"""Regularisation: the models, runs and measurements behind the
regularisation animations.

Four studies, each seeded and cached through :func:`voxanim.cache.cached_npz`:

* **Logistic regression, 3 against 8** (:func:`logistic_data`,
  :func:`paths`, :func:`plane`) -- each digit averaged over 7 x 7 blocks
  into 16 numbers (standardised), a linear classifier fitted by FISTA with
  an L1 or an L2 penalty at a ladder of strengths (warm-started, the
  intercept never penalised).  :func:`paths` is the 16-weight model's
  lasso and ridge paths; :func:`plane` is the same model on just two of the
  features (:data:`PAIR`), with the training loss over a grid of the two
  weights (the intercept fitted at every point) so the penalty's ball and
  the loss's contours can be drawn together.
* **An MLP's weights** (:func:`weight_runs`): the 784-64-10 ReLU MLP trained
  by SGD on all 60,000 digits with no penalty, L2, L1 (the proximal step)
  and both (elastic net), and L1 by its subgradient for contrast, with the
  first layer's weights recorded through training.
* **The race** (:func:`race_sweeps`, :func:`race_runs`): a 784-256-256-10
  MLP that memorises 1,000 digits (Adam), trained with no regulariser and
  with each of L2, L1 (its subgradient: Adam has no proximal step), dropout,
  input noise, max-norm and label smoothing at a ladder of strengths; each
  one's strength is the one with the best validation accuracy (2,000 other
  digits), and the test set (2,000 more) is only read for what the
  animation reports.
* **The autoencoder** (:func:`ae_runs`): the tied-weight autoencoder of
  :mod:`voxanim.regul_ae` trained plain and with each regulariser, each
  turned up until it costs the same clean reconstruction error
  (:data:`AE_BUDGET`), so what each buys can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import deepnet as DN
from . import regul_ae as AE
from .cache import cached_npz

# ------------------------------------------------------------------ 1. logistic regression

CLASSES = (3, 8)              # 0 is a 3, 1 is an 8
BLOCK = 7                     # 28 x 28 averaged over 7 x 7 blocks: 4 x 4 = 16 features
GRID_SIDE = 28 // BLOCK
PAIR = (6, 9)                 # the two features of the plane (block (1, 2) and block (2, 1))
L1_LAMS = np.geomspace(1e-3, 0.5, 46)
L2_LAMS = np.geomspace(1e-3, 50.0, 46)
FISTA_ITERS = 3000            # per strength, warm-started from the previous one
UNREG_ITERS = 20000
PLANE_GRID = 81               # the loss surface's samples per side
PLANE_RANGE = ((-2.6, 0.6), (-0.4, 3.6))      # (w6, w9) shown


@dataclass
class Logistic:
    """3s and 8s as 16 standardised block averages (``y`` 1 for an 8)."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    images: np.ndarray        # (n_test, 784) the test digits themselves


def pool(x):
    """``(N, 784)`` digits to ``(N, 16)``: the mean of each 7 x 7 block, row-major."""
    s = GRID_SIDE
    return x.reshape(-1, s, BLOCK, s, BLOCK).mean((2, 4)).reshape(-1, s * s)


def logistic_data(cache_dir=None):
    """The 3-against-8 problem (all of MNIST's 3s and 8s)."""
    from .mnist import load
    xtr, ytr, xte, yte = load(cache_dir)
    parts = []
    for x, y in ((xtr, ytr), (xte, yte)):
        keep = np.isin(y, CLASSES)
        parts += [x[keep], (y[keep] == CLASSES[1]).astype(np.float64)]
    ftr, fte = pool(parts[0]), pool(parts[2])
    mu, sd = ftr.mean(0), ftr.std(0)
    return Logistic((ftr - mu) / sd, parts[1], (fte - mu) / sd, parts[3], mu, sd, parts[2])


def logistic_loss(w, b, x, y):
    """Mean cross-entropy of ``sigmoid(x w + b)`` against ``y``."""
    z = x @ w + b
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def accuracy(w, b, x, y):
    """Fraction of ``x`` classified right (``x w + b > 0`` means an 8)."""
    return float(np.mean(((x @ w + b) > 0) == (y > 0.5)))


def penalty_of(kind, lam, w):
    """``lam * |w|_1`` or ``lam / 2 * |w|^2``."""
    if kind == "l1":
        return float(lam * np.abs(w).sum())
    return float(0.5 * lam * (w * w).sum())


def fista(x, y, lam, kind, w, b, iters=FISTA_ITERS):
    """Minimise the logistic loss plus the ``kind`` (``"l1"``/``"l2"``)
    penalty of strength ``lam`` from ``(w, b)``: accelerated proximal
    gradient (FISTA) with the fixed step ``1 / Lipschitz``; the L1 penalty's
    proximal step is the soft threshold, which is what puts weights at
    exactly 0.  Returns ``(w, b)``."""
    lip = np.linalg.eigvalsh(x.T @ x / len(y)).max() / 4.0 + 0.25 + (lam if kind == "l2" else 0.0)
    w, b = np.array(w, np.float64), float(b)
    v, vb, t = w.copy(), b, 1.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(x @ v + vb)))
        g, gb = x.T @ (p - y) / len(y), float(np.mean(p - y))
        if kind == "l2":
            g = g + lam * v
        wn, bn = v - g / lip, vb - gb / lip
        if kind == "l1":
            wn = DN.soft_threshold(wn, lam / lip)
        tn = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        v = wn + (t - 1.0) / tn * (wn - w)
        vb = bn + (t - 1.0) / tn * (bn - b)
        w, b, t = wn, bn, tn
    return w, b


@dataclass
class Path:
    """A regularisation path: the fitted model at each strength ``lams[i]``."""

    kind: str
    lams: np.ndarray          # (n,)
    w: np.ndarray             # (n, k)
    b: np.ndarray             # (n,)
    train_loss: np.ndarray    # (n,) the data term alone
    test_acc: np.ndarray      # (n,)

    def nonzero(self):
        """``(n,)`` how many weights are not exactly 0."""
        return (self.w != 0).sum(1)


def _path(dat, kind, lams, cols, w0, b0):
    x, xt = dat.x_train[:, cols], dat.x_test[:, cols]
    w, b = w0.copy(), b0
    W, B, L, A = [], [], [], []
    for lam in lams:
        w, b = fista(x, dat.y_train, lam, kind, w, b)
        W.append(w.copy())
        B.append(b)
        L.append(logistic_loss(w, b, x, dat.y_train))
        A.append(accuracy(w, b, xt, dat.y_test))
    return dict(w=np.array(W), b=np.array(B), loss=np.array(L), acc=np.array(A))


def _fit_paths(dat, cols, l1_lams, l2_lams):
    x, xt = dat.x_train[:, cols], dat.x_test[:, cols]
    w0, b0 = fista(x, dat.y_train, 0.0, "l2", np.zeros(len(cols)), 0.0, UNREG_ITERS)
    out = dict(w0=w0, b0=np.array(b0), loss0=np.array(logistic_loss(w0, b0, x, dat.y_train)),
               acc0=np.array(accuracy(w0, b0, xt, dat.y_test)))
    for kind, lams in (("l1", l1_lams), ("l2", l2_lams)):
        for k, v in _path(dat, kind, lams, cols, w0, b0).items():
            out["%s_%s" % (kind, k)] = v
    return out


def _paths_from(d, l1_lams, l2_lams):
    return {kind: Path(kind, lams, d[kind + "_w"], d[kind + "_b"], d[kind + "_loss"],
                       d[kind + "_acc"]) for kind, lams in (("l1", l1_lams), ("l2", l2_lams))}


@dataclass
class Paths:
    """The 16-weight model: unregularised, and its lasso and ridge paths."""

    w0: np.ndarray
    b0: float
    acc0: float
    l1: Path
    l2: Path


def paths(cache_dir=None) -> Paths:
    """Lasso and ridge paths of the 16-feature 3-against-8 classifier (cached)."""
    settings = dict(task="logistic_paths", classes=list(CLASSES), block=BLOCK,
                    l1=[float(v) for v in L1_LAMS], l2=[float(v) for v in L2_LAMS],
                    iters=FISTA_ITERS, unreg=UNREG_ITERS)
    d = cached_npz("regul_paths", settings,
                   lambda: _fit_paths(logistic_data(cache_dir), list(range(GRID_SIDE ** 2)),
                                      L1_LAMS, L2_LAMS), cache_dir)
    p = _paths_from(d, L1_LAMS, L2_LAMS)
    return Paths(d["w0"], float(d["b0"]), float(d["acc0"]), p["l1"], p["l2"])


PLANE_L1 = np.geomspace(1e-3, 0.6, 70)
PLANE_L2 = np.geomspace(1e-3, 60.0, 70)


@dataclass
class Plane:
    """The two-weight model of :data:`PAIR`: its loss surface and its paths."""

    w0: np.ndarray            # (2,) the unregularised optimum
    loss0: float
    acc0: float
    axes: tuple               # (w6 samples, w9 samples), PLANE_GRID each
    loss: np.ndarray          # (G, G) training loss, rows over w9, columns over w6,
    #                           the intercept fitted at every point
    l1: Path
    l2: Path


def profile_loss(x, y, w1s, w2s, iters=8):
    """``(len(w2s), len(w1s))`` the logistic loss at each weight pair with the
    intercept chosen best (Newton's method on ``b``, per point)."""
    out = np.empty((len(w2s), len(w1s)))
    for r, w2 in enumerate(w2s):
        z = x[:, :1] * w1s[None, :] + x[:, 1:2] * w2     # (N, G)
        b = np.zeros(len(w1s))
        for _ in range(iters):
            p = 1.0 / (1.0 + np.exp(-(z + b)))
            b -= (p - y[:, None]).mean(0) / np.maximum((p * (1 - p)).mean(0), 1e-12)
        zz = z + b
        out[r] = np.mean(np.logaddexp(0.0, zz) - y[:, None] * zz, 0)
    return out


def plane(cache_dir=None) -> Plane:
    """The two-weight model, its loss surface and its two paths (cached)."""
    settings = dict(task="logistic_plane", classes=list(CLASSES), block=BLOCK, pair=list(PAIR),
                    l1=[float(v) for v in PLANE_L1], l2=[float(v) for v in PLANE_L2],
                    grid=PLANE_GRID, range=[list(r) for r in PLANE_RANGE], iters=FISTA_ITERS,
                    unreg=UNREG_ITERS)
    w1s = np.linspace(*PLANE_RANGE[0], PLANE_GRID)
    w2s = np.linspace(*PLANE_RANGE[1], PLANE_GRID)

    def build():
        dat = logistic_data(cache_dir)
        out = _fit_paths(dat, list(PAIR), PLANE_L1, PLANE_L2)
        out["surface"] = profile_loss(dat.x_train[:, list(PAIR)], dat.y_train, w1s, w2s)
        return out

    d = cached_npz("regul_plane", settings, build, cache_dir)
    p = _paths_from(d, PLANE_L1, PLANE_L2)
    return Plane(d["w0"], float(d["loss0"]), float(d["acc0"]), (w1s, w2s), d["surface"],
                 p["l1"], p["l2"])


def first_zero(path, i):
    """The smallest strength at which weight ``i`` of ``path`` is exactly 0 and
    stays 0 at every larger strength (``inf`` if it never does)."""
    z = path.w[:, i] == 0
    if not z[-1]:
        return float("inf")
    k = len(z) - int(np.argmin(z[::-1])) if not z.all() else 0
    return float(path.lams[k])


# ------------------------------------------------------------------ 2. an MLP's weights

W_SIZES = (784, 64, 10)
W_TRAIN = dict(steps=9400, batch=64, lr=0.1, opt="sgd", every=235)     # 10 epochs, 40 records
W_RUNS = (("none", {}),
          ("L2", dict(weight_decay=2e-3)),
          ("L1", dict(l1=2e-4, l1_prox=True)),
          ("elastic net", dict(l1=1e-4, l1_prox=True, weight_decay=1e-3)),
          ("L1 by its slope", dict(l1=2e-4)))
W_LABELS = tuple(name for name, _ in W_RUNS)
W_SEED = 0


def weight_data(cache_dir=None):
    """All 60,000 training digits and 2,000 test digits (a seeded shuffle)."""
    return DN.subset(60000, 2000, seed=W_SEED, cache_dir=cache_dir)


def weight_runs(cache_dir=None, dat=None):
    """One run per entry of :data:`W_RUNS`, the first layer's weights recorded."""
    dat = weight_data(cache_dir) if dat is None else dat
    kw = dict(W_TRAIN, weights=("W0",))
    if cache_dir:
        kw["cache_dir"] = cache_dir
    return [DN.train(DN.Spec(W_SIZES, "relu", "he", **opts), dat, **kw) for _n, opts in W_RUNS]


def never_lit(dat):
    """``(784,)`` the pixels that are 0 in every training digit: no data
    gradient ever reaches their weights."""
    return (dat.x_train > 0).sum(0) == 0


def cut_pixels(W):
    """``(784,)`` the pixels whose every outgoing weight is exactly 0."""
    return (W != 0).sum(1) == 0


# ------------------------------------------------------------------ 3. the race

RACE_SIZES = (784, 256, 256, 10)
RACE_N, RACE_VAL, RACE_TEST = 1000, 2000, 2000
RACE_TRAIN = dict(steps=3000, batch=64, lr=1e-3, opt="adam", every=50)
RACE_SEED = 0


class GaussianNoise:
    """Input noise: every minibatch's pixels plus Gaussian noise of std ``sd``
    (an ``augment`` for :func:`voxanim.deepnet.train`)."""

    def __init__(self, sd):
        self.sd = float(sd)

    def settings(self):
        """Its cache key."""
        return dict(kind="gaussian", sd=self.sd)

    def __call__(self, rng, x):
        return x + rng.normal(0.0, self.sd, x.shape)


@dataclass(frozen=True)
class Reg:
    """One regulariser of the race and the strengths it is tried at."""

    key: str
    label: str
    symbol: str               # how its strength is written: "λ", "p", "σ", "c", "ε"
    grid: tuple
    ink: tuple

    def spec_and_augment(self, s):
        """``(Spec, augment)`` for strength ``s``."""
        opts, aug = {}, None
        if self.key == "l2":
            opts["weight_decay"] = s
        elif self.key == "l1":
            opts["l1"] = s
        elif self.key == "dropout":
            opts["dropout"] = s
        elif self.key == "noise":
            aug = GaussianNoise(s)
        elif self.key == "maxnorm":
            opts["max_norm"] = s
        elif self.key == "smooth":
            opts["label_smoothing"] = s
        return DN.Spec(RACE_SIZES, "relu", "he", **opts), aug


def _ink(h):
    from . import color as C
    return tuple(float(v) for v in C.rgb(h))


PLAIN_INK = (0.40, 0.42, 0.46)
REGS = (Reg("l2", "L2", "λ", (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1), _ink("#2f8a86")),
        Reg("l1", "L1", "λ", (1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2), _ink("#3f55c8")),
        Reg("dropout", "dropout", "p", (0.2, 0.35, 0.5, 0.65, 0.8), _ink("#b0318c")),
        Reg("noise", "input noise", "σ", (0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.3), _ink("#8a6a2e")),
        Reg("maxnorm", "max-norm", "c", (0.35, 0.5, 0.75, 1.0, 1.5, 2.0), _ink("#35803d")),
        Reg("smooth", "label smoothing", "ε", (0.02, 0.05, 0.1, 0.2, 0.3), _ink("#7a46d6")))
BY_KEY = {r.key: r for r in REGS}


def race_data(cache_dir=None):
    """1,000 training digits, and 2,000 validation and 2,000 test digits drawn
    from MNIST's test set (a seeded shuffle)."""
    d = DN.subset(RACE_N, RACE_VAL + RACE_TEST, seed=RACE_SEED, cache_dir=cache_dir)
    return DN.Data(d.x_train, d.y_train, d.x_test[RACE_VAL:], d.y_test[RACE_VAL:], 10,
                   d.x_test[:RACE_VAL], d.y_test[:RACE_VAL])


def _train(spec, dat, aug, cache_dir, **extra):
    kw = dict(RACE_TRAIN, augment=aug, seed=RACE_SEED, **extra)
    if cache_dir:
        kw["cache_dir"] = cache_dir
    return DN.train(spec, dat, **kw)


def plain_run(cache_dir=None, dat=None):
    """The race network with no regulariser."""
    dat = race_data(cache_dir) if dat is None else dat
    return _train(DN.Spec(RACE_SIZES, "relu", "he"), dat, None, cache_dir)


def val_acc(dat, run):
    """Validation accuracy of a run's final weights."""
    f = DN.forward(run.spec, run.P, dat.x_val, stats=run.stats)
    return float((f["p"].argmax(1) == dat.y_val).mean())


@dataclass
class Sweep:
    """One regulariser at every strength of its grid."""

    reg: Reg
    runs: list
    val_acc: np.ndarray       # (n,) final validation accuracy
    train_acc: np.ndarray     # (n,) final accuracy on the training digits

    @property
    def best(self):
        """Index of the chosen strength: the best validation accuracy (the
        weaker strength on a tie)."""
        return int(np.argmax(self.val_acc))


def race_sweeps(cache_dir=None, dat=None):
    """``{key: Sweep}`` for every regulariser of :data:`REGS`."""
    dat = race_data(cache_dir) if dat is None else dat
    out = {}
    for reg in REGS:
        runs, va, ta = [], [], []
        for s in reg.grid:
            spec, aug = reg.spec_and_augment(s)
            r = _train(spec, dat, aug, cache_dir)
            runs.append(r)
            va.append(val_acc(dat, r))
            f = DN.forward(r.spec, r.P, dat.x_train)
            ta.append(float((f["p"].argmax(1) == dat.y_train).mean()))
        out[reg.key] = Sweep(reg, runs, np.array(va), np.array(ta))
    return out


def race_runs(cache_dir=None, dat=None):
    """``(plain, [(Reg, strength, run), ...])``: the race's lineup, each
    regulariser at its chosen strength."""
    dat = race_data(cache_dir) if dat is None else dat
    sweeps = race_sweeps(cache_dir, dat)
    chosen = [(sw.reg, sw.reg.grid[sw.best], sw.runs[sw.best]) for sw in sweeps.values()]
    return plain_run(cache_dir, dat), chosen


def confidence(run, x):
    """``(N,)`` the network's top probability on each of ``x``."""
    return DN.forward(run.spec, run.P, x, stats=run.stats)["p"].max(1)


def logit_gap(run, x):
    """``(N,)`` the top logit minus the mean of the others: what label smoothing caps."""
    lg = DN.forward(run.spec, run.P, x, stats=run.stats)["logits"]
    top = lg.max(1)
    return top - (lg.sum(1) - top) / (lg.shape[1] - 1)


def smoothing_gap(eps, classes=10):
    """The logit gap at which a softmax hits the smoothed target exactly:
    ``log((1 - eps + eps / K) / (eps / K))`` (equal other logits)."""
    return float(np.log((1.0 - eps + eps / classes) / (eps / classes)))


# ------------------------------------------------------------------ 4. the autoencoder

AE_BUDGET = 0.0075            # the clean reconstruction error every regulariser is allowed
AE_RUNS = (("plain", AE.AESpec("plain")),
           ("L1 on the code", AE.AESpec("sparse", 1.1e-4)),
           ("weight decay", AE.AESpec("decay", 5e-5)),
           ("denoising", AE.AESpec("denoise", 0.55)),
           ("contractive", AE.AESpec("contract", 2.5e-5)))
AE_LABELS = tuple(name for name, _ in AE_RUNS)
AE_MEASURE = (2000, 4000)     # the test digits measured (apart from the EVAL_N training curve)


def ae_runs(cache_dir=None):
    """One :class:`~voxanim.regul_ae.AERun` per entry of :data:`AE_RUNS`."""
    kw = dict(cache_dir=cache_dir) if cache_dir else {}
    return [AE.train(spec, **kw) for _n, spec in AE_RUNS]


def ae_measures(runs, cache_dir=None):
    """:func:`voxanim.regul_ae.measure` of every run on the same test digits."""
    from .mnist import load
    xte = load(cache_dir)[2]
    x = xte[AE_MEASURE[0]:AE_MEASURE[1]]
    return [AE.measure(r.P, x) for r in runs]

"""The experiments behind animations 157-158: an anomaly search with two scores (see
:func:`voxanim.next_catalog.redesigned`).

Both build on ``autoencoder-anomaly-detection`` (:mod:`voxanim.study_phys_adv`): the same
MNIST digits, the same autoencoder trained with one digit class removed, the same seeds -
the networks are retrained here from the same random streams and come out identical, so
their reconstruction errors equal that study's to the last digit. That study scored a
picture by its reconstruction error alone and found the unseen 1s scoring *below* the seen
digits. Here every picture gets a second score, read in the code: how unusual its code is
among the codes of the training digits.

* ``anomaly-loss-and-latent``: the two scores side by side; which anomalies each one
  catches (both, the loss only, the code only, neither) at a fixed false-alarm rate, for
  every removed class, eight ways of spoiling a real digit and three seeds.
* ``anomaly-two-score-flow``: the whole procedure. Three densities fitted to the training
  codes (one Gaussian, a mixture of ten, a normalizing flow), thresholds set on held-out
  normal digits, three ways of combining the two scores, the false alarms that two cuts
  cost, and the same at four code sizes.

Thresholds, density fits and the choice of density use digits of the kept classes only.
Everything the drawings need is recorded here: :mod:`voxanim.mech_anomaly` computes nothing
but layout.
"""

from __future__ import annotations

import functools
import hashlib
import inspect

import numpy as np

from . import study_phys_adv as P
from .cache import cached_npz
from .optim import Adam
from .study import Panel, Study

VERSION = 1
SEEDS = P.AE_SEEDS
CODES = (2, 8, 32, 128)           # numbers in the code; a subset of the earlier study's seven
DRAWN, MAIN = 2, 8                # the code that can be drawn; the earlier study's best size
FEATURED = 1                      # the removed class the pictures follow: the one the loss misses
KNN = 5                           # the latent score "knn": distance to the 5th nearest code
KNN_REF = 6000                    # ... among this many training codes (a seeded draw)
MIX, MIX_STEPS = 10, 30           # components of the mixture; rounds of EM
MIX_FIT, MIX_RIDGE = 6000, 0.01   # training codes the mixture is fitted to; added variance
FLOW_LAYERS, FLOW_HIDDEN, FLOW_STEPS, FLOW_BATCH, FLOW_LR = 6, 48, 1200, 128, 2e-3
FLOW_SCALE = 1.5                  # a coupling layer scales by at most exp(+-1.5)
FLOW_EVERY = 50                   # the flow's training curve: the mean of every 50 updates
N_COR = 1000                      # held-out digits of kept classes that are spoiled
NOISE, DIMMED, SHIFT = 0.3, 0.4, 6
KINDS = ("rot90", "inverted", "noise", "shift", "flip", "half", "blend", "dimmed")
KIND_NAMES = {"rot90": "turned 90°", "inverted": "inverted", "noise": "pixel noise",
              "shift": "shifted 6 px", "flip": "mirrored", "half": "lower half erased",
              "blend": "two digits averaged", "dimmed": "dimmed to 40%"}
SCORES = ("loss", "gauss", "knn", "mix", "flow")
DENSITIES = ("gauss", "mix", "flow")
ALPHAS = (0.01, 0.05)             # false-alarm rates at which thresholds are set
ALPHA = 0.05                      # the one the pictures use
GRID = 44                         # cells a side of the drawn densities (2-D code)
N_CLOUD = 2000                    # training codes kept for the drawing of the 2-D code
BINS = 32


def _cache(name, build, sources, **settings):
    h = hashlib.sha256()
    for fn in sources:
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-anomaly-" + name, settings,
                                                    build).items()}


def _stat(a):
    a = np.asarray(a, float)
    return dict(mean=float(a.mean()), min=float(a.min()), max=float(a.max()))


# ====================================================================== the inputs
def groups(yt, left):
    """``(calibration, test, anomalies)``: held-out digits of the kept classes at even
    and at odd positions, and every held-out digit of the removed class. Thresholds are
    set on the first group and false alarms counted on the second."""
    at = np.arange(len(yt))
    kept = yt != left
    return (np.flatnonzero(kept & (at % 2 == 0)), np.flatnonzero(kept & (at % 2 == 1)),
            np.flatnonzero(~kept))


def spoil(v, rng):
    """Real digits spoiled in the eight ways of :data:`KINDS`: ``[kind, digit, 784]``."""
    v = np.asarray(v, np.float32)
    im = v.reshape(-1, 28, 28)
    flat = lambda a: np.ascontiguousarray(a).reshape(len(v), 784)  # noqa: E731
    half = im.copy()
    half[:, 14:] = 0.0
    out = dict(
        rot90=flat(np.rot90(im, 1, (1, 2))), inverted=1.0 - v,
        noise=np.clip(v + rng.normal(0.0, NOISE, v.shape), 0.0, 1.0),
        shift=flat(np.roll(im, SHIFT, 2)), flip=flat(im[:, :, ::-1]), half=flat(half),
        blend=(v + np.roll(v, 1, 0)) / 2, dimmed=DIMMED * v)
    return np.array([out[k] for k in KINDS], np.float32)


# ====================================================================== latent scores
def whitener(z):
    """``(mean, L)`` of codes ``z``: ``L`` is the Cholesky factor of their covariance (with
    a ridge of a millionth of the mean variance, for codes with unused directions)."""
    z = np.asarray(z, float)
    cov = np.atleast_2d(np.cov(z.T))
    cov = cov + 1e-6 * np.trace(cov) / len(cov) * np.eye(len(cov))
    return z.mean(0), np.linalg.cholesky(cov)


def whiten(z, mean, L):
    """Codes in the units of the training cloud: mean 0, covariance 1 on training codes.
    The length of a whitened code is its Mahalanobis distance from the cloud's centre."""
    return np.linalg.solve(L, (np.asarray(z, float) - mean).T).T


def gauss_nll(u):
    """Minus the log density of whitened codes under a standard normal."""
    return 0.5 * (u ** 2).sum(1) + 0.5 * u.shape[1] * np.log(2 * np.pi)


def knn_distance(z, ref, k=KNN):
    """Distance from each row of ``z`` to its ``k``-th nearest row of ``ref``."""
    z, ref = np.asarray(z, np.float32), np.asarray(ref, np.float32)
    r2, out = (ref ** 2).sum(1), []
    for i in range(0, len(z), 1000):
        q = z[i:i + 1000]
        d2 = (q ** 2).sum(1)[:, None] - 2.0 * (q @ ref.T) + r2[None]
        out.append(np.sqrt(np.maximum(np.partition(d2, k - 1, axis=1)[:, k - 1], 0.0)))
    return np.concatenate(out).astype(float)


def mix_parts(u, g):
    """``log(weight x density)`` of every row of ``u`` under every component of ``g``."""
    lp = []
    for mu, cov, pi in zip(g["mu"], g["cov"], g["pi"]):
        d = u - mu
        lp.append(np.log(pi) - 0.5 * (((d @ np.linalg.inv(cov)) * d).sum(1)
                                      + np.linalg.slogdet(cov)[1]
                                      + u.shape[1] * np.log(2 * np.pi)))
    return np.array(lp).T


def mix_nll(u, g):
    """Minus the log density of ``u`` under the Gaussian mixture ``g``."""
    lp = mix_parts(u, g)
    top = lp.max(1)
    return -(top + np.log(np.exp(lp - top[:, None]).sum(1)))


def mix_fit(u, rng, k=MIX, steps=MIX_STEPS, ridge=MIX_RIDGE):
    """A mixture of ``k`` full-covariance Gaussians fitted to whitened codes ``u`` by EM,
    from ``k`` of the codes as first centres. ``ridge`` is added to every variance: with
    128 numbers a component has more covariances than digits to estimate them from.

    :func:`voxanim.toy2d.gmm_em` does the same sums one pair at a time and has no ridge:
    a run with 128 numbers took 79 s with it, most of it there."""
    eye = np.eye(u.shape[1])
    g = dict(mu=u[rng.choice(len(u), k, replace=False)].copy(), cov=np.repeat(eye[None], k, 0),
             pi=np.full(k, 1.0 / k))
    for _ in range(steps):
        lp = mix_parts(u, g)
        r = np.exp(lp - lp.max(1, keepdims=True))
        r /= r.sum(1, keepdims=True)
        nk = r.sum(0) + 1e-9
        g["pi"], g["mu"] = nk / nk.sum(), r.T @ u / nk[:, None]
        for j in range(k):
            d = u - g["mu"][j]
            g["cov"][j] = (d.T * r[:, j]) @ d / nk[j] + ridge * eye
    return g


# ---------------------------------------------------------------------- the flow
# :mod:`voxanim.toy2d` has a flow for two numbers only; this is the same affine coupling
# layer for a code of any size. A layer leaves half of the numbers alone and uses them to
# scale and shift the other half; which half changes from layer to layer.
def flow_init(d, rng, layers=FLOW_LAYERS, hidden=FLOW_HIDDEN):
    p = {}
    for n in range(layers):
        order = rng.permutation(d)
        if d == 2:
            order = np.array([n % 2, 1 - n % 2])
        a, b = np.sort(order[:d // 2]), np.sort(order[d // 2:])
        p["ia%d" % n], p["ib%d" % n] = a, b
        p["W%d" % n] = rng.normal(0.0, 1.0 / np.sqrt(len(a)), (len(a), hidden))
        p["b%d" % n] = np.zeros(hidden)
        p["V%d" % n] = rng.normal(0.0, 0.01, (hidden, 2 * len(b)))
        p["c%d" % n] = np.zeros(2 * len(b))
    return p


def flow_layers(p):
    return sum(k.startswith("W") for k in p)


def flow_forward(p, x):
    """``(z, log det, tape)``: ``x`` carried through the layers to where a standard
    normal should describe it, and the log of how much each point's volume grew."""
    z, logdet, tape = np.array(x, float), np.zeros(len(x)), []
    for n in range(flow_layers(p)):
        ia, ib = p["ia%d" % n], p["ib%d" % n]
        a, b = z[:, ia], z[:, ib]
        h = np.tanh(a @ p["W%d" % n] + p["b%d" % n])
        st = h @ p["V%d" % n] + p["c%d" % n]
        th = np.tanh(st[:, :len(ib)])
        s = FLOW_SCALE * th
        out = z.copy()
        out[:, ib] = b * np.exp(s) + st[:, len(ib):]
        tape.append((a, b, h, th, s))
        z = out
        logdet += s.sum(1)
    return z, logdet, tape


def flow_nll(p, x):
    """Minus the log density of each row of ``x`` under the flow."""
    z, logdet, _ = flow_forward(p, x)
    return gauss_nll(z) - logdet


def flow_loss_grads(p, x):
    """The mean of :func:`flow_nll` over ``x`` and its gradient for every weight."""
    z, logdet, tape = flow_forward(p, x)
    n = len(x)
    loss = float(np.mean(gauss_nll(z) - logdet))
    dz, g = z / n, {}
    for k in reversed(range(len(tape))):
        ia, ib = p["ia%d" % k], p["ib%d" % k]
        a, b, h, th, s = tape[k]
        db = dz[:, ib]
        ds = db * b * np.exp(s) - 1.0 / n
        dst = np.concatenate([ds * FLOW_SCALE * (1 - th ** 2), db], 1)
        g["V%d" % k], g["c%d" % k] = h.T @ dst, dst.sum(0)
        dh = (dst @ p["V%d" % k].T) * (1 - h ** 2)
        g["W%d" % k], g["b%d" % k] = a.T @ dh, dh.sum(0)
        d = dz.copy()
        d[:, ib] = db * np.exp(s)
        d[:, ia] += dh @ p["W%d" % k].T
        dz = d
    return loss, g


def flow_fit(u, rng):
    """A flow fitted to whitened codes ``u`` by maximum likelihood (it starts as the
    Gaussian: the last weights of every layer start near zero). Returns the flow and its
    training curve."""
    p = flow_init(u.shape[1], rng)
    w = {k: v for k, v in p.items() if not k.startswith("i")}
    opt, curve = Adam(w, FLOW_LR), []
    for step in range(FLOW_STEPS):
        value, g = flow_loss_grads(p, u[rng.integers(0, len(u), FLOW_BATCH)])
        opt.step(w, g, lr=FLOW_LR * (0.3 if step >= 0.7 * FLOW_STEPS else 1.0))
        p.update(w)
        curve.append(value)
    return p, np.array(curve).reshape(-1, FLOW_EVERY).mean(1)


# ====================================================================== the runs
def train_autoencoder(left, code, seed):
    """The autoencoder of :func:`voxanim.study_phys_adv.ae_job` for the same arguments
    (same initial weights, same batches, so the same network), and its training digits."""
    x, y = P.ae_data()[:2]
    rng = np.random.default_rng([seed, left, code, 154])
    xs = x[y != left]
    net = P.ae_init(code, rng)
    opt = Adam(net, P.AE_LR)
    for _ in range(P.AE_STEPS):
        opt.step(net, P.ae_grads(net, xs[rng.integers(0, len(xs), P.AE_BATCH)]))
    return net, xs


def job(left, code, seed):
    """One autoencoder trained without class ``left``; the three densities fitted to its
    training codes; and the five scores of every held-out digit and of every spoiled one.
    Densities are in nats per code, in the code's own units."""
    xt, yt = P.ae_data()[2:4]
    net, xs = train_autoencoder(left, code, seed)
    rng = np.random.default_rng([seed, left, code, 157])
    _cal, test, anom = groups(yt, left)
    spoiled = spoil(xt[test[:N_COR]], rng)
    ztr = P.ae_forward(net, xs)[1].astype(float)
    mean, L = whitener(ztr)
    utr = whiten(ztr, mean, L)
    volume = float(np.log(np.diag(L)).sum())          # log det of the whitening
    mix = mix_fit(utr[rng.choice(len(utr), MIX_FIT, replace=False)], rng)
    ref = ztr[rng.choice(len(ztr), KNN_REF, replace=False)]
    flow, curve = flow_fit(utr, rng)

    def latent(z):
        u = whiten(z, mean, L)
        return [gauss_nll(u) + volume, knn_distance(z, ref), mix_nll(u, mix) + volume,
                flow_nll(flow, u) + volume]

    def score(v):
        z, out = P.ae_forward(net, v)[1::2]
        return np.array([P.rebuild_error(out, v)] + latent(z.astype(float))), z, out

    s, z, out = score(xt)
    parts = [score(v) for v in spoiled]
    scores = np.concatenate([s] + [q[0] for q in parts], 1)
    typical = [g[np.argsort(s[0][g], kind="stable")[len(g) // 2]] for g in (test, anom)]
    cor_typical = [int(np.argsort(q[0][0], kind="stable")[N_COR // 2]) for q in parts]
    pics = np.concatenate([out[typical]] + [q[2][[k]] for q, k in zip(parts, cor_typical)])
    res = dict(scores=scores.astype(np.float32), typical=np.array(typical),
               cor_typical=np.array(cor_typical), pics=pics.astype(np.float32),
               curve=curve, volume=np.array(volume), mean=mean, L=L,
               train_nll=np.array([np.mean(v) for v in latent(ztr[:4000])])[[0, 2, 3]])
    if code == DRAWN:
        lo, hi = np.quantile(ztr, 0.002, 0), np.quantile(ztr, 0.998, 0)
        lo, hi = lo - 0.12 * (hi - lo), hi + 0.12 * (hi - lo)
        gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], GRID), np.linspace(lo[1], hi[1], GRID))
        cells = np.column_stack([gx.ravel(), gy.ravel()])
        dens = np.array(latent(cells))[[0, 2, 3]].reshape(3, GRID, GRID)
        keep = np.random.default_rng([seed, left, 2]).choice(len(ztr), N_CLOUD, replace=False)
        res.update(z=z.astype(np.float32),
                   zc=np.array([q[1] for q in parts], np.float32),
                   cloud=ztr[keep].astype(np.float32), cloud_ids=keep,
                   centres=mean + mix["mu"] @ L.T, dens=dens.astype(np.float32),
                   box=np.array([lo[0], hi[0], lo[1], hi[1]]))
    return res


def plan(code):
    return [(left, code, seed) for left in range(10) for seed in SEEDS]


def _build(code):
    out = P._parallel(job, plan(code))
    shape = (10, len(SEEDS))
    return {k: np.array([o[k] for o in out]).reshape(shape + np.shape(out[0][k]))
            for k in out[0]}


@functools.cache
def arrays(code):
    """Every run at one code size: ``scores[left, seed, score, digit]`` (the 5,000 held-out
    digits, then 1,000 spoiled ones for each of :data:`KINDS`), the rebuilt pictures of
    the typical digits, and for the 2-D code the codes themselves."""
    return _cache("code%d" % code, functools.partial(_build, code),
                  (groups, spoil, whitener, whiten, gauss_nll, knn_distance, mix_parts, mix_nll,
                   mix_fit,
                   flow_init, flow_forward, flow_loss_grads, flow_fit, train_autoencoder,
                   job, plan, _build, P.ae_init, P.ae_grads, P.ae_forward),
                  code=code, seeds=SEEDS, train=P.AE_TRAIN, test=P.AE_TEST,
                  split_seed=P.AE_SPLIT_SEED, hidden=P.AE_HIDDEN, steps=P.AE_STEPS,
                  batch=P.AE_BATCH, lr=P.AE_LR, knn=(KNN, KNN_REF),
                  mix=(MIX, MIX_STEPS, MIX_FIT, MIX_RIDGE),
                  flow=(FLOW_LAYERS, FLOW_HIDDEN, FLOW_STEPS, FLOW_BATCH, FLOW_LR, FLOW_SCALE),
                  spoiled=(N_COR, NOISE, DIMMED, SHIFT, KINDS), grid=GRID, cloud=N_CLOUD)


# ====================================================================== reading the runs
HELD = P.AE_TEST
METHODS = ("loss", "latent", "either", "sum", "plane")
LOSS, GAUSS, KNN_, MIXT, FLOW = range(5)


def kind_slice(k):
    """Where the spoiled digits of kind ``k`` sit along a run's digit axis."""
    return slice(HELD + k * N_COR, HELD + (k + 1) * N_COR)


def cut(values, alpha):
    """The threshold that lets ``1 - alpha`` of ``values`` (normal digits) pass."""
    return float(np.quantile(np.asarray(values, float), 1.0 - alpha))


def quadrants(a, b):
    """Shares of ``(both, first only, second only, neither)`` from two flags per digit."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    return np.array([(a & b).mean(), (a & ~b).mean(), (~a & b).mean(), (~a & ~b).mean()])


def ranks(normal, v):
    """The share of ``normal`` values below each of ``v``: 0.95 means the digit scores
    above 95% of normal digits."""
    return np.searchsorted(np.sort(np.asarray(normal, float)), v, "left") / len(normal)


def combine(loss, latent, cal):
    """Both scores in the units of normal digits, and three ways of making one of them.

    ``z1`` is ln(loss) and ``z2`` the latent score, each minus its mean over the
    calibration digits ``cal`` and divided by their standard deviation. ``either``: the
    larger of the two ranks among calibration digits (a cut on it is the OR of two cuts
    at one shared rate); ``sum``: ``z1 + z2``; ``plane``: the squared Mahalanobis distance
    of ``(z1, z2)`` from the calibration cloud (a cut on it is an ellipse, and it flags
    a digit unusual in either direction). Returns the five scores and that cloud's
    covariance. Nothing here sees an anomaly."""
    a = np.log(np.asarray(loss, float))
    b = np.asarray(latent, float)
    z1 = (a - a[cal].mean()) / a[cal].std()
    z2 = (b - b[cal].mean()) / b[cal].std()
    pts = np.column_stack([z1, z2])
    cov = np.cov(pts[cal].T)
    plane = np.einsum("ni,ij,nj->n", pts, np.linalg.inv(cov), pts)
    return dict(loss=z1, latent=z2, either=np.maximum(ranks(z1[cal], z1), ranks(z2[cal], z2)),
                sum=z1 + z2, plane=plane), cov


@functools.cache
def summary(code):
    """What every run at one code size measured.

    ``auc[left, seed, score]`` and ``kind_auc[kind, left, seed, score]``: areas under the
    ROC curve of each of :data:`SCORES` alone, the unseen class (or the spoiled digits)
    against every held-out digit of the kept classes. ``normal_nll[left, seed, density]``:
    the mean of minus the log density over calibration digits - the number the density
    is chosen by. ``quad`` / ``kind_quad``: shares in the four quadrants at
    :data:`ALPHA`, the latent score being the 5th-neighbour distance. With the chosen
    density as the latent score: ``m_auc[left, seed, method]``, ``caught`` and ``alarms``
    ``[alpha, left, seed, method]`` (thresholds from calibration digits, false alarms
    counted on the other half), the same for two uncorrected cuts (``or_raw``), and with
    the neighbour distance in place of the density (``k_auc``, ``k_caught``)."""
    d = arrays(code)
    yt = P.ae_data()[3]
    n_seed, n_kind = len(SEEDS), len(KINDS)
    auc = np.zeros((10, n_seed, len(SCORES)))
    kind_auc = np.zeros((n_kind, 10, n_seed, len(SCORES)))
    normal_nll = np.zeros((10, n_seed, len(DENSITIES)))
    quad = np.zeros((10, n_seed, 4))
    kind_quad = np.zeros((n_kind, 10, n_seed, 4))
    alarms2 = np.zeros((10, n_seed, 4))
    for left, s in np.ndindex(10, n_seed):
        sc = d["scores"][left, s].astype(float)
        cal, test, anom = groups(yt, left)
        kept = np.concatenate([cal, test])
        for k in range(len(SCORES)):
            auc[left, s, k] = P.auc(sc[k, kept], sc[k, anom])
            for j in range(n_kind):
                kind_auc[j, left, s, k] = P.auc(sc[k, kept], sc[k, kind_slice(j)])
        normal_nll[left, s] = sc[[GAUSS, MIXT, FLOW]][:, cal].mean(1)
        t_loss, t_lat = cut(sc[LOSS, cal], ALPHA), cut(sc[KNN_, cal], ALPHA)
        quad[left, s] = quadrants(sc[LOSS, anom] > t_loss, sc[KNN_, anom] > t_lat)
        alarms2[left, s] = quadrants(sc[LOSS, test] > t_loss, sc[KNN_, test] > t_lat)
        for j in range(n_kind):
            v = sc[:, kind_slice(j)]
            kind_quad[j, left, s] = quadrants(v[LOSS] > t_loss, v[KNN_] > t_lat)
    chosen = int(normal_nll.mean((0, 1)).argmin())
    lat = (GAUSS, MIXT, FLOW)[chosen]
    n_m = len(METHODS)
    m_auc = np.zeros((10, n_seed, n_m))
    caught = np.zeros((len(ALPHAS), 10, n_seed, n_m))
    alarms = np.zeros_like(caught)
    k_auc, k_caught = np.zeros_like(m_auc), np.zeros_like(caught)
    or_raw = np.zeros((len(ALPHAS), 10, n_seed, 3))       # caught, false alarms, fair rate
    rho = np.zeros((10, n_seed))
    for left, s in np.ndindex(10, n_seed):
        sc = d["scores"][left, s, :, :HELD].astype(float)
        cal, test, anom = groups(yt, left)
        kept = np.concatenate([cal, test])
        c, cov = combine(sc[LOSS], sc[lat], cal)
        rho[left, s] = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        for k, name in enumerate(METHODS):
            v = c[name]
            m_auc[left, s, k] = P.auc(v[kept], v[anom])
            for a, alpha in enumerate(ALPHAS):
                t = cut(v[cal], alpha)
                caught[a, left, s, k] = (v[anom] > t).mean()
                alarms[a, left, s, k] = (v[test] > t).mean()
        for a, alpha in enumerate(ALPHAS):
            t1, t2 = cut(c["loss"][cal], alpha), cut(c["latent"][cal], alpha)
            flag = (c["loss"] > t1) | (c["latent"] > t2)
            or_raw[a, left, s] = (flag[anom].mean(), flag[test].mean(),
                                  1.0 - cut(c["either"][cal], alpha))
        c = combine(sc[LOSS], sc[KNN_], cal)[0]       # the same with the neighbour distance
        for k, name in enumerate(METHODS):
            k_auc[left, s, k] = P.auc(c[name][kept], c[name][anom])
            for a, alpha in enumerate(ALPHAS):
                k_caught[a, left, s, k] = (c[name][anom] > cut(c[name][cal], alpha)).mean()
    return dict(auc=auc, kind_auc=kind_auc, normal_nll=normal_nll, quad=quad,
                kind_quad=kind_quad, alarms2=alarms2, chosen=chosen, m_auc=m_auc,
                caught=caught, alarms=alarms, or_raw=or_raw, rho=rho, k_auc=k_auc,
                k_caught=k_caught,
                train_nll=d["train_nll"].astype(float))


def spoiled_inputs(left, code, seed):
    """The spoiled digits run ``(left, code, seed)`` scored: ``[kind, digit, 784]``."""
    xt, yt = P.ae_data()[2:4]
    return spoil(xt[groups(yt, left)[1][:N_COR]], np.random.default_rng([seed, left, code, 157]))


def mahalanobis(nll, volume, d):
    """The Mahalanobis distance behind a Gaussian's minus log density."""
    return np.sqrt(np.maximum(2.0 * (np.asarray(nll, float) - volume - 0.5 * d * np.log(2 * np.pi)),
                              0.0))


def _hist(values, edges):
    v = np.minimum(np.asarray(values, float), edges[-1] - 1e-12)
    return np.histogram(v, edges)[0] / max(len(v), 1)


COUNTERPART = ("A model-independent search: a model is trained on known processes only "
               "(simulated or measured background) and events it finds unusual are kept as "
               "candidates, with no signal model. Here real digits stand in for events, the "
               "nine kept digit classes for the known processes and the removed class (or a "
               "spoiled digit) for new physics.")
NORMAL_ONLY = ("Thresholds are quantiles of the scores of held-out digits of the kept classes "
               "at even positions; false alarms are counted on those at odd positions. "
               "Density fits use the training digits' codes. No anomaly enters either.")
CASES = ("both", "loss only", "code only", "neither")


# ====================================================================== 157
def anomaly_loss_and_latent():
    x, y, xt, yt, meta = P.ae_data()
    d2, d8 = arrays(DRAWN), arrays(MAIN)
    s2, s8 = summary(DRAWN), summary(MAIN)
    left, k0 = FEATURED, 0
    seed = SEEDS[k0]
    cal, test, anom = groups(yt, left)
    inv = KINDS.index("inverted")

    # ---- one event, two scores: the 2-D code
    sc = d2["scores"][left, k0].astype(float)
    typ = d2["typical"][left, k0]
    cor = int(d2["cor_typical"][left, k0, inv])
    spoiled2 = spoiled_inputs(left, DRAWN, seed)
    z, zc = d2["z"][left, k0].astype(float), d2["zc"][left, k0].astype(float)
    at = [int(typ[0]), int(typ[1]), HELD + inv * N_COR + cor]
    pics = np.array([xt[typ[0]], d2["pics"][left, k0, 0], xt[typ[1]], d2["pics"][left, k0, 1],
                     spoiled2[inv, cor], d2["pics"][left, k0, 2 + inv]], float)
    codes = np.array([z[typ[0]], z[typ[1]], zc[inv, cor]])
    thr2 = np.array([cut(sc[LOSS, cal], ALPHA), cut(sc[KNN_, cal], ALPHA)])
    cloud_labels = y[y != left][d2["cloud_ids"][left, k0]]
    e_loss = np.linspace(0.0, float(np.quantile(sc[LOSS, :HELD], 0.995)), BINS + 1)
    e_knn = np.linspace(0.0, float(np.quantile(sc[KNN_, :HELD], 0.995)) * 1.6, BINS + 1)
    members = (test, anom, np.arange(HELD + inv * N_COR, HELD + (inv + 1) * N_COR))
    flagged2 = np.array([[(sc[k, m] > t).mean() for m in members]
                         for k, t in ((LOSS, thr2[0]), (KNN_, thr2[1]))])

    # ---- the four cases, on the code of 8 numbers
    sc8 = d8["scores"][left, k0].astype(float)
    thr8 = np.array([cut(sc8[LOSS, cal], ALPHA), cut(sc8[KNN_, cal], ALPHA)])
    spoiled8 = spoiled_inputs(left, MAIN, seed)
    pools = [("unseen %d" % left, anom, xt[anom])] + [
        (KIND_NAMES[k], np.arange(HELD + j * N_COR, HELD + (j + 1) * N_COR), spoiled8[j])
        for j, k in enumerate(KINDS)]
    run_quad = np.array([quadrants(sc8[LOSS, m] > thr8[0], sc8[KNN_, m] > thr8[1])
                         for _, m, _ in pools])
    ex_pics, ex_names, ex_scores, ex_share, ex_kind = [], [], [], [], []
    for q in range(4):
        j = int(run_quad[:, q].argmax())
        name, m, pictures = pools[j]
        a, b = sc8[LOSS, m] > thr8[0], sc8[KNN_, m] > thr8[1]
        inside = np.flatnonzero((a if q in (0, 1) else ~a) & (b if q in (0, 2) else ~b))
        pick = inside[np.argsort(sc8[LOSS, m][inside], kind="stable")[len(inside) // 2]]
        ex_pics.append(pictures[pick])
        ex_names.append(name)
        ex_kind.append(j)
        ex_scores.append([sc8[LOSS, m][pick], sc8[KNN_, m][pick]])
        ex_share.append(run_quad[j, q])
    # the unseen class's own best example of "code only", if it has one
    a, b = sc8[LOSS, anom] > thr8[0], sc8[KNN_, anom] > thr8[1]
    only = np.flatnonzero(~a & b)
    own = anom[only[np.argsort(sc8[KNN_, anom][only], kind="stable")[len(only) // 2]]] \
        if len(only) else anom[0]

    e8 = (np.linspace(0.0, float(np.quantile(sc8[LOSS, :HELD], 0.995)), BINS + 1),
          np.linspace(0.0, float(np.quantile(sc8[KNN_, :HELD], 0.995)) * 1.3, BINS + 1))
    hist8 = np.array([[_hist(sc8[k, m], e) for m in (test, anom)]
                      for k, e in zip((LOSS, KNN_), e8)])
    flagged8 = np.array([[(sc8[k, m] > t).mean() for m in (test, anom)]
                         for k, t in zip((LOSS, KNN_), thr8)])
    quad = np.concatenate([s8["quad"][None], s8["kind_quad"]]).mean((1, 2))   # [9 kinds, 4]
    quad_by_left = s8["quad"]                                                  # [left, seed, 4]
    auc8, auc2 = s8["auc"], s2["auc"]
    order = [LOSS, KNN_, GAUSS]
    by_left = auc8.mean(1)                                                     # [left, score]
    rescued = [c for c in range(10) if by_left[c, LOSS] < 0.5 < by_left[c, KNN_]]
    kind_auc = s8["kind_auc"].mean((1, 2))                                     # [kind, score]
    # each kind's home: where the larger part of what is caught is caught
    names = ["unseen class"] + [KIND_NAMES[k] for k in KINDS]
    return Study(
        "Two places an anomaly can show",
        "An autoencoder trained on MNIST with one digit class removed; every held-out digit "
        "scored twice, by its reconstruction error and by the distance from its code to the "
        "training codes: 10 removed classes x %d seeds, codes of %d and %d numbers, and %d "
        "ways of spoiling a real digit" % (len(SEEDS), DRAWN, MAIN, len(KINDS)),
        (
            Panel("One digit, two scores",
                  "values: three held-out pictures and what the autoencoder with a code of "
                  "%d numbers returns (removed class %d, first seed): a digit of a kept "
                  "class, one of the removed class (each the one whose error is its group's "
                  "median) and an inverted digit. errors: mean squared pixel errors; codes: "
                  "their two code numbers; knn: distance from the code to its %dth nearest "
                  "training code; mahalanobis: distance from the centre of the training "
                  "codes in units of their spread." % (DRAWN, left, KNN), "images",
                  dict(values=pics.reshape(6, 28, 28),
                       labels=["seen: in", "seen: out", "unseen: in", "unseen: out",
                               "inverted: in", "inverted: out"],
                       errors=sc[LOSS, at], knn=sc[KNN_, at], codes=codes,
                       mahalanobis=mahalanobis(sc[GAUSS, at], float(d2["volume"][left, k0]),
                                               DRAWN),
                       classes=yt[typ], ids=typ, thresholds=thr2,
                       train_examples=np.array([x[np.flatnonzero(y == c)[0]] for c in
                                                range(10)], float).reshape(10, 28, 28))),
            Panel("The two spaces, for a code of two numbers",
                  "xy: the codes of %d training digits (classes: their digit classes); "
                  "unseen / inverted / seen: codes of held-out digits of the removed class, "
                  "of inverted digits and of kept classes. loss_hist / knn_hist: the share "
                  "of seen, unseen and inverted digits in each bin of each score (the last "
                  "bin holds everything above it). thresholds: the values 95%% of the "
                  "calibration digits stay under. flagged[score, group]: shares above them."
                  % N_CLOUD, "points",
                  dict(xy=d2["cloud"][left, k0].astype(float), classes=cloud_labels,
                       unseen=z[anom], inverted=zc[inv, :300], seen=z[test][:300],
                       box=d2["box"][left, k0], mean=d2["mean"][left, k0],
                       L=d2["L"][left, k0],
                       loss_edges=e_loss, knn_edges=e_knn,
                       loss_hist=np.array([_hist(sc[LOSS, m], e_loss) for m in members]),
                       knn_hist=np.array([_hist(sc[KNN_, m], e_knn) for m in members]),
                       thresholds=thr2, flagged=flagged2,
                       labels=["seen", "unseen", "inverted"])),
            Panel("Caught by both, by one, by neither",
                  "values[kind, case]: the share of each kind of anomaly (the unseen class, "
                  "then %d ways of spoiling a kept digit) that is above both thresholds, "
                  "above the loss threshold only, above the code threshold only, above "
                  "neither; code of %d numbers, each threshold passing 95%% of calibration "
                  "digits, mean of 10 removed classes x %d seeds. by_class[left, seed, case]: "
                  "the unseen class alone. normal[case]: the same shares for held-out digits "
                  "of kept classes (false alarms). examples: for each case, a real digit of "
                  "the kind most often in it (removed class %d, first seed). hist[score, group]: "
                  "the share of kept and of removed held-out digits in each bin of each "
                  "score in that run (edges; the last bin holds everything above it). "
                  "kind_pictures: one kept digit, and the same digit spoiled each way."
                  % (len(KINDS), MAIN, len(SEEDS), left), "matrix",
                  dict(values=quad, labels=names, cases=list(CASES), by_class=quad_by_left,
                       normal=s8["alarms2"].mean((0, 1)), seeds=np.concatenate(
                           [s8["quad"][None], s8["kind_quad"]]).mean(1),
                       run=run_quad, examples=np.array(ex_pics, float).reshape(4, 28, 28),
                       example_names=ex_names, example_kind=np.array(ex_kind),
                       example_scores=np.array(ex_scores), example_share=np.array(ex_share),
                       edges=np.array(e8), hist=hist8, flagged=flagged8,
                       medians=np.array([[np.median(sc8[k, m]) for m in (test, anom)]
                                         for k in (LOSS, KNN_)]),
                       run_auc=auc8[left, k0, [LOSS, KNN_]],
                       kind_pictures=np.concatenate(
                           [xt[test[:1]], spoiled8[:, 0]]).reshape(-1, 28, 28).astype(float),
                       thresholds=thr8, own=xt[own].reshape(28, 28).astype(float),
                       own_scores=sc8[[LOSS, KNN_]][:, own], own_found=len(only) > 0)),
            Panel("Area under the ROC curve of each score alone",
                  "x: the removed class; ys: area for the reconstruction error, the %dth-"
                  "neighbour distance and the Mahalanobis distance, mean of %d seeds, code "
                  "of %d numbers (seeds[score, class, seed]; drawn: the same for the code "
                  "of %d). kinds[kind, score]: the same areas for spoiled digits against "
                  "kept ones." % (KNN, len(SEEDS), MAIN, DRAWN), "curve",
                  dict(x=np.arange(10), ys=by_left[:, order].T,
                       labels=["reconstruction error", "distance to the %dth nearest code" % KNN,
                               "Mahalanobis distance"],
                       seeds=auc8[:, :, order].transpose(2, 0, 1),
                       drawn=auc2[:, :, order].transpose(2, 0, 1),
                       all_scores=auc8.mean((0, 1)), score_names=list(SCORES),
                       kinds=kind_auc[:, order], kind_names=names[1:])),
        ),
        dict(
            mnist=True, review_only=True, dataset=meta, source="NumPy training on MNIST",
            evidence="measured on MNIST: every number is from held-out digits",
            seeds=list(SEEDS), codes=[DRAWN, MAIN], featured=left, alpha=ALPHA,
            model="the autoencoder of autoencoder-anomaly-detection, retrained from the same "
                  "random streams: 784 -> %d ReLU -> code (linear) -> %d ReLU -> 784 sigmoid; "
                  "Adam %g, batch %d, %d updates on the training digits of the nine kept "
                  "classes" % (P.AE_HIDDEN, P.AE_HIDDEN, P.AE_LR, P.AE_BATCH, P.AE_STEPS),
            scores=dict(loss="mean over 784 pixels of (output - input)^2",
                        knn="Euclidean distance from the code to its %dth nearest among the "
                            "codes of %d training digits" % (KNN, KNN_REF),
                        gauss="Mahalanobis distance from the mean of the training codes"),
            latent_score_used="knn",
            chosen_by_looking_at_anomalies=(
                "The latent score used for the four cases is the %dth-neighbour distance "
                "rather than the Mahalanobis distance. That was decided after seeing the "
                "areas: the training codes form several clumps, an unseen digit's code "
                "falls between them, nearer the centre than a typical seen digit's, and "
                "the Mahalanobis distance then scores it as more normal (mean area %.2f "
                "against %.2f)." % (KNN, auc8[:, :, GAUSS].mean(), auc8[:, :, KNN_].mean())),
            normal_only=NORMAL_ONLY,
            auc_loss=_stat(auc8[:, :, LOSS]), auc_knn=_stat(auc8[:, :, KNN_]),
            auc_gauss=_stat(auc8[:, :, GAUSS]),
            auc_by_class={str(c): dict(loss=_stat(auc8[c, :, LOSS]), knn=_stat(auc8[c, :, KNN_]),
                                       gauss=_stat(auc8[c, :, GAUSS])) for c in range(10)},
            auc_drawn=dict(loss=float(auc2[:, :, LOSS].mean()), knn=float(auc2[:, :, KNN_].mean()),
                           gauss=float(auc2[:, :, GAUSS].mean())),
            rescued_classes=rescued,
            quadrants={n: {c: float(v) for c, v in zip(CASES, row)}
                       for n, row in zip(names, quad)},
            quadrants_featured={c: _stat(quad_by_left[left, :, q]) for q, c in enumerate(CASES)},
            false_alarms={c: float(v) for c, v in zip(CASES, s8["alarms2"].mean((0, 1)))},
            kinds=list(KINDS), spoiled_per_run=N_COR,
            split_sizes=[P.AE_TRAIN, P.AE_TEST],
            held_out_per_class=[int((yt == c).sum()) for c in range(10)],
            counterpart=COUNTERPART,
            limitations="One small dense autoencoder with no pressure on where its codes "
                        "go: nothing in the training keeps an unseen input's code away "
                        "from the training codes, so the latent score is weak for inputs "
                        "that resemble seen digits. A variational autoencoder was not "
                        "tried. The pictures use a code of two numbers because it can be "
                        "drawn; the measurements use %d, the best size of the earlier "
                        "animation, itself chosen on anomalies. The ways of spoiling a "
                        "digit were written down before their scores were read and all "
                        "%d are reported. MNIST only; %d updates." % (MAIN, len(KINDS),
                                                                    P.AE_STEPS),
            designs_tried=[
                "a one-seed probe (removed classes 0, 1, 7, 9; codes 2, 8, 32, 128) of the "
                "Mahalanobis distance and the 5th-neighbour distance: the first came out "
                "below one half at small codes, the second above it for the unseen 1s at 8",
                "the same probe with a mixture and a flow fitted to the codes and eight "
                "ways of spoiling a digit, to see which kinds each score catches; all "
                "eight were kept",
                "the design recorded here, fixed after those probes: ten removed classes, "
                "three seeds, thresholds at 5% from half of the held-out kept digits, the "
                "featured class (1) fixed beforehand as the one the loss missed",
            ],
        ),
    )


# ====================================================================== 158
DENSITY_NAMES = ("one Gaussian", "a mixture of %d" % MIX, "a normalizing flow")
METHOD_NAMES = ("loss alone", "code alone", "either cut (OR)", "sum of the two", "ellipse")
PLANE_SHOWN = 400                 # held-out kept digits drawn in the plane


def _plane(left, k0, lat):
    """One run's held-out digits in the plane of the two standardised scores, and the
    five thresholds at :data:`ALPHA`, all from calibration digits."""
    yt = P.ae_data()[3]
    sc = arrays(MAIN)["scores"][left, k0, :, :HELD].astype(float)
    cal, test, anom = groups(yt, left)
    c, cov = combine(sc[LOSS], sc[lat], cal)
    fair = 1.0 - cut(c["either"][cal], ALPHA)         # the shared rate of the two OR-ed cuts
    cuts = dict(loss=cut(c["loss"][cal], ALPHA), latent=cut(c["latent"][cal], ALPHA),
                or_loss=cut(c["loss"][cal], fair), or_latent=cut(c["latent"][cal], fair),
                sum=cut(c["sum"][cal], ALPHA), plane=cut(c["plane"][cal], ALPHA))
    pts = np.column_stack([c["loss"], c["latent"]])
    caught = {k: float((c[k][anom] > cut(c[k][cal], ALPHA)).mean()) for k in METHODS}
    return dict(normal=pts[test][:PLANE_SHOWN], anomalies=pts[anom], cov=cov, fair=fair,
                cuts=cuts, caught=caught, scores=c, raw=sc, groups=(cal, test, anom),
                stats=np.array([[np.log(sc[LOSS, cal]).mean(), np.log(sc[LOSS, cal]).std()],
                                [sc[lat, cal].mean(), sc[lat, cal].std()]]))


def anomaly_two_score_flow():
    x, y, xt, _yt, meta = P.ae_data()
    d2, d8 = arrays(DRAWN), arrays(MAIN)
    every = [summary(c) for c in CODES]
    s8 = every[CODES.index(MAIN)]
    left, k0 = FEATURED, 0
    chosen = s8["chosen"]
    lat = (GAUSS, MIXT, FLOW)[chosen]
    lat_name = DENSITY_NAMES[chosen]

    # ---- one event through the pipeline (code of 8 numbers)
    pl = _plane(left, k0, lat)
    other = _plane(0, k0, lat)
    cal, test, anom = pl["groups"]
    typ = int(d8["typical"][left, k0, 1])
    raw, c = pl["raw"], pl["scores"]
    event = dict(loss=float(raw[LOSS, typ]), latent=float(raw[lat, typ]),
                 z=[float(c["loss"][typ]), float(c["latent"][typ])],
                 loss_cut=cut(raw[LOSS, cal], ALPHA), latent_cut=cut(raw[lat, cal], ALPHA))
    decisions = [bool(c[k][typ] > cut(c[k][cal], ALPHA)) for k in METHODS]

    # ---- the densities
    nll = s8["normal_nll"]                                    # [left, seed, density]
    dens_auc = s8["auc"][:, :, [GAUSS, MIXT, FLOW, KNN_]]     # [left, seed, 4]
    box = d2["box"][left, k0]
    best_auc = int(dens_auc[:, :, :3].mean((0, 1)).argmax())

    # ---- the combinations
    m_auc = s8["m_auc"]                                       # [left, seed, method]
    caught, alarms, or_raw = s8["caught"], s8["alarms"], s8["or_raw"]
    a5 = ALPHAS.index(ALPHA)
    by_code = np.array([s["m_auc"] for s in every])           # [code, left, seed, method]
    knn_code = np.array([s["auc"][:, :, KNN_] for s in every])
    chosen_code = [int(s["chosen"]) for s in every]
    mean_code = by_code.mean((1, 2))                          # [code, method]
    _main, _last = CODES.index(MAIN), len(CODES) - 1
    best_method = int(m_auc.mean((0, 1)).argmax())
    best_caught = int(caught[a5].mean((0, 1)).argmax())
    return Study(
        "An anomaly search with two scores",
        "The whole procedure on MNIST with one digit class removed: an autoencoder trained "
        "on the kept classes, a density fitted to its codes (a Gaussian, a mixture, a "
        "normalizing flow), thresholds from held-out kept digits and three ways of combining "
        "the two scores: 10 removed classes x %d seeds x %d code sizes"
        % (len(SEEDS), len(CODES)),
        (
            Panel("One event through the procedure",
                  "values: a held-out digit of the removed class (%d; the one whose "
                  "reconstruction error is the median of its class) and what the "
                  "autoencoder with a code of %d numbers returns (first seed). event: its "
                  "reconstruction error and minus the log density of its code under %s, "
                  "with the thresholds that pass 95%% of calibration digits. decisions: "
                  "whether each of the five rules flags it. curve: the flow's training "
                  "curve (minus log density of whitened training codes, mean of every %d "
                  "updates)." % (left, MAIN, lat_name, FLOW_EVERY), "images",
                  dict(values=np.array([xt[typ], d8["pics"][left, k0, 1]],
                                       float).reshape(2, 28, 28),
                       labels=["in", "out"], event=np.array([event["loss"], event["latent"]]),
                       cuts=np.array([event["loss_cut"], event["latent_cut"]]),
                       z=np.array(event["z"]), stats=pl["stats"],
                       decisions=np.array(decisions), methods=list(METHOD_NAMES),
                       curve=d8["curve"][left, k0].astype(float),
                       curve_x=(np.arange(FLOW_STEPS // FLOW_EVERY) + 0.5) * FLOW_EVERY,
                       counts=np.array([int((y != left).sum()), len(cal), len(test),
                                        len(anom)]),
                       train_examples=np.array([x[np.flatnonzero(y == k)[0]] for k in
                                                range(10)], float).reshape(10, 28, 28))),
            Panel("Three densities fitted to the training codes",
                  "values[density]: minus the log density over a grid of the two code "
                  "numbers (code of %d, removed class %d, first seed; box: the grid's "
                  "extent), for one Gaussian, a mixture of %d and a flow of %d coupling "
                  "layers; cloud: training codes. normal_nll[class, seed, density]: mean "
                  "minus log density of calibration digits' codes at %d numbers - the "
                  "density with the lowest is the one used. auc[class, seed, density]: "
                  "area under the ROC curve of each as an anomaly score, with the %dth-"
                  "neighbour distance last." % (DRAWN, left, MIX, FLOW_LAYERS, MAIN, KNN),
                  "images",
                  dict(values=d2["dens"][left, k0].astype(float), box=box,
                       cloud=d2["cloud"][left, k0][:500].astype(float),
                       centres=d2["centres"][left, k0].astype(float),
                       labels=list(DENSITY_NAMES), normal_nll=nll, train_nll=s8["train_nll"],
                       auc=dens_auc, chosen=chosen, best_by_auc=best_auc,
                       drawn_nll=summary(DRAWN)["normal_nll"][left, k0])),
            Panel("The plane of the two scores",
                  "xy: normal then anomalies (classes: 0, 1). normal / anomalies: held-out "
                  "digits of kept classes (%d of them) and "
                  "of the removed class %d as (z1, z2): ln of the reconstruction error and "
                  "minus log density of the code, each in standard deviations of the "
                  "calibration digits from their mean (code of %d, first seed). cuts: the "
                  "thresholds at 5%% false alarms: loss alone, code alone, the two OR-ed "
                  "cuts at their shared rate (fair), the sum, the ellipse (cov: the "
                  "calibration cloud's covariance). other_*: the same with class 0 removed."
                  % (PLANE_SHOWN, left, MAIN), "points",
                  dict(xy=np.concatenate([pl["normal"], pl["anomalies"]]),
                       classes=np.repeat([0, 1], [len(pl["normal"]), len(pl["anomalies"])]),
                       normal=pl["normal"], anomalies=pl["anomalies"], cov=pl["cov"],
                       cuts=np.array([pl["cuts"][k] for k in
                                      ("loss", "latent", "or_loss", "or_latent", "sum",
                                       "plane")]),
                       fair=pl["fair"], caught=np.array([pl["caught"][k] for k in METHODS]),
                       other_normal=other["normal"], other_anomalies=other["anomalies"],
                       other_cov=other["cov"],
                       other_cuts=np.array([other["cuts"][k] for k in
                                            ("loss", "latent", "or_loss", "or_latent", "sum",
                                             "plane")]),
                       other_caught=np.array([other["caught"][k] for k in METHODS]),
                       labels=["kept classes", "removed class"], methods=list(METHOD_NAMES))),
            Panel("What each rule catches, and the code size",
                  "x: numbers in the code; ys[method]: area under the ROC curve, mean of "
                  "10 removed classes x %d seeds, for the loss alone, the code's density "
                  "alone and the three combinations (seeds[code, class, seed, method]); "
                  "knn: the %dth-neighbour distance. At %d numbers: caught[alpha, class, "
                  "seed, method] and alarms: the share of the removed class and of test "
                  "kept digits above each threshold set at 1%% and 5%%; or_raw[alpha, "
                  "class, seed]: (caught, false alarms, fair shared rate) when the two "
                  "single cuts are OR-ed without correction."
                  % (len(SEEDS), KNN, MAIN), "curve",
                  dict(x=np.array(CODES), ys=mean_code.T, labels=list(METHOD_NAMES),
                       seeds=by_code, knn=knn_code.mean((1, 2)), knn_seeds=knn_code,
                       chosen=np.array(chosen_code), auc=m_auc, caught=caught, alarms=alarms,
                       or_raw=or_raw, alphas=np.array(ALPHAS), rho=s8["rho"])),
        ),
        dict(
            mnist=True, review_only=True, dataset=meta, source="NumPy training on MNIST",
            evidence="measured on MNIST: every number is from held-out digits",
            seeds=list(SEEDS), codes=list(CODES), code_shown=MAIN, featured=left,
            alphas=list(ALPHAS),
            model="the autoencoder of autoencoder-anomaly-detection, retrained from the same "
                  "random streams (784 -> %d -> code -> %d -> 784, %d updates)"
                  % (P.AE_HIDDEN, P.AE_HIDDEN, P.AE_STEPS),
            densities=dict(
                gauss="one Gaussian: the mean and covariance of the training codes",
                mix="a mixture of %d full-covariance Gaussians, %d rounds of EM on %d "
                    "whitened training codes, %g added to every variance"
                    % (MIX, MIX_STEPS, MIX_FIT, MIX_RIDGE),
                flow="a normalizing flow on the whitened training codes: %d affine coupling "
                     "layers, %d tanh units each, scale at most exp(+-%g), Adam %g, batch "
                     "%d, %d updates" % (FLOW_LAYERS, FLOW_HIDDEN, FLOW_SCALE, FLOW_LR,
                                         FLOW_BATCH, FLOW_STEPS)),
            density_used=DENSITIES[chosen],
            density_chosen_by="the lowest mean minus log density of calibration digits' "
                              "codes (digits of kept classes only), over all removed "
                              "classes and seeds",
            density_best_by_auc=DENSITIES[best_auc],
            normal_nll={k: float(v) for k, v in zip(DENSITIES, nll.mean((0, 1)))},
            density_auc={k: _stat(dens_auc[:, :, n].mean(0))
                         for n, k in enumerate(DENSITIES + ("knn",))},
            combinations=dict(
                either="flag when either score is above its own threshold; the two "
                       "thresholds share one false-alarm rate, set so that the pair "
                       "flags the stated share of calibration digits",
                sum="z1 + z2 above a threshold (z: a score in standard deviations of the "
                    "calibration digits; z1 uses ln of the reconstruction error)",
                plane="outside an ellipse of constant Mahalanobis distance from the "
                      "calibration digits in the (z1, z2) plane; it also flags digits "
                      "unusually low in a score"),
            auc={k: _stat(m_auc[:, :, n].mean(0)) for n, k in enumerate(METHODS)},
            auc_by_class={str(k): {m: float(m_auc[k, :, n].mean())
                                   for n, m in enumerate(METHODS)} for k in range(10)},
            caught={str(a): {k: _stat(caught[i, :, :, n].mean(0))
                             for n, k in enumerate(METHODS)} for i, a in enumerate(ALPHAS)},
            false_alarms={str(a): {k: float(alarms[i, :, :, n].mean())
                                   for n, k in enumerate(METHODS)}
                          for i, a in enumerate(ALPHAS)},
            or_uncorrected={str(a): dict(caught=float(or_raw[i, :, :, 0].mean()),
                                         false_alarms=float(or_raw[i, :, :, 1].mean()),
                                         fair_rate=float(or_raw[i, :, :, 2].mean()),
                                         if_independent=1 - (1 - a) ** 2)
                            for i, a in enumerate(ALPHAS)},
            correlation_of_scores=_stat(s8["rho"]),
            best_auc=METHODS[best_method], best_caught=METHODS[best_caught],
            auc_by_code={str(k): {m: float(mean_code[i, n]) for n, m in enumerate(METHODS)}
                         for i, k in enumerate(CODES)},
            knn_by_code={str(k): float(v) for k, v in zip(CODES, knn_code.mean((1, 2)))},
            density_by_code={str(k): DENSITIES[v] for k, v in zip(CODES, chosen_code)},
            event=dict(event, decisions=dict(zip(METHODS, decisions))),
            normal_only=NORMAL_ONLY,
            chosen_by_looking_at_anomalies=(
                "Nothing in the procedure: the density is chosen on kept digits, the "
                "thresholds are set on kept digits. What does use anomalies is the "
                "comparison itself (which rule catches more), the code size of %d taken "
                "from the earlier animation, and the latent score's design probes." % MAIN),
            split_sizes=[P.AE_TRAIN, P.AE_TEST],
            counterpart=COUNTERPART,
            limitations="A plain autoencoder: nothing in its training keeps an unseen "
                        "input's code away from the training codes, so the density of the "
                        "code is a weak score for inputs that resemble seen digits. The "
                        "flow is small (%d updates) so that a run stays quick; a longer-"
                        "trained flow fitted the kept digits better in a one-run probe. "
                        "Thresholds at 1%% rest on about %d calibration digits above them. "
                        "Four code sizes, not the earlier animation's seven. MNIST only."
                        % (FLOW_STEPS, round(0.01 * len(cal))),
            designs_tried=[
                "a one-run probe of a larger flow (6 layers of 64 units, 2,000 updates) and "
                "toy2d's EM for the mixture: the flow fitted held-out kept digits slightly "
                "better than the mixture, but a run with 128 numbers took over a minute; "
                "both were made smaller and the mixture given a ridge",
                "a full build with a flow of 1,500 updates on batches of 256 and neighbours "
                "among all training codes: the same picture (areas at 8 numbers: loss 0.81, "
                "Gaussian 0.29, mixture 0.63, flow 0.61, neighbours 0.72; at 32: 0.70 for "
                "the loss, 0.86 for the density), but the four code sizes took four minutes "
                "on a loaded machine, so the flow and the neighbour search were trimmed. In "
                "that build the flow fitted kept digits' codes better than the mixture by "
                "0.02 nats and was the density used; with the trimmed flow the mixture is",
                "the ellipse was kept two-sided as fitted; a one-sided version was not "
                "tried",
                "the design recorded here, fixed after that probe: ten removed classes, "
                "three seeds, four code sizes, the density chosen by the calibration "
                "digits' log density, thresholds at 1% and 5%",
            ],
        ),
    )


STUDIES = {
    "anomaly-loss-and-latent": anomaly_loss_and_latent,
    "anomaly-two-score-flow": anomaly_two_score_flow,
}

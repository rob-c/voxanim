"""The experiment behind animation 176, ``kl-divergence``: cross-entropy is the target's
own entropy plus the KL divergence, a bridge from "Logits, loss and nats" (175) to the
places KL comes back later on the page (label smoothing, distillation, the KL term of a
VAE, t-SNE, RLHF's penalty).

* The trained classifier of :mod:`voxanim.study_info` (784-64-32-10, seed 0) is the model
  ``q``. For one test digit the identity ``CE(p, q) = H(p) + KL(p || q)`` is worked out
  term by term, class by class; each term is computed on its own, so that they add up is
  checked, not assumed.
* Three targets ``p`` for the same ``q`` on the 10,000 test digits: one-hot labels
  (``H = 0``, so the loss is all KL), label-smoothed targets (``eps = 0.1``) and a
  teacher's soft targets (the same classifier trained from seed 1, its logits softened at
  ``T = 2``). The untrained network (seed 0 before training) is scored too: ``H(p)``
  belongs to the target, so training can only shrink the KL part.
* KL is not symmetric: ``KL(p || q)`` against ``KL(q || p)`` for those targets (for a
  one-hot target the second is infinite: ``q`` puts mass where ``p`` has none), and one
  Gaussian fitted to a two-humped measured distribution, the training 0s and 1s along the
  difference of their average pictures, by minimising each direction: forward KL covers
  both humps, reverse KL sits on one.

Plain numpy on MNIST; the teacher's run and the fits are cached under their settings and
the source of the functions that make them.
"""

from __future__ import annotations

import functools
import hashlib
import inspect

import numpy as np

from . import mnist
from . import study_info as S
from .cache import cached_npz
from .study import Panel, Study

VERSION = 1
CLASSES = S.CLASSES
EPS = 0.1                          # label smoothing: this much of each target spread over all ten
TEACHER_SEED = 1                   # the teacher: the same classifier from another seed
TEACHER_T = 2.0                    # its logits divided by this before the softmax
SHOWN = S.SHOWN_DIGIT              # test digit 0, a 7, as in "Logits, loss and nats"
HUMPS = (1, 0)                     # the two digit classes of the two-humped distribution
BANDWIDTH = 0.2                    # kernel width of the smoothed distribution p
GRID = 1601                        # points of the grid p and the Gaussians live on
REACH = 8.0                        # the grid spans the data's mean +- REACH standard deviations
COARSE = (241, 121)                # the first search: centres x widths
ROUNDS = 10                        # then this many zooms of a 21 x 21 search
TARGETS = ("one-hot", "smoothed", "teacher")
EVIDENCE = dict(
    exact="exact arithmetic: H, KL and cross-entropy worked out from measured probabilities",
    measured="measured on MNIST: classifiers trained here in numpy, scored on the 10,000 "
             "held-out digits; the two-humped distribution from the training 0s and 1s",
)


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them."""
    h = hashlib.sha256()
    for fn in sources:
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-bridge-kl-" + name, settings,
                                                    build).items()}


def _floats(v):
    return [float(x) for x in np.asarray(v).ravel()]


# ====================================================================== the three quantities
def entropy_terms(p):
    """``-p ln p`` for each class (0 where ``p`` is 0), nats."""
    p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, -p * np.log(np.where(p > 0, p, 1.0)), 0.0)


def cross_entropy_terms(p, lq):
    """``-p ln q`` for each class, from ``q``'s log-probabilities ``lq``."""
    return -np.asarray(p, float) * np.asarray(lq, float)


def kl_terms(p, lq):
    """``p ln(p / q)`` for each class (0 where ``p`` is 0): computed on its own, not as
    the difference of the other two."""
    p = np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, p * (np.log(np.where(p > 0, p, 1.0)) - lq), 0.0)


def reverse_kl(lq, p):
    """``KL(q || p) = sum q ln(q / p)`` along the last axis: infinite where ``q`` puts
    mass on a class that ``p`` gives none."""
    q, p = np.exp(np.asarray(lq, float)), np.asarray(p, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(q > 0, q * (lq - np.log(p)), 0.0)
    return t.sum(-1)


def smoothed(y, eps=EPS):
    """Label-smoothed targets: ``1 - eps`` on the label plus ``eps / 10`` on every class."""
    return (1.0 - eps) * np.eye(CLASSES)[y] + eps / CLASSES


# ====================================================================== the teacher
def _teacher_build():
    _P0, P = S.train_mlp(TEACHER_SEED)[1:]
    xt = mnist.load()[2].astype(np.float32)
    return dict(logits=S.logits(P, xt).astype(np.float64))


@functools.lru_cache(maxsize=1)
def teacher_logits():
    """The seed-1 classifier's logits on the 10,000 test digits (cached)."""
    return _cache("teacher", _teacher_build,
                  (S.init_mlp, S.logits, S.log_softmax, S.mlp_grads, S.train_mlp,
                   _teacher_build),
                  sizes=S.SIZES, rate=S.RATE, batch=S.BATCH, epochs=S.EPOCHS,
                  seed=TEACHER_SEED, weight_seed=S.WEIGHT_SEED)["logits"]


def targets():
    """``(p, lq, lq0, yt)``: the three targets on the test digits (``{name: (n, 10)}``),
    the trained model's log-probabilities, the untrained one's, and the labels."""
    _z, lq, yt, _prior = S.test_scores()
    xt = mnist.load()[2].astype(np.float32)
    lq0 = S.log_softmax(S.logits(S.params("before"), xt))
    p = {"one-hot": np.eye(CLASSES)[yt], "smoothed": smoothed(yt),
         "teacher": np.exp(S.log_softmax(teacher_logits() / TEACHER_T))}
    return p, lq, lq0, yt


# ====================================================================== two humps, one bell
def hump_feature():
    """``(v, y, direction)``: every training 0 and 1 projected on the unit picture
    ``average 0 - average 1`` (one number a digit: more like a 0, more like a 1), their
    labels and the picture."""
    x, y = mnist.load()[:2]
    keep = np.isin(y, HUMPS)
    x, y = x[keep], y[keep]
    d = x[y == HUMPS[1]].mean(0) - x[y == HUMPS[0]].mean(0)
    d = d / np.linalg.norm(d)
    return x @ d, y.astype(np.int64), d


def logsumexp(a, axis=-1):
    m = np.max(a, axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis, keepdims=True))).squeeze(axis)


def kde_log_p(grid, v, bandwidth=BANDWIDTH):
    """The log of the digits' distribution on ``grid``: each digit a Gaussian bump of
    width ``bandwidth``, normalised over the grid's points. In logs, so that ``p`` is
    positive everywhere and the reverse KL of any Gaussian is finite."""
    out = np.empty(len(grid))
    for a in range(0, len(grid), 200):
        g = grid[a:a + 200, None]
        out[a:a + 200] = logsumexp(-0.5 * ((g - v[None]) / bandwidth) ** 2, 1)
    return out - logsumexp(out)


def gaussian_log_q(grid, mu, sd):
    """Log of Gaussians (centres ``mu``, widths ``sd``, broadcast) on ``grid``,
    normalised over its points: ``(..., len(grid))``."""
    mu, sd = np.asarray(mu, float)[..., None], np.asarray(sd, float)[..., None]
    a = -0.5 * ((grid - mu) / sd) ** 2
    return a - logsumexp(a)[..., None]


def forward_kl(lp, lq):
    """``KL(p || q)`` from logs, along the last axis."""
    return (np.exp(lp) * (lp - lq)).sum(-1)


def reverse_kl_grid(lp, lq):
    """``KL(q || p)`` from logs, along the last axis."""
    return (np.exp(lq) * (lq - lp)).sum(-1)


def fit_gaussian(grid, lp, direction, v):
    """``(mu, sd, kl)``: the Gaussian on ``grid`` closest to ``p`` by forward or reverse
    KL: a search over centres within the data and widths from the kernel's to twice the
    data's spread, then ``ROUNDS`` zooms of a 21 x 21 search around the best."""
    kl = forward_kl if direction == "forward" else reverse_kl_grid
    mus = np.linspace(v.min(), v.max(), COARSE[0])
    sds = np.geomspace(BANDWIDTH, 2 * v.std(), COARSE[1])
    best = (np.inf, 0.0, 0.0)
    for s in sds:
        val = kl(lp, gaussian_log_q(grid, mus, s))
        k = int(np.argmin(val))
        if val[k] < best[0]:
            best = (float(val[k]), float(mus[k]), float(s))
    _, mu, sd = best
    dm, ds = 2 * (mus[1] - mus[0]), 2 * (sds[1] / sds[0] - 1)
    for _ in range(ROUNDS):
        M, F = np.meshgrid(mu + np.linspace(-dm, dm, 21), 1 + np.linspace(-ds, ds, 21))
        val = kl(lp, gaussian_log_q(grid, M.ravel(), sd * F.ravel()))
        k = int(np.argmin(val))
        mu, sd = float(M.ravel()[k]), float(sd * F.ravel()[k])
        dm, ds = dm / 3, ds / 3
    return mu, sd, float(kl(lp, gaussian_log_q(grid, mu, sd)))


def _humps_build():
    v, y, d = hump_feature()
    m, s = float(v.mean()), float(v.std())
    grid = np.linspace(m - REACH * s, m + REACH * s, GRID)
    lp = kde_log_p(grid, v)
    out = dict(v=v, y=y, direction=d, grid=grid, lp=lp)
    for name in ("forward", "reverse"):
        out[name] = np.array(fit_gaussian(grid, lp, name, v))
    return out


@functools.lru_cache(maxsize=1)
def humps():
    """The two-humped distribution and its two Gaussian fits (cached)."""
    return _cache("humps", _humps_build,
                  (hump_feature, logsumexp, kde_log_p, gaussian_log_q, forward_kl,
                   reverse_kl_grid, fit_gaussian, _humps_build),
                  humps=HUMPS, bandwidth=BANDWIDTH, grid=GRID, reach=REACH, coarse=COARSE,
                  rounds=ROUNDS)


# ====================================================================== the study
def _split(p, lq):
    """Per digit: ``(H, KL, CE)``, each summed over the classes from its own terms."""
    return (entropy_terms(p).sum(1), kl_terms(p, lq).sum(1),
            cross_entropy_terms(p, lq).sum(1))


def kl_divergence():
    p, lq, lq0, yt = targets()
    n = len(yt)
    rows = {}
    for name in TARGETS:
        H, K, C = _split(p[name], lq)
        _, K0, C0 = _split(p[name], lq0)
        R = reverse_kl(lq, p[name])
        finite = np.isfinite(R)
        rows[name] = dict(
            entropy=float(H.mean()), kl=float(K.mean()), cross_entropy=float(C.mean()),
            kl_error=float(K.std(ddof=1) / np.sqrt(n)),
            identity_gap=float(np.abs(C - H - K).max()),
            untrained_kl=float(K0.mean()), untrained_cross_entropy=float(C0.mean()),
            reverse_kl=float(R.mean()) if finite.all() else None,
            reverse_infinite=int((~finite).sum()))
    # one digit, term by term
    i = SHOWN
    qi = np.exp(lq[i])
    ps = p["smoothed"][i]
    h, k, c = entropy_terms(ps), kl_terms(ps, lq[i]), cross_entropy_terms(ps, lq[i])
    label = int(yt[i])
    wrong = np.arange(CLASSES) != label
    shown = dict(digit=i, label=label, q=_floats(qi), p=_floats(ps),
                 entropy_terms=_floats(h), kl_terms=_floats(k), cross_entropy_terms=_floats(c),
                 entropy=float(h.sum()), kl=float(k.sum()), cross_entropy=float(c.sum()),
                 wrong_share=float(c[wrong].sum() / c.sum()),
                 onehot_loss=float(-lq[i, label]),
                 reverse_kl=float(reverse_kl(lq[i], ps)),
                 teacher=_floats(p["teacher"][i]))
    # two humps, one bell
    hb = humps()
    grid, lp, v, y = hb["grid"], hb["lp"], hb["v"], hb["y"]
    pg = np.exp(lp)
    mean_p = float((pg * grid).sum())
    sd_p = float(np.sqrt((pg * (grid - mean_p) ** 2).sum()))
    cls = {c: v[y == c] for c in HUMPS}
    m1, m0 = (float(cls[c].mean()) for c in HUMPS)
    inside = (grid > m1) & (grid < m0)
    valley = float(grid[inside][np.argmin(lp[inside])])
    fits = {}
    for name in ("forward", "reverse"):
        mu, sd, _ = (float(t) for t in hb[name])
        lqg = gaussian_log_q(grid, mu, sd)
        qg = np.exp(lqg)
        fits[name] = dict(mu=mu, sd=sd, kl_pq=float(forward_kl(lp, lqg)),
                          kl_qp=float(reverse_kl_grid(lp, lqg)),
                          mass_right=float(qg[grid > valley].sum()),
                          p_at_centre=float(np.exp(np.interp(mu, grid, lp)) / pg.max()))
    width = 12.0 * sd_p
    shown_x = (grid > mean_p - width / 2) & (grid < mean_p + width / 2)
    dx = float(grid[1] - grid[0])
    curves = np.array([pg] + [np.exp(gaussian_log_q(grid, fits[f]["mu"], fits[f]["sd"]))
                              for f in ("forward", "reverse")])[:, shown_x] / dx
    panels = (
        Panel("One digit, term by term", "test digit %d (a %d) against a label-smoothed "
              "target: -p ln q, -p ln p and p ln(p/q) for each class, nats" % (i, label),
              "bars",
              dict(values=c, entropy=h, kl=k, labels=[str(j) for j in range(CLASSES)],
                   p=ps, q=qi)),
        Panel("Three targets, one model", "mean over the 10,000 test digits: the target's "
              "entropy H(p), KL(p || q) and the cross-entropy, which is their sum", "bars",
              dict(values=np.array([rows[t]["cross_entropy"] for t in TARGETS]),
                   entropy=np.array([rows[t]["entropy"] for t in TARGETS]),
                   kl=np.array([rows[t]["kl"] for t in TARGETS]),
                   untrained=np.array([rows[t]["untrained_cross_entropy"] for t in TARGETS]),
                   labels=list(TARGETS))),
        Panel("Which way round", "mean KL(p || q) and KL(q || p) over the test digits for "
              "the two soft targets (for one-hot labels KL(q || p) is infinite)", "bars",
              dict(values=np.array([rows[t][w] for t in TARGETS[1:]
                                    for w in ("kl", "reverse_kl")]),
                   labels=["%s %s" % (t, w) for t in TARGETS[1:]
                           for w in ("KL(p||q)", "KL(q||p)")])),
        Panel("Two humps, one bell", "the training %ds and %ds along the difference of their "
              "average pictures, smoothed (p), and the Gaussian closest to it by forward "
              "and by reverse KL" % HUMPS, "curve",
              dict(x=grid[shown_x], ys=curves, labels=["p", "forward fit", "reverse fit"],
                   direction=hb["direction"].reshape(28, 28))),
    )
    return Study(
        "Cross-entropy is entropy plus KL", "CE(p, q) = H(p) + KL(p || q), measured on "
        "three targets for one classifier", panels,
        dict(source="the 784-64-32-10 classifier of 'Logits, loss and nats' (seed 0) as q; a "
                    "teacher from seed 1; every mean over the 10,000 test digits; the two "
                    "humps from the MNIST training 0s and 1s",
             dataset="MNIST", evidence=EVIDENCE, n=n, classes=CLASSES, eps=EPS,
             teacher_seed=TEACHER_SEED, teacher_temperature=TEACHER_T,
             model_seed=S.SEEDS[0], accuracy=float((lq.argmax(1) == yt).mean()),
             teacher_accuracy=float((teacher_logits().argmax(1) == yt).mean()),
             targets=list(TARGETS), rows=rows, shown=shown,
             humps=dict(classes=list(HUMPS), n={str(c): int(len(cls[c])) for c in HUMPS},
                        class_mean={str(c): float(cls[c].mean()) for c in HUMPS},
                        class_sd={str(c): float(cls[c].std()) for c in HUMPS},
                        share={str(c): float(len(cls[c]) / len(v)) for c in HUMPS},
                        bandwidth=BANDWIDTH, grid=GRID, p_mean=mean_p, p_sd=sd_p,
                        valley=valley, p_mass_right=float(pg[grid > valley].sum()),
                        forward=fits["forward"], reverse=fits["reverse"]),
             limitations=["one model and one teacher, each trained once",
                          "the teacher's temperature (2) and the smoothing (0.1) are fixed "
                          "choices, not tuned",
                          "the two humps are smoothed with a %.1f-wide kernel so that p is "
                          "never zero: the reverse KL of a Gaussian is then finite"
                          % BANDWIDTH]))


STUDIES = {"kl-divergence": kl_divergence}

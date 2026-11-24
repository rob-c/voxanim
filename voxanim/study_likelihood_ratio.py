"""A classifier learns the likelihood ratio, and reweights a simulation, for the two animations
of "Blocks and code" on the classifier and the likelihood ratio (catalogue 273-274):
``examples/likelihood_ratio.py`` in numpy. Everything is made up, with the script's seeds.

**Part one** (:func:`hypotheses`, :func:`run_one`): one measured value x. H0 (background): x
Gaussian at 0, spread 1; H1 (signal): at 1, spread 0.5. The exact likelihood ratio
p1(x) / p0(x) (:func:`log_ratio`) has a parabola for its log, highest at x = 4/3, since the
spreads differ: the best test keeps a window of x, not everything above a cut. 20,000 events
of each (the script's ``rng(273)``), a small MLP (1 -> 32 -> 32 -> 1, ReLU) trained with binary
cross-entropy (Adam, learning rate 0.0003, batches of :data:`BATCH`, :data:`STEPS_ONE` steps).
With equal numbers of each, the best output is s = p1 / (p0 + p1), so s / (1 - s) = p1 / p0.
Measured: the learned ratio on a grid of x against the exact one; the training loss against
the exact ratio's own loss on the same events (:func:`floor_on`; :func:`floor` for endless
events): no classifier can do better, beyond chance; the network's ROC on 20,000 fresh events of
each against the exact Neyman-Pearson ROC and the exact ROC of a single cut x > c; and the
network's own ROC computed exactly, from the window of x its output keeps (:func:`net_exact`).

**Part two** (:func:`samples`, :func:`run_two`): two measured values x, y. "Simulation": both
Gaussian at 0, spread 1, independent. "Data": x shifted to 0.4, y wider (1.25), correlation 0.4.
50,000 events of each (``rng(274)``) train the same kind of network (2 inputs,
:data:`STEPS_TWO` steps) to tell data (label 1) from simulation; each event of a fresh
simulated sample gets the weight s / (1 - s) = e^z (z the logit). The exact weight
(:func:`log_weight`) is known. Measured on fresh samples: histograms of x, y and x + y (a
combination the network was not given as a column) for data, simulation and weighted
simulation, with their chi-squares; where it fails: the largest weights against the exact
ones, the effective number of events (sum w)^2 / sum w^2, the data events in bins where the
simulation has none, and a network trained on only :data:`FEW` events of each.

The script draws its batches with ``torch.randint`` and starts from torch's own weights, so
its printed numbers differ a little from these: this module draws batches from its own seeded
generator and starts from :func:`init`, as PyTorch would (uniform within 1 / sqrt(fan in)).
``tests/test_likelihood_ratio_equivalence.py`` runs the script's own lines in torch from these
weights on these batches and matches the loss, the gradients and one Adam step. One run per
part (one seed each, plus one with few events); cached.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import math
import sys

import numpy as np

from . import study_course as S
from .cache import cached_npz

VERSION = 1
HIDDEN = 32
BATCH = 512
LR = 3e-4
KEYS = ("0.weight", "0.bias", "2.weight", "2.bias", "4.weight", "4.bias")
SEED_ONE, SEED_TWO = 273, 274            # the script's data seeds
N_ONE, N_TWO = 20000, 50000              # events of each kind
STEPS_ONE = STEPS_TWO = 10000
H0, H1 = (0.0, 1.0), (1.0, 0.5)          # (centre, spread) of x under each hypothesis
SIM_MEAN, DATA_MEAN = (0.0, 0.0), (0.4, 0.0)
DATA_COV = ((1.0, 0.5), (0.5, 1.5625))   # y spread 1.25, correlation 0.4
GRID = np.linspace(-3.0, 5.0, 161)       # the script's grid of x
BINS = np.linspace(-5.0, 5.0, 21)        # histograms of part two: 20 bins of 0.5
LOSS_EVERY = 50                          # the loss on all training events, every so many steps
ALPHA = 0.05                             # the false-alarm rate the power is read at
FEW = 1000                               # part two again, with this many events of each
TRACK = (-0.75, 2.25)                    # part one: x where both hypotheses have events
EDGE = (-1.0, 2.5)                       # part one: past these, too few events of one or both
FAR = 3.0                                # part one: an x beyond the events, where it drifts
FINE = np.linspace(-8.0, 8.0, 32001)     # part one: x for the network's exact ROC
SHOW = np.linspace(-3.0, 4.0, 29)        # part one: histograms on screen, bins of 0.25
PLANE = np.linspace(-4.0, 4.0, 41)       # part two: the plane of (x, y) the weights are drawn on


# ====================================================================== the network
def init(inputs, seed):
    """``classifier(inputs)`` as PyTorch starts it, keyed as its ``state_dict()``."""
    rng = np.random.default_rng([seed, 0])
    P = {}
    for k, (o, i) in zip(("0", "2", "4"), ((HIDDEN, inputs), (HIDDEN, HIDDEN), (1, HIDDEN)),
                         strict=True):
        b = 1 / np.sqrt(i)
        P[k + ".weight"] = rng.uniform(-b, b, (o, i)).astype(np.float32)
        P[k + ".bias"] = rng.uniform(-b, b, o).astype(np.float32)
    return P


def forward(P, x):
    """The logit ``z`` [n, 1] and what the backward pass needs."""
    p1 = x @ P["0.weight"].T + P["0.bias"]
    h1 = np.maximum(p1, 0)
    p2 = h1 @ P["2.weight"].T + P["2.bias"]
    h2 = np.maximum(p2, 0)
    z = h2 @ P["4.weight"].T + P["4.bias"]
    return z, (x, p1, h1, p2, h2)


def logit(P, x, chunk=20000):
    """The logit of every row of ``x``, as a flat float64 array."""
    x = np.asarray(x, np.float32)
    return np.concatenate([forward(P, x[s:s + chunk])[0][:, 0]
                           for s in range(0, len(x), chunk)]).astype(np.float64)


def sigmoid(z):
    z = np.asarray(z)
    with np.errstate(over="ignore"):
        return 1 / (1 + np.exp(-z))


def bce(z, y):
    """``nn.BCEWithLogitsLoss()`` (the mean over the batch)."""
    return float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))


def loss_and_grads(P, x, y):
    """``bce(net(x), y)`` and the gradient of every weight (float32, as torch's)."""
    z, (x, p1, h1, p2, h2) = forward(P, x)
    loss = bce(z.astype(np.float64), y)
    dz = ((sigmoid(z) - y) / len(x)).astype(np.float32)
    G = {"4.weight": dz.T @ h2, "4.bias": dz.sum(0)}
    d2 = (dz @ P["4.weight"]) * (p2 > 0)
    G["2.weight"], G["2.bias"] = d2.T @ h1, d2.sum(0)
    d1 = (d2 @ P["2.weight"]) * (p1 > 0)
    G["0.weight"], G["0.bias"] = d1.T @ x, d1.sum(0)
    return loss, {k: G[k].astype(np.float32) for k in KEYS}


def batch_indices(seed, steps, n):
    """The indices of every batch: ``[steps, BATCH]`` from 0 to ``n``."""
    return np.random.default_rng([seed, 1]).integers(0, n, (steps, BATCH))


def train(X, Y, steps, seed, record=None):
    """``train(net, X, Y, steps)`` of the script, from :func:`init`: the final weights, the
    loss on all of ``X`` every :data:`LOSS_EVERY` steps, the first step's loss and gradients
    and the weights after it."""
    P = init(X.shape[1], seed)
    st = S.adam_start(P)
    idx = batch_indices(seed, steps, len(X))
    out = {"start__" + k: v for k, v in P.items()}
    curve = []
    for step in range(steps):
        if step % LOSS_EVERY == 0:
            curve.append(bce(logit(P, X), Y[:, 0]))
        i = idx[step]
        loss, G = loss_and_grads(P, X[i], Y[i])
        P, st = S.adam_update(P, G, st, lr=LR)
        if step == 0:
            out["loss0"] = np.float64(loss)
            out.update({"grad0__" + k: v for k, v in G.items()})
            out.update({"step1__" + k: v for k, v in P.items()})
    curve.append(bce(logit(P, X), Y[:, 0]))
    out["curve"] = np.array(curve)
    out.update({"final__" + k: v for k, v in P.items()})
    return out


def weights(r, prefix):
    p = prefix + "__"
    return {k[len(p):]: v for k, v in r.items() if k.startswith(p)}


# ====================================================================== exact
def norm_cdf(z):
    z = np.asarray(z, float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))


def log_density(x, mu, sd):
    z = (np.asarray(x, float) - mu) / sd
    return -0.5 * z * z - math.log(sd) - 0.5 * math.log(2 * math.pi)


def log_ratio(x):
    """ln p1(x) / p0(x), exactly: ln 2 - 2 (x - 1)^2 + x^2 / 2."""
    return log_density(x, *H1) - log_density(x, *H0)


def peak():
    """Where the ratio is highest, and its value there."""
    (m0, s0), (m1, s1) = H0, H1
    a = 1 / s0 ** 2 - 1 / s1 ** 2              # ln r = a x^2 / 2 + b x + c, a < 0
    b = m1 / s1 ** 2 - m0 / s0 ** 2
    x = -b / a
    return float(x), float(np.exp(log_ratio(x)))


def np_window(d):
    """The best test keeps a window of x round the peak, ``peak ± d``: its exact false-alarm
    rate (H0 kept) and power (H1 kept)."""
    c = peak()[0]
    lo, hi = c - d, c + d
    alpha = norm_cdf((hi - H0[0]) / H0[1]) - norm_cdf((lo - H0[0]) / H0[1])
    power = norm_cdf((hi - H1[0]) / H1[1]) - norm_cdf((lo - H1[0]) / H1[1])
    return float(alpha), float(power)


def bisect(f, lo, hi, steps=100):
    """``x`` in ``[lo, hi]`` with ``f(x) = 0`` (``f`` rising)."""
    for _ in range(steps):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def np_at(alpha=ALPHA):
    """The Neyman-Pearson window at false-alarm rate ``alpha``: ``(lo, hi, power)``."""
    d = bisect(lambda d: np_window(d)[0] - alpha, 0.0, 10.0)
    c = peak()[0]
    return c - d, c + d, np_window(d)[1]


def cut_at(alpha=ALPHA):
    """A single cut x > c at false-alarm rate ``alpha``: ``(c, power)``, exact."""
    c = bisect(lambda c: alpha - (1 - float(norm_cdf((c - H0[0]) / H0[1]))), -10.0, 10.0)
    return c, 1 - float(norm_cdf((c - H1[0]) / H1[1]))


def exact_rocs(n=201):
    """``(alpha, power_np, power_cut)`` on a grid of false-alarm rates: the exact ROC of the
    likelihood ratio and of a single cut."""
    alphas = np.linspace(0.0, 1.0, n)
    a_np, p_np = zip(*(np_window(d) for d in np.linspace(0.0, 12.0, 4001)), strict=True)
    p_np = np.interp(alphas, np.array(a_np), np.array(p_np))
    p_cut = np.array([cut_at(a)[1] if 0 < a < 1 else float(a) for a in alphas])
    return alphas, p_np, p_cut


def floor():
    """The lowest binary cross-entropy any classifier can reach on equal numbers of H0 and H1:
    that of s = p1 / (p0 + p1), integrated exactly (on a fine grid)."""
    x = np.linspace(-9.0, 9.0, 180001)
    l0, l1 = log_density(x, *H0), log_density(x, *H1)
    lr = l1 - l0
    loss1 = np.logaddexp(0, -lr)               # -ln s for an H1 event
    loss0 = np.logaddexp(0, lr)                # -ln (1 - s) for an H0 event
    f = 0.5 * (np.exp(l1) * loss1 + np.exp(l0) * loss0)
    return float(np.sum(f) * (x[1] - x[0]))


# ====================================================================== part one: data, run
@functools.lru_cache(maxsize=1)
def hypotheses():
    """The script's events: ``(x0, x1, f0, f1)``, each ``[N_ONE, 1]`` float64 (training events
    of H0 and H1, then the fresh ones the power is measured on)."""
    rng = np.random.default_rng(SEED_ONE)
    x0 = rng.normal(*H0, (N_ONE, 1))
    x1 = rng.normal(*H1, (N_ONE, 1))
    f0 = rng.normal(*H0, (N_ONE, 1))
    f1 = rng.normal(*H1, (N_ONE, 1))
    return x0, x1, f0, f1


def tensors(a, b):
    """``X`` (float32, ``a`` then ``b``) and ``Y`` (0 for ``a``, 1 for ``b``), as the script's."""
    X = np.concatenate([a, b]).astype(np.float32)
    Y = np.zeros((len(X), 1), np.float32)
    Y[len(a):] = 1
    return X, Y


def _build_one():
    x0, x1, f0, f1 = hypotheses()
    X, Y = tensors(x0, x1)
    out = train(X, Y, STEPS_ONE, SEED_ONE)
    P = weights(out, "final")
    out["grid_logit"] = logit(P, GRID[:, None])
    out["fine_logit"] = logit(P, FINE[:, None])
    out["fresh0"] = logit(P, f0)
    out["fresh1"] = logit(P, f1)
    return out


# ====================================================================== part two: data, run
@functools.lru_cache(maxsize=1)
def samples():
    """The script's events: ``(sim, data, new, seen)``, each ``[N_TWO, 2]`` (training
    simulation and data, then fresh simulation to weight and fresh data to compare with)."""
    rng = np.random.default_rng(SEED_TWO)
    sim = rng.normal(0, 1, (N_TWO, 2))
    data = rng.multivariate_normal(DATA_MEAN, DATA_COV, N_TWO)
    new = rng.normal(0, 1, (N_TWO, 2))
    seen = rng.multivariate_normal(DATA_MEAN, DATA_COV, N_TWO)
    return sim, data, new, seen


def log_weight(p):
    """ln p_data(x, y) / p_sim(x, y), exactly."""
    p = np.atleast_2d(np.asarray(p, float))
    C = np.asarray(DATA_COV)
    d = p - np.asarray(DATA_MEAN)
    q = np.einsum("ni,ij,nj->n", d, np.linalg.inv(C), d)
    return -0.5 * q - 0.5 * math.log(np.linalg.det(C)) + 0.5 * (p ** 2).sum(1)


def _build_two():
    sim, data, new, _seen = samples()
    X, Y = tensors(sim, data)
    out = train(X, Y, STEPS_TWO, SEED_TWO)
    P = weights(out, "final")
    out["new_logit"] = logit(P, new)
    gx, gy = np.meshgrid(PLANE, PLANE)
    out["plane_logit"] = logit(P, np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)
    few = train(*tensors(sim[:FEW], data[:FEW]), STEPS_TWO, SEED_TWO + 1000)
    out["few_logit"] = logit(weights(few, "final"), new)
    out["few_curve"] = few["curve"]
    return out


def _settings(part):
    src = hashlib.sha256(inspect.getsource(sys.modules[__name__]).encode()).hexdigest()[:16]
    return dict(version=VERSION, part=part, hidden=HIDDEN, batch=BATCH, lr=LR,
                n=(N_ONE, N_TWO), steps=(STEPS_ONE, STEPS_TWO), few=FEW, source=src,
                adam=hashlib.sha256(inspect.getsource(S.adam_update).encode()).hexdigest()[:16])


@functools.lru_cache(maxsize=1)
def run_one():
    return {k: np.asarray(v) for k, v in cached_npz("study-lr-one", _settings(1),
                                                     _build_one).items()}


@functools.lru_cache(maxsize=1)
def run_two():
    return {k: np.asarray(v) for k, v in cached_npz("study-lr-two", _settings(2),
                                                     _build_two).items()}


# ====================================================================== measured
def roc(s0, s1, alphas):
    """The measured ROC of scores ``s0`` (H0) and ``s1`` (H1): the share of H1 above the cut
    that passes each false-alarm rate of H0."""
    s0 = np.sort(np.asarray(s0, float))
    out = []
    for a in alphas:
        if a <= 0:
            out.append(0.0)
            continue
        cut = np.quantile(s0, 1 - a)
        out.append(float((np.asarray(s1) > cut).mean()))
    return np.array(out)


def power_at(s0, s1, alpha=ALPHA):
    """As the script: the cut that 5% of H0 pass (``quantile(0.95)``), the share of H1 kept,
    and the share of H0 that really passes."""
    cut = float(np.quantile(s0, 1 - alpha))
    return cut, float((s1 > cut).mean()), float((s0 > cut).mean())


def floor_on(x0, x1):
    """The binary cross-entropy of the exact ratio's own output, s = p1 / (p0 + p1), on the
    events ``x0`` (H0) and ``x1`` (H1): what the best classifier scores on these events."""
    X, Y = tensors(x0, x1)
    return bce(log_ratio(X[:, 0].astype(np.float64)), Y[:, 0].astype(np.float64))


def net_exact(z_fine, alphas):
    """The network's own ROC, computed exactly: for each false-alarm rate, the cut on its
    output that keeps exactly that share of H0, and the share of H1 it keeps, integrating both
    densities over the values of x whose output passes (on :data:`FINE`)."""
    dx = FINE[1] - FINE[0]
    m0 = np.exp(log_density(FINE, *H0)) * dx
    m1 = np.exp(log_density(FINE, *H1)) * dx
    order = np.argsort(-z_fine, kind="stable")              # the highest outputs first
    a, p = np.cumsum(m0[order]), np.cumsum(m1[order])
    return np.interp(alphas, np.r_[0.0, a], np.r_[0.0, p])


def window(z_fine, alpha=ALPHA):
    """The values of x the network keeps at false-alarm rate ``alpha``: ``(lo, hi)``."""
    dx = FINE[1] - FINE[0]
    m0 = np.exp(log_density(FINE, *H0)) * dx
    order = np.argsort(-z_fine, kind="stable")
    k = int(np.searchsorted(np.cumsum(m0[order]), alpha))
    kept = FINE[order[:k + 1]]
    return float(kept.min()), float(kept.max())


def one():
    """Part one's measurements."""
    r = run_one()
    x0, x1, _f0, _f1 = hypotheses()
    z = r["grid_logit"]
    exact = log_ratio(GRID)
    inside = np.abs(GRID - np.mean(TRACK)) <= (TRACK[1] - TRACK[0]) / 2 + 1e-9
    off = z - exact                                   # ln(learned / exact)
    far = int(np.argmin(np.abs(GRID - FAR)))
    alphas = np.linspace(0, 1, 201)
    _a, p_np, p_cut = exact_rocs(len(alphas))
    lo, hi, np_power = np_at()
    c, cut_power = cut_at()
    cut, net_power, net_alpha = power_at(r["fresh0"], r["fresh1"])
    own = net_exact(r["fine_logit"], alphas)
    i5 = int(np.argmin(np.abs(alphas - ALPHA)))
    h0 = np.histogram(x0[:, 0], SHOW)[0]
    h1 = np.histogram(x1[:, 0], SHOW)[0]
    return dict(
        grid=GRID, learned=z, exact=exact, off=off,
        worst_inside=float(np.abs(off[inside]).max()),
        worst_inside_x=float(GRID[inside][np.argmax(np.abs(off[inside]))]),
        track=TRACK, edge=EDGE,
        far_x=float(GRID[far]), far_learned=float(np.exp(z[far])),
        far_exact=float(np.exp(exact[far])),
        above0=int((x0 > EDGE[1]).sum()), above1=int((x1 > EDGE[1]).sum()),
        below0=int((x0 < EDGE[0]).sum()), below1=int((x1 < EDGE[0]).sum()),
        beyond_far0=int((x0 > FAR).sum()), beyond_far1=int((x1 > FAR).sum()),
        peak=peak(), curve=r["curve"], curve_steps=np.arange(len(r["curve"])) * LOSS_EVERY,
        floor=floor(), floor_on=floor_on(x0, x1),
        first_loss=float(r["curve"][0]), last_loss=float(r["curve"][-1]),
        alphas=alphas, roc_np=p_np, roc_cut=p_cut, roc_net=roc(r["fresh0"], r["fresh1"], alphas),
        roc_own=own, own_power=float(own[i5]), own_window=window(r["fine_logit"]),
        np_window=(lo, hi), np_power=np_power, cut=c, cut_power=cut_power,
        net_power=net_power, net_alpha=net_alpha, net_cut=cut,
        net_power_err=float(np.sqrt(net_power * (1 - net_power) / N_ONE)),
        show=SHOW, h0=h0, h1=h1)


def histograms(v_seen, v_new, w):
    """Data, simulation and weighted simulation in :data:`BINS`, with the weighted
    simulation's error sqrt(sum w^2) and the data's sqrt(n)."""
    d = np.histogram(v_seen, BINS)[0].astype(float)
    a = np.histogram(v_new, BINS)[0].astype(float)
    b = np.histogram(v_new, BINS, weights=w)[0]
    b2 = np.histogram(v_new, BINS, weights=w ** 2)[0]
    return d, a, b, np.sqrt(b2)


def chi2(d, b, b_err, low=0):
    """Sum over bins with data above ``low`` of (data - b)^2 / (data + b_err^2), and the
    number of those bins."""
    keep = d > low
    return float(np.sum((d[keep] - b[keep]) ** 2 / (d[keep] + b_err[keep] ** 2))), int(keep.sum())


VARIABLES = ("x", "y", "x + y")


def columns(e):
    return (e[:, 0], e[:, 1], e[:, 0] + e[:, 1])


def two():
    """Part two's measurements."""
    r = run_two()
    _sim, _data, new, seen = samples()
    z = r["new_logit"]
    w = np.exp(z)                                     # s / (1 - s) = e^z
    lw = log_weight(new)
    we = np.exp(lw)
    few = np.exp(r["few_logit"])
    out = dict(curve=r["curve"], few_curve=r["few_curve"], weights=w, exact_weights=we)
    hists = {}
    for name, vs, vn in zip(VARIABLES, columns(seen), columns(new), strict=True):
        d, a, b, be = histograms(vs, vn, w)
        _, _, e, ee = histograms(vs, vn, we)
        _, _, f, fe = histograms(vs, vn, few)
        hists[name] = dict(data=d, sim=a, weighted=b, weighted_err=be, exact=e, exact_err=ee,
                           few=f, few_err=fe,
                           chi2_before=chi2(d, a, np.sqrt(a)), chi2_after=chi2(d, b, be),
                           chi2_exact=chi2(d, e, ee), chi2_few=chi2(d, f, fe),
                           empty=int(d[(a == 0)].sum()))
    out["hists"] = hists
    bulk = (np.abs(new) < 2).all(1)
    out.update(
        n_eff=float(w.sum() ** 2 / (w ** 2).sum()), n_eff_exact=float(we.sum() ** 2
                                                                     / (we ** 2).sum()),
        n_eff_few=float(few.sum() ** 2 / (few ** 2).sum()),
        w_max=float(w.max()), w_max_exact=float(we.max()),
        w_max_exact_learned=float(w[np.argmax(we)]), w_max_at=new[np.argmax(we)].tolist(),
        w_mean=float(w.mean()), w_mean_exact=float(we.mean()),
        rms_bulk=float(np.sqrt(np.mean((z - lw)[bulk] ** 2))),
        rms_bulk_few=float(np.sqrt(np.mean((r["few_logit"] - lw)[bulk] ** 2))),
        bulk_share=float(bulk.mean()),
        top1=float(np.sort(w)[-len(w) // 100:].sum() / w.sum()),
        top1_exact=float(np.sort(we)[-len(we) // 100:].sum() / we.sum()))
    return out


def summary():
    o, t = one(), two()
    return dict(
        one=dict(np_power=o["np_power"], cut_power=o["cut_power"], net_power=o["net_power"],
                 net_alpha=o["net_alpha"], floor=o["floor"], last_loss=o["last_loss"],
                 worst_inside=o["worst_inside"], far=(o["far_x"], o["far_learned"],
                                                      o["far_exact"]),
                 beyond=(o["beyond0"], o["beyond1"])),
        two=dict(n_eff=t["n_eff"], n_eff_exact=t["n_eff_exact"], w_max=t["w_max"],
                 w_max_exact=t["w_max_exact"], rms_bulk=t["rms_bulk"],
                 rms_bulk_few=t["rms_bulk_few"],
                 chi2={k: (h["chi2_before"], h["chi2_after"], h["chi2_exact"], h["chi2_few"],
                           h["empty"]) for k, h in t["hists"].items()}))

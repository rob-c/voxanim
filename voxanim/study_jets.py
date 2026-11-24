"""Jets as sets of particles, for the animation "Jets as sets of particles" of "Blocks and
code" (catalogue 279): ``examples/jets_as_sets.py`` in numpy.

* **Made-up jets** (pseudo-data; :func:`make_jets`, the script's recipe and seeds): 6 to 20
  particles, each with its momentum fraction z (Dirichlet(0.7), a jet's z add up to 1) and its
  offsets in eta and phi from the jet's axis. Half are one-prong (a toy QCD jet: one core,
  spread 0.1 to 0.35), half two-prong (a toy W: two cores 0.25 to 1.0 apart sharing the
  momentum f : 1 - f, f from 0.25 to 0.75, each spread 0.1). Each jet is centred, turned so
  its spread runs along eta (hardest particle on the + side), sorted by z, padded with zeros to
  20 rows, with a mask of its real rows. 40,000 for training (seed 1), 4,000 held out (seed 2).
* **The MLP** ``Flatten, 60 -> 64 -> 64 -> 1`` (ReLU) reads the 60 numbers in a row; **Deep
  Sets** applies ``phi`` (``3 -> 32 -> 32``, ReLU) to each row, multiplies by the mask, adds
  over the rows and reads the sum with ``rho`` (``32 -> 32 -> 1``). Both from PyTorch's
  default initialisation (uniform +-1/sqrt(fan-in), drawn here with seed :data:`SEED`), trained
  with Adam (lr 0.001) for 6,000 steps of 64 jets, the same batches for both (seed 5), on
  ``binary_cross_entropy_with_logits``.
* **Measured** on the 4,000 held-out jets: accuracy as sorted; then each jet's real rows
  shuffled (seed 3, the script's :func:`shuffled`) and both asked again: accuracy, how far each
  answer (the probability of "two prongs") moved, how many decisions changed. Sorting the
  shuffled rows again gives back the list exactly (:func:`sorted_again`).

A sum does not depend on the order of what it adds, so Deep Sets' answer cannot depend on the
order of the particles: exact in arithmetic. In float32 the additions round in a different
order, so the answer moves in its last bits (a few times 1e-7 here) and no decision changes.

One run each (one seed); cached. ``tests/test_jets_equivalence.py`` runs the script's lines in
torch from this module's starting weights on its first batch and matches the loss, the
gradients and one Adam step of both models.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import sys

import numpy as np

from . import study_course as S
from .cache import cached_npz

VERSION = 1
MAX, FEATURES = 20, 3
N_TRAIN, N_TEST = 40_000, 4_000
STEPS, BATCH, LR = 6000, 64, 1e-3
TRAIN_SEED, TEST_SEED, SHUFFLE_SEED, BATCH_SEED = 1, 2, 3, 5
SEED = 279                       # the starting weights
HIDDEN, WIDTH = 64, 32           # the MLP's hidden layers; phi's and rho's
MLP_KEYS = ("1.weight", "1.bias", "3.weight", "3.bias", "5.weight", "5.bias")
PHI_KEYS = RHO_KEYS = ("0.weight", "0.bias", "2.weight", "2.bias")
MODELS = ("mlp", "ds")


# ====================================================================== the made-up jets
def one_core(rng, n):
    return rng.normal(0.0, rng.uniform(0.1, 0.35), (n, 2))


def two_cores(rng, z):
    n = len(z)
    f, d = rng.uniform(0.25, 0.75), rng.uniform(0.25, 1.0)
    a = rng.uniform(0.0, 2 * np.pi)
    way = np.array([np.cos(a), np.sin(a)])
    first = rng.random(n) < 0.5
    first[0], first[1] = True, False
    z[first] *= f / z[first].sum()
    z[~first] *= (1 - f) / z[~first].sum()
    centre = np.where(first[:, None], (1 - f) * d * way, -f * d * way)
    return centre + rng.normal(0.0, 0.1, (n, 2))


def make_jet(rng, two_prong):
    n = int(rng.integers(6, MAX + 1))
    z = rng.dirichlet(np.full(n, 0.7))
    xy = two_cores(rng, z) if two_prong else one_core(rng, n)
    return np.c_[z, xy]


def turn(p):
    z, xy = p[:, 0], p[:, 1:]
    xy = xy - z @ xy
    spread = (z[:, None] * xy).T @ xy
    v = np.linalg.eigh(spread)[1][:, -1]
    if xy[np.argmax(z)] @ v < 0:
        v = -v
    return np.c_[z, xy @ np.array([v, [-v[1], v[0]]]).T]


def as_list(p):
    """The script's ``as_list``: rows sorted by z (hardest first), padded to 20, and the mask."""
    p = p[np.argsort(-p[:, 0])]
    x = np.zeros((MAX, 3), np.float32)
    x[:len(p)] = p
    mask = np.arange(MAX) < len(p)
    return x, mask


def make_jets(n, seed):
    """``(X [n, 20, 3], M [n, 20], y [n])``, y = 1 for a two-prong jet: the script's."""
    rng = np.random.default_rng(seed)
    X = np.zeros((n, MAX, 3), np.float32)
    M = np.zeros((n, MAX), bool)
    y = np.zeros(n, np.float32)
    for i in range(n):
        two = rng.random() < 0.5
        X[i], M[i] = as_list(turn(make_jet(rng, two)))
        y[i] = two
    return X, M, y


def shuffled(X, M, rng):
    """The script's ``shuffled``: each jet's real rows in a new order, the pads left last."""
    Xs = X.copy()
    for x, m in zip(Xs, M, strict=True):
        n = m.sum()
        x[:n] = x[rng.permutation(n)]
    return Xs


@functools.lru_cache(maxsize=1)
def train_set():
    return make_jets(N_TRAIN, TRAIN_SEED)


@functools.lru_cache(maxsize=1)
def test_set():
    """``(X, M, y, Xs)``: the held-out jets and the same jets shuffled."""
    X, M, y = make_jets(N_TEST, TEST_SEED)
    return X, M, y, shuffled(X, M, np.random.default_rng(SHUFFLE_SEED))


def sorted_again(Xs, M):
    """Each shuffled jet's real rows put through ``as_list`` again."""
    return np.stack([as_list(x[m])[0] for x, m in zip(Xs, M, strict=True)])


def batch_indices(step):
    """The rows of training batch ``step`` (``pick.integers`` with the script's seed, in order)."""
    return _batches()[step]


@functools.lru_cache(maxsize=1)
def _batches():
    pick = np.random.default_rng(BATCH_SEED)
    return np.stack([pick.integers(0, N_TRAIN, BATCH) for _ in range(STEPS)])


# ====================================================================== the two models
def _linear(rng, o, i):
    b = 1 / np.sqrt(i)
    return (rng.uniform(-b, b, (o, i)).astype(np.float32),
            rng.uniform(-b, b, o).astype(np.float32))


def init(seed=0):
    """``dict(mlp=..., phi=..., rho=...)``, each keyed as its ``state_dict()``."""
    rng = np.random.default_rng([SEED, int(seed)])
    mlp, phi, rho = {}, {}, {}
    mlp["1.weight"], mlp["1.bias"] = _linear(rng, HIDDEN, MAX * FEATURES)
    mlp["3.weight"], mlp["3.bias"] = _linear(rng, HIDDEN, HIDDEN)
    mlp["5.weight"], mlp["5.bias"] = _linear(rng, 1, HIDDEN)
    phi["0.weight"], phi["0.bias"] = _linear(rng, WIDTH, FEATURES)
    phi["2.weight"], phi["2.bias"] = _linear(rng, WIDTH, WIDTH)
    rho["0.weight"], rho["0.bias"] = _linear(rng, WIDTH, WIDTH)
    rho["2.weight"], rho["2.bias"] = _linear(rng, 1, WIDTH)
    return dict(mlp=mlp, phi=phi, rho=rho)


def _two(P, x, a="0", b="2"):
    """``Linear, ReLU, Linear`` with keys ``a`` and ``b``: ``(out, hidden, pre)``."""
    pre = x @ P[a + ".weight"].T + P[a + ".bias"]
    h = np.maximum(pre, 0)
    return h @ P[b + ".weight"].T + P[b + ".bias"], h, pre


def _two_back(P, x, h, pre, dout, a="0", b="2"):
    xs, hs, ps = x.reshape(-1, x.shape[-1]), h.reshape(-1, h.shape[-1]), pre.reshape(-1,
                                                                                    h.shape[-1])
    ds = dout.reshape(-1, dout.shape[-1])
    G = {b + ".weight": ds.T @ hs, b + ".bias": ds.sum(0)}
    dh = (ds @ P[b + ".weight"]) * (ps > 0)
    G[a + ".weight"], G[a + ".bias"] = dh.T @ xs, dh.sum(0)
    return G, (dh @ P[a + ".weight"]).reshape(x.shape)


def mlp_logits(P, X):
    """The MLP's logit per jet: ``[B]``."""
    x = X.reshape(len(X), -1)
    h1 = np.maximum(x @ P["1.weight"].T + P["1.bias"], 0)
    out, _h2, _p2 = _two(P, h1, "3", "5")
    return out[:, 0]


def ds_logits(phi, rho, X, M):
    """Deep Sets' logit per jet: ``rho(sum over the real rows of phi(row))``: ``[B]``."""
    h = np.maximum(_two(phi, X)[0], 0) * M[..., None]
    return _two(rho, h.sum(1))[0][:, 0]


def _bce(logit, y):
    return float((np.maximum(logit, 0) - logit * y + np.log1p(np.exp(-np.abs(logit)))).mean())


def sig(z):
    return 1 / (1 + np.exp(-np.asarray(z, np.float64)))


def mlp_grads(P, X, y):
    """``bce(mlp(X), y)`` (mean) and the MLP's gradients."""
    x = X.reshape(len(X), -1)
    p1 = x @ P["1.weight"].T + P["1.bias"]
    h1 = np.maximum(p1, 0)
    out, h2, p2 = _two(P, h1, "3", "5")
    logit = out[:, 0]
    d = ((sig(logit) - y) / len(y)).astype(np.float32)[:, None]
    G, dh1 = _two_back(P, h1, h2, p2, d, "3", "5")
    dp1 = dh1 * (p1 > 0)
    G["1.weight"], G["1.bias"] = dp1.T @ x, dp1.sum(0)
    return _bce(logit, y), {k: G[k].astype(np.float32) for k in MLP_KEYS}


def ds_grads(phi, rho, X, M, y):
    """``bce(deep_sets(X, M), y)`` (mean) and the gradients of ``phi`` and ``rho``."""
    o1, h1, p1 = _two(phi, X)
    r = np.maximum(o1, 0)
    s = (r * M[..., None]).sum(1)
    out, h2, p2 = _two(rho, s)
    logit = out[:, 0]
    d = ((sig(logit) - y) / len(y)).astype(np.float32)[:, None]
    Gr, ds = _two_back(rho, s, h2, p2, d)
    dr = ds[:, None, :] * M[..., None] * (o1 > 0)
    Gp, _ = _two_back(phi, X, h1, p1, dr.astype(np.float32))
    return (_bce(logit, y), {k: Gp[k].astype(np.float32) for k in PHI_KEYS},
            {k: Gr[k].astype(np.float32) for k in RHO_KEYS})


def count(P):
    return int(sum(v.size for v in P.values()))


# ====================================================================== the run
def _train_mlp(out):
    X, _M, y = train_set()
    P = init()["mlp"]
    state = S.adam_start(P)
    losses = []
    for step in range(STEPS):
        i = batch_indices(step)
        loss, G = mlp_grads(P, X[i], y[i])
        P, state = S.adam_update(P, G, state, lr=LR)
        if step == 0:
            out.update({"mlpG0__" + k: v for k, v in G.items()})
            out.update({"mlp1__" + k: v.copy() for k, v in P.items()})
        losses.append(loss)
    out["loss_mlp"] = np.array(losses)
    out.update({"mlp__" + k: v for k, v in P.items()})


def _train_ds(out):
    X, M, y = train_set()
    start = init()
    phi, rho = start["phi"], start["rho"]
    sp, sr = S.adam_start(phi), S.adam_start(rho)
    losses = []
    for step in range(STEPS):
        i = batch_indices(step)
        loss, Gp, Gr = ds_grads(phi, rho, X[i], M[i], y[i])
        phi, sp = S.adam_update(phi, Gp, sp, lr=LR)
        rho, sr = S.adam_update(rho, Gr, sr, lr=LR)
        if step == 0:
            out.update({"phiG0__" + k: v for k, v in Gp.items()})
            out.update({"rhoG0__" + k: v for k, v in Gr.items()})
            out.update({"phi1__" + k: v.copy() for k, v in phi.items()})
            out.update({"rho1__" + k: v.copy() for k, v in rho.items()})
        losses.append(loss)
    out["loss_ds"] = np.array(losses)
    out.update({"phi__" + k: v for k, v in phi.items()})
    out.update({"rho__" + k: v for k, v in rho.items()})


def _build():
    out = {}
    _train_mlp(out)
    _train_ds(out)
    w = {p: {k[len(p) + 2:]: v for k, v in out.items() if k.startswith(p + "__")}
         for p in ("mlp", "phi", "rho")}
    X, M, _y, Xs = test_set()
    again = sorted_again(Xs, M)
    out["sorted_again_same"] = np.int64(sum(np.array_equal(a, b) for a, b in zip(again, X,
                                                                                strict=True)))
    out["logit_mlp"] = mlp_logits(w["mlp"], X)
    out["logit_mlp_shuffled"] = mlp_logits(w["mlp"], Xs)
    out["logit_mlp_again"] = mlp_logits(w["mlp"], again)
    out["logit_ds"] = ds_logits(w["phi"], w["rho"], X, M)
    out["logit_ds_shuffled"] = ds_logits(w["phi"], w["rho"], Xs, M)
    Xr, Mr, yr = (a[:N_TEST] for a in train_set())
    out["train_right_mlp"] = np.float64(((mlp_logits(w["mlp"], Xr) > 0) == yr).mean())
    out["train_right_ds"] = np.float64(((ds_logits(w["phi"], w["rho"], Xr, Mr) > 0) == yr)
                                       .mean())
    return out


def _settings():
    return dict(version=VERSION, n=(N_TRAIN, N_TEST), steps=STEPS, batch=BATCH, lr=LR,
                seeds=(TRAIN_SEED, TEST_SEED, SHUFFLE_SEED, BATCH_SEED, SEED),
                sizes=(HIDDEN, WIDTH),
                source=hashlib.sha256(inspect.getsource(sys.modules[__name__]).encode())
                .hexdigest()[:16])


@functools.lru_cache(maxsize=1)
def run():
    return {k: np.asarray(v) for k, v in cached_npz("study-jets", _settings(), _build).items()}


def weights(which):
    """By prefix: ``mlp``, ``phi``, ``rho`` (trained); ``mlp1``, ``phi1``, ``rho1`` (after the
    first step); ``mlpG0``, ``phiG0``, ``rhoG0`` (the first step's gradients)."""
    p = which + "__"
    return {k[len(p):]: v for k, v in run().items() if k.startswith(p)}


def probs():
    """The probability of "two prongs" per held-out jet: ``dict(mlp, mlp_shuffled,
    mlp_again, ds, ds_shuffled)`` (float64, from the float32 logits)."""
    r = run()
    return dict(mlp=sig(r["logit_mlp"]), mlp_shuffled=sig(r["logit_mlp_shuffled"]),
                mlp_again=sig(r["logit_mlp_again"]), ds=sig(r["logit_ds"]),
                ds_shuffled=sig(r["logit_ds_shuffled"]))


def results():
    """The measured claims, on the 4,000 held-out jets."""
    r = run()
    y = test_set()[2]
    p = probs()
    right = {k: float(((v > 0.5) == y).mean()) for k, v in p.items()}
    out = dict(right=right, train_right=dict(mlp=float(r["train_right_mlp"]),
                                             ds=float(r["train_right_ds"])),
               sorted_again_same=int(r["sorted_again_same"]), n_test=N_TEST)
    for m in MODELS:
        moved = np.abs(p[m] - p[m + "_shuffled"])
        out[m] = dict(mean_moved=float(moved.mean()), median_moved=float(np.median(moved)),
                      max_moved=float(moved.max()),
                      over_10=float((moved > 0.1).mean()),
                      changed=int(((p[m] > 0.5) != (p[m + "_shuffled"] > 0.5)).sum()),
                      bitwise_same=int((r["logit_" + m] == r["logit_%s_shuffled" % m]).sum()))
    out["mlp_again_same"] = int((r["logit_mlp"] == r["logit_mlp_again"]).sum())
    out["count"] = dict(mlp=count(init()["mlp"]), ds=count(init()["phi"]) + count(init()["rho"]))
    out["loss"] = {m: (float(r["loss_" + m][0]), float(r["loss_" + m][-500:].mean()))
                   for m in MODELS}
    return out

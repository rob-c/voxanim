"""A vector database, for the vector-database animations of "Blocks and code" (catalogue
234-237): ``examples/vector_db.py`` in numpy, trained and searched once, with every number
the animations show.

* **Training** (:func:`train`): ``nn.Sequential(nn.Flatten(), nn.Linear(784, 128),
  nn.ReLU(), nn.Linear(128, 32))``, initialised as PyTorch does, trained for 938 steps of
  64 random training digits with Adam (lr 0.001). Each digit is paired with another drawing
  of the same digit; both are encoded and scaled to length 1, and InfoNCE (cross-entropy
  over the [64, 64] grid of dot products divided by 0.1, the right pair on the diagonal)
  pulls pairs together and pushes the other 63 apart. Labels pick the pairs; nothing else
  uses them. Two runs: ``seed`` 0 (the store's encoder) and 1 (a retrained one).
* **The store** (:func:`store`): the 60,000 training digits' vectors, ``[60000, 32]``, with
  ids and labels beside them. **Search** (:func:`search`): 60,000 dot products, the top k.
* **Quality** (:func:`quality`): for the 10,000 test digits, how often the nearest stored
  vector, and the 10 nearest, carry the query's label: for the trained encoder, an untrained
  one, raw pixels, and the retrained encoder's queries against the first encoder's store.
* **IVF** (:func:`ivf`): spherical k-means into 64 lists (10 steps, seeded); a query visits
  the ``nprobe`` lists whose centres are nearest; recall@10 against exact search, and the
  share of the store scanned, for nprobe 1, 2, 4, 8, 16, 32 and 64.

The run is cached by :func:`voxanim.cache.cached_npz`. ``tests/test_vectordb_equivalence.py``
runs the script's own lines in torch from :func:`init_params` on :func:`batch` ``(0)`` and
matches the loss, the gradients and Adam's step, and the search and IVF lines.
The one thing not the same as the script: its random batches, pairs and k-means start come
from torch's generator, the twin's from numpy's (seeded).
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
SIZES = (784, 128, 32)
KEYS = ("1.weight", "1.bias", "3.weight", "3.bias")
BATCH = 64
STEPS = 938
TEMPERATURE = 0.1
LR = 1e-3
SEEDS = (0, 1)                       # the store's encoder, and a retrained one
LISTS = 64
KMEANS_STEPS = 10
NPROBES = (1, 2, 4, 8, 16, 32, 64)
K = 10
SNAPS = (0, 100, 937)                # the grid of similarities kept at these steps
SCRIPT = "examples/vector_db.py"
SEED = 234


# ====================================================================== the encoder
def init_params(seed=0):
    """PyTorch's ``nn.Linear`` initialisation (uniform ``±1/sqrt(fan_in)``), float32."""
    rng = np.random.default_rng([SEED, 0, int(seed)])
    P = {}
    for i, (a, b) in zip((1, 3), ((784, 128), (128, 32)), strict=True):
        bound = 1.0 / np.sqrt(a)
        P["%d.weight" % i] = rng.uniform(-bound, bound, (b, a)).astype(np.float32)
        P["%d.bias" % i] = rng.uniform(-bound, bound, b).astype(np.float32)
    return P


def count(P=None):
    P = init_params() if P is None else P
    return int(sum(v.size for v in P.values()))


def encode(P, x, chunk=10_000):
    """``encoder(x)``: ``[N, 32]`` (not yet scaled to length 1)."""
    x = np.asarray(x, np.float32).reshape(len(x), -1)
    out = []
    for s in range(0, len(x), chunk):
        h = np.maximum(x[s:s + chunk] @ P["1.weight"].T + P["1.bias"], 0)
        out.append(h @ P["3.weight"].T + P["3.bias"])
    return np.concatenate(out)


def unit(v):
    """``F.normalize(v, dim=1)``: each row scaled to length 1."""
    v = np.asarray(v)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), np.float32(1e-12))


def batch(step, seed=0):
    """Step ``step``'s 64 random training digits and, for each, another drawing of the same
    digit: ``dict(i, x, y, j, p)`` (``j``: the partners' indices)."""
    x, y = S.data()[:2]
    rng = np.random.default_rng([SEED, 1, int(seed), int(step)])
    i = rng.integers(0, len(x), BATCH)
    where = _where()
    j = np.array([where[d][rng.integers(0, len(where[d]))] for d in y[i]])
    return dict(i=i, x=x[i], y=y[i], j=j, p=x[j])


@functools.lru_cache(maxsize=1)
def _where():
    y = S.data()[1]
    return [np.nonzero(y == d)[0] for d in range(10)]


def loss_and_grads(P, x, p, keep=False):
    """InfoNCE as region "vdb-loss" computes it, and every parameter's gradient; ``keep``:
    also the [64, 64] grid of dot products (before dividing by the temperature)."""
    x = np.asarray(x, np.float32).reshape(len(x), -1)
    p = np.asarray(p, np.float32).reshape(len(p), -1)
    n = len(x)
    hs, es = [], []
    for inp in (x, p):
        z = inp @ P["1.weight"].T + P["1.bias"]
        h = np.maximum(z, 0)
        hs.append((inp, z, h))
        es.append(h @ P["3.weight"].T + P["3.bias"])
    a, b = unit(es[0]), unit(es[1])
    grid = a @ b.T
    scores = grid / np.float32(TEMPERATURE)
    loss = float(S.loss_each(scores, np.arange(n)).mean())
    d = S._softmax(scores).astype(np.float32)
    d[np.arange(n), np.arange(n)] -= 1.0
    d /= np.float32(n)
    d /= np.float32(TEMPERATURE)
    da, db = d @ b, d.T @ a
    G = {k: np.zeros_like(v) for k, v in P.items()}
    for (inp, z, h), e, u, du in zip(hs, es, (a, b), (da, db), strict=True):
        norm = np.maximum(np.linalg.norm(e, axis=1, keepdims=True), np.float32(1e-12))
        de = (du - u * (u * du).sum(1, keepdims=True)) / norm
        G["3.weight"] += de.T @ h
        G["3.bias"] += de.sum(0)
        dz = (de @ P["3.weight"]) * (z > 0)
        G["1.weight"] += dz.T @ inp
        G["1.bias"] += dz.sum(0)
    G = {k: v.astype(np.float32) for k, v in G.items()}
    return (loss, G, grid) if keep else (loss, G)


def _train(seed):
    P = init_params(seed)
    state = S.adam_start(P)
    out = dict(loss=[], pos=[], neg=[])
    for step in range(STEPS):
        b = batch(step, seed)
        loss, G, grid = loss_and_grads(P, b["x"], b["p"], keep=True)
        if step == 0:
            out.update({"G0__" + k: G[k] for k in P})
        if step in SNAPS:
            out["grid%d" % step] = grid
        diag = np.eye(len(grid), dtype=bool)
        out["loss"].append(loss)
        out["pos"].append(float(grid[diag].mean()))
        out["neg"].append(float(grid[~diag].mean()))
        P, state = S.adam_update(P, G, state, lr=LR)
        if step == 0:
            out.update({"W1__" + k: P[k].copy() for k in P})
    out.update({"W__" + k: P[k] for k in P})
    return out


def _agreement(store_vecs, query_vecs, y, yt, k=K, chunk=2000):
    """Top-1 and top-k label agreement of exact search, and the top-k ids of every query."""
    first, agree, tops = 0, 0.0, []
    for s in range(0, len(query_vecs), chunk):
        sc = query_vecs[s:s + chunk] @ store_vecs.T
        top = np.argpartition(-sc, k, axis=1)[:, :k]
        order = np.take_along_axis(sc, top, 1).argsort(1)[:, ::-1]
        top = np.take_along_axis(top, order, 1)
        tops.append(top)
        first += int((y[top[:, 0]] == yt[s:s + chunk]).sum())
        agree += float((y[top] == yt[s:s + chunk, None]).sum())
    return first / len(yt), agree / (len(yt) * k), np.concatenate(tops)


def kmeans(v, k=LISTS, steps=KMEANS_STEPS, seed=SEED):
    """``kmeans`` of the script: spherical k-means, the start ``k`` random stored vectors."""
    rng = np.random.default_rng([SEED, 2, int(seed)])
    centres = v[rng.permutation(len(v))[:k]].copy()
    for _ in range(steps):
        home = (v @ centres.T).argmax(1)
        for j in range(k):
            if (home == j).any():
                centres[j] = unit(v[home == j].mean(0, keepdims=True))[0]
    return centres


def _build():
    out = {}
    runs = {seed: _train(seed) for seed in SEEDS}
    for seed, r in runs.items():
        out.update({"s%d__%s" % (seed, k): np.asarray(v) for k, v in r.items()})
    x, y, xt, yt = S.data()
    P0 = {k: runs[0]["W__" + k] for k in KEYS}
    P1 = {k: runs[1]["W__" + k] for k in KEYS}
    v0, q0 = unit(encode(P0, x)), unit(encode(P0, xt))
    v1, q1 = unit(encode(P1, x)), unit(encode(P1, xt))
    vu, qu = unit(encode(init_params(0), x)), unit(encode(init_params(0), xt))
    px, pq = unit(x), unit(xt)
    q = {}
    for name, (sv, qv) in dict(trained=(v0, q0), untrained=(vu, qu), pixels=(px, pq),
                               mixed=(v0, q1), reembedded=(v1, q1)).items():
        first, agree, tops = _agreement(sv, qv, y, yt)
        q[name] = (first, agree)
        if name in ("trained", "mixed"):
            out["top__" + name] = tops
    out["quality"] = np.array([q[n] for n in ("trained", "untrained", "pixels", "mixed",
                                              "reembedded")])
    out["cross_cos"] = np.float64((v0 * v1).sum(1).mean())
    centres = kmeans(v0)
    home = (v0 @ centres.T).argmax(1)
    exact = out["top__trained"]
    recall, scanned = [], []
    order = np.argsort(-(q0 @ centres.T), axis=1)
    sizes = np.bincount(home, minlength=LISTS)
    members = [np.nonzero(home == j)[0] for j in range(LISTS)]
    for nprobe in NPROBES:
        near = order[:, :nprobe]
        r, sc = 0.0, 0
        for t in range(len(q0)):
            cand = np.sort(np.concatenate([members[j] for j in near[t]]))
            sc += len(cand)
            s = v0[cand] @ q0[t]
            got = cand[np.argsort(-s)[:K]]
            r += len(np.intersect1d(got, exact[t])) / K
        recall.append(r / len(q0))
        scanned.append(sc / (len(q0) * len(v0)))
    out.update(centres=centres, home=home, ivf_recall=np.array(recall),
               ivf_scanned=np.array(scanned), list_sizes=sizes)
    return out


def _settings():
    return dict(version=VERSION, course=S.VERSION, sizes=SIZES, batch=BATCH, steps=STEPS,
                tau=TEMPERATURE, lr=LR, seeds=SEEDS, lists=LISTS, kmeans=KMEANS_STEPS,
                nprobes=NPROBES, k=K, snaps=SNAPS,
                source=hashlib.sha256(inspect.getsource(sys.modules[__name__]).encode())
                .hexdigest()[:16])


@functools.lru_cache(maxsize=1)
def run():
    """Everything recorded (cached)."""
    return {k: np.asarray(v) for k, v in cached_npz("study-vectordb", _settings(),
                                                    _build).items()}


def params(seed=0, which="end"):
    """An encoder's parameters: ``"start"``, ``"after_one_step"`` or ``"end"``."""
    if which == "start":
        return init_params(seed)
    prefix = {"after_one_step": "W1__", "end": "W__"}[which]
    return {k: run()["s%d__%s%s" % (seed, prefix, k)] for k in KEYS}


def first_grads(seed=0):
    return {k: run()["s%d__G0__%s" % (seed, k)] for k in KEYS}


def history(seed=0):
    """Per step: the loss, and the mean dot product of the right pairs (``pos``) and of the
    other 63 (``neg``); ``grids``: the [64, 64] grid at :data:`SNAPS`."""
    r = run()
    p = "s%d__" % seed
    return dict(loss=r[p + "loss"], pos=r[p + "pos"], neg=r[p + "neg"],
                grids={s: r[p + "grid%d" % s] for s in SNAPS})


@functools.lru_cache(maxsize=2)
def store(seed=0):
    """``dict(vectors [60000, 32], ids, labels)``: every training digit's unit vector."""
    x, y = S.data()[:2]
    return dict(vectors=unit(encode(params(seed), x)), ids=np.arange(len(x)), labels=y)


def query(t, seed=0):
    """Test digit ``t``'s unit vector from encoder ``seed``."""
    return unit(encode(params(seed), S.data()[2][t:t + 1]))[0]


def search(t, k=5, seed=0, store_seed=0):
    """Test digit ``t`` searched exactly: ``dict(ids, labels, scores, label)``."""
    st = store(store_seed)
    sc = st["vectors"] @ query(t, seed)
    top = np.argsort(-sc, kind="stable")[:k]
    return dict(ids=top, labels=st["labels"][top], scores=sc[top], label=int(S.data()[3][t]))


def quality():
    """Over the 10,000 test digits: ``{name: dict(top1, top10)}`` for ``trained``,
    ``untrained`` (the same network before training), ``pixels`` (the 784 pixels, scaled to
    length 1), ``mixed`` (the retrained encoder's queries, the first encoder's store) and
    ``reembedded`` (the retrained encoder for both)."""
    q = run()["quality"]
    names = ("trained", "untrained", "pixels", "mixed", "reembedded")
    return {n: dict(top1=float(a), top10=float(b)) for n, (a, b) in zip(names, q, strict=True)}


def ivf():
    """``dict(nprobe, recall, scanned, list_sizes, centres, home)``: recall@10 against exact
    search, and the share of the 60,000 vectors scanned, averaged over the 10,000 queries."""
    r = run()
    return dict(nprobe=NPROBES, recall=r["ivf_recall"], scanned=r["ivf_scanned"],
                list_sizes=r["list_sizes"], centres=r["centres"], home=r["home"])


def summary():
    q = quality()
    iv = ivf()
    h = history(0)
    return dict(parameters=count(), first_loss=float(h["loss"][0]),
                last_100=float(h["loss"][-100:].mean()), pos_end=float(h["pos"][-1]),
                neg_end=float(h["neg"][-1]), quality=q, cross_cos=float(run()["cross_cos"]),
                ivf={int(n): (float(r), float(s)) for n, r, s in
                     zip(iv["nprobe"], iv["recall"], iv["scanned"], strict=True)},
                list_sizes=(int(iv["list_sizes"].min()), int(iv["list_sizes"].max())))

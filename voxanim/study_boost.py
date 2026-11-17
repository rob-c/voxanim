"""The experiments behind animations 165-169: boosting, and the algorithm XGBoost uses
(see :func:`voxanim.next_catalog.redesigned`).

No physics dataset is used. Real handwritten digits (MNIST) stand in for events: the 4s
are the signal, the 9s the background, each read out as 4 x 4 cells (the mean of 7 x 7
pixels), so that the two classes overlap as signal and background do (:func:`data`,
:func:`draw`). Two of the sixteen cells, chosen by a stated rule (:func:`best_pair`),
are the two-feature task on which a partition of the plane can be drawn; all sixteen
are the task the measurements are made on.

Everything is plain numpy and written out here:

* ``adaboost``: discrete AdaBoost with stumps (:func:`stump`, :func:`adaboost`);
* ``boosting-shrinkage-and-early-stopping``: gradient boosting for the log-loss
  (:func:`boost` with ``second=False``: each tree is fitted by least squares to the
  gradient and a leaf's value is its mean);
* ``xgboost-split-gain``, ``xgboost-histograms-and-missing-values``,
  ``xgboost-one-round``: the algorithm XGBoost uses (Chen and Guestrin 2016), written
  out in numpy: the slope g and curvature h of the loss for every event, leaf values
  ``-G / (H + lambda)``, the split gain, histogram split finding over quantile bins with
  the subtraction trick, a learnt default direction for missing values, pruning with
  gamma, shrinkage, row and column subsampling (:func:`histograms`, :func:`scan`,
  :func:`grow`, :func:`prune`, :func:`boost`). The XGBoost library is never imported
  and no number here is the library's.

Seeds enter through ``numpy.random.default_rng`` generators whose seeds are recorded;
training, validation and held-out events are disjoint and every number reported is a
held-out one unless it says otherwise.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import time

import numpy as np

from . import mnist
from .cache import cached_npz
from .study import Panel, Study
from .study_phys_apps import auc, cells

VERSION = 1
STAND_IN = ("No physics dataset is used: real handwritten digits (MNIST) stand in for "
            "events, the 4s for signal and the 9s for background.")
COUNTERPART = ("An analysis separates signal from background events with boosted "
               "decision trees in the same way, from a handful of measured variables.")
OWN = "the algorithm XGBoost uses, written out in numpy"
EVIDENCE = dict(
    measured="measured on MNIST: trees grown here in numpy, numbers from held-out digits",
    exact="exact arithmetic on a stated handful of real events",
    timing="timings of this numpy code on the machine that built the study: they vary "
           "by machine and are never the XGBoost library's",
)


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them (an edit to one experiment re-runs that experiment only)."""
    h = hashlib.sha256()
    for fn in sources:
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-boost-" + name, settings,
                                                    build).items()}


def _parallel(fn, jobs):
    from .study_generative import parallel

    return parallel(fn, jobs)


def _floats(v):
    return [float(x) for x in np.asarray(v).ravel()]


# ====================================================================== the events
SIGNAL, BACKGROUND = 4, 9
SPLIT = 165                       # seed of the split into pool and held-out events
BLOCK = 7                         # cells are 7 x 7 pixels: 4 x 4 = 16 readings an event
CELLS = (28 // BLOCK) ** 2
POOL = 4000                       # events of each class that training may draw from
TRAIN, VALID = 1000, 500          # of each class, per seed
SEEDS = (0, 1, 2, 3, 4)
BINS = 32                         # quantile bins of a cell's reading


def cell_name(j):
    """Cell ``j`` of the 4 x 4 read-out as a row letter (A at the top) and a column."""
    return "ABCD"[int(j) // 4] + str(int(j) % 4 + 1)


@functools.lru_cache(maxsize=1)
def data():
    """The 4s (signal, label 1) and 9s (background, label 0) as 16 cells. ``pool``:
    ``POOL`` of each from the training file, for training and validation, with their
    row in that file (``pool_id``). ``held``: every other 4 and 9 of the training file
    and all of the test file."""
    xtr, ytr, xte, yte = mnist.load()
    rng = np.random.default_rng([SPLIT, 0])
    pool, pool_id, held, n_test = [], [], [], []
    for digit in (SIGNAL, BACKGROUND):
        ids = rng.permutation(np.flatnonzero(ytr == digit))
        pool.append(cells(xtr[ids[:POOL]], BLOCK))
        pool_id.append(ids[:POOL])
        held.append(np.concatenate([cells(xtr[ids[POOL:]], BLOCK),
                                    cells(xte[yte == digit], BLOCK)]))
        n_test.append(int((yte == digit).sum()))
    return dict(pool=pool, pool_id=pool_id, held_x=np.concatenate(held),
                held_y=np.r_[np.ones(len(held[0])), np.zeros(len(held[1]))],
                held_from_test=n_test)


def draw(seed, n=TRAIN, extra=VALID):
    """Seed ``seed``'s training sample, ``n`` events of each class from the pool, and
    ``extra`` more of each (never the same events) for validation. ``id``: the rows of
    the training events in the MNIST training file."""
    d = data()
    rng = np.random.default_rng([SPLIT, 1, seed])
    order = [rng.permutation(POOL) for _ in d["pool"]]
    return dict(
        x=np.concatenate([p[o[:n]] for p, o in zip(d["pool"], order)]),
        y=np.r_[np.ones(n), np.zeros(n)],
        id=np.concatenate([i[o[:n]] for i, o in zip(d["pool_id"], order)]),
        vx=np.concatenate([p[o[n:n + extra]] for p, o in zip(d["pool"], order)]),
        vy=np.r_[np.ones(extra), np.zeros(extra)])


def pictures(ids):
    """The 28 x 28 images of rows ``ids`` of the MNIST training file."""
    return mnist.load()[0][np.asarray(ids, np.int64)].reshape(-1, 28, 28)


# ====================================================================== bins
def quantile_edges(x, bins=BINS):
    """For each feature (column of ``x``) the cut points of at most ``bins`` bins holding
    equal shares of its values that are not missing (NaN): the ``1/bins, 2/bins, ...``
    quantiles, each a value that occurs, repeats dropped. A value ``v`` is in bin ``k``
    when ``edges[k - 1] <= v < edges[k]``."""
    out = []
    for col in np.asarray(x, float).T:
        v = np.sort(col[~np.isnan(col)])
        if len(v) == 0:
            out.append(np.zeros(0))
            continue
        q = v[np.minimum((np.arange(1, bins) * len(v)) // bins, len(v) - 1)]
        out.append(np.unique(q[q > v[0]]))
    return out


def value_edges(x):
    """Cut points that give every distinct value of each feature a bin of its own: with
    these the histogram method tries exactly the splits the exact method tries."""
    out = []
    for col in np.asarray(x, float).T:
        out.append(np.unique(col[~np.isnan(col)])[1:])
    return out


def width(edges):
    """The number of bins the widest feature has."""
    return max(len(e) for e in edges) + 1


def to_bins(x, edges, bins=None):
    """``x`` as bin numbers under ``edges``; a missing value (NaN) is given the number
    ``bins`` (one past the last bin: the slot the histograms keep for missing)."""
    x = np.asarray(x, float)
    bins = bins or width(edges)
    out = np.empty(x.shape, np.int64)
    for j, e in enumerate(edges):
        out[:, j] = np.searchsorted(e, x[:, j], side="right")
    out[np.isnan(x)] = bins
    return out


# ====================================================================== the loss
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def event_loss(score, y):
    """Each event's log-loss at its score (its log-odds): ``ln(1 + e^F) - y F``."""
    score = np.asarray(score, float)
    return np.logaddexp(0.0, score) - y * score


def log_loss(score, y):
    """The mean log-loss of scores against labels 0/1."""
    return float(np.mean(event_loss(score, y)))


def error_rate(score, y):
    """The share of events on the wrong side of probability one half."""
    return float(np.mean((np.asarray(score) > 0) != (np.asarray(y) > 0.5)))


def slopes(score, y):
    """The first and second derivative of each event's log-loss with respect to its
    score: ``g = p - y`` and ``h = p (1 - p)``, with ``p`` the sigmoid of the score."""
    p = sigmoid(score)
    return p - y, np.maximum(p * (1.0 - p), 1e-16)


def roc_area(score, y):
    y = np.asarray(y) > 0.5
    return auc(np.asarray(score)[y], np.asarray(score)[~y])


# ====================================================================== one node
def leaf_value(G, H, lam=0.0):
    """The value of a leaf that minimises ``G w + (H + lambda) w^2 / 2``."""
    return -G / (H + lam)


def leaf_score(G, H, lam=0.0):
    """``G^2 / (H + lambda)``: twice the loss a leaf with these sums can remove."""
    return G * G / (H + lam)


def split_gain(GL, HL, GR, HR, lam=0.0, gamma=0.0):
    """XGBoost's gain of a split: half of what the two children can remove beyond what
    the parent can, less ``gamma`` for the extra leaf."""
    return 0.5 * (leaf_score(GL, HL, lam) + leaf_score(GR, HR, lam)
                  - leaf_score(GL + GR, HL + HR, lam)) - gamma


def histograms(xb, g, h, rows, bins):
    """The node's histograms: for each feature, the sums of ``g`` and of ``h`` and the
    number of the events ``rows`` in each bin, slot ``bins`` holding the events whose
    value is missing. Shape ``(3, features, bins + 1)``. One pass over the events: no
    sorting."""
    sub = xb[rows]
    f = sub.shape[1]
    idx = (sub + np.arange(f) * (bins + 1)).ravel()
    size = f * (bins + 1)
    out = np.empty((3, f, bins + 1))
    out[0] = np.bincount(idx, weights=np.repeat(g[rows], f), minlength=size).reshape(f, -1)
    out[1] = np.bincount(idx, weights=np.repeat(h[rows], f), minlength=size).reshape(f, -1)
    out[2] = np.bincount(idx, minlength=size).reshape(f, -1)
    return out


def gains(hist, lam=0.0, mcw=0.0, force=None):
    """The gain of every split the histograms allow: ``[feature, cut, side]``, where
    events in bins up to ``cut`` go left and ``side`` says where the missing ones go
    (0: left, 1: right). A split that leaves a side empty, or with less summed
    curvature than ``mcw`` (min_child_weight), is minus infinity. ``force`` (0 or 1)
    allows one side only for the missing ones: no direction is learnt."""
    G, H, N = hist[0, :, :-1], hist[1, :, :-1], hist[2, :, :-1]
    Gm, Hm, Nm = hist[0, :, -1:], hist[1, :, -1:], hist[2, :, -1:]
    Gt, Ht, Nt = hist[0, 0].sum(), hist[1, 0].sum(), hist[2, 0].sum()
    GL = np.stack([np.cumsum(G, 1) + Gm, np.cumsum(G, 1)], -1)
    HL = np.stack([np.cumsum(H, 1) + Hm, np.cumsum(H, 1)], -1)
    NL = np.stack([np.cumsum(N, 1) + Nm, np.cumsum(N, 1)], -1)
    GR, HR, NR = Gt - GL, Ht - HL, Nt - NL
    with np.errstate(divide="ignore", invalid="ignore"):
        gain = 0.5 * (GL * GL / (HL + lam) + GR * GR / (HR + lam) - Gt * Gt / (Ht + lam))
    bad = (NL < 0.5) | (NR < 0.5) | (np.minimum(HL, HR) < mcw) | ~np.isfinite(gain)
    gain[bad] = -np.inf
    # where nothing is missing the two sides are one split: keep "left" (side 0) only
    none = np.broadcast_to(Nm < 0.5, gain.shape[:2])
    if force is None:
        gain[none, 1] = -np.inf
    else:
        gain[..., 1 - force] = -np.inf
    return gain


def scan(hist, lam=0.0, mcw=0.0, cols=None, force=None):
    """The best split of a node from its histograms: ``(gain, feature, cut,
    missing_left)``, or None when no split is allowed. ``cols``: the features that may
    be used. Ties go to the lowest feature, then the lowest cut, then missing left."""
    gain = gains(hist, lam, mcw, force)
    if cols is not None:
        keep = np.zeros(len(gain), bool)
        keep[cols] = True
        gain[~keep] = -np.inf
    k = int(gain.argmax())
    best = float(gain.ravel()[k])
    if not np.isfinite(best):
        return None
    f, c, side = np.unravel_index(k, gain.shape)
    return best, int(f), int(c), bool(side == 0)


def exact_split(xb, g, h, rows, lam=0.0, mcw=0.0):
    """The exact method: sort the node's events by each feature in turn and try a split
    between every two neighbouring values that differ. ``xb`` holds any numbers that
    keep the order of the values (no missing ones). Returns ``(gain, feature, cut,
    candidates)``: events with ``xb[:, feature] <= cut`` go left; ``candidates`` is how
    many splits were scored. None when no split is allowed."""
    sub = xb[rows]
    order = np.argsort(sub, axis=0, kind="stable")
    v = np.take_along_axis(sub, order, 0)
    GL = np.cumsum(g[rows][order], 0)[:-1]
    HL = np.cumsum(h[rows][order], 0)[:-1]
    Gt, Ht = g[rows].sum(), h[rows].sum()
    GR, HR = Gt - GL, Ht - HL
    gain = 0.5 * (GL * GL / (HL + lam) + GR * GR / (HR + lam) - Gt * Gt / (Ht + lam))
    allowed = (v[1:] != v[:-1]) & (np.minimum(HL, HR) >= mcw)
    gain[~allowed] = -np.inf
    k = int(gain.T.argmax())                          # lowest feature, then lowest value
    f, i = divmod(k, gain.shape[0])
    if not np.isfinite(gain[i, f]):
        return None
    return float(gain[i, f]), int(f), v[i, f].item(), int((v[1:] != v[:-1]).sum())


# ====================================================================== one tree
def grow(xb, g, h, bins, depth=3, lam=0.0, mcw=0.0, rows=None, cols=None, leaves=None,
         policy="level", node_cols=None, method="hist", subtract=True, stats=None,
         curve=None, force=None):
    """One tree, grown from the slopes ``g`` and curvatures ``h`` of the events ``rows``
    on binned features ``xb`` (missing values in slot ``bins``).

    A node's best split comes from its histograms (:func:`scan`; ``method="exact"``:
    from sorting, :func:`exact_split`) and is made when its gain is positive, until
    ``depth`` levels or ``leaves`` leaves. ``policy="level"`` splits the nodes in the
    order they were made, level by level, as XGBoost does by default; ``"leaf"`` always
    splits the leaf with the largest gain next. With ``subtract`` only the child with
    fewer events fills its histograms from its events: its sibling's are the parent's
    minus those. ``cols``: the features this tree may use; ``node_cols=(rng, k)``: ``k``
    features drawn afresh at every node (a forest's way). ``stats`` (a dict) counts
    the work: ``added`` values put into bins, ``sorted`` values sorted, ``scored``
    candidate splits, ``seconds`` spent finding splits. ``curve``: curvatures to work
    the leaf values out from when the splits were chosen with another ``h``.

    Returns the tree as a dict of arrays, one entry per node (node 0 is the root):
    ``feature``, ``cut`` (bins up to ``cut`` go left), ``miss_left``, ``left``,
    ``right`` (children; -1 at a leaf), ``value`` (``-G / (H + lambda)``), ``G``, ``H``,
    ``N``, ``gain`` (of the split made there, before gamma), ``level``, ``order`` (the
    order in which the splits were made; -1 at a leaf)."""
    rows = np.arange(len(xb)) if rows is None else np.asarray(rows)
    count = stats if stats is not None else {}
    for key in ("added", "sorted", "scored", "seconds"):
        count.setdefault(key, 0)
    nodes, members, hists, found = [], [], [], []

    def find(k):
        r = members[k]
        t0 = time.perf_counter()
        use = cols
        if node_cols is not None:
            use = node_cols[0].choice(xb.shape[1], node_cols[1], replace=False)
        if method == "exact":
            count["sorted"] += r.size * xb.shape[1]
            got = exact_split(xb, g, h, r, lam, mcw)
            if got is not None:
                count["scored"] += got[3]
                got = (got[0], got[1], got[2], True)
        else:
            if hists[k] is None:
                hists[k] = histograms(xb, g, h, r, bins)
                count["added"] += r.size * xb.shape[1]
            count["scored"] += int((hists[k][2, :, :-1] > 0).sum())
            got = scan(hists[k], lam, mcw, use, force)
        count["seconds"] += time.perf_counter() - t0
        return got

    def add(r, level, hist=None):
        G, H = float(g[r].sum()), float((h if curve is None else curve)[r].sum())
        nodes.append(dict(feature=-1, cut=-1, miss_left=True, left=-1, right=-1,
                          value=leaf_value(G, H, lam), G=G, H=H, N=len(r), gain=0.0,
                          level=level, order=-1))
        members.append(r)
        hists.append(hist)
        found.append(None)
        k = len(nodes) - 1
        if level < depth and len(r) > 1:
            found[k] = find(k)
        return k

    add(rows, 0)
    open_, made = [0], 0
    while open_ and (leaves is None or made + 1 < leaves):
        ready = [k for k in open_ if found[k] is not None and found[k][0] > 1e-12]
        if not ready:
            break
        k = max(ready, key=lambda j: (found[j][0], -j)) if policy == "leaf" else ready[0]
        open_.remove(k)
        gain, f, c, miss_left = found[k]
        r = members[k]
        v = xb[r, f]
        go_left = (v <= c) | (v == bins) if miss_left else (v <= c)
        rl, rr = r[go_left], r[~go_left]
        hl = hr = None
        if method == "hist" and subtract and nodes[k]["level"] + 1 < depth:
            small = rl if len(rl) <= len(rr) else rr
            t0 = time.perf_counter()
            hs = histograms(xb, g, h, small, bins)
            count["seconds"] += time.perf_counter() - t0
            count["added"] += small.size * xb.shape[1]
            hl, hr = (hs, hists[k] - hs) if small is rl else (hists[k] - hs, hs)
        hists[k] = None
        nodes[k].update(feature=f, cut=c, miss_left=bool(miss_left), gain=gain, order=made)
        nodes[k]["left"] = add(rl, nodes[k]["level"] + 1, hl)
        nodes[k]["right"] = add(rr, nodes[k]["level"] + 1, hr)
        open_ += [nodes[k]["left"], nodes[k]["right"]]
        made += 1
    return {key: np.array([n[key] for n in nodes]) for key in nodes[0]}


def prune(tree, gamma):
    """XGBoost's pruning: working up from the leaves, a split whose two children are
    leaves and whose gain is below ``gamma`` is undone (its node becomes a leaf).
    Returns ``(tree, removed)``; the nodes keep their numbers."""
    t = {k: v.copy() for k, v in tree.items()}
    removed = 0
    for k in np.argsort(-t["level"], kind="stable"):
        if t["left"][k] < 0:
            continue
        a, b = t["left"][k], t["right"][k]
        if t["left"][a] < 0 and t["left"][b] < 0 and t["gain"][k] < gamma:
            t["left"][k] = t["right"][k] = -1
            t["order"][k] = -1
            removed += 1
    return t, removed


def reached(tree):
    """``(inner, leaf)``: the numbers of the nodes still in the tree after pruning."""
    inner, leaf, todo = [], [], [0]
    while todo:
        k = todo.pop()
        if tree["left"][k] < 0:
            leaf.append(k)
        else:
            inner.append(k)
            todo += [int(tree["right"][k]), int(tree["left"][k])]
    return np.array(inner, np.int64), np.array(leaf, np.int64)


def n_leaves(tree):
    """Leaves that can be reached from the root."""
    return int(len(reached(tree)[1]))


def leaf_of(tree, xb, bins):
    """The number of the leaf each row of ``xb`` reaches."""
    node = np.zeros(len(xb), np.int64)
    rows = np.arange(len(xb))
    for _ in range(int(tree["level"].max())):
        inner = tree["left"][node] >= 0
        v = xb[rows, np.maximum(tree["feature"][node], 0)]
        left = np.where(v == bins, tree["miss_left"][node], v <= tree["cut"][node])
        node = np.where(inner, np.where(left, tree["left"][node], tree["right"][node]), node)
    return node


def predict(tree, xb, bins):
    """The leaf value each row of ``xb`` reaches."""
    return tree["value"][leaf_of(tree, xb, bins)]


# ====================================================================== many trees
def boost(xb, y, bins, rounds, rate=0.1, depth=3, lam=0.0, gamma=0.0, mcw=0.0,
          subsample=1.0, colsample=1.0, second=True, watch=(), seed=None, leaves=None,
          policy="level", keep=False, method="hist", stats=None, force=None,
          subtract=True):
    """Boosting for the log-loss from a score of zero (even odds).

    Every round: the slope ``g`` and curvature ``h`` of each event's loss at its
    current score; a tree grown from them (:func:`grow`) on a random ``subsample`` of
    the rows and ``colsample`` of the features, pruned with ``gamma``; ``rate`` times
    its leaf values added to every event's score. ``second=False`` is plain gradient
    boosting as Friedman gave it for this loss: the splits are chosen with the
    curvature taken as 1 for every event, so that the tree is the least-squares fit to
    ``-g``, and then each leaf is given the value ``-G / H`` (one Newton step).

    ``watch``: ``(xb, y)`` sets followed along. Returns a dict: ``loss`` and ``error``,
    ``[set, round]`` with set 0 the training events and round 0 the start; ``leaves``
    per tree; with ``keep``, ``trees`` and ``scores`` (``[set][round]``)."""
    rng = np.random.default_rng(seed)
    n, f = xb.shape
    sets = [(xb, y), *watch]
    score = [np.zeros(len(s[1])) for s in sets]
    loss = [[log_loss(s, t[1])] for s, t in zip(score, sets)]
    err = [[error_rate(s, t[1])] for s, t in zip(score, sets)]
    kept = [[s.astype(np.float32)] for s in score]
    trees, sizes = [], []
    for _ in range(rounds):
        g, h = slopes(score[0], y)
        curve = None
        if not second:
            curve, h = h, np.ones(n)
        rows = None
        if subsample < 1.0:
            rows = np.sort(rng.choice(n, max(2, int(round(subsample * n))), replace=False))
        cols = None
        if colsample < 1.0:
            cols = np.sort(rng.choice(f, max(1, int(round(colsample * f))), replace=False))
        tree = grow(xb, g, h, bins, depth, lam, mcw, rows, cols, leaves, policy,
                    method=method, stats=stats, curve=curve, force=force, subtract=subtract)
        if gamma > 0:
            tree, _ = prune(tree, gamma)
        sizes.append(n_leaves(tree))
        for k, (sx, sy) in enumerate(sets):
            score[k] = score[k] + rate * predict(tree, sx, bins)
            loss[k].append(log_loss(score[k], sy))
            err[k].append(error_rate(score[k], sy))
            if keep:
                kept[k].append(score[k].astype(np.float32))
        if keep:
            trees.append(tree)
    out = dict(loss=np.array(loss), error=np.array(err), leaves=np.array(sizes))
    if keep:
        out.update(trees=trees, scores=[np.array(k) for k in kept])
    return out


def stop_round(valid, patience):
    """Early stopping: the round with the lowest validation loss among those seen when
    training stops, which is the first time ``patience`` rounds in a row have failed to
    improve on the best (or the last round). Returns ``(chosen, stopped_at)``."""
    best, at = np.inf, 0
    for k, v in enumerate(valid):
        if v < best - 1e-12:
            best, at = v, k
        elif k - at >= patience:
            return at, k
    return at, len(valid) - 1


# ====================================================================== AdaBoost
def stump(xb, sy, w, bins):
    """The stump with the lowest weighted error: ``(feature, cut, sign, error)``. It
    answers ``sign`` for events in bins above ``cut`` and ``-sign`` for the others;
    ``sy`` are the labels as +1 / -1 and ``w`` the weights, which sum to 1. One pass:
    the sum of ``w * sy`` in every bin of every feature, then a running sum."""
    f = xb.shape[1]
    idx = (xb + np.arange(f) * bins).ravel()
    s = np.bincount(idx, weights=np.repeat(w * sy, f), minlength=f * bins).reshape(f, bins)
    left = np.cumsum(s, 1)[:, :-1]
    edge = s.sum(1, keepdims=True) - 2.0 * left       # sum of w * y * answer for sign +1
    k = int(np.abs(edge).argmax())
    j, c = divmod(k, bins - 1)
    return j, c, (1.0 if edge[j, c] >= 0 else -1.0), float((1.0 - abs(edge[j, c])) / 2.0)


def stump_answer(xb, feature, cut, sign):
    return np.where(xb[:, feature] > cut, sign, -sign)


def adaboost(xb, y, bins, rounds, watch=(), keep_weights=()):
    """Discrete AdaBoost with stumps. Every event starts with weight 1/N; each round
    fits the stump with the lowest weighted error ``err``, gives it the say
    ``alpha = ln((1 - err) / err) / 2``, multiplies the weight of every event it got
    wrong by ``e^alpha`` and of every other by ``e^-alpha``, and rescales the weights to
    sum to 1. The vote is the sign of the sum of ``alpha`` times each stump's answer.

    Returns a dict: ``feature``, ``cut``, ``sign``, ``alpha``, ``err`` per round;
    ``votes[set][round]``: the sum after 0, 1, ... rounds (as float32) for the training
    events (set 0) and each ``watch`` set of binned features; ``weights``: the weights
    after the rounds in ``keep_weights``; ``top_share``: the share of all the weight
    that the heaviest 1% of the events carry, after 0, 1, ... rounds."""
    sy = np.where(np.asarray(y) > 0.5, 1.0, -1.0)
    n = len(sy)
    w = np.full(n, 1.0 / n)
    sets = [xb, *watch]
    vote = [np.zeros(len(s)) for s in sets]
    votes = [[v.astype(np.float32)] for v in vote]
    rows = dict(feature=[], cut=[], sign=[], alpha=[], err=[])
    weights = {0: w.copy()}
    top = max(1, n // 100)
    share = [float(np.partition(w, n - top)[n - top:].sum())]
    for m in range(1, rounds + 1):
        j, c, sign, err = stump(xb, sy, w, bins)
        err = min(max(err, 1e-12), 1.0 - 1e-12)
        alpha = 0.5 * np.log((1.0 - err) / err)
        w = w * np.exp(-alpha * sy * stump_answer(xb, j, c, sign))
        w = w / w.sum()
        for k, s in enumerate(sets):
            vote[k] = vote[k] + alpha * stump_answer(s, j, c, sign)
            votes[k].append(vote[k].astype(np.float32))
        for key, v in zip(rows, (j, c, sign, alpha, err)):
            rows[key].append(v)
        if m in keep_weights:
            weights[m] = w.copy()
        share.append(float(np.partition(w, n - top)[n - top:].sum()))
    out = {k: np.array(v) for k, v in rows.items()}
    out.update(top_share=np.array(share),
               votes=[np.array(v) for v in votes],
               weights=np.array([weights[m] for m in keep_weights]))
    return out



# ====================================================================== the two cells
PAIR_DEPTH = 3


@functools.lru_cache(maxsize=1)
def best_pair():
    """The two cells of the two-feature task: the pair on which one tree of
    ``PAIR_DEPTH`` levels makes the fewest mistakes on seed 0's training events (all
    120 pairs tried; least squares on the labels, which is the Gini rule)."""
    tr = draw(0)
    edges = quantile_edges(tr["x"])
    bins = width(edges)
    xb = to_bins(tr["x"], edges, bins)
    g, h = 0.5 - tr["y"], np.ones(len(tr["y"]))
    best = (2.0, 0, 0)
    for a in range(CELLS):
        for b in range(a + 1, CELLS):
            tree = grow(xb[:, [a, b]], g, h, bins, PAIR_DEPTH)
            wrong = float(np.mean((predict(tree, xb[:, [a, b]], bins) > 0) != (tr["y"] > 0.5)))
            if wrong < best[0] - 1e-12:
                best = (wrong, a, b)
    return best[1], best[2]


def plane_grid(lo, hi, side):
    """``side x side`` points covering the plane of the two cells, row 0 at the top
    (the highest value of the second cell), as ``(side * side, 2)`` values."""
    xs = lo[0] + (hi[0] - lo[0]) * (np.arange(side) + 0.5) / side
    ys = hi[1] - (hi[1] - lo[1]) * (np.arange(side) + 0.5) / side
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


# ====================================================================== 165: AdaBoost
ADA_TWO_N = 100                   # events of each class in the two-feature example
ADA_TWO_BINS = 16
ADA_TWO_ROUNDS = 100
ADA_TWO_SHOWN = (1, 2, 3, 10, 100)
ADA_FIRST = 3                     # stumps drawn one by one
ADA_SIDE = 96                     # the plane is evaluated on a 96 x 96 grid
ADA_ROUNDS = 400
ADA_MARGIN_AT = (1, 10, 100, 400)
ADA_MARGIN_EDGES = np.linspace(-1.0, 1.0, 81)
ADA_HEAVY = 8                     # heaviest training events shown as pictures


def ada_two():
    """AdaBoost on the two cells: ``ADA_TWO_N`` events of each class (the first of seed
    0's training sample)."""
    tr, d = draw(0), data()
    a, b = best_pair()
    pick = np.r_[np.arange(ADA_TWO_N), TRAIN + np.arange(ADA_TWO_N)]
    x, y = tr["x"][pick][:, [a, b]], tr["y"][pick]
    edges = quantile_edges(x, ADA_TWO_BINS)
    bins = ADA_TWO_BINS
    xb = to_bins(x, edges, bins)
    hx = to_bins(d["held_x"][:, [a, b]], edges, bins)
    lo, hi = np.zeros(2), np.maximum(x.max(0), d["held_x"][:, [a, b]].max(0)) * 1.02
    grid = to_bins(plane_grid(lo, hi, ADA_SIDE), edges, bins)
    keep = tuple(range(ADA_FIRST + 1))
    r = adaboost(xb, y, bins, ADA_TWO_ROUNDS, (hx, grid), keep)
    sy, hy = np.where(y > 0.5, 1.0, -1.0), np.where(d["held_y"] > 0.5, 1.0, -1.0)
    answers = np.array([stump_answer(xb, r["feature"][m], r["cut"][m], r["sign"][m])
                        for m in range(ADA_FIRST)])
    return dict(
        x=x, y=y, pair=np.array([a, b]), lo=lo, hi=hi, feature=r["feature"],
        threshold=np.array([edges[f][c] for f, c in zip(r["feature"], r["cut"])]),
        sign=r["sign"], alpha=r["alpha"], err=r["err"], weights=r["weights"],
        wrong=answers != sy,
        train_error=(r["votes"][0] * sy <= 0).mean(1),
        held_error=(r["votes"][1] * hy <= 0).mean(1),
        planes=r["votes"][2][list(ADA_TWO_SHOWN)].reshape(-1, ADA_SIDE, ADA_SIDE),
        images=pictures(tr["id"][pick[[0, 1, ADA_TWO_N, ADA_TWO_N + 1]]]),
        image_cells=x[[0, 1, ADA_TWO_N, ADA_TWO_N + 1]])


def ada_run(seed):
    """AdaBoost on all 16 cells of seed ``seed``'s training events."""
    tr, d = draw(seed), data()
    edges = quantile_edges(tr["x"])
    bins = width(edges)
    xb, hb = to_bins(tr["x"], edges, bins), to_bins(d["held_x"], edges, bins)
    r = adaboost(xb, tr["y"], bins, ADA_ROUNDS, (hb,), ADA_MARGIN_AT)
    sy, hy = np.where(tr["y"] > 0.5, 1.0, -1.0), np.where(d["held_y"] > 0.5, 1.0, -1.0)
    total = np.maximum(np.cumsum(np.r_[0.0, r["alpha"]]), 1e-300)
    margin = np.clip([r["votes"][0][m] * sy / total[m] for m in ADA_MARGIN_AT], -1, 1)
    held_margin = np.clip([r["votes"][1][m] * hy / total[m] for m in ADA_MARGIN_AT], -1, 1)
    order = np.argsort(-r["weights"][-1], kind="stable")[:ADA_HEAVY]
    answers = np.array([stump_answer(xb[order], f, c, s) for f, c, s in
                        zip(r["feature"], r["cut"], r["sign"])])
    return dict(
        train_error=(r["votes"][0] * sy <= 0).mean(1),
        held_error=(r["votes"][1] * hy <= 0).mean(1),
        held_auc=roc_area(r["votes"][1][-1], d["held_y"]),
        alpha=r["alpha"], err=r["err"], feature=r["feature"], top_share=r["top_share"],
        margin=np.array([np.histogram(m, ADA_MARGIN_EDGES)[0] for m in margin]),
        held_margin=np.array([np.histogram(m, ADA_MARGIN_EDGES)[0] for m in held_margin]),
        low_margin=(margin <= 0.05).mean(1),
        heavy_id=tr["id"][order], heavy_label=tr["y"][order],
        heavy_weight=r["weights"][-1][order] * len(sy),
        heavy_wrong=(answers != sy[order]).mean(0),
        heavy_vote=r["votes"][0][-1][order] * sy[order] / total[-1],
        weights_last=np.sort(r["weights"][-1])[::-1] * len(sy))


def _ada_build():
    runs = [ada_run(s) for s in SEEDS]
    out = {"two_" + k: v for k, v in ada_two().items()}
    for key in runs[0]:
        out[key] = np.array([r[key] for r in runs])
    out["heavy_images"] = pictures(out["heavy_id"][0])
    return out


CORE = (cells, data, draw, quantile_edges, to_bins, slopes, histograms, gains, scan, grow,
        prune, leaf_of, boost, stump, adaboost)


def ada_data():
    return _cache("ada", _ada_build, (*CORE, best_pair, plane_grid, ada_two, ada_run),
                  two=(ADA_TWO_N, ADA_TWO_BINS, ADA_TWO_ROUNDS, ADA_TWO_SHOWN, ADA_FIRST,
                       ADA_SIDE), rounds=ADA_ROUNDS, at=ADA_MARGIN_AT, seeds=SEEDS,
                  bins=BINS, heavy=ADA_HEAVY)


def _base(seeds=SEEDS):
    """What every one of the five studies says about its data."""
    d = data()
    return dict(
        seeds=list(seeds), mnist=True, review_only=True, dataset="MNIST",
        stand_in=STAND_IN, physics_counterpart=COUNTERPART,
        task="signal: the digit %d; background: the digit %d; features: the 16 means of "
             "the 7 x 7 pixel cells of each image (A1 top left to D4 bottom right)"
             % (SIGNAL, BACKGROUND),
        split="per seed %d training and %d validation events of each class, drawn "
              "without overlap from a pool of %d of each from the MNIST training file; "
              "held-out: the %d other 4s and 9s of that file and of the test file, the "
              "same for every seed" % (TRAIN, VALID, POOL, len(d["held_y"])),
        where_the_seed_enters="numpy.random.default_rng([%d, 1, seed]) picks the "
                              "training and validation events; row and column "
                              "subsampling, where used, draws from "
                              "numpy.random.default_rng(seed)" % SPLIT,
        two_cells=[cell_name(j) for j in best_pair()],
        two_cells_rule="the pair of cells on which one tree of %d levels makes the "
                       "fewest mistakes on seed 0's training events, of all 120 pairs"
                       % PAIR_DEPTH,
        bins="features are cut at the quantiles of the training events: at most %d bins "
             "a cell" % BINS,
        evidence=dict(EVIDENCE))


def adaboost_study():
    d = ada_data()
    rounds = np.arange(ADA_ROUNDS + 1)
    held, train = d["held_error"], d["train_error"]
    best_at = held.mean(0).argmin()
    centres = (ADA_MARGIN_EDGES[:-1] + ADA_MARGIN_EDGES[1:]) / 2
    cumulative = np.cumsum(d["margin"][0], 1) / d["margin"][0].sum(1, keepdims=True)
    a, b = (cell_name(j) for j in d["two_pair"])
    first = [dict(cell=cell_name(d["two_pair"][d["two_feature"][m]]),
                  threshold=float(d["two_threshold"][m]), err=float(d["two_err"][m]),
                  alpha=float(d["two_alpha"][m]), wrong=int(d["two_wrong"][m].sum()),
                  signal_above=bool(d["two_sign"][m] > 0))
             for m in range(ADA_FIRST)]
    meta = _base()
    meta.update(
        source="NumPy AdaBoost with stumps on MNIST",
        model="discrete AdaBoost (Freund and Schapire): stumps on features cut at "
              "quantiles, weights multiplied by e^alpha on a mistake and e^-alpha "
              "otherwise, alpha = ln((1 - err) / err) / 2",
        two=dict(cells=[a, b], events=2 * ADA_TWO_N, bins=ADA_TWO_BINS, first=first,
                 shown=list(ADA_TWO_SHOWN),
                 train_error=_floats(d["two_train_error"][list(ADA_TWO_SHOWN)]),
                 held_error=_floats(d["two_held_error"][list(ADA_TWO_SHOWN)]),
                 heaviest_after_first=float(d["two_weights"][1].max() * 2 * ADA_TWO_N),
                 lightest_after_first=float(d["two_weights"][1].min() * 2 * ADA_TWO_N)),
        rounds=ADA_ROUNDS,
        train_error=dict(first=float(train[:, 1].mean()), last=float(train[:, -1].mean())),
        held_error=dict(first=float(held[:, 1].mean()), last=float(held[:, -1].mean()),
                        per_seed=_floats(held[:, -1]), lowest=float(held.mean(0).min()),
                        lowest_at=int(best_at)),
        held_auc=dict(mean=float(d["held_auc"].mean()), per_seed=_floats(d["held_auc"])),
        margin=dict(at=list(ADA_MARGIN_AT),
                    definition="label (+1 or -1) times the vote, divided by the sum of "
                               "the alphas: between -1 and 1, negative when wrong",
                    share_at_most_0_05=_floats(d["low_margin"].mean(0)),
                    median=[float(np.interp(0.5, c, ADA_MARGIN_EDGES[1:]))
                            for c in cumulative]),
        weights=dict(top_share_start=float(d["top_share"][:, 0].mean()),
                     top_share_end=float(d["top_share"][:, -1].mean()),
                     top_share_per_seed=_floats(d["top_share"][:, -1]),
                     heaviest=_floats(d["heavy_weight"][0]),
                     heaviest_labels=[int(SIGNAL if v > 0.5 else BACKGROUND)
                                      for v in d["heavy_label"][0]],
                     heaviest_wrong_share=_floats(d["heavy_wrong"][0]),
                     heaviest_margin=_floats(d["heavy_vote"][0])),
        limitations=[
            "Digits stand in for events; stumps only (one question each), so the "
            "training error never reaches zero here and the textbook picture of "
            "held-out error still falling after it does cannot be tested.",
            "The two-cell example uses %d events so that each can be seen."
            % (2 * ADA_TWO_N),
            "Whether a heavy event is mislabelled cannot be measured: the pictures are "
            "shown for the reader to judge.",
        ],
        designs_tried=[
            "The rounds (%d), the rounds at which margins are kept and the 1%% share "
            "of the heaviest events were fixed before any run." % ADA_ROUNDS,
        ])
    return Study(
        "AdaBoost",
        "Stumps fitted one after another to re-weighted events: on two cells of the "
        "digits, where the partition can be drawn, then on all sixteen",
        (
            Panel("Two cells: the events and the stumps",
                  "xy: cells %s and %s of %d training events; classes 1 signal (4s), 0 "
                  "background (9s). threshold, feature, sign, alpha, err: the stumps in "
                  "order; weights[m]: every event's weight after m stumps; planes: the "
                  "vote over the plane after %s stumps."
                  % (a, b, 2 * ADA_TWO_N, ", ".join(map(str, ADA_TWO_SHOWN))), "points",
                  dict(xy=d["two_x"], classes=d["two_y"], threshold=d["two_threshold"],
                       feature=d["two_feature"], sign=d["two_sign"], alpha=d["two_alpha"],
                       err=d["two_err"], weights=d["two_weights"], wrong=d["two_wrong"],
                       planes=d["two_planes"], lo=d["two_lo"], hi=d["two_hi"],
                       train_error=d["two_train_error"], held_error=d["two_held_error"],
                       images=d["two_images"], image_cells=d["two_image_cells"])),
            Panel("16 cells: error against rounds",
                  "ys: training and held-out error after x stumps, mean of %d seeds; "
                  "per_seed has the seeds." % len(SEEDS), "curve",
                  dict(x=rounds, ys=np.array([train.mean(0), held.mean(0)]),
                       labels=["training", "held-out"], per_seed=np.array([train, held]),
                       top_share=d["top_share"])),
            Panel("16 cells: the margins of the training events",
                  "ys: the share of seed 0's training events whose margin is at most x, "
                  "after %s stumps. counts: the histograms behind them, for every seed; "
                  "held_counts: the same for held-out events."
                  % ", ".join(map(str, ADA_MARGIN_AT)), "curve",
                  dict(x=ADA_MARGIN_EDGES[1:], ys=cumulative,
                       labels=["%d stumps" % m for m in ADA_MARGIN_AT], centres=centres,
                       counts=d["margin"], held_counts=d["held_margin"])),
            Panel("The heaviest training events after %d rounds" % ADA_ROUNDS,
                  "values: the %d training events of seed 0 with the largest weights; "
                  "weights: each as a multiple of the average weight; labels: their "
                  "true digit; all_weights: every event's, sorted, per seed."
                  % ADA_HEAVY, "images",
                  dict(values=d["heavy_images"], weights=d["heavy_weight"][0],
                       labels=[str(SIGNAL if v > 0.5 else BACKGROUND)
                               for v in d["heavy_label"][0]],
                       wrong_share=d["heavy_wrong"][0], margin=d["heavy_vote"][0],
                       all_weights=d["weights_last"])),
        ), meta)


# ====================================================================== shared by 166-169
def prep(seed, bins=BINS, change=None, edges=None):
    """Seed ``seed``'s events as binned features: a dict with ``xb``, ``y`` (training),
    ``valid`` and ``held`` (``(xb, y)`` pairs), ``bins``, ``edges`` and the raw ``x``.
    ``change(x, which)``: applied to the readings first (to remove some);
    ``edges``: a function of the training readings (default: quantiles)."""
    tr, d = draw(seed), data()
    x, vx, hx = tr["x"], tr["vx"], d["held_x"]
    if change is not None:
        x, vx, hx = change(x, 0), change(vx, 1), change(hx, 2)
    e = quantile_edges(x, bins) if edges is None else edges(x)
    b = max(width(e), 2)
    return dict(xb=to_bins(x, e, b), y=tr["y"], valid=(to_bins(vx, e, b), tr["vy"]),
                held=(to_bins(hx, e, b), d["held_y"]), bins=b, edges=e, x=x, id=tr["id"])


def run(seed, rounds, bins=BINS, change=None, edges=None, **settings):
    """One boosting run on seed ``seed``'s events, followed on the validation (set 1)
    and held-out (set 2) events."""
    q = prep(seed, bins, change, edges)
    return boost(q["xb"], q["y"], q["bins"], rounds, watch=(q["valid"], q["held"]),
                 seed=seed, **settings)


def chosen(r):
    """The round with the lowest validation loss, and the held-out loss there."""
    k = int(r["loss"][1].argmin())
    return k, float(r["loss"][2][k])


def pack(tree, edges):
    """A tree as arrays to keep: its nodes' fields and the reading each split tests
    (events with a reading below ``threshold`` go left; 0 at a leaf)."""
    out = {k: np.asarray(v) for k, v in tree.items()}
    out["threshold"] = np.array([edges[f][c] if f >= 0 and c < len(edges[f]) else 0.0
                                 for f, c in zip(tree["feature"], tree["cut"])])
    return out


def members(tree, xb, bins, rows=None):
    """The events of every node: ``{node: rows}``."""
    rows = np.arange(len(xb)) if rows is None else rows
    out, todo = {}, [(0, rows)]
    while todo:
        k, r = todo.pop()
        out[k] = r
        if tree["left"][k] >= 0:
            v = xb[r, tree["feature"][k]]
            left = np.where(v == bins, tree["miss_left"][k], v <= tree["cut"][k])
            todo += [(int(tree["left"][k]), r[left]), (int(tree["right"][k]), r[~left])]
    return out


STATE = dict(rate=0.3, depth=3, lam=1.0)
STATE_AT = 10                     # the rounds already made when a node is looked at


@functools.lru_cache(maxsize=2)
def state(seed=0):
    """Seed ``seed``'s training events after ``STATE_AT`` rounds of the XGBoost-style
    boosting with the settings ``STATE``: their scores, slopes and curvatures."""
    q = prep(seed)
    r = boost(q["xb"], q["y"], q["bins"], STATE_AT, keep=True, **STATE)
    score = r["scores"][0][-1].astype(float)
    g, h = slopes(score, q["y"])
    return dict(q, score=score, g=g, h=h, trees=r["trees"])


# ====================================================================== 166: settings
GB_RATES = (1.0, 0.3, 0.1, 0.03)
GB_DEPTH, GB_RATE = 3, 0.1        # the setting the other sweeps start from
GB_ROUNDS = 2000
GB_SUBSAMPLES = (1.0, 0.5, 0.25, 0.1)
GB_DEPTHS = (1, 2, 3, 4, 6)
GB_PATIENCE = 50
GB_ONE = dict(rate=0.3, depth=2)  # the round that is drawn
GB_ONE_AT = 10                    # trees already added when it is drawn
GB_ONE_SHOWN = 20                 # events of each class drawn


def gb_configs():
    """Every ``(rate, depth, subsample)`` the three sweeps need, once."""
    out = [(r, GB_DEPTH, 1.0) for r in GB_RATES]
    out += [(GB_RATE, GB_DEPTH, f) for f in GB_SUBSAMPLES]
    out += [(GB_RATE, d, 1.0) for d in GB_DEPTHS]
    return list(dict.fromkeys(out))


def gb_run(seed, rate, depth, subsample):
    r = run(seed, GB_ROUNDS, rate=rate, depth=depth, subsample=subsample, second=False)
    return r["loss"].astype(np.float32), r["error"][2].astype(np.float32)


def gb_one():
    """One round of gradient boosting drawn: seed 0's training events after
    ``GB_ONE_AT`` trees, the pull ``y - p`` on each, the tree fitted to the pulls and
    the step it makes."""
    q = prep(0)
    xb, y, bins = q["xb"], q["y"], q["bins"]
    r = boost(xb, y, bins, GB_ONE_AT + 1, second=False, keep=True, watch=(q["held"],),
              **GB_ONE)
    before, after = (r["scores"][0][k].astype(float) for k in (GB_ONE_AT, GB_ONE_AT + 1))
    tree = r["trees"][GB_ONE_AT]
    shown = np.r_[np.arange(GB_ONE_SHOWN), TRAIN + np.arange(GB_ONE_SHOWN)]
    out = {"tree_" + k: v for k, v in pack(tree, q["edges"]).items()}
    out.update(shown_before=before[shown], shown_after=after[shown], shown_y=y[shown],
               shown_leaf=leaf_of(tree, xb[shown], bins),
               loss=r["loss"][:, GB_ONE_AT:GB_ONE_AT + 2],
               mean_pull=np.array([(y - sigmoid(before))[m].mean() for _, m in
                                   sorted(members(tree, xb, bins).items())]))
    return out


def _gb_build():
    configs = gb_configs()
    runs = _parallel(gb_run, [(s, *c) for c in configs for s in SEEDS])
    shape = (len(configs), len(SEEDS))
    out = {"one_" + k: v for k, v in gb_one().items()}
    out.update(loss=np.array([r[0] for r in runs]).reshape(*shape, 3, -1),
               error=np.array([r[1] for r in runs]).reshape(*shape, -1),
               configs=np.array(configs))
    return out


def gb_data():
    return _cache("gb", _gb_build, (*CORE, prep, run, pack, members, gb_configs, gb_run,
                                    gb_one),
                  rates=GB_RATES, depths=GB_DEPTHS, subsamples=GB_SUBSAMPLES,
                  base=(GB_RATE, GB_DEPTH), rounds=GB_ROUNDS, one=(GB_ONE, GB_ONE_AT,
                                                                 GB_ONE_SHOWN),
                  seeds=SEEDS, bins=BINS)


def gb_summary():
    d = gb_data()
    configs = [tuple(c) for c in d["configs"].tolist()]
    held, valid = d["loss"][:, :, 2], d["loss"][:, :, 1]         # config, seed, round
    stops = np.array([[stop_round(v, GB_PATIENCE) for v in row] for row in valid])
    pick = np.take_along_axis(held, stops[..., :1], 2)[..., 0]
    out = dict(d=d, configs=configs, held=held, valid=valid, train=d["loss"][:, :, 0],
               stop=stops[..., 0], stopped_at=stops[..., 1], at_stop=pick,
               lowest=held.min(2), lowest_at=held.argmin(2), end=held[..., -1])
    out["rates"] = [configs.index((r, GB_DEPTH, 1.0)) for r in GB_RATES]
    out["subs"] = [configs.index((GB_RATE, GB_DEPTH, f)) for f in GB_SUBSAMPLES]
    out["depths"] = [configs.index((GB_RATE, dd, 1.0)) for dd in GB_DEPTHS]
    return out


# ====================================================================== 167: the gain
GAIN_LAM = 1.0                    # lambda of the worked example
GAIN_HAND = 4                     # events of each class in the worked example
GAIN_P = 0.3                      # the one event drawn: the signal event nearest this p
GAIN_W = np.linspace(-3.0, 3.0, 121)
GAIN_LAMS = (0.0, 1.0, 10.0)
GAIN_SWEEP = dict(rate=0.1, depth=4, lam=1.0)
GAIN_SWEEP_ROUNDS = 600
GAIN_GAMMAS = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0)
GAIN_LAMBDAS = (0.0, 1.0, 10.0, 100.0)
GAIN_MCWS = (0.0, 1.0, 5.0, 20.0)
GAIN_COMPARE = dict(rate=0.1, depth=3)
GAIN_COMPARE_ROUNDS = 1000
GAIN_DIFFER_ROUNDS = 100
GAIN_NODE_MCW = 1.0               # min_child_weight drawn on the small node
GAIN_NODE_GAMMA = 1.0


def approx(G, H, lam, w):
    """The second-order estimate of how the loss of a leaf's events changes when ``w``
    is added to their scores, with the penalty: ``G w + (H + lambda) w^2 / 2``."""
    return G * w + 0.5 * (H + lam) * w * w


def gain_one():
    """What the split-gain animation draws from seed 0's state: one event's loss
    curve, a handful of events as a leaf and as a split, a small real node under three
    lambdas, and the root as the two criteria see it."""
    st = state(0)
    xb, y, g, h, score, bins = st["xb"], st["y"], st["g"], st["h"], st["score"], st["bins"]
    p = sigmoid(score)
    one = int(np.flatnonzero(y > 0.5)[np.abs(p[y > 0.5] - GAIN_P).argmin()])
    f = best_pair()[0]                               # the handful is laid out along this cell
    hand = np.r_[np.arange(GAIN_HAND), TRAIN + np.arange(GAIN_HAND)]
    hand = hand[np.argsort(st["x"][hand, f], kind="stable")]
    hg, hh = g[hand], h[hand]
    GL, HL = np.cumsum(hg)[:-1], np.cumsum(hh)[:-1]
    exact = np.array([(event_loss(score[hand] + w, y[hand]) - event_loss(score[hand],
                                                                          y[hand])).sum()
                      + 0.5 * GAIN_LAM * w * w for w in GAIN_W])
    # a small real node: the smallest of the next tree's nodes on its last level but one
    tree = grow(xb, g, h, bins, STATE["depth"], STATE["lam"])
    rows = members(tree, xb, bins)
    level = [k for k in rows if tree["level"][k] == STATE["depth"] - 1 and tree["left"][k] >= 0]
    k = min(level, key=lambda j: len(rows[j]))
    nh = histograms(xb, g, h, rows[k], bins)
    nf = int(tree["feature"][k])
    valid = np.flatnonzero(np.cumsum(nh[2, nf, :-1])[:-1] > 0)
    valid = valid[np.cumsum(nh[2, nf, :-1])[valid] < len(rows[k])]
    cg, ch = np.cumsum(nh[0, nf, :-1])[valid], np.cumsum(nh[1, nf, :-1])[valid]
    Gk, Hk = float(g[rows[k]].sum()), float(h[rows[k]].sum())
    return dict(
        one=np.array([score[one], p[one], g[one], h[one]]),
        one_curve=event_loss(score[one] + GAIN_W, 1.0), one_id=st["id"][one],
        feature=f, hand_x=st["x"][hand, f], hand_y=y[hand], hand_g=hg, hand_h=hh,
        hand_score=score[hand], hand_GL=GL, hand_HL=HL,
        hand_gain=split_gain(GL, HL, hg.sum() - GL, hh.sum() - HL, GAIN_LAM),
        hand_exact=exact, hand_id=st["id"][hand],
        node=np.array([k, nf, len(rows[k]), Gk, Hk]),
        node_x=np.array([st["edges"][nf][v] for v in valid]),
        node_GL=cg, node_HL=ch,
        node_gain=np.array([split_gain(cg, ch, Gk - cg, Hk - ch, lam) for lam in GAIN_LAMS]),
        h_range=np.percentile(h, [0, 25, 50, 75, 100]), **gain_disagree(st))


def gain_disagree(st):
    """The first node, going on from the state round by round and through each tree
    from the root, at which the best split by ``G^2 / H`` (second order, lambda 0) is
    not the best by ``G^2 / N`` (least squares on the gradient): both splits, each
    scored both ways, and the leaf values each way of working them out gives."""
    xb, y, bins, edges = st["xb"], st["y"], st["bins"], st["edges"]
    score, ones = st["score"].copy(), np.ones(len(y))
    for more in range(1, 200):
        g, h = slopes(score, y)
        tree = grow(xb, g, h, bins, STATE["depth"])
        for k, rows in sorted(members(tree, xb, bins).items()):
            if tree["left"][k] < 0:
                continue
            second = histograms(xb, g, h, rows, bins)
            first = histograms(xb, g, ones, rows, bins)
            a, b = scan(second), scan(first)
            if a[1:3] == b[1:3]:
                continue
            gs, gf = gains(second), gains(first)
            sides = []
            for _, f, c, _ in (a, b):
                left = rows[xb[rows, f] <= c]
                right = rows[xb[rows, f] > c]
                sides.append([[g[r].sum(), h[r].sum(), len(r)] for r in (left, right)])
            return dict(
                dis_where=np.array([STATE_AT + more, k, len(rows)]),
                dis_split=np.array([[a[1], edges[a[1]][a[2]]], [b[1], edges[b[1]][b[2]]]]),
                dis_score=np.array([[gs[a[1], a[2], 0], gs[b[1], b[2], 0]],
                                    [gf[a[1], a[2], 0], gf[b[1], b[2], 0]]]),
                dis_sides=np.array(sides))
        score = score + STATE["rate"] * predict(tree, xb, bins)
    raise RuntimeError("the two criteria never disagreed")


def gain_sweep(seed, knob, value):
    r = run(seed, GAIN_SWEEP_ROUNDS, **dict(GAIN_SWEEP, **{knob: value}))
    k, loss = chosen(r)
    return loss, float(r["loss"][2][-1]), float(r["leaves"].mean()), k


def gain_compare(seed, second, lam):
    r = run(seed, GAIN_COMPARE_ROUNDS, second=bool(second), lam=lam, **GAIN_COMPARE)
    return r["loss"][2].astype(np.float32), chosen(r)[0]


def gain_differ(seed):
    """For every node of the trees of ``GAIN_DIFFER_ROUNDS`` rounds of second-order
    boosting (lambda 0): does the best split by ``G^2 / H`` differ from the best by
    ``G^2 / N``, the least-squares fit that plain gradient boosting makes? Returns per
    round the number of nodes that split and the number where the two differ."""
    q = prep(seed)
    xb, y, bins = q["xb"], q["y"], q["bins"]
    score, ones = np.zeros(len(y)), np.ones(len(y))
    out = []
    for _ in range(GAIN_DIFFER_ROUNDS):
        g, h = slopes(score, y)
        tree = grow(xb, g, h, bins, GAIN_COMPARE["depth"])
        n = differ = 0
        for k, rows in members(tree, xb, bins).items():
            if tree["left"][k] < 0:
                continue
            plain = scan(histograms(xb, g, ones, rows, bins))
            n += 1
            differ += plain is None or plain[1:3] != (tree["feature"][k], tree["cut"][k])
        out.append((n, differ))
        score = score + GAIN_COMPARE["rate"] * predict(tree, xb, bins)
    return np.array(out)


def _gain_build():
    sweeps = ([("gamma", v) for v in GAIN_GAMMAS] + [("lam", v) for v in GAIN_LAMBDAS]
              + [("mcw", v) for v in GAIN_MCWS])
    jobs = [(s, k, v) for k, v in sweeps for s in SEEDS]
    swept = np.array(_parallel(gain_sweep, jobs)).reshape(len(sweeps), len(SEEDS), 4)
    kinds = [(0, 0.0), (1, 0.0), (1, 1.0)]
    cmp_ = _parallel(gain_compare, [(s, a, b) for a, b in kinds for s in SEEDS])
    out = {"one_" + k: v for k, v in gain_one().items()}
    out.update(sweep=swept,
               compare=np.array([c[0] for c in cmp_]).reshape(len(kinds), len(SEEDS), -1),
               compare_at=np.array([c[1] for c in cmp_]).reshape(len(kinds), len(SEEDS)),
               differ=np.array(_parallel(gain_differ, [(s,) for s in SEEDS])))
    return out


def gain_data():
    return _cache("gain", _gain_build, (*CORE, prep, run, pack, members, state, approx,
                                        best_pair, gain_disagree, gain_one, gain_sweep,
                                        gain_compare, gain_differ),
                  state=(STATE, STATE_AT), lam=GAIN_LAM, hand=GAIN_HAND, p=GAIN_P,
                  lams=GAIN_LAMS, sweep=(GAIN_SWEEP, GAIN_SWEEP_ROUNDS, GAIN_GAMMAS,
                                         GAIN_LAMBDAS, GAIN_MCWS),
                  compare=(GAIN_COMPARE, GAIN_COMPARE_ROUNDS, GAIN_DIFFER_ROUNDS),
                  seeds=SEEDS, bins=BINS)


# ====================================================================== 168: histograms
HIST_SHOW_BINS = 8                # bins of the drawn histograms
HIST_SORTED = 12                  # events of the sorting example
HIST_DROPPED = 48                 # events dropped into the drawn bins
HIST_SET = dict(rate=0.1, depth=3, lam=1.0)
HIST_ROUNDS = 600
HIST_BINS = (2, 4, 8, 16, 32, 64, 128, 256)
HIST_TIMED_ROUNDS = 40
HIST_TIMED_REPEATS = 3
HIST_LEAVES = 8
HIST_RANDOM = 0.3                 # the share of readings removed at random
HIST_FLOOR = 0.1                  # readings below this are not reported
HIST_RULES = ("random", "floor")
HIST_WAYS = ("learnt", "left", "right", "mean")


def remove(rule, seed):
    """The rule that makes readings missing, as a function for :func:`prep`.
    ``"random"``: every reading is removed with probability ``HIST_RANDOM``, by a
    generator seeded with the seed and the set; ``"floor"``: a reading below
    ``HIST_FLOOR`` is not reported, as a read-out that suppresses zeros would."""
    def change(x, which):
        x = np.array(x, float)
        if rule == "random":
            rng = np.random.default_rng([SPLIT, 2, seed, which])
            x[rng.random(x.shape) < HIST_RANDOM] = np.nan
        else:
            x[x < HIST_FLOOR] = np.nan
        return x
    return change


def impute(change):
    """``change``, then every missing reading replaced by the mean of the readings of
    that cell that are present (in that set's own events)."""
    def filled(x, which):
        x = change(x, which)
        there = ~np.isnan(x)
        means = np.where(there, x, 0.0).sum(0) / np.maximum(there.sum(0), 1)
        return np.where(there, x, means)
    return filled


def hist_missing(seed, rule, way):
    """Boosting on readings with holes. ``way``: ``learnt`` (each split keeps the
    better side for the missing), ``left`` / ``right`` (always that side), ``mean``
    (holes filled with the cell's mean first). Returns the held-out loss at the round
    validation picks, that round, and for ``learnt`` the splits that send the missing
    left and the splits counted."""
    change = remove(rule, seed)
    q = prep(seed, change=impute(change) if way == "mean" else change)
    force = {"left": 0, "right": 1}.get(way)
    r = boost(q["xb"], q["y"], q["bins"], HIST_ROUNDS, watch=(q["valid"], q["held"]),
              keep=True, force=force, **HIST_SET)
    k, loss = chosen(r)
    left = total = 0
    for tree in r["trees"][:k]:
        inner = tree["left"] >= 0
        left += int(tree["miss_left"][inner].sum())
        total += int(inner.sum())
    return loss, k, left, total


def hist_growth(seed, policy):
    depth = 3 if policy == "level" else HIST_LEAVES - 1
    r = run(seed, HIST_ROUNDS, policy=policy, leaves=HIST_LEAVES,
            **dict(HIST_SET, depth=depth))
    return chosen(r)[1], chosen(r)[0]


def hist_bins(seed, bins):
    """Held-out loss at the round validation picks with ``bins`` quantile bins; 0:
    every distinct reading its own bin, which makes the splits the exact method's."""
    r = run(seed, HIST_ROUNDS, bins=bins or BINS, edges=None if bins else value_edges,
            **HIST_SET)
    return chosen(r)[1], chosen(r)[0]


def hist_timing():
    """Split finding timed on this machine for seed 0, ``HIST_TIMED_ROUNDS`` rounds:
    seconds (the least of ``HIST_TIMED_REPEATS`` repeats) and counted operations per
    tree, for each number of bins and for the exact method (last row). Columns:
    seconds, values added to bins, values added without the subtraction trick, values
    sorted, candidate splits scored."""
    out = []
    for bins in (*HIST_BINS, 0):
        q = prep(0, bins or BINS, edges=None if bins else value_edges)
        best, kept = np.inf, None
        for _ in range(HIST_TIMED_REPEATS):
            stats = {}
            boost(q["xb"], q["y"], q["bins"], HIST_TIMED_ROUNDS, stats=stats,
                  method="hist" if bins else "exact", **HIST_SET)
            if stats["seconds"] < best:
                best, kept = stats["seconds"], stats
        plain = {}
        if bins:
            boost(q["xb"], q["y"], q["bins"], HIST_TIMED_ROUNDS, stats=plain, subtract=False,
                  **HIST_SET)
        out.append([best, kept["added"], plain.get("added", 0), kept["sorted"],
                    kept["scored"]])
    return np.array(out, float) / HIST_TIMED_ROUNDS


def hist_one():
    """What the histogram animation draws, from seed 0's state: a few events sorted, a
    few dropped into ``HIST_SHOW_BINS`` bins, the root's histograms and those of its
    children, one split with missing readings, and two trees of ``HIST_LEAVES`` leaves."""
    st = state(0)
    xb, y, g, h, bins, x = st["xb"], st["y"], st["g"], st["h"], st["bins"], st["x"]
    every = np.arange(len(y))
    root = histograms(xb, g, h, every, bins)
    _, rf, rc, _ = scan(root, GAIN_LAM)               # the root's own best split
    f = best_pair()[0]                                # the cell the drawings use
    few = np.r_[np.arange(HIST_SORTED // 2), TRAIN + np.arange(HIST_SORTED // 2)]
    drop = np.r_[np.arange(HIST_DROPPED // 2), TRAIN + np.arange(HIST_DROPPED // 2)]
    rng = np.random.default_rng([SPLIT, 3])
    drop = drop[rng.permutation(len(drop))]           # the order in which they fall
    e8 = quantile_edges(x[:, [f]], HIST_SHOW_BINS)
    b8 = HIST_SHOW_BINS
    col = to_bins(x[:, [f]], e8, b8)
    dh = histograms(col, g, h, drop, b8)[:, 0, :-1]
    dgain = gains(histograms(col, g, h, drop, b8), GAIN_LAM)[0, :b8 - 1, 0]
    # the children hold the events on each side of the root's split
    left = every[xb[:, rf] <= rc]
    right = every[xb[:, rf] > rc]
    sub = [histograms(col, g, h, r, b8)[:, 0, :-1] for r in (every, left, right)]
    # one split with holes: the same cell with readings removed at random
    change = remove("random", 0)
    xm = change(x, 0)[:, [f]]
    em = quantile_edges(xm, HIST_SHOW_BINS)
    mh = histograms(to_bins(xm, em, b8), g, h, every, b8)
    mgain = gains(mh, GAIN_LAM)[0]
    _, _, mc, mleft = scan(mh, GAIN_LAM)
    trees = {}
    for policy in ("level", "leaf"):
        depth = 3 if policy == "level" else HIST_LEAVES - 1
        t = pack(grow(xb, g, h, bins, depth, GAIN_LAM, leaves=HIST_LEAVES, policy=policy),
                 st["edges"])
        trees.update({"%s_%s" % (policy, k): v for k, v in t.items()})
    out = dict(
        feature=f, root=np.array([rf, st["edges"][rf][rc]]),
        few_x=x[few, f], few_y=y[few], few_g=g[few],
        distinct=np.array([len(np.unique(x[:, j])) for j in range(x.shape[1])]),
        edges8=e8[0], drop_x=x[drop, f], drop_y=y[drop], drop_g=g[drop], drop_h=h[drop],
        drop_bin=col[drop, 0], drop_hist=dh, drop_gain=dgain,
        sub_parent=sub[0], sub_left=sub[1],
        sub_right=sub[2], sub_n=np.array([len(every), len(left), len(right)]),
        miss_cut=mc, miss_left=mleft, miss_hist=mh[:, 0], miss_gain=mgain,
        miss_edges=em[0],
        miss_sample=change(x[np.r_[0:3, TRAIN:TRAIN + 3]], 0)[:, :8])
    out.update(trees)
    return out


def _hist_build():
    jobs = [(s, r, w) for r in HIST_RULES for w in HIST_WAYS for s in SEEDS]
    miss = np.array(_parallel(hist_missing, jobs), float).reshape(
        len(HIST_RULES), len(HIST_WAYS), len(SEEDS), 4)
    grown = np.array(_parallel(hist_growth, [(s, p) for p in ("level", "leaf")
                                             for s in SEEDS])).reshape(2, len(SEEDS), 2)
    by_bins = np.array(_parallel(hist_bins, [(s, b) for b in (*HIST_BINS, 0)
                                             for s in SEEDS])).reshape(-1, len(SEEDS), 2)
    full = np.array(_parallel(hist_bins, [(s, BINS) for s in SEEDS]))
    out = {"one_" + k: v for k, v in hist_one().items()}
    out.update(missing=miss, growth=grown, by_bins=by_bins, complete=full,
               timing=hist_timing())
    return out


def hist_data():
    return _cache("hist", _hist_build, (*CORE, exact_split, value_edges, prep, run, pack,
                                        members, state, best_pair, remove, impute, hist_missing,
                                        hist_growth, hist_bins, hist_timing, hist_one),
                  state=(STATE, STATE_AT), show=(HIST_SHOW_BINS, HIST_SORTED, HIST_DROPPED),
                  set=(HIST_SET, HIST_ROUNDS), bins_tried=HIST_BINS,
                  timed=(HIST_TIMED_ROUNDS, HIST_TIMED_REPEATS), leaves=HIST_LEAVES,
                  rules=(HIST_RANDOM, HIST_FLOOR, HIST_RULES, HIST_WAYS), seeds=SEEDS,
                  bins=BINS, lam=GAIN_LAM)


# ====================================================================== 169: one round
ROUND = dict(rate=0.3, depth=3, lam=1.0, gamma=5.0, mcw=1.0)      # the round drawn
ROUND_SHOWN = 4                   # events of each class in the table
ROUND_MANY = dict(rate=0.1, depth=3, lam=1.0, gamma=0.0, mcw=1.0)
ROUND_SAMPLED = dict(subsample=0.5, colsample=0.5)
ROUND_MANY_ROUNDS = 600
LOOSE = dict(rate=0.3, depth=6, lam=0.0, gamma=0.0, mcw=0.0, subsample=1.0, colsample=1.0)
LOOSE_ROUNDS = 200
KNOBS = (("rate", 0.03), ("depth", 2), ("lam", 10.0), ("gamma", 2.0), ("mcw", 10.0),
         ("subsample", 0.5), ("colsample", 0.5))
RIVALS = ("one tree", "forest", "AdaBoost", "gradient boosting", "XGBoost-style")
TREE_DEPTHS = tuple(range(1, 11))
TREE_MIN_LEAF = 5
FOREST = dict(trees=200, depth=10, tried=4)
RIVAL_ROUNDS = 1000
RIVAL_GB = dict(rate=0.1, depth=3)
RIVAL_XGB = dict(rate=0.1, depth=3, lam=1.0, mcw=1.0, subsample=0.8, colsample=0.8)


def round_one():
    """Two rounds of the XGBoost-style loop on seed 0's training events, kept step by
    step: the table of a few events, the root's histograms, the tree before and after
    pruning with gamma, and the loss."""
    q = prep(0)
    xb, y, bins = q["xb"], q["y"], q["bins"]
    hx, hy = q["held"]
    shown = np.r_[np.arange(ROUND_SHOWN), TRAIN + np.arange(ROUND_SHOWN)]
    score, held = np.zeros(len(y)), np.zeros(len(hy))
    out = dict(shown_id=q["id"][shown], shown_y=y[shown], shown_cells=q["x"][shown])
    table, loss = [], [[log_loss(score, y), log_loss(held, hy)]]
    for k in (1, 2):
        g, h = slopes(score, y)
        full = grow(xb, g, h, bins, ROUND["depth"], ROUND["lam"], ROUND["mcw"])
        tree, removed = prune(full, ROUND["gamma"])
        w = predict(tree, xb, bins)
        table.append(np.array([score[shown], sigmoid(score[shown]), g[shown], h[shown],
                               w[shown], leaf_of(tree, xb[shown], bins)]))
        if k == 1:
            root = histograms(xb, g, h, np.arange(len(y)), bins)
            f = int(full["feature"][0])
            out.update(root_hist=root[:, f, :-1], root_edges=q["edges"][f],
                       root_gain=gains(root, ROUND["lam"], ROUND["mcw"])[f, :, 0],
                       best_by_feature=gains(root, ROUND["lam"], ROUND["mcw"]).max((1, 2)),
                       removed=removed, kept=tree["left"] >= 0)
            out.update({"tree_" + key: v for key, v in pack(full, q["edges"]).items()})
        score = score + ROUND["rate"] * w
        held = held + ROUND["rate"] * predict(tree, hx, bins)
        loss.append([log_loss(score, y), log_loss(held, hy)])
    g, h = slopes(score, y)
    table.append(np.array([score[shown], sigmoid(score[shown]), g[shown], h[shown],
                           np.zeros(len(shown)), np.zeros(len(shown))]))
    out.update(table=np.array(table), loss=np.array(loss))
    return out


def round_many(seed, sampled):
    r = run(seed, ROUND_MANY_ROUNDS, **dict(ROUND_MANY, **(ROUND_SAMPLED if sampled else {})))
    return r["loss"].astype(np.float32)


def round_knob(seed, knob):
    """The loose setting with one knob turned (``knob``: an index into ``KNOBS``; -1:
    none): held-out loss after the last round, at the round validation picks, that
    round, and the mean number of leaves."""
    settings = dict(LOOSE)
    if knob >= 0:
        settings[KNOBS[knob][0]] = KNOBS[knob][1]
    r = run(seed, LOOSE_ROUNDS, **settings)
    k, loss = chosen(r)
    return float(r["loss"][2][-1]), loss, k, float(r["leaves"].mean())


def rivals(seed):
    """Five ways to use trees on seed ``seed``'s events, each tuned on the validation
    events only: held-out error and ROC area, in the order of ``RIVALS``, and what was
    picked."""
    q = prep(seed)
    xb, y, bins = q["xb"], q["y"], q["bins"]
    (vb, vy), (hb, hy) = q["valid"], q["held"]
    out, picked = [], []
    g, ones = 0.5 - y, np.ones(len(y))                # least squares on the labels
    # one tree, its depth chosen on validation
    best = None
    for depth in TREE_DEPTHS:
        tree = grow(xb, g, ones, bins, depth, mcw=TREE_MIN_LEAF)
        err = error_rate(predict(tree, vb, bins), vy)
        if best is None or err < best[0] - 1e-12:
            best = (err, depth, tree)
    s = predict(best[2], hb, bins)
    out.append((error_rate(s, hy), roc_area(s, hy)))
    picked.append(best[1])
    # forest-style: deep trees on bootstrap samples, a few cells tried at each node
    rng = np.random.default_rng([SPLIT, 4, seed])
    s = np.zeros(len(hy))
    for _ in range(FOREST["trees"]):
        rows = np.sort(rng.integers(0, len(y), len(y)))
        tree = grow(xb, g, ones, bins, FOREST["depth"], rows=rows,
                    node_cols=(rng, FOREST["tried"]), subtract=False)
        s += predict(tree, hb, bins)
    out.append((error_rate(s, hy), roc_area(s, hy)))
    picked.append(FOREST["trees"])
    # AdaBoost with stumps, the number of stumps chosen on validation
    a = adaboost(xb, y, bins, ADA_ROUNDS, (vb, hb))
    verr = (a["votes"][1] * np.where(vy > 0.5, 1.0, -1.0) <= 0).mean(1)
    k = int(verr[1:].argmin()) + 1
    s = a["votes"][2][k]
    out.append((error_rate(s, hy), roc_area(s, hy)))
    picked.append(k)
    # the two boosters, the number of trees chosen on validation
    for second, settings in ((False, RIVAL_GB), (True, RIVAL_XGB)):
        r = boost(xb, y, bins, RIVAL_ROUNDS, watch=((vb, vy), (hb, hy)), seed=seed,
                  second=second, keep=True, **settings)
        k = int(r["loss"][1].argmin())
        s = r["scores"][2][k]
        out.append((error_rate(s, hy), roc_area(s, hy)))
        picked.append(k)
    return np.array(out), np.array(picked)


def _round_build():
    many = np.array(_parallel(round_many, [(s, k) for k in (0, 1) for s in SEEDS]))
    knobs = np.array(_parallel(round_knob, [(s, k) for k in range(-1, len(KNOBS))
                                            for s in SEEDS]))
    riv = _parallel(rivals, [(s,) for s in SEEDS])
    out = {"one_" + k: v for k, v in round_one().items()}
    out["one_images"] = pictures(out["one_shown_id"])
    out.update(many=many.reshape(2, len(SEEDS), 3, -1),
               knobs=knobs.reshape(len(KNOBS) + 1, len(SEEDS), 4),
               rivals=np.array([r[0] for r in riv]), picked=np.array([r[1] for r in riv]))
    return out


def round_data():
    return _cache("round", _round_build, (*CORE, prep, run, pack, members, round_one,
                                          round_many, round_knob, rivals),
                  round=(ROUND, ROUND_SHOWN), many=(ROUND_MANY, ROUND_SAMPLED,
                                                    ROUND_MANY_ROUNDS),
                  loose=(LOOSE, LOOSE_ROUNDS, KNOBS),
                  rivals=(TREE_DEPTHS, TREE_MIN_LEAF, FOREST, RIVAL_ROUNDS, RIVAL_GB,
                          RIVAL_XGB, ADA_ROUNDS), seeds=SEEDS, bins=BINS)


# ====================================================================== the four studies
def _spread(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()),
                per_seed=_floats(v))


def _tree_meta(t, names=None):
    """A kept tree's nodes as plain lists."""
    return dict(cell=[cell_name(f) if f >= 0 else "" for f in t["feature"]],
                threshold=[float(np.nan_to_num(v)) for v in t["threshold"]],
                left=[int(v) for v in t["left"]], right=[int(v) for v in t["right"]],
                value=_floats(t["value"]), G=_floats(t["G"]), H=_floats(t["H"]),
                N=[int(v) for v in t["N"]], gain=_floats(t["gain"]),
                order=[int(v) for v in t["order"]])


def _sub(d, prefix):
    return {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}


def shrinkage_study():
    r = gb_summary()
    d = r["d"]
    rounds = np.arange(GB_ROUNDS + 1)
    held = r["held"].mean(1)                           # config, round
    tree = _sub(d, "one_tree_")
    main = r["configs"].index((GB_RATE, GB_DEPTH, 1.0))

    def rows(index, names):
        return {n: dict(lowest=_spread(r["lowest"][i]),
                        lowest_at=float(r["lowest_at"][i].mean()),
                        end=float(r["end"][i].mean()),
                        stop=float(r["stop"][i].mean()),
                        stopped_at=float(r["stopped_at"][i].mean()),
                        at_stop=_spread(r["at_stop"][i])) for n, i in zip(names, index)}
    rate_names = ["rate %g" % v for v in GB_RATES]
    sub_names = ["%g%% of the rows" % (100 * v) for v in GB_SUBSAMPLES]
    depth_names = ["depth %d" % v for v in GB_DEPTHS]
    half = GB_SUBSAMPLES.index(0.5)
    meta = _base()
    meta.update(
        source="NumPy gradient boosting for the log-loss on MNIST",
        model="gradient boosting for the log-loss as Friedman gave it: every round a "
              "tree is fitted by least squares to the pulls y - p, each leaf gets the "
              "value sum(y - p) / sum(p (1 - p)), and the learning rate times it is "
              "added to the scores; %d rounds" % GB_ROUNDS,
        one=dict(after_trees=GB_ONE_AT, events_drawn=2 * GB_ONE_SHOWN, **GB_ONE,
                 train_loss=_floats(d["one_loss"][0]), held_loss=_floats(d["one_loss"][1]),
                 mean_pull=_floats(d["one_mean_pull"]), tree=_tree_meta(tree)),
        rates=rows(r["rates"], rate_names), subsamples=rows(r["subs"], sub_names),
        depths=rows(r["depths"], depth_names),
        base=dict(rate=GB_RATE, depth=GB_DEPTH),
        half_rows_gain=_spread(r["lowest"][r["subs"][0]] - r["lowest"][r["subs"][half]]),
        early_stopping=dict(
            patience=GB_PATIENCE, rate=GB_RATE,
            rule="stop when %d rounds in a row have not lowered the validation loss; "
                 "keep the trees up to its lowest point" % GB_PATIENCE,
            seed0=dict(chosen=int(r["stop"][main][0]), stopped_at=int(r["stopped_at"][main][0]),
                       held_there=float(r["at_stop"][main][0]),
                       held_lowest=float(r["lowest"][main][0]),
                       held_lowest_at=int(r["lowest_at"][main][0]),
                       held_end=float(r["end"][main][0]))),
        limitations=[
            "Digits stand in for events; 16 coarse cells, %d training events."
            % (2 * TRAIN),
            "Every run stops at %d rounds: rate %g is close to its lowest point there "
            "but was still within reach of more." % (GB_ROUNDS, GB_RATES[-1]),
            "Depth, rows and rate are varied one at a time from rate %g, depth %d, all "
            "rows: no joint search." % (GB_RATE, GB_DEPTH),
        ],
        designs_tried=[
            "A first version gave each leaf the mean of the pulls (a plain gradient "
            "step). Its steps are about four times smaller than the Newton value used "
            "here, since p (1 - p) is at most 1/4, so a learning rate of 1 was not "
            "large: Friedman's leaf value was used instead and the rates kept.",
            "The rates, depths, row shares, rounds and patience were fixed before the "
            "runs reported here.",
        ])
    return Study(
        "Gradient boosting for classification: the settings",
        "Trees fitted to the gradient of the log-loss: one round drawn, then the "
        "learning rate, the share of rows per tree, the depth and early stopping",
        (
            Panel("One round: scores, pulls, a tree, the step",
                  "xy: the score before the round and the label of %d training events "
                  "of seed 0 after %d trees. after: their scores after it; leaf: the "
                  "leaf each reaches; tree_*: the tree fitted to the pulls y - p."
                  % (2 * GB_ONE_SHOWN, GB_ONE_AT), "points",
                  dict(xy=np.column_stack([d["one_shown_before"], d["one_shown_y"]]),
                       classes=d["one_shown_y"], after=d["one_shown_after"],
                       leaf=d["one_shown_leaf"], loss=d["one_loss"],
                       mean_pull=d["one_mean_pull"],
                       **{"tree_" + k: v for k, v in tree.items()})),
            Panel("Held-out loss against trees, four learning rates",
                  "ys: held-out log-loss after x trees, mean of %d seeds, depth %d. "
                  "per_seed has the seeds." % (len(SEEDS), GB_DEPTH), "curve",
                  dict(x=rounds, ys=held[r["rates"]], labels=rate_names,
                       per_seed=r["held"][r["rates"]], lowest=r["lowest"][r["rates"]],
                       lowest_at=r["lowest_at"][r["rates"]])),
            Panel("Rows per tree, and depth",
                  "values: the lowest held-out loss on the way, mean of %d seeds, for "
                  "each share of the rows and each depth at rate %g. per_seed has the "
                  "seeds; curves the mean loss against trees."
                  % (len(SEEDS), GB_RATE), "bars",
                  dict(values=np.r_[r["lowest"][r["subs"]].mean(1),
                                    r["lowest"][r["depths"]].mean(1)],
                       labels=sub_names + depth_names,
                       per_seed=np.concatenate([r["lowest"][r["subs"]],
                                                r["lowest"][r["depths"]]]),
                       curves=held[r["subs"] + r["depths"]])),
            Panel("Early stopping on a validation set",
                  "ys: training, validation and held-out loss of seed 0 at rate %g "
                  "against trees. stop[rate, seed]: the round kept; stopped_at: when "
                  "training stopped; at_stop, lowest, end: held-out loss there, at its "
                  "lowest and after %d trees." % (GB_RATE, GB_ROUNDS), "curve",
                  dict(x=rounds, ys=d["loss"][main, 0], labels=["training", "validation",
                                                                "held-out"],
                       stop=r["stop"][r["rates"]], stopped_at=r["stopped_at"][r["rates"]],
                       at_stop=r["at_stop"][r["rates"]], lowest=r["lowest"][r["rates"]],
                       end=r["end"][r["rates"]])),
        ), meta)


def gain_study():
    d = gain_data()
    o = _sub(d, "one_")
    F, p, g, h = o["one"]
    G, H = float(o["hand_g"].sum()), float(o["hand_h"].sum())
    best = int(o["hand_gain"].argmax())
    _, node_f, node_n, Gk, Hk = o["node"]
    sweep = d["sweep"]
    ng, nl = len(GAIN_GAMMAS), len(GAIN_LAMBDAS)
    parts = dict(gamma=(GAIN_GAMMAS, sweep[:ng]), lam=(GAIN_LAMBDAS, sweep[ng:ng + nl]),
                 mcw=(GAIN_MCWS, sweep[ng + nl:]))
    cmp_, at = d["compare"], d["compare_at"]
    pick = np.take_along_axis(cmp_, at[..., None], 2)[..., 0]
    names = ["plain gradient boosting", "second order, lambda 0", "second order, lambda 1"]
    differ = d["differ"].sum(0)                        # round, (nodes, differing)
    share = differ[:, 1] / np.maximum(differ[:, 0], 1)
    nb = [int(np.argmax(row)) for row in o["node_gain"]]
    where, split, score, sides = o["dis_where"], o["dis_split"], o["dis_score"], o["dis_sides"]
    meta = _base()
    meta.update(
        source="The split gain of XGBoost's algorithm, written out in NumPy, on MNIST",
        own=OWN, model="second-order boosting for the log-loss: g = p - y, "
                       "h = p (1 - p); leaf value -G / (H + lambda); gain "
                       "(GL^2/(HL+lambda) + GR^2/(HR+lambda) - G^2/(H+lambda)) / 2 - gamma",
        state=dict(seed=0, rounds_done=STATE_AT, **STATE),
        one_event=dict(score=float(F), p=float(p), g=float(g), h=float(h), label=1),
        handful=dict(events=2 * GAIN_HAND, cell=cell_name(o["feature"]), lam=GAIN_LAM,
                     x=_floats(o["hand_x"]), g=_floats(o["hand_g"]), h=_floats(o["hand_h"]),
                     G=G, H=H, value=float(leaf_value(G, H, GAIN_LAM)),
                     score=float(-0.5 * leaf_score(G, H, GAIN_LAM)),
                     exact_minimum=float(o["hand_exact"].min()),
                     exact_minimum_at=float(GAIN_W[o["hand_exact"].argmin()]),
                     gain=_floats(o["hand_gain"]), best=best,
                     GL=float(o["hand_GL"][best]), HL=float(o["hand_HL"][best]),
                     GR=float(G - o["hand_GL"][best]), HR=float(H - o["hand_HL"][best])),
        node=dict(events=int(node_n), cell=cell_name(node_f), G=float(Gk), H=float(Hk),
                  lams=list(GAIN_LAMS), best_gain=[float(row.max()) for row in o["node_gain"]],
                  best_at=[float(o["node_x"][k]) for k in nb],
                  left_value=[float(leaf_value(o["node_GL"][k], o["node_HL"][k], lam))
                              for k, lam in zip(nb, GAIN_LAMS)],
                  right_value=[float(leaf_value(Gk - o["node_GL"][k], Hk - o["node_HL"][k],
                                                lam)) for k, lam in zip(nb, GAIN_LAMS)],
                  mcw=GAIN_NODE_MCW, gamma=GAIN_NODE_GAMMA,
                  allowed_by_mcw=int(((o["node_HL"] >= GAIN_NODE_MCW)
                                      & (Hk - o["node_HL"] >= GAIN_NODE_MCW)).sum()),
                  thresholds=len(o["node_x"])),
        sweeps={k: dict(values=list(v), held=_floats(a[..., 0].mean(1)),
                        held_end=_floats(a[..., 1].mean(1)),
                        leaves=_floats(a[..., 2].mean(1)), **GAIN_SWEEP)
                for k, (v, a) in parts.items()},
        disagree=dict(round=int(where[0]), events=int(where[2]),
                      second=dict(cell=cell_name(split[0, 0]), threshold=float(split[0, 1])),
                      plain=dict(cell=cell_name(split[1, 0]), threshold=float(split[1, 1])),
                      second_scores=_floats(score[0]), plain_scores=_floats(score[1]),
                      mean_pull=_floats(-sides[0, :, 0] / sides[0, :, 2]),
                      newton=_floats(-sides[0, :, 0] / sides[0, :, 1]),
                      share_by_round=dict(first=float(share[0]),
                                          rounds_3_to_10=float(differ[2:10, 1].sum()
                                                               / differ[2:10, 0].sum()),
                                          rounds_31_to_100=float(differ[30:, 1].sum()
                                                                 / differ[30:, 0].sum()))),
        compare=dict(names=names, rounds=GAIN_COMPARE_ROUNDS, **GAIN_COMPARE,
                     held={n: _spread(v) for n, v in zip(names, pick)},
                     plain_minus_second=_spread(pick[0] - pick[1]),
                     plain_minus_lambda_1=_spread(pick[0] - pick[2]),
                     trees={n: float(v) for n, v in zip(names, at.mean(1))}),
        limitations=[
            "Digits stand in for events. Plain gradient boosting here already gives "
            "each leaf the Newton value -G / H (Friedman): what differs is the split "
            "choice and lambda.",
            "The worked example has %d events so that the sums can be followed; with so "
            "few, lambda = %g is as large as their summed curvature."
            % (2 * GAIN_HAND, GAIN_LAM),
            "The held-out losses are compared at the round the validation loss picks.",
        ],
        designs_tried=[
            "The handful was first laid out along the cell the root splits on (C4), in "
            "which most readings are zero and no threshold can slide: cell %s is drawn "
            "instead." % cell_name(o["feature"]),
            "min_child_weight 5 was first drawn on the small node: its summed curvature "
            "is %.1f, so no split at all was allowed. 1 is drawn." % Hk,
            "gamma was first swept over 0, 0.5, 1, 2, 5: every value above 0 made the "
            "held-out loss worse, so smaller values were added (0.1, 0.2).",
        ])
    w = GAIN_W
    return Study(
        "How XGBoost scores a split",
        "The slope g and curvature h of every event's loss, the value of a leaf and "
        "the gain of a split built from their sums: " + OWN,
        (
            Panel("One event's loss, its tangent and its parabola",
                  "ys: the change in the log-loss of one training event (a 4, p = "
                  "%.2f) when x is added to its score; g x; g x + h x^2 / 2." % p,
                  "curve", dict(x=w, ys=np.array([o["one_curve"] - event_loss(F, 1.0),
                                                 g * w, g * w + 0.5 * h * w * w]),
                                labels=["log-loss", "slope only", "slope and curvature"],
                                event=o["one"])),
            Panel("A handful of events: one leaf, then a split",
                  "xy: cell %s and g of %d training events of seed 0 after %d rounds, "
                  "in the order of the cell; h, score, classes: theirs. exact: the "
                  "true change of their summed loss (plus lambda w^2 / 2) when w is "
                  "added to every score, over w; approx: G w + (H + lambda) w^2 / 2. "
                  "gain: of a split after each event, with GL and HL."
                  % (cell_name(o["feature"]), 2 * GAIN_HAND, STATE_AT), "points",
                  dict(xy=np.column_stack([o["hand_x"], o["hand_g"]]), classes=o["hand_y"],
                       h=o["hand_h"], score=o["hand_score"], w=w, exact=o["hand_exact"],
                       approx=approx(G, H, GAIN_LAM, w), gain=o["hand_gain"],
                       GL=o["hand_GL"], HL=o["hand_HL"])),
            Panel("A small real node under three lambdas",
                  "ys: the gain of a split of a node of %d events at each threshold x "
                  "of cell %s, for lambda 0, 1 and 10. GL, HL: the sums on the left."
                  % (node_n, cell_name(node_f)), "curve",
                  dict(x=o["node_x"], ys=o["node_gain"],
                       labels=["lambda %g" % v for v in GAIN_LAMS], GL=o["node_GL"],
                       HL=o["node_HL"], node=o["node"], sweep=sweep)),
            Panel("Second order against plain gradient boosting",
                  "ys: held-out loss against trees, mean of %d seeds. at: the round "
                  "validation picks, per seed; picked: the loss there. differ: per "
                  "round, the nodes split and those where the two criteria disagree."
                  % len(SEEDS), "curve",
                  dict(x=np.arange(GAIN_COMPARE_ROUNDS + 1), ys=cmp_.mean(1), labels=names,
                       at=at, picked=pick, differ=d["differ"], share=share,
                       dis_score=score, dis_sides=sides)),
        ), meta)


def hist_study():
    d = hist_data()
    o = _sub(d, "one_")
    miss = d["missing"]                                # rule, way, seed, (loss, round, ..)
    loss = miss[..., 0]
    left_share = miss[:, 0, :, 2].sum(1) / miss[:, 0, :, 3].sum(1)
    by_bins, timing = d["by_bins"][..., 0], d["timing"]
    growth = d["growth"][..., 0]
    names = ["%d bins" % b for b in HIST_BINS] + ["exact"]
    k32 = HIST_BINS.index(BINS)
    mg = o["miss_gain"]
    mc = int(o["miss_cut"])
    sub_ok = bool(np.allclose(o["sub_parent"] - o["sub_left"], o["sub_right"]))
    level, leaf = _sub(o, "level_"), _sub(o, "leaf_")
    meta = _base()
    meta.update(
        source="Histogram split finding of XGBoost's algorithm, written out in NumPy, "
               "on MNIST", own=OWN,
        model="second-order boosting (%s) with quantile bins; missing readings kept in "
              "a slot of their own and sent to the side with the larger gain"
              % ", ".join("%s %g" % kv for kv in HIST_SET.items()),
        state=dict(seed=0, rounds_done=STATE_AT, **STATE),
        drawn=dict(cell=cell_name(o["feature"]), bins=HIST_SHOW_BINS,
                   sorted_events=HIST_SORTED, dropped_events=HIST_DROPPED,
                   distinct_readings={cell_name(j): int(v)
                                      for j, v in enumerate(o["distinct"])},
                   scan_gain=_floats(o["drop_gain"]), scan_best=int(o["drop_gain"].argmax()),
                   lam=GAIN_LAM),
        subtraction=dict(parent=int(o["sub_n"][0]), left=int(o["sub_n"][1]),
                         right=int(o["sub_n"][2]), root_cell=cell_name(o["root"][0]),
                         root_threshold=float(o["root"][1]), holds_exactly=sub_ok),
        missing=dict(
            rules=dict(random="every reading removed with probability %g" % HIST_RANDOM,
                       floor="readings below %g are not reported" % HIST_FLOOR),
            ways=list(HIST_WAYS), rounds=HIST_ROUNDS,
            held={r: {w: _spread(loss[i, j]) for j, w in enumerate(HIST_WAYS)}
                  for i, r in enumerate(HIST_RULES)},
            sent_left={r: float(v) for r, v in zip(HIST_RULES, left_share)},
            complete=_spread(d["complete"][:, 0]),
            drawn=dict(missing=int(o["miss_hist"][2, -1]), threshold=float(o["miss_edges"][mc]),
                       gain_left=float(mg[mc, 0]), gain_right=float(mg[mc, 1]),
                       sent_left=bool(o["miss_left"]))),
        growth=dict(leaves=HIST_LEAVES, level=_spread(growth[0]), leaf=_spread(growth[1]),
                    level_minus_leaf=_spread(growth[0] - growth[1]),
                    leaf_depth=int(leaf["level"].max()), level_depth=int(level["level"].max())),
        by_bins={n: _spread(v) for n, v in zip(names, by_bins)},
        timing=dict(
            note=EVIDENCE["timing"], rounds=HIST_TIMED_ROUNDS, repeats=HIST_TIMED_REPEATS,
            ms_per_tree={n: float(1000 * v) for n, v in zip(names, timing[:, 0])},
            speedup_at_32=float(timing[-1, 0] / timing[k32, 0]),
            added_per_tree={n: float(v) for n, v in zip(names, timing[:, 1])},
            added_without_subtraction=float(timing[k32, 2]),
            sorted_per_tree=float(timing[-1, 3]),
            scored_per_tree={n: float(v) for n, v in zip(names, timing[:, 4])}),
        limitations=[
            "Digits stand in for events. The timings are of this numpy code, where a "
            "histogram is one call and a sort another: a compiled library's ratio "
            "will differ, and all of them vary by machine. The counts do not.",
            "2,000 training events: sorting is cheap at this size, and the histogram "
            "method's advantage grows with the number of events.",
            "Each split's default direction is learnt from the training events of its "
            "node only.",
        ],
        designs_tried=[
            "The drawings first used the cell the root splits on (C4), where most "
            "readings are zero and eight quantile bins collapse into three: cell %s "
            "is drawn instead." % cell_name(o["feature"]),
            "The two rules for removing readings, the four ways of handling them and "
            "the numbers of bins were fixed before the runs.",
        ])
    return Study(
        "XGBoost's histograms and missing values",
        "Split finding from bin totals, the subtraction trick, a learnt side for "
        "missing readings and two ways to grow a tree: " + OWN,
        (
            Panel("Events dropped into bins, and the scan",
                  "values: the sum of g in each of %d bins of cell %s for %d training "
                  "events; H, N: the sums of h and the counts; gain: of a split at each "
                  "bin boundary; drop_*: the events in the order they fall."
                  % (HIST_SHOW_BINS, cell_name(o["feature"]), HIST_DROPPED), "bars",
                  dict(values=o["drop_hist"][0], labels=["bin %d" % (k + 1)
                                                         for k in range(HIST_SHOW_BINS)],
                       H=o["drop_hist"][1], N=o["drop_hist"][2], gain=o["drop_gain"],
                       edges=o["edges8"], drop_x=o["drop_x"], drop_y=o["drop_y"],
                       drop_g=o["drop_g"], drop_h=o["drop_h"], drop_bin=o["drop_bin"],
                       few_x=o["few_x"], few_y=o["few_y"], few_g=o["few_g"],
                       sub_parent=o["sub_parent"], sub_left=o["sub_left"],
                       sub_right=o["sub_right"], sub_n=o["sub_n"])),
            Panel("Missing readings: four ways to handle them",
                  "values: held-out loss, mean of %d seeds, for each rule (%s) and way "
                  "(%s). per_seed has the seeds; miss_*: one split with holes."
                  % (len(SEEDS), ", ".join(HIST_RULES), ", ".join(HIST_WAYS)), "bars",
                  dict(values=loss.mean(2).ravel(),
                       labels=["%s, %s" % (r, w) for r in HIST_RULES for w in HIST_WAYS],
                       per_seed=loss.reshape(-1, len(SEEDS)), sent_left=left_share,
                       miss_hist=o["miss_hist"],
                       miss_gain=np.where(np.isfinite(mg), mg, 0.0), miss_edges=o["miss_edges"],
                       miss_sample=np.nan_to_num(o["miss_sample"], nan=-1.0))),
            Panel("Held-out loss and time against the number of bins",
                  "ys: held-out loss (mean of %d seeds) and milliseconds of split "
                  "finding per tree (seed 0, this machine) for x bins. exact: the "
                  "same two for the exact method; timing: every column per tree."
                  % len(SEEDS), "curve",
                  dict(x=np.array(HIST_BINS, float),
                       ys=np.array([by_bins[:-1].mean(1), 1000 * timing[:-1, 0]]),
                       labels=["held-out loss", "ms per tree"],
                       exact=np.array([by_bins[-1].mean(), 1000 * timing[-1, 0]]),
                       per_seed=by_bins, timing=timing)),
            Panel("Level-wise against leaf-wise growth",
                  "values: held-out loss with %d leaves a tree, mean of %d seeds. "
                  "level_*, leaf_*: one tree grown each way from the same events."
                  % (HIST_LEAVES, len(SEEDS)), "bars",
                  dict(values=growth.mean(1), labels=["level-wise", "leaf-wise"],
                       per_seed=growth,
                       **{"level_" + k: np.nan_to_num(v) for k, v in level.items()},
                       **{"leaf_" + k: np.nan_to_num(v) for k, v in leaf.items()})),
        ), meta)


KNOB_NAMES = dict(rate="learning rate", depth="max depth", lam="lambda", gamma="gamma",
                  mcw="min child weight", subsample="subsample", colsample="colsample")
KNOB_HOLDS = dict(rate="the size of every step", depth="cells combined by one tree",
                  lam="the size of leaf values", gamma="splits that gain little",
                  mcw="leaves with few events", subsample="reliance on single events",
                  colsample="reliance on single cells")


def round_study():
    d = round_data()
    o = _sub(d, "one_")
    tree = _sub(o, "tree_")
    many = d["many"].mean(1)                           # sampled?, set, round
    knobs = d["knobs"]                                 # knob, seed, (end, chosen, ...)
    riv = d["rivals"]                                  # seed, method, (error, auc)
    table = o["table"]
    pruned = [int(k) for k in np.flatnonzero((tree["left"] >= 0) & ~o["kept"])]
    knob_rows = {}
    for i, (key, value) in enumerate(KNOBS, start=1):
        knob_rows[KNOB_NAMES[key]] = dict(
            value=value, loose=LOOSE[key], restrains=KNOB_HOLDS[key],
            end=_spread(knobs[i, :, 0]), stopped=_spread(knobs[i, :, 1]),
            leaves=float(knobs[i, :, 3].mean()))
    gap = riv[:, 3] - riv[:, 4]
    meta = _base()
    meta.update(
        source="One round of XGBoost's algorithm, written out in NumPy, on MNIST", own=OWN,
        model="second-order boosting for the log-loss with histograms over quantile "
              "bins, pruning with gamma, shrinkage, row and column subsampling",
        round=dict(seed=0, **ROUND, events_shown=2 * ROUND_SHOWN,
                   root_cell=cell_name(tree["feature"][0]),
                   root_threshold=float(tree["threshold"][0]),
                   root_gain=float(tree["gain"][0]), pruned_nodes=pruned,
                   pruned_gain=[float(tree["gain"][k]) for k in pruned],
                   leaves_before=int((tree["left"] < 0).sum()),
                   leaves_after=int((tree["left"] < 0).sum()) - len(pruned),
                   train_loss=_floats(o["loss"][:, 0]), held_loss=_floats(o["loss"][:, 1]),
                   candidates=int(sum(len(e) for e in prep(0)["edges"])),
                   tree=_tree_meta(tree)),
        many=dict(rounds=ROUND_MANY_ROUNDS, **ROUND_MANY, sampled=dict(ROUND_SAMPLED),
                  held_lowest=dict(all=float(many[0, 2].min()),
                                   sampled=float(many[1, 2].min())),
                  held_lowest_at=dict(all=int(many[0, 2].argmin()),
                                      sampled=int(many[1, 2].argmin())),
                  train_end=dict(all=float(many[0, 0, -1]), sampled=float(many[1, 0, -1])),
                  held_end=dict(all=float(many[0, 2, -1]), sampled=float(many[1, 2, -1]))),
        loose=dict(rounds=LOOSE_ROUNDS, **LOOSE, end=_spread(knobs[0, :, 0]),
                   stopped=_spread(knobs[0, :, 1]), leaves=float(knobs[0, :, 3].mean())),
        knobs=knob_rows,
        rivals={n: dict(error=_spread(riv[:, i, 0]), auc=_spread(riv[:, i, 1]),
                        picked=[int(v) for v in d["picked"][:, i]])
                for i, n in enumerate(RIVALS)},
        rival_settings=dict(
            tree="one tree, least squares on the labels (the Gini rule), at least %d "
                 "events a leaf, depth 1 to %d chosen on validation error"
                 % (TREE_MIN_LEAF, TREE_DEPTHS[-1]),
            forest="%d trees of depth %d, each on a bootstrap sample, %d of the 16 "
                   "cells tried at every node, averaged" % (FOREST["trees"],
                                                            FOREST["depth"],
                                                            FOREST["tried"]),
            adaboost="stumps, up to %d, the number chosen on validation error"
                     % ADA_ROUNDS,
            gradient_boosting=dict(RIVAL_GB, rounds=RIVAL_ROUNDS,
                                   picked="the round with the lowest validation loss"),
            xgboost_style=dict(RIVAL_XGB, rounds=RIVAL_ROUNDS,
                               picked="the round with the lowest validation loss")),
        boosting_gap=dict(gradient_minus_xgboost_error=_spread(gap[:, 0]),
                          xgboost_minus_gradient_auc=_spread(-gap[:, 1])),
        limitations=[
            "Digits stand in for events; 16 coarse cells and %d training events, on "
            "which every method is far from the error a network reaches on whole "
            "images." % (2 * TRAIN),
            "The forest and AdaBoost are simple versions written here; none of the "
            "five was searched over more than the one setting named.",
            "The knobs are turned one at a time from a deliberately loose setting; "
            "together they interact.",
        ],
        designs_tried=[
            "The loose setting first had learning rate 1: with lambda 0 its leaf "
            "values exploded (held-out loss above 10^13 for some knobs), so rate 0.3 "
            "is the loose value and 0.03 the restrained one.",
            "gamma = 1 pruned nothing in the round that is drawn (its weakest split "
            "gains %.1f), so the drawn round uses gamma = %g."
            % (min(float(tree["gain"][k]) for k in np.flatnonzero(tree["left"] >= 0)),
               ROUND["gamma"]),
            "The five methods' settings were fixed before their held-out numbers were "
            "looked at.",
        ])
    order = np.r_[0, np.arange(1, len(KNOBS) + 1)]
    return Study(
        "One round of XGBoost, end to end",
        "From the current predictions to g and h, histograms, the best split, a "
        "pruned tree, its leaf weights and the new predictions; then many rounds, the "
        "knobs, and four other tree methods: " + OWN,
        (
            Panel("Eight events through two rounds",
                  "values: %d training events of seed 0. table[k]: their score, p, g, "
                  "h, the leaf weight they receive and the leaf, at the start of round "
                  "k + 1; root_*: the root's histograms of the cell it splits on; "
                  "tree_*: round one's tree before pruning; kept: the splits left "
                  "after pruning." % (2 * ROUND_SHOWN), "images",
                  dict(values=d["one_images"], labels=[str(SIGNAL if v > 0.5 else BACKGROUND)
                                                       for v in o["shown_y"]],
                       table=table, loss=o["loss"], root_hist=o["root_hist"],
                       root_edges=o["root_edges"],
                       root_gain=np.where(np.isfinite(o["root_gain"]), o["root_gain"], 0.0),
                       best_by_feature=np.where(np.isfinite(o["best_by_feature"]),
                                                o["best_by_feature"], 0.0),
                       kept=o["kept"],
                       **{"tree_" + k: np.nan_to_num(v) for k, v in tree.items()})),
            Panel("Many rounds, with and without subsampling",
                  "ys: training and held-out loss against rounds, mean of %d seeds, "
                  "with every row and cell and with %g of the rows and %g of the cells "
                  "per tree." % (len(SEEDS), ROUND_SAMPLED["subsample"],
                                 ROUND_SAMPLED["colsample"]), "curve",
                  dict(x=np.arange(ROUND_MANY_ROUNDS + 1),
                       ys=np.array([many[0, 0], many[0, 2], many[1, 0], many[1, 2]]),
                       labels=["training, all", "held-out, all", "training, sampled",
                               "held-out, sampled"], per_seed=d["many"][:, :, 2])),
            Panel("The knobs, one at a time",
                  "values: held-out loss after %d rounds, mean of %d seeds, for the "
                  "loose setting and with each knob turned alone. stopped: at the "
                  "round validation picks." % (LOOSE_ROUNDS, len(SEEDS)), "bars",
                  dict(values=knobs[order, :, 0].mean(1),
                       labels=["loose"] + [KNOB_NAMES[k] for k, _ in KNOBS],
                       stopped=knobs[order, :, 1].mean(1), per_seed=knobs[order, :, 0],
                       leaves=knobs[order, :, 3].mean(1))),
            Panel("Five ways to use trees",
                  "values: held-out error, mean of %d seeds, same events for all. auc: "
                  "ROC area; per_seed, auc_per_seed: the seeds." % len(SEEDS), "bars",
                  dict(values=riv[:, :, 0].mean(0), labels=list(RIVALS),
                       auc=riv[:, :, 1].mean(0), per_seed=riv[:, :, 0].T,
                       auc_per_seed=riv[:, :, 1].T, picked=d["picked"])),
        ), meta)


STUDIES = {
    "adaboost": adaboost_study,
    "boosting-shrinkage-and-early-stopping": shrinkage_study,
    "xgboost-split-gain": gain_study,
    "xgboost-histograms-and-missing-values": hist_study,
    "xgboost-one-round": round_study,
}

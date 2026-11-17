"""The experiments behind animations 160-164: one decision tree, why it is not enough, and
what averaging many does (see :func:`voxanim.next_catalog.redesigned`).

No physics dataset is used. Real handwritten digits (MNIST) stand in for events: the 4s
for signal, the 9s for background, two digits people write alike. Every event is read
two ways (:func:`events`):

* two features that can be drawn as a plane: the ink at the top centre of the picture
  (:func:`top_ink`: a 9 is closed there, a 4 is open) and how far the ink is spread up
  and down the picture (:func:`spread`), in pixels;
* 49 cells, the mean of each 4 x 4 square of pixels, as a detector reads an event.

The five experiments:

* ``tree-splits-and-impurity``: one node; every threshold of both features scored by
  the impurity it removes (Gini, entropy, misclassification error), on twelve events
  and on 2,000; whole trees grown with each criterion on the 49 cells.
* ``tree-pruning-and-depth``: a tree grown until nothing can be split, then held back
  by depth, by events per leaf, by gain, and by cost-complexity pruning; the setting
  chosen by cross-validation.
* ``tree-instability``: the same recipe on resamples of the same events: how often two
  such trees disagree, and the squared error split into the error of the average tree
  and the variance around it, against depth.
* ``bagging``: bootstrap resamples, a tree on each, their vote; the variance of the
  vote against the number of trees and the formula with the measured correlation; the
  out-of-bag error against the held-out one.
* ``random-forest``: bagging with a random handful of cells tried at each split:
  correlation between trees, error of single trees and of the forest against the size
  of the handful; importance of each cell by total gain and by permutation.

Trees are plain numpy (:func:`grow`, :func:`leaves_of`, :func:`pruning`), from
``numpy.random.default_rng`` generators whose seeds are recorded. Training and held-out
events are disjoint and every error reported is a held-out one unless it says
otherwise.
"""

from __future__ import annotations

import functools
import hashlib
import inspect

import numpy as np

from . import mnist
from .cache import cached_npz
from .study import Panel, Study
from .study_phys_apps import cells

VERSION = 1
STAND_IN = ("No physics dataset is used: real handwritten digits (MNIST) stand in for "
            "events, the 4s for signal and the 9s for background.")
COUNTERPART = ("An analysis separates signal events from background events with trees "
               "grown on detector variables in the same way.")
EVIDENCE = dict(
    exact="exact arithmetic on the events drawn: counts, impurities and gains worked "
          "out from them",
    measured="measured on MNIST: trees grown here in numpy, errors from held-out digits",
)


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them (an edit to one experiment re-runs that experiment only)."""
    h = hashlib.sha256()
    for fn in (*CORE, *sources):
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-trees-" + name, settings,
                                                    build).items()}


def _parallel(fn, jobs):
    from .study_generative import parallel

    return parallel(fn, jobs)


def _floats(v):
    return [float(x) for x in np.asarray(v).ravel()]


# ====================================================================== the events
SIGNAL, BACKGROUND = 4, 9
SPLIT = 160                       # seed of the division into pool and held-out events
POOL = 4000                       # events of each class that training may draw from
BLOCK = 4                         # cells are 4 x 4 pixels: 7 x 7 = 49 readings an event
SIDE = 28 // BLOCK
CELLS = SIDE * SIDE
TOP_ROWS, TOP_COLS = (5, 10), (10, 18)            # the patch :func:`top_ink` averages
FEATURES = ("ink at the top", "vertical spread")
CELL_BINS = 32                    # a cell's reading is cut at 31 evenly spaced thresholds
CELL_CUTS = np.tile(np.arange(1, CELL_BINS) / CELL_BINS, (CELLS, 1))


def top_ink(images):
    """The mean ink of rows 5-9, columns 10-17 of each 28 x 28 image, between 0 and 1:
    the top centre of the picture, where a 9 is closed and a 4 is open."""
    img = np.asarray(images, float).reshape(-1, 28, 28)
    return img[:, TOP_ROWS[0]:TOP_ROWS[1], TOP_COLS[0]:TOP_COLS[1]].mean((1, 2))


def spread(images):
    """How far the ink of each image is spread up and down, in pixels: the
    root-mean-square distance of its ink from the ink's mean row, every pixel weighted
    by its ink (the square root of the second moment about the centre)."""
    img = np.asarray(images, float).reshape(-1, 28, 28)
    rows = np.arange(28.0)[None, :, None]
    mass = np.maximum(img.sum((1, 2)), 1e-12)
    mean = (img * rows).sum((1, 2)) / mass
    return np.sqrt((img * (rows - mean[:, None, None]) ** 2).sum((1, 2)) / mass)


def two_features(images):
    """``(n, 2)``: :func:`top_ink` and :func:`spread`."""
    return np.column_stack([top_ink(images), spread(images)])


@functools.lru_cache(maxsize=1)
def events():
    """The 4s (label 1, signal) and 9s (label 0, background). ``pool_*``: ``POOL`` of
    each from the training file, signal first, for training and validation. ``held_*``:
    every other 4 and 9 of the training file and all of the test file. For each: ``two``
    (the two features), ``cells`` (the 49 cells), ``y`` and ``index`` (where the image
    is in its file; test images are numbered from 60,000)."""
    xtr, ytr, _, yte = mnist.load()
    rng = np.random.default_rng([SPLIT, 0])
    pool, held = [], []
    for digit in (SIGNAL, BACKGROUND):
        ids = rng.permutation(np.flatnonzero(ytr == digit))
        pool.append(ids[:POOL])
        held.append(np.r_[ids[POOL:], len(xtr) + np.flatnonzero(yte == digit)])
    out = {}
    for name, parts in (("pool", pool), ("held", held)):
        index = np.concatenate(parts)
        img = picture(index)
        out[name + "_index"] = index
        out[name + "_two"] = two_features(img)
        out[name + "_cells"] = cells(img, BLOCK)
        out[name + "_y"] = np.r_[np.ones(len(parts[0]), np.int64),
                                 np.zeros(len(parts[1]), np.int64)]
    return out


def picture(index):
    """The 28 x 28 images at ``index`` (as :func:`events` numbers them)."""
    xtr, _, xte, _ = mnist.load()
    index = np.atleast_1d(index)
    out = np.where((index < len(xtr))[:, None], xtr[np.minimum(index, len(xtr) - 1)],
                   xte[np.maximum(index - len(xtr), 0)])
    return out.reshape(-1, 28, 28)


def draw(n, seed, extra=0):
    """Seed ``seed``'s training sample: the rows of the pool of ``n`` events of each
    class (signal first), and of ``extra`` more of each, never the same events."""
    rng = np.random.default_rng([SPLIT, 1, seed])
    order = [k * POOL + rng.permutation(POOL) for k in range(2)]
    return (np.concatenate([o[:n] for o in order]),
            np.concatenate([o[n:n + extra] for o in order]))


# ====================================================================== impurity
CRITERIA = ("gini", "entropy", "error")
CRITERION_NAMES = dict(gini="Gini", entropy="entropy", error="misclassification")


def impurity(p, criterion="gini"):
    """The impurity of a node whose share of signal is ``p``. Gini: 2p(1 - p), which
    is 1 - p^2 - (1 - p)^2, the chance that two events drawn from it differ. Entropy:
    -p log2 p - (1 - p) log2 (1 - p), in bits. Error: min(p, 1 - p), the share the
    majority answer gets wrong. All are 0 for a pure node and largest at p = 1/2."""
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    if criterion == "gini":
        return 2.0 * p * (1.0 - p)
    if criterion == "error":
        return np.minimum(p, 1.0 - p)
    q = 1.0 - p
    with np.errstate(divide="ignore", invalid="ignore"):
        return -(np.where(p > 0, p * np.log2(p), 0.0) + np.where(q > 0, q * np.log2(q), 0.0))


def split_gain(n, n1, left, left1, criterion="gini"):
    """What a split removes: the node's impurity minus the mean impurity of its two
    sides weighted by their share of the node's ``n`` events (``n1`` of them signal;
    ``left``, ``left1`` the same counts for the left side). 0 where a side is empty."""
    n, n1 = np.asarray(n, float), np.asarray(n1, float)
    left, left1 = np.asarray(left, float), np.asarray(left1, float)
    right, right1 = n - left, n1 - left1
    nn, ll, rr = np.maximum(n, 1e-12), np.maximum(left, 1e-12), np.maximum(right, 1e-12)
    g = (impurity(n1 / nn, criterion) - left / nn * impurity(left1 / ll, criterion)
         - right / nn * impurity(right1 / rr, criterion))
    return np.where((left > 0) & (right > 0), g, 0.0)


# ====================================================================== trees
PAD = 1e30                        # fills a feature's list of thresholds to full width


def midpoints(x, most=None):
    """Thresholds for each column of ``x``: halfway between each pair of neighbouring
    distinct values, so that every way of dividing the events along it is tried.
    ``(features, most)``, padded with ``PAD`` (which no value exceeds, so nothing is
    split there); with more distinct values than ``most + 1``, ``most`` of them evenly
    spaced through the list."""
    x = np.asarray(x, float)
    cols = []
    for f in range(x.shape[1]):
        v = np.unique(x[:, f])
        mid = (v[:-1] + v[1:]) / 2
        if most is not None and len(mid) > most:
            mid = mid[np.round(np.linspace(0, len(mid) - 1, most)).astype(int)]
        cols.append(mid)
    width = max(1, max(len(c) for c in cols))
    out = np.full((len(cols), width), PAD)
    for f, c in enumerate(cols):
        out[f, :len(c)] = c
    return out


def binned(x, cuts):
    """For each value, how many of its feature's thresholds ``cuts`` lie below it: an
    event goes left at threshold number ``c`` when this is at most ``c``, that is when
    its value is at most ``cuts[feature, c]``."""
    x = np.asarray(x, float)
    return np.column_stack([np.searchsorted(cuts[f], x[:, f], side="left")
                            for f in range(x.shape[1])]).astype(np.int64)


def grow(xb, y, bins, w=None, criterion="gini", max_depth=None, min_leaf=1, min_gain=0.0,
         mtry=None, rng=None):
    """A classification tree on binned features ``xb`` (:func:`binned`; values below
    ``bins``) and labels ``y`` (1 signal, 0 background), each event counted ``w`` times
    (a bootstrap resample is its counts).

    Level by level, every node that is not pure takes the feature and threshold with
    the largest :func:`split_gain` (ties: the first feature, the lowest threshold),
    provided both sides keep ``min_leaf`` events and the gain times the node's share
    of all events is at least ``min_gain``; a node whose best gain is zero stays a
    leaf. ``mtry``: at each node only that many features, drawn afresh from ``rng``,
    are tried (a random forest); if none of them can split the node, features are
    added in the drawn order until one can.

    Returns the node table: ``feature`` (-1 for a leaf), ``cut``, ``left``, ``right``
    (children always have larger numbers than their parent), ``depth``, ``n`` and
    ``n1`` (events and signal events in the node), ``gain`` (of its split) and
    ``tried`` (with ``mtry``: the features drawn, in order, -1 past the last tried)."""
    xb = np.asarray(xb, np.int64)
    n, nf = xb.shape
    y = np.asarray(y, float)
    w = np.ones(n) if w is None else np.asarray(w, float)
    keep = np.flatnonzero(w > 0)
    xb, y, w = xb[keep], y[keep], w[keep]
    total = float(w.sum())
    T = dict(feature=[-1], cut=[0], left=[-1], right=[-1], depth=[0], n=[total],
             n1=[float((w * y).sum())], gain=[0.0])
    tried = [np.full(nf, -1, np.int64)]
    node = np.zeros(len(xb), np.int64)
    rows = np.arange(len(xb)) if 0 < T["n1"][0] < total else np.arange(0)
    frontier = np.array([0])
    level = 0
    offsets = np.arange(nf) * bins
    while len(rows) and (max_depth is None or level < max_depth):
        k = len(frontier)
        local = np.full(len(T["feature"]), -1, np.int64)
        local[frontier] = np.arange(k)
        loc = local[node[rows]]
        idx = (loc[:, None] * (nf * bins) + offsets + xb[rows]).ravel()
        size = k * nf * bins
        h = np.bincount(idx, weights=np.repeat(w[rows], nf), minlength=size)
        h1 = np.bincount(idx, weights=np.repeat(w[rows] * y[rows], nf), minlength=size)
        h, h1 = h.reshape(k, nf, bins), h1.reshape(k, nf, bins)
        left = np.cumsum(h, 2)[:, :, :-1]
        left1 = np.cumsum(h1, 2)[:, :, :-1]
        n_all = np.array(T["n"])[frontier][:, None, None]
        n1_all = np.array(T["n1"])[frontier][:, None, None]
        gain = split_gain(n_all, n1_all, left, left1, criterion)
        ok = ((left >= min_leaf) & (n_all - left >= min_leaf) & (gain > 1e-12)
              & (gain * n_all / total >= min_gain))
        gain = np.where(ok, gain, -1.0)
        if mtry is not None and mtry < nf:
            order = np.argsort(rng.random((k, nf)), 1)
            can = np.take_along_axis(ok.any(2), order, 1)
            first = np.where(can.any(1), can.argmax(1) + 1, 0)
            limit = np.maximum(mtry, first)[:, None]
            allowed = np.zeros((k, nf), bool)
            np.put_along_axis(allowed, order, np.arange(nf)[None] < limit, 1)
            gain = np.where(allowed[:, :, None], gain, -1.0)
            drawn = np.where(np.arange(nf)[None] < limit, order, -1)
            for j, t in enumerate(frontier):
                tried[t] = drawn[j]
        flat = gain.reshape(k, -1)
        best = flat.argmax(1)
        children = np.full((len(T["feature"]), 2), -1, np.int64)
        for j, t in enumerate(frontier):
            if flat[j, best[j]] <= 0:
                continue
            f, c = divmod(int(best[j]), bins - 1)
            T["feature"][t], T["cut"][t], T["gain"][t] = f, c, float(flat[j, best[j]])
            sides = ((left[j, f, c], left1[j, f, c]),
                     (T["n"][t] - left[j, f, c], T["n1"][t] - left1[j, f, c]))
            for side, (count, count1) in enumerate(sides):
                new = len(T["feature"])
                children[t, side] = new
                T["left" if side == 0 else "right"][t] = new
                for key, v in (("feature", -1), ("cut", 0), ("left", -1), ("right", -1),
                               ("depth", level + 1), ("n", float(count)),
                               ("n1", float(count1)), ("gain", 0.0)):
                    T[key].append(v)
                tried.append(np.full(nf, -1, np.int64))
        at = node[rows]
        split = children[at, 0] >= 0
        rows, at = rows[split], at[split]
        feature, cut = np.array(T["feature"]), np.array(T["cut"])
        node[rows] = children[at, (xb[rows, feature[at]] > cut[at]).astype(int)]
        counts, counts1 = np.array(T["n"]), np.array(T["n1"])
        mixed = (counts1[node[rows]] > 0) & (counts1[node[rows]] < counts[node[rows]])
        rows = rows[mixed]
        frontier = np.unique(node[rows])
        level += 1
    out = {k: np.array(v, np.int64 if k in ("feature", "cut", "left", "right", "depth")
                       else float) for k, v in T.items()}
    out["tried"] = np.array(tried)
    return out


def _asks(tree, internal=None, max_depth=None):
    asks = tree["feature"] >= 0 if internal is None else np.asarray(internal, bool)
    return asks if max_depth is None else asks & (tree["depth"] < max_depth)


def reached(tree, internal=None, max_depth=None):
    """Which nodes of the table are part of the tree pruned to ``internal`` or cut at
    ``max_depth``."""
    asks = _asks(tree, internal, max_depth)
    reach = np.zeros(len(asks), bool)
    reach[0] = True
    for t in range(len(asks)):                         # parents come before children
        if reach[t] and asks[t]:
            reach[tree["left"][t]] = reach[tree["right"][t]] = True
    return reach


def leaves_of(tree, xb, internal=None, max_depth=None):
    """The node each row of binned features ``xb`` ends in. ``internal``: which nodes
    still ask their question (a pruned tree is the same table with fewer of them);
    ``max_depth``: nodes at that depth answer instead of asking."""
    asks = _asks(tree, internal, max_depth)
    xb = np.asarray(xb, np.int64)
    node = np.zeros(len(xb), np.int64)
    rows = np.flatnonzero(asks[node])
    while len(rows):
        at = node[rows]
        right = xb[rows, tree["feature"][at]] > tree["cut"][at]
        node[rows] = np.where(right, tree["right"][at], tree["left"][at])
        rows = rows[asks[node[rows]]]
    return node


def share(tree, xb, internal=None, max_depth=None):
    """The share of signal among the training events of the leaf each row reaches."""
    leaf = leaves_of(tree, xb, internal, max_depth)
    return tree["n1"][leaf] / np.maximum(tree["n"][leaf], 1e-12)


def vote(tree, xb, internal=None, max_depth=None):
    """The answer of the leaf each row reaches: 1 (signal) where more than half of the
    leaf's training events are signal, else 0."""
    return (share(tree, xb, internal, max_depth) > 0.5).astype(np.int64)


def error(tree, xb, y, internal=None, max_depth=None):
    """The share of rows the tree answers wrongly."""
    return float(np.mean(vote(tree, xb, internal, max_depth) != np.asarray(y)))


def leaf_count(tree, internal=None, max_depth=None):
    """How many leaves the tree has, pruned to ``internal`` or cut at ``max_depth``."""
    return int((reached(tree, internal, max_depth) & ~_asks(tree, internal, max_depth)).sum())


def pruning(tree):
    """Cost-complexity pruning (Breiman, Friedman, Olshen and Stone, 1984). A tree's
    cost is its training error plus ``alpha`` for every leaf. For a node that asks a
    question, cutting everything below it costs the same as keeping it at
    ``g = (R(node) - R(leaves below it)) / (leaves below it - 1)``, with ``R`` the
    share of all training events answered wrongly. The node with the smallest ``g`` is
    the weakest link and is cut first (ties together); then ``g`` is worked out again.

    Returns ``(alphas, internals, links)``: ``internals[k]`` is the tree that is best
    from ``alphas[k]`` up to ``alphas[k + 1]``. The first, at 0, is the smallest tree
    with the training error of the tree as grown; the last is the root alone.
    ``links[k]`` is the node cut to reach tree ``k`` (-1 for the first)."""
    total = tree["n"][0]
    alone = np.minimum(tree["n1"], tree["n"] - tree["n1"]) / total
    internal = tree["feature"] >= 0
    alphas, internals, links = [0.0], [internal.copy()], [-1]
    while internal[0]:
        below, leaves = alone.copy(), np.ones(len(internal))
        for t in range(len(internal) - 1, -1, -1):     # children before parents
            if internal[t]:
                a, b = tree["left"][t], tree["right"][t]
                below[t], leaves[t] = below[a] + below[b], leaves[a] + leaves[b]
        g = np.where(internal, (alone - below) / np.maximum(leaves - 1, 1), np.inf)
        a = float(g.min())
        cut = np.flatnonzero(g <= a + 1e-12)
        internal = internal.copy()
        internal[cut] = False
        for t in range(len(internal)):                 # everything under a cut goes too
            if tree["feature"][t] >= 0 and not internal[t]:
                internal[tree["left"][t]] = internal[tree["right"][t]] = False
        if a <= alphas[-1] + 1e-15:
            internals[-1] = internal.copy()            # no dearer than the last: same tree
        else:
            alphas.append(a)
            internals.append(internal.copy())
            links.append(int(cut[0]))
    return np.array(alphas), np.array(internals), np.array(links)


def link_arithmetic(tree, internal, node):
    """What cutting below ``node`` of the tree pruned to ``internal`` does:
    ``(wrong, wrong_below, leaves_below)``: training events answered wrongly if the
    node answers by itself, by the leaves now below it, and how many those are. The
    node's ``g`` is ``(wrong - wrong_below) / total / (leaves_below - 1)``."""
    alone = np.minimum(tree["n1"], tree["n"] - tree["n1"])
    todo, below, leaves = [int(node)], 0.0, 0
    while todo:
        t = todo.pop()
        if internal[t]:
            todo += [int(tree["left"][t]), int(tree["right"][t])]
        else:
            below, leaves = below + alone[t], leaves + 1
    return int(round(alone[node])), int(round(below)), leaves


def pruned_at(alphas, alpha):
    """Which tree of a :func:`pruning` sequence is best at ``alpha``."""
    return int(np.searchsorted(alphas, alpha, side="right") - 1)


def regions(tree, cuts, box, internal=None, max_depth=None):
    """The rectangles a two-feature tree divides ``box`` (``((x0, x1), (y0, y1))``)
    into: rows ``(node, x0, x1, y0, y1)``, one for each leaf."""
    asks = _asks(tree, internal, max_depth)
    out, todo = [], [(0, box[0][0], box[0][1], box[1][0], box[1][1])]
    while todo:
        t, x0, x1, y0, y1 = todo.pop()
        if not asks[t]:
            out.append((t, x0, x1, y0, y1))
            continue
        v = float(cuts[tree["feature"][t], tree["cut"][t]])
        if tree["feature"][t] == 0:
            v = min(max(v, x0), x1)
            todo += [(tree["left"][t], x0, v, y0, y1), (tree["right"][t], v, x1, y0, y1)]
        else:
            v = min(max(v, y0), y1)
            todo += [(tree["left"][t], x0, x1, y0, v), (tree["right"][t], x0, x1, v, y1)]
    return np.array(sorted(out), float)


def node_scan(xb, y, bins, w=None, criterion="gini"):
    """One node's events scored at every threshold of every feature: ``(left, left1,
    gain)``, each ``(features, bins - 1)``: the events and the signal events that go
    left, and the :func:`split_gain`."""
    xb = np.asarray(xb, np.int64)
    n, nf = xb.shape
    y = np.asarray(y, float)
    w = np.ones(n) if w is None else np.asarray(w, float)
    idx = (np.arange(nf) * bins + xb).ravel()
    h = np.bincount(idx, weights=np.repeat(w, nf), minlength=nf * bins).reshape(nf, bins)
    h1 = np.bincount(idx, weights=np.repeat(w * y, nf), minlength=nf * bins).reshape(nf, bins)
    left, left1 = np.cumsum(h, 1)[:, :-1], np.cumsum(h1, 1)[:, :-1]
    return left, left1, split_gain(w.sum(), (w * y).sum(), left, left1, criterion)


def fit_two(x, y, w=None, **settings):
    """A tree on the two features of events ``x``, every midpoint between neighbouring
    values (of the events counted at least once) a candidate threshold. Returns
    ``(tree, cuts)``."""
    x = np.asarray(x, float)
    cuts = midpoints(x if w is None else x[np.asarray(w) > 0])
    return grow(binned(x, cuts), y, cuts.shape[1] + 1, w=w, **settings), cuts


def fit_cells(x, y, w=None, **settings):
    """A tree on the 49 cells, each cut at the ``CELL_BINS - 1`` fixed thresholds."""
    return grow(binned(x, CELL_CUTS), y, CELL_BINS, w=w, **settings)


def bootstrap(rng, n):
    """A bootstrap resample of ``n`` events: ``n`` picks with replacement, as how many
    times each event was picked."""
    return np.bincount(rng.integers(0, n, n), minlength=n)


PLANE = ((0.0, 0.8), (3.2, 6.4))  # the part of the feature plane that is drawn
SEEDS = (0, 1, 2, 3, 4)
TWO_N = 250                       # training events of each class on the two features
CELLS_N = 1000                    # training events of each class on the 49 cells


def sample(kind, n, seed):
    """Seed ``seed``'s training sample of ``n`` events of each class: ``(x, y, rows)``
    with ``kind`` ``"two"`` (the two features) or ``"cells"``."""
    e = events()
    rows, _ = draw(n, seed)
    return e["pool_" + kind][rows], e["pool_y"][rows], rows


# ====================================================================== 160: one split
HANDFUL = 6                       # events of each class in the node worked by hand
HANDFUL_SEED = 1000               # the first seed tried for it
NODE_GRID = 200                   # thresholds scanned on each feature of the large node
BLIND_MIN = 100                   # the smallest node looked at for the criteria example
HIST_BINS = 32


def handful():
    """The twelve events worked by hand: ``HANDFUL`` of each class, the first seeded
    draw (seeds from ``HANDFUL_SEED`` up) in which neither feature has a threshold
    that separates the classes completely, so that every impurity on screen is the
    impurity of a mixed node. Returns ``(rows, seed)``."""
    e = events()
    for seed in range(HANDFUL_SEED, HANDFUL_SEED + 100):
        rows, _ = draw(HANDFUL, seed)
        x, y = e["pool_two"][rows], e["pool_y"][rows]
        cuts = midpoints(x)
        gain = node_scan(binned(x, cuts), y, cuts.shape[1] + 1)[2]
        if ((cuts < PAD).all() and cuts.shape[1] == 2 * HANDFUL - 1
                and gain.max(1).max() < 0.5 - 1e-9):
            return rows, seed
    raise RuntimeError("no mixed handful")


def scan_all(x, y, cuts):
    """Every threshold in ``cuts`` on events ``x``: ``(left, left1, gains)`` with
    ``gains[criterion, feature, threshold]``."""
    xb = binned(x, cuts)
    out = [node_scan(xb, y, cuts.shape[1] + 1, criterion=c) for c in CRITERIA]
    return out[0][0], out[0][1], np.array([o[2] for o in out])


def node_grid(x):
    """``NODE_GRID`` evenly spaced thresholds on each feature, from the lowest value
    of ``x`` to the highest (the ends themselves left out)."""
    lo, hi = x.min(0), x.max(0)
    steps = np.arange(1, NODE_GRID + 1) / (NODE_GRID + 1)
    return lo[:, None] + (hi - lo)[:, None] * steps[None]


def blind_node(x, y, cuts):
    """A node at which the misclassification error sees nothing: grow the Gini tree
    and, of its nodes of at least ``BLIND_MIN`` events at which no threshold of either
    feature lowers the error, take the one whose Gini split gains most. Returns the
    node's counts and those of its Gini split, or None."""
    bins = cuts.shape[1] + 1
    xb = binned(x, cuts)
    tree = grow(xb, y, bins)
    found = None
    for t in np.flatnonzero((tree["feature"] >= 0) & (tree["n"] >= BLIND_MIN)):
        inside = leaves_of(tree, xb, max_depth=int(tree["depth"][t])) == t
        gains = [node_scan(xb[inside], y[inside], bins, criterion=c)[2] for c in CRITERIA]
        if gains[2].max() <= 1e-12 and (found is None or tree["gain"][t] > found["gain"]):
            left = tree["left"][t]
            f, c = int(tree["feature"][t]), int(tree["cut"][t])
            found = dict(node=int(t), depth=int(tree["depth"][t]), feature=f,
                         threshold=float(cuts[f, c]), gain=float(tree["gain"][t]),
                         counts=np.array([[tree["n"][t], tree["n1"][t]],
                                          [tree["n"][left], tree["n1"][left]],
                                          [tree["n"][t] - tree["n"][left],
                                           tree["n1"][t] - tree["n1"][left]]]),
                         gains=np.array([g[f, c] for g in gains]),
                         best=np.array([g.max() for g in gains]))
    return found


def criterion_run(seed):
    """Seed ``seed``'s 49-cell sample: a tree grown to the end with each criterion.
    Returns held-out and training error and the number of leaves of each, and, over
    the nodes the Gini tree splits: the share at which entropy would take the same
    split, the share at which the error would, and the share at which no threshold
    lowers the error at all."""
    e = events()
    x, y, _ = sample("cells", CELLS_N, seed)
    xb, hb = binned(x, CELL_CUTS), binned(e["held_cells"], CELL_CUTS)
    out = []
    for c in CRITERIA:
        tree = grow(xb, y, CELL_BINS, criterion=c)
        out.append((error(tree, hb, e["held_y"]), error(tree, xb, y), leaf_count(tree)))
        if c == "gini":
            first = tree
    same = np.zeros(3)
    asks = np.flatnonzero(first["feature"] >= 0)
    for t in asks:
        inside = leaves_of(first, xb, max_depth=int(first["depth"][t])) == t
        chosen = first["feature"][t] * (CELL_BINS - 1) + first["cut"][t]
        gains = [node_scan(xb[inside], y[inside], CELL_BINS, criterion=c)[2].ravel()
                 for c in CRITERIA]
        same += (gains[1].argmax() == chosen, gains[2].max() > 1e-12
                 and gains[2].argmax() == chosen, gains[2].max() <= 1e-12)
    return np.array(out), same / len(asks), len(asks)


def _splits_build():
    e = events()
    rows, seed = handful()
    x, y = e["pool_two"][rows], e["pool_y"][rows]
    cuts = midpoints(x)
    left, left1, gains = scan_all(x, y, cuts)
    out = dict(hand_rows=rows, hand_seed=seed, hand_x=x, hand_y=y, hand_cuts=cuts,
               hand_left=left, hand_left1=left1, hand_gains=gains,
               hand_images=picture(e["pool_index"][rows]))
    roots, hists = [], None
    for s in SEEDS:
        xs, ys, _ = sample("two", CELLS_N, s)
        grid = node_grid(xs)
        nl, nl1, g = scan_all(xs, ys, grid)
        best = g.reshape(3, -1).argmax(1)
        roots.append([(b // NODE_GRID, grid[b // NODE_GRID, b % NODE_GRID], g[k].ravel()[b])
                      for k, b in enumerate(best)])
        if s == SEEDS[0]:
            edges = np.array([np.linspace(*PLANE[f], HIST_BINS + 1) for f in range(2)])
            hists = np.array([[np.histogram(xs[ys == c, f], edges[f])[0] for c in (1, 0)]
                              for f in range(2)])
            blind = blind_node(xs, ys, grid)
            out.update(node_grid=grid, node_left=nl, node_left1=nl1, node_gains=g,
                       node_hist=hists, node_edges=edges, node_xy=xs[::8], node_cls=ys[::8])
            if blind is not None:
                out.update(blind_counts=blind["counts"], blind_gains=blind["gains"],
                           blind_best=blind["best"],
                           blind_where=np.array([blind["node"], blind["depth"],
                                                 blind["feature"], blind["threshold"]]))
    runs = _parallel(criterion_run, [(s,) for s in SEEDS])
    out.update(roots=np.array(roots, float), crit=np.array([r[0] for r in runs]),
               same=np.array([r[1] for r in runs]), asked=np.array([r[2] for r in runs]))
    return out


def splits_data():
    return _cache("splits", _splits_build, (node_scan, handful, scan_all, node_grid,
                                            blind_node, criterion_run, sample),
                  seeds=SEEDS, handful=(HANDFUL, HANDFUL_SEED), grid=NODE_GRID,
                  n=CELLS_N, blind=BLIND_MIN, plane=PLANE, hist=HIST_BINS)




# ====================================================================== 161: holding back
DEPTHS = tuple(range(1, 17))      # depth 16 or more: grown to the end
LEAF_SIZES = (1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120)
MIN_GAINS = (0.0, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.016)
ALPHAS = (0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.004, 0.006, 0.008, 0.012, 0.02, 0.04)
FOLDS = 5
SHOWN_DEPTH, SHOWN_LEAF, SHOWN_GAIN = 3, 20, 0.004     # the settings drawn on the tree


def tree_arrays(tree, prefix):
    """A node table as arrays to keep, without the forest's list of features tried."""
    return {prefix + k: v for k, v in tree.items() if k != "tried"}


def tree_from(d, prefix):
    """The node table kept by :func:`tree_arrays`."""
    return {k[len(prefix):]: np.asarray(v) for k, v in d.items() if k.startswith(prefix)}


def folds(y, seed):
    """Which of ``FOLDS`` folds each training event is held back in: each class dealt
    out in a seeded random order, so that every fold has both in proportion."""
    rng = np.random.default_rng([SPLIT, 2, seed])
    fold = np.zeros(len(y), np.int64)
    for c in (0, 1):
        ids = rng.permutation(np.flatnonzero(np.asarray(y) == c))
        fold[ids] = np.arange(len(ids)) % FOLDS
    return fold


def sweep(kind, x, y, hx, hy):
    """One training sample against every setting of each control: ``depth``,
    ``leaf``, ``gain`` and ``alpha``, each ``(settings, 3)``: training error, error on
    ``hx, hy`` and number of leaves."""
    fit = fit_two if kind == "two" else fit_cells

    def made(x):
        if kind == "two":
            tree, cuts = fit(x, y)
            return tree, cuts, binned(x, cuts), binned(hx, cuts)
        return fit(x, y), None, binned(x, CELL_CUTS), binned(hx, CELL_CUTS)

    def row(tree, xb, hb, **how):
        return (error(tree, xb, y, **how), error(tree, hb, hy, **how), leaf_count(tree, **how))
    tree, _, xb, hb = made(x)
    out = dict(depth=[row(tree, xb, hb, max_depth=d if d < DEPTHS[-1] else None)
                      for d in DEPTHS])
    alphas, internals, _ = pruning(tree)
    out["alpha"] = [row(tree, xb, hb, internal=internals[pruned_at(alphas, a)])
                    for a in ALPHAS]
    for name, key, values in (("leaf", "min_leaf", LEAF_SIZES), ("gain", "min_gain",
                                                                 MIN_GAINS)):
        rows = []
        for v in values:
            t = fit(x, y, **{key: v})
            t = t[0] if kind == "two" else t
            rows.append(row(t, xb, hb))
        out[name] = rows
    return {k: np.array(v, float) for k, v in out.items()}


def sweep_run(kind, seed):
    e = events()
    x, y, _ = sample(kind, TWO_N if kind == "two" else CELLS_N, seed)
    return sweep(kind, x, y, e["held_" + kind], e["held_y"])


def cross_validate(x, y, seed):
    """Five-fold cross-validation of each control on the two-feature sample: the tree
    is grown on four fifths of the training events and judged on the fifth held back,
    each fifth in turn. Returns the mean error over the five for every setting:
    ``depth``, ``leaf``, ``gain``, ``alpha``."""
    fold = folds(y, seed)
    runs = []
    for k in range(FOLDS):
        out = fold == k
        r = sweep("two", x[~out], y[~out], x[out], y[out])
        runs.append({name: v[:, 1] for name, v in r.items()})
    return {name: np.mean([r[name] for r in runs], 0) for name in runs[0]}


def choose(cv, values):
    """The setting with the lowest cross-validated error; of equals, the one that
    holds the tree back most (the last)."""
    cv = np.asarray(cv)
    return int(np.flatnonzero(cv <= cv.min() + 1e-12)[-1]), values


def _pruning_build():
    e = events()
    x, y, rows = sample("two", TWO_N, 0)
    tree, cuts = fit_two(x, y)
    xb, hb = binned(x, cuts), binned(e["held_two"], cuts)
    alphas, internals, links = pruning(tree)
    out = dict(x=x, y=y, rows=rows, cuts=cuts, **tree_arrays(tree, "tree_"),
               alphas=alphas, internals=internals, links=links,
               seq=np.array([(error(tree, xb, y, internal=i),
                              error(tree, hb, e["held_y"], internal=i),
                              leaf_count(tree, internal=i)) for i in internals]))
    for name, key, v in (("leaf", "min_leaf", SHOWN_LEAF), ("gain", "min_gain", SHOWN_GAIN)):
        t, c = fit_two(x, y, **{key: v})
        out.update(tree_arrays(t, name + "_tree_"))
        out[name + "_cuts"] = c
    cv = cross_validate(x, y, 0)
    out.update({"cv_" + k: v for k, v in cv.items()})
    jobs = [(kind, s) for kind in ("two", "cells") for s in SEEDS]
    runs = _parallel(sweep_run, jobs)
    for i, kind in enumerate(("two", "cells")):
        for name in ("depth", "leaf", "gain", "alpha"):
            out["%s_%s" % (kind, name)] = np.array([runs[i * len(SEEDS) + j][name]
                                                    for j in range(len(SEEDS))])
    return out


def pruning_data():
    return _cache("pruning", _pruning_build, (fit_two, fit_cells, sample, folds, sweep,
                                              sweep_run, cross_validate, tree_arrays),
                  seeds=SEEDS, two=TWO_N, cells=CELLS_N, depths=DEPTHS, leaves=LEAF_SIZES,
                  gains=MIN_GAINS, alphas=ALPHAS, folds=FOLDS,
                  shown=(SHOWN_DEPTH, SHOWN_LEAF, SHOWN_GAIN))




# ====================================================================== 162: instability
INST_SHOWN = 4                    # resamples whose trees are drawn
INST_DEPTH = 2                    # levels of questions drawn for each
INST_TREES = 20                   # resamples per seed for the disagreement
INST_DEPTHS = (1, 2, 3, 4, 6, 8, 10, 99)          # 99: grown to the end
INST_SWAPS = (0.002, 0.01, 0.05, 0.2)             # share of the training events replaced
INST_SWAP_REPEATS = 10
INST_SAMPLES = 40                 # separate training samples for the bias and variance
INST_SAMPLE_SEED = 100            # their seeds start here
INST_EVENTS = 3                   # held-out events of each class whose votes are listed
INST_POINTS = 400                 # held-out events drawn on the plane
KINDS = ("two", "cells")


def kind_n(kind):
    return TWO_N if kind == "two" else CELLS_N


def fit_kind(kind, x, y, w=None, **settings):
    """A tree on ``x`` and what turns other events into its bins: ``(tree, cuts)``."""
    if kind == "two":
        return fit_two(x, y, w, **settings)
    return fit_cells(x, y, w, **settings), CELL_CUTS


def depth_of(d):
    return None if d >= 99 else d


def disagreement(votes):
    """How often two of the trees give different answers to the same event: the share
    of (pair of trees, event) combinations that differ. ``votes``: ``(trees, events)``
    of 0 and 1."""
    votes = np.asarray(votes)
    r = len(votes)
    k = votes.sum(0)
    return float(np.mean(k * (r - k)) / (r * (r - 1) / 2))


def resample_run(kind, seed):
    """Seed ``seed``'s training sample resampled ``INST_TREES`` times (bootstrap), a
    tree grown to the end on each. Returns the disagreement between pairs of them on
    the held-out events at each depth of ``INST_DEPTHS``, each tree's held-out error
    at each depth, each tree's root (feature, threshold), and the disagreement between
    the tree of the sample itself and that of the sample with a share of its events
    replaced by fresh ones (``INST_SWAPS``, mean of ``INST_SWAP_REPEATS``)."""
    e = events()
    x, y, rows = sample(kind, kind_n(kind), seed)
    hx, hy = e["held_" + kind], e["held_y"]
    rng = np.random.default_rng([SPLIT, 3, seed])
    votes, roots = [], []
    for _ in range(INST_TREES):
        tree, cuts = fit_kind(kind, x, y, bootstrap(rng, len(y)))
        hb = binned(hx, cuts)
        votes.append([vote(tree, hb, max_depth=depth_of(d)) for d in INST_DEPTHS])
        roots.append((tree["feature"][0], cuts[tree["feature"][0], tree["cut"][0]]))
    votes = np.array(votes)                            # tree, depth, event
    tree, cuts = fit_kind(kind, x, y)
    base = vote(tree, binned(hx, cuts))
    spare = np.setdiff1d(np.arange(2 * POOL), rows)
    swapped = []
    for share_ in INST_SWAPS:
        k = max(1, int(round(share_ * len(y))))
        diff = []
        for _ in range(INST_SWAP_REPEATS):
            out = rng.choice(len(y), k, replace=False)
            new = rows.copy()
            new[out] = rng.choice(spare, k, replace=False)
            t2, c2 = fit_kind(kind, e["pool_" + kind][new], e["pool_y"][new])
            diff.append(np.mean(vote(t2, binned(hx, c2)) != base))
        swapped.append(np.mean(diff))
    return (np.array([disagreement(votes[:, j]) for j in range(len(INST_DEPTHS))]),
            (votes != hy).mean(2), np.array(roots, float), np.array(swapped))


def spread_run(kind, event):
    """``INST_SAMPLES`` separate training samples, a tree on each, and each tree's
    answer as the share of signal in the leaf an event reaches. On the held-out
    events, at each depth: the mean squared error of that share against the label
    (1 or 0), and its two parts, which add up to it exactly: the squared error of the
    average over the trees (bias, with whatever no tree could know) and the variance
    of the trees around their average. Also the error of single trees and of the
    majority answer of them all. Rows: ``(mse, bias2, variance, single, together)``.
    Second: every tree's answer for held-out event ``event`` at the first depth and at
    the last."""
    e = events()
    hx, hy = e["held_" + kind], e["held_y"]
    p = []
    for s in range(INST_SAMPLES):
        x, y, _ = sample(kind, kind_n(kind), INST_SAMPLE_SEED + s)
        tree, cuts = fit_kind(kind, x, y)
        hb = binned(hx, cuts)
        p.append([share(tree, hb, max_depth=depth_of(d)) for d in INST_DEPTHS])
    p = np.array(p)                                    # sample, depth, event
    mean = p.mean(0)
    return (np.array([((p - hy) ** 2).mean((0, 2)), ((mean - hy) ** 2).mean(1),
                      p.var(0).mean(1), ((p > 0.5) != hy).mean((0, 2)),
                      ((mean > 0.5) != hy).mean(1)]), p[:, [0, -1], event].T)


def _instability_build():
    e = events()
    x, y, rows = sample("two", TWO_N, 0)
    rng = np.random.default_rng([SPLIT, 4, 0])
    order = rng.permutation(len(e["held_y"]))
    points = order[:INST_POINTS]
    out = dict(x=x, y=y, rows=rows, points=points)
    rng = np.random.default_rng([SPLIT, 3, 0])         # the first resamples of resample_run
    votes, weights = [], []
    for b in range(INST_SHOWN):
        w = bootstrap(rng, len(y))
        tree, cuts = fit_two(x, y, w)
        out.update(tree_arrays(tree, "tree%d_" % b))
        out["cuts%d" % b] = cuts
        votes.append(vote(tree, binned(e["held_two"], cuts)))
        weights.append(w)
    votes = np.array(votes)
    split = votes.min(0) != votes.max(0)               # the drawn trees do not all agree
    # listed: the first held-out events of each class, in a seeded order, that the
    # drawn trees disagree about (chosen as examples of disagreement; split_share
    # says how common they are)
    listed = np.concatenate([order[(e["held_y"][order] == c) & split[order]][:INST_EVENTS]
                             for c in (1, 0)])
    out.update(weights=np.array(weights), point_votes=votes[:, points], listed=listed,
               listed_votes=votes[:, listed], split_share=split.mean(),
               shown_errors=(votes != e["held_y"]).mean(1),
               listed_images=picture(e["held_index"][listed]))
    jobs = [(kind, s) for kind in KINDS for s in SEEDS]
    runs = _parallel(resample_run, jobs)
    spreads = _parallel(spread_run, [(k, int(listed[0])) for k in KINDS])
    for i, kind in enumerate(KINDS):
        mine = runs[i * len(SEEDS):(i + 1) * len(SEEDS)]
        out[kind + "_disagree"] = np.array([r[0] for r in mine])
        out[kind + "_errors"] = np.array([r[1] for r in mine])
        out[kind + "_roots"] = np.array([r[2] for r in mine])
        out[kind + "_swapped"] = np.array([r[3] for r in mine])
        out[kind + "_parts"], out[kind + "_one"] = spreads[i]
    return out


def instability_data():
    return _cache("instability", _instability_build,
                  (fit_two, fit_cells, sample, bootstrap, fit_kind, disagreement,
                   resample_run, spread_run, tree_arrays),
                  seeds=SEEDS, two=TWO_N, cells=CELLS_N, shown=INST_SHOWN, trees=INST_TREES,
                  depths=INST_DEPTHS, swaps=INST_SWAPS, repeats=INST_SWAP_REPEATS,
                  samples=(INST_SAMPLES, INST_SAMPLE_SEED), events=INST_EVENTS,
                  points=INST_POINTS)




# ====================================================================== 163: bagging
BAG_PICKED = 8                    # events of each class in the resample drawn pick by pick
BAG_TREES = 100
BAG_STEPS = (1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 70, 100)
BAG_GRID = 48                     # the plane is voted on at 48 x 48 places
BAG_MAPS = 6                      # single trees whose own answers are kept as maps
BAG_SEEDS = tuple(range(20))      # training samples, a bag of trees on each


def plane_grid(n=BAG_GRID):
    """The centres of ``n x n`` squares covering ``PLANE``, row 0 at the top of the
    second feature: ``(n * n, 2)``."""
    (x0, x1), (y0, y1) = PLANE
    xs = x0 + (np.arange(n) + 0.5) * (x1 - x0) / n
    ys = y1 - (np.arange(n) + 0.5) * (y1 - y0) / n
    return np.column_stack([np.tile(xs, n), np.repeat(ys, n)])


def bag_plane():
    """Bagging on the two features: ``BAG_TREES`` bootstrap resamples of seed 0's
    sample, a tree grown to the end on each. Kept: each resample's counts, the first
    trees' answers over the plane, the share of trees answering signal over the plane
    after each number of trees in ``BAG_STEPS``, and the held-out error of a single
    tree and of the majority after each."""
    e = events()
    x, y, rows = sample("two", TWO_N, 0)
    rng = np.random.default_rng([SPLIT, 5, 1])
    grid = plane_grid()
    maps, held, weights = [], [], []
    for _ in range(BAG_TREES):
        w = bootstrap(rng, len(y))
        tree, cuts = fit_two(x, y, w)
        maps.append(vote(tree, binned(grid, cuts)))
        held.append(vote(tree, binned(e["held_two"], cuts)))
        weights.append(w)
    maps, held = np.array(maps, float), np.array(held, float)
    steps = np.array(BAG_STEPS)
    cum = np.cumsum(held, 0)[steps - 1] / steps[:, None]
    n = BAG_GRID
    return dict(x=x, y=y, rows=rows, weights=np.array(weights),
                maps=maps[:BAG_MAPS].reshape(-1, n, n),
                shares=(np.cumsum(maps, 0)[steps - 1] / steps[:, None]).reshape(-1, n, n),
                single=(held != e["held_y"]).mean(1),
                together=((cum > 0.5) != e["held_y"]).mean(1))


def bag_run(seed, mtry=None, keep=False):
    """One bag of ``BAG_TREES`` trees on seed ``seed``'s 49-cell sample (``mtry``: a
    random forest). Returns ``shares`` (the share of trees answering signal for every
    held-out event after each number of trees in ``BAG_STEPS``), the held-out error of
    their majority after each, the out-of-bag error after each (every training event
    judged only by the trees whose resample left it out; events no tree has left out
    yet are not counted), the held-out error of each single tree, the share of the
    sample each resample contains, the mean correlation between two trees of whether
    they answer a held-out event correctly, and the cell each tree asks about first.
    ``keep``: the trees as well."""
    e = events()
    x, y, _ = sample("cells", CELLS_N, seed)
    xb, hb, hy = binned(x, CELL_CUTS), binned(e["held_cells"], CELL_CUTS), e["held_y"]
    rng = np.random.default_rng([SPLIT, 6, seed] + ([] if mtry is None else [mtry]))
    total, out_votes, out_count = np.zeros(len(hy)), np.zeros(len(y)), np.zeros(len(y))
    shares, held_err, oob_err, single, unique, right, trees = [], [], [], [], [], [], []
    first = []
    for b in range(1, BAG_TREES + 1):
        w = bootstrap(rng, len(y))
        tree = grow(xb, y, CELL_BINS, w=w, mtry=mtry, rng=rng)
        v = vote(tree, hb)
        total += v
        out = w == 0
        out_votes[out] += vote(tree, xb[out])
        out_count += out
        single.append(np.mean(v != hy))
        unique.append(np.mean(~out))
        right.append(v == hy)
        first.append(tree["feature"][0])
        if keep:
            trees.append(tree)
        if b in BAG_STEPS:
            shares.append(total / b)
            held_err.append(np.mean((total / b > 0.5) != hy))
            seen = out_count > 0
            oob_err.append(np.mean((out_votes[seen] / out_count[seen] > 0.5) != y[seen]))
    corr = np.corrcoef(np.array(right, float))
    alike = (corr.sum() - len(corr)) / (len(corr) * (len(corr) - 1))
    res = dict(shares=np.array(shares, np.float32), held=np.array(held_err),
               oob=np.array(oob_err), single=np.array(single), unique=np.array(unique),
               alike=float(alike), first=np.array(first))
    return (res, trees, (xb, y, hb, hy)) if keep else res


def vote_variance(shares):
    """How much the share of trees answering signal for an event changes from one
    training sample to the next, and what the formula says it should be.
    ``shares[sample, step, event]`` from :func:`bag_run`. For each event, over the
    samples: ``sigma2``, the variance of one tree's answer; ``cov``, the covariance of
    two different trees' answers (from the last step: the mean of the products of
    pairs of trees of the same bag, minus the squared mean). Averaged over the events:
    ``rho = cov / sigma2``, the measured variance of the share after each number of
    trees B, and the formula ``rho sigma2 + (1 - rho) sigma2 / B``."""
    v = np.asarray(shares, float)
    r, b = len(v), BAG_STEPS[-1]
    last = v[:, -1]
    total = last.sum(0)
    mean = total / r
    mean2 = (total ** 2 - (last ** 2).sum(0)) / (r * (r - 1))     # unbiased squared mean
    pairs = ((b * last ** 2 - last) / (b - 1)).mean(0)             # answers are 0 or 1
    sigma2, cov = float((mean - mean2).mean()), float((pairs - mean2).mean())
    steps = np.array(BAG_STEPS, float)
    return dict(sigma2=sigma2, cov=cov, rho=cov / sigma2,
                measured=v.var(0, ddof=1).mean(1), formula=cov + (sigma2 - cov) / steps)


def bag_job(seed):
    return bag_run(seed)


def _bagging_build():
    e = events()
    rows, _ = draw(BAG_PICKED, 0)
    picks = np.random.default_rng([SPLIT, 5, 0]).integers(0, len(rows), len(rows))
    out = {"plane_" + k: v for k, v in bag_plane().items()}
    out.update(picked_rows=rows, picks=picks, picked_y=e["pool_y"][rows],
               picked_images=picture(e["pool_index"][rows]))
    runs = _parallel(bag_job, [(s,) for s in BAG_SEEDS])
    for k in ("held", "oob", "single", "unique"):
        out[k] = np.array([r[k] for r in runs])
    out["alike"] = np.array([r["alike"] for r in runs])
    var = vote_variance(np.array([r["shares"] for r in runs]))
    out.update(var_measured=var["measured"], var_formula=var["formula"],
               var_terms=np.array([var["sigma2"], var["cov"], var["rho"]]))
    return out


def bagging_data():
    return _cache("bagging", _bagging_build, (fit_two, sample, bootstrap, plane_grid,
                                              bag_plane, bag_run, vote_variance),
                  seeds=BAG_SEEDS, two=TWO_N, cells=CELLS_N, trees=BAG_TREES,
                  steps=BAG_STEPS, grid=BAG_GRID, maps=BAG_MAPS, picked=BAG_PICKED,
                  plane=PLANE)


# ====================================================================== 164: random forest
MTRY = (1, 2, 4, 7, 14, 28, 49)   # cells tried at each split; 49 is bagging
RF_MTRY = 7                       # the forest whose importances are read: 7 = sqrt(49)
RF_NODES = 3                      # nodes of the first tree whose candidates are drawn
RF_REPEATS = 3                    # shuffles of each cell for the permutation importance


def gain_importance(trees):
    """Each cell's share of the impurity the forest removed: over every node that
    splits on it, the node's gain times its share of the tree's events, summed over
    the trees and divided by the total."""
    total = np.zeros(CELLS)
    for t in trees:
        asks = t["feature"] >= 0
        total += np.bincount(t["feature"][asks], weights=(t["gain"] * t["n"])[asks] / t["n"][0],
                             minlength=CELLS)
    return total / total.sum()


def permutation_importance(trees, hb, hy, rng, repeats=RF_REPEATS):
    """Each cell's permutation importance: its readings are shuffled among the
    held-out events, so that it says nothing but keeps its distribution, and the rise
    of the forest's held-out error is recorded (mean of ``repeats`` shuffles; trees
    that never ask about the cell answer as before)."""
    votes = np.array([vote(t, hb) for t in trees], float)
    base = np.mean((votes.mean(0) > 0.5) != hy)
    uses = [np.unique(t["feature"][t["feature"] >= 0]) for t in trees]
    out = np.zeros(CELLS)
    for f in range(CELLS):
        asking = [i for i, u in enumerate(uses) if f in u]
        if not asking:
            continue
        rise = []
        for _ in range(repeats):
            mixed = hb.copy()
            mixed[:, f] = hb[rng.permutation(len(hb)), f]
            total = votes.sum(0)
            for i in asking:
                total += vote(trees[i], mixed) - votes[i]
            rise.append(np.mean((total / len(trees) > 0.5) != hy) - base)
        out[f] = np.mean(rise)
    return out, float(base)


def candidates(tree, xb, y, w):
    """What the first ``RF_NODES`` nodes of a forest's tree were offered: for each,
    the cells drawn, the best gain each could give there (0: it cannot split the
    node), and the cell taken."""
    cells_, gains, taken = [], [], []
    for t in range(RF_NODES):
        inside = (leaves_of(tree, xb, max_depth=int(tree["depth"][t])) == t) & (w > 0)
        best = node_scan(xb[inside], y[inside], CELL_BINS, w[inside])[2].max(1)
        drawn = tree["tried"][t]
        cells_.append(drawn)
        gains.append(np.where(drawn >= 0, best[np.maximum(drawn, 0)], 0.0))
        taken.append(tree["feature"][t])
    return np.array(cells_), np.array(gains), np.array(taken)


def forest_run(seed):
    """Seed ``seed``'s 49-cell sample: a forest of ``BAG_TREES`` trees for each number
    of cells tried in ``MTRY``. Returns, for each: the forest's held-out error, the
    mean held-out error of its single trees and the mean correlation between two
    trees; and for ``RF_MTRY``: both importances of every cell, and the first tree's
    candidates."""
    rows, extra, first = [], {}, []
    for m in MTRY:
        keep = m == RF_MTRY
        res = bag_run(seed, None if m == CELLS else m, keep)
        if keep:
            res, trees, (xb, y, hb, hy) = res
            extra["gain"] = gain_importance(trees)
            extra["permutation"], extra["base"] = permutation_importance(
                trees, hb, hy, np.random.default_rng([SPLIT, 8, seed]))
            rng = np.random.default_rng([SPLIT, 6, seed, m])       # the first tree again
            w = bootstrap(rng, len(y))
            tree = grow(xb, y, CELL_BINS, w=w, mtry=m, rng=rng)
            extra["tried"], extra["tried_gain"], extra["taken"] = candidates(tree, xb, y, w)
            extra["tried_cut"] = tree["cut"][:RF_NODES]
            extra["tried_n"] = np.array([tree["n"][:RF_NODES], tree["n1"][:RF_NODES]])
        rows.append((res["held"][-1], res["single"].mean(), res["alike"],
                     res["held"][BAG_STEPS.index(10)]))
        first.append(res["first"])
    extra["first"] = np.array(first)
    return np.array(rows), extra


def _forest_build():
    e = events()
    runs = _parallel(forest_run, [(s,) for s in SEEDS])
    out = dict(rows=np.array([r[0] for r in runs]))
    for k in ("gain", "permutation", "base"):
        out[k] = np.array([r[1][k] for r in runs])
    for k in ("tried", "tried_gain", "taken", "tried_cut", "tried_n", "first"):
        out[k] = runs[0][1][k]
    rows, _ = draw(1, 0)
    out["example_images"] = picture(e["pool_index"][rows])
    x, y, _ = sample("cells", CELLS_N, 0)
    out["mean_cells"] = np.array([x[y == 1].mean(0), x[y == 0].mean(0)])
    return out


def forest_data():
    return _cache("forest", _forest_build, (sample, bootstrap, node_scan, bag_run,
                                            gain_importance, permutation_importance,
                                            candidates, forest_run),
                  seeds=SEEDS, cells=CELLS_N, trees=BAG_TREES, mtry=MTRY, read=RF_MTRY,
                  nodes=RF_NODES, repeats=RF_REPEATS, steps=BAG_STEPS)




CORE = (top_ink, spread, two_features, events, draw, impurity, split_gain, midpoints, binned, grow,
        leaves_of, share, vote, pruning)


# ====================================================================== the studies
def _spread(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()))


def _meta(seeds, where, limitations, designs, **more):
    return dict(seeds=list(seeds), mnist=True, review_only=True,
                source="NumPy decision trees on MNIST", dataset="MNIST", stand_in=STAND_IN,
                physics_counterpart=COUNTERPART,
                features=dict(two="%s: mean ink of rows %d-%d, columns %d-%d of the picture, "
                                  "0 to 1; %s: root-mean-square distance of the ink from its "
                                  "mean row, in pixels"
                                  % (FEATURES[0], TOP_ROWS[0], TOP_ROWS[1] - 1, TOP_COLS[0],
                                     TOP_COLS[1] - 1, FEATURES[1]),
                              cells="the mean of each %d x %d square of pixels: %d cells, "
                                    "each cut at %d evenly spaced thresholds"
                                    % (BLOCK, BLOCK, CELLS, CELL_BINS - 1)),
                split="training: events drawn from a pool of %d of each class from the MNIST "
                      "training file; held-out: the other %d 4s and 9s of the training file "
                      "and all of the test file, never trained on"
                      % (POOL, len(events()["held_y"])),
                where_the_seed_enters=where, evidence=dict(EVIDENCE),
                limitations=limitations, designs_tried=designs, **more)


FEATURE_CHOICE = ("For the two features, the ink at the top centre was fixed first (it is "
                  "where a 9 is closed and a 4 open); for the second, the tilt of the ink, "
                  "its width, its total and five patches were tried with it on 3,000 "
                  "training events (held-out error of a small tree 15 to 18%) before the "
                  "vertical spread of the ink (10%), which was kept.")


def worked(n, n1, left, left1):
    """The arithmetic of one split as it is written on screen: for the node, its left
    side and its right side, the counts ``(signal, background)`` and the Gini and
    entropy impurities; the weighted impurity of the two sides and the gain, for each."""
    sides = ((n, n1), (left, left1), (n - left, n1 - left1))
    out = dict(counts=[(int(round(b)), int(round(a - b))) for a, b in sides])
    for c in ("gini", "entropy"):
        imp = [float(impurity(b / a, c)) for a, b in sides]
        after = (left * imp[1] + (n - left) * imp[2]) / n
        out[c] = dict(node=imp[0], left=imp[1], right=imp[2], after=float(after),
                      gain=float(imp[0] - after))
    return out


def tree_splits_and_impurity():
    d = splits_data()
    g = d["hand_gains"]
    f, c = np.unravel_index(int(g[0].argmax()), g[0].shape)
    hand = worked(2 * HANDFUL, HANDFUL, d["hand_left"][f, c], d["hand_left1"][f, c])
    ng = d["node_gains"]
    nf, nc = np.unravel_index(int(ng[0].argmax()), ng[0].shape)
    node = worked(2 * CELLS_N, CELLS_N, d["node_left"][nf, nc], d["node_left1"][nf, nc])
    crit = d["crit"]
    names = [CRITERION_NAMES[c] for c in CRITERIA]
    bc = d["blind_counts"]
    blind = worked(bc[0, 0], bc[0, 1], bc[1, 0], bc[1, 1])
    wrong = [int(round(min(b, a - b))) for a, b in bc]
    return Study(
        "Choosing a split",
        "One node of a tree: every threshold of two real features scored by the impurity "
        "it removes, on twelve digits and on 2,000, with three ways of scoring",
        (
            Panel("Twelve events and every threshold",
                  "xy: the two features of %d 4s (class 1) and %d 9s (class 0). cuts: the "
                  "11 thresholds of each feature, halfway between neighbouring events; "
                  "left and left1: how many events and how many 4s go left at each; "
                  "gains[criterion, feature, threshold]: Gini, entropy, error."
                  % (HANDFUL, HANDFUL), "points",
                  dict(xy=d["hand_x"], classes=d["hand_y"], cuts=d["hand_cuts"],
                       left=d["hand_left"], left1=d["hand_left1"], gains=g,
                       images=d["hand_images"])),
            Panel("2,000 events: gain against threshold",
                  "ys: the Gini gain at each of %d evenly spaced thresholds x (numbered) "
                  "of each feature, on seed 0's %d training events; grid has the "
                  "thresholds, gains all three criteria, hist the events of each class "
                  "(4s first) in bins of each feature." % (NODE_GRID, 2 * CELLS_N), "curve",
                  dict(x=np.arange(NODE_GRID), ys=ng[0], labels=list(FEATURES),
                       grid=d["node_grid"], gains=ng, left=d["node_left"],
                       left1=d["node_left1"], hist=d["node_hist"], edges=d["node_edges"],
                       xy=d["node_xy"], classes=d["node_cls"], roots=d["roots"])),
            Panel("Whole trees grown with each criterion, 49 cells",
                  "values: held-out error of a tree grown to the end with each criterion "
                  "on %d training events, mean of %d seeds. per_seed, train and leaves "
                  "the same per seed; same[seed]: over the nodes the Gini tree splits, "
                  "the share where entropy takes the same split, where the error does, "
                  "and where no threshold lowers the error."
                  % (2 * CELLS_N, len(SEEDS)), "bars",
                  dict(values=crit[:, :, 0].mean(0), labels=names, per_seed=crit[:, :, 0],
                       train=crit[:, :, 1], leaves=crit[:, :, 2], same=d["same"],
                       asked=d["asked"])),
            Panel("A node the error cannot improve",
                  "values: events and 4s in a node of the two-feature Gini tree, in its "
                  "left side and in its right side. gains: what its split removes by "
                  "Gini, entropy and error; best: the most any threshold removes.",
                  "matrix", dict(values=bc, gains=d["blind_gains"], best=d["blind_best"],
                                 where=d["blind_where"])),
        ),
        _meta(
            SEEDS,
            "numpy.random.default_rng([160, 1, seed]) draws the training events of a seed "
            "from the pool; the twelve events are the draw of seed %d" % int(d["hand_seed"]),
            ["Digits stand in for events; two features and 49 cells of one pair of "
             "digits only.",
             "The twelve events are for following the arithmetic: their best threshold "
             "is not the 2,000 events' best.",
             "On the 2,000 events %d evenly spaced thresholds are tried on each feature, "
             "not every midpoint." % NODE_GRID],
            [FEATURE_CHOICE,
             "The first twelve events drawn (seed 0) were separated completely by one "
             "threshold, which makes every impurity after the split zero; the twelve "
             "shown are the first seeded draw (seeds from %d) without such a threshold."
             % HANDFUL_SEED,
             "The criteria were to be compared on the root; all three took the same root "
             "split in every seed, so their difference is shown on a node further "
             "down: of the nodes of the two-feature Gini tree with at least %d events "
             "where no threshold lowers the error, the one whose Gini split gains most "
             "(the first such node, level by level, is 96%% pure and its gain too small "
             "to draw)." % BLIND_MIN],
            handful=dict(seed=int(d["hand_seed"]), feature=int(f),
                         threshold=float(d["hand_cuts"][f, c]), **hand),
            node=dict(events=2 * CELLS_N, feature=int(nf),
                      threshold=float(d["node_grid"][nf, nc]), **node,
                      best_other=float(ng[0, 1 - nf].max()),
                      best_other_threshold=float(d["node_grid"][1 - nf,
                                                                ng[0, 1 - nf].argmax()])),
            roots=dict(feature=[[int(v) for v in r[:, 0]] for r in d["roots"]],
                       threshold=[_floats(r[:, 1]) for r in d["roots"]],
                       same_in_every_seed=bool(np.all(d["roots"][:, :, :2]
                                                      == d["roots"][:, :1, :2]))),
            criteria={n: dict(held=_spread(crit[:, i, 0]), train=_spread(crit[:, i, 1]),
                              leaves=_spread(crit[:, i, 2])) for i, n in enumerate(names)},
            same_split=dict(entropy=_spread(d["same"][:, 0]), error=_spread(d["same"][:, 1]),
                            error_sees_nothing=_spread(d["same"][:, 2])),
            blind=dict(node=int(d["blind_where"][0]), depth=int(d["blind_where"][1]),
                       feature=int(d["blind_where"][2]), threshold=float(d["blind_where"][3]),
                       wrong=wrong, error_gain=float(d["blind_gains"][2]),
                       error_best=float(d["blind_best"][2]), **blind),
        ),
    )


def tree_pruning_and_depth():
    d, e = pruning_data(), events()
    tree, cuts = tree_from(d, "tree_"), d["cuts"]
    hb = binned(e["held_two"], cuts)
    grown = dict(leaves=leaf_count(tree), depth=int(tree["depth"].max()),
                 train=error(tree, binned(d["x"], cuts), d["y"]),
                 held=error(tree, hb, e["held_y"]))
    chosen = {}
    for name, values in (("depth", DEPTHS), ("leaf", LEAF_SIZES), ("gain", MIN_GAINS),
                         ("alpha", ALPHAS)):
        k, _ = choose(d["cv_" + name], values)
        row = d["two_" + name][0, k]
        chosen[name] = dict(setting=float(values[k]), index=k, cv=float(d["cv_" + name][k]),
                            train=float(row[0]), held=float(row[1]), leaves=int(row[2]))
    k = pruned_at(d["alphas"], chosen["alpha"]["setting"])
    chosen["alpha"]["tree"] = k
    over = {}
    for kind in KINDS:
        for name in ("depth", "leaf", "gain", "alpha"):
            a = d["%s_%s" % (kind, name)]              # seed, setting, (train, held, leaves)
            m = a[:, :, 1].mean(0)
            b = int(m.argmin())
            over["%s_%s" % (kind, name)] = dict(
                best_index=b, best=float(m[b]), best_per_seed=_floats(a[:, b, 1]),
                best_leaves=float(a[:, b, 2].mean()), end=float(m[0 if name != "depth"
                                                                 else -1]),
                end_per_seed=_floats(a[:, 0 if name != "depth" else -1, 1]))
    depth = np.array(DEPTHS)
    means = np.array([d["%s_depth" % kind][:, :, j].mean(0) for kind in KINDS
                      for j in (0, 1)])
    return Study(
        "Holding a tree back",
        "A tree grown until every leaf is pure, then limited by depth, by events per "
        "leaf, by gain and by cost-complexity pruning, the setting chosen by "
        "cross-validation",
        (
            Panel("The tree grown to the end, and the controls drawn on it",
                  "xy: the two features of %d training events (classes: 1 for a 4). "
                  "tree_*: the node table of the tree grown until nothing can be split, "
                  "cuts its thresholds; leaf_tree_* and gain_tree_*: grown with at least "
                  "%d events a leaf, and with a gain of at least %g."
                  % (2 * TWO_N, SHOWN_LEAF, SHOWN_GAIN), "points",
                  dict(xy=d["x"], classes=d["y"], cuts=cuts,
                       **{k: v for k, v in d.items() if "tree_" in k},
                       leaf_cuts=d["leaf_cuts"], gain_cuts=d["gain_cuts"])),
            Panel("That tree's error against depth",
                  "ys: training and held-out error of the drawn tree cut at depth x (%d: "
                  "grown to the end); leaves: how many leaves." % DEPTHS[-1], "curve",
                  dict(x=depth, ys=d["two_depth"][0, :, :2].T,
                       labels=["training", "held-out"], leaves=d["two_depth"][0, :, 2])),
            Panel("Pruning and cross-validation",
                  "ys: against alpha x, the five-fold cross-validated error on the "
                  "training events, and the held-out and training error of the drawn "
                  "tree pruned at that alpha. alphas, internals, links, seq: the pruning "
                  "sequence (seq: training error, held-out error, leaves); cv_*: the "
                  "cross-validated error against each control.", "curve",
                  dict(x=np.array(ALPHAS), ys=np.array([d["cv_alpha"], d["two_alpha"][0, :, 1],
                                                        d["two_alpha"][0, :, 0]]),
                       labels=["cross-validated", "held-out", "training"],
                       alphas=d["alphas"], internals=d["internals"], links=d["links"],
                       seq=d["seq"], cv_depth=d["cv_depth"], cv_leaf=d["cv_leaf"],
                       cv_gain=d["cv_gain"], leaves=d["two_alpha"][0, :, 2])),
            Panel("Every control over seeds, two features and 49 cells",
                  "ys: mean over %d seeds of the training and held-out error against "
                  "depth x, two features (%d events) then 49 cells (%d). two_* and "
                  "cells_*[seed, setting, (training, held-out, leaves)] for depth, leaf "
                  "size, gain and alpha." % (len(SEEDS), 2 * TWO_N, 2 * CELLS_N), "curve",
                  dict(x=depth, ys=means,
                       labels=["two features, training", "two features, held-out",
                               "49 cells, training", "49 cells, held-out"],
                       **{k: v for k, v in d.items() if k.startswith(("two_", "cells_"))},
                       leaf_sizes=np.array(LEAF_SIZES), min_gains=np.array(MIN_GAINS),
                       alpha_values=np.array(ALPHAS))),
        ),
        _meta(
            SEEDS,
            "numpy.random.default_rng([160, 1, seed]) draws a seed's training events; "
            "default_rng([160, 2, seed]) deals them into the five folds; the tree drawn "
            "is seed 0's",
            ["Digits stand in for events; one pair of digits.",
             "Cross-validation is shown for the drawn sample only; over seeds the "
             "settings are compared on the held-out events directly.",
             "Alpha is tried on a fixed list of %d values, not at every value where the "
             "pruned tree changes." % len(ALPHAS),
             "On the 49 cells the held-out error does not turn up with depth at this "
             "size: it flattens, and the controls gain little there."],
            [FEATURE_CHOICE,
             "The lists of depths, leaf sizes, gains and alphas were fixed before the "
             "held-out errors were looked at.",
             "150 and 250 training events of each class were tried for the drawn tree; "
             "250 was kept because its tree grown to the end is worse on held-out events "
             "in every seed, where with 150 it was in two of three."],
            grown=grown, chosen=chosen, over_seeds=over,
            shown=dict(depth=SHOWN_DEPTH, leaf=SHOWN_LEAF, gain=SHOWN_GAIN),
            sequence=dict(alphas=_floats(d["alphas"]), leaves=[int(v) for v in d["seq"][:, 2]],
                          cuts=[link_arithmetic(tree, d["internals"][k - 1], d["links"][k])
                                for k in range(1, len(d["links"]))],
                          train=_floats(d["seq"][:, 0]), held=_floats(d["seq"][:, 1]),
                          links=[int(v) for v in d["links"]]),
            folds=FOLDS,
        ),
    )


def tree_instability():
    d = instability_data()
    x = np.arange(len(INST_DEPTHS))
    two, cel = d["two_parts"], d["cells_parts"]
    roots = {}
    for kind in KINDS:
        r = d[kind + "_roots"]                         # seed, tree, (feature, threshold)
        usual = np.array([np.bincount(s[:, 0].astype(int)).argmax() for s in r])
        same = r[:, :, 0] == usual[:, None]
        roots[kind] = dict(same_feature=float(same.mean()),
                           threshold_sd=float(np.mean([s[m, 1].std() for s, m in
                                                       zip(r, same)])),
                           usual=[int(v) for v in usual])
    shown = []
    for b in range(INST_SHOWN):
        t, c = tree_from(d, "tree%d_" % b), d["cuts%d" % b]
        shown.append(dict(feature=[int(t["feature"][k]) for k in range(3)],
                          threshold=[float(c[t["feature"][k], t["cut"][k]])
                                     for k in range(3)], leaves=leaf_count(t)))
    e = events()
    return Study(
        "One tree is unstable",
        "The same recipe on resamples of the same events: different questions, "
        "different answers, and the error split into bias and variance against depth",
        (
            Panel("Four resamples of the same 500 events and their trees",
                  "xy: the two features of the training events (classes: 1 for a 4); "
                  "weights[b]: how often each is in resample b; tree<b>_* and cuts<b>: "
                  "the tree grown to the end on it.", "points",
                  dict(xy=d["x"], classes=d["y"], weights=d["weights"],
                       **{k: v for k, v in d.items() if k.startswith(("tree", "cuts"))})),
            Panel("The four trees' answers to held-out events",
                  "values[tree, event]: each tree's answer (1: a 4) for six held-out "
                  "events the four do not agree about (labels: what they are). "
                  "point_votes the same for %d held-out events at points_xy."
                  % INST_POINTS, "matrix",
                  dict(values=d["listed_votes"].astype(float),
                       labels=e["held_y"][d["listed"]], images=d["listed_images"],
                       listed_xy=e["held_two"][d["listed"]],
                       point_votes=d["point_votes"], points_xy=e["held_two"][d["points"]],
                       points_y=e["held_y"][d["points"]])),
            Panel("How often two trees disagree",
                  "ys: the share of held-out events two trees grown on different "
                  "resamples answer differently, against depth (depths; 99: grown to "
                  "the end), mean of %d seeds with %d resamples each. per_seed; "
                  "swapped_*: the same between a tree and one grown after replacing a "
                  "share (swaps) of its training events." % (len(SEEDS), INST_TREES),
                  "curve",
                  dict(x=x, ys=np.array([d["two_disagree"].mean(0),
                                         d["cells_disagree"].mean(0)]),
                       labels=["two features", "49 cells"], depths=np.array(INST_DEPTHS),
                       per_seed=np.array([d["two_disagree"], d["cells_disagree"]]),
                       swaps=np.array(INST_SWAPS),
                       swapped=np.array([d["two_swapped"], d["cells_swapped"]]),
                       errors=np.array([d["two_errors"].mean((0, 1)),
                                        d["cells_errors"].mean((0, 1))]))),
            Panel("Bias and variance against depth",
                  "ys: on the 49 cells, over %d separate training samples, the mean "
                  "squared error of a tree's answer (the share of 4s in the leaf) "
                  "against the label, and its two parts: the squared error of the "
                  "average tree and the variance around it. two has the same for the "
                  "two features; single and together the error of one tree and of the "
                  "majority of all; one[depth, sample]: every tree's answer for one "
                  "held-out event at depth 1 and grown to the end." % INST_SAMPLES, "curve",
                  dict(x=x, ys=cel[:3], labels=["squared error", "bias squared", "variance"],
                       depths=np.array(INST_DEPTHS), two=two[:3], single=cel[3],
                       together=cel[4], two_single=two[3], two_together=two[4],
                       one=d["cells_one"], one_two=d["two_one"],
                       one_image=d["listed_images"][0])),
        ),
        _meta(
            SEEDS,
            "numpy.random.default_rng([160, 1, seed]) draws a seed's training events; "
            "default_rng([160, 3, seed]) draws its resamples and the events replaced; "
            "the separate samples of the last panel are seeds %d to %d"
            % (INST_SAMPLE_SEED, INST_SAMPLE_SEED + INST_SAMPLES - 1),
            ["Digits stand in for events; one pair of digits.",
             "The separate training samples are drawn from the same pool of %d events "
             "of each class, so two of them share some events." % POOL,
             "What no tree could know (the noise) cannot be told apart from bias here: "
             "both are in the error of the average tree.",
             "The six events listed were chosen because the four trees disagree about "
             "them; the share of all held-out events like that is given."],
            [FEATURE_CHOICE,
             "The first six held-out events in a seeded order were to be listed with "
             "each tree's answer: all four trees agreed on all six, so the first three "
             "of each class that they disagree about are listed instead, and said to be.",
             "The plan expected the root question to change between resamples. It "
             "mostly does not: the feature asked first is the same in most resamples "
             "and its threshold moves a little; the questions below it change."],
            shown=shown, split_share=float(d["split_share"]),
            shown_errors=_floats(d["shown_errors"]), depths=list(INST_DEPTHS),
            disagreement={k: dict(by_depth=_floats(d[k + "_disagree"].mean(0)),
                                  end=_spread(d[k + "_disagree"][:, -1]),
                                  shallow=_spread(d[k + "_disagree"][:, 0]),
                                  single_error_end=float(d[k + "_errors"][:, :, -1].mean()))
                          for k in KINDS},
            swapped={k: dict(shares=list(INST_SWAPS),
                             events=[max(1, int(round(s_ * 2 * kind_n(k))))
                                     for s_ in INST_SWAPS],
                             changed=_floats(d[k + "_swapped"].mean(0))) for k in KINDS},
            roots=roots,
            parts={k: dict(mse=_floats(p[0]), bias2=_floats(p[1]), variance=_floats(p[2]),
                           single=_floats(p[3]), together=_floats(p[4]))
                   for k, p in (("two", two), ("cells", cel))},
            samples=INST_SAMPLES, resamples=INST_TREES,
            one_event=dict(label=int(e["held_y"][d["listed"][0]]),
                           **{name: dict(mean=float(row.mean()),
                                         bias2=float((row.mean() - e["held_y"][d["listed"][0]])
                                                     ** 2),
                                         variance=float(row.var()),
                                         mse=float(((row - e["held_y"][d["listed"][0]])
                                                    ** 2).mean()))
                              for name, row in zip(("shallow", "deep"), d["cells_one"])}),
        ),
    )


def bagging():
    d, e = bagging_data(), events()
    steps = np.array(BAG_STEPS)
    counts = np.bincount(d["picks"], minlength=len(d["picked_rows"]))
    n = len(counts)
    sigma2, cov, rho = (float(v) for v in d["var_terms"])
    gap = d["oob"][:, -1] - d["held"][:, -1]
    lag = float(np.interp(BAG_TREES / np.e, steps, d["held"].mean(0)))
    return Study(
        "Bagging",
        "Bootstrap resamples of the training events, a tree grown to the end on each, "
        "and their majority: the error and the variance of the vote against the number "
        "of trees",
        (
            Panel("One bootstrap resample, pick by pick",
                  "xy: the two features of %d training events (classes: 1 for a 4). "
                  "picks: the %d picks with replacement, in order; counts: how often "
                  "each event was picked." % (n, n), "points",
                  dict(xy=e["pool_two"][d["picked_rows"]], classes=d["picked_y"],
                       picks=d["picks"], counts=counts, images=d["picked_images"])),
            Panel("The vote of the trees over the feature plane",
                  "values[k]: the share of trees answering 4 at each of %d x %d places "
                  "of the plane (row 0 at the top) after steps[k] trees, on the two "
                  "features. maps: the first %d trees' own answers; xy, classes, "
                  "weights: the training events and each resample's counts; single and "
                  "together: held-out error of each tree and of the majority."
                  % (BAG_GRID, BAG_GRID, BAG_MAPS), "images",
                  dict(values=d["plane_shares"], steps=steps, maps=d["plane_maps"],
                       xy=d["plane_x"], classes=d["plane_y"], weights=d["plane_weights"],
                       single=d["plane_single"], together=d["plane_together"])),
            Panel("49 cells: error against the number of trees",
                  "ys: held-out error of the majority of x trees, and the out-of-bag "
                  "error of the same trees, mean of %d training samples. per_seed for "
                  "both; single[sample, tree]: each tree's held-out error."
                  % len(BAG_SEEDS), "curve",
                  dict(x=steps, ys=np.array([d["held"].mean(0), d["oob"].mean(0)]),
                       labels=["held-out", "out-of-bag"],
                       per_seed=np.array([d["held"], d["oob"]]), single=d["single"],
                       unique=d["unique"])),
            Panel("49 cells: the variance of the vote",
                  "ys: the variance, from one training sample to the next, of the share "
                  "of x trees answering 4 for a held-out event (mean over the events), "
                  "measured over %d samples, and rho sigma2 + (1 - rho) sigma2 / x with "
                  "the measured terms (sigma2, covariance, rho)." % len(BAG_SEEDS), "curve",
                  dict(x=steps, ys=np.array([d["var_measured"], d["var_formula"]]),
                       labels=["measured", "formula"], terms=d["var_terms"])),
        ),
        _meta(
            BAG_SEEDS,
            "numpy.random.default_rng([160, 1, seed]) draws a seed's training events; "
            "default_rng([160, 6, seed]) its bootstrap resamples (49 cells); "
            "default_rng([160, 5, 0]) the sixteen picks and [160, 5, 1] the resamples "
            "of the two-feature picture",
            ["Digits stand in for events; one pair of digits.",
             "The formula's rho and sigma2 are measured from all %d trees of each bag, "
             "the same trees whose vote is compared with it: at %d trees the two agree "
             "by construction, and the check is at smaller numbers."
             % (BAG_TREES, BAG_TREES),
             "The %d training samples come from one pool of %d events of each class, "
             "so they share some events." % (len(BAG_SEEDS), POOL),
             "All samples are judged on the same held-out events."],
            [FEATURE_CHOICE,
             "The variance was first to be taken over bootstrap resamples of one "
             "sample; that leaves out the part the trees share (the sample itself), "
             "which is the part that does not average away, so separate samples are "
             "used."],
            picked=dict(events=n, unique=int((counts > 0).sum()),
                        left_out=int((counts == 0).sum()), most=int(counts.max()),
                        expected_share=float(1 - (1 - 1 / n) ** n)),
            unique_share=dict(limit=float(1 - 1 / np.e), measured=float(d["unique"].mean()),
                              events=2 * CELLS_N),
            trees=BAG_TREES, steps=list(BAG_STEPS),
            plane=dict(single=float(d["plane_single"].mean()),
                       together=float(d["plane_together"][-1]), events=2 * TWO_N),
            single=_spread(d["single"].mean(1)),
            held={int(b): _spread(d["held"][:, i]) for i, b in enumerate(BAG_STEPS)},
            oob={int(b): _spread(d["oob"][:, i]) for i, b in enumerate(BAG_STEPS)},
            oob_minus_held=dict(mean=float(gap.mean()), mean_abs=float(np.abs(gap).mean()),
                                correlation=float(np.corrcoef(d["oob"][:, -1],
                                                              d["held"][:, -1])[0, 1])),
            held_at_share_of_trees=dict(trees=float(BAG_TREES / np.e), held=lag),
            variance=dict(sigma2=sigma2, covariance=cov, rho=rho, floor=cov,
                          measured=_floats(d["var_measured"]),
                          formula=_floats(d["var_formula"]),
                          worst_ratio=float(np.max(np.abs(d["var_measured"] / d["var_formula"]
                                                          - 1)))),
            tree_correlation_of_correctness=_spread(d["alike"]),
        ),
    )


def random_forest():
    d = forest_data()
    rows = d["rows"]                                   # seed, mtry, (forest, single, alike, 10)
    mean = rows.mean(0)
    best = int(mean[:, 0].argmin())
    gain, perm = d["gain"].mean(0), d["permutation"].mean(0)
    share_ = perm.clip(0) / perm.clip(0).sum()
    rank_g, rank_p = np.argsort(np.argsort(-gain)), np.argsort(np.argsort(-perm))
    top = [int(v) for v in np.argsort(-perm)[:3]]
    over = [int(v) for v in np.argsort(share_ - gain)[:3]]     # gain credits most above perm
    quiet = (gain > 0.01) & (perm < 0.001)
    one, bag, mid = MTRY.index(1), MTRY.index(CELLS), MTRY.index(RF_MTRY)
    return Study(
        "A random forest",
        "Bagging with only a random handful of the 49 cells tried at each split: trees "
        "less alike, the forest's error against the size of the handful, and which "
        "cells the forest uses",
        (
            Panel("The cells offered to the first nodes of one tree",
                  "values[node]: the %d cells drawn for the root and its two children "
                  "in the first tree of seed 0's forest (cell = 7 x row + column). "
                  "gains: the most each could remove there; taken: the cell split on; "
                  "cut: at which of the 31 thresholds; counts: events and 4s in the "
                  "node." % RF_MTRY, "matrix",
                  dict(values=d["tried"][:, :RF_MTRY].astype(float),
                       gains=d["tried_gain"][:, :RF_MTRY], taken=d["taken"],
                       cut=d["tried_cut"], counts=d["tried_n"],
                       images=d["example_images"], mean_cells=d["mean_cells"])),
            Panel("Against the number of cells tried",
                  "ys: against the cells tried at each split x (49: all, which is "
                  "bagging), mean of %d seeds: the held-out error of a forest of %d "
                  "trees, the mean held-out error of its single trees, and the mean "
                  "correlation between two trees of whether they are right about a "
                  "held-out event. per_seed[seed, x, (forest, single, correlation, "
                  "forest of 10)]; first[x, tree]: the cell each tree of seed 0 asks "
                  "about first." % (len(SEEDS), BAG_TREES), "curve",
                  dict(x=np.array(MTRY), ys=mean[:, :3].T,
                       labels=["forest", "single trees", "correlation"], per_seed=rows,
                       first=d["first"])),
            Panel("Importance by total gain",
                  "values: each cell's share of all the impurity removed by the forest "
                  "with %d cells tried, as the 7 x 7 picture, mean of %d seeds."
                  % (RF_MTRY, len(SEEDS)), "matrix",
                  dict(values=gain.reshape(SIDE, SIDE), per_seed=d["gain"])),
            Panel("Importance by permutation",
                  "values: the rise of the forest's held-out error when one cell's "
                  "readings are shuffled among the held-out events, mean of %d shuffles "
                  "and %d seeds. base: the error with nothing shuffled; rank_gain and "
                  "rank_permutation: each cell's place by either measure."
                  % (RF_REPEATS, len(SEEDS)), "matrix",
                  dict(values=perm.reshape(SIDE, SIDE), per_seed=d["permutation"],
                       base=d["base"], rank_gain=rank_g, rank_permutation=rank_p,
                       share=share_.reshape(SIDE, SIDE))),
        ),
        _meta(
            SEEDS,
            "numpy.random.default_rng([160, 1, seed]) draws a seed's training events; "
            "default_rng([160, 6, seed, cells tried]) its resamples and the cells "
            "offered to each node; default_rng([160, 8, seed]) the shuffles",
            ["Digits stand in for events; one pair of digits.",
             "If none of the cells drawn can split a node, further cells are drawn "
             "until one can; with one cell tried this happens often, since the cells "
             "at the edge of the picture are blank.",
             "Permutation importance is of the forest's majority answer on the "
             "held-out events, three shuffles a cell.",
             "Neighbouring cells carry much the same information: shuffling one leaves "
             "the forest its neighbours, which gain importance does not show."],
            [FEATURE_CHOICE,
             "The list of handful sizes was fixed before any forest was grown; the "
             "importances are read from the forest with %d cells tried, the usual "
             "square root of the number of features, not from the best." % RF_MTRY],
            mtry=list(MTRY), trees=BAG_TREES, read=RF_MTRY,
            forest={int(m): _spread(rows[:, i, 0]) for i, m in enumerate(MTRY)},
            single={int(m): _spread(rows[:, i, 1]) for i, m in enumerate(MTRY)},
            correlation={int(m): _spread(rows[:, i, 2]) for i, m in enumerate(MTRY)},
            best=dict(mtry=int(MTRY[best]), held=float(mean[best, 0]),
                      per_seed=_floats(rows[:, best, 0])),
            one=float(mean[one, 0]), all=float(mean[bag, 0]), usual=float(mean[mid, 0]),
            best_beats_ends_in_every_seed=bool(np.all(rows[:, best, 0] < rows[:, one, 0])
                                               and np.all(rows[:, best, 0] < rows[:, bag, 0])),
            first_cells={int(m): int(len(np.unique(d["first"][i])))
                         for i, m in enumerate(MTRY)},
            first_node=dict(cells=[int(v) for v in d["tried"][0, :RF_MTRY]],
                            gains=_floats(d["tried_gain"][0, :RF_MTRY]),
                            taken=int(d["taken"][0])),
            importance=dict(
                base=float(d["base"].mean()), top=top,
                top_gain=_floats(gain[top]), top_rise=_floats(perm[top]),
                correlation=float(np.corrcoef(gain, perm)[0, 1]),
                rank_correlation=float(np.corrcoef(rank_g, rank_p)[0, 1]),
                overcredited=over, overcredited_gain=_floats(gain[over]),
                overcredited_rise=_floats(perm[over]),
                overcredited_share=_floats(share_[over]),
                quiet_cells=int(quiet.sum()), quiet_gain=float(gain[quiet].sum()),
                top_share_gain=float(gain[top[0]]), top_share_permutation=float(share_[top[0]])),
        ),
    )


STUDIES = {
    "tree-splits-and-impurity": tree_splits_and_impurity,
    "tree-pruning-and-depth": tree_pruning_and_depth,
    "tree-instability": tree_instability,
    "bagging": bagging,
    "random-forest": random_forest,
}

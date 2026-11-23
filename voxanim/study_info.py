"""The experiments behind animations 172-175: one measure of uncertainty, entropy, in a
decision tree's nodes, in a trained network's weights and in a classifier's loss (see
:func:`voxanim.next_catalog.redesigned`).

* ``entropy-and-gini-by-hand``: a tree's nodes on all ten MNIST digits, their impurity
  worked out term by term, entropy ``-sum p log2 p`` in bits and Gini ``1 - sum p^2``,
  and what each counts: Gini is the chance that two digits drawn from the node differ
  (checked by drawing pairs), entropy the fewest yes/no questions a code needs on average
  to name the digit (checked with a Huffman code).
* ``entropy-down-a-tree``: a tree grown on the 49 cells of 10,000 training digits by
  information gain; the entropy left at each depth, the gains that remove it, and the
  tree read as a probability model, scored in nats on 10,000 held-out digits.
* ``weight-entropy``: every free parameter of an MNIST classifier (784-64-32-10, 52,650
  numbers) measured before and after training. The entropy of the weights as a pile of
  numbers goes *up* (they spread out and stay bell-shaped); the structure training makes
  is in how they are arranged: neighbouring pixels' weights agree (mutual information),
  and each layer's matrix concentrates on fewer directions (spectral entropy).
* ``logits-loss-and-nats``: the same trained classifier on the 10,000 test digits:
  logits, probabilities, the loss of one digit in nats and bits, the mean over the test
  set as an estimate with its standard error, what it is measured against (a uniform
  guess, ln 10; the class shares), the likelihood it is the negative log of, and
  temperature, which moves the loss and not the accuracy.

Everything is plain numpy on MNIST from seeded generators whose seeds are recorded.
Training and held-out digits are disjoint, and every loss and accuracy is a held-out one
unless it says otherwise.
"""

from __future__ import annotations

import functools
import hashlib
import heapq
import inspect
import itertools

import numpy as np

from . import mnist
from .cache import cached_npz
from .study import Panel, Study
from .study_phys_apps import cells

VERSION = 1
CLASSES = 10
LOG2_10 = float(np.log2(CLASSES))
EVIDENCE = dict(
    exact="exact arithmetic: counts, proportions, entropies and Gini worked out from them",
    measured="measured on MNIST: models trained here in numpy, scored on held-out digits",
)


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them."""
    h = hashlib.sha256()
    for fn in (*CORE, *sources):
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-info-" + name, settings,
                                                    build).items()}


def _floats(v):
    return [float(x) for x in np.asarray(v).ravel()]


# ====================================================================== impurity
def proportions(counts):
    """Counts of each class as shares that add up to one."""
    c = np.asarray(counts, float)
    return c / max(c.sum(), 1e-300)


def entropy_terms(counts, base=2.0):
    """``-p log p`` for each class (0 where ``p`` is 0), in bits (``base`` 2) or nats."""
    p = proportions(counts)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p > 0, -p * np.log(p) / np.log(base), 0.0)


def gini_terms(counts):
    """``p (1 - p)`` for each class: they add up to ``1 - sum p^2``."""
    p = proportions(counts)
    return p * (1.0 - p)


def entropy(counts, base=2.0):
    return float(entropy_terms(counts, base).sum())


def gini(counts):
    return float(1.0 - (proportions(counts) ** 2).sum())


def huffman_lengths(counts):
    """The length in bits of each class's codeword in a Huffman code for these counts
    (classes with no events get none: length 0): the shortest prefix code on average,
    whose mean length L satisfies H <= L < H + 1."""
    c = np.asarray(counts, float)
    live = [i for i in range(len(c)) if c[i] > 0]
    lengths = np.zeros(len(c), np.int64)
    if len(live) == 1:
        lengths[live[0]] = 1
        return lengths
    heap = [(c[i], k, [i]) for k, i in enumerate(live)]
    heapq.heapify(heap)
    tie = len(heap)
    while len(heap) > 1:
        a, _, xs = heapq.heappop(heap)
        b, _, ys = heapq.heappop(heap)
        for i in xs + ys:
            lengths[i] += 1
        heapq.heappush(heap, (a + b, tie, xs + ys))
        tie += 1
    return lengths


# ====================================================================== a ten-class tree
BLOCK = 4                         # 7 x 7 cells of 4 x 4 pixels: 49 readings a digit
BINS = 32                         # a cell is cut at 31 evenly spaced thresholds
CUTS = np.arange(1, BINS) / BINS
N_TRAIN = 10000                   # training digits for the tree (seeded draw from 60,000)
TREE_SEED = 172
MAX_DEPTH = 14
ALPHA = 1.0                       # Laplace smoothing of a leaf's shares, as a model


def tree_data():
    """``(xb, y, xbt, yt, images)``: 10,000 seeded training digits and the 10,000 test
    digits as binned cells (bin = how many of the 31 thresholds lie below the reading)."""
    x, y, xt, yt = mnist.load()
    rows = np.sort(np.random.default_rng(TREE_SEED).choice(len(x), N_TRAIN, replace=False))
    xb = np.searchsorted(CUTS, cells(x[rows], BLOCK), side="left").astype(np.int64)
    xbt = np.searchsorted(CUTS, cells(xt, BLOCK), side="left").astype(np.int64)
    return xb, y[rows].astype(np.int64), xbt, yt.astype(np.int64), x[rows]


def impurity_of(counts, criterion):
    """Impurity of count vectors along the last axis: entropy (bits) or Gini."""
    c = np.asarray(counts, float)
    n = np.maximum(c.sum(-1, keepdims=True), 1e-300)
    p = c / n
    if criterion == "gini":
        return 1.0 - (p * p).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.where(p > 0, p * np.log2(p), 0.0).sum(-1)


def grow_multi(xb, y, criterion="entropy", max_depth=MAX_DEPTH, min_leaf=1):
    """A ten-class tree on binned features, grown depth-first by the largest gain
    (parent impurity minus the children's, weighted by their share of the node; ties: the
    first feature, the lowest threshold). A node stays a leaf when it is pure, at
    ``max_depth``, or when no split gains. Returns the node table: ``feature`` (-1: leaf),
    ``cut`` (left: bin <= cut), ``left``, ``right``, ``depth``, ``counts`` (n, 10),
    ``gain``; and ``node_of`` (the deepest node each training digit reaches)."""
    nf = xb.shape[1]
    feature, cut, left, right, depth, counts, gain = [], [], [], [], [], [], []

    def new(d, idx):
        feature.append(-1), cut.append(0), left.append(-1), right.append(-1)
        depth.append(d), gain.append(0.0)
        counts.append(np.bincount(y[idx], minlength=CLASSES).astype(float))
        return len(feature) - 1

    root = new(0, np.arange(len(y)))
    stack = [(root, np.arange(len(y)))]
    while stack:
        t, idx = stack.pop()
        c = counts[t]
        if depth[t] >= max_depth or (c > 0).sum() < 2 or len(idx) < 2 * min_leaf:
            continue
        flat = ((np.arange(nf) * BINS + xb[idx]) * CLASSES + y[idx, None]).ravel()
        h = np.bincount(flat, minlength=nf * BINS * CLASSES).reshape(nf, BINS, CLASSES)
        lc = np.cumsum(h, 1)[:, :-1]                    # (nf, BINS-1, 10): left counts
        rc = c - lc
        nl, nr = lc.sum(-1), rc.sum(-1)
        n = c.sum()
        g = impurity_of(c, criterion) - nl / n * impurity_of(lc, criterion) \
            - nr / n * impurity_of(rc, criterion)
        g = np.where((nl >= min_leaf) & (nr >= min_leaf) & (g > 1e-12), g, -1.0)
        best = int(np.argmax(g))
        if g.ravel()[best] <= 0:
            continue
        f, k = divmod(best, BINS - 1)
        feature[t], cut[t], gain[t] = f, k, float(g.ravel()[best])
        go = xb[idx, f] <= k
        a, b = new(depth[t] + 1, idx[go]), new(depth[t] + 1, idx[~go])
        left[t], right[t] = a, b
        stack += [(b, idx[~go]), (a, idx[go])]
    return dict(feature=np.array(feature), cut=np.array(cut), left=np.array(left),
                right=np.array(right), depth=np.array(depth), counts=np.array(counts),
                gain=np.array(gain))


def path_nodes(tree, xb, max_depth):
    """The node each digit sits in when the tree is cut at ``max_depth``."""
    node = np.zeros(len(xb), np.int64)
    for _ in range(max_depth):
        f = tree["feature"][node]
        go = (f >= 0) & (tree["depth"][node] < max_depth)
        if not go.any():
            break
        side = np.where(xb[np.arange(len(xb)), np.maximum(f, 0)] <= tree["cut"][node],
                        tree["left"][node], tree["right"][node])
        node = np.where(go, side, node)
    return node


def leaf_probabilities(tree, alpha=ALPHA):
    """Each node's shares as a model: ``(counts + alpha) / (n + 10 alpha)``."""
    c = tree["counts"]
    return (c + alpha) / (c.sum(1, keepdims=True) + CLASSES * alpha)


def depth_table(tree, xb, y, xbt, yt):
    """Per depth ``d`` (0 .. MAX_DEPTH), the tree cut at ``d``: the training entropy left
    (each digit's node's entropy, averaged), its Gini, the number of leaves, and as a
    model the training and held-out cross-entropy (nats) of the smoothed shares, held-out
    accuracy, the training cross-entropy of the unsmoothed shares (which is the entropy
    left, in nats) and how many held-out digits the unsmoothed shares give probability 0."""
    P = leaf_probabilities(tree)
    R = leaf_probabilities(tree, alpha=0.0)
    H = impurity_of(tree["counts"], "entropy")
    G = impurity_of(tree["counts"], "gini")
    rows = []
    for d in range(MAX_DEPTH + 1):
        nt, nh = path_nodes(tree, xb, d), path_nodes(tree, xbt, d)
        ce_train = float(-np.log(P[nt, y]).mean())
        ce_test = float(-np.log(P[nh, yt]).mean())
        acc = float((P[nh].argmax(1) == yt).mean())
        raw = float(-np.log(R[nt, y]).mean())
        zero = int((R[nh, yt] == 0).sum())
        rows.append([d, float(H[nt].mean()), float(G[nt].mean()), len(np.unique(nt)),
                     ce_train, ce_test, acc, raw, zero])
    return np.array(rows)


def _tree_build():
    xb, y, xbt, yt, images = tree_data()
    out = {}
    for criterion in ("entropy", "gini"):
        T = grow_multi(xb, y, criterion)
        for k, v in T.items():
            out["%s_%s" % (criterion, k)] = v
        out[criterion + "_depths"] = depth_table(T, xb, y, xbt, yt)
    # the nodes drawn by hand: the root, its two children, and the deepest node on the
    # left-most path that still holds at least 40 digits of at least two classes
    T = {k[len("entropy_"):]: v for k, v in out.items() if k.startswith("entropy_")}
    node, path = 0, [0]
    while T["left"][node] >= 0:
        nxt = T["left"][node]
        if T["counts"][nxt].sum() < 40 or (T["counts"][nxt] > 0).sum() < 2:
            break
        node = nxt
        path.append(node)
    out["hand_nodes"] = np.array([0, T["left"][0], T["right"][0], path[-1]])
    out["hand_path"] = np.array(path)
    out["root_digits"] = images[:24].astype(np.float32)   # a few digits, to draw
    return out


@functools.lru_cache(maxsize=1)
def tree_runs():
    """The entropy and Gini trees on 10,000 training digits (cached)."""
    return _cache("trees", _tree_build, (tree_data, grow_multi, path_nodes, depth_table,
                                         leaf_probabilities, _tree_build),
                  n=N_TRAIN, seed=TREE_SEED, block=BLOCK, bins=BINS, depth=MAX_DEPTH,
                  alpha=ALPHA)


def tree_of(criterion="entropy"):
    r = tree_runs()
    return {k[len(criterion) + 1:]: r[k] for k in r if k.startswith(criterion + "_")}


PAIRS = 200000                    # random pairs drawn from a node to measure Gini
PAIR_SEED = 7


def pairs_differ(counts, n=PAIRS, seed=PAIR_SEED):
    """The share of ``n`` random pairs (drawn with replacement) from a node with these
    counts whose two digits differ, and its standard error."""
    rng = np.random.default_rng(seed)
    p = proportions(counts)
    a = rng.choice(len(p), n, p=p)
    b = rng.choice(len(p), n, p=p)
    share = float((a != b).mean())
    return share, float(np.sqrt(share * (1 - share) / n))


# ====================================================================== a classifier's weights
SIZES = (784, 64, 32, 10)
RATE = 1e-3
BATCH = 128
EPOCHS = 5
SEEDS = (0, 1, 2)
SNAPS = 26                        # parameter snapshots per run, step 0 included
HIST_BIN = 0.01                   # bin width of the weights' histogram entropy
MI_BINS = 16                      # equal-count bins of the neighbour mutual information
WEIGHT_SEED = 175


def init_mlp(seed):
    """He-normal weights, zero biases, float32."""
    rng = np.random.default_rng([WEIGHT_SEED, int(seed)])
    P = {}
    for k, (a, b) in enumerate(itertools.pairwise(SIZES)):
        P["W%d" % k] = rng.normal(0.0, np.sqrt(2.0 / a), (a, b)).astype(np.float32)
        P["b%d" % k] = np.zeros(b, np.float32)
    return P


def logits(P, x):
    """The classifier's ten scores for each image: ReLU, ReLU, then a linear layer."""
    h = x
    for k in range(len(SIZES) - 1):
        h = h @ P["W%d" % k] + P["b%d" % k]
        if k < len(SIZES) - 2:
            h = np.maximum(h, 0)
    return h


def log_softmax(z):
    z = np.asarray(z, np.float64)
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def mlp_grads(P, x, y):
    """Mean cross-entropy (nats) of a batch and its gradient for every parameter."""
    acts = [x]
    for k in range(len(SIZES) - 1):
        z = acts[-1] @ P["W%d" % k] + P["b%d" % k]
        acts.append(np.maximum(z, 0) if k < len(SIZES) - 2 else z)
    lp = log_softmax(acts[-1])
    rows = np.arange(len(y))
    loss = float(-lp[rows, y].mean())
    d = np.exp(lp)
    d[rows, y] -= 1.0
    d = (d / len(y)).astype(np.float32)
    G = {}
    for k in reversed(range(len(SIZES) - 1)):
        G["W%d" % k] = acts[k].T @ d
        G["b%d" % k] = d.sum(0)
        if k:
            d = (d @ P["W%d" % k].T) * (acts[k] > 0)
    return loss, G


def flat(P):
    """Every free parameter, weights and biases, in one float64 vector."""
    return np.concatenate([np.asarray(P[k], np.float64).ravel() for k in sorted(P)])


def histogram_entropy(v, width=HIST_BIN):
    """Entropy in bits of the values ``v`` counted in bins of ``width``: the pile of
    numbers, with no regard to where each sits."""
    c = np.unique(np.floor(np.asarray(v) / width), return_counts=True)[1]
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


def spectral_entropy(W):
    """Entropy in bits of a matrix's squared singular values as shares; ``2**H`` is its
    effective number of directions (its effective rank)."""
    s = np.linalg.svd(np.asarray(W, np.float64), compute_uv=False)
    q = s ** 2 / (s ** 2).sum()
    q = q[q > 0]
    return float(-(q * np.log2(q)).sum())


def neighbour_pairs(W0):
    """First-layer weights of horizontally neighbouring pixels, for every hidden unit."""
    img = np.asarray(W0, np.float64).T.reshape(-1, 28, 28)
    return img[:, :, :-1].ravel(), img[:, :, 1:].ravel()


def mutual_information(a, b, bins=MI_BINS):
    """Mutual information in bits between ``a`` and ``b``, each cut into ``bins``
    equal-count bins (so each alone has entropy log2 bins, whatever its spread). Two
    independent samples of this size read about 0.003 bits."""
    def q(v):
        return np.searchsorted(np.quantile(v, np.linspace(0, 1, bins + 1)[1:-1]), v)
    J = np.zeros((bins, bins))
    np.add.at(J, (q(a), q(b)), 1.0)
    J /= J.sum()
    pa, pb = J.sum(1), J.sum(0)
    nz = J > 0
    return float((J[nz] * np.log2(J[nz] / np.outer(pa, pb)[nz])).sum())


def measures(P):
    """Everything measured on one set of parameters."""
    v = flat(P)
    a, b = neighbour_pairs(P["W0"])
    out = [histogram_entropy(v), float(v.std()), float(np.corrcoef(a, b)[0, 1]),
           mutual_information(a, b)]
    out += [spectral_entropy(P["W%d" % k]) for k in range(len(SIZES) - 1)]
    return out


MEASURES = ("histogram entropy", "spread", "neighbour correlation", "neighbour information",
            "spectral entropy W0", "spectral entropy W1", "spectral entropy W2")


def train_mlp(seed):
    """One run: snapshots of every parameter's measures, held-out accuracy and loss, and
    the parameters before and after."""
    from .optim import Adam

    x, y, xt, yt = mnist.load()
    x, xt = x.astype(np.float32), xt.astype(np.float32)
    P = init_mlp(seed)
    P0 = {k: v.copy() for k, v in P.items()}
    opt = Adam(P, np.float32(RATE))
    rng = np.random.default_rng([WEIGHT_SEED, int(seed), 1])
    per = len(x) // BATCH
    steps = EPOCHS * per
    at = np.unique(np.round(np.linspace(0, steps, SNAPS)).astype(np.int64))
    rows, step = [], 0

    def snap():
        lp = log_softmax(logits(P, xt))
        rows.append([step] + measures(P) + [float((lp.argmax(1) == yt).mean()),
                                            float(-lp[np.arange(len(yt)), yt].mean())])
    snap()
    for _ in range(EPOCHS):
        order = rng.permutation(len(x))
        for b in range(per):
            idx = order[b * BATCH:(b + 1) * BATCH]
            opt.step(P, mlp_grads(P, x[idx], y[idx])[1])
            step += 1
            if step in at:
                snap()
    return np.array(rows), P0, P


def _weights_build():
    out = {}
    for seed in SEEDS:
        rows, P0, P = train_mlp(seed)
        out["run_%d" % seed] = rows
        if seed == SEEDS[0]:
            for k in P:
                out["before_" + k] = P0[k]
                out["after_" + k] = P[k]
    x = mnist.load()[0]
    out["never_lit"] = np.flatnonzero(x.max(0) == 0)
    return out


@functools.lru_cache(maxsize=1)
def weight_runs():
    """Three seeds of the classifier, measured through training (cached)."""
    return _cache("weights", _weights_build, (init_mlp, logits, log_softmax, mlp_grads, flat,
                                              histogram_entropy, spectral_entropy,
                                              neighbour_pairs, mutual_information, measures,
                                              train_mlp, _weights_build),
                  sizes=SIZES, rate=RATE, batch=BATCH, epochs=EPOCHS, seeds=SEEDS,
                  snaps=SNAPS, bin=HIST_BIN, mi_bins=MI_BINS, seed=WEIGHT_SEED)


def params(which="after"):
    """The seed-0 classifier's parameters ``before`` or ``after`` training."""
    r = weight_runs()
    return {k[len(which) + 1:]: r[k] for k in r if k.startswith(which + "_")}


# ====================================================================== logits and loss
TEMPERATURES = np.round(np.geomspace(0.25, 4.0, 33), 4)
SHOWN_DIGIT = 0                   # the first test digit is the one followed
HALF = 5000                       # temperature is chosen on the first half, scored on the second


def test_scores():
    """``(z, lp, yt, prior)``: the trained classifier's logits and log-probabilities on
    the 10,000 test digits, their labels, and the training set's class shares."""
    _, y, xt, yt = mnist.load()
    z = logits(params("after"), xt.astype(np.float32)).astype(np.float64)
    return z, log_softmax(z), yt.astype(np.int64), np.bincount(y, minlength=CLASSES) / len(y)


def per_digit_loss(lp, y):
    """``-ln p(true digit)`` for each digit: its loss in nats."""
    return -lp[np.arange(len(y)), y]


# ====================================================================== the studies
CORE = (proportions, entropy_terms, gini_terms, impurity_of)


def _shares(counts):
    return [round(float(v), 6) for v in proportions(counts)]


def entropy_and_gini_by_hand():
    T = tree_of("entropy")
    hand = [int(v) for v in tree_runs()["hand_nodes"]]
    nodes = []
    for t in hand:
        c = T["counts"][t]
        L = huffman_lengths(c)
        share, se = pairs_differ(c)
        nodes.append(dict(node=t, depth=int(T["depth"][t]), counts=_floats(c),
                          n=float(c.sum()), shares=_shares(c),
                          entropy_terms=_floats(entropy_terms(c)),
                          gini_terms=_floats(gini_terms(c)), entropy=entropy(c),
                          entropy_nats=entropy(c, np.e), gini=gini(c),
                          huffman_lengths=[int(v) for v in L],
                          huffman_mean=float((proportions(c) * L).sum()),
                          pairs_differ=share, pairs_error=se,
                          feature=int(T["feature"][t]), cut=int(T["cut"][t]),
                          gain=float(T["gain"][t])))
    root, a, b, _ = nodes
    p = np.linspace(0.0, 1.0, 101)
    two = np.stack([p, 1 - p], 1)
    curves = np.array([impurity_of(two * 100, "entropy"), impurity_of(two * 100, "gini") * 2,
                       np.minimum(p, 1 - p) * 2])
    weighted = (a["n"] * a["entropy"] + b["n"] * b["entropy"]) / root["n"]
    panels = (
        Panel("Entropy at the root, term by term", "-p log2 p for each digit among the %d "
              "training digits; they add up to %.3f bits of a possible %.3f"
              % (root["n"], root["entropy"], LOG2_10), "bars",
              dict(values=np.array(root["entropy_terms"]), labels=[str(k) for k in
                                                                    range(CLASSES)],
                   gini=np.array(root["gini_terms"]), shares=np.array(root["shares"]))),
        Panel("After the first question", "entropy and Gini of the root, of its two "
              "children and of a node further down", "bars",
              dict(values=np.array([n["entropy"] for n in nodes]),
                   gini=np.array([n["gini"] for n in nodes]),
                   labels=["root", "left", "right", "deeper"],
                   n=np.array([n["n"] for n in nodes]))),
        Panel("Two classes: the three measures", "entropy (bits), twice Gini and twice the "
              "misclassification error against the share of one class: each 0 when pure, "
              "1 at an even mix", "curve",
              dict(x=p, ys=curves, labels=["entropy", "2 x Gini", "2 x error"])),
        Panel("What the numbers count", "Gini against the share of %d random pairs that "
              "differ; entropy against the mean length of a Huffman code, per node"
              % PAIRS, "bars",
              dict(values=np.array([v for n in nodes for v in (n["gini"], n["pairs_differ"],
                                                               n["entropy"],
                                                               n["huffman_mean"])]),
                   labels=["%s %s" % (w, k) for w in ("root", "left", "right", "deeper")
                           for k in ("Gini", "pairs", "entropy", "Huffman")])),
    )
    return Study(
        "Entropy and Gini by hand", "How impure a tree's node is, worked out from its "
        "counts", panels,
        dict(source="exact arithmetic from the counts of real MNIST training digits; pairs "
                    "drawn from a seeded generator",
             dataset="MNIST", evidence=EVIDENCE, nodes=nodes, classes=CLASSES,
             max_entropy=LOG2_10, max_gini=1 - 1 / CLASSES, n_train=N_TRAIN,
             pairs=PAIRS, pair_seed=PAIR_SEED, weighted_children_entropy=weighted,
             first_gain=root["gain"], block=BLOCK, bins=BINS, tree_seed=TREE_SEED,
             nats_vs_gini=[dict(nats=n["entropy_nats"], gini=n["gini"]) for n in nodes],
             limitations=["one tree, grown on one seeded draw of 10,000 training digits",
                          "the Huffman code is the best code for the node's own counts: "
                          "on new digits it would do a little worse"]))


def entropy_down_a_tree():
    T, Tg = tree_of("entropy"), tree_of("gini")
    D, Dg = T["depths"], Tg["depths"]
    internal = T["feature"] >= 0
    n0 = T["counts"][0].sum()
    weighted_gain = T["gain"] * T["counts"].sum(1) / n0
    by_depth = np.array([weighted_gain[internal & (T["depth"] == d)].sum()
                         for d in range(MAX_DEPTH)])
    best = int(np.argmin(D[:, 5]))
    panels = (
        Panel("Entropy left at each depth", "the training digits' mean node entropy (bits) "
              "with the tree cut at each depth, entropy and Gini trees", "curve",
              dict(x=D[:, 0], ys=np.array([D[:, 1], Dg[:, 1]]),
                   labels=["entropy tree", "Gini tree"])),
        Panel("Information gained at each depth", "the gains of a depth's splits, each "
              "weighted by its node's share of the digits: they add up to what has gone",
              "bars", dict(values=by_depth, labels=[str(d) for d in range(MAX_DEPTH)])),
        Panel("The tree as a model, in nats", "cross-entropy of the leaves' smoothed "
              "shares on the training digits and on 10,000 held-out digits", "curve",
              dict(x=D[:, 0], ys=np.array([D[:, 4], D[:, 5]]),
                   labels=["training", "held out"])),
        Panel("Held-out accuracy", "the share of held-out digits the tree names correctly, "
              "cut at each depth", "curve",
              dict(x=D[:, 0], ys=np.array([D[:, 6], Dg[:, 6]]),
                   labels=["entropy tree", "Gini tree"])),
    )
    return Study(
        "Entropy down a tree", "Where a tree's information gain goes, and what is left",
        panels,
        dict(source="trees grown here in numpy on the 49 cells of 10,000 seeded training "
                    "MNIST digits; held-out figures on the 10,000 official test digits",
             dataset="MNIST", evidence=EVIDENCE, n_train=N_TRAIN, tree_seed=TREE_SEED,
             max_depth=MAX_DEPTH, alpha=ALPHA, root_entropy=float(D[0, 1]),
             nodes=int(len(T["feature"])), leaves=int((~internal).sum()),
             depths=[dict(depth=int(r[0]), entropy=float(r[1]), gini=float(r[2]),
                          leaves=int(r[3]), train_nats=float(r[4]), test_nats=float(r[5]),
                          test_accuracy=float(r[6]), train_nats_unsmoothed=float(r[7]),
                          test_zero_probability=int(r[8])) for r in D],
             gini_depths=[dict(depth=int(r[0]), entropy=float(r[1]),
                               test_nats=float(r[5]), test_accuracy=float(r[6]))
                          for r in Dg],
             gain_by_depth=_floats(by_depth), total_gain=float(by_depth.sum()),
             best_depth=best, best_test_nats=float(D[best, 5]),
             same_root=bool(T["feature"][0] == Tg["feature"][0]
                            and T["cut"][0] == Tg["cut"][0]),
             limitations=["a leaf's shares are smoothed (one imaginary digit of each kind) "
                          "so that a held-out digit never gets probability zero",
                          "one tree on one draw of training digits"]))


def weight_entropy():
    r = weight_runs()
    runs = np.array([r["run_%d" % s] for s in SEEDS])      # (seeds, snaps, columns)
    steps = runs[0, :, 0]
    col = {name: 1 + i for i, name in enumerate(MEASURES)}
    acc_col, loss_col = 1 + len(MEASURES), 2 + len(MEASURES)
    before, after = runs[:, 0], runs[:, -1]

    def both(name):
        c = col[name]
        return dict(before=_floats(before[:, c]), after=_floats(after[:, c]))
    P0, P1 = params("before"), params("after")
    never = r["never_lit"]
    W0b, W0a = P0["W0"], P1["W0"]
    unchanged = int(np.all(W0b[never] == W0a[never], axis=1).sum())
    panels = (
        Panel("The weights as a pile of numbers", "histogram entropy of all %d free "
              "parameters (bins of %g), bits each, through training, three seeds"
              % (flat(P0).size, HIST_BIN), "curve",
              dict(x=steps, ys=runs[:, :, col["histogram entropy"]],
                   labels=["seed %d" % s for s in SEEDS],
                   spread=runs[:, :, col["spread"]])),
        Panel("Neighbouring pixels agree", "mutual information (bits) between the first-layer "
              "weights of neighbouring pixels, three seeds", "curve",
              dict(x=steps, ys=runs[:, :, col["neighbour information"]],
                   labels=["seed %d" % s for s in SEEDS],
                   correlation=runs[:, :, col["neighbour correlation"]])),
        Panel("Fewer directions", "spectral entropy (bits) of each layer's weight matrix, "
              "seed 0", "curve",
              dict(x=steps, ys=runs[0][:, [col["spectral entropy W%d" % k]
                                          for k in range(3)]].T,
                   labels=["W0", "W1", "W2"])),
        Panel("First-layer weights before and after", "the 784 weights of eight hidden "
              "units as 28 x 28 pictures, seed 0", "images",
              dict(values=np.concatenate([W0b.T[:8], W0a.T[:8]]).reshape(16, 28, 28),
                   labels=["before"] * 8 + ["after"] * 8)),
    )
    n_params = int(flat(P0).size)
    return Study(
        "The entropy of a network's weights", "Every free parameter of a classifier, "
        "before and after training", panels,
        dict(source="a 784-64-32-10 classifier trained here in numpy on MNIST, three seeds; "
                    "accuracy on the 10,000 test digits",
             dataset="MNIST", evidence=EVIDENCE, sizes=list(SIZES), parameters=n_params,
             weights=int(sum(a * b for a, b in itertools.pairwise(SIZES))),
             biases=int(sum(SIZES[1:])), epochs=EPOCHS, batch=BATCH, learning_rate=RATE,
             optimiser="Adam", seeds=list(SEEDS), steps=_floats(steps),
             histogram_bin=HIST_BIN, mi_bins=MI_BINS,
             histogram_entropy=both("histogram entropy"), spread=both("spread"),
             neighbour_correlation=both("neighbour correlation"),
             neighbour_information=both("neighbour information"),
             spectral_entropy={("W%d" % k): both("spectral entropy W%d" % k)
                               for k in range(3)},
             accuracy=dict(before=_floats(before[:, acc_col]),
                           after=_floats(after[:, acc_col])),
             test_nats=dict(before=_floats(before[:, loss_col]),
                            after=_floats(after[:, loss_col])),
             never_lit=int(len(never)), never_lit_unchanged=unchanged,
             # a pile of numbers spread wider by a factor s has log2(s) more bits in fixed
             # bins: how much of the rise is the spread alone
             spread_bits=_floats(np.log2(after[:, col["spread"]] / before[:, col["spread"]])),
             limitations=["entropies of continuous numbers are estimated from bins: the "
                          "histogram entropy depends on the bin width, which is fixed, so "
                          "only differences mean anything",
                          "mutual information from 16 equal-count bins; two independent "
                          "samples of this size read about 0.003 bits",
                          "one architecture, three seeds"]))


def logits_loss_and_nats():
    z, lp, yt, prior = test_scores()
    loss = per_digit_loss(lp, yt)
    n = len(yt)
    mean, se = float(loss.mean()), float(loss.std(ddof=1) / np.sqrt(n))
    order = np.sort(loss)[::-1]
    worst = [float(order[:k].sum() / order.sum()) for k in (10, 100, 1000)]
    uniform = float(np.log(CLASSES))
    prior_nats = float(-np.log(prior[yt]).mean())
    acc = float((lp.argmax(1) == yt).mean())
    # temperature: chosen on the first half of the test digits, scored on the second
    first = [float(per_digit_loss(log_softmax(z[:HALF] / T), yt[:HALF]).mean())
             for T in TEMPERATURES]
    second = [float(per_digit_loss(log_softmax(z[HALF:] / T), yt[HALF:]).mean())
              for T in TEMPERATURES]
    best_T = float(TEMPERATURES[int(np.argmin(first))])
    i = SHOWN_DIGIT
    zi, pi = z[i], np.exp(lp[i])
    top2 = np.argsort(-zi)[:2]
    w = int(np.argmax(loss))
    edges = np.geomspace(1e-6, 30, 41)
    hist = np.histogram(np.clip(loss, edges[0], edges[-1]), edges)[0]
    panels = (
        Panel("One digit: logits to probabilities", "the ten logits of test digit %d and "
              "the softmax of them" % i, "bars",
              dict(values=zi, labels=[str(k) for k in range(CLASSES)], probabilities=pi)),
        Panel("Every test digit's loss", "how many of the 10,000 test digits have each loss "
              "-ln p (nats, log scale)", "curve",
              dict(x=np.sqrt(edges[:-1] * edges[1:]), ys=hist[None].astype(float),
                   labels=["test digits"])),
        Panel("What the mean is measured against", "mean loss in nats: a uniform guess, the "
              "class shares, the trained classifier", "bars",
              dict(values=np.array([uniform, prior_nats, mean]),
                   labels=["uniform", "class shares", "classifier"])),
        Panel("Temperature moves the loss, not the accuracy", "mean loss of the second "
              "half of the test digits with the logits divided by T", "curve",
              dict(x=TEMPERATURES.astype(float), ys=np.array([first, second]),
                   labels=["first half (chooses T)", "second half"])),
    )
    return Study(
        "Logits, loss and nats", "What a classifier's loss is, in statistical terms",
        panels,
        dict(source="the 784-64-32-10 classifier trained here (seed 0) on MNIST; every "
                    "figure on the 10,000 test digits",
             dataset="MNIST", evidence=EVIDENCE, n=n, shown_digit=i,
             shown_label=int(yt[i]), shown_logits=_floats(zi), shown_probabilities=_floats(pi),
             shown_loss_nats=float(loss[i]), shown_top2=[int(v) for v in top2],
             shown_top2_gap=float(zi[top2[0]] - zi[top2[1]]),
             shown_top2_log_ratio=float(lp[i, top2[0]] - lp[i, top2[1]]),
             worst_digit=w, worst_label=int(yt[w]), worst_named=int(lp[w].argmax()),
             worst_probability=float(np.exp(lp[w, yt[w]])), worst_loss_nats=float(loss[w]),
             mean_nats=mean, mean_bits=mean / np.log(2), standard_error=se,
             median_nats=float(np.median(loss)), accuracy=acc,
             worst_shares=dict(zip(("10", "100", "1000"), worst)),
             uniform_nats=uniform, uniform_bits=LOG2_10, prior_nats=prior_nats,
             perplexity=float(np.exp(mean)),
             total_bits=float(loss.sum() / np.log(2)), raw_bits=float(n * LOG2_10),
             log_likelihood=float(-loss.sum()),
             temperatures=_floats(TEMPERATURES), best_temperature=best_T,
             loss_first_half=first, loss_second_half=second, half=HALF,
             second_half_at_1=float(second[int(np.argmin(np.abs(TEMPERATURES - 1.0)))]),
             second_half_at_best=float(second[int(np.argmin(first))]),
             limitations=["the classifier is small (97% accurate): its loss is mostly a "
                          "few confident mistakes",
                          "temperature is chosen on one half of the test digits and only "
                          "scored on the other"]))


STUDIES = {
    "entropy-and-gini-by-hand": entropy_and_gini_by_hand,
    "entropy-down-a-tree": entropy_down_a_tree,
    "weight-entropy": weight_entropy,
    "logits-loss-and-nats": logits_loss_and_nats,
}

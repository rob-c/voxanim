"""The experiments behind animations 106-109: a class label as a learned vector.

A *label embedding* is one learned vector per class label: a row of a small table,
looked up by the label, used where a one-hot marker would be. Four experiments on
MNIST's ten digit labels, each a fair comparison over several seeds:

* ``label-embedding``: a table trained by a task that has nothing but the label to go
  on (draw the digit from its label's row), at five table widths and five seeds. Which
  labels end up near each other is compared with two things measured without the table:
  the distance between the digits' mean images and how an independent classifier
  confuses them. The untrained table is the control, and the size of the table's random
  start is swept too (a table started large keeps its start).
* ``label-embedding-classifier``: the last layer of a classifier as a table of label
  vectors. Four ways to treat that table (ordinary dot-product scores with one-hot
  targets; the same with label smoothing; cosine scores against a table trained with
  the network; cosine scores against the same table left at its random start), four
  feature widths, five seeds.
* ``label-embedding-autoencoder``: an autoencoder whose decoder is given the label's
  row beside the code, one whose encoder is given it too, the plain autoencoder of the
  same size, and a control given a random label's row; four code sizes, three seeds.
* ``label-embedding-generation``: the same trained autoencoders (trained once, shared),
  asked to decode an image's code with another label's row.

Every model trains on one split of the MNIST training images and is measured on
official test images, which the independent classifier used as a judge
(:func:`voxanim.mnist.train`) never saw either. Everything the drawings need is
recorded here: :mod:`voxanim.mech_labels` computes nothing but layout.
"""

from __future__ import annotations

import functools
import hashlib
import inspect

import numpy as np

from .autoencoder import dense_init, recon_grad, sigmoid
from .cache import cached_npz
from .minibatch import minibatches
from .optim import Adam
from .study import Panel, Study

VERSION = 1
SEED = 17                         # of the data split
N_TRAIN = 20000                   # training images (of the first 50,000 training images)
N_TEST = 5000                     # held-out images (of the official test images)
PAIRS = np.triu_indices(10, 1)    # the 45 pairs of labels


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them (an edit to one experiment re-runs that experiment only)."""
    h = hashlib.sha256()
    for fn in sources:
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, seed=SEED, n_train=N_TRAIN, n_test=N_TEST,
                    source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-labels-" + name, settings,
                                                    build).items()}


def _parallel(fn, jobs):
    from .study_generative import parallel

    return parallel(fn, jobs)


@functools.lru_cache(maxsize=1)
def data():
    """``(x, y, xt, yt, meta)``: the training split and the held-out split, 28 x 28."""
    from . import realdata

    x, y, m = realdata.mnist_split(N_TRAIN, "train", 28, SEED)
    xt, yt, mt = realdata.mnist_split(N_TEST, "test", 28, SEED)
    keep = lambda d: {k: v for k, v in d.items() if k != "example_indices"}  # noqa: E731
    return x, y, xt, yt, dict(train=keep(m), held_out=keep(mt))


@functools.lru_cache(maxsize=1)
def judge():
    """The independent classifier: the repo's 784-64-32-10 network, trained on the
    official training images. Returns ``(read, accuracy)``; ``read(x)`` gives its ten
    probabilities per image."""
    from . import mnist

    params, acc = mnist.train()

    def read(x):
        return mnist._forward(params, np.asarray(x, float).reshape(-1, 784))[2]
    return read, acc


def ranks(v):
    """Ranks of ``v`` (ties share their mean rank)."""
    v = np.asarray(v, float)
    order = np.argsort(v, kind="stable")
    r = np.empty(len(v))
    r[order] = np.arange(len(v), dtype=float)
    for value in np.unique(v):
        same = v == value
        if same.sum() > 1:
            r[same] = r[same].mean()
    return r


def spearman(a, b):
    """Rank correlation of two vectors (0 when either is constant)."""
    ra, rb = ranks(a), ranks(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def distances(table):
    """The 10 x 10 distances between a table's rows, scaled so that the 45 pairs
    average 1."""
    t = np.asarray(table, float)
    d = np.sqrt(((t[:, None] - t[None]) ** 2).sum(-1))
    return d / max(d[PAIRS].mean(), 1e-12)


def cosines(table):
    """Cosines between a table's rows after their mean row is taken out (a softmax
    does not see a vector added to every row)."""
    t = np.asarray(table, float)
    t = t - t.mean(0)
    t = t / np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    return t @ t.T


def references():
    """What is known about the ten digits without any label table: the distance between
    their mean training images, and how the judge confuses them on held-out images
    (the mean probability it gives digit b on images of digit a, both ways)."""
    x, y, xt, yt, _ = data()
    means = np.array([x[y == c].mean(0) for c in range(10)])
    read, _ = judge()
    p = read(xt)
    soft = np.array([p[yt == c].mean(0) for c in range(10)])
    return means, distances(means), (soft + soft.T) / 2


# ====================================================================== 106: the table
TAB_WIDTHS = (2, 4, 8, 16, 32)
TAB_SEEDS = (17, 18, 19, 20, 21)
TAB_HIDDEN = 64
TAB_EPOCHS, TAB_BATCH, TAB_LR = 3, 64, 2e-3
TAB_INIT = 0.01                   # the table starts as normal draws of this size
TAB_STARTS = (1.0, 0.1, 0.01, 0.001)      # the start sizes compared at TAB_START_WIDTH
TAB_START_WIDTH = 8
TAB_SNAPS = 48                    # table states kept along each run


def tab_draw(P, rows):
    """The picture drawn from label rows ``rows`` ``(n, width)``."""
    h = np.maximum(rows @ P["W0"] + P["b0"], 0)
    return h, sigmoid(h @ P["W1"] + P["b1"])


def tab_run(width, seed, init=TAB_INIT):
    """One table of ``width`` trained to draw each training image from its label. The
    same seed draws the same start whatever ``init``, only its size differs."""
    x, y, xt, yt, _ = data()
    rng = np.random.default_rng([seed, width])
    P = {"E": init * rng.normal(0, 1.0, (10, width))}
    dense_init(P, "0", width, TAB_HIDDEN, rng)
    dense_init(P, "1", TAB_HIDDEN, 784, rng)
    start = P["E"].copy()
    total = TAB_EPOCHS * int(np.ceil(len(x) / TAB_BATCH))
    keep = np.unique(np.round(np.geomspace(1, total, TAB_SNAPS - 1)).astype(int))
    snaps, steps = [start.copy()], [0]

    def state():
        drawn = tab_draw(P, P["E"])[1]
        return drawn.astype(np.float32), np.mean((drawn[yt] - xt) ** 2)
    states = [state()]
    opt = Adam(P, TAB_LR)
    for step, idx in enumerate(minibatches(rng, len(x), TAB_EPOCHS, TAB_BATCH), 1):
        rows = P["E"][y[idx]]
        h, r = tab_draw(P, rows)
        d = recon_grad(r, x[idx])
        g = {"W1": h.T @ d, "b1": d.sum(0)}
        d = (d @ P["W1"].T) * (h > 0)
        g["W0"], g["b0"] = rows.T @ d, d.sum(0)
        g["E"] = np.zeros_like(P["E"])
        np.add.at(g["E"], y[idx], d @ P["W0"].T)
        opt.step(P, g)
        if step in keep:
            snaps.append(P["E"].copy())
            steps.append(step)
            states.append(state())
    while len(snaps) < TAB_SNAPS:                     # short runs repeat their last state
        snaps.append(P["E"].copy())
        steps.append(total)
        states.append(state())
    drawn = tab_draw(P, P["E"])[1]
    return dict(start=start, table=P["E"], snaps=np.array(snaps), steps=np.array(steps),
                drawn=drawn, loss=np.mean((drawn[yt] - xt) ** 2),
                curve=np.array([v for _, v in states]),
                drawn_snaps=np.array([a for a, _ in states]))


def _tab_build():
    _x, _y, xt, yt, _ = data()
    means, mean_dist, soft = references()
    jobs = [(w, s, TAB_INIT) for w in TAB_WIDTHS for s in TAB_SEEDS]
    sweep = [(TAB_START_WIDTH, s, a) for a in TAB_STARTS for s in TAB_SEEDS]
    runs = _parallel(tab_run, jobs + sweep)
    runs, swept = runs[:len(jobs)], runs[len(jobs):]
    nw, ns = len(TAB_WIDTHS), len(TAB_SEEDS)
    out = dict(
        means=means, mean_dist=mean_dist, soft=soft,
        floor=np.mean((means[yt] - xt) ** 2),         # the best any label-only drawing can do
        one_hot_dist=distances(np.eye(10)),
        loss=np.array([r["loss"] for r in runs]).reshape(nw, ns),
        dist=np.array([distances(r["table"]) for r in runs]).reshape(nw, ns, 10, 10),
        start_dist=np.array([distances(r["start"]) for r in runs]).reshape(nw, ns, 10, 10),
        sweep_dist=np.array([distances(r["table"]) for r in swept]).reshape(
            len(TAB_STARTS), ns, 10, 10),
        sweep_start_dist=np.array([distances(r["start"]) for r in swept]).reshape(
            len(TAB_STARTS), ns, 10, 10),
        sweep_loss=np.array([r["loss"] for r in swept]).reshape(len(TAB_STARTS), ns),
    )
    for i, w in enumerate(TAB_WIDTHS):
        out["table_%d" % w] = np.array([r["table"] for r in runs[i * ns:(i + 1) * ns]])
    two = runs[:ns] if TAB_WIDTHS[0] == 2 else []
    out["snaps_2"] = np.array([r["snaps"] for r in two])          # (seed, snap, 10, 2)
    out["steps_2"] = np.array([r["steps"] for r in two])
    out["curve_2"] = np.array([r["curve"] for r in two])          # held-out error at each
    out["drawn_snaps_2"] = two[0]["drawn_snaps"]                  # first seed: what the rows draw
    out["curve_8"] = runs[TAB_WIDTHS.index(8) * ns]["curve"]
    out["steps_8"] = runs[TAB_WIDTHS.index(8) * ns]["steps"]
    out["drawn"] = runs[TAB_WIDTHS.index(8) * ns]["drawn"]        # width 8, first seed
    out["drawn_2"] = runs[0]["drawn"]                             # width 2, first seed
    return out


def table_data():
    """The table experiment's arrays (cached)."""
    return _cache("table", _tab_build, (_tab_build, tab_run, tab_draw, references, distances),
                  widths=TAB_WIDTHS, seeds=TAB_SEEDS, hidden=TAB_HIDDEN, epochs=TAB_EPOCHS,
                  batch=TAB_BATCH, lr=TAB_LR, init=TAB_INIT, snaps=TAB_SNAPS,
                  starts=TAB_STARTS, start_width=TAB_START_WIDTH)


def tab_start_sweep(d):
    """Per start size and seed, at width ``TAB_START_WIDTH``: the rank correlation of the
    trained table's 45 distances with the mean-image distances, with the judge's
    confusion, and with the distances of the table it started as."""
    na, ns = d["sweep_dist"].shape[:2]
    out = np.zeros((3, na, ns))
    for i in range(na):
        for j in range(ns):
            t = d["sweep_dist"][i, j][PAIRS]
            out[:, i, j] = (spearman(t, d["mean_dist"][PAIRS]), spearman(t, -d["soft"][PAIRS]),
                            spearman(t, d["sweep_start_dist"][i, j][PAIRS]))
    return out


def tab_agreement(d):
    """Rank correlations over the 45 label pairs, each ``(width, seed)``: the trained
    table's distances against the mean-image distances, against the judge's confusion
    (near labels should be confused more: the sign is turned so that agreement is
    positive), and the same two for the untrained table."""
    nw, ns = d["dist"].shape[:2]
    ref_mean, ref_conf = d["mean_dist"][PAIRS], -d["soft"][PAIRS]
    out = np.zeros((4, nw, ns))
    for i in range(nw):
        for j in range(ns):
            trained, start = d["dist"][i, j][PAIRS], d["start_dist"][i, j][PAIRS]
            out[:, i, j] = (spearman(trained, ref_mean), spearman(trained, ref_conf),
                            spearman(start, ref_mean), spearman(start, ref_conf))
    return out


def tab_between_seeds(d):
    """Per width, over the 10 pairs of seeds: the rank correlation of the two tables'
    45 distances, and the correlation of the two tables' numbers themselves."""
    nw, ns = d["dist"].shape[:2]
    geometry, numbers = np.zeros((nw, ns * (ns - 1) // 2)), np.zeros((nw, ns * (ns - 1) // 2))
    for i, w in enumerate(TAB_WIDTHS):
        tables = d["table_%d" % w]
        k = 0
        for a in range(ns):
            for b in range(a + 1, ns):
                geometry[i, k] = spearman(d["dist"][i, a][PAIRS], d["dist"][i, b][PAIRS])
                numbers[i, k] = np.corrcoef(tables[a].ravel(), tables[b].ravel())[0, 1]
                k += 1
    return geometry, numbers


def nearest_counts(dist):
    """``(label, label)`` counts over tables: how often the column label is the row
    label's nearest."""
    dist = np.asarray(dist, float).reshape(-1, 10, 10)
    counts = np.zeros((10, 10), int)
    for m in dist:
        m = m + np.eye(10) * 1e9
        counts[np.arange(10), m.argmin(1)] += 1
    return counts


def label_embedding():
    d = table_data()
    agree = tab_agreement(d)
    geometry, numbers = tab_between_seeds(d)
    starts = tab_start_sweep(d)
    w8 = TAB_WIDTHS.index(8)
    mean8 = d["dist"][w8].mean(0)
    near = nearest_counts(d["dist"][w8])
    order = np.argsort(mean8[PAIRS])
    pair = lambda k: (int(PAIRS[0][k]), int(PAIRS[1][k]))                      # noqa: E731
    _, acc = judge()
    return Study(
        "A label as a vector",
        "Ten digit labels as rows of a table trained to draw the digit from its label: "
        "%d widths, %d seeds" % (len(TAB_WIDTHS), len(TAB_SEEDS)),
        (
            Panel("Distances between the ten labels, width 8",
                  "one_hot: every pair equally far. trained: mean over %d seeds of the "
                  "trained table's row distances (scaled to average 1). mean_images: the "
                  "same for the digits' mean training images." % len(TAB_SEEDS), "matrix",
                  dict(values=mean8, one_hot=d["one_hot_dist"], mean_images=d["mean_dist"],
                       judge_confusion=d["soft"], per_seed=d["dist"][w8])),
            Panel("Agreement with what was measured without the table",
                  "Rank correlation over the 45 label pairs between table distance and "
                  "(ys) mean-image distance; conf: the judge's confusion; the untrained "
                  "table is the control. Mean over seeds; per-seed values in *_seeds.",
                  "curve",
                  dict(x=np.array(TAB_WIDTHS), ys=np.array([agree[0].mean(1), agree[1].mean(1),
                                                            agree[2].mean(1), agree[3].mean(1)]),
                       labels=["trained vs mean images", "trained vs confusion",
                               "untrained vs mean images", "untrained vs confusion"],
                       seeds=agree)),
            Panel("The same geometry from different numbers",
                  "Between pairs of seeds, per width: rank correlation of the two tables' "
                  "45 distances (geometry) and correlation of their entries (numbers). "
                  "start_sweep: at width %d, per start size, the trained table's agreement "
                  "with the mean images, the confusion, and its own start." % TAB_START_WIDTH,
                  "curve",
                  dict(x=np.array(TAB_WIDTHS), ys=np.array([geometry.mean(1), numbers.mean(1)]),
                       labels=["geometry", "numbers"], geometry=geometry, numbers=numbers,
                       start_sizes=np.array(TAB_STARTS), start_sweep=starts.mean(2),
                       start_sweep_seeds=starts)),
            Panel("What a row draws, and the error of drawing from the label alone",
                  "values: the ten pictures a width-8 table's rows decode to, then the ten "
                  "mean training images. loss: held-out squared error per width and seed; "
                  "floor: the error of the mean images themselves.", "images",
                  dict(values=np.vstack([d["drawn"], d["means"]]).reshape(20, 28, 28),
                       labels=["row %d" % c for c in range(10)]
                       + ["mean %d" % c for c in range(10)],
                       loss=d["loss"], floor=d["floor"])),
        ),
        dict(
            seed=SEED, seeds=list(TAB_SEEDS), widths=list(TAB_WIDTHS), mnist=True,
            review_only=True, dataset=data()[4],
            source="NumPy training on MNIST",
            model="table (10 x width, normal %.2f) -> %d ReLU -> 784 sigmoid; squared error "
                  "to the image; Adam %.0e, batch %d, %d epochs of %d training images"
                  % (TAB_INIT, TAB_HIDDEN, TAB_LR, TAB_BATCH, TAB_EPOCHS, N_TRAIN),
            judge="voxanim.mnist.train(): 784-64-32-10, %.4f of the 10,000 test images; its "
                  "confusion here is its mean probability on %d held-out images" % (acc, N_TEST),
            agreement=dict(
                widths=list(TAB_WIDTHS),
                trained_vs_mean_images=agree[0].mean(1).tolist(),
                trained_vs_confusion=agree[1].mean(1).tolist(),
                untrained_vs_mean_images=agree[2].mean(1).tolist(),
                untrained_vs_confusion=agree[3].mean(1).tolist(),
                trained_vs_mean_images_min=agree[0].min(1).tolist(),
                trained_vs_mean_images_max=agree[0].max(1).tolist()),
            between_seeds=dict(geometry=geometry.mean(1).tolist(),
                               numbers=numbers.mean(1).tolist()),
            start_size=dict(width=TAB_START_WIDTH, sizes=list(TAB_STARTS),
                            trained_vs_mean_images=starts[0].mean(1).tolist(),
                            trained_vs_confusion=starts[1].mean(1).tolist(),
                            trained_vs_its_own_start=starts[2].mean(1).tolist()),
            closest_pairs_width_8=[pair(k) for k in order[:5]],
            farthest_pairs_width_8=[pair(k) for k in order[-3:]],
            nearest_label_counts_width_8=near.tolist(),
            held_out_error=dict(per_width=d["loss"].mean(1).tolist(), floor=float(d["floor"])),
            limitations="The task has only the label to go on, so the best it can do is each "
                        "digit's mean image: the geometry it learns is the geometry of ten "
                        "mean images, and the mean-image comparison is close to that by "
                        "construction; the judge's confusion is the independent one. "
                        "Three epochs: the narrowest table has not quite reached the floor.",
            designs_tried=[
                "first: every table started as normal draws of size 0.1, no start-size "
                "sweep. Agreement with the mean images was 0.33 (width 2) to 0.76 (width "
                "32) and two seeds' geometries agreed only 0.16 to 0.66: a trial of smaller "
                "starts (0.01, 0.001; 3 and 10 epochs; 3 seeds) showed the table was keeping "
                "its random start",
                "this design: tables start at 0.01, and the start size is itself measured "
                "(1, 0.1, 0.01, 0.001 at width 8, 5 seeds) and shown",
            ],
        ),
    )


# ====================================================================== 107: the classifier
CLF_VARIANTS = ("one-hot", "smoothed", "learned", "random")
CLF_WIDTHS = (2, 4, 8, 16)
CLF_SEEDS = (17, 18, 19, 20, 21)
CLF_HIDDEN = 256
CLF_EPOCHS, CLF_BATCH, CLF_LR = 6, 100, 1e-3
CLF_SMOOTH = 0.1                  # label smoothing: this much of each target spread over all ten
CLF_SCALE = 16.0                  # cosine scores are multiplied by this before the softmax
CLF_TABLE_START = 0.01            # the table's rows start as random directions of this length
CLF_KEEP = 200                    # held-out images whose features are kept (first seed)


def clf_feature(P, x):
    h = np.maximum(x @ P["W0"] + P["b0"], 0)
    return h, h @ P["W1"] + P["b1"]


def clf_scores(variant, P, f):
    """The ten scores of features ``f``: a dot product with each label's row plus a
    bias, or ``CLF_SCALE`` times the cosine to each row."""
    if variant in ("one-hot", "smoothed"):
        return f @ P["T"].T + P["c"]
    fn = f / np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
    tn = P["T"] / np.linalg.norm(P["T"], axis=1, keepdims=True)
    return CLF_SCALE * fn @ tn.T


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def clf_grads(variant, P, x, y):
    """Gradients of the mean cross-entropy to the (smoothed) targets."""
    n = len(x)
    h, f = clf_feature(P, x)
    target = np.eye(10)[y]
    if variant == "smoothed":
        target = target * (1 - CLF_SMOOTH) + CLF_SMOOTH / 10
    d = (softmax(clf_scores(variant, P, f)) - target) / n
    g = {}
    if variant in ("one-hot", "smoothed"):
        g["T"], g["c"] = d.T @ f, d.sum(0)
        df = d @ P["T"]
    else:
        fl = np.maximum(np.linalg.norm(f, axis=1, keepdims=True), 1e-12)
        tl = np.linalg.norm(P["T"], axis=1, keepdims=True)
        fn, tn = f / fl, P["T"] / tl
        dfn, dtn = CLF_SCALE * d @ tn, CLF_SCALE * d.T @ fn
        df = (dfn - fn * (dfn * fn).sum(1, keepdims=True)) / fl
        g["T"] = (dtn - tn * (dtn * tn).sum(1, keepdims=True)) / tl
        if variant == "random":
            g["T"] = np.zeros_like(P["T"])            # the table is left where it started
        g["c"] = np.zeros(10)
    g["W1"], g["b1"] = h.T @ df, df.sum(0)
    dh = (df @ P["W1"].T) * (h > 0)
    g["W0"], g["b0"] = x.T @ dh, dh.sum(0)
    return g


def clf_init(width, seed):
    """Every variant of one seed and width starts from the same network and table."""
    rng = np.random.default_rng([seed, width, 107])
    P = {}
    dense_init(P, "0", 784, CLF_HIDDEN, rng)
    dense_init(P, "1", CLF_HIDDEN, width, rng)
    P["T"] = CLF_TABLE_START * rng.normal(0, 1.0 / np.sqrt(width), (10, width))
    P["c"] = np.zeros(10)
    return P, rng


def clf_run(variant, width, seed):
    x, y, xt, yt, _ = data()
    P, rng = clf_init(width, seed)
    opt = Adam(P, CLF_LR)
    for idx in minibatches(rng, len(x), CLF_EPOCHS, CLF_BATCH):
        opt.step(P, clf_grads(variant, P, x[idx], y[idx]))
    _, f = clf_feature(P, xt)
    scores = clf_scores(variant, P, f)
    said = scores.argmax(1)
    confusion = np.zeros((10, 10), int)
    np.add.at(confusion, (yt, said), 1)
    table = np.zeros((10, max(CLF_WIDTHS)))
    table[:, :width] = P["T"]
    feats = np.zeros((CLF_KEEP, max(CLF_WIDTHS)))
    feats[:, :width] = f[:CLF_KEEP]
    return dict(acc=(said == yt).mean(), confusion=confusion, table=table, bias=P["c"],
                feats=feats, scores=scores[:CLF_KEEP])


def _clf_build():
    _means, mean_dist, soft = references()
    jobs = [(v, w, s) for v in CLF_VARIANTS for w in CLF_WIDTHS for s in CLF_SEEDS]
    runs = _parallel(clf_run, jobs)
    shape = (len(CLF_VARIANTS), len(CLF_WIDTHS), len(CLF_SEEDS))

    def stack(k):
        return np.array([r[k] for r in runs]).reshape(shape + np.shape(runs[0][k]))
    starts = np.zeros(shape[1:] + (10, max(CLF_WIDTHS)))
    for i, w in enumerate(CLF_WIDTHS):
        for j, s in enumerate(CLF_SEEDS):
            starts[i, j, :, :w] = clf_init(w, s)[0]["T"]
    return dict(acc=stack("acc"), confusion=stack("confusion"), table=stack("table"),
                bias=stack("bias"), feats=stack("feats")[:, :, 0], scores=stack("scores")[:, :, 0],
                start=starts, soft=soft, mean_dist=mean_dist)


def classifier_data():
    """The classifier experiment's arrays (cached)."""
    return _cache("classifier", _clf_build,
                  (_clf_build, clf_run, clf_init, clf_grads, clf_scores, clf_feature, references),
                  variants=CLF_VARIANTS, widths=CLF_WIDTHS, seeds=CLF_SEEDS, hidden=CLF_HIDDEN,
                  epochs=CLF_EPOCHS, batch=CLF_BATCH, lr=CLF_LR, smooth=CLF_SMOOTH,
                  scale=CLF_SCALE, keep=CLF_KEEP, table_start=CLF_TABLE_START)


def clf_table(d, variant, width, seed_index):
    """One run's table of label vectors ``(10, width)``."""
    return d["table"][CLF_VARIANTS.index(variant), CLF_WIDTHS.index(width), seed_index, :, :width]


def clf_structure(d):
    """Per ``(variant, width, seed)``: the rank correlation over the 45 label pairs of
    the table's cosines with the judge's confusion, the same with the run's own mistakes
    (both ways summed), the spread (standard deviation) of the 45 cosines, and the cosine
    of the 4 and 9 rows. Then the cosine matrices themselves."""
    nv, nw, ns = d["acc"].shape
    out = np.zeros((4, nv, nw, ns))
    cos = np.zeros((nv, nw, ns, 10, 10))
    for v in range(nv):
        for i, w in enumerate(CLF_WIDTHS):
            for j in range(ns):
                c = cos[v, i, j] = cosines(d["table"][v, i, j, :, :w])
                own = d["confusion"][v, i, j] + d["confusion"][v, i, j].T
                out[:, v, i, j] = (spearman(c[PAIRS], d["soft"][PAIRS]),
                                   spearman(c[PAIRS], own[PAIRS]), c[PAIRS].std(), c[4, 9])
    return out, cos


def clf_mistakes(d):
    """Per ``(variant, width)``: the mistakes summed over seeds as ``(true, said)`` counts
    with the diagonal emptied; and per variant and width the rank correlation of its 45
    pair counts with the one-hot classifier's."""
    wrong = d["confusion"].sum(2) * (1 - np.eye(10, dtype=int))
    both = wrong + np.swapaxes(wrong, -1, -2)
    same = np.array([[spearman(both[v, i][PAIRS], both[0, i][PAIRS])
                      for i in range(len(CLF_WIDTHS))] for v in range(len(CLF_VARIANTS))])
    return wrong, same


def label_embedding_classifier():
    d = classifier_data()
    acc = d["acc"]
    structure, cos = clf_structure(d)
    wrong, same = clf_mistakes(d)
    wide = len(CLF_WIDTHS) - 1
    top = {}
    for v, name in enumerate(CLF_VARIANTS):
        both = wrong[v, wide] + wrong[v, wide].T
        k = int(np.argmax(both[PAIRS]))
        top[name] = dict(pair=[int(PAIRS[0][k]), int(PAIRS[1][k])], mistakes=int(both[PAIRS][k]),
                         of=int(wrong[v, wide].sum()))
    by = lambda a: {name: a[v].tolist() for v, name in enumerate(CLF_VARIANTS)}   # noqa: E731
    _, judge_acc = judge()
    return Study(
        "Label vectors in a digit classifier",
        "The last layer of a classifier as a table of ten label vectors, treated four ways: "
        "%d feature widths, %d seeds" % (len(CLF_WIDTHS), len(CLF_SEEDS)),
        (
            Panel("Held-out accuracy",
                  "Fraction of %d held-out images right, mean over %d seeds, per feature "
                  "width; seeds: every run." % (N_TEST, len(CLF_SEEDS)), "curve",
                  dict(x=np.array(CLF_WIDTHS), ys=acc.mean(2), labels=list(CLF_VARIANTS),
                       seeds=acc)),
            Panel("The label vectors' cosines, width %d" % CLF_WIDTHS[wide],
                  "values: the one-hot classifier's ten rows (mean row removed), cosine "
                  "between each pair, mean over seeds; by_variant: the same for all four.",
                  "matrix",
                  dict(values=cos[0, wide].mean(0), by_variant=cos[:, wide].mean(1),
                       first_seed=cos[:, wide, 0], judge_confusion=d["soft"])),
            Panel("Do near label vectors mean confused digits?",
                  "Rank correlation over the 45 label pairs of table cosine with the judge's "
                  "confusion, mean over seeds, per width; own: with the run's own mistakes; "
                  "spread: standard deviation of the 45 cosines; four_nine: cosine of the 4 "
                  "and 9 rows.", "curve",
                  dict(x=np.array(CLF_WIDTHS), ys=structure[0].mean(2), labels=list(CLF_VARIANTS),
                       seeds=structure[0], own=structure[1], spread=structure[2],
                       four_nine=structure[3])),
            Panel("Where the mistakes go, width %d" % CLF_WIDTHS[wide],
                  "values: the one-hot classifier's mistakes (true digit down, answer "
                  "across) summed over seeds; by_variant: all four; same_pairs: rank "
                  "correlation of each variant's 45 pair counts with the one-hot ones.",
                  "matrix",
                  dict(values=wrong[0, wide].astype(float), by_variant=wrong[:, wide].astype(float),
                       same_pairs=same)),
        ),
        dict(
            seed=SEED, seeds=list(CLF_SEEDS), widths=list(CLF_WIDTHS), mnist=True,
            review_only=True, dataset=data()[4], source="NumPy training on MNIST",
            model="784 -> %d ReLU -> feature (width) ; ten scores from the feature and a "
                  "10 x width table; cross-entropy; Adam %.0e, batch %d, %d epochs of %d "
                  "training images; all four variants of a seed start from the same weights "
                  "and the same table" % (CLF_HIDDEN, CLF_LR, CLF_BATCH, CLF_EPOCHS, N_TRAIN),
            variants={
                "one-hot": "score = feature . row + bias, one-hot targets (the ordinary "
                           "classifier)",
                "smoothed": "the same, each target %.1f spread over the ten labels" % CLF_SMOOTH,
                "learned": "score = %.0f x cosine(feature, row), the table trained with the "
                           "network" % CLF_SCALE,
                "random": "score = %.0f x cosine(feature, row), the table never updated "
                          "(the control for 'learned')" % CLF_SCALE},
            judge_accuracy=judge_acc,
            accuracy=by(acc.mean(2)), accuracy_min=by(acc.min(2)), accuracy_max=by(acc.max(2)),
            cosine_vs_confusion=by(structure[0].mean(2)),
            cosine_vs_own_mistakes=by(structure[1].mean(2)),
            cosine_spread=by(structure[2].mean(2)),
            four_nine_cosine=by(structure[3].mean(2)),
            most_confused_pair_widest=top,
            mistakes_same_pairs_as_one_hot=by(same),
            limitations="One body (a 256-unit layer), one data size, one cosine scale; "
                        "smoothing 0.1 only. Starting the table small costs the two "
                        "dot-product classifiers about a point after 6 epochs (one-hot "
                        "0.962 at width 16 with the first design's unit-length start). "
                        "Mistakes are counted on %d held-out images per "
                        "run, so single cells of the mistake tables are small counts."
                        % N_TEST,
            designs_tried=[
                "first: the table started as random rows of length about 1. After 6 epochs "
                "the ordinary classifier's table was still mostly that start: its cosines "
                "agreed with the judge's confusion only 0.22 at width 16 (learned cosine "
                "table 0.16, random 0), and 4 and 9 were no nearer than average",
                "this design: the same random directions, started at length %.2f, so that "
                "what the table ends as is what training made it (tried on 2 seeds at "
                "widths 4 and 16 with 6 and 15 epochs before the full run; 6 kept). A "
                "cosine score ignores a row's length, so the fixed random table is the "
                "same control as before" % CLF_TABLE_START,
            ],
        ),
    )


# ====================================================================== 108, 109: the autoencoders
AE_VARIANTS = ("plain", "decoder", "both", "shuffled")
AE_CODES = (2, 4, 8, 16)
AE_SEEDS = (17, 18, 19)
AE_HIDDEN = (256, 64)             # the encoder's two ReLU layers; the decoder mirrors them
AE_LABEL = 8                      # width of the label table
AE_EPOCHS, AE_BATCH, AE_LR = 10, 128, 1e-3
AE_PROBE_TRAIN, AE_PROBE_TEST = 5000, 2000
AE_NEIGHBOURS = 10
AE_SWAP = 1000                    # held-out images whose label is swapped (to each other label)
AE_ALPHAS = np.linspace(0.0, 1.0, 11)
AE_STRIPS = ((4, 9), (3, 8), (1, 7), (0, 6))      # label pairs whose in-between is kept as pictures
AE_CLOUD = 1500                   # held-out codes kept for the drawing


def ae_init(variant, code, seed):
    rng = np.random.default_rng([seed, code, AE_VARIANTS.index(variant), 108])
    told = variant != "plain"
    a, b = AE_HIDDEN
    P = {}
    dense_init(P, "0", 784 + (AE_LABEL if variant == "both" else 0), a, rng)
    dense_init(P, "1", a, b, rng)
    dense_init(P, "2", b, code, rng)
    dense_init(P, "3", code + (AE_LABEL if told else 0), b, rng)
    dense_init(P, "4", b, a, rng)
    dense_init(P, "5", a, 784, rng)
    if told:
        P["E"] = rng.normal(0, 1.0, (10, AE_LABEL))
    return P, rng


def ae_encode(variant, P, x, rows=None):
    """``(inputs, h0, h1, code)``; ``rows``: the label rows the 'both' encoder is given."""
    a0 = np.hstack([x, rows]) if variant == "both" else x
    h0 = np.maximum(a0 @ P["W0"] + P["b0"], 0)
    h1 = np.maximum(h0 @ P["W1"] + P["b1"], 0)
    return a0, h0, h1, h1 @ P["W2"] + P["b2"]


def ae_decode(variant, P, z, rows=None):
    """``(inputs, h3, h4, picture)``; ``rows``: the label rows the decoder is given."""
    a3 = z if variant == "plain" else np.hstack([z, rows])
    h3 = np.maximum(a3 @ P["W3"] + P["b3"], 0)
    h4 = np.maximum(h3 @ P["W4"] + P["b4"], 0)
    return a3, h3, h4, sigmoid(h4 @ P["W5"] + P["b5"])


def ae_grads(variant, P, x, y):
    rows = None if variant == "plain" else P["E"][y]
    a0, h0, h1, z = ae_encode(variant, P, x, rows)
    a3, h3, h4, r = ae_decode(variant, P, z, rows)
    g = {}
    d = recon_grad(r, x)
    g["W5"], g["b5"] = h4.T @ d, d.sum(0)
    d = (d @ P["W5"].T) * (h4 > 0)
    g["W4"], g["b4"] = h3.T @ d, d.sum(0)
    d = (d @ P["W4"].T) * (h3 > 0)
    g["W3"], g["b3"] = a3.T @ d, d.sum(0)
    d = d @ P["W3"].T
    code = z.shape[1]
    drows = d[:, code:]
    d = d[:, :code]
    g["W2"], g["b2"] = h1.T @ d, d.sum(0)
    d = (d @ P["W2"].T) * (h1 > 0)
    g["W1"], g["b1"] = h0.T @ d, d.sum(0)
    d = (d @ P["W1"].T) * (h0 > 0)
    g["W0"], g["b0"] = a0.T @ d, d.sum(0)
    if variant != "plain":
        if variant == "both":
            drows = drows + (d @ P["W0"].T)[:, 784:]
        g["E"] = np.zeros_like(P["E"])
        np.add.at(g["E"], y, drows)
    return g


def ae_loss(variant, P, x, y):
    rows = None if variant == "plain" else P["E"][y]
    z = ae_encode(variant, P, x, rows)[3]
    return float(np.mean((ae_decode(variant, P, z, rows)[3] - x) ** 2))


def ink_measures(img):
    """``(ink, slant, thickness)`` of pictures ``(n, 784)``: the summed ink; the shear
    of the ink about its centre (positive: leaning right, as most handwriting does);
    and the stroke's thickness, as inked pixels (above half) per inked pixel on the
    stroke's edge."""
    v = np.asarray(img, float).reshape(-1, 28, 28)
    ink = v.sum((1, 2))
    rows, cols = np.mgrid[0:28, 0:28]
    total = np.maximum(ink, 1e-9)
    cy = (v * rows).sum((1, 2)) / total
    cx = (v * cols).sum((1, 2)) / total
    dy, dx = rows[None] - cy[:, None, None], cols[None] - cx[:, None, None]
    slant = -(v * dx * dy).sum((1, 2)) / np.maximum((v * dy * dy).sum((1, 2)), 1e-9)
    m = np.pad(v > 0.5, ((0, 0), (1, 1), (1, 1)))
    inner = m[:, 1:-1, 1:-1]
    edge = inner & ~(m[:, :-2, 1:-1] & m[:, 2:, 1:-1] & m[:, 1:-1, :-2] & m[:, 1:-1, 2:])
    thickness = inner.sum((1, 2)) / np.maximum(edge.sum((1, 2)), 1)
    return np.array([ink, slant, thickness])


def knn_accuracy(ref, ref_y, query, query_y, k=AE_NEIGHBOURS):
    """How often the commonest digit among a code's ``k`` nearest reference codes is
    its own."""
    d = ((query ** 2).sum(1)[:, None] - 2 * query @ ref.T + (ref ** 2).sum(1)[None])
    near = ref_y[np.argpartition(d, k, axis=1)[:, :k]]
    votes = np.array([(near == c).sum(1) for c in range(10)])
    return float((votes.argmax(0) == query_y).mean())


def linear_probe(train, train_y, test, test_y, steps=300, lr=0.05):
    """Held-out accuracy of a softmax regression reading the digit from the code."""
    mu, sd = train.mean(0), train.std(0) + 1e-9
    a, b = (train - mu) / sd, (test - mu) / sd
    P = {"W": np.zeros((a.shape[1], 10)), "b": np.zeros(10)}
    opt = Adam(P, lr)
    target = np.eye(10)[train_y]
    for _ in range(steps):
        d = (softmax(a @ P["W"] + P["b"]) - target) / len(a)
        opt.step(P, {"W": a.T @ d, "b": d.sum(0)})
    return float(((b @ P["W"] + P["b"]).argmax(1) == test_y).mean())


def first_of_each(y):
    """The first image of each digit, 0 to 9."""
    return np.array([int(np.flatnonzero(y == c)[0]) for c in range(10)])


def pooled_r(a, b, groups):
    """Correlation of ``a`` and ``b`` within each group, averaged over the groups."""
    return float(np.mean([np.corrcoef(a[groups == c], b[groups == c])[0, 1]
                          for c in np.unique(groups)]))


def ae_run(variant, code, seed):
    """Train one autoencoder and measure everything animations 108 and 109 use."""
    x, y, xt, yt, _ = data()
    read, _ = judge()
    P, rng = ae_init(variant, code, seed)
    opt = Adam(P, AE_LR)
    draws = np.random.default_rng([seed, code, 9])    # the control's random labels
    for idx in minibatches(rng, len(x), AE_EPOCHS, AE_BATCH):
        lab = draws.integers(0, 10, len(idx)) if variant == "shuffled" else y[idx]
        opt.step(P, ae_grads(variant, P, x[idx], lab))
    told = variant != "plain"
    E = P["E"] if told else np.zeros((10, AE_LABEL))
    lab_train = draws.integers(0, 10, len(x)) if variant == "shuffled" else y
    lab_test = draws.integers(0, 10, len(xt)) if variant == "shuffled" else yt

    def code_of(xs, labs):
        return ae_encode(variant, P, xs, E[labs] if told else None)[3]

    def draw(z, rows):
        return ae_decode(variant, P, z, rows if told else None)[3]

    z_test = code_of(xt, lab_test)
    recon = draw(z_test, E[lab_test])
    out = dict(loss=np.mean((recon - xt) ** 2), table=E)
    # what the code holds: the digit read from the code alone
    z_ref = code_of(x[:AE_PROBE_TRAIN], lab_train[:AE_PROBE_TRAIN])
    zq, yq = z_test[:AE_PROBE_TEST], yt[:AE_PROBE_TEST]
    out["knn"] = knn_accuracy(z_ref, y[:AE_PROBE_TRAIN], zq, yq)
    out["probe"] = linear_probe(z_ref, y[:AE_PROBE_TRAIN], zq, yq)
    out["recon_read"] = (read(recon).argmax(1) == yt).mean()
    # swap the label row: every one of the first AE_SWAP held-out images as each other digit
    n = AE_SWAP
    src, zs = yt[:n], z_test[:n]
    want = np.array([[c for c in range(10) if c != s] for s in src])          # (n, 9)
    zz = np.repeat(zs, 9, axis=0)
    swapped = draw(zz, E[want.ravel()])
    said = read(swapped).argmax(1).reshape(n, 9)
    out["swap_as_asked"] = (said == want).mean()
    out["swap_as_source"] = (said == src[:, None]).mean()
    grid = np.zeros((10, 10))                                                 # source, asked
    np.add.at(grid, (np.repeat(src, 9), want.ravel()), (said == want).ravel())
    out["swap_grid"] = grid / np.maximum(np.bincount(src, minlength=10), 1)[:, None]
    # how much of the handwriting survives: source against its swapped pictures, within
    # each asked digit; against its own reconstruction; and against an unrelated real
    # image of the asked digit (the floor)
    m_src = ink_measures(xt[:n])
    m_swap = ink_measures(swapped).reshape(3, n, 9)
    m_rec = ink_measures(recon[:n])
    other = np.random.default_rng([seed, 4])
    pools = [np.flatnonzero(y == c) for c in range(10)]
    unrelated = np.array([other.choice(pools[c]) for c in want.ravel()])
    m_other = ink_measures(x[unrelated]).reshape(3, n, 9)
    groups = want.ravel()
    out["style_swap"] = np.array([pooled_r(np.repeat(m_src[k], 9), m_swap[k].ravel(), groups)
                                  for k in range(3)])
    out["style_recon"] = np.array([pooled_r(m_src[k], m_rec[k], src) for k in range(3)])
    out["style_floor"] = np.array([pooled_r(np.repeat(m_src[k], 9), m_other[k].ravel(), groups)
                                   for k in range(3)])
    # a row between two labels
    half = 300
    mix = np.zeros((3, len(AE_ALPHAS)))               # reads as: the source, the asked, neither
    for i, alpha in enumerate(AE_ALPHAS):
        rows = (1 - alpha) * E[np.repeat(src[:half], 9)] + alpha * E[want[:half].ravel()]
        s = read(draw(zz[:half * 9], rows)).argmax(1)
        a, b = s == np.repeat(src[:half], 9), s == want[:half].ravel()
        mix[:, i] = a.mean(), b.mean(), 1 - a.mean() - b.mean()
    out["mix"] = mix
    # pictures: the first held-out image of each digit, drawn as every digit; four
    # in-between strips; the held-out codes themselves
    first = first_of_each(yt)
    zf = z_test[first]
    out["pictures"] = draw(np.repeat(zf, 10, axis=0), E[np.tile(np.arange(10), 10)]).reshape(
        10, 10, 784).astype(np.float32)
    out["read_pictures"] = read(out["pictures"].reshape(100, 784)).argmax(1).reshape(10, 10)
    strips = []
    for a, b in AE_STRIPS:
        rows = (1 - AE_ALPHAS[:, None]) * E[a] + AE_ALPHAS[:, None] * E[b]
        strips.append(draw(np.repeat(zf[a:a + 1], len(AE_ALPHAS), axis=0), rows))
    out["strips"] = np.array(strips, np.float32)
    out["strip_read"] = read(out["strips"].reshape(-1, 784)).reshape(len(AE_STRIPS), -1, 10)
    cloud = np.zeros((AE_CLOUD, max(AE_CODES)), np.float32)
    cloud[:, :code] = z_test[:AE_CLOUD]
    out["cloud"] = cloud
    return out


def _ae_build():
    _x, _y, xt, yt, _ = data()
    judge()                                           # trained before the workers start
    jobs = [(v, c, s) for v in AE_VARIANTS for c in AE_CODES for s in AE_SEEDS]
    runs = _parallel(ae_run, jobs)
    shape = (len(AE_VARIANTS), len(AE_CODES), len(AE_SEEDS))
    out = {}
    for key in ("loss", "knn", "probe", "recon_read", "swap_as_asked", "swap_as_source",
                "swap_grid", "style_swap", "style_recon", "style_floor", "mix", "table"):
        out[key] = np.array([r[key] for r in runs]).reshape(shape + np.shape(runs[0][key]))
    for key in ("pictures", "read_pictures", "strips", "strip_read", "cloud"):   # first seed
        a = np.array([r[key] for r in runs]).reshape(shape + np.shape(runs[0][key]))
        out[key] = a[:, :, 0]
    first = first_of_each(yt)
    out["first"] = first
    out["sources"] = xt[first].astype(np.float32)
    out["cloud_y"] = yt[:AE_CLOUD]
    read, _acc = judge()
    out["judge_held_out"] = np.array((read(xt).argmax(1) == yt).mean())
    out["source_measures"] = ink_measures(xt[first])
    return out


def autoencoder_data():
    """Every autoencoder of animations 108 and 109, trained once (cached)."""
    return _cache("autoencoders", _ae_build,
                  (_ae_build, ae_run, ae_init, ae_encode, ae_decode, ae_grads, ink_measures,
                   knn_accuracy, linear_probe, pooled_r),
                  variants=AE_VARIANTS, codes=AE_CODES, seeds=AE_SEEDS, hidden=AE_HIDDEN,
                  label=AE_LABEL, epochs=AE_EPOCHS, batch=AE_BATCH, lr=AE_LR,
                  probe=(AE_PROBE_TRAIN, AE_PROBE_TEST, AE_NEIGHBOURS), swap=AE_SWAP,
                  alphas=len(AE_ALPHAS), strips=AE_STRIPS, cloud=AE_CLOUD)


AE_MODEL = ("%d -> %d -> %d ReLU -> code (linear) -> %d -> %d ReLU -> 784 sigmoid; the label "
            "table is 10 x %d (normal 1), its row joined to the code at the decoder's input "
            "('decoder'), and to the image at the encoder's input too ('both'); 'shuffled' is "
            "the 'decoder' network given a random label's row for every image; squared "
            "error; Adam %.0e, batch %d, %d epochs of %d training images"
            % (784, AE_HIDDEN[0], AE_HIDDEN[1], AE_HIDDEN[1], AE_HIDDEN[0], AE_LABEL, AE_LR,
               AE_BATCH, AE_EPOCHS, N_TRAIN))


def _by(a):
    """``{variant: per-code list}`` of an array whose first two axes are variant, code."""
    return {name: np.asarray(a[v]).tolist() for v, name in enumerate(AE_VARIANTS)}


def label_embedding_autoencoder():
    d = autoencoder_data()
    loss, knn, probe = d["loss"], d["knn"], d["probe"]
    two = AE_CODES.index(2)
    return Study(
        "Label embeddings in an autoencoder",
        "An autoencoder whose decoder is given the label's row beside the code, against the "
        "plain one of the same size: %d code sizes, %d seeds" % (len(AE_CODES), len(AE_SEEDS)),
        (
            Panel("Held-out reconstruction error",
                  "Mean squared error on %d held-out images per code size, mean over %d "
                  "seeds; seeds: every run." % (N_TEST, len(AE_SEEDS)), "curve",
                  dict(x=np.array(AE_CODES), ys=loss.mean(2), labels=list(AE_VARIANTS),
                       seeds=loss)),
            Panel("The digit read from the code alone",
                  "How often the commonest digit among a held-out code's %d nearest "
                  "training codes is its own (%d queries, %d references), mean over seeds; "
                  "probe: a softmax regression on the code instead."
                  % (AE_NEIGHBOURS, AE_PROBE_TEST, AE_PROBE_TRAIN), "curve",
                  dict(x=np.array(AE_CODES), ys=knn.mean(2), labels=list(AE_VARIANTS),
                       seeds=knn, probe=probe.mean(2), probe_seeds=probe)),
            Panel("Two-number codes of held-out digits",
                  "xy: the plain autoencoder's codes of %d held-out images (first seed); "
                  "decoder, both, shuffled: the other three's." % AE_CLOUD, "points",
                  dict(xy=d["cloud"][0, two, :, :2].astype(float), classes=d["cloud_y"],
                       decoder=d["cloud"][1, two, :, :2].astype(float),
                       both=d["cloud"][2, two, :, :2].astype(float),
                       shuffled=d["cloud"][3, two, :, :2].astype(float))),
            Panel("The same ten digits rebuilt from two numbers",
                  "values: the first held-out image of each digit, then its reconstruction "
                  "by the plain, decoder-told, both-told and shuffled-label autoencoders "
                  "(code size 2, first seed).", "images",
                  dict(values=np.vstack(
                      [d["sources"]] + [d["pictures"][v, two][np.arange(10), np.arange(10)]
                                        for v in range(4)]).reshape(50, 28, 28).astype(float),
                       labels=["%s %d" % (name, c) for name in ("image",) + AE_VARIANTS
                               for c in range(10)])),
        ),
        dict(
            seed=SEED, seeds=list(AE_SEEDS), codes=list(AE_CODES), mnist=True, review_only=True,
            dataset=data()[4], source="NumPy training on MNIST", model=AE_MODEL,
            held_out_error=_by(loss.mean(2)), held_out_error_min=_by(loss.min(2)),
            held_out_error_max=_by(loss.max(2)),
            digit_from_code_neighbours=_by(knn.mean(2)),
            digit_from_code_neighbours_min=_by(knn.min(2)),
            digit_from_code_neighbours_max=_by(knn.max(2)),
            digit_from_code_linear=_by(probe.mean(2)),
            reconstruction_read_right=_by(d["recon_read"].mean(2)),
            judge="voxanim.mnist.train(): 784-64-32-10; reconstruction_read_right is how "
                  "often it reads a held-out reconstruction as the image's own digit",
            limitations="The decoder is told the label; nothing in the loss asks the encoder "
                        "to leave the digit out of the code, so whatever leaves it does so "
                        "only because the room is better used. 'shuffled' has exactly the "
                        "told decoder's weights and a useless label: it is the same-size "
                        "control for the extra inputs. For 'both', and for the label rows "
                        "at test time, the true held-out labels are used.",
            designs_tried=["this design, fixed before the comparison was run"],
        ),
    )


def label_embedding_generation():
    d = autoencoder_data()
    asked, kept = d["swap_as_asked"], d["swap_as_source"]
    # how often a plain reconstruction is read as one given other digit
    plain_floor = (1 - d["recon_read"][0]) / 9
    names = ("ink", "slant", "thickness")
    small = AE_CODES.index(4)
    dec = AE_VARIANTS.index("decoder")
    style = lambda a: {name: {n: np.asarray(a[v, :, :, k]).mean(1).tolist()         # noqa: E731
                              for k, n in enumerate(names)}
                       for v, name in enumerate(AE_VARIANTS) if name != "plain"}
    _, judge_acc = judge()
    return Study(
        "Changing the label, keeping the style",
        "An image's code decoded with another label's row; a separate classifier reads the "
        "result: %d code sizes, %d seeds" % (len(AE_CODES), len(AE_SEEDS)),
        (
            Panel("Every digit drawn as every digit, code size %d" % AE_CODES[small],
                  "values[s, a]: the first held-out image of digit s, its code decoded with "
                  "label a's row (decoder-told autoencoder, first seed); by_code: the same "
                  "at every code size and variant; sources: the ten images.", "images",
                  dict(values=d["pictures"][dec, small].reshape(100, 28, 28).astype(float),
                       labels=["%d as %d" % (s, a) for s in range(10) for a in range(10)],
                       by_code=d["pictures"], read=d["read_pictures"], sources=d["sources"])),
            Panel("Read as the digit asked for",
                  "Fraction of %d x 9 swaps the judge reads as the asked digit, per code "
                  "size, mean over seeds; as_source: read as the image's own digit instead; "
                  "plain: a plain reconstruction read as one given other digit."
                  % AE_SWAP, "curve",
                  dict(x=np.array(AE_CODES), ys=asked[1:].mean(2), labels=list(AE_VARIANTS[1:]),
                       seeds=asked[1:], as_source=kept[1:].mean(2), plain=plain_floor.mean(1),
                       grid=d["swap_grid"][1:].mean(2))),
            Panel("How much of the handwriting survives the swap",
                  "Correlation, within each asked digit, between the source image's ink, "
                  "slant and stroke thickness and the swapped picture's (decoder-told, per "
                  "code size, mean over seeds); recon: against its own reconstruction; "
                  "floor: against an unrelated real image of the asked digit.", "curve",
                  dict(x=np.array(AE_CODES), ys=d["style_swap"][dec].mean(1).T,
                       labels=list(names), seeds=d["style_swap"][dec],
                       recon=d["style_recon"][dec].mean(1).T,
                       floor=d["style_floor"][dec].mean(1).T,
                       both=d["style_swap"][2].mean(1).T)),
            Panel("A row between two labels",
                  "The label row slid from the image's own to another's: fraction read as "
                  "the source digit, as the asked digit, as neither (decoder-told, code "
                  "size %d, mean over seeds); strips: four in-between rows as pictures."
                  % AE_CODES[small], "curve",
                  dict(x=AE_ALPHAS, ys=d["mix"][dec, small].mean(0),
                       labels=["source", "asked", "neither"], seeds=d["mix"][dec, small],
                       by_code=d["mix"][dec].mean(1), strips=d["strips"][dec],
                       strip_read=d["strip_read"][dec],
                       pairs=np.array(AE_STRIPS))),
        ),
        dict(
            seed=SEED, seeds=list(AE_SEEDS), codes=list(AE_CODES), mnist=True, review_only=True,
            dataset=data()[4], source="NumPy training on MNIST", model=AE_MODEL,
            judge="voxanim.mnist.train(): 784-64-32-10, %.4f of the 10,000 test images, "
                  "%.4f of the %d held-out images used here"
                  % (judge_acc, float(d["judge_held_out"]), N_TEST),
            read_as_asked=_by(asked.mean(2)), read_as_asked_min=_by(asked.min(2)),
            read_as_asked_max=_by(asked.max(2)), read_as_source=_by(kept.mean(2)),
            plain_read_as_another_digit=plain_floor.mean(1).tolist(),
            reconstruction_read_right=_by(d["recon_read"].mean(2)),
            style_swap=style(d["style_swap"]), style_recon=style(d["style_recon"]),
            style_floor=style(d["style_floor"]),
            in_between=dict(alphas=AE_ALPHAS.tolist(),
                            decoder_code_4=d["mix"][dec, small].mean(0).tolist()),
            limitations="The judge is a classifier, not a person: 'read as the asked digit' "
                        "is its answer. The style measures are three simple ones of the "
                        "picture (ink, slant, stroke thickness); blurred outputs make "
                        "thickness coarse. The plain autoencoder has no label input, so its "
                        "number is how often its reconstruction is read as one given other "
                        "digit.",
            designs_tried=["this design, fixed before the comparison was run"],
        ),
    )


STUDIES = {
    "label-embedding": label_embedding,
    "label-embedding-classifier": label_embedding_classifier,
    "label-embedding-autoencoder": label_embedding_autoencoder,
    "label-embedding-generation": label_embedding_generation,
}

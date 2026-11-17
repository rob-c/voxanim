"""The experiments behind animations 150-153: machine learning the way a physics analysis
uses it (see :func:`voxanim.next_catalog.redesigned`).

No physics dataset is used. Real handwritten digits (MNIST) stand in for events, read
out coarsely, as a detector reads an event in cells:

* ``network-regression``: every digit is turned by a known angle (bilinear
  interpolation, :func:`rotate`) and a network with one linear output unit is trained
  with the squared error to give that angle back from the 7 x 7 cell averages. Held-out
  resolution (the standard deviation of predicted minus true) and bias (its mean),
  against the true angle, over seeds. An analysis regresses a particle's energy from
  detector readings in the same way.
* ``boosted-regression-trees``: the same events and the same target (:func:`reg_train`,
  :func:`reg_heldout`), with boosted regression trees: first on one real feature (the
  tilt of the ink, :func:`tilt`), where each tree is a staircase, then on all 49 cells
  for several learning rates and depths, against the network.
* ``classifier-score-as-variable``: the 4s stand for signal, the 9s for background, read
  out as 4 x 4 cells so that the two overlap as signal and background do. The score of
  a small network on held-out digits: a cut and what it keeps, the best cut with fifty
  times more background, and the signal fraction of a known mixture recovered by
  fitting the two score shapes (seeded pseudo-experiments drawn from held-out digits).
* ``overtraining-check``: the same two classes and a large network. The score
  distributions of training and held-out events, their Kolmogorov-Smirnov distance and
  the gap in ROC area, against the size of the training sample and the time trained,
  with early stopping and weight decay.

Everything is plain numpy, from ``numpy.random.default_rng`` generators whose seeds are
recorded; training and held-out events are disjoint and every number reported is a
held-out one unless it says otherwise.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import itertools

import numpy as np

from . import mnist
from .cache import cached_npz
from .optim import Adam
from .study import Panel, Study

VERSION = 1
STAND_IN = ("No physics dataset is used: real handwritten digits (MNIST) stand in for "
            "events.")
EVIDENCE = dict(
    measured="measured on MNIST: networks and trees trained here in numpy, numbers from "
             "held-out digits",
    toys="pseudo-experiments: seeded random draws of held-out MNIST digits with a known "
         "mixture fraction",
)


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them (an edit to one experiment re-runs that experiment only)."""
    h = hashlib.sha256()
    for fn in sources:
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-physapps-" + name, settings,
                                                    build).items()}


def _parallel(fn, jobs):
    from .study_generative import parallel

    return parallel(fn, jobs)


def _floats(v):
    return [float(x) for x in np.asarray(v).ravel()]


# ====================================================================== shared arithmetic
def rotate(images, degrees):
    """28 x 28 images turned anticlockwise (as drawn, row 0 at the top) by ``degrees``
    about the centre of the picture, by bilinear interpolation: each new pixel is the
    weighted mean of the four old pixels around the place it came from, and whatever
    comes from outside the picture is blank."""
    img = np.asarray(images, float).reshape(-1, 28, 28)
    n = len(img)
    t = np.deg2rad(np.broadcast_to(np.asarray(degrees, float), (n,)))[:, None, None]
    r, c = np.mgrid[0:28, 0:28] - 13.5
    src_r = np.cos(t) * r + np.sin(t) * c + 13.5        # where each new pixel came from:
    src_c = -np.sin(t) * r + np.cos(t) * c + 13.5       # the turn undone (rows count down)
    r0, c0 = np.floor(src_r).astype(int), np.floor(src_c).astype(int)
    fr, fc = src_r - r0, src_c - c0
    out = np.zeros((n, 28, 28))
    which = np.arange(n)[:, None, None]
    for dr, wr in ((0, 1 - fr), (1, fr)):
        for dc, wc in ((0, 1 - fc), (1, fc)):
            rr, cc = r0 + dr, c0 + dc
            inside = (rr >= 0) & (rr < 28) & (cc >= 0) & (cc < 28)
            out += wr * wc * inside * img[which, np.clip(rr, 0, 27), np.clip(cc, 0, 27)]
    return out


def cells(images, block):
    """Images as the mean of each ``block x block`` square of pixels: ``(n, cells)``."""
    img = np.asarray(images, float).reshape(-1, 28, 28)
    side = 28 // block
    return img.reshape(len(img), side, block, side, block).mean((2, 4)).reshape(len(img), -1)


def tilt(images):
    """The tilt of the ink in each image, in degrees anticlockwise from upright: the
    direction of the long axis of the ink, from its second moments about its centre.
    Between -90 and 90; turning an image by a few degrees turns this by the same."""
    img = np.asarray(images, float).reshape(-1, 28, 28)
    r, c = np.mgrid[0:28, 0:28].astype(float)
    mass = np.maximum(img.sum((1, 2)), 1e-12)
    mr = (img * r).sum((1, 2)) / mass
    mc = (img * c).sum((1, 2)) / mass
    dr, dc = r[None] - mr[:, None, None], c[None] - mc[:, None, None]
    rr = (img * dr * dr).sum((1, 2)) / mass
    cc = (img * dc * dc).sum((1, 2)) / mass
    rc = (img * dr * dc).sum((1, 2)) / mass
    # rows count downwards: a long axis leaning to the left (turned anticlockwise) has rc > 0
    return np.rad2deg(0.5 * np.arctan2(2 * rc, rr - cc))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def net_init(rng, sizes):
    """He-normal weights and zero biases for ReLU layers of ``sizes``; the last layer has
    one unit and no squashing."""
    P = {}
    for i, (a, b) in enumerate(itertools.pairwise(sizes)):
        P["W%d" % i] = rng.normal(0.0, np.sqrt(2.0 / a), (a, b))
        P["b%d" % i] = np.zeros(b)
    return P


def net_forward(P, x):
    """Every layer's output for inputs ``x``; the last is the output unit, ``(n,)``."""
    layers = len(P) // 2
    hs = [np.asarray(x, float)]
    for i in range(layers):
        z = hs[-1] @ P["W%d" % i] + P["b%d" % i]
        hs.append(np.maximum(z, 0.0) if i < layers - 1 else z[:, 0])
    return hs


def net_output(P, x):
    return net_forward(P, x)[-1]


def net_loss(P, x, y, loss):
    """Mean loss: ``"square"`` is (output - y)^2; ``"logistic"`` is the cross-entropy of
    sigmoid(output) against a label y of 0 or 1."""
    out = net_output(P, x)
    if loss == "square":
        return float(np.mean((out - y) ** 2))
    return float(np.mean(np.logaddexp(0.0, out) - y * out))


def net_grads(P, x, y, loss):
    """The gradient of :func:`net_loss` by backpropagation."""
    layers = len(P) // 2
    hs = net_forward(P, x)
    out = hs[-1]
    d = (2.0 * (out - y) if loss == "square" else sigmoid(out) - y)[:, None] / len(out)
    G = {}
    for i in range(layers - 1, -1, -1):
        G["W%d" % i] = hs[i].T @ d
        G["b%d" % i] = d.sum(0)
        if i:
            d = (d @ P["W%d" % i].T) * (hs[i] > 0)
    return G


def train_net(P, x, y, loss, rng, updates, batch, rate, snaps=(), measure=None, decay=0.0):
    """``updates`` minibatch Adam steps on ``(x, y)``, each epoch's order drawn from
    ``rng``. ``measure(P)`` is recorded before the first update if 0 is in ``snaps`` and
    after each update count in ``snaps``. ``decay``: decoupled weight decay (after each
    step every weight matrix is multiplied by 1 - rate * decay)."""
    opt, out, done, snaps = Adam(P, rate), [], 0, {int(s) for s in snaps}
    if 0 in snaps:
        out.append(measure(P))
    while done < updates:
        order = rng.permutation(len(x))
        for s in range(0, len(x), batch):
            idx = order[s:s + batch]
            opt.step(P, net_grads(P, x[idx], y[idx], loss))
            if decay:
                for k in P:
                    if k.startswith("W"):
                        P[k] *= 1.0 - rate * decay
            done += 1
            if done in snaps:
                out.append(measure(P))
            if done >= updates:
                break
    return out


# ====================================================================== the regression data
REG_SPLIT = 150                   # seed of the event selection and of every rotation
REG_RANGE = 45.0                  # angles are uniform between minus and plus this, in degrees
REG_TRAIN, REG_TEST = 10000, 5000
REG_BLOCK = 4                     # cells are 4 x 4 pixels: 7 x 7 = 49 readings an event
REG_CELLS = (28 // REG_BLOCK) ** 2
REG_SEEDS = (0, 1, 2, 3, 4)
REG_EDGES = np.linspace(-45.0, 45.0, 10)          # nine bins of the true angle


@functools.lru_cache(maxsize=1)
def reg_heldout():
    """The held-out events: ``REG_TEST`` of the official test images, each turned once
    by its own angle. The same for every seed and for both models."""
    x, y = mnist.load()[2:]
    rng = np.random.default_rng([REG_SPLIT, 0])
    index = np.sort(rng.choice(len(x), REG_TEST, replace=False))
    angle = rng.uniform(-REG_RANGE, REG_RANGE, REG_TEST)
    image = rotate(x[index], angle)
    return dict(index=index, angle=angle, image=image, x=cells(image, REG_BLOCK),
                label=y[index], tilt=tilt(image), original=x[index].reshape(-1, 28, 28))


@functools.lru_cache(maxsize=len(REG_SEEDS))
def reg_train(seed):
    """Seed ``seed``'s training events: ``REG_TRAIN`` of the first 50,000 training
    images, each turned once by its own angle."""
    x = mnist.load()[0][:50000]
    rng = np.random.default_rng([REG_SPLIT, 1, seed])
    index = np.sort(rng.choice(len(x), REG_TRAIN, replace=False))
    angle = rng.uniform(-REG_RANGE, REG_RANGE, REG_TRAIN)
    image = rotate(x[index], angle)
    return dict(index=index, angle=angle, x=cells(image, REG_BLOCK), tilt=tilt(image))


def resolution(pred, truth, edges=REG_EDGES):
    """``(bias, resolution)`` of predictions in each bin of the true value: the mean
    and the standard deviation of predicted minus true."""
    res = np.asarray(pred, float) - np.asarray(truth, float)
    which = np.clip(np.digitize(truth, edges) - 1, 0, len(edges) - 2)
    bias = np.array([res[which == k].mean() for k in range(len(edges) - 1)])
    width = np.array([res[which == k].std() for k in range(len(edges) - 1)])
    return bias, width


# ====================================================================== 150: the network
NET_HIDDEN = (128, 64)
NET_RATE, NET_BATCH, NET_EPOCHS = 1e-3, 64, 12
NET_UPDATES = NET_EPOCHS * int(np.ceil(REG_TRAIN / NET_BATCH))
NET_SNAPS = tuple(int(v) for v in np.unique(np.r_[0, np.round(np.geomspace(1, NET_UPDATES,
                                                                           26))]))
NET_TRAIN_SHOWN = 2000            # training events whose error is followed
NET_SCATTER = 260                 # held-out events drawn as points
NET_BINS = np.linspace(-60.0, 60.0, 41)          # of predicted minus true, in degrees


def net_run(seed):
    """One network: 49 cells -> 128 -> 64 ReLU -> one linear unit, trained on the angle
    divided by ``REG_RANGE``. Returns its predictions in degrees for the held-out events
    and for some training events at every snapshot, and the final weights."""
    tr, te = reg_train(seed), reg_heldout()
    P = net_start(seed)
    seen = tr["x"][:NET_TRAIN_SHOWN]

    def measure(P):
        return (REG_RANGE * net_output(P, te["x"]), REG_RANGE * net_output(P, seen))
    snaps = train_net(P, tr["x"], tr["angle"] / REG_RANGE, "square",
                      np.random.default_rng([REG_SPLIT, 3, seed]), NET_UPDATES, NET_BATCH,
                      NET_RATE, NET_SNAPS, measure)
    held = np.array([s[0] for s in snaps], np.float32)
    train = np.array([s[1] for s in snaps], np.float32)
    return held, train, P


def _net_build():
    runs = _parallel(net_run, [(s,) for s in REG_SEEDS])
    out = dict(held=np.array([r[0] for r in runs]), train=np.array([r[1] for r in runs]))
    for k, v in runs[0][2].items():
        out["P_" + k] = v
    return out


def net_data():
    """Every seed's predictions at every snapshot (``held[seed, snap, event]`` and
    ``train``), and seed 0's trained weights."""
    return _cache("network", _net_build, (rotate, cells, net_grads, train_net, net_run,
                                          net_start, reg_train, reg_heldout),
                  seeds=REG_SEEDS, train=REG_TRAIN, test=REG_TEST, hidden=NET_HIDDEN,
                  rate=NET_RATE, batch=NET_BATCH, epochs=NET_EPOCHS, snaps=NET_SNAPS)


def net_weights(d=None):
    d = net_data() if d is None else d
    return {k[2:]: v for k, v in d.items() if k.startswith("P_")}


def net_start(seed=0):
    """Seed ``seed``'s network before any training."""
    return net_init(np.random.default_rng([REG_SPLIT, 2, seed]), (REG_CELLS, *NET_HIDDEN, 1))


def typical_event(d=None):
    """The held-out event worked through on screen: of the events drawn as points, the
    one whose error after training (seed 0) is the median, so that it is neither the
    luckiest nor the unluckiest."""
    d = net_data() if d is None else d
    err = np.abs(d["held"][0, -1, :NET_SCATTER] - reg_heldout()["angle"][:NET_SCATTER])
    return int(np.argsort(err, kind="stable")[NET_SCATTER // 2])


def net_example(P, event):
    """One held-out event through the network, in degrees: each last-hidden unit's
    contribution (its output times its weight to the output unit), the output unit's
    bias, their sum (the prediction), the true angle and the squared error."""
    te = reg_heldout()
    hs = net_forward(P, te["x"][event:event + 1])
    last = len(P) // 2 - 1
    terms = REG_RANGE * hs[-2][0] * P["W%d" % last][:, 0]
    bias = REG_RANGE * float(P["b%d" % last][0])
    pred, truth = float(terms.sum() + bias), float(te["angle"][event])
    return dict(terms=terms, bias=bias, prediction=pred, truth=truth,
                loss=(pred - truth) ** 2, slope=2 * (pred - truth), hidden=hs[-2][0])


def net_summary():
    d, te = net_data(), reg_heldout()
    truth = te["angle"]
    tr_truth = np.array([reg_train(s)["angle"][:NET_TRAIN_SHOWN] for s in REG_SEEDS])
    res = d["held"] - truth                                       # seed, snap, event
    held_rms = np.sqrt((res ** 2).mean(-1))
    train_rms = np.sqrt(((d["train"] - tr_truth[:, None]) ** 2).mean(-1))
    per = [resolution(d["held"][k, -1], truth) for k in range(len(REG_SEEDS))]
    bias, width = np.array([p[0] for p in per]), np.array([p[1] for p in per])
    hist = np.array([np.histogram(np.clip(r, NET_BINS[0], NET_BINS[-1] - 1e-9), NET_BINS)[0]
                     for r in res[0]]) / res.shape[-1]
    return dict(d=d, truth=truth, res=res, held_rms=held_rms, train_rms=train_rms, bias=bias,
                width=width, hist=hist, overall_width=res[:, -1].std(-1),
                overall_bias=res[:, -1].mean(-1),
                snap_width=res[0].std(-1), snap_bias=res[0].mean(-1),
                guess_zero=float(np.sqrt((truth ** 2).mean())))


def network_regression():
    r = net_summary()
    te = reg_heldout()
    centres = (REG_EDGES[:-1] + REG_EDGES[1:]) / 2
    snaps = np.array(NET_SNAPS)
    event = typical_event(r["d"])
    ex0, ex1 = net_example(net_start(), event), net_example(net_weights(r["d"]), event)
    shown = np.arange(NET_SCATTER)
    w, b = r["overall_width"], r["overall_bias"]
    return Study(
        "Regression with a network",
        "A network with one linear output unit trained with the squared error to give "
        "back the angle a real digit was turned by: held-out resolution and bias",
        (
            Panel("Error during training",
                  "ys: root-mean-square of predicted minus true angle, in degrees, on the "
                  "%d held-out events and on %d training events, mean of %d seeds, at "
                  "each update count x (0: before training). per_seed has every seed."
                  % (REG_TEST, NET_TRAIN_SHOWN, len(REG_SEEDS)), "curve",
                  dict(x=snaps, ys=np.array([r["held_rms"].mean(0), r["train_rms"].mean(0)]),
                       labels=["held-out", "training"], per_seed=r["held_rms"],
                       per_seed_train=r["train_rms"])),
            Panel("Predicted against true angle",
                  "xy: true and predicted angle of %d held-out events after training "
                  "(seed 0). snapshots[snap, event]: the same events' predictions at each "
                  "update count in updates." % NET_SCATTER, "points",
                  dict(xy=np.column_stack([r["truth"][shown], r["d"]["held"][0, -1, shown]]),
                       snapshots=r["d"]["held"][0][:, shown], updates=snaps,
                       images=te["image"][shown[:12]].astype(np.float32),
                       originals=te["original"][shown[:12]].astype(np.float32),
                       labels=te["label"][shown[:12]])),
            Panel("The distribution of predicted minus true",
                  "values[snap, bin]: share of all %d held-out events whose residual "
                  "falls in each 3-degree bin (edges in edges), at each update count "
                  "(seed 0). width and mean: its standard deviation and mean."
                  % REG_TEST, "matrix",
                  dict(values=r["hist"], edges=NET_BINS, updates=snaps,
                       width=r["snap_width"], mean=r["snap_bias"])),
            Panel("Resolution and bias against the true angle",
                  "ys: standard deviation (resolution) and mean (bias) of predicted "
                  "minus true in nine 10-degree bins of the true angle x, mean of %d "
                  "seeds; width and bias have each seed." % len(REG_SEEDS), "curve",
                  dict(x=centres, ys=np.array([r["width"].mean(0), r["bias"].mean(0)]),
                       labels=["resolution", "bias"], width=r["width"], bias=r["bias"])),
        ),
        dict(
            seeds=list(REG_SEEDS), mnist=True, review_only=True,
            source="NumPy training on MNIST", dataset="MNIST", stand_in=STAND_IN,
            physics_counterpart="An analysis regresses a particle's energy from detector "
                                "readings in the same way.",
            task="Each 28 x 28 digit is turned about its centre by an angle drawn "
                 "uniformly between -%d and %d degrees (bilinear interpolation, blank "
                 "outside), then read as 7 x 7 cells (4 x 4 pixel means). The target is "
                 "the angle." % (REG_RANGE, REG_RANGE),
            model="49 -> %d -> %d ReLU -> 1 linear; squared error on angle / %d; Adam "
                  "%.0e, batch %d, %d epochs (%d updates)"
                  % (*NET_HIDDEN, REG_RANGE, NET_RATE, NET_BATCH, NET_EPOCHS, NET_UPDATES),
            split="training: %d of the first 50,000 training images per seed; held-out: "
                  "%d official test images, the same for every seed" % (REG_TRAIN, REG_TEST),
            where_the_seed_enters="numpy.random.default_rng([150, k, seed]): k = 1 picks "
                                  "the training events and their angles, 2 the initial "
                                  "weights, 3 each epoch's order; [150, 0] the held-out "
                                  "events and angles",
            resolution=dict(mean=float(w.mean()), min=float(w.min()), max=float(w.max()),
                            per_seed=_floats(w)),
            bias=dict(mean=float(b.mean()), min=float(b.min()), max=float(b.max())),
            before_training=dict(width=float(r["res"][:, 0].std(-1).mean()),
                                 rms=float(r["held_rms"][:, 0].mean())),
            guess_zero_rms=r["guess_zero"],
            train_rms=float(r["train_rms"][:, -1].mean()),
            held_rms=float(r["held_rms"][:, -1].mean()),
            edge_bias=dict(lowest_bin=float(r["bias"][:, 0].mean()),
                           highest_bin=float(r["bias"][:, -1].mean()),
                           middle_bin=float(r["bias"][:, 4].mean()),
                           lowest_range=[float(r["bias"][:, 0].min()),
                                         float(r["bias"][:, 0].max())],
                           highest_range=[float(r["bias"][:, -1].min()),
                                          float(r["bias"][:, -1].max())]),
            example=dict(event=event, truth=ex1["truth"],
                         before=ex0["prediction"], after=ex1["prediction"],
                         loss_before=ex0["loss"], loss_after=ex1["loss"],
                         chosen="the median absolute error of the %d events drawn"
                                % NET_SCATTER),
            evidence=dict(EVIDENCE),
            limitations=[
                "Digits stand in for events: the size of the resolution is this task's, "
                "set largely by how differently people slant their digits.",
                "The angle is uniform between two limits, so near a limit the errors can "
                "only go one way; the bias found at the edges is of that kind.",
                "One architecture and one training length, fixed before the comparison "
                "with the trees; the training error keeps falling after the held-out "
                "error has stopped.",
            ],
            designs_tried=[
                "A pilot compared 28 x 28 pixels, 14 x 14 and 7 x 7 cells as input (one "
                "seed): held-out resolution was 10 to 13 degrees for all, so the "
                "coarsest was kept and given to the trees as well.",
                "The fraction of the image that is ink was considered as the target and "
                "dropped: it is a plain sum of the inputs, so nothing has to be learnt.",
            ],
        ),
    )


# ====================================================================== 152: boosted trees
BINS = 16                         # a cell's reading is cut at 15 evenly spaced thresholds


def binned(x, lo, hi, bins=BINS):
    """Which of ``bins`` equal slices of ``lo..hi`` each value falls in."""
    return np.clip(np.floor((np.asarray(x, float) - lo) / (hi - lo) * bins), 0,
                   bins - 1).astype(np.int64)


def threshold(cut, lo, hi, bins=BINS):
    """The value a split at bin ``cut`` tests: below it goes left."""
    return lo + (np.asarray(cut) + 1) * (hi - lo) / bins


def fit_tree(xb, target, depth, bins=BINS, min_leaf=5):
    """A regression tree of ``depth`` levels fitted to ``target`` by least squares on
    binned features ``xb``: every node takes the feature and threshold that most reduce
    the summed squared error of its events, both sides keeping ``min_leaf`` events; a
    leaf's value is the mean target of its events. Returns ``(feature, cut, leaf)``:
    node ``j`` of level ``d`` is entry ``2^d - 1 + j`` and sends ``xb[:, feature] > cut``
    right; a node with no allowed split sends everything left."""
    n, f = xb.shape
    base = np.arange(f) * bins + xb
    node = np.zeros(n, np.int64)
    feature = np.zeros(2 ** depth - 1, np.int64)
    cut = np.full(2 ** depth - 1, bins - 1, np.int64)
    target = np.asarray(target, float)
    weights = np.broadcast_to(target[:, None], (n, f)).ravel()
    for d in range(depth):
        k = 2 ** d
        idx = (node[:, None] * (f * bins) + base).ravel()
        count = np.bincount(idx, minlength=k * f * bins).reshape(k, f, bins)
        total = np.bincount(idx, weights=weights, minlength=k * f * bins).reshape(k, f, bins)
        nl, sl = np.cumsum(count, 2)[:, :, :-1], np.cumsum(total, 2)[:, :, :-1]
        n_all, s_all = count[:, 0].sum(1)[:, None, None], total[:, 0].sum(1)[:, None, None]
        nr, sr = n_all - nl, s_all - sl
        gain = (sl ** 2 / np.maximum(nl, 1) + sr ** 2 / np.maximum(nr, 1)
                - s_all ** 2 / np.maximum(n_all, 1))
        gain[(nl < min_leaf) | (nr < min_leaf)] = -np.inf
        flat = gain.reshape(k, -1)
        best = flat.argmax(1)
        for j in range(k):
            if flat[j, best[j]] > 1e-12:
                feature[k - 1 + j], cut[k - 1 + j] = divmod(int(best[j]), bins - 1)
        at = k - 1 + node
        node = 2 * node + (xb[np.arange(n), feature[at]] > cut[at])
    size = np.bincount(node, minlength=2 ** depth)
    leaf = np.bincount(node, weights=target, minlength=2 ** depth) / np.maximum(size, 1)
    return feature, cut, leaf


def tree_predict(xb, tree):
    """The leaf value each row of binned features ``xb`` reaches."""
    feature, cut, leaf = tree
    node = np.zeros(len(xb), np.int64)
    rows = np.arange(len(xb))
    for d in range(int(round(np.log2(len(leaf))))):
        at = 2 ** d - 1 + node
        node = 2 * node + (xb[rows, feature[at]] > cut[at])
    return leaf[node]


def boost(xb, y, depth, rate, trees, watch=(), bins=BINS):
    """Gradient boosting for the squared error: start from the mean of ``y``; each tree
    is fitted to the residuals (``y`` minus the sum so far) and ``rate`` times its
    prediction is added. Returns ``(start, [tree, ...], fitted, watched)``: the sum on
    the training events and on each ``watch`` set after 0, 1, ... trees."""
    start = float(np.mean(y))
    f = np.full(len(y), start)
    fitted, grown = [f.copy()], []
    watched = [[np.full(len(w), start)] for w in watch]
    for _ in range(trees):
        tree = fit_tree(xb, y - f, depth, bins)
        f = f + rate * tree_predict(xb, tree)
        grown.append(tree)
        fitted.append(f.copy())
        for w, rows in zip(watch, watched):
            rows.append(rows[-1] + rate * tree_predict(w, tree))
    return start, grown, np.array(fitted), [np.array(rows) for rows in watched]


# ---------------------------------------------------------------------- one feature
ONE_N = 300                       # training events of the one-feature example
ONE_DEPTH, ONE_RATE, ONE_TREES = 2, 0.3, 50
ONE_RANGE, ONE_BINS = (-90.0, 90.0), 18           # the tilt axis, cut at every 10 degrees
ONE_SHOWN = (1, 2, 5, 50)
ONE_PICTURES = 3                  # events drawn as pictures with their tilt
ONE_GRID = np.linspace(-89.9, 89.9, 360)


def one_feature():
    """Boosting on one real feature: the tilt of the ink (:func:`tilt`) of the first
    ``ONE_N`` training events of seed 0, against the angle they were turned by. Kept:
    the events, every partial sum as a staircase over the tilt axis, each tree on its
    own, the residuals each was fitted to, and the held-out error."""
    tr, te = reg_train(0), reg_heldout()
    x, y = tr["tilt"][:ONE_N], tr["angle"][:ONE_N]
    xb = binned(x, *ONE_RANGE, ONE_BINS)[:, None]
    grid = binned(ONE_GRID, *ONE_RANGE, ONE_BINS)[:, None]
    held = binned(te["tilt"], *ONE_RANGE, ONE_BINS)[:, None]
    start, trees, fitted, (steps, held_f) = boost(xb, y, ONE_DEPTH, ONE_RATE, ONE_TREES,
                                                  (grid, held), ONE_BINS)
    alone = np.array([tree_predict(grid, t) for t in trees])
    images = rotate(mnist.load()[0][tr["index"][:ONE_PICTURES]], y[:ONE_PICTURES])
    return dict(x=x, y=y, images=images, start=start, fitted=fitted, steps=steps, alone=alone,
                residuals=y[None] - fitted,
                train_rms=np.sqrt(((fitted - y) ** 2).mean(1)),
                held_rms=np.sqrt(((held_f - te["angle"]) ** 2).mean(1)),
                cuts=np.array([threshold(t[1], *ONE_RANGE, ONE_BINS) for t in trees]),
                leaves=np.array([t[2] for t in trees]))


# ---------------------------------------------------------------------- all 49 cells
BRT_RATES = (0.03, 0.1, 0.3, 1.0)
BRT_DEPTHS = (1, 3, 5)
BRT_TREES = 300
BRT_SEEDS = (0, 1, 2)
BRT_MAIN = (0.1, 3)               # the setting drawn first


def brt_run(seed, rate, depth):
    """Boosted trees on seed ``seed``'s training events (the network's, cell for cell):
    training and held-out root-mean-square error after 0, 1, ... trees, and the
    held-out predictions after the last tree."""
    tr, te = reg_train(seed), reg_heldout()
    xb, hb = binned(tr["x"], 0.0, 1.0), binned(te["x"], 0.0, 1.0)
    _, _, fitted, (held,) = boost(xb, tr["angle"], depth, rate, BRT_TREES, (hb,))
    return (np.sqrt(((fitted - tr["angle"]) ** 2).mean(1)),
            np.sqrt(((held - te["angle"]) ** 2).mean(1)), held[-1])


def _brt_build():
    jobs = [(s, r, d) for r in BRT_RATES for d in BRT_DEPTHS for s in BRT_SEEDS]
    runs = _parallel(brt_run, jobs)
    shape = (len(BRT_RATES), len(BRT_DEPTHS), len(BRT_SEEDS))
    out = {"one_" + k: v for k, v in one_feature().items()}
    out.update(train=np.array([r[0] for r in runs]).reshape(*shape, -1),
               held=np.array([r[1] for r in runs]).reshape(*shape, -1),
               final=np.array([r[2] for r in runs], np.float32).reshape(*shape, -1))
    return out


def brt_data():
    """``held[rate, depth, seed, trees]`` and ``train``: error against the number of
    trees; ``final``: held-out predictions after the last tree; ``one_*``: the
    one-feature example."""
    return _cache("trees", _brt_build, (rotate, cells, tilt, binned, fit_tree, tree_predict,
                                        boost, one_feature, brt_run, reg_train, reg_heldout),
                  rates=BRT_RATES, depths=BRT_DEPTHS, trees=BRT_TREES, seeds=BRT_SEEDS,
                  bins=BINS, one=(ONE_N, ONE_DEPTH, ONE_RATE, ONE_TREES, ONE_RANGE, ONE_BINS))


def brt_summary():
    d, net = brt_data(), net_summary()
    truth = reg_heldout()["angle"]
    mean = d["held"].mean(2)                           # rate, depth, trees
    best_at = mean.argmin(-1)                          # trees at the lowest mean error
    lowest = mean.min(-1)
    rise = mean[..., -1] - lowest                      # how far it climbs back by the end
    width = (d["final"] - truth).std(-1)               # resolution after the last tree
    i, j = BRT_RATES.index(BRT_MAIN[0]), BRT_DEPTHS.index(BRT_MAIN[1])
    k = np.unravel_index(np.argmin(mean[..., -1]), mean[..., -1].shape)
    per = [resolution(d["final"][k[0], k[1], s], truth) for s in range(len(BRT_SEEDS))]
    n = len(BRT_SEEDS)
    return dict(d=d, mean=mean, best_at=best_at, lowest=lowest, rise=rise, width=width,
                main=(i, j), winner=(int(k[0]), int(k[1])),
                winner_bias=np.array([p[0] for p in per]),
                winner_width=np.array([p[1] for p in per]),
                net_rms=net["held_rms"][:n, -1], net_width=net["overall_width"][:n],
                net_bins=net["width"][:n], net_bias=net["bias"][:n])


def boosted_regression_trees():
    r = brt_summary()
    d = r["d"]
    count = np.arange(BRT_TREES + 1)
    i, j = r["main"]
    wi, wj = r["winner"]
    names = ["rate %g, depth %d" % (a, b) for a in BRT_RATES for b in BRT_DEPTHS]
    final = r["mean"][..., -1]
    centres = (REG_EDGES[:-1] + REG_EDGES[1:]) / 2
    return Study(
        "Boosted trees for regression",
        "Small regression trees added one at a time, each fitted to the residuals the "
        "sum so far leaves: on one real feature, then on the network's 49 cells",
        (
            Panel("One feature: the sum of trees as a staircase",
                  "xy: tilt of the ink and true angle of %d training events. steps[m]: "
                  "the sum of the first m trees over the tilt values in grid (depth %d, "
                  "each tree added at %g of its size); alone[m]: tree m + 1 by itself; "
                  "residuals[m]: what the events still lack after m trees."
                  % (ONE_N, ONE_DEPTH, ONE_RATE), "points",
                  dict(xy=np.column_stack([d["one_x"], d["one_y"]]), steps=d["one_steps"],
                       alone=d["one_alone"], grid=ONE_GRID, residuals=d["one_residuals"],
                       start=d["one_start"], train_rms=d["one_train_rms"],
                       held_rms=d["one_held_rms"], cuts=d["one_cuts"],
                       images=d["one_images"],
                       leaves=d["one_leaves"])),
            Panel("49 cells: held-out error against the number of trees",
                  "ys: held-out root-mean-square error in degrees after x trees, mean of "
                  "%d seeds, for each learning rate at depth %d. per_seed has the seeds; "
                  "train the same on the training events."
                  % (len(BRT_SEEDS), BRT_DEPTHS[j]), "curve",
                  dict(x=count, ys=r["mean"][:, j], labels=["rate %g" % a for a in BRT_RATES],
                       per_seed=d["held"][:, j], train=d["train"][:, j].mean(1))),
            Panel("49 cells: every learning rate and depth",
                  "values: held-out root-mean-square error after %d trees, mean of %d "
                  "seeds, for each setting. lowest: the lowest it reached on the way; "
                  "lowest_at: after how many trees." % (BRT_TREES, len(BRT_SEEDS)), "bars",
                  dict(values=final.ravel(), labels=names, lowest=r["lowest"].ravel(),
                       lowest_at=r["best_at"].ravel(), grid=final,
                       all_curves=r["mean"], all_train=d["train"].mean(2))),
            Panel("Trees against the network, by true angle",
                  "ys: resolution (standard deviation of predicted minus true) in nine "
                  "bins of the true angle x for the best tree setting (%s) and for the "
                  "network, same training events and held-out events, mean of %d seeds."
                  % (names[wi * len(BRT_DEPTHS) + wj], len(BRT_SEEDS)), "curve",
                  dict(x=centres, ys=np.array([r["winner_width"].mean(0),
                                               r["net_bins"].mean(0)]),
                       labels=["boosted trees", "network"], trees_bias=r["winner_bias"],
                       network_bias=r["net_bias"])),
        ),
        dict(
            seeds=list(BRT_SEEDS), mnist=True, review_only=True,
            source="NumPy boosted trees on MNIST", dataset="MNIST", stand_in=STAND_IN,
            physics_counterpart="An analysis regresses a particle's energy from detector "
                                "readings with boosted trees in the same way.",
            task="The network's task and events (voxanim.study_phys_apps.reg_train, "
                 "reg_heldout): the angle a digit was turned by, from its 7 x 7 cells.",
            model="least-squares regression trees on features cut at %d evenly spaced "
                  "thresholds, at least 5 events a leaf; boosting from the mean, each "
                  "tree fitted to the residuals and added at the learning rate; %d trees"
                  % (BINS - 1, BRT_TREES),
            one_feature=dict(feature="tilt of the ink from its second moments, degrees",
                             events=ONE_N, depth=ONE_DEPTH, rate=ONE_RATE, trees=ONE_TREES,
                             start=float(d["one_start"]),
                             train_rms=[float(d["one_train_rms"][m]) for m in (0, *ONE_SHOWN)],
                             held_rms=[float(d["one_held_rms"][m]) for m in (0, *ONE_SHOWN)],
                             held_lowest=float(d["one_held_rms"].min()),
                             held_lowest_at=int(d["one_held_rms"].argmin())),
            rates=list(BRT_RATES), depths=list(BRT_DEPTHS),
            held_rms={n: float(v) for n, v in zip(names, final.ravel())},
            lowest_rms={n: float(v) for n, v in zip(names, r["lowest"].ravel())},
            lowest_at={n: int(v) for n, v in zip(names, r["best_at"].ravel())},
            rise_after_lowest={n: float(v) for n, v in zip(names, r["rise"].ravel())},
            main=dict(rate=BRT_MAIN[0], depth=BRT_MAIN[1], held=float(final[i, j]),
                      train=float(d["train"][i, j].mean(0)[-1])),
            best=dict(rate=BRT_RATES[wi], depth=BRT_DEPTHS[wj], held=float(final[wi, wj]),
                      per_seed=_floats(d["held"][wi, wj, :, -1]),
                      train=float(d["train"][wi, wj].mean(0)[-1]),
                      resolution=float(r["width"][wi, wj].mean())),
            network=dict(held=float(r["net_rms"].mean()), per_seed=_floats(r["net_rms"]),
                         resolution=float(r["net_width"].mean())),
            split="training: the network's %d events per seed; held-out: its %d events"
                  % (REG_TRAIN, REG_TEST),
            where_the_seed_enters="the trees are deterministic; a seed is the network's "
                                  "seed of the same number: numpy.random.default_rng("
                                  "[150, 1, seed]) picks the training events and angles",
            evidence=dict(EVIDENCE),
            limitations=[
                "Digits stand in for events, and the comparison with the network is on "
                "this task, these 49 inputs and %d training events only." % REG_TRAIN,
                "Features are cut at 15 fixed thresholds, a coarser search than a "
                "library's; %d trees is where every run stops, whether or not its "
                "error was still falling." % BRT_TREES,
                "The one-feature example uses %d events so that each can be seen."
                % ONE_N,
            ],
            designs_tried=[
                "For the one-feature example the mean ink of the image was looked at "
                "first: its correlation with the angle is -0.01, so a staircase in it "
                "stays flat. The tilt of the ink (correlation 0.6) was used instead.",
                "The grid of rates and depths was fixed before any run; the network was "
                "trained first and not changed afterwards.",
            ],
        ),
    )


# ====================================================================== the two classes
CLS_SIGNAL, CLS_BACKGROUND = 4, 9
CLS_SPLIT = 151
CLS_BLOCK = 7                     # cells are 7 x 7 pixels: 4 x 4 = 16 readings an event
CLS_POOL = 4000                   # events of each class that training may draw from
CLS_PICTURES = 4                  # held-out events of each class drawn as pictures


@functools.lru_cache(maxsize=1)
def cls_data():
    """The 4s (signal, label 1) and 9s (background, label 0) as 16 cells. ``pool``:
    ``CLS_POOL`` of each from the training file, for training and validation.
    ``held``: every other 4 and 9 of the training file and all of the test file."""
    xtr, ytr, xte, yte = mnist.load()
    rng = np.random.default_rng([CLS_SPLIT, 0])
    pool, held, held_image = [], [], []
    for digit in (CLS_SIGNAL, CLS_BACKGROUND):
        ids = rng.permutation(np.flatnonzero(ytr == digit))
        pool.append(cells(xtr[ids[:CLS_POOL]], CLS_BLOCK))
        rest = np.concatenate([xtr[ids[CLS_POOL:]], xte[yte == digit]])
        held.append(cells(rest, CLS_BLOCK))
        held_image.append(rest[:CLS_PICTURES].reshape(-1, 28, 28))
    return dict(pool=pool, held_x=np.concatenate(held),
                held_y=np.r_[np.ones(len(held[0])), np.zeros(len(held[1]))],
                image=np.array(held_image))


def cls_draw(n, seed, extra=0):
    """Seed ``seed``'s training sample, ``n`` events of each class from the pool, and
    ``extra`` more of each (never the same events) for validation."""
    pool = cls_data()["pool"]
    rng = np.random.default_rng([CLS_SPLIT, 1, seed])
    order = [rng.permutation(CLS_POOL) for _ in pool]
    x = np.concatenate([p[o[:n]] for p, o in zip(pool, order)])
    v = np.concatenate([p[o[n:n + extra]] for p, o in zip(pool, order)])
    return x, np.r_[np.ones(n), np.zeros(n)], v, np.r_[np.ones(extra), np.zeros(extra)]


def auc(signal, background):
    """Area under the ROC curve: the chance that a random signal event scores above a
    random background event (ties count a half)."""
    b = np.sort(np.asarray(background, float))
    below = np.searchsorted(b, signal, side="left")
    upto = np.searchsorted(b, signal, side="right")
    return float((below + 0.5 * (upto - below)).mean() / len(b))


def ks_distance(a, b):
    """The Kolmogorov-Smirnov distance of two samples: the largest gap between their
    cumulative distributions. Returns ``(D, where)``."""
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    every = np.concatenate([a, b])
    gap = np.abs(np.searchsorted(a, every, side="right") / len(a)
                 - np.searchsorted(b, every, side="right") / len(b))
    k = int(gap.argmax())
    return float(gap[k]), float(every[k])


def kolmogorov_tail(x):
    """Kolmogorov's distribution: the chance that sqrt(n) D exceeds ``x`` when a sample
    of size n is compared with the distribution it was drawn from, as n grows:
    2 sum_k (-1)^(k-1) exp(-2 k^2 x^2)."""
    x = np.atleast_1d(np.asarray(x, float))
    k = np.arange(1, 101)[:, None]
    total = 2.0 * ((-1.0) ** (k - 1) * np.exp(-2.0 * k ** 2 * np.maximum(x, 0.2) ** 2)).sum(0)
    return np.where(x < 0.2, 1.0, np.clip(total, 0.0, 1.0))


def ks_p_value(d, n, m):
    """The chance of a distance of ``d`` or more between samples of ``n`` and ``m``
    events drawn from one distribution (large-sample formula)."""
    return float(kolmogorov_tail(np.sqrt(n * m / (n + m)) * d)[0])


def ks_critical(n, m, level=0.05):
    """The distance that samples of ``n`` and ``m`` events from one distribution exceed
    with chance ``level`` (large-sample formula; 1.358 sqrt((n + m) / (n m)) at 5%)."""
    lo, hi = 0.2, 3.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if kolmogorov_tail(mid)[0] > level else (lo, mid)
    return float(lo * np.sqrt((n + m) / (n * m)))


def efficiencies(signal, background, cuts):
    """The share of signal and of background scores above each cut."""
    s, b = np.sort(signal), np.sort(background)
    return (1.0 - np.searchsorted(s, cuts, side="right") / len(s),
            1.0 - np.searchsorted(b, cuts, side="right") / len(b))


# ====================================================================== 151: the score
SCORE_N = 2000                    # training events of each class
SCORE_HIDDEN = (32, 32)
SCORE_RATE, SCORE_BATCH, SCORE_UPDATES = 1e-3, 50, 2400
SCORE_SEEDS = (0, 1, 2, 3, 4)
SCORE_BINS = 25
CUTS = np.round(np.linspace(0.0, 1.0, 101), 2)
RATIO = 50                        # background events for each signal event, when imbalanced
FIT_BINS = 20
FIT_FRACTIONS = (0.05, 0.1, 0.2, 0.3, 0.5)
FIT_EVENTS = 1000                 # events in each pseudo-experiment
FIT_TOYS = 2000                   # pseudo-experiments per fraction and seed
FIT_SHOWN = 2                     # which fraction's first pseudo-experiment is drawn
FIT_CUT = 0.5                     # the cut of the count-above-a-cut estimate


def score_run(seed):
    """One classifier: 16 cells -> 32 -> 32 ReLU -> one unit -> sigmoid, cross-entropy.
    Returns the output unit's value (before the sigmoid) for the held-out and the
    training events."""
    x, y, _, _ = cls_draw(SCORE_N, seed)
    P = net_init(np.random.default_rng([CLS_SPLIT, 2, seed]), (x.shape[1], *SCORE_HIDDEN, 1))
    train_net(P, x, y, "logistic", np.random.default_rng([CLS_SPLIT, 3, seed]), SCORE_UPDATES,
              SCORE_BATCH, SCORE_RATE)
    return net_output(P, cls_data()["held_x"]), net_output(P, x)


def significance(s, b):
    """s / sqrt(s + b): the signal in units of the spread of the count."""
    return s / np.sqrt(np.maximum(s + b, 1e-12))


def fit_fraction(counts, signal, background):
    """The signal fraction f that maximises the binned log-likelihood
    ln L(f) = sum_i n_i ln(f S_i + (1 - f) B_i) of ``counts`` (``(toys, bins)``) given
    the two normalised shapes; by bisection on its slope, which falls steadily with f.
    Returns ``(f, sigma)``: sigma from the curvature of ln L at its maximum."""
    n = np.atleast_2d(np.asarray(counts, float))
    diff = signal - background
    lo, hi = np.zeros(len(n)), np.ones(len(n))
    for _ in range(40):
        mid = (lo + hi) / 2
        slope = (n * diff / (mid[:, None] * signal + (1 - mid[:, None]) * background)).sum(1)
        lo, hi = np.where(slope > 0, mid, lo), np.where(slope > 0, hi, mid)
    f = (lo + hi) / 2
    curve = (n * diff ** 2 / (f[:, None] * signal + (1 - f[:, None]) * background) ** 2).sum(1)
    return f, 1.0 / np.sqrt(np.maximum(curve, 1e-12))


def log_likelihood(counts, signal, background, f):
    """ln L at each fraction in ``f`` for one histogram of ``counts``."""
    f = np.asarray(f, float)[:, None]
    return (np.asarray(counts, float) * np.log(f * signal + (1 - f) * background)).sum(1)


def shapes(scores, bins=FIT_BINS):
    """Scores as a normalised histogram that is nowhere zero (half an event is added to
    every bin, so that no event is impossible under the fit)."""
    h = np.histogram(scores, np.linspace(0, 1, bins + 1))[0] + 0.5
    return h / h.sum()


def fit_toys(signal_scores, background_scores, seed):
    """The mixture fit for one classifier. The held-out scores of each class are split
    in two at random: the first halves give the shapes (and the efficiencies of the
    cut), the second halves are drawn from, with replacement, to make ``FIT_TOYS``
    samples of ``FIT_EVENTS`` events at each known signal fraction f: every event is a
    signal event with chance f. Returns the fitted fractions, their errors from the
    likelihood, the count-above-a-cut estimates, the first sample of each fraction and
    how many signal events it held."""
    rng = np.random.default_rng([CLS_SPLIT, 4, seed])
    halves = []
    for scores in (signal_scores, background_scores):
        order = rng.permutation(len(scores))
        halves.append((scores[order[:len(scores) // 2]], scores[order[len(scores) // 2:]]))
    (s_shape, s_draw), (b_shape, b_draw) = halves
    S, B = shapes(s_shape), shapes(b_shape)
    eff_s, eff_b = (float((v > FIT_CUT).mean()) for v in (s_shape, b_shape))
    fitted, errors, counted, first, first_signal = [], [], [], [], []
    for frac in FIT_FRACTIONS:
        is_signal = rng.random((FIT_TOYS, FIT_EVENTS)) < frac
        s = s_draw[rng.integers(0, len(s_draw), (FIT_TOYS, FIT_EVENTS))]
        b = b_draw[rng.integers(0, len(b_draw), (FIT_TOYS, FIT_EVENTS))]
        both = np.where(is_signal, s, b)
        which = np.clip((both * FIT_BINS).astype(int), 0, FIT_BINS - 1)
        counts = np.array([np.bincount(row, minlength=FIT_BINS) for row in which])
        f, sigma = fit_fraction(counts, S, B)
        fitted.append(f)
        errors.append(sigma)
        counted.append(((both > FIT_CUT).mean(1) - eff_b) / (eff_s - eff_b))
        first.append(counts[0])
        first_signal.append(int(is_signal[0].sum()))
    return dict(fitted=np.array(fitted), errors=np.array(errors), counted=np.array(counted),
                first=np.array(first), first_signal=np.array(first_signal), S=S, B=B,
                eff=np.array([eff_s, eff_b]))


def _score_build():
    runs = _parallel(score_run, [(s,) for s in SCORE_SEEDS])
    held = np.array([r[0] for r in runs])
    y = cls_data()["held_y"]
    toys = [fit_toys(sigmoid(h[y == 1]), sigmoid(h[y == 0]), s)
            for h, s in zip(held, SCORE_SEEDS)]
    out = dict(held=held, train=np.array([r[1] for r in runs]))
    for k in toys[0]:
        out["fit_" + k] = np.array([t[k] for t in toys])
    return out


def score_data():
    """``held[seed, event]`` and ``train``: every classifier's output-unit values; and
    ``fit_*``: the mixture fits (:func:`fit_toys`), per seed."""
    return _cache("score", _score_build, (cells, net_grads, train_net, cls_data, cls_draw,
                                          score_run, fit_fraction, shapes, fit_toys),
                  n=SCORE_N, hidden=SCORE_HIDDEN, rate=SCORE_RATE, batch=SCORE_BATCH,
                  updates=SCORE_UPDATES, seeds=SCORE_SEEDS, fit=(FIT_BINS, FIT_FRACTIONS,
                                                                FIT_EVENTS, FIT_TOYS, FIT_CUT))


def cut_table(signal, background, ratio):
    """For every cut in :data:`CUTS`: signal efficiency, background efficiency, and,
    with ``ratio`` background events for each signal event, purity s / (s + b) and
    significance s / sqrt(s + b) for 1,000 signal events."""
    eff_s, eff_b = efficiencies(signal, background, CUTS)
    s, b = 1000.0 * eff_s, 1000.0 * ratio * eff_b
    return dict(eff_s=eff_s, eff_b=eff_b, purity=s / np.maximum(s + b, 1e-12),
                z=significance(s, b))


def score_summary():
    d, y = score_data(), cls_data()["held_y"]
    scores = sigmoid(d["held"])
    sig, bkg = scores[:, y == 1], scores[:, y == 0]
    edges = np.linspace(0, 1, SCORE_BINS + 1)
    hist = np.array([[np.histogram(v, edges)[0] / len(v) for v in (s, b)]
                     for s, b in zip(sig, bkg)])                  # seed, class, bin
    even = [cut_table(s, b, 1) for s, b in zip(sig, bkg)]
    many = [cut_table(s, b, RATIO) for s, b in zip(sig, bkg)]
    best_even = np.array([int(np.argmax(t["z"])) for t in even])
    best_many = np.array([int(np.argmax(t["z"])) for t in many])
    half = int(np.flatnonzero(CUTS == 0.5)[0])
    tr = d["train"]
    return dict(
        d=d, sig=sig, bkg=bkg, hist=hist, even=even, many=many, best_even=best_even,
        best_many=best_many, half=half,
        auc=np.array([auc(a, b) for a, b in zip(sig, bkg)]),
        train_auc=np.array([auc(t[:SCORE_N], t[SCORE_N:]) for t in tr]),
        fit_mean=d["fit_fitted"].mean(-1), fit_std=d["fit_fitted"].std(-1),
        fit_error=d["fit_errors"].mean(-1),
        count_mean=d["fit_counted"].mean(-1), count_std=d["fit_counted"].std(-1))


def _pick(tables, key, at):
    return np.array([t[key][a] for t, a in zip(tables, at)])


def classifier_score_as_variable():
    r = score_summary()
    d = r["d"]
    half = np.full(len(SCORE_SEEDS), r["half"])
    even, many = r["even"], r["many"]
    n_s, n_b = r["sig"].shape[1], r["bkg"].shape[1]
    return Study(
        "A classifier's score as a variable",
        "The 4s as signal and the 9s as background: a network's score on held-out "
        "digits, a cut on it, the cut that is best with %d times more background, and "
        "the signal fraction of a mixture fitted from the score's shape" % RATIO,
        (
            Panel("The score of held-out signal and background",
                  "values[class, bin]: share of the %d held-out 4s (signal, row 0) and "
                  "%d held-out 9s (background, row 1) in each of %d bins of the score "
                  "(seed 0). per_seed has every seed." % (n_s, n_b, SCORE_BINS), "matrix",
                  dict(values=r["hist"][0], per_seed=r["hist"],
                       edges=np.linspace(0, 1, SCORE_BINS + 1), images=cls_data()["image"],
                       example_scores=np.array([r["sig"][0, :CLS_PICTURES],
                                                r["bkg"][0, :CLS_PICTURES]]))),
            Panel("What a cut keeps: the ROC curve",
                  "ys: share of signal and of background with a score above each cut x "
                  "(mean of %d seeds); per_seed_signal and per_seed_background have the "
                  "seeds. The ROC curve is one plotted against the other."
                  % len(SCORE_SEEDS), "curve",
                  dict(x=CUTS, ys=np.array([np.mean([t["eff_s"] for t in even], 0),
                                            np.mean([t["eff_b"] for t in even], 0)]),
                       labels=["signal efficiency", "background efficiency"],
                       per_seed_signal=np.array([t["eff_s"] for t in even]),
                       per_seed_background=np.array([t["eff_b"] for t in even]))),
            Panel("Purity and significance with %d times more background" % RATIO,
                  "ys: purity s / (s + b) and significance s / sqrt(s + b) above each "
                  "cut x for 1,000 signal events and %d background events (the held-out "
                  "shares scaled to those numbers), and the same with equal numbers; "
                  "seed 0. per_seed_z has the significance of every seed."
                  % (1000 * RATIO), "curve",
                  dict(x=CUTS, ys=np.array([many[0]["purity"], many[0]["z"],
                                            even[0]["purity"], even[0]["z"]]),
                       labels=["purity, 1:%d" % RATIO, "significance, 1:%d" % RATIO,
                               "purity, 1:1", "significance, 1:1"],
                       per_seed_z=np.array([t["z"] for t in many]),
                       per_seed_purity=np.array([t["purity"] for t in many]))),
            Panel("The signal fraction recovered by fitting the score's shape",
                  "values: mean fitted fraction over %d pseudo-experiments of %d events "
                  "for each true fraction in true (seed 0). spread: their standard "
                  "deviation; counted, counted_spread: the same for counting events "
                  "above a cut of %g; per_seed arrays have every seed."
                  % (FIT_TOYS, FIT_EVENTS, FIT_CUT), "bars",
                  dict(values=r["fit_mean"][0], labels=["%g" % f for f in FIT_FRACTIONS],
                       true=np.array(FIT_FRACTIONS), spread=r["fit_std"][0],
                       error=r["fit_error"][0], counted=r["count_mean"][0],
                       counted_spread=r["count_std"][0], per_seed=r["fit_mean"],
                       per_seed_spread=r["fit_std"], per_seed_counted_spread=r["count_std"],
                       first=d["fit_first"][0], signal_shape=d["fit_S"][0],
                       background_shape=d["fit_B"][0],
                       first_signal=d["fit_first_signal"][0],
                       fitted_shown=d["fit_fitted"][0, FIT_SHOWN],
                       error_shown=d["fit_errors"][0, FIT_SHOWN])),
        ),
        dict(
            seeds=list(SCORE_SEEDS), mnist=True, review_only=True,
            source="NumPy training on MNIST", dataset="MNIST", stand_in=STAND_IN,
            physics_counterpart="An analysis cuts on, or fits, the score of a classifier "
                                "trained to tell simulated signal from background in "
                                "the same way.",
            classes=dict(signal=CLS_SIGNAL, background=CLS_BACKGROUND,
                         why="4 and 9 are a pair of digits that are easily confused, and "
                             "read as 4 x 4 cells they overlap as signal and background "
                             "do; the rarer thing looked for is called the signal"),
            model="16 -> %d -> %d ReLU -> 1 -> sigmoid; cross-entropy; Adam %.0e, batch "
                  "%d, %d updates on %d events of each class"
                  % (*SCORE_HIDDEN, SCORE_RATE, SCORE_BATCH, SCORE_UPDATES, SCORE_N),
            split="pool: %d of each class from the training file (seeded); held-out: the "
                  "other 4s and 9s of the training file and those of the test file, %d "
                  "4s and %d 9s" % (CLS_POOL, n_s, n_b),
            heldout=dict(signal=n_s, background=n_b),
            where_the_seed_enters="numpy.random.default_rng([151, k, seed]): k = 1 draws "
                                  "the training events, 2 the initial weights, 3 each "
                                  "epoch's order, 4 the halves and the pseudo-experiments; "
                                  "[151, 0] the pool",
            auc=dict(mean=float(r["auc"].mean()), min=float(r["auc"].min()),
                     max=float(r["auc"].max()), train=float(r["train_auc"].mean())),
            cut_at_half=dict(
                signal_efficiency=_floats(_pick(even, "eff_s", half)),
                background_rejection=_floats(1 - _pick(even, "eff_b", half)),
                purity_even=_floats(_pick(even, "purity", half)),
                purity_imbalanced=_floats(_pick(many, "purity", half))),
            best_cut=dict(
                even=_floats(CUTS[r["best_even"]]),
                imbalanced=_floats(CUTS[r["best_many"]]),
                purity_even=_floats(_pick(even, "purity", r["best_even"])),
                purity_imbalanced=_floats(_pick(many, "purity", r["best_many"])),
                signal_efficiency_imbalanced=_floats(_pick(many, "eff_s", r["best_many"])),
                z_imbalanced=_floats(_pick(many, "z", r["best_many"])),
                z_imbalanced_at_half=_floats(_pick(many, "z", half))),
            ratio=RATIO,
            fit=dict(fractions=list(FIT_FRACTIONS), events=FIT_EVENTS, toys=FIT_TOYS,
                     bins=FIT_BINS,
                     mean=_floats(r["fit_mean"].mean(0)),
                     spread=_floats(r["fit_std"].mean(0)),
                     likelihood_error=_floats(r["fit_error"].mean(0)),
                     counted_mean=_floats(r["count_mean"].mean(0)),
                     counted_spread=_floats(r["count_std"].mean(0)),
                     perfect_spread=_floats(np.sqrt(np.array(FIT_FRACTIONS)
                                                    * (1 - np.array(FIT_FRACTIONS))
                                                    / FIT_EVENTS)),
                     worst_offset=float(np.abs(r["fit_mean"] - np.array(FIT_FRACTIONS)).max())),
            evidence=dict(EVIDENCE),
            limitations=[
                "Digits stand in for events. The coarse 4 x 4 read-out was chosen so "
                "that the classes overlap; with every pixel the same kind of network "
                "separates 4s from 9s almost perfectly.",
                "The imbalanced numbers are the held-out shares scaled to 1,000 signal "
                "and %d background events, not a sample of that size: the background "
                "share above a high cut rests on few held-out 9s." % (1000 * RATIO),
                "The pseudo-experiments draw with replacement from half of the held-out "
                "scores, with the shapes from the other half; the shapes' own "
                "uncertainty shows as a small offset of the mean fitted fraction.",
            ],
            designs_tried=[
                "A pilot (one seed, before any cut or fit was computed) compared inputs "
                "of 784 pixels, 14 x 14, 7 x 7 and 4 x 4 cells: ROC areas 0.998, 0.997, "
                "0.995 and 0.92. The 4 x 4 read-out was kept for its overlap.",
                "The figure of merit s / sqrt(s + b), the 1:%d ratio, the bins and the "
                "fractions were fixed before the cuts were scanned." % RATIO,
            ],
        ),
    )


# ====================================================================== 153: overtraining
OT_SIZES = (25, 50, 100, 200, 500, 1000, 2000)    # training events of each class
OT_HIDDEN = (256, 256)
OT_RATE, OT_BATCH, OT_UPDATES = 1e-3, 50, 3000
OT_SNAPS = tuple(int(v) for v in np.unique(np.round(np.geomspace(10, OT_UPDATES, 18))))
OT_SEEDS = (0, 1, 2, 3, 4)
OT_DECAY = 3.0                    # the weight decay of the remedy (decoupled, times the rate)
OT_VALID = 500                    # validation events of each class, for early stopping
OT_SMALL, OT_LARGE = 100, 2000    # the two sizes drawn
OT_BINS = 20


def ot_run(n, seed, decay, keep=False):
    """One large classifier (16 -> 256 -> 256 -> 1) on ``n`` events of each class. At
    every snapshot: the Kolmogorov-Smirnov distance between the training and held-out
    scores of each class; the ROC area on each; the losses (training, held-out,
    validation); and the score histograms. ``keep``: also the final scores."""
    data = cls_data()
    x, y, v, vy = cls_draw(n, seed, OT_VALID)
    hx, hy = data["held_x"], data["held_y"]
    P = net_init(np.random.default_rng([CLS_SPLIT, 2, seed]), (x.shape[1], *OT_HIDDEN, 1))
    edges = np.linspace(0, 1, OT_BINS + 1)

    def measure(P):
        t, h = net_output(P, x), net_output(P, hx)
        return dict(
            ks=[ks_distance(t[y == c], h[hy == c])[0] for c in (1, 0)],
            auc=[auc(t[y == 1], t[y == 0]), auc(h[hy == 1], h[hy == 0])],
            valid=net_loss(P, v, vy, "logistic"),
            loss=[net_loss(P, x, y, "logistic"), net_loss(P, hx, hy, "logistic")],
            hist=[[np.histogram(sigmoid(z[lab == c]), edges)[0] / (lab == c).sum()
                   for z, lab in ((t, y), (h, hy))] for c in (1, 0)])
    snaps = train_net(P, x, y, "logistic", np.random.default_rng([CLS_SPLIT, 3, seed]),
                      OT_UPDATES, OT_BATCH, OT_RATE, OT_SNAPS, measure, decay)
    out = {k: np.array([s[k] for s in snaps]) for k in snaps[0]}
    if keep:
        t, h = net_output(P, x), net_output(P, hx)
        out.update(train_signal=sigmoid(t[y == 1]), held_signal=sigmoid(h[hy == 1]))
    return out


def _ot_build():
    jobs = [(n, s, w) for w in (0.0, OT_DECAY) for n in OT_SIZES for s in OT_SEEDS]
    runs = _parallel(ot_run, jobs)
    shape = (2, len(OT_SIZES), len(OT_SEEDS))
    out = {k: np.array([r[k] for r in runs]).reshape(*shape, *np.shape(runs[0][k]))
           for k in runs[0]}
    one = ot_run(OT_SMALL, 0, 0.0, keep=True)
    out.update(example_train=np.sort(one["train_signal"]),
               example_held=np.sort(one["held_signal"]))
    return out


def ot_data():
    """Indexed ``[decay (none, some), size, seed, snapshot, ...]``: ``ks`` (signal,
    background), ``auc`` (training, held-out), ``valid`` (validation loss), ``loss``
    (training, held-out) and ``hist[class, (training, held-out), bin]``. And the final
    signal scores of the small sample (seed 0), sorted (``example_*``)."""
    return _cache("overtraining", _ot_build, (cells, net_grads, train_net, cls_data, cls_draw,
                                              ks_distance, auc, ot_run),
                  sizes=OT_SIZES, hidden=OT_HIDDEN, rate=OT_RATE, batch=OT_BATCH,
                  updates=OT_UPDATES, snaps=OT_SNAPS, seeds=OT_SEEDS, decay=OT_DECAY,
                  valid=OT_VALID, bins=OT_BINS)


OT_ROWS = ("trained to the end", "stopped early", "weight decay")


def ot_summary():
    d = ot_data()
    y = cls_data()["held_y"]
    n_held = int((y == 1).sum()), int((y == 0).sum())
    ks = d["ks"].mean(-1)                               # decay, size, seed, snap
    gap = d["auc"][..., 0] - d["auc"][..., 1]
    stop = d["valid"].argmin(-1)                        # decay, size, seed

    def at_stop(a):
        return np.take_along_axis(a, stop[0][..., None], -1)[..., 0]
    critical = np.array([np.mean([ks_critical(n, m) for m in n_held]) for n in OT_SIZES])
    ex_d, ex_at = ks_distance(d["example_train"], d["example_held"])
    return dict(d=d, ks=ks, gap=gap, stop=stop, critical=critical, n_held=n_held,
                held_auc=d["auc"][..., 1], example=(ex_d, ex_at),
                rows=dict(
                    ks=np.array([ks[0, ..., -1], at_stop(ks[0]), ks[1, ..., -1]]),
                    gap=np.array([gap[0, ..., -1], at_stop(gap[0]), gap[1, ..., -1]]),
                    auc=np.array([d["auc"][0, ..., -1, 1], at_stop(d["auc"][0, ..., 1]),
                                  d["auc"][1, ..., -1, 1]])))


def overtraining_check():
    r = ot_summary()
    d, rows = r["d"], r["rows"]
    sizes, snaps = np.array(OT_SIZES), np.array(OT_SNAPS)
    a, b = OT_SIZES.index(OT_SMALL), OT_SIZES.index(OT_LARGE)

    def spread(v):
        return dict(mean=_floats(v.mean(-1)), min=_floats(v.min(-1)), max=_floats(v.max(-1)))
    ex_d, ex_at = r["example"]
    return Study(
        "The overtraining check",
        "The score distributions of a classifier's training events and of held-out "
        "events, their Kolmogorov-Smirnov distance and the gap in ROC area, against the "
        "size of the training sample and the time trained",
        (
            Panel("Training against held-out scores, small and large samples",
                  "values[4 x case + 2 x class + set, bin]: share of events in each of "
                  "%d score bins after the last update (seed 0); case 0 is %d training "
                  "events a class, case 1 is %d; class 0 is signal; set 0 is training, 1 "
                  "held-out. during and during_large: the same at every snapshot."
                  % (OT_BINS, OT_SMALL, OT_LARGE), "matrix",
                  dict(values=np.array([d["hist"][0, a, 0, -1], d["hist"][0, b, 0, -1]])
                       .reshape(8, OT_BINS),
                       during=d["hist"][0, a, 0], during_large=d["hist"][0, b, 0],
                       updates=snaps, example_train=d["example_train"],
                       example_held=d["example_held"])),
            Panel("Distance and gap during training",
                  "ys: Kolmogorov-Smirnov distance between training and held-out scores "
                  "(mean of the two classes) and training-minus-held-out ROC area, for "
                  "%d events a class, mean of %d seeds, after x updates; then the same "
                  "for %d a class." % (OT_SMALL, len(OT_SEEDS), OT_LARGE), "curve",
                  dict(x=snaps, ys=np.array([r["ks"][0, a].mean(0), r["gap"][0, a].mean(0),
                                             r["ks"][0, b].mean(0), r["gap"][0, b].mean(0)]),
                       labels=["distance, small", "gap, small", "distance, large",
                               "gap, large"],
                       held_auc=r["held_auc"][0, a].mean(0), valid=d["valid"][0, a].mean(0),
                       held_loss=d["loss"][0, a, :, :, 1].mean(0),
                       train_loss=d["loss"][0, a, :, :, 0].mean(0),
                       stop=snaps[r["stop"][0, a]],
                       decay_ks=r["ks"][1, a].mean(0), decay_gap=r["gap"][1, a].mean(0),
                       decay_held_auc=r["held_auc"][1, a].mean(0))),
            Panel("Distance against the size of the training sample",
                  "ys: Kolmogorov-Smirnov distance (mean of the two classes, mean of %d "
                  "seeds) for x training events a class: trained to the end, stopped "
                  "where the validation loss was lowest, and with weight decay. "
                  "critical: the distance two samples of one distribution exceed 5%% of "
                  "the time (large-sample formula)." % len(OT_SEEDS), "curve",
                  dict(x=sizes, ys=rows["ks"].mean(-1), labels=list(OT_ROWS),
                       per_seed=rows["ks"], critical=r["critical"])),
            Panel("ROC-area gap against the size of the training sample",
                  "ys: ROC area on the training events minus on the held-out events for "
                  "the same three, mean of %d seeds; held: the held-out ROC area."
                  % len(OT_SEEDS), "curve",
                  dict(x=sizes, ys=rows["gap"].mean(-1), labels=list(OT_ROWS),
                       per_seed=rows["gap"], held=rows["auc"].mean(-1),
                       held_per_seed=rows["auc"])),
        ),
        dict(
            seeds=list(OT_SEEDS), mnist=True, review_only=True,
            source="NumPy training on MNIST", dataset="MNIST", stand_in=STAND_IN,
            physics_counterpart="An analysis overlays the score of its training events "
                                "on that of independent test events, for simulated signal "
                                "and background, in the same way.",
            classes=dict(signal=CLS_SIGNAL, background=CLS_BACKGROUND),
            model="16 -> %d -> %d ReLU -> 1 -> sigmoid; cross-entropy; Adam %.0e, batch "
                  "%d, %d updates whatever the sample size"
                  % (*OT_HIDDEN, OT_RATE, OT_BATCH, OT_UPDATES),
            remedies=dict(
                early_stopping="the snapshot with the lowest loss on %d validation "
                               "events a class, drawn from the pool and never trained on"
                               % OT_VALID,
                weight_decay="after every step each weight matrix is multiplied by "
                             "1 - %g x the learning rate" % OT_DECAY,
                stopped_after=_floats(snaps[r["stop"][0]].mean(-1))),
            split="training and validation: from a pool of %d events a class; held-out: "
                  "%d 4s and %d 9s never in the pool" % (CLS_POOL, *r["n_held"]),
            where_the_seed_enters="numpy.random.default_rng([151, k, seed]): k = 1 draws "
                                  "the training and validation events, 2 the initial "
                                  "weights, 3 each epoch's order",
            sizes=list(OT_SIZES), rows=list(OT_ROWS),
            ks={name: spread(rows["ks"][k]) for k, name in enumerate(OT_ROWS)},
            auc_gap={name: spread(rows["gap"][k]) for k, name in enumerate(OT_ROWS)},
            heldout_auc={name: spread(rows["auc"][k]) for k, name in enumerate(OT_ROWS)},
            critical=_floats(r["critical"]),
            example=dict(events=OT_SMALL, distance=ex_d, at=ex_at,
                         p_value=ks_p_value(ex_d, OT_SMALL, r["n_held"][0])),
            small=OT_SMALL, large=OT_LARGE,
            evidence=dict(EVIDENCE),
            limitations=[
                "Digits stand in for events; 16 inputs and one large network.",
                "The 5% level is Kolmogorov's large-sample formula for two samples of "
                "one distribution; with 25 training events a class only a distance "
                "above about 0.27 stands out from chance.",
                "Early stopping uses %d validation events a class, more than the "
                "smallest training samples themselves." % OT_VALID,
            ],
            designs_tried=[
                "Weight decays of 3, 30 and 100 (times the learning rate) were tried at "
                "100 and 2,000 events a class, one seed: at 30 and 100 the network "
                "stops learning, so 3 was kept.",
                "The number of updates is the same for every sample size, so that "
                "'training time' means the same thing along the size axis.",
            ],
        ),
    )


STUDIES = {
    "network-regression": network_regression,
    "boosted-regression-trees": boosted_regression_trees,
    "classifier-score-as-variable": classifier_score_as_variable,
    "overtraining-check": overtraining_check,
}

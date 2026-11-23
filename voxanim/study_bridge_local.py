"""The experiment behind animation 179, ``dense-layer-learns-local``: the bridge from a
network's capacity to convolutions (see :func:`voxanim.next_catalog.redesigned`).

"The entropy of a network's weights" (174, :mod:`voxanim.study_info`) found that the
first layer of an MNIST classifier, trained, gives neighbouring pixels similar weights.
Here the same layer is read again, and what a convolution makes of it is measured:

* how alike two of a hidden unit's weights are as a function of how many pixels apart
  they sit (pooled over the 64 units, along rows and columns), before and after training;
* the weights a dense layer spends against a 3 x 3 convolution with as many output maps,
  and against the layers in between (one weight per pixel per output; local but not
  shared), counted exactly;
* a small convolutional network (3 x 3 filters, :mod:`voxanim.study_vision`'s numpy CNN)
  trained with the same budget as the classifier (the same 60,000 digits, epochs, batch,
  optimiser and learning rate), and both scored on the 10,000 held-out digits shifted by
  0 to 4 pixels in each of four directions;
* that a convolution's maps of a shifted digit are its maps shifted, checked exactly.

Plain numpy on MNIST, from seeded generators whose seeds are recorded. Training and
held-out digits are disjoint; every accuracy is a held-out one.
"""

from __future__ import annotations

import functools
import hashlib
import inspect

import numpy as np

from . import mnist
from . import study_info as I
from . import study_vision as V
from .cache import cached_npz
from .optim import Adam
from .study import Panel, Study

VERSION = 1
SIDE = 28
PIXELS = SIDE * SIDE
UNITS = I.SIZES[1]                     # the dense first layer: 784 -> 64
KERNEL = 3                             # a 3 x 3 convolution
DISTANCES = np.arange(1, 13)           # pixels apart, along a row or a column
SHIFTS = (0, 1, 2, 3, 4)               # pixels a held-out digit is moved by
DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))     # (down, right): right, left, down, up
DIRECTION_NAMES = ("right", "left", "down", "up")
CNN_FILTERS = (8, 16)                  # 3 x 3 conv, ReLU, 2 x 2 mean pool; twice; dense
CNN_SEED = 179
EPOCHS, BATCH, RATE = I.EPOCHS, I.BATCH, I.RATE     # the classifier's budget, exactly
SHOWN_UNITS = 8                        # hidden units drawn as pictures
EQUIVARIANCE_DIGITS = 100              # held-out digits the exact shift check runs on
EVIDENCE = dict(
    exact="exact arithmetic: weights counted from the layers' shapes",
    measured="measured on MNIST: models trained here in numpy, scored on held-out digits",
)


def _cache(name, build, sources, **settings):
    """``build()``'s arrays, cached under every setting and the source of the functions
    that produce them."""
    h = hashlib.sha256()
    for fn in sources:
        h.update(inspect.getsource(fn).encode())
    settings = dict(version=VERSION, source=h.hexdigest()[:16], **settings)
    return {k: np.asarray(v) for k, v in cached_npz("study-bridge-local-" + name, settings,
                                                    build).items()}


def _floats(v):
    return [float(x) for x in np.asarray(v).ravel()]


# ====================================================================== locality
def pairs_at(W0, d):
    """Every pair of one hidden unit's weights ``d`` pixels apart along a row or a column,
    over all units: ``(a, b)``, the first-layer weights ``W0`` ``(784, units)`` read as
    28 x 28 pictures."""
    img = np.asarray(W0, np.float64).T.reshape(-1, SIDE, SIDE)
    a = np.concatenate([img[:, :, :-d].ravel(), img[:, :-d, :].ravel()])
    b = np.concatenate([img[:, :, d:].ravel(), img[:, d:, :].ravel()])
    return a, b


def correlation_by_distance(W0, distances=DISTANCES):
    """The correlation of two weights of the same hidden unit against the pixel distance
    between them (along rows and columns, pooled over the units)."""
    return np.array([np.corrcoef(*pairs_at(W0, int(d)))[0, 1] for d in distances])


def horizontal_neighbours(W0):
    """The correlation of horizontally neighbouring weights: study 174's measure."""
    return float(np.corrcoef(*I.neighbour_pairs(W0))[0, 1])


def falls_to(distances, corr, share=0.5):
    """The distance (pixels, interpolated) at which ``corr`` first falls to ``share`` of its
    value at the first distance."""
    target = share * corr[0]
    for k in range(1, len(corr)):
        if corr[k] <= target:
            d0, d1, c0, c1 = distances[k - 1], distances[k], corr[k - 1], corr[k]
            return float(d0 + (d1 - d0) * (c0 - target) / (c0 - c1))
    return float("nan")


# ====================================================================== counting weights
def weight_counts(units=UNITS, kernel=KERNEL, side=SIDE):
    """Weights (no biases) of four layers reading one ``side x side`` picture:

    * ``dense``: ``units`` outputs, each with one weight for every pixel;
    * ``conv``: a ``kernel x kernel`` convolution with ``units`` output maps, one small
      filter per map shared by every position;
    * ``outputs``: how many numbers the convolution gives (``units`` maps of
      ``(side - kernel + 1)^2``, no padding);
    * ``dense_same_outputs``: a dense layer giving that many outputs;
    * ``local``: each of those outputs reading its own ``kernel x kernel`` patch, with
      weights of its own (local, not shared)."""
    pixels, out_side = side * side, side - kernel + 1
    outputs = units * out_side * out_side
    return dict(dense=pixels * units, conv=kernel * kernel * units, outputs=outputs,
                out_side=out_side, dense_same_outputs=pixels * outputs,
                local=kernel * kernel * outputs, pixels=pixels, units=units, kernel=kernel)


def cnn_parameters(p):
    return int(sum(np.asarray(v).size for v in p.values()))


# ====================================================================== shifted digits
def shift(x, k, direction):
    """Images ``x`` ``(N, 784)`` moved ``k`` pixels in ``direction`` (``(down, right)``
    unit steps), what leaves the frame dropped and what enters it blank."""
    img = np.asarray(x).reshape(-1, SIDE, SIDE)
    sy, sx = k * direction[0], k * direction[1]
    out = np.zeros_like(img)
    out[:, max(sy, 0):SIDE + min(sy, 0), max(sx, 0):SIDE + min(sx, 0)] = \
        img[:, max(-sy, 0):SIDE + min(-sy, 0), max(-sx, 0):SIDE + min(-sx, 0)]
    return out.reshape(len(img), -1)


def shifted_accuracy(predict, x, y):
    """``(shifts, directions)``: the share of digits named right when every digit is moved
    by each shift in each direction (shift 0: the digits as they are, in every column)."""
    out = np.zeros((len(SHIFTS), len(DIRECTIONS)))
    for i, k in enumerate(SHIFTS):
        for j, direction in enumerate(DIRECTIONS):
            if k == 0 and j:
                out[i, j] = out[i, 0]
                continue
            out[i, j] = float((predict(shift(x, k, direction)) == y).mean())
    return out


def ink_kept(x):
    """``(shifts,)``: the share of the digits' ink still in the frame after each shift,
    over the four directions."""
    total = np.asarray(x, np.float64).sum()
    return np.array([np.mean([shift(x, k, d).sum(dtype=np.float64) / total
                              for d in DIRECTIONS]) for k in SHIFTS])


def mlp_predict(x):
    return I.logits(I.params("after"), np.asarray(x, np.float32)).argmax(1)


# ====================================================================== the CNN
def cnn_loss_grads(p, x, y):
    """Mean cross-entropy (nats) of a batch through :func:`voxanim.study_vision.cnn_forward`
    and its gradient for every parameter."""
    f = V.cnn_forward(p, x)
    lp = I.log_softmax(f["logits"])
    rows = np.arange(len(y))
    loss = float(-lp[rows, y].mean())
    d = np.exp(lp)
    d[rows, y] -= 1.0
    g = V.cnn_backward(p, f, d / len(y))
    del g["x"]
    return loss, g


def train_cnn(seed=CNN_SEED):
    """The CNN trained on the 60,000 training digits with the classifier's budget: ``EPOCHS``
    shuffled passes in batches of ``BATCH``, Adam at ``RATE``. Returns the parameters, the
    batch losses and the held-out accuracy after each epoch."""
    x, y, xt, yt = mnist.load()
    rng = np.random.default_rng([CNN_SEED, int(seed)])
    p = V.cnn_init(rng, CNN_FILTERS, False, False)
    opt = Adam(p, RATE)
    per = len(x) // BATCH
    losses, epoch_acc = [], []
    for _ in range(EPOCHS):
        order = rng.permutation(len(x))
        for b in range(per):
            idx = order[b * BATCH:(b + 1) * BATCH]
            loss, g = cnn_loss_grads(p, x[idx], y[idx])
            opt.step(p, g)
            losses.append(loss)
        epoch_acc.append(float((V.cnn_predict(p, xt) == yt).mean()))
    return p, np.array(losses), np.array(epoch_acc)


def equivariance(p, x, k=2, direction=(0, 1)):
    """The largest difference, over the maps' inner part (where neither picture has met the
    frame), between the first convolution's maps of shifted digits and the maps of the
    digits, shifted. The digits must have ``k`` blank columns or rows on the side they move
    to."""
    z = V.cnn_forward(p, x)["z1"]
    zs = V.cnn_forward(p, shift(x, k, direction))["z1"]
    n = z.shape[1]
    sy, sx = k * direction[0], k * direction[1]
    a = zs[:, max(sy, 0):n + min(sy, 0), max(sx, 0):n + min(sx, 0)]
    b = z[:, max(-sy, 0):n + min(-sy, 0), max(-sx, 0):n + min(-sx, 0)]
    return float(np.abs(a - b).max()), float(np.abs(z).max())


def _build():
    xt, yt = mnist.load()[2:]
    p, losses, epoch_acc = train_cnn()
    out = {"cnn_" + k: v for k, v in p.items()}
    out["cnn_losses"], out["cnn_epoch_accuracy"] = losses, epoch_acc
    out["acc_cnn"] = shifted_accuracy(lambda v: V.cnn_predict(p, v), xt, yt)
    out["acc_mlp"] = shifted_accuracy(mlp_predict, xt, yt)
    out["ink_kept"] = ink_kept(xt)
    # the exact check runs on held-out digits with two blank columns on the right
    img = xt.reshape(-1, SIDE, SIDE)
    room = np.flatnonzero(img[:, :, -2:].max((1, 2)) == 0)[:EQUIVARIANCE_DIGITS]
    out["equivariance"] = np.array(equivariance(p, xt[room]))
    out["equivariance_digits"] = room
    return out


@functools.lru_cache(maxsize=1)
def runs():
    """The trained CNN and both models' held-out accuracy on shifted digits (cached)."""
    return _cache("cnn", _build, (I.init_mlp, I.logits, I.mlp_grads, I.train_mlp, shift,
                                  shifted_accuracy, ink_kept, mlp_predict, cnn_loss_grads,
                                  train_cnn, equivariance, _build, V.conv_cols, V.cols_back,
                                  V.pool, V.pool_back, V.cnn_init, V.cnn_forward,
                                  V.cnn_backward, V.cnn_predict),
                  filters=CNN_FILTERS, seed=CNN_SEED, epochs=EPOCHS, batch=BATCH, rate=RATE,
                  shifts=SHIFTS, directions=DIRECTIONS, digits=EQUIVARIANCE_DIGITS)


def cnn_params():
    r = runs()
    return {k[4:]: r[k] for k in r if k.startswith("cnn_") and k[4:] in
            ("W1", "b1", "W2", "b2", "Wd", "bd")}


# ====================================================================== the study
def dense_layer_learns_local():
    P0, P1 = I.params("before"), I.params("after")
    W0b, W0a = P0["W0"], P1["W0"]
    corr = np.array([correlation_by_distance(W0b), correlation_by_distance(W0a)])
    r = runs()
    acc_mlp, acc_cnn = r["acc_mlp"], r["acc_cnn"]
    mlp_mean, cnn_mean = acc_mlp.mean(1), acc_cnn.mean(1)
    counts = weight_counts()
    pc = cnn_params()
    n_mlp = int(I.flat(P0).size)
    n_cnn = cnn_parameters(pc)
    per = len(mnist.load()[0]) // BATCH
    eq, eq_scale = (float(v) for v in r["equivariance"])
    ladder = [counts["dense_same_outputs"], counts["local"], counts["dense"], counts["conv"]]
    panels = (
        Panel("First-layer weights before and after", "the 784 weights of %d hidden units "
              "of the 784-64-32-10 classifier as 28 x 28 pictures, before and after training"
              % SHOWN_UNITS, "images",
              dict(values=np.concatenate([W0b.T[:SHOWN_UNITS],
                                          W0a.T[:SHOWN_UNITS]]).reshape(-1, SIDE, SIDE),
                   labels=["before"] * SHOWN_UNITS + ["after"] * SHOWN_UNITS)),
        Panel("Near weights agree, far ones do not", "the correlation of two weights of the "
              "same hidden unit against how many pixels apart they are, along rows and "
              "columns, over all 64 units", "curve",
              dict(x=DISTANCES.astype(float), ys=corr, labels=["before", "after"])),
        Panel("Weights to read one picture", "weights of a dense layer of 64 units, of a "
              "3 x 3 convolution with 64 maps, and of the layers giving the convolution's "
              "outputs: dense, and local but not shared (log scale)", "bars",
              dict(values=np.array(ladder, float),
                   labels=["dense, conv's outputs", "local, unshared", "dense, 64 units",
                           "3 x 3 conv, 64 maps"])),
        Panel("Held-out digits, shifted", "the share of the 10,000 held-out digits named "
              "right when each is moved by 0-4 pixels, averaged over four directions",
              "curve", dict(x=np.array(SHIFTS, float), ys=np.array([mlp_mean, cnn_mean]),
                            labels=["dense classifier", "CNN"])),
    )
    return Study(
        "A dense layer learns to be local", "A dense layer's trained weights agree with "
        "their neighbours; a convolution keeps only the neighbours, and shares them",
        panels,
        dict(source="the 784-64-32-10 classifier of study 174 (seed 0) and a 3 x 3 CNN, both "
                    "trained here in numpy on the 60,000 MNIST training digits; accuracy on "
                    "the 10,000 test digits",
             dataset="MNIST", evidence=EVIDENCE, comes_from="The entropy of a network's "
             "weights", leads_to="Convolution, 3×3 kernel",
             distances=[int(d) for d in DISTANCES],
             correlation=dict(before=_floats(corr[0]), after=_floats(corr[1])),
             horizontal_neighbours=dict(before=horizontal_neighbours(W0b),
                                        after=horizontal_neighbours(W0a)),
             half_distance=falls_to(DISTANCES, corr[1]),
             counts=counts, ratio=counts["dense"] / counts["conv"],
             mlp_sizes=list(I.SIZES), mlp_parameters=n_mlp, cnn_filters=list(CNN_FILTERS),
             cnn_parameters=n_cnn, cnn_seed=CNN_SEED, mlp_seed=0, epochs=EPOCHS,
             batch=BATCH, learning_rate=RATE, optimiser="Adam", updates=EPOCHS * per,
             shifts=list(SHIFTS), directions=list(DIRECTION_NAMES),
             accuracy=dict(dense=_floats(mlp_mean), cnn=_floats(cnn_mean)),
             accuracy_by_direction=dict(dense=acc_mlp.tolist(), cnn=acc_cnn.tolist()),
             cnn_epoch_accuracy=_floats(r["cnn_epoch_accuracy"]),
             ink_kept=_floats(r["ink_kept"]),
             equivariance=dict(shift=2, direction="right", digits=int(
                 len(r["equivariance_digits"])), largest_difference=eq, largest_value=eq_scale),
             shown_units=SHOWN_UNITS,
             limitations=["one seed of each model; the classifier's other two seeds of study "
                          "174 are not kept",
                          "the CNN pools and ends in a dense layer, so it is not shift-"
                          "invariant: only its convolutions are shift-equivariant",
                          "shifted digits lose the ink that leaves the 28 x 28 frame",
                          "the correlation pools all pixels, the 67 never lit included"]))


STUDIES = {"dense-layer-learns-local": dense_layer_learns_local}

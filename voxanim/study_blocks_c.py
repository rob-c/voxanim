"""The numbers behind animations 191-193 of the "Blocks and code" series (group ``c``):
``nn.Conv2d``, ``nn.MaxPool2d`` and ``nn.Dropout``.

* ``blocks-conv2d``: region ``conv`` of ``examples/blocks_parts.py``: ``conv =
  nn.Conv2d(1, 4, 3)``, ``maps = conv(x)``, run here on the course's first batch (64 MNIST
  digits as ``examples/mnist_blocks.py`` loads them, :mod:`voxanim.study_course`). The
  classifier of the course has no convolution, so its 4 filters are the ones a fresh
  ``nn.Conv2d(1, 4, 3)`` starts with: PyTorch draws the weights and the biases uniformly
  in ``+-1/sqrt(fan_in)`` = ``+-1/3`` (fan_in = 1 x 3 x 3); here they are drawn the same
  way from a fixed seed (:func:`filters`). One output number is followed in full: 9
  pixels times 9 weights, added up, plus the bias. ``[64, 1, 28, 28] -> [64, 4, 26, 26]``,
  and 4 x 1 x 3 x 3 + 4 = 40 numbers learned, the same 40 at every one of the 676 places.
* ``blocks-pooling``: region ``pool``: ``nn.MaxPool2d(2)`` on those maps: each 2 x 2
  square becomes its biggest number; ``[64, 4, 26, 26] -> [64, 4, 13, 13]``, a quarter
  of the numbers, nothing learned.
* ``blocks-dropout``: region ``dropout`` (``nn.Dropout(0.2)``, ``drop.train()``,
  ``drop.eval()``) on the course's dropout run (``study_course``, run ``dropout``: the
  model of region ``dropout-model``, ``nn.Dropout(0.2)`` after each ReLU): the 64 hidden
  numbers of the first digit of the first batch after the first ReLU, and the switches
  that run really drew for them at its first three steps (each kept number is scaled by
  1 / (1 - 0.2) = 1.25, as ``nn.Dropout`` scales); in ``eval()`` nothing is switched off.
  Measured: the held-out accuracy after one epoch, scored with ``model.eval()`` and
  without it (left in training mode).

Every number shown is exact arithmetic on MNIST digits (the convolution and the pooling
are checked against direct loops; the dropout switches reproduce the recorded run's first
loss exactly), except the two accuracies, which are measured.
"""

from __future__ import annotations

import numpy as np

from . import study_course as course
from .blocks import code_line, code_lines, shape
from .study import Panel, Study

SIDE = course.SIDE
IN_CH, OUT_CH, KERNEL = 1, 4, 3           # nn.Conv2d(1, 4, 3)
POOL = 2                                  # nn.MaxPool2d(2)
P_DROP = course.DROPOUT                   # nn.Dropout(0.2)
CONV_SIDE = SIDE - KERNEL + 1             # 26: no padding
POOL_SIDE = CONV_SIDE // POOL             # 13
FILTER_SEED = 191
DIGIT = 0                                 # the first digit of the first batch
SHOWN_STEPS = (0, 1, 2)                   # the dropout switches drawn
GRID = 8                                  # 64 hidden numbers drawn as 8 rows of 8
SOURCE = ("MNIST, as examples/mnist_blocks.py loads it (ToTensor, Normalize((0.1307,), "
          "(0.3081,))); the first batch of voxanim.study_course (seeded shuffle)")


# ====================================================================== the arithmetic
def filters(seed=FILTER_SEED):
    """``(W [4, 1, 3, 3], b [4])`` as a fresh ``nn.Conv2d(1, 4, 3)`` starts them: weights
    and biases uniform in ``+-1/sqrt(1 * 3 * 3)`` (PyTorch's default), from a fixed seed."""
    rng = np.random.default_rng([course.SEED, int(seed)])
    bound = 1.0 / np.sqrt(IN_CH * KERNEL * KERNEL)
    W = rng.uniform(-bound, bound, (OUT_CH, IN_CH, KERNEL, KERNEL)).astype(np.float32)
    b = rng.uniform(-bound, bound, OUT_CH).astype(np.float32)
    return W, b


def conv2d(x, W, b):
    """``nn.Conv2d`` with no padding and stride 1 (cross-correlation, as PyTorch does it):
    ``x [B, C, H, W]`` -> ``[B, F, H - k + 1, W - k + 1]``."""
    x = np.asarray(x, np.float32)
    k = W.shape[-1]
    win = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(2, 3))  # B,C,h,w,k,k
    out = np.einsum("bchwij,fcij->bfhw", win, W, optimize=True)
    return (out + b[None, :, None, None]).astype(np.float32)


def maxpool2d(x, k=POOL):
    """``nn.MaxPool2d(k)``: each ``k x k`` square of every map becomes its biggest number."""
    x = np.asarray(x)
    B, C, H, W = x.shape
    h, w = H // k, W // k
    return x[:, :, :h * k, :w * k].reshape(B, C, h, k, w, k).max(axis=(3, 5))


def one_output(img, w, bias, r, c):
    """Output ``(r, c)`` of one filter on one picture, by hand: the 3 x 3 window of
    pixels, the 9 products, and their sum plus the bias."""
    k = w.shape[-1]
    patch = np.asarray(img, np.float32)[r:r + k, c:c + k]
    products = patch * w
    return patch, products, float(products.sum(dtype=np.float64) + float(bias))


def busy_window(img, size, step=1):
    """The ``size x size`` window (on a grid of ``step``) of ``img`` holding the most
    different values, then the widest spread: ``(row, col)``."""
    img = np.round(np.asarray(img, float), 4)
    best = None
    for r in range(0, img.shape[0] - size + 1, step):
        for c in range(0, img.shape[1] - size + 1, step):
            w = img[r:r + size, c:c + size]
            key = (len(np.unique(w)), float(w.max() - w.min()), -r, -c)
            if best is None or key > best:
                best = key
    return int(-best[2]), int(-best[3])


def _batch_maps():
    b = course.first_batch()
    W, bias = filters()
    maps = conv2d(b["x"], W, bias)
    return b, W, bias, maps


# ====================================================================== 191
def conv():
    b, W, bias, maps = _batch_maps()
    x = b["x"]
    img = x[DIGIT, 0]
    if maps.shape != (course.BATCH, OUT_CH, CONV_SIDE, CONV_SIDE):
        raise ValueError("conv(x) is not [64, 4, 26, 26]")
    r, c = busy_window(img, KERNEL)
    patch, products, total = one_output(img, W[0, 0], bias[0], r, c)
    if not np.isclose(total, maps[DIGIT, 0, r, c], atol=1e-5):
        raise ValueError("the hand-made output is not the convolution's")
    weights = OUT_CH * IN_CH * KERNEL * KERNEL
    places = CONV_SIDE * CONV_SIDE
    lines = list(code_lines("conv"))
    meta = dict(
        source=SOURCE, evidence="exact arithmetic on a real digit",
        filters="as a fresh nn.Conv2d(1, 4, 3) starts them: weights and biases uniform in "
                "±1/3 (PyTorch's default), drawn from a fixed seed (%d); not trained"
                % FILTER_SEED,
        label=int(b["y"][DIGIT]), index=int(b["indices"][DIGIT]),
        in_channels=IN_CH, out_channels=OUT_CH, kernel=KERNEL,
        weight=W.tolist(), bias=bias.tolist(), weight_shape=list(W.shape),
        at=[r, c], patch=patch.tolist(), products=products.tolist(),
        products_sum=float(products.sum(dtype=np.float64)), output=total,
        side_in=SIDE, side_out=CONV_SIDE, places=places,
        places_sum="26 × 26 = %d" % places, multiplies=places * KERNEL * KERNEL,
        shape_in=list(x.shape), shape_out=list(maps.shape),
        shape_in_text=shape(x.shape), shape_out_text=shape(maps.shape),
        weights=weights, biases=OUT_CH, learned=weights + OUT_CH,
        learned_sum="4 × 3 × 3 + 4 = %d" % (weights + OUT_CH),
        maps_low=float(maps[DIGIT].min()), maps_high=float(maps[DIGIT].max()),
        code=lines, make=code_line("conv", "nn.Conv2d"), run=code_line("conv", "conv(x)"),
        linear_learned=784 * 64 + 64,
        limitations=[
            "the course's classifier has no convolution: the 4 filters are untrained, as "
            "nn.Conv2d(1, 4, 3) starts them (uniform in ±1/3 from a fixed seed), so the "
            "sheets are rough copies of the digit, not trained edge detectors",
            "the input is the course's first batch (examples/blocks_parts.py runs the "
            "same lines on random numbers of the same shape)"])
    panels = (
        Panel("The digit", "the first digit of the first batch, 28 x 28 numbers", "images",
              dict(values=img[None].astype(float), labels=["label %d" % meta["label"]])),
        Panel("The 4 stamps", "the 4 filters of 3 x 3 weights, as nn.Conv2d(1, 4, 3) starts "
              "them (seeded)", "images", dict(values=W[:, 0].astype(float),
                                             labels=["filter %d" % i for i in range(OUT_CH)])),
        Panel("The 4 sheets", "conv(x) for this digit: 4 maps of 26 x 26", "images",
              dict(values=maps[DIGIT].astype(float),
                   labels=["sheet %d" % i for i in range(OUT_CH)])),
        Panel("Numbers learned", "4 x 3 x 3 weights and 4 biases, used at all 676 places",
              "bars", dict(values=np.array([weights, OUT_CH, weights + OUT_CH], float),
                           labels=["weights", "biases", "in all"])),
    )
    return Study("nn.Conv2d: one small stamp, used everywhere", "A 3 x 3 stamp slides over "
                 "the digit: the same 9 numbers at all 676 places, 4 stamps, 4 sheets",
                 panels, meta)


# ====================================================================== 192
def pool():
    b, _W, _bias, maps = _batch_maps()
    small = maxpool2d(maps)
    if small.shape != (course.BATCH, OUT_CH, POOL_SIDE, POOL_SIDE):
        raise ValueError("pool(maps) is not [64, 4, 13, 13]")
    sheet = maps[DIGIT, 0]
    # a direct loop over the squares, against the reshape
    loop = np.array([[sheet[2 * i:2 * i + 2, 2 * j:2 * j + 2].max() for j in range(POOL_SIDE)]
                     for i in range(POOL_SIDE)])
    if not np.array_equal(loop, small[DIGIT, 0]):
        raise ValueError("max-pooling is not the biggest of each square")
    r, c = busy_window(sheet, POOL, step=POOL)
    square = sheet[r:r + POOL, c:c + POOL]
    win = np.unravel_index(int(np.argmax(square)), square.shape)
    before = int(maps[DIGIT].size)
    after = int(small[DIGIT].size)
    lines = list(code_lines("pool"))
    meta = dict(
        source=SOURCE, evidence="exact arithmetic on a real digit",
        filters="the 4 sheets of blocks-conv2d: the untrained filters of a fresh "
                "nn.Conv2d(1, 4, 3) (seeded) on the first digit of the first batch",
        label=int(b["y"][DIGIT]), kernel=POOL, side_in=CONV_SIDE, side_out=POOL_SIDE,
        at=[r, c], square=square.tolist(), winner=[int(win[0]), int(win[1])],
        biggest=float(square.max()), at_out=[r // POOL, c // POOL],
        squares=POOL_SIDE * POOL_SIDE, squares_sum="13 × 13 = %d" % (POOL_SIDE ** 2),
        shape_in=list(maps.shape), shape_out=list(small.shape),
        shape_in_text=shape(maps.shape), shape_out_text=shape(small.shape),
        per_digit_in=before, per_digit_out=after,
        per_digit_in_sum="4 × 26 × 26 = %s" % format(before, ","),
        per_digit_out_sum="4 × 13 × 13 = %d" % after,
        weights=0, sheet_low=float(sheet.min()), sheet_high=float(sheet.max()),
        code=lines, make=code_line("pool", "nn.MaxPool2d"), run=code_line("pool", "pool(maps)"),
        limitations=["the sheets come from untrained filters (blocks-conv2d); pooling picks "
                     "the biggest of each square whatever the filters are"])
    panels = (
        Panel("One sheet", "conv(x)'s first sheet for the first digit: 26 x 26", "images",
              dict(values=sheet[None].astype(float), labels=["sheet 0"])),
        Panel("One square", "a 2 x 2 square of it; the biggest is kept", "matrix",
              dict(values=square.astype(float))),
        Panel("Pooled", "pool(maps) for this digit: 4 sheets of 13 x 13", "images",
              dict(values=small[DIGIT].astype(float),
                   labels=["sheet %d" % i for i in range(OUT_CH)])),
        Panel("Numbers per digit", "before and after nn.MaxPool2d(2); it learns none",
              "bars", dict(values=np.array([before, after, 0], float),
                           labels=["in", "out", "learned"])),
    )
    return Study("nn.MaxPool2d: keep the biggest", "Each 2 x 2 square becomes its biggest "
                 "number: 26 x 26 sheets become 13 x 13, and nothing is learned", panels, meta)


# ====================================================================== 193
def switches(step, row=DIGIT):
    """The dropout run's switches after the first ReLU at training step ``step``, for digit
    ``row`` of that step's batch: 64 numbers, 0 (off) or 1.25 (kept, scaled)."""
    return course._masks(step, course.BATCH)[0][row]


def dropout():
    b = course.first_batch()
    P = course.init_params()                     # every run starts from these
    masks = course._masks(0, course.BATCH)
    recorded = float(course.losses("dropout")[0])
    again = course.loss_of(P, b["x"], b["y"], masks=masks)
    if not np.isclose(again, recorded, rtol=0, atol=1e-6):
        raise ValueError("the switches do not reproduce the dropout run's first loss")
    h = course.forward(P, b["x"])["h1"][DIGIT]
    n = h.size
    steps = []
    for s in SHOWN_STEPS:
        m = switches(s)
        seen = h * m
        steps.append(dict(step=s, off=[int(i) for i in np.flatnonzero(m == 0)],
                          count=int((m == 0).sum()), seen=seen.tolist()))
    m0 = switches(0)
    kept = np.flatnonzero((m0 > 0) & (h > 0))
    ex = int(kept[np.argmax(h[kept])])
    scale = 1.0 / (1.0 - P_DROP)
    # every switch of the epoch's first dropout layer: the share switched off
    total_off = sum(int((course._masks(s, len(course.batch_indices(s)))[0] == 0).sum())
                    for s in range(course.EPOCH_STEPS))
    total = course.TRAIN * course.SIZES[1]
    held = course.heldout("dropout")
    plain = course.heldout("sgd")
    acc_eval = float(held["accuracy"][-1])
    acc_train = float(held["accuracy_train_mode"][-1])
    meta = dict(
        source=SOURCE, evidence="exact arithmetic on the dropout run's switches",
        run="study_course run 'dropout': the course's model with nn.Dropout(0.2) after "
            "each ReLU (region dropout-model), SGD lr 0.1, one epoch",
        label=int(b["y"][DIGIT]), p=P_DROP, numbers=int(n), grid=GRID,
        h=h.tolist(), relu_zeros=int((h == 0).sum()), steps=steps,
        scale=scale, scale_text="× %.2f" % scale, example=ex, example_in=float(h[ex]),
        example_out=float(h[ex] * m0[ex]),
        epoch_off=total_off, epoch_switches=total, epoch_share=total_off / total,
        first_loss=recorded, same_in_eval=True,
        accuracy_eval=acc_eval, accuracy_train_mode=acc_train,
        accuracy_no_dropout=float(plain["accuracy"][-1]),
        right_eval=int(round(acc_eval * course.TEST)),
        right_train_mode=int(round(acc_train * course.TEST)), test=course.TEST,
        code=list(code_lines("dropout")),
        switch=[code_line("train-mode", "model.train()"), code_line("evaluate", "model.eval()")],
        limitations=[
            "the 64 numbers are drawn as 8 rows of 8: they are one row of h, [64]",
            "the three sets of switches are the run's for the first digit of steps 0, 1 and "
            "2; at steps 1 and 2 the run fed other digits, so only the switches are "
            "theirs (drawn over the same 64 numbers)",
            "the dropout model's lines (region dropout-model, 9 lines) do not fit the "
            "8-line code panel; the block's own lines (region dropout) are shown"])
    panels = (
        Panel("64 hidden numbers", "h: the first digit's 64 numbers after the first ReLU "
              "(the untrained model, step 0), as 8 rows of 8", "matrix",
              dict(values=h.reshape(GRID, GRID).astype(float))),
        Panel("In training", "drop(h) at step 0: 1 in 5 switched off at random, the rest "
              "times 1.25", "matrix",
              dict(values=np.asarray(steps[0]["seen"]).reshape(GRID, GRID))),
        Panel("Switched off per step", "how many of the 64 the run switched off at steps "
              "0, 1, 2", "bars", dict(values=np.array([s["count"] for s in steps], float),
                                      labels=["step %d" % s for s in SHOWN_STEPS])),
        Panel("Held-out accuracy", "the dropout run on the 10,000 test digits, scored with "
              "model.eval() and left in training mode", "curve",
              dict(x=np.asarray(held["steps"], float),
                   ys=np.stack([held["accuracy"], held["accuracy_train_mode"]]).astype(float),
                   labels=["model.eval()", "training mode"])),
    )
    return Study("nn.Dropout: switched off at random", "In training nn.Dropout(0.2) switches "
                 "1 in 5 numbers off at random; in eval() it switches none", panels, meta)


STUDIES = {"blocks-conv2d": conv, "blocks-pooling": pool, "blocks-dropout": dropout}

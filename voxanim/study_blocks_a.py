"""The numbers behind animations 185-187 of the "Blocks and code" series (group ``a``): a
tensor, a batch and ``nn.Flatten``, all on the course's data as ``examples/mnist_blocks.py``
loads it (:mod:`voxanim.study_course`).

* ``blocks-a-tensor``: ``x, y = train_set[0]``, the first training digit after
  ``ToTensor`` and ``Normalize``: 784 numbers in a block of shape ``[1, 28, 28]``, label 5;
  a 4 x 4 corner of it read out number by number.
* ``blocks-a-batch``: ``x, y = next(iter(loader))``: 64 digits in one block of shape
  ``[64, 1, 28, 28]`` and their 64 labels, shape ``[64]``; what is in this shuffled batch.
* ``blocks-flatten``: ``nn.Flatten()`` unrolls each digit row by row into 784 numbers:
  pixel (row r, column c) lands at place ``28 r + c``; ``[64, 1, 28, 28] -> [64, 784]``,
  the same numbers in the same order, and no weights.

Every number shown is exact arithmetic on the data, checked here (the flattening against
numpy's own row-major reshape; the shapes against the arrays).
"""

from __future__ import annotations

import numpy as np

from . import study_course as course
from .blocks import code_line, code_lines, shape, torch_size
from .study import Panel, Study

SIDE = course.SIDE
DIGIT = 0                     # train_set[0]
ZOOM = 3                      # the corner read out number by number: ZOOM x ZOOM
ROW = 14                      # the pixel followed into the strip: the first brightest of this row
SOURCE = ("MNIST, as examples/mnist_blocks.py loads it: ToTensor, then Normalize((0.1307,), "
          "(0.3081,)); the batches in the order of voxanim.study_course (seeded)")


def zoom_window(img, size=ZOOM):
    """The ``size x size`` window of ``img`` holding the most different values (then the
    widest spread of them), so that its numbers show paper, ink and the shades between:
    ``(row, col)``."""
    win = np.lib.stride_tricks.sliding_window_view(np.round(np.asarray(img, float), 4),
                                                   (size, size))
    rows, cols = win.shape[:2]
    best = max((len(np.unique(win[r, c])), float(win[r, c].std()), -r, -c)
                for r in range(rows) for c in range(cols))
    return int(-best[2]), int(-best[3])


def flat_index(row, col, side=SIDE):
    """Where ``nn.Flatten`` puts pixel ``(row, col)`` of a ``side x side`` digit."""
    return int(row) * side + int(col)


# ====================================================================== 185
def a_tensor():
    x, y = course.digit(DIGIT)
    img = x[0]
    stats = course.pixel_stats()
    r0, c0 = zoom_window(img)
    patch = img[r0:r0 + ZOOM, c0:c0 + ZOOM]
    raw = course.pixels()[0][DIGIT].reshape(SIDE, SIDE)
    if x.shape != (1, SIDE, SIDE) or x.size != 784:
        raise ValueError("train_set[0] is not [1, 28, 28]")
    paper = float(stats["low"])
    inked = int((raw > 0).sum())
    meta = dict(
        source=SOURCE, evidence="exact arithmetic on the data", index=DIGIT, label=y,
        shape=list(x.shape), shape_text=shape(x.shape), printed="%s %d" % (torch_size(
            x.shape), y), numbers=int(x.size), product="1 × 28 × 28 = 784",
        low=float(img.min()), high=float(img.max()), paper=paper, inked=inked,
        blank=int(x.size - inked), zoom=[r0, c0], patch=patch.tolist(),
        mean=stats["mean"], std=stats["std"],
        code=list(code_lines("one-digit")), load=code_line("one-digit", "train_set[0]"),
        limitations=["the values are the normalised pixels train_set gives, not 0..255: "
                     "Normalize subtracts 0.1307 and divides by 0.3081, so the paper reads "
                     "%.4f and full ink %.4f" % (paper, float(stats["high"]))])
    panels = (
        Panel("train_set[0]", "the first training digit as 28 x 28 numbers, after ToTensor "
              "and Normalize", "images", dict(values=img[None].astype(float),
                                              labels=["label %d" % y])),
        Panel("A corner, number by number", "the %d x %d window at row %d, column %d"
              % (ZOOM, ZOOM, r0, c0), "matrix", dict(values=patch.astype(float))),
        Panel("Its shape", "x.shape: 1 channel, 28 rows, 28 columns", "bars",
              dict(values=np.array(x.shape, float), labels=["channels", "rows", "columns"])),
        Panel("Paper and ink", "tiles at the paper's value and tiles with ink", "bars",
              dict(values=np.array([x.size - inked, inked], float),
                   labels=["paper", "ink"])),
    )
    return Study("A tensor is a block of numbers", "One digit is 784 numbers in a block of "
                 "shape [1, 28, 28]", panels, meta)


# ====================================================================== 186
def a_batch():
    b = course.first_batch()
    xb, yb = b["x"], b["y"]
    if xb.shape != (course.BATCH, 1, SIDE, SIDE) or yb.shape != (course.BATCH,):
        raise ValueError("the first batch is not [64, 1, 28, 28] and [64]")
    counts = np.bincount(yb, minlength=course.CLASSES)
    one, _label = course.digit(int(b["indices"][0]))
    if not np.array_equal(one, xb[0]):
        raise ValueError("the batch's first digit is not the digit at its index")
    meta = dict(
        source=SOURCE, evidence="exact arithmetic on the data", batch=course.BATCH,
        indices=[int(i) for i in b["indices"]], labels=[int(v) for v in yb],
        shape=list(xb.shape), label_shape=list(yb.shape), shape_text=shape(xb.shape),
        label_shape_text=shape(yb.shape), printed="%s %s" % (torch_size(xb.shape),
                                                               torch_size(yb.shape)),
        one_shape=shape(one.shape), numbers=int(xb.size), product="64 × 784 = 50,176",
        counts=[int(c) for c in counts], most=int(np.argmax(counts)),
        most_count=int(counts.max()), fewest=int(np.argmin(counts)),
        fewest_count=int(counts.min()), train=course.TRAIN, steps=course.EPOCH_STEPS,
        last_batch=course.LAST_BATCH,
        loader=list(code_lines("loader")), take=list(code_lines("one-batch")),
        limitations=["the shuffled order is the numpy twin's (voxanim.study_course.order), "
                     "not PyTorch's: the run of the script draws other digits, of the same "
                     "shapes"])
    panels = (
        Panel("The first batch", "the 64 digits of next(iter(loader)), in order", "images",
              dict(values=b["pixels"].reshape(-1, SIDE, SIDE).astype(float),
                   labels=["%d" % v for v in yb])),
        Panel("Its labels", "y: the 64 labels, one per digit, in the batch's order", "bars",
              dict(values=yb.astype(float), labels=["%d" % i for i in range(len(yb))])),
        Panel("Digits in the batch", "how many of each digit this shuffled batch holds",
              "bars", dict(values=counts.astype(float), labels=[str(d) for d in range(10)])),
        Panel("Its shape", "x.shape: 64 digits, 1 channel, 28 rows, 28 columns", "bars",
              dict(values=np.array(xb.shape, float),
                   labels=["digits", "channels", "rows", "columns"])),
    )
    return Study("A batch: many digits in one block", "64 digits make one block of shape "
                 "[64, 1, 28, 28], their labels one of shape [64]", panels, meta)


# ====================================================================== 187
def flatten():
    x, y = course.digit(DIGIT)
    img = x[0]
    strip = course.forward(course.init_params(), x[None])["x"][0]     # nn.Flatten's output
    rows = np.concatenate([img[r] for r in range(SIDE)])
    if not np.array_equal(strip, rows) or not np.array_equal(strip, x.reshape(-1)):
        raise ValueError("flattening is not row after row")
    b = course.first_batch()
    flat = course.forward(course.init_params(), b["x"])["x"]
    if flat.shape != (course.BATCH, SIDE * SIDE):
        raise ValueError("the flattened batch is not [64, 784]")
    r, c = ROW, int(np.argmax(img[ROW]))
    at = flat_index(r, c)
    if strip[at] != img[r, c]:
        raise ValueError("pixel (%d, %d) is not at %d" % (r, c, at))
    model = list(code_lines("model"))
    flatten_line = next(i for i, s in enumerate(model) if "Flatten" in s)
    meta = dict(
        source=SOURCE, evidence="exact arithmetic: the same numbers, row after row",
        label=y, side=SIDE, length=int(strip.size), shape_in=[1, SIDE, SIDE],
        batch_in=list(b["x"].shape), batch_out=list(flat.shape),
        batch_in_text=shape(b["x"].shape), batch_out_text=shape(flat.shape),
        pixel=[r, c], place=at, place_sum="%d × 28 + %d = %d" % (r, c, at),
        value=float(img[r, c]), row_starts=[flat_index(k, 0) for k in range(SIDE)],
        weights=0, same_sum=bool(np.isclose(strip.sum(), img.sum())),
        model=model, flatten_line=flatten_line,
        limitations=["the strip is drawn 784 numbers wide in about 400 pixels, so "
                     "neighbouring numbers share a pixel there; the rows themselves are "
                     "shown tile by tile"])
    panels = (
        Panel("The square", "train_set[0]: 28 x 28 numbers", "images",
              dict(values=img[None].astype(float), labels=["label %d" % y])),
        Panel("The strip", "nn.Flatten(): the same 784 numbers, row 0, then row 1, ...",
              "matrix", dict(values=strip[None].astype(float))),
        Panel("The batch, flattened", "the first batch after nn.Flatten(): 64 rows of 784 "
              "numbers, one digit to a row", "matrix", dict(values=flat.astype(float))),
        Panel("Weights", "numbers nn.Flatten() learns: none", "bars",
              dict(values=np.array([0.0, 784 * 64 + 64]),
                   labels=["nn.Flatten()", "nn.Linear(784, 64)"])),
    )
    return Study("Flatten: a square becomes a strip", "nn.Flatten unrolls each 28 x 28 "
                 "digit row by row into 784 numbers, and learns nothing", panels, meta)


STUDIES = {"blocks-a-tensor": a_tensor, "blocks-a-batch": a_batch, "blocks-flatten": flatten}

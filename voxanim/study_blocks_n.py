"""The numbers behind animations 224-226, the first three of "Blocks and code: padding" (group
``n``): why a convolution's maps shrink, a ring of padding, and "same" against "valid", from
the lines of ``examples/padding.py`` (regions ``pad-none``, ``pad-one``, ``pad-same``) and
:mod:`voxanim.study_padding`.

* ``padding-maps-shrink``: ``nn.Conv2d(1, 8, 3)`` on the course's first training digit
  (``x[0]``, a 5): a 3 x 3 stamp fits 26 places across 28 pixels and 26 down, 676 places,
  so the maps are 26 x 26; and the coverage map, how many of those places read each pixel:
  a corner pixel 1, an edge pixel 3, a middle one 9.
* ``padding-a-ring``: ``padding=1`` puts a ring of zeros round the digit, 30 x 30; the stamp
  fits 28 places across, so the maps keep 28 x 28; the corner is now read 4 times and the
  edge 6, the middle still 9 (padding does not make the border read as often as the
  middle). After Normalize the paper is -0.42 (:data:`voxanim.study_padding.BG`), so the
  ring of zeros is a little ink, not paper.
* ``padding-same-and-valid``: for stamps of 3, 5 and 7, ``padding="same"`` pads
  (k - 1) / 2 = 1, 2, 3 and keeps 28; "valid", the default, loses k - 1: 26, 24, 22.

Every number is exact arithmetic: places and coverage are counted here place by place and
checked against :func:`voxanim.study_padding.coverage`; the stamp's settings are read from
the script's own lines. The three pixels called out are the same in 224 and 225: a corner,
an edge pixel and a middle pixel of the real 28 x 28.
"""

from __future__ import annotations

import re

import numpy as np

from . import study_course as course
from . import study_padding as pad
from .blocks import code_line, code_lines, shape, torch_size, val
from .study import Panel, Study

DIGIT = 0                      # x[0] of the script, the course's first training digit (a 5)
SIDE = pad.SIDE                # 28
BATCH = course.BATCH           # the script's x: the first 64 digits
# the three pixels called out (row, column of the real 28 x 28): a corner, a pixel of the
# right edge, and one in the middle
CORNER, EDGE, MIDDLE = (0, 27), (24, 27), (14, 14)
SPOTS = (("corner", CORNER), ("edge", EDGE), ("middle", MIDDLE))
SOURCE = ("examples/padding.py (regions pad-none, pad-one, pad-same) and "
          "voxanim.study_padding, on the course's first MNIST training digit")
_CONV = re.compile(r"nn\.Conv2d\((\d+), (\d+), (\w+)(?:,\s*padding=([^)]+))?\)")


# ====================================================================== counting
def conv_settings(region):
    """``(in, out, k, padding)`` of the region's first ``nn.Conv2d(...)`` (which may run
    over two lines; ``k`` may be the loop's ``k``; padding 0 when not given, else as
    written)."""
    text = " ".join(s.strip() for s in code_lines(region))
    m = _CONV.search(text)
    if m is None:
        raise ValueError("no nn.Conv2d(...) in region %r" % region)
    cin, cout, k, padding = m.groups()
    k = int(k) if k.isdigit() else k
    if padding is None:
        padding = 0
    elif padding.isdigit():
        padding = int(padding)
    else:
        padding = padding.strip("\"'")
    return int(cin), int(cout), k, padding


def places(side=SIDE, k=3, padding=0):
    """Every place of a ``k x k`` stamp on ``side x side`` with ``padding`` rings, in reading
    order, as the top-left ``(row, col)`` of the stamp in the padded picture."""
    n = pad.out_side(side, k, padding)
    return [(r, c) for r in range(n) for c in range(n)]


def covering(pixel, side=SIDE, k=3, padding=0):
    """The places (as :func:`places` gives them) whose stamp reads the real pixel
    ``(row, col)``, in reading order."""
    r, c = pixel[0] + padding, pixel[1] + padding
    return [(pr, pc) for pr, pc in places(side, k, padding)
            if pr <= r < pr + k and pc <= c < pc + k]


def coverage_after(n, side=SIDE, k=3, padding=0):
    """How many of the first ``n`` places (reading order) read each real pixel: the coverage
    map as it fills while the stamp sweeps."""
    total = side + 2 * padding
    count = np.zeros((total, total), int)
    for pr, pc in places(side, k, padding)[:int(n)]:
        count[pr:pr + k, pc:pc + k] += 1
    return count[padding:padding + side, padding:padding + side]


def read_counts(cov):
    """``{times: pixels}``: how many pixels are read each number of times."""
    times, pixels = np.unique(np.asarray(cov), return_counts=True)
    return {int(t): int(p) for t, p in zip(times, pixels, strict=True)}


def _coverage_meta(side, k, padding):
    cov = coverage_after(len(places(side, k, padding)), side, k, padding)
    if not np.array_equal(cov, pad.coverage(side, k, padding)):
        raise ValueError("the places counted one by one are not study_padding.coverage")
    spots = {}
    for name, px in SPOTS:
        where = covering(px, side, k, padding)
        if len(where) != cov[px]:
            raise ValueError("the %s pixel's places are not its coverage" % name)
        spots[name] = dict(pixel=list(px), times=int(cov[px]), places=[list(p) for p in where])
    return cov, spots


def _shapes(region, side_out):
    cin, cout, _k, _padding = conv_settings(region)
    before = [BATCH, cin, SIDE, SIDE]
    after = [BATCH, cout, side_out, side_out]
    return before, after


# ====================================================================== 224
def maps_shrink():
    cin, cout, k, padding = conv_settings("pad-none")
    if (cin, k, padding) != (1, 3, 0):
        raise ValueError("pad-none is not a 3 x 3 stamp without padding")
    x, label = course.digit(DIGIT)
    img = np.asarray(x[0], float)
    across = pad.out_side(SIDE, k, padding)
    every = places(SIDE, k, padding)
    if len(every) != across * across:
        raise ValueError("the places are not %d x %d" % (across, across))
    cov, spots = _coverage_meta(SIDE, k, padding)
    if int(cov.sum()) != len(every) * k * k:
        raise ValueError("every place reads 9 pixels")
    counts = read_counts(cov)
    before, after = _shapes("pad-none", across)
    meta = dict(
        source=SOURCE, evidence="exact arithmetic: the places of a 3 x 3 stamp, counted",
        digit=DIGIT, label=int(label), side=SIDE, k=k, padding=padding, channels=cout,
        across=across, places=len(every),
        across_sum="%d − %d = %d" % (SIDE, k - 1, across),
        places_sum="%d × %d = %d" % (across, across, len(every)),
        spots=spots, corner=spots["corner"]["times"], edge=spots["edge"]["times"],
        middle=spots["middle"]["times"], read_counts=counts,
        fewer=int(sum(p for t, p in counts.items() if t < k * k)),
        nine=counts.get(k * k, 0), most=int(cov.max()),
        shape_in=before, shape_out=after,
        shape_change="%s → %s" % (shape(before), shape(after)),
        printed=torch_size(after), code=list(code_lines("pad-none")),
        conv_line=code_line("pad-none", "nn.Conv2d("),
        limitations=["the coverage map counts the places of one 3 x 3 stamp; every one of "
                     "the 8 stamps of nn.Conv2d(1, 8, 3) sits at the same places",
                     "the shapes are those the script's comments give, checked against torch "
                     "by tests/test_padding_equivalence.py"])
    panels = (
        Panel("The digit", "x[0], normalised: 28 x 28 pixels, the paper at -0.42", "matrix",
              dict(values=img)),
        Panel("How often each pixel is read", "of the 676 places of a 3 x 3 stamp without "
              "padding, how many read each pixel", "matrix", dict(values=cov.astype(float))),
        Panel("Pixels by times read", "how many of the 784 pixels are read 1, 2, 3, 4, 6 or "
              "9 times", "bars", dict(values=np.array(list(counts.values()), float),
                                      labels=[str(t) for t in counts])),
        Panel("Places across", "28 pixels across; a 3 x 3 stamp fits 26 places", "bars",
              dict(values=np.array([SIDE, across], float), labels=["pixels", "places"])),
    )
    return Study("Why maps shrink", "a 3 x 3 stamp fits 26 places across 28 pixels, and "
                 "reads a corner pixel once, an edge pixel 3 times and a middle one 9",
                 panels, meta)


# ====================================================================== 225
def a_ring():
    cin, cout, k, padding = conv_settings("pad-one")
    if (cin, k, padding) != (1, 3, 1):
        raise ValueError("pad-one is not a 3 x 3 stamp with padding=1")
    x, label = course.digit(DIGIT)
    img = np.asarray(x[0], float)
    padded = np.pad(img, padding)                     # nn.Conv2d's padding: zeros
    big = SIDE + 2 * padding
    ring = np.ones((big, big), bool)
    ring[padding:-padding, padding:-padding] = False
    if not (padded[ring] == 0).all():
        raise ValueError("the ring is not 0")
    bg = pad.BG
    if not np.isclose(img.min(), bg, atol=1e-6):
        raise ValueError("the digit's blank paper is not Normalize's blank, %.4f" % bg)
    paper = img == img.min()
    across = pad.out_side(SIDE, k, padding)
    if across != SIDE:
        raise ValueError("padding=1 does not keep 28")
    every = places(SIDE, k, padding)
    cov, spots = _coverage_meta(SIDE, k, padding)
    counts = read_counts(cov)
    corner3 = padded[:k, -k:]                   # the top-right 3 x 3: the last place across
    before, after = _shapes("pad-one", across)
    meta = dict(
        source=SOURCE, evidence="exact arithmetic: padding=1 on a real digit",
        digit=DIGIT, label=int(label), side=SIDE, big=big, k=k, padding=padding,
        channels=cout, ring_value=0.0, paper_value=float(bg), ring_cells=int(ring.sum()),
        paper_share=float(paper.mean()),
        # where 0 sits on the digit's shading, paper (-0.42) to full ink (2.82)
        ring_shade=float((0.0 - bg) / ((1.0 - course.MEAN) / course.STD - bg)),
        bg_sum="(0 − %.4f) ÷ %.4f = %s" % (course.MEAN, course.STD, val(bg)),
        big_sum="%d + %d + %d = %d" % (padding, SIDE, padding, big),
        across=across, places=len(every),
        across_sum="%d − %d = %d" % (big, k - 1, across),
        places_sum="%d × %d = %d" % (across, across, len(every)),
        corner3=corner3.tolist(),
        spots=spots, corner=spots["corner"]["times"], edge=spots["edge"]["times"],
        middle=spots["middle"]["times"], read_counts=counts,
        fewer=int(sum(p for t, p in counts.items() if t < k * k)),
        shape_in=before, shape_out=after,
        shape_change="%s → %s" % (shape(before), shape(after)),
        printed=torch_size(after), code=list(code_lines("pad-one")),
        conv_line=code_line("pad-one", "nn.Conv2d("),
        limitations=["padding=1 pads with 0, which after Normalize is not the paper (-0.42): "
                     "the ring is a faint frame round every digit",
                     "the corner is read 4 times and the edge 6: padding keeps the size, it "
                     "does not make the border read as often as the middle (9)"])
    panels = (
        Panel("The padded digit", "x[0] with padding=1: a ring of zeros, 30 x 30", "matrix",
              dict(values=padded)),
        Panel("How often each pixel is read", "of the 784 places of a 3 x 3 stamp with "
              "padding=1, how many read each real pixel", "matrix",
              dict(values=cov.astype(float))),
        Panel("Pixels by times read", "how many of the 784 pixels are read 4, 6 or 9 times",
              "bars", dict(values=np.array(list(counts.values()), float),
                           labels=[str(t) for t in counts])),
        Panel("The ring and the paper", "the ring's value, 0, against a blank pixel after "
              "Normalize, -0.42", "bars", dict(values=np.array([0.0, bg]),
                                               labels=["ring", "paper"])),
    )
    return Study("A ring of padding", "padding=1: a ring of zeros makes the digit 30 x 30, "
                 "so a 3 x 3 stamp fits 28 places and the maps keep their size", panels, meta)


# ====================================================================== 226
def same_and_valid():
    line = code_lines("pad-same")[0]
    m = re.search(r"for k in \(([\d, ]+)\)", line)
    kernels = tuple(int(v) for v in m.group(1).split(","))
    if kernels != pad.KERNELS:
        raise ValueError("the script's kernels are not study_padding's")
    if conv_settings("pad-same")[3] != "same":
        raise ValueError("pad-same's first conv is not padding=\"same\"")
    cin, cout = conv_settings("pad-same")[:2]
    rows = pad.same_and_valid(kernels, SIDE)
    for r in rows:
        if r["pad"] * 2 != r["k"] - 1 or pad.out_side(SIDE, r["k"], r["pad"]) != SIDE:
            raise ValueError("same does not keep 28 for k = %d" % r["k"])
        if r["valid"] != SIDE - (r["k"] - 1):
            raise ValueError("valid does not lose k - 1 for k = %d" % r["k"])
        r["pad_sum"] = "(%d − 1) ÷ 2 = %d" % (r["k"], r["pad"])
        r["valid_sum"] = "%d − %d = %d" % (SIDE, r["k"] - 1, r["valid"])
        r["printed"] = ["%d %s" % (r["k"], torch_size([BATCH, cout, side, side]))
                        for side in (r["same"], r["valid"])]
    meta = dict(
        source=SOURCE, evidence="exact arithmetic: the sides, (k − 1) ÷ 2 and k − 1",
        side=SIDE, kernels=list(kernels), channels=cout, in_channels=cin, rows=rows,
        code=list(code_lines("pad-same")),
        limitations=["\"same\" pads (k - 1) / 2 on every side for the odd stamps shown; "
                     "for an even k PyTorch pads one side one more than the other",
                     "the printed lines are those the script's region prints, checked against "
                     "torch by tests/test_padding_equivalence.py"])
    ks = [str(r["k"]) for r in rows]
    panels = (
        Panel("Padding for same", "(k - 1) / 2 on every side", "bars",
              dict(values=np.array([r["pad"] for r in rows], float), labels=ks)),
        Panel("Side with same", "the maps keep 28 x 28 for every k", "bars",
              dict(values=np.array([r["same"] for r in rows], float), labels=ks)),
        Panel("Side with valid", "no padding: 28 - (k - 1)", "bars",
              dict(values=np.array([r["valid"] for r in rows], float), labels=ks)),
        Panel("The table", "k, padding, same, valid", "matrix",
              dict(values=np.array([[r["k"], r["pad"], r["same"], r["valid"]] for r in rows],
                                   float))),
    )
    return Study("Same and valid", "padding=\"same\" pads (k - 1) / 2 and keeps 28; "
                 "\"valid\", the default, loses k - 1", panels, meta)


STUDIES = {"padding-maps-shrink": maps_shrink, "padding-a-ring": a_ring,
           "padding-same-and-valid": same_and_valid}

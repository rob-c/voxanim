"""The numbers behind animations 227-229 of "Blocks and code: padding" (group ``o``), from
:mod:`voxanim.study_padding` (the lines of ``examples/padding.py`` in numpy) and the course's
CNN without padding (:mod:`voxanim.study_course_cnn`).

* ``padding-modes``: the 8 x 8 piece of training digit 0 that region ``pad-modes`` pads
  (:func:`voxanim.study_padding.part`) and ``F.pad(part, [2] * 4, mode=m)`` for each of the
  four modes (:func:`voxanim.study_padding.modes`). For the three modes that copy, where each
  border number is copied from (:func:`source`, checked here against every border number);
  for "constant", 0 against the paper, which after Normalize is -0.42.
* ``padding-deep-stacks``: twelve 3 x 3 stamps take 28 down to 4 without padding and keep 28
  with ``padding=1`` (:func:`voxanim.study_padding.deep`); the padded CNN of region
  ``pad-model`` walks 28 -> 28 -> 14 -> 14 -> 7, so its last layer reads 16 x 7 x 7 = 784
  numbers, against the CNN without padding's 400. The squares show one real map of each
  stage of the padded CNN after its epoch, for training digit 0.
* ``padding-in-the-cnn``: the two CNNs after one epoch on the same batches, one run each
  (:func:`voxanim.study_padding.compare`): 5,258 numbers against 9,098 (the stamps are the
  same; all 3,840 more are in the last layer, which reads 784 numbers, not 400), 96.4%
  against 97.6% held out, 364 digits missed against 237; and on test digits moved 1 to 6
  pixels towards an edge (:func:`voxanim.study_padding.shifted`, the mean over the four
  edges) both fall alike, to 18.5% and 16.9% at 6 pixels. So the better accuracy is not
  padding's alone, and padding does not help with moved digits.
"""

from __future__ import annotations

import numpy as np

from . import study_course as course
from . import study_course_cnn as cnn
from . import study_padding as pad
from .blocks import code_line, code_lines, commas, shape, torch_size, val
from .study import Panel, Study
from .study_blocks_j import pct

DIGIT = 0                     # x[0] of the script: training digit 0, a 5
ROW = 6                       # the row of the 12 x 12 whose two left border numbers are traced
TRACED = (1, 0)               # those two border columns, the inner one first
PAPER_TILE = (7, 2)           # a tile of the piece that is blank paper (-0.42)
RING_TILE = (ROW, 0)          # a tile of "constant"'s ring (0)
MOVED = (0, 2, 4, 6)          # the moved digit drawn, pixels to the right
MOVED_DIGIT = 0               # test digit 0, a 7
SOURCE = ("MNIST; examples/padding.py's lines in numpy (voxanim.study_padding), and the "
          "course's CNN without padding (voxanim.study_course_cnn)")
ONE_RUN = ("one run of each CNN from one seed: another seed would move these numbers a little, "
           "by an amount not measured here")


def number(v):
    """A tile's value as a label: ``0`` for zero, else :func:`~voxanim.blocks.val`."""
    return "0" if float(v) == 0.0 else val(v)


# ====================================================================== 227
def source(mode, i, n=8, width=pad.PART_PAD):
    """Which row (or column) of the ``n``-long piece index ``i`` of the padded block copies,
    for ``F.pad``'s ``mode``: reflect mirrors about the edge (the edge itself not repeated),
    replicate repeats the edge, circular wraps round to the other side. None: "constant"'s
    border, which copies nothing (it is 0)."""
    j = i - width
    if 0 <= j < n:
        return j
    if mode == "constant":
        return None
    if mode == "reflect":
        return -j if j < 0 else 2 * (n - 1) - j
    if mode == "replicate":
        return min(max(j, 0), n - 1)
    if mode == "circular":
        return j % n
    raise ValueError("no mode %r" % mode)


def readout(block, mode, row=ROW, width=pad.PART_PAD):
    """Row ``row`` of a padded block as a short line: its left border, an edge mark, then
    the piece's first numbers (for "circular", also the piece's last two, which the left
    border copies)."""
    r = [number(v) for v in block[row]]
    left = "  ".join(r[:width])
    if mode == "circular":
        right = "  ".join(r[width:width + 2]) + "  ...  " + "  ".join(r[-2 * width:-width])
    else:
        right = "  ".join(r[width:width + 3])
    return left + " | " + right


def modes_study():
    part = pad.part()
    blocks = pad.modes()
    n, w = part.shape[0], pad.PART_PAD
    side = n + 2 * w
    for m, b in blocks.items():                  # every number is where `source` says it is
        if b.shape != (side, side):
            raise ValueError("%s: not %d x %d" % (m, side, side))
        for r in range(side):
            for c in range(side):
                sr, sc = source(m, r), source(m, c)
                want = 0.0 if sr is None or sc is None else part[sr, sc]
                if b[r, c] != want:
                    raise ValueError("%s: (%d, %d) is not where source() says" % (m, r, c))
    if part[PAPER_TILE[0] - w, PAPER_TILE[1] - w] != np.float32(pad.BG):
        raise ValueError("the paper tile is not blank paper")
    arrows = {m: [[[ROW, source(m, c) + w], [ROW, c]] for c in TRACED]
              for m in pad.MODES if m != "constant"}
    lines = list(code_lines("pad-modes"))
    printed = {m: "%s %s" % (m, torch_size((1, 1, side, side))) for m in pad.MODES}
    rows, cols = pad.PART
    img, label = course.digit(DIGIT)
    meta = dict(
        source=SOURCE, evidence="exact arithmetic: F.pad's four modes on a real digit",
        digit=DIGIT, label=int(label), image=np.asarray(img[0], float).tolist(),
        box=[rows.start, rows.stop - 1, cols.start, cols.stop - 1],
        box_text="rows %d–%d, columns %d–%d" % (rows.start, rows.stop - 1, cols.start,
                                                cols.stop - 1),
        part=part.tolist(), side=n, width=w, padded=side, modes=list(pad.MODES),
        blocks={m: b.tolist() for m, b in blocks.items()},
        part_shape=[1, 1, n, n], part_text=shape((1, 1, n, n)),
        padded_shape=[1, 1, side, side], padded_text=shape((1, 1, side, side)),
        printed=printed, row=ROW, traced=list(TRACED), arrows=arrows,
        readout={m: readout(blocks[m], m) for m in pad.MODES},
        copied={m: [number(blocks[m][ROW, c]) for c in TRACED] for m in arrows},
        bg=pad.BG, bg_text=val(pad.BG), zero_text="0", ring_tile=list(RING_TILE),
        paper_tile=list(PAPER_TILE), code=lines,
        limitations=["the piece is one 8 x 8 corner of one digit; F.pad does the same to any "
                     "block of numbers",
                     "the arrows trace two border numbers of one row; source() is checked "
                     "against all of them"])
    panels = (
        Panel("The piece", "x[:1, :, 5:13, 8:16]: 8 x 8 of training digit 0, normalised",
              "images", dict(values=part[None].astype(float), labels=["part"])),
        Panel("The four modes", "F.pad(part, [2] * 4, mode=m): 12 x 12 each", "images",
              dict(values=np.stack([blocks[m] for m in pad.MODES]).astype(float),
                   labels=list(pad.MODES))),
        Panel("One row of each", "row %d of each padded block (the piece is columns 2-9)" % ROW,
              "curve", dict(x=np.arange(side, dtype=float),
                            ys=np.stack([blocks[m][ROW] for m in pad.MODES]).astype(float),
                            labels=list(pad.MODES))),
        Panel("Zero is not the paper", "what \"constant\" pads with, and a blank pixel after "
              "Normalize", "bars", dict(values=np.array([0.0, pad.BG]),
                                        labels=["constant's border", "the paper"])),
    )
    return Study("What goes in the border", "F.pad's four modes on an 8 x 8 piece of a real "
                 "digit", panels, meta)


# ====================================================================== 228
def walk_maps(a):
    """One map of each stage of the padded CNN's activations ``a`` (a batch of one): the
    digit, the first convolution's busiest map before and after pooling, then the second's:
    sides 28, 28, 14, 14, 7. Returns the maps and the two maps' indices."""
    c1 = int(a["h1"][0].reshape(len(a["h1"][0]), -1).sum(1).argmax())
    c2 = int(a["h2"][0].reshape(len(a["h2"][0]), -1).sum(1).argmax())
    maps = [a["x"][0, 0], a["h1"][0, c1], a["p1"][0, c1], a["h2"][0, c2], a["p2"][0, c2]]
    return [np.asarray(m, float) for m in maps], c1, c2


def deep_study():
    d = pad.deep()
    walk = list(pad.pad_side_walk())
    img, label = course.digit(DIGIT)
    P = pad.params("end")
    a = pad.forward(P, img[None])
    maps, c1, c2 = walk_maps(a)
    if [m.shape[0] for m in maps] != walk:
        raise ValueError("the maps' sides are not the side walk")
    batch = course.first_batch()["x"]
    b = pad.forward(P, batch)
    shapes = [list(b[k].shape) for k in ("x", "z1", "p1", "z2", "p2", "flat", "logits")]
    channels = pad.PAD_CHANNELS[-1]
    side = walk[-1]
    in_features = channels * side * side
    if in_features != pad.PAD_IN_FEATURES or P["7.weight"].shape[1] != in_features:
        raise ValueError("16 x 7 x 7 is not the padded Linear's in_features")
    plain_side = cnn.side_walk()[-1]
    plain_flat = np.asarray(cnn.activations(cnn.EPOCH_STEPS, DIGIT)["p2"][0], float).ravel()
    if plain_flat.size != cnn.IN_FEATURES:
        raise ValueError("the CNN without padding does not hand its Linear 400 numbers")
    steps = []
    for n in range(len(walk) - 1):
        before, after = walk[n], walk[n + 1]
        if n % 2 == 0:
            steps.append(dict(kind="conv", before=before, after=after, op="pad 1",
                              sum="%d + 2 − 2 = %d" % (before, after),
                              why="a ring adds 2, the stamp loses 2"))
        else:
            steps.append(dict(kind="pool", before=before, after=after, op="÷2",
                              sum="%d ÷ 2 = %d" % (before, after),
                              why="2 × 2 squares become 1"))
    if any(s["after"] != (s["before"] if s["kind"] == "conv" else s["before"] // 2)
           for s in steps):
        raise ValueError("the steps are not the side walk")
    linear = code_line("pad-model", "nn.Linear(")
    if "16 * 7 * 7" not in linear:
        raise ValueError("the padded model's Linear is not 16 * 7 * 7")
    meta = dict(
        source=SOURCE, evidence="exact arithmetic: the sides of the maps",
        layers=pad.DEEP_LAYERS, valid=d["valid"], same=d["same"],
        printed_deep=pad.PRINTED_DEEP, printed_one="padding=1: %s"
        % torch_size((course.BATCH, 8, course.SIDE, course.SIDE)),
        digit=DIGIT, label=int(label), maps_used=[c1, c2], side_walk=walk, steps=steps,
        walk_text=" → ".join(str(v) for v in walk),
        maps=channels, side=side, in_features=in_features,
        in_features_sum="%d × %d × %d = %d" % (channels, side, side, in_features),
        plain_walk=list(cnn.side_walk()), plain_in_features=cnn.IN_FEATURES,
        plain_sum="%d × %d × %d = %d" % (channels, plain_side, plain_side, cnn.IN_FEATURES),
        plain_flat=plain_flat.tolist(),
        shapes=shapes, shape_texts=[shape(s) for s in shapes],
        transitions=["%s → %s" % (shape(shapes[i]), shape(shapes[i + 1]))
                     for i in range(len(shapes) - 1)],
        linear_code=linear, code_deep=list(code_lines("pad-deep")),
        code_one=list(code_lines("pad-one")), code_model=list(code_lines("pad-model")),
        limitations=["the twelve stamps of region pad-deep are untrained; only their sides "
                     "are shown, as squares to scale",
                     "the walk's squares show one map at each stage (the busiest of the "
                     "digit's) of the padded CNN after its epoch; there are 8, then 16",
                     "the strip of 400 is the same digit's numbers into the last layer of the "
                     "CNN without padding, after its epoch"])
    panels = (
        Panel("Twelve stamps", "the side after each of 12 layers of 3 x 3 stamps, without and "
              "with padding=1", "curve",
              dict(x=np.arange(pad.DEEP_LAYERS + 1, dtype=float),
                   ys=np.array([d["valid"], d["same"]], float),
                   labels=["no padding", "padding=1"])),
        Panel("The padded CNN's sides", "the side of the maps after each block that changes "
              "it", "bars", dict(values=np.array(walk, float),
                                 labels=["input", "Conv2d", "MaxPool2d", "Conv2d",
                                         "MaxPool2d"])),
        Panel("One map at each stage", "training digit 0 and one of its maps after each block, "
              "through the padded CNN after its epoch", "matrix",
              dict(values=maps[0], conv1=maps[1], pool1=maps[2], conv2=maps[3],
                   pool2=maps[4])),
        Panel("16 maps of 7 x 7", "what nn.Flatten() unrolls into 784 numbers", "images",
              dict(values=np.asarray(a["p2"][0], float),
                   labels=[str(k) for k in range(channels)])),
    )
    return Study("Deep stacks need padding", "Twelve 3 x 3 stamps: 28 -> 4 without padding, "
                 "28 -> 28 with it; the padded CNN's last layer reads 784", panels, meta)


# ====================================================================== 229
def per_digit_missed(predictions):
    yt = course.data()[3]
    p = np.asarray(predictions)
    return [int(((yt == d) & (p != yt)).sum()) for d in range(course.CLASSES)]


def in_the_cnn():
    c = pad.compare()
    sh = pad.shifted()
    plain_counts = [v["total"] for v in cnn.counts().values()]
    P = pad.init_params()
    padded_counts = [int(P["%d.weight" % i].size + P["%d.bias" % i].size) for i in (0, 3, 7)]
    if sum(plain_counts) != c["plain_params"] or sum(padded_counts) != c["padded_params"]:
        raise ValueError("the per-block counts do not add up")
    if plain_counts[:2] != padded_counts[:2]:
        raise ValueError("padding changed the stamps' counts")
    if not np.array_equal(c["steps"], cnn.heldout()["steps"]):
        raise ValueError("the two held-out curves are not on the same steps")
    yt = course.data()[3]
    pred = pad.run()["predictions"]
    per_plain = per_digit_missed(cnn.predictions())
    per_padded = per_digit_missed(pred)
    if sum(per_plain) != c["plain_missed"] or sum(per_padded) != c["padded_missed"]:
        raise ValueError("the per-digit misses do not add up")
    xt = course.data()[2]
    moved = np.stack([pad.shift(xt[MOVED_DIGIT:MOVED_DIGIT + 1], s, "right")[0]
                      .reshape(course.SIDE, course.SIDE) for s in MOVED])
    if np.abs(moved[0] - xt[MOVED_DIGIT].reshape(course.SIDE, course.SIDE)).max() != 0:
        raise ValueError("a digit moved 0 pixels is not the digit")
    pm, qm = np.asarray(sh["plain_mean"]), np.asarray(sh["padded_mean"])
    if abs(pm[0] - c["plain_accuracy"]) > 1e-9 or abs(qm[0] - c["padded_accuracy"]) > 1e-9:
        raise ValueError("unmoved digits do not give the held-out accuracy")
    right_plain = course.TEST - c["plain_missed"]
    right_padded = course.TEST - c["padded_missed"]
    meta = dict(
        source=SOURCE, evidence="exact: the numbers counted; measured: 10,000 held-out digits",
        runs="one run each",
        plain_params=c["plain_params"], padded_params=c["padded_params"],
        plain_params_text=commas(c["plain_params"]),
        padded_params_text=commas(c["padded_params"]),
        plain_counts=plain_counts, padded_counts=padded_counts,
        plain_counts_text=[commas(v) for v in plain_counts],
        padded_counts_text=[commas(v) for v in padded_counts],
        extra=c["padded_params"] - c["plain_params"],
        extra_text=commas(c["padded_params"] - c["plain_params"]),
        plain_in=cnn.IN_FEATURES, padded_in=pad.PAD_IN_FEATURES,
        plain_accuracy=c["plain_accuracy"], padded_accuracy=c["padded_accuracy"],
        plain_accuracy_text=pct(c["plain_accuracy"]),
        padded_accuracy_text=pct(c["padded_accuracy"]),
        plain_missed=c["plain_missed"], padded_missed=c["padded_missed"],
        printed_plain=repr(right_plain / course.TEST),
        printed_padded=repr(right_padded / course.TEST),
        per_digit_plain=per_plain, per_digit_padded=per_padded,
        per_digit_total=[int((yt == d).sum()) for d in range(course.CLASSES)],
        eval_steps=[int(s) for s in c["steps"]],
        plain_curve=[float(v) for v in c["plain_curve"]],
        padded_curve=[float(v) for v in c["padded_curve"]],
        shifts=list(sh["shifts"]), directions=list(sh["directions"]),
        plain_shifted=[float(v) for v in pm], padded_shifted=[float(v) for v in qm],
        plain_shifted_text=[pct(v) for v in pm], padded_shifted_text=[pct(v) for v in qm],
        plain_by_direction=np.asarray(sh["plain"]).tolist(),
        padded_by_direction=np.asarray(sh["padded"]).tolist(),
        moved=list(MOVED), moved_digit=MOVED_DIGIT, moved_label=int(yt[MOVED_DIGIT]),
        moved_images=moved.tolist(), bg=pad.BG,
        steps=course.EPOCH_STEPS, test=course.TEST,
        linear_code=code_line("pad-model", "nn.Linear("),
        code_model=list(code_lines("pad-model")),
        limitations=[
            ONE_RUN,
            "the padded CNN's last layer reads 784 numbers, not 400, so it has 3,840 more "
            "numbers to learn: its better accuracy is not padding's alone",
            "moved digits: the 10,000 test digits moved 1 to 6 pixels towards each of the "
            "four edges, the space left filled with the paper; the accuracy shown is the mean "
            "over the four edges",
            "both trained one epoch with SGD at lr 0.1 on the same batches in the same order; "
            "examples/padding.py has no lines for the training or the moving: the training is "
            "examples/mnist_cnn.py's loop with the model swapped"])
    panels = (
        Panel("Numbers to learn", "each block that learns, without and with padding=1",
              "matrix", dict(values=np.array([plain_counts, padded_counts], float),
                             rows=["no padding", "padding=1"],
                             cols=["Conv2d(1, 8)", "Conv2d(8, 16)", "Linear"])),
        Panel("Held-out accuracy while it trains", "share of the 10,000 test digits named "
              "right, every 25 steps of one run each", "curve",
              dict(x=np.asarray(c["steps"], float),
                   ys=np.stack([c["plain_curve"], c["padded_curve"]]).astype(float),
                   labels=["no padding", "padding=1"])),
        Panel("Digits missed, by digit", "test digits of each label named wrong after one "
              "epoch", "matrix", dict(values=np.array([per_plain, per_padded], float),
                                      rows=["no padding", "padding=1"],
                                      cols=[str(d) for d in range(course.CLASSES)])),
        Panel("Moved digits", "held-out accuracy on the test digits moved towards an edge "
              "(the mean over the four edges)", "curve",
              dict(x=np.asarray(sh["shifts"], float), ys=np.stack([pm, qm]),
                   labels=["no padding", "padding=1"])),
    )
    return Study("The CNN with padding, measured", "The course's CNN with and without "
                 "padding=1, one run each: accuracy, digits missed, and moved digits",
                 panels, meta)


STUDIES = {"padding-modes": modes_study, "padding-deep-stacks": deep_study,
           "padding-in-the-cnn": in_the_cnn}

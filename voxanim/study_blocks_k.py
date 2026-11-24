"""The numbers behind animations 215-217, the first three of "Blocks and code: a CNN" (group
``k``): the course's program with its model changed to a small CNN
(``examples/mnist_cnn.py``, :mod:`voxanim.study_course_cnn`).

* ``blocks-cnn-the-model``: the eight blocks of ``nn.Sequential`` as the model lines make
  them, each block's output shape on the first batch, ``print(model)``'s real text
  (:data:`voxanim.study_course_cnn.PRINTED_MODEL`, wrapped to fit the picture, nothing left
  out), and each block's count of learned numbers: 80 + 1,168 + 4,010 = 5,258, against the
  MLP's 52,650.
* ``blocks-cnn-shapes``: where the 400 of ``nn.Linear(16 * 5 * 5, 10)`` comes from: the side
  of the maps walked down, 28, -2 -> 26, /2 -> 13, -2 -> 11, /2 -> 5 (the odd last row and
  column dropped), and 16 maps x 5 x 5 = 400; with 784 there instead, torch's real error
  (:data:`voxanim.study_course_cnn.ERRORS`), in plain words: 400 numbers arrive, the layer
  expects 784.
* ``blocks-cnn-forward``: one real training digit, ``train_set[0]`` (a 5), through the CNN
  after its epoch (:func:`voxanim.study_course_cnn.activations`): 8 maps, pooled, 16 maps,
  pooled, 400 numbers in a strip, 10 scores; the 5 wins.

The squares of ``blocks-cnn-shapes`` show the same digit's real maps. Every number shown is
exact arithmetic on the CNN's recorded parameters and data, checked here against its own
forward pass.
"""

from __future__ import annotations

import itertools
import re

import numpy as np

from . import study_course as course
from . import study_course_cnn as cnn
from .blocks import code_line, code_lines, shape, torch_size
from .study import Panel, Study
from .study_blocks_b import unwrap

DIGIT = 0                     # train_set[0], the 5 the program's cnn-one-digit lines load
STEP = cnn.EPOCH_STEPS        # the CNN after its one epoch
MLP_COUNT = 52_650            # the course's MLP (blocks-a-model-is-a-stack)
WRAP = 37                     # print(model)'s lines, wrapped to fit the picture
INDENT = 6                    # and the indent of the rest of a wrapped line
SOURCE = ("the course's CNN (voxanim.study_course_cnn, examples/mnist_cnn.py's model trained "
          "one epoch with SGD on the course's batches), on MNIST")
_BLOCK = re.compile(r"nn\.\w+\([^()]*\)")


def model_blocks(region="cnn-model"):
    """The region's lines, each block written in them (``nn.Conv2d(1, 8, 3)`` ...), and the
    index of the line each block is on (two blocks share a line)."""
    lines = list(code_lines(region))
    names, at = [], []
    for i, s in enumerate(lines):
        for hit in _BLOCK.findall(s):
            names.append(hit)
            at.append(i)
    return lines, names, at


def kind(name):
    """``nn.Conv2d(1, 8, 3)`` -> ``Conv2d``."""
    return name[3:name.index("(")]


def stamps(w):
    """Convolution weights ``[out, in, 3, 3]`` as one picture: a row of 3 x 3 stamps per
    output map, one stamp per input map: ``[out x 3, in x 3]``."""
    o, i, k, _ = w.shape
    return np.asarray(w, float).transpose(0, 2, 1, 3).reshape(o * k, i * k)


def wrap_printed(text=cnn.PRINTED_MODEL, width=WRAP, indent=INDENT):
    """``print(model)``'s lines, each one too wide for the picture broken after a comma
    between two of its module's settings (never inside ``(3, 3)``), the rest indented
    ``indent`` spaces, with the module each line belongs to: ``[(line, block)]``, block -1
    for ``Sequential(`` and ``)``. Joined back (:func:`voxanim.study_blocks_b.unwrap`), the
    pieces are the printed text."""
    out = []
    for raw in text.splitlines():
        block = -1
        s = raw.strip()
        if s.startswith("(") and "):" in s:
            block = int(s[1:s.index(")")])
        # the places a line may break: after a comma at depth 1 (in the module's call)
        cuts, depth = [], 0
        for i, ch in enumerate(raw):
            depth += (ch == "(") - (ch == ")")
            if raw.startswith(", ", i) and depth == 1:
                cuts.append(i)
        # the pieces between those places, each but the last ending in its comma
        pieces = [raw[a:b + 1] for a, b in zip([0] + [c + 2 for c in cuts], cuts,
                                                strict=False)] + \
            [raw[cuts[-1] + 2:] if cuts else raw]
        line = pieces[0]
        for piece in pieces[1:]:
            if len(line) + 1 + len(piece) > width:
                out.append((line, block))
                line = " " * indent + piece
            else:
                line += " " + piece
        out.append((line, block))
        if any(len(s) > width for s, b in out if b == block >= 0):
            raise ValueError("cannot wrap %r to %d" % (raw, width))
    return out


def pages(pieces):
    """``print(model)``'s wrapped lines in two pages, so each fits the picture: blocks 0-3
    with ``Sequential(``, and blocks 4-7 with ``)``. Returns ``[[(line, block)], ...]``."""
    cut = next(n for n, (_s, b) in enumerate(pieces) if b == 4)
    return [pieces[:cut], pieces[cut:]]


# ====================================================================== 215
def the_model():
    P = cnn.params(STEP)
    lines, names, at = model_blocks()
    kinds = [kind(n) for n in names]
    if tuple(kinds) != cnn.BLOCKS:
        raise ValueError("the model lines are not the CNN's blocks: %s" % kinds)
    shapes = [list(s) for _name, s in cnn.shapes()]
    per = cnn.counts(P)
    counts = [per[k]["total"] if k in per else 0 for k in range(len(names))]
    weights = [per[k]["weights"] if k in per else 0 for k in range(len(names))]
    biases = [per[k]["biases"] if k in per else 0 for k in range(len(names))]
    total = int(sum(counts))
    if total != cnn.count(P) or total != 5_258:
        raise ValueError("the blocks' counts do not add up to the CNN's %d" % cnn.count(P))
    if course.count() != MLP_COUNT:
        raise ValueError("the MLP has %d numbers, not %d" % (course.count(), MLP_COUNT))
    pieces = wrap_printed()
    if unwrap(pieces) != cnn.PRINTED_MODEL:
        raise ValueError("the wrapped print(model) is not the printed text")
    split = pages(pieces)
    logits = cnn.forward(P, course.first_batch()["x"])["logits"]
    sums = ["%s + %d = %s" % (" × ".join(str(d) for d in per[k]["shape"]), per[k]["biases"],
                              format(per[k]["total"], ",")) for k in cnn.LEARNS]
    meta = dict(
        source=SOURCE, evidence="exact arithmetic on the CNN's parameters and the first batch",
        names=names, kinds=kinds, lines=at, model=lines, sequential_line=0,
        shapes=shapes[1:], input_shape=shapes[0], input_text=shape(shapes[0]),
        shape_texts=[shape(s) for s in shapes[1:]],
        per_digit=[int(np.prod(s[1:])) for s in shapes],
        side_walk=list(cnn.side_walk()), counts=counts, weights=weights, biases=biases,
        total=total, total_weights=int(sum(weights)), total_biases=int(sum(biases)),
        learning=[k for k, c in enumerate(counts) if c], sums=sums,
        mlp_total=MLP_COUNT, times_fewer=MLP_COUNT / total,
        printed=cnn.PRINTED_MODEL, printed_lines=[s for s, _ in pieces],
        printed_blocks=[b for _, b in pieces],
        pages=[[s for s, _ in page] for page in split],
        page_blocks=[sorted({b for _, b in page if b >= 0}) for page in split],
        limitations=["print(model)'s lines are up to 81 characters, too wide for the picture: "
                     "they are broken after a comma (nothing left out) and shown in two "
                     "halves, blocks 0-3 then 4-7, as all 20 lines do not fit at once",
                     "examples/mnist_cnn.py has no line that prints the count; it is "
                     "sum(p.numel() for p in model.parameters()), as the MLP's in "
                     "examples/blocks_parts.py (region count)",
                     "the weights drawn in the blocks are those after the epoch; a count does "
                     "not depend on training"])
    panels = (
        Panel("The blocks", "learned numbers per block of the CNN's nn.Sequential", "bars",
              dict(values=np.array(counts, float), labels=names)),
        Panel("Their outputs", "numbers per digit out of each block, on the first batch "
              "(the input first)", "bars",
              dict(values=np.array(meta["per_digit"], float),
                   labels=["input"] + names)),
        Panel("The weights", "0.weight (8 stamps of 3 x 3), 3.weight (16 x 8 stamps) and "
              "7.weight (10 x 400), after one epoch", "matrix",
              dict(values=stamps(P["0.weight"]), second=stamps(P["3.weight"]),
                   third=P["7.weight"].astype(float))),
        Panel("What comes out", "model(x) for the first batch: 10 scores per digit", "matrix",
              dict(values=logits.astype(float))),
    )
    return Study("The CNN, block by block", "The course's program with a CNN for its model: "
                 "eight blocks, and 5,258 numbers to learn instead of 52,650", panels, meta)


# ====================================================================== 216
def walk_maps(a):
    """One map of each stage of activations ``a`` (a batch of one): the digit, then the first
    convolution's busiest map before and after pooling, then the second's: ``[(28 x 28),
    (26 x 26), (13 x 13), (11 x 11), (5 x 5)]`` and the two maps' indices."""
    c1 = int(a["h1"][0].reshape(len(a["h1"][0]), -1).sum(1).argmax())
    c2 = int(a["h2"][0].reshape(len(a["h2"][0]), -1).sum(1).argmax())
    maps = [a["x"][0, 0], a["h1"][0, c1], a["p1"][0, c1], a["h2"][0, c2], a["p2"][0, c2]]
    return [np.asarray(m, float) for m in maps], c1, c2


def the_shapes():
    a = cnn.activations(STEP, DIGIT)
    walk = list(cnn.side_walk())
    maps, c1, c2 = walk_maps(a)
    if [m.shape[0] for m in maps] != walk:
        raise ValueError("the maps' sides are not the side walk")
    p1, _ = cnn.pool(a["h1"])
    p2, _ = cnn.pool(a["h2"])
    if not (np.array_equal(p1, a["p1"]) and np.array_equal(p2, a["p2"])):
        raise ValueError("the pooled maps are not MaxPool2d(2) of the maps")
    # pooling an 11 x 11 map reads its first 10 rows and columns only
    if not np.array_equal(p2, cnn.pool(a["h2"][:, :, :10, :10])[0]):
        raise ValueError("pooling 11 x 11 reads more than 10 x 10")
    maps16 = np.asarray(a["p2"][0], float)
    channels = cnn.CHANNELS[-1]
    side = walk[-1]
    in_features = channels * side * side
    if in_features != cnn.IN_FEATURES or in_features != cnn.counts()[7]["shape"][1]:
        raise ValueError("16 x 5 x 5 is not nn.Linear's in_features")
    error = cnn.ERRORS["cnn-mistake-in-features"]
    m = re.search(r"\((\d+)x(\d+) and (\d+)x(\d+)\)", error)
    rows, got, wants, outs = (int(v) for v in m.groups())
    if (rows, got, outs) != (course.BATCH, in_features, cnn.CLASSES):
        raise ValueError("the error is not about %d rows of %d" % (course.BATCH, in_features))
    wrong = code_line("cnn-mistake-in-features", "nn.Linear(")
    if "(%d," % wants not in wrong:
        raise ValueError("the broken line does not ask for %d" % wants)
    shapes = [list(s) for _name, s in cnn.shapes()]
    steps = []
    for n, (a_side, b_side) in enumerate(itertools.pairwise(walk)):
        conv = n % 2 == 0
        steps.append(dict(kind="conv" if conv else "pool", before=a_side, after=b_side,
                          op="−2" if conv else "÷2",
                          sum=("%d − 2 = %d" % (a_side, b_side) if conv
                               else "%d ÷ 2 = %s" % (a_side, ("%d" % b_side) if a_side % 2 == 0
                                                     else "%.1f → %d" % (a_side / 2, b_side)))))
    cut = error.index(" shapes")
    cut2 = error.index(" (")
    b = course.first_batch()
    meta = dict(
        source=SOURCE, evidence="exact arithmetic: the shapes, and torch's own error",
        digit=DIGIT, label=int(a["y"]), maps_used=[c1, c2], side_walk=walk, steps=steps,
        kernel=cnn.KERNEL, pool=cnn.POOL, channels=list(cnn.CHANNELS),
        maps=channels, side=side, in_features=in_features,
        in_features_sum="%d × %d × %d = %d" % (channels, side, side, in_features),
        dropped=walk[-2] ** 2 - (2 * side) ** 2, odd=walk[-2],
        shapes=shapes, shape_texts=[shape(s) for s in shapes],
        transitions=["%s → %s" % (shape(shapes[i]), shape(shapes[j]))
                     for i, j in ((0, 2), (2, 3), (3, 5), (5, 6), (6, 7), (7, 8))],
        batch_printed="%s\n%s" % (torch_size(b["x"].shape), torch_size(b["y"].shape)),
        linear_code=code_line("cnn-model", "nn.Linear("), wrong_code=wrong,
        error=error, error_lines=[error[:cut], error[cut + 1:cut2], error[cut2 + 1:]],
        rows=rows, got=got, wants=wants, outs=outs,
        limitations=["the squares show one map at each stage (the busiest of the digit's), "
                     "drawn to scale; there are 8, then 16, of them",
                     "the error is torch's, checked against torch by "
                     "tests/test_mnist_blocks_equivalence.py; the broken program is region "
                     "cnn-mistake-in-features of examples/blocks_parts.py"])
    panels = (
        Panel("The side walked down", "the side of the square maps after each block that "
              "changes it", "bars", dict(values=np.array(walk, float),
                                         labels=["input", "Conv2d", "MaxPool2d", "Conv2d",
                                                 "MaxPool2d"])),
        Panel("One map at each stage", "train_set[0] and one of its maps after each block, "
              "through the trained CNN", "matrix",
              dict(values=maps[0], conv1=maps[1], pool1=maps[2], conv2=maps[3],
                   pool2=maps[4])),
        Panel("16 maps of 5 x 5", "what nn.Flatten() unrolls into 400 numbers", "images",
              dict(values=maps16, labels=[str(k) for k in range(channels)])),
        Panel("What arrives, what is expected", "numbers per digit reaching nn.Linear, "
              "against its in_features: right (400) and wrong (784)", "bars",
              dict(values=np.array([got, in_features, wants], float),
                   labels=["arrive", "nn.Linear(16 * 5 * 5, 10)", "nn.Linear(784, 10)"])),
    )
    return Study("Where the 400 comes from", "28 -> 26 -> 13 -> 11 -> 5: sixteen maps of "
                 "5 x 5 are the 400 numbers the CNN's last layer reads", panels, meta)


# ====================================================================== 217
def forward_one():
    P = cnn.params(STEP)
    a = cnn.activations(STEP, DIGIT)
    y = int(a["y"])
    logits = np.asarray(a["logits"][0], float)
    said = int(logits.argmax())
    if said != y:
        raise ValueError("the trained CNN gets train_set[%d] wrong" % DIGIT)
    flat = np.asarray(a["flat"][0], float)
    if not np.array_equal(a["flat"][0], a["p2"][0].reshape(-1)):
        raise ValueError("flat is not the 16 maps unrolled")
    if not np.allclose(flat @ P["7.weight"].T.astype(float) + P["7.bias"], logits, atol=1e-4):
        raise ValueError("the scores are not nn.Linear of the 400 numbers")
    order = np.argsort(-logits)
    second = int(order[1])
    x, label = course.digit(DIGIT)
    if label != y or not np.array_equal(x[0], a["x"][0, 0]):
        raise ValueError("the digit is not train_set[%d]" % DIGIT)
    stages = ("x", "h1", "p1", "h2", "p2", "flat", "logits")
    shapes = {k: list(a[k].shape) for k in stages}
    zero = {k: float((a[k] <= 0).mean()) for k in ("h1", "h2")}
    meta = dict(
        source=SOURCE, evidence="exact arithmetic on the CNN after one epoch",
        digit=DIGIT, label=y, said=said, score=float(logits[y]), second=second,
        second_score=float(logits[second]), prob=float(a["probs"][0, y]),
        scores=[float(v) for v in logits],
        shapes=shapes, shape_texts={k: shape(s) for k, s in shapes.items()},
        per_digit={k: int(np.prod(s[1:])) for k, s in shapes.items()},
        maps={"h1": int(a["h1"].shape[1]), "h2": int(a["h2"].shape[1])},
        zero_share=zero,
        ranges={k: [float(a[k].min()), float(a[k].max())] for k in stages},
        dead=[int(c) for c in np.flatnonzero(a["h2"][0].reshape(16, -1).max(1) <= 0)],
        printed="%s %d" % (torch_size(x.shape), y),
        model=list(code_lines("cnn-model")),
        limitations=["one digit is a batch of one, [1, 1, 28, 28], as model(x[0:1]) gives "
                     "(region cnn-fix-no-batch of examples/blocks_parts.py)",
                     "the maps are shaded from 0 (paper) to the biggest number of their "
                     "stage; ReLU leaves no number below 0"])
    panels = (
        Panel("The digit", "train_set[0], normalised: the input", "matrix",
              dict(values=np.asarray(a["x"][0, 0], float))),
        Panel("8 maps", "nn.Conv2d(1, 8, 3) and nn.ReLU(), then nn.MaxPool2d(2)", "images",
              dict(values=np.asarray(a["h1"][0], float), pooled=np.asarray(a["p1"][0], float),
                   labels=[str(k) for k in range(8)])),
        Panel("16 maps", "nn.Conv2d(8, 16, 3) and nn.ReLU(), then nn.MaxPool2d(2)", "images",
              dict(values=np.asarray(a["h2"][0], float), pooled=np.asarray(a["p2"][0], float),
                   labels=[str(k) for k in range(16)])),
        Panel("10 scores", "nn.Linear(16 * 5 * 5, 10) of the 400 numbers: the biggest wins",
              "bars", dict(values=logits, labels=[str(d) for d in range(cnn.CLASSES)],
                           flat=flat)),
    )
    return Study("One digit through the CNN", "train_set[0], a 5, through the trained CNN: "
                 "8 maps, 16 maps, 400 numbers, and 10 scores where the 5 wins", panels, meta)


STUDIES = {"blocks-cnn-the-model": the_model, "blocks-cnn-shapes": the_shapes,
           "blocks-cnn-forward": forward_one}

"""The numbers behind animations 188-190 of the "Blocks and code" series (group ``b``):
``nn.Linear``, ``nn.ReLU`` and ``nn.Sequential``, on the course's model after its one epoch
of SGD (:func:`voxanim.study_course.params`) and the first batch's real activations
(:func:`voxanim.study_course.activations`).

* ``blocks-a-linear-layer``: ``nn.Linear(784, 64)`` is a grid of 64 x 784 weights and a
  strip of 64 biases. For the first digit of the first batch, output :data:`OUTPUT` (the
  biggest) is its 784 numbers times the grid's row of 784 weights, added up, plus that
  row's bias; the three biggest products are read out. Out of the whole batch comes
  ``[64, 64]``.
* ``blocks-relu``: ``nn.ReLU()`` on those 64 numbers: every negative one becomes 0;
  :data:`WINDOW` of them read out. Measured: the share it zeroes in this digit, in the
  first batch and over all 60,000 training digits (and at the second ReLU).
* ``blocks-a-model-is-a-stack``: ``nn.Sequential`` stacks the six blocks into the
  784-64-32-10 model: each block's output shape on the first batch, ``print(model)``'s
  real text (:data:`voxanim.study_course.PRINTED_MODEL`), and each block's count of
  learned numbers, 52,650 in all (as "Counting the weights: a DNN and a CNN", 183, counts
  them).

Every number shown is exact arithmetic on the recorded weights and data, checked here
against the model's own forward pass.
"""

from __future__ import annotations

import numpy as np

from . import study_course as course
from .blocks import code_line, code_lines, shape
from .study import Panel, Study

DIGIT = 0                     # the first digit of the first batch
OUTPUT = None                 # the output followed in 188: None picks the biggest
PAIRS = 3                     # the products read out: the biggest
WINDOW = 8                    # the numbers ReLU is shown on, from the followed output
COUNTED = "Counting the weights: a DNN and a CNN"      # catalogue 183
COUNTED_TOTAL = 52_650
WRAP = 34                     # print(model)'s lines, wrapped to fit the picture
SOURCE = ("the course's model after one epoch of SGD (voxanim.study_course, as "
          "examples/mnist_blocks.py trains it), on the first batch of MNIST digits")


def trained():
    """The parameters after the epoch, and the first batch through them."""
    return course.params(course.EPOCH_STEPS), course.activations(course.EPOCH_STEPS, 0)


def wrap_printed(text=course.PRINTED_MODEL, width=WRAP, indent=6):
    """``print(model)``'s lines, each one too wide for the picture broken after a comma
    (the rest indented ``indent`` spaces), with the module each line belongs to: ``[(line,
    block)]``, block -1 for ``Sequential(`` and ``)``. Nothing is left out: joined back,
    the pieces are the printed text."""
    out = []
    for raw in text.splitlines():
        block = -1
        s = raw.strip()
        if s.startswith("(") and "):" in s:
            block = int(s[1:s.index(")")])
        line = raw
        while len(line) > width:
            cut = line.rfind(", ", 0, width)
            if cut < 0:
                raise ValueError("cannot wrap %r" % raw)
            out.append((line[:cut + 1], block))
            line = " " * indent + line[cut + 2:]
        out.append((line, block))
    return out


def unwrap(pieces, indent=6):
    """The printed text back from :func:`wrap_printed`'s pieces."""
    lines = []
    for s, _block in pieces:
        if lines and s.startswith(" " * indent) and lines[-1].endswith(","):
            lines[-1] += " " + s[indent:]
        else:
            lines.append(s)
    return "\n".join(lines)


def model_lines():
    """The region ``model``'s lines, and the index of each block's line in it (0 Flatten,
    1 Linear, ...)."""
    lines = list(code_lines("model"))
    at = [i for i, s in enumerate(lines) if s.strip().startswith("nn.")]
    return lines, at


# ====================================================================== 188
def a_linear_layer():
    P, act = trained()
    b = course.first_batch()
    W, bias = P["1.weight"], P["1.bias"]
    x = act["x"][DIGIT]
    z = act["z1"][DIGIT]
    j = int(np.argmax(z)) if OUTPUT is None else OUTPUT
    prod = (W[j] * x).astype(np.float64)
    added = float(prod.sum())
    out = float(z[j])
    if not np.isclose(added + float(bias[j]), out, atol=1e-4):
        raise ValueError("output %d is not the products added up plus the bias" % j)
    if not np.allclose(x @ W.T + bias, z, atol=1e-4):
        raise ValueError("the outputs are not x @ W.T + b")
    top = [int(i) for i in np.argsort(-prod)[:PAIRS]]
    lines, at = model_lines()
    meta = dict(
        source=SOURCE, evidence="exact arithmetic on the trained weights", digit=DIGIT,
        label=int(b["y"][DIGIT]), index=int(b["indices"][DIGIT]), output=j,
        inputs=int(W.shape[1]), outputs=int(W.shape[0]), weight_shape=list(W.shape),
        weights=int(W.size), biases=int(bias.size), numbers=int(W.size + bias.size),
        pairs=[dict(at=i, row=i // course.SIDE, col=i % course.SIDE, x=float(x[i]),
                    w=float(W[j, i]), product=float(prod[i])) for i in top],
        added=added, bias=float(bias[j]), out=out,
        positive_products=int((prod > 0).sum()),
        batch_in=list(act["x"].shape), batch_out=list(act["z1"].shape),
        batch_in_text=shape(act["x"].shape), batch_out_text=shape(act["z1"].shape),
        w_range=[float(W.min()), float(W.max())], out_range=[float(z.min()), float(z.max())],
        model=lines, linear_line=at[1], code=code_line("model", "Linear(784"),
        limitations=["the weights are drawn 784 columns wide in about 300 pixels, so "
                     "neighbouring weights share a pixel there; the numbers read out are "
                     "single weights",
                     "the weights are those after one epoch of the numpy twin's SGD run "
                     "(voxanim.study_course), which the script's own run matches step for "
                     "step from the same start; its shuffle differs"])
    panels = (
        Panel("x: one digit", "the first digit of the first batch after nn.Flatten(): "
              "784 numbers", "matrix", dict(values=x[None].astype(float))),
        Panel("nn.Linear(784, 64)", "1.weight, 64 rows of 784 weights, and 1.bias, after "
              "one epoch", "matrix", dict(values=W.astype(float), bias=bias.astype(float))),
        Panel("One output, added up", "the products x[i] * W[%d, i], added up from i = 0 "
              "to 783" % j, "curve", dict(x=np.arange(W.shape[1], dtype=float),
                                          ys=np.cumsum(prod)[None],
                                          labels=["added so far"])),
        Panel("Its 64 outputs", "nn.Linear(784, 64) of the digit: one number per row of "
              "weights", "bars", dict(values=z.astype(float),
                                      labels=[str(k) for k in range(len(z))],
                                      batch=act["z1"].astype(float))),
    )
    return Study("nn.Linear: a grid of weights", "One output is the digit's 784 numbers "
                 "times one row of weights, added up, plus a bias", panels, meta)


# ====================================================================== 189
def relu():
    P, act = trained()
    z, h = act["z1"], act["h1"]
    if not np.array_equal(h, np.maximum(z, 0)):
        raise ValueError("h1 is not ReLU of z1")
    j = int(np.argmax(z[DIGIT]))
    window = list(range(j, j + WINDOW))
    x = course.data()[0]
    Z = x @ P["1.weight"].T + P["1.bias"]
    Z2 = np.maximum(Z, 0) @ P["3.weight"].T + P["3.bias"]
    digit = z[DIGIT]
    lines, at = model_lines()
    meta = dict(
        source=SOURCE, evidence="exact arithmetic; the shares measured on MNIST",
        digit=DIGIT, output=j, window=window, before=[float(v) for v in digit[window]],
        after=[float(v) for v in h[DIGIT][window]],
        values=[float(v) for v in digit], size=int(digit.size),
        zeroed=int((digit <= 0).sum()), kept=int((digit > 0).sum()),
        share=float((digit <= 0).mean()),
        batch_shape=list(z.shape), shape_text=shape(z.shape), batch_numbers=int(z.size),
        batch_zeroed=int((z <= 0).sum()), batch_share=float((z <= 0).mean()),
        train_digits=int(len(x)), train_share=float((Z <= 0).mean()),
        second_batch_share=float((act["z2"] <= 0).mean()),
        second_train_share=float((Z2 <= 0).mean()),
        never=int((Z <= 0).all(0).sum()), weights=0,
        model=lines, relu_line=at[2], code=lines[at[2]].strip(),
        limitations=["a number exactly 0 stays 0; the share counts numbers at or below 0 "
                     "(none of these is exactly 0)",
                     "the shares are for the model after one epoch of SGD"])
    panels = (
        Panel("Out of nn.Linear", "the digit's 64 numbers before ReLU", "bars",
              dict(values=digit.astype(float), labels=[str(k) for k in range(64)])),
        Panel("Out of nn.ReLU", "the same 64 numbers after ReLU: the negative ones are 0",
              "bars", dict(values=h[DIGIT].astype(float), labels=[str(k) for k in range(64)])),
        Panel("The whole batch", "nn.ReLU of the first batch: [64, 64], zeros marked",
              "matrix", dict(values=h.astype(float), zero=(z <= 0).astype(float))),
        Panel("How many become 0", "the share of numbers ReLU sets to 0", "bars",
              dict(values=np.array([meta["share"], meta["batch_share"], meta["train_share"],
                                    meta["second_train_share"]]),
                   labels=["this digit", "first batch", "60,000 digits",
                           "second ReLU, 60,000"])),
    )
    return Study("nn.ReLU: negatives become 0", "nn.ReLU turns every negative number into 0: "
                 "about half of the model's first hidden numbers", panels, meta)


# ====================================================================== 190
def a_model_is_a_stack():
    P, act = trained()
    lines, at = model_lines()
    names = [lines[i].strip().rstrip(",") for i in at]
    b = course.first_batch()
    shapes = [list(act[k].shape) for k in ("x", "z1", "h1", "z2", "h2", "logits")]
    counts, weights, biases = [], [], []
    for k in range(len(names)):
        w, bb = P.get("%d.weight" % k), P.get("%d.bias" % k)
        weights.append(0 if w is None else int(w.size))
        biases.append(0 if bb is None else int(bb.size))
        counts.append(weights[-1] + biases[-1])
    total = int(sum(counts))
    if total != course.count(P) or total != COUNTED_TOTAL:
        raise ValueError("the blocks' counts do not add up to the model's %d" % course.count(P))
    pieces = wrap_printed()
    if unwrap(pieces) != course.PRINTED_MODEL:
        raise ValueError("the wrapped print(model) is not the printed text")
    sums = ["%d × %d + %d = %s" % (w // bb, bb, bb, format(w + bb, ","))
            for w, bb in zip(weights, biases) if bb]
    meta = dict(
        source=SOURCE, evidence="exact arithmetic on the model's parameters",
        names=names, kinds=[n[3:n.index("(")] for n in names], lines=at, model=lines,
        sequential_line=0, input_shape=list(b["x"].shape), shapes=shapes,
        shape_texts=[shape(s) for s in shapes], input_text=shape(b["x"].shape),
        sizes=list(course.SIZES), counts=counts, weights=weights, biases=biases,
        total=total, total_weights=int(sum(weights)), total_biases=int(sum(biases)),
        sums=sums, first_share=counts[1] / total,
        printed=course.PRINTED_MODEL, printed_lines=[s for s, _ in pieces],
        printed_blocks=[k for _, k in pieces], counted=COUNTED,
        count_code=list(code_lines("count")), printed_sum=str(total),
        limitations=["print(model)'s lines are up to 58 characters, too wide for the "
                     "picture: the three Linear lines and the Flatten line are broken "
                     "after a comma, nothing left out",
                     "the sum line, print(sum(p.numel() for p in params)), is in "
                     "examples/blocks_parts.py (region count), the same model untrained: "
                     "a count does not depend on training"])
    panels = (
        Panel("The blocks", "learned numbers per block of nn.Sequential", "bars",
              dict(values=np.array(counts, float), labels=names)),
        Panel("Their outputs", "numbers per digit out of each block, on the first batch",
              "bars", dict(values=np.array([s[1] for s in shapes], float), labels=names)),
        Panel("The weight grids", "1.weight, 3.weight and 5.weight after one epoch",
              "matrix", dict(values=P["1.weight"].astype(float),
                             second=P["3.weight"].astype(float),
                             third=P["5.weight"].astype(float))),
        Panel("What comes out", "model(x) for the first batch: 10 scores per digit",
              "matrix", dict(values=act["logits"].astype(float))),
    )
    return Study("A model is a stack of blocks", "nn.Sequential stacks six blocks into the "
                 "784-64-32-10 digit classifier, which learns 52,650 numbers", panels, meta)


STUDIES = {"blocks-a-linear-layer": a_linear_layer, "blocks-relu": relu,
           "blocks-a-model-is-a-stack": a_model_is_a_stack}

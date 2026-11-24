"""The numbers behind animations 200-202 of the "Blocks and code" series (group ``f``): the
forward pass, the loss and ``loss.backward()``, all on the first batch of the course's run
at step 0, with the untrained weights (:mod:`voxanim.study_course`).

* ``blocks-forward-pass``: ``logits = model(x)``: the batch through every block of
  ``nn.Sequential``, with the shape it has after each one: ``[64, 1, 28, 28]`` ->
  ``[64, 784]`` -> ``[64, 64]`` (Linear, then ReLU, which zeroes about half) -> ``[64, 32]``
  -> ``[64, 10]``, the logits.
* ``blocks-the-loss``: ``loss = loss_fn(logits, y)``: each digit's loss is ``-ln`` of the
  chance the model gives its right label; the 64 losses' mean is the loss, 2.322 for the
  untrained model, close to ``ln 10 = 2.303``, the loss of a blind guess (1 in 10 for every
  digit).
* ``blocks-backward``: ``loss.backward()``: a gradient for every weight and bias, the last
  layer's first; each ``.grad`` has its parameter's shape, 52,650 numbers in all.

Every number is the course's own (``activations(0)`` and ``grads(0, "sgd")``, which
``tests/test_mnist_blocks_equivalence.py`` checks against the script's PyTorch lines), and
the study checks the identities it shows: the shapes against the arrays, the mean against
the sum, each loss against ``-ln p``, each gradient's shape against its weight's.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import study_course as course
from .blocks import code_line, code_lines, commas, shape
from .study import Panel, Study

STEP = 0                      # the first step: the untrained weights, the first batch
DIGIT = 0                     # the batch's digit whose loss is worked out in full
SOURCE = ("MNIST, as examples/mnist_blocks.py loads it; the first batch (voxanim.study_course, "
          "seeded) and the weights nn.Linear starts with, before any step")


def block_name(line):
    """A line of region ``model`` as the block's short name: ``nn.Linear(784, 64),`` ->
    ``Linear(784, 64)``."""
    return line.strip().rstrip(",").removeprefix("nn.")


def stages():
    """The forward pass of the first batch at step 0, block by block: a list of
    ``dict(name, line, kind, values, zero)`` (``line``: the block's line in region
    ``model``; ``values``: the batch after that block, ``[64, n]``, or the batch itself,
    ``[64, 1, 28, 28]``, for the input; ``zero``: the numbers ReLU set to 0)."""
    a = course.activations(STEP, STEP)
    b = course.batch(STEP)
    model = list(code_lines("model"))
    lines = [i for i, s in enumerate(model) if s.strip().startswith("nn.")]
    names = [block_name(model[i]) for i in lines]
    outs = [a["x"], a["z1"], a["h1"], a["z2"], a["h2"], a["logits"]]
    out = [dict(name="x", line=None, kind="input", values=b["x"], zero=None)]
    for name, line, v, before in zip(names, lines, outs, [None, None, a["z1"], None, a["z2"],
                                                          None]):
        kind = name.split("(")[0].lower()
        zero = before <= 0 if kind == "relu" else None
        out.append(dict(name=name, line=line, kind=kind, values=v, zero=zero))
    return out


# ====================================================================== 200
def forward_pass():
    st = stages()
    a = course.activations(STEP, STEP)
    if [s["kind"] for s in st] != ["input", "flatten", "linear", "relu", "linear", "relu",
                                   "linear"]:
        raise ValueError("the model is not Flatten, Linear, ReLU, Linear, ReLU, Linear")
    P = course.params(STEP)
    again = course.forward(P, course.batch(STEP)["x"])
    if not np.array_equal(again["logits"], a["logits"]):
        raise ValueError("the forward pass does not give the recorded logits")
    shapes = [list(s["values"].shape) for s in st]
    zeroed = [float(s["zero"].mean()) for s in st if s["kind"] == "relu"]
    for s in st:
        if s["kind"] == "relu" and not np.array_equal(s["zero"], s["values"] == 0):
            raise ValueError("ReLU's zeros are not where its input was not positive")
    logits = a["logits"]
    meta = dict(
        source=SOURCE, evidence="exact arithmetic: the first batch through the untrained model",
        step=STEP, batch=int(len(logits)), names=[s["name"] for s in st],
        kinds=[s["kind"] for s in st], lines=[s["line"] for s in st], shapes=shapes,
        shape_texts=[shape(v) for v in shapes],
        changes=["%s → %s" % (shape(a_), shape(b_)) for a_, b_ in itertools.pairwise(shapes)],
        zeroed=zeroed, zeroed_text=["%d%%" % round(100 * z) for z in zeroed],
        logits_low=float(logits.min()), logits_high=float(logits.max()),
        label=int(a["y"][DIGIT]), digit_logits=[float(v) for v in logits[DIGIT]],
        model=list(code_lines("model")), step_code=list(code_lines("step")),
        call=code_line("step", "model(x)"),
        limitations=["the blocks of 64 rows are drawn a few pixels tall and up to 125 wide, "
                     "so neighbouring numbers share a pixel; their colours are the averages",
                     "the weights are the untrained ones of step 0, so the logits are small "
                     "and nearly the same for every digit"])
    panels = (
        Panel("The batch", "x: the first batch, 64 digits, as the loader gives it", "images",
              dict(values=course.batch(STEP)["x"][:, 0].astype(float),
                   labels=["%d" % v for v in a["y"]])),
        Panel("After Linear(784, 64) and ReLU", "the 64 numbers per digit after the first "
              "Linear (before) and after ReLU (values)", "matrix",
              dict(values=a["h1"].astype(float), before=a["z1"].astype(float))),
        Panel("After Linear(64, 32) and ReLU", "the 32 numbers per digit after the second "
              "Linear (before) and after ReLU (values)", "matrix",
              dict(values=a["h2"].astype(float), before=a["z2"].astype(float))),
        Panel("The logits", "model(x): 10 scores per digit, [64, 10]", "matrix",
              dict(values=logits.astype(float))),
    )
    return Study("The forward pass", "logits = model(x): one batch through every block, "
                 "its shape changing at each", panels, meta)


# ====================================================================== 201
def the_loss():
    a = course.activations(STEP, STEP)
    y, probs, each = a["y"], a["probs"].astype(np.float64), a["loss_each"]
    if not np.allclose(each, -np.log(probs[np.arange(len(y)), y]), atol=1e-5):
        raise ValueError("a digit's loss is not -ln of its right label's chance")
    loss = float(each.mean())
    recorded = float(course.losses("sgd")[STEP])
    if abs(loss - recorded) > 1e-5:
        raise ValueError("the mean of the 64 losses is not the recorded loss")
    total = float(each.astype(np.float64).sum())
    p = float(probs[DIGIT, y[DIGIT]])
    ln10 = float(np.log(course.CLASSES))
    meta = dict(
        source=SOURCE, evidence="measured on MNIST: the first batch, untrained",
        step=STEP, batch=int(len(y)), labels=[int(v) for v in y],
        losses=[float(v) for v in each], loss=loss, loss_text="%.3f" % loss,
        total=total, total_text="%.2f" % total, mean_text="%.2f ÷ %d = %.3f" % (
            total, len(y), loss),
        digit=DIGIT, label=int(y[DIGIT]), chances=[float(v) for v in probs[DIGIT]],
        chance=p, chance_text="%.3f" % p, digit_loss=float(each[DIGIT]),
        digit_loss_text="−ln %.3f = %.2f" % (p, float(each[DIGIT])),
        lowest=float(each.min()), highest=float(each.max()),
        guess=1.0 / course.CLASSES, ln10=ln10, ln10_text="%.3f" % ln10,
        over=loss - ln10, define=code_line("loss", "CrossEntropyLoss"),
        call=code_line("step", "loss_fn("),
        limitations=["each digit's chance is the softmax of its logits, which "
                     "nn.CrossEntropyLoss works out inside: the model itself outputs logits",
                     "the loss is of the untrained model on one batch; ln 10 is what a model "
                     "giving every digit exactly 1 in 10 would score"])
    panels = (
        Panel("One digit's chances", "softmax of the logits of the batch's first digit "
              "(label %d)" % y[DIGIT], "bars",
              dict(values=probs[DIGIT], labels=[str(d) for d in range(course.CLASSES)])),
        Panel("One loss per digit", "-ln of the chance given to each digit's right label",
              "bars", dict(values=each.astype(float), labels=["%d" % v for v in y])),
        Panel("Their mean, and a blind guess", "the loss (the mean of the 64) against ln 10",
              "bars", dict(values=np.array([loss, ln10]), labels=["loss", "ln 10"])),
        Panel("Every digit's chances", "softmax of the logits, [64, 10]", "matrix",
              dict(values=probs)),
    )
    return Study("The loss", "loss = loss_fn(logits, y): the mean of 64 losses, 2.322, close "
                 "to ln 10 for an untrained model", panels, meta)


# ====================================================================== 202
def backward():
    P = course.params(STEP)
    G = course.grads(STEP, "sgd")
    b = course.batch(STEP)
    loss, again = course.loss_and_grads(P, b["x"], b["y"])
    for k in course.KEYS:
        if G[k].shape != P[k].shape:
            raise ValueError("%s: .grad is not its weight's shape" % k)
        if not np.allclose(G[k], again[k], atol=1e-7):
            raise ValueError("%s: the recorded gradient is not loss.backward()'s" % k)
    layers = []
    total = 0
    for i, n in zip(course.LINEAR, ("Linear(784, 64)", "Linear(64, 32)", "Linear(32, 10)")):
        w, bias = "%d.weight" % i, "%d.bias" % i
        total += P[w].size + P[bias].size
        layers.append(dict(index=i, name=n, weight=w, bias=bias, shape=list(P[w].shape),
                           bias_shape=list(P[bias].shape), shape_text=shape(P[w].shape),
                           bias_text=shape(P[bias].shape), numbers=int(P[w].size + P[bias].size),
                           largest=float(np.abs(G[w]).max())))
    # loss.backward() works from the last layer back to the first
    order = list(reversed(range(len(layers))))
    running, so_far = [], 0
    for k in order:
        so_far += layers[k]["numbers"]
        running.append(so_far)
    if so_far != course.count(P):
        raise ValueError("the gradients are not one per number")
    meta = dict(
        source=SOURCE, evidence="measured on MNIST: loss.backward() on the first batch",
        step=STEP, loss=float(loss), loss_text="%.3f" % loss, layers=layers, order=order,
        running=running, running_text=[commas(v) for v in running], total=int(so_far),
        total_text=commas(so_far), printed=course.PRINTED_GRAD,
        call=code_line("step", "backward"), grad=list(code_lines("grad")),
        names=list(course.KEYS),
        limitations=["each tile's shade is its gradient's value, lightest for the most "
                     "negative of that layer and deepest red for the most positive; the "
                     "blocks are drawn smaller than their numbers, so a pixel there is an "
                     "average", "the weights' blocks are not drawn to scale with each other"])
    panels = tuple(
        Panel("%s.grad" % layers[k]["weight"], "the gradient loss.backward() leaves in "
              "model[%d].weight.grad, shape %s, beside the weight and the bias"
              % (layers[k]["index"], layers[k]["shape_text"]), "matrix",
              dict(values=G[layers[k]["weight"]].astype(float),
                   weight=P[layers[k]["weight"]].astype(float),
                   bias=P[layers[k]["bias"]].astype(float),
                   bias_grad=G[layers[k]["bias"]].astype(float)))
        for k in range(len(layers))) + (
        Panel("Gradients, last layer first", "how many numbers have a gradient as "
              "loss.backward() reaches each layer", "bars",
              dict(values=np.array(running, float),
                   labels=[layers[k]["name"] for k in order])),)
    return Study("loss.backward()", "loss.backward() puts a gradient of its weight's shape "
                 "behind every weight, from the last layer back to the first", panels, meta)


STUDIES = {"blocks-forward-pass": forward_pass, "blocks-the-loss": the_loss,
           "blocks-backward": backward}

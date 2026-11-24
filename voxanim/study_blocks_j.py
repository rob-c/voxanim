"""The numbers behind animations 212-214 of the "Blocks and code" series (group ``j``, the
last three: the whole picture), all from the course's one run (:mod:`voxanim.study_course`)
and the script ``examples/mnist_blocks.py``.

* ``blocks-one-epoch-fast``: the 938 steps of the epoch, sped up. The run is replayed here
  step by step from :func:`~voxanim.study_course.init_params` over the same batches (it is
  deterministic: every loss and the final weights come out bit for bit as the recorded
  ``sgd`` run, which :func:`epoch_replay` checks), keeping the first layer's weights at
  :data:`KEEP` steps, more often early on, where they change fastest. Four units of the 64
  (the first four: not chosen) are drawn as 28 x 28 pictures; how alike next-door weights
  are (:func:`alike`) says, as one number, that they turn from noise into strokes. With it:
  the loss of every step and the held-out accuracy every 25 steps.
* ``blocks-the-whole-program``: the whole script, :func:`voxanim.blocks.program`, 63
  lines, split into its parts (:data:`PARTS`, its top-level regions in order) and into the
  windows of at most 8 lines the code panel scrolls through (:data:`WINDOWS`); what the
  script prints on the way, and the run's numbers.
* ``blocks-common-mistakes``: the four broken programs of ``examples/blocks_parts.py``, each
  with its real run: ``zero_grad`` forgotten (:func:`~voxanim.study_course.grad_norms`: the
  gradient used piles up), ``model.eval()`` forgotten with dropout (scored both ways), the
  softmax put in front of ``CrossEntropyLoss`` (its loss cannot go below ``ln(e + 9) - 1``,
  :func:`softmax_floor`) and no ``nn.Flatten`` (:data:`NO_FLATTEN_ERROR`, what torch
  raises, checked against torch by ``tests/test_study_blocks_j.py``).
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

import numpy as np

from . import study_course as course
from .blocks import code_lines, program, shape
from .cache import cached_npz
from .study import Panel, Study

VERSION = 1
SIDE = course.SIDE
UNITS = (0, 1, 2, 3)            # the first layer's units drawn: the first four, not picked
# the steps whose first-layer weights are kept (before that step; 938: after the epoch)
KEEP = tuple(range(0, 11)) + (12, 14, 16, 18, 20, 23, 26, 30, 35, 40, 45, 50, 60, 70, 80,
                              90, 100, 115, 130, 150, 175, 200) + \
    tuple(range(250, 938, 50)) + (course.EPOCH_STEPS,)
SOURCE = ("MNIST, as examples/mnist_blocks.py loads it; the course's one run "
          "(voxanim.study_course): SGD lr 0.1, batches of 64, one epoch")

# what torch raises when the model without nn.Flatten() (region mistake-no-flatten) is given
# a batch [64, 1, 28, 28]; checked against torch by tests/test_study_blocks_j.py
NO_FLATTEN_ERROR = "addmm: mat1 and mat2 shapes cannot be multiplied (1792x28 and 784x64)"
NO_FLATTEN_TYPE = "RuntimeError"

# the script's top-level regions, in order: together they are the whole program
TOP = ("imports", "device", "data", "one-digit", "loader", "one-batch", "model",
       "print-model", "loss", "optimizer", "train-mode", "train", "grad", "evaluate",
       "save", "load")
# the parts of the program, as the picture shows them: (name, its regions)
PARTS = (("imports", ("imports",)), ("device", ("device",)), ("data", ("data",)),
         ("batches", ("one-digit", "loader", "one-batch")), ("model", ("model",)),
         ("loss", ("print-model", "loss")), ("optimizer", ("optimizer", "train-mode")),
         ("train", ("train",)), ("grad", ("grad",)), ("test", ("evaluate",)),
         ("save", ("save", "load")))
# the code panel's windows, one per phase: (regions, picked lines of them or None: all)
WINDOWS = ((("device",), None), (("data",), None), (("one-digit", "loader", "one-batch"), None),
           (("model",), None), (("print-model", "loss", "optimizer", "train-mode"), None),
           (("train",), tuple(range(8))), (("evaluate",), None), (("save", "load"), None))
CORNERS = ((8, 16), (8, 8), (6, 8))       # the weight grids' corners drawn (rows, columns)
WEIGHTS = ("1.weight", "3.weight", "5.weight")
MASK_SEED = 214                 # the dropout mask drawn in blocks-common-mistakes


def pct(a, places=1):
    """An accuracy as a percentage, rounded half up: ``0.9435 -> "94.4%"``."""
    d = Decimal(repr(float(a))) * 100
    return "%s%%" % d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def alike(W):
    """How alike next-door weights are: for each unit's 28 x 28 picture of weights, the
    correlation of each weight with its right-hand and its lower neighbour (the mean of
    the two), averaged over the units. 0 for noise, 1 for a smooth picture."""
    W = np.asarray(W, np.float64).reshape(-1, SIDE, SIDE)
    out = []
    for w in W:
        rows = np.corrcoef(w[:, :-1].ravel(), w[:, 1:].ravel())[0, 1]
        cols = np.corrcoef(w[:-1, :].ravel(), w[1:, :].ravel())[0, 1]
        out.append((rows + cols) / 2)
    return float(np.mean(out))


def softmax_floor(classes=course.CLASSES):
    """The lowest loss ``CrossEntropyLoss`` can give when the model ends in a softmax: its
    inputs are then in 0..1, and the best is 1 for the right digit and 0 for the others,
    ``-log(e / (e + 9)) = ln(e + 9) - 1``."""
    return float(np.log(np.e + classes - 1) - 1)


# ====================================================================== 212
def _replay():
    P = course.init_params()
    W, A, L = [], [], []
    for step in range(course.EPOCH_STEPS):
        b = course.batch(step)
        if step in KEEP:
            W.append(P["1.weight"][list(UNITS)].copy())
            A.append(alike(P["1.weight"]))
        loss, G = course.loss_and_grads(P, b["x"], b["y"])
        L.append(loss)
        P = course.sgd_update(P, G)
    W.append(P["1.weight"][list(UNITS)].copy())
    A.append(alike(P["1.weight"]))
    return dict(steps=np.array(KEEP), units=np.stack(W), alike=np.array(A), loss=np.array(L),
                end=P["1.weight"])


def epoch_replay():
    """The epoch replayed: ``dict(steps, units, alike, loss, end)``: the :data:`KEEP` steps,
    the first layer's :data:`UNITS` weights before each (``(len(KEEP), 4, 784)``), how alike
    next-door weights are then (all 64 units), every step's loss, and the whole first layer
    at the end. Cached; checked against the recorded run."""
    settings = dict(version=VERSION, units=UNITS, keep=KEEP, course=course._settings())
    r = cached_npz("blocks-j-epoch", settings, _replay)
    if not np.array_equal(r["loss"], course.losses("sgd")):
        raise ValueError("the replayed epoch's losses are not the recorded run's")
    if not np.array_equal(r["end"], course.params(course.EPOCH_STEPS)["1.weight"]):
        raise ValueError("the replayed epoch does not end at the recorded weights")
    return r


def one_epoch_fast():
    r = epoch_replay()
    loss = course.losses("sgd")
    h = course.heldout("sgd")
    start = course.params(0)["1.weight"]
    printed = "%d %r" % (0, float(loss[-1]))          # print(epoch, loss.item())
    train = list(code_lines("train"))
    meta = dict(
        source=SOURCE, evidence="measured on MNIST: the course's run, replayed step by step",
        steps=course.EPOCH_STEPS, batch=course.BATCH, train=course.TRAIN,
        last_batch=course.LAST_BATCH, units=list(UNITS), layer_units=course.SIZES[1],
        keep=[int(s) for s in r["steps"]], weights=r["units"].tolist(),
        alike=[float(v) for v in r["alike"]], alike_start=float(r["alike"][0]),
        alike_end=float(r["alike"][-1]), alike_step_100=float(r["alike"][KEEP.index(100)]),
        loss=[float(v) for v in loss], first_loss=float(loss[0]), last_loss=float(loss[-1]),
        mean_last_100=float(loss[-100:].mean()), ln10=float(np.log(10)),
        eval_steps=[int(s) for s in h["steps"]], accuracy=[float(v) for v in h["accuracy"]],
        accuracy_start=float(h["accuracy"][0]), accuracy_end=float(h["accuracy"][-1]),
        heldout_loss_end=float(h["loss"][-1]), printed=printed,
        scale=float(np.percentile(np.abs(r["end"]), 95)),
        bound=float(1 / np.sqrt(course.SIZES[0])), code=train,
        limitations=[
            "the weights are kept at %d steps of the 938, more often early on; between two "
            "kept steps the picture shows the last one kept" % len(KEEP),
            "the shuffled order of the batches is the numpy twin's, not PyTorch's",
            "how alike next-door weights are is a correlation, averaged over the 64 units of "
            "the first layer: the four drawn are units 0-3"])
    panels = (
        Panel("The loss of every step", "loss.item() at each of the 938 steps of the epoch",
              "curve", dict(x=np.arange(course.EPOCH_STEPS, dtype=float), ys=loss[None],
                            labels=["training loss"])),
        Panel("Held-out accuracy", "share of the 10,000 test digits named right, every 25 "
              "steps", "curve", dict(x=np.asarray(h["steps"], float), ys=h["accuracy"][None],
                                     labels=["held-out accuracy"])),
        Panel("The first layer's units, before and after", "units 0-3 of nn.Linear(784, 64) "
              "as 28 x 28 pictures of weights: at step 0 and after the epoch", "images",
              dict(values=np.concatenate([start[list(UNITS)], r["units"][-1]]).reshape(
                  -1, SIDE, SIDE).astype(float),
                   labels=["unit %d, step %d" % (u, s) for s in (0, course.EPOCH_STEPS)
                           for u in UNITS])),
        Panel("Next-door weights alike", "correlation of next-door weights in the first "
              "layer's 64 pictures, at the kept steps", "curve",
              dict(x=np.asarray(r["steps"], float), ys=r["alike"][None],
                   labels=["next-door weights alike"])),
    )
    return Study("One epoch, fast", "938 steps of the course's run: the first layer's weights "
                 "turn from noise into strokes while the loss falls and the accuracy rises",
                 panels, meta)


# ====================================================================== 213
def program_parts():
    """The whole program and its parts: ``(lines, parts)``, ``parts`` a list of ``(name,
    first, last)`` line indices into ``lines`` (inclusive). Checks that the script's
    top-level regions, in order, are the whole program."""
    lines = program()
    joined = tuple(s for name in TOP for s in code_lines(name))
    if joined != lines:
        raise ValueError("the regions of %s are not the whole program" % "mnist_blocks.py")
    first, at = {}, 0
    for name in TOP:
        first[name] = at
        at += len(code_lines(name))
    parts = []
    for name, regs in PARTS:
        a = first[regs[0]]
        b = first[regs[-1]] + len(code_lines(regs[-1])) - 1
        parts.append((name, a, b))
    return lines, parts


def window_lines(k):
    """The lines of window ``k`` of :data:`WINDOWS`, and their indices in the program."""
    lines, _parts = program_parts()
    regs, pick = WINDOWS[k]
    shown = [s for name in regs for s in code_lines(name)]
    at = sum(len(code_lines(n)) for n in TOP[:TOP.index(regs[0])])
    idx = list(range(at, at + len(shown)))
    if pick is not None:
        shown, idx = [shown[i] for i in pick], [idx[i] for i in pick]
    if [lines[i] for i in idx] != shown:
        raise ValueError("window %d is not the program's lines" % k)
    return shown, idx


def whole_program():
    lines, parts = program_parts()
    loss = course.losses("sgd")
    h = course.heldout("sgd")
    acc = float(h["accuracy"][-1])
    right = int(round(acc * course.TEST))
    windows = [window_lines(k) for k in range(len(WINDOWS))]
    shown = sorted(i for _s, idx in windows for i in idx)
    b = course.first_batch()
    one, label = course.digit(0)
    P = course.params(course.EPOCH_STEPS)
    P0, G0 = course.init_params(), course.grads(0)
    meta = dict(
        source=SOURCE, evidence="exact: the script's own lines; measured: the course's run",
        program=list(lines), lines=len(lines), parts=[list(p) for p in parts],
        windows=[dict(lines=s, index=i) for s, i in windows], shown=shown,
        not_shown=[i for i in range(len(lines)) if i not in shown],
        widest=max(len(s) for s in lines), train=course.TRAIN, test=course.TEST,
        batch=course.BATCH, steps=course.EPOCH_STEPS, parameters=course.count(P),
        tensors={k: list(v.shape) for k, v in P.items()},
        printed_digit="%s %d" % ("torch.Size(%s)" % shape(one.shape), label),
        printed_batch="torch.Size(%s)\ntorch.Size(%s)" % (shape(b["x"].shape),
                                                         shape(b["y"].shape)),
        printed_epoch="%d %r" % (0, float(loss[-1])), printed_grad=course.PRINTED_GRAD,
        printed_accuracy=repr(right / course.TEST), correct=right, accuracy=acc,
        first_loss=float(loss[0]), last_loss=float(loss[-1]),
        device_line=[i for i, s in enumerate(lines) if "to(device)" in s],
        # the model's three weight grids, drawn as their top-left corners: as initialised,
        # and the gradient of the first step
        corners=[list(c) for c in CORNERS],
        model_tiles=[P0[k][:r, :c].tolist() for k, (r, c) in zip(WEIGHTS, CORNERS)],
        model_bounds=[float(1 / np.sqrt(P0[k].shape[1])) for k in WEIGHTS],
        grad_tiles=[G0[k][:r, :c].tolist() for k, (r, c) in zip(WEIGHTS, CORNERS)],
        limitations=[
            "the panel shows the program in windows of at most 8 lines; the imports (one "
            "line is 63 characters, wider than the panel) and the two lines that print the "
            "last loss and a gradient's shape are on the program's map only",
            "the device is the GPU when there is one, else the CPU; the course's run is the "
            "numpy twin's, on a CPU"])
    sizes = [b_ - a + 1 for _n, a, b_ in parts]
    panels = (
        Panel("The program's parts", "lines of code in each part of examples/mnist_blocks.py",
              "bars", dict(values=np.array(sizes, float), labels=[n for n, _a, _b in parts])),
        Panel("The loss while it trains", "loss.item() at each of the 938 steps", "curve",
              dict(x=np.arange(course.EPOCH_STEPS, dtype=float), ys=loss[None],
                   labels=["training loss"])),
        Panel("What the model learns", "numbers in each tensor of model.state_dict()", "bars",
              dict(values=np.array([v.size for v in P.values()], float), labels=list(P))),
        Panel("The test", "held-out accuracy before and after the epoch", "bars",
              dict(values=np.array([h["accuracy"][0], acc]), labels=["step 0", "step 938"])),
    )
    return Study("The whole program", "The whole training script, part by part, beside the "
                 "picture of what each part makes", panels, meta)


# ====================================================================== 214
def dropout_mask(n=course.SIZES[1], p=course.DROPOUT, seed=MASK_SEED):
    """One random dropout mask over ``n`` hidden numbers: True where a number is switched
    off (each with chance ``p``), as ``nn.Dropout(p)`` does in training mode."""
    return np.random.default_rng(seed).random(n) < p


def common_mistakes():
    sgd = course.heldout("sgd")
    right = float(sgd["accuracy"][-1])
    ng = course.heldout("no_zero_grad")
    norms = course.grad_norms("no_zero_grad")
    do = course.heldout("dropout")
    ds = course.heldout("double_softmax")
    ds_loss = course.losses("double_softmax")
    loss = course.losses("sgd")
    mask = dropout_mask()
    floor = softmax_floor()
    at = 100
    # without Flatten, Linear reads the last dimension (28) as the features: every other
    # dimension becomes a row
    xb = course.first_batch()["x"].shape
    rows = int(np.prod(xb[:-1]))
    error = "%s: %s" % (NO_FLATTEN_TYPE, NO_FLATTEN_ERROR)
    meta = dict(
        source=SOURCE, evidence="measured on MNIST: each broken program run for one epoch "
        "from the same weights over the same batches",
        right=right, right_text=pct(right),
        no_zero_grad=dict(
            accuracy=float(ng["accuracy"][-1]), text=pct(ng["accuracy"][-1]),
            best=float(ng["accuracy"].max()), best_step=int(ng["steps"][int(ng["accuracy"]
                                                                            .argmax())]),
            used=[float(v) for v in norms["used"]], fresh=[float(v) for v in norms["fresh"]],
            at=at, used_at=float(norms["used"][at]), fresh_at=float(norms["fresh"][at]),
            times_at=float(norms["used"][at] / norms["fresh"][at]),
            loss=[float(v) for v in course.losses("no_zero_grad")],
            last_100=float(course.losses("no_zero_grad")[-100:].mean()),
            code=list(code_lines("mistake-no-zero-grad"))),
        no_eval=dict(
            accuracy=float(do["accuracy_train_mode"][-1]), text=pct(do["accuracy_train_mode"][-1]),
            with_eval=float(do["accuracy"][-1]), with_eval_text=pct(do["accuracy"][-1]),
            mask=[bool(v) for v in mask], off=int(mask.sum()), p=course.DROPOUT,
            code=list(code_lines("mistake-no-eval")), model=list(code_lines("dropout-model"))),
        double_softmax=dict(
            accuracy=float(ds["accuracy"][-1]), text=pct(ds["accuracy"][-1]), floor=floor,
            floor_text="%.2f" % floor, lowest=float(ds_loss.min()),
            last_100=float(ds_loss[-100:].mean()), right_last_100=float(loss[-100:].mean()),
            right_lowest=float(loss.min()), loss=[float(v) for v in ds_loss],
            first=float(ds_loss[0]), code=list(code_lines("mistake-double-softmax"))),
        no_flatten=dict(
            error=error, message=NO_FLATTEN_ERROR, kind=NO_FLATTEN_TYPE, batch=list(xb),
            batch_text=shape(xb), rows=rows, rows_sum="%s = %s" % (
                " × ".join(str(d) for d in xb[:-1]), format(rows, ",")),
            got=int(xb[-1]), wants=course.SIZES[0], weight=[course.SIZES[0], course.SIZES[1]],
            code=list(code_lines("mistake-no-flatten"))),
        loss=[float(v) for v in loss],
        limitations=[
            "the dropout mask drawn is one random draw (1 in 5 off), not the one PyTorch "
            "would draw",
            "the error message is torch %s's; other versions may word it differently"
            % "2.15",
            "the no-zero_grad run keeps every step's gradient in .grad, as PyTorch does when "
            "optimizer.zero_grad() is left out"])
    steps = np.arange(course.EPOCH_STEPS, dtype=float)
    panels = (
        Panel("The loss, right and wrong", "loss.item() at each step: the script, zero_grad "
              "forgotten, and softmax before the loss", "curve",
              dict(x=steps, ys=np.stack([loss, course.losses("no_zero_grad"), ds_loss]),
                   labels=["the script", "no zero_grad", "double softmax"])),
        Panel("Held-out accuracy after one epoch", "share of the 10,000 test digits named "
              "right", "bars",
              dict(values=np.array([right, ng["accuracy"][-1], do["accuracy"][-1],
                                    do["accuracy_train_mode"][-1], ds["accuracy"][-1]]),
                   labels=["the script", "no zero_grad", "dropout, eval()",
                           "dropout, no eval()", "double softmax"])),
        Panel("The gradient used, without zero_grad", "length of the gradient the optimizer "
              "used against the batch's own, each step", "curve",
              dict(x=steps, ys=np.stack([norms["used"], norms["fresh"]]),
                   labels=[".grad, never emptied", "this batch's gradient"])),
        Panel("Dropout, tested two ways", "held-out accuracy of the dropout model every 25 "
              "steps, with model.eval() and without", "curve",
              dict(x=np.asarray(do["steps"], float),
                   ys=np.stack([do["accuracy"], do["accuracy_train_mode"]]),
                   labels=["model.eval()", "still training mode"])),
    )
    return Study("Common mistakes", "Four broken programs: what each mistake does, measured",
                 panels, meta)


STUDIES = {"blocks-one-epoch-fast": one_epoch_fast, "blocks-the-whole-program": whole_program,
           "blocks-common-mistakes": common_mistakes}

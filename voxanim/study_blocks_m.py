"""The numbers behind animations 221-223 of "Blocks and code" (group ``m``, the last three of
the CNN part): the CNN of ``examples/mnist_cnn.py`` against the course's MLP, what it gets
wrong, and three broken CNN programs. Everything comes from :mod:`voxanim.study_course_cnn`
(the CNN's one run) and :mod:`voxanim.study_course` (the MLP's), both one epoch of SGD at lr
0.1 on the same batches in the same order, from one seed each.

* ``blocks-cnn-against-the-mlp``: :func:`~voxanim.study_course_cnn.compare`: 5,258
  numbers against 52,650, drawn as two stacks of tiles of :data:`PER_TILE` numbers each, to
  scale; the held-out accuracy every 25 steps of both runs (the same steps); the digits each
  misses, per digit. One run of each: nothing here says how much another seed would move.
* ``blocks-cnn-what-it-gets-wrong``: the 364 test digits the CNN misses, per digit and per
  confusion (:func:`confusions`), with real examples: for each of the three most common
  confusions, the first :data:`SHOWN` such digits in test-set order (not picked), what the
  CNN said and how sure it was (its softmax), and what the MLP said. :data:`LOOKS_LIKE` is
  the one example that, looked at by eye, a person might read the CNN's way: a judgement,
  not a measurement, and said only of the examples that were looked at.
* ``blocks-cnn-common-mistakes``: the three broken programs of ``examples/blocks_parts.py``
  and torch's own errors (:data:`~voxanim.study_course_cnn.ERRORS`), each error wrapped for
  the code panel (:func:`wrap_error`); the shapes that explain them, worked out here by
  :func:`walk` (PyTorch's rules for a 3 x 3 convolution without padding, a 2 x 2 pooling
  and ``Flatten(start_dim=1)``; ``tests/test_study_blocks_m.py`` checks them against
  torch), and the fix's shape, :data:`~voxanim.study_course_cnn.FIXED_SHAPES`.
"""

from __future__ import annotations

import textwrap

import numpy as np

from . import study_course as course
from . import study_course_cnn as cnn
from .blocks import code_lines, commas, shape
from .study import Panel, Study
from .study_blocks_j import pct

PER_TILE = 250                  # numbers per tile in the two stacks
STACK_COLS = 10                 # tiles per row of a stack
SHOWN = 4                       # examples drawn of each confusion
TOP = 3                         # confusions drawn
# Looked at by eye (the author of this module, at 8x): of the examples drawn, test digit
# 947, an 8 the CNN calls a 9 (softmax 0.99), has its lower loop filled in and reads as a
# 9. The other eleven drawn read plainly as their labels. A judgement, not a measurement.
LOOKS_LIKE = {947: 9}
WIDTH = 39                      # characters of an error line in the output slot ("→ " first)
SOURCE = ("MNIST; one run each of examples/mnist_cnn.py (voxanim.study_course_cnn) and "
          "examples/mnist_blocks.py (voxanim.study_course): SGD lr 0.1, batches of 64, one "
          "epoch, the same batches in the same order")
ONE_RUN = ("one run of each model from one seed: another seed would move these numbers a "
           "little, by an amount not measured here")


def printed_accuracy(right, total=course.TEST):
    """What ``print(int(correct) / len(test_set))`` prints."""
    return repr(int(right) / int(total))


# ====================================================================== 221
def stack_tiles(n, per=PER_TILE):
    """``n`` numbers as tiles of ``per``: ``(whole tiles, the last one's fraction)``."""
    whole, rest = divmod(int(n), per)
    return int(whole), rest / per


def first_at_least(steps, acc, target):
    """The first step whose accuracy reaches ``target`` (None if none)."""
    hit = np.flatnonzero(np.asarray(acc) >= target)
    return int(np.asarray(steps)[hit[0]]) if len(hit) else None


def against_the_mlp():
    c = cnn.compare()
    hc, hm = cnn.heldout(), course.heldout("sgd")
    if not np.array_equal(hc["steps"], hm["steps"]):
        raise ValueError("the two held-out curves are not on the same steps")
    counts = cnn.counts()
    P = course.params(0)
    mlp_counts = [int(P["%d.weight" % i].size + P["%d.bias" % i].size) for i in course.LINEAR]
    cw, cf = stack_tiles(c["cnn_params"])
    mw, mf = stack_tiles(c["mlp_params"])
    yt = course.data()[3]
    right_cnn = course.TEST - c["cnn_missed"]
    right_mlp = course.TEST - c["mlp_missed"]
    meta = dict(
        source=SOURCE, evidence="exact: the numbers counted; measured: 10,000 held-out digits",
        runs="one run each", per_tile=PER_TILE, stack_cols=STACK_COLS,
        cnn_params=c["cnn_params"], mlp_params=c["mlp_params"],
        cnn_params_text=commas(c["cnn_params"]), mlp_params_text=commas(c["mlp_params"]),
        cnn_counts=[v["total"] for v in counts.values()], mlp_counts=mlp_counts,
        cnn_tiles=[cw, cf], mlp_tiles=[mw, mf],
        ratio=c["mlp_params"] / c["cnn_params"],
        ratio_text="%d" % round(c["mlp_params"] / c["cnn_params"]),
        cnn_accuracy=c["cnn_accuracy"], mlp_accuracy=c["mlp_accuracy"],
        cnn_accuracy_text=pct(c["cnn_accuracy"]), mlp_accuracy_text=pct(c["mlp_accuracy"]),
        cnn_missed=c["cnn_missed"], mlp_missed=c["mlp_missed"],
        cnn_right=right_cnn, mlp_right=right_mlp,
        printed_cnn=printed_accuracy(right_cnn), printed_mlp=printed_accuracy(right_mlp),
        both_right=c["both_right"], only_cnn=c["only_cnn"], only_mlp=c["only_mlp"],
        both_wrong=c["both_wrong"],
        per_digit_cnn=c["per_digit_cnn"], per_digit_mlp=c["per_digit_mlp"],
        per_digit_total=[int((yt == d).sum()) for d in range(course.CLASSES)],
        eval_steps=[int(s) for s in hc["steps"]],
        accuracy_cnn=[float(v) for v in hc["accuracy"]],
        accuracy_mlp=[float(v) for v in hm["accuracy"]],
        start_cnn=float(hc["accuracy"][0]), start_mlp=float(hm["accuracy"][0]),
        cnn_passes_mlp_end=first_at_least(hc["steps"], hc["accuracy"], c["mlp_accuracy"]),
        ahead_at=[int(s) for s, a, b in zip(hc["steps"], hc["accuracy"], hm["accuracy"])
                  if a > b],
        steps=course.EPOCH_STEPS, test=course.TEST,
        code_mlp=list(code_lines("model")), code_cnn=list(code_lines("cnn-model")),
        limitations=[
            ONE_RUN,
            "both run for one epoch only; more epochs would change both",
            "the script tests once, after the epoch; the held-out accuracy every 25 steps is "
            "measured by the study on the same 10,000 test digits",
            "a tile stands for %d numbers; the last tile of each stack is drawn as the "
            "fraction it holds" % PER_TILE])
    steps = np.asarray(hc["steps"], float)
    panels = (
        Panel("Numbers to learn", "sum(p.numel() for p in model.parameters()), each model",
              "bars", dict(values=np.array([c["mlp_params"], c["cnn_params"]], float),
                           labels=["MLP", "CNN"])),
        Panel("Held-out accuracy while it trains", "share of the 10,000 test digits named "
              "right, every 25 steps of one run each", "curve",
              dict(x=steps, ys=np.stack([hm["accuracy"], hc["accuracy"]]),
                   labels=["MLP", "CNN"])),
        Panel("Digits missed, by digit", "test digits of each label named wrong after one "
              "epoch (rows: MLP, CNN)", "matrix",
              dict(values=np.array([c["per_digit_mlp"], c["per_digit_cnn"]], float),
                   rows=["MLP", "CNN"], cols=[str(d) for d in range(course.CLASSES)])),
        Panel("Who gets which right", "the 10,000 test digits: both right, only the CNN, "
              "only the MLP, both wrong", "bars",
              dict(values=np.array([c["both_right"], c["only_cnn"], c["only_mlp"],
                                    c["both_wrong"]], float),
                   labels=["both right", "only the CNN", "only the MLP", "both wrong"])),
    )
    return Study("The CNN against the MLP", "Ten times fewer numbers, fewer digits missed: "
                 "one run of each, measured on the same 10,000 test digits", panels, meta)


# ====================================================================== 222
def misses():
    """The test digits the CNN gets wrong: ``(indices, true labels, what it said)``."""
    yt = course.data()[3]
    p = cnn.predictions()
    idx = np.flatnonzero(p != yt)
    return idx, yt[idx], p[idx]


def confusions(top=None):
    """``[(true, said, how many), ...]``, most common first (ties: by true, then said)."""
    c = cnn.compare()["confusions"]
    return c if top is None else c[:top]


def examples(true, said, n=SHOWN):
    """The first ``n`` test digits (in test-set order: not picked) with label ``true`` that
    the CNN calls ``said``: ``dict(index, image [n, 28, 28], sure, sure_true, mlp)``, the
    CNN's softmax for ``said`` and for ``true``, and the MLP's answer."""
    _x, _y, xt, yt = course.data()
    p = cnn.predictions()
    idx = np.flatnonzero((yt == true) & (p == said))[:n]
    a = cnn.forward(cnn.params(), xt[idx])
    probs = course._softmax(a["logits"])
    if not np.array_equal(probs.argmax(1), p[idx]):
        raise ValueError("the recomputed answers are not the recorded predictions")
    return dict(index=[int(i) for i in idx], image=xt[idx].reshape(-1, course.SIDE, course.SIDE),
                sure=[float(v) for v in probs[:, said]],
                sure_true=[float(v) for v in probs[:, true]],
                mlp=[int(v) for v in cnn.mlp_predictions()[idx]])


def what_it_gets_wrong():
    idx, true, said = misses()
    yt = course.data()[3]
    xt = course.data()[2]
    per_digit = [int((true == d).sum()) for d in range(course.CLASSES)]
    groups = []
    for t, s, n in confusions(TOP):
        e = examples(t, s)
        groups.append(dict(true=int(t), said=int(s), count=int(n), index=e["index"],
                           image=e["image"].tolist(), sure=e["sure"], sure_true=e["sure_true"],
                           mlp=e["mlp"],
                           looks_like=[LOOKS_LIKE.get(i) for i in e["index"]],
                           of_digit=per_digit[int(t)]))
    first = idx[:SHOWN]
    matrix = np.zeros((course.CLASSES, course.CLASSES))
    for t, s in zip(true, said):
        matrix[t, s] += 1
    c = cnn.compare()
    meta = dict(
        source=SOURCE, evidence="measured on MNIST: the CNN's answers on the 10,000 test digits",
        missed=int(len(idx)), missed_text=commas(len(idx)), test=course.TEST,
        accuracy=c["cnn_accuracy"], accuracy_text=pct(c["cnn_accuracy"]),
        printed=printed_accuracy(course.TEST - len(idx)),
        per_digit=per_digit, per_digit_total=[int((yt == d).sum()) for d in range(10)],
        worst=int(np.argmax(per_digit)), worst_count=int(max(per_digit)),
        per_digit_mlp=c["per_digit_mlp"],
        confusions=[list(map(int, x)) for x in confusions()], groups=groups,
        top_total=int(sum(g["count"] for g in groups)),
        first=dict(index=[int(i) for i in first], true=[int(v) for v in true[:SHOWN]],
                   said=[int(v) for v in said[:SHOWN]],
                   image=xt[first].reshape(-1, course.SIDE, course.SIDE).tolist()),
        looks_like={str(k): v for k, v in LOOKS_LIKE.items()},
        code=list(code_lines("cnn-evaluate")),
        limitations=[
            ONE_RUN,
            "the examples are the first of each kind in test-set order, not picked",
            "that test digit 947 reads as a 9 is a judgement made by eye, not a measurement; "
            "the other examples drawn read plainly as their labels"])
    panels = (
        Panel("Missed, by digit", "test digits of each label the CNN names wrong", "bars",
              dict(values=np.array(per_digit, float), labels=[str(d) for d in range(10)])),
        Panel("Missed, by confusion", "the most common (label, what the CNN said) pairs",
              "bars", dict(values=np.array([n for _t, _s, n in confusions(10)], float),
                           labels=["%d → %d" % (t, s) for t, s, _n in confusions(10)])),
        Panel("All its mistakes", "rows: the label; columns: what the CNN said (the "
              "diagonal, right answers, left out)", "matrix", dict(values=matrix)),
        Panel("Examples", "the first %d test digits of each of the %d most common confusions"
              % (SHOWN, TOP), "images",
              dict(values=np.concatenate([np.asarray(g["image"]) for g in groups]),
                   labels=["%d, said %d" % (g["true"], g["said"]) for g in groups
                           for _i in g["index"]])),
    )
    return Study("What the CNN gets wrong", "The 364 test digits the CNN misses, by digit "
                 "and by confusion, with real examples", panels, meta)


# ====================================================================== 223
def wrap_error(text, width=WIDTH):
    """An error as lines of at most ``width`` characters: a ``(...)`` at the end is kept
    whole on a line of its own (the numbers that matter), the rest wrapped at spaces, never
    inside a shape ``[...]``. The lines joined by spaces are the error again."""
    head, tail = text, ""
    if text.endswith(")") and " (" in text:
        cut = text.rindex(" (")
        head, tail = text[:cut], text[cut + 1:]
    keep, depth = [], 0
    for ch in head:                       # spaces inside [...] do not break a line
        depth += (ch == "[") - (ch == "]")
        keep.append("\0" if ch == " " and depth > 0 else ch)
    lines = [s.replace("\0", " ") for s in textwrap.wrap(
        "".join(keep), width, break_long_words=False, break_on_hyphens=False)]
    if tail:
        lines.append(tail)
    if max(len(s) for s in lines) > width:
        raise ValueError("an error line is wider than %d characters" % width)
    return lines


def walk(dims):
    """``dims`` through the CNN's blocks by PyTorch's rules (a 3 x 3 convolution without
    padding: the side loses 2; a 2 x 2 pooling halves it, rounding down; ``Flatten()``
    keeps the first dimension and multiplies the rest): ``[(block, shape), ...]`` up to the
    input of ``nn.Linear``. A 3-dimensional ``dims`` is one digit without its batch, which
    a convolution takes as ``[channels, height, width]``."""
    dims = tuple(int(d) for d in dims)
    out = [("input", dims)]
    lead, (c, h, w) = dims[:-3], dims[-3:]
    for block in cnn.BLOCKS[:-1]:
        if block == "Conv2d":
            c = cnn.CHANNELS[cnn.CHANNELS.index(c) + 1]
            h, w = h - (cnn.KERNEL - 1), w - (cnn.KERNEL - 1)
        elif block == "MaxPool2d":
            h, w = h // cnn.POOL, w // cnn.POOL
        if block == "Flatten":
            full = lead + (c, h, w)
            out.append((block, (full[0], int(np.prod(full[1:])))))
        else:
            out.append((block, lead + (c, h, w)))
    return out


def common_mistakes():
    E = cnn.ERRORS
    batch = (course.BATCH, 1, course.SIDE, course.SIDE)
    one, fixed = batch[1:], (1,) + batch[1:]
    right = walk(batch)
    nobatch = walk(one)
    fix = walk(fixed)
    flat = right[-1][1]
    wrong_in = 784                  # nn.Linear(784, 10) in region cnn-mistake-in-features
    errs = {k: "RuntimeError: " + v for k, v in E.items()}
    wrapped = {k: wrap_error(v) for k, v in errs.items()}
    nb = nobatch[-1][1]
    checks = {
        "cnn-mistake-in-features": "(%dx%d and %dx%d)" % (flat + (wrong_in, cnn.CLASSES)),
        "cnn-mistake-channels": "weight of size %s, expected input%s to have 3 channels, but "
                                "got 1 channels" % (shape((8, 3, 3, 3)), shape(batch)),
        "cnn-mistake-no-batch": "(%dx%d and %dx%d)" % (nb + (cnn.IN_FEATURES, cnn.CLASSES)),
    }
    for k, s in checks.items():
        if s not in E[k]:
            raise ValueError("%s: the shapes worked out here are not torch's: %r" % (k, s))
    if (fix[0][1][0], cnn.CLASSES) != cnn.FIXED_SHAPES["cnn-fix-no-batch"]:
        raise ValueError("the fix's shape is not torch's")
    digit = course.first_batch()["x"][0, 0]
    meta = dict(
        source=SOURCE, evidence="exact arithmetic: the shapes, and torch's own errors",
        errors=errs, wrapped=wrapped, width=WIDTH,
        batch=list(batch), batch_text=shape(batch),
        in_features=dict(arrive=cnn.IN_FEATURES, wants=wrong_in, right=cnn.IN_FEATURES,
                         flat=list(flat), flat_text=shape(flat),
                         sum="%d × %d × %d = %d" % (cnn.CHANNELS[-1], cnn.side_walk()[-1],
                                                     cnn.side_walk()[-1], cnn.IN_FEATURES),
                         code=list(code_lines("cnn-mistake-in-features"))),
        channels=dict(wants=3, got=1, weight=[8, 3, 3, 3], weight_text=shape((8, 3, 3, 3)),
                      input_text=shape(batch), code=list(code_lines("cnn-mistake-channels"))),
        no_batch=dict(one=list(one), one_text=shape(one),
                      walk=[[b, list(s)] for b, s in nobatch], flat=list(nb),
                      flat_text=shape(nb), rows=nb[0], row=nb[1], wants=cnn.IN_FEATURES,
                      code=list(code_lines("cnn-mistake-no-batch"))),
        fix=dict(one=list(fixed), one_text=shape(fixed),
                 walk=[[b, list(s)] for b, s in fix], flat=list(fix[-1][1]),
                 flat_text=shape(fix[-1][1]),
                 out=list(cnn.FIXED_SHAPES["cnn-fix-no-batch"]),
                 out_text=shape(cnn.FIXED_SHAPES["cnn-fix-no-batch"]),
                 printed="torch.Size(%s)" % shape(cnn.FIXED_SHAPES["cnn-fix-no-batch"]),
                 code=list(code_lines("cnn-fix-no-batch"))),
        right_walk=[[b, list(s)] for b, s in right],
        digit=digit.tolist(),
        limitations=[
            "examples/blocks_parts.py runs the broken programs on a made-up batch shaped like "
            "64 digits; the digit drawn is the course's first training digit: the errors "
            "depend on the shapes only",
            "the error messages are torch 2.15's; other versions may word them differently",
            "the output slot wraps each error over lines; the words are torch's, unchanged"])
    panels = (
        Panel("in_features", "numbers per digit nn.Linear(784, 10) wants, and what arrives "
              "after Flatten", "bars", dict(values=np.array([wrong_in, cnn.IN_FEATURES], float),
                                            labels=["wanted", "arrive"])),
        Panel("Channels", "channels nn.Conv2d(3, 8, 3) wants, and a grey digit's", "bars",
              dict(values=np.array([3.0, 1.0]), labels=["wanted", "a grey digit"])),
        Panel("Rows into the Linear", "rows Flatten makes from one digit: x[0] (no batch) "
              "and x[0:1] (a batch of one)", "bars",
              dict(values=np.array([nb[0], fix[-1][1][0]], float),
                   labels=["x[0]", "x[0:1]"])),
        Panel("The digit", "the first digit of the course's first batch, normalised",
              "images", dict(values=np.asarray(digit, float)[None], labels=["x[0]"])),
    )
    return Study("CNN mistakes", "Three broken CNN programs, torch's own errors, and the fix",
                 panels, meta)


STUDIES = {"blocks-cnn-against-the-mlp": against_the_mlp,
           "blocks-cnn-what-it-gets-wrong": what_it_gets_wrong,
           "blocks-cnn-common-mistakes": common_mistakes}

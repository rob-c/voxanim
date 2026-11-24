"""The numbers behind animations 218-220 of the "Blocks and code" series (group ``l``, the
CNN part): ``loss.backward()`` on the CNN's shared stamps, the training loop the CNN and
the MLP share, and the first layer's stamps as they learn. Everything comes from
:mod:`voxanim.study_course_cnn` (the CNN of ``examples/mnist_cnn.py``, trained once in numpy
on :mod:`voxanim.study_course`'s batches) and, for the MLP, :mod:`voxanim.study_course`.

* ``blocks-cnn-backward``: at step 0 (the first batch, the untrained CNN),
  ``model[0].weight.grad`` has the stamps' shape ``[8, 1, 3, 3]``. One of its 72 numbers
  (stamp 1, its centre weight) is followed as the sum of its parts
  (:func:`~voxanim.study_course_cnn.stamp_grad_parts`): the stamp sat at 26 x 26 places on
  each of the 64 digits, and each place adds (the pixel under that weight) x (the gradient
  of the map number the stamp made there, its "red"). Three places of one digit are worked
  out, then that digit's 676 parts, then the 64 digits' sums, adding up to PyTorch's number.
* ``blocks-cnn-same-loop``: the two programs' ``model`` regions; the step's six lines are
  the same in both files (checked), as are all but a few lines of the programs (counted with
  :mod:`difflib`); both runs' losses on the same 938 batches, each smoothed the same way:
  the mean of each 25 steps (:func:`chunk_means`).
* ``blocks-cnn-stamps-learn``: ``params(step)["0.weight"]`` at steps 0, 10, 100, 500 and
  938; how far they have moved from the start; and the 8 maps (the first ``nn.Conv2d``'s
  output, before ReLU) they make of training digit 0, a 5, before and after the epoch.

Numbers computed here and not in ``study_course_cnn``: the per-digit sums of the parts, the
three worked places, the count of lines the two programs share, the 25-step means, the
stamps' mean move, and the maps of one digit (re-computed by a direct loop as a check).
"""

from __future__ import annotations

import difflib
import re

import numpy as np

from . import study_course as course
from . import study_course_cnn as cnn
from .blocks import (
    CNN_SCRIPT,
    SCRIPT,
    code_line,
    code_lines,
    commas,
    program,
    regions,
    shape,
    val,
)
from .study import Panel, Study

STEP = 0                       # 218: the first batch, the untrained CNN
STAMP, ROW, COL = 0, 1, 1      # stamp 1 (index 0), its centre weight
DIGIT = 4                      # 218: the batch's digit whose parts are worked out
CHUNK = 25                     # 219: the loss curves, the mean of each 25 steps
LAST = 100                     # 219: the steps the final means are taken over
STAMP_STEPS = (0, 10, 100, 500, cnn.EPOCH_STEPS)     # 220
TRAIN_DIGIT = 0                # 220: train_set[0], a 5 (what region cnn-one-digit prints)
SOURCE = ("MNIST, trained as examples/mnist_cnn.py trains it (voxanim.study_course_cnn: "
          "one epoch of SGD, lr=0.1, on voxanim.study_course's batches)")


def five(v):
    """A small number with 5 decimals and a real minus: ``0.00034``, ``−0.00021``."""
    return val(v, 5)


def chunk_means(values, size=CHUNK):
    """``(x, means)``: the mean of each ``size`` steps of ``values`` (the last chunk may be
    shorter), at ``x`` = the middle of its steps."""
    v = np.asarray(values, float)
    ends = np.r_[np.arange(size, len(v), size), len(v)]
    starts = np.r_[0, ends[:-1]]
    return (starts + ends) / 2.0, np.array([v[a:b].mean() for a, b in zip(starts, ends)])


def block_names(region):
    """The blocks of a ``model`` region in order: ``nn.Conv2d(1, 8, 3), nn.ReLU(),`` ->
    ``Conv2d(1, 8, 3)``, ``ReLU()``."""
    out = []
    for line in code_lines(region)[1:-1]:
        out += re.findall(r"nn\.(\w+\((?:[^()]*)\))", line)
    return out


# ====================================================================== 218
def cnn_backward():
    P = cnn.params(STEP)
    b = course.batch(STEP)
    loss, G, dz1 = cnn.loss_and_grads(P, b["x"], b["y"], keep=True)
    recorded = cnn.grads(STEP)
    for k in cnn.KEYS:
        if G[k].shape != P[k].shape or not np.allclose(G[k], recorded[k], atol=1e-7):
            raise ValueError("%s: .grad is not loss.backward()'s, or not its weight's shape" % k)
    got = cnn.stamp_grad_parts(STEP, STAMP, ROW, COL)
    parts = got["parts"]                                       # [64, 26, 26]
    side = parts.shape[1]
    x = np.asarray(b["x"], np.float64)[:, 0]
    red = dz1[:, STAMP].astype(np.float64)                     # [64, 26, 26]
    pixels = x[:, ROW:ROW + side, COL:COL + side]
    if not np.allclose(parts, pixels * red, rtol=1e-6, atol=1e-12):
        raise ValueError("a part is not pixel x red")
    grad = float(G["0.weight"][STAMP, 0, ROW, COL])
    if abs(got["total"] - grad) > 1e-6 or got["grad"] != grad:
        raise ValueError("the parts do not add up to PyTorch's number")
    sums = parts.reshape(len(parts), -1).sum(1)
    if abs(sums.sum() - got["total"]) > 1e-12:
        raise ValueError("the digits' sums do not add up to the total")
    p = parts[DIGIT]
    # three places of the digit, worked out: the biggest positive part, the most negative,
    # and the place on the stroke (its brightest pixel) where the red is 0
    pos = np.unravel_index(int(np.argmax(p)), p.shape)
    neg = np.unravel_index(int(np.argmin(p)), p.shape)
    zeros = np.where(red[DIGIT] == 0, pixels[DIGIT], -np.inf)
    nil = np.unravel_index(int(np.argmax(zeros)), p.shape)
    places = [tuple(int(v) for v in rc) for rc in (pos, neg, nil)]
    worked = [dict(place=list(rc), pixel=float(pixels[DIGIT][rc]), red=float(red[DIGIT][rc]),
                   part=float(p[rc]),
                   text=("%s × 0 = 0" % val(pixels[DIGIT][rc]) if red[DIGIT][rc] == 0 else
                         "%s × %s = %s" % (val(pixels[DIGIT][rc]), five(red[DIGIT][rc]),
                                           five(p[rc]))))
              for rc in places]
    # the order the places are added in: the three worked ones, then the rest row by row
    order = places + [(r, c) for r in range(side) for c in range(side)
                      if (r, c) not in places]
    running = np.cumsum([p[rc] for rc in order])
    n_parts = int(parts.size)
    count = cnn.count(P)
    w = P["0.weight"]
    meta = dict(
        source=SOURCE, evidence="measured on MNIST: loss.backward() on the first batch",
        step=STEP, loss=float(loss), loss_text="%.3f" % loss, count=count,
        count_text=commas(count), weight_shape=list(w.shape), shape_text=shape(w.shape),
        weights=int(w.size), stamps=int(w.shape[0]), printed=cnn.PRINTED_GRAD,
        stamp=STAMP, row=ROW, col=COL, digit=DIGIT, label=int(b["y"][DIGIT]),
        batch=int(len(parts)), side=int(side), places=int(side * side),
        n_parts=n_parts, n_parts_text=commas(n_parts),
        parts_sum="%d × %d = %s" % (len(parts), side * side, commas(n_parts)),
        nonzero=int((parts != 0).sum()), digit_nonzero=int((p != 0).sum()),
        worked=worked, order=[list(rc) for rc in order], running=running.tolist(),
        digit_sum=float(sums[DIGIT]), digit_sum_text=five(sums[DIGIT]),
        sums=sums.tolist(), positive=float(parts[parts > 0].sum()),
        negative=float(parts[parts < 0].sum()), total=float(got["total"]),
        total_text=five(got["total"]), grad=grad, grad_text=five(grad),
        call=code_line("cnn-step", "backward"), step_code=list(code_lines("cnn-step")),
        grad_code=list(code_lines("cnn-grad")),
        limitations=[
            "the parts are added in float64; PyTorch's .grad is float32 and agrees to %.0e"
            % abs(got["total"] - grad),
            "red tiles are shaded from the most negative (lightest) to the most positive "
            "(deepest); a grey tile is a 0 (pooling or ReLU passed no gradient there)",
            "the stamp's other 71 weights and its bias get their gradients the same way"])
    panels = (
        Panel("The digit", "digit %d of the first batch (a %d), as the stamp sees it"
              % (DIGIT + 1, int(b["y"][DIGIT])), "images",
              dict(values=x[DIGIT][None], labels=["%d" % int(b["y"][DIGIT])],
                   weight=w.astype(float), grad=G["0.weight"].astype(float))),
        Panel("Its map's gradient", "the gradient of the 26 x 26 numbers stamp %d made of "
              "this digit (its red)" % (STAMP + 1), "matrix", dict(values=red[DIGIT])),
        Panel("Its parts", "pixel under the centre weight x red, at each of the 676 places",
              "matrix", dict(values=p)),
        Panel("One sum per digit", "each digit's 676 parts added; the 64 add up to "
              "model[0].weight.grad[%d, 0, %d, %d]" % (STAMP, ROW, COL), "bars",
              dict(values=sums, labels=["%d" % v for v in b["y"]])),
    )
    return Study("A shared stamp's gradient", "one number of the stamps' .grad is the sum of "
                 "64 x 676 parts, one per place the stamp sat", panels, meta)


# ====================================================================== 219
def same_loop():
    if code_lines("step") != code_lines("cnn-step"):
        raise ValueError("the step's lines are not the same in the two programs")
    mlp_prog, cnn_prog = program(SCRIPT), program(CNN_SCRIPT)
    sm = difflib.SequenceMatcher(None, mlp_prog, cnn_prog, autojunk=False)
    same = sum(blk.size for blk in sm.get_matching_blocks())
    changed = [(mlp_prog[i1:i2], cnn_prog[j1:j2]) for tag, i1, i2, j1, j2 in sm.get_opcodes()
               if tag != "equal"]
    model_mlp, model_cnn = list(code_lines("model")), list(code_lines("cnn-model"))
    model_changed = sum(len(a) for a, _ in changed if any(s in model_mlp for s in a))
    shared = [r for r, (path, _lines) in regions().items() if path == SCRIPT
              and "cnn-" + r in regions() and code_lines(r) == code_lines("cnn-" + r)]
    mlp_loss, cnn_loss = course.losses("sgd").astype(float), cnn.losses().astype(float)
    if len(mlp_loss) != len(cnn_loss):
        raise ValueError("the two runs are not the same length")
    x, mlp_mean = chunk_means(mlp_loss)
    _x, cnn_mean = chunk_means(cnn_loss)
    lower = int((cnn_mean < mlp_mean).sum())
    end_mlp, end_cnn = float(mlp_loss[-LAST:].mean()), float(cnn_loss[-LAST:].mean())
    meta = dict(
        source=SOURCE, evidence="measured on MNIST: both runs, the same 938 batches",
        mlp_blocks=block_names("model"), cnn_blocks=block_names("cnn-model"),
        mlp_count=course.count(), cnn_count=cnn.count(), mlp_count_text=commas(course.count()),
        cnn_count_text=commas(cnn.count()), mlp_model=model_mlp, cnn_model=model_cnn,
        step_code=list(code_lines("cnn-step")), step_lines=len(code_lines("cnn-step")),
        train_code=list(code_lines("cnn-train")),
        program_lines=len(mlp_prog), cnn_program_lines=len(cnn_prog), same_lines=int(same),
        changed=[dict(mlp=list(a), cnn=list(b_)) for a, b_ in changed],
        model_changed=int(model_changed), shared_regions=shared,
        steps=len(mlp_loss), chunk=CHUNK, chunk_x=x.tolist(), mlp_curve=mlp_mean.tolist(),
        cnn_curve=cnn_mean.tolist(), chunks=len(x), cnn_lower=lower,
        first_mlp=float(mlp_loss[0]), first_cnn=float(cnn_loss[0]),
        last=LAST, end_mlp=end_mlp, end_cnn=end_cnn, end_mlp_text="%.3f" % end_mlp,
        end_cnn_text="%.3f" % end_cnn,
        limitations=[
            "each point is the mean of %d steps' losses (the last of 13 steps), the same for "
            "both runs; single batches are noisier" % CHUNK,
            "these are training losses on the batches being learned from; held-out accuracy "
            "is the next animation's",
            "besides the model, the programs differ in a print of a .grad's shape and the "
            "saved file's name"])
    panels = (
        Panel("Numbers to learn", "the MLP's and the CNN's parameters", "bars",
              dict(values=np.array([course.count(), cnn.count()], float),
                   labels=["MLP", "CNN"])),
        Panel("Lines of the program", "lines the two programs share, and lines that differ",
              "bars", dict(values=np.array([same, len(mlp_prog) - same], float),
                           labels=["the same", "different"])),
        Panel("The two losses", "the mean of each %d steps' losses, same batches" % CHUNK,
              "curve", dict(x=x, ys=np.stack([mlp_mean, cnn_mean]), labels=["MLP", "CNN"])),
        Panel("Every step", "the loss of every step of both runs", "curve",
              dict(x=np.arange(len(mlp_loss), dtype=float), ys=np.stack([mlp_loss, cnn_loss]),
                   labels=["MLP", "CNN"])),
    )
    return Study("The same training loop", "change the model, keep everything else: the "
                 "same lines train the CNN", panels, meta)


# ====================================================================== 220
def conv_loop(img, W, bias):
    """The first ``nn.Conv2d`` by direct loops (a check of the maps)."""
    k = W.shape[-1]
    side = img.shape[0] - k + 1
    out = np.zeros((len(W), side, side))
    for o in range(len(W)):
        for r in range(side):
            for c in range(side):
                out[o, r, c] = float((img[r:r + k, c:c + k] * W[o, 0]).sum()) + float(bias[o])
    return out


def stamps_learn():
    stamps = np.stack([cnn.params(s)["0.weight"] for s in STAMP_STEPS]).astype(float)
    start = stamps[0]
    moved = [float(np.abs(s - start).mean()) for s in stamps]
    img, label = course.digit(TRAIN_DIGIT)
    img = np.asarray(img, float)[0]
    maps = {}
    for s in (STAMP_STEPS[0], STAMP_STEPS[-1]):
        a = cnn.activations(s, TRAIN_DIGIT)
        P = cnn.params(s)
        again = conv_loop(img, P["0.weight"], P["0.bias"])
        if not np.allclose(again, a["z1"][0], atol=1e-4):
            raise ValueError("the maps are not the stamps on the digit")
        maps[s] = a["z1"][0].astype(float)
    before, after = cnn.activations(0, TRAIN_DIGIT), cnn.activations(cnn.EPOCH_STEPS,
                                                                    TRAIN_DIGIT)
    biggest = [float(np.abs(s).max()) for s in stamps]
    meta = dict(
        source=SOURCE, evidence="measured on MNIST: the first layer's stamps",
        steps=list(STAMP_STEPS), stamps=int(stamps.shape[1]), shape_text=shape(start.shape),
        moved=moved, moved_text=[val(v) for v in moved], biggest=biggest,
        biggest_text=[val(v) for v in biggest], bound=float(1 / 3),
        lim=float(np.abs(stamps).max()), digit=TRAIN_DIGIT, label=int(label),
        map_shape=list(maps[0].shape),
        chance_before=float(before["probs"][0, label]),
        chance_after=float(after["probs"][0, label]),
        conv_line=code_line("cnn-model", "Conv2d(1, 8, 3)"),
        step_line=code_line("cnn-step", "optimizer.step()"),
        limitations=[
            "each stamp's 9 tiles are shaded on one scale for all stamps and steps, from "
            "the most negative weight (lightest) to the most positive (deepest)",
            "each map is shaded from its own smallest to its own biggest number; the maps "
            "are the first nn.Conv2d's output, before ReLU",
            "steps 10, 100 and 500 are the weights before those steps; 938 after the epoch"])
    panels = (
        Panel("The 8 stamps", "model[0].weight at steps %s" % ", ".join(map(str, STAMP_STEPS)),
              "images", dict(values=stamps[:, :, 0].reshape(-1, 3, 3),
                             labels=["%d/%d" % (s, j + 1) for s in STAMP_STEPS
                                     for j in range(stamps.shape[1])])),
        Panel("How far they moved", "mean |weight - its start| at each step", "curve",
              dict(x=np.array(STAMP_STEPS, float), ys=np.array([moved]),
                   labels=["mean move"])),
        Panel("Maps before", "the 8 maps of training digit %d (a %d), step 0"
              % (TRAIN_DIGIT, label), "images", dict(values=maps[0], digit=img)),
        Panel("Maps after", "the same digit, the stamps after one epoch", "images",
              dict(values=maps[cnn.EPOCH_STEPS])),
    )
    return Study("The stamps learn", "the CNN's first eight stamps from random to trained, "
                 "and the maps they make of one digit", panels, meta)


STUDIES = {"blocks-cnn-backward": cnn_backward, "blocks-cnn-same-loop": same_loop,
           "blocks-cnn-stamps-learn": stamps_learn}

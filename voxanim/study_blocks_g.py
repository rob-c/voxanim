"""The numbers behind animations 203-205 of the "Blocks and code" series (group ``g``):
``optimizer.zero_grad()``, ``optimizer.step()`` and the learning rate, all from the course's
one MNIST classifier and its recorded runs (:mod:`voxanim.study_course`).

* ``blocks-zero-grad``: the red tiles (``.grad``) of the last layer's weight,
  ``model[5].weight`` ``[10, 32]``, after the first batch's ``loss.backward()``;
  ``zero_grad()`` sets all 52,650 of them to 0; the second batch's own gradient fills them
  again. Then the real run that forgets ``zero_grad`` (``no_zero_grad``): ``.grad`` holds
  the sum of every gradient so far, which grows (its length 0.84, 1.30, 1.78 ... 37.6) while
  each batch's own gradient does not; its training loss climbs back up, and it gets 10% of
  the 10,000 held-out digits right (the SGD run: 95%).
* ``blocks-optimizer-step``: one ``SGD(lr=0.1)`` step from the first weights with the first
  batch's gradient: three tiles of ``model[5].weight`` before, ``- 0.1 x`` their gradient,
  and after (which is exactly the recorded weights before step 1); the loss of the same
  batch before (2.3223) and after (2.2616), computed here with study_course's functions.
* ``blocks-learning-rate``: the same start with ``lr`` 0.001, 0.1 and 0.8 (runs
  ``lr_small``, ``sgd``, ``lr_large``): their training-loss curves (means of each 25 steps)
  and held-out accuracies after one epoch (55%, 95%, 27%), and how far one tile moves on the
  first step at each rate.

Every number shown is measured on MNIST by the recorded runs or is exact arithmetic on
them, checked here.
"""

from __future__ import annotations

import numpy as np

from . import study_course as course
from .blocks import code_line, code_lines
from .study import Panel, Study

LAYER = "5.weight"            # the weight drawn: model[5].weight, [10, 32]
LAYER_NAME = "model[5].weight"
TILES = 3                     # tiles read out number by number
GAP = 8                       # and at least this many columns apart
CHUNK = 25                    # the loss curves: the mean of each 25 steps
COUNT_AT = (0, 1, 2, 3, 9, 19, 49, 99, 937)     # 0-based steps the pile is read out at
SOURCE = ("MNIST, trained as examples/mnist_blocks.py trains it (voxanim.study_course: one "
          "epoch of SGD, lr=0.1, batches of 64, from the same seeded start)")


def chunk_means(values, size=CHUNK):
    """``(x, means)``: the mean of each ``size`` steps of ``values`` (the last chunk may be
    shorter), at ``x`` = the number of steps done at its end."""
    v = np.asarray(values, float)
    ends = np.r_[np.arange(size, len(v), size), len(v)]
    starts = np.r_[0, ends[:-1]]
    return ends.astype(float), np.array([v[a:b].mean() for a, b in zip(starts, ends)])


def norm(G):
    """The length of a whole gradient: the square root of the sum of every number squared."""
    return float(np.sqrt(sum((np.asarray(g, np.float64) ** 2).sum() for g in G.values())))


def agrees(w, g, lr=course.LR, places=4):
    """True when the step's arithmetic on the numbers as shown (``places`` decimals) gives
    the new weight as shown: ``w - lr g``, rounded, is the rounded result."""
    shown = round(w, places) - lr * round(g, places)
    new = float(np.float32(w) - np.float32(lr) * np.float32(g))
    return abs(shown - round(new, places)) < 0.5 * 10.0 ** -places - 1e-9


def biggest(g, n=TILES, gap=GAP, w=None):
    """``n`` tiles of ``g`` with large gradients, read out side by side under the grid: the
    largest by size, then the largest of the other sign, then the largest left, none of them
    fewer than ``gap`` columns from another (and, given the weights ``w``, only tiles whose
    step adds up as shown, :func:`agrees`); ``[(row, col), ...]``, left to right."""
    g = np.asarray(g)
    order = [tuple(int(v) for v in np.unravel_index(i, g.shape))
             for i in np.argsort(-np.abs(g).ravel(), kind="stable")]
    if w is not None:
        order = [rc for rc in order if agrees(float(w[rc]), float(g[rc]))]
    out = []

    def take(want=None):
        for r, c in order:
            if (r, c) in out or any(abs(c - c2) < gap for _r2, c2 in out):
                continue
            if want is None or np.sign(g[r, c]) == want:
                out.append((r, c))
                return

    take()
    if n > 1:
        take(-np.sign(g[out[0]]))
    while len(out) < n:
        take()
    return sorted(out, key=lambda rc: rc[1])


def two_figures(v):
    """A small positive number to two significant figures, written out: 0.000045, 0.0045,
    0.036."""
    v = float(v)
    places = max(0, 1 - int(np.floor(np.log10(abs(v)))))
    return "%.*f" % (places, v)


def percent(a):
    return "%d%%" % round(100 * float(a))


# ====================================================================== 203
def zero_grad():
    P0, G0, G1 = course.params(0), course.grads(0), course.grads(1)
    tiles = biggest(G0[LAYER], w=P0[LAYER])            # the tiles of blocks-optimizer-step
    plain, piled = course.grad_norms("sgd"), course.grad_norms("no_zero_grad")
    pile = {k: G0[k] + G1[k] for k in G0}               # .grad after two batches, no zero_grad
    if not np.isclose(norm(G0), plain["fresh"][0], rtol=1e-5):
        raise ValueError("the first gradient's length is not the recorded one")
    if not np.isclose(norm(pile), piled["used"][1], rtol=1e-4):
        raise ValueError("two gradients added up are not what the run without zero_grad used")
    sgd_loss, bad_loss = course.losses("sgd"), course.losses("no_zero_grad")
    if sgd_loss[1] != bad_loss[1]:
        raise ValueError("the two runs part before their second step")
    x, sgd_mean = chunk_means(sgd_loss)
    _x, bad_mean = chunk_means(bad_loss)
    h_sgd, h_bad = course.heldout("sgd"), course.heldout("no_zero_grad")
    meta = dict(
        source=SOURCE, evidence="measured on MNIST: the recorded runs", layer=LAYER,
        layer_name=LAYER_NAME, shape=list(P0[LAYER].shape), numbers=course.count(),
        weight=P0[LAYER].tolist(), grad=G0[LAYER].tolist(), grad_next=G1[LAYER].tolist(),
        grad_pile=pile[LAYER].tolist(), tiles=tiles,
        g0=[float(G0[LAYER][r, c]) for r, c in tiles],
        g1=[float(G1[LAYER][r, c]) for r, c in tiles],
        pile=[float(pile[LAYER][r, c]) for r, c in tiles],
        length_first=float(plain["fresh"][0]), length_next=float(plain["fresh"][1]),
        length_pile=norm(pile),
        count_at=list(COUNT_AT), batches=[s + 1 for s in COUNT_AT],
        fresh=[float(piled["fresh"][s]) for s in COUNT_AT],
        used=[float(piled["used"][s]) for s in COUNT_AT],
        used_max=float(piled["used"].max()), fresh_last=float(piled["fresh"][-1]),
        fresh_all=piled["fresh"].tolist(), used_all=piled["used"].tolist(),
        chunk=CHUNK, chunk_x=x.tolist(), loss_sgd=sgd_mean.tolist(), loss_bad=bad_mean.tolist(),
        first_loss=float(sgd_loss[0]), last_sgd=float(sgd_mean[-1]),
        last_bad=float(bad_mean[-1]), low_bad=float(bad_mean.min()),
        low_bad_at=float(x[int(np.argmin(bad_mean))]),
        accuracy_sgd=float(h_sgd["accuracy"][-1]), accuracy_bad=float(h_bad["accuracy"][-1]),
        percent_sgd=percent(h_sgd["accuracy"][-1]), percent_bad=percent(h_bad["accuracy"][-1]),
        test=course.TEST,
        code=list(code_lines("step")), mistake=list(code_lines("mistake-no-zero-grad")),
        zero_line=code_line("step", "zero_grad"),
        limitations=["the red tiles drawn are one block's (model[5].weight, 320 of the "
                     "52,650); the length of the gradient is over all of them",
                     "the loss curves are the mean of each 25 steps of the per-batch "
                     "loss; single batches of the run without zero_grad reach %.1f"
                     % float(bad_loss.max()),
                     "the run without zero_grad is the same run as the script's until its "
                     "second step, which adds the first batch's gradient to the second's",
                     "PyTorch's zero_grad() sets .grad to None by default (set_to_none=True), "
                     "which the next backward() treats as 0: drawn as grey tiles of 0"])
    panels = (
        Panel("The red tiles", "model[5].weight.grad after the first batch's loss.backward()",
              "matrix", dict(values=G0[LAYER].astype(float))),
        Panel("The gradients pile up", "without zero_grad: the length of each batch's own "
              "gradient, and of what .grad holds (their sum so far)", "curve",
              dict(x=np.arange(course.EPOCH_STEPS) + 1.0,
                   ys=np.stack([piled["fresh"], piled["used"]]),
                   labels=["this batch's gradient", "what .grad holds"])),
        Panel("The loss goes wrong", "training loss, mean of each 25 steps: the script, and "
              "the script without zero_grad", "curve",
              dict(x=x, ys=np.stack([sgd_mean, bad_mean]),
                   labels=["with zero_grad", "without zero_grad"])),
        Panel("Held-out accuracy", "share of the 10,000 test digits right after one epoch",
              "bars", dict(values=np.array([h_sgd["accuracy"][-1], h_bad["accuracy"][-1]]),
                           labels=["with zero_grad", "without zero_grad"])),
    )
    return Study("optimizer.zero_grad()", "zero_grad empties the gradients before each "
                 "backward; forgotten, they pile up and the model learns nothing", panels, meta)


# ====================================================================== 204
def optimizer_step():
    P0, G0 = course.params(0), course.grads(0)
    b = course.batch(0)
    after = course.sgd_update(P0, G0, course.LR)
    recorded = course.params(1)
    if not all(np.array_equal(after[k], recorded[k]) for k in P0):
        raise ValueError("one step of SGD does not give the recorded weights")
    before_loss = course.loss_of(P0, b["x"], b["y"])
    after_loss = course.loss_of(after, b["x"], b["y"])
    if not np.isclose(before_loss, course.losses("sgd")[0], rtol=1e-6):
        raise ValueError("the first batch's loss is not the recorded one")
    if not after_loss < before_loss:
        raise ValueError("the step did not lower the batch's loss")
    w, g, new = P0[LAYER], G0[LAYER], after[LAYER]
    tiles = biggest(g, w=w)
    sizes = {k: int(v.size) for k, v in P0.items()}
    meta = dict(
        source=SOURCE, evidence="exact arithmetic on the recorded run", layer=LAYER,
        layer_name=LAYER_NAME, shape=list(w.shape), lr=course.LR,
        weight=w.tolist(), grad=g.tolist(), after=new.tolist(), tiles=tiles,
        before_w=[float(w[r, c]) for r, c in tiles], g=[float(g[r, c]) for r, c in tiles],
        move=[float(-course.LR * g[r, c]) for r, c in tiles],
        after_w=[float(new[r, c]) for r, c in tiles],
        layer_numbers=int(w.size), numbers=course.count(), sizes=sizes,
        changed=int(sum(int((after[k] != P0[k]).sum()) for k in P0)),
        loss_before=before_loss, loss_after=after_loss, drop=before_loss - after_loss,
        batch=len(b["y"]),
        code=list(code_lines("optimizer")) + list(code_lines("step")),
        step_line=code_line("step", "optimizer.step()"),
        limitations=["three tiles of model[5].weight are read out; the same rule moves "
                     "all 52,650 numbers (a gradient of exactly 0 leaves its number where "
                     "it is)",
                     "the loss after the step is of the same batch: whether the step helps "
                     "other digits is the loss curve's question"])
    panels = (
        Panel("Weights before", "model[5].weight before the first step (as initialised)",
              "matrix", dict(values=w.astype(float))),
        Panel("Their gradients", "model[5].weight.grad from the first batch", "matrix",
              dict(values=g.astype(float))),
        Panel("How far each moved", "after minus before: -0.1 times the gradient", "matrix",
              dict(values=(new - w).astype(float))),
        Panel("The same batch's loss", "loss_fn(model(x), y) on the first batch, before and "
              "after optimizer.step()", "bars",
              dict(values=np.array([before_loss, after_loss]), labels=["before", "after"])),
    )
    return Study("optimizer.step()", "SGD moves every weight by -0.1 times its gradient, and "
                 "the same batch's loss goes down", panels, meta)


# ====================================================================== 205
RATE_RUNS = (("lr_small", course.RATES[0]), ("sgd", course.RATES[1]),
             ("lr_large", course.RATES[2]))


def learning_rate():
    G0 = course.grads(0)
    tile = biggest(G0[LAYER], 1, w=course.params(0)[LAYER])[0]
    g = float(G0[LAYER][tile])
    curves, accs, lasts, highs, starts, held = [], [], [], [], [], []
    x = None
    for run, _lr in RATE_RUNS:
        loss = course.losses(run)
        x, m = chunk_means(loss)
        h = course.heldout(run)
        curves.append(m)
        accs.append(float(h["accuracy"][-1]))
        lasts.append(float(m[-1]))
        highs.append(float(loss.max()))
        starts.append(float(loss[0]))
        held.append(h["accuracy"])
    if len(set(starts)) != 1:
        raise ValueError("the three runs do not start from the same loss")
    rates = [lr for _run, lr in RATE_RUNS]
    if rates[1] != course.LR:
        raise ValueError("the middle rate is not the script's")
    big = course.losses("lr_large")
    meta = dict(
        source=SOURCE, evidence="measured on MNIST: three recorded runs",
        runs=[r for r, _ in RATE_RUNS], rates=rates, rate_text=["%g" % r for r in rates],
        first_loss=starts[0], chunk=CHUNK, chunk_x=x.tolist(),
        curves=[c.tolist() for c in curves], last=lasts, highest=highs,
        highest_at=int(np.argmax(big)), spikes=int((big > course.losses("sgd")[0]).sum()),
        accuracy=accs, percent=[percent(a) for a in accs], test=course.TEST,
        tile=list(tile), tile_grad=g, moves=[abs(lr * g) for lr in rates],
        moves_text=[two_figures(abs(lr * g)) for lr in rates],
        code=list(code_lines("optimizer")), lr_line=code_line("optimizer", "lr="),
        limitations=["the script sets lr=0.1; the other two runs are the same script with "
                     "that number changed",
                     "the loss curves are the mean of each 25 steps; single batches at "
                     "lr=0.8 reach %.2f" % highs[2]])
    panels = (
        Panel("Three learning rates", "training loss, mean of each 25 steps, from the same "
              "start", "curve", dict(x=x, ys=np.stack(curves),
                                     labels=["lr=%g" % r for r in rates])),
        Panel("Held-out accuracy as they learn", "share of the 10,000 test digits right, "
              "every 25 steps", "curve",
              dict(x=np.asarray(course.heldout("sgd")["steps"], float), ys=np.stack(held),
                   labels=["lr=%g" % r for r in rates])),
        Panel("After one epoch", "held-out accuracy after 938 steps", "bars",
              dict(values=np.array(accs), labels=["lr=%g" % r for r in rates])),
        Panel("One step of one tile", "how far model[5].weight[%d, %d] moves on the first "
              "step: lr times its gradient" % tile, "bars",
              dict(values=np.array([abs(lr * g) for lr in rates]),
                   labels=["lr=%g" % r for r in rates])),
    )
    return Study("The learning rate", "From the same start, lr=0.001 crawls, lr=0.1 learns "
                 "and lr=0.8 bounces", panels, meta)


STUDIES = {"blocks-zero-grad": zero_grad, "blocks-optimizer-step": optimizer_step,
           "blocks-learning-rate": learning_rate}

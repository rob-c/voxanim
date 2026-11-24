"""The numbers behind animations 206-208 of the "Blocks and code" series (group ``h``):
Adam, the training loop and epochs, all from the course's one recorded run
(:mod:`voxanim.study_course`).

* ``blocks-adam``: ``Adam(model.parameters())`` keeps two more numbers for every number
  the model learns, ``m`` and ``v`` (both 0 at the start), so it holds 3 x 52,650; one
  weight of the last layer (``5.weight``, ``[10, 32]``) followed through the Adam run's
  second step with its real numbers (``m``: 0.9 of the old ``m`` plus 0.1 of the gradient;
  ``v``: the same for the squared gradient with 0.999; the move: the learning rate times
  ``m`` over ``sqrt v``, each corrected for the early steps), checked against the
  recorded weights; and the real Adam and SGD runs from the same start.
* ``blocks-the-training-loop``: the region ``train`` as a track with the five lines of a
  step as its stations; what each station makes on the first step; the step counter and
  the real per-step loss going round 938 times, and what ``print(epoch, loss.item())``
  writes at the end.
* ``blocks-epochs-and-batches``: 60,000 / 64 = 937.5, so 937 batches of 64 and one of 32,
  938 steps an epoch; the loader's new order every epoch (each epoch's first digits); the
  loss per step against the loss per epoch over three epochs (the SGD run continued,
  :func:`~voxanim.study_course.more_epochs`).

Every number shown is either exact arithmetic or read from the recorded runs, and checked
here (the Adam update against the recorded weights, each epoch's order against a
permutation of the 60,000 digits).
"""

from __future__ import annotations

import numpy as np

from . import study_course as course
from .blocks import code_line, code_lines, commas, shape
from .study import Panel, Study

SOURCE = ("the course's MNIST classifier (784-64-32-10) trained in numpy exactly as "
          "examples/mnist_blocks.py trains it (voxanim.study_course)")
LAYER = "5.weight"            # the layer whose blocks 206 draws: [10, 32], every tile shown
ADAM_T = 2                    # the Adam update worked with real numbers: the second
SMOOTH = 50                   # the Adam and SGD curves: the mean of the last 50 steps' losses
EPOCHS = 1 + course.MORE_EPOCHS
FIRST_SHOWN = 7               # each epoch's first digits shown
# the five lines of a step, as the stations of the track (region "train", without the
# line that moves x and y to the device: the five lines TODO.md names)
STEP_LINES = (("data", "for x, y in loader:"), ("forward", "logits = model(x)"),
              ("loss", "loss = loss_fn(logits, y)"), ("backward", "optimizer.zero_grad()"),
              ("backward", "loss.backward()"), ("step", "optimizer.step()"))


def sci(v, places=2):
    """A small positive number as ``2.06×10⁻⁶`` (a real minus sign, superscript power)."""
    v = float(v)
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(abs(v))))
    mant = v / 10 ** e
    if round(abs(mant), places) >= 10:
        mant, e = mant / 10, e + 1
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return ("%.*f×10%s" % (places, mant, str(e).translate(sup))).replace("-", "−")


def num(v, places):
    """``v`` with ``places`` decimals and a real minus sign."""
    s = "%.*f" % (places, float(v))
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "−")


def signed(v, places):
    """``v`` with its sign always written: ``+0.00093``."""
    s = num(v, places)
    return s if s.startswith("−") else "+" + s


def moving_mean(x, w=SMOOTH):
    """The mean of the last ``w`` values (fewer at the start)."""
    x = np.asarray(x, float)
    cs = np.r_[0.0, np.cumsum(x)]
    i = np.arange(1, len(x) + 1)
    lo = np.maximum(i - w, 0)
    return (cs[i] - cs[lo]) / (i - lo)


def pick_weight(g_before, g_now):
    """The weight of the layer whose gradient is biggest at the worked step among those
    whose gradient kept its sign from the step before (so ``m`` grows): ``(row, col)``."""
    score = np.abs(g_now) * (np.sign(g_now) == np.sign(g_before))
    r, c = np.unravel_index(int(np.argmax(score)), np.shape(g_now))
    return int(r), int(c)


def adam_move(m, v, t, lr=course.ADAM_LR, betas=course.BETAS, eps=course.EPS):
    """Adam's move of one number from its ``m`` and ``v`` after ``t`` steps, in float64:
    ``-lr * m_hat / (sqrt(v_hat) + eps)``, with ``m_hat = m / (1 - b1^t)`` and
    ``v_hat = v / (1 - b2^t)``. Returns ``(m_hat, sqrt(v_hat), move)``."""
    b1, b2 = betas
    m_hat = float(m) / (1 - b1 ** t)
    root = float(np.sqrt(float(v) / (1 - b2 ** t)))
    return m_hat, root, -lr * m_hat / (root + eps)


# ====================================================================== 206
def adam():
    t = ADAM_T
    b1, b2 = course.BETAS
    g_prev = course.grads(t - 2, "adam")[LAYER]           # the gradient of step 1
    g = course.grads(t - 1, "adam")[LAYER]                # ... and of step 2
    s_prev, s_now = course.adam_state(t - 1), course.adam_state(t)
    w_before = course.params(t - 1, "adam")
    w_after = course.params(t, "adam")
    r, c = pick_weight(g_prev, g)
    at = (r, c)
    m_old, v_old = float(s_prev["m"][LAYER][at]), float(s_prev["v"][LAYER][at])
    m_new, v_new = float(s_now["m"][LAYER][at]), float(s_now["v"][LAYER][at])
    gv = float(g[at])
    # the recorded state is Adam's arithmetic on the recorded gradient
    if not np.isclose(m_new, b1 * m_old + (1 - b1) * gv, rtol=1e-5, atol=1e-10):
        raise ValueError("m is not 0.9 m + 0.1 g")
    if not np.isclose(v_new, b2 * v_old + (1 - b2) * gv * gv, rtol=1e-4, atol=1e-14):
        raise ValueError("v is not 0.999 v + 0.001 g²")
    m_hat, root, move = adam_move(m_new, v_new, t)
    wb, wa = float(w_before[LAYER][at]), float(w_after[LAYER][at])
    if not np.isclose(wa - wb, move, rtol=1e-3, atol=1e-8):
        raise ValueError("the recorded move is not lr × m̂ / √v̂")
    moves = np.concatenate([np.abs(w_after[k] - w_before[k]).ravel() for k in course.KEYS])
    n = course.count()
    sgd, adm = course.losses("sgd").astype(float), course.losses("adam").astype(float)
    ms, ma = moving_mean(sgd), moving_mean(adm)
    ahead = np.flatnonzero(ma < ms)
    h_sgd, h_adam = course.heldout("sgd"), course.heldout("adam")
    zeros = course.adam_start(course.init_params())
    if any(np.any(zeros[w][k]) for w in ("m", "v") for k in course.KEYS):
        raise ValueError("Adam's state does not start at zero")
    line = code_line("optimizer", "Adam(")
    meta = dict(
        source=SOURCE, layer=LAYER, layer_shape=list(g.shape), layer_shape_text=shape(g.shape),
        layer_numbers=int(g.size), t=t, weight=[r, c], lr=course.ADAM_LR, betas=list(
            course.BETAS), eps=course.EPS,
        parameters=n, adam_numbers=3 * n, times=3, per_weight=2, start=0.0,
        g_prev=float(g_prev[at]), g=gv, m_old=m_old, v_old=v_old, m=m_new, v=v_new,
        m_hat=m_hat, root_v_hat=root, move=move, w_before=wb, w_after=wa,
        recorded_move=wa - wb, sgd_lr=course.LR, sgd_move=-course.LR * gv,
        m_fix=1 - b1 ** t, v_fix=1 - b2 ** t,
        largest_move=float(moves.max()), median_move=float(np.median(moves)),
        moved=int((moves > 0).sum()),
        sgd_curve=ms.tolist(), adam_curve=ma.tolist(),
        adam_ahead_until=int(ahead[-1]) if len(ahead) else -1,
        adam_ahead_steps=int(len(ahead)), adam_ahead_contiguous=bool(
            len(ahead) and np.array_equal(ahead, np.arange(ahead[0], ahead[-1] + 1))),
        sgd_end=float(ms[-1]), adam_end=float(ma[-1]),
        sgd_accuracy=float(h_sgd["accuracy"][-1]), adam_accuracy=float(h_adam["accuracy"][-1]),
        first_loss=float(sgd[0]),
        code=list(code_lines("optimizer")), adam_line=line,
        adam_line_index=list(code_lines("optimizer")).index(line),
        step=list(code_lines("step")), step_line=code_line("step", "optimizer.step()"),
        text=dict(
            g=num(gv, 6), m_old=num(m_old, 6), m=num(m_new, 6),
            v_old=sci(v_old, 3), v=sci(v_new, 3),
            m_sum="0.9 × %s + 0.1 × %s" % (num(m_old, 6), num(gv, 6)),
            v_sum="0.999 × %s + 0.001 × %s²" % (sci(v_old, 3), num(abs(gv), 6)),
            m_hat="÷ %s = %s" % (num(1 - b1 ** t, 2), num(m_hat, 4)),
            root="√(v ÷ %s) = %s" % (num(1 - b2 ** t, 6), num(root, 4)),
            move=signed(move, 5),
            move_sum="0.001 × %s ÷ %s" % (num(abs(m_hat), 4), num(root, 4)),
            w_before=num(wb, 5), w_after=num(wa, 5), sgd_move=signed(-course.LR * gv, 5),
            largest=num(moves.max(), 4)),
        limitations=[
            "the blocks are the last layer's weights (5.weight, [10, 32]); Adam keeps an m "
            "and a v for every one of the model's 52,650 numbers, biases included",
            "the curves are the mean of the last %d steps' losses, so the noise of single "
            "batches is smoothed out" % SMOOTH,
            "Adam runs at its default learning rate 0.001 and SGD at the script's 0.1: "
            "the comparison is of the two settings the script uses, not of the best of each"])
    steps = np.arange(len(sgd), dtype=float)
    panels = (
        Panel("Three blocks", "the last layer's weights after %d Adam steps, and Adam's m "
              "and v for them: the same shape" % t, "images",
              dict(values=np.stack([w_after[LAYER], s_now["m"][LAYER], s_now["v"][LAYER]])
                   .astype(float), labels=["weight", "m", "v"])),
        Panel("Memory", "numbers kept: the model's, then Adam's m and v for each of them",
              "bars", dict(values=np.array([n, n, n], float), labels=["weights", "m", "v"])),
        Panel("One weight, step %d" % t, "weight %s[%d, %d]: its gradient, m, v, Adam's "
              "corrected ratio and its move" % (LAYER, r, c), "bars",
              dict(values=np.array([gv, m_old, m_new, m_hat, root, move, wb, wa], float),
                   labels=["g", "m before", "m", "m corrected", "√v corrected", "move",
                           "weight before", "weight after"])),
        Panel("Adam and SGD", "the loss, mean of the last %d steps, of the two real runs "
              "from the same start" % SMOOTH, "curve",
              dict(x=steps, ys=np.stack([ms, ma]), labels=["SGD lr 0.1", "Adam lr 0.001"])),
    )
    return Study("Adam keeps two more blocks", "Adam keeps an m and a v for every weight and "
                 "moves each by about its learning rate", panels, meta)


# ====================================================================== 207
def training_loop():
    region = list(code_lines("train"))
    device = code_line("train", ".to(device)")
    pick = [i for i, s in enumerate(region) if s != device]
    shown = [region[i] for i in pick]
    if len(shown) > 8:
        raise ValueError("the loop is more than 8 lines")
    lines = {s.strip(): i for i, s in enumerate(shown)}
    stations = [dict(station=st, line=lines[code], code=code) for st, code in STEP_LINES]
    b = course.batch(0)
    P = course.init_params()
    a = course.forward(P, b["x"])
    loss0, G = course.loss_and_grads(P, b["x"], b["y"])
    losses = course.losses("sgd").astype(float)
    if not np.isclose(loss0, losses[0], rtol=1e-6):
        raise ValueError("the first step's loss is not the recorded one")
    n = course.count()
    grads = int(sum(v.size for v in G.values()))
    last = float(course.losses("sgd")[-1])
    printed = "0 %s" % repr(last)
    reads = dict(data="x: %s" % shape(b["x"].shape), forward="logits: %s" % shape(
        a["logits"].shape), loss="loss: %s" % num(loss0, 3), zero="gradients: all 0",
        backward="%s gradients" % commas(grads), step="%s numbers move" % commas(n))
    meta = dict(
        source=SOURCE, steps=course.EPOCH_STEPS, epochs=1, batch=course.BATCH,
        last_batch=course.LAST_BATCH, x_shape=list(b["x"].shape),
        logits_shape=list(a["logits"].shape), first_loss=float(loss0), last_loss=last,
        gradients=grads, parameters=n, losses=losses.tolist(),
        loss_max=float(losses.max()), loss_min=float(losses.min()),
        mean_last_100=float(losses[-100:].mean()),
        code=shown, pick=pick, left_out=device, stations=stations, reads=reads,
        printed=printed, print_line=code_line("train", "print("),
        epoch_line=code_line("train", "for epoch"),
        limitations=[
            "the loop shown leaves out the line x, y = x.to(device), y.to(device), so that "
            "the five lines of a step and the print fit in 8 lines",
            "the numbers are the numpy twin's run, whose shuffled order is not PyTorch's: "
            "the script itself prints another last loss, of the same size"])
    steps = np.arange(len(losses), dtype=float)
    counts = [int(np.prod(b["x"].shape)), int(np.prod(a["logits"].shape)), 1, grads, n]
    milestones = [0, 1, 2, 3, 10, 100, 500, course.EPOCH_STEPS - 1]
    panels = (
        Panel("The loss, step by step", "the loss of each of the 938 steps of one epoch",
              "curve", dict(x=steps, ys=losses[None], labels=["loss"])),
        Panel("What each station makes", "on the first step: the batch, the logits, the loss, "
              "the gradients, the numbers moved", "bars",
              dict(values=np.array(counts, float),
                   labels=["x", "logits", "loss", "gradients", "moved"])),
        Panel("Round the loop", "the loss after 0, 1, 2, 3, 10, 100, 500 and 937 times round",
              "bars", dict(values=losses[milestones], labels=[str(s) for s in milestones])),
        Panel("The logits of step 0", "logits = model(x) on the first batch: [64, 10]",
              "matrix", dict(values=a["logits"].astype(float))),
    )
    return Study("The training loop", "The five lines of a step, 938 times round: the loss "
                 "falls from %.2f to %.2f" % (loss0, last), panels, meta)


# ====================================================================== 208
def epochs_and_batches():
    train, b = course.TRAIN, course.BATCH
    full, last = divmod(train, b)
    steps = full + (last > 0)
    if (steps, last) != (course.EPOCH_STEPS, course.LAST_BATCH) or full * b + last != train:
        raise ValueError("60,000 / 64 is not 937 batches of 64 and one of 32")
    sizes = np.array([len(course.batch_indices(s)) for s in range(steps)])
    if sizes.sum() != train or sizes[-1] != last or (sizes[:-1] != b).any():
        raise ValueError("the batches do not add up to the training set")
    more = course.more_epochs()
    loss = np.asarray(more["loss"], float)
    first = np.asarray(more["first"])
    for e in range(EPOCHS):
        if not np.array_equal(np.sort(course.order(e)), np.arange(train)):
            raise ValueError("epoch %d does not see every digit once" % e)
        if not np.array_equal(course.order(e)[:len(first[e])], first[e]):
            raise ValueError("epoch %d's first digits" % e)
    _x, y = course.data()[:2]
    pix = course.pixels()[0]
    shown = first[:, :FIRST_SHOWN]
    imgs = pix[shown.ravel()].reshape(-1, course.SIDE, course.SIDE)
    labels = y[shown]
    means = loss.mean(1)
    meta = dict(
        source=SOURCE, train=train, batch=b, quotient=train / b, full=full, last_batch=last,
        steps=steps, check="%s × %d + %d = %s" % (commas(full), b, last, commas(train)),
        division="%s ÷ %d = %s" % (commas(train), b, ("%.1f" % (train / b))),
        epochs=EPOCHS, total_steps=EPOCHS * steps, digits_seen=EPOCHS * train,
        first=first.tolist(), shown=shown.tolist(), shown_labels=labels.tolist(),
        epoch_mean=means.tolist(), epoch_last=loss[:, -1].tolist(),
        epoch_min=loss.min(1).tolist(), epoch_max=loss.max(1).tolist(),
        accuracy=np.asarray(more["accuracy"], float).tolist(),
        test_loss=np.asarray(more["test_loss"], float).tolist(),
        first_loss=float(loss[0, 0]), losses=loss.ravel().tolist(),
        same_first=bool(len({tuple(r) for r in first.tolist()}) < EPOCHS),
        loader=list(code_lines("loader")), batch_line=code_line("loader", "batch_size"),
        shuffle_line=code_line("loader", "shuffle"), loop=list(code_lines("train")),
        epoch_line=code_line("train", "for epoch"),
        limitations=[
            "the script trains one epoch (range(1)); the second and third are the same run "
            "continued, as range(3) would run it (voxanim.study_course.more_epochs)",
            "the shuffled orders are the numpy twin's (seeded), not PyTorch's"])
    panels = (
        Panel("One epoch in batches", "the size of each of the 938 batches: 937 of 64, the "
              "last of 32", "bars", dict(values=sizes.astype(float),
                                         labels=[str(s) for s in range(steps)])),
        Panel("A new order each epoch", "the first %d digits of epochs 1, 2 and 3"
              % FIRST_SHOWN, "images", dict(values=imgs.astype(float),
                                            labels=[str(v) for v in labels.ravel()])),
        Panel("Loss per step", "the loss of every step of three epochs (2,814 steps)", "curve",
              dict(x=np.arange(loss.size, dtype=float), ys=loss.ravel()[None],
                   labels=["loss per step"])),
        Panel("Loss per epoch", "the mean loss of each epoch's 938 steps", "bars",
              dict(values=means, labels=["epoch %d" % (e + 1) for e in range(EPOCHS)])),
    )
    return Study("Epochs and batches", "60,000 digits in batches of 64 make 938 steps, one "
                 "epoch; per epoch the loss falls %.3f, %.3f, %.3f" % tuple(means), panels, meta)


STUDIES = {"blocks-adam": adam, "blocks-the-training-loop": training_loop,
           "blocks-epochs-and-batches": epochs_and_batches}

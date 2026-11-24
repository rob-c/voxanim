"""The numbers behind animations 209-211 of the "Blocks and code" series (group ``i``):
testing the trained model, saving and loading it, and one whole training step, all from the
course's one recorded run (:mod:`voxanim.study_course`, the SGD run of
``examples/mnist_blocks.py``).

* ``blocks-evaluation``: region ``evaluate`` on the model trained for one epoch:
  ``model.eval()`` (this model has no dropout, so no number of it changes), ``with
  torch.no_grad():`` (no gradient is made: 0 numbers, against 52,650 in a training step),
  then the 10,000 held-out digits in the script's ``test_loader`` order (batches of 1,000,
  not shuffled): the first ten counted one by one (test digit 8 is the first one it gets
  wrong), then ``correct`` after each batch, up to the accuracy, 9,494 / 10,000 = 0.9494.
* ``blocks-save-and-load``: regions ``save`` and ``load``: ``model.state_dict()`` holds the
  six tensors :data:`~voxanim.study_course.KEYS` with their shapes (52,650 numbers, 210,600
  bytes of float32); saved and loaded (:func:`~voxanim.study_course.roundtrip`), every
  number comes back the same, so the 10,000 held-out predictions are the same, 10,000 of
  10,000.
* ``blocks-one-step-slow-motion``: region ``step``, the recorded step :data:`STEP` (the
  run's second) station by station: its batch, the shapes through the model, the loss, the
  gradients (and the last step's, which ``zero_grad`` empties first), and one weight moved
  by ``-lr × grad``; then the loss on the same batch again, lower.

Every number is checked here: the accuracy against :func:`~voxanim.study_course.score`
and the recorded held-out curve, the running total against the per-digit hits, the step's
loss and gradients recomputed from its weights and batch, and its update against the
recorded weights of the step after.
"""

from __future__ import annotations

import math

import numpy as np

from . import study_course as course
from .blocks import code_line, code_lines, commas, shape
from .study import Panel, Study

TRAINED = course.EPOCH_STEPS        # the model as the script leaves it: after one epoch
TEST_BATCH = 1000                   # test_loader = DataLoader(test_set, 1000)
SHOWN = 10                          # held-out digits counted one by one
FILE = "mnist.pt"                   # torch.save(state, "mnist.pt")
FRESH_SEED = 1                      # the new model a later program loads into
STEP = 1                            # the step shown in slow motion: the run's second
CORNER = 8                          # the tiles drawn of each weight: an 8 x 8 corner
PIXEL_ROW = 14                      # 1.weight's corner: inputs from this row of the digit
PIXEL_COL = 10                      # ... columns 10-17, where the digits' strokes are
SOURCE = ("MNIST; the course's one run of examples/mnist_blocks.py (SGD, lr 0.1, batches "
          "of 64, one epoch), recorded by voxanim.study_course in numpy")
LIMIT_ORDER = ("the batches come in the numpy twin's shuffled order "
               "(voxanim.study_course.order), not PyTorch's, so the script run with torch "
               "trains on other batches and ends within a fraction of a percent of these "
               "numbers, not on them")


def heldout_predictions(P):
    """``(pred, right)``: the model's answer for each of the 10,000 test digits, in
    ``test_loader``'s order, and whether it is the label."""
    _x, _y, xt, yt = course.data()
    pred = course.forward(P, xt)["logits"].argmax(1)
    return pred, pred == yt


def running_correct(right, batch=TEST_BATCH):
    """``correct`` after each batch of ``test_loader``: the running total of hits."""
    return np.cumsum(np.asarray(right).reshape(-1, batch).sum(1))


def corner(key, a):
    """The tiles drawn of parameter ``key`` (an array of its shape): an 8 x 8 corner of a
    weight (1.weight: its first 8 outputs, from inputs row 14, columns 10-17 of the digit),
    the first 8 of a bias."""
    a = np.asarray(a)
    if a.ndim == 1:
        return a[:CORNER]
    if key == "1.weight":
        cols = PIXEL_ROW * course.SIDE + PIXEL_COL + np.arange(CORNER)
        return a[:CORNER, cols]
    return a[:CORNER, :CORNER]


# ====================================================================== 209
def evaluation():
    P = course.params(TRAINED)
    yt = course.data()[3]
    pred, right = heldout_predictions(P)
    acc, loss = course.score(P)
    run = running_correct(right)
    recorded = float(course.heldout("sgd")["accuracy"][-1])
    if not (math.isclose(acc, right.mean()) and math.isclose(acc, recorded)
            and int(run[-1]) == int(right.sum())):
        raise ValueError("the accuracy does not agree with the recorded run")
    wrong = np.flatnonzero(~right)
    first_wrong = int(wrong[0])
    if first_wrong >= SHOWN:
        raise ValueError("no mistake among the first %d test digits" % SHOWN)
    per_digit = np.array([right[yt == d].mean() for d in range(course.CLASSES)])
    dropout = course.heldout("dropout")
    h = course.heldout("sgd")
    meta = dict(
        source=SOURCE, evidence="measured on MNIST: the trained model on the 10,000 test "
        "digits", trained_steps=TRAINED, test=len(yt), test_batch=TEST_BATCH,
        batches=len(yt) // TEST_BATCH,
        shown_labels=[int(v) for v in yt[:SHOWN]], shown_preds=[int(v) for v in pred[:SHOWN]],
        shown_right=[bool(v) for v in right[:SHOWN]], shown_correct=int(right[:SHOWN].sum()),
        first_wrong=first_wrong, first_wrong_label=int(yt[first_wrong]),
        first_wrong_pred=int(pred[first_wrong]),
        per_batch=[int(v) for v in right.reshape(-1, TEST_BATCH).sum(1)],
        running=[int(v) for v in run], correct=int(right.sum()), wrong=int((~right).sum()),
        accuracy=float(acc), accuracy_text="%.4f" % acc, heldout_loss=float(loss),
        printed="%s" % (int(right.sum()) / len(yt)),
        fraction="%s / %s = %.4f" % (commas(right.sum()), commas(len(yt)), acc),
        grads_made=0, grads_training=course.count(P), numbers=course.count(P),
        dropout_eval=float(dropout["accuracy"][-1]),
        dropout_train_mode=float(dropout["accuracy_train_mode"][-1]),
        code=list(code_lines("evaluate")), train_mode=code_line("train-mode", "train"),
        eval_line=code_line("evaluate", "eval"), no_grad=code_line("evaluate", "no_grad"),
        loader_line=code_line("loader", "test_loader"),
        limitations=[
            "model.eval() changes nothing in this model, which has no dropout: on the course's "
            "dropout run it matters (%.4f held out in test mode, %.4f left in training mode)"
            % (float(dropout["accuracy"][-1]), float(dropout["accuracy_train_mode"][-1])),
            "no_grad is shown as the gradients it does not make: numpy keeps no graph, so "
            "the 0 is what PyTorch's no_grad guarantees, not a measurement of memory",
            LIMIT_ORDER])
    shown = course.pixels()[1][:SHOWN].reshape(-1, course.SIDE, course.SIDE)
    panels = (
        Panel("One by one", "the first %d test digits; the trained model's answer under each"
              % SHOWN, "images", dict(values=shown.astype(float), labels=[
                  "%d: %d" % (y, p) for y, p in zip(yt[:SHOWN], pred[:SHOWN])])),
        Panel("Correct, batch by batch", "right answers in each batch of 1,000 of "
              "test_loader", "bars", dict(values=right.reshape(-1, TEST_BATCH).sum(1)
                                          .astype(float), labels=["%d" % (i + 1) for i in
                                                                  range(len(yt) // TEST_BATCH)])),
        Panel("Accuracy during the epoch", "held-out accuracy every 25 steps of the run",
              "curve", dict(x=np.asarray(h["steps"], float), ys=np.asarray(h["accuracy"],
                                                                         float)[None],
                            labels=["SGD, lr 0.1"])),
        Panel("Per digit", "the share of each digit it gets right", "bars",
              dict(values=per_digit.astype(float), labels=[str(d) for d in range(10)])),
    )
    return Study("Testing the model", "model.eval() and torch.no_grad(): %s of %s held-out "
                 "digits right, %.4f" % (commas(right.sum()), commas(len(yt)), acc), panels,
                 meta)


# ====================================================================== 210
def save_and_load():
    P = course.params(TRAINED)
    r = course.roundtrip()
    total = course.count(P)
    if r["numbers"] != total or r["max_change"] != 0.0 or r["same_predictions"] != course.TEST:
        raise ValueError("saving and loading changed the model")
    pred, _right = heldout_predictions(P)
    F = course.init_params(FRESH_SEED)          # a new model, before load_state_dict
    keys = list(course.KEYS)
    sizes = [int(P[k].size) for k in keys]
    meta = dict(
        source=SOURCE, evidence="exact arithmetic: the same numbers, back from the file",
        keys=keys, shapes=[list(P[k].shape) for k in keys],
        shape_texts=[shape(P[k].shape) for k in keys], sizes=sizes,
        size_texts=[commas(n) for n in sizes], numbers=total, numbers_text=commas(total),
        bytes=4 * total, bytes_text=commas(4 * total), file=FILE,
        corners={k: corner(k, P[k]).tolist() for k in keys},
        fresh={k: corner(k, F[k]).tolist() for k in keys}, fresh_seed=FRESH_SEED,
        max_change=r["max_change"], same=r["same_predictions"], test=course.TEST,
        accuracy_before=r["accuracy_before"], accuracy_after=r["accuracy_after"],
        shown_preds=[int(v) for v in pred[:SHOWN]],
        save=list(code_lines("save")), load=list(code_lines("load")),
        limitations=[
            "saved and loaded as numpy does it (np.savez to bytes and back), not with "
            "torch.save: what is shown is that every number comes back the same; the "
            "file's size is not shown, only its %s bytes of numbers (4 per float32; "
            "numpy's file is %s bytes, torch's adds its own labels)"
            % (commas(4 * total), commas(r["file_bytes"])),
            "each tensor is drawn as an 8 x 8 corner of its tiles (a bias: its first 8); "
            "the new model they are loaded into is drawn with fresh random numbers "
            "(PyTorch's initialisation, seed %d), as a new program would make it; the "
            "script itself loads into the model it just saved" % FRESH_SEED,
            LIMIT_ORDER])
    panels = (
        Panel("model.state_dict()", "the six tensors by name: numbers in each", "bars",
              dict(values=np.array(sizes, float), labels=keys)),
        Panel("5.weight, as saved", "the last layer's 10 x 32 weights, the same after loading",
              "matrix", dict(values=P["5.weight"].astype(float))),
        Panel("Predictions", "each test digit's answer before saving (x) and after loading "
              "(y)", "points", dict(xy=np.stack([pred, pred], 1).astype(float))),
        Panel("Accuracy", "held out, before saving and after loading", "bars",
              dict(values=np.array([r["accuracy_before"], r["accuracy_after"]]),
                   labels=["before", "after"])),
    )
    return Study("Saving and loading", "torch.save(model.state_dict()): %s numbers in six "
                 "named tensors, loaded back the same" % commas(total), panels, meta)


# ====================================================================== 211
def one_step(step=STEP):
    P, G, Q = course.params(step), course.grads(step), course.params(step + 1)
    before = course.grads(step - 1)              # what .grad holds until zero_grad
    b = course.batch(step)
    loss, G2 = course.loss_and_grads(P, b["x"], b["y"])
    if not math.isclose(loss, float(course.losses()[step]), rel_tol=1e-6) or any(
            not np.allclose(G2[k], G[k], atol=1e-7) for k in G):
        raise ValueError("the recorded step does not recompute")
    new = course.sgd_update(P, G)
    if any(not np.array_equal(new[k], Q[k]) for k in Q):
        raise ValueError("the update is not the next step's weights")
    a = course.forward(P, b["x"])
    each = course.loss_each(a["logits"], b["y"])
    after = course.loss_of(Q, b["x"], b["y"])
    key = "5.weight"                             # the largest gradient among its tiles shown
    i, j = (int(v) for v in np.unravel_index(np.abs(corner(key, G[key])).argmax(),
                                             (CORNER, CORNER)))
    w0, g, w1 = float(P[key][i, j]), float(G[key][i, j]), float(Q[key][i, j])
    shapes = [list(b["x"].shape), list(a["x"].shape), list(a["z1"].shape),
              list(a["z2"].shape), list(a["logits"].shape)]
    weights = [k for k in course.KEYS if k.endswith("weight")]
    meta = dict(
        source=SOURCE, evidence="measured on MNIST: one recorded step", step=step,
        number=step + 1, steps=course.EPOCH_STEPS, batch=len(b["y"]),
        labels=[int(v) for v in b["y"]], x_shape=shape(b["x"].shape),
        y_shape=shape(b["y"].shape), shapes=[shape(s) for s in shapes],
        logits=[float(v) for v in a["logits"][0]], first_label=int(b["y"][0]),
        first_pred=int(a["logits"][0].argmax()), first_loss=float(each[0]),
        loss=float(loss), ln10=math.log(10), loss_after=float(after),
        numbers=course.count(P), grad_shape=shape(G["1.weight"].shape),
        grad_norm=float(course.grad_norms("sgd")["fresh"][step]), lr=course.LR,
        weight=dict(key=key, index=[i, j], before=w0, grad=g, after=w1,
                    check=float(np.float32(w0) - np.float32(course.LR) * np.float32(g))),
        weights=weights,
        corners={k: corner(k, P[k]).tolist() for k in weights},
        grads={k: corner(k, G[k]).tolist() for k in weights},
        old_grads={k: corner(k, before[k]).tolist() for k in weights},
        code=list(code_lines("step")),
        limitations=[
            "the tiles are an 8 x 8 corner of each weight (1.weight: inputs from row 14 of "
            "the digit); the weight read out is the one with the largest gradient among "
            "the 8 x 8 tiles of 5.weight shown",
            "optimizer.zero_grad() sets .grad to None in PyTorch 2 (set_to_none=True): the "
            "red tiles are taken away, not set to 0",
            LIMIT_ORDER])
    panels = (
        Panel("The batch", "batch %d of the epoch: its first 16 digits" % (step + 1),
              "images", dict(values=b["pixels"][:16].reshape(-1, course.SIDE, course.SIDE)
                             .astype(float), labels=[str(v) for v in b["y"][:16]])),
        Panel("Logits", "the first digit's 10 scores (label %d)" % int(b["y"][0]), "bars",
              dict(values=a["logits"][0].astype(float), labels=[str(d) for d in range(10)])),
        Panel("Gradient of 5.weight", "loss.backward(): one red tile per weight", "matrix",
              dict(values=G[key].astype(float))),
        Panel("The loss on this batch", "before and after optimizer.step()", "bars",
              dict(values=np.array([loss, after]), labels=["before", "after"])),
    )
    return Study("One training step, slowly", "data, forward, loss, backward, step: the loss "
                 "on its batch goes from %.2f to %.2f" % (loss, after), panels, meta)


STUDIES = {"blocks-evaluation": evaluation, "blocks-save-and-load": save_and_load,
           "blocks-one-step-slow-motion": one_step}

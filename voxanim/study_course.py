"""The course's one MNIST classifier, trained once in numpy exactly as the PyTorch program
``examples/mnist_blocks.py`` trains it, with everything the "Blocks and code" animations
(catalogue 185-214, :mod:`voxanim.blocks`) show recorded on the way.

The program, and its twin here:

* data: MNIST (:func:`voxanim.mnist.load`), pixels 0..1 (``ToTensor``), then
  ``Normalize((0.1307,), (0.3081,))``: :data:`MEAN` and :data:`STD` are the script's
  constants, the mean and spread of every training pixel (:func:`pixel_stats` computes
  them: 0.13066 and 0.30811);
* the model: ``nn.Sequential(nn.Flatten(), nn.Linear(784, 64), nn.ReLU(), nn.Linear(64,
  32), nn.ReLU(), nn.Linear(32, 10))``, its parameters kept as PyTorch keeps them (the keys
  of ``model.state_dict()``, :data:`KEYS`; a weight is ``[out, in]``), initialised as PyTorch
  initialises ``nn.Linear``: weights and biases uniform in ``±1/sqrt(fan_in)``;
* ``nn.CrossEntropyLoss()`` (mean over the batch) on the logits;
* batches of 64 in a new shuffled order each epoch, 938 steps an epoch (937 of 64 and one
  of 32: ``drop_last`` is off);
* ``SGD(lr=0.1)``, or with ``adam`` ``Adam()`` (lr 0.001, betas 0.9 and 0.999, eps 1e-8),
  updated in PyTorch's operation order; one epoch.

Everything is float32, as in PyTorch (the model's functions work in the precision of the
arrays they are given: the tests check the gradients in float64).
``tests/test_mnist_blocks_equivalence.py`` loads :func:`init_params` into the script's
own model and runs one SGD and one Adam step of the script's own lines on :func:`batch`
``(0)``: the loss, the gradients and the new weights agree with the recorded runs to 1e-5
(measured: 2e-7 for the loss, 1e-8 for the gradients, 1e-6 for Adam's new weights). The one
thing not the same is the order of the batches: PyTorch's shuffle comes from its own
generator, the twin's from numpy's (seeded, recorded by :func:`order`).

The API the animations use (arrays are numpy; a ``params`` dict is keyed by :data:`KEYS`):

* data: :func:`data` ``-> (x, y, xt, yt)`` normalised float32 ``(N, 784)`` and labels;
  :func:`pixels` ``-> (x, xt)`` the 0..1 pixels; :func:`pixel_stats`; :func:`digit`
  ``(i)`` one training digit as ``train_set[i]`` gives it, ``(x (1, 28, 28), y)``;
  :func:`order` ``(epoch)``; :func:`batch` ``(step, epoch=0)`` ``-> dict(indices, x
  (B, 1, 28, 28), y, pixels)``; :func:`first_batch`;
* the model: :func:`init_params`, :func:`forward` ``(P, x) -> dict(x, z1, h1, z2, h2,
  logits)``, :func:`loss_of`, :func:`loss_and_grads` ``(P, x, y) -> (loss, grads)``,
  :func:`sgd_update`, :func:`adam_update`, :func:`count` (52,650);
* what the script prints: :data:`PRINTED_MODEL` (``print(model)``, region
  ``print-model``) and :data:`PRINTED_GRAD` (``print(model[1].weight.grad.shape)``,
  region ``grad``);
* the recorded runs (:data:`RUNS`): :func:`losses` ``(run)`` (938 per-step losses),
  :func:`heldout` ``(run) -> dict(steps, accuracy, loss)`` on the 10,000 test digits,
  :func:`params` ``(step, run)`` (the weights *before* step ``step``; ``EPOCH_STEPS``: the
  end), :func:`grads` ``(step, run)`` (the gradient computed at that step),
  :func:`adam_state` ``(step) -> dict(m, v)`` after ``step`` updates, :func:`grad_norms`,
  :func:`more_epochs` (two more epochs of the SGD run), :func:`activations` ``(step,
  batch_step=0)``, :func:`roundtrip` (save and load), :func:`summary` (the headline
  numbers, JSON-able).

The runs, each from the same initial weights and the same batches:

* ``sgd``: the script as it is (and two more epochs after it, :func:`more_epochs`);
* ``adam``: the script with ``adam``;
* ``lr_small``, ``lr_large``: SGD at :data:`RATES` 0.001 and 0.8 (``sgd`` is 0.1);
* ``no_zero_grad``: ``optimizer.zero_grad()`` left out, so ``.grad`` adds up;
* ``dropout``: the model with ``nn.Dropout(0.2)`` after each ReLU, scored at test with
  and without ``model.eval()`` (examples/blocks_parts.py, region ``dropout-model``);
* ``double_softmax``: ``nn.Softmax(dim=1)`` at the end of the model, so the loss takes
  probabilities (region ``mistake-double-softmax``).

The runs are built once (a few seconds) and cached by :func:`voxanim.cache.cached_npz`,
keyed by every setting and this module's source.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import io
import itertools
import sys

import numpy as np

from . import mnist
from .cache import cached_npz

VERSION = 1
SIZES = (784, 64, 32, 10)
SIDE = 28
CLASSES = 10
TRAIN, TEST = 60_000, 10_000
BATCH = 64
EPOCH_STEPS = -(-TRAIN // BATCH)            # 938: 937 batches of 64 and one of 32
LAST_BATCH = TRAIN - (EPOCH_STEPS - 1) * BATCH
LR = 0.1                                    # SGD(model.parameters(), lr=0.1)
ADAM_LR, BETAS, EPS = 1e-3, (0.9, 0.999), 1e-8     # Adam's defaults
MEAN, STD = 0.1307, 0.3081                  # Normalize((0.1307,), (0.3081,))
SEED = 185
RATES = (0.001, 0.1, 0.8)                   # the learning-rate dial: crawls, learns, bounces
DROPOUT = 0.2
MORE_EPOCHS = 2                             # the SGD run continued, for the epoch animations
# The keys of model.state_dict(): nn.Sequential numbers its blocks 0 Flatten, 1 Linear,
# 2 ReLU, 3 Linear, 4 ReLU, 5 Linear; only the Linear blocks hold numbers.
LINEAR = (1, 3, 5)
KEYS = tuple("%d.%s" % (i, w) for i in LINEAR for w in ("weight", "bias"))
SNAP_STEPS = (0, 1, 2, 3, 10, 100, 500, 937)       # weights and gradients kept at these
ADAM_STEPS = (1, 2, 10, 100, EPOCH_STEPS)          # Adam's m and v kept after these
EVAL_STEPS = tuple(range(0, EPOCH_STEPS, 25)) + (EPOCH_STEPS,)
# What the script's print lines write (checked against torch by
# tests/test_mnist_blocks_equivalence.py): region "print-model", then region "grad"
PRINTED_MODEL = """Sequential(
  (0): Flatten(start_dim=1, end_dim=-1)
  (1): Linear(in_features=784, out_features=64, bias=True)
  (2): ReLU()
  (3): Linear(in_features=64, out_features=32, bias=True)
  (4): ReLU()
  (5): Linear(in_features=32, out_features=10, bias=True)
)"""
PRINTED_GRAD = "torch.Size([%d, %d])" % (SIZES[1], SIZES[0])     # model[1].weight.grad.shape
RUNS = ("sgd", "adam", "lr_small", "lr_large", "no_zero_grad", "dropout", "double_softmax")
SCRIPT = "examples/mnist_blocks.py"


# ====================================================================== data
@functools.lru_cache(maxsize=1)
def pixels():
    """``(x, xt)``: every training and test digit as ``ToTensor`` gives it, float32 pixels
    in 0..1, ``(N, 784)`` in row order."""
    x, _y, xt, _yt = mnist.load()
    return x.astype(np.float32), xt.astype(np.float32)


def normalise(p):
    """``Normalize((MEAN,), (STD,))`` of pixels ``p``, in float32 as PyTorch does it."""
    return (np.asarray(p, np.float32) - np.float32(MEAN)) / np.float32(STD)


@functools.lru_cache(maxsize=1)
def data():
    """``(x, y, xt, yt)``: the normalised training and test digits ``(N, 784)`` float32
    and their labels (int64)."""
    x, xt = pixels()
    _x, y, _xt, yt = mnist.load()
    return normalise(x), y.astype(np.int64), normalise(xt), yt.astype(np.int64)


def pixel_stats():
    """The mean and spread (population standard deviation) of every training pixel, which
    ``Normalize`` subtracts and divides by, and the script's rounded constants."""
    x = pixels()[0].astype(np.float64)
    return dict(mean=float(x.mean()), std=float(x.std()), script_mean=MEAN, script_std=STD,
                low=float(normalise(0.0)), high=float(normalise(1.0)))


def digit(i=0):
    """Training digit ``i`` as ``train_set[i]`` gives it: ``(x (1, 28, 28) float32, y)``."""
    x, y = data()[:2]
    return x[i].reshape(1, SIDE, SIDE), int(y[i])


def order(epoch=0):
    """The shuffled order of the 60,000 training digits in ``epoch`` (0-based)."""
    return np.random.default_rng([SEED, 1, int(epoch)]).permutation(TRAIN)


def batch_indices(step, epoch=0):
    """The training indices of batch ``step`` (0..937) of ``epoch``."""
    return order(epoch)[step * BATCH:(step + 1) * BATCH]


def batch(step=0, epoch=0):
    """Batch ``step`` of ``epoch`` as ``for x, y in loader`` gives it: ``dict(indices,
    x (B, 1, 28, 28) normalised, y (B,), pixels (B, 784) 0..1)``."""
    idx = batch_indices(step, epoch)
    x, y = data()[:2]
    return dict(indices=idx, x=x[idx].reshape(-1, 1, SIDE, SIDE), y=y[idx],
                pixels=pixels()[0][idx])


def first_batch():
    """The first batch of the first epoch (what ``next(iter(loader))`` gives)."""
    return batch(0, 0)


# ====================================================================== the model
def init_params(seed=SEED):
    """The model's parameters as PyTorch initialises ``nn.Linear``: weight ``[out, in]``
    and bias ``[out]``, both uniform in ``±1/sqrt(in)`` (kaiming_uniform with a = sqrt 5
    gives that bound for the weights), float32, keyed as ``model.state_dict()``."""
    rng = np.random.default_rng([SEED, 0, int(seed)])
    P = {}
    for i, (a, b) in zip(LINEAR, itertools.pairwise(SIZES)):
        bound = 1.0 / np.sqrt(a)
        P["%d.weight" % i] = rng.uniform(-bound, bound, (b, a)).astype(np.float32)
        P["%d.bias" % i] = rng.uniform(-bound, bound, b).astype(np.float32)
    return P


def count(P=None):
    """Every number the model learns (``sum(p.numel() for p in model.parameters())``)."""
    P = init_params() if P is None else P
    return int(sum(np.asarray(v).size for v in P.values()))


def _flat(x):
    """``nn.Flatten()``: ``[B, 1, 28, 28] -> [B, 784]``, row by row."""
    x = np.asarray(x)
    return x.reshape(len(x), -1)


def _masks(step, rows, sizes=SIZES[1:-1], p=DROPOUT, salt=0):
    """The dropout masks of one training step, already scaled by ``1 / (1 - p)`` as
    ``nn.Dropout`` scales what it keeps."""
    rng = np.random.default_rng([SEED, 7, int(salt), int(step)])
    return [(rng.random((rows, n)) >= p).astype(np.float32) / np.float32(1 - p) for n in sizes]


def forward(P, x, masks=None, softmax=False):
    """One pass through the model: ``dict(x, z1, h1, z2, h2, logits)`` (``x`` flattened,
    ``z``: a Linear's output, ``h``: after ReLU, and after dropout when ``masks`` are
    given); ``softmax``: the model ends in ``nn.Softmax(dim=1)`` (``out``: its output)."""
    a = dict(x=_flat(x))
    h = a["x"]
    for k, i in enumerate(LINEAR):
        z = h @ P["%d.weight" % i].T + P["%d.bias" % i]
        if k == len(LINEAR) - 1:
            a["logits"] = z
            break
        a["z%d" % (k + 1)] = z
        h = np.maximum(z, 0).astype(z.dtype)
        if masks is not None:
            h = h * masks[k]
        a["h%d" % (k + 1)] = h
    a["out"] = _softmax(a["logits"]) if softmax else a["logits"]
    return a


def _softmax(z):
    z = np.asarray(z)
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def log_softmax(z):
    z = np.asarray(z)
    m = z.max(1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(1, keepdims=True))


def loss_each(scores, y):
    """``nn.CrossEntropyLoss(reduction="none")``: ``-log softmax(scores)[y]`` per digit."""
    return -log_softmax(scores)[np.arange(len(y)), np.asarray(y)]


def loss_of(P, x, y, softmax=False, masks=None):
    """The mean cross-entropy of a batch, as ``loss_fn(model(x), y)`` gives it."""
    return float(loss_each(forward(P, x, masks, softmax)["out"], y).mean())


def loss_and_grads(P, x, y, masks=None, softmax=False):
    """``loss.backward()``: the batch's mean cross-entropy and its gradient for every
    parameter (same keys and shapes as ``P``)."""
    a = forward(P, x, masks, softmax)
    y = np.asarray(y)
    n = len(y)
    rows = np.arange(n)
    q = _softmax(a["out"])
    loss = float(loss_each(a["out"], y).mean())
    d = q.copy()
    d[rows, y] -= 1.0
    d /= d.dtype.type(n)
    if softmax:                        # back through the extra softmax: p * (d - <d, p>)
        p = a["out"]
        d = p * (d - (d * p).sum(1, keepdims=True))
    G = {}
    hs = [a["x"], a["h1"], a["h2"]]
    zs = [None, a["z1"], a["z2"]]
    for k in reversed(range(len(LINEAR))):
        i = LINEAR[k]
        G["%d.weight" % i] = d.T @ hs[k]
        G["%d.bias" % i] = d.sum(0)
        if k:
            d = d @ P["%d.weight" % i]
            if masks is not None:
                d = d * masks[k - 1]
            d = d * (zs[k] > 0)
    return loss, {k: G[k].astype(P[k].dtype) for k in P}


def sgd_update(P, G, lr=LR):
    """``optimizer.step()`` of ``SGD``: every number moves by ``-lr`` times its gradient."""
    return {k: (P[k] - P[k].dtype.type(lr) * G[k]).astype(P[k].dtype) for k in P}


def adam_update(P, G, state, lr=ADAM_LR, betas=BETAS, eps=EPS):
    """``optimizer.step()`` of ``Adam``, in PyTorch's order: ``state`` is ``dict(t, m, v)``
    (``t`` updates so far); returns the new parameters and the new state."""
    b1, b2 = betas
    t = state["t"] + 1
    m = {k: state["m"][k] + (G[k] - state["m"][k]) * np.float32(1 - b1) for k in P}
    v = {k: state["v"][k] * np.float32(b2) + np.float32(1 - b2) * G[k] * G[k] for k in P}
    step = lr / (1 - b1 ** t)
    root = np.sqrt(1 - b2 ** t)
    new = {k: (P[k] - np.float32(step) * m[k] / (np.sqrt(v[k]) / np.float32(root)
                                                + np.float32(eps))).astype(np.float32)
           for k in P}
    return new, dict(t=t, m=m, v=v)


def adam_start(P):
    """Adam's state before its first step: no steps, ``m`` and ``v`` zero."""
    return dict(t=0, m={k: np.zeros_like(v) for k, v in P.items()},
                v={k: np.zeros_like(v) for k, v in P.items()})


def score(P, softmax=False, masks=None, chunk=2000):
    """Held-out ``(accuracy, mean loss)`` on the 10,000 test digits; ``masks(rows, start)``
    gives dropout masks (a model left in training mode)."""
    _x, _y, xt, yt = data()
    right, total = 0, 0.0
    for s in range(0, len(xt), chunk):
        m = None if masks is None else masks(min(chunk, len(xt) - s), s)
        out = forward(P, xt[s:s + chunk], m, softmax)["out"]
        right += int((out.argmax(1) == yt[s:s + chunk]).sum())
        total += float(loss_each(out, yt[s:s + chunk]).astype(np.float64).sum())
    return right / len(xt), total / len(xt)


# ====================================================================== the runs
def _train(kind, lr=LR, epochs=1, record=True):
    """One run of the script, or of one of its variants, from :func:`init_params` over the
    same batches. Returns a dict of arrays: per-step ``loss``, the held-out curve, the
    gradient norms, and the snapshots the animations use."""
    P = init_params()
    x, y = data()[:2]
    adam = kind == "adam"
    state = adam_start(P) if adam else None
    acc_grad = {k: np.zeros_like(v) for k, v in P.items()}
    softmax = kind == "double_softmax"
    dropout = kind == "dropout"
    out = dict(loss=[], grad_norm=[], used_norm=[], eval_steps=[], accuracy=[], test_loss=[])
    if dropout:
        out.update(accuracy_train_mode=[], test_loss_train_mode=[])

    def evaluate(step):
        out["eval_steps"].append(step)
        a, l = score(P, softmax)
        out["accuracy"].append(a)
        out["test_loss"].append(l)
        if dropout:
            a, l = score(P, masks=lambda rows, s: _masks(s, rows, salt=10_000 + step))
            out["accuracy_train_mode"].append(a)
            out["test_loss_train_mode"].append(l)

    step = 0
    for epoch in range(epochs):
        idx_all = order(epoch)
        for b in range(EPOCH_STEPS):
            if epoch == 0 and step in EVAL_STEPS:
                evaluate(step)
            idx = idx_all[b * BATCH:(b + 1) * BATCH]
            masks = _masks(step, len(idx)) if dropout else None
            loss, G = loss_and_grads(P, x[idx], y[idx], masks, softmax)
            if record and epoch == 0 and step in SNAP_STEPS:
                for k in P:
                    out["W%d__%s" % (step, k)] = P[k].copy()
                    out["G%d__%s" % (step, k)] = G[k]
            if kind == "no_zero_grad":
                acc_grad = {k: acc_grad[k] + G[k] for k in P}
                used = acc_grad
            else:
                used = G
            out["loss"].append(loss)
            out["grad_norm"].append(float(np.sqrt(sum((G[k].astype(np.float64) ** 2).sum()
                                                      for k in P))))
            out["used_norm"].append(float(np.sqrt(sum((used[k].astype(np.float64) ** 2).sum()
                                                      for k in P))))
            if adam:
                P, state = adam_update(P, used, state)
                if record and state["t"] in ADAM_STEPS:
                    for k in P:
                        out["M%d__%s" % (state["t"], k)] = state["m"][k]
                        out["V%d__%s" % (state["t"], k)] = state["v"][k]
            else:
                P = sgd_update(P, used, lr)
            step += 1
            if not all(np.isfinite(v).all() for v in P.values()):
                raise FloatingPointError("%s: the weights stopped being finite at step %d"
                                         % (kind, step))
        if epoch == 0:
            evaluate(step)
            if record:
                for k in P:
                    out["W%d__%s" % (EPOCH_STEPS, k)] = P[k].copy()
        else:
            a, l = score(P, softmax)
            out.setdefault("epoch_accuracy", []).append(a)
            out.setdefault("epoch_test_loss", []).append(l)
    return {k: np.asarray(v) for k, v in out.items()}


def _build():
    runs = dict(sgd=("sgd", LR, 1 + MORE_EPOCHS), adam=("adam", ADAM_LR, 1),
                lr_small=("sgd", RATES[0], 1), lr_large=("sgd", RATES[2], 1),
                no_zero_grad=("no_zero_grad", LR, 1), dropout=("dropout", LR, 1),
                double_softmax=("double_softmax", LR, 1))
    out = {}
    for name, (kind, lr, epochs) in runs.items():
        r = _train(kind, lr, epochs, record=name in ("sgd", "adam"))
        for k, v in r.items():
            out["%s__%s" % (name, k)] = v
    return out


def _settings():
    return dict(version=VERSION, sizes=SIZES, batch=BATCH, lr=LR, adam=(ADAM_LR, BETAS, EPS),
                mean=MEAN, std=STD, seed=SEED, rates=RATES, dropout=DROPOUT,
                more=MORE_EPOCHS, snaps=SNAP_STEPS, adam_steps=ADAM_STEPS, evals=EVAL_STEPS,
                source=hashlib.sha256(inspect.getsource(sys.modules[__name__]).encode())
                .hexdigest()[:16])


@functools.lru_cache(maxsize=1)
def runs():
    """Every recorded run, as one flat dict of arrays ``"<run>__<what>"`` (cached)."""
    return {k: np.asarray(v) for k, v in cached_npz("study-course", _settings(),
                                                    _build).items()}


def _run(name):
    if name not in RUNS:
        raise KeyError("no run %r: the runs are %s" % (name, ", ".join(RUNS)))
    p = name + "__"
    return {k[len(p):]: v for k, v in runs().items() if k.startswith(p)}


def losses(run="sgd"):
    """The loss of every step of the first epoch, as ``loss.item()`` reads it (938)."""
    return _run(run)["loss"][:EPOCH_STEPS]


def grad_norms(run="sgd"):
    """``dict(fresh, used)``: per step, the length of the batch's own gradient, and of the
    gradient the optimizer used (the same unless ``zero_grad`` was left out, when ``.grad``
    holds the sum of every step so far)."""
    r = _run(run)
    return dict(fresh=r["grad_norm"][:EPOCH_STEPS], used=r["used_norm"][:EPOCH_STEPS])


def heldout(run="sgd"):
    """The held-out curve: ``dict(steps, accuracy, loss)`` on the 10,000 test digits, every
    25 steps and at the end; the dropout run adds ``accuracy_train_mode`` and
    ``loss_train_mode`` (scored without ``model.eval()``)."""
    r = _run(run)
    out = dict(steps=r["eval_steps"], accuracy=r["accuracy"], loss=r["test_loss"])
    if "accuracy_train_mode" in r:
        out.update(accuracy_train_mode=r["accuracy_train_mode"],
                   loss_train_mode=r["test_loss_train_mode"])
    return out


def _snap(run, prefix, step):
    r = _run(run)
    got = {k: r["%s%d__%s" % (prefix, step, k)] for k in KEYS
           if "%s%d__%s" % (prefix, step, k) in r}
    if not got:
        raise KeyError("%s: nothing kept at step %s (kept: %s)" % (run, step, SNAP_STEPS))
    return got


def params(step=EPOCH_STEPS, run="sgd"):
    """The parameters before step ``step`` (0: as initialised; ``EPOCH_STEPS``: after the
    epoch), for ``step`` in :data:`SNAP_STEPS` or ``EPOCH_STEPS``; runs ``sgd`` and
    ``adam``."""
    if step == "end":
        step = EPOCH_STEPS
    if step == 0:
        return init_params()
    return _snap(run, "W", step)


def grads(step=0, run="sgd"):
    """The gradient ``loss.backward()`` leaves in ``.grad`` at step ``step`` (in
    :data:`SNAP_STEPS`), computed from :func:`params` ``(step)`` on :func:`batch` ``(step)``."""
    return _snap(run, "G", step)


def adam_state(step=1):
    """Adam's ``dict(m, v)`` (each keyed as the parameters) after ``step`` updates, for
    ``step`` in :data:`ADAM_STEPS`."""
    return dict(m=_snap("adam", "M", step), v=_snap("adam", "V", step))


def more_epochs():
    """The SGD run continued for :data:`MORE_EPOCHS` more epochs, each in a new order:
    ``dict(loss (epochs, 938), epoch_mean, accuracy, test_loss, first)``, every epoch
    including the first, ``first``: each epoch's first 8 training indices."""
    r = _run("sgd")
    loss = r["loss"].reshape(1 + MORE_EPOCHS, EPOCH_STEPS)
    return dict(loss=loss, epoch_mean=loss.mean(1),
                accuracy=np.r_[r["accuracy"][-1], r["epoch_accuracy"]],
                test_loss=np.r_[r["test_loss"][-1], r["epoch_test_loss"]],
                first=np.stack([order(e)[:8] for e in range(1 + MORE_EPOCHS)]))


def activations(step=EPOCH_STEPS, batch_step=0, run="sgd"):
    """Batch ``batch_step`` through the model with :func:`params` ``(step)``: ``dict(x,
    z1, h1, z2, h2, logits, probs, loss_each, y)``."""
    b = batch(batch_step)
    a = forward(params(step, run), b["x"])
    a.pop("out")
    a["probs"] = _softmax(a["logits"])
    a["loss_each"] = loss_each(a["logits"], b["y"])
    a["y"] = b["y"]
    return a


@functools.lru_cache(maxsize=1)
def roundtrip():
    """``torch.save(model.state_dict())`` and ``load_state_dict`` as numpy does it: the
    trained parameters written to bytes and read back. ``dict(numbers, bytes_of_numbers,
    file_bytes, max_change, same_predictions, accuracy_before, accuracy_after)``."""
    P = params(EPOCH_STEPS)
    buf = io.BytesIO()
    np.savez(buf, **P)
    buf.seek(0)
    with np.load(buf) as d:
        Q = {k: d[k] for k in d.files}
    xt = data()[2]
    before = forward(P, xt)["logits"].argmax(1)
    after = forward(Q, xt)["logits"].argmax(1)
    yt = data()[3]
    return dict(numbers=count(P), bytes_of_numbers=4 * count(P), file_bytes=len(buf.getvalue()),
                max_change=float(max(np.abs(P[k] - Q[k]).max() for k in P)),
                same_predictions=int((before == after).sum()),
                accuracy_before=float((before == yt).mean()),
                accuracy_after=float((after == yt).mean()))


def summary():
    """The headline numbers of every run, JSON-able."""
    out = dict(pixel_stats=pixel_stats(), steps=EPOCH_STEPS, last_batch=LAST_BATCH,
               parameters=count())
    for name in RUNS:
        h, l = heldout(name), losses(name)
        out[name] = dict(first_loss=float(l[0]), last_loss=float(l[-1]),
                         mean_last_100=float(l[-100:].mean()),
                         accuracy=float(h["accuracy"][-1]), test_loss=float(h["loss"][-1]))
        if "accuracy_train_mode" in h:
            out[name]["accuracy_train_mode"] = float(h["accuracy_train_mode"][-1])
    return out

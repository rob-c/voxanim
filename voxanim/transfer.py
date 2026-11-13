"""Fine-tuning and transfer, in plain numpy, on split MNIST.

**The split.**  MNIST is cut into two tasks: the *old* digits 0-4 (30,596
training images) and the *new* digits 5-9 (29,404).  A **few-shot** set
takes :data:`N_FEW` training images of each new digit, drawn once with a
seed; a **rehearsal buffer** takes :data:`N_BUFFER` of each old digit.  Test
accuracy during a run is measured on a fixed :data:`EVAL_PER_DIGIT` images
of each digit (the *eval set*), the final numbers on the whole test set.

**The network** is a CNN with two convolutional layers, so that there is a
*feature stack* worth transferring::

    x 28x28 -> conv 3x3, F1 -> ReLU -> pool 2x2 -> conv 3x3, F2 -> ReLU -> pool 2x2 -> dense 10

(:data:`FILTERS`; 26x26 -> 13x13 -> 11x11 -> 5x5, so the dense layer reads
400 values; He-normal initialisation, cross-entropy, Adam).  The first
layer's filters are what the animations show before and after.  The
backward pass is checked against finite differences in
``tests/test_transfer.py``.  **Pretraining** (:func:`pretrain`) runs one
epoch on the old digits with a 10-way head, so the new digits' outputs exist
but have only ever been pushed down.

*Why two conv layers.*  With the one-conv CNN of :mod:`voxanim.mnist` the
36 frozen conv weights are too few to matter (frozen and scratch tie within
a point), and a frozen *dense* hidden layer trained on 0-4 loses 13 points
to scratch: its units only know what told 0-4 apart.  Two conv layers are
generic (strokes, edges) and hard to learn from 500 images: frozen beats
scratch in every seed tried.

**Per-layer learning rates and freezing.**  :class:`PerLayerAdam` is Adam
(the update of :mod:`voxanim.optim`, same literals) with a learning rate per
parameter; a rate of 0 *freezes* the parameter: it is skipped, so its array
is bit for bit what it was.

**Three ways to learn the new digits from 100 a class**
(:func:`transfer_runs`): *scratch* (a fresh network with a 5-way head),
*frozen* (the two pretrained conv layers frozen, a fresh 5-way head
trained), *fine-tune* (the pretrained layers at :data:`LOW` of the rate, the
fresh head at the full rate).  Each run records the test accuracy on the new
digits and the first-layer filters at every recorded step.

**Catastrophic forgetting** (:func:`forgetting_runs`): the pretrained 10-way
network is trained on the new digits, in three ways: *plain*; with
*rehearsal*, :data:`REHEARSE` of every :data:`FT_BATCH`-image minibatch
drawn from the buffer of old digits; and with *lower rates on the early
layers* (:data:`FT_LOW`: the conv layers at a fiftieth of the rate, the head
at a fifth; the head must slow down too, since at the full rate it forgets
on its own, and the conv layers at a tenth alone change nothing).  Each run
records the
per-digit test accuracy and the first images of the minibatch at every
recorded step.

**The learning-rate finder and warm restarts** (on the MLP of
:mod:`voxanim.mnist`, 784-64-32-10, Adam): :func:`lr_range_test` raises
the learning rate geometrically from :data:`LR_LO` to :data:`LR_HI` over
:data:`LR_STEPS` minibatches and records the loss of each; :func:`lr_analysis`
smooths it and finds the *knee* (the steepest fall against log learning
rate, before the minimum), the *minimum* and the *divergence* (the first
rate past the minimum at which the loss is back above where it started).
:func:`restart_runs` then trains for :data:`EPOCHS` epochs with a cosine
schedule restarted every epoch, peak at the knee, against one cosine over the
same steps; :func:`landscape_restarts` runs minibatch SGD with the same shape
of schedule on the two-weight landscape of :mod:`voxanim.landscape`.  Only
noise can take a walker out of a minimum (full-batch descent and Adam, whose
second moment remembers the large early gradients, just sit there when the
rate jumps), so the landscape run is minibatch SGD with a peak high enough
for the kick to show.

Everything is seeded and cached by :func:`voxanim.cache.cached_npz` under
its settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np

from . import landscape as LS
from .cache import cached_npz
from .minibatch import minibatches
from .mnist import _as_layers, _forward, _init, _mlp_grads, load
from .optim import BETA1, BETA2, EPS, GAIN1, GAIN2, SGD, Adam

OLD = (0, 1, 2, 3, 4)            # the first task
NEW = (5, 6, 7, 8, 9)            # the second
N_FEW = 100                      # training images of each new digit in the few-shot set
N_BUFFER = 100                   # of each old digit in the rehearsal buffer
EVAL_PER_DIGIT = 250             # test images of each digit the recorded accuracies use
SPLIT_SEED = 0

FILTERS = (8, 16)                # conv filters of the two layers
HW = ((28, 26, 13), (13, 11, 5))     # each conv layer's (input, conv, pooled) side
FLAT = HW[1][2] ** 2 * FILTERS[1]    # 400 values into the dense layer
CLASSES = 10
PRE = dict(epochs=1, batch=64, lr=2e-3, seed=0)                # pretraining on the old digits
LR = 2e-3                        # the full learning rate of every run below
LOW = 0.1                        # the early layers' share of it when "lower"
PRE_EVERY = 15                   # steps between recorded filters while pretraining
STEPS, BATCH, EVERY = 200, 50, 20                               # the few-shot runs
FT_STEPS, FT_BATCH, REHEARSE = 200, 64, 16                      # the forgetting runs
FT_EVERY, FT_DENSE, FT_SPARSE = 10, 100, 25   # recorded every 10 steps to 100, then every 25
FT_LOW = (0.02, 0.2)             # "lower rates": the conv layers' and the head's share of LR
N_SHOWN = 8                      # minibatch images recorded at each step, for the animation
EARLY = ("W1", "b1", "W2", "b2")                                # the pretrained conv layers
HEAD = ("Wo", "bo")

# the learning-rate finder and the restarts (the MLP of voxanim.mnist)
LR_LO, LR_HI, LR_STEPS, LR_BATCH = 1e-5, 1.0, 300, 128
SMOOTH = 21                      # centred window of the smoothed range-test loss
EPOCHS, CYCLES = 3, 3            # warm restarts: one cosine per epoch
PER_EPOCH = 469                  # minibatches of 128 in an epoch
SUB_EVERY, SUB_N = 10, 2000      # the training loss on a fixed subset, every so many steps
LAND_STEPS, LAND_CYCLES, LAND_PEAK, LAND_BATCH, LAND_SEED = 180, 3, 8.0, 4, 6
LS_START = (2.5, -2.0)           # where every walk on the round bowl starts (optim_anim.START)


# ------------------------------------------------------------------ data


@dataclass
class Split:
    """Split MNIST (images ``(N, 784)`` in [0, 1], integer labels 0-9)."""

    x_old: np.ndarray            # every training image of an old digit
    y_old: np.ndarray
    x_new: np.ndarray            # every training image of a new digit
    y_new: np.ndarray
    x_few: np.ndarray            # N_FEW of each new digit
    y_few: np.ndarray            # ... labelled 0-4 (the new digit minus 5)
    x_buffer: np.ndarray         # N_BUFFER of each old digit, for rehearsal
    y_buffer: np.ndarray
    x_eval: np.ndarray           # EVAL_PER_DIGIT test images of each digit
    y_eval: np.ndarray
    x_test: np.ndarray           # the whole test set
    y_test: np.ndarray

    def settings(self):
        """What identifies the split in a cache key."""
        return dict(old=list(OLD), new=list(NEW), few=N_FEW, buffer=N_BUFFER,
                    eval_per_digit=EVAL_PER_DIGIT, seed=SPLIT_SEED,
                    checksum=float(self.x_few.sum() + self.x_buffer.sum() + self.x_eval.sum()))


def _per_digit(rng, y, digits, n):
    """``n`` indices of each digit in ``digits``, drawn without replacement."""
    return np.concatenate([rng.choice(np.nonzero(y == d)[0], n, replace=False) for d in digits])


def split(cache_dir=None) -> Split:
    """The split of the module docstring, from one seeded draw."""
    xtr, ytr, xte, yte = load(cache_dir)
    rng = np.random.default_rng(SPLIT_SEED)
    old, new = np.isin(ytr, OLD), np.isin(ytr, NEW)
    few = _per_digit(rng, ytr, NEW, N_FEW)
    buf = _per_digit(rng, ytr, OLD, N_BUFFER)
    ev = _per_digit(rng, yte, range(CLASSES), EVAL_PER_DIGIT)
    return Split(xtr[old], ytr[old], xtr[new], ytr[new], xtr[few], ytr[few] - NEW[0],
                 xtr[buf], ytr[buf], xte[ev], yte[ev], xte, yte)


# ------------------------------------------------------------------ the network


def im2col(x):
    """``(N, H, W, C)`` -> ``(N, H-2, W-2, 9C)``: every 3x3 patch, channel-major
    (channel, then the 3x3 taps row-major): the layout numpy copies fastest."""
    s = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(1, 2))    # N,H-2,W-2,C,3,3
    n, h, w, c = s.shape[:4]
    return s.reshape(n, h, w, 9 * c)


def _wmat(W):
    """``W (3, 3, C, F)`` as the ``(9C, F)`` matrix that matches :func:`im2col`."""
    return W.transpose(2, 0, 1, 3).reshape(-1, W.shape[-1])


def conv(x, W, b):
    """Valid 3x3 convolution of ``x (N, H, W, C)`` with ``W (3, 3, C, F)``:
    ``(patches, z)``, ``z`` ``(N, H-2, W-2, F)``."""
    cols = im2col(x)
    return cols, cols @ _wmat(W) + b


def pool(a, winners=True):
    """2x2 max-pool of ``a (N, H, W, C)`` (an odd last row or column is dropped):
    ``(arg, pooled)``, ``arg`` which cell of each window won, in the order
    ``(0,0), (0,1), (1,0), (1,1)`` (None unless ``winners``: evaluation skips it)."""
    h2, w2 = a.shape[1] // 2, a.shape[2] // 2
    a = a[:, :2 * h2, :2 * w2]
    rows = np.maximum(a[:, 0::2], a[:, 1::2])              # two stages: fewer strided reads
    pooled = np.maximum(rows[:, :, 0::2], rows[:, :, 1::2])
    if not winners:
        return None, pooled
    cells = (a[:, 0::2, 0::2], a[:, 0::2, 1::2], a[:, 1::2, 0::2], a[:, 1::2, 1::2])
    return np.stack(cells, -1).argmax(-1), pooled


def unpool(arg, d, h, w):
    """Route the pooled gradient ``d (N, H/2, W/2, C)`` back to each window's
    winner, in an ``(N, h, w, C)`` array (dropped rows and columns get 0)."""
    n, h2, w2, c = d.shape
    route = np.eye(4)[arg] * d[..., None]
    route = route.reshape(n, h2, w2, c, 2, 2).transpose(0, 1, 4, 2, 5, 3)
    out = np.zeros((n, h, w, c))
    out[:, :2 * h2, :2 * w2] = route.reshape(n, 2 * h2, 2 * w2, c)
    return out


def col2im(dcols, h, w, c):
    """Patch gradients ``(N, h-2, w-2, 9c)`` (the layout of :func:`im2col`)
    summed back onto the input ``(N, h, w, c)``."""
    n = dcols.shape[0]
    d = dcols.reshape(n, h - 2, w - 2, c, 3, 3)
    out = np.zeros((n, h, w, c))
    for i in range(3):
        for j in range(3):
            out[:, i:i + h - 2, j:j + w - 2] += d[:, :, :, :, i, j]
    return out


def init(rng, classes=CLASSES, filters=FILTERS):
    """He-normal weights, zero biases: ``W1 (3, 3, 1, F1), b1, W2 (3, 3, F1, F2),
    b2, Wo (flat, classes), bo``."""
    f1, f2 = filters
    flat = HW[1][2] ** 2 * f2
    return dict(W1=rng.normal(0, np.sqrt(2 / 9), (3, 3, 1, f1)), b1=np.zeros(f1),
                W2=rng.normal(0, np.sqrt(2 / (9 * f1)), (3, 3, f1, f2)), b2=np.zeros(f2),
                Wo=rng.normal(0, np.sqrt(2 / flat), (flat, classes)), bo=np.zeros(classes))


def forward(P, x, train=True):
    """The batch ``x (N, 784)`` through the network: a dict of every intermediate
    (``c1 z1 arg1`` and ``c2 z2 arg2`` the two conv layers' patches,
    pre-activations and pool winners, ``flat`` what the dense layer reads,
    ``p`` the softmax).  Without ``train`` the pool winners are skipped (they
    are only needed by :func:`backward`)."""
    c1, z1 = conv(x.reshape(-1, 28, 28, 1), P["W1"], P["b1"])
    arg1, p1 = pool(np.maximum(z1, 0.0), train)
    c2, z2 = conv(p1, P["W2"], P["b2"])
    arg2, p2 = pool(np.maximum(z2, 0.0), train)
    flat = p2.reshape(len(x), -1)
    zo = flat @ P["Wo"] + P["bo"]
    e = np.exp(zo - zo.max(-1, keepdims=True))
    return dict(c1=c1, z1=z1, arg1=arg1, c2=c2, z2=z2, arg2=arg2, flat=flat,
                p=e / e.sum(-1, keepdims=True))


def cross_entropy(p, y_onehot):
    """Mean ``-log p[label]`` over the batch."""
    return float(-np.mean(np.log(np.sum(p * y_onehot, 1) + 1e-300)))


def backward(P, f, y_onehot):
    """Gradients of the mean cross-entropy, with the parameters' keys."""
    f1, f2 = P["W1"].shape[-1], P["W2"].shape[-1]
    (_, h1, q1), (_, h2, q2) = HW
    do = (f["p"] - y_onehot) / len(y_onehot)
    G = dict(Wo=f["flat"].T @ do, bo=do.sum(0))
    dp2 = (do @ P["Wo"].T).reshape(-1, q2, q2, f2)
    d2 = unpool(f["arg2"], dp2, h2, h2) * (f["z2"] > 0)                  # pool, ReLU
    # the patch matrices are channel-major (im2col), so a (9C, F) gradient
    # comes out as (C, 3, 3, F) and is put back in the weights' (3, 3, C, F) order
    G["W2"] = (f["c2"].reshape(-1, 9 * f1).T @ d2.reshape(-1, f2)).reshape(f1, 3, 3, f2)
    G["W2"] = G["W2"].transpose(1, 2, 0, 3)
    G["b2"] = d2.sum((0, 1, 2))
    dcols = (d2.reshape(-1, f2) @ _wmat(P["W2"]).T).reshape(-1, h2, h2, 9 * f1)
    dp1 = col2im(dcols, q1, q1, f1)
    d1 = unpool(f["arg1"], dp1, h1, h1) * (f["z1"] > 0)
    G["W1"] = (f["c1"].reshape(-1, 9).T @ d1.reshape(-1, f1)).reshape(3, 3, 1, f1)
    G["b1"] = d1.sum((0, 1, 2))
    return G


def grads(P, x, y_onehot):
    """Forward and backward in one call."""
    return backward(P, forward(P, x), y_onehot)


def predict(P, x, chunk=500):
    """The predicted digit of every row of ``x``, in chunks."""
    return np.concatenate([forward(P, x[s:s + chunk], train=False)["p"].argmax(1)
                           for s in range(0, len(x), chunk)])


def per_digit_accuracy(P, x, y, classes=CLASSES):
    """``(classes,)``: the fraction of each digit's images predicted right."""
    pred = predict(P, x)
    return np.array([(pred[y == d] == d).mean() if (y == d).any() else np.nan
                     for d in range(classes)])


def filters(W1):
    """The first layer's filters ``(..., 3, 3, 1, F1)`` as ``(..., 3, 3, F1)``."""
    return np.asarray(W1)[..., 0, :]


# ------------------------------------------------------------------ the optimiser


class PerLayerAdam:
    """Adam over a dict of parameters with a learning rate per key (``lrs``);
    a rate of 0 freezes that parameter: it is not touched at all.

    The update is :class:`voxanim.optim.Adam`'s, literal for literal, so
    with every rate equal the two agree bit for bit.
    """

    def __init__(self, params, lrs):
        self.lrs = dict(lrs)
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params, grads):
        """One update of every unfrozen key in ``params``."""
        self.t += 1
        t = self.t
        for k in params:
            lr = self.lrs[k]
            if lr == 0:
                continue
            g = grads[k]
            self.m[k] = BETA1 * self.m[k] + GAIN1 * g
            self.v[k] = BETA2 * self.v[k] + GAIN2 * g ** 2
            params[k] = params[k] - lr * (self.m[k] / (1 - BETA1 ** t)) / (
                np.sqrt(self.v[k] / (1 - BETA2 ** t)) + EPS)


def rates(early, head):
    """``{key: lr}`` with the pretrained layers at ``early`` and the head at ``head``."""
    return {**dict.fromkeys(EARLY, early), **dict.fromkeys(HEAD, head)}


# ------------------------------------------------------------------ pretraining


def _settings(data, **kw):
    return dict(model="cnn2", filters=list(FILTERS), init="he_normal", optimizer="adam",
                data=data.settings(), **kw)


@dataclass
class Pretrained:
    """The 10-way network after :data:`PRE` on the old digits."""

    P: dict                      # the trained parameters
    acc: np.ndarray              # (10,) per-digit test accuracy: the new digits' five are 0
    steps: np.ndarray            # (S,) steps at which the filters were recorded (the last: the end)
    W1: np.ndarray               # (S, 3, 3, F1) the first layer's filters at those steps

    @property
    def old_acc(self):
        """Mean test accuracy over the old digits."""
        return float(self.acc[list(OLD)].mean())


def pretrain(data=None, cache_dir=None) -> Pretrained:
    """Train the 10-way network for :data:`PRE` on the old digits (or load it)."""
    data = data or split(cache_dir)

    def build():
        rng = np.random.default_rng(PRE["seed"])
        P = init(rng)
        opt = PerLayerAdam(P, dict.fromkeys(P, PRE["lr"]))
        eye = np.eye(CLASSES)
        steps, W1s = [], []
        for k, idx in enumerate(minibatches(rng, len(data.x_old), PRE["epochs"], PRE["batch"])):
            if k % PRE_EVERY == 0:
                steps.append(k)
                W1s.append(filters(P["W1"]))
            opt.step(P, grads(P, data.x_old[idx], eye[data.y_old[idx]]))
        steps.append(k + 1)
        W1s.append(filters(P["W1"]))
        return dict(P, acc=per_digit_accuracy(P, data.x_test, data.y_test), steps=np.array(steps),
                    W1_steps=np.array(W1s))

    d = cached_npz("transfer_pretrain", _settings(data, **PRE, task="old", every=PRE_EVERY), build,
                   cache_dir)
    meta = ("acc", "steps", "W1_steps")
    return Pretrained({k: v for k, v in d.items() if k not in meta}, d["acc"], d["steps"],
                      d["W1_steps"])


# ------------------------------------------------------------------ transfer: three ways


@dataclass
class TransferRun:
    """One few-shot run on the new digits (``S`` recorded steps)."""

    steps: np.ndarray            # (S,)
    acc: np.ndarray              # (S,) accuracy on the eval set's new digits
    W1: np.ndarray               # (S, 3, 3, F1) the first layer's filters at each recorded step
    test_acc: float              # on every test image of a new digit, at the end
    P: dict                      # the final parameters


WAYS = ("scratch", "frozen", "finetune")


def _record_steps(steps, every):
    return np.unique(np.concatenate([np.arange(0, steps, every), [steps]]))


def transfer_runs(data=None, cache_dir=None):
    """``{way: TransferRun}`` for the three ways of :data:`WAYS` (seed 0)."""
    data = data or split(cache_dir)
    pre = pretrain(data, cache_dir).P
    marks = _record_steps(STEPS, EVERY)
    new = np.isin(data.y_eval, NEW)
    x_ev, y_ev = data.x_eval[new], data.y_eval[new] - NEW[0]
    test = np.isin(data.y_test, NEW)
    x_te, y_te = data.x_test[test], data.y_test[test] - NEW[0]
    n_new = len(NEW)

    def one(way, seed=0):
        head = init(np.random.default_rng(100 + seed), n_new)
        if way == "scratch":
            P, lrs = head, rates(LR, LR)
        else:
            P = {**{k: pre[k].copy() for k in EARLY}, "Wo": head["Wo"], "bo": head["bo"]}
            lrs = rates(0.0, LR) if way == "frozen" else rates(LOW * LR, LR)
        opt = PerLayerAdam(P, lrs)
        rng = np.random.default_rng(seed)
        eye = np.eye(n_new)
        want = set(marks.tolist())
        accs, W1s = [], []
        per_epoch = -(-len(data.x_few) // BATCH)
        for k, idx in enumerate(minibatches(rng, len(data.x_few), -(-(STEPS + 1) // per_epoch),
                                            BATCH)):
            if k in want:
                accs.append(float((predict(P, x_ev) == y_ev).mean()))
                W1s.append(filters(P["W1"]))
            if k == STEPS:
                break
            opt.step(P, grads(P, data.x_few[idx], eye[data.y_few[idx]]))
        test_acc = float((predict(P, x_te) == y_te).mean())
        return dict(acc=np.array(accs), W1=np.array(W1s), test_acc=np.array(test_acc),
                    **{"P_" + k: v for k, v in P.items()})

    def build():
        out = {"steps": marks}
        for way in WAYS:
            out.update({"%s.%s" % (way, k): v for k, v in one(way).items()})
        return out

    d = cached_npz("transfer_fewshot", _settings(data, pre=PRE, steps=STEPS, batch=BATCH,
                                                 every=EVERY, lr=LR, low=LOW, ways=list(WAYS)),
                   build, cache_dir)
    runs = {}
    for way in WAYS:
        P = {k[len(way) + 3:]: v for k, v in d.items() if k.startswith(way + ".P_")}
        runs[way] = TransferRun(d["steps"], d[way + ".acc"], d[way + ".W1"],
                                float(d[way + ".test_acc"]), P)
    return runs


# ------------------------------------------------------------------ forgetting: three ways


@dataclass
class ForgettingRun:
    """The pretrained network fine-tuned on the new digits (``S`` recorded steps)."""

    steps: np.ndarray            # (S,)
    per_digit: np.ndarray        # (S, 10) accuracy on each digit's eval images
    shown: np.ndarray            # (S, N_SHOWN, 784) the first images of the minibatch, float32
    shown_y: np.ndarray          # (S, N_SHOWN) their labels
    test: np.ndarray             # (10,) per-digit accuracy on the whole test set, at the end

    def old(self, i=-1):
        """Mean accuracy over the old digits at recorded step ``i``."""
        return float(self.per_digit[i, list(OLD)].mean())

    def new(self, i=-1):
        """Mean accuracy over the new digits at recorded step ``i``."""
        return float(self.per_digit[i, list(NEW)].mean())

    @property
    def test_old(self):
        """Final test accuracy over the old digits."""
        return float(self.test[list(OLD)].mean())

    @property
    def test_new(self):
        """Final test accuracy over the new digits."""
        return float(self.test[list(NEW)].mean())


REMEDIES = ("plain", "rehearsal", "lowlr")


def forgetting_runs(data=None, cache_dir=None):
    """``{remedy: ForgettingRun}`` for :data:`REMEDIES` (seed 0): plain
    fine-tuning, rehearsal of :data:`REHEARSE` buffer images a minibatch, and
    lower rates (:data:`FT_LOW`)."""
    data = data or split(cache_dir)
    pre = pretrain(data, cache_dir).P
    # dense while the old digits collapse (the first few dozen steps), sparse after
    marks = np.unique(np.concatenate([np.arange(0, FT_DENSE, FT_EVERY),
                                      np.arange(FT_DENSE, FT_STEPS + 1, FT_SPARSE)]))
    eye = np.eye(CLASSES)

    def one(remedy, seed=0):
        P = {k: v.copy() for k, v in pre.items()}
        low = rates(FT_LOW[0] * LR, FT_LOW[1] * LR)
        opt = PerLayerAdam(P, low if remedy == "lowlr" else rates(LR, LR))
        rehearse = REHEARSE if remedy == "rehearsal" else 0
        rng = np.random.default_rng(seed)
        want = set(marks.tolist())
        accs, shown, shown_y = [], [], []
        per_epoch = -(-len(data.x_new) // (FT_BATCH - rehearse))
        for k, idx in enumerate(minibatches(rng, len(data.x_new),
                                            -(-(FT_STEPS + 1) // per_epoch), FT_BATCH - rehearse)):
            x, y = data.x_new[idx], data.y_new[idx]
            if rehearse:
                # the buffer's share of the minibatch, drawn with replacement, then
                # shuffled in so the recorded images are a fair sample of the mix
                j = rng.integers(0, len(data.x_buffer), rehearse)
                x, y = np.concatenate([x, data.x_buffer[j]]), np.concatenate([y, data.y_buffer[j]])
                order = rng.permutation(len(x))
                x, y = x[order], y[order]
            if k in want:
                accs.append(per_digit_accuracy(P, data.x_eval, data.y_eval))
                shown.append(x[:N_SHOWN].astype(np.float32))
                shown_y.append(y[:N_SHOWN])
            if k == FT_STEPS:
                break
            opt.step(P, grads(P, x, eye[y]))
        return dict(per_digit=np.array(accs), shown=np.array(shown), shown_y=np.array(shown_y),
                    test=per_digit_accuracy(P, data.x_test, data.y_test))

    def build():
        out = {"steps": marks}
        for remedy in REMEDIES:
            out.update({"%s.%s" % (remedy, k): v for k, v in one(remedy).items()})
        return out

    d = cached_npz("transfer_forgetting",
                   _settings(data, pre=PRE, steps=FT_STEPS, batch=FT_BATCH, marks=marks.tolist(),
                             lr=LR, low=list(FT_LOW), rehearse=REHEARSE, shown=N_SHOWN,
                             remedies=list(REMEDIES)),
                   build, cache_dir)
    return {r: ForgettingRun(d["steps"], d[r + ".per_digit"], d[r + ".shown"], d[r + ".shown_y"],
                             d[r + ".test"]) for r in REMEDIES}


# ------------------------------------------------------------------ the learning-rate finder


def _mlp(seed):
    rng = np.random.default_rng(seed)
    P = {"%s%d" % (n, k): a for k, layer in enumerate(_init(rng))
         for n, a in zip("Wb", layer, strict=True)}
    return P, rng


def _mlp_loss(P, x, y):
    p = _forward(_as_layers(P, 3), x)[2]
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))


def _mlp_accuracy(P, x, y):
    return float((_forward(_as_layers(P, 3), x)[2].argmax(1) == y).mean())


def range_lr(k, steps=LR_STEPS, lo=LR_LO, hi=LR_HI):
    """The learning rate of step ``k`` of the range test: geometric from ``lo``
    (step 0) to ``hi`` (the last step)."""
    return lo * (hi / lo) ** (k / (steps - 1))


def lr_range_test(seed=0, cache_dir=None):
    """``(lrs, losses)``, one per step: the MLP's minibatch loss as the
    learning rate rises geometrically over :data:`LR_STEPS` steps of Adam."""
    settings = dict(model="mlp", sizes=[784, 64, 32, 10], optimizer="adam", lo=LR_LO, hi=LR_HI,
                    steps=LR_STEPS, batch=LR_BATCH, seed=seed, init="he_normal", data="mnist/255")

    def build():
        xtr, ytr, _xte, _yte = load(cache_dir)
        P, rng = _mlp(seed)
        opt = Adam(P, LR_LO)
        eye = np.eye(CLASSES)
        lrs, losses = [], []
        epochs = -(-LR_STEPS // (len(xtr) // LR_BATCH))
        for k, idx in enumerate(minibatches(rng, len(xtr), epochs, LR_BATCH)):
            if k == LR_STEPS:
                break
            x, y = xtr[idx], ytr[idx]
            lrs.append(range_lr(k))
            losses.append(_mlp_loss(P, x, y))
            opt.step(P, _mlp_grads(P, x, eye[y]), lrs[-1])
        return dict(lrs=np.array(lrs), losses=np.array(losses))

    d = cached_npz("lr_range_test", settings, build, cache_dir)
    return d["lrs"], d["losses"]


def centred_mean(x, window=SMOOTH):
    """Running mean of ``x`` over a centred window (odd); at the ends the
    window shrinks to what is there."""
    x = np.asarray(x, np.float64)
    n, h = len(x), window // 2
    c = np.cumsum(np.concatenate([[0.0], x]))
    i = np.arange(n)
    lo, hi = np.maximum(0, i - h), np.minimum(n, i + h + 1)
    return (c[hi] - c[lo]) / (hi - lo)


@dataclass
class LRAnalysis:
    """What the range test says (indices into the recorded steps)."""

    smooth: np.ndarray           # the smoothed loss
    slope: np.ndarray            # its derivative against log10 of the learning rate
    knee: int                    # the steepest fall before the minimum
    minimum: int                 # the lowest smoothed loss
    diverge: int                 # first step past the minimum with the loss back above its start


def lr_analysis(lrs, losses, window=SMOOTH) -> LRAnalysis:
    """Smooth the range test and find the knee, the minimum and the divergence."""
    sm = centred_mean(losses, window)
    slope = np.gradient(sm, np.log10(lrs))
    minimum = int(np.argmin(sm))
    knee = int(np.argmin(slope[:minimum]))
    above = np.nonzero(sm[minimum:] > sm[0])[0]
    diverge = minimum + int(above[0]) if len(above) else len(sm) - 1
    return LRAnalysis(sm, slope, knee, minimum, diverge)


# ------------------------------------------------------------------ warm restarts


def cosine_restarts(steps, cycles, peak):
    """``lr(k)``: a cosine from ``peak`` to 0 over ``steps / cycles`` steps,
    restarted ``cycles`` times (one cycle is a plain cosine)."""
    length = steps // cycles

    def lr_at(k):
        return peak * 0.5 * (1 + np.cos(np.pi * ((k % length) / length)))
    return lr_at


@dataclass
class RestartRun:
    """The MLP trained with a restarted (or a single) cosine."""

    lrs: np.ndarray              # (steps,) the learning rate of each step
    losses: np.ndarray           # (steps,) minibatch loss before each step
    sub_steps: np.ndarray        # (S,) steps at which the subset loss was measured
    sub_loss: np.ndarray         # (S,) cross-entropy on SUB_N fixed training images
    cycle_acc: np.ndarray        # test accuracy at the end of each cycle (the last is the end)

    @property
    def steps(self):
        """Number of training steps."""
        return len(self.losses)

    @property
    def acc(self):
        """Test accuracy at the end."""
        return float(self.cycle_acc[-1])


def restart_runs(cache_dir=None):
    """``(peak, {"restarts": RestartRun, "cosine": RestartRun})``: the MLP for
    :data:`EPOCHS` epochs with a cosine restarted every epoch and with one
    cosine, both peaking at the knee of :func:`lr_range_test`."""
    lrs, losses = lr_range_test(cache_dir=cache_dir)
    peak = float(lrs[lr_analysis(lrs, losses).knee])
    steps = EPOCHS * PER_EPOCH
    settings = dict(model="mlp", sizes=[784, 64, 32, 10], optimizer="adam", epochs=EPOCHS,
                    batch=LR_BATCH, peak=peak, cycles=CYCLES, sub_every=SUB_EVERY, sub_n=SUB_N,
                    seed=0, init="he_normal", data="mnist/255")

    def one(xtr, ytr, xte, yte, cycles):
        P, rng = _mlp(0)
        opt = Adam(P, peak)
        lr_at = cosine_restarts(steps, cycles, peak)
        sub = np.random.default_rng(0).permutation(len(xtr))[:SUB_N]
        eye = np.eye(CLASSES)
        lr_rec, loss_rec, sub_steps, sub_loss, accs = [], [], [], [], []
        length = steps // cycles
        for k, idx in enumerate(minibatches(rng, len(xtr), EPOCHS, LR_BATCH)):
            if k % SUB_EVERY == 0:
                sub_steps.append(k)
                sub_loss.append(_mlp_loss(P, xtr[sub], ytr[sub]))
            if k and k % length == 0:
                accs.append(_mlp_accuracy(P, xte, yte))
            x, y = xtr[idx], ytr[idx]
            lr_rec.append(float(lr_at(k)))
            loss_rec.append(_mlp_loss(P, x, y))
            opt.step(P, _mlp_grads(P, x, eye[y]), lr_rec[-1])
        sub_steps.append(steps)
        sub_loss.append(_mlp_loss(P, xtr[sub], ytr[sub]))
        accs.append(_mlp_accuracy(P, xte, yte))
        return dict(lrs=np.array(lr_rec), losses=np.array(loss_rec), sub_steps=np.array(sub_steps),
                    sub_loss=np.array(sub_loss), cycle_acc=np.array(accs))

    def build():
        xtr, ytr, xte, yte = load(cache_dir)
        out = {}
        for name, cycles in (("restarts", CYCLES), ("cosine", 1)):
            out.update({"%s.%s" % (name, k): v for k, v in one(xtr, ytr, xte, yte, cycles).items()})
        return out

    d = cached_npz("mlp_restarts", settings, build, cache_dir)
    fields = ("lrs", "losses", "sub_steps", "sub_loss", "cycle_acc")
    return peak, {name: RestartRun(*(d["%s.%s" % (name, f)] for f in fields))
                  for name in ("restarts", "cosine")}


def landscape_restarts(cache_dir=None):
    """Minibatch SGD (:data:`LAND_BATCH` examples a step) on the round bowl of
    :mod:`voxanim.landscape` under a cosine restarted :data:`LAND_CYCLES`
    times, peak :data:`LAND_PEAK`: a :class:`voxanim.landscape.Run`."""
    settings = dict(problem="mnist-0v1-ink-width", start=LS_START, steps=LAND_STEPS,
                    cycles=LAND_CYCLES, peak=LAND_PEAK, batch=LAND_BATCH, seed=LAND_SEED,
                    optimizer="sgd", sampling="with replacement")

    def build():
        return {"restarts": LS.descend(LS.problem(cache_dir), partial(SGD, lr=LAND_PEAK), LS_START,
                                       LAND_STEPS, batch=LAND_BATCH,
                                       rng=np.random.default_rng(LAND_SEED),
                                       lr_at=cosine_restarts(LAND_STEPS, LAND_CYCLES, LAND_PEAK))}
    return LS.run_cache("landscape_restarts", settings, build, cache_dir)["restarts"]

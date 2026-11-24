"""The course's CNN: ``examples/mnist_cnn.py`` trained once in numpy, as PyTorch trains it,
for the CNN animations of "Blocks and code" (catalogue 215-223, :mod:`voxanim.blocks`).

The program is ``examples/mnist_blocks.py`` (:mod:`voxanim.study_course`) with one block
changed, the model::

    nn.Sequential(nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2),
                  nn.Conv2d(8, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
                  nn.Flatten(), nn.Linear(16 * 5 * 5, 10))

so everything else is :mod:`~voxanim.study_course`'s: the same normalised digits, the same
batches in the same order (:func:`voxanim.study_course.batch`), the same loss and the same
``SGD(lr=0.1)``, one epoch of 938 steps. The parameters are kept as PyTorch keeps them
(:data:`KEYS`, the keys of ``model.state_dict()``: a convolution's weight is ``[out, in,
3, 3]``) and initialised as PyTorch initialises ``nn.Conv2d`` and ``nn.Linear``: uniform
in ``±1/sqrt(fan_in)``, ``fan_in`` being ``in x 3 x 3`` for a convolution.

The convolutions are matrix products over 3 x 3 patches (``im2col``); max pooling keeps
the first biggest number of each 2 x 2 square and sends its gradient there, as PyTorch
does, and drops the odd last row and column (11 -> 5). Everything is float32.
``tests/test_mnist_blocks_equivalence.py`` runs the script's own lines in PyTorch from
:func:`init_params` on the first batch: the loss, every gradient and the new weights agree
with this twin to 1e-5.

The API (``P`` is a dict keyed by :data:`KEYS`):

* the model: :func:`init_params`, :func:`count` (5,258), :func:`counts` (per block),
  :func:`forward` ``(P, x) -> dict(x, z1, h1, p1, z2, h2, p2, flat, logits)``,
  :func:`loss_and_grads`, :func:`shapes` (every block's output shape on the first batch),
  :func:`side_walk` (28 -> 26 -> 13 -> 11 -> 5);
* what the script prints: :data:`PRINTED_MODEL`, :data:`PRINTED_GRAD`, and the errors of
  the broken programs of ``examples/blocks_parts.py`` (:data:`ERRORS`);
* the run: :func:`losses`, :func:`grad_norms`, :func:`heldout` (every 25 steps),
  :func:`params` ``(step)`` and :func:`grads` ``(step)`` at :data:`SNAP_STEPS`,
  :func:`activations` ``(step, i)`` (training digit ``i`` through the model),
  :func:`stamp_grad_parts` (one stamp weight's gradient as the sum of its parts),
  :func:`predictions` and :func:`compare` (against the MLP of
  :mod:`~voxanim.study_course`, on the same 10,000 test digits), :func:`summary`.

The run is built once (about a minute) and cached by :func:`voxanim.cache.cached_npz`.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import sys

import numpy as np

from . import study_course as S
from .cache import cached_npz

VERSION = 1
SIDE = S.SIDE
KERNEL = 3
CHANNELS = (1, 8, 16)                      # nn.Conv2d(1, 8, 3), nn.Conv2d(8, 16, 3)
POOL = 2
CLASSES = S.CLASSES
BATCH, EPOCH_STEPS, LR = S.BATCH, S.EPOCH_STEPS, S.LR
SEED = 215
# The keys of model.state_dict(): nn.Sequential numbers its blocks 0 Conv2d, 1 ReLU,
# 2 MaxPool2d, 3 Conv2d, 4 ReLU, 5 MaxPool2d, 6 Flatten, 7 Linear.
LEARNS = (0, 3, 7)
KEYS = tuple("%d.%s" % (i, w) for i in LEARNS for w in ("weight", "bias"))
BLOCKS = ("Conv2d", "ReLU", "MaxPool2d", "Conv2d", "ReLU", "MaxPool2d", "Flatten", "Linear")
SNAP_STEPS = S.SNAP_STEPS
EVAL_STEPS = S.EVAL_STEPS
SCRIPT = "examples/mnist_cnn.py"
PRINTED_MODEL = """Sequential(
  (0): Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1))
  (1): ReLU()
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): Conv2d(8, 16, kernel_size=(3, 3), stride=(1, 1))
  (4): ReLU()
  (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (6): Flatten(start_dim=1, end_dim=-1)
  (7): Linear(in_features=400, out_features=10, bias=True)
)"""
PRINTED_GRAD = "torch.Size([8, 1, 3, 3])"              # model[0].weight.grad.shape
# What PyTorch says when the broken programs of examples/blocks_parts.py run (regions
# "cnn-mistake-..."), checked against torch by tests/test_mnist_blocks_equivalence.py.
ERRORS = {
    "cnn-mistake-in-features":
        "addmm: mat1 and mat2 shapes cannot be multiplied (64x400 and 784x10)",
    "cnn-mistake-channels":
        "Given groups=1, weight of size [8, 3, 3, 3], expected input[64, 1, 28, 28] to have "
        "3 channels, but got 1 channels instead",
    "cnn-mistake-no-batch":
        "addmm: mat1 and mat2 shapes cannot be multiplied (16x25 and 400x10)",
}
FIXED_SHAPES = {"cnn-fix-no-batch": (1, CLASSES)}       # what the fixed line gives


# ====================================================================== the model
def side_walk(side=SIDE):
    """The side of the square maps after each block that changes it: a 3 x 3 stamp with no
    padding loses 2, a 2 x 2 pool halves (rounding down): ``(28, 26, 13, 11, 5)``."""
    out = [side]
    for _conv in CHANNELS[1:]:
        side = side - (KERNEL - 1)
        out.append(side)
        side = side // POOL
        out.append(side)
    return tuple(out)


IN_FEATURES = CHANNELS[-1] * side_walk()[-1] ** 2          # 16 x 5 x 5 = 400


def init_params(seed=SEED):
    """The parameters as PyTorch initialises them (uniform in ``±1/sqrt(fan_in)``),
    float32, keyed as ``model.state_dict()``."""
    rng = np.random.default_rng([S.SEED, 2, int(seed)])
    shapes = {0: (CHANNELS[1], CHANNELS[0], KERNEL, KERNEL),
              3: (CHANNELS[2], CHANNELS[1], KERNEL, KERNEL),
              7: (CLASSES, IN_FEATURES)}
    P = {}
    for i in LEARNS:
        shape = shapes[i]
        bound = 1.0 / np.sqrt(np.prod(shape[1:]))
        P["%d.weight" % i] = rng.uniform(-bound, bound, shape).astype(np.float32)
        P["%d.bias" % i] = rng.uniform(-bound, bound, shape[0]).astype(np.float32)
    return P


def count(P=None):
    """Every number the model learns (``sum(p.numel() for p in model.parameters())``)."""
    P = init_params() if P is None else P
    return int(sum(np.asarray(v).size for v in P.values()))


def counts(P=None):
    """Per block that learns: ``{index: dict(block, weights, biases, total, shape)}``."""
    P = init_params() if P is None else P
    out = {}
    for i in LEARNS:
        w, b = P["%d.weight" % i], P["%d.bias" % i]
        out[i] = dict(block=BLOCKS[i], weights=int(w.size), biases=int(b.size),
                      total=int(w.size + b.size), shape=tuple(int(s) for s in w.shape))
    return out


def _patches(x):
    """``[B, C, H, W] -> [B * H' * W', C * 9]``: every 3 x 3 patch, in PyTorch's weight
    order (channel, row, column)."""
    B, C, H, W = x.shape
    win = np.lib.stride_tricks.sliding_window_view(x, (KERNEL, KERNEL), axis=(2, 3))
    h, w = H - KERNEL + 1, W - KERNEL + 1
    return np.ascontiguousarray(win.transpose(0, 2, 3, 1, 4, 5)).reshape(B * h * w, C * 9)


def conv(x, w, b):
    """``nn.Conv2d(C, O, 3)`` (stride 1, no padding): ``[B, C, H, W] -> [B, O, H-2, W-2]``."""
    B, _C, H, W = x.shape
    h, wd = H - KERNEL + 1, W - KERNEL + 1
    out = _patches(x) @ w.reshape(len(w), -1).T + b
    return np.ascontiguousarray(out.reshape(B, h, wd, len(w)).transpose(0, 3, 1, 2))


def conv_back(d, x, w):
    """The gradients of :func:`conv`: ``(dx, dw, db)`` for the gradient ``d`` of its output."""
    B, C = x.shape[:2]
    O, h, wd = d.shape[1], d.shape[2], d.shape[3]
    dm = d.transpose(0, 2, 3, 1).reshape(-1, O)
    dw = (dm.T @ _patches(x)).reshape(w.shape)
    db = dm.sum(0)
    dcols = (dm @ w.reshape(O, -1)).reshape(B, h, wd, C, KERNEL, KERNEL)
    dx = np.zeros_like(x)
    for i in range(KERNEL):
        for j in range(KERNEL):
            dx[:, :, i:i + h, j:j + wd] += dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    return dx, dw, db


def pool(x):
    """``nn.MaxPool2d(2)``: each 2 x 2 square becomes its biggest number; an odd last row
    and column are dropped. Returns ``(out, where)``, ``where`` the position (0..3, row by
    row) of the first biggest in each square, where the gradient goes back."""
    B, C, H, W = x.shape
    h, w = H // POOL, W // POOL
    sq = x[:, :, :h * POOL, :w * POOL].reshape(B, C, h, POOL, w, POOL)
    sq = sq.transpose(0, 1, 2, 4, 3, 5).reshape(B, C, h, w, POOL * POOL)
    where = sq.argmax(-1)
    return np.take_along_axis(sq, where[..., None], -1)[..., 0], where


def pool_back(d, where, shape):
    """The gradient of :func:`pool`'s input: each square's gradient goes to its biggest."""
    B, C = shape[:2]
    h, w = d.shape[2], d.shape[3]
    sq = np.zeros((B, C, h, w, POOL * POOL), d.dtype)
    np.put_along_axis(sq, where[..., None], d[..., None], -1)
    sq = sq.reshape(B, C, h, w, POOL, POOL).transpose(0, 1, 2, 4, 3, 5)
    out = np.zeros(shape, d.dtype)
    out[:, :, :h * POOL, :w * POOL] = sq.reshape(B, C, h * POOL, w * POOL)
    return out


def _image(x):
    x = np.asarray(x, np.float32)
    return x.reshape(len(x), 1, SIDE, SIDE)


def forward(P, x):
    """One pass: ``dict(x [B,1,28,28], z1, h1 [B,8,26,26], p1 [B,8,13,13], z2, h2
    [B,16,11,11], p2 [B,16,5,5], flat [B,400], logits [B,10], where1, where2)``."""
    a = dict(x=_image(x))
    a["z1"] = conv(a["x"], P["0.weight"], P["0.bias"])
    a["h1"] = np.maximum(a["z1"], 0)
    a["p1"], a["where1"] = pool(a["h1"])
    a["z2"] = conv(a["p1"], P["3.weight"], P["3.bias"])
    a["h2"] = np.maximum(a["z2"], 0)
    a["p2"], a["where2"] = pool(a["h2"])
    a["flat"] = a["p2"].reshape(len(a["p2"]), -1)
    a["logits"] = a["flat"] @ P["7.weight"].T + P["7.bias"]
    return a


def loss_and_grads(P, x, y, keep=False):
    """``loss.backward()``: the batch's mean cross-entropy and the gradient of every
    parameter (same keys and shapes as ``P``); ``keep``: also return the gradient of the
    first convolution's output (``"dz1"``), for :func:`stamp_grad_parts`."""
    a = forward(P, x)
    y = np.asarray(y)
    n = len(y)
    loss = float(S.loss_each(a["logits"], y).mean())
    d = S._softmax(a["logits"]).astype(np.float32)
    d[np.arange(n), y] -= 1.0
    d /= np.float32(n)
    G = {"7.weight": d.T @ a["flat"], "7.bias": d.sum(0)}
    d = (d @ P["7.weight"]).reshape(a["p2"].shape)
    d = pool_back(d, a["where2"], a["h2"].shape) * (a["z2"] > 0)
    dp1, G["3.weight"], G["3.bias"] = conv_back(d, a["p1"], P["3.weight"])
    dz1 = pool_back(dp1, a["where1"], a["h1"].shape) * (a["z1"] > 0)
    _dx, G["0.weight"], G["0.bias"] = conv_back(dz1, a["x"], P["0.weight"])
    G = {k: G[k].astype(P[k].dtype) for k in P}
    return (loss, G, dz1) if keep else (loss, G)


def shapes():
    """Every block's output shape on the first batch, in order: ``[(block, shape), ...]``
    from the input ``("input", (64, 1, 28, 28))`` to ``("Linear", (64, 10))``."""
    a = forward(init_params(), S.first_batch()["x"])
    order = ("x", "z1", "h1", "p1", "z2", "h2", "p2", "flat", "logits")
    return [(name, tuple(int(s) for s in a[k].shape))
            for name, k in zip(("input",) + BLOCKS, order, strict=True)]


def predict(P, x, chunk=2000):
    """The predicted digit of every row of ``x`` (normalised, ``(N, 784)``)."""
    return np.concatenate([forward(P, x[s:s + chunk])["logits"].argmax(1)
                           for s in range(0, len(x), chunk)])


def score(P, chunk=2000):
    """Held-out ``(accuracy, mean loss)`` on the 10,000 test digits."""
    _x, _y, xt, yt = S.data()
    right, total = 0, 0.0
    for s in range(0, len(xt), chunk):
        logits = forward(P, xt[s:s + chunk])["logits"]
        right += int((logits.argmax(1) == yt[s:s + chunk]).sum())
        total += float(S.loss_each(logits, yt[s:s + chunk]).astype(np.float64).sum())
    return right / len(xt), total / len(xt)


# ====================================================================== the run
def _build():
    P = init_params()
    out = dict(loss=[], grad_norm=[], eval_steps=[], accuracy=[], test_loss=[])
    for step in range(EPOCH_STEPS):
        if step in EVAL_STEPS:
            a, l = score(P)
            out["eval_steps"].append(step)
            out["accuracy"].append(a)
            out["test_loss"].append(l)
        b = S.batch(step)
        loss, G = loss_and_grads(P, b["x"], b["y"])
        if step in SNAP_STEPS:
            for k in P:
                out["W%d__%s" % (step, k)] = P[k].copy()
                out["G%d__%s" % (step, k)] = G[k]
        out["loss"].append(loss)
        out["grad_norm"].append(float(np.sqrt(sum((G[k].astype(np.float64) ** 2).sum()
                                                  for k in P))))
        P = S.sgd_update(P, G, LR)
        if not all(np.isfinite(v).all() for v in P.values()):
            raise FloatingPointError("the CNN's weights stopped being finite at step %d" % step)
    a, l = score(P)
    out["eval_steps"].append(EPOCH_STEPS)
    out["accuracy"].append(a)
    out["test_loss"].append(l)
    for k in P:
        out["W%d__%s" % (EPOCH_STEPS, k)] = P[k].copy()
    out["predictions"] = predict(P, S.data()[2])
    return {k: np.asarray(v) for k, v in out.items()}


def _settings():
    return dict(version=VERSION, course=S.VERSION, channels=CHANNELS, kernel=KERNEL, pool=POOL,
                batch=BATCH, lr=LR, seed=SEED, snaps=SNAP_STEPS, evals=EVAL_STEPS,
                source=hashlib.sha256(inspect.getsource(sys.modules[__name__]).encode())
                .hexdigest()[:16])


@functools.lru_cache(maxsize=1)
def run():
    """The recorded run, a dict of arrays (cached)."""
    return {k: np.asarray(v) for k, v in cached_npz("study-course-cnn", _settings(),
                                                    _build).items()}


def losses():
    """The loss of every step of the epoch (938), on the same batches as the MLP's."""
    return run()["loss"]


def grad_norms():
    """The length of every step's whole gradient (all 5,258 numbers)."""
    return run()["grad_norm"]


def heldout():
    """``dict(steps, accuracy, loss)`` on the 10,000 test digits, every 25 steps and at
    the end (the same steps as :func:`voxanim.study_course.heldout`)."""
    r = run()
    return dict(steps=r["eval_steps"], accuracy=r["accuracy"], loss=r["test_loss"])


def _snap(prefix, step):
    r = run()
    got = {k: r["%s%d__%s" % (prefix, step, k)] for k in KEYS
           if "%s%d__%s" % (prefix, step, k) in r}
    if not got:
        raise KeyError("nothing kept at step %s (kept: %s)" % (step, SNAP_STEPS))
    return got


def params(step=EPOCH_STEPS):
    """The parameters before step ``step`` (0: as initialised; ``EPOCH_STEPS``: after the
    epoch), for ``step`` in :data:`SNAP_STEPS` or ``EPOCH_STEPS``."""
    if step == 0:
        return init_params()
    return _snap("W", step)


def grads(step=0):
    """The gradient ``loss.backward()`` leaves in ``.grad`` at step ``step`` (in
    :data:`SNAP_STEPS`), from :func:`params` ``(step)`` on batch ``step``."""
    return _snap("G", step)


def activations(step=EPOCH_STEPS, i=0):
    """Training digit ``i`` (``train_set[i]``) through the model with :func:`params`
    ``(step)``: :func:`forward`'s dict for a batch of one, plus ``probs`` and ``y``."""
    x, y = S.digit(i)
    a = forward(params(step), x[None])
    a["probs"] = S._softmax(a["logits"])
    a["y"] = y
    return a


def stamp_grad_parts(step=0, stamp=0, row=0, col=0):
    """One number of ``model[0].weight.grad`` (stamp ``stamp``, weight ``row, col``) at
    step ``step`` as the sum of its parts: the stamp sat at 26 x 26 places on each of the
    batch's digits, and every place adds (the pixel under that weight) x (the gradient of
    the number the stamp made there). Returns ``dict(parts [B, 26, 26], total, grad)``;
    ``total`` (the sum of the parts, in float64) is ``grad`` (PyTorch's number)."""
    P = params(step)
    b = S.batch(step)
    _loss, G, dz1 = loss_and_grads(P, b["x"], b["y"], keep=True)
    h = dz1.shape[2]
    x = _image(b["x"])[:, 0, row:row + h, col:col + h]
    parts = dz1[:, stamp].astype(np.float64) * x
    return dict(parts=parts, total=float(parts.sum()),
                grad=float(G["0.weight"][stamp, 0, row, col]))


def predictions():
    """The CNN's answer for each of the 10,000 test digits after the epoch."""
    return run()["predictions"]


@functools.lru_cache(maxsize=1)
def mlp_predictions():
    """The MLP's (:mod:`voxanim.study_course`, after its epoch) on the same test digits."""
    return S.forward(S.params(S.EPOCH_STEPS), S.data()[2])["logits"].argmax(1)


def compare():
    """The CNN against the MLP, both after one epoch on the same batches: parameters,
    accuracy, digits missed, and which of the 10,000 each gets right; ``confusions``: the
    CNN's most common mistakes ``[(true, said, how many), ...]``."""
    yt = S.data()[3]
    cnn, mlp = predictions() == yt, mlp_predictions() == yt
    pairs = {}
    for t, p in zip(yt[~cnn], predictions()[~cnn], strict=True):
        pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    confusions = sorted(((t, p, n) for (t, p), n in pairs.items()), key=lambda c: (-c[2], c))
    return dict(cnn_params=count(), mlp_params=S.count(),
                cnn_accuracy=float(heldout()["accuracy"][-1]),
                mlp_accuracy=float(S.heldout("sgd")["accuracy"][-1]),
                cnn_missed=int((~cnn).sum()), mlp_missed=int((~mlp).sum()),
                both_right=int((cnn & mlp).sum()), only_cnn=int((cnn & ~mlp).sum()),
                only_mlp=int((~cnn & mlp).sum()), both_wrong=int((~cnn & ~mlp).sum()),
                per_digit_cnn=[int((~cnn & (yt == d)).sum()) for d in range(CLASSES)],
                per_digit_mlp=[int((~mlp & (yt == d)).sum()) for d in range(CLASSES)],
                confusions=confusions)


def summary():
    """The headline numbers, JSON-able."""
    c = compare()
    c.pop("confusions")
    l = losses()
    return dict(parameters=count(), counts={str(k): v for k, v in counts().items()},
                side_walk=side_walk(), in_features=IN_FEATURES, first_loss=float(l[0]),
                last_loss=float(l[-1]), mean_last_100=float(l[-100:].mean()), **c)

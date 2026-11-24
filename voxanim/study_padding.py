"""Padding, for the padding animations of "Blocks and code" (catalogue 224-233): every number
they show, computed here in numpy from the lines of ``examples/padding.py``.

Two halves, as in the script:

* **A CNN's padding.** How many places a k x k stamp fits (:func:`out_side`), how often each
  pixel is read with and without padding (:func:`coverage`), "same" and "valid"
  (:func:`same_and_valid`), what each ``F.pad`` mode puts in the border (:func:`modes`, on
  an 8 x 8 piece of the course's first digit), how a stack of stamps runs out of map
  (:func:`deep`), and the course's CNN with ``padding=1`` trained once, as
  :mod:`voxanim.study_course_cnn` trains it without (:func:`run`, :func:`compare`,
  :func:`shifted`: both CNNs on test digits moved towards an edge).
* **A batch is one block.** The first four digits cut to their ink have four sizes, so
  ``torch.stack`` fails (:data:`STACK_ERROR`, PyTorch's message, checked against torch);
  padded to 28 x 28 with the background they stack (:func:`crops`). Lines of Tiny
  Shakespeare: the first eight padded to the longest, with their mask (:func:`lines_batch`),
  what padding does to a mean unless it is masked (:func:`masked_mean`), and how much of a
  batch is padding when lines are batched in file order, shuffled, or sorted by length
  (:func:`waste`).

One fact runs through the first half: after ``Normalize`` a blank pixel is not 0 but
(0 - 0.1307) / 0.3081 = -0.42 (:data:`BG`), so ``padding=1`` and ``F.pad``'s "constant"
mode put a ring that is not paper but a faint frame of ink (0 is more than -0.42) around
every digit.

The padded CNN's run is built once (about a minute) and cached by
:func:`voxanim.cache.cached_npz`; ``tests/test_padding_equivalence.py`` runs the script's own
lines in torch and matches its first step, the shapes, the modes and the errors.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import sys

import numpy as np

from . import realdata
from . import study_course as S
from . import study_course_cnn as C
from .cache import cached_npz

VERSION = 1
SIDE = S.SIDE
BG = float((np.float32(0.0) - np.float32(S.MEAN)) / np.float32(S.STD))     # -0.4242
KERNELS = (3, 5, 7)
DEEP_LAYERS = 12
MODES = ("constant", "reflect", "replicate", "circular")      # F.pad's names
NUMPY_MODES = {"constant": "constant", "reflect": "reflect", "replicate": "edge",
               "circular": "wrap"}
PART = (slice(5, 13), slice(8, 16))          # x[:1, :, 5:13, 8:16] of training digit 0
PART_PAD = 2
CUT_DIGITS = 4
TEXT_LINES = 8
BATCH = S.BATCH
SHIFTS = (0, 1, 2, 3, 4, 5, 6)
DIRECTIONS = ("right", "left", "down", "up")
SEED = 224
SCRIPT = "examples/padding.py"
# What PyTorch says (checked against torch by tests/test_padding_equivalence.py)
STACK_ERROR = ("stack expects each tensor to be equal size, but got [1, 20, 20] at entry 0 "
               "and [1, 20, 17] at entry 1")
PRINTED_DEEP = "torch.Size([64, 1, 4, 4])"
# The padded CNN of region "pad-model": its blocks that learn and their shapes
PAD_CHANNELS = (1, 8, 16)
PAD_KEYS = ("0.weight", "0.bias", "3.weight", "3.bias", "7.weight", "7.bias")


# ====================================================================== a CNN's padding
def out_side(side, k=3, pad=0, stride=1):
    """The side of a conv's output: ``(side + 2 pad - k) // stride + 1``."""
    return (side + 2 * pad - k) // stride + 1


def coverage(side=SIDE, k=3, pad=0):
    """How many of a ``k x k`` stamp's places read each real pixel (``side x side``), with
    ``pad`` rings of padding. For a 3 x 3 stamp on 28 x 28: without padding a corner pixel
    is read by 1 place, an edge pixel by 3 and a middle one by 9; with ``padding=1`` the
    corner by 4 and the edge by 6 (the middle still 9): fewer, still, at the border."""
    n = out_side(side, k, pad)
    count = np.zeros((side + 2 * pad, side + 2 * pad), int)
    for r in range(n):
        for c in range(n):
            count[r:r + k, c:c + k] += 1
    return count[pad:pad + side, pad:pad + side]


def same_and_valid(kernels=KERNELS, side=SIDE):
    """For each kernel: the padding "same" needs, ``(k - 1) // 2``, and both output sides."""
    return [dict(k=k, pad=(k - 1) // 2, same=out_side(side, k, (k - 1) // 2),
                 valid=out_side(side, k, 0), lost=k - 1) for k in kernels]


def part():
    """The 8 x 8 piece of training digit 0 that region "pad-modes" pads (normalised)."""
    return S.digit(0)[0][0][PART]


def modes(width=PART_PAD):
    """``F.pad(part, [width] * 4, mode=m)`` for each of :data:`MODES`: ``{mode: 12 x 12}``.
    "constant" pads with 0, which after Normalize is not the background (:data:`BG`)."""
    p = part()
    return {m: np.pad(p, width, mode=NUMPY_MODES[m]) for m in MODES}


def deep(layers=DEEP_LAYERS, side=SIDE, k=3):
    """The side after each of ``layers`` 3 x 3 stamps, without and with ``padding=1``."""
    valid, same = [side], [side]
    for _ in range(layers):
        valid.append(out_side(valid[-1], k, 0))
        same.append(out_side(same[-1], k, 1))
    return dict(valid=valid, same=same)


# ---------------------------------------------------------------- the padded CNN
def pad_side_walk(side=SIDE):
    """The padded CNN's map sides: 28 -> 28 (conv) -> 14 (pool) -> 14 -> 7."""
    out = [side]
    for _ in PAD_CHANNELS[1:]:
        side = out_side(side, 3, 1)
        out.append(side)
        side //= 2
        out.append(side)
    return tuple(out)


PAD_IN_FEATURES = PAD_CHANNELS[-1] * pad_side_walk()[-1] ** 2       # 16 x 7 x 7 = 784


def init_params(seed=SEED):
    """The padded CNN's parameters, initialised as PyTorch does (uniform ``±1/sqrt(fan_in)``),
    keyed as ``model.state_dict()``."""
    rng = np.random.default_rng([S.SEED, 3, int(seed)])
    shapes = {0: (8, 1, 3, 3), 3: (16, 8, 3, 3), 7: (10, PAD_IN_FEATURES)}
    P = {}
    for i, shape in shapes.items():
        bound = 1.0 / np.sqrt(np.prod(shape[1:]))
        P["%d.weight" % i] = rng.uniform(-bound, bound, shape).astype(np.float32)
        P["%d.bias" % i] = rng.uniform(-bound, bound, shape[0]).astype(np.float32)
    return P


def count(P=None):
    P = init_params() if P is None else P
    return int(sum(v.size for v in P.values()))


def _pad1(x):
    """``padding=1`` of ``nn.Conv2d``: one ring of zeros (zeros, not the background)."""
    return np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))


def forward(P, x):
    """The padded CNN: ``dict(x, z1 [B,8,28,28], h1, p1 [B,8,14,14], z2 [B,16,14,14], h2,
    p2 [B,16,7,7], flat [B,784], logits, where1, where2)``."""
    a = dict(x=np.asarray(x, np.float32).reshape(-1, 1, SIDE, SIDE))
    a["z1"] = C.conv(_pad1(a["x"]), P["0.weight"], P["0.bias"])
    a["h1"] = np.maximum(a["z1"], 0)
    a["p1"], a["where1"] = C.pool(a["h1"])
    a["z2"] = C.conv(_pad1(a["p1"]), P["3.weight"], P["3.bias"])
    a["h2"] = np.maximum(a["z2"], 0)
    a["p2"], a["where2"] = C.pool(a["h2"])
    a["flat"] = a["p2"].reshape(len(a["p2"]), -1)
    a["logits"] = a["flat"] @ P["7.weight"].T + P["7.bias"]
    return a


def loss_and_grads(P, x, y):
    """The batch's mean cross-entropy and every parameter's gradient."""
    a = forward(P, x)
    y = np.asarray(y)
    n = len(y)
    loss = float(S.loss_each(a["logits"], y).mean())
    d = S._softmax(a["logits"]).astype(np.float32)
    d[np.arange(n), y] -= 1.0
    d /= np.float32(n)
    G = {"7.weight": d.T @ a["flat"], "7.bias": d.sum(0)}
    d = (d @ P["7.weight"]).reshape(a["p2"].shape)
    d = C.pool_back(d, a["where2"], a["h2"].shape) * (a["z2"] > 0)
    dp1, G["3.weight"], G["3.bias"] = C.conv_back(d, _pad1(a["p1"]), P["3.weight"])
    dp1 = dp1[:, :, 1:-1, 1:-1]                       # the padding's gradient is dropped
    dz1 = C.pool_back(dp1, a["where1"], a["h1"].shape) * (a["z1"] > 0)
    _dx, G["0.weight"], G["0.bias"] = C.conv_back(dz1, _pad1(a["x"]), P["0.weight"])
    return loss, {k: G[k].astype(P[k].dtype) for k in P}


def predict(P, x, chunk=2000, fwd=forward):
    return np.concatenate([fwd(P, x[s:s + chunk])["logits"].argmax(1)
                           for s in range(0, len(x), chunk)])


def shift(x, s, direction):
    """Digits ``x`` (``(N, 784)``, normalised) moved ``s`` pixels towards ``direction``, the
    space they leave filled with the background, as a blank pixel looks after Normalize."""
    img = np.asarray(x, np.float32).reshape(-1, SIDE, SIDE)
    if s == 0:
        return img.reshape(len(img), -1)
    out = np.full_like(img, np.float32(BG))
    if direction == "right":
        out[:, :, s:] = img[:, :, :-s]
    elif direction == "left":
        out[:, :, :-s] = img[:, :, s:]
    elif direction == "down":
        out[:, s:, :] = img[:, :-s, :]
    else:
        out[:, :-s, :] = img[:, s:, :]
    return out.reshape(len(img), -1)


def _build():
    P = init_params()
    xt, yt = S.data()[2:]
    out = dict(loss=[], eval_steps=[], accuracy=[])
    for step in range(S.EPOCH_STEPS):
        if step in S.EVAL_STEPS:
            out["eval_steps"].append(step)
            out["accuracy"].append(float((predict(P, xt) == yt).mean()))
        b = S.batch(step)
        loss, G = loss_and_grads(P, b["x"], b["y"])
        if step == 0:
            for k in P:
                out["G0__" + k] = G[k]
        out["loss"].append(loss)
        P = S.sgd_update(P, G, S.LR)
        if step == 0:
            for k in P:
                out["W1__" + k] = P[k].copy()
    out["eval_steps"].append(S.EPOCH_STEPS)
    out["accuracy"].append(float((predict(P, xt) == yt).mean()))
    for k in P:
        out["W__" + k] = P[k].copy()
    out["predictions"] = predict(P, xt)
    plain = C.params(C.EPOCH_STEPS)
    acc = np.zeros((2, len(DIRECTIONS), len(SHIFTS)))
    for j, d in enumerate(DIRECTIONS):
        for i, s in enumerate(SHIFTS):
            moved = shift(xt, s, d)
            acc[0, j, i] = (predict(plain, moved, fwd=C.forward) == yt).mean()
            acc[1, j, i] = (predict(P, moved) == yt).mean()
    out["shifted"] = acc
    return {k: np.asarray(v) for k, v in out.items()}


def _settings():
    return dict(version=VERSION, course=S.VERSION, cnn=C.VERSION, seed=SEED, shifts=SHIFTS,
                directions=DIRECTIONS,
                source=hashlib.sha256(inspect.getsource(sys.modules[__name__]).encode())
                .hexdigest()[:16])


@functools.lru_cache(maxsize=1)
def run():
    """The padded CNN's recorded run (cached): per-step ``loss``, the held-out curve, its
    final parameters ``W__<key>``, its first step (``G0__``, ``W1__``), ``predictions``
    and ``shifted`` (see :func:`shifted`)."""
    return {k: np.asarray(v) for k, v in cached_npz("study-padding", _settings(),
                                                    _build).items()}


def params(which="end"):
    """The padded CNN's parameters: ``"start"``, ``"after_one_step"`` or ``"end"``."""
    if which == "start":
        return init_params()
    prefix = {"after_one_step": "W1__", "end": "W__"}[which]
    return {k: run()[prefix + k] for k in PAD_KEYS}


def first_grads():
    return {k: run()["G0__" + k] for k in PAD_KEYS}


def compare():
    """The two CNNs after one epoch on the same batches, one run each: numbers to learn,
    held-out accuracy, digits missed, and the held-out curves on the same steps."""
    yt = S.data()[3]
    r = run()
    return dict(plain_params=C.count(), padded_params=count(),
                plain_accuracy=float(C.heldout()["accuracy"][-1]),
                padded_accuracy=float(r["accuracy"][-1]),
                plain_missed=int((C.predictions() != yt).sum()),
                padded_missed=int((r["predictions"] != yt).sum()),
                steps=r["eval_steps"], plain_curve=C.heldout()["accuracy"],
                padded_curve=r["accuracy"],
                plain_last_100=float(C.losses()[-100:].mean()),
                padded_last_100=float(r["loss"][-100:].mean()))


def shifted():
    """Held-out accuracy of both CNNs on the 10,000 test digits moved 0..6 pixels towards
    each edge: ``dict(shifts, directions, plain [4, 7], padded [4, 7], plain_mean [7],
    padded_mean [7])`` (the means over the four directions)."""
    acc = run()["shifted"]
    return dict(shifts=SHIFTS, directions=DIRECTIONS, plain=acc[0], padded=acc[1],
                plain_mean=acc[0].mean(0), padded_mean=acc[1].mean(0))


# ====================================================================== a batch is one block
def cut(img):
    """``cut`` of the script: the digit cut to the smallest box holding every pixel above
    its lowest value (the paper). Returns ``(piece [1, h, w], (r0, r1, c0, c1))``."""
    img = np.asarray(img, np.float32)
    ink = img[0] > img.min() + 1e-3
    rows, cols = np.nonzero(ink.any(1))[0], np.nonzero(ink.any(0))[0]
    r0, r1, c0, c1 = int(rows[0]), int(rows[-1]), int(cols[0]), int(cols[-1])
    return img[:, r0:r1 + 1, c0:c1 + 1], (r0, r1, c0, c1)


def pad_to(piece, side=SIDE):
    """``pad_to`` of the script: room on the right and at the bottom, filled with :data:`BG`."""
    h, w = piece.shape[-2:]
    return np.pad(piece, ((0, 0), (0, side - h), (0, side - w)), constant_values=np.float32(BG))


def crops(n=CUT_DIGITS):
    """The first ``n`` training digits (``x[i]`` of the script) cut to their ink, their
    shapes and boxes, the padded block, and how much of it is padding."""
    pieces, boxes, labels = [], [], []
    for i in range(n):
        img, y = S.digit(i)
        piece, box = cut(img)
        pieces.append(piece)
        boxes.append(box)
        labels.append(y)
    block = np.stack([pad_to(p) for p in pieces])
    real = sum(p.size for p in pieces)
    return dict(pieces=pieces, shapes=[list(p.shape) for p in pieces], boxes=boxes,
                labels=labels, block=block, block_shape=list(block.shape), real=int(real),
                cells=int(block.size), padding=int(block.size - real))


# ---------------------------------------------------------------- lines of text
@functools.lru_cache(maxsize=1)
def corpus():
    """The whole of Tiny Shakespeare, as ``open(TEXT).read()`` reads it."""
    return "".join(realdata.text(s) for s in ("train", "validation", "test"))


@functools.lru_cache(maxsize=1)
def vocab():
    """``["<pad>"] + sorted(set(text))``: id 0 is the padding."""
    return ["<pad>"] + sorted(set(corpus()))


@functools.lru_cache(maxsize=1)
def lines():
    """Every line that is not empty, in file order."""
    return [s for s in corpus().split("\n") if s]


def encode(s):
    ids = {c: i for i, c in enumerate(vocab())}
    return np.array([ids[c] for c in s], np.int64)


def lines_batch(n=TEXT_LINES):
    """The first ``n`` lines padded to the longest (``pad_sequence(batch_first=True)``):
    ``dict(text, lengths, batch [n, L], mask, real, cells, padding_share)``."""
    text = lines()[:n]
    lengths = [len(s) for s in text]
    L = max(lengths)
    batch = np.zeros((n, L), np.int64)
    for i, s in enumerate(text):
        batch[i, :len(s)] = encode(s)
    mask = batch != 0
    return dict(text=text, lengths=lengths, batch=batch, mask=mask, real=int(mask.sum()),
                cells=int(batch.size), padding_share=float(1 - mask.mean()))


def masked_mean(n=TEXT_LINES):
    """With the pad's vector at 0 (``padding_idx=0``), the plain mean over a padded line is
    its true mean times ``length / L``: ``dict(lengths, longest, ratio)``, exact."""
    b = lines_batch(n)
    L = b["batch"].shape[1]
    return dict(lengths=b["lengths"], longest=L,
                ratio=[length / L for length in b["lengths"]])


def waste(batch=BATCH, seed=SEED):
    """How much of every batch of ``batch`` lines is padding, over all the lines: in file
    order, shuffled, and sorted by length (region "pad-sort"). ``dict(lines, characters,
    file, shuffled, sorted)``, each ``dict(cells, padding, share)``."""
    ln = np.array([len(s) for s in lines()])

    def cost(lengths):
        cells = 0
        for i in range(0, len(lengths), batch):
            g = lengths[i:i + batch]
            cells += len(g) * int(g.max())
        return dict(cells=int(cells), padding=int(cells - lengths.sum()),
                    share=float(1 - lengths.sum() / cells))

    rng = np.random.default_rng([S.SEED, 4, int(seed)])
    return dict(lines=len(ln), characters=int(ln.sum()), shortest=int(ln.min()),
                longest=int(ln.max()), file=cost(ln), shuffled=cost(rng.permutation(ln)),
                sorted=cost(np.sort(ln, kind="stable")))


def summary():
    """The headline numbers, JSON-able."""
    c = compare()
    for k in ("steps", "plain_curve", "padded_curve"):
        c.pop(k)
    sh = shifted()
    w = waste()
    cr = crops()
    lb = lines_batch()
    return dict(bg=BG, same_and_valid=same_and_valid(), deep=deep(),
                corner=int(coverage()[0, 0]), edge=int(coverage()[0, 5]),
                middle=int(coverage()[14, 14]),
                pad_side_walk=pad_side_walk(), **c,
                shifted_plain=[float(v) for v in sh["plain_mean"]],
                shifted_padded=[float(v) for v in sh["padded_mean"]],
                crops=cr["shapes"], block=cr["block_shape"], block_padding=cr["padding"],
                text_batch=list(lb["batch"].shape), text_real=lb["real"],
                text_padding_share=lb["padding_share"],
                waste={k: w[k]["share"] for k in ("file", "shuffled", "sorted")})

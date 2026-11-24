"""Thinking and effort, measured on a model small enough to see (``examples/thinking.py`` in
numpy), for the "thinking" animations of the hosting series.

The task: the parity of 12 bits (1 if the number of ones is odd). A sequence is
``[effort token] b1 ... b12 "=" s1 ... sk answer``: before the answer the model may write ``k``
steps, the parity so far after each group of ``12 / k`` bits, where ``k`` is the effort:
0, 1, 2, 3, 4, 6 or 12 (:data:`EFFORTS`). One tiny transformer (:func:`forward`: 2 pre-norm
layers of width 64 with 4 heads, causal attention, a ReLU MLP of 256; token and position
embeddings; 103,114 numbers) is trained on every effort with next-token cross-entropy on the
places from "=" on (Adam, lr 0.003, 6,000 steps of 256), on the 3,628 bit strings not held
out; then it is tested on the 468 held-out strings, writing its own steps and answer one
token at a time (greedy), at every effort (:func:`evaluate`).

What it shows: a step is a token like any other, written by the model and read back by it;
an answer needs one pass through the model per token, so effort ``k`` costs ``k + 1`` passes;
and whether the answer is right depends on how much each step must do at once. Everything is
from one run (one seed); where the cliff falls is this model's, not a law.

``tests/test_thinking_equivalence.py`` loads :func:`init_params` into the script's model and
matches the loss, every gradient and Adam's first step on :func:`batch` ``(0)``.
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
BITS = 12
EFFORTS = (0, 1, 2, 3, 4, 6, 12)
EQUALS = 2
VOCAB = 3 + len(EFFORTS)
LONGEST = 1 + BITS + 1 + max(EFFORTS) + 1          # 27
D, HEADS, LAYERS = 64, 4, 2
BATCH, STEPS, LR = 256, 6000, 3e-3
HELD = 468
SNAPS = (0, 500, 2000, 6000)
LN_EPS = 1e-5
SEED = 260


# ====================================================================== the data
def example(bits, e):
    """``example`` of the script: the full sequence for ``bits`` at effort index ``e``."""
    steps = EFFORTS[e]
    seq = [3 + e] + list(bits) + [EQUALS]
    group = BITS // max(steps, 1)
    for k in range(1, steps + 1):
        seq.append(sum(bits[:k * group]) % 2)
    seq.append(sum(bits) % 2)
    return seq


@functools.lru_cache(maxsize=1)
def strings():
    """``(every [4096, 12] bit string, held [468] the held-out rows)``: string ``n``'s bit
    ``i`` is ``(n >> i) & 1``, as in the script."""
    every = (np.arange(2 ** BITS)[:, None] >> np.arange(BITS)) & 1
    held = np.random.default_rng([SEED, 0]).permutation(2 ** BITS)[:HELD]
    return every, np.sort(held)


def batch(step):
    """Step ``step``'s 256 training examples: ``dict(x [256, 27], learn [256, 27], efforts)``
    (padded with 0; ``learn`` marks the places scored, from "=" on)."""
    every, held = strings()
    seen = np.setdiff1d(np.arange(2 ** BITS), held)
    rng = np.random.default_rng([SEED, 1, int(step)])
    rows = seen[rng.integers(0, len(seen), BATCH)]
    eff = rng.integers(0, len(EFFORTS), BATCH)
    x = np.zeros((BATCH, LONGEST), np.int64)
    learn = np.zeros((BATCH, LONGEST), np.float32)
    for i, (r, e) in enumerate(zip(rows, eff, strict=True)):
        s = example(every[r].tolist(), int(e))
        x[i, :len(s)] = s
        learn[i, 1 + BITS:len(s) - 1] = 1
    return dict(x=x, learn=learn, efforts=eff)


# ====================================================================== the model
def keys():
    out = ["tok.weight", "pos.weight"]
    for i in range(LAYERS):
        p = "blocks.%d." % i
        out += [p + k for k in ("n1.weight", "n1.bias", "n2.weight", "n2.bias", "qkv.weight",
                                "qkv.bias", "out.weight", "out.bias", "mlp.0.weight",
                                "mlp.0.bias", "mlp.2.weight", "mlp.2.bias")]
    return tuple(out + ["norm.weight", "norm.bias", "head.weight", "head.bias"])


def init_params(seed=0):
    """As PyTorch starts them: embeddings normal (spread 1), LayerNorms 1 and 0, Linear layers
    uniform in ``±1/sqrt(fan_in)``; keyed as ``model.state_dict()``."""
    rng = np.random.default_rng([SEED, 2, int(seed)])
    P = {"tok.weight": rng.normal(0, 1, (VOCAB, D)), "pos.weight": rng.normal(0, 1, (LONGEST, D))}

    def linear(name, o, i):
        b = 1 / np.sqrt(i)
        P[name + ".weight"] = rng.uniform(-b, b, (o, i))
        P[name + ".bias"] = rng.uniform(-b, b, o)

    for l in range(LAYERS):
        p = "blocks.%d." % l
        for n in ("n1", "n2"):
            P[p + n + ".weight"], P[p + n + ".bias"] = np.ones(D), np.zeros(D)
        linear(p + "qkv", 3 * D, D)
        linear(p + "out", D, D)
        linear(p + "mlp.0", 4 * D, D)
        linear(p + "mlp.2", D, 4 * D)
    P["norm.weight"], P["norm.bias"] = np.ones(D), np.zeros(D)
    linear("head", VOCAB, D)
    return {k: P[k].astype(np.float32) for k in keys()}


def count(P=None):
    P = init_params() if P is None else P
    return int(sum(v.size for v in P.values()))


def _ln(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    inv = 1 / np.sqrt(var + LN_EPS)
    xh = (x - mu) * inv
    return xh * g + b, (xh, inv, g)


def _ln_back(dy, c):
    xh, inv, g = c
    dxh = dy * g
    dx = inv * (dxh - dxh.mean(-1, keepdims=True) - xh * (dxh * xh).mean(-1, keepdims=True))
    return dx, (dy * xh).reshape(-1, xh.shape[-1]).sum(0), dy.reshape(-1, dy.shape[-1]).sum(0)


def forward(P, x, keep=False):
    """``model(x)``: ``[B, T, VOCAB]`` scores for the next token at every place."""
    x = np.asarray(x)
    B, T = x.shape
    h = P["tok.weight"][x] + P["pos.weight"][:T]
    caches = []
    mask = np.triu(np.ones((T, T), bool), 1)
    dh = D // HEADS
    for l in range(LAYERS):
        p = "blocks.%d." % l
        a1, c1 = _ln(h, P[p + "n1.weight"], P[p + "n1.bias"])
        qkv = a1 @ P[p + "qkv.weight"].T + P[p + "qkv.bias"]
        q, k, v = (t.reshape(B, T, HEADS, dh).transpose(0, 2, 1, 3)
                   for t in np.split(qkv, 3, -1))
        s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh).astype(q.dtype)
        s = np.where(mask, -np.inf, s)
        pr = np.exp(s - s.max(-1, keepdims=True))
        pr /= pr.sum(-1, keepdims=True)
        o = (pr @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        h1 = h + o @ P[p + "out.weight"].T + P[p + "out.bias"]
        a2, c2 = _ln(h1, P[p + "n2.weight"], P[p + "n2.bias"])
        z = a2 @ P[p + "mlp.0.weight"].T + P[p + "mlp.0.bias"]
        r = np.maximum(z, 0)
        h2 = h1 + r @ P[p + "mlp.2.weight"].T + P[p + "mlp.2.bias"]
        caches.append((a1, c1, q, k, v, pr, o, a2, c2, z, r))
        h = h2
    hf, cf = _ln(h, P["norm.weight"], P["norm.bias"])
    out = (hf @ P["head.weight"].T + P["head.bias"]).astype(P["head.weight"].dtype)
    return (out, (x, caches, hf, cf)) if keep else out


def loss_and_grads(P, x, learn):
    """Region "th-train"'s loss on ``model(x[:, :-1])`` and every parameter's gradient."""
    inp, tgt, w = x[:, :-1], x[:, 1:], learn[:, :-1]
    out, (inp, caches, hf, cf) = forward(P, inp, keep=True)
    B, T, _ = out.shape
    total = w.sum()
    each = S.loss_each(out.reshape(-1, VOCAB), tgt.reshape(-1)).reshape(B, T)
    loss = float((each * w).sum() / total)
    d = S._softmax(out.reshape(-1, VOCAB)).reshape(B, T, VOCAB).astype(out.dtype)
    d[np.arange(B)[:, None], np.arange(T)[None], tgt] -= 1
    d *= (w / total)[..., None]
    G = {k: np.zeros_like(v) for k, v in P.items()}
    G["head.weight"] = d.reshape(-1, VOCAB).T @ hf.reshape(-1, D)
    G["head.bias"] = d.reshape(-1, VOCAB).sum(0)
    dh_, G["norm.weight"], G["norm.bias"] = _ln_back(d @ P["head.weight"], cf)
    dhd = D // HEADS
    for l in reversed(range(LAYERS)):
        p = "blocks.%d." % l
        a1, c1, q, k, v, pr, o, a2, c2, z, r = caches[l]
        # MLP
        G[p + "mlp.2.weight"] = dh_.reshape(-1, D).T @ r.reshape(-1, 4 * D)
        G[p + "mlp.2.bias"] = dh_.reshape(-1, D).sum(0)
        dr = (dh_ @ P[p + "mlp.2.weight"]) * (z > 0)
        G[p + "mlp.0.weight"] = dr.reshape(-1, 4 * D).T @ a2.reshape(-1, D)
        G[p + "mlp.0.bias"] = dr.reshape(-1, 4 * D).sum(0)
        da2, G[p + "n2.weight"], G[p + "n2.bias"] = _ln_back(dr @ P[p + "mlp.0.weight"], c2)
        dh1 = dh_ + da2
        # attention
        G[p + "out.weight"] = dh1.reshape(-1, D).T @ o.reshape(-1, D)
        G[p + "out.bias"] = dh1.reshape(-1, D).sum(0)
        do = (dh1 @ P[p + "out.weight"]).reshape(B, T, HEADS, dhd).transpose(0, 2, 1, 3)
        dpr = do @ v.transpose(0, 1, 3, 2)
        dv = pr.transpose(0, 1, 3, 2) @ do
        ds = pr * (dpr - (dpr * pr).sum(-1, keepdims=True)) / np.sqrt(dhd).astype(pr.dtype)
        dq = ds @ k
        dk = ds.transpose(0, 1, 3, 2) @ q
        dqkv = np.concatenate([t.transpose(0, 2, 1, 3).reshape(B, T, D) for t in (dq, dk, dv)],
                              -1)
        G[p + "qkv.weight"] = dqkv.reshape(-1, 3 * D).T @ a1.reshape(-1, D)
        G[p + "qkv.bias"] = dqkv.reshape(-1, 3 * D).sum(0)
        da1, G[p + "n1.weight"], G[p + "n1.bias"] = _ln_back(dqkv @ P[p + "qkv.weight"], c1)
        dh_ = dh1 + da1
    np.add.at(G["tok.weight"], inp.reshape(-1), dh_.reshape(-1, D))
    G["pos.weight"][:T] = dh_.sum(0)
    return loss, {k: v.astype(P[k].dtype) for k, v in G.items()}


def generate(P, bits, e):
    """``answer`` of the script, for many bit strings at once (``bits [N, 12]``): the model
    writes its ``EFFORTS[e]`` steps and its answer greedily (0 or 1); ``[N, 14 + k + 1]``."""
    bits = np.asarray(bits)
    seq = np.concatenate([np.full((len(bits), 1), 3 + e), bits,
                          np.full((len(bits), 1), EQUALS)], 1)
    for _ in range(EFFORTS[e] + 1):
        nxt = forward(P, seq)[:, -1, :2].argmax(-1)
        seq = np.concatenate([seq, nxt[:, None]], 1)
    return seq


# ====================================================================== the run
def _train():
    P = init_params()
    state = S.adam_start(P)
    out = dict(loss=[])
    for step in range(STEPS):
        if step in SNAPS:
            out.update({"W%d__%s" % (step, k): P[k].copy() for k in P})
        b = batch(step)
        loss, G = loss_and_grads(P, b["x"], b["learn"])
        if step == 0:
            out.update({"G0__" + k: G[k] for k in P})
        out["loss"].append(loss)
        P, state = S.adam_update(P, G, state, lr=LR)
        if step == 0:
            out.update({"W1__" + k: P[k].copy() for k in P})
    out.update({"W%d__%s" % (STEPS, k): P[k] for k in P})
    every, held = strings()
    for snap in SNAPS:
        Q = P if snap == STEPS else {k: out["W%d__%s" % (snap, k)] for k in P}
        right = []
        for e in range(len(EFFORTS)):
            seqs = generate(Q, every[held], e)
            right.append(int((seqs[:, -1] == every[held].sum(1) % 2).sum()))
            if snap == STEPS:
                out["seqs%d" % e] = seqs
                want = np.array([example(b.tolist(), e) for b in every[held]])
                out["steps_right%d" % e] = (seqs[:, 14:-1] == want[:, 14:-1]).all(1)
        out["right%d" % snap] = np.array(right)
    return {k: np.asarray(v) for k, v in out.items()}


def _settings():
    return dict(version=VERSION, bits=BITS, efforts=EFFORTS, d=D, heads=HEADS, layers=LAYERS,
                batch=BATCH, steps=STEPS, lr=LR, held=HELD, snaps=SNAPS,
                source=hashlib.sha256(inspect.getsource(sys.modules[__name__]).encode())
                .hexdigest()[:16])


@functools.lru_cache(maxsize=1)
def run():
    """The training run and the tests (cached)."""
    return {k: np.asarray(v) for k, v in cached_npz("study-thinking", _settings(),
                                                    _train).items()}


def params(step=STEPS):
    if step == 0:
        return init_params()
    pre = "W1__" if step == 1 else "W%d__" % step
    return {k: run()[pre + k] for k in keys()}


def results():
    """Per effort, on the 468 held-out strings: answers right, steps all right, and the
    tokens the model writes (``k + 1``) and passes through the model it makes."""
    r = run()
    out = []
    for e, k in enumerate(EFFORTS):
        out.append(dict(effort=k, bits_per_step=BITS // max(k, 1) if k else BITS,
                        right=int(r["right%d" % STEPS][e]), of=HELD,
                        steps_right=int(r["steps_right%d" % e].sum()), tokens=k + 1,
                        passes=k + 1))
    return out


def multiply_adds_per_token(T=LONGEST - 1):
    """The model's multiply-adds to produce one token with ``T`` tokens before it (exact,
    counting the matrix products of every layer and the attention over the ``T`` places)."""
    per_layer = D * 3 * D + T * D * 2 + D * D + D * 4 * D * 2
    return int(LAYERS * per_layer + D * VOCAB)


def summary():
    return dict(parameters=count(), first_loss=float(run()["loss"][0]),
                last_loss=float(run()["loss"][-100:].mean()), results=results(),
                over_training={int(s): run()["right%d" % s].tolist() for s in SNAPS},
                multiply_adds_per_token=multiply_adds_per_token())

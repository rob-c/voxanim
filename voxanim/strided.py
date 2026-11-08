"""A small classifier built from *strided* convolutions, in plain numpy.

    x 28x28x1 -> conv 3x3 stride 2, 4 filters -> ReLU -> 13x13x4
              -> conv 3x3 stride 2, 8 filters -> ReLU ->  6x6x8
              -> dense 288 -> 10 -> softmax

Each stride-2 convolution reads every other window, so the data per channel
drops 784 -> 169 -> 36 while the channel count grows: no pooling layers are
needed to shrink the image.  Weights are stored ``(kh, kw, C_in, C_out)``.
Training is seeded and cached in ``build/.cache``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .mnist import load
from .scene import DEFAULT_CACHE

LAYERS = ((3, 2, 1, 4), (3, 2, 4, 8))          # kernel, stride, C_in, C_out


def out_size(n, k, s):
    return (n - k) // s + 1


def _windows(x, k, s):
    """``(N, H, W, C)`` -> ``(N, Ho, Wo, k*k*C)`` patches, rows (a, b, c)."""
    v = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(1, 2))[:, ::s, ::s]
    n, ho, wo, c = v.shape[:4]
    return v.transpose(0, 1, 2, 4, 5, 3).reshape(n, ho, wo, k * k * c)


def conv(x, W, b, s):
    k = W.shape[0]
    cols = _windows(x, k, s)
    return cols @ W.reshape(-1, W.shape[-1]) + b, cols


def conv_backward(dz, cols, W, x_shape, s):
    """Gradients of a strided valid convolution: (dW, db, dx)."""
    k, _, c, f = W.shape
    dW = cols.reshape(-1, cols.shape[-1]).T @ dz.reshape(-1, f)
    db = dz.sum((0, 1, 2))
    dcols = (dz @ W.reshape(-1, f).T).reshape(dz.shape[:3] + (k, k, c))
    dx = np.zeros(x_shape)
    ho, wo = dz.shape[1:3]
    for a in range(k):
        for bb in range(k):
            dx[:, a:a + s * (ho - 1) + 1:s, bb:bb + s * (wo - 1) + 1:s] += dcols[:, :, :, a, bb]
    return dW.reshape(W.shape), db, dx


def init(rng):
    P = {}
    for n, (k, _s, c, f) in enumerate(LAYERS):
        P["W%d" % n] = rng.normal(0, np.sqrt(2.0 / (k * k * c)), (k, k, c, f))
        P["b%d" % n] = np.zeros(f)
    k, _s, c, f = LAYERS[-1]
    size = 28
    for k_, s_, _c, _f in LAYERS:
        size = out_size(size, k_, s_)
    P["Wd"] = rng.normal(0, np.sqrt(1.0 / (size * size * f)), (size * size * f, 10))
    P["bd"] = np.zeros(10)
    return P


def forward(P, x):
    a = x.reshape(-1, 28, 28, 1)
    f = dict(acts=[a], zs=[], cols=[])
    for n, (_k, s, _c, _f) in enumerate(LAYERS):
        z, cols = conv(a, P["W%d" % n], P["b%d" % n], s)
        a = np.maximum(z, 0.0)
        f["zs"].append(z)
        f["cols"].append(cols)
        f["acts"].append(a)
    flat = a.reshape(len(a), -1)
    logits = flat @ P["Wd"] + P["bd"]
    p = np.exp(logits - logits.max(-1, keepdims=True))
    f.update(flat=flat, logits=logits, p=p / p.sum(-1, keepdims=True))
    return f


def backward(P, f, y_onehot):
    g = {}
    d = (f["p"] - y_onehot) / len(f["p"])
    g["Wd"], g["bd"] = f["flat"].T @ d, d.sum(0)
    da = (d @ P["Wd"].T).reshape(f["acts"][-1].shape)
    for n in range(len(LAYERS) - 1, -1, -1):
        dz = da * (f["zs"][n] > 0)
        g["W%d" % n], g["b%d" % n], da = conv_backward(dz, f["cols"][n], P["W%d" % n],
                                                      f["acts"][n].shape, LAYERS[n][1])
    return g


def train(epochs=3, batch=64, lr=2e-3, seed=0, cache_dir=DEFAULT_CACHE):
    tag = "strided_%s_e%d_s%d.npz" % ("-".join("%dx%ds%d" % (k, f, s) for k, s, _c, f in LAYERS),
                                      epochs, seed)
    path = os.path.join(cache_dir, tag)
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files if k != "acc"}, float(d["acc"])
    xtr, ytr, xte, yte = load(cache_dir)
    rng = np.random.default_rng(seed)
    P = init(rng)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(a) for k, a in P.items()}
    eye = np.eye(10)
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(xtr))
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            g = backward(P, forward(P, xtr[idx]), eye[ytr[idx]])
            step += 1
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                P[k] = P[k] - lr * (m[k] / (1 - 0.9 ** step)) / (
                    np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
    acc = float(np.mean([(forward(P, xte[s:s + 1000])["p"].argmax(1) == yte[s:s + 1000]).mean()
                         for s in range(0, len(xte), 1000)]))
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, acc=acc, **P)
    return P, acc


@dataclass
class StridedTrace:
    x: np.ndarray            # (28, 28)
    label: int
    P: dict
    acts: list               # [x (28,28,1), a1 (13,13,4), a2 (6,6,8)]
    zs: list                 # pre-activations
    logits: np.ndarray
    p: np.ndarray
    accuracy: float

    @property
    def prediction(self):
        return int(self.p.argmax())


def trace(index, **kw) -> StridedTrace:
    P, acc = train(**kw)
    _a, _b, xte, yte = load()
    f = forward(P, xte[index:index + 1])
    return StridedTrace(x=xte[index].reshape(28, 28), label=int(yte[index]), P=P,
                        acts=[a[0] for a in f["acts"]], zs=[z[0] for z in f["zs"]],
                        logits=f["logits"][0], p=f["p"][0], accuracy=acc)

"""MNIST and a small fully-connected classifier, in plain numpy.

The animations show real numbers, so this trains a real network:

    x (784) -> dense 64 -> ReLU -> dense 32 -> ReLU -> dense 10 -> softmax

with cross-entropy against a one-hot label.  The dataset (the standard Keras
mirror, ~11 MB) and the trained weights are cached in ``build/.cache``;
training is seeded and takes a few seconds.

:func:`trace` runs one image forward and backward and returns every
intermediate the animations draw.
"""

from __future__ import annotations

import itertools
import os
import urllib.request
from dataclasses import dataclass

import numpy as np

from .scene import DEFAULT_CACHE

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SIZES = (784, 64, 32, 10)


def load(cache_dir=DEFAULT_CACHE):
    """``(x_train, y_train, x_test, y_test)`` with pixels scaled to [0, 1]."""
    path = os.path.join(cache_dir, "mnist.npz")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        tmp = path + ".part"
        urllib.request.urlretrieve(URL, tmp)
        os.replace(tmp, path)
    with np.load(path, allow_pickle=False) as d:
        return (d["x_train"].reshape(-1, 784) / 255.0, d["y_train"],
                d["x_test"].reshape(-1, 784) / 255.0, d["y_test"])


def _init(rng, sizes=SIZES):
    params = []
    for a, b in itertools.pairwise(sizes):
        params.append([rng.normal(0, np.sqrt(2.0 / a), (a, b)), np.zeros(b)])
    return params


def _forward(params, x):
    acts, pre = [x], []
    for n, (W, b) in enumerate(params):
        z = acts[-1] @ W + b
        pre.append(z)
        if n < len(params) - 1:
            acts.append(np.maximum(z, 0.0))
    z = pre[-1] - pre[-1].max(axis=-1, keepdims=True)
    p = np.exp(z)
    return acts, pre, p / p.sum(axis=-1, keepdims=True)


def _backward(params, acts, pre, p, y_onehot):
    """Gradients of the mean cross-entropy; also the per-layer deltas."""
    n = len(params)
    delta = (p - y_onehot) / len(p)
    grads, deltas, dacts = [None] * n, [None] * n, [None] * n
    for k in range(n - 1, -1, -1):
        deltas[k] = delta
        W, _b = params[k]
        grads[k] = (acts[k].T @ delta, delta.sum(0))
        da = delta @ W.T
        dacts[k] = da
        if k > 0:
            delta = da * (pre[k - 1] > 0)
    return grads, deltas, dacts


def train(epochs=4, batch=128, lr=1e-3, seed=0, cache_dir=DEFAULT_CACHE, sizes=SIZES):
    """Train (Adam) or load the cached weights.  Returns ``(params, accuracy)``."""
    tag = "mlp_%s_e%d_b%d_s%d.npz" % ("-".join(map(str, sizes)), epochs, batch, seed)
    path = os.path.join(cache_dir, tag)
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as d:
            params = [[d["W%d" % k], d["b%d" % k]] for k in range(len(sizes) - 1)]
            return params, float(d["acc"])
    xtr, ytr, xte, yte = load(cache_dir)
    rng = np.random.default_rng(seed)
    params = _init(rng, sizes)
    m = [[np.zeros_like(a) for a in layer] for layer in params]
    v = [[np.zeros_like(a) for a in layer] for layer in params]
    eye = np.eye(10)
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(xtr))
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            acts, pre, p = _forward(params, xtr[idx])
            grads, _, _ = _backward(params, acts, pre, p, eye[ytr[idx]])
            step += 1
            for k, g in enumerate(grads):
                for j in (0, 1):
                    m[k][j] = 0.9 * m[k][j] + 0.1 * g[j]
                    v[k][j] = 0.999 * v[k][j] + 0.001 * g[j] ** 2
                    mh = m[k][j] / (1 - 0.9 ** step)
                    vh = v[k][j] / (1 - 0.999 ** step)
                    params[k][j] = params[k][j] - lr * mh / (np.sqrt(vh) + 1e-8)
    acc = float((_forward(params, xte)[2].argmax(1) == yte).mean())
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, acc=acc, **{"W%d" % k: W for k, (W, _) in enumerate(params)},
             **{"b%d" % k: b for k, (_, b) in enumerate(params)})
    return params, acc


@dataclass
class Trace:
    """One image through the network, forward and backward."""

    x: np.ndarray            # (784,) input pixels in [0, 1]
    label: int
    Ws: list                 # weight matrices (in, out)
    bs: list
    zs: list                 # pre-activations per layer
    acts: list               # [x, h1, h2]: the input of each dense layer
    p: np.ndarray            # softmax probabilities
    loss: float
    deltas: list             # dL/dz per layer
    dWs: list                # dL/dW per layer
    dacts: list              # dL/d(input of layer k)
    accuracy: float

    @property
    def prediction(self):
        return int(self.p.argmax())


def _make_trace(params, x, label, accuracy) -> Trace:
    acts, pre, p = _forward(params, x[None])
    grads, deltas, dacts = _backward(params, acts, pre, p, np.eye(10)[[label]])
    return Trace(x=x, label=label, Ws=[W for W, _ in params], bs=[b for _, b in params],
                 zs=[z[0] for z in pre], acts=[a[0] for a in acts], p=p[0],
                 loss=float(-np.log(p[0, label] + 1e-12)), deltas=[d[0] for d in deltas],
                 dWs=[g[0] for g in grads], dacts=[d[0] for d in dacts], accuracy=accuracy)


def trace(index=0, **train_kw) -> Trace:
    """Forward + backward for test image ``index`` (0 is the classic '7')."""
    params, acc = train(**train_kw)
    _xtr, _ytr, xte, yte = load()
    return _make_trace(params, xte[index], int(yte[index]), acc)


def sgd_step(tr: Trace, lr) -> Trace:
    """The same image after one plain-SGD step on it: ``W <- W - lr * dW``."""
    params = [[W - lr * dW, b - lr * d] for W, b, dW, d in zip(tr.Ws, tr.bs, tr.dWs, tr.deltas)]
    return _make_trace(params, tr.x, tr.label, tr.accuracy)


# ------------------------------------------------------------------ a small CNN
#
#   x 28x28 -> conv 3x3, 4 filters -> ReLU -> max-pool 2x2 -> dense 676 -> 10 -> softmax

CNN_FILTERS = 4


def _im2col(x):
    """``(N, 28, 28)`` -> ``(N, 676, 9)``: every 3x3 patch, row-major."""
    n = x.shape[0]
    s = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(1, 2))   # N,26,26,3,3
    return s.reshape(n, 26 * 26, 9)


def _cnn_forward(params, x):
    (Wc, bc), (Wd, bd) = params
    cols = _im2col(x.reshape(-1, 28, 28))
    z1 = (cols @ Wc.reshape(9, CNN_FILTERS) + bc).reshape(-1, 26, 26, CNN_FILTERS)
    a1 = np.maximum(z1, 0.0)
    win = a1.reshape(-1, 13, 2, 13, 2, CNN_FILTERS).transpose(0, 1, 3, 5, 2, 4)
    win = win.reshape(-1, 13, 13, CNN_FILTERS, 4)
    arg = win.argmax(-1)                                   # which of the 2x2 won
    pooled = win.max(-1)                                   # N,13,13,F
    flat = pooled.reshape(len(x), -1)
    z2 = flat @ Wd + bd
    z = z2 - z2.max(-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(-1, keepdims=True)
    return dict(cols=cols, z1=z1, a1=a1, arg=arg, pooled=pooled, flat=flat, z2=z2, p=p)


def _cnn_backward(params, f, y_onehot):
    (_Wc, _bc), (Wd, _bd) = params
    n = len(f["p"])
    d2 = (f["p"] - y_onehot) / n
    dWd, dbd = f["flat"].T @ d2, d2.sum(0)
    dpool = (d2 @ Wd.T).reshape(-1, 13, 13, CNN_FILTERS)
    # max-pool routes each gradient to the cell that won its 2x2 window
    route = np.eye(4)[f["arg"]] * dpool[..., None]          # N,13,13,F,4
    route = route.reshape(-1, 13, 13, CNN_FILTERS, 2, 2).transpose(0, 1, 4, 2, 5, 3)
    da1 = route.reshape(-1, 26, 26, CNN_FILTERS)
    d1 = da1 * (f["z1"] > 0)                                # ReLU
    dWc = (f["cols"].reshape(-1, 9).T @ d1.reshape(-1, CNN_FILTERS)).reshape(3, 3, CNN_FILTERS)
    return dict(dWc=dWc, dbc=d1.sum((0, 1, 2)), dWd=dWd, dbd=dbd, d2=d2, dpool=dpool,
                da1=da1, d1=d1)


def cnn_train(epochs=2, batch=64, lr=2e-3, seed=0, cache_dir=DEFAULT_CACHE):
    """Train (Adam) or load the cached CNN.  Returns ``(params, accuracy)``."""
    path = os.path.join(cache_dir, "cnn%d_e%d_b%d_s%d.npz" % (CNN_FILTERS, epochs, batch, seed))
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as d:
            return [[d["Wc"], d["bc"]], [d["Wd"], d["bd"]]], float(d["acc"])
    xtr, ytr, xte, yte = load(cache_dir)
    rng = np.random.default_rng(seed)
    params = [[rng.normal(0, np.sqrt(2 / 9), (3, 3, CNN_FILTERS)), np.zeros(CNN_FILTERS)],
              [rng.normal(0, np.sqrt(2 / 676), (676, 10)), np.zeros(10)]]
    m = [[np.zeros_like(a) for a in layer] for layer in params]
    v = [[np.zeros_like(a) for a in layer] for layer in params]
    eye = np.eye(10)
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(xtr))
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            f = _cnn_forward(params, xtr[idx])
            g = _cnn_backward(params, f, eye[ytr[idx]])
            grads = [(g["dWc"], g["dbc"]), (g["dWd"], g["dbd"])]
            step += 1
            for k in range(2):
                for j in (0, 1):
                    m[k][j] = 0.9 * m[k][j] + 0.1 * grads[k][j]
                    v[k][j] = 0.999 * v[k][j] + 0.001 * grads[k][j] ** 2
                    mh = m[k][j] / (1 - 0.9 ** step)
                    vh = v[k][j] / (1 - 0.999 ** step)
                    params[k][j] = params[k][j] - lr * mh / (np.sqrt(vh) + 1e-8)
    acc = float(np.mean([(_cnn_forward(params, xte[s:s + 1000])["p"].argmax(1)
                          == yte[s:s + 1000]).mean() for s in range(0, len(xte), 1000)]))
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, acc=acc, Wc=params[0][0], bc=params[0][1], Wd=params[1][0], bd=params[1][1])
    return params, acc


@dataclass
class CnnTrace:
    """One image through the CNN, forward and backward (batch of one)."""

    x: np.ndarray              # (784,)
    label: int
    params: list               # [[Wc (3,3,F), bc], [Wd (676,10), bd]]
    fwd: dict                  # z1, a1 (26,26,F), pooled (13,13,F), arg, z2 (10,), p
    grad: dict                 # dWc, dbc, dWd, dbd, d2, dpool, da1, d1
    p: np.ndarray
    loss: float
    accuracy: float

    @property
    def prediction(self):
        return int(self.p.argmax())


def _cnn_trace(params, x, label, acc):
    f = _cnn_forward(params, x[None])
    g = _cnn_backward(params, f, np.eye(10)[[label]])
    f1 = {k: v[0] for k, v in f.items()}
    g1 = {k: (v[0] if k in ("d2", "dpool", "da1", "d1") else v) for k, v in g.items()}
    return CnnTrace(x=x, label=label, params=params, fwd=f1, grad=g1, p=f1["p"],
                    loss=float(-np.log(f1["p"][label] + 1e-12)), accuracy=acc)


def cnn_trace(index, **train_kw) -> CnnTrace:
    params, acc = cnn_train(**train_kw)
    _xtr, _ytr, xte, yte = load()
    return _cnn_trace(params, xte[index], int(yte[index]), acc)


def cnn_sgd_step(tr: CnnTrace, lr) -> CnnTrace:
    g = tr.grad
    (Wc, bc), (Wd, bd) = tr.params
    params = [[Wc - lr * g["dWc"], bc - lr * g["dbc"]], [Wd - lr * g["dWd"], bd - lr * g["dbd"]]]
    return _cnn_trace(params, tr.x, tr.label, tr.accuracy)

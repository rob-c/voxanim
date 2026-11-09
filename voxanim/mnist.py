"""MNIST and two small classifiers trained on it, in plain numpy.

The animations show real numbers, so they need real networks:

* an **MLP**, ``x (784) -> dense 64 -> ReLU -> dense 32 -> ReLU -> dense 10 ->
  softmax`` (:data:`SIZES`; other sizes are a keyword away), and
* a **CNN**, ``x 28x28 -> conv 3x3, 4 filters -> ReLU -> max-pool 2x2 ->
  dense 676 -> 10 -> softmax`` (:data:`CNN_FILTERS`),

both with cross-entropy against a one-hot label, He-normal initialisation and
minibatch Adam (:func:`fit`).  The dataset (the standard Keras mirror, ~11 MB,
pinned by SHA-256) and the trained weights are cached by :mod:`voxanim.cache`;
training is seeded, deterministic and takes seconds (MLP) to a minute (CNN).

:func:`trace` / :func:`cnn_trace` run one test image forward and backward
and return every intermediate the animations draw; :func:`sgd_step` /
:func:`cnn_sgd_step` return the same image after one plain-SGD step on it.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass

import numpy as np

from .cache import DEFAULT_CACHE, cached_npz, download
from .minibatch import chunk_mean, minibatches
from .model_io import forward as dense_forward
from .optim import Adam

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
SHA256 = "731c5ac602752760c8e48fbffcf8c3b850d9dc2a2aedcf2cc48468fc17b673d1"
TIMEOUT = 60                # seconds a stalled download may block before it fails
CLASSES = 10
SIZES = (784, 64, 32, 10)   # the MLP the mlp-* animations show


def load(cache_dir=None):
    """``(x_train, y_train, x_test, y_test)``: images ``(N, 784)`` in [0, 1], labels ``(N,)``.

    Downloads (and checks) the dataset on first use.
    """
    path = os.path.join(cache_dir or DEFAULT_CACHE, "mnist.npz")
    download(URL, path, SHA256, timeout=TIMEOUT)
    with np.load(path, allow_pickle=False) as d:
        return (d["x_train"].reshape(-1, 784) / 255.0, d["y_train"],
                d["x_test"].reshape(-1, 784) / 255.0, d["y_test"])


def fit(P, grads, x, y, epochs, batch, lr, rng):
    """Minibatch Adam: ``epochs`` shuffled passes over ``(x, y)``; updates ``P`` in place.

    ``P`` is a dict of parameter arrays and ``grads(P, x_batch, y_onehot)``
    returns a dict with the same keys.  Each epoch draws its order from
    ``rng``, so the result depends only on the initial ``P`` and the seed.
    """
    opt = Adam(P, lr)
    eye = np.eye(CLASSES)
    for idx in minibatches(rng, len(x), epochs, batch):
        opt.step(P, grads(P, x[idx], eye[y[idx]]))
    return P


def chunked_accuracy(predict, x, y, chunk=1000):
    """Mean over ``chunk``-sized slices of the per-slice accuracy of ``predict(x) -> p``."""
    return chunk_mean(lambda s: (predict(x[s]).argmax(1) == y[s]).mean(), len(x), chunk)


# ------------------------------------------------------------------ the MLP


def _init(rng, sizes=SIZES):
    """He-normal weights ``N(0, sqrt(2 / fan_in))`` and zero biases: ``[[W, b], ...]``."""
    params = []
    for a, b in itertools.pairwise(sizes):
        params.append([rng.normal(0, np.sqrt(2.0 / a), (a, b)), np.zeros(b)])
    return params


def _forward(params, x):
    """``(acts, pre, p)``: each dense layer's input, its pre-activation, and the softmax."""
    zs, acts, p = dense_forward(params, x)
    return acts, zs, p


def _backward(params, acts, pre, p, y_onehot):
    """Gradients of the mean cross-entropy; also the per-layer deltas.

    Returns ``(grads, deltas, dacts)``: ``grads[k] = (dW, db)``, ``deltas[k]``
    is dL/dz of layer ``k`` and ``dacts[k]`` dL/d(input of layer ``k``).
    """
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


def _as_layers(P, n):
    """``{"W0", "b0", ...}`` -> ``[[W0, b0], ...]`` for ``n`` layers."""
    return [[P["W%d" % k], P["b%d" % k]] for k in range(n)]


def _mlp_grads(P, x, y_onehot):
    n = len(P) // 2
    params = _as_layers(P, n)
    acts, pre, p = _forward(params, x)
    grads, _, _ = _backward(params, acts, pre, p, y_onehot)
    return {"%s%d" % (name, k): g[j] for k, g in enumerate(grads)
            for j, name in enumerate("Wb")}


def train(epochs=4, batch=128, lr=1e-3, seed=0, cache_dir=None, sizes=SIZES):
    """Train an MLP of layer ``sizes`` (or load it from the cache).

    Returns ``(params, accuracy)``: ``params`` is ``[[W (in, out), b], ...]``
    and ``accuracy`` the fraction of the 10 000 test images it gets right.
    """
    settings = dict(model="mlp", sizes=list(sizes), epochs=epochs, batch=batch, lr=lr,
                    seed=seed, init="he_normal", optimizer="adam", data="mnist/255")

    def build():
        xtr, ytr, xte, yte = load(cache_dir)
        rng = np.random.default_rng(seed)
        P = {"%s%d" % (name, k): a for k, layer in enumerate(_init(rng, sizes))
             for name, a in zip("Wb", layer, strict=True)}
        fit(P, _mlp_grads, xtr, ytr, epochs, batch, lr, rng)
        acc = float((_forward(_as_layers(P, len(sizes) - 1), xte)[2].argmax(1) == yte).mean())
        return dict(P, acc=np.array(acc))

    d = cached_npz("mlp_%s" % "-".join(map(str, sizes)), settings, build, cache_dir)
    return _as_layers(d, len(sizes) - 1), float(d["acc"])


@dataclass
class Trace:
    """One image through the MLP, forward and backward (lists run over layers)."""

    x: np.ndarray            # (784,) input pixels in [0, 1]
    label: int               # the true digit
    Ws: list                 # weight matrices (in, out)
    bs: list                 # biases (out,)
    zs: list                 # pre-activations per layer
    acts: list               # [x, h1, h2]: the input of each dense layer
    p: np.ndarray            # (10,) softmax probabilities
    loss: float              # cross-entropy -log p[label]
    deltas: list             # dL/dz per layer
    dWs: list                # dL/dW per layer
    dacts: list              # dL/d(input of layer k)
    accuracy: float          # test accuracy of the network

    @property
    def prediction(self):
        """The most probable digit."""
        return int(self.p.argmax())


def _make_trace(params, x, label, accuracy) -> Trace:
    acts, pre, p = _forward(params, x[None])
    grads, deltas, dacts = _backward(params, acts, pre, p, np.eye(CLASSES)[[label]])
    return Trace(x=x, label=label, Ws=[W for W, _ in params], bs=[b for _, b in params],
                 zs=[z[0] for z in pre], acts=[a[0] for a in acts], p=p[0],
                 loss=float(-np.log(p[0, label] + 1e-12)), deltas=[d[0] for d in deltas],
                 dWs=[g[0] for g in grads], dacts=[d[0] for d in dacts], accuracy=accuracy)


def trace(index=0, **train_kw) -> Trace:
    """Forward + backward for test image ``index`` (0 is the classic '7').

    ``train_kw`` go to :func:`train`.
    """
    params, acc = train(**train_kw)
    _xtr, _ytr, xte, yte = load()
    return _make_trace(params, xte[index], int(yte[index]), acc)


def sgd_step(tr: Trace, lr) -> Trace:
    """The same image after one plain-SGD step on it: ``W <- W - lr * dW``."""
    params = [[W - lr * dW, b - lr * d] for W, b, dW, d in zip(tr.Ws, tr.bs, tr.dWs, tr.deltas,
                                                              strict=True)]
    return _make_trace(params, tr.x, tr.label, tr.accuracy)


# ------------------------------------------------------------------ a small CNN
#
#   x 28x28 -> conv 3x3, 4 filters -> ReLU -> max-pool 2x2 -> dense 676 -> 10 -> softmax

CNN_FILTERS = 4
CONV_HW = 26                 # 28 - 3 + 1: valid 3x3 convolution
POOL_HW = 13                 # 26 / 2: 2x2 max-pool
CNN_DENSE_IN = POOL_HW * POOL_HW * CNN_FILTERS     # 676 inputs to the dense layer


def _im2col(x):
    """``(N, 28, 28)`` -> ``(N, 676, 9)``: every 3x3 patch, row-major."""
    n = x.shape[0]
    s = np.lib.stride_tricks.sliding_window_view(x, (3, 3), axis=(1, 2))   # N,26,26,3,3
    return s.reshape(n, CONV_HW * CONV_HW, 9)


def _cnn_forward(params, x):
    """Forward pass of the CNN on ``x`` ``(N, 784)``; a dict of every intermediate.

    ``z1``/``a1`` ``(N, 26, 26, F)`` (conv, ReLU), ``arg`` which cell of each
    2x2 window won, ``pooled`` ``(N, 13, 13, F)``, ``flat``, ``z2`` (logits),
    ``p`` (softmax) and ``cols`` (the im2col patches).
    """
    (Wc, bc), (Wd, bd) = params
    F = CNN_FILTERS
    cols = _im2col(x.reshape(-1, 28, 28))
    z1 = (cols @ Wc.reshape(9, F) + bc).reshape(-1, CONV_HW, CONV_HW, F)
    a1 = np.maximum(z1, 0.0)
    win = a1.reshape(-1, POOL_HW, 2, POOL_HW, 2, F).transpose(0, 1, 3, 5, 2, 4)
    win = win.reshape(-1, POOL_HW, POOL_HW, F, 4)
    arg = win.argmax(-1)                                   # which of the 2x2 won
    pooled = win.max(-1)                                   # N,13,13,F
    flat = pooled.reshape(len(x), -1)
    z2 = flat @ Wd + bd
    z = z2 - z2.max(-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(-1, keepdims=True)
    return dict(cols=cols, z1=z1, a1=a1, arg=arg, pooled=pooled, flat=flat, z2=z2, p=p)


def _cnn_backward(params, f, y_onehot):
    """Gradients of the mean cross-entropy from a :func:`_cnn_forward` dict.

    Parameter gradients ``dWc dbc dWd dbd`` and the intermediates the
    animation draws: ``d2`` (dL/dz2), ``dpool``, ``da1`` (after max-pool
    routing) and ``d1`` (after ReLU).
    """
    (_Wc, _bc), (Wd, _bd) = params
    F = CNN_FILTERS
    n = len(f["p"])
    d2 = (f["p"] - y_onehot) / n
    dWd, dbd = f["flat"].T @ d2, d2.sum(0)
    dpool = (d2 @ Wd.T).reshape(-1, POOL_HW, POOL_HW, F)
    # max-pool routes each gradient to the cell that won its 2x2 window
    route = np.eye(4)[f["arg"]] * dpool[..., None]          # N,13,13,F,4
    route = route.reshape(-1, POOL_HW, POOL_HW, F, 2, 2).transpose(0, 1, 4, 2, 5, 3)
    da1 = route.reshape(-1, CONV_HW, CONV_HW, F)
    d1 = da1 * (f["z1"] > 0)                                # ReLU
    dWc = (f["cols"].reshape(-1, 9).T @ d1.reshape(-1, F)).reshape(3, 3, F)
    return dict(dWc=dWc, dbc=d1.sum((0, 1, 2)), dWd=dWd, dbd=dbd, d2=d2, dpool=dpool,
                da1=da1, d1=d1)


_CNN_KEYS = (("Wc", "bc"), ("Wd", "bd"))


def _cnn_params(P):
    """``{"Wc", "bc", "Wd", "bd"}`` -> ``[[Wc, bc], [Wd, bd]]``."""
    return [[P[w], P[b]] for w, b in _CNN_KEYS]


def _cnn_grads(P, x, y_onehot):
    g = _cnn_backward(_cnn_params(P), _cnn_forward(_cnn_params(P), x), y_onehot)
    return {k: g["d" + k] for pair in _CNN_KEYS for k in pair}


def cnn_train(epochs=2, batch=64, lr=2e-3, seed=0, cache_dir=None):
    """Train the CNN (or load it from the cache).

    Returns ``(params, accuracy)`` with ``params = [[Wc (3, 3, F), bc (F,)],
    [Wd (676, 10), bd (10,)]]``.
    """
    settings = dict(model="cnn", filters=CNN_FILTERS, kernel=3, pool=2, epochs=epochs,
                    batch=batch, lr=lr, seed=seed, init="he_normal", optimizer="adam",
                    data="mnist/255")

    def build():
        xtr, ytr, xte, yte = load(cache_dir)
        rng = np.random.default_rng(seed)
        P = dict(Wc=rng.normal(0, np.sqrt(2 / 9), (3, 3, CNN_FILTERS)), bc=np.zeros(CNN_FILTERS),
                 Wd=rng.normal(0, np.sqrt(2 / CNN_DENSE_IN), (CNN_DENSE_IN, CLASSES)),
                 bd=np.zeros(CLASSES))
        fit(P, _cnn_grads, xtr, ytr, epochs, batch, lr, rng)
        acc = chunked_accuracy(lambda x: _cnn_forward(_cnn_params(P), x)["p"], xte, yte)
        return dict(P, acc=np.array(acc))

    d = cached_npz("cnn%d" % CNN_FILTERS, settings, build, cache_dir)
    return _cnn_params(d), float(d["acc"])


@dataclass
class CnnTrace:
    """One image through the CNN, forward and backward (batch of one)."""

    x: np.ndarray              # (784,)
    label: int                 # the true digit
    params: list               # [[Wc (3,3,F), bc], [Wd (676,10), bd]]
    fwd: dict                  # z1, a1 (26,26,F), pooled (13,13,F), arg, z2 (10,), p
    grad: dict                 # dWc, dbc, dWd, dbd, d2, dpool, da1, d1
    p: np.ndarray              # (10,) softmax probabilities
    loss: float                # cross-entropy -log p[label]
    accuracy: float            # test accuracy of the network

    @property
    def prediction(self):
        """The most probable digit."""
        return int(self.p.argmax())


def _cnn_trace(params, x, label, acc):
    f = _cnn_forward(params, x[None])
    g = _cnn_backward(params, f, np.eye(CLASSES)[[label]])
    f1 = {k: v[0] for k, v in f.items()}
    g1 = {k: (v[0] if k in ("d2", "dpool", "da1", "d1") else v) for k, v in g.items()}
    return CnnTrace(x=x, label=label, params=params, fwd=f1, grad=g1, p=f1["p"],
                    loss=float(-np.log(f1["p"][label] + 1e-12)), accuracy=acc)


def cnn_trace(index, **train_kw) -> CnnTrace:
    """Forward + backward of the CNN for test image ``index``; ``train_kw`` go to
    :func:`cnn_train`."""
    params, acc = cnn_train(**train_kw)
    _xtr, _ytr, xte, yte = load()
    return _cnn_trace(params, xte[index], int(yte[index]), acc)


def cnn_sgd_step(tr: CnnTrace, lr) -> CnnTrace:
    """The same image after one plain-SGD step of the CNN on it."""
    g = tr.grad
    (Wc, bc), (Wd, bd) = tr.params
    params = [[Wc - lr * g["dWc"], bc - lr * g["dbc"]], [Wd - lr * g["dWd"], bd - lr * g["dbd"]]]
    return _cnn_trace(params, tr.x, tr.label, tr.accuracy)

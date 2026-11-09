"""A small classifier built from *strided* convolutions, in plain numpy.

    x 28x28x1 -> conv 3x3 stride 2, 4 filters -> ReLU -> 13x13x4
              -> conv 3x3 stride 2, 8 filters -> ReLU ->  6x6x8
              -> dense 288 -> 10 -> softmax

Each stride-2 convolution reads every other window, so the data per channel
drops 784 -> 169 -> 36 while the channel count grows: no pooling layers are
needed to shrink the image.  Weights are stored ``(kh, kw, C_in, C_out)``.
Training (:func:`train`: He-normal init, minibatch Adam via
:func:`voxanim.mnist.fit`) is seeded and cached by :mod:`voxanim.cache`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import cached_npz
from .mnist import chunked_accuracy, fit, load

LAYERS = ((3, 2, 1, 4), (3, 2, 4, 8))          # kernel, stride, C_in, C_out


def out_size(n, k, s):
    """Output length of a valid convolution: ``n`` inputs, kernel ``k``, stride ``s``."""
    return (n - k) // s + 1


def _windows(x, k, s):
    """``(N, H, W, C)`` -> ``(N, Ho, Wo, k*k*C)`` patches, rows (a, b, c)."""
    v = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(1, 2))[:, ::s, ::s]
    n, ho, wo, c = v.shape[:4]
    return v.transpose(0, 1, 2, 4, 5, 3).reshape(n, ho, wo, k * k * c)


def conv(x, W, b, s):
    """Valid convolution of ``x`` ``(N, H, W, C)`` with ``W`` ``(k, k, C, F)``, stride ``s``.

    Returns ``(z, cols)``: the pre-activation ``(N, Ho, Wo, F)`` and the
    patches it was computed from (kept for :func:`conv_backward`).
    """
    k = W.shape[0]
    cols = _windows(x, k, s)
    return cols @ W.reshape(-1, W.shape[-1]) + b, cols


def conv_backward(dz, cols, W, x_shape, s):
    """Gradients of a strided valid convolution: ``(dW, db, dx)``.

    ``dz`` is dL/dz ``(N, Ho, Wo, F)``; ``cols`` and ``x_shape`` come from the
    forward pass.  Each tap ``(a, b)`` scatters its share back onto the input
    rows/cols it read.
    """
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
    """Initial parameters: He-normal convolutions, ``N(0, sqrt(1 / fan_in))`` dense, zero biases."""
    P = {}
    for n, (k, _s, c, f) in enumerate(LAYERS):
        P["W%d" % n] = rng.normal(0, np.sqrt(2.0 / (k * k * c)), (k, k, c, f))
        P["b%d" % n] = np.zeros(f)
    f = LAYERS[-1][3]                    # channels of the last feature map
    size = 28
    for k_, s_, _c, _f in LAYERS:
        size = out_size(size, k_, s_)
    P["Wd"] = rng.normal(0, np.sqrt(1.0 / (size * size * f)), (size * size * f, 10))
    P["bd"] = np.zeros(10)
    return P


def forward(P, x):
    """Forward pass on ``x`` ``(N, 784)``.

    Returns a dict: ``acts`` (each convolution's input, then the last
    feature map), ``zs`` and ``cols`` per convolution, ``flat``, ``logits``
    and the softmax ``p`` ``(N, 10)``.
    """
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
    """Gradients of the mean cross-entropy, a dict with the keys of ``P``."""
    g = {}
    d = (f["p"] - y_onehot) / len(f["p"])
    g["Wd"], g["bd"] = f["flat"].T @ d, d.sum(0)
    da = (d @ P["Wd"].T).reshape(f["acts"][-1].shape)
    for n in range(len(LAYERS) - 1, -1, -1):
        dz = da * (f["zs"][n] > 0)
        g["W%d" % n], g["b%d" % n], da = conv_backward(dz, f["cols"][n], P["W%d" % n],
                                                      f["acts"][n].shape, LAYERS[n][1])
    return g


def train(epochs=3, batch=64, lr=2e-3, seed=0, cache_dir=None):
    """Train the strided classifier (or load it from the cache).

    Returns ``(P, accuracy)``: ``P`` holds ``W0, b0, W1, b1`` (the
    convolutions, ``(k, k, C_in, C_out)``) and ``Wd, bd`` (the dense layer).
    """
    settings = dict(model="strided", layers=[list(layer) for layer in LAYERS], epochs=epochs,
                    batch=batch, lr=lr, seed=seed, init="he_normal conv, lecun_normal dense",
                    optimizer="adam", data="mnist/255")

    def build():
        xtr, ytr, xte, yte = load(cache_dir)
        rng = np.random.default_rng(seed)
        P = init(rng)
        fit(P, lambda Q, x, y: backward(Q, forward(Q, x), y), xtr, ytr, epochs, batch, lr, rng)
        acc = chunked_accuracy(lambda x: forward(P, x)["p"], xte, yte)
        return dict(P, acc=np.array(acc))

    tag = "strided_%s" % "-".join("%dx%ds%d" % (k, f, s) for k, s, _c, f in LAYERS)
    d = cached_npz(tag, settings, build, cache_dir)
    return {k: v for k, v in d.items() if k != "acc"}, float(d["acc"])


@dataclass
class StridedTrace:
    """One test image through the strided classifier (batch of one)."""

    x: np.ndarray            # (28, 28) pixels in [0, 1]
    label: int               # the true digit
    P: dict                  # the trained parameters (see :func:`train`)
    acts: list               # [x (28,28,1), a1 (13,13,4), a2 (6,6,8)]
    zs: list                 # pre-activations of the two convolutions
    logits: np.ndarray       # (10,)
    p: np.ndarray            # (10,) softmax probabilities
    accuracy: float          # test accuracy of the model

    @property
    def prediction(self):
        """The most probable digit."""
        return int(self.p.argmax())


def trace(index, **kw) -> StridedTrace:
    """Test image ``index`` through the trained model; ``kw`` go to :func:`train`."""
    P, acc = train(**kw)
    _a, _b, xte, yte = load()
    f = forward(P, xte[index:index + 1])
    return StridedTrace(x=xte[index].reshape(28, 28), label=int(yte[index]), P=P,
                        acts=[a[0] for a in f["acts"]], zs=[z[0] for z in f["zs"]],
                        logits=f["logits"][0], p=f["p"][0], accuracy=acc)

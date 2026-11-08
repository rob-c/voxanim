"""Autoencoders for MNIST in plain numpy: a dense one and a convolutional one,
each with a 2-D or 3-D latent space (the bottleneck the animations draw).

Dense:  784 -> 256 -> 64 -> L -> 64 -> 256 -> 784        (ReLU, sigmoid output)
Conv:   28x28x1 -> conv 3x3 s2 -> 14x14x8 -> conv 3x3 s2 -> 7x7x16 -> dense -> L
        L -> dense -> 7x7x16 -> up x2, conv 3x3 -> 14x14x8 -> up x2, conv 3x3 -> 28x28x1

Loss: mean squared error between the input and its reconstruction.  Training
is seeded (Adam) and cached in ``build/.cache`` like the other models.
"""

from __future__ import annotations

import itertools
import os
from dataclasses import dataclass

import numpy as np

from .mnist import load
from .scene import DEFAULT_CACHE
from .strided import _windows

DENSE = (784, 256, 64)                       # encoder widths before the latent


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


# ------------------------------------------------------------------ convolution helpers


def conv(x, W, b, stride, pad):
    """``(N, H, W, C)`` -> conv with zero ``pad`` and ``stride``; returns (z, cols, padded shape)."""
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    cols = _windows(xp, W.shape[0], stride)
    return cols @ W.reshape(-1, W.shape[-1]) + b, cols, xp.shape


def conv_back(dz, cols, W, xp_shape, stride, pad):
    k, _, c, f = W.shape
    dW = cols.reshape(-1, cols.shape[-1]).T @ dz.reshape(-1, f)
    db = dz.sum((0, 1, 2))
    dcols = (dz @ W.reshape(-1, f).T).reshape(dz.shape[:3] + (k, k, c))
    dxp = np.zeros(xp_shape)
    ho, wo = dz.shape[1:3]
    for a in range(k):
        for bb in range(k):
            dxp[:, a:a + stride * (ho - 1) + 1:stride,
                bb:bb + stride * (wo - 1) + 1:stride] += dcols[:, :, :, a, bb]
    dx = dxp[:, pad:xp_shape[1] - pad, pad:xp_shape[2] - pad] if pad else dxp
    return dW.reshape(W.shape), db, dx


def up2(x):
    return x.repeat(2, axis=1).repeat(2, axis=2)


def up2_back(d):
    n, h, w, c = d.shape
    return d.reshape(n, h // 2, 2, w // 2, 2, c).sum((2, 4))


# ------------------------------------------------------------------ models


def init(kind, latent, rng):
    P = {}

    def dense(name, a, b):
        P["W" + name] = rng.normal(0, np.sqrt(2.0 / a), (a, b))
        P["b" + name] = np.zeros(b)

    def convw(name, k, c, f):
        P["W" + name] = rng.normal(0, np.sqrt(2.0 / (k * k * c)), (k, k, c, f))
        P["b" + name] = np.zeros(f)

    if kind == "dense":
        widths = list(DENSE) + [latent] + list(DENSE[::-1])
        for n, (a, b) in enumerate(itertools.pairwise(widths)):
            dense(str(n), a, b)
    else:
        convw("e0", 3, 1, 8)
        convw("e1", 3, 8, 16)
        dense("z", 7 * 7 * 16, latent)
        dense("d", latent, 7 * 7 * 16)
        convw("d0", 3, 16, 8)
        convw("d1", 3, 8, 1)
    return P


def forward(kind, P, x):
    """Every intermediate, plus ``recon`` and ``z`` (the latent code)."""
    n = len(x)
    f = {}
    if kind == "dense":
        L = len([k for k in P if k.startswith("W")])
        a = x.reshape(n, -1)
        acts, pre = [a], []
        for k in range(L):
            zk = a @ P["W%d" % k] + P["b%d" % k]
            pre.append(zk)
            last = k == L - 1
            mid = k == len(DENSE) - 1                      # the latent layer: linear
            a = _sig(zk) if last else (zk if mid else np.maximum(zk, 0))
            acts.append(a)
        f.update(acts=acts, pre=pre, z=acts[len(DENSE)], recon=acts[-1])
        return f
    a0 = x.reshape(n, 28, 28, 1)
    z0, c0, s0 = conv(a0, P["We0"], P["be0"], 2, 1)          # 14x14x8
    a1 = np.maximum(z0, 0)
    z1, c1, s1 = conv(a1, P["We1"], P["be1"], 2, 1)          # 7x7x16
    a2 = np.maximum(z1, 0)
    lat = a2.reshape(n, -1) @ P["Wz"] + P["bz"]
    zd = lat @ P["Wd"] + P["bd"]
    a3 = np.maximum(zd, 0).reshape(n, 7, 7, 16)
    u3 = up2(a3)
    z4, c4, s4 = conv(u3, P["Wd0"], P["bd0"], 1, 1)          # 14x14x8
    a4 = np.maximum(z4, 0)
    u4 = up2(a4)
    z5, c5, s5 = conv(u4, P["Wd1"], P["bd1"], 1, 1)          # 28x28x1
    out = _sig(z5)
    f.update(a0=a0, z0=z0, c0=c0, s0=s0, a1=a1, z1=z1, c1=c1, s1=s1, a2=a2, z=lat, zd=zd,
             a3=a3, u3=u3, z4=z4, c4=c4, s4=s4, a4=a4, u4=u4, z5=z5, c5=c5, s5=s5,
             recon=out.reshape(n, -1),
             acts=[a0, a1, a2, lat, a3, a4, out])
    return f


def backward(kind, P, f, x):
    n = len(x)
    g = {}
    r = f["recon"]
    d = 2.0 * (r - x.reshape(n, -1)) / r.size * r * (1 - r)   # through the sigmoid
    if kind == "dense":
        acts, pre = f["acts"], f["pre"]
        L = len(pre)
        for k in range(L - 1, -1, -1):
            g["W%d" % k] = acts[k].T @ d
            g["b%d" % k] = d.sum(0)
            if k:
                da = d @ P["W%d" % k].T
                d = da if k == len(DENSE) else da * (pre[k - 1] > 0)
        return g
    d5 = d.reshape(n, 28, 28, 1)
    g["Wd1"], g["bd1"], du4 = conv_back(d5, f["c5"], P["Wd1"], f["s5"], 1, 1)
    d4 = up2_back(du4) * (f["z4"] > 0)
    g["Wd0"], g["bd0"], du3 = conv_back(d4, f["c4"], P["Wd0"], f["s4"], 1, 1)
    dzd = up2_back(du3).reshape(n, -1) * (f["zd"] > 0)
    g["Wd"], g["bd"] = f["z"].T @ dzd, dzd.sum(0)
    dlat = dzd @ P["Wd"].T
    g["Wz"], g["bz"] = f["a2"].reshape(n, -1).T @ dlat, dlat.sum(0)
    da2 = (dlat @ P["Wz"].T).reshape(f["a2"].shape) * (f["z1"] > 0)
    g["We1"], g["be1"], da1 = conv_back(da2, f["c1"], P["We1"], f["s1"], 2, 1)
    da1 = da1 * (f["z0"] > 0)
    g["We0"], g["be0"], _ = conv_back(da1, f["c0"], P["We0"], f["s0"], 2, 1)
    return g


def loss(f, x):
    return float(np.mean((f["recon"] - x.reshape(len(x), -1)) ** 2))


EPOCHS = {"dense": 12, "conv": 4}


def train(kind="dense", latent=3, epochs=None, batch=128, lr=1e-3, seed=0,
          cache_dir=DEFAULT_CACHE):
    epochs = EPOCHS[kind] if epochs is None else epochs
    path = os.path.join(cache_dir, "ae_%s_l%d_e%d_s%d.npz" % (kind, latent, epochs, seed))
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files if k != "loss"}, float(d["loss"])
    xtr, _ytr, xte, _yte = load(cache_dir)
    rng = np.random.default_rng(seed)
    P = init(kind, latent, rng)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(a) for k, a in P.items()}
    step = 0
    for _ in range(epochs):
        order = rng.permutation(len(xtr))
        for s in range(0, len(order), batch):
            x = xtr[order[s:s + batch]]
            g = backward(kind, P, forward(kind, P, x), x)
            step += 1
            for k in P:
                m[k] = 0.9 * m[k] + 0.1 * g[k]
                v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
                P[k] = P[k] - lr * (m[k] / (1 - 0.9 ** step)) / (
                    np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
    test = float(np.mean([loss(forward(kind, P, xte[s:s + 1000]), xte[s:s + 1000])
                          for s in range(0, 10000, 1000)]))
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, loss=test, **P)
    return P, test


@dataclass
class AETrace:
    kind: str
    latent: int
    P: dict
    images: np.ndarray            # (k, 784) the digits evaluated
    labels: np.ndarray
    fwd: list                     # forward dict per image (batch removed)
    cloud: np.ndarray             # (m, latent) test codes for the scatter
    cloud_labels: np.ndarray
    test_loss: float

    def decode(self, z):
        """Decoder only: latent ``z`` -> (activations, 784 image)."""
        P, z = self.P, np.asarray(z, np.float64)[None]
        if self.kind == "dense":
            L = len([k for k in P if k.startswith("W")])
            a, acts = z, [z[0]]
            for k in range(len(DENSE), L):
                zk = a @ P["W%d" % k] + P["b%d" % k]
                a = _sig(zk) if k == L - 1 else np.maximum(zk, 0)
                acts.append(a[0])
            return acts, a[0]
        a3 = np.maximum(z @ P["Wd"] + P["bd"], 0).reshape(1, 7, 7, 16)
        a4 = np.maximum(conv(up2(a3), P["Wd0"], P["bd0"], 1, 1)[0], 0)
        out = _sig(conv(up2(a4), P["Wd1"], P["bd1"], 1, 1)[0])
        return [z[0], a3[0], a4[0], out[0]], out[0].reshape(-1)


def trace(kind="dense", latent=3, digits=(3, 2, 1), cloud=600) -> AETrace:
    """Evaluate test images ``digits`` (indices); also encode ``cloud`` test
    images for the latent-space scatter."""
    P, test = train(kind, latent)
    _a, _b, xte, yte = load()
    imgs = xte[list(digits)]
    fw = []
    for x in imgs:
        f = forward(kind, P, x[None])
        fw.append({k: (v[0] if isinstance(v, np.ndarray) and v.ndim and len(v) == 1 else v)
                   for k, v in f.items() if k not in ("acts", "pre")} |
                  {"acts": [a[0] for a in f["acts"]]})
    sel = np.arange(2000, 2000 + cloud)
    codes = np.concatenate([forward(kind, P, xte[s:s + 200])["z"]
                            for s in range(sel[0], sel[-1] + 1, 200)])[:cloud]
    return AETrace(kind, latent, P, imgs, yte[list(digits)], fw, codes, yte[sel], test)

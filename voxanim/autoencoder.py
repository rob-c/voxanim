"""Autoencoders for MNIST in plain numpy: a dense one and a convolutional one,
each with a 2-D or 3-D latent space (the bottleneck the animations draw).

Dense:  784 -> 256 -> 64 -> L -> 64 -> 256 -> 784        (ReLU, sigmoid output)
Conv:   28x28x1 -> conv 3x3 s2 -> 14x14x8 -> conv 3x3 s2 -> 7x7x16 -> dense -> L
        L -> dense -> 7x7x16 -> up x2, conv 3x3 -> 14x14x8 -> up x2, conv 3x3 -> 28x28x1

The latent layer is linear.  Loss: mean squared error between the input and
its reconstruction.  Training is seeded (Adam, :class:`voxanim.optim.Adam`)
and cached by :func:`voxanim.cache.cached_npz`.

The dense layers are numbered ``0 .. 5`` (parameters ``W0, b0 .. W5, b5``);
:func:`dense_layers` and :func:`dense_backward` run any consecutive range of
them, so :mod:`voxanim.deep_models` reuses the same encoder and decoder.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from . import accel
from .cache import DEFAULT_CACHE, cached_npz
from .minibatch import batched, chunk_mean, minibatches
from .mnist import load
from .optim import Adam
from .strided import _windows

DENSE = (784, 256, 64)                       # encoder widths before the latent
LATENT_LAYER = len(DENSE) - 1                # the dense layer producing the code: linear
N_DENSE = 2 * len(DENSE)                     # dense layers in all; the last is a sigmoid
ENCODER = range(0, len(DENSE))               # dense layer indices of the encoder ...
DECODER = range(len(DENSE), N_DENSE)         # ... and the decoder


def sigmoid(z):
    """Logistic sigmoid, with ``z`` clipped to [-30, 30] so ``exp`` cannot overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def dense_init(P, name, a, b, rng):
    """Add He-initialised ``W<name>`` ``(a, b)`` and a zero ``b<name>`` to ``P``."""
    P["W" + name] = rng.normal(0, np.sqrt(2.0 / a), (a, b))
    P["b" + name] = np.zeros(b)


# ------------------------------------------------------------------ convolution helpers


def conv(x, W, b, stride, pad):
    """Convolve ``(N, H, W, C)`` with zero ``pad`` and ``stride``.

    Returns ``(z, cols, padded shape)``: the output, the im2col windows and
    the padded input's shape (both needed by :func:`conv_back`).
    """
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad), (0, 0)))
    cols = _windows(xp, W.shape[0], stride)
    return cols @ W.reshape(-1, W.shape[-1]) + b, cols, xp.shape


def conv_back(dz, cols, W, xp_shape, stride, pad):
    """Gradients of :func:`conv`: ``(dW, db, dx)`` for the output gradient ``dz``."""
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
    """Nearest-neighbour upsampling x2 of ``(N, H, W, C)``."""
    return x.repeat(2, axis=1).repeat(2, axis=2)


def up2_back(d):
    """Gradient of :func:`up2`: each 2x2 block summed."""
    n, h, w, c = d.shape
    return d.reshape(n, h // 2, 2, w // 2, 2, c).sum((2, 4))


# ------------------------------------------------------------------ dense layers


def dense_layers(P, a, layers):
    """Run dense ``layers`` (consecutive indices) on ``a`` ``(N, width)``.

    ReLU everywhere except the latent layer (linear) and the last layer
    (sigmoid).  Returns ``(acts, pre)``: ``acts[i]`` is the input of
    ``layers[i]`` (``acts[-1]`` the output) and ``pre[i]`` its pre-activation.
    """
    acts, pre = [a], []
    for k in layers:
        zk = a @ P["W%d" % k] + P["b%d" % k]
        pre.append(zk)
        if k == N_DENSE - 1:
            a = sigmoid(zk)
        elif k == LATENT_LAYER:
            a = zk
        else:
            a = np.maximum(zk, 0)
        acts.append(a)
    return acts, pre


def dense_backward(P, acts, d, layers, g):
    """Backpropagate ``d`` (the gradient at the pre-activation of the last of
    ``layers``) through ``layers``, given their ``acts`` from :func:`dense_layers`.

    Writes the ``W`` / ``b`` gradients into ``g`` and returns the gradient at
    ``acts[0]`` (``None`` when the range starts at layer 0, the image).
    """
    for i in reversed(range(len(layers))):
        k = layers[i]
        g["W%d" % k] = acts[i].T @ d
        g["b%d" % k] = d.sum(0)
        if k == 0:
            return None
        d = d @ P["W%d" % k].T
        if k - 1 != LATENT_LAYER:                    # back through the ReLU
            d = d * (acts[i] > 0)
    return d


def recon_grad(r, x):
    """Gradient of ``mean((r - x)^2)`` at the pre-sigmoid of the output ``r``."""
    return 2.0 * (r - x) / r.size * r * (1 - r)


# ------------------------------------------------------------------ models


def init(kind, latent, rng):
    """Random parameters for a ``"dense"`` or ``"conv"`` autoencoder."""
    P = {}

    def convw(name, k, c, f):
        P["W" + name] = rng.normal(0, np.sqrt(2.0 / (k * k * c)), (k, k, c, f))
        P["b" + name] = np.zeros(f)

    if kind == "dense":
        widths = list(DENSE) + [latent] + list(DENSE[::-1])
        for n, (a, b) in enumerate(itertools.pairwise(widths)):
            dense_init(P, str(n), a, b, rng)
    else:
        convw("e0", 3, 1, 8)
        convw("e1", 3, 8, 16)
        dense_init(P, "z", 7 * 7 * 16, latent, rng)
        dense_init(P, "d", latent, 7 * 7 * 16, rng)
        convw("d0", 3, 16, 8)
        convw("d1", 3, 8, 1)
    return P


def forward(kind, P, x):
    """Encode and decode ``x`` ``(N, 784)``.

    Returns every intermediate; for both kinds ``z`` (the latent code,
    ``(N, L)``), ``recon`` ``(N, 784)`` and ``acts``, the seven tensors of the
    chain (input, two encoder layers, code, two decoder layers, output).
    Dense also has ``pre`` (pre-activations); conv the per-layer
    pre-activations ``z0 .. z5``, im2col windows ``c*`` and padded shapes ``s*``.
    """
    n = len(x)
    if kind == "dense":
        enc, pre_e = dense_layers(P, x.reshape(n, -1), ENCODER)
        dec, pre_d = dense_layers(P, enc[-1], DECODER)
        return dict(acts=enc + dec[1:], pre=pre_e + pre_d, z=enc[-1], recon=dec[-1])
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
    out = sigmoid(z5)
    return dict(a0=a0, z0=z0, c0=c0, s0=s0, a1=a1, z1=z1, c1=c1, s1=s1, a2=a2, z=lat, zd=zd,
                a3=a3, u3=u3, z4=z4, c4=c4, s4=s4, a4=a4, u4=u4, z5=z5, c5=c5, s5=s5,
                recon=out.reshape(n, -1), acts=[a0, a1, a2, lat, a3, a4, out])


def backward(kind, P, f, x):
    """Gradients of :func:`loss` with respect to every parameter."""
    n = len(x)
    g = {}
    d = recon_grad(f["recon"], x.reshape(n, -1))
    if kind == "dense":
        dense_backward(P, f["acts"], d, range(N_DENSE), g)
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


# ------------------------------------------------------------------ the same model in PyTorch
#
# For training on a GPU (voxanim.accel).  The tensors are channels-first, as
# torch's convolutions want; the weights keep their numpy shapes and are
# permuted on the way in, and flattened activations are reordered to the
# numpy (height, width, channel) layout, so the trained weights come back as
# exactly the arrays the numpy model reads.


def torch_forward(kind, Pt, x):
    """``(z, recon)`` of :func:`forward`, for tensor parameters ``Pt`` and a
    ``(N, 784)`` tensor ``x``."""
    from .accel import torch
    t = torch()
    F = t.nn.functional
    n = x.shape[0]
    if kind == "dense":
        a = x
        for k in range(N_DENSE):
            zk = a @ Pt["W%d" % k] + Pt["b%d" % k]
            if k == N_DENSE - 1:
                a = t.sigmoid(zk)
            elif k == LATENT_LAYER:
                a = zk
                z = zk
            else:
                a = t.relu(zk)
        return z, a

    def conv_t(h, name, stride):
        W = Pt["W" + name].permute(3, 2, 0, 1)          # (k, k, c, f) -> (f, c, k, k)
        return F.conv2d(h, W, Pt["b" + name], stride=stride, padding=1)

    def up(h):
        return h.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

    a1 = t.relu(conv_t(x.reshape(n, 1, 28, 28), "e0", 2))
    a2 = t.relu(conv_t(a1, "e1", 2))
    lat = a2.permute(0, 2, 3, 1).reshape(n, -1) @ Pt["Wz"] + Pt["bz"]
    a3 = t.relu(lat @ Pt["Wd"] + Pt["bd"]).reshape(n, 7, 7, 16).permute(0, 3, 1, 2)
    a4 = t.relu(conv_t(up(a3), "d0", 1))
    out = t.sigmoid(conv_t(up(a4), "d1", 1))
    return lat, out.reshape(n, -1)


def torch_loss(kind, Pt, x):
    """:func:`loss` of the torch forward pass (mean squared error)."""
    _z, r = torch_forward(kind, Pt, x)
    return ((r - x) ** 2).mean()


def loss(f, x):
    """Mean squared reconstruction error of the forward pass ``f`` of ``x``."""
    return float(np.mean((f["recon"] - x.reshape(len(x), -1)) ** 2))


EPOCHS = {"dense": 12, "conv": 4}


def train_settings(kind="dense", latent=3, epochs=None, batch=128, lr=1e-3, seed=0):
    """Every setting that determines :func:`train`'s weights (its cache key)."""
    epochs = EPOCHS[kind] if epochs is None else epochs
    return dict(kind=kind, latent=latent, dense=DENSE, epochs=epochs, batch=batch, lr=lr,
                seed=seed)


def train(kind="dense", latent=3, epochs=None, batch=128, lr=1e-3, seed=0,
          cache_dir=DEFAULT_CACHE):
    """Train with Adam (or load the cache): ``(params, mean test reconstruction error)``."""
    settings = train_settings(kind, latent, epochs, batch, lr, seed)

    def build():
        xtr, _ytr, xte, _yte = load(cache_dir)
        rng = np.random.default_rng(seed)
        P = init(kind, latent, rng)
        batches = minibatches(rng, len(xtr), settings["epochs"], batch)
        dev = accel.device(prefer=(kind == "conv"))      # measured: conv 9x, dense ~1x
        if dev is not None:
            accel.fit(P, lambda Pt, xb, _i: torch_loss(kind, Pt, xb), batches, lr, data=xtr,
                      dev=dev)
        else:
            opt = Adam(P, lr)
            for idx in batches:
                x = xtr[idx]
                opt.step(P, backward(kind, P, forward(kind, P, x), x))
        test = chunk_mean(lambda s: loss(forward(kind, P, xte[s]), xte[s]), len(xte))
        return dict(P, loss=np.array(test))

    d = cached_npz("ae_%s" % kind, settings, build, cache_dir)
    return {k: v for k, v in d.items() if k != "loss"}, float(d["loss"])


@dataclass
class AETrace:
    """Digits through a trained autoencoder, plus a cloud of test codes."""

    kind: str
    latent: int
    P: dict
    images: np.ndarray            # (k, 784) the digits evaluated
    labels: np.ndarray            # (k,)
    fwd: list                     # forward dict per image (batch removed)
    cloud: np.ndarray             # (m, latent) test codes for the scatter
    cloud_labels: np.ndarray      # (m,)

    def decode(self, z):
        """Decoder only: latent ``z`` ``(L,)`` -> (the decoder's tensors, 784 image).

        The tensors are the code and the decoder's layers, batch removed (for
        conv in HWC shape).
        """
        P, z = self.P, np.asarray(z, np.float64)[None]
        if self.kind == "dense":
            acts, _pre = dense_layers(P, z, DECODER)
            return [a[0] for a in acts], acts[-1][0]
        a3 = np.maximum(z @ P["Wd"] + P["bd"], 0).reshape(1, 7, 7, 16)
        a4 = np.maximum(conv(up2(a3), P["Wd0"], P["bd0"], 1, 1)[0], 0)
        out = sigmoid(conv(up2(a4), P["Wd1"], P["bd1"], 1, 1)[0])
        return [z[0], a3[0], a4[0], out[0]], out[0].reshape(-1)


CLOUD_START = 2000                # the scatter's test digits: CLOUD_START, CLOUD_START + 1, ...
SNAPSHOT_RATIO = 1.08             # a run's snapshot steps grow by 8% a time (equal time per decade)
EVAL_TEST = 2000                  # test images a snapshot's loss is measured on


def snapshot_steps(total, ratio=SNAPSHOT_RATIO):
    """The steps a run is snapshotted at: 0 (untrained), 1, 2, ... then
    thinning out geometrically (each about ``ratio`` times the last) to ``total``."""
    steps, s = [0], 0
    while s < total:
        s = min(total, max(s + 1, int(s * ratio)))
        steps.append(s)
    return np.array(steps)


@dataclass
class AERun:
    """Snapshots through one training run of :func:`train` (:func:`training_run`).

    Arrays with a leading ``S`` axis have one entry per snapshot.
    """

    kind: str
    latent: int
    steps: np.ndarray             # (S,) the training step of each snapshot; 0 is untrained
    steps_per_epoch: int
    losses: np.ndarray            # (S,) mean reconstruction error on EVAL_TEST test images
    cloud: np.ndarray             # (S, m, latent) codes of the cloud digits (as AETrace.cloud)
    cloud_labels: np.ndarray      # (m,)
    images: np.ndarray            # (k, 784) the digits followed through the chain
    labels: np.ndarray            # (k,)
    acts: list                    # per chain tensor: (S, k, values) float32, flattened
    P0: dict                      # the untrained parameters
    P: dict                       # the trained ones: the same bytes as train()'s


def training_run(kind="dense", latent=2, digits=(3, 2, 1), cloud=600,
                 cache_dir=DEFAULT_CACHE) -> AERun:
    """:func:`train` again, keeping snapshots on the way (cached).

    At every step of :func:`snapshot_steps` it records the codes of the
    ``cloud`` test digits (from :data:`CLOUD_START`), the loss on the first
    :data:`EVAL_TEST` test images and the chain activations of test images
    ``digits``.  The random draws are exactly :func:`train`'s, so the final
    weights are identical to its cache.
    """
    settings = dict(train_settings(kind, latent), snapshot_ratio=SNAPSHOT_RATIO,
                    digits=list(digits), cloud=cloud, cloud_start=CLOUD_START,
                    eval_test=EVAL_TEST)
    seed, batch, lr, epochs = (settings[k] for k in ("seed", "batch", "lr", "epochs"))
    xtr, _ytr, xte, yte = load(cache_dir)
    sel = np.arange(CLOUD_START, CLOUD_START + cloud)
    imgs = xte[list(digits)]

    def build():
        rng = np.random.default_rng(seed)
        P = init(kind, latent, rng)
        opt = Adam(P, lr)
        per_epoch = -(-len(xtr) // batch)
        marks = snapshot_steps(epochs * per_epoch)
        rec = dict(losses=[], cloud=[], acts=[])

        def snapshot():
            rec["cloud"].append(forward(kind, P, xte[sel])["z"])
            rec["losses"].append(chunk_mean(lambda s: loss(forward(kind, P, xte[s]), xte[s]),
                                            EVAL_TEST))
            rec["acts"].append([a.reshape(len(imgs), -1) for a in forward(kind, P, imgs)["acts"]])

        snapshot()
        want = set(marks.tolist())
        batches = minibatches(rng, len(xtr), epochs, batch)
        dev = accel.device(prefer=(kind == "conv"))
        if dev is not None:
            def after(step, Pt):
                if step in want:
                    P.update(accel.to_numpy(Pt))
                    snapshot()
            accel.fit(P, lambda Pt, xb, _i: torch_loss(kind, Pt, xb), batches, lr, after, xtr,
                      dev=dev)
        else:
            for step, idx in enumerate(batches, 1):
                x = xtr[idx]
                opt.step(P, backward(kind, P, forward(kind, P, x), x))
                if step in want:
                    snapshot()
        out = dict(P, steps=marks, steps_per_epoch=np.array(per_epoch),
                   losses=np.array(rec["losses"]), cloud=np.array(rec["cloud"]))
        for j in range(len(rec["acts"][0])):
            out["acts%d" % j] = np.array([a[j] for a in rec["acts"]], np.float32)
        return out

    d = cached_npz("ae_%s_run" % kind, settings, build, cache_dir)
    meta = ("steps", "steps_per_epoch", "losses", "cloud")
    acts = [d["acts%d" % j] for j in range(7)]
    P = {k: v for k, v in d.items() if k not in meta and not k.startswith("acts")}
    return AERun(kind, latent, d["steps"], int(d["steps_per_epoch"]), d["losses"], d["cloud"],
                 yte[sel], imgs, yte[list(digits)], acts,
                 init(kind, latent, np.random.default_rng(seed)), P)


def _unbatch(f):
    """A single-image forward dict without its batch dimension."""
    out = {k: (v[0] if isinstance(v, np.ndarray) and v.ndim and len(v) == 1 else v)
           for k, v in f.items() if k not in ("acts", "pre")}
    return out | {"acts": [a[0] for a in f["acts"]]}


def trace(kind="dense", latent=3, digits=(3, 2, 1), cloud=600,
          cache_dir=DEFAULT_CACHE) -> AETrace:
    """Evaluate test images ``digits`` (indices); also encode ``cloud`` test
    images (from :data:`CLOUD_START`) for the latent-space scatter."""
    P, _test = train(kind, latent, cache_dir=cache_dir)
    _xtr, _ytr, xte, yte = load(cache_dir)
    imgs = xte[list(digits)]
    fw = [_unbatch(forward(kind, P, x[None])) for x in imgs]
    sel = np.arange(CLOUD_START, CLOUD_START + cloud)
    codes = batched(lambda xs: forward(kind, P, xs)["z"], xte[sel], 200)
    return AETrace(kind, latent, P, imgs, yte[list(digits)], fw, codes, yte[sel])

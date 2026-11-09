"""A tiny vision transformer for MNIST, in plain numpy.

    image 28x28 -> 16 patches of 7x7 -> tokens: [CLS; patch @ E] + position   (17 x 24)
    -> multi-head self-attention, 3 heads of width 8:
           Q = T Wq,  K = T Wk,  V = T Wv
           A_h = softmax(Q_h K_h^T / sqrt(8))          (17 x 17 per head)
           O_h = A_h V_h,   O = [O_1 O_2 O_3]
    -> Y = T + O Wo                                     (residual)
    -> CLS row: ReLU(Y_0 W1 + b1) W2 + b2 -> softmax -> cross-entropy vs one-hot

Training is seeded (Adam, :class:`voxanim.optim.Adam`) and cached by
:func:`voxanim.cache.cached_npz`.

Shapes (``N`` images): tokens ``T`` ``(N, N_TOK, D)``; per-head ``Qh``, ``Kh``,
``Vh``, ``Oh`` ``(N, HEADS, N_TOK, DH)``; scores ``S`` and weights ``A``
``(N, HEADS, N_TOK, N_TOK)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import DEFAULT_CACHE, cached_npz
from .minibatch import chunk_mean, minibatches
from .mnist import load
from .optim import Adam

D, HEADS, PATCH, HIDDEN = 24, 3, 7, 32    # model width, heads, patch side, MLP width
DH = D // HEADS                           # width of one head
GRID = 28 // PATCH                        # patches per side
N_TOK = GRID ** 2 + 1                     # 16 patches + CLS
NAMES = ("E", "bE", "cls", "pos", "Wq", "Wk", "Wv", "Wo", "W1", "b1", "W2", "b2")


def patches(x):
    """``(N, 784)`` -> ``(N, 16, 49)``, patches in row-major order."""
    x = x.reshape(-1, GRID, PATCH, GRID, PATCH).transpose(0, 1, 3, 2, 4)
    return x.reshape(-1, GRID * GRID, PATCH * PATCH)


def init(rng):
    """Random parameters (keys :data:`NAMES`); He-style scale on ``W1``."""
    s = lambda a, *shape: rng.normal(0, np.sqrt(1.0 / a), shape)   # noqa: E731
    return dict(E=s(49, 49, D), bE=np.zeros(D), cls=s(D, D), pos=0.1 * s(1, N_TOK, D),
                Wq=s(D, D, D), Wk=s(D, D, D), Wv=s(D, D, D), Wo=s(D, D, D),
                W1=s(D, D, HIDDEN) * np.sqrt(2), b1=np.zeros(HIDDEN),
                W2=s(HIDDEN, HIDDEN, 10), b2=np.zeros(10))


def _heads(m):
    """``(N, T, D)`` -> ``(N, H, T, DH)``."""
    return m.reshape(m.shape[0], m.shape[1], HEADS, DH).transpose(0, 2, 1, 3)


def _merge(m):
    """``(N, H, T, DH)`` -> ``(N, T, D)``, the inverse of :func:`_heads`."""
    return m.transpose(0, 2, 1, 3).reshape(m.shape[0], m.shape[2], D)


def forward(P, x):
    """Classify ``x`` ``(N, 784)``; every intermediate, by the names in the
    module docstring, plus ``z`` (the CLS row of ``Y``), ``pre1``, ``u``
    (MLP), ``logits`` and ``p`` (softmax) ``(N, 10)``."""
    n = len(x)
    pt = patches(x)
    T = np.concatenate([np.broadcast_to(P["cls"], (n, 1, D)), pt @ P["E"] + P["bE"]], 1)
    T = T + P["pos"]
    Q, K, V = T @ P["Wq"], T @ P["Wk"], T @ P["Wv"]
    Qh, Kh, Vh = _heads(Q), _heads(K), _heads(V)
    S = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(DH)
    A = np.exp(S - S.max(-1, keepdims=True))
    A /= A.sum(-1, keepdims=True)
    Oh = A @ Vh
    O = _merge(Oh)
    Y = T + O @ P["Wo"]
    z = Y[:, 0]
    pre1 = z @ P["W1"] + P["b1"]
    u = np.maximum(pre1, 0)
    logits = u @ P["W2"] + P["b2"]
    p = np.exp(logits - logits.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return dict(pt=pt, T=T, Q=Q, K=K, V=V, Qh=Qh, Kh=Kh, Vh=Vh, S=S, A=A, Oh=Oh, O=O,
                Y=Y, z=z, pre1=pre1, u=u, logits=logits, p=p)


def backward(P, f, y_onehot):
    """Cross-entropy gradients for the forward pass ``f``.

    Returns ``(g, acts)``: parameter gradients (keys :data:`NAMES`) and the
    activation gradients ``dlog``, ``dY``, ``dO``, ``dA``, ``dS``, ``dQ``,
    ``dK``, ``dV``, ``dT`` (shaped like their forward counterparts).
    """
    n = len(f["p"])
    g = {}
    dlog = (f["p"] - y_onehot) / n
    g["W2"], g["b2"] = f["u"].T @ dlog, dlog.sum(0)
    dpre1 = (dlog @ P["W2"].T) * (f["pre1"] > 0)
    g["W1"], g["b1"] = f["z"].T @ dpre1, dpre1.sum(0)
    dY = np.zeros_like(f["Y"])
    dY[:, 0] = dpre1 @ P["W1"].T                      # only the CLS row is read
    g["Wo"] = np.einsum("ntd,nte->de", f["O"], dY)
    dO = dY @ P["Wo"].T
    dOh = _heads(dO)
    dA = dOh @ f["Vh"].transpose(0, 1, 3, 2)
    dVh = f["A"].transpose(0, 1, 3, 2) @ dOh
    dS = f["A"] * (dA - (dA * f["A"]).sum(-1, keepdims=True))
    dQh = dS @ f["Kh"] / np.sqrt(DH)
    dKh = dS.transpose(0, 1, 3, 2) @ f["Qh"] / np.sqrt(DH)
    dQ, dK, dV = _merge(dQh), _merge(dKh), _merge(dVh)
    T = f["T"]
    g["Wq"] = np.einsum("ntd,nte->de", T, dQ)
    g["Wk"] = np.einsum("ntd,nte->de", T, dK)
    g["Wv"] = np.einsum("ntd,nte->de", T, dV)
    dT = dY + dQ @ P["Wq"].T + dK @ P["Wk"].T + dV @ P["Wv"].T
    g["pos"] = dT.sum(0)
    g["cls"] = dT[:, 0].sum(0)
    g["E"] = np.einsum("npk,npd->kd", f["pt"], dT[:, 1:])
    g["bE"] = dT[:, 1:].sum((0, 1))
    acts = dict(dlog=dlog, dY=dY, dO=dO, dA=dA, dS=dS, dQ=dQ, dK=dK, dV=dV, dT=dT)
    return g, acts


def train(epochs=10, batch=128, lr=2e-3, seed=0, cache_dir=DEFAULT_CACHE):
    """Train with Adam (or load the cached weights): ``(params, test accuracy)``."""
    settings = dict(D=D, heads=HEADS, patch=PATCH, hidden=HIDDEN, epochs=epochs, batch=batch,
                    lr=lr, seed=seed)

    def build():
        xtr, ytr, xte, yte = load(cache_dir)
        rng = np.random.default_rng(seed)
        P = init(rng)
        opt = Adam(P, lr)
        eye = np.eye(10)
        for idx in minibatches(rng, len(xtr), epochs, batch):
            g, _ = backward(P, forward(P, xtr[idx]), eye[ytr[idx]])
            opt.step(P, g)
        acc = chunk_mean(lambda s: (forward(P, xte[s])["p"].argmax(1) == yte[s]).mean(),
                         len(xte))
        return dict(P, acc=np.array(acc))

    d = cached_npz("vit", settings, build, cache_dir)
    return {k: d[k] for k in NAMES}, float(d["acc"])


@dataclass
class VitTrace:
    """One image through the ViT: forward, loss and gradients."""

    x: np.ndarray            # (784,) the image
    label: int
    P: dict
    f: dict                  # forward intermediates, batch dimension removed
    g: dict                  # parameter gradients
    d: dict                  # activation gradients, batch dimension removed
    p: np.ndarray            # (10,) softmax probabilities
    loss: float              # cross-entropy -log p[label]
    accuracy: float          # the model's test accuracy

    @property
    def prediction(self):
        """The most probable digit."""
        return int(self.p.argmax())


def _trace(P, x, label, acc):
    f = forward(P, x[None])
    g, d = backward(P, f, np.eye(10)[[label]])
    f1 = {k: v[0] for k, v in f.items()}
    d1 = {k: v[0] for k, v in d.items()}
    return VitTrace(x=x, label=label, P=P, f=f1, g=g, d=d1, p=f1["p"],
                    loss=float(-np.log(f1["p"][label] + 1e-12)), accuracy=acc)


def trace(index, cache_dir=DEFAULT_CACHE, **kw) -> VitTrace:
    """Test image ``index`` through the trained model (``kw`` go to :func:`train`)."""
    P, acc = train(cache_dir=cache_dir, **kw)
    _xtr, _ytr, xte, yte = load(cache_dir)
    return _trace(P, xte[index], int(yte[index]), acc)


def sgd_step(tr: VitTrace, lr) -> VitTrace:
    """``tr`` again after one plain SGD step ``P - lr * g`` on its own gradient."""
    P = {k: tr.P[k] - lr * tr.g[k] for k in NAMES}
    return _trace(P, tr.x, tr.label, tr.accuracy)

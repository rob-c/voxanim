"""Patch embeddings, the first layer of a vision transformer, on MNIST.

A 28 × 28 digit is cut into a 4 × 4 grid of 7 × 7 patches (:func:`patches`);
each patch's 49 pixels, flattened, are multiplied by one shared matrix ``E``
(49 × :data:`DIM`) -- the patch embedding, a 7 × 7 convolution with stride 7
-- and, optionally, a learned position vector ``pos[k]`` is added for the
patch's place ``k`` in the grid.  The 16 vectors are then what a ViT's
attention reads; here, to keep the study about the embedding, they go
straight to a classifier:

    h_k = ReLU(x_k E + b + pos[k]),   logits = mean_k(h_k) W + c

Mean pooling forgets the order of the patches, so without ``pos`` the model
sees a bag of patches: shuffle a digit's patches and its prediction does not
change at all (:func:`shuffled_accuracy`).  With ``pos`` it knows where each
patch was.  Both are trained the same way (:func:`train`: Adam, seeded,
cached).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import cached_npz
from .optim import Adam

SIDE, P = 4, 7                 # a 4 × 4 grid of 7 × 7 patches
N_PATCH = SIDE * SIDE
DIM = 64                       # embedding width
EPOCHS = 12
BATCH = 128
LR = 2e-3
SEED = 0
N_SNAPS = 41                   # snapshots of E (and pos) through training
SHUFFLE_SEED = 5               # the fixed shuffle of the patch grid


def patches(x):
    """``(N, 784)`` digits to ``(N, 16, 49)``: the patches, row-major."""
    return (x.reshape(-1, SIDE, P, SIDE, P).transpose(0, 1, 3, 2, 4)
            .reshape(-1, N_PATCH, P * P))


def unpatch(xp):
    """The inverse of :func:`patches`."""
    return (xp.reshape(-1, SIDE, SIDE, P, P).transpose(0, 1, 3, 2, 4).reshape(-1, 784))


def init(rng, positions):
    """He-normal ``E`` and ``W``, zero biases, small random positions."""
    Pm = dict(E=rng.normal(0, np.sqrt(2.0 / (P * P)), (P * P, DIM)), b=np.zeros(DIM),
              W=rng.normal(0, np.sqrt(2.0 / DIM), (DIM, 10)), c=np.zeros(10))
    if positions:
        Pm["pos"] = rng.normal(0, 0.02, (N_PATCH, DIM))
    return Pm


def forward(Pm, xp):
    """Logits ``(N, 10)`` and what the backward pass needs."""
    a = xp @ Pm["E"] + Pm["b"]
    if "pos" in Pm:
        a = a + Pm["pos"]
    h = np.maximum(a, 0.0)
    m = h.mean(1)
    return m @ Pm["W"] + Pm["c"], (a, h, m)


def loss_grads(Pm, xp, y):
    """Mean cross-entropy and its gradients."""
    z, (a, _h, m) = forward(Pm, xp)
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(1, keepdims=True)
    n = len(y)
    loss = float(-np.log(p[np.arange(n), y]).mean())
    dz = p.copy()
    dz[np.arange(n), y] -= 1.0
    dz /= n
    G = dict(W=m.T @ dz, c=dz.sum(0))
    dm = dz @ Pm["W"].T
    da = (dm[:, None, :] / N_PATCH) * (a > 0)
    G["E"] = np.einsum("nkp,nkd->pd", xp, da)
    G["b"] = da.sum((0, 1))
    if "pos" in Pm:
        G["pos"] = da.sum(0)
    return loss, G


def predict(Pm, xp, chunk=2000):
    """Predicted classes, in chunks."""
    return np.concatenate([forward(Pm, xp[s:s + chunk])[0].argmax(1)
                           for s in range(0, len(xp), chunk)])


def shuffle_order():
    """The fixed permutation of the 16 patches used to scramble digits."""
    return np.random.default_rng(SHUFFLE_SEED).permutation(N_PATCH)


@dataclass
class Run:
    """One trained patch-embedding classifier."""

    positions: bool
    P: dict
    E_snaps: np.ndarray            # (N_SNAPS, 49, DIM)
    pos_snaps: np.ndarray          # (N_SNAPS, 16, DIM) (zeros without positions)
    snap_steps: np.ndarray
    train_loss: np.ndarray         # (steps,)
    test_acc: float
    shuffled_acc: float            # test accuracy with every digit's patches shuffled


def _fit(positions, cache_dir):
    from .mnist import load
    xtr, ytr, xte, yte = load(cache_dir)
    ptr, pte = patches(xtr), patches(xte)
    rng = np.random.default_rng(SEED)
    Pm = init(rng, positions)
    opt = Adam(Pm, LR)
    steps = EPOCHS * (len(ptr) // BATCH)
    snap_steps = np.unique(np.round(np.linspace(0, steps, N_SNAPS)).astype(np.int64))
    E_s, pos_s = [Pm["E"].copy()], [Pm.get("pos", np.zeros((N_PATCH, DIM))).copy()]
    losses, k = np.empty(steps), 0
    for _ in range(EPOCHS):
        order = rng.permutation(len(ptr))
        for b in range(len(ptr) // BATCH):
            idx = order[b * BATCH:(b + 1) * BATCH]
            loss, G = loss_grads(Pm, ptr[idx], ytr[idx])
            opt.step(Pm, G)
            losses[k] = loss
            k += 1
            if k in snap_steps:
                E_s.append(Pm["E"].copy())
                pos_s.append(Pm.get("pos", np.zeros((N_PATCH, DIM))).copy())
    acc = float((predict(Pm, pte) == yte).mean())
    sh = float((predict(Pm, pte[:, shuffle_order()]) == yte).mean())
    out = dict(E_snaps=np.array(E_s), pos_snaps=np.array(pos_s), snap_steps=snap_steps,
               train_loss=losses, test_acc=np.array(acc), shuffled_acc=np.array(sh))
    out.update({"P_" + k: v for k, v in Pm.items()})
    return out


def train(positions, cache_dir=None):
    """The classifier with (``positions``) or without learned position vectors."""
    settings = dict(side=SIDE, p=P, dim=DIM, epochs=EPOCHS, batch=BATCH, lr=LR, seed=SEED,
                    snaps=N_SNAPS, shuffle_seed=SHUFFLE_SEED, positions=bool(positions), v=1)
    d = cached_npz("patch-embed", settings, lambda: _fit(positions, cache_dir), cache_dir)
    Pm = {k[2:]: v for k, v in d.items() if k.startswith("P_")}
    return Run(bool(positions), Pm, d["E_snaps"], d["pos_snaps"], d["snap_steps"],
               d["train_loss"], float(d["test_acc"]), float(d["shuffled_acc"]))


def position_similarity(pos):
    """``(16, 16)`` cosine similarity of the learned position vectors."""
    U = pos / np.maximum(np.linalg.norm(pos, axis=1, keepdims=True), 1e-12)
    return U @ U.T

"""A dense VAE, a dense U-Net (denoising) and DEC / IDEC clustering, in plain
numpy, on MNIST.  Each is seeded, trained with Adam and cached in
``build/.cache`` like the other models.

VAE      784 -> 256 -> 64 -> (mu, log var) in L dims -> sample z -> 64 -> 256 -> 784
         loss = binary cross-entropy (summed over pixels) + KL(N(mu, var) || N(0, I))

U-Net    noisy 784 -> e1 256 -> e2 64 -> b 16 -> d2 64 [b ++ e2] -> d1 256 [d2 ++ e1] -> 784
         ("++" = concatenation: the skip connections); trained to output the clean
         digit (MSE).  A copy without the skips is trained for comparison.

DEC      the dense 3-D autoencoder's encoder (784-256-64-3) + 10 cluster centres
         initialised by k-means; soft assignment q_ij = Student-t kernel,
         target p_ij = q_ij^2 / f_j (renormalised), loss = KL(P || Q)
IDEC     DEC + the decoder: loss = reconstruction + gamma * KL(P || Q)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from . import autoencoder as AE
from .mnist import load
from .scene import DEFAULT_CACHE

CACHE = DEFAULT_CACHE


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _adam(P, g, st, lr, step):
    for k in P:
        st["m"][k] = 0.9 * st["m"][k] + 0.1 * g[k]
        st["v"][k] = 0.999 * st["v"][k] + 0.001 * g[k] ** 2
        P[k] = P[k] - lr * (st["m"][k] / (1 - 0.9 ** step)) / (
            np.sqrt(st["v"][k] / (1 - 0.999 ** step)) + 1e-8)


def _adam_state(P):
    return {"m": {k: np.zeros_like(v) for k, v in P.items()},
            "v": {k: np.zeros_like(v) for k, v in P.items()}}


def _dense(P, name, a, b, rng):
    P["W" + name] = rng.normal(0, np.sqrt(2.0 / a), (a, b))
    P["b" + name] = np.zeros(b)


# ================================================================== VAE


def vae_init(L, rng):
    P = {}
    for name, a, b in (("e0", 784, 256), ("e1", 256, 64), ("mu", 64, L), ("lv", 64, L),
                       ("d0", L, 64), ("d1", 64, 256), ("d2", 256, 784)):
        _dense(P, name, a, b, rng)
    P["Wlv"] *= 0.1
    return P


def vae_forward(P, x, eps):
    h0 = np.maximum(x @ P["We0"] + P["be0"], 0)
    h1 = np.maximum(h0 @ P["We1"] + P["be1"], 0)
    mu = h1 @ P["Wmu"] + P["bmu"]
    lv = h1 @ P["Wlv"] + P["blv"]
    z = mu + np.exp(0.5 * lv) * eps
    g0 = np.maximum(z @ P["Wd0"] + P["bd0"], 0)
    g1 = np.maximum(g0 @ P["Wd1"] + P["bd1"], 0)
    logits = g1 @ P["Wd2"] + P["bd2"]
    return dict(x=x, eps=eps, h0=h0, h1=h1, mu=mu, lv=lv, z=z, g0=g0, g1=g1, logits=logits,
                recon=_sig(logits))


def vae_loss(f):
    x, r = f["x"], np.clip(f["recon"], 1e-7, 1 - 1e-7)
    bce = -(x * np.log(r) + (1 - x) * np.log(1 - r)).sum(1)
    kl = -0.5 * (1 + f["lv"] - f["mu"] ** 2 - np.exp(f["lv"])).sum(1)
    return float(np.mean(bce + kl)), float(np.mean(bce)), float(np.mean(kl))


def vae_backward(P, f):
    n = len(f["x"])
    g = {}
    d = (f["recon"] - f["x"]) / n                       # BCE through the sigmoid
    g["Wd2"], g["bd2"] = f["g1"].T @ d, d.sum(0)
    d = (d @ P["Wd2"].T) * (f["g1"] > 0)
    g["Wd1"], g["bd1"] = f["g0"].T @ d, d.sum(0)
    d = (d @ P["Wd1"].T) * (f["g0"] > 0)
    g["Wd0"], g["bd0"] = f["z"].T @ d, d.sum(0)
    dz = d @ P["Wd0"].T
    s = np.exp(0.5 * f["lv"])
    dmu = dz + f["mu"] / n                              # + KL
    dlv = dz * 0.5 * s * f["eps"] + 0.5 * (np.exp(f["lv"]) - 1) / n
    g["Wmu"], g["bmu"] = f["h1"].T @ dmu, dmu.sum(0)
    g["Wlv"], g["blv"] = f["h1"].T @ dlv, dlv.sum(0)
    d = (dmu @ P["Wmu"].T + dlv @ P["Wlv"].T) * (f["h1"] > 0)
    g["We1"], g["be1"] = f["h0"].T @ d, d.sum(0)
    d = (d @ P["We1"].T) * (f["h0"] > 0)
    g["We0"], g["be0"] = f["x"].T @ d, d.sum(0)
    return g


def vae_decode(P, z):
    g0 = np.maximum(np.atleast_2d(z) @ P["Wd0"] + P["bd0"], 0)
    g1 = np.maximum(g0 @ P["Wd1"] + P["bd1"], 0)
    return g0, g1, _sig(g1 @ P["Wd2"] + P["bd2"])


def vae_train(L=3, epochs=15, batch=128, lr=1e-3, seed=0):
    path = os.path.join(CACHE, "vae_l%d_e%d_s%d.npz" % (L, epochs, seed))
    if os.path.exists(path):
        with np.load(path) as d:
            return {k: d[k] for k in d.files if k != "loss"}, float(d["loss"])
    xtr, _y, xte, _ = load(CACHE)
    rng = np.random.default_rng(seed)
    P = vae_init(L, rng)
    st, step = _adam_state(P), 0
    for _ in range(epochs):
        order = rng.permutation(len(xtr))
        for s in range(0, len(order), batch):
            x = xtr[order[s:s + batch]]
            f = vae_forward(P, x, rng.normal(size=(len(x), L)))
            step += 1
            _adam(P, vae_backward(P, f), st, lr, step)
    te = vae_loss(vae_forward(P, xte[:2000], np.zeros((2000, L))))[0]
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, loss=te, **P)
    return P, te


# ================================================================== U-Net (denoising)

NOISE = 0.45                       # std of the Gaussian noise added to the digits


def noisy(x, rng):
    return np.clip(x + rng.normal(0, NOISE, x.shape), 0, 1)


def unet_init(rng, skips=True):
    P = {}
    _dense(P, "e1", 784, 256, rng)
    _dense(P, "e2", 256, 64, rng)
    _dense(P, "b", 64, 16, rng)
    _dense(P, "d2", 16 + (64 if skips else 0), 64, rng)
    _dense(P, "d1", 64 + (256 if skips else 0), 256, rng)
    _dense(P, "o", 256, 784, rng)
    return P


def unet_forward(P, x, skips=True):
    e1 = np.maximum(x @ P["We1"] + P["be1"], 0)
    e2 = np.maximum(e1 @ P["We2"] + P["be2"], 0)
    b = np.maximum(e2 @ P["Wb"] + P["bb"], 0)
    c2 = np.concatenate([b, e2], 1) if skips else b
    d2 = np.maximum(c2 @ P["Wd2"] + P["bd2"], 0)
    c1 = np.concatenate([d2, e1], 1) if skips else d2
    d1 = np.maximum(c1 @ P["Wd1"] + P["bd1"], 0)
    out = _sig(d1 @ P["Wo"] + P["bo"])
    return dict(x=x, e1=e1, e2=e2, b=b, c2=c2, d2=d2, c1=c1, d1=d1, out=out)


def unet_backward(P, f, clean, skips=True):
    g = {}
    r = f["out"]
    d = 2 * (r - clean) / r.size * r * (1 - r)
    g["Wo"], g["bo"] = f["d1"].T @ d, d.sum(0)
    d = (d @ P["Wo"].T) * (f["d1"] > 0)
    g["Wd1"], g["bd1"] = f["c1"].T @ d, d.sum(0)
    dc1 = d @ P["Wd1"].T
    dd2 = dc1[:, :64] * (f["d2"] > 0)
    de1_skip = dc1[:, 64:] if skips else 0
    g["Wd2"], g["bd2"] = f["c2"].T @ dd2, dd2.sum(0)
    dc2 = dd2 @ P["Wd2"].T
    db = dc2[:, :16] * (f["b"] > 0)
    de2_skip = dc2[:, 16:] if skips else 0
    g["Wb"], g["bb"] = f["e2"].T @ db, db.sum(0)
    de2 = (db @ P["Wb"].T + de2_skip) * (f["e2"] > 0)
    g["We2"], g["be2"] = f["e1"].T @ de2, de2.sum(0)
    de1 = (de2 @ P["We2"].T + de1_skip) * (f["e1"] > 0)
    g["We1"], g["be1"] = f["x"].T @ de1, de1.sum(0)
    return g


def unet_train(skips=True, epochs=8, batch=128, lr=1e-3, seed=0):
    path = os.path.join(CACHE, "unet_%s_e%d_s%d.npz" % ("skips" if skips else "plain",
                                                         epochs, seed))
    if os.path.exists(path):
        with np.load(path) as d:
            return {k: d[k] for k in d.files if k != "loss"}, float(d["loss"])
    xtr, _y, xte, _ = load(CACHE)
    rng = np.random.default_rng(seed)
    P = unet_init(rng, skips)
    st, step = _adam_state(P), 0
    for _ in range(epochs):
        order = rng.permutation(len(xtr))
        for s in range(0, len(order), batch):
            x = xtr[order[s:s + batch]]
            f = unet_forward(P, noisy(x, rng), skips)
            step += 1
            _adam(P, unet_backward(P, f, x, skips), st, lr, step)
    trng = np.random.default_rng(99)
    te = float(np.mean((unet_forward(P, noisy(xte[:2000], trng), skips)["out"]
                        - xte[:2000]) ** 2))
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, loss=te, **P)
    return P, te


# ================================================================== DEC / IDEC

K = 10


def soft_assign(z, mu):
    """Student-t soft assignment q (alpha = 1) and the kernel values."""
    d2 = ((z[:, None, :] - mu[None]) ** 2).sum(-1)
    k = 1.0 / (1.0 + d2)
    return k / k.sum(1, keepdims=True), k


def target(q):
    w = q ** 2 / q.sum(0)
    return w / w.sum(1, keepdims=True)


def kl_grads(z, mu, p, q, k):
    """dKL(P||Q)/dz and /dmu (mean over the batch)."""
    n = len(z)
    c = 2.0 * k * (p - q)                              # (n, K)
    diff = z[:, None, :] - mu[None]                    # (n, K, L)
    dz = (c[..., None] * diff).sum(1) / n
    dmu = -(c[..., None] * diff).sum(0) / n
    return dz, dmu


def kl_loss(p, q):
    return float(np.mean((p * np.log((p + 1e-12) / (q + 1e-12))).sum(1)))


def kmeans(z, k, rng, iters=60):
    c = z[rng.choice(len(z), k, replace=False)]
    for _ in range(iters):
        lab = ((z[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.array([z[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    return c


def cluster_accuracy(pred, true, k=K):
    """Best one-to-one mapping of clusters to labels (Hungarian, by brute
    greedy on the confusion matrix -- exact enough for 10 classes)."""
    M = np.zeros((k, k), int)
    for a, b in zip(pred, true):
        M[a, b] += 1
    # Hungarian via scipy when available, greedy otherwise
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-M)
        mapping = dict(zip(r, c))
    except ImportError:                                  # pragma: no cover
        mapping, used = {}, set()
        for a, b in sorted(np.ndindex(k, k), key=lambda ij: -M[ij]):
            if a not in mapping and b not in used:
                mapping[a] = b
                used.add(b)
    acc = sum(M[a, mapping[a]] for a in range(k)) / max(len(pred), 1)
    return float(acc), mapping


@dataclass
class ClusterRun:
    kind: str                       # "dec" or "idec"
    P0: dict                        # pretrained autoencoder
    snaps: list = field(default_factory=list)   # dict(it, z, mu, q, acc, lr, lc, recon)
    cloud_labels: np.ndarray = None
    show: np.ndarray = None         # images whose reconstructions are tracked


def _encode(P, x):
    a = x
    acts = [a]
    for k in range(3):
        z = a @ P["W%d" % k] + P["b%d" % k]
        a = z if k == 2 else np.maximum(z, 0)
        acts.append(a)
    return acts


def _decode(P, z):
    a = z
    acts = [a]
    for k in range(3, 6):
        zz = a @ P["W%d" % k] + P["b%d" % k]
        a = _sig(zz) if k == 5 else np.maximum(zz, 0)
        acts.append(a)
    return acts


def _encoder_backward(P, acts, dz, g):
    d = dz
    for k in (2, 1, 0):
        g["W%d" % k] = g.get("W%d" % k, 0) + acts[k].T @ d
        g["b%d" % k] = g.get("b%d" % k, 0) + d.sum(0)
        if k:
            d = (d @ P["W%d" % k].T) * (acts[k] > 0)


def cluster_train(kind="dec", iters=1500, batch=256, lr=1e-3, gamma=0.1, update_p=100,
                  snap_every=50, cloud=800, seed=0):
    """DEC / IDEC from the dense 3-D autoencoder; keeps snapshots of the cloud,
    the centres, the soft assignments and accuracy for the animation."""
    path = os.path.join(CACHE, "%s_i%d_s%d.npz" % (kind, iters, seed))
    P0, _ = AE.train("dense", 3)
    xtr, ytr, xte, yte = load(CACHE)
    sel = np.arange(3000, 3000 + cloud)
    xc, yc = xte[sel], yte[sel]
    show = xte[[3, 2, 1, 18]]
    run = ClusterRun(kind, P0, cloud_labels=yc, show=show)
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as d:
            run.snaps = list(d["snaps"])
        return run
    rng = np.random.default_rng(seed)
    P = {k: np.array(v) for k, v in P0.items()}
    ztr = _encode(P, xtr[:10000])[-1]
    mu = kmeans(ztr, K, rng)
    st = _adam_state({**P, "mu": mu})
    Pall = {**P, "mu": mu}

    def snap(it, lr_, lc_):
        z = _encode(Pall, xc)[-1]
        q, _k = soft_assign(z, Pall["mu"])
        acc, _m = cluster_accuracy(q.argmax(1), yc)
        rec = _decode(Pall, _encode(Pall, show)[-1])[-1]
        run.snaps.append(dict(it=it, z=z, mu=Pall["mu"].copy(), q=q, acc=acc, lr=lr_,
                              lc=lc_, recon=rec))

    p_all = None
    snap(0, 0.0, 0.0)
    for it in range(1, iters + 1):
        if (it - 1) % update_p == 0:                     # refresh the target distribution
            zz = _encode(Pall, xtr[:10000])[-1]
            q_all, _ = soft_assign(zz, Pall["mu"])
            p_all = target(q_all)
        idx = rng.integers(0, 10000, batch)
        x = xtr[idx]
        acts = _encode(Pall, x)
        z = acts[-1]
        q, kk = soft_assign(z, Pall["mu"])
        p = p_all[idx]
        dz, dmu = kl_grads(z, Pall["mu"], p, q, kk)
        lc = kl_loss(p, q)
        g = {}
        lr_ = 0.0
        scale = gamma if kind == "idec" else 1.0
        dz = dz * scale
        dmu = dmu * scale
        if kind == "idec":                               # + reconstruction
            dacts = _decode(Pall, z)
            r = dacts[-1]
            lr_ = float(np.mean((r - x) ** 2))
            d = 2 * (r - x) / r.size * r * (1 - r)
            for k in (5, 4, 3):
                g["W%d" % k] = dacts[k - 3].T @ d
                g["b%d" % k] = d.sum(0)
                d = d @ Pall["W%d" % k].T
                if k > 3:
                    d = d * (dacts[k - 3] > 0)
            dz = dz + d
        _encoder_backward(Pall, acts, dz, g)
        g["mu"] = dmu
        for k in Pall:
            g.setdefault(k, np.zeros_like(Pall[k]))
        _adam(Pall, g, st, lr, it)
        if it % snap_every == 0:
            snap(it, lr_, lc)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, snaps=np.array(run.snaps, dtype=object))
    return run

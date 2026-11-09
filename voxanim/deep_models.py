"""A dense VAE, a dense U-Net (denoising) and DEC / IDEC clustering, in plain
numpy, on MNIST.  Each is seeded, trained with Adam
(:class:`voxanim.optim.Adam`) and cached by :func:`voxanim.cache.cached_npz`.

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

from dataclasses import dataclass

import numpy as np

from . import autoencoder as AE
from .autoencoder import dense_init, sigmoid
from .cache import DEFAULT_CACHE, cached_npz
from .minibatch import minibatches
from .mnist import load
from .optim import Adam

TEST_N = 2000                      # test images the VAE and U-Net losses are measured on


# ================================================================== VAE


def vae_init(L, rng):
    """Random VAE parameters for an ``L``-D latent (``Wlv`` starts small)."""
    P = {}
    for name, a, b in (("e0", 784, 256), ("e1", 256, 64), ("mu", 64, L), ("lv", 64, L),
                       ("d0", L, 64), ("d1", 64, 256), ("d2", 256, 784)):
        dense_init(P, name, a, b, rng)
    P["Wlv"] *= 0.1
    return P


def vae_forward(P, x, eps):
    """Encode ``x`` ``(N, 784)``, sample ``z = mu + exp(lv / 2) eps`` and decode.

    ``eps`` ``(N, L)`` is the standard-normal noise (zeros: decode the mean).
    Returns ``x, eps, h0, h1, mu, lv`` (log variance), ``z, g0, g1, logits``
    and ``recon`` ``(N, 784)``.
    """
    h0 = np.maximum(x @ P["We0"] + P["be0"], 0)
    h1 = np.maximum(h0 @ P["We1"] + P["be1"], 0)
    mu = h1 @ P["Wmu"] + P["bmu"]
    lv = h1 @ P["Wlv"] + P["blv"]
    z = mu + np.exp(0.5 * lv) * eps
    g0 = np.maximum(z @ P["Wd0"] + P["bd0"], 0)
    g1 = np.maximum(g0 @ P["Wd1"] + P["bd1"], 0)
    logits = g1 @ P["Wd2"] + P["bd2"]
    return dict(x=x, eps=eps, h0=h0, h1=h1, mu=mu, lv=lv, z=z, g0=g0, g1=g1, logits=logits,
                recon=sigmoid(logits))


def vae_loss(f):
    """``(loss, bce, kl)``: per-image means of the summed binary cross-entropy,
    the KL term and their sum."""
    x, r = f["x"], np.clip(f["recon"], 1e-7, 1 - 1e-7)
    bce = -(x * np.log(r) + (1 - x) * np.log(1 - r)).sum(1)
    kl = -0.5 * (1 + f["lv"] - f["mu"] ** 2 - np.exp(f["lv"])).sum(1)
    return float(np.mean(bce + kl)), float(np.mean(bce)), float(np.mean(kl))


def vae_backward(P, f):
    """Gradients of the batch-mean VAE loss (``eps`` held fixed: reparameterised)."""
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
    """Decoder only: ``z`` ``(L,)`` or ``(N, L)`` -> ``(g0, g1, image)``, each 2-D."""
    g0 = np.maximum(np.atleast_2d(z) @ P["Wd0"] + P["bd0"], 0)
    g1 = np.maximum(g0 @ P["Wd1"] + P["bd1"], 0)
    return g0, g1, sigmoid(g1 @ P["Wd2"] + P["bd2"])


def vae_train(L=3, epochs=15, batch=128, lr=1e-3, seed=0, cache_dir=DEFAULT_CACHE):
    """Train with Adam (or load the cache): ``(params, test loss)``.

    The test loss is :func:`vae_loss` of the first :data:`TEST_N` test
    images, decoded from their means.
    """
    settings = dict(latent=L, epochs=epochs, batch=batch, lr=lr, seed=seed, test_n=TEST_N)

    def build():
        xtr, _ytr, xte, _yte = load(cache_dir)
        rng = np.random.default_rng(seed)
        P = vae_init(L, rng)
        opt = Adam(P, lr)
        for idx in minibatches(rng, len(xtr), epochs, batch):
            x = xtr[idx]
            f = vae_forward(P, x, rng.normal(size=(len(x), L)))
            opt.step(P, vae_backward(P, f))
        te = vae_loss(vae_forward(P, xte[:TEST_N], np.zeros((TEST_N, L))))[0]
        return dict(P, loss=np.array(te))

    d = cached_npz("vae", settings, build, cache_dir)
    return {k: v for k, v in d.items() if k != "loss"}, float(d["loss"])


# ================================================================== U-Net (denoising)

NOISE = 0.45                       # std of the Gaussian noise added to the digits
TEST_NOISE_SEED = 99               # the noise on the U-Net's test images


def noisy(x, rng):
    """``x`` plus Gaussian noise of std :data:`NOISE`, clipped to [0, 1]."""
    return np.clip(x + rng.normal(0, NOISE, x.shape), 0, 1)


def unet_init(rng, skips=True):
    """Random U-Net parameters; with ``skips`` the decoder layers are wider."""
    P = {}
    dense_init(P, "e1", 784, 256, rng)
    dense_init(P, "e2", 256, 64, rng)
    dense_init(P, "b", 64, 16, rng)
    dense_init(P, "d2", 16 + (64 if skips else 0), 64, rng)
    dense_init(P, "d1", 64 + (256 if skips else 0), 256, rng)
    dense_init(P, "o", 256, 784, rng)
    return P


def unet_forward(P, x, skips=True):
    """Denoise ``x`` ``(N, 784)``: every layer, the concatenations ``c2`` and
    ``c1`` (just ``b`` and ``d2`` without ``skips``) and ``out`` ``(N, 784)``."""
    e1 = np.maximum(x @ P["We1"] + P["be1"], 0)
    e2 = np.maximum(e1 @ P["We2"] + P["be2"], 0)
    b = np.maximum(e2 @ P["Wb"] + P["bb"], 0)
    c2 = np.concatenate([b, e2], 1) if skips else b
    d2 = np.maximum(c2 @ P["Wd2"] + P["bd2"], 0)
    c1 = np.concatenate([d2, e1], 1) if skips else d2
    d1 = np.maximum(c1 @ P["Wd1"] + P["bd1"], 0)
    out = sigmoid(d1 @ P["Wo"] + P["bo"])
    return dict(x=x, e1=e1, e2=e2, b=b, c2=c2, d2=d2, c1=c1, d1=d1, out=out)


def unet_backward(P, f, clean, skips=True):
    """Gradients of ``mean((out - clean)^2)`` for the forward pass ``f``."""
    g = {}
    d = AE.recon_grad(f["out"], clean)
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


def unet_train(skips=True, epochs=8, batch=128, lr=1e-3, seed=0, cache_dir=DEFAULT_CACHE):
    """Train with Adam (or load the cache): ``(params, test error)``.

    Each batch gets fresh noise from the training generator; the test error
    is the MSE on the first :data:`TEST_N` test images, noised with
    :data:`TEST_NOISE_SEED`.
    """
    settings = dict(skips=skips, epochs=epochs, batch=batch, lr=lr, seed=seed, noise=NOISE,
                    test_n=TEST_N, test_noise_seed=TEST_NOISE_SEED)

    def build():
        xtr, _ytr, xte, _yte = load(cache_dir)
        rng = np.random.default_rng(seed)
        P = unet_init(rng, skips)
        opt = Adam(P, lr)
        for idx in minibatches(rng, len(xtr), epochs, batch):
            x = xtr[idx]
            f = unet_forward(P, noisy(x, rng), skips)
            opt.step(P, unet_backward(P, f, x, skips))
        test_rng = np.random.default_rng(TEST_NOISE_SEED)
        out = unet_forward(P, noisy(xte[:TEST_N], test_rng), skips)["out"]
        return dict(P, loss=np.array(float(np.mean((out - xte[:TEST_N]) ** 2))))

    d = cached_npz("unet", settings, build, cache_dir)
    return {k: v for k, v in d.items() if k != "loss"}, float(d["loss"])


# ================================================================== DEC / IDEC

N_CLUSTERS = 10
TRAIN_SUBSET = 10000               # training images DEC sees (k-means, targets, batches)
KMEANS_ITERS = 60
CLOUD_START = 3000                 # the animated codes: test digits CLOUD_START, ...
SHOW = (3, 2, 1, 18)               # test digits whose reconstructions are tracked


def soft_assign(z, mu):
    """Student-t soft assignment ``q`` (alpha = 1) and the kernel values.

    ``z`` ``(n, L)`` codes, ``mu`` ``(K, L)`` centres; both results ``(n, K)``.
    """
    d2 = ((z[:, None, :] - mu[None]) ** 2).sum(-1)
    k = 1.0 / (1.0 + d2)
    return k / k.sum(1, keepdims=True), k


def target(q):
    """DEC's sharpened target ``p_ij = (q_ij^2 / f_j)`` renormalised per row."""
    w = q ** 2 / q.sum(0)
    return w / w.sum(1, keepdims=True)


def kl_grads(z, mu, p, q, k):
    """dKL(P||Q)/dz ``(n, L)`` and /dmu ``(K, L)`` (mean over the batch)."""
    n = len(z)
    c = 2.0 * k * (p - q)                              # (n, K)
    diff = z[:, None, :] - mu[None]                    # (n, K, L)
    dz = (c[..., None] * diff).sum(1) / n
    dmu = -(c[..., None] * diff).sum(0) / n
    return dz, dmu


def kl_loss(p, q):
    """Batch mean of KL(p || q) over rows."""
    return float(np.mean((p * np.log((p + 1e-12) / (q + 1e-12))).sum(1)))


def kmeans(z, k, rng, iters=KMEANS_ITERS):
    """Lloyd's k-means from ``k`` random codes: the centres ``(k, L)``.

    An empty cluster keeps its previous centre.
    """
    c = z[rng.choice(len(z), k, replace=False)]
    for _ in range(iters):
        lab = ((z[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.array([z[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    return c


def linear_assignment(cost):
    """Exact minimum-cost assignment for a square ``cost`` matrix.

    The Hungarian method with potentials (shortest augmenting paths,
    O(n^3)); returns ``col`` with row ``i`` assigned to column ``col[i]``.
    Pure numpy, so results never depend on whether scipy is installed.
    """
    cost = np.asarray(cost, np.float64)
    n = len(cost)
    u, v = np.zeros(n + 1), np.zeros(n + 1)          # row / column potentials
    match = np.zeros(n + 1, int)                    # column j (1-based) -> row, 0: free
    way = np.zeros(n + 1, int)
    for row in range(1, n + 1):
        match[0], j0 = row, 0
        minv = np.full(n + 1, np.inf)
        used = np.zeros(n + 1, bool)
        while match[j0]:
            used[j0] = True
            i0 = match[j0]
            free = ~used[1:]
            cur = cost[i0 - 1] - u[i0] - v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            j1 = 1 + int(np.argmin(np.where(free, minv[1:], np.inf)))
            delta = minv[j1]
            u[match[used]] += delta
            v[used] -= delta
            minv[~used] -= delta
            j0 = j1
        while j0:                                      # flip the augmenting path
            j1 = way[j0]
            match[j0] = match[j1]
            j0 = j1
    col = np.zeros(n, int)
    col[match[1:] - 1] = np.arange(n)
    return col


def cluster_accuracy(pred, true, k=N_CLUSTERS):
    """Accuracy under the best one-to-one mapping of clusters to labels.

    Returns ``(accuracy, mapping)`` with ``mapping[cluster] = label``, found
    exactly by :func:`linear_assignment` on the confusion matrix.
    """
    M = np.zeros((k, k), int)
    for a, b in zip(pred, true):
        M[a, b] += 1
    mapping = dict(enumerate(linear_assignment(-M).tolist()))
    acc = sum(M[a, mapping[a]] for a in range(k)) / max(len(pred), 1)
    return float(acc), mapping


@dataclass
class ClusterRun:
    """Snapshots of a DEC / IDEC run, stacked over ``S`` snapshots."""

    kind: str                       # "dec" or "idec"
    it: np.ndarray                  # (S,) iteration of each snapshot (0 = after k-means)
    z: np.ndarray                   # (S, m, L) codes of the cloud digits
    mu: np.ndarray                  # (S, K, L) cluster centres
    q: np.ndarray                   # (S, m, K) soft assignments of the cloud
    acc: np.ndarray                 # (S,) cluster accuracy on the cloud
    recon: np.ndarray               # (S, 4, 784) reconstructions of the SHOW digits
    cloud_labels: np.ndarray        # (m,) true labels of the cloud digits
    show: np.ndarray                # (4, 784) the SHOW digits

    def __len__(self):
        return len(self.it)


def _encode(P, x):
    """The autoencoder's code for ``x``."""
    return AE.dense_layers(P, x, AE.ENCODER)[0][-1]


def _decode(P, z):
    """The autoencoder's reconstruction of codes ``z``."""
    return AE.dense_layers(P, z, AE.DECODER)[0][-1]


def cluster_train(kind="dec", iters=1500, batch=256, lr=1e-3, gamma=0.1, update_p=100,
                  snap_every=50, cloud=800, seed=0, cache_dir=DEFAULT_CACHE):
    """DEC / IDEC, starting from the dense 3-D autoencoder
    (:func:`voxanim.autoencoder.train`).

    Adam on the encoder and the centres (IDEC: also the decoder, with the
    clustering loss weighted by ``gamma``); the target ``p`` is refreshed
    every ``update_p`` iterations; a snapshot of the cloud, the centres, the
    soft assignments, the accuracy and the reconstructions is kept every
    ``snap_every`` iterations.  The cache key includes the autoencoder's own
    settings.
    """
    settings = dict(kind=kind, iters=iters, batch=batch, lr=lr, gamma=gamma,
                    update_p=update_p, snap_every=snap_every, cloud=cloud, seed=seed,
                    clusters=N_CLUSTERS, train_subset=TRAIN_SUBSET, kmeans_iters=KMEANS_ITERS,
                    cloud_start=CLOUD_START, show=SHOW,
                    autoencoder=AE.train_settings("dense", 3))
    xtr, _ytr, xte, yte = load(cache_dir)
    sel = np.arange(CLOUD_START, CLOUD_START + cloud)
    xc, yc = xte[sel], yte[sel]
    show = xte[list(SHOW)]

    def build():
        P0, _ = AE.train("dense", 3, cache_dir=cache_dir)
        return _cluster_run(kind, P0, xtr[:TRAIN_SUBSET], xc, yc, show, iters, batch, lr,
                            gamma, update_p, snap_every, seed)

    d = cached_npz(kind, settings, build, cache_dir)
    return ClusterRun(kind, cloud_labels=yc, show=show, **d)


def _cluster_run(kind, P0, xtr, xc, yc, show, iters, batch, lr, gamma, update_p, snap_every,
                 seed):
    """The training loop of :func:`cluster_train`: the stacked snapshot arrays."""
    rng = np.random.default_rng(seed)
    P = {k: np.array(v) for k, v in P0.items()}
    P["mu"] = kmeans(_encode(P, xtr), N_CLUSTERS, rng)
    opt = Adam(P, lr)
    snaps = []

    def snap(it):
        z = _encode(P, xc)
        q, _k = soft_assign(z, P["mu"])
        acc, _m = cluster_accuracy(q.argmax(1), yc)
        snaps.append(dict(it=it, z=z, mu=P["mu"].copy(), q=q, acc=acc,
                          recon=_decode(P, _encode(P, show))))

    p_all = None
    snap(0)
    for it in range(1, iters + 1):
        if (it - 1) % update_p == 0:                     # refresh the target distribution
            q_all, _ = soft_assign(_encode(P, xtr), P["mu"])
            p_all = target(q_all)
        idx = rng.integers(0, len(xtr), batch)
        x = xtr[idx]
        acts, _pre = AE.dense_layers(P, x, AE.ENCODER)
        z = acts[-1]
        q, kk = soft_assign(z, P["mu"])
        dz, dmu = kl_grads(z, P["mu"], p_all[idx], q, kk)
        g = {}
        if kind == "idec":                               # + reconstruction
            dz, dmu = dz * gamma, dmu * gamma
            dec, _pre = AE.dense_layers(P, z, AE.DECODER)
            dz = dz + AE.dense_backward(P, dec, AE.recon_grad(dec[-1], x), AE.DECODER, g)
        AE.dense_backward(P, acts, dz, AE.ENCODER, g)
        g["mu"] = dmu
        for k in P:
            g.setdefault(k, np.zeros_like(P[k]))
        opt.step(P, g)
        if it % snap_every == 0:
            snap(it)
    return {k: np.array([s[k] for s in snaps]) for k in snaps[0]}

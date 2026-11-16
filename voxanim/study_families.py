"""The experiments behind animations 101-105: four generative families compared.

An autoencoder (AE), a variational autoencoder (VAE), a generative adversarial network
(GAN) and deep embedded clustering (DEC, and IDEC, the form that keeps its decoder) are
trained here ONCE, as one shared set (:func:`models`), and the five studies all read it:

* the same data: 20,000 MNIST training digits at 28 x 28 (``realdata.mnist_split``), and
  2,000 digits of the official test partition held out for every number reported;
* the same networks: every decoder and the GAN's generator is ``code -> 64 -> 256 -> 784``
  (ReLU, sigmoid output); every encoder is ``784 -> 256 -> 64 -> code``; the GAN's judge
  (its discriminator) has the encoder's widths with one output;
* the same code sizes: 2 numbers (so the code space can be drawn: ``generative-latent-spaces``)
  and 16 numbers (the other four animations);
* the same optimiser (:class:`voxanim.optim.Adam`), three seeds of everything.

What cannot be held equal is the objective - that is the comparison - and what follows
from it: the GAN has no encoder and takes more updates of a smaller batch; DEC and IDEC
start from the trained autoencoder of the same seed (as the method prescribes) and DEC
then moves only the encoder, so the decoder it is shown with is the autoencoder's.

The yardsticks are model-independent: a separately trained classifier
(:func:`voxanim.mnist.train`, "the reader") and distances between images.
"""

from __future__ import annotations

import os
from functools import cache, lru_cache

import numpy as np

from . import autoencoder as AE
from . import deep_models as dm
from . import mnist, realdata
from .cache import DEFAULT_CACHE, cached_npz, settings_key, sha256_of
from .next_vision import bars, curve, images, matrix, points
from .optim import Adam
from .study import Study
from .study_generative import parallel

VERSION = 1                            # bump when the shared training changes
STUDY_VERSION = 1                      # bump when a study's own measurement changes

TRAIN_N, TEST_N = 20000, 2000
SEEDS = (1, 2, 3)
CODES = (2, 16)
WIDE = 16                              # the code size of every animation but the latent spaces
STEPS, BATCH, LR = 4000, 128, 1e-3     # AE and VAE
CLUSTER_STEPS, CLUSTER_BATCH, GAMMA, UPDATE_P = 1500, 256, 0.1, 100
CLUSTERS = dm.N_CLUSTERS
GAN_STEPS, GAN_BATCH, GAN_SMOOTH, GAN_LEAK = 12000, 64, 0.9, 0.2
GAN_LR_G, GAN_LR_D, GAN_BETA1, GAN_DROP = 1e-3, 2e-4, 0.5, 0.3
SNAPS = 40                             # snapshots kept of every training run (plus the start)
EVAL_N = 500                           # held-out digits the training traces are measured on
SHOWN = 6                              # held-out digits whose outputs are kept at each snapshot
# The one of them the animations follow: a 9, the one of the six whose AE rebuilding
# error (0.012) is nearest the held-out mean (0.013).
TRACK = 3
FIXED = 200                            # fixed noise points a GAN is followed on
CLOUD = 300                            # held-out digits whose DEC / IDEC codes are kept
FAMILIES = ("ae", "vae", "gan", "dec", "idec")
NAMES = {"ae": "AE", "vae": "VAE", "gan": "GAN", "dec": "DEC", "idec": "IDEC"}


# ====================================================================== data and the reader
@lru_cache(maxsize=1)
def data():
    """The two disjoint parts every model and number here uses."""
    x, y, meta = realdata.mnist_split(TRAIN_N, "train", 28)
    xt, yt, meta_t = realdata.mnist_split(TEST_N, "test", 28)
    return dict(x=x, y=y, xt=xt, yt=yt, train_x_sha256=meta["x_sha256"],
                test_x_sha256=meta_t["x_sha256"])


@lru_cache(maxsize=1)
def reader():
    """The separately trained classifier that reads generated digits: the repository's
    MNIST MLP (784-64-32-10, trained on the official training partition)."""
    layers, acc = mnist.train()
    return [(np.asarray(W, float), np.asarray(b, float)) for W, b in layers], float(acc)


def read(x):
    """The reader's class probabilities for images ``x`` ``(n, 784)``."""
    a = np.asarray(x, float)
    layers = reader()[0]
    for W, b in layers[:-1]:
        a = np.maximum(a @ W + b, 0)
    z = a @ layers[-1][0] + layers[-1][1]
    z = z - z.max(1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(1, keepdims=True)


# ====================================================================== the networks
def decode(P, z):
    """Images from codes through a ``code-64-256-784`` decoder / generator (the dense
    autoencoder's layers 3-5)."""
    return AE.dense_layers(P, np.atleast_2d(z), AE.DECODER)[0][-1]


def encode(P, x):
    """Codes of images through a ``784-256-64-code`` encoder (dense layers 0-2)."""
    return AE.dense_layers(P, x, AE.ENCODER)[0][-1]


def vae_as_dense(P):
    """A VAE's parameters under the autoencoder's names: its mean encoder as layers
    0-2 and its decoder as layers 3-5, so :func:`encode` and :func:`decode` read it
    (``Wlv`` / ``blv``, the spread of each code, are kept beside them)."""
    names = ("e0", "e1", "mu", "d0", "d1", "d2")
    out = {"Wlv": P["Wlv"], "blv": P["blv"]}
    for k, n in enumerate(names):
        out["W%d" % k], out["b%d" % k] = P["W" + n], P["b" + n]
    return out


def judge_init(rng):
    """The GAN's discriminator: the encoder's widths, one output."""
    P = {}
    for name, a, b in (("a", 784, 256), ("b", 256, 64), ("c", 64, 1)):
        AE.dense_init(P, name, a, b, rng)
    return P


def judge_forward(D, x, rng=None):
    """``(logit, tape)`` of the discriminator on ``x`` (leaky ReLU). With ``rng`` (in
    training) each hidden unit is dropped with probability :data:`GAN_DROP`."""
    n = len(x)
    ma = mb = 1.0
    if rng is not None:
        ma = (rng.random((n, D["Wa"].shape[1])) > GAN_DROP) / (1 - GAN_DROP)
        mb = (rng.random((n, D["Wb"].shape[1])) > GAN_DROP) / (1 - GAN_DROP)
    pa = x @ D["Wa"] + D["ba"]
    ha = np.where(pa > 0, pa, GAN_LEAK * pa) * ma
    pb = ha @ D["Wb"] + D["bb"]
    hb = np.where(pb > 0, pb, GAN_LEAK * pb) * mb
    return hb @ D["Wc"] + D["bc"], (x, pa, ha, pb, hb, ma, mb)


def judge_backward(D, tape, d, grads=True):
    """Backpropagate ``d`` (gradient at the logit): ``(parameter gradients, gradient at
    the input image)``."""
    x, pa, ha, pb, hb, ma, mb = tape
    g = {}
    if grads:
        g["Wc"], g["bc"] = hb.T @ d, d.sum(0)
    d = (d @ D["Wc"].T) * np.where(pb > 0, 1.0, GAN_LEAK) * mb
    if grads:
        g["Wb"], g["bb"] = ha.T @ d, d.sum(0)
    d = (d @ D["Wb"].T) * np.where(pa > 0, 1.0, GAN_LEAK) * ma
    if grads:
        g["Wa"], g["ba"] = x.T @ d, d.sum(0)
    return g, d @ D["Wa"].T


def _sig(z):
    return AE.sigmoid(z)


class GanAdam(Adam):
    """:class:`voxanim.optim.Adam` with the first-moment decay a GAN is usually trained
    with (:data:`GAN_BETA1` = 0.5 instead of 0.9). With 0.9 the discriminator here called
    every batch correctly from the first few hundred updates and the generator never
    produced more than a few digits (see ``designs_tried``)."""

    def step(self, params, grads, lr=None):
        self.t += 1
        lr = self.lr if lr is None else lr
        t = self.t
        for k in params:
            g = grads[k]
            self.m[k] = GAN_BETA1 * self.m[k] + (1 - GAN_BETA1) * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g ** 2
            params[k] = params[k] - lr * (self.m[k] / (1 - GAN_BETA1 ** t)) / (
                np.sqrt(self.v[k] / (1 - 0.999 ** t)) + 1e-8)


def snapshot_steps(total):
    """The updates a run is recorded at: 0, then :data:`SNAPS` evenly spaced."""
    return np.round(np.linspace(0, total, SNAPS + 1)).astype(int)


# ====================================================================== training one set
def _rng(seed, code, k):
    return np.random.default_rng([seed, code, k])


def train_ae(code, seed, x, xe, shown):
    rng = _rng(seed, code, 0)
    P = AE.init("dense", code, rng)
    opt = Adam(P, LR)
    at = set(snapshot_steps(STEPS).tolist())
    mse, out = [], []
    for it in range(STEPS + 1):
        if it in at:
            mse.append(np.mean((AE.forward("dense", P, xe)["recon"] - xe) ** 2))
            out.append(AE.forward("dense", P, shown)["recon"])
        if it == STEPS:
            break
        xb = x[rng.integers(0, len(x), BATCH)]
        opt.step(P, AE.backward("dense", P, AE.forward("dense", P, xb), xb))
    return P, dict(mse=np.array(mse), shown=np.array(out))


def train_vae(code, seed, x, xe, shown):
    rng = _rng(seed, code, 1)
    P = dm.vae_init(code, rng)
    opt = Adam(P, LR)
    at = set(snapshot_steps(STEPS).tolist())
    mse, kl, bce, out = [], [], [], []
    for it in range(STEPS + 1):
        if it in at:
            f = dm.vae_forward(P, xe, np.zeros((len(xe), code)))
            _l, b, k = dm.vae_loss(f)
            mse.append(np.mean((f["recon"] - xe) ** 2))
            kl.append(k)
            bce.append(b)
            out.append(dm.vae_forward(P, shown, np.zeros((len(shown), code)))["recon"])
        if it == STEPS:
            break
        xb = x[rng.integers(0, len(x), BATCH)]
        f = dm.vae_forward(P, xb, rng.normal(size=(BATCH, code)))
        opt.step(P, dm.vae_backward(P, f))
    return P, dict(mse=np.array(mse), kl=np.array(kl), bce=np.array(bce), shown=np.array(out))


def gan_step(G, D, go, do, real, z, rng):
    """One update of the discriminator, then one of the generator (non-saturating
    loss, real labels smoothed to :data:`GAN_SMOOTH`). Returns ``(d loss, g loss, share
    of this batch the discriminator called correctly)``."""
    n = len(real)
    acts, _pre = AE.dense_layers(G, z, AE.DECODER)
    fake = acts[-1]
    xx = np.vstack((real, fake))
    target = np.r_[np.full(n, GAN_SMOOTH), np.zeros(n)][:, None]
    logit, tape = judge_forward(D, xx, rng)
    p = _sig(logit)
    d_loss = float(np.mean(np.logaddexp(0, logit) - target * logit))
    right = float(np.mean((p > 0.5) == (target > 0.5)))
    dg, _ = judge_backward(D, tape, (p - target) / len(xx))
    do.step(D, dg)
    logit, tape = judge_forward(D, fake, rng)
    g_loss = float(np.mean(np.logaddexp(0, logit) - logit))
    _, dx = judge_backward(D, tape, (_sig(logit) - 1.0) / n, grads=False)
    gg = {}
    AE.dense_backward(G, acts, dx * fake * (1 - fake), AE.DECODER, gg)
    go.step(G, gg)
    return d_loss, g_loss, right


def train_gan(code, seed, x, shown):
    rng = _rng(seed, code, 2)
    full = AE.init("dense", code, rng)
    G = {k: v for k, v in full.items() if int(k[1:]) in AE.DECODER}
    D = judge_init(rng)
    go, do = GanAdam(G, GAN_LR_G), GanAdam(D, GAN_LR_D)
    fixed = rng.normal(size=(FIXED, code))
    steps = snapshot_steps(GAN_STEPS)
    at = set(steps.tolist())
    conf, out, dl, gl, right, run, pf, pr = [], [], [], [], [], [], [], []
    for it in range(GAN_STEPS + 1):
        if it in at:
            img = decode(G, fixed)
            conf.append(read(img).max(1).mean())
            out.append(img[:SHOWN])
            pf.append(_sig(judge_forward(D, img[:SHOWN])[0])[:, 0])   # the judge, no dropout
            pr.append(_sig(judge_forward(D, shown)[0])[:, 0])
            block = np.array(run) if run else np.full((1, 3), np.nan)
            dl.append(block[:, 0].mean())
            gl.append(block[:, 1].mean())
            right.append(block[:, 2].mean())
            run = []
        if it == GAN_STEPS:
            break
        real = x[rng.integers(0, len(x), GAN_BATCH)]
        run.append(gan_step(G, D, go, do, real, rng.normal(size=(GAN_BATCH, code)), rng))
    dl[0], gl[0], right[0] = dl[1], gl[1], right[1]      # nothing was run before update 0
    return G, D, dict(conf=np.array(conf), shown=np.array(out), d_loss=np.array(dl),
                      g_loss=np.array(gl), right=np.array(right), fixed=fixed,
                      p_fake=np.array(pf), p_real=np.array(pr))


def assign(P, x):
    """DEC's cluster of each image: the nearest centre in its code space."""
    return dm.soft_assign(encode(P, x), P["mu"])[0].argmax(1)


def train_cluster(kind, code, seed, P0, x, xe, ye, shown):
    """DEC / IDEC from the trained autoencoder ``P0`` (k-means centres, then
    KL(P || Q); IDEC adds the reconstruction and weights the clustering by
    :data:`GAMMA`). DEC's decoder receives no gradient and stays the autoencoder's."""
    rng = _rng(seed, code, 3)                            # the same k-means start for both
    P = {k: np.array(v) for k, v in P0.items()}
    P["mu"] = dm.kmeans(encode(P, x), CLUSTERS, rng)
    opt = Adam(P, LR)
    at = set(snapshot_steps(CLUSTER_STEPS).tolist())
    mse, kl, acc, out, cloud, centres = [], [], [], [], [], []
    p_all = None
    for it in range(CLUSTER_STEPS + 1):
        if it in at:
            z = encode(P, xe)
            q, _k = dm.soft_assign(z, P["mu"])
            mse.append(np.mean((decode(P, z) - xe) ** 2))
            kl.append(dm.kl_loss(dm.target(q), q))
            acc.append(dm.cluster_accuracy(q.argmax(1), ye)[0])
            out.append(decode(P, encode(P, shown)))
            cloud.append(z[:CLOUD])
            centres.append(P["mu"].copy())
        if it == CLUSTER_STEPS:
            break
        if it % UPDATE_P == 0:
            p_all = dm.target(dm.soft_assign(encode(P, x), P["mu"])[0])
        idx = rng.integers(0, len(x), CLUSTER_BATCH)
        xb = x[idx]
        acts, _pre = AE.dense_layers(P, xb, AE.ENCODER)
        z = acts[-1]
        q, kk = dm.soft_assign(z, P["mu"])
        dz, dmu = dm.kl_grads(z, P["mu"], p_all[idx], q, kk)
        g = {}
        if kind == "idec":
            dz, dmu = dz * GAMMA, dmu * GAMMA
            dec, _pre = AE.dense_layers(P, z, AE.DECODER)
            dz = dz + AE.dense_backward(P, dec, AE.recon_grad(dec[-1], xb), AE.DECODER, g)
        AE.dense_backward(P, acts, dz, AE.ENCODER, g)
        g["mu"] = dmu
        for k in P:
            g.setdefault(k, np.zeros_like(P[k]))
        opt.step(P, g)
    return P, dict(mse=np.array(mse), kl=np.array(kl), acc=np.array(acc), shown=np.array(out),
                   cloud=np.array(cloud), centres=np.array(centres))


def train_set(code, seed):
    """Every family at one code size and seed: weights (float32) and training traces."""
    d = data()
    x, xe, ye, shown = d["x"], d["xt"][:EVAL_N], d["yt"][:EVAL_N], d["xt"][:SHOWN]
    out = {}

    def keep(name, P, trace):
        for k, v in P.items():
            out["%s_%s" % (name, k)] = np.asarray(v, np.float32)
        for k, v in trace.items():
            out["%s_trace_%s" % (name, k)] = np.asarray(v, np.float32)

    ae, tr = train_ae(code, seed, x, xe, shown)
    keep("ae", ae, tr)
    keep("vae", *train_vae(code, seed, x, xe, shown))
    G, D, tr = train_gan(code, seed, x, shown)
    keep("gan", G, tr)
    keep("gand", D, {})
    ae32 = {k: np.asarray(v, np.float32).astype(float) for k, v in ae.items()}
    for kind in ("dec", "idec"):                         # from the autoencoder as it is saved
        keep(kind, *train_cluster(kind, code, seed, ae32, x, xe, ye, shown))
    return out


SETTINGS = dict(version=VERSION, train_n=TRAIN_N, test_n=TEST_N, seeds=list(SEEDS),
                codes=list(CODES), steps=STEPS, batch=BATCH, lr=LR,
                cluster=dict(steps=CLUSTER_STEPS, batch=CLUSTER_BATCH, gamma=GAMMA,
                             update_p=UPDATE_P, clusters=CLUSTERS),
                gan=dict(steps=GAN_STEPS, batch=GAN_BATCH, lr_generator=GAN_LR_G,
                         lr_discriminator=GAN_LR_D, beta1=GAN_BETA1, dropout=GAN_DROP,
                         smooth=GAN_SMOOTH, leak=GAN_LEAK),
                snaps=SNAPS, eval_n=EVAL_N, shown=SHOWN, fixed=FIXED, cloud=CLOUD,
                widths="784-256-64-code-64-256-784")


@lru_cache(maxsize=1)
def models():
    """The shared set: every family x code size x seed, trained once and cached.
    Keys are ``c<code>_s<seed>_<family>_<array>``."""
    def build():
        reader()                                         # trained before the workers start
        jobs = [(code, seed) for code in CODES for seed in SEEDS]
        out = {}
        for (code, seed), got in zip(jobs, parallel(train_set, jobs)):
            for k, v in got.items():
                out["c%d_s%d_%s" % (code, seed, k)] = v
        d = data()
        out["train_x_sha256"] = np.array(d["train_x_sha256"])
        out["test_x_sha256"] = np.array(d["test_x_sha256"])
        return out
    return cached_npz("study_families_models", SETTINGS, build)


@cache
def params(family, code, seed):
    """One trained network of the shared set, as float64 arrays under the dense
    autoencoder's names (``W0 .. b5``; a GAN has only the generator's 3-5; DEC and IDEC
    also ``mu``, and DEC's decoder is the autoencoder's of the same seed)."""
    d = models()
    head = "c%d_s%d_%s_" % (code, seed, family)
    P = {k[len(head):]: np.asarray(v, float) for k, v in d.items()
         if k.startswith(head) and "trace_" not in k}
    return vae_as_dense(P) if family == "vae" else P


def trace(family, code, seed, name):
    return np.asarray(models()["c%d_s%d_%s_trace_%s" % (code, seed, family, name)], float)


# ====================================================================== yardsticks
# 99% of a standard bell's draws lie within this squared distance of its centre (the
# chi-square 99% point for 2 and for 16 numbers): "inside the bell" below.
BELL_R2 = {2: 9.2103, 16: 31.9999}
COVERED = 0.03                         # a digit is covered when the reader reads it in this share


def nearest(a, b, chunk=1000):
    """``(distance, index)`` of each row of ``a``'s nearest row of ``b`` (Euclidean)."""
    b2 = (b ** 2).sum(1)
    dist, index = [], []
    for s in range(0, len(a), chunk):
        part = a[s:s + chunk]
        d = (part ** 2).sum(1)[:, None] - 2 * part @ b.T + b2[None]
        index.append(d.argmin(1))
        dist.append(np.sqrt(np.maximum(d.min(1), 0.0)))
    return np.concatenate(dist), np.concatenate(index)


def purity(z, y, k=10):
    """Mean share of each point's ``k`` nearest neighbours (in ``z``) with its label."""
    d = (z ** 2).sum(1)[:, None] - 2 * z @ z.T + (z ** 2).sum(1)[None]
    np.fill_diagonal(d, np.inf)
    near = np.argpartition(d, k, axis=1)[:, :k]
    return float((y[near] == y[:, None]).mean())


def kmeans_accuracy(z_train, z_test, y_test, rng):
    """Cluster accuracy of k-means (fitted on training codes) on held-out codes."""
    c = dm.kmeans(z_train, CLUSTERS, rng)
    lab = ((z_test[:, None] - c[None]) ** 2).sum(-1).argmin(1)
    return dm.cluster_accuracy(lab, y_test)[0]


def edge(x):
    """Stroke sharpness of images ``(n, 784)``: mean absolute difference between
    neighbouring pixels."""
    v = np.asarray(x, float).reshape(-1, 28, 28)
    return np.abs(np.diff(v, axis=1)).mean((1, 2)) + np.abs(np.diff(v, axis=2)).mean((1, 2))


def own(key, build, **settings):
    """A study's measurements, cached under the shared set's settings and its own."""
    return cached_npz("study_families_" + key,
                      dict(version=STUDY_VERSION, models=settings_key(SETTINGS), **settings),
                      build)


# ====================================================================== sampling
SAMPLES, KEPT, PAIRS_VARIETY = 2000, 20, 4000
METHODS = ("ae", "ae_fit", "vae", "gan", "dec", "idec")
METHOD_NAMES = {"ae": "AE, codes from the bell", "ae_fit": "AE, codes from its own spread",
                "vae": "VAE, codes from the bell", "gan": "GAN, noise from the bell",
                "dec": "DEC, around its centres", "idec": "IDEC, around its centres"}
MEASURES = ("confidence", "covered", "variety", "nearest")


def cluster_spread(P, x):
    """Per cluster of a DEC / IDEC model: the per-number spread of the training codes
    assigned to it (zero for an empty cluster)."""
    z = encode(P, x)
    lab = dm.soft_assign(z, P["mu"])[0].argmax(1)
    return np.array([z[lab == j].std(0) if (lab == j).sum() > 1 else np.zeros(z.shape[1])
                     for j in range(CLUSTERS)])


def draw(method, code, seed, eps, pick):
    """``len(eps)`` new images from one model. ``eps`` is standard normal noise shared
    by every method of a seed; ``pick`` the cluster each DEC / IDEC sample starts from."""
    x = data()["x"]
    if method in ("ae", "vae", "gan"):
        return decode(params(method, code, seed), eps)
    if method == "ae_fit":
        P = params("ae", code, seed)
        z = encode(P, x)
        chol = np.linalg.cholesky(np.cov(z.T) + 1e-9 * np.eye(code))
        return decode(P, z.mean(0) + eps @ chol.T)
    P = params(method, code, seed)
    return decode(P, P["mu"][pick] + cluster_spread(P, x)[pick] * eps)


def judged(img, pairs):
    """The four model-independent measures of a set of images, and the reader's class
    histogram: mean confidence, digits covered, variety (mean distance between two of
    them) and mean distance to the nearest training image."""
    p = read(img)
    hist = np.bincount(p.argmax(1), minlength=10) / len(img)
    variety = np.sqrt(((img[pairs[:, 0]] - img[pairs[:, 1]]) ** 2).sum(1)).mean()
    near = nearest(img, data()["x"])[0]
    return np.array([p.max(1).mean(), (hist >= COVERED).sum(), variety, near.mean()]), hist


def sampling_arrays():
    def build():
        d = data()
        pairs = np.random.default_rng(11).integers(0, SAMPLES, (PAIRS_VARIETY, 2))
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        shape = (len(CODES), len(METHODS), len(SEEDS))
        metrics, hist = np.zeros(shape + (len(MEASURES),)), np.zeros(shape + (10,))
        kept = np.zeros((len(CODES), len(METHODS), KEPT, 784))
        kept_near = np.zeros_like(kept)
        kept_dist = np.zeros((len(CODES), len(METHODS), KEPT))
        centres = np.zeros((len(CODES), 2, CLUSTERS, 784))
        for ci, code in enumerate(CODES):
            for si, seed in enumerate(SEEDS):
                rng = np.random.default_rng([seed, code, 7])
                eps = rng.normal(size=(SAMPLES, code))
                pick = rng.integers(0, CLUSTERS, SAMPLES)
                for mi, method in enumerate(METHODS):
                    img = draw(method, code, seed, eps, pick)
                    metrics[ci, mi, si], hist[ci, mi, si] = judged(img, pairs)
                    if si == 0:
                        kept[ci, mi] = img[:KEPT]
                        dist, index = nearest(img[:KEPT], d["x"])
                        kept_near[ci, mi], kept_dist[ci, mi] = d["x"][index], dist
            for k, kind in enumerate(("dec", "idec")):
                P = params(kind, code, SEEDS[0])
                centres[ci, k] = decode(P, P["mu"])
        real, real_hist = judged(d["xt"], pairs)
        return dict(metrics=metrics, hist=hist, kept=kept.astype(np.float32),
                    kept_near=kept_near.astype(np.float32), kept_dist=kept_dist,
                    centres=centres.astype(np.float32), real=real, real_hist=real_hist,
                    train_x_sha256=np.array(d["train_x_sha256"]),
                    test_x_sha256=np.array(d["test_x_sha256"]))
    return own("sampling", build, samples=SAMPLES, covered=COVERED, methods=list(METHODS),
               pairs=PAIRS_VARIETY)


# ====================================================================== a GAN has no encoder
INV_N, INV_OWN, INV_STEPS, INV_RESTARTS, INV_LR = 500, 200, 300, 8, 0.05
INV_AT = (0, 1, 2, 5, 10, 20, 40, 80, 150, 300)        # search steps whose images are kept


def invert(G, x, code, rng):
    """Search a generator's noise space for each image of ``x``: gradient descent
    (Adam) on ``z`` from :data:`INV_RESTARTS` random starts, minimising the squared
    error of ``G(z)``; the best start is kept. Returns ``z`` ``(n, code)``, its error
    per image, the mean error after every step (of the start that is best at that
    step), and for the first :data:`SHOWN` images the kept start's ``z`` and image at
    the steps :data:`INV_AT``."""
    n, R = len(x), INV_RESTARTS
    xx = np.repeat(x, R, 0)
    P = {"z": rng.normal(size=(n * R, code))}
    opt = Adam(P, INV_LR)
    curve_, zs, imgs = [], [], []
    for it in range(INV_STEPS + 1):
        acts, _pre = AE.dense_layers(G, P["z"], AE.DECODER)
        r = acts[-1]
        err = ((r - xx) ** 2).mean(1).reshape(n, R)
        curve_.append(err.min(1).mean())
        if it in INV_AT:
            zs.append(P["z"].reshape(n, R, code)[:SHOWN].copy())
            imgs.append(r.reshape(n, R, 784)[:SHOWN].copy())
        if it == INV_STEPS:
            break
        d = 2.0 * (r - xx) / 784.0 * r * (1 - r)
        opt.step(P, {"z": AE.dense_backward(G, acts, d, AE.DECODER, {})})
    best = err.argmin(1)
    rows = np.arange(n)
    z = P["z"].reshape(n, R, code)[rows, best]
    shown = np.arange(min(SHOWN, n))
    return (z, err[rows, best], np.array(curve_), np.array(zs)[:, shown, best[:SHOWN]],
            np.array(imgs)[:, shown, best[:SHOWN]])


def invert_job(code, seed):
    d = data()
    G = params("gan", code, seed)
    rng = np.random.default_rng([seed, code, 9])
    z, err, curve_, zs, imgs = invert(G, d["xt"][:INV_N], code, rng)
    # the control: digits the generator certainly can draw (its own samples)
    truth = rng.normal(size=(INV_OWN, code))
    _z, own_err, own_curve, _zs, _imgs = invert(G, decode(G, truth), code, rng)
    return dict(z=z, err=err, curve=curve_, path_z=zs, path_img=imgs.astype(np.float32),
                own_err=own_err, own_curve=own_curve)


def inversions():
    """Every GAN's noise point for each of the first :data:`INV_N` held-out digits."""
    def build():
        jobs = [(code, seed) for code in CODES for seed in SEEDS]
        out = {}
        for (code, seed), got in zip(jobs, parallel(invert_job, jobs)):
            for k, v in got.items():
                out["c%d_s%d_%s" % (code, seed, k)] = v
        return out
    models()                                             # before any worker needs them
    return own("inversions", build, n=INV_N, own=INV_OWN, steps=INV_STEPS,
               restarts=INV_RESTARTS, lr=INV_LR, at=list(INV_AT))


def codes_of(family, code, seed, x=None):
    """Each family's code for the first :data:`INV_N` held-out digits: an encoder's
    output (the VAE's mean), or for the GAN the noise point found by :func:`invert`."""
    if family == "gan":
        return np.asarray(inversions()["c%d_s%d_z" % (code, seed)], float)
    return encode(params(family, code, seed), data()["xt"][:INV_N] if x is None else x)


# ====================================================================== reconstruction
RECON_KEPT = 8


def recon_arrays():
    def build():
        d = data()
        x, y = d["xt"][:INV_N], d["yt"][:INV_N]
        inv = inversions()
        shape = (len(CODES), len(FAMILIES), len(SEEDS))
        mse, right, sharp = np.zeros(shape), np.zeros(shape), np.zeros(shape)
        maps = np.zeros((len(CODES), len(FAMILIES), 784))
        kept = np.zeros((len(CODES), len(FAMILIES), RECON_KEPT, 784))
        per_digit = np.zeros((len(CODES), len(FAMILIES), 10))
        e0 = edge(x)
        for ci, code in enumerate(CODES):
            for fi, family in enumerate(FAMILIES):
                for si, seed in enumerate(SEEDS):
                    r = decode(params(family, code, seed), codes_of(family, code, seed))
                    err = (r - x) ** 2
                    mse[ci, fi, si] = err.mean()
                    right[ci, fi, si] = (read(r).argmax(1) == y).mean()
                    sharp[ci, fi, si] = (edge(r) / e0).mean()
                    maps[ci, fi] += err.mean(0) / len(SEEDS)
                    per_digit[ci, fi] += np.array([err[y == k].mean() for k in range(10)]
                                                  ) / len(SEEDS)
                    if si == 0:
                        kept[ci, fi] = r[:RECON_KEPT]
        seeds = [(code, seed) for code in CODES for seed in SEEDS]
        return dict(
            mse=mse, right=right, sharp=sharp, maps=maps, per_digit=per_digit,
            kept=kept.astype(np.float32), originals=x[:RECON_KEPT], labels=y[:RECON_KEPT],
            right_original=np.array((read(x).argmax(1) == y).mean()),
            search=np.array([inv["c%d_s%d_curve" % k] for k in seeds]).reshape(
                len(CODES), len(SEEDS), -1),
            search_own=np.array([inv["c%d_s%d_own_curve" % k] for k in seeds]).reshape(
                len(CODES), len(SEEDS), -1),
            own_mse=np.array([inv["c%d_s%d_own_err" % k].mean() for k in seeds]).reshape(
                len(CODES), len(SEEDS)),
            path_img=np.array([inv["c%d_s%d_path_img" % (code, SEEDS[0])] for code in CODES]),
            train_x_sha256=np.array(d["train_x_sha256"]),
            test_x_sha256=np.array(d["test_x_sha256"]))
    return own("recon", build, n=INV_N, kept=RECON_KEPT)


# ====================================================================== interpolation
WALK_PAIRS, WALK = 200, 9
WALKERS = FAMILIES + ("pixels",)       # the control: a cross-fade of the two images
WALK_MEASURES = ("confidence", "nearest", "neither")
WALK_KEPT = 4


def walk_pairs():
    """:data:`WALK_PAIRS` pairs of held-out digits with different labels."""
    y = data()["yt"][:INV_N]
    rng = np.random.default_rng(5)
    pairs = []
    while len(pairs) < WALK_PAIRS:
        i, j = rng.integers(0, INV_N, 2)
        if y[i] != y[j]:
            pairs.append((int(i), int(j)))
    return np.array(pairs)


def walk_images(family, code, seed, pairs):
    """The images along a straight line between the two digits of each pair, in
    ``family``'s code space (``pixels``: in image space): ``(pairs, WALK, 784)``."""
    t = np.linspace(0.0, 1.0, WALK)[None, :, None]
    if family == "pixels":
        x = data()["xt"][:INV_N]
        return x[pairs[:, 0]][:, None] * (1 - t) + x[pairs[:, 1]][:, None] * t
    z = codes_of(family, code, seed)
    line = z[pairs[:, 0]][:, None] * (1 - t) + z[pairs[:, 1]][:, None] * t
    return decode(params(family, code, seed), line.reshape(-1, code)).reshape(
        len(pairs), WALK, 784)


def walk_job(code, seed):
    d = data()
    pairs = walk_pairs()
    y = d["yt"][:INV_N]
    out = np.zeros((len(WALKERS), len(WALK_MEASURES), WALK))
    kept = np.zeros((len(WALKERS), WALK_KEPT, WALK, 784), np.float32)
    for fi, family in enumerate(WALKERS):
        img = walk_images(family, code, seed, pairs)
        flat = img.reshape(-1, 784)
        p = read(flat)
        top = p.argmax(1).reshape(len(pairs), WALK)
        out[fi, 0] = p.max(1).reshape(len(pairs), WALK).mean(0)
        out[fi, 1] = nearest(flat, d["x"])[0].reshape(len(pairs), WALK).mean(0)
        out[fi, 2] = ((top != y[pairs[:, 0]][:, None]) & (top != y[pairs[:, 1]][:, None])
                      ).mean(0)
        kept[fi] = img[:WALK_KEPT]
    return out, kept


def interp_arrays():
    def build():
        d = data()
        jobs = [(code, seed) for code in CODES for seed in SEEDS]
        got = parallel(walk_job, jobs)
        walk = np.array([g[0] for g in got]).reshape(
            (len(CODES), len(SEEDS), len(WALKERS), len(WALK_MEASURES), WALK))
        kept = np.array([got[ci * len(SEEDS)][1] for ci in range(len(CODES))])
        pairs = walk_pairs()
        return dict(walk=walk, kept=kept, pairs=pairs,
                    pair_labels=d["yt"][:INV_N][pairs],
                    ends=d["xt"][:INV_N][pairs[:WALK_KEPT]],
                    train_x_sha256=np.array(d["train_x_sha256"]),
                    test_x_sha256=np.array(d["test_x_sha256"]))
    inversions()
    return own("interp", build, pairs=WALK_PAIRS, walk=WALK, kept=WALK_KEPT,
               walkers=list(WALKERS))


# ====================================================================== latent spaces
ENCODERS = ("ae", "vae", "dec", "idec")
MAP_SIDE, MAP_LIM, TILES = 41, 3.0, 9
REACH = 0.25                           # a bell draw is "used" when a held-out code is this near


def latent_arrays():
    def build():
        d = data()
        x, xt, yt = d["x"], d["xt"], d["yt"]
        shape = (len(CODES), len(ENCODERS), len(SEEDS))
        pure, acc, inside, used = (np.zeros(shape) for _ in range(4))
        codes2 = np.zeros((len(ENCODERS), len(SEEDS), TEST_N, 2))
        centres2 = np.zeros((2, len(SEEDS), CLUSTERS, 2))
        for ci, code in enumerate(CODES):
            for si, seed in enumerate(SEEDS):
                bell = np.random.default_rng([seed, code, 13]).normal(size=(TEST_N, code))
                for fi, family in enumerate(ENCODERS):
                    P = params(family, code, seed)
                    z = encode(P, xt)
                    pure[ci, fi, si] = purity(z, yt)
                    if family in ("dec", "idec"):
                        acc[ci, fi, si] = dm.cluster_accuracy(assign(P, xt), yt)[0]
                    else:
                        acc[ci, fi, si] = kmeans_accuracy(
                            encode(P, x), z, yt, np.random.default_rng([seed, code, 3]))
                    inside[ci, fi, si] = ((z ** 2).sum(1) <= BELL_R2[code]).mean()
                    used[ci, fi, si] = (nearest(bell, z)[0] <= REACH).mean()
                    if code == 2:
                        codes2[fi, si] = z
                        if family in ("dec", "idec"):
                            centres2[ENCODERS.index(family) - 2, si] = P["mu"]
        # the reference: neighbours and k-means in pixel space, no model at all
        pixel_purity = purity(xt, yt)
        pixel_acc = np.array([kmeans_accuracy(x[:5000], xt, yt,
                                              np.random.default_rng([seed, 0, 3]))
                              for seed in SEEDS])
        # the GAN's noise plane: what the generator draws at each point, as the reader reads it
        axis = np.linspace(-MAP_LIM, MAP_LIM, MAP_SIDE)
        grid = np.array([(a, b) for b in axis[::-1] for a in axis])     # row 0 at the top
        tiles_axis = np.linspace(-MAP_LIM * 0.8, MAP_LIM * 0.8, TILES)
        tile_grid = np.array([(a, b) for b in tiles_axis[::-1] for a in tiles_axis])
        gan_class = np.zeros((len(SEEDS), MAP_SIDE, MAP_SIDE), int)
        gan_conf = np.zeros((len(SEEDS), MAP_SIDE, MAP_SIDE))
        gan_pure, gan_covered = np.zeros(len(SEEDS)), np.zeros(len(SEEDS))
        tiles = np.zeros((TILES * TILES, 784), np.float32)
        for si, seed in enumerate(SEEDS):
            G = params("gan", 2, seed)
            p = read(decode(G, grid))
            gan_class[si] = p.argmax(1).reshape(MAP_SIDE, MAP_SIDE)
            gan_conf[si] = p.max(1).reshape(MAP_SIDE, MAP_SIDE)
            z = np.random.default_rng([seed, 2, 13]).normal(size=(TEST_N, 2))
            read_as = read(decode(G, z)).argmax(1)
            gan_pure[si] = purity(z, read_as)
            gan_covered[si] = (np.bincount(read_as, minlength=10) / TEST_N >= COVERED).sum()
            if si == 0:
                tiles = decode(G, tile_grid).astype(np.float32)
        return dict(purity=pure, accuracy=acc, inside=inside, used=used, codes=codes2,
                    centres=centres2, labels=yt, pixel_purity=np.array(pixel_purity),
                    pixel_accuracy=pixel_acc, gan_class=gan_class, gan_conf=gan_conf,
                    gan_purity=gan_pure, gan_covered=gan_covered, tiles=tiles,
                    tile_grid=tile_grid, shown=xt[:SHOWN],
                    train_x_sha256=np.array(d["train_x_sha256"]),
                    test_x_sha256=np.array(d["test_x_sha256"]))
    return own("latent", build, map_side=MAP_SIDE, map_lim=MAP_LIM, tiles=TILES, reach=REACH,
               bell_r2=BELL_R2)


# ====================================================================== the five studies
HELD_EQUAL = [
    "data: the same %s MNIST training digits at 28 x 28; every number is measured on "
    "held-out digits of the official test partition" % f"{TRAIN_N:,}",
    "networks: every decoder and the GAN's generator is code-64-256-784 (ReLU, sigmoid); "
    "every encoder is 784-256-64-code; the GAN's discriminator has the encoder's widths",
    "code size: 2 numbers (generative-latent-spaces) or 16 (the other four animations)",
    "optimiser Adam; seeds %s for every family and code size" % (list(SEEDS),),
    "yardsticks: one separately trained classifier and distances between images, the same "
    "for every family",
]
NOT_EQUAL = [
    "the objective (the comparison itself): AE mean squared error; VAE cross-entropy plus "
    "KL to a standard normal; GAN non-saturating adversarial loss; DEC KL(P || Q) on soft "
    "cluster assignments; IDEC reconstruction + %.1f x that" % GAMMA,
    "updates: AE and VAE %d of batch %d; GAN %d of batch %d (generator rate %g, "
    "discriminator %g, first-moment decay %g, discriminator dropout %g); DEC and IDEC %d "
    "of batch %d on top of the trained AE of the same seed" % (
        STEPS, BATCH, GAN_STEPS, GAN_BATCH, GAN_LR_G, GAN_LR_D, GAN_BETA1, GAN_DROP,
        CLUSTER_STEPS, CLUSTER_BATCH),
    "the GAN has no encoder: wherever a code of a real digit is needed it is found by "
    "gradient descent on the noise (%d steps, %d starts)" % (INV_STEPS, INV_RESTARTS),
    "DEC trains only the encoder and the centres; the decoder it is shown with is the "
    "autoencoder's, left behind. IDEC is the form that keeps training the decoder",
]
LIMITS = [
    "small dense networks on 20,000 digits: far from the quality any of these families "
    "reaches with convolutions and more data; the comparison is at equal small size",
    "the reader is a small MLP (784-64-32-10): its confidence on images unlike any digit "
    "is not calibrated, so confidence is reported beside coverage, variety and distances",
    "distance to the nearest training image favours blurred images; it is read against the "
    "same distance for real held-out digits",
    "cluster accuracy of DEC / IDEC is about 0.5 here: the autoencoder they start from is "
    "small and not pre-trained layer-wise as in the paper",
]
DESIGNS = [
    "GAN with the repository's Adam (first-moment decay 0.9), both rates 2e-4, no dropout: "
    "the discriminator called every batch correctly and the generator drew 2 to 4 digits; "
    "decay 0.5 with discriminator dropout 0.3 gave 8 to 9 digits, and a generator rate of "
    "1e-3 all ten at 16 numbers. Chosen on the GAN's own coverage, before any comparison "
    "between families was computed",
    "DEC and IDEC with 6,000 instead of 1,500 clustering updates: cluster accuracy flat "
    "after about 1,000 (0.53 at both); 1,500 kept",
    "every yardstick (reader confidence, digits covered at 3%, variety, nearest training "
    "image, neighbour purity at k = 10, the bell's 99% radius, reach 0.25) was fixed before "
    "the families were compared; none was changed afterwards. 'Bell used' is reported only "
    "for 2 numbers: at 16 numbers no draw has a held-out code within reach for any family",
]


def each(names, n):
    """A label per image of an images panel: ``n`` in a row for each of ``names``."""
    return ["%s %d" % (name, k + 1) for name in names for k in range(n)]


def stat(a):
    """``mean (min-max)`` of the per-seed values ``a`` as plain floats."""
    a = np.asarray(a, float)
    return dict(mean=float(a.mean()), min=float(a.min()), max=float(a.max()))


def cache_file(key, settings):
    return os.path.abspath(os.path.join(DEFAULT_CACHE, "study_families_%s-%s.npz" % (
        key, settings_key(settings))))


def result(name, subtitle, panels, **metadata):
    d = models()
    path = cache_file("models", SETTINGS)
    return Study(
        name.replace("-", " ").capitalize(), subtitle, tuple(panels),
        dict(source="NumPy measurements on MNIST", dataset="MNIST", review_only=True,
             new=True, seeds=list(SEEDS), codes=list(CODES), run_settings=SETTINGS,
             artifact=path, artifact_sha256=sha256_of(path),
             data=dict(source_url=mnist.URL, raw_sha256=mnist.SHA256,
                       train_x_sha256=str(d["train_x_sha256"]),
                       test_x_sha256=str(d["test_x_sha256"]),
                       splits="training: realdata.mnist_split(%d, 'train', 28); held out: "
                              "realdata.mnist_split(%d, 'test', 28), the official test "
                              "partition" % (TRAIN_N, TEST_N)),
             reader="voxanim.mnist.train(): MLP 784-64-32-10, test accuracy %.4f" % reader()[1],
             held_equal=HELD_EQUAL, not_equal=NOT_EQUAL, limitations=LIMITS,
             designs_tried=DESIGNS, **metadata))


# ---------------------------------------------------------------------- 101 objectives
YARDSTICKS = ("rebuild error", "codes inside the bell", "cluster accuracy", "samples read")
RUN_SHOTS = (0, 1, 3, 8, 20, 40)       # the snapshots of each run that are shown


def yardsticks(code=WIDE):
    """Every family on four yardsticks, per seed: ``(values (family, yardstick, seed),
    defined (family, yardstick))``. Each yardstick is what one family is trained for:
    held-out rebuilding error (AE), share of held-out codes inside the standard bell's
    99% radius (VAE), cluster accuracy (DEC) and the reader's confidence in new samples
    (GAN). For the GAN the rebuilding error is by search and the bell is exact by
    construction; it has no clusters."""
    ci = CODES.index(code)
    r, lat, s = recon_arrays(), latent_arrays(), sampling_arrays()
    v = np.zeros((len(FAMILIES), len(YARDSTICKS), len(SEEDS)))
    defined = np.ones(v.shape[:2], bool)
    enc = [FAMILIES.index(f) for f in ENCODERS]
    gan = FAMILIES.index("gan")
    v[:, 0] = r["mse"][ci]
    v[enc, 1] = lat["inside"][ci]
    v[gan, 1] = 1.0
    v[enc, 2] = lat["accuracy"][ci]
    defined[gan, 2] = False
    v[:, 3] = s["metrics"][ci][[METHODS.index(f) for f in FAMILIES], :, 0]
    return v, defined


def run_traces(code=WIDE):
    """Held-out rebuilding error through each run that has an encoder (per seed), the
    GAN's game (share of batches' images its discriminator calls correctly, and the
    reader's confidence in its samples), DEC and IDEC cluster accuracy."""
    def get(f, n):
        return np.array([trace(f, code, s, n) for s in SEEDS])

    return dict(mse={f: get(f, "mse") for f in ENCODERS}, right=get("gan", "right"),
                conf=get("gan", "conf"), acc={f: get(f, "acc") for f in ("dec", "idec")},
                kl=get("vae", "kl"))


def objectives():
    tr = run_traces()
    v, defined = yardsticks()
    frac = np.linspace(0.0, 1.0, SNAPS + 1)
    shots = np.array([trace(f, WIDE, SEEDS[0], "shown")[list(RUN_SHOTS), 0 if f == "gan" else TRACK]
                      for f in FAMILIES])
    m = v.mean(2)
    return result(
        "generative-objectives",
        "AE, VAE, GAN, DEC and IDEC with the same decoder, data and 16-number code: what "
        "each is trained to do and what that buys, three seeds each",
        [
            curve("Rebuilding held-out digits through training",
                  "Mean squared error on 500 held-out digits, mean of three seeds, against "
                  "the share of each run done. DEC's rises: its decoder is left behind.",
                  np.array([tr["mse"][f].mean(0) for f in ENCODERS]),
                  [NAMES[f] for f in ENCODERS], frac),
            curve("The GAN's game through training",
                  "Share of images its discriminator calls correctly, and the reader's "
                  "confidence in 200 fixed samples; mean of three seeds.",
                  np.array([tr["right"].mean(0), tr["conf"].mean(0)]),
                  ["discriminator right", "reader confidence"], frac),
            matrix_with("Every family on every family's yardstick",
                         "Rows: AE, VAE, GAN, DEC, IDEC. Columns: %s. Mean of three "
                         "seeds; 'seeds' holds each." % ", ".join(YARDSTICKS),
                         m, seeds=v, defined=defined),
            images("The same held-out digit at six moments of each run",
                   "Seed 1, snapshots %s of 40; the GAN's row is one fixed noise point."
                   % (list(RUN_SHOTS),), shots.reshape(-1, 28, 28),
                   each([NAMES[f] for f in FAMILIES], len(RUN_SHOTS))),
        ],
        code=WIDE, yardsticks=list(YARDSTICKS), families=list(FAMILIES),
        table=m.tolist(), table_min=v.min(2).tolist(), table_max=v.max(2).tolist(),
        final_mse={f: stat(tr["mse"][f][:, -1]) for f in ENCODERS},
        start_mse_cluster=stat(tr["mse"]["dec"][:, 0]),
        vae_kl=stat(tr["kl"][:, -1]),
        gan_right=stat(tr["right"][:, -1]), gan_conf=stat(tr["conf"][:, -1]),
        cluster_accuracy_start=stat(tr["acc"]["dec"][:, 0]),
        cluster_accuracy_end={f: stat(tr["acc"][f][:, -1]) for f in ("dec", "idec")},
        scope="one small dense architecture; DEC's cluster picture is drawn from the "
              "2-number models, every number from the 16-number ones")


def matrix_with(title, caption, values, **extra):
    p = matrix(title, caption, np.asarray(values))
    p.data.update(extra)
    return p


# ---------------------------------------------------------------------- 102 latent spaces
def latent_spaces():
    d = latent_arrays()
    two = CODES.index(2)
    return result(
        "generative-latent-spaces",
        "Where 2,000 held-out digits sit in the 2-number code space of an AE, a VAE, DEC and "
        "IDEC, and what a GAN draws at each point of its noise plane; three seeds",
        [
            points("Held-out digits in the autoencoder's code space",
                   "Seed 1: each point is one of 2,000 held-out digits at its two code "
                   "numbers, coloured by digit.", d["codes"][0, 0], d["labels"]),
            matrix_with("What the GAN draws across its noise plane",
                         "Seed 1: the digit the reader reads in the image generated at each "
                         "of 41 x 41 noise points from -3 to 3. A GAN has no encoder, so "
                         "real digits cannot be placed here.", d["gan_class"][0],
                         confidence=d["gan_conf"][0]),
            bars("Neighbours that are the same digit",
                 "Share of each held-out digit's 10 nearest held-out neighbours in code "
                 "space with its label, mean of three seeds: AE, VAE, DEC, IDEC.",
                 d["purity"][two].mean(1), [NAMES[f] for f in ENCODERS]),
            bars("Codes inside the standard bell",
                 "Share of held-out codes within the radius holding 99% of a standard "
                 "bell's draws, mean of three seeds: AE, VAE, DEC, IDEC.",
                 d["inside"][two].mean(1), [NAMES[f] for f in ENCODERS]),
        ],
        code=2, encoders=list(ENCODERS),
        purity={f: stat(d["purity"][two, i]) for i, f in enumerate(ENCODERS)},
        purity_16={f: stat(d["purity"][1 - two, i]) for i, f in enumerate(ENCODERS)},
        cluster_accuracy={f: stat(d["accuracy"][two, i]) for i, f in enumerate(ENCODERS)},
        cluster_accuracy_16={f: stat(d["accuracy"][1 - two, i])
                             for i, f in enumerate(ENCODERS)},
        inside_bell={f: stat(d["inside"][two, i]) for i, f in enumerate(ENCODERS)},
        bell_used={f: stat(d["used"][two, i]) for i, f in enumerate(ENCODERS)},
        pixel_purity=float(d["pixel_purity"]), pixel_cluster_accuracy=stat(d["pixel_accuracy"]),
        gan_noise_purity=stat(d["gan_purity"]), gan_digits_covered=d["gan_covered"].tolist(),
        bell_radius=float(np.sqrt(BELL_R2[2])), reach=REACH,
        scope="2-number codes, so that the space itself can be drawn; cluster accuracy for "
              "AE and VAE is k-means on their codes (they have no clusters of their own); "
              "the GAN's purity is of the digit the reader reads in what it draws, not of "
              "true labels")


# ---------------------------------------------------------------------- 103 sampling
def sampling():
    d = sampling_arrays()
    ci = CODES.index(WIDE)
    m = d["metrics"][ci]                                   # method, seed, measure
    def table(c):
        return {meth: {name: stat(d["metrics"][c][i, :, k]) for k, name in enumerate(MEASURES)}
                for i, meth in enumerate(METHODS)}

    return result(
        "generative-sampling",
        "2,000 new digits from each of AE, VAE, GAN, DEC and IDEC (16-number codes), judged "
        "by a separate classifier and by distances between images; three seeds",
        [
            images("Twelve draws from each",
                   "Seed 1, the first 12 of 2,000, from the same noise: %s."
                   % "; ".join(METHOD_NAMES[k] for k in METHODS),
                   d["kept"][ci][:, :12].reshape(-1, 28, 28),
                   each([METHOD_NAMES[k] for k in METHODS], 12)),
            bars("The reader's confidence in the draws",
                 "Mean highest class probability over 2,000 draws, mean of three seeds; "
                 "real held-out digits: %.3f." % d["real"][0],
                 m[:, :, 0].mean(1), [METHOD_NAMES[k] for k in METHODS]),
            matrix_with("Four measures of each family's draws",
                         "Rows: the six ways of drawing. Columns: reader confidence, digits "
                         "covered (read in at least 3% of draws), variety (mean distance "
                         "between two draws), mean distance to the nearest of 20,000 "
                         "training images. Mean of three seeds.", m.mean(1), seeds=m,
                         real=d["real"]),
            matrix_with("Which digits the reader reads in the draws",
                         "Share of 2,000 draws read as each digit 0-9, mean of three "
                         "seeds; rows as before.", d["hist"][ci].mean(1),
                         real=d["real_hist"]),
        ],
        code=WIDE, methods=list(METHODS), measures=list(MEASURES), samples=SAMPLES,
        covered_threshold=COVERED, table=table(ci), table_2=table(1 - ci),
        real={name: float(d["real"][k]) for k, name in enumerate(MEASURES)},
        scope="DEC and IDEC have no prior to draw from: a cluster is picked evenly and its "
              "centre is moved by that cluster's own spread; DEC's decoder is the "
              "autoencoder's. 'AE, codes from its own spread' fits one bell to the AE's "
              "training codes, which the AE itself never does")


# ---------------------------------------------------------------------- 104 reconstruction
def reconstruction():
    d = recon_arrays()
    ci = CODES.index(WIDE)
    def per(key, c):
        return {f: stat(d[key][c, i]) for i, f in enumerate(FAMILIES)}

    return result(
        "generative-reconstruction",
        "The same 500 held-out digits through AE, VAE, DEC and IDEC and back, and searched "
        "for in a GAN's noise space; 16-number codes, three seeds",
        [
            images("Eight held-out digits and what comes back",
                   "Seed 1. Rows: the originals, then AE, VAE, GAN (by search), DEC (the "
                   "autoencoder's decoder), IDEC.",
                   np.concatenate([d["originals"][None], d["kept"][ci]]).reshape(-1, 28, 28),
                   each(["original"] + [NAMES[f] for f in FAMILIES], RECON_KEPT)),
            bars("Held-out rebuilding error",
                 "Mean squared error per pixel over 500 held-out digits, mean of three "
                 "seeds: AE, VAE, GAN by search, DEC, IDEC.",
                 d["mse"][ci].mean(1), [NAMES[f] for f in FAMILIES]),
            curve("The GAN's search",
                  "Error of the best of 8 starts after each of 300 steps of gradient "
                  "descent on the noise, mean of three seeds: for held-out digits, and for "
                  "the generator's own samples (the control: digits it certainly can draw).",
                  np.array([d["search"][ci].mean(0), d["search_own"][ci].mean(0)]),
                  ["held-out digits", "its own samples"], np.arange(INV_STEPS + 1)),
            matrix_with("Where the error falls",
                         "Mean squared error at each of the 784 pixels, rows AE, VAE, GAN, "
                         "DEC, IDEC; mean of 500 digits and three seeds.", d["maps"][ci]),
        ],
        code=WIDE, families=list(FAMILIES), digits=INV_N,
        mse=per("mse", ci), mse_2=per("mse", 1 - ci),
        read_right=per("right", ci), read_right_original=float(d["right_original"]),
        sharpness=per("sharp", ci),
        gan_own_sample_mse=stat(d["own_mse"][ci]),
        search=dict(steps=INV_STEPS, restarts=INV_RESTARTS, lr=INV_LR),
        scope="the VAE rebuilds from the mean of its code, without its noise; the GAN's "
              "number is the best that 300 steps from 8 starts found, an upper bound on "
              "what its generator can do")


# ---------------------------------------------------------------------- 105 interpolation
def bump(walk):
    """How much a measure at the middle of the walk exceeds the mean of its two ends:
    ``walk`` ``(..., WALK)`` -> ``(...)``."""
    return walk[..., WALK // 2] - 0.5 * (walk[..., 0] + walk[..., -1])


def interpolation():
    d = interp_arrays()
    ci = CODES.index(WIDE)
    w = d["walk"][ci]                                      # seed, walker, measure, step
    t = np.linspace(0.0, 1.0, WALK)
    names = [NAMES.get(f, "cross-fade") for f in WALKERS]
    far = bump(w[:, :, 1])                                 # seed, walker
    def per(a):
        return {f: stat(a[:, i]) for i, f in enumerate(WALKERS)}

    return result(
        "generative-interpolation",
        "A straight walk between 200 pairs of different held-out digits in each family's "
        "16-number code space, against a cross-fade of the two images; three seeds",
        [
            images("One walk in each space",
                   "Seed 1, the first pair. Rows: AE, VAE, GAN (between the two searched "
                   "noise points), DEC, IDEC, then a cross-fade of the pixels.",
                   d["kept"][ci][:, 0].reshape(-1, 28, 28), each(names, WALK)),
            curve("The reader's confidence along the walk",
                  "Mean highest class probability at 9 points of the walk, over 200 pairs "
                  "and three seeds.", w[:, :, 0].mean(0), names, t),
            curve("Distance to the nearest training digit along the walk",
                  "Mean distance to the nearest of 20,000 training images at each point.",
                  w[:, :, 1].mean(0), names, t),
            bars("How much further from any training digit the middle is than the ends",
                 "Distance at the midpoint minus the mean of the two ends, mean of three "
                 "seeds.", far.mean(0), names),
        ],
        code=WIDE, walkers=list(WALKERS), pairs=WALK_PAIRS, points=WALK,
        middle_further=per(far),
        middle_confidence=per(w[:, :, 0, WALK // 2]),
        end_confidence=per(0.5 * (w[:, :, 0, 0] + w[:, :, 0, -1])),
        middle_neither=per(w[:, :, 2, WALK // 2]),
        end_neither=per(0.5 * (w[:, :, 2, 0] + w[:, :, 2, -1])),
        middle_further_2=per(bump(d["walk"][1 - ci][:, :, 1])),
        scope="the ends of each walk are that family's own codes of the two digits (the "
              "GAN's: found by search), so each row starts and ends at its own rebuilding, "
              "not at the digits themselves; the cross-fade has no seed and is the same "
              "in all three")


STUDIES = {
    "generative-objectives": objectives,
    "generative-latent-spaces": latent_spaces,
    "generative-sampling": sampling,
    "generative-reconstruction": reconstruction,
    "generative-interpolation": interpolation,
}

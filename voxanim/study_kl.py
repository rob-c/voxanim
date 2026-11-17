"""The experiments behind animations 170-171 (see :func:`voxanim.next_catalog.redesigned`).

``vae-kl-term``: the dense VAE of :mod:`voxanim.deep_models` with a code of two numbers,
trained on the 20,000 MNIST digits of :func:`voxanim.study_families.data` and recorded at
14 checkpoints (six in the first 80 updates, then after 1 to 20 passes' worth of updates).
At each checkpoint the 2,000 held-out digits are encoded and the two loss terms and four
measures of how Gaussian their cloud is are taken. The same network is trained without
the KL term and without code noise (a plain autoencoder: the same initial weights and the
same batches), and with the KL term weighted by 0.1 and by 10. Three seeds of each.

``dec-centroids-training``: the 2-number autoencoder of the generative-families set
(:func:`voxanim.study_families.train_ae`, so the same weights), k-means on its training
codes recorded iteration by iteration, then DEC and IDEC from those centres recorded at
11 checkpoints with the codes of the same 2,000 held-out digits, the centres, and what
they do to label purity (accuracy, NMI, confusion) and to separation (silhouette). Three
seeds, four more k-means starts of the drawn autoencoder, and the same with 10 numbers.
"""

from __future__ import annotations

import copy
import math
import os

import numpy as np

from . import autoencoder as AE
from . import deep_models as dm
from . import mnist
from . import study_families as F
from .cache import DEFAULT_CACHE, cached_npz, settings_key, sha256_of
from .next_vision import bars, curve, matrix, points
from .optim import Adam
from .study import Study
from .study_generative import parallel

VERSION = 1
SEEDS = F.SEEDS
DRAWN = SEEDS[0]                       # the seed whose run is drawn: fixed before any run
TRACK = F.TRACK                        # the held-out digit followed (a 9), as in 101-105
CODE = 2

# ====================================================================== 170: the KL term
EPOCH = F.TRAIN_N // F.BATCH           # 156 updates see as many digits as the training set has
EARLY = (0, 5, 10, 20, 40, 80)
PASSES = (1, 2, 3, 5, 8, 12, 16, 20)
CHECKS = EARLY + tuple(EPOCH * p for p in PASSES)
STEPS = CHECKS[-1]
# The runs: a plain autoencoder (no KL term, no noise on the code), then the VAE at three
# weights of the KL term. "vae" (weight 1) is the model the animation follows.
RUNS = ("ae", "low", "vae", "high")
BETA = {"ae": 0.0, "low": 0.1, "vae": 1.0, "high": 10.0}
NOISY = {"ae": 0.0, "low": 1.0, "vae": 1.0, "high": 1.0}
EVAL_NOISE_SEED = 77                   # the one draw per held-out digit used for "a sample"
RING_SHARE = tuple(1.0 - math.exp(-0.5 * r * r) for r in (1, 2, 3))   # a 2-D standard bell
_erf = np.frompyfunc(math.erf, 1, 1)


def phi(v):
    """The standard bell's cumulative distribution."""
    return 0.5 * (1.0 + (_erf(np.asarray(v, float) / math.sqrt(2.0))).astype(float))


def kl_pieces(mu, sd):
    """The KL of N(mu, sd^2) from N(0, 1), as its two pieces ``(mu^2 / 2,
    (sd^2 - ln sd^2 - 1) / 2)``: the pull to the middle and the pull of the width to 1."""
    mu, var = np.asarray(mu, float), np.asarray(sd, float) ** 2
    return 0.5 * mu ** 2, 0.5 * (var - np.log(var) - 1.0)


def kl_only_grads(P, f):
    """Gradients of the batch-mean KL term alone (it reaches only the encoder)."""
    n = len(f["x"])
    g = {k: np.zeros_like(v) for k, v in P.items()}
    dmu, dlv = f["mu"] / n, 0.5 * (np.exp(f["lv"]) - 1.0) / n
    g["Wmu"], g["bmu"] = f["h1"].T @ dmu, dmu.sum(0)
    g["Wlv"], g["blv"] = f["h1"].T @ dlv, dlv.sum(0)
    d = (dmu @ P["Wmu"].T + dlv @ P["Wlv"].T) * (f["h1"] > 0)
    g["We1"], g["be1"] = f["h0"].T @ d, d.sum(0)
    d = (d @ P["We1"].T) * (f["h0"] > 0)
    g["We0"], g["be0"] = f["x"].T @ d, d.sum(0)
    return g


def beta_grads(P, f, beta):
    """Gradients of ``reconstruction + beta * KL``: :func:`deep_models.vae_backward`
    (which is beta = 1) plus ``beta - 1`` times the KL term's own."""
    g = dm.vae_backward(P, f)
    if beta != 1.0:
        extra = kl_only_grads(P, f)
        g = {k: g[k] + (beta - 1.0) * extra[k] for k in g}
    return g


def gaussianity(z):
    """How like a standard bell a cloud of 2-number codes is:

    ``fit``    the KL of the best-fit Gaussian (the cloud's mean and covariance) from
               N(0, I): 0 when the mean is 0 and the covariance the identity;
    ``ks``     per code number, the Kolmogorov-Smirnov distance (largest gap between the
               cloud's cumulative share and the bell's) from N(0, 1);
    ``shape``  the same after the cloud is centred and scaled to unit covariance: what is
               left when position and size no longer count;
    ``rings``  the share of codes within 1, 2 and 3 of the origin (:data:`RING_SHARE`);
    ``mean``, ``cov`` the cloud's own.
    """
    z = np.asarray(z, float)
    n, L = z.shape
    m, S = z.mean(0), np.cov(z.T)
    fit = 0.5 * (np.trace(S) + m @ m - L - np.log(np.linalg.det(S)))
    white = np.linalg.solve(np.linalg.cholesky(S), (z - m).T).T

    def ks(v):
        c = phi(np.sort(v))
        k = np.arange(1, n + 1)
        return float(max((k / n - c).max(), (c - (k - 1) / n).max()))

    r = np.sqrt((z ** 2).sum(1))
    return dict(fit=float(fit), ks=np.array([ks(z[:, j]) for j in range(L)]),
                shape=np.array([ks(white[:, j]) for j in range(L)]),
                rings=np.array([(r <= k).mean() for k in (1, 2, 3)]), mean=m, cov=S)


def vae_job(run, seed):
    """One training run of the 2-number network, recorded at :data:`CHECKS`."""
    d = F.data()
    x, xt = d["x"], d["xt"]
    beta, noisy = BETA[run], NOISY[run]
    rng = np.random.default_rng([seed, CODE, 1])         # as study_families.train_vae
    P = dm.vae_init(CODE, rng)
    opt = Adam(P, F.LR)
    eps_eval = np.random.default_rng(EVAL_NOISE_SEED).normal(size=(len(xt), CODE))
    keep = {k: [] for k in ("mu", "lv", "bce", "kl", "mse", "track")}
    at = set(CHECKS)
    for it in range(STEPS + 1):
        if it in at:
            f = dm.vae_forward(P, xt, eps_eval * noisy)   # one draw per digit, as trained
            r = np.clip(f["recon"], 1e-7, 1 - 1e-7)
            keep["bce"].append(-(xt * np.log(r) + (1 - xt) * np.log(1 - r)).sum(1).mean())
            keep["kl"].append(sum(kl_pieces(f["mu"], np.exp(0.5 * f["lv"]))).mean(0))
            keep["mu"].append(f["mu"].astype(np.float32))
            keep["lv"].append(f["lv"].astype(np.float32))
            mean = dm.vae_forward(P, xt, np.zeros_like(eps_eval))["recon"]
            keep["mse"].append(np.mean((mean - xt) ** 2))
            keep["track"].append(mean[TRACK].astype(np.float32))
        if it == STEPS:
            break
        xb = x[rng.integers(0, len(x), F.BATCH)]
        eps = rng.normal(size=(F.BATCH, CODE)) * noisy    # drawn either way: same batches
        opt.step(P, beta_grads(P, dm.vae_forward(P, xb, eps), beta))
    out = {k: np.array(v) for k, v in keep.items()}
    out["sharp"] = np.array((F.edge(mean) / F.edge(xt)).mean())
    out["shown"] = mean[:F.SHOWN].astype(np.float32)
    if seed == DRAWN:
        out.update({"P_" + k: np.asarray(v, np.float32) for k, v in P.items()})
    return out


KL_SETTINGS = dict(version=VERSION, train_n=F.TRAIN_N, test_n=F.TEST_N, seeds=list(SEEDS),
                   code=CODE, checks=list(CHECKS), batch=F.BATCH, lr=F.LR, runs=list(RUNS),
                   beta=BETA, noisy=NOISY, eval_noise_seed=EVAL_NOISE_SEED, track=TRACK,
                   widths="784-256-64-(mu, log var)-64-256-784")


def kl_arrays():
    """Every run of animation 170: ``<run>_<array>`` stacked over seeds (then
    checkpoints), the measures of the cloud of means (``g_``) and of one sample per
    digit (``s_``), and the drawn seed's final weights (``<run>_P_<name>``)."""
    def build():
        jobs = [(run, seed) for run in RUNS for seed in SEEDS]
        got = dict(zip(jobs, parallel(vae_job, jobs)))
        eps = np.random.default_rng(EVAL_NOISE_SEED).normal(size=(F.TEST_N, CODE))
        out = {}
        for run in RUNS:
            for k in ("mu", "lv", "bce", "kl", "mse", "track", "sharp", "shown"):
                out["%s_%s" % (run, k)] = np.array([got[run, s][k] for s in SEEDS])
            for k, v in got[run, DRAWN].items():
                if k.startswith("P_"):
                    out["%s_%s" % (run, k)] = v
            mu = out[run + "_mu"].astype(float)
            sd = np.exp(0.5 * out[run + "_lv"].astype(float))
            for tag, cloud in (("g", mu), ("s", mu + sd * eps * NOISY[run])):
                g = [[gaussianity(z) for z in per_seed] for per_seed in cloud]
                for k in ("fit", "ks", "shape", "rings", "mean", "cov"):
                    out["%s_%s_%s" % (run, tag, k)] = np.array([[c[k] for c in s] for s in g])
        d = F.data()
        out["labels"] = d["yt"]
        out["train_x_sha256"] = np.array(d["train_x_sha256"])
        out["test_x_sha256"] = np.array(d["test_x_sha256"])
        return out
    return cached_npz("study_kl_vae", KL_SETTINGS, build)


def kl_params(run):
    """The drawn seed's final weights of ``run`` (float64, :mod:`deep_models` names)."""
    d = kl_arrays()
    head = run + "_P_"
    return {k[len(head):]: np.asarray(v, float) for k, v in d.items() if k.startswith(head)}


SCAN_CENTRE = tuple(round(0.05 * k, 2) for k in range(31))       # blob centres times 0 .. 1.5
SCAN_WIDTH = tuple(float(v) for v in np.round(2.0 ** np.linspace(-1, 3, 17), 4))  # widths x


def pull_scan():
    """The trained VAE's two loss terms on the held-out digits when every blob's centre
    is scaled towards the middle (:data:`SCAN_CENTRE`) and when every blob's width is
    scaled (:data:`SCAN_WIDTH`): ``centre`` and ``width``, rows of ``(reconstruction,
    KL)`` in nats per digit, with the study's one noise draw per digit."""
    def build():
        P, x = kl_params("vae"), F.data()["xt"]
        eps = np.random.default_rng(EVAL_NOISE_SEED).normal(size=(len(x), CODE))
        f = dm.vae_forward(P, x, np.zeros_like(eps))
        mu, sd = f["mu"], np.exp(0.5 * f["lv"])

        def terms(centre, width):
            r = np.clip(dm.vae_decode(P, centre * mu + width * sd * eps)[2], 1e-7, 1 - 1e-7)
            bce = -(x * np.log(r) + (1 - x) * np.log(1 - r)).sum(1).mean()
            return bce, sum(kl_pieces(centre * mu, width * sd)).sum(1).mean()

        return dict(centre=np.array([terms(s, 1.0) for s in SCAN_CENTRE]),
                    width=np.array([terms(1.0, w) for w in SCAN_WIDTH]))
    return cached_npz("study_kl_scan", dict(KL_SETTINGS, centre=list(SCAN_CENTRE),
                                            width=list(SCAN_WIDTH)), build)


def _meta(path, **metadata):
    d = F.data()
    return dict(
        source="NumPy measurements on MNIST", dataset="MNIST", review_only=True, new=True,
        seeds=list(SEEDS), drawn_seed=DRAWN, artifact=path, artifact_sha256=sha256_of(path),
        evidence=dict(
            measured="models trained here in numpy on MNIST; every number is taken on the "
                     "2,000 held-out digits, three seeds, the drawn seed fixed in advance",
            exact="formulas evaluated on those measured codes (the Gaussian KL; DEC's q, p "
                  "and KL for single codes)"),
        data=dict(source_url=mnist.URL, raw_sha256=mnist.SHA256,
                  train_x_sha256=d["train_x_sha256"], test_x_sha256=d["test_x_sha256"],
                  splits="training: realdata.mnist_split(%d, 'train', 28); held out: "
                         "realdata.mnist_split(%d, 'test', 28), the official test partition"
                         % (F.TRAIN_N, F.TEST_N)), **metadata)


def _file(name, settings):
    return os.path.abspath(os.path.join(DEFAULT_CACHE, "%s-%s.npz" % (
        name, settings_key(settings))))


def vae_kl_term():
    d = kl_arrays()
    steps = np.array(CHECKS, float)

    def seeds(key):
        return np.asarray(d[key], float)

    def stat(key, at=-1):
        return F.stat(seeds(key)[:, at] if seeds(key).ndim > 1 else seeds(key))

    kl_total = {run: seeds(run + "_kl").sum(2) for run in RUNS}       # seed, checkpoint
    table = {run: dict(
        beta=BETA[run], reconstruction=stat(run + "_bce"), kl=F.stat(kl_total[run][:, -1]),
        mse=stat(run + "_mse"), sharpness=stat(run + "_sharp"),
        fit=stat(run + "_g_fit"), ks=F.stat(seeds(run + "_g_ks")[:, -1].mean(1)),
        shape=F.stat(seeds(run + "_g_shape")[:, -1].mean(1)),
        rings=seeds(run + "_g_rings")[:, -1].mean(0).tolist(),
        sample_fit=stat(run + "_s_fit"),
        sample_ks=F.stat(seeds(run + "_s_ks")[:, -1].mean(1)),
        sample_shape=F.stat(seeds(run + "_s_shape")[:, -1].mean(1)),
        sample_rings=seeds(run + "_s_rings")[:, -1].mean(0).tolist(),
        spread=F.stat(np.exp(0.5 * seeds(run + "_lv")[:, -1]).mean((1, 2))),
    ) for run in RUNS}
    drawn = SEEDS.index(DRAWN)
    scan = pull_scan()
    return Study(
        "Vae kl term",
        "A 2-number VAE on MNIST followed through training: its two loss terms and how "
        "Gaussian the cloud of 2,000 held-out codes is at 14 checkpoints, against the same "
        "network without the KL term and with it weighted 0.1 and 10; three seeds",
        (
            points("Held-out digits in the trained VAE's code space",
                   "Seed %d after %d updates: the centre of each of 2,000 held-out digits' "
                   "blobs, coloured by digit." % (DRAWN, STEPS),
                   d["vae_mu"][drawn, -1], d["labels"]),
            curve("The two loss terms through training",
                  "Held-out reconstruction term (cross-entropy summed over pixels, one code "
                  "draw per digit) and KL term, nats per digit, mean of three seeds, at "
                  "each checkpoint (updates).",
                  np.array([seeds("vae_bce").mean(0), kl_total["vae"].mean(0)]),
                  ["reconstruction", "KL"], steps),
            curve("How Gaussian the cloud of codes is",
                  "KL of the best-fit Gaussian of the 2,000 held-out code centres from "
                  "N(0, I), mean of three seeds: the VAE and the same network without a KL "
                  "term.",
                  np.array([seeds("vae_g_fit").mean(0), seeds("ae_g_fit").mean(0)]),
                  ["VAE", "no KL term"], steps),
            bars("The KL term at four weights",
                 "Final held-out KL term (nats per digit), mean of three seeds: no KL term "
                 "(measured only), weight 0.1, 1 and 10.",
                 np.array([table[run]["kl"]["mean"] for run in RUNS]),
                 ["no KL", "0.1", "1", "10"]),
        ),
        _meta(_file("study_kl_vae", KL_SETTINGS), run_settings=KL_SETTINGS,
              checks=list(CHECKS), updates_per_pass=EPOCH, table=table,
              ring_share_of_a_bell=list(RING_SHARE),
              pull_scan=dict(centre_scale=list(SCAN_CENTRE), width_scale=list(SCAN_WIDTH),
                             centre=scan["centre"].tolist(), width=scan["width"].tolist(),
                             note="drawn seed's trained VAE, held-out digits: (reconstruction, "
                                  "KL) with every blob centre, or width, scaled"),
              vae_curves=dict(
                  reconstruction=seeds("vae_bce").mean(0).tolist(),
                  kl=kl_total["vae"].mean(0).tolist(),
                  kl_per_number=seeds("vae_kl").mean(0).tolist(),
                  fit=seeds("vae_g_fit").mean(0).tolist(),
                  ae_fit=seeds("ae_g_fit").mean(0).tolist()),
              measures=dict(
                  reconstruction="binary cross-entropy summed over the 784 pixels, one "
                                 "fixed noise draw per held-out digit (the autoencoder: no "
                                 "noise)",
                  kl="sum over the two code numbers of (mu^2 + sigma^2 - ln sigma^2 - 1)/2, "
                     "mean over held-out digits",
                  fit="KL of the Gaussian with the cloud's mean and covariance from N(0, I)",
                  ks="Kolmogorov-Smirnov distance of each code number's marginal from "
                     "N(0, 1), mean of the two",
                  shape="the same after centring and whitening the cloud",
                  rings="share of codes within 1, 2, 3 of the origin; a 2-D standard bell: "
                        "%.3f, %.3f, %.3f" % RING_SHARE,
                  cloud="g_*: the blob centres (what is drawn); s_*: one draw per digit "
                        "from its blob (the aggregate posterior)"),
              limitations=[
                  "one small dense architecture and a 2-number code, chosen so the code "
                  "space can be drawn exactly; wider codes behave differently (the KL term "
                  "can switch code numbers off)",
                  "an 'epoch' here is %d updates of %d digits drawn with replacement, not a "
                  "shuffled pass" % (EPOCH, F.BATCH),
                  "the KS distance is reported as a distance, not as a test: with 2,000 "
                  "points even a small departure from the bell would be 'significant'"],
              designs_tried=[
                  "KL weights 0.1, 1 and 10 and the checkpoint spacing were fixed before "
                  "the runs were looked at; nothing was retuned",
                  "the cloud is measured two ways (blob centres, and one draw per digit) "
                  "because the centres alone understate how much of the bell the blobs "
                  "cover; both are reported",
                  "the tug of war was first drawn as two gradients on each digit class's "
                  "blob centres (the KL term's, towards the middle, against the "
                  "reconstruction term's): on held-out digits the two were unrelated in "
                  "direction (mean cosine -0.02), a single digit's gradient being mostly "
                  "its own misfit. It was replaced by pull_scan: both terms with every blob "
                  "slid to the middle, or widened, whose sum is lowest as trained"]))


# ====================================================================== 171: DEC's centres
K = F.CLUSTERS
KM_ITERS = dm.KMEANS_ITERS
DEC_CHECKS = (0, 10, 20, 40, 80, 150, 300, 500, 800, 1100, F.CLUSTER_STEPS)
DEC_CODES = (2, 10)                    # the drawn one, and a second run with more numbers
EXTRA_STARTS = 4                       # further k-means starts of the drawn autoencoder
KINDS = ("dec", "idec")
MEASURES = ("acc", "nmi", "maxq", "sil", "within", "gap", "mse", "moved", "kl")


def kmeans_trace(z, k, rng, iters=KM_ITERS):
    """:func:`deep_models.kmeans` with every iteration kept: the centres after 0 ..
    ``iters`` iterations ``(iters + 1, k, L)`` and the mean squared distance of a code to
    its nearest centre at each."""
    c = z[rng.choice(len(z), k, replace=False)]
    trace, within = [c], []
    for _ in range(iters):
        d2 = ((z[:, None] - c[None]) ** 2).sum(-1)
        lab = d2.argmin(1)
        within.append(d2.min(1).mean())
        c = np.array([z[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        trace.append(c)
    within.append(((z[:, None] - c[None]) ** 2).sum(-1).min(1).mean())
    return np.array(trace), np.array(within)


def nmi(a, b):
    """Normalised mutual information of two labellings: I(a; b) over the mean of the
    two entropies (0: independent, 1: the same partition)."""
    a, b = np.asarray(a), np.asarray(b)
    joint = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(joint, (a, b), 1.0)
    joint /= joint.sum()
    pa, pb = joint.sum(1), joint.sum(0)

    def h(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    nz = joint > 0
    mutual = float((joint[nz] * np.log(joint[nz] / np.outer(pa, pb)[nz])).sum())
    return mutual / max(0.5 * (h(pa) + h(pb)), 1e-12)


def silhouette(z, lab, k=K):
    """Mean silhouette of codes ``z`` under clusters ``lab``: per code, (b - a) /
    max(a, b) with a its mean distance to its own cluster and b to the nearest other
    one. 1: tight, far-apart clusters; 0: overlapping. Needs no labels."""
    z = np.asarray(z, float)
    sq = (z ** 2).sum(1)
    dist = np.sqrt(np.maximum(sq[:, None] - 2 * z @ z.T + sq[None], 0.0))
    onehot = np.eye(k)[lab]
    count = onehot.sum(0)
    total = dist @ onehot                                # code, cluster
    own = count[lab]
    rows = np.arange(len(z))
    a = total[rows, lab] / np.maximum(own - 1, 1)
    other = total / np.maximum(count, 1)
    other[:, count == 0] = np.inf
    other[rows, lab] = np.inf
    b = other.min(1)
    s = np.where(own > 1, (b - a) / np.maximum(np.maximum(a, b), 1e-12), 0.0)
    return float(s.mean())


def confusion(lab, y, k=K):
    """``(cluster, digit)`` counts."""
    M = np.zeros((k, 10), int)
    np.add.at(M, (lab, y), 1)
    return M


def cluster_measures(z, mu, y):
    """Label purity and separation of held-out codes ``z`` around centres ``mu``."""
    q, _k = dm.soft_assign(z, mu)
    lab = q.argmax(1)
    d = np.sqrt(((mu[:, None] - mu[None]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return dict(acc=dm.cluster_accuracy(lab, y)[0], nmi=nmi(lab, y), maxq=q.max(1).mean(),
                sil=silhouette(z, lab),
                within=np.sqrt(((z - mu[lab]) ** 2).sum(1)).mean(), gap=d.min(1).mean(),
                kl=dm.kl_loss(dm.target(q), q)), lab


def cluster_run(kind, P0, centres, x, xt, yt, rng):
    """DEC / IDEC from autoencoder ``P0`` and k-means ``centres``, as
    :func:`study_families.train_cluster` trains it (the same updates from the same
    generator state), recorded at :data:`DEC_CHECKS` on the held-out digits."""
    P = {k: np.array(v) for k, v in P0.items()}
    P["mu"] = np.array(centres)
    opt = Adam(P, F.LR)
    at = set(DEC_CHECKS)
    keep = {k: [] for k in MEASURES + ("z", "mu", "conf")}
    p_all = freq0 = None
    for it in range(F.CLUSTER_STEPS + 1):
        if it in at:
            z = F.encode(P, xt)
            m, lab = cluster_measures(z, P["mu"], yt)
            m["mse"] = np.mean((F.decode(P, z) - xt) ** 2)
            m["moved"] = np.sqrt(((P["mu"] - centres) ** 2).sum(1)).mean()
            for k in MEASURES:
                keep[k].append(m[k])
            keep["z"].append(z.astype(np.float32))
            keep["mu"].append(P["mu"].copy())
            keep["conf"].append(confusion(lab, yt))
        if it == F.CLUSTER_STEPS:
            break
        if it % F.UPDATE_P == 0:
            q_all = dm.soft_assign(F.encode(P, x), P["mu"])[0]
            p_all = dm.target(q_all)
            if freq0 is None:
                freq0 = q_all.sum(0)
        idx = rng.integers(0, len(x), F.CLUSTER_BATCH)
        xb = x[idx]
        acts, _pre = AE.dense_layers(P, xb, AE.ENCODER)
        z = acts[-1]
        q, kk = dm.soft_assign(z, P["mu"])
        dz, dmu = dm.kl_grads(z, P["mu"], p_all[idx], q, kk)
        g = {}
        if kind == "idec":
            dz, dmu = dz * F.GAMMA, dmu * F.GAMMA
            dec, _pre = AE.dense_layers(P, z, AE.DECODER)
            dz = dz + AE.dense_backward(P, dec, AE.recon_grad(dec[-1], xb), AE.DECODER, g)
        AE.dense_backward(P, acts, dz, AE.ENCODER, g)
        g["mu"] = dmu
        for k in P:
            g.setdefault(k, np.zeros_like(P[k]))
        opt.step(P, g)
    out = {k: np.array(v) for k, v in keep.items()}
    out["freq0"] = freq0
    return out


def dec_job(code, seed):
    """One autoencoder (the generative-families one of this code size and seed), its
    k-means and its DEC and IDEC runs; for the drawn 2-number one, four more starts."""
    d = F.data()
    x, xt, yt = d["x"], d["xt"], d["yt"]
    ae, _trace = F.train_ae(code, seed, x, xt[:F.EVAL_N], xt[:F.SHOWN])
    P0 = {k: np.asarray(v, np.float32).astype(float) for k, v in ae.items()}   # as saved there
    z0 = F.encode(P0, x)
    out = {}
    starts = range(1 + (EXTRA_STARTS if (code == CODE and seed == DRAWN) else 0))
    for start in starts:
        rng = np.random.default_rng([seed, code, 3] + ([start] if start else []))
        centres, within = kmeans_trace(z0, K, rng)
        zt = F.encode(P0, xt)
        head = "start%d_" % start
        out[head + "km"], out[head + "km_within"] = centres, within
        out[head + "km_acc"] = np.array([
            dm.cluster_accuracy(((zt[:, None] - c[None]) ** 2).sum(-1).argmin(1), yt)[0]
            for c in centres])
        for kind in (KINDS if start == 0 else KINDS[:1]):
            run = cluster_run(kind, P0, centres[-1], x, xt, yt, copy.deepcopy(rng))
            for k, v in run.items():
                out["%s%s_%s" % (head, kind, k)] = v
    out["scale"] = np.array(np.sqrt(((z0 - z0.mean(0)) ** 2).sum(1).mean()))
    # a fixed 2-D view of a wider code: the two widest directions of the starting codes
    vals, vecs = np.linalg.eigh(np.cov(z0.T))
    out["view_mean"], out["view_axes"] = z0.mean(0), vecs[:, np.argsort(vals)[::-1][:2]]
    out["view_share"] = np.array(np.sort(vals)[::-1][:2].sum() / vals.sum())
    return out


DEC_SETTINGS = dict(version=VERSION, train_n=F.TRAIN_N, test_n=F.TEST_N, seeds=list(SEEDS),
                    codes=list(DEC_CODES), ae_steps=F.STEPS, batch=F.BATCH, lr=F.LR,
                    cluster=dict(steps=F.CLUSTER_STEPS, batch=F.CLUSTER_BATCH, gamma=F.GAMMA,
                                 update_p=F.UPDATE_P, clusters=K),
                    kmeans_iters=KM_ITERS, checks=list(DEC_CHECKS), extra_starts=EXTRA_STARTS,
                    widths="784-256-64-code-64-256-784")


def dec_arrays():
    """Every run of animation 171, keys ``c<code>_s<seed>_start<n>_<kind>_<array>``."""
    def build():
        jobs = [(code, seed) for code in DEC_CODES for seed in SEEDS]
        out = {}
        for (code, seed), got in zip(jobs, parallel(dec_job, jobs)):
            for k, v in got.items():
                out["c%d_s%d_%s" % (code, seed, k)] = v
        d = F.data()
        out["labels"] = d["yt"]
        out["train_x_sha256"] = np.array(d["train_x_sha256"])
        out["test_x_sha256"] = np.array(d["test_x_sha256"])
        return out
    return cached_npz("study_kl_dec", DEC_SETTINGS, build)


def dec_run(kind="dec", code=CODE, seed=DRAWN, start=0):
    """One recorded run as a dict of float arrays (``z``, ``mu``, the measures, ...)."""
    d = dec_arrays()
    head = "c%d_s%d_start%d_%s_" % (code, seed, start, kind)
    return {k[len(head):]: np.asarray(v, float) for k, v in d.items() if k.startswith(head)}


def dec_start(code=CODE, seed=DRAWN, start=0):
    """The k-means trace of one start: ``km``, ``km_within``, ``km_acc``; and the
    autoencoder's ``scale`` and the fixed 2-D ``view_*`` of its codes."""
    d = dec_arrays()
    head = "c%d_s%d_" % (code, seed)
    out = {k: np.asarray(d[head + "start%d_%s" % (start, k)], float)
           for k in ("km", "km_within", "km_acc")}
    out.update({k: np.asarray(d[head + k], float)
                for k in ("scale", "view_mean", "view_axes", "view_share")})
    return out


def shared_digits(conf, least=0.25):
    """Which digits share clusters: per cluster of a ``(cluster, digit)`` count matrix,
    the digits that each make up at least ``least`` of it, largest first."""
    out = []
    for row in np.asarray(conf, float):
        share = row / max(row.sum(), 1.0)
        out.append([int(j) for j in np.argsort(-share) if share[j] >= least])
    return out


def dec_centroids_training():
    d = dec_arrays()
    steps = np.array(DEC_CHECKS, float)

    def seeds(kind, key, code=CODE):
        return np.array([dec_run(kind, code, s)[key] for s in SEEDS])

    def ends(kind, key, code=CODE):
        v = seeds(kind, key, code)
        return dict(start=F.stat(v[:, 0]), end=F.stat(v[:, -1]))

    table = {"c%d" % code: {kind: {k: ends(kind, k, code) for k in MEASURES}
                            for kind in KINDS} for code in DEC_CODES}
    starts = np.array([dec_run("dec", CODE, DRAWN, s)["acc"][[0, -1]]
                       for s in range(1 + EXTRA_STARTS)])
    starts_sil = np.array([dec_run("dec", CODE, DRAWN, s)["sil"][[0, -1]]
                           for s in range(1 + EXTRA_STARTS)])
    run = dec_run("dec")
    conf = run["conf"][-1]
    return Study(
        "Dec centroids training",
        "K-means centres in a 2-number autoencoder's code space, then DEC and IDEC moving "
        "the codes and the centres: label purity and separation of 2,000 held-out digits "
        "at 11 checkpoints; three seeds, five k-means starts, and a 10-number code",
        (
            points("Held-out digits after DEC",
                   "Seed %d, 2-number code, after %d DEC updates: 2,000 held-out digits, "
                   "coloured by digit." % (DRAWN, F.CLUSTER_STEPS),
                   run["z"][-1], d["labels"]),
            curve("Label purity through DEC training",
                  "Cluster accuracy (best one-to-one matching of clusters to digits) and "
                  "normalised mutual information of 2,000 held-out digits, 2-number code, "
                  "mean of three seeds, at each checkpoint (updates).",
                  np.array([seeds("dec", "acc").mean(0), seeds("dec", "nmi").mean(0)]),
                  ["accuracy", "NMI"], steps),
            curve("Separation through DEC training",
                  "Mean silhouette of the held-out codes under their clusters (needs no "
                  "labels) and mean largest soft assignment, mean of three seeds.",
                  np.array([seeds("dec", "sil").mean(0), seeds("dec", "maxq").mean(0)]),
                  ["silhouette", "mean max q"], steps),
            matrix("Which digits share a cluster",
                   "Seed %d after DEC: rows are the ten clusters, columns the digits 0 to "
                   "9, entries counts of the 2,000 held-out digits." % DRAWN, conf),
        ),
        _meta(_file("study_kl_dec", DEC_SETTINGS), run_settings=DEC_SETTINGS,
              checks=list(DEC_CHECKS), codes=list(DEC_CODES), table=table,
              starts=dict(accuracy_start=starts[:, 0].tolist(),
                          accuracy_end=starts[:, 1].tolist(),
                          silhouette_end=starts_sil[:, 1].tolist(),
                          note="five k-means starts of the drawn autoencoder (seed %d); "
                               "start 0 is the one drawn" % DRAWN),
              kmeans_accuracy={"c%d" % code: F.stat(
                  [dec_start(code, s)["km_acc"][-1] for s in SEEDS]) for code in DEC_CODES},
              shared_digits=shared_digits(conf),
              view_share={"c%d" % code: float(dec_start(code)["view_share"])
                          for code in DEC_CODES},
              measures=dict(
                  acc="share of held-out digits whose cluster is matched to their digit "
                      "under the best one-to-one matching (Hungarian method, numpy)",
                  nmi="mutual information of cluster and digit over the mean of their "
                      "entropies",
                  maxq="mean over held-out digits of the largest soft assignment",
                  sil="mean silhouette of the held-out codes under their nearest centre",
                  moved="mean distance of the ten centres from their k-means start, in "
                        "code units ('scale' is the RMS spread of the starting codes)",
                  mse="mean squared error of the decoder on the held-out digits; DEC's "
                      "decoder is the autoencoder's, untouched"),
              limitations=[
                  "one small dense autoencoder, 4,000 updates of pretraining and 1,500 of "
                  "DEC: the published DEC uses a much wider stacked autoencoder and a "
                  "10-number code and reaches far higher accuracy",
                  "the 2-number code is drawn exactly; the 10-number one through a fixed "
                  "projection onto the two widest directions of its starting codes, which "
                  "hides separation along the other eight",
                  "silhouette is measured in each model's own code space, which DEC is "
                  "free to reshape",
                  "the drawn run (seed %d, fixed in advance) is the least accurate of the "
                  "three seeds" % DRAWN],
              designs_tried=[
                  "the autoencoder, k-means and DEC settings are those of the "
                  "generative-families study, unchanged; the checkpoints, the measures and "
                  "the 10-number second run were fixed before the runs were looked at",
                  "the 10-number run is reported as numbers only: its accuracy is not "
                  "clearly better than the 2-number one's, so it is not drawn",
                  "the three codes of the q-and-p picture were first the held-out codes "
                  "whose largest q is nearest 0.36, 0.6 and 0.9 whatever their digit; all "
                  "three were 1s at the edge of the picture, so the rule became a 4, a 3 "
                  "and a 0 with those q"]))


STUDIES = {"vae-kl-term": vae_kl_term, "dec-centroids-training": dec_centroids_training}

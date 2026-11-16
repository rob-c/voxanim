"""Redesigned experiments for four of the classical-ML animations (67, 71, 72, 73).

The first versions (:mod:`voxanim.next_vision`) could not show their concept:

* ``bias-variance`` predicted right-half ink from left-half ink, a relation so nearly
  straight that no polynomial degree had any bias to lose (the mean fit's error stayed at
  0.0006 from degree 1 to 6): only the variance half of the trade-off existed.
* ``gaussian-mixture-em`` compared two starts on eight heavily overlapping digit classes
  (label silhouette 0.02): the likelihood surface had no separate basins, the two starts
  ended 0.003 nats apart, and two starts could not have shown a distribution anyway.
* ``curse-of-dimensionality`` added *PCA* coordinates, every one of which carries label
  signal, so nearest-neighbour accuracy rose with dimension: signal and dimension were
  confounded.
* ``tsne-and-umap`` fitted no UMAP.

Each is run again here with a design that gives the concept a fair chance. Every function
ending ``_data`` is the experiment (seeded, plain numpy, cached); the function registered in
:data:`STUDIES` wraps its numbers as a four-panel :class:`~voxanim.study.Study`.
:mod:`voxanim.mech_classical` draws from both.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from . import mnist
from . import next_vision as nv
from .cache import cached_npz
from .next_vision_models import downsample
from .study import Panel, Study

VERSION = 1
SEED = 51


def _cached(name, build, **settings):
    return cached_npz("study_classical_" + name,
                      dict(version=VERSION, seed=SEED, mnist_sha256=mnist.SHA256, **settings),
                      lambda: {k: np.asarray(v) for k, v in build().items()})


def _meta(**extra):
    return dict(source="NumPy measurements on MNIST", dataset="MNIST",
                dataset_sha256=mnist.SHA256, dataset_url=mnist.URL, review_only=True, seed=SEED,
                study_version=VERSION, **extra)


# ====================================================================== 67 bias-variance
BV = dict(pool=5000, sample_size=40, resamples=200, degrees=list(range(1, 11)), fine=4)


def column_ink(images):
    """``[image, column]``: the share of each of an image's 28 columns that is inked."""
    return np.asarray(images, float).reshape(-1, 28, 28).mean(1)


def fit_columns(columns, inks, degree):
    """The least-squares polynomial of ``degree`` through (column, ink) pairs."""
    return np.polynomial.Polynomial.fit(np.asarray(columns, float), inks, degree, domain=[0, 27])


@lru_cache(maxsize=1)
def bias_variance_data():
    """Polynomials of degree 1 to 10 fitted to 200 resamples of 40 measurements each.

    One measurement: pick a training image and one of its 28 columns, and measure the share
    of that column that is inked. Input: the column's position. Target: the ink. The mean
    target is a hump (no ink at the edges, most in the middle), which no straight line fits.
    The fits are judged on every column of all 10,000 test images, whose own per-column mean
    and variance split the squared error exactly into bias, variance and noise."""
    def build():
        x, _, xt, _ = nv.mnist_data()
        pool, test = column_ink(x[:BV["pool"]]), column_ink(xt)
        rng = np.random.default_rng(SEED)
        fine = np.linspace(0, 27, 27 * BV["fine"] + 1)
        ids, curves = [], []
        for degree in BV["degrees"]:
            for _ in range(BV["resamples"]):
                image = rng.integers(0, len(pool), BV["sample_size"])
                column = rng.integers(0, 28, BV["sample_size"])
                curves.append(fit_columns(column, pool[image, column], degree)(fine))
                ids.append((image, column))
        shape = (len(BV["degrees"]), BV["resamples"])
        curves = np.array(curves).reshape(*shape, len(fine))
        ids = np.array(ids).reshape(*shape, 2, BV["sample_size"])
        pred = curves[:, :, ::BV["fine"]]                   # at the 28 columns
        truth, spread = test.mean(0), test.var(0)
        # the average squared error, counted directly over every fit, test image and column
        mse = np.array([np.mean([((p - test) ** 2).mean() for p in rows]) for rows in pred])
        bias2 = ((pred.mean(1) - truth) ** 2).mean(1)
        variance = pred.var(1).mean(1)
        noise = float(spread.mean())
        return dict(pool=pool, sampled=ids, fine=fine, curves=curves, truth=truth,
                    test_variance=spread, bias2=bias2, variance=variance, noise=noise, mse=mse,
                    closure=np.abs(mse - bias2 - variance - noise),
                    median_variance=np.median(pred.var(1), axis=1))
    return _cached("bias_variance", build, **BV)


def bias_variance():
    d = bias_variance_data()
    degrees = np.array(BV["degrees"])
    best = int(np.argmin(d["mse"]))
    return Study(
        "Bias and variance",
        "Polynomial fits to resampled measurements of where the ink is in MNIST digits",
        [
            Panel("Different samples, different fits",
                  "Eight fits at the highest degree, each to 40 resampled (column, ink) pairs.",
                  "curve", dict(x=d["fine"], ys=np.clip(d["curves"][-1, :8], -1, 2),
                                labels=["fit %d" % i for i in range(8)]),
                  ("clipped to -1..2 for display",)),
            Panel("The mean fit against the test images' mean",
                  "Mean of 200 fits at the lowest, the best and the highest degree.",
                  "curve", dict(x=np.arange(28.0),
                                ys=[d["curves"][i].mean(0)[::BV["fine"]]
                                    for i in (0, best, len(degrees) - 1)] + [d["truth"]],
                                labels=["degree %d" % degrees[0], "degree %d" % degrees[best],
                                        "degree %d" % degrees[-1], "test mean"])),
            Panel("Squared error on 10,000 test images, decomposed",
                  "Bias squared + variance + noise equals the average squared error, exactly.",
                  "curve", dict(x=degrees, ys=[d["bias2"], d["variance"],
                                               np.full(len(degrees), d["noise"]), d["mse"]],
                                labels=["bias squared", "variance", "noise", "average error"])),
            Panel("Check the identity",
                  "Directly counted error minus the three parts, per degree.",
                  "bars", dict(values=d["closure"], labels=[str(v) for v in degrees])),
        ],
        _meta(
            task="input: a column's position (0..27); target: share of that column inked, in "
                 "one randomly drawn training image",
            **BV, test_images=10000, best_degree=int(degrees[best]),
            truth="per-column mean over the 10,000 test images (so the split is exact on them)",
            caveat="bias at the highest degrees is the squared error of a mean of 200 very "
                   "variable fits, so it includes about variance/200 of sampling error",
            designs_tried=["column position -> column ink, 40 per resample, degrees 1-10 "
                           "(fixed before the decomposition was looked at; the only design run)"],
            replaces="next_vision.bias_variance: right-half ink from left-half ink, a nearly "
                     "linear relation with no bias to trade",
        ),
    )


# ====================================================================== 71 gaussian-mixture-em
GMM = dict(digits=[0, 1, 6, 7], train=1200, test=600, components=4, rounds=100, starts=300,
           covariance_floor=1e-5)
KINDS = ("random", "kmeans++", "kmeans")
KMEANS_ROUNDS = 20


def log_terms(x, mu, cov, pi):
    """``[point, component]``: log of (weight x Gaussian density)."""
    out = []
    for j in range(len(mu)):
        d = x - mu[j]
        out.append(np.log(pi[j]) - 0.5 * (
            np.einsum("ni,ij,nj->n", d, np.linalg.inv(cov[j]), d)
            + np.linalg.slogdet(cov[j])[1] + x.shape[1] * np.log(2 * np.pi)))
    return np.array(out).T


def mean_log_density(x, mu, cov, pi):
    return float(np.logaddexp.reduce(log_terms(x, mu, cov, pi), axis=1).mean())


def em_history(x, start, cov=None, pi=None, rounds=GMM["rounds"],
               floor=GMM["covariance_floor"]):
    """EM for a full-covariance Gaussian mixture from the centres ``start`` (with unit
    covariances and equal weights unless ``cov`` and ``pi`` are given), keeping every
    round: the shapes before each round and after the last (``mu``, ``cov``, ``pi``:
    ``rounds + 1`` states), the shares each round's E-step gave (``r``) and its mean
    log-likelihood (``loss``)."""
    k, dim = len(start), x.shape[1]
    mu = np.array(start, float)
    cov = np.repeat(np.eye(dim)[None], k, axis=0) if cov is None else np.array(cov, float)
    pi = np.ones(k) / k if pi is None else np.array(pi, float)
    out = dict(mu=[mu.copy()], cov=[cov.copy()], pi=[pi.copy()], r=[], loss=[])
    for _ in range(rounds):
        lp = log_terms(x, mu, cov, pi)
        mx = lp.max(1, keepdims=True)
        r = np.exp(lp - mx)
        total = r.sum(1, keepdims=True)
        r /= total
        out["loss"].append(float(np.mean(mx + np.log(total))))
        out["r"].append(r.copy())
        nk = r.sum(0)
        pi = nk / len(x)
        mu = r.T @ x / nk[:, None]
        for j in range(k):
            d = x - mu[j]
            cov[j] = (d.T * r[:, j]) @ d / nk[j] + np.eye(dim) * floor
        for key, value in (("mu", mu), ("cov", cov), ("pi", pi)):
            out[key].append(value.copy())
    return {key: np.array(value) for key, value in out.items()}


def unit(centres):
    """A start from centres alone: unit covariances, equal weights."""
    k, dim = centres.shape
    return centres, np.repeat(np.eye(dim)[None], k, axis=0), np.ones(k) / k


def random_start(x, k, rng):
    """``k`` of the points, drawn uniformly."""
    return unit(x[rng.choice(len(x), k, replace=False)].copy())


def spread_start(x, k, rng):
    """k-means++ seeding: each next centre a point drawn with probability proportional to
    its squared distance from the nearest centre already chosen."""
    centres = [x[rng.integers(len(x))]]
    for _ in range(k - 1):
        d2 = ((x[:, None] - np.array(centres)[None]) ** 2).sum(-1).min(1)
        centres.append(x[rng.choice(len(x), p=d2 / d2.sum())])
    return unit(np.array(centres))


def kmeans_start(x, k, rng, rounds=KMEANS_ROUNDS, floor=GMM["covariance_floor"]):
    """The usual library start: k-means (k-means++ seeding, then ``rounds`` rounds of
    assigning each point to its nearest centre and moving the centres), handing EM each
    cluster's centre, covariance and share of the points."""
    centres = spread_start(x, k, rng)[0]
    for _ in range(rounds):
        owner = ((x[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
        centres = np.array([x[owner == j].mean(0) if (owner == j).any() else centres[j]
                            for j in range(k)])
    owner = ((x[:, None] - centres[None]) ** 2).sum(-1).argmin(1)
    cov = np.array([np.cov(x[owner == j].T, bias=True) + np.eye(x.shape[1]) * floor
                    if (owner == j).sum() > 2 else np.eye(x.shape[1]) for j in range(k)])
    return centres, cov, np.maximum(np.bincount(owner, minlength=k), 1) / len(x)


def digit_plane(digits, n_train, n_test):
    """Training-only PCA to two coordinates of pooled 7 x 7 images of ``digits``."""
    x, y, xt, yt = nv.mnist_data()
    tr = np.flatnonzero(np.isin(y, digits))[:n_train]
    te = np.flatnonzero(np.isin(yt, digits))[:n_test]
    a, b = downsample(x[tr]), downsample(xt[te])
    mean = a.mean(0)
    axes = np.linalg.svd(a - mean, full_matrices=False)[2][:2]
    scale = float(((a - mean) @ axes.T).std(0).mean())
    return dict(train=(a - mean) @ axes.T / scale, test=(b - mean) @ axes.T / scale,
                train_labels=y[tr], test_labels=yt[te], train_indices=tr, test_indices=te)


def silhouette(z, labels, every=4):
    """Mean silhouette of the labelled groups (how distinct the clusters are)."""
    dist = np.linalg.norm(z[:, None] - z[None], axis=-1)
    groups, out = np.unique(labels), []
    for i in range(0, len(z), every):
        own = labels == labels[i]
        a = dist[i, own].sum() / max(own.sum() - 1, 1)
        b = min(dist[i, labels == g].mean() for g in groups if g != labels[i])
        out.append((b - a) / max(a, b))
    return float(np.mean(out))


@lru_cache(maxsize=1)
def gaussian_mixture_data():
    """300 starts of each kind (uniformly random points; k-means++ seeds; a k-means
    solution) of the same EM, on the same 1,200 training points; every fit scored on 600
    held-out points."""
    def build():
        data = digit_plane(GMM["digits"], GMM["train"], GMM["test"])
        x, k = data["train"], GMM["components"]
        starts, covs, pis, final, losses, train_ll, test_ll = [], [], [], [], [], [], []
        for kind, make in enumerate((random_start, spread_start, kmeans_start)):
            for s in range(GMM["starts"]):
                start = make(x, k, np.random.default_rng([SEED, kind, s]))
                run = em_history(x, *start)
                fit = run["mu"][-1], run["cov"][-1], run["pi"][-1]
                starts.append(start[0])
                covs.append(start[1])
                pis.append(start[2])
                final.append(np.concatenate([v.ravel() for v in fit]))
                losses.append(run["loss"])
                train_ll.append(mean_log_density(x, *fit))
                test_ll.append(mean_log_density(data["test"], *fit))
        shape = (len(KINDS), GMM["starts"])
        return dict(**data, starts=np.array(starts).reshape(*shape, k, 2),
                    start_cov=np.array(covs).reshape(*shape, k, 2, 2),
                    start_pi=np.array(pis).reshape(*shape, k),
                    final=np.array(final).reshape(*shape, -1),
                    losses=np.array(losses).reshape(*shape, -1),
                    train_ll=np.array(train_ll).reshape(shape),
                    test_ll=np.array(test_ll).reshape(shape),
                    silhouette=silhouette(x[:600], data["train_labels"][:600]))
    return _cached("gaussian_mixture", build, kinds=list(KINDS), kmeans_rounds=KMEANS_ROUNDS,
                   **GMM)


NEAR_BEST = 0.01                   # nats per point: "ended at the best optimum found"
CLEARLY_WORSE = 0.05               # nats per point on held-out images: "a worse fit"


def mixture_summary(d):
    """Per kind of start: the share of starts whose training log-likelihood ends within
    :data:`NEAR_BEST` of the best any start reached, the share whose held-out
    log-likelihood ends more than :data:`CLEARLY_WORSE` below that optimum's, and the best,
    median and worst held-out log-likelihood."""
    top = float(d["train_ll"].max())
    at_best = d["train_ll"] > top - NEAR_BEST
    usual = float(np.median(d["test_ll"][at_best]))      # held-out value of the best optimum
    return dict(
        reached=at_best.mean(1), usual=usual,
        bad=(d["test_ll"] < usual - CLEARLY_WORSE).mean(1),
        best=d["test_ll"].max(1), median=np.median(d["test_ll"], axis=1),
        worst=d["test_ll"].min(1),
        # the starts shown: chosen by TRAINING likelihood, as a user without test data would
        best_start=d["train_ll"].argmax(1), worst_start=d["train_ll"].argmin(1))


def gaussian_mixture():
    d = gaussian_mixture_data()
    s = mixture_summary(d)
    rounds = np.arange(1, GMM["rounds"] + 1)
    b, w = int(s["best_start"][0]), int(s["worst_start"][0])
    return Study(
        "Gaussian mixture, by EM",
        "Four-component EM from 300 starts of each of three kinds on real MNIST coordinates",
        [
            Panel("Real digit images as points",
                  "Training-only PCA of zeros, ones, sixes and sevens; EM never sees the labels.",
                  "points", dict(xy=d["train"][:400], classes=d["train_labels"][:400])),
            Panel("Log-likelihood by round: the best and the worst random start",
                  "Chosen by final training log-likelihood among 300 random starts.",
                  "curve", dict(x=rounds, ys=[d["losses"][0, b], d["losses"][0, w]],
                                labels=["best random start", "worst random start"])),
            Panel("Held-out log-likelihood of every start, sorted",
                  "Mean log density of 600 held-out images under each of the 900 fits.",
                  "curve", dict(x=np.arange(1, GMM["starts"] + 1),
                                ys=np.sort(d["test_ll"], axis=1), labels=list(KINDS))),
            Panel("Share of starts ending at the best optimum found",
                  "Within %.2f nats per point of the best training log-likelihood." % NEAR_BEST,
                  "bars", dict(values=s["reached"], labels=list(KINDS)),
                  ("worst held-out: " + ", ".join("%.3f %s" % (v, k)
                                                  for v, k in zip(s["worst"], KINDS)),)),
        ],
        _meta(
            **GMM, start_kinds=list(KINDS), kmeans_rounds=KMEANS_ROUNDS,
            label_silhouette=float(d["silhouette"]),
            reached_best=[float(v) for v in s["reached"]],
            clearly_worse_heldout=[float(v) for v in s["bad"]],
            heldout_at_best_optimum=s["usual"],
            heldout_worst=[float(v) for v in s["worst"]],
            heldout_median=[float(v) for v in s["median"]],
            heldout_best=[float(v) for v in s["best"]],
            start_seeds="numpy default_rng([51, kind, start]) for start 0..299",
            designs_tried=[
                "CHOSEN PLANE: unsupervised PCA plane of digits 0,1,6,7 (label silhouette 0.46, "
                "the most distinct unsupervised plane of those compared), 4 components. The "
                "plane was chosen by silhouette before any EM comparison",
                "LDA plane of digits 0,1,3,6,7 (silhouette 0.49, but its coordinates are fitted "
                "with labels), 5 components, 100 starts: more optima; 50 of 100 random and 62 "
                "of 100 k-means++ starts reached the best one. Not used: label-fitted plane",
                "first run on the chosen plane: 100 starts each of random and k-means++ seeds "
                "(80 and 88 reached the best optimum). Raised to 300 starts each to settle "
                "whether that gap was real: it mostly was not (259 and 267 of 300)",
                "the third kind of start (a full k-means solution, as libraries use) was added "
                "after k-means++ seeding alone was seen not to help; all three are reported",
            ],
            replaces="next_vision.gaussian_mixture: two starts, eight overlapping classes",
        ),
    )


# ====================================================================== 72 curse-of-dimensionality
CURSE = dict(test=1000, sizes=[125, 250, 500, 1000, 2000, 4000, 8000],
             blocks=[0, 1, 3, 7, 15, 31], shuffles=5, shown_size=1000, bins=24,
             pca_dimensions=[2, 4, 8, 16, 32, 49])


def shuffled_block(rng, pixels):
    """49 no-signal coordinates: each pixel column of ``pixels`` (images x 49), with its
    values dealt out again to the images at random, independently per column. Every value
    is a real pixel measurement and each column keeps its distribution; none says anything
    about the image it now belongs to."""
    return np.column_stack([pixels[rng.permutation(len(pixels)), j]
                            for j in range(pixels.shape[1])])


def squared_distances(p, q):
    return (p * p).sum(1)[:, None] - 2 * p @ q.T + (q * q).sum(1)[None]


def ball_in_box(d):
    """Exact: the share of a box's volume inside the largest ball that fits in it."""
    return math.exp(d / 2 * math.log(math.pi) - math.lgamma(d / 2 + 1) - d * math.log(2))


@lru_cache(maxsize=1)
def curse_data():
    """1-nearest-neighbour classification of 1,000 test images from 125 to 8,000 training
    images, described by their 49 real pooled pixels plus 0 to 1,519 shuffled pixel
    coordinates (five independent shuffles); and, for contrast, by 2 to 49 real PCA
    coordinates."""
    def build():
        x, y, xt, yt = nv.mnist_data()
        n = max(CURSE["sizes"])
        a, b, la, lb = downsample(x[:n]), downsample(xt[:CURSE["test"]]), y[:n], yt[:CURSE["test"]]
        base = squared_distances(b, a)
        shown = CURSE["shown_size"]
        accuracy, ratio, hist, nearest, tiles = [], [], [], [], []
        for seed in range(CURSE["shuffles"]):
            rng = np.random.default_rng([SEED, seed])
            dist, done = base.copy(), 0
            for blocks in CURSE["blocks"]:
                while done < blocks:
                    extra_test, extra_train = shuffled_block(rng, b), shuffled_block(rng, a)
                    dist += squared_distances(extra_test, extra_train)
                    done += 1
                    if seed == 0 and done <= 3:
                        tiles.append(extra_test[:16])
                accuracy.append([np.mean(la[dist[:, :m].argmin(1)] == lb)
                                 for m in CURSE["sizes"]])
                near = np.sqrt(np.maximum(dist[:, :shown], 0))
                ratio.append(np.mean(near.min(1) / near.max(1)))
                if seed == 0:
                    hist.append(np.histogram((near / near.max(1, keepdims=True)).ravel(),
                                             CURSE["bins"], (0, 1))[0] / near.size)
                    nearest.append(near.argmin(1))
        levels = len(CURSE["blocks"])
        # the contrast: added coordinates that DO carry signal (training-only PCA)
        mean = a[:shown].mean(0)
        axes = np.linalg.svd(a[:shown] - mean, full_matrices=False)[2]
        pa, pb = (a[:shown] - mean) @ axes.T, (b - mean) @ axes.T
        pca_accuracy, pca_ratio = [], []
        for dim in CURSE["pca_dimensions"]:
            near = np.sqrt(np.maximum(squared_distances(pb[:, :dim], pa[:, :dim]), 0))
            pca_accuracy.append(np.mean(la[near.argmin(1)] == lb))
            pca_ratio.append(np.mean(near.min(1) / near.max(1)))
        first = np.sqrt(np.maximum(base[:, :shown], 0))
        return dict(
            dims=49 * (1 + np.array(CURSE["blocks"])),
            accuracy=np.array(accuracy).reshape(CURSE["shuffles"], levels, -1),
            ratio=np.array(ratio).reshape(CURSE["shuffles"], levels),
            hist=hist, nearest=nearest, farthest=first.argmax(1),
            near_distance=first.min(1), far_distance=first.max(1), noise_tiles=tiles,
            pca_accuracy=pca_accuracy, pca_ratio=pca_ratio,
            volume=[ball_in_box(v) for v in CURSE["pca_dimensions"]],
            train_labels=la, test_labels=lb)
    return _cached("curse", build, **CURSE)


def curse_dimensions():
    d = curse_data()
    acc = d["accuracy"]                                  # [shuffle, level, size]
    at = CURSE["sizes"].index(CURSE["shown_size"])
    mean = acc.mean(0)
    return Study(
        "The curse of dimensionality",
        "1-NN on real MNIST pixels as shuffled, no-signal pixel coordinates are added",
        [
            Panel("Distances concentrate",
                  "Mean nearest/farthest distance ratio, 1,000 test to 1,000 training images, "
                  "five shuffles.",
                  "curve", dict(x=d["dims"], ys=[d["ratio"].mean(0), d["ratio"].min(0),
                                                 d["ratio"].max(0)],
                                labels=["mean", "min", "max"])),
            Panel("Nearest-neighbour accuracy falls as no-signal coordinates are added",
                  "1,000 training images; the first 49 coordinates are the real pooled pixels.",
                  "curve", dict(x=d["dims"], ys=[acc[:, :, at].mean(0), acc[:, :, at].min(0),
                                                 acc[:, :, at].max(0)],
                                labels=["mean of 5 shuffles", "min", "max"])),
            Panel("Accuracy by coordinates (rows) and training images (columns)",
                  "Rows: %s coordinates. Columns: %s training images. Mean of five shuffles." % (
                      ", ".join(str(v) for v in d["dims"]),
                      ", ".join(str(v) for v in CURSE["sizes"])),
                  "matrix", dict(values=mean)),
            Panel("The contrast: added coordinates that carry signal",
                  "Keeping 2 to 49 real PCA coordinates (1,000 training images): accuracy rises.",
                  "curve", dict(x=np.array(CURSE["pca_dimensions"]),
                                ys=[d["pca_accuracy"], d["pca_ratio"]],
                                labels=["1-NN accuracy", "nearest/farthest ratio"])),
        ],
        _meta(
            **CURSE,
            signal_coordinates="49 real pixels: each test/train image mean-pooled to 7 x 7",
            added_coordinates="DISCLOSED TRANSFORMATION: blocks of 49 real pixel columns, each "
                              "column permuted across the images of its split (training and "
                              "test separately), so they keep each pixel's distribution and "
                              "carry no information about the image or its label",
            classifier="1-nearest neighbour, Euclidean distance, first N training images",
            volume="ball-in-box volumes are an exact formula, not a measurement",
            designs_tried=["49 real pixels + shuffled pixel blocks, 1-NN, sizes 125-8000 (fixed "
                           "before looking; a 3-shuffle, 500-test pilot of the same design was "
                           "run first to check its run time)"],
            replaces="next_vision.curse_dimensions: added PCA coordinates, which carry signal",
        ),
    )


# ====================================================================== 73 tsne-and-umap
MAPS = dict(samples=240, perplexities=[5, 30, 100], neighbours=[5, 15, 50], seeds=3,
            tsne_steps=400, umap_epochs=500, negatives=5, a=1.577, b=0.8951, every=5)


def pair_distances(x):
    return np.sqrt(np.maximum(((x[:, None] - x[None]) ** 2).sum(-1), 0))


def ranks(v):
    r = np.empty(len(v))
    r[np.argsort(v, kind="stable")] = np.arange(len(v))
    return r


def map_faithfulness(pixels, z, k=10):
    """How a map keeps the images' own distances: the mean share of each image's ``k``
    nearest images that are also its ``k`` nearest in the map, and the rank correlation of
    all pairwise distances."""
    d, m = pair_distances(pixels), pair_distances(z)
    a = np.argsort(d, 1, kind="stable")[:, 1:k + 1]
    b = np.argsort(m, 1, kind="stable")[:, 1:k + 1]
    kept = float(np.mean([len(set(u) & set(v)) / k for u, v in zip(a, b)]))
    iu = np.triu_indices(len(z), 1)
    return kept, float(np.corrcoef(ranks(d[iu]), ranks(m[iu]))[0, 1])


def fuzzy_memberships(x, k):
    """UMAP's directed graph: row ``i`` gives image ``i``'s ``k`` nearest images a
    membership ``exp(-(d - rho_i) / sigma_i)``: 1 for the nearest (``rho_i`` is its
    distance), falling with distance at a rate ``sigma_i`` found by bisection so that the
    row sums to ``log2(k)``."""
    dist = pair_distances(x)
    order = np.argsort(dist, 1, kind="stable")[:, 1:k + 1]
    out, target = np.zeros_like(dist), np.log2(k)
    for i in range(len(x)):
        d = dist[i, order[i]]
        lo, hi, sigma = 0.0, np.inf, 1.0
        for _ in range(64):
            total = np.exp(-np.maximum(d - d[0], 0) / sigma).sum()
            if abs(total - target) < 1e-5:
                break
            if total > target:
                hi = sigma
                sigma = (lo + hi) / 2
            else:
                lo = sigma
                sigma = sigma * 2 if np.isinf(hi) else (lo + hi) / 2
        out[i, order[i]] = np.exp(-np.maximum(d - d[0], 0) / sigma)
    return out


def fuzzy_graph(x, k):
    """The symmetric graph UMAP lays out: an edge is present if either end claims it
    (``a + b - a b``)."""
    v = fuzzy_memberships(x, k)
    return v + v.T - v * v.T


def spectral_start(graph, rng):
    """UMAP's start: the graph's two smoothest non-constant eigenvectors (normalised
    Laplacian), scaled to +-10, with a little noise."""
    degree = graph.sum(1)
    laplacian = np.eye(len(graph)) - graph / np.sqrt(degree[:, None] * degree[None])
    z = np.linalg.eigh(laplacian)[1][:, 1:3]
    return 10 * z / np.abs(z).max() + rng.normal(0, 1e-4, z.shape)


def umap_layout(graph, seed, epochs=MAPS["umap_epochs"], negatives=MAPS["negatives"],
                a=MAPS["a"], b=MAPS["b"], every=MAPS["every"]):
    """UMAP's layout by stochastic gradient descent on the fuzzy cross-entropy, as the
    reference implementation does it: each edge is visited in proportion to its weight and
    pulls its two ends together; each visit also pushes one end away from ``negatives``
    randomly drawn points; steps are clipped at 4 and the rate falls linearly from 1 to 0.
    ``a`` and ``b`` shape the map's similarity ``1 / (1 + a d^(2b))`` (the values UMAP
    derives for its default ``min_dist`` 0.1). Returns the map every ``every`` epochs
    (first: the spectral start)."""
    rng = np.random.default_rng(seed)
    z = spectral_start(graph, rng)
    ii, jj = np.nonzero(np.triu(graph, 1))
    w = graph[ii, jj]
    keep = w >= w.max() / epochs
    ii, jj, w = ii[keep].tolist(), jj[keep].tolist(), w[keep]
    period = (w.max() / w).tolist()
    due_at = list(period)
    # plain floats: the updates are sequential, one edge at a time
    zx, zy = z[:, 0].tolist(), z[:, 1].tolist()
    frames, n = [z.copy()], len(graph)

    def clip(v):
        return max(-4.0, min(4.0, v))
    for epoch in range(1, epochs + 1):
        rate = 1.0 - (epoch - 1) / epochs
        due = [e for e in range(len(ii)) if due_at[e] <= epoch]
        others = rng.integers(0, n, (len(due), negatives)).tolist()
        for e, away in zip(due, others):
            due_at[e] += period[e]
            i, j = ii[e], jj[e]
            dx, dy = zx[i] - zx[j], zy[i] - zy[j]
            d2 = dx * dx + dy * dy
            if d2 > 0:
                c = -2 * a * b * d2 ** (b - 1) / (1 + a * d2 ** b)
                gx, gy = clip(c * dx) * rate, clip(c * dy) * rate
                zx[i] += gx
                zy[i] += gy
                zx[j] -= gx
                zy[j] -= gy
            for k in away:
                if k == i:
                    continue
                dx, dy = zx[i] - zx[k], zy[i] - zy[k]
                d2 = dx * dx + dy * dy
                if d2 > 0:
                    c = 2 * b / ((0.001 + d2) * (1 + a * d2 ** b))
                    zx[i] += clip(c * dx) * rate
                    zy[i] += clip(c * dy) * rate
                else:
                    zx[i] += 4 * rate
                    zy[i] += 4 * rate
        if epoch % every == 0:
            frames.append(np.column_stack((zx, zy)))
    return np.array(frames)


def tsne_seed(s):
    return nv.SEED + s


@lru_cache(maxsize=1)
def maps_data():
    """The same 240 MNIST images (pooled to 7 x 7) mapped by exact t-SNE at three
    perplexities and by the UMAP procedure at three neighbourhood sizes, three seeds each."""
    def build():
        _, _, xt, yt = nv.mnist_data()
        x = downsample(xt[:MAPS["samples"]])
        tsne, losses, umap, frames, graphs = [], [], [], None, []
        for perplexity in MAPS["perplexities"]:
            for s in range(MAPS["seeds"]):
                z, loss = nv.tsne_fit(x, perplexity, MAPS["tsne_steps"], tsne_seed(s))
                tsne.append(z)
                losses.append(loss)
        for k in MAPS["neighbours"]:
            graph = fuzzy_graph(x, k)
            graphs.append(graph)
            for s in range(MAPS["seeds"]):
                path = umap_layout(graph, [SEED, k, s])
                umap.append(path[-1])
                if k == 15 and s == 0:
                    frames = path
        shape = (3, MAPS["seeds"])
        tsne, umap = (np.array(v).reshape(*shape, len(x), 2) for v in (tsne, umap))
        faith = np.array([[[map_faithfulness(x, z) for z in row] for row in maps]
                          for maps in (tsne, umap)])          # [method, setting, seed, measure]
        start = map_faithfulness(x, frames[0])
        return dict(pixels=x, labels=yt[:MAPS["samples"]], indices=np.arange(MAPS["samples"]),
                    tsne=tsne, tsne_losses=np.array(losses).reshape(*shape, -1), umap=umap,
                    umap_frames=frames, graph=graphs[1], faith=faith, spectral_faith=start)
    return _cached("maps", build, **MAPS)


def tsne_umap():
    d = maps_data()
    faith = d["faith"]
    rows = ["t-SNE p=%d" % p for p in MAPS["perplexities"]] + [
        "UMAP k=%d" % k for k in MAPS["neighbours"]]
    return Study(
        "t-SNE and UMAP",
        "The same 240 MNIST images mapped by exact t-SNE and by UMAP, both implemented here",
        [
            Panel("t-SNE, perplexity 30", "Seed %d of three; exact gradients, 400 steps."
                  % tsne_seed(0), "points", dict(xy=d["tsne"][1, 0], classes=d["labels"])),
            Panel("UMAP, 15 neighbours",
                  "Fuzzy 15-nearest-neighbour graph, spectral start, 500 epochs of sampled "
                  "attraction and repulsion.",
                  "points", dict(xy=d["umap"][1, 0], classes=d["labels"])),
            Panel("Share of each image's 10 nearest images kept nearby in the map",
                  "Rows: %s. Columns: three seeds." % ", ".join(rows),
                  "matrix", dict(values=faith[..., 0].reshape(6, -1))),
            Panel("Rank correlation of map distance with image distance, all pairs",
                  "Rows: %s. Columns: three seeds." % ", ".join(rows),
                  "matrix", dict(values=faith[..., 1].reshape(6, -1))),
        ],
        _meta(
            **MAPS, tsne_seeds=[tsne_seed(s) for s in range(MAPS["seeds"])],
            umap_seeds="numpy default_rng([51, neighbours, seed])",
            fit_split="the first 240 images of the MNIST test split are the embedding data; "
                      "nothing is trained and no held-out claim is made",
            umap="implemented here in numpy after McInnes et al.: fuzzy k-NN memberships, "
                 "probabilistic union, spectral start, edge-weighted sampling with negative "
                 "samples; all pairwise distances are exact (no approximate neighbour search)",
            measures="chosen before UMAP was fitted: the two the t-SNE animation already used",
            designs_tried=["the design above, the only one run"],
            replaces="next_vision.tsne_umap: t-SNE only",
        ),
    )


STUDIES = {
    "bias-variance": bias_variance,
    "gaussian-mixture-em": gaussian_mixture,
    "curse-of-dimensionality": curse_dimensions,
    "tsne-and-umap": tsne_umap,
}

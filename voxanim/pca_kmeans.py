"""Principal component analysis of MNIST and k-means on its first two
components, in plain numpy -- the two classical unsupervised methods the
``pca-and-kmeans`` animation shows next to the autoencoders.

**PCA** (:func:`pca`): centre the 60,000 training images on their mean,
take the covariance ``C = X^T X / (N - 1)`` (784 x 784) and its
eigendecomposition.  The eigenvectors, sorted by eigenvalue, are the
principal components: each is a 784-vector, an image with positive and
negative pixels, and the eigenvalue is the variance of the data along it.
Projecting an image onto the first ``k`` components gives ``k`` numbers
(:func:`project`); ``mean + z @ V_k^T`` rebuilds it from those numbers
(:func:`reconstruct`), and the error of that rebuild is the variance the
dropped components carried.  The same components come out of the SVD of
the centred data, which the tests check.

**k-means** (:func:`kmeans`): Lloyd's algorithm with ``k = 10`` on the 2-D
projection of 600 test digits (the same digits as the autoencoders' latent
clouds): random points become the first centroids; then, repeated to
convergence, every point joins its nearest centroid and every centroid moves
to the mean of its points.  The inertia (the sum of squared distances of
points to their centroids) falls at every step and never rises.  The clusters
are matched to labels one-to-one afterwards (:func:`voxanim.deep_models.cluster_accuracy`)
to say, honestly, how well two numbers per digit separate the ten digits.

Everything is deterministic (seeded) and cached by
:func:`voxanim.cache.cached_npz`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .autoencoder import CLOUD_START
from .cache import DEFAULT_CACHE, cached_npz
from .deep_models import cluster_accuracy
from .mnist import load

N_COMPONENTS = 50                # components kept in the cache (the largest reconstruction)
RECON_K = (2, 10, 50)            # components the reconstructions use
K_CLUSTERS = 10
CLOUD = 600                      # test digits projected and clustered (from CLOUD_START)
DIGIT = 18                       # the test digit reconstructed: a '3'
KMEANS_SEED = 3                  # converges in 16 iterations (seed 0 takes 23)
MAX_ITERS = 60                   # k-means gives up after this many (it converges long before)


# ------------------------------------------------------------------ PCA


def pca(x):
    """``(mean, components, variances)`` of ``x`` ``(N, D)``.

    ``components`` is ``(D, D)``, one unit eigenvector of the covariance per
    column, sorted by decreasing ``variances`` ``(D,)`` (the eigenvalues).
    The sign of each component is fixed so that its largest entry is positive
    (an eigenvector's sign is arbitrary; this makes the result reproducible).
    """
    x = np.asarray(x, np.float64)
    mean = x.mean(0)
    xc = x - mean
    cov = xc.T @ xc / (len(x) - 1)
    w, V = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    flip = np.sign(V[np.argmax(np.abs(V), axis=0), np.arange(V.shape[1])])
    return mean, V * flip[None], w


def project(x, mean, V, k):
    """The first ``k`` principal coordinates ``(N, k)`` of images ``x`` ``(N, D)``."""
    return (np.asarray(x, np.float64) - mean) @ V[:, :k]


def reconstruct(z, mean, V):
    """Images ``(N, D)`` rebuilt from ``k`` coordinates ``z`` ``(N, k)``."""
    z = np.asarray(z, np.float64)
    return mean + z @ V[:, :z.shape[1]].T


# ------------------------------------------------------------------ k-means


def assign(z, c):
    """Index of the nearest centroid for each of ``z`` ``(n, d)`` given ``c`` ``(k, d)``."""
    return ((z[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)


def inertia(z, c, lab):
    """Sum of squared distances of every point to its centroid."""
    return float(((z - c[lab]) ** 2).sum())


def kmeans(z, k, rng, max_iters=MAX_ITERS):
    """Lloyd's k-means on ``z`` ``(n, d)`` from ``k`` random points of ``z``.

    Returns ``(centroids, labels, inertias)``, one row per iteration ``I``:
    the centroids ``(I, k, d)`` the iteration assigned points to, the
    assignment ``labels`` ``(I, n)`` it made and the ``inertias`` ``(I,)`` of
    that assignment.  Iteration ``i + 1``'s centroids are the means of
    iteration ``i``'s clusters.  It stops when an assignment repeats
    (converged: the means, and so the centroids, would not change again).
    An empty cluster keeps its centroid.
    """
    c = z[rng.choice(len(z), k, replace=False)]
    cs, labs, costs = [c], [], []
    for _ in range(max_iters):
        lab = assign(z, c)
        labs.append(lab)
        costs.append(inertia(z, c, lab))
        if len(labs) > 1 and np.array_equal(lab, labs[-2]):
            break
        c = np.array([z[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        cs.append(c)
    return np.array(cs[:len(labs)]), np.array(labs), np.array(costs)


def confusion(lab, true, k=K_CLUSTERS):
    """``(k, 10)`` counts: how many points of cluster ``i`` carry label ``j``."""
    M = np.zeros((k, 10), int)
    np.add.at(M, (lab, true), 1)
    return M


# ------------------------------------------------------------------ the trace


@dataclass
class PCATrace:
    """PCA of the training set and k-means on the 2-D projection of the test cloud."""

    mean: np.ndarray            # (784,) the mean image
    components: np.ndarray      # (784, N_COMPONENTS)
    variances: np.ndarray       # (784,) all eigenvalues, decreasing
    cloud_z: np.ndarray         # (CLOUD, 2) the test digits' first two coordinates
    cloud_labels: np.ndarray    # (CLOUD,)
    digit: np.ndarray           # (784,) the digit reconstructed
    digit_label: int
    digit_z: np.ndarray         # (2,) its first two coordinates
    recons: np.ndarray          # (len(RECON_K), 784) rebuilt from RECON_K components
    centroids: np.ndarray       # (I, K, 2) the centroids each k-means iteration assigned to
    assignments: np.ndarray     # (I, CLOUD)
    inertias: np.ndarray        # (I,)

    @property
    def explained(self):
        """Fraction of the total variance each component explains, ``(784,)``."""
        return self.variances / self.variances.sum()

    @property
    def recon_errors(self):
        """Mean squared error of each reconstruction in :attr:`recons`."""
        return np.mean((self.recons - self.digit) ** 2, axis=1)

    @property
    def iterations(self):
        """Number of k-means iterations run (the last one changed nothing)."""
        return len(self.assignments)

    @property
    def final_labels(self):
        """The converged assignment ``(CLOUD,)``."""
        return self.assignments[-1]

    def matching(self):
        """``(accuracy, mapping)``: the best one-to-one match of clusters to
        digits and the fraction of the cloud it gets right."""
        return cluster_accuracy(self.final_labels, self.cloud_labels, K_CLUSTERS)

    def confusion(self):
        """``(K, 10)`` cluster-by-label counts of the converged assignment."""
        return confusion(self.final_labels, self.cloud_labels)


def trace(cloud=CLOUD, digit=DIGIT, seed=KMEANS_SEED, cache_dir=DEFAULT_CACHE) -> PCATrace:
    """PCA of the training images and k-means on the test cloud (cached)."""
    settings = dict(components=N_COMPONENTS, recon_k=list(RECON_K), cloud=cloud,
                    cloud_start=CLOUD_START, digit=digit, k=K_CLUSTERS, seed=seed,
                    max_iters=MAX_ITERS)
    _xtr, _ytr, xte, yte = load(cache_dir)
    sel = np.arange(CLOUD_START, CLOUD_START + cloud)

    def build():
        xtr, _ytr, _xte, _yte = load(cache_dir)
        mean, V, w = pca(xtr)
        V = V[:, :N_COMPONENTS]
        z = project(xte[sel], mean, V, 2)
        recons = np.array([reconstruct(project(xte[digit:digit + 1], mean, V, k), mean, V)[0]
                           for k in RECON_K])
        cs, labs, costs = kmeans(z, K_CLUSTERS, np.random.default_rng(seed))
        return dict(mean=mean, components=V, variances=w, cloud_z=z, recons=recons,
                    centroids=cs, assignments=labs, inertias=costs)

    d = cached_npz("pca_kmeans", settings, build, cache_dir)
    digit_z = project(xte[digit:digit + 1], d["mean"], d["components"], 2)[0]
    return PCATrace(d["mean"], d["components"], d["variances"], d["cloud_z"], yte[sel],
                    xte[digit], int(yte[digit]), digit_z, d["recons"], d["centroids"],
                    d["assignments"], d["inertias"])

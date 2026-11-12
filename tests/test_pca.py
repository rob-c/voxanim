"""PCA and k-means (:mod:`voxanim.pca_kmeans`) and what ``pca-and-kmeans`` says
about them."""

from __future__ import annotations

import numpy as np
from conftest import needs_mnist

from voxanim import pca_kmeans as PK


def test_pca_matches_svd():
    """The covariance eigenvectors are the right singular vectors of the
    centred data, with the variances the squared singular values / (N - 1)."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (300, 12)) @ rng.normal(0, 1, (12, 12)) + rng.normal(0, 1, 12)
    mean, V, w = PK.pca(x)
    xc = x - x.mean(0)
    _u, s, vt = np.linalg.svd(xc, full_matrices=False)
    assert np.allclose(mean, x.mean(0))
    assert np.allclose(w, s ** 2 / (len(x) - 1))
    assert np.allclose(np.abs(V.T), np.abs(vt))             # up to each vector's sign
    assert np.allclose(V.T @ V, np.eye(12))
    assert (V[np.argmax(np.abs(V), 0), np.arange(12)] > 0).all()   # the sign convention
    # projecting onto every component and back is the identity
    assert np.allclose(PK.reconstruct(PK.project(x, mean, V, 12), mean, V), x)
    # and the error of a k-component rebuild is the variance left out (per entry)
    k = 3
    err = np.mean((PK.reconstruct(PK.project(x, mean, V, k), mean, V) - x) ** 2)
    assert np.isclose(err, w[k:].sum() * (len(x) - 1) / x.size)


def test_kmeans_inertia_is_monotone_and_converges():
    rng = np.random.default_rng(1)
    z = np.concatenate([rng.normal(c, 0.5, (50, 2)) for c in ((0, 0), (4, 0), (0, 4))])
    cs, labs, costs = PK.kmeans(z, 3, np.random.default_rng(2))
    assert len(cs) == len(labs) == len(costs)
    assert (np.diff(costs) <= 1e-9).all()                      # never rises
    assert np.array_equal(labs[-1], labs[-2])                  # stopped on a repeat
    # each centroid after an iteration is the mean of the points assigned in it
    for i in range(len(labs) - 1):
        for j in range(3):
            if (labs[i] == j).any():
                assert np.allclose(cs[i + 1][j], z[labs[i] == j].mean(0))
    # the recorded inertia is the inertia of that iteration's assignment
    assert np.isclose(costs[0], PK.inertia(z, cs[0], labs[0]))
    assert np.isclose(costs[-1], PK.inertia(z, cs[-1], labs[-1]))


@needs_mnist
def test_trace_numbers():
    """The numbers the subtitles quote."""
    tr = PK.trace()
    e = tr.explained
    assert 0.09 < e[0] < 0.105 and 0.065 < e[1] < 0.075           # PC 1 9.7%, PC 2 7.1%
    assert 0.46 < np.cumsum(e)[9] < 0.52 and 0.80 < np.cumsum(e)[49] < 0.85
    assert np.allclose(tr.variances, np.sort(tr.variances)[::-1])
    err = tr.recon_errors
    assert err[0] > err[1] > err[2]                                  # 2 > 10 > 50 components
    assert 0.08 < err[0] < 0.12 and 0.015 < err[2] < 0.03
    assert tr.digit_label == 3
    # k-means on the 600 projected test digits
    assert tr.iterations == 16
    assert (np.diff(tr.inertias) <= 1e-9).all()
    assert 1150 < tr.inertias[0] < 1250 and 570 < tr.inertias[-1] < 600
    acc, mapping = tr.matching()
    assert 0.38 < acc < 0.46                                         # "42% of the time"
    assert sorted(mapping.values()) == list(range(10))              # one-to-one
    M = tr.confusion()
    assert M.sum() == len(tr.cloud_labels) == 600
    assert sum(M[j, mapping[j]] for j in range(10)) == round(acc * 600)


@needs_mnist
def test_animation_reads_the_trace():
    from voxanim.pca_anim import PCAKMeans
    anim = PCAKMeans(PK.trace())
    assert 40 <= anim.duration <= 60
    # the grid's diagonal count is the matched count
    tr = anim.tr
    M = anim.confusions[-1]
    assert int(np.trace(M)) == round(anim.purity * 600)
    # the cloud sits inside the plane
    u = (tr.cloud_z - anim.lo) / (anim.hi - anim.lo)
    assert (u > 0).all() and (u < 1).all()
    # every phase's subtitle is one or two lines
    from voxanim import layout
    for t in np.linspace(0, anim.t_done, 40):
        anim.scene(float(t))
        lines = [ln for part in anim.subtitle.split("\n")
                 for ln in layout.wrap(part, layout.SUB_SIZE, 1 - 2 * layout.SAFE)]
        assert len(lines) <= 2, anim.subtitle


def test_confusion_counts():
    lab = np.array([0, 0, 1, 2, 2, 2])
    true = np.array([5, 5, 1, 7, 7, 0])
    M = PK.confusion(lab, true, k=3)
    assert M.shape == (3, 10) and M[0, 5] == 2 and M[2, 7] == 2 and M.sum() == 6

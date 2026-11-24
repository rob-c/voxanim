"""voxanim.study_vectordb: InfoNCE's gradients, and the measured claims."""

from __future__ import annotations

import numpy as np
from conftest import needs_mnist

from voxanim import study_vectordb as D


def test_the_gradients_against_finite_differences():
    rng = np.random.default_rng(3)
    P = {k: v.astype(np.float64) for k, v in D.init_params(0).items()}
    x, p = rng.normal(size=(5, 784)), rng.normal(size=(5, 784))
    _l, G = D.loss_and_grads(P, x, p)
    for k in D.KEYS:
        for _ in range(3):
            idx = tuple(rng.integers(0, s) for s in P[k].shape)
            up, down = {**P, k: P[k].copy()}, {**P, k: P[k].copy()}
            up[k][idx] += 1e-6
            down[k][idx] -= 1e-6
            num = (D.loss_and_grads(up, x, p)[0] - D.loss_and_grads(down, x, p)[0]) / 2e-6
            assert abs(num - G[k][idx]) < 1e-5 * max(1.0, abs(num)), (k, idx)


def test_the_encoder():
    assert D.count() == 784 * 128 + 128 + 128 * 32 + 32 == 104_608
    u = D.unit(np.array([[3.0, 4.0]]))
    assert np.allclose(u, [[0.6, 0.8]])


@needs_mnist
def test_training_pulls_pairs_together():
    h = D.history(0)
    assert len(h["loss"]) == D.STEPS
    assert abs(h["loss"][0] - np.log(64)) < 0.3                 # untrained: about ln 64
    assert h["loss"][-100:].mean() < h["loss"][0] - 1.5
    assert h["pos"][-1] > 0.85 > 0.5 > h["neg"][-1]
    b = D.batch(5)
    assert (b["y"] == np.array([D.S.data()[1][j] for j in b["j"]])).all()   # same digits


@needs_mnist
def test_search_and_its_quality():
    q = D.quality()
    assert q["trained"]["top10"] > q["pixels"]["top10"] > q["untrained"]["top10"]
    assert q["mixed"]["top1"] < 0.15 and q["mixed"]["top10"] < 0.15     # about chance or less
    assert q["reembedded"]["top10"] > 0.95
    s = D.search(0)
    assert (np.diff(s["scores"]) <= 0).all() and len(s["ids"]) == 5
    st = D.store(0)
    assert st["vectors"].shape == (60_000, 32)
    assert np.allclose(np.linalg.norm(st["vectors"], axis=1), 1, atol=1e-5)


@needs_mnist
def test_ivf():
    iv = D.ivf()
    assert iv["list_sizes"].sum() == 60_000 and len(iv["list_sizes"]) == D.LISTS
    assert (np.diff(iv["recall"]) >= -1e-3).all() and (np.diff(iv["scanned"]) > 0).all()
    assert iv["recall"][-1] > 0.999 and abs(iv["scanned"][-1] - 1) < 1e-12
    k4 = list(iv["nprobe"]).index(4)
    assert iv["recall"][k4] > 0.99 and iv["scanned"][k4] < 0.1

"""voxanim.study_faces and voxanim.olivetti: the arithmetic, the gradients, and the measured
claims of the face animations."""

from __future__ import annotations

import numpy as np
from conftest import needs_faces

from voxanim import study_faces as D


def test_boxes_and_nms():
    assert np.allclose(D.corners([(10, 20, 8, 4)]), [[6, 18, 14, 22]])
    a = np.array([[0, 0, 10, 10]])
    assert np.allclose(D.iou(a, [[5, 0, 15, 10]]), 50 / 150)
    assert np.allclose(D.iou(a, a), 1) and np.allclose(D.iou(a, [[20, 20, 30, 30]]), 0)
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]], np.float32)
    assert list(D.nms(boxes, np.array([0.9, 0.95, 0.5]), 0.3)) == [1, 2]
    r, c, say = D.offsets(20.0, 44.0, 32.0, 16.0)
    assert (r, c) == (5, 2) and np.allclose(say, [0.5, 0.5, np.log(2), 0])


def test_the_gradients_against_finite_differences():
    rng = np.random.default_rng(4)
    P = {k: v.astype(np.float64) for k, v in D.det_init().items()}
    x = rng.random((2, 1, 96, 96))
    has = np.zeros((2, 12, 12))
    has[0, 3, 4] = has[1, 7, 2] = 1
    want = rng.normal(size=(2, 4, 12, 12)) * 0.3
    _l, _t, _w, G = D.det_loss_and_grads(P, x, has, want, 10.0)
    for k in D.DET_KEYS:
        idx = np.unravel_index(np.argmax(np.abs(G[k])), G[k].shape)
        up, down = {**P, k: P[k].copy()}, {**P, k: P[k].copy()}
        up[k][idx] += 1e-7                 # small: max-pool and ReLU have kinks
        down[k][idx] -= 1e-7
        f = lambda Q: D.det_loss_and_grads(Q, x, has, want, 10.0)[0]   # noqa: E731
        num = (f(up) - f(down)) / 2e-7
        assert abs(num - G[k][idx]) < 1e-4 * max(1.0, abs(num)), (k, num, G[k][idx])


@needs_faces
def test_the_data():
    x, who = D.photos()
    assert x.shape == (400, 64, 64) and x.min() == 0 and x.max() == 1
    assert list(np.bincount(who)) == [10] * 40
    assert D.small().shape == (400, 1, 32, 32)
    rng = np.random.default_rng(0)
    sc = D.scene(D.train_pool(), rng)
    assert sc["canvas"].shape == (96, 96) and 1 <= sc["faces"] <= 3
    assert len(sc["boxes"]) == sc["faces"] and all(16 <= s <= 32 for _x, _y, s in sc["squares"])
    img = x[0]
    sh = D.scramble(img, np.random.default_rng(1))
    assert sorted(sh.ravel()) == sorted(img.ravel()) and not np.array_equal(sh, img)
    assert all(who[s["ids"][: s["faces"]]].min() >= 30 for s in D.test_scenes()[:20])


@needs_faces
def test_the_detector():
    d, d1 = D.detection(10.0), D.detection(1.0)
    assert d["faces"] == d1["faces"] > 300
    assert d["ap"] > 0.85 > 0.6 > d1["ap"]                   # weighting the face cells matters
    assert d["found"] > 3 * d1["found"]
    assert len(D.det_run()["loss"]) == D.DET_STEPS
    assert D.det_run()["loss"][-100:].mean() < D.det_run()["loss"][0] / 5


@needs_faces
def test_the_recogniser():
    r, px = D.recognition(), D.recognition("pixels")
    assert r["pairs"] == 4950 and r["same_pairs"] == 450
    assert r["right"] == r["true_accept"] + r["true_reject"]
    assert r["test_eer"] < px["test_eer"]                    # verification: better than pixels
    assert r["identified"] <= r["probes"] == 45 and r["strangers"] == 50
    assert abs(r["threshold"] - D.trained_threshold()) < 1e-6
    p = D.pipeline_rates()
    assert p["found"] <= p["faces"] and p["right_detected"] <= p["found"]
    assert p["right_true"] > p["right_detected"]             # imperfect boxes cost names

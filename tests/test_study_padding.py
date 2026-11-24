"""voxanim.study_padding: the arithmetic of padding, and the measured claims."""

from __future__ import annotations

import numpy as np
from conftest import needs_mnist, needs_text

from voxanim import study_course as S
from voxanim import study_padding as P


def test_the_arithmetic():
    assert P.out_side(28, 3) == 26 and P.out_side(28, 3, 1) == 28
    assert P.out_side(28, 3, 1, 2) == 14
    c = P.coverage()
    assert c.shape == (28, 28) and c[0, 0] == 1 and c[0, 5] == 3 and c[14, 14] == 9
    assert c.sum() == 26 * 26 * 9                          # every place reads 9 pixels
    cp = P.coverage(pad=1)                                 # padded: the border still less
    assert (cp[0, 0], cp[0, 5], cp[14, 14]) == (4, 6, 9)
    assert cp.sum() == 28 * 28 * 9 - 4 * 26 * 3 - 4 * 5     # 9 everywhere, less the border
    assert (cp[1:-1, 1:-1] == 9).all()
    assert [(r["k"], r["pad"], r["same"], r["valid"]) for r in P.same_and_valid()] == [
        (3, 1, 28, 26), (5, 2, 28, 24), (7, 3, 28, 22)]
    d = P.deep()
    assert d["valid"] == list(range(28, 3, -2)) and set(d["same"]) == {28}
    assert P.pad_side_walk() == (28, 28, 14, 14, 7) and P.PAD_IN_FEATURES == 784
    assert P.count() == 8 * 9 + 8 + 16 * 8 * 9 + 16 + 784 * 10 + 10 == 9098
    assert abs(P.BG - (0 - 0.1307) / 0.3081) < 1e-6


def test_the_padded_gradients_against_finite_differences():
    rng = np.random.default_rng(2)
    W = {k: v.astype(np.float64) for k, v in P.init_params().items()}
    x = rng.normal(size=(2, 1, 28, 28))
    y = np.array([4, 7])
    _l, G = P.loss_and_grads(W, x, y)
    for k in P.PAD_KEYS:
        for _ in range(3):
            idx = tuple(rng.integers(0, s) for s in W[k].shape)
            up, down = {**W, k: W[k].copy()}, {**W, k: W[k].copy()}
            up[k][idx] += 1e-6
            down[k][idx] -= 1e-6
            num = (P.loss_and_grads(up, x, y)[0] - P.loss_and_grads(down, x, y)[0]) / 2e-6
            assert abs(num - G[k][idx]) < 1e-6 * max(1.0, abs(num)), (k, idx)


@needs_mnist
def test_the_modes_and_the_crops():
    m = P.modes()
    p = P.part()
    for mode, a in m.items():
        assert a.shape == (12, 12)
        assert np.array_equal(a[2:10, 2:10], p), mode
    assert (m["constant"][:2] == 0).all() and P.BG < -0.4      # zeros are not the paper
    assert np.array_equal(m["replicate"][0, 2:10], p[0])
    assert np.array_equal(m["circular"][0, 2:10], p[-2])
    assert np.array_equal(m["reflect"][0, 2:10], p[2])
    c = P.crops()
    assert c["shapes"] == [[1, 20, 20], [1, 20, 17], [1, 20, 20], [1, 20, 14]]
    assert "[1, 20, 20] at entry 0 and [1, 20, 17] at entry 1" in P.STACK_ERROR
    assert c["block_shape"] == [4, 1, 28, 28] and c["real"] + c["padding"] == 4 * 784
    for piece, block in zip(c["pieces"], c["block"], strict=True):
        h, w = piece.shape[1:]
        assert np.array_equal(block[:, :h, :w], piece)
        assert np.allclose(block[:, h:, :], P.BG) and np.allclose(block[:, :, w:], P.BG)


@needs_mnist
def test_the_padded_cnn_run():
    c = P.compare()
    assert (c["plain_params"], c["padded_params"]) == (5258, 9098)
    assert c["padded_accuracy"] > 0.95 and c["plain_accuracy"] > 0.95
    assert list(c["steps"]) == list(S.heldout("sgd")["steps"])
    assert c["padded_missed"] == round((1 - c["padded_accuracy"]) * 10_000)
    sh = P.shifted()
    assert sh["plain"].shape == sh["padded"].shape == (4, 7)
    assert np.allclose(sh["plain"][:, 0], c["plain_accuracy"])
    assert np.allclose(sh["padded"][:, 0], c["padded_accuracy"])
    assert (np.diff(sh["plain_mean"]) < 0).all() and (np.diff(sh["padded_mean"]) < 0).all()
    moved = P.shift(S.data()[2][:5], 3, "right").reshape(-1, 28, 28)
    assert np.allclose(moved[:, :, :3], P.BG)


@needs_text
def test_lines_of_text():
    b = P.lines_batch()
    assert b["lengths"] == [len(s) for s in P.lines()[:8]] and b["batch"].shape == (8, 50)
    assert b["real"] == sum(b["lengths"]) == 163 and b["cells"] == 400
    assert P.vocab()[0] == "<pad>" and len(P.vocab()) == 66
    assert (b["batch"][~b["mask"]] == 0).all() and (b["batch"][b["mask"]] > 0).all()
    assert P.masked_mean()["ratio"][5] == 1.0                      # the longest line
    w = P.waste()
    assert w["sorted"]["share"] < 0.01 < 0.3 < w["file"]["share"]
    assert w["file"]["cells"] - w["file"]["padding"] == w["characters"]

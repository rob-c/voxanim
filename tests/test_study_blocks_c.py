"""The numbers behind animations 191-193 (voxanim.study_blocks_c): a convolution and a
max-pool on a real digit, checked against direct loops (and against torch when it is
installed), and the dropout run's switches."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import study_blocks_c as S
from voxanim import study_course as course


# ---------------------------------------------------------------------- no data needed
def test_filters_start_as_pytorch_starts_them():
    W, b = S.filters()
    assert W.shape == (4, 1, 3, 3) and b.shape == (4,)
    assert np.abs(W).max() <= 1 / 3 and np.abs(b).max() <= 1 / 3
    W2, b2 = S.filters()
    np.testing.assert_array_equal(W, W2)                  # seeded: the same every time
    np.testing.assert_array_equal(b, b2)


def test_conv2d_is_a_sliding_dot_product():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 1, 7, 6)).astype(np.float32)
    W, b = S.filters()
    out = S.conv2d(x, W, b)
    assert out.shape == (2, 4, 5, 4)
    for n, f, i, j in ((0, 0, 0, 0), (1, 3, 4, 3), (0, 2, 2, 1)):
        want = float((x[n, 0, i:i + 3, j:j + 3] * W[f, 0]).sum() + b[f])
        assert np.isclose(out[n, f, i, j], want, atol=1e-5)
    patch, products, total = S.one_output(x[1, 0], W[3, 0], b[3], 4, 3)
    assert patch.shape == products.shape == (3, 3)
    assert np.isclose(total, out[1, 3, 4, 3], atol=1e-5)


def test_maxpool_keeps_the_biggest_of_each_square():
    a = np.arange(16, dtype=float).reshape(1, 1, 4, 4)
    np.testing.assert_array_equal(S.maxpool2d(a)[0, 0], [[5, 7], [13, 15]])
    a = -a
    np.testing.assert_array_equal(S.maxpool2d(a)[0, 0], [[0, -2], [-8, -10]])


def test_the_busiest_window():
    img = np.zeros((6, 6))
    img[2:4, 4:6] = [[1, 2], [3, 4]]
    assert S.busy_window(img, 2, step=2) == (2, 4)


def test_conv_and_pool_match_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    x = rng.normal(size=(3, 1, 28, 28)).astype(np.float32)
    W, b = S.filters()
    conv = torch.nn.Conv2d(1, 4, 3)
    with torch.no_grad():
        conv.weight.copy_(torch.from_numpy(W))
        conv.bias.copy_(torch.from_numpy(b))
        maps = conv(torch.from_numpy(x))
        small = torch.nn.MaxPool2d(2)(maps)
    np.testing.assert_allclose(S.conv2d(x, W, b), maps.numpy(), atol=1e-5)
    np.testing.assert_allclose(S.maxpool2d(S.conv2d(x, W, b)), small.numpy(), atol=1e-5)
    assert sum(p.numel() for p in conv.parameters()) == 40


# ---------------------------------------------------------------------- the data
@needs_mnist
def test_conv_on_the_first_digit():
    s = S.conv().validate()
    m = s.metadata
    assert m["shape_in"] == [64, 1, 28, 28] and m["shape_out"] == [64, 4, 26, 26]
    assert m["places"] == 676 == 26 * 26 and m["multiplies"] == 676 * 9
    assert (m["weights"], m["biases"], m["learned"]) == (36, 4, 40)
    assert m["learned_sum"] == "4 × 3 × 3 + 4 = 40"
    W, b = S.filters()
    np.testing.assert_allclose(m["weight"], W)
    img = course.first_batch()["x"][0, 0]
    r, c = m["at"]
    np.testing.assert_array_equal(np.asarray(m["patch"], np.float32), img[r:r + 3, c:c + 3])
    np.testing.assert_allclose(m["products"], np.asarray(m["patch"]) * W[0, 0], atol=1e-6)
    assert np.isclose(m["output"], m["products_sum"] + b[0], atol=1e-6)
    maps = np.asarray(s.panels[2].data["values"])
    assert np.isclose(maps[0, r, c], m["output"], atol=1e-5)
    assert len(np.unique(np.round(m["patch"], 3))) >= 7       # paper, ink and between
    assert m["make"] == "conv = nn.Conv2d(1, 4, 3)" and m["run"] == "maps = conv(x)"


@needs_mnist
def test_pool_keeps_a_quarter_and_learns_nothing():
    s = S.pool().validate()
    m = s.metadata
    assert m["shape_out"] == [64, 4, 13, 13] and m["squares"] == 169
    assert (m["per_digit_in"], m["per_digit_out"]) == (2704, 676) and 676 * 4 == 2704
    sheet = np.asarray(s.panels[0].data["values"][0])
    r, c = m["at"]
    assert r % 2 == 0 and c % 2 == 0
    square = np.asarray(m["square"])
    np.testing.assert_array_equal(square, sheet[r:r + 2, c:c + 2])
    assert m["biggest"] == square.max() == square[tuple(m["winner"])]
    assert len(np.unique(square)) == 4
    small = np.asarray(s.panels[2].data["values"])
    assert small[0][tuple(m["at_out"])] == m["biggest"]
    conv = S.conv()
    np.testing.assert_array_equal(sheet, np.asarray(conv.panels[2].data["values"][0]))
    assert m["weights"] == 0


@needs_mnist
def test_dropout_switches_are_the_runs():
    s = S.dropout().validate()
    m = s.metadata
    P = course.init_params()
    b = course.first_batch()
    h = course.forward(P, b["x"])["h1"][0]
    np.testing.assert_array_equal(np.asarray(m["h"], np.float32), h)
    assert m["relu_zeros"] == int((h == 0).sum())
    # the switches reproduce the recorded run's first loss
    again = course.loss_of(P, b["x"], b["y"], masks=course._masks(0, 64))
    assert np.isclose(again, course.losses("dropout")[0], atol=1e-6)
    for st in m["steps"]:
        sw = S.switches(st["step"])
        assert set(np.unique(sw)) <= {0.0, np.float32(1.25)}
        assert st["off"] == [int(i) for i in np.flatnonzero(sw == 0)]
        assert st["count"] == len(st["off"]) and 5 <= st["count"] <= 25
        np.testing.assert_allclose(st["seen"], h * sw, atol=1e-6)
    assert [st["count"] for st in m["steps"]] == [15, 12, 15]
    assert m["scale"] == 1.25 and m["example_out"] == pytest.approx(m["example_in"] * 1.25)
    assert abs(m["epoch_share"] - 0.2) < 0.002 and m["epoch_switches"] == 60_000 * 64


@needs_mnist
def test_eval_scores_better_than_training_mode():
    m = S.dropout().metadata
    assert m["accuracy_eval"] == pytest.approx(0.9435)
    assert m["accuracy_train_mode"] == pytest.approx(0.919)
    assert m["right_eval"] - m["right_train_mode"] == 245
    assert m["switch"] == ["model.train()", "model.eval()"]

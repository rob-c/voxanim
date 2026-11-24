"""The numbers behind animation 271 (voxanim.study_blocks_ze): a 9 x 9 kernel on a 6 x 6 picture
with padding 8 gives 14 x 14; the middle output pixel is all 36 pixels times the kernel; the
stamps added up are conv2d and M @ x; a column of M is the kernel stamped at its pixel; M is
V turned on its side; the total is kept. The photo parts need the Olivetti faces."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_faces

from voxanim import study_blocks_ze as Z
from voxanim import study_upconv as U


# ---------------------------------------------------------------------- no download
def test_formats_and_the_rule():
    assert Z.num(22.98766) == "22.99" and Z.num(0.0010681, 3) == "0.001"
    x = np.zeros((6, 6))
    x[4, 1] = 1.0
    x[5, 5] = 1.0                                   # tied: the first
    assert Z.brightest(x) == (4, 1)


def test_the_stamps_added_up_are_conv2d():
    x = np.random.default_rng(2).random((6, 6))
    k = U.kernel()
    np.testing.assert_allclose(Z.stamps_added(x), U.conv(x, k, 8), atol=1e-15)
    np.testing.assert_allclose(Z.stamps_added(x), (U.matrix() @ x.ravel()).reshape(14, 14),
                               atol=1e-15)
    # a lopsided kernel too: the stamp is the kernel flipped, as conv2d places it
    lop = np.arange(81, dtype=float).reshape(9, 9)
    np.testing.assert_allclose(Z.stamps_added(x, lop), U.conv(x, lop, 8), atol=1e-9)


def test_the_kernel_is_bigger_and_its_middle_row():
    k = U.kernel()
    assert k.shape == (9, 9) and 9 > 6 and np.isclose(k.sum(), 1.0)
    row = ["%.3f" % v for v in k[4]]
    assert row == ["0.001", "0.009", "0.030", "0.060", "0.075", "0.060", "0.030", "0.009",
                   "0.001"]
    assert np.allclose(np.array([1, 8, 28, 56, 70, 56, 28, 8, 1]) / 256, U.ROW)


def test_against_torch():
    torch = pytest.importorskip("torch")
    x = np.random.default_rng(3).random((6, 6))
    t = torch.nn.functional.conv2d(torch.tensor(x)[None, None],
                                   torch.tensor(U.kernel())[None, None], padding=8)[0, 0]
    np.testing.assert_allclose(t.numpy(), Z.stamps_added(x), atol=1e-12)
    # V (no padding, 14 x 14 to 6 x 6) is what conv2d does without padding
    y = np.random.default_rng(4).random((14, 14))
    tv = torch.nn.functional.conv2d(torch.tensor(y)[None, None],
                                    torch.tensor(U.kernel())[None, None])[0, 0]
    np.testing.assert_allclose(tv.numpy().ravel(), U.down_matrix() @ y.ravel(), atol=1e-12)


# ---------------------------------------------------------------------- photo 0
@needs_faces
def test_the_study():
    s = Z.big_kernel_study().validate()
    m = s.metadata
    assert s.title == "Bigger: a kernel larger than the picture" and len(s.panels) == 4
    assert m["equation"] == "6 + 2 × 8 − 9 + 1 = 14" and m["side"] == 14
    assert m["padded_side"] == 22 and np.asarray(m["padded"]).shape == (22, 22)
    assert m["outputs"] == 196 and m["M_shape"] == [196, 36] and m["V_shape"] == [36, 196]
    assert m["middle_row_text"][4] == "0.075" and m["kernel_sum"] == pytest.approx(1.0)
    # the middle output: all 36 under the kernel, 45 on zeros, 0.61
    assert m["worked_at"] == [7, 7] and m["under"] == 36 and m["on_zeros"] == 45
    assert m["worked_text"] == "0.61" and m["worked"] == pytest.approx(m["out"][7, 7])
    assert m["middle_reads"] == 36
    # the column shown: the brightest pixel, (4, 1), column 25
    assert m["column_at"] == [4, 1] and m["column"] == 25 and m["column_value_text"] == "0.76"
    assert m["per_column"] == 81 and m["stamp_corner"] == [4, 1]
    stamp = np.asarray(m["stamp"])
    assert (stamp > 0).sum() == 81 and np.allclose(stamp[4:13, 1:10], U.kernel())
    # exact: conv2d is M @ x, M is V on its side, columns add to 1, total kept
    assert m["same"] == 0.0 and m["transpose"] == 0.0
    assert np.array_equal(np.asarray(m["M"]), np.asarray(m["V"]).T)
    assert np.allclose(m["column_sums"], 1.0) and np.allclose(m["V_row_sums"], 1.0)
    assert m["total_in_text"] == m["total_out_text"] == "22.99"
    assert m["total_in"] == pytest.approx(m["total_out"], abs=1e-12)
    assert m["out_max_text"] == "0.62"
    assert np.asarray(m["x"]).shape == (6, 6) and 0 <= np.min(m["x"]) <= np.max(m["x"]) <= 1


@needs_faces
def test_the_picture_is_photo_0_shrunk():
    m = Z.big_kernel_study().metadata
    np.testing.assert_array_equal(m["x"], U.picture())
    assert m["photo_index"] == 0 and m["person"] == 0

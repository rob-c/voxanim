"""The numbers behind animations 263-264 (voxanim.study_blocks_zb): the kernel adds up to 1, the
pixel worked out is conv2d's, D is the kernel moved 2 places a row and D X D.T is conv2d,
spreading then the kernel x 4 is conv_transpose2d (bilinear), the edge receives half, and
the measured false rings and round trips are the ones on screen. The photo parts need the
Olivetti faces."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_faces

from voxanim import blocks as B
from voxanim import study_blocks_zb as Z
from voxanim import study_resample as R


# ---------------------------------------------------------------------- no download
def test_formats():
    assert Z.frac(2, 16) == "2/16"
    assert Z.err(0.0012903) == "0.00129" and Z.err(0.00092336) == "0.00092"


def test_the_sliding_kernel_is_down_and_the_matrix():
    plate = R.zone_plate()
    small, skip = R.down(plate)
    slid = Z.conv3(plate, R.KERNEL, 2)
    assert slid.shape == (32, 32)
    np.testing.assert_allclose(slid, small, atol=1e-6)
    D = R.down_matrix()
    np.testing.assert_allclose(D @ plate @ D.T, slid, atol=1e-6)
    assert np.array_equal(skip, plate[::2, ::2])


def test_d_is_the_kernel_moved_two_places_a_row():
    D = R.down_matrix()
    assert D.shape == (32, 64)
    assert D[0, :3].tolist() == [0.5, 0.25, 0.0]           # its first number is past the edge
    assert D[1, 1:4].tolist() == [0.25, 0.5, 0.25]
    for r in range(1, 32):
        assert np.array_equal(D[r, 2:], D[r - 1, :-2])
        assert D[r, 2 * r] == 0.5                          # row i is centred on pixel 2i
    assert D.sum(1)[0] == 0.75 and np.allclose(D.sum(1)[1:], 1.0)


def test_spread_then_the_kernel_x4_is_bilinear():
    small = R.down(R.zone_plate())[0]
    sp = Z.spread(small)
    assert sp.shape == (64, 64) and np.array_equal(sp[::2, ::2], small)
    assert (sp[1::2] == 0).all() and (sp[:, 1::2] == 0).all()
    big = R.up(small)[0]
    np.testing.assert_allclose(Z.conv3(sp, 4 * R.KERNEL), big, atol=1e-6)
    np.testing.assert_allclose(big[::2, ::2], small, atol=1e-6)           # kept: itself
    np.testing.assert_allclose(big[1:-1:2, ::2], (small[:-1] + small[1:]) / 2, atol=1e-6)
    np.testing.assert_allclose(big[1:-1:2, 1:-1:2], (small[:-1, :-1] + small[1:, :-1]
                                                     + small[:-1, 1:] + small[1:, 1:]) / 4,
                               atol=1e-6)
    np.testing.assert_allclose(big[-1, ::2], small[-1] / 2, atol=1e-6)   # the edge: half


def test_the_fine_rings_lie_outside_the_circle():
    a = R.aliasing()
    assert a["fine_pixels"] == 827
    assert (round(a["skip"], 3), round(a["blur"], 3), round(a["average"], 3)) == \
        (0.354, 0.079, 0.130)


def test_what_the_script_prints_for_d_x_dt():
    torch = pytest.importorskip("torch", reason="torch is not installed")
    import runpy
    ns = runpy.run_path(str(B.RESAMPLE), run_name="test")
    plate = ns["zone_plate"]()
    small, _skip = ns["down"](plate)
    run = dict(ns, plate=plate, small=small, D=ns["down_matrix"]())
    exec(B.code_lines("rs-same")[0], run)
    assert str((run["by_matrix"] - small).abs().max()) == Z.PRINTED_SAME
    assert isinstance(run["by_matrix"], torch.Tensor)


# ---------------------------------------------------------------------- the photos
@needs_faces
def test_the_worked_pixel_is_picked_by_its_rule():
    photo = R.photo(Z.PHOTO)
    at = Z.worked_at(photo)
    assert at == (9, 23)
    i, j = at
    spread = photo[2 * i - 1:2 * i + 2, 2 * j - 1:2 * j + 2]
    best = spread.max() - spread.min()
    for a in range(8, 24):
        for b in range(8, 24):
            p = photo[2 * a - 1:2 * a + 2, 2 * b - 1:2 * b + 2]
            assert p.max() - p.min() <= best


@needs_faces
def test_smaller():
    s = Z.down_study().validate()
    m = s.metadata
    assert len(s.panels) == 4
    assert m["row_text"] == ["1/4", "2/4", "1/4"]
    assert m["kernel_text"] == [["1/16", "2/16", "1/16"], ["2/16", "4/16", "2/16"],
                                ["1/16", "2/16", "1/16"]]
    assert (m["total16"], m["total"]) == (16, 1.0)
    assert m["at"] == [9, 23] and m["centre"] == [18, 46]
    assert m["patch_text"] == [["0.57", "0.38", "0.40"], ["0.69", "0.54", "0.45"],
                               ["0.64", "0.60", "0.52"]]
    patch, kernel = np.asarray(m["patch"]), np.asarray(m["kernel"])
    assert abs((patch * kernel).sum() - m["small"][9, 23]) < 1e-6
    assert m["output_text"] == "0.53" and m["outputs_text"] == "1,024"
    assert m["D_corner_text"][0][:3] == ["0.5", "0.25", "0"]
    assert m["D_corner_text"][3][5:8] == ["0.25", "0.5", "0.25"]
    assert m["D_first_row_sum"] == 0.75 and m["D_shape"] == [32, 64]
    assert m["matrix_gap"] < 1e-6 and m["printed_same"] == "tensor(1.7881e-07)"
    assert np.asarray(m["plate_dx"]).shape == (32, 64)
    np.testing.assert_allclose(m["by_matrix"], m["plate_small"], atol=1e-6)
    assert (m["fine_pixels"], m["rms_skip_text"], m["rms_blur_text"],
            m["rms_average_text"]) == (827, "0.354", "0.079", "0.130")
    assert int(np.asarray(m["fine"]).sum()) == 827
    assert any("0.75" in x for x in m["limitations"])
    assert "0.354" in s.subtitle and "0.079" in s.subtitle
    assert m["code_matrix"][-1] == "            D[i, j] = w"


@needs_faces
def test_bigger():
    s = Z.up_study().validate()
    m = s.metadata
    assert len(s.panels) == 4
    assert m["kernel4_text"] == [["0.25", "0.5", "0.25"], ["0.5", "1", "0.5"],
                                 ["0.25", "0.5", "0.25"]]
    assert m["spread_close_text"][0] == ["0.58", "0", "0.42", "0", "0.33"]
    assert m["spread_close_text"][1] == ["0"] * 5
    assert m["big_close_text"][2][:3] == ["0.64", "0.59", "0.53"]
    assert m["blocks_close_text"][0] == ["0.58", "0.58", "0.42", "0.42", "0.33"]
    assert (m["two_text"], m["gap2_text"], m["gap4_text"]) == (["0.64", "0.53"], "0.59", "0.54")
    assert abs(m["gap2_value"] - sum(m["two"]) / 2) < 1e-6
    assert abs(m["gap4_value"] - sum(m["four"]) / 4) < 1e-6
    assert (m["zeros"], m["kept"]) == (3072, 1024)
    assert m["got_corner_text"] == [["1", "1", "0.5"], ["1", "1", "0.5"],
                                    ["0.5", "0.5", "0.25"]]
    np.testing.assert_allclose(m["fixed"], np.asarray(m["big"]) / np.asarray(m["got"]))
    assert (m["last_row_text"], m["last_row_fixed_text"]) == ("0.28", "0.57")
    assert (m["inner_bilinear_text"], m["inner_nearest_text"], m["full_bilinear_text"],
            m["full_even_text"], m["inner_skip_bilinear_text"]) == \
        ("0.00129", "0.00302", "0.00343", "0.00199", "0.00092")
    # honest: skip then bilinear lands closer; the edge fix changes only the edge
    assert m["inner_skip_bilinear"] < m["inner_bilinear"] < m["inner_nearest"]
    assert m["trips"]["blur_even"]["inner"] == m["trips"]["blur_bilinear"]["inner"]
    assert m["full_even"] < m["full_bilinear"]
    assert any("not a closer copy" in x for x in m["limitations"])
    assert m["code_edge"] == ["ones = torch.ones_like(small)", "got, _ = up(ones)",
                              "fixed = big / got"]


def test_the_studies_are_registered():
    assert set(Z.STUDIES) == {"resample-down", "resample-up"}

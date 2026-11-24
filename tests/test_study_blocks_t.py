"""The numbers behind animations 238-240 (voxanim.study_blocks_t): the box and its corners,
IoU as areas, the grid's cell and its numbers, the detector's shapes and counts, and what one
cell can see, checked against study_faces, torch's printing and the trained model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import needs_faces

from voxanim import study_blocks_t as T
from voxanim import study_faces as F

CATALOGUE = Path(__file__).resolve().parent.parent / "voxanim" / "next_catalog.json"


def catalogue_titles():
    return {item["number"]: item["title"] for item in json.loads(CATALOGUE.read_text())}


# ---------------------------------------------------------------------- no data needed
def test_the_titles_are_the_catalogues():
    titles = catalogue_titles()
    assert (titles[238], titles[239], titles[240]) == (
        "A box is four numbers", "A grid of guesses", "From pixels to a grid")


def test_formats():
    assert (T.num(27.5), T.num(19.0), T.num(18.926), T.num(69.057)) == ("27.5", "19", "18.9",
                                                                        "69.1")
    assert T.two(0.7077) == "0.71"
    assert T.area([18, 52, 37, 71]) == 361 and T.area([5, 5, 2, 9]) == 0
    assert T.overlap([0, 0, 10, 10], [5, -2, 15, 4]) == [5, 0, 10, 4]


def test_the_printed_tensors_are_torchs():
    torch = pytest.importorskip("torch", reason="torch is not installed")
    for a in ([[55.0, 32, 87, 64], [18, 52, 37, 71]], [[0.74515, 0.0], [0.0, 0.70767]],
              [0.4375, 0.6875, 0.171850, 0.171850], [[1.5, -2.25], [10.0, 0.0]]):
        assert T.tensor_text(np.array(a, np.float32)) == str(torch.tensor(a))


def test_reach_grows_3_4_8_10_18_22():
    assert [s for _n, s, _j in T.reach()] == [3, 4, 8, 10, 18, 22, 22]
    assert [j for _n, _s, j in T.reach()][-1] == 8 == F.CELL
    (r0, r1), (c0, c1) = T.field(7, 3)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (22, 22)
    assert ((r0 + r1) / 2, (c0 + c1) / 2) == (7 * 8 + 3.5, 3 * 8 + 3.5)    # centred on the cell


def test_reach_by_brute_force_on_the_structure():
    """Which input pixels can reach one output cell, followed back through the layers as
    index sets (padding 1, stride 1 convolutions; 2 x 2 pools): the same 22 x 22 square."""
    rows = {7}
    for name, k in reversed(T.LAYERS):
        if name == "pool":
            rows = {2 * r + d for r in rows for d in range(k)}
        else:
            rows = {r + d for r in rows for d in range(-(k // 2), k // 2 + 1)}
    assert (min(rows), max(rows)) == tuple(T.field(7, 3)[0])


# ---------------------------------------------------------------------- 238
@needs_faces
def test_a_box():
    s = T.a_box().validate()
    m = s.metadata
    assert len(s.panels) == 4 and s.title == "A box is four numbers"
    sc = F.test_scenes()[T.SCENE]
    assert (m["faces"], m["nots"]) == (2, 2) and m["paste"] == [18, 52, 19]
    assert m["box"] == list(sc["boxes"][T.FACE]) == [27.5, 61.5, 19.0, 19.0]
    assert m["box_text"] == "(27.5, 61.5, 19, 19)" and m["centre_text"] == "(27.5, 61.5)"
    assert m["corners"] == [18, 52, 37, 71] and m["corners_text"] == "(18, 52, 37, 71)"
    assert np.allclose(F.corners([m["box"]]), [m["corners"]])
    assert m["corner_sums"] == ["x1 = 27.5 − 9.5 = 18", "y1 = 61.5 − 9.5 = 52",
                                "x2 = 27.5 + 9.5 = 37", "y2 = 61.5 + 9.5 = 71"]
    assert m["printed_real"] == "tensor([[55., 32., 87., 64.],\n        [18., 52., 37., 71.]])"
    # the detector's box for this face, and IoU as areas
    found = F.find(F.det_params(), sc["canvas"])
    assert np.allclose(m["guess"], found["found"][m["guess_index"]])
    assert m["guess_text"] == "(18.9, 49.4, 38.4, 69.1)" and m["guess_score_text"] == "0.97"
    assert (m["area_texts"]["true"], m["area_texts"]["guess"], m["area_texts"]["inter"],
            m["area_texts"]["union"]) == ("361", "383", "308", "436")
    assert m["union_sum"] == "361 + 383 − 308 = 436" and m["iou_sum"] == "308 ÷ 436 = 0.71"
    assert np.isclose(m["union"], m["area_true"] + m["area_guess"] - m["area_inter"])
    assert np.isclose(m["iou"], F.iou([m["guess"]], [m["corners"]])[0, 0])
    assert 361 + 383 - 308 == 436 and round(308 / 436, 2) == 0.71       # the rounded sums agree
    assert m["printed_iou"] == "tensor([[0.7452, 0.0000],\n        [0.0000, 0.7077]])"
    assert m["mean_iou_text"] == "0.78" == "%.2f" % F.detection()["mean_iou"]
    assert any("made" in t for t in m["limitations"])


# ---------------------------------------------------------------------- 239
@needs_faces
def test_the_grid():
    s = T.the_grid().validate()
    m = s.metadata
    assert len(s.panels) == 4 and s.title == "A grid of guesses"
    assert (m["grid"], m["cell"], m["grid_sum"]) == (12, 8, "96 ÷ 8 = 12")
    assert (m["row"], m["col"]) == (7, 3) == (int(61.5 // 8), int(27.5 // 8))
    assert (m["row_div"], m["col_div"]) == ("61.5 ÷ 8 = 7.69", "27.5 ÷ 8 = 3.44")
    assert (m["dx"], m["dy"]) == (0.4375, 0.6875) and (m["dx_text"], m["dy_text"]) == (
        "0.44", "0.69")
    assert m["in_cell"] == [3.5, 5.5] and np.allclose(np.array(m["in_cell"]) / 8,
                                                      [m["dx"], m["dy"]])
    assert np.isclose(m["lw"], np.log(19 / 16)) and m["log_sum"] == "log(19 ÷ 16) = 0.17"
    assert m["size_logs_text"] == ["0.00", "0.69"]
    assert m["five_text"] == ["1", "0.44", "0.69", "0.17", "0.17"]
    assert m["printed_offsets"] == "(7, 3, tensor([0.4375, 0.6875, 0.1719, 0.1719]))"
    assert m["printed_offsets_wrapped"].replace("\n" + " " * 15, " ") == m["printed_offsets"]
    has, want = np.asarray(m["has"]), np.asarray(m["want"])
    h2, w2 = F.targets(F.test_scenes()[T.SCENE]["boxes"])
    assert np.array_equal(has, h2) and np.allclose(want, w2)
    assert (m["has_ones"], m["has_zeros"]) == (2, 142) and has.sum() == 2
    assert sorted(map(tuple, m["face_cells"])) == [(6, 8), (7, 3)]
    assert np.allclose(want[:, 7, 3], m["say"])
    assert np.allclose(m["other_say"], [0.875, 0.0, np.log(2), np.log(2)], atol=1e-6)
    # the not-faces' cells say 0, and every cell but the two faces' holds zeros
    assert all(v == 0.0 for _r, _c, v in m["not_cells"])
    mask = np.ones((12, 12), bool)
    mask[7, 3] = mask[6, 8] = False
    assert not has[mask].any() and not want[:, mask].any()
    torch = pytest.importorskip("torch", reason="torch is not installed")
    say = torch.tensor([m["dx"], m["dy"], m["lw"], m["lw"]])
    assert m["printed_offsets"] == str((7, 3, say))


# ---------------------------------------------------------------------- 240
@needs_faces
def test_the_detector():
    s = T.the_detector().validate()
    m = s.metadata
    assert len(s.panels) == 4 and s.title == "From pixels to a grid"
    assert m["shape_texts"] == ["[1, 1, 96, 96]", "[1, 8, 48, 48]", "[1, 16, 24, 24]",
                                "[1, 32, 12, 12]", "[1, 5, 12, 12]"]
    assert m["counts"] == [80, 1168, 4640, 165] and m["total"] == 6053 == F.count(F.det_init())
    assert m["count_texts"] == ["80", "1,168", "4,640", "165"] and m["total_text"] == "6,053"
    assert m["counts"] == [o * i * k * k + o for o, i, k in ((8, 1, 3), (16, 8, 3), (32, 16, 3),
                                                             (5, 32, 1))]
    # the real maps, from the trained detector on the scene
    sc = F.test_scenes()[T.SCENE]
    out = F.det_forward(F.det_params(), sc["canvas"][None, None])[0]
    assert np.allclose(m["out"], out)
    assert np.allclose(m["cell_out"], out[:, 7, 3]) and m["cell_sure_text"] == "0.97"
    assert np.isclose(m["cell_sure"], 1 / (1 + np.exp(-out[0, 7, 3])))
    assert [7, 3] in m["above"] and m["above_n"] == 4
    assert np.allclose(np.asarray(m["maps"][2]).shape, (12, 12))
    # 1 x 1: the cell's 5 numbers are the 32 numbers under it times the 5 x 32 weights
    P = F.det_params()
    W = P["9.weight"].reshape(5, 32)
    assert np.allclose(W @ np.asarray(m["cell_in"]) + P["9.bias"], m["cell_out"], atol=1e-5)
    # what one cell sees: exactly its 22 x 22 square, on the trained model
    assert m["field"] == ([49, 70], [17, 38]) and m["field_side"] == 22
    assert m["outside_change"] == 0.0 and m["ring_change"] > 0.0

"""The numbers behind animations 241-244 (voxanim.study_blocks_u): the detector's two losses,
its training, one box per face, and precision and recall."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_faces

from voxanim import study_blocks_u as S
from voxanim import study_faces as F


# ---------------------------------------------------------------------- no data needed
def test_percentages_and_rounding():
    assert S.pct(38 / 2304) == "1.6%" and S.pct(380 / 2646) == "14.4%"
    assert S.pct(348 / 412) == "84.5%" and S.pct(72 / 412) == "17.5%"
    assert S.two(0.4687) == "0.47" and S.three(0.85312) == "0.853"


def test_smoothed_starts_with_step_0():
    v = np.arange(1000.0)
    xs, ys = S.smoothed(v, 100)
    assert xs[0] == 0 and ys[0] == 0                     # step 0 itself
    assert list(xs[1:3]) == [50, 150] and list(ys[1:3]) == [49.5, 149.5]
    assert len(xs) == len(ys) == 11


def test_area_is_precision_times_each_step_of_recall():
    # 4 faces; boxes surest first: right, right, wrong, right
    d = dict(recall=np.array([0.25, 0.5, 0.5, 0.75]),
             precision=np.array([1.0, 1.0, 2 / 3, 0.75]),
             scores=np.array([0.9, 0.8, 0.6, 0.4]))
    assert S.area(d) == pytest.approx(0.25 + 0.25 + 0.0 + 0.25 * 0.75)
    assert S.at_keep(d, 0.7) == (0.5, 1.0, 2)
    assert S.at_keep(d, 0.5)[2] == 3
    assert S.at_keep(d, 0.95) == (0.0, 1.0, 0)


# ---------------------------------------------------------------------- the faces
@needs_faces
def test_two_losses():
    s = S.two_losses().validate()
    m = s.metadata
    assert (m["scenes"], m["cells"], m["face_cells"], m["empty_cells"]) == (16, 2304, 38, 2266)
    assert (m["face_share_text"], m["weighted_share_text"]) == ("1.6%", "14.4%")
    assert (m["weighted_face_text"], m["weighted_total_text"]) == ("380", "2,646")
    assert (m["there_text"], m["there_unweighted_text"]) == ("0.853", "0.760")
    assert (m["where_text"], m["loss_text"]) == ("0.438", "3.044")
    assert m["loss"] == pytest.approx(m["there"] + 5 * m["where"])
    # the grids are the script's targets of step 0's scenes, one face to a cell
    b = F.det_batch(0)
    has = np.asarray(m["has"])
    for h, sc in zip(has, b["scenes"], strict=True):
        assert np.array_equal(h, F.targets(sc["boxes"])[0])
        assert h.sum() == sc["faces"]
    assert sum(m["per_scene"]) == 38
    # "there" by hand: every cell, a face cell's -log p counted 10 times
    out = F.det_forward(F.det_params(0), b["x"])
    p = 1 / (1 + np.exp(-out[:, 0].astype(np.float64)))
    by_hand = -(10 * has * np.log(p) + (1 - has) * np.log(1 - p)).mean()
    assert by_hand == pytest.approx(m["there"], abs=1e-5)
    assert (m["p_lo_text"], m["p_hi_text"]) == ("0.53", "0.54")
    # the cell worked: what offsets() wants, what the starting weights say, the miss
    c = m["cell"]
    r, col, want = F.offsets(*b["scenes"][0]["boxes"][0])
    assert (c["scene"], c["row"], c["col"]) == (0, r, col) == (0, 5, 3)
    assert np.allclose(c["want"], want)
    assert c["want_text"] == ["0.06", "0.56", "0.66", "0.66"]
    assert c["miss"] == pytest.approx(sum((a - w) ** 2 for a, w in zip(c["said"], c["want"],
                                                                       strict=True)))
    # the thumbnails are the scenes, area-resized to 48
    assert np.allclose(m["thumbs"][3], F.resize(b["x"][3, 0], 48), atol=1e-3)
    # why the weight: the two trained runs on the 200 test scenes
    assert (m["ap_text"], m["ap_unweighted_text"]) == ("0.92", "0.47")
    assert (m["faces"], m["found"], m["found_unweighted"]) == (412, 348, 72)
    assert any("made" in x for x in m["limitations"])


@needs_faces
def test_training():
    s = S.training().validate()
    m = s.metadata
    run = F.det_run()
    assert m["there"][0] == pytest.approx(run["there"][0])
    assert m["there"][-1] == pytest.approx(run["there"][-100:].mean())
    assert len(m["curve_x"]) == 61 and m["curve_x"][-1] == 5950
    assert (m["there_first_text"], m["there_last_text"]) == ("0.85", "0.06")
    assert (m["where_first_text"], m["where_last_text"]) == ("0.44", "0.03")
    assert (m["loss_first_text"], m["loss_last_text"]) == ("3.04", "0.22")
    assert (m["scene"], m["faces"], m["not_faces"]) == (0, 2, 2)
    assert np.array_equal(np.asarray(m["canvas"], np.float32), F.test_scenes()[0]["canvas"])
    snaps = m["snaps"]
    assert [x["step"] for x in snaps] == list(F.DET_SNAPS)
    assert [x["above"] for x in snaps] == [144, 0, 0, 5, 4]
    assert [x["boxes"] for x in snaps[1:]] == [0, 0, 3, 2]
    assert [x["right_count"] for x in snaps[3:]] == [2, 2]
    assert snaps[0]["p_max_text"] == "0.54" and snaps[0]["p_min_text"] == "0.53"
    assert snaps[2]["p_max_text"] == "0.22"
    assert snaps[4]["score_texts"] == ["0.98", "0.97"]
    # the grid is the sigmoid of the detector's first number at that step
    sc = F.test_scenes()[0]
    out = F.det_forward(F.det_params(2000), sc["canvas"][None, None])[0]
    assert np.allclose(snaps[3]["p"], 1 / (1 + np.exp(-out[0].astype(np.float64))), atol=1e-6)
    assert np.allclose(snaps[4]["found"], F.find(F.det_params(), sc["canvas"])["found"])


@needs_faces
def test_one_box_per_face():
    s = S.one_box().validate()
    m = s.metadata
    assert m["scene"] == S.nms_scene() == 0             # the first test scene where NMS drops
    assert (m["n_before"], m["n_after"], m["faces"]) == (4, 2, 2)
    assert [m["score_texts"][j] for j in m["order"]] == ["0.98", "0.97", "0.96", "0.75"]
    assert [m["cells"][j] for j in m["order"]] == [[5, 8], [7, 3], [5, 9], [5, 7]]
    top = m["order"][0]
    assert m["kept"] == [int(k) for k in F.nms(np.asarray(m["corners"]),
                                               np.asarray(m["scores"]))]
    found = F.find(F.det_params(), F.test_scenes()[0]["canvas"])["found"]
    assert np.allclose(np.asarray(m["corners"])[m["kept"]], found)
    dropped = [f for f in m["fate"] if not f["kept"]]
    assert sorted(f["iou_text"] for f in dropped) == ["0.39", "0.77"]
    assert all(f["by"] == top and f["iou"] > 0.3 for f in dropped)
    assert len(m["claimed"]) == 3 and top in m["claimed"]
    # the worked cell: logits to pixels, as decode() does
    w = m["worked"]
    assert (w["row"], w["col"]) == (5, 8)
    assert (w["cx_text"], w["cy_text"], w["w_text"]) == ("69.9", "47.2", "27.9")
    assert np.allclose(m["boxes"][top], [w["cx"], w["cy"], w["w"], w["h"]], atol=1e-4)
    assert m["kept_iou_truth_text"] == ["0.75", "0.71"]
    # over the 200 test scenes: the boxes kept are the evaluation's at 0.5
    d = F.detection()
    assert (m["total_before"], m["total_after"]) == (685, 412) == (685, d["found"] + d["wrong"])


@needs_faces
def test_precision_and_recall():
    s = S.precision_recall().validate()
    m = s.metadata
    assert m["scene"] == S.wrong_scene() == 3
    P = F.det_params()
    for i in range(3):                                   # earlier scenes: no wrong box
        sc = F.test_scenes()[i]
        f = F.find(P, sc["canvas"])
        assert all(g for _s, g in F._match(f["found"], f["scores"], sc["boxes"])[0])
    assert m["right"] == [True, True, False, True]
    assert m["score_texts"] == ["0.95", "0.80", "0.58", "0.53"]
    assert m["iou_texts"] == ["0.84", "0.90", "0.00", "0.73"]
    w, u = m["weighted"], m["unweighted"]
    assert (w["faces"], w["found"], w["wrong"], w["boxes05"]) == (412, 348, 64, 412)
    assert (w["recall05_text"], w["precision05_text"]) == ("84.5%", "84.5%")
    assert (w["ap_text"], u["ap_text"]) == ("0.92", "0.47")
    assert (u["found"], u["recall05_text"], u["precision05_text"]) == (72, "17.5%", "93.5%")
    assert u["last_recall_text"] == "56.8%" and w["last_recall_text"] == "98.3%"
    assert [x["recall_text"] for x in w["marks"]] == ["55.8%", "75.2%", "84.5%", "89.8%",
                                                      "97.8%"]
    assert [x["precision_text"] for x in w["marks"]] == ["99.6%", "93.7%", "84.5%", "69.9%",
                                                         "52.1%"]
    # recall never falls as the confidence falls; AP is the area under the curve
    for c in (w, u):
        assert np.all(np.diff(c["recall"]) >= 0)
        assert S.area(c) == pytest.approx(c["ap"], abs=1e-6)
    assert any("by chance" in x for x in m["limitations"])

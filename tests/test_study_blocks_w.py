"""The numbers behind animations 249-251 (voxanim.study_blocks_w): identification from one
enrolled photo, the two models together on one scene and over 200, and the errors person by
person."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import needs_faces

from voxanim import study_blocks_w as W
from voxanim import study_faces as F
from voxanim.blocks import code_lines

CATALOGUE = Path(__file__).resolve().parent.parent / "voxanim" / "next_catalog.json"


def catalogue():
    return {item["number"]: item for item in json.loads(CATALOGUE.read_text())}


# ---------------------------------------------------------------------- no data needed
def test_the_titles_and_modules_are_the_catalogues():
    items = catalogue()
    assert [items[n]["title"] for n in (249, 250, 251)] == [
        "Who is it?", "Find, cut out, compare", "What 40 people can't tell you"]
    assert {items[n]["module"] for n in (249, 250, 251)} == {"study_blocks_w"}
    assert set(W.STUDIES) == {items[n]["name"] for n in (249, 250, 251)}


def test_formats_and_tags():
    assert W.pct(0.6656891) == "67%" and W.pct(0.8184143) == "82%"
    assert W.pct(0.0191111, 1) == "1.9%" and W.pct(0.12444, 1) == "12.4%"
    assert W.dist(0.7246) == "0.72" and W.dist(0.17984) == "0.18"
    assert W.measured("x") == "measured on Olivetti faces: x" and W.is_tag(W.measured("x"))
    assert W.is_tag(W.about("40 people")) and not W.is_tag("measured on MNIST: x")


def test_the_regions_shown_are_the_scripts():
    assert code_lines("fc-identify")[0] == "gallery = v[:50:10]"
    assert code_lines("fc-strangers")[-1] == 'print("let in:", (d < THRESHOLD).sum())'
    assert code_lines("fc-pipeline")[-1] == "who = int(d.argmin()) + 30"


def test_the_printed_count_is_torchs():
    torch = pytest.importorskip("torch", reason="torch is not installed")
    assert "let in: %s" % torch.tensor(6) == "let in: tensor(6)"


# ---------------------------------------------------------------------- the faces: 249
@needs_faces
def test_the_gallery_and_the_probes_are_the_scripts():
    assert W.gallery_ids().tolist() == [300, 310, 320, 330, 340]       # v[:50:10]
    p = W.probe_ids()
    assert len(p) == 45 and 300 not in p and p[0] == 301 and p[-1] == 349
    assert W.stranger_ids().tolist() == list(range(350, 400))         # v[50:]


@needs_faces
def test_identification_runs_as_the_script_does():
    """The script's own lines (regions "fc-identify" and "fc-strangers") in torch, on the
    study's unit vectors of the 100 test photos, give the same names and the same count."""
    torch = pytest.importorskip("torch", reason="torch is not installed")
    v = torch.tensor(W.vectors()[300:400])
    ns = dict(torch=torch, v=v, THRESHOLD=F.trained_threshold())
    exec("\n".join(code_lines("fc-identify")), ns)
    idn = W.identify()
    assert np.allclose(ns["d"].numpy(), idn["d"], atol=1e-5)
    assert (np.array(F.ENROLLED)[ns["who"].numpy()] == idn["guess"]).all()
    printed = []
    ns["print"] = lambda *a: printed.append(" ".join(str(x) for x in a))
    exec("\n".join(code_lines("fc-strangers")), ns)
    assert printed == ["let in: tensor(6)"]


@needs_faces
def test_identification():
    s = W.identification().validate()
    m = s.metadata
    r = F.recognition()
    assert m["threshold"] == pytest.approx(F.trained_threshold()) == pytest.approx(r["threshold"])
    assert m["threshold_text"] == "0.72"
    # the two probes followed, each picked by its rule
    a, b = m["sure_probe"], m["unsure_probe"]
    assert (a["photo"], a["person"], a["guess"], a["nearest_text"], a["sure"]) == (
        305, 30, 30, "0.18", True)
    assert (b["photo"], b["person"], b["guess"], b["nearest_text"], b["sure"]) == (
        301, 30, 31, "0.81", False)
    sure = np.asarray(m["sure"])
    assert not sure[:a["index"]].any() and sure[a["index"]]
    assert b["index"] == 0 and not sure[0]
    assert a["d_text"] == ["0.18", "1.36", "1.40", "1.08", "1.04"]
    assert b["d_text"] == ["1.00", "0.81", "1.56", "0.88", "1.18"]
    # the counts, as study_faces measured them, and raw pixels' the same
    assert (m["identified"], m["probes"], m["unsure"], m["identified_sure"]) == (36, 45, 14, 30)
    assert (m["identified"], m["unsure"]) == (r["identified"], r["unsure"])
    assert m["pixels_identified"] == F.recognition("pixels")["identified"] == 36
    assert m["unsure_right"] + m["identified_sure"] == m["identified"]
    assert m["sure_wrong"] == 1
    assert np.asarray(m["grid"]).sum() == 36 and np.asarray(m["grid"]).shape == (5, 9)
    assert (m["eer_text"], m["pixels_eer_text"]) == ("12.4%", "22.0%")
    # the strangers
    near = np.asarray(m["stranger_near"])
    assert m["stranger_count"] == 50 and m["let_in"] == r["let_in"] == 6
    assert int((near < m["threshold"]).sum()) == 6
    assert m["stranger_sorted"] == sorted(m["stranger_sorted"])
    assert m["let_in_ids"] == [359, 376, 380, 383, 384, 388]
    assert len(m["gallery_vectors"]) == 5 and m["dims"] == 32
    assert np.allclose(np.linalg.norm(m["gallery_vectors"], axis=1), 1, atol=1e-5)
    assert any("raw pixels" in x and "verification" in x for x in m["limitations"])


@needs_faces
def test_the_threshold_is_chosen_on_the_training_people_only():
    who = F.photos()[1]
    v = W.vectors()[who < F.TRAIN_PEOPLE]
    d, same, _i, _j = F._pairs(v, who[who < F.TRAIN_PEOPLE])
    assert F.eer_threshold(d, same) == pytest.approx(W.identification().metadata["threshold"])


# ---------------------------------------------------------------------- 250
@needs_faces
def test_the_scene_is_picked_by_its_rule():
    rank, counts = W.scene_rank()
    assert rank == 3 and counts == [1, 2, 3]                # the first with three faces
    m = W.pipeline().metadata
    assert m["scene_rank"] == 3 and m["earlier_faces"] == [1, 2]
    assert np.allclose(m["canvas"], F.pipeline_scene()["scene"]["canvas"])


@needs_faces
def test_pipeline():
    s = W.pipeline().validate()
    m = s.metadata
    f = m["faces"]
    assert m["faces_in_scene"] == 3 and len(f) == 3 and m["before"] == 7
    assert [x["person"] for x in f] == [x["truth"] for x in f] == [30, 34, 31]
    assert m["photo_ids"] == [307, 348, 314]               # never an enrolled photo
    assert not set(m["photo_ids"]) & set(m["gallery"])
    assert [x["distance_text"] for x in f] == ["0.34", "0.84", "0.71"]
    assert [x["sure"] for x in f] == [True, False, True]
    assert [x["score_text"] for x in f] == ["0.96", "0.93", "0.92"]
    assert (m["named_right"], m["unsure"]) == (3, 1)
    for x in f:
        assert np.asarray(x["crop"]).shape == (32, 32)
        assert np.isclose(np.linalg.norm(x["q"]), 1, atol=1e-5) and len(x["q"]) == 32
        assert min(x["d"]) == pytest.approx(x["distance"], abs=1e-5)
        assert x["true_person"] == x["truth"]
    assert [round(x["iou"], 2) for x in f] == [0.93, 0.9, 0.73]
    # the rates over 200 scenes, with their own denominators
    assert (m["scenes"], m["all_faces"], m["found_count"]) == (200, 391, 341)
    assert (m["right_detected"], m["right_true"]) == (227, 320)
    assert (m["det_text"], m["true_text"], m["found_text"]) == ("67%", "82%", "87%")
    assert m["det_share"] == pytest.approx(227 / 341)
    assert m["true_share"] == pytest.approx(320 / 391)
    assert m["mean_iou_text"] == "0.76"
    assert any("different denominators" in x for x in m["limitations"])
    assert any("picked by that rule" in x for x in m["limitations"])


@needs_faces
def test_the_crop_is_the_scripts_resize():
    torch = pytest.importorskip("torch", reason="torch is not installed")
    F_ = pytest.importorskip("torch.nn.functional")
    m = W.pipeline().metadata
    canvas = torch.tensor(np.asarray(m["canvas"], np.float32))[None]
    for x in m["faces"]:
        x1, y1, x2, y2 = x["box"]
        crop = canvas[:, y1:y2, x1:x2]
        crop = F_.interpolate(crop[None], size=32, mode="area")
        assert np.allclose(crop[0, 0].numpy(), x["crop"], atol=1e-5)


# ---------------------------------------------------------------------- 251
@needs_faces
def test_limits():
    s = W.limits().validate()
    m = s.metadata
    r = F.recognition()
    assert m["people"] == list(range(30, 40))
    assert m["rejects"] == [25, 2, 9, 0, 17, 33, 0, 3, 0, 22]
    assert sum(m["rejects"]) == r["false_reject"] == 111
    assert m["accepts"] == [7, 17, 2, 29, 8, 11, 34, 32, 28, 4]
    assert sum(m["accepts"]) == 2 * r["false_accept"] == 172     # each pair counts for both
    assert np.allclose(m["frr"], np.asarray(m["rejects"]) / 45)
    assert np.allclose(m["far"], np.asarray(m["accepts"]) / 900)
    assert (m["worst"], m["worst_text"], m["best_count"]) == (35, "73%", 3)
    assert (m["frr_all_text"], m["far_all_text"]) == ("25%", "1.9%")
    assert (m["far_lo_text"], m["far_hi_text"]) == ("0.2%", "3.8%")
    assert (m["people_all"], m["photos_each"], m["test_people"]) == (40, 10, 10)
    assert m["cannot"] == ["ages", "skin tones", "lighting", "cameras", "poses"]
    assert any("still err" in x for x in m["limitations"])

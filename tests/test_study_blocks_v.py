"""The numbers behind animations 245-248 (voxanim.study_blocks_v): a face as numbers, the
encoder, the triplet loss and verification, all from voxanim.study_faces."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_faces

from voxanim import study_blocks_v as S
from voxanim import study_faces as F


# ---------------------------------------------------------------------- no data needed
def test_formats():
    assert S.pct(0.12444) == "12.4%" and S.pct(0.22044) == "22.0%" and S.pct(0.18) == "18.0%"
    assert S.dist(0.7246) == "0.72" and S.dist(0.34026, 3) == "0.340"
    assert S.means(np.arange(50.0), 25).tolist() == [12.0, 37.0]


def test_histogram_keeps_every_value():
    d = np.array([0.0, 0.01, 0.5, 2.0, 2.5])
    h = S.histogram(d, S.BINS)
    assert h.sum() == len(d) and h[0] == 2 and h[-1] == 2


# ---------------------------------------------------------------------- the faces
@needs_faces
def test_a_face_is_numbers():
    s = S.a_face_is_numbers().validate()
    m = s.metadata
    x = F.photos()[0]
    assert (m["photos"], m["people"], m["per_person"], m["numbers"]) == (400, 40, 10, 4096)
    assert m["numbers_text"] == "4,096" and m["photos_shape"] == "[400, 64, 64]"
    assert (m["photo"], m["photo_person"]) == (300, 30)
    assert np.array_equal(m["image"], x[300])
    r0, c0 = m["patch_at"]
    assert np.array_equal(m["patch"], x[300][r0:r0 + 3, c0:c0 + 3])
    assert m["patch_text"][0] == ["%.2f" % v for v in x[300][r0, c0:c0 + 3]]
    assert m["lo_value"] >= 0.0 and m["hi_value"] <= 1.0
    # the thumbnails are the photos averaged down, as F.interpolate(mode="area") does
    assert m["thumbs"].shape == (400, 16, 16)
    assert np.allclose(m["thumbs"][123], x[123].reshape(16, 4, 16, 4).mean((1, 3)), atol=1e-6)
    # the two pairs: one person, then two people; the two people are the closer
    assert m["same_people"] == [30, 30] and m["diff_people"] == [30, 31]
    assert (m["same_text"], m["diff_text"]) == ("0.340", "0.336")
    assert m["diff_distance"] < m["same_distance"]
    v = x.reshape(400, -1) / np.linalg.norm(x.reshape(400, -1), axis=1, keepdims=True)
    assert m["same_distance"] == pytest.approx(float(np.linalg.norm(v[300] - v[301])), abs=1e-5)
    px = F.recognition("pixels")
    assert np.isclose(px["d_same"], m["same_distance"], atol=1e-5).any()
    assert np.isclose(px["d_diff"], m["diff_distance"], atol=1e-5).any()
    # all the pairs of people 30-39: 450 of one person, 4,500 of two; they overlap
    assert (m["pairs"], m["same_pairs"], m["diff_pairs"]) == (4950, 450, 4500)
    assert m["hist_same"].sum() == 450 and m["hist_diff"].sum() == 4500
    assert m["same_range_text"] == ("0.06", "0.42") and m["diff_range_text"] == ("0.17", "0.52")
    assert m["overlap_text"] == ("0.17", "0.42")
    assert m["overlap"][0] < m["overlap"][1]          # the piles do overlap
    assert (m["in_overlap_same"], m["in_overlap_diff"]) == (377, 4158)
    assert any("rule" in x for x in m["limitations"])


@needs_faces
def test_an_embedding():
    s = S.an_embedding().validate()
    m = s.metadata
    assert [c["total"] for c in m["counts"]] == [80, 1168, 4640, 16416]
    assert m["counts"][0]["weights"] == 8 * 1 * 3 * 3 and m["counts"][3]["weights"] == 512 * 32
    assert m["total"] == 22304 == F.count(F.rec_init()) and m["total_text"] == "22,304"
    assert m["shapes"] == ["[1, 32, 32]", "[8, 16, 16]", "[16, 8, 8]", "[32, 4, 4]"]
    assert m["sizes"] == [1024, 2048, 1024, 512] and m["flat_len"] == 512 and m["dims"] == 32
    # the 32 numbers: the trained encoder's, before and after scaling to length 1
    e = F.rec_forward(F.rec_params(), F.small()[300:301])[0]
    assert np.allclose(m["e"], e, atol=1e-6)
    assert np.allclose(m["v"], F.embed(F.rec_params(), F.small())[300], atol=1e-6)
    assert m["length_text"] == "0.09" and m["unit_length_text"] == "1.00"
    assert np.isclose(np.linalg.norm(m["v"]), 1.0, atol=1e-5)
    assert np.allclose(m["v"] * m["length"], m["e"], atol=1e-6)
    # the maps on the front cards: block outputs of that photo, the widest spread of each
    assert [np.shape(x) for x in m["maps"]] == [(16, 16), (8, 8), (4, 4)]
    assert np.array_equal(m["flat"].reshape(32, 4, 4)[m["map_index"][2]], m["maps"][2])
    assert all(np.ptp(x) > 0 for x in m["maps"])
    assert m["code_normalize"] == "a = F.normalize(encoder(anchor))"


@needs_faces
def test_triplets():
    s = S.triplets_study().validate()
    m = s.metadata
    i, j, k = F.triplets(0)
    assert m["ids"] == [int(i[0]), int(j[0]), int(k[0])] == [144, 149, 30]
    assert m["people"] == [14, 14, 3]
    assert all(p < F.TRAIN_PEOPLE for p in m["people"])     # training people only
    b, a = m["before"], m["after"]
    assert (b["ap_text"], b["an_text"], b["loss_text"]) == ("0.04", "0.09", "0.45")
    assert (a["ap_text"], a["an_text"], a["loss"]) == ("0.24", "1.27", 0.0)
    for d in (b, a):                                     # the loss of one triplet
        assert d["loss"] == pytest.approx(max(d["ap"] - d["an"] + 0.5, 0.0), abs=1e-6)
    assert b["short"] and not a["short"]
    # the curves: the run's numbers, means of 25 steps
    run = F.rec_run()
    assert m["steps"] == 1500 and len(m["loss_means"]) == 60
    assert m["loss_means"][0] == pytest.approx(run["loss"][:25].mean())
    assert m["active_means"][-1] == pytest.approx(run["active"][-25:].mean())
    assert m["first_loss_text"] == "0.45" and m["first_active_text"] == "100%"
    assert m["first_short"] == 32
    assert m["last_active_text"] == "1.7%"
    assert m["curve_steps"][0] == 25 and m["curve_steps"][-1] == 1500


@needs_faces
def test_verification():
    s = S.verification().validate()
    m = s.metadata
    assert (m["pairs"], m["same_pairs"], m["diff_pairs"]) == (4950, 450, 4500)
    assert m["photos"] == 100 and m["test_people"] == (30, 39)
    assert m["hist_same"].sum() == 450 and m["hist_diff"].sum() == 4500
    # the threshold comes from the training people's pairs only
    who = F.photos()[1]
    v = F.embed(F.rec_params(), F.small())
    tr = who < F.TRAIN_PEOPLE
    d, same, _i, _j = F._pairs(v[tr], who[tr])
    assert m["threshold"] == pytest.approx(F.eer_threshold(d, same))
    assert m["threshold_text"] == "0.72"
    assert (m["false_accept"], m["false_reject"], m["right"]) == (86, 111, 4753)
    assert m["false_accept"] + m["false_reject"] + m["right"] == 4950
    assert (m["far_text"], m["frr_text"]) == ("1.9%", "24.7%")
    assert m["eer_text"] == dict(trained="12.4%", pixels="22.0%", untrained="18.0%")
    assert m["mistakes"] == dict(trained=197, pixels=390, untrained=534)
    assert m["printed"] == "right: tensor(4753)"
    # the unseen people's 100 photos, in person order: a row of the sheet is one person
    assert np.array_equal(m["thumbs"], S.thumbnails()[300:400])
    assert any("not a threshold" in x for x in m["limitations"])

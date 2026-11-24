"""The numbers behind animation 277 (voxanim.study_blocks_zi): unfolding a made-up spectrum.
Everything comes from voxanim.study_unfolding (the numpy twin of examples/unfolding.py, checked
against the script by tests/test_study_unfolding.py); the claims shown on screen are checked
here. No dataset is needed: the events are made up."""

from __future__ import annotations

import numpy as np

from voxanim import blocks as B
from voxanim import study_blocks_zi as Z
from voxanim import study_unfolding as S
from voxanim.next_catalog import ITEMS, build_study


def meta():
    return build_study("study_blocks_zi", "unfolding").metadata


def test_the_numbers_are_written_as_the_screen_writes_them():
    assert Z.signed(-899.7) == "−900" and Z.signed(3475.14) == "3,475" and Z.signed(0.2) == "0"
    assert Z.sci(3.637978807091713e-12) == "3.6×10⁻¹²"


def test_the_study_is_registered_with_four_panels():
    study = build_study("study_blocks_zi", "unfolding").validate()
    assert len(study.panels) == 4
    assert [p.kind for p in study.panels] == ["matrix", "curve", "bars", "curve"]
    item = next(i for i in ITEMS if i["name"] == "unfolding")
    assert item["module"] == "study_blocks_zi" and item["number"] == 277
    assert study.title == item["title"]
    assert "1,166" in study.subtitle and "bias 41, spread 53" in study.subtitle


def test_the_truth_and_r_on_screen():
    m = meta()
    assert m["truth_text"][0] == "2,519" and m["truth_text"][5] == "1,194"
    assert m["bump_bin"] == 5 and m["truth_total_text"] == "10,000"
    assert m["simulated_text"] == "1,000,000" and m["counted_text"] == "582,195"
    assert m["in_range_text"] == "964,628"
    R = np.asarray(m["R"])
    np.testing.assert_allclose(R, np.asarray(m["H"]) / np.asarray(m["n"]))
    # the dots are found pairs, measured inside 0-100 GeV, the first 500 of the simulation
    t, r = np.asarray(m["sim_true"]), np.asarray(m["sim_measured"])
    assert m["dots"] == len(t) == Z.DOTS
    assert ((t >= 0) & (t < 100) & (r >= 0) & (r < 100)).all()
    w = S.world()
    kept = w["found"] & (w["r"] >= 0) & (w["r"] < 100)
    np.testing.assert_array_equal(t, w["t"][kept][:Z.DOTS])


def test_measured_inverted_and_why():
    m = meta()
    assert m["expected_total_text"] == "6,345" and m["data_total_text"] == "6,362"
    assert m["lost_text"] == "37%" and m["data_noise_text"] == "25"
    assert np.allclose(np.asarray(m["expected"]), np.asarray(m["R"]) @ np.asarray(m["truth"]))
    assert m["back_off_text"] == "3.6×10⁻¹²"
    assert m["naive_min_text"] == "−900" and m["naive_max_text"] == "3,475"
    assert m["naive_negative"] == 3
    assert m["s_text"] == ["0.75", "0.62", "0.49", "0.36", "0.24", "0.15", "0.082", "0.041",
                           "0.019", "0.0078"]
    assert m["blown_text"][-3:] == ["1,789", "−2,319", "2,759"]
    # the last three directions make nearly all of the swing
    assert m["last3_share"] > 0.99 and m["last3_share_text"] == "99.9%"
    blown = np.asarray(m["blown"])
    assert np.all(np.abs(blown[:7]) < 120) and np.all(np.abs(blown[7:]) > 1700)


def test_iterations_and_pseudo_experiments():
    m = meta()
    its = np.asarray(m["iterates"])
    assert its.shape == (11, 10)
    np.testing.assert_allclose(its[0], m["prior"])
    # the prior has no bump; the bump's bin climbs to the truth's
    assert [round(v) for v in its[[0, 1, 4, 10], 5]] == [587, 909, 1121, 1198]
    assert abs(its[10, 5] - m["truth"][5]) < 10
    assert m["best"] == 10 and m["near_best"] == [8, 13]
    assert (m["bias_text"][10], m["spread_text"][10]) == ("41", "53")
    assert m["inverse_spread_text"] == "1,166" and m["inverse_spread_exact_text"] == "1,180"
    assert m["ratio_text"] == "46" and m["inverse_negative_text"] == "90%"
    assert m["printed"] == S.printed()
    for name, lines in m["code"].items():
        assert lines == list(B.code_lines(name))
        assert B.code_file(name) == "examples/unfolding.py"

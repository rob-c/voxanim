"""The numbers behind animations 269-270 (voxanim.study_blocks_zd): Minuit's calls and SGD's walk
on one small fit. Everything comes from voxanim.study_minuit (the numpy twin of
examples/minuit_vs_sgd.py and the record of iminuit's own calls); the claims shown on screen
are checked against it here. No dataset is needed: the data are made up by the script."""

from __future__ import annotations

import math

import numpy as np

from voxanim import blocks as B
from voxanim import study_blocks_zd as Z
from voxanim import study_minuit as M
from voxanim.next_catalog import build_study


def feels():
    return build_study("study_blocks_zd", "minuit-feels-its-way").metadata


def sgd():
    return build_study("study_blocks_zd", "minuit-and-sgd").metadata


# ---------------------------------------------------------------------- helpers
def test_the_numbers_are_written_as_the_screen_writes_them():
    assert Z.chi2_text(2691.6126) == "2,691.61" and Z.chi2_text(108.9191) == "108.92"
    assert Z.chi2_text(33.944675) == "33.9447"
    assert [Z.short(v) for v in (2691.6, 108.9, 35.30, 34.01, 33.94)] == [
        "2,692", "109", "35.3", "34.0", "33.9"]
    assert Z.delta_text(-0.10601) == "−0.11" and Z.delta_text(4.848) == "+4.8"
    assert Z.delta_text(762.24) == "+762" and Z.delta_text(-2582.7) == "−2,583"
    assert Z.delta_text(0.00055) == "+0.00055" and Z.delta_text(-0.00009) == "flat"
    assert Z.step_text(0.00025335) == "+0.00025" and Z.step_text(-0.0025335) == "−0.0025"
    assert Z.step_text(7.5e-05) == "+0.000075"
    assert [Z.change(d) for d in (-1.0, 1.0, 5e-5, -5e-5)] == ["lower", "higher", "flat",
                                                                 "flat"]
    assert Z.walk_indices(3001)[:3] == [0, 1, 2] and Z.walk_indices(3001)[-1] == 3000
    assert Z.walk_indices(3001)[200:202] == [200, 205]


# ---------------------------------------------------------------------- 269
def test_every_call_is_labelled_and_in_iminuits_order():
    tl = Z.timeline()
    calls = M.calls()
    assert len(tl) == len(calls) == 81
    for e, row in zip(tl, calls, strict=True):
        assert e["point"] == row[:3].tolist() and e["chi2"] == row[3]
    kinds = [e["kind"] for e in tl]
    assert kinds.count("probe") == 52 and kinds.count("try") == 12 and kinds.count("hesse") == 16
    assert sum(e["kept"] for e in tl if e["kind"] == "try") == 5
    assert kinds[:65].count("hesse") == 0 and set(kinds[65:]) == {"hesse"}


def test_a_probe_moves_one_parameter_from_the_point_it_is_asked_from():
    tl = Z.timeline()
    for e in tl:
        base = tl[e["base"]]
        assert base["kind"] == "start" or (base["kind"] == "try" and base["kept"])
        d = np.asarray(e["point"]) - np.asarray(base["point"])
        np.testing.assert_allclose(d, e["d"])
        assert math.isclose(e["delta"], e["chi2"] - base["chi2"])
        if e["kind"] == "probe":
            moved = np.nonzero(np.abs(d) > 1e-12)[0]
            assert moved.tolist() == [e["axis"]]
            assert np.sign(d[e["axis"]]) == e["sign"]
            assert np.abs(d).max() < 0.003                # far too small to see on the path
            assert e["near"]
        assert e["change"] == Z.change(e["delta"])


def test_the_first_probes_are_up_and_down_each_axis_two_step_sizes():
    tl = Z.timeline()
    first = tl[1:13]
    assert all(e["kind"] == "probe" and e["base"] == 0 for e in first)
    assert [e["axis"] for e in first] == [0] * 4 + [1] * 4 + [2] * 4
    assert [e["sign"] for e in first] == [1, -1] * 6
    sizes = [abs(e["d"][e["axis"]]) for e in first]
    assert [round(s, 5) for s in sizes[:4]] == [0.00025, 0.00025, 0.00253, 0.00253]
    assert [round(s, 5) for s in sizes[8:]] == [0.00253, 0.00253, 0.0012, 0.0012]
    # each pair: up lowers chi-square, down raises it (the slope points down every way)
    assert [e["change"] for e in first] == ["lower", "higher"] * 6
    m = feels()
    assert m["first_probes"] == 12 and m["first_steps"] == [0.000253, 0.001202, 0.002534]


def test_the_first_step_overshoots_and_the_line_search_comes_back():
    m = feels()
    tl = m["timeline"]
    assert tl[13]["kind"] == "try" and not tl[13]["kept"]
    assert tl[14]["kind"] == "try" and tl[14]["kept"]
    assert m["overshoot_text"] == "3,454" and m["back_text"] == "109"
    assert tl[13]["chi2"] > tl[0]["chi2"] > tl[14]["chi2"]
    # the way back is on the line to the overshoot
    p0, p1, p2 = (np.asarray(tl[i]["point"]) for i in (0, 13, 14))
    t = (p2 - p0) / (p1 - p0)
    assert np.allclose(t, t[0], atol=1e-6) and 0 < t[0] < 1
    # then the probes again round the kept point: a round per kept try, then HESSE's
    assert [e["round"] for e in tl[15:25]] == [1] * 10
    assert m["rounds"] == 7 and m["kept_text"] == ["2,692", "109", "35.3", "34.0", "33.9",
                                                   "33.9"]


def test_the_minimum_the_edm_and_the_errors():
    m = feels()
    assert (m["migrad_calls"], m["probes"], m["tries"], m["kept"], m["hesse"]) == (
        65, 52, 12, 5, 16)
    assert m["values_text"] == ["5.14", "2.41", "1.007"]
    assert m["errors_text"] == ["0.18", "0.21", "0.11"]
    assert m["fval_text"] == "33.9" and m["edm_text"] == "0.00014"
    assert m["start_text"] == "2,692" and m["iminuit"] == "2.33.0"
    assert m["printed"] == "MIGRAD: 65 calls; HESSE: 16 more"
    assert m["point_evaluations"] == 65 * 40 == 2600
    # the record's minimum is the chi-square there, recomputed
    assert math.isclose(M.chi2(m["values"]), m["fval"], rel_tol=1e-9)
    # every probe round the last kept point changes chi-square by less than 0.0001: flat
    tl = m["timeline"]
    assert all(e["change"] == "flat" for e in tl if e["base"] == 49)
    assert tl[49]["point"] == tl[42]["point"]           # kept again where it already was


def test_the_curves_are_the_models_at_each_point_asked():
    m = feels()
    grid = np.asarray(m["grid"])
    for i, ys in m["curves"].items():
        p = m["timeline"][i]["point"]
        np.testing.assert_allclose(ys, p[0] * np.exp(-grid / p[1]) + p[2])
    assert sorted(m["curves"]) == [0] + [e["i"] for e in m["timeline"] if e["kind"] == "try"]
    x, y = M.data()
    assert m["x"] == x.tolist() and m["y"] == y.tolist() and len(m["x"]) == 40


def test_the_code_is_the_scripts():
    m = feels()
    assert m["code_migrad"][0] == "m = Minuit(chi2, a=1.0, tau=1.0, b=0.0)"
    assert "m.migrad()" in m["code_migrad"] and "m.hesse()" in m["code_migrad"]
    assert m["code_model"][1].strip() == "return a * np.exp(-x / tau) + b"
    assert B.code_file("mn-chi2") == "examples/minuit_vs_sgd.py"


# ---------------------------------------------------------------------- 270
def test_sgds_first_step_is_a_batch_of_8_and_a_small_step():
    m = sgd()
    f = m["first"]
    assert len(f["idx"]) == 8 == len(set(f["idx"])) and all(0 <= i < 40 for i in f["idx"])
    np.testing.assert_allclose(f["p1"], m["path"][1])
    np.testing.assert_allclose(f["step"], -0.005 * np.asarray(f["grad"]))
    g, _ = M.gradient(np.array(M.START), np.array(f["idx"]))
    np.testing.assert_allclose(g, f["grad"])
    assert (m["steps"], m["batch"], m["lr"], m["seed"]) == (3000, 8, 0.005, 7)


def test_the_walk_ends_wandering_and_its_average_is_near_minuits_minimum():
    m = sgd()
    path = np.asarray(m["path"])
    assert path.shape == (3001, 3) and path[0].tolist() == [1.0, 1.0, 0.0]
    tail = path[-1000:]
    np.testing.assert_allclose(m["sgd_average"], tail.mean(0))
    assert m["average_text"] == ["5.131", "2.399", "1.020"]
    assert m["average_chi2_text"] == "33.96" and m["minuit_chi2_text"] == "33.94"
    assert m["sgd_average_chi2"] > m["minuit_chi2"]      # Minuit's is the minimum
    assert m["spread_text"] == "0.02"
    assert all(0.05 < r < 0.2 for r in m["spread_ratio"])  # about a tenth of the errors
    # the end itself is further off than the average: the noise averages out
    assert m["sgd_end_chi2"] > m["sgd_average_chi2"]
    early = np.linalg.norm(np.diff(path[:20], axis=0), axis=1).mean()
    late = np.linalg.norm(np.diff(tail, axis=0), axis=1).mean()
    assert early > 10 * late                              # big steps first, small ones last


def test_the_counts():
    m = sgd()
    assert m["sgd_evaluations"] == 3000 * 8 == 24000
    assert m["minuit_evaluations"] == 65 * 40 == 2600
    assert (m["sgd_evaluations_text"], m["minuit_evaluations_text"]) == ("24,000", "2,600")
    assert m["code_sgd"][-1] == "p = p - lr * g"


def test_four_panels_each():
    for name in ("minuit-feels-its-way", "minuit-and-sgd"):
        s = build_study("study_blocks_zd", name)
        assert len(s.panels) == 4 and s.title and s.subtitle
        assert {p.kind for p in s.panels} <= {"curve", "bars", "matrix", "points", "images"}

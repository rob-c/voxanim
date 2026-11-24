"""The numbers behind animations 273-274 (voxanim.study_blocks_zg): every number shown is
voxanim.study_likelihood_ratio's, written as the screen writes it; each study has four panels.
No dataset is needed: the events are made up by the script."""

from __future__ import annotations

import numpy as np

from voxanim import study_blocks_zg as Z
from voxanim import study_likelihood_ratio as L
from voxanim.next_catalog import build_study

NAMES = ("classifier-learns-the-likelihood-ratio", "reweighting-with-a-classifier")


def meta(name):
    return build_study("study_blocks_zg", name).metadata


def test_the_numbers_are_written_as_the_screen_writes_them():
    assert [Z.num(v) for v in (0.2404, 0.0604, 3.895, 151.14, 38.04, 0.006)] == [
        "0.24", "0.060", "3.9", "151", "38", "0.006"]
    assert Z.pct(0.19254) == "19.3%" and Z.pct(0.0986) == "9.9%"


def test_four_panels_each():
    for name in NAMES:
        s = build_study("study_blocks_zg", name)
        assert len(s.panels) == 4 and s.validate() is s
        assert {p.kind for p in s.panels} <= {"curve", "points"}


def test_273_is_the_studys():
    m = meta(NAMES[0])
    o = L.one()
    assert m["np_text"] == "19.3%" and m["cut_text"] == "9.9%" and m["net_text"] == "19.6%"
    assert m["own_text"] == "19.0%" and m["err_text"] == "0.3"
    assert m["last_text"] == "%.4f" % o["last_loss"] == "0.4691"
    assert m["floor_text"] == "0.4692" and m["first_text"] == "0.682"
    assert m["far_learned_text"] == "0.24" and m["far_exact_text"] == "0.060"
    assert (m["beyond_far0"], m["beyond_far1"]) == (20, 2)
    assert m["within_text"] == "10%" and m["worst_inside"] < 0.1
    assert m["peak_text"] == "3.9" and m["peak_x_text"] == "1.33"
    assert m["printed_power"] == "0.196"
    grid = np.asarray(m["grid"])
    assert grid[0] == -2.0 and grid[-1] == 4.0
    np.testing.assert_allclose(np.log(m["exact"]), L.log_ratio(grid), atol=1e-9)


def test_274_is_the_studys():
    m = meta(NAMES[1])
    t = L.two()
    assert m["w_max_text"] == "38" and m["w_max_exact_text"] == "151"
    assert m["n_eff_text"] == "30,400" and m["n_text"] == "50,000"
    assert m["empty_y"] == 38 and m["empty_bins"] == [0, 1]
    assert m["bulk_pct"] == "5%" and m["few_pct"] == "24%"
    h = m["hists"]
    assert (h["x"]["off_middle_text"], h["y"]["off_middle_text"],
            h["x + y"]["off_middle_text"]) == ("12%", "14%", "12%")
    assert h["x + y"]["off_before_text"] == "80%"
    assert h["y"]["middle_share"] > 0.95
    np.testing.assert_allclose(h["x"]["weighted"], t["hists"]["x"]["weighted"])
    # every event with an exact weight above 5 is in the scatter
    assert len(m["scatter_exact"]) == Z.SCATTER + int((t["exact_weights"] > Z.BIG_W).sum())
    assert max(m["scatter_exact"]) == t["w_max_exact"]

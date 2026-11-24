"""voxanim.study_weights is examples/weights_and_scale_factors.py: the script run again makes the
same events, weights, classifier, cut, control region and data, number for number; and the
arithmetic and the measured claims of the two animations hold. No dataset: the data are made
up by the script."""

from __future__ import annotations

import contextlib
import io
import math
import runpy

import numpy as np
import pytest

from voxanim import blocks
from voxanim import study_weights as W

SCRIPT = blocks.EXAMPLES / "weights_and_scale_factors.py"


@pytest.fixture(scope="module")
def ns():
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(SCRIPT), run_name="test")


# ---------------------------------------------------------------------- the twin is the script
def test_the_samples_and_weights_are_the_scripts(ns):
    assert ns["L"] == W.L and ns["samples"] == W.SAMPLES and ns["SHAPES"] == W.SHAPES
    x, w = W.simulated(), W.weights()
    for k in W.NAMES:
        np.testing.assert_array_equal(ns["x"][k], x[k])
        np.testing.assert_array_equal(ns["w"][k], w[k])
    np.testing.assert_array_equal(ns["edges"], W.EDGES)
    assert ns["cut"] == W.cut()


def test_the_training_is_the_scripts(ns):
    for weighted in (False, True):
        a, b, losses = ns["train"](weighted)
        t = W.train(weighted)
        assert a == t["a"] and b == t["b"]
        np.testing.assert_array_equal(losses, t["losses"])


def test_the_efficiency_and_negative_weights_are_the_scripts(ns):
    e = W.efficiency()
    assert ns["keep"].mean() == e["counted"]
    assert ns["wb"][ns["keep"]].sum() / ns["wb"].sum() == e["weighted"]
    sign, wn, n_eff = ns["negative"]()
    ng = W.negative()
    np.testing.assert_array_equal(sign, ng["sign"])
    np.testing.assert_array_equal(wn, ng["w"])
    assert n_eff == ng["n_eff"]


def test_the_made_up_data_and_scale_factor_are_the_scripts(ns):
    d, c, r = W.made_up_data(), W.control(), W.signal_region()
    np.testing.assert_array_equal(ns["probe_mc"], d["probe_mc"])
    np.testing.assert_array_equal(ns["probe_data"], d["probe_data"])
    np.testing.assert_array_equal(ns["data"], d["data"])
    assert (ns["eff_mc"], ns["eff_data"], ns["sf"], ns["err"]) == (
        c["eff_mc"], c["eff_data"], c["sf"], c["sf_err"])
    assert (ns["predicted"], ns["corrected"], ns["observed"]) == (
        r["predicted"], r["corrected"], r["observed"])


# ---------------------------------------------------------------------- the arithmetic
def test_a_weight_is_cross_section_times_luminosity_over_events_made():
    assert [W.weight(k) for k in W.NAMES] == [0.25, 0.4, 1.2]
    assert [W.expected(k) for k in W.NAMES] == [5000.0, 20000.0, 12000.0]
    h = W.histograms()
    assert h["counts"].sum(1).tolist() == [20000, 50000, 10000]       # none outside the bins
    np.testing.assert_allclose(h["weighted"].sum(1), [5000, 20000, 12000], rtol=1e-12)


def test_negative_weights_keep_the_sum_and_cost_events():
    ng = W.negative()
    assert (ng["pos"], ng["neg"], ng["weight"]) == (8000, 2000, 2.0)
    assert math.isclose(ng["w"].sum(), 12000.0)
    assert ng["n_eff"] == (8000 - 2000) ** 2 / 10000 == 3600.0      # (sum w)^2 / sum w^2
    assert math.isclose(ng["n_eff_plain"], 10000.0)                  # equal weights: N
    np.testing.assert_allclose(ng["net"], ng["hist_pos"] + ng["hist_neg"])
    assert (ng["hist_neg"] <= 0).all() and (ng["hist_pos"] >= 0).all()
    # each bin's error grows by sqrt(10,000 / 3,600) = 5/3 (all weights are +-2 against 1.2)
    nz = ng["err_plain"] > 0
    np.testing.assert_allclose(ng["err"][nz] / ng["err_plain"][nz], 2.0 / 1.2)


def test_the_scale_factor_and_its_error():
    c = W.control()
    assert c["sf"] == c["eff_data"] / c["eff_mc"]
    rel = math.hypot(c["err_data"] / c["eff_data"], c["err_mc"] / c["eff_mc"])
    assert math.isclose(c["sf_err"], c["sf"] * rel)


# ---------------------------------------------------------------------- the measured claims
def test_the_cut_keeps_80_percent_of_simulated_signal():
    e = W.efficiency()
    assert e["each"]["signal"] == 0.8
    assert round(e["cut"], 3) == 0.915


def test_trained_with_weights_the_mean_score_is_the_weighted_share():
    un, we = W.train(False), W.train(True)
    assert math.isclose(un["losses"][0], math.log(2)) and math.isclose(we["losses"][0],
                                                                          math.log(2))
    assert un["share"] == 0.25 and math.isclose(we["share"], 5000 / 37000)
    assert abs(un["mean_score"] - un["share"]) < 1e-4
    assert abs(we["mean_score"] - we["share"]) < 1e-4
    assert un["half"] < we["half"]                    # with the weights it is more cautious
    for t in (un, we):                                 # a large rate: it bounces at first,
        assert (np.diff(t["losses"][40:]) <= 1e-12).all()   # then every step goes down


def test_weighted_efficiency_differs_from_counted():
    e = W.efficiency()
    assert e["each"]["bkg 2"] > 10 * e["each"]["bkg 1"]
    assert round(e["counted"], 3) == 0.018 and round(e["weighted"], 3) == 0.035


def test_the_data_need_the_scale_factor():
    c, r = W.control(), W.signal_region()
    assert round(c["eff_mc"], 3) == 0.805 and round(c["eff_data"], 3) == 0.745
    assert round(c["sf"], 3) == 0.925 and round(c["sf_err"], 3) == 0.008
    assert math.isclose(r["predicted"], 5123.6) and r["observed"] == 4739
    assert r["pull_before"] > 4 and abs(r["pull_after"]) < 1
    assert math.isclose(r["corrected"], r["background"] + c["sf"] * r["signal"])

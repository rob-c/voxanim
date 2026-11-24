"""voxanim.study_ensembles is examples/ensembles.py in numpy (its runs, its resamples, HESSE's
errors), and the measured claims of "Error bars on a network" hold: new seeds spread by the
optimiser's noise only; resamples spread near HESSE's errors in a and b and short in tau."""

from __future__ import annotations

import runpy

import numpy as np
import pytest

from voxanim import blocks
from voxanim import study_ensembles as E
from voxanim import study_minuit as M

SCRIPT = str(blocks.EXAMPLES / "ensembles.py")


@pytest.fixture(scope="module")
def ns():
    pytest.importorskip("iminuit", reason="examples/ensembles.py imports iminuit")
    return runpy.run_path(SCRIPT, run_name="test")


# ---------------------------------------------------------------------- the twin is the script
def test_the_data_are_minuit_and_sgds(ns):
    x, y = M.data()
    np.testing.assert_array_equal(ns["x"], x)
    np.testing.assert_array_equal(ns["y"], y)


def test_the_runs_are_the_scripts(ns):
    """The script's train, run by run, against the study's runs side by side."""
    everyone = np.arange(M.N)
    for r in (0, 1, 57):
        np.testing.assert_allclose(ns["train"](everyone, E.SEEDS[r]), E.seeds()[r], rtol=0,
                                   atol=1e-12)
    draw = np.random.default_rng(E.DRAW_SEED)
    picks = [draw.integers(0, 40, 40) for _ in range(E.RUNS)]
    np.testing.assert_array_equal(np.array(picks), E.resamples())
    for r in (0, 1, 99):
        np.testing.assert_allclose(ns["train"](picks[r], E.BOOT_SEEDS[r]), E.boots()[r],
                                   rtol=0, atol=1e-12)
    # one run, alone, is the same as one of many side by side
    np.testing.assert_allclose(E.train(everyone, 0), E.seeds()[0], rtol=0, atol=1e-12)


def test_hesse_is_the_record(ns):
    import iminuit

    if iminuit.__version__ != M.record()["iminuit"]:
        pytest.skip("iminuit %s installed; the record is of %s"
                    % (iminuit.__version__, M.record()["iminuit"]))
    m = ns["run_minuit"]()
    np.testing.assert_allclose(list(m.values), M.record()["values"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(list(m.errors), M.record()["errors"], rtol=1e-9)
    np.testing.assert_allclose(np.array(m.covariance), M.record()["covariance"], rtol=1e-9)


def test_the_printed_numbers_are_numpys():
    p = E.printed()
    c = E.compare()
    assert p["seeds"] == str(c["seeds_spread"].round(3)) == "[0.017 0.016 0.021]"
    assert p["boots"] == "[0.15 0.14 0.11]"
    assert p["hesse"] == "[0.18 0.21 0.11]"
    assert p["cov"].splitlines()[0] == "[[ 0.022  0.001 -0.008]"


# ---------------------------------------------------------------------- what was measured
def test_new_seeds_show_only_the_optimisers_noise():
    c = E.compare()
    assert (c["seeds_spread"] < 0.025).all()
    assert (c["seeds_spread"] < 0.2 * c["hesse"]).all()           # a tenth to a fifth
    # every run arrives: its chi-square on the 40 points is within 3 of Minuit's
    assert max(M.chi2(p) for p in c["seeds"]) < E.minuit()["fval"] + 3
    assert np.abs(c["seeds_mean"] - c["values"]).max() < 0.01


def test_resamples_come_near_hesse_in_a_and_b_and_short_in_tau():
    c = E.compare()
    a, tau, b = c["ratio"]
    assert 0.75 < a < 0.9 and 0.9 < b < 1.05            # near
    assert 0.6 < tau < 0.72                             # about a third short
    assert (c["boots_spread"] > 5 * c["seeds_spread"]).all()
    # the ellipses tilt alike
    assert c["corr_tau_b_boots"] < -0.75 and c["corr_tau_b_hesse"] < -0.75
    assert abs(c["corr_tau_b_boots"] - c["corr_tau_b_hesse"]) < 0.05


def test_the_short_tau_is_the_bootstraps_not_sgds():
    c = E.compare()
    # SGD's answers are the resamples' own minima, give or take the optimiser's noise
    np.testing.assert_allclose(c["exact_spread"], c["boots_spread"], rtol=0.12)
    assert c["exact_spread"][1] < 0.75 * c["hesse"][1]
    # new made-up noise each time: the spread HESSE describes
    np.testing.assert_allclose(c["toys_spread"], c["hesse"], rtol=0.06)


def test_hesses_ellipse_is_near_chi_square_plus_one():
    c = E.compare()
    lo, hi = c["profile_on_ellipse"]
    assert 0.85 < lo < 1.0 < hi < 1.15
    ell = c["hesse_ellipse"]
    # its shadow on each axis is HESSE's error
    np.testing.assert_allclose(ell.max(0) - c["values"][1:], c["hesse"][1:], rtol=2e-3)
    np.testing.assert_allclose(c["values"][1:] - ell.min(0), c["hesse"][1:], rtol=2e-3)


def test_the_resamples_are_picks_with_repeats():
    c = E.compare()
    counts = c["counts"]
    assert (counts.sum(1) == 40).all()
    assert c["distinct"].min() >= 20 and c["distinct"].max() <= 33
    assert abs(c["distinct"].mean() - 40 * (1 - (39 / 40) ** 40)) < 1.0    # about 63%

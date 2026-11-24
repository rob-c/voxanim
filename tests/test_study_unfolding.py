"""voxanim.study_unfolding: the numpy twin of examples/unfolding.py. The script is run (it needs
numpy only) and every number the study keeps is checked against it; then the arithmetic of
unfolding and the measured claims. No dataset is needed: the events are made up."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from voxanim import study_unfolding as S

SCRIPT = Path(__file__).resolve().parent.parent / "examples" / "unfolding.py"


@pytest.fixture(scope="module")
def script():
    """The script's globals, run without its main block (the pseudo-experiments not drawn)."""
    return runpy.run_path(str(SCRIPT), run_name="unfolding_under_test")


# ---------------------------------------------------------------------- the script is the study
def test_the_script_prints_what_the_study_works_out():
    out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True,
                         check=True, timeout=120).stdout.splitlines()
    assert out == S.printed()


def test_every_array_is_the_scripts(script):
    w = S.world()
    for key in ("edges", "truth", "t", "found", "r", "H", "n", "R", "eff", "expected", "data",
                "back", "naive", "s", "noise", "blown", "prior"):
        np.testing.assert_array_equal(script[key], w[key], err_msg=key)
    np.testing.assert_array_equal(script["U"], w["U"])
    np.testing.assert_array_equal(script["Vt"], w["Vt"])
    assert script["N"] == S.N and script["SIMULATED"] == S.SIMULATED and script["TOYS"] == S.TOYS
    for k in (0, 1, 4, 10):
        np.testing.assert_array_equal(script["bayes"](w["data"], w["prior"], k),
                                      S.bayes(w["data"], w["prior"], w["R"], w["eff"], k))


def test_the_pseudo_experiments_are_the_scripts(script):
    toy = S.toys()
    got = script["pseudo_experiments"](S.SCRIPT_ITERATIONS)   # the next draws of its rng
    for k in S.SCRIPT_ITERATIONS:
        bias, spread = got[k]
        np.testing.assert_allclose(bias, toy["bias"][k], rtol=0, atol=1e-9)
        np.testing.assert_allclose(spread, toy["spread"][k], rtol=0, atol=1e-9)
    bias, spread = got["inverse"]
    np.testing.assert_allclose(bias, toy["inverse_bias"], atol=1e-6)
    np.testing.assert_allclose(spread, toy["inverse_spread"], atol=1e-6)


# ---------------------------------------------------------------------- the arithmetic
def test_the_truth_and_the_response():
    w = S.world()
    assert np.isclose(w["truth"].sum(), S.N)
    assert np.isclose(S.falling(w["edges"]).sum(), 1) and np.isclose(S.bump(w["edges"]).sum(), 1)
    assert int(np.argmax(w["truth"][3:])) + 3 == 5               # the bump: 50-60 GeV
    R = w["R"]
    assert R.shape == (10, 10) and (R >= 0).all()
    assert (R.sum(0) < 1).all() and np.allclose(R.sum(0), w["eff"])
    # column j: of the events simulated in true bin j, the share found and measured in bin i
    t, r, found = w["t"], w["r"], w["found"]
    j = 4
    inside = (t >= 40) & (t < 50)
    hit = found & inside & (r >= 30) & (r < 40)
    assert R[3, j] == hit.sum() / inside.sum()
    assert np.allclose(w["n"], np.histogram(t, w["edges"])[0])


def test_inverting_and_why():
    w = S.world()
    assert np.abs(w["back"] - w["truth"]).max() < 1e-9
    R, U, s, Vt = w["R"], w["U"], w["s"], w["Vt"]
    np.testing.assert_allclose(U @ np.diag(s) @ Vt, R, atol=1e-14)
    # the inversion's error is the noise along U's directions, divided by s, along Vt's
    np.testing.assert_allclose(w["naive"] - w["truth"], Vt.T @ (w["noise"] / s), atol=1e-8)
    np.testing.assert_allclose(np.linalg.inv(R), Vt.T @ np.diag(1 / s) @ U.T, atol=1e-8)


def test_bayes_keeps_the_data_total_and_its_fixed_point():
    w = S.world()
    R, eff = w["R"], w["eff"]
    for k in (1, 3, 10):
        t = S.bayes(w["data"], w["prior"], R, eff, k)
        assert np.isclose((R @ t).sum(), w["data"].sum())      # it folds back to the data's total
    # the truth folded is the expected counts: on them the iterations stay at the truth
    t = S.bayes(w["expected"], w["truth"], R, eff, 5)
    np.testing.assert_allclose(t, w["truth"], rtol=1e-12)
    # the prior's scale does not matter after one iteration
    np.testing.assert_allclose(S.bayes(w["data"], 3 * w["prior"], R, eff, 2),
                               S.bayes(w["data"], w["prior"], R, eff, 2))


# ---------------------------------------------------------------------- what was measured
def test_the_measured_claims():
    sm = S.summary()
    w = S.world()
    assert sm["simulated_in_range"] == 964_628 and sm["counted"] == 582_195
    assert round(sm["expected_total"]) == 6345 and sm["data_total"] == 6362
    assert np.allclose(w["s"][[0, -1]], [0.7458, 0.0078], atol=5e-5)
    assert round(sm["condition"]) == 96
    naive = np.round(w["naive"]).astype(int)
    assert naive.min() == -900 and naive.max() == 3475 and (naive < 0).sum() == 3
    # the data's noise is about 25 a bin, and about 25 along each of U's directions too
    assert 24 < sm["data_noise"] < 26
    assert np.all((S.noise_size() > 22) & (S.noise_size() < 27))
    # R inverted spreads 1,166 events a bin (rms), as worked out (1,180), 46 times the noise
    assert round(sm["inverse_spread"]) == 1166 and round(sm["inverse_spread_exact"]) == 1180
    assert abs(sm["inverse_spread"] / sm["inverse_spread_exact"] - 1) < 0.03
    assert 45 < sm["inverse_spread"] / sm["data_noise"] < 47


def test_bias_falls_and_spread_grows_with_the_iterations():
    toy = S.toys()
    b, s, e = toy["bias_rms"], toy["spread_rms"], toy["error_rms"]
    assert (np.diff(b) < 0).all() and (np.diff(s) > 0).all()
    assert s[0] < 1e-9                                      # the prior does not see the data
    assert [round(b[k]) for k in (0, 4, 10, 20)] == [423, 82, 41, 24]
    assert [round(s[k]) for k in (1, 4, 10, 20)] == [24, 38, 53, 69]
    assert int(np.argmin(e)) == 10 and round(e[10], 1) == 66.8
    np.testing.assert_allclose(toy["mean_squared"], e ** 2)
    assert s[S.UPTO] < 0.1 * toy["inverse_spread_rms"]     # even 50 iterations spread far less
    assert toy["negative"] > 0.85                            # most inversions have a negative bin

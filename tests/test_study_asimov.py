"""voxanim.study_asimov is examples/asimov.py in numpy (its fit, its toys, what it prints), and
the measured claims of "The Asimov dataset" hold: the Asimov dataset's significance is the
toys' median, and s / sqrt(b) is too high, far too high when the background is small."""

from __future__ import annotations

import contextlib
import io
import math
import runpy

import numpy as np
import pytest

from voxanim import blocks
from voxanim import study_asimov as A

SCRIPT = str(blocks.EXAMPLES / "asimov.py")


@pytest.fixture(scope="module")
def ns():
    return runpy.run_path(SCRIPT, run_name="test")


# ---------------------------------------------------------------------- the twin is the script
def test_the_counts_are_the_scripts(ns):
    np.testing.assert_array_equal(ns["s"], A.S)
    np.testing.assert_array_equal(ns["b"], A.B)


def test_the_fit_and_z_are_the_scripts(ns):
    for name in A.CASES:
        c = A.case(name)
        for i in (0, 1, 2, 17, 4321, 9999):
            n = c["toys"][i]
            assert math.isclose(ns["fit_mu"](n, A.S, c["b"]), c["mu"][i], rel_tol=0,
                                abs_tol=1e-12)
            assert math.isclose(ns["z0"](n, A.S, c["b"]), c["z"][i], rel_tol=0, abs_tol=1e-12)


def test_the_script_prints_what_the_study_shows(ns):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        toys, z, _za, _simple = ns["expected"](A.S, A.B, 7)
    np.testing.assert_array_equal(toys, A.case("b")["toys"])
    np.testing.assert_allclose(z, A.case("b")["z"], rtol=0, atol=1e-12)
    p = A.printed("b")
    assert out.getvalue().split() == [p["median"], p["asimov"], p["simple"]] == [
        "2.89", "2.93", "3.16"]
    with contextlib.redirect_stdout(io.StringIO()) as out:
        ns["expected"](A.S, A.B / 10, 8)
    p = A.printed("b/10")
    assert out.getvalue().split() == [p["median"], p["asimov"], p["simple"]] == [
        "6.74", "6.77", "9.98"]


# ---------------------------------------------------------------------- what was measured
def test_every_fit_converged():
    for name in A.CASES:
        c = A.case(name)
        assert c["converged"] == A.TOYS
        assert np.isfinite(c["z"]).all() and (c["z"] >= 0).all()
        assert (c["toys"].sum(1) > 0).all()


def test_the_asimov_dataset_fits_mu_one_and_is_the_formula():
    for name in A.CASES:
        c = A.case(name)
        assert abs(c["asimov_mu"] - 1.0) < 1e-12
        assert math.isclose(c["asimov_z"], c["asimov_formula"], rel_tol=1e-12)
        assert math.isclose(c["asimov_z"], math.sqrt((c["asimov_bins"] ** 2).sum()),
                            rel_tol=1e-12)
    # one bin by hand: s = 3, b = 30
    assert math.isclose(A.formula(np.array([3.0]), np.array([30.0]))[0],
                        math.sqrt(2 * (33 * math.log(1.1) - 3)))


def test_the_asimov_z_is_the_toys_median():
    for name in A.CASES:
        c = A.case(name)
        assert abs(c["asimov_z"] - c["median"]) < 0.05
        assert 0.47 < c["above"] < 0.51                  # about half the toys above it


def test_s_over_root_b_is_high_and_far_too_high_with_little_background():
    near, far = A.case("b"), A.case("b/10")
    assert 1.0 < near["simple"] / near["asimov_z"] < 1.1
    assert far["simple"] / far["asimov_z"] > 1.4
    assert (far["simple_bins"] > far["asimov_bins"]).all()
    assert far["simple"] - far["median"] > 3.0


def test_the_first_toys():
    c = A.case("b")
    assert c["first"].tolist() == [35, 33, 17, 14]
    assert round(c["first_z"], 2) == 3.38 and round(c["first_mu"], 2) == 1.19
    assert A.case("b/10")["asimov"].tolist() == [6.0, 10.0, 9.2, 3.8]

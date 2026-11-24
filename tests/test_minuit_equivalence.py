"""voxanim/minuit_calls.json is what iminuit really asks: examples/minuit_vs_sgd.py run again
with the same iminuit version makes every call again, in order; and the twin's chi-square, data
and SGD are the script's."""

from __future__ import annotations

import runpy

import numpy as np
import pytest

from voxanim import blocks
from voxanim import study_minuit as M

iminuit = pytest.importorskip("iminuit", reason="iminuit is not installed: the recorded calls "
                                                "are checked with it")


@pytest.fixture(scope="module")
def ns():
    return runpy.run_path(str(blocks.MINUIT), run_name="test")


def test_the_recorded_calls_are_iminuits(ns):
    if iminuit.__version__ != M.record()["iminuit"]:
        pytest.skip("iminuit %s installed; the record is of %s"
                    % (iminuit.__version__, M.record()["iminuit"]))
    m, n = ns["run_minuit"]()
    assert n == M.record()["migrad_calls"]
    np.testing.assert_array_equal(np.array(ns["asked"]), M.calls())
    np.testing.assert_allclose(list(m.values), M.record()["values"], rtol=0, atol=0)


def test_the_twin_is_the_script(ns):
    x, y = M.data()
    np.testing.assert_array_equal(ns["x"], x)
    np.testing.assert_array_equal(ns["y"], y)
    for a, tau, b, v in M.calls()[::7]:
        assert abs(M.chi2((a, tau, b)) - v) < 1e-9 * max(1, v)
    path = ns["run_sgd"]()
    np.testing.assert_allclose(path, M.sgd()["path"], rtol=0, atol=1e-12)

"""voxanim.study_minuit: the labels on Minuit's calls, and the measured comparison with SGD."""

from __future__ import annotations

import numpy as np

from voxanim import study_minuit as M


def test_every_call_is_labelled():
    s = M.steps()
    assert s[0]["kind"] == "start" and s[0]["point"] == list(M.START)
    kinds = [e["kind"] for e in s]
    assert set(kinds) == {"start", "probe", "try", "hesse"}
    assert kinds.count("hesse") == len(s) - M.record()["migrad_calls"]
    for e in s:
        if e["kind"] == "probe":                       # one axis only, from the current point
            d = np.array(e["point"]) - np.array(e["base"])
            assert np.count_nonzero(np.abs(d) > 1e-12) == 1 and e["axis"] in (0, 1, 2)
    first = s[1:13]
    assert [e["kind"] for e in first] == ["probe"] * 12      # it feels all 3 axes first
    assert [e["axis"] for e in first] == [0] * 4 + [1] * 4 + [2] * 4
    tries = [e for e in s if e["kind"] == "try"]
    assert any(not e["kept"] for e in tries) and any(e["kept"] for e in tries)
    assert tries[0]["chi2"] > s[0]["chi2"] and not tries[0]["kept"]   # the first step overshoots


def test_minuit_and_sgd():
    m = M.summary_minuit()
    assert m["migrad_calls"] < 100 and m["fval"] < 40 < m["start_chi2"]
    assert all(abs(v - t) < 3 * e for v, t, e in zip(m["values"], M.TRUE, m["errors"],
                                                       strict=True))
    c = M.compare()
    assert c["sgd_evaluations"] > 5 * c["minuit_evaluations"]
    assert all(abs(a - b) < 0.05 for a, b in zip(c["sgd_average"], c["minuit"], strict=True))
    assert c["sgd_average_chi2"] < c["sgd_end_chi2"]       # averaging the walk lands closer
    assert max(c["sgd_spread"]) < 0.2 * min(c["minuit_errors"])

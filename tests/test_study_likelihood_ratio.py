"""voxanim.study_likelihood_ratio: the exact arithmetic (the likelihood ratio, the best test, a
single cut, the loss floor, the backward pass), and the measured claims the two animations make.
No dataset is needed: the events are made up with the script's seeds."""

from __future__ import annotations

import math

import numpy as np

from voxanim import study_likelihood_ratio as L


# ---------------------------------------------------------------------- exact
def test_the_ratio_is_a_parabola_in_its_log():
    x = np.linspace(-3, 4, 15)
    np.testing.assert_allclose(L.log_ratio(x), math.log(2) - 2 * (x - 1) ** 2 + x ** 2 / 2,
                               atol=1e-12)
    px, pv = L.peak()
    assert math.isclose(px, 4 / 3) and math.isclose(pv, math.exp(L.log_ratio(4 / 3)))
    assert 3.8 < pv < 4.0
    # not a straight line: it falls on both sides of its peak
    assert L.log_ratio(0.0) < L.log_ratio(px) > L.log_ratio(3.0)


def test_the_best_test_is_a_window_and_beats_a_cut():
    lo, hi, power = L.np_at()
    assert math.isclose(lo + hi, 2 * L.peak()[0])
    a, p = L.np_window((hi - lo) / 2)
    assert math.isclose(a, L.ALPHA, abs_tol=1e-9) and math.isclose(p, power)
    c, cut_power = L.cut_at()
    assert math.isclose(c, 1.6448536, abs_tol=1e-6)               # 5% of a standard normal
    assert math.isclose(cut_power, 1 - float(L.norm_cdf((c - 1.0) / 0.5)))
    assert 0.19 < power < 0.195 and 0.098 < cut_power < 0.099
    _alphas, p_np, p_cut = L.exact_rocs(101)
    assert (p_np >= p_cut - 1e-6).all()                           # Neyman-Pearson: never worse
    assert np.all(np.diff(p_np) >= -1e-9)


def test_the_floor_is_the_exact_ratios_loss():
    f = L.floor()
    assert 0.468 < f < 0.470
    x0, x1, _f0, _f1 = L.hypotheses()
    assert abs(L.floor_on(x0, x1) - f) < 0.002                    # on 40,000 events: close


def test_the_backward_pass_by_finite_differences():
    P = {k: v.astype(np.float64) for k, v in L.init(2, 5).items()}
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (16, 2))
    y = (rng.random((16, 1)) < 0.5).astype(np.float64)
    _loss, G = L.loss_and_grads(P, x, y)
    for k in ("0.weight", "2.bias", "4.weight"):
        flat = P[k].ravel()
        for j in (0, 3):
            e = 1e-6
            flat[j] += e
            up = L.bce(L.forward(P, x)[0], y)
            flat[j] -= 2 * e
            down = L.bce(L.forward(P, x)[0], y)
            flat[j] += e
            assert abs((up - down) / (2 * e) - G[k].ravel()[j]) < 1e-5


def test_the_exact_weight():
    p = np.array([[0.4, 0.0], [1.0, -2.0]])
    C = np.asarray(L.DATA_COV)
    d = p - np.asarray(L.DATA_MEAN)
    data = np.exp(-0.5 * np.einsum("ni,ij,nj->n", d, np.linalg.inv(C), d)) / (
        2 * np.pi * math.sqrt(np.linalg.det(C)))
    sim = np.exp(-0.5 * (p ** 2).sum(1)) / (2 * np.pi)
    np.testing.assert_allclose(np.exp(L.log_weight(p)), data / sim, rtol=1e-12)
    assert math.isclose(C[0][1] / math.sqrt(C[0][0] * C[1][1]), 0.4)


# ---------------------------------------------------------------------- measured: part one
def test_the_network_learns_the_ratio_where_there_are_events():
    o = L.one()
    assert o["worst_inside"] < 0.1                     # within 10% from x = -0.75 to 2.25
    assert o["last_loss"] < 0.47 < 0.6 < o["first_loss"]
    assert abs(o["last_loss"] - o["floor_on"]) < 5e-4
    assert o["far_x"] == 3.0 and o["far_learned"] > 3 * o["far_exact"]   # drifts past the events
    assert o["beyond_far0"] + o["beyond_far1"] < 30 and o["below1"] == 0
    off = np.asarray(o["off"])
    grid = np.asarray(o["grid"])
    assert np.abs(off[grid >= 3.5]).min() > 2.0        # far out: wrong by a factor over 7


def test_its_roc_is_the_best_tests():
    o = L.one()
    assert abs(o["net_power"] - o["np_power"]) < 3 * o["net_power_err"]
    assert o["own_power"] <= o["np_power"] + 1e-9      # its own window, exactly: no better
    assert o["own_power"] > 0.97 * o["np_power"]
    assert o["cut_power"] < 0.55 * o["np_power"]
    al = np.asarray(o["alphas"])
    sel = (al > 0.02) & (al < 0.3)
    assert np.abs(np.asarray(o["roc_net"])[sel] - np.asarray(o["roc_np"])[sel]).max() < 0.02


# ---------------------------------------------------------------------- measured: part two
def test_the_weighted_simulation_matches_the_data():
    t = L.two()
    for name, h in t["hists"].items():
        before, after = h["chi2_before"][0], h["chi2_after"][0]
        assert after < before / 20, name
        assert h["chi2_exact"][0] <= after + 1e-9, name   # the exact weights do better still
    assert abs(t["w_mean"] - 1) < 0.02                  # the weights' mean: 1, as it should be
    assert t["rms_bulk"] < 0.05


def test_where_it_fails():
    t = L.two()
    assert t["hists"]["y"]["empty"] == 38               # data where the simulation has none
    assert t["w_max"] < t["w_max_exact"] / 3            # the largest weights, too small
    assert 25000 < t["n_eff"] < 35000 and t["n_eff_exact"] < t["n_eff"]
    assert t["rms_bulk_few"] > 3 * t["rms_bulk"]        # 1,000 events of each: much worse
    for h in t["hists"].values():
        assert h["chi2_few"][0] > h["chi2_after"][0]

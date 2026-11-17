"""Animations 141-144 (voxanim.mech_stats_tests) and their experiments
(voxanim.study_stats_tests): the layout is clean, the loop closes, every phase says what
kind of evidence it shows, the numbers on screen are the study's, the special functions
written for them give known values, each study's claim holds with a margin, and the
pseudo-experiments are the ones the recorded seed gives.

None needs a dataset: everything here runs from an empty cache."""

from __future__ import annotations

import math

import numpy as np
import pytest

from voxanim import cache, layout, presets, qa
from voxanim import mech_stats_tests as M
from voxanim import study_stats_tests as S
from voxanim.next_catalog import NO_DATA, build_study

ERRORS, PVALUES, NP, LEE = ("type-i-and-type-ii-errors", "p-values", "neyman-pearson",
                            "look-elsewhere-effect")
NAMES = (ERRORS, PVALUES, NP, LEE)
EACH = [pytest.param(name, id=name) for name in NAMES]


def study(name):
    return build_study("study_stats_tests", name)


def anim(name):
    return presets.get(name).build()


def subtitles(a):
    return [a.subtitle_for(k) for k in range(len(a.PHASES))]


def texts(a, t):
    return [x.s for x in a.scene(t).texts]


def at(a, phase, u):
    lo, hi = a.spans[phase]
    return min(lo + (hi - lo) * u, hi - a.FADE - 0.05)


def late(a, phase):
    """Everything of a phase is on screen."""
    return texts(a, at(a, phase, 0.97))


def within(measured, exact, n, sigmas=4.0):
    """A share counted in ``n`` draws agrees with its exact chance."""
    return abs(measured - exact) < sigmas * math.sqrt(exact * (1 - exact) / n)


# ---------------------------------------------------------------------- all four
def test_registered_and_need_no_data():
    assert set(M.ANIMATIONS) == set(S.STUDIES) == set(NAMES) <= NO_DATA


def test_everything_builds_from_an_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_CACHE", str(tmp_path))
    for name in NAMES:                                # offline (conftest): a download would raise
        M.ANIMATIONS[name](S.STUDIES[name]().validate()).scene(3.0)
    assert list(tmp_path.iterdir()) == []             # and nothing is slow enough to cache


@pytest.mark.parametrize("name", EACH)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", EACH)
def test_drawn_by_this_module_and_loop_closes(name):
    a = anim(name)
    assert type(a) is M.ANIMATIONS[name]
    assert 45.0 <= a.t_done <= 60.0 and len(a.PHASES) == 5

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:                            # every phase draws things, not only text
        assert len(a.scene((lo + hi) / 2).quads) >= 3


@pytest.mark.parametrize("name", EACH)
def test_subtitles_fit_two_lines(name):
    for sub in subtitles(anim(name)):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE


@pytest.mark.parametrize("name", EACH)
def test_every_phase_is_tagged_with_its_kind_of_evidence(name):
    a = anim(name)
    tags = []
    draw = a.tag
    a.tag = lambda sc, kind, op: (tags.append(kind), draw(sc, kind, op))
    for k in range(len(a.PHASES)):
        for u in (0.3, 0.6, 0.9):
            tags.clear()
            shown = texts(a, at(a, k, u))
            assert len(tags) == 1 and tags[0] in shown
            assert S.EXACT in tags[0] or S.PSEUDO in tags[0]


@pytest.mark.parametrize("name", EACH)
def test_study_says_what_was_run(name):
    s = study(name)
    m = s.metadata
    assert len(s.panels) == 4 and m["seeds"] and m["model"]
    assert m["designs_tried"] and m["limitations"] and "default_rng" in m["where_the_seed_enters"]
    assert set(m["evidence"]) == {S.EXACT, S.PSEUDO}


# ---------------------------------------------------------------------- special functions
def test_normal_tail_and_its_inverse():
    assert S.norm_sf(0.0) == 0.5
    assert np.isclose(S.norm_sf(1.6448536269514722), 0.05, rtol=1e-12)
    assert np.isclose(S.norm_sf(3.0), 1.3498980316301e-3, rtol=1e-12)
    assert np.isclose(S.norm_sf(5.0), 2.866515718791933e-7, rtol=1e-12)
    assert np.isclose(S.norm_sf(-1.0) + S.norm_sf(1.0), 1.0)
    p = np.array([1e-12, 2.866515718791933e-7, 1e-4, 0.02, 0.025, 0.3, 0.5, 0.7, 0.999])
    z = S.norm_isf(p)
    assert np.allclose(S.norm_sf(z), p, rtol=1e-10, atol=0)
    assert np.isclose(z[1], 5.0) and np.isclose(z[4], 1.959963984540054) and z[6] == 0.0
    assert np.all(np.diff(z) < 0)


def test_poisson_sums():
    pmf = S.poisson_pmf(10.0, 80)
    assert np.isclose(pmf.sum(), 1.0) and np.isclose(pmf[0], math.exp(-10.0))
    assert np.isclose(pmf[10], 10.0 ** 10 * math.exp(-10.0) / math.factorial(10))
    sf = S.poisson_sf(3.0, 12)
    assert sf[0] == 1.0 and np.isclose(sf[2], 1 - 4 * math.exp(-3.0))
    assert np.allclose(sf[:-1] - sf[1:], S.poisson_pmf(3.0, 11))
    # a far tail, by the identity P(n or more | mu) = P(chi-square with 2n degrees <= 2 mu)
    assert np.isclose(S.poisson_sf(100.0, 200)[150], 1 - S.chi2_sf(200.0, 300), rtol=1e-7)
    assert np.isclose(S.poisson_sf(10.0, 30)[16], 1 - S.chi2_sf(20.0, 32), rtol=1e-12)


def test_chi_square_tails():
    assert np.isclose(S.chi2_sf(5.991464547107979, 2), 0.05)
    assert np.isclose(S.chi2_sf(18.307038053275146, 10), 0.05, rtol=1e-9)
    assert np.isclose(S.chi2_sf(9.487729036781154, 4), 0.05, rtol=1e-9)
    assert S.chi2_sf(0.0, 10) == 1.0
    with pytest.raises(ValueError):
        S.chi2_sf(1.0, 3)
    x = np.linspace(0.0, 80.0, 16001)
    pdf = S.chi2_pdf(x, 10)
    area = float(((pdf[1:] + pdf[:-1]) / 2 * np.diff(x)).sum())
    mean = float(((pdf[1:] * x[1:] + pdf[:-1] * x[:-1]) / 2 * np.diff(x)).sum())
    assert np.isclose(area, 1.0, atol=1e-6) and np.isclose(mean, 10.0, atol=1e-4)
    step = 1e-4                                       # the density is minus the tail's slope
    assert np.isclose((S.chi2_sf(12.0 - step, 10) - S.chi2_sf(12.0 + step, 10)) / (2 * step),
                      S.chi2_pdf(12.0, 10), rtol=1e-6)


def test_non_central_chi_square_against_draws():
    assert np.isclose(S.ncx2_sf(18.3, 10, 1e-12), S.chi2_sf(18.3, 10))
    rng = np.random.default_rng(0)
    shift = np.zeros(10)
    shift[:3] = 2.0                                   # non-centrality 3 x 2^2 = 12
    draws = ((rng.normal(0, 1, (200000, 10)) + shift) ** 2).sum(1)
    for x in (10.0, 18.307, 30.0):
        assert within((draws >= x).mean(), S.ncx2_sf(x, 10, 12.0), len(draws))


def test_small_helpers():
    assert np.isclose(S.bisect(lambda x: x * x - 2.0, 0.0, 2.0), math.sqrt(2.0))
    assert S.window_sums(np.arange(6.0), 4).tolist() == [6.0, 10.0, 14.0]
    assert S.window_sums(np.ones((3, 60))).shape == (3, 57)
    a, b = M.clip_line((1.0, 0.0), 2.0, (-4, 6), (-5, 5))            # the line x = 2
    assert np.allclose(sorted([a[1], b[1]]), [-5, 5]) and a[0] == b[0] == 2.0
    assert M.clip_line((1.0, 0.0), 9.0, (-4, 6), (-5, 5)) is None
    a, b = M.clip_line((1.0, 1.0), 0.0, (-1, 1), (-2, 2))            # x + y = 0
    assert np.allclose(np.abs([a, b]), 1.0) and np.allclose(a + b, 0.0)
    tex = M.scatter([0.0], [0.0], (-1, 1), (-1, 1), (10, 10), [M.RED], fat=2)
    assert (tex != np.asarray(M.PAPER, np.float32)).any(-1).sum() == 4


# ---------------------------------------------------------------------- 141
def test_errors_are_exact_poisson_sums():
    d = S.errors_data()
    m = study(ERRORS).metadata
    cut = m["threshold"]
    pmf0, pmf1 = S.poisson_pmf(10.0, 200), S.poisson_pmf(18.0, 200)
    assert np.isclose(m["alpha"], pmf0[cut:].sum()) and np.isclose(m["beta"], pmf1[:cut].sum())
    assert np.isclose(m["power"], 1 - m["beta"])
    assert d["alpha"][cut] <= S.ERR_ALPHA < d["alpha"][cut - 1]      # the smallest such threshold
    assert cut == 16 and round(m["alpha"], 4) == 0.0487 and round(m["beta"], 4) == 0.2867


def test_errors_trade_against_each_other():
    d = S.errors_data()
    assert np.all(np.diff(d["alpha"]) <= 0) and np.all(np.diff(d["beta"]) >= 0)
    lo, hi = 12, 20                                   # a stricter threshold: fewer false alarms,
    assert d["alpha"][hi] < d["alpha"][lo] - 0.2      # more misses
    assert d["beta"][hi] > d["beta"][lo] + 0.5
    assert np.all(d["power"] >= d["alpha"] - 1e-12)   # the ROC curve is above the diagonal
    assert d["power"][16] - d["alpha"][16] > 0.6


def test_errors_table_is_what_the_seed_gives():
    d = S.errors_data()
    rng = np.random.default_rng(S.ERR_SEED)
    h0 = rng.poisson(S.ERR_B, S.ERR_N)
    h1 = rng.poisson(S.ERR_B, S.ERR_N) + rng.poisson(S.ERR_S, S.ERR_N)
    assert np.array_equal(h0, d["h0"]) and np.array_equal(h1, d["h1"])
    table = np.asarray(study(ERRORS).panels[2].data["values"], int)
    cut = d["n_c"]
    assert table.tolist() == [[(h0 < cut).sum(), (h0 >= cut).sum()],
                              [(h1 < cut).sum(), (h1 >= cut).sum()]]
    assert table.sum(1).tolist() == [S.ERR_N, S.ERR_N]
    assert within(table[0, 1] / S.ERR_N, d["alpha_c"], S.ERR_N)
    assert within(table[1, 0] / S.ERR_N, d["beta_c"], S.ERR_N)
    assert np.abs(d["roc_mc"] - np.stack([d["alpha"], d["power"]], 1)).max() < 0.03


def test_power_grows_with_the_data():
    d = S.errors_data()
    runs = [float(v) for v in d["runs"]]
    one, two, four = (d["power_run"][runs.index(t)] for t in S.ERR_SHOWN)
    assert two > one + 0.15 and four > two + 0.08 and four > 0.99
    assert np.all(d["alpha_run"] <= S.ERR_ALPHA) and np.all(d["alpha_run"] > 0.03)
    for i, t in enumerate(runs):                      # each threshold is the smallest allowed
        assert S.poisson_sf(10.0 * t, 200)[d["cuts"][i] - 1] > S.ERR_ALPHA
    assert np.abs(d["power_mc"] - d["power_run"]).max() < 0.02
    assert np.abs(d["alpha_mc"] - d["alpha_run"]).max() < 0.01
    assert d["power_run"][-1] - d["power_run"][0] > 0.6


def test_errors_numbers_on_screen():
    a = anim(ERRORS)
    d, m = S.errors_data(), a.study.metadata
    two = late(a, 1)
    assert "= %.4f" % m["alpha"] in two and "= %.4f" % m["beta"] in two
    assert "power = 1 − β = %.4f" % m["power"] in two
    assert "= %.4f + %.4f + %.4f + …" % tuple(d["pmf0"][16:19]) in two
    four = late(a, 3)
    for v in np.asarray(a.study.panels[2].data["values"], int).ravel():
        assert format(int(v), ",") in four
    assert a.cum[-1].tolist() == a.table.ravel().tolist() and a.arrived(1.0) == 2 * S.ERR_N
    first = texts(a, at(a, 0, 0.9))
    assert ["n = %d" % v for v in d["h0"][:6]] == [s for s in first if s.startswith("n = ")][::2]
    assert "%.3f" % m["alpha"] in subtitles(a)[1] and "240 false alarms" in subtitles(a)[3]
    cut, low = a.cut_at(1.0)
    assert cut == m["threshold"] and low == 4         # the slide ends on the chosen threshold


# ---------------------------------------------------------------------- 142
def test_p_value_of_the_observed_data():
    d = S.pvalue_data()
    m = study(PVALUES).metadata
    rng = np.random.default_rng(S.PV_SEED)
    y = np.round(rng.normal(S.pv_truth(), S.PV_SIGMA), 1)
    assert np.array_equal(y, d["y"])                  # the first draw of the recorded seed
    pulls = (y - S.PV_MU) / S.PV_SIGMA
    assert np.allclose(pulls, study(PVALUES).panels[0].data["values"])
    assert np.isclose(m["chi2_observed"], (pulls ** 2).sum())
    assert np.isclose(m["p_exact"], S.chi2_sf(m["chi2_observed"], S.PV_N))
    assert np.isclose(S.norm_sf(m["z"]), m["p_exact"], rtol=1e-9)
    assert m["tail_count"] == int((d["null"] >= d["obs"]).sum())
    assert within(m["p_pseudo"], m["p_exact"], S.PV_TOYS)
    assert S.pv_truth().tolist() == [20, 20, 20, 24, 24, 24, 20, 20, 20, 20]


def test_null_chi_square_follows_the_exact_curve():
    d = S.pvalue_data()
    assert np.array_equal(d["null"][:40], S.chi2_of(d["null_y"]))
    assert abs(d["null"].mean() - S.PV_N) < 0.15 and abs(d["null"].var() - 2 * S.PV_N) < 1.5
    centres = np.asarray(study(PVALUES).panels[1].data["x"])
    measured, exact = study(PVALUES).panels[1].data["ys"]
    assert np.abs(measured - exact).max() < 0.008     # a 20,000-draw histogram on the curve
    assert centres[np.argmax(exact)] == 8.5           # the mode of chi-square(10) is 8


def test_null_p_values_are_uniform():
    d = S.pvalue_data()
    m = study(PVALUES).metadata
    hist = np.asarray(study(PVALUES).panels[2].data["values"])
    assert hist.sum() == S.PV_TOYS and np.abs(hist - 1000).max() < 5 * math.sqrt(950)
    p = np.sort(d["p_null"])
    assert np.abs(p - (np.arange(len(p)) + 0.5) / len(p)).max() < 0.015
    assert within(m["share_below_005"], 0.05, S.PV_TOYS)
    assert np.allclose(S.chi2_sf(d["quantiles"], S.PV_N), np.arange(0.95, 0.0, -0.05))
    assert np.isclose(S.chi2_sf(d["crit"], S.PV_N), S.PV_CUT)


def test_p_below_005_is_not_a_5_percent_chance_of_h0():
    d = S.pvalue_data()
    m = study(PVALUES).metadata
    pop, grid = d["pop"], d["grid"]
    assert pop.sum() == S.PV_POP and pop[1].sum() == S.PV_PRIOR * S.PV_POP
    assert grid.sum() == S.PV_GRID and grid[1].sum() == S.PV_PRIOR * S.PV_GRID
    assert int((d["grid_p"] < S.PV_CUT).sum()) == grid[:, 1].sum()
    assert int(d["grid_real"].sum()) == grid[1].sum()
    assert within(m["power_pseudo"], m["power_exact"], pop[1].sum())
    assert np.isclose(m["power_exact"], S.ncx2_sf(d["crit"], S.PV_N, 12.0))
    exact = 0.9 * 0.05 / (0.9 * 0.05 + 0.1 * m["power_exact"])
    assert np.isclose(m["false_alarm_share_exact"], exact)
    assert abs(m["false_alarm_share_pseudo"] - exact) < 0.04
    for share in (exact, m["false_alarm_share_pseudo"], m["false_alarm_share_on_screen"]):
        assert share > S.PV_CUT + 0.25                # many times the 5% a newcomer expects


def test_p_value_numbers_on_screen():
    a = anim(PVALUES)
    d, m = S.pvalue_data(), a.study.metadata
    one = late(a, 0)
    assert "%.2f" % m["chi2_observed"] in one
    for i in range(S.PV_N):
        assert "%.1f" % d["y"][i] in one and "%.2f" % d["pulls"][i] ** 2 in one
    three = late(a, 2)
    assert "pseudo-experiments: %d / 20,000 = %.4f" % (m["tail_count"], m["p_pseudo"]) in three
    assert "exact area of the tail:  p = %.4f" % m["p_exact"] in three
    assert "Z = %.2fσ" % m["z"] in three
    assert "%d of 20,000 have p < 0.05: %.2f%%" % (
        a.study.panels[2].data["values"][0], 100 * m["share_below_005"]) in late(a, 3)
    g = np.asarray(a.study.panels[3].data["on_screen"], int)
    five = late(a, 4)
    assert "%d real" % g[1, 1] in five and "%d false alarms" % g[0, 1] in five
    assert "%d / %d = %.0f%% false" % (g[0, 1], g[:, 1].sum(),
                                       100 * g[0, 1] / g[:, 1].sum()) in five
    assert a.toys_at(1.0) == S.PV_TOYS
    dots = (a.textures[1] != np.asarray(M.PAPER, np.float32)).any(-1)
    assert dots[::a.CELL, ::a.CELL].shape == (a.ROWS, a.COLS) and a.ROWS * a.COLS == S.PV_GRID


# ---------------------------------------------------------------------- 143
def test_the_statistic_is_the_log_likelihood_ratio():
    d = S.np_data()
    rng = np.random.default_rng(1)
    xy = rng.normal(0, 3, (50, 2))
    l0, l1 = S.log_likelihoods(xy)
    assert np.allclose(l1 - l0, xy @ S.np_weights()[2] + S.np_constant())
    assert np.allclose(S.np_weights()[2], [1.5, 0.6]) and np.isclose(S.np_constant(), -1.845)
    assert np.isclose(math.exp(S.log_likelihoods([[0.0, 0.0]])[0][0]),
                      1 / (2 * math.pi * 1.0 * 2.0))  # the density at H0's centre
    assert np.isclose(d["event_l1"] / d["event_l0"],
                      math.exp(d["event"] @ d["w"][2] + d["constant"]))
    rng = np.random.default_rng(S.NP_SEED)
    e0 = rng.normal(0.0, 1.0, (S.NP_N, 2)) * np.array(S.NP_SIGMA)
    assert np.array_equal(e0, d["e0"]) and np.array_equal(d["event"], d["e1"][0])
    assert abs(d["e1"].mean(0) - np.array(S.NP_SHIFT)).max() < 0.05
    assert abs(d["e0"].std(0) - np.array(S.NP_SIGMA)).max() < 0.03


def test_likelihood_ratio_has_the_highest_power():
    d = S.np_data()
    m = study(NP).metadata
    x, both, lr = m["power_exact"]
    assert lr > both + 0.05 and both > x + 0.05       # exact, at alpha = 5%
    x, both, lr = m["power_pseudo"]
    assert lr > both + 0.04 and both > x + 0.04       # and on 20,000 pseudo-events
    for j in range(3):
        assert within(m["power_pseudo"][j], m["power_exact"][j], S.NP_N)
        assert within(d["alpha_mc"][j], S.NP_ALPHA, S.NP_N)
    exact, measured = d["roc_exact"], d["roc_mc"]
    assert np.all(exact[2] >= exact[1] - 1e-12) and np.all(exact[2] >= exact[0] - 1e-12)
    inside = (d["alphas"] > 0.01) & (d["alphas"] < 0.3)      # further on all three near 1
    assert np.all(exact[2][inside] > exact[:2].max(0)[inside] + 0.02)
    assert np.all(measured[2] >= measured[:2].max(0) - 0.012)
    assert np.abs(measured - exact).max() < 0.03
    assert np.isclose(m["separation"][2], math.hypot(1.5 / 1.0, 2.4 / 2.0))


def test_no_direction_beats_the_likelihood_ratio():
    d = S.np_data()
    m = study(NP).metadata
    best = m["power_exact"][2]
    assert np.all(d["fine_power"] <= best + 1e-12) and np.all(d["ang_exact"] <= best + 1e-12)
    assert abs(m["best_angle"] - m["lr_angle"]) <= 0.5
    assert np.isclose(m["lr_angle"], math.degrees(math.atan2(0.6, 1.5)))
    assert np.abs(d["ang_mc"] - d["ang_exact"]).max() < 0.015
    i0, i45 = (list(S.NP_ANGLES).index(v) for v in (0, 45))
    assert np.isclose(d["ang_exact"][i0], m["power_exact"][0])
    assert np.isclose(d["ang_exact"][i45], m["power_exact"][1])


def test_a_monotonic_function_of_the_ratio_is_the_same_test():
    d = S.np_data()
    m = study(NP).metadata
    assert m["monotonic_same_events"] is True
    assert m["monotonic_power"] == m["power_pseudo"][2]
    assert d["post_alpha"] == d["alpha_mc"][2] and 0 < d["post_cut"] < 1


def test_neyman_pearson_numbers_on_screen():
    a = anim(NP)
    d, m = S.np_data(), a.study.metadata
    two = late(a, 1)
    assert "L₁ / L₀ = %.1f" % (d["event_l1"] / d["event_l0"]) in two
    (a0, b0), (a1, b1) = a.factors
    assert np.isclose(a0 * b0, d["event_l0"]) and np.isclose(a1 * b1, d["event_l1"])
    assert "= %.4f × %.4f = %.5f" % (a1, b1, d["event_l1"]) in two
    three = late(a, 2)
    for j in range(3):
        assert "power %.1f%%" % (100 * m["power_pseudo"][j]) in three
        assert "exact %.1f%%" % (100 * m["power_exact"][j]) in three
    assert any(s.startswith("cut direction %d°" % round(m["lr_angle"])) for s in late(a, 4))
    assert np.isclose(a.angle_at(1.0), m["lr_angle"])
    assert any("power %.1f%%" % (100 * m["monotonic_power"]) in s for s in late(a, 4))


# ---------------------------------------------------------------------- 144
def test_local_significance_is_an_exact_poisson_tail():
    bw, ptab, ztab = S.lee_tables()
    b = S.lee_background()
    assert len(bw) == S.LEE_BINS - S.LEE_WIDTH + 1 == 57
    assert np.isclose(bw[0], b[:4].sum()) and np.isclose(bw[-1], b[-4:].sum())
    assert np.isclose(b[0], 200 * math.exp(-0.5 / 30)) and np.all(np.diff(b) < 0)
    w, n = 28, 358
    assert np.isclose(ptab[w, n], S.poisson_pmf(bw[w], 2000)[n:].sum(), rtol=1e-9)
    assert np.isclose(S.norm_sf(ztab[w, n]), ptab[w, n], rtol=1e-8)
    assert np.all(np.diff(ztab, axis=1) >= 0)         # more events, more significant


def test_shown_spectrum_is_the_first_past_three_sigma():
    d = S.lee_data()
    m = study(LEE).metadata
    rng = np.random.default_rng(S.LEE_SEED)
    counts = rng.poisson(S.lee_background(), (S.LEE_TOYS, S.LEE_BINS))
    k = m["shown_experiment"] - 1
    assert np.array_equal(counts[k], d["counts"])
    assert np.array_equal(counts[:S.LEE_TRACES], d["trace_counts"])
    assert np.all(d["zmax"][:k] < S.LEE_Z) and d["zmax"][k] >= S.LEE_Z
    w = m["shown_window"]
    assert m["shown_n"] == counts[k, w:w + S.LEE_WIDTH].sum()
    assert np.isclose(m["shown_b"], S.lee_background()[w:w + S.LEE_WIDTH].sum())
    assert np.isclose(S.norm_sf(m["shown_z"]), m["shown_p"], rtol=1e-8) and m["shown_z"] >= 3
    assert np.isclose(d["z_first"].max(), m["shown_z"]) and int(d["z_first"].argmax()) == w
    assert np.allclose(d["traces"].max(1), d["zmax"][:S.LEE_TRACES])


def test_scanning_makes_three_sigma_common():
    d = S.lee_data()
    m = study(LEE).metadata
    assert m["p_global_3"] > 20 * m["p_local_3"]      # measured: 38 times
    assert np.isclose(m["trials_factor_3"], m["p_global_3"] / m["p_local_3"])
    assert m["count_3"] == int((d["zmax"] >= 3.0).sum()) > 3000
    assert 1.5 < m["median_max"] < 2.5
    assert m["z_global_3"] < 3.0 - 1.0 and m["z_global_4"] < 4.0 - 0.7
    assert np.isclose(S.norm_sf(m["z_global_3"]), m["p_global_3"])
    assert np.all(np.diff(d["p_glob"]) <= 0) and np.all(d["p_glob"] >= d["p_apart"])


def test_effective_number_of_places_lies_between_the_two_counts():
    d = S.lee_data()
    m = study(LEE).metadata
    i3 = d["i3"]
    assert len(d["apart"]) == 15 and np.all(np.diff(d["apart"]) == S.LEE_WIDTH)
    # windows that share no bin are independent: there the exact answer exists
    for i in (lee for lee in range(len(d["grid"])) if d["grid"][lee] <= 3.5):
        assert within(d["p_apart"][i], d["apart_exact"][i], S.LEE_TOYS)
    assert d["apart_exact"][i3] * 1.5 < m["p_global_3"] < d["all_indep"][i3] / 1.15
    assert 15 + 10 < m["n_eff_3"] < 57 - 5 and 15 + 10 < m["n_eff_4"] < 57
    assert np.isclose(1 - (1 - d["p_mean"][i3]) ** m["n_eff_3"], m["p_global_3"])
    assert d["p_mean"][i3] < m["p_local_3"] < 1.2 * d["p_mean"][i3]    # counts are whole numbers
    assert 4.0 < m["z_global_5_estimate"] < 4.5
    assert np.abs(d["z_fit"] - d["z_glob"])[d["grid"] >= 2.5].max() < 0.12


def test_look_elsewhere_numbers_on_screen():
    a = anim(LEE)
    d, m = S.lee_data(), a.study.metadata
    one = late(a, 0)
    assert "Z = %.1fσ" % m["shown_z"] in one and "counted:   n = %d" % m["shown_n"] in one
    assert "expected:  b = %.1f" % m["shown_b"] in one
    two = late(a, 1)
    assert "window %d of 57" % (m["shown_window"] + 1) in two and "n = %d" % m["shown_n"] in two
    assert a.sums[m["shown_window"]] == m["shown_n"] and np.allclose(a.zs, d["z_first"])
    assert "largest ≥ 3σ in %s of 100,000: %.1f%%" % (
        format(m["count_3"], ","), 100 * m["p_global_3"]) in late(a, 2)
    four = late(a, 3)
    assert "global p = %.4f  (measured)" % m["p_global_3"] in four
    assert "trials factor = %.4f / %.5f = %d" % (
        m["p_global_3"], m["p_local_3"], round(m["trials_factor_3"])) in four
    five = late(a, 4)
    assert "global %.1fσ" % m["z_global_3"] in five and "global %.1fσ" % m["z_global_4"] in five
    assert "global ≈ %.1fσ" % m["z_global_5_estimate"] in five
    assert a.toys_at(1.0) == S.LEE_TOYS

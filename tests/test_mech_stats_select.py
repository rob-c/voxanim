"""Animations 137-140 (voxanim.mech_stats_select) and their experiments
(voxanim.study_stats_select): the layout is clean, the loop closes, every phase says what
kind of evidence it shows, the numbers on screen are the study's, the special functions
match known values, each study's claim holds as a relative claim with a margin, and the
recorded arrays are what they say they are.

None of the four needs data: they are built here from an empty cache. Every event is
pseudo-data from a model written in the study's metadata.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from voxanim import cache, layout, presets, qa
from voxanim import mech_stats_select as M
from voxanim import study_stats_select as S
from voxanim.next_catalog import NO_DATA, build_study

NAMES = ("moving-a-cut", "cuts-vs-multivariate", "sideband-background", "selection-sculpting")
EACH = [pytest.param(n) for n in NAMES]


def study(name):
    return build_study("study_stats_select", name)


def subtitles(anim):
    return [anim.subtitle_for(k) for k in range(len(anim.PHASES))]


def texts(anim, t):
    return [x.s for x in anim.scene(t).texts]


def mid(anim, k, at=0.97):
    a, b = anim.spans[k]
    return min(a + (b - a) * at, b - anim.FADE - 0.05)


# ---------------------------------------------------------------------- all four
def test_registered_and_need_no_data():
    assert set(NAMES) <= NO_DATA
    assert set(M.ANIMATIONS) == set(S.STUDIES) == set(NAMES)


@pytest.mark.parametrize("name", EACH)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", EACH)
def test_drawn_by_this_module_and_loop_closes(name):
    anim = presets.get(name).build()
    assert type(anim) is M.ANIMATIONS[name]
    assert 45.0 <= anim.t_done <= 60.0

    def summary(t):
        sc = anim.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(anim.duration)
    for a, b in anim.spans:                           # every phase draws things, not only text
        assert len(anim.scene((a + b) / 2).quads) >= 3


@pytest.mark.parametrize("name", EACH)
def test_subtitles_fit_two_lines_and_every_phase_is_tagged(name):
    anim = presets.get(name).build()
    for sub in subtitles(anim):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    for k in range(len(anim.PHASES)):                 # exact, pseudo-data or pseudo-experiments
        for at in (0.2, 0.5, 0.9):
            tags = [t.s for t in anim.scene(mid(anim, k, at)).texts     # bottom right
                    if t.anchor == "rm" and t.opacity > 0.9 and t.s.startswith(M.KINDS)]
            assert len(tags) == 1, (k, at, tags)
    assert M.toys(5000) == "pseudo-experiments: 5,000 seeded repeats"


@pytest.mark.parametrize("name", EACH)
def test_study_says_what_was_run(name):
    s = study(name)
    m = s.metadata
    assert len(s.panels) == 4 and m["designs_tried"] and m["limitations"]
    assert m["evidence"] and m["model"] and m["seeds"]
    assert any(k.startswith("pseudo") for k in m["evidence"])


def test_studies_build_from_an_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_CACHE", str(tmp_path))
    for name in NAMES:                                # offline (conftest): a download would raise
        M.ANIMATIONS[name](S.STUDIES[name]().validate()).scene(3.0)
    made = sorted(p.name.split("-")[1] for p in tmp_path.iterdir())
    assert made == ["cut", "mv", "sculpt", "sideband"]


# ---------------------------------------------------------------------- special functions
def test_normal_and_poisson_functions_match_known_values():
    assert np.allclose(S.norm_sf([0.0, 1.0, 1.959964, 5.0]),
                       [0.5, 0.158655254, 0.025, 2.866515719e-7], rtol=1e-6)
    assert np.allclose(S.norm_cdf([-1.0, 0.0, 1.0]), [0.158655254, 0.5, 0.841344746])
    assert np.allclose(S.norm_isf([0.5, 0.025, 0.975, 2.866515719e-7, 1.349898e-3]),
                       [0.0, 1.959964, -1.959964, 5.0, 3.0], atol=1e-6)
    z = np.linspace(-6, 8, 57)
    assert np.allclose(S.norm_isf(S.norm_sf(z)), z, atol=1e-8)
    assert np.isclose(S.norm_pdf(0.0), 1 / math.sqrt(2 * math.pi))
    assert np.isclose(S.poisson_pmf(3, 2.0), math.exp(-2) * 8 / 6)
    assert S.poisson_sf(0, 3.0) == 1.0 and np.isclose(S.poisson_sf(1, 3.0), 1 - math.exp(-3))
    exact = 1 - sum(math.exp(-1) / math.factorial(k) for k in range(11))
    assert np.isclose(S.poisson_sf(11, 1.0), exact, rtol=1e-6) and 1.0e-8 < exact < 1.01e-8
    assert np.isclose(S.poisson_sf(110, 100.0), 0.1707, atol=2e-4)
    assert [S.poisson_median(mu) for mu in (0.5, 1.0, 10.0, 100.0)] == [0, 1, 10, 100]


def test_significance_formulas():
    assert S.z_simple(10, 4) == 5.0 and np.isclose(S.z_total(10, 6), 2.5)
    assert np.isclose(S.z_asimov(10, 1), math.sqrt(2 * (11 * math.log(11) - 10)))
    assert np.isclose(S.z_asimov(1.0, 1e6), 1e-3, rtol=1e-3)        # large b: s / sqrt(b)
    assert S.z_observed(5, 5.0) == 0.0 and S.z_observed(3, 5.0) == 0.0
    assert np.isclose(S.z_observed(11, 1.0), S.z_asimov(10, 1))
    assert np.isclose(S.z_poisson(11, 1.0), 5.611, atol=2e-3)       # p = 1.005e-8


# ---------------------------------------------------------------------- 137
def test_cut_claims():
    s = study("moving-a-cut")
    best = s.metadata["best"]
    order = [best[k]["cut"] for k in ("z_total", "z_asimov", "z_simple", "purity")]
    assert order == sorted(order) and min(np.diff(order)) > 0.3     # four different cuts
    simple, _total, asimov = np.asarray(s.panels[2].data["ys"])
    assert (simple >= asimov - 1e-9).all()
    top = best["z_simple"]
    assert top["b"] < 1 and top["value"] > 1.5 * top["z_asimov"]    # overstated at small b
    assert abs(top["z_poisson"] - top["z_asimov"]) < 0.5
    grid = np.asarray(s.panels[2].data["x"])
    at = np.searchsorted(grid, np.asarray(s.panels[2].data["poisson_at"]) - 1e-9)
    assert np.abs(np.asarray(s.panels[2].data["z_poisson"]) - asimov[at]).max() < 0.75
    table = np.asarray(s.panels[2].data["table"])                   # rows: b = 100, 25, 4, 1
    assert (table[:, 0] > table[:, 1]).all() and (table[:, 1] > table[:, 2]).all()
    assert (np.diff(table[:, 0] / table[:, 2]) > 0).all()           # worse as b falls
    assert table[0, 0] / table[0, 2] < 1.06 and table[3, 0] / table[3, 2] > 1.7
    p = s.panels[3].data
    fixed, same, other = np.asarray(p["values"])
    se = np.asarray(p["standard_error"])
    assert abs(fixed - 50) < 4 * se[0] and abs(other - 50) < 4 * se[2]
    assert same > 50 + 10 and same > fixed + 20 * se[1]
    null = np.asarray(p["null_above_2"])
    assert null[1] > 3 * null[0] and abs(null[0] - float(p["null_fixed_exact"])) < 0.008
    assert np.asarray(p["s_hat"]).shape == (3, S.CUT_TOYS) and S.CUT_TOYS >= 5000


def test_cut_data_is_what_it_says():
    d = S.cut_data()
    assert len(d["xs"]) == d["hist_s"].sum() == S.CUT["S"]
    assert len(d["xb"]) == d["hist_b"].sum() == S.CUT["B"]
    for c in (0.0, 1.0, 2.47):
        i = int(np.argmin(np.abs(S.CUT_GRID - c)))
        assert d["count_s"][i] == (d["xs"] >= c).sum() and d["count_b"][i] == (d["xb"] >= c).sum()
    s, b = S.cut_expected(np.array([-50.0, 0.0, 2.0]))
    assert np.allclose(s, [50, 50 * S.norm_sf(-2.0), 25]) and np.allclose(b[:2], [1000, 500])
    picks = d["sig_pick"]
    assert (picks[0] == picks[0][0]).all() and len(np.unique(picks[1])) > 5
    assert S.CUT_TUNE[int(picks[0][0])] == float(d["fixed_cut"])
    rng = np.random.default_rng(S.CUT_SEED)                         # the seeded draw, again
    xs, xb = S.cut_sample(rng, S.CUT["S"], S.CUT["B"])
    assert np.array_equal(xs, d["xs"]) and np.array_equal(xb, d["xb"])


def test_cut_numbers_on_screen():
    s, d = study("moving-a-cut"), S.cut_data()
    anim = M.MovingACut(s)
    sub = subtitles(anim)
    ns, nb = int((d["xs"] >= 1.0).sum()), int((d["xb"] >= 1.0).sum())
    assert "s = %d signal and b = %d background" % (ns, nb) in sub[0]
    assert "of %.1f" % (ns / math.sqrt(nb)) in sub[0]
    best = s.metadata["best"]
    assert "x > %.2f for s/√(s+b), %.2f for the" % (best["z_total"]["cut"],
                                                   best["z_asimov"]["cut"]) in sub[2]
    assert "claims %.1fσ" % best["z_simple"]["value"] in sub[3]
    assert "averages %.0f where the truth is 50" % s.panels[3].data["values"][1] in sub[4]
    shown = texts(anim, mid(anim, 0))
    assert "s = %d signal,  b = %d background" % (ns, nb) in shown
    assert "purity: s / (s + b) = %d / %d = %.0f%%" % (ns, ns + nb, 100 * ns / (ns + nb)) in shown
    shown = texts(anim, mid(anim, 3))
    assert {"%.1f" % v for v in np.asarray(s.panels[2].data["table"])[:, :3].ravel()} <= set(shown)
    shown = texts(anim, mid(anim, 4))
    for mean, se in zip(s.panels[3].data["values"], s.panels[3].data["standard_error"]):
        assert "mean %.1f ± %.1f" % (mean, se) in shown
    assert M.toys(S.CUT_TOYS) in shown


# ---------------------------------------------------------------------- 138
def test_two_variable_claims():
    s = study("cuts-vs-multivariate")
    at = int(np.argmin(np.abs(S.MV_EFF - S.MV_AT)))
    p = s.panels[1].data
    rect, line = np.asarray(p["rect_seeds"]), np.asarray(p["line_seeds"])
    assert rect.shape == line.shape == (len(S.MV_SEEDS), len(S.MV_EFF)) and len(S.MV_SEEDS) >= 3
    assert (line[:, at] < rect[:, at] - 0.15).all()                 # every seed, at 80%
    assert (line.mean(0) < rect.mean(0)).all()                      # ... and at every efficiency
    _, _, rect_exact, line_exact = np.asarray(p["ys"])
    assert np.abs(rect.mean(0) - rect_exact).max() < 0.02           # measured against exact
    assert np.abs(line.mean(0) - line_exact).max() < 0.02
    assert abs(rect[:, at].mean() - np.asarray(p["one_variable"])[at]) < 0.03   # cuts gain nothing
    assert np.abs(np.asarray(p["rect_efficiency"]) - S.MV_EFF).max() < 0.03
    assert np.abs(np.asarray(p["line_efficiency"]) - S.MV_EFF).max() < 0.03
    f = s.metadata["fisher"]
    assert np.abs(np.asarray(f["w"]) - np.asarray(f["exact_w"])).max() < 0.15
    assert np.isclose(f["distance"], 1 / math.sqrt(1 - S.MV["rho"] ** 2))
    p = s.panels[3].data
    assert (np.asarray(p["curve_seeds"])[:, at] < np.asarray(p["line_seeds"])[:, at] - 0.15).all()
    auc = np.asarray(p["auc"])
    assert (auc[:, 1] > auc[:, 0] + 0.1).all() and auc[:, 1].min() > 0.97


def test_two_variable_data_is_what_it_says():
    d = S.mv_data()
    rng = np.random.default_rng(S.MV_SEEDS[0])
    sig, bkg = S.mv_sample(rng, S.MV_TRAIN)
    w, ms, mb, w_mat = S.fisher(sig, bkg)
    assert np.allclose(d["w"], w) and np.allclose(d["mean_s"], ms)
    assert np.allclose(d["w_mat"], w_mat)
    assert np.allclose(w, np.linalg.inv(w_mat) @ (ms - mb))
    assert np.array_equal(d["shown_s"], sig[:S.MV_SHOWN])
    assert abs(np.corrcoef(bkg.T)[0, 1] - S.MV["rho"]) < 0.02
    cut = d["rect_cut"]
    assert S.rect_apply(sig, cut).mean() >= S.MV_AT                  # chosen on these events
    assert (sig @ w > float(d["line_thr"])).mean() == pytest.approx(S.MV_AT, abs=1e-3)
    kept = S._kept(np.array([[1.0, 2.0], [3.0, 4.0]]))              # by hand: 2 x 2 counts
    assert kept.shape == (4, 3, 3) and np.allclose(kept[:, 1, 1], [0.4, 0.3, 0.2, 0.1])
    assert np.allclose(kept[0, 0, 0], 1.0) and np.allclose(kept[3, 2, 2], 1.0)
    x = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0]])
    assert list(S.rect_apply(x, (1, 1.0, 1.0))) == [False, True, False]      # x1 > 1, x2 < 1
    assert S.auc(np.array([2.0, 3.0]), np.array([0.0, 1.0])) == 1.0
    assert S.auc(np.array([0.0, 3.0]), np.array([1.0, 2.0])) == 0.5
    x = np.array([[0.0, -1.2], [0.0, 0.6]])                         # on the crescent; on the blob
    ratio = S.banana_log_ratio(x)
    assert ratio[0] > 3 and ratio[1] < -3
    assert d["ban_region"].shape == (len(S.BAN_GRID[1]), len(S.BAN_GRID[0]))


def test_two_variable_numbers_on_screen():
    s = study("cuts-vs-multivariate")
    anim = M.CutsVsMultivariate(s)
    sub = subtitles(anim)
    kept = np.asarray(s.panels[2].data["values"])
    w = np.asarray(s.panels[0].data["w"])
    assert "still keeps %.0f%% of the background" % (100 * kept[1]) in sub[1]
    assert "t = %.2f x₁ − %.2f x₂ keeps %.0f%%" % (w[0], -w[1], 100 * kept[2]) in sub[2]
    assert "%.1f%% against %.1f%% at 80%%" % (100 * kept[2], 100 * kept[1]) in sub[3]
    c = s.metadata["crescent_at"]
    assert "keeps %.0f%% of the background, a curved boundary %.1f%%" % (
        100 * c["line"], 100 * c["curve"]) in sub[4]
    shown = texts(anim, mid(anim, 2))
    inv = np.linalg.inv(np.asarray(s.panels[0].data["w_mat"]))
    for v in list(np.asarray(s.panels[0].data["w_mat"]).ravel()) + list(inv.ravel()):
        assert M.minus("%.2f" % v) in shown
    assert "w = W⁻¹ × difference = (%.2f, −%.2f)" % (w[0], -w[1]) in shown
    n = S.MV_SHOWN
    d = S.mv_data()
    assert {"%d signal" % d["shown_counts"][2],
            "%d background kept, of %d each" % (d["shown_counts"][3], n)} <= set(shown)
    assert any(t.startswith("pseudo-data (dots") for t in texts(anim, mid(anim, 3)))


# ---------------------------------------------------------------------- 139
def test_sideband_claims():
    s = study("sideband-background")
    right, wrong, tails = (s.metadata["results"][k] for k in S.SB_CASES)
    se = right["spread"] / math.sqrt(S.SB_TOYS)
    assert S.SB_TOYS >= 5000 and abs(right["bias"] - right["bias_expected"]) < 4 * se
    assert abs(right["bias"]) < 0.1 * right["spread"]               # unbiased: 0.04 sigma
    assert abs(right["pull_width"] - 1) < 0.05 and abs(right["within_1_sigma"] - 0.6827) < 0.02
    assert abs(right["mean_quoted_error"] / right["spread"] - 1) < 0.05
    assert wrong["bias"] < -1.5 * wrong["spread"]                   # a line: far too low
    assert abs(wrong["bias"] - wrong["bias_expected"]) < 5
    assert tails["bias"] < -0.5 * tails["spread"]
    assert abs(tails["bias"] - tails["bias_expected"]) < 5
    assert wrong["within_1_sigma"] < 0.3 and tails["within_1_sigma"] < right["within_1_sigma"]
    assert wrong["chi2_per_ndf"] > right["chi2_per_ndf"] + 0.2
    hist = np.asarray(s.panels[1].data["ys"])
    assert (hist.sum(1) > 0.995 * S.SB_TOYS).all()                  # the histograms hold them all


def test_sideband_data_is_what_it_says():
    bkg, sig = S.sb_expected()
    window, side = S.sb_masks()
    assert np.isclose(bkg.sum(), S.SB["B"]) and abs(sig.sum() - S.SB["S"]) < 1e-6
    assert window.sum() == 10 and side.sum() == 30 and not (window & side).any()
    assert np.isclose(bkg[1] / bkg[0], math.exp(-1 / S.SB["tau"]))
    # a spectrum that is exactly the assumed shape, with no signal: the yield is zero
    assert abs(S.sb_estimate(bkg, "log")["s"][0]) < 1e-6
    line = 200.0 - 2.0 * (S.SB_M - 100.0)
    est = S.sb_estimate(line, "line")
    assert abs(est["s"][0]) < 1e-6 and np.allclose(est["p"][0], [150.0, -2.0])
    p, cov = S.poisson_fit(np.exp(3.0 - 0.05 * np.arange(20.0))[None], np.arange(20.0), "log")
    assert np.allclose(p[0], [3.0, -0.05]) and cov.shape == (1, 2, 2) and cov[0, 0, 0] > 0
    # the quoted error of the fitted background against its spread over pseudo-experiments
    rng = np.random.default_rng(1)
    est = S.sb_estimate(rng.poisson(bkg, (2000, len(bkg))), "log")
    assert abs(est["b_err"].mean() / est["b"].std() - 1) < 0.08
    d = S.sb_data()
    one = S.sb_estimate(d["data"], "log")
    assert np.isclose(one["n"][0], d["data"][window].sum()) and np.isclose(
        one["s"][0], d["one_0"][3])
    assert np.isclose(float(d["truth_0"]), sig[window].sum())
    assert np.array_equal(d["data"], np.random.default_rng(S.SB_SEED).poisson(bkg + sig))


def test_sideband_numbers_on_screen():
    s = study("sideband-background")
    anim = M.SidebandBackground(s)
    sub = subtitles(anim)
    n, b, b_err, sig, sig_err = np.asarray(s.panels[0].data["one"])[0][:5]
    res = s.metadata["results"]
    assert "N = %s events" % M.commas(n) in sub[0]
    assert "is %s ± %.0f" % (M.commas(round(b)), b_err) in sub[1]
    assert "%.0f ± %.0f events" % (sig, sig_err) in sub[1]
    assert "%.1f%% of the results" % (100 * res["right shape"]["within_1_sigma"]) in sub[2]
    assert "%.0f events too low on average" % -res["wrong shape"]["bias"] in sub[3]
    assert "%.0f events too low" % -res["signal in the sidebands"]["bias"] in sub[4]
    shown = texts(anim, mid(anim, 1))
    assert "signal = N − b = %.0f" % sig in shown
    assert "error = √(N + %.0f²) = √(%s + %.0f) = %.0f" % (
        b_err, M.commas(n), b_err ** 2, sig_err) in shown
    assert abs(sig - (n - b)) < 1e-9 and np.isclose(sig_err, math.sqrt(n + b_err ** 2))
    assert M.toys(S.SB_TOYS) in texts(anim, mid(anim, 2))


# ---------------------------------------------------------------------- 140
def test_sculpting_claims():
    s = study("selection-sculpting")
    true, rec, _trig = np.asarray(s.panels[0].data["ys"])
    eff = np.asarray(s.panels[0].data["efficiency"])
    energy = S.SC_E
    assert (np.diff(true) < 0).all()                                # the truth only falls
    peak = int(np.argmax(rec))
    assert 0 < peak < len(rec) - 1 and rec[peak] > 5 * rec[peak - 4]        # a peak, made
    assert abs(s.metadata["recorded_peak"]["exact"] - S.SC_EDGES[peak]) < 4
    ratio, spread, formula = np.asarray(s.panels[1].data["ys"])
    high = energy > S.SC["threshold"] + 8
    assert np.abs(ratio[high] - 1).max() < 0.005 and spread[high].max() < 0.06
    ok = eff > 0.003
    assert np.abs(spread[ok] / formula[ok] - 1).max() < 0.1         # the formula holds
    low = (eff < 0.002) & (np.asarray(s.panels[1].data["usable"]) > 0.98)
    assert low.any() and spread[low].min() > 0.3 and (ratio[low] > 1.1).all()
    assert np.asarray(s.panels[1].data["usable"]).min() < 0.5
    peaks = np.asarray(s.panels[3].data["values"])
    assert (np.diff(peaks) > 5).all()                               # the peak follows the cut
    assert 1.5 < s.metadata["pair_peak_slope"] < 2.6
    assert float(s.panels[3].data["correlation"]) > 0.5
    assert (np.diff(np.asarray(s.panels[3].data["kept"])) < 0).all()


def test_sculpting_data_is_what_it_says():
    d = S.sc_data()
    nu, eff = S.sc_expected()
    assert np.isclose(nu.sum(), S.SC["n_true"]) and d["true"].sum() == S.SC["n_true"]
    assert (d["recorded"] <= d["triggered"]).all() and (d["triggered"] <= d["true"]).all()
    assert np.abs(d["true"] - nu).max() < 5 * np.sqrt(nu.max())
    high = S.SC_E > 60.0
    assert np.allclose(eff[high], S.SC["acceptance"], atol=1e-3)
    assert abs(d["recorded"][high].sum() / d["true"][high].sum() - S.SC["acceptance"]) < 0.02
    assert S.turn_on(S.SC["threshold"]) == 0.5 and S.efficiency(1e3) == S.SC["acceptance"]
    tau, w = S.SC["tau"], S.SC["width"]
    assert abs(S.sc_peak() - (S.SC["threshold"] + w * math.log(tau / w - 1))) < 0.01
    value, err = S.corrected(100.0, 0.5, 0.05)
    assert value == 200.0 and np.isclose(err, 200 * math.sqrt(1 / 100 + 0.01))
    k = 15
    eff_hat = d["cal_pass"][k] / S.SC["calibration"]
    assert np.isclose(d["corrected"][k], d["recorded"][k] / eff_hat)
    assert np.isclose(S.pair_mass(10.0, 10.0, 0.0, np.pi), 20.0)
    assert S.pair_mass(10.0, 10.0, 0.0, 0.0) == 0.0
    assert S.peak_of(np.array([0, 1, 5, 9, 5, 1, 0, 0.0]), np.arange(9.0), smooth=3) == 3.5
    rng = np.random.default_rng(S.PAIR_SEED)
    pt = rng.exponential(S.PAIR["pt_mean"], (2, S.PAIR["n"]))
    assert np.allclose(d["pair_soft"], pt.min(0)[:S.PAIR_SHOWN])
    assert np.isclose(d["pair_kept"][3], math.exp(-2 * 20 / S.PAIR["pt_mean"]), atol=2e-3)


def test_sculpting_numbers_on_screen():
    s = study("selection-sculpting")
    anim = M.SelectionSculpting(s)
    sub = subtitles(anim)
    true, rec, _ = np.asarray(s.panels[0].data["ys"])
    w, hi = anim.WORKED, anim.HIGH
    eff = np.asarray(s.panels[0].data["efficiency"])
    assert "%s true events × efficiency %.3f: %s were recorded" % (
        M.commas(true[w]), eff[w], M.commas(rec[w])) in sub[0]
    assert abs(rec[w] - true[w] * eff[w]) < 4 * math.sqrt(true[w] * eff[w])
    peak = int(np.argmax(rec))
    assert "peaks at %d–%d GeV" % (S.SC_EDGES[peak], S.SC_EDGES[peak + 1]) in sub[1]
    fix, err = (np.asarray(s.panels[1].data[k]) for k in ("corrected", "corrected_err"))
    assert "%s ± %.0f for a true %s" % (M.commas(round(fix[hi])), err[hi],
                                        M.commas(true[hi])) in sub[2]
    assert "%.1f GeV higher for each GeV" % s.metadata["pair_peak_slope"] in sub[4]
    shown = texts(anim, mid(anim, 0))
    assert "%s true events" % M.commas(true[w]) in shown
    assert "recorded: %s" % M.commas(rec[w]) in shown
    shown = texts(anim, mid(anim, 4))
    peaks = np.asarray(s.panels[3].data["values"])
    assert any(t.startswith("> 30 GeV: peak at %.0f GeV" % peaks[-1]) for t in shown)
    assert M.toys(S.SC_TOYS) in texts(anim, mid(anim, 3))

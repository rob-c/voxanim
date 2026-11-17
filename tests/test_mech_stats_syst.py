"""Animations 133-136 (voxanim.mech_stats_syst) and their experiments
(voxanim.study_stats_syst): the layout is clean, the loop closes, every phase says what
kind of evidence it shows, the numbers on screen are the study's, each study's claim
holds as a relative claim with a margin on every seed, the few special functions agree
with known values, and the recorded arrays are what they say they are."""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest

from voxanim import cache, layout, presets, qa
from voxanim import mech_stats_syst as M
from voxanim import study_stats_syst as S
from voxanim.next_catalog import NO_DATA

ELLIPSE, SYST = "error-ellipse", "statistical-vs-systematic"
NUISANCE, COMBINE = "nuisance-parameters-profiling", "combining-measurements"
NAMES = (ELLIPSE, SYST, NUISANCE, COMBINE)
EACH = [pytest.param(n, id=n) for n in NAMES]


@functools.cache
def study(name):
    return S.STUDIES[name]().validate()


@functools.cache
def anim(name):
    return presets.get(name).build()


def subtitles(a):
    return [a.subtitle_for(k) for k in range(len(a.PHASES))]


def texts(a, t):
    return [x.s for x in a.scene(t).texts if x.opacity > 0.5]


def late(a, k, at=0.97):
    lo, hi = a.spans[k]
    return min(lo + (hi - lo) * at, hi - a.FADE - 0.05)


def on_screen(a, k, s, at=0.97):
    """True when some text of phase ``k`` contains ``s``."""
    return any(s in t for t in texts(a, late(a, k, at)))


# ---------------------------------------------------------------------- all four
def test_registered_and_need_no_data():
    assert set(M.ANIMATIONS) == set(S.STUDIES) == set(NAMES)
    assert set(NAMES) <= NO_DATA


@pytest.mark.parametrize("name", EACH)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", EACH)
def test_drawn_by_this_module_and_loop_closes(name):
    a = anim(name)
    assert type(a) is M.ANIMATIONS[name]
    assert 45.0 <= a.t_done <= 60.0

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:                            # every phase draws things, not only text
        sc = a.scene((lo + hi) / 2)                   # (error bars and curves are lines)
        assert len(sc.quads) >= 1 and len(sc.quads) + len(sc.lines) >= 12


@pytest.mark.parametrize("name", EACH)
def test_subtitles_fit_two_lines_and_every_phase_is_tagged(name):
    a = anim(name)
    for sub in subtitles(a):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    for k in range(len(a.PHASES)):                    # what kind of evidence is on screen
        for at in (0.3, 0.6, 0.97):
            tags = [t for t in texts(a, late(a, k, at))
                    if any(t == kind or t.startswith(kind + ": ") for kind in M.KINDS)]
            assert len(tags) == 1, (name, k, at)


@pytest.mark.parametrize("name", EACH)
def test_study_says_what_was_run(name):
    m = study(name).metadata
    assert len(study(name).panels) == 4 and len(m["seeds"]) >= 3
    assert len(m["by_seed"]) == len(m["seeds"]) and m["toys"] >= 4000
    assert m["designs_tried"] and m["limitations"] and m["model"]
    assert set(m["evidence"]) == {"exact", "pseudo_experiments"}
    assert "default_rng" in m["where_the_seed_enters"]


def test_studies_build_from_an_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_CACHE", str(tmp_path))
    for name in NAMES:                                # offline (conftest): a download would raise
        M.ANIMATIONS[name](S.STUDIES[name]().validate()).scene(3.0)
    made = {p.name.split("-")[2] for p in tmp_path.iterdir()}
    assert made == {"ellipse", "syst", "nuisance", "combine"}


def test_special_functions_against_known_values():
    np.testing.assert_allclose(S.gauss_within([1.0, 2.0, 3.0]),
                               [0.6826895, 0.9544997, 0.9973002], atol=1e-7)
    assert pytest.approx(0.6826895, abs=1e-7) == S.ONE_SIGMA
    # chi-square with 2 degrees of freedom: 39.35% below 1; 5.991 is the 95% point
    np.testing.assert_allclose(S.chi2_cdf_2dof([1.0, 2.2957, 5.9915]),
                               [0.3934693, 0.6826895, 0.95], atol=2e-5)
    assert pytest.approx(math.log(2.5 ** 3 * math.exp(-2.5) / 6)) == S.log_poisson(3, 2.5)
    assert pytest.approx(1.0, abs=1e-12) == np.exp(S.log_poisson(np.arange(80), 7.0)).sum()
    x = np.linspace(-3, 3, 601)
    left, right, best = S.crossings(x, (x - 0.4) ** 2 / 0.25, 1.0)
    assert (left, right, best) == pytest.approx((-0.1, 0.9, 0.4), abs=1e-3)


# ---------------------------------------------------------------------- 133: error ellipse
def test_ellipse_fit_and_covariance_are_least_squares():
    d = S.ellipse_data()
    a, b = S.line_fit(S.ELL_X, d["y"])
    for k in range(3):
        slope, intercept = np.polyfit(S.ELL_X, d["y"][k], 1)
        assert (a[k], b[k]) == pytest.approx((intercept, slope))
        assert tuple(d["fits"][k]) == pytest.approx((intercept, slope))
    np.testing.assert_allclose(S.line_cov(S.ELL_X, S.SIGMA_Y),
                               [[0.275, -0.075], [-0.075, 0.025]], atol=1e-12)
    # the recorded data are the seed's: the first rows of the same draw
    rng = np.random.default_rng(S.ELL_SEED)
    y = S.A_TRUE + S.B_TRUE * S.ELL_X + rng.normal(0.0, S.SIGMA_Y, (S.TOYS, 5))
    np.testing.assert_array_equal(y[:3], d["y"])


def test_ellipse_contour_is_where_chi_square_rises_by_one():
    d = S.ellipse_data()
    rise = S.chi2_line(d["ring"][:, 0], d["ring"][:, 1], S.ELL_X, d["y"][0], S.SIGMA_Y)
    np.testing.assert_allclose(rise - d["chi_min"][0], 1.0, atol=1e-9)
    # ... and it is the exact covariance's ellipse around that fit
    off = d["ring"] - d["fits"][0]
    q = np.einsum("ni,ij,nj->n", off, np.linalg.inv(d["exact"]), off)
    np.testing.assert_allclose(q, 1.0, atol=1e-9)
    p = study(ELLIPSE).panels[2].data
    np.testing.assert_allclose((p["free_at"][1] - p["free_at"][0]) / 2,
                               np.sqrt(d["exact"][0, 0]), rtol=2e-3)
    np.testing.assert_allclose((p["held_at"][1] - p["held_at"][0]) / 2,
                               S.SIGMA_Y / np.sqrt(5), rtol=2e-3)


def test_ellipse_claims_hold_on_every_seed():
    m = study(ELLIPSE).metadata
    e = m["exact"]
    assert e["rho"] == pytest.approx(-3 / math.sqrt(11))
    assert e["sigma_a"] * math.sqrt(1 - e["rho"] ** 2) == pytest.approx(e["conditional"])
    for row in m["by_seed"]:
        assert abs(row["rho"] - e["rho"]) < 0.01                    # the cloud is tilted ...
        assert abs(row["sigma_a"] - e["sigma_a"]) < 0.02
        assert row["sigma_a_fixed"] < 0.5 * row["sigma_a"]          # ... fixing b narrows a
        assert abs(row["sigma_a_fixed"] - e["conditional"]) < 0.01
        assert abs(row["rho_centred"]) < 0.03                       # ... centring removes it
        assert abs(row["inside"] - e["inside"]) < 0.015             # 39.3%, not 68.3%
    assert abs(m["inside_a"] - e["inside_one"]) < 0.015
    assert m["contour_mismatch"] < 0.03


def test_ellipse_numbers_on_screen_are_the_studys():
    a, m = anim(ELLIPSE), study(ELLIPSE).metadata
    subs = subtitles(a)
    assert "ρ = %s" % M.num(m["rho"], 3) in subs[1]
    assert M.pct(m["inside"]) in subs[2]
    assert M.num(m["conditional"], 3) in subs[3] and M.num(m["centred"]["rho"]) in subs[4]
    cov = study(ELLIPSE).panels[1].data["values"]
    shown = texts(a, late(a, 2))
    for v in (M.num(cov[0, 0], 3), M.num(cov[0, 1], 4), M.num(cov[1, 1], 4)):
        assert v in shown
    assert on_screen(a, 2, "(exact: %s)" % M.pct(m["exact"]["inside"]))
    assert on_screen(a, 0, "a = %s,  b = %s" % tuple(M.num(v) for v in m["first_fit"]), 0.2)
    assert on_screen(a, 3, "spread of a = %s" % M.num(m["conditional"], 3))
    assert on_screen(a, 4, "ρ: %s → %s" % (M.num(m["rho"], 3), M.num(m["centred"]["rho"])))
    # the ellipse drawn is the measured covariance's
    pts = M.ellipse(cov, a.truth) - a.truth
    np.testing.assert_allclose(np.einsum("ni,ij,nj->n", pts, np.linalg.inv(cov), pts), 1.0)


# ---------------------------------------------------------------------- 134: stat and syst
def test_syst_running_mean_is_the_mean_of_the_events():
    d = S.syst_data()
    np.testing.assert_allclose(np.cumsum(d["events"]) / np.arange(1, 101), d["running"][:100])
    rng = np.random.default_rng(S.SYS_SEED)
    events = S.MU_TRUE + S.SHIFT_DRAWN + rng.normal(0.0, S.RESOLUTION, S.N_MAX)
    np.testing.assert_array_equal(events[:100], d["events"])
    assert d["running"][-1] == pytest.approx(events.mean())


def test_syst_formulas():
    assert S.stat_error(400) == pytest.approx(S.SCALE_ERR)           # where the two are equal
    assert S.total_error(400) == pytest.approx(S.SCALE_ERR * math.sqrt(2))
    m = study(SYST).metadata
    assert m["crossover"] == 400 and m["control_pays_from"] == 1600
    assert S.control_error(1600) == pytest.approx(S.total_error(1600))
    assert S.control_error(10000) < S.total_error(10000)


def test_syst_claims_hold():
    s = study(SYST)
    m = s.metadata
    stat, _, total, rms = s.panels[1].data["ys"]
    np.testing.assert_allclose(rms, total, rtol=0.04)                 # quadrature sum, measured
    np.testing.assert_allclose(m["rms_shift_known"], stat, rtol=0.04)
    assert rms[-1] > 4 * stat[-1]                                     # the floor: not 1/sqrt(N)
    assert rms[-1] > 0.95 * S.SCALE_ERR
    np.testing.assert_allclose(m["cover_stat"], m["cover_stat_exact"], atol=0.03)
    np.testing.assert_allclose(m["cover_total"], S.ONE_SIGMA, atol=0.03)
    np.testing.assert_allclose(m["control"]["rms"], m["control"]["exact"], rtol=0.05)
    np.testing.assert_allclose(m["control"]["coverage"], S.ONE_SIGMA, atol=0.03)
    for row in m["by_seed"]:
        assert row["cover_stat_last"] < 0.25 < 0.6 < row["cover_total_last"]
        assert row["rms_control_last"] < 0.6 * row["rms_last"]        # the control beats the floor
    c = m["control"]
    assert c["shift"] == pytest.approx(c["mean"] - S.REFERENCE)
    assert abs(c["shift"] - S.SHIFT_DRAWN) < 3 * c["shift_error"]


def test_syst_numbers_on_screen_are_the_studys():
    a, m = anim(SYST), study(SYST).metadata
    subs = subtitles(a)
    assert "%s ± %s" % (M.num(m["mean_at"]["10000"]), M.num(m["stat_at"]["10000"])) in subs[1]
    assert M.num(m["rms"][-1], 3) in subs[2] and M.num(m["total_at"]["400"]) in subs[2]
    assert M.pct(m["cover_stat"][-1], 0) in subs[3]
    assert "%s ± %s" % (M.num(m["control"]["shift"]), M.num(m["control"]["shift_error"])) in subs[4]
    assert on_screen(a, 0, "mean = %s" % M.num(m["mean_at"]["100"]))
    assert on_screen(a, 1, "%s ± %s" % (M.num(m["mean_at"]["10000"]), M.num(0.1)))
    assert on_screen(a, 2, "measured at 10,000: %s" % M.num(m["rms"][-1], 3))
    means = study(SYST).panels[2].data["first_means"][:, -1]
    inside = int((np.abs(means - S.MU_TRUE) <= S.total_error(S.N_MAX)).sum())
    assert on_screen(a, 3, "holds the truth in %d of %d" % (inside, len(means)))
    c = m["control"]
    assert on_screen(a, 4, "= %s ± %s" % (M.num(c["corrected"]), M.num(c["corrected_error"])))


# ---------------------------------------------------------------------- 135: nuisance
def test_profiled_background_is_the_best_background():
    b = np.linspace(1.0, 120.0, 119001)
    for n in (40, 80, 110):
        for mu in (-0.5, 0.0, 1.0, 2.5):
            best = b[np.argmin(S.nll2(mu, b, n))]
            assert S.profiled_b(mu, n) == pytest.approx(best, abs=2e-3)
    # the likelihood's own maximum: the count explained exactly, background at nominal
    lo, hi, best = S.intervals(80)[1]
    assert best == pytest.approx((80 - S.B_NOMINAL) / S.SIGNAL, abs=5e-3)
    assert lo < best < hi


def test_nuisance_count_and_path_are_what_they_say():
    d, s = S.nuisance_data(), study(NUISANCE)
    rng = np.random.default_rng(S.NUI_SEED)
    b_true = rng.normal(S.B_NOMINAL, S.B_ERROR, S.TOYS)
    n = rng.poisson(S.MU_SIGNAL * S.SIGNAL + b_true)
    assert int(d["n"]) == int(n[0]) == s.metadata["count"]
    assert float(d["b_true"]) == pytest.approx(b_true[0])
    p = s.panels[0].data
    np.testing.assert_allclose(p["path_b"], S.profiled_b(p["path_mu"], int(d["n"])))
    np.testing.assert_allclose(
        p["values"], S.nll2(p["mu"][None, :], p["b"][:, None], int(d["n"]))
        - S.nll2(p["mu"][None, :], p["b"][:, None], int(d["n"])).min())
    # the profiled curve is the smallest -2 ln L over b at each mu, so never above the cut
    fixed, prof, _ = s.panels[1].data["ys"]
    assert (prof <= fixed + 1e-9).all()
    assert s.metadata["counts_outside_table"] == 0


def test_nuisance_claims_hold_on_every_seed():
    m = study(NUISANCE).metadata
    hw = m["half_width"]
    assert hw["profiled"] > hw["fixed"] + 0.15                      # profiling widens it
    assert abs(hw["marginalised"] - hw["profiled"]) < 0.02          # marginalising: the same
    assert hw["fixed"] == pytest.approx(m["rule_of_thumb"]["fixed"], abs=0.01)
    assert hw["profiled"] == pytest.approx(m["rule_of_thumb"]["profiled"], abs=0.01)
    for row in m["by_seed"]:
        fixed, prof, marg = row["coverage"]
        assert fixed < prof - 0.12                                  # fixed: over-confident
        assert abs(prof - m["nominal"]) < 0.02 and abs(marg - m["nominal"]) < 0.02
    for k in S.METHODS:                                             # toys against the exact sum
        assert abs(m["coverage"][k] - m["coverage_exact"][k]) < 0.012
    never = m["coverage_if_background_never_varied"]
    assert abs(never["fixed"] - m["nominal"]) < 0.03 < never["profiled"] - m["nominal"]


def test_nuisance_numbers_on_screen_are_the_studys():
    a, m = anim(NUISANCE), study(NUISANCE).metadata
    subs = subtitles(a)
    assert "n = %d" % m["count"] in subs[0]
    iv = m["intervals"]
    assert "%s to %s" % (M.num(iv["fixed"][0]), M.num(iv["fixed"][1])) in subs[3]
    for k in S.METHODS:
        assert M.pct(m["coverage"][k]) in subs[4]
        assert on_screen(a, 4, M.pct(m["coverage"][k]))
        assert on_screen(a, 3, "%s to %s:  ± %s" % (M.num(iv[k][0]), M.num(iv[k][1]),
                                                   M.num(m["half_width"][k])))
    assert on_screen(a, 4, ", ".join(M.pct(m["coverage_exact"][k]) for k in S.METHODS))
    # the contour drawn is where -2 ln L is 1 above its minimum
    low = S.nll2((m["count"] - S.B_NOMINAL) / S.SIGNAL, S.B_NOMINAL, m["count"])
    np.testing.assert_allclose(S.nll2(a.ring[:, 0], a.ring[:, 1], m["count"]) - low, 1.0,
                               atol=1e-8)


# ---------------------------------------------------------------------- 136: combining
def test_best_weights_minimise_the_error():
    cov = S.shared_cov()
    np.testing.assert_allclose(cov, [[1.0, 1.92], [1.92, 6.76]])
    w = S.blue_weights(cov)
    assert w.sum() == pytest.approx(1.0) and w[1] < -0.2
    grid = np.linspace(-1.0, 3.0, 40001)
    errs = [S.combined_error(np.array([g, 1 - g]), cov) for g in grid[::40]]
    assert grid[::40][int(np.argmin(errs))] == pytest.approx(w[0], abs=5e-3)
    assert S.combined_error(w, cov) <= min(errs) + 1e-12
    # independent errors: the best weights are the 1/sigma^2 weights
    diag = np.diag(np.diag(cov))
    np.testing.assert_allclose(S.blue_weights(diag), S.naive_weights(cov))
    assert S.combined_error(S.naive_weights(cov), diag) == pytest.approx(
        1 / math.sqrt(1 / 1.0 + 1 / 6.76))


def test_combine_sweep_and_pairs_are_what_they_say():
    s, d = study(COMBINE), S.combine_data()
    m = s.metadata
    rho, (w1, w2, err) = s.panels[1].data["x"], s.panels[1].data["ys"]
    np.testing.assert_allclose(w1 + w2, 1.0)
    k = int(np.argmin(np.abs(w2)))
    assert rho[k] == pytest.approx(m["negative_from"], abs=0.006)     # zero at sigma1/sigma2
    assert int(np.argmax(err)) == k and err[k] == pytest.approx(m["sigmas"][0], abs=1e-3)
    assert (w2[rho > m["negative_from"] + 0.01] < 0).all() and err[-1] < 0.6 * err[k]
    rng = np.random.default_rng(S.COM_SEED)
    own = rng.normal(0.0, 1.0, (S.TOYS, 2)) * S.STAT
    shared = rng.normal(0.0, 1.0, S.TOYS)
    np.testing.assert_allclose(d["x"], (S.X_TRUE + own + shared[:, None] * S.SENS)[:2000])
    assert m["first_combined"]["best"] == pytest.approx(
        float(np.dot(m["weights"]["best"], m["first_pair"])))
    np.testing.assert_allclose(d["cov"], S.shared_cov(), rtol=0.04)
    assert d["hist"].sum(1).min() > 0.99 * S.TOYS


def test_combine_claims_hold_on_every_seed():
    m = study(COMBINE).metadata
    q, e = m["quoted"], m["actual_exact"]
    assert e["best"] < e["alone"] - 0.08 and e["naive"] > e["alone"] + 0.1
    assert q["naive"] < e["naive"] - 0.15                             # it under-states its error
    assert q["best"] == pytest.approx(e["best"])
    for row in m["by_seed"]:
        alone, naive, best = row["spread"]
        assert best < alone - 0.08 < naive - 0.2                      # the smallest spread
        assert naive > q["naive"] + 0.15
        assert row["coverage"][1] < 0.62 and abs(row["coverage"][2] - S.ONE_SIGMA) < 0.02
    assert abs(m["coverage"]["naive"] - m["coverage_exact"]["naive"]) < 0.015
    assert m["outside_both"] == {"naive": 0.0, "best": 1.0}
    assert abs(m["independent"]["spread"] - q["naive"]) < 0.02        # right when independent
    assert abs(m["independent"]["coverage"] - S.ONE_SIGMA) < 0.02


def test_combine_numbers_on_screen_are_the_studys():
    a, m = anim(COMBINE), study(COMBINE).metadata
    subs = subtitles(a)
    assert "%s ± %s" % (M.num(m["first_combined"]["naive"]), M.num(m["quoted"]["naive"])) in subs[0]
    assert "ρ = %s" % M.num(m["rho"]) in subs[1] and M.num(m["weights"]["best"][1], 3) in subs[2]
    assert M.num(m["spread"]["naive"]) in subs[4] and M.pct(m["coverage"]["naive"]) in subs[4]
    assert on_screen(a, 0, "average = %s ± %s" % (M.num(m["first_combined"]["naive"]),
                                                  M.num(m["quoted"]["naive"])))
    assert on_screen(a, 1, "= %s" % M.num(m["covariance"][0][1]))
    assert on_screen(a, 2, "= %s" % M.num(m["weights"]["best"][0], 3))
    shown = texts(a, late(a, 4))
    for k in ("alone", "naive", "best"):
        assert M.num(m["spread"][k]) in shown and M.pct(m["coverage"][k]) in shown
    # the pair on screen in phase 3 and its combination
    x = study(COMBINE).panels[0].data["xy"][a.PAIRS - 1]
    best = float(np.dot(m["weights"]["best"], x))
    assert on_screen(a, 2, "= %s ± %s" % (M.num(best), M.num(m["quoted"]["best"])))
    assert not min(x) <= best <= max(x)

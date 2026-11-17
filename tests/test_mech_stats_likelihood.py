"""Animations 128-132 (voxanim.mech_stats_likelihood) and their experiments
(voxanim.study_stats_likelihood): the layout is clean, the loop closes, every phase says
what kind of evidence it shows, the numbers on screen are the study's, the special
functions give known values, each study's claim holds as a relative claim with a margin
over its seeds, and the recorded arrays are what they say they are.

None of the five needs a dataset: they are built here from an empty cache.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from conftest import data_marks

from voxanim import cache, layout, presets, qa
from voxanim import mech_stats_likelihood as M
from voxanim import study_stats_likelihood as S
from voxanim.next_catalog import NO_DATA, build_study

CURVE, INTERVALS, PROPAGATION, MEAN, BOOTSTRAP = NAMES = (
    "likelihood-curve", "likelihood-intervals", "error-propagation", "error-on-the-mean",
    "bootstrap-errors")
EACH = [pytest.param(n, marks=data_marks(n)) for n in NAMES]


def study(name):
    return build_study("study_stats_likelihood", name)


def anim(name):
    return presets.get(name).build()


def subtitles(a):
    return [a.subtitle_for(k) for k in range(len(a.PHASES))]


def texts(a, t):
    return [x.s for x in a.scene(t).texts if x.opacity > 0.5]


def at(a, k, u=0.97):
    lo, hi = a.spans[k]
    return min(lo + (hi - lo) * u, hi - a.FADE - 0.05)


def on_screen(a, k, u=0.97):
    return " | ".join(texts(a, at(a, k, u)))


# ---------------------------------------------------------------------- all five
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
        sc = a.scene((lo + hi) / 2)
        assert len(sc.quads) >= 1 and len(sc.quads) + len(sc.lines) >= 10


@pytest.mark.parametrize("name", EACH)
def test_subtitles_fit_two_lines_and_every_phase_is_tagged(name):
    a = anim(name)
    for sub in subtitles(a):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    evidence = a.study.metadata["evidence"]
    assert list(evidence) == [phase for phase, _ in a.PHASES]
    for k, (phase, _) in enumerate(a.PHASES):         # every phase says what kind of evidence
        for u in (0.3, 0.6, 0.97):
            tags = [t for t in texts(a, at(a, k, u))
                    if t.startswith(("exact", "pseudo-experiment"))]
            assert len(tags) == 1, (phase, tags)
            assert tags[0].split()[0].rstrip("s:") == evidence[phase].split()[0].rstrip("s")


@pytest.mark.parametrize("name", EACH)
def test_study_says_what_was_run(name):
    s = study(name)
    m = s.metadata
    assert len(s.panels) == 4 and len(m["seeds"]) >= 3
    assert m["designs_tried"] and m["limitations"] and m["model"] and m["what_was_run"]


def test_studies_build_from_an_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_CACHE", str(tmp_path))
    for name in NAMES:                                # offline (conftest): a download would raise
        M.ANIMATIONS[name](S.STUDIES[name]().validate()).scene(3.0)
    made = sorted(p.name.split("-")[2] for p in tmp_path.iterdir())
    assert made == ["bootstrap", "curve", "intervals", "mean", "propagation"]


# ---------------------------------------------------------------------- special functions
def test_special_functions_against_known_values():
    assert np.isclose(S.ONE_SIGMA, 0.6826894921)
    # a sum of n unit exponentials: n = 1 is the exponential itself; n = 2 by hand
    assert np.allclose(S.gamma_cdf(1, [0.5, 2.0]), 1 - np.exp([-0.5, -2.0]))
    assert np.isclose(S.gamma_cdf(2, 1.0)[0], 1 - 2 / math.e)
    assert np.isclose(S.gamma_cdf(3, 2.0)[0], 1 - 5 * math.exp(-2.0))
    assert S.gamma_cdf(5, 0.0)[0] == 0.0 and np.isclose(S.gamma_cdf(200, 1e4)[0], 1.0)
    assert abs(S.gamma_cdf(200, 200.0)[0] - 0.5) < 0.02       # median close to the mean
    x = np.linspace(0, 40, 40001)
    for n, scale in ((1, 2.0), (5, 0.44), (20, 0.5)):
        pdf = S.gamma_pdf(x, n, scale)
        assert np.isclose(np.trapezoid(pdf, x), 1.0, atol=1e-4)
        assert np.isclose(np.trapezoid(pdf * x, x), n * scale, atol=1e-3)
        # ... and the distribution function is the integral of the density
        assert np.isclose(np.trapezoid(pdf[x <= 3.0], x[x <= 3.0]),
                          S.gamma_cdf(n, 3.0 / scale)[0], atol=1e-4)
    assert np.isclose(S.c4(2), math.sqrt(2 / math.pi)) and np.isclose(S.c4(10), 0.97266, atol=1e-5)
    assert np.isclose(np.trapezoid(S.normal_pdf(x, 20, 2), x), 1.0, atol=1e-6)
    assert np.isclose(np.trapezoid(S.max_pdf(x), x), 1.0, atol=1e-4)
    harmonic = sum(1 / k for k in range(1, S.BS_N + 1))
    assert np.isclose(np.trapezoid(S.max_pdf(x) * x, x), S.BS_TAU * harmonic, atol=1e-3)


def test_likelihood_arithmetic():
    times = np.array([1.0, 2.0, 6.0])
    assert np.allclose(S.exp_density(times, 2.0), np.exp(-times / 2) / 2)
    for tau in (0.7, 3.0, 9.0):                       # ln L is the log of the product
        assert np.isclose(S.log_likelihood(times, tau),
                          np.log(S.exp_density(times, tau).prod()))
    grid = np.linspace(0.5, 12, 23001)
    assert abs(grid[np.argmax(S.log_likelihood(times, grid))] - times.mean()) < 1e-3
    for n in (1, 2, 5, 50, 1000):
        lo, hi = S.half_unit_ratios(n)
        assert lo < 1 < hi
        for r in (lo, hi):                            # ln L has fallen by exactly one half
            assert np.isclose(-n * (math.log(r) + 1 / r - 1), -0.5, atol=1e-9)
    lo, hi = S.half_unit_ratios(10000)                # many events: best +- best / sqrt(N)
    assert np.isclose(1 - lo, 0.01, rtol=0.02) and np.isclose(hi - 1, 0.01, rtol=0.02)
    cover, below, above = S.exact_coverage(10000, lo, hi)
    assert np.isclose(cover, S.ONE_SIGMA, atol=2e-3) and np.isclose(cover + below + above, 1)


# ---------------------------------------------------------------------- 128: the curve
def test_curve_data_is_the_seeded_draw_and_the_peak_is_the_mean():
    s = study(CURVE)
    m = s.metadata
    stream = np.round(np.random.default_rng(S.LC_SEED).exponential(S.LC_TAU, 80), 2)
    times = s.panels[0].data["times"]
    assert np.array_equal(times, stream[:5]) and m["times"] == list(times)
    assert np.isclose(m["best_tau"], times.mean())
    assert abs(m["grid_peak"] - m["best_tau"]) <= m["grid_step"] / 2 + 1e-12
    d = s.panels[0].data
    for k, tau in enumerate(d["trials"]):
        assert np.allclose(d["values"][k], S.exp_density(times, tau))
        assert np.isclose(d["product"][k], d["values"][k].prod())
        assert np.isclose(d["log_sum"][k], math.log(d["product"][k]))
    assert d["product"][2] > max(d["product"][:2]) * 1.5      # the best beats both trials
    like = s.panels[1].data["ys"][0]
    assert np.allclose(np.log(like), S.log_likelihood(times, S.LC_GRID))
    # more events: a narrower peak (the width where ln L is within 1/2 of its maximum)
    widths = [float(np.ptp(S.LC_GRID[row > -0.5])) for row in s.panels[2].data["ys"]]
    assert widths[0] > 1.5 * widths[1] > 2.2 * widths[2]
    assert np.allclose(s.panels[2].data["best"], [stream[:n].mean() for n in S.LC_MORE])


def test_curve_repeated_experiments_match_the_exact_distribution():
    s = study(CURVE)
    m = s.metadata
    assert np.isclose(m["sd_exact"], S.LC_TAU / math.sqrt(S.LC_N))
    for mean, sd in zip(m["toy_mean"], m["toy_sd"]):          # every seed
        assert abs(mean - S.LC_TAU) < 0.03 and abs(sd / m["sd_exact"] - 1) < 0.03
    hist, exact = s.panels[3].data["ys"]
    assert np.abs(hist - exact).max() < 0.03 and exact.max() > 0.4
    first = np.random.default_rng(S.LC_TOY_SEEDS[0]).exponential(S.LC_TAU, (60, S.LC_N)).mean(1)
    assert np.allclose(s.panels[3].data["first"], first)


def test_curve_numbers_on_screen():
    a = anim(CURVE)
    m = a.study.metadata
    d = a.study.panels[0].data
    subs = subtitles(a)
    assert "%.3f" % m["grid_peak"] in subs[2] and "%.3f" % m["best_tau"] in subs[2]
    assert "%.3f" % m["toy_sd"][0] in subs[4] and "%.3f" % m["sd_exact"] in subs[4]
    shown = on_screen(a, 1, 0.45)                     # the first trial lifetime
    for t, f in zip(d["times"], d["values"][0]):
        assert "f(%.2f) = %.3f" % (t, f) in shown
    assert "=  %s" % M.pow10(float(d["product"][0]), 1) in shown
    shown = on_screen(a, 1)                           # the second
    assert "f(%.2f) = %.3f" % (d["times"][0], d["values"][1][0]) in shown
    assert "=  %s" % M.pow10(float(d["product"][1]), 1) in shown
    assert "/ 5 = %.3f μs" % m["best_tau"] in on_screen(a, 2)
    assert "ln L = their sum = %s" % M.signed(d["log_sum"][2]) in on_screen(a, 3)
    assert "spread: %.3f μs" % m["toy_sd"][0] in on_screen(a, 4)
    assert abs(a._tau_slide(0.9) - m["best_tau"]) < 1e-9      # the slider comes to rest there


# ---------------------------------------------------------------------- 129: intervals
def test_intervals_are_where_ln_l_has_fallen_by_a_half():
    s = study(INTERVALS)
    m = s.metadata
    stream = np.round(np.random.default_rng(S.LI_SEED).exponential(S.LI_TAU, 200), 2)
    table = s.panels[2].data["values"]
    ratios = []
    for n, (best, minus, plus, sigma) in zip(S.LI_NS, table):
        times = stream[:n]
        assert np.isclose(best, times.mean()) and np.isclose(sigma, best / math.sqrt(n))
        top = S.log_likelihood(times, best)
        for end in (best - minus, best + plus):
            assert np.isclose(S.log_likelihood(times, end) - top, -0.5, atol=1e-8)
        assert plus > sigma > minus                   # lopsided, around the symmetric error
        ratios.append(plus / minus)
    assert np.all(np.diff(ratios) < 0)                # less so with every step in N
    assert ratios[S.LI_NS.index(5)] > 1.6 and ratios[-1] < 1.15
    assert m["few_row"] == list(table[S.LI_NS.index(5)])
    for panel, n in ((s.panels[0], S.LI_MANY), (s.panels[1], S.LI_FEW)):
        grid, (dlnl, parabola) = panel.data["x"], panel.data["ys"]
        times = stream[:n]
        assert np.allclose(dlnl, S.log_likelihood(times, grid)
                           - S.log_likelihood(times, times.mean()))
        assert dlnl.max() <= 1e-12 and np.all(dlnl[grid > times.mean() * 1.2] > parabola[
            grid > times.mean() * 1.2])               # the slow fall towards long lifetimes
    # the parabola is a better description with 50 events than with 5: one sigma from the
    # peak it should read -1/2
    off = []
    for panel, row in ((s.panels[0], m["many_row"]), (s.panels[1], m["few_row"])):
        ends = [row[0] - row[3], row[0] + row[3]]
        off.append(np.abs(np.interp(ends, panel.data["x"], panel.data["ys"][0]) + 0.5).max())
    assert off[0] < 0.4 * off[1]


def test_intervals_cover_as_the_exact_calculation_says():
    s = study(INTERVALS)
    c = s.panels[3].data
    measured, exact = c["ys"][:2], c["ys"][2:]
    assert np.abs(measured - exact).max() < 0.006             # 60,000 pseudo-experiments
    assert np.abs(c["per_seed"] - exact[None]).max() < 0.012  # each seed of 20,000
    assert np.abs(c["misses"] - c["exact_misses"]).max() < 0.006
    assert np.allclose(exact + c["exact_misses"].sum(-1), 1.0)
    i5 = S.LI_NS.index(5)
    # both kinds hold the truth about 68% of the time from 5 events on ...
    assert np.abs(exact[:, i5:] - S.ONE_SIGMA).max() < 0.01
    # ... what differs at small N is the side they miss on: the symmetric interval lies
    # below the truth far more often than above it
    (half_below, half_above), (sym_below, sym_above) = c["misses"][:, i5]
    assert sym_below > half_below + 0.04 and sym_above < half_above - 0.04
    assert sym_below > 3 * sym_above and half_below < 2 * half_above
    assert np.all(np.diff(c["exact_misses"][1, :, 0]) < 0)    # and evens out as N grows
    # the intervals drawn one by one are the first pseudo-experiments of the first seed
    sums = np.random.default_rng(S.LI_SEEDS[0]).exponential(S.LI_TAU, (S.LI_SHOWN, 200))
    assert np.allclose(s.panels[2].data["shown"], sums[:, :S.LI_FEW].mean(1))


def test_intervals_numbers_on_screen():
    a = anim(INTERVALS)
    m = a.study.metadata
    best, minus, plus, sigma = m["many_row"]
    subs = subtitles(a)
    assert "σ = %.3f μs" % sigma in subs[0] and "%.2f −%.2f +%.2f" % (best, minus, plus) in subs[1]
    assert "σ = 1/√%.2f = %.3f μs" % (m["many"] / best ** 2, sigma) in on_screen(a, 0)
    assert "τ = %.2f  −%.2f  +%.2f μs" % (best, minus, plus) in on_screen(a, 1)
    lo, hi = a._cross("many", -0.5)                   # where the falling line stops
    assert abs(lo - (best - minus)) < 2e-3 and abs(hi - (best + plus)) < 2e-3
    _, fminus, fplus, _ = m["few_row"]
    shown = texts(a, at(a, 2))
    assert "half-unit drop:  −%.2f  +%.2f μs" % (fminus, fplus) in shown
    assert "%.2f" % (fplus / fminus) in shown and "−%.2f" % fminus in shown
    i5 = S.LI_NS.index(5)
    for kind, key in enumerate(("coverage_half_unit", "coverage_symmetric")):
        assert M.pct(m[key][i5]) in subs[3]
        assert "of all 60,000: %s hold the truth" % M.pct(m[key][i5]) in on_screen(a, 3)
        hits = int(a.hits[kind].sum())
        assert "30 experiments: %d hold the truth, %d miss (red)" % (hits, 30 - hits) in \
            on_screen(a, 3)
    assert M.pct(a.miss[1, i5, 0], 0) in subs[4] and M.pct(a.miss[0, i5, 1], 0) in subs[4]


# ---------------------------------------------------------------------- 130: propagation
def test_propagation_formula_and_pairs():
    assert np.isclose(S.ratio_sigma(0.5, 0.25), 2 * math.sqrt(0.0025 + 0.0025))
    assert np.isclose(S.ratio_sigma(0.5, 0.25, 0.6), math.sqrt(0.008))
    assert np.isclose(S.ratio_sigma(0.5, 0.25, 1.0), 0.0)     # equal relative errors cancel
    z = np.random.default_rng(0).standard_normal((200000, 2))
    for rho in (-0.7, 0.0, 0.6):
        x, y = S.pair(z, 0.5, 0.25, rho)
        assert abs(x.mean() - S.EP_X) < 0.01 and abs(y.mean() - S.EP_Y) < 0.01
        assert abs(x.std() - 0.5) < 0.01 and abs(y.std() - 0.25) < 0.01
        assert abs(np.corrcoef(x, y)[0, 1] - rho) < 0.01


def test_propagation_formula_matches_the_cloud_where_the_function_is_gentle():
    s = study(PROPAGATION)
    m = s.metadata
    assert np.isclose(m["one_variable"]["formula"], 0.1)
    assert abs(m["one_variable"]["cloud"] / 0.1 - 1) < 0.02
    cloud, with_term, without = s.panels[1].data["ys"]
    per_seed = s.panels[1].data["per_seed"]
    assert np.abs(per_seed / with_term - 1).max() < 0.02      # every seed, every correlation
    assert np.all(np.diff(cloud) < 0)                 # errors that rise together cancel
    i6 = S.EP_RHOS.index(S.EP_RHO)
    assert without[i6] > 1.5 * per_seed[:, i6].max()  # dropping the term: far too large
    assert without[0] < 0.75 * per_seed[:, 0].min()   # ... or, anticorrelated, too small
    # the recorded spread is the seeded draw pushed through the ratio
    z = np.random.default_rng(S.EP_SEEDS[0]).standard_normal((S.EP_DRAWS, 2))
    f = np.divide(*S.pair(z, S.EP_SX, S.EP_SY, S.EP_RHO))
    assert np.isclose(f.std(ddof=1), per_seed[0, i6])
    assert np.allclose(s.panels[0].data["z"], z[:S.EP_SHOWN])
    assert np.allclose(s.panels[0].data["xy"],
                       np.stack(S.pair(z[:S.EP_SHOWN], S.EP_SX, S.EP_SY, S.EP_RHO), 1))


def test_propagation_formula_fails_for_a_poorly_measured_denominator():
    s = study(PROPAGATION)
    m = s.metadata
    formula, minus, plus = s.panels[3].data["ys"]
    assert plus[0] / minus[0] < 1.03 and abs(formula[0] / plus[0] - 1) < 0.02     # y to 2%
    assert np.all(np.diff(plus / minus) > 0)          # more lopsided with every step
    wide = m["wide"]
    assert wide["plus"] > 1.6 * wide["minus"] and wide["minus"] < 0.85 * wide["formula"]
    assert wide["plus"] > 1.3 * wide["formula"]
    assert abs(wide["median"] - m["f0"]) < 0.01 and wide["mean"] > m["f0"] + 0.15
    # no finite variance: the standard deviation is far above the formula and unstable
    sd = np.array(wide["sd_per_seed"])
    assert sd.min() > 5 * wide["formula"] and sd.max() > 1.5 * sd.min()
    hist, bell = s.panels[2].data["ys"]
    x = s.panels[2].data["x"]
    assert x[np.argmax(hist)] < m["f0"] - 0.15        # the cloud peaks below 2, tail above
    assert hist[x > 3.5].sum() > 5 * bell[x > 3.5].sum()


def test_propagation_numbers_on_screen():
    a = anim(PROPAGATION)
    m = a.study.metadata
    subs = subtitles(a)
    assert "%.3f" % m["one_variable"]["cloud"] in subs[0]
    assert "σ = %.3f" % m["two_variables"]["formula"] in subs[1]
    assert "%.3f" % m["two_variables"]["cloud"] in subs[1]
    assert "%.3f" % m["correlated"]["cloud"] in subs[2]
    wide = m["wide"]
    assert "−%.2f +%.2f" % (wide["minus"], wide["plus"]) in subs[3]
    assert "spread %.3f" % m["one_variable"]["cloud"] in on_screen(a, 0)
    shown = on_screen(a, 1)
    assert "formula: σf = %.3f" % m["two_variables"]["formula"] in shown
    assert "cloud: %.3f" % m["two_variables"]["cloud"] in shown
    assert "correlation ρ = %.2f" % m["rho"] in on_screen(a, 2)
    assert abs(a._rho(0.9) - m["rho"]) < 1e-9 and abs(a._rho(0.24) - 0.9) < 1e-9
    assert "cloud:  %.2f  −%.2f  +%.2f" % (wide["median"], wide["minus"], wide["plus"]) in \
        on_screen(a, 3)
    formula, minus, plus = a.study.panels[3].data["ys"]
    j = S.EP_RELS.index(0.20)
    shown = texts(a, at(a, 4))
    assert "± %.2f" % formula[j] in shown and "+%.2f" % plus[j] in shown
    assert "−%.2f" % minus[j] in shown


# ---------------------------------------------------------------------- 131: the mean
def test_mean_spread_falls_as_one_over_root_n():
    s = study(MEAN)
    m = s.metadata
    law = s.panels[1].data["ys"][2]
    assert np.allclose(law, S.EM_SIGMA / np.sqrt(S.EM_NS))
    per_seed = s.panels[1].data["per_seed"]           # [seed, distribution, N]
    assert np.abs(per_seed[:, 0] / law - 1).max() < 0.03      # every seed, every N
    assert np.abs(per_seed[:, 1] / law - 1).max() < 0.04      # the lopsided distribution too
    assert abs(m["slope"] + 0.5) < 0.005
    cost = s.panels[3].data["values"]
    assert np.abs(cost[:-1] / cost[1:] - 2).max() < 0.06      # four times the data: half
    hists, bells = s.panels[0].data["ys"], s.panels[0].data["exact"]
    assert np.abs(hists - bells).max() < 0.07 and bells.max() > 1.5
    assert np.all(np.diff(hists.max(1)) > 0)          # narrower, so taller, with every N


def test_mean_sample_variance_needs_n_minus_one():
    s = study(MEAN)
    ys = s.panels[2].data["ys"]
    measured, exact = ys[:3], ys[3:]
    ns = np.array(S.EM_SD_NS)
    assert np.allclose(exact[0], (ns - 1) / ns) and np.allclose(exact[1], 1.0)
    assert np.allclose(exact[2], [S.c4(int(n)) for n in ns])
    assert np.abs(measured - exact).max() < 0.02
    # dividing by N is too small by 1/N; N - 1 is right; the root is still a little low
    assert np.all(measured[0] < measured[1] - 0.9 / ns)
    assert np.all(measured[2, :6] < measured[1, :6] - 0.02)
    assert np.all(measured[2] > measured[0])
    per_seed = s.panels[2].data["per_seed"]
    assert np.abs(per_seed[:, 1] / S.EM_SIGMA ** 2 - 1).max() < 0.03


def test_mean_drawn_experiments_are_the_seeded_draws():
    s = study(MEAN)
    d = s.panels[0].data
    shown = np.round(np.random.default_rng(S.EM_SEEDS[0]).normal(
        S.EM_MU, S.EM_SIGMA, (S.EM_SHOWN, S.EM_SHOWN_N)), 1)
    assert np.array_equal(d["shown"], shown) and np.allclose(d["shown_means"], shown.mean(1))
    own, true = d["shown_ss"].T
    # the identity behind N - 1: squares about mu = squares about the mean + N (mean - mu)^2
    assert np.allclose(true, own + S.EM_SHOWN_N * (shown.mean(1) - S.EM_MU) ** 2)
    assert np.all(own <= true + 1e-12)
    assert np.isclose(s.metadata["shown_spread"], shown.mean(1).std(ddof=1))
    assert s.metadata["shown_spread"] < 0.75 * S.EM_SIGMA


def test_mean_numbers_on_screen():
    a = anim(MEAN)
    m = a.study.metadata
    d = a.study.panels[0].data
    subs = subtitles(a)
    assert "%.2f" % m["shown_spread"] in subs[0] and "%.2f" % m["spread"]["16"] in subs[1]
    assert M.signed(m["slope"]) in subs[2]
    assert "%d%% of σ² at N = 4" % round(100 * m["sd_ratio"]["4"][0]) in subs[3]
    # the first experiment, number by number
    assert (a._schedule(0.2)[0], a._schedule(0.99)) == (0, (S.EM_SHOWN - 1, 1.0))
    shown = on_screen(a, 0, 0.34)
    row = d["shown"][0]
    for v in row:
        assert "%.1f" % v in shown
    assert "mean = %.1f / 4 = %.3f" % (row.sum(), row.mean()) in shown
    assert "spread by %.2f" % m["shown_spread"] in on_screen(a, 0)
    shown = texts(a, at(a, 1))
    for n in S.EM_HIST_NS:
        assert "%.3f" % m["spread"][str(n)] in shown
        assert "%.3f" % (S.EM_SIGMA / math.sqrt(n)) in shown
    assert "measured slope: %s" % M.signed(m["slope"]) in on_screen(a, 2)
    assert "average from own mean: %.1f   ((N − 1) σ² = 12)" % m["shown_ss"][0] in \
        on_screen(a, 3)
    shown = texts(a, at(a, 4))
    for v in a.study.panels[3].data["values"]:
        assert "± %.3f" % v in shown


# ---------------------------------------------------------------------- 132: the bootstrap
def test_bootstrap_statistics_by_hand():
    x = np.array([1.0, 2.0, 3.0, 10.0])
    y = 2 * x + 1
    mean, median, ratio, r, top = S.statistics(x, y)
    assert (mean, median, top) == (4.0, 2.5, 10.0)
    assert np.isclose(ratio, 9.0 / 4.0) and np.isclose(r, 1.0)
    assert np.isclose(S.statistics(x, -y)[3], -1.0)
    both = S.statistics(np.stack([x, x[::-1]]), np.stack([y, y[::-1]]))       # row by row
    assert both.shape == (2, 5) and np.allclose(both[0], both[1])
    idx = np.array([[0, 0, 3, 3], [1, 1, 1, 1]])
    out = S.bootstrap(x, y, idx[:1])
    assert np.allclose(out[0, [0, 1, 4]], [5.5, 5.5, 10.0])


def test_bootstrap_data_and_resamples_are_the_seeded_draws():
    s = study(BOOTSTRAP)
    m = s.metadata
    d = s.panels[0].data
    rng = np.random.default_rng(S.BS_SEED)
    x, y = S.model(rng, S.BS_N)
    idx = rng.integers(0, S.BS_N, (S.BS_B_MAIN, S.BS_N))
    assert np.allclose(d["data_x"], x) and np.allclose(d["data_y"], y)
    assert np.array_equal(d["picks"], idx[:S.BS_SHOWN])
    assert d["picks"].min() >= 0 and d["picks"].max() < S.BS_N
    boot = S.bootstrap(x, y, idx)
    assert np.allclose(d["first"], boot[:60])
    assert np.allclose(m["bootstrap"], boot.std(0, ddof=1))
    assert np.allclose(m["here"], S.statistics(x, y))
    assert np.isclose(m["formula_mean"], x.std(ddof=1) / math.sqrt(S.BS_N))
    t = m["true_values"]                              # the model's own values, checked by draws
    big = S.statistics(*S.model(np.random.default_rng(7), (1, 400000)))[0]
    assert abs(big[0] - t["mean"]) < 0.02 and abs(big[1] - t["median"]) < 0.02
    assert abs(big[2] - t["ratio of means"]) < 0.01 and abs(big[3] - t["correlation"]) < 0.01


def test_bootstrap_agrees_with_the_formula_and_with_the_true_spread_on_average():
    s = study(BOOTSTRAP)
    m = s.metadata
    # where the answer is known: the mean. The bootstrap reproduces s / sqrt(N) (it
    # divides by N where the formula divides by N - 1)
    expect = m["formula_mean"] * math.sqrt((S.BS_N - 1) / S.BS_N)
    assert abs(m["bootstrap"][0] / expect - 1) < 0.05
    true_per_seed = s.panels[1].data["true_per_seed"]
    assert np.abs(true_per_seed[:, 0] / m["exact_mean"] - 1).max() < 0.02
    assert np.abs(true_per_seed[:, 4] / m["exact_max"] - 1).max() < 0.04
    assert np.abs(true_per_seed / true_per_seed.mean(0) - 1).max() < 0.03     # truth is stable
    # over 400 datasets: right on average for the four smooth statistics ...
    for k in range(4):
        assert 0.8 < m["ratio_median"][k] < 1.15 and 0.85 < m["ratio_mean"][k] < 1.15
        assert m["ratio_low"][k] < 0.8 and m["ratio_high"][k] > 1.15  # not for every dataset
    # ... and badly low for the maximum
    assert m["ratio_median"][4] < min(m["ratio_median"][:4]) - 0.3
    assert m["ratio_median"][4] < 0.6 and m["ratio_high"][4] < 1.0
    assert abs(m["top_share"] - m["top_share_exact"]) < 0.03
    assert np.isclose(m["top_share_exact"], 1 - 0.95 ** 20)
    assert m["top_values"][0][0] == max(s.panels[0].data["data_x"])
    assert np.isclose(m["top_values"][0][1], m["top_share"])
    boot, exact, fresh = s.panels[3].data["ys"]
    assert np.abs(fresh - exact).max() < 0.02         # fresh datasets follow the exact curve
    assert boot.max() > 5 * exact.max()               # the resamples pile up on one value


def test_bootstrap_settles_with_a_few_hundred_resamples():
    s = study(BOOTSTRAP)
    m = s.metadata
    scatter, reference = s.panels[2].data["ys"]
    assert np.all(np.diff(scatter) < 0)
    assert np.all((scatter > 0.9 * reference) & (scatter < 1.5 * reference))
    assert scatter[S.BS_BS.index(2000)] < scatter[0] / 8
    # having one dataset rather than another matters far more than resampling noise
    between = 0.5 * (m["ratio_high"][1] - m["ratio_low"][1])
    assert between > 5 * scatter[S.BS_BS.index(200)]
    runs = s.panels[2].data["runs"]
    assert np.ptp(runs[:, -1]) < 0.15 * np.ptp(runs[:, 0])


def test_bootstrap_numbers_on_screen():
    a = anim(BOOTSTRAP)
    m = a.study.metadata
    subs = subtitles(a)
    assert "s/√N = %.2f" % m["formula_mean"] in subs[0]
    assert "%.3f" % m["bootstrap"][0] in subs[1] and "%.3f" % m["formula_mean"] in subs[1]
    assert "%d%% of the true spread" % round(100 * m["ratio_median"][4]) in subs[4]
    shown = texts(a, at(a, 0))
    for v in m["here"]:
        assert "%.2f" % v in shown
    # the picking: the row drawn is the dataset at the positions picked
    r, t = a._picking(0.39)
    assert (r, t) == (0, 20.0) and a._picking(0.62)[0] == 3
    shown = texts(a, at(a, 1, 0.395))
    picks = a.study.panels[0].data["picks"][0]
    counts = np.bincount(picks, minlength=20)
    assert "%d events left out, %d picked more than once;  its mean = %.2f" % (
        (counts == 0).sum(), (counts > 1).sum(), a.x[picks].mean()) in shown
    assert shown.count("×0") == (counts == 0).sum()
    assert "%.3f, the bootstrap error" % m["bootstrap"][0] in on_screen(a, 1)
    shown = texts(a, at(a, 2))
    for k in range(4):
        assert "%.3f" % m["bootstrap"][k] in shown and "%.3f" % m["true"][k] in shown
    assert "bootstrap error here: %.2f" % m["bootstrap"][4] in on_screen(a, 4)
    assert "true spread: %.2f (exact %.2f)" % (m["true"][4], m["exact_max"]) in on_screen(a, 4)

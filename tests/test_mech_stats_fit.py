"""Animations 124-127 (voxanim.mech_stats_fit) and their experiments
(voxanim.study_stats_fit): the layout is clean, the loop closes, every phase says what kind
of evidence it shows, the numbers on screen are the study's, each study's claim holds as a
relative claim with a margin, the special functions and minimisers give known answers, and
the recorded arrays are what they say they are. None needs a dataset."""

from __future__ import annotations

import math

import numpy as np
import pytest

from voxanim import cache, layout, presets, qa
from voxanim import mech_stats_fit as M
from voxanim import study_stats_fit as S
from voxanim.next_catalog import NO_DATA, build_study

FIT, DOF, MIN, BIN = ("chi-square-fit", "chi-square-per-dof", "minimisers-on-chi-square",
                      "binned-vs-unbinned-fits")
NAMES = (FIT, DOF, MIN, BIN)


def study(name):
    return build_study("study_stats_fit", name)


def anim(name):
    return presets.get(name).build()


def subtitles(a):
    return [a.subtitle_for(k) for k in range(len(a.PHASES))]


def at(a, phase, u):
    lo, hi = a.spans[phase]
    return lo + (hi - lo) * u


def texts(a, phase, u=0.9):
    return [x.s for x in a.scene(at(a, phase, u)).texts]


# ---------------------------------------------------------------------- all four
def test_registered_and_need_no_data():
    assert set(M.ANIMATIONS) == set(S.STUDIES) == set(NAMES)
    assert set(NAMES) <= NO_DATA


@pytest.mark.parametrize("name", NAMES)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", NAMES)
def test_drawn_by_this_module_and_loop_closes(name):
    a = anim(name)
    assert type(a) is M.ANIMATIONS[name]
    assert 45.0 <= a.t_done <= 60.0

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:                            # every phase draws things, not only text
        assert len(a.scene((lo + hi) / 2).quads) >= 3


@pytest.mark.parametrize("name", NAMES)
def test_subtitles_fit_two_lines_and_every_phase_is_tagged(name):
    a = anim(name)
    for sub in subtitles(a):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    for k in range(len(a.PHASES)):                    # exact, or pseudo-experiments
        for u in (0.3, 0.9):
            tags = [t for t in texts(a, k, u) if t.startswith((S.EXACT, S.TOYS))]
            assert len(tags) == 1, (name, k, u)
            if tags[0].startswith(S.TOYS):            # ... and then from what, or how many
                assert any(w in tags[0] for w in ("from the true", "per point", "samples"))


@pytest.mark.parametrize("name", NAMES)
def test_study_says_what_was_run(name):
    s = study(name)
    m = s.metadata
    assert len(s.panels) == 4 and m["designs_tried"] and m["limitations"] and m["seeds"]
    assert "seed" in m["evidence"]["data"]
    kinds = " ".join(m["evidence"].values())
    assert S.EXACT in kinds


def test_studies_build_from_an_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_CACHE", str(tmp_path))
    for name in NAMES:                                # offline (conftest): a download would raise
        M.ANIMATIONS[name](S.STUDIES[name]().validate()).scene(3.0)
    made = sorted(p.name.split("-")[2] for p in tmp_path.iterdir())
    assert made == ["binned", "dof", "fit", "minimisers"]


def test_pseudo_experiments_are_seeded():
    for build in (S._fit_build, S._dof_build):
        a, b = build(), build()
        assert all(np.array_equal(a[k], b[k]) for k in a)
    rng = np.random.default_rng(S.BIN_SEED)
    assert np.array_equal(np.sort(S.decay_sample(rng, S.DEMO_N)), S.bin_data()["demo_times"])


# ---------------------------------------------------------------------- special functions
def test_chi_square_distribution_against_known_values():
    # the 5% points of the chi-square table
    for k, x in ((1, 3.841459), (2, 5.991465), (4, 9.487729), (9, 16.918978), (10, 18.307038)):
        assert abs(float(S.chi2_sf(x, k)) - 0.05) < 1e-6
    xs = np.array([0.3, 2.0, 7.5, 40.0])
    assert np.allclose(S.chi2_sf(xs, 2), np.exp(-xs / 2))             # two degrees: exact
    assert np.allclose(S.chi2_cdf(xs, 1), [math.erf(math.sqrt(x / 2)) for x in xs])
    assert np.allclose(S.chi2_cdf(xs, 7) + S.chi2_sf(xs, 7), 1.0)
    assert 1e-7 < float(S.chi2_sf(46.0866, 10)) < 2e-6                # a tail, not 1 - (almost 1)
    assert float(S.chi2_sf(0.0, 5)) == 1.0 and float(S.chi2_cdf(0.0, 5)) == 0.0
    x = np.linspace(0, 80, 80001)
    for k in (1, 4, 9):
        pdf = S.chi2_pdf(x[1:], k)
        assert abs(np.sum(pdf) * (x[1] - x[0]) - 1) < (0.02 if k == 1 else 1e-3)
        if k > 1:                                                     # mean k, variance 2k
            assert abs(np.sum(x[1:] * pdf) * (x[1] - x[0]) - k) < 1e-2
            assert abs(np.sum((x[1:] - k) ** 2 * pdf) * (x[1] - x[0]) - 2 * k) < 2e-2
        mid = (x[1:] + x[:-1])[2000:20000:3000] / 2
        slope = np.diff(S.chi2_cdf(x[:20001], k))[2000:20000:3000] / (x[1] - x[0])
        assert np.allclose(slope, S.chi2_pdf(mid, k), rtol=1e-4)


# ---------------------------------------------------------------------- 124
def test_line_fit_is_least_squares():
    y = S.fit_dataset()
    a, b, cov, _ = S.line_fit(S.FIT_X, y, S.FIT_SIGMA)
    slope, intercept = np.polyfit(S.FIT_X, y, 1, w=1 / S.FIT_SIGMA)
    assert np.isclose(a, intercept) and np.isclose(b, slope)
    best = S.line_chi2(a, b, S.FIT_X, y, S.FIT_SIGMA)
    for da, db in ((1e-3, 0), (-1e-3, 0), (0, 1e-3), (0, -1e-3), (1e-3, -1e-3)):
        assert S.line_chi2(a + da, b + db, S.FIT_X, y, S.FIT_SIGMA) > best
    # one standard deviation along each parameter, the other re-fitted, costs exactly 1
    sa, sb = np.sqrt(np.diag(cov))
    rho = cov[0, 1] / (sa * sb)
    assert np.isclose(S.line_chi2(a + sa, b + rho * sb, S.FIT_X, y, S.FIT_SIGMA) - best, 1.0)
    assert np.isclose(S.line_chi2(a + rho * sa, b + sb, S.FIT_X, y, S.FIT_SIGMA) - best, 1.0)
    r, pull, sq = S.line_terms(a, b, S.FIT_X, y, S.FIT_SIGMA)
    assert np.allclose(pull, r / S.FIT_SIGMA) and np.isclose(sq.sum(), best)


def test_fit_data_is_what_it_says():
    d, s = S.fit_data(), study(FIT)
    y = d["y"]
    rng = np.random.default_rng(S.FIT_SEED)
    assert np.allclose(y, S.FIT_TRUE[0] + S.FIT_TRUE[1] * S.FIT_X
                       + S.FIT_SIGMA * rng.standard_normal(6))
    surface = s.panels[1].data["values"]
    for r, c in ((0, 0), (17, 93), (120, 120)):
        assert np.isclose(surface[r, c], S.line_chi2(d["A"][c], d["B"][r], S.FIT_X, y,
                                                     S.FIT_SIGMA))
    assert surface.min() >= d["chi_min"] and surface.min() < d["chi_min"] + 0.01
    assert len(d["toy_a"]) == S.FIT_TOYS == s.metadata["toys"]
    path = d["path"]
    assert tuple(path[0]) == S.FIT_START and np.allclose(path[-1], d["best"])
    assert path[1][0] == path[0][0] and path[2][1] == path[1][1]    # one parameter at a time
    for k, step in ((1, (0, 1e-3)), (2, (1e-3, 0))):                # each the best of its line
        here = S.line_chi2(path[k][0], path[k][1], S.FIT_X, y, S.FIT_SIGMA)
        for sign in (1, -1):
            assert S.line_chi2(path[k][0] + sign * step[0], path[k][1] + sign * step[1],
                               S.FIT_X, y, S.FIT_SIGMA) > here
    assert (np.diff(d["path_chi"]) < 0).all()


def test_chi_square_fit_claims():
    m = study(FIT).metadata
    # the contour chi-square_min + 1, found on a grid, against the covariance matrix
    for found, exact in zip(m["contour_half_widths"], m["errors"]):
        assert abs(found / exact - 1) < 0.01
    # ... and against the spread of 10,000 repeated experiments
    for found, exact in zip(m["toy_sd"], m["errors"]):
        assert abs(found / exact - 1) < 0.03
    assert abs(m["toy_correlation"] - m["correlation"]) < 0.01 and m["correlation"] < -0.8
    assert np.allclose(m["toy_mean"], m["true"], atol=0.01)         # unbiased
    for share in m["within_one_sigma"]:
        assert abs(share - m["within_exact"]) < 0.015
    assert abs(m["inside_contour"] - m["inside_contour_exact"]) < 0.015
    assert m["inside_contour"] < m["within_one_sigma"][0] - 0.2     # jointly: far fewer
    assert abs(m["toy_chi_mean"] - m["dof"]) < 0.1
    assert np.allclose(m["grid_minimum"], m["best"], atol=0.03)
    assert m["start_chi"] > 10 * m["chi_min"]


def test_chi_square_fit_numbers_on_screen():
    s = study(FIT)
    m, a = s.metadata, anim(FIT)
    sub = subtitles(a)
    assert "here %.1f for the trial line" % m["start_chi"] in sub[0]
    assert "falls from %.1f to %.2f" % (m["start_chi"], m["chi_min"]) in sub[1]
    assert "a = %.3f, b = %.3f" % tuple(m["best"]) in sub[2]
    assert "± %.2f" % m["contour_half_widths"][0] in sub[3]
    assert "%.1f%% of the time" % (100 * m["within_one_sigma"][0]) in sub[4]
    assert "only %.1f%%" % (100 * m["inside_contour"]) in sub[4]
    shown = texts(a, 0, 0.95)
    _, _, sq = S.line_terms(*S.FIT_START, S.FIT_X, S.fit_dataset(), S.FIT_SIGMA)
    assert {"%.2f" % v for v in sq} <= set(shown) and "%.2f" % sq.sum() in shown
    assert abs(sq.sum() - m["start_chi"]) < 1e-9
    assert "%s = %.2f" % (M.CHI, m["chi_min"]) in texts(a, 2)
    assert "± %.3f" % m["contour_half_widths"][0] in texts(a, 3)
    assert any("%.3f measured,  %.3f from the formula" % (m["toy_sd"][0], m["errors"][0]) in t
               for t in texts(a, 4))


# ---------------------------------------------------------------------- 125
def test_polynomial_fits_are_least_squares():
    y = S.dof_dataset()
    for deg in (1, 2, 5):
        chi, fitted = S.poly_chi2(y, deg, S.DOF_SIGMA)
        ref = np.polyval(np.polyfit(S.DOF_X, y, deg), S.DOF_X)
        assert np.allclose(fitted, ref, atol=1e-8)
        assert np.isclose(chi, (((y - ref) / S.DOF_SIGMA) ** 2).sum())
    H = S.poly_hat(S.DOF_X, 2)
    assert np.allclose(H @ H, H) and np.isclose(np.trace(H), 3)     # a projection of rank 3


def test_dof_data_is_what_it_says():
    d, s = S.dof_data(), study(DOF)
    m = s.metadata
    rng = np.random.default_rng(S.DOF_SEED)
    assert np.allclose(d["y"], S.dof_truth(S.DOF_X) + S.DOF_SIGMA * rng.standard_normal(12))
    assert d["toy_chi"].shape == (9, S.DOF_TOYS)
    assert np.isclose(d["toy_chi"][2][0], S.poly_chi2(d["toys_y"][0], 2, S.DOF_SIGMA)[0])
    for deg in S.DOF_ALL:
        assert m["dof"][str(deg)] == 12 - (deg + 1)
        assert np.isclose(m["probability"][str(deg)],
                          float(S.chi2_sf(m["chi2"][str(deg)], 12 - deg - 1)))
    assert (np.diff([m["chi2"][str(deg)] for deg in S.DOF_ALL]) <= 1e-9).all()
    for k, deg in enumerate(S.DOF_DEGREES):           # the curves drawn are those fits
        ref = np.polyval(np.polyfit(S.DOF_X, d["y"], deg), d["fine"])
        assert np.allclose(s.panels[0].data["ys"][k], ref, atol=1e-6)


def test_chi_square_per_dof_claims():
    s = study(DOF)
    m = s.metadata
    # the right model and the one with too many parameters: chi-square averages its dof
    for deg in range(2, 9):
        assert abs(m["toy_mean"][str(deg)] - m["dof"][str(deg)]) < 0.1
    assert abs(m["toy_variance_right"] - 2 * 9) < 0.6
    hist, exact = (np.asarray(s.panels[1].data[k]) for k in ("ys", "exact"))
    assert np.abs(hist[1] - exact[1]).max() < 0.006 and np.abs(hist[2] - exact[2]).max() < 0.01
    # the too-simple one does not follow its curve: three times its dof on average
    assert m["toy_mean"]["1"] > m["dof"]["1"] + 15 and np.abs(hist[0] - exact[0]).max() > 0.05
    assert m["probability"]["1"] < 1e-3 < 0.05 < min(m["probability"]["2"], m["probability"]["7"])
    # the needed parameter buys a lot, each unneeded one about 1
    assert abs(m["gain_per_extra_parameter"] - 1) < 0.05
    assert m["gain_line_to_parabola"] > 10 * m["gain_per_extra_parameter"]
    # ... and costs distance from the truth
    for deg in ("2", "7"):
        assert abs(m["distance_from_truth"][deg] - m["distance_exact"][deg]) < 0.01
    assert m["distance_from_truth"]["7"] > m["distance_from_truth"]["2"] + 0.25
    # wrong errors
    small, right, large = m["reduced_mean"]
    assert abs(small - 4) < 0.1 and abs(right - 1) < 0.02 and abs(large - 0.25) < 0.01
    assert m["p_below_5pc"][0] > 0.8 and abs(m["p_below_5pc"][1] - 0.05) < 0.015
    assert m["p_below_5pc"][2] < 0.005 and m["p_above_95pc"][0] < 0.005
    assert m["p_above_95pc"][2] > 0.75 and abs(m["p_above_95pc"][1] - 0.05) < 0.015
    red, red_exact = (np.asarray(s.panels[3].data[k]) for k in ("ys", "exact"))
    assert np.abs(red - red_exact)[1].max() < 0.06


def test_chi_square_per_dof_numbers_on_screen():
    s = study(DOF)
    m, a = s.metadata, anim(DOF)
    sub = subtitles(a)
    chi = [m["chi2"][str(d)] for d in S.DOF_DEGREES]
    assert "%.1f for the line, %.1f for the parabola, %.1f for degree 7" % tuple(chi) in sub[0]
    assert "their mean is %.2f" % m["toy_mean"]["2"] in sub[1]
    assert "%.2f for the parabola and %.2f for degree 7" % (
        m["probability"]["2"], m["probability"]["7"]) in sub[2]
    assert M.small(m["probability"]["1"]) in sub[2]
    assert "by only %.2f" % m["gain_per_extra_parameter"] in sub[3]
    assert "this one gives %.2f" % m["reduced_this"][1] in sub[4]
    shown = texts(a, 0, 0.95)
    assert {"%.1f" % c for c in chi} <= set(shown) and {"10", "9", "4"} <= set(shown)
    assert "p = %.2f" % m["probability"]["2"] in texts(a, 2)
    assert "20,000 pseudo-experiments" in texts(a, 1, 0.8)
    assert any("p > 95%% in %.0f%% of repeats" % (100 * m["p_above_95pc"][2]) == t
               for t in texts(a, 4))
    assert M.small(1.38e-6) == "1.4 × 10⁻⁶" and M.small(0.94) == "0.94"


# ---------------------------------------------------------------------- 126
class Bowl:
    """(p0 - 2)^2 + 10 (p1 - 1)^2, counting its evaluations."""

    def __init__(self):
        self.calls = 0

    def __call__(self, p):
        self.calls += 1
        return float((p[0] - 2) ** 2 + 10 * (p[1] - 1) ** 2)


@pytest.mark.parametrize("method", S.METHODS)
def test_minimisers_find_the_bottom_of_a_bowl(method):
    f = Bowl()
    path = S.RUN[method](f, (4.0, 2.0))
    assert np.allclose(path[-1], [2, 1], atol=0.02) and np.allclose(path[0], [4, 2])
    assert 5 < f.calls < 400
    if method == "Newton":                            # a paraboloid: one jump, then it checks
        assert len(path) == 3 and np.allclose(path[1], [2, 1], atol=1e-5) and f.calls == 14


def test_the_surface_has_two_bottoms():
    y, err = S.peak_dataset()
    minima = S.peak_minima(y, err)
    assert len(minima) == 2
    for chi, mu, sigma in minima:
        f = S.Counted(y, err)
        assert np.abs(S._gradient(f, np.array([mu, sigma]))).max() < 0.05
        assert np.linalg.eigvalsh(S.hessian((mu, sigma), y, err)).min() > 0
        assert np.isclose(chi, S.peak_chi2(mu, sigma, y, err))
    best, other = minima
    assert abs(best[1] - S.MIN_PEAK[0]) < 0.2 and abs(other[1] - S.MIN_SECOND[1]) < 0.2
    assert other[0] > best[0] + 100
    assert S.Counted(y, err)((0.0, 1.0)) == math.inf   # outside the box
    assert S.classify(best[1:], minima) == 0 and S.classify(other[1:], minima) == 1
    assert S.classify(np.array([6.0, 1.0]), minima) == 2
    assert S.classify(np.array([np.nan, 1.0]), minima) == 2


def test_minimiser_data_is_what_it_says():
    d, s = S.min_data(), study(MIN)
    y, err = S.peak_dataset()
    assert np.array_equal(d["y"], y) and np.array_equal(
        y, np.random.default_rng(S.MIN_SEED).poisson(S.peak_truth(S.MIN_X)))
    assert np.allclose(d["err"], np.sqrt(y)) and y.min() >= 1
    assert np.isclose(d["surface"][40, 100], S.peak_chi2(d["mu"][100], d["sigma"][40], y, err))
    ends = s.panels[2].data["ends"]
    assert ends.shape == (4, S.MIN_GRID[1], S.MIN_GRID[0])
    for i, r, k in ((0, 3, 7), (1, 6, 14), (2, 20, 30), (3, 11, 40), (0, 22, 44)):
        path, calls = S.run(S.METHODS[i], (d["start_mu"][k], d["start_sigma"][r]))
        assert S.classify(path[-1], d["minima"]) == ends[i, r, k]
        assert calls == d["evals"][i, r, k]
    for n, start in enumerate(S.MIN_STARTS):
        for i, method in enumerate(S.METHODS):
            path, calls = S.run(method, start)
            assert np.array_equal(path, d["path_%d_%d" % (n, i)])
            assert calls == d["calls_%d_%d" % (n, i)] == s.metadata["demo_evaluations"][n][i]
            seen = d["at_%d_%d" % (n, i)]
            assert len(seen) == len(path) and (np.diff(seen) >= 0).all() and seen[-1] <= calls
    tri = d["triangles_0"]
    assert np.allclose(tri[0][np.argsort(tri[0].sum(1))][0], S.MIN_STARTS[0])
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    assert area[-1] < 1e-4 * area[0]


def test_minimiser_claims():
    s = study(MIN)
    m = s.metadata
    share = {k: np.array(v) for k, v in m["share"].items()}
    for v in share.values():
        assert np.isclose(v.sum(), 1)
    # from the first start every method reaches the best fit; from the second none does
    assert m["demo_end"][0] == [0, 0, 0, 0]
    assert m["demo_end"][1] == [1, 2, 1, 1]
    c1, c2 = np.array(m["start_curvature"])
    assert c1.min() > 0 and c2.max() < 0              # a bowl at start 1, a dome at start 2
    # Newton unguarded: quick where it works, and it works from few starts
    med = m["median_evaluations"]
    assert med["Newton"] < 0.7 * min(med[k] for k in S.METHODS if k != "Newton")
    assert share["Newton"][0] < 0.25 * min(share[k][0] for k in S.METHODS if k != "Newton")
    assert share["Newton"][2] > 0.8
    # the others: each reaches the wrong bottom from a good share of starts
    for k in ("gradient descent", "simplex", "variable metric"):
        assert share[k][0] > 0.4 and share[k][1] > 0.2
    # what gradient descent pays for using the slope only
    assert med["gradient descent"] > med["variable metric"] > med["Newton"]
    assert m["demo_evaluations"][0][0] > m["demo_evaluations"][0][1]


def test_minimiser_numbers_on_screen():
    s = study(MIN)
    m, a = s.metadata, anim(MIN)
    sub = subtitles(a)
    assert "%.0f with the peak in place, %.0f with it sitting" % (
        m["minima"][0][0], m["minima"][1][0]) in sub[0]
    ev, st = m["demo_evaluations"], m["demo_steps"]
    assert "%d steps and %d evaluations" % (st[0][0], ev[0][0]) in sub[1]
    assert "out: %d steps" % st[0][1] in sub[2]
    assert "%d evaluations from the first start" % ev[0][2] in sub[3]
    for k in S.METHODS:
        assert "%.0f%%" % (100 * m["share"][k][0]) in sub[4]
    assert "1,035 starts" in sub[4] and m["starts"] == 1035
    shown = texts(a, 1, 0.5)                          # the first walk is over
    assert "evaluations of %s so far:  %d" % (M.CHI, ev[0][0]) in shown
    assert "start 1:  step %d of %d" % (st[0][0], st[0][0]) in shown
    g = m["start_gradient"]
    assert any("+%.0f per unit of μ" % g[0] in t for t in shown)
    assert any("off the map" in t for t in texts(a, 2, 0.97))
    last = texts(a, 4)
    for k in S.METHODS:
        assert "best fit from %.0f%%" % (100 * m["share"][k][0]) in last
        assert "median %d evaluations" % m["median_evaluations"][k] in last
    assert any("MINUIT" in t for t in last)


# ---------------------------------------------------------------------- 127
def test_decay_model_and_estimators_on_known_input():
    t = np.linspace(0, S.WINDOW, 200001)
    pdf = S.decay_pdf(t, S.TAU)
    dt = t[1] - t[0]
    assert abs((pdf[1:] + pdf[:-1]).sum() * dt / 2 - 1) < 1e-8
    edges = np.linspace(0, S.WINDOW, 11)
    p = S.bin_probs(np.array([1.0, S.TAU, 5.0]), edges)
    assert np.allclose(p.sum(1), 1) and (np.diff(p, axis=1) < 0).all()
    assert np.isclose(p[1, 0], (1 - math.exp(-1 / S.TAU)) / (1 - math.exp(-S.WINDOW / S.TAU)))
    times = S.decay_sample(np.random.default_rng(0), (1, 400000))
    assert times.min() >= 0 and times.max() <= S.WINDOW
    mean = ((t * pdf)[1:] + (t * pdf)[:-1]).sum() * dt / 2
    assert abs(times.mean() - mean) < 0.01
    # a very large sample: all three fits return the true lifetime
    est, edge, counts = S.fit_all(times, 10)
    assert counts.sum() == 400000 and not edge.any()
    assert np.abs(est[:, 0] - S.TAU).max() < 0.02
    # counts exactly as expected: both binned fits return it to the precision of the scan
    exact = 1e6 * S.bin_probs(np.array([S.TAU]), edges)
    for score in (S.chi2_score(exact, edges), S.binned_score(exact, edges)):
        found, at_edge = S._refine(score)
        assert abs(found[0] - S.TAU) < 1e-3 and not at_edge[0]
    # the Cramer-Rao bound against the Fisher information integrated numerically
    h = 1e-4
    d = (np.log(S.decay_pdf(t, S.TAU + h)) - np.log(S.decay_pdf(t, S.TAU - h))) / (2 * h)
    info = ((d ** 2 * pdf)[1:] + (d ** 2 * pdf)[:-1]).sum() * dt / 2
    assert np.isclose(S.exact_spread(100), 1 / math.sqrt(100 * info), rtol=1e-5)


def test_binned_chi_square_is_what_it_says():
    d = S.bin_data()
    counts, edges = d["demo_counts"], np.linspace(0, S.WINDOW, S.DEMO_BINS + 1)
    assert counts.sum() == S.DEMO_N and (counts == 0).any()
    assert np.array_equal(counts, np.histogram(d["demo_times"], edges)[0])
    est, _, _ = S.fit_all(d["demo_times"][None], S.DEMO_BINS)
    assert np.allclose(est[:, 0], d["demo_fit"])
    filled = counts > 0

    def chi2(tau, norm):                              # sqrt(n) errors, empty bins left out
        p = S.bin_probs(np.array([tau]), edges)[0]
        return (((counts - norm * p) ** 2)[filled] / counts[filled]).sum()
    tau = float(d["demo_fit"][0])
    norm = S.chi2_norm(counts, edges, tau)
    best = chi2(tau, norm)
    for dt, dn in ((0.01, 0), (-0.01, 0), (0, 0.5), (0, -0.5)):
        other = chi2(tau + dt, S.chi2_norm(counts, edges, tau + dt) if dt else norm + dn)
        assert other > best
    score = S.chi2_score(counts[None], edges)[0]      # the scan's score is minus chi-square
    k = int(np.argmin(np.abs(S.TAU_GRID - 2.0)))
    assert np.isclose(counts[filled].sum() - score[k],
                      chi2(S.TAU_GRID[k], S.chi2_norm(counts, edges, S.TAU_GRID[k])))
    # the unbinned likelihood: its maximum is where the sample mean equals the model's
    t = np.linspace(0, S.WINDOW, 100001)
    pdf = S.decay_pdf(t, float(d["demo_fit"][2]))
    assert abs((t * pdf).sum() * (t[1] - t[0]) - d["demo_times"].mean()) < 2e-3


def test_binned_vs_unbinned_claims():
    s = study(BIN)
    m = s.metadata
    sizes = np.array(m["sizes"])
    bias, spread = np.array(m["bias_by_size"]), np.array(m["spread_by_size"])
    se = spread / math.sqrt(m["toys"])
    assert np.array(m["failed_by_size"]).max() < 0.01 and np.array(m["failed_by_bins"]).max() == 0
    mid = (sizes >= 120) & (sizes <= 1000)
    # chi-square with sqrt(n) errors: too short a lifetime, far beyond its statistical error
    assert (bias[mid, 0] < -5 * se[mid, 0]).all() and (bias[mid, 0] < -0.03).all()
    assert (np.abs(bias[mid, 0]) > 4 * np.abs(bias[mid, 1:]).max(1)).all()
    assert (np.abs(bias[mid, 0]) > 0.3 * spread[mid, 0]).all()
    # ... and at 30 events the other way, which is the empty bins being left out
    assert bias[0, 0] > 0.1 and bias[0, 0] > 3 * np.abs(bias[0, 1:]).max()
    kept = np.array(m["chi2_bias_keeping_empty_bins"]["by_size"])
    assert kept[0] < -0.1 and (kept < 0).all()
    # the likelihoods: within 1.5% from 60 events up, the bias falling with the sample
    assert (np.abs(bias[1:, 1:]) < 0.015 * m["tau"]).all()
    assert np.abs(bias[-3:, 1:]).max() < 0.25 * np.abs(bias[:2, 1:]).min()
    # spread: the unbinned likelihood is at the bound, the chi-square above it
    exact = np.asarray(s.panels[2].data["exact_spread"])
    assert (np.abs(spread[2:, 2] / exact[2:] - 1) < 0.04).all()
    assert (spread[:, 0] > spread[:, 2]).all() and spread[0, 0] > 1.5 * spread[0, 2]
    assert (spread[:, 1] >= spread[:, 2] - 1e-9).all()
    # coarser bins: the chi-square's bias fades, the binned fits lose precision
    bb, sb = np.array(m["bias_by_bins"]), np.array(m["spread_by_bins"])
    assert m["binnings"] == [50, 20, 10, 5, 3, 2]
    assert (np.diff(np.abs(bb[1:, 0])) < 0).all() and abs(bb[-1, 0]) < 0.01
    assert np.abs(bb[:, 1:]).max() < 0.01
    assert sb[-1, 1] > 1.25 * sb[-1, 2] and abs(sb[0, 1] / sb[0, 2] - 1) < 0.02
    assert np.abs(sb[:, 2] / sb[0, 2] - 1).max() < 0.03   # the unbinned fit has no bins
    assert sb[0, 0] > 1.3 * sb[0, 2]                  # 50 bins: mostly empty or nearly so


def test_binned_vs_unbinned_numbers_on_screen():
    s = study(BIN)
    m, a = s.metadata, anim(BIN)
    demo = m["demo"]
    sub = subtitles(a)
    assert "%.2f from χ², %.2f and %.2f from the likelihoods" % tuple(demo["fitted"]) in sub[0]
    ratio = abs(demo["bias"][0]) / demo["spread"][0]
    assert "a third of its own spread" in sub[2] and 0.26 < ratio < 0.40
    assert "averages −%.1f%%" % (100 * abs(demo["bias"][0]) / m["tau"]) in sub[2]
    b30, b250 = (100 * m["bias_by_size"][m["sizes"].index(n)][0] / m["tau"] for n in (30, 250))
    assert "+%.0f%% at 30 events" % b30 in sub[3] and "−%.1f%% at 250" % abs(b250) in sub[3]
    sb = np.array(m["spread_by_bins"])
    assert "spread %.2f times wider" % (sb[-1, 1] / sb[-1, 2]) in sub[4]
    assert abs(sb[-1, 0] / sb[-1, 1] - 1) < 0.01      # "both binned fits"
    shown = texts(a, 0, 0.97)
    assert {"τ = %.2f" % v for v in demo["fitted"]} <= set(shown)
    errors = texts(a, 1)
    assert {"±%.1f" % math.sqrt(n) for n in demo["counts"] if n} <= set(errors)
    assert "n = 2:   (2 − 4)² ÷ 2  =  2.00" in errors and "n = 0: no error, left out" in errors
    expected = s.panels[0].data["expected"][demo["counts"].index(0)]
    assert any(t.startswith("P(0 | %.1f) = %.2f" % (expected, math.exp(-round(expected, 1))))
               for t in errors)
    assert "P(2 | 4) = %.3f,   P(6 | 4) = %.3f" % (
        math.exp(-4) * 8, math.exp(-4) * 4 ** 6 / 720) in errors
    rows = texts(a, 2)
    for k in range(3):
        assert "spread %.3f" % demo["spread"][k] in rows
    assert "smallest spread possible: %.3f" % demo["exact_spread"] in rows

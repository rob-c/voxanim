"""Animations 145-149 (voxanim.mech_stats_discovery) and their experiments
(voxanim.study_stats_discovery): the layout is clean, the loop closes, every phase says
what kind of evidence it shows, the numbers on screen are the study's, the special
functions give the tabulated values, each study's claim holds as a relative claim, and
the recorded arrays are what they say they are.

None of the five needs a dataset: every spectrum and count is seeded pseudo-data, and
all five are built here from an empty cache.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from voxanim import cache, layout, presets, qa
from voxanim import mech_stats_discovery as M
from voxanim import study_stats_discovery as S
from voxanim.next_catalog import NO_DATA, build_study

NAMES = ("bump-hunt", "poisson-upper-limits", "cls-method", "brazil-band", "why-five-sigma")


def study(name):
    return build_study("study_stats_discovery", name)


def anim_of(name):
    return M.ANIMATIONS[name](study(name))


def subtitles(anim):
    return [anim.subtitle_for(k) for k in range(len(anim.PHASES))]


def texts(anim, t):
    return [x.s for x in anim.scene(t).texts if x.opacity > 0.5]


def late(anim, k, at=0.97):
    a, b = anim.spans[k]
    return min(a + (b - a) * at, b - anim.FADE - 0.05)


# ---------------------------------------------------------------------- all five
def test_registered_and_need_no_data():
    assert set(M.ANIMATIONS) == set(S.STUDIES) == set(NAMES)
    assert set(NAMES) <= NO_DATA


@pytest.mark.parametrize("name", NAMES)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", NAMES)
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


@pytest.mark.parametrize("name", NAMES)
def test_subtitles_fit_two_lines_and_every_phase_is_tagged(name):
    anim = anim_of(name)
    assert len(anim.TAGS) == len(anim.PHASES)
    for k, sub in enumerate(subtitles(anim)):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
        tag = anim.TAGS[k]                            # the kind of evidence, on screen
        assert tag.startswith((M.EXACT, M.PSEUDO))
        assert tag in texts(anim, late(anim, k, 0.5))
    assert any(t.startswith(M.PSEUDO) for t in anim.TAGS)


@pytest.mark.parametrize("name", NAMES)
def test_study_says_what_was_run(name):
    s = study(name).validate()
    m = s.metadata
    assert len(s.panels) == 4 and m["designs_tried"] and m["limitations"]
    assert m["evidence"]["exact"] and m["evidence"]["pseudo_experiments"]
    assert m["seeds"] and all(isinstance(v, int) for v in m["seeds"])


def test_all_five_build_from_an_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_CACHE", str(tmp_path))
    for name in NAMES:                                # offline (conftest): a download would raise
        M.ANIMATIONS[name](S.STUDIES[name]().validate()).scene(3.0)
    made = sorted(p.name.split("-")[2] for p in tmp_path.iterdir())
    assert made == ["band", "bump", "cls", "five", "limits"]


# ---------------------------------------------------------------------- special functions
def test_normal_tail_and_its_inverse():
    assert S.norm_sf(0.0) == pytest.approx(0.5)
    assert S.norm_sf(1.0) == pytest.approx(0.158655254, rel=1e-8)
    assert S.norm_sf(3.0) == pytest.approx(1.349898e-3, rel=1e-6)
    assert S.norm_sf(5.0) == pytest.approx(2.866516e-7, rel=1e-6)
    assert S.norm_sf(-1.0) == pytest.approx(1 - 0.158655254, rel=1e-8)
    z = np.array([-2.0, 0.0, 1.0, 3.0, 5.0, 7.0])
    assert np.allclose(S.norm_isf(S.norm_sf(z)), z, atol=1e-9)
    assert S.norm_isf(0.05) == pytest.approx(1.6448536, abs=1e-6)


def test_poisson_probabilities():
    k = np.arange(60)
    for mu in (0.3, 3.0, 10.0, 25.0):
        p = S.poisson_pmf(k, mu)
        assert p.sum() == pytest.approx(1.0, abs=1e-8)
        assert (p * k).sum() == pytest.approx(mu, abs=1e-6)
        assert p[4] == pytest.approx(math.exp(-mu) * mu ** 4 / 24, rel=1e-12)
        assert S.poisson_cdf(7, mu) == pytest.approx(p[:8].sum(), rel=1e-12)
    assert list(S.poisson_pmf(np.arange(3), 0.0)) == [1.0, 0.0, 0.0]
    assert S.poisson_cdf(4, 10.0) == pytest.approx(0.029252688, rel=1e-7)
    assert S.poisson_cdf(-1, 2.0) == 0.0


def test_upper_limits_are_the_textbook_ones():
    # the classical 90% and 95% limits (Particle Data Group, Poisson, no background)
    assert np.allclose(S.upper_limit(np.arange(4), 0.0, 0.90), [2.3026, 3.8897, 5.3223, 6.6808],
                       atol=1e-4)
    assert np.allclose(S.upper_limit(np.arange(4), 0.0, 0.95), [2.9957, 4.7439, 6.2958, 7.7537],
                       atol=1e-4)
    n = np.arange(30)
    plain, cls = S.upper_limit(n, 10.0, 0.95), S.upper_limit(n, 10.0, 0.95, "cls")
    assert np.allclose(S.poisson_cdf(n, plain + 10.0), 0.05, atol=1e-9)
    assert np.allclose(S.poisson_cdf(n, cls + 10.0) / S.poisson_cdf(n, 10.0), 0.05, atol=1e-9)
    assert (cls > plain).all() and (cls > 0).all() and (np.diff(cls) > 0).all()
    assert cls[0] == pytest.approx(-math.log(0.05), abs=1e-6)     # n = 0: b drops out
    assert np.allclose(S.upper_limit(n, 0.0, 0.95, "cls"), S.upper_limit(n, 0.0, 0.95))


def test_counting_significance():
    assert S.z_counting(100.0, 100.0) == 0.0
    assert S.z_counting(130.0, 100.0) == pytest.approx(
        math.sqrt(2 * (130 * math.log(1.3) - 30)), rel=1e-12)
    assert S.z_counting(80.0, 100.0) < 0 and S.z_counting(0.0, 5.0) == pytest.approx(-math.sqrt(10))
    assert S.z_asimov(5.0, 25.0) == pytest.approx(S.z_counting(30.0, 25.0))
    # both counts grow with luminosity: Z grows as its square root
    assert S.z_asimov(5.0 * 16, 25.0 * 16) == pytest.approx(4 * S.z_asimov(5.0, 25.0))
    assert S.lumi_for(5.0, 5.0) == pytest.approx(25 / S.z_asimov(5.0, 25.0) ** 2)


# ---------------------------------------------------------------------- the bump hunt
def test_bump_data_is_what_it_says():
    d, s = S.bump_data(), study("bump-hunt")
    back, sig = S.bump_truth()
    assert back.sum() == pytest.approx(S.N_BKG) and sig.sum() == pytest.approx(S.N_SIG, rel=1e-6)
    inside = S.signal_shape(S.SCAN).sum(1)             # the outermost masses lose a 3-sigma tail
    assert (inside > 0.998).all() and (inside <= 1 + 1e-12).all() and inside[24] > 1 - 1e-9
    n = np.random.default_rng(S.BUMP_SEED).poisson(back + sig)
    assert np.array_equal(d["n"], n) and s.metadata["events"] == n.sum()
    # the fits are maxima: no gradient left, totals conserved, and the likelihood rose
    assert d["gradient_left"] < 1e-3
    assert d["fit_b"].sum() == pytest.approx(n.sum(), rel=1e-9)
    assert d["fit_sb"].sum() == pytest.approx(n.sum(), rel=1e-6)
    assert d["q_obs"] == pytest.approx(2 * (d["ll_sb"] - d["ll_b"]))
    # the scan at the true mass is the fit at the true mass
    at = int(np.argmin(np.abs(S.SCAN - S.M_SIG)))
    assert d["scan_q"][at] == pytest.approx(float(d["q_obs"]), rel=1e-9)
    assert np.allclose(d["scan_p"], S.norm_sf(np.sqrt(d["scan_q"])))
    assert d["z_max"].shape == (S.N_SCAN,) and d["q_hist"].sum() <= S.N_FIXED
    assert int(d["beyond"]) == int((d["z_max"] >= d["scan_z"].max()).sum())
    # a fresh background-only spectrum gives the recorded kind of q0 (the scan is redone)
    toy = np.random.default_rng(S.BUMP_TOY_SEEDS[1]).poisson(back, (3, len(back)))
    q, _, left = S.q0_scan(toy, S.SCAN)
    assert np.allclose(np.sqrt(q.max(1)), d["z_max"][:3]) and left < 1e-3


def test_bump_claims():
    m = study("bump-hunt").metadata
    t, g = m["q0_tails"], m["global_"]
    # q0 is half a chi-square: half at zero, tails as the normal's (3 sigma: 135 +- 12 expected)
    assert abs(m["q0_zero_share"] - 0.5) < 0.01
    assert abs(t["pseudo_experiments"][0] - t["exact"][0]) < 0.005
    assert abs(t["pseudo_experiments"][1] - t["exact"][1]) < 0.002
    assert 0.6 * t["exact"][2] < t["pseudo_experiments"][2] < 1.4 * t["exact"][2]
    # the fit finds the injected signal, where it was injected
    assert abs(m["fitted_signal"] - S.N_SIG) < 2 * m["fitted_signal_error"]
    assert abs(m["scan"]["best_mass"] - S.M_SIG) <= S.WIDTH
    assert abs(m["with_signal"]["mean_fitted_signal"] - S.N_SIG) < 2.0
    lo, _, hi = m["with_signal"]["z_16_50_84"]
    assert lo - 1.0 < m["z_at_true_mass"] < hi + 1.0   # this spectrum is not an outlier
    # the look-elsewhere cost: the global p-value is many times the local one
    assert g["at_or_above"] >= 5
    assert g["p"] - 2 * g["p_error"] > 5 * m["scan"]["local_p"]
    assert g["z"] < m["scan"]["local_z"] - 0.5


def test_bump_numbers_on_screen():
    anim = anim_of("bump-hunt")
    m, sub = anim.meta, subtitles(anim)
    assert "q0 = %.2f" % m["q0"] in sub[1] and "%.2f sigma" % m["z_at_true_mass"] in sub[1]
    assert "%.3f%% pass 3" % (100 * m["q0_tails"]["pseudo_experiments"][2]) in sub[2]
    assert M.sci(m["scan"]["local_p"]) in sub[3]
    assert "%d reach" % m["global_"]["at_or_above"] in sub[4]
    assert "%.1f sigma instead of %.1f" % (m["global_"]["z"], m["scan"]["local_z"]) in sub[4]
    shown = texts(anim, late(anim, 1))
    assert "Z = √q0 = %.2f sigma" % m["z_at_true_mass"] in shown
    assert "n = %d observed, b = %.1f fitted" % (m["window"]["observed"],
                                                  m["window"]["background_fit"]) in shown
    shown = texts(anim, late(anim, 4))
    assert "= %.2f sigma" % m["global_"]["z"] in shown
    assert any("pseudo-data from a stated model" in t for t in texts(anim, late(anim, 0)))


# ---------------------------------------------------------------------- upper limits
def test_limits_data_is_what_it_says():
    d, s = S.limits_data(), study("poisson-upper-limits")
    assert np.allclose(d["limits"][:4], [2.3026, 3.8897, 5.3223, 6.6808], atol=1e-4)
    assert np.allclose(S.poisson_cdf(S.LIMIT_N, d["limits"]), 1 - S.CL, atol=1e-9)
    assert np.allclose(d["with_b"], d["limits"] - S.LIMIT_B)
    rng = np.random.default_rng(S.LIMIT_SEED)
    assert np.array_equal(d["example"], rng.poisson(S.EXAMPLE_S, S.N_EXAMPLE))
    assert np.allclose(d["example_limits"], S.upper_limit(d["example"], 0.0, S.CL))
    assert np.allclose(d["cross_section"], d["limits"] / (S.LUMI * S.EFFICIENCY))
    # the curves cross 10% at the limits
    for n, row in zip(S.SHOWN_N, d["cdf"]):
        assert np.interp(d["limits"][n], S.S_GRID, row) == pytest.approx(0.10, abs=2e-3)
    assert s.metadata["example"]["covered"] == int((d["example_limits"] >= S.EXAMPLE_S).sum())


def test_limits_claims():
    d, m = S.limits_data(), study("poisson-upper-limits").metadata
    # the procedure never covers less than it promises (exactly), and over-covers on average
    assert d["coverage_fine"].min() >= S.CL - 1e-9
    assert d["coverage_fine"].mean() > S.CL + 0.02
    # pseudo-experiments agree with the exact coverage to binomial accuracy
    sd = np.sqrt(0.1 * 0.9 / S.N_COVER)
    assert np.abs(d["coverage"] - d["coverage_exact"]).max() < 4 * sd
    assert d["coverage"].min() > S.CL - 4 * sd
    # below the n = 0 limit every count covers
    low = d["limits"][0] > S.COVER_S
    assert low.any() and (d["coverage"][low] == 1.0).all()
    # with a background, a count of zero gives a negative limit
    assert m["limits_with_background"]["0"] < 0 < m["limits_with_background"]["1"]


def test_limits_numbers_on_screen():
    anim = anim_of("poisson-upper-limits")
    m, sub = anim.meta, subtitles(anim)
    assert "s = %.2f" % m["limits"]["0"] in sub[0] and "2.30" in sub[0]
    assert "%.2f, %.2f and %.2f" % tuple(m["limits"][k] for k in "123") in sub[1]
    assert M.signed(m["limits_with_background"]["0"]) in sub[2]
    assert M.pct(m["coverage"]["lowest_pseudo"]) in sub[3]
    assert "%.2f fb" % m["cross_section"]["limit_fb"][0] in sub[4]
    assert "s = %.2f" % m["limits"]["0"] in texts(anim, late(anim, 0))
    shown = texts(anim, late(anim, 1))
    for n in (0, 1, 2, 3):
        assert "n = %d:  s < %.2f" % (n, m["limits"][str(n)]) in shown
    assert "σ < %.2f fb" % m["cross_section"]["limit_fb"][0] in texts(anim, late(anim, 4))


# ---------------------------------------------------------------------- CLs
def test_cls_data_is_what_it_says():
    d, m = S.cls_data(), study("cls-method").metadata
    e = m["example"]
    assert e["cls_b"] == pytest.approx(S.poisson_cdf(S.CLS_N, S.CLS_TEST + S.CLS_B))
    assert e["cl_b"] == pytest.approx(S.poisson_cdf(S.CLS_N, S.CLS_B))
    assert e["cls"] == pytest.approx(e["cls_b"] / e["cl_b"])
    assert np.allclose(d["plain"], S.upper_limit(S.CLS_COUNTS, S.CLS_B, S.CLS_CL))
    assert np.allclose(d["cls"], S.upper_limit(S.CLS_COUNTS, S.CLS_B, S.CLS_CL, "cls"))
    n0 = np.random.default_rng(S.CLS_SEED).poisson(S.CLS_B, S.N_CLS)
    assert np.array_equal(d["counts_hist"], np.bincount(n0, minlength=80)[:len(S.CLS_COUNTS)])
    assert d["below"][0][0] == pytest.approx((n0 <= np.flatnonzero(d["plain"] < 0).max()).mean())


def test_cls_claims():
    d, m = S.cls_data(), study("cls-method").metadata
    e, x = m["example"], m["excludes"]
    # the worked count: the plain rule excludes even no signal, CLs does not exclude s = 3
    assert e["plain_limit"] < 0 < e["cls_limit"]
    assert e["cls_b"] < 0.05 < e["cls"]
    # background-only experiments: the plain limit rules out signals it cannot see, CLs never
    assert x["plain"][0] > 0.02 and x["plain"][1] > x["plain"][0] + 0.02
    assert x["cls"] == [0.0, 0.0, 0.0] and x["cls_exact"] == [0.0, 0.0, 0.0]
    assert np.allclose(x["plain"], x["plain_exact"], atol=4 * np.sqrt(0.13 * 0.87 / S.N_CLS))
    # the price: CLs covers more everywhere, most at small signals, and its limit is weaker
    assert (d["coverage_fine"][1] >= d["coverage_fine"][0] - 1e-12).all()
    assert d["coverage_fine"][0].min() >= S.CLS_CL - 1e-9
    assert d["coverage"][1][0] > d["coverage"][0][0] + 0.01
    assert np.abs(d["coverage"] - d["coverage_exact"]).max() < 4 * np.sqrt(0.05 * 0.95 / 20000)
    assert m["median_limit"]["cls"] > m["median_limit"]["plain"] + 0.5


def test_cls_numbers_on_screen():
    anim = anim_of("cls-method")
    m, sub = anim.meta, subtitles(anim)
    e = m["example"]
    assert M.signed(e["plain_limit"]) in sub[0] and "−" in sub[0]
    assert "CLs = %.3f" % e["cls"] in sub[1]
    assert "it is %.2f" % e["cls_limit"] in sub[2]
    assert M.pct(m["excludes"]["plain"][1]) in sub[3] and M.pct(m["excludes"]["plain"][0]) in sub[3]
    assert "%.1f against %.1f" % (m["median_limit"]["cls"], m["median_limit"]["plain"]) in sub[4]
    shown = texts(anim, late(anim, 1))
    assert "= %.5f / %.4f = %.3f" % (e["cls_b"], e["cl_b"], e["cls"]) in shown
    assert "the plain 95%% limit: s < %s" % M.signed(e["plain_limit"]) in texts(anim, late(anim, 0))


# ---------------------------------------------------------------------- the bands
def test_band_data_is_what_it_says():
    d, m = S.band_data(), study("brazil-band").metadata
    back, ref = S.band_model()
    assert np.allclose(d["back"], back) and np.allclose(d["ref"], ref)
    i = S.BAND_AT
    counts = np.arange(100)
    assert np.allclose(d["table"][i, :100],
                       S.upper_limit(counts, back[i], S.BAND_CL, "cls") / ref[i])
    toys = np.random.default_rng(S.BAND_SEEDS[0]).poisson(back[:, None], (len(back), S.N_BAND))
    assert np.array_equal(d["counts_at"][:100], np.bincount(toys[i], minlength=100)[:100])
    limits = d["table"][i][toys[i]]
    assert np.allclose(d["bands"][:, i],
                       np.percentile(limits, S.QUANTILES, method="inverted_cdf"))
    expect = back.copy()
    expect[S.BAND_INJECT] += S.BAND_MU * ref[S.BAND_INJECT]
    obs = np.random.default_rng(S.BAND_SEEDS[1]).poisson(expect)
    assert np.array_equal(d["observed"], obs)
    assert np.allclose(d["observed_limit"], d["table"][np.arange(len(back)), obs])
    assert d["out_count"].sum() == S.N_EXITS and m["leaving"]["plots"] == S.N_EXITS
    assert np.array_equal(d["lines_out"],
                          (d["lines"] < d["bands"][0]) | (d["lines"] > d["bands"][4]))


def test_band_claims():
    d, m = S.band_data(), study("brazil-band").metadata
    bands, lv = d["bands"], m["leaving"]
    assert (np.diff(bands, axis=0) > 0).all()             # 2.5 < 16 < 50 < 84 < 97.5 at every mass
    # the bands hold what they say, or a little more (counts come in steps)
    step = np.abs(np.diff(d["table"], axis=1))[:, :80].max()
    assert np.abs(bands - d["bands_exact"]).max() <= step + 1e-9
    assert m["band_edges_equal_to_exact"] >= 0.75 * m["band_edges"]
    assert 0.02 < lv["per_point"] <= 0.05 + 0.002
    assert 0.25 < lv["outside_68_per_point"] <= 0.32 + 0.005
    assert abs(lv["per_point"] - lv["per_point_exact"]) < 0.003
    assert abs(lv["any_point"] - lv["any_point_exact"]) < 0.015
    # leaving the band somewhere is common although each point rarely does
    assert lv["any_point"] > 5 * lv["per_point"]
    # the observed line: far above the band where the signal was injected, inside elsewhere
    obs, k = d["observed_limit"], S.BAND_INJECT
    assert obs[k] > bands[4, k] + 0.3 and obs[k] > 2 * bands[2, k]
    assert m["observed"]["outside_95"] == [float(S.BAND_M[k])]


def test_band_numbers_on_screen():
    anim = anim_of("brazil-band")
    m, sub = anim.meta, subtitles(anim)
    q = [m["bands"][k][S.BAND_AT] for k in ("2.5", "16", "50", "84", "97.5")]
    assert "median of %.2f" % q[2] in sub[1] and "%.2f to %.2f" % (q[1], q[3]) in sub[1]
    assert "%.2f to %.2f" % (q[0], q[4]) in sub[1]
    k = S.BAND_INJECT
    assert "rises to %.2f" % m["observed"]["limits"][k] in sub[3]
    assert "expected %.2f" % m["bands"]["50"][k] in sub[3]
    assert M.pct(m["leaving"]["per_point"]) in sub[4]
    assert M.pct(m["leaving"]["any_point"], 0) in sub[4]
    assert "central 68%%: %.2f to %.2f" % (q[1], q[3]) in texts(anim, late(anim, 1))
    assert "at least one: %s of plots" % M.pct(m["leaving"]["any_point"], 0) in texts(
        anim, late(anim, 4))
    # the pseudo-experiment worked on screen: its limit is its count's
    n = int(anim.first[anim.TOYS - 1])
    mu = S.upper_limit(n, anim.b, S.BAND_CL, "cls") / anim.ref
    assert "μ < %.1f / %.1f = %.2f" % (mu * anim.ref, anim.ref, mu) in texts(anim, late(anim, 0))


# ---------------------------------------------------------------------- five sigma
def test_five_data_is_what_it_says():
    d, s = S.five_data(), study("why-five-sigma")
    assert d["none"].shape == d["with_"].shape == (len(S.LOOKS), S.N_BINS)
    assert (np.diff(d["none"], axis=0) >= 0).all() and (np.diff(d["with_"], axis=0) >= 0).all()
    expect = S.B_RATE * S.LOOKS[:, None]
    assert np.allclose(d["z_none"], S.z_counting(d["none"], expect))
    assert np.allclose(d["z_with"], S.z_counting(d["with_"], expect))
    rng = np.random.default_rng(S.FIVE_SEEDS[0])
    assert np.array_equal(d["none"], np.cumsum(rng.poisson(S.B_RATE, (len(S.LOOKS), S.N_BINS)), 0))
    assert np.allclose(d["tails"], [1.349898e-3, 2.866516e-7], rtol=1e-6)
    assert list(d["reach_of"]) == [S.N_SEARCH] * 3 + [S.N_SEARCH // S.GROUP]
    assert np.allclose(s.panels[0].data["ys"][1], S.z_asimov(S.S_RATE * S.LOOKS,
                                                             S.B_RATE * S.LOOKS))


def test_five_claims():
    d, m = S.five_data(), study("why-five-sigma").metadata
    r = m["reach"]
    three, five = r["three_share"], r["five_share"]
    # one bin, one look: close to the Gaussian tail (counts are whole numbers)
    assert 0.6 * 1.35e-3 < three[0] < 1.4 * 1.35e-3
    # every further place and time to look multiplies the chance of a 3-sigma excess
    assert three[1] > 4 * three[0] and three[2] > 10 * three[1] and three[3] > 3 * three[2]
    assert three[2] > 0.1 and three[3] > 0.7
    # ... while 5 sigma stays rare: at least a thousand times rarer than 3 sigma
    assert five[2] < three[2] / 1000 and r["five"][0] == 0
    # background-only bins rise and fade; the signal's bin keeps growing, about as sqrt(L)
    f = m["followed"]
    assert f["background_peak"] > 2.0 and f["background_last"] < f["background_peak"] - 0.5
    z_sig = d["z_with"][:, S.SIG_BIN]
    assert z_sig[-1] > 5.0 and z_sig[-1] > z_sig[4] + 2.0
    assert r["one_bin_ever_two"] > 0.05
    med = d["median"][1]                                   # the signal followed, over 5,000
    for lum in (4, 16, 64):
        assert med[lum - 1] == pytest.approx(S.z_asimov(S.S_RATE * lum, S.B_RATE * lum), abs=0.12)
    assert med[63] / med[15] == pytest.approx(2.0, abs=0.1)
    # half the signal needs about four times the data
    lum, formula = m["lumi_for_five"]["pseudo_experiments"], m["lumi_for_five"]["formula"]
    assert np.allclose(lum, formula, atol=1.5)
    assert 3.3 < lum[0] / lum[1] < 4.5 and 3.3 < formula[1] / formula[2] < 4.5


def test_five_numbers_on_screen():
    anim = anim_of("why-five-sigma")
    m, sub = anim.meta, subtitles(anim)
    r = m["reach"]
    assert "%.1f sigma at L = %g" % (anim.z_sig[-1], S.LOOKS[-1]) in sub[1]
    assert "%.1f sigma at" % m["followed"]["background_peak"] in sub[1]
    assert M.pct(r["three_share"][2]) in sub[2] and M.pct(r["three_share"][0], 2) in sub[2]
    assert M.sci(m["tails"]["3"]) in sub[3] and M.sci(m["tails"]["5"]) in sub[3]
    assert "%d of %s searches" % (r["five"][2], M.commas(S.N_SEARCH)) in sub[3]
    assert "L = %g for this signal" % m["lumi_for_five"]["pseudo_experiments"][1] in sub[4]
    shown = texts(anim, late(anim, 2))
    assert "%d of %s" % (r["five"][2], M.commas(S.N_SEARCH)) in shown
    assert M.pct(r["three_share"][2], 1) in shown
    shown = texts(anim, late(anim, 3))
    assert "beyond 3 sigma: %s, 1 in 741" % M.sci(m["tails"]["3"]) in shown
    assert "beyond 5 sigma: %s, 1 in 3.5 million" % M.sci(m["tails"]["5"]) in shown

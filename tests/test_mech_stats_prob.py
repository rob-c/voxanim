"""Animations 118-123 (voxanim.mech_stats_prob) and their experiments
(voxanim.study_stats_prob): the layout is clean, the loop closes, every phase says what
kind of evidence it shows, the special functions give known values, each study's claim
holds (exactly where it is exact, with a margin where it is counted from seeded draws),
and the numbers on screen are the study's.

None of the six needs data: they are built here from an empty cache.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from voxanim import cache, layout, presets, qa
from voxanim import mech_stats_prob as M
from voxanim import study_stats_prob as S
from voxanim.next_catalog import NO_DATA, build_study

FB, AREAS, PRIOR = "frequentist-and-bayesian", "bayes-theorem-areas", "prior-washing-out"
COVER, BPG, CLT = "interval-coverage", "binomial-poisson-gaussian", "central-limit-theorem"
NAMES = (FB, AREAS, PRIOR, COVER, BPG, CLT)
EACH = [pytest.param(n) for n in NAMES]


def study(name):
    return build_study("study_stats_prob", name)


def anim(name):
    return presets.get(name).build()


def subtitles(a):
    return [a.subtitle_for(k) for k in range(len(a.PHASES))]


def texts(a, t):
    return [x.s for x in a.scene(t).texts]


def at(a, phase, u):
    lo, hi = a.spans[phase]
    return min(lo + (hi - lo) * u, hi - a.FADE - 0.05)


# ---------------------------------------------------------------------- all six
def test_registered_and_need_no_data():
    assert set(M.ANIMATIONS) == set(S.STUDIES) == set(NAMES) <= NO_DATA


@pytest.mark.parametrize("name", EACH)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", EACH)
def test_drawn_by_this_module_and_loop_closes(name):
    a = anim(name)
    assert type(a) is M.ANIMATIONS[name]
    assert 45.0 <= a.t_done <= 60.0 and len(a.PHASES) >= 4

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:                            # every phase draws things, not only text
        sc = a.scene((lo + hi) / 2)
        assert len(sc.quads) + len(sc.lines) >= 3


@pytest.mark.parametrize("name", EACH)
def test_subtitles_fit_two_lines_and_every_phase_is_tagged(name):
    a = anim(name)
    for sub in subtitles(a):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    for k in range(len(a.PHASES)):                    # the tag: bottom right, right-anchored
        tags = [t.s for t in a.scene(at(a, k, 0.6)).texts if t.anchor == "rm" and t.y > 0.86]
        assert len(tags) == 1
        assert tags[0].startswith("exact") or "pseudo-experiments" in tags[0] \
            or "seeded" in tags[0]


@pytest.mark.parametrize("name", EACH)
def test_study_says_what_was_run(name):
    s = study(name)
    m = s.metadata
    assert len(s.panels) == 4 and m["designs_tried"] and m["limitations"]
    assert set(m["evidence"]) == {"exact", "pseudo_experiments"}
    assert m["seed"] in m["seeds"] and "default_rng" in m["where_the_seed_enters"]


def test_all_six_build_from_an_empty_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache, "DEFAULT_CACHE", str(tmp_path))
    for fn in (S.fb_data, S.bayes_data, S.prior_data, S.cover_data, S.bpg_data, S.clt_data):
        fn.cache_clear()
    for name in NAMES:                                # offline (conftest): a download would raise
        M.ANIMATIONS[name](S.STUDIES[name]().validate()).scene(3.0)
    assert list(tmp_path.iterdir()) == []             # nothing fetched, nothing stored


def test_seeded_draws_repeat():
    first = {k: v.copy() for k, v in S.clt_data().items()}
    S.clt_data.cache_clear()
    again = S.clt_data()
    assert all(np.array_equal(first[k], again[k]) for k in first)
    assert np.array_equal(S.flips(7, 50), S.flips(7, 50))
    assert not np.array_equal(S.flips(7, 50), S.flips(8, 50))


# ---------------------------------------------------------------------- special functions
def test_distributions_give_known_values():
    assert np.isclose(S.CL, 0.6826894921) and np.isclose(S.TAIL, 0.1586552539)
    p = S.poisson_pmf(3.0, 60)
    assert np.isclose(p[3], 27 * math.exp(-3) / 6) and np.isclose(p.sum(), 1.0)
    assert np.isclose((np.arange(61) * p).sum(), 3.0)
    assert S.poisson_pmf(0.0, 3).tolist() == [1, 0, 0, 0]
    assert np.isclose(S.poisson_cdf(1, 2.0), 3 * math.exp(-2)) and S.poisson_cdf(-1, 2.0) == 0
    b = S.binomial_pmf(10, 0.3)
    assert np.isclose(b[3], 120 * 0.3 ** 3 * 0.7 ** 7) and np.isclose(b.sum(), 1.0)
    assert S.binomial_pmf(4, 1.0).tolist() == [0, 0, 0, 0, 1]
    assert np.allclose(S.normal_cdf([0.0, 1.0, -1.0, 1.959964]),
                       [0.5, 0.8413447461, 0.1586552539, 0.975])
    assert np.isclose(S.normal_pdf(0.0), 1 / math.sqrt(2 * math.pi))
    assert np.isclose(S.solve(lambda x: x * x - 2.0, 0.0, 2.0), math.sqrt(2.0))
    assert S.tv([1, 0], [0, 1]) == 1.0 and S.tv([0.5, 0.5], [0.5, 0.5]) == 0.0
    assert np.isclose(S.poisson_tail(26, 10.0), 1.0 - S.poisson_cdf(25, 10.0), rtol=1e-6)


def test_intervals_give_known_values():
    # the 68.27% central limits for small counts, as tabulated (e.g. PDG, Statistics)
    for n, (lo, hi) in {0: (0.0, 1.841), 1: (0.173, 3.300), 2: (0.708, 4.638),
                        3: (1.367, 5.918), 10: (6.891, 14.267)}.items():
        assert np.allclose(S.poisson_central(n), (lo, hi), atol=2e-3)
    assert np.isclose(S.poisson_central(0)[1], -math.log(S.TAIL))
    # the two ends solve what they are defined to solve
    lo, hi = S.poisson_central(7)
    assert np.isclose(1 - S.poisson_cdf(6, lo), S.TAIL) and np.isclose(S.poisson_cdf(7, hi),
                                                                     S.TAIL)
    # no count: the posterior under a flat prior is exp(-mu)
    assert np.allclose(S.poisson_credible(0), (-math.log(1 - S.TAIL), -math.log(S.TAIL)))
    assert np.allclose(S.poisson_credible(0, b=3.0), S.poisson_credible(0))
    # Clopper-Pearson at the ends has a closed form
    assert np.allclose(S.clopper_pearson(0, 20), (0.0, 1 - S.TAIL ** (1 / 20)))
    assert np.allclose(S.clopper_pearson(20, 20), (S.TAIL ** (1 / 20), 1.0))
    lo, hi = S.clopper_pearson(7, 20)
    assert np.isclose(S.binomial_pmf(20, lo)[7:].sum(), S.TAIL)
    assert np.isclose(S.binomial_pmf(20, hi)[:8].sum(), S.TAIL)
    assert S.wald(20, 20) == (1.0, 1.0) and np.allclose(S.wald(10, 20), (0.5 - 0.1118, 0.6118),
                                                       atol=1e-4)
    assert S.poisson_sqrt(4) == (2.0, 6.0)
    pmf = np.array([0.2, 0.5, 0.3])
    assert np.isclose(S.coverage(pmf, [0, 0, 2], [1, 3, 3], 2.0), 0.8)


# ---------------------------------------------------------------------- 118
def test_fb_the_belt_read_across_is_the_interval():
    d = S.fb_data()
    for n in (0, 1, 3, 6):
        lo, hi = d["freq"][n]
        holds = (d["belt"][:, 0] <= n) & (n <= d["belt"][:, 1])
        inside = (lo <= S.BELT_MUS) & (hi >= S.BELT_MUS)
        assert np.array_equal(holds, inside)          # every mean whose range includes n
    for mu, (a, b) in zip(S.BELT_MUS, d["belt"]):     # each range holds at least 68.27%
        assert S.poisson_pmf(float(mu), 60)[a:b + 1].sum() >= S.CL


def test_fb_where_the_two_intervals_agree_and_differ():
    d = S.fb_data()
    f, b = d["freq"], d["bayes"]
    assert np.allclose(f[:, 1], b[:, 1], atol=1e-9)   # flat prior: the same upper end, exactly
    assert (b[:, 0] > f[:, 0] + 0.15).all()           # and a higher lower end
    assert np.allclose(d["inverse"][:, 0], f[1:, 0], atol=1e-9)   # prior 1/mu: the lower end
    assert np.allclose(b[:-1, 0], f[1:, 0], atol=1e-9)
    gap = (b[:, 0] - f[:, 0]) / (f[:, 1] - f[:, 0])
    assert gap[20] < 0.1 < gap[3] and (np.diff(gap[1:]) < 0).all()   # they close up as n grows
    post = d["like"]
    area = post.sum() * (d["grid"][1] - d["grid"][0])   # up to mu = 12; the rest is beyond
    assert np.isclose(area, 1 - S.poisson_cdf(S.N_OBS, 12.0), atol=1e-4)
    lo, hi = b[S.N_OBS]
    inside = (d["grid"] >= lo) & (d["grid"] <= hi)
    assert abs(post[inside].sum() * (d["grid"][1] - d["grid"][0]) - S.CL) < 0.01


def test_fb_coverage_and_the_boundary():
    s = study(FB)
    cov = s.metadata["coverage_at_truth"]
    assert np.allclose(cov["exact"], cov["pseudo_experiments"], atol=0.012)
    x, ys = s.panels[3].data["x"], s.panels[3].data["ys"]
    assert ys[0].min() >= S.CL                        # confidence: never under, whatever mu
    assert ys[1][x >= 1].min() < S.CL - 0.1           # credible: no such promise
    d = S.fb_data()
    assert d["b_freq"][0, 1] < -1.0                   # n = 0 on background 3: no allowed value
    assert (d["b_bayes"] >= 0).all() and (d["b_bayes"][:, 0] > 0).all()
    assert abs(d["b_bayes"][-1, 1] - d["b_freq"][-1, 1]) < 0.02   # far from the boundary: alike


def test_fb_numbers_on_screen():
    a, s = anim(FB), study(FB)
    m = s.metadata
    subs = subtitles(a)
    ci, cr = m["confidence_interval"], m["credible_interval"]
    assert "from %.2f to %.2f" % tuple(ci) in subs[1]
    assert "from %.2f to %.2f" % tuple(cr) in subs[2]
    assert M.pct(m["coverage_at_truth"]["pseudo_experiments"][0]) in subs[3]
    late = texts(a, at(a, 1, 0.95))
    assert "%.2f ≤ μ ≤ %.2f" % tuple(ci) in late
    assert "%.2f ≤ μ ≤ %.2f" % tuple(cr) in texts(a, at(a, 2, 0.95))
    assert any("P(3 | μ = 3) = %.3f" % S.poisson_pmf(3.0, 3)[3] == t for t in texts(a, 4.0))
    assert a.row3 == S.poisson_band(3.0)


# ---------------------------------------------------------------------- 119
def test_areas_are_bayes_theorem():
    s = study(AREAS)
    v = s.panels[0].data["values"]
    assert np.isclose(v.sum(), 1.0)
    post = v[0] / (v[0] + v[2])
    assert np.isclose(post, 0.99 * 0.001 / (0.99 * 0.001 + 0.05 * 0.999))
    assert np.isclose(post, s.metadata["disease"]["posterior"]) and 0.019 < post < 0.02
    chain = s.panels[1].data["ys"][0]
    assert np.allclose(chain[1:], [S.posterior(p, 0.99, 0.05)[2] for p in chain[:-1]])
    assert chain[3] > 0.85 > 0.3 > chain[2] > chain[1]


def test_areas_kaon_sample_is_what_it_says():
    d = S.bayes_data()
    assert d["counts"].sum() == S.PARTICLES
    x, y = d["xy"].T
    kaon = x < S.KAON_FRACTION
    tagged = y < np.where(kaon, S.EFFICIENCY, S.MISID)
    assert np.array_equal(d["classes"], 2 * kaon + tagged)
    m = study(AREAS).metadata["kaon"]
    assert np.isclose(m["purity"], 2 / 3) and abs(m["purity_counted"] - m["purity"]) < 0.015
    exact, toys = d["exact"], d["toys"]
    assert (np.diff(exact, axis=1) > 0).all()         # purer as the prior grows
    assert (exact[0] < exact[1]).all() and (exact[1] < exact[2]).all()
    at_toys = np.array([[S.posterior(p, S.EFFICIENCY, mis)[2] for p in S.PRIORS]
                        for mis in S.MISIDS])
    assert np.abs(toys - at_toys).max() < 0.06 and np.abs(toys - at_toys)[:, 3:].max() < 0.02


def test_areas_numbers_on_screen():
    a, s = anim(AREAS), study(AREAS)
    subs = subtitles(a)
    assert (a.n_tp, a.n_fn, a.n_fp, a.n_tn) == (99, 1, 4995, 94905)
    assert M.pct(s.metadata["disease"]["posterior"], 2) in subs[1] and "1.94%" in subs[1]
    assert any(t.startswith("P(ill | positive) = 99 / (99 + 4,995) = 1.94%")
               for t in texts(a, at(a, 1, 0.9)))
    c = s.metadata["kaon"]["counted"]
    assert any("purity = %s / (%s + %s)" % (M.commas(c[0]), M.commas(c[0]), M.commas(c[2])) in t
               for t in texts(a, at(a, 3, 0.95)))
    assert M.pct(s.metadata["kaon"]["purity_counted"]) in subs[3]


# ---------------------------------------------------------------------- 120
def test_prior_posteriors_are_exact():
    priors = S.log_priors()
    p = S.p_grid()
    post = S.posteriors(priors, 8, 10)
    assert np.allclose(post.sum(1) / S.GRID, 1.0)
    mean = (post * p).sum(1) / S.GRID
    assert np.isclose(mean[0], 9 / 12, atol=1e-6)                 # Beta(9, 3)
    assert np.isclose(mean[1], (18 + 8) / (20 + 10), atol=1e-6)   # Beta(26, 4)
    assert np.isclose(mean[2], (15 + 8) / (30 + 10), atol=1e-6)
    outside = (p < S.DOGMA[0]) | (p > S.DOGMA[1])
    assert (post[4][outside] == 0).all() and S.density_tv(post[0], post[0]) == 0.0
    d = S.prior_data()
    seq = S.flips(S.PRIOR_SEED)
    assert np.array_equal(d["first"], seq[:40])
    assert np.array_equal(d["heads_at"], [seq[:n].sum() for n in S.CHECKS])
    assert abs(seq.mean() - S.P_TRUE) < 0.005


def test_priors_wash_out_unless_they_are_zero_at_the_truth():
    s = study(PRIOR)
    far, x = s.panels[1].data["ys"], list(s.panels[1].data["x"])
    med = s.panels[2].data["ys"]
    for run in (far, med):
        assert run[0, -1] < 0.03 and run[1, -1] < 0.03        # optimistic, sceptical: gone
        assert run[0, x.index(10)] > 0.4 and run[1, x.index(10)] > 0.4
        assert run[2, x.index(1000)] > 0.7                    # the narrow prior needs far more
        assert 0.05 < run[2, -1] < 0.2
        assert run[3, -1] > 0.999 and run[3, x.index(1000)] > 0.99   # the dogmatic one: never
        assert (run[2] >= run[1] - 0.05).all()
    means = s.panels[3].data["ys"][:, -1]
    assert np.abs(means[:3] - S.P_TRUE).max() < 0.005 and abs(means[3] - S.P_TRUE) < 0.01
    assert means[4] <= S.DOGMA[1] and means[4] > 0.599


def test_prior_numbers_on_screen():
    a, s = anim(PRIOR), study(PRIOR)
    subs = subtitles(a)
    far = s.panels[1].data["ys"]
    assert "%.3f (optimistic), %.3f (sceptical)" % (far[0, -1], far[1, -1]) in subs[4]
    d = S.prior_data()
    assert a.heads[10] == int(d["first"][:10].sum()) and "%d heads" % a.heads[10] in subs[0]
    full = (d["shown"] * d["p"]).sum(-1) / d["shown"].sum(-1)
    assert np.allclose(a.means, full, atol=2e-3)      # the thinned curves keep their means
    i = list(S.SHOWN_NS).index(10)
    post = S.posteriors(S.log_priors(), int(d["shown_heads"][i]), 10)
    assert abs(a.gap[i, 2] - S.density_tv(post[0], post[2])) < 5e-3
    k = a.heads[10]
    assert "its mean: (%d + 1) / (10 + 2) = %.3f" % (k, (k + 1) / 12) in texts(a, at(a, 0, 0.95))


# ---------------------------------------------------------------------- 121
def test_coverage_of_one_sigma_is_68_percent():
    s = study(COVER)
    g = s.metadata["gaussian"]
    d = S.cover_data()
    assert g["first"] == int(d["inside"].sum()) and 55 <= g["first"] <= 80
    assert abs(g["all"] - S.CL) < 0.012
    assert np.array_equal(d["inside"], np.abs(d["x"] - S.G_MU) <= S.G_SIGMA)
    assert np.isclose(d["running"][-1], g["all"])


def test_coverage_naive_intervals_fail_and_exact_ones_do_not():
    s = study(COVER)
    m = s.metadata
    ex = m["poisson_example"]
    p = S.poisson_pmf(2.5, 9)
    assert np.isclose(ex["naive"], p[2] + p[3] + p[4]) and ex["naive"] < S.CL - 0.05
    assert s.panels[1].data["holds"].tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    cover, bcover = s.panels[2].data["ys"], s.panels[3].data["ys"]
    assert cover[1].min() >= S.CL and bcover[1].min() >= S.CL     # exact: never under
    assert cover[0].min() < 0.1 and (cover[0] < S.CL - 0.02).mean() > 0.2
    assert bcover[0].min() < 0.15 and (bcover[0] < S.CL - 0.02).mean() > 0.3
    mus, ps = s.panels[2].data["x"], s.panels[3].data["x"]
    assert (cover[0][mus < 1] < 0.6).all()            # small means: sqrt(n) at its worst
    assert (bcover[0][(ps < 0.03) | (ps > 0.97)] < 0.5).all()
    # the pseudo-experiments agree with the exact sums at the same true values
    for xs, toy_x, toys in ((mus, S.TOY_MUS, s.panels[2].data["toys"]),
                            (ps, S.TOY_PS, s.panels[3].data["toys"])):
        iv = [(S.cover_data()["naive"], S.cover_data()["exact"]),
              (S.cover_data()["wald"], S.cover_data()["cp"])][xs is ps]
        for j, t in enumerate(toy_x):
            pmf = S.binomial_pmf(S.TRIALS, float(t)) if xs is ps else S.poisson_pmf(float(t), 80)
            for i in range(2):
                exact = S.coverage(pmf, iv[i][:, 0], iv[i][:, 1], float(t))
                assert abs(toys[i, j] - exact) < 0.02


def test_coverage_numbers_on_screen():
    a, s = anim(COVER), study(COVER)
    m = s.metadata
    subs = subtitles(a)
    assert "%d contain μ" % m["gaussian"]["first"] in subs[1] and M.pct(m["gaussian"]["all"]) \
        in subs[1]
    assert M.pct(m["poisson_example"]["naive"]) in subs[2] and "n = 2, 3, 4" in subs[2]
    assert "%d contain μ" % m["gaussian"]["first"] in texts(a, at(a, 1, 0.6))
    assert any(t.startswith("coverage = P(2) + P(3) + P(4) = 0.257 + 0.214 + 0.134 = 60.4%")
               for t in texts(a, at(a, 2, 0.95)))
    assert M.pct(m["binomial_range"]["clopper_pearson"][0]) in subs[4]


# ---------------------------------------------------------------------- 122
def test_bpg_distances():
    s = study(BPG)
    d = S.bpg_data()
    seq_tv = s.panels[0].data["tv"]
    assert (np.diff(seq_tv) < 0).all() and seq_tv[0] > 0.2 and seq_tv[-1] < 2e-4
    n, ys = s.panels[1].data["x"], s.panels[1].data["ys"]
    many = n >= 30
    assert np.allclose((ys[0] * n)[many], ys[0][-1] * n[-1], rtol=0.15)   # falls as 1 / n
    assert (ys[0] <= np.minimum(S.LAM / n, S.LAM ** 2 / n) + 1e-12).all()   # Le Cam's bound
    assert ys[1].min() > 0.08 and ys[1][-1] > 100 * ys[0][-1]     # p not small: no approach
    assert ys[2][-1] < 0.002 and (np.diff(ys[2][5:]) < 0).all()   # ... but to the Gaussian
    mu, g = s.panels[2].data["x"], s.panels[2].data["ys"][0]
    big = mu > 10
    assert np.allclose(g[big] * np.sqrt(mu[big]), g[-1] * np.sqrt(mu[-1]), rtol=0.1)
    assert np.isclose(d["negative"][0], S.normal_cdf(-1.5)) and d["negative"][0] > 0.06
    assert np.isclose(S.tv_binomial_poisson(10, 0.3),
                      S.tv(np.pad(S.binomial_pmf(10, 0.3), (0, 40)), S.poisson_pmf(3.0, 50)),
                      atol=1e-9)
    assert abs(d["counted"] - d["example"]).max() < 0.01          # 20,000 rows of 10
    assert np.array_equal(d["hits"].sum(1), d["hits"].sum(1).astype(int)) and \
        d["hits"].shape == (S.ROWS, S.SLOTS)


def test_bpg_the_gaussian_tail_is_too_small():
    s = study(BPG)
    ratio = s.panels[3].data["values"]
    assert ratio[0, 4] > 30 and ratio[1, 4] > 4 and ratio[2, 4] > 1.5    # at 5 sigma
    assert (np.diff(ratio, axis=1)[:, 1:] > 0).all()   # worse the further out
    assert (np.diff(ratio[:, 2:], axis=0) < 0).all()   # better, slowly, as the mean grows
    k = int(s.panels[3].data["threshold"][0, 4])
    assert k == 26 and np.isclose(s.panels[3].data["exact"][0, 4], S.poisson_tail(26, 10.0))
    assert np.isclose(s.panels[3].data["gaussian"][0, 4],
                      1 - S.normal_cdf(15.5 / math.sqrt(10.0)), rtol=1e-6)


def test_bpg_numbers_on_screen():
    a, s = anim(BPG), study(BPG)
    subs = subtitles(a)
    tv = s.panels[0].data["tv"]
    assert "distance %.3f at n = 10" % tv[S.SEQ_N.index(10)] in subs[1]
    assert "%.0f times too small" % s.panels[3].data["values"][0, 4] in subs[4]
    assert any(t.endswith("= %.3f" % S.binomial_pmf(10, 0.3)[3]) and t.startswith("P(3)")
               for t in texts(a, at(a, 0, 0.95)))
    assert "%.4f" % tv[-1] in texts(a, at(a, 1, 0.95))
    assert "%.4f" % s.panels[2].data["at_means"][-1] in texts(a, at(a, 2, 0.95))
    for (ks, pmf, _, _), mu in zip(a.shapes, a.means):
        assert np.allclose(pmf, S.poisson_pmf(mu, int(ks[-1]))[int(ks[0]):])


# ---------------------------------------------------------------------- 123
def test_clt_exact_distributions():
    assert np.allclose(S.dice_sum_pmf(1), 1 / 6)
    assert np.allclose(S.dice_sum_pmf(2) * 36, [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1])
    p = S.dice_sum_pmf(30)
    total = np.arange(30, 181)
    assert np.isclose(p.sum(), 1) and np.isclose((p * total).sum(), 105.0)
    assert np.isclose((p * (total - 105.0) ** 2).sum(), 30 * 35 / 12)
    assert np.isclose(S.exp_mean_cdf(1.0, 1)[0], 1 - math.exp(-1))
    x = np.linspace(0, 12, 24001)
    for n in (1, 2, 5, 30):
        f = S.exp_mean_pdf(x, n)
        step = x[1] - x[0]
        assert np.isclose(f.sum() * step, 1.0, atol=1e-3)
        assert np.isclose((f * x).sum() * step, 1.0, atol=1e-3)
        assert np.allclose(np.cumsum(f)[::2000] * step, S.exp_mean_cdf(x[::2000], n), atol=2e-3)
    assert np.isclose(S.exp_distance(1), S.normal_cdf(-1.0), atol=1e-4)   # the gap at x = 0
    rng = np.random.default_rng(0)
    assert S.sample_distance(rng.normal(2.0, 3.0, 20000), 2.0, 3.0) < 0.012
    q = np.quantile(S.cauchy(rng, 200000), [0.25, 0.5, 0.75])
    assert np.allclose(q, [-1, 0, 1], atol=0.02)


def test_clt_width_falls_as_root_n_and_the_bell_arrives():
    s = study(CLT)
    d = s.panels[2].data
    n = d["x"]
    assert np.allclose(d["sd"][:2], d["sd_exact"], rtol=0.03)     # measured against sigma/sqrt N
    assert np.allclose(d["sd_exact"][0] * np.sqrt(n), S.DIE_SD)
    exact, measured = d["ys"], d["measured"]
    assert (np.diff(exact, axis=1) < 0).all()
    assert (exact[:, -1] < exact[:, 0] / 9).all()     # 100 draws: a tenth of the gap or less
    assert np.abs(measured - exact).max() < 0.015
    assert exact[1, 3] > exact[0, 3]                  # the lopsided start is further away at N = 5
    for i, k in enumerate(S.NS):
        counted = s.panels[0].data["counted_%d" % k]
        assert abs(counted - S.dice_sum_pmf(k)).max() < 0.012
        assert np.isclose(s.panels[1].data["ys"][i].sum() * (S.EXP_RANGE[1] / S.BINS), 1.0,
                          atol=0.03)


def test_clt_a_breit_wigner_never_settles():
    s = study(CLT)
    d = s.panels[3].data
    assert np.abs(d["half_width"] - 1.0).max() < 0.05          # as wide at N = 100 as at 1
    sd = s.panels[2].data["sd"]
    assert sd[2].min() > 20 * sd[1].max()             # no sigma/sqrt(N): no sigma
    assert np.abs(d["settles"][:, -1] - 1.0).max() < 0.03
    late = d["x"] >= 1000                             # the last 9,000 of 10,000 draws
    moved = np.ptp(d["ys"][:, late], axis=1)
    assert moved.min() > 10 * np.ptp(d["settles"][:, late], axis=1).max()   # still jumping
    for k in S.NS:
        centre = d["density_%d" % k][S.BINS // 2 - 1:S.BINS // 2 + 1].mean()
        assert abs(centre - 1 / math.pi) < 0.03       # the same height at 0 for every N


def test_clt_numbers_on_screen():
    a, s = anim(CLT), study(CLT)
    subs = subtitles(a)
    d = s.panels[2].data
    i30 = list(d["x"]).index(30)
    assert "%.2f at N = 30" % d["sd_exact"][0, i30] in subs[1]
    assert "to %.3f at N = 30" % d["ys"][1, i30] in subs[3]
    assert "%.2f at 30" % s.panels[3].data["half_width"][i30] in subs[4]
    late = texts(a, at(a, 1, 0.95))
    assert "σ/√N = %.2f" % (S.DIE_SD / math.sqrt(30)) in late and "N = 30" in late
    assert any("P(mean = 3.5) = 6 / 36 = 0.167" in t for t in texts(a, at(a, 0, 0.95)))
    assert np.array_equal(a.rolls, S.clt_data()["rolls"])

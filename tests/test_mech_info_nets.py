"""Animations 174-175 (voxanim.mech_info_nets) and their experiments (voxanim.study_info):
the layout is clean, the loop closes, every phase says what kind of evidence it shows,
the numbers on screen are the study's, and what the studies claim holds when it is
worked out again from their cached parameters and test scores."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import layout, presets, qa
from voxanim import mech_info as I
from voxanim import mech_info_nets as M
from voxanim import study_info as S
from voxanim.next_catalog import build_study

WEIGHTS, LOSS = "weight-entropy", "logits-loss-and-nats"
EACH = [pytest.param(name, marks=needs_mnist) for name in (WEIGHTS, LOSS)]
_built = {}


def built(name):
    """``(animation, study)`` of catalogue ``name``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_info", name))
    return _built[name]


def subtitles(a):
    return [a.subtitle_for(k) for k in range(len(a.PHASES))]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.5]


# ---------------------------------------------------------------------- both
@pytest.mark.parametrize("name", EACH)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", EACH)
def test_drawn_by_this_module_and_loop_closes(name):
    a, _ = built(name)
    assert type(a) is M.ANIMATIONS[name] is I.ANIMATIONS[name]
    assert 45.0 <= a.t_done <= 61.0 and a.duration <= 66.0 and len(a.PHASES) >= 4

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:                            # every phase draws things, not only text
        assert len(a.scene((lo + hi) / 2).quads) >= 3


@pytest.mark.parametrize("name", EACH)
def test_every_frame_of_a_coarse_pass_builds(name):
    a, _ = built(name)
    for t in np.arange(0.0, a.duration, 0.37):
        assert a.scene(float(t)).texts is not None


@pytest.mark.parametrize("name", EACH)
def test_subtitles_fit_and_every_phase_says_what_it_shows(name):
    a, _ = built(name)
    for sub in subtitles(a):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert 1 <= len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    for k in range(len(a.PHASES)):                    # the kind of evidence, in every phase
        for u in (0.2, 0.6, 0.9):
            assert any(M.is_evidence(t) for t in texts(a, k, u))
        assert sum(bool(t.strip()) for t in texts(a, k, 0.6)) >= 3


@pytest.mark.parametrize("name", EACH)
def test_study_says_what_was_run(name):
    _, s = built(name)
    m = s.metadata
    assert len(s.panels) == 4 and m["dataset"] == "MNIST" and m["limitations"]
    assert m["evidence"]["measured"] and m["evidence"]["exact"]


# ---------------------------------------------------------------------- no data needed
def test_labels():
    assert M.fixed(-0.0012) == "0.00" and M.fixed(-0.5) == "−0.50" and M.fixed(3.0, 0) == "3"
    assert M.span_of([4.571, 4.573], lambda v: "%.2f" % v) == "4.57"
    assert M.span_of([0.1008, 0.1576], lambda v: "%.0f%%" % (100 * v)) == "10%–16%"
    assert M.tenths(-1128.026) == "−1,128.0"
    assert M.is_evidence(M.RUNS) and M.is_evidence(M.SEED0) and M.is_evidence(M.HELD)
    assert M.is_evidence(M.HALVES) and M.is_evidence(M.EXACT)
    assert not M.is_evidence("measured elsewhere")


def test_the_neighbour_picture_counts_pairs_where_they_fall():
    a, b = np.array([0.1, 0.1, -0.25]), np.array([0.2, 0.2, -0.25])
    tex = M.density(a, b, 0.3, M.AFTER, bins=6)
    assert tex.shape == (6, 6, 3)
    np.testing.assert_allclose(tex[0, 4], M.AFTER, atol=1e-6)        # two pairs: the darkest
    assert np.abs(tex[5, 0] - M.PAPER).sum() > 0.05                   # one pair, bottom left
    blank = np.abs(tex - np.asarray(M.PAPER)).sum(-1) < 1e-6
    assert blank.sum() == 34                                          # nothing anywhere else


# ---------------------------------------------------------------------- 174
@needs_mnist
def test_weight_numbers_on_screen_are_the_study():
    a, s = built(WEIGHTS)
    m = s.metadata
    H = m["histogram_entropy"]
    first = texts(a, 0, 0.95)
    assert "52,650 free parameters" in first                          # as the catalogue says
    assert "%s free parameters" % format(m["parameters"], ",") in first
    assert "H = −Σ p log₂ p = %.2f bits a parameter" % H["before"][0] in first
    assert "10%–16% of test digits right, loss 2.28–2.38 nats" in first
    trained = texts(a, 1, 0.95)
    assert "three seeds: 4.57 → 5.23 bits" in trained
    assert "before: %.2f bits" % H["before"][0] in trained
    assert "after: %.2f bits" % H["after"][0] in trained
    grow = m["spread"]["after"][0] / m["spread"]["before"][0]
    assert any("= %.2f bits more" % m["spread_bits"][0] in t for t in trained)
    assert any("× %.2f" % grow in t for t in trained)
    where = texts(a, 2, 0.95)
    for w in ("before", "after"):
        assert "correlation %s" % M.fixed(m["neighbour_correlation"][w][0]) in where
    assert "information %.3f bits" % m["neighbour_information"]["before"][0] in where
    assert "information %.2f bits" % m["neighbour_information"]["after"][0] in where
    assert {"correlation 0.00", "correlation 0.50", "information 0.003 bits",
            "information 0.19 bits"} <= set(where)
    assert "their weights get no gradient; 67 of 67 end" in where
    se = m["spectral_entropy"]
    directions = texts(a, 3, 0.95)
    w0 = [float(np.mean(se["W0"][w])) for w in ("before", "after")]
    assert "first layer, 784 × 64: %.2f → %.2f bits" % tuple(w0) in directions
    assert {"61 of 64", "45 of 64"} <= set(directions)
    verdict = texts(a, 4, 0.95)
    assert {"4.57", "5.23", "0.00", "0.50", "0.003", "0.19", "5.94", "5.48", "4.63", "4.50",
            "10%–16%", "96.5%–97.0%"} <= set(verdict)


@needs_mnist
def test_weight_claims_hold():
    _, s = built(WEIGHTS)
    m = s.metadata
    H, sd = m["histogram_entropy"], m["spread"]
    for k in range(len(m["seeds"])):                  # the pile's entropy rises in every seed
        assert H["after"][k] > H["before"][k] + 0.5
        rise = H["after"][k] - H["before"][k]
        assert np.isclose(m["spread_bits"][k], np.log2(sd["after"][k] / sd["before"][k]))
        assert 0.8 < m["spread_bits"][k] / rise < 1.0  # most of it is the spread
        assert m["neighbour_correlation"]["after"][k] > 0.4 > 0.01 > abs(
            m["neighbour_correlation"]["before"][k])
        assert m["neighbour_information"]["after"][k] > 30 * m["neighbour_information"][
            "before"][k]
        se = m["spectral_entropy"]
        assert se["W0"]["after"][k] < se["W0"]["before"][k] - 0.3
        assert m["accuracy"]["after"][k] > 0.96 and m["accuracy"]["before"][k] < 0.2
    P0, P1 = S.params("before"), S.params("after")
    assert S.flat(P0).size == m["parameters"] == m["weights"] + m["biases"] == 52650
    never = S.weight_runs()["never_lit"]
    assert len(never) == m["never_lit"] == m["never_lit_unchanged"]
    np.testing.assert_array_equal(P0["W0"][never], P1["W0"][never])
    lit = np.setdiff1d(np.arange(784), never)
    moved = P0["W0"][lit] != P1["W0"][lit]            # every lit pixel moves in some unit
    assert moved.any(1).all() and moved.mean() > 0.95
    # rescaled to one spread, the piles differ by what the spread does not explain
    a, _ = built(WEIGHTS)
    rise = H["after"][0] - H["before"][0]
    assert abs(a.shape_bits - (rise - m["spread_bits"][0])) < 0.02
    # the singular values drawn give the study's spectral entropies
    for k in range(3):
        for j, w in enumerate(("before", "after")):
            q = a.shares[k][j]
            assert np.isclose(-(q * np.log2(q)).sum(), m["spectral_entropy"]["W%d" % k][w][0])


# ---------------------------------------------------------------------- 175
@needs_mnist
def test_loss_numbers_on_screen_are_the_study():
    a, s = built(LOSS)
    m = s.metadata
    digit = texts(a, 0, 0.95)
    assert "softmax: p(7) = 0.9745, p(3) = 0.0251" in digit
    assert "9.51 − 5.85 = 3.66 = ln(0.9745 / 0.0251)" in digit
    assert "= %.3f nats = %.3f bits" % (m["shown_loss_nats"], m["shown_loss_nats"] / np.log(2)) \
        in digit
    assert "= 0.026 nats = 0.037 bits" in digit and "1 nat = 1 / ln 2 = 1.443 bits" in digit
    shifted = texts(a, 0, 0.41)                       # all ten logits moved: nothing changes
    assert any(t.startswith("all ten logits −") and t.endswith("still 0.9745") for t in shifted)
    tail = texts(a, 1, 0.95)
    assert {"median %.3f" % m["median_nats"], "mean %.3f" % m["mean_nats"],
            "worst 100: 42%", "the worst: a 1, called a 2", "p = 0.00005, 9.86 nats",
            "median 0.002", "mean 0.113"} <= set(tail)
    est = texts(a, 2, 0.95)
    assert {"0.113 ± 0.006 nats", "2.303 nats = 3.322 bits", "2.301 nats = 3.320 bits",
            "perplexity exp(0.113) = 1.12", "0.113 nats = 0.163 bits"} <= set(est)
    assert "%.3f ± %.3f nats" % (m["mean_nats"], m["standard_error"]) in est
    code = texts(a, 3, 0.95)
    assert {"with it: 1,627", "without the model: 33,219",
            "in bits: 1,627 with the model against 10,000 × log₂ 10 = 33,219 without"} <= set(
        code)
    assert any("= −(log-likelihood) / 10,000 = 1,128.0 / 10,000 = 0.113 nats" in t for t in code)
    temp = texts(a, 4, 0.95)
    assert {"T = 1.00", "second half: %.3f nats" % m["second_half_at_1"],
            "best on the first half: T = %.2f" % m["best_temperature"]} <= set(temp)


@needs_mnist
def test_loss_claims_hold():
    a, s = built(LOSS)
    m = s.metadata
    z, lp, yt, _prior = S.test_scores()
    loss = S.per_digit_loss(lp, yt)
    n = len(yt)
    assert n == m["n"] == 10000 and np.isclose(loss.mean(), m["mean_nats"])
    # logits are log-odds up to a shared constant
    i, (top, second) = m["shown_digit"], m["shown_top2"]
    p = np.exp(lp[i])
    assert np.isclose(z[i, top] - z[i, second], np.log(p[top] / p[second]))
    np.testing.assert_allclose(S.log_softmax(z[i] + 123.0), lp[i], atol=1e-9)
    # nats and bits, the mean, and the likelihood it is the negative log of
    assert np.isclose(m["mean_bits"], m["mean_nats"] / np.log(2))
    assert np.isclose(m["log_likelihood"], lp[np.arange(n), yt].sum())
    assert np.isclose(m["mean_nats"], -m["log_likelihood"] / n)
    assert np.isclose(m["total_bits"], n * m["mean_bits"]) and np.isclose(m["raw_bits"],
                                                                          n * np.log2(10))
    assert np.isclose(a.cum_bits[-1], m["total_bits"]) and a.upto[-1] == n
    assert m["median_nats"] < m["mean_nats"] / 10                     # a heavy tail
    assert m["mean_nats"] + 3 * m["standard_error"] < m["prior_nats"] < m["uniform_nats"]
    assert np.isclose(m["perplexity"], np.exp(m["mean_nats"]))
    # the histogram split into digits named right and wrongly adds up to the study's
    np.testing.assert_array_equal(a.counts_right + a.counts_wrong, a.counts)
    assert a.counts.sum() == n
    # temperature moves the loss and never the accuracy
    assert np.ptp(a.acc_T) == 0.0
    assert m["best_temperature"] == 1.0 and m["second_half_at_best"] == m["second_half_at_1"]
    assert min(m["loss_second_half"]) >= m["second_half_at_1"] - 0.005

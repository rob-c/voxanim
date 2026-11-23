"""Animation 177 (voxanim.mech_bridge_loss): the layout is clean, the loop closes, every
phase says what kind of evidence it shows, the first phase names where the bridge comes
from and the last where it goes, and the numbers on screen are the study's."""

from __future__ import annotations

import numpy as np
from conftest import needs_mnist

from voxanim import layout, presets, qa
from voxanim import mech_bridge_loss as M
from voxanim.losschart import sup
from voxanim.next_catalog import build_study, drawing

NAME = "every-loss-is-a-likelihood"
_built = {}


def built():
    """``(animation, study)``, built once."""
    if NAME not in _built:
        _built[NAME] = (presets.get(NAME).build(), build_study("study_bridge_loss", NAME))
    return _built[NAME]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.5]


# ---------------------------------------------------------------------- no data needed
def test_labels():
    assert M.signed(-83.108) == "−83.11" and M.signed(5, 0) == "5"
    assert M.thousands(-2747.33) == "−2,747.3" and M.thousands(-212.47) == "−212.5"
    for tag in (M.BINNED, M.SAMPLED, M.DIGITS, M.TEN, M.TWO, M.TABLE):
        assert M.is_evidence(tag)
    assert not M.is_evidence("measured elsewhere")
    assert drawing(NAME) is M.EveryLossIsALikelihood is M.ANIMATIONS[NAME]


# ---------------------------------------------------------------------- the animation
@needs_mnist
def test_layout_is_clean():
    assert qa.check(NAME, samples=24, verbose=False) == 0


@needs_mnist
def test_loop_closes_and_every_phase_draws():
    a, _ = built()
    assert type(a) is M.EveryLossIsALikelihood
    assert 45.0 <= a.duration <= 66.0 and len(a.PHASES) >= 4

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:
        assert len(a.scene((lo + hi) / 2).quads) >= 3


@needs_mnist
def test_every_frame_of_a_coarse_pass_builds():
    a, _ = built()
    for t in np.arange(0.0, a.duration, 0.41):
        assert a.scene(float(t)).texts is not None


@needs_mnist
def test_subtitles_fit_and_every_phase_says_what_it_shows():
    a, _ = built()
    for k in range(len(a.PHASES)):
        lines = layout.wrap(a.subtitle_for(k), layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert 1 <= len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
        for u in (0.2, 0.6, 0.9):
            assert any(M.is_evidence(t) for t in texts(a, k, u))
        assert sum(bool(t.strip()) for t in texts(a, k, 0.5)) >= 3


@needs_mnist
def test_it_is_a_bridge():
    a, _ = built()
    assert "from: Chi-square as a landscape · The likelihood curve" in texts(a, 0, 0.5)
    assert "from: Errors from the likelihood" in texts(a, 1, 0.5)
    assert "next: the same cross-entropy trains every network on this page" in texts(
        a, len(a.PHASES) - 1, 0.95)
    assert a.subtitle_for(0).startswith("from Part 1")
    assert "Next: the same cross-entropy" in a.subtitle_for(len(a.PHASES) - 1)


@needs_mnist
def test_the_numbers_on_screen_are_the_study():
    a, s = built()
    m = s.metadata
    g, one, ten, two = m["gauss"], m["one_sigma"], m["softmax"], m["logistic"]
    first = texts(a, 0, 0.95)
    assert "lowest chi-square: b = %.3f" % g["b"] in first
    assert "highest likelihood: b = %.3f" % g["b"] in first
    assert "−2 ln L = χ² + Σ ln 2πσᵢ² = χ² − %.2f" % -g["constant"] in first
    assert "%d training digits in %d bins: mean ± its error" % (g["digits_used"], g["points"]) \
        in first
    width = texts(a, 1, 0.95)
    assert "error 1 / √%s = ±%.3f, where −ln L is ½ up" % (
        format(int(round(g["curvature"])), ","), g["sigma_b"]) in width
    assert "the slopes scatter by %.3f ± %.3f" % (g["spread"], g["spread_error"]) in width
    assert "green: the curvature's errors, mean %.3f" % g["mean_sigma_b"] in width
    assert "120 disjoint samples of 500 digits:" in width
    sq = texts(a, 2, 0.95)
    assert "least squares, every digit: b = %.3f" % one["b"] in sq
    assert "chi-square, binned: b = %.3f ± %.3f" % (g["b"], g["sigma_b"]) in sq
    assert "%.6f on 10,000 held out" % one["mse_test"] in sq
    assert "the best σ is √MSE = %.4f," % one["sigma"] in sq
    assert "and then χ² = N = 500" in sq and "Σ rᵢ² / σ² = χ² = N · MSE / σ²" in sq
    cat = texts(a, 3, 0.95)
    assert "−ln L / 10,000 = %.3f nats" % ten["cross_entropy"] in cat
    assert "ln L = Σ ln p = %s" % M.thousands(ten["log_likelihood"]) in cat
    assert "Π p(right digit) ≈ 10%s" % sup(int(round(ten["log10_product"]))) in cat
    assert "%.1f%% of the test digits right" % (100 * ten["accuracy"]) in cat
    assert "p(%d) = %.4f" % (ten["shown_label"],
                             ten["shown_probabilities"][ten["shown_label"]]) in cat
    assert "starts at ln 10 = 2.303: every digit 1 in 10" in cat
    ber = texts(a, 4, 0.95)
    assert "−ln L / 1,991 = %.3f nats" % two["bce"] in ber
    assert "ln L = %s over 1,991 digits" % M.thousands(two["log_likelihood"]) in ber
    assert "%.1f%% right; a coin: ln 2 = 0.693" % (100 * two["accuracy"]) in ber
    assert "σ(z₉ − z₄): %.3f nats, %.1f%% right" % (two["ten_class_bce"],
                                                    100 * two["ten_class_accuracy"]) in ber
    assert {"the 982 held-out 4s", "the 1,009 held-out 9s"} <= set(ber)
    table = texts(a, 5, 0.95)
    assert {"%.1f, %d d.o.f." % (g["chi2_min"], g["dof"]),
            "MSE %.6f held out" % one["mse_test"], "%.3f nats held out" % two["bce"],
            "%.3f nats held out" % ten["cross_entropy"], "binary cross-entropy",
            "cross-entropy", "squared error", "chi-square"} <= set(table)


@needs_mnist
def test_what_is_drawn_matches_the_study():
    a, s = built()
    g = s.metadata["gauss"]
    assert np.isclose(a.lik.max(), 1.0) and int(np.argmax(a.lik)) == g["argmax"]
    assert np.isclose(a.dnll.min(), 0.0)
    # the held-out logits' histograms count every 4 and 9
    assert a.z4.sum() + a.z9.sum() == s.metadata["logistic"]["n_test"]
    assert a.z9.sum() == s.metadata["logistic"]["nines"]
    assert a.templates.shape == (10, 28, 28)


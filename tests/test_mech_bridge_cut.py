"""Animation 178 (voxanim.mech_bridge_cut) and its experiment (voxanim.study_bridge_cut): the
layout is clean, the loop closes, every phase says what kind of evidence it shows, the
bridge names where it comes from and where it leads, and the numbers on screen are the
study's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import layout, presets, qa
from voxanim import mech_bridge_cut as M
from voxanim import study_bridge_cut as S
from voxanim.next_catalog import build_study, drawing

NAME = S.NAME
_built = {}


def built():
    """``(animation, study)``, built once."""
    if NAME not in _built:
        _built[NAME] = (presets.get(NAME).build(), build_study("study_bridge_cut", NAME))
    return _built[NAME]


def subtitles(a):
    return [a.subtitle_for(k) for k in range(len(a.PHASES))]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.5]


# ---------------------------------------------------------------------- no data needed
def test_labels():
    assert M.px(4.8803) == "4.88"
    assert M.is_evidence(M.exact(8000)) and M.is_evidence(M.grown(8000))
    assert M.is_evidence(M.held(5782)) and not M.is_evidence("measured elsewhere")
    assert M.exact(8000) == "exact arithmetic on 8,000 training events"
    f, g = dict(cut=4.899), dict(cut=4.8803)
    assert M.agree(f, g) == "picks 4.88, 0.02 pixels from s / √(s + b)"
    assert M.agree(g, g) == "picks the same cut, 4.88 pixels"
    r = dict(same=True, by_fom=dict(cut=4.644), by_gini=dict(cut=4.644))
    assert M.moved(r) == "both move to 4.64" and M.picks(r) == "both pick 4.64"
    r = dict(same=False, by_fom=dict(cut=4.644), by_gini=dict(cut=4.70))
    assert M.moved(r) == "s / √(s + b) moves to 4.64 and Gini to 4.70"


def test_a_curve_is_cropped_where_it_leaves_the_chart():
    xs, ys = np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 0.2, 0.6, 0.9])
    cx, cy = M.crop(xs, ys, 0.4)
    np.testing.assert_allclose(cx, [0.0, 1.0, 1.5])
    np.testing.assert_allclose(cy, [0.0, 0.2, 0.4])
    cx, cy = M.crop(xs, ys, 1.0)
    np.testing.assert_array_equal(cx, xs)


# ---------------------------------------------------------------------- the animation
@needs_mnist
def test_layout_is_clean():
    assert qa.check(NAME, samples=24, verbose=False) == 0


@needs_mnist
def test_drawn_by_this_module_and_loop_closes():
    a, _ = built()
    assert type(a) is M.ANIMATIONS[NAME] is drawing(NAME)
    assert 45.0 <= a.t_done <= 61.0 and a.duration <= 66.0 and len(a.PHASES) >= 4

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:                            # every phase draws things, not only text
        assert len(a.scene((lo + hi) / 2).quads) >= 3


@needs_mnist
def test_every_frame_of_a_coarse_pass_builds():
    a, _ = built()
    for t in np.arange(0.0, a.duration, 0.37):
        assert a.scene(float(t)).texts is not None


@needs_mnist
def test_subtitles_fit_and_every_phase_says_what_it_shows():
    a, _ = built()
    for sub in subtitles(a):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert 1 <= len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    for k in range(len(a.PHASES)):                    # the kind of evidence, in every phase
        for u in (0.2, 0.6, 0.9):
            assert any(M.is_evidence(t) for t in texts(a, k, u))
        assert sum(bool(t.strip()) for t in texts(a, k, 0.6)) >= 3


@needs_mnist
def test_it_is_a_bridge():
    a, _ = built()
    assert M.FROM_CUT in texts(a, 0, 0.5)             # where it comes from, on screen
    assert M.FROM_TEST in texts(a, 3, 0.5)
    assert M.NEXT in texts(a, len(a.PHASES) - 1, 0.95)
    assert "Moving a cut" in a.subtitle_for(0)


@needs_mnist
def test_study_says_what_was_run():
    _, s = built()
    m = s.metadata
    assert len(s.panels) == 4 and m["dataset"] == "MNIST" and m["limitations"]
    assert m["evidence"]["measured"] and m["evidence"]["exact"]
    assert "s / sqrt(s + b)" in m["figure_of_merit"]


@needs_mnist
def test_numbers_on_screen_are_the_study():
    a, s = built()
    m = s.metadata
    f, g = m["one_feature"]["by_fom"], m["one_feature"]["by_gini"]
    cut = texts(a, 0, 0.95)
    assert "best: %.2f pixels, s / √(s + b) = %.1f" % (f["cut"], f["fom"]) in cut
    assert "best: 4.90 pixels, s / √(s + b) = 54.3" in cut
    assert "s = %s 4s kept of 4,000" % format(int(f["s"]), ",") in cut
    assert "b = %d 9s kept of 4,000" % f["b"] in cut
    assert {"s = 3,537 4s kept of 4,000", "b = 707 9s kept of 4,000"} <= set(cut)
    gini = texts(a, 1, 0.95)
    assert "s / √(s + b) picks %.2f pixels" % f["cut"] in gini
    assert "Gini gain picks %.2f pixels" % g["cut"] in gini
    assert "held out: %.0f%% and %.0f%% of 4s kept" % (100 * f["held"]["efficiency"],
                                                     100 * g["held"]["efficiency"]) in gini
    left, right = m["root"]["left"], m["root"]["right"]
    assert "%d | %d" % tuple(left) in gini and "%d | %d" % tuple(right) in gini
    assert {"3480 | 632", "520 | 3368", "spread ≤ 4.88?"} <= set(gini)
    assert "4s rare, 50 per 1,000 9s: both pick %.2f" % m["rare"]["by_fom"]["cut"] in gini
    boxes = texts(a, 2, 0.95)
    best = m["depths"][m["best_depth"] - 1]
    assert "depth %d: %d boxes" % (m["best_depth"], best["leaves"]) in boxes
    assert "depth 1: 2 boxes" in texts(a, 2, 0.1)
    roc = texts(a, 3, 0.95)
    assert "9s kept where %.0f%% of 4s are" % (100 * m["stump_held"]["efficiency"]) in roc
    assert "the stump: %.1f%%" % (100 * m["stump_held"]["background"]) in roc
    assert "tree, depth %d: %.1f%%" % (m["best_depth"], 100 * best["background_at_stump"]) in roc
    assert "likelihood ratio: %.1f%%" % (100 * m["ratio"]["background_at_stump"]) in roc
    assert "spread alone, every cut: %.1f%%" % (100 * m["cut_background_at_stump"]) in roc
    deep = texts(a, 4, 0.95)
    last = m["depths"][-1]
    assert "held out, best at depth %d: %.3f" % (m["best_depth"], m["best_area"]) in deep
    assert "%.3f held out, %.3f training" % (last["held_area"], last["train_area"]) in deep
    assert "the cut alone: %.3f" % m["cut_area"] in deep
    assert "grown to the end, depth %d:" % m["deepest"] in deep


@needs_mnist
def test_what_is_drawn_matches_the_study():
    a, s = built()
    m = s.metadata
    # the score curves peak at the chosen cuts
    assert a.cuts_x[int(np.argmax(a.fom))] == pytest.approx(m["one_feature"]["by_fom"]["cut"])
    assert a.cuts_x[int(np.argmax(a.gain))] == pytest.approx(m["one_feature"]["by_gini"]["cut"])
    # the histograms hold the training events drawn inside the plane
    assert a.h_sig.sum() <= m["train_signal"] and a.h_sig.sum() > 0.99 * m["train_signal"]
    # the boxes at each depth are the tree's leaves there, and cover the plane
    for d, boxes in a.boxes.items():
        assert len(boxes) == m["depths"][d - 1]["leaves"]
        area = ((boxes[:, 2] - boxes[:, 1]) * (boxes[:, 4] - boxes[:, 3])).sum()
        (x0, x1), (y0, y1) = S.T.PLANE
        assert area == pytest.approx((x1 - x0) * (y1 - y0))
    # one ROC curve per name on the chart
    assert len(a.curves) == len(a._roc_names())

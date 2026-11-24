"""Animations 275-276 (voxanim.mech_blocks_zh): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are short,
and the numbers and code lines on screen are the study's and the script's. No dataset is
needed: the data are made up by the script."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_zh as M
from voxanim.next_catalog import ITEMS, build_study

NAMES = ("event-weights", "scale-factors")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_zh", name))
    return _built[name]


def at(a, phase, u):
    lo, hi = a.spans[phase]
    return lo + (hi - lo) * u


def texts(a, phase, u=0.6, kind=None):
    return [x.s for x in a.scene(at(a, phase, u)).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


def picture(a, t):
    """The texts in the picture (left of the code panel, under the heading)."""
    return [x.s for x in a.scene(float(t)).texts if x.opacity > 0.5 and x.x < B.PANEL[0]
            and x.y > B.HEAD_Y + 0.03]


# ---------------------------------------------------------------------- no drawing needed
def test_the_titles_and_the_chain_are_the_catalogues():
    from voxanim.mech_phys_apps import OvertrainingCheck

    titles = {i["name"]: i["title"] for i in ITEMS}
    classes = [M.ANIMATIONS[n] for n in NAMES]
    for name, cls in zip(NAMES, classes, strict=True):
        assert titles[name] == cls.TITLE
    assert M.EventWeights.LAST == "The overtraining check" == OvertrainingCheck.TITLE
    for before, after in pairwise(classes):
        assert before.NEXT == after.TITLE and after.LAST == before.TITLE
    assert M.ScaleFactors.NEXT == titles["unfolding"] == "Unfolding: running the smearing " \
                                                         "backwards"
    assert {i["section"] for i in ITEMS if i["name"] in NAMES} == {
        "p. From simulation to a result (4)"}


# ---------------------------------------------------------------------- every one of them
@pytest.mark.parametrize("name", NAMES)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", NAMES)
def test_drawn_by_this_module_and_loop_closes(name):
    a, _ = built(name)
    assert type(a) is M.ANIMATIONS[name] and isinstance(a, B.Blocks)
    assert 45.0 <= a.t_done <= 61.5 and a.duration <= 66.0 and len(a.PHASES) >= 4

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:
        assert len(a.scene((lo + hi) / 2).quads) >= 3
    for t in np.arange(0.0, a.duration, 0.37):           # a coarse pass: every frame builds
        assert a.scene(float(t)).texts is not None


@pytest.mark.parametrize("name", NAMES)
def test_nothing_blanks_between_phases(name):
    a, _ = built(name)
    for k, (_lo, hi) in enumerate(a.spans[:-1]):
        sc = a.scene(hi)                                   # the content's dip
        shown = [x.s for x in sc.texts if x.opacity > 0.5]
        if a.evidence_for(k) == a.evidence_for(k + 1):
            assert a.evidence_for(k) in shown
        if a.code_for(k).lines() == a.code_for(k + 1).lines():
            assert any(x.kind == "mono" and x.opacity > 0.5 for x in sc.texts)
        assert len(sc.quads) >= 6
        assert sum(x.opacity > 0.5 and bool(x.s.strip()) for x in sc.texts) >= 8
        # the picture itself is never empty: a chart's ticks, the samples or the counts stay
        assert len(picture(a, hi)) >= 3, (name, k)


@pytest.mark.parametrize("name", NAMES)
def test_subtitles_fit_and_every_phase_says_what_it_shows(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        sub = a.subtitle_for(k)
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert 1 <= len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
        for u in (0.2, 0.6, 0.9):
            assert any(B.is_evidence(t) for t in texts(a, k, u))
        assert sum(bool(t.strip()) for t in texts(a, k, 0.6)) >= 3
        tag = a.evidence_for(k)
        assert layout.text_width(tag, B.SIZE) <= 0.95 - B.PANEL[0]
        assert tag.startswith(("measured on made-up data: ", "exact arithmetic: "))
    said = " ".join(a.subtitle_for(k) for k in range(2)).lower()
    assert "made-up" in said or "made up" in said        # the made-up data, said on screen


@pytest.mark.parametrize("name", NAMES)
def test_headings_are_short_and_start_and_end_the_same_way(name):
    a, _ = built(name)
    for t in np.arange(0.0, a.duration, 0.5):
        h = a.heading_at(float(t))
        assert B.words(h) <= 12, h
        assert layout.text_width(h, B.HEAD) <= B.PANEL[0] - B.HEAD_X - 0.01, h
    first = a.heading_at(1.0)
    assert first.startswith("last time: ")
    assert a.heading_at(a.t_done - 1.0).startswith("next: ")
    assert first in texts(a, 0, 0.1)


@pytest.mark.parametrize("name", NAMES)
def test_the_code_on_screen_is_the_scripts(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        shown = mono(a, k, 0.6)
        assert shown[:len(code.lines())] == list(code.lines())
        assert all(t.startswith(("→ ", "  ")) for t in shown[len(code.lines()):])
        lines = [s for n in code.names() for s in B.code_lines(n)]
        assert all(s in lines for s in code.lines())
        assert {B.code_file(n) for n in code.names()} == {
            "examples/weights_and_scale_factors.py"}
        assert all(n.startswith("ws-") for n in code.names())


def test_only_the_loss_is_a_station_of_the_training_loop():
    a, _ = built("event-weights")
    assert [a.station_for(k) for k in range(len(a.PHASES))] == [None, None, None, "loss",
                                                                None, None]
    assert "you are here" in texts(a, 3, 0.5)
    b, _ = built("scale-factors")
    assert all(b.station_for(k) is None for k in range(len(b.PHASES)))
    assert "you are here" not in texts(b, 2, 0.5)


# ---------------------------------------------------------------------- 275
def test_event_weights_numbers_on_screen():
    a, s = built("event-weights")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"signal", "bkg 1", "bkg 2", "N made", "20,000", "50,000", "10,000",
            "80,000 simulated events", "events per bin of x"} <= set(one)
    assert '    "signal": (50.0, 20000),' in mono(a, 0, 0.9)
    two = texts(a, 1, 0.9)
    assert {"σ (fb)", "w", "50", "200", "120", "0.25", "0.4", "1.2", "L = 100 fb⁻¹",
            "signal: 50 × 100 ÷ 20,000 = 0.25"} <= set(two)
    assert a.heading_for(1, 0.5) == "a weight: w = σ × L ÷ N"
    three = texts(a, 2, 0.95)
    assert {"Σw", "5,000", "20,000", "12,000", "37,000 expected in all"} <= set(three)
    assert {"→ signal 5000", "  bkg 1 20000", "  bkg 2 12000"} <= set(mono(a, 2, 0.95))
    assert "13.5%" in a.subtitle_for(2) and "25.0%" in a.subtitle_for(2)
    four = texts(a, 3, 0.95)
    assert {"no weights: mean score 0.250", "weights: mean score 0.135", "score, 0 to 1",
            "= signal's share: 25% made, 13.5% expected"} <= set(four)
    assert "loss = (w * ce).sum() / w.sum()" in mono(a, 3)
    five = texts(a, 4, 0.95)
    assert {"cut", "cut: x > 0.915", m["eff_counted_row"], m["eff_weighted_row"]} <= set(five)
    assert {"→ 0.01795", "  0.0351125"} <= set(mono(a, 4, 0.95))
    six = texts(a, 5, 0.95)
    assert {"8,000 × (+2), 2,000 × (−2)", "the outline: the sum, 12,000 as before",
            "N_eff = (Σw)² ÷ Σw² = 3,600 of 10,000"} <= set(six)
    assert "next: Simulation is not data: scale factors" in six
    assert "3,600" in a.subtitle_for(5) and "negative" in a.subtitle_for(5)


def test_the_bars_follow_the_weights():
    a, s = built("event-weights")
    m = s.metadata
    a._t = at(a, 0, 0.05)
    assert all(np.allclose(h, 0) for h in a.heights())             # nothing grown yet
    a._t = at(a, 1, 0.5)
    np.testing.assert_allclose(a.heights(), m["counts"])           # counted
    a._t = at(a, 3, 0.5)
    np.testing.assert_allclose(a.heights(), m["weighted"])         # weighted
    assert a.y_range() == a.RANGES[1]
    a._t = at(a, 0, 0.5)
    assert a.y_range() == a.RANGES[0]
    a._t = at(a, 5, 0.9)
    assert a.y_range() == a.RANGES[2]
    lo, hi = a.RANGES[2]
    assert lo < min(m["neg"]["hist_neg"]) and max(m["neg"]["hist_pos"]) < hi


# ---------------------------------------------------------------------- 276
def test_scale_factors_numbers_on_screen():
    a, s = built("scale-factors")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"cut", "cut: x > 0.915", "keeps 16,000 of 20,000 simulated: 80%"} <= set(one)
    assert 'cut = np.quantile(x["signal"], 0.2)' in mono(a, 0)
    two = texts(a, 1, 0.9)
    assert {"simulation: 20,000 probes", "data: 5,000 probes",
            "made up: data spread 0.13, not 0.10", "probes, share per bin of x"} <= set(two)
    three = texts(a, 2, 0.9)
    assert {"simulation: 16,106 ÷ 20,000 = 0.805", "data: 3,724 ÷ 5,000 = 0.745"} <= set(three)
    four = texts(a, 3, 0.9)
    assert {"scale factor: 0.745 ÷ 0.805 = 0.925", "± 0.008: the statistical error"} <= set(
        four)
    assert "sf = eff_data / eff_mc" in mono(a, 3)
    five = texts(a, 4, 0.9)
    assert {"5,124", "4,739 ± 69", "simulation", "data", "events above the cut",
            "simulation − data = 385: 4.6 × the error"} <= set(five)
    six = texts(a, 5, 0.9)
    assert {"4,823 ± 57", "was 5,124", "4,739 ± 69",
            "signal 4,000 × 0.925 = 3,699; background kept"} <= set(six)
    assert "corrected = sf * sig + bkg" in mono(a, 5)
    assert "next: Unfolding: running the smearing backwards" in six
    assert m["corrected_text"] in a.subtitle_for(5) and "not the shape" in a.subtitle_for(5)
    # the efficiencies and the scale factor stay on screen to the end
    end = texts(a, 5, 0.99)
    assert {"simulation: 16,106 ÷ 20,000 = 0.805", "data: 3,724 ÷ 5,000 = 0.745",
            "scale factor: 0.745 ÷ 0.805 = 0.925"} <= set(end)


def test_the_simulations_count_comes_down_to_the_corrected_one():
    a, s = built("scale-factors")
    sr = s.metadata["sr"]
    ch = a.sr_chart()
    lo, hi = ch.y_range
    for v in (sr["predicted"], sr["corrected"], sr["observed"]):
        assert lo < v - 100 and v + 100 < hi
    a._t = at(a, 5, 0.1)
    assert a.prog(5, *a.MOVE) == 0.0
    a._t = at(a, 5, 0.6)
    assert a.prog(5, *a.MOVE) == 1.0

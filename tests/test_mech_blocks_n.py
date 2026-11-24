"""Animations 224-226 (voxanim.mech_blocks_n): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the program's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_n as M
from voxanim import study_padding as pad
from voxanim.next_catalog import build_study

NAMES = ("padding-maps-shrink", "padding-a-ring", "padding-same-and-valid")
STAYS = {"padding-maps-shrink": "28 × 28", "padding-a-ring": "30 × 30",
         "padding-same-and-valid": "padding"}       # a text of the picture that never dips
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_n", name))
    return _built[name]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.5]


def panel(a, phase, u=0.6):
    """The mono text in the code panel: the region's lines, then the output slot."""
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and x.kind == "mono" and x.x >= B.PANEL[0]]


# ---------------------------------------------------------------------- every one of them
@needs_mnist
@pytest.mark.parametrize("name", NAMES)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@needs_mnist
@pytest.mark.parametrize("name", NAMES)
def test_drawn_by_this_module_and_loop_closes(name):
    a, _ = built(name)
    assert type(a) is M.ANIMATIONS[name] and isinstance(a, B.Blocks)
    assert 45.0 <= a.t_done <= 61.0 and a.duration <= 66.0 and len(a.PHASES) >= 4

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:
        assert len(a.scene((lo + hi) / 2).quads) >= 3
    for t in np.arange(0.0, a.duration, 0.37):           # a coarse pass: every frame builds
        assert a.scene(float(t)).texts is not None


@needs_mnist
@pytest.mark.parametrize("name", NAMES)
def test_nothing_blanks_between_phases(name):
    a, _ = built(name)
    for k, (_lo, hi) in enumerate(a.spans[:-1]):
        sc = a.scene(hi)                                   # the content's dip
        shown = [x.s for x in sc.texts if x.opacity > 0.5]
        if a.station_for(k) == a.station_for(k + 1):
            assert "you are here" in shown
        if a.evidence_for(k) == a.evidence_for(k + 1):
            assert a.evidence_for(k) in shown
        if a.code_for(k).lines() == a.code_for(k + 1).lines():
            assert any(x.kind == "mono" and x.opacity > 0.5 for x in sc.texts)
        assert len(sc.quads) >= 8
        assert STAYS[name] in shown


@needs_mnist
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
        assert layout.text_width(a.evidence_for(k), B.SIZE) <= 0.95 - B.PANEL[0]


@needs_mnist
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


@needs_mnist
@pytest.mark.parametrize("name", NAMES)
def test_the_code_on_screen_is_the_programs(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        lines = list(code.lines())
        shown = panel(a, k, 0.6)
        assert shown[:len(lines)] == lines == [s for r in code.names()
                                               for s in B.code_lines(r)]
        assert B.code_file(code.names()[0]) == "examples/padding.py"
        out = shown[len(lines):]
        assert not out or out[0].startswith("→ ")


def test_the_series_runs_on():
    shrink, ring, same = (M.ANIMATIONS[n] for n in NAMES)
    assert shrink.LAST == "CNN mistakes"            # the end of the CNN's series
    assert shrink.NEXT == ring.TITLE and ring.LAST == shrink.TITLE
    assert ring.NEXT == same.TITLE and same.LAST == ring.TITLE
    assert same.NEXT == "What goes in the border"


# ---------------------------------------------------------------------- the numbers shown
@needs_mnist
def test_maps_shrink_shows_the_places_and_the_coverage():
    a, s = built("padding-maps-shrink")
    m = s.metadata
    assert "3 × 3" in texts(a, 0, 0.9)
    across = texts(a, 1, 0.45)
    assert "26" in across and m["across_sum"] in across and "places across" in across
    end = texts(a, 1, 0.95)
    assert "676" in end and "26 × 26 = 676" in end and "maps of 26 × 26" in end
    assert "→ [64, 1, 28, 28] → [64, 8, 26, 26]" in panel(a, 1, 0.95)
    # the coverage map fills as the stamp sweeps, and ends as the study's map
    assert a.sweep(1.0) == 676 and a.sweep(0.0) == 0
    np.testing.assert_array_equal(a.coverage(a.sweep(1.0)), pad.coverage())
    assert "times each pixel is read" in texts(a, 2, 0.5)
    shown = texts(a, 3, 0.99)
    for name, n in (("corner", "1"), ("edge", "3"), ("middle", "9")):
        assert name in shown and n in shown


@needs_mnist
def test_a_ring_shows_the_ring_the_paper_and_the_coverage():
    a, _s = built("padding-a-ring")
    assert "30 × 30" in texts(a, 0, 0.95) and "1 + 28 + 1 = 30" in texts(a, 0, 0.95)
    zoom = texts(a, 1, 0.8)
    assert zoom.count("0.00") == 5 and zoom.count("−0.42") >= 4
    assert "(0 − 0.1307) ÷ 0.3081" in zoom and "0: the ring" in zoom
    assert "−0.42: the paper" in zoom
    end = texts(a, 2, 0.95)
    assert "784" in end and "28 × 28 = 784" in end
    assert "30 − 2 = 28" in texts(a, 2, 0.48)
    assert "→ [64, 1, 28, 28] → [64, 8, 28, 28]" in panel(a, 2, 0.95)
    np.testing.assert_array_equal(a.coverage(a.sweep(1.0)), pad.coverage(pad=1))
    shown = texts(a, 4, 0.99)
    for name, n in (("corner", "4"), ("edge", "6"), ("middle", "9")):
        assert name in shown and n in shown


@needs_mnist
def test_same_and_valid_fills_the_table_row_by_row():
    a, s = built("padding-same-and-valid")
    rows = s.metadata["rows"]
    for k, r in enumerate(rows):
        shown = texts(a, k, 0.93)
        for v in (r["k"], r["pad"], r["same"], r["valid"]):
            assert str(v) in shown
        assert r["valid_sum"] in shown
        assert r["pad_sum"] in texts(a, k, 0.42)
        out = panel(a, k, 0.93)
        assert "→ " + r["printed"][0] in out and "  " + r["printed"][1] in out
        if k < 2:                                  # the next row is not there yet
            assert str(rows[k + 1]["valid"]) not in shown
    rule = texts(a, 3, 0.9)
    assert "(k − 1) ÷ 2" in rule and "28 − (k − 1)" in rule
    assert a.heading(3, 0.2) == "same pads (k − 1) ÷ 2"
    assert a.heading(3, 0.55) == "valid, the default, loses k − 1"

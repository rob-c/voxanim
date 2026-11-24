"""Animations 232-233 (voxanim.mech_blocks_q): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_text

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_q as M
from voxanim.next_catalog import build_study

NAMES = ("padding-masks", "padding-sort-by-length")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_q", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


# ---------------------------------------------------------------------- every one of them
@needs_text
@pytest.mark.parametrize("name", NAMES)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@needs_text
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


@needs_text
@pytest.mark.parametrize("name", NAMES)
def test_nothing_blanks_between_phases(name):
    a, _ = built(name)
    for k, (_lo, hi) in enumerate(a.spans[:-1]):
        sc = a.scene(hi)                                   # the content's dip
        shown = [x.s for x in sc.texts if x.opacity > 0.5]
        if a.station_for(k) == a.station_for(k + 1) is not None:
            assert "you are here" in shown
        if a.evidence_for(k) == a.evidence_for(k + 1):     # a new tag fades over
            assert a.evidence_for(k) in shown
        if a.code_for(k).lines() == a.code_for(k + 1).lines():   # new lines fade over
            assert any(x.kind == "mono" and x.opacity > 0.5 for x in sc.texts)
        # the furniture (panel, five stations) and what the picture keeps
        assert len(sc.quads) >= 8
        assert sum(x.opacity > 0.5 and bool(x.s.strip()) for x in sc.texts) >= 8


@needs_text
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
        assert a.evidence_for(k).startswith("exact arithmetic: ")   # counted, not measured


@needs_text
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


@needs_text
@pytest.mark.parametrize("name", NAMES)
def test_the_code_on_screen_is_the_scripts(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        shown = [s for s in mono(a, k, 0.6) if not s.startswith("→")]
        assert shown == list(code.lines())
        lines = [s for n in code.names() for s in B.code_lines(n)]
        assert all(s in lines for s in shown)
        assert {B.code_file(n) for n in code.names()} == {"examples/padding.py"}


@needs_text
def test_the_titles_and_the_chain_are_the_catalogues():
    from voxanim.next_catalog import ITEMS
    titles = {i["name"]: i["title"] for i in ITEMS}
    for name in NAMES:
        assert titles[name] == M.ANIMATIONS[name].TITLE
    assert titles["padding-lines-of-text"] == M.Masks.LAST
    assert M.Masks.NEXT == titles["padding-sort-by-length"] == M.SortByLength.TITLE
    assert titles["padding-masks"] == M.SortByLength.LAST
    assert M.SortByLength.NEXT == "Automatic differentiation"


# ---------------------------------------------------------------------- 232
@needs_text
def test_masks():
    a, s = built("padding-masks")
    m = s.metadata
    for k in range(5):                                   # the block stays all through
        assert "[8, 50]: 163 real, 237 pads" in texts(a, k, 0.6)
    one = texts(a, 0, 0.9)
    assert {'vecs[2], "All:": 50 vectors of 16', "4 real, random", "46 pads: all 0"} <= set(one)
    assert a.heading(0, 0.6) == "padding_idx=0: each pad's vector is 0"
    assert "                     padding_idx=0)" in mono(a, 0, 0.9)
    two = texts(a, 1, 0.9)
    assert {"plain mean: the sum ÷ 50", "true mean: the sum ÷ 4",
            "4 ÷ 50 = 0.08 of the true mean"} <= set(two)
    assert "4 ÷ 50 = 0.08" in a.subtitle_for(1) and "plain = vecs.mean(1)" in mono(a, 1)
    plain = texts(a, 2, 0.4)                             # the plain means, then the masked
    assert "plain mean ÷ true mean" in plain
    assert [t for t in plain if t in m["ratio_texts"]] == m["ratio_texts"]
    masked = texts(a, 2, 0.9)
    assert "masked mean ÷ true mean" in masked and masked.count("1.00") == 8
    assert "mean = total / keep.sum(1)" in mono(a, 2, 0.9)
    four = texts(a, 3, 0.9)
    assert {"w[:, 0]: each first character's weights", "w[1, 0]: 45 weights, adding to 1.00",
            "its 5 pads: exactly 0", "the weights are random; the zeros are not"} <= set(four)
    assert "              key_padding_mask=~mask)" in mono(a, 3, 0.9)
    assert "random" in a.subtitle_for(3) and "random" in a.evidence_for(3)
    five = texts(a, 4, 0.9)
    assert {"163 counted", "237 skipped", "the loss: ÷ 163, not ÷ 400",
            "next: Sort by length, pad less"} <= set(five)
    assert "    ignore_index=0)" in mono(a, 4, 0.9)
    assert a.station_for(4) == "loss"


# ---------------------------------------------------------------------- 233
@needs_text
def test_sort_by_length():
    a, s = built("padding-sort-by-length")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"batch 1 · longest 59", "45% pads", "batch 2 · longest 54", "32% pads",
            "padding, all 513 batches", "file order", "38.0%"} <= set(one)
    assert "38.7%" not in one and "0.16%" not in one      # each order's share once counted
    assert "lines.sort(key=len)" not in mono(a, 0)        # file order: no sort yet
    assert "1,734,987" in a.subtitle_for(0)
    two = texts(a, 1, 0.9)
    assert {"batch 1 · longest 51", "batch 2 · longest 50", "38.0%", "38.7%"} <= set(two)
    three = texts(a, 2, 0.9)
    assert {"batch 172 · longest 29", "2% pads", "batch 343 · longest 43", "0% pads",
            "38.0%", "38.7%", "0.16%"} <= set(three)
    assert mono(a, 2, 0.9)[0] == "lines.sort(key=len)"
    assert "1,717 pads in 1,077,111 cells, 0.16%" in a.subtitle_for(2)
    four = texts(a, 3, 0.9)
    assert {"the same batches, shuffled", "the longest line in each", "padding: still 0.16%",
            "next: Automatic differentiation"} <= set(four)
    shown = [int(t) for t in four if t.isdigit()]
    assert sorted(shown) == m["mini_longest"]
    # shuffled as whole batches: the order on screen is the study's shuffle
    lo, hi = a.spans[3]
    sc = a.scene(lo + 0.9 * (hi - lo))
    left_to_right = [x.s for x in sorted((x for x in sc.texts
                                          if x.opacity > 0.5 and x.s.isdigit()),
                                         key=lambda x: x.x)]
    slots = [None] * len(m["mini"])
    for i, j in enumerate(m["mini_order"]):              # batch i moves to slot j
        slots[j] = str(m["mini_longest"][i])
    assert left_to_right == slots
    start = texts(a, 3, 0.3)                             # before: short to long
    assert [t for t in start if t.isdigit()] == [str(v) for v in m["mini_longest"]]
    assert "similar lines together" in a.subtitle_for(3)
    assert not any(w in a.subtitle_for(k) for k in range(4) for w in ("faster", "better model"))

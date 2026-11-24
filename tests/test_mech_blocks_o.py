"""Animations 227-229 (voxanim.mech_blocks_o): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the programs'."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_o as M
from voxanim.next_catalog import ITEMS, build_study

NAMES = ("padding-modes", "padding-deep-stacks", "padding-in-the-cnn")
STAYS = {"padding-modes": "x[0], a 5", "padding-deep-stacks": "28",
         "padding-in-the-cnn": "one run each"}       # a text of the picture that never dips
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_o", name))
    return _built[name]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.5]


def panel(a, phase, u=0.6):
    """The mono text in the code panel."""
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
        if a.station_for(k) == a.station_for(k + 1):        # a new station fades over
            assert "you are here" in shown
        if a.evidence_for(k) == a.evidence_for(k + 1):     # a new tag fades over
            assert a.evidence_for(k) in shown
        if a.code_for(k).lines() == a.code_for(k + 1).lines():   # new lines fade over
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
        full = [s for r in code.names() for s in B.code_lines(r)]
        assert shown[:len(lines)] == lines
        assert lines == (full if code.pick is None else [full[i] for i in code.pick])
        out = shown[len(lines):]                 # then the output slot, if anything
        assert not out or out[0].startswith("→ ")


def test_the_series_runs_on():
    modes, deep, cnn = (M.ANIMATIONS[n] for n in NAMES)
    titles = {item["name"]: item["title"] for item in ITEMS}
    assert [titles[n] for n in NAMES] == [modes.TITLE, deep.TITLE, cnn.TITLE]
    assert modes.LAST == "Same and valid"                  # 226
    assert modes.NEXT == deep.TITLE and deep.LAST == modes.TITLE
    assert deep.NEXT == cnn.TITLE and cnn.LAST == deep.TITLE
    assert cnn.NEXT == "A batch is one block"              # 230


# ---------------------------------------------------------------------- 227
@needs_mnist
def test_the_modes():
    a, s = built("padding-modes")
    m = s.metadata
    first = texts(a, 0, 0.6)
    assert {"x[0], a 5", "8 × 8"} <= set(first) and "12 × 12" not in first
    assert "→ [1, 1, 8, 8]" in panel(a, 0, 0.6)
    assert a.heading(0, 0.5) == "an 8 × 8 piece of a real 5"
    const = texts(a, 1, 0.8)
    assert {"12 × 12", "the ring: 0", "the paper: −0.42", m["readout"]["constant"]} <= set(const)
    assert "→ constant torch.Size([1, 1, 12, 12])" in panel(a, 1, 0.8)
    assert "    big = F.pad(part, [2] * 4, mode=mode)" in panel(a, 1, 0.4)
    for k, mode in enumerate(("reflect", "replicate", "circular"), 2):
        assert m["readout"][mode] in texts(a, k, 0.8)
        assert m["readout"][mode] not in texts(a, k, 0.3)        # after the arrows
        assert "→ %s torch.Size([1, 1, 12, 12])" % mode in panel(a, k, 0.8)
        assert a.heading(k, 0.5).startswith(mode + ": ")
    assert "0.94  1.56 | 0.59  1.56  0.94" in texts(a, 2, 0.8)
    # the border turns from one mode into the next, never blank
    assert np.array_equal(a.border(2, 0.0), np.asarray(m["blocks"]["constant"]))
    assert np.array_equal(a.border(2, 0.5), np.asarray(m["blocks"]["reflect"]))
    assert a.border_on(0, 0.9) == 0.0 and a.border_on(3, 0.0) == 1.0
    assert "next: Deep stacks need padding" in texts(a, 4, 0.95)


# ---------------------------------------------------------------------- 228
@needs_mnist
def test_the_deep_stacks():
    a, _ = built("padding-deep-stacks")
    assert "layer 12: 4 × 4" not in texts(a, 0, 0.3)               # still stacking
    one = texts(a, 0, 0.9)
    assert {"no padding", "layer 12: 4 × 4", "28"} <= set(one)
    assert "→ torch.Size([64, 1, 4, 4])" in panel(a, 0, 0.9)
    assert "    h = nn.Conv2d(1, 1, 3)(h)" in panel(a, 0, 0.5)
    two = texts(a, 1, 0.9)
    assert {"no padding", "layer 12: 4 × 4", "padding=1", "layer 12: 28 × 28"} <= set(two)
    assert "→ padding=1: torch.Size([64, 8, 28, 28])" in panel(a, 1, 0.9)
    walk = texts(a, 2, 0.9)
    assert [t for t in walk if t in ("28", "14", "7")] == ["28", "28", "14", "14", "7"]
    assert {"pad 1", "÷2", "14 ÷ 2 = 7", "2 × 2 squares become 1"} <= set(walk)
    assert "28 + 2 − 2 = 28" in texts(a, 2, 0.2)
    assert "→ [64, 16, 14, 14] → [64, 16, 7, 7]" in panel(a, 2, 0.9)
    assert a.heading(2, 0.5) == "the padded CNN: 28 → 28 → 14 → 14 → 7"
    last = texts(a, 3, 0.9)
    assert {"16 × 7 × 7 = 784", "784", "400 without padding"} <= set(last)
    assert "    nn.Linear(16 * 7 * 7, 10)," in panel(a, 3, 0.9)
    assert "→ [64, 784] → [64, 10]" in panel(a, 3, 0.9)
    assert "next: The CNN with padding, measured" in texts(a, 3, 0.95)


# ---------------------------------------------------------------------- 229
@needs_mnist
def test_the_cnn_with_padding():
    a, _ = built("padding-in-the-cnn")
    sizes = texts(a, 0, 0.9)
    assert {"80", "1,168", "4,010", "7,850", "Linear reads", "400", "784", "5,258",
            "9,098", "no padding", "padding=1", "one run each"} <= set(sizes)
    assert sizes.count("80") == 2 and sizes.count("1,168") == 2       # the same stamps
    assert "    nn.Linear(16 * 7 * 7, 10)," in panel(a, 0, 0.9)
    acc = texts(a, 1, 0.9)
    assert {"96.4%", "97.6%", "no padding 96.4%", "padding=1 97.6%"} <= set(acc)
    missed = texts(a, 2, 0.9)
    assert {"364", "237", "131", "66"} <= set(missed)
    assert a.heading(2, 0.5) == "fewer missed, but not padding alone"
    assert "→ no padding 0.9636, padding=1 0.9763" in panel(a, 2, 0.9)
    assert "not padding's alone" in a.subtitle_for(2)
    moved = texts(a, 3, 0.9)
    assert {"0 px", "2 px", "4 px", "6 px"} <= set(moved)
    fall = texts(a, 4, 0.9)
    assert {"moved 6 px", "18.5%", "16.9%", "no padding 18.5% at 6",
            "padding=1 16.9% at 6"} <= set(fall)
    assert "moved 6 px" not in texts(a, 3, 0.9)
    assert "Padding does not help" in a.subtitle_for(4)
    assert "next: A batch is one block" in texts(a, 4, 0.95)

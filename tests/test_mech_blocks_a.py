"""Animations 185-187 (voxanim.mech_blocks_a): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_a as M
from voxanim.next_catalog import build_study

NAMES = ("blocks-a-tensor", "blocks-a-batch", "blocks-flatten")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_a", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


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
        assert "you are here" in shown
        if a.evidence_for(k) == a.evidence_for(k + 1):     # a new tag fades over
            assert a.evidence_for(k) in shown
        if a.code_for(k).lines() == a.code_for(k + 1).lines():   # new lines fade over
            assert any(x.kind == "mono" and x.opacity > 0.5 for x in sc.texts)
        # the furniture (panel, running line, five stations) and the picture's card
        assert len(sc.quads) >= 10


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
    assert first == "start here" if name == NAMES[0] else first.startswith("last time: ")
    assert a.heading_at(a.t_done - 1.0).startswith("next: ")
    assert first in texts(a, 0, 0.1)


@needs_mnist
@pytest.mark.parametrize("name", NAMES)
def test_the_code_on_screen_is_the_scripts(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        shown = [s for s in mono(a, k, 0.6) if not s.startswith(("→", "  torch"))]
        assert shown == list(B.code_lines(code.region))


# ---------------------------------------------------------------------- 185
@needs_mnist
def test_a_tensor():
    a, s = built("blocks-a-tensor")
    m = s.metadata
    one = texts(a, 0, 0.6)
    assert {"5", "y: its label", "x"} <= set(one)
    assert "x, y = train_set[0]" in mono(a, 0, 0.6)
    numbers = texts(a, 1, 0.95)
    for v in np.ravel(m["patch"]):
        assert B.val(v) in numbers
    assert {"−0.42: paper", "2.82: darkest ink"} <= set(numbers)
    shape = texts(a, 2, 0.95)
    assert {"[1, 28, 28]", "1 channel", "28 rows", "28 columns", "784", "1 × 28 × 28 ="} <= set(
        shape)
    assert shape.count("28") >= 2
    assert "→ torch.Size([1, 28, 28]) 5" in mono(a, 2, 0.95)
    end = texts(a, 3, 0.9)
    assert {"the numbers", "the shape", "[1, 28, 28]", "784"} <= set(end)
    assert "next: 64 digits in one block" in end


# ---------------------------------------------------------------------- 186
@needs_mnist
def test_a_batch():
    a, s = built("blocks-a-batch")
    m = s.metadata
    assert "[1, 28, 28]" in texts(a, 0, 0.6) and "train_set" in texts(a, 0, 0.6)
    assert "64 at a time" in texts(a, 0, 0.9)
    counts = [int(x) for x in texts(a, 1, 0.3) if x.isdigit()]
    assert counts and 1 < max(counts) < 64                 # cards still arriving
    assert "64" in texts(a, 1, 0.97)
    labels = texts(a, 2, 0.95)
    for v in m["labels"][:M.ABatch.SHOWN]:
        assert str(v) in labels
    assert "y, all 64" in labels
    shape = texts(a, 3, 0.95)
    assert {"[64, 1, 28, 28]", "50,176", "64 × 784 =", "64"} <= set(shape)
    assert shape.count("28") >= 2
    out = mono(a, 3, 0.95)
    assert "→ torch.Size([64, 1, 28, 28])" in out and "  torch.Size([64])" in out
    mix = texts(a, 4, 0.95)
    for d, c in enumerate(m["counts"]):
        assert str(d) in mix and str(c) in mix
    assert "next: Flatten: a square becomes a strip" in mix


# ---------------------------------------------------------------------- 187
@needs_mnist
def test_flatten():
    a, s = built("blocks-flatten")
    m = s.metadata
    assert {"28 rows", "28 columns"} <= set(texts(a, 0, 0.9))
    assert "row 0" in texts(a, 1, 0.1)
    assert "a strip of 784: [784]" in texts(a, 1, 0.97) and "784" in texts(a, 1, 0.97)
    r, c = m["pixel"]
    where = texts(a, 2, 0.95)
    assert {"row %d, column %d" % (r, c), "%d × 28 + %d" % (r, c), "= %d" % m["place"],
            "same number: %s" % B.val(m["value"])} <= set(where)
    batch = texts(a, 3, 0.8)
    assert {"[64, 784]", "0", "numbers Flatten learns", "64 strips, one per digit"} <= set(batch)
    assert "→ [64, 1, 28, 28] → [64, 784]" in mono(a, 3, 0.8)

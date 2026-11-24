"""Animations 203-205 (voxanim.mech_blocks_g): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_g as M
from voxanim.next_catalog import build_study

NAMES = ("blocks-zero-grad", "blocks-optimizer-step", "blocks-learning-rate")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_g", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


def running(a, phase, u=0.6):
    """The code line boxed at ``u`` of ``phase`` (drawn at full strength)."""
    lo, hi = a.spans[phase]
    return [x.s.strip() for x in a.scene(lo + (hi - lo) * u).texts
            if x.kind == "mono" and x.opacity > 0.97 and not x.s.startswith("→")]


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
        if a.station_for(k) == a.station_for(k + 1):      # a new station fades over
            assert "you are here" in shown
        if a.evidence_for(k) == a.evidence_for(k + 1):
            assert a.evidence_for(k) in shown
        if a.code_for(k).lines() == a.code_for(k + 1).lines():
            assert any(x.kind == "mono" and x.opacity > 0.5 for x in sc.texts)
        assert len(sc.quads) >= 7                          # the panel, its line, 5 stations
        # and something of the picture stays: a running number, or a block's label
        picture = [x for x in sc.texts if x.opacity > 0.5 and x.x < B.PANEL[0] - 0.01
                   and x.y > B.PIC_TOP]
        assert picture, (name, k)


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
def test_the_code_on_screen_is_the_scripts(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        shown = [s for s in mono(a, k, 0.6) if not s.startswith("→")]
        assert shown == list(code.lines())
        script = {B.code_lines(n) for n in code.names()}
        assert all(any(s in lines for lines in script) for s in shown)


# ---------------------------------------------------------------------- 203
@needs_mnist
def test_zero_grad():
    a, s = built("blocks-zero-grad")
    m = s.metadata
    assert a.heading_at(1.0) == "last time: loss.backward() makes red tiles"
    one = texts(a, 0, 0.6)
    assert {"model[5].weight: [10, 32]", ".grad", "size of .grad", "0.84"} <= set(one)
    for g in m["g0"]:
        assert B.val(g, 4) in one
    assert running(a, 0, 0.5) == ["loss.backward()"]
    zero = texts(a, 1, 0.9)
    assert zero.count("0") == 3 and {"0.00", "all 52,650 set to 0"} <= set(zero)
    assert running(a, 1, 0.6) == ["optimizer.zero_grad()"]
    assert not any(B.val(g, 4) in zero for g in m["g0"])
    again = texts(a, 2, 0.9)
    assert "0.78" in again and all(B.val(g, 4) in again for g in m["g1"])
    piled = texts(a, 3, 0.9)
    assert {"1.29", "batch 1 + batch 2"} <= set(piled)
    assert all(B.val(g, 4) in piled for g in m["pile"])
    assert "optimizer.zero_grad()" not in mono(a, 3)
    assert "examples/blocks_parts.py" in texts(a, 3)
    pile = texts(a, 4, 0.97)
    assert {"938", "what .grad holds", "this batch's own", "batches", "size"} <= set(pile)
    assert B.val(m["used"][-1]) in pile
    assert "100" in texts(a, 4, 0.55) and B.val(m["used"][m["batches"].index(100)]) in texts(
        a, 4, 0.55)
    end = texts(a, 5, 0.9)
    assert {"95%", "10%", "with zero_grad", "without zero_grad", "test digits right"} <= set(end)
    assert "next: optimizer.step() moves the tiles" in end


# ---------------------------------------------------------------------- 204
@needs_mnist
def test_optimizer_step():
    a, s = built("blocks-optimizer-step")
    m = s.metadata
    one = texts(a, 0, 0.8)
    assert {"SGD", "lr=0.1", "52,650", "model[5].weight: [10, 32]"} <= set(one)
    assert running(a, 0, 0.8) == ["lr=0.1)"]
    rows = texts(a, 1, 0.95)
    for w, g, new in zip(m["before_w"], m["g"], m["after_w"]):
        assert {B.val(w, 4), B.val(g, 4), B.val(new, 4)} <= set(rows)
    assert rows.count("− 0.1 ×") == 3 and rows.count("=") == 3
    assert running(a, 1) == ["optimizer.step()"]
    every = texts(a, 2, 0.9)
    assert {"320", "52,650", "moved in this block", "in the whole model"} <= set(every)
    assert "0" in texts(a, 2, 0.15) and "320" not in texts(a, 2, 0.15)
    batch = texts(a, 3, 0.9)
    assert {"before", "after", "2.3223", "2.2616", "0.0607", "lower by"} <= set(batch)
    assert running(a, 3, 0.9) == ["loss = loss_fn(logits, y)"]
    assert "next: the learning rate: how far to move" in batch


# ---------------------------------------------------------------------- 205
@needs_mnist
def test_learning_rate():
    a, s = built("blocks-learning-rate")
    m = s.metadata
    start = texts(a, 0, 0.8)
    assert {"lr=0.1", "start: 2.32", "0.001", "0.1", "0.8", "938 steps", "loss",
            "correct"} <= set(start)
    assert "one step moves a tile 0.0045" in start
    assert running(a, 0, 0.8) == ["lr=0.1)"]
    rows = zip(m["rate_text"], m["percent"], M.LearningRate.WORDS, m["moves_text"])
    for k, (rate, pct, word, move) in enumerate(rows, 1):
        mid = texts(a, k, 0.5)
        assert "lr=%s" % rate in mid and "one step moves a tile %s" % move in mid
        assert pct not in texts(a, k, 0.3)                   # not before its curve is drawn
        late = texts(a, k, 0.95)
        assert {pct, word} <= set(late)
        assert running(a, k, 0.6) == ["optimizer.step()"]
    end = texts(a, 4, 0.9)
    assert {"55%", "95%", "27%", "crawls", "learns", "bounces"} <= set(end)
    assert "next: Adam keeps two more blocks" in end

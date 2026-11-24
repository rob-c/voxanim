"""Animations 209-211 (voxanim.mech_blocks_i): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_i as M
from voxanim.next_catalog import build_study

NAMES = ("blocks-evaluation", "blocks-save-and-load", "blocks-one-step-slow-motion")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_i", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


def code_shown(a, phase, u=0.6):
    """The code panel's lines (the output slot left out)."""
    lines = set(a.code_for(phase).lines())
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and x.kind == "mono" and x.x >= B.PANEL[0] and x.s in lines]


def anywhere(a, phase, u=0.6):
    """Every text drawn, however faint."""
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.05]


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
        if a.station_for(k) is not None and a.station_for(k) == a.station_for(k + 1):
            assert "you are here" in shown
        if a.evidence_for(k) == a.evidence_for(k + 1):     # a new tag fades over
            assert a.evidence_for(k) in shown
        if a.code_for(k).lines() == a.code_for(k + 1).lines():   # new lines fade over
            assert any(x.kind == "mono" and x.opacity > 0.5 for x in sc.texts)
        # the furniture (panel, running line, five stations) and the picture's content
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
    assert first.startswith("last time: ")
    assert a.heading_at(a.t_done - 1.0).startswith("next: ")
    assert first in texts(a, 0, 0.1)


@needs_mnist
@pytest.mark.parametrize("name", NAMES)
def test_at_most_twelve_words_of_explanation_at_once(name):
    a, _ = built(name)
    for lo, hi in a.spans:
        for t in np.linspace(lo, hi, 25):
            shown = [x.s for x in a.scene(float(t)).texts
                     if x.opacity > 0.5 and x.kind != "mono" and x.x < B.PANEL[0]
                     and 0.25 < x.y < 0.93]
            assert sum(B.words(s) for s in shown) <= 12, shown


@needs_mnist
@pytest.mark.parametrize("name", NAMES)
def test_the_code_on_screen_is_the_scripts(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        regions = code.names()
        assert list(code.lines()) == [s for r in regions for s in B.code_lines(r)]
        assert code_shown(a, k, 0.6) == list(code.lines())


# ---------------------------------------------------------------------- 209
@needs_mnist
def test_evaluation():
    a, s = built("blocks-evaluation")
    m = s.metadata
    mode = texts(a, 0, 0.9)
    assert {"eval", "no dropout here: nothing changes", "correct", "0"} <= set(mode)
    assert "model.eval()" in mono(a, 0, 0.9) and "model.train()" in anywhere(a, 0, 0.9)
    assert "model.train()" in mono(a, 0, 0.1)                   # the switch starts at train
    assert "model.eval(): switch to test mode" in mode
    grads = texts(a, 1, 0.9)
    assert {"1.weight", "3.weight", "5.weight", "red tiles made", "0",
            "training makes 52,650"} <= set(grads)
    one = texts(a, 2, 0.95)
    for p in m["shown_preds"]:
        assert str(p) in one
    assert {"says %d, not %d" % (m["first_wrong_pred"], m["first_wrong_label"]),
            "%d of 10" % m["shown_correct"], str(m["shown_correct"])} <= set(one)
    early = [int(x) for x in texts(a, 2, 0.3) if x.isdigit() and x in {"1", "2", "3", "4"}]
    assert early                                                # counted one by one
    every = texts(a, 3, 0.95)
    for n in m["per_batch"]:
        assert str(n) in every
    assert {"9,494", m["fraction"], "of 1,000"} <= set(every)
    assert "→ %s" % m["printed"] in mono(a, 3, 0.95)
    middle = texts(a, 3, 0.25)
    assert any(B.commas(v) in middle for v in m["running"][1:-1])   # the total, growing
    assert "next: saving the model to a file" in texts(a, 3, 0.95)


# ---------------------------------------------------------------------- 210
@needs_mnist
def test_save_and_load():
    a, s = built("blocks-save-and-load")
    m = s.metadata
    boxes = texts(a, 0, 0.9)
    assert set(m["keys"]) <= set(boxes) and set(m["shape_texts"]) <= set(boxes)
    assert {"model", "after training"} <= set(boxes)
    count = texts(a, 1, 0.9)
    assert {"50,176 + 64", "2,048 + 32", "320 + 10", "52,650", "numbers",
            "× 4 bytes = 210,600 bytes"} <= set(count)
    assert "mnist.pt" in texts(a, 2, 0.9)
    assert "a new model" in texts(a, 3, 0.3)
    back = texts(a, 3, 0.95)
    assert {"0", "largest change", "model", "mnist.pt"} <= set(back)
    same = texts(a, 4, 0.9)
    assert {"10,000", "same answers", "of 10,000 test digits"} <= set(same)
    assert "next: one training step, slowly" in same
    assert 'model.load_state_dict(state)' in mono(a, 4, 0.5)


# ---------------------------------------------------------------------- 211
@needs_mnist
def test_one_step():
    a, s = built("blocks-one-step-slow-motion")
    m = s.metadata
    w = m["weight"]
    assert [a.station_for(k) for k in range(5)] == list(B.STATIONS)
    data = texts(a, 0, 0.9)
    assert {"x: [64, 1, 28, 28]", "y: [64]", "x, y", "1.weight", "logits"} <= set(data)
    for v in m["labels"][:8]:
        assert str(v) in data
    fwd = texts(a, 1, 0.95)
    assert set(m["shapes"][1:]) <= set(fwd)
    assert {"first digit: a %d" % m["first_label"], "highest score: %d" % m["first_pred"]} <= set(
        fwd)
    loss = texts(a, 2, 0.9)
    assert {"%.2f" % m["loss"], "ln 10 = 2.30", "loss"} <= set(loss)
    assert "zero_grad: empty last step's red tiles" in texts(a, 3, 0.2)
    assert "old red tiles: gone" in texts(a, 3, 0.2)
    assert {"52,650", "red tiles"} <= set(texts(a, 3, 0.95))
    step = texts(a, 4, 0.95)
    assert {B.val(w["before"], 4), B.val(w["grad"], 4), B.val(w["after"], 4),
            "%.2f → %.2f" % (m["loss"], m["loss_after"]), "%.2f" % m["loss_after"]} <= set(step)
    assert "next: 938 steps, fast" in step
    running = [code_shown(a, k, 0.9) for k in range(5)]
    assert all(r == list(B.code_lines("step")) for r in running)

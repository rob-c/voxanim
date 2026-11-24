"""Animations 206-208 (voxanim.mech_blocks_h): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_h as M
from voxanim.next_catalog import build_study

NAMES = ("blocks-adam", "blocks-the-training-loop", "blocks-epochs-and-batches")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_h", name))
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
        # the furniture (panel, five stations; the running line too, unless the code
        # changes) and the picture's blocks
        same = a.code_for(k).lines() == a.code_for(k + 1).lines()
        assert len(sc.quads) >= (10 if same else 9)


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
    script = B.program()
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        shown = [s for s in mono(a, k, 0.6) if not s.startswith("→ ")]
        assert shown == list(code.lines())
        for s in shown:                                     # every line is the program's
            assert s.strip() in {x.strip() for x in script}


# ---------------------------------------------------------------------- 206
@needs_mnist
def test_adam():
    a, s = built("blocks-adam")
    m = s.metadata
    T = m["text"]
    one = texts(a, 0, 0.9)
    assert {"weight [10, 32]", "m [10, 32]", "v [10, 32]", "all 0", "same shape"} <= set(one)
    assert "    optimizer = Adam(model.parameters())" in mono(a, 0, 0.9)
    memory = texts(a, 1, 0.95)
    assert memory.count("52,650") == 3 and {"157,950", "3 × the model", "numbers kept"} <= set(
        memory)
    assert "105,300" in texts(a, 1, 0.4)                    # the running total, half way
    step = texts(a, 2, 0.97)
    assert {"weight", "m", "v", T["w_before"], "g = " + T["g"], T["m_sum"], T["m"],
            T["v_sum"], T["v"]} <= set(step)
    assert "optimizer.step()" in mono(a, 2, 0.6)
    move = texts(a, 3, 0.95)
    assert {"m " + T["m_hat"], T["root"], T["move"], T["move_sum"], T["w_after"],
            "largest move of all: " + T["largest"]} <= set(move)
    assert T["w_before"] not in move and T["m_sum"] not in move
    curves = texts(a, 4, 0.95)
    assert {"SGD, lr 0.1", "Adam, lr 0.001", "237", "938", "↑ from 2.32"} <= set(curves)
    assert "next: the training loop" in curves
    assert "94.61%" in a.subtitle_for(4) and "94.94%" in a.subtitle_for(4)


# ---------------------------------------------------------------------- 207
@needs_mnist
def test_the_training_loop():
    a, s = built("blocks-the-training-loop")
    m = s.metadata
    track = texts(a, 0, 0.9)
    assert set(B.STATIONS) <= set(track) and {"steps done", "0"} <= set(track)
    assert "a batch of 64 digits" in track
    # one step: the card goes round; the running line, the readout and the corner map
    # follow it station by station
    for (station, key, arrive, leave) in M.TrainingLoop.STOPS:
        u = (arrive + leave) / 2
        shown = texts(a, 1, u)
        assert m["reads"][key] in shown
        lo, hi = a.spans[1]
        t = lo + (hi - lo) * u
        assert a._tl_station.at(t) == station
        lines, line = a._tl_run.at(t)
        assert lines[line].strip() == next(x["code"] for x in m["stations"]
                                           if x["line"] == line)
    assert "1" in texts(a, 1, 0.97)                         # one step done
    assert "4" in texts(a, 2, 0.39) and "938" in texts(a, 2, 0.97)
    assert "loss: %.3f" % m["last_loss"] in texts(a, 2, 0.97)
    epoch = texts(a, 3, 0.8)
    assert {"938 steps = 1 epoch", "last: %.4f" % m["last_loss"]} <= set(epoch)
    assert "→ " + m["printed"] in mono(a, 3, 0.8)
    assert "next: epochs and batches" in texts(a, 3, 0.95)


# ---------------------------------------------------------------------- 208
@needs_mnist
def test_epochs_and_batches():
    a, s = built("blocks-epochs-and-batches")
    m = s.metadata
    split = texts(a, 0, 0.95)
    assert {"60,000 ÷ 64 = 937.5", "937 × 64 + 32 = 60,000", "937 + 1 = 938 steps",
            "the last: 32", "938", "one tile:", "64 digits"} <= set(split)
    assert "                    batch_size=64," in mono(a, 0, 0.6)
    epoch = texts(a, 1, 0.95)
    assert {"938", "60,000", "digits used", "each digit once", "= 1 epoch"} <= set(epoch)
    order = texts(a, 2, 0.95)
    assert {"epoch 1", "epoch 2", "epoch 3"} <= set(order)
    assert "                    shuffle=True)" in mono(a, 2, 0.6)
    per = texts(a, 3, 0.95)
    for mean in m["epoch_mean"]:
        assert "%.3f" % mean in per
    assert {"× 3 = 2,814", "2,814", "1,876", "↑ from 2.32"} <= set(per)
    assert "for epoch in range(1):" in mono(a, 3, 0.6)
    assert "the script runs one" in a.subtitle_for(3)

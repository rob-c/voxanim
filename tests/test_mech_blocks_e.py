"""Animations 197-199 (voxanim.mech_blocks_e): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_e as M
from voxanim.next_catalog import build_study

NAMES = ("blocks-logits-and-softmax", "blocks-which-is-which", "blocks-the-data-pipeline")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_e", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


def panel(a, phase, u=0.6):
    """The code panel's lines (the picture's monospace block names left out)."""
    x0 = B.PANEL[0]
    lo, hi = a.spans[phase]
    sc = a.scene(lo + (hi - lo) * u)
    return [x.s for x in sc.texts if x.opacity > 0.5 and x.kind == "mono" and x.x > x0]


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
        if a.station_for(k) == a.station_for(k + 1):       # a new station fades over
            assert "you are here" in shown
        if a.evidence_for(k) == a.evidence_for(k + 1):     # a new tag fades over
            assert a.evidence_for(k) in shown
        if a.code_for(k).lines() == a.code_for(k + 1).lines():   # new lines fade over
            assert any(x.kind == "mono" and x.opacity > 0.5 for x in sc.texts)
        # the furniture (panel, running line, five stations) and the picture's steady part
        assert len(sc.quads) >= 10
        # and something of the picture besides the furniture's words
        furniture = {"you are here", *B.STATIONS, a.evidence_for(k), a.evidence_for(k + 1)}
        assert any(x.x < B.PANEL[0] and x.s not in furniture and x.y > B.PIC_TOP - 0.02
                   for x in sc.texts if x.opacity > 0.5)


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
        shown = [s for s in panel(a, k, 0.6) if not s.startswith(("→", "  "))
                 or s in code.lines()]
        assert shown == list(code.lines())
        program = [s for n in code.names() for s in B.code_lines(n)]
        assert all(s in program for s in code.lines())


# ---------------------------------------------------------------------- 197
@needs_mnist
def test_logits_and_softmax():
    a, s = built("blocks-logits-and-softmax")
    m = s.metadata
    y = m["label"]
    one = texts(a, 0, 0.6)
    assert {"logits", "label: %d" % y, "10", "numbers", "0"} <= set(one)
    for d, v in enumerate(m["logits"]):
        assert str(d) in one and B.val(v) in one
    assert "nn.Linear(32, 10)," in [x.strip() for x in mono(a, 0, 0.6)]
    soft = texts(a, 1, 0.9)
    assert "softmax" in soft and "all positive" in soft
    for v in m["probs"]:
        assert B.val(v) in soft
    assert panel(a, 1) == ["probs = logits.softmax(dim=1)"]
    # adding up: a running total, the four biggest one at a time, then all ten
    assert "0.00" in texts(a, 2, 0.04) and "added up" in texts(a, 2, 0.04)
    p = m["probs"]
    first = B.val(p[m["order"][0]])
    assert texts(a, 2, 0.15).count(first) >= 2                # in its row and as the total
    assert {"1.00", "= 1"} <= set(texts(a, 2, 0.95))
    loss = texts(a, 3, 0.9)
    assert {B.val(m["loss"]), "−ln %s =" % B.val(m["prob_right"]), "loss"} <= set(loss)
    assert panel(a, 3) == ["loss_fn = nn.CrossEntropyLoss()", "logits = model(x)",
                           "loss = loss_fn(logits, y)"]
    twice = texts(a, 4, 0.95)
    assert {"softmax twice", B.val(m["loss_twice"]), "at least 1.46",
            "−ln %s =" % B.val(m["twice_right"]), "85% right, not 95%"} <= set(twice)
    for v in m["twice"]:
        assert B.val(v) in twice
    assert "nn.Softmax(dim=1)," in [x.strip() for x in mono(a, 4, 0.6)]
    assert "next: every block, side by side" in twice
    assert a.station_for(0) == "forward" and a.station_for(4) == "loss"


# ---------------------------------------------------------------------- 198
@needs_mnist
def test_which_is_which():
    a, s = built("blocks-which-is-which")
    m = s.metadata
    for k, group in enumerate(m["groups"]):
        guess = texts(a, k, 0.36)
        answer = texts(a, k, 0.97)
        assert guess.count("?") >= 2 * len(group)          # the shape out and the weights
        for i in group:
            b = m["blocks"][i]
            name = a.call_text(b)
            assert name in mono(a, k, 0.36) and name in mono(a, k, 0.97)
            assert "%s → ?" % b["shape_in_text"] in guess
            assert "%s → %s" % (b["shape_in_text"], b["shape_out_text"]) in answer
            assert B.commas(b["weights"]) in answer
            assert ("weights: %s =" % b["how"] if b["learns"] else "weights: none") in answer
            # the running line is the block's own line
        assert len(group) <= 4
    # the block names are the code's calls (the long one without its keyword setting)
    assert "nn.Conv2d(1, 4, 3)" in mono(a, 1, 0.97)
    assert "nn.MultiheadAttention(16, 2)" in mono(a, 3, 0.97)
    assert "nn.Linear(784, 64)" in mono(a, 0, 0.97)
    # the tally along the bottom, all through
    assert "1 of 3 learn" in texts(a, 0, 0.97) and "1 of 3 learn" in texts(a, 1, 0.1)
    end = texts(a, 3, 0.97)
    assert {"6 of 11 learn", "6 of 11", "blocks that learn weights",
            "next: the data pipeline"} <= set(end)
    assert "1,088" in end and "4 × 16 × 16 + 4 × 16" in m["blocks"][9]["how"]
    assert a.heading(0, 0.3).startswith("guess") and a.heading(0, 0.9).startswith("answers")


# ---------------------------------------------------------------------- 199
@needs_mnist
def test_the_data_pipeline():
    a, s = built("blocks-the-data-pipeline")
    m = s.metadata
    stages = {"MNIST", "ToTensor", "Normalize", "DataLoader"}
    one = texts(a, 0, 0.9)
    assert stages <= set(one) and {"60,000", "10,000", "label: 5"} <= set(one)
    two_a, two_b = texts(a, 1, 0.3), texts(a, 1, 0.9)
    for v in np.ravel(m["raw"]):
        assert str(v) in two_a
    for v in np.ravel(m["tensor"]):
        assert "%.2f" % v in two_b
    assert {"÷ 255", "0..1"} <= set(two_b) and "0..255" in two_a
    three = texts(a, 2, 0.95)
    for v in np.ravel(m["normalised"]):
        assert B.val(v) in three
    assert {"mean 0.1307", "spread 0.3081", m["pixel_sum"], "= 0.61"} <= set(three)
    assert stages <= set(three)
    four = texts(a, 3, 0.9)
    assert {"epoch 1", "epoch 2"} <= set(four)
    for i in m["first"][:M.DataPipeline.SHOWN] + m["again"][:M.DataPipeline.SHOWN]:
        assert str(i) in four
    assert "shuffle=True)" in [x.strip() for x in mono(a, 3, 0.9)]
    # the belt: the counter runs 1, 2, 3, then 938 with the last batch of 32
    counts = [texts(a, 4, u) for u in (0.2, 0.33, 0.5, 0.97)]
    assert "1" in counts[0] and "2" in counts[1] and "3" in counts[2]
    assert "64" in counts[0] and "32" in counts[2]
    assert {"938", "937 × 64 + 32 = 60,000", "one epoch", "batches in"} <= set(counts[3])
    assert "for x, y in loader:" in [x.strip() for x in mono(a, 4, 0.5)]
    assert "next: the forward pass" in counts[3]

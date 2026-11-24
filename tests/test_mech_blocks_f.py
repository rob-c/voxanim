"""Animations 200-202 (voxanim.mech_blocks_f): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_f as M
from voxanim.next_catalog import build_study

NAMES = ("blocks-forward-pass", "blocks-the-loss", "blocks-backward")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_f", name))
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
        # the furniture (panel, running line, five stations) and the picture's blocks
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
def test_the_code_on_screen_is_the_scripts(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        shown = [s for s in mono(a, k, 0.6) if not s.startswith(("→", "  "))
                 or s in code.lines()]
        assert shown == list(code.lines())
        assert list(code.lines()) == [s for r in code.names() for s in B.code_lines(r)]


# ---------------------------------------------------------------------- 200
@needs_mnist
def test_forward_pass():
    a, s = built("blocks-forward-pass")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"x: the batch", "[64, 1, 28, 28]", "64 digits"} <= set(one)
    assert set(m["names"][1:]) <= set(one)                 # the whole stack, from the start
    assert "[64, 784]" not in one                          # the batch has not reached it
    assert "logits = model(x)" in mono(a, 0, 0.9)
    assert {"[64, 784]", "784 = 28 × 28"} <= set(texts(a, 1, 0.9))
    assert "→ [64, 1, 28, 28] → [64, 784]" in mono(a, 1, 0.9)
    assert {"[64, 64]", "64 per digit"} <= set(texts(a, 2, 0.9))
    assert "[64, 32]" not in texts(a, 2, 0.9)
    assert "%s are 0" % m["zeroed_text"][0] in texts(a, 3, 0.25)
    assert "→ [64, 64] → [64, 64]" in mono(a, 3, 0.25)
    assert "→ [64, 64] → [64, 32]" in mono(a, 3, 0.6)
    assert "%s are 0" % m["zeroed_text"][1] in texts(a, 3, 0.95)
    end = texts(a, 4, 0.95)
    for shape in m["shape_texts"]:
        assert shape in end
    assert "logits" in end and "→ [64, 32] → [64, 10]" in mono(a, 4, 0.95)
    assert "next: the loss: how wrong are the logits?" in end
    # the batch reaches the blocks one at a time, in order
    assert a.reach == sorted(a.reach) and len(set(a.reach)) == len(a.reach)


# ---------------------------------------------------------------------- 201
@needs_mnist
def test_the_loss():
    a, s = built("blocks-the-loss")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"y = 0", "0.094", "−ln 0.094 = 2.36", "its chances, 0 to 9"} <= set(one)
    assert {str(d) for d in range(10)} <= set(one)
    assert "loss = loss_fn(logits, y)" in mono(a, 0, 0.9)
    assert "loss_fn = nn.CrossEntropyLoss()" in mono(a, 0, 0.9)
    counts = [int(x) for x in texts(a, 1, 0.4) if x.isdigit()]
    assert counts and 1 < max(counts) < 64                 # bars still arriving
    assert {"64", "of 64", "one bar per digit", "digits with a loss"} <= set(texts(a, 1, 0.97))
    mean = texts(a, 2, 0.95)
    assert {"add the 64 losses: 148.63", "148.63 ÷ 64 = 2.322", "2.322"} <= set(mean)
    assert "→ loss: 2.322" in mono(a, 2, 0.95)
    end = texts(a, 3, 0.9)
    assert {"−ln 0.1 = ln 10 = 2.303", "this model: 2.322", "2.322"} <= set(end)
    assert "next: loss.backward(): a gradient for every weight" in end
    assert m["ln10_text"] == "2.303"


# ---------------------------------------------------------------------- 202
@needs_mnist
def test_backward():
    a, s = built("blocks-backward")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"loss", "2.322", "starts here", "[64, 784]", "[32, 64]", "[10, 32]"} <= set(one)
    assert {"Linear(784, 64)", "Linear(64, 32)", "Linear(32, 10)"} <= set(one)
    assert not any(t.startswith(".grad") for t in one)
    assert "loss.backward()" in mono(a, 0, 0.9)
    last = texts(a, 1, 0.9)
    assert {".grad [10, 32]", "330", "numbers with a .grad"} <= set(last)
    assert ".grad [64, 784]" not in last and ".grad [32, 64]" not in last
    assert "2,410" in texts(a, 2, 0.4) and ".grad [64, 784]" not in texts(a, 2, 0.4)
    back = texts(a, 2, 0.95)
    assert {".grad [64, 784]", ".grad [32, 64]", ".grad [10, 32]", "52,650"} <= set(back)
    same = texts(a, 3, 0.9)
    assert same.count("=") == 3
    assert "→ torch.Size([64, 784])" in mono(a, 3, 0.9)
    assert "print(model[1].weight.grad.shape)" in mono(a, 3, 0.9)
    assert "next: zero_grad(): empty the red tiles" in same
    # from the last layer back to the first
    assert a.reach[2] < a.reach[1] < a.reach[0]
    assert m["running_text"] == ["330", "2,410", "52,650"]

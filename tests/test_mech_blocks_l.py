"""Animations 218-220 (voxanim.mech_blocks_l): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the CNN script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_l as M
from voxanim.next_catalog import build_study

NAMES = ("blocks-cnn-backward", "blocks-cnn-same-loop", "blocks-cnn-stamps-learn")
TITLES = {"blocks-cnn-backward": "A shared stamp's gradient",
          "blocks-cnn-same-loop": "The same training loop",
          "blocks-cnn-stamps-learn": "The stamps learn"}
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_l", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


# ---------------------------------------------------------------------- no data needed
def test_small_numbers_keep_their_sign():
    assert M.val5(0.059336) == "0.05934"
    assert M.val5(-0.000588) == "−0.00059"
    assert M.val5(-1e-9) == "0.00000"


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
    assert TITLES[name] == a.TITLE
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
        else:
            assert set(B.STATIONS) <= set(shown)          # the map itself stays
        if a.evidence_for(k) == a.evidence_for(k + 1):
            assert a.evidence_for(k) in shown
        if a.code_for(k).lines() == a.code_for(k + 1).lines():
            assert any(x.kind == "mono" and x.opacity > 0.5 for x in sc.texts)
        # the furniture (panel, running line, stations) and the picture's shapes; or, in
        # 219 from phase 3 on, the band of the shared step lines
        assert len(sc.quads) >= 10 or "step: the same 6 lines in both files" in shown


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
        every = [s for r in code.names() for s in B.code_lines(r)]
        assert list(code.lines()) == (every if code.pick is None
                                      else [every[i] for i in code.pick])
        assert all(B.code_file(r) == "examples/mnist_cnn.py" or r == "model"
                   for r in code.names())


# ---------------------------------------------------------------------- 218
@needs_mnist
def test_cnn_backward():
    a, s = built("blocks-cnn-backward")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"2.370", "5,258", "numbers with a .grad", "weight [8, 1, 3, 3]", "8 stamps"} <= set(one)
    assert "last time: one digit through the CNN" in texts(a, 0, 0.1)
    assert "loss.backward()" in mono(a, 0, 0.9)
    assert ".grad [8, 1, 3, 3]" not in one
    same = texts(a, 1, 0.9)
    assert {".grad [8, 1, 3, 3]", "weight [8, 1, 3, 3]", "one red number per weight: 72"} <= set(
        same)
    assert "→ torch.Size([8, 1, 3, 3])" in mono(a, 1, 0.9)
    assert "print(model[0].weight.grad.shape)" in mono(a, 1, 0.9)
    # one place at a time: pixel x red = part, the running count and sum never blank
    for i, u in enumerate((0.2, 0.4, 0.6)):
        shown = texts(a, 2, u)
        assert m["worked"][i]["text"] in shown
        assert {"parts added", str(i + 1), "this digit", "pixel × red = part"} <= set(shown)
    assert "0.00041" in texts(a, 2, 0.4)                  # 0.00100 − 0.00059
    swept = texts(a, 2, 0.95)
    assert {"676", "0.00599", "a 4, digit 5 of 64"} <= set(swept)
    assert "676 places: 0.00599" in texts(a, 2, 0.95)
    # every digit: the sum grows to PyTorch's number
    assert {"64 × 676 = 43,264 parts", "one sum per digit"} <= set(texts(a, 3, 0.2))
    assert "PyTorch's .grad" not in texts(a, 3, 0.4)
    end = texts(a, 3, 0.95)
    assert end.count("0.05934") == 2 and {"=", "PyTorch's .grad", "all parts added"} <= set(end)
    assert "next: the same training loop" in end
    assert m["total_text"] == m["grad_text"] == "0.05934"


# ---------------------------------------------------------------------- 219
@needs_mnist
def test_same_loop():
    a, s = built("blocks-cnn-same-loop")
    m = s.metadata
    one = texts(a, 0, 0.95)
    assert set(m["mlp_blocks"]) <= set(one) and "52,650 numbers" in one
    assert "Conv2d(1, 8, 3)" not in one
    assert "nn.Linear(784, 64)," in [x.strip() for x in mono(a, 0, 0.6)]
    two = texts(a, 1, 0.95)
    assert set(m["cnn_blocks"]) <= set(two) and "5,258 numbers" in two
    assert "nn.Linear(16 * 5 * 5, 10)," in [x.strip() for x in mono(a, 1, 0.6)]
    three = texts(a, 2, 0.95)
    assert {"step: the same 6 lines in both files", "55 of 63 lines the same"} <= set(three)
    assert mono(a, 2, 0.5) == list(B.code_lines("step"))
    curves = texts(a, 3, 0.9)
    assert {"MLP", "CNN", "each point: mean of 25 steps", "938 steps"} <= set(curves)
    end = texts(a, 4, 0.9)
    assert {"last 100 steps", "0.164", "0.095", "CNN lower: 38 of 38 points"} <= set(end)
    assert "next: the stamps learn" in end
    assert "loss = loss_fn(logits, y)" in [x.strip() for x in mono(a, 3, 0.5)]


# ---------------------------------------------------------------------- 220
@needs_mnist
def test_stamps_learn():
    a, s = built("blocks-cnn-stamps-learn")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"step", "0", "stamp 1: 9 numbers", "−0.12", "0.26"} <= set(one)
    assert {str(j) for j in range(1, 9)} <= set(one)
    assert "10" not in one
    assert "nn.Conv2d(1, 8, 3), nn.ReLU()," in [x.strip() for x in mono(a, 0, 0.9)]
    # the rows arrive one at a time; the average move swaps in place
    assert "10" in texts(a, 1, 0.2) and "100" not in texts(a, 1, 0.2)
    assert "0.02" in texts(a, 1, 0.2)
    rows = texts(a, 1, 0.95)
    assert {"0", "10", "100", "500", "938", "0.14", "average move from step 0"} <= set(rows)
    assert "optimizer.step()" in mono(a, 1, 0.5)
    before = texts(a, 2, 0.9)
    assert {"0", "938", "one map per stamp"} <= set(before) and "100" not in before
    end = texts(a, 3, 0.95)
    assert "next: the CNN against the MLP" in end
    assert "each column: one stamp, before and after" in end
    assert m["moved_text"][-1] == "0.14" and m["label"] == 5

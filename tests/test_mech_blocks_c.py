"""Animations 191-193 (voxanim.mech_blocks_c): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the scripts'."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_c as M
from voxanim.next_catalog import build_study

NAMES = ("blocks-conv2d", "blocks-pooling", "blocks-dropout")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_c", name))
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
        # the furniture (panel, running line, five stations: 7) and something of the
        # picture that stays: a card, or the dropout switch
        assert len(sc.quads) >= 8
        if name == "blocks-dropout":
            assert "eval" in shown if k >= 3 else "h: 64 numbers" in shown
        else:
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
    # every heading stays up for 3 s or more
    ts = np.arange(0.0, a.t_done, 0.05)
    runs, last, start = [], None, 0.0
    for t in ts:
        h = a.heading_at(float(t))
        if h != last:
            if last is not None:
                runs.append((last, float(t) - start))
            last, start = h, float(t)
    assert all(d >= 3.0 for _h, d in runs), runs


@needs_mnist
@pytest.mark.parametrize("name", NAMES)
def test_the_code_on_screen_is_the_scripts(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        shown = [s for s in mono(a, k, 0.6) if not s.startswith(("→", "  "))]
        assert shown == list(code.lines())
        whole = [s for n in code.names() for s in B.code_lines(n)]
        assert all(s in whole for s in shown)


# ---------------------------------------------------------------------- 191
@needs_mnist
def test_conv2d():
    a, s = built("blocks-conv2d")
    m = s.metadata
    stamp = texts(a, 0, 0.8)
    for w in np.ravel(m["weight"][0]):
        assert B.val(w) in stamp
    assert {"stamp 1 of 4", "bias %s" % B.val(m["bias"][0]), "9 weights,",
            "random to start"} <= set(stamp)
    assert "conv = nn.Conv2d(1, 4, 3)" in mono(a, 0, 0.8)
    one = texts(a, 1, 0.7)
    for v in np.ravel(m["patch"]):
        assert B.val(v) in one
    assert {"9 pixels", "×", "+ bias", B.val(m["output"])} <= set(one)
    assert B.val(m["products_sum"]) in texts(a, 1, 0.5) and "add up" in texts(a, 1, 0.5)
    slide = texts(a, 2, 0.95)
    assert {"places", "676", "26 × 26 = 676"} <= set(slide)
    counts = [int(x.replace(",", "")) for x in texts(a, 2, 0.6) if x.replace(",", "").isdigit()]
    assert any(26 < c < 676 for c in counts)                # still sliding
    four = texts(a, 3, 0.9)
    assert {"1", "2", "3", "4", "4 stamps", "40", "4 × 3 × 3 + 4 =",
            "the same 40 at all 676 places"} <= set(four)
    assert "→ [64, 1, 28, 28] → [64, 4, 26, 26]" in mono(a, 3, 0.9)
    assert a.heading_at(a.spans[3][0] + 1.0) == "4 stamps make 4 sheets"
    assert "40 numbers, used everywhere" in [a.heading(3, u) for u in (0.5, 0.6, 0.7)]
    assert "next: MaxPool: keep the biggest" in texts(a, 3, 0.97)
    assert "untrained" in a.evidence_for(0)


# ---------------------------------------------------------------------- 192
@needs_mnist
def test_pooling():
    a, s = built("blocks-pooling")
    m = s.metadata
    assert {"26 × 26", "squares of 2 × 2,", "side by side"} <= set(texts(a, 0, 0.9))
    one = texts(a, 1, 0.9)
    for v in np.ravel(m["square"]):
        assert B.val(v) in one
    assert {"biggest: %s" % B.val(m["biggest"]), "4 numbers → 1"} <= set(one)
    every = texts(a, 2, 0.95)
    assert {"squares", "169", "13 × 13 = 169", "13 × 13"} <= set(every)
    four = texts(a, 3, 0.8)
    assert {"numbers per digit", "2,704", "→ 676", "a quarter", "0",
            "numbers it learns"} <= set(four)
    assert "→ [64, 4, 26, 26] → [64, 4, 13, 13]" in mono(a, 3, 0.8)
    assert "next: Dropout: switched off at random" in texts(a, 3, 0.97)


# ---------------------------------------------------------------------- 193
@needs_mnist
def test_dropout():
    a, s = built("blocks-dropout")
    m = s.metadata
    assert {"h: 64 numbers", "nn.Dropout(0.2)", "p = 0.2", "1 in 5"} <= set(texts(a, 0, 0.9))
    train = texts(a, 1, 0.9)
    assert {"train", "switched off", "15 of 64",
            "kept: %s × 1.25" % B.val(m["example_in"]),
            "= %s" % B.val(m["example_out"])} <= set(train)
    assert "seen = drop(h)" in mono(a, 1, 0.9)
    assert "12 of 64" in texts(a, 2, 0.3) and "step 1" in texts(a, 2, 0.3)
    end = texts(a, 2, 0.95)
    assert {"15 of 64", "step 2", "over the epoch:", "20.0% off"} <= set(end)
    test = texts(a, 3, 0.9)
    assert {"0 of 64", "same = h"} <= set(test)
    assert "same = drop(h)" in mono(a, 3, 0.9)
    acc = texts(a, 4, 0.9)
    assert {"91.90%", "94.35%", "9,190 of 10,000 right", "9,435 of 10,000 right",
            "still in train()", "245 more right with model.eval()"} <= set(acc)
    assert mono(a, 4, 0.9) == ["model.train()", "model.eval()"]
    assert a.evidence_for(4).startswith("measured on MNIST")

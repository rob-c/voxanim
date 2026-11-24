"""Animations 215-217 (voxanim.mech_blocks_k): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the programs'."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_k as M
from voxanim import study_course_cnn as cnn
from voxanim.next_catalog import build_study

NAMES = ("blocks-cnn-the-model", "blocks-cnn-shapes", "blocks-cnn-forward")
STAYS = {"blocks-cnn-the-model": "nn.Flatten()", "blocks-cnn-shapes": "28",
         "blocks-cnn-forward": "y = 5"}             # a text of the picture that never dips
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_k", name))
    return _built[name]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.5]


def panel(a, phase, u=0.6):
    """The mono text in the code panel (print(model) and the error are mono too, but drawn
    in the picture, left of the panel)."""
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and x.kind == "mono" and x.x >= B.PANEL[0]]


def picture_mono(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and x.kind == "mono" and x.x < B.PANEL[0]]


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
        # the furniture (panel, running line, five stations) and the picture, which stays
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
        assert shown[:len(lines)] == lines == [s for r in code.names()
                                               for s in B.code_lines(r)]
        out = shown[len(lines):]                 # then the output slot, if anything
        assert not out or out[0].startswith("→ ")


def test_the_series_runs_on():
    model, shapes, forward = (M.ANIMATIONS[n] for n in NAMES)
    assert model.LAST == "Common mistakes"          # the end of the MLP's series
    assert model.NEXT == shapes.TITLE and shapes.LAST == model.TITLE
    assert shapes.NEXT == forward.TITLE and forward.LAST == shapes.TITLE
    assert forward.NEXT == "A shared stamp's gradient"


# ---------------------------------------------------------------------- 215
@needs_mnist
def test_the_model():
    a, s = built("blocks-cnn-the-model")
    m = s.metadata
    blocks = texts(a, 0, 0.9)
    assert set(m["names"]) <= set(blocks) and "one model: 8 blocks" in blocks
    assert "model = nn.Sequential(" in panel(a, 0, 0.9)
    assert B.code_file("cnn-model") == "examples/mnist_cnn.py"
    assert "examples/mnist_cnn.py" in texts(a, 0, 0.9)
    shapes = texts(a, 1, 0.9)
    assert set(m["shape_texts"]) <= set(shapes) and "[64, 16, 11, 11]" in shapes
    assert "28 → 26 → 13 → 11 → 5" in shapes and "the side of the maps" in shapes
    assert a.heading(1, 0.5) == "[64, 1, 28, 28] in, [64, 10] out"
    # print(model), in two halves, each beside the blocks it lists
    assert picture_mono(a, 2, 0.45) == m["pages"][0]
    assert picture_mono(a, 2, 0.88) == m["pages"][1]
    assert "\n".join(m["pages"][0] + m["pages"][1]).replace(",\n      ", ", ") == cnn.PRINTED_MODEL
    assert panel(a, 2, 0.6) == ["print(model)"]
    assert not set(m["names"]) & set(texts(a, 2, 0.6))        # the names make room
    assert a.lit(0, 2, 0.3) == 1.0 and a.lit(7, 2, 0.3) < 0.5
    assert a.lit(7, 2, 0.75) == 1.0 and a.lit(0, 2, 0.75) < 0.5
    counts = texts(a, 3, 0.9)
    assert {"80", "1,168", "4,010", "5,258", "numbers the CNN learns",
            "the MLP: 52,650"} <= set(counts)
    assert counts.count("0") == 5
    midway = [t for t in texts(a, 3, 0.3) if t in ("80", "1,248", "5,258")]
    assert "5,258" not in midway                               # still adding
    assert "next: Where the 400 comes from" in texts(a, 3, 0.95)


# ---------------------------------------------------------------------- 216
@needs_mnist
def test_the_shapes():
    a, s = built("blocks-cnn-shapes")
    m = s.metadata
    first = texts(a, 0, 0.6)
    assert {"28", "28 × 28", "784 numbers: one map"} <= set(first)
    assert "→ torch.Size([64, 1, 28, 28])" in panel(a, 0, 0.6)
    sliding = texts(a, 1, 0.2)
    assert "places across" in sliding
    one = texts(a, 1, 0.45)
    assert {"28", "26", "−2", "28 − 2 = 26", "a 3 × 3 stamp loses 2"} <= set(one)
    assert "→ [64, 1, 28, 28] → [64, 8, 26, 26]" in panel(a, 1, 0.45)
    two = texts(a, 1, 0.9)
    assert {"13", "÷2", "26 ÷ 2 = 13"} <= set(two)
    three = texts(a, 2, 0.9)
    assert {"11", "5", "11 ÷ 2 = 5.5 → 5", "the odd row is dropped"} <= set(three)
    assert [t for t in three if t in ("28", "26", "13", "11", "5")] == ["28", "26", "13",
                                                                        "11", "5"]
    assert "→ [64, 16, 11, 11] → [64, 16, 5, 5]" in panel(a, 2, 0.9)
    four = texts(a, 3, 0.9)
    assert {"16 × 5 × 5 = 400", "in_features = 400"} <= set(four)
    assert "    nn.Linear(16 * 5 * 5, 10)," in panel(a, 3, 0.9)
    assert "→ [64, 400] → [64, 10]" in panel(a, 3, 0.9)
    five = texts(a, 4, 0.8)
    assert {"400 arrive", "784 expected"} <= set(five)
    assert picture_mono(a, 4, 0.8) == m["error_lines"]
    assert " ".join(picture_mono(a, 4, 0.8)) == cnn.ERRORS["cnn-mistake-in-features"]
    assert "    nn.Linear(784, 10)," in panel(a, 4, 0.8)
    assert "examples/blocks_parts.py" in five
    assert a.heading(4, 0.5) == "nn.Linear(784, 10): torch stops"
    assert "next: One digit through the CNN" in texts(a, 4, 0.95)


# ---------------------------------------------------------------------- 217
@needs_mnist
def test_forward():
    a, s = built("blocks-cnn-forward")
    m = s.metadata
    first = texts(a, 0, 0.8)
    assert {"y = 5", "28 × 28 numbers"} <= set(first)
    assert "→ torch.Size([1, 28, 28]) 5" in panel(a, 0, 0.8)
    assert "x, y = train_set[0]" in panel(a, 0, 0.8)
    assert a.heading(0, 0.5) == "one digit: train_set[0], a 5"
    assert "→ [1, 1, 28, 28] → [1, 8, 26, 26]" in panel(a, 1, 0.8)
    assert "→ [1, 8, 26, 26] → [1, 8, 13, 13]" in panel(a, 2, 0.8)
    assert "→ [1, 8, 13, 13] → [1, 16, 11, 11]" in panel(a, 3, 0.3)
    assert "→ [1, 16, 11, 11] → [1, 16, 5, 5]" in panel(a, 3, 0.9)
    assert a.heading(3, 0.9) == "pooled: 16 maps of 5 × 5"
    flat = texts(a, 4, 0.3)
    assert "400 numbers" in flat and "→ [1, 16, 5, 5] → [1, 400]" in panel(a, 4, 0.3)
    scores = texts(a, 4, 0.85)
    assert {str(d) for d in range(10)} <= set(scores)
    assert {B.val(m["score"]), B.val(m["second_score"]), "the 5 wins"} <= set(scores)
    assert (B.val(m["score"]), B.val(m["second_score"])) == ("10.16", "8.22")
    assert "→ [1, 400] → [1, 10]" in panel(a, 4, 0.85)
    assert "    nn.Linear(16 * 5 * 5, 10)," in panel(a, 4, 0.85)
    assert "next: A shared stamp's gradient" in texts(a, 4, 0.95)

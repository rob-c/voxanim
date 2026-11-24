"""Animations 188-190 (voxanim.mech_blocks_b): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_b as M
from voxanim import study_course as course
from voxanim.next_catalog import build_study

NAMES = ("blocks-a-linear-layer", "blocks-relu", "blocks-a-model-is-a-stack")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_b", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def panel(a, phase, u=0.6):
    """The mono text in the code panel (print(model)'s output is mono too, but drawn in
    the picture, left of the panel)."""
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
        shown = [s for s in panel(a, k, 0.6) if not s.startswith("→")]
        assert shown == list(code.lines()) == [s for r in code.names()
                                                for s in B.code_lines(r)]


@needs_mnist
def test_the_series_runs_on():
    lin, relu, stack = (M.ANIMATIONS[n] for n in NAMES)
    assert lin.LAST.startswith("Flatten") and lin.NEXT == relu.TITLE
    assert relu.NEXT == stack.TITLE and stack.LAST == relu.TITLE
    assert stack.NEXT.startswith("nn.Conv2d")


# ---------------------------------------------------------------------- 188
@needs_mnist
def test_a_linear_layer():
    a, s = built("blocks-a-linear-layer")
    m = s.metadata
    assert {"x", "784", "numbers in x", "= 28 × 28"} <= set(texts(a, 0, 0.8))
    assert "    nn.Linear(784, 64)," in panel(a, 0, 0.8)
    weights = texts(a, 1, 0.9)
    assert {"50,176", "784 × 64 weights", "+ 64 biases", "64", "bias"} <= set(weights)
    one = texts(a, 2, 0.95)
    j = m["output"]
    assert "row %d" % j in one
    for p in m["pairs"]:
        assert "%s × %s = %s" % (B.val(p["x"]), B.val(p["w"], 3), B.val(p["product"])) in one
    assert {"added: %s" % B.val(m["added"]), "+ bias: %s" % B.val(m["bias"]),
            "= %s" % B.val(m["out"])} <= set(one)
    shown = ("added: %s" % B.val(m["added"]), "+ bias: %s" % B.val(m["bias"]),
             "= %s" % B.val(m["out"]))
    assert shown == ("added: 8.09", "+ bias: −0.02", "= 8.07")
    sweeping = [t for t in texts(a, 2, 0.52) if t.startswith("added: ")]
    assert sweeping and sweeping[0] != "added: %s" % B.val(m["added"])   # still adding
    headings = {a.heading(2, u) for u in (0.05, 0.3, 0.55, 0.9)}
    assert headings == {"one output: row %d" % j, "multiply the matching numbers",
                        "add them all up", "add the bias"}
    out = texts(a, 3, 0.8)
    assert {"out", B.val(m["out"]), "the batch: [64, 64]"} <= set(out)
    assert "→ [64, 784] → [64, 64]" in panel(a, 3, 0.8)
    assert "next: nn.ReLU: negatives become 0" in texts(a, 3, 0.95)


# ---------------------------------------------------------------------- 189
@needs_mnist
def test_relu():
    a, s = built("blocks-relu")
    m = s.metadata
    first = texts(a, 0, 0.8)
    for v in m["before"]:
        assert B.val(v) in first
    assert "%d of them below 0" % m["zeroed"] in first
    assert "    nn.ReLU()," in panel(a, 0, 0.8) and m["code"] == "nn.ReLU(),"
    turned = texts(a, 1, 0.95)
    assert turned.count("0") >= sum(v <= 0 for v in m["after"])
    for v in m["after"]:
        if v > 0:
            assert turned.count(B.val(v)) >= 2                 # before and after
    assert "%d of %d" % (m["zeroed"], m["size"]) in turned and "became 0" in turned
    midway = [t for t in texts(a, 1, 0.3) if " of 64" in t]
    assert midway and midway[0] != "%d of 64" % m["zeroed"]   # still counting
    same = texts(a, 2, 0.9)
    assert {"0", "numbers it learns", "64"} <= set(same) and same.count("64") >= 2
    assert "→ [64, 64] → [64, 64]" in panel(a, 2, 0.9)
    batch = texts(a, 3, 0.85)
    assert {"%s of %s" % (B.commas(m["batch_zeroed"]), B.commas(m["batch_numbers"])),
            "= %d%%" % round(100 * m["batch_share"]),
            "all 60,000 digits: %d%%" % round(100 * m["train_share"])} <= set(batch)
    shown = ("%s of %s" % (B.commas(m["batch_zeroed"]), B.commas(m["batch_numbers"])),
             "= %d%%" % round(100 * m["batch_share"]))
    assert shown == ("1,954 of 4,096", "= 48%")
    assert "next: A model is a stack of blocks" in texts(a, 3, 0.95)


# ---------------------------------------------------------------------- 190
@needs_mnist
def test_a_model_is_a_stack():
    a, s = built("blocks-a-model-is-a-stack")
    m = s.metadata
    blocks = texts(a, 0, 0.9)
    assert set(m["names"]) <= set(blocks) and "one model: 6 blocks" in blocks
    assert "model = nn.Sequential(" in panel(a, 0, 0.9)
    shapes = texts(a, 1, 0.9)
    assert set(m["shape_texts"]) <= set(shapes)
    assert "784 → 64 → 32 → 10" in shapes
    assert a.heading(1, 0.5) == "[64, 1, 28, 28] in, [64, 10] out"
    printed = picture_mono(a, 2, 0.6)
    assert printed == m["printed_lines"]
    assert "\n".join(printed).replace(",\n      ", ", ") == course.PRINTED_MODEL
    assert panel(a, 2, 0.6) == ["print(model)"]
    assert not set(m["names"]) & set(texts(a, 2, 0.6))        # the names make room
    counts = texts(a, 3, 0.85)
    assert {"0", "50,240", "2,080", "330", "52,650", "numbers the model learns",
            "as in “Counting the weights: a DNN and a CNN”"} <= set(counts)
    assert "→ 52650" in panel(a, 3, 0.85)
    assert B.code_file("count") == "examples/blocks_parts.py"
    assert "examples/blocks_parts.py" in texts(a, 3, 0.85)
    assert "next: nn.Conv2d: one small stamp" in texts(a, 3, 0.95)

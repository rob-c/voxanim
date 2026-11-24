"""Animations 221-223 (voxanim.mech_blocks_m): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_m as M
from voxanim import study_course_cnn as cnn
from voxanim.next_catalog import build_study

NAMES = ("blocks-cnn-against-the-mlp", "blocks-cnn-what-it-gets-wrong",
         "blocks-cnn-common-mistakes")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_m", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


def output(a, phase, u=0.9):
    """The output slot's lines, joined again (the first starts "→ ", the rest "  ")."""
    out = [s for s in mono(a, phase, u) if s.startswith("→ ")]
    lines = mono(a, phase, u)
    i = lines.index(out[0])
    rest = [s[2:] for s in lines[i + 1:] if s.startswith("  ") and not s.startswith("   ")]
    return " ".join([out[0][2:]] + rest)


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
    assert 45.0 <= a.t_done <= 61.5 and a.duration <= 66.0 and len(a.PHASES) >= 4

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
        if a.station_for(k) == a.station_for(k + 1) is not None:
            assert "you are here" in shown
        if a.evidence_for(k) == a.evidence_for(k + 1):     # a new tag fades over
            assert a.evidence_for(k) in shown
        if a.code_for(k).lines() == a.code_for(k + 1).lines():   # new lines fade over
            assert any(x.kind == "mono" and x.opacity > 0.5 for x in sc.texts)
        # the furniture (panel, five stations) and what the picture keeps
        assert len(sc.quads) >= 8
        assert sum(x.opacity > 0.5 and bool(x.s.strip()) for x in sc.texts) >= 8


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
        out = B._pick(code.out, 0.6, "")
        more = ["  " + s for s in out.split("\n")[1:]] if out else []
        shown = [s for s in mono(a, k, 0.6) if not s.startswith("→") and s not in more]
        assert shown == list(code.lines())
        lines = [s for n in code.names() for s in B.code_lines(n)]
        assert all(s in lines for s in shown)
        # one file per window: the panel names it
        assert len({B.code_file(n) for n in code.names()}) == 1


@needs_mnist
def test_the_titles_are_the_catalogues():
    from voxanim.next_catalog import ITEMS
    titles = {i["name"]: i["title"] for i in ITEMS}
    for name in NAMES:
        assert titles[name] == M.ANIMATIONS[name].TITLE


# ---------------------------------------------------------------------- 221
@needs_mnist
def test_the_cnn_against_the_mlp():
    a, s = built("blocks-cnn-against-the-mlp")
    m = s.metadata
    for k in range(4):                                   # said on screen all through
        assert "one run each" in texts(a, k, 0.6)
    one = texts(a, 0, 0.6)
    assert {"MLP", "numbers", "52,650", "= 250 numbers"} <= set(one)
    assert "5,258" not in one and "    nn.Linear(784, 64)," in mono(a, 0, 0.6)
    two = texts(a, 1, 0.6)
    assert {"CNN", "52,650", "5,258", "10× fewer"} <= set(two)
    assert a.heading(1, 0.6) == "the CNN: 5,258 numbers, 10 times fewer"
    assert "    nn.Conv2d(1, 8, 3), nn.ReLU()," in mono(a, 1, 0.6)
    three = texts(a, 2, 0.9)
    assert {"MLP 94.9%", "CNN 96.4%", "94.9%", "96.4%", "accuracy",
            "held-out accuracy"} <= set(three)
    assert "        optimizer.step()" in mono(a, 2, 0.9)
    four = texts(a, 3, 0.9)
    assert {"missed", "506", "364", "117", "33", "131", "64", "missed, by digit"} <= set(four)
    assert "→ 0.9636" in mono(a, 3, 0.9) and m["printed_cnn"] == "0.9636"
    assert "next: what the CNN gets wrong" in texts(a, 3, 0.9)
    assert "one run each" in a.subtitle_for(2)
    # the stacks are to scale: tiles of 250, ten to a row
    assert m["mlp_tiles"] == [210, 0.6] and m["cnn_tiles"] == [21, pytest.approx(0.032)]


# ---------------------------------------------------------------------- 222
@needs_mnist
def test_what_the_cnn_gets_wrong():
    a, s = built("blocks-cnn-what-it-gets-wrong")
    m = s.metadata
    one = texts(a, 0, 0.6)
    assert {"364", "missed", "of 10,000", "131", "71", "3 → 2", "4 → 9", "8 → 3", "8 → 0",
            "the first four missed: true → said"} <= set(one)
    assert "→ 0.9636" in mono(a, 0, 0.6)
    for k, (t, said, n) in enumerate(((8, 9, 36), (2, 7, 33), (8, 0, 22)), 1):
        shown = texts(a, k, 0.5)
        assert {"%d said %d" % (n, said), "%d → %d" % (t, said), "364"} <= set(shown)
        assert shown.count("%d → %d" % (t, said)) == 4
        assert "%d times" % n in a.heading(k, 0.5)
        assert "        pred = model(x).argmax(1)" in mono(a, k, 0.5)
    assert a.heading(1, 0.5) == "an 8 read as a 9: 36 times"
    # only the example looked at by eye is said to read the CNN's way
    assert "the 4th: a person might say 9 too" in texts(a, 1, 0.9)
    assert not any("a person" in t for k in (2, 3) for t in texts(a, k, 0.9))
    assert "the first four in the test set" in texts(a, 2, 0.9)
    assert "%d of the 364" % m["top_total"] in a.subtitle_for(3)
    assert "next: three broken CNN programs" in texts(a, 3, 0.95)


# ---------------------------------------------------------------------- 223
@needs_mnist
def test_three_cnn_mistakes():
    a, s = built("blocks-cnn-common-mistakes")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"nn.Linear(784, 10) wants 784", "after Flatten: 16 × 5 × 5 = 400",
            "[64, 400]: 64 rows of 400", "in_features: 16 × 5 × 5 = 400"} <= set(one)
    assert {"→ RuntimeError: addmm: mat1 and mat2", "  shapes cannot be multiplied",
            "  (64x400 and 784x10)"} <= set(mono(a, 0, 0.9))
    assert "    nn.Linear(784, 10)," in mono(a, 0, 0.9)
    two = texts(a, 1, 0.9)
    assert {"a stamp: 3 × 3 × 3", "wants 3 channels", "x: [64, 1, 28, 28]",
            "a grey digit: 1 channel", "in_channels: 1 for a grey digit"} <= set(two)
    three = texts(a, 2, 0.9)
    assert {"x[0]: [1, 28, 28]", "Flatten: [16, 25]", "16 rows of 25",
            "nn.Linear wants rows of 400"} <= set(three)
    four = texts(a, 3, 0.9)
    assert {"x[0:1]: [1, 1, 28, 28]", "Flatten: [1, 400]", "scores: [1, 10]",
            "keep the batch: x[0:1], not x[0]"} <= set(four)
    assert "→ torch.Size([1, 10])" in mono(a, 3, 0.9)
    five = texts(a, 4, 0.6)
    assert {"x", "[64, 1, 28, 28]", "after Flatten", "[64, 400]", "scores", "[64, 10]",
            "in_features: 16 × 5 × 5 = 400", "in_channels: 1 for a grey digit",
            "keep the batch: x[0:1], not x[0]"} <= set(five)
    assert "    nn.Linear(16 * 5 * 5, 10)," in mono(a, 4, 0.6)
    assert "next: Why maps shrink" in texts(a, 4, 0.9)
    # each error in the output slot is torch's, word for word
    for k, name in enumerate(("cnn-mistake-in-features", "cnn-mistake-channels",
                              "cnn-mistake-no-batch")):
        assert output(a, k) == "RuntimeError: " + cnn.ERRORS[name] == m["errors"][name]

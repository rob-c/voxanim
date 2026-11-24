"""Animations 212-214 (voxanim.mech_blocks_j): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_j as M
from voxanim.next_catalog import build_study

NAMES = ("blocks-one-epoch-fast", "blocks-the-whole-program", "blocks-common-mistakes")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_j", name))
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


# ---------------------------------------------------------------------- 212
@needs_mnist
def test_one_epoch_fast():
    a, s = built("blocks-one-epoch-fast")
    m = s.metadata
    start = texts(a, 0, 0.6)
    assert {"first layer: 4 of 64 units", "step 0 of 938", "2.32", "7.6%", "loss",
            "held-out accuracy"} <= set(start)
    assert a.heading(0, 0.8) == "60,000 digits ÷ 64 → 938 steps"
    assert "for epoch in range(1):" in mono(a, 0, 0.6)
    # the counter runs, never blank, and the units show the last kept step
    for u in (0.3, 0.5, 0.7):
        n = a.step_at(1, u)
        assert 0 < n < 938
        assert "step %s of 938" % B.commas(n) in texts(a, 1, u)
        kept = a.shown_step(1, u)
        assert kept <= n and kept in m["keep"]
    assert "step 938 of 938" in texts(a, 1, 0.97)
    # before and after
    assert "step 0 of 938" in texts(a, 2, 0.3)
    assert a.heading(2, 0.3) == "at the start: random noise"
    assert a.heading(2, 0.9) == "after one epoch: strokes"
    after = texts(a, 2, 0.9)
    assert "next-door alike: 0.0 → 0.4" in after and "step 938 of 938" in after
    end = texts(a, 3, 0.6)
    assert {"0.16", "94.9%", "2.32", "7.6%"} <= set(end)
    assert "→ " + m["printed"] in mono(a, 3, 0.6) and m["printed"] == "0 0.1583479940891266"
    assert "    print(epoch, loss.item())" in mono(a, 3, 0.6)
    assert "next: A dense layer learns to be local" in texts(a, 3, 0.9)


# ---------------------------------------------------------------------- 213
@needs_mnist
def test_the_whole_program():
    a, s = built("blocks-the-whole-program")
    m = s.metadata
    assert {"device: GPU or CPU", "all 63 lines"} <= set(texts(a, 0, 0.8))
    assert 'device = "cuda" if use_gpu else "cpu"' in mono(a, 0, 0.8)
    assert {"train_set", "60,000", "test_set", "10,000"} <= set(texts(a, 1, 0.9))
    assert "64" in texts(a, 2, 0.9)
    assert "→ torch.Size([1, 28, 28]) 5" in mono(a, 2, 0.9)
    assert "model" in texts(a, 3, 0.9) and ").to(device)" in mono(a, 3, 0.9)
    assert {"loss_fn", "optimizer"} <= set(texts(a, 4, 0.9))
    assert "2.32" in texts(a, 5, 0.5)
    assert {"× 938 steps", "0.16"} <= set(texts(a, 5, 0.97))
    assert "9,494 right" in texts(a, 6, 0.9) and "→ 0.9494" in mono(a, 6, 0.9)
    assert {"mnist.pt", "52,650"} <= set(texts(a, 7, 0.5))
    assert "next: Training on a CPU and on a GPU" in texts(a, 7, 0.9)
    # the loop map follows the running line round the loop
    for u, station in ((0.2, "data"), (0.35, "forward"), (0.5, "loss"), (0.6, "backward"),
                       (0.85, "step")):
        assert a._tl_station.at(a.T(5, u)) == station
    # every line of the program is on its map, and every line but the imports and two
    # prints passes through the panel, in order
    panel = [i for k in range(len(a.PHASES)) for i in a.windows[k]]
    assert panel == sorted(panel) and len(panel) == m["lines"] - len(m["not_shown"])
    for k in range(len(a.PHASES)):
        assert [m["program"][i] for i in a.windows[k]] == list(a.code_for(k).lines())


# ---------------------------------------------------------------------- 214
@needs_mnist
def test_common_mistakes():
    a, s = built("blocks-common-mistakes")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {".grad: 37.6", "one batch: 0.28", "with zero_grad", "without", "94.9%",
            "10.1%", "held-out accuracy"} <= set(one)
    assert "loss.backward()" in mono(a, 0, 0.9)
    two = texts(a, 1, 0.9)
    assert {"model.eval()", "no eval()", "64 of 64 on", "%d of 64 off" % m["no_eval"]["off"],
            "94.4%", "91.9%", "with eval()"} <= set(two)
    three = texts(a, 2, 0.9)
    assert {"lowest possible: 1.46", "85.0%", "one softmax", "softmax twice"} <= set(three)
    assert "    nn.Softmax(dim=1)," in mono(a, 2, 0.9)
    four = texts(a, 3, 0.9)
    assert {"nn.Linear(784, 64) wants rows of 784", "without Flatten, a row is 28",
            "64 × 1 × 28 = 1,792 rows of 28", "an error: no result"} <= set(four)
    out = mono(a, 3, 0.9)
    assert "→ RuntimeError: addmm: mat1 and mat2" in out
    assert "  (1792x28 and 784x64)" in out
    assert "".join(a.error_lines).replace("mat2shapes", "mat2 shapes").replace(
        "multiplied(", "multiplied (") == m["no_flatten"]["error"]
    five = texts(a, 4, 0.5)
    assert {"zero_grad forgotten", "10.1%", "eval() forgotten", "91.9%", "softmax twice",
            "85.0%", "no Flatten", "an error", "the script"} <= set(five)
    assert "next: The CNN, block by block" in texts(a, 4, 0.9)

"""Animations 230-231 (voxanim.mech_blocks_p): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the program's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist, needs_text

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_p as M
from voxanim import study_padding as P
from voxanim.next_catalog import build_study

BATCH, LINES = "padding-a-batch-is-a-block", "padding-lines-of-text"
NAMES = (pytest.param(BATCH, marks=needs_mnist), pytest.param(LINES, marks=needs_text))
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_p", name))
    return _built[name]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.5]


def panel(a, phase, u=0.6):
    """The mono text in the code panel."""
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and x.kind == "mono" and x.x >= B.PANEL[0]]


def stays(name, shown):
    """A text of the picture that never dips between phases: the sizes under the four
    cards (230), the lines' lengths (231)."""
    if name == BATCH:
        return any(s.endswith(" × 20") or s.endswith(" × 17") or s == "28 × 28" for s in shown)
    return {"14", "45", "50", "19"} <= set(shown)


# ---------------------------------------------------------------------- both of them
@pytest.mark.parametrize("name", NAMES)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


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


@pytest.mark.parametrize("name", NAMES)
def test_nothing_blanks_between_phases(name):
    a, _ = built(name)
    for k, (_lo, hi) in enumerate(a.spans[:-1]):
        sc = a.scene(hi)                                   # the content's dip
        shown = [x.s for x in sc.texts if x.opacity > 0.5]
        if a.station_for(k) == a.station_for(k + 1):
            assert "you are here" in shown
        if a.evidence_for(k) == a.evidence_for(k + 1):
            assert a.evidence_for(k) in shown
        if a.code_for(k).lines() == a.code_for(k + 1).lines():
            assert any(x.kind == "mono" and x.opacity > 0.5 for x in sc.texts)
        assert len(sc.quads) >= 8
        assert stays(name, shown), (k, shown)


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


@pytest.mark.parametrize("name", NAMES)
def test_the_code_on_screen_is_the_programs(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        lines = list(code.lines())
        shown = panel(a, k, 0.6)
        assert shown[:len(lines)] == lines
        every = [s for r in code.names() for s in B.code_lines(r)]
        assert lines == (every if code.pick is None else [every[i] for i in code.pick])
        out = shown[len(lines):]
        assert not out or out[0].startswith("→ ")
        assert B.code_file(code.names()[0]) == "examples/padding.py"


def test_the_series_runs_on():
    batch, lines = M.ANIMATIONS[BATCH], M.ANIMATIONS[LINES]
    assert batch.LAST == "The CNN with padding, measured"
    assert batch.NEXT == lines.TITLE and lines.LAST == batch.TITLE
    assert lines.NEXT == "Masks: ignore the padding"
    assert (batch.TITLE, lines.TITLE) == ("A batch is one block", "Padding lines of text")


def test_padding_is_striped_and_the_rest_is_not():
    rgb = np.full((2, 2, 3), 0.9, np.float32)
    pad = np.array([[False, True], [False, True]])
    tex = M.hatched(rgb, pad, 3, 3, 6)
    assert tex.shape == (6, 6, 3)
    assert np.all(tex[:, :3] == 0.9)                     # the real tiles are untouched
    assert np.any(tex[:, 3:] < 0.8)                      # the pads carry stripes
    assert np.all(M.hatched(rgb, None, 3, 3, 6) == 0.9)


# ---------------------------------------------------------------------- 230
@needs_mnist
def test_a_batch_is_a_block():
    a, s = built(BATCH)
    m = s.metadata
    # 1: the four digits, 28 x 28, then cut to their ink
    assert texts(a, 0, 0.5).count("28 × 28") == 4
    assert "cut: the box round the ink" in texts(a, 0, 0.5)
    assert set(m["sizes"]) <= set(texts(a, 0, 0.9))
    assert panel(a, 0, 0.9)[-2:] == ["→ [[1, 20, 20], [1, 20, 17],", "  [1, 20, 20], [1, 20, 14]]"]
    assert "small = [cut(x[i]) for i in range(4)]" in panel(a, 0, 0.1)
    # 2: torch.stack's own error, and what it means
    two = texts(a, 1, 0.8)
    assert {"entry 0", "entry 1", "entry 2", "entry 3", "torch.stack needs one size",
            "entry 1: 20 × 17, not 20 × 20", "red: holes, no numbers there"} <= set(two)
    shown = panel(a, 1, 0.8)
    assert shown[-3:] == ["→ " + m["error_lines"][0]] + ["  " + s for s in m["error_lines"][1:]]
    assert " ".join(s[2:] for s in shown[-3:]) == "RuntimeError: " + P.STACK_ERROR
    assert a.heading(1, 0.5) == "torch.stack stops: the sizes differ"
    # 3: padded with the paper's value, one card at a time
    three = texts(a, 2, 0.3)
    assert "padding: −0.42, the paper's value" in three
    assert "→ BG = -0.4242" in panel(a, 2, 0.3)
    done = texts(a, 2, 0.9)
    assert done.count("28 × 28") == 4
    assert {"+384 pads", "+444 pads", "+504 pads"} <= set(done)
    assert "    return F.pad(img, room, value=BG)" in panel(a, 2, 0.9)
    # 4: one block, and how much of it is padding
    early = texts(a, 3, 0.1)
    assert "384 of 784" in early and "1,716 of 3,136" not in early
    four = texts(a, 3, 0.8)
    assert {"[4, 1, 28, 28]", "one block", "3,136 numbers", "1,716 of 3,136",
            "numbers that are padding", "55% padding"} <= set(four)
    assert "→ padded: torch.Size([4, 1, 28, 28])" in panel(a, 3, 0.8)
    assert "batch = torch.stack(padded)" in panel(a, 3, 0.8)
    assert a.heading(3, 0.6) == "any input: one full block, no holes"
    assert "next: Padding lines of text" in texts(a, 3, 0.95)


@needs_mnist
def test_the_cards_are_drawn_to_scale():
    a, s = built(BATCH)
    m = s.metadata
    for i, (_c, h, w) in enumerate(m["shapes"]):
        img, pad, left, top = a.state(i, 1, 0.5)          # cut, in the slot's corner
        assert img.shape == (h, w) and pad is None
        assert (left, top) == (a.left(i), a.TOP)
        img, pad, _l, _t = a.state(i, 2, 1.0)             # padded to 28 x 28
        assert img.shape == (28, 28) and int(pad.sum()) == m["pads"][i]
        img, _pad, _l, _t = a.state(i, 0, 0.0)            # the whole digit first
        assert img.shape == (28, 28)


# ---------------------------------------------------------------------- 231
@needs_text
def test_lines_of_text():
    a, s = built(LINES)
    m = s.metadata
    one = texts(a, 0, 0.8)
    assert set(m["text"]) <= set(one)
    assert "4 to 50" in one and "characters per line" in one
    assert "→ lengths 14, 45, 4, 13, 14, 50, 4, 19" in panel(a, 0, 0.8)
    two = texts(a, 1, 0.8)
    assert "F → 19   i → 48   r → 57" in two and "id 0 is kept for padding" in two
    assert not set(m["text"]) & set(two)                 # the text has become tiles
    assert panel(a, 1, 0.8)[:4] == ['vocab = ["<pad>"] + sorted(set(text))',
                                    "ids = {c: i for i, c in enumerate(vocab)}",
                                    "seqs = [torch.tensor([ids[c] for c in s])",
                                    "        for s in lines[:8]]"]
    assert "→ seqs[0] = tensor([19, 48, 57, ...])" in panel(a, 1, 0.8)
    assert a.heading(2, 0.1) == "the longest line: 50 characters"
    assert a.heading(2, 0.8) == "pad with 0 up to 50"
    three = texts(a, 2, 0.9)
    assert {"[8, 50]", "one block", "a pad: id 0", "237 pads"} <= set(three)
    assert "→ torch.Size([8, 50])" in panel(a, 2, 0.9)
    early = texts(a, 3, 0.3)
    assert "163 of 400" not in early
    four = texts(a, 3, 0.9)
    assert {"163 of 400", "real characters", "True: a character", "False: a pad",
            "59% padding"} <= set(four)
    assert panel(a, 3, 0.9)[-2:] == ["→ lines: torch.Size([8, 50])", "  163 real of 400"]
    assert "mask = batch != 0" in panel(a, 3, 0.9)
    assert "next: Masks: ignore the padding" in texts(a, 3, 0.95)

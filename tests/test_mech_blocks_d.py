"""Animations 194-196 (voxanim.mech_blocks_d): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the parts file's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_d as M
from voxanim.next_catalog import build_study

NAMES = ("blocks-normalisation", "blocks-embedding", "blocks-attention")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_d", name))
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
        if a.evidence_for(k) == a.evidence_for(k + 1):
            assert a.evidence_for(k) in shown
        if a.code_for(k).lines() == a.code_for(k + 1).lines():
            assert any(x.kind == "mono" and x.opacity > 0.5 for x in sc.texts)
        # the furniture (panel, running line, five stations: 7) and the picture's block
        # (its tiles and its frame) stay
        assert len(sc.quads) >= 9


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
def test_the_code_on_screen_is_the_parts_file(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        assert B.code_file(code.names()[0]) == "examples/blocks_parts.py"
        shown = [s for s in mono(a, k, 0.6) if not s.startswith(("→", "  "))]
        assert shown == [s for s in code.lines() if not s.startswith("  ")]
        assert list(code.lines()) == [s for n in code.names() for s in B.code_lines(n)]


# ---------------------------------------------------------------------- 194
@needs_mnist
def test_normalisation():
    a, s = built("blocks-normalisation")
    m = s.metadata
    h = texts(a, 0, 0.9)
    assert {"[64, 64]", "a row: a digit", "a column: a feature", "column means"} <= set(h)
    assert h.count("64") >= 2
    assert mono(a, 0, 0.9) == list(B.code_lines("batchnorm")) + list(B.code_lines("layernorm"))
    col = texts(a, 1, 0.25)
    assert {"column %d" % m["col"], B.val(m["col_before"][0]), B.val(m["col_before"][1]),
            "mean", "spread"} <= set(col)
    done = texts(a, 1, 0.95)
    assert {"0.00", "1.00", "all 64 columns"} <= set(done)
    assert "→ torch.Size([64, 64])" in mono(a, 1, 0.95)
    row = texts(a, 2, 0.28)
    assert {"row %d" % m["row"], B.val(m["row_before"][0]), B.val(m["row_before"][1]),
            "row means"} <= set(row)
    assert {"0.00", "1.00", "all 64 rows"} <= set(texts(a, 2, 0.95))
    both = texts(a, 3, 0.95)
    lo, hi = m["bn_row_means_range"]
    lo2, hi2 = m["ln_col_means_range"]
    assert {"BatchNorm", "LayerNorm", "column means: 0", "row means: 0", "row means:",
            "column means:", "%s to %s" % (B.val(lo), B.val(hi)),
            "%s to %s" % (B.val(lo2), B.val(hi2))} <= set(both)
    assert "next: nn.Embedding: a table you look up" in both
    assert "2.33" in a.subtitle_for(1) and "1.16" in a.subtitle_for(2)


# ---------------------------------------------------------------------- 195
@needs_mnist
def test_embedding():
    a, s = built("blocks-embedding")
    m = s.metadata
    table = texts(a, 0, 0.9)
    assert {"10 rows of 16", "160", "numbers", "it learns", "16"} <= set(table)
    assert {str(r) for r in range(10)} <= set(table)
    first = texts(a, 1, 0.4)
    assert "y = %d" % m["first"] in first
    assert "  ".join(B.val(v) for v in m["first_row"][:4]) + "  …" in first
    second = texts(a, 1, 0.95)
    row8 = np.asarray(s.panels[0].data["values"])[m["second"]]
    assert {"y = %d" % m["second"], "multiplications", "0"} <= set(second)
    assert "  ".join(B.val(v) for v in row8[:4]) + "  …" in second
    batch = texts(a, 2, 0.95)
    assert {"64", "lookups", "[64] → [64, 16]", "y", "embed(y)"} <= set(batch)
    assert "→ torch.Size([64, 16])" in mono(a, 2, 0.95)
    assert any(t.isdigit() and 0 < int(t) < 64 for t in texts(a, 2, 0.3))   # rows arriving
    hot = texts(a, 3, 0.95)
    assert {"one-hot of 8", "1", "× 0", "row 8", "160", "one-hot × table", "lookup", "0"} <= set(
        hot)
    assert "next: nn.MultiheadAttention as blocks" in hot
    assert "144 of them by 0" in a.subtitle_for(3)


# ---------------------------------------------------------------------- 196
@needs_mnist
def test_attention():
    a, s = built("blocks-attention")
    m = s.metadata
    tokens = texts(a, 0, 0.9)
    assert {"[64, 5, 16]", "5 tokens of 4 × 4 = 16", "a 0 from the first batch", "x"} <= set(
        tokens)
    assert "x: 5 patches of a digit, not randn" in texts(a, 0, 0.5)
    qkv = texts(a, 1, 0.9)
    assert {"Q", "K", "V", "x", "3 grids of 16 × 16", "2 heads: 8 + 8"} <= set(qkv)
    i, j = m["query"], m["key"]
    one = texts(a, 2, 0.3)
    assert "q · k ÷ √8 = %s ÷ %s = %s" % (B.val(m["dot"]), B.val(m["scale"]),
                                          B.val(m["raw_cell"])) in one
    assert "head 1" in one
    row = texts(a, 2, 0.95)
    assert "row %d: " % (i + 1) + "  ".join(B.val(v) for v in m["query_row"]) in row
    assert {"adds up to 1.00", "head 1", "head 2", "scores: the 2 grids, averaged"} <= set(row)
    assert "→ scores: torch.Size([64, 5, 5])" in mono(a, 2, 0.95)
    out = texts(a, 3, 0.95)
    p = m["query_row"]
    assert {"out", "out_proj", "V", "head 1", "head 2",
            "head 1: %s V1 + %s V2 + …" % (B.val(p[0]), B.val(p[1])),
            "the count: “Counting the weights: a transformer”",
            "a real one: “Inside GPT-2”", "next: Logits and softmax"} <= set(out)
    assert "Q" not in out and "K" not in out
    assert "→ out: torch.Size([64, 5, 16])" in mono(a, 3, 0.95)
    assert B.val(p[j]) in a.subtitle_for(3) and "token %d" % (j + 1) in a.subtitle_for(3)

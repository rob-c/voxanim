"""Animations 261-262 (voxanim.mech_blocks_za): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers, tokens and code lines on screen are the study's and the script's.
They need no download."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pytest

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_za as M
from voxanim.next_catalog import build_study

NAMES = ("thinking-tokens", "thinking-effort")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_za", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


def digit_rows(a, phase, u):
    """The 0s and 1s in the picture's token columns (not the effort's), row by row, left
    to right, keyed by their height."""
    lo, hi = a.spans[phase]
    rows = defaultdict(list)
    left = (M.col(0) + M.col(1)) / 2
    for x in a.scene(lo + (hi - lo) * u).texts:
        if x.opacity > 0.5 and x.s in ("0", "1") and left < x.x < B.PANEL[0]:
            rows[round(x.y, 3)].append((x.x, int(x.s)))
    return {y: [v for _x, v in sorted(r)] for y, r in sorted(rows.items())}


# ---------------------------------------------------------------------- every one of them
@pytest.mark.parametrize("name", NAMES)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


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


@pytest.mark.parametrize("name", NAMES)
def test_nothing_blanks_between_phases(name):
    a, _ = built(name)
    for k, (_lo, hi) in enumerate(a.spans[:-1]):
        sc = a.scene(hi)                                   # the content's dip
        shown = [x.s for x in sc.texts if x.opacity > 0.5]
        if a.evidence_for(k) == a.evidence_for(k + 1):     # a new tag fades over
            assert a.evidence_for(k) in shown
        if a.code_for(k).lines() == a.code_for(k + 1).lines():   # new lines fade over
            assert any(x.kind == "mono" and x.opacity > 0.5 for x in sc.texts)
        # the furniture (panel, five stations) and what the picture keeps
        assert len(sc.quads) >= 8
        assert sum(x.opacity > 0.5 and bool(x.s.strip()) for x in sc.texts) >= 8
        # the picture itself is never empty: something of it stays at full strength
        assert any(x.opacity > 0.5 and x.x < B.PANEL[0] and x.y > B.HEAD_Y + 0.03
                   for x in sc.texts), (name, k)


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
        tag = a.evidence_for(k)
        assert layout.text_width(tag, B.SIZE) <= 0.95 - B.PANEL[0]
        assert tag.startswith(("measured on a tiny transformer: ", "exact arithmetic: ",
                               "about the data: "))


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
def test_the_code_on_screen_is_the_scripts(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        shown = [s for s in mono(a, k, 0.6) if not s.startswith("→")]
        assert shown == list(code.lines())
        lines = [s for n in code.names() for s in B.code_lines(n)]
        assert all(s in lines for s in shown)
        assert {B.code_file(n) for n in code.names()} == {"examples/thinking.py"}


def test_the_titles_and_the_chain_are_the_catalogues():
    from voxanim.next_catalog import ITEMS
    titles = {i["name"]: i["title"] for i in ITEMS}
    for name in NAMES:
        assert titles[name] == M.ANIMATIONS[name].TITLE
    assert titles["hosting-a-coding-agent"] == M.ThinkingTokens.LAST
    assert M.ThinkingTokens.NEXT == titles["thinking-effort"] == M.ThinkingEffort.TITLE
    assert titles["thinking-tokens"] == M.ThinkingEffort.LAST
    assert M.ThinkingEffort.NEXT == "The cost of context: attention, RNN, SSM"


@pytest.mark.parametrize("name", NAMES)
def test_nothing_here_is_training_and_nothing_is_timed(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        assert a.station_for(k) is None
    words = " ".join([a.subtitle_for(k) for k in range(len(a.PHASES))]
                     + [t for k in range(len(a.PHASES)) for t in texts(a, k, 0.95)])
    for w in ("second", "faster", "slower", " ms", "$", "price", "GPU"):
        assert w not in words, w


# ---------------------------------------------------------------------- 261
def test_thinking_tokens():
    a, s = built("thinking-tokens")
    m = s.metadata
    for k in range(5):                                   # the question stays all through
        assert {"effort", "12 bits it never saw", "="} <= set(texts(a, k, 0.6))
        assert next(iter(digit_rows(a, k, 0.6).values())) == m["bits"]
    one = texts(a, 0, 0.9)
    assert {"answer: 1 if the count of 1s is odd", "this string: 3 ones, odd: 1",
            "the tokens it writes will go here", "0"} <= set(one)
    assert "seq = [3 + e] + bits + [EQUALS]" in mono(a, 0)
    assert "held-out string 1 of 468" in a.evidence_for(0)
    # effort 0: one pass, the answer at once, wrong
    two = texts(a, 1, 0.9)
    assert {"the model", "103,114 numbers", "reads", "writes", "pass 1", "reads 14 tokens",
            "answer 0: wrong", "3 ones is odd: 1", "1 token written · 1 pass"} <= set(two)
    assert list(digit_rows(a, 1, 0.9).values())[1] == m["none"]["tokens"] == [0]
    # effort 12: the steps, one pass each, each read back by the next
    for u, n in ((0.3, 1), (0.5, 2), (0.7, 3), (0.95, 4)):
        three = texts(a, 2, u)
        assert "12" in three
        assert {"pass %d" % n, "reads %d tokens" % (13 + n)} <= set(three)
        assert list(digit_rows(a, 2, u).values())[1] == m["deep"]["tokens"][:n]
    assert "a step: 1 if the ones so far are odd" in texts(a, 2, 0.95)
    four = texts(a, 3, 0.95)
    assert {"pass 13", "reads 26 tokens", "answer 1: right", "13 tokens written · 13 passes",
            "all 468 new strings: 12 steps, 465 right", "no steps: 215 right (46%, chance)"
            } <= set(four)
    rows = list(digit_rows(a, 3, 0.95).values())
    assert rows == [m["bits"], m["deep"]["tokens"]]       # its own 13 tokens, under the bits
    assert "    seq.append(nxt)" in mono(a, 3)
    assert "465" in a.subtitle_for(3) and "215 (46%)" in a.subtitle_for(3)
    # a large served model: the same kind of thing
    five = texts(a, 4, 0.95)
    assert {"a large served model", "question", "thinking", "answer",
            "written, read back, counted", "thinking: often", "not shown to you",
            "next: Effort: how much thinking is allowed"} <= set(five)
    assert "the model" not in five                       # the toy's card has gone
    assert a.evidence_for(4).startswith("about the data: ")


# ---------------------------------------------------------------------- 262
def test_thinking_effort():
    a, s = built("thinking-effort")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"effort", "the same 12 bits", "0 steps: 1 token, the answer",
            "3 steps of 4 bits: 4 tokens", "12 steps of 1 bit: 13 tokens",
            "a step: the parity so far"} <= set(one)
    rows = list(digit_rows(a, 0, 0.9).values())
    assert rows == [m["bits"]] + [r["truth"] for r in m["shown"]]
    assert a.evidence_for(0).startswith("exact arithmetic: ")
    assert "        seq.append(sum(bits[:k * g]) % 2)" in mono(a, 0)
    two = texts(a, 1, 0.9)
    assert {"a pass: all 103,114 numbers, once", "passes: 1, 4, 13"} <= set(two)
    assert "    scores = model(torch.tensor([seq]))" in mono(a, 1)
    three = texts(a, 2, 0.95)
    assert {"right, of 468", "tokens written", "chance", "steps"} <= set(three)
    for r in m["results"]:                               # two bars per effort, their numbers
        assert {str(r["right"]), str(r["tokens"]), str(r["effort"])} <= set(three)
    assert "bits each" not in three
    assert "234" in a.subtitle_for(2)
    four = texts(a, 3, 0.9)
    assert {"up to 6 steps: 213 to 241", "bits each", "465", "13"} <= set(four)
    assert "one run" in a.evidence_for(3)
    assert "One run" in a.subtitle_for(3)
    five = texts(a, 4, 0.95)
    assert {"coding agent", "the service", "effort: more", "thinking tokens",
            "the same in large models", "more tokens: more passes, time and cost",
            "helps when each step is one it can do", "what this toy cannot show",
            "by how much it helps", "that more always helps",
            "next: The cost of context: attention, RNN, SSM"} <= set(five)
    assert "effort: less" in texts(a, 4, 0.15) and "effort: less" not in five
    assert "per request" in a.subtitle_for(4)
    assert a.evidence_for(4).startswith("about the data: ")

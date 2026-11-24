"""Animations 258-260 (voxanim.mech_blocks_z): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, hosting is never put on the training loop, and the numbers and code lines on screen
are the study's and the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_gpt2

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_z as M
from voxanim.next_catalog import build_study

NAMES = ("hosting-many-people-one-model", "hosting-memory-per-conversation",
         "hosting-a-coding-agent")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get("gpt2-" + name).build(), build_study("study_blocks_z", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def panel(a, phase, u=0.6):
    """The monospace lines in the code panel (the picture may show text in monospace too:
    the model's text and a file's lines in 260)."""
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and x.kind == "mono" and x.x > B.PANEL[0]]


def picture_mono(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and x.kind == "mono" and x.x < B.PANEL[0]]


# ---------------------------------------------------------------------- every one of them
@needs_gpt2
@pytest.mark.parametrize("name", NAMES)
def test_layout_is_clean(name):
    assert qa.check("gpt2-" + name, samples=24, verbose=False) == 0


@needs_gpt2
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


@needs_gpt2
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


@needs_gpt2
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
        assert a.evidence_for(k).startswith(("measured on GPT-2: ", "exact arithmetic: ",
                                             "about the data: "))
        # counts, not speeds: no time, no price, no product
        assert not any(w in sub.lower() for w in ("second", "faster", "slower", "$", "cost "))


@needs_gpt2
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


@needs_gpt2
@pytest.mark.parametrize("name", NAMES)
def test_the_code_on_screen_is_the_scripts(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        shown = [s for s in panel(a, k, 0.6) if not s.startswith("→")]
        assert shown == list(code.lines())
        lines = [s for n in code.names() for s in B.code_lines(n)]
        assert all(s in lines for s in shown)
        assert {B.code_file(n) for n in code.names()} == {"examples/serve.py"}


@needs_gpt2
@pytest.mark.parametrize("name", NAMES)
def test_hosting_is_not_training(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        assert a.station_for(k) is None
        assert "you are here" not in texts(a, k, 0.6)


@needs_gpt2
def test_the_titles_and_the_chain_are_the_catalogues():
    from voxanim.next_catalog import ITEMS
    titles = {i["name"]: i["title"] for i in ITEMS}
    for name in NAMES:
        assert titles[name] == M.ANIMATIONS[name].TITLE
    assert titles["hosting-read-once-remember"] == M.ManyPeople.LAST
    assert M.ManyPeople.NEXT == titles["hosting-memory-per-conversation"] == M.Memory.TITLE
    assert titles["hosting-many-people-one-model"] == M.Memory.LAST
    assert M.Memory.NEXT == titles["hosting-a-coding-agent"] == M.Agent.TITLE
    assert titles["hosting-memory-per-conversation"] == M.Agent.LAST
    assert titles["thinking-tokens"] == M.Agent.NEXT


# ---------------------------------------------------------------------- 258
@needs_gpt2
def test_many_people():
    a, s = built("hosting-many-people-one-model")
    m = s.metadata
    for k in range(4):                                   # the server and its model stay
        assert {"the server", "one model", "498 MB"} <= set(texts(a, k, 0.6))
    one = texts(a, 0, 0.9)
    assert set(m["questions"]) <= set(one)
    assert {"A", "It", "I", "Hello"} <= set(one)
    assert "written for the animation" in a.subtitle_for(0)
    assert "    send(tok.decode([nxt]))" in panel(a, 0, 0.9)
    # one step each: the weights read once per person, counted
    for u, n in ((0.2, 1), (0.42, 2), (0.65, 3)):
        assert "%d × 498 MB" % n in texts(a, 1, u)
    two = texts(a, 1, 0.97)
    assert {"one each", "weights read for the next tokens", "4 × 498 MB = 1,991 MB", "server",
            "'s", "'m", "!"} <= set(two)
    assert "1,991 MB" in a.subtitle_for(1) and "124,439,808" in a.subtitle_for(1)
    assert "    scores, cache = step(W, [nxt], cache)" in panel(a, 1)
    # one step for all: a row per person, read once, the same tokens
    three = texts(a, 2, 0.95)
    assert {"together", "[4, 768]", "1 × 498 MB, for all 4", "and the same 4 tokens"
            } <= set(three)
    assert three.count("server") == 2 and three.count("=") == 4
    assert "0 × 498 MB" in texts(a, 2, 0.3)
    assert panel(a, 2, 0.9) == ['w = W[p + "attn.c_attn.weight"]', "qkv = h @ w"]
    assert "batch_step" in a.evidence_for(2)
    four = texts(a, 3, 0.95)
    assert {"people", "weights read per token", "1", "64", "498 MB", "249 MB", "7.8 MB",
            "next: Why a conversation has a limit"} <= set(four)
    assert "Why is the sky blue?" not in four             # the people have made room
    assert "Counts, not speeds" in a.subtitle_for(3) and "7.8 MB" in a.subtitle_for(3)
    assert a.evidence_for(3) == "exact arithmetic: 497,759,232 bytes ÷ people in a step"


# ---------------------------------------------------------------------- 259
@needs_gpt2
def test_memory():
    a, s = built("hosting-memory-per-conversation")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"one token", "server", "key", "value", "layer 1", "layer 12", "768 numbers each",
            "12 layers", "× 2: key, value", "× 768 numbers", "× 4 bytes", "kept per token",
            "73,728 bytes"} <= set(one)
    assert "new_cache.append((k, v))" in panel(a, 0)
    for k in range(1, 4):                                # the bytes per token stay
        assert {"kept per token", "73,728 bytes"} <= set(texts(a, k, 0.6))
    two = texts(a, 1, 0.4)
    assert {"one conversation", "0", "512", "1,024", "tokens", "30 tokens: 10 asked, 20 written",
            "2.2 MB"} <= set(two)
    full = texts(a, 1, 0.95)
    assert {"1,024 tokens: full", "75 MB", "GPT-2 has 1,024 places: the limit"} <= set(full)
    assert "2.2 MB" not in full
    assert "[12, 30, 64]" in a.evidence_for(1)
    three = texts(a, 2, 0.9)
    assert {"1 GB of memory: 10 shelves", "100 MB a shelf", "the model: 498 MB",
            "shared by everyone", "a full conversation", "75 MB, its own",
            "6 fit beside the model"} <= set(three)
    assert "W = gpt2.weights()" in panel(a, 2)
    four = texts(a, 3, 0.4)
    assert {"1 GB of cache:", "13,563 tokens", "13 conversations", "of 1,024 tokens"
            } <= set(four)
    later = texts(a, 3, 0.9)
    assert {"or 452 conversations", "of 30 tokens", "next: A coding agent's loop"} <= set(later)
    assert "not published" in a.subtitle_for(3)
    assert a.evidence_for(3) == "exact arithmetic: 10⁹ bytes ÷ 73,728 = 13,563 tokens"
    assert m["full_per_gb"] == 13 and m["short_per_gb"] == 452


# ---------------------------------------------------------------------- 260
@needs_gpt2
def test_coding_agent():
    a, s = built("hosting-a-coding-agent")
    m = s.metadata
    for k in range(5):                                   # the two sides stay, always named
        shown = set(texts(a, k, 0.9))
        assert {"your laptop", "agent", "parser.py", "tests", "the service",
                "only writes text", "the whole conversation, sent each call"} <= shown
    one = texts(a, 0, 0.9)
    assert {"you", "The tests fail in parser.py.", "Please fix it."} <= set(one)
    assert a.evidence_for(0) == "about the data: a conversation written for the animation"
    assert "written for the animation" in a.subtitle_for(0)
    # each call: the whole conversation sent, the model's text back, the tool run on the laptop
    for k, c in enumerate(m["calls"], start=1):
        shown = texts(a, k, 0.9)
        assert "all %d tokens" % c["tokens_sent"] in shown
        assert {"service", "laptop", "text"} <= set(shown)
        assert c["reply_lines"] == [x for x in picture_mono(a, k, 0.9)
                                    if x in c["reply_lines"]]
        for prev in m["calls"][:k]:                       # every call's sending, kept
            assert str(prev["tokens_sent"]) in shown
        assert str(c["tokens_sent"]) in a.subtitle_for(k)
        assert "call %d" % k in a.evidence_for(k) and a.evidence_for(k).startswith(
            "measured on GPT-2: ")
    two = texts(a, 1, 0.9)
    assert "reads parser.py" in two
    assert picture_mono(a, 1, 0.9) == ['{"tool": "read_file",', '"path": "parser.py"}',
                                       "def parse(line):", "    return line.split(',')[1]"]
    assert "edits parser.py" in texts(a, 2, 0.9) and "parser.py: 1 line changed" in \
        picture_mono(a, 2, 0.9)
    assert "runs pytest" in texts(a, 3, 0.9) and "3 passed" in picture_mono(a, 3, 0.9)
    five = texts(a, 4, 0.9)
    assert {"shows you the answer", "no tool: the loop ends", "25", "87", "150", "189",
            "next: Thinking: tokens it writes for itself"} <= set(five)
    assert picture_mono(a, 4, 0.9) == ["Fixed: parse() took the second",
                                       "field; it now takes the first.", "The 3 tests pass."]
    assert "451" in a.subtitle_for(4)
    # the code: the client sends the whole talk; the service only turns text into text
    assert 'request = {"messages": [said],' in panel(a, 1, 0.02)
    assert "reply(W, tok, text, most, send)" in panel(a, 4)

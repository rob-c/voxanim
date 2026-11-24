"""Animations 255-257 (voxanim.mech_blocks_y): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, hosting is not training, and the numbers, tokens and code lines on screen are the
study's and the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_gpt2

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_y as M
from voxanim.mech import dark
from voxanim.next_catalog import build_study

NAMES = ("hosting-text-to-tokens", "hosting-one-token-at-a-time", "hosting-read-once-remember")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get("gpt2-" + name).build(), build_study("study_blocks_y", name))
    return _built[name]


def shown(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.5]


def texts(a, phase, u=0.6):
    return [x.s for x in shown(a, phase, u)]


def picture(a, phase, u=0.6):
    """The texts of the picture (left of the code panel)."""
    return [x.s for x in shown(a, phase, u) if x.x < B.PANEL[0]]


OUT_INK = np.asarray(dark(B.DATA, 0.35), float)           # the output slot's ink


def printed(x):
    return np.allclose(np.asarray(x.color, float), OUT_INK, atol=1e-3)


def code(a, phase, u=0.6):
    """The code panel's lines (mono, right of the picture), the output slot left out."""
    return [x.s for x in shown(a, phase, u) if x.kind == "mono" and x.x > B.PANEL[0]
            and not printed(x)]


def output(a, phase, u=0.6):
    """What the output slot shows."""
    return [x.s for x in shown(a, phase, u) if x.kind == "mono" and printed(x)]


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
        on = [x.s for x in sc.texts if x.opacity > 0.5]
        if a.evidence_for(k) == a.evidence_for(k + 1):     # a new tag fades over
            assert a.evidence_for(k) in on
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
        assert B.is_evidence(a.evidence_for(k))
        # counts, not speeds: no time, price or company anywhere
        for s in [sub, a.evidence_for(k), *texts(a, k, 0.9)]:
            assert not any(w in s.lower() for w in ("second", "faster", "price", "$", "cost "))


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
        c = a.code_for(k)
        assert code(a, k, 0.6) == list(c.lines())
        lines = [s for n in c.names() for s in B.code_lines(n)]
        assert all(s in lines for s in c.lines())
        assert {B.code_file(n) for n in c.names()} == {"examples/serve.py"}


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
    chain = [titles[n] for n in ("hosting-the-model-is-a-file", *NAMES,
                                 "hosting-many-people-one-model")]
    for i, name in enumerate(NAMES, 1):
        cls = M.ANIMATIONS[name]
        assert tuple(chain[i - 1:i + 2]) == (cls.LAST, cls.TITLE, cls.NEXT)


# ---------------------------------------------------------------------- 255
@needs_gpt2
def test_text_to_tokens():
    a, s = built("hosting-text-to-tokens")
    m = s.metadata
    for k in range(4):                                   # the server and the text stay
        assert {"on the server", "User: What is a server?", "Assistant:",
                "34 characters"} <= set(picture(a, k, 0.6))
    one = picture(a, 0, 0.9)
    assert {"the request (an example): 82 bytes", '{"messages": [{"role": "user",',
            '"content": "What is a server?"}],', '"max_tokens": 20}', "prompt(messages)"
            } <= set(one)
    assert 'text = prompt(request["messages"])' in code(a, 0)
    assert "example" in a.evidence_for(0)
    # the tokens, as the tokenizer cut them
    two = picture(a, 1, 0.9)
    assert set(m["shown"]) <= set(two)
    assert {"␣What", "␣server", "↵"} <= set(two)
    assert {"34 characters → 10 tokens", "␣ a space    ↵ a new line"} <= set(two)
    assert not {str(i) for i in m["ids"]} & set(two)        # no numbers yet
    assert "ids = tok.encode(text)" in code(a, 1)
    assert "differently" in a.subtitle_for(1)
    # the ids under the tokens, and the printed list
    three = picture(a, 2, 0.9)
    assert [str(i) for i in m["ids"]] == [t for t in three if t.isdigit()]
    assert {"the same token, the same number: 25", "each from GPT-2's list of 50,257"
            } <= set(three)
    out = output(a, 2, 0.9)
    assert "→ ids = [12982, 25, 1867, 318, 257," in out
    assert "        4382, 30, 198, 48902, 25]" in out
    # counted, limited and billed
    four = picture(a, 3, 0.9)
    assert {"read: 10 tokens", "written: at most 20", "GPT-2's limit: 1,024 tokens in all",
            "at most 30 of 1,024", "next: One token at a time"} <= set(four)
    assert 'most = request["max_tokens"]' in code(a, 3)
    assert a.heading(3, 0.5) == "tokens are counted, limited and billed"
    assert "bills" in a.subtitle_for(3)


# ---------------------------------------------------------------------- 256
@needs_gpt2
def test_one_token_at_a_time():
    a, s = built("hosting-one-token-at-a-time")
    m = s.metadata
    for k in range(4):                                   # the server, the cable, the phone
        assert {"the server: GPT-2", "the internet", "a phone", "What is a", "server?"
                } <= set(picture(a, k, 0.6))
    one = picture(a, 0, 0.9)
    top = m["steps"][0]["top"]
    assert [t["shown"] for t in top] == ["␣A", "␣It", "␣Server", "␣The", "␣I"]
    assert {t["shown"] for t in top} | {t["p_text"] for t in top} <= set(one)
    assert {"11.3%", "the 5 likeliest of 50,257", "picked: the likeliest",
            "the text so far: 10 tokens"} <= set(one)
    assert "A" not in one                                # nothing sent yet
    assert "    nxt = int(scores.argmax())" in code(a, 0)
    assert '"A"' in a.subtitle_for(0) and "11.3%" in a.subtitle_for(0)
    # sent at once, then added to the text
    assert "A" in picture(a, 1, 0.45) and "the text so far: 10 tokens" in picture(a, 1, 0.45)
    two = picture(a, 1, 0.9)
    assert {"A", "sent at once", "the text so far: 11 tokens", "written: 1 of 20"} <= set(two)
    assert "    send(tok.decode([nxt]))" in code(a, 1, 0.3)
    assert "    scores, cache = step(W, [nxt], cache)" in code(a, 1, 0.9)
    # again: each step's top five, the phone's text one token longer
    for u, i, said in ((0.25, 1, "A server"), (0.49, 2, "A server is"), (0.73, 3, "A server is a")):
        here = picture(a, 2, u)
        st = m["steps"][i]
        assert {st["shown"], st["p_text"]} <= set(here)
        assert said in here and "written: %d of 20" % (i + 1) in here
        assert "the text so far: %d tokens" % (11 + i) in here
    assert "68.2%" in picture(a, 2, 0.25) and "51.4%" in picture(a, 2, 0.49)
    # the whole reply, and what is wrong with it
    four = picture(a, 3, 0.95)
    lines = [t for t in four if t in m["reply"]]
    assert " ".join(lines) == m["reply_text"]
    assert {"20 tokens: max_tokens", "stops mid-sentence", "GPT-2 was not trained to chat:",
            "a poor reply", "the text so far: 30 tokens", "written: 20 of 20",
            "next: Read once, then remember"} <= set(four)
    out = [s.strip(" →") for s in output(a, 3, 0.95)]
    assert " ".join(out) == m["reply_text"]
    assert "    sys.stdout.write(line.decode())" in code(a, 3)
    assert "not trained to chat" in a.subtitle_for(3) and "poor" in a.subtitle_for(3)


# ---------------------------------------------------------------------- 257
@needs_gpt2
def test_read_once_remember():
    a, s = built("hosting-read-once-remember")
    m = s.metadata
    assert a.rows_shown(0, 0.9) == 21 and a.rows_shown(0, 0.0) == 0
    one = picture(a, 0, 0.9)
    assert {"tokens run, step by step", "the prompt, then 20 steps",
            "without a cache: 410 tokens run"} <= set(one)
    assert "scores, cache = step(W, ids, [])" in code(a, 0)
    assert "11 tokens" in a.subtitle_for(0) and "30" in a.subtitle_for(0)
    two = picture(a, 1, 0.9)
    assert {"the cache, kept", "a shelf a layer", "a column a token",
            "without a cache: 410 tokens run"} <= set(two)
    assert "new_cache.append((k, v))" in code(a, 1, 0.9)
    assert "12 layers" in a.subtitle_for(1) and "keys and values" in a.subtitle_for(1)
    # with the cache: a step at a time, the shelves one column longer each
    assert a.steps_cached(2, 0.0) == 0 and a.steps_cached(2, 0.9) == 20
    assert a.steps_cached(1, 0.9) == 0 and a.steps_cached(3, 0.1) == 20
    three = picture(a, 2, 0.9)
    assert {"grey: kept, not run again", "with the cache: 20 tokens run",
            "without a cache: 410 tokens run"} <= set(three)
    assert "    scores, cache = step(W, [nxt], cache)" in code(a, 2, 0.5)
    assert "20 tokens run, not 410" in a.subtitle_for(2)
    # the same reply
    four = picture(a, 3, 0.9)
    assert {"with the cache", "reading it all again", "the same 20 tokens",
            "scores within 0.0000068"} <= set(four)
    assert four.count("A server is a place where you can") == 2
    assert "0.0000068" in a.subtitle_for(3)
    assert a.evidence_for(3).startswith("measured on GPT-2: ")
    # less work, counted
    five = picture(a, 4, 0.9)
    assert {"without a cache: 35.7 billion multiply-adds", "with the cache: 2.5 billion",
            "14.4 times fewer", "next: Many people, one model"} <= set(five)
    assert a.heading(4, 0.5) == "14 times fewer multiply-adds"
    assert "Counts, not times" in a.subtitle_for(4)
    assert a.evidence_for(4).startswith("exact arithmetic: ")
    assert m["ratio_text"] in a.subtitle_for(4)

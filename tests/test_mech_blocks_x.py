"""Animations 252-254 (voxanim.mech_blocks_x): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, serving is not training, and the numbers and code lines on screen are the study's and
the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_gpt2

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_x as M
from voxanim.next_catalog import build_study

NAMES = ("hosting-device-and-server", "hosting-a-request-is-text", "hosting-the-model-is-a-file")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get("gpt2-" + name).build(), build_study("study_blocks_x", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


def picture_quads(a, sc):
    """How many shapes are drawn in the picture, left of the code panel."""
    xs = [a.cam.project(q.p0 + q.e1 / 2 + q.e2 / 2)[0] / a.cam.width for q in sc.quads]
    return sum(x < B.PANEL[0] for x in xs)


# ---------------------------------------------------------------------- no data needed
def test_wrapping_keeps_every_word():
    tokens = [" A", " server", " is", " a", " place", " where", " you", " can", " go"]
    states = M.wrap_words(tokens, 90.0, 3)
    assert len(states) == len(tokens) + 1 and states[0] == []
    assert states[1] == ["A"] and states[2] == ["A server"]
    for rows in states:
        assert len(rows) <= 3
        assert all(M.width_px(r) <= 90.0 for r in rows)
    text = "A server is a place where you can communicate with other people."
    assert " ".join(M.wrap_chars(text, 39)) == text
    assert all(len(r) <= 39 for r in M.wrap_chars(text, 39))


def test_the_titles_and_the_chain_are_the_catalogues():
    from voxanim.next_catalog import ITEMS
    titles = {i["name"]: i["title"] for i in ITEMS}
    for name in NAMES:
        assert titles[name] == M.ANIMATIONS[name].TITLE
    assert titles["vector-db-versions"] == M.DeviceAndServer.LAST
    assert M.DeviceAndServer.NEXT == titles["hosting-a-request-is-text"] == M.RequestIsText.TITLE
    assert titles["hosting-device-and-server"] == M.RequestIsText.LAST
    assert M.RequestIsText.NEXT == titles["hosting-the-model-is-a-file"] == M.ModelIsAFile.TITLE
    assert titles["hosting-a-request-is-text"] == M.ModelIsAFile.LAST
    assert titles["hosting-text-to-tokens"] == M.ModelIsAFile.NEXT


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
        # the furniture (panel, five stations), and something of the picture that stays
        assert len(sc.quads) >= 7 and picture_quads(a, sc) >= 1
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


@needs_gpt2
@pytest.mark.parametrize("name", NAMES)
def test_counts_not_speeds(name):
    """No time, price, company or chip: a count of numbers or bytes, never a speed."""
    a, _ = built(name)
    said = " ".join([a.subtitle_for(k) for k in range(len(a.PHASES))]
                    + [a.heading(k, u) for k in range(len(a.PHASES)) for u in (0.3, 0.9)]
                    + [t for k in range(len(a.PHASES)) for t in texts(a, k, 0.9)]).lower()
    for word in ("second", " ms", "fast", "slow", "$", "price", "gpu", "openai", "cloud"):
        assert word not in said, word


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
        shown = [s for s in mono(a, k, 0.6) if not s.startswith("→")]
        assert shown == list(code.lines())
        lines = [s for n in code.names() for s in B.code_lines(n)]
        assert all(s in lines for s in shown)
        assert {B.code_file(n) for n in code.names()} == {"examples/serve.py"}


@needs_gpt2
@pytest.mark.parametrize("name", NAMES)
def test_serving_is_not_training(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        assert a.station_for(k) is None
    assert "you are here" not in texts(a, 1, 0.5)


# ---------------------------------------------------------------------- 252
@needs_gpt2
def test_device_and_server():
    a, s = built("hosting-device-and-server")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"your phone or laptop", "chat app", "What is", "a server?", "on your phone:",
            "the app, your words"} <= set(one)
    assert "the server, far away" not in one and "the model" not in one
    assert '"content": question}' in [x.strip() for x in mono(a, 0)]
    two = texts(a, 1, 0.9)
    assert {"the server, far away", "the model", "on the server:", "the model,",
            "124,439,808 numbers", "498 MB", "not the model"} <= set(two)
    assert mono(a, 1) == ["W = gpt2.weights()", "tok = gpt2.Tokenizer()"]
    assert "real services run far bigger" in a.subtitle_for(1)
    three = texts(a, 2, 0.9)
    assert {"82 bytes", "a request:", "your words as text", "the internet"} <= set(three)
    assert "82 bytes" in a.subtitle_for(2) and m["request_bytes"] == 82
    # tokens back, one at a time: the counter is the tokens that have arrived
    for u in (0.3, 0.5):
        assert "tokens back: %d" % a.arrived(u) in texts(a, 3, u)
    assert 0 < a.arrived(0.3) < a.arrived(0.5) < 20
    four = texts(a, 3, 0.95)
    assert {"tokens back: 20", "92 bytes", "a token:", "a piece of a word", "the reply",
            "A server is a", "place where", "you can…"} <= set(four)
    out = [x.strip() for x in mono(a, 3, 0.95) if x not in a.code_for(3).lines()]
    assert " ".join(x.lstrip("→ ") for x in out) == m["reply"].strip()
    assert "not trained to chat" in a.subtitle_for(3)
    five = texts(a, 4, 0.9)
    assert {"bytes, on a log scale: each mark ×1,000", "request: 82", "reply: 92",
            "model: 497,759,232", "1", "1,000", "1 million", "1 billion"} <= set(five)
    assert "6,070,234 times" in a.subtitle_for(4) and "log scale" in a.subtitle_for(4)
    assert a.heading_at(a.t_done - 1.0) == "next: A request is text"


# ---------------------------------------------------------------------- 253
@needs_gpt2
def test_request_is_text():
    a, s = built("hosting-a-request-is-text")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {'{"role": "user",', '"content": "What is a server?"}', "role: who is speaking",
            "content: and what they said"} <= set(one)
    assert '{"messages": [' not in one
    two = texts(a, 1, 0.9)
    assert {'{"messages": [', '"max_tokens": 20}', "messages: the whole talk so far",
            "max_tokens: how long a reply may be"} <= set(two)
    three = texts(a, 2, 0.9)
    assert {"body: one line of text", "82 bytes", "one per letter, space and mark",
            "the server"} <= set(three)
    assert all(r.strip() in three for r in m["rows"])
    assert "".join(m["rows"]) == m["json"] and len(m["json"]) == 82
    four = texts(a, 3, 0.95)
    assert {"turn 1", "turn 2", "turn 3", "82 bytes", "252 bytes", "422 bytes", "10 tokens",
            "36 tokens", "66 tokens", "an example talk, written for this animation"
            } <= set(four)
    assert four.count("you") == 1 + 2 + 3 and four.count("reply") == 0 + 1 + 2
    assert "turn 3" not in texts(a, 3, 0.3)                  # one turn at a time
    assert mono(a, 3) == ['request = json.loads(self.rfile.read(n))',
                          'text = prompt(request["messages"])']
    assert "written for this animation" in a.subtitle_for(3)
    assert a.heading_at(a.t_done - 1.0) == "next: The model is a file of numbers"


# ---------------------------------------------------------------------- 254
@needs_gpt2
def test_model_is_a_file():
    a, s = built("hosting-the-model-is-a-file")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"the model: a file", "numbers in it", "124,439,808",
            "an orange tile: one number"} <= set(one)
    assert mono(a, 0) == ["W = gpt2.weights()", "tok = gpt2.Tokenizer()"]
    assert "added up: 38,597,376" in texts(a, 1, 0.15)
    two = texts(a, 1, 0.95)
    assert {"token table", "38,597,376", "12 layers", "7,087,872 each",
            "and positions 786,432, last norm 1,536", "added up: 124,439,808"} <= set(two)
    three = texts(a, 2, 0.9)
    assert {"one number: −0.1101", "4 bytes", "124,439,808 × 4 bytes",
            "= 497,759,232 bytes", "about 498 MB"} <= set(three)
    assert m["first_text"] == "-0.1101"
    four = texts(a, 3, 0.9)
    assert {"the server's memory", "loaded: once", "497,759,232 bytes",
            "memory: what the computer"} <= set(four)
    five = texts(a, 4, 0.95)
    assert {"four people", "requests: 4", "loaded: once, 497,759,232 bytes",
            "the server's memory"} <= set(five)
    assert "reply(W, tok, text, most, send)" in mono(a, 4)
    assert a.heading_at(a.t_done - 1.0) == "next: Text becomes tokens"

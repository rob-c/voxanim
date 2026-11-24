"""Animations 245-248 (voxanim.mech_blocks_v): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, the people are named only by number, and the numbers and code lines on screen are the
study's and the script's."""

from __future__ import annotations

import re
from itertools import pairwise

import numpy as np
import pytest
from conftest import needs_faces

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_v as M
from voxanim.next_catalog import build_study

NAMES = ("faces-a-face-is-numbers", "faces-an-embedding", "faces-triplets",
         "faces-verification")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_v", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


def is_tag(text):
    return B.is_evidence(text) or text.startswith(M.TAG)


# ---------------------------------------------------------------------- no data needed
def test_tags_and_colours():
    assert M.faces("x") == "measured on Olivetti faces: x" and is_tag(M.faces("x"))
    rgb = M.face_rgb(np.array([[0.0, 1.0]]))
    assert rgb.shape == (1, 2, 3) and rgb[0, 0].sum() < rgb[0, 1].sum()   # 0 dark, 1 light
    assert np.array_equal(rgb[0, :, 0], [0.0, 1.0]) and np.ptp(rgb, axis=-1).max() == 0  # grey
    sheet = M.mosaic(np.zeros((2, 3, 4, 4)))
    assert sheet.shape == (2 * 5 - 1, 3 * 5 - 1, 3)


# ---------------------------------------------------------------------- every one of them
@needs_faces
@pytest.mark.parametrize("name", NAMES)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@needs_faces
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


@needs_faces
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
        assert len(sc.quads) >= 8
        assert sum(x.opacity > 0.5 and bool(x.s.strip()) for x in sc.texts) >= 8


@needs_faces
@pytest.mark.parametrize("name", NAMES)
def test_subtitles_fit_and_every_phase_says_what_it_shows(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        sub = a.subtitle_for(k)
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert 1 <= len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
        for u in (0.2, 0.6, 0.9):
            assert any(is_tag(t) for t in texts(a, k, u))
        assert sum(bool(t.strip()) for t in texts(a, k, 0.6)) >= 3
        tag = a.evidence_for(k)
        assert layout.text_width(tag, B.SIZE) <= 0.95 - B.PANEL[0]
        assert tag.startswith((M.TAG, "exact arithmetic: "))
        assert "MNIST" not in tag


@needs_faces
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


@needs_faces
@pytest.mark.parametrize("name", NAMES)
def test_the_code_on_screen_is_the_scripts(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        shown = [s for s in mono(a, k, 0.6) if not s.startswith("→")]
        assert shown == list(code.lines())
        lines = [s for n in code.names() for s in B.code_lines(n)]
        assert all(s in lines for s in shown)
        assert {B.code_file(n) for n in code.names()} == {"examples/faces.py"}


@needs_faces
@pytest.mark.parametrize("name", NAMES)
def test_people_are_only_numbers(name):
    """Nobody is named or described: every word about a person is "person" and a number."""
    a, _ = built(name)
    said = [a.subtitle_for(k) for k in range(len(a.PHASES))]
    for k in range(len(a.PHASES)):
        for u in (0.3, 0.9):
            said += texts(a, k, u)
    words = " ".join(said).lower()
    for w in ("man", "woman", "male", "female", "old", "young", "smil", "glasses", "beard",
              "skin", "race", "age ", "happy", "sad", "anyone", "everyone"):
        assert not re.search(r"\b" + w, words), w
    for s in said:                                       # "person" is followed by a number
        assert not re.search(r"\bperson [A-Z]", s), s
        for n in re.findall(r"\bpersons? (\d+)", s):
            assert 0 <= int(n) < 40


@needs_faces
def test_the_titles_and_the_chain_are_the_catalogues():
    from voxanim.next_catalog import ITEMS
    titles = {i["name"]: i["title"] for i in ITEMS}
    for name in NAMES:
        assert titles[name] == M.ANIMATIONS[name].TITLE
    chain = [M.ANIMATIONS[n] for n in NAMES]
    assert titles["faces-precision-and-recall"] == chain[0].LAST
    for a, b in pairwise(chain):
        assert a.NEXT == b.TITLE and b.LAST == a.TITLE
    assert chain[-1].NEXT == titles["faces-identification"] == "Who is it?"


# ---------------------------------------------------------------------- 245
@needs_faces
def test_a_face_is_numbers():
    a, s = built("faces-a-face-is-numbers")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"photo 300 · person 30", "64 × 64 = 4,096 numbers", "9 of the 4,096",
            "0 is dark, 1 is light"} <= set(one)
    assert all(v in one for row in m["patch_text"] for v in row)
    assert "photos = torch.tensor(faces.images)" in mono(a, 0)
    assert a.station_for(0) == "data" and a.station_for(2) is None
    two = texts(a, 1, 0.9)
    assert {"people 0–29: training", "30–39: unseen", "400 photos", "a column: one person",
            "person 30"} <= set(two)
    for k in (1, 2, 3):                                  # the 400 photos stay
        assert "people 0–29: training" in texts(a, k, 0.6)
    three = texts(a, 2, 0.9)
    assert {"person 30: 0.340", "person 31: 0.336", "one person", "two people, yet closer",
            "each photo scaled to length 1"} <= set(three)
    assert "two people" in texts(a, 2, 0.5) and "two people, yet closer" not in texts(a, 2, 0.5)
    four = texts(a, 3, 0.9)
    assert {"one person: 450 pairs", "two people: 4,500", "both kinds", "0.17–0.42",
            "next: A face as 32 numbers"} <= set(four)
    assert "photo 300 · person 30" not in four
    assert "0.340" in a.subtitle_for(2) and "0.336" in a.subtitle_for(2)
    assert "0.06 to 0.42" in a.subtitle_for(3) and "0.17 to 0.52" in a.subtitle_for(3)
    assert "4,950 pairs" in a.evidence_for(3)


# ---------------------------------------------------------------------- 246
@needs_faces
def test_an_embedding():
    a, s = built("faces-an-embedding")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"photo 300 · person 30, 32 × 32", "[1, 32, 32]", "[8, 16, 16]", "[16, 8, 8]",
            "[32, 4, 4]"} <= set(one)
    assert "    *block(16, 32)," in mono(a, 0)
    assert "512" not in one                              # not flattened yet
    two = texts(a, 1, 0.9)
    assert {"512", "32", "Linear(512, 32)", "length 0.09", "[32, 4, 4]"} <= set(two)
    assert "    nn.Linear(512, 32)," in mono(a, 1)
    assert "length 0.09" in texts(a, 2, 0.3) and "÷ 0.09" in texts(a, 2, 0.3)
    assert "length 1.00" in texts(a, 2, 0.9)
    assert "a = F.normalize(encoder(anchor))" in mono(a, 2)
    four = texts(a, 3, 0.8)
    assert {"80", "1,168", "4,640", "16,416", "= 22,304", "length 1.00"} <= set(four)
    assert a.evidence_for(3) == "exact arithmetic: 80 + 1,168 + 4,640 + 16,416"
    assert a.heading(3, 0.5) == "22,304 numbers learned"
    assert [a.station_for(k) for k in range(4)] == ["forward"] * 3 + [None]
    assert m["length_text"] in a.subtitle_for(2) and "22,304" in a.subtitle_for(3)


# ---------------------------------------------------------------------- 247
@needs_faces
def test_triplets():
    a, s = built("faces-triplets")
    m = s.metadata
    for k in range(4):                                   # the triplet stays all through
        assert {"anchor", "positive", "negative", "person 14", "person 3"} <= set(texts(a, k))
    one = texts(a, 0, 0.9)
    assert {"same person: anchor, positive", "somebody else: negative",
            "photos 144, 149 and 30"} <= set(one)
    assert "n = F.normalize(encoder(other))" in mono(a, 0)
    two = texts(a, 1, 0.9)
    assert {"p 0.04", "n 0.09", "+ 0.5", "distance from the anchor",
            "loss = max(0.04 − 0.09 + 0.5, 0)", "= 0.45"} <= set(two)
    assert "    a, p, n, margin=0.5)" in mono(a, 1)
    three = texts(a, 2, 0.9)
    assert {"loss", "not yet 0.5 apart", "0.45", "100%", "1,500", "0.0004", "1%",
            "step 0, then means of 25 steps"} <= set(three)
    assert a.curves[0][0] == m["first_loss"] and a.curves[1][0] == 1.0
    assert np.allclose(a.curves[0][1:], m["loss_means"])
    four = texts(a, 3, 0.9)
    assert {"p 0.24", "n 1.27", "loss = max(0.24 − 1.27 + 0.5, 0)", "= 0",
            "next: Same person or not?"} <= set(four)
    assert [a.station_for(k) for k in range(4)] == ["forward", "loss", "loss", "forward"]
    assert "person 14" in a.subtitle_for(0) and "person 3" in a.subtitle_for(0)
    assert "0.45" in a.subtitle_for(1) and "1.27" in a.subtitle_for(3)


# ---------------------------------------------------------------------- 248
@needs_faces
def test_verification():
    a, s = built("faces-verification")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"100 × 99 ÷ 2 = 4,950", "450 of one person", "4,500 of two people",
            "a row: one person", "4,950 pairs: 450 one person, 4,500 two people"} <= set(one)
    assert "i, j = torch.triu_indices(100, 100, 1)" in mono(a, 0)
    for k in (1, 2, 3):                                   # the two piles stay
        shown = texts(a, k, 0.6)
        assert {"one person", "two people", "0", "1", "2",
                "4,950 pairs: 450 one person, 4,500 two people"} <= set(shown)
    assert "0.72" not in texts(a, 1, 0.9)                 # no threshold yet
    three = texts(a, 2, 0.9)
    assert {"0.72", "86 false accepts · 1.9% of 4,500", "111 false rejects · 24.7% of 450",
            "4,753 of 4,950 right"} <= set(three)
    assert "→ right: tensor(4753)" in mono(a, 2, 0.9)
    assert "people 0–29" in a.heading(2, 0.5) and "people 0-29" in a.evidence_for(2)
    four = texts(a, 3, 0.8)
    assert {"raw pixels", "22.0%", "untrained", "18.0%", "trained", "12.4%",
            "threshold from people 0–29: 197 mistakes, pixels 390", "0.72"} <= set(four)
    assert "12.4%" in a.subtitle_for(3) and "22.0%" in a.subtitle_for(3)
    assert m["threshold_text"] in a.subtitle_for(2)
    assert [a.station_for(k) for k in range(4)] == ["forward", None, None, None]

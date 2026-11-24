"""Animations 249-251 (voxanim.mech_blocks_w): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, people are named by number only, and the numbers and code lines on screen are the
study's and the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_faces

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_w as M
from voxanim import study_blocks_w as W
from voxanim.next_catalog import build_study

NAMES = ("faces-identification", "faces-the-whole-pipeline", "faces-what-40-people-cannot-show")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_w", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


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
            assert any(W.is_tag(t) for t in texts(a, k, u))
        assert sum(bool(t.strip()) for t in texts(a, k, 0.6)) >= 3
        assert layout.text_width(a.evidence_for(k), B.SIZE) <= 0.95 - B.PANEL[0]
        assert W.is_tag(a.evidence_for(k))


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
def test_matching_faces_is_not_training(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        assert a.station_for(k) in (None, "forward")


@needs_faces
@pytest.mark.parametrize("name", NAMES)
def test_people_are_named_by_number_only(name):
    """Nothing on screen or in a subtitle labels anyone's looks or who they are."""
    a, _ = built(name)
    said = " ".join([a.subtitle_for(k) for k in range(len(a.PHASES))]
                    + [t for k in range(len(a.PHASES)) for u in (0.3, 0.9) for t in texts(a, k, u)])
    words = set(said.lower().replace(",", " ").replace(".", " ").replace(":", " ").split())
    for w in ("man", "men", "woman", "women", "male", "female", "boy", "girl", "old", "young",
              "older", "younger", "white", "black", "asian", "smiling", "smile", "happy", "sad",
              "glasses", "beard", "he", "she", "his", "her", "anyone", "everyone"):
        assert w not in words, w


@needs_faces
def test_the_titles_and_the_chain_are_the_catalogues():
    from voxanim.next_catalog import ITEMS
    titles = {i["name"]: i["title"] for i in ITEMS}
    for name in NAMES:
        assert titles[name] == M.ANIMATIONS[name].TITLE
    assert titles["faces-verification"] == M.Identification.LAST
    assert M.Identification.NEXT == titles["faces-the-whole-pipeline"] == M.Pipeline.TITLE
    assert titles["faces-identification"] == M.Pipeline.LAST
    assert M.Pipeline.NEXT == titles["faces-what-40-people-cannot-show"] == M.Limits.TITLE
    assert titles["faces-the-whole-pipeline"] == M.Limits.LAST
    assert M.Limits.NEXT == "Regression with a network"
    assert "Regression with a network" in titles.values()


# ---------------------------------------------------------------------- 249
@needs_faces
def test_identification():
    a, s = built("faces-identification")
    m = s.metadata
    for k in range(5):                                   # the gallery stays all through
        assert {"enrolled people", "30", "31", "32", "33", "34"} <= set(texts(a, k, 0.6))
    assert "gallery = v[:50:10]" in mono(a, 0)
    assert "32 numbers" in texts(a, 0, 0.9)
    assert a.station_for(0) == "forward" and a.station_for(1) is None
    two = texts(a, 1, 0.9)
    assert {"new photo", "photo 305", "nearest:", "person 30", "threshold 0.72", "0.18",
            "1.36", "1.40", "1.08", "1.04", "distance"} <= set(two)
    assert "0.18" in a.subtitle_for(1) and "0.72" in a.subtitle_for(1)
    three = texts(a, 2, 0.9)
    assert {"photo 301", "nearest: 31", "unsure", "it is person 30", "0.81", "1.00", "1.56",
            "threshold 0.72"} <= set(three)
    assert "photo 305" not in three
    assert "yes = d < THRESHOLD" in mono(a, 2)
    four = texts(a, 3, 0.9)
    assert {"nearest is right", "36 of 45", "raw pixels", "right", "wrong"} <= set(four)
    assert four.count("36 of 45") == 2                   # the encoder's and raw pixels'
    assert "raw pixels also get 36" in a.subtitle_for(3)
    assert "verification" in a.subtitle_for(3)
    five = texts(a, 4, 0.9)
    assert {"6 of 50 let in", "people 35-39", "never enrolled", "threshold 0.72",
            "→ let in: tensor(6)", "next: Find, cut out, compare"} <= set(five)
    assert m["let_in"] == 6 and "6 are under the threshold" in a.subtitle_for(4)
    assert a.evidence_for(4) == "measured on Olivetti faces: 50 photos of people 35-39"


# ---------------------------------------------------------------------- 250
@needs_faces
def test_pipeline():
    a, s = built("faces-the-whole-pipeline")
    m = s.metadata
    for k in range(5):                                   # the scene stays all through
        assert "faces pasted on paper" in texts(a, k, 0.6)
    assert "the first scene with 3 faces" in texts(a, 0, 0.6)
    assert "pasted on paper" in a.subtitle_for(0) and "first scene with 3" in a.subtitle_for(0)
    two = texts(a, 1, 0.9)
    assert {"0.96", "0.93", "0.92", "boxes over 0.5", "3 after nms"} <= set(two)
    assert "kept = nms(corners, score, 0.3)" in mono(a, 1)
    assert a.station_for(1) == "forward"
    assert "7" in texts(a, 1, 0.3)
    assert "32 × 32" in texts(a, 2, 0.9)
    assert "crop = canvas[:, y1:y2, x1:x2]" in mono(a, 2)
    four = texts(a, 3, 0.95)
    assert {"person · distance", "30 · 0.34", "34 · 0.84", "31 · 0.71", "unsure",
            "threshold 0.72", "30", "34", "31"} <= set(four)
    assert a.station_for(3) == "forward"
    assert "q = F.normalize(encoder(crop))" in mono(a, 3)
    five = texts(a, 4, 0.95)
    assert {"200 scenes · 391 faces", "found 341", "named right:", "from the detector's box",
            "67%", "227 of 341", "from the true box", "82%", "320 of 391",
            "the truth by 0.76", "grey: the true boxes"} <= set(five)
    assert "67%" in a.subtitle_for(4) and "82%" in a.subtitle_for(4)
    assert m["det_text"] == "67%" and m["true_text"] == "82%"
    assert "200 scenes" in a.evidence_for(4)


# ---------------------------------------------------------------------- 251
@needs_faces
def test_limits():
    a, s = built("faces-what-40-people-cannot-show")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"own pairs rejected, of 45 each", "all: 25%", "73%"} <= set(one)
    assert all(str(p) in one for p in range(30, 40))
    for v in m["rejects"]:                               # each person's count, over its bar
        assert str(v) in one
    assert "73%" in a.subtitle_for(0) and "0%" in a.subtitle_for(0)
    two = texts(a, 1, 0.9)
    assert {"pairs with others accepted, of 900 each", "all: 1.9%",
            "own pairs rejected, of 45 each", "29", "34", "32", "28"} <= set(two)
    assert "0.2%" in a.subtitle_for(1) and "3.8%" in a.subtitle_for(1)
    three = texts(a, 2, 0.9)
    assert {"40 people × 10 photos", "trained on: 0-29", "tested: 30-39", "one lab · 1992–94",
            "own pairs rejected, of 45 each"} <= set(three)          # the ten, among the 40
    four = texts(a, 3, 0.9)
    assert {"how errors differ across", "ages", "skin tones", "lighting", "cameras", "poses",
            "that it never saw: not measured", "results here do not transfer",
            "next: Regression with a network"} <= set(four)
    assert "still err unevenly" in a.subtitle_for(3)
    assert a.evidence_for(2).startswith(W.ABOUT) and a.evidence_for(0).startswith(W.TAG)
    assert [a.station_for(k) for k in range(4)] == [None] * 4

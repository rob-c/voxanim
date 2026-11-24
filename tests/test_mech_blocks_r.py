"""Animations 234-235 (voxanim.mech_blocks_r): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_r as M
from voxanim.next_catalog import build_study

NAMES = ("vector-db-training", "vector-db-store-and-search")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_r", name))
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
        assert a.evidence_for(k).startswith("measured on MNIST: ")


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
        shown = [s for s in mono(a, k, 0.6) if not s.startswith(("→", "  "))
                 or s in code.lines()]
        assert shown == list(code.lines())
        lines = [s for n in code.names() for s in B.code_lines(n)]
        assert all(s in lines for s in shown)
        assert {B.code_file(n) for n in code.names()} == {"examples/vector_db.py"}


@needs_mnist
def test_the_titles_and_the_chain_are_the_catalogues():
    from voxanim.next_catalog import ITEMS
    titles = {i["name"]: i["title"] for i in ITEMS}
    for name in NAMES:
        assert titles[name] == M.ANIMATIONS[name].TITLE
    assert M.Training.LAST == "How wide should an embedding be?"
    assert M.Training.NEXT == titles["vector-db-store-and-search"] == M.StoreSearch.TITLE
    assert titles["vector-db-training"] == M.StoreSearch.LAST
    assert M.StoreSearch.NEXT == titles["vector-db-ivf"] == "An index: look in fewer places"


# ---------------------------------------------------------------------- 234
@needs_mnist
def test_training():
    a, s = built("vector-db-training")
    m = s.metadata
    for k in range(5):                                   # the pair stays all through
        assert any(t.startswith("x[0]: a ") for t in texts(a, k, 0.9))
        assert any(t.startswith("p[0]: a ") for t in texts(a, k, 0.9))
    one = texts(a, 0, 0.9)
    assert {"x[0]: a 5", "p[0]: a 5", "a pair: the same digit", "two drawings of a 5",
            "labels only pick the pair"} <= set(one)
    assert "p = partner(y)" in mono(a, 0) and a.station_for(0) == "data"
    assert "labels only pick the pairs" in a.subtitle_for(0)
    two = texts(a, 1, 0.9)
    assert {"encoder", "a[0]: 32 numbers", "b[0]: 32 numbers", "a[0] · b[0] = 0.61"} <= set(two)
    assert "    nn.Linear(128, 32)," in mono(a, 1) and a.station_for(1) == "forward"
    three = texts(a, 2, 0.9)
    assert {"a[0] · b[0] = 0.61", "the diagonal: the right pairs", "4.19", "loss",
            "64 digits × 64 partners"} <= set(three)
    assert "target = torch.arange(len(a))" in mono(a, 2)
    assert "4.19" in a.subtitle_for(2) and "ln 64 = 4.16" in a.subtitle_for(2)
    # training: step 0, then 100, then 937, the pair, the numbers and the loss swapping
    for u, step, pos, neg, loss, digit in ((0.2, 0, "0.59", "0.49", "4.19", 5),
                                           (0.5, 100, "0.85", "0.43", "2.58", 9),
                                           (0.9, 937, "0.93", "0.36", "2.13", 8)):
        shown = texts(a, 3, u)
        assert {"step %d" % step, "right pairs: " + pos, "the other 63: " + neg, loss,
                "x[0]: a %d" % digit, "p[0]: a %d" % digit} <= set(shown)
        assert m["snap"][step]["label"] == digit
    assert "optimizer.step()" in mono(a, 3) and a.station_for(3) == "step"
    five = texts(a, 4, 0.9)
    assert {"step 937, sorted by digit", "right pairs 0.93 · same digit 0.92",
            "other digits: 0.29", "next: Store, then search"} <= set(five)
    assert [t for t in five if t.isdigit() and len(t) == 1] == [str(d) for d in range(10)]
    assert "counted it wrong" in a.subtitle_for(4)


# ---------------------------------------------------------------------- 235
@needs_mnist
def test_store_and_search():
    a, s = built("vector-db-store-and-search")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"X: 60,000 digits", "encoder", "id", "label", "vector", "0", "5", "59999", "8",
            "store: [60000, 32]", "ids and labels beside it"} <= set(one)
    assert "labels = Y" in mono(a, 0) and a.station_for(0) == "forward"
    two = texts(a, 1, 0.9)
    assert {"XT[0]: a 7", "q: 32 numbers", "store: [60000, 32]"} <= set(two)
    assert "    q = encoder(XT[:1])" in mono(a, 1)
    three = texts(a, 2, 0.9)
    assert {"score", "60,000 dot products", "7s: 0.96 · others: 0.26"} <= set(three)
    assert [t for t in three if t in m["row_score_texts"]] == m["row_score_texts"]
    assert a.station_for(2) is None
    four = texts(a, 3, 0.9)
    assert {"X[id]", "5 of 5: a 7"} <= set(four)
    assert [t for t in four if t.isdigit() and len(t) > 3] == [str(i) for i in m["top"]]
    assert four.count("0.999") == 5
    out = [t for t in mono(a, 3, 0.9) if t.startswith(("→", "  "))]
    assert out[-2:] == ["→ tensor([38743, 53843, 24372, 50115,",
                        "          1833]) tensor([7, 7, 7, 7, 7])"]
    five = texts(a, 4, 0.9)
    assert {"10,000 test digits as queries", "32, trained", "96.4%", "96.0%", "784 pixels",
            "97.0%", "95.4%", "32, untrained", "87.0%", "81.9%", "32 numbers instead of 784",
            "next: An index: look in fewer places"} <= set(five)
    assert "XT[0]: a 7" in five                          # the query stays
    for k in range(5):
        assert not any(w in a.subtitle_for(k) for w in ("better than pixels", "beats"))

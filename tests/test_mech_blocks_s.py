"""Animations 236-237 (voxanim.mech_blocks_s): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_s as M
from voxanim.next_catalog import build_study

NAMES = ("vector-db-ivf", "vector-db-versions")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_s", name))
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
        shown = [s for s in mono(a, k, 0.6) if not s.startswith("→")]
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
    assert titles["vector-db-store-and-search"] == M.Ivf.LAST
    assert M.Ivf.NEXT == titles["vector-db-versions"] == M.Versions.TITLE
    assert titles["vector-db-ivf"] == M.Versions.LAST
    assert M.Versions.NEXT == "Your device and the server"


@needs_mnist
@pytest.mark.parametrize("name", NAMES)
def test_searching_is_not_training(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        assert a.station_for(k) in (None, "forward")


# ---------------------------------------------------------------------- 236
@needs_mnist
def test_ivf():
    a, s = built("vector-db-ivf")
    m = s.metadata
    for k in range(4):                                   # the 64 lists stay all through
        assert {"64 lists · 60,000 vectors", "4,761",
                "each bar: one list, 89 to 4,761 vectors"} <= set(texts(a, k, 0.6))
    assert "centres = kmeans(store, 64)" in mono(a, 0)
    two = texts(a, 1, 0.9)
    assert {"q · each centre: how near", "test digit 1, a 2: q",
            "the 4 nearest centres: lit"} <= set(two)
    assert "near = (centres @ q[0]).topk(4).indices" in mono(a, 1)
    # the lists visited, one step at a time: the counts are the study's
    for u, i in ((0.3, 0), (0.6, 1), (0.95, 2)):
        st = m["steps"][i]
        shown = texts(a, 2, u)
        assert "%d list%s visited" % (st["nprobe"], "" if i == 0 else "s") in shown
        assert "%s of 60,000 scanned · %s" % (st["scanned_text"], st["share_text"]) in shown
        assert "found %d of its 10 nearest" % st["found"] in shown
    assert "1,923 of 60,000 scanned · 3.2%" in texts(a, 2, 0.3)
    assert "found 10 of its 10 nearest" in texts(a, 2, 0.95)
    assert "scores = store[cand] @ q[0]" in mono(a, 2)
    four = texts(a, 3, 0.9)
    assert {"4 lists: 99.5% found", "8.1% scanned", "1 list: 80.9%", "64 lists: 100.0%",
            "found", "share of the store scanned", "next: A new encoder needs a new store"
            } <= set(four)
    assert "comparisons, not time" in a.subtitle_for(3)
    assert "99.5%" in a.subtitle_for(3) and "8.1%" in a.subtitle_for(3)
    assert "10,000 queries" in a.evidence_for(3)
    assert not any(w in a.subtitle_for(k) for k in range(4) for w in ("faster", "seconds"))


# ---------------------------------------------------------------------- 237
@needs_mnist
def test_versions():
    a, s = built("vector-db-versions")
    m = s.metadata
    for k in range(4):                                   # the digit stays all through
        assert "test digit 0, a 7" in texts(a, k, 0.6)
    one = texts(a, 0, 0.9)
    assert {"encoder(x): q", "encoder2(x): q2", "q · q2 = 0.12",
            "all 60,000 digits: 0.08 on average"} <= set(one)
    assert "    q2 = encoder2(XT[:1])" in mono(a, 0)
    assert a.station_for(0) == "forward" and a.station_for(1) is None
    two = texts(a, 1, 0.9)
    assert "store: encoder · query: encoder2" in two and "not a 7" in two
    assert [t for t in two if t.isdigit()] == [str(v) for v in m["mixed"]["labels"]]
    assert "scores = store @ q2[0]" in mono(a, 1)
    three = texts(a, 2, 0.9)
    assert {"right digit, all 10,000 queries", "one encoder", "96.4%", "mixed encoders",
            "7.6%", "chance", "10.0%", "mixed: 759 of 10,000 right", "below chance"
            } <= set(three)
    assert "96.6%" not in three                          # not re-embedded yet
    assert a.heading(2, 0.6) == "a vector means something only to its encoder"
    assert "7.6%" in a.subtitle_for(2) and "chance" in a.subtitle_for(2)
    four = texts(a, 3, 0.9)
    assert {"store: encoder2 · query: encoder2", "re-embedded", "96.6%", "7.6%",
            "the store [60000, 32], embedded again",
            "next: Your device and the server"} <= set(four)
    assert [t for t in four if t.isdigit()] == [str(v) for v in m["again"]["labels"]]
    assert "    v = encoder2(X)" in mono(a, 3)
    assert a.station_for(3) == "forward"
    assert "all 10,000" in a.evidence_for(2)

"""Animation 271 (voxanim.mech_blocks_ze): the layout is clean, the loop closes, nothing blanks
between phases, every phase says what kind of evidence it shows, the headings are short,
there is no code panel and no loop map (the matrix takes that space), the photo is named
by number only, and the numbers on screen are the study's. It shows photo 0 of the Olivetti
faces."""

from __future__ import annotations

import numpy as np
from conftest import needs_faces

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_ze as M
from voxanim.next_catalog import build_study

NAME = "resample-big-kernel"
_built = {}


def built():
    """``(animation, study)``, built once."""
    if NAME not in _built:
        _built[NAME] = (presets.get(NAME).build(), build_study("study_blocks_ze", NAME))
    return _built[NAME]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.5]


@needs_faces
def test_layout_is_clean():
    assert qa.check(NAME, samples=24, verbose=False) == 0


@needs_faces
def test_drawn_by_this_module_and_loop_closes():
    a, _ = built()
    assert type(a) is M.BigKernel and isinstance(a, B.Blocks)
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
def test_nothing_blanks_between_phases():
    a, _ = built()
    for _lo, hi in a.spans[:-1]:
        sc = a.scene(hi)                                   # the content's dip
        shown = [x.s for x in sc.texts if x.opacity > 0.5]
        assert "photo 0, 6 × 6" in shown or "22 × 22: zeros round photo 0" in shown
        assert len(sc.quads) >= 3                         # at least the picture on its card
        assert sum(x.opacity > 0.5 and bool(x.s.strip()) for x in sc.texts) >= 3


@needs_faces
def test_subtitles_fit_and_every_phase_says_what_it_shows():
    a, _ = built()
    for k in range(len(a.PHASES)):
        sub = a.subtitle_for(k)
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert 1 <= len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
        for u in (0.2, 0.6, 0.9):
            assert any(B.is_evidence(t) for t in texts(a, k, u))
        assert sum(bool(t.strip()) for t in texts(a, k, 0.6)) >= 3
        assert layout.text_width(a.evidence_for(k), B.SIZE) <= 0.95 - B.PANEL[0]
        assert a.evidence_for(k).startswith("exact arithmetic: ")


@needs_faces
def test_headings_are_short_and_start_and_end_the_same_way():
    a, _ = built()
    for t in np.arange(0.0, a.duration, 0.5):
        h = a.heading_at(float(t))
        assert B.words(h) <= 12, h
        assert layout.text_width(h, B.HEAD) <= B.PANEL[0] - B.HEAD_X - 0.01, h
    first = a.heading_at(1.0)
    assert first == "last time: Bigger: spread, then blur"
    assert a.heading_at(a.t_done - 1.0) == "next: A CNN training step"
    assert first in texts(a, 0, 0.1)


@needs_faces
def test_no_code_no_loop_map_nothing_trained():
    a, _ = built()
    for k in range(len(a.PHASES)):
        assert a.code_for(k) is None and a.station_for(k) is None
        lo, hi = a.spans[k]
        sc = a.scene(lo + (hi - lo) * 0.6)
        assert not any(x.kind == "mono" for x in sc.texts)
        shown = texts(a, k, 0.6)
        assert "you are here" not in shown
        assert not {"data", "forward", "loss", "backward", "step"} & set(shown)
        assert not any(s.startswith("examples/") for s in shown)


@needs_faces
def test_the_photo_is_named_by_number_only():
    a, _ = built()
    said = " ".join([a.subtitle_for(k) for k in range(len(a.PHASES))]
                    + [t for k in range(len(a.PHASES)) for u in (0.3, 0.9) for t in texts(a, k, u)])
    words = set(said.lower().replace(",", " ").replace(".", " ").replace(":", " ").split())
    for w in ("man", "men", "woman", "women", "male", "female", "boy", "girl", "old", "young",
              "white", "black", "smiling", "glasses", "beard", "he", "she", "his", "her"):
        assert w not in words, w
    assert "photo 0" in said.lower()


@needs_faces
def test_the_title_and_the_chain_are_the_catalogues():
    from voxanim.next_catalog import ITEMS
    titles = {i["name"]: i["title"] for i in ITEMS}
    item = next(i for i in ITEMS if i["name"] == NAME)
    assert item["number"] == 271 and item["section"].startswith("l. Resizing pictures")
    assert titles[NAME] == M.BigKernel.TITLE == "Bigger: a kernel larger than the picture"
    assert titles["resample-up"] == M.BigKernel.LAST
    assert M.BigKernel.NEXT == "A CNN training step"


@needs_faces
def test_the_numbers_on_screen_are_the_studys():
    a, s = built()
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"photo 0, 6 × 6", "kernel, 9 × 9", "its middle row", "it does not fit inside",
            "all 81 add up to", "1"} <= set(one)
    assert set(m["middle_row_text"]) <= set(one) and one.count("0.001") == 2
    assert "bigger than the picture" in a.subtitle_for(0)
    two = texts(a, 1, 0.9)
    assert {"22 × 22: zeros round photo 0", "14 × 14", "places a side",
            "6 + 2 × 8 − 9 + 1 = 14", "output pixels", "196", "shaded 0 to 0.62"} <= set(two)
    assert "14 places" in a.subtitle_for(1) and "own shading" in a.subtitle_for(1)
    three = texts(a, 2, 0.9)
    assert {"36 pixels × kernel numbers", "45 kernel numbers × 0", "added up", "0.61"} <= \
        set(three)
    assert "0.61" in a.subtitle_for(2) and a.evidence_for(2).endswith("(7, 7) of photo 0's 6 × 6")
    four = texts(a, 3, 0.9)
    assert {"M: 196 × 36", "x: 36", "196", "@", "=", "pixel 25: 0.76",
            "column 25, as 14 × 14", "81 outputs from one pixel"} <= set(four)
    assert "Column 25" in a.subtitle_for(3) and "81" in a.subtitle_for(3)
    assert "× pixel 0: 0.57" in texts(a, 4, 0.12) and "1 of 36" in texts(a, 4, 0.12)
    five = texts(a, 4, 0.95)
    assert {"stamps added", "36 of 36", "each × its pixel", "each column, as 14 × 14",
            "out: 14 × 14", "shaded 0 to 0.62", "smooth values the 36 never showed"} <= set(five)
    assert a.evidence_for(4) == "exact arithmetic: M @ x against conv2d: 0 apart"
    six = texts(a, 5, 0.9)
    assert {"V: 36 × 196, no padding", "each row: an average of 81",
            "row 25 of V = column 25 of M", "M = V turned on its side", "0 apart"} <= set(six)
    assert "average of 81" in a.subtitle_for(5) and "turned on its side" in a.subtitle_for(5)
    seven = texts(a, 6, 0.6)
    assert seven.count("22.99") == 2 and seven.count("total") == 2
    assert {"each column of M adds up to 1", "nothing new, only spread out"} <= set(seven)
    assert "22.99" in a.subtitle_for(6)


@needs_faces
def test_the_output_drawn_is_the_stamps_added_up():
    a, s = built()
    m = s.metadata
    assert len(a.partial) == 37 and np.allclose(a.partial[0], 0)
    np.testing.assert_allclose(a.partial[36], m["out"], atol=1e-12)
    assert [a.added(u) for u in (0.0, 0.1, 0.2, 0.35, 0.95)] == [0, 1, 2, 3, 36]
    assert all(a.added(u) <= a.added(u + 0.01) for u in np.arange(0.0, 1.0, 0.01))
    assert a.scan(0.0) == 0 and a.scan(1.0) == 196

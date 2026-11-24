"""Animations 238-240 (voxanim.mech_blocks_t): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the script's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_faces

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_t as M
from voxanim.next_catalog import build_study

NAMES = ("faces-a-box", "faces-the-grid", "faces-the-detector")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_t", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


def out(a, phase, u=0.9):
    return [s for s in mono(a, phase, u) if s.startswith(("→", "  ")) and s not in
            a.code_for(phase).lines()]


# ---------------------------------------------------------------------- no data needed
def test_the_tags_are_the_series_and_say_the_data():
    assert M.measured("x", on=M.FACES) == "measured on Olivetti faces: x"
    assert B.is_evidence(M.measured("x", on=M.FACES))


def test_gray_keeps_the_photo():
    img = np.array([[0.0, 0.5], [1.0, 1.2]])
    g = M.gray(img)
    assert g.shape == (2, 2, 3) and np.allclose(g[..., 0], g[..., 2])
    assert np.allclose(g[..., 1], [[0, 0.5], [1, 1]])


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
        # the furniture (panel, five stations) and the scene the picture keeps
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
            assert any(B.is_evidence(t) for t in texts(a, k, u))
        assert sum(bool(t.strip()) for t in texts(a, k, 0.6)) >= 3
        assert layout.text_width(a.evidence_for(k), B.SIZE) <= 0.95 - B.PANEL[0]
        assert a.evidence_for(k).startswith(("exact arithmetic: ",
                                             "measured on Olivetti faces: "))


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
        shown = [s for s in mono(a, k, 0.6) if not s.startswith(("→", "  "))
                 or s in code.lines()]
        assert shown == list(code.lines())
        lines = [s for n in code.names() for s in B.code_lines(n)]
        assert all(s in lines for s in shown)
        assert {B.code_file(n) for n in code.names()} == {"examples/faces.py"}


@needs_faces
@pytest.mark.parametrize("name", NAMES)
def test_the_scene_stays_and_is_the_held_out_one(name):
    from voxanim import study_faces as F
    a, _ = built(name)
    assert np.array_equal(a.canvas, F.test_scenes()[0]["canvas"].astype(float))

    def photo(t):
        """The opacity of the 96 x 96 photo on screen at ``t`` (0 if it is not there)."""
        keys = [q.shader.key for q in a.scene(t).quads if hasattr(q.shader, "key")]
        return max([k[6] for k in keys if k[0] == "image" and k[2] == (96, 96, 3)] or [0.0])
    for lo, hi in a.spans:                  # in every phase, and between them, at full strength
        for t in (max(lo, 0.6), (lo + hi) / 2, hi):
            assert photo(t) > 0.95, t


@needs_faces
def test_the_titles_and_the_chain_are_the_catalogues():
    from voxanim.next_catalog import ITEMS
    titles = {i["name"]: i["title"] for i in ITEMS}
    for name in NAMES:
        assert titles[name] == M.ANIMATIONS[name].TITLE
    assert M.ABox.LAST == "Style transfer with Gram matrices"
    assert M.ABox.NEXT == titles["faces-the-grid"] == M.TheGrid.TITLE
    assert M.TheGrid.LAST == M.ABox.TITLE and M.TheGrid.NEXT == M.TheDetector.TITLE
    assert M.TheDetector.LAST == M.TheGrid.TITLE
    assert M.TheDetector.NEXT == titles["faces-two-losses"] == "Two losses: is it there, and where"


# ---------------------------------------------------------------------- 238
@needs_faces
def test_a_box():
    a, s = built("faces-a-box")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"96 × 96 pixels", "2 faces, pasted", "2 not-faces:", "photos, shuffled",
            "we know every box"} <= set(one)
    assert "not-faces" in a.subtitle_for(0) and "shuffled in 8 × 8 blocks" in a.subtitle_for(0)
    assert a.evidence_for(0) == "exact arithmetic: we pasted the faces, so the boxes are exact"
    two = texts(a, 1, 0.9)
    assert {"cx = 27.5", "cy = 61.5", "w = 19", "h = 19"} <= set(two)
    assert out(a, 1) == ["→ (27.5, 61.5, 19, 19)"]
    assert "box = (x + s / 2, y + s / 2, s, s)" in mono(a, 1)
    assert "x 18, y 52, 19 pixels wide" in a.subtitle_for(1)
    three = texts(a, 2, 0.9)
    assert {"x1 = 27.5 − 9.5 = 18", "y1 = 61.5 − 9.5 = 52", "x2 = 27.5 + 9.5 = 37",
            "y2 = 61.5 + 9.5 = 71", "(18, 52, 37, 71)"} <= set(three)
    assert out(a, 2) == ["→ tensor([[55., 32., 87., 64.],", "          [18., 52., 37., 71.]])"]
    assert "real = box_convert(real, *FMT)" in mono(a, 2) and a.station_for(2) == "data"
    four = texts(a, 3, 0.9)
    assert {"true box: (18, 52, 37, 71)", "the detector's box: (18.9, 49.4, 38.4, 69.1)",
            "it is 0.97 sure of a face"} <= set(four)
    assert a.station_for(3) == "forward"
    assert a.evidence_for(3) == "measured on Olivetti faces: the trained detector, held-out"
    five = texts(a, 4, 0.9)
    assert {"overlap: 308", "together: 361 + 383 − 308 = 436", "IoU = 308 ÷ 436 = 0.71"} <= \
        set(five)
    assert out(a, 4) == ["→ tensor([[0.7452, 0.0000],", "          [0.0000, 0.7077]])"]
    assert a.station_for(4) is None
    assert "0.78 over 200 held-out scenes" in a.subtitle_for(4)
    assert m["iou_text"] == "0.71" and m["mean_iou_text"] == "0.78"


# ---------------------------------------------------------------------- 239
@needs_faces
def test_the_grid():
    a, s = built("faces-the-grid")
    m = s.metadata
    assert {"12 × 12 cells", "each 8 × 8 pixels", "96 ÷ 8 = 12"} <= set(texts(a, 0, 0.9))
    two = texts(a, 1, 0.9)
    assert {"centre (27.5, 61.5)", "61.5 ÷ 8 = 7.69", "row 7", "27.5 ÷ 8 = 3.44",
            "column 3"} <= set(two)
    assert "    r, c = int(y // CELL), int(x // CELL)" in mono(a, 1)
    assert {"across: 0.44", "down: 0.69", "0 to 1 inside the cell"} <= set(texts(a, 2, 0.9))
    four = texts(a, 3, 0.9)
    assert {"w = h = 19 pixels", "log(19 ÷ 16) = 0.17", "16 → 0.00", "32 → 0.69"} <= set(four)
    assert out(a, 3) == ["→ (7, 3, tensor([0.4375, 0.6875,", "                 0.1719, 0.1719]))"]
    five = texts(a, 4, 0.9)
    assert {"has", "2 cells say 1", "142 cells say 0", "not-faces too", "its cell:"} <= set(five)
    row = [t for t in five if t in ("1", "0.44", "0.69", "0.17")]
    assert row == ["1", "0.44", "0.69", "0.17", "0.17"] == m["five_text"]
    assert [t for t in five if t in ("dx", "dy", "lw", "lh")] == ["dx", "dy", "lw", "lh"]
    assert "        has[r, c] = 1" in mono(a, 4)
    assert all(a.station_for(k) == "data" for k in range(5))


# ---------------------------------------------------------------------- 240
@needs_faces
def test_the_detector():
    a, s = built("faces-the-detector")
    m = s.metadata
    shapes = ["[1, 1, 96, 96]", "[1, 8, 48, 48]", "[1, 16, 24, 24]", "[1, 32, 12, 12]",
              "[1, 5, 12, 12]"]
    one = texts(a, 0, 0.9)
    assert set(shapes[:2]) <= set(one) and not set(shapes[2:]) & set(one)
    assert {"conv 3 × 3", "ReLU", "max 2 × 2", "8 stamps: 8 maps, half as wide"} <= set(one)
    assert out(a, 0) == ["→ [1, 1, 96, 96] → [1, 8, 48, 48]"]
    two = texts(a, 1, 0.9)
    assert set(shapes[:4]) <= set(two) and shapes[4] not in two
    assert out(a, 1) == ["→ [1, 16, 24, 24] → [1, 32, 12, 12]"]
    three = texts(a, 2, 0.9)
    assert set(shapes) <= set(three)
    assert {"this cell: 32 numbers → 5", "a face here? 0.97"} <= set(three)
    assert out(a, 2) == ["→ [1, 32, 12, 12] → [1, 5, 12, 12]"]
    assert "    nn.Conv2d(32, 5, 1)," in mono(a, 2)
    four = texts(a, 3, 0.9)
    assert {"80 numbers", "1,168 numbers", "4,640 numbers", "165 numbers", "6,053",
            "numbers to learn"} <= set(four)
    assert "80 + 1,168 + 4,640 + 165 = 6,053" in a.subtitle_for(3)
    assert a.evidence_for(3).startswith("exact arithmetic")
    # what one cell sees: the square grows step by step to 22 x 22
    seen = [next(t for t in texts(a, 4, u) if not B.is_evidence(t) and (
        t.endswith(" pixels") or t == "1 pixel")) for u in (0.09, 0.15, 0.27, 0.39, 0.51, 0.63,
                                                             0.9)]
    assert seen == ["1 pixel", "3 × 3 pixels", "4 × 4 pixels", "8 × 8 pixels",
                    "10 × 10 pixels", "18 × 18 pixels", "22 × 22 pixels"]
    assert m["reach_sizes"][:6] == [3, 4, 8, 10, 18, 22]
    assert all(a.station_for(k) == "forward" for k in range(5))

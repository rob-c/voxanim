"""Animations 263-264 (voxanim.mech_blocks_zb): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, nothing is trained, the photo is named by number only, and the numbers and code
lines on screen are the study's and the script's. Both show photo 0 of the Olivetti faces."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_faces

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_zb as M
from voxanim.next_catalog import build_study

NAMES = ("resample-down", "resample-up")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_zb", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


# ---------------------------------------------------------------------- both
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
            assert any(B.is_evidence(t) for t in texts(a, k, u))
        assert sum(bool(t.strip()) for t in texts(a, k, 0.6)) >= 3
        assert layout.text_width(a.evidence_for(k), B.SIZE) <= 0.95 - B.PANEL[0]
        assert B.is_evidence(a.evidence_for(k))


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
        assert {B.code_file(n) for n in code.names()} == {"examples/resample.py"}


@needs_faces
@pytest.mark.parametrize("name", NAMES)
def test_nothing_is_trained(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        assert a.station_for(k) is None
        assert "you are here" not in texts(a, k, 0.6)


@needs_faces
@pytest.mark.parametrize("name", NAMES)
def test_the_photo_is_named_by_number_only(name):
    a, _ = built(name)
    said = " ".join([a.subtitle_for(k) for k in range(len(a.PHASES))]
                    + [t for k in range(len(a.PHASES)) for u in (0.3, 0.9) for t in texts(a, k, u)])
    words = set(said.lower().replace(",", " ").replace(".", " ").replace(":", " ").split())
    for w in ("man", "men", "woman", "women", "male", "female", "boy", "girl", "old", "young",
              "white", "black", "smiling", "glasses", "beard", "he", "she", "his", "her"):
        assert w not in words, w
    assert "photo 0" in said.lower()


@needs_faces
def test_the_titles_and_the_chain_are_the_catalogues():
    from voxanim.next_catalog import ITEMS
    titles = {i["name"]: i["title"] for i in ITEMS}
    for name in NAMES:
        assert titles[name] == M.ANIMATIONS[name].TITLE
    assert M.Smaller.LAST == "Edge detection with Sobel filters"
    assert M.Smaller.NEXT == titles["resample-up"] == M.Bigger.TITLE
    assert M.Bigger.LAST == titles["resample-down"] == M.Smaller.TITLE
    assert M.Bigger.NEXT == "Bigger: a kernel larger than the picture"


# ---------------------------------------------------------------------- 263
@needs_faces
def test_smaller():
    a, s = built("resample-down")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert one.count("1/4") == 4 and one.count("2/4") == 2          # the row and the column
    assert {"1/16", "2/16", "4/16", "[1, 2, 1] / 4", "×", "adds up to", "16/16 = 1"} <= set(one)
    assert one.count("1/16") == 4 and one.count("2/16") == 4 and one.count("4/16") == 1
    assert "blur = torch.tensor([1.0, 2.0, 1.0]) / 4" in mono(a, 0)
    two = texts(a, 1, 0.9)
    assert {"photo 0, 64 × 64", "32 × 32", "9 pixels", "kernel", "=", "0.53"} <= set(two)
    assert {"0.57", "0.38", "0.40", "0.69", "0.54", "0.45", "0.64", "0.60", "0.52"} <= set(two)
    assert two.count("1/16") == 4                                   # the kernel stays
    assert "0.53" in a.subtitle_for(1) and m["output_text"] == "0.53"
    assert a.evidence_for(1) == "exact arithmetic: photo 0's 9 pixels at (18, 46)"
    assert "small = F.conv2d(x, k, stride=2," in mono(a, 1)
    three = texts(a, 2, 0.9)
    assert {"new pixels", "1,024", "2 at a time"} <= set(three)
    assert "1,024" in a.subtitle_for(2)
    four = texts(a, 3, 0.9)
    assert {"row 0", "row 3", "pixel", "each row: 2 places on", "D: 32 × 64"} <= set(four)
    assert four.count("0.5") == 4 and four.count("0.25") == 7       # row 0 holds one 0.25
    assert "past the edge" in a.subtitle_for(3)
    assert "            D[i, j] = w" in mono(a, 3)
    five = texts(a, 4, 0.9)
    assert {"zone plate", "D @", "@ D.T", "32 × 64", "32 × 32", "conv2d",
            "1.8e-7 apart: rounding"} <= set(five)
    assert "→ tensor(1.7881e-07)" in mono(a, 4, 0.9)
    assert "rounding" in a.subtitle_for(4)
    six = texts(a, 5, 0.9)
    assert {"zone plate", "skip only", "blur, then skip", "false rings", "0.354", "0.079",
            "outside the circle: too fine for 32 × 32"} <= set(six)
    assert "0.354" in a.subtitle_for(5) and "0.079" in a.subtitle_for(5)
    assert a.evidence_for(5) == "exact arithmetic: the zone plate, 827 pixels too fine"
    assert all(e.startswith("exact arithmetic: ") for e in a.EVIDENCE)


# ---------------------------------------------------------------------- 264
@needs_faces
def test_bigger():
    a, s = built("resample-up")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"32 × 32", "64 × 64", "1,024 pixels kept", "3,072 zeros added"} <= set(one)
    assert one.count("0") >= 16                                      # the zeros of the close-up
    assert {"0.58", "0.42", "0.33", "0.64", "0.53", "0.44", "0.55"} <= set(one)
    two = texts(a, 1, 0.3)
    assert "(0.64 + 0.53) / 2 = 0.59" in two and "0.59" in two
    assert "average of 4 = 0.54" in texts(a, 1, 0.55)
    done = texts(a, 1, 0.9)
    assert {"bilinear", "4 × k", "each gap: an average", "0.50", "0.61", "0.54", "0.59"} <= \
        set(done)
    assert done.count("0.25") == 4 and done.count("1") == 1          # the kernel x 4
    assert "bilinear" in a.subtitle_for(1)
    assert "    y, 4 * k, stride=2, padding=1," in mono(a, 1)
    three = texts(a, 2, 0.9)
    assert {"nearest", "bilinear", "copies: 2 × 2 blocks"} <= set(three)
    assert three.count("0.58") >= 6 and three.count("0.42") == 4
    assert "blocks = blocks.repeat_interleave(2, 3)" in mono(a, 2)
    four = texts(a, 3, 0.9)
    assert {"got = up(ones)", "big / got", "last row, on average", "0.28 → 0.57", "0.25"} <= \
        set(four)
    assert four.count("0.5") == 4 and four.count("1") >= 4
    assert "0.28" in texts(a, 3, 0.5)
    assert "fixed = big / got" in mono(a, 3)
    five = texts(a, 4, 0.9)
    assert {"inside the 1-pixel border", "the whole picture", "blur, bilinear", "blur, nearest",
            "blur, even edge", "0.00129", "0.00302", "0.00343", "0.00199", "× 400 photos"} <= \
        set(five)
    assert "0.00092" not in five
    assert a.evidence_for(4) == "measured on Olivetti faces: 400 photos, down then up"
    six = texts(a, 5, 0.9)
    assert {"skip, bilinear", "0.00092", "rings 0.354", "rings 0.079", "0.00129"} <= set(six)
    assert "0.00092" in a.subtitle_for(5) and "no false patterns" in a.subtitle_for(5)
    assert a.heading(5, 0.2) == "skip lands closer; blur avoids false rings"
    assert m["inner_skip_bilinear"] < m["inner_bilinear"]
    assert a.evidence_for(5).startswith("measured on Olivetti faces: ")

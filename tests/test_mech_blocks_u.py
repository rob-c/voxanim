"""Animations 241-244 (voxanim.mech_blocks_u): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the script's."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import needs_faces

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_u as M
from voxanim.next_catalog import build_study

NAMES = ("faces-two-losses", "faces-training-the-detector", "faces-nms",
         "faces-precision-and-recall")
TAG = "measured on Olivetti faces: "
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_u", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


def test_the_evidence_tag_names_the_data():
    assert M.faces("x") == TAG + "x" == B.measured("x", on="Olivetti faces")
    assert B.is_evidence(M.faces("x")) and "MNIST" not in M.faces("x")


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
            assert a.evidence_for(k) in texts(a, k, u)
        assert sum(bool(t.strip()) for t in texts(a, k, 0.6)) >= 3
        assert layout.text_width(a.evidence_for(k), B.SIZE) <= 0.95 - B.PANEL[0]
        assert a.evidence_for(k).startswith(TAG) and B.is_evidence(a.evidence_for(k))


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
def test_the_titles_and_the_chain_are_the_catalogues():
    from voxanim.next_catalog import ITEMS
    titles = {i["name"]: i["title"] for i in ITEMS}
    for name in NAMES:
        assert titles[name] == M.ANIMATIONS[name].TITLE
    assert M.TwoLosses.LAST == titles["faces-the-detector"] == "From pixels to a grid"
    for a, b in itertools.pairwise(NAMES):
        assert titles[b] == M.ANIMATIONS[a].NEXT and titles[a] == M.ANIMATIONS[b].LAST
    assert titles["faces-a-face-is-numbers"] == M.PrecisionRecall.NEXT


@needs_faces
def test_only_training_lights_the_loop():
    stations = {name: [built(name)[0].station_for(k) for k in range(4)] for name in NAMES}
    assert stations["faces-two-losses"] == ["loss", "loss", "loss", None]
    assert stations["faces-training-the-detector"] == ["loss", "forward", "forward", "forward"]
    assert stations["faces-nms"] == ["forward", None, None, None]
    assert stations["faces-precision-and-recall"] == [None] * 4


@needs_faces
@pytest.mark.parametrize("name", NAMES)
def test_privacy_no_names_or_traits(name):
    """People are never named or described: no traits, no claim about "anyone"."""
    a, _ = built(name)
    said = [s for k in range(4) for u in (0.3, 0.9) for s in texts(a, k, u)]
    said += [a.subtitle_for(k) for k in range(4)]
    text = " %s " % " ".join(said).lower()
    for w in ("anyone", " man ", "woman", "male", "female", " age ", "smil", "happy", " race",
              "skin"):
        assert w not in text, w


# ---------------------------------------------------------------------- 241
@needs_faces
def test_two_losses():
    a, s = built("faces-two-losses")
    one = texts(a, 0, 0.9)
    assert {"step 0: 16 made scenes", "16 × 144 cells", "2,304", "hold a face", "38",
            "of the cells", "1.6%", "38 of 2,304 cells hold a face"} <= set(one)
    assert "obj, box = out[:, 0], out[:, 1:]" in mono(a, 0)
    assert "→ obj: [16, 12, 12]" in mono(a, 0, 0.9)
    assert "faces: 38 of 2,304" in texts(a, 1, 0.3) and "1.6%" in texts(a, 1, 0.3)
    two = texts(a, 1, 0.9)
    assert {"what 'there' counts", "faces × 10: 380 of 2,646", "14.4%", "there = 0.853",
            "no weight: 0.760", "there", "0.853"} <= set(two)
    assert "PW = torch.tensor(10.0)" in mono(a, 1)
    assert "14.4%" in a.subtitle_for(1) and "1.6%" in a.subtitle_for(1)
    three = texts(a, 2, 0.9)
    assert {"scene 0, cell (5, 3)", "want 0.06 0.56 0.66 0.66", "where: mean of 38",
            "= 0.438", "0.853 + 5 × 0.438", "= 3.044", "loss", "3.044",
            "grey: not counted in where"} <= set(three)
    assert "where = (miss * has).sum() / has.sum()" in mono(a, 2)
    assert "→ where: 0.438" in mono(a, 2, 0.9) and "  loss: 3.044" in mono(a, 2, 0.9)
    four = texts(a, 3, 0.95)
    assert {"AP, 200 unseen scenes", "faces × 10", "no weight", "0.92", "0.47",
            "found at 0.5, of 412", "348", "72", "one run each,", "same start, same scenes",
            "38 of 2,304 cells hold a face", "next: Training the detector"} <= set(four)
    assert four.count("faces × 10") == 2 and four.count("no weight") == 2
    assert "200 unseen scenes" in a.evidence_for(3)
    m = s.metadata
    assert m["cells_text"] in a.subtitle_for(0) and "38" in a.subtitle_for(0)


# ---------------------------------------------------------------------- 242
@needs_faces
def test_training():
    a, _ = built("faces-training-the-detector")
    one = texts(a, 0, 0.9)
    assert {"loss", "step", "there 0.85", "where 0.44", "there 0.06", "where 0.03",
            "after step 0: means of 100 steps", "loss = there + 5 × where",
            "3.04 at the start, 0.22 at the end"} <= set(one)
    assert "loss = detector_loss(out, has, want, pw)" in mono(a, 0)
    for k, u, step, above in ((1, 0.3, "step 0", 144), (1, 0.9, "step 100", 0),
                              (2, 0.3, "step 500", 0), (2, 0.9, "step 2,000", 5),
                              (3, 0.6, "step 6,000", 4)):
        shown = texts(a, k, u)
        assert step in shown and "above 0.5: %d" % above in shown
        assert "test scene 0" in shown and "a face here?" in shown
        assert "→ above 0.5: %d" % above in mono(a, k, u)
    assert "highest 0.54" in texts(a, 1, 0.3)
    assert "boxes: 3" in texts(a, 2, 0.9) and "1 too many" in texts(a, 2, 0.9)
    assert {"boxes: 2", "2 right", "next: One box per face"} <= set(texts(a, 3, 0.95))
    assert "0.98 and 0.97" in a.subtitle_for(3)
    assert "unseen" in a.subtitle_for(1)


# ---------------------------------------------------------------------- 243
@needs_faces
def test_one_box_per_face():
    a, _ = built("faces-nms")
    one = texts(a, 0, 0.9)
    assert {"test scene 0", "score · cell", "0.98 · (5, 8)", "0.97 · (7, 3)",
            "0.96 · (5, 9)", "0.75 · (5, 7)", "lit: above 0.5"} <= set(one)
    assert "→ 4 cells above 0.5" in mono(a, 0, 0.9)
    two = texts(a, 1, 0.9)
    assert {"sigmoid: 0.73, 0.90", "cx = (8 + 0.73) × 8 = 69.9", "cy = (5 + 0.90) × 8 = 47.2",
            "w = 16 × e^0.56 = 27.9", "4 boxes, in pixels"} <= set(two)
    assert "s = 16 * out[3:5, r, c].exp()" in mono(a, 1)
    three = texts(a, 2, 0.9)
    assert {"IoU", "0.77", "0.39", "IoU with the surest, 0.98", "3 boxes, one face"} <= set(three)
    assert "corners = box_convert(boxes, *FMT)" in mono(a, 2)
    four = texts(a, 3, 0.95)
    assert four.count("kept") == 2 and four.count("dropped") == 2
    assert {"drop if IoU > 0.3", "2 boxes, 2 faces", "200 scenes: 685 → 412 boxes",
            "next: Is the detector any good?"} <= set(four)
    assert "→ 2 boxes kept of 4" in mono(a, 3, 0.9)
    assert "kept = nms(corners, score, 0.3)" in mono(a, 3)
    assert "first test scene" in a.subtitle_for(0)


# ---------------------------------------------------------------------- 244
@needs_faces
def test_precision_and_recall():
    a, _ = built("faces-precision-and-recall")
    one = texts(a, 0, 0.95)
    assert {"test scene 3", "score · IoU", "0.95 · 0.84", "0.80 · 0.90", "0.58 · 0.00",
            "0.53 · 0.73", "wrong", "thin: true face · thick: found",
            "right: IoU ≥ 0.5"} <= set(one)
    assert one.count("right") == 3
    assert "print(box_iou(found, real))" in mono(a, 0)
    two = texts(a, 1, 0.9)
    assert {"at 0.5, over 200 unseen scenes", "faces", "boxes", "right", "wrong", "412", "348",
            "64", "recall: 348 of 412 faces found", "precision: 348 of 412 boxes right",
            "84.5%"} <= set(two)
    assert "r, c = torch.nonzero(p > keep).T" in mono(a, 1)
    assert "from 0.9: found 55.8%, right 99.6%" in texts(a, 2, 0.33)
    three = texts(a, 2, 0.95)
    assert {"precision: boxes right", "recall: faces found", "AP = area = 0.92", "0.5",
            "from 0.1: found 97.8%, right 52.1%"} <= set(three)
    four = texts(a, 3, 0.95)
    assert {"AP = area = 0.92", "no weight", "AP 0.47",
            "at 0.5: 348 found with the weight, 72 without",
            "next: A face is 4,096 numbers"} <= set(four)
    assert "PW = torch.tensor(10.0)" in mono(a, 3)
    assert "56.8%" in a.subtitle_for(3) and "0.92" in a.subtitle_for(2)

"""Animations 269-270 (voxanim.mech_blocks_zd): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, the 3-D space is drawn the way its labels say, every call of the record is shown in
its order, and the numbers and code lines on screen are the study's and the script's. No
dataset is needed: the data are made up by the script."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_zd as M
from voxanim.next_catalog import ITEMS, build_study

NAMES = ("minuit-feels-its-way", "minuit-and-sgd")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_zd", name))
    return _built[name]


def at(a, phase, u):
    lo, hi = a.spans[phase]
    return lo + (hi - lo) * u


def texts(a, phase, u=0.6, kind=None):
    return [x.s for x in a.scene(at(a, phase, u)).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def texts_at(a, t):
    return [x.s for x in a.scene(float(t)).texts if x.opacity > 0.5]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


# ---------------------------------------------------------------------- no drawing needed
def test_the_titles_and_the_chain_are_the_catalogues():
    from voxanim.mech_stats_fit import BinnedOrNot as Binned
    from voxanim.mech_stats_likelihood import ANIMATIONS as LIKELIHOOD

    titles = {i["name"]: i["title"] for i in ITEMS}
    classes = [M.ANIMATIONS[n] for n in NAMES]
    for name, cls in zip(NAMES, classes, strict=True):
        assert titles[name] == cls.TITLE
    assert M.Feels.LAST == "Binned or unbinned?" == Binned.TITLE
    for before, after in pairwise(classes):
        assert before.NEXT == after.TITLE and after.LAST == before.TITLE
    assert M.Versus.NEXT == "The likelihood curve"
    assert M.Versus.NEXT in {c.TITLE for c in LIKELIHOOD.values()}
    assert {i["section"] for i in ITEMS if i["name"] in NAMES} == {"n. Minuit and SGD (2)"}


def test_the_space_is_drawn_as_its_labels_say():
    S = M.Space()
    lo = np.array(S.LO)
    x0, y0 = S.at(lo)
    # a runs left to right, b up, tau back (up the picture and smaller: perspective)
    assert S.at(lo + [9, 0, 0])[0] > x0 + 0.1
    assert S.at(lo + [0, 0, 2.2])[1] < y0 - 0.1
    back = S.at(lo + [0, 3.5, 0])
    assert back[1] < y0 and back[0] > x0
    near = np.hypot(*np.subtract(S.at([9, 0, 2.2]), S.at([0, 0, 2.2])))
    far = np.hypot(*np.subtract(S.at([9, 3.5, 2.2]), S.at([0, 3.5, 2.2])))
    assert far < near
    for i in range(3):
        assert np.isclose(np.hypot(*S.axis_dir([5, 2, 1], i)), 1.0)
    # every point the fits visit is inside the picture's space
    x, x1, top, bottom = M.SPACE_BOX
    m = build_study("study_blocks_zd", "minuit-and-sgd").metadata
    pts = [e["point"] for e in build_study("study_blocks_zd", "minuit-feels-its-way")
           .metadata["timeline"]] + m["path"]
    for p in pts:
        fx, fy = S.at(p)
        assert x - 1e-6 <= fx <= x1 + 1e-6 and top - 1e-6 <= fy <= bottom + 1e-6


def test_the_ways_a_probe_moves_are_the_axes_directions():
    assert M.WAY == (("left", "right"), ("forward", "back"), ("down", "up"))


# ---------------------------------------------------------------------- every one of them
@pytest.mark.parametrize("name", NAMES)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


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


@pytest.mark.parametrize("name", NAMES)
def test_nothing_blanks_between_phases(name):
    a, _ = built(name)
    for k, (_lo, hi) in enumerate(a.spans[:-1]):
        sc = a.scene(hi)                                   # the content's dip
        shown = [x.s for x in sc.texts if x.opacity > 0.5]
        if a.evidence_for(k) == a.evidence_for(k + 1):
            assert a.evidence_for(k) in shown
        if a.code_for(k).lines() == a.code_for(k + 1).lines():
            assert any(x.kind == "mono" and x.opacity > 0.5 for x in sc.texts)
        assert len(sc.quads) >= 6
        assert sum(x.opacity > 0.5 and bool(x.s.strip()) for x in sc.texts) >= 8
        # the picture itself is never empty: the space's axes and the counter stay
        picture = [x.s for x in sc.texts if x.opacity > 0.5 and x.x < B.PANEL[0]
                   and x.y > B.HEAD_Y + 0.03]
        assert {"a", "τ", "b"} <= set(picture), (name, k)
        assert any(s.startswith("χ² ") for s in picture)


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
        tag = a.evidence_for(k)
        assert layout.text_width(tag, B.SIZE) <= 0.95 - B.PANEL[0]
        assert tag.startswith(("measured on made-up data: ", "exact arithmetic: "))


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


@pytest.mark.parametrize("name", NAMES)
def test_the_code_on_screen_is_the_scripts(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        shown = [s for s in mono(a, k, 0.6) if not s.startswith("→")]
        assert shown == list(code.lines())
        lines = [s for n in code.names() for s in B.code_lines(n)]
        assert all(s in lines for s in shown)
        assert {B.code_file(n) for n in code.names()} == {"examples/minuit_vs_sgd.py"}


@pytest.mark.parametrize("name", NAMES)
def test_fitting_is_not_the_training_loop(name):
    a, _ = built(name)
    assert all(a.station_for(k) is None for k in range(len(a.PHASES)))
    assert "you are here" not in texts(a, 2, 0.5)


# ---------------------------------------------------------------------- 269
def test_every_call_is_shown_in_minuits_order():
    a, s = built("minuit-feels-its-way")
    tl = s.metadata["timeline"]
    assert len(a.call_t) == len(tl) == 81
    assert (np.diff(a.call_t) > 0.1).all()              # one at a time, in order
    assert a.call_t[-1] < a.t_done - 2.0
    # the first 12 probes slowly, nearly a second each, before the first step
    assert (np.diff(a.call_t[1:13]) > 0.8).all() and a.phase_at(a.call_t[1]) == 1
    assert a.phase_at(a.call_t[13]) == a.phase_at(a.call_t[14]) == 2
    assert a.phase_at(a.call_t[64]) == 4 and a.phase_at(a.call_t[65]) == 5
    # each call's chi-square is on screen while it is the latest
    for i in (0, 1, 3, 9, 13, 14, 26, 42, 64, 80):
        t = a.call_t[i] + 0.12 if i == 80 else (a.call_t[i] + a.call_t[i + 1]) / 2
        assert "χ² " + s.metadata["chi2_text"][i] in texts_at(a, t), i


def test_the_counter_says_which_way_each_probe_went():
    a, s = built("minuit-feels-its-way")
    m = s.metadata
    mid = lambda i: (a.call_t[i] + a.call_t[i + 1]) / 2       # noqa: E731
    assert {"call 1: a right 0.00025", "χ² 2,691.51", "−0.11  lower"} <= set(texts_at(a, mid(1)))
    assert {"call 4: a left 0.0025", "+1.1  higher"} <= set(texts_at(a, mid(4)))
    assert "call 7: τ back 0.0025" in texts_at(a, mid(7))
    assert {"call 9: b up 0.0025", "χ² 2,686.77", "−4.8  lower"} <= set(texts_at(a, mid(9)))
    assert "call 12: b down 0.0012" in texts_at(a, mid(12))
    assert "flat: under 0.0001" in texts_at(a, mid(44))
    assert m["delta_text"][13] == "+762"


def test_the_probes_are_drawn_along_their_axes_in_the_magnified_inset():
    a, s = built("minuit-feels-its-way")
    tl = s.metadata["timeline"]
    assert "magnified" in texts(a, 1, 0.5)
    for rnd in range(s.metadata["rounds"]):
        near = [e for e in tl if e["round"] == rnd and e["near"]]
        base = tl[near[0]["base"]]["point"]
        scale = a.round_scale[rnd]
        reach = [np.hypot(*a.span(*a.offset(base, e["d"], scale))) for e in near]
        assert np.isclose(max(reach), M.INSET_R)          # each round fits the inset
        assert all(r <= M.INSET_R + 1e-6 for r in reach)
        for e in near:
            if e["kind"] != "probe":
                continue
            v = np.array(a.span(*a.offset(base, e["d"], scale)))
            d = a.space.axis_dir(base, e["axis"]) * e["sign"]
            assert np.allclose(v / np.hypot(*v), d, atol=1e-6)   # along its axis, its way
    # the inset stays inside its frame
    hw, hh = M.INSET_W / 2, M.INSET_H / 2
    assert min(hw, hh) > M.INSET_R and hh > M.AXIS_LEN + 9 + 8


def test_feels_numbers_on_screen():
    a, s = built("minuit-feels-its-way")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"a", "τ", "b", "the start", "χ² 2,691.61", "2,692"} <= set(one)
    assert "    loss = float((r ** 2).sum())" in mono(a, 0, 0.9)
    assert "m.migrad()" in mono(a, 1)
    three = texts(a, 2, 0.95)
    assert {"too far: 3,454", "109", "call 14: a step", "χ² 108.92", "−2,583  lower"} <= set(
        three)
    assert "3,454" in a.subtitle_for(2) and "109" in a.subtitle_for(2)
    five = texts(a, 4, 0.9)
    assert {"65 calls", "χ² 33.9447", "EDM 0.00014", "a = 5.14", "τ = 2.41",
            "b = 1.007"} <= set(five)
    assert "→ 65 calls" in mono(a, 4, 0.9)
    assert "EDM" in a.subtitle_for(4) and "52 probes, 12 tries, 5 kept" in a.subtitle_for(4)
    six = texts(a, 5, 0.97)
    assert {"65 + 16 calls", "± 0.18", "± 0.21", "± 0.11"} <= set(six)
    assert "→ " + m["printed"] in mono(a, 5, 0.97)
    assert "next: Minuit and SGD" in six
    assert "MIGRAD" in a.subtitle_for(1) and "HESSE" in a.subtitle_for(5)


# ---------------------------------------------------------------------- 270
def test_versus_numbers_on_screen():
    a, s = built("minuit-and-sgd")
    one = texts(a, 0, 0.9)
    assert {"65 calls, 5 kept", "χ² 33.94", "Minuit"} <= set(one)
    assert "→ MIGRAD: 65 calls; HESSE: 16 more" in mono(a, 0, 0.9)
    assert "step 1 of 3,000" in texts(a, 1, 0.7)
    two = texts(a, 1, 0.97)
    assert {"step 10 of 3,000", "batch: 8 of 40 points", "learning rate: 0.005",
            "step: rate × slope"} <= set(two)
    assert "g, _ = gradient(p, idx)" in mono(a, 1)
    for word in ("batch", "learning rate", "step"):
        assert word in a.subtitle_for(1)
    three = texts(a, 2, 0.95)
    assert "step 3,000 of 3,000" in three
    assert "χ² 34.36" in three                         # where the last step happens to be
    four = texts(a, 3, 0.95)
    assert {"last 1,000, averaged", "χ² 33.96", "Minuit: 33.94", "magnified",
            "SGD's spread ≈ 0.02", "Minuit's errors ≈ 0.2"} <= set(four)
    five = texts(a, 4, 0.95)
    assert {"24,000", "2,600", "SGD   3,000 × 8", "MIGRAD   65 × 40",
            "small fit: Minuit · huge data: SGD"} <= set(five)
    assert a.heading_for(4, 0.5) == "points the model ran on, each way"
    assert "next: The likelihood curve" in five
    m = s.metadata
    assert m["sgd_evaluations_text"] in a.subtitle_for(4)
    assert m["average_chi2_text"] in a.subtitle_for(3)


def test_the_walk_is_drawn_step_by_step():
    a, _ = built("minuit-and-sgd")
    seen = []
    for t in np.linspace(0.0, a.t_done, 400):
        a._t = float(t)
        seen.append(a.walk_upto())
    assert seen[0] == 0 and seen[-1] == 3000 and seen == sorted(seen)
    a._t = at(a, 1, a.FIRST_U + 0.05)
    assert a.walk_upto() == 1                           # the first step, alone
    a._t = at(a, 1, 0.99)
    assert a.walk_upto() == a.FIRST == 10
    a._t = at(a, 2, 0.4)
    assert a.walk_upto() == 200                         # the first 200 slowly, then the rest

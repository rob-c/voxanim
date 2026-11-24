"""Animations 272 and 278 (voxanim.mech_blocks_zf): the layout is clean, the loop closes,
nothing blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the scripts'. No dataset is
needed: the data are made up by the scripts."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_zf as M
from voxanim.next_catalog import ITEMS, build_study

NAMES = ("error-bars-on-a-network", "asimov-dataset")
SCRIPTS = {"error-bars-on-a-network": "examples/ensembles.py",
           "asimov-dataset": "examples/asimov.py"}
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_zf", name))
    return _built[name]


def at(a, phase, u):
    lo, hi = a.spans[phase]
    return lo + (hi - lo) * u


def texts(a, phase, u=0.6, kind=None):
    return [x.s for x in a.scene(at(a, phase, u)).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


# ---------------------------------------------------------------------- no drawing needed
def test_the_titles_and_the_chain_are_the_catalogues():
    from voxanim.mech_blocks_zd import Versus
    from voxanim.mech_stats_likelihood import ANIMATIONS as LIKELIHOOD

    titles = {i["name"]: i["title"] for i in ITEMS}
    for name in NAMES:
        assert titles[name] == M.ANIMATIONS[name].TITLE
    assert M.ErrorBars.LAST == Versus.TITLE == "Minuit and SGD"
    assert Versus.NEXT == M.ErrorBars.TITLE
    assert M.ErrorBars.NEXT == "The likelihood curve"
    assert M.ErrorBars.NEXT in {c.TITLE for c in LIKELIHOOD.values()}
    assert M.Asimov.LAST == titles["unfolding"] == "Unfolding: running the smearing backwards"
    assert M.Asimov(build_study("study_blocks_zf", "asimov-dataset")).last_line() == (
        "last time: Unfolding")
    assert M.Asimov.NEXT == titles["jets-as-sets"] == "Jets as sets of particles"


# ---------------------------------------------------------------------- both of them
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
        # the picture itself is never empty: its axes and the counter stay
        picture = {x.s for x in sc.texts if x.opacity > 0.5 and x.x < B.PANEL[0]
                   and x.y > B.HEAD_Y + 0.03}
        need = {"τ", "b"} if name == NAMES[0] else {"events", "mass"}
        assert need <= picture, (name, k)
        assert len(picture) >= 8, (name, k)


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
    assert "made-up" in a.subtitle_for(0)                 # said where the data first appear


@pytest.mark.parametrize("name", NAMES)
def test_headings_are_short_and_start_and_end_the_same_way(name):
    a, _ = built(name)
    for t in np.arange(0.0, a.duration, 0.5):
        h = a.heading_at(float(t))
        assert B.words(h) <= 12, h
        assert layout.text_width(h, B.HEAD) <= B.PANEL[0] - B.HEAD_X - 0.01, h
    first = a.heading_at(1.0)
    assert first.startswith("last time: ") and a.LAST.startswith(first[len("last time: "):])
    assert a.heading_at(a.t_done - 1.0) == "next: " + a.NEXT
    assert first in texts(a, 0, 0.1)


@pytest.mark.parametrize("name", NAMES)
def test_the_code_on_screen_is_the_scripts(name):
    a, _ = built(name)
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        n = len(code.lines())
        shown = mono(a, k, 0.6)
        assert shown[:n] == list(code.lines())             # the code, then the output slot
        assert all(s.startswith(("→ ", "  ")) for s in shown[n:])
        assert {B.code_file(r) for r in code.names()} == {SCRIPTS[name]}


@pytest.mark.parametrize("name", NAMES)
def test_fitting_and_counting_are_not_the_training_loop(name):
    a, _ = built(name)
    assert all(a.station_for(k) is None for k in range(len(a.PHASES)))
    assert "you are here" not in texts(a, 2, 0.5)


# ---------------------------------------------------------------------- 272
def test_error_bars_numbers_on_screen():
    a, s = built("error-bars-on-a-network")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"40 made-up points", "Minuit", "run 1, seed 0", "τ = 2.44", "b = 1.02"} <= set(one)
    assert "    return p" in mono(a, 0, 0.9)
    two = texts(a, 1, 0.95)
    assert {"100 new seeds", "± 0.02", "magnified × 4"} <= set(two)
    assert "→ [0.017 0.016 0.021]" in mono(a, 1, 0.95)
    assert "0.02" in a.subtitle_for(1)
    three = texts(a, 2, 0.9)
    assert {"resample 1", "12 left out", "11 twice or more",
            "bigger: picked more often · hollow: left out"} <= set(three)
    assert "12 are left out; 11 count twice or more" in a.subtitle_for(2)
    four = texts(a, 3, 0.95)
    assert {"100 resamples", "τ ± 0.14", "b ± 0.11"} <= set(four)
    counted = [s for t in np.linspace(0.1, 0.6, 60) for s in texts(a, 3, t)
               if s.startswith("resample ") and s.endswith(" of 100")]
    assert counted[0] == "resample 1 of 100" and counted[-1] == "resample 100 of 100"
    assert len(set(counted)) > 10                      # counted up, one run at a time
    five = texts(a, 4, 0.9)
    assert {"tilt −0.82"} <= set(five)
    assert "→ [0.15 0.14 0.11]" in mono(a, 4, 0.9)
    assert m["corr_boots_text"] in a.subtitle_for(4)
    six = texts(a, 5, 0.95)
    assert {"new seeds", "resamples", "HESSE", "± 0.017", "± 0.15", "± 0.18", "± 0.016",
            "± 0.14", "± 0.21", "± 0.021", "± 0.11", "τ ± 0.21", "tilt −0.81"} <= set(six)
    assert "→ [0.18 0.21 0.11]" in mono(a, 5, 0.95)
    assert "next: The likelihood curve" in six
    assert "a third short in τ" in a.subtitle_for(5)


def test_the_answers_arrive_one_run_at_a_time():
    a, _ = built("error-bars-on-a-network")
    seen_s, seen_b = [], []
    for t in np.linspace(0.0, a.t_done, 500):
        a._t = float(t)
        seen_s.append(a.n_seeds())
        seen_b.append(a.n_boots())
    assert seen_s == sorted(seen_s) and seen_s[0] == 0 and seen_s[-1] == 100
    assert seen_b == sorted(seen_b) and seen_b[0] == 0 and seen_b[-1] == 100
    a._t = at(a, 0, 0.9)
    assert a.n_seeds() == 1 and a.n_boots() == 0          # one run's answer, alone
    a._t = at(a, 2, 0.9)
    assert a.n_seeds() == 100 and a.n_boots() == 1        # resample 1's answer, alone


def test_the_plane_draws_tau_across_and_b_up():
    a, s = built("error-bars-on-a-network")
    lo = a.at((0.0, 2.2, 0.8))
    assert a.at((0.0, 2.8, 0.8))[0] > lo[0] + 0.1          # tau to the right
    assert a.at((0.0, 2.2, 1.2))[1] < lo[1] - 0.1          # b up
    (x0, x1), (top, bottom) = M.PLANE
    for p in s.metadata["boots"]:
        fx, fy = a.at(p)
        assert x0 <= fx <= x1 and top <= fy <= bottom


# ---------------------------------------------------------------------- 278
def test_asimov_numbers_on_screen():
    a, s = built("asimov-dataset")
    one = texts(a, 0, 0.9)
    assert {"33", "28", "20", "11", "signal s", "background b", "4 made-up bins"} <= set(one)
    assert "b = np.array([30.0, 20.0, 12.0, 8.0])" in mono(a, 0)
    two = texts(a, 1, 0.9)
    assert {"35", "33", "17", "14", "toy 1", "Z = 3.38 σ", "signal × 1.19"} <= set(two)
    three = texts(a, 2, 0.95)
    assert {"10,000 toys", "median 2.89"} <= set(three)
    assert "→ 2.89" in mono(a, 2, 0.95)
    four = texts(a, 3, 0.95)
    assert {"33", "28", "20", "11", "median 2.89", "Asimov 2.93"} <= set(four)
    assert "→ 2.93" in mono(a, 3, 0.95)
    five = texts(a, 4, 0.9)
    assert {"median 2.89", "Asimov 2.93", "s/√b 3.16"} <= set(five)
    assert "→ 3.16" in mono(a, 4, 0.9)
    six = texts(a, 5, 0.95)
    assert {"6", "10", "9.2", "3.8", "median 6.74", "Asimov 6.77", "s/√b 9.98"} <= set(six)
    assert {"→ background [3.  2.  1.2 0.8]", "  6.74", "  6.77", "  9.98"} <= set(mono(a, 5,
                                                                                       0.95))
    assert "next: Jets as sets of particles" in six
    m = s.metadata
    for k, key in ((2, "median_text"), (3, "asimov_text"), (4, "simple_text"),
                   (5, "simple_text_less")):
        assert m[key] in a.subtitle_for(k)


def test_the_toys_pile_up_and_the_markers_move_one_at_a_time():
    a, _ = built("asimov-dataset")
    counts = []
    for u in np.linspace(0.05, 0.95, 40):
        rows = [s for s in texts(a, 2, float(u)) if s.endswith((" toy", " toys"))]
        counts.append(int(rows[0].split()[0].replace(",", "")))
    assert counts[0] == 1 and counts[-1] == 10000 and counts == sorted(counts)
    # phase 6: first the background shrinks, then the toys, then the Asimov dot, then s/√b
    spans = (M.Asimov.SHRINK_U, M.Asimov.MORPH_U, M.Asimov.MOVE_U, M.Asimov.JUMP_U)
    for (_a0, a1), (b0, _b1) in pairwise(spans):
        assert a1 < b0
    a._t = at(a, 5, 0.3)
    assert a.shrink() == 1.0

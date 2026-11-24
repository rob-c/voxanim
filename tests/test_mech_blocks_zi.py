"""Animation 277 (voxanim.mech_blocks_zi): the layout is clean, the loop closes, nothing blanks
between phases, every phase says what kind of evidence it shows, the headings are short, the
matrix is drawn with true bins as columns and measured bins as rows, and the numbers and code
lines on screen are the study's and the script's. No dataset is needed: the events are made
up by the script."""

from __future__ import annotations

import numpy as np

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_zi as M
from voxanim.next_catalog import ITEMS, build_study

NAME = "unfolding"
_built = {}


def built():
    """``(animation, study)``, built once."""
    if NAME not in _built:
        _built[NAME] = (presets.get(NAME).build(), build_study("study_blocks_zi", NAME))
    return _built[NAME]


def at(a, phase, u):
    lo, hi = a.spans[phase]
    return lo + (hi - lo) * u


def texts(a, phase, u=0.6, kind=None):
    return [x.s for x in a.scene(at(a, phase, u)).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


def picture(sc):
    """The texts of the picture: left of the code panel, under the heading."""
    return [x.s for x in sc.texts if x.opacity > 0.5 and x.x < B.PANEL[0]
            and x.y > B.HEAD_Y + 0.03]


# ---------------------------------------------------------------------- no drawing needed
def test_the_title_and_the_chain_are_the_catalogues():
    by_number = {i["number"]: i for i in ITEMS}
    item = by_number[277]
    assert item["name"] == NAME and item["title"] == M.Unfolding.TITLE
    assert M.Unfolding.LAST == by_number[276]["title"] == "Simulation is not data: scale factors"
    assert M.Unfolding.NEXT == by_number[278]["title"] == "The Asimov dataset"
    assert set(M.ANIMATIONS) == {NAME}


def test_the_layout_fits_the_picture_area():
    a, _ = built()
    x0, y0, x1, _y1 = a.mat_box()
    (tx0, tx1), (_ty0, ty1) = M.TOP_BOX
    assert np.isclose(tx0, x0) and np.isclose(tx1, x1) and ty1 < y0    # truth over R's columns
    right_end = M.RIGHT_X + M.RIGHT_LEN * 1000
    assert x1 < M.RIGHT_X and right_end < B.PIC[1]                    # data beside its rows
    (bx0, bx1), (_by0, by1) = M.BIG_BOX
    assert B.PIC[0] < bx0 and bx1 <= B.PIC[1] and by1 < M.LINE1 < M.LINE2 <= 0.9
    blown_end = M.B_X + M.B_LEN * 2800
    assert blown_end < B.PIC[1] and M.S_X + M.S_LEN < M.N_X - M.N_LEN * 75


# ---------------------------------------------------------------------- the animation
def test_layout_is_clean():
    assert qa.check(NAME, samples=24, verbose=False) == 0


def test_drawn_by_this_module_and_loop_closes():
    a, _ = built()
    assert type(a) is M.Unfolding and isinstance(a, B.Blocks)
    assert 45.0 <= a.t_done <= 61.5 and a.duration <= 66.0 and len(a.PHASES) >= 4

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:
        assert len(a.scene((lo + hi) / 2).quads) >= 3
    for t in np.arange(0.0, a.duration, 0.37):           # a coarse pass: every frame builds
        assert a.scene(float(t)).texts is not None


def test_nothing_blanks_between_phases():
    a, _ = built()
    for k, (_lo, hi) in enumerate(a.spans[:-1]):
        sc = a.scene(hi)                                   # the content's dip
        shown = [x.s for x in sc.texts if x.opacity > 0.5]
        if a.evidence_for(k) == a.evidence_for(k + 1):
            assert a.evidence_for(k) in shown
        if a.code_for(k).lines() == a.code_for(k + 1).lines():
            assert any(x.kind == "mono" and x.opacity > 0.5 for x in sc.texts)
        assert len(sc.quads) >= 6
        assert sum(x.opacity > 0.5 and bool(x.s.strip()) for x in sc.texts) >= 8
        # the picture itself is never empty: the phase's chart, matrix or table stays
        assert len(picture(sc)) >= 3, k


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
        tag = a.evidence_for(k)
        assert layout.text_width(tag, B.SIZE) <= 0.95 - B.PANEL[0]
        assert tag.startswith(("measured on made-up data: ", "exact arithmetic: "))


def test_headings_are_short_and_start_and_end_the_same_way():
    a, _ = built()
    for t in np.arange(0.0, a.duration, 0.5):
        h = a.heading_at(float(t))
        assert B.words(h) <= 12, h
        assert layout.text_width(h, B.HEAD) <= B.PANEL[0] - B.HEAD_X - 0.01, h
    first = a.heading_at(1.0)
    assert first == "last time: Simulation is not data: scale factors"
    assert a.heading_at(a.t_done - 1.0) == "next: The Asimov dataset"
    assert first in texts(a, 0, 0.1)


def test_the_code_on_screen_is_the_scripts():
    a, s = built()
    printed = s.metadata["printed"]
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        shown = [t for t in mono(a, k, 0.6)
                 if not (t.startswith("→") or t.strip() in printed)]     # the output slot
        assert shown == list(code.lines())
        lines = [t for n in code.names() for t in B.code_lines(n)]
        assert all(t in lines for t in shown)
        assert {B.code_file(n) for n in code.names()} == {"examples/unfolding.py"}
        for _u, out in code.out:                           # the output slot: what it prints
            assert all(line in printed for line in out.split("\n"))


def test_unfolding_is_not_the_training_loop():
    a, _ = built()
    assert all(a.station_for(k) is None for k in range(len(a.PHASES)))
    assert "you are here" not in texts(a, 2, 0.5)


# ---------------------------------------------------------------------- what is on screen
def test_the_truth_and_r():
    a, s = built()
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"10,000 made-up events", "a bump at 60 GeV", "true energy, GeV", "events"} <= set(one)
    assert "→ truth total 10000; 50-60 GeV: 1194" in mono(a, 0, 0.9)
    two = texts(a, 1, 0.9)
    assert {"truth", "measured", "bin ↓", "true bin →", "R", "1,000,000", "582,195",
            "each column ÷", "simulated in it"} <= set(two)
    assert "500 dots shown" in texts(a, 1, 0.3)
    assert "R = H / n" in mono(a, 1, 0.9)
    # the dots: true energy to the right, measured energy down, inside R's square
    x0, y0, x1, y1 = a.mat_box()
    for tt, rr in zip(m["sim_true"][:20], m["sim_measured"][:20], strict=True):
        fx, fy = x0 + (x1 - x0) * tt / 100, y0 + (y1 - y0) * rr / 100
        assert x0 <= fx <= x1 and y0 <= fy <= y1
    # R's row i is measured bin i, its column j true bin j
    cell = a.mat_cell()
    assert cell(0, 9)[0] > cell(0, 0)[0] and cell(9, 0)[1] > cell(0, 0)[1]


def test_measured_inverted_why():
    a, s = built()
    m = s.metadata
    three = texts(a, 2, 0.2)
    assert {"50–60 GeV: 1,194 true", "→ 911 found, spread", "measured"} <= set(three)
    assert {"all bins: 10,000 true", "→ 6,345 expected"} <= set(texts(a, 2, 0.5))
    assert {"data: 6,362 counts", "noise: about ±25 a bin"} <= set(texts(a, 2, 0.9))
    assert "→ expected 6345, data 6362" in mono(a, 2, 0.9)
    assert "37%" in a.subtitle_for(2)
    four = texts(a, 3, 0.9)
    assert {"no noise: the truth, 3.6×10⁻¹² apart", "with noise: −900 to 3,475"} <= set(four)
    assert {"back - truth: 3.6e-12", "naive: -900 to 3475"} <= {t.lstrip("→ ") for t in
                                                               mono(a, 3, 0.9)}
    five = texts(a, 4, 0.95)
    assert {"s", "noise", "noise ÷ s", "0.75", "0.0078", "1,789", "−2,319", "2,759",
            "the last 3 rows: 99.9% of the swing"} <= set(five)
    assert m["s_text"][0] in a.subtitle_for(4) and m["s_text"][-1] in a.subtitle_for(4)


def test_iterations_and_pseudo_experiments():
    a, s = built()
    m = s.metadata
    assert a.iteration(0.1) == 0 and a.iteration(0.99) == 10
    seen = [a.iteration(u) for u in np.linspace(0, 1, 200)]
    assert seen == sorted(seen) and set(seen) == set(range(11))
    six = texts(a, 5, 0.15)
    assert {"start", "start: the simulation's shape, no bump"} <= set(six)
    assert "iteration 10" in texts(a, 5, 0.95)
    assert "→ 10 iterations: 50-60 GeV holds 1198" in mono(a, 5, 0.95)
    assert "D'Agostini" in a.subtitle_for(5)
    seven = texts(a, 6, 0.9)
    assert {"10 iterations: bias 41, spread 53", "R inverted: spread 1,166", "bias", "spread",
            "iterations", "2,000 pseudo-experiments"} <= set(seven)
    assert "→ 10 iterations: bias 41, spread 53" in mono(a, 6, 0.9)
    assert "near 10" in a.subtitle_for(6) and "1,166" in a.subtitle_for(6)
    assert m["best"] == 10

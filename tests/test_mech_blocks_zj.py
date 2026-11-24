"""Animation 279 (voxanim.mech_blocks_zj): the layout is clean, the loop closes, nothing blanks
between phases, every phase says what kind of evidence it shows, the headings are short, the
list's rows move as the shuffle says, and the numbers and code lines on screen are the study's
and the script's. No dataset is needed: the jets are made up by the script."""

from __future__ import annotations

import numpy as np
import pytest

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_zj as M
from voxanim import study_jets as J
from voxanim.next_catalog import ITEMS, NO_DATA, build_study

NAME = "jets-as-sets"
_built = {}


def built():
    """``(animation, study)``, built once."""
    if NAME not in _built:
        _built[NAME] = (presets.get(NAME).build(), build_study("study_blocks_zj", NAME))
    return _built[NAME]


def at(a, phase, u):
    lo, hi = a.spans[phase]
    return lo + (hi - lo) * u


def texts(a, phase, u=0.6, kind=None):
    return [x.s for x in a.scene(at(a, phase, u)).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


# ---------------------------------------------------------------------- no drawing needed
def test_the_title_and_the_chain_are_the_catalogues():
    titles = {i["number"]: i["title"] for i in ITEMS}
    item = next(i for i in ITEMS if i["name"] == NAME)
    assert item["number"] == 279 and item["title"] == M.JetsAsSets.TITLE
    assert item["section"] == "q. Jets as sets of particles (1)"
    assert M.JetsAsSets.LAST == titles[278] == "The Asimov dataset"
    assert M.JetsAsSets.NEXT == "Newton's step and gradient descent"
    assert M.JetsAsSets.NEXT in titles.values()
    assert NAME in NO_DATA                              # made up: nothing to download
    assert M.J_MAX == J.MAX


# ---------------------------------------------------------------------- the whole animation
def test_layout_is_clean():
    assert qa.check(NAME, samples=24, verbose=False) == 0


def test_drawn_by_this_module_and_loop_closes():
    a, _ = built()
    assert type(a) is M.JetsAsSets and isinstance(a, B.Blocks)
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
        assert len(sc.quads) >= 6
        assert sum(x.opacity > 0.5 and bool(x.s.strip()) for x in sc.texts) >= 8
        # the list stays on the left from phase 2 on, its columns named
        if k >= 1:
            assert {"z", "Δη", "Δφ", "mask", "11 real, 9 pads"} <= set(shown), k


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
        assert tag.startswith(("measured on made-up data: ", "exact arithmetic: ",
                               "about the data: made-up"))


def test_headings_are_short_and_start_and_end_the_same_way():
    a, _ = built()
    for t in np.arange(0.0, a.duration, 0.5):
        h = a.heading_at(float(t))
        assert B.words(h) <= 12, h
        assert layout.text_width(h, B.HEAD) <= B.PANEL[0] - B.HEAD_X - 0.01, h
    first = a.heading_at(1.0)
    assert first == "last time: The Asimov dataset"
    assert a.heading_at(a.t_done - 1.0) == "next: Newton's step and gradient descent"
    assert first in texts(a, 0, 0.1)
    # each phase's own heading is up for 3 s or more before "next" takes over
    for k in range(len(a.PHASES)):
        lo, hi = a.spans[k]
        own = [t for t in np.arange(lo, hi, 0.1) if a.heading_at(float(t)) == a.heading_for(k, 0)]
        assert len(own) * 0.1 >= 3.0, k


def test_the_code_on_screen_is_the_scripts():
    a, _ = built()
    for k in range(len(a.PHASES)):
        code = a.code_for(k)
        shown = [s for s in mono(a, k, 0.6) if not s.startswith("→")]
        assert shown == list(code.lines())
        lines = [s for n in code.names() for s in B.code_lines(n)]
        assert all(s in lines for s in shown)
        assert {B.code_file(n) for n in code.names()} == {"examples/jets_as_sets.py"}
    assert "x[:n] = x[rng.permutation(n)]" in [s.strip() for s in mono(a, 4)]
    assert "h = h * mask[..., None]" in [s.strip() for s in mono(a, 3)]
    assert "→ True" in mono(a, 6, 0.9)


def test_no_phase_is_a_training_step():
    a, _ = built()
    assert all(a.station_for(k) is None for k in range(len(a.PHASES)))
    assert "you are here" not in texts(a, 2, 0.5)


# ---------------------------------------------------------------------- what moves
def test_the_rows_move_as_the_shuffle_says():
    a, s = built()
    f = s.metadata["followed"]
    n = f["n"]
    a._t = at(a, 3, 0.9)
    np.testing.assert_array_equal(a.slots(), np.arange(n))       # sorted
    a._t = at(a, 5, 0.5)
    slots = a.slots()                                             # shuffled: row perm[s] at s
    for s_, j in enumerate(f["perm"]):
        assert slots[j] == s_
    a._t = at(a, 6, 0.9)
    np.testing.assert_allclose(a.slots(), np.arange(n))          # sorted again
    seen = []
    for t in np.linspace(a.spans[4][0], a.spans[4][1], 50):
        a._t = float(t)
        seen.append(a.slots().copy())
    assert np.isclose(seen[0], np.arange(n)).all() and np.isclose(seen[-1], slots).all()


def test_the_numbers_on_screen():
    a, s = built()
    m = s.metadata
    pt, rt = m["p_text"], m["right_text"]
    one = texts(a, 0, 0.9)
    assert {"one prong · toy QCD", "two prongs · toy W", "11 particles", "Δη", "Δφ"} <= set(one)
    two = texts(a, 1, 0.95)
    assert {"z", "mask", "11 real, 9 pads"} <= set(two)
    assert "11 real rows + 9 pads = 20" in a.evidence_for(1)
    three = texts(a, 2, 0.95)
    assert {"60 in a row", "MLP", "8,129 numbers", "p = " + pt["mlp"],
            rt["mlp"] + " right"} <= set(three)
    four = texts(a, 3, 0.97)
    assert {"φ on each row", "sum", "ρ", "2,273 numbers", "Deep Sets", "p = " + pt["ds"],
            rt["ds"] + " right"} <= set(four)
    before, after = texts(a, 4, a.SWAP_U - 0.05), texts(a, 4, 0.9)
    assert "p = " + pt["mlp"] in before and "p = " + pt["mlp_shuffled"] in after
    assert {"was " + pt["mlp"], "was " + pt["ds"], "p = " + pt["ds"], "the same"} <= set(after)
    assert pt["mlp"] == "0.72" and pt["mlp_shuffled"] == "0.01" and pt["ds"] == "0.86"
    six = texts(a, 5, 0.9)
    assert {rt["mlp"], rt["mlp_shuffled"], rt["ds"], "flipped: MLP %s, Deep Sets 0"
            % m["changed_text"]} <= set(six)
    assert six.count(rt["ds"]) == 2                                # sorted and shuffled
    seven = texts(a, 6, 0.95)
    assert {"sorted again", "1  sort first: " + rt["mlp_again"], "2  build the symmetry in",
            "GNNs: messages between particles",
            "next: Newton's step and gradient descent"} <= set(seven)
    assert m["changed_text"] in a.subtitle_for(4) and m["changed_text"] in a.subtitle_for(5)
    assert "Graph networks" in a.subtitle_for(6) and "padding" in a.subtitle_for(1)
    assert m["ds_bound_text"] in a.subtitle_for(5) and "most move little" in a.subtitle_for(5)


@pytest.mark.parametrize("phase", [1, 2, 3])
def test_one_thing_moves_at_a_time(phase):
    """The ring that links a row to its particle (phase 2) or to its place in the MLP's
    column (phase 3), and the φ box (phase 4), each sweep down the rows in order."""
    a, _ = built()
    lo, hi = a.spans[phase]
    if phase == 1:
        rings = []
        for t in np.linspace(lo, hi, 200):
            a._t = float(t)
            r = a.n * a.ease_in(1, *a.ROWS_U)
            rings.append(int(r) if 0 < r < a.n else None)
        got = [r for r in rings if r is not None]
        assert got == sorted(got) and set(got) == set(range(a.n))
    elif phase == 2:
        got = []
        for t in np.linspace(lo, hi, 300):
            a._t = float(t)
            r = a.strip_ring()
            if r is not None:
                got.append(r)
        assert got == sorted(got) and set(got) == set(range(a.n))
    else:
        assert a.PHI_U[1] < a.MASK_U < a.SWAP_U + 1             # phi first, then the mask

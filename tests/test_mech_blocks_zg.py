"""Animations 273-274 (voxanim.mech_blocks_zg): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, and the numbers and code lines on screen are the study's and the script's. No dataset is
needed: the events are made up by the script."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_zg as M
from voxanim.next_catalog import ITEMS, build_study

NAMES = ("classifier-learns-the-likelihood-ratio", "reweighting-with-a-classifier")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_zg", name))
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
    from voxanim.mech_stats_tests import LookElsewhere

    titles = {i["name"]: i["title"] for i in ITEMS}
    classes = [M.ANIMATIONS[n] for n in NAMES]
    for name, cls in zip(NAMES, classes, strict=True):
        assert titles[name] == cls.TITLE
    assert M.LearnsRatio.LAST == "The look-elsewhere effect" == LookElsewhere.TITLE
    for before, after in pairwise(classes):
        assert before.NEXT == after.TITLE and after.LAST == before.TITLE
    assert M.Reweights.NEXT == "A bump hunt, from fit to global significance"
    assert M.Reweights.NEXT in titles.values()
    assert {i["section"] for i in ITEMS if i["name"] in NAMES} == {
        "o. The classifier and the likelihood ratio (2)"}


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
        assert len(sc.quads) >= 20                         # the histograms stay
        picture = [x.s for x in sc.texts if x.opacity > 0.5 and x.x < B.PANEL[0]
                   and x.y > B.HEAD_Y + 0.03]
        if name == NAMES[0]:
            assert {"H0", "H1"} <= set(picture), k
        else:
            assert {"x", "y"} <= set(picture), k


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
        assert tag.startswith("measured on made-up data: ")


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
        assert {B.code_file(n) for n in code.names()} == {"examples/likelihood_ratio.py"}


@pytest.mark.parametrize("name", NAMES)
def test_only_training_is_on_the_loop(name):
    a, _ = built(name)
    stations = [a.station_for(k) for k in range(len(a.PHASES))]
    assert stations[1] == "loss" and stations.count(None) == len(stations) - 1
    assert "you are here" in texts(a, 1, 0.5)
    assert "you are here" not in texts(a, 2, 0.5)


# ---------------------------------------------------------------------- 273
def test_learns_ratio_numbers_on_screen():
    a, s = built(NAMES[0])
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"H0", "H1", "made-up: 20,000 events of each", "H0: wide, centred at 0",
            "H1: narrow, centred at 1"} <= set(one)
    assert "x1 = rng.normal(1.0, 0.5, (n, 1))" in mono(a, 0)
    two = texts(a, 1, 0.95)
    assert {"loss 0.4691", "exact ratio's: 0.4692", "10,000 steps"} <= set(two)
    assert "        loss = bce(net(X[i]), Y[i])" in mono(a, 1)
    three = texts(a, 2, 0.95)
    assert {"exact", "network: s / (1 − s)", "within 10%", "10⁻⁴", "1", "10"} <= set(three)
    assert "learned = s / (1 - s)" in mono(a, 2)
    four = texts(a, 3, 0.9)
    assert {"past x = 3: 20 H0, 2 H1 events", "at x = 3:", "network: 0.24", "exact: 0.060",
            "no H1"} <= set(four)
    five = texts(a, 4, 0.95)
    assert {"best, a window: 19.3%", "network: 19.6%", "one cut: 9.9%", "at 5% of H0",
            "H1 kept against H0 kept"} <= set(five)
    assert "→ power: 0.196" in mono(a, 4, 0.95)
    assert "next: Reweighting simulation with a classifier" in five
    for k, needle in ((1, m["floor_text"]), (2, "−0.75 to 2.25"), (3, "0.060"),
                      (4, m["np_text"]), (4, m["cut_text"]), (4, "±0.3")):
        assert needle in a.subtitle_for(k), (k, needle)


def test_the_curves_are_the_studys():
    a, s = built(NAMES[0])
    m = s.metadata
    np.testing.assert_allclose(a.learned, m["learned"])
    np.testing.assert_allclose(a.exact, m["exact"])
    ch = a.ratio
    assert ch.ylog and ch.y_range == (1e-4, 10.0)
    # the drift is drawn where it is: past x = 3, the network's curve above the exact one
    x = m["far_x"]
    i = int(np.argmin(np.abs(a.grid - x)))
    assert ch.fy(a.learned[i]) < ch.fy(a.exact[i]) - 0.02


# ---------------------------------------------------------------------- 274
def test_reweights_numbers_on_screen():
    a, s = built(NAMES[1])
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"x", "y", "data", "simulation", "made-up:", "50,000 events each"} <= set(one)
    assert "data = rng.multivariate_normal(" in mono(a, 0)
    two = texts(a, 1, 0.95)
    assert {"w = s / (1 − s)", "orange: w above 1", "blue: w below 1"} <= set(two)
    assert "train(judge, X, Y, 10000)" in mono(a, 1)
    three = texts(a, 2, 0.95)
    assert {"weighted", "off by at most", "x: 12%", "y: 14%", "before: 110%, 49%"} <= set(three)
    assert "w = (s / (1 - s)).numpy()[:, 0]" in mono(a, 2)
    four = texts(a, 3, 0.95)
    assert {"x + y", "not an input", "before: 80%", "weighted: 12%"} <= set(four)
    five = texts(a, 4, 0.95)
    assert {"38 data events,", "no simulation", "largest: 38", "exact: 151", "network's w",
            "exact w"} <= set(five)
    assert "→ big: %.1f" % m["w_max"] in mono(a, 4, 0.95)
    six = texts(a, 5, 0.95)
    assert {"50,000 events", "count like 30,400", "1,000 of each:", "typically off by 24%",
            "50,000: 5%"} <= set(six)
    assert "→ n_eff: 30,400" in mono(a, 5, 0.95)
    assert "next: A bump hunt, from fit to global significance" in six
    assert "151" in a.subtitle_for(4) and "30,400" in a.subtitle_for(5)


def test_the_weights_reshape_the_simulation():
    a, s = built(NAMES[1])
    a._t = at(a, 2, 0.0)
    assert a.weighted_share() == 0.0
    a._t = at(a, 2, 0.9)
    assert a.weighted_share() == 1.0
    # the plane is the study's weights: orange where w > 1, blue where w < 1
    w = np.asarray(s.metadata["plane_w"])[::-1]
    tex = a.plane_tex
    hot = w > 2
    cold = w < 0.5
    assert hot.any() and cold.any()
    assert (tex[hot][:, 0] > tex[hot][:, 2]).all() and (tex[cold][:, 2] > tex[cold][:, 0]).all()

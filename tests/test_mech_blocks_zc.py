"""Animations 265-268 (voxanim.mech_blocks_zc): the layout is clean, the loop closes, nothing
blanks between phases, every phase says what kind of evidence it shows, the headings are
short, the blocks move without their tiles changing, and the numbers and code lines on
screen are the study's and the script's."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import layout, presets, qa
from voxanim import mech_blocks_zc as M
from voxanim.next_catalog import build_study

NAMES = ("ae-to-gan-rewire", "ae-gan-blurry-or-sharp", "ae-gan-rebuild-or-invent",
         "ae-gan-one-loss-or-two")
_built = {}


def built(name):
    """``(animation, study)``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("study_blocks_zc", name))
    return _built[name]


def texts(a, phase, u=0.6, kind=None):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and (kind is None or x.kind == kind)]


def mono(a, phase, u=0.6):
    return texts(a, phase, u, "mono")


# ---------------------------------------------------------------------- no data needed
def test_swapped_texts_are_never_both_shown():
    for p in np.linspace(0.0, 1.0, 41):
        out, into = M.swap(p)
        assert min(out, into) == 0.0 and 0.0 <= out <= 1.0 and 0.0 <= into <= 1.0
    assert M.swap(0.0) == (1.0, 0.0) and M.swap(1.0) == (0.0, 1.0)


def test_the_titles_and_the_chain_are_the_catalogues():
    from voxanim.next_catalog import ITEMS
    titles = {i["name"]: i["title"] for i in ITEMS}
    classes = [M.ANIMATIONS[n] for n in NAMES]
    for name, cls in zip(NAMES, classes, strict=True):
        assert titles[name] == cls.TITLE
    # the last of "Autoencoders and latent codes", as the website names it
    from pathlib import Path
    site = (Path(B.__file__).resolve().parent.parent / "website" / "build_site.py").read_text()
    assert '("idec-dense", "%s")' % M.Rewire.LAST in site
    for before, after in pairwise(classes):
        assert before.NEXT == after.TITLE and after.LAST == before.TITLE
    assert M.Losses.NEXT == "Four families, four objectives"
    assert {i["section"] for i in ITEMS if i["name"] in NAMES} == {
        "m. From autoencoder to GAN (4)"}


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
        if a.evidence_for(k) == a.evidence_for(k + 1):
            assert a.evidence_for(k) in shown
        if a.code_for(k).lines() == a.code_for(k + 1).lines():
            assert any(x.kind == "mono" and x.opacity > 0.5 for x in sc.texts)
        assert len(sc.quads) >= 6                          # the panel, the five stations
        assert sum(x.opacity > 0.5 and bool(x.s.strip()) for x in sc.texts) >= 8
        # the picture itself is never empty: something of it stays at full strength
        assert any(x.opacity > 0.5 and x.x < B.PANEL[0] and x.y > B.HEAD_Y + 0.03
                   for x in sc.texts), (name, k)


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
        tag = a.evidence_for(k)
        assert layout.text_width(tag, B.SIZE) <= 0.95 - B.PANEL[0]
        assert tag.startswith(("measured on MNIST: ", "exact arithmetic: "))


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
        assert {B.code_file(n) for n in code.names()} == {"examples/ae_gan.py"}


@needs_mnist
@pytest.mark.parametrize("name", NAMES)
def test_only_training_phases_light_a_station(name):
    a, _ = built(name)
    lit = [a.station_for(k) for k in range(len(a.PHASES))]
    assert lit == list(type(a).STATION)
    assert set(lit) <= {None, "forward", "loss"}


# ---------------------------------------------------------------------- 265
@needs_mnist
def test_the_blocks_move_and_their_tiles_do_not_change():
    a, s = built("ae-to-gan-rewire")
    m = s.metadata
    drawn = []

    def spy(sc, kind, fx, fy, op, patches, **kw):
        drawn.append((kind, round(fx, 4), round(fy, 4), patches, kw.get("ghost", False)))
    a.block = spy
    try:
        for k, u in ((0, 0.5), (2, 0.99), (3, 0.99), (4, 0.5)):
            drawn.clear()
            lo, hi = a.spans[k]
            a.scene(lo + (hi - lo) * u)
            where = {kind: (fx, fy) for kind, fx, fy, _p, ghost in drawn if not ghost}
            # every block, wherever it is, is drawn from the same tiles: the study's
            assert all(p is a.meta["patches"] for *_r, p, _g in drawn)
            assert all(np.array_equal(a.meta["patches"][key], m["patches"][key])
                       for key in m["patches"])
            if k == 0:
                assert where == {"enc": (a.AX["enc"], a.ROW_A), "dec": (a.AX["dec"], a.ROW_A)}
            if k >= 2:
                assert where["dec"] == (a.GX["gen"], a.ROW_G)         # the generator's place
            if k >= 3:
                assert where["enc"] == (a.GX["judge"], a.ROW_G)       # the judge's place
    finally:
        del a.block
    # the decoder moves first, then the encoder: one thing at a time
    assert a.DEC_LEFT[0] < a.ENC_RIGHT[0]


@needs_mnist
def test_rewire():
    a, s = built("ae-to-gan-rewire")
    m = s.metadata
    one = texts(a, 0, 0.9)
    assert {"encoder", "decoder", "205,072", "205,840", "numbers"} <= set(one)
    assert "encoder = nn.Sequential(" in mono(a, 0)
    two = texts(a, 1, 0.95)
    assert {"16 numbers", "pixel by pixel: 784 scores", "loss 0.07"} <= set(two)
    assert "code = encoder(x)" in mono(a, 1)
    assert a.station_for(1) == "loss"
    three = texts(a, 2, 0.95)
    assert {"generator", "made up", "generator = decoder", "205,840 numbers kept"} <= set(three)
    assert "decoder" not in three                       # renamed once in its new place
    assert "the wires come off" in texts(a, 2, 0.15)
    four_early, four = texts(a, 3, 0.25), texts(a, 3, 0.95)
    assert "left out: 4,112" in four_early
    assert {"judge", "new: 257", "judge = encoder's body + new head",
            "200,960 kept + 257 new"} <= set(four)
    assert "encoder" not in four
    assert "4,112 left out; 256 × 1 + 1 = 257 new" in a.evidence_for(3)
    five = texts(a, 4, 0.95)
    assert {"real", "real?", "no digit goes in:", "it starts from a made-up code"} <= set(five)
    assert "said_real = judge(x)" in mono(a, 4)
    six = texts(a, 5, 0.5)
    assert {"encoder", "decoder", "16 numbers", "784 scores", "1 score", "generator",
            "judge", "real?"} <= set(six)
    assert "In the GAN no digit goes through a code" in a.subtitle_for(5)
    assert m["counts_text"]["head"] in a.subtitle_for(3)


# ---------------------------------------------------------------------- 266
@needs_mnist
def test_blurry():
    a, s = built("ae-gan-blurry-or-sharp")
    m = s.metadata
    assert {"real", "rebuilt"} <= set(texts(a, 0, 0.9))
    assert {"before", "after", "GAN game"} <= set(texts(a, 1, 0.95))
    three = texts(a, 2, 0.95)
    assert {"6.0%", "9.4%", "9.1%", "7.4%", "grey pixels"} <= set(three)
    four = texts(a, 3, 0.95)
    assert {"0.94", "0.74", "0.84", "classifier's certainty"} <= set(four)
    assert not {"6.0%", "9.4%"} & set(four)              # swapped in place, not stacked
    five = texts(a, 4, 0.9)
    assert {"real", "rebuilt", "6.0%", "9.4%", "one pixel:", "ink in half the digits",
            "1.20", "0.69", "lowest: 0.5, grey"} <= set(five)
    assert not {"before", "after", "7.4%"} & set(five)
    assert "loss = F.binary_cross_entropy(" in mono(a, 4)
    assert a.evidence_for(4).startswith("exact arithmetic: ")
    assert m["grey_text"]["before"] in a.subtitle_for(2)


# ---------------------------------------------------------------------- 267
@needs_mnist
def test_rebuild():
    a, s = built("ae-gan-rebuild-or-invent")
    m = s.metadata
    one = texts(a, 0, 0.95)
    assert {"reads 7", "reads 2", "reads 1", "reads 0", "given", "rebuilt"} <= set(one)
    two = texts(a, 1, 0.9)
    assert {"928 of 1,000", "949 of 1,000", "rebuilds read right", "real digits"} <= set(two)
    three = texts(a, 2, 0.95)
    assert {"generator", "made up", "made", "a 7, please?", "×", "no input for a digit",
            "only 16 made-up numbers"} <= set(three)
    assert "fake = generator(z)" in mono(a, 2)
    four = texts(a, 3, 0.95)
    for d, n in enumerate(m["made_counts"]["after"]):
        assert {str(d), str(n)} <= set(four)
    assert {"63", "135"} <= set(four)
    five = texts(a, 4, 0.9)
    assert [t for t in five if t in "0123456789" and len(t) == 1].count("8") >= \
        m["made_read"].count(8)
    assert "none chosen" in a.subtitle_for(4)


# ---------------------------------------------------------------------- 268
@needs_mnist
def test_losses():
    a, _s = built("ae-gan-one-loss-or-two")
    one = texts(a, 0, 0.9)
    assert {"autoencoder", "0.10", "0.69", "3,000 steps"} <= set(one)
    assert "loss = F.binary_cross_entropy(" in mono(a, 0)
    two = texts(a, 1, 0.9)
    assert {"judge 1.20", "generator 0.88", "GAN, means of 100 steps", "6,000 steps"} <= set(two)
    assert "lg = bce(judge(fake), ones)" in mono(a, 1)
    three = texts(a, 2, 0.9)
    assert {"judge 1.43 → 1.20", "generator 0.67 → 0.88", "cannot tell", "1.39"} <= set(three)
    assert "autoencoder" not in three
    four = texts(a, 3, 0.95)
    assert {"step 0", "step 500", "step 2,000", "step 6,000", "judge 1.20"} <= set(four)
    assert "next: Four families, four objectives" in texts(a, 4, 0.95)
    assert "1.43 to 1.20" in a.subtitle_for(1) and "0.67" in a.subtitle_for(1)

"""Animation 182 (voxanim.mech_bridge_bits): the layout is clean, the loop closes, every
phase says what kind of evidence it shows, it opens on the animation it comes from and
closes on the one it leads to, and the numbers on screen are the study's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import layout, presets, qa
from voxanim import mech_bridge_bits as M
from voxanim import study_bridge_bits as B
from voxanim.mech_phys_apps import pct
from voxanim.next_catalog import build_study

NAME = "bits-in-a-weight"
_built = {}


def built():
    """``(animation, study)``, built once."""
    if NAME not in _built:
        _built[NAME] = (presets.get(NAME).build(), build_study("study_bridge_bits", NAME))
    return _built[NAME]


def subtitles(a):
    return [a.subtitle_for(k) for k in range(len(a.PHASES))]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.5]


# ---------------------------------------------------------------------- no data needed
def test_labels():
    assert M.float_bits(1.0) == ("0", "01111111", "0" * 23)
    assert M.float_bits(-0.25) == ("1", "01111101", "0" * 23)
    _s, e, f = M.float_bits(0.1)
    assert (1 + int(f, 2) / 2 ** 23) * 2.0 ** (int(e, 2) - 127) == float(np.float32(0.1))
    assert M.minus("-0.25") == "−0.25"
    for tag in (M.WEIGHTS, M.ROUNDED, M.HELD, M.EXACT):
        assert M.is_evidence(tag)
    assert not M.is_evidence("measured elsewhere")


# ---------------------------------------------------------------------- the animation
@needs_mnist
def test_layout_is_clean():
    assert qa.check(NAME, samples=24, verbose=False) == 0


@needs_mnist
def test_drawn_by_this_module_and_loop_closes():
    a, _ = built()
    assert type(a) is M.ANIMATIONS[NAME] is M.BitsInAWeight
    assert 45.0 <= a.t_done <= 61.0 and a.duration <= 66.0 and len(a.PHASES) >= 4

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:                            # every phase draws things, not only text
        assert len(a.scene((lo + hi) / 2).quads) >= 3


@needs_mnist
def test_every_frame_of_a_coarse_pass_builds():
    a, _ = built()
    for t in np.arange(0.0, a.duration, 0.37):
        assert a.scene(float(t)).texts is not None


@needs_mnist
def test_subtitles_fit_and_every_phase_says_what_it_shows():
    a, _ = built()
    for sub in subtitles(a):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert 1 <= len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
    for k in range(len(a.PHASES)):                    # the kind of evidence, in every phase
        for u in (0.2, 0.6, 0.9):
            assert any(M.is_evidence(t) for t in texts(a, k, u))
        assert sum(bool(t.strip()) for t in texts(a, k, 0.6)) >= 3


@needs_mnist
def test_it_is_a_bridge():
    a, s = built()
    assert "from: The entropy of a network's weights" in texts(a, 0, 0.5)
    assert "next: Quantization" in texts(a, len(a.PHASES) - 1, 0.95)
    assert s.metadata["comes_from"] == "The entropy of a network's weights"
    assert s.metadata["leads_to"] == "Quantization"


# ---------------------------------------------------------------------- numbers on screen
@needs_mnist
def test_numbers_on_screen_are_the_study():
    a, s = built()
    m = s.metadata
    f = m["formats"]
    first = texts(a, 0, 0.95)
    assert "52,544 weights of a 784-64-32-10 classifier" in first
    assert "174: all its parameters, bins of 0.01: %.2f bits each" % m["histogram_174"] in first
    assert "52,544 × 32 = 1,681,408 bits in all" in first
    w = a.block[0, 0]
    assert "one weight, W0[384, 0] = %s" % M.minus("%.8f" % w) in first
    sign, expo, frac = M.float_bits(w)
    assert {sign, expo, frac, "sign 1", "exponent 8", "fraction 23"} <= set(first)
    carried = texts(a, 1, 0.95)
    assert "≥ %.2f of 32 bits, %s values" % (f["FP32"]["entropy"],
                                             format(f["FP32"]["distinct"], ",")) in carried
    for name in ("FP16", "BF16", "FP8", "FP4", "FP8 scaled", "FP4 scaled"):
        assert "%.2f of %g bits, %s values" % (f[name]["entropy"], f[name]["stored"],
                                               format(f[name]["distinct"], ",")) in carried
    assert any("log₂ 52,544 = %.2f bits" % m["log2_weights"] in t for t in carried)
    acc = texts(a, 2, 0.95)
    for name in B.FORMATS:
        assert pct(f[name]["accuracy"], 1) in acc
    assert "%s of weights fall in it" % pct(f["FP4"]["zeros"], 1) in acc   # the band is FP4's 0
    assert "FP4 keeps %d values: %.2f bits" % (f["FP4"]["distinct"], f["FP4"]["entropy"]) in acc
    scale = texts(a, 3, 0.95)
    assert {"%d of 32 at 0" % m["block_zero_fp4"], "%d of 32 at 0" % m["block_zero_scaled"],
            "FP4, scale %g" % m["block_scale"], pct(f["FP4 scaled"]["zeros"], 1),
            "%.2f" % f["FP4 scaled"]["entropy"], pct(f["FP4 scaled"]["accuracy"], 1),
            "= 0.25 bits a weight"} <= set(scale)
    few = texts(a, 4, 0.95)
    b = m["fewest_int_bits"]
    assert "holds to %d-bit integers: %.2f bits," % (b, m["fewest_entropy"]) in few
    lo = m["ints"][str(m["below_int_bits"])]
    assert "%d bits: %.2f carried, %s" % (m["below_int_bits"], lo["entropy"],
                                          pct(lo["accuracy"], 1)) in few
    assert "float32: %s" % pct(m["float32"]["accuracy"], 1) in few


@needs_mnist
def test_what_is_drawn_matches_the_study():
    a, s = built()
    m = s.metadata
    # the band drawn is exactly the weights FP4 rounds to 0
    assert a.inside == pytest.approx(m["formats"]["FP4"]["zeros"])
    # the block: as trained, then FP4 and FP4 with its scale, as study_hardware rounds them
    blk = a.block
    np.testing.assert_array_equal(blk[1], B.H.to_format("FP4", blk[0]))
    np.testing.assert_array_equal(blk[2], a.codes * a.scale)
    assert int((blk[1] == 0).sum()) == m["block_zero_fp4"]
    # the bit pattern drawn is the weight
    assert (-1) ** int(M.float_bits(a.shown)[0]) * a.mant * 2.0 ** a.e == \
        float(np.float32(a.shown))
    # the histogram is the weights'
    assert a.counts.sum() == int(((a.w >= a.edges[0]) & (a.w <= a.edges[-1])).sum())

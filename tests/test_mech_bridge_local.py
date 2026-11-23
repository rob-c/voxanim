"""Animation 179 (voxanim.mech_bridge_local): the layout is clean, the loop closes, every
phase says what kind of evidence it shows, it names where it comes from and where it
leads, and the numbers on screen are the study's."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import layout, presets, qa
from voxanim import mech_bridge_local as M
from voxanim import study_bridge_local as B
from voxanim.next_catalog import build_study, drawing

NAME = "dense-layer-learns-local"
_built = {}


def built():
    """``(animation, study)``, built once."""
    if not _built:
        _built["a"] = (presets.get(NAME).build(), build_study("study_bridge_local", NAME))
    return _built["a"]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.5]


# ---------------------------------------------------------------------- no data needed
def test_labels():
    assert M.times(784, 64, 50176) == "784 × 64 = 50,176"
    assert M.number_or_zero(0.0) == "0" and M.number_or_zero(2.5e-15) == "2.5e−15"
    for tag in (M.LAYER, M.PAIRS, M.COUNTED, M.SHIFTED, M.BOTH):
        assert M.is_evidence(tag)
    assert not M.is_evidence("measured elsewhere")


# ---------------------------------------------------------------------- MNIST
@needs_mnist
def test_layout_is_clean():
    assert qa.check(NAME, samples=24, verbose=False) == 0


@needs_mnist
def test_drawn_by_this_module_and_loop_closes():
    a, _ = built()
    assert type(a) is M.ANIMATIONS[NAME] is drawing(NAME)
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
    for k in range(len(a.PHASES)):
        sub = a.subtitle_for(k)
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert 1 <= len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE
        for u in (0.2, 0.6, 0.9):
            assert any(M.is_evidence(t) for t in texts(a, k, u))
        assert sum(bool(t.strip()) for t in texts(a, k, 0.6)) >= 3


@needs_mnist
def test_it_is_a_bridge():
    a, _ = built()
    assert "from: The entropy of a network's weights" in texts(a, 0, 0.5)
    assert "The entropy of a network's weights" in a.subtitle_for(0)
    assert "next: Convolution, 3×3 kernel" in texts(a, len(a.PHASES) - 1, 0.9)
    assert a.subtitle_for(len(a.PHASES) - 1).endswith("Next: Convolution, 3×3 kernel")


@needs_mnist
def test_numbers_on_screen_are_the_study():
    a, s = built()
    m = s.metadata
    h = m["horizontal_neighbours"]
    recall = texts(a, 0, 0.95)
    assert "correlation %s → %s" % (M.fixed(h["before"]), M.fixed(h["after"])) in recall
    assert "correlation 0.00 → 0.50" in recall                        # as 174 shows it
    assert "the first layer of the 784-64-32-10 classifier" in recall
    dist = texts(a, 1, 0.95)
    after = m["correlation"]["after"]
    assert "d = 12: correlation %s" % M.fixed(after[11]) in dist
    assert "half the one-pixel value by %.1f pixels" % m["half_distance"] in dist
    assert any(t.startswith("before training: ") and t.endswith(" at every distance")
               for t in dist)
    early = texts(a, 1, 0.31)
    assert "d = 1: correlation %s" % M.fixed(after[0]) in early
    counted = texts(a, 2, 0.95)
    assert {"dense, 64 units: 784 × 64 = 50,176", "3 × 3 conv, 64 maps: 9 × 64 = 576",
            "dense, as many outputs: 784 × 43,264 = 33,918,976",
            "local, not shared: 9 × 43,264 = 389,376",
            "87 times fewer weights, 676 times the outputs"} <= set(counted)
    shifted = texts(a, 3, 0.95)
    acc = m["accuracy"]
    assert "moved 4 px: dense %.0f%%, CNN %.0f%%" % (100 * acc["dense"][-1],
                                                    100 * acc["cnn"][-1]) in shifted
    assert "in place: dense %.1f%%, CNN %.1f%%" % (100 * acc["dense"][0],
                                                  100 * acc["cnn"][0]) in shifted
    assert {"dense 784-64-32-10: 52,650 parameters", "CNN, 3 × 3, 8 then 16 maps: 5,258",
            "2,340 updates of 128, Adam"} <= set(shifted)
    assert "difference 0 (values up to %.1f)" % m["equivariance"]["largest_value"] in shifted
    verdict = texts(a, 4, 0.95)
    assert "576 weights for 64 maps, not 50,176" in verdict
    assert "the CNN's 8 first-layer 3 × 3 filters, trained" in verdict


@needs_mnist
def test_pictures_are_the_study_parameters():
    a, s = built()
    np.testing.assert_array_equal(a.pics, s.panels[0].data["values"])
    np.testing.assert_allclose(a.filters[0], B.cnn_params()["W1"][:, 0].reshape(3, 3))
    np.testing.assert_array_equal(a.moved[0], a.digit_img)
    np.testing.assert_array_equal(a.moved[2].reshape(28, 28)[:, 2:],
                                  a.digit_img.reshape(28, 28)[:, :-2])


@pytest.mark.parametrize("u", [0.0, 0.5, 1.0])
def test_the_sweep_stays_on_the_distances(u):
    class Stub:
        d = B.DISTANCES.astype(float)
    k = M.DenseLayerLearnsLocal._sweep(Stub, u)
    assert 1 <= k <= len(B.DISTANCES)

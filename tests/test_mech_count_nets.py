"""Animation 183 (voxanim.mech_count_nets): the layout is clean, the loop closes, every
phase says what kind of evidence it shows, the numbers on screen are the study's, and the
PyTorch lines appear beside the layers they make."""

from __future__ import annotations

import numpy as np
from conftest import needs_mnist

from voxanim import layout, presets, qa
from voxanim import mech_count_nets as M
from voxanim.next_catalog import build_study

NAME = "counting-weights-dnn-and-cnn"
_built = {}


def built():
    """``(animation, study)``, built once."""
    if NAME not in _built:
        _built[NAME] = (presets.get(NAME).build(), build_study("study_count_nets", NAME))
    return _built[NAME]


def texts(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts if x.opacity > 0.5]


def mono(a, phase, u=0.6):
    lo, hi = a.spans[phase]
    return [x.s for x in a.scene(lo + (hi - lo) * u).texts
            if x.opacity > 0.5 and x.kind == "mono"]


def joined(a, phase, u):
    """Every visible text of the frame in one string (a sum is drawn in coloured pieces)."""
    return " ".join(texts(a, phase, u))


# ---------------------------------------------------------------------- no data needed
def test_labels():
    assert M.is_evidence(M.DNN) and M.is_evidence(M.CNN) and M.is_evidence(M.BOTH)
    assert not M.is_evidence("measured on MNIST")
    assert M.product(784, 64) == "784 × 64" and M.product(4, 13, 13) == "4 × 13 × 13"
    assert M.cells(784) > M.cells(64) > M.cells(32) > M.cells(10) >= 3
    assert M.cells(10 ** 6) == 15


# ---------------------------------------------------------------------- the page's models
@needs_mnist
def test_layout_is_clean():
    assert qa.check(NAME, samples=24, verbose=False) == 0


@needs_mnist
def test_drawn_by_this_module_and_loop_closes():
    a, _ = built()
    assert type(a) is M.ANIMATIONS[NAME]
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
def test_one_weight_then_a_small_layer():
    a, s = built()
    m = s.metadata
    first = texts(a, 0, 0.35)
    assert {"1 weight", "1 bias", "input", "output"} <= set(first)
    assert "nn.Linear(1, 1)" in mono(a, 0, 0.35)
    assert "1 × 1 + 1 = 2" in joined(a, 0, 0.35)
    small = texts(a, 0, 0.95)
    assert M.RULE in small
    assert {"%d weights" % m["toy"]["weights"], "%d biases" % m["toy"]["biases"],
            "3 inputs", "2 outputs"} <= set(small)
    assert "nn.Linear(3, 2)" in mono(a, 0, 0.95)
    assert "3 × 2 + 2 = 8" in joined(a, 0, 0.95)


@needs_mnist
def test_the_dnn_layer_by_layer():
    a, s = built()
    m = s.metadata["dnn"]
    lines = [x["code"] for x in m["layers"]]
    assert mono(a, 1, 0.3) == lines                     # the whole model's code, all along
    sums = ("784 × 64 + 64 = 50,240", "64 × 32 + 32 = 2,080", "32 × 10 + 10 = 330")
    for (_lo, hi), want, run, n_in in zip(M.CountingWeights.DNN_SLOTS, sums, m["running"],
                                         (784, 64, 32)):
        u = hi - 0.04
        shown = texts(a, 1, u)
        assert want in joined(a, 1, u)
        assert "%s inputs" % format(n_in, ",") in shown
        assert format(run, ",") in shown                # the running total
    end = texts(a, 1, 0.95)
    assert "52,650" in end and "total so far" in end
    assert {"50,240", "2,080", "330"} <= set(end) and end.count("0") >= 3 + 1
    assert "nothing to learn" in end


@needs_mnist
def test_the_cnn_filters_and_dense_layer():
    a, s = built()
    m = s.metadata["cnn"]
    assert mono(a, 2, 0.3) == [x["code"] for x in m["layers"]]
    filters = texts(a, 2, 0.45)
    assert {"4 filters", "4 biases", "40"} <= set(filters)
    assert "4 × 1 × 3 × 3 + 4 = 40" in joined(a, 2, 0.45)
    sizes = joined(a, 2, 0.95)
    for k in m["any_size"]:
        assert "%d × %d: %d" % (k["side"], k["side"], k["parameters"]) in sizes
    assert "any picture size: the same 40 numbers" in sizes
    zeros = texts(a, 3, 0.1)
    assert "nothing to learn" in zeros and "ReLU, MaxPool2d, Flatten:" in zeros
    dense = texts(a, 3, 0.95)
    assert "676 × 10 + 10 = 6,770" in joined(a, 3, 0.95)
    assert {"676 inputs", "(4 × 13 × 13)", "10 outputs", "10 biases", "6,810"} <= set(dense)


@needs_mnist
def test_both_totals_and_the_line_that_counts_them():
    a, s = built()
    m = s.metadata
    end = texts(a, 4, 0.95)
    assert {"DNN", "CNN", "52,650", "6,810"} <= set(end)
    assert "gives 52,650 and 6,810" in end
    code = mono(a, 4, 0.95)
    assert m["code"] in code
    assert {"nn.Linear(784, 64)", "nn.Linear(4 * 13 * 13, 10)"} <= set(code)
    assert {"50,240", "6,770"} <= set(end)

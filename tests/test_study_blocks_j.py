"""The numbers behind animations 212-214 (voxanim.study_blocks_j): one epoch replayed step by
step, the whole program cut into its parts and windows, and the four broken programs' runs."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import study_blocks_j as J
from voxanim import study_course as course


# ---------------------------------------------------------------------- no data needed
def test_percentages_round_half_up():
    assert J.pct(0.9435) == "94.4%" and J.pct(0.9494) == "94.9%"
    assert J.pct(0.8495) == "85.0%" and J.pct(0.1009) == "10.1%" and J.pct(0.919) == "91.9%"


def test_next_door_weights_alike():
    rng = np.random.default_rng(0)
    assert abs(J.alike(rng.uniform(-1, 1, (8, 784)))) < 0.05          # noise
    r, c = np.mgrid[0:28, 0:28]
    smooth = np.stack([np.sin(r / 5.0 + k) + np.cos(c / 4.0) for k in range(3)])
    assert J.alike(smooth.reshape(3, 784)) > 0.9


def test_the_softmax_floor():
    """With a softmax in front, the loss reads probabilities: its best case is 1 for the
    right digit and 0 elsewhere, and the loss of that is ln(e + 9) - 1."""
    best = np.zeros((1, 10))
    best[0, 3] = 1.0
    assert np.isclose(course.loss_each(best, [3])[0], J.softmax_floor())
    assert np.isclose(J.softmax_floor(), np.log(np.e + 9) - 1) and f"{J.softmax_floor():.2f}" \
        == "1.46"
    worse = np.full((1, 10), 0.1)
    assert course.loss_each(worse, [3])[0] > J.softmax_floor()


def test_the_dropout_mask():
    mask = J.dropout_mask()
    assert mask.shape == (64,) and mask.dtype == bool
    assert np.array_equal(mask, J.dropout_mask())                      # seeded
    assert 5 <= mask.sum() <= 20                                       # about 1 in 5


def test_the_whole_program_is_its_parts_and_windows():
    lines, parts = J.program_parts()
    assert len(lines) == 63 and lines == B.program()
    assert parts[0][1] == 0 and parts[-1][2] == len(lines) - 1
    for (_n, _a, b), (_m, a2, _b2) in itertools.pairwise(parts):
        assert a2 == b + 1                                             # no gaps
    shown = []
    for k in range(len(J.WINDOWS)):
        window, idx = J.window_lines(k)
        assert 1 <= len(window) <= B.MAX_LINES
        assert max(len(s) for s in window) <= B.MAX_CHARS
        assert idx == list(range(idx[0], idx[0] + len(idx)))          # contiguous
        shown += idx
    assert shown == sorted(shown)                                      # in order, once each
    # left out: the imports (one line is wider than the panel) and the two prints after
    # the loop
    assert sorted(set(range(63)) - set(shown)) == list(range(7)) + [49, 50]
    assert max(len(s) for s in B.code_lines("imports")) > B.MAX_CHARS


def test_the_no_flatten_error_is_torchs():
    """The model without nn.Flatten(), the region's own lines, given a batch shaped like
    MNIST's: torch raises exactly the error the animation shows."""
    torch = pytest.importorskip("torch", reason="torch is not installed")
    ns = dict(torch=torch, nn=torch.nn)
    exec("\n".join(B.code_lines("mistake-no-flatten")), ns)
    with pytest.raises(RuntimeError) as error:
        ns["model"](torch.randn(64, 1, 28, 28))
    assert str(error.value) == J.NO_FLATTEN_ERROR
    assert type(error.value).__name__ == J.NO_FLATTEN_TYPE
    # the same model with Flatten in front runs
    ns["model"].insert(0, torch.nn.Flatten())
    assert tuple(ns["model"](torch.randn(64, 1, 28, 28)).shape) == (64, 10)


# ---------------------------------------------------------------------- the data
@needs_mnist
def test_one_epoch_is_the_recorded_run_replayed():
    r = J.epoch_replay()                       # raises unless it matches the recorded run
    assert tuple(r["steps"]) == J.KEEP and r["units"].shape == (len(J.KEEP), 4, 784)
    for s in (0, 1, 10, 100, 500):
        np.testing.assert_array_equal(r["units"][J.KEEP.index(s)],
                                      course.params(s)["1.weight"][list(J.UNITS)])
    np.testing.assert_array_equal(r["units"][-1], course.params(938)["1.weight"][:4])
    s = J.one_epoch_fast().validate()
    m = s.metadata
    assert m["steps"] == 938 and m["keep"][-1] == 938 and m["keep"][0] == 0
    assert abs(m["alike_start"]) < 0.02 and m["alike_end"] > 0.35       # noise to strokes
    assert m["alike_start"] < m["alike_step_100"] < m["alike_end"]
    assert np.isclose(m["first_loss"], 2.322, atol=1e-3) and m["last_loss"] < 0.2
    assert abs(m["first_loss"] - m["ln10"]) < 0.05
    assert m["accuracy_start"] < 0.15 and m["accuracy_end"] == 0.9494
    assert m["printed"] == "0 %r" % float(course.losses()[-1])
    assert m["code"] == list(B.code_lines("train"))


@needs_mnist
def test_the_whole_programs_numbers():
    m = J.whole_program().validate().metadata
    assert m["lines"] == 63 and m["parameters"] == 52_650 == course.count()
    assert m["correct"] == 9494 and m["printed_accuracy"] == "0.9494"
    assert m["printed_digit"] == "torch.Size([1, 28, 28]) 5"
    assert m["printed_grad"] == "torch.Size([64, 784])"
    assert [m["program"][i] for i in m["device_line"]] == [
        ").to(device)", "        x, y = x.to(device), y.to(device)",
        "        x, y = x.to(device), y.to(device)"]
    assert m["not_shown"] == list(range(7)) + [49, 50]
    assert [np.shape(t) for t in m["model_tiles"]] == [tuple(c) for c in J.CORNERS]


@needs_mnist
def test_the_four_mistakes_measured():
    m = J.common_mistakes().validate().metadata
    assert m["right_text"] == "94.9%"
    ng = m["no_zero_grad"]
    assert ng["text"] == "10.1%" and ng["best"] > 0.5             # it learnt, then broke
    assert ng["used"][0] == ng["fresh"][0]                       # the first step is the same
    assert ng["used"][-1] > 100 * ng["fresh"][-1] > 0
    ne = m["no_eval"]
    assert (ne["text"], ne["with_eval_text"]) == ("91.9%", "94.4%")
    assert ne["off"] == sum(ne["mask"])
    ds = m["double_softmax"]
    assert ds["text"] == "85.0%" and ds["floor_text"] == "1.46"
    assert ds["lowest"] > ds["floor"]                           # never below the floor
    assert ds["right_lowest"] < ds["floor"]                     # the script goes far lower
    nf = m["no_flatten"]
    assert nf["batch"] == [64, 1, 28, 28] and nf["rows"] == 1792 == 64 * 28
    assert "(%dx%d and %dx%d)" % (nf["rows"], nf["got"], nf["wants"], 64) in nf["error"]
    assert nf["rows_sum"] == "64 × 1 × 28 = 1,792"

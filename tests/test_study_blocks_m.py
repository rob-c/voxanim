"""The numbers behind animations 221-223 (voxanim.study_blocks_m): the CNN against the MLP,
what the CNN gets wrong, and three broken CNN programs."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import study_blocks_m as M
from voxanim import study_course as course
from voxanim import study_course_cnn as cnn


# ---------------------------------------------------------------------- no data needed
def test_the_stacks_are_to_scale():
    assert M.stack_tiles(52_650) == (210, 0.6)
    assert M.stack_tiles(5_258) == (21, pytest.approx(0.032))
    whole = [w + f for w, f in (M.stack_tiles(52_650), M.stack_tiles(5_258))]
    assert np.isclose(whole[0] / whole[1], 52_650 / 5_258)


def test_what_the_script_prints():
    assert M.printed_accuracy(9636) == "0.9636" and M.printed_accuracy(9494) == "0.9494"


@pytest.mark.parametrize("name", sorted(cnn.ERRORS))
def test_the_errors_wrap_without_losing_a_word(name):
    error = "RuntimeError: " + cnn.ERRORS[name]
    lines = M.wrap_error(error)
    assert " ".join(lines) == error
    assert all(len(s) <= M.WIDTH for s in lines) and len(lines) <= 4
    for s in lines:                                       # a shape is never cut in two
        assert s.count("[") == s.count("]")
    if "addmm" in error:                                  # the numbers that matter, whole
        assert lines[-1].startswith("(") and lines[-1].endswith(")")
        assert lines[-1] in ("(64x400 and 784x10)", "(16x25 and 400x10)")


def test_the_shapes_walked():
    walk = M.walk((64, 1, 28, 28))
    assert [s for _b, s in walk] == [(64, 1, 28, 28), (64, 8, 26, 26), (64, 8, 26, 26),
                                     (64, 8, 13, 13), (64, 16, 11, 11), (64, 16, 11, 11),
                                     (64, 16, 5, 5), (64, 400)]
    assert [b for b, _s in walk] == ["input"] + list(cnn.BLOCKS[:-1])
    assert M.walk((1, 28, 28))[-1] == ("Flatten", (16, 25))           # no batch
    assert M.walk((1, 1, 28, 28))[-1] == ("Flatten", (1, 400))        # a batch of one


def test_the_shapes_walked_are_torchs():
    torch = pytest.importorskip("torch", reason="torch is not installed")
    nn = torch.nn
    blocks = [nn.Conv2d(1, 8, 3), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(8, 16, 3), nn.ReLU(),
              nn.MaxPool2d(2), nn.Flatten()]
    for dims in ((64, 1, 28, 28), (1, 28, 28), (1, 1, 28, 28)):
        x = torch.randn(*dims)
        got = [("input", tuple(x.shape))]
        with torch.no_grad():
            for b in blocks:
                x = b(x)
                got.append((type(b).__name__, tuple(x.shape)))
        assert got == M.walk(dims)


# ---------------------------------------------------------------------- the data
@needs_mnist
def test_the_cnn_against_the_mlp():
    s = M.against_the_mlp().validate()
    m = s.metadata
    assert (m["cnn_params"], m["mlp_params"]) == (5_258, 52_650) == (cnn.count(), course.count())
    assert m["cnn_counts"] == [80, 1_168, 4_010] and sum(m["mlp_counts"]) == 52_650
    assert m["ratio_text"] == "10" and 10.0 < m["ratio"] < 10.1
    assert (m["cnn_accuracy_text"], m["mlp_accuracy_text"]) == ("96.4%", "94.9%")
    assert (m["cnn_missed"], m["mlp_missed"]) == (364, 506)
    assert (m["printed_cnn"], m["printed_mlp"]) == ("0.9636", "0.9494")
    assert sum(m["per_digit_cnn"]) == 364 and sum(m["per_digit_mlp"]) == 506
    assert sum(m["per_digit_total"]) == course.TEST
    assert m["both_right"] + m["only_cnn"] + m["only_mlp"] + m["both_wrong"] == course.TEST
    assert m["only_mlp"] + m["both_wrong"] == m["cnn_missed"]
    assert m["only_cnn"] + m["both_wrong"] == m["mlp_missed"]
    # the two curves are on the same steps, and end at the accuracies in the table
    assert m["eval_steps"] == [int(v) for v in course.heldout("sgd")["steps"]]
    assert m["accuracy_cnn"][-1] == m["cnn_accuracy"] and m["accuracy_mlp"][-1] == m[
        "mlp_accuracy"]
    # after step 0 (both at chance) the CNN is ahead at every test of the one run each
    assert [s for s in m["eval_steps"] if s > 0] == [s for s in m["ahead_at"] if s > 0]
    # the digits the subtitle names: fewer 5s, more 8s
    c, d = m["per_digit_cnn"], m["per_digit_mlp"]
    assert (c[5], d[5], c[8], d[8]) == (33, 117, 131, 64)
    assert m["runs"] == "one run each" and any("one run" in s for s in m["limitations"])


@needs_mnist
def test_what_the_cnn_gets_wrong():
    m = M.what_it_gets_wrong().validate().metadata
    assert m["missed"] == 364 and sum(m["per_digit"]) == 364
    assert (m["worst"], m["worst_count"]) == (8, 131)
    assert [(g["true"], g["said"], g["count"]) for g in m["groups"]] == [(8, 9, 36), (2, 7, 33),
                                                                          (8, 0, 22)]
    assert m["top_total"] == 91 and m["printed"] == "0.9636"
    _x, _y, xt, yt = course.data()
    p = cnn.predictions()
    for g in m["groups"]:
        assert len(g["index"]) == M.SHOWN
        # the first of their kind in test-set order: none skipped
        kind = np.flatnonzero((yt == g["true"]) & (p == g["said"]))
        assert g["index"] == [int(i) for i in kind[:M.SHOWN]]
        np.testing.assert_array_equal(np.asarray(g["image"]),
                                      xt[g["index"]].reshape(-1, 28, 28))
        assert all(a > b for a, b in zip(g["sure"], g["sure_true"]))
    # the one example said to read the CNN's way is among those drawn, and only it
    looked = [i for g in m["groups"] for i, v in zip(g["index"], g["looks_like"]) if v]
    assert looked == [947] and m["groups"][0]["looks_like"][3] == 9
    f = m["first"]
    miss = np.flatnonzero(p != yt)[:M.SHOWN]
    assert f["index"] == [int(i) for i in miss]
    assert list(zip(f["true"], f["said"])) == [(3, 2), (4, 9), (8, 3), (8, 0)]


@needs_mnist
def test_three_cnn_mistakes():
    m = M.common_mistakes().validate().metadata
    for name, error in cnn.ERRORS.items():
        assert m["errors"][name] == "RuntimeError: " + error
        assert " ".join(m["wrapped"][name]) == m["errors"][name]
    f = m["in_features"]
    assert (f["arrive"], f["wants"]) == (400, 784) and f["sum"] == "16 × 5 × 5 = 400"
    assert f["flat_text"] == "[64, 400]"
    c = m["channels"]
    assert (c["wants"], c["got"]) == (3, 1) and c["input_text"] == "[64, 1, 28, 28]"
    nb = m["no_batch"]
    assert nb["one_text"] == "[1, 28, 28]" and (nb["rows"], nb["row"]) == (16, 25)
    assert "(%dx%d and %dx10)" % (nb["rows"], nb["row"], nb["wants"]) in cnn.ERRORS[
        "cnn-mistake-no-batch"]
    fix = m["fix"]
    assert fix["one_text"] == "[1, 1, 28, 28]" and fix["flat_text"] == "[1, 400]"
    assert fix["printed"] == "torch.Size([1, 10])" and tuple(fix["out"]) == cnn.FIXED_SHAPES[
        "cnn-fix-no-batch"]
    # the batched walk is the CNN's own shapes on the first batch, up to the Linear
    assert [tuple(s) for _b, s in m["right_walk"]] == [s for _b, s in cnn.shapes()[:-1]]
    assert m["in_features"]["code"] == list(B.code_lines("cnn-mistake-in-features"))

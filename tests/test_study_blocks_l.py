"""The numbers behind animations 218-220 (voxanim.study_blocks_l): one stamp weight's
gradient as the sum of its parts, the lines and losses the CNN and the MLP share, and the
first layer's stamps as they learn."""

from __future__ import annotations

import numpy as np
from conftest import needs_mnist

from voxanim import blocks as B
from voxanim import study_blocks_l as L
from voxanim import study_course as course
from voxanim import study_course_cnn as cnn


# ---------------------------------------------------------------------- no data needed
def test_block_names_come_from_the_code():
    assert L.block_names("cnn-model") == [
        "Conv2d(1, 8, 3)", "ReLU()", "MaxPool2d(2)", "Conv2d(8, 16, 3)", "ReLU()",
        "MaxPool2d(2)", "Flatten()", "Linear(16 * 5 * 5, 10)"]
    assert L.block_names("model") == ["Flatten()", "Linear(784, 64)", "ReLU()",
                                      "Linear(64, 32)", "ReLU()", "Linear(32, 10)"]


def test_chunk_means_are_means_of_25():
    x, m = L.chunk_means(np.arange(60, dtype=float))
    assert list(x) == [12.5, 37.5, 55.0]
    assert np.allclose(m, [12.0, 37.0, 54.5])


def test_the_step_is_the_same_lines_in_both_programs():
    assert B.code_lines("step") == B.code_lines("cnn-step")
    assert len(B.code_lines("cnn-step")) == 6


def test_a_conv_by_loops():
    img = np.arange(25, dtype=float).reshape(5, 5)
    W = np.zeros((1, 1, 3, 3))
    W[0, 0, 1, 1] = 2.0
    out = L.conv_loop(img, W, np.array([1.0]))
    np.testing.assert_allclose(out[0], 2 * img[1:4, 1:4] + 1)


# ---------------------------------------------------------------------- 218
@needs_mnist
def test_the_parts_add_up_to_pytorchs_number():
    s = L.cnn_backward().validate()
    m = s.metadata
    assert m["shape_text"] == "[8, 1, 3, 3]" and m["printed"] == "torch.Size([8, 1, 3, 3])"
    assert m["weights"] == 72 and m["count"] == cnn.count() == 5258
    assert m["places"] == 676 and m["n_parts"] == 64 * 676 == 43264
    assert m["parts_sum"] == "64 × 676 = 43,264"
    assert abs(m["total"] - m["grad"]) < 1e-6
    assert m["grad"] == float(cnn.grads(0)["0.weight"][0, 0, 1, 1])
    assert m["total_text"] == m["grad_text"] == "0.05934"
    assert np.isclose(sum(m["sums"]), m["total"], atol=1e-12)
    assert m["loss_text"] == "2.370"
    # the digit worked out: its parts are pixel x red, and add up to its sum
    parts = np.asarray(s.panels[2].data["values"])
    red = np.asarray(s.panels[1].data["values"])
    img = np.asarray(s.panels[0].data["values"])[0]
    np.testing.assert_allclose(parts, img[1:27, 1:27] * red, rtol=1e-6, atol=1e-12)
    assert np.isclose(parts.sum(), m["digit_sum"]) and m["digit_sum_text"] == "0.00599"
    assert m["digit_nonzero"] == int((parts != 0).sum()) == 50
    assert np.isclose(m["running"][-1], m["digit_sum"])
    assert len(m["order"]) == 676 and len({tuple(rc) for rc in m["order"]}) == 676
    w = m["worked"]
    assert w[0]["part"] == parts.max() > 0 and w[1]["part"] == parts.min() < 0
    assert w[2]["red"] == 0 and w[2]["part"] == 0
    for ex in w:
        assert np.isclose(ex["pixel"] * ex["red"], ex["part"], rtol=1e-6, atol=1e-12)
    assert [ex["text"] for ex in w] == ["2.59 × 0.00039 = 0.00100",
                                        "2.50 × −0.00023 = −0.00059", "2.82 × 0 = 0"]
    assert m["label"] == 4 and m["call"] == "loss.backward()"


# ---------------------------------------------------------------------- 219
@needs_mnist
def test_the_same_loop_trains_both():
    s = L.same_loop().validate()
    m = s.metadata
    assert m["mlp_count_text"] == "52,650" and m["cnn_count_text"] == "5,258"
    assert m["program_lines"] == m["cnn_program_lines"] == 63 and m["same_lines"] == 55
    assert m["model_changed"] == 5
    assert {"data", "loader", "loss", "optimizer", "step", "train", "evaluate"} <= set(
        m["shared_regions"])
    assert "model" not in m["shared_regions"]
    assert m["chunks"] == 38 and m["cnn_lower"] == 38
    mlp, cnn_ = course.losses("sgd"), cnn.losses()
    assert np.isclose(m["mlp_curve"][0], mlp[:25].mean())
    assert np.isclose(m["cnn_curve"][-1], cnn_[925:].mean())
    assert m["end_mlp_text"] == "0.164" and m["end_cnn_text"] == "0.095"
    assert np.isclose(m["first_cnn"], 2.370, atol=5e-4)


# ---------------------------------------------------------------------- 220
@needs_mnist
def test_the_stamps_learn():
    s = L.stamps_learn().validate()
    m = s.metadata
    assert m["steps"] == [0, 10, 100, 500, 938]
    stamps = np.asarray(s.panels[0].data["values"]).reshape(5, 8, 3, 3)
    np.testing.assert_array_equal(stamps[0], cnn.init_params()["0.weight"][:, 0])
    np.testing.assert_array_equal(stamps[-1], cnn.params(938)["0.weight"][:, 0])
    assert np.abs(stamps[0]).max() <= 1 / 3
    assert m["moved"][0] == 0 and all(a < b for a, b in zip(m["moved"], m["moved"][1:]))
    assert m["moved_text"] == ["0.00", "0.02", "0.08", "0.12", "0.14"]
    assert m["label"] == 5 and m["map_shape"] == [8, 26, 26]
    before = np.asarray(s.panels[2].data["values"])
    np.testing.assert_allclose(before, cnn.activations(0, 0)["z1"][0], atol=1e-6)
    assert not np.allclose(before, s.panels[3].data["values"])
    assert m["conv_line"].strip() == "nn.Conv2d(1, 8, 3), nn.ReLU(),"

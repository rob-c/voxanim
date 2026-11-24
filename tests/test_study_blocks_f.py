"""The numbers behind animations 200-202 (voxanim.study_blocks_f): the shapes through the
model, the loss as the mean of -ln p, and loss.backward()'s gradients, one per number, each
its weight's shape."""

from __future__ import annotations

import numpy as np
from conftest import needs_mnist

from voxanim import study_blocks_f as F
from voxanim import study_course as course


# ---------------------------------------------------------------------- no data needed
def test_block_names_come_from_the_code():
    assert F.block_name("    nn.Linear(784, 64),") == "Linear(784, 64)"
    assert F.block_name("    nn.ReLU(),") == "ReLU()"
    assert F.block_name("nn.Flatten(),") == "Flatten()"


# ---------------------------------------------------------------------- the data
@needs_mnist
def test_the_forward_pass_changes_the_shape_at_each_block():
    s = F.forward_pass().validate()
    m = s.metadata
    assert m["names"] == ["x", "Flatten()", "Linear(784, 64)", "ReLU()", "Linear(64, 32)",
                          "ReLU()", "Linear(32, 10)"]
    assert m["shapes"] == [[64, 1, 28, 28], [64, 784], [64, 64], [64, 64], [64, 32], [64, 32],
                           [64, 10]]
    assert m["changes"][0] == "[64, 1, 28, 28] → [64, 784]"
    model = m["model"]
    for name, line in zip(m["names"][1:], m["lines"][1:]):
        assert model[line].strip() == "nn." + name + ","
    # ReLU is max(z, 0): its zeros are exactly where z was not positive
    p1, p2 = s.panels[1].data, s.panels[2].data
    np.testing.assert_array_equal(p1["values"], np.maximum(p1["before"], 0))
    np.testing.assert_array_equal(p2["values"], np.maximum(p2["before"], 0))
    assert m["zeroed"][0] == float((p1["before"] <= 0).mean())
    assert all(0.3 < z < 0.7 for z in m["zeroed"]) and m["zeroed_text"] == ["46%", "45%"]
    # the logits are the untrained model's, and small
    logits = course.forward(course.init_params(), course.first_batch()["x"])["logits"]
    np.testing.assert_allclose(s.panels[3].data["values"], logits)
    assert -0.5 < m["logits_low"] < 0 < m["logits_high"] < 0.5
    assert m["call"] == "logits = model(x)"


@needs_mnist
def test_the_loss_is_the_mean_of_minus_ln_p():
    s = F.the_loss().validate()
    m = s.metadata
    each = np.asarray(m["losses"])
    assert len(each) == 64
    assert np.isclose(each.mean(), m["loss"]) and np.isclose(m["total"] / 64, m["loss"])
    assert np.isclose(m["loss"], course.losses("sgd")[0], atol=1e-6)
    assert m["loss_text"] == "2.322" and m["ln10_text"] == "2.303"
    assert 0 < m["loss"] - np.log(10) < 0.05                    # an untrained model: a guess
    chances = np.asarray(m["chances"])
    assert np.isclose(chances.sum(), 1.0)
    assert np.isclose(-np.log(m["chance"]), m["digit_loss"], atol=1e-5)
    assert m["chance"] == chances[m["label"]] and m["chance_text"] == "0.094"
    assert m["digit_loss_text"] == "−ln 0.094 = 2.36"
    assert m["mean_text"] == "148.63 ÷ 64 = 2.322"
    assert m["lowest"] == each.min() < m["loss"] < each.max() == m["highest"]
    assert m["define"] == "loss_fn = nn.CrossEntropyLoss()"
    assert m["call"] == "loss = loss_fn(logits, y)"


@needs_mnist
def test_backward_gives_every_weight_a_gradient_of_its_shape():
    s = F.backward().validate()
    m = s.metadata
    P = course.init_params()
    for panel, L in zip(s.panels[:3], m["layers"]):
        g, w = panel.data["values"], panel.data["weight"]
        assert g.shape == w.shape == P[L["weight"]].shape == tuple(L["shape"])
        assert panel.data["bias_grad"].shape == P[L["bias"]].shape
    assert [L["shape_text"] for L in m["layers"]] == ["[64, 784]", "[32, 64]", "[10, 32]"]
    assert m["order"] == [2, 1, 0]                              # the last layer first
    assert m["running"] == [330, 2410, 52650] and m["total"] == course.count() == 52650
    assert m["printed"] == "torch.Size([64, 784])" == course.PRINTED_GRAD
    assert m["grad"] == ["print(model[1].weight.grad.shape)"]
    assert m["call"] == "loss.backward()"


@needs_mnist
def test_the_gradients_are_the_slope_of_the_loss():
    """A few of the gradients against the loss's change when that one number is nudged
    (in float64, where the nudge is not lost in rounding)."""
    P = {k: v.astype(np.float64) for k, v in course.init_params().items()}
    b = course.first_batch()
    x, y = b["x"].astype(np.float64), b["y"]
    _loss, G = course.loss_and_grads(P, x, y)
    recorded = course.grads(0, "sgd")
    rng = np.random.default_rng(0)
    for k in ("5.weight", "3.weight", "1.weight", "1.bias"):
        np.testing.assert_allclose(recorded[k], G[k], atol=1e-6)
        for _ in range(3):
            i = tuple(int(rng.integers(n)) for n in P[k].shape)
            h = 1e-6
            up = {kk: v.copy() for kk, v in P.items()}
            down = {kk: v.copy() for kk, v in P.items()}
            up[k][i] += h
            down[k][i] -= h
            slope = (course.loss_of(up, x, y) - course.loss_of(down, x, y)) / (2 * h)
            assert abs(slope - G[k][i]) < 1e-6

"""The numbers behind animations 206-208 (voxanim.study_blocks_h): Adam's m and v and one
weight's update, the training loop's stations, and 60,000 digits in batches of 64 over
three epochs."""

from __future__ import annotations

import numpy as np
from conftest import needs_mnist

from voxanim import study_blocks_h as H
from voxanim import study_course as course


# ---------------------------------------------------------------------- no data needed
def test_small_numbers_are_written_with_a_power_of_ten():
    assert H.sci(2.0566e-6) == "2.06×10⁻⁶"
    assert H.sci(9.996e-6) == "1.00×10⁻⁵"
    assert H.num(-0.0045351, 5) == "−0.00454" and H.num(-0.0000001, 3) == "0.000"
    assert H.signed(0.000926, 5) == "+0.00093" and H.signed(-0.02, 2) == "−0.02"


def test_adams_move_is_the_learning_rate_over_the_ratio():
    # the first step: m = 0.1 g and v = 0.001 g², so the move is lr times the sign of g
    for g in (0.5, -0.003, 2e-4):
        _m_hat, _root, move = H.adam_move(0.1 * g, 0.001 * g * g, 1)
        assert np.isclose(move, -1e-3 * np.sign(g), rtol=1e-4)
    m_hat, root, move = H.adam_move(-0.00625, 2.53e-6, 2)
    assert np.isclose(m_hat, -0.00625 / 0.19) and np.isclose(root, np.sqrt(2.53e-6 / 0.001999))
    assert np.isclose(move, 1e-3 * 0.00625 / 0.19 / root, rtol=1e-6)


def test_the_mean_of_the_last_steps():
    np.testing.assert_allclose(H.moving_mean([1, 2, 3, 4], 2), [1, 1.5, 2.5, 3.5])
    assert H.pick_weight(np.array([[1.0, -1.0], [1.0, 1.0]]),
                         np.array([[0.2, -0.9], [0.1, 0.3]])) == (0, 1)


def test_sixty_thousand_digits_make_938_steps():
    full, last = divmod(course.TRAIN, course.BATCH)
    assert (full, last) == (937, 32) and full * 64 + last == 60_000
    assert course.EPOCH_STEPS == 938 == full + 1


# ---------------------------------------------------------------------- the runs
@needs_mnist
def test_adam_keeps_m_and_v_and_moves_each_weight_by_about_the_rate():
    s = H.adam().validate()
    m = s.metadata
    assert m["parameters"] == 52_650 and m["adam_numbers"] == 3 * 52_650 == 157_950
    w, mm, v = np.asarray(s.panels[0].data["values"])
    assert w.shape == mm.shape == v.shape == (10, 32) and (v >= 0).all()
    # the worked step, from the recorded state and gradient
    assert np.isclose(m["m"], 0.9 * m["m_old"] + 0.1 * m["g"], rtol=1e-5)
    assert np.isclose(m["v"], 0.999 * m["v_old"] + 0.001 * m["g"] ** 2, rtol=1e-4)
    assert np.isclose(m["m_old"], 0.1 * m["g_prev"], rtol=1e-5)          # m after step 1
    assert np.isclose(m["recorded_move"], m["move"], rtol=1e-3)
    assert np.isclose(m["w_after"] - m["w_before"], m["move"], rtol=1e-3)
    assert np.sign(m["move"]) == -np.sign(m["g"])
    assert 0.0005 < abs(m["move"]) <= 0.00101 and m["largest_move"] < 0.00101
    T = m["text"]
    assert T["m_sum"] == "0.9 × −0.004535 + 0.1 × −0.021709" and T["m"] == "−0.006252"
    assert T["v"] == "2.526×10⁻⁶" and T["move"] == "+0.00093" and T["largest"] == "0.0010"
    assert T["root"] == "√(v ÷ 0.001999) = 0.0355" and T["m_hat"] == "÷ 0.19 = −0.0329"

    # the arithmetic on screen adds up in the digits shown
    def f(s):
        return float(s.replace("−", "-").replace("×10⁻⁶", "e-6"))
    assert H.num(0.9 * f(T["m_old"]) + 0.1 * f(T["g"]), 6) == T["m"]
    assert H.sci(0.999 * f(T["v_old"]) + 0.001 * f(T["g"]) ** 2, 3) == T["v"]
    assert H.num(f(T["m"]) / 0.19, 4) == "−0.0329"
    assert H.num(np.sqrt(f(T["v"]) / 0.001999), 4) == "0.0355"
    assert H.signed(0.001 * 0.0329 / 0.0355, 5) == T["move"]
    # the real runs: Adam lower at first, SGD lower from step 237 on
    sgd, adam = np.asarray(m["sgd_curve"]), np.asarray(m["adam_curve"])
    k = m["adam_ahead_until"] + 1
    assert k == 237 and (sgd[k:] <= adam[k:]).all() and (adam[1:50] < sgd[1:50]).all()
    assert (m["sgd_accuracy"], m["adam_accuracy"]) == (0.9494, 0.9461)
    assert m["code"][m["adam_line_index"]].strip() == "optimizer = Adam(model.parameters())"


@needs_mnist
def test_the_loop_has_five_lines_as_stations():
    s = H.training_loop().validate()
    m = s.metadata
    assert len(m["code"]) == 8 and "x.to(device)" not in " ".join(m["code"])
    assert [x["station"] for x in m["stations"]] == ["data", "forward", "loss", "backward",
                                                     "backward", "step"]
    for x in m["stations"]:
        assert m["code"][x["line"]].strip() == x["code"]
    assert m["reads"]["data"] == "x: [64, 1, 28, 28]" and m["reads"]["forward"] == \
        "logits: [64, 10]"
    assert m["reads"]["loss"] == "loss: 2.322" and abs(m["first_loss"] - np.log(10)) < 0.05
    assert m["gradients"] == m["parameters"] == 52_650
    assert len(m["losses"]) == 938 and m["losses"][-1] == m["last_loss"]
    assert m["printed"] == "0 " + repr(float(course.losses("sgd")[-1]))
    assert m["last_loss"] < 0.2 < 2.3 < m["first_loss"]


@needs_mnist
def test_epochs_see_every_digit_once_in_a_new_order():
    s = H.epochs_and_batches().validate()
    m = s.metadata
    assert (m["steps"], m["full"], m["last_batch"], m["quotient"]) == (938, 937, 32, 937.5)
    assert m["division"] == "60,000 ÷ 64 = 937.5" and m["check"] == "937 × 64 + 32 = 60,000"
    sizes = np.asarray(s.panels[0].data["values"])
    assert sizes.sum() == 60_000 and len(sizes) == 938 and sizes[-1] == 32
    first = np.asarray(m["first"])
    assert len({tuple(r) for r in first.tolist()}) == 3 and not m["same_first"]
    for e in range(3):
        assert (course.order(e)[:8] == first[e]).all()
    _x, y = course.data()[:2]
    assert m["shown_labels"] == [[int(v) for v in y[r]] for r in m["shown"]]
    loss = np.asarray(m["losses"]).reshape(3, 938)
    np.testing.assert_allclose(loss.mean(1), m["epoch_mean"])
    assert m["epoch_mean"][0] > m["epoch_mean"][1] > m["epoch_mean"][2]
    np.testing.assert_allclose(loss[0], course.losses("sgd"))
    assert m["accuracy"][0] == 0.9494 and m["accuracy"][2] > m["accuracy"][0]
    # per step the loss jumps: within the third epoch it spans far more than the means differ
    assert m["epoch_max"][2] - m["epoch_min"][2] > m["epoch_mean"][0] - m["epoch_mean"][2]

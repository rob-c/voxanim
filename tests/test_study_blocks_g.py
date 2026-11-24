"""The numbers behind animations 203-205 (voxanim.study_blocks_g): zero_grad, one SGD step and
three learning rates, from the course's recorded runs."""

from __future__ import annotations

import numpy as np
from conftest import needs_mnist

from voxanim import study_blocks_g as S
from voxanim import study_course as course
from voxanim.blocks import val


# ---------------------------------------------------------------------- no data needed
def test_chunk_means_cover_every_step_once():
    x, m = S.chunk_means(np.arange(60.0), 25)
    assert list(x) == [25, 50, 60]
    np.testing.assert_allclose(m, [12.0, 37.0, 54.5])


def test_the_tiles_read_out_are_large_spread_out_and_of_both_signs():
    g = np.zeros((4, 20))
    g[0, 0], g[1, 1], g[2, 9], g[3, 18] = -0.9, -0.8, 0.5, -0.4
    assert S.biggest(g, 3, 8) == [(0, 0), (2, 9), (3, 18)]   # (1, 1) is too close to (0, 0)


def test_a_step_adds_up_as_shown():
    assert S.agrees(0.15205856, -0.045350302)          # 0.1521 + 0.00454 = 0.1566
    assert not S.agrees(0.027738405, 0.017832227)      # 0.0277 - 0.00178 = 0.0259, not 0.0260


def test_two_figures():
    assert [S.two_figures(v) for v in (4.535e-5, 0.004535, 0.03628)] == [
        "0.000045", "0.0045", "0.036"]


# ---------------------------------------------------------------------- 203
@needs_mnist
def test_zero_grad_empties_and_the_forgotten_run_piles_up():
    s = S.zero_grad().validate()
    m = s.metadata
    assert m["shape"] == [10, 32] and m["numbers"] == 52_650
    G0, G1 = course.grads(0), course.grads(1)
    np.testing.assert_array_equal(np.asarray(m["grad"]), G0["5.weight"])
    np.testing.assert_array_equal(np.asarray(m["grad_pile"]), G0["5.weight"] + G1["5.weight"])
    for i, (r, c) in enumerate(m["tiles"]):
        assert m["g0"][i] == G0["5.weight"][r, c] and m["g1"][i] == G1["5.weight"][r, c]
        assert np.isclose(m["pile"][i], m["g0"][i] + m["g1"][i])
    # the sizes: the first gradient, the second, and the two added (as the run used them)
    assert val(m["length_first"]) == "0.84" and val(m["length_next"]) == "0.78"
    assert val(m["length_pile"]) == "1.29"
    assert m["used"][0] == m["fresh"][0]
    assert all(b > a for a, b in zip(m["used"][:-1], m["used"][1:-1]))      # it grows
    at100 = m["batches"].index(100)
    assert m["used"][at100] > 30 > 4 > m["fresh"][at100]
    assert len(m["used_all"]) == len(m["fresh_all"]) == 938
    # the loss goes wrong, and the model learns nothing
    assert m["last_bad"] > 2.3 > 0.2 > m["last_sgd"]
    assert (m["percent_sgd"], m["percent_bad"]) == ("95%", "10%")
    assert m["accuracy_bad"] < 0.11
    assert "optimizer.zero_grad()" in m["code"] and "optimizer.zero_grad()" not in m["mistake"]


# ---------------------------------------------------------------------- 204
@needs_mnist
def test_one_sgd_step_with_real_numbers():
    s = S.optimizer_step().validate()
    m = s.metadata
    W, G, after = (np.asarray(m[k]) for k in ("weight", "grad", "after"))
    np.testing.assert_array_equal(after, course.params(1)["5.weight"])
    np.testing.assert_allclose(after, W - 0.1 * G, atol=1e-7)
    signs = set()
    for i, (r, c) in enumerate(m["tiles"]):
        assert m["before_w"][i] == W[r, c] and m["after_w"][i] == after[r, c]
        assert np.isclose(m["move"][i], -0.1 * m["g"][i])
        # the arithmetic on screen, to four places, adds up
        shown = round(m["before_w"][i], 4) - 0.1 * round(m["g"][i], 4)
        assert val(shown, 4) == val(m["after_w"][i], 4)
        signs.add(np.sign(m["g"][i]))
    assert signs == {-1.0, 1.0}                                  # one moves up, one down
    assert (val(m["loss_before"], 4), val(m["loss_after"], 4)) == ("2.3223", "2.2616")
    b = course.batch(0)
    assert m["loss_after"] == course.loss_of(course.params(1), b["x"], b["y"])
    assert m["drop"] > 0.05
    assert sum(m["sizes"].values()) == m["numbers"] == 52_650 and m["layer_numbers"] == 320
    assert 52_000 < m["changed"] <= 52_650


# ---------------------------------------------------------------------- 205
@needs_mnist
def test_three_learning_rates_from_one_start():
    s = S.learning_rate().validate()
    m = s.metadata
    assert m["rates"] == [0.001, 0.1, 0.8] and m["runs"] == ["lr_small", "sgd", "lr_large"]
    assert m["percent"] == ["55%", "95%", "27%"]
    for run, curve in zip(m["runs"], m["curves"]):
        _x, means = S.chunk_means(course.losses(run))
        np.testing.assert_allclose(curve, means)
        assert course.losses(run)[0] == m["first_loss"]
    small, right, large = m["last"]
    assert small > 1.8 and right < 0.2 and large > 1.5
    assert m["highest"][2] > 15 and m["highest"][1] == m["first_loss"]
    assert m["moves_text"] == ["0.000045", "0.0045", "0.036"]
    assert m["lr_line"].strip() == "lr=0.1)"

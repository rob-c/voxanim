"""The experiments behind animations 172-175 (voxanim.study_info): the impurity
arithmetic and what it counts, the ten-class tree and its identities, the classifier's
gradients, the entropy estimators on cases with known answers, and every measured claim,
as relative claims (CI retrains on another machine, where the last bits differ)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fd_check, needs_mnist

from voxanim import study_info as I

# ---------------------------------------------------------------------- impurity


def test_entropy_and_gini_of_known_counts():
    even = np.ones(10)
    assert I.entropy(even) == pytest.approx(np.log2(10))
    assert I.gini(even) == pytest.approx(0.9)
    assert I.entropy([5, 0, 0]) == 0 and I.gini([5, 0, 0]) == 0
    assert I.entropy([1, 1]) == pytest.approx(1.0) and I.gini([1, 1]) == pytest.approx(0.5)
    c = np.array([3, 1, 0, 4])
    assert I.entropy_terms(c).sum() == pytest.approx(I.entropy(c))
    assert I.gini_terms(c).sum() == pytest.approx(I.gini(c))
    assert I.entropy(c, np.e) == pytest.approx(I.entropy(c) * np.log(2))
    # termwise -ln p >= 1 - p: in nats entropy is never below Gini
    rng = np.random.default_rng(0)
    for _ in range(200):
        c = rng.integers(0, 20, 10)
        if c.sum():
            assert I.entropy(c, np.e) >= I.gini(c) - 1e-12


def test_huffman_is_within_one_bit_of_entropy_and_a_prefix_code():
    rng = np.random.default_rng(1)
    for _ in range(100):
        c = rng.integers(0, 50, 10).astype(float)
        if (c > 0).sum() < 2:
            continue
        L = I.huffman_lengths(c)
        mean = float((I.proportions(c) * L).sum())
        assert I.entropy(c) - 1e-9 <= mean < I.entropy(c) + 1
        live = L[c > 0]
        assert np.isclose((2.0 ** -live).sum(), 1.0)          # Kraft: a full prefix code
    assert list(I.huffman_lengths([4, 2, 1, 1])) == [1, 2, 3, 3]


def test_drawn_pairs_measure_gini():
    c = np.array([50, 30, 20])
    share, se = I.pairs_differ(c)
    assert abs(share - I.gini(c)) < 4 * se


# ---------------------------------------------------------------------- the ten-class tree


def test_tree_on_a_step_splits_once_and_is_pure():
    xb = np.repeat(np.arange(8), 10)[:, None]
    y = (np.arange(80) >= 40).astype(np.int64) * 3
    T = I.grow_multi(xb, y)
    assert T["feature"][0] == 0 and T["cut"][0] == 3
    assert T["gain"][0] == pytest.approx(1.0)
    assert (T["feature"][1:] == -1).all()
    assert I.path_nodes(T, xb, 5).tolist() == [1] * 40 + [2] * 40


@needs_mnist
def test_the_tree_spends_its_entropy_and_the_leaves_are_its_loss():
    T = I.tree_of("entropy")
    D = T["depths"]
    n0 = T["counts"][0].sum()
    gains = (T["gain"] * T["counts"].sum(1) / n0)[T["feature"] >= 0].sum()
    assert gains == pytest.approx(D[0, 1] - D[-1, 1], abs=1e-9)      # gains add up
    assert np.all(np.diff(D[:, 1]) <= 1e-12)                          # entropy only falls
    assert np.allclose(D[:, 7], D[:, 1] * np.log(2), atol=1e-9)        # = unsmoothed loss
    assert D[0, 1] == pytest.approx(np.log2(10), abs=0.01)
    assert D[-1, 8] > 100                                              # zero probabilities
    best = int(np.argmin(D[:, 5]))
    assert 3 < best < len(D) - 1 and D[-1, 5] > D[best, 5]             # held-out turns up
    assert D[best, 6] > 0.75
    # each child's counts add up to its parent's
    inner = np.flatnonzero(T["feature"] >= 0)
    assert np.array_equal(T["counts"][T["left"][inner]] + T["counts"][T["right"][inner]],
                          T["counts"][inner])


# ---------------------------------------------------------------------- the classifier


def test_classifier_gradients_match_finite_differences(monkeypatch):
    monkeypatch.setattr(I, "SIZES", (6, 5, 4, 3))
    rng = np.random.default_rng(2)
    P = {k: v.astype(np.float64) + 0.1 * rng.normal(size=v.shape)
         for k, v in I.init_mlp(0).items()}
    x, y = rng.random((7, 6)), rng.integers(0, 3, 7)
    _, G = I.mlp_grads(P, x, y)
    fd_check(lambda Q: I.mlp_grads(Q, x, y)[0], P, G, list(P))


def test_entropy_estimators_on_known_cases():
    rng = np.random.default_rng(3)
    a, b = rng.normal(size=50000), rng.normal(size=50000)
    assert I.mutual_information(a, b) < 0.01                       # independent: about 0
    assert I.mutual_information(a, a) == pytest.approx(4.0, abs=0.01)   # = log2 16
    assert I.spectral_entropy(np.eye(8)) == pytest.approx(3.0)       # 8 equal directions
    assert I.spectral_entropy(np.outer([1, 2, 3], [1, 1])) == pytest.approx(0.0, abs=1e-9)
    v = rng.normal(0, 1, 200000)
    # a pile spread twice as wide has one more bit in fixed bins
    assert I.histogram_entropy(2 * v, 0.01) - I.histogram_entropy(v, 0.01) == \
        pytest.approx(1.0, abs=0.02)


@needs_mnist
def test_weights_spread_out_but_structure_forms():
    m = I.weight_entropy().validate().metadata
    for k in range(len(I.SEEDS)):
        h, s = m["histogram_entropy"], m["spread_bits"]
        rise = h["after"][k] - h["before"][k]
        assert rise > 0.3 and 0.6 * rise < s[k] < 1.1 * rise          # up, mostly spread
        assert m["neighbour_information"]["before"][k] < 0.01
        assert m["neighbour_information"]["after"][k] > 0.1
        assert abs(m["neighbour_correlation"]["before"][k]) < 0.02
        assert m["neighbour_correlation"]["after"][k] > 0.3
        se = m["spectral_entropy"]["W0"]
        assert se["after"][k] < se["before"][k] - 0.2                 # fewer directions
        assert m["accuracy"]["before"][k] < 0.25 and m["accuracy"]["after"][k] > 0.95
        assert m["test_nats"]["before"][k] == pytest.approx(np.log(10), abs=0.2)
    assert m["never_lit_unchanged"] == m["never_lit"] > 30
    assert m["parameters"] == m["weights"] + m["biases"] == 52650


@needs_mnist
def test_loss_in_statistical_terms():
    m = I.logits_loss_and_nats().validate().metadata
    assert m["shown_top2_gap"] == pytest.approx(m["shown_top2_log_ratio"], abs=1e-9)
    assert m["shown_loss_nats"] == pytest.approx(-np.log(m["shown_probabilities"][
        m["shown_label"]]))
    assert m["mean_bits"] == pytest.approx(m["mean_nats"] / np.log(2))
    assert m["mean_nats"] > 10 * m["median_nats"]                       # a heavy tail
    assert m["worst_shares"]["100"] > 0.3
    assert m["uniform_nats"] == pytest.approx(np.log(10))
    assert m["prior_nats"] > 2.2 and m["mean_nats"] < 0.2
    assert m["standard_error"] < 0.1 * m["mean_nats"]
    assert m["total_bits"] == pytest.approx(-m["log_likelihood"] / np.log(2))
    assert m["total_bits"] == pytest.approx(m["n"] * m["mean_bits"])
    assert m["raw_bits"] == pytest.approx(m["n"] * np.log2(10))
    assert m["perplexity"] == pytest.approx(np.exp(m["mean_nats"]))
    assert m["worst_named"] != m["worst_label"] and m["worst_loss_nats"] > 5
    # temperature: the second half's loss is lowest near the T the first half chose
    second = np.array(m["loss_second_half"])
    assert m["second_half_at_best"] <= second.min() + 0.01
    assert second[0] > m["second_half_at_best"] and second[-1] > m["second_half_at_best"]


def test_temperature_moves_loss_not_accuracy():
    rng = np.random.default_rng(4)
    z, y = rng.normal(0, 3, (500, 10)), rng.integers(0, 10, 500)
    acc = [(I.log_softmax(z / T).argmax(1) == y).mean() for T in (0.5, 1, 2)]
    loss = [I.per_digit_loss(I.log_softmax(z / T), y).mean() for T in (0.5, 1, 2)]
    assert len(set(acc)) == 1 and len(set(np.round(loss, 6))) == 3

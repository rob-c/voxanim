"""The experiment behind animation 181 (voxanim.study_bridge_reward): the two gradients on
the logits and the network's backward pass against finite differences, the exact
expectation of the reward's gradient, the reward's entropy, the helpers on cases with known
answers, and every measured claim as a relative claim (CI retrains on another machine,
where the last bits differ)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fd_check, needs_mnist

from voxanim import study_bridge_reward as S

NAME = "labels-against-rewards"


def _probabilities(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return S.softmax(rng.normal(0, 2, (n, S.CLASSES))), rng.integers(0, S.CLASSES, n)


# ---------------------------------------------------------------------- the two gradients
def test_label_gradient_is_the_derivative_of_cross_entropy():
    rng = np.random.default_rng(1)
    z = rng.normal(0, 2, (4, S.CLASSES))
    y = np.array([0, 3, 9, 3])
    g = S.label_gradient(S.softmax(z), y)
    e = 1e-6
    for i in range(4):
        for k in range(S.CLASSES):
            dz = np.zeros_like(z)
            dz[i, k] = e
            fd = (-np.log(S.softmax(z + dz)[i, y[i]]) + np.log(S.softmax(z - dz)[i, y[i]])) / (
                2 * e)
            assert fd == pytest.approx(g[i, k], abs=1e-7)
    np.testing.assert_allclose(g.sum(1), 0.0, atol=1e-12)        # p sums to one


def test_reward_gradient_is_the_derivative_of_the_weighted_log_probability():
    rng = np.random.default_rng(2)
    z = rng.normal(0, 2, (5, S.CLASSES))
    a = np.array([1, 1, 4, 7, 0])
    r = np.array([1.0, 0.0, 1.0, 0.0, 0.0])
    b = 0.3
    g = S.reward_gradient(S.softmax(z), a, r, b)
    e = 1e-6
    for i in range(5):
        for k in range(S.CLASSES):
            dz = np.zeros_like(z)
            dz[i, k] = e

            def loss(zz, i=i):
                return -(r[i] - b) * np.log(S.softmax(zz)[i, a[i]])
            assert (loss(z + dz) - loss(z - dz)) / (2 * e) == pytest.approx(g[i, k], abs=1e-7)


def test_a_right_guess_is_the_labels_gradient_scaled_and_a_wrong_one_pushes_it_down():
    p, y = _probabilities()
    b = 0.4
    right = S.reward_gradient(p, y, np.ones(len(y)), b)
    np.testing.assert_allclose(right, (1 - b) * S.label_gradient(p, y), atol=1e-15)
    a = (y + 1) % S.CLASSES
    wrong = S.reward_gradient(p, a, np.zeros(len(y)), b)
    rows = np.arange(len(y))
    # descent subtracts the gradient: the wrong guess's entry b(1 - p_a) is positive, so its
    # logit falls, and every other entry -b p_k is negative, so the rest rise a little
    np.testing.assert_allclose(wrong[rows, a], b * (1 - p[rows, a]))
    others = np.ones_like(wrong, bool)
    others[rows, a] = False
    assert np.all(wrong[rows, a] > 0) and np.all(wrong[others] < 0)
    # with no baseline a wrong guess teaches nothing
    np.testing.assert_allclose(S.reward_gradient(p, a, np.zeros(len(y)), 0.0), 0.0)


def test_the_expected_reward_gradient_is_p_of_the_label_times_the_labels():
    p, y = _probabilities(8, 3)
    rows = np.arange(len(y))
    want = p[rows, y][:, None] * S.label_gradient(p, y)
    for b in (0.0, 0.1, 0.47, 1.0, -2.0):                       # whatever the baseline
        np.testing.assert_allclose(S.expected_reward_gradient(p, y, b), want, atol=1e-14)


def test_sampled_guesses_average_to_the_expectation():
    p, y = _probabilities(1, 4)
    rng = np.random.default_rng(5)
    n = 200000
    a = S.sample(np.repeat(p, n, 0), rng.random(n))
    np.testing.assert_allclose(np.bincount(a, minlength=S.CLASSES) / n, p[0], atol=0.005)
    g = S.reward_gradient(np.repeat(p, n, 0), a, (a == y[0]).astype(float), 0.3).mean(0)
    np.testing.assert_allclose(g, S.expected_reward_gradient(p, y, 0.3)[0], atol=0.006)
    assert list(S.sample(np.eye(S.CLASSES)[[2, 7]], [0.0, 0.999])) == [2, 7]


def test_backward_matches_finite_differences():
    rng = np.random.default_rng(6)
    P = {k: v.astype(np.float64) for k, v in S.init(0).items()}
    P["b0"] = rng.normal(0, 0.1, P["b0"].shape)                 # some units off, some on
    x = rng.random((5, 784))
    y = np.array([1, 2, 3, 4, 5])

    def loss(Q):
        return float(-np.log(S.softmax(S.forward(Q, x)[1])[np.arange(5), y]).mean())
    h, z = S.forward(P, x)
    G = S.backward(P, x, h, S.label_gradient(S.softmax(z), y) / 5)
    fd_check(loss, P, G, ["W0", "b0", "W1", "b1"])


# ---------------------------------------------------------------------- measures
def test_binary_entropy():
    assert S.binary_entropy(0.5) == pytest.approx(1.0)
    assert S.binary_entropy(0.0) == 0 and S.binary_entropy(1.0) == 0
    q = np.linspace(0.01, 0.99, 99)
    np.testing.assert_allclose(S.binary_entropy(q), S.binary_entropy(1 - q))
    assert np.all(S.binary_entropy(q) <= 1.0) and S.binary_entropy(0.9) == pytest.approx(
        0.4689955935892812)


def test_first_reach_windows_and_stalls():
    probes = np.array([0, 10, 20, 100, 200, 300])
    acc = np.array([0.1, 0.5, 0.85, 0.8, 0.92, 0.95])
    assert S.first_reach(acc, 0.8, probes) == 20 and S.first_reach(acc, 0.9, probes) == 200
    assert S.first_reach(acc, 0.99, probes) is None
    means, ends = S.windows(np.arange(10.0), 4)
    np.testing.assert_allclose(means, [1.5, 5.5])
    assert list(ends) == [4, 8]
    by = np.full((6, 3), 0.9)
    by[1:5, 1] = 0.0                       # digit 1: unnamed from probe 10 to probe 200
    by[0:2, 2] = 0.0                       # digit 2: unnamed only before update 100
    assert S.stalls(by, probes, after=100) == [dict(digit=1, since=100, until=200)]


# ---------------------------------------------------------------------- measured
@pytest.fixture(scope="module")
def study():
    return S.STUDIES[NAME]().validate()


@needs_mnist
def test_both_kinds_start_alike_and_see_the_same_digits(study):
    m = study.metadata
    for k in range(len(m["seeds"])):
        assert m["accuracy"]["label"][k][0] == m["accuracy"]["reward"][k][0]
    assert m["examples"] == m["updates"] * m["batch"]
    r = S.runs()
    base = r["reward_0_base"]
    assert base[0] == S.B0 and np.all((base >= 0) & (base <= 1))
    # the baseline is the moving average of the rewards before each update
    rew = r["reward_0_reward"]
    assert base[1] == pytest.approx(S.BETA * S.B0 + (1 - S.BETA) * rew[0])


@needs_mnist
def test_the_shown_digit_is_worked_out_exactly(study):
    sh = study.metadata["shown"]
    p, y, b = np.array(sh["probabilities"]), sh["label"], sh["baseline"]
    assert 0.3 < sh["p_label"] < 0.7 and p.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(sh["expected_gradient"], sh["p_label"] * np.array(
        sh["label_gradient"]), atol=1e-12)
    for a, r, g, c in zip(sh["draws"], sh["draw_rewards"], sh["draw_gradients"],
                          sh["draw_cosines"]):
        assert r == int(a == y)
        np.testing.assert_allclose(g, S.reward_gradient(p[None], [a], [r], b)[0])
        if r:
            assert c == pytest.approx(1.0)
    assert 0 < sh["draw_rewards"].count(1) < len(sh["draws"])        # some right, some wrong
    # the mean of many draws is the expectation, up to sampling noise
    assert sh["mean_error"] < 0.01 and sh["mean_cosine"] > 0.999
    assert abs(sh["many_right"] - sh["p_label"]) < 4 * np.sqrt(0.25 / sh["many"])
    assert sh["right_cosine"] == pytest.approx(1.0) and 0 < sh["wrong_cosine"] < 1


@needs_mnist
def test_rewards_learn_slower_than_labels(study):
    m = study.metadata
    reach = m["reach"]
    for k in range(len(m["seeds"])):
        la, re = reach["0.90"]["label"][k], reach["0.90"]["reward"][k]
        assert la is not None and (re is None or re > 2 * la)
        assert m["final_accuracy"]["label"][k] > m["final_accuracy"]["reward"][k]
        assert m["final_accuracy"]["reward"][k] > 0.9                 # it does learn
        assert m["final_accuracy"]["label"][k] > 0.96
    for g in m["gap_90"]:
        assert g is None or g > 2
    for c in m["batch_cosine"]["mean"]:                             # same way, much noise
        assert 0.1 < c < 0.7


@needs_mnist
def test_a_reward_carries_at_most_a_bit(study):
    info = study.metadata["information"]
    H = np.array(info["entropy"])
    rate = np.array(info["reward_rate"])
    np.testing.assert_allclose(H, S.binary_entropy(rate))
    assert H.max() <= 1.0 < info["label_bits"] <= info["label_bits_max"]
    assert info["label_bits"] == pytest.approx(np.log2(10), abs=0.01)
    for k in range(len(H)):
        assert info["peak"][k] > 0.8 > 0.4 > info["final_entropy"][k]   # it falls as q rises
        assert info["final_rate"][k] > 0.9
    total = np.array(info["reward_total_bits"])
    assert np.all(total < info["label_total_bits"] / 4)
    np.testing.assert_allclose(total, H.mean(1) * study.metadata["examples"])


@needs_mnist
def test_digits_a_rewarded_run_stopped_naming(study):
    m = study.metadata
    assert not any(m["stalls"]["label"])                            # labels name every digit
    assert m["stalled_runs"] == sum(bool(s) for s in m["stalls"]["reward"])
    for seed, ss in zip(m["seeds"], m["stalls"]["reward"]):
        by = S.run("reward", seed, "by_class")
        for st in ss:
            i, j = (int(np.flatnonzero(st[w] == S.PROBES)[0]) for w in ("since", "until"))
            assert by[i, st["digit"]] < S.DEAD and by[j, st["digit"]] < S.DEAD
    # the labelled runs' slowest digit after 100 updates beats the rewarded runs'
    late = S.PROBES >= 100
    for seed in m["seeds"]:
        assert S.run("label", seed, "by_class")[late].min() > 0.5
    worst = min(S.run("reward", s, "by_class")[late].min() for s in m["seeds"])
    assert worst < min(S.run("label", s, "by_class")[late].min() for s in m["seeds"])

"""The experiment behind animation 180 (voxanim.study_bridge_text): the GRU's gradients and
its streaming evaluation, the tables' per-position bits and buckets, and every measured
claim, as relative claims (CI retrains on another machine, where the last bits differ)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fd_check, needs_text

from voxanim import study_bridge_text as S
from voxanim.study_chance import fit_order, order_bits


def tiny(seed=0, V=7, H=5):
    rng = np.random.default_rng(seed)
    P = S.gru_init(rng, V, H, np.float64)
    for k in P:                                   # away from zero biases: every term counts
        P[k] = P[k] + rng.standard_normal(P[k].shape) * 0.3
    return P, rng


# ---------------------------------------------------------------------- the network
def test_gru_gradients_match_finite_differences():
    P, rng = tiny()
    x, y = rng.integers(0, 7, (3, 6)), rng.integers(0, 7, (3, 6))
    h = rng.standard_normal((3, 5)) * 0.5
    _, G, _ = S.gru_loss_grad(P, x, y, h)
    assert set(G) == set(P) and all(G[k].shape == P[k].shape for k in P)
    fd_check(lambda Q: S.gru_loss_grad(Q, x, y, h)[0], P, G, list(P))


def test_gru_loss_is_the_mean_of_its_steps_and_carries_the_state():
    P, rng = tiny(1)
    x, y = rng.integers(0, 7, (2, 8)), rng.integers(0, 7, (2, 8))
    h0 = np.zeros((2, 5))
    loss, _, hT = S.gru_loss_grad(P, x, y, h0)
    # the same eight steps by hand
    h, total = h0, 0.0
    for t in range(8):
        h = S.gru_step(P, P["Wx"][x[:, t]] + P["b"], h)[0]
        total += -S.log_probs(P, h)[np.arange(2), y[:, t]].sum()
    assert loss == pytest.approx(total / 16) and np.allclose(h, hT)
    # two windows of four with the state carried = one window of eight
    l1, _, h1 = S.gru_loss_grad(P, x[:, :4], y[:, :4], h0)
    l2, _, h2 = S.gru_loss_grad(P, x[:, 4:], y[:, 4:], h1)
    assert (l1 + l2) / 2 == pytest.approx(loss) and np.allclose(h2, hT)


def test_sigmoid_and_the_parameter_count():
    v = np.linspace(-30, 30, 61)
    np.testing.assert_allclose(S.sigmoid(v), 1 / (1 + np.exp(-v)), atol=1e-12)
    P = S.gru_init(np.random.default_rng(0), 65, 128)
    assert S.n_params(P) == S.gru_count(65, 128) == 82881
    assert P["Wx"].dtype == np.float32


def test_streaming_evaluation_scores_every_position_with_its_warm_up():
    P, rng = tiny(2)
    seq = rng.integers(0, 7, 61)
    warm = 5

    def by_hand(i, start):
        h = np.zeros((1, 5))
        for t in range(start, i):
            h = S.gru_step(P, P["Wx"][seq[t:t + 1]] + P["b"], h)[0]
        return -S.log_probs(P, h)[0, seq[i]] / np.log(2)
    one = S.gru_bits(P, seq, warm=warm, streams=1)
    assert one.shape == (len(seq) - warm,)
    np.testing.assert_allclose(one, [by_hand(i, 0) for i in range(warm, len(seq))])
    four = S.gru_bits(P, seq, warm=warm, streams=4)
    L = -(-(len(seq) - warm) // 4)
    want = [by_hand(i, warm + (i - warm) // L * L - warm) for i in range(warm, len(seq))]
    np.testing.assert_allclose(four, want)


def test_clipping_and_the_stream_batches():
    G = dict(a=np.array([3.0, 0.0]), b=np.array([[4.0]]))
    C, norm = S.clip(G, 1.0)
    assert norm == pytest.approx(5.0)
    assert np.sqrt(sum((g ** 2).sum() for g in C.values())) == pytest.approx(1.0)
    C, _ = S.clip(G, 10.0)
    assert all(np.array_equal(C[k], G[k]) for k in G)
    gen = S.stream_batches(1001, streams=4, window=10)
    seen = [next(gen) for _ in range(60)]
    assert seen[0][1] and not seen[1][1]
    np.testing.assert_array_equal(seen[0][0], [0, 250, 500, 750])
    np.testing.assert_array_equal(seen[1][0] - seen[0][0], 10)
    fresh = [i for i, (_, f) in enumerate(seen) if f]
    assert fresh == [0, 25, 50]                   # 25 windows of 10 fill a stream of 250
    assert max(s.max() for s, _ in seen) + 10 <= 1000


# ---------------------------------------------------------------------- the tables
def test_position_bits_average_to_order_bits_and_buckets():
    rng = np.random.default_rng(3)
    train, test = rng.integers(0, 5, 3000), rng.integers(0, 5, 400)
    for k in (0, 1, 3):
        model = fit_order(train, k, 5)
        pb = S.position_bits(model, test, k, 5, 0.1, 6)
        mean, unseen = order_bits(model, test, k, 5, 0.1, 6)
        assert pb.mean() == pytest.approx(mean)
        cc = S.context_counts(model, test, k, 5, 6)
        assert (cc == 0).mean() == pytest.approx(unseen)
        # an unseen context gives every character 1/n
        np.testing.assert_allclose(pb[cc == 0], np.log2(5))
    np.testing.assert_array_equal(S.bucket_of([0, 1, 9, 10, 99, 100, 5000]),
                                  [0, 1, 1, 2, 2, 3, 3])
    assert S.bucket_names() == ["never", "1-9", "10-99", "100+"]
    b = S.buckets(np.array([0, 0, 5, 50, 500]), np.array([6.0, 6.0, 3.0, 2.0, 1.0]),
                  np.array([4.0, 2.0, 2.0, 2.0, 1.0]))
    np.testing.assert_allclose(b[:, 0], [0.4, 0.2, 0.2, 0.2])
    np.testing.assert_allclose(b[0, 1:], [6.0, 3.0, 2])


# ---------------------------------------------------------------------- measured
@pytest.fixture(scope="module")
def study():
    return S.markov_chain_to_rnn().validate()


@needs_text
def test_tables_overfit_with_order(study):
    m = study.metadata
    train, held = np.array(m["training_bits"]), np.array(m["held_out_bits"])
    assert (np.diff(train) < 0).all()                 # on their own text: always better
    k = m["best_order"]
    assert 0 < k < max(m["orders"]) and held[k] == held.min()
    assert (np.diff(held[k:]) > 0).all()              # held out: worse after the best order
    assert held[-1] > held[k] + 1.0
    assert (np.diff(m["unseen_share"]) >= 0).all() and m["unseen_share"][-1] > 0.4
    assert (held >= train - 1e-9).all()
    assert m["table_gap"] == pytest.approx(held[k] - train[k]) and m["table_gap"] > 0.5
    assert m["contexts"][0] == 1 and m["pairs"][0] == m["characters"] == 65
    assert m["possible_cells"] == 65 ** (k + 1) and m["best_pairs"] == m["pairs"][k]


@needs_text
def test_the_network_beats_the_best_table_on_held_out_text(study):
    m = study.metadata
    r = m["rnn"]
    held = np.array(m["rnn_held_out"])
    assert held[0] == pytest.approx(m["uniform_bits"], abs=0.1)   # untrained: about uniform
    assert m["rnn_final"] == held[-1] < m["best_bits"] - 0.05
    assert held[-1] < held[len(held) // 2] < held[1]              # still falling
    assert m["rnn_gap"] < m["table_gap"] / 2                       # it generalises better
    assert r["parameters"] == S.gru_count(m["characters"], r["hidden"])
    assert r["parameters"] < m["best_pairs"]                       # and is smaller
    # the cached network gives the recorded bits again
    P = S.gru_params()
    assert S.n_params(P) == r["parameters"]
    from voxanim.study_chance import char_ids
    assert S.gru_bits(P, char_ids("test")).mean() == pytest.approx(m["rnn_final"], abs=1e-4)


@needs_text
def test_buckets_show_where_the_table_has_no_data(study):
    m = study.metadata
    share = np.array(m["bucket_share"])
    tb, rb = np.array(m["bucket_table_bits"]), np.array(m["bucket_rnn_bits"])
    assert share.sum() == pytest.approx(1.0)
    assert sum(m["bucket_positions"]) == m["scored_positions"]
    assert share[0] == pytest.approx(m["unseen_share"][m["best_order"]])
    assert (share * tb).sum() == pytest.approx(m["best_bits"])     # identities
    assert (share * rb).sum() == pytest.approx(m["rnn_final"])
    assert tb[0] == pytest.approx(m["uniform_bits"])                # never counted: 1/65
    assert rb[0] < tb[0] - 1.0 and rb[1] < tb[1]                    # where data is rare
    assert abs(rb[3] - tb[3]) < 0.15                                # common: about even


@needs_text
def test_perplexity_and_the_example(study):
    m = study.metadata
    for name, b in m["perplexity_bits"].items():
        assert m["perplexity"][name] == pytest.approx(2.0 ** b)
    assert m["perplexity"]["uniform"] == pytest.approx(65.0)
    ex = m["example"]
    k = m["best_order"]
    assert ex["counts"][-1] == 0 and ex["counts"][k] >= S.EXAMPLE_MIN
    assert ex["table_bits"][-1] == pytest.approx(m["uniform_bits"])
    assert [len(c) for c in ex["contexts"]] == list(m["orders"])
    assert all(c == ex["before"][len(ex["before"]) - len(c):] for c in ex["contexts"])
    assert (np.diff(ex["counts"]) <= 0).all()

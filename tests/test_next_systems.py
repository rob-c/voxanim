"""Scientific invariants for unpublished systems, optimisation and RL studies."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import data_marks, fd_check, needs_gpt2, needs_mnist, needs_text

from voxanim import next_systems as S


def test_mlp_gradient():
    rng = np.random.default_rng(1)
    p = S.mlp_init(3, 4, 2, rng)
    x = rng.normal(size=(5, 3))
    y = np.array([0, 1, 0, 0, 1])
    _, g = S.mlp_loss(p, x, y)
    fd_check(lambda q: S.mlp_loss(q, x, y)[0], p, g, list(p))


@pytest.mark.parametrize("temperature", [0.05, 0.2, 1.0])
def test_infonce_gradient(temperature):
    rng = np.random.default_rng(2)
    p = {"W": rng.normal(size=(5, 3))}
    a = rng.normal(size=(6, 5))
    b = rng.normal(size=(6, 5))
    loss, grad, u, v = S.infonce_loss(p["W"], a, b, temperature)
    fd_check(lambda q: S.infonce_loss(q["W"], a, b, temperature)[0], p, {"W": grad}, ["W"])
    assert np.isfinite(loss)
    np.testing.assert_allclose(np.linalg.norm(u, axis=1), 1)
    np.testing.assert_allclose(np.linalg.norm(v, axis=1), 1)


def test_q_network_gradient():
    rng = np.random.default_rng(3)
    p = S.qnet_init(rng)
    x = rng.normal(size=(5, 2))
    a = np.array([0, 1, 2, 3, 0])
    target = rng.normal(size=5)
    _, g = S.qnet_loss(p, x, a, target)
    fd_check(lambda q: S.qnet_loss(q, x, a, target)[0], p, g, list(p))


def test_neural_ode_gradient():
    rng = np.random.default_rng(4)
    p = dict(
        W=rng.normal(size=(2, 3)),
        b=rng.normal(size=3),
        U=rng.normal(size=(3, 2)),
        v=rng.normal(size=2),
    )
    x = rng.normal(size=(4, 2))
    target = rng.normal(size=(4, 2))
    _, g, _ = S.ode_loss(p, x, target, 4)
    fd_check(lambda q: S.ode_loss(q, x, target, 4)[0], p, g, list(p))


def test_neural_cellular_gradient():
    rng = np.random.default_rng(5)
    p = dict(
        A=rng.normal(0, 0.1, (9, 3)),
        b=rng.normal(0, 0.1, 3),
        B=rng.normal(0, 0.1, (3, 2)),
        c=rng.normal(0, 0.1, 2),
    )
    state = rng.normal(0, 0.2, (4, 4, 2))
    seed = np.zeros((4, 4))
    seed[2, 2] = 1
    target = rng.random((4, 4))
    _, g = S.nca_loss(p, state, seed, target, 3)
    fd_check(lambda q: S.nca_loss(q, state, seed, target, 3)[0], p, g, list(p))


def test_checkpoint_gradients_independently():
    rng = np.random.default_rng(6)
    w = [rng.normal(0, 0.3, (3, 3)) for _ in range(5)]
    x, y = rng.normal(size=(2, 3)), rng.normal(size=(2, 3))
    p = {str(k): a for k, a in enumerate(w)}

    def objective(q):
        a = x
        for k in range(5):
            a = np.tanh(a @ q[str(k)])
        return 0.5 * np.mean((a - y) ** 2)

    for spacing in (1, 2, 3, 5):
        grads, _, _ = S.checkpoint_grad(w, x, y, spacing)
        fd_check(objective, p, {str(k): g for k, g in enumerate(grads)}, list(p))


@needs_mnist
def test_allreduce_matches_whole_batch():
    study = S.allreduce().validate()
    partial = study.panels[0].data["values"]
    gathered = study.panels[2].data["values"]
    expected = partial.mean(0)
    np.testing.assert_allclose(gathered, np.broadcast_to(expected, gathered.shape), atol=1e-14)


def test_sharding_counts():
    s = S.zero_sharding()
    memory = s.panels[2].data["values"]
    assert memory[0] == 112
    assert memory[-1] == 14
    assert np.all(np.diff(memory) < 0)


def test_hnsw_exhaustive_search():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(60, 4))
    graph = S.HNSW(x, m=5)
    for q in rng.normal(size=(12, 4)):
        result, work, path = graph.query(q, ef=60)
        assert result == np.sum((x - q) ** 2, axis=1).argmin()
        assert 1 <= work <= len(x)
        assert path


def test_mcts_finds_forced_win():
    s = S.mcts()
    p = s.panels[-1]
    assert p.data["labels"][np.argmax(p.data["values"])] == "cell 2"
    assert S.tictactoe_winner([1, 1, 1, -1, -1, 0, 0, 0, 0]) == 1


@needs_mnist
def test_hopfield_energy():
    s = S.hopfield()
    energy = s.panels[2].data["ys"][0]
    assert np.all(np.diff(energy) <= 1e-10)
    assert energy[-1] < energy[0]


@needs_mnist
def test_nca_training_reduces_loss():
    s = S.cellular()
    loss = s.panels[1].data["ys"][0]
    assert np.mean(loss[-30:]) < np.mean(loss[:30])
    before, damaged, _after5, after31 = s.panels[2].data["values"]
    assert np.max(np.abs(before - damaged)) > 0.1
    assert np.max(np.abs(after31 - damaged)) > 0.1


@needs_mnist
def test_batch_gradient_noise_decreases():
    s = S.batch_noise()
    noise = s.panels[0].data["ys"][0]
    assert noise[-1] < noise[0] / 80


@needs_gpt2
@needs_text
def test_retrieval_uses_normalized_probability():
    s = S.make("retrieval-augmented-generation")
    p = s.panels[2].data["values"]
    assert np.all((p > 0) & (p < 1))
    assert p[0] != p[1]


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(name, marks=data_marks(name))
        for name in (
            "newton-vs-gradient-descent",
            "data-parallel-allreduce",
            "tensor-and-pipeline-parallel",
            "zero-sharding",
            "activation-checkpointing",
            "roofline",
            "multi-armed-bandits",
            "policy-gradient",
            "actor-critic",
            "dqn-replay-buffer",
            "mcts-tic-tac-toe",
            "reward-shaping",
        )
    ],
)
def test_study_arrays(name):
    study = S.make(name).validate()
    assert len(study.panels) == 4
    for panel in study.panels:
        if panel.kind == "curve":
            assert panel.data["ys"].shape[1] == len(panel.data["x"])
            assert len(panel.data["labels"]) == panel.data["ys"].shape[0]


def test_adjoint_converges_to_discrete_gradient():
    rng = np.random.default_rng(9)
    p = dict(
        W=rng.normal(0, 0.2, (2, 3)), b=np.zeros(3), U=rng.normal(0, 0.2, (3, 2)), v=np.zeros(2)
    )
    x, target = rng.normal(size=(5, 2)), rng.normal(size=(5, 2))
    errors = []
    for steps in (8, 64):
        _, exact, states = S.ode_loss(p, x, target, steps)
        adjoint, start = S.ode_adjoint(p, states[-1], target, steps)
        errors.append(np.linalg.norm(S._pack(adjoint) - S._pack(exact)))
        assert np.linalg.norm(start - x) < 0.03
    assert errors[1] < errors[0] / 6


def test_joint_encoder_decoder_gradients():
    from dataclasses import replace

    from voxanim import tinylm as T
    from voxanim.next_attention import encoder_decoder_loss

    enc = T.Config(vocab=8, width=4, heads=1, kv_heads=1, ff=5, max_len=8, mask="bidirectional")
    dec = replace(enc, mask="causal", cross_attention=True)
    p = {"encoder." + k: v for k, v in T.init(enc, 2).items()}
    p.update({"decoder." + k: v for k, v in T.init(dec, 3).items()})
    source = np.array([[1, 2, 3], [3, 1, 4]])
    target = source[:, ::-1].copy()
    shifted = np.c_[np.zeros(2, int), target[:, :-1]]
    _, grad = encoder_decoder_loss(p, source, shifted, target, enc, dec)
    fd_check(
        lambda q: encoder_decoder_loss(q, source, shifted, target, enc, dec, False)[0],
        p,
        grad,
        list(p),
        tol=3e-4,
    )


@needs_gpt2
def test_cache_eviction_preserves_absolute_positions():
    from voxanim import gpt2_hooks

    ids = [10, 20, 30, 40]
    first = gpt2_hooks.forward(ids)
    cache = [(k[:, 2:], v[:, 2:]) for k, v in first["cache"]]
    implicit = gpt2_hooks.forward([50], cache=cache)["logits"]
    absolute = gpt2_hooks.forward([50], cache=cache, position_ids=[4])["logits"]
    assert not np.allclose(implicit, absolute)
    with pytest.raises(ValueError, match="position_ids"):
        gpt2_hooks.forward([50], cache=cache, position_ids=[1024])


@needs_text
def test_text_targets_are_actual_next_characters():
    from voxanim import realdata

    x, target, weights = realdata.language_data(64, 12, "train", 19)
    np.testing.assert_array_equal(x[:, 1:], target[:, :-1])
    assert np.all(weights == 1)
    assert (
        realdata.text_metadata("train")["character_range"][1]
        <= realdata.text_metadata("test")["character_range"][0]
    )


def test_depth_width_parameter_match():
    from dataclasses import replace

    from voxanim import tinylm as T
    from voxanim.next_attention import counts

    c = T.Config(width=16, heads=4, kv_heads=4, ff=32, max_len=48)
    sizes = [
        counts(T.init(replace(c, width=d, ff=f, layers=depth)))
        for d, f, depth in [(36, 67, 2), (28, 35, 4), (12, 88, 8)]
    ]
    assert sizes == [22680, 22680, 22680]


@pytest.mark.parametrize("kind", ["dense", "strided", "block", "global", "random"])
def test_sparse_mask_counts_and_causality(kind):
    from voxanim.next_attention import _pattern_counts, sparse_mask

    mask = sparse_mask(300, kind, 32)
    assert np.all(np.diag(mask))
    assert not np.triu(mask, 1).any()
    assert _pattern_counts(300, kind, 32) == mask.sum()

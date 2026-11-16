"""Numerical contracts for the new language and intervention studies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from conftest import fd_check, needs_gpt2, needs_mnist

from voxanim import next_language as n
from voxanim import toycorpus
from voxanim.cache import DEFAULT_CACHE

needs_text = pytest.mark.skipif(
    not (Path(DEFAULT_CACHE) / "tinyshakespeare.txt").exists(),
    reason="Pinned TinyShakespeare corpus is not downloaded",
)


def test_preference_gradient():
    rng = np.random.default_rng(3)
    P = {"w": rng.normal(size=4)}
    x = rng.normal(size=(8, 4))
    a = np.array([1, 4, 3, 1])
    b = np.array([0, 1, 6, 2])
    _, g = n.preference_loss(P, x, a, b)
    fd_check(lambda p: n.preference_loss(p, x, a, b)[0], P, g, ["w"])


def test_superposition_gradient():
    rng = np.random.default_rng(4)
    P = {"W": rng.normal(size=(2, 5)), "b": rng.normal(size=5)}
    x = rng.uniform(size=(8, 5))
    _, g = n.superposition_loss(P, x)
    fd_check(lambda p: n.superposition_loss(p, x)[0], P, g, list(P))


def test_sparse_autoencoder_gradient():
    rng = np.random.default_rng(5)
    P = {
        "E": rng.normal(size=(5, 8)),
        "b": rng.normal(size=8),
        "D": rng.normal(size=(8, 5)),
        "c": rng.normal(size=5),
    }
    x = rng.normal(size=(6, 5))
    _, g = n.sparse_loss(P, x, 0.13)
    fd_check(lambda p: n.sparse_loss(p, x, 0.13)[0], P, g, list(P))


def test_dpo_gradient():
    rng = np.random.default_rng(6)
    P = {"z": rng.normal(size=8)}
    ref = n.gpt2.softmax(rng.normal(size=8))
    a = np.array([1, 4, 3, 1])
    b = np.array([0, 1, 6, 2])
    _, g = n.dpo_loss(P["z"], ref, a, b)
    fd_check(lambda p: n.dpo_loss(p["z"], ref, a, b)[0], P, {"z": g}, ["z"])


def test_kl_regularized_policy_gradient():
    rng = np.random.default_rng(8)
    P = {"z": rng.normal(size=7)}
    ref = n.gpt2.softmax(rng.normal(size=7))
    reward = rng.normal(size=7)
    _, g = n.policy_grads(P["z"], ref, reward, 0.2)
    fd_check(lambda p: n.policy_grads(p["z"], ref, reward, 0.2)[0], P, {"z": g}, ["z"])


def test_speculative_distribution_identity():
    rng = np.random.default_rng(7)
    for _ in range(20):
        p, q = rng.dirichlet(np.ones(11), size=2)
        assert np.max(np.abs(n.speculative_distribution(p, q) - p)) < 1e-15


@needs_text
def test_bpe_roundtrip_on_collected_text():
    text = n.realdata.text("train")[:600]
    rules = toycorpus.bpe_train(text, 20)
    assert "".join(toycorpus.bpe_apply(text, rules)) == text
    assert len(toycorpus.bpe_apply(text, rules)) < len(text)
    assert rules == toycorpus.bpe_train(text, 20)


@needs_gpt2
def test_gpt2_hooks_match_reference_and_cached_logits():
    from voxanim import gpt2, gpt2_hooks

    text = "The cat sat on the"
    ids = gpt2.Tokenizer().encode(text)
    reference = gpt2.run(text)
    got = gpt2_hooks.forward(ids, record=True)
    assert np.allclose(got["logits"], reference.logits, atol=5e-5, rtol=1e-5)
    cache = None
    rows = []
    for token in ids:
        row = gpt2_hooks.forward([token], cache=cache)
        cache = row["cache"]
        rows.append(row["logits"][0])
    assert np.allclose(np.array(rows), got["logits"], atol=5e-5, rtol=1e-5)
    same = gpt2_hooks.forward(ids, patches={("resid", 4, 1): got["resid"][4, 1]})
    assert np.array_equal(same["logits"], got["logits"])
    before = gpt2.weights()["h.2.attn.c_proj.weight"].copy()
    gpt2_hooks.forward(ids, ablate={(2, 0): 0})
    assert np.array_equal(before, gpt2.weights()["h.2.attn.c_proj.weight"])


@needs_mnist
def test_input_attribution_and_feature_gradient():
    from voxanim import mnist

    rng = np.random.default_rng(4)
    params = mnist._init(rng, (784, 10))
    x = rng.uniform(0.2, 0.8, 784)
    ig = n.integrated_gradients(params, x, 3, 32)
    assert ig.sum() == pytest.approx(x @ params[0][0][:, 3], abs=1e-10)
    cnn = [[rng.normal(size=(3, 3, 4)), np.ones(4)], [np.zeros((676, 10)), np.zeros(10)]]
    score, g = n.cnn_feature_grad(cnn, x, 2)
    for ij in ((14, 14), (4, 8)):
        plus = x.reshape(28, 28).copy()
        minus = plus.copy()
        plus[ij] += 1e-6
        minus[ij] -= 1e-6
        fd = (n.cnn_feature_grad(cnn, plus, 2)[0] - n.cnn_feature_grad(cnn, minus, 2)[0]) / 2e-6
        assert g[ij] == pytest.approx(fd, abs=1e-7)
    assert np.isfinite(score)


@pytest.mark.parametrize(
    "name",
    [
        "reward-model",
        "rlhf-ppo",
        "dpo",
        "reward-hacking",
        "superposition",
        "teacher-forcing",
        "packing-and-padding",
    ],
)
@needs_text
@needs_mnist
def test_cached_studies_are_measured_and_finite(name):
    study = n.make(name).validate()
    assert len(study.panels) == 4
    assert study.metadata["seed"] == n.SEED
    if name == "reward-model":
        loss = study.panels[1].data["ys"][0]
        assert loss[-1] < loss[0]
    if name == "teacher-forcing":
        loss = study.panels[2].data["ys"][0]
        assert loss[-20:].mean() < loss[:20].mean()


@pytest.mark.parametrize("method", ["soft prompt", "KV prefix", "LoRA", "full tuning"])
def test_adaptation_gradients(method):
    from voxanim import tinylm

    rng = np.random.default_rng(17)
    config = tinylm.Config(width=8, heads=2, kv_heads=2, ff=12)
    params = tinylm.init(config, 17)
    extras = {
        "soft prompt": {"prompt": rng.normal(size=(1, 2, 8))},
        "KV prefix": {
            "prefix_k": rng.normal(size=(1, 2, 2, 4)),
            "prefix_v": rng.normal(size=(1, 2, 2, 4)),
        },
        "LoRA": {"A": rng.normal(size=(8, 2)), "B": rng.normal(size=(2, 8))},
        "full tuning": {},
    }[method]
    x = rng.integers(8, size=(2, 4))
    y = rng.integers(8, size=(2, 4))
    _, gp, ga = n.adaptation_loss(params, extras, x, y, config, method, True)
    if extras:
        fd_check(
            lambda p: n.adaptation_loss(params, p, x, y, config, method), extras, ga, list(extras)
        )
    else:
        fd_check(
            lambda p: n.adaptation_loss(p, extras, x, y, config, method), params, gp, list(params)
        )


def test_covariance_weighted_rank_one_edit():
    rng = np.random.default_rng(42)
    key = rng.normal(size=12)
    delta = rng.normal(size=7)
    background = rng.normal(size=(30, 12))
    edit = n.rank_one_edit(key, delta, background, ridge=0.2)
    assert np.allclose(key @ edit, delta, atol=1e-12)
    assert np.linalg.matrix_rank(edit, tol=1e-10) == 1
    cov = background.T @ background / len(background) + 0.2 * np.eye(12)
    direct = np.linalg.solve(cov, key)
    assert np.allclose(edit, np.outer(direct, delta) / (key @ direct), atol=1e-12)


@needs_text
def test_reward_pressure_records_outcome_without_requiring_hacking():
    study = n.make("reward-hacking")
    proxy, true = study.panels[1].data["ys"]
    assert np.all(np.diff(proxy) > -1e-10)
    assert np.isfinite(true).all()
    assert "TinyShakespeare" in study.metadata["source"]
    assert study.metadata["weights_artifacts"]


@pytest.mark.slow
@needs_text
@needs_mnist
def test_mnist_long_run_reports_measured_generalization_without_promised_transition():
    study = n.make("grokking")
    steps = study.panels[1].data["x"]
    train, test = study.panels[1].data["ys"]
    assert steps[0] == 0 and steps[-1] == 3000
    assert np.all(np.diff(steps) > 0)
    assert np.all((train >= 0) & (train <= 1))
    assert np.all((test >= 0) & (test <= 1))
    assert study.metadata["dataset"] == "MNIST"
    assert study.metadata["dataset_provenance"]["test"]["official_partition"] == "test"


@needs_text
def test_document_packing_resets_match_individual():
    study = n.make("packing-and-padding")
    individual, leaked, reset = study.panels[3].data["values"]
    assert reset == pytest.approx(individual, abs=1e-12)
    assert np.isfinite(leaked)


@needs_text
def test_learning_windows_are_unmodified_collected_text_in_disjoint_splits():
    for split in ("train", "test"):
        windows = n._text_windows(8, 16, split, 31)
        source = n.realdata.text(split)
        assert all(n.realdata.decode(row) in source for row in windows)
    train = n.realdata.text_metadata("train")["character_range"]
    test = n.realdata.text_metadata("test")["character_range"]
    assert train[1] <= test[0]


@needs_text
def test_automatic_preferences_use_observed_corpus_words_and_save_weights():
    preferences = n._preferences()
    source = n.realdata.text("train").lower()
    assert all(word in source for word in preferences["words"])
    assert preferences["observed_train"] > 0
    assert preferences["observed_test"] > 0
    artifact = Path(preferences["artifact"]["artifact"])
    with np.load(artifact) as saved:
        assert "parameter_w" in saved
        assert "history_train_winner" in saved


def test_neural_dpo_gradient():
    from voxanim import tinylm

    rng = np.random.default_rng(19)
    cfg = tinylm.Config(vocab=8, width=8, heads=2, kv_heads=2, ff=12)
    params = tinylm.init(cfg, 19)
    prompt = np.array([[0, 2]])
    reference = n.gpt2.softmax(rng.normal(size=8))
    a = np.array([1, 4, 3, 1])
    b = np.array([0, 1, 6, 2])
    offset = np.zeros(8)
    _, g = n.neural_dpo_loss(params, prompt, cfg, reference, a, b, offset)
    fd_check(
        lambda p: n.neural_dpo_loss(p, prompt, cfg, reference, a, b, offset)[0],
        params,
        g,
        list(params),
    )

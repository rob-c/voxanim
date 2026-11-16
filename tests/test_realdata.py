"""Evidence checks for collected-data training and its review artifacts."""

import json

import numpy as np
import pytest
from conftest import needs_mnist, needs_text

from voxanim import realdata as R
from voxanim.cache import sha256_of


def test_array_identity_includes_shape_and_dtype():
    a = np.arange(12, dtype=np.int64)
    assert R.array_hash(a.reshape(2, 6)) != R.array_hash(a.reshape(3, 4))
    assert R.array_hash(a) != R.array_hash(a.view(np.float64))


@needs_text
def test_text_splits_are_contiguous_disjoint_source_ranges():
    train, validation, test = [R.text_metadata(s) for s in ("train", "validation", "test")]
    assert train["character_range"][0] == 0
    assert train["character_range"][1] == validation["character_range"][0]
    assert validation["character_range"][1] == test["character_range"][0]
    assert train["raw_sha256"] == validation["raw_sha256"] == test["raw_sha256"]
    assert all(len(R.text(s)) > 10000 for s in ("train", "validation", "test"))
    for split in ("train", "test"):
        for row in R.text_sequences(4, 30, split, 13):
            assert R.decode(row) in R.text(split)


@needs_mnist
def test_mnist_receipt_identifies_exact_data_and_nonoverlapping_splits():
    x, y, train = R.mnist_split(64, "train", 7, 31)
    xv, yv, val = R.mnist_split(64, "validation", 7, 31)
    assert not set(train["example_indices"]) & set(val["example_indices"])
    assert train["x_sha256"] == R.array_hash(x)
    assert train["y_sha256"] == R.array_hash(y)
    assert val["x_sha256"] == R.array_hash(xv)
    assert val["y_sha256"] == R.array_hash(yv)
    assert x.shape == (64, 49) and np.all((x >= 0) & (x <= 1))


@needs_mnist
@pytest.mark.parametrize("newton", [False, True])
def test_ridge_checkpoint_reproduces_the_plotted_final_test_result(newton):
    from voxanim.next_systems_real import digits, precondition

    study = precondition(newton)
    receipt = study.metadata["training_provenance"]
    assert sha256_of(receipt["artifact"]) == receipt["artifact_sha256"]
    with np.load(receipt["artifact"], allow_pickle=False) as arrays:
        weights = arrays["parameter_weights"]
        expected = arrays["history_accuracy"][:, -1]
        losses = arrays["history_loss"]
    x, y, xt, yt, _ = digits()
    xt = np.c_[xt, np.ones(len(xt))]
    actual = [np.mean((xt @ w).argmax(1) == yt) for w in weights]
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, study.panels[-1].data["values"])
    # A disposable parameter dictionary used to leave the Adam weights at zero
    # while recording plausible, but completely unchanged, evaluation arrays.
    assert all(np.any(w != 0) for w in weights)
    if newton:
        assert np.any(losses[1, 1:] != losses[1, 0])
    x = np.c_[x, np.ones(len(x))]
    target = np.eye(10)[y]
    ridge = receipt["settings"]["ridge"]
    objectives = [
        0.5 * np.mean(np.sum((x @ w - target) ** 2, axis=1)) + 0.5 * ridge * np.sum(w * w)
        for w in weights
    ]
    np.testing.assert_allclose(objectives, losses[:, -1], rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(losses, study.panels[1].data["ys"])
    assert receipt["settings"]["updates"] == 80


@needs_mnist
def test_bandit_arms_are_updated_and_replay_observed_validation_rewards():
    from voxanim.next_systems_real import bandits, digits

    study = bandits()
    receipt = study.metadata["training_provenance"]
    assert sha256_of(receipt["artifact"]) == receipt["artifact_sha256"]
    _, _, xt, yt, _ = digits()
    validation = receipt["dataset"]["validation"]
    xv, yv, _ = R.mnist_split(1000, "validation", seed=validation["seed"])
    validation_rates, test_rates = [], []
    with np.load(receipt["artifact"], allow_pickle=False) as arrays:
        expected_rewards = arrays["history_reward_table"]
        for arm in range(len(expected_rewards)):
            weights = arrays["parameter_arm%d" % arm]
            features = arrays["history_feature_indices%d" % arm]
            assert np.any(weights != 0), "Every classifier must leave its zero initialization"
            observed = (np.c_[xv[:, features], np.ones(len(xv))] @ weights).argmax(1) == yv
            np.testing.assert_array_equal(observed, expected_rewards[arm])
            validation_rates.append(observed.mean())
            test_rates.append(
                np.mean((np.c_[xt[:, features], np.ones(len(xt))] @ weights).argmax(1) == yt)
            )
        np.testing.assert_array_equal(validation_rates, arrays["history_means"])
        np.testing.assert_array_equal(test_rates, arrays["history_test"])
    np.testing.assert_array_equal(validation_rates, study.panels[0].data["values"])
    np.testing.assert_array_equal(test_rates, study.panels[-1].data["values"])


def test_evidence_receipt_checksums_actual_saved_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "DEFAULT_CACHE", str(tmp_path))
    receipt = R.save_evidence(
        "fixture",
        {"w": np.array([1.0, 2.0])},
        {"loss": np.array([3.0, 2.0])},
        {"dataset": "unit-test fixture"},
        {"updates": 2},
    )
    assert sha256_of(receipt["artifact"]) == receipt["artifact_sha256"]
    saved = json.loads(next((tmp_path / "run-evidence").glob("*.json")).read_text())
    assert saved["run_id"] == receipt["run_id"]
    with np.load(receipt["artifact"], allow_pickle=False) as arrays:
        assert R.array_hash(arrays["parameter_w"]) == saved["arrays"]["parameter_w"]["sha256"]

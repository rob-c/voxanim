"""Numerical contracts for the new NumPy vision and generative models."""

import json

import numpy as np
import pytest
from conftest import needs_mnist

from voxanim import next_vision as studies
from voxanim import next_vision_models as vision
from voxanim import toy2d


def finite_difference(fn, p, g, keys):
    for key in keys:
        idx = np.unravel_index(np.argmax(np.abs(g[key])), g[key].shape)
        plus, minus = {k: v.copy() for k, v in p.items()}, {k: v.copy() for k, v in p.items()}
        plus[key][idx] += 1e-5
        minus[key][idx] -= 1e-5
        measured = (fn(plus) - fn(minus)) / 2e-5
        assert measured == pytest.approx(g[key][idx], rel=3e-4, abs=1e-7), key


def test_mlp_and_flow_gradients():
    rng = np.random.default_rng(71)
    x, y = rng.normal(size=(8, 2)), rng.normal(size=(8, 2))
    p = toy2d.init((2, 6, 2), rng)
    _, g = toy2d.loss_grads(p, x, y)
    finite_difference(lambda q: toy2d.loss_grads(q, x, y)[0], p, g, list(p))
    p = toy2d.flow_init(rng, layers=3, hidden=6)
    _, g = toy2d.flow_loss_grads(p, x)
    finite_difference(lambda q: toy2d.flow_loss_grads(q, x)[0], p, g, list(p))
    z, _, _ = toy2d.flow_forward(p, x)
    assert np.allclose(toy2d.flow_inverse(p, z), x, atol=1e-12)


def test_infonce_gradients():
    rng = np.random.default_rng(72)
    p = dict(I=rng.normal(size=(5, 3)), T=rng.normal(size=(4, 3)))
    x, y = rng.normal(size=(7, 5)), np.arange(7) % 4
    _, g, _ = vision.info_nce(p, x, y)
    finite_difference(lambda q: vision.info_nce(q, x, y)[0], p, g, list(p))


def test_pixelcnn_mask_and_gradients():
    rng = np.random.default_rng(73)
    p = dict(
        W=rng.normal(0, 0.1, (3, 3, 1, 4)),
        b=np.zeros(4),
        O=rng.normal(0, 0.1, (4, 1)),
        c=np.zeros(1),
    )
    x = rng.random((2, 6, 6, 1))
    _, g, prediction = vision.masked_conv_loss(p, x)
    finite_difference(lambda q: vision.masked_conv_loss(q, x)[0], p, g, list(p))
    changed = x.copy()
    changed[:, 3:, 3:] = 1 - changed[:, 3:, 3:]
    after = vision.masked_conv_loss(p, changed)[2]
    assert np.array_equal(prediction[:, 3, 3], after[:, 3, 3])
    assert np.all(g["W"][1, 1:] == 0) and np.all(g["W"][2] == 0)


def test_unet_gradients():
    rng = np.random.default_rng(74)
    p = vision.unet_init(rng)
    x, y = rng.random((2, 6, 6, 1)), rng.integers(0, 3, (2, 6, 6))
    _, g, _ = vision.unet_loss(p, x, y)
    finite_difference(lambda q: vision.unet_loss(q, x, y)[0], p, g, list(p))


def test_small_vit_gradients():
    rng = np.random.default_rng(75)
    p = vision.small_vit_init(rng, 2)
    x, y = rng.random((2, 784)), np.array([1, 7])
    _, g, _ = vision.small_vit_loss(p, x, y)
    finite_difference(
        lambda q: vision.small_vit_loss(q, x, y, False)[0],
        p,
        g,
        ["E", "cls", "pos", "U", "Q0", "K0", "V0", "O0", "A0", "B0", "Q1"],
    )


def test_em_monotonic_and_svm_separates_rings():
    x, y = toy2d.dataset("rings", n=120, seed=7)
    model = toy2d.svm(x, y, c=10.0, passes=12)
    assert np.mean((toy2d.svm_predict(model, x) > 0) == y) > 0.95
    x, _ = toy2d.dataset("gaussians", n=240)
    run = toy2d.gmm_em(x, k=8)
    assert np.min(np.diff(run["loss"])) > -1e-7
    assert np.allclose(run["responsibility"].sum(1), 1)


def test_nms_and_iou():
    boxes = np.array([[0, 0, 2, 2], [0, 0, 2, 2], [4, 4, 6, 6]])
    assert np.array_equal(studies.iou(boxes[0], boxes), [1, 1, 0])
    assert np.array_equal(studies.nms(boxes, np.array([0.9, 0.8, 0.5])), [0, 2])


def test_tsne_probabilities_and_objective():
    x, _ = toy2d.dataset("moons", n=50)
    p = studies.tsne_affinity(x, 8)
    assert np.allclose(p, p.T) and np.isclose(p.sum(), 1)
    assert np.all(np.diag(p) == 0)
    z, losses = studies.tsne_fit(x, 8, steps=150)
    assert np.isfinite(z).all() and losses[-1] < losses[0]


@needs_mnist
def test_empirical_bias_variance_decomposition():
    s = studies.bias_variance().validate()
    error = s.panels[-1].data["values"]
    assert np.max(error) < 1e-10


def test_autoregressive_vae_gradients_and_causality():
    rng = np.random.default_rng(76)
    p = dict(
        E=rng.normal(0, 0.1, (6, 2)),
        eb=np.zeros(2),
        L=rng.normal(0, 0.01, (6, 2)),
        lb=np.zeros(2),
        D=rng.normal(0, 0.1, (2, 6)),
        db=np.zeros(6),
        AR=rng.normal(0, 0.1, (6, 6)),
    )
    x, noise = rng.random((3, 6)), rng.normal(size=(3, 2))
    _, g, _ = vision.autoregressive_vae_loss(p, x, noise, beta=0.4)
    finite_difference(
        lambda q: vision.autoregressive_vae_loss(q, x, noise, beta=0.4)[0], p, g, list(p)
    )
    assert np.all(np.tril(g["AR"]) == 0)


@needs_mnist
def test_trained_visual_claims():
    """Measured results are finite and replayable; no method is required to win."""
    for name in (
        "clip-contrastive",
        "object-detection-basics",
        "semantic-segmentation",
        "flow-matching",
        "vae-posterior-collapse",
        "vq-vae-codebook",
    ):
        study = studies.make(name).validate()
        assert study.metadata["dataset"] == "MNIST"
        evidence = study.metadata["run_evidence"]
        with np.load(evidence["artifact"], allow_pickle=False) as artifact:
            assert any(key.startswith("parameter_") for key in artifact.files)
        assert study.metadata["run_settings"]["source_sha256"]


def test_minibatch_discriminator_gradient():
    rng = np.random.default_rng(77)
    x = rng.normal(size=(8, 2))
    augmented, tape = vision.minibatch_features(x)
    weights = rng.normal(size=augmented.shape)
    analytic = vision.minibatch_backward(weights, tape)
    for idx in [(0, 0), (5, 1), (7, 0)]:
        plus, minus = x.copy(), x.copy()
        plus[idx] += 1e-6
        minus[idx] -= 1e-6
        fd = (
            np.sum(
                (vision.minibatch_features(plus)[0] - vision.minibatch_features(minus)[0]) * weights
            )
            / 2e-6
        )
        assert fd == pytest.approx(analytic[idx], rel=1e-5, abs=1e-7)


def test_vq_decoder_gradient_and_straight_through():
    rng = np.random.default_rng(78)
    p = dict(
        E=rng.normal(0, 0.1, (6, 2)),
        eb=np.zeros(2),
        C=rng.normal(0, 0.2, (4, 2)),
        D=rng.normal(0, 0.1, (2, 6)),
        db=np.zeros(6),
    )
    x = rng.random((5, 6))
    _, g, (h, z, ids, recon) = vision.vq_loss_grads(p, x)
    finite_difference(lambda q: vision.vq_loss_grads(q, x)[0], p, g, ["D", "db"])
    assert np.array_equal(z, p["C"][ids])
    # The non-differentiable argmin uses the specified STE, not a false numerical gradient.
    incoming = ((recon - x) / x.size) @ p["D"].T
    incoming += 0.5 * (h - z) / h.size
    expected = x.T @ (incoming * (1 - h**2))
    assert np.allclose(g["E"], expected)


def test_cnn_pixel_gradients():
    from voxanim import mnist

    rng = np.random.default_rng(79)
    p = [
        [rng.normal(0, 0.1, (3, 3, 4)), rng.normal(0.5, 0.1, 4)],
        [rng.normal(0, 0.1, (676, 10)), rng.normal(0, 0.1, 10)],
    ]
    x = rng.random((1, 784))
    upstream = rng.normal(size=(1, 10))
    analytic = vision.cnn_input_grad(p, x, upstream)
    gram = np.eye(4) * 0.2
    _, feature, _ = vision.cnn_feature_grad(p, x, 1, target_gram=gram)
    for idx in (100, 390, 630):
        plus, minus = x.copy(), x.copy()
        plus[0, idx] += 1e-6
        minus[0, idx] -= 1e-6
        f1 = np.sum(mnist._cnn_forward(p, plus)["z2"] * upstream)
        f0 = np.sum(mnist._cnn_forward(p, minus)["z2"] * upstream)
        assert (f1 - f0) / 2e-6 == pytest.approx(analytic[0, idx], abs=1e-7)
        f1 = vision.cnn_feature_grad(p, plus, 1, target_gram=gram)[0]
        f0 = vision.cnn_feature_grad(p, minus, 1, target_gram=gram)[0]
        assert (f1 - f0) / 2e-6 == pytest.approx(feature[0, idx], abs=1e-7)


def test_receptive_field_stride_dilation():
    assert np.array_equal(
        studies.receptive_field_sizes([3, 2, 3], [1, 2, 1], [1, 1, 1]), [1, 3, 4, 8]
    )
    assert np.array_equal(studies.receptive_field_sizes([3, 3], [1, 1], [2, 2]), [1, 5, 9])


def test_linear_svm_gradient_and_nonlinear_lift():
    rng = np.random.default_rng(82)
    x, y = toy2d.dataset("rings", n=120, seed=82)
    p = dict(w=rng.normal(size=2), b=np.array(0.25))
    _, g = toy2d.linear_svm_loss(p, x, 2 * y - 1, c=3.0)
    finite_difference(lambda q: toy2d.linear_svm_loss(q, x, 2 * y - 1, c=3.0)[0], p, g, list(p))
    linear = toy2d.linear_svm(x, y, c=10.0)
    lifted_x = np.column_stack((x, np.sum(x**2, axis=1)))
    lifted = toy2d.linear_svm(lifted_x, y, c=10.0)
    linear_acc = np.mean((x @ linear["w"] + linear["b"] > 0) == y)
    lifted_acc = np.mean((lifted_x @ lifted["w"] + lifted["b"] > 0) == y)
    assert lifted_acc > 0.95 and lifted_acc > linear_acc + 0.25


@needs_mnist
def test_cross_validation_selection_optimism_is_measured():
    study = studies.cross_validation()
    scores = study.panels[2].data["values"]
    outer = study.panels[3].data["values"]
    assert scores.shape == (6, 5) and len(outer) == 5
    assert scores.max(axis=0).mean() >= outer.mean()
    assert any("optimism gap" in note for note in study.panels[3].notes)


@needs_mnist
def test_boosting_bagging_and_shrinkage_are_real_fits():
    study = studies.gradient_boosting()
    curves = study.panels[2].data["ys"]
    assert curves.shape == (2, 60)
    assert np.isfinite(curves).all()
    values = study.panels[3].data["values"]
    assert len(values) == 3 and np.isfinite(values).all()
    with np.load(study.metadata["run_evidence"]["artifact"], allow_pickle=False) as artifact:
        trees = json.loads(str(artifact["history_trees"]))
        # JSON preserves the fitted split values; replay the actual final predictions.
        x, y = artifact["history_test_x"], artifact["history_test_y"]
        prediction = np.full(len(y), float(artifact["history_initial_prediction"]))
        for tree in trees:
            prediction += 0.12 * toy2d.tree_predict(tree, x)
        assert values[0] == pytest.approx(np.mean((prediction - y) ** 2))


@needs_mnist
def test_gan_comparison_counts_and_density():
    study = studies.gan_training()
    assert study.panels[0].data["xy"].shape == (240, 2)
    counts = study.panels[3].data["values"]
    assert counts.shape == (2, 10) and np.all(counts.sum(1) == 240)
    with np.load(study.metadata["run_evidence"]["artifact"], allow_pickle=False) as artifact:
        assert artifact["history_loss"].shape == (2, 1800)
        assert np.array_equal(np.unique(artifact["history_data_train_labels"]), np.arange(10))
        generator = {
            k[len("parameter_plain_generator_") :]: artifact[k]
            for k in artifact.files
            if k.startswith("parameter_plain_generator_")
        }
        assert generator


@needs_mnist
def test_adversarial_training_tradeoff_is_measured():
    study = studies.adversarial_examples()
    before_clean, after_clean, before_attack, after_attack = study.panels[3].data["values"]
    assert np.all(
        (np.array([before_clean, after_clean, before_attack, after_attack]) >= 0)
        & (np.array([before_clean, after_clean, before_attack, after_attack]) <= 1)
    )


def test_training_only_projection_does_not_leak_test_pixels(monkeypatch):
    rng = np.random.default_rng(71)
    x, y = rng.random((20, 784)), np.arange(20) % 10
    xt, yt = rng.random((10, 784)), np.arange(10)
    monkeypatch.setattr(studies, "mnist_data", lambda: (x, y, xt, yt))
    before = studies.mnist_projection(20, 10)
    monkeypatch.setattr(studies, "mnist_data", lambda: (x, y, xt * 0.1, yt))
    after = studies.mnist_projection(20, 10)
    assert np.array_equal(before["pca_components"], after["pca_components"])
    assert np.array_equal(before["train"], after["train"])
    assert not np.array_equal(before["test"], after["test"])


def test_clip_declared_steps_and_final_snapshot():
    rng = np.random.default_rng(53)
    x, y = rng.random((20, 4)), np.arange(20) % 10
    p, loss, snapshots = vision.fit_clip(x, y, steps=4)
    assert len(loss) == 4
    actual = vision.info_nce(p, x[:10], y[:10])[2]
    assert np.allclose(snapshots[-1], actual)


@needs_mnist
def test_saved_flow_replays_heldout_likelihood():
    study = studies.normalizing_flows()
    with np.load(study.metadata["run_evidence"]["artifact"], allow_pickle=False) as artifact:
        parameters = {
            k[len("parameter_weights_") :]: artifact[k]
            for k in artifact.files
            if k.startswith("parameter_weights_")
        }
        heldout = artifact["history_data_test"]
        measured = toy2d.flow_loss_grads(parameters, heldout)[0]
        assert measured == pytest.approx(study.panels[3].data["values"][1])
        raw, _, raw_test, _ = studies.mnist_data()
        ids = artifact["history_data_train_indices"]
        test_ids = artifact["history_data_test_indices"]
        mean = artifact["history_data_pca_mean"]
        axes = artifact["history_data_pca_components"]
        scale = artifact["history_data_pca_scale"]
        assert np.allclose(mean, vision.downsample(raw[ids]).mean(0))
        assert np.allclose(heldout, (vision.downsample(raw_test[test_ids]) - mean) @ axes.T / scale)

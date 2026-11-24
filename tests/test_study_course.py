"""The course's classifier (voxanim.study_course), the numpy twin of examples/mnist_blocks.py:
its gradients against finite differences, PyTorch's initialisation, the optimizers' update
rules, the script's settings, and the recorded runs (each claim relative: one run against
another, or a bound the arithmetic gives)."""

from __future__ import annotations

import re

import numpy as np
import pytest
from conftest import fd_check, needs_mnist

from voxanim import blocks
from voxanim import study_course as S


def small(seed=0, n=6, dtype=np.float64):
    """A few made-up digits and labels, and the initial weights, in ``dtype``."""
    rng = np.random.default_rng(seed)
    P = {k: v.astype(dtype) for k, v in S.init_params().items()}
    x = rng.normal(0.0, 1.0, (n, 1, 28, 28)).astype(dtype)
    y = rng.integers(0, 10, n)
    return P, x, y


# ---------------------------------------------------------------------- no data needed
def test_the_model_is_kept_as_pytorch_keeps_it():
    P = S.init_params()
    assert S.KEYS == ("1.weight", "1.bias", "3.weight", "3.bias", "5.weight", "5.bias")
    assert [P[k].shape for k in S.KEYS] == [(64, 784), (64,), (32, 64), (32,), (10, 32), (10,)]
    assert all(v.dtype == np.float32 for v in P.values())
    assert S.count() == 52_650 == 784 * 64 + 64 + 64 * 32 + 32 + 32 * 10 + 10


def test_initialisation_is_pytorchs_uniform_range():
    P = S.init_params()
    for i, (a, _b) in zip(S.LINEAR, ((784, 64), (64, 32), (32, 10))):
        bound = 1 / np.sqrt(a)
        for k in ("%d.weight" % i, "%d.bias" % i):
            v = P[k]
            assert np.abs(v).max() <= bound
            if v.size >= 64:                       # a uniform spread fills its range
                assert np.abs(v).max() > 0.9 * bound
                assert abs(v.mean()) < 0.2 * bound
                assert abs(v.std() - bound / np.sqrt(3)) < 0.15 * bound
    np.testing.assert_array_equal(P["1.weight"], S.init_params()["1.weight"])   # seeded


def test_gradients_match_finite_differences():
    P, x, y = small()

    def loss(Q):
        return S.loss_of(Q, x, y)
    _l, G = S.loss_and_grads(P, x, y)
    fd_check(loss, P, G, S.KEYS)


def test_gradients_with_dropout_and_with_a_double_softmax():
    P, x, y = small(1)
    masks = [m.astype(np.float64) for m in S._masks(3, len(y))]
    _l, G = S.loss_and_grads(P, x, y, masks=masks)
    fd_check(lambda Q: S.loss_of(Q, x, y, masks=masks), P, G, S.KEYS)
    _l, G = S.loss_and_grads(P, x, y, softmax=True)
    fd_check(lambda Q: S.loss_of(Q, x, y, softmax=True), P, G, S.KEYS)


def test_dropout_keeps_four_in_five_and_scales_them():
    m = S._masks(0, 1000)
    for mask in m:
        assert set(np.unique(mask)) <= {0.0, np.float32(1 / 0.8)}
        assert abs((mask > 0).mean() - 0.8) < 0.02


def test_the_loss_of_a_uniform_guess_is_ln_10_and_its_identities():
    z = np.zeros((5, 10))
    np.testing.assert_allclose(S.loss_each(z, np.arange(5)), np.log(10))
    z = np.random.default_rng(2).normal(size=(4, 10))
    np.testing.assert_allclose(np.exp(S.log_softmax(z)).sum(1), 1.0)
    np.testing.assert_allclose(S.log_softmax(z + 7.0), S.log_softmax(z))     # shift-free
    # a softmax in front of the loss: the loss can never go below log(1 + 9 / e)
    p = np.eye(10)[:3] * 1.0
    assert np.allclose(S.loss_each(p, np.arange(3)), np.log(1 + 9 / np.e))


def test_the_optimizers_update_rules():
    P, x, y = small(dtype=np.float32)
    _l, G = S.loss_and_grads(P, x, y)
    new = S.sgd_update(P, G, 0.1)
    for k in S.KEYS:
        np.testing.assert_allclose(new[k], P[k] - 0.1 * G[k], atol=1e-7)
    new, st = S.adam_update(P, G, S.adam_start(P))
    assert st["t"] == 1
    for k in S.KEYS:
        np.testing.assert_allclose(st["m"][k], 0.1 * G[k], rtol=1e-5, atol=1e-12)
        np.testing.assert_allclose(st["v"][k], 0.001 * G[k] ** 2, rtol=1e-5, atol=1e-16)
        step = np.abs(new[k] - P[k])
        # Adam's first step: lr times g / (|g| + eps): at most lr, about lr where |g| >> eps
        assert step.max() <= S.ADAM_LR * 1.0001
        big = np.abs(G[k]) > 1e-5
        np.testing.assert_allclose(step[big], S.ADAM_LR, rtol=1e-2)


def test_an_epoch_is_938_steps_and_the_order_is_seeded():
    assert S.EPOCH_STEPS == 938 and S.LAST_BATCH == 32
    assert len(S.batch_indices(0)) == 64 and len(S.batch_indices(937)) == 32
    o = S.order(0)
    np.testing.assert_array_equal(np.sort(o), np.arange(60_000))
    np.testing.assert_array_equal(o, S.order(0))
    assert not np.array_equal(o[:64], S.order(1)[:64])           # a new order every epoch


def test_the_twin_has_the_scripts_settings():
    """What the twin trains with is what the script says (read from its regions)."""
    data = " ".join(blocks.code_lines("data"))
    m = re.search(r"Normalize\(\(([\d.]+),\), \(([\d.]+),\)\)", data)
    assert (float(m.group(1)), float(m.group(2))) == (S.MEAN, S.STD)
    loader = " ".join(blocks.code_lines("loader"))
    assert "batch_size=%d" % S.BATCH in loader and "shuffle=True" in loader
    opt = " ".join(blocks.code_lines("optimizer"))
    assert "lr=%g" % S.LR in opt and "Adam(model.parameters())" in opt   # Adam's defaults
    assert blocks.code_lines("train")[0] == "for epoch in range(1):"
    assert blocks.code_lines("loss") == ("loss_fn = nn.CrossEntropyLoss()",)
    model = blocks.code_lines("model")
    sizes = [tuple(int(v) for v in re.findall(r"\d+", s)) for s in model if "Linear" in s]
    assert sizes == list(zip(S.SIZES, S.SIZES[1:]))
    parts = " ".join(blocks.code_lines("dropout-model"))
    assert parts.count("nn.Dropout(%g)" % S.DROPOUT) == 2


# ---------------------------------------------------------------------- the data and the runs
@needs_mnist
def test_the_normalisation_is_the_training_pixels_mean_and_spread():
    st = S.pixel_stats()
    assert round(st["mean"], 4) == S.MEAN and round(st["std"], 4) == S.STD
    x = S.data()[0]
    assert x.dtype == np.float32 and x.shape == (60_000, 784)
    assert abs(float(x.mean())) < 1e-3 and abs(float(x.std()) - 1.0) < 1e-3
    assert np.isclose(x.min(), st["low"]) and np.isclose(x.max(), st["high"])
    one, label = S.digit(0)
    assert one.shape == (1, 28, 28) and label == 5


@needs_mnist
def test_the_batches_are_the_shuffled_training_digits():
    b = S.batch(3)
    x, y = S.data()[:2]
    np.testing.assert_array_equal(b["y"], y[S.batch_indices(3)])
    np.testing.assert_array_equal(b["x"].reshape(64, -1), x[S.batch_indices(3)])
    assert b["x"].shape == (64, 1, 28, 28)
    f = S.first_batch()
    assert f["x"].shape == (64, 1, 28, 28) and f["pixels"].max() <= 1.0


@needs_mnist
def test_the_snapshots_are_consistent_with_each_other():
    for s in (0, 1, 10, 500):
        b = S.batch(s)
        loss, G = S.loss_and_grads(S.params(s), b["x"], b["y"])
        assert abs(loss - S.losses()[s]) < 1e-6
        for k in S.KEYS:
            np.testing.assert_array_equal(G[k], S.grads(s)[k])
    nxt = S.sgd_update(S.params(0), S.grads(0))
    for k in S.KEYS:
        np.testing.assert_array_equal(nxt[k], S.params(1)[k])
        assert S.grads(0)[k].shape == S.params(0)[k].shape
    # the same start and batch for every run: Adam's first gradient is SGD's
    for k in S.KEYS:
        np.testing.assert_array_equal(S.grads(0, "adam")[k], S.grads(0)[k])
    st = S.adam_state(1)
    for k in S.KEYS:
        np.testing.assert_allclose(st["m"][k], 0.1 * S.grads(0)[k], rtol=1e-5, atol=1e-12)
    with pytest.raises(KeyError):
        S.params(7)


@needs_mnist
def test_untrained_the_loss_is_about_ln_10_and_training_brings_it_down():
    l, h = S.losses(), S.heldout()
    assert abs(l[0] - np.log(10)) < 0.1
    assert abs(h["loss"][0] - np.log(10)) < 0.1 and h["accuracy"][0] < 0.2
    assert l[-100:].mean() < 0.15 * l[0]
    assert h["accuracy"][-1] > 0.9 and h["steps"][-1] == S.EPOCH_STEPS
    assert len(l) == S.EPOCH_STEPS and np.isfinite(l).all()
    a = S.activations(0)
    assert a["x"].shape == (64, 784) and a["h1"].shape == (64, 64) and a["logits"].shape == (
        64, 10)
    assert (a["h1"] >= 0).all() and (a["h1"] == 0).any()
    np.testing.assert_allclose(a["probs"].sum(1), 1.0, rtol=1e-5)
    assert abs(a["loss_each"].mean() - l[0]) < 1e-5


@needs_mnist
def test_the_three_learning_rates_crawl_learn_and_bounce():
    small, good, large = (S.losses(r) for r in ("lr_small", "sgd", "lr_large"))
    acc = {r: S.heldout(r)["accuracy"][-1] for r in ("lr_small", "sgd", "lr_large")}
    assert acc["sgd"] > acc["lr_small"] > acc["lr_large"]
    # small: still falling slowly at the end; good: low; large: jumps about
    assert small[-100:].mean() < small[:100].mean() and small[-100:].mean() > 5 * good[
        -100:].mean()
    assert np.std(large[-200:]) > 5 * np.std(good[-200:])
    assert np.max(large) > 2 * np.log(10)


@needs_mnist
def test_forgetting_zero_grad_piles_the_gradients_up():
    n = S.grad_norms("no_zero_grad")
    ok = S.grad_norms("sgd")
    np.testing.assert_allclose(ok["used"], ok["fresh"])
    assert n["used"][0] == pytest.approx(n["fresh"][0])
    assert np.median(n["used"][100:] / n["fresh"][100:]) > 10
    assert S.heldout("no_zero_grad")["accuracy"][-1] < S.heldout("sgd")["accuracy"][-1] - 0.3
    assert S.losses("no_zero_grad")[-100:].mean() > 5 * S.losses("sgd")[-100:].mean()


@needs_mnist
def test_dropout_must_be_switched_off_to_test():
    h = S.heldout("dropout")
    assert h["accuracy"][-1] > h["accuracy_train_mode"][-1]
    assert h["loss"][-1] < h["loss_train_mode"][-1]


@needs_mnist
def test_a_softmax_before_the_loss_learns_worse():
    l = S.losses("double_softmax")
    assert l.min() >= np.log(1 + 9 / np.e) - 1e-4               # its floor
    assert S.heldout("double_softmax")["accuracy"][-1] < S.heldout("sgd")["accuracy"][-1]
    assert l[-100:].mean() > 5 * S.losses("sgd")[-100:].mean()


@needs_mnist
def test_adam_learns_too_and_keeps_two_more_numbers_per_weight():
    assert S.heldout("adam")["accuracy"][-1] > 0.9
    st = S.adam_state(100)
    assert set(st["m"]) == set(st["v"]) == set(S.KEYS)
    assert all(st["v"][k].min() >= 0 for k in S.KEYS)
    assert sum(st["m"][k].size + st["v"][k].size for k in S.KEYS) == 2 * S.count()


@needs_mnist
def test_more_epochs_reshuffle_and_keep_learning():
    m = S.more_epochs()
    assert m["loss"].shape == (1 + S.MORE_EPOCHS, S.EPOCH_STEPS)
    assert m["epoch_mean"][0] > m["epoch_mean"][1] > m["epoch_mean"][2]
    assert m["accuracy"][-1] > m["accuracy"][0]
    assert len({tuple(r) for r in m["first"]}) == 1 + S.MORE_EPOCHS


@needs_mnist
def test_save_and_load_gives_back_the_same_model():
    r = S.roundtrip()
    assert r["max_change"] == 0.0 and r["same_predictions"] == 10_000
    assert r["accuracy_before"] == r["accuracy_after"] == S.heldout()["accuracy"][-1]
    assert r["numbers"] == 52_650 and r["bytes_of_numbers"] == 4 * 52_650


@needs_mnist
def test_the_summary_is_plain_numbers():
    import json
    s = S.summary()
    json.dumps(s)
    assert set(S.RUNS) <= set(s) and s["parameters"] == 52_650 and s["steps"] == 938

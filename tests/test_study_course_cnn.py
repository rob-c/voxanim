"""voxanim.study_course_cnn: the CNN twin's arithmetic, against direct loops and finite
differences, and the recorded run's claims."""

from __future__ import annotations

import numpy as np
from conftest import needs_mnist

from voxanim import study_course as S
from voxanim import study_course_cnn as C


def test_the_shapes_and_the_counts():
    assert C.side_walk() == (28, 26, 13, 11, 5)
    assert C.IN_FEATURES == 16 * 5 * 5 == 400
    c = C.counts()
    assert [c[i]["total"] for i in C.LEARNS] == [8 * 1 * 9 + 8, 16 * 8 * 9 + 16, 400 * 10 + 10]
    assert [c[i]["total"] for i in C.LEARNS] == [80, 1168, 4010]
    assert C.count() == 5258 == sum(v["total"] for v in c.values())
    assert C.KEYS == ("0.weight", "0.bias", "3.weight", "3.bias", "7.weight", "7.bias")
    assert "in_features=%d" % C.IN_FEATURES in C.PRINTED_MODEL
    assert "torch.Size([%s])" % ", ".join(map(str, c[0]["shape"])) == C.PRINTED_GRAD


def test_conv_and_pool_against_direct_loops():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2, 3, 7, 6)).astype(np.float64)
    w = rng.normal(size=(4, 3, 3, 3))
    b = rng.normal(size=4)
    out = C.conv(x, w, b)
    assert out.shape == (2, 4, 5, 4)
    for n in range(2):
        for o in range(4):
            for i in range(5):
                for j in range(4):
                    want = (x[n, :, i:i + 3, j:j + 3] * w[o]).sum() + b[o]
                    assert abs(out[n, o, i, j] - want) < 1e-12
    p, where = C.pool(out)
    assert p.shape == (2, 4, 2, 2)                   # 5 x 4 -> 2 x 2: the odd row is dropped
    for i in range(2):
        for j in range(2):
            sq = out[:, :, 2 * i:2 * i + 2, 2 * j:2 * j + 2].reshape(2, 4, 4)
            assert np.array_equal(p[:, :, i, j], sq.max(-1))
            assert np.array_equal(where[:, :, i, j], sq.argmax(-1))


def test_the_gradients_against_finite_differences():
    rng = np.random.default_rng(1)
    P = {k: v.astype(np.float64) for k, v in C.init_params().items()}
    x = rng.normal(size=(3, 1, 28, 28))
    y = np.array([3, 1, 7])
    _loss, G = C.loss_and_grads(P, x, y)
    for k in C.KEYS:
        for idx in [tuple(rng.integers(0, s) for s in P[k].shape) for _ in range(3)]:
            h = 1e-6
            up = {**P, k: P[k].copy()}
            up[k][idx] += h
            down = {**P, k: P[k].copy()}
            down[k][idx] -= h
            num = (C.loss_and_grads(up, x, y)[0] - C.loss_and_grads(down, x, y)[0]) / (2 * h)
            assert abs(num - G[k][idx]) < 1e-6 * max(1.0, abs(num)), (k, idx)


@needs_mnist
def test_the_shapes_on_the_first_batch():
    assert C.shapes() == [("input", (64, 1, 28, 28)), ("Conv2d", (64, 8, 26, 26)),
                          ("ReLU", (64, 8, 26, 26)), ("MaxPool2d", (64, 8, 13, 13)),
                          ("Conv2d", (64, 16, 11, 11)), ("ReLU", (64, 16, 11, 11)),
                          ("MaxPool2d", (64, 16, 5, 5)), ("Flatten", (64, 400)),
                          ("Linear", (64, 10))]


@needs_mnist
def test_a_stamp_weights_gradient_is_the_sum_of_its_parts():
    for step, stamp, r, c in ((0, 0, 1, 1), (100, 5, 0, 2)):
        g = C.stamp_grad_parts(step, stamp, r, c)
        assert g["parts"].shape == (64, 26, 26)            # 64 digits x 676 places
        assert abs(g["total"] - g["grad"]) < 1e-6
        assert abs(g["grad"] - C.grads(step)["0.weight"][stamp, 0, r, c]) < 1e-7


@needs_mnist
def test_the_run():
    l = C.losses()
    assert len(l) == C.EPOCH_STEPS == 938
    assert abs(l[0] - np.log(10)) < 0.15                   # untrained: about ln 10
    assert l[-100:].mean() < 0.15 < S.losses("sgd")[-100:].mean()
    h = C.heldout()
    assert list(h["steps"]) == list(S.heldout("sgd")["steps"])
    assert h["accuracy"][0] < 0.2 and h["accuracy"][-1] > 0.95
    # the step really is SGD's: new = old - 0.1 x gradient
    for k in C.KEYS:
        assert np.allclose(C.params(1)[k], C.params(0)[k] - 0.1 * C.grads(0)[k], atol=1e-7)
    a = C.activations(C.EPOCH_STEPS, 0)
    assert a["p2"].shape == (1, 16, 5, 5) and a["probs"].shape == (1, 10)
    assert int(a["probs"].argmax()) == a["y"] == S.digit(0)[1]


@needs_mnist
def test_against_the_mlp():
    c = C.compare()
    assert (c["cnn_params"], c["mlp_params"]) == (5258, 52650)
    assert c["cnn_accuracy"] > c["mlp_accuracy"]
    assert c["both_right"] + c["only_cnn"] + c["only_mlp"] + c["both_wrong"] == 10_000
    assert c["cnn_missed"] == c["only_mlp"] + c["both_wrong"] == sum(c["per_digit_cnn"])
    assert c["mlp_missed"] == c["only_cnn"] + c["both_wrong"] == sum(c["per_digit_mlp"])
    assert abs(c["cnn_accuracy"] - (1 - c["cnn_missed"] / 10_000)) < 1e-12
    assert sum(n for _t, _p, n in c["confusions"]) == c["cnn_missed"]

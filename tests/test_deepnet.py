"""The deep MLP behind the training-deep-networks animations: every option's
gradients against finite differences, and the claims the subtitles make."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fd_check, needs_mnist

from voxanim import deepnet as D

# ------------------------------------------------------------------ gradients

OPTIONS = [
    dict(sizes=(5, 4, 4, 4, 3), activation="tanh", init="xavier"),
    dict(sizes=(5, 4, 4, 4, 3), activation="relu", init="he"),
    dict(sizes=(5, 4, 4, 4, 3), activation="tanh", init="normal", init_std=0.5),
    dict(sizes=(5, 4, 4, 4, 4, 3), activation="tanh", residual=1),
    dict(sizes=(5, 4, 4, 4, 4, 3), activation="relu", residual=2),
    dict(sizes=(5, 4, 4, 3), activation="tanh", norm="batch"),
    dict(sizes=(5, 4, 4, 3), activation="relu", norm="batch"),
    dict(sizes=(5, 4, 4, 3), activation="tanh", norm="layer"),
    dict(sizes=(5, 4, 4, 3), activation="relu", norm="layer"),
    dict(sizes=(5, 4, 4, 3), activation="tanh", dropout=0.5),
    dict(sizes=(5, 4, 4, 3), activation="relu", weight_decay=0.1),
    dict(sizes=(5, 4, 4, 4, 4, 3), activation="tanh", residual=1, norm="batch", dropout=0.3),
    dict(sizes=(5, 4, 4, 4, 4, 3), activation="relu", residual=2, norm="layer", dropout=0.25,
         weight_decay=0.05),
]


def _problem(spec, seed=0):
    """A small batch, random biases (so no ReLU sits exactly on its kink) and
    fixed dropout masks."""
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (6, spec.sizes[0]))
    y = np.eye(spec.sizes[-1])[rng.integers(0, spec.sizes[-1], 6)]
    P = D.init(spec, rng)
    for k in P:
        if k.startswith("b"):
            P[k] = rng.normal(0, 0.3, P[k].shape)
    masks = [rng.random((6, w)) >= spec.dropout for w in spec.sizes[1:-1]] \
        if spec.dropout else None
    return x, y, P, masks


@pytest.mark.parametrize("opts", OPTIONS, ids=lambda o: "-".join(
    "%s=%s" % (k, v) for k, v in o.items() if k != "sizes"))
def test_gradients_match_finite_differences(opts):
    """Every parameter's gradient, through norm, residual path, dropout and
    weight decay, matches central differences of the objective."""
    spec = D.Spec(**opts)
    x, y, P, masks = _problem(spec)

    def loss(Q):
        f = D.forward(spec, Q, x, train=True, masks=masks, stats=D.init_stats(spec))
        return D.objective(spec, Q, f, y)

    f = D.forward(spec, P, x, train=True, masks=masks, stats=D.init_stats(spec))
    G, _D = D.backward(spec, P, f, y)
    assert set(G) == set(P)
    fd_check(loss, P, G, list(P), tol=1e-5)


@pytest.mark.parametrize("residual", [1, 2])
def test_gradient_at_the_input_through_skips(residual):
    """The gradient reaching each hidden layer (what the bands draw) is the
    true derivative with respect to that layer's output."""
    spec = D.Spec((5, 4, 4, 4, 4, 3), "tanh", residual=residual)
    x, y, P, _m = _problem(spec)
    f = D.forward(spec, P, x)
    _G, Dd = D.backward(spec, P, f, y)
    e = 1e-6
    for idx in ((0, 0), (3, 2)):
        xp, xm = x.copy(), x.copy()
        xp[idx] += e
        xm[idx] -= e
        fd = (D.objective(spec, P, D.forward(spec, P, xp), y) -
              D.objective(spec, P, D.forward(spec, P, xm), y)) / (2 * e)
        assert abs(fd - Dd["dh"][0][idx]) <= 1e-6 * max(1e-3, abs(fd))


def test_skips_only_where_widths_match():
    spec = D.Spec((784, 16, 16, 16, 16, 10), "relu", residual=2)
    assert [spec.skip(l) for l in range(4)] == [False, False, False, True]
    spec = D.Spec((784, 16, 16, 16, 10), "relu", residual=1)
    assert [spec.skip(l) for l in range(3)] == [False, True, True]


def test_batch_norm_running_statistics_are_used_when_evaluating():
    spec = D.Spec((3, 4, 2), "tanh", norm="batch")
    rng = np.random.default_rng(1)
    P = D.init(spec, rng)
    stats = D.init_stats(spec)
    x = rng.normal(2.0, 3.0, (64, 3))
    for _ in range(200):
        D.forward(spec, P, x, train=True, stats=stats)
    f_train = D.forward(spec, P, x, train=True, stats=dict(stats))
    f_eval = D.forward(spec, P, x, stats=stats)
    assert np.allclose(f_train["p"], f_eval["p"], atol=1e-3)
    assert np.allclose(f_train["zn"][0].mean(0), 0, atol=1e-9)
    assert np.allclose(f_train["zn"][0].var(0), 1, atol=1e-3)


def test_layer_norm_standardises_each_row():
    spec = D.Spec((3, 5, 2), "relu", norm="layer")
    rng = np.random.default_rng(2)
    f = D.forward(spec, D.init(spec, rng), rng.normal(0, 1, (7, 3)))
    assert np.allclose(f["zn"][0].mean(1), 0, atol=1e-9)
    assert np.allclose(f["zn"][0].var(1), 1, atol=1e-3)


def test_sweep_schedule():
    from voxanim.conv import Tempo
    from voxanim.sweep import Sweep
    s = Sweep(2.0, 5, Tempo(slow=1.0, fast=0.5, ramp_in=1.0, ramp_out=1.0))
    assert s.at(1.0) == (0, -1, 0.0)
    assert s.at(s.t1 + 1) == (5, 5, 0.0)
    done, cur, u = s.at(2.0 + s.durs[0] + 0.5 * s.durs[1])
    assert (done, cur) == (1, 1) and abs(u - 0.5) < 1e-9


# ------------------------------------------------------------------ the animations' claims


@needs_mnist
def test_initialisation_claims():
    """Only the Xavier network learns; the small init's activations and
    gradients vanish, the large init's gradient explodes on the way back."""
    from voxanim import init_anim as A
    small, large, xavier = A.runs()
    assert small.act_std[0][-1] < 1e-15 and small.grad_rms[0][0] < 1e-15
    assert large.act_std[0].min() > 0.9                      # saturated tanh
    assert large.grad_rms[0][0] / large.grad_rms[0][-1] > 1e5
    assert xavier.act_std[0].min() > 0.1 and xavier.act_std[0].max() < 0.5
    assert 0.2 < xavier.grad_rms[0][0] / xavier.grad_rms[0][-1] < 5
    assert small.test_acc[-1] < 0.2 and large.test_acc[-1] < 0.2 and xavier.test_acc[-1] > 0.85
    hist = A.histogram(small.probe["a"][0])
    assert hist[A.BINS // 2 - 1] + hist[A.BINS // 2] > 0.99     # all at 0
    hist = A.histogram(large.probe["a"][0])
    assert hist[0] + hist[-1] > 0.8                             # at +-1


@needs_mnist
def test_skip_connection_claims():
    """The plain 30-layer network's gradient shrinks a hundred-thousand-fold
    and it never learns; the residual one's arrives larger and it learns."""
    from voxanim import residual_anim as A
    (plain, res), x, label = A.runs()
    assert plain.grad_rms[0][0] / plain.grad_rms[0][-1] < 1e-4
    assert res.grad_rms[0][0] / res.grad_rms[0][-1] > 1
    assert plain.act_std[0][-1] < 1e-4 and res.act_std[0][-1] > 1
    assert plain.test_acc[-1] < 0.2 and abs(plain.train_loss[-1] - np.log(10)) < 0.01
    assert res.test_acc[-1] > 0.85
    h, dh = A.example_through(plain, x, label)
    assert h.shape == dh.shape == (A.WIDTH, A.DEPTH)
    assert np.abs(dh[:, 0]).max() < 1e-6 * np.abs(dh[:, -1]).max()


@needs_mnist
def test_normalisation_claims():
    """At the high learning rate the plain network collapses; batch norm and
    layer norm learn.  The normalisation the animation recomputes from the
    recorded pre-activations is the network's own."""
    from voxanim import norm_anim as A
    plain, bn, ln = A.runs()
    assert plain.test_acc[-1] < 0.2 and bn.test_acc[-1] > 0.9 and ln.test_acc[-1] > 0.7
    assert plain.batch_loss[-50:].mean() > 2.0                 # back at chance
    k = int(np.searchsorted(bn.steps, A.CHECK_STEP))
    zn, _mu, _sd = A.standardise(bn.probe["z"][k].astype(np.float64), 0)
    assert np.allclose(bn.probe["g"][k] * zn + bn.probe["beta"][k], bn.probe["y"][k], atol=1e-5)
    zn, _mu, _sd = A.standardise(ln.probe["z"][k].astype(np.float64), 1)
    assert np.allclose(ln.probe["g"][k] * zn + ln.probe["beta"][k], ln.probe["y"][k], atol=1e-5)
    assert np.allclose(zn.mean(1), 0, atol=1e-6) and np.allclose(zn.std(1), 1, atol=1e-2)


@needs_mnist
def test_overfitting_claims():
    """The plain run overfits (training loss to 0, test loss and error up);
    each cure keeps the test loss down; early stopping finds an early step."""
    from voxanim import overfit_anim as A
    data, images, (plain, wd, drop) = A.runs()
    assert data.x_train.shape == (A.N_TRAIN, 2) and images.shape == (A.N_TRAIN, 784)
    assert np.allclose(data.x_train.std(0), 1.0)
    assert plain.train_loss[-1] < 0.01 and plain.test_loss[-1] > 1.5
    b = plain.best
    assert plain.steps[b] < 100 and plain.test_acc[b] - plain.test_acc[-1] > 0.05
    assert wd.test_loss[-1] < 0.6 and wd.test_acc[-1] >= 0.78 and wd.train_loss[-1] > 0.2
    assert drop.test_loss[-1] < 0.6 and drop.test_acc[-1] >= 0.75
    assert plain.field.shape == (len(plain.steps), A.GRID * A.GRID)
    assert drop.probe["m"].shape == (len(drop.steps), A.N_TRAIN, A.SIZES[1])
    assert abs(drop.probe["m"].mean() - (1 - A.DROPOUT)) < 0.02

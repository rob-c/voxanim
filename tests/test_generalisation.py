"""The generalisation animations: the options added to the deep MLP's
training (gradients unchanged, augmentation reproducible), the measurements
of :mod:`voxanim.generalise`, and the claims the five animations make, as
relative claims (CI retrains every model on another CPU)."""

from __future__ import annotations

import numpy as np
from conftest import fd_check

from voxanim import deepnet as D
from voxanim import generalise as G

# ------------------------------------------------------------------ the model's additions


def test_gradients_still_match_finite_differences():
    """The one-hidden-layer ReLU network every animation here trains: the
    gradients of the objective against central differences."""
    spec = D.Spec((6, 5, 3), "relu", "he")
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (7, 6))
    y = np.eye(3)[rng.integers(0, 3, 7)]
    P = D.init(spec, rng)
    P["b0"] = rng.normal(0, 0.3, P["b0"].shape)        # no ReLU exactly on its kink

    def loss(Q):
        return D.objective(spec, Q, D.forward(spec, Q, x), y)

    G_, _D = D.backward(spec, P, D.forward(spec, P, x), y)
    fd_check(loss, P, G_, list(P))


def test_example_losses_are_the_per_row_cross_entropy():
    spec = D.Spec((4, 3, 2), "tanh")
    rng = np.random.default_rng(1)
    P = D.init(spec, rng)
    x, y = rng.normal(0, 1, (2500, 4)), rng.integers(0, 2, 2500)
    data = D.Data(x, y, x[:5], y[:5], 2)
    el = D.example_losses(spec, P, data)
    assert el.shape == (2500,)
    assert np.isclose(el.mean(), D.cross_entropy(D.forward(spec, P, x)["p"], np.eye(2)[y]))


def _tiny_data(n=40, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((n, 784))
    y = rng.integers(0, 10, n)
    return D.Data(x, y, x[:10], y[:10], 10)


class _Identity:
    """An augmentation that changes nothing (but draws from its generator)."""

    def settings(self):
        return dict(kind="identity")

    def __call__(self, rng, x):
        rng.random(len(x))
        return x


def test_augmentation_draws_from_its_own_stream(tmp_path):
    """With an identity augmentation the run is bit-identical to the plain
    one: the minibatch order and the initial weights do not depend on the
    augmentation's draws."""
    spec = D.Spec((784, 4, 10), "relu", "he")
    data = _tiny_data()
    kw = dict(steps=12, batch=8, lr=0.01, opt="adam", cache_dir=str(tmp_path))
    plain = D.train(spec, data, **kw)
    same = D.train(spec, data, augment=_Identity(), **kw)
    assert all(np.array_equal(plain.P[k], same.P[k]) for k in plain.P)
    assert np.array_equal(plain.batch_loss, same.batch_loss)
    shifted = D.train(spec, data, augment=G.Augment(2.0, 10.0, 0.1), **kw)
    assert not np.array_equal(plain.P["W0"], shifted.P["W0"])
    assert np.array_equal(plain.P0["W0"], shifted.P0["W0"])


def test_probe_rows_and_example_loss_are_recorded(tmp_path):
    spec = D.Spec((784, 4, 10), "relu", "he")
    data = _tiny_data()
    run = D.train(spec, data, 6, 8, 0.01, "adam", every=2, probe_layer=0, probe_keys=("h",),
                  probe_rows=3, example_loss=True, cache_dir=str(tmp_path))
    assert run.probe["h"].shape == (4, 3, 784)
    assert run.example_loss.shape == (4, 40)
    assert np.isclose(run.example_loss[0].mean(), run.train_loss[0], atol=1e-5)


# ------------------------------------------------------------------ the measurements


def test_corrupt_changes_exactly_the_fraction_and_never_to_the_truth():
    y = np.random.default_rng(0).integers(0, 10, 1000)
    yn, mask = G.corrupt(y, 0.25, 10, np.random.default_rng(1))
    assert mask.sum() == 250 and (yn[mask] != y[mask]).all() and (yn[~mask] == y[~mask]).all()
    assert yn.min() >= 0 and yn.max() < 10


def test_affine_identity_shift_and_rotation():
    rng = np.random.default_rng(2)
    x = rng.random((3, 784))
    assert np.allclose(G.affine(x, 0, 0, 0, 1), x)
    # an integer shift is a roll (with zeros pulled in at the edge)
    img = x.reshape(3, 28, 28)
    right = G.affine(x, 2, 0, 0, 1).reshape(3, 28, 28)
    assert np.allclose(right[:, :, 2:], img[:, :, :-2]) and np.allclose(right[:, :, :2], 0)
    down = G.affine(x, 0, 3, 0, 1).reshape(3, 28, 28)
    assert np.allclose(down[:, 3:], img[:, :-3]) and np.allclose(down[:, :3], 0)
    # a quarter turn is exact on the pixel grid: four of them give the image back
    q = x
    for _ in range(4):
        q = G.affine(q, 0, 0, np.pi / 2, 1)
    assert np.allclose(q, x, atol=1e-9)
    # a mild transform keeps the ink roughly (bilinear sampling is mass-preserving in the interior)
    blob = np.zeros((1, 28, 28))
    blob[0, 10:18, 10:18] = 1.0
    t = G.affine(blob.reshape(1, 784), 1.5, -1.0, 0.2, 1.0)
    assert abs(t.sum() / blob.sum() - 1) < 0.05


def test_confusion_and_errors():
    y = np.array([0, 1, 2, 2, 1])
    pred = np.array([0, 2, 2, 1, 1])
    m = G.confusion(pred, y, 3)
    assert m.sum() == 5 and m[0, 0] == 1 and m[1, 2] == 1 and m[2, 1] == 1 and m[1, 1] == 1
    assert list(G.misclassified(pred, y)) == [1, 3]


def test_slices_of_the_loss_agree_with_direct_evaluation():
    spec = D.Spec((4, 3, 2), "tanh")
    rng = np.random.default_rng(3)
    Pa, Pb = D.init(spec, rng), D.init(spec, rng)
    x, y = rng.normal(0, 1, (50, 4)), rng.integers(0, 2, 50)
    line = G.along_line(spec, Pa, Pb, [0.0, 0.5, 1.0], x, y, 2)
    assert np.isclose(line[0], G.loss_on(spec, Pa, x, y, 2))
    assert np.isclose(line[2], G.loss_on(spec, Pb, x, y, 2))
    d = G.random_direction(Pa, rng)
    assert np.isclose(np.sqrt(sum((v * v).sum() for v in d.values())), 1.0)
    grid = G.plane_grid(spec, Pa, Pb, d, [0.0, 1.0], [0.0], x, y, 2)
    assert np.allclose(grid[:, 0], line[[0, 2]])
    assert np.isclose(G.param_distance(Pa, G.moved(Pa, d, 2.5)), 2.5)
    a = G.around(spec, Pa, rng, [0.0, 1.0], x, y, 2, directions=2)
    assert a.shape == (2, 2) and np.allclose(a[:, 0], line[0])

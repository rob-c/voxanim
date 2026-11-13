"""The generalisation animations: the options added to the deep MLP's
training (gradients unchanged, augmentation reproducible), the measurements
of :mod:`voxanim.generalise`, and the claims the five animations make, as
relative claims (CI retrains every model on another CPU)."""

from __future__ import annotations

import numpy as np
from conftest import fd_check, needs_mnist

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


# ------------------------------------------------------------------ the animations' claims
#
# Relative claims only: CI retrains every run on another CPU, and after
# thousands of Adam steps the last bits differ.


def _settings(run):
    """The settings stored in a run's cache file (what ``train`` keyed it by)."""
    import json
    import os

    from voxanim.cache import DEFAULT_CACHE, settings_key
    for name in os.listdir(DEFAULT_CACHE):
        if not name.startswith("deepnet-") or not name.endswith(".npz"):
            continue
        with np.load(os.path.join(DEFAULT_CACHE, name), allow_pickle=False) as d:
            s = json.loads(str(d["_settings"]))
        if settings_key(s) == name[len("deepnet-"):-len(".npz")] and \
                s["sizes"] == list(run.spec.sizes) and len(run.batch_loss) == s["steps"]:
            yield s


@needs_mnist
def test_double_descent_claims():
    """The test loss peaks at the first widths that fit every label and falls
    again past them, below the U's minimum; the training error reaches zero
    from the threshold on; the widest interpolator is the smoothest."""
    from voxanim import ddescent_anim as A
    data, mask, runs = A.runs()
    anim = A.DoubleDescent(data, mask, runs)
    assert mask.sum() == round(A.NOISE * A.N_TRAIN)
    assert all(len(r.batch_loss) == A.EPOCHS * A.N_TRAIN // A.TRAIN["batch"] for r in runs)
    u, p, w = anim.u_min, anim.peak, anim.wide
    widths = np.array(A.WIDTHS, float)
    assert widths[u] < widths[p] < widths[w]
    # the peak sits within a factor of three of the parameter-count threshold
    assert 1 / 3 < widths[p] / A.threshold_width() < 3
    tl = anim.test_loss
    assert tl[p] > 1.25 * tl[u] and tl[p] > 1.5 * tl[w] and tl[w] < tl[u]
    # trained to interpolation from the threshold on
    assert anim.train_err[p] < 0.02 and (anim.train_err[p + 1:] <= A.INTERPOLATES).all()
    # the test error only falls here (no U is claimed for it)
    assert anim.test_err[0] > anim.test_err[p] > anim.test_err[w]
    # smoothness: most sensitive near the threshold, least at the widest
    top = int(np.argmax(anim.sens))
    assert widths[top] <= 2 * widths[p] and anim.sens[top] > 2 * anim.sens[w]


@needs_mnist
def test_memorising_noise_claims():
    """All three runs fit their labels (training loss far below chance,
    training error under 1%); the test accuracy falls with the noise; clean
    examples are learned before shuffled ones; early stopping beats the end."""
    from voxanim import memorise_anim as A
    sets, runs = A.runs()
    anim = A.MemorisingNoise(sets, runs)
    masks = [m for _d, m in sets]
    assert [int(m.sum()) for m in masks] == [round(n * A.N_TRAIN) for n in A.NOISES]
    for r, (d, _m) in zip(runs, sets):
        assert r.example_loss.shape == (len(r.steps), A.N_TRAIN)
        assert r.example_loss[-1].mean() < 0.05
        assert A.error_rate(r.spec, r.P, d.x_train, d.y_train) < 0.01
    acc = [1 - e for e in anim.test_err]
    assert acc[0] > 0.9 and acc[0] > acc[1] + 0.1 > acc[2] + 0.2
    # the 50% run: at every recorded step the clean half has the lower loss
    # once learning has started, and half-way through it is far lower
    k = int(np.searchsorted(runs[-1].steps, 1000))
    assert anim.clean_curve[k] < 0.5 * anim.shuffled_curve[k]
    assert (anim.clean_curve[5:] < anim.shuffled_curve[5:]).all()
    b = anim.best
    assert runs[-1].steps[b] < A.TRAIN["steps"] / 4
    assert runs[-1].test_acc[b] > acc[2] + 0.15
    assert anim.heat.shape == (len(runs[-1].steps), A.HEAT_COLS, 3)


@needs_mnist
def test_data_vs_model_size_claims():
    """More data helps every model; the small one saturates (its last
    sixfold of data buys under a point) while the large one keeps gaining;
    with all the data the models rank by size; with 100 images they are
    within a few points."""
    from voxanim import datasize_anim as A
    data, table = A.runs()
    anim = A.DataVsModelSize(data, table)
    e = anim.err
    assert e.shape == (3, 6)
    for row in e:
        assert row[0] > row[2] > row[5]                      # more data helps
    assert e[0, 4] - e[0, 5] < 0.01                          # the small one saturates
    assert e[2, 4] - e[2, 5] > 0.008                         # the large one keeps gaining
    assert e[0, 5] > e[1, 5] > e[2, 5]                       # ranked by size at 60,000
    assert e[0, 5] > 2.5 * e[2, 5]
    assert abs(e[0, 0] - e[2, 0]) < 0.08                     # close at 100 images
    assert all(len(idx) == A.N_WRONG for idx, _p in anim.wrong)
    assert all((p != data.y_test[idx]).all() for idx, p in anim.wrong)


@needs_mnist
def test_augmentation_claims():
    """The plain run memorises (training loss near zero, test loss rising
    from its low); the augmented run generalises better and is far more
    invariant to shifts and rotations."""
    from voxanim import augment_anim as A
    data, plain, aug = A.runs()
    assert plain.probe["h"].shape == (len(plain.steps), A.PROBE_ROWS, 784)
    assert np.allclose(plain.P0["W0"], aug.P0["W0"])         # the same start
    assert plain.train_loss[-1] < 0.01
    assert plain.test_loss[-1] > 1.15 * plain.test_loss.min()
    assert aug.test_loss[-1] < 0.6 * plain.test_loss[-1]
    pa, aa = A.invariance(data, plain), A.invariance(data, aug)
    assert aa[0] > pa[0] + 0.04                              # clean test accuracy
    assert pa[1] < 0.7 and aa[1] > pa[1] + 0.3               # shifted test digits
    assert aa[2] > pa[2] + 0.03                              # rotated ones
    # the recorded augmented minibatch differs from the stored images, the plain one does not
    assert np.abs(aug.probe["h"][1] - plain.probe["h"][1]).max() > 0.1
    rows = plain.probe["h"][1]
    assert all(any(np.allclose(r, x) for x in data.x_train) for r in rows[:2])


@needs_mnist
def test_sharp_vs_flat_claims():
    """Both runs reach a low training loss; the large-batch minimum is the
    sharper one (narrower bowl, more loss at the same distance), has the
    higher test error and suffers more from weight noise; the line between
    them has a barrier in the training loss."""
    from voxanim import minima_anim as A
    data, small, large, m = A.runs()
    anim = A.SharpVsFlat(data, small, large, m)
    assert len(small.batch_loss) == A.EPOCHS * (A.N_TRAIN // A.SMALL["batch"])
    assert len(large.batch_loss) == A.EPOCHS * -(-A.N_TRAIN // A.LARGE["batch"])
    assert m["train_loss"].max() < 0.02
    assert m["test_err"][1] > m["test_err"][0] + 0.003
    w_small, w_large = anim.width
    assert w_large < 0.7 * w_small
    radii = m["radii"]
    far = np.abs(radii) >= A.RADIUS / 2
    assert (anim.prof[1][far] > anim.prof[0][far]).all()
    assert m["perturbed"][1, -1] - m["test_err"][1] > m["perturbed"][0, -1] - m["test_err"][0]
    us = m["us"]
    inside = (us > 0) & (us < 1)
    assert m["line_train"][inside].max() > 2 * max(m["line_train"][np.argmin(np.abs(us))],
                                                   m["line_train"][np.argmin(np.abs(us - 1))])
    assert m["grid"].shape == (A.GRID_N, A.GRID_N)
    # the plane's u axis passes through both minima: the grid agrees with the line (its
    # nearest cell is 0.02 in u and 1.4 in v away, so to the loss's own scale)
    j0 = int(np.argmin(np.abs(m["gv"])))
    for u in (0.0, 1.0):
        i = int(np.argmin(np.abs(m["gu"] - u)))
        k = int(np.argmin(np.abs(us - u)))
        assert abs(m["grid"][i, j0] - m["line_train"][k]) < 1e-3


def _stored(run, **expect):
    """True when a cache file of ``run``'s shape stores settings with every
    item of ``expect`` (nested dicts compared whole)."""
    return any(all(s.get(k) == v for k, v in expect.items()) for s in _settings(run))


@needs_mnist
def test_runs_match_their_stored_settings():
    """What each animation says it trained is what its cache files say they
    hold: optimiser, batch, rate, the augmentation, the per-example record."""
    from voxanim import augment_anim as AU
    from voxanim import memorise_anim as ME
    _data, plain, aug = AU.runs()
    assert _stored(plain, opt="adam", batch=AU.TRAIN["batch"], lr=AU.TRAIN["lr"],
                   probe_layer=0, probe_rows=AU.PROBE_ROWS, probe_keys=["h"])
    assert _stored(aug, augment=dict(kind="affine", shift=AU.SHIFT, degrees=AU.DEGREES,
                                     scale=AU.SCALE), batch=AU.TRAIN["batch"])
    _sets, runs = ME.runs()
    for r in runs:
        assert _stored(r, example_loss=True, opt="adam", batch=ME.TRAIN["batch"],
                       lr=ME.TRAIN["lr"], steps=ME.TRAIN["steps"])

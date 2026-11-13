"""The activation-function family: the new activations' gradients against
finite differences through :mod:`voxanim.deepnet`, the recorded runs against
their settings, and every claim the five animations make, as relative claims
(CI retrains on another machine, where the last bits differ)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fd_check, needs_mnist

from voxanim import deepnet as D

NEW = ("sigmoid", "leaky_relu", "gelu", "silu", "elu")


# ------------------------------------------------------------------ the functions


@pytest.mark.parametrize("name", D.ACTIVATIONS)
def test_derivative_matches_finite_differences(name):
    """``act_grad`` is the derivative of ``act`` (away from ReLU's kink at 0)."""
    z = np.linspace(-5, 5, 101) + 0.0137
    e = 1e-6
    fd = (D.act(name, z + e) - D.act(name, z - e)) / (2 * e)
    assert np.allclose(fd, D.act_grad(name, z), atol=1e-6)
    assert np.allclose(D.act_grad(name, z), D.act_grad(name, z, D.act(name, z)))


def test_tanh_and_relu_are_unchanged():
    """The two old activations compute exactly what they did (cached runs depend on it)."""
    y = np.random.default_rng(0).normal(0, 2, (5, 7))
    assert np.array_equal(D.act("tanh", y), np.tanh(y))
    assert np.array_equal(D.act("relu", y), np.maximum(y, 0.0))
    assert np.array_equal(D.act_grad("tanh", y, np.tanh(y)), 1.0 - np.tanh(y) ** 2)
    assert np.array_equal(D.act_grad("relu", y), (y > 0).astype(np.float64))
    with pytest.raises(ValueError):
        D.act("softsign", y)


def test_function_facts():
    """The facts the subtitles state about the functions themselves."""
    z = np.linspace(-8, 8, 4001)
    assert D.act_grad("sigmoid", z).max() <= 0.25 + 1e-12
    assert abs(D.act_grad("tanh", z).max() - 1.0) < 1e-6
    assert set(np.unique(D.act_grad("relu", z))) <= {0.0, 1.0}
    assert D.act("gelu", z).min() < -0.1 and D.act("silu", z).min() < -0.2
    assert D.act("elu", z).min() > -1.0 and D.act("elu", -50.0) == pytest.approx(-1.0)
    assert D.act_grad("gelu", z).max() > 1.0 and D.act_grad("silu", z).max() > 1.0
    assert np.all(D.act_grad("leaky_relu", z) >= D.LEAKY_SLOPE)


@pytest.mark.parametrize("name", NEW)
def test_network_gradients_match_finite_differences(name):
    """Every parameter's gradient through a small network of each new activation."""
    spec = D.Spec((5, 4, 4, 3), name, "he")
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (6, 5))
    y = np.eye(3)[rng.integers(0, 3, 6)]
    P = D.init(spec, rng)
    for k in P:
        if k.startswith("b"):
            P[k] = rng.normal(0, 0.3, P[k].shape)

    def loss(Q):
        return D.objective(spec, Q, D.forward(spec, Q, x), y)

    f = D.forward(spec, P, x)
    G, Dd = D.backward(spec, P, f, y)
    fd_check(loss, P, G, list(P), tol=1e-5)
    # the gradient at the input, too (what the gradient bands draw)
    e = 1e-6
    xp, xm = x.copy(), x.copy()
    xp[1, 2] += e
    xm[1, 2] -= e
    fd = (D.objective(spec, P, D.forward(spec, P, xp), y)
          - D.objective(spec, P, D.forward(spec, P, xm), y)) / (2 * e)
    assert abs(fd - Dd["dh"][0][1, 2]) <= 1e-6 * max(1e-3, abs(fd))


def test_sparsity_is_recorded(tmp_path):
    """``train(sparse=True)`` records the three per-layer fractions at every
    recorded step; ReLU's exact zeros are its near-zeros and its flat part."""
    rng = np.random.default_rng(0)
    data = D.Data(rng.normal(0, 1, (40, 6)), rng.integers(0, 3, 40), rng.normal(0, 1, (10, 6)),
                  rng.integers(0, 3, 10), classes=3)
    spec = D.Spec((6, 8, 8, 3), "relu", "he")
    run = D.train(spec, data, steps=6, batch=8, lr=0.01, every=2, sparse=True,
                  cache_dir=str(tmp_path))
    S = len(run.steps)
    assert set(run.sparse) == {"zero", "small", "flat"}
    assert run.sparse["zero"].shape == (S, 2)
    assert np.all(run.sparse["zero"] <= run.sparse["small"])
    assert np.allclose(run.sparse["zero"], run.sparse["flat"])
    assert 0.2 < run.sparse["zero"][0].mean() < 0.8          # He init: about half negative
    plain = D.train(spec, data, steps=6, batch=8, lr=0.01, every=2, cache_dir=str(tmp_path))
    assert plain.sparse == {}
    assert np.allclose(plain.test_acc, run.test_acc)        # the option changes nothing else


# ------------------------------------------------------------------ the runs


def _index():
    from voxanim import actfun
    return {a.key: k for k, a in enumerate(actfun.ACTS)}


def _consistent(runs, sizes, settings):
    from voxanim import actfun
    for a, r in zip(actfun.ACTS, runs):
        assert r.spec.activation == a.key and r.spec.init == a.init
        assert tuple(r.spec.sizes) == sizes and r.steps[-1] == settings["steps"]
        assert r.spec.init == ("xavier" if a.key in ("sigmoid", "tanh") else "he")


@needs_mnist
def test_race_claims():
    """Seven runs on one network: sigmoid learns slowest and ends worst, all
    seven learn; ReLU switches far more of a layer off than GELU; the time per
    step is measured and the function alone costs most for GELU."""
    from voxanim import actfun
    from voxanim.actfun_race import names, ties
    dat = actfun.data()
    runs = actfun.race_runs(dat=dat)
    _consistent(runs, actfun.RACE_SIZES, actfun.RACE)
    i = _index()
    acc = np.array([r.test_acc[-1] for r in runs])
    loss = np.array([r.train_loss[-1] for r in runs])
    assert acc.min() > 0.9 and acc.argmin() == i["sigmoid"] and loss.argmax() == i["sigmoid"]
    k = int(np.searchsorted(runs[0].steps, 50))
    assert runs[i["sigmoid"]].test_acc[k] < runs[i["relu"]].test_acc[k] - 0.1
    assert i["sigmoid"] not in ties(acc) and names([0, 2]) == "sigmoid and ReLU"
    x = dat.x_test[:64]
    zero_relu, small_relu, _f = actfun.fractions(runs[i["relu"]].spec, runs[i["relu"]].P, x)
    _z, small_gelu, _f = actfun.fractions(runs[i["gelu"]].spec, runs[i["gelu"]].P, x)
    assert zero_relu.min() > 0.2 and np.allclose(zero_relu, small_relu, atol=0.01)
    assert zero_relu.min() > 4 * small_gelu.max()
    for a, r in zip(actfun.ACTS, runs):
        assert r.sparse["zero"].shape == (len(r.steps), 2)
        if a.key != "relu":
            assert r.sparse["zero"].max() < 0.01     # the others only by underflow, far out
    times = actfun.step_times(dat=dat)
    assert times["step"].shape == (7,) and np.all(times["step"] > 0)
    assert times["fn"][i["gelu"]] > 2 * times["fn"][i["relu"]]


@needs_mnist
def test_gradient_claims():
    """Eight layers: sigmoid's gradient vanishes on the way back, ReLU's does
    not, GELU's and SiLU's sit in between; after training Adam has levelled them."""
    from voxanim import actfun
    from voxanim.actfun_gradients import HIST_LAYER, flat_bins, histogram
    runs = actfun.deep8_runs()
    _consistent(runs, actfun.DEEP8_SIZES, actfun.DEEP8)
    i = _index()
    r0 = np.array([r.grad_rms[0][0] / r.grad_rms[0][-1] for r in runs])
    r1 = np.array([r.grad_rms[-1][0] / r.grad_rms[-1][-1] for r in runs])
    assert r0[i["sigmoid"]] < 1e-3
    assert 0.3 < r0[i["relu"]] < 3 and 0.3 < r0[i["leaky_relu"]] < 3
    for k in ("gelu", "silu"):
        assert r0[i["sigmoid"]] < r0[i[k]] < 0.5
    assert r0[i["elu"]] > 1
    assert r1.min() > 0.1 and r1[i["sigmoid"]] > 100 * r0[i["sigmoid"]]
    acc = np.array([r.test_acc[-1] for r in runs])
    assert acc[i["sigmoid"]] < 0.7 < 0.9 < np.delete(acc, i["sigmoid"]).min()
    dat = actfun.data()
    ys, _a = actfun.hidden(runs[0].spec, runs[0].P0, dat.x_test[:64])
    h = histogram(ys[HIST_LAYER])
    assert h.shape == (30,) and abs(h.sum() - 1) < 1e-9
    assert flat_bins("relu").sum() == 15 and flat_bins("sigmoid").sum() == 10   # left half; ends


@needs_mnist
def test_smoothness_claims():
    """Along the line between a 3 and an 8 the ReLU logit is kinked and the
    smooth networks' are not; the zoomed loss slice says the same."""
    from voxanim import actfun
    from voxanim.actfun_smoothness import CLASS_A, CLASS_B, LINE_N, ZOOM_N, first_of, loss_slices
    dat = actfun.data()
    runs = actfun.race_runs(dat=dat)
    i = _index()
    line = actfun.line_between(dat.x_test[first_of(dat, CLASS_A)],
                               dat.x_test[first_of(dat, CLASS_B)], LINE_N)
    rough, kinks = {}, {}
    for a, r in zip(actfun.ACTS, runs):
        z, p = actfun.along(r.spec, r.P, line, CLASS_B)
        assert p[0] < 0.1 and p[-1] > 0.9                   # every network reads 3 then 8
        rough[a.key] = actfun.roughness(z)
        kinks[a.key] = actfun.kinks(r.spec, r.P, line)
    assert kinks["relu"] > 10 and kinks["leaky_relu"] > 10
    for k in ("gelu", "silu", "tanh", "sigmoid", "elu"):
        assert rough["relu"] > 5 * rough[k] and rough["leaky_relu"] > 5 * rough[k]
    sl = loss_slices(runs, dat)
    assert sl["zoom"].shape == (7, ZOOM_N) and np.isfinite(sl["wide"]).all()
    rz = [actfun.roughness(v) for v in sl["zoom"]]
    for k in ("gelu", "silu", "tanh"):
        assert rz[i["relu"]] > 5 * rz[i[k]] and rz[i["leaky_relu"]] > 5 * rz[i[k]]
    # the wide slices are bowls: the minimum is inside, not at an end
    for v in sl["wide"]:
        assert 0 < v.argmin() < len(v) - 1


@needs_mnist
def test_depth_claims():
    """Twenty layers, no normalisation: tanh and ReLU train, sigmoid, GELU and
    SiLU stall (their activations collapse), ELU's loss overflows."""
    from voxanim import actfun
    from voxanim.actfun_depth import diverged, histogram
    runs = actfun.deep20_runs()
    _consistent(runs, actfun.DEEP20_SIZES, actfun.DEEP20)
    i = _index()
    acc = np.array([r.test_acc[-1] for r in runs])
    last = np.array([r.act_std[0][-1] for r in runs])        # layer 20's std, untrained
    assert acc[i["tanh"]] > 0.8 and acc[i["relu"]] > 0.5 and acc[i["leaky_relu"]] > 0.3
    for k in ("sigmoid", "gelu", "silu"):
        assert acc[i[k]] < 0.2
        assert diverged(runs[i[k]]) is None
    assert last[i["gelu"]] < 0.05 and last[i["silu"]] < 0.05 and last[i["elu"]] > 1.0
    assert 0.05 < last[i["sigmoid"]] < 0.2 and 0.1 < last[i["tanh"]] < 0.5
    assert diverged(runs[i["elu"]]) is not None and diverged(runs[i["elu"]]) < 50
    assert histogram(np.full(10, np.nan)).sum() == 0
    h = histogram(np.array([0.0, 0.0, 5.0, -5.0]))
    assert h[10] == 0.5 and h[-1] == 0.25 and h[0] == 0.25

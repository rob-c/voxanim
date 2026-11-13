"""The regularisation family: the new deepnet options and the tied
autoencoder against finite differences, the helpers, and every claim the
five animations make, as relative claims (CI retrains on another machine,
where the last bits differ)."""

from __future__ import annotations

import re

import numpy as np
import pytest
from conftest import fd_check, needs_mnist

from voxanim import deepnet as D
from voxanim import regul_ae as A

NAMES = ("l1-vs-l2", "l1-l2-weights", "regularisers", "regularisation-race",
         "regularised-autoencoder")


# ------------------------------------------------------------------ deepnet's new options


def _small(rng, **opts):
    spec = D.Spec((5, 4, 4, 3), "tanh", "xavier", **opts)
    x = rng.normal(0, 1, (6, 5))
    y = np.eye(3)[rng.integers(0, 3, 6)]
    P = D.init(spec, rng)
    for k in P:
        if k.startswith("b"):
            P[k] = rng.normal(0, 0.3, P[k].shape)
    return spec, x, y, P


@pytest.mark.parametrize("opts", [dict(l1=0.05), dict(label_smoothing=0.2),
                                  dict(l1=0.03, weight_decay=0.02, label_smoothing=0.1)])
def test_new_options_gradients(opts):
    """L1 (by its subgradient, away from 0), label smoothing and the lot
    together: every parameter's gradient matches finite differences."""
    rng = np.random.default_rng(4)
    spec, x, y, P = _small(rng, **opts)

    def loss(Q):
        return D.objective(spec, Q, D.forward(spec, Q, x), y)

    G, _ = D.backward(spec, P, D.forward(spec, P, x), y)
    fd_check(loss, P, G, list(P), tol=1e-5)


def test_old_cache_keys_are_unchanged():
    """The new options are absent from a spec's settings when off, so every
    run cached before them keeps its file."""
    s = D.Spec((2, 3, 2), "tanh").settings()
    assert s == dict(sizes=[2, 3, 2], activation="tanh", init="xavier", init_std=0.0, residual=0,
                     norm=None, dropout=0.0, weight_decay=0.0)
    on = D.Spec((2, 3, 2), "tanh", l1=0.1, l1_prox=True, max_norm=2.0, label_smoothing=0.1)
    assert {"l1", "l1_prox", "max_norm", "label_smoothing"} <= set(on.settings())


def test_proximal_l1_and_max_norm():
    """The soft threshold moves every weight t towards 0 and stops there; the
    proximal L1 term is left out of the gradient; max-norm caps each hidden
    unit's incoming weights and leaves the output layer alone."""
    w = np.array([-0.5, -0.05, 0.0, 0.02, 0.3])
    assert np.allclose(D.soft_threshold(w, 0.1), [-0.4, 0.0, 0.0, 0.0, 0.2])
    rng = np.random.default_rng(1)
    spec, x, y, P = _small(rng, l1=0.5, l1_prox=True)
    Gp, _ = D.backward(spec, P, D.forward(spec, P, x), y)
    G0, _ = D.backward(D.Spec(spec.sizes, "tanh", "xavier"), P, D.forward(spec, P, x), y)
    assert all(np.array_equal(Gp[k], G0[k]) for k in P)
    Q = {k: v.copy() for k, v in P.items()}
    D.constrain(spec, Q, lr=0.1)
    assert np.allclose(Q["W0"], D.soft_threshold(P["W0"], 0.05)) and (Q["W0"] == 0).any()
    spec2 = D.Spec(spec.sizes, "tanh", "xavier", max_norm=0.4)
    Q = {k: v * 3 for k, v in P.items()}
    D.constrain(spec2, Q, lr=0.1)
    for l in range(spec2.depth):
        assert D.unit_norms(Q["W%d" % l]).max() <= 0.4 + 1e-12
    assert np.array_equal(Q["W2"], 3 * P["W2"])
    with pytest.raises(ValueError):
        D.train(spec, None, steps=1, batch=1, lr=0.1, opt="adam")


def test_weight_snapshots_and_smoothing_record(tmp_path):
    """``weights`` records the named arrays at every recorded step; label
    smoothing's recorded losses stay the plain cross-entropy."""
    rng = np.random.default_rng(0)
    data = D.Data(rng.random((40, 6)), rng.integers(0, 3, 40), rng.random((10, 6)),
                  rng.integers(0, 3, 10), classes=3)
    spec = D.Spec((6, 8, 3), "relu", "he", label_smoothing=0.3)
    run = D.train(spec, data, steps=6, batch=8, lr=0.01, every=2, weights=("W0",),
                  cache_dir=str(tmp_path))
    assert run.weights["W0"].shape == (len(run.steps), 6, 8)
    assert np.allclose(run.weights["W0"][-1], run.P["W0"], atol=1e-6)
    f = D.forward(spec, run.P, data.x_train[:D.EVAL_N])
    assert run.train_loss[-1] == pytest.approx(D.cross_entropy(f["p"], data.onehot(data.y_train)))


# ------------------------------------------------------------------ the autoencoder


@pytest.mark.parametrize("kind,strength", [("plain", 0.0), ("sparse", 0.01), ("decay", 0.01),
                                           ("denoise", 0.3), ("contract", 0.05)])
def test_autoencoder_gradients(kind, strength):
    """The tied autoencoder's objective under each regulariser: gradients
    match finite differences (the denoiser's corruption is in the input)."""
    rng = np.random.default_rng(3)
    x = rng.random((5, 784)) * (rng.random((5, 784)) > 0.6)
    spec = A.AESpec(kind, strength, code=7)
    P = A.init(spec, rng)
    P["be"] = rng.normal(0, 0.3, 7)
    P["bd"] = rng.normal(0, 0.3, 784)
    x_in = A.corrupt(rng, x, strength) if kind == "denoise" else x

    def loss(Q):
        return A.objective(spec, Q, A.forward(Q, x_in), x)

    fd_check(loss, P, A.backward(spec, P, A.forward(P, x_in), x), list(P), tol=1e-5)


def test_relu_jacobian():
    """``|dh/dx|_F^2`` of a ReLU code is the squared weights of the units that
    are on (checked against the Jacobian by finite differences)."""
    rng = np.random.default_rng(2)
    P = A.init(A.AESpec(code=9), rng)
    P["be"] = rng.normal(0, 0.5, 9)
    x = rng.random((1, 784))
    z, h = A.encode(P, x)
    e = 1e-6
    J = np.empty((9, 784))
    for i in range(784):
        d = np.zeros((1, 784))
        d[0, i] = e
        J[:, i] = (A.encode(P, x + d)[1] - A.encode(P, x - d)[1])[0] / (2 * e)
    assert A.jacobian_sq(P, z)[0] == pytest.approx((J ** 2).sum(), rel=1e-6)
    assert 0 < (h > 0).sum() < 9


# ------------------------------------------------------------------ helpers


def test_contours_and_mosaics():
    """Marching squares traces a circle at its radius; a mosaic packs its
    tiles with gaps; an exactly-zero weight is blank paper, a tiny one is not."""
    from voxanim.regul_plots import ZERO, contour_segments, mosaic, signed_texture
    xs, ys = np.linspace(-2, 2, 41), np.linspace(-1.5, 2.5, 41)
    F = xs[None, :] ** 2 + ys[:, None] ** 2
    pts = np.array(contour_segments(F, xs, ys, 1.0)).reshape(-1, 2)
    assert len(pts) > 100 and np.allclose(np.hypot(pts[:, 0], pts[:, 1]), 1.0, atol=0.01)
    tex = mosaic([np.zeros((3, 2, 3))] * 5, cols=2, gap=1)
    assert tex.shape == (3 * 3 + 2, 2 * 2 + 1, 3)
    t = signed_texture(np.array([0.0, 1e-9, -0.5, 0.5]), 0.5, side=2)
    assert np.allclose(t[0, 0], ZERO) and not np.allclose(t[0, 1], ZERO)


def test_fista_matches_newton_and_zeroes():
    """FISTA without a penalty finds the logistic optimum; with L1 it puts
    weights at exactly 0, with L2 it does not."""
    from voxanim import regul as R
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, (400, 5))
    y = (x @ np.array([2.0, -1.0, 0.3, 0.0, 0.05]) + 0.2 * rng.normal(0, 1, 400) > 0) * 1.0
    w, b = R.fista(x, y, 0.0, "l2", np.zeros(5), 0.0, iters=20000)
    p = 1 / (1 + np.exp(-(x @ w + b)))
    assert np.abs(x.T @ (p - y)).max() / 400 < 1e-5
    w1, _ = R.fista(x, y, 0.05, "l1", w, b)
    w2, _ = R.fista(x, y, 0.05, "l2", w, b)
    assert (w1 == 0).sum() >= 2 and (w2 == 0).sum() == 0


# ------------------------------------------------------------------ the claims


@needs_mnist
def test_geometry_claims():
    """The plane: L1 zeroes w6 well before w9 and L2 zeroes neither; the
    sixteen-weight paths: lasso drops weights one by one, ridge keeps all."""
    from voxanim import regul as R
    pl = R.plane()
    assert pl.loss.min() == pytest.approx(pl.loss0, abs=1e-3)
    assert 0.8 < pl.acc0 < 0.9 and pl.w0[0] < 0 < pl.w0[1] and abs(pl.w0[0]) < pl.w0[1]
    z6, z9 = R.first_zero(pl.l1, 0), R.first_zero(pl.l1, 1)
    assert z6 < z9 / 2 < np.inf
    assert np.all(pl.l2.w != 0) and abs(pl.l2.w[-1, 0]) < 0.01
    # the path is the penalised optimum: the data loss rises along it
    assert np.all(np.diff(pl.l1.train_loss) >= -1e-9) and np.all(np.diff(pl.l2.train_loss) >= -1e-9)
    pa = R.paths()
    nz = pa.l1.nonzero()
    assert nz[0] == 16 and nz[-1] == 0 and len(np.unique(nz)) >= 8
    assert np.all(pa.l2.nonzero() == 16)
    from voxanim.regul_geometry import KEEP_WITHIN, L1VsL2
    anim = L1VsL2(pl, pa)
    k = anim.keep_i
    assert pa.l1.test_acc[k] >= pa.acc0 - KEEP_WITHIN and nz[k] <= 12
    assert (D.soft_threshold(pa.w0, 1.0) == 0).sum() >= 6     # the one-weight phase at λ = 1


@needs_mnist
def test_weights_claims():
    """The MLP's first layer: the plain network never moves the never-lit
    pixels' weights, L2 shrinks them, L1 zeroes them; proximal L1 makes exact
    zeros where its subgradient makes none; every penalty costs a little."""
    from voxanim import regul as R
    dat = R.weight_data()
    runs = dict(zip(R.W_LABELS, R.weight_runs(dat=dat)))
    never = R.never_lit(dat)
    assert 40 < never.sum() < 100
    plain, l2, l1, en, slope = (runs[k] for k in R.W_LABELS)
    assert np.array_equal(plain.P["W0"][never], plain.P0["W0"][never])
    left = np.abs(l2.P["W0"][never]).sum() / np.abs(l2.P0["W0"][never]).sum()
    assert 0.0 < left < 0.4 and np.all(l2.P["W0"] != 0)
    assert R.cut_pixels(l1.P["W0"]).sum() > 100
    assert (R.cut_pixels(l1.P["W0"]) & never).sum() > 0.8 * never.sum()
    assert (l1.P["W0"] == 0).mean() > 0.3 and (en.P["W0"] == 0).mean() > 0.2
    assert (slope.P["W0"] == 0).sum() == 0
    near = np.abs(slope.P["W0"]) <= R.W_TRAIN["lr"] * slope.spec.l1
    assert near.mean() > 0.3
    assert abs(slope.test_acc[-1] - l1.test_acc[-1]) < 0.01
    assert np.linalg.norm(slope.P["W0"]) == pytest.approx(np.linalg.norm(l1.P["W0"]), rel=0.05)
    acc = {k: r.test_acc[-1] for k, r in runs.items()}
    assert acc["none"] == max(acc.values()) and min(acc.values()) > acc["none"] - 0.02
    assert l1.weights["W0"].shape[0] == len(l1.steps)


@needs_mnist
def test_race_claims():
    """Plain memorises 1,000 digits and its test loss climbs; input noise and
    label smoothing beat it on accuracy, L2 and L1 on test loss; the sweeps
    run from overfitting (training accuracy 100%) to underfitting."""
    from voxanim import regul as R
    dat = R.race_data()
    plain, chosen = R.race_runs(dat=dat)
    assert plain.train_loss[-1] < 1e-3 and plain.test_loss[-1] > 1.3 * plain.test_loss.min()
    assert plain.steps[plain.best] < 500
    got = {reg.key: run for reg, _s, run in chosen}
    for k in ("noise", "smooth"):
        assert got[k].test_acc[-1] > plain.test_acc[-1] + 0.007
    for k in ("l2", "l1"):
        assert got[k].test_loss[-1] < plain.test_loss[-1] - 0.05
    assert got["smooth"].train_loss[-1] >= -np.log(1 - got["smooth"].spec.label_smoothing
                                                   + got["smooth"].spec.label_smoothing / 10)
    sweeps = R.race_sweeps(dat=dat)
    for k in ("l2", "l1", "noise"):
        sw = sweeps[k]
        assert sw.train_acc[0] > 0.99 and sw.train_acc[-1] < sw.train_acc[0] - 0.03
        assert 0 < sw.best < len(sw.reg.grid) - 1 or k == "l1"
        assert sw.val_acc[-1] < sw.val_acc.max()
    assert sweeps["l1"].val_acc[-1] < 0.2


@needs_mnist
def test_regularisers_claims():
    """Dropout: the dropout-trained network survives losing half its hidden
    units far better; input noise: it reads noisy digits; max-norm: the cap
    holds; label smoothing: the logit gap sits under the smoothed target's."""
    from voxanim import regul as R
    dat = R.race_data()
    plain, chosen = R.race_runs(dat=dat)
    got = {reg.key: (s, run) for reg, s, run in chosen}
    c, mn = got["maxnorm"]
    for l in range(mn.spec.depth):
        assert D.unit_norms(mn.P["W%d" % l]).max() <= c + 1e-9
    assert np.median(D.unit_norms(plain.P["W0"])) > 0.9 * c
    eps, sm = got["smooth"]
    x = dat.x_test
    assert np.median(R.logit_gap(sm, x)) < R.smoothing_gap(eps) < np.median(R.logit_gap(plain, x))
    assert np.median(R.confidence(sm, x)) < np.median(R.confidence(plain, x))
    sd, nz = got["noise"]
    xn = x + np.random.default_rng(0).normal(0, sd, x.shape)

    def acc(run, xs):
        return (D.forward(run.spec, run.P, xs)["p"].argmax(1) == dat.y_test).mean()

    assert acc(nz, xn) > acc(plain, xn) + 0.1
    p, dr = got["dropout"]
    rng = np.random.default_rng(0)
    masks = [rng.random((len(x), w)) >= p for w in plain.spec.sizes[1:-1]]

    def dropped(run):
        f = D.forward(D.Spec(run.spec.sizes, "relu", "he", dropout=p), run.P, x, train=True,
                      masks=masks)
        return (f["p"].argmax(1) == dat.y_test).mean()

    assert dropped(dr) > dropped(plain) + 0.05


@needs_mnist
def test_autoencoder_claims():
    """Each regulariser holds the clean error near the budget and wins its own
    column: L1 the sparsest code, denoising the noisy digits, weight decay and
    the contractive penalty the steadiest code, alike."""
    from voxanim import regul as R
    runs = R.ae_runs()
    m = dict(zip(R.AE_LABELS, R.ae_measures(runs)))
    plain, l1, wd, dn, ct = (m[k] for k in R.AE_LABELS)
    for v in (l1, wd, dn, ct):
        assert abs(v["error"] - R.AE_BUDGET) < 0.1 * R.AE_BUDGET
    assert plain["error"] < 0.5 * R.AE_BUDGET
    assert l1["active"] < 0.5 * plain["active"]
    assert dn["noisy_error"] == min(v["noisy_error"] for v in m.values())
    assert dn["noisy_error"] < 0.7 * plain["noisy_error"]
    for k in ("shift", "jac"):
        assert wd[k] == pytest.approx(ct[k], rel=0.05)
        assert wd[k] < 0.6 * plain[k]
    assert l1["shift"] > plain["shift"] and ct["active"] > 0.95
    for r in runs:
        assert r.test_error[-1] < r.test_error[0] and r.W_snaps.shape[0] == len(r.snap_steps)


@needs_mnist
@pytest.mark.parametrize("name", NAMES)
def test_words_are_finite(name):
    """No subtitle ever prints a nan or an inf, at any time."""
    from voxanim import presets
    anim = presets.build(name)
    for t in np.linspace(0, anim.duration, 90):
        anim.scene(float(t))
        assert not re.search(r"\b(nan|inf)\b", anim.subtitle.lower()), (t, anim.subtitle)

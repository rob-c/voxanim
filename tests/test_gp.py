"""Gaussian processes: the kernels, the posterior, the marginal likelihood and
its gradient, expected improvement, and the claims the five animations make."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from conftest import fd_check

from voxanim import gp, gp_bayesopt, gp_hyper, gp_kernels, gp_posterior, gp_prior, presets

KERNELS = [
    gp.RBF(variance=1.3, lengthscale=1.7),
    gp.Matern32(variance=0.8, lengthscale=2.1),
    gp.Periodic(variance=1.1, lengthscale=0.9, period=3.3),
    gp.Linear(variance=0.2, offset=4.0),
    gp.Periodic(variance=1.1, lengthscale=0.9, period=3.3) + gp.RBF(variance=0.5,
                                                                  lengthscale=2.0),
]


def _data(n=7, seed=0):
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(0, 10, n))
    return X, np.sin(X) + 0.3 * X - 1.5 + rng.normal(0, 0.1, n)


# ------------------------------------------------------------------ kernels


@pytest.mark.parametrize("k", KERNELS, ids=[k.NAME for k in KERNELS])
def test_kernel_matrices_are_symmetric_and_positive_semidefinite(k):
    x = np.random.default_rng(1).uniform(0, 10, 30)
    K = k(x, x)
    assert np.allclose(K, K.T)
    assert np.linalg.eigvalsh(K).min() > -1e-8
    # the sum, scaling and blending are what they say
    assert np.allclose((k + k)(x, x), 2 * K)
    assert np.allclose(gp.scaled(k, 0.5)(x, x), 0.5 * K)
    other = gp.RBF(variance=2.0, lengthscale=0.5)
    assert np.allclose(gp.blend(k, other, 0.25)(x, x), 0.75 * K + 0.25 * other(x, x))
    assert gp.blend(k, other, 0.0) is k and gp.blend(k, other, 1.0) is other


def test_kernel_derivatives_match_finite_differences():
    x = np.random.default_rng(2).uniform(0, 10, 6)
    for k in KERNELS:
        dK = k.dK(x, x)
        for name in k.params:
            e = 1e-6
            plus = k.set_params({name: k.params[name] + e})(x, x)
            minus = k.set_params({name: k.params[name] - e})(x, x)
            assert np.allclose(dK[name], (plus - minus) / (2 * e), atol=1e-6), (k.NAME, name)


# ------------------------------------------------------------------ posterior


def test_posterior_matches_the_direct_formula():
    X, y = _data()
    xs = np.linspace(0, 10, 40)
    k = gp.RBF(variance=1.0, lengthscale=1.5)
    noise = np.full(len(X), 0.04)
    noise[2] = 0.5                                   # one noisier point
    post = gp.posterior(k, X, y, xs, noise)
    Kinv = np.linalg.inv(k(X, X) + np.diag(noise))
    assert np.allclose(post.mu, k(xs, X) @ Kinv @ y, atol=1e-6)
    assert np.allclose(post.cov, k(xs, xs) - k(xs, X) @ Kinv @ k(X, xs), atol=1e-6)
    assert np.allclose(post.alpha, Kinv @ y, atol=1e-6)
    # no data: the prior
    pri = gp.posterior(k, [], [], xs, 0.1)
    assert np.allclose(pri.mu, 0) and np.allclose(pri.cov, k(xs, xs))
    # samples: the same draws give exact draws from N(mu, cov)
    z = np.random.default_rng(0).standard_normal((20000, len(xs)))
    f = post.samples(z)
    assert np.abs(f.mean(0) - post.mu).max() < 0.03
    assert np.abs(np.cov(f.T) - post.cov).max() < 0.03


# ------------------------------------------------------------------ marginal likelihood


def test_log_marginal_value_and_gradient():
    X, y = _data()
    for k in KERNELS:
        noise = 0.09
        ev = gp.log_marginal(k, X, y, noise)
        K = k(X, X) + noise * np.eye(len(X))
        direct = -0.5 * y @ np.linalg.solve(K, y) - 0.5 * np.linalg.slogdet(K)[1] \
            - 0.5 * len(X) * np.log(2 * np.pi)
        assert abs(ev.value - direct) < 1e-6
        assert abs(ev.fit + ev.complexity + ev.const - ev.value) < 1e-12
        names = list(k.params) + ["noise"]
        P = {n: np.array([v]) for n, v in dict(k.params, noise=noise).items()}
        G = {n: np.array([ev.grad[n]]) for n in names}

        def loss(Q, k=k):
            p = {n: float(v[0]) for n, v in Q.items()}
            return gp.log_marginal(k.set_params(p), X, y, p["noise"]).value
        fd_check(loss, P, G, names, e=1e-6, tol=1e-5)


def test_ascent_climbs():
    X, y = _data()
    ks, nz, evs = gp.ascend(gp.RBF(variance=1.0, lengthscale=0.5), X, y, 0.5,
                            ["lengthscale", "noise"], 0.02, 100)
    vals = [e.value for e in evs]
    assert len(ks) == len(nz) == len(vals) == 101
    assert all(b >= a - 1e-9 for a, b in pairwise(vals))
    assert vals[-1] > vals[0] + 1.0
    assert all(n > 0 for n in nz) and all(k.p["lengthscale"] > 0 for k in ks)


# ------------------------------------------------------------------ expected improvement


def test_expected_improvement_matches_monte_carlo():
    z = np.random.default_rng(0).standard_normal(1_000_000)
    for mu, sd, best, xi in ((0.3, 0.5, 0.0, 0.0), (-1.0, 0.2, 0.0, 0.0), (0.0, 1.0, 1.0, 0.0),
                             (0.5, 0.3, 0.4, 0.1)):
        mc = np.maximum(mu + sd * z - best - xi, 0).mean()
        assert abs(gp.expected_improvement(mu, sd, best, xi) - mc) < 3e-3
    ei = gp.expected_improvement(np.array([0.0, 2.0]), np.array([0.0, 0.0]), 1.0)
    assert np.allclose(ei, [0.0, 1.0])              # no uncertainty: the plain improvement
    assert np.all(gp.expected_improvement(np.linspace(-2, 2, 9), 0.5, 0.0) >=
                  np.maximum(np.linspace(-2, 2, 9), 0.0) - 1e-12)
    from math import erf
    zz = np.linspace(-5, 5, 101)
    assert np.abs(gp.norm_cdf(zz) - 0.5 * (1 + np.array([erf(v / np.sqrt(2)) for v in zz]))
                  ).max() < 1e-6


# ------------------------------------------------------------------ the animations' claims


def test_durations():
    for name in ("gp-prior", "gp-posterior", "gp-kernels", "gp-hyperparameters",
                 "gp-bayesian-optimisation"):
        assert 35.0 <= presets.build(name).duration <= 55.0, name


def test_prior_animation_claims():
    a = gp_prior.PriorAnimation()
    # the hyperparameters visit the values the subtitles quote, in order
    ends = [a._value(t1, a.l_moves, gp_prior.L0) for _t0, t1, _a, _c in a.l_moves]
    assert np.allclose(ends, gp_prior.L_PATH)
    ends = [a._value(t1, a.var_moves, gp_prior.VAR0) for _t0, t1, _a, _c in a.var_moves]
    assert np.allclose(ends, gp_prior.VAR_PATH)
    # the band is +-2 sqrt(variance): the prior's sd is constant
    pri = gp.prior(a.kernel(0.7, 2.25), a.xs)
    assert np.allclose(pri.sd, 1.5) and np.allclose(pri.mu, 0.0)
    # "95% of samples lie within +-2 sd": many seeded draws
    z = np.random.default_rng(0).standard_normal((4000, len(a.xs)))
    assert 0.94 < np.mean(np.abs(pri.samples(z)) < 3.0) < 0.96
    # the matrix fills row by row and is the kernel over the grid
    K = a.kernel(1.0, 1.0)(a.grid, a.grid)
    assert np.allclose(K, K.T) and np.allclose(np.diag(K), 1.0)


def test_posterior_animation_claims():
    a = gp_posterior.PosteriorAnimation()
    # noise-free: the mean passes through every observation and the band closes there
    f = a._frame(a.Tm["formula"][0] + 0.1, 0.0)
    assert np.all(f.noise == gp_posterior.NOISE_FREE)
    assert np.allclose(f.y, gp_posterior.truth(a.X))
    post = a.posterior(f, a.X)
    assert np.abs(post.mu - f.y).max() < 1e-3
    assert post.sd.max() < 1e-2
    # the joint matrix's blocks are the formula's matrices
    J, mask = a._joint(f)
    n = len(a.X)
    pg = a.posterior(f, a.grid)
    order = a.order
    assert np.allclose(J[:n, :n], pg.K[np.ix_(order, order)])
    assert np.allclose(J[:n, n:], pg.ks[order])
    assert np.allclose(J[n:, n:], pg.kss) and mask.all()
    # with noise: the band stays open and the mean need not pass through
    f = a._frame(a.Tm["noise"][1] - 0.1, 0.0)
    assert np.allclose(f.noise, gp_posterior.NOISE_SD ** 2)
    post = a.posterior(f, a.X)
    assert post.sd.min() > 0.1 and np.abs(post.mu - f.y).max() > 0.05
    # a point enters with no influence: the posterior just before equals the one without it
    t0 = a.obs_t[3]
    before = a.posterior(a._frame(t0 - 0.01, 0.0), a.xs)
    entering = a.posterior(a._frame(t0 + gp_posterior.POP_S + 0.001, 0.0), a.xs)
    assert np.abs(before.mu - entering.mu).max() < 5e-3


def test_kernels_animation_claims():
    a = gp_kernels.KernelsAnimation()
    names = [k[0] for k in gp_kernels.KERNELS]
    lml = {n: ev.value for n, (_k, ev) in zip(names, a.fits)}
    assert max(lml, key=lml.get) == "sum" and min(lml, key=lml.get) == "linear"
    X, y = a.X, a.y
    noise = a.noise
    # fitting never lowered the evidence, and each fitted kernel beats its start
    for (name, _label, make, free), (_k, ev) in zip(gp_kernels.KERNELS, a.fits):
        _ks, _nz, evs = gp.ascend(make(), X, y, noise, free, gp_kernels.FIT_LR,
                                  gp_kernels.FIT_STEPS)
        vals = [e.value for e in evs]
        assert all(b >= a_ - 1e-9 for a_, b in pairwise(vals)), name
        assert abs(vals[-1] - ev.value) < 1e-6 and ev.value > vals[0], name
    fits = dict(zip(names, (k for k, _ev in a.fits)))
    xs = a.xs
    # RBF: far from the data the mean returns to 0 and the band to +-2 sd
    post = gp.posterior(fits["RBF"], X, y, xs, noise)
    assert abs(post.mu[-1]) < 0.05 and abs(post.sd[-1] ** 2 - fits["RBF"].p["variance"]) < 0.02
    # periodic: what it saw it expects again, every period, as sure as at the data
    post = gp.posterior(fits["periodic"], X, y, xs, noise)
    p = gp_kernels.PERIOD
    mu_a, sd_a = np.interp(3.0, xs, post.mu), np.interp(3.0, xs, post.sd)
    mu_b, sd_b = np.interp(3.0 + p, xs, post.mu), np.interp(3.0 + p, xs, post.sd)
    assert abs(mu_a - mu_b) < 1e-3 and abs(sd_a - sd_b) < 1e-3
    # linear: only straight lines
    post = gp.posterior(fits["linear"], X, y, xs, noise)
    assert np.abs(np.diff(post.mu, 2)).max() < 1e-6
    # the morph between kernels is a kernel (a sum of scaled parts)
    f = a._frame(a.seg_t[1] + 0.5 * gp_kernels.MORPH, 0.0)
    k = a.kernel_at(f)
    assert isinstance(k, gp.Sum) and 0 < f.u < 1


def test_hyperparameters_animation_claims():
    a = gp_hyper.HyperAnimation()
    X, y = a.X, a.y
    evs = [a.evidence(l, sd) for l, sd in a.path]
    vals = [e.value for e in evs]
    # the walk climbs, from too smooth to the landscape's top
    assert tuple(a.path[0]) == gp_hyper.START
    assert all(b >= a_ - 1e-9 for a_, b in pairwise(vals))
    assert vals[-1] >= a.zmax - 0.2
    i, j = np.unravel_index(np.argmax(a.Z), a.Z.shape)
    ls = np.exp(np.linspace(np.log(gp_hyper.L_RANGE[0]), np.log(gp_hyper.L_RANGE[1]),
                            gp_hyper.N_L))
    ns = np.exp(np.linspace(np.log(gp_hyper.N_RANGE[0]), np.log(gp_hyper.N_RANGE[1]),
                            gp_hyper.N_N))
    assert abs(np.log(ls[i] / a.path[-1, 0])) < 0.25 and abs(np.log(ns[j] / a.path[-1, 1])) < 0.25
    # the trade-off the subtitles describe: complexity rises, data fit drops, the sum climbs
    assert evs[-1].complexity > evs[0].complexity + 5
    assert evs[-1].fit < evs[0].fit - 1
    assert evs[-1].value > evs[0].value + 5
    # the optimum is sensible: around the true noise (0.3) and a lengthscale of order 1
    assert 0.15 < a.path[-1, 1] < 0.45 and 1.0 < a.path[-1, 0] < 3.0
    # the landscape is the evidence on the grid, and the gradient is the ascent's direction
    assert abs(a.Z[i, j] - a.evidence(ls[i], ns[j]).value) < 1e-9
    g = a.log_gradient(*a.path[0])
    step = np.log(a.path[1] / a.path[0])
    assert np.allclose(step, gp_hyper.LR * g)
    assert len(X) == len(y) == gp_hyper.N_OBS


def test_bayesian_optimisation_claims():
    a = gp_bayesopt.BayesOptAnimation()
    X, y = a.X, a.y
    n0 = len(gp_bayesopt.X_START)
    assert len(X) == n0 + gp_bayesopt.ROUNDS
    assert np.allclose(y, gp_bayesopt.unknown(X))
    # every pick is the maximum of EI under the posterior of the evaluations before it
    for i in range(n0, len(X)):
        post = gp.posterior(gp_bayesopt.kernel(), X[:i], y[:i], a.xs, gp_bayesopt.NOISE)
        ei = gp.expected_improvement(post.mu, post.sd, y[:i].max(), gp_bayesopt.XI)
        assert X[i] == a.xs[int(np.argmax(ei))]
    # explore and exploit both happen
    kinds = [gp_bayesopt.explores(X, i) for i in range(n0, len(X))]
    assert any(kinds) and not all(kinds)
    # the best evaluation reaches the true maximum
    assert abs(y.max() - a.true_max) < 0.01 and abs(X[int(np.argmax(y))] - a.true_x) < 0.1
    # the posterior on show is exact for the evaluations in: before a round's
    # update the new point has no influence
    t0 = a.round_t[2] + gp_bayesopt.ACQ_S + gp_bayesopt.FLY_S
    before = a.posterior(a._frame(t0 - 0.01, 0.0))
    entering = a.posterior(a._frame(t0 + 0.001, 0.0))
    assert np.abs(before.mu - entering.mu).max() < 5e-3

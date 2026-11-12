"""The diffusion model (:mod:`voxanim.diffusion`): the schedule's identities,
the denoiser's gradients, the reverse step against the DDPM formulas, and the
numbers the ``diffusion`` animation quotes."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fd_check, needs_mnist

from voxanim import diffusion as D


@pytest.mark.parametrize("kind", ["cosine", "linear"])
def test_schedule_identities(kind):
    beta, alpha, abar = D.schedule(D.T_STEPS, kind)
    T = D.T_STEPS
    assert beta.shape == alpha.shape == abar.shape == (T + 1,)
    assert beta[0] == 0 and abar[0] == 1
    assert (beta[1:] > 0).all() and (beta <= D.BETA_MAX).all()
    assert np.allclose(alpha, 1 - beta)
    assert np.allclose(abar, np.cumprod(1 - beta))                 # abar_t = prod (1 - beta_s)
    assert (np.diff(abar) < 0).all()                                # strictly falling
    if kind == "cosine":
        f = np.cos((np.arange(T + 1) / T + D.COSINE_S) / (1 + D.COSINE_S) * np.pi / 2) ** 2
        assert np.allclose(abar[:-1], f[:-1] / f[0], atol=1e-9)    # abar_t = f(t) / f(0)
        assert abar[T] < 1e-6                                       # ends at pure noise
        assert 0.45 < abar[T // 2] < 0.55                           # and halfway at t = T / 2
    else:
        assert np.isclose(beta[1], D.LINEAR_BETA[0]) and np.isclose(beta[T], D.LINEAR_BETA[1])


def test_forward_walk_has_the_closed_form_marginals():
    """Stepping x_t = sqrt(1 - beta) x_{t-1} + sqrt(beta) z reaches, in
    distribution, sqrt(abar_t) x_0 + sqrt(1 - abar_t) eps."""
    beta, _alpha, abar = D.schedule()
    rng = np.random.default_rng(0)
    x0 = np.full((4000, 784), 0.6)
    walk = D.forward_walk(x0, rng, beta)
    for t in (10, 60, 120, 200):
        assert np.isclose(walk[t].mean(), np.sqrt(abar[t]) * 0.6, atol=0.01)
        assert np.isclose(walk[t].var(), 1 - abar[t], rtol=0.03)
    # the closed form itself
    eps = rng.standard_normal((3, 784))
    t = np.array([1, 60, 200])
    xt = D.noised(x0[:3], t, eps, abar)
    assert np.allclose(xt, np.sqrt(abar[t])[:, None] * 0.6 + np.sqrt(1 - abar[t])[:, None] * eps)


def test_time_embedding():
    e = D.time_embedding(np.array([0, 7, 200]))
    assert e.shape == (3, D.EMBED)
    assert np.allclose(e[0, :D.EMBED // 2], 0) and np.allclose(e[0, D.EMBED // 2:], 1)
    assert np.allclose(e[:, :D.EMBED // 2] ** 2 + e[:, D.EMBED // 2:] ** 2, 1)
    assert not np.allclose(e[1], e[2])                                  # steps are told apart


def test_denoiser_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    P = D.init(rng)
    x = rng.normal(0, 1, (3, 784))
    t = np.array([1, 60, 200])
    eps = rng.standard_normal((3, 784))
    L, g = D.loss_grads(P, D.forward(P, x, t), eps)
    assert set(g) == set(P)
    assert np.isclose(L, np.mean((D.forward(P, x, t)["eps"] - eps) ** 2))
    fd_check(lambda Q: D.loss_grads(Q, D.forward(Q, x, t), eps)[0], P, g, list(P), tol=1e-4)


def test_reverse_step_is_the_ddpm_posterior():
    """The clipped-x0 posterior mean equals DDPM's eps form of the mean,
    (x_t - beta_t / sqrt(1 - abar_t) eps) / sqrt(alpha_t), when x0 is not
    clipped; the variance is beta~_t; and the x0 estimate inverts the forward
    closed form."""
    sched = beta, alpha, abar = D.schedule()
    rng = np.random.default_rng(0)
    P = D.init(rng)
    x0 = np.clip(rng.normal(0, 0.5, (2, 784)), -1, 1)
    for t in (1, 60, 199, 200):
        eps = rng.standard_normal(x0.shape)
        xt = D.noised(x0, np.array([t, t]), eps, abar)
        # exact noise in -> exact x0 out
        assert np.allclose(D.predict_x0(xt, eps, t, abar, clip=False), x0)
        # the two forms of the posterior mean agree (no clipping)
        eps_hat = D.forward(P, xt, np.array([t, t]))["eps"]
        x0_hat = D.predict_x0(xt, eps_hat, t, abar, clip=False)
        mean, var = D.posterior(xt, x0_hat, t, beta, alpha, abar)
        eps_form = (xt - beta[t] / np.sqrt(1 - abar[t]) * eps_hat) / np.sqrt(alpha[t])
        assert np.allclose(mean, eps_form)
        assert np.isclose(var, beta[t] * (1 - abar[t - 1]) / (1 - abar[t]))
        # the posterior of the true x0 is centred where the forward walk came from, on average
        m_true, _ = D.posterior(xt, x0, t, beta, alpha, abar)
        coef = np.sqrt(abar[t - 1]) * beta[t] / (1 - abar[t]) + \
            np.sqrt(alpha[t]) * (1 - abar[t - 1]) / (1 - abar[t]) * np.sqrt(abar[t])
        assert np.isclose(coef, np.sqrt(abar[t - 1]))        # the x0 coefficient, overall
        assert np.allclose(m_true - np.sqrt(abar[t - 1]) * x0,
                           np.sqrt(alpha[t]) * (1 - abar[t - 1]) / (1 - abar[t])
                           * np.sqrt(1 - abar[t]) * eps)
        # one reverse step: mean + sigma z, x0 clipped to the image range
        z = rng.standard_normal(x0.shape)
        x_prev, x0c, e = D.reverse_step(P, xt, t, z, sched)
        assert np.allclose(e, eps_hat) and np.abs(x0c).max() <= 1
        mc, vc = D.posterior(xt, x0c, t, beta, alpha, abar)
        assert np.allclose(x_prev, mc + np.sqrt(vc) * z)
    # the last step adds no noise, and the chain is recorded step for step
    xs, x0s = D.sample(P, np.random.default_rng(1), 2, sched, record=True)
    assert xs.shape == x0s.shape == (D.T_STEPS + 1, 2, 784)
    assert np.allclose(x0s[0], xs[0])
    again = D.sample(P, np.random.default_rng(1), 2, sched)
    assert np.allclose(again, xs[0])


@needs_mnist
def test_trained_denoiser_and_the_animation_numbers():
    """The cached run: the loss falls, the trained model beats predicting
    zero noise by a wide margin, and the subtitles' numbers are the trace's."""
    tr = D.trace()
    assert tr.T == 200 and tr.losses.shape == (D.EPOCHS * tr.steps_per_epoch,)
    first = float(tr.losses[0])
    last = float(tr.losses[-200:].mean())
    # the untrained net passes x_t through its skip (gain 1): the loss is
    # E|x_t - eps|^2 = abar_t E x0^2, well below 1 already; then it falls to 0.16
    assert 0.5 < first < 1.0
    assert 0.14 < last < 0.2 and 0.14 < tr.test_loss < 0.2     # "0.69 -> 0.16 (test 0.16)"
    assert abs(tr.test_loss - last) < 0.02       # no overfitting to speak of
    # the worked example is what the subtitle says it is
    assert tr.example_t == D.EXAMPLE_T
    assert np.isclose(tr.example_loss, np.mean((tr.example_eps_hat - tr.example_eps) ** 2))
    assert tr.example_loss < 0.3
    # the forward walk ends in noise: unit variance, no digit left
    assert np.isclose(tr.walk[-1].var(), 1.0, atol=0.08)
    assert abs(np.corrcoef(tr.walk[-1].ravel(), tr.digits.ravel())[0, 1]) < 0.05
    # the reverse chains end inside the image range and are not the noise they started from
    assert np.abs(tr.chain[0]).max() <= 1.0 + 1e-9
    assert tr.chain[0].std() > 0.35               # ink and paper (a test digit: 0.6), not grey mush
    assert np.isclose(tr.chain[0], tr.chain_x0[0]).all()
    # the samples are not copies of training digits (nearest-neighbour distance is
    # that of a typical test digit to the training set, not ~0)
    from voxanim.mnist import load
    xtr = load()[0][:20000]
    s = D.to_image(tr.chain[0])
    d_samp = np.sqrt(((s[:, None, :] - xtr[None, :, :]) ** 2).sum(-1)).min(1)
    assert (d_samp > 2.0).all()


@needs_mnist
def test_animation_claims():
    from voxanim import layout
    from voxanim.diffusion_anim import DiffusionAnimation
    anim = DiffusionAnimation(D.trace())
    assert 40 <= anim.duration <= 60
    assert anim.curve.min() < 0.2 and anim.curve[0] > 0.3
    for t in np.linspace(0, anim.t_done, 48):
        anim.scene(float(t))
        lines = [ln for part in anim.subtitle.split("\n")
                 for ln in layout.wrap(part, layout.SUB_SIZE, 1 - 2 * layout.SAFE)]
        assert len(lines) <= 2, anim.subtitle

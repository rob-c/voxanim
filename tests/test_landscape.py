"""The optimisation family: the landscape model's gradients, the optimisers'
update rules against hand-computed steps, and the claims the five
animations' subtitles make."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fd_check, needs_mnist

from voxanim import landscape as LS
from voxanim.optim import SGD, Adam, AdamW, Momentum, RMSProp

# ------------------------------------------------------------------ the optimisers (no data)


def _step(cls, w, g, **kw):
    """One ``cls`` step on parameter ``w`` with gradient ``g``; returns the new ``w``."""
    P = {"w": np.array(w, np.float64)}
    opt = cls(P, **kw)
    opt.step(P, {"w": np.array(g, np.float64)})
    return opt, P["w"]


def test_sgd_momentum_rmsprop_match_hand_computed_steps():
    """Each update rule, checked on two steps against the formula by hand."""
    g1, g2 = np.array([0.5, -2.0]), np.array([0.25, -1.0])
    # SGD: p - lr g
    opt, w = _step(SGD, [1.0, 1.0], g1, lr=0.1)
    assert np.allclose(w, [0.95, 1.2])
    # momentum: m = beta m + g; p - lr m
    P = {"w": np.array([1.0, 1.0])}
    opt = Momentum(P, lr=0.1, beta=0.9)
    opt.step(P, {"w": g1})
    assert np.allclose(P["w"], [1.0, 1.0] - 0.1 * g1)
    opt.step(P, {"w": g2})
    m2 = 0.9 * g1 + g2
    assert np.allclose(P["w"], [1.0, 1.0] - 0.1 * g1 - 0.1 * m2)
    # RMSProp: v = beta v + (1 - beta) g^2; p - lr g / (sqrt(v) + eps)
    P = {"w": np.array([1.0, 1.0])}
    opt = RMSProp(P, lr=0.1, beta=0.9, eps=1e-8)
    opt.step(P, {"w": g1})
    v1 = 0.1 * g1 ** 2
    w1 = [1.0, 1.0] - 0.1 * g1 / (np.sqrt(v1) + 1e-8)
    assert np.allclose(P["w"], w1)
    # the first RMSProp step is sqrt(10) lr in size, whatever the gradient
    assert np.allclose(np.abs(P["w"] - [1.0, 1.0]), 0.1 * np.sqrt(10), rtol=1e-6)
    opt.step(P, {"w": g2})
    v2 = 0.9 * v1 + 0.1 * g2 ** 2
    assert np.allclose(P["w"], w1 - 0.1 * g2 / (np.sqrt(v2) + 1e-8))


def test_adamw_decouples_the_decay_and_reduces_to_adam():
    """AdamW with wd = 0 is Adam bit for bit; with wd > 0 the extra term is
    exactly ``lr wd w`` on top of Adam's step."""
    rng = np.random.default_rng(0)
    P, Q = {"w": rng.normal(size=20)}, None
    Q = {"w": P["w"].copy()}
    a, b = Adam(P, 0.01), AdamW(Q, 0.01, 0.0)
    for _ in range(30):
        g = {"w": rng.normal(size=20)}
        a.step(P, g)
        b.step(Q, g)
    assert np.array_equal(P["w"], Q["w"])
    w0 = rng.normal(size=5)
    g = rng.normal(size=5)
    _, w_adam = _step(Adam, w0, g, lr=0.01)
    _, w_adamw = _step(AdamW, w0, g, lr=0.01, weight_decay=0.1)
    assert np.allclose(w_adam - w_adamw, 0.01 * 0.1 * w0)
    # Adam's (and AdamW's) first step is exactly lr in every coordinate
    assert np.allclose(np.abs(w_adam - w0), 0.01, rtol=1e-6)


def test_schedules_have_the_shapes_the_subtitles_describe():
    from voxanim.mlp_runs import schedule_fn
    steps, peak = 1200, 1e-3
    const = schedule_fn("constant", steps, peak)
    assert all(const(k) == peak for k in (0, 600, 1199))
    step = schedule_fn("step", steps, peak)
    assert (step(0), step(400), step(800)) == pytest.approx((peak, peak / 10, peak / 100))
    cos = schedule_fn("cosine", steps, peak)
    assert cos(0) == pytest.approx(peak / 60) and cos(59) == pytest.approx(peak)
    assert cos(1199) < 1e-8 and cos(600) < cos(100)
    one = schedule_fn("onecycle", steps, peak)
    assert one(359) == pytest.approx(peak) and one(0) < peak / 5
    assert one(1199) == pytest.approx(peak / 100, rel=1e-3)


# ------------------------------------------------------------------ the model


@pytest.fixture(scope="module")
def prob():
    return LS.problem()


@needs_mnist
def test_features_and_standardisation(prob):
    """12,665 zeros and ones, two standardised features, ones are the label 1."""
    assert prob.n == 12665 and prob.F.shape == (12665, 2)
    assert np.allclose(prob.F.mean(0), 0, atol=1e-9) and np.allclose(prob.F.std(0), 1)
    assert set(np.unique(prob.y)) == {0.0, 1.0}
    # ones are fainter and thinner than zeros: both features are lower for them
    assert (prob.F[prob.y == 1].mean(0) < prob.F[prob.y == 0].mean(0)).all()


@needs_mnist
def test_gradient_hessian_and_minibatch_gradient_match_finite_differences(prob):
    """The gradient (full and minibatch) and the Hessian against finite differences."""
    for w in ([0.0, 0.0], [2.5, -2.0], [-4.0, -1.5]):
        P = {"w": np.array(w)}
        fd_check(lambda Q: prob.loss(Q["w"]), P, {"w": prob.grad(w)}, ["w"], tol=1e-6)
        idx = np.arange(0, prob.n, 97)
        fd_check(lambda Q, idx=idx: prob.loss(Q["w"], idx), P, {"w": prob.grad(w, idx)}, ["w"],
                 tol=1e-6)
        e = 1e-5
        H = prob.hessian(w)
        for j in range(2):
            d = np.zeros(2)
            d[j] = e
            assert np.allclose((prob.grad(np.add(w, d)) - prob.grad(np.subtract(w, d))) / (2 * e),
                               H[:, j], rtol=1e-4, atol=1e-7)
    # the vectorised grid agrees with the scalar loss
    w1s, w2s, L = prob.grid((-7, -7), (3, 3), 11)
    assert L.shape == (11, 11)
    assert abs(L[3, 8] - prob.loss([w1s[3], w2s[8]])) < 1e-12


@needs_mnist
def test_the_bowl_and_the_valley(prob):
    """The round bowl's minimum, its accuracy and conditioning; the scaled
    problem's narrow valley."""
    wmin = prob.minimum()
    assert np.allclose(wmin, [-4.29, -1.77], atol=0.01)
    assert abs(prob.loss(wmin) - 0.137) < 0.001
    assert abs(prob.accuracy(wmin) - 0.947) < 0.002
    assert np.linalg.norm(prob.grad(wmin)) < 1e-9
    ev = np.linalg.eigvalsh(prob.hessian(wmin))
    assert 2.0 < ev[1] / ev[0] < 3.0
    ill = prob.scaled(LS.ILL_SCALE)
    wm = ill.minimum()
    assert np.allclose(wm, [wmin[0] / LS.ILL_SCALE, wmin[1]], atol=1e-6)
    assert abs(ill.loss(wm) - prob.loss(wmin)) < 1e-9
    ev = np.linalg.eigvalsh(ill.hessian(wm))
    assert 40 < ev[1] / ev[0] < 50 or 25 < ev[1] / ev[0] < 50
    ev0 = np.linalg.eigvalsh(ill.hessian([0.0, 0.0]))
    assert ev0[1] / ev0[0] > 150


@needs_mnist
def test_minibatch_gradient_covariance_is_sigma_over_b(prob):
    """The spread the drunken walk draws: 4000 minibatch gradients at one point
    have the covariance ``grad_cov / B`` (within 10% in each standard deviation)."""
    w = np.array([-2.0, -2.5])
    rng = np.random.default_rng(0)
    for b in (1, 8, 64):
        gs = np.array([prob.grad(w, rng.integers(0, prob.n, b)) for _ in range(4000)])
        emp = np.sqrt(np.diag(np.cov(gs, rowvar=False, bias=True)))
        pred = np.sqrt(np.diag(prob.grad_cov(w) / b))
        assert np.allclose(emp, pred, rtol=0.1), (b, emp, pred)
        assert np.allclose(gs.mean(0), prob.grad(w), atol=3 * pred.max() / np.sqrt(4000))


# ------------------------------------------------------------------ the animations' claims


@needs_mnist
def test_gradient_descent_walk():
    from voxanim import optim_anim as A
    run = A.gd_runs()["gd"]
    prob = LS.problem()
    Lmin = prob.loss(prob.minimum())
    assert run.steps == A.GD_STEPS and abs(run.loss[0] - 1.15) < 0.01
    # every step is exactly -lr times the gradient there, and the loss never rises
    assert np.allclose(run.w[1:] - run.w[:-1], -A.GD_LR * run.g)
    assert np.all(np.diff(run.loss) < 0)
    assert run.loss[-1] - Lmin < 0.003
    assert run.first_within(Lmin, A.ARRIVED) == 34
    # the steps shrink with the gradient: the last is a hundredth of the first
    lens = np.linalg.norm(np.diff(run.w, axis=0), axis=1)
    assert lens[0] > 2.0 and lens[-1] < 0.02


@needs_mnist
def test_drunken_walk_claims():
    from voxanim import optim_anim as A
    runs = A.sgd_runs()
    prob = LS.problem()
    wmin = prob.minimum()
    Lmin = prob.loss(wmin)
    # all three minibatch walkers still get down: the loss at the end is near the minimum
    for b in A.BATCHES:
        r = runs["b%d" % b]
        assert r.steps == A.SGD_STEPS
        assert r.loss[-1] - Lmin < 0.03
        # and each step was lr times that step's (minibatch) gradient
        assert np.allclose(r.w[1:] - r.w[:-1], -A.SGD_LR * r.g)
    # the stagger shrinks with sqrt(B): the ellipses' sigma ratio is sqrt(8)
    sig = {b: np.sqrt(np.diag(prob.grad_cov(runs["b%d" % b].w[-1]) / b)) for b in A.BATCHES}
    assert np.allclose(sig[1] / sig[8], np.sqrt(8), rtol=0.3)
    assert np.allclose(sig[8] / sig[64], np.sqrt(8), rtol=0.3)
    # too high: the first step lands beyond the minimum and the later steps stay long
    high, low = runs["high"], runs["low"]
    assert np.linalg.norm(high.w[1] - high.w[0]) > 5.0
    half = A.SGD_STEPS // 2
    step_len = lambda r: np.linalg.norm(np.diff(r.w[half:], axis=0), axis=1).mean()  # noqa: E731
    assert step_len(high) > 2 * step_len(runs["b8"])
    # too low: it stops short
    assert np.linalg.norm(low.w[-1] - wmin) > 3.5 and low.loss[-1] > 0.2
    assert step_len(low) < 0.05


@needs_mnist
def test_optimisers_on_the_valley():
    from voxanim import optim_anim as A
    runs = A.optimiser_runs()
    ill = LS.problem().scaled(LS.ILL_SCALE)
    wm = ill.minimum()
    Lm = ill.loss(wm)
    arrive = {k: r.first_within(Lm, A.ARRIVED) for k, r in runs.items()}
    # who arrives: RMSProp, AdamW, Adam, momentum; SGD not in 150 steps
    assert 30 <= arrive["RMSProp"] <= 40
    assert arrive["RMSProp"] < arrive["AdamW"] < arrive["Adam"] < arrive["momentum"] < A.ILL_STEPS
    assert arrive["SGD"] == -1
    # the start is on the steep wall: curvature 19 across the valley, ratio about 280;
    # SGD's rate is nine tenths of the stability limit of the steepest curvature it meets
    ev = np.linalg.eigvalsh(ill.hessian(A.ILL_START))
    assert 18 < ev[1] < 20 and 250 < ev[1] / ev[0] < 320
    curv = [np.linalg.eigvalsh(ill.hessian(w))[1] for w in runs["SGD"].w]
    assert max(curv) == curv[0]
    assert A.OPTIMISERS["SGD"][1]["lr"] == pytest.approx(A.SGD_STABLE * 2 / curv[0], rel=0.02)
    # SGD crosses the valley in one step, then creeps: loss 1.2 -> 0.22, and 0.18 at the end
    L = runs["SGD"].loss
    assert L[0] > 1.1 and L[1] < 0.25 and L[-1] > 0.17
    # momentum overshoots: after its first dip the loss doubles before it comes back
    L = runs["momentum"].loss
    dip = int(np.nonzero(np.diff(L) > 0)[0][0])
    assert L[dip:dip + 40].max() > 1.8 * L[dip] and dip <= 3
    assert runs["momentum"].w[:, 0].min() > A.ILL_LO[0]       # the loop stays in the box
    # Adam's first step is (-0.1, -0.1) exactly; the raw gradient is mostly across the valley
    assert np.allclose(runs["Adam"].w[1] - runs["Adam"].w[0], [-0.1, -0.1], rtol=1e-6)
    g = runs["SGD"].g[0]
    assert abs(g[0]) > 4 * abs(g[1])
    # AdamW's decay holds it short of the minimum; RMSProp lands on it
    assert 0.3 < np.linalg.norm(runs["AdamW"].w[-1] - wm) < 0.8
    assert np.linalg.norm(runs["RMSProp"].w[-1] - wm) < 0.1


@needs_mnist
def test_adamw_vs_l2_claims():
    from voxanim import adamw_anim as W
    runs = W.runs()
    plain, l2, adamw = runs["Adam"], runs["Adam + L2"], runs["AdamW"]
    assert plain.steps == l2.steps == adamw.steps == 1407
    # all three learn (test accuracy about 96%)
    assert min(plain.acc, l2.acc, adamw.acc) > 0.95
    # the effective decay of Adam + L2 varies by orders of magnitude; AdamW's is uniform
    ratio = W.effective_decay(l2.v[-1]) / (W.LR * W.WD)
    assert np.percentile(ratio, 10) < 0.7 and np.percentile(ratio, 90) > 4
    assert ratio.max() > 500
    # and it is largest exactly where the gradient is smallest
    assert np.corrcoef(np.log(l2.v[-1]).ravel(), np.log(ratio).ravel())[0, 1] < -0.999
    # the norms: plain Adam grows, AdamW less, Adam + L2 collapses W0 and holds W1 flat
    n = {k: r.norms[-1] for k, r in runs.items()}
    assert n["Adam"][0] > n["AdamW"][0] > n["Adam + L2"][0] * 1.5
    assert n["Adam"][1] > n["AdamW"][1] > n["Adam + L2"][1]
    assert abs(l2.norms[-1, 1] - l2.norms[0, 1]) < 0.1
    assert n["Adam"][0] > 16 and n["Adam + L2"][0] < 10


@needs_mnist
def test_schedule_claims():
    from voxanim import mlp_runs as R
    from voxanim import schedule_anim as S
    mlp, warm, land = S.mlp_runs(), S.warmup_runs(), S.landscape_runs()
    final = {k: R.smooth(r.losses, S.SMOOTH)[-1] for k, r in mlp.items()}
    # the order: one cycle best, then constant, cosine, step decay (too early)
    assert final["onecycle"] < final["constant"] < final["cosine"] < final["step"]
    assert mlp["onecycle"].acc > 0.965 and mlp["step"].acc < mlp["constant"].acc
    # the landscape walkers finish in the same order
    end = {k: land[k].loss[-1] for k, _ in S.SCHEDULES}
    assert end["onecycle"] < end["constant"] < end["cosine"] < end["step"]
    # the rates the runs used are the schedules
    assert np.allclose(mlp["constant"].lrs, S.PEAK)
    assert mlp["onecycle"].lrs.max() == pytest.approx(S.ONECYCLE_PEAK)
    # warmup: Adam's first step is exactly eta per weight; warmed up, eta / 8
    assert np.allclose(land["nowarm"].w[1] - land["nowarm"].w[0], -S.WARM_PEAK, rtol=1e-6)
    assert np.allclose(land["warm"].w[1] - land["warm"].w[0], -S.WARM_PEAK / 8, rtol=1e-6)
    # on the MLP the weights move far more in the first 20 steps without warmup
    moved = [np.sqrt((r.norms[S.EARLY] ** 2).sum()) - np.sqrt((r.norms[0] ** 2).sum())
             for r in warm]
    assert moved[0] > 10 * moved[1] and moved[0] > 0.5


@needs_mnist
def test_landscape_drawing_geometry():
    """The height-field: column heights follow the loss, walkers sit above the
    surface at their own loss, the step vector maps weights linearly."""
    from voxanim import presets
    anim = presets.build("loss-landscape-descent")
    land = anim.land
    assert land.heights.min() == 1 and land.heights.max() == land.levels
    i, j = np.unravel_index(land.L.argmin(), land.L.shape)
    assert land.heights[i, j] == 1
    wmin = anim.wmin
    p = land.point(wmin)
    assert abs(p[2] - (land.z_of(land.L_min) + LS.LIFT * land.block.scale)) < 0.02
    d = land.step_vec([0.0, 0.0], [1.0, 0.0])
    assert d[0] > 0 and abs(d[1]) < 1e-12 and d[2] == 0
    assert land.inside(wmin) and not land.inside(np.array([10.0, 0.0]))


@needs_mnist
@pytest.mark.parametrize("name", ["loss-landscape-descent", "drunken-walk", "optimisers",
                                  "adamw-vs-l2", "learning-rate-schedule"])
def test_durations_and_phases(name):
    from voxanim import presets
    anim = presets.build(name)
    assert 35 <= anim.duration <= 60, anim.duration
    assert len(anim.PHASES) >= 4
    # the opening subtitle returns for the loop
    anim.scene(0.5)
    first = anim.subtitle
    anim.scene(anim.duration - 0.2)
    assert anim.subtitle == first

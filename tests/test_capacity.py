"""Capacity and bottlenecks: the squeezed classifier's gradients against finite
differences, the SVD identities, the recorded runs consistent with their
settings, and the claims the four animations make, as relative claims."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fd_check, needs_mnist

from voxanim import capacity as C
from voxanim import presets

# ------------------------------------------------------------------ gradients and identities


def test_squeeze_gradients_match_finite_differences():
    """The linear-bottleneck classifier's gradients, every parameter."""
    rng = np.random.default_rng(0)
    P = C.squeeze_init(2, rng)
    for k in P:
        if k.startswith("b"):
            P[k] = rng.normal(0, 0.3, P[k].shape)
    x = rng.random((6, 784))
    y = np.eye(10)[rng.integers(0, 10, 6)]
    G = C.squeeze_backward(P, x, C.squeeze_forward(P, x), y)
    assert set(G) == set(P)
    fd_check(lambda Q: C.squeeze_loss(C.squeeze_forward(Q, x), y), P, G, list(P), tol=1e-5)


def test_svd_reconstruction_identity_on_a_random_matrix():
    """``U Σ Vᵀ`` rebuilds the matrix; the energy of all components is 1."""
    rng = np.random.default_rng(1)
    W = rng.normal(0, 1, (40, 12))
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    assert np.allclose((U * s) @ Vt, W, atol=1e-12)
    assert np.all(np.diff(s) <= 0)                       # largest first
    k = 3
    low = (U[:, :k] * s[:k]) @ Vt[:k]
    assert np.linalg.matrix_rank(low) == k
    # the best rank-k approximation: its error is the dropped singular values
    assert np.isclose(np.linalg.norm(W - low), np.sqrt((s[k:] ** 2).sum()))


def test_params_kept():
    """Two thin factors hold k(m + n) numbers, capped at the full matrix."""
    assert np.isclose(C.params_kept([(784, 64)], 8), 8 * (784 + 64) / (784 * 64))
    assert C.params_kept([(784, 64), (64, 32)], 64) == 1.0
    assert C.params_kept([(64, 32)], 32) == 1.0


def test_parameter_budget():
    """The three shapes of width-vs-depth are within 2% of 50,000 parameters."""
    for _name, sizes in C.SHAPES:
        assert abs(C.n_params(sizes) - 50000) < 1000, (sizes, C.n_params(sizes))
    assert C.n_params((784, 64, 10)) == 784 * 64 + 64 + 64 * 10 + 10


# ------------------------------------------------------------------ the recorded models


@pytest.fixture(scope="module")
def sweep():
    return C.bottleneck_sweep()


@pytest.fixture(scope="module")
def squeeze():
    return [C.squeeze_run(h) for h in C.HIDDEN]


@pytest.fixture(scope="module")
def rank():
    return C.rank_study()


@pytest.fixture(scope="module")
def width_depth():
    runs, data = C.width_depth_runs()
    return runs, data, C.step_times(data)


@needs_mnist
def test_sweep_error_falls_with_width_then_flattens(sweep):
    """bottleneck-sweep: the test error falls at every step of width, steeply
    at first and hardly at all from 32 to 256; the reconstructions are the
    trained networks' outputs on the four digits."""
    assert sweep.widths == C.WIDTHS
    assert np.all(np.diff(sweep.losses) < 0)
    i8, i32 = sweep.widths.index(8), sweep.widths.index(32)
    steep = 1 - sweep.losses[i8] / sweep.losses[0]
    flat = 1 - sweep.losses[-1] / sweep.losses[i32]
    assert steep > 0.4 and flat < 0.3 and flat < steep / 3
    assert sweep.recon.shape == (len(C.WIDTHS), len(C.DIGITS), 784)
    assert sweep.error(len(C.WIDTHS) - 1, 0) < sweep.error(0, 0)
    for L in C.CLOUD_WIDTHS:
        assert sweep.clouds[L].shape == (C.CLOUD, L)
    # a wider code keeps more about the digit: 3 numbers beat 1, 256 beat 3
    acc = dict(zip(sweep.widths, sweep.centroid_acc))
    assert acc[1] < acc[2] < acc[3] and acc[3] < acc[256]
    assert acc[1] > 0.1                                   # but even one number beats chance


@needs_mnist
def test_squeeze_runs_are_consistent_with_their_settings(squeeze):
    """information-bottleneck: snapshots, shapes and counts of both runs."""
    for run, h in zip(squeeze, C.HIDDEN):
        S = len(run.steps)
        assert run.hidden == h and run.steps[0] == 0
        assert run.steps[-1] == C.SQUEEZE["epochs"] * run.steps_per_epoch
        assert run.steps_per_epoch == -(-60000 // C.SQUEEZE["batch"])
        assert np.all(np.diff(run.steps) > 0)
        assert run.cloud.shape == (S, C.CLOUD, h) and run.test_acc.shape == (S,)
        assert [e.shape for e in run.example] == [(S, C.FIRST), (S, h), (S, 10)]
        assert np.allclose(run.example[2].sum(1), 1.0)
        assert run.confusion.sum() == 10000
        assert np.isclose(run.final_acc, np.trace(run.confusion) / 10000)
        assert run.errors == 10000 - np.trace(run.confusion)
        # the recorded cloud is the trained network's middle layer
        _xtr, _ytr, xte, _yte = C.mnist.load()
        sel = np.arange(C.AE.CLOUD_START, C.AE.CLOUD_START + C.CLOUD)
        assert np.allclose(C.squeeze_forward(run.P, xte[sel])[1], run.cloud[-1], atol=1e-5)


@needs_mnist
def test_two_units_lose_to_sixty_four_by_a_margin(squeeze):
    """information-bottleneck: the squeezed network learns (far above chance)
    but ends well below the roomy one, with several times its errors; its
    worst confusion is a real pair of look-alikes."""
    narrow, wide = squeeze
    assert narrow.final_acc > 0.8
    assert wide.final_acc - narrow.final_acc > 0.03
    assert narrow.errors > 2 * wide.errors
    assert narrow.test_acc[-1] > narrow.test_acc[0] + 0.5
    (a, b, n), *_ = narrow.confusions(1)
    assert n > 50 and {a, b} in ({4, 9}, {3, 5}, {5, 8}, {7, 9}, {3, 8}, {5, 6}, {2, 7})


@needs_mnist
def test_rank_study(rank):
    """rank-of-a-layer: the SVD identities on the trained matrices, and the
    accuracy after truncation: rank 32 loses little, rank 4 loses most."""
    for m in range(2):
        U, s, Vt = rank.svd[m]
        assert np.allclose((U * s) @ Vt, rank.Ws[m], atol=1e-10)
        assert np.allclose(rank.truncated(m, len(s)), rank.Ws[m], atol=1e-10)
        assert np.isclose(rank.energy(m, len(s)), 1.0) and rank.energy(m, 1) < 0.5
    assert rank.ranks == C.RANKS + (64,)
    assert np.isclose(rank.acc[-1], rank.full_acc) and rank.full_acc > 0.95
    acc = dict(zip(rank.ranks, rank.acc))
    assert rank.full_acc - acc[32] < 0.02
    assert acc[32] - acc[4] > 0.3
    assert np.all(np.diff(rank.acc) >= 0)
    kept = dict(zip(rank.ranks, rank.kept))
    assert kept[64] == 1.0 and kept[32] < 0.6 and kept[8] < 0.2


@needs_mnist
def test_width_vs_depth(width_depth):
    """width-vs-depth: three runs with the recorded settings; all learn, none
    by a wide margin; He weights keep the gradient of the deep one; the deep
    narrow network costs more wall-clock time a step than the wide one."""
    runs, data, times = width_depth
    assert len(data.x_train) == C.N_TRAIN
    for run, (_name, sizes) in zip(runs, C.SHAPES):
        assert run.spec.sizes == sizes and run.steps[-1] == C.TRAIN["steps"]
        assert run.act_std.shape == (len(run.steps), run.spec.depth)
        assert run.grad_rms.shape == run.act_std.shape
        assert run.test_acc[-1] > 0.9
    acc = [float(r.test_acc[-1]) for r in runs]
    assert max(acc) - min(acc) < 0.03
    g = runs[-1].grad_rms[0]
    assert 0.2 < g[0] / g[-1] < 5                       # neither vanishing nor exploding
    assert np.all(runs[-1].act_std[0] > 0.05)
    assert runs[-1].act_std[-1][-1] > 2 * runs[-1].act_std[0][-1]   # activations grow in training
    assert times.shape == (3,) and np.all(times > 0)
    assert times[-1] > times[0]                         # sixteen small layers cost more than one


# ------------------------------------------------------------------ the animations

CAPACITY = ("bottleneck-sweep", "information-bottleneck", "rank-of-a-layer", "width-vs-depth")


@needs_mnist
@pytest.mark.parametrize("name", CAPACITY)
def test_animation_runs_forty_to_sixty_seconds(name):
    """Each animation is 40 to 60 s, has a subtitle in every phase and loops
    back to its opening subtitle."""
    anim = presets.build(name)
    assert 40 <= anim.duration <= 60, anim.duration
    opening = None
    for t in np.linspace(0.5, anim.t_done, 12):
        anim.scene(float(t))
        assert anim.subtitle, (name, t)
        opening = opening or anim.subtitle
    anim.scene(anim.duration)
    assert anim.subtitle == opening

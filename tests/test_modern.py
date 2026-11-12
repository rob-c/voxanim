"""The modern sequence models: gradients against finite differences, the
identities the animations rest on, and the claims their subtitles make."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fd_check, needs_gpt2

from voxanim import memory as Mem
from voxanim import ssm

# ------------------------------------------------------------------ state space model


def test_ssm_recurrence_equals_convolution():
    """Unrolling the recurrence from h_{-1} = 0 is the convolution with K_k = C Ā^k B̄."""
    rng = np.random.default_rng(1)
    P = ssm.init(rng, n=5, dt=0.2)
    x = rng.normal(0, 1, (3, 20))
    f = ssm.recurrent(P, x)
    assert np.allclose(f["y"], ssm.convolve(P, x), rtol=0, atol=1e-12)
    # the animated case, as its subtitle says: identical to rounding
    anim_x = ssm.example()[None]
    Pa = ssm.init(np.random.default_rng(ssm.SEED))
    diff = np.abs(ssm.recurrent(Pa, anim_x)["y"] - ssm.convolve(Pa, anim_x)).max()
    assert diff < 1e-14


def test_ssm_gradients_match_finite_differences():
    rng = np.random.default_rng(2)
    P = ssm.init(rng, n=4, dt=0.15)
    x = rng.normal(0, 1, (2, 9))
    target = rng.normal(0, 1, x.shape)
    f = ssm.recurrent(P, x)
    g = ssm.backward(P, f, ssm.mse_grad(f["y"], target))
    fd_check(lambda Q: ssm.mse_loss(ssm.recurrent(Q, x)["y"], target), P, g, list(P))


def test_ssm_step_sets_the_memory():
    """A smaller Δ keeps Ā nearer 1: the slowest state remembers longer and the kernel
    decays more slowly (the numbers the step-Δ phase prints)."""
    P = ssm.init(np.random.default_rng(ssm.SEED))
    Abar, _ = ssm.discretize(P)
    assert np.allclose(Abar, np.exp(0.1 * P["A"]))
    assert abs(ssm.memory_steps(P)[0] - 10.0) < 1e-9             # −1 / log(exp(−0.1))
    slow, fast = ssm.with_dt(P, 0.02), ssm.with_dt(P, 0.5)
    assert ssm.memory_steps(slow)[0] > 45 and ssm.memory_steps(fast)[0] < 2.5
    K_slow, K_fast = np.abs(ssm.kernel(slow, 48)), np.abs(ssm.kernel(fast, 48))
    assert K_slow[20] / K_slow[0] > K_fast[20] / K_fast[0] * 100


# ------------------------------------------------------------------ Mamba


def test_mamba_gradients_match_finite_differences():
    from voxanim import mamba as M
    rng = np.random.default_rng(3)
    P = M.init(rng, d=5, n=3, K=3)
    x, cue = Mem.memory_batch(rng, 2, 7)
    f = M.forward(P, x)
    g, _ = M.backward(P, f, M.grad_last(f, cue))
    fd_check(lambda Q: Mem.loss_last(M.forward(Q, x)["out"], cue), P, g, list(P))
    # and for a loss on every step (the backward takes any dL/dout)
    dout = rng.normal(0, 1, x.shape)
    g, _ = M.backward(P, f, dout)
    fd_check(lambda Q: float((M.forward(Q, x)["out"] * dout).sum()), P, g, list(P))


def test_parallel_scan_equals_the_sequential_recurrence():
    from voxanim import mamba as M
    rng = np.random.default_rng(4)
    for T in (1, 2, 7, 50, 64):
        a = rng.uniform(0.5, 1.0, (2, T, 3, 4))
        b = rng.normal(0, 1, (2, T, 3, 4))
        h, trail = M.parallel_scan(a, b, record=True)
        assert np.allclose(h, M.scan(a, b), rtol=0, atol=1e-12)
        assert len(M.scan_levels(T)) == (int(np.ceil(np.log2(T))) if T > 1 else 0)
        assert trail.shape[0] == len(M.scan_levels(T))
    assert M.scan_levels(50) == [1, 2, 4, 8, 16, 32]


def test_mamba_learns_the_memory_task_by_selection():
    """What mamba-selective-scan says: the block answers the cue, Δ opens at the cue and
    shuts on the noise, less of the state survives the cue step, and the parallel scan
    reproduces the recurrence."""
    from voxanim import mamba as M
    tr = M.trace()
    f = tr.f
    assert abs(tr.y - tr.cue) < 0.1
    assert tr.losses[-200:].mean() < Mem.CHANCE / 20
    dt_cue, dt_noise = f["dt"][0].mean(), f["dt"][1:].mean()
    assert dt_cue > 10 * dt_noise
    assert f["Abar"][0].mean() < f["Abar"][1:].mean() - 0.1
    assert np.abs(tr.trail[-1] - f["h"][1:]).max() < 1e-12
    # the RNN and LSTM it is drawn against are the ones gated-memory shows
    rnn, lstm = Mem.trace("rnn"), Mem.trace("lstm", Mem.FORGET_BIAS)
    assert len(rnn.losses) == len(lstm.losses) == Mem.STEPS and len(tr.losses) == M.STEPS


# ------------------------------------------------------------------ mixture of experts


def test_moe_gradients_match_finite_differences():
    """Through the top-2 routing (locally constant) and the balancing loss."""
    from voxanim import moe
    rng = np.random.default_rng(5)
    P = moe.init(rng, d=6, h=5, E=4)
    cur, nxt = moe.batch(rng, 12)
    f = moe.forward(P, cur, 0.05)
    ps = np.sort(f["p"], 1)
    assert (ps[:, -2] - ps[:, -3]).min() > 1e-7           # no routing flips in the differences
    g = moe.backward(P, f, nxt)
    fd_check(lambda Q: moe.loss(moe.forward(Q, cur, 0.05), nxt), P, g, list(P))


def test_moe_routing_and_balance():
    """What mixture-of-experts says: two experts per token with renormalised gates, both
    models near the chain's entropy floor, the balancing loss evening the load, and the
    compute / parameter count."""
    from voxanim import moe
    T = moe.transition()
    assert np.allclose(T.sum(1), 1.0)
    bal, plain = moe.train(moe.ALPHA), moe.train(0.0)
    for run in (bal, plain):
        assert moe.test_loss(run.P) < moe.entropy_floor() + 0.1
    fw, _ = moe.utilisation(bal.P)
    fwo, _ = moe.utilisation(plain.P)
    assert fw.max() / fw.min() < fwo.max() / fwo.min()
    f = moe.forward(bal.P, np.arange(len(moe.VOCAB)), 0.0)
    assert f["top"].shape == (len(moe.VOCAB), 2) and np.allclose(f["g"].sum(1), 1.0)
    assert f["f"].sum() == pytest.approx(1.0) and f["aux"] == 0.0
    assert moe.compute_per_token() == 1024 and moe.parameters() == 4352


# ------------------------------------------------------------------ the cost of context


def test_context_cost_arithmetic():
    from voxanim import cost_anim as CA
    assert CA.KV_PER_TOKEN == 12 * 2 * 12 * 64 == 18432
    assert CA.MAMBA_STATE == 24 * 1536 * 16 == 589824
    assert CA.MAMBA_STATE // CA.KV_PER_TOKEN == 32            # the cache passes the state here
    assert CA.KV_ROWS == 24 and CA.STATE_COLS == 32           # the same 768-number voxel
    from voxanim.mamba import scan_levels
    assert scan_levels(16) == [1, 2, 4, 8]


# ------------------------------------------------------------------ rotary embeddings


def test_rope_depends_on_the_relative_position_only():
    from voxanim import rope_anim as R
    rng = np.random.default_rng(6)
    q, k = rng.normal(0, 1, 64), rng.normal(0, 1, 64)
    th = R.frequencies()
    assert th[0] == 1.0 and abs(th[-1] - 10000 ** (-62 / 64)) < 1e-12
    assert abs(R.score(q, k, 3, 7, th) - R.score(q, k, 10, 14, th)) < 1e-10
    assert abs(R.score(q, k, 3, 7, th) - R.score(q, k, 3, 10, th)) > 1e-3
    assert abs(R.score(q, k, 3, 7, th) - q @ R.rotate(k, 4, th)) < 1e-10
    assert abs(np.linalg.norm(R.rotate(q, 5, th)) - np.linalg.norm(q)) < 1e-12
    assert np.allclose(R.self_score_by_pair(q, 0, th).sum(), q @ q)
    # the animated pair of positions, to the decimals the subtitle prints
    anim = R.RotaryEmbeddings.__new__(R.RotaryEmbeddings)
    rng = np.random.default_rng(R.SEED)
    qa, ka = rng.normal(0, 1, 64) / 8, rng.normal(0, 1, 64) / 8
    assert abs(R.score(qa, ka, 3, 7, th) - R.score(qa, ka, 10, 14, th)) < 1e-10
    del anim


@needs_gpt2
def test_gpt2_positions_fade_with_the_gap():
    """GPT-2's learned position vectors: similar near the diagonal, less so further out."""
    from voxanim import rope_anim as R
    sim = R.gpt2_similarity(64)
    assert np.allclose(np.diag(sim), 1.0)
    near, mid, far = (np.mean(np.diag(sim, g)) for g in (1, 16, 48))
    assert near > mid > far

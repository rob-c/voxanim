"""A minimal Mamba block (a selective state space model), in plain numpy.

The SSM of :mod:`voxanim.ssm` has one fixed step ``Δ`` and fixed ``B``, ``C``:
every input is written into the state the same way, so it cannot tell a cue
from noise.  Mamba makes them functions of the input, step by step --
*selection*.  For a scalar sequence ``x_t`` the block computes, with ``d``
channels and ``n`` states per channel:

    u_t = W_in x_t + b_in,   z_t = W_z x_t + b_z                     (d,)   two branches
    v_t = silu(b_conv + Σ_k W_conv[:, k] u_{t−K+1+k})                (d,)   causal conv, width K
    Δ_t = softplus(W_dt v_t + b_dt)                                  (d,)   the step
    B_t = W_B v_t,   C_t = W_C v_t                                   (n,)   what is written, read
    A   = −exp(A_log)                                                (d, n) fixed, negative
    h_t = exp(Δ_t A) ⊙ h_{t−1} + (Δ_t ⊗ B_t) v_t                     (d, n) the selective scan
    y_t = h_t C_t + D ⊙ v_t,   o_t = y_t ⊙ silu(z_t),   out_t = W_out . o_t + b_out

``exp(Δ_t A)`` is how much of each state survives the step: ``Δ_t`` small
keeps it (the block ignores ``x_t``), large replaces it (the block attends to
``x_t`` and forgets).  The ``B̄ = Δ B`` here is Mamba's (Euler) rather than
the SSM's zero-order hold, as in the paper.

On the memory task of :mod:`voxanim.memory` (say the first value after 49
steps of noise) the cue and the noise come from the same distribution, so
the *value* of ``x_t`` cannot tell them apart.  What can is the causal
convolution's zero padding: at ``t = 0`` the window is ``(0, 0, 0, u_0)``,
afterwards it is full of noise.  Mamba's selection is a function of the
input window only, and it learns to open ``Δ`` on that start-of-sequence
signature and shut it on everything else -- which is the whole trick.

The scan ``h_t = a_t h_{t−1} + b_t`` is associative: ``(a, b) ∘ (a', b') =
(a a', a' b + b')``, so :func:`parallel_scan` computes every ``h_t`` in
``log₂ T`` combine levels (Hillis-Steele) with the same result as the
sequential loop (:func:`scan`); the training runs the sequential one.

:func:`backward` is BPTT through all of the above for a given ``dL/dout``;
:mod:`tests.test_modern` checks it against finite differences.  :func:`train`
fits the block on the memory task with Adam (same gap, seed and batch as
:func:`voxanim.memory.train`); the result is cached.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import DEFAULT_CACHE, cached_npz
from .memory import CHANCE, T_GAP, example, loss_last, memory_batch
from .optim import Adam

D_INNER = 16                 # channels d
D_STATE = 8                  # states per channel n
CONV_WIDTH = 4               # K: the causal convolution's taps
STEPS = 8000                 # training steps (as memory.STEPS)
BATCH = 32
LR = 1e-2
SEED = 0
DT_RANGE = (0.001, 0.1)      # Mamba's initial Δ: softplus(b_dt) log-uniform in this range


def silu(z):
    """``z σ(z)``."""
    return z / (1.0 + np.exp(-z))


def _silu_grad(z):
    """``d silu / dz = σ(z) (1 + z (1 − σ(z)))``."""
    s = 1.0 / (1.0 + np.exp(-z))
    return s * (1.0 + z * (1.0 - s))


def softplus(s):
    """``log(1 + e^s)``, stably."""
    return np.logaddexp(0.0, s)


def init(rng, d=D_INNER, n=D_STATE, K=CONV_WIDTH):
    """Mamba's initialisation for ``d`` channels, ``n`` states, conv width ``K``.

    ``A_log = log(1 .. n)`` per channel (S4D-Real), ``D = 1``, the step's bias
    the inverse softplus of a log-uniform draw in :data:`DT_RANGE`.
    """
    dt = np.exp(rng.uniform(np.log(DT_RANGE[0]), np.log(DT_RANGE[1]), d))
    return dict(
        W_in=rng.normal(0, 1.0, d), b_in=np.zeros(d),
        W_z=rng.normal(0, 1.0, d), b_z=np.zeros(d),
        W_conv=rng.uniform(-1, 1, (d, K)) / np.sqrt(K), b_conv=np.zeros(d),
        W_dt=rng.uniform(-1, 1, (d, d)) / np.sqrt(d), b_dt=dt + np.log(-np.expm1(-dt)),
        W_B=rng.normal(0, 1 / np.sqrt(d), (d, n)), W_C=rng.normal(0, 1 / np.sqrt(d), (d, n)),
        A_log=np.tile(np.log(1.0 + np.arange(n)), (d, 1)), D=np.ones(d),
        W_out=rng.normal(0, 1 / np.sqrt(d), d), b_out=np.zeros(()))


def scan(Abar, inp):
    """The sequential scan ``h_t = Ā_t h_{t−1} + inp_t`` over axis 1:
    ``(N, T, ...)`` -> ``h`` ``(N, T + 1, ...)`` with ``h[:, 0] = 0``."""
    h = np.zeros((Abar.shape[0], Abar.shape[1] + 1) + Abar.shape[2:])
    for t in range(Abar.shape[1]):
        h[:, t + 1] = Abar[:, t] * h[:, t] + inp[:, t]
    return h


def scan_levels(T):
    """The Hillis-Steele combine offsets for ``T`` steps: ``1, 2, 4, ...`` below ``T``
    (``ceil(log₂ T)`` levels).  At level ``s`` every step ``t ≥ s`` combines with ``t − s``."""
    levels, s = [], 1
    while s < T:
        levels.append(s)
        s *= 2
    return levels


def parallel_scan(Abar, inp, record=False):
    """The same ``h`` as :func:`scan` by the associative (prefix) scan.

    Each step carries ``(a, b)`` with ``h_t = a h_{t−s−1} + b`` after level ``s``
    has combined ``t`` with ``t − s``: ``(a, b) ← (a_t a_{t−s}, a_t b_{t−s} + b_t)``.
    With ``record`` also returns ``b`` after every level, ``(levels, N, T, ...)``.
    """
    a, b = Abar.copy(), inp.copy()
    T = a.shape[1]
    trail = []
    for s in scan_levels(T):
        a_prev, b_prev = a[:, :T - s], b[:, :T - s]
        b = np.concatenate([b[:, :s], a[:, s:] * b_prev + b[:, s:]], 1)
        a = np.concatenate([a[:, :s], a[:, s:] * a_prev], 1)
        if record:
            trail.append(b.copy())
    h = np.concatenate([np.zeros_like(b[:, :1]), b], 1)
    return (h, np.array(trail)) if record else h


def forward(P, x):
    """Run the block over ``x`` ``(N, T)``; every intermediate, for the backward pass
    and the animation.  ``h`` is ``(N, T + 1, d, n)`` with ``h[:, 0] = 0``."""
    K = P["W_conv"].shape[1]
    u = x[..., None] * P["W_in"] + P["b_in"]
    z = x[..., None] * P["W_z"] + P["b_z"]
    up = np.pad(u, ((0, 0), (K - 1, 0), (0, 0)))               # zeros before the start
    T = x.shape[1]
    c = P["b_conv"] + sum(up[:, k:k + T] * P["W_conv"][:, k] for k in range(K))
    v = silu(c)
    s = v @ P["W_dt"] + P["b_dt"]
    dt = softplus(s)
    Bt, Ct = v @ P["W_B"], v @ P["W_C"]
    A = -np.exp(P["A_log"])
    Abar = np.exp(dt[..., None] * A)
    inp = dt[..., None] * Bt[:, :, None, :] * v[..., None]
    h = scan(Abar, inp)
    y = np.einsum("ntdk,ntk->ntd", h[:, 1:], Ct) + P["D"] * v
    o = y * silu(z)
    out = o @ P["W_out"] + P["b_out"]
    return dict(x=x, u=u, z=z, up=up, c=c, v=v, s=s, dt=dt, Bt=Bt, Ct=Ct, A=A, Abar=Abar,
                h=h, y=y, o=o, out=out)


def backward(P, f, dout):
    """BPTT through the block for ``dL/dout`` ``(N, T)``: gradients for every key
    of ``P``, and ``dh`` ``(N, T, d, n)``, the gradient reaching each state."""
    x, v, z, dt, Bt, Ct, A, Abar, h, y, o = (f[k] for k in
                                             ("x", "v", "z", "dt", "Bt", "Ct", "A", "Abar",
                                              "h", "y", "o"))
    T = x.shape[1]
    K = P["W_conv"].shape[1]
    g = {}
    g["W_out"] = np.einsum("nt,ntd->d", dout, o)
    g["b_out"] = np.array(dout.sum())
    do = dout[..., None] * P["W_out"]
    sz = silu(z)
    dy = do * sz
    dz = do * y * _silu_grad(z)
    g["D"] = (dy * v).sum((0, 1))
    dv = dy * P["D"]
    dCt = np.einsum("ntd,ntdk->ntk", dy, h[:, 1:])
    # back through the scan
    dh_all = np.zeros_like(h[:, 1:])
    dh_next = np.zeros(h.shape[0:1] + h.shape[2:])
    for t in range(T - 1, -1, -1):
        dh = dy[:, t, :, None] * Ct[:, t, None, :] + dh_next
        dh_all[:, t] = dh
        dh_next = dh * Abar[:, t]
    dAbar = dh_all * h[:, :-1]
    dinp = dh_all
    dA = (dAbar * Abar * dt[..., None]).sum((0, 1))
    g["A_log"] = dA * A
    ddt = (dAbar * Abar * A).sum(-1) + np.einsum("ntdk,ntk,ntd->ntd", dinp, Bt, v)
    dBt = np.einsum("ntdk,ntd,ntd->ntk", dinp, dt, v)
    dv = dv + np.einsum("ntdk,ntd,ntk->ntd", dinp, dt, Bt)
    # the selection: Δ, B, C from v
    ds = ddt / (1.0 + np.exp(-f["s"]))
    g["W_dt"] = np.einsum("ntd,nte->de", v, ds)
    g["b_dt"] = ds.sum((0, 1))
    g["W_B"] = np.einsum("ntd,ntk->dk", v, dBt)
    g["W_C"] = np.einsum("ntd,ntk->dk", v, dCt)
    dv = dv + ds @ P["W_dt"].T + dBt @ P["W_B"].T + dCt @ P["W_C"].T
    # the convolution and the two input branches
    dc = dv * _silu_grad(f["c"])
    up = f["up"]
    g["W_conv"] = np.stack([(dc * up[:, k:k + T]).sum((0, 1)) for k in range(K)], 1)
    g["b_conv"] = dc.sum((0, 1))
    dup = np.zeros_like(up)
    for k in range(K):
        dup[:, k:k + T] += dc * P["W_conv"][:, k]
    du = dup[:, K - 1:]
    g["W_in"] = (du * x[..., None]).sum((0, 1))
    g["b_in"] = du.sum((0, 1))
    g["W_z"] = (dz * x[..., None]).sum((0, 1))
    g["b_z"] = dz.sum((0, 1))
    return g, dict(dh=dh_all)


def grad_last(f, cue):
    """``dL/dout`` of :func:`voxanim.memory.loss_last` for the forward pass ``f``."""
    N, T = f["out"].shape
    dout = np.zeros((N, T))
    dout[:, -1] = (f["out"][:, -1] - cue) / N
    return dout


@dataclass
class MambaRun:
    """A trained block and the loss of every training step."""

    P: dict
    losses: np.ndarray        # (steps,) batch loss before each update


def train(d=D_INNER, n=D_STATE, K=CONV_WIDTH, steps=STEPS, batch=BATCH, lr=LR, seed=SEED,
          T=T_GAP, cache_dir=DEFAULT_CACHE) -> MambaRun:
    """Adam on fresh batches of the memory task (gap ``T − 1``), as
    :func:`voxanim.memory.train` trains the RNN and the LSTM.  Cached."""
    settings = dict(task="memory", kind="mamba", d=d, n=n, K=K, T=T, steps=steps, batch=batch,
                    lr=lr, seed=seed)

    def build():
        rng = np.random.default_rng(seed)
        P = init(rng, d, n, K)
        opt = Adam(P, lr)
        losses = []
        for _ in range(steps):
            x, cue = memory_batch(rng, batch, T)
            f = forward(P, x)
            losses.append(loss_last(f["out"], cue))
            opt.step(P, backward(P, f, grad_last(f, cue))[0])
        return dict(P, losses=np.array(losses))

    dct = cached_npz("memory_mamba", settings, build, cache_dir)
    return MambaRun({k: v for k, v in dct.items() if k != "losses"}, dct["losses"])


@dataclass
class MambaTrace:
    """The memory example through a trained block: the forward intermediates
    (batch removed), the parallel scan's trail, and the training losses."""

    x: np.ndarray            # (T,)
    cue: float
    P: dict
    f: dict                  # forward(), batch dimension removed
    trail: np.ndarray        # (levels, T, d, n): the scan's partial sums after each level
    losses: np.ndarray

    @property
    def y(self):
        """The answer: the output at the last step."""
        return float(self.f["out"][-1])

    @property
    def kept(self):
        """``exp(Δ_t A)`` averaged over the states: the fraction of the state each
        channel keeps at each step, ``(T, d)``."""
        return self.f["Abar"].mean(-1)


def trace(cache_dir=DEFAULT_CACHE, **kw) -> MambaTrace:
    """:func:`voxanim.memory.example` through the trained block."""
    run = train(cache_dir=cache_dir, **kw)
    x = example()
    f = forward(run.P, x[None])
    h, trail = parallel_scan(f["Abar"], f["dt"][..., None] * f["Bt"][:, :, None, :]
                             * f["v"][..., None], record=True)
    assert np.allclose(h, f["h"], atol=1e-12)
    return MambaTrace(x, float(x[0]), run.P, {k: v[0] for k, v in f.items()}, trail[:, 0],
                      run.losses)


def learned(run: MambaRun, last=200):
    """True when the run ends well below chance (mean loss of the ``last`` steps)."""
    return float(run.losses[-last:].mean()) < CHANCE / 4

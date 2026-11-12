"""A diagonal linear state space model (SSM), in plain numpy: one recurrence, two views.

A continuous-time linear system with a diagonal state matrix, read at a step
``Δ`` with the zero-order hold (the input is held constant over each step):

    continuous   h'(s) = A h(s) + B x(s),        y(s) = C . h(s)
    discrete     Ā = exp(Δ A),  B̄ = (Ā − 1) / A . B          (ZOH, elementwise)
    recurrence   h_t = Ā h_{t−1} + B̄ x_t,       y_t = C . h_t
    convolution  y_t = Σ_k K_k x_{t−k},          K_k = C . Ā^k B̄

The two forms give the same ``y`` to rounding (:func:`recurrent`,
:func:`convolve`): unrolling the recurrence from ``h_{−1} = 0`` *is* the
convolution.  The recurrence costs one state update per step at inference;
the convolution is one long FFT-able filter over the whole sequence at
training time.  ``K`` decays with the eigenvalues ``Ā_n = exp(Δ A_n)`` of the
state matrix (all in (0, 1) for ``A_n < 0``), so ``Δ`` sets how long the
state remembers: a small step keeps ``Ā`` near 1 and the kernel long.

Parameters ``P``: ``A`` (n,) negative reals, ``B`` (n,), ``C`` (n,) and
``log_dt`` (the step as ``Δ = exp(log_dt)``).  :func:`backward` is BPTT
through the recurrence for a given ``dL/dy``; :mod:`tests.test_modern`
checks it against finite differences and the two forms against each other.
:func:`init` is S4D-Real's: ``A_n = −(n + 1)``, ``B = 1``, random ``C``.
"""

from __future__ import annotations

import numpy as np

N_STATE = 6                 # states of the animated SSM
DT = 0.1                    # its step Δ
T_STEPS = 48                # length of the animated sequence
PERIOD = 12.0               # steps per cycle of the sine
STEP_AT = 24                # the step in the input happens here
STEP_SIZE = 0.7
AMPLITUDE = 0.6
SEED = 0


def init(rng, n=N_STATE, dt=DT):
    """S4D-Real parameters for ``n`` states: ``A_n = −(n + 1)``, ``B = 1``,
    ``C ~ N(0, 1/n)`` and the step ``dt``."""
    return dict(A=-(1.0 + np.arange(n, dtype=np.float64)), B=np.ones(n),
                C=rng.normal(0, 1 / np.sqrt(n), n), log_dt=np.array(np.log(dt)))


def example(T=T_STEPS):
    """The animated input ``(T,)``: a sine wave that steps up half way."""
    t = np.arange(T, dtype=np.float64)
    return AMPLITUDE * np.sin(2 * np.pi * t / PERIOD) + STEP_SIZE * (t >= STEP_AT)


def discretize(P):
    """``(Ā, B̄)`` of ``P`` by the zero-order hold at the step ``Δ = exp(log_dt)``."""
    dt = np.exp(P["log_dt"])
    Abar = np.exp(dt * P["A"])
    Bbar = (Abar - 1.0) / P["A"] * P["B"]
    return Abar, Bbar


def kernel(P, L):
    """The convolution kernel ``K_k = C . Ā^k B̄`` for ``k = 0 .. L − 1``, ``(L,)``."""
    Abar, Bbar = discretize(P)
    powers = Abar[None, :] ** np.arange(L)[:, None]          # (L, n)
    return powers @ (P["C"] * Bbar)


def recurrent(P, x):
    """Run the recurrence over ``x`` ``(N, T)``.

    Returns ``x``, ``h`` ``(N, T + 1, n)`` (``h[:, 0]`` the zero initial
    state) and ``y`` ``(N, T)``.
    """
    N, T = x.shape
    Abar, Bbar = discretize(P)
    h = np.zeros((N, T + 1, len(Abar)))
    for t in range(T):
        h[:, t + 1] = Abar * h[:, t] + Bbar * x[:, t, None]
    return dict(x=x, h=h, y=h[:, 1:] @ P["C"])


def convolve(P, x):
    """The same output by the causal convolution of ``x`` ``(N, T)`` with
    :func:`kernel` of length ``T``: ``(N, T)``."""
    N, T = x.shape
    K = kernel(P, T)
    y = np.zeros((N, T))
    for k in range(T):
        y[:, k:] += K[k] * x[:, :T - k]
    return y


def backward(P, f, dy):
    """BPTT of the recurrence for ``dL/dy`` ``(N, T)``: gradients for every key of ``P``."""
    x, h = f["x"], f["h"]
    N, T = x.shape
    A, B = P["A"], P["B"]
    dt = np.exp(P["log_dt"])
    Abar, _Bbar = discretize(P)
    g = dict(A=np.zeros_like(A), B=np.zeros_like(B), C=np.einsum("nt,ntk->k", dy, h[:, 1:]))
    dAbar, dBbar = np.zeros_like(A), np.zeros_like(B)
    dh_next = np.zeros((N, len(A)))
    for t in range(T - 1, -1, -1):
        dh = dy[:, t, None] * P["C"] + dh_next
        dAbar += (dh * h[:, t]).sum(0)
        dBbar += (dh * x[:, t, None]).sum(0)
        dh_next = dh * Abar
    # through the zero-order hold: Ā = exp(Δ A), B̄ = (Ā − 1) / A . B
    g["A"] = dAbar * Abar * dt + dBbar * B * (dt * Abar * A - (Abar - 1.0)) / A ** 2
    g["B"] = dBbar * (Abar - 1.0) / A
    ddt = (dAbar * Abar * A).sum() + (dBbar * B * Abar).sum()
    g["log_dt"] = np.array(ddt * dt)
    return g


def mse_loss(y, target):
    """Half the mean squared error, ``0.5 mean((y − target)²)``, as a float."""
    return float(0.5 * np.mean((y - target) ** 2))


def mse_grad(y, target):
    """``dL/dy`` of :func:`mse_loss`."""
    return (y - target) / y.size


def with_dt(P, dt):
    """``P`` with its step replaced by ``dt``."""
    return dict(P, log_dt=np.array(np.log(dt)))


def memory_steps(P):
    """Each state's memory in steps, ``−1 / log Ā_n`` (the e-folding time of the kernel)."""
    Abar, _ = discretize(P)
    return -1.0 / np.log(Abar)

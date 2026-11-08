"""A vanilla RNN and an LSTM that learn to continue a sine wave, in plain numpy.

Task: given ``x_0 .. x_{T-1}`` of ``sin(w t + phi)`` (random ``w``, ``phi``
per sequence), predict the next value at every step; loss is the mean squared
error over the sequence.  Both are trained with backprop through time (BPTT).

    RNN:   h_t = tanh(Wx x_t + Wh h_{t-1} + b),            y_t = Wy . h_t + by
    LSTM:  [i f g o] = [sig sig tanh sig](W [x_t; h_{t-1}] + b)
           c_t = f * c_{t-1} + i * g,   h_t = o * tanh(c_t),  y_t = Wy . h_t + by

The animations show a network part-way through training (``CHECKPOINT``
steps), so its predictions are visibly off and one more step visibly helps.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .scene import DEFAULT_CACHE

T_STEPS = 16
OMEGA = (0.35, 0.75)


def sine_batch(rng, n, T=T_STEPS):
    w = rng.uniform(*OMEGA, (n, 1))
    phi = rng.uniform(0, 2 * np.pi, (n, 1))
    s = np.sin(w * np.arange(T + 1)[None] + phi)
    return s[:, :-1], s[:, 1:]                       # inputs, targets (next value)


def example():
    """The sequence the animations use."""
    t = np.arange(T_STEPS + 1)
    s = np.sin(0.5 * t + 0.4)
    return s[:-1], s[1:]


def _sig(z):
    return 1.0 / (1.0 + np.exp(-z))


# ------------------------------------------------------------------ RNN


def rnn_init(rng, n):
    return dict(Wx=rng.normal(0, 0.5, (n,)), Wh=rng.normal(0, 1 / np.sqrt(n), (n, n)),
                b=np.zeros(n), Wy=rng.normal(0, 1 / np.sqrt(n), (n,)), by=np.zeros(()))


def rnn_forward(P, x):
    N, T = x.shape
    n = len(P["b"])
    h = np.zeros((N, T + 1, n))                      # h[:, 0] is h_{-1} = 0
    for t in range(T):
        h[:, t + 1] = np.tanh(x[:, t, None] * P["Wx"] + h[:, t] @ P["Wh"].T + P["b"])
    y = h[:, 1:] @ P["Wy"] + P["by"]
    return dict(x=x, h=h, y=y)


def rnn_backward(P, f, target):
    x, h, y = f["x"], f["h"], f["y"]
    N, T = x.shape
    dy = (y - target) / (N * T)                      # d mean(0.5 (y - t)^2)
    g = {k: np.zeros_like(v) for k, v in P.items()}
    g["Wy"] = np.einsum("nt,ntk->k", dy, h[:, 1:])
    g["by"] = dy.sum()
    dh_next = np.zeros_like(h[:, 0])
    dh_all = np.zeros_like(h[:, 1:])                 # total gradient reaching h_t
    dz_all = np.zeros_like(h[:, 1:])                 # ... and through the tanh
    for t in range(T - 1, -1, -1):
        dh = dy[:, t, None] * P["Wy"] + dh_next
        dh_all[:, t] = dh
        dz = dh * (1 - h[:, t + 1] ** 2)
        dz_all[:, t] = dz
        g["Wx"] += (dz * x[:, t, None]).sum(0)
        g["Wh"] += dz.T @ h[:, t]
        g["b"] += dz.sum(0)
        dh_next = dz @ P["Wh"]
    return g, dict(dy=dy, dh=dh_all, dz=dz_all)


def rnn_loss(f, target):
    return float(0.5 * np.mean((f["y"] - target) ** 2))


# ------------------------------------------------------------------ LSTM


def lstm_init(rng, n):
    W = rng.normal(0, 1 / np.sqrt(n + 1), (4 * n, n + 1))
    b = np.zeros(4 * n)
    b[n:2 * n] = 1.0                                 # forget gate starts open
    return dict(W=W, b=b, Wy=rng.normal(0, 1 / np.sqrt(n), (n,)), by=np.zeros(()))


def lstm_forward(P, x):
    N, T = x.shape
    n = len(P["Wy"])
    h = np.zeros((N, T + 1, n))
    c = np.zeros((N, T + 1, n))
    gates = np.zeros((N, T, 4, n))                   # i, f, g, o
    for t in range(T):
        z = np.concatenate([x[:, t, None], h[:, t]], 1) @ P["W"].T + P["b"]
        i, fg, gg, o = _sig(z[:, :n]), _sig(z[:, n:2 * n]), np.tanh(z[:, 2 * n:3 * n]), \
            _sig(z[:, 3 * n:])
        c[:, t + 1] = fg * c[:, t] + i * gg
        h[:, t + 1] = o * np.tanh(c[:, t + 1])
        gates[:, t] = np.stack([i, fg, gg, o], 1)
    y = h[:, 1:] @ P["Wy"] + P["by"]
    return dict(x=x, h=h, c=c, gates=gates, y=y)


def lstm_backward(P, f, target):
    x, h, c, G, y = f["x"], f["h"], f["c"], f["gates"], f["y"]
    N, T = x.shape
    n = len(P["Wy"])
    dy = (y - target) / (N * T)
    g = {k: np.zeros_like(v) for k, v in P.items()}
    g["Wy"] = np.einsum("nt,ntk->k", dy, h[:, 1:])
    g["by"] = dy.sum()
    dh_next = np.zeros((N, n))
    dc_next = np.zeros((N, n))
    dh_all, dc_all = np.zeros_like(h[:, 1:]), np.zeros_like(c[:, 1:])
    dz_all = np.zeros((N, T, 4 * n))
    for t in range(T - 1, -1, -1):
        i, fg, gg, o = G[:, t, 0], G[:, t, 1], G[:, t, 2], G[:, t, 3]
        tc = np.tanh(c[:, t + 1])
        dh = dy[:, t, None] * P["Wy"] + dh_next
        dc = dh * o * (1 - tc ** 2) + dc_next
        dh_all[:, t], dc_all[:, t] = dh, dc
        dz = np.concatenate([dc * gg * i * (1 - i), dc * c[:, t] * fg * (1 - fg),
                             dc * i * (1 - gg ** 2), dh * tc * o * (1 - o)], 1)
        dz_all[:, t] = dz
        inp = np.concatenate([x[:, t, None], h[:, t]], 1)
        g["W"] += dz.T @ inp
        g["b"] += dz.sum(0)
        dinp = dz @ P["W"]
        dh_next = dinp[:, 1:]
        dc_next = dc * fg
    return g, dict(dy=dy, dh=dh_all, dc=dc_all, dz=dz_all)


# ------------------------------------------------------------------ training


MODELS = {
    "rnn": dict(init=rnn_init, fwd=rnn_forward, bwd=rnn_backward, n=8),
    "lstm": dict(init=lstm_init, fwd=lstm_forward, bwd=lstm_backward, n=6),
}
CHECKPOINT = {"rnn": 5, "lstm": 20}         # early: visibly wrong, one step visibly helps
STEP_LR = {"rnn": 0.1, "lstm": 0.5}


def train(kind, steps=None, batch=32, lr=1e-2, seed=0, cache_dir=DEFAULT_CACHE):
    """Adam on random sine sequences; returns params after ``steps`` updates."""
    steps = CHECKPOINT[kind] if steps is None else steps
    M = MODELS[kind]
    path = os.path.join(cache_dir, "%s_n%d_s%d_seed%d.npz" % (kind, M["n"], steps, seed))
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files}
    rng = np.random.default_rng(seed)
    P = M["init"](rng, M["n"])
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(a) for k, a in P.items()}
    for step in range(1, steps + 1):
        x, tgt = sine_batch(rng, batch)
        g, _ = M["bwd"](P, M["fwd"](P, x), tgt)
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
            P[k] = P[k] - lr * (m[k] / (1 - 0.9 ** step)) / (
                np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
    os.makedirs(cache_dir, exist_ok=True)
    np.savez(path, **P)
    return P


@dataclass
class SeqTrace:
    kind: str
    x: np.ndarray            # (T,) inputs
    target: np.ndarray       # (T,) next values
    P: dict
    f: dict                  # forward intermediates (batch removed)
    g: dict                  # parameter gradients
    d: dict                  # activation gradients (batch removed)
    loss: float


def _trace(kind, P, x, target):
    M = MODELS[kind]
    f = M["fwd"](P, x[None])
    g, d = M["bwd"](P, f, target[None])
    return SeqTrace(kind=kind, x=x, target=target, P=P,
                    f={k: v[0] for k, v in f.items()}, g=g, d={k: v[0] for k, v in d.items()},
                    loss=float(0.5 * np.mean((f["y"][0] - target) ** 2)))


def trace(kind) -> SeqTrace:
    x, tgt = example()
    return _trace(kind, train(kind), x, tgt)


def sgd_step(tr: SeqTrace, lr) -> SeqTrace:
    return _trace(tr.kind, {k: tr.P[k] - lr * tr.g[k] for k in tr.P}, tr.x, tr.target)


# ------------------------------------------------------------------ a 2-layer RNN, trained in batches
#
#   h1_t = tanh(Wx1 x_t + Wh1 h1_{t-1} + b1)
#   h2_t = tanh(W12 h1_t + Wh2 h2_{t-1} + b2)
#   y_t  = Wy . h2_t + by


def rnn2_init(rng, n=8):
    s = 1 / np.sqrt(n)
    return dict(Wx1=rng.normal(0, 0.5, (n,)), Wh1=rng.normal(0, s, (n, n)), b1=np.zeros(n),
                W12=rng.normal(0, s, (n, n)), Wh2=rng.normal(0, s, (n, n)), b2=np.zeros(n),
                Wy=rng.normal(0, s, (n,)), by=np.zeros(()))


def rnn2_forward(P, x):
    N, T = x.shape
    n = len(P["b1"])
    h1 = np.zeros((N, T + 1, n))
    h2 = np.zeros((N, T + 1, n))
    for t in range(T):
        h1[:, t + 1] = np.tanh(x[:, t, None] * P["Wx1"] + h1[:, t] @ P["Wh1"].T + P["b1"])
        h2[:, t + 1] = np.tanh(h1[:, t + 1] @ P["W12"].T + h2[:, t] @ P["Wh2"].T + P["b2"])
    y = h2[:, 1:] @ P["Wy"] + P["by"]
    return dict(x=x, h1=h1, h2=h2, y=y)


def rnn2_backward(P, f, target):
    x, h1, h2, y = f["x"], f["h1"], f["h2"], f["y"]
    N, T = x.shape
    dy = (y - target) / (N * T)
    g = {k: np.zeros_like(v) for k, v in P.items()}
    g["Wy"] = np.einsum("nt,ntk->k", dy, h2[:, 1:])
    g["by"] = dy.sum()
    n1 = np.zeros_like(h1[:, 0])
    n2 = np.zeros_like(h2[:, 0])
    for t in range(T - 1, -1, -1):
        dh2 = dy[:, t, None] * P["Wy"] + n2
        dz2 = dh2 * (1 - h2[:, t + 1] ** 2)
        g["W12"] += dz2.T @ h1[:, t + 1]
        g["Wh2"] += dz2.T @ h2[:, t]
        g["b2"] += dz2.sum(0)
        n2 = dz2 @ P["Wh2"]
        dh1 = dz2 @ P["W12"] + n1
        dz1 = dh1 * (1 - h1[:, t + 1] ** 2)
        g["Wx1"] += (dz1 * x[:, t, None]).sum(0)
        g["Wh1"] += dz1.T @ h1[:, t]
        g["b1"] += dz1.sum(0)
        n1 = dz1 @ P["Wh1"]
    return g


@dataclass
class BatchRecord:
    step: int                 # 1-based training step
    x: np.ndarray             # (B, T) inputs
    target: np.ndarray        # (B, T)
    f: dict                   # forward pass with the weights *before* this step
    g: dict                   # gradients (averaged over the batch)
    loss: float
    P: dict                   # weights before the step


@dataclass
class TrainingRun:
    losses: np.ndarray        # loss of every step
    shown: list               # BatchRecords for the steps the animation shows in detail
    P_final: dict


def rnn2_training(steps=400, batch=4, n=8, lr=0.004, seed=3, show=(1, 2, 3, 4)):
    """Train from scratch, keeping every loss and full records for ``show``
    steps and the last one."""
    rng = np.random.default_rng(seed)
    P = rnn2_init(rng, n)
    m = {k: np.zeros_like(v) for k, v in P.items()}
    v = {k: np.zeros_like(a) for k, a in P.items()}
    losses, shown = [], []
    keep = set(show) | {steps}
    for step in range(1, steps + 1):
        x, tgt = sine_batch(rng, batch)
        f = rnn2_forward(P, x)
        g = rnn2_backward(P, f, tgt)
        loss = float(0.5 * np.mean((f["y"] - tgt) ** 2))
        losses.append(loss)
        if step in keep:
            shown.append(BatchRecord(step, x, tgt, f, g, loss,
                                     {k: np.array(a) for k, a in P.items()}))
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
            P[k] = P[k] - lr * (m[k] / (1 - 0.9 ** step)) / (
                np.sqrt(v[k] / (1 - 0.999 ** step)) + 1e-8)
    return TrainingRun(np.array(losses), shown, P)

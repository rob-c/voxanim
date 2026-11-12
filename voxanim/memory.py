"""A task that needs gated memory: say the first value of a sequence after a
gap of noise.

    x_0 = cue ~ U(-1, 1),   x_1 .. x_{T-1} ~ U(-1, 1) (noise),   loss = 0.5 (y_{T-1} - cue)^2

The cells are those of :mod:`voxanim.sequence` (a tanh RNN and an LSTM),
with the same number of units and the same training (Adam, fresh batches).
Across a gap of 49 steps, in four seeds out of four:

* the RNN stays at chance (loss 1/6, what always answering 0 scores);
* so does an LSTM whose forget gate starts at the usual bias of 1 (the gate
  opens to 0.73, and what the cell state holds leaks away within the gap);
* an LSTM whose forget gate starts at :data:`FORGET_BIAS` (open to 0.92)
  solves the task.

The gate *is* the memory.  The gradient of the final error shrinks through
every tanh step of the RNN and reaches the cue vanishingly small; along the
cell state it is multiplied only by the forget gate.  Shorter gaps do not
separate the cells: with Adam the RNN learns a gap of 19 to 31 steps in most
seeds.  :mod:`voxanim.gated_anim` shows the RNN against the gated LSTM, and
the training curves of all three.

Gradients come from the cells' own backward passes.  Those differentiate the
mean squared error over *all* steps, so the targets of the other steps are
set to the predictions themselves (no error there) and the result is scaled
by ``T``; exact, because the backward pass is linear in the error.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import DEFAULT_CACHE, cached_npz
from .optim import Adam
from .sequence import MODELS

T_GAP = 50                   # sequence length: the cue, then 49 steps of noise
UNITS = 8                    # hidden units of both networks
STEPS = 8000                 # training steps
FORGET_BIAS = 2.5            # the gated LSTM starts with its forget gate open to 0.92
LEAKY_BIAS = 1.0             # sequence.lstm_init's usual bias: the gate opens to 0.73
CUE = 0.8                    # the example's cue
EXAMPLE_SEED = 1             # the example's noise
CHANCE = 1.0 / 6.0           # loss of always answering 0: 0.5 E[cue^2]


def memory_batch(rng, n, T=T_GAP):
    """``n`` sequences: ``(x (n, T), cue (n,))``, the cue first, then noise."""
    x = rng.uniform(-1, 1, (n, T))
    return x, x[:, 0].copy()


def example(cue=CUE, T=T_GAP, seed=EXAMPLE_SEED):
    """The animated sequence ``(T,)``: ``cue`` followed by fixed noise."""
    x = np.random.default_rng(seed).uniform(-1, 1, T)
    x[0] = cue
    return x


def loss_last(y, cue):
    """``0.5 mean((y_last - cue)^2)`` over the batch, as a float."""
    return float(0.5 * np.mean((y[:, -1] - cue) ** 2))


def backward_last(kind, P, f, cue):
    """Gradients of :func:`loss_last` for the forward pass ``f``: ``(g, d)`` as
    the cell's backward pass returns them (see the module docstring)."""
    y = f["y"]
    target = np.array(y)
    target[:, -1] = cue
    g, d = MODELS[kind]["bwd"](P, f, target)
    T = y.shape[1]
    return {k: v * T for k, v in g.items()}, {k: v * T for k, v in d.items()}


@dataclass
class MemoryRun:
    """A trained network and the loss of every training step."""

    kind: str
    forget_bias: float | None
    P: dict
    losses: np.ndarray        # (steps,) batch loss before each update


def train(kind, forget_bias=None, steps=STEPS, batch=32, lr=1e-2, seed=0,
          cache_dir=DEFAULT_CACHE) -> MemoryRun:
    """Adam on fresh batches of the task; ``kind`` is ``"rnn"`` or ``"lstm"``.

    ``forget_bias`` (LSTM only) sets the forget gate's bias at the start.
    Cached by :func:`voxanim.cache.cached_npz`.
    """
    M = MODELS[kind]
    settings = dict(task="memory", kind=kind, n=UNITS, T=T_GAP, steps=steps, batch=batch, lr=lr,
                    seed=seed, forget_bias=forget_bias)

    def build():
        rng = np.random.default_rng(seed)
        P = M["init"](rng, UNITS)
        if forget_bias is not None:
            P["b"][UNITS:2 * UNITS] = forget_bias
        opt = Adam(P, lr)
        losses = []
        for _ in range(steps):
            x, cue = memory_batch(rng, batch)
            f = M["fwd"](P, x)
            losses.append(loss_last(f["y"], cue))
            opt.step(P, backward_last(kind, P, f, cue)[0])
        return dict(P, losses=np.array(losses))

    d = cached_npz("memory_" + kind, settings, build, cache_dir)
    return MemoryRun(kind, forget_bias, {k: v for k, v in d.items() if k != "losses"},
                     d["losses"])


@dataclass
class MemoryTrace:
    """The example through a trained network, twice: with its cue and with the
    cue flipped.  ``f`` and ``d`` are the first run's forward intermediates
    and activation gradients (batch removed); ``f2`` the flipped run's."""

    kind: str
    x: np.ndarray            # (T,)
    cue: float
    P: dict
    f: dict
    d: dict
    f2: dict
    losses: np.ndarray       # the training run's

    @property
    def y(self):
        """The answer: the prediction at the last step."""
        return float(self.f["y"][-1])

    @property
    def y2(self):
        """The flipped run's answer."""
        return float(self.f2["y"][-1])


def _one(d):
    """A batch-of-one dict without its batch dimension."""
    return {k: v[0] for k, v in d.items()}


def trace(kind, forget_bias=None, cache_dir=DEFAULT_CACHE) -> MemoryTrace:
    """:func:`example` through the trained network, and with the cue flipped."""
    run = train(kind, forget_bias, cache_dir=cache_dir)
    M = MODELS[kind]
    x = example()
    x2 = np.array(x)
    x2[0] = -x[0]
    f = M["fwd"](run.P, x[None])
    _g, d = backward_last(kind, run.P, f, np.array([x[0]]))
    f2 = M["fwd"](run.P, x2[None])
    return MemoryTrace(kind, x, float(x[0]), run.P, _one(f), _one(d), _one(f2), run.losses)

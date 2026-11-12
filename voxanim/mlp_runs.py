"""MNIST MLP training runs recorded step by step, for the optimisation animations.

The network is the one :mod:`voxanim.mnist` trains (784-64-32-10, ReLU,
He-normal initialisation, cross-entropy) with the same minibatch loop, so
``optimizer=Adam, lr=1e-3`` here reproduces that training exactly.  What is
new is the recording: every step's minibatch loss and learning rate, the
Frobenius norm of each weight matrix after every step, and, at chosen steps,
snapshots of one weight matrix and of the optimiser's root second moment
``sqrt(v_hat)`` for it (what AdamW-versus-L2 draws).  Runs are cached by
:func:`voxanim.cache.cached_npz` under all their settings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cache import cached_npz
from .minibatch import minibatches
from .mnist import CLASSES, SIZES, _as_layers, _forward, _init, _mlp_grads, load
from .optim import BETA2

WATCH = "W1"                     # the matrix the sheets show: 64 x 32, the middle layer
LAYERS = ("W0", "W1", "W2")      # the weight matrices whose norms are recorded


@dataclass
class MLPRun:
    """One recorded training run (see the module docstring)."""

    losses: np.ndarray          # (steps,) minibatch cross-entropy before each step
    lrs: np.ndarray             # (steps,) the learning rate each step used
    norms: np.ndarray           # (steps + 1, 3) ||W0||, ||W1||, ||W2|| before each step and after
    snap_steps: np.ndarray      # (S,) the steps after which snapshots were taken
    W: np.ndarray               # (S, 64, 32) the watched matrix at those steps
    v: np.ndarray               # (S, 64, 32) sqrt of the bias-corrected second moment there
    acc: float                  # test accuracy at the end

    @property
    def steps(self):
        """Number of training steps."""
        return len(self.losses)


def schedule_fn(kind, steps, peak, warmup=0.05):
    """``lr(step)`` for ``kind``:

    * ``constant``: ``peak`` throughout;
    * ``step``: ``peak``, divided by 10 after each third of training;
    * ``cosine``: linear warmup from 0 over the first ``warmup`` of the steps,
      then a cosine from ``peak`` down to 0;
    * ``onecycle``: linear from ``peak / 10`` up to ``peak`` over the first 30%,
      then a cosine down to ``peak / 100``.
    """
    def constant(k):
        return peak

    def step(k):
        return peak * 0.1 ** int(3 * k / steps)

    def cosine(k):
        w = max(1, int(round(warmup * steps)))
        if k < w:
            return peak * (k + 1) / w
        return peak * 0.5 * (1 + np.cos(np.pi * (k - w) / max(1, steps - w)))

    def onecycle(k):
        up = int(round(0.3 * steps))
        if k < up:
            return peak / 10 + (peak - peak / 10) * (k + 1) / up
        u = (k - up) / max(1, steps - up)
        return peak / 100 + (peak - peak / 100) * 0.5 * (1 + np.cos(np.pi * u))

    return dict(constant=constant, step=step, cosine=cosine, onecycle=onecycle)[kind]


def train(name, settings, make_opt, lr_at=None, epochs=3, batch=128, seed=0, decay=0.0,
          snapshots=24, cache_dir=None):
    """Train the MLP with ``make_opt(params)`` (any optimiser of
    :mod:`voxanim.optim`) and record it; cached under ``settings``.

    ``lr_at(k)`` is the learning rate of step ``k`` (else the optimiser's);
    ``decay`` adds ``decay * W`` to every weight matrix's gradient (L2 in the
    loss, biases excluded); ``snapshots`` snapshots are spaced evenly in the
    log of the step (plus the last step).
    """
    settings = dict(settings, model="mlp", sizes=list(SIZES), epochs=epochs, batch=batch,
                    seed=seed, decay=decay, snapshots=snapshots, init="he_normal", data="mnist/255")

    def build():
        xtr, ytr, xte, yte = load(cache_dir)
        rng = np.random.default_rng(seed)
        P = {"%s%d" % (n, k): a for k, layer in enumerate(_init(rng))
             for n, a in zip("Wb", layer, strict=True)}
        opt = make_opt(P)
        eye = np.eye(CLASSES)
        steps = epochs * ((len(xtr) + batch - 1) // batch)
        snap_at = set(np.unique(np.round(np.logspace(0, np.log10(steps), snapshots)).astype(int)))
        losses, lrs, norms, snap_steps, Ws, vs = [], [], [_norms(P)], [], [], []
        for k, idx in enumerate(minibatches(rng, len(xtr), epochs, batch)):
            x, y = xtr[idx], eye[ytr[idx]]
            p = _forward(_as_layers(P, 3), x)[2]
            losses.append(float(-np.mean(np.log(p[np.arange(len(idx)), ytr[idx]] + 1e-12))))
            g = _mlp_grads(P, x, y)
            if decay:
                for key in LAYERS:
                    g[key] = g[key] + decay * P[key]
            lr = None if lr_at is None else float(lr_at(k))
            opt.step(P, g, lr)
            lrs.append(opt.lr if lr is None else lr)
            norms.append(_norms(P))
            if k + 1 in snap_at:
                snap_steps.append(k + 1)
                Ws.append(P[WATCH].copy())
                v = getattr(opt, "v", None)
                vs.append(np.sqrt(v[WATCH] / (1 - BETA2 ** (k + 1))) if v is not None
                          else np.zeros_like(P[WATCH]))
        acc = float((_forward(_as_layers(P, 3), xte)[2].argmax(1) == yte).mean())
        return dict(losses=np.array(losses), lrs=np.array(lrs), norms=np.array(norms),
                    snap_steps=np.array(snap_steps), W=np.array(Ws), v=np.array(vs),
                    acc=np.array(acc))

    d = cached_npz("mlp_run_%s" % name, settings, build, cache_dir)
    return MLPRun(d["losses"], d["lrs"], d["norms"], d["snap_steps"], d["W"], d["v"],
                  float(d["acc"]))


def _norms(P):
    """Frobenius norms of the weight matrices, in :data:`LAYERS` order."""
    return [float(np.linalg.norm(P[k])) for k in LAYERS]


def smooth(x, window):
    """Running mean of ``x`` over ``window`` steps (same length; the start is
    averaged over what is there)."""
    c = np.cumsum(np.concatenate([[0.0], np.asarray(x, np.float64)]))
    n = np.arange(1, len(x) + 1)
    lo = np.maximum(0, n - window)
    return (c[n] - c[lo]) / (n - lo)

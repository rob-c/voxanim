"""Activation functions in training: the runs and measurements behind the
``activation-*`` animations.

Seven activations (:data:`ACTS`: sigmoid, tanh, ReLU, Leaky ReLU, GELU, SiLU,
ELU) each train the same MNIST classifier with :mod:`voxanim.deepnet`, under
the initialisation that fits the family (Xavier for the squashing sigmoid
and tanh, He for the rectifiers), the same data, batches, steps and
optimiser.  Three networks are used:

* **the race** (:func:`race_runs`): 784 → 128 → 64 → 10, 1,500 Adam steps
  (learning rate 0.001, batches of 64) on 10,000 training images; the run
  also records how much of each layer is off, near zero or on a flat part of
  the activation at every recorded step (``sparse``),
* **eight layers** (:func:`deep8_runs`): 784 → 64 × 8 → 10, 1,000 Adam
  steps: the gradient reaching each layer, at the start and after training,
* **twenty layers** (:func:`deep20_runs`): 784 → 100 × 20 → 10, the network
  of the ``initialisation`` animation, 400 SGD steps at learning rate 0.1
  with no normalisation: which activations train at that depth.

The measurements the animations read out are here too: the time a training
step costs with each activation (:func:`step_times`, measured, not
estimated), the fraction of a layer that is switched off (:func:`fractions`)
and the units that never fire (:func:`dead_units`), the network's output
along the straight line between two digits and the kinks in it
(:func:`line_between`, :func:`kinks`), and a one-dimensional slice of the
training loss along a random weight direction (:func:`loss_slice`).
Everything is seeded and cached through :func:`voxanim.cache.cached_npz`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from . import color as C
from .cache import cached_npz
from .deepnet import (
    FLAT,
    NEAR_ZERO,
    Adam,
    Spec,
    act,
    act_grad,
    backward,
    forward,
    objective,
    subset,
    train,
)


@dataclass(frozen=True)
class Act:
    """One activation as the animations present it."""

    key: str            # the deepnet name
    label: str          # on screen
    init: str           # the initialisation its family uses
    ink: tuple          # its colour for curves and text

    @property
    def init_label(self):
        """``Xavier`` or ``He``."""
        return "Xavier" if self.init == "xavier" else "He"


# Inks chosen for contrast against the backdrop (text uses them too).
ACTS = (Act("sigmoid", "sigmoid", "xavier", (0.75, 0.22, 0.22)),
        Act("tanh", "tanh", "xavier", tuple(float(v) for v in C.rgb("#8a6a2e"))),
        Act("relu", "ReLU", "he", tuple(float(v) for v in C.rgb("#2f8a86"))),
        Act("leaky_relu", "Leaky ReLU", "he", tuple(float(v) for v in C.rgb("#3f55c8"))),
        Act("gelu", "GELU", "he", tuple(float(v) for v in C.rgb("#7a46d6"))),
        Act("silu", "SiLU", "he", tuple(float(v) for v in C.rgb("#b0318c"))),
        Act("elu", "ELU", "he", tuple(float(v) for v in C.rgb("#35803d"))))
BY_KEY = {a.key: a for a in ACTS}
SMOOTH = ("sigmoid", "tanh", "gelu", "silu")       # f is C-infinity (ELU is only C-1)
KINKED = ("relu", "leaky_relu")                    # f' jumps at 0

N_TRAIN, N_TEST = 10000, 2000
RACE_SIZES = (784, 128, 64, 10)
RACE = dict(steps=1500, batch=64, lr=0.001, opt="adam", sparse=True)
DEEP8_SIZES = (784,) + (64,) * 8 + (10,)
DEEP8 = dict(steps=1000, batch=64, lr=0.001, opt="adam", every=50)
DEEP20_SIZES = (784,) + (100,) * 20 + (10,)
DEEP20 = dict(steps=400, batch=64, lr=0.1, opt="sgd", every=20)   # as the initialisation anim
SEED = 0


def spec(a, sizes):
    """The :class:`~voxanim.deepnet.Spec` of activation ``a`` on ``sizes``."""
    return Spec(sizes, a.key, a.init)


def data(cache_dir=None):
    """The MNIST subset every run trains on."""
    return subset(N_TRAIN, N_TEST, seed=SEED, cache_dir=cache_dir)


def runs(sizes, settings, cache_dir=None, dat=None):
    """One recorded run per activation of :data:`ACTS` on ``sizes`` (cached)."""
    dat = data(cache_dir) if dat is None else dat
    kw = dict(settings)
    if cache_dir:
        kw["cache_dir"] = cache_dir
    return [train(spec(a, sizes), dat, **kw) for a in ACTS]


def race_runs(cache_dir=None, dat=None):
    """The seven runs of the 784-128-64-10 network."""
    return runs(RACE_SIZES, RACE, cache_dir, dat)


def deep8_runs(cache_dir=None, dat=None):
    """The seven runs of the eight-layer network."""
    return runs(DEEP8_SIZES, DEEP8, cache_dir, dat)


def deep20_runs(cache_dir=None, dat=None):
    """The seven runs of the twenty-layer network."""
    return runs(DEEP20_SIZES, DEEP20, cache_dir, dat)


# ------------------------------------------------------------------ time per step

TIMED_STEPS = 40            # steps in one timed block
TIMED_REPS = 12             # blocks per activation, interleaved round-robin over the activations
FN_CALLS = 300              # calls of f and f' in one timed block, on a (batch, 128) array


def step_times(cache_dir=None, dat=None):
    """``{"step": (7,), "fn": (7,)}`` seconds: the least time one training step of
    the race network took with each activation, and the least time the
    activation and its derivative alone took on a layer's ``(64, 128)``
    pre-activations.  Blocks are interleaved across the activations and the
    minimum is kept, so whatever else the machine was doing cancels out
    (cached: a measurement of the machine it ran on)."""
    settings = dict(task="step_times", sizes=list(RACE_SIZES), batch=RACE["batch"],
                    acts=[a.key for a in ACTS], steps=TIMED_STEPS, reps=TIMED_REPS,
                    fn_calls=FN_CALLS)

    def build():
        dat_ = data(cache_dir) if dat is None else dat
        rng = np.random.default_rng(SEED)
        x = dat_.x_train[:RACE["batch"]]
        y = dat_.onehot(dat_.y_train[:RACE["batch"]])
        specs = [spec(a, RACE_SIZES) for a in ACTS]
        params = [__import__("voxanim.deepnet", fromlist=["init"]).init(s, rng) for s in specs]
        opts = [Adam(P, RACE["lr"]) for P in params]
        z = rng.normal(0, 2, (RACE["batch"], RACE_SIZES[1]))
        step = np.full(len(ACTS), np.inf)
        fn = np.full(len(ACTS), np.inf)
        for _rep in range(TIMED_REPS):
            for i, (s, P, opt) in enumerate(zip(specs, params, opts)):
                t0 = time.perf_counter()
                for _ in range(TIMED_STEPS):
                    f = forward(s, P, x, train=True)
                    G, _D = backward(s, P, f, y)
                    opt.step(P, G)
                step[i] = min(step[i], (time.perf_counter() - t0) / TIMED_STEPS)
                t0 = time.perf_counter()
                for _ in range(FN_CALLS):
                    a = act(s.activation, z)
                    act_grad(s.activation, z, a)
                fn[i] = min(fn[i], (time.perf_counter() - t0) / FN_CALLS)
        return dict(step=step, fn=fn)

    return cached_npz("actfun_timing", settings, build, cache_dir)


# ------------------------------------------------------------------ what a layer leaves on


def hidden(sp, P, x):
    """``(y, a)``: lists over hidden layers of the pre-activations and
    activations of the batch ``x`` through parameters ``P``."""
    f = forward(sp, P, x)
    return f["y"], f["a"]


def fractions(sp, P, x):
    """``(zero, small, flat)``, each ``(L,)``: per hidden layer, the fraction of
    activations of ``x`` that are exactly 0, within :data:`~voxanim.deepnet.NEAR_ZERO`
    of 0, and whose slope ``|f'(y)|`` is below :data:`~voxanim.deepnet.FLAT`."""
    ys, As = hidden(sp, P, x)
    zero = np.array([float((a == 0.0).mean()) for a in As])
    small = np.array([float((np.abs(a) < NEAR_ZERO).mean()) for a in As])
    flat = np.array([float((np.abs(act_grad(sp.activation, y, a)) < FLAT).mean())
                     for y, a in zip(ys, As)])
    return zero, small, flat


def off(sp, P, x):
    """Per hidden layer, the ``(N, width)`` mask of activations of ``x`` that are
    off: within :data:`~voxanim.deepnet.NEAR_ZERO` of 0.  ReLU's exact zeros
    are inside that, and the others make an exact 0 only by underflow (sigmoid
    or GELU of a very negative value), so one threshold serves all seven."""
    _ys, As = hidden(sp, P, x)
    return [np.abs(a) < NEAR_ZERO for a in As]


def dead_units(sp, P, x):
    """Per hidden layer, the ``(width,)`` mask of units that are off on every
    row of ``x``: dead, as far as these inputs can tell."""
    return [m.all(0) for m in off(sp, P, x)]


# ------------------------------------------------------------------ along a line


def line_between(xa, xb, n):
    """``(n, 784)`` images along the straight line from ``xa`` to ``xb``
    (``alpha`` from 0 to 1 inclusive)."""
    alpha = np.linspace(0.0, 1.0, n)[:, None]
    return (1.0 - alpha) * xa[None] + alpha * xb[None]


def along(sp, P, line, cls):
    """``(logit, p)`` of class ``cls`` at every point of ``line``."""
    f = forward(sp, P, line)
    return f["logits"][:, cls], f["p"][:, cls]


def kinks(sp, P, line):
    """How many times a hidden pre-activation changes sign between consecutive
    points of ``line``: the kinks of a ReLU network's output along it (a
    smooth activation has no kink there, only a bend)."""
    ys, _As = hidden(sp, P, line)
    return int(sum(((y[1:] > 0) != (y[:-1] > 0)).sum() for y in ys))


# ------------------------------------------------------------------ a slice of the loss


def direction(P, rng):
    """A random direction in parameter space, each array's part scaled to
    that array's own norm (Li et al.'s filter normalisation: a step of ``s``
    moves every layer by the same fraction ``s`` of its size)."""
    d = {}
    for k, v in P.items():
        r = rng.normal(0.0, 1.0, v.shape)
        scale = np.linalg.norm(v) / max(np.linalg.norm(r), 1e-12)
        d[k] = r * scale
    return d


def loss_slice(sp, P, x, y_onehot, d, s):
    """The objective at ``P + s_i d`` for each ``s_i`` of ``s``, on ``(x, y_onehot)``."""
    out = np.empty(len(s))
    for i, si in enumerate(s):
        Q = {k: v + si * d[k] for k, v in P.items()}
        out[i] = objective(sp, Q, forward(sp, Q, x), y_onehot)
    return out


def roughness(values):
    """The largest jump of the slope of ``values`` (sampled evenly), relative to
    the mean slope size: 0 for a straight line, large for a kinked curve."""
    v = np.asarray(values, np.float64)
    slope = np.diff(v)
    return float(np.abs(np.diff(slope)).max() / max(np.abs(slope).mean(), 1e-300))

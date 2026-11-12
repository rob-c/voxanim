"""Activation functions: their shapes and slopes, a real layer through each,
and ReLU units dying under too large a learning rate.

    sigmoid   tanh   ReLU   GELU        four plots: f (teal) and f' (magenta)

    z  ████████████████████████████     a real layer's 64 pre-activations
    σ  ████████████████████████████     ...through each function, as voxels
    tanh / ReLU / GELU rows

1. **functions** -- σ(z) = 1 / (1 + exp(−z)), tanh, ReLU = max(0, z) and
   GELU = z Φ(z) (GPT-2's tanh form), each with its derivative: the sigmoid
   and tanh slopes vanish at both ends, ReLU's is exactly 0 or 1, GELU's
   dips below 0 and rises above 1,
2. **a real layer** -- the 64 pre-activations of the MNIST MLP's first
   hidden layer on test image 18 (:func:`voxanim.mnist.trace`), drawn as
   dots on each curve and as a row of voxels before and after each
   function: saturation (sigmoid, tanh), sparsity (ReLU: exact zeros, dark),
3. **dead ReLUs** -- a 784-128-10 ReLU network trained by plain SGD on MNIST
   with a learning rate of 1.0 (:data:`LR_HOT`).  Within the first 30 steps
   31 of its 128 hidden units stop firing on every one of the 10,000 test
   images and never fire again: their voxels go dark for good, the count
   rises on the chart, and the gradient through a dead unit is exactly 0
   (ReLU′(z) = 0 for z < 0), so nothing can revive them.  At 0.3
   (:data:`LR_SANE`) no unit dies and the accuracy after the epoch is the
   same (94.5% against 94.8%): the dead units were lost for nothing.  Both
   runs are seeded and cached (:func:`dead_relu_run`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import color as C
from . import layout
from .cache import cached_npz
from .conv import Tempo
from .diagram import DATA, GRAD, Diagram
from .easing import remap, smoothstep
from .minibatch import minibatches
from .netanim import amax, div_colors, lit_colors
from .text import num

# ------------------------------------------------------------------ the functions

GELU_C = math.sqrt(2.0 / math.pi)
GELU_K = 0.044715


def sigmoid(z):
    """``1 / (1 + exp(−z))``."""
    return 1.0 / (1.0 + np.exp(-np.asarray(z, np.float64)))


def d_sigmoid(z):
    """``σ(z) (1 − σ(z))``."""
    s = sigmoid(z)
    return s * (1.0 - s)


def tanh(z):
    """``tanh(z)``."""
    return np.tanh(np.asarray(z, np.float64))


def d_tanh(z):
    """``1 − tanh²(z)``."""
    return 1.0 - np.tanh(np.asarray(z, np.float64)) ** 2


def relu(z):
    """``max(0, z)``."""
    return np.maximum(np.asarray(z, np.float64), 0.0)


def d_relu(z):
    """``1`` where ``z > 0``, else ``0`` (exactly: nothing passes a switched-off unit)."""
    return (np.asarray(z, np.float64) > 0).astype(np.float64)


def gelu(z):
    """GELU in its tanh form (GPT-2's ``gelu_new``):
    ``½ z (1 + tanh(√(2/π) (z + 0.044715 z³)))``."""
    z = np.asarray(z, np.float64)
    return 0.5 * z * (1.0 + np.tanh(GELU_C * (z + GELU_K * z ** 3)))


def d_gelu(z):
    """The derivative of :func:`gelu`: ``½(1 + t) + ½ z (1 − t²) √(2/π) (1 + 3·0.044715 z²)``
    with ``t = tanh(√(2/π) (z + 0.044715 z³))``."""
    z = np.asarray(z, np.float64)
    t = np.tanh(GELU_C * (z + GELU_K * z ** 3))
    return 0.5 * (1.0 + t) + 0.5 * z * (1.0 - t ** 2) * GELU_C * (1.0 + 3.0 * GELU_K * z ** 2)


# name, label, f, f', plot y range, signed (diverging colours) or not (lit colours)
FUNCTIONS = (("sigmoid", "sigmoid σ(z)", sigmoid, d_sigmoid, (-0.3, 1.3), False),
             ("tanh", "tanh(z)", tanh, d_tanh, (-1.3, 1.3), True),
             ("relu", "ReLU max(0, z)", relu, d_relu, (-1.0, 4.3), False),
             ("gelu", "GELU z·Φ(z)", gelu, d_gelu, (-1.0, 4.3), True))
Z_RANGE = (-4.5, 4.5)               # the plots' z axis
SATURATED = 0.05                    # |f'| below this: the unit is on a flat end
LAYER_INDEX = 18                    # the MNIST test image (the mlp-* animations' 3)


def layer_preactivations(index=LAYER_INDEX):
    """``(z (64,), label)``: the first hidden layer's pre-activations of the MNIST
    MLP on test image ``index``."""
    from . import mnist
    tr = mnist.trace(index)
    return np.asarray(tr.zs[0], np.float64), tr.label


# ------------------------------------------------------------------ dead ReLUs

HIDDEN, BATCH, EPOCHS, SEED = 128, 128, 1, 0
LR_HOT, LR_SANE = 1.0, 0.3          # too large, and sane (3x smaller: same accuracy, no deaths)
EARLY, EVERY = 60, 10               # checkpoints: every step up to EARLY, then every EVERY


def _forward(P, x):
    """``(z1, h, p)``: pre-activations, ReLU outputs and softmax of the 784-H-10 net."""
    z1 = x @ P["W1"] + P["b1"]
    h = np.maximum(z1, 0.0)
    z2 = h @ P["W2"] + P["b2"]
    z2 = z2 - z2.max(1, keepdims=True)
    p = np.exp(z2)
    p /= p.sum(1, keepdims=True)
    return z1, h, p


def _grads(P, x, y_onehot):
    """Gradients of the mean cross-entropy, and ``d1`` = dL/dz1 per image."""
    z1, h, p = _forward(P, x)
    d2 = (p - y_onehot) / len(x)
    d1 = (d2 @ P["W2"].T) * (z1 > 0)
    return dict(W1=x.T @ d1, b1=d1.sum(0), W2=h.T @ d2, b2=d2.sum(0)), d1


@dataclass
class DeadRun:
    """One training run and what it did to the hidden units, on the whole
    MNIST test set at every checkpoint."""

    lr: float
    steps: np.ndarray           # (n,) training step of each checkpoint
    dead: np.ndarray            # (n, H) bool: z <= 0 on every test image
    active: np.ndarray          # (n, H) fraction of test images the unit fires on
    accuracy: float             # test accuracy at the end
    z_max: np.ndarray           # (H,) the unit's largest z over the test set, at the end
    grad_max: np.ndarray        # (H,) largest |dL/dz| over the test set, at the end
    P: dict                     # the final parameters

    @property
    def counts(self):
        """Dead units per checkpoint."""
        return self.dead.sum(1)

    @property
    def n_dead(self):
        """Dead units at the end."""
        return int(self.dead[-1].sum())


def dead_relu_run(lr, hidden=HIDDEN, batch=BATCH, epochs=EPOCHS, seed=SEED, cache_dir=None):
    """Plain SGD at ``lr`` on MNIST, one 784-``hidden``-10 ReLU network, with the
    hidden units' state on the test set recorded at every checkpoint (cached)."""
    from . import mnist
    settings = dict(task="dead_relu", sizes=[784, hidden, 10], batch=batch, epochs=epochs, lr=lr,
                    seed=seed, init="he_normal", optimizer="sgd", data="mnist/255",
                    checkpoints=[EARLY, EVERY])

    def build():
        xtr, ytr, xte, yte = mnist.load(cache_dir)
        rng = np.random.default_rng(seed)
        P = dict(W1=rng.normal(0, np.sqrt(2.0 / 784), (784, hidden)), b1=np.zeros(hidden),
                 W2=rng.normal(0, np.sqrt(2.0 / hidden), (hidden, 10)), b2=np.zeros(10))
        eye = np.eye(10)
        steps, dead, active = [], [], []

        def record(n):
            z1 = xte @ P["W1"] + P["b1"]
            steps.append(n)
            dead.append((z1 <= 0).all(0))
            active.append((z1 > 0).mean(0))
        n = 0
        for idx in minibatches(rng, len(xtr), epochs, batch):
            if n <= EARLY or n % EVERY == 0:
                record(n)
            g, _d1 = _grads(P, xtr[idx], eye[ytr[idx]])
            for k in P:
                P[k] = P[k] - lr * g[k]
            n += 1
        record(n)
        z1, _h, p = _forward(P, xte)
        _g, d1 = _grads(P, xte, eye[yte])
        return dict(P, steps=np.array(steps), dead=np.array(dead), active=np.array(active),
                    acc=np.array((p.argmax(1) == yte).mean()), z_max=z1.max(0),
                    grad_max=np.abs(d1).max(0))

    d = cached_npz("dead_relu", settings, build, cache_dir)
    return DeadRun(lr, d["steps"], d["dead"].astype(bool), d["active"], float(d["acc"]),
                   d["z_max"], d["grad_max"], {k: d[k] for k in ("W1", "b1", "W2", "b2")})


# ------------------------------------------------------------------ layout

PLOT_X = (0.03, 0.27, 0.51, 0.75)       # each plot's left edge (layout fraction)
PLOT_W, PLOT_Y = 0.21, (0.27, 0.50)     # width; top and bottom
PLOT_PAD = 6.0                          # px inside the plot frame
STACK_AT = (0.49, 0.72)                 # centre of the voxel rows
VOX = 0.9                               # voxel size of the rows
ROW_GAP = 2.6                           # world units between rows (room for a label each)
GRID_AT, GRID_SCALE = (0.21, 0.60), 1.7 # the 8 x 16 unit grid of the dead-ReLU network
CHART = ((0.46, 0.97), (0.30, 0.66))    # dead-count chart: (x0, x1), (top, bottom)
NOTE_X, NOTE_Y = 0.0, 0.875             # the read-outs along the bottom
NOTE_STEP = 0.05
LEGEND_AT = (0.49, 0.36)                # the chart's legend (top left, inside)
FRAME = (0.96, 0.96, 0.96)
PANEL = (0.955, 0.955, 0.95)
PANEL_EDGE = (0.70, 0.71, 0.72)
AXIS = (0.5, 0.52, 0.54)
HOT_INK = (0.75, 0.22, 0.22)
SANE_INK = tuple(float(v) for v in C.INPUT)
GRAD_INK = tuple(float(v) for v in C.darken(GRAD, 0.2))

T_FUNC0, FUNC_DT = 1.2, 2.2             # the four plots, one after another
T_LAYER0, LAYER_DT = 10.8, 2.4          # the z row, then one row per function
T_SWAP = 23.4                           # the dead-ReLU part fades in
SWEEP_EARLY = Tempo(slow=0.55, fast=0.11, ramp_in=8.0, ramp_out=1.0)
SWEEP_LATE_DT = 0.07                    # per later checkpoint
NOTES_DT = 5.0


@dataclass
class Frame:
    """What one frame shows (see :meth:`Activations._frame`)."""

    back: float
    intro: float
    plots: list                 # per function: 0..1 drawn (f first, then f')
    layer: list                 # z row, then per function: 0..1 appeared
    gone: float                 # the functions and the layer fading out (0 -> 1)
    part: float                 # the dead-ReLU part fading in afterwards (0 -> 1)
    cp: int                     # dead-ReLU checkpoint on show
    sweep: float                # progress through the sweep
    notes: float                # the read-outs appearing


class Activations(Diagram):
    """The functions, a real layer through each, then the dead ReLUs."""

    PHASES = ("functions", "a real layer", "dead ReLUs")

    def __init__(self, runs=None):
        super().__init__("Activation functions", "")
        self.z, self.digit = layer_preactivations()
        self.hot, self.sane = runs or (dead_relu_run(LR_HOT), dead_relu_run(LR_SANE))
        self.n = len(self.z)
        self.rows = self._rows()
        self.grid = self.sheet((HIDDEN // 16, 16), GRID_AT, GRID_SCALE, "units")
        self.hold, self.reset = 3.0, 1.2
        self._timeline()
        self.finish(self.Tm["notes"][1])
        # the dead unit the read-out quotes: the first to die for good
        first = np.argmax(self.hot.dead, axis=0)
        first[~self.hot.dead[-1]] = len(self.hot.steps)
        self.quoted = int(np.argmin(first))

    def _rows(self):
        """The z row and one row per function, stacked in world space."""
        rows = {}
        centre = self.cam.frac(*STACK_AT)
        keys = ["z"] + [f[0] for f in FUNCTIONS]
        height = (len(keys) - 1) * ROW_GAP
        for k, key in enumerate(keys):
            s = self.sheet((1, self.n), STACK_AT, VOX, key)
            c = centre.copy()
            c[2] = centre[2] + height / 2 - k * ROW_GAP
            s.block.place(c)
            rows[key] = s
        return rows

    def _timeline(self):
        Tm = dict(intro=(0.2, 1.2))
        for k, f in enumerate(FUNCTIONS):
            Tm["plot:" + f[0]] = (T_FUNC0 + k * FUNC_DT, T_FUNC0 + (k + 1) * FUNC_DT)
        Tm["layer:z"] = (T_LAYER0, T_LAYER0 + LAYER_DT)
        for k, f in enumerate(FUNCTIONS):
            Tm["layer:" + f[0]] = (T_LAYER0 + (k + 1) * LAYER_DT, T_LAYER0 + (k + 2) * LAYER_DT)
        Tm["swap"] = (T_SWAP, T_SWAP + 2.0)
        steps = self.hot.steps
        early = int(np.searchsorted(steps, EARLY, side="right"))
        durs = np.concatenate([SWEEP_EARLY.durations(early),
                               np.full(len(steps) - early, SWEEP_LATE_DT)])
        t0 = Tm["swap"][1] + 0.4
        self.cp_starts = t0 + np.concatenate([[0.0], np.cumsum(durs)[:-1]])
        Tm["sweep"] = (t0, t0 + float(durs.sum()))
        Tm["notes"] = (Tm["sweep"][1] + 0.3, Tm["sweep"][1] + 0.3 + NOTES_DT)
        self.Tm = Tm

    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        if t >= self.Tm["swap"][0]:
            return 2
        return 1 if t >= self.Tm["layer:z"][0] else 0

    # ------------------------------------------------------------------ state
    def _frame(self, t, reset):
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        intro = smoothstep(0.0, 1.0, self.progress("intro", t)) * back
        plots = [self.progress("plot:" + f[0], t) for f in FUNCTIONS]
        layer = [self.progress("layer:" + k, t) for k in ["z"] + [f[0] for f in FUNCTIONS]]
        swap = self.progress("swap", t)
        gone, part = smoothstep(0.0, 0.45, swap), smoothstep(0.55, 1.0, swap)
        sweep = self.progress("sweep", t)
        cp = 0
        if t >= self.Tm["sweep"][0]:
            cp = int(np.searchsorted(self.cp_starts, t, side="right") - 1)
        cp = min(cp, len(self.hot.steps) - 1)
        notes = smoothstep(0.0, 0.4, self.progress("notes", t)) * back
        return Frame(back, intro, plots, layer, gone, part, cp, sweep, notes)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        f = self._frame(t, reset)
        first = (1.0 - f.gone) * f.back
        for k, fn in enumerate(FUNCTIONS):
            self._plot(sc, k, fn, f, first)
        self._layer_rows(sc, f, first)
        self._dead(sc, f, f.part * f.back)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, f, reset)

    # -- the plots -----------------------------------------------------------------------
    def _plot_point(self, k, z, v, yr, lift=0.3):
        """World point of ``(z, v)`` in plot ``k`` whose y range is ``yr``."""
        x0, (y0, y1) = PLOT_X[k], PLOT_Y
        fx = x0 + PLOT_W * (z - Z_RANGE[0]) / (Z_RANGE[1] - Z_RANGE[0])
        fy = y1 - (y1 - y0) * (v - yr[0]) / (yr[1] - yr[0])
        return self.P(fx, fy, lift)

    def _plot(self, sc, k, fn, f, op):
        """Plot ``k``: the frame, the axes, f drawn then f', and the layer's
        values as dots on f once the layer is on show."""
        _key, name, fun, dfun, yr, _signed = fn
        u = f.plots[k]
        op *= f.intro
        if op <= 0.01:
            return
        x0, (y0, y1) = PLOT_X[k], PLOT_Y
        a, b = self.P(x0, y0), self.P(x0 + PLOT_W, y1)
        w, h = self.px(b) - self.px(a)
        self.rect(sc, (a + b) / 2, w, h, PANEL, op, border=PANEL_EDGE, border_w=1.0, radius=4,
                  lift=0.1)
        zs = np.linspace(*Z_RANGE, 121)
        zs = zs[zs != 0.0]                               # ReLU' jumps at 0
        self.polyline(sc, [self._plot_point(k, z, 0.0, yr, 0.2) for z in Z_RANGE], 1.0, AXIS,
                      0.6 * op, bias=0.1)
        self.polyline(sc, [self._plot_point(k, 0.0, v, yr, 0.2) for v in yr], 1.0, AXIS,
                      0.6 * op, bias=0.1)
        self.label(sc, self.P(x0, y0), name, layout.MIN_TEXT, anchor="lb", dx=2, dy=-5,
                   opacity=op, ink=layout.STRONG_INK)
        for vals, color, width, u0, u1 in ((fun(zs), DATA, 2.0, 0.0, 0.55),
                                           (dfun(zs), GRAD, 1.6, 0.5, 1.0)):
            m = int(np.ceil(len(zs) * remap(u, u0, u1)))
            if m > 1:
                pts = [self._plot_point(k, z, float(np.clip(v, *yr)), yr) for z, v in
                       zip(zs[:m], vals[:m])]
                self.polyline(sc, pts, width, color, op * 0.95, bias=0.2)
        self.label(sc, self.P(x0 + PLOT_W, y0), "f′", layout.MIN_TEXT, anchor="rb", dx=-2, dy=-5,
                   opacity=op * smoothstep(0.5, 0.7, u), ink=GRAD_INK)
        dots = smoothstep(0.0, 0.6, f.layer[k + 1]) if f.layer[0] > 0 else 0.0
        if dots > 0:
            for z in self.z:
                self.disk(sc, self._plot_point(k, float(np.clip(z, *Z_RANGE)),
                                               float(np.clip(fun(z), *yr)), yr, 0.5),
                          2.6, DATA, 0.6 * op * dots, lift=0.6, outline=0.0)

    # -- the layer's rows --------------------------------------------------------------
    def row_values(self, key):
        """The 64 values of row ``key`` and whether they are signed."""
        if key == "z":
            return self.z, True
        _k, _name, fun, _d, _yr, signed = next(fn for fn in FUNCTIONS if fn[0] == key)
        return fun(self.z), signed

    def counts(self, key):
        """What the row's read-out counts: ``(n, what)``."""
        _k, _name, fun, dfun, _yr, _signed = next(fn for fn in FUNCTIONS if fn[0] == key)
        if key == "relu":
            return int((fun(self.z) == 0).sum()), "exactly 0"
        if key == "gelu":
            return int((fun(self.z) < 0).sum()), "below 0"
        return int((np.abs(dfun(self.z)) < SATURATED).sum()), "saturated"

    def _layer_rows(self, sc, f, op):
        keys = ["z"] + [fn[0] for fn in FUNCTIONS]
        labels = {"z": "z: %d pre-activations" % self.n, "sigmoid": "sigmoid σ(z)",
                  "tanh": "tanh(z)", "relu": "ReLU", "gelu": "GELU"}
        col = np.arange(self.n)
        for k, key in enumerate(keys):
            s = self.rows[key]
            on = f.layer[k]
            if on <= 0 or op <= 0.01:
                s.visible = False
                continue
            v, signed = self.row_values(key)
            s.fill(div_colors(v, amax(v)) if signed else lit_colors(v, amax(v)))
            s.show(col < on * self.n * 1.08)
            s.cage = op
            s.fade(op)
            self.caption_left(sc, s, labels[key], op)
            if key != "z":
                n, what = self.counts(key)
                self.caption_side(sc, s, "%d of %d %s" % (n, self.n, what),
                                  op * smoothstep(0.9, 1.0, on))
            else:
                self.caption_side(sc, s, "%d of %d negative" % (int((v < 0).sum()), self.n),
                                  op * smoothstep(0.9, 1.0, on))

    # -- dead ReLUs --------------------------------------------------------------------
    def _dead(self, sc, f, op):
        g = self.grid
        if op <= 0.01:
            g.visible = False
            return
        cp = f.cp
        active = self.hot.active[cp]
        g.fill(lit_colors(active, 1.0))
        g.cage = op
        g.fade(op)
        if f.notes > 0:                                   # the quoted unit framed
            r, c = divmod(self.quoted, 16)
            lo, hi = g.block.box((r, r + 1), (c, c + 1), (0, 1))
            sc.wire_box(lo, hi, inflate=0.15 * GRID_SCALE, color=FRAME, width=1.6,
                        opacity=0.95 * f.notes, bias=0.1)
        step = int(self.hot.steps[cp])
        n_dead = int(self.hot.dead[cp].sum())
        self.caption(sc, g, "%d hidden units, lr %s, step %d" % (HIDDEN, num(LR_HOT, 1), step),
                     op, above=True, size=layout.MIN_TEXT)
        self.caption(sc, g, "lit: how often a unit fires on the test set\n"
                     "dark: %d dead (never fire)" % n_dead, op, size=layout.MIN_TEXT,
                     max_width=0.36)
        self._chart(sc, f, op)
        self._notes(sc, f, op)

    def _chart(self, sc, f, op):
        """Dead units against the training step (log axis), both learning rates."""
        (x0, x1), (y0, y1) = CHART
        steps = self.hot.steps
        last = float(steps[-1])
        top = max(int(self.hot.counts.max()), 1)
        top = int(np.ceil(top / 10.0) * 10)

        def fx(step):
            return x0 + (x1 - x0) * np.log10(1 + step) / np.log10(1 + last)

        def pt(step, n):
            return self.P(fx(step), y1 - (y1 - y0) * n / top, 0.3)
        sc.line(self.P(x0, y1), self.P(x1, y1), width=1.0, color=AXIS, opacity=0.7 * op, bias=0.0)
        sc.line(self.P(x0, y1), self.P(x0, y0), width=1.0, color=AXIS, opacity=0.7 * op, bias=0.0)
        for step in (1, 10, 100):
            self.label(sc, self.P(fx(step), y1), str(step), layout.MIN_TEXT, anchor="mt", dy=5,
                       opacity=op, ink=layout.NOTE_INK)
        self.label(sc, self.P(x1, y1), "%d" % last, layout.MIN_TEXT, anchor="mt", dy=5, opacity=op,
                   ink=layout.NOTE_INK)
        self.label(sc, self.P((x0 + x1) / 2, y1), "training step (1 epoch = %d steps of %d images)"
                   % (last, BATCH), layout.MIN_TEXT, anchor="mt", dy=26, opacity=op)
        self.label(sc, self.P(x0, y0), "dead units", layout.MIN_TEXT, anchor="lb", dy=-6,
                   opacity=op)
        self.label(sc, self.P(x0, y0), "%d" % top, layout.MIN_TEXT, anchor="rm", dx=-6,
                   opacity=op, ink=layout.NOTE_INK)
        self.label(sc, self.P(x0, y1), "0", layout.MIN_TEXT, anchor="rm", dx=-6, opacity=op,
                   ink=layout.NOTE_INK)
        lx, ly = LEGEND_AT
        for n, (run, ink) in enumerate(((self.hot, HOT_INK), (self.sane, SANE_INK))):
            k = f.cp
            pts = [pt(float(s), float(c)) for s, c in zip(run.steps[:k + 1], run.counts[:k + 1])]
            if len(pts) > 1:
                self.polyline(sc, pts, 2.0, ink, 0.9 * op, bias=0.2)
            self.disk(sc, pts[-1], 3.4, ink, op, lift=0.6, outline=0.0)
            self.label(sc, self.P(lx, ly + 0.05 * n), "lr %s: %d dead"
                       % (num(run.lr, 1), int(run.counts[k])), layout.MIN_TEXT, anchor="lm",
                       opacity=op, ink=ink)

    def _notes(self, sc, f, op):
        """Along the bottom: the end of training, and the gradient through one
        dead unit, exactly 0."""
        op *= f.notes
        if op <= 0.01:
            return
        hot, sane, u = self.hot, self.sane, self.quoted
        died = int(hot.steps[np.argmax(hot.dead[:, u])])
        lines = [
            ("after 1 epoch: %d of %d dead at lr %s (%.1f%% test accuracy); %d at lr %s (%.1f%%)"
             % (hot.n_dead, HIDDEN, num(LR_HOT, 1), 100 * hot.accuracy, sane.n_dead,
                num(LR_SANE, 1), 100 * sane.accuracy), layout.STRONG_INK),
            ("framed unit: dead since step %d; largest z on the 10,000 test images %s"
             % (died, num(hot.z_max[u], 2)), layout.CAPTION_INK),
            ("ReLU′(z) = 0 there: its gradient ∂L/∂z = %s exactly, for every image"
             % num(hot.grad_max[u], 6), GRAD_INK),
        ]
        for n, (text, ink) in enumerate(lines):
            self.label(sc, self.P(NOTE_X, NOTE_Y + n * NOTE_STEP), text, layout.MIN_TEXT,
                       anchor="lm", opacity=op, ink=ink)

    # ------------------------------------------------------------------ words
    def _words(self, t, f, reset):
        T = self.Tm
        if reset > 0 or t < T["plot:sigmoid"][0]:
            return ("an activation function f turns a neuron's pre-activation z = w · x + b into "
                    "its output f(z); backprop multiplies the gradient by the slope f′(z)")
        if t < T["layer:z"][0]:
            k = max(i for i, fn in enumerate(FUNCTIONS) if t >= T["plot:" + fn[0]][0])
            return [
                "sigmoid σ(z) = 1 / (1 + exp(−z)) squashes z into (0, 1); its slope σ′ = σ(1 − σ) "
                "peaks at 0.25 and vanishes at both ends",
                "tanh squashes into (−1, 1), centred on 0; its slope 1 − tanh² reaches 1 and "
                "also vanishes at both ends",
                "ReLU = max(0, z): slope exactly 1 for z > 0, exactly 0 for z < 0. No saturation "
                "on the right; nothing passes on the left",
                "GELU = z Φ(z) (tanh form, as in GPT-2): a smooth ReLU that lets small negative "
                "values through; its slope dips to %s and tops out above 1"
                % num(d_gelu(np.linspace(-4, 0, 400)).min(), 2),
            ][k]
        if t < T["swap"][0]:
            return self._layer_words(t)
        if t < T["sweep"][0]:
            return ("dead ReLUs: a %d-%d-10 ReLU network trained on MNIST by plain SGD with a "
                    "learning rate of %s, batches of %d (%s would do)"
                    % (784, HIDDEN, num(LR_HOT, 1), BATCH, num(LR_SANE, 1)))
        if t < T["notes"][0]:
            step, n = int(self.hot.steps[f.cp]), int(self.hot.dead[f.cp].sum())
            return ("step %d: %d of %d units output 0 for every one of the 10,000 test images. "
                    "Dead: ReLU′ = 0 means no gradient reaches their weights" % (step, n, HIDDEN))
        since = int(self.hot.steps[np.argmax(self.hot.counts == self.hot.counts.max())])
        return ("after one epoch %d of %d are dead, all since step %d, none revived; at lr %s none "
                "die. Both reach the same accuracy: %.1f%% and %.1f%%: the %d units were lost for "
                "nothing" % (self.hot.n_dead, HIDDEN, since, num(LR_SANE, 1),
                             100 * self.hot.accuracy, 100 * self.sane.accuracy, self.hot.n_dead))

    def _layer_words(self, t):
        T = self.Tm
        z = self.z
        if t < T["layer:sigmoid"][0]:
            return ("a real layer: the %d pre-activations of the MNIST MLP's first hidden layer on "
                    "test image %d (a %d): z from %s to %s, %d of %d negative"
                    % (self.n, LAYER_INDEX, self.digit, num(z.min(), 2), num(z.max(), 2),
                       int((z < 0).sum()), self.n))
        key = max((fn[0] for fn in FUNCTIONS if t >= T["layer:" + fn[0]][0]),
                  key=lambda k: T["layer:" + k][0])
        n, _what = self.counts(key)
        return {
            "sigmoid": "through the sigmoid (the dots on its curve): %d of %d sit on the flat ends "
                       "(σ′ < %g), saturated; their gradient is almost blocked"
                       % (n, self.n, SATURATED),
            "tanh": "through tanh: %d of %d saturated; signed outputs (orange +, blue −) with "
                    "the large |z| all at ±1" % (n, self.n),
            "relu": "through ReLU: the %d negative ones become exactly 0 (dark): a sparse layer, "
                    "and their gradient is exactly 0 too" % n,
            "gelu": "through GELU: %d of %d come out slightly negative (down to %s) instead of 0; "
                    "the large positive ones pass unchanged"
                    % (n, self.n, num(gelu(z).min(), 2)),
        }[key]

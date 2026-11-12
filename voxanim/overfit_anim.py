"""Overfitting, and four cures, on a two-dimensional problem where it is real.

The data: MNIST 3s and 8s, each image reduced to two numbers, its
coordinates on the first two principal components of the 60 training
images (computed here, in numpy, by an SVD of the centred images).  Sixty
training points in a plane, a 2-128-128-2 tanh MLP trained by full-batch
Adam, a validation set of 200 held-out digits and a test set of 1000 more.

    the plane: the decision field p(8), the training points      loss curves

1. **data** -- the points,
2. **overfitting** -- 2000 steps (on a log time axis): the field wraps itself
   around every training point, the training loss falls to 0.001 while the
   test loss climbs from 0.46 to 1.9 and the test accuracy drops from 79%
   to 71%,
3. **weight decay** -- the same with an L2 penalty on the weights: the
   boundary stays smooth, the training loss stops at 0.30, test accuracy
   80%,
4. **dropout** -- each hidden unit dropped with probability 0.5 on every
   step (the live mask of one example is shown): test accuracy 78%,
5. **early stopping** -- the first run again, watching the validation loss:
   keep the weights from its lowest point, step 24, where the test accuracy
   is 79%.

Every run is a recorded :func:`voxanim.deepnet.train`; the field is the
network's p(8) on a 40 x 40 grid at each recorded step, interpolated
linearly between steps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import color as C
from . import layout
from .deepnet import Data, Spec, subset, train
from .diagram import EDGE, Diagram, draw_image
from .easing import ease_in_out, remap, smoothstep
from .losschart import LossChart
from .netanim import ACT_HI, OFF

CLASSES = (3, 8)
N_TRAIN, N_VAL, N_TEST, SEED = 60, 200, 1000, 0
GRID, RANGE = 40, 3.0                            # the decision field: 40 x 40 over (-3, 3)^2
SIZES = (2, 128, 128, 2)
WEIGHT_DECAY, DROPOUT = 1e-2, 0.5
RUNS = (("plain", Spec(SIZES, "tanh")),
        ("weight decay", Spec(SIZES, "tanh", weight_decay=WEIGHT_DECAY)),
        ("dropout", Spec(SIZES, "tanh", dropout=DROPOUT)))
TRAIN = dict(steps=2000, batch=N_TRAIN, lr=0.001, opt="adam", probe_layer=0)
CLASS_COLORS = (C.rgb("#e0962a"), C.rgb("#3f7fd0"))          # a 3, an 8
TRAIN_INK = tuple(float(v) for v in C.INPUT)
TEST_INK = (0.75, 0.22, 0.22)
VAL_INK = (0.40, 0.42, 0.46)
PLOT_AT, PLOT_PX = (0.255, 0.60), 290.0          # the plane: layout centre, size in px
CHART = ((0.58, 0.94), (0.30, 0.60))             # the loss chart ((x0, x1), (top, bottom))
NOTE_AT = (0.58, 0.70)                           # the read-outs, layout
MASK_AT = (0.76, 0.915)                          # the dropout mask strip, layout centre
DIGITS_AT, DIGIT_PX, DIGIT_COLS = (0.76, 0.56), 26.0, 10     # the 60 training images, data phase
MASK_SHAPE = (4, 32)                             # 128 hidden units as a strip
MASK_CELL = 6.0                                  # px per unit
SPANS = (9.0, 7.0, 7.0, 7.0)                     # seconds each run plays for
R_TRAIN, R_VAL = 4.2, 1.8                        # point radii, px


def pca2(x):
    """``(mean, components (2, 784))``: the first two principal axes of the rows of ``x``."""
    mu = x.mean(0)
    _u, _s, vt = np.linalg.svd(x - mu, full_matrices=False)
    return mu, vt[:2]


def dataset(cache_dir=None):
    """``(data, images)``: the 2-D problem, 3s and 8s on their first two
    principal components (fitted on the training images, each component
    scaled to unit std), and the training images themselves ``(60, 784)``."""
    raw = subset(N_TRAIN, N_VAL + N_TEST, CLASSES, SEED, cache_dir)
    mu, V = pca2(raw.x_train)
    ztr = (raw.x_train - mu) @ V.T
    scale = ztr.std(0)
    zte = ((raw.x_test - mu) @ V.T) / scale
    return Data(ztr / scale, raw.y_train, zte[N_VAL:], raw.y_test[N_VAL:], 2,
                zte[:N_VAL], raw.y_test[:N_VAL]), raw.x_train


def grid():
    """``(GRID * GRID, 2)`` inputs, row-major from the top-left corner of the plane."""
    gx, gy = np.meshgrid(np.linspace(-RANGE, RANGE, GRID), np.linspace(RANGE, -RANGE, GRID))
    return np.stack([gx.ravel(), gy.ravel()], 1)


def runs(cache_dir=None):
    """``(data, images, runs)``: the problem, its training images and the
    three recorded runs."""
    data, images = dataset(cache_dir)
    kw = dict(TRAIN, grid=grid())
    if cache_dir:
        kw["cache_dir"] = cache_dir
    return data, images, [train(spec, data, **kw) for _name, spec in RUNS]


@dataclass
class Frame:
    """What one instant shows: the run on screen, how far through it, the
    (fractional) recorded-step index and the loop fade."""

    back: float
    phase: int                  # index into PHASES
    run: int                    # index into the runs (-1: none yet)
    u: float                    # 0..1 through the run's playback
    x: float                    # fractional index into the run's recorded steps
    stop: float                 # early stopping: 1 once the run has passed its best step


class Overfitting(Diagram):
    """A wide MLP overfitting 60 points, then weight decay, dropout and early
    stopping (see the module docstring)."""

    PHASES = ("data", "overfitting", "weight decay", "dropout", "early stopping")

    def __init__(self, data, images, runs):
        super().__init__("Overfitting, and four cures", "")
        self.data, self.images, self.runs = data, images, runs
        self.hold = 2.5
        self.plot = self.P(*PLOT_AT)
        self.plain = runs[0]
        self.best = self.plain.best                              # early stopping's step index
        self.S = len(self.plain.steps)
        self.chart = LossChart(CHART, (1, TRAIN["steps"]), (0.0, 2.5), xlog=True)
        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("data", 4.5)
        for name, span, hold in zip(("plain", "wd", "drop", "stop"), SPANS, (2.0, 1.5, 1.5, 0.3)):
            add(name, span, hold)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["plain"][0], Tm["wd"][0], Tm["drop"][0], Tm["stop"][0]]

    # ------------------------------------------------------------------ the frame's state
    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _frame(self, t, reset):
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        if ph == 0:
            return Frame(back, 0, -1, 0.0, 0.0, 0.0)
        name = ("plain", "wd", "drop", "stop")[ph - 1]
        u = self.progress(name, t)
        run = (0, 1, 2, 0)[ph - 1]
        x = u * (self.S - 1)
        stop = smoothstep(self.best, self.best + 1.0, x) if ph == 4 else 0.0
        return Frame(back, ph, run, u, x, stop)

    def field(self, f):
        """The decision field p(8) on the grid at frame ``f``, ``(GRID, GRID)``."""
        fields = self.runs[f.run].field
        x = min(f.x, self.best) if f.phase == 4 else f.x
        i = int(np.floor(x))
        a = x - i
        p = fields[i] if a <= 0 or i + 1 >= self.S else fields[i] * (1 - a) + fields[i + 1] * a
        return p.reshape(GRID, GRID)

    def at_step(self, f):
        """``(step, train loss, test loss, test accuracy)`` on show at frame ``f``."""
        run = self.runs[f.run]
        x = min(f.x, self.best) if f.phase == 4 else f.x
        i = int(round(x))
        return int(run.steps[i]), run.train_loss[i], run.test_loss[i], run.test_acc[i]

    # ------------------------------------------------------------------ drawing
    def pp(self, x1, x2, lift=0.0):
        """Feature space to a world point on the plane."""
        return self.shift(self.plot, x1 / RANGE * PLOT_PX / 2, -x2 / RANGE * PLOT_PX / 2) + \
            self.view * lift

    def draw(self, sc, t, reset):
        """Draw the frame at time ``t``."""
        f = self._frame(t, reset)
        self.subtitle = self._words(f)
        op = f.back * smoothstep(0.0, 0.5, t)
        self._plane(sc, t, f, op)
        self._digits(sc, t, f, op)
        self._chart(sc, f, op)
        self._notes(sc, f, op)
        self._mask(sc, f, op)
        self.step_indicator(sc, f.phase, self.PHASES)

    def _plane(self, sc, t, f, op):
        """The field, the axes and the points."""
        c = self.plot
        if f.run >= 0:
            p = self.field(f)
            tex = CLASS_COLORS[0][None, None] * (1 - p[..., None]) + \
                CLASS_COLORS[1][None, None] * p[..., None]
            tex = tex + (1.0 - tex) * 0.55                             # pale, points on top
            draw_image(self, sc, c, tex, PLOT_PX, PLOT_PX, op, grid=0.0, lift=0.1)
        half = PLOT_PX / 2
        corners = [self.shift(c, sx * half, sy * half) for sx, sy in
                   ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]
        self.polyline(sc, corners, width=1.2, color=EDGE, opacity=0.8 * op, bias=0.3)
        self.label(sc, self.shift(c, 0, half + 10), "PC 1", anchor="mt", opacity=op,
                   ink=layout.NOTE_INK)
        self.label(sc, self.shift(c, -half - 8, 0), "PC 2", anchor="rm", opacity=op,
                   ink=layout.NOTE_INK)
        # held-out points, faint; training points popping in during the data phase
        pop = self.progress("data", t) if f.phase == 0 else 1.0
        for (x1, x2), k in zip(self.data.x_val, self.data.y_val):
            if max(abs(x1), abs(x2)) < RANGE:
                self.disk(sc, self.pp(x1, x2), R_VAL, CLASS_COLORS[k], 0.35 * op * pop, lift=0.5,
                          outline=0.0)
        n = len(self.data.y_train)
        for i, ((x1, x2), k) in enumerate(zip(self.data.x_train, self.data.y_train)):
            on = smoothstep(0.0, 0.2, remap(pop, 0.7 * i / n, 0.7 * i / n + 0.3)) * op
            self.disk(sc, self.pp(x1, x2), R_TRAIN, CLASS_COLORS[k], on, lift=0.8, outline=0.3)
        # legend, top-left corner of the plane
        for j, (name, col) in enumerate((("3", CLASS_COLORS[0]), ("8", CLASS_COLORS[1]))):
            q = self.shift(c, -half + 14 + 40 * j, -half - 12)
            self.disk(sc, q, 4.0, col, op, lift=0.8, outline=0.3)
            self.label(sc, q, name, anchor="lm", dx=9, opacity=op)

    def _digits(self, sc, t, f, op):
        """The data phase: the 60 training images, where the chart will be."""
        if f.phase != 0:
            return
        op *= self.progress("data", t) if t < self.Tm["data"][1] else 1.0
        if op <= 0.01:
            return
        rows = -(-len(self.images) // DIGIT_COLS)
        c = self.P(*DIGITS_AT)
        for i, (img, k) in enumerate(zip(self.images, self.data.y_train)):
            r, col = divmod(i, DIGIT_COLS)
            q = self.shift(c, (col - (DIGIT_COLS - 1) / 2) * (DIGIT_PX + 3),
                           (r - (rows - 1) / 2) * (DIGIT_PX + 3))
            tint = C.mix((1, 1, 1), CLASS_COLORS[k], 0.75)
            tex = (1.0 - img.reshape(28, 28))[..., None] * (1.0 - tint)[None, None]
            draw_image(self, sc, q, 1.0 - tex, DIGIT_PX, DIGIT_PX, op, grid=0.0, lift=0.1)
        self.label(sc, self.shift(c, 0, -rows / 2 * (DIGIT_PX + 3) - 10),
                   "the %d training images" % len(self.images), anchor="mb", opacity=op)

    def _chart(self, sc, f, op):
        """Training and test loss against the step (log axis), drawn as the run plays."""
        if f.run < 0:
            return
        run = self.runs[f.run]
        place = lambda fx, fy: self.P(fx, fy, 0.3)                          # noqa: E731
        ticks = [(s, f"{s:,}") for s in (1, 10, 100, 1000)]
        self.chart.axes(sc, self.cam, place, op, "training steps", "loss", ticks,
                        [(v, "%g" % v) for v in (0, 1, 2)])
        i = int(np.floor(f.x))
        a = f.x - i
        upto = run.steps[i] if i + 1 >= self.S else run.steps[i] * (1 - a) + run.steps[i + 1] * a
        upto = max(upto, 1.0)
        curves = [(run.train_loss, TRAIN_INK), (run.test_loss, TEST_INK)]
        if f.phase == 4:
            curves.append((run.val_loss, VAL_INK))
        for values, ink in curves:
            if f.phase == 4 and f.stop > 0:
                stop_step = run.steps[self.best]
                self.chart.curve(sc, place, run.steps, values, stop_step, ink, op)
                self.chart.curve(sc, place, run.steps, values, upto, ink, 0.3 * op, dot=False)
            else:
                self.chart.curve(sc, place, run.steps, values, upto, ink, op)
        if f.phase == 4 and f.stop > 0:
            self.chart.mark(sc, place, run.steps[self.best], layout.CAPTION_INK, op * f.stop)
            self.label(sc, place(self.chart.fx(run.steps[self.best]), CHART[1][0] - 0.02),
                       "stop: step %d" % run.steps[self.best], anchor="mb", opacity=op * f.stop)
        legend = [("training loss", TRAIN_INK), ("test loss", TEST_INK)]
        if f.phase == 4:
            legend.append(("validation loss", VAL_INK))
        self.chart.legend(sc, self.cam, legend, op, (CHART[0][0] + 0.02, CHART[1][0] + 0.03),
                          step=0.038)

    def _notes(self, sc, f, op):
        """The step and the numbers at it."""
        if f.run < 0:
            return
        step, tr, te, acc = self.at_step(f)
        fx, fy = NOTE_AT
        lines = (("step %s" % f"{step:,}", layout.CAPTION_INK, 0.029),
                 ("training loss %.3f    test loss %.2f" % (tr, te), layout.STRONG_INK, 0.031),
                 ("test accuracy %d%% (%s points)" % (round(100 * acc), f"{N_TEST:,}"),
                  layout.STRONG_INK, 0.031))
        for n, (text, ink, size) in enumerate(lines):
            self.label(sc, self.P(fx, fy + 0.055 * n), text, size, anchor="lm", ink=ink,
                       opacity=op)

    def _mask(self, sc, f, op):
        """Dropout: the first hidden layer's mask for one example at the step on show."""
        if f.phase != 3:
            return
        run = self.runs[f.run]
        i = int(round(f.x))
        m = run.probe["m"][i, 0].reshape(MASK_SHAPE)
        tex = np.where(m[..., None] > 0, ACT_HI[None, None], OFF[None, None])
        c = self.P(*MASK_AT)
        w, h = MASK_SHAPE[1] * MASK_CELL, MASK_SHAPE[0] * MASK_CELL
        draw_image(self, sc, c, tex, w, h, op, grid=0.5, lift=0.2)
        kept = int(m.sum())
        self.label(sc, self.shift(c, 0, -h / 2 - 8), "hidden layer 1 at this step: %d of %d units "
                   "kept" % (kept, m.size), anchor="mb", opacity=op)

    # ------------------------------------------------------------------ words
    def _words(self, f):
        """The subtitle for frame ``f``: what is happening, with the real numbers."""
        plain, wd, drop = self.runs
        end = self.S - 1
        b = self.best
        return [
            "two MNIST classes, 3 and 8, as two numbers each: the first two principal "
            "components of the %d training images; %d held-out digits faint" % (N_TRAIN, N_VAL),
            "a 2-%d-%d-2 tanh MLP, Adam, full batch: the training loss falls to %.3f while the "
            "test loss climbs from %.2f to %.2f; test accuracy %d%% → %d%%"
            % (SIZES[1], SIZES[2], plain.train_loss[end], plain.test_loss[b],
               plain.test_loss[end], round(100 * plain.test_acc[b]),
               round(100 * plain.test_acc[end])),
            "weight decay %g: the L2 penalty keeps the weights small and the boundary smooth; "
            "the training loss stops at %.2f, test accuracy %d%%"
            % (WEIGHT_DECAY, wd.train_loss[end], round(100 * wd.test_acc[end])),
            "dropout: on every step each hidden unit is dropped with probability %.1f (one "
            "example's mask below); test loss %.2f, accuracy %d%%"
            % (DROPOUT, drop.test_loss[end], round(100 * drop.test_acc[end])),
            "early stopping: watch a validation set and keep the weights from its lowest loss, "
            "step %d: test loss %.2f, accuracy %d%% (%d%% at step %s)"
            % (plain.steps[b], plain.test_loss[b], round(100 * plain.test_acc[b]),
               round(100 * plain.test_acc[end]), f"{TRAIN['steps']:,}"),
        ][f.phase]

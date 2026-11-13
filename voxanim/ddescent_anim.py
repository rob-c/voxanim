"""Double descent: test error against width for a one-hidden-layer MLP
trained to interpolation on a small, noisily labelled MNIST subset.

The setting (chosen by sweeping, see the report in the README): 1,000
training images, 20% of their labels replaced by a random wrong class
(:func:`voxanim.generalise.corrupt`), a 784-w-10 ReLU network (He init) for
each width ``w`` in :data:`WIDTHS`, every one trained by Adam (lr 0.001,
minibatches of 100) for 1,000 epochs (10,000 steps) -- long enough that
every width from 8 up fits all 1,000 labels, the false ones included.  The
test error is on all 10,000 test digits.  A 784-w-10 network has
``795 w + 10`` parameters; ``n x classes = 10,000`` labels' worth of
constraints puts the interpolation threshold near ``w = 12``, and the
networks here first reach zero training error at ``w = 8``.

1. **setup** -- the data: training digits with their labels, the false ones
   in red; the model,
2. **sweep** -- the widths one by one: each trained network's hidden layer
   to size, its confusion matrix on the test set, and its training and test
   error dropping onto the chart (log width axis),
3. **the U** -- the classic curve: error falls with width, then rises to its
   peak at the width that just manages to fit every label,
4. **second descent** -- past the peak the error falls again, below the U's
   minimum, with the training error at zero throughout,
5. **why** -- how smooth the function is: the output's response to pixel
   noise (mean L1 change of the softmax under noise of std 0.1 on 2,000 test
   images), per width: the wide interpolators are the smoothest.

(The numbers are this machine's; CI retrains.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import color as C
from . import layout
from .conv import Tempo
from .deepnet import Data, Spec, subset, train
from .diagram import Diagram, draw_image
from .easing import ease_in_out, ease_out, remap, smoothstep
from .gen_common import (
    DIGIT_INK,
    PAPER,
    TEST_INK,
    TRAIN_INK,
    WRONG_INK,
    draw_digit,
    note,
    readout,
    strong,
)
from .generalise import confusion, corrupt, error_rate, predictions, sensitivity
from .losschart import LossChart
from .sweep import Sweep

N_TRAIN, N_TEST, SEED = 1000, 10000, 0
NOISE, NOISE_SEED = 0.2, 0
WIDTHS = (2, 3, 4, 6, 8, 12, 16, 25, 40, 64, 100, 200, 400)
EPOCHS = 1000
TRAIN = dict(batch=100, lr=0.001, opt="adam")
INTERPOLATES = 0.005                              # training error counted as zero below this
SENS_SIGMA, SENS_N, SENS_SEED = 0.1, 2000, 1      # the smoothness measurement
INK_WIDTH = C.rgb("#3d9e45")
GRID_COLS, GRID_ROWS, GRID_PX = 10, 4, 28.0       # training digits of the setup phase
GRID_AT = (0.74, 0.50)
CHART = ((0.09, 0.45), (0.30, 0.74))              # error against width
ERR_RANGE = (0.0, 0.7)
GLYPH_AT, GLYPH_PX, GLYPH_COLS = (0.63, 0.46), 7.0, 20   # the hidden layer, sweep phase
CONF_AT, CONF_PX = (0.84, 0.46), 130.0            # the confusion matrix, sweep phase
TRIO_Y, TRIO_PX, TRIO_X = 0.50, 110.0, (0.60, 0.76, 0.92)   # three matrices, later phases
WHY_CHART = ((0.60, 0.94), (0.32, 0.64))          # sensitivity against width
NOTE_AT = (0.60, 0.80)


def dataset(cache_dir=None):
    """``(data, mask)``: the noisy problem and which labels are false."""
    raw = subset(N_TRAIN, N_TEST, None, SEED, cache_dir)
    y, mask = corrupt(raw.y_train, NOISE, 10, np.random.default_rng(NOISE_SEED))
    return Data(raw.x_train, y, raw.x_test, raw.y_test, 10), mask


def runs(cache_dir=None):
    """``(data, mask, runs)``: one recorded run per width."""
    data, mask = dataset(cache_dir)
    steps = EPOCHS * (N_TRAIN // TRAIN["batch"])
    kw = dict(TRAIN, steps=steps)
    if cache_dir:
        kw["cache_dir"] = cache_dir
    return data, mask, [train(Spec((784, w, 10), "relu", "he"), data, **kw) for w in WIDTHS]


def params(w):
    """Parameters of a 784-w-10 network."""
    return 784 * w + w + w * 10 + 10


def threshold_width():
    """The width at which the parameters equal the labels' constraints,
    ``n_train * classes``."""
    return (N_TRAIN * 10 - 10) / 795.0


def landmarks(train_err, test_err):
    """``(u_min, peak, wide)``: indices of the U's lowest point, the peak and
    the widest network.  The peak is the largest test error among the widths
    within a factor of two of the first that interpolates; the U's minimum
    the smallest test error at a narrower width than the peak."""
    widths = np.asarray(WIDTHS, float)
    fits = np.nonzero(train_err <= INTERPOLATES)[0]
    first = widths[fits[0]] if len(fits) else widths[-1]
    near = np.nonzero((widths >= first / 2) & (widths <= first * 2))[0]
    peak = int(near[np.argmax(test_err[near])])
    u_min = int(np.argmin(test_err[:peak])) if peak > 0 else 0
    return u_min, peak, len(WIDTHS) - 1


@dataclass
class Frame:
    """What one instant shows."""

    back: float
    phase: int
    done: int        # widths finished (sweep)
    cur: int         # the width under way (-1: none)
    cu: float        # its progress
    grow: float      # the why-chart's growth


class DoubleDescent(Diagram):
    """Test error against width through the interpolation threshold (see the
    module docstring)."""

    PHASES = ("setup", "sweep", "the U", "second descent", "why")

    def __init__(self, data, mask, runs):
        super().__init__("Double descent", "")
        self.data, self.mask, self.runs = data, mask, runs
        self.hold = 2.5
        x, y = data.x_train, data.y_train
        self.train_err = np.array([error_rate(r.spec, r.P, x, y) for r in runs])
        self.test_err = np.array([error_rate(r.spec, r.P, data.x_test, data.y_test)
                                  for r in runs])
        self.sens = np.array([sensitivity(r.spec, r.P, data.x_test[:SENS_N],
                                          np.random.default_rng(SENS_SEED), SENS_SIGMA)
                              for r in runs])
        self.conf = [confusion(predictions(r.spec, r.P, data.x_test), data.y_test, 10)
                     for r in runs]
        self.u_min, self.peak, self.wide = landmarks(self.train_err, self.test_err)
        self.chart = LossChart(CHART, (WIDTHS[0], WIDTHS[-1]), ERR_RANGE, xlog=True)
        self.why = LossChart(WHY_CHART, (WIDTHS[0], WIDTHS[-1]),
                             (0.0, float(self.sens.max()) * 1.15), xlog=True)
        Tm, t = {}, 0.0
        Tm["setup"] = (t, t + 6.0)
        t += 6.0 + 0.3
        self.sweep = Sweep(t, len(WIDTHS), Tempo(slow=1.6, fast=0.8, ramp_in=3.0, ramp_out=2.0))
        Tm["sweep"] = (t, self.sweep.t1 + 1.0)
        t = self.sweep.t1 + 1.0 + 0.3
        for name in ("u", "descent", "why"):
            Tm[name] = (t, t + 7.0)
            t += 7.0 + 0.3
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["sweep"][0], Tm["u"][0], Tm["descent"][0], Tm["why"][0]]

    # ------------------------------------------------------------------ state
    def _frame(self, t, reset):
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else int(np.searchsorted(self.marks, t, side="right") - 1)
        done, cur, cu = self.sweep.at(t) if ph >= 1 else (0, -1, 0.0)
        grow = ease_out(self.progress("why", t)) if ph == 4 else 0.0
        return Frame(back, ph, done, cur if ph == 1 else -1, cu, grow)

    # ------------------------------------------------------------------ drawing
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t``."""
        f = self._frame(t, reset)
        self.subtitle = self._words(f)
        op = f.back * smoothstep(0.0, 0.5, t)
        if f.phase == 0:
            self._setup(sc, t, op)
        elif f.phase == 1:
            self._current(sc, f, op)
        elif f.phase in (2, 3):
            self._trio(sc, f, op)
        else:
            self._why(sc, f, op)
        if f.phase >= 1:
            self._chart(sc, f, op)
        self.step_indicator(sc, f.phase, self.PHASES)

    def _setup(self, sc, t, op):
        """Training digits with their labels, the false ones in red."""
        pop = smoothstep(0.0, 0.4, self.progress("setup", t))
        c = self.P(*GRID_AT)
        px, py = GRID_PX + 6, GRID_PX + 24
        for i in range(GRID_COLS * GRID_ROWS):
            r, col = divmod(i, GRID_COLS)
            q = self.shift(c, (col - (GRID_COLS - 1) / 2) * px, (r - (GRID_ROWS - 1) / 2) * py)
            wrong = bool(self.mask[i])
            draw_digit(self, sc, q, self.data.x_train[i], GRID_PX, op * pop,
                       ink=WRONG_INK if wrong else DIGIT_INK)
            self.label(sc, self.shift(q, 0, GRID_PX / 2 + 3), "%d" % self.data.y_train[i],
                       anchor="mt", opacity=op * pop,
                       ink=WRONG_INK if wrong else layout.CAPTION_INK)
        self.label(sc, self.shift(c, 0, -GRID_ROWS / 2 * py - 8),
                   "%s training images, %d%% of the labels false (%d of them)"
                   % (f"{N_TRAIN:,}", round(100 * NOISE), int(self.mask.sum())),
                   anchor="mb", opacity=op * pop)
        lines = [strong("784-w-10 ReLU networks, w from %d to %d" % (WIDTHS[0], WIDTHS[-1])),
                 note("%d epochs of Adam each: trained until the labels are fitted"
                      % EPOCHS),
                 note("parameters 795 w + 10; labels' constraints %s: threshold near w = %.0f"
                      % (f"{N_TRAIN * 10:,}", threshold_width()))]
        readout(self, sc, 0.09, 0.40, lines, op * smoothstep(0.4, 0.8, pop), step=0.05)

    def _glyph(self, sc, c, w, op, px=GLYPH_PX, cols=GLYPH_COLS):
        """A hidden layer of ``w`` units as a grid of cells."""
        rows = -(-w // cols)
        cells = np.zeros((rows, cols, 3), np.float32) + np.asarray(PAPER, np.float32)
        on = np.arange(rows * cols) < w
        cells.reshape(-1, 3)[on] = INK_WIDTH
        cw = min(cols, w)
        draw_image(self, sc, c, cells[:, :cw], cw * px, rows * px, op, grid=0.6, lift=0.2)
        return rows * px

    def _confusion(self, sc, c, k, op, px):
        """Run ``k``'s confusion matrix on the test set: rows true, columns predicted."""
        m = self.conf[k].astype(np.float64)
        frac = m / m.sum(1, keepdims=True)
        v = np.clip(frac / 0.5, 0, 1)[..., None] ** 0.6
        tex = np.asarray(PAPER)[None, None] * (1 - v) + np.asarray(DIGIT_INK)[None, None] * v
        draw_image(self, sc, c, tex.astype(np.float32), px, px, op, grid=0.3, lift=0.2)
        self.label(sc, self.shift(c, 0, px / 2 + 4), "predicted 0–9", anchor="mt", opacity=op,
                   ink=layout.NOTE_INK)
        self.label(sc, self.shift(c, -px / 2 - 4, 0), "true", anchor="rm", opacity=op,
                   ink=layout.NOTE_INK)

    def _current(self, sc, f, op):
        """The width training now: its hidden layer and confusion matrix."""
        k = f.cur if f.cur >= 0 else min(f.done, len(WIDTHS)) - 1
        if k < 0:
            return
        on = op * (smoothstep(0.0, 0.3, f.cu) if f.cur >= 0 else 1.0)
        w = WIDTHS[k]
        c = self.P(*GLYPH_AT)
        h = self._glyph(sc, c, w, on)
        self.label(sc, self.shift(c, 0, -h / 2 - 10), "w = %d hidden units" % w, anchor="mb",
                   opacity=on, ink=layout.STRONG_INK)
        self.label(sc, self.shift(c, 0, h / 2 + 8), "%s parameters" % f"{params(w):,}",
                   anchor="mt", opacity=on, ink=layout.NOTE_INK)
        self._confusion(sc, self.P(*CONF_AT), k, on, CONF_PX)
        lines = [note("after %s steps" % f"{EPOCHS * N_TRAIN // TRAIN['batch']:,}"),
                 strong("training error %.1f%%   test error %.1f%%"
                        % (100 * self.train_err[k], 100 * self.test_err[k]))]
        readout(self, sc, NOTE_AT[0], NOTE_AT[1], lines, on, step=0.05)

    def _trio(self, sc, f, op):
        """Three confusion matrices: the U's minimum, the peak, the widest."""
        for fx, k, name in zip(TRIO_X, (self.u_min, self.peak, self.wide),
                               ("the U's low point", "the peak", "far past")):
            c = self.P(fx, TRIO_Y)
            dim = 1.0
            if f.phase == 2 and k == self.wide:
                dim = 0.35
            self._confusion(sc, c, k, op * dim, TRIO_PX)
            self.label(sc, self.shift(c, 0, -TRIO_PX / 2 - 26), name, anchor="mb",
                       opacity=op * dim, ink=layout.STRONG_INK)
            self.label(sc, self.shift(c, 0, -TRIO_PX / 2 - 8), "w = %d: %.1f%% wrong"
                       % (WIDTHS[k], 100 * self.test_err[k]), anchor="mb", opacity=op * dim)

    def _why(self, sc, f, op):
        """Sensitivity to pixel noise against width."""
        place = lambda fx, fy: self.P(fx, fy, 0.3)                          # noqa: E731
        ticks = [(w, str(w)) for w in (2, 10, 100, 400)]
        top = self.why.y_range[1]
        self.why.axes(sc, self.cam, place, op, "hidden units w (log scale)",
                      "output change under pixel noise", ticks,
                      [(v, "%.1f" % v) for v in np.arange(0, top, 0.2)])
        xs = np.array(WIDTHS, float)
        upto = xs[0] * (xs[-1] / xs[0]) ** f.grow
        self.why.curve(sc, place, xs, self.sens, upto, INK_WIDTH, op)
        for k, name in ((self.peak, "peak"), (self.wide, "w = %d" % WIDTHS[self.wide])):
            if xs[k] <= upto:
                p = place(self.why.fx(xs[k]), self.why.fy(self.sens[k]))
                self.disk(sc, p, 3.8, INK_WIDTH, op, lift=0.6, outline=0.3)
                self.label(sc, self.shift(p, 0, -8), "%s: %.2f" % (name, self.sens[k]),
                           anchor="mb", opacity=op)
        lines = [note("mean L1 change of the softmax output when pixel noise"),
                 note("of std %.1f is added to %s test images" % (SENS_SIGMA, f"{SENS_N:,}"))]
        readout(self, sc, NOTE_AT[0], NOTE_AT[1], lines, op, step=0.045)

    def _chart(self, sc, f, op):
        """Training and test error against width."""
        place = lambda fx, fy: self.P(fx, fy, 0.3)                          # noqa: E731
        ticks = [(w, str(w)) for w in (2, 10, 100, 400)]
        self.chart.axes(sc, self.cam, place, op, "hidden units w (log scale)", "error",
                        ticks, [(v, "%d%%" % round(100 * v)) for v in (0, 0.2, 0.4, 0.6)])
        done = min(f.done, len(WIDTHS))
        if done == 0:
            return
        xs = np.array(WIDTHS[:done], float)
        for values, ink in ((self.train_err, TRAIN_INK), (self.test_err, TEST_INK)):
            self.chart.curve(sc, place, xs, values[:done], xs[-1], ink, op, dot=False)
            for x, y in zip(xs, values[:done]):
                dim = 1.0
                if f.phase == 2 and x > WIDTHS[self.peak]:
                    dim = 0.3
                if f.phase == 3 and x < WIDTHS[self.peak]:
                    dim = 0.3
                self.disk(sc, place(self.chart.fx(x), self.chart.fy(y)), 3.4, ink, op * dim,
                          lift=0.6, outline=0.3)
        if 0 <= f.cur < len(WIDTHS):
            x = WIDTHS[f.cur]
            u = ease_out(f.cu)
            for values, ink in ((self.train_err, TRAIN_INK), (self.test_err, TEST_INK)):
                p = place(self.chart.fx(x), self.chart.fy(values[f.cur]))
                self.disk(sc, p, 3.4 + 6 * (1 - u), ink, op * u, lift=0.7, glow=0.4 * (1 - u),
                          outline=0.3)
        if f.phase >= 2:
            x = WIDTHS[self.peak]
            self.chart.mark(sc, place, x, layout.CAPTION_INK, 0.8 * op)
            self.label(sc, place(self.chart.fx(x), CHART[1][0] - 0.02), "peak: w = %d" % x,
                       anchor="mb", opacity=op)
        self.chart.legend(sc, self.cam, [("test error", TEST_INK), ("training error", TRAIN_INK)],
                          op, (CHART[0][1] - 0.01, CHART[1][0] + 0.03), step=0.038, anchor="rm")

    # ------------------------------------------------------------------ words
    def _words(self, f):
        """The subtitle for frame ``f``."""
        te, tr = self.test_err, self.train_err
        u, p, w = self.u_min, self.peak, self.wide
        return [
            "%s MNIST images with %d%% of the labels false; 784-w-10 ReLU networks trained %d "
            "epochs each, until they fit every label" % (f"{N_TRAIN:,}", round(100 * NOISE),
                                                          EPOCHS),
            "width by width: the training error (teal) and the error on %s test digits (red)"
            % f"{N_TEST:,}",
            "the classic U: test error %.1f%% at w = %d, %.1f%% at w = %d, then up to %.1f%% at "
            "w = %d, the width that just fits all %s labels"
            % (100 * te[0], WIDTHS[0], 100 * te[u], WIDTHS[u], 100 * te[p], WIDTHS[p],
               f"{N_TRAIN:,}"),
            "past the peak the training error stays at %.0f%% and the test error falls again, to "
            "%.1f%% at w = %d: wider interpolators generalise better"
            % (100 * tr[w], 100 * te[w], WIDTHS[w]),
            "why: the output's change under pixel noise is %.2f at the peak and %.2f at w = %d: "
            "of all the networks that fit the labels, the wide one is the smoothest"
            % (self.sens[p], self.sens[w], WIDTHS[w]),
        ][f.phase]

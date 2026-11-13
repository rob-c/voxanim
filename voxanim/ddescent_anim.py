"""Double descent: a one-hidden-layer MLP of every width trained to
interpolation on a small, noisily labelled MNIST subset.

The setting: 1,000 training images, 20% of their labels replaced by a random
wrong class (:func:`voxanim.generalise.corrupt`), a 784-w-10 ReLU network
(He init) for each width ``w`` in :data:`WIDTHS`, every one trained by Adam
(lr 0.001, minibatches of 100) for 1,000 epochs (10,000 steps) -- long
enough that every width from 8 up fits all 1,000 labels, the false ones
included.  The test loss and error are on all 10,000 test digits.  A
784-w-10 network has ``795 w + 10`` parameters, and ``n x classes = 10,000``
labels' worth of constraints puts the interpolation threshold near
``w = 13``; the networks here first reach zero training error at ``w = 8``.

**What the sweep found.**  The double descent is in the test *loss*
(cross-entropy): 2.4 at w = 3, a peak of 3.6 at w = 6 -- the width that just
manages to fit the labels is confidently wrong on the test set -- then down
to 1.5 at w = 100 and 200.  The test *error* has no U in this
setting: it only falls, 65% at w = 2, 45% at the peak, 24% at w = 100
(100, 300 and 1,000 epochs, Adam and SGD, 20% and 30% noise, 1,000 and
2,000 images were swept; the error bump at the threshold never exceeded
1.5 points, so it is not claimed).  The animation says exactly this.

1. **setup** -- the data: training digits with their labels, the false ones
   in red; the model and where its threshold lies,
2. **sweep** -- the widths one by one: each trained network's hidden layer to
   size, its confusion matrix on the test set, and its losses and errors
   dropping onto the two charts (log width axis),
3. **the U** -- the test loss falls with width, then rises to its peak at the
   width that just fits every label,
4. **second descent** -- past the peak the training error stays at zero and
   the test loss falls again, below the U's minimum; the test error with it,
5. **why** -- how smooth the function is: the output's response to pixel
   noise (mean L1 change of the softmax under noise of std 0.1 on 2,000
   test images) per width peaks at the threshold and is smallest for the
   widest interpolators.

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
from .generalise import confusion, corrupt, error_rate, loss_on, predictions, sensitivity
from .losschart import LossChart
from .sweep import Sweep

N_TRAIN, N_TEST, SEED = 1000, 10000, 0
NOISE, NOISE_SEED = 0.2, 0
WIDTHS = (2, 3, 4, 6, 8, 12, 16, 25, 40, 64, 100, 200)
EPOCHS = 1000
TRAIN = dict(batch=100, lr=0.001, opt="adam")
INTERPOLATES = 0.005                              # training error counted as zero below this
SENS_SIGMA, SENS_N, SENS_SEED = 0.1, 2000, 1      # the smoothness measurement
INK_WIDTH = C.rgb("#3d9e45")
GRID_COLS, GRID_ROWS, GRID_PX = 10, 4, 28.0       # training digits of the setup phase
GRID_AT = (0.74, 0.50)
LOSS_CHART = ((0.09, 0.45), (0.30, 0.56))         # test and training loss against width
ERR_CHART = ((0.09, 0.45), (0.68, 0.90))          # test and training error against width
LOSS_RANGE, ERR_RANGE = (0.0, 4.0), (0.0, 0.7)
GLYPH_AT, GLYPH_PX, GLYPH_COLS = (0.63, 0.42), 7.0, 20   # the hidden layer, sweep phase
CONF_AT, CONF_PX = (0.84, 0.44), 130.0            # the confusion matrix, sweep phase
TRIO_Y, TRIO_PX, TRIO_X = 0.54, 100.0, (0.585, 0.755, 0.925)   # three matrices, later phases
WHY_CHART = ((0.60, 0.94), (0.34, 0.66))          # sensitivity against width
NOTE_AT = (0.58, 0.70)
WIDTH_TICKS = (2, 10, 100, 200)


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


def landmarks(train_err, test_loss):
    """``(u_min, peak, wide)``: indices of the U's lowest point, the peak and
    the widest network.  The peak is the largest test loss among the widths
    within a factor of two of the first that interpolates; the U's minimum
    the smallest test loss at a narrower width than the peak."""
    widths = np.asarray(WIDTHS, float)
    fits = np.nonzero(train_err <= INTERPOLATES)[0]
    first = widths[fits[0]] if len(fits) else widths[-1]
    near = np.nonzero((widths >= first / 2) & (widths <= first * 2))[0]
    peak = int(near[np.argmax(test_loss[near])])
    u_min = int(np.argmin(test_loss[:peak])) if peak > 0 else 0
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
    """Test loss and error against width through the interpolation threshold
    (see the module docstring)."""

    PHASES = ("setup", "sweep", "the U", "second descent", "why")

    def __init__(self, data, mask, runs):
        super().__init__("Double descent", "")
        self.data, self.mask, self.runs = data, mask, runs
        self.hold = 2.5
        x, y = data.x_train, data.y_train
        xt, yt = data.x_test, data.y_test
        self.train_err = np.array([error_rate(r.spec, r.P, x, y) for r in runs])
        self.test_err = np.array([error_rate(r.spec, r.P, xt, yt) for r in runs])
        self.train_loss = np.array([loss_on(r.spec, r.P, x, y, 10) for r in runs])
        self.test_loss = np.array([loss_on(r.spec, r.P, xt, yt, 10) for r in runs])
        self.sens = np.array([sensitivity(r.spec, r.P, xt[:SENS_N],
                                          np.random.default_rng(SENS_SEED), SENS_SIGMA)
                              for r in runs])
        self.conf = [confusion(predictions(r.spec, r.P, xt), yt, 10) for r in runs]
        self.u_min, self.peak, self.wide = landmarks(self.train_err, self.test_loss)
        self.loss_chart = LossChart(LOSS_CHART, (WIDTHS[0], WIDTHS[-1]), LOSS_RANGE, xlog=True)
        self.err_chart = LossChart(ERR_CHART, (WIDTHS[0], WIDTHS[-1]), ERR_RANGE, xlog=True)
        self.why = LossChart(WHY_CHART, (WIDTHS[0], WIDTHS[-1]),
                             (0.0, float(self.sens.max()) * 1.5), xlog=True)
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
        cur = cur if ph == 1 and cur < len(WIDTHS) else -1
        return Frame(back, ph, done, cur, cu, grow)

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
            self._charts(sc, f, op)
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
                   "%s training images; %d%% of the labels false, in red"
                   % (f"{N_TRAIN:,}", round(100 * NOISE)),
                   anchor="mb", opacity=op * pop)
        lines = [strong("784-w-10 ReLU networks, w from %d to %d" % (WIDTHS[0], WIDTHS[-1])),
                 note("%s Adam steps each (%d epochs):"
                      % (f"{EPOCHS * N_TRAIN // TRAIN['batch']:,}", EPOCHS)),
                 note("long enough to fit every label, false ones too"),
                 note("795 w + 10 parameters against %s labels:" % f"{N_TRAIN * 10:,}"),
                 note("the interpolation threshold is near w = %.0f" % threshold_width())]
        readout(self, sc, 0.09, 0.40, lines, op * smoothstep(0.4, 0.8, pop), step=0.05)

    def _glyph(self, sc, c, w, op, px=GLYPH_PX, cols=GLYPH_COLS):
        """A hidden layer of ``w`` units as a grid of cells; returns its height in px."""
        rows = -(-w // cols)
        cells = np.zeros((rows, cols, 3), np.float32) + np.asarray(PAPER, np.float32)
        on = np.arange(rows * cols) < w
        cells.reshape(-1, 3)[on] = INK_WIDTH
        cw = min(cols, w)
        draw_image(self, sc, c, cells[:, :cw], cw * px, rows * px, op, grid=0.6, lift=0.2)
        return rows * px

    def _confusion(self, sc, c, k, op, px, axes=(True, True)):
        """Run ``k``'s confusion matrix on the test set: rows true, columns
        predicted; ``axes`` says which of the two axis labels to write."""
        m = self.conf[k].astype(np.float64)
        frac = m / m.sum(1, keepdims=True)
        v = np.clip(frac / 0.5, 0, 1)[..., None] ** 0.6
        tex = np.asarray(PAPER)[None, None] * (1 - v) + np.asarray(DIGIT_INK)[None, None] * v
        draw_image(self, sc, c, tex.astype(np.float32), px, px, op, grid=0.3, lift=0.2)
        if axes[1]:
            self.label(sc, self.shift(c, 0, px / 2 + 4), "predicted 0–9", anchor="mt",
                       opacity=op, ink=layout.NOTE_INK)
        if axes[0]:
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
                        % (100 * self.train_err[k], 100 * self.test_err[k])),
                 strong("training loss %.4f   test loss %.2f"
                        % (self.train_loss[k], self.test_loss[k]))]
        readout(self, sc, NOTE_AT[0], NOTE_AT[1], lines, on, step=0.05)

    def _trio(self, sc, f, op):
        """Three confusion matrices: the U's minimum, the peak, the widest."""
        for n, (fx, k, name) in enumerate(zip(TRIO_X, (self.u_min, self.peak, self.wide),
                                              ("the U's low point", "the peak", "far past"))):
            c = self.P(fx, TRIO_Y)
            dim = 0.35 if f.phase == 2 and k == self.wide else 1.0
            self._confusion(sc, c, k, op * dim, TRIO_PX, axes=(n == 0, n == 1))
            self.label(sc, self.shift(c, 0, -TRIO_PX / 2 - 72), name, anchor="mb",
                       opacity=op * dim, ink=layout.STRONG_INK)
            for row, text in enumerate(("w = %d" % WIDTHS[k], "test loss %.2f" % self.test_loss[k],
                                        "%.0f%% wrong" % (100 * self.test_err[k]))):
                self.label(sc, self.shift(c, 0, -TRIO_PX / 2 - 50 + 21 * row), text, anchor="mb",
                           opacity=op * dim)

    def _why(self, sc, f, op):
        """Sensitivity to pixel noise against width."""
        place = lambda fx, fy: self.P(fx, fy, 0.3)                          # noqa: E731
        ticks = [(w, str(w)) for w in WIDTH_TICKS]
        top = self.why.y_range[1]
        self.why.axes(sc, self.cam, place, op, "hidden units w (log scale)",
                      "output change under pixel noise", ticks,
                      [(v, "%.1f" % v) for v in np.arange(0, top, 0.4)])
        xs = np.array(WIDTHS, float)
        upto = xs[0] * (xs[-1] / xs[0]) ** f.grow
        self.why.curve(sc, place, xs, self.sens, upto, INK_WIDTH, op)
        top_k = int(np.argmax(self.sens))
        for k in (top_k, self.wide):
            name = "w = %d" % WIDTHS[k]
            if xs[k] <= upto:
                p = place(self.why.fx(xs[k]), self.why.fy(self.sens[k]))
                self.disk(sc, p, 3.8, INK_WIDTH, op, lift=0.6, outline=0.3)
                # the peak's label goes above it (the chart leaves headroom); the
                # widest point sits near the right edge: below-left of it
                self.label(sc, self.shift(p, 6 if k == self.wide else 0, 8 if k == self.wide
                                          else -8), "%s: %.2f" % (name, self.sens[k]),
                           anchor="rt" if k == self.wide else "mb", opacity=op)
        lines = [note("mean L1 change of the softmax output"),
                 note("under pixel noise of std %.1f, %s test images"
                      % (SENS_SIGMA, f"{SENS_N:,}"))]
        readout(self, sc, NOTE_AT[0], NOTE_AT[1] + 0.06, lines, op, step=0.045)

    def _points(self, sc, place, chart, xs, values, ink, op, f):
        """A curve with a dot per width, dimmed outside the phase's focus."""
        chart.curve(sc, place, xs, values, xs[-1], ink, op, dot=False)
        for x, y in zip(xs, values):
            dim = 1.0
            if f.phase == 2 and x > WIDTHS[self.peak]:
                dim = 0.3
            if f.phase == 3 and x < WIDTHS[self.peak]:
                dim = 0.3
            self.disk(sc, place(chart.fx(x), chart.fy(y)), 3.2, ink, op * dim, lift=0.6,
                      outline=0.3)

    def _charts(self, sc, f, op):
        """Losses (top) and errors (bottom) against width."""
        place = lambda fx, fy: self.P(fx, fy, 0.3)                          # noqa: E731
        ticks = [(w, str(w)) for w in WIDTH_TICKS]
        self.loss_chart.axes(sc, self.cam, place, op, "", "loss", ticks,
                             [(v, "%d" % v) for v in (1, 2, 3, 4)])
        self.err_chart.axes(sc, self.cam, place, op, "hidden units w (log scale)", "error",
                            ticks, [(v, "%d%%" % round(100 * v)) for v in (0.2, 0.4, 0.6)])
        done = min(f.done, len(WIDTHS))
        if done == 0:
            return
        xs = np.array(WIDTHS[:done], float)
        self._points(sc, place, self.loss_chart, xs, self.train_loss[:done], TRAIN_INK, op, f)
        self._points(sc, place, self.loss_chart, xs, self.test_loss[:done], TEST_INK, op, f)
        self._points(sc, place, self.err_chart, xs, self.train_err[:done], TRAIN_INK, op, f)
        self._points(sc, place, self.err_chart, xs, self.test_err[:done], TEST_INK, op, f)
        if 0 <= f.cur < len(WIDTHS):
            x = WIDTHS[f.cur]
            u = ease_out(f.cu)
            for chart, values, ink in ((self.loss_chart, self.train_loss, TRAIN_INK),
                                       (self.loss_chart, self.test_loss, TEST_INK),
                                       (self.err_chart, self.train_err, TRAIN_INK),
                                       (self.err_chart, self.test_err, TEST_INK)):
                p = place(chart.fx(x), chart.fy(values[f.cur]))
                self.disk(sc, p, 3.2 + 6 * (1 - u), ink, op * u, lift=0.7, glow=0.4 * (1 - u),
                          outline=0.3)
        if f.phase >= 2:
            x = WIDTHS[self.peak]
            for chart in (self.loss_chart, self.err_chart):
                chart.mark(sc, place, x, layout.CAPTION_INK, 0.8 * op)
            self.label(sc, place(self.loss_chart.fx(x) + 0.006, LOSS_CHART[1][0] - 0.02),
                       "peak: w = %d" % x, anchor="lb", opacity=op)
        self.loss_chart.legend(sc, self.cam, [("test", TEST_INK), ("training", TRAIN_INK)],
                               op, (LOSS_CHART[0][1] - 0.01, LOSS_CHART[1][0] + 0.03),
                               step=0.038, anchor="rm")

    # ------------------------------------------------------------------ words
    def _words(self, f):
        """The subtitle for frame ``f``."""
        te, tr, tl = self.test_err, self.train_err, self.test_loss
        u, p, w = self.u_min, self.peak, self.wide
        top = int(np.argmax(self.sens))
        return [
            "%s MNIST images with %d%% of the labels false; 784-w-10 ReLU networks trained %d "
            "epochs each, until they fit every label" % (f"{N_TRAIN:,}", round(100 * NOISE),
                                                          EPOCHS),
            "width by width: training (teal) and test (red) loss above, error below, the test "
            "set %s digits" % f"{N_TEST:,}",
            "the classic U, in the test loss: %.2f at w = %d, up to %.2f at w = %d, the width "
            "that just about fits all %s labels: confidently wrong"
            % (tl[u], WIDTHS[u], tl[p], WIDTHS[p], f"{N_TRAIN:,}"),
            "past the peak the training error stays at %.0f%% and the test loss falls to %.2f at "
            "w = %d; the test error only ever falls: %.0f%% → %.0f%% → %.0f%%"
            % (100 * tr[w], tl[w], WIDTHS[w], 100 * te[0], 100 * te[p], 100 * te[w]),
            "why: the output's change under pixel noise peaks at the threshold (%.2f at w = %d) "
            "and is %.2f at w = %d: the widest fit is the smoothest"
            % (self.sens[top], WIDTHS[top], self.sens[w], WIDTHS[w]),
        ][f.phase]

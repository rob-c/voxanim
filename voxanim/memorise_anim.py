"""Networks memorise anything: the MNIST MLP trained on 5,000 images whose
labels are 0%, 25% and 50% wrong.

The data: the first 5,000 images of a seeded shuffle of the MNIST training
set; a validation set of 2,000 test digits and a test set of the other
8,000.  :func:`voxanim.generalise.corrupt` replaces a seeded fraction of the
training labels by a random *other* class (so 25% means exactly 1,250 false
labels).  The network, a 784-256-10 ReLU MLP (He init), is trained by Adam
(lr 0.001, minibatches of 100) for 120 epochs on each label set, and
:func:`voxanim.deepnet.train` records the loss of every one of the 5,000
training examples at every recorded step (``example_loss``).

1. **labels** -- the training digits with their labels; the false ones turn
   red as the fraction goes 0% -> 25% -> 50%,
2. **training** -- the training loss of the three runs on a log axis: all
   three fall towards zero (the network can fit the labels whatever they
   are),
3. **test** -- what that buys: test accuracy 95%, 82% and 60%, and the test
   loss curves, which turn back up,
4. **per example** -- the recorded per-example losses of the 50% run sorted
   by status, clean examples left, shuffled right, one row per recorded
   step: the clean half is learned first (its mean loss 0.40 at step 1,000
   against 1.23), the shuffled half last, by memorising,
5. **early stopping** -- the validation loss of the 50% run bottoms out at
   step 498: stopping there gives 83% test accuracy instead of 60%.

(The numbers are this machine's; CI retrains.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import color as C
from . import layout
from .deepnet import Data, Spec, subset, train
from .diagram import Diagram, draw_image, shrink_columns
from .easing import ease_in_out, remap, smoothstep
from .gen_common import DIGIT_INK, PAPER, WRONG_INK, draw_digit, note, pct, readout, strong
from .generalise import corrupt, error_rate
from .losschart import LossChart
from .netanim import ACT_HI, OFF

N_TRAIN, N_VAL, N_TEST, SEED = 5000, 2000, 8000, 0
NOISES = (0.0, 0.25, 0.5)
NOISE_SEED = 0                                   # the generator that picks and relabels
SIZES = (784, 256, 10)
TRAIN = dict(steps=6000, batch=100, lr=0.001, opt="adam")
INKS = (C.rgb("#3d9e45"), C.rgb("#e0952e"), C.rgb("#d84e4e"))     # 0%, 25%, 50%
GRID_COLS, GRID_ROWS, GRID_PX = 10, 5, 30.0      # training digits shown in the labels phase
GRID_AT = (0.26, 0.52)
HEAT_AT, HEAT_W, HEAT_H = (0.26, 0.50), 330.0, 200.0     # the per-example loss image
HEAT_COLS = 330                                  # examples averaged down to this many columns
HEAT_MAX = 4.0                                   # loss drawn at full brightness
CHART = ((0.57, 0.94), (0.32, 0.66))
LOSS_RANGE = (1e-3, 3.0)
NOTE_AT = (0.57, 0.77)


def datasets(cache_dir=None):
    """``[(data, mask), ...]`` for the three noise levels: the same images,
    the labels of a seeded fraction replaced; ``mask`` marks them."""
    raw = subset(N_TRAIN, N_VAL + N_TEST, None, SEED, cache_dir)
    out = []
    for noise in NOISES:
        y, mask = corrupt(raw.y_train, noise, 10, np.random.default_rng(NOISE_SEED))
        out.append((Data(raw.x_train, y, raw.x_test[N_VAL:], raw.y_test[N_VAL:], 10,
                         raw.x_test[:N_VAL], raw.y_test[:N_VAL]), mask))
    return out


def runs(cache_dir=None):
    """``(datasets, runs)``: the three label sets and their recorded runs."""
    sets = datasets(cache_dir)
    spec = Spec(SIZES, "relu", "he")
    kw = dict(TRAIN, example_loss=True)
    if cache_dir:
        kw["cache_dir"] = cache_dir
    return sets, [train(spec, data, **kw) for data, _mask in sets]


def heat_image(example_loss, mask):
    """The per-example losses ``(S, n)`` as an ``(S, HEAT_COLS, 3)`` texture:
    columns are the examples, clean ones first then the shuffled, averaged
    down to the width; brightness is the loss (saturating at HEAT_MAX)."""
    order = np.concatenate([np.nonzero(~mask)[0], np.nonzero(mask)[0]])
    v = np.clip(example_loss[:, order] / HEAT_MAX, 0.0, 1.0) ** 0.5
    tex = OFF[None, None] + (ACT_HI - OFF)[None, None] * v[..., None]
    return shrink_columns(tex.astype(np.float32), HEAT_COLS)


@dataclass
class Frame:
    """What one instant shows."""

    back: float
    phase: int
    level: int       # labels phase: which label set is on show
    x: float         # training phases: fractional recorded-step index
    stop: float      # early stopping: the mark's opacity


class MemorisingNoise(Diagram):
    """Training on partly false labels (see the module docstring)."""

    PHASES = ("labels", "training", "test", "per example", "early stopping")

    def __init__(self, sets, runs):
        super().__init__("Networks memorise anything", "")
        self.sets, self.runs = sets, runs
        self.hold = 2.5
        self.S = len(runs[0].steps)
        noisy = runs[-1]
        self.mask = sets[-1][1]
        self.heat = heat_image(noisy.example_loss, self.mask)
        self.clean_curve = noisy.example_loss[:, ~self.mask].mean(1)
        self.shuffled_curve = noisy.example_loss[:, self.mask].mean(1)
        self.test_err = [error_rate(r.spec, r.P, d.x_test, d.y_test)
                         for r, (d, _m) in zip(runs, sets)]
        self.best = noisy.best
        self.chart = LossChart(CHART, (1, TRAIN["steps"]), LOSS_RANGE, xlog=True, ylog=True)
        Tm, t = {}, 0.0
        Tm["labels"] = (t, t + 7.0)
        t += 7.0 + 0.3
        Tm["training"] = (t, t + 10.0)
        t += 10.0 + 1.5
        Tm["test"] = (t, t + 6.0)
        t += 6.0 + 0.3
        Tm["example"] = (t, t + 9.0)
        t += 9.0 + 1.0
        Tm["stop"] = (t, t + 5.0)
        t += 5.0 + 1.5
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["training"][0], Tm["test"][0], Tm["example"][0], Tm["stop"][0]]

    # ------------------------------------------------------------------ state
    def _frame(self, t, reset):
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else int(np.searchsorted(self.marks, t, side="right") - 1)
        level = min(int(self.progress("labels", t) * 3), 2) if ph == 0 else 2
        if ph == 0:
            x = 0.0
        elif ph == 1:
            x = self.progress("training", t) * (self.S - 1)
        elif ph == 3:
            x = self.progress("example", t) * (self.S - 1)
        else:
            x = float(self.S - 1)
        stop = smoothstep(0.2, 0.5, self.progress("stop", t)) if ph == 4 else 0.0
        return Frame(back, ph, level, x, stop)

    def at(self, run, x):
        """``(step, train loss, test loss)`` of ``run`` at fractional index ``x``."""
        i = int(round(x))
        return int(run.steps[i]), float(run.example_loss[i].mean()), float(run.test_loss[i])

    # ------------------------------------------------------------------ drawing
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t``."""
        f = self._frame(t, reset)
        self.subtitle = self._words(f)
        op = f.back * smoothstep(0.0, 0.5, t)
        if f.phase <= 2:
            self._grid(sc, t, f, op)
        else:
            self._heat(sc, f, op)
        if f.phase >= 1:
            self._chart(sc, f, op)
            self._notes(sc, f, op)
        self.step_indicator(sc, f.phase, self.PHASES)

    def _grid(self, sc, t, f, op):
        """Fifty training digits and their labels at the level on show."""
        data, mask = self.sets[f.level]
        c = self.P(*GRID_AT)
        pitch_x, pitch_y = GRID_PX + 6, GRID_PX + 24
        pop = smoothstep(0.0, 0.3, self.progress("labels", t)) if f.phase == 0 else 1.0
        for i in range(GRID_COLS * GRID_ROWS):
            r, col = divmod(i, GRID_COLS)
            q = self.shift(c, (col - (GRID_COLS - 1) / 2) * pitch_x,
                           (r - (GRID_ROWS - 1) / 2) * pitch_y)
            wrong = bool(mask[i])
            draw_digit(self, sc, q, data.x_train[i], GRID_PX, op * pop,
                       ink=WRONG_INK if wrong else DIGIT_INK, paper=PAPER)
            self.label(sc, self.shift(q, 0, GRID_PX / 2 + 3), "%d" % data.y_train[i], anchor="mt",
                       opacity=op * pop, ink=WRONG_INK if wrong else layout.CAPTION_INK)
        n_wrong = int(mask.sum())
        self.label(sc, self.shift(c, 0, -GRID_ROWS / 2 * pitch_y - 8),
                   "%d%% of the labels false: %s of %s" % (round(100 * NOISES[f.level]),
                                                            f"{n_wrong:,}", f"{N_TRAIN:,}"),
                   anchor="mb", opacity=op * pop, ink=INKS[f.level])

    def _heat(self, sc, f, op):
        """The per-example losses of the 50% run, rows revealed as training plays."""
        c = self.P(*HEAT_AT)
        i = int(round(f.x))
        tex = self.heat.copy()
        tex[i + 1:] = PAPER                                   # rows still to come
        draw_image(self, sc, c, tex, HEAT_W, HEAT_H, op, grid=0.0, lift=0.2)
        self.label(sc, self.shift(c, -HEAT_W / 4, HEAT_H / 2 + 6), "%s clean" %
                   f"{int((~self.mask).sum()):,}", anchor="mt", opacity=op, ink=INKS[0])
        self.label(sc, self.shift(c, HEAT_W / 4, HEAT_H / 2 + 6), "%s shuffled" %
                   f"{int(self.mask.sum()):,}", anchor="mt", opacity=op, ink=INKS[2])
        self.label(sc, self.shift(c, 0, -HEAT_H / 2 - 8), "each example's loss, bright = high; "
                   "one row per recorded step, step 0 on top", anchor="mb", opacity=op)
        # the row on show
        y = -HEAT_H / 2 + (i + 0.5) / self.S * HEAT_H
        self.polyline(sc, [self.shift(c, -HEAT_W / 2 - 6, y), self.shift(c, -HEAT_W / 2 - 1, y)],
                      width=1.4, color=layout.CAPTION_INK, opacity=op)
        self.label(sc, self.shift(c, -HEAT_W / 2 - 8, y), "step %s" %
                   f"{int(self.runs[-1].steps[i]):,}", anchor="rm", opacity=op,
                   ink=layout.NOTE_INK)

    def _chart(self, sc, f, op):
        """Training losses (phase 2 adds test losses; 4 the clean / shuffled
        means; 5 the validation loss and the stopping point)."""
        place = lambda fx, fy: self.P(fx, fy, 0.3)                          # noqa: E731
        ticks = [(s, f"{s:,}") for s in (1, 10, 100, 1000)]
        self.chart.axes(sc, self.cam, place, op, "training steps", "loss (log)", ticks,
                        [(v, "%g" % v) for v in (0.001, 0.01, 0.1, 1)])
        i = int(np.floor(f.x))
        a = f.x - i
        steps = self.runs[0].steps
        upto = steps[i] if i + 1 >= self.S else steps[i] * (1 - a) + steps[i + 1] * a
        upto = max(float(upto), 1.0)
        noisy = self.runs[-1]
        legend = []
        if f.phase in (1, 2):
            for run, ink, noise in zip(self.runs, INKS, NOISES):
                self.chart.curve(sc, place, run.steps, run.example_loss.mean(1), upto, ink, op)
                legend.append(("%d%% false: training loss" % round(100 * noise), ink))
            if f.phase == 2:
                for run, ink in zip(self.runs, INKS):
                    self.chart.curve(sc, place, run.steps, run.test_loss, upto, ink, 0.5 * op,
                                     width=1.0, dot=False)
                legend.append(("test losses, faint", layout.NOTE_INK))
        elif f.phase == 3:
            self.chart.curve(sc, place, noisy.steps, self.clean_curve, upto, INKS[0], op)
            self.chart.curve(sc, place, noisy.steps, self.shuffled_curve, upto, INKS[2], op)
            legend = [("50% run: loss of the clean examples", INKS[0]),
                      ("loss of the shuffled examples", INKS[2])]
        else:
            self.chart.curve(sc, place, noisy.steps, noisy.example_loss.mean(1), upto, INKS[2],
                             0.5 * op, width=1.0, dot=False)
            self.chart.curve(sc, place, noisy.steps, noisy.test_loss, upto, INKS[2], op,
                             dot=False)
            self.chart.curve(sc, place, noisy.steps, noisy.val_loss, upto, layout.CAPTION_INK,
                             op, dot=False)
            legend = [("50% run: validation loss", layout.CAPTION_INK),
                      ("test loss", INKS[2]), ("training loss, faint", layout.NOTE_INK)]
            if f.stop > 0:
                stop_step = noisy.steps[self.best]
                self.chart.mark(sc, place, stop_step, layout.CAPTION_INK, op * f.stop)
                self.label(sc, place(self.chart.fx(stop_step), CHART[1][0] - 0.02),
                           "stop: step %d" % stop_step, anchor="mb", opacity=op * f.stop)
        self.chart.legend(sc, self.cam, legend, op, (CHART[0][0] + 0.02, CHART[1][1] - 0.03
                                                      - 0.038 * (len(legend) - 1)), step=0.038)

    def _notes(self, sc, f, op):
        """The step and the numbers at it."""
        if f.phase == 1:
            step = self.at(self.runs[0], f.x)[0]
            lines = [note("step %s" % f"{step:,}")] + [
                strong("%d%% false: training loss %.3f" % (round(100 * n), self.at(r, f.x)[1]))
                for r, n in zip(self.runs, NOISES)]
        elif f.phase == 2:
            lines = [note("after %s steps" % f"{TRAIN['steps']:,}")] + [
                strong("%d%% false: test accuracy %s" % (round(100 * n), pct(1 - e)))
                for e, n in zip(self.test_err, NOISES)]
        elif f.phase == 3:
            i = int(round(f.x))
            lines = [note("step %s" % f"{int(self.runs[-1].steps[i]):,}"),
                     strong("clean examples: mean loss %.2f" % self.clean_curve[i]),
                     strong("shuffled examples: mean loss %.2f" % self.shuffled_curve[i])]
        else:
            noisy = self.runs[-1]
            b = self.best
            lines = [note("50%% false, validation loss lowest at step %d" % noisy.steps[b]),
                     strong("test accuracy there %s" % pct(noisy.test_acc[b])),
                     strong("after %s steps %s" % (f"{TRAIN['steps']:,}",
                                                   pct(1 - self.test_err[2])))]
        readout(self, sc, NOTE_AT[0], NOTE_AT[1], lines, op, step=0.045)

    # ------------------------------------------------------------------ words
    def _words(self, f):
        """The subtitle for frame ``f``."""
        end = self.S - 1
        final = [r.example_loss[end].mean() for r in self.runs]
        noisy = self.runs[-1]
        b = self.best
        k = int(np.searchsorted(noisy.steps, 1000))
        return [
            "%s MNIST training images, a %d-%d-%d ReLU network; a seeded fraction of the labels "
            "is replaced by a wrong class" % (f"{N_TRAIN:,}", *SIZES),
            "Adam, %d epochs: the training loss falls to %.3f, %.3f and %.3f: the network fits "
            "the labels whether they are true or not"
            % (TRAIN["steps"] * TRAIN["batch"] // N_TRAIN, *final),
            "but it only generalises what was true: test accuracy %s, %s and %s on %s held-out "
            "digits" % (*[pct(1 - e) for e in self.test_err], f"{N_TEST:,}"),
            "the 50%% run, every example's loss: the clean half is learned first (%.2f at step "
            "%s against %.2f), the shuffled half is memorised last"
            % (self.clean_curve[k], f"{int(noisy.steps[k]):,}", self.shuffled_curve[k]),
            "early stopping: the validation loss is lowest at step %d, before the memorising; "
            "test accuracy %s there, %s at the end"
            % (noisy.steps[b], pct(noisy.test_acc[b]), pct(1 - self.test_err[2])),
        ][f.phase]

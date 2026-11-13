"""Seven activations, one network: the same MNIST classifier trained with
sigmoid, tanh, ReLU, Leaky ReLU, GELU, SiLU and ELU.

    784 -> 128 -> 64 -> 10, Adam at learning rate 0.001, batches of 64,
    1,500 steps on 10,000 MNIST images; Xavier weights for sigmoid and tanh,
    He weights for the rest; the same data order for all seven

1. **functions** -- the seven functions drawn one after another, each with
   its slope f′ (magenta): the squashing pair, the rectifier, its leaky
   variant, the two smooth rectifiers and ELU,
2. **training** -- the training loss and the test accuracy of all seven,
   racing on a log time axis (steps 1 to 1,500),
3. **results** -- the final numbers in a table: training loss, test accuracy
   and, honestly, who ties with whom,
4. **cost** -- the measured time of one training step (forward, backward,
   Adam) under each activation, as bars: the exponentials and tanh of the
   smooth functions cost more than ReLU's comparison, but in a network this
   small the matrix products dominate and the differences are a few percent
   (the function alone is measured too and quoted in the subtitle).

Every number is from the recorded runs of :func:`voxanim.actfun.race_runs`
and the timing of :func:`voxanim.actfun.step_times`.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .actfun import ACTS, RACE, RACE_SIZES, race_runs, step_times
from .actfun_plots import function_plot
from .deepnet import act
from .diagram import Diagram
from .easing import remap, smoothstep
from .losschart import LossChart
from .text import num

STEPS = RACE["steps"]
TIE = 0.005                                 # accuracies this close count as a tie
PLOT_X0, PLOT_W, PLOT_GAP = 0.005, 0.125, 0.0155
PLOT_Y = (0.30, 0.64)
LOSS_BOX = ((0.07, 0.46), (0.28, 0.80))     # the two race charts
ACC_BOX = ((0.58, 0.97), (0.28, 0.80))
LOSS_RANGE, ACC_RANGE = (0.0, 2.5), (0.0, 1.0)
LEGEND_Y = 0.93                             # the row of names under the charts
TABLE_Y0, ROW = 0.27, 0.075                 # the table: first row and row step
COLS = (0.02, 0.21, 0.46, 0.66, 0.72)       # name, init, loss, accuracy, time (layout x)
BAR_X, BAR_W = 0.735, 0.20                  # the time bars: left edge and the longest bar
BAR_H = 14.0                                # px
FUNC_DT = 1.3                               # seconds per function plot
RACE_T = 13.0                               # seconds the race takes
RESULTS_T, COST_T = 7.0, 7.0
HEAD_INK = layout.NOTE_INK


def ties(acc):
    """Indices of the activations within :data:`TIE` of the best accuracy."""
    acc = np.asarray(acc)
    return [int(i) for i in np.nonzero(acc >= acc.max() - TIE)[0]]


def names(idx):
    """``"ReLU, GELU and SiLU"`` for the activations ``idx``."""
    labels = [ACTS[i].label for i in idx]
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


class ActivationRace(Diagram):
    """The seven functions, the race, the table and the cost (see the module)."""

    PHASES = ("functions", "training", "results", "cost")

    def __init__(self, runs=None, times=None):
        super().__init__("Seven activations, one network", "")
        self.runs = race_runs() if runs is None else runs
        self.times = step_times() if times is None else times
        self.hold = 2.5
        self.steps = self.runs[0].steps
        self.loss = [r.train_loss for r in self.runs]
        self.acc = [r.test_acc for r in self.runs]
        self.final_acc = np.array([float(a[-1]) for a in self.acc])
        self.final_loss = np.array([float(v[-1]) for v in self.loss])
        self.ms = 1e3 * np.asarray(self.times["step"], np.float64)
        self.us = 1e6 * np.asarray(self.times["fn"], np.float64)
        self.loss_chart = LossChart(LOSS_BOX, (1, STEPS), LOSS_RANGE, xlog=True)
        self.acc_chart = LossChart(ACC_BOX, (1, STEPS), ACC_RANGE, xlog=True)
        Tm, t = {}, 0.3
        for k, a in enumerate(ACTS):
            Tm["plot:" + a.key] = (t + k * FUNC_DT, t + (k + 1) * FUNC_DT)
        t += len(ACTS) * FUNC_DT + 1.5
        Tm["race"] = (t, t + RACE_T)
        t += RACE_T + 1.5
        Tm["results"] = (t, t + RESULTS_T)
        t += RESULTS_T
        Tm["cost"] = (t, t + COST_T)
        t += COST_T
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["race"][0], Tm["results"][0], Tm["cost"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        ph = 0 if reset > 0 else self.step_of(t)
        race = self.progress("race", t)
        swap = smoothstep(0.0, 0.5, remap(t, self.Tm["race"][0] - 1.5, self.Tm["race"][0] - 0.3))
        results = smoothstep(0.0, 0.25, self.progress("results", t))
        funcs = (1.0 - swap) * back
        charts = swap * (1.0 - results) * back
        table = results * back
        self._functions(sc, t, funcs)
        self._race(sc, race, charts)
        self._table(sc, t, table)
        self.step_indicator(sc, ph, self.PHASES)
        self.subtitle = self._words(t, race, reset)

    # -- the functions ---------------------------------------------------------------
    def _functions(self, sc, t, op):
        if op <= 0.01:
            return
        for k, a in enumerate(ACTS):
            u = smoothstep(0.0, 1.0, self.progress("plot:" + a.key, t))
            if u <= 0:
                continue
            x0 = PLOT_X0 + k * (PLOT_W + PLOT_GAP)
            function_plot(self, sc, ((x0, x0 + PLOT_W), PLOT_Y), a, u, op,
                          sub="%s init" % a.init_label)

    # -- the race ----------------------------------------------------------------------
    def upto(self, race):
        """The training step the race has reached: constant speed on the log axis."""
        return 10.0 ** (remap(race, 0.0, 0.92) * np.log10(STEPS))

    def _race(self, sc, race, op):
        op *= self._cap
        if op <= 0.01:
            return
        upto = self.upto(race)
        xt = [(1, "1"), (10, "10"), (100, "100"), (1000, "1,000")]
        self.loss_chart.axes(sc, self.cam, self.P, op, "training step (log)", "training loss",
                             xt, [(v, "%g" % v) for v in (0, 1, 2)])
        self.acc_chart.axes(sc, self.cam, self.P, op, "training step (log)", "test accuracy",
                            xt, [(v, "%d%%" % round(100 * v)) for v in (0, 0.5, 1.0)])
        steps = self.steps[1:]
        for i, a in enumerate(ACTS):
            self.loss_chart.curve(sc, self.P, steps, self.loss[i][1:], upto, a.ink, op, width=1.5)
            self.acc_chart.curve(sc, self.P, steps, self.acc[i][1:], upto, a.ink, op, width=1.5)
        # the names in a row under the charts
        labels = [a.label for a in ACTS]
        widths = [layout.text_width(s, layout.MIN_TEXT) for s in labels]
        gap = 0.03
        total = sum(widths) + gap * (len(labels) - 1)
        x = 0.5 - total / 2
        fy = self.fr(self.P(0.5, LEGEND_Y))[1]
        for a, s, w in zip(ACTS, labels, widths):
            sc.text(x, fy, s, layout.MIN_TEXT, "regular", a.ink, op, "la")
            x += w + gap
        sc.probe(("race",), (np.log10(upto) * 100, 0, 0), tol=6.0)

    # -- the table ----------------------------------------------------------------------
    def _row_y(self, r):
        return TABLE_Y0 + r * ROW

    def _table(self, sc, t, op):
        if op <= 0.01:
            return
        cost = smoothstep(0.0, 0.3, self.progress("cost", t))
        heads = ("activation", "weights", "training loss", "test accuracy", "time per step")
        anchors = ("lm", "lm", "rm", "rm", "lm")
        for x, s, anc in zip(COLS, heads, anchors):
            o = op * (cost if s == "time per step" else 1.0)
            self.label(sc, self.P(x, self._row_y(0)), s, layout.MIN_TEXT, anchor=anc, ink=HEAD_INK,
                       opacity=o)
        best = ties(self.final_acc)
        longest = float(self.ms.max())
        for i, a in enumerate(ACTS):
            y = self._row_y(i + 1)
            strong = layout.STRONG_INK if i in best else layout.CAPTION_INK
            cells = ((COLS[0], a.label, "lm", a.ink), (COLS[1], a.init_label, "lm", HEAD_INK),
                     (COLS[2], num(self.final_loss[i], 3), "rm", layout.CAPTION_INK),
                     (COLS[3], "%.1f%%" % (100 * self.final_acc[i]), "rm", strong))
            for x, s, anc, ink in cells:
                self.label(sc, self.P(x, y), s, 0.03, anchor=anc, ink=ink, opacity=op)
            if cost > 0:
                grow = smoothstep(0.0, 1.0, remap(self.progress("cost", t), 0.1, 0.6))
                w = BAR_W * self.ms[i] / longest * grow
                a0, a1 = self.P(BAR_X, y), self.P(BAR_X + w, y)
                px = self.px(a1)[0] - self.px(a0)[0]
                if px > 1:
                    self.rect(sc, (a0 + a1) / 2, px, BAR_H, a.ink, 0.85 * op * cost, radius=2.0,
                              lift=0.2)
                self.label(sc, a1, "%.1f ms" % self.ms[i], layout.MIN_TEXT, anchor="lm", dx=6,
                           ink=layout.CAPTION_INK, opacity=op * cost * smoothstep(0.9, 1.0, grow))

    # ------------------------------------------------------------------ words
    def _words(self, t, race, reset):
        T = self.Tm
        if reset > 0 or t < T["plot:" + ACTS[0].key][0]:
            return ("one network, 784 → %d → %d → 10, trained seven times on MNIST, each with a "
                    "different activation and the matching init" % RACE_SIZES[1:3])
        if t < T["race"][0] - 1.5:
            k = max(i for i, a in enumerate(ACTS) if t >= T["plot:" + a.key][0])
            return self._function_words()[k]
        if t < T["results"][0]:
            step = int(round(self.upto(race)))
            return ("the same data, batches and %s Adam steps (lr %g): step %s of %s; training "
                    "loss left, test accuracy right, time on a log axis"
                    % (f"{STEPS:,}", RACE["lr"], f"{step:,}", f"{STEPS:,}"))
        if t < T["cost"][0]:
            return self._results_words()
        return self._cost_words()

    def _function_words(self):
        return [
            "sigmoid σ(z) = 1 / (1 + e⁻ᶻ) squashes z into (0, 1); its slope is at most 0.25; "
            "Xavier weights, std 1/√n",
            "tanh squashes into (−1, 1), centred on 0, slope up to 1; Xavier weights",
            "ReLU = max(0, z): slope exactly 1 or exactly 0; He weights, std √(2/n), keep the "
            "variance through the zeros",
            "Leaky ReLU: 0.01 z for z < 0, so a unit is never completely off; He weights",
            "GELU = z Φ(z) (tanh form, as GPT-2): a smooth ReLU that dips to %s; He weights"
            % num(_min_of("gelu"), 2),
            "SiLU = z σ(z) (Swish): smooth like GELU, dips to %s; He weights"
            % num(_min_of("silu"), 2),
            "ELU: z for z > 0, e^z − 1 below: smooth, saturating at −1 on the left; He weights",
        ]

    def _results_words(self):
        best = ties(self.final_acc)
        order = np.argsort(-self.final_acc)
        rest = [int(i) for i in order if i not in best]
        lo, hi = 100 * self.final_acc[best].min(), 100 * self.final_acc[best].max()
        if len(best) > 1:
            head = "%s tie at %.1f–%.1f%% test accuracy" % (names(best), lo, hi)
        else:
            head = "%s leads at %.1f%%" % (names(best), hi)
        tail = "; ".join("%s %.1f%%" % (ACTS[i].label, 100 * self.final_acc[i]) for i in rest)
        return "after %s steps: %s (within %.1f%%); %s" % (f"{STEPS:,}", head, 100 * TIE, tail)

    def _cost_words(self):
        i_relu = next(i for i, a in enumerate(ACTS) if a.key == "relu")
        slowest = int(np.argmax(self.ms))
        fn_slow = int(np.argmax(self.us))
        return ("one step measured: %.1f ms with ReLU, %.1f with %s (slowest); the function alone "
                "costs %.0f µs for ReLU, %.0f for %s on a 64 × 128 layer: the matrix products "
                "dominate" % (self.ms[i_relu], self.ms[slowest], ACTS[slowest].label,
                              self.us[i_relu], self.us[fn_slow], ACTS[fn_slow].label))


def _min_of(key):
    """The lowest value of activation ``key`` over the plotted range."""
    return float(act(key, np.linspace(-6, 0, 2000)).min())

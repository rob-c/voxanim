"""Seven regularisers, one network: the 784-256-256-10 MLP memorising 1,000
MNIST digits, trained plain and with L2, L1, dropout, input noise, max-norm
and label smoothing, each at the strength that did best on 2,000 validation
digits (:func:`voxanim.regul.race_sweeps`).

    3,000 Adam steps (learning rate 0.001, batches of 64), He weights, ReLU;
    the same data, batches and initial weights for every run; 2,000 test
    digits that no choice was made on

1. **memorising** -- the plain network alone: its training loss falls to
   about 10⁻⁵ (log axis) while its test loss, lowest at the step where the
   validation loss is lowest (early stopping's pick, marked), climbs for
   the rest of training; its test accuracy hardly moves -- it grows sure of
   its mistakes,
2. **regularisers** -- the six regularised runs join the two charts one at a
   time, each with what it does and where its curves end,
3. **results** -- the final table: chosen strength, training loss, test loss
   and test accuracy, honest about which differences the 2,000 test digits
   can resolve (:data:`TIE`),
4. **strength** -- the sweeps behind the choices for L2, L1 and input noise:
   accuracy on the training digits and on the validation digits against the
   strength (log axis): too little and the network memorises, too much and
   it cannot fit even the training digits.

Every number on screen is read from the recorded runs.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .diagram import Diagram
from .easing import remap, smoothstep
from .losschart import LossChart, sup
from .regul import (
    PLAIN_INK,
    RACE_N,
    RACE_SIZES,
    RACE_TEST,
    RACE_TRAIN,
    RACE_VAL,
    race_data,
    race_runs,
    race_sweeps,
    val_acc,
)
from .text import TIMES, num

STEPS = RACE_TRAIN["steps"]
TIE = 0.007                     # accuracy differences under this are within the test set's noise
NEAR = 0.05                     # test losses this close to plain's count as no change
TRAIN_BOX = ((0.08, 0.46), (0.30, 0.80))      # the two race charts ((x0, x1), (top, bottom))
TEST_BOX = ((0.58, 0.96), (0.30, 0.80))
TRAIN_RANGE, TEST_RANGE = (1e-5, 3.0), (0.0, 0.8)
LEGEND_Y = 0.93
TABLE_Y0, ROW = 0.27, 0.074
COLS = (0.03, 0.27, 0.58, 0.76, 0.96)       # name, strength, training loss, test loss, accuracy
SWEEP_KEYS = ("l2", "l1", "noise")
SWEEP_X = ((0.08, 0.32), (0.40, 0.64), (0.72, 0.96))
SWEEP_Y = (0.33, 0.78)
ACC_RANGE = (0.75, 1.0)
TRAIN_ACC_INK = (0.55, 0.57, 0.59)
HEAD_INK = layout.NOTE_INK
PLAIN_T, REG_DT = 7.0, 3.3      # seconds: the plain run, then each regulariser
RESULTS_T, SWEEP_DT, SUMMARY_T = 7.5, 4.0, 3.0


def strength_text(v):
    """A strength as it is written: ``3×10⁻⁴``, ``10⁻⁵``, ``0.5``."""
    v = float(v)
    if 0.01 <= v < 1000:
        return "%g" % v
    e = int(np.floor(np.log10(v) + 1e-9))
    m = v / 10.0 ** e
    if abs(m - 1.0) < 1e-6:
        return "10" + sup(e)
    return "%g%s10%s" % (round(m, 2), TIMES, sup(e))


def loss_text(v):
    """A loss: ``0.56``, ``0.010``, ``1.6×10⁻³``."""
    v = float(v)
    if v >= 0.1:
        return num(v, 2)
    if v >= 0.01:
        return num(v, 3)
    return strength_text(float("%.2g" % v))


def pct(v):
    """``0.9065`` as ``90.7%``; a perfect score as ``100%``."""
    return "100%" if v >= 1.0 else "%.1f%%" % (100 * v)


def names(items):
    """``"a, b and c"``."""
    items = list(items)
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def runs():
    """``(plain, chosen, sweeps, plain_val)``: everything the animation reads."""
    dat = race_data()
    sweeps = race_sweeps(dat=dat)
    plain, chosen = race_runs(dat=dat)
    return plain, chosen, sweeps, val_acc(dat, plain)


class RegularisationRace(Diagram):
    """The plain run, the race, the table and the sweeps (see the module)."""

    PHASES = ("memorising", "regularisers", "results", "strength")

    def __init__(self, data=None):
        super().__init__("Seven regularisers, one network", "")
        self.plain, self.chosen, self.sweeps, self.plain_val = runs() if data is None else data
        self.hold = 2.5
        self.steps = self.plain.steps
        self.train_chart = LossChart(TRAIN_BOX, (0, STEPS), TRAIN_RANGE, ylog=True)
        self.test_chart = LossChart(TEST_BOX, (0, STEPS), TEST_RANGE)
        self.sweep_charts = {}
        for key, xs in zip(SWEEP_KEYS, SWEEP_X):
            g = self.sweeps[key].reg.grid
            self.sweep_charts[key] = LossChart((xs, SWEEP_Y), (g[0], g[-1]), ACC_RANGE,
                                               xlog=True)
        # every line of the table: plain first
        self.rows = [("no regulariser", None, None, self.plain, PLAIN_INK)]
        self.rows += [(reg.label, reg, s, r, reg.ink) for reg, s, r in self.chosen]
        Tm, t = {}, 0.6
        Tm["plain"] = (t, t + PLAIN_T)
        t += PLAIN_T + 0.6
        for reg, _s, _r in self.chosen:
            Tm["reg:" + reg.key] = (t, t + REG_DT)
            t += REG_DT
        t += 1.5
        Tm["results"] = (t, t + RESULTS_T)
        t += RESULTS_T + 0.5
        for key in SWEEP_KEYS:
            Tm["sweep:" + key] = (t, t + SWEEP_DT)
            t += SWEEP_DT
        Tm["summary"] = (t, t + SUMMARY_T)
        t += SUMMARY_T
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["reg:" + self.chosen[0][0].key][0], Tm["results"][0],
                      Tm["sweep:" + SWEEP_KEYS[0]][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        ph = 0 if reset > 0 else self.step_of(t)
        results = self.progress("results", t)
        s0 = self.Tm["sweep:" + SWEEP_KEYS[0]][0]
        leaving = smoothstep(0.0, 1.0, remap(t, s0 - 1.0, s0 - 0.55))      # the table goes,
        sweeping = smoothstep(0.0, 1.0, remap(t, s0 - 0.5, s0 + 0.2))      # then the sweeps come
        intro = smoothstep(0.0, 1.0, remap(t, 0.0, self.Tm["plain"][0] + 0.4))
        charts = intro * (1.0 - smoothstep(0.0, 0.08, results)) * back
        table = smoothstep(0.1, 0.25, results) * (1.0 - leaving) * back
        self._race(sc, t, charts)
        self._table(sc, table)
        self._sweeps(sc, t, sweeping * back)
        self.step_indicator(sc, ph, self.PHASES)
        self.subtitle = self._words(t, reset)

    # -- the race ----------------------------------------------------------------------
    def _upto(self, key, t):
        """The training step run ``key`` has been drawn to at time ``t``."""
        return STEPS * remap(self.progress(key, t), 0.0, 0.9)

    def _race(self, sc, t, op):
        op *= self._cap
        if op <= 0.01:
            return
        xt = [(0, "0"), (1000, "1,000"), (2000, "2,000"), (3000, "3,000")]
        self.train_chart.axes(sc, self.cam, self.P, op, "training step", "training loss (log)",
                              xt, [(1e-4, "10" + sup(-4)), (1e-2, "10" + sup(-2)), (1.0, "1")])
        self.test_chart.axes(sc, self.cam, self.P, op, "training step", "test loss", xt,
                             [(v, "%g" % v) for v in (0, 0.2, 0.4, 0.6)])
        plain = self.plain
        early = smoothstep(0.0, 0.3, remap(self.progress("plain", t), 0.5, 1.0))
        if early > 0:
            s = float(plain.steps[plain.best])
            self.test_chart.mark(sc, self.P, s, PLAIN_INK, op * early)
            self.label(sc, self.P(self.test_chart.fx(s), TEST_BOX[1][0] + 0.012),
                       "lowest validation loss: step %d" % s, layout.MIN_TEXT, anchor="lt",
                       dx=5, ink=HEAD_INK, opacity=op * early)
        shown = []
        current = None
        for name, reg, _s, r, ink in self.rows:
            key = "plain" if reg is None else "reg:" + reg.key
            upto = self._upto(key, t)
            if t < self.Tm[key][0]:
                continue
            shown.append(name)
            if self.progress(key, t) < 1.0:
                current = key
            width = 2.2 if current == key else 1.5
            self.train_chart.curve(sc, self.P, r.steps, r.train_loss, upto, ink, op, width=width)
            self.test_chart.curve(sc, self.P, r.steps, r.test_loss, upto, ink, op, width=width)
        widths = [layout.text_width(row[0], layout.MIN_TEXT) for row in self.rows]
        gap = 0.026
        total = sum(widths) + gap * (len(widths) - 1)
        x = 0.5 - total / 2
        fy = self.fr(self.P(0.5, LEGEND_Y))[1]
        for (name, _reg, _s, _r, ink), w in zip(self.rows, widths):
            if name in shown:
                sc.text(x, fy, name, layout.MIN_TEXT, "regular", ink, op, "la")
            x += w + gap
        sc.probe(("race",), (t * 10.0, 0, 0), tol=6.0)

    # -- the table ----------------------------------------------------------------------
    def _row_y(self, r):
        return TABLE_Y0 + r * ROW

    def better(self):
        """Indices of :attr:`rows` whose test accuracy beats plain's by more than :data:`TIE`."""
        base = float(self.plain.test_acc[-1])
        return [i for i, row in enumerate(self.rows)
                if float(row[3].test_acc[-1]) > base + TIE]

    def _table(self, sc, op):
        if op <= 0.01:
            return
        heads = ("regulariser", "strength", "training loss", "test loss", "test accuracy")
        anchors = ("lm", "lm", "rm", "rm", "rm")
        for x, s, anc in zip(COLS, heads, anchors):
            self.label(sc, self.P(x, self._row_y(0)), s, layout.MIN_TEXT, anchor=anc,
                       ink=HEAD_INK, opacity=op)
        best = self.better()
        for i, (name, reg, s, r, ink) in enumerate(self.rows):
            y = self._row_y(i + 1)
            strong = layout.STRONG_INK if i in best else layout.CAPTION_INK
            stren = "—" if reg is None else "%s = %s" % (reg.symbol, strength_text(s))
            cells = ((COLS[0], name, "lm", ink), (COLS[1], stren, "lm", HEAD_INK),
                     (COLS[2], loss_text(r.train_loss[-1]), "rm", layout.CAPTION_INK),
                     (COLS[3], loss_text(r.test_loss[-1]), "rm", layout.CAPTION_INK),
                     (COLS[4], pct(r.test_acc[-1]), "rm", strong))
            for x, txt, anc, ink_ in cells:
                self.label(sc, self.P(x, y), txt, 0.03, anchor=anc, ink=ink_, opacity=op)

    # -- the sweeps ----------------------------------------------------------------------
    def _sweeps(self, sc, t, op):
        op *= self._cap
        if op <= 0.01:
            return
        for key in SWEEP_KEYS:
            u = self.progress("sweep:" + key, t)
            if t < self.Tm["sweep:" + key][0] - 0.3:
                continue
            o = op * smoothstep(0.0, 0.15, remap(t, self.Tm["sweep:" + key][0] - 0.3,
                                                 self.Tm["sweep:" + key][0] + 0.3))
            self._sweep(sc, key, u, o)
        fy = self.fr(self.P(0.5, LEGEND_Y))[1]
        entries = (("grey: accuracy on the %s training digits" % f"{RACE_N:,}", TRAIN_ACC_INK),
                   ("coloured: on %s validation digits" % f"{RACE_VAL:,}",
                    layout.CAPTION_INK),
                   ("dashed: no regulariser", HEAD_INK))
        widths = [layout.text_width(e, layout.MIN_TEXT) for e, _ in entries]
        gap = 0.035
        x = 0.5 - (sum(widths) + gap * (len(widths) - 1)) / 2
        for (e, ink), w in zip(entries, widths):
            sc.text(x, fy, e, layout.MIN_TEXT, "regular", ink, op, "la")
            x += w + gap

    def _sweep(self, sc, key, u, op):
        sw = self.sweeps[key]
        reg, chart = sw.reg, self.sweep_charts[key]
        g = np.asarray(reg.grid, np.float64)
        decades = range(int(np.ceil(np.log10(g[0]) - 1e-9)), int(np.floor(np.log10(g[-1])
                                                                          + 1e-9)) + 1)
        if key == "noise":
            xt = [(v, "%g" % v) for v in (0.2, 0.5, 1.0) if g[0] <= v <= g[-1]]
        else:
            xt = [(10.0 ** e, "10" + sup(e)) for e in decades]
        chart.axes(sc, self.cam, self.P, op, "%s (log)" % reg.symbol, "", xt,
                   [(v, "%d%%" % round(100 * v)) for v in (0.8, 0.9, 1.0)])
        x0, y0, y1 = chart.x0, chart.y0, chart.y1
        self.label(sc, self.P(x0, y0 - 0.035), reg.label, 0.031, anchor="lb", ink=reg.ink,
                   opacity=op)
        # plain's validation accuracy: the line every strength is measured against
        chart.hline(sc, self.P, self.plain_val, op)
        upto = float(np.exp(np.log(g[0]) + remap(u, 0.05, 0.7) * (np.log(g[-1]) - np.log(g[0]))))
        for vals, ink in ((sw.train_acc, TRAIN_ACC_INK), (sw.val_acc, reg.ink)):
            chart.curve(sc, self.P, g, vals, upto, ink, op, width=1.6, dot=False)
            for gi, v in zip(g, vals):
                if gi <= upto * (1 + 1e-9):
                    self.disk(sc, self.P(chart.fx(gi), chart.fy(v)), 3.0, ink, op, lift=0.6)
        # a value below the axis: written next to its pinned point
        lo = [(gi, v) for gi, v in zip(g, sw.val_acc) if v < ACC_RANGE[0] and gi <= upto]
        for gi, v in lo:
            self.label(sc, self.P(chart.fx(gi), y1), pct(v) + " ↓", layout.MIN_TEXT,
                       anchor="rb", dx=-6, dy=-4, ink=reg.ink, opacity=op)
        pick = smoothstep(0.75, 0.9, u)
        if pick > 0:
            gb = g[sw.best]
            chart.mark(sc, self.P, gb, reg.ink, op * pick)

    # ------------------------------------------------------------------ words
    def _words(self, t, reset):
        T = self.Tm
        p = self.plain
        if reset > 0 or t < T["plain"][0] + 0.8:
            return ("784 → %d → %d → 10 and 3,000 Adam steps (lr %g) on just %s digits: "
                    "enough weights to memorise every one of them"
                    % (RACE_SIZES[1], RACE_SIZES[2], RACE_TRAIN["lr"], f"{RACE_N:,}"))
        if t < T["reg:" + self.chosen[0][0].key][0]:
            if self.progress("plain", t) < 0.55:
                return ("no regulariser: the training loss falls to %s; the test loss (%s "
                        "digits it never sees) stops falling at step %d"
                        % (loss_text(p.train_loss.min()), f"{RACE_TEST:,}",
                           p.steps[int(np.argmin(p.test_loss))]))
            b = p.best
            return ("then climbs from %s to %s, while test accuracy moves only from %s to %s: "
                    "it grows sure of its mistakes; early stopping would keep step %d"
                    % (num(p.test_loss[b], 2), num(p.test_loss[-1], 2), pct(p.test_acc[b]),
                       pct(p.test_acc[-1]), p.steps[b]))
        if t < T["results"][0]:
            for reg, s, r in self.chosen:
                if t < T["reg:" + reg.key][1] or reg is self.chosen[-1][0]:
                    return self._reg_words(reg, s, r)
        if t < T["sweep:" + SWEEP_KEYS[0]][0]:
            return self._results_words(self.progress("results", t) >= 0.5)
        for key in SWEEP_KEYS:
            if t < T["sweep:" + key][1]:
                return self._sweep_words(key, self.progress("sweep:" + key, t) >= 0.72)
        return self._summary_words()

    def _vs_plain(self, r):
        """How a run's final test loss compares with plain's."""
        a, b = float(r.test_loss[-1]), float(self.plain.test_loss[-1])
        if abs(a - b) < NEAR:
            return "test loss %s, close to plain's %s" % (num(a, 2), num(b, 2))
        return "test loss %s (plain %s)" % (num(a, 2), num(b, 2))

    def _reg_words(self, reg, s, r):
        st = strength_text(s)
        tr = loss_text(r.train_loss[-1])
        acc = pct(r.test_acc[-1])
        vs = self._vs_plain(r)
        if reg.key == "l2":
            return ("L2, λ = %s: λ/2 · Σw² added to the loss pulls every weight towards 0; "
                    "training loss %s, %s, %s accuracy" % (st, tr, vs, acc))
        if reg.key == "l1":
            return ("L1, λ = %s: λ · Σ|w| (by its slope, under Adam); training loss %s, %s, "
                    "%s accuracy" % (st, tr, vs, acc))
        if reg.key == "dropout":
            return ("dropout, p = %s: each step drops that share of hidden units at random; it "
                    "still memorises (training loss %s): %s, %s accuracy" % (st, tr, vs, acc))
        if reg.key == "noise":
            return ("input noise, σ = %s: fresh Gaussian noise on every training image, so no "
                    "pixel can be memorised; training loss %s, %s, %s accuracy"
                    % (st, tr, vs, acc))
        if reg.key == "maxnorm":
            return ("max-norm, c = %s: each hidden unit's incoming weights capped at length %s; "
                    "training loss %s, %s, %s accuracy" % (st, st, tr, vs, acc))
        eps = float(s)
        target = 1.0 - eps + eps / 10
        return ("label smoothing, ε = %s: the target is %s on the label, so the training loss "
                "cannot fall below −log %s = %s (it ends at %s); %s, %s accuracy"
                % (st, num(target, 2), num(target, 2), num(-np.log(target), 3),
                   num(r.train_loss[-1], 3), vs, acc))

    def _results_words(self, second):
        """The table in words: who beats plain (``second`` False), then the ties
        and what they still change."""
        base = float(self.plain.test_acc[-1])
        best = self.better()
        win = [self.rows[i] for i in sorted(best, key=lambda i: -self.rows[i][3].test_acc[-1])]
        ties = [row[0] for i, row in enumerate(self.rows[1:], 1) if i not in best]
        head = ("%s beat%s plain's %s"
                % (names("%s %s" % (row[0], pct(row[3].test_acc[-1])) for row in win),
                   "" if len(win) > 1 else "s", pct(base))) if win else \
            "no regulariser beats plain's %s" % pct(base)
        if not second:
            return ("after %s steps: %s by more than %.1f points, the spread of %s test digits"
                    % (f"{STEPS:,}", head, 100 * TIE, f"{RACE_TEST:,}"))
        tail = "%s tie with plain on accuracy" % names(ties) if ties else "nothing ties"
        cut = [row for row in self.rows[1:] if row[0] in ties
               and float(row[3].test_loss[-1]) < float(self.plain.test_loss[-1]) - NEAR]
        extra = ""
        if cut:
            extra = ", but %s cut%s the test loss from %s to %s: less sure of the mistakes" % (
                names(row[0] for row in cut), "" if len(cut) > 1 else "s",
                num(self.plain.test_loss[-1], 2),
                names(num(row[3].test_loss[-1], 2) for row in cut))
        return tail + extra

    def _sweep_words(self, key, second):
        """One sweep in words: too little and the best (``second`` False), then too much."""
        sw = self.sweeps[key]
        reg = sw.reg
        g = reg.grid
        b = sw.best
        sym = reg.symbol
        lo = "too little (%s = %s): %s on the training digits, %s on validation" % (
            sym, strength_text(g[0]), pct(sw.train_acc[0]), pct(sw.val_acc[0]))
        mid = "best %s = %s (%s; none: %s)" % (sym, strength_text(g[b]), pct(sw.val_acc[b]),
                                              pct(self.plain_val))
        if sw.val_acc[-1] < 0.2:
            zero = float(np.mean([np.mean(np.abs(sw.runs[-1].P["W%d" % l]) < 1e-3)
                                  for l in range(len(RACE_SIZES) - 1)]))
            hi = ("too much (%s = %s): %s of the weights within 10⁻³ of 0, accuracy at chance "
                  "(%s)" % (sym, strength_text(g[-1]), "%.0f%%" % (100 * zero),
                            pct(sw.val_acc[-1])))
        else:
            hi = ("too much (%s = %s) and it cannot fit even the training digits: %s on them, "
                  "%s on validation" % (sym, strength_text(g[-1]), pct(sw.train_acc[-1]),
                                        pct(sw.val_acc[-1])))
        if not second:
            return "%s, %d strengths: %s; %s" % (reg.label, len(g), lo, mid)
        return "%s, %s" % (reg.label, hi)

    def _summary_words(self):
        n = self.sweeps["noise"]
        return ("each regulariser is a dial, set on held-out digits: input noise's validation "
                "accuracy rises from %s to %s and falls to %s; the test digits only report"
                % (pct(n.val_acc[0]), pct(n.val_acc[n.best]), pct(n.val_acc[-1])))


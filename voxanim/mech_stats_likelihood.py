"""Parameter estimates, maximum-likelihood fits and errors: animations 128-132 of the
"next" catalogue (see :mod:`voxanim.mech`).

* :class:`LikelihoodCurve`: five decay times, the product of their probability densities,
  and the lifetime at which it peaks;
* :class:`LikelihoodIntervals`: the error read off where ln L has fallen by one half;
* :class:`ErrorPropagation`: a cloud of measurements pushed through a ratio, against the
  linear formula;
* :class:`ErrorOnTheMean`: an experiment repeated, and the spread of its mean;
* :class:`BootstrapErrors`: one dataset resampled to get an error without a formula.

Every number drawn is computed by :mod:`voxanim.study_stats_likelihood`; this module lays
them out (and evaluates that module's formulas at the value a slider is passing through).
Each phase says, bottom right, what kind of evidence it shows: "exact" (a formula or a
distribution evaluated) or "pseudo-experiments" (seeded draws from the model stated on
screen, with how many).
"""

from __future__ import annotations

import math

import numpy as np

from . import layout
from . import study_stats_likelihood as S
from .easing import smoothstep
from .losschart import LossChart, pow10
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIGIT_INK,
    DIM,
    GREEN,
    MAGENTA,
    PAPER,
    RED,
    TEAL,
    VIOLET,
    dark,
    soft,
)
from .mech_curios import progress
from .mech_random import Random, raster

EXACT, TOYS = S.EXACT, S.TOYS
BIG, HEAD = 0.036, 0.033
GREY = (0.50, 0.52, 0.54)
STRONG = layout.STRONG_INK
EVENT_INK = (RED, BLUE, GREEN, AMBER, VIOLET)


def commas(n):
    return format(int(n), ",")


def pct(v, digits=1):
    return "%.*f%%" % (digits, 100.0 * float(v))


def signed(v, digits=3):
    """A number with a true minus sign."""
    return ("%.*f" % (digits, float(v))).replace("-", "−")


class Stats(Random):
    """What the five animations share: curves and histograms on a chart, dashed marks,
    legends and the evidence tag."""

    def tag(self, sc, kind, op):
        """The kind of evidence on screen, bottom right."""
        self.text(sc, 0.95, 0.925, kind, op, anchor="rm", ink=dark(TEAL, 0.25))

    def path(self, sc, chart, xs, ys, ink, op, width=2.6, go=1.0, lift=0.0):
        """``ys`` against ``xs`` on ``chart``, as far as fraction ``go`` of the points;
        parts outside the chart's range are left out."""
        if op <= 0.01 or go <= 0:
            return
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        if len(xs) < 2:                 # a curve still growing from one point: nothing yet
            return
        n = len(xs) if go >= 1 else max(2, int(np.ceil(len(xs) * go)))
        step = max(1, n // 140)
        keep = list(range(0, n, step))
        if keep[-1] != n - 1:
            keep.append(n - 1)
        (xa, xb), (ya, yb) = sorted(chart.x_range), sorted(chart.y_range)
        run = []
        for i in keep:
            if xa <= xs[i] <= xb and ya <= ys[i] <= yb:
                run.append(self.P(chart.fx(float(xs[i])), chart.fy(float(ys[i])), lift=lift))
                continue
            if len(run) > 1:
                self.polyline(sc, run, width, ink, op, bias=0.3)
            run = []
        if len(run) > 1:
            self.polyline(sc, run, width, ink, op, bias=0.3)

    def hist(self, sc, chart, mids, dens, ink, op, grow=1.0, gap=0.14):
        """A histogram as bars rising from the chart's floor."""
        if op <= 0.01 or grow <= 0:
            return
        w = self.span(abs(chart.fx(float(mids[1])) - chart.fx(float(mids[0]))) * (1 - gap))[0]
        floor = chart.y1
        for m, v in zip(mids, dens):
            h = floor - chart.fy(float(v) * grow + chart.y_range[0])
            if h * self.cam.height < 0.7:
                continue
            self.rect(sc, self.P(chart.fx(float(m)), floor - h / 2), w, self.span(0, h)[1], ink,
                      op, radius=min(1.5, w / 2), lift=0.3)

    def dash(self, sc, a, b, ink, op, width=1.8, n=12):
        """A dashed straight line between two layout points."""
        if op <= 0.01:
            return
        (ax, ay), (bx, by) = a, b
        for i in range(n):
            u0, u1 = i / n, (i + 0.55) / n
            self.polyline(sc, [self.P(ax + (bx - ax) * u0, ay + (by - ay) * u0, lift=0.8),
                               self.P(ax + (bx - ax) * u1, ay + (by - ay) * u1, lift=0.8)],
                          width, ink, op)

    def vmark(self, sc, chart, x, ink, op, width=1.8, dashed=True, top=None):
        """A vertical mark at ``x`` over the chart's height (or up to value ``top``)."""
        fx = chart.fx(float(x))
        y0 = chart.y0 if top is None else chart.fy(float(top))
        if dashed:
            self.dash(sc, (fx, chart.y1), (fx, y0), ink, op, width)
        else:
            self.polyline(sc, [self.P(fx, chart.y1, lift=0.8), self.P(fx, y0, lift=0.8)], width,
                          ink, op)

    def hmark(self, sc, chart, y, ink, op, width=1.8, n=16):
        fy = chart.fy(float(y))
        self.dash(sc, (chart.x0, fy), (chart.x1, fy), ink, op, width, n)

    def point(self, sc, chart, x, y, ink, op, r=5.0, glow=0.0):
        self.disk(sc, self.P(chart.fx(float(x)), chart.fy(float(y))), r, ink, op, lift=1.5,
                  outline=0.0, glow=glow)

    def key(self, sc, fx, fy, entries, op, step=0.046, size=0.030, line=False):
        """A legend: a colour chip (or a short line) and a name per entry, stacked down."""
        for n, (name, ink) in enumerate(entries):
            y = fy + n * step
            if line:
                self.polyline(sc, [self.P(fx - 0.012, y), self.P(fx + 0.012, y)], 3.2, ink, op)
            else:
                self.box(sc, fx, y, 14, 14, ink, op, radius=3)
            self.text(sc, fx + 0.022, y, name, op, anchor="lm", ink=dark(ink, 0.3), size=size)

    def stack(self, sc, values, at, fy, ink, op, shown=None, r=3.4, bin_px=7.4):
        """A dot per value on a horizontal axis (``at``: value -> layout x), piled up
        where they share a ``bin_px``-wide bin; the first ``shown`` only."""
        piles = {}
        for n, v in enumerate(values):
            if shown is not None and n >= shown:
                break
            b = int(round(self.span(at(float(v)))[0] / bin_px))
            k = piles[b] = piles.get(b, 0) + 1
            dy = self.frac(0, (k - 0.5) * (2 * r + 0.6))[1]
            self.disk(sc, self.P(self.frac(b * bin_px)[0], fy - dy), r, ink, op, lift=1.2,
                      outline=0.12)


# ====================================================================== 128: the curve
class LikelihoodCurve(Stats):
    """Five decay times under an exponential; for one lifetime, the five densities and
    their product; the product traced as the lifetime slides, peaking at the sample
    mean; the same as a sum of logs; and where the peak lands when the experiment is
    repeated."""

    TITLE = "The likelihood curve"
    PHASES = (("the data", 9.0), ("one lifetime", 13.0), ("slide it", 13.0),
              ("sum of logs", 9.0), ("repeat", 10.0))
    T_MAX, F_MAX = 4.5, 1.3

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        self.times = np.asarray(s.panels[0].data["times"], float)
        self.trials = [float(v) for v in s.panels[0].data["trials"]]
        self.product = [float(v) for v in s.panels[0].data["product"]]
        self.logs = np.asarray(s.panels[0].data["logs"], float)
        self.log_sum = [float(v) for v in s.panels[0].data["log_sum"]]
        self.grid = np.asarray(s.panels[1].data["x"], float)
        self.like = np.asarray(s.panels[1].data["ys"], float)[0]
        self.best, self.peak = float(m["best_tau"]), float(m["grid_peak"])
        self.more = np.asarray(s.panels[2].data["ys"], float)
        self.more_best = [float(v) for v in s.panels[2].data["best"]]
        stream = S.likelihood_data()["stream"]
        self.lnl_all = float(S.log_likelihood(stream, stream.mean()))
        self.toy_x = np.asarray(s.panels[3].data["x"], float)
        self.toy_hist, self.toy_exact = np.asarray(s.panels[3].data["ys"], float)
        self.toy_first = np.asarray(s.panels[3].data["first"], float)
        self.toy_mean, self.toy_sd = float(m["toy_mean"][0]), float(m["toy_sd"][0])

    def subtitle_for(self, k):
        m = self.meta
        return [
            "five muon decay times, drawn from an exponential whose mean lifetime τ is "
            "%.1f μs (seed %d). The curve gives the probability density of each time; τ is "
            "what we want to estimate from the five" % (m["true_tau"], m["seed"]),
            "for one trial value of τ, each event's density is the height of the curve at its "
            "time. The likelihood L is the product of the five: %s at τ = %.0f μs, %s at "
            "τ = %.0f μs" % (pow10(self.product[0], 1), self.trials[0],
                             pow10(self.product[1], 1), self.trials[1]),
            "slide τ and the product rises, then falls. Its peak is the maximum-likelihood "
            "estimate: τ = %.3f μs on this grid of step %.4f. The mean of the five times is "
            "%.3f μs" % (self.peak, m["grid_step"], self.best),
            "the log of a product is a sum: ln L = %s at the peak, and the peak has not "
            "moved. With 20 and 80 events of the same run the peak narrows, at τ = %.2f and "
            "%.2f μs" % (signed(self.log_sum[2]), self.more_best[1], self.more_best[2]),
            "%s pseudo-experiments of five decays each. Their best τ averages %.3f μs (truth "
            "%.1f) and spreads by %.3f μs; exact: τ/√5 = %.3f. One experiment lands "
            "anywhere in that spread" % (commas(m["toys"]), self.toy_mean, m["true_tau"],
                                         self.toy_sd, m["sd_exact"]),
        ][k]

    # ------------------------------------------------------------------ the model chart
    def _model(self, sc, box, tau, op, rise=None, curve=1.0, ink=DIGIT_INK, events=None,
               ticks=True):
        """The exponential of lifetime ``tau`` over the time axis, with a bar from each
        event up to the curve (``rise``: how far each has risen, 0..1)."""
        chart = LossChart(box, (0.0, self.T_MAX), (0.0, self.F_MAX))
        self.axes(sc, chart, op, "decay time t (μs)" if ticks else "",
                  "probability density f(t; τ)",
                  xticks=[(v, "%d" % v) for v in range(5)] if ticks else [],
                  yticks=[(0, "0"), (0.5, "0.5"), (1.0, "1")])
        ts = np.linspace(0.0, self.T_MAX, 90)
        self.path(sc, chart, ts, S.exp_density(ts, tau), ink, op, 2.8, go=curve)
        for i, t in enumerate(self.times):
            on = 1.0 if events is None else events[i]
            if on <= 0:
                continue
            x = chart.fx(t)
            r = 0.0 if rise is None else float(rise[i])
            if r > 0:
                top = chart.fy(min(float(S.exp_density(t, tau)), self.F_MAX) * r)
                self.polyline(sc, [self.P(x, chart.y1, lift=0.9), self.P(x, top, lift=0.9)],
                              5.0, EVENT_INK[i], op * on)
                self.disk(sc, self.P(x, top), 5.0, EVENT_INK[i], op * on, lift=1.6, outline=0.0)
            self.disk(sc, self.P(x, chart.y1), 5.5, EVENT_INK[i], op * on, lift=1.5,
                      outline=0.15)
        return chart

    def phase_1(self, sc, u, op):
        m = self.meta
        box = ((0.07, 0.60), (0.40, 0.73))
        land = [smoothstep(0.04 + 0.09 * i, 0.14 + 0.09 * i, u) for i in range(len(self.times))]
        curve = smoothstep(0.58, 0.82, u)
        chart = self._model(sc, box, m["true_tau"], op, curve=curve, ink=GREY,
                            events=[1.0 if v >= 1 else 0.0 for v in land], ticks=False)
        self.text(sc, 0.335, chart.y1 + 0.085, "decay time t (μs)", op)
        for i, t in enumerate(self.times):
            x = chart.fx(t)
            if 0 < land[i] < 1:                        # the decay arriving
                self.disk(sc, self.P(x, chart.y0 + (chart.y1 - chart.y0) * land[i]), 6.0,
                          EVENT_INK[i], op, lift=1.6, outline=0.0, glow=0.4)
            if land[i] >= 1:
                self.text(sc, x, chart.y1 + 0.034, "%.2f" % t, op,
                          ink=dark(EVENT_INK[i], 0.3), size=0.031)
        self.text(sc, 0.34, 0.325, "five measured decay times", op, strong=True, size=BIG)
        x = 0.66
        self.text(sc, x, 0.385, "the model", op, anchor="lm", strong=True, size=HEAD)
        self.lines(sc, x, 0.44, ["f(t; τ) = (1/τ) exp(−t/τ)", "τ: the mean lifetime",
                                 "f: probability per μs of", "a decay at time t"], op,
                   size=0.031, step=0.046)
        late = op * smoothstep(0.62, 0.78, u)
        self.lines(sc, x, 0.66, [("grey curve: the truth,", dark(GREY, 0.2)),
                                 ("τ = %.1f μs" % m["true_tau"], dark(GREY, 0.2))], late,
                   size=0.031, step=0.046)
        self.text(sc, 0.07, 0.868, "which τ would you infer from these five times alone?",
                  op * smoothstep(0.84, 0.95, u), anchor="lm", strong=True, size=0.032)
        self.tag(sc, "%s: 5 times drawn from the model, seed %d" % (TOYS[:-1], m["seed"]), op)

    def _tau_two(self, u):
        return self.trials[0] + (self.trials[1] - self.trials[0]) * smoothstep(0.52, 0.62, u)

    def phase_2(self, sc, u, op):
        tau = self._tau_two(u)
        dens = S.exp_density(self.times, tau)
        rise = [smoothstep(0.03 + 0.045 * i, 0.10 + 0.045 * i, u) for i in range(5)]
        self._model(sc, ((0.09, 0.50), (0.40, 0.70)), tau, op, rise=rise)
        x = 0.60
        self.text(sc, x, 0.33, "try τ = %.1f μs" % tau, op, anchor="lm", strong=True, size=BIG)
        self.text(sc, x, 0.385, "height of the curve at each time:", op, anchor="lm",
                  size=0.030)
        for i, t in enumerate(self.times):
            on = op * smoothstep(0.6, 1.0, rise[i])
            self.box(sc, x + 0.008, 0.435 + 0.046 * i, 13, 13, EVENT_INK[i], on, radius=3)
            self.text(sc, x + 0.028, 0.435 + 0.046 * i, "f(%.2f) = %.3f" % (t, dens[i]), on,
                      anchor="lm", ink=dark(EVENT_INK[i], 0.3), size=0.032)
        # the numbers multiplying
        y = 0.86
        self.text(sc, 0.09, y, "L(τ) =", op * smoothstep(0.26, 0.3, u), anchor="lm",
                  strong=True, size=0.033)
        for i in range(5):
            on = op * smoothstep(0.27 + 0.03 * i, 0.31 + 0.03 * i, u)
            cx = 0.205 + 0.092 * i
            self.box(sc, cx, y, 62, 30, soft(EVENT_INK[i], 0.5), on, "%.3f" % dens[i],
                     border=EVENT_INK[i], size=0.031)
            if i:
                self.text(sc, cx - 0.046, y, "×", on, size=0.034, strong=True)
        self.text(sc, 0.205 + 0.092 * 4 + 0.043, y, "=  %s" % pow10(float(dens.prod()), 1),
                  op * smoothstep(0.42, 0.47, u), anchor="lm", strong=True, size=0.034)
        self.lines(sc, x, 0.685, ["the likelihood L: the product", "of the five densities"],
                   op * smoothstep(0.26, 0.32, u), size=0.030, step=0.042)
        self.text(sc, x, 0.775, "τ = %.0f:  L = %s" % (self.trials[0],
                                                      pow10(self.product[0], 1)),
                  op * smoothstep(0.44, 0.5, u), anchor="lm", size=0.032, ink=dark(TEAL, 0.3))
        self.text(sc, x + 0.19, 0.775, "τ = %.0f:  L = %s" % (self.trials[1],
                                                             pow10(self.product[1], 1)),
                  op * smoothstep(0.68, 0.76, u), anchor="lm", size=0.032, ink=dark(TEAL, 0.3))
        self.tag(sc, EXACT, op)

    def _tau_slide(self, u):
        lo, hi = float(self.grid[0]), float(self.grid[-1])
        out = lo + (hi - lo) * smoothstep(0.03, 0.56, u) ** 1.5
        return out + (self.best - out) * smoothstep(0.60, 0.74, u)

    def phase_3(self, sc, u, op):
        tau = self._tau_slide(u)
        dens = S.exp_density(self.times, tau)
        self._model(sc, ((0.08, 0.42), (0.40, 0.68)), tau, op, rise=[1.0] * 5)
        self.text(sc, 0.08, 0.815, "τ = %.2f μs" % tau, op, anchor="lm", strong=True, size=BIG)
        self.text(sc, 0.25, 0.815, "L = %s" % pow10(float(dens.prod()) * 1e4, 2) + "×10⁻⁴", op,
                  anchor="lm", size=0.033, ink=dark(TEAL, 0.3))
        scale = 1e4
        chart = LossChart(((0.55, 0.93), (0.40, 0.68)), (float(self.grid[0]),
                                                         float(self.grid[-1])), (0.0, 5.6))
        self.axes(sc, chart, op, "lifetime τ (μs)", "likelihood L (×10⁻⁴)",
                  xticks=[(v, "%d" % v) for v in range(1, 8)],
                  yticks=[(v, "%d" % v) for v in (0, 2, 4)])
        sweeping = u < 0.58
        upto = (tau - self.grid[0]) / (self.grid[-1] - self.grid[0]) if sweeping else 1.0
        self.path(sc, chart, self.grid, self.like * scale, TEAL, op, 3.2, go=upto)
        self.point(sc, chart, tau, float(dens.prod()) * scale, TEAL, op, 6.0, glow=0.4)
        late = op * smoothstep(0.74, 0.84, u)
        self.vmark(sc, chart, self.best, dark(TEAL, 0.3), late, top=self.like.max() * scale)
        self.vmark(sc, chart, self.meta["true_tau"], GREY, late)
        self.lines(sc, 0.93, 0.33, [("peak: τ = %.2f μs" % self.best, dark(TEAL, 0.3)),
                                     ("truth: τ = %.1f μs" % self.meta["true_tau"],
                                      dark(GREY, 0.2))], late, anchor="rm", size=0.031,
                   step=0.042)
        total = " + ".join("%.2f" % t for t in self.times)
        self.text(sc, 0.08, 0.872, "mean of the five times: (%s) / 5 = %.3f μs"
                  % (total, self.best), op * smoothstep(0.82, 0.92, u), anchor="lm",
                  strong=True, size=0.031)
        self.tag(sc, EXACT, op)

    def phase_4(self, sc, u, op):
        x = 0.06
        self.text(sc, x, 0.33, "take logs: the product becomes a sum", op, anchor="lm",
                  strong=True, size=HEAD)
        self.text(sc, x, 0.385, "at the peak, τ = %.3f μs:" % self.best, op, anchor="lm",
                  size=0.030)
        for i, t in enumerate(self.times):
            on = op * smoothstep(0.05 + 0.05 * i, 0.12 + 0.05 * i, u)
            self.box(sc, x + 0.008, 0.435 + 0.044 * i, 13, 13, EVENT_INK[i], on, radius=3)
            self.text(sc, x + 0.028, 0.435 + 0.044 * i, "ln f(%.2f) = %s"
                      % (t, signed(self.logs[2, i])), on, anchor="lm",
                      ink=dark(EVENT_INK[i], 0.3), size=0.032)
        self.text(sc, x, 0.672, "ln L = their sum = %s" % signed(self.log_sum[2]),
                  op * smoothstep(0.32, 0.4, u), anchor="lm", strong=True, size=0.034)
        late = op * smoothstep(0.45, 0.55, u)
        self.lines(sc, x, 0.745, [
            "with all 80 events of the run:",
            "L = 10^(%s),  ln L = %s" % (signed(self.lnl_all / math.log(10), 0),
                                        signed(self.lnl_all, 1)),
            "a sum of logs stays a number of ordinary size"], late, size=0.030, step=0.044)
        chart = LossChart(((0.59, 0.93), (0.40, 0.72)), (float(self.grid[0]),
                                                         float(self.grid[-1])), (-4.0, 0.25))
        self.axes(sc, chart, op, "lifetime τ (μs)", "ln L minus its maximum",
                  xticks=[(v, "%d" % v) for v in range(1, 8)],
                  yticks=[(v, signed(v, 0)) for v in (0, -1, -2, -3, -4)])
        inks = (TEAL, AMBER, MAGENTA)
        for k in range(3):
            go = smoothstep(0.1 + 0.25 * k, 0.4 + 0.25 * k, u)
            self.path(sc, chart, self.grid, self.more[k], inks[k], op, 3.0, go=go)
        self.vmark(sc, chart, self.meta["true_tau"], GREY, op)
        self.key(sc, 0.80, 0.435, [("%d events" % n, inks[k]) for k, n in enumerate(S.LC_MORE)],
                 op, line=True, step=0.044)
        self.text(sc, 0.93, 0.862, "peaks at %.2f, %.2f and %.2f μs; dashes: the truth, %.1f"
                  % (*self.more_best, self.meta["true_tau"]), op * smoothstep(0.8, 0.9, u),
                  anchor="rm", size=0.030)
        self.tag(sc, EXACT, op)

    def phase_5(self, sc, u, op):
        m = self.meta
        chart = LossChart(((0.08, 0.60), (0.40, 0.74)), (0.0, 6.0), (0.0, 0.5))
        self.axes(sc, chart, op, "best τ of one experiment (μs)", "share of experiments per μs",
                  xticks=[(v, "%d" % v) for v in range(7)],
                  yticks=[(v, "%g" % v) for v in (0, 0.2, 0.4)])
        n_dots = int(smoothstep(0.03, 0.32, u) * len(self.toy_first))
        grow = smoothstep(0.36, 0.6, u)
        self.stack(sc, self.toy_first, chart.fx, chart.y1, soft(BLUE, 0.15),
                   op * (1 - smoothstep(0.36, 0.5, u)), shown=n_dots)
        self.hist(sc, chart, self.toy_x, self.toy_hist, soft(BLUE, 0.3), op, grow=grow)
        self.path(sc, chart, self.toy_x, self.toy_exact, dark(BLUE, 0.35), op, 2.8,
                  go=smoothstep(0.6, 0.78, u))
        self.vmark(sc, chart, m["true_tau"], GREY, op)
        self.vmark(sc, chart, self.best, TEAL, op, dashed=False, top=0.1, width=3.0)
        self.text(sc, chart.fx(self.best) - 0.008, chart.fy(0.47), "ours: %.2f" % self.best, op,
                  anchor="rm", ink=dark(TEAL, 0.3), size=0.030)
        self.text(sc, chart.fx(m["true_tau"]) + 0.008, chart.fy(0.47),
                  "truth: %.1f" % m["true_tau"], op, anchor="lm", ink=dark(GREY, 0.2),
                  size=0.030)
        x = 0.66
        done = int(n_dots + grow * (m["toys"] - n_dots))
        self.text(sc, x, 0.385, "repeat the experiment", op, anchor="lm", strong=True,
                  size=HEAD)
        self.lines(sc, x, 0.44, ["five new decays each time,", "the best τ of each:",
                                 "experiments: %s" % commas(done)], op, size=0.031, step=0.046)
        late = op * smoothstep(0.66, 0.8, u)
        self.lines(sc, x, 0.60, [
            ("mean of best τ: %.3f μs" % self.toy_mean, dark(BLUE, 0.3)),
            ("spread: %.3f μs" % self.toy_sd, dark(BLUE, 0.3)),
            ("formula: τ/√5 = %.3f μs" % m["sd_exact"], STRONG)], late, size=0.031, step=0.046)
        self.key(sc, 0.62, 0.86, [("curve: the exact distribution", dark(BLUE, 0.35))], late,
                 line=True)
        self.text(sc, 0.08, 0.86, "bars: the pseudo-experiments", op * grow, anchor="lm",
                  ink=dark(BLUE, 0.2), size=0.030)
        self.tag(sc, "%s: %s × 5 decays, τ = %.1f μs, seed %d"
                 % (TOYS, commas(m["toys"]), m["true_tau"], S.LC_TOY_SEEDS[0]), op)

# ====================================================================== 129: intervals
HALF_INK, SYM_INK = TEAL, AMBER               # the half-unit interval; best +- sigma


class LikelihoodIntervals(Stats):
    """ln L of fifty decays against the parabola of the same curvature; the error read
    off where ln L has fallen by one half; the same with five decays, lopsided; thirty
    pseudo-experiments' intervals against the truth; and how often each kind of interval
    holds the truth, and on which side it misses, from 2 to 200 events."""

    TITLE = "Errors from the likelihood"
    PHASES = (("curvature", 11.0), ("half a unit", 10.0), ("few events", 12.0),
              ("does it cover", 11.0), ("against N", 10.0))
    Y_RANGE = (-2.6, 0.3)
    TAU_AXIS = 9.5

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        self.curves = {}
        for name, panel in (("many", s.panels[0]), ("few", s.panels[1])):
            ys = np.asarray(panel.data["ys"], float)
            self.curves[name] = (np.asarray(panel.data["x"], float), ys[0], ys[1])
        self.table = np.asarray(s.panels[2].data["values"], float)
        self.ns = [int(n) for n in m["events"]]
        self.many, self.few = m["many_row"], m["few_row"]
        self.shown = np.asarray(s.panels[2].data["shown"], float)
        self.intervals = [(self.shown * lo, self.shown * hi)
                          for lo, hi in S.interval_factors(m["few"])]
        tau = m["true_tau"]
        self.hits = [(a <= tau) & (tau <= b) for a, b in self.intervals]
        c = s.panels[3].data
        self.cov = np.asarray(c["ys"], float)                 # half, sym, exact half, exact sym
        self.cov_seeds = np.asarray(c["per_seed"], float)     # [seed, kind, N]
        self.miss = np.asarray(c["misses"], float)            # [kind, N, (below, above)]
        self.miss_exact = np.asarray(c["exact_misses"], float)
        self.i_few = self.ns.index(int(m["few"]))
        self.total = int(m["toys"]) * (len(m["seeds"]) - 1)

    def subtitle_for(self, k):
        m = self.meta
        best, minus, plus, sigma = self.many
        fbest, fminus, fplus, fsigma = self.few
        i = self.i_few
        return [
            "with %d decay times, ln L is close to a parabola around its maximum at "
            "τ = %.3f μs. A parabola's curvature is 1/σ²: here N/τ² = %.2f per μs², which "
            "gives σ = %.3f μs" % (m["many"], best, m["many"] / best ** 2, sigma),
            "on a parabola ln L has fallen by exactly ½ one σ from the peak, and by 2 at two "
            "σ. So read the error off the curve itself: within ½ of the maximum from %.2f to "
            "%.2f μs, that is %.2f −%.2f +%.2f" % (best - minus, best + plus, best, minus,
                                                   plus),
            "with only the first %d decays ln L is not a parabola: it falls slowly towards "
            "long lifetimes. The half-unit interval is lopsided, %.2f −%.2f +%.2f μs; the "
            "curvature alone says ±%.2f" % (m["few"], fbest, fminus, fplus, fsigma),
            "does the interval hold the truth 68%% of the time? In %s pseudo-experiments of "
            "%d decays (truth τ = %.1f μs) the half-unit interval contains it in %s, the "
            "symmetric one in %s" % (commas(self.total), m["few"], m["true_tau"],
                                     pct(self.cov[0, i]), pct(self.cov[1, i])),
            "both hold the truth about 68%% of the time at every N. They differ in where they "
            "miss: at N = %d the symmetric interval falls below the truth %s of the time and "
            "above it %s; the half-unit one %s and %s"
            % (m["few"], pct(self.miss[1, i, 0], 0), pct(self.miss[1, i, 1], 0),
               pct(self.miss[0, i, 0], 0), pct(self.miss[0, i, 1], 0)),
        ][k]

    # ------------------------------------------------------------------ the curve
    def _curve(self, sc, box, which, op, go=1.0, parabola=1.0, xticks=()):
        """ln L minus its maximum (and the parabola of the same curvature, drawn outward
        from the peak as far as fraction ``parabola``)."""
        grid, dlnl, par = self.curves[which]
        chart = LossChart(box, (float(grid[0]), float(grid[-1])), self.Y_RANGE)
        self.axes(sc, chart, op, "lifetime τ (μs)", "ln L minus its maximum",
                  xticks=[(v, "%g" % v) for v in xticks],
                  yticks=[(0, "0"), (-0.5, "−½"), (-1, "−1"), (-2, "−2")])
        if parabola > 0:
            best = (self.many if which == "many" else self.few)[0]
            near = np.abs(grid - best) <= parabola * (grid[-1] - grid[0])
            self.path(sc, chart, grid[near], par[near], soft(SYM_INK, 0.2), op, 6.0)
        self.path(sc, chart, grid, dlnl, HALF_INK, op, 2.8, go=go)
        if go >= 1:                                      # the maximum: the best lifetime
            self.point(sc, chart, grid[int(np.argmax(dlnl))], 0.0, dark(HALF_INK, 0.2), op, 5.0)
        return chart

    def _cross(self, which, level):
        """Where ln L minus its maximum equals ``level``, left and right of the peak."""
        grid, dlnl, _ = self.curves[which]
        top = int(np.argmax(dlnl))
        return (float(np.interp(level, dlnl[:top + 1], grid[:top + 1])),
                float(np.interp(level, dlnl[top:][::-1], grid[top:][::-1])))

    def _drop(self, sc, chart, x, level, ink, op, width=2.4):
        fx = chart.fx(x)
        self.polyline(sc, [self.P(fx, chart.fy(level), lift=0.9), self.P(fx, chart.y1, lift=0.9)],
                      width, ink, op)
        self.disk(sc, self.P(fx, chart.fy(level)), 5.0, ink, op, lift=1.6, outline=0.0)

    def phase_1(self, sc, u, op):
        m = self.meta
        best, _minus, _plus, sigma = self.many
        chart = self._curve(sc, ((0.08, 0.53), (0.40, 0.76)), "many", op,
                            go=smoothstep(0.03, 0.3, u), parabola=smoothstep(0.38, 0.7, u),
                            xticks=(2, 2.5, 3, 3.5))
        self.vmark(sc, chart, best, GREY, op * smoothstep(0.25, 0.35, u))
        self.key(sc, 0.10, 0.872, [("ln L of the %d decays" % m["many"], HALF_INK)], op,
                 line=True)
        self.key(sc, 0.36, 0.872, [("a parabola", SYM_INK)], op * smoothstep(0.38, 0.5, u),
                 line=True)
        x = 0.60
        self.text(sc, x, 0.33, "%d decay times" % m["many"], op, anchor="lm", strong=True,
                  size=BIG)
        self.lines(sc, x, 0.39, ["ln L peaks at the best τ:", "the mean, %.3f μs" % best], op,
                   size=0.031, step=0.045)
        mid = op * smoothstep(0.38, 0.5, u)
        self.lines(sc, x, 0.505, ["near the peak it is a parabola:",
                                  "ln L ≈ max − (τ − best)² / (2σ²)"], mid, size=0.031,
                   step=0.045)
        late = op * smoothstep(0.62, 0.74, u)
        self.lines(sc, x, 0.62, [
            "its curvature is 1/σ². Here",
            "1/σ² = N / τ² = %d / %.3f²" % (m["many"], best),
            "      = %.2f per μs²" % (m["many"] / best ** 2)], late, size=0.031, step=0.045)
        self.text(sc, x, 0.775, "σ = 1/√%.2f = %.3f μs" % (m["many"] / best ** 2, sigma),
                  op * smoothstep(0.8, 0.9, u), anchor="lm", strong=True, size=0.034)
        self.tag(sc, EXACT, op)

    def phase_2(self, sc, u, op):
        best, minus, plus, sigma = self.many
        chart = self._curve(sc, ((0.08, 0.53), (0.40, 0.76)), "many", op,
                            xticks=(2, 2.5, 3, 3.5))
        level = -0.5 * smoothstep(0.06, 0.42, u)
        a, b = self._cross("many", level)
        y = chart.fy(level)
        self.dash(sc, (chart.x0, y), (chart.x1, y), DIGIT_INK, op, 1.8, 18)
        if level < -0.01:
            self._drop(sc, chart, a, level, HALF_INK, op)
            self._drop(sc, chart, b, level, HALF_INK, op)
            self.polyline(sc, [self.P(chart.fx(a), chart.y1, lift=1.0),
                               self.P(chart.fx(b), chart.y1, lift=1.0)], 6.0, HALF_INK, op)
        two = op * smoothstep(0.72, 0.84, u)
        a2, b2 = self._cross("many", -2.0)
        self.dash(sc, (chart.x0, chart.fy(-2.0)), (chart.x1, chart.fy(-2.0)), GREY, two, 1.6, 18)
        self._drop(sc, chart, a2, -2.0, GREY, two, 1.8)
        self._drop(sc, chart, b2, -2.0, GREY, two, 1.8)
        self.text(sc, chart.fx(best), 0.872, "ln L within %.2f of its maximum: τ from %.2f to "
                  "%.2f μs" % (-level, a, b), op, size=0.031, strong=True)
        x = 0.60
        self.text(sc, x, 0.33, "where has ln L fallen by ½?", op, anchor="lm", strong=True,
                  size=HEAD)
        self.lines(sc, x, 0.39, ["on the parabola, at τ = best ± σ:",
                                 "(τ − best)² / (2σ²) = σ² / (2σ²) = ½"], op, size=0.031,
                   step=0.045)
        mid = op * smoothstep(0.42, 0.52, u)
        self.lines(sc, x, 0.505, ["so the half-unit drop marks", "the one-σ interval:"], mid,
                   size=0.031, step=0.045)
        self.text(sc, x, 0.61, "τ = %.2f  −%.2f  +%.2f μs" % (best, minus, plus), mid,
                  anchor="lm", ink=dark(HALF_INK, 0.35), size=0.034)
        self.text(sc, x, 0.66, "curvature:  ± %.2f μs" % sigma, mid, anchor="lm",
                  ink=dark(SYM_INK, 0.35), size=0.034)
        self.lines(sc, x, 0.735, ["a drop of 2 marks two σ:",
                                  "%.2f to %.2f μs" % (a2, b2)], two, size=0.031, step=0.045)
        self.tag(sc, EXACT, op)

    def phase_3(self, sc, u, op):
        m = self.meta
        best, minus, plus, sigma = self.few
        chart = self._curve(sc, ((0.08, 0.50), (0.40, 0.76)), "few", op,
                            go=smoothstep(0.03, 0.25, u), parabola=smoothstep(0.25, 0.45, u),
                            xticks=(2, 4, 6, 8, 10))
        marks = op * smoothstep(0.45, 0.55, u)
        self.dash(sc, (chart.x0, chart.fy(-0.5)), (chart.x1, chart.fy(-0.5)), DIGIT_INK, marks,
                  1.8, 18)
        for x in (best - minus, best + plus):
            self._drop(sc, chart, x, -0.5, HALF_INK, marks)
        for x in (best - sigma, best + sigma):
            self._drop(sc, chart, x, -0.5, SYM_INK, marks, 1.8)
        self.vmark(sc, chart, best, GREY, op * smoothstep(0.2, 0.3, u))
        self.key(sc, 0.10, 0.872, [("ln L of the %d decays" % m["few"], HALF_INK)], op,
                 line=True)
        self.key(sc, 0.34, 0.872, [("parabola", SYM_INK)], op * smoothstep(0.25, 0.4, u),
                 line=True)
        x = 0.565
        self.text(sc, x, 0.33, "the first %d decays only" % m["few"], op, anchor="lm",
                  strong=True, size=BIG)
        self.text(sc, x, 0.385, "best τ = %.3f μs" % best, op, anchor="lm", size=0.031)
        self.text(sc, x, 0.432, "half-unit drop:  −%.2f  +%.2f μs" % (minus, plus), marks,
                  anchor="lm", ink=dark(HALF_INK, 0.35), size=0.033)
        self.text(sc, x, 0.478, "curvature:  ± %.2f μs" % sigma, marks, anchor="lm",
                  ink=dark(SYM_INK, 0.35), size=0.033)
        # the same run stopped after N decays
        cols = (x + 0.035, x + 0.125, x + 0.215, x + 0.30, x + 0.385)
        ty = 0.565
        table = op * smoothstep(0.58, 0.66, u)
        self.text(sc, x, ty - 0.045, "the same run, read after N decays:", table, anchor="lm",
                  size=0.030)
        heads = (("N", STRONG), ("down", dark(HALF_INK, 0.35)), ("up", dark(HALF_INK, 0.35)),
                 ("± σ", dark(SYM_INK, 0.35)), ("up ÷ down", STRONG))
        for cx, (name, ink) in zip(cols, heads):
            self.text(sc, cx, ty, name, table, ink=ink, size=0.030)
        for r, n in enumerate((5, 20, 50, 200)):
            _b, lo, hi, sg = self.table[self.ns.index(n)]
            on = op * smoothstep(0.62 + 0.07 * r, 0.69 + 0.07 * r, u)
            y = ty + 0.048 * (r + 1)
            digits = 2 if n < 100 else 3
            for cx, text, ink in zip(cols, ("%d" % n, "−%.*f" % (digits, lo),
                                            "+%.*f" % (digits, hi), "%.*f" % (digits, sg),
                                            "%.2f" % (hi / lo)),
                                     [h[1] for h in heads]):
                self.text(sc, cx, y, text, on, ink=ink, size=0.031)
        self.tag(sc, EXACT, op)

    def _stack(self, sc, kind, x0, x1, op, u, name, ink):
        """Thirty pseudo-experiments' intervals as rows, against the true lifetime."""
        m = self.meta
        lo, hi = self.intervals[kind]
        rows = len(self.shown)
        n = int(np.clip((u - 0.04) / 0.55, 0, 1) * rows + 1e-9)

        def at(v):
            return x0 + (x1 - x0) * min(float(v), self.TAU_AXIS) / self.TAU_AXIS
        base = 0.795
        self.polyline(sc, [self.P(x0, base), self.P(x1, base)], 1.2, AXIS, 0.8 * op)
        for v in (0, 2, 4, 6, 8):
            self.polyline(sc, [self.P(at(v), base), self.P(at(v), base + 0.01)], 1.2, AXIS,
                          0.8 * op)
            self.text(sc, at(v), base + 0.032, "%d" % v, op, size=0.029)
        self.text(sc, x1 + 0.012, base + 0.032, "τ (μs)", op, size=0.029, anchor="rm")
        self.dash(sc, (at(m["true_tau"]), 0.395), (at(m["true_tau"]), base), DIGIT_INK, op, 2.0,
                  16)
        for r in range(n):
            y = 0.408 + 0.0125 * r
            col = ink if self.hits[kind][r] else RED
            self.polyline(sc, [self.P(at(lo[r]), y), self.P(at(hi[r]), y)], 3.4,
                          col if self.hits[kind][r] else RED, op)
            self.disk(sc, self.P(at(self.shown[r]), y), 3.2, dark(col, 0.25), op, lift=1.4,
                      outline=0.0)
        self.text(sc, x0, 0.325, name, op, anchor="lm", ink=dark(ink, 0.35), size=0.033)
        held = int(self.hits[kind][:n].sum())
        self.text(sc, x0, 0.368, "%d experiments: %d hold the truth, %d miss (red)"
                  % (n, held, n - held), op, anchor="lm", size=0.030)
        self.text(sc, x0, 0.868, "of all %s: %s hold the truth"
                  % (commas(self.total), pct(self.cov[kind, self.i_few])),
                  op * smoothstep(0.66, 0.78, u), anchor="lm", strong=True, size=0.031)

    def phase_4(self, sc, u, op):
        m = self.meta
        self._stack(sc, 0, 0.06, 0.46, op, u, "ln L within ½ of its maximum", HALF_INK)
        self._stack(sc, 1, 0.55, 0.95, op, u, "best τ ± σ, from the curvature", SYM_INK)
        self.tag(sc, "%s: %d decays each, truth τ = %.1f μs (dashes), seed %d"
                 % (TOYS, m["few"], m["true_tau"], S.LI_SEEDS[0]), op)

    def phase_5(self, sc, u, op):
        m = self.meta
        ns = np.asarray(self.ns, float)
        ticks = [(v, "%d" % v) for v in (2, 5, 20, 50, 200)]
        go = (u - 0.04) / 0.4
        left = LossChart(((0.08, 0.45), (0.43, 0.79)), (2.0, 200.0), (0.652, 0.692), xlog=True)
        self.axes(sc, left, op, "decays per experiment, N", "share holding the truth "
                  "(dashes: 68.3%)", xticks=ticks,
                  yticks=[(v, "%d%%" % round(100 * v)) for v in (0.66, 0.67, 0.68, 0.69)])
        self.hmark(sc, left, m["one_sigma"], DIGIT_INK, op)
        right = LossChart(((0.58, 0.95), (0.43, 0.79)), (2.0, 200.0), (0.0, 0.36), xlog=True)
        self.axes(sc, right, op, "decays per experiment, N", "share missing on each side "
                  "(dashes: 15.9%)", xticks=ticks,
                  yticks=[(v, "%d%%" % round(100 * v)) for v in (0, 0.1, 0.2, 0.3)])
        self.hmark(sc, right, (1 - m["one_sigma"]) / 2, DIGIT_INK, op)
        for kind, ink in ((0, HALF_INK), (1, SYM_INK)):
            progress(self, sc, left, ns, self.cov[2 + kind], ink, op, go, 2.6, xlog=True,
                     dot=False)
            for side, width in ((0, 2.8), (1, 1.6)):
                progress(self, sc, right, ns, np.maximum(self.miss_exact[kind, :, side], 1e-4),
                         ink, op, go, width, xlog=True, dot=False)
            shown = int(np.clip(go, 0, 1) * len(ns) + 1e-9)
            for j in range(shown):
                lo, hi = self.cov_seeds[:, kind, j].min(), self.cov_seeds[:, kind, j].max()
                fx = left.fx(ns[j]) + (0.004 if kind else -0.004)
                self.polyline(sc, [self.P(fx, left.fy(lo)), self.P(fx, left.fy(hi))], 1.6,
                              dark(ink, 0.2), op)
                self.disk(sc, self.P(fx, left.fy(self.cov[kind, j])), 4.2, ink, op, lift=1.5,
                          outline=0.15)
                for side in (0, 1):
                    self.point(sc, right, ns[j], self.miss[kind, j, side], ink, op, 3.6)
        self.key(sc, 0.09, 0.332, [("ln L within ½", HALF_INK)], op)
        self.key(sc, 0.27, 0.332, [("best τ ± σ", SYM_INK)], op)
        self.text(sc, 0.95, 0.332, "thick: wholly below the truth;  thin: above", op,
                  anchor="rm", size=0.029)
        self.tag(sc, "%s (dots: %s per seed; bars: range of 3 seeds) against exact values "
                 "(lines)" % (TOYS, commas(m["toys"])), op)

# ====================================================================== 130: propagation
BAND_INK = (BLUE, TEAL, GREEN, AMBER, RED)    # f from far below its centre to far above
FORMULA_INK, CLOUD_INK = AMBER, BLUE


def band_colours(f, centre, sigma):
    """A colour per value: which of five bands of width ``sigma`` around ``centre``."""
    k = np.clip(np.floor((np.asarray(f, float) - centre) / sigma + 2.5), 0, 4).astype(int)
    return np.asarray(BAND_INK, np.float32)[k]


class ErrorPropagation(Stats):
    """A measured y carried through 10 / y along the curve and its tangent; a cloud of
    (x, y) pushed through x / y against the formula's two terms; the cloud turning as
    the correlation changes, with and without the third term; a denominator known to
    30%, where the curve is no straight line; and formula against cloud as that error
    grows."""

    TITLE = "Error propagation by cloud"
    PHASES = (("one number", 11.0), ("two numbers", 11.0), ("correlation", 11.0),
              ("a curve", 11.0), ("its limits", 10.0))
    XR, YR = (8.2, 11.8), (4.05, 5.95)
    CARRIED = 40

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        self.z = np.asarray(s.panels[0].data["z"], float)
        self.rhos = np.asarray(s.panels[1].data["x"], float)
        self.rho_cloud, self.rho_with, self.rho_without = np.asarray(s.panels[1].data["ys"],
                                                                     float)
        self.wide_x = np.asarray(s.panels[2].data["x"], float)
        self.wide_hist, self.wide_formula = np.asarray(s.panels[2].data["ys"], float)
        self.gentle_x = np.asarray(s.panels[2].data["gentle_x"], float)
        self.gentle_hist = np.asarray(s.panels[2].data["gentle_hist"], float)
        self.rels = np.asarray(s.panels[3].data["x"], float)
        self.rel_formula, self.rel_minus, self.rel_plus = np.asarray(s.panels[3].data["ys"],
                                                                     float)
        self.sigma0 = float(m["two_variables"]["formula"])
        self.f0 = float(m["f0"])
        self.ys1 = S.EP_Y + S.EP_SY * self.z[:self.CARRIED, 1]      # the y values carried

    def subtitle_for(self, k):
        m = self.meta
        one, two, cor, wide = (m[k] for k in ("one_variable", "two_variables", "correlated",
                                              "wide"))
        j = list(S.EP_RELS).index(0.10)
        return [
            "a measured y = %.2f ± %.2f, pushed through f = 10/y. Over so small a range the "
            "curve is nearly its tangent, so the spread of f is the slope times σ: 0.40 × "
            "0.25 = %.3f. A cloud of %s draws gives %.3f"
            % (S.EP_Y, S.EP_SY, one["formula"], commas(m["draws"]), one["cloud"]),
            "now x = %.1f ± %.1f is measured too, independently, and f = x/y. The formula "
            "adds the two contributions in quadrature: σ = %.3f. The cloud of %s pairs "
            "spreads by %.3f" % (S.EP_X, S.EP_SX, two["formula"], commas(m["draws"]),
                                 two["cloud"]),
            "if x and y are correlated the formula has a third term. At ρ = %.1f it gives "
            "%.3f, and the cloud %.3f. Without the term: %.3f, %d%% too large. Errors that "
            "rise together cancel in a ratio"
            % (m["rho"], cor["formula"], cor["cloud"], cor["without_term"],
               round(100 * (cor["without_term"] / cor["cloud"] - 1))),
            "now y is known only to 30%%: %.1f ± %.1f. Over that range 1/y is no straight "
            "line. The formula says %.2f ± %.2f; the middle 68%% of the cloud runs from "
            "%.2f to %.2f, that is −%.2f +%.2f"
            % (S.EP_Y, wide["sigma_y"], self.f0, wide["formula"],
               wide["median"] - wide["minus"], wide["median"] + wide["plus"], wide["minus"],
               wide["plus"]),
            "formula against cloud as the denominator's error grows. At 2%% they agree; at "
            "10%% the cloud already reaches %d%% further up than down; at 30%% nearly "
            "twice as far. The linear formula is for small errors"
            % round(100 * (self.rel_plus[j] / self.rel_minus[j] - 1)),
        ][k]

    # ------------------------------------------------------------------ one variable
    def _curve_1d(self, sc, box, yr, fr, op, tangent=1.0, yticks=(), xticks=()):
        """f = 10 / y over a range of y, with its tangent at y = 5."""
        chart = LossChart(box, yr, fr)
        self.axes(sc, chart, op, "measured y", "f = 10 / y",
                  xticks=[(v, "%g" % v) for v in xticks], yticks=[(v, "%g" % v) for v in yticks])
        ys = np.linspace(yr[0], yr[1], 80)
        slope = -S.EP_X / S.EP_Y ** 2
        if tangent > 0:
            half = 0.5 * (yr[1] - yr[0]) * tangent
            ty = np.linspace(S.EP_Y - half, S.EP_Y + half, 30)
            self.path(sc, chart, ty, self.f0 + slope * (ty - S.EP_Y), FORMULA_INK, op, 2.4)
        self.path(sc, chart, ys, S.EP_X / ys, DIGIT_INK, op, 3.0)
        return chart

    def phase_1(self, sc, u, op):
        m = self.meta
        chart = self._curve_1d(sc, ((0.12, 0.55), (0.40, 0.76)), (4.2, 5.8), (1.68, 2.42), op,
                               tangent=smoothstep(0.5, 0.66, u), yticks=(1.8, 2.0, 2.2, 2.4),
                               xticks=(4.5, 5.0, 5.5))
        n = len(self.ys1)
        for i, y in enumerate(self.ys1):                 # each value: up to the curve, across
            a = 0.04 + 0.42 * i / n
            up, over = smoothstep(a, a + 0.06, u), smoothstep(a + 0.06, a + 0.12, u)
            if up <= 0:
                continue
            f = S.EP_X / y
            fx, fy = chart.fx(y), chart.fy(f)
            ink = soft(CLOUD_INK, 0.1)
            self.disk(sc, self.P(fx, chart.y1), 3.4, ink, op, lift=1.2, outline=0.1)
            if over < 1:
                px = fx + (chart.x0 - fx) * over
                py = chart.y1 + (fy - chart.y1) * up
                trail = [self.P(fx, chart.y1), self.P(fx, py)] + (
                    [self.P(px, fy)] if over > 0 else [])
                self.polyline(sc, trail, 1.4, CLOUD_INK, 0.7 * op)
                self.disk(sc, self.P(px, py), 4.6, CLOUD_INK, op, lift=1.6, outline=0.0,
                          glow=0.4)
            else:
                self.disk(sc, self.P(chart.x0, fy), 3.4, ink, op, lift=1.2, outline=0.1)
        band = op * smoothstep(0.52, 0.64, u)
        for sign in (-1, 1):                             # y +- sigma and where it lands
            y = S.EP_Y + sign * S.EP_SY
            f = S.EP_X / y
            self.dash(sc, (chart.fx(y), chart.y1), (chart.fx(y), chart.fy(f)), GREY, band, 1.6,
                      8)
            self.dash(sc, (chart.fx(y), chart.fy(f)), (chart.x0, chart.fy(f)), GREY, band, 1.6,
                      8)
        x = 0.62
        self.text(sc, x, 0.33, "one measured number", op, anchor="lm", strong=True, size=BIG)
        self.lines(sc, x, 0.39, ["y = %.2f ± %.2f  (σ: its error)" % (S.EP_Y, S.EP_SY),
                                 "f = 10 / y:  each y goes up to",
                                 "the curve and across to f"], op, size=0.031, step=0.045)
        mid = op * smoothstep(0.52, 0.64, u)
        self.lines(sc, x, 0.55, [("the tangent at y = 5 has slope", dark(FORMULA_INK, 0.35)),
                                 ("df/dy = −10 / y² = −0.40", dark(FORMULA_INK, 0.35))], mid,
                   size=0.031, step=0.045)
        late = op * smoothstep(0.7, 0.8, u)
        self.text(sc, x, 0.665, "σf = |df/dy| σ = 0.40 × 0.25", late, anchor="lm",
                  size=0.032, strong=True)
        self.text(sc, x, 0.712, "     = %.3f" % m["one_variable"]["formula"], late, anchor="lm",
                  size=0.032, strong=True)
        self.text(sc, x, 0.775, "cloud of %s: spread %.3f"
                  % (commas(m["draws"]), m["one_variable"]["cloud"]),
                  op * smoothstep(0.84, 0.92, u), anchor="lm", ink=dark(CLOUD_INK, 0.3),
                  size=0.032)
        self.tag(sc, "%s (the formula) and %s: %s draws, %d seeds"
                 % (EXACT, TOYS, commas(m["draws"]), len(m["seeds"])), op)

    # ------------------------------------------------------------------ two variables
    def _cloud(self, sc, box, rho, op, shown=None, sy=None):
        """The cloud of (x, y) pairs, coloured by the band of f = x / y each falls in,
        over rays of constant f."""
        x, y = S.pair(self.z if shown is None else self.z[:shown], S.EP_SX,
                      S.EP_SY if sy is None else sy, rho)
        chart = LossChart(box, self.XR, self.YR)
        w, h = self.span(box[0][1] - box[0][0], box[1][1] - box[1][0])
        tex = raster(x, y, self.XR, self.YR, (int(h / 1.5), int(w / 1.5)),
                     band_colours(x / y, self.f0, self.sigma0), fat=2)
        self.picture(sc, tex, box, op)
        xs = np.linspace(self.XR[0], self.XR[1], 40)
        for k in range(-2, 3):                           # rays of constant f
            f = self.f0 + (k - 0.5) * self.sigma0 if k > -2 else None
            if f is not None:
                self.path(sc, chart, xs, xs / f, DIGIT_INK, 0.7 * op, 1.3, lift=0.9)
        self.axes(sc, chart, op, "x", "y", xticks=[(v, "%g" % v) for v in (9, 10, 11)],
                  yticks=[(v, "%g" % v) for v in (4.5, 5.0, 5.5)], grid=False)
        return chart

    def _spread(self, sc, box, op, grow, x=None, hist=None, rng=(1.4, 2.6), top=3.4,
                sigma=None, ticks=(1.6, 2.0, 2.4)):
        """The cloud's values of f as a histogram, with the formula's bell curve."""
        x = self.gentle_x if x is None else x
        hist = self.gentle_hist if hist is None else hist
        chart = LossChart(box, rng, (0.0, top))
        w = self.span(abs(chart.fx(float(x[1])) - chart.fx(float(x[0]))) * 0.86)[0]
        for mid, v in zip(x, hist):
            hgt = chart.y1 - chart.fy(float(v) * grow)
            if hgt * self.cam.height < 0.7:
                continue
            ink = band_colours([mid], self.f0, self.sigma0)[0]
            self.rect(sc, self.P(chart.fx(float(mid)), chart.y1 - hgt / 2), w,
                      self.span(0, hgt)[1], ink, op, radius=1.0, lift=0.3)
        self.axes(sc, chart, op, "f = x / y", "", xticks=[(v, "%g" % v) for v in ticks],
                  yticks=[], grid=False)
        if sigma:
            fs = np.linspace(rng[0], rng[1], 120)
            self.path(sc, chart, fs, S.normal_pdf(fs, self.f0, sigma), DIGIT_INK, op, 2.6)
        return chart

    def phase_2(self, sc, u, op):
        m = self.meta
        shown = int(smoothstep(0.03, 0.4, u) * len(self.z))
        self._cloud(sc, ((0.09, 0.43), (0.42, 0.78)), 0.0, op, shown=max(shown, 1))
        self.text(sc, 0.09, 0.33, "x = %.1f ± %.1f,  y = %.2f ± %.2f" % (
            S.EP_X, S.EP_SX, S.EP_Y, S.EP_SY), op, anchor="lm", strong=True, size=HEAD)
        self.text(sc, 0.26, 0.885, "lines: f = x / y constant;  colour: the band of f",
                  op, size=0.029)
        self._spread(sc, ((0.56, 0.94), (0.38, 0.56)), op, smoothstep(0.03, 0.4, u),
                     sigma=self.sigma0 if u > 0.72 else None)
        self.text(sc, 0.94, 0.345, "the cloud's f: %s pairs" % commas(m["draws"]), op,
                  anchor="rm", size=0.030)
        x = 0.53
        mid = op * smoothstep(0.42, 0.52, u)
        self.lines(sc, x, 0.675, ["σf² = (∂f/∂x)² σx² + (∂f/∂y)² σy²",
                                  "∂f/∂x = 1/y = 0.20;   ∂f/∂y = −x/y² = −0.40"], mid,
                   size=0.031, step=0.045)
        late = op * smoothstep(0.58, 0.68, u)
        self.text(sc, x, 0.775, "σf² = (0.20 × 0.5)² + (0.40 × 0.25)² = 0.01 + 0.01", late,
                  anchor="lm", size=0.031)
        last = op * smoothstep(0.72, 0.82, u)
        self.text(sc, x, 0.83, "formula: σf = %.3f" % m["two_variables"]["formula"], last,
                  anchor="lm", strong=True, size=0.033)
        self.text(sc, x + 0.225, 0.83, "cloud: %.3f" % m["two_variables"]["cloud"], last,
                  anchor="lm", ink=dark(CLOUD_INK, 0.3), size=0.033)
        self.tag(sc, "%s: Gaussian x and y, %s pairs, %d seeds"
                 % (TOYS, commas(m["draws"]), len(m["seeds"])), op)

    def _rho(self, u):
        a = 0.9 * smoothstep(0.04, 0.22, u)
        b = -1.8 * smoothstep(0.26, 0.54, u)
        return a + b + (0.9 + self.meta["rho"]) * smoothstep(0.58, 0.72, u)

    def phase_3(self, sc, u, op):
        m = self.meta
        rho = self._rho(u)
        self._cloud(sc, ((0.09, 0.43), (0.42, 0.78)), rho, op)
        self.text(sc, 0.09, 0.33, "correlation ρ = %s" % signed(rho, 2).replace("−0.00", "0.00"),
                  op, anchor="lm", strong=True, size=BIG)
        self.text(sc, 0.26, 0.885, "the same %s points at every ρ" % commas(len(self.z)), op,
                  size=0.029)
        chart = LossChart(((0.57, 0.94), (0.40, 0.66)), (-0.95, 0.95), (0.0, 0.22))
        self.axes(sc, chart, op, "correlation ρ of x and y", "spread of f = x / y",
                  xticks=[(v, signed(v, 1)) for v in (-0.6, 0, 0.6)],
                  yticks=[(v, "%g" % v) for v in (0, 0.1, 0.2)])
        self.path(sc, chart, self.rhos, self.rho_without, GREY, op, 2.2)
        self.path(sc, chart, self.rhos, self.rho_with, FORMULA_INK, op, 3.0)
        for r, v in zip(self.rhos, self.rho_cloud):
            self.point(sc, chart, r, v, CLOUD_INK, op, 4.4)
        now = S.ratio_sigma(S.EP_SX, S.EP_SY, rho)
        self.vmark(sc, chart, rho, DIGIT_INK, op, top=now)
        self.point(sc, chart, rho, now, FORMULA_INK, op, 6.0, glow=0.4)
        self.key(sc, 0.58, 0.782, [("formula with the ρ term", FORMULA_INK),
                                   ("formula without it", GREY)], op, line=True, step=0.04)
        self.key(sc, 0.82, 0.782, [("the cloud", CLOUD_INK)], op)
        late = op * smoothstep(0.76, 0.86, u)
        self.text(sc, 0.95, 0.872, "third term, at ρ = 0.6:  2ρ (∂f/∂x)(∂f/∂y) σx σy = −0.012",
                  late, anchor="rm", size=0.029, strong=True)
        self.tag(sc, "%s: %s pairs per ρ, %d seeds" % (TOYS, commas(m["draws"]),
                                                      len(m["seeds"])), op)

    def phase_4(self, sc, u, op):
        m = self.meta
        wide = m["wide"]
        sy = wide["sigma_y"]
        chart = self._curve_1d(sc, ((0.09, 0.45), (0.43, 0.76)), (2.6, 8.4), (0.9, 3.6), op,
                               tangent=smoothstep(0.1, 0.25, u), yticks=(1, 2, 3),
                               xticks=(3.5, 5.0, 6.5))
        slope = -S.EP_X / S.EP_Y ** 2
        mark = op * smoothstep(0.25, 0.4, u)
        for sign in (-1, 1):
            y = S.EP_Y + sign * sy
            f, lin = S.EP_X / y, self.f0 + slope * sign * sy
            self.dash(sc, (chart.fx(y), chart.y1), (chart.fx(y), chart.fy(max(f, lin))), GREY,
                      mark, 1.6, 10)
            self.polyline(sc, [self.P(chart.fx(y), chart.fy(f), lift=0.9),
                               self.P(chart.x0, chart.fy(f), lift=0.9)], 2.4, CLOUD_INK, mark)
            self.polyline(sc, [self.P(chart.fx(y), chart.fy(lin), lift=0.9),
                               self.P(chart.x0, chart.fy(lin), lift=0.9)], 1.8, FORMULA_INK,
                          mark)
            self.point(sc, chart, y, f, CLOUD_INK, mark, 5.0)
            self.point(sc, chart, y, lin, FORMULA_INK, mark, 4.4)
        self.text(sc, 0.09, 0.33, "y = %.1f ± %.1f:  the same curve, a wider range"
                  % (S.EP_Y, sy), op, anchor="lm", strong=True, size=HEAD)
        self.lines(sc, 0.09, 0.868, [
            ("curve:  10/6.5 = %.2f,  10/3.5 = %.2f" % (10 / 6.5, 10 / 3.5),
             dark(CLOUD_INK, 0.3)),
            ("tangent:  2 − 0.6 = 1.40,  2 + 0.6 = 2.60", dark(FORMULA_INK, 0.35))], mark,
            size=0.030, step=0.042)
        grow = smoothstep(0.42, 0.62, u)
        right = LossChart(((0.56, 0.94), (0.43, 0.63)), (0.5, 5.0), (0.0, 0.95))
        self.hist(sc, right, self.wide_x, self.wide_hist, soft(CLOUD_INK, 0.25), op, grow=grow)
        self.path(sc, right, self.wide_x, self.wide_formula, FORMULA_INK, op, 3.0,
                  go=smoothstep(0.6, 0.72, u))
        self.axes(sc, right, op, "f = x / y", "", xticks=[(v, "%d" % v) for v in (1, 2, 3, 4, 5)],
                  yticks=[], grid=False)
        self.text(sc, 0.94, 0.395, "x / y for %s pairs (bars)" % commas(m["draws"]), op,
                  anchor="rm", size=0.030)
        late = op * smoothstep(0.7, 0.8, u)
        x = 0.56
        self.text(sc, x, 0.735, "formula:  %.2f ± %.2f" % (self.f0, wide["formula"]), late,
                  anchor="lm", ink=dark(FORMULA_INK, 0.35), size=0.033)
        self.text(sc, x, 0.782, "cloud:  %.2f  −%.2f  +%.2f" % (wide["median"], wide["minus"],
                                                              wide["plus"]), late,
                  anchor="lm", ink=dark(CLOUD_INK, 0.3), size=0.033)
        sd = wide["sd_per_seed"]
        self.text(sc, x, 0.84, "its standard deviation: %.0f to %.0f over %d seeds;"
                  % (min(sd), max(sd), len(sd)), op * smoothstep(0.84, 0.92, u), anchor="lm",
                  size=0.029)
        self.text(sc, x, 0.878, "it never settles, since y can land near 0",
                  op * smoothstep(0.84, 0.92, u), anchor="lm", size=0.029)
        self.tag(sc, "%s (left) and %s: %s pairs" % (EXACT, TOYS, commas(m["draws"])), op)

    def phase_5(self, sc, u, op):
        m = self.meta
        pc = 100 * self.rels
        go = (u - 0.04) / 0.45
        chart = LossChart(((0.09, 0.50), (0.47, 0.78)), (0.0, 31.0), (0.0, 0.92))
        self.axes(sc, chart, op, "relative error of y (%)", "error on f = x / y",
                  xticks=[(v, "%d" % v) for v in (2, 10, 20, 30)],
                  yticks=[(v, "%g" % v) for v in (0, 0.2, 0.4, 0.6, 0.8)])
        for ys, ink, width in ((self.rel_formula, FORMULA_INK, 3.0),
                               (self.rel_minus, CLOUD_INK, 2.6), (self.rel_plus, RED, 2.6)):
            progress(self, sc, chart, pc, ys, ink, op, go, width, dot=False)
            for j in range(int(np.clip(go, 0, 1) * len(pc) + 1e-9)):
                self.point(sc, chart, pc[j], ys[j], ink, op, 4.2)
        self.key(sc, 0.11, 0.335, [("formula", FORMULA_INK)], op, line=True)
        self.key(sc, 0.25, 0.335, [("cloud: 68% upwards", RED)], op, line=True)
        self.key(sc, 0.25, 0.378, [("cloud: 68% downwards", CLOUD_INK)], op, line=True)
        # the error bars themselves, to scale
        base, unit = 0.60, 0.21                          # layout y of f = 2; layout per unit f
        self.text(sc, 0.76, 0.335, "the error bar on f = 2, to scale", op, strong=True,
                  size=0.031)
        for n, rel in enumerate((0.05, 0.10, 0.20, 0.30)):
            j = list(S.EP_RELS).index(rel)
            on = op * smoothstep(0.3 + 0.1 * n, 0.4 + 0.1 * n, u)
            cx = 0.61 + 0.10 * n
            f, lo, hi = self.rel_formula[j], self.rel_minus[j], self.rel_plus[j]
            self.polyline(sc, [self.P(cx - 0.014, base - unit * f),
                               self.P(cx - 0.014, base + unit * f)], 4.0, FORMULA_INK, on)
            self.polyline(sc, [self.P(cx + 0.014, base), self.P(cx + 0.014, base - unit * hi)],
                          4.0, RED, on)
            self.polyline(sc, [self.P(cx + 0.014, base), self.P(cx + 0.014, base + unit * lo)],
                          4.0, CLOUD_INK, on)
            self.disk(sc, self.P(cx + 0.014, base), 4.0, DIGIT_INK, on, lift=1.5, outline=0.0)
            self.disk(sc, self.P(cx - 0.014, base), 4.0, DIGIT_INK, on, lift=1.5, outline=0.0)
            self.text(sc, cx, 0.765, "y to %d%%" % round(100 * rel), on, size=0.030, strong=True)
            self.text(sc, cx, 0.807, "± %.2f" % f, on, size=0.029, ink=dark(FORMULA_INK, 0.35))
            self.text(sc, cx, 0.846, "+%.2f" % hi, on, size=0.029, ink=dark(RED, 0.3))
            self.text(sc, cx, 0.885, "−%.2f" % lo, on, size=0.029, ink=dark(CLOUD_INK, 0.3))
        self.tag(sc, "%s: %s pairs, %d seeds" % (TOYS, commas(m["draws"]), len(m["seeds"])), op)

# ====================================================================== 131: the mean
SINGLE_INK, MEAN_INK = AMBER, RED             # one measurement; the mean of an experiment
N_INK = (AMBER, RED, MAGENTA, VIOLET)         # means of 1, 4, 16, 64 measurements


class ErrorOnTheMean(Stats):
    """Four measurements and their mean, twelve times over; the means of thousands of
    experiments for 1, 4, 16 and 64 measurements against bell curves of width
    sigma / sqrt(N); their spread against N on log axes; why the sample's own spread
    is divided by N - 1; and what halving the error costs."""

    TITLE = "The error on the mean"
    PHASES = (("one experiment", 10.0), ("repeat it", 11.0), ("1/√N", 11.0), ("N − 1", 12.0),
              ("the cost", 9.0))
    AXIS = (4.0, 16.0)

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        d = s.panels[0].data
        self.shown = np.asarray(d["shown"], float)
        self.shown_means = np.asarray(d["shown_means"], float)
        self.ss = np.asarray(d["shown_ss"], float)
        self.mids = np.asarray(d["x"], float)
        self.hists, self.bells = np.asarray(d["ys"], float), np.asarray(d["exact"], float)
        self.ns = np.asarray(s.panels[1].data["x"], float)
        self.spread, self.spread_skew, self.law = np.asarray(s.panels[1].data["ys"], float)
        self.sd_ns = np.asarray(s.panels[2].data["x"], float)
        ys = np.asarray(s.panels[2].data["ys"], float)
        self.sd_measured, self.sd_exact = ys[:3], ys[3:]
        self.cost = np.asarray(s.panels[3].data["values"], float)
        self.mu, self.sigma = float(m["mu"]), float(m["sigma"])

    def subtitle_for(self, k):
        m = self.meta
        sp = m["spread"]
        r4 = m["sd_ratio"]["4"]
        return [
            "one experiment: N = 4 measurements of a quantity whose true value is μ = %g, "
            "each with error σ = %g. Their mean is the result. Repeated 12 times, the means "
            "scatter by %.2f: less than single measurements do"
            % (self.mu, self.sigma, m["shown_spread"]),
            "%s experiments for each N. Single measurements spread by %.2f; means of 4 by "
            "%.2f, of 16 by %.2f, of 64 by %.2f. Each histogram follows the bell curve of "
            "width σ/√N" % (commas(m["toys"]), sp["1"], sp["4"], sp["16"], sp["64"]),
            "on log axes the spread of the mean against N is a straight line of slope %s: "
            "σ/√N. Variances add, so a sum of N has variance Nσ²; dividing the sum by N "
            "leaves σ²/N" % signed(m["slope"]),
            "σ is rarely known: it is estimated from the same data. Deviations from the "
            "sample's own mean are too small, so dividing their squares by N gives %d%% of "
            "σ² at N = 4. Dividing by N − 1 gives %d%%"
            % (round(100 * r4[0]), round(100 * r4[1])),
            "what precision costs: 4 measurements give ±%.2f, 16 give ±%.2f, 64 give ±%.2f, "
            "256 give ±%.3f. Each halving of the error takes four times the data"
            % tuple(self.cost),
        ][k]

    # ------------------------------------------------------------------ phase 1
    def _schedule(self, u):
        """Which experiment is being done at ``u``, and how far through it."""
        if u < 0.36:
            return 0, float(np.clip((u - 0.03) / 0.33, 0, 1))
        k = (u - 0.36) / 0.045
        n = len(self.shown)
        return (int(k) + 1, k - int(k)) if k < n - 1 else (n - 1, 1.0)

    def phase_1(self, sc, u, op):
        k, f = self._schedule(u)
        top = LossChart(((0.08, 0.60), (0.40, 0.58)), self.AXIS, (0.0, 0.215))
        xs = np.linspace(*self.AXIS, 100)
        self.path(sc, top, xs, S.normal_pdf(xs, self.mu, self.sigma), GREY, op, 2.4)
        ticks = [(v, "%d" % v) for v in range(4, 17, 2)]
        self.axes(sc, top, op, "", "", xticks=ticks, yticks=[], grid=False)
        self.vmark(sc, top, self.mu, GREY, op)
        self.text(sc, top.fx(self.mu) + 0.01, 0.385, "true value μ = %g" % self.mu, op,
                  anchor="lm", ink=dark(GREY, 0.25), size=0.030)
        y_sigma = top.fy(float(S.normal_pdf(self.mu + self.sigma, self.mu, self.sigma)))
        self.polyline(sc, [self.P(top.fx(self.mu), y_sigma),
                           self.P(top.fx(self.mu + self.sigma), y_sigma)], 2.2, GREY, op)
        self.text(sc, top.fx(self.mu + self.sigma) + 0.012, y_sigma, "σ = %g" % self.sigma, op,
                  anchor="lm", ink=dark(GREY, 0.25), size=0.030)
        self.text(sc, 0.08, 0.345, "single measurements", op, anchor="lm",
                  ink=dark(SINGLE_INK, 0.35), size=0.031)
        low = 0.80
        self.polyline(sc, [self.P(top.x0, low), self.P(top.x1, low)], 1.2, AXIS, 0.8 * op)
        for v, name in ticks:
            self.polyline(sc, [self.P(top.fx(v), low), self.P(top.fx(v), low + 0.01)], 1.2, AXIS,
                          0.8 * op)
            self.text(sc, top.fx(v), low + 0.032, name, op, size=0.029)
        self.text(sc, 0.08, 0.69, "the mean of each experiment", op, anchor="lm",
                  ink=dark(MEAN_INK, 0.3), size=0.031)
        self.dash(sc, (top.fx(self.mu), low - 0.075), (top.fx(self.mu), low), GREY, op, 1.8, 5)
        # experiments already done: their measurements pale, their means below
        self.stack(sc, self.shown[:k].ravel(), top.fx, top.y1, soft(SINGLE_INK, 0.45), op, r=3.6)
        self.stack(sc, self.shown_means[:k], top.fx, low, MEAN_INK, op, r=4.6, bin_px=9.6)
        row, mean = self.shown[k], float(self.shown_means[k])
        for j, v in enumerate(row):
            fall = smoothstep(0.04 + 0.11 * j, 0.2 + 0.11 * j, f)
            if fall > 0:
                y = top.y0 + (top.y1 - top.y0 - 0.012) * fall
                self.disk(sc, self.P(top.fx(v), y), 6.0, SINGLE_INK, op, lift=1.6, outline=0.1,
                          glow=0.3 * (1 - fall))
        got = smoothstep(0.58, 0.7, f)
        if got > 0:
            y = top.y1 - 0.012 + (low - 0.012 - top.y1 + 0.012) * smoothstep(0.74, 0.96, f)
            self.disk(sc, self.P(top.fx(mean), y), 7.0, MEAN_INK, op * got, lift=1.8,
                      outline=0.0, glow=0.4)
        x = 0.66
        self.text(sc, x, 0.345, "experiment %d of %d" % (k + 1, len(self.shown)), op,
                  anchor="lm", strong=True, size=BIG)
        self.text(sc, x, 0.40, "N = 4 measurements:", op, anchor="lm", size=0.031)
        for j, v in enumerate(row):
            on = op * smoothstep(0.1 + 0.11 * j, 0.2 + 0.11 * j, f)
            self.box(sc, x + 0.03 + 0.068 * j, 0.455, 54, 30, soft(SINGLE_INK, 0.5), on,
                     "%.1f" % v, border=SINGLE_INK, size=0.031)
        self.lines(sc, x, 0.52, ["sum = %.1f" % row.sum(),
                                 "mean = %.1f / 4 = %.3f" % (row.sum(), mean)],
                   op * got, size=0.032, step=0.047)
        late = op * smoothstep(0.86, 0.92, u)
        self.lines(sc, x, 0.66, [("single measurements", dark(SINGLE_INK, 0.35)),
                                 ("spread by σ = %g" % self.sigma, dark(SINGLE_INK, 0.35)),
                                 ("these 12 means", dark(MEAN_INK, 0.3)),
                                 ("spread by %.2f" % self.meta["shown_spread"],
                                  dark(MEAN_INK, 0.3))], late, size=0.031, step=0.044)
        self.tag(sc, "%s: Gaussian, μ = %g, σ = %g, seed %d"
                 % (TOYS, self.mu, self.sigma, S.EM_SEEDS[0]), op)

    # ------------------------------------------------------------------ phase 2
    def phase_2(self, sc, u, op):
        m = self.meta
        x0, x1 = 0.08, 0.58
        cols = (0.655, 0.775, 0.895)
        self.text(sc, x0, 0.335, "means of %s experiments, for four values of N"
                  % commas(m["toys"]), op, anchor="lm", strong=True, size=HEAD)
        for cx, name in zip(cols, ("N", "spread", "σ/√N")):
            self.text(sc, cx, 0.39, name, op, size=0.031, strong=True)
        chart = None
        for r, n in enumerate(S.EM_HIST_NS):
            base = 0.475 + 0.105 * r
            grow = smoothstep(0.04 + 0.17 * r, 0.2 + 0.17 * r, u)
            peak = float(self.bells[r].max()) * 1.12
            chart = LossChart(((x0, x1), (base - 0.085, base)), self.AXIS, (0.0, peak))
            self.polyline(sc, [self.P(x0, base), self.P(x1, base)], 1.0, AXIS, 0.7 * op)
            self.hist(sc, chart, self.mids, self.hists[r], soft(N_INK[r], 0.2), op, grow=grow)
            curve = smoothstep(0.16 + 0.17 * r, 0.26 + 0.17 * r, u)
            self.path(sc, chart, self.mids, self.bells[r], DIGIT_INK, op, 1.8, go=curve)
            sd = m["spread"][str(n)]
            on = op * curve
            self.polyline(sc, [self.P(chart.fx(self.mu - sd), base + 0.011, lift=1.0),
                               self.P(chart.fx(self.mu + sd), base + 0.011, lift=1.0)], 4.0,
                          dark(N_INK[r], 0.25), on)
            y = base - 0.04
            self.text(sc, cols[0], y, "%d" % n, op, ink=dark(N_INK[r], 0.3), size=0.034)
            self.text(sc, cols[1], y, "%.3f" % sd, on, ink=dark(N_INK[r], 0.3), size=0.034)
            self.text(sc, cols[2], y, "%.3f" % (self.sigma / math.sqrt(n)), on, size=0.034,
                      strong=True)
        for v in range(4, 17, 2):
            self.polyline(sc, [self.P(chart.fx(v), chart.y1 + 0.018),
                               self.P(chart.fx(v), chart.y1 + 0.028)], 1.2, AXIS, 0.8 * op)
            self.text(sc, chart.fx(v), chart.y1 + 0.05, "%d" % v, op, size=0.029)
        self.lines(sc, 0.62, 0.815, ["black: the bell curve of width σ/√N",
                                     "thick bar: ± the measured spread"], op, size=0.029,
                   step=0.042)
        self.tag(sc, "%s against exact curves" % TOYS, op)

    # ------------------------------------------------------------------ phase 3
    def phase_3(self, sc, u, op):
        m = self.meta
        go = (u - 0.04) / 0.5
        chart = LossChart(((0.10, 0.52), (0.40, 0.76)), (1.0, 1024.0), (0.045, 2.8), xlog=True,
                          ylog=True)
        self.axes(sc, chart, op, "measurements per experiment, N", "spread of the mean",
                  xticks=[(v, "%d" % v) for v in (1, 4, 16, 64, 256, 1024)],
                  yticks=[(v, "%g" % v) for v in (0.0625, 0.125, 0.25, 0.5, 1, 2)])
        progress(self, sc, chart, self.ns, self.law, GREY, op, 1.0, 2.0, xlog=True, ylog=True,
                 dot=False)
        shown = int(np.clip(go, 0, 1) * len(self.ns) + 1e-9)
        for j in range(shown):
            self.disk(sc, self.P(chart.fx(self.ns[j]), chart.fy(self.spread_skew[j])), 7.0,
                      TEAL, op, lift=1.3, outline=0.0)
            self.disk(sc, self.P(chart.fx(self.ns[j]), chart.fy(self.spread[j])), 4.4, MEAN_INK,
                      op, lift=1.6, outline=0.0)
        self.key(sc, 0.11, 0.872, [("σ/√N", GREY)], op, line=True)
        self.key(sc, 0.21, 0.872, [("Gaussian", MEAN_INK)], op)
        self.key(sc, 0.34, 0.872, [("lopsided, same σ", TEAL)], op)
        x = 0.60
        self.text(sc, x, 0.335, "why 1/√N: variances add", op, anchor="lm", strong=True,
                  size=HEAD)
        rows = (("one measurement: variance σ²", 0.1),
                ("sum of N independent ones:", 0.25), ("    variance N σ²", 0.25),
                ("mean = sum / N:", 0.42), ("    variance N σ² / N² = σ² / N", 0.42),
                ("spread of the mean = σ / √N", 0.6))
        for r, (line, t0) in enumerate(rows):
            self.text(sc, x, 0.395 + 0.046 * r, line, op * smoothstep(t0, t0 + 0.1, u),
                      anchor="lm", size=0.031, strong=r == len(rows) - 1)
        late = op * smoothstep(0.74, 0.84, u)
        self.lines(sc, x, 0.705, [
            ("measured slope: %s" % signed(m["slope"]), dark(MEAN_INK, 0.3)),
            ("N = 1024: %.4f; σ/√N = %.4f" % (m["spread"]["1024"], self.sigma / 32), STRONG),
            ("the shape of the distribution", dark(TEAL, 0.3)),
            ("does not matter, only its σ", dark(TEAL, 0.3))], late, size=0.030, step=0.043)
        self.tag(sc, "%s: %s experiments per N, %d seeds"
                 % (TOYS, commas(m["toys"]), len(m["seeds"])), op)

    # ------------------------------------------------------------------ phase 4
    def phase_4(self, sc, u, op):
        m = self.meta
        n = len(self.ss)
        x0, x1, base = 0.07, 0.47, 0.735
        unit = 0.26 / float(self.ss.max())                       # layout y per unit of the sum
        self.text(sc, x0, 0.335, "12 experiments of N = 4: squared deviations, summed", op,
                  anchor="lm", strong=True, size=0.031)
        self.polyline(sc, [self.P(x0, base), self.P(x1, base)], 1.2, AXIS, 0.8 * op)
        step = (x1 - x0) / n
        w = self.span(step * 0.36)[0]
        for e in range(n):
            on = smoothstep(0.03 + 0.03 * e, 0.09 + 0.03 * e, u)
            cx = x0 + (e + 0.5) * step
            for side, (value, ink) in enumerate(((self.ss[e, 1], GREY), (self.ss[e, 0],
                                                                         MEAN_INK))):
                h = unit * float(value) * on
                if h * self.cam.height < 0.7:
                    continue
                self.rect(sc, self.P(cx + (side - 0.5) * step * 0.4, base - h / 2), w,
                          self.span(0, h)[1], ink, op, radius=1.5, lift=0.3)
        self.key(sc, x0 + 0.01, 0.385, [("from the true value μ", GREY)], op)
        self.key(sc, x0 + 0.01, 0.428, [("from the sample's own mean: never larger", MEAN_INK)],
                 op)
        mid = op * smoothstep(0.42, 0.52, u)
        own, true = m["shown_ss"]
        self.lines(sc, x0, 0.775, [
            ("average from μ: %.1f   (N σ² = %g)" % (true, 4 * self.sigma ** 2),
             dark(GREY, 0.3)),
            ("average from own mean: %.1f   ((N − 1) σ² = %g)" % (own, 3 * self.sigma ** 2),
             dark(MEAN_INK, 0.3)),
            ("so divide by N − 1, not N, to estimate σ²", STRONG)], mid, size=0.030, step=0.043)
        go = (u - 0.5) / 0.3
        chart = LossChart(((0.60, 0.94), (0.42, 0.72)), (2.0, 64.0), (0.45, 1.06), xlog=True)
        self.axes(sc, chart, op, "measurements per experiment, N", "estimate ÷ truth, averaged",
                  xticks=[(v, "%d" % v) for v in (2, 4, 8, 16, 32, 64)],
                  yticks=[(v, "%g" % v) for v in (0.5, 0.75, 1.0)])
        inks = (AMBER, GREEN, VIOLET)
        for r in (0, 2, 1):
            progress(self, sc, chart, self.sd_ns, self.sd_exact[r], inks[r], op, go, 2.4,
                     xlog=True, dot=False)
            for j in range(int(np.clip(go, 0, 1) * len(self.sd_ns) + 1e-9)):
                self.point(sc, chart, self.sd_ns[j], self.sd_measured[r, j], inks[r], op, 4.0)
        late = op * smoothstep(0.55, 0.65, u)
        self.key(sc, 0.61, 0.335, [("s² with ÷ (N − 1)", GREEN)], late, size=0.029)
        self.key(sc, 0.80, 0.335, [("s² with ÷ N", AMBER)], late, size=0.029)
        self.text(sc, 0.94, 0.868, "violet: s itself, the square root; it stays a little low",
                  op * smoothstep(0.82, 0.9, u), anchor="rm", ink=dark(VIOLET, 0.3), size=0.029)
        self.tag(sc, "%s (dots: %s per N) against exact values (lines)"
                 % (TOYS, commas(m["sd_toys"])), op)

    # ------------------------------------------------------------------ phase 5
    def phase_5(self, sc, u, op):
        self.text(sc, 0.07, 0.335, "halving the error takes four times the measurements", op,
                  anchor="lm", strong=True, size=HEAD)
        cell, floor = 10.0, 0.715
        centres = (0.125, 0.26, 0.42, 0.645)
        for r, n in enumerate(S.EM_COST_NS):
            on = op * smoothstep(0.04 + 0.16 * r, 0.16 + 0.16 * r, u)
            side = int(round(math.sqrt(n)))
            px = cell * side
            dy = self.frac(0, px)[1]
            self.cells(sc, np.ones((side, side)), centres[r], floor - dy / 2, px, px, on,
                       ink=soft(SINGLE_INK, 0.15), grid=0.5)
            self.text(sc, centres[r], floor - dy - 0.035, "N = %d" % n, on, strong=True,
                      size=0.032)
            half = self.frac(60.0 * float(self.cost[r]))[0]
            y = 0.765
            self.polyline(sc, [self.P(centres[r] - half, y), self.P(centres[r] + half, y)], 5.0,
                          MEAN_INK, on)
            self.disk(sc, self.P(centres[r], y), 5.0, DIGIT_INK, on, lift=1.6, outline=0.0)
            self.text(sc, centres[r], 0.812, "± %.3f" % self.cost[r], on,
                      ink=dark(MEAN_INK, 0.3), size=0.032)
            if r:
                self.text(sc, centres[r], 0.858, "error ÷ %.2f"
                          % (self.cost[r - 1] / self.cost[r]), on, size=0.029)
        late = op * smoothstep(0.72, 0.84, u)
        self.lines(sc, 0.80, 0.42, ["for ± 0.01:", "N = (σ / 0.01)²", "   = 40,000"], late,
                   size=0.032, step=0.048)
        self.lines(sc, 0.80, 0.60, ["one square:", "one measurement"], op, size=0.029,
                   step=0.042)
        self.tag(sc, "%s: %s per N" % (TOYS, commas(self.meta["toys"])), op)

# ====================================================================== 132: the bootstrap
DATA_INK, PICK_INK, TRUE_INK = BLUE, AMBER, TEAL   # the dataset; a resample; fresh datasets
SHORT = ("mean", "median", "ratio", "correl.")


class BootstrapErrors(Stats):
    """Twenty events and five statistics; the events picked again with replacement, some
    twice and some never, and the mean recomputed two thousand times; the same for
    statistics with no error formula, against the true spread over fresh datasets; how
    the answer settles as resamples are added; and the largest value, which no resample
    can exceed."""

    TITLE = "The bootstrap"
    PHASES = (("one dataset", 8.0), ("resample", 14.0), ("no formula", 12.0),
              ("how many", 9.0), ("it fails", 10.0))
    X0, STEP = 0.0735, 0.0449

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        d = s.panels[0].data
        self.x, self.y = np.asarray(d["data_x"], float), np.asarray(d["data_y"], float)
        self.picks = np.asarray(d["picks"], int)
        self.first = np.asarray(d["first"], float)             # statistics of 60 resamples
        self.mean_x = np.asarray(d["x"], float)
        self.boot_mean_hist, self.true_mean_hist = np.asarray(d["ys"], float)
        self.band = np.asarray(s.panels[1].data["ratio_band"], float)
        self.bs = np.asarray(s.panels[2].data["x"], float)
        self.scatter, self.scatter_ref = np.asarray(s.panels[2].data["ys"], float)
        self.runs = np.asarray(s.panels[2].data["runs"], float)
        self.runs_mean = np.asarray(s.panels[2].data["mean"], float)
        self.max_x = np.asarray(s.panels[3].data["x"], float)
        self.boot_max, self.exact_max, self.true_max = np.asarray(s.panels[3].data["ys"], float)
        self.n = int(m["events"])
        self.here, self.se, self.true = m["here"], m["bootstrap"], m["true"]
        self.xs = [self.X0 + self.STEP * i for i in range(self.n)]

    def subtitle_for(self, k):
        m = self.meta
        sc = m["scatter"]
        return [
            "one dataset of %d events from a stated model, each with two measured numbers, "
            "x and y. The mean has an error formula, s/√N = %.2f. The median, a ratio and a "
            "correlation have no simple one" % (self.n, m["formula_mean"]),
            "the bootstrap: draw %d events from the dataset itself, with replacement, so "
            "some repeat and some are left out, and recompute. %s recomputed means spread "
            "by %.3f; the formula s/√N gives %.3f"
            % (self.n, commas(m["resamples"]), self.se[0], m["formula_mean"]),
            "the same resamples give an error for any statistic. Against the true spread "
            "over %s fresh datasets: median %.2f against %.2f, correlation %.2f against "
            "%.2f. Right on average, not for every dataset"
            % (commas(m["fresh"]), self.se[1], self.true[1], self.se[3], self.true[3]),
            "how many resamples? Repeating the whole bootstrap %d times, its answer scatters "
            "by %s after 10 resamples, %s after 200 and %s after 2,000. Having only one "
            "dataset costs far more than that"
            % (S.BS_REPEATS, pct(sc["10"], 0), pct(sc["200"]), pct(sc["2000"])),
            "where it fails: the largest value. No resample can exceed the largest value "
            "seen, and %s of them return exactly it. Over %d datasets the bootstrap error is "
            "typically %d%% of the true spread"
            % (pct(m["top_share"], 0), m["datasets"], round(100 * m["ratio_median"][4])),
        ][k]

    # ------------------------------------------------------------------ rows of events
    def _event(self, sc, fx, fy, value, ink, op, strength=0.5, w=36.0, h=29.0):
        self.box(sc, fx, fy, w, h, soft(ink, 1.0 - strength) if strength > 0 else PAPER, op,
                 "%.1f" % value if abs(value) < 9.95 else "%.0f" % value,
                 border=ink if strength > 0 else DIM, size=0.029, radius=4,
                 ink=None if strength > 0 else GREY)

    def phase_1(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.055, 0.335, "%d events (seed %d).  Model: x exponential with mean %g;  "
                  "y = 1 + x/2 + noise (σ = 1)" % (self.n, m["seed"], S.BS_TAU), op,
                  anchor="lm", strong=True, size=0.031)
        shown = int(smoothstep(0.03, 0.3, u) * self.n + 1e-9)
        for i in range(shown):
            self._event(sc, self.xs[i], 0.41, self.x[i], DATA_INK, op)
            self._event(sc, self.xs[i], 0.475, self.y[i], VIOLET, op)
        self.text(sc, 0.052, 0.41, "x", op, anchor="rm", ink=dark(DATA_INK, 0.3), size=0.032)
        self.text(sc, 0.052, 0.475, "y", op, anchor="rm", ink=dark(VIOLET, 0.3), size=0.032)
        late = smoothstep(0.34, 0.44, u)
        chart = LossChart(((0.09, 0.40), (0.58, 0.80)), (0.0, 13.0), (-1.0, 8.5))
        self.axes(sc, chart, op * late, "x", "y", xticks=[(v, "%d" % v) for v in (0, 4, 8, 12)],
                  yticks=[(v, "%d" % v) for v in (0, 4, 8)])
        for i in range(self.n):
            self.point(sc, chart, self.x[i], self.y[i], DATA_INK, op * late, 4.4)
        x = 0.50
        self.text(sc, x, 0.565, "computed from these %d events:" % self.n, op * late,
                  anchor="lm", strong=True, size=0.031)
        truth = m["true_values"]
        rows = (("mean of x", self.here[0], truth["mean"]),
                ("median of x", self.here[1], truth["median"]),
                ("mean of y ÷ mean of x", self.here[2], truth["ratio of means"]),
                ("correlation of x and y", self.here[3], truth["correlation"]))
        for r, (name, value, true) in enumerate(rows):
            on = op * smoothstep(0.42 + 0.08 * r, 0.5 + 0.08 * r, u)
            y = 0.615 + 0.045 * r
            self.text(sc, x, y, name, on, anchor="lm", size=0.031)
            self.text(sc, x + 0.27, y, "%.2f" % value, on, anchor="rm", size=0.032, strong=True)
            self.text(sc, x + 0.29, y, "(model: %.2f)" % true, on, anchor="lm", size=0.029,
                      ink=dark(TRUE_INK, 0.3))
        on = op * smoothstep(0.74, 0.82, u)
        self.text(sc, x, 0.795, "largest x", on, anchor="lm", size=0.031)
        self.text(sc, x + 0.27, 0.795, "%.2f" % self.here[4], on, anchor="rm", size=0.032,
                  strong=True)
        self.text(sc, x, 0.868, "error on the mean: s/√N = %.2f.  On the others?"
                  % m["formula_mean"], op * smoothstep(0.86, 0.94, u), anchor="lm",
                  ink=dark(RED, 0.25), size=0.031)
        self.tag(sc, "%s: one dataset drawn from the model" % TOYS[:-1], op)

    # ------------------------------------------------------------------ phase 2
    def _picking(self, u):
        """Which resample is being drawn at ``u`` and how many of its picks are made."""
        if u < 0.40:
            t = (u - 0.03) / 0.35
            return 0, float(np.clip(t, 0, 1)) * self.n
        r = min(int((u - 0.40) / 0.08) + 1, len(self.picks) - 1)
        t = (u - 0.40 - 0.08 * (r - 1)) / 0.065
        return r, float(np.clip(t, 0, 1)) * self.n

    def phase_2(self, sc, u, op):
        m = self.meta
        r, t = self._picking(u)
        done = int(min(t, self.n))
        complete = done >= self.n
        idx = self.picks[r]
        counts = np.bincount(idx[:done], minlength=self.n)
        top, mid = 0.435, 0.59
        self.text(sc, 0.055, 0.335, "the dataset: %d values of x.  Pick one at random, copy "
                  "it, put it back; %d times" % (self.n, self.n), op, anchor="lm", strong=True,
                  size=0.031)
        for i in range(self.n):
            c = int(counts[i])
            left_out = complete and c == 0
            self._event(sc, self.xs[i], top, self.x[i], DATA_INK, op,
                        strength=0.0 if left_out else min(0.3 + 0.25 * c, 0.9))
            if c > 1 or left_out:
                self.text(sc, self.xs[i], top - 0.046, "×%d" % c, op, size=0.029,
                          ink=dark(RED, 0.2) if left_out else dark(PICK_INK, 0.4))
        for j in range(done):
            self._event(sc, self.xs[j], mid, self.x[idx[j]], PICK_INK, op)
        if not complete:                                # the pick being made
            j = done
            f = t - done
            a, b = (self.xs[idx[j]], top + 0.032), (self.xs[j], mid - 0.032)
            self.arrow(sc, a, b, dark(PICK_INK, 0.2), op, u=smoothstep(0.0, 0.7, f), width=2.4)
            self.rect(sc, self.P(self.xs[idx[j]], top), 42, 35, PICK_INK, op, border=PICK_INK,
                      border_w=2.6, radius=5, lift=0.9, fill_op=0.0)
        self.text(sc, 0.055, mid + 0.054, "resample %d of %s:" % (r + 1, commas(m["resamples"])),
                  op, anchor="lm", ink=dark(PICK_INK, 0.4), size=0.031)
        if complete:
            self.text(sc, 0.27, mid + 0.054, "%d events left out, %d picked more than once;  "
                      "its mean = %.2f" % ((counts == 0).sum(), (counts > 1).sum(),
                                           self.first[r, 0]), op, anchor="lm", size=0.031)
        chart = LossChart(((0.075, 0.56), (0.705, 0.835)), (0.0, 4.5), (0.0, 0.72))
        self.axes(sc, chart, op, "mean of a resample", "",
                  xticks=[(v, "%d" % v) for v in range(5)], yticks=[], grid=False)
        grow = smoothstep(0.68, 0.84, u)
        finished = r + (1 if complete else 0)
        self.stack(sc, self.first[:finished, 0], chart.fx, chart.y1, PICK_INK,
                   op * (1 - smoothstep(0.68, 0.76, u)), r=4.4)
        self.hist(sc, chart, self.mean_x, self.boot_mean_hist, soft(PICK_INK, 0.15), op,
                  grow=grow)
        self.vmark(sc, chart, self.here[0], DATA_INK, op, dashed=False, width=2.4)
        late = op * smoothstep(0.82, 0.9, u)
        self.polyline(sc, [self.P(chart.fx(self.here[0] - self.se[0]), chart.fy(0.66), lift=1.0),
                           self.P(chart.fx(self.here[0] + self.se[0]), chart.fy(0.66), lift=1.0)],
                      4.0, dark(PICK_INK, 0.3), late)
        x = 0.61
        self.lines(sc, x, 0.705, [("blue line: the dataset's mean, %.2f" % self.here[0],
                                   dark(DATA_INK, 0.3))], op, size=0.030)
        self.lines(sc, x, 0.752, [
            ("spread of %s resampled means:" % commas(m["resamples"]), dark(PICK_INK, 0.4)),
            ("%.3f, the bootstrap error" % self.se[0], dark(PICK_INK, 0.4)),
            ("formula s/√N: %.3f" % m["formula_mean"], STRONG)], late, size=0.031, step=0.045)
        self.tag(sc, "%s: resamples of the one dataset, seed %d" % (TOYS, m["seed"]), op)

    # ------------------------------------------------------------------ phase 3
    def phase_3(self, sc, u, op):
        m = self.meta
        cols = (0.245, 0.325, 0.405, 0.485)
        x = 0.055
        self.text(sc, x, 0.335, "recomputed on every resample", op, anchor="lm", strong=True,
                  size=0.031)
        for cx, name in zip(cols, SHORT):
            self.text(sc, cx, 0.385, name, op, size=0.030, strong=True)

        def row(y, name, values, on, ink=None, fmt="%.2f", strong=False, chip=None):
            if chip is not None:
                self.box(sc, x + 0.008, y, 13, 13, chip, on, radius=3)
            self.text(sc, x + (0.026 if chip is not None else 0.0), y, name, on, anchor="lm",
                      size=0.030, ink=ink, strong=strong)
            for cx, v in zip(cols, values):
                self.text(sc, cx, y, fmt % v, on, size=0.031, ink=ink, strong=strong)
        row(0.432, "this dataset", self.here[:4], op, dark(DATA_INK, 0.3), chip=DATA_INK)
        for r in range(4):
            row(0.478 + 0.042 * r, "resample %d" % (r + 1), self.first[r, :4],
                op * smoothstep(0.04 + 0.04 * r, 0.1 + 0.04 * r, u), dark(PICK_INK, 0.4),
                chip=PICK_INK)
        self.text(sc, x, 0.648, "… and %s more" % commas(m["resamples"] - 4),
                  op * smoothstep(0.2, 0.26, u), anchor="lm", size=0.030,
                  ink=dark(PICK_INK, 0.4))
        row(0.70, "their spread", self.se[:4], op * smoothstep(0.26, 0.33, u), strong=True,
            fmt="%.3f")
        row(0.748, "true spread", self.true[:4], op * smoothstep(0.34, 0.41, u),
            dark(TRUE_INK, 0.3), fmt="%.3f", chip=TRUE_INK)
        self.lines(sc, x, 0.815, [
            "their spread: the bootstrap error, from this one dataset",
            ("true spread: over %s fresh datasets from the model" % commas(m["fresh"]),
             dark(TRUE_INK, 0.3))], op * smoothstep(0.34, 0.41, u), size=0.029, step=0.04)
        # bootstrap error over true spread: this dataset, and many others
        late = op * smoothstep(0.42, 0.5, u)
        ax0, ax1, base = 0.70, 0.94, 0.745

        def at(v):
            return ax0 + (ax1 - ax0) * float(v) / 2.0
        self.text(sc, 0.60, 0.335, "bootstrap error ÷ true spread", late, anchor="lm",
                  strong=True, size=0.031)
        self.polyline(sc, [self.P(ax0, base), self.P(ax1, base)], 1.2, AXIS, 0.8 * late)
        for v in (0, 0.5, 1, 1.5, 2):
            self.polyline(sc, [self.P(at(v), base), self.P(at(v), base + 0.01)], 1.2, AXIS,
                          0.8 * late)
            self.text(sc, at(v), base + 0.032, "%g" % v, late, size=0.029)
        self.dash(sc, (at(1), 0.385), (at(1), base), DIGIT_INK, late, 1.8, 14)
        for r, name in enumerate(SHORT):
            y = 0.435 + 0.075 * r
            on = late * smoothstep(0.48 + 0.07 * r, 0.56 + 0.07 * r, u)
            self.text(sc, 0.60, y, name, on, anchor="lm", size=0.031)
            self.polyline(sc, [self.P(at(self.band[0, r]), y), self.P(at(self.band[2, r]), y)],
                          9.0, soft(TRUE_INK, 0.35), on)
            self.polyline(sc, [self.P(at(self.band[1, r]), y - 0.016, lift=0.8),
                               self.P(at(self.band[1, r]), y + 0.016, lift=0.8)], 2.6,
                          dark(TRUE_INK, 0.3), on)
            self.disk(sc, self.P(at(self.se[r] / self.true[r]), y), 5.5, RED, on, lift=1.6,
                      outline=0.0)
        self.key(sc, 0.61, 0.832, [("this dataset", RED)], late, size=0.029)
        self.key(sc, 0.61, 0.872, [("%d other datasets: middle 68%%" % m["datasets"],
                                    soft(TRUE_INK, 0.2))], late, size=0.029)
        self.tag(sc, "%s: %s resamples; %s fresh datasets × %d seeds"
                 % (TOYS, commas(m["resamples"]), commas(m["fresh"]), len(S.BS_TRUE_SEEDS)), op)

    # ------------------------------------------------------------------ phase 4
    def phase_4(self, sc, u, op):
        m = self.meta
        go = (u - 0.04) / 0.5
        ticks = [(v, commas(v)) for v in (10, 100, 1000, 5000)]
        left = LossChart(((0.09, 0.47), (0.42, 0.76)), (10.0, 5000.0), (0.17, 0.56), xlog=True)
        self.axes(sc, left, op, "resamples used, B", "bootstrap error of the median",
                  xticks=ticks, yticks=[(v, "%g" % v) for v in (0.2, 0.3, 0.4, 0.5)])
        for run in self.runs:
            progress(self, sc, left, self.bs, run, soft(PICK_INK, 0.25), op, go, 1.5, xlog=True,
                     dot=False)
        progress(self, sc, left, self.bs, self.runs_mean, dark(PICK_INK, 0.2), op, go, 3.2,
                 xlog=True)
        self.text(sc, 0.09, 0.335, "the whole bootstrap, repeated: 12 of %d runs" % S.BS_REPEATS,
                  op, anchor="lm", strong=True, size=0.031)
        right = LossChart(((0.60, 0.94), (0.42, 0.76)), (10.0, 5000.0), (0.008, 0.7), xlog=True,
                          ylog=True)
        self.axes(sc, right, op, "resamples used, B", "scatter of the answer, run to run",
                  xticks=ticks, yticks=[(v, "%g%%" % (100 * v)) for v in (0.01, 0.03, 0.1, 0.3)])
        progress(self, sc, right, self.bs, self.scatter_ref, GREY, op, 1.0, 2.0, xlog=True,
                 ylog=True, dot=False)
        for j in range(int(np.clip(go, 0, 1) * len(self.bs) + 1e-9)):
            self.point(sc, right, self.bs[j], self.scatter[j], PICK_INK, op, 5.0)
        late = op * smoothstep(0.62, 0.74, u)
        between = 0.5 * (m["ratio_high"][1] - m["ratio_low"][1])
        self.hmark(sc, right, between, TRUE_INK, late, 2.2)
        self.text(sc, 0.935, right.fy(between) + 0.03, "one dataset or another: ±%s"
                  % pct(between, 0), late, anchor="rm", ink=dark(TRUE_INK, 0.3), size=0.029)
        self.key(sc, 0.61, 0.335, [("measured", PICK_INK)], op, size=0.029)
        self.key(sc, 0.75, 0.335, [("1/√(2B)", GREY)], op, size=0.029, line=True)
        self.text(sc, 0.09, 0.872, "a few hundred resamples are enough: the rest of the "
                  "uncertainty is the dataset's", late, anchor="lm", strong=True, size=0.030)
        self.tag(sc, "%s: %d repeats of the resampling" % (TOYS, S.BS_REPEATS), op)

    # ------------------------------------------------------------------ phase 5
    def phase_5(self, sc, u, op):
        m = self.meta
        x0, x1 = 0.075, 0.55
        high = 0.15
        rows = ((0.535, self.true_max, soft(TRUE_INK, 0.2), "largest x of %s fresh datasets"
                 % commas(m["fresh"]), dark(TRUE_INK, 0.3), 0.04),
                (0.805, self.boot_max, soft(RED, 0.1), "largest x of %s resamples of ours"
                 % commas(m["resamples"]), dark(RED, 0.25), 0.3))
        for base, hist, ink, name, text_ink, t0 in rows:
            chart = LossChart(((x0, x1), (base - high, base)), (0.0, 16.0),
                              (0.0, float(hist.max()) * 1.08))
            self.polyline(sc, [self.P(x0, base), self.P(x1, base)], 1.0, AXIS, 0.7 * op)
            self.hist(sc, chart, self.max_x, hist, ink, op, grow=smoothstep(t0, t0 + 0.2, u))
            self.text(sc, x0, base - high - 0.03, name, op * smoothstep(t0, t0 + 0.1, u),
                      anchor="lm",
                      ink=text_ink, size=0.031)
            if base < 0.6:
                self.path(sc, chart, self.max_x, self.exact_max, DIGIT_INK, op, 2.2,
                          go=smoothstep(0.2, 0.3, u))
                self.text(sc, x1, base - 0.1, "black: exact", op * smoothstep(0.2, 0.3, u),
                          anchor="rm", size=0.029)
            else:
                on = op * smoothstep(0.48, 0.58, u)
                for n, (value, share) in enumerate(m["top_values"][:2]):
                    self.text(sc, chart.fx(value) - 0.012, base - 0.125 + 0.08 * n,
                              "%s are %.1f" % (pct(share, 0), value), on, anchor="rm",
                              size=0.029, ink=text_ink)
            for v in range(0, 17, 4):
                self.polyline(sc, [self.P(chart.fx(v), base), self.P(chart.fx(v), base + 0.01)],
                              1.2, AXIS, 0.8 * op)
                self.text(sc, chart.fx(v), base + 0.032, "%d" % v, op, size=0.029)
        x = 0.61
        self.text(sc, x, 0.335, "the largest of %d values" % self.n, op, anchor="lm",
                  strong=True, size=HEAD)
        mid = op * smoothstep(0.5, 0.6, u)
        self.lines(sc, x, 0.40, [
            "no resample can go above", "the largest value seen, %.1f;" % self.here[4],
            "%s of them contain it" % pct(m["top_share"], 0),
            "(exact: 1 − (19/20)²⁰ = %s)" % pct(m["top_share_exact"], 0)], mid, size=0.030,
            step=0.043)
        late = op * smoothstep(0.7, 0.8, u)
        self.lines(sc, x, 0.60, [
            ("bootstrap error here: %.2f" % self.se[4], dark(RED, 0.25)),
            ("true spread: %.2f (exact %.2f)" % (self.true[4], m["exact_max"]),
             dark(TRUE_INK, 0.3)),
            ("close only by luck: our", STRONG), ("top two are far apart", STRONG)], late,
            size=0.030, step=0.043)
        self.lines(sc, x, 0.795, [
            "over %d datasets, bootstrap ÷ true:" % m["datasets"],
            "typically %.2f (68%%: %.2f to %.2f)" % (m["ratio_median"][4], m["ratio_low"][4],
                                                    m["ratio_high"][4])],
            op * smoothstep(0.84, 0.92, u), size=0.030, step=0.043)
        self.tag(sc, "%s against the exact distribution" % TOYS, op)


ANIMATIONS = {
    "likelihood-curve": LikelihoodCurve,
    "likelihood-intervals": LikelihoodIntervals,
    "error-propagation": ErrorPropagation,
    "error-on-the-mean": ErrorOnTheMean,
    "bootstrap-errors": BootstrapErrors,
}

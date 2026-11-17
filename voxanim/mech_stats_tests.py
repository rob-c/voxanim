"""Hypothesis tests: animations 141-144 of the "next" catalogue (see :mod:`voxanim.mech`).

* :class:`TypeErrors`: a counting experiment, a threshold, and the two ways it errs;
* :class:`PValues`: one result against the distribution it would have if H0 were true;
* :class:`NeymanPearson`: the likelihood ratio against two other test statistics;
* :class:`LookElsewhere`: a scan of many windows and the largest excess it finds.

They draw the experiments of :mod:`voxanim.study_stats_tests`, which records every
number shown: this module computes layout, and the running counts of the
pseudo-experiments it shows arriving one by one. Every phase carries a tag saying what
is on screen: ``exact`` (a formula or a distribution evaluated) or
``pseudo-experiments`` (seeded draws from the stated model), or which part is which.

One meaning per colour in all four: blue is H0 (background only, nothing there), red is
H1 (a signal, a real effect), amber the observed value, the threshold and a false alarm.
"""

from __future__ import annotations

import math

import numpy as np

from . import layout
from . import study_stats_tests as S
from .diagram import draw_image
from .easing import smoothstep
from .losschart import LossChart
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIM,
    GREEN,
    PAPER,
    RED,
    TEAL,
    VIOLET,
    Mechanism,
    dark,
    soft,
)

BIG, HEAD = 0.036, 0.033
STRONG = layout.STRONG_INK
MINUS = "−"


def commas(n):
    """An integer with thousands separators."""
    return format(int(n), ",")


def pct(v, digits=1):
    return "%.*f%%" % (digits, 100.0 * float(v))


def signed(v, fmt="%.2f"):
    """A number with a typographic minus or a plus."""
    return ("+" if v >= 0 else MINUS) + fmt % abs(v)


def _rgb(c):
    return np.asarray(c, np.float32)


def scatter(x, y, xr, yr, shape, colours, back=PAPER, fat=3):
    """Points drawn into a ``shape`` (rows, cols) texture: ``xr`` and ``yr`` are the
    ranges shown, ``colours`` one colour per point, y upwards; later points on top."""
    rows, cols = shape
    tex = np.empty((rows, cols, 3), np.float32)
    tex[:] = _rgb(back)
    x, y = np.asarray(x, float), np.asarray(y, float)
    col = ((x - xr[0]) / (xr[1] - xr[0]) * cols).astype(int) - fat // 2
    row = ((1.0 - (y - yr[0]) / (yr[1] - yr[0])) * rows).astype(int) - fat // 2
    ok = (col >= 0) & (col <= cols - fat) & (row >= 0) & (row <= rows - fat)
    paint = np.asarray(colours, np.float32)[ok]
    for a in range(fat):
        for b in range(fat):
            tex[row[ok] + a, col[ok] + b] = paint
    return tex


def clip_line(w, c, xr, yr):
    """The two ends, inside the rectangle ``xr`` x ``yr``, of the line ``w . (x, y) = c``
    (None when it misses the rectangle)."""
    w = np.asarray(w, float)
    p = w * c / float(w @ w)
    d = np.array([-w[1], w[0]])
    lo, hi = -1e9, 1e9
    for k, (a, b) in enumerate((xr, yr)):
        if abs(d[k]) < 1e-12:
            if not a <= p[k] <= b:
                return None
            continue
        t0, t1 = sorted(((a - p[k]) / d[k], (b - p[k]) / d[k]))
        lo, hi = max(lo, t0), min(hi, t1)
    if lo >= hi:
        return None
    return p + lo * d, p + hi * d


class Tests(Mechanism):
    """What the four animations share: the evidence tag, bars and curves on a chart,
    dashed lines, pictures of many points, a legend."""

    def tag(self, sc, kind, op):
        """The kind of evidence on screen, bottom right."""
        self.text(sc, 0.95, 0.925, kind, op, anchor="rm", ink=dark(TEAL, 0.25))

    def graph(self, box, xr, yr, ylog=False):
        return LossChart(box, (float(xr[0]), float(xr[1])), (float(yr[0]), float(yr[1])),
                         ylog=ylog)

    def bars(self, sc, chart, xs, hs, width, inks, op, gap=1.0, grow=1.0):
        """Vertical bars from the chart's floor: ``width`` in units of x; ``inks`` one
        colour or one per bar."""
        one = np.ndim(inks) == 1
        floor = chart.y1
        for i, (x, h) in enumerate(zip(xs, hs)):
            if h <= 0:
                continue
            xa, xb = chart.fx(x - width / 2), chart.fx(x + width / 2)
            top = chart.fy(float(h) * grow)
            wpx, hpx = self.span(xb - xa, floor - top)
            if hpx < 0.5 or wpx < 0.3:
                continue
            self.rect(sc, self.P((xa + xb) / 2, (top + floor) / 2), max(wpx - gap, 0.8), hpx,
                      inks if one else inks[i], op, radius=min(1.5, wpx / 3), lift=0.4)

    def path(self, sc, chart, xs, ys, ink, op, width=2.6, go=1.0, lift=0.6):
        """A polyline through chart coordinates, as far as fraction ``go`` of its points."""
        m = int(round(len(xs) * float(np.clip(go, 0, 1))))
        if op <= 0.01 or m < 2:
            return
        pts = [self.P(chart.fx(float(a)), chart.fy(float(b)), lift=lift)
               for a, b in zip(xs[:m], ys[:m])]
        self.polyline(sc, pts, width, ink, op, bias=0.3)

    def steps(self, sc, chart, edges, hs, ink, op, width=2.2):
        """A histogram's outline."""
        xs = np.repeat(np.asarray(edges, float), 2)[1:-1]
        self.path(sc, chart, xs, np.repeat(np.asarray(hs, float), 2), ink, op, width)

    def vline(self, sc, fx, y0, y1, ink, op, width=2.6):
        self.polyline(sc, [self.P(fx, y0, lift=0.9), self.P(fx, y1, lift=0.9)], width, ink, op)

    def dashed(self, sc, a, b, ink, op, n=14, width=1.6):
        """A dashed straight line between two layout points."""
        (ax, ay), (bx, by) = a, b
        for i in range(n):
            u0, u1 = i / n, (i + 0.55) / n
            self.polyline(sc, [self.P(ax + (bx - ax) * u0, ay + (by - ay) * u0, lift=0.6),
                               self.P(ax + (bx - ax) * u1, ay + (by - ay) * u1, lift=0.6)],
                          width, ink, 0.45 * op)

    def picture(self, sc, tex, box, op, frame=AXIS):
        """A texture filling the layout box ``((x0, x1), (top, bottom))``."""
        (x0, x1), (y0, y1) = box
        w, h = self.span(x1 - x0, y1 - y0)
        c = self.P((x0 + x1) / 2, (y0 + y1) / 2)
        if frame is not None:
            self.rect(sc, c, w + 6, h + 6, PAPER, op, border=frame, border_w=1.2, radius=4,
                      lift=0.1)
        draw_image(self, sc, c, np.asarray(tex, np.float32), w, h, op, grid=0)

    def key(self, sc, fx, fy, entries, op, step=0.046, size=0.030, across=False):
        """A legend: a colour chip and a name per entry, stacked down from ``(fx, fy)``
        (or set side by side)."""
        x = fx
        for n, (name, ink) in enumerate(entries):
            y = fy if across else fy + n * step
            self.box(sc, x, y, 14, 14, ink, op, radius=3)
            self.text(sc, x + 0.014, y, name, op, anchor="lm", ink=dark(ink, 0.3), size=size)
            if across:
                x += 0.014 + layout.text_width(name, size) + 0.035

    def ticks(self, values, fmt="%g"):
        return [(v, (fmt % v).replace("-", MINUS)) for v in values]


# ====================================================================== 141: two errors
ALPHA_INK, BETA_INK = AMBER, VIOLET           # a false alarm; a miss


class TypeErrors(Tests):
    """A counting experiment under two hypotheses; a threshold on the count and the two
    areas it cuts off; the threshold sliding and tracing the ROC curve; the four
    outcomes counted in pseudo-experiments; more data pulling the hypotheses apart."""

    TITLE = "Two ways for a test to be wrong"
    PHASES = (("two hypotheses", 10.0), ("threshold", 12.0), ("sliding it", 11.0),
              ("outcomes", 11.0), ("more data", 12.0))
    NSHOW = 35                                # counts on the axis
    XR, YR = (-0.5, 35.5), (0.0, 0.136)

    def setup(self):
        d = self.d = S.errors_data()
        self.meta = self.study.metadata
        self.n_c = int(self.meta["threshold"])
        n = S.ERR_N
        # the pseudo-experiments in the order they arrive: H0, H1, H0, H1, ...
        self.truth = np.tile([0, 1], n)
        self.count = np.stack([d["h0"], d["h1"]], 1).ravel()
        cell = 2 * self.truth + (self.count >= self.n_c)
        self.cum = np.concatenate([np.zeros((1, 4), int),
                                   np.cumsum(np.eye(4, dtype=int)[cell], 0)])
        self.table = np.asarray(self.study.panels[2].data["values"], int)
        runs = [float(v) for v in d["runs"]]
        self.shown = [runs.index(t) for t in S.ERR_SHOWN]

    def subtitle_for(self, k):
        d, m = self.d, self.meta
        t = self.table
        i1, i4 = self.shown[0], self.shown[-1]
        return [
            "one experiment counts n events. H₀ says background only: b = %d expected. H₁ "
            "says a signal too: s + b = %d. Either way n is a Poisson draw, so the two "
            "distributions overlap" % (m["b"], m["b"] + m["s"]),
            "reject H₀ when n ≥ %d. Type I error: H₀ was true, chance α = %.3f (its bars "
            "beyond the threshold). Type II: H₁ was true but kept H₀, chance β = %.3f. "
            "Power = 1 − β = %.3f" % (self.n_c, m["alpha"], m["beta"], m["power"]),
            "slide the threshold: as α falls, β rises. Each threshold is one point of "
            "power against α, and together they trace the ROC curve; no threshold makes "
            "both errors small at once",
            "%s pseudo-experiments with each hypothesis true: %d false alarms, %s of H₀ "
            "(exact α: %s), and %s misses, %s of H₁ (exact β: %s)"
            % (commas(S.ERR_N), t[0, 1], pct(t[0, 1] / S.ERR_N), pct(m["alpha"]),
               commas(t[1, 0]), pct(t[1, 0] / S.ERR_N), pct(m["beta"])),
            "more data: both counts grow, their spread only as the square root. At α ≤ 5%% "
            "the power climbs from %s with one run to %s with four (exact; %s "
            "pseudo-experiments each agree)"
            % (pct(d["power_run"][i1], 0), pct(d["power_run"][i4], 1),
               commas(S.ERR_POWER_N)),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def pair(self, sc, chart, op, cut=None, grow0=1.0, grow1=1.0, lit=None):
        """The two Poisson distributions as bars side by side; with ``cut``, the H₀ bars
        at or beyond it and the H₁ bars below it take the colours of the two errors."""
        d = self.d
        n = np.arange(self.NSHOW + 1)
        ink0 = [BLUE if cut is None or k < cut else ALPHA_INK for k in n]
        ink1 = [RED if cut is None or k >= cut else BETA_INK for k in n]
        if lit is not None:                           # the errors not yet named stay pale
            ink0 = [c if (k < cut or lit[0]) else soft(BLUE, 0.2) for k, c in zip(n, ink0)]
            ink1 = [c if (k >= cut or lit[1]) else soft(RED, 0.2) for k, c in zip(n, ink1)]
        self.bars(sc, chart, n - 0.21, d["pmf0"][:len(n)], 0.42, ink0, op, grow=grow0)
        self.bars(sc, chart, n + 0.21, d["pmf1"][:len(n)], 0.42, ink1, op, grow=grow1)

    def count_axes(self, sc, chart, op, ylabel="probability"):
        self.axes(sc, chart, op, "n: events counted", ylabel,
                  xticks=self.ticks((0, 10, 20, 30), "%d"),
                  yticks=self.ticks((0.0, 0.05, 0.10), "%.2f"), grid=False)

    def cluster(self, sc, fx, fy, nb, ns, op):
        """One pseudo-experiment as its events: a dot each, background blue, signal red."""
        self.box(sc, fx, fy, 62, 60, PAPER, op, border=DIM, radius=5)
        c = self.P(fx, fy)
        for k in range(min(nb + ns, 42)):
            r, col = divmod(k, 7)
            self.disk(sc, self.shift(c, (col - 3) * 8.0, (r - 2.5) * 9.0), 3.1,
                      BLUE if k < nb else RED, op, lift=1.2, outline=0.1)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        d = self.d
        self.text(sc, 0.06, 0.318, "H₀: background only, b = 10 events expected", op,
                  anchor="lm", ink=dark(BLUE, 0.3), size=HEAD)
        self.text(sc, 0.06, 0.578, "H₁: signal too, s + b = 8 + 10 = 18 expected", op,
                  anchor="lm", ink=dark(RED, 0.3), size=HEAD)
        for i in range(6):
            x = 0.095 + 0.071 * i
            on = op * smoothstep(0.04 + 0.05 * i, 0.09 + 0.05 * i, u)
            self.cluster(sc, x, 0.42, int(d["h0"][i]), 0, on)
            self.text(sc, x, 0.505, "n = %d" % d["h0"][i], on, size=0.030)
            on = op * smoothstep(0.36 + 0.05 * i, 0.41 + 0.05 * i, u)
            self.cluster(sc, x, 0.68, int(d["h1_b"][i]), int(d["h1_s"][i]), on)
            self.text(sc, x, 0.765, "n = %d" % d["h1"][i], on, size=0.030)
        self.text(sc, 0.06, 0.845, "six pseudo-experiments of each: every dot one event",
                  op, anchor="lm")
        self.text(sc, 0.06, 0.89, "the detector reports only the total n", op * smoothstep(
            0.7, 0.8, u), anchor="lm", strong=True)
        chart = self.graph(((0.58, 0.94), (0.40, 0.78)), self.XR, self.YR)
        self.count_axes(sc, chart, op, "probability of counting n")
        self.pair(sc, chart, op, grow0=smoothstep(0.1, 0.4, u), grow1=smoothstep(0.42, 0.72, u))
        self.key(sc, chart.fx(21.5), 0.435, [("H₀: Poisson, mean 10", BLUE)], op)
        self.key(sc, chart.fx(21.5), 0.481, [("H₁: Poisson, mean 18", RED)],
                 op * smoothstep(0.42, 0.5, u))
        self.tag(sc, "dots: pseudo-experiments, seed %d; bars: exact" % S.ERR_SEED, op)

    def phase_2(self, sc, u, op):
        d, cut = self.d, self.n_c
        chart = self.graph(((0.08, 0.55), (0.42, 0.78)), self.XR, self.YR)
        self.count_axes(sc, chart, op)
        a_on, b_on, p_on = (smoothstep(a, a + 0.08, u) for a in (0.16, 0.50, 0.80))
        self.pair(sc, chart, op, cut=cut, lit=(a_on > 0.5, b_on > 0.5))
        x = chart.fx(cut - 0.5)
        self.vline(sc, x, 0.385, 0.78, dark(AMBER, 0.1), op * smoothstep(0.02, 0.12, u), 3.0)
        self.text(sc, x, 0.345, "reject H₀ if n ≥ %d" % cut, op * smoothstep(0.02, 0.12, u),
                  strong=True, size=HEAD)
        pm = d["pmf0"]
        x0 = 0.60
        self.lines(sc, x0, 0.345, [
            ("type I error: reject a true H₀", dark(ALPHA_INK, 0.35)),
            "α = P(n ≥ %d | b = 10)" % cut,
            "= %.4f + %.4f + %.4f + …" % (pm[cut], pm[cut + 1], pm[cut + 2])], op * a_on,
            step=0.05)
        self.text(sc, x0, 0.497, "= %.4f" % d["alpha_c"], op * smoothstep(0.3, 0.38, u),
                  anchor="lm", strong=True, size=BIG)
        self.lines(sc, x0, 0.58, [
            ("type II error: keep a false H₀", dark(BETA_INK, 0.25)),
            "β = P(n ≤ %d | s + b = 18)" % (cut - 1)], op * b_on, step=0.05)
        self.text(sc, x0, 0.682, "= %.4f" % d["beta_c"], op * smoothstep(0.62, 0.7, u),
                  anchor="lm", strong=True, size=BIG)
        self.text(sc, x0, 0.765, "power = 1 − β = %.4f" % (1 - d["beta_c"]), op * p_on,
                  anchor="lm", ink=dark(RED, 0.3), size=HEAD)
        self.text(sc, x0, 0.812, "the chance to see a real signal", op * p_on, anchor="lm")
        self.tag(sc, "exact: sums of Poisson probabilities", op)

    def cut_at(self, u):
        """The threshold shown at ``u`` of phase 3: down from 30 to 4, then back to the
        chosen one; and the lowest it has reached."""
        down = 30 - 26 * float(np.clip((u - 0.06) / 0.62, 0, 1))
        back = 4 + (self.n_c - 4) * smoothstep(0.74, 0.9, u)
        cut = int(round(down if u < 0.72 else back))
        return cut, int(round(down))

    def phase_3(self, sc, u, op):
        d = self.d
        cut, low = self.cut_at(u)
        chart = self.graph(((0.08, 0.50), (0.42, 0.76)), self.XR, self.YR)
        self.count_axes(sc, chart, op)
        self.pair(sc, chart, op, cut=cut)
        x = chart.fx(cut - 0.5)
        self.vline(sc, x, 0.42, 0.76, dark(AMBER, 0.1), op, 3.0)
        self.text(sc, 0.29, 0.345, "reject H₀ if n ≥ %d" % cut, op, strong=True, size=HEAD)
        self.box(sc, 0.19, 0.885, 150, 30, soft(ALPHA_INK, 0.6), op,
                 "α = %.4f" % d["alpha"][cut], border=ALPHA_INK, size=0.032)
        self.box(sc, 0.39, 0.885, 150, 30, soft(BETA_INK, 0.6), op,
                 "β = %.4f" % d["beta"][cut], border=BETA_INK, size=0.032)
        roc = self.graph(((0.64, 0.93), (0.42, 0.76)), (0, 1), (0, 1))
        self.axes(sc, roc, op, "α: chance of a false alarm", "power = 1 − β",
                  xticks=self.ticks((0, 0.5, 1)), yticks=self.ticks((0, 0.5, 1)), grid=False)
        self.dashed(sc, (roc.fx(0), roc.fy(0)), (roc.fx(1), roc.fy(1)), AXIS, op)
        cs = np.arange(low, 31)
        if len(cs) > 1:
            self.path(sc, roc, d["alpha"][cs], d["power"][cs], STRONG, op, 2.4)
        for c in cs:
            self.disk(sc, self.P(roc.fx(d["roc_mc"][c, 0]), roc.fy(d["roc_mc"][c, 1])), 3.4,
                      GREEN, op, lift=1.1, outline=0.0)
        self.disk(sc, self.P(roc.fx(d["alpha"][cut]), roc.fy(d["power"][cut])), 6.5, AMBER, op,
                  glow=0.4, lift=1.6, outline=0.15)
        self.text(sc, 0.785, 0.345, "one point per threshold: the ROC curve", op, size=0.030)
        self.text(sc, roc.fx(0.56), roc.fy(0.2), "a coin toss", op * smoothstep(0.3, 0.4, u),
                  anchor="lm")
        self.tag(sc, "line: exact; green dots: pseudo-experiments", op)

    def arrived(self, u):
        """How many of the 10,000 pseudo-experiments have been counted at ``u``."""
        total = len(self.count)
        if u < 0.34:
            return int(np.clip((u - 0.04) / 0.05, 0, 6))
        return int(min(total, round(6 * (total / 6.0) ** min((u - 0.34) / 0.38, 1.0))))

    def phase_4(self, sc, u, op):
        cut = self.n_c
        k = self.arrived(u)
        c = self.cum[k]
        xs, ys = (0.61, 0.845), (0.50, 0.71)
        self.text(sc, xs[0], 0.375, "keep H₀  (n ≤ %d)" % (cut - 1), op, strong=True, size=HEAD)
        self.text(sc, xs[1], 0.375, "reject H₀  (n ≥ %d)" % cut, op, strong=True, size=HEAD)
        names = (("correct", GREEN), ("type I error", ALPHA_INK), ("type II error", BETA_INK),
                 ("correct: a discovery", GREEN))
        done = smoothstep(0.76, 0.84, u)
        rate = ("", "α ", "β ", "")
        for j in range(4):
            r, col = divmod(j, 2)
            name, ink = names[j]
            self.box(sc, xs[col], ys[r], 174, 84, soft(ink, 0.68), op, border=ink, radius=7)
            self.text(sc, xs[col], ys[r] - 0.052, name, op, ink=dark(ink, 0.4), size=0.030)
            self.text(sc, xs[col], ys[r] - 0.002, commas(c[j]), op, strong=True, size=0.046)
            share = c[j] / max(c[2 * r] + c[2 * r + 1], 1)
            exact = self.study.panels[2].data["exact"][r, col] / S.ERR_N
            note = "%s;  exact %s%s" % (pct(share), rate[j], pct(exact))
            self.text(sc, xs[col], ys[r] + 0.05, note, op * done, size=0.029)
        for r, (name, ink) in enumerate((("H₀ true", BLUE), ("H₁ true", RED))):
            self.text(sc, 0.495, ys[r] - 0.02, name, op, anchor="rm", ink=dark(ink, 0.3),
                      size=HEAD)
            self.text(sc, 0.495, ys[r] + 0.025, "%s runs" % commas(c[2 * r] + c[2 * r + 1]), op,
                      anchor="rm", size=0.029)
        # the experiment just counted
        if k:
            j = k - 1
            truth, n = int(self.truth[j]), int(self.count[j])
            ink = RED if truth else BLUE
            self.text(sc, 0.06, 0.375, "pseudo-experiment %s" % commas(k), op, anchor="lm",
                      strong=True, size=HEAD)
            self.box(sc, 0.17, 0.47, 210, 40, soft(ink, 0.6), op,
                     "H₁ true: s + b = 18" if truth else "H₀ true: b = 10", border=ink,
                     size=0.031)
            self.text(sc, 0.17, 0.575, "n = %d" % n, op, strong=True, size=0.05)
            verdict = "≥ %d: reject H₀" % cut if n >= cut else "≤ %d: keep H₀" % (cut - 1)
            self.text(sc, 0.17, 0.655, verdict, op, size=HEAD)
            cell = 2 * truth + (n >= cut)
            self.box(sc, 0.17, 0.74, 210, 38, soft(names[cell][1], 0.6), op, names[cell][0],
                     border=names[cell][1], size=0.031)
            if k <= 6:
                r, col = divmod(cell, 2)
                self.pulse(sc, (0.285, 0.74), (xs[col] - 0.1, ys[r]),
                           (u - 0.04) / 0.05 - (k - 1) if k < 6 else (u - 0.29) / 0.05,
                           names[cell][1], op, bend=0.05, r=6.0)
        self.text(sc, 0.06, 0.86, "each run: draw n from its hypothesis, compare with the "
                  "threshold", op, anchor="lm")
        self.tag(sc, "pseudo-experiments: %s per hypothesis, seed %d"
                 % (commas(S.ERR_N), S.ERR_SEED), op)

    def phase_5(self, sc, u, op):
        d = self.d
        b, sb = S.ERR_B, S.ERR_B + S.ERR_S
        for row, i in enumerate(self.shown):
            t = float(d["runs"][i])
            top = 0.375 + 0.152 * row
            on = op * smoothstep(0.03 + 0.2 * row, 0.13 + 0.2 * row, u)
            chart = self.graph(((0.08, 0.46), (top, top + 0.095)), (0, 34), (0, 0.27))
            nmax = int(34 * t)
            n = np.arange(nmax + 1)
            for mean, ink in ((b * t, BLUE), (sb * t, RED)):
                pmf = S.poisson_pmf(mean, nmax) * t
                keep = pmf > 0.002                     # not along the floor
                self.path(sc, chart, n[keep] / t, pmf[keep], ink, on, 2.8)
            self.polyline(sc, [self.P(0.08, top + 0.095), self.P(0.46, top + 0.095)], 1.2,
                          AXIS, 0.45 * on)
            x = chart.fx((d["cuts"][i] - 0.5) / t)
            self.vline(sc, x, top - 0.005, top + 0.095, dark(AMBER, 0.1), on, 2.6)
            self.text(sc, 0.08, top - 0.03, "%s: reject if n ≥ %d;  power %s"
                      % ("1 run" if t == 1 else "%d runs" % t, d["cuts"][i],
                         pct(d["power_run"][i])), on, anchor="lm", size=0.030)
        for v in (0, 10, 20, 30):
            self.text(sc, 0.08 + 0.38 * v / 34.0, 0.80, "%d" % v, op, size=0.029)
        self.text(sc, 0.27, 0.845, "events per run, n / runs", op, size=0.030)
        self.key(sc, 0.08, 0.89, [("H₀", BLUE), ("H₁", RED), ("threshold, α ≤ 5%", AMBER)],
                 op, across=True)
        chart = self.graph(((0.60, 0.93), (0.40, 0.76)), (0, 6.2), (0, 1.04))
        self.axes(sc, chart, op, "amount of data (runs)", "power at α ≤ 5%",
                  xticks=self.ticks((0, 2, 4, 6), "%d"),
                  yticks=self.ticks((0, 0.5, 1.0)), grid=False)
        go = smoothstep(0.25, 0.8, u)
        self.path(sc, chart, d["runs"], d["power_run"], STRONG, op, 2.4, go=go)
        for i in range(int(round(go * len(d["runs"])))):
            lit = i in self.shown
            self.disk(sc, self.P(chart.fx(d["runs"][i]), chart.fy(d["power_mc"][i])),
                      6.0 if lit else 3.6, AMBER if lit else GREEN, op, lift=1.3,
                      outline=0.1)
        self.text(sc, chart.fx(2.5), chart.fy(0.38), "steps: n is a whole number,", op
                  * smoothstep(0.8, 0.9, u), anchor="lm")
        self.text(sc, chart.fx(2.5), chart.fy(0.38) + 0.042, "so α sits a little under 5%",
                  op * smoothstep(0.8, 0.9, u), anchor="lm")
        self.tag(sc, "lines: exact; dots: pseudo-experiments", op)


# ====================================================================== 142: p-values
OBS = AMBER


class PValues(Tests):
    """Ten measurements against a prediction and their chi-square; the distribution of
    chi-square if H0 is true, from pseudo-experiments and exactly; the tail beyond the
    observation as the p-value and as Z; the p-values of null experiments, uniform; and
    a population of experiments in which most 'significant' results are not rare."""

    TITLE = "What a p-value is, and is not"
    PHASES = (("one result", 11.0), ("if H₀ is true", 12.0), ("the p-value", 11.0),
              ("uniform", 10.0), ("not P(H₀)", 12.0))
    COLS, ROWS, CELL = 50, 20, 20

    def setup(self):
        d = self.d = S.pvalue_data()
        self.meta = self.study.metadata
        self.obs = float(self.meta["chi2_observed"])
        self.grid_counts = np.asarray(self.study.panels[3].data["on_screen"], int)
        self.pop_counts = np.asarray(self.study.panels[3].data["values"], int)
        hit = d["grid_p"] < S.PV_CUT
        real = d["grid_real"]
        self.textures = [
            self.population(np.where(real[:, None], RED, soft(BLUE, 0.25))),
            self.population(np.where(hit[:, None], np.where(real[:, None], RED, BLUE),
                                     np.where(real[:, None], soft(RED, 0.72),
                                              soft(BLUE, 0.80)))),
        ]
        self.xs = np.linspace(0.0, 40.0, 161)
        self.pdf = S.chi2_pdf(self.xs, S.PV_N)

    def population(self, colours):
        """The 1,000 experiments as a texture of round dots, 50 across and 20 down."""
        c = self.CELL
        yy, xx = np.mgrid[0:c, 0:c]
        dot = np.hypot(xx - (c - 1) / 2, yy - (c - 1) / 2) <= c * 0.36
        tex = np.empty((self.ROWS * c, self.COLS * c, 3), np.float32)
        tex[:] = _rgb(PAPER)
        for k, colour in enumerate(np.asarray(colours, np.float32)):
            r, col = divmod(k, self.COLS)
            tex[r * c:(r + 1) * c, col * c:(col + 1) * c][dot] = colour
        return tex

    def subtitle_for(self, k):
        m = self.meta
        g = self.grid_counts
        return [
            "ten measurements y, each with error σ = %g, against the prediction of H₀: "
            "μ = %g everywhere. Each pull (y − μ)/σ is squared; the sum is the test "
            "statistic, χ² = %.2f" % (S.PV_SIGMA, S.PV_MU, self.obs),
            "is %.1f large? Make data in which H₀ is true, %s times over, and compute χ² "
            "each time. The histogram settles onto the exact χ² curve for 10 degrees of "
            "freedom" % (self.obs, commas(S.PV_TOYS)),
            "the p-value: the share of H₀ experiments at least as extreme as ours. %d of "
            "%s, %.4f; exactly %.4f. As a Gaussian's one-sided tail that is Z = %.2fσ"
            % (m["tail_count"], commas(S.PV_TOYS), m["p_pseudo"], m["p_exact"], m["z"]),
            "if H₀ is true, every p-value is as likely as any other: %s of the %s null "
            "experiments have p < 0.05. A threshold on p is a choice of α, the type I "
            "error rate" % (pct(m["share_below_005"], 2), commas(S.PV_TOYS)),
            "p is not the chance that H₀ is true. If %d%% of experiments test a real "
            "effect (seen with power %s), then %d of the %d with p < 0.05 here are false "
            "alarms: %s, not 5%%"
            % (round(100 * S.PV_PRIOR), pct(m["power_exact"], 0), g[0, 1],
               g[0, 1] + g[1, 1], pct(m["false_alarm_share_on_screen"], 0)),
        ][k]

    def data_points(self, sc, chart, ys, op, shown=None, ink=STRONG, r=4.6):
        """Measurements with their error bars of one sigma."""
        dy = (chart.y1 - chart.y0) * S.PV_SIGMA / (chart.y_range[1] - chart.y_range[0])
        for i, y in enumerate(ys):
            if shown is not None and i >= shown:
                break
            x, fy = chart.fx(i + 1), chart.fy(float(y))
            self.polyline(sc, [self.P(x, fy - dy), self.P(x, fy + dy)], 2.0, ink, 0.8 * op)
            self.disk(sc, self.P(x, fy), r, ink, op, lift=1.3, outline=0.1)

    def phase_1(self, sc, u, op):
        d = self.d
        chart = self.graph(((0.09, 0.50), (0.40, 0.76)), (0.3, 10.7), (11.0, 31.0))
        self.axes(sc, chart, op, "measurement", "measured y  (error bar: σ = 2.0)",
                  xticks=self.ticks(range(1, 11), "%d"),
                  yticks=self.ticks((15, 20, 25, 30), "%d"), grid=False)
        y0 = chart.fy(S.PV_MU)
        self.polyline(sc, [self.P(chart.x0, y0), self.P(chart.x1, y0)], 2.8, BLUE, op)
        self.text(sc, chart.fx(8.4), chart.fy(13.2), "H₀: μ = 20", op, ink=dark(BLUE, 0.3),
                  size=HEAD)
        lo, hi = S.PV_RAISED[0] + 0.5, S.PV_RAISED[-1] + 1.5
        yt = chart.fy(d["truth"][S.PV_RAISED[0]])
        self.dashed(sc, (chart.fx(lo), yt), (chart.fx(hi), yt), RED, 2 * op, n=5, width=2.4)
        self.text(sc, chart.fx(8.4), chart.fy(28.6), "truth: 24 here", op, ink=dark(RED, 0.3))
        self.arrow(sc, (chart.fx(7.1), chart.fy(28.0)), (chart.fx(6.6), chart.fy(24.6)),
                   dark(RED, 0.2), op, 1.0, 1.8)
        k = float(np.clip((u - 0.05) / 0.55, 0, 1)) * S.PV_N
        self.data_points(sc, chart, d["y"], op, shown=int(math.ceil(k)))
        cols = (0.585, 0.675, 0.78, 0.895)
        for x, name in zip(cols, ("point", "y", "(y − 20) / 2", "squared")):
            self.text(sc, x, 0.345, name, op, size=0.030, strong=True)
        for i in range(S.PV_N):
            on = op * smoothstep(i, i + 0.6, k)
            y = 0.392 + 0.0385 * i
            raised = i in S.PV_RAISED
            ink = dark(RED, 0.3) if raised else None
            z = d["pulls"][i]
            for x, s in zip(cols, ("%d" % (i + 1), "%.1f" % d["y"][i], signed(z),
                                   "%.2f" % (z * z))):
                self.text(sc, x, y, s, on, size=0.029, ink=ink)
        total = op * smoothstep(0.66, 0.76, u)
        self.polyline(sc, [self.P(0.55, 0.762), self.P(0.94, 0.762)], 1.4, AXIS, 0.45 * total)
        self.text(sc, 0.895, 0.80, "%.2f" % self.obs, total, strong=True, size=BIG)
        self.text(sc, 0.845, 0.80, "χ² = sum of the squares =", total, anchor="rm", size=0.031)
        self.text(sc, 0.09, 0.885, "the three points made higher are in red", op, anchor="lm")
        self.tag(sc, "pseudo-experiments: one data set, seed %d" % S.PV_SEED, op)

    def toys_at(self, u):
        """How many H0 pseudo-experiments are in the histogram at ``u`` of phase 2."""
        if u < 0.1:
            return max(1, int(u / 0.1 * 20))
        return int(round(20 * (S.PV_TOYS / 20.0) ** min((u - 0.1) / 0.5, 1.0)))

    def null_bars(self, sc, chart, n, op, ink=None):
        d = self.d
        hist = d["hist"] if n >= S.PV_TOYS else np.histogram(d["null"][:n], d["edges"])[0]
        self.bars(sc, chart, d["edges"][:-1] + 0.5, hist / float(n), 1.0,
                  soft(BLUE, 0.25) if ink is None else ink, op)

    def phase_2(self, sc, u, op):
        d = self.d
        n = self.toys_at(u)
        j = n - 1 if n <= 40 else int(u * 400) % 40
        small = self.graph(((0.08, 0.36), (0.42, 0.68)), (0.3, 10.7), (11.0, 31.0))
        self.axes(sc, small, op, "", "", xticks=[], yticks=self.ticks((15, 20, 25), "%d"),
                  grid=False)
        y0 = small.fy(S.PV_MU)
        self.polyline(sc, [self.P(small.x0, y0), self.P(small.x1, y0)], 2.8, BLUE, op)
        self.data_points(sc, small, d["null_y"][j], op, ink=dark(BLUE, 0.45), r=4.0)
        self.text(sc, 0.22, 0.345, "data made with H₀ true", op, strong=True, size=HEAD)
        self.text(sc, 0.22, 0.735, "pseudo-experiment %s" % commas(j + 1 if n <= 40 else n),
                  op, size=0.031)
        self.text(sc, 0.22, 0.785, "χ² = %.1f" % d["null"][j if n <= 40 else n - 1], op,
                  strong=True, size=BIG)
        self.arrow(sc, (0.375, 0.56), (0.455, 0.56), AXIS, op, 1.0, 2.6)
        chart = self.graph(((0.50, 0.93), (0.42, 0.76)), (0, 40), (0, 0.112))
        self.axes(sc, chart, op, "χ²", "share of experiments per unit of χ²",
                  xticks=self.ticks((0, 10, 20, 30, 40), "%d"),
                  yticks=self.ticks((0, 0.05, 0.10), "%.2f"), grid=False)
        self.null_bars(sc, chart, n, op)
        self.text(sc, chart.fx(39.5), 0.455, "%s pseudo-experiments" % commas(n), op,
                  anchor="rm", ink=dark(BLUE, 0.3), size=0.031)
        exact = smoothstep(0.64, 0.8, u)
        self.path(sc, chart, self.xs, self.pdf, STRONG, op, 2.6, go=exact)
        self.text(sc, chart.fx(39.5), 0.50, "curve: exact χ², 10 degrees", op * smoothstep(
            0.66, 0.74, u), anchor="rm", size=0.031)
        seen = op * smoothstep(0.84, 0.92, u)
        x = chart.fx(self.obs)
        self.vline(sc, x, 0.60, 0.76, dark(OBS, 0.1), seen, 3.0)
        self.text(sc, x, 0.57, "ours: %.1f" % self.obs, seen, ink=dark(OBS, 0.4), size=HEAD)
        self.tag(sc, "bars: pseudo-experiments, seed %d; curve: exact" % S.PV_SEED, op)

    def phase_3(self, sc, u, op):
        d, m = self.d, self.meta
        chart = self.graph(((0.08, 0.50), (0.42, 0.76)), (0, 40), (0, 0.112))
        self.axes(sc, chart, op, "χ² if H₀ is true", "",
                  xticks=self.ticks((0, 10, 20, 30, 40), "%d"), yticks=[], grid=False)
        self.null_bars(sc, chart, S.PV_TOYS, op)
        self.path(sc, chart, self.xs, self.pdf, STRONG, op, 2.4)
        x = chart.fx(self.obs)
        self.vline(sc, x, 0.655, 0.76, dark(OBS, 0.1), op, 3.0)
        # the tail, drawn ten times taller: whole bins, and the bin the observation splits
        zoom = self.graph(((0.285, 0.49), (0.40, 0.56)), (20, 40), (0, 0.0112))
        on = op * smoothstep(0.06, 0.2, u)
        self.rect(sc, self.P(0.3875, 0.48), self.span(0.205)[0] + 10, self.span(0, 0.16)[1] + 10,
                  PAPER, on, border=DIM, radius=4, lift=0.2)
        edges, hist = d["edges"], d["hist"]
        k = int(self.obs)
        low = int(((d["null"] >= k) & (d["null"] < self.obs)).sum())
        frac = self.obs - k
        cs = np.arange(20, 40)
        inks = [soft(BLUE, 0.25) if c < k else OBS for c in cs]
        hs = [0.0 if c == k else hist[c] / S.PV_TOYS for c in cs]
        self.bars(sc, zoom, edges[cs] + 0.5, hs, 1.0, inks, on)
        self.bars(sc, zoom, [k + frac / 2, k + (1 + frac) / 2],
                  [low / frac / S.PV_TOYS, (hist[k] - low) / (1 - frac) / S.PV_TOYS],
                  1.0, [soft(BLUE, 0.25), OBS], on, gap=0.0)
        self.bars(sc, zoom, [k + frac / 2], [low / frac / S.PV_TOYS], frac, soft(BLUE, 0.25),
                  on, gap=0.0)
        self.vline(sc, zoom.fx(self.obs), 0.40, 0.56, dark(OBS, 0.1), on, 2.4)
        for v in (20, 30, 40):
            self.text(sc, zoom.fx(v), 0.595, "%d" % v, on, size=0.029)
        self.text(sc, 0.3875, 0.345, "the tail, 10 times taller", on, size=0.030)
        self.text(sc, x + 0.012, 0.685, "ours: %.2f" % self.obs, op, anchor="lm",
                  ink=dark(OBS, 0.4))
        x0 = 0.57
        self.lines(sc, x0, 0.345, ["p-value: if H₀ is true, how often is",
                                   "χ² at least as large as ours?"], op, step=0.042,
                   size=0.031)
        self.text(sc, x0, 0.455, "pseudo-experiments: %d / %s = %.4f"
                  % (m["tail_count"], commas(S.PV_TOYS), m["p_pseudo"]),
                  op * smoothstep(0.2, 0.3, u), anchor="lm", ink=dark(OBS, 0.45), size=0.031)
        self.text(sc, x0, 0.505, "exact area of the tail:  p = %.4f" % m["p_exact"],
                  op * smoothstep(0.36, 0.46, u), anchor="lm", strong=True, size=0.033)
        late = op * smoothstep(0.55, 0.68, u)
        gauss = self.graph(((0.60, 0.92), (0.635, 0.80)), (-4, 4), (0, 0.42))
        self.axes(sc, gauss, late, "", "", xticks=self.ticks((-2, 0, 2, 4), "%d"), yticks=[],
                  grid=False)
        zs = np.linspace(-4, 4, 97)
        self.path(sc, gauss, zs, S.norm_pdf(zs), STRONG, late, 2.4)
        tail = np.linspace(m["z"], 4.0, 13)
        self.bars(sc, gauss, 0.5 * (tail[1:] + tail[:-1]), S.norm_pdf(0.5 * (tail[1:]
                  + tail[:-1])), tail[1] - tail[0], OBS, late, gap=0.0)
        self.vline(sc, gauss.fx(m["z"]), 0.70, 0.80, dark(OBS, 0.1), late, 2.4)
        self.text(sc, x0, 0.585, "the same area as a Gaussian's tail:", late, anchor="lm",
                  size=0.031)
        self.text(sc, 0.935, 0.66, "Z = %.2fσ" % m["z"], op * smoothstep(0.7, 0.8, u),
                  anchor="rm", strong=True, size=BIG)
        self.tag(sc, "count: pseudo-experiments; p and Z: exact", op)

    def phase_4(self, sc, u, op):
        d, m = self.d, self.meta
        chart = self.graph(((0.08, 0.45), (0.42, 0.74)), (0, 32), (0, 0.112))
        self.axes(sc, chart, op, "χ² if H₀ is true", "", xticks=self.ticks((0, 10, 20, 30),
                  "%d"), yticks=[], grid=False)
        xs = self.xs[self.xs <= 32]
        cutx = d["quantiles"][-1]                     # the slice furthest out: p < 0.05
        tail = np.linspace(cutx, 32.0, 25)
        mid = 0.5 * (tail[1:] + tail[:-1])
        self.bars(sc, chart, mid, S.chi2_pdf(mid, S.PV_N), tail[1] - tail[0], OBS, op, gap=0.0)
        self.path(sc, chart, xs, self.pdf[:len(xs)], STRONG, op, 2.4)
        cuts = smoothstep(0.05, 0.4, u) * len(d["quantiles"])
        for i, q in enumerate(d["quantiles"]):
            if i < cuts:
                self.vline(sc, chart.fx(q), chart.fy(float(S.chi2_pdf(q, S.PV_N))), 0.74,
                           dark(BLUE, 0.2), op, 1.6)
        self.text(sc, 0.265, 0.345, "the exact curve, cut into", op, size=0.031)
        self.text(sc, 0.265, 0.387, "20 slices of equal area", op, size=0.031, strong=True)
        self.text(sc, chart.fx(24.5), chart.fy(0.03), "p < 0.05", op, ink=dark(OBS, 0.4),
                  size=0.031)
        self.arrow(sc, (0.47, 0.58), (0.54, 0.58), AXIS, op, smoothstep(0.3, 0.45, u), 2.6)
        hist = self.graph(((0.60, 0.93), (0.42, 0.74)), (0, 1), (0, 1250))
        self.axes(sc, hist, op, "p-value", "",
                  xticks=self.ticks((0, 0.5, 1)), yticks=self.ticks((0, 500, 1000), "%d"),
                  grid=False)
        grow = smoothstep(0.35, 0.72, u)
        inks = [OBS] + [soft(BLUE, 0.25)] * 19
        self.bars(sc, hist, np.arange(20) / 20.0 + 0.025, d["p_hist"], 0.05, inks, op, grow=grow)
        self.text(sc, 0.765, 0.345, "p of %s H₀ experiments" % commas(S.PV_TOYS), op,
                  size=0.031)
        self.text(sc, 0.765, 0.387, "flat: 1,000 expected per slice", op * smoothstep(
            0.72, 0.8, u), size=0.031, strong=True)
        self.text(sc, 0.08, 0.875, "%s of %s have p < 0.05: %s" % (
            commas(d["p_hist"][0]), commas(S.PV_TOYS), pct(m["share_below_005"], 2)),
            op * smoothstep(0.78, 0.88, u), anchor="lm", ink=dark(OBS, 0.45), size=HEAD)
        self.tag(sc, "bars: pseudo-experiments; curve: exact", op)

    def phase_5(self, sc, u, op):
        m = self.meta
        g, pop = self.grid_counts, self.pop_counts
        box = ((0.06, 0.60), (0.395, 0.755))
        cut = smoothstep(0.3, 0.42, u)
        self.picture(sc, self.textures[0], box, op * (1 - cut))
        self.picture(sc, self.textures[1], box, op * cut, frame=None)
        self.key(sc, 0.068, 0.345, [("%d test a real effect" % g[1].sum(), RED),
                                    ("%d test nothing: H₀ is true" % g[0].sum(), BLUE)],
                 op, across=True, size=0.031)
        self.text(sc, 0.06, 0.80, "kept bright: the %d with p < 0.05" % g[:, 1].sum(),
                  op * cut, anchor="lm", size=0.031, strong=True)
        self.text(sc, 0.06, 0.845, "%d of the %d real effects (the power), %d of the %d "
                  "others (α)" % (g[1, 1], g[1].sum(), g[0, 1], g[0].sum()), op * cut,
                  anchor="lm")
        late = op * smoothstep(0.5, 0.6, u)
        x0 = 0.66
        self.text(sc, x0, 0.40, "among p < 0.05:", late, anchor="lm", strong=True, size=HEAD)
        full = self.span(0.16)[0] / float(g[:, 1].sum())
        self.hbar(sc, x0, 0.46, full * g[1, 1], RED, late, "", "%d real" % g[1, 1],
                  (u - 0.5) / 0.12, thick=20.0)
        self.hbar(sc, x0, 0.515, full * g[0, 1], BLUE, late, "", "%d false alarms" % g[0, 1],
                  (u - 0.56) / 0.12, thick=20.0)
        last = op * smoothstep(0.7, 0.8, u)
        self.text(sc, x0, 0.59, "%d / %d = %s false" % (
            g[0, 1], g[:, 1].sum(), pct(m["false_alarm_share_on_screen"], 0)), last,
            anchor="lm", strong=True, size=BIG)
        self.lines(sc, x0, 0.65, [
            "all %s: %s" % (commas(pop.sum()), pct(m["false_alarm_share_pseudo"], 0)),
            "exact: %s" % pct(m["false_alarm_share_exact"], 0),
            ("p < 0.05 is not", dark(OBS, 0.45)),
            ("'H₀ has a 5% chance'", dark(OBS, 0.45))], last, step=0.044, size=0.031)
        self.tag(sc, "pseudo-experiments: %d%% real effects assumed"
                 % round(100 * S.PV_PRIOR), op)


# ====================================================================== 143: Neyman-Pearson
STAT_INKS = (TEAL, VIOLET, GREEN)             # x alone, the sum, the likelihood ratio
FORMULAS = ("t = x", "t = x + y", "t = 1.5 x + 0.6 y")


class NeymanPearson(Tests):
    """Events with two measured values under two hypotheses; one event's two likelihoods
    and their ratio; three tests at the same false-alarm rate, drawn as lines in the
    plane; their ROC curves; and the power of every direction, which peaks at the
    likelihood ratio's."""

    TITLE = "The likelihood ratio is the best test"
    PHASES = (("events", 9.0), ("one event", 12.0), ("three tests", 13.0),
              ("ROC", 11.0), ("directions", 12.0))
    XR, YR = (-3.6, 5.4), (-6.2, 8.8)

    def setup(self):
        d = self.d = S.np_data()
        self.meta = self.study.metadata
        xy = np.asarray(self.study.panels[0].data["xy"], float)
        n = S.NP_SHOWN
        self.p0, self.p1 = xy[:n], xy[n:]
        self.power_mc = np.asarray(self.study.panels[2].data["values"], float)
        self.power_exact = np.asarray(self.study.panels[2].data["exact"], float)
        x, y = (float(v) for v in d["event"])
        sx, sy = S.NP_SIGMA

        def g(v, mean, sd):
            return float(S.norm_pdf((v - mean) / sd) / sd)
        self.factors = ((g(x, 0.0, sx), g(y, 0.0, sy)),
                        (g(x, S.NP_SHIFT[0], sx), g(y, S.NP_SHIFT[1], sy)))
        edges = np.linspace(-8.0, 8.0, 65)
        self.lr_edges = edges
        self.lr_hist = [np.histogram(t, edges)[0] / float(len(t)) / (edges[1] - edges[0])
                        for t in (d["t0_lr"], d["t1_lr"])]

    def subtitle_for(self, k):
        d, m = self.d, self.meta
        pe, pm = m["power_exact"], m["power_pseudo"]
        return [
            "each event has two measured values. Under H₀ (background) both centre on 0; "
            "under H₁ (signal) x centres on %g and y on %g. x is known to ±%g, y only to "
            "±%g" % (S.NP_SHIFT + S.NP_SIGMA),
            "the likelihood L of an event is its chance density under a hypothesis. This "
            "event is %.1f times likelier under H₁ than H₀; the log of the ratio weighs x "
            "by 1.5 and y by only 0.6" % (d["event_l1"] / d["event_l0"]),
            "three tests, each cut to let through α = 5%% of H₀. Of %s H₁ events they "
            "catch %s (x alone), %s (x + y) and %s (the likelihood ratio)"
            % (commas(S.NP_N), pct(pm[0]), pct(pm[1]), pct(pm[2])),
            "at every α the likelihood ratio has the highest power: its ROC curve lies "
            "above the others (exact curves; dots from %s pseudo-events of each kind). "
            "At 5%%: %s, %s, %s" % (commas(S.NP_N), pct(pe[0]), pct(pe[1]), pct(pe[2])),
            "turn the cut through every direction: the power peaks at %.0f°, the "
            "likelihood ratio's. Any increasing function of the ratio, such as "
            "L₁/(L₀ + L₁), passes the very same events" % m["best_angle"],
        ][k]

    # ------------------------------------------------------------------ the plane
    def at(self, box, x, y):
        (x0, x1), (y0, y1) = box
        return (x0 + (x1 - x0) * (x - self.XR[0]) / (self.XR[1] - self.XR[0]),
                y1 - (y1 - y0) * (y - self.YR[0]) / (self.YR[1] - self.YR[0]))

    def plane(self, sc, box, op, n0=None, n1=None, colours=None, shape=(156, 255), fat=3,
              names=True):
        """The first events of each kind in the (x, y) plane; ``colours`` overrides the
        colour of each (H0 events first)."""
        n0 = S.NP_SHOWN if n0 is None else n0
        n1 = S.NP_SHOWN if n1 is None else n1
        pts = np.concatenate([self.p0[:n0], self.p1[:n1]])
        if colours is None:
            colours = np.concatenate([np.tile(_rgb(BLUE), (n0, 1)), np.tile(_rgb(RED), (n1, 1))])
        tex = scatter(pts[:, 0], pts[:, 1], self.XR, self.YR, shape, colours, fat=fat)
        self.picture(sc, tex, box, op)
        (x0, x1), (y0, y1) = box
        if names:
            self.text(sc, (x0 + x1) / 2, y1 + 0.035, "x", op, size=0.031)
            self.text(sc, x0 - 0.022, (y0 + y1) / 2, "y", op, size=0.031)

    def cut_line(self, sc, box, w, c, ink, op, width=3.0):
        ends = clip_line(w, c, self.XR, self.YR)
        if ends is not None:
            self.polyline(sc, [self.P(*self.at(box, *e), lift=0.9) for e in ends], width, ink, op)

    def ellipse(self, sc, box, mean, k, ink, op):
        a = np.linspace(0, 2 * np.pi, 49)
        pts = [self.P(*self.at(box, mean[0] + k * S.NP_SIGMA[0] * math.cos(t),
                               mean[1] + k * S.NP_SIGMA[1] * math.sin(t)), lift=0.8) for t in a]
        self.polyline(sc, pts, 2.2, ink, op)

    def passing(self, w, c):
        """Colours of the drawn events for the test ``w . (x, y) > c``: bright where an
        event passes, pale where it does not."""
        a, b = self.p0 @ w > c, self.p1 @ w > c
        return np.concatenate([np.where(a[:, None], _rgb(dark(BLUE, 0.25)), _rgb(soft(BLUE, 0.7))),
                               np.where(b[:, None], _rgb(RED), _rgb(soft(RED, 0.7)))])

    BOX = ((0.09, 0.50), (0.385, 0.80))

    def phase_1(self, sc, u, op):
        n0 = int(S.NP_SHOWN * smoothstep(0.05, 0.4, u))
        n1 = int(S.NP_SHOWN * smoothstep(0.4, 0.75, u))
        self.plane(sc, self.BOX, op, n0, n1)
        for mean, ink in (((0.0, 0.0), BLUE), (S.NP_SHIFT, RED)):
            self.disk(sc, self.P(*self.at(self.BOX, *mean), lift=1.2), 6.5, PAPER, op,
                      ring=dark(ink, 0.45), ring_w=3.0, lift=1.5, outline=0.0)
        self.text(sc, 0.295, 0.345, "every dot: one event, two measured values", op,
                  size=0.031)
        x0 = 0.57
        self.text(sc, x0, 0.40, "H₀: background", op, anchor="lm", ink=dark(BLUE, 0.3), size=BIG)
        self.text(sc, x0, 0.45, "x = 0 ± 1,   y = 0 ± 2", op, anchor="lm", size=HEAD)
        on = op * smoothstep(0.4, 0.5, u)
        self.text(sc, x0, 0.545, "H₁: signal", on, anchor="lm", ink=dark(RED, 0.3), size=BIG)
        self.text(sc, x0, 0.595, "x = 1.5 ± 1,   y = 2.4 ± 2", on, anchor="lm", size=HEAD)
        late = op * smoothstep(0.78, 0.88, u)
        self.lines(sc, x0, 0.70, ["a test needs one number t(x, y)",
                                  "to cut on. Which is best?"], late, step=0.046, size=HEAD)
        self.tag(sc, "pseudo-experiments: %d events of each kind, seed %d"
                 % (S.NP_SHOWN, S.NP_SEED), op)

    def phase_2(self, sc, u, op):
        d = self.d
        pale = np.concatenate([np.tile(_rgb(soft(BLUE, 0.6)), (S.NP_SHOWN, 1)),
                               np.tile(_rgb(soft(RED, 0.6)), (S.NP_SHOWN, 1))])
        self.plane(sc, self.BOX, op, colours=pale)
        for mean, ink in (((0.0, 0.0), BLUE), (S.NP_SHIFT, RED)):
            for k in (1, 2):
                self.ellipse(sc, self.BOX, mean, k, dark(ink, 0.15), op)
        x, y = (float(v) for v in d["event"])
        self.disk(sc, self.P(*self.at(self.BOX, x, y), lift=1.4), 7.5, AMBER, op, glow=0.4,
                  lift=1.8, outline=0.2)
        self.text(sc, 0.295, 0.345, "ellipses: 1σ and 2σ of each hypothesis", op, size=0.031)
        (a0, b0), (a1, b1) = self.factors
        x0 = 0.57
        self.text(sc, x0, 0.345, "one event:  x = %.2f,  y = %.2f" % (x, y), op, anchor="lm",
                  ink=dark(AMBER, 0.45), size=HEAD)
        on = op * smoothstep(0.1, 0.2, u)
        self.lines(sc, x0, 0.41, [("L₀ = G(x; 0, 1) × G(y; 0, 2)", dark(BLUE, 0.3)),
                                  "= %.4f × %.4f = %.5f" % (a0, b0, d["event_l0"])], on,
                   step=0.043, size=0.031)
        on = op * smoothstep(0.3, 0.4, u)
        self.lines(sc, x0, 0.52, [("L₁ = G(x; 1.5, 1) × G(y; 2.4, 2)", dark(RED, 0.3)),
                                  "= %.4f × %.4f = %.5f" % (a1, b1, d["event_l1"])], on,
                   step=0.043, size=0.031)
        on = op * smoothstep(0.5, 0.6, u)
        self.text(sc, x0, 0.64, "L₁ / L₀ = %.1f" % (d["event_l1"] / d["event_l0"]), on,
                  anchor="lm", strong=True, size=BIG)
        on = op * smoothstep(0.68, 0.78, u)
        t = float(d["w"][2] @ d["event"] + d["constant"])
        self.lines(sc, x0, 0.715, ["ln(L₁ / L₀) = 1.5 x + 0.6 y − 1.845",
                                   ("= %.2f" % t, STRONG)], on, step=0.043, size=0.031)
        for c in (-2.0, 1.0, 4.0, 7.0):                # lines of equal ratio
            self.cut_line(sc, self.BOX, d["w"][2], c, GREEN, 0.6 * on, 2.0)
        self.text(sc, 0.295, 0.885, "green: lines of equal ratio.  G: a Gaussian density",
                  on, size=0.030)
        self.tag(sc, "exact: two Gaussian densities multiplied", op)

    def phase_3(self, sc, u, op):
        d = self.d
        self.text(sc, 0.5, 0.33, "each line lets through 5% of H₀ (dark blue, right of it): "
                  "how much of H₁?", op, size=0.031)
        for j in range(3):
            cx = 0.20 + 0.30 * j
            box = ((cx - 0.125, cx + 0.125), (0.455, 0.755))
            w = d["w"][j]
            go = smoothstep(0.06 + 0.22 * j, 0.26 + 0.22 * j, u)
            sd = d["cuts"][j] / float(S.norm_isf(S.NP_ALPHA))
            c = d["cuts"][j] + (1 - go) * 3.0 * sd
            self.plane(sc, box, op, colours=self.passing(w, c), shape=(130, 190), names=False)
            self.cut_line(sc, box, w, c, dark(STAT_INKS[j], 0.1), op, 3.4)
            self.text(sc, cx, 0.375, S.NP_NAMES[j], op, ink=dark(STAT_INKS[j], 0.3), size=HEAD)
            self.text(sc, cx, 0.416, FORMULAS[j], op, size=0.030)
            on = op * smoothstep(0.24 + 0.22 * j, 0.3 + 0.22 * j, u)
            self.text(sc, cx, 0.80, "power %s" % pct(self.power_mc[j]), on, strong=True,
                      size=BIG)
            self.text(sc, cx, 0.845, "exact %s" % pct(self.power_exact[j]), on, size=0.030)
        self.tag(sc, "pseudo-experiments: %s events each (%d drawn)"
                 % (commas(S.NP_N), S.NP_SHOWN), op)

    def phase_4(self, sc, u, op):
        d = self.d
        go = smoothstep(0.08, 0.7, u)
        left = self.graph(((0.08, 0.44), (0.42, 0.76)), (-8, 8), (0, 0.24))
        self.axes(sc, left, op, "t = ln(L₁ / L₀)", "", xticks=self.ticks((-6, -3, 0, 3, 6),
                  "%d"), yticks=[], grid=False)
        mid = 0.5 * (self.lr_edges[1:] + self.lr_edges[:-1])
        self.bars(sc, left, mid - 0.0625, self.lr_hist[0], 0.125, BLUE, op, gap=0.0)
        self.bars(sc, left, mid + 0.0625, self.lr_hist[1], 0.125, RED, op, gap=0.0)
        half = int(np.searchsorted(d["alphas"], 0.5 + 1e-9))      # alpha up to 0.5
        k = max(int(round(go * half)), 1)                         # the alpha reached
        alpha = float(d["alphas"][k - 1])
        cut = float(S.norm_isf(alpha)) * d["d"][2] - d["d"][2] ** 2 / 2
        self.vline(sc, left.fx(cut), 0.40, 0.76, dark(AMBER, 0.1), op, 3.0)
        self.key(sc, 0.09, 0.345, [("H₀", BLUE), ("H₁", RED), ("the cut slides", AMBER)], op,
                 across=True)
        self.text(sc, 0.26, 0.885, "right of the cut: α = %.2f of H₀" % alpha, op,
                  size=0.031)
        roc = self.graph(((0.57, 0.93), (0.42, 0.76)), (0, 0.5), (0, 1))
        self.axes(sc, roc, op, "α: share of H₀ let through", "power: share of H₁ caught",
                  xticks=self.ticks((0, 0.1, 0.2, 0.3, 0.4, 0.5)),
                  yticks=self.ticks((0, 0.5, 1)), grid=False)
        for j in range(3):
            self.path(sc, roc, d["alphas"][:k], d["roc_exact"][j][:k], STAT_INKS[j], op, 2.8)
            for i in range(10, k, 6):
                self.disk(sc, self.P(roc.fx(d["alphas"][i]), roc.fy(d["roc_mc"][j][i])), 3.6,
                          dark(STAT_INKS[j], 0.25), op, lift=1.2, outline=0.0)
        self.key(sc, roc.fx(0.27), roc.fy(0.36), [(S.NP_NAMES[j], STAT_INKS[j])
                                                  for j in (2, 1, 0)], op, step=0.044)
        self.tag(sc, "curves: exact; dots and bars: pseudo-experiments", op)

    def angle_at(self, u):
        a = -60.0 + 150.0 * smoothstep(0.05, 0.6, u)
        return a + (self.d["lr_angle"] - 90.0) * smoothstep(0.64, 0.78, u)

    def phase_5(self, sc, u, op):
        d, m = self.d, self.meta
        a = self.angle_at(u)
        w = np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
        box = ((0.09, 0.43), (0.40, 0.76))
        self.plane(sc, box, op, colours=self.passing(w, S.exact_cut(w, S.NP_ALPHA)),
                   shape=(146, 245))
        self.cut_line(sc, box, w, S.exact_cut(w, S.NP_ALPHA), dark(AMBER, 0.1), op, 3.4)
        power = float(S.exact_power(w, S.NP_ALPHA))
        self.text(sc, 0.26, 0.345, "cut direction %s°:  power %s" % (
            ("%d" % round(a)).replace("-", MINUS), pct(power)), op, strong=True, size=HEAD)
        chart = self.graph(((0.56, 0.93), (0.44, 0.76)), (-60, 90), (0, 0.66))
        self.axes(sc, chart, op, "direction of the cut (degrees)", "",
                  xticks=self.ticks((-60, -30, 0, 30, 60, 90), "%d"),
                  yticks=self.ticks((0.2, 0.4, 0.6)), grid=False)
        reach = -60.0 + 150.0 * smoothstep(0.05, 0.6, u)
        sel = d["fine"] <= reach
        self.path(sc, chart, d["fine"][sel], d["fine_power"][sel], STRONG, op, 2.6)
        for ang, p in zip(d["angles"], d["ang_mc"]):
            if ang <= reach:
                self.disk(sc, self.P(chart.fx(ang), chart.fy(p)), 3.6, dark(AMBER, 0.1), op,
                          lift=1.2, outline=0.0)
        marks = ((0.0, 0, 0.385, "x alone"), (d["lr_angle"], 2, 0.345, "likelihood ratio"),
                 (45.0, 1, 0.385, "x + y"))
        for ang, j, y, name in marks:
            on = op * smoothstep(ang - 4, ang + 4, reach)
            x = chart.fx(ang)
            self.text(sc, x + (0.03 if j == 1 else 0.0), y, name, on,
                      ink=dark(STAT_INKS[j], 0.3), size=0.030)
            self.vline(sc, x, y + 0.025, chart.fy(self.power_exact[j]) - 0.012,
                       STAT_INKS[j], 0.9 * on, 2.0)
            self.disk(sc, self.P(x, chart.fy(self.power_exact[j])), 6.0, STAT_INKS[j], on,
                      lift=1.5, outline=0.15)
        late = op * smoothstep(0.8, 0.9, u)
        self.text(sc, 0.09, 0.86, "L₁ / (L₀ + L₁) in place of the ratio: the same events "
                  "pass, power %s" % pct(m["monotonic_power"]), late, anchor="lm",
                  ink=dark(GREEN, 0.4))
        self.text(sc, chart.fx(-38), chart.fy(0.50), "power at", op, size=0.030)
        self.text(sc, chart.fx(-38), chart.fy(0.50) + 0.042, "α = 5%", op, size=0.030)
        self.tag(sc, "curve: exact; dots: pseudo-experiments", op)


# ====================================================================== 144: look elsewhere
MAX_INK = VIOLET                              # the largest local significance of a scan


class LookElsewhere(Tests):
    """One background-only spectrum with a window that looks like a signal; the scan
    that found it; many more spectra with nothing in them and the largest excess of
    each; the trials factor; and what a local significance is worth globally."""

    TITLE = "The look-elsewhere effect"
    PHASES = (("one spectrum", 11.0), ("the scan", 12.0), ("many spectra", 12.0),
              ("trials factor", 11.0), ("global", 11.0))
    MASS = (100.0, 160.0)

    def setup(self):
        d = self.d = S.lee_data()
        self.meta = m = self.study.metadata
        self.mass = np.asarray(self.study.panels[0].data["x"], float)
        self.counts = np.asarray(self.study.panels[0].data["ys"][1], float)
        self.zs = np.asarray(self.study.panels[1].data["ys"][0], float)
        self.nw = int(m["windows"])
        self.centre = S.LEE_M0 + np.arange(self.nw) + S.LEE_WIDTH / 2.0
        self.sums = S.window_sums(self.counts).astype(int)
        self.i3, self.i4 = d["i3"], d["i4"]

    def subtitle_for(self, k):
        d, m = self.d, self.meta
        i3 = self.i3
        return [
            "a spectrum with background only: no signal was put in. Yet one window holds "
            "%d events where b = %.1f are expected: a local p-value of %.5f, Z = %.1fσ. "
            "A discovery?" % (m["shown_n"], m["shown_b"], m["shown_p"], m["shown_z"]),
            "that window was not chosen in advance. A window of %d bins was slid across "
            "the spectrum and Z computed at each of %d positions: %d chances for the "
            "background to fluctuate up" % (m["width"], self.nw, self.nw),
            "%s more spectra with nothing in them. The largest Z of a scan is typically "
            "%.1fσ, and it passes 3σ in %s of them: %d times more often than one window "
            "fixed in advance" % (commas(S.LEE_TOYS), m["median_max"],
                                  pct(m["p_global_3"]), round(m["trials_factor_3"])),
            "the trials factor: global p over local p, %d here. %d independent windows "
            "would give %s, the %d sharing no bin %s; the measured %s is what %d "
            "independent places would give"
            % (round(m["trials_factor_3"]), self.nw, pct(d["all_indep"][i3]),
               len(d["apart"]), pct(d["apart_exact"][i3]), pct(m["p_global_3"]),
               round(m["n_eff_3"])),
            "so a local 3σ found by this scan is a global %.1fσ, and a local 4σ a global "
            "%.1fσ (%d of %s spectra). This is why a search quotes both, and why "
            "discovery asks for 5σ" % (m["z_global_3"], m["z_global_4"], m["count_4"],
                                       commas(S.LEE_TOYS)),
        ][k]

    def spectrum(self, sc, chart, counts, op, window=None, label=True):
        """Counts as bars, the expected background as a curve, one window in amber."""
        inks = [AMBER if window is not None and window <= i < window + S.LEE_WIDTH
                else soft(BLUE, 0.45) for i in range(S.LEE_BINS)]
        self.bars(sc, chart, self.mass, counts, 1.0, inks, op)
        self.path(sc, chart, self.mass, self.d["b"], dark(BLUE, 0.25), op, 2.8)

    def phase_1(self, sc, u, op):
        m = self.meta
        chart = self.graph(((0.09, 0.62), (0.42, 0.76)), self.MASS, (0, 245))
        self.axes(sc, chart, op, "mass (GeV), one bin per GeV", "events per bin",
                  xticks=self.ticks((100, 120, 140, 160), "%d"),
                  yticks=self.ticks((100, 200), "%d"), grid=False)
        found = smoothstep(0.3, 0.42, u)
        w = int(m["shown_window"])
        self.spectrum(sc, chart, self.counts, op, window=w if found > 0.5 else None)
        self.key(sc, chart.fx(131), 0.45, [("counts of one pseudo-experiment",
                                            soft(BLUE, 0.3))], op)
        self.key(sc, chart.fx(131), 0.496, [("expected background b", dark(BLUE, 0.25))], op)
        self.text(sc, 0.355, 0.345, "truth: b = 200 exp(−(m − 100) / 30) per bin, no signal",
                  op, size=0.031)
        x0 = 0.67
        on = op * found
        lo = S.LEE_M0 + w
        self.text(sc, x0, 0.40, "window %d–%d GeV" % (lo, lo + S.LEE_WIDTH), on, anchor="lm",
                  ink=dark(AMBER, 0.45), size=HEAD)
        self.lines(sc, x0, 0.455, ["counted:   n = %d" % m["shown_n"],
                                   "expected:  b = %.1f" % m["shown_b"]], on, step=0.044,
                   size=0.031)
        on = op * smoothstep(0.52, 0.62, u)
        self.lines(sc, x0, 0.565, ["p = P(n ≥ %d | b)" % m["shown_n"],
                                   "   = %.5f" % m["shown_p"]], on, step=0.044, size=0.031)
        self.text(sc, x0, 0.69, "Z = %.1fσ" % m["shown_z"], op * smoothstep(0.7, 0.8, u),
                  anchor="lm", strong=True, size=0.046)
        self.text(sc, x0, 0.745, "local significance", op * smoothstep(0.7, 0.8, u),
                  anchor="lm")
        self.text(sc, 0.09, 0.885, "spectrum %d of %s: the first to hold a window past 3σ"
                  % (m["shown_experiment"], commas(S.LEE_TOYS)), op, anchor="lm")
        self.tag(sc, "pseudo-experiments, seed %d; p, Z: exact" % S.LEE_SEED, op)

    def phase_2(self, sc, u, op):
        m = self.meta
        top = self.graph(((0.09, 0.70), (0.375, 0.55)), self.MASS, (0, 245))
        self.axes(sc, top, op, "", "events per bin", xticks=[],
                  yticks=self.ticks((0, 100, 200), "%d"), grid=False)
        best = int(m["shown_window"])
        pos = smoothstep(0.04, 0.74, u) * (self.nw - 1)
        w = int(round(pos)) if u < 0.8 else best
        self.spectrum(sc, top, self.counts, op, window=w)
        low = self.graph(((0.09, 0.70), (0.625, 0.80)), self.MASS, (-3.6, 4.4))
        self.axes(sc, low, op, "mass (GeV)", "local Z of the window",
                  xticks=self.ticks((100, 120, 140, 160), "%d"),
                  yticks=self.ticks((-2, 0, 2, 4), "%d"), grid=False)
        self.dashed(sc, (0.09, low.fy(3.0)), (0.70, low.fy(3.0)), RED, 2 * op, n=30, width=2.0)
        self.text(sc, 0.71, low.fy(3.0), "3σ", op, anchor="lm", ink=dark(RED, 0.3))
        self.polyline(sc, [self.P(0.09, low.fy(0)), self.P(0.70, low.fy(0))], 1.2, AXIS,
                      0.45 * op)
        done = self.nw if u >= 0.8 else w + 1
        self.path(sc, low, self.centre[:done], self.zs[:done], dark(AMBER, 0.15), op, 2.4)
        for i in range(done):
            hot = i == w
            self.disk(sc, self.P(low.fx(self.centre[i]), low.fy(self.zs[i])),
                      6.5 if hot else 3.0, AMBER if hot else dark(AMBER, 0.15), op,
                      lift=1.6 if hot else 1.2, outline=0.1, glow=0.4 if hot else 0.0)
        self.vline(sc, low.fx(self.centre[w]), 0.555, 0.80, AMBER, 0.4 * op, 1.6)
        x0 = 0.765
        self.text(sc, x0, 0.40, "window %d of %d" % (w + 1, self.nw), op, anchor="lm",
                  strong=True, size=HEAD)
        z = self.zs[w]
        self.lines(sc, x0, 0.455, ["n = %d" % self.sums[w], "b = %.1f" % self.d["bw"][w]], op,
                   step=0.044, size=0.031)
        self.text(sc, x0, 0.56, "Z = %sσ" % signed(z, "%.1f"), op, anchor="lm", strong=True,
                  size=BIG)
        self.lines(sc, x0, 0.66, ["%d windows," % self.nw, "%d chances" % self.nw],
                   op * smoothstep(0.8, 0.9, u), step=0.044, size=HEAD)
        self.tag(sc, "pseudo-experiments, seed %d; Z: exact" % S.LEE_SEED, op)

    def toys_at(self, u):
        return int(round(8 * (S.LEE_TOYS / 8.0) ** float(np.clip((u - 0.42) / 0.38, 0, 1))))

    def phase_3(self, sc, u, op):
        d, m = self.d, self.meta
        self.text(sc, 0.06, 0.33, "the first 8 of %s scans: nothing in any" % commas(S.LEE_TOYS),
                  op, anchor="lm", size=0.031)
        for k in range(S.LEE_TRACES):
            r, col = divmod(k, 2)
            x0, top = 0.06 + 0.22 * col, 0.375 + 0.122 * r
            on = op * smoothstep(0.03 + 0.045 * k, 0.09 + 0.045 * k, u)
            chart = self.graph(((x0, x0 + 0.19), (top, top + 0.066)), (0, self.nw - 1),
                               (-3.6, 4.4))
            self.rect(sc, self.P(x0 + 0.095, top + 0.033), self.span(0.19)[0] + 6,
                      self.span(0, 0.066)[1] + 6, PAPER, on, border=DIM, radius=3, lift=0.1)
            self.dashed(sc, (x0, chart.fy(3.0)), (x0 + 0.19, chart.fy(3.0)), RED, 2 * on, n=12,
                        width=1.6)
            z = d["traces"][k]
            self.path(sc, chart, np.arange(self.nw), z, dark(AMBER, 0.15), on, 2.2)
            best = int(np.argmax(z))
            self.disk(sc, self.P(chart.fx(best), chart.fy(z[best])), 5.0, MAX_INK, on,
                      lift=1.5, outline=0.1)
            self.text(sc, x0, top + 0.094, "scan %d: largest %.1fσ" % (k + 1, z[best]), on,
                      anchor="lm", size=0.029, ink=dark(MAX_INK, 0.3))
        n = self.toys_at(u)
        width = d["edges"][1] - d["edges"][0]
        mid = 0.5 * (d["edges"][1:] + d["edges"][:-1])
        hist = d["hist_max"] if n >= S.LEE_TOYS else np.histogram(d["zmax"][:n], d["edges"])[0]
        chart = self.graph(((0.57, 0.93), (0.40, 0.70)), (-3, 5), (0, 0.95))
        self.axes(sc, chart, op, "Z (σ)", "", xticks=self.ticks((-2, 0, 2, 4), "%d"),
                  yticks=[], grid=False)
        on = op * smoothstep(0.4, 0.46, u)
        inks = [AMBER if e >= S.LEE_Z - 1e-9 else soft(MAX_INK, 0.2) for e in d["edges"][:-1]]
        self.bars(sc, chart, mid, hist / float(n) / width, width, inks, on, gap=0.6)
        self.steps(sc, chart, d["edges"], d["hist_one"] / float(S.LEE_TOYS) / width,
                   dark(BLUE, 0.1), op * smoothstep(0.3, 0.4, u), 2.4)
        self.key(sc, 0.58, 0.33, [("a single window", BLUE)], op * smoothstep(0.3, 0.4, u))
        self.key(sc, 0.775, 0.33, [("largest of %d" % self.nw, MAX_INK)], on)
        self.text(sc, 0.93, 0.375, "%s scans" % commas(n), on, anchor="rm", size=0.030)
        late = op * smoothstep(0.82, 0.9, u)
        self.text(sc, 0.57, 0.815, "largest ≥ 3σ in %s of %s: %s" % (
            commas(m["count_3"]), commas(S.LEE_TOYS), pct(m["p_global_3"])), late,
            anchor="lm", strong=True, size=0.031)
        self.text(sc, 0.57, 0.86, "one fixed window: %s" % pct(m["p_local_3"], 2), late,
                  anchor="lm", ink=dark(BLUE, 0.3), size=0.031)
        self.tag(sc, "pseudo-experiments: %s spectra" % commas(S.LEE_TOYS), op)

    def phase_4(self, sc, u, op):
        d, m = self.d, self.meta
        i3 = self.i3
        x0 = 0.06
        self.text(sc, x0, 0.345, "one window fixed in advance:", op, anchor="lm",
                  ink=dark(BLUE, 0.3), size=0.031)
        self.text(sc, x0, 0.39, "local p = P(Z ≥ 3) = %.5f" % m["p_local_3"], op, anchor="lm",
                  strong=True, size=HEAD)
        on = op * smoothstep(0.1, 0.2, u)
        self.text(sc, x0, 0.45, "the largest of the %d windows:" % self.nw, on, anchor="lm",
                  ink=dark(AMBER, 0.45), size=0.031)
        self.text(sc, x0, 0.50, "global p = %.4f  (measured)" % m["p_global_3"], on,
                  anchor="lm", strong=True, size=HEAD)
        on = op * smoothstep(0.26, 0.36, u)
        self.box(sc, 0.31, 0.58, 380, 40, soft(AMBER, 0.6), on,
                 "trials factor = %.4f / %.5f = %d" % (m["p_global_3"], m["p_local_3"],
                                                       round(m["trials_factor_3"])),
                 border=AMBER, size=0.032)
        on = op * smoothstep(0.46, 0.56, u)
        self.text(sc, x0, 0.665, "N independent places:  1 − (1 − p) to the power N", on,
                  anchor="lm", size=0.030)
        rows = (("N = %d, every window:  %.4f" % (self.nw, d["all_indep"][i3]), VIOLET, 0.56),
                ("N = %d, no shared bins:  %.4f" % (len(d["apart"]), d["apart_exact"][i3]),
                 TEAL, 0.64),
                ("measured %.4f:  N = %d" % (m["p_global_3"], round(m["n_eff_3"])),
                 AMBER, 0.72))
        for r, (s, ink, t0) in enumerate(rows):
            self.text(sc, x0, 0.715 + 0.045 * r, s, op * smoothstep(t0, t0 + 0.08, u),
                      anchor="lm", ink=dark(ink, 0.4), size=0.031)
        chart = self.graph(((0.60, 0.93), (0.40, 0.76)), (1.5, 4.5), (2e-6, 1.0), ylog=True)
        self.axes(sc, chart, op, "local Z (σ)", "chance of reaching that Z",
                  xticks=self.ticks((2, 3, 4), "%d"),
                  yticks=[(10.0 ** -k, "1" if k == 0 else "10" + S_SUP[k]) for k in (0, 2, 4)],
                  grid=False)
        g = d["grid"]
        self.path(sc, chart, g, d["p_nominal"], BLUE, op, 2.6)
        self.path(sc, chart, g, d["all_indep"], VIOLET, op * smoothstep(0.56, 0.66, u), 2.4)
        self.path(sc, chart, g, d["apart_exact"], TEAL, op * smoothstep(0.64, 0.74, u), 2.4)
        shown = int(round(smoothstep(0.1, 0.5, u) * len(g)))
        for i in range(shown):
            if d["n_glob"][i] >= 10:
                self.disk(sc, self.P(chart.fx(g[i]), chart.fy(d["p_glob"][i])), 4.2, AMBER, op,
                          lift=1.3, outline=0.15)
        self.vline(sc, chart.fx(3.0), chart.fy(m["p_global_3"]), chart.fy(m["p_local_3"]),
                   dark(AMBER, 0.2), op * smoothstep(0.3, 0.4, u), 2.2)
        self.tag(sc, "dots: pseudo-experiments; lines: exact", op)

    def phase_5(self, sc, u, op):
        d, m = self.d, self.meta
        self.text(sc, 0.06, 0.345, "what a local excess is worth after this scan", op,
                  anchor="lm", size=HEAD, strong=True)
        rows = (("3.0σ", "%.1fσ" % m["z_global_3"],
                 ["%s of %s" % (commas(m["count_3"]), commas(S.LEE_TOYS)), "spectra reach it"]),
                ("4.0σ", "%.1fσ" % m["z_global_4"],
                 ["%s of %s" % (commas(m["count_4"]), commas(S.LEE_TOYS)), "spectra reach it"]),
                ("5.0σ", "≈ %.1fσ" % m["z_global_5_estimate"],
                 ["never reached: an", "estimate with N = %d" % round(m["n_eff_4"])]))
        for r, (loc, glob, note) in enumerate(rows):
            y = 0.455 + 0.135 * r
            on = op * smoothstep(0.06 + 0.2 * r, 0.16 + 0.2 * r, u)
            self.box(sc, 0.125, y, 120, 46, soft(BLUE, 0.6), on, "local " + loc, border=BLUE,
                     size=0.032)
            self.arrow(sc, (0.198, y), (0.247, y), AXIS, on, 1.0, 2.6)
            self.box(sc, 0.33, y, 146, 46, soft(AMBER, 0.55), on, "global " + glob,
                     border=AMBER, size=0.032)
            self.lines(sc, 0.42, y - 0.021, note, on, step=0.042, size=0.029)
        chart = self.graph(((0.66, 0.93), (0.42, 0.76)), (1.5, 4.5), (-1.5, 4.5))
        self.axes(sc, chart, op, "local Z (σ)", "global Z (σ)",
                  xticks=self.ticks((2, 3, 4), "%d"), yticks=self.ticks((-1, 0, 1, 2, 3, 4),
                  "%d"), grid=False)
        self.dashed(sc, (chart.fx(1.5), chart.fy(1.5)), (chart.fx(4.5), chart.fy(4.5)), AXIS,
                    op, n=16)
        g = d["grid"]
        go = smoothstep(0.1, 0.6, u)
        self.path(sc, chart, g, d["z_fit"], dark(AMBER, 0.2), op, 2.2, go=go)
        for i in range(int(round(go * len(g)))):
            if d["n_glob"][i] >= 10:
                lit = i in (self.i3, self.i4)
                self.disk(sc, self.P(chart.fx(g[i]), chart.fy(d["z_glob"][i])),
                          6.2 if lit else 3.6, AMBER, op, lift=1.3, outline=0.15)
        self.text(sc, chart.fx(2.2), chart.fy(3.5), "no scan:", op, size=0.029)
        self.text(sc, chart.fx(2.2), chart.fy(3.5) + 0.04, "global = local", op, size=0.029)
        self.text(sc, 0.06, 0.875, "line: %d independent places, the number measured at 3σ"
                  % round(m["n_eff_3"]), op * smoothstep(0.6, 0.7, u), anchor="lm")
        self.tag(sc, "dots: pseudo-experiments; 5σ: extrapolated", op)


S_SUP = {2: "⁻²", 4: "⁻⁴"}

ANIMATIONS = {
    "type-i-and-type-ii-errors": TypeErrors,
    "p-values": PValues,
    "neyman-pearson": NeymanPearson,
    "look-elsewhere-effect": LookElsewhere,
}

"""Probability, read the frequentist way and the Bayesian way: animations 118-123 of
the "next" catalogue (see :mod:`voxanim.mech`).

* :class:`FrequentistAndBayesian`: one count, a confidence interval and a posterior;
* :class:`BayesTheoremAreas`: Bayes' theorem as rectangles of a unit square;
* :class:`PriorWashingOut`: five priors updated by the same coin flips;
* :class:`IntervalCoverage`: how often an interval contains the truth;
* :class:`BinomialPoissonGaussian`: one histogram, three distributions;
* :class:`CentralLimitTheorem`: means of dice and decay times becoming a bell, and a
  Breit-Wigner that never does.

They draw the experiments of :mod:`voxanim.study_stats_prob`, which records every array
shown here; this module computes layout only. Each starts from one small case whose
arithmetic is on screen, and every phase says, bottom right, what kind of evidence it
shows: exact (a formula or a distribution evaluated) or pseudo-experiments (seeded draws
from a model whose true values are stated).
"""

from __future__ import annotations

import math

import numpy as np

from . import layout
from . import study_stats_prob as S
from .diagram import draw_image
from .easing import smoothstep
from .losschart import LossChart, sup
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIM,
    GREEN,
    MAGENTA,
    OCHRE,
    PAPER,
    RED,
    TEAL,
    VIOLET,
    Mechanism,
    dark,
    soft,
)
from .mech_random import raster

EXACT = "exact"
BIG, HEAD = 0.036, 0.034
STRONG = layout.STRONG_INK
GREY = (0.56, 0.58, 0.60)


def toys(n, what="pseudo-experiments"):
    """The evidence tag of a phase that shows seeded draws."""
    return "%s %s" % (format(int(n), ","), what)


def commas(n):
    return format(int(round(float(n))), ",")


def pct(v, digits=1):
    return "%.*f%%" % (digits, 100.0 * float(v))


def signed(v, digits=2):
    """A number with a real minus sign."""
    return ("%.*f" % (digits, float(v))).replace("-", "−")


def sci(v):
    """A small probability as ``m × 10⁻ⁿ``."""
    from .losschart import pow10

    return "%.3f" % v if v >= 0.01 else pow10(float(v), 1)


class Stats(Mechanism):
    """What the six animations share: the evidence tag, histograms, curves through
    values, shaded areas and interval bars on a chart."""

    def tag(self, sc, kind, op):
        """The kind of evidence on screen, bottom right."""
        self.text(sc, 0.95, 0.925, kind, op, anchor="rm", ink=dark(TEAL, 0.25))

    def frame(self, box, xr, yr, xlog=False, ylog=False):
        """A chart in the layout box ``((x0, x1), (top, bottom))`` over exactly the
        ranges given."""
        return LossChart(box, (float(xr[0]), float(xr[1])), (float(yr[0]), float(yr[1])),
                         xlog=xlog, ylog=ylog)

    def bars(self, sc, chart, xs, hs, ink, op, width=0.8, base=None, lift=0.4, grow=1.0):
        """Vertical bars of heights ``hs`` at ``xs`` (``width`` in units of x; ``ink``:
        one colour or one per bar), rising from ``base`` (default: the chart's floor)."""
        if op <= 0.01:
            return
        y0 = chart.y1 if base is None else chart.fy(base)
        one = isinstance(ink[0], (float, np.floating))
        for i, (x, h) in enumerate(zip(xs, hs)):
            a, b = chart.fx(float(x) - width / 2), chart.fx(float(x) + width / 2)
            top = y0 + (chart.fy(float(h)) - y0) * grow
            hpx = self.span(0, y0 - top)[1]
            if hpx < 0.5 or b <= a:
                continue
            wpx = self.span(b - a)[0]
            self.rect(sc, self.P((a + b) / 2, (y0 + top) / 2), wpx, hpx,
                      ink if one else ink[i], op, radius=min(2.5, wpx / 2, hpx / 2), lift=lift)

    def curve(self, sc, chart, xs, ys, ink, op, go=1.0, width=2.6):
        """``ys`` against ``xs`` as a line, drawn as far as fraction ``go`` of the points."""
        if op <= 0.01 or go <= 0:
            return
        m = max(2, int(np.ceil(len(xs) * min(float(go), 1.0))))
        pts = [self.P(chart.fx(float(x)), chart.fy(float(y))) for x, y in zip(xs[:m], ys[:m])]
        self.polyline(sc, pts, width, ink, op, bias=0.3)

    def area(self, sc, chart, xs, ys, a, b, ink, op, strips=36):
        """The area under ``ys(xs)`` between ``a`` and ``b``, filled."""
        if op <= 0.01 or b <= a:
            return
        edges = np.linspace(a, b, strips + 1)
        mid = 0.5 * (edges[1:] + edges[:-1])
        self.bars(sc, chart, mid, np.interp(mid, xs, ys), ink, op, width=(b - a) / strips,
                  lift=0.2)

    def marks(self, sc, chart, xs, ys, ink, op, r=4.2, ring=None):
        """A disk at each ``(x, y)`` of the chart."""
        for x, y in zip(xs, ys):
            self.disk(sc, self.P(chart.fx(float(x)), chart.fy(float(y))), r, ink, op, lift=1.4,
                      outline=0.1, ring=ring, ring_w=1.6)

    def bar(self, sc, a, b, ink, op, thick=5.0, lift=0.6):
        """A straight bar between two layout points on one horizontal or vertical."""
        (ax, ay), (bx, by) = a, b
        w, h = self.span(abs(bx - ax), abs(by - ay))
        self.rect(sc, self.P((ax + bx) / 2, (ay + by) / 2, ), max(w, thick), max(h, thick), ink,
                  op, radius=thick / 2, lift=lift)

    def rule(self, sc, a, b, ink, op, width=1.6, dashes=0):
        """A thin line between two layout points, dashed when ``dashes`` > 0. Kept under
        half strength, as a guide behind what it measures."""
        pa, pb = self.P(*a), self.P(*b)
        if not dashes:
            self.polyline(sc, [pa, pb], width, ink, op)
            return
        for i in range(dashes):
            u0, u1 = i / dashes, (i + 0.55) / dashes
            self.polyline(sc, [pa + (pb - pa) * u0, pa + (pb - pa) * u1], width, ink, op)

    def key(self, sc, fx, fy, entries, op, step=0.046, size=0.031):
        """A legend: a colour chip and a name per entry, stacked down from ``(fx, fy)``."""
        for n, (name, ink) in enumerate(entries):
            y = fy + n * step
            self.box(sc, fx, y, 14, 14, ink, op, radius=3)
            self.text(sc, fx + 0.016, y, name, op, anchor="lm", ink=dark(ink, 0.3), size=size)

    def ticks(self, values, fmt="%g"):
        return [(float(v), fmt % v) for v in values]


# ====================================================================== 118: two readings
FREQ, BAYES, SEEN = BLUE, GREEN, AMBER


class FrequentistAndBayesian(Stats):
    """One count of three events read twice: as the set of means for which three is a
    likely count (the confidence belt, read across), and as a posterior for the mean
    (prior times likelihood); the two intervals for every count; how often each contains
    a known truth; and a count below a known background, where they part."""

    TITLE = "Two readings of one count"
    PHASES = (("the count", 8.0), ("confidence", 13.0), ("posterior", 12.0),
              ("side by side", 11.0), ("a boundary", 10.0))

    def setup(self):
        s, d = self.study, S.fb_data()
        m = self.meta = s.metadata
        self.n = int(m["observed"])
        self.pmfs = np.asarray(s.panels[0].data["pmfs"], float)
        self.pmf_means = [float(v) for v in s.panels[0].data["pmf_means"]]
        self.belt_mu = np.asarray(s.panels[0].data["x"], float)
        self.belt = np.asarray(s.panels[0].data["ys"], float)
        self.grid = np.asarray(s.panels[1].data["x"], float)
        self.post = np.asarray(s.panels[1].data["ys"], float)[1]
        self.ci = [float(v) for v in m["confidence_interval"]]
        self.cr = [float(v) for v in m["credible_interval"]]
        self.ns = np.asarray(s.panels[2].data["x"], int)
        self.iv = np.asarray(s.panels[2].data["ys"], float)
        cov = m["coverage_at_truth"]
        self.cov_exact, self.cov_toys = cov["exact"], cov["pseudo_experiments"]
        self.repeats, self.truth = int(cov["repeats"]), float(cov["mu"])
        x = np.asarray(s.panels[3].data["x"], float)
        cred = np.asarray(s.panels[3].data["ys"], float)[1][x >= 1.0]
        self.cred_range = float(cred.min()), float(cred.max())
        self.conf_min = float(np.asarray(s.panels[3].data["ys"], float)[0].min())
        self.b = float(m["background"])
        self.b_ns = [int(v) for v in s.panels[2].data["boundary_counts"]]
        self.b_ci = np.asarray(s.panels[2].data["boundary_confidence"], float)
        self.b_cr = np.asarray(s.panels[2].data["boundary_credible"], float)
        row = int(np.argmin(np.abs(self.belt_mu - 3.0)))
        self.row3 = int(self.belt[0, row]), int(self.belt[1, row])
        self.row3_p = float(d["pmfs"][1][self.row3[0]:self.row3[1] + 1].sum())

    def subtitle_for(self, k):
        n, ci, cr = self.n, self.ci, self.cr
        return [
            "a counter records n = %d events. Counts scatter: if the true mean is μ, the "
            "chance of n is the Poisson formula. Which values of μ does n = %d point to? "
            "Two readings follow" % (n, n),
            "frequentist: for every possible μ, mark the counts that are likely (the "
            "central 68%%). The confidence interval is every μ whose range holds the "
            "observed %d: from %.2f to %.2f" % (n, ci[0], ci[1]),
            "Bayesian: start from a prior for μ (here flat), multiply by the chance of "
            "n = %d at each μ, and scale to area 1. The central 68%% of that posterior "
            "runs from %.2f to %.2f" % (n, cr[0], cr[1]),
            "at n = %d the upper ends agree exactly and the lower ends do not. In %s "
            "pseudo-experiments with μ = %g the confidence interval held the truth in "
            "%s, the credible one in %s"
            % (n, commas(self.repeats), self.truth, pct(self.cov_toys[0]),
               pct(self.cov_toys[1])),
            "with a known background of %g the signal s = μ − %g cannot be negative. At "
            "n = 0 the confidence interval for s is %s to %s, no allowed value; the "
            "posterior, zero below s = 0, gives %.2f to %.2f"
            % (self.b, self.b, signed(self.b_ci[0, 0]), signed(self.b_ci[0, 1]),
               self.b_cr[0, 0], self.b_cr[0, 1]),
        ][k]

    def phase_1(self, sc, u, op):
        n = self.n
        self.text(sc, 0.5, 0.315, "one experiment counted n = %d events. What is the mean "
                  "μ behind it?" % n, op, strong=True, size=BIG)
        ks = np.arange(self.pmfs.shape[1])
        for i, mu in enumerate(self.pmf_means):
            show = op * smoothstep(0.06 + 0.16 * i, 0.16 + 0.16 * i, u)
            x0 = 0.085 + 0.31 * i
            chart = self.frame(((x0, x0 + 0.245), (0.44, 0.70)), (-0.7, 12.7), (0.0, 0.36))
            self.axes(sc, chart, show, "count n", "", xticks=self.ticks((0, 3, 6, 9, 12)),
                      yticks=self.ticks((0, 0.1, 0.2, 0.3)) if i == 0 else [], grid=False)
            inks = [SEEN if k == n else soft(FREQ, 0.35) for k in ks]
            self.bars(sc, chart, ks, self.pmfs[i], inks, show)
            self.text(sc, x0 + 0.1225, 0.395, "if μ = %g" % mu, show, strong=True, size=HEAD)
            self.text(sc, x0 + 0.1225, 0.835, "P(%d | μ = %g) = %.3f" % (n, mu, self.pmfs[i, n]),
                      show, ink=dark(SEEN, 0.35), size=0.032)
        late = op * smoothstep(0.62, 0.74, u)
        self.text(sc, 0.06, 0.895, "P(n | μ) = μⁿ exp(−μ) / n!     at μ = 3:  3³ × exp(−3) / 3! "
                  "= 27 × 0.0498 / 6 = %.3f" % self.pmfs[1, n], late, anchor="lm")
        self.tag(sc, EXACT, op)

    def phase_2(self, sc, u, op):
        n, (lo, hi) = self.n, self.ci
        chart = self.frame(((0.10, 0.56), (0.37, 0.79)), (-0.6, 14.6), (0.0, 10.2))
        self.axes(sc, chart, op, "count n", "true mean μ", xticks=self.ticks(range(0, 15, 2)),
                  yticks=self.ticks((0, 2, 4, 6, 8, 10)), grid=False)
        rows = len(self.belt_mu)
        shown = int(smoothstep(0.03, 0.42, u) * rows + 1e-9)
        read = smoothstep(0.50, 0.60, u)
        for r in range(shown):
            mu = self.belt_mu[r]
            a, b = self.belt[0, r] - 0.42, self.belt[1, r] + 0.42
            holds = lo <= mu <= hi
            ink = soft(FREQ, 0.15 if (holds and read > 0.5) else 0.6)
            y = chart.fy(mu)
            self.bar(sc, (chart.fx(a), y), (chart.fx(b), y), ink, op, thick=4.6)
        if read > 0:
            x = chart.fx(n)
            self.bar(sc, (x, chart.y1), (x, chart.y1 - (chart.y1 - chart.y0) * read), SEEN, op,
                     thick=3.0, lift=0.9)
        got = op * smoothstep(0.60, 0.68, u)
        self.bar(sc, (chart.x0, chart.fy(lo)), (chart.x0, chart.fy(hi)), FREQ, got, thick=7.0,
                 lift=1.0)
        tx = 0.62
        self.lines(sc, tx, 0.385, ["for each true μ (a row):", "the counts that are likely,",
                                   "the central 68% or more"], op, size=0.032, step=0.045)
        eg = op * smoothstep(0.2, 0.3, u)
        self.lines(sc, tx, 0.545, [
            ("row μ = 3: n from %d to %d," % self.row3, dark(FREQ, 0.25)),
            ("probability %s" % pct(self.row3_p), dark(FREQ, 0.25))], eg, size=0.032,
            step=0.045)
        self.lines(sc, tx, 0.66, [("read up the column n = %d:" % n, dark(SEEN, 0.4))],
                   op * smoothstep(0.5, 0.58, u), size=0.032)
        self.text(sc, tx, 0.72, "%.2f ≤ μ ≤ %.2f" % (lo, hi), got, anchor="lm", size=0.046,
                  ink=dark(FREQ, 0.2))
        self.lines(sc, tx, 0.80, ["what it says: of many repeats, at least",
                                  "68% of such intervals contain the true μ"],
                   op * smoothstep(0.76, 0.86, u), size=0.031, step=0.043)
        self.tag(sc, EXACT, op)

    def phase_3(self, sc, u, op):
        n, (lo, hi) = self.n, self.cr
        chart = self.frame(((0.10, 0.56), (0.39, 0.79)), (0.0, 12.0), (0.0, 0.26))
        self.axes(sc, chart, op, "mean μ", "", xticks=self.ticks(range(0, 13, 2)), yticks=[],
                  grid=False)
        g, post = self.grid, self.post
        self.rule(sc, (chart.fx(0), chart.fy(0.045)), (chart.fx(12), chart.fy(0.045)), OCHRE,
                  0.45 * op, 2.4, dashes=26)
        self.text(sc, chart.fx(9.6), chart.fy(0.045) - 0.032, "prior: flat", op,
                  ink=dark(OCHRE, 0.3))
        like = smoothstep(0.08, 0.36, u)
        became = smoothstep(0.42, 0.52, u)
        self.area(sc, chart, g, post, lo, hi, soft(BAYES, 0.45), op * smoothstep(0.58, 0.68, u))
        self.curve(sc, chart, g, post, SEEN if became < 0.5 else BAYES, op, like, 3.0)
        k = int(np.argmax(post))
        self.text(sc, chart.fx(g[k]) + 0.075, chart.fy(post[k]) - 0.03,
                  "P(%d | μ), as a function of μ" % n if became < 0.5 else "the posterior for μ",
                  op * smoothstep(0.2, 0.3, u), anchor="lm",
                  ink=dark(SEEN if became < 0.5 else BAYES, 0.35))
        shaded = op * smoothstep(0.62, 0.72, u)
        self.text(sc, chart.fx((lo + hi) / 2), chart.fy(0.075), "68.3%", shaded, strong=True,
                  size=HEAD)
        tx = 0.62
        self.lines(sc, tx, 0.385, ["posterior = prior × likelihood,", "scaled to area 1"], op,
                   size=0.032, step=0.045)
        self.lines(sc, tx, 0.50, [
            ("likelihood: μ³ exp(−μ) / 3!", dark(SEEN, 0.4)),
            ("the chance of the %d seen, at each μ" % n, dark(SEEN, 0.4))],
            op * smoothstep(0.1, 0.2, u), size=0.032, step=0.045)
        self.lines(sc, tx, 0.615, ["flat prior: the same curve,", "now a density for μ"],
                   op * smoothstep(0.42, 0.52, u), size=0.032, step=0.045)
        self.text(sc, tx, 0.72, "%.2f ≤ μ ≤ %.2f" % (lo, hi), shaded, anchor="lm", size=0.046,
                  ink=dark(BAYES, 0.3))
        self.lines(sc, tx, 0.80, ["what it says: given this count and this",
                                  "prior, μ is in there with probability 68%"],
                   op * smoothstep(0.76, 0.86, u), size=0.031, step=0.043)
        self.tag(sc, EXACT, op)

    def phase_4(self, sc, u, op):
        n = self.n
        last = 12
        chart = self.frame(((0.07, 0.46), (0.37, 0.79)), (-0.7, last + 0.7), (0.0, 18.0))
        self.axes(sc, chart, op, "count n", "interval for μ",
                  xticks=self.ticks(range(0, last + 1, 2)), yticks=self.ticks((0, 5, 10, 15)),
                  grid=False)
        shown = int(smoothstep(0.02, 0.4, u) * (last + 1) + 1e-9)
        for k in range(shown):
            for j, (ink, dx) in enumerate(((FREQ, -0.19), (BAYES, 0.19))):
                x = chart.fx(k + dx)
                self.bar(sc, (x, chart.fy(self.iv[2 * j, k])), (x, chart.fy(self.iv[2 * j + 1, k])),
                         ink, op, thick=6.0)
        if shown > n:
            self.rect(sc, self.P(chart.fx(n), chart.fy((self.ci[0] + self.ci[1]) / 2)), 30,
                      self.span(0, chart.fy(self.ci[0]) - chart.fy(self.ci[1]))[1] + 12, SEEN,
                      op, border=SEEN, border_w=2.2, radius=5, fill_op=0.0, lift=0.9)
        self.key(sc, 0.095, 0.405, [("confidence interval", FREQ),
                                    ("credible interval, flat prior", BAYES)], op, step=0.044)
        tx = 0.52
        a = op * smoothstep(0.3, 0.4, u)
        self.lines(sc, tx, 0.385, [
            ("n = %d, confidence: %.2f to %.2f" % (n, *self.ci), dark(FREQ, 0.2)),
            ("n = %d, credible: %.2f to %.2f" % (n, *self.cr), dark(BAYES, 0.3)),
            "same upper end; they close up as n grows"], a, size=0.032, step=0.045)
        b = op * smoothstep(0.5, 0.6, u)
        self.lines(sc, tx, 0.55, ["%s pseudo-experiments, true μ = %g:"
                                  % (commas(self.repeats), self.truth),
                                  "how many intervals contain %g?" % self.truth], b,
                   size=0.032, step=0.045)
        full = self.span(0.15)[0]
        for j, (name, ink) in enumerate((("confidence", FREQ), ("credible", BAYES))):
            go = (u - 0.58 - 0.08 * j) / 0.14
            y = 0.665 + 0.058 * j
            self.hbar(sc, tx + 0.105, y, full * self.cov_toys[j], ink, b, name,
                      "%s  (exact %s)" % (pct(self.cov_toys[j]), pct(self.cov_exact[j])), go,
                      thick=18.0, track=full)
        self.lines(sc, tx, 0.80, [
            ("confidence: never under %s, whatever μ" % pct(min(S.CL, self.conf_min)),
             dark(FREQ, 0.2)),
            ("credible: %.0f%% to %.0f%% as μ runs from 1 to 15"
             % (100 * self.cred_range[0], 100 * self.cred_range[1]), dark(BAYES, 0.3))],
            op * smoothstep(0.8, 0.9, u), size=0.031, step=0.043)
        self.tag(sc, "exact; bars: " + toys(self.repeats), op)

    def phase_5(self, sc, u, op):
        self.text(sc, 0.5, 0.315, "a known background b = %g: the count is signal plus "
                  "background, μ = s + %g, and s ≥ 0" % (self.b, self.b), op, strong=True,
                  size=HEAD)
        x0, x1, lo, hi = 0.20, 0.93, -3.6, 9.6

        def at(s):
            return x0 + (x1 - x0) * (s - lo) / (hi - lo)
        top, step = 0.42, 0.064
        rows = len(self.b_ns)
        bottom = top + step * (rows - 1) + 0.045
        w, h = self.span(at(0) - x0, bottom - 0.375)
        self.rect(sc, self.P((x0 + at(0)) / 2, (0.375 + bottom) / 2), w, h, soft(RED, 0.78), op,
                  radius=4, lift=0.1)
        self.text(sc, (x0 + at(0)) / 2, 0.385 + 0.012, "s < 0: not allowed", op,
                  ink=dark(RED, 0.2))
        self.rule(sc, (x0, bottom), (x1, bottom), AXIS, 0.45 * op, 1.4)
        for v in (-3, 0, 3, 6, 9):
            self.text(sc, at(v), bottom + 0.03, signed(v, 0), op)
        self.text(sc, (x0 + x1) / 2, bottom + 0.072, "signal s", op)
        for r, n in enumerate(self.b_ns):
            y = top + step * r + 0.022
            show = op * smoothstep(0.05 + 0.09 * r, 0.13 + 0.09 * r, u)
            self.text(sc, x0 - 0.02, y, "n = %d" % n, show, anchor="rm", strong=True,
                      size=0.032)
            self.bar(sc, (at(self.b_ci[r, 0]), y - 0.011), (at(self.b_ci[r, 1]), y - 0.011),
                     FREQ, show, thick=6.5)
            self.bar(sc, (at(self.b_cr[r, 0]), y + 0.011), (at(self.b_cr[r, 1]), y + 0.011),
                     BAYES, show, thick=6.5)
        self.key(sc, 0.635, 0.405, [("confidence interval for s", FREQ),
                                    ("credible interval, prior zero for s < 0", BAYES)],
                 op * smoothstep(0.1, 0.2, u), step=0.044)
        self.text(sc, 0.06, 0.925, "n = 0: the confidence interval is correct as a procedure "
                  "and empty as an answer", op * smoothstep(0.7, 0.8, u), anchor="lm")
        self.tag(sc, EXACT, op)


# ====================================================================== 119: Bayes as areas
ILL, WELL, FALSE = RED, BLUE, AMBER            # the hypothesis, its alternative, false alarms


class BayesTheoremAreas(Stats):
    """A population as a unit square: its width split by the prior, each column's height
    by how the test comes out. The posterior is one rectangle over the sum of two: for a
    rare disease, for a test repeated, for a kaon tag counted in seeded particles, and
    as the prior moves."""

    TITLE = "Bayes' theorem as areas"
    PHASES = (("the square", 10.0), ("the posterior", 11.0), ("test again", 9.0),
              ("a kaon tag", 12.0), ("the prior moves", 11.0))
    SIDE = 236.0

    def setup(self):
        s, d = self.study, S.bayes_data()
        m = self.meta = s.metadata
        self.dis, self.kaon = m["disease"], m["kaon"]
        self.people = S.PEOPLE
        a = np.asarray(s.panels[0].data["values"], float) * self.people
        self.n_tp, self.n_fn, self.n_fp, self.n_tn = (int(round(v)) for v in a)
        self.chain = [float(v) for v in s.panels[1].data["ys"][0]]
        self.xy = np.asarray(s.panels[2].data["xy"], float)
        self.kinds = np.asarray(s.panels[2].data["classes"], int)
        self.counts = [int(v) for v in s.panels[2].data["counts"]]
        self.grid = np.asarray(s.panels[3].data["x"], float)
        self.purity = np.asarray(s.panels[3].data["ys"], float)
        self.toy_x = np.asarray(s.panels[3].data["toy_x"], float)
        self.toys = np.asarray(s.panels[3].data["toys"], float)
        self.misids = S.MISIDS
        self.particles = S.PARTICLES
        inks = np.array([soft(WELL, 0.25), FALSE, soft(ILL, 0.35), ILL], np.float32)
        self.stages = [int(v) for v in np.linspace(0, len(self.xy), 25)]
        self.textures = [raster(self.xy[:k, 0], self.xy[:k, 1], (0, 1), (0, 1), (118, 118),
                                inks[self.kinds[:k]] if k else inks[0], fat=2)
                         for k in self.stages]
        del d

    def subtitle_for(self, k):
        dis, kn = self.dis, self.kaon
        return [
            "of %s people, 1 in 1,000 is ill: a sliver of the square. A test finds %g%% "
            "of the ill and wrongly flags %g%% of the healthy. Each column is split by its "
            "own test result" % (commas(self.people), 100 * dis["sensitivity"],
                                 100 * dis["false_positive"]),
            "someone tests positive. They are in one of two rectangles: %s ill, %s "
            "healthy. The chance they are ill is the first over the sum: %s. That is "
            "Bayes' theorem" % (commas(self.n_tp), commas(self.n_fp), pct(dis["posterior"], 2)),
            "the posterior is the prior for the next test. Each positive result widens "
            "the ill column: %s, then %s, then %s (if the tests err independently)"
            % tuple(pct(v) for v in self.chain[1:]),
            "particle identification is the same square: %g%% kaons, a tag that fires "
            "for %g%% of kaons and %g%% of pions. Of %s seeded particles, %s of the "
            "tagged are kaons; exact %s"
            % (100 * kn["prior"], 100 * kn["efficiency"], 100 * kn["misid"],
               commas(self.particles), pct(kn["purity_counted"]), pct(kn["purity"])),
            "the same tag on a sample with fewer kaons is less pure: at 1 kaon in 1,000 "
            "only %s of tagged particles are kaons. A pion rejection 25 times better "
            "brings that to %s (counted: %s)"
            % (pct(self.purity[0, 0]), pct(self.purity[2, 0]), pct(self.toys[2, 0])),
        ][k]

    # ------------------------------------------------------------------ the square
    def block(self, sc, x0, x1, y0, y1, ink, op, lift=0.3, border=None):
        w, h = self.span(x1 - x0, y1 - y0)
        if w <= 0 or h <= 0:
            return
        self.rect(sc, self.P((x0 + x1) / 2, (y0 + y1) / 2), max(w, 0.5), max(h, 0.5), ink, op,
                  radius=0.0, lift=lift, border=border, border_w=1.4)

    def square(self, sc, x0, y0, side, prior, hit, false, op, split=1.0, lit=1.0, least=1.6):
        """The unit square with its top-left corner at ``(x0, y0)``: the hypothesis's
        column ``prior`` wide (never thinner than ``least`` px), each column's positive
        share rising from the floor by ``split``; ``lit`` < 1 fades the negatives.
        Returns ``(column edge x, right x, floor y, height)``."""
        w, h = self.frac(side, side)
        cut = x0 + max(prior * w, self.frac(least)[0])
        self.block(sc, x0, x0 + w, y0, y0 + h, soft(WELL, 0.72 + 0.2 * (1 - lit)), op, 0.2)
        self.block(sc, x0, cut, y0, y0 + h, soft(ILL, 0.60 + 0.3 * (1 - lit)), op, 0.3)
        self.block(sc, cut, x0 + w, y0 + h * (1 - false * split), y0 + h, FALSE, op, 0.4)
        self.block(sc, x0, cut, y0 + h * (1 - hit * split), y0 + h, ILL, op, 0.5)
        self.rect(sc, self.P(x0 + w / 2, y0 + h / 2), side + 3, side + 3, AXIS, op,
                  border=AXIS, border_w=1.3, radius=1.0, fill_op=0.0, lift=0.6)
        return cut, x0 + w, y0 + h, h

    def phase_1(self, sc, u, op):
        dis = self.dis
        self.text(sc, 0.5, 0.315, "%s people as a square of area 1: across, who is ill; up, "
                  "who tests positive" % commas(self.people), op, strong=True, size=HEAD)
        split = smoothstep(0.42, 0.62, u)
        x0, y0 = 0.07, 0.385
        _, x1, _, h = self.square(sc, x0, y0, self.SIDE, dis["prior"], dis["sensitivity"],
                                  dis["false_positive"], op, split)
        self.text(sc, (x0 + x1) / 2, y0 + 0.2, "healthy", op, ink=dark(WELL, 0.2), size=HEAD)
        self.text(sc, (x0 + x1) / 2, y0 + 0.25, commas(self.n_fp + self.n_tn), op,
                  ink=dark(WELL, 0.2))
        self.arrow(sc, (x0 - 0.035, y0 - 0.02), (x0 - 0.002, y0 + 0.03), ILL,
                   op * smoothstep(0.05, 0.15, u), 1.0, 2.2)
        # the ill column again, 200 times wider, so that it can be seen
        zx = 0.43
        zoom = op * smoothstep(0.12, 0.24, u)
        zw = self.frac(self.SIDE * 0.2)[0]
        self.block(sc, zx, zx + zw, y0, y0 + h, soft(ILL, 0.60), zoom, 0.3)
        self.block(sc, zx, zx + zw, y0 + h * (1 - dis["sensitivity"] * split), y0 + h, ILL, zoom,
                   0.5)
        self.text(sc, zx + zw / 2, y0 + h + 0.035, "the ill column,", zoom, ink=dark(ILL, 0.2))
        self.text(sc, zx + zw / 2, y0 + h + 0.075, "200 times wider", zoom, ink=dark(ILL, 0.2))
        tx = 0.56
        self.lines(sc, tx, 0.40, [
            ("ill: 1 in 1,000", dark(ILL, 0.2)),
            ("%s × 0.001 = %s people" % (commas(self.people), commas(self.n_tp + self.n_fn)),
             dark(ILL, 0.2))], op * smoothstep(0.05, 0.15, u), size=0.033, step=0.046)
        late = op * smoothstep(0.5, 0.62, u)
        self.lines(sc, tx, 0.53, [
            ("the test, on the ill: %g%% positive" % (100 * dis["sensitivity"]), None),
            ("%s × %g = %s true positives" % (commas(self.n_tp + self.n_fn),
                                              dis["sensitivity"], commas(self.n_tp)),
             dark(ILL, 0.2))], late, size=0.033, step=0.046)
        later = op * smoothstep(0.66, 0.78, u)
        self.lines(sc, tx, 0.66, [
            ("on the healthy: %g%% positive" % (100 * dis["false_positive"]), None),
            ("%s × %g = %s false positives" % (commas(self.n_fp + self.n_tn),
                                               dis["false_positive"], commas(self.n_fp)),
             dark(FALSE, 0.4))], later, size=0.033, step=0.046)
        self.text(sc, tx, 0.80, "many more false alarms than true ones", later, anchor="lm",
                  strong=True, size=0.033)
        self.tag(sc, EXACT, op)

    def phase_2(self, sc, u, op):
        dis = self.dis
        self.text(sc, 0.5, 0.315, "a positive result: which of the two positive rectangles "
                  "is this person in?", op, strong=True, size=HEAD)
        x0, y0 = 0.07, 0.385
        lit = 1.0 - smoothstep(0.03, 0.15, u)
        _, x1, floor, _ = self.square(sc, x0, y0, self.SIDE, dis["prior"], dis["sensitivity"],
                                      dis["false_positive"], op, 1.0, lit)
        self.text(sc, (x0 + x1) / 2, floor + 0.035, "the positives: the bottom strip", op,
                  ink=dark(FALSE, 0.4))
        self.text(sc, (x0 + x1) / 2, floor + 0.075, "and the red sliver", op,
                  ink=dark(ILL, 0.2))
        # the two rectangles, laid end to end
        bx0, bx1, by = 0.42, 0.96, 0.44
        go = smoothstep(0.15, 0.32, u)
        share = dis["posterior"]
        mid = bx0 + (bx1 - bx0) * share
        self.block(sc, bx0, mid, by - 0.03, by + 0.03, ILL, op * go, 0.5)
        self.block(sc, mid, mid + (bx1 - mid) * go, by - 0.03, by + 0.03, FALSE, op * go, 0.4)
        self.pulse(sc, (x1 - 0.05, floor - 0.012), ((mid + bx1) / 2, by), (u - 0.1) / 0.2,
                   FALSE, op, bend=0.05)
        self.pulse(sc, (x0, floor - 0.1), (bx0 + 0.005, by), (u - 0.08) / 0.2, ILL, op, bend=0.1)
        named = op * smoothstep(0.3, 0.4, u)
        self.text(sc, bx0, by - 0.065, "%s ill" % commas(self.n_tp), named, anchor="lm",
                  ink=dark(ILL, 0.2), size=0.033)
        self.text(sc, bx1, by - 0.065, "%s healthy" % commas(self.n_fp), named, anchor="rm",
                  ink=dark(FALSE, 0.4), size=0.033)
        self.text(sc, bx0, 0.535, "everyone who tests positive: %s people"
                  % commas(self.n_tp + self.n_fp), named, anchor="lm")
        ans = op * smoothstep(0.42, 0.54, u)
        self.text(sc, bx0, 0.615, "P(ill | positive) = %s / (%s + %s) = %s"
                  % (commas(self.n_tp), commas(self.n_tp), commas(self.n_fp),
                     pct(share, 2)), ans, anchor="lm", size=0.037, ink=dark(ILL, 0.25))
        law = op * smoothstep(0.62, 0.74, u)
        self.lines(sc, bx0, 0.71, [
            "Bayes' theorem, with H = ill and + = positive:",
            ("P(H | +) = P(+ | H) P(H) / [ P(+ | H) P(H) + P(+ | not H) P(not H) ]", STRONG),
            "= %g × %g / (%g × %g + %g × %g)" % (
                dis["sensitivity"], dis["prior"], dis["sensitivity"], dis["prior"],
                dis["false_positive"], 1 - dis["prior"])], law, size=0.029, step=0.054)
        self.tag(sc, EXACT, op)

    def phase_3(self, sc, u, op):
        dis = self.dis
        self.text(sc, 0.5, 0.315, "test again: the posterior of one test is the prior of the "
                  "next", op, strong=True, size=HEAD)
        side = 168.0
        w, h = self.frac(side, side)
        for k in range(3):
            show = op * smoothstep(0.04 + 0.26 * k, 0.14 + 0.26 * k, u)
            x0, y0 = 0.085 + 0.315 * k, 0.43
            prior, after = self.chain[k], self.chain[k + 1]
            split = smoothstep(0.10 + 0.26 * k, 0.22 + 0.26 * k, u)
            self.square(sc, x0, y0, side, prior, dis["sensitivity"], dis["false_positive"],
                        show, split)
            self.text(sc, x0 + w / 2, y0 - 0.04, "prior %s" % pct(prior), show,
                      ink=dark(ILL, 0.2), size=0.033)
            got = show * smoothstep(0.2 + 0.26 * k, 0.3 + 0.26 * k, u)
            self.text(sc, x0 + w / 2, y0 + h + 0.04, "positive: P(ill) = %s" % pct(after), got,
                      strong=True, size=0.033)
            if k < 2:
                self.arrow(sc, (x0 + w + 0.025, y0 + h / 2), (x0 + 0.315 - 0.025, y0 + h / 2),
                           AXIS, got, smoothstep(0.24 + 0.26 * k, 0.32 + 0.26 * k, u), 2.6)
        self.text(sc, 0.5, 0.875, "red over red plus amber, each time: one positive is weak "
                  "evidence, three are strong", op * smoothstep(0.8, 0.9, u))
        self.tag(sc, EXACT, op)

    def phase_4(self, sc, u, op):
        kn, c = self.kaon, self.counts
        self.text(sc, 0.5, 0.315, "a kaon tag: every particle is a random point of the same "
                  "square", op, strong=True, size=HEAD)
        x0, y0 = 0.07, 0.385
        w, h = self.frac(self.SIDE, self.SIDE)
        k = min(int(smoothstep(0.04, 0.5, u) * (len(self.stages) - 1) + 1e-9),
                len(self.stages) - 1)
        centre = self.P(x0 + w / 2, y0 + h / 2)
        self.rect(sc, centre, self.SIDE + 6, self.SIDE + 6, PAPER, op, border=AXIS,
                  border_w=1.2, radius=2, lift=0.1)
        draw_image(self, sc, centre, self.textures[k], self.SIDE, self.SIDE, op, grid=0)
        cut = x0 + w * kn["prior"]
        self.rule(sc, (cut, y0), (cut, y0 + h), AXIS, 0.45 * op, 1.6)
        self.rule(sc, (x0, y0 + h * (1 - kn["efficiency"])), (cut, y0 + h * (1 - kn["efficiency"])),
                  ILL, 0.45 * op, 1.6)
        self.rule(sc, (cut, y0 + h * (1 - kn["misid"])), (x0 + w, y0 + h * (1 - kn["misid"])),
                  FALSE, 0.45 * op, 1.6)
        self.text(sc, x0 + w / 2, y0 + h + 0.035, "%s of %s particles drawn"
                  % (commas(self.stages[k]), commas(self.particles)), op)
        tx = 0.42
        self.key(sc, tx + 0.008, 0.40, [
            ("kaon, tagged: left of %g, below %g" % (kn["prior"], kn["efficiency"]), ILL),
            ("pion, tagged: right of %g, below %g" % (kn["prior"], kn["misid"]), FALSE)], op,
            step=0.046, size=0.032)
        a = op * smoothstep(0.2, 0.3, u)
        self.lines(sc, tx, 0.515, [
            ("exact areas: %g × %g = %g kaon," % (kn["prior"], kn["efficiency"],
                                                  kn["prior"] * kn["efficiency"]), None),
            ("%g × %g = %g pion;  purity %g / %g = %s" % (
                1 - kn["prior"], kn["misid"], (1 - kn["prior"]) * kn["misid"],
                kn["prior"] * kn["efficiency"],
                kn["prior"] * kn["efficiency"] + (1 - kn["prior"]) * kn["misid"],
                pct(kn["purity"])), None)], a, size=0.032, step=0.046)
        b = op * smoothstep(0.52, 0.62, u)
        self.text(sc, tx, 0.655, "counted in all %s particles (seed %d):"
                  % (commas(self.particles), self.meta["seed"]), b, anchor="lm", size=0.032)
        full = self.span(0.30)[0]
        top = float(c[0] + c[2])
        for j, (name, n, ink) in enumerate((("tagged kaons", c[0], ILL),
                                            ("tagged pions", c[2], FALSE))):
            self.hbar(sc, tx + 0.15, 0.715 + 0.052 * j, full * n / top, ink, b, name, commas(n),
                      (u - 0.56 - 0.06 * j) / 0.12, thick=18.0)
        self.text(sc, tx, 0.84, "purity = %s / (%s + %s) = %s" % (
            commas(c[0]), commas(c[0]), commas(c[2]), pct(kn["purity_counted"])),
            op * smoothstep(0.74, 0.84, u), anchor="lm", size=0.036, ink=dark(ILL, 0.25))
        self.tag(sc, "exact areas; counts: " + toys(self.particles, "seeded particles"), op)

    def phase_5(self, sc, u, op):
        kn = self.kaon
        self.text(sc, 0.5, 0.315, "the tag is the same; only the share of kaons in the sample "
                  "changes", op, strong=True, size=HEAD)
        g = self.grid
        k = int(np.clip(smoothstep(0.06, 0.62, u), 0, 1) * (len(g) - 1) + 1e-9)
        k = len(g) - 1 - k                                # from half kaons down to 1 in 1,000
        prior = float(g[k])
        side = 190.0
        x0, y0 = 0.065, 0.42
        w, h = self.frac(side, side)
        self.square(sc, x0, y0, side, prior, kn["efficiency"], kn["misid"], op)
        self.text(sc, x0 + w / 2, y0 - 0.04, "kaons: %s of the sample" % pct(prior, 1), op,
                  ink=dark(ILL, 0.2), size=0.032)
        self.text(sc, x0 + w / 2, y0 + h + 0.04, "purity %s" % pct(self.purity[0, k]), op,
                  strong=True, size=HEAD)
        chart = self.frame(((0.40, 0.95), (0.40, 0.79)), (g[0], g[-1]), (0.0, 1.04), xlog=True)
        self.axes(sc, chart, op, "kaon fraction of the sample (the prior)",
                  "purity: share of tagged particles that are kaons",
                  xticks=[(0.001, "0.1%"), (0.01, "1%"), (0.1, "10%"), (0.5, "50%")],
                  yticks=[(0.5, "50%"), (1.0, "100%")], grid=False)
        inks = (FALSE, VIOLET, TEAL)
        for i, ink in enumerate(inks):
            show = op * (1.0 if i == 0 else smoothstep(0.62 + 0.08 * i, 0.70 + 0.08 * i, u))
            self.curve(sc, chart, g, self.purity[i], ink, show, 1.0, 2.8)
            self.marks(sc, chart, self.toy_x, self.toys[i], ink, show, r=3.6)
        self.disk(sc, self.P(chart.fx(prior), chart.fy(self.purity[0, k])), 7.0, ILL, op,
                  glow=0.4, lift=1.8, outline=0.0)
        self.key(sc, 0.79, 0.625, [
            ("pions tagged: %g%%" % (100 * self.misids[0]), inks[0]),
            ("pions tagged: %g%%" % (100 * self.misids[1]), inks[1]),
            ("pions tagged: %g%%" % (100 * self.misids[2]), inks[2])], op, step=0.042)
        self.text(sc, 0.782, 0.752, "lines exact, dots counted", op, anchor="lm")
        self.tag(sc, "exact; dots: " + toys(self.particles, "seeded particles each"), op)


# ====================================================================== 120: priors
PRIOR_INK = (BLUE, GREEN, AMBER, VIOLET, RED)   # flat, optimistic, sceptical, narrow, dogmatic
PRIOR_SAYS = ("flat: no opinion", "optimistic: mostly heads", "sceptical: a fair coin",
              "narrow: surely near fair", "dogmatic: 0.4 to 0.6 only")


class PriorWashingOut(Stats):
    """One seeded run of coin flips updating a flat prior flip by flip; three priors
    converging on the same data; the distance between their posteriors; a prior that is
    zero where the truth is and never gets there; and all the distances measured."""

    TITLE = "A prior washing out"
    PHASES = (("flip by flip", 11.0), ("three priors", 12.0), ("distance", 9.0),
              ("a dogmatic prior", 12.0), ("measured", 9.0))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        self.truth = float(m["truth"])
        p0 = s.panels[0].data
        self.p = np.asarray(p0["p"], float)[::5]
        self.dens = np.asarray(p0["densities"], float)[:, :, ::5]
        self.ns = [int(v) for v in p0["flips"]]
        self.heads = [int(v) for v in p0["heads"]]
        self.first = [int(v) for v in p0["first"]]
        self.means = (self.dens * self.p).sum(-1) / self.dens.sum(-1)
        step = self.p[1] - self.p[0]
        self.gap = 0.5 * np.abs(self.dens - self.dens[:, :1]).sum(-1) * step
        self.checks = np.asarray(s.panels[1].data["x"], float)
        self.far = np.asarray(s.panels[1].data["ys"], float)
        self.low = np.asarray(s.panels[2].data["low"], float)
        self.high = np.asarray(s.panels[2].data["high"], float)
        self.runs = len(m["seeds"]) - 1
        self.post_mean = np.asarray(s.panels[3].data["ys"], float)
        self.flips = int(m["flips"])

    def at_checks(self, n):
        return int(np.flatnonzero(self.checks == n)[0])

    def subtitle_for(self, k):
        far, c = self.far, self.at_checks
        return [
            "a coin lands heads with unknown chance p (the simulation uses p = %g, seed "
            "%d). Start flat. Each head multiplies the curve by p, each tail by 1 − p: "
            "after 10 flips, %d heads, it has moved" % (self.truth, self.meta["seed"],
                                                       self.heads[10]),
            "three people start from different priors and see the same flips. After "
            "%s flips their posteriors have drawn together, means %.3f, %.3f and %.3f: "
            "the data outweigh where they began" % (commas(self.ns[-1]), *self.means[-1, :3]),
            "distance between two posteriors: half the area between the curves (0 the "
            "same, 1 nothing shared). To the flat prior's, after 10, 100 and 1,000 flips: "
            "sceptical %.2f, %.2f, %.2f" % (far[1, c(10)], far[1, c(100)], far[1, c(1000)]),
            "a prior that is zero outside 0.4 to 0.6 stays zero there: zero times any "
            "likelihood is zero. Its posterior piles against 0.6, mean %.3f after %s "
            "flips, while the truth is %g" % (self.post_mean[4, -1], commas(self.flips),
                                              self.truth),
            "after %s flips the distance to the flat prior's posterior is %.3f "
            "(optimistic), %.3f (sceptical), %.2f for the narrow prior, which needs far "
            "more data, and %.0f for the dogmatic one"
            % (commas(self.flips), far[0, -1], far[1, -1], far[2, -1], far[3, -1]),
        ][k]

    def step(self, u, a, b, last=None):
        """The index into the recorded flips counts as ``u`` runs from ``a`` to ``b``."""
        last = len(self.ns) - 1 if last is None else last
        return int(np.clip((u - a) / (b - a), 0, 1) * last + 1e-9)

    def posterior_chart(self, sc, box, i, which, op, top=None, label="chance of heads p"):
        """The posteriors ``which`` after the ``i``-th recorded number of flips."""
        peak = max(float(self.dens[i, k].max()) for k in which)
        top = max(4.2, 1.12 * peak) if top is None else top
        chart = self.frame(box, (0.0, 1.0), (0.0, top))
        self.axes(sc, chart, op, label, "", xticks=self.ticks((0, 0.2, 0.4, 0.6, 0.8, 1.0)),
                  yticks=[], grid=False)
        x = chart.fx(self.truth)
        self.rule(sc, (x, chart.y0), (x, chart.y1), STRONG, 0.45 * op, 2.0, dashes=14)
        for k in which:
            self.curve(sc, chart, self.p, np.minimum(self.dens[i, k], top), PRIOR_INK[k], op,
                       1.0, 2.8)
        return chart

    def phase_1(self, sc, u, op):
        k = self.step(u, 0.10, 0.80, 10)
        self.text(sc, 0.06, 0.315, "the flips, as they arrive:", op, anchor="lm", strong=True,
                  size=HEAD)
        for j in range(10):
            on = j < k
            head = self.first[j] == 1
            ink = GREEN if head else MAGENTA
            self.box(sc, 0.345 + 0.052 * j, 0.315, 34, 30, soft(ink, 0.5) if on else PAPER,
                     op * (1.0 if on else 0.35), ("H" if head else "T") if on else "",
                     ink=dark(ink, 0.45), border=ink if on else DIM, size=0.032,
                     glow=0.25 if j == k - 1 else 0.0)
        chart = self.posterior_chart(sc, ((0.08, 0.58), (0.42, 0.79)), k, (0,), op, top=3.6)
        self.text(sc, chart.fx(self.truth), chart.y0 - 0.025, "the truth, %g" % self.truth, op)
        h = self.heads[k]
        tx = 0.64
        self.lines(sc, tx, 0.43, [("flat prior: every p as likely", dark(BLUE, 0.2)),
                                  "a head: multiply the curve by p",
                                  "a tail: multiply it by (1 − p)",
                                  "then scale to area 1"], op, size=0.032, step=0.046)
        self.text(sc, tx, 0.665, "after %d flips: %d heads, %d tails" % (k, h, k - h), op,
                  anchor="lm", strong=True, size=0.033)
        self.text(sc, tx, 0.72, "posterior:  p%s (1 − p)%s × constant" % (sup(h), sup(k - h)),
                  op, anchor="lm", ink=dark(BLUE, 0.2), size=0.033)
        self.text(sc, tx, 0.775, "its mean: (%d + 1) / (%d + 2) = %.3f" % (h, k, (h + 1) / (k + 2)),
                  op, anchor="lm", size=0.032)
        self.tag(sc, "exact posterior of 10 seeded flips", op)

    def phase_2(self, sc, u, op):
        i = self.step(u, 0.14, 0.86)
        chart = self.posterior_chart(sc, ((0.08, 0.58), (0.40, 0.79)), i, (0, 1, 2), op)
        self.text(sc, 0.06, 0.325, "the same flips for all three:  %s flips, %s heads"
                  % (commas(self.ns[i]), commas(self.heads[i])), op, anchor="lm", strong=True,
                  size=HEAD)
        tx = 0.64
        self.key(sc, tx + 0.008, 0.43, [(PRIOR_SAYS[k], PRIOR_INK[k]) for k in range(3)], op,
                 step=0.048, size=0.032)
        self.text(sc, tx, 0.60, "where each posterior sits (its mean):", op, anchor="lm")
        full = self.span(0.22)[0]
        for k in range(3):
            self.hbar(sc, tx + 0.008, 0.65 + 0.05 * k, full * self.means[i, k], PRIOR_INK[k], op,
                      "", "%.3f" % self.means[i, k], thick=16.0, track=full)
        self.text(sc, tx, 0.81, "the truth: %g" % self.truth, op, anchor="lm", strong=True)
        del chart
        self.tag(sc, "exact posteriors of %s seeded flips" % commas(self.ns[-1]), op)

    def phase_3(self, sc, u, op):
        i = self.step(u, 0.10, 0.80)
        box = ((0.06, 0.40), (0.42, 0.74))
        chart = self.posterior_chart(sc, box, i, (0, 2), op, label="p")
        between = np.abs(self.dens[i, 0] - self.dens[i, 2])
        lo = np.minimum(self.dens[i, 0], self.dens[i, 2])
        top = chart.y_range[1]
        edges = np.linspace(0, 1, 61)
        mid = 0.5 * (edges[1:] + edges[:-1])
        a = np.minimum(np.interp(mid, self.p, lo), top)
        b = np.minimum(np.interp(mid, self.p, lo + between), top)
        for x, y0, y1 in zip(mid, a, b):
            if y1 - y0 > top * 0.004:
                self.bars(sc, chart, [x], [y1], [soft(RED, 0.45)], op, width=1 / 60.0, base=y0,
                          lift=0.2)
        self.text(sc, 0.06, 0.325, "after %s flips: flat and sceptical" % commas(self.ns[i]), op,
                  anchor="lm", strong=True, size=HEAD)
        self.text(sc, 0.23, 0.855, "half the area between them: %.2f" % self.gap[i, 2], op,
                  ink=dark(RED, 0.2), size=0.033)
        right = self.frame(((0.52, 0.95), (0.40, 0.76)), (1.0, 1000.0), (0.008, 1.2), xlog=True,
                           ylog=True)
        self.axes(sc, right, op, "flips seen", "distance to the flat prior's posterior",
                  xticks=[(1, "1"), (10, "10"), (100, "100"), (1000, "1,000")],
                  yticks=[(0.01, "0.01"), (0.1, "0.1"), (1.0, "1")], grid=False)
        keep = (self.checks >= 1) & (self.checks <= 1000)
        x = self.checks[keep]
        upto = max(float(self.ns[i]), 1.0)
        for k in (1, 2):
            seen = x <= upto
            if seen.sum() >= 2:
                self.curve(sc, right, x[seen], self.far[k - 1][keep][seen], PRIOR_INK[k], op,
                           1.0, 2.8)
            self.marks(sc, right, x[seen], self.far[k - 1][keep][seen], PRIOR_INK[k], op, r=4.0)
        self.key(sc, 0.545, 0.665, [("optimistic", PRIOR_INK[1]), ("sceptical", PRIOR_INK[2])],
                 op, step=0.044)
        self.tag(sc, "exact posteriors of one seeded run", op)

    def phase_4(self, sc, u, op):
        i = self.step(u, 0.16, 0.80)
        box = ((0.08, 0.58), (0.40, 0.79))
        top = max(6.0, min(1.12 * float(self.dens[i, 0].max()), 30.0))
        chart = self.frame(box, (0.0, 1.0), (0.0, top))
        self.block(sc, chart.fx(S.DOGMA[0]), chart.fx(S.DOGMA[1]), chart.y0, chart.y1,
                   soft(RED, 0.9), op)
        self.posterior_chart(sc, box, i, (0, 4), op, top=top)
        self.text(sc, 0.06, 0.325, "%s flips, %s heads" % (commas(self.ns[i]),
                                                          commas(self.heads[i])), op,
                  anchor="lm", strong=True, size=HEAD)
        tx = 0.64
        self.key(sc, tx + 0.008, 0.43, [(PRIOR_SAYS[0], PRIOR_INK[0]),
                                        (PRIOR_SAYS[4], PRIOR_INK[4])], op, step=0.048,
                 size=0.032)
        self.lines(sc, tx, 0.555, ["outside the shaded band the", "dogmatic prior is exactly 0;",
                                   "0 × any likelihood = 0"], op, size=0.032, step=0.046)
        self.text(sc, tx, 0.715, "posterior means:", op, anchor="lm")
        self.text(sc, tx, 0.762, "flat %.3f" % self.means[i, 0], op, anchor="lm",
                  ink=dark(BLUE, 0.2), size=0.034)
        self.text(sc, tx + 0.16, 0.762, "dogmatic %.3f" % self.means[i, 4], op, anchor="lm",
                  ink=dark(RED, 0.2), size=0.034)
        self.text(sc, tx, 0.81, "the truth: %g" % self.truth, op, anchor="lm", strong=True)
        self.tag(sc, "exact posteriors of %s seeded flips" % commas(self.ns[-1]), op)

    def block(self, sc, x0, x1, y0, y1, ink, op, lift=0.15):
        w, h = self.span(x1 - x0, y1 - y0)
        self.rect(sc, self.P((x0 + x1) / 2, (y0 + y1) / 2), w, h, ink, op, radius=0.0, lift=lift)

    def phase_5(self, sc, u, op):
        chart = self.frame(((0.10, 0.60), (0.37, 0.79)), (1.0, float(self.flips)), (0.008, 1.3),
                           xlog=True, ylog=True)
        self.axes(sc, chart, op, "flips seen", "distance to the flat prior's posterior",
                  xticks=[(1, "1"), (100, "100"), (10000, "10,000"), (100000, "100,000")],
                  yticks=[(0.01, "0.01"), (0.1, "0.1"), (1.0, "1")], grid=False)
        keep = self.checks >= 1
        x = self.checks[keep]
        go = smoothstep(0.05, 0.6, u)
        for k in (1, 2, 3, 4):
            self.curve(sc, chart, x, np.maximum(self.far[k - 1][keep], 0.008), PRIOR_INK[k], op,
                       go, 2.8)
            for a, b, xx in zip(self.low[k - 1][keep], self.high[k - 1][keep], x):
                if xx <= x[0] * (x[-1] / x[0]) ** go:
                    self.bar(sc, (chart.fx(xx), chart.fy(max(a, 0.008))),
                             (chart.fx(xx), chart.fy(max(b, 0.008))), soft(PRIOR_INK[k], 0.45),
                             op, thick=3.0, lift=0.2)
        tx = 0.66
        self.text(sc, tx, 0.385, "after %s flips:" % commas(self.flips), op, anchor="lm",
                  strong=True, size=0.033)
        for j, k in enumerate((1, 2, 3, 4)):
            y = 0.445 + 0.097 * j
            show = op * smoothstep(0.5 + 0.07 * j, 0.6 + 0.07 * j, u)
            self.box(sc, tx + 0.008, y, 14, 14, PRIOR_INK[k], show, radius=3)
            self.text(sc, tx + 0.026, y, PRIOR_SAYS[k], show, anchor="lm",
                      ink=dark(PRIOR_INK[k], 0.3), size=0.031)
            self.text(sc, tx + 0.026, y + 0.045, "distance %s" % (
                "%.3f" % self.far[k - 1, -1] if self.far[k - 1, -1] < 0.995 else "1: nothing "
                "shared"), show, anchor="lm", strong=True, size=0.031)
        self.text(sc, 0.35, 0.895, "lines: the run drawn; bars: the middle 68%% of %d other "
                  "runs" % self.runs, op)
        self.tag(sc, "exact posteriors of %d seeded runs" % (self.runs + 1), op)


# ====================================================================== 121: coverage
HOLDS, MISSES, NAIVE, EXACT_INK = GREEN, RED, AMBER, BLUE


class IntervalCoverage(Stats):
    """Intervals x ± σ drawn one by one around a known truth, then a hundred, then ten
    thousand; n ± √n at a small Poisson mean, count by count; its coverage against the
    true mean beside the exact interval's; and the same for an efficiency from 20
    trials."""

    TITLE = "What coverage means"
    PHASES = (("one interval", 8.0), ("repeat", 12.0), ("n ± √n", 12.0),
              ("against μ", 11.0), ("an efficiency", 11.0))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        g = m["gaussian"]
        self.mu, self.sigma = float(g["mu"]), float(g["sigma"])
        self.first, self.all = int(g["first"]), float(g["all"])
        self.x = np.asarray(s.panels[0].data["xy"], float)[:, 1]
        self.inside = np.asarray(s.panels[0].data["classes"], int) == 1
        self.run_x = np.asarray(s.panels[0].data["experiments"], float)
        self.run = np.asarray(s.panels[0].data["running"], float)
        self.repeats = int(m["repeats"])
        p1 = s.panels[1].data
        self.ex_pmf = np.asarray(p1["values"], float)
        self.ex_holds = np.asarray(p1["holds"], int) == 1
        self.ex_lo, self.ex_hi = np.asarray(p1["low"], float), np.asarray(p1["high"], float)
        self.ex = m["poisson_example"]
        p2 = s.panels[2].data
        self.mus, self.cover = np.asarray(p2["x"], float), np.asarray(p2["ys"], float)
        self.toy_mu, self.toy = np.asarray(p2["toy_x"], float), np.asarray(p2["toys"], float)
        p3 = s.panels[3].data
        self.ps, self.bcover = np.asarray(p3["x"], float), np.asarray(p3["ys"], float)
        self.toy_p, self.btoy = np.asarray(p3["toy_x"], float), np.asarray(p3["toys"], float)
        self.wald = np.asarray(p3["wald"], float)
        self.cp = np.asarray(p3["clopper_pearson"], float)
        self.trials = len(self.wald) - 1
        self.level = float(m["level"])

    def subtitle_for(self, k):
        m = self.meta
        yes = np.flatnonzero(self.ex_holds)
        return [
            "a measurement x scatters around the true value μ = %g with a known σ = %g. "
            "The interval x ± σ either contains μ or misses it; which, depends on the "
            "draw" % (self.mu, self.sigma),
            "coverage is how often. Of the first 100 seeded experiments %d contain μ; of "
            "all %s, %s. Exactly: %s. A 68%% interval is a procedure that is right 68%% "
            "of the time" % (self.first, commas(self.repeats), pct(self.all),
                             pct(self.level, 2)),
            "counting: n ± √n at a true mean of %g. Only n = %s give an interval that "
            "contains %g, so the coverage is their summed probability, %s, not 68%%"
            % (self.ex["mu"], ", ".join(str(v) for v in yes), self.ex["mu"],
               pct(self.ex["naive"])),
            "n ± √n covers between %.0f%% and %.0f%% of the time, depending on the true "
            "mean. The exact interval never falls below %s (lowest: %s), at the price "
            "of covering more than asked"
            % (100 * m["poisson_range"]["naive"][0], 100 * m["poisson_range"]["naive"][1],
               pct(self.level), pct(m["poisson_range"]["exact"][0])),
            "an efficiency from %d trials: the binomial error bar has zero width at "
            "%d of %d and covers as little as %.0f%% near 0 and 1. The Clopper-Pearson "
            "interval never covers less than %s"
            % (self.trials, self.trials, self.trials,
               100 * m["binomial_range"]["wald"][0],
               pct(m["binomial_range"]["clopper_pearson"][0])),
        ][k]

    def phase_1(self, sc, u, op):
        mu, sg = self.mu, self.sigma
        self.text(sc, 0.5, 0.315, "the model: x is drawn from a Gaussian, true μ = %g, "
                  "σ = %g" % (mu, sg), op, strong=True, size=HEAD)
        chart = self.frame(((0.06, 0.62), (0.36, 0.50)), (3.0, 17.0), (0.0, 0.21))
        xs = np.linspace(3, 17, 141)
        self.curve(sc, chart, xs, S.normal_pdf(xs, mu, sg), GREY, op, 1.0, 2.4)
        floor = 0.85
        self.rule(sc, (chart.x0, floor), (chart.x1, floor), AXIS, 0.45 * op, 1.4)
        for v in (4, 6, 8, 10, 12, 14, 16):
            self.text(sc, chart.fx(v), floor + 0.03, "%d" % v, op)
        self.text(sc, chart.x1 + 0.03, floor + 0.03, "x", op)
        x = chart.fx(mu)
        self.rule(sc, (x, 0.36), (x, floor), STRONG, 0.45 * op, 2.0, dashes=22)
        self.text(sc, x, floor + 0.075, "the truth: μ = %g" % mu, op, strong=True)
        for i in range(6):
            go = (u - 0.08 - 0.13 * i) / 0.1
            if go <= 0:
                break
            show = op * smoothstep(0, 1, go)
            y = 0.545 + 0.05 * i
            v, ok = float(self.x[i]), bool(self.inside[i])
            ink = HOLDS if ok else MISSES
            self.bar(sc, (chart.fx(v - sg), y), (chart.fx(v + sg), y), ink, show, thick=5.0)
            self.disk(sc, self.P(chart.fx(v), y), 5.5, ink, show, lift=1.4, outline=0.15)
            self.text(sc, 0.66, y, "x = %.1f:  %.1f to %.1f" % (v, v - sg, v + sg), show,
                      anchor="lm", size=0.031)
            self.text(sc, 0.99, y, "contains μ" if ok else "misses", show, anchor="rm",
                      ink=dark(ink, 0.3), size=0.031)
        self.tag(sc, "6 pseudo-experiments", op)

    def phase_2(self, sc, u, op):
        mu, sg = self.mu, self.sigma
        n = len(self.x)
        k = int(smoothstep(0.03, 0.42, u) * n + 1e-9)
        top = self.frame(((0.08, 0.95), (0.37, 0.60)), (-1.0, n), (2.5, 17.5))
        self.axes(sc, top, op, "", "", xticks=[], yticks=self.ticks((5, 10, 15)), grid=False)
        y = top.fy(mu)
        self.rule(sc, (top.x0, y), (top.x1, y), STRONG, 0.45 * op, 2.0)
        for i in range(k):
            ink = HOLDS if self.inside[i] else MISSES
            x = top.fx(i)
            self.bar(sc, (x, top.fy(self.x[i] - sg)), (x, top.fy(self.x[i] + sg)), ink, op,
                     thick=4.2)
        got = int(self.inside[:k].sum())
        self.text(sc, 0.08, 0.325, "%d experiments, each with its x ± σ:" % k, op, anchor="lm",
                  strong=True, size=HEAD)
        self.text(sc, 0.47, 0.325, "%d contain μ" % got, op, anchor="lm", ink=dark(HOLDS, 0.35),
                  size=HEAD)
        self.text(sc, 0.66, 0.325, "%d miss" % (k - got), op, anchor="lm",
                  ink=dark(MISSES, 0.2), size=HEAD)
        low = self.frame(((0.08, 0.60), (0.67, 0.83)), (10.0, float(self.repeats)), (0.55, 0.80),
                         xlog=True)
        later = op * smoothstep(0.46, 0.54, u)
        self.axes(sc, low, later, "experiments", "share that contain μ",
                  xticks=[(10, "10"), (100, "100"), (1000, "1,000"), (10000, "10,000")],
                  yticks=[(0.6, "60%"), (0.7, "70%"), (0.8, "80%")], grid=False)
        yy = low.fy(self.level)
        self.rule(sc, (low.x0, yy), (low.x1, yy), STRONG, 0.45 * later, 1.8, dashes=24)
        go = smoothstep(0.52, 0.86, u)
        self.curve(sc, low, self.run_x, self.run, HOLDS, later, go, 2.8)
        m = max(1, int(np.ceil(len(self.run_x) * go))) - 1
        self.lines(sc, 0.66, 0.70, [
            ("after %s experiments:" % commas(self.run_x[m]), None),
            ("%s contain μ" % pct(self.run[m]), dark(HOLDS, 0.35)),
            ("exact: %s (dashed)" % pct(self.level, 2), STRONG)], later, size=0.033,
            step=0.048)
        self.tag(sc, toys(self.repeats), op)

    def phase_3(self, sc, u, op):
        mu = float(self.ex["mu"])
        self.text(sc, 0.5, 0.315, "a count n with the usual error bar √n, when the true mean "
                  "is μ = %g" % mu, op, strong=True, size=HEAD)
        ks = np.arange(len(self.ex_pmf))
        left = self.frame(((0.08, 0.40), (0.42, 0.74)), (-0.7, 9.7), (0.0, 0.29))
        self.axes(sc, left, op, "count n", "P(n | μ = %g)" % mu,
                  xticks=self.ticks(range(0, 10)), yticks=self.ticks((0, 0.1, 0.2)), grid=False)
        judged = smoothstep(0.2, 0.55, u) * 8
        inks = [(HOLDS if self.ex_holds[k] else MISSES) if k < judged else soft(GREY, 0.3)
                for k in ks]
        self.bars(sc, left, ks, self.ex_pmf, inks, op)
        x0, x1 = 0.56, 0.95

        def at(v):
            return x0 + (x1 - x0) * v / 9.5
        rows = 7
        y0, step = 0.415, 0.05
        floor = y0 + step * (rows - 1) + 0.035
        self.rule(sc, (x0, floor), (x1, floor), AXIS, 0.45 * op, 1.4)
        for v in (0, 2, 4, 6, 8):
            self.text(sc, at(v), floor + 0.028, "%d" % v, op)
        self.text(sc, x1 + 0.025, floor + 0.028, "μ", op)
        self.rule(sc, (at(mu), y0 - 0.03), (at(mu), floor), STRONG, 0.45 * op, 2.0, dashes=18)
        for k in range(rows):
            show = op * smoothstep(0.1 + 0.06 * k, 0.16 + 0.06 * k, u)
            y = y0 + step * k
            ink = HOLDS if self.ex_holds[k] else MISSES
            self.text(sc, x0 - 0.015, y, "n = %d" % k, show, anchor="rm", size=0.031)
            self.bar(sc, (at(max(self.ex_lo[k], 0.0)), y), (at(self.ex_hi[k]), y), ink, show,
                     thick=5.5)
            self.disk(sc, self.P(at(k), y), 5.0, ink, show, lift=1.4, outline=0.15)
        yes = np.flatnonzero(self.ex_holds)
        total = op * smoothstep(0.62, 0.74, u)
        self.text(sc, 0.06, 0.865, "coverage = %s = %s = %s" % (
            " + ".join("P(%d)" % k for k in yes),
            " + ".join("%.3f" % self.ex_pmf[k] for k in yes), pct(self.ex["naive"])),
            total, anchor="lm", size=0.035, ink=dark(HOLDS, 0.4))
        self.text(sc, 0.06, 0.92, "n = 0 gives 0 ± 0: it can never contain μ", total,
                  anchor="lm", ink=dark(MISSES, 0.2))
        self.tag(sc, EXACT, op)

    def cover_chart(self, sc, box, xs, ys, toy_x, toy, op, go, xlabel, xticks, names):
        chart = self.frame(box, (0.0, float(xs[-1]) if xs[-1] > 1.5 else 1.0), (0.0, 1.04))
        self.axes(sc, chart, op, xlabel, "coverage: how often the interval contains the truth",
                  xticks=xticks, yticks=[(0.25, "25%"), (0.5, "50%"), (0.75, "75%"),
                                         (1.0, "100%")], grid=False)
        y = chart.fy(self.level)
        self.rule(sc, (chart.x0, y), (chart.x1, y), STRONG, 0.45 * op, 2.0, dashes=30)
        for i, ink in enumerate((NAIVE, EXACT_INK)):
            show = op * (1.0 if i == 0 else smoothstep(0.45, 0.55, go))
            self.curve(sc, chart, xs, ys[i], ink, show, go / 0.45 if i == 0 else
                       (go - 0.45) / 0.45, 2.4)
            self.marks(sc, chart, toy_x, toy[i], ink, show * smoothstep(0.9, 1.0, go), r=4.0,
                       ring=PAPER)
        self.key(sc, box[0][1] - 0.30, box[1][1] - 0.155, [(names[0], NAIVE),
                                                           (names[1], EXACT_INK)], op,
                 step=0.044)
        self.text(sc, box[0][1] - 0.308, box[1][1] - 0.06, "dashed: the 68.3% asked for", op,
                  anchor="lm")
        return chart

    def phase_4(self, sc, u, op):
        self.text(sc, 0.06, 0.325, "the same sum at every true mean μ:", op, anchor="lm",
                  strong=True, size=HEAD)
        chart = self.cover_chart(sc, ((0.09, 0.95), (0.42, 0.79)), self.mus, self.cover,
                                 self.toy_mu, self.toy, op, smoothstep(0.04, 0.8, u),
                                 "true mean μ", self.ticks((0, 5, 10, 15, 20)),
                                 ("n ± √n", "exact central interval"))
        mu = float(self.ex["mu"])
        self.disk(sc, self.P(chart.fx(mu), chart.fy(self.ex["naive"])), 6.5, MISSES, op,
                  glow=0.4, lift=1.8, outline=0.0)
        self.text(sc, 0.52, 0.325, "at μ = %g: %s, the case just counted"
                  % (mu, pct(self.ex["naive"])), op, anchor="lm", ink=dark(MISSES, 0.2),
                  size=0.032)
        self.tag(sc, "lines exact; dots: " + toys(self.repeats) + " each", op)

    def phase_5(self, sc, u, op):
        n = self.trials
        self.text(sc, 0.06, 0.325, "k of %d particles fire a detector: the efficiency ε is "
                  "near k / %d" % (n, n), op, anchor="lm", strong=True, size=HEAD)
        x0, x1 = 0.12, 0.38

        def at(v):
            return x0 + (x1 - x0) * (v - 0.6) / 0.4
        floor = 0.72
        self.rule(sc, (x0, floor), (x1, floor), AXIS, 0.45 * op, 1.4)
        for v in (0.6, 0.8, 1.0):
            self.text(sc, at(v), floor + 0.028, "%g" % v, op)
        self.text(sc, (x0 + x1) / 2, floor + 0.07, "interval for ε", op)
        for r, k in enumerate((n, n - 1, n - 3)):
            y = 0.44 + 0.095 * r
            show = op * smoothstep(0.05 + 0.1 * r, 0.15 + 0.1 * r, u)
            self.text(sc, x0 - 0.015, y, "%d of %d" % (k, n), show, anchor="rm", size=0.031)
            w = self.wald[k]
            self.bar(sc, (at(w[0]), y - 0.013), (at(w[1]), y - 0.013), NAIVE, show, thick=6.0)
            c = self.cp[k]
            self.bar(sc, (at(max(c[0], 0.6)), y + 0.013), (at(c[1]), y + 0.013), EXACT_INK, show,
                     thick=6.0)
        self.text(sc, 0.03, 0.86, "%d of %d: the error bar √(ε(1 − ε)/%d) is 0" % (n, n, n),
                  op * smoothstep(0.3, 0.4, u), anchor="lm", ink=dark(NAIVE, 0.4))
        self.cover_chart(sc, ((0.50, 0.98), (0.42, 0.79)), self.ps, self.bcover, self.toy_p,
                         self.btoy, op, smoothstep(0.3, 0.85, u), "true efficiency ε",
                         self.ticks((0, 0.25, 0.5, 0.75, 1.0)),
                         ("k/n ± binomial error", "Clopper-Pearson"))
        self.tag(sc, "lines exact; dots: " + toys(self.repeats) + " each", op)


# ====================================================================== 122: three distributions
BINOM, POIS, GAUSS = BLUE, AMBER, GREEN


class BinomialPoissonGaussian(Stats):
    """The count in one bin. Ten events that each land in it with chance 0.3: a
    binomial. More events, each less likely, the mean held at 3: the histogram settles on
    a Poisson. A larger mean: the Poisson settles on a Gaussian. The distances between
    them measured, and the tails, where the Gaussian stays wrong."""

    TITLE = "Binomial, Poisson, Gaussian"
    PHASES = (("ten chances", 10.0), ("to Poisson", 12.0), ("to Gaussian", 12.0),
              ("distances", 9.0), ("the tails", 11.0))

    def setup(self):
        s = self.study
        self.meta = s.metadata
        p0 = s.panels[0].data
        self.hits = np.asarray(p0["hits"], int)
        self.example = np.asarray(p0["example"], float)
        self.counted = np.asarray(p0["counted"], float)
        self.seq = np.asarray(p0["values"], float)
        self.trials = [int(v) for v in p0["trials"]]
        self.pois = np.asarray(p0["poisson"], float)
        self.seq_tv = np.asarray(p0["tv"], float)
        p1, p2, p3 = (s.panels[k].data for k in (1, 2, 3))
        self.tv_n, self.tv_b = np.asarray(p1["x"], float), np.asarray(p1["ys"], float)
        self.tv_mu, self.tv_g = np.asarray(p2["x"], float), np.asarray(p2["ys"], float)[0]
        self.means = [float(v) for v in p2["means"]]
        self.mean_tv = np.asarray(p2["at_means"], float)
        self.negative = np.asarray(p2["negative"], float)
        self.ratio = np.asarray(p3["values"], float)
        self.thresholds = np.asarray(p3["threshold"], float)
        self.tail_exact = np.asarray(p3["exact"], float)
        self.tail_gauss = np.asarray(p3["gaussian"], float)
        self.tail_means = [float(v) for v in p3["means"]]
        self.slots, self.p_slot, self.lam = S.SLOTS, S.P_SLOT, S.LAM
        self.repeats = S.BPG_REPEATS
        self.shapes = []                    # per mean: counts, their chances, the range drawn
        for mu in self.means:
            sd = math.sqrt(mu)
            lo, hi = max(0, int(mu - 4.5 * sd)), int(mu + 4.5 * sd) + 1
            self.shapes.append((np.arange(lo, hi + 1), S.poisson_pmf(mu, hi)[lo:], mu - 4.7 * sd,
                                mu + 4.7 * sd))
        self.tail_k = np.arange(0, 33)
        self.tail_pmf = S.poisson_pmf(self.tail_means[0], 32)

    def subtitle_for(self, k):
        t = self.seq_tv
        return [
            "%d events, each landing in one bin with chance %g. The count k in the bin "
            "follows the binomial distribution; %s seeded rows of %d agree with it to "
            "within %.3f" % (self.slots, self.p_slot, commas(self.repeats), self.slots,
                             self.meta["counted_vs_exact"]),
            "more events, each less likely, the mean n × p held at %g. The histogram "
            "settles on the Poisson distribution: distance %.3f at n = %d, %.4f at "
            "n = %s" % (self.lam, t[self.trials.index(10)], 10, t[self.trials.index(1000)],
                        commas(1000)),
            "now let the mean μ grow. A Poisson of mean μ approaches a Gaussian of mean "
            "μ and width √μ: distance %.2f at μ = 1, %.3f at μ = 10, %.3f at μ = 100"
            % (self.mean_tv[0], self.mean_tv[2], self.mean_tv[4]),
            "total-variation distance: half the summed difference in probability. "
            "Binomial to Poisson falls as 1/n if n × p is fixed, and not at all if p "
            "stays %g; Poisson to Gaussian falls as 1/√μ" % self.p_slot,
            "the tails are another matter. With μ = %g expected, %d or more is 5σ: "
            "exact chance %s, Gaussian %s, %.0f times too small. At μ = 1,000 it is "
            "still %.1f times" % (self.tail_means[0], self.thresholds[0, 4],
                                  sci(self.tail_exact[0, 4]), sci(self.tail_gauss[0, 4]),
                                  self.ratio[0, 4], self.ratio[2, 4]),
        ][k]

    def phase_1(self, sc, u, op):
        n, p = self.slots, self.p_slot
        self.text(sc, 0.06, 0.325, "%d events; each lands in this bin with chance %g" % (n, p),
                  op, anchor="lm", strong=True, size=HEAD)
        rows = len(self.hits)
        shown = int(smoothstep(0.03, 0.45, u) * rows + 1e-9)
        for r in range(rows):
            on = r < shown
            y = 0.40 + 0.052 * r
            for c in range(n):
                hit = bool(self.hits[r, c]) and on
                self.box(sc, 0.075 + 0.031 * c, y, 22, 20, BINOM if hit else PAPER,
                         op * (1.0 if on else 0.3), border=BINOM if hit else DIM, radius=3)
            if on:
                self.text(sc, 0.40, y, "k = %d" % self.hits[r].sum(), op, anchor="lm",
                          ink=dark(BINOM, 0.2), size=0.031)
        self.text(sc, 0.215, 0.835, "one row = one experiment", op)
        ks = np.arange(n + 1)
        chart = self.frame(((0.56, 0.96), (0.42, 0.72)), (-0.7, n + 0.7), (0.0, 0.30))
        self.axes(sc, chart, op, "count in the bin, k", "P(k)", xticks=self.ticks(range(0, 11, 2)),
                  yticks=self.ticks((0, 0.1, 0.2)), grid=False)
        self.bars(sc, chart, ks, self.example, soft(BINOM, 0.25), op,
                  grow=smoothstep(0.3, 0.5, u))
        self.marks(sc, chart, ks, self.counted, STRONG, op * smoothstep(0.55, 0.65, u), r=3.6)
        late = op * smoothstep(0.62, 0.74, u)
        self.text(sc, 0.56, 0.845, "bars: the binomial, exact;  dots: counted", late,
                  anchor="lm")
        self.text(sc, 0.06, 0.925, "P(3) = (10 choose 3) × 0.3³ × 0.7⁷ = 120 × 0.027 × 0.0824 "
                  "= %.3f" % self.example[3], late, anchor="lm", ink=dark(BINOM, 0.2))
        self.tag(sc, "exact; dots: " + toys(self.repeats, "rows"), op)

    def phase_2(self, sc, u, op):
        i = int(np.clip((u - 0.08) / 0.74, 0, 1) * (len(self.trials) - 1) + 1e-9)
        n = self.trials[i]
        ks = np.arange(self.seq.shape[1])
        chart = self.frame(((0.08, 0.58), (0.40, 0.79)), (-0.7, 12.7), (0.0, 0.36))
        self.axes(sc, chart, op, "count in the bin, k", "P(k)",
                  xticks=self.ticks(range(0, 13, 2)), yticks=self.ticks((0, 0.1, 0.2, 0.3)),
                  grid=False)
        self.bars(sc, chart, ks, self.seq[i], soft(BINOM, 0.2), op)
        for k, v in zip(ks, self.pois):
            self.bar(sc, (chart.fx(k - 0.46), chart.fy(v)), (chart.fx(k + 0.46), chart.fy(v)),
                     POIS, op, thick=4.0, lift=0.9)
        self.text(sc, 0.06, 0.325, "n = %s events, chance %g / %s = %s each:  mean n × p = %g"
                  % (commas(n), self.lam, commas(n), "%.4g" % (self.lam / n), self.lam), op,
                  anchor="lm", strong=True, size=HEAD)
        tx = 0.65
        self.key(sc, tx + 0.008, 0.43, [("binomial, n = %s" % commas(n), BINOM),
                                        ("Poisson, mean %g" % self.lam, POIS)], op,
                 step=0.048, size=0.032)
        self.lines(sc, tx, 0.555, ["Poisson: P(k) = μᵏ exp(−μ) / k!",
                                   "one number, the mean μ;", "n and p have dropped out"], op,
                   size=0.032, step=0.046)
        self.text(sc, tx, 0.715, "distance between the two:", op, anchor="lm")
        self.text(sc, tx, 0.77, "%.4f" % self.seq_tv[i], op, anchor="lm", strong=True,
                  size=0.046)
        self.tag(sc, EXACT, op)

    def phase_3(self, sc, u, op):
        i = int(np.clip((u - 0.06) / 0.76, 0, 1) * (len(self.means) - 1) + 1e-9)
        mu = self.means[i]
        ks, pmf, lo, hi = self.shapes[i]
        sd = math.sqrt(mu)
        chart = self.frame(((0.08, 0.58), (0.40, 0.79)), (lo, hi), (0.0, 1.12 * pmf.max()))
        ticks = (mu - 4 * sd, mu - 2 * sd, mu, mu + 2 * sd, mu + 4 * sd)
        self.axes(sc, chart, op, "count n", "",
                  xticks=[(v, "%.0f" % v) for v in ticks if v >= 0], yticks=[], grid=False)
        if lo < -0.5:
            self.block(sc, chart.x0, chart.fx(-0.5), chart.y0, chart.y1, soft(RED, 0.82), op)
        self.bars(sc, chart, ks, pmf, soft(POIS, 0.15), op, width=0.84 if len(ks) < 80 else 1.0)
        xs = np.linspace(lo, hi, 161)
        self.curve(sc, chart, xs, S.normal_pdf(xs, mu, sd), GAUSS, op, 1.0, 3.0)
        self.text(sc, 0.06, 0.325, "a bin that expects μ = %s events:  σ = √μ = %.3g"
                  % (commas(mu), sd), op, anchor="lm", strong=True, size=HEAD)
        tx = 0.65
        self.key(sc, tx + 0.008, 0.43, [("Poisson, mean μ", POIS),
                                        ("Gaussian, mean μ, width √μ", GAUSS)], op, step=0.048,
                 size=0.032)
        self.text(sc, tx, 0.56, "distance between the two:", op, anchor="lm")
        self.text(sc, tx, 0.615, "%.4f" % self.mean_tv[i], op, anchor="lm", strong=True,
                  size=0.046)
        self.lines(sc, tx, 0.705, [
            ("the Gaussian puts %s" % (pct(self.negative[i]) if self.negative[i] > 5e-4
                                       else "almost nothing"), dark(RED, 0.2)),
            ("on counts below zero", dark(RED, 0.2)),
            ("(the shaded part)" if lo < -0.5 else "", dark(RED, 0.2))], op, size=0.032,
            step=0.046)
        self.tag(sc, EXACT, op)

    def block(self, sc, x0, x1, y0, y1, ink, op, lift=0.15):
        w, h = self.span(x1 - x0, y1 - y0)
        self.rect(sc, self.P((x0 + x1) / 2, (y0 + y1) / 2), w, h, ink, op, radius=0.0, lift=lift)

    def phase_4(self, sc, u, op):
        go = smoothstep(0.05, 0.6, u)
        left = self.frame(((0.09, 0.47), (0.40, 0.72)), (4.0, 10000.0), (1e-5, 1.0), xlog=True,
                          ylog=True)
        self.axes(sc, left, op, "events n", "binomial: distance to ...",
                  xticks=[(10, "10"), (100, "100"), (1000, "1,000"), (10000, "10,000")],
                  yticks=[(1e-4, "0.0001"), (1e-2, "0.01"), (1.0, "1")], grid=False)
        inks = (BINOM, RED, GAUSS)
        for k in range(3):
            self.curve(sc, left, self.tv_n, self.tv_b[k], inks[k], op, go, 2.8)
        self.key(sc, 0.075, 0.835, [("Poisson, n × p = %g" % self.lam, BINOM),
                                    ("Poisson, p = %g" % self.p_slot, RED)], op, step=0.044)
        self.key(sc, 0.305, 0.835, [("Gaussian, p = %g" % self.p_slot, GAUSS)], op)
        right = self.frame(((0.60, 0.96), (0.40, 0.72)), (0.3, 1000.0), (1e-3, 1.0), xlog=True,
                           ylog=True)
        self.axes(sc, right, op, "mean μ", "Poisson: distance to the Gaussian",
                  xticks=[(1, "1"), (10, "10"), (100, "100"), (1000, "1,000")],
                  yticks=[(1e-3, "0.001"), (1e-2, "0.01"), (0.1, "0.1"), (1.0, "1")],
                  grid=False)
        self.curve(sc, right, self.tv_mu, self.tv_g, POIS, op, go, 2.8)
        self.marks(sc, right, self.means, self.mean_tv, POIS, op * smoothstep(0.6, 0.7, u),
                   r=4.4)
        self.lines(sc, 0.60, 0.835, ["100 times the mean: 10 times closer", "(dots: the means "
                                     "just drawn)"], op * smoothstep(0.6, 0.7, u), step=0.044)
        self.tag(sc, EXACT, op)

    def phase_5(self, sc, u, op):
        mu = self.tail_means[0]
        sd = math.sqrt(mu)
        self.text(sc, 0.06, 0.325, "μ = %g expected in the bin, on a logarithmic scale" % mu, op,
                  anchor="lm", strong=True, size=HEAD)
        chart = self.frame(((0.09, 0.52), (0.40, 0.79)), (-0.5, 32.5), (1e-10, 1.0), ylog=True)
        self.axes(sc, chart, op, "count n", "probability",
                  xticks=self.ticks((0, 10, 20, 30)),
                  yticks=[(1e-9, "10⁻⁹"), (1e-6, "10⁻⁶"), (1e-3, "10⁻³"), (1.0, "1")],
                  grid=False)
        xs = np.linspace(-0.5, 32.5, 133)
        xs = xs[S.normal_pdf(xs, mu, sd) > 1e-10]
        self.curve(sc, chart, xs, S.normal_pdf(xs, mu, sd), GAUSS, op,
                   smoothstep(0.05, 0.35, u), 3.0)
        self.marks(sc, chart, self.tail_k, np.maximum(self.tail_pmf, 1e-10), POIS,
                   op * smoothstep(0.1, 0.3, u), r=4.4)
        self.key(sc, 0.125, 0.70, [("Poisson", POIS), ("Gaussian", GAUSS)], op, step=0.044)
        tx = 0.58
        cols = (tx, tx + 0.105, tx + 0.215, tx + 0.335)
        for x, head in zip(cols, ("n at least", "exact", "Gaussian", "ratio")):
            self.text(sc, x, 0.40, head, op, anchor="lm", strong=True)
        for r, j in enumerate((1, 2, 3, 4)):
            y = 0.455 + 0.052 * r
            show = op * smoothstep(0.3 + 0.08 * r, 0.4 + 0.08 * r, u)
            vals = ("%d (%dσ)" % (self.thresholds[0, j], j + 1), sci(self.tail_exact[0, j]),
                    sci(self.tail_gauss[0, j]), "× %.3g" % self.ratio[0, j])
            for x, v, ink in zip(cols, vals, (None, dark(POIS, 0.4), dark(GAUSS, 0.35),
                                              dark(RED, 0.2))):
                self.text(sc, x, y, v, show, anchor="lm", ink=ink, size=0.031)
        late = op * smoothstep(0.7, 0.8, u)
        self.lines(sc, tx, 0.70, [
            "the same 5σ ratio at larger means:",
            ("μ = 100: × %.2g      μ = 1,000: × %.2g" % (self.ratio[1, 4], self.ratio[2, 4]),
             dark(RED, 0.2)),
            "a discovery needs the exact tail"], late, size=0.032, step=0.048)
        self.tag(sc, EXACT, op)


# ====================================================================== 123: central limit
DICE, DECAY, WIDE = BLUE, AMBER, RED


class CentralLimitTheorem(Stats):
    """Two dice rolled and averaged; the mean of 1, 2, 5 and 30 dice with the Gaussian
    over it; the same for decay times, which start lopsided; the width falling as
    σ/√N and the distance to the Gaussian measured; and a Breit-Wigner, whose mean is as
    wide for 30 draws as for one."""

    TITLE = "The central limit theorem"
    PHASES = (("two dice", 9.0), ("more dice", 11.0), ("decay times", 11.0),
              ("width, distance", 11.0), ("no bell", 12.0))
    BOXES = tuple(((0.065 + 0.24 * i, 0.255 + 0.24 * i), (0.42, 0.69)) for i in range(4))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (pn.data for pn in s.panels)
        self.ns = [int(v) for v in p0["ns"]]
        self.rolls = np.asarray(p0["rolls"], int)
        self.dice = [np.asarray(p0["values"], float)[i][:5 * n + 1]
                     for i, n in enumerate(self.ns)]
        self.dice_counted = [np.asarray(p0["counted_%d" % n], float) for n in self.ns]
        self.exp_x = np.asarray(p1["x"], float)
        self.exp_hist = np.asarray(p1["ys"], float)
        self.width_n = np.asarray(p2["x"], float)
        self.far_exact = np.asarray(p2["ys"], float)
        self.far = np.asarray(p2["measured"], float)
        self.sd = np.asarray(p2["sd"], float)
        self.sd_exact = np.asarray(p2["sd_exact"], float)
        self.run_x = np.asarray(p3["x"], float)
        self.run_bw = np.asarray(p3["ys"], float)
        self.run_exp = np.asarray(p3["settles"], float)
        self.half = np.asarray(p3["half_width"], float)
        self.bw = [np.asarray(p3["density_%d" % n], float) for n in self.ns]
        self.bw_x = np.linspace(*S.CAUCHY_RANGE, S.BINS + 1)
        self.bw_x = 0.5 * (self.bw_x[1:] + self.bw_x[:-1])
        self.repeats = int(m["repeats"])
        self.sigma = m["sigma"]

    def at(self, n):
        return int(np.flatnonzero(self.width_n == n)[0])

    def subtitle_for(self, k):
        a = self.at
        return [
            "one die: every face equally likely. The mean of two dice is not flat: of "
            "the 36 equally likely pairs, six average 3.5 and only one averages 1. "
            "Middling means have more ways to happen",
            "average N dice and the distribution of the mean narrows and rounds into a "
            "Gaussian of width σ/√N, with σ = %.2f for one die: %.2f at N = 5, %.2f at "
            "N = 30 (%s pseudo-experiments: %.2f, %.2f)"
            % (self.sigma["die"], self.sd_exact[0, a(5)], self.sd_exact[0, a(30)],
               commas(self.repeats), self.sd[0, a(5)], self.sd[0, a(30)]),
            "it does not need a symmetric start. Decay times are lopsided (mean life τ "
            "= 1, σ = 1); the mean of N of them still becomes a bell around 1, though "
            "at N = 5 it is visibly skewed",
            "the width of the mean follows σ/√N. The largest gap between its cumulative "
            "distribution and the Gaussian's falls from %.3f (one decay time) to %.3f "
            "at N = 30 and %.3f at N = 100"
            % (self.far_exact[1, 0], self.far_exact[1, a(30)], self.far_exact[1, a(100)]),
            "the theorem needs a finite σ. A Breit-Wigner line shape has none: the mean "
            "of N draws is as wide as one draw (half-width %.2f at N = 1, %.2f at 30, "
            "%.2f at 100) and its running mean never settles"
            % (self.half[0], self.half[a(30)], self.half[a(100)]),
        ][k]

    def die(self, sc, fx, fy, v, op, ink=DICE, size=30):
        self.box(sc, fx, fy, size, size, soft(ink, 0.55), op, str(v), ink=dark(ink, 0.45),
                 border=ink, size=0.033, radius=5)

    def phase_1(self, sc, u, op):
        self.text(sc, 0.06, 0.325, "roll two dice, take their mean:", op, anchor="lm",
                  strong=True, size=HEAD)
        rolls = self.rolls
        k = int(smoothstep(0.04, 0.6, u) * len(rolls) + 1e-9)
        for r in range(min(k, 6)):
            j = k - 1 - r                              # the newest roll on top
            y = 0.41 + 0.066 * r
            fade = op * (1.0 if r == 0 else 0.55)
            self.die(sc, 0.085, y, rolls[j, 0], fade)
            self.die(sc, 0.135, y, rolls[j, 1], fade)
            self.text(sc, 0.175, y, "mean %.1f" % rolls[j].mean(), fade, anchor="lm",
                      strong=r == 0, size=0.032)
        chart = self.frame(((0.38, 0.95), (0.42, 0.74)), (0.6, 6.4), (0.0, 0.2))
        self.axes(sc, chart, op, "mean of two dice", "",
                  xticks=self.ticks((1, 2, 3, 4, 5, 6)), yticks=[], grid=False)
        means = np.arange(2, 13) / 2.0
        self.bars(sc, chart, means, self.dice[1], soft(DICE, 0.62), op, width=0.42)
        stack = {}
        for j in range(k):
            v = rolls[j].mean()
            n = stack[v] = stack.get(v, 0) + 1
            self.disk(sc, self.P(chart.fx(v), chart.y1 - self.frac(0, 9 + 15 * (n - 1))[1]), 6.5,
                      DICE, op, lift=1.4, outline=0.15)
        ways = [min(t - 1, 13 - t) for t in range(2, 13)]
        late = op * smoothstep(0.6, 0.7, u)
        for v, w, p in zip(means, ways, self.dice[1]):
            self.text(sc, chart.fx(v), chart.fy(p) - 0.028, "%d" % w, late, size=0.030)
        self.text(sc, 0.665, 0.345, "bars: exact, with the number of pairs out of 36", late)
        self.text(sc, 0.38, 0.875, "dots: the %d seeded rolls;  P(mean = 3.5) = 6 / 36 = %.3f"
                  % (len(rolls), self.dice[1][5]), late, anchor="lm")
        self.tag(sc, "exact; dots: %d seeded rolls" % len(rolls), op)

    def panel(self, i, u, op, start=0.04, each=0.17):
        return op * smoothstep(start + each * i, start + 0.1 + each * i, u)

    def phase_2(self, sc, u, op):
        self.text(sc, 0.06, 0.325, "the mean of N dice, and the Gaussian of the same mean and "
                  "width (green)", op, anchor="lm", strong=True, size=HEAD)
        sd1 = self.sigma["die"]
        for i, n in enumerate(self.ns):
            show = self.panel(i, u, op)
            pmf = self.dice[i]
            mean = np.arange(n, 6 * n + 1) / n
            chart = self.frame(self.BOXES[i], (0.5, 6.5), (0.0, 1.18 * pmf.max()))
            self.axes(sc, chart, show, "mean", "", xticks=self.ticks((1, 3.5, 6)), yticks=[],
                      grid=False)
            self.bars(sc, chart, mean, pmf, soft(DICE, 0.3), show, width=0.84 / n)
            if n > 1:
                xs = np.linspace(0.5, 6.5, 121)
                self.curve(sc, chart, xs, S.normal_pdf(xs, 3.5, sd1 / math.sqrt(n)) / n, GAUSS,
                           show, 1.0, 2.8)
            if n <= 5:
                self.marks(sc, chart, mean, self.dice_counted[i], STRONG, show, r=3.0)
            x = 0.5 * (self.BOXES[i][0][0] + self.BOXES[i][0][1])
            self.text(sc, x, 0.385, "N = %d" % n, show, strong=True, size=HEAD)
            self.text(sc, x, 0.815, "σ/√N = %.2f" % (sd1 / math.sqrt(n)), show,
                      ink=dark(DICE, 0.2), size=0.032)
        self.text(sc, 0.06, 0.875, "bars: exact;  dots: %s pseudo-experiments;  one die: "
                  "σ = %.2f" % (commas(self.repeats), sd1), op, anchor="lm")
        self.tag(sc, "exact; dots: " + toys(self.repeats), op)

    def phase_3(self, sc, u, op):
        self.text(sc, 0.06, 0.325, "decay times, mean life τ = 1: the mean of N of them", op,
                  anchor="lm", strong=True, size=HEAD)
        xs = np.linspace(0.0, 4.0, 161)
        for i, n in enumerate(self.ns):
            show = self.panel(i, u, op)
            exact = S.exp_mean_pdf(np.maximum(xs, 1e-9), n)
            top = 1.18 * max(float(self.exp_hist[i].max()), float(exact.max()))
            lo, hi = (0.0, 4.0) if n < 30 else (0.0, 2.0)
            chart = self.frame(self.BOXES[i], (lo, hi), (0.0, top))
            self.axes(sc, chart, show, "mean / τ", "",
                      xticks=self.ticks((0, 1, 2, 3, 4) if n < 30 else (0, 1, 2)), yticks=[],
                      grid=False)
            keep = self.exp_x < hi
            self.bars(sc, chart, self.exp_x[keep], self.exp_hist[i][keep], soft(DECAY, 0.3),
                      show, width=self.exp_x[1] - self.exp_x[0])
            if n > 1:
                g = xs[xs <= hi]
                self.curve(sc, chart, g, np.minimum(S.normal_pdf(g, 1.0, 1 / math.sqrt(n)), top),
                           GAUSS, show, 1.0, 2.8)
            x = 0.5 * (self.BOXES[i][0][0] + self.BOXES[i][0][1])
            self.text(sc, x, 0.385, "N = %d" % n, show, strong=True, size=HEAD)
            self.text(sc, x, 0.815, "σ/√N = %.2f" % (1 / math.sqrt(n)), show,
                      ink=dark(DECAY, 0.4), size=0.032)
        self.text(sc, 0.06, 0.875, "bars: %s pseudo-experiments per N;  green: the Gaussian of "
                  "the same mean and width" % commas(self.repeats), op, anchor="lm")
        self.tag(sc, toys(self.repeats) + " per N", op)

    def phase_4(self, sc, u, op):
        n = self.width_n
        go = smoothstep(0.05, 0.55, u)
        left = self.frame(((0.09, 0.46), (0.40, 0.70)), (1.0, 100.0), (0.08, 2.2), xlog=True,
                          ylog=True)
        self.axes(sc, left, op, "N: draws averaged", "width of the mean (its σ)",
                  xticks=[(1, "1"), (10, "10"), (100, "100")],
                  yticks=[(0.1, "0.1"), (1.0, "1")], grid=False)
        right = self.frame(((0.60, 0.97), (0.40, 0.70)), (1.0, 100.0), (0.008, 0.25), xlog=True,
                           ylog=True)
        self.axes(sc, right, op, "N: draws averaged", "largest gap to the Gaussian's "
                  "cumulative curve", xticks=[(1, "1"), (10, "10"), (100, "100")],
                  yticks=[(0.01, "0.01"), (0.1, "0.1")], grid=False)
        for k, ink in enumerate((DICE, DECAY)):
            self.curve(sc, left, n, self.sd_exact[k], ink, op, go, 2.6)
            self.marks(sc, left, n, self.sd[k], ink, op * smoothstep(0.5, 0.6, u), r=4.4,
                       ring=PAPER)
            self.curve(sc, right, n, self.far_exact[k], ink, op, go, 2.6)
            self.marks(sc, right, n, self.far[k], ink, op * smoothstep(0.5, 0.6, u), r=4.4,
                       ring=PAPER)
        self.key(sc, 0.075, 0.825, [("dice: σ/√N, σ = %.2f" % self.sigma["die"], DICE),
                                    ("decay times: σ/√N, σ = 1", DECAY)], op, step=0.044)
        self.lines(sc, 0.56, 0.825, ["lines: exact;  dots: %s pseudo-experiments"
                                     % commas(self.repeats),
                                     "4 times the draws: half the width, half the gap"], op,
                   step=0.044)
        self.tag(sc, "exact; dots: " + toys(self.repeats), op)

    def phase_5(self, sc, u, op):
        self.text(sc, 0.06, 0.325, "a Breit-Wigner (Cauchy) line shape, centre 0, half-width 1: "
                  "the mean of N draws", op, anchor="lm", strong=True, size=HEAD)
        xs = np.linspace(-8, 8, 161)
        exact = 1.0 / (math.pi * (1.0 + xs * xs))
        for i, n in enumerate(self.ns):
            show = self.panel(i, u, op, 0.03, 0.08)
            box = ((0.065 + 0.125 * i, 0.165 + 0.125 * i), (0.43, 0.60))
            chart = self.frame(box, (-8.0, 8.0), (0.0, 0.36))
            self.axes(sc, chart, show, "", "", xticks=[(0, "0")], yticks=[], grid=False)
            self.bars(sc, chart, self.bw_x, self.bw[i], soft(WIDE, 0.3), show,
                      width=self.bw_x[1] - self.bw_x[0])
            self.curve(sc, chart, xs, exact, STRONG, 0.45 * show, 1.0, 1.6)
            self.text(sc, 0.5 * (box[0][0] + box[0][1]), 0.39, "N = %d" % n, show, strong=True)
        note = op * smoothstep(0.34, 0.44, u)
        self.lines(sc, 0.065, 0.70, [
            ("the same shape for every N", dark(WIDE, 0.2)),
            ("half-width of the mean, %s" % commas(self.repeats), None),
            ("pseudo-experiments:  N = 1: %.2f" % self.half[0], None),
            ("N = 30: %.2f     N = 100: %.2f" % (self.half[self.at(30)],
                                                 self.half[self.at(100)]), None)],
            note, step=0.045)
        lo = 5.0 * math.floor(min(self.run_bw.min(), -1.0) / 5.0)
        hi = 5.0 * math.ceil(max(self.run_bw.max(), 2.0) / 5.0)
        chart = self.frame(((0.62, 0.97), (0.42, 0.70)), (1.0, float(self.run_x[-1])),
                           (lo, hi), xlog=True)
        self.axes(sc, chart, op, "draws so far", "running mean of one sequence",
                  xticks=[(1, "1"), (100, "100"), (10000, "10,000")],
                  yticks=[(v, signed(v, 0)) for v in np.arange(lo, hi + 1, 5.0)], grid=False)
        go = smoothstep(0.4, 0.9, u)
        for k in range(len(self.run_bw)):
            self.curve(sc, chart, self.run_x, self.run_bw[k], (WIDE, MAGENTA, VIOLET)[k], op,
                       go, 2.2)
        self.curve(sc, chart, self.run_x, self.run_exp[0], DECAY, op, go, 2.6)
        self.key(sc, 0.635, 0.875, [("three Breit-Wigner sequences", WIDE)], op)
        self.key(sc, 0.635, 0.833, [("decay times: settles at 1", DECAY)], op)
        self.tag(sc, toys(self.repeats) + "; 4 seeded sequences", op)


ANIMATIONS = {
    "frequentist-and-bayesian": FrequentistAndBayesian,
    "bayes-theorem-areas": BayesTheoremAreas,
    "prior-washing-out": PriorWashingOut,
    "interval-coverage": IntervalCoverage,
    "binomial-poisson-gaussian": BinomialPoissonGaussian,
    "central-limit-theorem": CentralLimitTheorem,
}

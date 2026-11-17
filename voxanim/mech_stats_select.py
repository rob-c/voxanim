"""Selecting events, and what a selection does to what is measured: animations 137-140
of the "next" catalogue (see :mod:`voxanim.mech`).

* :class:`MovingACut`: one cut sliding across a signal and a background histogram;
* :class:`CutsVsMultivariate`: two thresholds against one line, then a curved boundary;
* :class:`SidebandBackground`: the background under a peak, from the spectrum beside it;
* :class:`SelectionSculpting`: a trigger and a cut giving a smooth spectrum a peak.

They draw the experiments of :mod:`voxanim.study_stats_select`, which records every
array shown here. Nothing on screen is measured data: every event is pseudo-data from a
model that is written on screen, and each phase carries a tag (bottom right) saying
whether it shows the model's exact formulas, one seeded draw, or many seeded repeats.
"""

from __future__ import annotations

import numpy as np

from . import study_stats_select as S
from .easing import ease_in_out, smoothstep
from .losschart import pow10
from .mech import (
    AMBER,
    AXIS,
    BLUE,
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
from .mech_random import Random, commas, raster

EXACT = "exact: the model's formulas"
DATA = "pseudo-data: one seeded draw from the model"
KINDS = ("exact", "pseudo-data", "pseudo-experiments")
BIG, HEAD = 0.034, 0.032
INK = (0.13, 0.15, 0.17)


def toys(n):
    """The tag of a phase that shows ``n`` seeded repeats."""
    return "pseudo-experiments: %s seeded repeats" % commas(n)


def pct(v, digits=0):
    return "%.*f%%" % (digits, 100.0 * float(v))


def minus(s):
    """A number's text with a real minus sign."""
    return str(s).replace("-", "−")


class Select(Random):
    """What the four animations share: histograms on a chart, cut lines, shaded
    ranges, legends, and scatters with the kept region tinted."""

    def bars(self, sc, chart, edges, values, ink, op, base=None, split=None, pale=None,
             gap=1.0, front=0):
        """A histogram on ``chart``: one bar per bin from ``base`` (default: the bottom
        of the chart) to ``base + values``. ``split``: the part of every bar left of
        this x is drawn in ``pale``. ``front``: 1, 2 ... for a histogram drawn over
        another (bars at the same depth hide one another at random)."""
        if op <= 0.01:
            return
        floor = chart.y_range[0]
        for k, v in enumerate(values):
            lo = floor if base is None else float(base[k])
            hi = float(v) if base is None else lo + float(v)
            if hi <= lo:
                continue
            y0, y1 = chart.fy(hi), chart.fy(lo)
            h = self.span(0, y1 - y0)[1]
            if h < 0.5:
                continue
            parts = [(float(edges[k]), float(edges[k + 1]), ink)]
            if split is not None and split > edges[k]:
                cut = min(float(split), float(edges[k + 1]))
                parts = [(float(edges[k]), cut, pale), (cut, float(edges[k + 1]), ink)]
            for a, b, col in parts:
                x0, x1 = chart.fx(a), chart.fx(b)
                w = self.span(x1 - x0)[0] - gap
                if w > 0.3:
                    self.rect(sc, self.P((x0 + x1) / 2, (y0 + y1) / 2), w, h, col, op,
                              radius=min(1.5, w / 2, h / 2), lift=0.4 + 0.15 * front)

    def vline(self, sc, chart, x, ink, op, width=2.4, top=None, bottom=None):
        """A vertical line at ``x`` over the height of ``chart``."""
        fx = chart.fx(x)
        self.polyline(sc, [self.P(fx, chart.y0 if top is None else top, lift=0.9),
                           self.P(fx, chart.y1 if bottom is None else bottom, lift=0.9)],
                      width, ink, op)

    def shade(self, sc, chart, xa, xb, ink, op, strength=0.22):
        """The range ``xa..xb`` of ``chart`` tinted over its whole height."""
        x0, x1 = chart.fx(xa), chart.fx(xb)
        w, h = self.span(x1 - x0, chart.y1 - chart.y0)
        self.rect(sc, self.P((x0 + x1) / 2, (chart.y0 + chart.y1) / 2), w, h, ink,
                  op * strength, radius=0.0, lift=0.05)

    def curve(self, sc, chart, xs, ys, ink, op, go=1.0, width=2.6, dot=False, **kw):
        return progress(self, sc, chart, xs, ys, ink, op, go, width, dot=dot, **kw)

    def marks(self, sc, chart, xs, ys, ink, op, r=3.2):
        """A dot at each point."""
        for x, y in zip(xs, ys):
            self.disk(sc, self.P(chart.fx(float(x)), chart.fy(float(y))), r, ink, op, lift=1.3,
                      outline=0.0)

    def key(self, sc, fx, fy, entries, op, step=0.040, size=0.029, across=0.0):
        """A legend: a chip and a name per entry, down from ``(fx, fy)`` (or along the
        row, ``across`` apart)."""
        for n, (name, ink) in enumerate(entries):
            x, y = fx + n * across, fy + (0 if across else n * step)
            self.box(sc, x, y, 13, 13, ink, op, radius=3)
            self.text(sc, x + 0.014, y, name, op, anchor="lm", ink=dark(ink, 0.3), size=size)

    def said(self, sc, fx, fy, rows, u, op, start=0.1, every=0.1, step=0.058, size=0.030):
        """Lines of text that appear one after another as ``u`` grows. A row is a
        string, ``(string, ink)`` or ``(string, None, True)`` for a heading."""
        for n, row in enumerate(rows):
            row = row if isinstance(row, tuple) else (row, None)
            on = op * smoothstep(start + n * every, start + n * every + 0.05, u)
            self.text(sc, fx, fy + n * step, row[0], on, anchor="lm", size=size,
                      ink=None if row[1] is None else dark(row[1], 0.3),
                      strong=len(row) > 2 and row[2])

    def cloud(self, sc, box, xr, yr, groups, op, kept=None, tint=GREEN, shape=(132, 192),
              fat=2):
        """Points on a plane as a texture in the layout box. ``groups``: ``(points,
        colour, how many)`` each; ``kept``: a function of an (..., 2) array of
        positions giving the region to tint. Returns ``at(x, y)``: plane -> layout."""
        rows, cols = shape
        back = np.empty((rows, cols, 3), np.float32)
        back[:] = PAPER
        if kept is not None:
            gx = xr[0] + (np.arange(cols) + 0.5) / cols * (xr[1] - xr[0])
            gy = yr[1] - (np.arange(rows) + 0.5) / rows * (yr[1] - yr[0])
            inside = kept(np.stack(np.meshgrid(gx, gy), -1))
            back[inside] = np.asarray(soft(tint, 0.72), np.float32)
        xy = np.concatenate([np.asarray(p, float)[:n] for p, _c, n in groups])
        col = np.concatenate([np.tile(np.asarray(c, np.float32), (len(p[:n]), 1))
                              for p, c, n in groups])
        self.picture(sc, raster(xy[:, 0], xy[:, 1], xr, yr, shape, col, back=back, fat=fat),
                     box, op)
        (x0, x1), (y0, y1) = box

        def at(x, y):
            return (x0 + (x1 - x0) * (x - xr[0]) / (xr[1] - xr[0]),
                    y1 - (y1 - y0) * (y - yr[0]) / (yr[1] - yr[0]))
        return at

    def segment(self, sc, at, xr, yr, a, b, ink, op, width=2.8, head=False):
        """The part of the line through plane points ``a`` and ``b`` that lies inside
        the plane's range (with ``head``: the arrow from ``a`` to ``b`` itself)."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = b - a
        t0, t1 = (0.0, 1.0) if head else (-np.inf, np.inf)
        for k, (lo, hi) in enumerate((xr, yr)):
            if abs(d[k]) < 1e-12:
                if not lo <= a[k] <= hi:
                    return
                continue
            near, far = sorted(((lo - a[k]) / d[k], (hi - a[k]) / d[k]))
            t0, t1 = max(t0, near), min(t1, far)
        if t0 >= t1:
            return
        p, q = self.P(*at(*(a + t0 * d)), lift=0.9), self.P(*at(*(a + t1 * d)), lift=0.9)
        self.polyline(sc, [p, q], width, ink, op)
        if head:
            self._head(sc, p, q, ink, op, width)


# ====================================================================== 137: moving a cut
SIG, BKG, CUT = RED, BLUE, AMBER
Z_INK = (MAGENTA, TEAL, VIOLET)               # s/sqrt(b), s/sqrt(s+b), the formula
WAY_INK = (GREEN, AMBER, TEAL)                # fixed, tuned on the same events, independent
FORMULA = "√(2((s+b)·ln(1+s/b) − s))"


class MovingACut(Select):
    """A cut on one variable, number by number; the cut sliding; where each figure of
    merit is best; where s/sqrt(b) overstates; and the bias of tuning the cut on the
    events that are then measured."""

    TITLE = "Moving a cut"
    PHASES = (("one cut", 11.0), ("sliding", 11.0), ("which is best", 12.0),
              ("small b", 11.0), ("tuned on data", 13.0))
    BOX = ((0.09, 0.50), (0.40, 0.79))
    SLIDE = (0.0, 3.3)

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        self.model = m["model"]
        self.edges = np.asarray(s.panels[0].data["edges"], float)
        self.hs, self.hb = np.asarray(s.panels[0].data["ys"], float)
        p = s.panels[1].data
        self.cuts = np.asarray(p["x"], float)
        self.s, self.b = np.asarray(p["s"], float), np.asarray(p["b"], float)
        self.ns, self.nb = np.asarray(p["count_s"], int), np.asarray(p["count_b"], int)
        self.purity = np.asarray(p["ys"], float)[2]
        p = s.panels[2].data
        self.z = np.asarray(p["ys"], float)
        self.pois_x, self.pois_z = np.asarray(p["poisson_at"]), np.asarray(p["z_poisson"])
        self.table, self.table_sb = np.asarray(p["table"]), np.asarray(p["table_s_b"])
        p = s.panels[3].data
        self.s_hat = np.asarray(p["s_hat"], float)
        self.mean, self.se = np.asarray(p["values"]), np.asarray(p["standard_error"])
        self.null, self.null_exact = np.asarray(p["null_above_2"]), float(p["null_fixed_exact"])
        self.best = m["best"]
        self.first = float(m["first_cut"])
        self.toys = self.s_hat.shape[1]

    def at(self, cut):
        """Index of ``cut`` on the grid of cuts."""
        return int(np.clip(round((cut - self.cuts[0]) / 0.01), 0, len(self.cuts) - 1))

    def counts(self, cut):
        i = self.at(cut)
        return int(self.ns[i]), int(self.nb[i])

    def subtitle_for(self, k):
        b, mo = self.best, self.model
        s1, b1 = self.counts(self.first)
        lo, hi = (self.counts(c) for c in self.SLIDE)
        return [
            "a cut keeps the events with x above a threshold. Here s = %d signal and b = %d "
            "background events pass: efficiency %s, purity %s, and a significance s/√b "
            "of %.1f" % (s1, b1, pct(s1 / mo["S"]), pct(s1 / (s1 + b1)), s1 / np.sqrt(b1)),
            "slide the cut and every number moves. A tighter cut rejects more background, "
            "so the sample gets purer (%s to %s here), but signal is thrown away too: %d "
            "of %d events are left" % (pct(lo[0] / sum(lo)), pct(hi[0] / sum(hi)), hi[0],
                                       mo["S"]),
            "the best cut depends on the question: x > %.2f for s/√(s+b), %.2f for the "
            "likelihood-ratio formula, %.2f for s/√b. Purity alone asks for the tightest "
            "cut, which keeps nothing" % (b["z_total"]["cut"], b["z_asimov"]["cut"],
                                          b["z_simple"]["cut"]),
            "s/√b overstates when b is small. At its own best cut b is %.2f events: it "
            "claims %.1fσ, where the formula gives %.1f and the exact Poisson probability "
            "%.1f" % (b["z_simple"]["b"], b["z_simple"]["value"], b["z_simple"]["z_asimov"],
                      b["z_simple"]["z_poisson"]),
            "choose the cut on the very events you then measure and the result is biased: "
            "over %s pseudo-experiments the signal estimate averages %.0f where the truth "
            "is %d. A cut fixed beforehand: %.1f" % (commas(self.toys), self.mean[1],
                                                         mo["S"], self.mean[0]),
        ][k]

    # ------------------------------------------------------------------ drawing
    def spectrum(self, sc, cut, op, fade=1.0, box=None):
        """The pseudo-dataset's two histograms with the cut; bars left of it paled."""
        chart = self.chart(self.BOX if box is None else box, [-3.5, 5.5], [0.5, 220.0],
                           ylog=True)
        split = cut if fade > 0 else None
        self.bars(sc, chart, self.edges, self.hb, BKG, op, split=split,
                  pale=soft(BKG, 0.55 * fade))
        self.bars(sc, chart, self.edges, self.hs, SIG, op, split=split,
                  pale=soft(SIG, 0.55 * fade), gap=4.0, front=1)
        self.axes(sc, chart, op, "x (events per bin of 0.25, log scale)", "",
                  xticks=[(v, "%d" % v) for v in (-2, 0, 2, 4)],
                  yticks=[(v, "%d" % v) for v in (1, 10, 100)], grid=False)
        if fade > 0:
            self.vline(sc, chart, cut, dark(CUT, 0.1), op * fade, 3.0, top=chart.y0 - 0.012)
            self.text(sc, chart.fx(cut), chart.y0 - 0.034, "cut: x > %.2f" % cut, op * fade,
                      ink=dark(CUT, 0.35), size=0.030)
        self.key(sc, 0.10, 0.905, [("signal", SIG), ("background", BKG)], op, across=0.11)
        return chart

    def phase_1(self, sc, u, op):
        mo = self.model
        self.text(sc, 0.06, 0.315, "the model: %d signal events (μ = %g, σ = %g) on %s "
                  "background events (μ = %g, σ = %g)" % (
                      mo["S"], mo["mu_s"], mo["sigma"], commas(mo["B"]), mo["mu_b"],
                      mo["sigma"]), op, anchor="lm", strong=True)
        self.spectrum(sc, self.first, op, fade=smoothstep(0.16, 0.26, u))
        s, b = self.counts(self.first)
        self.said(sc, 0.56, 0.385, [
            ("keep the events with x > %g:" % self.first, None, True),
            "s = %d signal,  b = %d background" % (s, b),
            ("signal efficiency: %d / %d = %s" % (s, mo["S"], pct(s / mo["S"])), SIG),
            ("background rejection: 1 − %d / %s = %s" % (b, commas(mo["B"]),
                                                         pct(1 - b / mo["B"])), BKG),
            ("purity: s / (s + b) = %d / %d = %s" % (s, s + b, pct(s / (s + b))), GREEN),
            ("significance: Z = s / √b = %d / √%d = %.1f" % (s, b, s / np.sqrt(b)), MAGENTA),
            "√b: how much the background count",
            "fluctuates by itself. Z counts the signal in such σ",
        ], u, op, start=0.24, every=0.085, step=0.056)
        self.tag(sc, DATA, op)

    def phase_2(self, sc, u, op):
        mo = self.model
        cut = self.SLIDE[0] + (self.SLIDE[1] - self.SLIDE[0]) * ease_in_out(
            float(np.clip((u - 0.06) / 0.80, 0, 1)))
        cut = round(cut, 2)
        self.text(sc, 0.06, 0.315, "the same events, the cut sliding to the right", op,
                  anchor="lm", strong=True)
        self.spectrum(sc, cut, op)
        s, b = self.counts(cut)
        self.text(sc, 0.56, 0.385, "kept:  s = %d,  b = %d" % (s, b), op, anchor="lm",
                  strong=True, size=BIG)
        rows = (("signal efficiency", s / mo["S"], SIG, 0),
                ("background rejection", 1 - b / mo["B"], BKG, 1),
                ("purity", s / max(s + b, 1), GREEN, 0))
        for n, (name, v, ink, digits) in enumerate(rows):
            self.hbar(sc, 0.755, 0.47 + 0.075 * n, 132 * v, ink, op, name, pct(v, digits),
                      track=132)
        self.text(sc, 0.56, 0.715, "s / √b = %d / √%d = %.1f" % (s, b, s / np.sqrt(max(b, 1))),
                  op, anchor="lm", size=BIG, ink=dark(MAGENTA, 0.3))
        self.lines(sc, 0.56, 0.785, ["counted in this one dataset: the numbers",
                                     "step as single events drop out"], op, step=0.042)
        self.tag(sc, DATA, op)

    def phase_3(self, sc, u, op):
        b = self.best
        go = smoothstep(0.05, 0.72, u)
        lo, hi = 0.0, 4.5
        cut = round(lo + (hi - lo) * go, 2)
        self.text(sc, 0.06, 0.315, "what the model expects at every cut: which cut is best?",
                  op, anchor="lm", strong=True)
        self.spectrum(sc, cut, op)
        top = 7.5
        chart = self.chart(((0.60, 0.91), (0.375, 0.635)), [lo, hi], [0, top], pad=0.0)
        self.axes(sc, chart, op, "cut", "Z (σ)", xticks=[(v, "%d" % v) for v in range(5)],
                  yticks=[(v, "%d" % v) for v in (0, 2, 4, 6)])
        keep = (self.cuts >= lo) & (self.cuts <= hi)
        g = self.cuts[keep]
        self.curve(sc, chart, g, self.purity[keep] * top, GREEN, op, go, 2.0)
        self.text(sc, 0.918, chart.fy(top * 0.985), "100%", op, anchor="lm",
                  ink=dark(GREEN, 0.3))
        self.text(sc, 0.918, chart.fy(top * 0.5), "50%", op, anchor="lm", ink=dark(GREEN, 0.3))
        for n, name in enumerate(("z_simple", "z_total", "z_asimov")):
            self.curve(sc, chart, g, self.z[n][keep], Z_INK[n], op, go, 2.8)
            if cut >= b[name]["cut"]:
                self.disk(sc, self.P(chart.fx(b[name]["cut"]), chart.fy(b[name]["value"])), 5.5,
                          Z_INK[n], op, glow=0.4, lift=1.5, outline=0.0)
        self.vline(sc, chart, cut, dark(CUT, 0.1), op * 0.45, 1.6)
        rows = (("s/√(s+b): best at x > %.2f" % b["z_total"]["cut"], TEAL, "z_total"),
                ("%s: best at %.2f" % (FORMULA, b["z_asimov"]["cut"]), VIOLET, "z_asimov"),
                ("s/√b: best at %.2f, where b = %.2f" % (b["z_simple"]["cut"],
                                                        b["z_simple"]["b"]), MAGENTA,
                 "z_simple"),
                ("purity (right scale): only rises", GREEN, None))
        for n, (text, ink, name) in enumerate(rows):
            on = op * (smoothstep(0.74, 0.8, u) if name is None else
                       float(cut >= b[name]["cut"]))
            self.key(sc, 0.59, 0.752 + 0.042 * n, [(text, ink)], on)
        self.tag(sc, EXACT, op)

    def phase_4(self, sc, u, op):
        self.text(sc, 0.06, 0.315, "s/√b treats the background count as a Gaussian of width "
                  "√b. A small count is not one", op, anchor="lm", strong=True)
        xs = (0.085, 0.155, 0.25, 0.37, 0.49)
        heads = ("s", "b", "s/√b", "formula", "exact")
        inks = (None, None, MAGENTA, VIOLET, GREEN)
        for x, h, ink in zip(xs, heads, inks):
            self.text(sc, x, 0.385, h, op, ink=None if ink is None else dark(ink, 0.3),
                      size=0.031)
        for r, ((s, b), row) in enumerate(zip(self.table_sb, self.table)):
            on = op * smoothstep(0.08 + 0.11 * r, 0.14 + 0.11 * r, u)
            y = 0.45 + 0.066 * r
            cells = ("%d" % s, "%d" % b, "%.1f" % row[0], "%.1f" % row[1], "%.1f" % row[2])
            for x, c, ink in zip(xs, cells, inks):
                self.box(sc, x, y, 76 if ink is not None else 44, 30,
                         PAPER if ink is None else soft(ink, 0.6), on, c,
                         border=DIM if ink is None else ink, size=0.031, radius=4)
        s, b = self.table_sb[-1]
        self.said(sc, 0.06, 0.735, [
            "formula: %s" % FORMULA,
            "exact: the p-value, the chance that background alone gives ≥ s + b",
            ("s = %d on b = %d: p = P(n ≥ %d) = %s, which is Z = %.1f, not %.0f" % (
                s, b, s + b, pow10(self.table[-1][3], 1), self.table[-1][2],
                self.table[-1][0]), GREEN),
        ], u, op, start=0.5, every=0.09, step=0.046)
        lo, hi = 2.0, 3.5
        chart = self.chart(((0.65, 0.94), (0.40, 0.66)), [lo, hi], [0, 7.5], pad=0.0)
        self.axes(sc, chart, op, "cut  (b: 23 → 0.2 events)", "Z in the model",
                  xticks=[(v, "%.1f" % v) for v in (2.0, 2.5, 3.0, 3.5)],
                  yticks=[(v, "%d" % v) for v in (2, 4, 6)])
        keep = (self.cuts >= lo) & (self.cuts <= hi)
        go = smoothstep(0.3, 0.7, u)
        self.curve(sc, chart, self.cuts[keep], self.z[0][keep], MAGENTA, op, go, 2.8)
        self.curve(sc, chart, self.cuts[keep], self.z[2][keep], VIOLET, op, go, 2.8)
        shown = (self.pois_x >= lo) & (self.pois_x <= lo + (hi - lo) * go + 1e-9) & (go > 0)
        self.marks(sc, chart, self.pois_x[shown], self.pois_z[shown], dark(GREEN, 0.15), op)
        self.key(sc, 0.66, 0.775, [("s/√b", MAGENTA), ("the formula", VIOLET),
                                   ("exact Poisson, median outcome", GREEN)], op,
                 step=0.038)
        self.tag(sc, EXACT, op)

    def phase_5(self, sc, u, op):
        truth = self.model["S"]
        self.text(sc, 0.06, 0.315, "three ways to choose the cut; then estimate the signal "
                  "from the events it keeps", op, anchor="lm", strong=True)
        names = (("cut fixed beforehand", "at x > %.1f" % self.meta["tuning"]["fixed_cut"]),
                 ("cut with the best Z", "in these same events"),
                 ("cut with the best Z", "in an independent sample"))
        n = int(round(self.toys * smoothstep(0.04, 0.55, u)))
        edges = np.arange(-20.0, 160.1, 10.0)
        full = max(np.histogram(self.s_hat[k], edges)[0].max() for k in range(3))
        x0, x1 = 0.31, 0.63
        for k in range(3):
            y0 = 0.37 + 0.155 * k
            chart = self.chart(((x0, x1), (y0, y0 + 0.13)), [-20, 160], [0, full * 1.05],
                               pad=0.0)
            self.bars(sc, chart, edges, np.histogram(self.s_hat[k, :n], edges)[0], WAY_INK[k],
                      op)
            self.polyline(sc, [self.P(x0, chart.y1), self.P(x1, chart.y1)], 1.0, AXIS,
                          0.7 * op)
            self.vline(sc, chart, truth, INK, 0.45 * op, 1.6)
            self.text(sc, 0.06, y0 + 0.022, names[k][0], op, anchor="lm",
                      ink=dark(WAY_INK[k], 0.3))
            self.text(sc, 0.06, y0 + 0.06, names[k][1], op, anchor="lm",
                      ink=dark(WAY_INK[k], 0.3))
            late = op * smoothstep(0.56, 0.64, u)
            self.text(sc, 0.06, y0 + 0.098, "mean %.1f ± %.1f" % (self.mean[k], self.se[k]),
                      late, anchor="lm", strong=True)
            self.disk(sc, self.P(chart.fx(self.mean[k]), chart.y1), 5.0, INK, late, lift=1.6,
                      outline=0.0)
        for v in (0, 50, 100, 150):
            self.text(sc, chart.fx(v), chart.y1 + 0.03, "%d" % v, op)
        self.text(sc, (x0 + x1) / 2, chart.y1 + 0.072, "estimated signal (true: %d)" % truth,
                  op)
        late = op * smoothstep(0.66, 0.74, u)
        self.text(sc, 0.68, 0.385, "and with no signal at all:", late, anchor="lm",
                  strong=True)
        self.text(sc, 0.68, 0.43, "how often is Z above 2?", late, anchor="lm")
        for k, name in enumerate(("fixed", "same", "indep.")):
            self.hbar(sc, 0.76, 0.50 + 0.07 * k, 520 * self.null[k], WAY_INK[k], late, name,
                      pct(self.null[k], 1), u=smoothstep(0.7, 0.85, u))
        self.lines(sc, 0.68, 0.72, ["exact for the fixed cut: %s" % pct(self.null_exact, 1),
                                    "the tuned cut finds the upward",
                                    "fluctuations and keeps them"], late, step=0.042)
        self.tag(sc, toys(self.toys), op)


# ====================================================================== 138: two variables
MV_X, MV_Y = (-3.5, 4.5), (-2.7, 2.7)
BAN_X, BAN_Y = (-4.0, 4.0), (-2.0, 3.4)
RECT_INK, LINE_INK, CURVE_INK = AMBER, VIOLET, GREEN


class CutsVsMultivariate(Select):
    """Two tilted clouds; the best pair of thresholds; Fisher's line computed from
    means and covariance; background kept against signal efficiency for both; and a
    crescent of signal that no line can separate."""

    TITLE = "Cuts against a combined variable"
    PHASES = (("clouds", 10.0), ("two cuts", 11.0), ("one line", 14.0), ("compared", 10.0),
              ("curved", 12.0))
    BOX = ((0.07, 0.47), (0.36, 0.84))
    BAN_BOX = ((0.07, 0.47), (0.36, 0.84))

    def setup(self):
        self.d = S.mv_data()
        s = self.study
        m = self.meta = s.metadata
        p = s.panels[0].data
        n = len(p["xy"]) // 2
        self.sig, self.bkg = np.asarray(p["xy"][:n]), np.asarray(p["xy"][n:])
        self.w, self.w_mat = np.asarray(p["w"], float), np.asarray(p["w_mat"], float)
        self.ms, self.mb = np.asarray(p["mean_s"], float), np.asarray(p["mean_b"], float)
        self.rect_cut = np.asarray(p["rect_cut"], float)
        self.thr = float(p["line_threshold"])
        p = s.panels[1].data
        self.eff = np.asarray(p["x"], float)
        self.roc = np.asarray(p["ys"], float)         # cuts, line, cuts exact, line exact
        self.one = np.asarray(p["one_variable"], float)
        p = s.panels[2].data
        self.kept, self.spread = np.asarray(p["values"]), np.asarray(p["spread"])
        self.exact = np.asarray(p["exact"], float)
        p = s.panels[3].data
        self.ban_roc = np.asarray(p["ys"], float)
        self.ban_s, self.ban_b = np.asarray(p["xy"][:n]), np.asarray(p["xy"][n:])
        self.ban = m["crescent_at"]
        self.at_eff = float(m["at_efficiency"])
        self.seeds = len(m["seeds"])

    def subtitle_for(self, k):
        w, kept, at = self.w, self.kept, pct(self.at_eff)
        return [
            "two variables are measured for every event, and signal and background are "
            "two tilted clouds. In x₁ alone they overlap; in x₂ alone they are identical. "
            "Yet together they separate",
            "rectangular cuts, one threshold on each variable: the best pair that keeps "
            "%s of the signal still keeps %s of the background on held-out events, no "
            "better than a cut on x₁ alone (%s)" % (at, pct(kept[1]), pct(kept[0])),
            "one diagonal line instead: Fisher's discriminant, computed from the events' "
            "means and covariance. Cutting on t = %.2f x₁ − %.2f x₂ keeps %s of the "
            "background at the same %s" % (w[0], -w[1], pct(kept[2]), at),
            "at every signal efficiency the line keeps less background than the cuts: "
            "%s against %s at %s, the mean of %d seeds on held-out events; the model's "
            "exact values are %s and %s" % (pct(kept[2], 1), pct(kept[1], 1), at, self.seeds,
                                            pct(self.exact[2], 1), pct(self.exact[1], 1)),
            "when the signal curls round the background no straight line is enough: at "
            "%s signal the best line keeps %s of the background, a curved boundary %s. "
            "Finding such boundaries from data is what classifiers do" % (
                at, pct(self.ban["line"]), pct(self.ban["curve"], 1)),
        ][k]

    # ------------------------------------------------------------------ drawing
    def plane(self, sc, op, kept=None, shown=None, tint=GREEN):
        n = len(self.sig) if shown is None else shown
        at = self.cloud(sc, self.BOX, MV_X, MV_Y, [(self.bkg, BKG, n), (self.sig, SIG, n)], op,
                        kept=kept, tint=tint)
        (x0, x1), (y0, y1) = self.BOX
        for v in (-2, 0, 2, 4):
            self.text(sc, at(v, 0)[0], y1 + 0.03, "%d" % v, op)
        for v in (-2, 0, 2):
            self.text(sc, x0 - 0.014, at(0, v)[1], "%d" % v, op, anchor="rm")
        self.text(sc, x1 + 0.012, y1 + 0.03, "x₁", op, anchor="lm", strong=True)
        self.text(sc, x0 - 0.014, y0 + 0.012, "x₂", op, anchor="rm", strong=True)
        return at

    def tally(self, sc, ks, kb, n, op, fy=0.915):
        """How many of the events drawn the selection keeps (and which colour is which)."""
        self.key(sc, 0.09, fy, [("%d signal" % ks, SIG),
                                ("%d background kept, of %d each" % (kb, n), BKG)], op,
                 across=0.115, size=0.030)

    def rect_kept(self, cut):
        return lambda x: S.rect_apply(x.reshape(-1, 2), cut).reshape(x.shape[:-1])

    def rect_lines(self, sc, at, cut, op, width=2.8):
        a, b = float(cut[1]), float(cut[2])
        ink = dark(RECT_INK, 0.1)
        if MV_X[0] < a < MV_X[1]:
            self.segment(sc, at, MV_X, MV_Y, (a, 0), (a, 1), ink, op, width)
        if MV_Y[0] < b < MV_Y[1]:
            self.segment(sc, at, MV_X, MV_Y, (0, b), (1, b), ink, op, width)

    def line_kept(self, w, thr):
        return lambda x: x @ w > thr

    def fisher_line(self, sc, at, w, thr, op, xr=MV_X, yr=MV_Y, width=2.8):
        p = w * thr / (w @ w)
        self.segment(sc, at, xr, yr, p, p + np.array([-w[1], w[0]]), dark(LINE_INK, 0.1), op,
                     width)

    def matrix(self, sc, fx, fy, values, op, ink):
        for r in range(2):
            for c in range(2):
                self.box(sc, fx + 0.062 * c, fy + 0.058 * r, 54, 27, soft(ink, 0.7), op,
                         minus("%.2f" % values[r, c]), border=ink, size=0.029, radius=4)

    def small(self, sc, box, hs, hb, op, label):
        d = self.d
        chart = self.chart(box, [-4, 6], [0, max(hs.max(), hb.max()) * 1.05], pad=0.0)
        self.bars(sc, chart, d["proj_edges"], hb, BKG, 0.8 * op)
        self.bars(sc, chart, d["proj_edges"], hs, SIG, 0.62 * op, gap=3.0, front=1)
        self.axes(sc, chart, op, label, "", xticks=[(v, "%d" % v) for v in (-2, 0, 2, 4)],
                  yticks=[], grid=False)
        return chart

    def phase_1(self, sc, u, op):
        mo = self.meta["model"]
        self.text(sc, 0.06, 0.315, "the model: two Gaussian clouds, σ = 1 in each variable, "
                  "correlation ρ = %g; the signal shifted by %g in x₁ only" % (
                      mo["rho"], mo["mu_s"][0] - mo["mu_b"][0]), op, anchor="lm", strong=True)
        n = int(round(len(self.sig) * smoothstep(0.03, 0.4, u)))
        self.plane(sc, op, shown=max(n, 1))
        self.key(sc, 0.08, 0.915, [("signal", SIG), ("background", BKG)], op, across=0.11)
        d = self.d
        for k, (hs, hb, name, note) in enumerate((
                (d["x1_s"], d["x1_b"], "x₁", "x₁ alone: the two overlap"),
                (d["x2_s"], d["x2_b"], "x₂", "x₂ alone: no difference at all"))):
            on = op * smoothstep(0.42 + 0.2 * k, 0.52 + 0.2 * k, u)
            y0 = 0.41 + 0.27 * k
            self.text(sc, 0.58, y0 - 0.04, note, on, anchor="lm", strong=True)
            self.small(sc, ((0.58, 0.93), (y0, y0 + 0.125)), hs, hb, on, name)
        self.tag(sc, DATA, op)

    def phase_2(self, sc, u, op):
        go = smoothstep(0.05, 0.45, u)
        o, a, b = self.rect_cut
        far_a = MV_X[0] if S.ORIENT[int(o)][0] == ">" else MV_X[1]
        far_b = MV_Y[1] if S.ORIENT[int(o)][1] == "<" else MV_Y[0]
        cut = np.array([o, far_a + (a - far_a) * go, far_b + (min(b, MV_Y[1] - 0.25) - far_b)
                        * go if go < 1 else b])
        self.text(sc, 0.06, 0.315, "rectangular cuts: a threshold on x₁ and a threshold on "
                  "x₂; keep the corner between them", op, anchor="lm", strong=True)
        at = self.plane(sc, op, kept=self.rect_kept(cut), tint=RECT_INK)
        self.rect_lines(sc, at, cut, op)
        ks = int(S.rect_apply(self.sig, cut).sum())
        kb = int(S.rect_apply(self.bkg, cut).sum())
        self.tally(sc, ks, kb, len(self.sig), op)
        signs = S.ORIENT[int(o)]
        k, sp = self.kept, self.spread
        self.said(sc, 0.53, 0.40, [
            ("the best pair that keeps %s of the signal," % pct(self.at_eff), None, True),
            ("searched over 200 × 200 thresholds:", None, True),
            ("x₁ %s %s  and  x₂ %s %s" % (signs[0], minus("%.2f" % a), signs[1],
                                         minus("%.2f" % b)), RECT_INK),
            "the threshold on x₂ removes almost nothing",
            ("background kept, held-out events:", None, True),
            ("two cuts: %s  (%d seeds: %s to %s)" % (pct(k[1], 1), self.seeds,
                                                    pct(sp[1][0], 1), pct(sp[1][1], 1)),
             RECT_INK),
            ("a cut on x₁ alone: %s" % pct(k[0], 1), BLUE),
            "straight cuts cannot use the tilt",
        ], u, op, start=0.34, every=0.07, step=0.056)
        self.tag(sc, DATA, op)

    def phase_3(self, sc, u, op):
        w, thr = self.w, self.thr
        self.text(sc, 0.06, 0.315, "Fisher's line: weigh the difference of the means by the "
                  "inverse of the clouds' covariance W", op, anchor="lm", strong=True)
        go = smoothstep(0.52, 0.8, u)
        proj = np.concatenate([self.sig @ w, self.bkg @ w])
        now = proj.min() + (thr - proj.min()) * go
        at = self.plane(sc, op, kept=self.line_kept(w, now) if go > 0 else None,
                        tint=LINE_INK)
        if go > 0:
            self.fisher_line(sc, at, w, now, op)
        mid = (self.ms + self.mb) / 2
        tip = mid + w / np.linalg.norm(w) * 1.6
        grow = smoothstep(0.4, 0.52, u)
        if grow > 0:
            self.segment(sc, at, MV_X, MV_Y, mid, mid + (tip - mid) * grow, INK, op, 3.0,
                         head=True)
            self.text(sc, at(*tip)[0] + 0.02, at(*tip)[1] + 0.03, "w",
                      op * grow * (1 - smoothstep(0.5, 0.53, u)), strong=True, size=BIG)
        ks, kb = int((self.sig @ w > now).sum()), int((self.bkg @ w > now).sum())
        self.tally(sc, ks, kb, len(self.sig), op)
        x = 0.53
        delta = self.ms - self.mb
        inv = np.linalg.inv(self.w_mat)
        on = [op * smoothstep(0.06 + 0.085 * n, 0.11 + 0.085 * n, u) for n in range(6)]
        self.text(sc, x, 0.385, "signal mean − background mean = (%s, %s)" % (
            minus("%.2f" % delta[0]), minus("%.2f" % delta[1])), on[0], anchor="lm")
        self.text(sc, x, 0.468, "W =", on[1], anchor="lm")
        self.matrix(sc, x + 0.075, 0.44, self.w_mat, on[1], BLUE)
        self.text(sc, x + 0.19, 0.468, "W⁻¹ =", on[2], anchor="lm")
        self.matrix(sc, x + 0.285, 0.44, inv, on[2], LINE_INK)
        self.text(sc, x, 0.56, "w = W⁻¹ × difference = (%s, %s)" % (
            minus("%.2f" % w[0]), minus("%.2f" % w[1])), on[3], anchor="lm", strong=True)
        self.text(sc, x, 0.61, "t = %.2f x₁ − %.2f x₂ : one number per event" % (
            w[0], -w[1]), on[4], anchor="lm", ink=dark(LINE_INK, 0.3))
        d = self.d
        chart = self.small(sc, ((0.58, 0.93), (0.645, 0.765)), d["proj_s"], d["proj_b"], on[5],
                           "t, in units of its spread")
        scale = float(d["proj_scale"])
        if go > 0:
            self.vline(sc, chart, now * scale, dark(LINE_INK, 0.1), on[5], 2.6)
        late = op * smoothstep(0.84, 0.92, u)
        self.text(sc, x, 0.872, "keeps %s of signal, %s of background" % (
            pct(self.at_eff), pct(self.kept[2], 1)), late, anchor="lm", strong=True)
        self.tag(sc, DATA, op)

    def roc_chart(self, sc, box, op, top):
        chart = self.chart(box, [0, 1], [0, top], pad=0.0)
        self.axes(sc, chart, op, "signal efficiency", "background kept",
                  xticks=[(v, pct(v)) for v in (0, 0.5, 1.0)],
                  yticks=[(v, pct(v)) for v in np.arange(0.2, top + 1e-9, 0.2)])
        return chart

    def phase_4(self, sc, u, op):
        self.text(sc, 0.06, 0.315, "both selections at every signal efficiency: lower is "
                  "better", op, anchor="lm", strong=True)
        at = self.plane(sc, op, kept=self.line_kept(self.w, self.thr), tint=LINE_INK)
        self.fisher_line(sc, at, self.w, self.thr, op)
        self.rect_lines(sc, at, self.rect_cut, op)
        chart = self.roc_chart(sc, ((0.60, 0.93), (0.385, 0.70)), op, 0.8)
        go = smoothstep(0.08, 0.6, u)
        self.curve(sc, chart, self.eff, self.roc[2], soft(RECT_INK, 0.2), op, go, 2.2)
        self.curve(sc, chart, self.eff, self.roc[3], soft(LINE_INK, 0.2), op, go, 2.2)
        shown = self.eff <= self.eff[0] + (self.eff[-1] - self.eff[0]) * go
        self.marks(sc, chart, self.eff[shown], self.roc[0][shown], dark(RECT_INK, 0.15), op * go)
        self.marks(sc, chart, self.eff[shown], self.roc[1][shown], dark(LINE_INK, 0.15), op * go)
        late = op * smoothstep(0.62, 0.72, u)
        self.vline(sc, chart, self.at_eff, INK, 0.45 * late, 1.6)
        self.key(sc, 0.08, 0.915, [("two cuts", RECT_INK), ("Fisher's line", LINE_INK)], op,
                 across=0.12)
        k, e = self.kept, self.exact
        self.lines(sc, 0.59, 0.80, [
            ("at %s: two cuts keep %s (exact %s)" % (pct(self.at_eff), pct(k[1], 1),
                                                    pct(e[1], 1)), dark(RECT_INK, 0.3)),
            ("the line keeps %s (exact %s)" % (pct(k[2], 1), pct(e[2], 1)),
             dark(LINE_INK, 0.3)),
        ], late, step=0.042)
        self.tag(sc, "pseudo-data (dots, %d seeds) and exact (curves)" % self.seeds, op)

    def phase_5(self, sc, u, op):
        d = self.d
        w, thr, cthr = d["ban_w"], float(d["ban_line_thr"]), float(d["ban_curve_thr"])
        self.text(sc, 0.06, 0.315, "another model: the signal a crescent, the background a "
                  "blob inside it", op, anchor="lm", strong=True)
        curved = smoothstep(0.5, 0.56, u)
        n = len(self.ban_s)
        if curved < 0.5:
            go = smoothstep(0.1, 0.36, u)
            proj = np.concatenate([self.ban_s @ w, self.ban_b @ w])
            now = proj.min() + (thr - proj.min()) * go
            kept, tint = (self.line_kept(w, now) if go > 0 else None), LINE_INK
            ks, kb = int((self.ban_s @ w > now).sum()), int((self.ban_b @ w > now).sum())
        else:
            kept, tint = (lambda x: S.banana_log_ratio(x) > cthr), CURVE_INK
            ks = int((S.banana_log_ratio(self.ban_s) > cthr).sum())
            kb = int((S.banana_log_ratio(self.ban_b) > cthr).sum())
        at = self.cloud(sc, self.BAN_BOX, BAN_X, BAN_Y,
                        [(self.ban_b, BKG, n), (self.ban_s, SIG, n)], op, kept=kept, tint=tint)
        if curved < 0.5 and kept is not None:
            self.fisher_line(sc, at, w, now, op, BAN_X, BAN_Y)
        self.tally(sc, ks, kb, n, op, 0.885)
        self.text(sc, 0.27, 0.925, "the best straight line" if curved < 0.5 else
                  "a curved boundary", op, ink=dark(tint, 0.3))
        chart = self.roc_chart(sc, ((0.60, 0.93), (0.385, 0.61)), op, 1.0)
        go = smoothstep(0.1, 0.4, u)
        self.curve(sc, chart, self.eff, self.ban_roc[0], LINE_INK, op, go, 2.8)
        self.curve(sc, chart, self.eff, self.ban_roc[1], CURVE_INK, op,
                   smoothstep(0.56, 0.8, u), 2.8)
        self.vline(sc, chart, self.at_eff, INK, 0.45 * op, 1.6)
        self.said(sc, 0.59, 0.715, [
            ("a line keeps %s of the background" % pct(self.ban["line"], 1), LINE_INK),
            ("the curved boundary keeps %s" % pct(self.ban["curve"], 1), CURVE_INK),
            "(both at %s signal efficiency)" % pct(self.at_eff),
            "that boundary is the model's own likelihood",
            "ratio: from data, it has to be learned",
        ], u, op, start=0.4, every=0.1, step=0.04)
        self.tag(sc, "pseudo-data: %d seeded draws, held-out" % self.seeds, op)


# ====================================================================== 139: sidebands
FIT_INK = (GREEN, MAGENTA, VIOLET)            # right shape, wrong shape, signal in the sides
WINDOW, SIDE = AMBER, TEAL


class SidebandBackground(Select):
    """A peak on a falling spectrum; the background fitted in the sidebands and carried
    under the peak, number by number; the yield over many pseudo-experiments; and its
    bias when the shape is wrong or the sidebands hold signal."""

    TITLE = "Background from the sidebands"
    PHASES = (("data", 10.0), ("fit", 13.0), ("repeat", 11.0), ("wrong shape", 12.0),
              ("tails", 11.0))
    BOX = ((0.09, 0.55), (0.42, 0.78))

    def setup(self):
        d = self.d = S.sb_data()
        s = self.study
        m = self.meta = s.metadata
        self.model = m["model"]
        p = s.panels[0].data
        self.m = np.asarray(p["x"], float)
        self.data, self.bkg, self.total, self.fit_exp, self.fit_line = np.asarray(p["ys"])
        self.one = np.asarray(p["one"], float)        # n, b, b_err, s, s_err, chi2, ndf
        self.data_wide, self.total_wide = np.asarray(p["data_wide"]), np.asarray(
            p["truth_wide"])
        self.fit_wide = np.asarray(p["fit_wide"], float)
        self.window = np.asarray(p["window"]) > 0
        self.hist = np.asarray(s.panels[1].data["ys"], float)
        self.hist_edges = np.asarray(s.panels[1].data["edges"], float)
        self.res = [m["results"][name] for name in m["cases"]]
        self.truth = m["truth_in_window"]
        self.diff = [np.asarray(d["s_%d" % k], float) - self.truth[k] for k in range(3)]
        self.toys = len(self.diff[0])

    def subtitle_for(self, k):
        one, res = self.one, self.res
        return [
            "a mass spectrum: a smooth, falling background with a peak on it. The window "
            "round the peak holds N = %s events. How many of them are background?"
            % commas(one[0][0]),
            "fit the background's shape in the sidebands only and carry the curve under "
            "the peak: the background there is %s ± %.0f. What is left is the signal, "
            "%.0f ± %.0f events; the true number is %.0f" % (
                commas(round(one[0][1])), one[0][2], one[0][3], one[0][4], self.truth[0]),
            "over %s pseudo-experiments the estimate is right on average (bias %s events "
            "of %.0f) and its quoted error is honest: %s of the results lie within one σ "
            "of the truth" % (commas(self.toys), minus("%+.1f" % res[0]["bias"]),
                              self.truth[0], pct(res[0]["within_1_sigma"], 1)),
            "with the wrong shape the answer is biased. A straight line fitted to the "
            "curved background lies above it under the peak, and the signal comes out "
            "%.0f events too low on average: %.1fσ" % (-res[1]["bias"], -res[1]["pull_mean"]),
            "signal in the sidebands biases it too. When the peak is wider than the "
            "window allows for, its tails lift the fit, and the signal comes out %.0f "
            "events too low: %.1fσ" % (-res[2]["bias"], -res[2]["pull_mean"]),
        ][k]

    # ------------------------------------------------------------------ drawing
    ZOOM = ((103.0, 147.0), (40.0, 215.0))    # the part of the spectrum phases 4 and 5 show

    def spectrum(self, sc, op, data=None, shown=1.0, bands=1.0, names=True, zoom=False):
        mo = self.model
        xr, yr = self.ZOOM if zoom else ((mo["lo"], mo["hi"]), (0.0, 290.0))
        chart = self.chart(self.BOX, xr, yr, pad=0.0)
        if bands > 0:
            self.shade(sc, chart, *mo["window"], WINDOW, op * bands, 0.30)
            for a, b in mo["sidebands"]:
                self.shade(sc, chart, a, b, SIDE, op * bands, 0.22)
            if names:
                y = chart.y0 - 0.03
                for (a, b), name, ink in ((mo["sidebands"][0], "sideband", SIDE),
                                          (mo["window"], "window", WINDOW),
                                          (mo["sidebands"][1], "sideband", SIDE)):
                    self.text(sc, chart.fx((a + b) / 2), y, name, op * bands,
                              ink=dark(ink, 0.35))
        self.axes(sc, chart, op, "mass m (GeV); events per GeV", "",
                  xticks=[(v, "%d" % v) for v in ((105, 120, 130, 145) if zoom else
                                                  (100, 120, 140, 160))],
                  yticks=[(v, "%d" % v) for v in ((50, 100, 150, 200) if zoom else
                                                  (100, 200))], grid=False)
        data = self.data if data is None else data
        n = int(round(len(self.m) * shown))
        inside = (self.m > xr[0]) & (self.m < xr[1]) & (data > yr[0]) & (data < yr[1])
        inside[n:] = False
        for x, v in zip(self.m[inside], data[inside]):
            err = np.sqrt(v)
            self.polyline(sc, [self.P(chart.fx(x), chart.fy(v - err)),
                               self.P(chart.fx(x), chart.fy(v + err))], 1.4, INK, 0.42 * op)
        self.marks(sc, chart, self.m[inside], data[inside], INK, op, r=3.2 if zoom else 2.7)
        return chart

    def toy_hist(self, sc, box, cases, op, n=None, gauss=None, label=True):
        """Estimate minus truth over the pseudo-experiments, for the ``cases`` given."""
        e = self.hist_edges
        top = self.hist.max() * 1.08
        chart = self.chart(box, [e[0], e[-1]], [0, top], pad=0.0)
        for k in cases:
            h = self.hist[k] if n is None else np.histogram(self.diff[k][:n], e)[0]
            self.bars(sc, chart, e, h, FIT_INK[k], op * (0.85 if len(cases) == 1 else 0.7),
                      gap=0.6, front=k)
        self.axes(sc, chart, op, "estimated − true signal (events)" if label else "", "",
                  xticks=[(v, minus("%d" % v)) for v in (-200, -100, 0, 100)], yticks=[],
                  grid=False)
        self.vline(sc, chart, 0.0, INK, 0.45 * op, 1.6)
        if gauss is not None:
            xs = np.linspace(e[0], e[-1], 121)
            self.curve(sc, chart, xs, self.toys * (e[1] - e[0]) * S.norm_pdf(xs / gauss) / gauss,
                       INK, op, 1.0, 1.8)
        return chart

    def phase_1(self, sc, u, op):
        mo = self.model
        self.text(sc, 0.06, 0.315, "the model: %s background events falling as exp(−m / %g "
                  "GeV), and a peak of %d at %g GeV (σ = %g GeV)" % (
                      commas(mo["B"]), mo["tau"], mo["S"], mo["peak"], mo["sigma"]), op,
                  anchor="lm", strong=True)
        bands = smoothstep(0.5, 0.6, u)
        chart = self.spectrum(sc, op, shown=smoothstep(0.03, 0.3, u), bands=bands)
        go = smoothstep(0.3, 0.46, u)
        self.curve(sc, chart, self.m, self.total, SIG, op, go, 2.4)
        self.curve(sc, chart, self.m, self.bkg, BKG, op, go, 2.4)
        self.key(sc, 0.10, 0.915, [("true background", BKG), ("true background + peak", SIG)],
                 op * go, across=0.19)
        w, sb = mo["window"], mo["sidebands"]
        self.said(sc, 0.61, 0.42, [
            ("dots: one pseudo-experiment,", None, True),
            ("counted in bins of 1 GeV, ± √count", None, True),
            ("the window, %g to %g GeV:" % w, WINDOW),
            ("N = %s events, signal and background" % commas(self.one[0][0]), WINDOW),
            ("the sidebands, %g to %g and %g to %g:" % (sb[0] + sb[1]), SIDE),
            ("almost only background", SIDE),
            "they show what lies under the peak",
        ], u, op, start=0.34, every=0.075, step=0.056)
        self.tag(sc, DATA, op)

    def phase_2(self, sc, u, op):
        mo = self.model
        n, b, b_err, s, s_err = self.one[0][:5]
        self.text(sc, 0.06, 0.315, "fit in the sidebands only; carry the curve under the "
                  "peak; subtract", op, anchor="lm", strong=True)
        chart = self.spectrum(sc, op)
        lo, hi = mo["sidebands"][0][0], mo["sidebands"][1][1]
        keep = (self.m >= lo) & (self.m <= hi)
        go = smoothstep(0.06, 0.42, u)
        gap = smoothstep(0.44, 0.54, u)
        if gap > 0:
            wide = self.window
            edges = np.append(self.m[wide] - 0.5, self.m[wide][-1] + 0.5)
            self.bars(sc, chart, edges, np.maximum(self.data[wide] - self.fit_exp[wide], 0),
                      soft(SIG, 0.25), op * gap * 0.8, base=self.fit_exp[wide], front=1)
        self.curve(sc, chart, self.m[keep], self.fit_exp[keep], dark(FIT_INK[0], 0.1), op, go,
                   3.2, dot=go < 1)
        self.said(sc, 0.60, 0.40, [
            ("fit A · exp(−k·m) to the sideband bins", FIT_INK[0]),
            ("carried under the peak: b = %s ± %.0f" % (commas(round(b)), b_err), FIT_INK[0]),
            ("counted in the window: N = %s" % commas(n), WINDOW),
            ("signal = N − b = %.0f" % s, SIG),
            "error = √(N + %.0f²) = √(%s + %.0f) = %.0f" % (b_err, commas(n), b_err ** 2,
                                                            s_err),
            ("true signal in the window: %.0f" % self.truth[0], None, True),
            "the estimate is %.1fσ from the truth" % (abs(s - self.truth[0]) / s_err),
        ], u, op, start=0.1, every=0.11, step=0.058)
        self.tag(sc, DATA, op)

    def phase_3(self, sc, u, op):
        r = self.res[0]
        self.text(sc, 0.06, 0.315, "the same recipe on %s new spectra drawn from the same "
                  "model" % commas(self.toys), op, anchor="lm", strong=True)
        n = int(round(self.toys * smoothstep(0.04, 0.5, u)))
        late = op * smoothstep(0.52, 0.6, u)
        box = ((0.09, 0.55), (0.40, 0.76))
        self.toy_hist(sc, box, (0,), op, n=n)
        e = self.hist_edges
        chart = self.chart(box, [e[0], e[-1]], [0, self.hist.max() * 1.08], pad=0.0)
        xs = np.linspace(e[0], e[-1], 121)
        width = r["mean_quoted_error"]
        self.curve(sc, chart, xs, self.toys * (e[1] - e[0]) * S.norm_pdf(xs / width) / width,
                   INK, late, smoothstep(0.52, 0.7, u), 2.0)
        self.text(sc, 0.32, 0.375, "%s experiments" % commas(n), op, strong=True, size=HEAD)
        self.text(sc, 0.32, 0.895, "curve: a Gaussian of the quoted error's width, at zero",
                  late)
        self.said(sc, 0.61, 0.41, [
            ("estimate − truth, on average:", None, True),
            ("%s ± %.1f events" % (minus("%+.1f" % r["bias"]),
                                   r["spread"] / np.sqrt(self.toys)), FIT_INK[0]),
            "spread of the estimates: %.1f" % r["spread"],
            "average quoted error: %.1f" % r["mean_quoted_error"],
            ("pull = (estimate − truth) / error", None, True),
            "its width: %.2f  (ideal: 1)" % r["pull_width"],
            "within ±1σ: %s  (a Gaussian: 68.3%%)" % pct(r["within_1_sigma"], 1),
        ], u, op, start=0.5, every=0.055, step=0.056)
        self.tag(sc, toys(self.toys), op)

    def biased(self, sc, u, op, k, fit, data, total, head, rows):
        """Phases 4 and 5: one spectrum with a fit that sits too high under the peak,
        and the yields of all the pseudo-experiments beside the unbiased ones."""
        mo = self.model
        self.text(sc, 0.06, 0.315, head, op, anchor="lm", strong=True)
        chart = self.spectrum(sc, op, data=data, names=False, zoom=True)
        lo, hi = mo["sidebands"][0][0], mo["sidebands"][1][1]
        keep = (self.m >= lo) & (self.m <= hi)
        seen = (self.m > self.ZOOM[0][0]) & (self.m < self.ZOOM[0][1])
        self.curve(sc, chart, self.m[seen], self.bkg[seen], BKG, op, 1.0, 2.4)
        if total is not None:
            self.curve(sc, chart, self.m[seen], total[seen], SIG, op, 1.0, 2.0)
        go = smoothstep(0.08, 0.36, u)
        wide = self.window
        gap = smoothstep(0.36, 0.46, u)
        if gap > 0:
            edges = np.append(self.m[wide] - 0.5, self.m[wide][-1] + 0.5)
            self.bars(sc, chart, edges, np.maximum(fit[wide] - self.bkg[wide], 0),
                      soft(FIT_INK[k], 0.15), op * gap, base=self.bkg[wide], gap=0.0, front=1)
        self.curve(sc, chart, self.m[keep], fit[keep], dark(FIT_INK[k], 0.1), op, go, 3.2,
                   dot=go < 1)
        self.key(sc, 0.10, 0.915, [("true background", BKG), ("the fit", FIT_INK[k])], op,
                 across=0.19)
        late = op * smoothstep(0.46, 0.56, u)
        self.toy_hist(sc, ((0.62, 0.94), (0.385, 0.56)), (0, k), late, label=False)
        self.said(sc, 0.61, 0.655, rows, u, op, start=0.5, every=0.08, step=0.046)
        self.tag(sc, toys(self.toys), op)

    def phase_4(self, sc, u, op):
        r, one = self.res[1], self.one[1]
        self.biased(sc, u, op, 1, self.fit_line, None, None,
                    "the wrong shape: a straight line fitted to the same sidebands", [
                        ("this spectrum: b = %s, signal %.0f ± %.0f" % (
                            commas(round(one[1])), one[3], one[4]), FIT_INK[1]),
                        ("%s spectra: bias %s events = %sσ" % (
                            commas(self.toys), minus("%.0f" % r["bias"]),
                            minus("%.1f" % r["pull_mean"])), None, True),
                        "expected from the model alone: %s" % minus("%.0f" % r["bias_expected"]),
                        "sideband χ² per degree of freedom: %.2f" % r["chi2_per_ndf"],
                        "(with the right shape: %.2f)" % self.res[0]["chi2_per_ndf"],
                    ])

    def phase_5(self, sc, u, op):
        r, mo = self.res[2], self.model
        outside = 1.0 - self.truth[2] / mo["S"]
        self.biased(sc, u, op, 2, self.fit_wide, self.data_wide, self.total_wide,
                    "the right shape, but a peak of σ = %g GeV: its tails reach into the "
                    "sidebands" % mo["wide_sigma"], [
                        ("%s of the peak lies outside the window" % pct(outside), SIG),
                        ("%s spectra: bias %s events = %sσ" % (
                            commas(self.toys), minus("%.0f" % r["bias"]),
                            minus("%.1f" % r["pull_mean"])), None, True),
                        "expected from the model alone: %s" % minus("%.0f" % r["bias_expected"]),
                        ("clean sidebands, right shape: %s" % minus(
                            "%+.1f" % self.res[0]["bias"]), FIT_INK[0]),
                    ])


# ====================================================================== 140: sculpting
TRUE_INK, REC_INK, EFF_INK, FIX_INK = BLUE, RED, AMBER, GREEN
PT_INK = (DIM, TEAL, AMBER, VIOLET, MAGENTA, RED)


class SelectionSculpting(Select):
    """A falling spectrum and a trigger's turn-on, one bin worked through; the peak in
    what is recorded; dividing by the measured efficiency, with its error; that error
    over many pseudo-experiments; and a momentum cut putting a peak in a mass spectrum."""

    TITLE = "How a selection sculpts a spectrum"
    PHASES = (("truth", 11.0), ("recorded", 10.0), ("corrected", 12.0), ("error", 10.0),
              ("a cut", 14.0))
    BOX = ((0.10, 0.55), (0.52, 0.80))
    EFF_BOX = ((0.10, 0.55), (0.375, 0.455))
    WORKED = 7                                # the bin worked through: 28 to 32 GeV
    HIGH, LOW = 15, 3                         # bins of the two corrections worked through
    PAIR_SHOWN = (0, 1, 3, 5)                 # thresholds drawn: 0, 10, 20, 30 GeV

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        self.model = m["model"]
        p = s.panels[0].data
        self.e, self.edges = np.asarray(p["x"], float), np.asarray(p["edges"], float)
        self.true, self.rec, _trig = np.asarray(p["ys"], float)
        self.eff = np.asarray(p["efficiency"], float)
        p = s.panels[1].data
        self.ratio, self.spread, self.formula = np.asarray(p["ys"], float)
        self.usable = np.asarray(p["usable"], float)
        self.fix, self.fix_err = np.asarray(p["corrected"]), np.asarray(p["corrected_err"])
        self.cal = np.asarray(p["calibration_pass"], float)
        p = s.panels[2].data
        self.mass_x, self.mass_edges = np.asarray(p["x"], float), np.asarray(p["edges"], float)
        self.mass = np.asarray(p["ys"], float)
        p = s.panels[3].data
        self.cuts, self.peaks = np.asarray(p["cuts"], float), np.asarray(p["values"], float)
        self.kept, self.corr = np.asarray(p["kept"], float), float(p["correlation"])
        self.pairs = np.asarray(p["xy"], float)       # softer momentum, mass
        self.slope = float(m["pair_peak_slope"])
        self.toys = int(m["repeats"])
        self.peak_bin = int(np.argmax(self.rec))
        self.worst = int(np.argmax(self.spread * (self.usable > 0.98)))

    def bin_name(self, k):
        return "%d–%d GeV" % (self.edges[k], self.edges[k + 1])

    def subtitle_for(self, k):
        w, hi, lo, mo = self.WORKED, self.HIGH, self.LOW, self.model
        worst = self.worst
        return [
            "a smooth true spectrum meets a trigger that fires only above about %g GeV, in "
            "a detector that sees %s of directions. At %s, %s true events × efficiency "
            "%.3f: %s were recorded" % (mo["threshold"], pct(mo["acceptance"]),
                                        self.bin_name(w), commas(self.true[w]), self.eff[w],
                                        commas(self.rec[w])),
            "what is recorded has a different shape: it climbs through the trigger's "
            "turn-on and peaks at %s. The true spectrum has no peak anywhere: the "
            "selection made this one" % self.bin_name(self.peak_bin),
            "dividing each bin by its measured efficiency recovers the truth where the "
            "efficiency is high: %s ± %.0f for a true %s. At %s, where %s pass, the error "
            "is %s" % (commas(round(self.fix[hi])), self.fix_err[hi], commas(self.true[hi]),
                       self.bin_name(lo), pct(self.cal[lo] / mo["calibration"], 1),
                       pct(self.fix_err[lo] / self.fix[lo])),
            "over %s pseudo-experiments the corrected spectrum is unbiased above the "
            "turn-on, with errors of a few percent. At %s, where %s pass, the error is %s "
            "and the mean is off by %s" % (commas(self.toys), self.bin_name(worst),
                                           pct(self.eff[worst], 1), pct(self.spread[worst]),
                                           minus("%+.0f%%" % (100 * (self.ratio[worst] - 1)))),
            "a cut on one variable sculpts another. Requiring both momenta of a pair above "
            "a threshold puts a peak in the pair's mass, and the peak follows the cut: "
            "%.1f GeV higher for each GeV of threshold" % self.slope,
        ][k]

    # ------------------------------------------------------------------ drawing
    def turn_on(self, sc, op, go=1.0):
        mo = self.model
        chart = self.chart(self.EFF_BOX, [0, mo["e_max"]], [0, 1], pad=0.0)
        self.axes(sc, chart, op, "", "", xticks=[], yticks=[(0, "0"), (1, "1")], grid=False)
        xs = np.linspace(0, mo["e_max"], 201)
        self.curve(sc, chart, xs, S.efficiency(xs), dark(EFF_INK, 0.1), op, go, 3.0)
        return chart

    def spectrum(self, sc, op, top, rec=0.0, truth=1.0, worked=None):
        """True (and, by ``rec``, recorded) events per bin up to ``top``."""
        chart = self.chart(self.BOX, [0, self.model["e_max"]], [0, top], pad=0.0)
        if rec < 1:
            self.bars(sc, chart, self.edges, self.true, TRUE_INK, op * truth)
        else:
            self.bars(sc, chart, self.edges, self.true, soft(TRUE_INK, 0.6), op * truth)
        if rec > 0:
            self.bars(sc, chart, self.edges, self.true + (self.rec - self.true) * rec, REC_INK,
                      op, gap=3.0, front=1)
        ticks = [v for v in (0, 5000, 10000, 20000, 30000) if v <= top]
        self.axes(sc, chart, op, "energy E (GeV); events per 4 GeV", "",
                  xticks=[(v, "%d" % v) for v in (0, 20, 40, 60, 80, 100)],
                  yticks=[(v, commas(v)) for v in ticks], grid=False)
        if worked is not None:
            k = worked
            x = (chart.fx(self.edges[k]) + chart.fx(self.edges[k + 1])) / 2
            w = self.span(chart.fx(self.edges[k + 1]) - chart.fx(self.edges[k]))[0]
            self.ring(sc, x, (chart.y0 + chart.y1) / 2, w + 2,
                      self.span(0, chart.y1 - chart.y0)[1] + 4, dark(EFF_INK, 0.1), op)
        return chart

    def phase_1(self, sc, u, op):
        mo, w = self.model, self.WORKED
        self.text(sc, 0.06, 0.315, "the model: %s events falling as exp(−E / %g GeV); a "
                  "trigger; a detector that sees %s of directions" % (
                      commas(mo["n_true"]), mo["tau"], pct(mo["acceptance"])), op,
                  anchor="lm", strong=True)
        show = smoothstep(0.2, 0.3, u)
        self.turn_on(sc, op * show, smoothstep(0.22, 0.5, u))
        self.text(sc, 0.565, 0.415, "efficiency ε(E)", op * show, anchor="lm",
                  ink=dark(EFF_INK, 0.35))
        work = smoothstep(0.52, 0.6, u)
        self.spectrum(sc, op, 32000.0, worked=w if work > 0.5 else None)
        self.text(sc, 0.565, 0.56, "true spectrum", op, anchor="lm", ink=dark(TRUE_INK, 0.3))
        self.said(sc, 0.70, 0.40, [
            ("ε(E) = %g × trigger" % mo["acceptance"], EFF_INK),
            ("trigger: 1 / (1 + e^(−(E − %g) / %g))" % (mo["threshold"], mo["width"]), EFF_INK),
        ], u, op, start=0.3, every=0.08, step=0.048, size=0.029)
        self.said(sc, 0.70, 0.56, [
            ("the bin at %s:" % self.bin_name(w), None, True),
            ("%s true events" % commas(self.true[w]), TRUE_INK),
            ("× ε = %.3f" % self.eff[w], EFF_INK),
            ("expected: %s" % commas(round(self.true[w] * self.eff[w])), None),
            ("recorded: %s" % commas(self.rec[w]), REC_INK),
        ], u, op, start=0.56, every=0.065, step=0.05)
        self.tag(sc, DATA, op)

    def phase_2(self, sc, u, op):
        go = smoothstep(0.08, 0.45, u)
        top = 32000.0 + (11000.0 - 32000.0) * smoothstep(0.45, 0.7, u)
        self.text(sc, 0.06, 0.315, "every bin multiplied by its efficiency: what the "
                  "experiment records", op, anchor="lm", strong=True)
        self.turn_on(sc, op)
        self.text(sc, 0.565, 0.415, "efficiency ε(E)", op, anchor="lm", ink=dark(EFF_INK, 0.35))
        chart = self.spectrum(sc, op, top, rec=go, truth=1.0)
        k = self.peak_bin
        late = op * smoothstep(0.72, 0.8, u)
        x = (chart.fx(self.edges[k]) + chart.fx(self.edges[k + 1])) / 2
        w = self.span(chart.fx(self.edges[k + 1]) - chart.fx(self.edges[k]))[0]
        self.ring(sc, x, (chart.fy(self.rec[k]) + chart.y1) / 2, w + 3,
                  self.span(0, chart.y1 - chart.fy(self.rec[k]))[1] + 5, INK, late)
        self.key(sc, 0.11, 0.915, [("true", soft(TRUE_INK, 0.3)), ("recorded", REC_INK)], op,
                 across=0.09)
        self.said(sc, 0.60, 0.53, [
            ("recorded: a peak at %s" % self.bin_name(k), REC_INK),
            "the truth only falls",
            ("exact: the maximum is at %.1f GeV," % self.meta["recorded_peak"]["exact"],
             None, True),
            "where the turn-on climbs as fast",
            "as the spectrum falls",
            "(scale enlarged: the true bars run off the top)",
        ], u, op, start=0.6, every=0.045, step=0.05, size=0.029)
        self.tag(sc, DATA, op)

    def log_chart(self, box):
        return self.chart(box, [0, self.model["e_max"]], [200.0, 150000.0], ylog=True)

    def phase_3(self, sc, u, op):
        mo, hi, lo = self.model, self.HIGH, self.LOW
        self.text(sc, 0.06, 0.315, "undo it: divide each bin by its efficiency, measured on "
                  "%s calibration events per bin" % commas(mo["calibration"]), op,
                  anchor="lm", strong=True)
        box = ((0.10, 0.55), (0.38, 0.80))
        chart = self.log_chart(box)
        self.bars(sc, chart, self.edges, self.true, soft(TRUE_INK, 0.55), op)
        self.bars(sc, chart, self.edges, np.maximum(self.rec, 0), soft(REC_INK, 0.35), op,
                  gap=3.0, front=1)
        self.axes(sc, chart, op, "energy E (GeV); events per 4 GeV, log scale", "",
                  xticks=[(v, "%d" % v) for v in (0, 20, 40, 60, 80, 100)],
                  yticks=[(v, commas(v)) for v in (1000, 10000, 100000)], grid=False)
        n = int(round(len(self.e) * smoothstep(0.08, 0.5, u)))
        first = len(self.e) - n                       # from high energy down to low
        for k in range(first, len(self.e)):
            if self.fix[k] <= 0:
                continue
            x = chart.fx(self.e[k])
            self.polyline(sc, [self.P(x, chart.fy(max(self.fix[k] - self.fix_err[k], 200.0)),
                                      lift=1.0),
                               self.P(x, chart.fy(self.fix[k] + self.fix_err[k]), lift=1.0)],
                          2.2, dark(FIX_INK, 0.2), op)
            self.disk(sc, self.P(x, chart.fy(self.fix[k])), 3.4, dark(FIX_INK, 0.15), op,
                      lift=1.4, outline=0.0)
        self.key(sc, 0.11, 0.915, [("true", soft(TRUE_INK, 0.3)), ("recorded", REC_INK)], op,
                 across=0.09)
        self.key(sc, 0.32, 0.915, [("recorded ÷ efficiency", FIX_INK)], op)
        cal = mo["calibration"]
        rows = []
        for k in (hi, lo):
            rows += [
                ("%s:  ε = %s / %s = %.3f" % (self.bin_name(k), commas(self.cal[k]),
                                              commas(cal), self.cal[k] / cal), None, True),
                ("%s ÷ %.3f = %s ± %s" % (commas(self.rec[k]), self.cal[k] / cal,
                                         commas(round(self.fix[k])),
                                         commas(round(self.fix_err[k]))), FIX_INK),
                ("true: %s" % commas(self.true[k]), TRUE_INK),
            ]
        self.said(sc, 0.60, 0.40, rows[:3], u, op, start=0.3, every=0.06, step=0.05)
        self.said(sc, 0.60, 0.585, rows[3:], u, op, start=0.56, every=0.06, step=0.05)
        self.said(sc, 0.60, 0.77, ["relative error: √(1/N + (σ_ε/ε)²)",
                                   "small ε, few events: a large error"], u, op, start=0.76,
                  every=0.06, step=0.044)
        self.tag(sc, DATA, op)

    def phase_4(self, sc, u, op):
        worst = self.worst
        self.text(sc, 0.06, 0.315, "the correction repeated on %s pseudo-experiments: its "
                  "error, and its mean" % commas(self.toys), op, anchor="lm", strong=True)
        ok = self.usable > 0.98
        chart = self.chart(((0.10, 0.55), (0.40, 0.76)), [0, self.model["e_max"]], [0.01, 1.5],
                           ylog=True)
        self.axes(sc, chart, op, "energy E (GeV)", "relative error of the corrected count",
                  xticks=[(v, "%d" % v) for v in (0, 20, 40, 60, 80, 100)],
                  yticks=[(v, pct(v)) for v in (0.01, 0.1, 1.0)])
        go = smoothstep(0.08, 0.5, u)
        self.curve(sc, chart, self.e, self.formula, soft(FIX_INK, 0.1), op, go, 2.6,
                   ylog=True)
        shown = ok & (self.e <= self.e[0] + (self.e[-1] - self.e[0]) * go) & (go > 0)
        self.marks(sc, chart, self.e[shown], self.spread[shown], INK, op, r=3.4)
        self.key(sc, 0.11, 0.87, [("line: the formula √(1/N + (σ_ε/ε)²)", FIX_INK),
                                  ("dots: the spread over the experiments", INK)], op)
        high = self.e > self.model["threshold"] + 8
        dead = int(np.argmin(self.usable))
        self.said(sc, 0.60, 0.40, [
            ("above %d GeV:" % (self.model["threshold"] + 8), None, True),
            ("mean of corrected / true: %.3f to %.3f" % (self.ratio[high].min(),
                                                        self.ratio[high].max()), FIX_INK),
            "error %s to %s" % (pct(self.spread[high].min(), 1),
                                pct(self.spread[high].max(), 1)),
            ("at %s, ε = %s:" % (self.bin_name(worst), pct(self.eff[worst], 1)), None, True),
            ("error %s, mean ratio %.2f" % (pct(self.spread[worst]), self.ratio[worst]),
             REC_INK),
            ("at %s:" % self.bin_name(dead), None, True),
            ("no estimate at all in %s" % pct(1 - self.usable[dead]), REC_INK),
            "of the experiments",
        ], u, op, start=0.4, every=0.06, step=0.05)
        self.tag(sc, toys(self.toys), op)

    def phase_5(self, sc, u, op):
        self.text(sc, 0.06, 0.315, "another model: pairs of particles with falling momenta "
                  "and random angles; no particle decays into them", op, anchor="lm",
                  strong=True)
        shown = self.PAIR_SHOWN
        stage = min(int(u / 0.21), len(shown) - 1)
        within = smoothstep(0.0, 0.35, (u - 0.21 * stage) / 0.21)
        prev = self.cuts[shown[max(stage - 1, 0)]]
        cut = prev + (self.cuts[shown[stage]] - prev) * within
        box = ((0.10, 0.44), (0.40, 0.80))
        xr, yr = (0.0, 45.0), (0.0, 150.0)
        soft_pt, mass = self.pairs[:, 0], self.pairs[:, 1]
        keep = soft_pt > cut
        ink = PT_INK[shown[stage]] if stage else INK
        cols = np.where(keep[:, None], np.asarray(dark(ink, 0.1), np.float32),
                        np.asarray(soft(INK, 0.72), np.float32))
        self.picture(sc, raster(soft_pt, mass, xr, yr, (108, 163), cols, fat=2), box, op)
        (x0, x1), (y0, y1) = box

        def at(x, y):
            return (x0 + (x1 - x0) * (x - xr[0]) / (xr[1] - xr[0]),
                    y1 - (y1 - y0) * (y - yr[0]) / (yr[1] - yr[0]))
        if cut > 0.2:
            self.segment(sc, at, xr, yr, (cut, 0), (cut, 1), dark(CUT, 0.1), op, 3.0)
        for v in (0, 20, 40):
            self.text(sc, at(v, 0)[0], y1 + 0.03, "%d" % v, op)
        for v in (0, 50, 100, 150):
            self.text(sc, x0 - 0.012, at(0, v)[1], "%d" % v, op, anchor="rm")
        self.text(sc, (x0 + x1) / 2, y1 + 0.072, "the softer particle's momentum (GeV)", op)
        self.text(sc, x0, y0 - 0.03, "mass (GeV)", op, anchor="lm")
        self.text(sc, x1, y0 - 0.03, "both momenta > %.0f GeV" % cut, op, anchor="rm",
                  ink=dark(CUT, 0.35))
        box2 = ((0.56, 0.93), (0.40, 0.66))
        chart = self.chart(box2, [0, 150], [0, 1.08], pad=0.0)
        self.axes(sc, chart, op, "mass of the pair (GeV)", "pairs kept, each scaled to its "
                  "peak", xticks=[(v, "%d" % v) for v in (0, 50, 100, 150)], yticks=[],
                  grid=False)
        for n in range(stage + 1):
            k = shown[n]
            h = self.mass[k] / self.mass[k].max()
            on = op * (within if n == stage and stage else 1.0)
            if n == stage:
                self.bars(sc, chart, self.mass_edges, h, PT_INK[k] if k else soft(INK, 0.5),
                          on * 0.85)
            else:
                self.curve(sc, chart, self.mass_x, h, PT_INK[k] if k else soft(INK, 0.4), on,
                           1.0, 2.0)
        rows = []
        for n in range(stage + 1):
            k = shown[n]
            rows.append(("no cut: highest at %.0f GeV" % self.peaks[k] if k == 0 else
                         "> %.0f GeV: peak at %.0f GeV  (%s kept)" % (
                             self.cuts[k], self.peaks[k], pct(self.kept[k], 1)),
                         dark(PT_INK[k], 0.3) if k else None))
        self.lines(sc, 0.57, 0.765, rows, op, step=0.04)
        self.tag(sc, DATA, op)


ANIMATIONS = {
    "moving-a-cut": MovingACut,
    "cuts-vs-multivariate": CutsVsMultivariate,
    "sideband-background": SidebandBackground,
    "selection-sculpting": SelectionSculpting,
}

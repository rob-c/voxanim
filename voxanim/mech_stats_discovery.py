"""Discoveries, exclusion and upper limits: animations 145-149 of the "next" catalogue
(see :mod:`voxanim.mech`).

* :class:`BumpHunt`: one mass spectrum, two fits, ``q0``, the scan, the global significance;
* :class:`PoissonLimits`: where "2.30 at 90%" comes from, and what it promises;
* :class:`ClsMethod`: two tail areas and their ratio;
* :class:`BrazilBand`: the green and yellow bands built from pseudo-experiments;
* :class:`WhyFiveSigma`: an excess that grows and excesses that fade.

They draw the experiments of :mod:`voxanim.study_stats_discovery`. Nothing on screen is
measured data: every spectrum and count is pseudo-data from a model whose true values
are written beside it. Each phase says, bottom right, which kind of evidence it shows:
``exact`` (a formula or a distribution evaluated) or ``pseudo-experiments`` (seeded
draws from the stated model, and how many).

This module computes layout only, except where a distribution slides on screen: those
bars are :func:`~voxanim.study_stats_discovery.poisson_pmf` evaluated at the mean shown.
"""

from __future__ import annotations

import numpy as np

from . import study_stats_discovery as S
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
    OCHRE,
    RED,
    TEAL,
    VIOLET,
    dark,
    soft,
)
from .mech_chance import Chance

EXACT = "exact"
PSEUDO = "pseudo-experiments"
BIG, BODY = 0.034, 0.031
INV = "⁻¹"                                    # as in fb⁻¹


def commas(n):
    """An integer with thousands separators."""
    return format(int(round(float(n))), ",")


def pct(v, digits=1):
    return "%.*f%%" % (digits, 100.0 * float(v))


def sci(v):
    """A small probability as 2.1×10⁻⁵."""
    return pow10(float(v), 1)


def signed(v, fmt="%.2f"):
    """A number with a real minus sign."""
    return (fmt % float(v)).replace("-", "−")


def ramp(u, a, b):
    """0 before ``a``, 1 after ``b``, straight between."""
    return float(np.clip((u - a) / (b - a), 0.0, 1.0))


class Discovery(Chance):
    """What the five animations share: the evidence tag, bars and bands in chart
    coordinates, dashed rules, points with error bars."""

    TAGS: tuple = ()

    def tag(self, sc, k, op):
        """The kind of evidence phase ``k`` (0-based) shows, bottom right."""
        self.text(sc, 0.95, 0.925, self.TAGS[k], op, anchor="rm", ink=dark(TEAL, 0.25))

    def trace(self, sc, chart, xs, ys, ink, op, go=1.0, width=2.6, dots=True, r=4.0, lift=1.0):
        """``ys`` against ``xs`` through every point, drawn as far as fraction ``go`` of
        the axis, in front of bars and bands (:meth:`Chance.trace` draws behind them)."""
        if op <= 0.01 or go <= 0 or len(xs) == 0:
            return
        fx = [chart.fx(float(x)) for x in xs]
        fy = [chart.fy(float(y)) for y in ys]
        upto = fx[0] + (fx[-1] - fx[0]) * min(float(go), 1.0)
        pts = []
        for i in range(len(fx)):
            if fx[i] <= upto + 1e-9:
                pts.append((fx[i], fy[i]))
                if dots:
                    self.disk(sc, self.P(fx[i], fy[i]), r, ink, op, lift=lift + 0.4,
                              outline=0.0)
            else:
                a = (upto - fx[i - 1]) / (fx[i] - fx[i - 1])
                pts.append((upto, fy[i - 1] + a * (fy[i] - fy[i - 1])))
                break
        if len(pts) > 1:
            self.polyline(sc, [self.P(x, y, lift=lift) for x, y in pts], width, ink, op,
                          bias=0.3)

    def bars(self, sc, chart, xs, hs, wide, ink, op, inks=None, base=None, grow=1.0, lift=0.4):
        """Vertical bars at chart positions ``xs`` of heights ``hs`` (chart units),
        ``wide`` chart units across, standing on ``base`` (default: the chart's floor)."""
        if op <= 0.01:
            return
        y_base = chart.y1 if base is None else chart.fy(base)
        a, b = chart.x_range
        w = self.span((chart.x1 - chart.x0) * wide / (b - a))[0]
        for i, (x, h) in enumerate(zip(xs, hs)):
            if not a <= x <= b:
                continue
            y = y_base + (chart.fy(float(h)) - y_base) * grow
            tall = self.span(0, abs(y - y_base))[1]
            if tall < 0.5:
                continue
            self.rect(sc, self.P(chart.fx(float(x)), (y + y_base) / 2), w, tall,
                      ink if inks is None else inks[i], op, radius=min(2.0, w / 2, tall / 2),
                      lift=lift)

    def band(self, sc, chart, xs, lo, hi, ink, op, go=1.0, strips=60, lift=0.1):
        """The region between ``lo`` and ``hi`` over ``xs``, filled as far as ``go``."""
        if op <= 0.01 or go <= 0:
            return
        xs = np.asarray(xs, float)
        end = xs[0] + (xs[-1] - xs[0]) * min(float(go), 1.0)
        n = max(1, int(np.ceil(strips * min(float(go), 1.0))))
        cut = np.linspace(xs[0], end, n + 1)
        mid = 0.5 * (cut[:-1] + cut[1:])
        w = self.span(chart.fx(cut[1]) - chart.fx(cut[0]))[0] + 0.8
        for x, a, b in zip(mid, np.interp(mid, xs, lo), np.interp(mid, xs, hi)):
            ya, yb = chart.fy(float(a)), chart.fy(float(b))
            tall = self.span(0, abs(ya - yb))[1]
            if tall >= 0.5:
                self.rect(sc, self.P(chart.fx(float(x)), (ya + yb) / 2), w, tall, ink, op,
                          radius=0.0, lift=lift)

    def dash(self, sc, a, b, ink, op, n=14, width=1.6):
        """A dashed straight line between two layout points."""
        if op <= 0.01:
            return
        (ax, ay), (bx, by) = a, b
        for i in range(n):
            u0, u1 = i / n, (i + 0.55) / n
            self.polyline(sc, [self.P(ax + (bx - ax) * u0, ay + (by - ay) * u0, lift=0.8),
                               self.P(ax + (bx - ax) * u1, ay + (by - ay) * u1, lift=0.8)],
                          width, ink, op)

    def level(self, sc, chart, y, ink, op, n=18, width=1.6):
        """A dashed horizontal rule across ``chart`` at ``y``."""
        f = chart.fy(float(y))
        self.dash(sc, (chart.x0, f), (chart.x1, f), ink, op, n, width)

    def post(self, sc, chart, x, ink, op, width=2.4, top=None, bottom=None, dashed=False):
        """A vertical rule on ``chart`` at ``x`` (between chart values ``bottom``, ``top``)."""
        if op <= 0.01:
            return
        f = chart.fx(float(x))
        y0 = chart.y0 if top is None else chart.fy(top)
        y1 = chart.y1 if bottom is None else chart.fy(bottom)
        if dashed:
            self.dash(sc, (f, y0), (f, y1), ink, op, 12, width)
        else:
            self.polyline(sc, [self.P(f, y0, lift=0.9), self.P(f, y1, lift=0.9)], width, ink, op)

    def errors(self, sc, chart, xs, ys, es, ink, op, shown=None, r=3.4):
        """Points with a vertical bar of ``es`` either side; the first ``shown`` only."""
        for i, (x, y, e) in enumerate(zip(xs, ys, es)):
            if shown is not None and i >= shown:
                break
            f = chart.fx(float(x))
            self.polyline(sc, [self.P(f, chart.fy(float(y - e)), lift=0.6),
                               self.P(f, chart.fy(float(y + e)), lift=0.6)], 1.5, ink, 0.8 * op)
            self.disk(sc, self.P(f, chart.fy(float(y))), r, ink, op, lift=1.3, outline=0.0)

    def spots(self, sc, chart, xs, ys, ink, op, r=3.6, shown=None):
        """A dot per point of ``chart``."""
        for i, (x, y) in enumerate(zip(xs, ys)):
            if shown is not None and i >= shown:
                break
            self.disk(sc, self.P(chart.fx(float(x)), chart.fy(float(y))), r, ink, op, lift=1.3,
                      outline=0.0)

    def circle(self, sc, fx, fy, ink, op, size=20.0):
        """An outline around a point."""
        self.rect(sc, self.P(float(fx), float(fy)), size, size, ink, op, border=ink,
                  border_w=2.6, radius=size / 2, lift=1.0, fill_op=0.0)

    def rows(self, sc, fx, fy, rows, op, u=1.0, starts=(), step=0.05, size=BODY, width=None):
        """Lines of text (``str``, ``(str, ink)`` or ``(str, ink, True)`` for strong)
        stacked down from ``(fx, fy)``; row ``i`` fades in from ``starts[i]``."""
        y = fy
        for i, row in enumerate(rows):
            s, ink, strong = (row, None, False) if isinstance(row, str) else (*row, False)[:3]
            on = op * (smoothstep(starts[i], starts[i] + 0.06, u) if i < len(starts) else 1.0)
            if s:
                self.text(sc, fx, y, s, on, size=size, anchor="lm", ink=ink,
                          strong=strong and ink is None)
            y += step
        return y


# ====================================================================== 145: a bump hunt
class BumpHunt(Discovery):
    """One pseudo-data mass spectrum with a small injected signal: the background-only
    fit; the signal-plus-background fit and ``q0`` with its arithmetic; ``q0`` over
    background-only pseudo-experiments against half a chi-square; the scan in mass and
    its dip; the largest ``Z`` of background-only scans and the global significance."""

    TITLE = "A bump hunt, from fit to global significance"
    PHASES = (("the spectrum", 10.0), ("two fits", 13.0), ("q0", 11.0), ("the scan", 11.0),
              ("global", 11.0))
    TAGS = (PSEUDO + ": one spectrum, seed %d" % S.BUMP_SEED,
            PSEUDO + ": one spectrum, seed %d" % S.BUMP_SEED,
            PSEUDO + ": %s background-only spectra" % commas(S.N_FIXED),
            PSEUDO + ": one spectrum, seed %d" % S.BUMP_SEED,
            PSEUDO + ": %s background-only scans" % commas(S.N_SCAN))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p = s.panels
        self.x = np.asarray(p[0].data["x"], float)
        self.n, self.fit_b, self.fit_sb, self.truth = (np.asarray(v, float)
                                                       for v in p[0].data["ys"])
        self.res = self.n - self.fit_b
        self.err = np.sqrt(self.n)
        d = S.bump_data()
        self.tau_fit = float(-0.5 * (S.M_HI - S.M_LO) / d["theta_b"][1])
        self.q_x = np.asarray(p[1].data["x"], float)
        self.q_toys, self.q_exact = (np.asarray(v, float) for v in p[1].data["ys"])
        self.tails = np.asarray(p[1].data["tails"], float)
        self.tails_exact = np.asarray(p[1].data["tails_exact"], float)
        self.scan_m = np.asarray(p[2].data["x"], float)
        self.scan_p, self.scan_z = (np.asarray(v, float) for v in p[2].data["ys"])
        self.scan_s = np.asarray(p[2].data["fitted"], float)
        self.shapes = S.signal_shape(self.scan_m)          # the shape slid along the axis
        self.zmax_hist = np.asarray(p[3].data["values"], float) / m["global_"][
            "pseudo_experiments"]
        self.zmax_edges = np.asarray(p[3].data["edges"], float)
        self.zmax_median = float(np.median(p[3].data["z_max"]))
        lo, hi = self.zmax_edges[1:-1], self.zmax_edges[2:]
        self.one_mass = S.norm_sf(lo) - S.norm_sf(hi)      # Z at one fixed mass, exact
        self.g = m["global_"]

    def subtitle_for(self, k):
        m, g = self.meta, self.g
        t = m["q0_tails"]
        return [
            "a spectrum of pseudo-data: %s events drawn from a falling background plus a "
            "small bump of %d at m = %g. First fit the smooth background alone: two free "
            "numbers, no signal" % (commas(m["events"]), m["truth"]["signal_events"],
                                    m["truth"]["signal_mass"]),
            "then fit signal plus background. The likelihood L rises; twice the gain in "
            "ln L is the test statistic q0 = %.2f, and Z = √q0 = %.2f sigma: the local "
            "significance at this mass" % (m["q0"], m["z_at_true_mass"]),
            "is Z = √q0 honest? Over %s background-only pseudo-experiments q0 follows half "
            "a chi-square with one degree of freedom: %.2f%% pass 2 sigma (exact %.2f%%), "
            "%.3f%% pass 3 (exact %.3f%%)"
            % (commas(S.N_FIXED), 100 * t["pseudo_experiments"][1], 100 * t["exact"][1],
               100 * t["pseudo_experiments"][2], 100 * t["exact"][2]),
            "the mass is not known beforehand, so test all %d. The local p-value, the "
            "chance of so large a q0 from background alone at that mass, dips to %s at "
            "m = %g" % (m["scan"]["masses"], sci(m["scan"]["local_p"]), m["scan"]["best_mass"]),
            "but a scan gives chance many places to make a bump. Of %s background-only "
            "scans, %d reach Z = %.2f somewhere: the global p-value is %s, %.1f sigma "
            "instead of %.1f" % (commas(g["pseudo_experiments"]), g["at_or_above"],
                                 m["scan"]["local_z"], sci(g["p"]), g["z"],
                                 m["scan"]["local_z"]),
        ][k]

    # ------------------------------------------------------------------ phases
    def _model_line(self, sc, op):
        t = self.meta["truth"]
        self.text(sc, 0.5, 0.315, "pseudo-data from a stated model: %s background events "
                  "falling as exp(−m / %g)," % (commas(t["background_events"]),
                                                t["fall_length"]), op, size=BODY)
        self.text(sc, 0.5, 0.358, "plus %d signal events at m = %g, spread by a resolution "
                  "of %g" % (t["signal_events"], t["signal_mass"], t["resolution"]), op,
                  size=BODY, ink=dark(RED, 0.25))

    def phase_1(self, sc, u, op):
        self._model_line(sc, op)
        ch = LossChart(((0.10, 0.92), (0.43, 0.82)), (S.M_LO, S.M_HI), (0.0, 260.0))
        self.axes(sc, ch, op, "mass m", "events per bin of %g" % S.BIN_W,
                  xticks=[(v, "%d" % v) for v in range(100, 161, 10)],
                  yticks=[(v, "%d" % v) for v in (50, 100, 150, 200, 250)], grid=False)
        self.errors(sc, ch, self.x, self.n, self.err, DIGIT_INK, op,
                    shown=int(ramp(u, 0.04, 0.42) * len(self.x) + 1e-9))
        go = ramp(u, 0.48, 0.78)
        self.trace(sc, ch, self.x, self.fit_b, BLUE, op, go, width=3.0, dots=False)
        self.key(sc, 0.56, 0.47, [("pseudo-data, each bin ± √n", DIGIT_INK)], op)
        on = op * smoothstep(0.5, 0.6, u)
        self.key(sc, 0.56, 0.52, [("background-only fit: N exp(−m / τ)", BLUE)], on)
        self.text(sc, 0.576, 0.57, "fitted τ = %.1f (true: %g)"
                  % (self.tau_fit, self.meta["truth"]["fall_length"]),
                  op * smoothstep(0.78, 0.86, u), anchor="lm", size=BODY, ink=dark(BLUE, 0.3))
        self.tag(sc, 0, op)

    def _residuals(self, sc, box, op, shown=None, window=True):
        ch = LossChart(box, (S.M_LO, S.M_HI), (-48.0, 62.0))
        if window:
            a, b = S.WINDOW
            self.band(sc, ch, [a, b], [-48, -48], [62, 62], soft(AMBER, 0.72), op, strips=1)
        self.axes(sc, ch, op, "mass m" if window else "", "data − background fit",
                  xticks=[(v, "%d" % v) for v in range(100, 161, 10)],
                  yticks=[(v, "%d" % v) for v in (-40, 0, 40)])
        self.errors(sc, ch, self.x, self.res, self.err, DIGIT_INK, op, shown=shown, r=3.0)
        return ch

    def phase_2(self, sc, u, op):
        m = self.meta
        ch = self._residuals(sc, ((0.09, 0.55), (0.37, 0.80)), op)
        self.trace(sc, ch, self.x, self.fit_sb - self.fit_b, RED, op, ramp(u, 0.36, 0.56),
                   width=3.0, dots=False)
        w, ll = m["window"], m["log_likelihood"]
        rows = [
            ("in the window %g < m < %g:" % tuple(w["range"]), dark(AMBER, 0.45)),
            "n = %d observed, b = %.1f fitted" % (w["observed"], w["background_fit"]),
            "excess %.1f = %.1f × √b" % (w["observed"] - w["background_fit"], w["z"]),
            "",
            ("fit signal + background at m = %g:" % S.M_SIG, dark(RED, 0.25)),
            "s = %.0f ± %.0f events (true: %d)" % (m["fitted_signal"], m["fitted_signal_error"],
                                                    m["truth"]["signal_events"]),
            "",
            "L: the probability of this spectrum under a fit",
            "ln L(b) = %s" % "{:,.3f}".format(ll["background"]),
            "ln L(s + b) = %s" % "{:,.3f}".format(ll["signal_plus_background"]),
            "q0 = 2 × %.3f = %.2f" % (ll["signal_plus_background"] - ll["background"], m["q0"]),
            ("Z = √q0 = %.2f sigma" % m["z_at_true_mass"], None, True),
        ]
        self.rows(sc, 0.60, 0.335, rows, op, u, step=0.0455,
                  starts=(0.06, 0.12, 0.2, 0, 0.36, 0.48, 0, 0.6, 0.66, 0.7, 0.78, 0.87))
        self.tag(sc, 1, op)

    def phase_3(self, sc, u, op):
        m = self.meta
        ch = LossChart(((0.10, 0.57), (0.38, 0.80)), (0.0, 20.0), (1e-6, 1.0), ylog=True)
        self.axes(sc, ch, op, "q0", "share of spectra per bin of 0.5",
                  xticks=[(v, "%d" % v) for v in (0, 4, 9, 16)],
                  yticks=[(10.0 ** e, pow10(10.0 ** e)) for e in (-6, -4, -2, 0)])
        shown = int(ramp(u, 0.04, 0.45) * len(self.q_x) + 1e-9)
        self.bars(sc, ch, self.q_x[:shown], self.q_toys[:shown], 0.4, TEAL, op)
        self.trace(sc, ch, self.q_x, np.maximum(self.q_exact, 1e-7), VIOLET, op,
                   ramp(u, 0.45, 0.7), width=2.4, dots=False)
        seen = op * smoothstep(0.8, 0.88, u)
        self.post(sc, ch, m["q0"], RED, seen, top=3e-3)
        self.text(sc, ch.fx(m["q0"]), ch.fy(3.5e-2), "this spectrum", seen, size=BODY,
                  ink=dark(RED, 0.25))
        x0 = 0.63
        self.rows(sc, x0, 0.335, [
            ("%s spectra with no signal," % commas(S.N_FIXED), None, True),
            ("each fitted both ways", None, True),
            ("bars: their q0", dark(TEAL, 0.3)),
            ("line: half a chi-square, exact", dark(VIOLET, 0.25)),
            "%s fit s < 0: q0 = 0" % pct(m["q0_zero_share"]),
        ], op, u, starts=(0.02, 0.02, 0.1, 0.45, 0.52))
        on = op * smoothstep(0.62, 0.7, u)
        for c, head in enumerate(("√q0 above", "found", "exact")):
            self.text(sc, x0 + 0.045 + 0.105 * c, 0.615, head, on, size=BODY)
        for r in range(3):
            y = 0.665 + 0.048 * r
            self.text(sc, x0 + 0.045, y, "%d" % (r + 1), on, size=BODY)
            fmt = "%.1f%%" if r == 0 else "%.2f%%" if r == 1 else "%.3f%%"
            self.text(sc, x0 + 0.15, y, fmt % (100 * self.tails[r]), on, size=BODY,
                      ink=dark(TEAL, 0.3))
            self.text(sc, x0 + 0.255, y, fmt % (100 * self.tails_exact[r]), on, size=BODY,
                      ink=dark(VIOLET, 0.25))
        self.text(sc, x0, 0.835, "so p = tail beyond %.2f = %s"
                  % (m["z_at_true_mass"], sci(m["p_at_true_mass"])), seen, anchor="lm",
                  size=BODY, strong=True)
        self.tag(sc, 2, op)

    def phase_4(self, sc, u, op):
        m = self.meta
        k = int(ramp(u, 0.06, 0.72) * (len(self.scan_m) - 1) + 1e-9)
        done = u >= 0.72
        top = self._residuals(sc, ((0.10, 0.86), (0.37, 0.53)), op, window=False)
        ch = LossChart(((0.10, 0.86), (0.63, 0.83)), (S.M_LO, S.M_HI), (3e-6, 1.0), ylog=True)
        self.axes(sc, ch, op, "mass m tested", "local p-value",
                  xticks=[(v, "%d" % v) for v in range(100, 161, 10)],
                  yticks=[(10.0 ** e, pow10(10.0 ** e)) for e in (-4, -2, 0)])
        for z in (1, 2, 3, 4):
            self.level(sc, ch, float(S.norm_sf(z)), AXIS, 0.45 * op, n=26, width=1.2)
            if z > 1:
                self.text(sc, ch.x1 + 0.012, ch.fy(float(S.norm_sf(z))), "%dσ" % z, op,
                          anchor="lm", size=BODY)
        if not done:
            tpl = self.scan_s[k] * self.shapes[k]
            self.trace(sc, top, self.x, tpl, RED, op, 1.0, width=2.8, dots=False)
            self.post(sc, ch, self.scan_m[k], RED, 0.45 * op, width=1.6)
        else:
            best = int(np.argmax(self.scan_z))
            self.trace(sc, top, self.x, self.scan_s[best] * self.shapes[best], RED, op, 1.0,
                       width=2.8, dots=False)
        self.trace(sc, ch, self.scan_m[:k + 1], self.scan_p[:k + 1], BLUE, op, 1.0, width=2.8,
                   dots=False)
        self.disk(sc, self.P(ch.fx(self.scan_m[k]), ch.fy(self.scan_p[k])), 5.0, BLUE, op,
                  lift=1.5, outline=0.0)
        if not done:
            self.text(sc, 0.86, 0.315, "signal shape at m = %d: fitted s = %.0f, Z = %.2f"
                      % (self.scan_m[k], self.scan_s[k], self.scan_z[k]), op, anchor="rm",
                      size=BODY, strong=True)
        else:
            self.text(sc, 0.86, 0.315, "the dip: p = %s at m = %g, Z = %.2f sigma, local"
                      % (sci(m["scan"]["local_p"]), m["scan"]["best_mass"],
                         m["scan"]["local_z"]), op, anchor="rm", size=BODY, strong=True)
        self.tag(sc, 3, op)

    def phase_5(self, sc, u, op):
        m, g = self.meta, self.g
        ch = LossChart(((0.10, 0.58), (0.38, 0.80)), (0.0, 5.0), (0.0, 0.2))
        self.axes(sc, ch, op, "largest Z anywhere in the scan", "share of scans",
                  xticks=[(v, "%d" % v) for v in range(6)],
                  yticks=[(v, "%d%%" % round(100 * v)) for v in (0, 0.05, 0.1, 0.15, 0.2)])
        mid = 0.5 * (self.zmax_edges[:-1] + self.zmax_edges[1:])
        self.trace(sc, ch, mid[1:], self.one_mass, VIOLET, op, ramp(u, 0.02, 0.2), width=2.4,
                   dots=False)
        self.bars(sc, ch, mid, self.zmax_hist, 0.2, TEAL, op, grow=smoothstep(0.2, 0.5, u))
        z = m["scan"]["local_z"]
        seen = op * smoothstep(0.55, 0.63, u)
        self.post(sc, ch, z, RED, seen, top=0.12)
        self.text(sc, ch.fx(z), ch.fy(0.135), "observed %.2f" % z, seen, size=BODY,
                  ink=dark(RED, 0.25))
        self.rows(sc, 0.635, 0.335, [
            ("%s background-only spectra," % commas(g["pseudo_experiments"]), None, True),
            ("each scanned over %d masses" % m["scan"]["masses"], None, True),
            ("line: Z at one fixed mass (exact)", dark(VIOLET, 0.25)),
            ("bars: the largest Z of a scan,", dark(TEAL, 0.3)),
            ("typically %.1f" % self.zmax_median, dark(TEAL, 0.3)),
            "at or above %.2f: %d of %s" % (z, g["at_or_above"],
                                            commas(g["pseudo_experiments"])),
            "global p = %s ± %s" % (sci(g["p"]), sci(g["p_error"])),
            ("= %.2f sigma" % g["z"], None, True),
            "the scan costs a factor %.0f in p" % g["trials_factor"],
        ], op, u, starts=(0.2, 0.2, 0.04, 0.3, 0.3, 0.62, 0.72, 0.8, 0.88))
        self.tag(sc, 4, op)


# ====================================================================== 146: upper limits
class PoissonLimits(Discovery):
    """The Poisson distribution of the count for a sliding signal mean, with the chance
    of "n observed or fewer" shaded until it is 10%; the limits for n = 0..3; a known
    background; coverage over pseudo-experiments against the exact curve; and the
    limit as a cross-section."""

    TITLE = "An upper limit from counting"
    PHASES = (("zero events", 10.0), ("n or fewer", 13.0), ("a background", 10.0),
              ("coverage", 11.0), ("cross-section", 8.0))
    TAGS = (EXACT + ": Poisson probabilities", EXACT + ": Poisson probabilities",
            EXACT + ": Poisson probabilities",
            PSEUDO + ": %s counts per true s; line: exact" % commas(S.N_COVER),
            EXACT + ": one division")

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p = s.panels
        self.grid_s = np.asarray(p[0].data["x"], float)
        self.cdf = np.asarray(p[0].data["ys"], float)
        self.lim = np.asarray(p[1].data["values"], float)
        self.with_b = np.asarray(p[1].data["with_background"], float)
        self.cover_s = np.asarray(p[2].data["x"], float)
        self.cover, self.cover_exact = (np.asarray(v, float) for v in p[2].data["ys"])
        self.fine_x = np.asarray(p[2].data["fine_x"], float)
        self.fine = np.asarray(p[2].data["fine"], float)
        self.example = np.asarray(p[3].data["xy"], float)
        self.sigma = m["cross_section"]["limit_fb"]
        self.b = float(m["background"])

    def subtitle_for(self, k):
        m, lim = self.meta, self.lim
        c, x = m["coverage"], m["cross_section"]
        return [
            "zero events seen. A signal expected to give s events on average gives none "
            "with probability exp(−s). That falls to 10%% at s = %.2f: larger signals are "
            "excluded at 90%% confidence level" % lim[0],
            "with n events seen the question is the same: for which s would n or fewer "
            "happen only 10%% of the time? Sliding s until the shaded probability is 10%% "
            "gives %.2f, %.2f and %.2f for n = 1, 2, 3" % (lim[1], lim[2], lim[3]),
            "with a known background b = %g the count has mean s + b, so b comes off the "
            "limit: s below %.2f for n = 3. For n = 0 it is %s: a deficit excludes every "
            "signal, the flaw the CLs method repairs"
            % (self.b, self.with_b[3], signed(self.with_b[0])),
            "does 90%% mean 90%%? Over %s pseudo-experiments per true s the limit lies "
            "above the truth at least %s of the time, within %.1f points of the exact "
            "curve, whose lowest value is %s"
            % (commas(S.N_COVER), pct(c["lowest_pseudo"]), 100 * c["largest_gap"],
               pct(c["lowest_exact"])),
            "a limit on events becomes one on physics: s = cross-section × luminosity × "
            "efficiency. With %g fb%s and an efficiency of %g, zero events give a "
            "cross-section below %.2f fb" % (x["luminosity_fb"], INV, x["efficiency"],
                                             self.sigma[0]),
        ][k]

    def _poisson(self, sc, box, mean, n_obs, op, top, ymax):
        """The distribution of the count for ``mean``, with ``n_obs`` or fewer in red."""
        ch = LossChart(box, (-0.6, top + 0.6), (0.0, ymax))
        ks = np.arange(top + 1)
        self.axes(sc, ch, op, "events counted, n", "probability",
                  xticks=[(k, "%d" % k) for k in range(0, top + 1, 2)],
                  yticks=[(v, "%d%%" % round(100 * v))
                          for v in np.arange(0.0, ymax + 1e-9, 0.1 if ymax < 0.5 else 0.2)])
        self.bars(sc, ch, ks, S.poisson_pmf(ks, mean), 0.72, BLUE, op,
                  inks=[RED if k <= n_obs else soft(BLUE, 0.35) for k in ks])
        return ch

    def phase_1(self, sc, u, op):
        s = 0.4 + (self.lim[0] - 0.4) * smoothstep(0.1, 0.72, u)
        ch = self._poisson(sc, ((0.10, 0.55), (0.40, 0.82)), s, 0, op, 9, 0.7)
        self.level(sc, ch, 0.10, AMBER, op, n=20, width=2.0)
        self.text(sc, ch.x1 + 0.012, ch.fy(0.10), "10%", op, anchor="lm", size=BODY,
                  ink=dark(AMBER, 0.4))
        x0 = 0.63
        self.rows(sc, x0, 0.335, [
            ("an experiment counts events:", None, True),
            ("it sees n = 0", None, True),
            "s: the number of signal events",
            "expected on average",
        ], op)
        self.text(sc, x0, 0.555, "s = %.2f" % s, op, anchor="lm", size=0.044, strong=True)
        self.text(sc, x0, 0.625, "P(0 events) = exp(−s) = %s" % pct(np.exp(-s)), op,
                  anchor="lm", size=BIG, ink=dark(RED, 0.25))
        self.rows(sc, x0, 0.705, [
            "exp(−s) = 10%% at s = ln 10 = %.2f" % self.lim[0],
            ("s above %.2f: excluded," % self.lim[0], None, True),
            ("90% confidence level", None, True),
        ], op, u, starts=(0.74, 0.82, 0.82))
        self.tag(sc, 0, op)

    def phase_2(self, sc, u, op):
        seg = min(int(u * 3 / 0.97), 2)
        f = float(np.clip(u * 3 / 0.97 - seg, 0, 1))
        n = seg + 1
        s = self.lim[n - 1] + (self.lim[n] - self.lim[n - 1]) * smoothstep(0.1, 0.7, f)
        self._poisson(sc, ((0.09, 0.50), (0.46, 0.82)), s, n, op, 12, 0.4)
        self.text(sc, 0.295, 0.315, "n = %d seen" % n, op, size=BIG, strong=True)
        self.text(sc, 0.295, 0.365, "P(N ≤ %d) at s = %.2f:  %s"
                  % (n, s, pct(S.poisson_cdf(n, s))), op, size=BODY, ink=dark(RED, 0.25))
        inks = (OCHRE, VIOLET, TEAL, MAGENTA)
        cv = LossChart(((0.62, 0.93), (0.36, 0.62)), (0.0, 9.0), (0.0, 1.0))
        self.axes(sc, cv, op, "signal mean s", "P(n or fewer)",
                  xticks=[(v, "%d" % v) for v in (0, 2, 4, 6, 8)],
                  yticks=[(0.1, "10%"), (0.5, "50%"), (1.0, "100%")])
        self.level(sc, cv, 0.10, AMBER, op, n=16, width=1.8)
        for j in range(n + 1):
            self.trace(sc, cv, self.grid_s, self.cdf[j], inks[j], op, 1.0, width=2.4,
                       dots=False)
            if j < n or f > 0.7:
                self.post(sc, cv, self.lim[j], inks[j], 0.45 * op, width=1.4, top=0.10)
                self.disk(sc, self.P(cv.fx(self.lim[j]), cv.fy(0.10)), 5.0, inks[j], op,
                          lift=1.5, outline=0.0)
                self.text(sc, 0.63 + 0.155 * (j % 2), 0.745 + 0.05 * (j // 2),
                          "n = %d:  s < %.2f" % (j, self.lim[j]), op, anchor="lm", size=BODY,
                          ink=dark(inks[j], 0.3))
        self.disk(sc, self.P(cv.fx(s), cv.fy(float(S.poisson_cdf(n, s)))), 5.5, RED, op,
                  lift=1.6, outline=0.0, glow=0.4)
        self.text(sc, 0.63, 0.85, "the 90% upper limits", op, anchor="lm", size=BODY)
        self.tag(sc, 1, op)

    def phase_3(self, sc, u, op):
        self.text(sc, 0.5, 0.315, "a known background b = %g: the count has mean s + b, so "
                  "the limit is on s + b" % self.b, op, size=BIG, strong=True)
        x0, x1 = 0.20, 0.74

        def at(v):
            return x0 + (x1 - x0) * v / 10.0
        self.text(sc, 0.5 * (x0 + x1), 0.37, "largest mean of the count allowed at 90%", op,
                  size=BODY)
        for v in range(0, 11, 2):
            self.text(sc, at(v), 0.415, "%d" % v, op, size=BODY)
        px = self.span(x1 - x0)[0] / 10.0
        for i in range(6):
            y = 0.47 + 0.064 * i
            on = op * smoothstep(0.06 + 0.1 * i, 0.12 + 0.1 * i, u)
            total = self.lim[i]
            self.text(sc, x0 - 0.02, y, "n = %d" % i, on, anchor="rm", size=BODY)
            self.rect(sc, self.P(at(self.b / 2), y), self.b * px, 20, soft(BLUE, 0.2), on,
                      radius=3)
            if total >= self.b:
                self.rect(sc, self.P(at((self.b + total) / 2), y), (total - self.b) * px, 20,
                          GREEN, on, radius=3)
                self.text(sc, 0.77, y, "s < %.2f − %g = %.2f" % (total, self.b, self.with_b[i]),
                          on, anchor="lm", size=BODY)
            else:
                self.rect(sc, self.P(at((self.b + total) / 2), y), (self.b - total) * px, 20,
                          RED, on, radius=3, lift=0.8)
                self.text(sc, 0.77, y, "s < %.2f − %g = %s"
                          % (total, self.b, signed(self.with_b[i])), on, anchor="lm",
                          size=BODY, ink=dark(RED, 0.25))
        self.key(sc, 0.08, 0.875, [("background b", soft(BLUE, 0.2))], op)
        self.key(sc, 0.24, 0.875, [("room left for a signal s", GREEN)], op)
        self.key(sc, 0.52, 0.875, [("n = 0: less than b is allowed, s < 0",
                                    RED)], op * smoothstep(0.7, 0.8, u))
        self.tag(sc, 2, op)

    def phase_4(self, sc, u, op):
        m = self.meta
        true = m["example"]["true_signal"]
        n = len(self.example)
        ch = LossChart(((0.09, 0.50), (0.46, 0.82)), (0.0, n + 1.0), (0.0, 14.0))
        self.axes(sc, ch, op, "pseudo-experiment", "upper limit on s",
                  xticks=[(v, "%d" % v) for v in (1, 10, 20, 30, 40)],
                  yticks=[(v, "%d" % v) for v in (0, 4, 8, 12)])
        shown = int(ramp(u, 0.04, 0.42) * n + 1e-9)
        for i in range(shown):
            ok = self.example[i, 1] >= true
            ink = TEAL if ok else RED
            f = ch.fx(i + 1.0)
            self.polyline(sc, [self.P(f, ch.fy(0.0), lift=0.6),
                               self.P(f, ch.fy(self.example[i, 1]), lift=0.6)], 2.4, ink, op)
            self.disk(sc, self.P(f, ch.fy(self.example[i, 1])), 3.2, ink, op, lift=1.3,
                      outline=0.0)
        self.level(sc, ch, true, DIGIT_INK, op, n=22, width=1.8)
        self.text(sc, 0.295, 0.315, "true s = %g: each draws a count, sets its limit" % true,
                  op, size=BODY, strong=True)
        self.text(sc, 0.295, 0.362, "%d of %d limits lie above the truth (red: missed)"
                  % (m["example"]["covered"], n), op * smoothstep(0.44, 0.52, u), size=BODY)
        cv = LossChart(((0.62, 0.93), (0.46, 0.82)), (0.0, 10.0), (0.86, 1.0))
        self.axes(sc, cv, op, "true signal s", "coverage",
                  xticks=[(v, "%d" % v) for v in (0, 2, 4, 6, 8, 10)],
                  yticks=[(v, "%d%%" % round(100 * v)) for v in (0.9, 0.95, 1.0)])
        self.level(sc, cv, 0.90, AMBER, op, n=16, width=2.0)
        go = ramp(u, 0.5, 0.8)
        self.trace(sc, cv, self.fine_x, self.fine, VIOLET, op, go, width=2.0, dots=False)
        self.spots(sc, cv, self.cover_s, self.cover, TEAL, op, r=3.2,
                   shown=int(go * len(self.cover_s) + 1e-9))
        self.text(sc, 0.775, 0.315, "share of limits above the truth", op, size=BODY,
                  strong=True)
        self.text(sc, 0.775, 0.362, "never under 90%: it over-covers",
                  op * smoothstep(0.82, 0.9, u), size=BODY)
        self.tag(sc, 3, op)

    def phase_5(self, sc, u, op):
        x = self.meta["cross_section"]
        self.text(sc, 0.5, 0.325, "s = σ × L × ε", op, size=0.046, strong=True)
        self.rows(sc, 0.16, 0.40, [
            "σ: the cross-section, how often the process happens (in fb)",
            "L: the integrated luminosity, how much data: here %g fb%s" % (x["luminosity_fb"],
                                                                         INV),
            "ε: the efficiency, the share of signal events that pass the selection: %g"
            % x["efficiency"],
        ], op, step=0.046)
        y = 0.60
        cells = (("s < %.2f" % self.lim[0], 0.20, GREEN, 104),
                 ("%g fb%s" % (x["luminosity_fb"], INV), 0.40, DIM, 96),
                 ("%g" % x["efficiency"], 0.56, DIM, 76),
                 ("σ < %.2f fb" % self.sigma[0], 0.78, AMBER, 132))
        for i, (s, fx, ink, w) in enumerate(cells):
            self.box(sc, fx, y, w, 40, soft(ink, 0.5), op * smoothstep(0.1 + 0.12 * i,
                                                                      0.18 + 0.12 * i, u),
                     s, border=ink, size=BIG)
        self.text(sc, 0.30, y, "÷ (", op * smoothstep(0.22, 0.3, u), size=BIG, strong=True)
        self.text(sc, 0.485, y, "×", op * smoothstep(0.34, 0.42, u), size=BIG, strong=True)
        self.text(sc, 0.645, y, ")  =", op * smoothstep(0.46, 0.54, u), size=BIG, strong=True)
        for i in range(4):
            self.hbar(sc, 0.30, 0.70 + 0.05 * i, 190 * self.sigma[i], AMBER,
                      op * smoothstep(0.55, 0.62, u), "n = %d seen" % i,
                      "σ < %.2f fb" % self.sigma[i], u=ramp(u, 0.58 + 0.05 * i, 0.72 + 0.05 * i),
                      thick=16)
        self.tag(sc, 4, op)


# ====================================================================== 147: CLs
class ClsMethod(Discovery):
    """A count far below the expected background: the plain limit sliding to a negative
    signal; CLs+b and CLb as the two left tails and their ratio; both limits for every
    count; how often each excludes a signal it cannot see; and the coverage."""

    TITLE = "The CLs method"
    PHASES = (("a deficit", 11.0), ("two tails", 12.0), ("count by count", 10.0),
              ("small signals", 11.0), ("coverage", 11.0))
    TAGS = (EXACT + ": Poisson probabilities", EXACT + ": Poisson probabilities",
            EXACT + " limits; counts: %s " % commas(S.N_CLS) + PSEUDO,
            PSEUDO + ": %s, background only" % commas(S.N_CLS),
            PSEUDO + ": %s per true s; lines: exact" % commas(S.N_CLS_COVER))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p = s.panels
        self.counts = np.asarray(p[0].data["x"], float)
        self.plain, self.cls = (np.asarray(v, float) for v in p[0].data["ys"])
        self.below = np.asarray(p[1].data["values"], float).reshape(2, -1)
        self.below_exact = np.asarray(p[1].data["exact"], float).reshape(2, -1)
        self.n_hist = np.asarray(p[1].data["counts"], float) / S.N_CLS
        self.cover_s = np.asarray(p[2].data["x"], float)
        self.cover = np.asarray(p[2].data["ys"], float)
        self.fine_x = np.asarray(p[2].data["fine_x"], float)
        self.fine = np.asarray(p[2].data["fine"], float)
        self.b, self.n_obs = float(m["background"]), int(m["example"]["observed"])
        self.first_ok = int(np.flatnonzero(self.plain >= 0)[0])

    def subtitle_for(self, k):
        m = self.meta
        e, x, c = m["example"], m["excludes"], m["coverage"]
        return [
            "expect b = %g background events, count %d. The plain rule excludes a signal s "
            "when %d or fewer would turn up less than 5%% of the time. Here that holds "
            "even for s = 0: the limit comes out at %s"
            % (self.b, self.n_obs, self.n_obs, signed(e["plain_limit"])),
            "the fix compares two tail areas: CLs+b, the chance of %d or fewer with the "
            "signal, and CLb, the same with background alone. Their ratio CLs = %.3f is "
            "above 0.05: a signal of %g is not excluded"
            % (self.n_obs, e["cls"], e["tested_signal"]),
            "count by count: the plain 95%% limit is negative below n = %d, while the CLs "
            "limit never drops under %.1f; for n = %d it is %.2f. Well above the expected "
            "%g the two agree" % (self.first_ok, self.cls.min(), self.n_obs, e["cls_limit"],
                                  self.b),
            "over %s background-only pseudo-experiments the plain limit rules out a signal "
            "of %g event %s of the time, and no signal at all %s of the time. The CLs "
            "limit never does"
            % (commas(S.N_CLS), x["signals"][1], pct(x["plain"][1]), pct(x["plain"][0])),
            "the price is over-coverage. A 95%% limit should hold 95%% of the time; CLs "
            "holds %s of the time for a true s = %g, and with no signal its typical limit "
            "is weaker: %.1f against %.1f"
            % (pct(c["cls_at_smallest"], 0), c["smallest_signal"], m["median_limit"]["cls"],
               m["median_limit"]["plain"]),
        ][k]

    def _two(self, sc, box, s, op, both=True, zoom=0.0):
        """The count's distribution with background only (blue) and with a signal
        ``s`` added (red); the bars at or below the observed count at full strength.
        ``zoom`` (0..1) closes in on the low counts, where the two tails are."""
        top = 25.8 + (6.7 - 25.8) * zoom
        ymax = 0.135 + (0.066 - 0.135) * zoom
        ch = LossChart(box, (-0.8, top), (0.0, ymax))
        ks = np.arange(26)
        self.axes(sc, ch, op, "events counted, n", "",
                  xticks=[(k, "%d" % k) for k in range(0, int(top) + 1, 5 if top > 12 else 1)],
                  yticks=[(v, "%d%%" % round(100 * v))
                          for v in ((0, 0.05, 0.1) if ymax > 0.09 else (0, 0.02, 0.04, 0.06))],
                  grid=False)
        tail = ks <= self.n_obs
        self.bars(sc, ch, ks - 0.2, S.poisson_pmf(ks, self.b), 0.36, BLUE, op,
                  inks=[BLUE if (t and both) else soft(BLUE, 0.45) for t in tail])
        self.bars(sc, ch, ks + 0.2, S.poisson_pmf(ks, s + self.b), 0.36, RED, op,
                  inks=[RED if t else soft(RED, 0.5) for t in tail])
        f = ch.fx(self.n_obs + 0.5)
        self.dash(sc, (f, ch.y0 + 0.03), (f, ch.y1), AMBER, op, 14, 2.2)
        self.text(sc, f, ch.y0, "seen: n = %d" % self.n_obs, op, size=BODY,
                  ink=dark(AMBER, 0.45))
        return ch

    def phase_1(self, sc, u, op):
        e = self.meta["example"]
        s = 6.0 + (e["plain_limit"] - 6.0) * smoothstep(0.12, 0.78, u)
        self._two(sc, ((0.09, 0.58), (0.40, 0.82)), s, op, both=False)
        p = float(S.poisson_cdf(self.n_obs, s + self.b))
        x0 = 0.635
        self.rows(sc, x0, 0.335, [
            ("background: b = %g expected, known" % self.b, dark(BLUE, 0.25)),
            ("with a signal: s + b expected", dark(RED, 0.25)),
            ("counted: n = %d" % self.n_obs, dark(AMBER, 0.45)),
        ], op)
        self.text(sc, x0, 0.515, "test s = %s" % signed(s), op, anchor="lm", size=0.042,
                  strong=True)
        self.text(sc, x0, 0.585, "P(N ≤ %d | s + b) = %s" % (self.n_obs, pct(p, 2)), op,
                  anchor="lm", size=BIG, ink=dark(RED, 0.25))
        self.text(sc, x0, 0.64, "below 5%: excluded" if p < 0.04999 else "5%: the limit",
                  op, anchor="lm", size=BODY)
        self.rows(sc, x0, 0.72, [
            "s = 0 gives %s, still below 5%%" % pct(e["cl_b"]),
            ("the plain 95%% limit: s < %s" % signed(e["plain_limit"]), None, True),
            ("every signal excluded, even none", None, True),
        ], op, u, starts=(0.6, 0.8, 0.86))
        self.tag(sc, 0, op)

    def phase_2(self, sc, u, op):
        e = self.meta["example"]
        s = e["tested_signal"]
        self._two(sc, ((0.09, 0.56), (0.40, 0.82)), s, op, zoom=smoothstep(0.04, 0.24, u))
        x0 = 0.61
        self.rows(sc, x0, 0.335, [
            ("test s = %g with n = %d counted" % (s, self.n_obs), None, True),
            ("CLs+b: the red tail, P(N ≤ %d | %g)" % (self.n_obs, s + self.b),
             dark(RED, 0.25)),
            ("= %.5f" % e["cls_b"], dark(RED, 0.25)),
            ("CLb: the blue tail, P(N ≤ %d | %g)" % (self.n_obs, self.b), dark(BLUE, 0.25)),
            ("= %.4f" % e["cl_b"], dark(BLUE, 0.25)),
            ("CLs = CLs+b / CLb", dark(GREEN, 0.35)),
            ("= %.5f / %.4f = %.3f" % (e["cls_b"], e["cl_b"], e["cls"]), dark(GREEN, 0.35)),
            "",
            "plain: %.4f < 0.05, excluded" % e["cls_b"],
            ("CLs: %.3f > 0.05, not excluded" % e["cls"], None, True),
        ], op, u, step=0.0495, starts=(0.02, 0.1, 0.16, 0.3, 0.36, 0.5, 0.56, 0, 0.72, 0.82))
        self.tag(sc, 1, op)

    def phase_3(self, sc, u, op):
        e = self.meta["example"]
        ks = self.counts[:23]
        ch = LossChart(((0.10, 0.90), (0.38, 0.66)), (-0.6, 22.6), (-8.5, 22.0))
        self.band(sc, ch, [-0.6, 22.6], [-8.5, -8.5], [0, 0], soft(RED, 0.78), op, strips=1)
        self.axes(sc, ch, op, "", "95% upper limit on s", xticks=[],
                  yticks=[(v, "%d" % v) for v in (0, 10, 20)])
        go = ramp(u, 0.05, 0.5)
        self.trace(sc, ch, ks, self.plain[:23], RED, op, go, width=2.6, r=3.6)
        self.trace(sc, ch, ks, self.cls[:23], GREEN, op, ramp(u, 0.3, 0.75), width=2.6, r=3.6)
        self.key(sc, 0.14, 0.415, [("plain limit: negative for n < %d" % self.first_ok, RED),
                                   ("CLs limit", GREEN)], op)
        on = op * smoothstep(0.78, 0.86, u)
        fx = ch.fx(self.n_obs)
        self.circle(sc, fx, ch.fy(e["plain_limit"]), DIGIT_INK, on)
        self.circle(sc, fx, ch.fy(e["cls_limit"]), DIGIT_INK, on)
        self.text(sc, 0.70, 0.545, "n = %d:  plain %s,  CLs %.2f"
                  % (self.n_obs, signed(e["plain_limit"]), e["cls_limit"]), on, size=BODY,
                  strong=True)
        lo = LossChart(((0.10, 0.90), (0.74, 0.86)), (-0.6, 22.6), (0.0, 0.14))
        self.axes(sc, lo, op, "", "", xticks=[(k, "%d" % k) for k in range(0, 23, 2)],
                  yticks=[(0.0, "0"), (0.1, "10%")], grid=False)
        self.bars(sc, lo, ks, self.n_hist[:23], 0.7, BLUE, op)
        self.text(sc, 0.5, 0.705, "how often a background-only experiment counts n", op,
                  size=BODY)
        self.tag(sc, 2, op)

    def phase_4(self, sc, u, op):
        x = self.meta["excludes"]
        step = min(int(u * 3 / 0.75), 2)
        cut = int(np.flatnonzero(self.plain < x["signals"][step]).max())
        ch = LossChart(((0.09, 0.50), (0.44, 0.82)), (-0.6, 22.6), (0.0, 0.14))
        self.axes(sc, ch, op, "events counted, n", "share of experiments",
                  xticks=[(k, "%d" % k) for k in range(0, 23, 5)],
                  yticks=[(v, "%d%%" % round(100 * v)) for v in (0, 0.05, 0.1)])
        ks = self.counts[:23]
        self.bars(sc, ch, ks, self.n_hist[:23], 0.72, BLUE, op,
                  inks=[RED if k <= cut else soft(BLUE, 0.35) for k in ks])
        self.text(sc, 0.295, 0.315, "no signal: the count is background only", op, size=BODY,
                  strong=True)
        self.text(sc, 0.295, 0.365, "plain limit below s = %g when n ≤ %d: %s"
                  % (x["signals"][step], cut, pct(x["plain"][step])), op, size=BODY,
                  ink=dark(RED, 0.25))
        self.text(sc, 0.79, 0.315, "experiments that exclude s", op, size=BODY, strong=True)
        for i, sig in enumerate(x["signals"]):
            y = 0.40 + 0.155 * i
            on = op * smoothstep(0.25 * i, 0.25 * i + 0.08, u)
            self.text(sc, 0.62, y - 0.025, "s = %g" % sig, on, anchor="lm", size=BODY,
                      strong=True)
            self.hbar(sc, 0.68, y + 0.025, 450 * x["plain"][i], RED, on, "plain",
                      "%s (exact %s)" % (pct(x["plain"][i]), pct(x["plain_exact"][i])),
                      u=ramp(u, 0.25 * i + 0.02, 0.25 * i + 0.2), thick=15)
            self.hbar(sc, 0.68, y + 0.07, 450 * x["cls"][i], GREEN, on, "CLs",
                      "none" if x["cls"][i] == 0 else pct(x["cls"][i]), thick=15)
        self.text(sc, 0.79, 0.875, "s = 1 moves the mean from %g to %g" % (self.b, self.b + 1),
                  op * smoothstep(0.8, 0.88, u), size=BODY)
        self.tag(sc, 3, op)

    def phase_5(self, sc, u, op):
        m = self.meta
        c = m["coverage"]
        ch = LossChart(((0.10, 0.60), (0.40, 0.82)), (0.0, 15.0), (0.94, 1.0))
        self.axes(sc, ch, op, "true signal s", "share of limits above the truth",
                  xticks=[(v, "%d" % v) for v in (0, 5, 10, 15)],
                  yticks=[(v, "%d%%" % round(100 * v)) for v in (0.95, 0.97, 0.99)])
        self.level(sc, ch, 0.95, AMBER, op, n=22, width=2.0)
        for j, ink in enumerate((RED, GREEN)):
            go = ramp(u, 0.05 + 0.3 * j, 0.4 + 0.3 * j)
            self.trace(sc, ch, self.fine_x, self.fine[j], ink, op, go, width=2.0, dots=False)
            self.spots(sc, ch, self.cover_s, self.cover[j], ink, op, r=3.2,
                       shown=int(go * len(self.cover_s) + 1e-9))
        self.rows(sc, 0.65, 0.335, [
            ("true signal s on b = %g:" % self.b, None, True),
            ("how often does the limit hold?", None, True),
            ("plain: %s at its lowest," % pct(c["plain_lowest"]), dark(RED, 0.25)),
            ("as promised", dark(RED, 0.25)),
            ("CLs: never under %s," % pct(c["cls_lowest"]), dark(GREEN, 0.35)),
            ("%s for s = %g" % (pct(c["cls_at_smallest"], 0), c["smallest_signal"]),
             dark(GREEN, 0.35)),
            "",
            "typical limit with no signal:",
            "plain %.2f, CLs %.2f" % (m["median_limit"]["plain"], m["median_limit"]["cls"]),
        ], op, u, starts=(0.02, 0.02, 0.2, 0.2, 0.5, 0.5, 0, 0.8, 0.8))
        self.tag(sc, 4, op)


# ====================================================================== 148: the bands
class BrazilBand(Discovery):
    """One mass point, one background-only pseudo-experiment at a time, each giving an
    upper limit; five thousand of them and their median, central 68% and 95%; the same
    at every mass as the green and yellow bands; an observed limit with an injected
    signal; and how often background alone takes an observed point out of the band."""

    TITLE = "The bands of a limit plot"
    PHASES = (("one mass", 11.0), ("many", 11.0), ("all masses", 10.0), ("observed", 11.0),
              ("chance", 12.0))
    TAGS = (PSEUDO + ": background only, seed %d" % S.BAND_SEEDS[0],
            PSEUDO + ": %s, background only" % commas(S.N_BAND),
            PSEUDO + ": %s per mass; ticks: exact" % commas(S.N_BAND),
            PSEUDO + ": one dataset, seed %d" % S.BAND_SEEDS[1],
            PSEUDO + ": %s background-only plots" % commas(S.N_EXITS))
    TOYS = 9

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p = s.panels
        self.mass = np.asarray(p[0].data["x"], float)
        self.bands = np.asarray(p[0].data["ys"], float)
        self.exact = np.asarray(p[0].data["exact"], float)
        self.obs = np.asarray(p[1].data["ys"], float)[0]
        self.n_at = np.asarray(p[2].data["values"], float)
        self.lim_at = np.asarray(p[2].data["limits"], float)
        self.first = np.asarray(p[2].data["first"], int)
        self.out_count = np.asarray(p[3].data["values"], float)
        d = S.band_data()
        self.lines, self.lines_out = d["lines"], d["lines_out"].astype(bool)
        self.at = S.BAND_AT
        self.q = self.bands[:, self.at]
        self.b, self.ref = m["followed"]["background"], m["followed"]["signal"]
        self.inj = S.BAND_INJECT
        self.excluded = int((self.obs < 1.0).sum())

    def subtitle_for(self, k):
        m, q = self.meta, self.q
        lv = m["leaving"]
        return [
            "what limit should an experiment expect if there is no signal? Take one mass: "
            "draw a background-only count, work out its 95% upper limit on the signal "
            "strength μ, and do it again",
            "after %s pseudo-experiments the limits at m = %g pile up around a median of "
            "%.2f. The central 68%% of them, %.2f to %.2f, is the green band; the central "
            "95%%, %.2f to %.2f, the yellow"
            % (commas(S.N_BAND), m["followed"]["mass"], q[2], q[1], q[3], q[0], q[4]),
            "repeat at each of the %d mass points and join them up: the dashed median is "
            "the expected limit, and the two bands show how far background fluctuations "
            "alone move a limit" % len(self.mass),
            "now an observed limit, from one pseudo-dataset with a signal injected at "
            "m = %g. It stays inside the bands where there is nothing and rises to %.2f, "
            "against an expected %.2f, where the excess sits"
            % (m["injected"]["mass"], self.obs[self.inj], self.bands[2, self.inj]),
            "leaving the yellow band is not rare. Over %s background-only plots a given "
            "point is outside it %s of the time, and at least one of the %d points is "
            "outside in %s of plots"
            % (commas(lv["plots"]), pct(lv["per_point"]), len(self.mass),
               pct(lv["any_point"], 0)),
        ][k]

    def phase_1(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.5, 0.315, "one mass point, m = %g: background b = %.1f expected; "
                  "the model predicts %.1f signal events" % (m["followed"]["mass"], self.b,
                                                             self.ref), op, size=BODY,
                  strong=True)
        self.text(sc, 0.5, 0.36, "μ: the signal strength, a signal as a multiple of that "
                  "prediction (μ = 1 is the model)", op, size=BODY)
        h = ramp(u, 0.05, 0.9) * self.TOYS
        i = int(min(h, self.TOYS - 1))
        f = h - i if h < self.TOYS else 1.0
        n = int(self.first[i])
        ch = LossChart(((0.09, 0.50), (0.45, 0.82)), (19.4, 68.6), (0.0, 0.07))
        ks = np.arange(20, 69)
        self.axes(sc, ch, op, "events counted, n", "probability with no signal",
                  xticks=[(k, "%d" % k) for k in range(20, 69, 10)],
                  yticks=[(v, "%d%%" % round(100 * v)) for v in (0, 0.02, 0.04, 0.06)])
        self.bars(sc, ch, ks, S.poisson_pmf(ks, self.b), 0.7, BLUE, op,
                  inks=[AMBER if k == n else soft(BLUE, 0.35) for k in ks])
        x0 = 0.60
        mu = float(self.lim_at[n])
        self.rows(sc, x0, 0.44, [
            ("pseudo-experiment %d" % (i + 1), None, True),
            ("counts n = %d" % n, dark(AMBER, 0.45)),
            "95%% limit on the signal: s < %.1f" % (mu * self.ref),
            ("μ < %.1f / %.1f = %.2f" % (mu * self.ref, self.ref, mu), None, True),
        ], op, f, starts=(0.0, 0.05, 0.3, 0.5), step=0.052)
        def at(v):
            return x0 + (0.93 - x0) * float(v) / 2.4
        self.polyline(sc, [self.P(x0, 0.79), self.P(0.93, 0.79)], 1.2, AXIS, 0.8 * op)
        for v in (0.0, 0.5, 1.0, 1.5, 2.0):
            self.polyline(sc, [self.P(at(v), 0.79), self.P(at(v), 0.802)], 1.2, AXIS, 0.8 * op)
            self.text(sc, at(v), 0.826, "%g" % v, op, size=BODY)
        self.text(sc, 0.765, 0.872, "the upper limits on μ so far", op, size=BODY)
        done = i + (1 if f > 0.7 else 0)
        self.dots(sc, self.lim_at[self.first[:done]], at, 0.768, GREEN, op, r=4.2)
        self.tag(sc, 0, op)

    def _ink(self, v):
        q = self.q
        return GREEN if q[1] <= v <= q[3] else AMBER if q[0] <= v <= q[4] else DIM

    def phase_2(self, sc, u, op):
        m, q = self.meta, self.q
        keep = np.flatnonzero((self.n_at > 0) & (self.lim_at <= 2.4))
        ch = LossChart(((0.10, 0.62), (0.38, 0.82)), (0.0, 2.4), (0.0, 340.0))
        self.axes(sc, ch, op, "upper limit on μ", "pseudo-experiments",
                  xticks=[(v, "%g" % v) for v in (0, 0.5, 1.0, 1.5, 2.0)],
                  yticks=[(v, "%d" % v) for v in (0, 100, 200, 300)])
        col = op * smoothstep(0.42, 0.55, u)
        self.bars(sc, ch, self.lim_at[keep], self.n_at[keep], 0.036, BLUE, op * (1 - 0.999 * (
            col / max(op, 1e-6))), grow=smoothstep(0.02, 0.38, u))
        self.bars(sc, ch, self.lim_at[keep], self.n_at[keep], 0.036, GREEN, col,
                  inks=[self._ink(v) for v in self.lim_at[keep]], lift=0.5)
        self.post(sc, ch, q[2], DIGIT_INK, col, dashed=True, width=2.2)
        self.rows(sc, 0.66, 0.335, [
            ("%s pseudo-experiments" % commas(S.N_BAND), None, True),
            ("at m = %g, background only" % m["followed"]["mass"], None, True),
            "one bar per count n:",
            "limits come in steps",
            "",
            ("median %.2f:" % q[2], None, True),
            ("the expected limit", None, True),
            ("central 68%%: %.2f to %.2f" % (q[1], q[3]), dark(GREEN, 0.35)),
            ("central 95%%: %.2f to %.2f" % (q[0], q[4]), dark(AMBER, 0.45)),
        ], op, u, starts=(0.02, 0.02, 0.1, 0.1, 0, 0.5, 0.5, 0.64, 0.76))
        self.tag(sc, 1, op)

    def _plot(self, sc, box, op, go=1.0, exact=False, model=True):
        ch = LossChart(box, (106.0, 169.0), (0.0, 2.8))
        self.band(sc, ch, self.mass, self.bands[0], self.bands[4], soft(AMBER, 0.25), op, go)
        self.band(sc, ch, self.mass, self.bands[1], self.bands[3], soft(GREEN, 0.2), op, go,
                  lift=0.3)
        self.axes(sc, ch, op, "mass m", "95% upper limit on μ",
                  xticks=[(v, "%d" % v) for v in range(110, 166, 10)],
                  yticks=[(v, "%g" % v) for v in (0, 1, 2)], grid=False)
        end = self.mass[0] + (self.mass[-1] - self.mass[0]) * min(go, 1.0)
        for a, b in zip(range(len(self.mass) - 1), range(1, len(self.mass))):
            if self.mass[b] <= end + 1e-9:
                self.dash(sc, (ch.fx(self.mass[a]), ch.fy(self.bands[2, a])),
                          (ch.fx(self.mass[b]), ch.fy(self.bands[2, b])), DIGIT_INK, op, 3, 2.0)
        if exact:
            dx = self.frac(5.0)[0]
            for i, x in enumerate(self.mass):
                if x > end + 1e-9:
                    break
                for row in self.exact:
                    y = ch.fy(row[i])
                    self.polyline(sc, [self.P(ch.fx(x) - dx, y, lift=1.0),
                                       self.P(ch.fx(x) + dx, y, lift=1.0)], 1.8, DIGIT_INK,
                                  0.85 * op)
        if model:
            self.level(sc, ch, 1.0, RED, 0.9 * op, n=30, width=1.6)
        return ch

    def phase_3(self, sc, u, op):
        go = ramp(u, 0.05, 0.75)
        self._plot(sc, ((0.10, 0.90), (0.40, 0.82)), op, go, exact=True, model=False)
        self.key(sc, 0.12, 0.33, [("central 95% of the limits", soft(AMBER, 0.25))], op)
        self.key(sc, 0.40, 0.33, [("central 68%", soft(GREEN, 0.2))], op)
        self.key(sc, 0.58, 0.33, [("dashed: median;  ticks: exact quantiles", DIGIT_INK)], op)
        self.tag(sc, 2, op)

    def phase_4(self, sc, u, op):
        m = self.meta
        ch = self._plot(sc, ((0.10, 0.90), (0.40, 0.82)), op)
        self.trace(sc, ch, self.mass, self.obs, DIGIT_INK, op, ramp(u, 0.06, 0.6), width=2.8,
                   r=4.4)
        self.key(sc, 0.12, 0.33, [("observed limit (pseudo-data)", DIGIT_INK)], op)
        self.text(sc, 0.905, ch.fy(1.0), "μ = 1", op, anchor="lm", size=BODY,
                  ink=dark(RED, 0.25))
        on = op * smoothstep(0.62, 0.7, u)
        self.circle(sc, ch.fx(self.mass[self.inj]), ch.fy(self.obs[self.inj]), RED, on, 24)
        self.text(sc, 0.88, 0.33, "signal injected at m = %g: limit %.2f, expected %.2f"
                  % (m["injected"]["mass"], self.obs[self.inj], self.bands[2, self.inj]), on,
                  anchor="rm", size=BODY, ink=dark(RED, 0.25))
        self.text(sc, 0.06, 0.925, "under the red line the model is excluded: %d of %d masses"
                  % (self.excluded, len(self.mass)), op * smoothstep(0.82, 0.9, u), size=BODY,
                  anchor="lm")
        self.tag(sc, 3, op)

    def phase_5(self, sc, u, op):
        lv = self.meta["leaving"]
        ch = self._plot(sc, ((0.09, 0.58), (0.40, 0.82)), op, model=False)
        n = len(self.lines)
        i = int(min(ramp(u, 0.02, 0.8) * n, n - 1))
        self.trace(sc, ch, self.mass, self.lines[i], DIGIT_INK, op, 1.0, width=2.6, r=4.0)
        for j in np.flatnonzero(self.lines_out[i]):
            self.circle(sc, ch.fx(self.mass[j]), ch.fy(min(self.lines[i][j], 2.75)), RED, op, 22)
        self.text(sc, 0.335, 0.33, "background-only plot %d of %s: %d outside the yellow"
                  % (i + 1, commas(lv["plots"]), int(self.lines_out[i].sum())), op, size=BODY,
                  strong=True)
        x0 = 0.635
        self.rows(sc, x0, 0.39, [
            ("a point outside the yellow band:", None, True),
            "%s of the time (exact %s)" % (pct(lv["per_point"]), pct(lv["per_point_exact"])),
            "not 5%: counts come in steps",
            "",
            ("points outside, of %d:" % len(self.mass), None, True),
        ], op, u, starts=(0.1, 0.2, 0.3, 0, 0.45), step=0.048)
        shares = (self.out_count[0], self.out_count[1], self.out_count[2:].sum())
        for r, (name, v) in enumerate(zip(("none", "one", "two +"), shares)):
            self.hbar(sc, x0 + 0.075, 0.645 + 0.052 * r, 210 * v, AMBER if r else GREEN,
                      op * smoothstep(0.5, 0.58, u), name, pct(v, 0),
                      u=ramp(u, 0.52 + 0.08 * r, 0.7 + 0.08 * r), thick=16)
        self.text(sc, x0, 0.835, "at least one: %s of plots" % pct(lv["any_point"], 0),
                  op * smoothstep(0.84, 0.9, u), anchor="lm", size=BODY, strong=True)
        self.tag(sc, 4, op)


# ====================================================================== 149: five sigma
class WhyFiveSigma(Discovery):
    """Two spectra filling as luminosity accumulates, one with a signal and one
    without; their significance against luminosity; every bin and every look of a
    background-only search and how often 3 and 5 sigma are reached; the two tail
    probabilities; and the luminosity a signal needs for 5 sigma."""

    TITLE = "Why five sigma"
    PHASES = (("two spectra", 11.0), ("Z grows", 12.0), ("every look", 12.0),
              ("two tails", 10.0), ("how much data", 12.0))
    TAGS = (PSEUDO + ": two spectra, seed %d" % S.FIVE_SEEDS[0],
            PSEUDO + ": two spectra, seed %d" % S.FIVE_SEEDS[0],
            PSEUDO + ": %s background-only searches" % commas(S.N_SEARCH),
            EXACT + ": Gaussian tail probabilities",
            PSEUDO + ": %s per signal size" % commas(S.N_SIGNAL))
    S_INKS = (OCHRE, RED, MAGENTA)

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p = s.panels
        d = S.five_data()
        self.looks = np.asarray(p[0].data["x"], float)
        self.z_sig, self.z_exp, self.z_max = (np.asarray(v, float) for v in p[0].data["ys"])
        self.z_bins = np.asarray(p[0].data["all_bins"], float)
        self.n_none, self.n_with = d["none"], d["with_"]
        self.z_with = d["z_with"]
        self.share3 = np.asarray(p[1].data["values"], float)
        self.five = [int(v) for v in p[1].data["five"]]
        self.of = [int(v) for v in p[1].data["of"]]
        self.tails = [float(v) for v in p[2].data["values"]]
        self.long = np.asarray(p[3].data["x"], float)
        self.median = np.asarray(p[3].data["ys"], float)
        self.lumi5 = [float(v) for v in p[3].data["lumi5"]]
        self.formula = [float(v) for v in p[3].data["formula"]]
        self.b, self.s = m["model"]["background_rate"], m["model"]["signal_rate"]
        self.sig = int(m["model"]["signal_bin"])
        self.zs = np.linspace(0.0, 6.0, 121)
        self.tail_curve = S.norm_sf(self.zs)

    def subtitle_for(self, k):
        m = self.meta
        f, r = m["followed"], m["reach"]
        return [
            "one counting analysis, two spectra, as data accumulate: one has a real signal "
            "in bin %d, the other none. Signal events pile up in step with the luminosity "
            "L; the background's noise grows only as √L" % (self.sig + 1),
            "so the real signal's significance climbs roughly as √L, to %.1f sigma at "
            "L = %g here. Background-only bins wander: the largest reaches %.1f sigma at "
            "L = %g and has faded to %.1f by the end"
            % (self.z_sig[-1], self.looks[-1], f["background_peak"], f["background_peak_at"],
               f["background_last"]),
            "%d bins looked at %d times give chance many openings. Of %s background-only "
            "searches, %s show a 3-sigma excess somewhere at some time; one bin at one "
            "look does so in %s"
            % (m["model"]["bins"], m["model"]["looks"], commas(self.of[2]),
               pct(r["three_share"][2]), pct(r["three_share"][0], 2)),
            "three sigma is a one-sided tail probability of %s, five sigma of %s. Times "
            "the places and moments one looks, 3 sigma is routine; 5 sigma turned up in "
            "%d of %s searches" % (sci(self.tails[0]), sci(self.tails[1]), self.five[2],
                                   commas(self.of[2])),
            "and a real signal gets there, given data. The median significance over %s "
            "pseudo-experiments reaches 5 sigma at L = %g for this signal; half the signal "
            "needs four times as much: %g"
            % (commas(S.N_SIGNAL), self.lumi5[1], self.lumi5[0]),
        ][k]

    def _excess(self, sc, box, counts, k, hot, ink, op):
        ch = LossChart(box, (0.4, S.N_BINS + 0.6), (-110.0, 190.0))
        b = self.b * self.looks[k]
        sd = float(np.sqrt(b))
        self.band(sc, ch, [0.4, S.N_BINS + 0.6], [-sd, -sd], [sd, sd], soft(DIM, 0.1), op,
                  strips=1)
        self.axes(sc, ch, op, "bin of the spectrum", "n − b" if ink is RED else "",
                  xticks=[(v, "%d" % v) for v in (1, 5, 10, 15, 20)],
                  yticks=[(v, "%d" % v) for v in (-100, 0, 100)], grid=False)
        bins = np.arange(1, S.N_BINS + 1)
        self.bars(sc, ch, bins, counts[k] - b, 0.7, ink, op, base=0.0,
                  inks=[ink if j == hot else soft(ink, 0.5) for j in range(S.N_BINS)])
        return ch

    def phase_1(self, sc, u, op):
        k = int(ramp(u, 0.04, 0.9) * (len(self.looks) - 1) + 1e-9)
        lum, b = self.looks[k], self.b * self.looks[k]
        self.text(sc, 0.5, 0.315, "luminosity L = %d fb%s: the background, known, is b = %g × "
                  "L = %d events in every bin" % (lum, INV, self.b, b), op, size=BODY,
                  strong=True)
        top = int(np.argmax(self.z_bins[:, k]))
        self._excess(sc, ((0.09, 0.47), (0.46, 0.76)), self.n_with, k, self.sig, RED, op)
        self._excess(sc, ((0.58, 0.95), (0.46, 0.76)), self.n_none, k, top, BLUE, op)
        self.text(sc, 0.47, 0.375, "a signal of %g events per fb%s in bin %d"
                  % (self.s, INV, self.sig + 1), op, anchor="rm", size=BODY,
                  ink=dark(RED, 0.25))
        self.text(sc, 0.95, 0.375, "background only", op, anchor="rm", size=BODY,
                  ink=dark(BLUE, 0.25))
        self.text(sc, 0.28, 0.865, "bin %d: n = %d, Z = %.1f sigma"
                  % (self.sig + 1, self.n_with[k, self.sig], self.z_with[k, self.sig]), op,
                  size=BODY, strong=True)
        self.text(sc, 0.765, 0.865, "largest, bin %d: n = %d, Z = %.1f"
                  % (top + 1, self.n_none[k, top], self.z_bins[top, k]), op, size=BODY,
                  strong=True)
        self.text(sc, 0.06, 0.925, "n − b: events above background;  grey: ± √b", op,
                  anchor="lm",
                  size=BODY)
        self.tag(sc, 0, op)

    def phase_2(self, sc, u, op):
        ch = LossChart(((0.10, 0.88), (0.42, 0.82)), (0.0, 30.0), (-2.6, 6.4))
        self.axes(sc, ch, op, "luminosity L, fb%s" % INV, "significance Z, in sigma",
                  xticks=[(v, "%d" % v) for v in (0, 10, 20, 30)],
                  yticks=[(v, "%d" % v) for v in (-2, 0, 2, 4, 6)])
        for z, ink in ((3.0, AMBER), (5.0, VIOLET)):
            self.level(sc, ch, z, ink, op, n=30, width=1.8)
            self.text(sc, ch.x1 + 0.012, ch.fy(z), "%dσ" % z, op, anchor="lm", size=BODY,
                      ink=dark(ink, 0.4))
        go = ramp(u, 0.04, 0.8)
        for row in self.z_bins:
            self.trace(sc, ch, self.looks, row, soft(BLUE, 0.45), 0.45 * op, go, width=1.3,
                       dots=False)
        self.trace(sc, ch, self.looks, self.z_max, BLUE, op, go, width=3.0, dots=False)
        self.trace(sc, ch, self.looks, self.z_exp, soft(RED, 0.4), op, 1.0, width=1.8,
                   dots=False)
        self.trace(sc, ch, self.looks, self.z_sig, RED, op, go, width=3.2, dots=False)
        self.key(sc, 0.12, 0.33, [("the signal's bin", RED)], op)
        self.key(sc, 0.30, 0.33, [("expected: grows as √L", soft(RED, 0.4))], op)
        self.key(sc, 0.56, 0.33, [("background only: 20 bins, the largest bold", BLUE)], op)
        self.tag(sc, 1, op)

    def phase_3(self, sc, u, op):
        self.text(sc, 0.275, 0.315, "one background-only search: Z of each bin", op,
                  size=BODY, strong=True)
        self.text(sc, 0.275, 0.358, "(a row) at each look (a column)", op, size=BODY,
                  strong=True)
        k = int(ramp(u, 0.04, 0.6) * (len(self.looks) - 1) + 1e-9)
        shown = np.where(np.arange(len(self.looks))[None, :] <= k, self.z_bins, 0.0)
        hot = [(int(a), int(c)) for a, c in zip(*np.nonzero(shown >= 2.0))]
        self.grid(sc, shown, 0.275, 0.615, 345, 196, op, vmax=3.0, ring=hot, ring_ink=DIGIT_INK)
        self.text(sc, 0.275, 0.87, "red: above background; ringed: 2 sigma or more",
                  op, size=BODY)
        x0 = 0.735
        self.text(sc, 0.56, 0.315, "background-only searches that reach 3 sigma", op,
                  anchor="lm", size=BODY, strong=True)
        for i, name in enumerate(S.FIVE_LEVELS):
            v = self.share3[i]
            self.hbar(sc, x0, 0.375 + 0.055 * i, 105 * v, AMBER,
                      op * smoothstep(0.1 + 0.12 * i, 0.16 + 0.12 * i, u), name,
                      pct(v, 2 if v < 0.02 else 0 if v > 0.5 else 1),
                      u=ramp(u, 0.12 + 0.12 * i, 0.3 + 0.12 * i), thick=16)
        on = op * smoothstep(0.7, 0.78, u)
        self.text(sc, 0.56, 0.635, "and 5 sigma", on, anchor="lm", size=BODY, strong=True)
        for i, name in enumerate(S.FIVE_LEVELS):
            self.text(sc, x0 - 0.01, 0.69 + 0.048 * i, name, on, anchor="rm", size=BODY)
            self.text(sc, x0 + 0.008, 0.69 + 0.048 * i, "%d of %s" % (self.five[i],
                                                                      commas(self.of[i])),
                      on, anchor="lm", size=BODY, ink=dark(VIOLET, 0.25))
        self.tag(sc, 2, op)

    def phase_4(self, sc, u, op):
        self.text(sc, 0.5, 0.315, "the chance that background alone fluctuates up by Z "
                  "sigma or more: one place, one look", op, size=BODY, strong=True)
        zs = np.linspace(-3.0, 6.0, 91)
        pdf = np.exp(-0.5 * zs * zs) / np.sqrt(2 * np.pi)
        ch = LossChart(((0.08, 0.47), (0.39, 0.70)), (-3.0, 6.0), (0.0, 0.42))
        self.axes(sc, ch, op, "fluctuation, in sigma", "",
                  xticks=[(v, "%d" % v) for v in (-2, 0, 2, 3, 5)], yticks=[], grid=False)
        fill = np.linspace(3.0, 6.0, 25)
        self.band(sc, ch, fill, 0 * fill, np.exp(-0.5 * fill * fill) / np.sqrt(2 * np.pi),
                  AMBER, op * smoothstep(0.2, 0.3, u), strips=24)
        self.trace(sc, ch, zs, pdf, DIGIT_INK, op, ramp(u, 0.02, 0.2), width=2.4, dots=False)
        self.post(sc, ch, 3.0, AMBER, op * smoothstep(0.2, 0.3, u), top=0.16, width=2.0)
        self.post(sc, ch, 5.0, VIOLET, op * smoothstep(0.45, 0.55, u), top=0.16, width=2.0)
        lg = LossChart(((0.60, 0.93), (0.39, 0.70)), (0.0, 6.0), (1e-9, 1.0), ylog=True)
        self.axes(sc, lg, op, "Z", "tail probability p",
                  xticks=[(v, "%d" % v) for v in range(7)],
                  yticks=[(10.0 ** e, pow10(10.0 ** e)) for e in (-9, -6, -3, 0)])
        self.trace(sc, lg, self.zs, self.tail_curve, DIGIT_INK, op, ramp(u, 0.1, 0.5),
                   width=2.4, dots=False)
        for z, p, ink, t in ((3.0, self.tails[0], AMBER, 0.3), (5.0, self.tails[1], VIOLET,
                                                                0.52)):
            on = op * smoothstep(t, t + 0.08, u)
            self.disk(sc, self.P(lg.fx(z), lg.fy(p)), 6.0, ink, on, lift=1.5, outline=0.0)
            self.dash(sc, (lg.x0, lg.fy(p)), (lg.fx(z), lg.fy(p)), ink, 0.45 * on, 10, 1.4)
        self.text(sc, 0.275, 0.825, "beyond 3 sigma: %s, 1 in %s"
                  % (sci(self.tails[0]), commas(1 / self.tails[0])),
                  op * smoothstep(0.3, 0.38, u), size=BODY, ink=dark(AMBER, 0.45))
        self.text(sc, 0.765, 0.825, "beyond 5 sigma: %s, 1 in %.1f million"
                  % (sci(self.tails[1]), 1e-6 / self.tails[1]),
                  op * smoothstep(0.52, 0.6, u), size=BODY, ink=dark(VIOLET, 0.25))
        self.text(sc, 0.5, 0.875, "%d bins × %d looks: 3 sigma in %s of searches, 5 sigma "
                  "in %d of %s" % (S.N_BINS, len(self.looks), pct(self.share3[2], 0),
                                   self.five[2], commas(self.of[2])),
                  op * smoothstep(0.75, 0.83, u), size=BODY, strong=True)
        self.tag(sc, 3, op)

    def phase_5(self, sc, u, op):
        ch = LossChart(((0.10, 0.62), (0.39, 0.82)), (0.0, 130.0), (0.0, 8.0))
        self.axes(sc, ch, op, "luminosity L, fb%s" % INV, "median Z, in sigma",
                  xticks=[(v, "%d" % v) for v in (0, 25, 50, 75, 100, 125)],
                  yticks=[(v, "%d" % v) for v in (0, 2, 4, 6, 8)])
        self.level(sc, ch, 5.0, VIOLET, op, n=26, width=1.8)
        self.text(sc, ch.x1 + 0.012, ch.fy(5.0), "5σ", op, anchor="lm", size=BODY,
                  ink=dark(VIOLET, 0.25))
        x0 = 0.70
        self.text(sc, x0, 0.335, "signal events per fb%s," % INV, op, anchor="lm", size=BODY,
                  strong=True)
        self.text(sc, x0, 0.38, "on a background of %g:" % self.b, op, anchor="lm", size=BODY,
                  strong=True)
        for j in (2, 1, 0):
            ink = self.S_INKS[j]
            t0 = 0.05 + 0.22 * (2 - j)
            go = ramp(u, t0, t0 + 0.24)
            keep = self.median[j] <= 8.0
            self.trace(sc, ch, self.long[keep], self.median[j][keep], ink, op,
                       go * 130.0 / self.long[keep][-1] if keep.sum() < len(keep) else go,
                       width=2.8, dots=False)
            on = op * smoothstep(t0 + 0.2, t0 + 0.27, u)
            self.post(sc, ch, self.lumi5[j], ink, 0.45 * on, width=1.6, top=5.0)
            self.disk(sc, self.P(ch.fx(self.lumi5[j]), ch.fy(5.0)), 5.5, ink, on, lift=1.5,
                      outline=0.0)
            y = 0.455 + 0.11 * (2 - j)
            self.key(sc, x0 + 0.008, y, [("s = %g:  5 sigma at L = %g"
                                          % (S.S_RATES[j], self.lumi5[j]), ink)], on, size=BODY)
            self.text(sc, x0, y + 0.044, "(expected Z does at %.1f)" % self.formula[j], on,
                      anchor="lm", size=BODY)
        self.text(sc, x0, 0.80, "half the signal:", op * smoothstep(0.82, 0.9, u),
                  anchor="lm", size=BODY, strong=True)
        self.text(sc, x0, 0.845, "four times the data", op * smoothstep(0.82, 0.9, u),
                  anchor="lm", size=BODY, strong=True)
        self.tag(sc, 4, op)


ANIMATIONS = {
    "bump-hunt": BumpHunt,
    "poisson-upper-limits": PoissonLimits,
    "cls-method": ClsMethod,
    "brazil-band": BrazilBand,
    "why-five-sigma": WhyFiveSigma,
}

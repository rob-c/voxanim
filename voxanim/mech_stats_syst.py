"""Correlated errors and systematic uncertainties: animations 133-136 of the "next"
catalogue (see :mod:`voxanim.mech`).

* :class:`ErrorEllipse`: a straight line fitted again and again; the tilted cloud of
  (intercept, slope), its covariance matrix, and one fit's chi-square + 1 contour;
* :class:`StatVsSyst`: the mean of N events from a detector whose scale is off: the
  error that falls as 1/sqrt(N), the one that does not, and a control sample;
* :class:`Profiling`: a count with an uncertain background: the likelihood valley, and
  the interval with the background fixed, profiled and marginalised;
* :class:`Combining`: two measurements with a shared systematic: weights 1/sigma^2, the
  best linear combination, and a negative weight.

They draw the experiments of :mod:`voxanim.study_stats_syst`. Each phase says, bottom
right, what kind of evidence is on screen: a formula evaluated (:data:`EXACT`), seeded
draws from the stated model (:data:`TOYS`), or both.
"""

from __future__ import annotations

import numpy as np

from . import study_stats_syst as S
from .diagram import draw_image
from .easing import remap, smoothstep
from .losschart import LossChart
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIGIT_INK,
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

EXACT = "exact"
TOYS = "pseudo-experiments"
BOTH = "exact + pseudo-experiments"
KINDS = (EXACT, TOYS, BOTH)
BIG, MID = 0.036, 0.032
GREY = (0.50, 0.52, 0.54)


def num(v, digits=2):
    """A number with a real minus sign."""
    return ("%.*f" % (digits, float(v))).replace("-", "−")


def signed(v, digits=2):
    return ("+" if v >= 0 else "") + num(v, digits)


def pct(v, digits=1):
    return "%.*f%%" % (digits, 100.0 * float(v))


def commas(n):
    return format(int(n), ",")


def ellipse(cov, centre, n=64):
    """Points of the one-sigma ellipse of the covariance matrix ``cov``."""
    t = np.linspace(0.0, 2 * np.pi, n + 1)
    return np.asarray(centre) + np.stack([np.cos(t), np.sin(t)], 1) @ np.linalg.cholesky(cov).T


class Syst(Mechanism):
    """What the four animations share: the evidence tag, curves and points on a chart,
    error bars, and a number line."""

    def tag(self, sc, kind, op, detail=""):
        """The kind of evidence on screen, bottom right."""
        self.text(sc, 0.95, 0.925, kind + (": " + detail if detail else ""), op, anchor="rm",
                  ink=dark(TEAL, 0.25))

    def curve(self, sc, chart, xs, ys, ink, op, go=1.0, width=2.6, lift=0.4):
        """``ys`` against ``xs`` on ``chart`` as far as fraction ``go`` of its points,
        broken where a value is not finite. Returns the layout point it ends at."""
        if op <= 0.01 or go <= 0:
            return None
        k = max(2, int(round(1 + (len(xs) - 1) * min(float(go), 1.0))))
        run, end = [], None
        for x, y in zip(xs[:k], ys[:k]):
            if np.isfinite(y):
                end = (chart.fx(float(x)), chart.fy(float(y)))
                run.append(self.P(end[0], end[1], lift=lift))
            else:
                if len(run) > 1:
                    self.polyline(sc, run, width, ink, op, bias=0.3)
                run = []
        if len(run) > 1:
            self.polyline(sc, run, width, ink, op, bias=0.3)
        return end

    def seg(self, sc, a, b, ink, op, width=2.2, lift=0.5):
        """A straight line between two layout points."""
        if op > 0.01:
            self.polyline(sc, [self.P(a[0], a[1], lift=lift), self.P(b[0], b[1], lift=lift)],
                          width, ink, op, bias=0.3)

    def dash(self, sc, a, b, ink, op, n=14, width=1.6):
        """A dashed line between two layout points."""
        (ax, ay), (bx, by) = a, b
        for i in range(n):
            u0, u1 = i / n, (i + 0.55) / n
            self.seg(sc, (ax + (bx - ax) * u0, ay + (by - ay) * u0),
                     (ax + (bx - ax) * u1, ay + (by - ay) * u1), ink, op, width, 0.3)

    def spot(self, sc, fx, fy, r, ink, op, lift=1.2, glow=0.0):
        self.disk(sc, self.P(float(fx), float(fy)), r, ink, op, lift=lift, outline=0.0,
                  glow=glow)

    def hbar_err(self, sc, lo, hi, fy, ink, op, width=3.0, cap=6.0, dot=None, r=5.0):
        """A horizontal error bar from layout x ``lo`` to ``hi`` with end caps, and a
        point at ``dot``."""
        if op <= 0.01:
            return
        dy = self.frac(0, cap)[1]
        self.seg(sc, (lo, fy), (hi, fy), ink, op, width, 0.8)
        for x in (lo, hi):
            self.seg(sc, (x, fy - dy), (x, fy + dy), ink, op, width, 0.8)
        if dot is not None:
            self.spot(sc, dot, fy, r, ink, op, lift=1.4)

    def vbar_err(self, sc, fx, lo, hi, ink, op, width=2.0, cap=4.0):
        if op <= 0.01:
            return
        dx = self.frac(cap)[0]
        self.seg(sc, (fx, lo), (fx, hi), ink, op, width, 0.6)
        for y in (lo, hi):
            self.seg(sc, (fx - dx, y), (fx + dx, y), ink, op, width, 0.6)

    def scale(self, sc, lo, hi, x0, x1, fy, op, ticks, fmt="%g"):
        """A horizontal axis from ``lo`` to ``hi`` between ``x0`` and ``x1``; returns the
        value -> layout x map."""
        def at(v):
            return x0 + (x1 - x0) * (float(v) - lo) / (hi - lo)
        self.seg(sc, (x0, fy), (x1, fy), AXIS, 0.45 * op, 1.4, 0.2)
        for v in ticks:
            self.seg(sc, (at(v), fy), (at(v), fy + 0.012), AXIS, 0.45 * op, 1.4, 0.2)
            self.text(sc, at(v), fy + 0.036, fmt % v, op)
        return at

    def key(self, sc, fx, fy, entries, op, step=0.044):
        """A legend: a colour chip and a name per entry, stacked down from ``(fx, fy)``."""
        for n, (name, ink) in enumerate(entries):
            y = fy + n * step
            self.box(sc, fx, y, 14, 14, ink, op, radius=3)
            self.text(sc, fx + 0.016, y, name, op, anchor="lm", ink=dark(ink, 0.3))

    def rows(self, sc, fx, fy, rows, op, step=0.045, size=None, anchor="lm"):
        """Lines of text ``(string, ink, opacity factor)`` stacked down from ``(fx, fy)``."""
        for n, row in enumerate(rows):
            s, ink, on = row if len(row) == 3 else row + (1.0,)
            self.text(sc, fx, fy + n * step, s, op * on, size or MID * 0.94, anchor,
                      None if ink is None else dark(ink, 0.3), strong=ink is None)


# ====================================================================== 133: the error ellipse
class ErrorEllipse(Syst):
    """A straight line fitted to five noisy points, experiment after experiment: each
    fit one point (a, b); the tilted cloud and its covariance matrix; the chi-square + 1
    contour of a single fit as the same ellipse; the width of a with b free and with b
    fixed; and the cloud made upright by counting x from the centre of the data."""

    TITLE = "The error ellipse"
    PHASES = (("one fit", 10.0), ("many fits", 10.0), ("the ellipse", 13.0),
              ("fix one", 11.0), ("move the origin", 11.0))
    DATA = ((0.09, 0.43), (0.385, 0.79))
    PLANE = ((0.58, 0.93), (0.385, 0.79))
    SHOWN, LINES = 300, 40

    def setup(self):
        p = self.study.panels
        m = self.meta = self.study.metadata
        self.x = np.asarray(p[0].data["x"], float)
        self.y = np.asarray(p[0].data["y"], float)
        self.fits = np.asarray(p[0].data["xy"], float)
        self.ring = np.asarray(p[0].data["ring"], float)
        self.cov = np.asarray(p[1].data["values"], float)
        self.exact = np.asarray(p[1].data["exact"], float)
        self.cov_mid = np.asarray(p[1].data["centred"], float)
        self.grid = np.asarray(p[2].data["x"], float)
        self.free, self.held = (np.asarray(v, float) for v in p[2].data["ys"])
        self.free_at = np.asarray(p[2].data["free_at"], float)
        self.held_at = np.asarray(p[2].data["held_at"], float)
        self.first = self.fits[0]
        self.sa, self.sb, self.rho = m["sigma_a"], m["sigma_b"], m["rho"]
        self.half_free = (self.free_at[1] - self.free_at[0]) / 2
        self.half_held = (self.held_at[1] - self.held_at[0]) / 2
        self.dc = LossChart(self.DATA, (0.0, 6.0), (-1.0, 6.0))
        self.pc = LossChart(self.PLANE, (-1.0, 3.5), (0.0, 1.0))
        b = self.fits[:self.LINES, 1]
        self.steep, self.shallow = int(np.argmax(b)), int(np.argmin(b))
        self.truth = np.array([S.A_TRUE, S.B_TRUE])

    def subtitle_for(self, k):
        m = self.meta
        return [
            "fit a straight line y = a + bx to five noisy points and you get two numbers: "
            "the intercept a and the slope b. Repeat the experiment with fresh noise and "
            "both come out different",
            "over %s repeats the fitted pairs form a tilted cloud: a line that comes out "
            "too steep also starts too low. The errors on a and b are correlated, with "
            "coefficient ρ = %s" % (commas(m["toys"]), num(self.rho, 3)),
            "the cloud's covariance matrix is an ellipse. One fit alone draws the same "
            "one: where its χ² (the squared misses, in units of σ, summed) is 1 above its "
            "minimum. It holds the truth in %s of fits" % pct(m["inside"]),
            "the tilt widens each error. With b refitted a is known to ± %s; hold b fixed "
            "and it is ± %s, smaller by the factor √(1 − ρ²). Measured with b fixed at "
            "the truth: %s" % (num(self.half_free, 3), num(self.half_held, 3),
                               num(m["conditional"], 3)),
            "the tilt depends on where x = 0 is. Count x from the centre of the data and "
            "the same fits make an upright cloud: ρ = %s, and the line's height there is "
            "known to ± %s" % (num(m["centred"]["rho"]), num(m["centred"]["sigma"], 3)),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _data_axes(self, sc, op):
        self.axes(sc, self.dc, op, "x", "y", xticks=[(v, str(v)) for v in range(7)],
                  yticks=[(v, str(v)) for v in (0, 2, 4, 6)])

    def _plane_axes(self, sc, op, xlabel="intercept a: the height at x = 0"):
        self.axes(sc, self.pc, op, xlabel, "slope b",
                  xticks=[(v, "%d" % v) for v in (0, 1, 2, 3)],
                  yticks=[(v, "%g" % v) for v in (0, 0.5, 1.0)])

    def _line(self, sc, a, b, ink, op, go=1.0, width=2.4):
        x1 = 6.0 * min(max(go, 0.0), 1.0)
        if x1 > 0:
            self.seg(sc, (self.dc.fx(0.0), self.dc.fy(a)), (self.dc.fx(x1), self.dc.fy(a + b * x1)),
                     ink, op, width)

    def _points(self, sc, y, op):
        for xv, yv in zip(self.x, y):
            fx = self.dc.fx(xv)
            self.vbar_err(sc, fx, self.dc.fy(yv - S.SIGMA_Y), self.dc.fy(yv + S.SIGMA_Y),
                          DIGIT_INK, op)
            self.spot(sc, fx, self.dc.fy(yv), 4.5, DIGIT_INK, op)

    def _cloud(self, sc, op, count, ink=None, shear=0.0, skip=()):
        ink = soft(BLUE, 0.25) if ink is None else ink
        for i in range(count):
            if i in skip:
                continue
            a, b = self.fits[i]
            self.spot(sc, self.pc.fx(a + 3.0 * b * shear), self.pc.fy(b), 2.6, ink, 0.85 * op,
                      lift=0.8)

    def _cross(self, sc, op):
        """The true (a, b)."""
        x, y = self.pc.fx(S.A_TRUE), self.pc.fy(S.B_TRUE)
        dx, dy = self.frac(7.0, 7.0)
        self.seg(sc, (x - dx, y), (x + dx, y), DIGIT_INK, op, 2.6, 1.6)
        self.seg(sc, (x, y - dy), (x, y + dy), DIGIT_INK, op, 2.6, 1.6)

    def _closed(self, sc, pts, ink, op, go=1.0, width=2.6, shear=0.0):
        self.curve(sc, self.pc, pts[:, 0] + 3.0 * pts[:, 1] * shear, pts[:, 1], ink, op, go,
                   width, lift=1.0)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        e = min(int(u * 3), 2)
        f = u * 3 - e
        self.text(sc, 0.26, 0.305, "experiment %d of %s: five points, each ± %s"
                  % (e + 1, commas(self.meta["toys"]), num(S.SIGMA_Y, 1)), op, MID,
                  strong=True)
        self.text(sc, 0.755, 0.305, "each fit is one point: (a, b)", op, MID, strong=True)
        self._data_axes(sc, op)
        self._plane_axes(sc, op)
        self._line(sc, S.A_TRUE, S.B_TRUE, GREY, 0.45 * op, width=2.0)
        self._cross(sc, op)
        self.text(sc, self.pc.fx(S.A_TRUE) + 0.014, self.pc.fy(S.B_TRUE) + 0.036, "the truth",
                  op, anchor="lm")
        for k in range(e):                              # the experiments already fitted
            a, b = self.fits[k]
            self._line(sc, a, b, RED if k == 0 else BLUE, 0.3 * op, width=1.8)
            self.spot(sc, self.pc.fx(a), self.pc.fy(b), 5.0, RED if k == 0 else BLUE, op)
        a, b = self.fits[e]
        ink = RED if e == 0 else BLUE
        self._points(sc, self.y[e], op * smoothstep(0.0, 0.12, f))
        self._line(sc, a, b, ink, op, smoothstep(0.12, 0.4, f), 3.0)
        on = op * smoothstep(0.34, 0.44, f)
        self.spot(sc, self.dc.fx(0.0), self.dc.fy(a), 6.0, ink, on, lift=1.5)
        x0 = self.dc.x0 + 0.014
        self.text(sc, x0, 0.415, "fit:  a = %s,  b = %s" % (num(a), num(b)), on, anchor="lm",
                  ink=dark(ink, 0.3))
        self.text(sc, x0, 0.46, "truth:  a = %s,  b = %s" % (num(S.A_TRUE, 1),
                                                              num(S.B_TRUE, 1)),
                  op, anchor="lm", ink=dark(GREY, 0.2))
        start, end = (self.dc.fx(0.0), self.dc.fy(a)), (self.pc.fx(a), self.pc.fy(b))
        self.pulse(sc, start, end, remap(f, 0.44, 0.8), ink, op, bend=0.10, r=6.0)
        self.spot(sc, end[0], end[1], 5.0, ink, op * smoothstep(0.76, 0.82, f), glow=0.3)
        self.tag(sc, TOYS, op, "y = %g + %gx + noise (σ = %g), seed %d"
                 % (S.A_TRUE, S.B_TRUE, S.SIGMA_Y, self.meta["seed"]))

    def phase_2(self, sc, u, op):
        lines = int(1 + (self.LINES - 1) * smoothstep(0.04, 0.5, u))
        count = int(3 + (self.SHOWN - 3) * smoothstep(0.04, 0.6, u))
        self.text(sc, 0.26, 0.305, "%d of the fitted lines" % lines, op, MID, strong=True)
        self.text(sc, 0.755, 0.305, "%d of %s fits" % (count, commas(self.meta["toys"])), op,
                  MID, strong=True)
        self._data_axes(sc, op)
        self._plane_axes(sc, op)
        pick = op * smoothstep(0.55, 0.65, u)
        for i in range(lines):
            a, b = self.fits[i]
            self._line(sc, a, b, BLUE, 0.3 * op, width=1.4)
        self._cloud(sc, op, count, skip=(self.steep, self.shallow) if pick > 0.01 else ())
        self._cross(sc, op)
        for i, ink, name, yv in ((self.steep, VIOLET, "steepest line: lowest start", 0.5),
                                 (self.shallow, TEAL, "shallowest: highest start", -0.3)):
            a, b = self.fits[i]
            self._line(sc, a, b, ink, pick, width=3.0)
            self.spot(sc, self.pc.fx(a), self.pc.fy(b), 6.5, ink, pick, lift=1.6, glow=0.3)
            self.text(sc, self.dc.x1 - 0.008, self.dc.fy(yv), name, pick, anchor="rm",
                      ink=dark(ink, 0.3))
        last = op * smoothstep(0.74, 0.84, u)
        x0 = self.pc.x0 + 0.016
        self.text(sc, x0, self.pc.fy(0.17), "correlation ρ = %s" % num(self.rho, 3), last,
                  MID, anchor="lm", strong=True)
        self.text(sc, x0, self.pc.fy(0.06), "high b goes with low a", last, anchor="lm")
        self.tag(sc, TOYS, op, "%s fits, seed %d" % (commas(self.meta["toys"]),
                                                     self.meta["seed"]))

    def _matrix(self, sc, v, cx, cy, op, title, ink):
        self.text(sc, cx, cy - 0.082, title, op, ink=dark(ink, 0.3))
        for r in range(2):
            for c in range(2):
                self.box(sc, cx + (c - 0.5) * 0.094, cy + (r - 0.5) * 0.07, 84, 31,
                         soft(ink, 0.6 if r == c else 0.8), op,
                         num(v[r, c], 4 if abs(v[r, c]) < 0.1 else 3), border=ink,
                         radius=4)

    def phase_3(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.06, 0.305, "the cloud as four numbers: a covariance matrix V", op, MID,
                  anchor="lm", strong=True)
        self.text(sc, 0.755, 0.305, "the ellipse those numbers describe", op, MID,
                  strong=True)
        self._matrix(sc, self.cov, 0.17, 0.455, op, "from %s fits" % commas(m["toys"]), BLUE)
        self._matrix(sc, self.exact, 0.40, 0.455, op * smoothstep(0.08, 0.16, u),
                     "exact formula", GREEN)
        sa, sb = np.sqrt(np.diag(self.cov))
        self.rows(sc, 0.06, 0.555, [
            ("diagonal: σ(a)² and σ(b)²; off it: the covariance", GREY,
             smoothstep(0.1, 0.18, u)),
            ("σ(a) = √%s = %s      σ(b) = √%s = %s"
             % (num(self.cov[0, 0], 3), num(sa, 3), num(self.cov[1, 1], 4), num(sb, 3)), BLUE,
             smoothstep(0.16, 0.24, u)),
            ("ρ = %s / (%s × %s) = %s" % (num(self.cov[0, 1], 4), num(sa, 3), num(sb, 3),
                                          num(self.rho, 3)), BLUE,
             smoothstep(0.24, 0.32, u)),
        ], op)
        self.rows(sc, 0.06, 0.715, [
            ("experiment 1 alone: where its χ² is χ²min + 1", RED, smoothstep(0.46, 0.54, u)),
            ("the same ellipse, to within %s" % pct(m["contour_mismatch"]), RED,
             smoothstep(0.78, 0.86, u)),
            ("it holds the truth in %s of fits (exact: %s)"
             % (pct(m["inside"]), pct(m["exact"]["inside"])), None, smoothstep(0.84, 0.9, u)),
            ("a ± σ(a) holds the true a in %s (exact: %s)"
             % (pct(m["inside_a"]), pct(m["exact"]["inside_one"])), None,
             smoothstep(0.88, 0.94, u)),
        ], op)
        self._plane_axes(sc, op)
        self._cloud(sc, op, self.SHOWN, soft(BLUE, 0.45))
        self._closed(sc, ellipse(self.cov, self.truth), dark(BLUE, 0.2), op,
                     smoothstep(0.28, 0.44, u), 3.2)
        self._cross(sc, op)
        on = op * smoothstep(0.46, 0.52, u)
        self.spot(sc, self.pc.fx(self.first[0]), self.pc.fy(self.first[1]), 5.0, RED, on,
                  lift=1.6)
        self._closed(sc, self.ring, RED, op, smoothstep(0.5, 0.64, u), 3.0)
        s = smoothstep(0.68, 0.84, u)
        if s > 0:                                       # the same contour carried to the truth
            moved = np.vstack([self.ring, self.ring[:1]]) + (self.truth - self.first) * s
            self._closed(sc, moved, RED, op, 1.0, 1.6)
        self.tag(sc, BOTH, op, "one dataset's χ², %s fits" % commas(m["toys"]))

    def phase_4(self, sc, u, op):
        m = self.meta
        hf, hh = self.half_free, self.half_held
        self.rows(sc, 0.06, 0.305, [
            ("b refitted at every a:  a ± %s" % num(hf, 3), RED),
            ("b held fixed:  a ± %s" % num(hh, 3), AMBER, smoothstep(0.3, 0.4, u)),
            ("%s × √(1 − ρ²) = %s × √(1 − %s²) = %s"
             % (num(hf, 3), num(hf, 3), num(abs(m["exact"]["rho"]), 3),
                num(hf * np.sqrt(1 - m["exact"]["rho"] ** 2), 3)), None,
             smoothstep(0.62, 0.72, u)),
        ], op, step=0.042)
        self.text(sc, 0.755, 0.305, "the same two widths on the ellipse", op, MID, strong=True)
        a0 = self.first[0]
        chart = LossChart(((0.09, 0.47), (0.47, 0.79)), (a0 - 1.2, a0 + 1.2), (0.0, 4.0))
        self.axes(sc, chart, op, "intercept a", "χ² of experiment 1, above its minimum",
                  xticks=[(v, num(v, 1)) for v in (-1.0, -0.5, 0.0, 0.5, 1.0)],
                  yticks=[(v, "%d" % v) for v in range(5)])
        self.dash(sc, (chart.x0, chart.fy(1.0)), (chart.x1, chart.fy(1.0)), AXIS, 0.45 * op, 18)
        keep = np.abs(self.grid - a0) <= 1.2
        g = self.grid[keep]
        for ys, ink, t0, at in ((self.free, RED, 0.04, self.free_at),
                                (self.held, AMBER, 0.3, self.held_at)):
            y = np.where(ys[keep] <= 4.0, ys[keep], np.nan)
            self.curve(sc, chart, g, y, ink, op, smoothstep(t0, t0 + 0.22, u), 3.0)
            on = op * smoothstep(t0 + 0.2, t0 + 0.26, u)
            for v in at[:2]:
                self.spot(sc, chart.fx(v), chart.fy(1.0), 5.5, ink, on, lift=1.5)
        # the plane: the full shadow of the contour on the a axis, and its slice at fixed b
        self._plane_axes(sc, op)
        self._cloud(sc, op, self.SHOWN, soft(BLUE, 0.6))
        self._closed(sc, np.vstack([self.ring, self.ring[:1]]), RED, op, 1.0, 2.8)
        self.spot(sc, self.pc.fx(a0), self.pc.fy(self.first[1]), 4.5, RED, op, lift=1.6)
        top = self.pc.fy(0.965)
        on = op * smoothstep(0.1, 0.2, u)
        lo, hi = int(np.argmin(self.ring[:, 0])), int(np.argmax(self.ring[:, 0]))
        for k in (lo, hi):
            x = self.pc.fx(self.ring[k, 0])
            self.seg(sc, (x, self.pc.fy(self.ring[k, 1])), (x, top), RED, 0.45 * on, 1.6)
        self.hbar_err(sc, self.pc.fx(self.ring[lo, 0]), self.pc.fx(self.ring[hi, 0]), top, RED,
                      on, 3.2)
        on = op * smoothstep(0.34, 0.44, u)
        yb = self.pc.fy(self.first[1])
        self.hbar_err(sc, self.pc.fx(self.held_at[0]), self.pc.fx(self.held_at[1]), yb, AMBER,
                      on, 3.6)
        self.rows(sc, self.pc.x0 + 0.016, self.pc.fy(0.27), [
            ("full shadow on the a axis: ± %s" % num(hf, 3), RED, smoothstep(0.1, 0.2, u)),
            ("slice at fixed b: ± %s" % num(hh, 3), AMBER, smoothstep(0.34, 0.44, u)),
        ], op, step=0.046)
        self.text(sc, 0.06, 0.925, "b fixed in %s fits: spread of a = %s"
                  % (commas(m["toys"]), num(m["conditional"], 3)),
                  op * smoothstep(0.76, 0.86, u), anchor="lm", strong=True)
        self.tag(sc, BOTH, op, "σ/√5 = %s" % num(m["exact"]["conditional"], 3))

    def phase_5(self, sc, u, op):
        m = self.meta
        s = smoothstep(0.32, 0.7, u)
        self.text(sc, 0.26, 0.305, "count x from the middle of the data", op, MID,
                  strong=True)
        self.text(sc, 0.755, 0.305, "the same %d fits, re-plotted" % self.SHOWN, op, MID,
                  strong=True)
        self._data_axes(sc, op)
        for i in range(self.LINES):
            a, b = self.fits[i]
            self._line(sc, a, b, BLUE, 0.3 * op, width=1.4)
        on = op * smoothstep(0.06, 0.18, u)
        xm = self.dc.fx(self.x.mean())
        self.seg(sc, (xm, self.dc.fy(4.6)), (xm, self.dc.fy(1.1)), AMBER, on, 3.0, 1.0)
        x0 = self.dc.x0 + 0.014
        self.text(sc, x0, 0.415, "new origin:  x′ = x − %g" % self.x.mean(), on, anchor="lm",
                  ink=dark(AMBER, 0.3))
        self.text(sc, self.dc.x1 - 0.008, self.dc.fy(0.5), "the lines cross near x = 3:", on,
                  anchor="rm")
        self.text(sc, self.dc.x1 - 0.008, self.dc.fy(-0.3), "there the height is best known",
                  on, anchor="rm")
        before, after = op * (1 - smoothstep(0.3, 0.4, u)), op * smoothstep(0.62, 0.72, u)
        self.axes(sc, self.pc, op, "", "slope b",
                  xticks=[(v, "%d" % v) for v in (0, 1, 2, 3)],
                  yticks=[(v, "%g" % v) for v in (0, 0.5, 1.0)])
        xl = ((self.pc.x0 + self.pc.x1) / 2, self.pc.y1 + 0.068)
        self.text(sc, xl[0], xl[1], "intercept a: the height at x = 0", before)
        self.text(sc, xl[0], xl[1], "a′: the height at x′ = 0 (x = 3)", after,
                  ink=dark(AMBER, 0.3))
        self._cloud(sc, op, self.SHOWN, shear=s)
        self._closed(sc, ellipse(self.cov, self.truth), dark(BLUE, 0.2), op, 1.0, 3.0, shear=s)
        self.rows(sc, self.pc.x0 + 0.016, self.pc.fy(0.84), [
            ("ρ: %s → %s" % (num(self.rho, 3), num(m["centred"]["rho"])), None),
            ("width: ± %s → ± %s" % (num(self.sa, 3), num(m["centred"]["sigma"], 3)), None),
            ("slope: ± %s, as before" % num(self.sb, 3), GREY),
        ], after, step=0.046)
        self.text(sc, 0.06, 0.925, "same data, same lines: only the question changed",
                  after, anchor="lm", strong=True)
        self.tag(sc, TOYS, op, "%s fits, seed %d" % (commas(m["toys"]), m["seed"]))


# ====================================================================== 134: stat and syst
class StatVsSyst(Syst):
    """Events arriving one by one and their mean; the mean against N homing in on the
    wrong value; statistical, systematic and total error against N; how often each
    quoted interval holds the truth; and a control sample that measures the shift."""

    TITLE = "Statistical against systematic error"
    PHASES = (("events", 10.0), ("more data", 11.0), ("two errors", 12.0),
              ("coverage", 10.0), ("a control", 12.0))
    STAGES = (1, 2, 4, 10, 25, 100)

    def setup(self):
        p = self.study.panels
        m = self.meta = self.study.metadata
        self.events = np.asarray(p[0].data["events"], float)
        self.running = np.asarray(p[0].data["ys"], float)[0]
        self.ns = np.asarray(p[1].data["x"], float)
        self.stat, self.syst, self.total, self.rms = (np.asarray(v, float)
                                                      for v in p[1].data["ys"])
        self.cover_stat, self.cover_exact, self.cover_total = (
            np.asarray(v, float) for v in p[2].data["ys"])
        self.means = np.asarray(p[2].data["first_means"], float)[:, -1]
        self.nc = np.asarray(p[3].data["x"], float)
        self.ctrl, self.ctrl_rms, self.floor = (np.asarray(v, float)
                                                for v in p[3].data["ys"])
        self.n_max = int(m["model"]["n_max"])
        self.model = m["model"]
        pick = np.unique(np.round(np.logspace(0, np.log10(self.n_max), 260)).astype(int))
        self.run_n = pick.astype(float)
        self.run_mean = self.running[pick - 1]
        self.i400 = list(m["ns"]).index(400)

    def subtitle_for(self, k):
        m = self.meta
        c = m["control"]
        last = self.n_max
        off = abs(m["mean_at"][str(last)] - self.model["mu"]) / m["stat_at"][str(last)]
        return [
            "each event measures the same quantity with a resolution σ = %g. The mean of "
            "N events has a statistical error σ/√N: %s after 4 events, %s after 25, %s "
            "after 100" % (self.model["resolution"], num(m["stat_at"]["4"]),
                           num(m["stat_at"]["25"]), num(m["stat_at"]["100"])),
            "more events, smaller error. But this detector reads %g too high in every "
            "event, and averaging cannot remove that: after %s events the mean is %s ± %s, "
            "%.0f errors from the truth"
            % (self.model["shift_drawn"], commas(last), num(m["mean_at"][str(last)]),
               num(m["stat_at"][str(last)]), off),
            "the shift is only known to be within ± %g: a systematic error. It adds in "
            "quadrature, so the total flattens: %s at N = 400, %s at %s. Measured over %s "
            "pseudo-experiments: %s"
            % (self.model["scale_error"], num(m["total_at"]["400"]),
               num(m["total_at"][str(last)]), commas(last), commas(m["toys"]),
               num(m["rms"][-1], 3)),
            "quote only the statistical error and the interval holds the truth less and "
            "less often: %s of the time at N = 400, %s at %s, where 68%% is promised. With "
            "the total error it stays at %s"
            % (pct(m["cover_stat"][self.i400], 0), pct(m["cover_stat"][-1], 0), commas(last),
               pct(m["cover_total"][-1], 0)),
            "a control sample measures the shift instead: a reference of known value in "
            "the same detector reads %s ± %s too high. That error shrinks with more data "
            "too, and beats the floor from N = %s"
            % (num(c["shift"]), num(c["shift_error"]), commas(m["control_pays_from"])),
        ][k]

    def phase_1(self, sc, u, op):
        m = self.meta
        j = min(int(np.clip((u - 0.03) / 0.72, 0, 0.999) * len(self.STAGES)),
                len(self.STAGES) - 1)
        n = self.STAGES[j]
        self.text(sc, 0.06, 0.315, "events arrive one by one; each scatters by σ = %g"
                  % self.model["resolution"], op, MID, anchor="lm", strong=True)
        at = self.scale(sc, 70.0, 130.0, 0.08, 0.60, 0.67, op, (70, 80, 90, 100, 110, 120, 130))
        base, r = 0.67, 4.0
        dy = self.frac(0, 2 * r + 0.8)[1]
        self.seg(sc, (at(100), 0.385), (at(100), base), GREEN, 0.45 * op, 2.2, 0.2)
        self.text(sc, at(100), 0.365, "true value: %g" % self.model["mu"], op,
                  ink=dark(GREEN, 0.3))
        stack = {}
        for i in range(n):
            b = int(round(float(np.clip(self.events[i], 70, 130))))
            k = stack[b] = stack.get(b, 0) + 1
            self.spot(sc, at(b), base - (k - 0.5) * dy - 0.004, r, BLUE,
                      op * (1.0 if i < n - 1 or n > 4 else 0.9))
        mean, err = self.running[n - 1], float(self.model["resolution"] / np.sqrt(n))
        self.hbar_err(sc, at(max(mean - err, 70)), at(min(mean + err, 130)), 0.765,
                      dark(BLUE, 0.15), op, 3.4, dot=at(mean), r=6.0)
        self.text(sc, 0.34, 0.82, "the mean of the first %d, ± σ/√N" % n, op,
                  ink=dark(BLUE, 0.3))
        self.rows(sc, 0.665, 0.40, [
            ("N = %d" % n, None),
            ("mean = %s" % num(mean), None),
            ("σ/√N = %g/√%d = %s" % (self.model["resolution"], n, num(err)), BLUE),
        ], op, step=0.052, size=MID)
        self.rows(sc, 0.665, 0.585, [
            ("four times the events:", GREY),
            ("half the error", GREY),
        ], op * smoothstep(0.3, 0.4, u), step=0.042)
        self.rows(sc, 0.665, 0.705, [
            ("this detector reads %g" % self.model["shift_drawn"], AMBER),
            ("too high in every event;", AMBER),
            ("we only know: 0 ± %g" % self.model["scale_error"], AMBER),
        ], op * smoothstep(0.6, 0.72, u), step=0.042)
        self.tag(sc, TOYS, op, "one run, event = %g + %g + noise (σ = %g), seed %d"
                 % (self.model["mu"], self.model["shift_drawn"], self.model["resolution"],
                    m["seed"]))

    def _n_ticks(self, extra=()):
        ticks = [(1, "1"), (10, "10"), (100, "100"), (1000, "1,000"), (10000, "10,000")]
        return [t for t in ticks if not any(0.5 < t[0] / e[0] < 2.6 for e in extra)] + list(extra)

    def phase_2(self, sc, u, op):
        m = self.meta
        mu, shift = self.model["mu"], self.model["shift_drawn"]
        go = float(np.clip((u - 0.06) / 0.74, 0, 1))
        chart = LossChart(((0.10, 0.57), (0.37, 0.79)), (1.0, float(self.n_max)),
                          (mu - 4, mu + 4), xlog=True)
        self.axes(sc, chart, op, "N: events averaged", "mean of the first N events",
                  xticks=self._n_ticks(), yticks=[(v, "%g" % v) for v in (97, 98, 99, 100,
                                                                         101, 102, 103)])
        self.seg(sc, (chart.x0, chart.fy(mu)), (chart.x1, chart.fy(mu)), GREEN, 0.45 * op, 2.6)
        self.dash(sc, (chart.x0, chart.fy(mu + shift)), (chart.x1, chart.fy(mu + shift)),
                  AMBER, 0.45 * op, 30, 2.6)
        err = self.model["resolution"] / np.sqrt(self.run_n)

        def inside(v):
            return np.where(np.abs(v - mu) <= 4, v, np.nan)
        for band in (self.run_mean - err, self.run_mean + err):
            self.curve(sc, chart, self.run_n, inside(band), soft(BLUE, 0.2), 0.45 * op, go, 2.0)
        end = self.curve(sc, chart, self.run_n, inside(self.run_mean), BLUE, op, go, 3.0)
        if end is not None:
            self.spot(sc, end[0], end[1], 5.5, BLUE, op, glow=0.4)
        k = max(2, int(round(1 + (len(self.run_n) - 1) * go)))
        n, mean, e = int(self.run_n[k - 1]), float(self.run_mean[k - 1]), float(err[k - 1])
        x = 0.63
        self.rows(sc, x, 0.375, [
            ("N = %s" % commas(n), None),
            ("mean = %s ± %s" % (num(mean), num(e)), BLUE),
        ], op, step=0.05, size=MID)
        # the same interval, close up: it narrows onto the wrong value
        lo, hi = mu - 1.0, mu + 2.0
        at = self.scale(sc, lo, hi, x, 0.93, 0.575, op, (mu - 1, mu, mu + 1, mu + 2))
        self.seg(sc, (at(mu), 0.50), (at(mu), 0.575), GREEN, 0.45 * op, 3.0, 0.2)
        self.seg(sc, (at(mu + shift), 0.50), (at(mu + shift), 0.575), AMBER, 0.45 * op, 3.0, 0.2)
        self.hbar_err(sc, at(float(np.clip(mean - e, lo, hi))),
                      at(float(np.clip(mean + e, lo, hi))), 0.535, BLUE, op, 3.4, cap=7.0,
                      dot=at(mean) if lo <= mean <= hi else None, r=5.0)
        self.key(sc, x + 0.008, 0.672, [
            ("the true value: %g" % mu, GREEN),
            ("this detector heads for %s" % num(mu + shift, 1), AMBER)], op)
        off = (m["mean_at"][str(self.n_max)] - mu) / m["stat_at"][str(self.n_max)]
        self.text(sc, x, 0.78, "the truth is now %.1f error bars away" % off,
                  op * smoothstep(0.84, 0.92, u), anchor="lm", strong=True)
        self.tag(sc, TOYS, op, "the same run, to %s events" % commas(self.n_max))

    def phase_3(self, sc, u, op):
        m = self.meta
        go = smoothstep(0.04, 0.5, u)
        chart = LossChart(((0.10, 0.60), (0.37, 0.79)), (1.0, float(self.n_max)), (0.05, 12.0),
                          xlog=True, ylog=True)
        self.axes(sc, chart, op, "N: events averaged", "error on the mean",
                  xticks=self._n_ticks([(400, "400")]),
                  yticks=[(v, "%g" % v) for v in (0.1, 0.5, 1, 10)])
        cross = float(m["crossover"])
        self.dash(sc, (chart.fx(cross), chart.y0), (chart.fx(cross), chart.y1), AXIS,
                  0.45 * op * smoothstep(0.5, 0.6, u), 16)
        self.curve(sc, chart, self.ns, self.stat, BLUE, op, go, 3.0)
        self.curve(sc, chart, self.ns, self.syst, AMBER, op, go, 3.0)
        self.curve(sc, chart, self.ns, self.total, RED, op, smoothstep(0.3, 0.7, u), 3.6,
                   lift=0.8)
        shown = int(len(self.ns) * smoothstep(0.62, 0.86, u))
        for i in range(shown):
            self.disk(sc, self.P(chart.fx(self.ns[i]), chart.fy(self.rms[i])), 5.0,
                      dark(RED, 0.45), op, lift=1.6, outline=0.0)
        self.key(sc, chart.fx(cross) + 0.03, 0.40, [
            ("statistical: σ/√N", BLUE), ("systematic: %g" % self.model["scale_error"], AMBER),
            ("total", RED)], op)
        self.text(sc, chart.fx(cross) + 0.03 - 0.008, 0.535, "● measured", op * smoothstep(
            0.62, 0.7, u), anchor="lm", ink=dark(RED, 0.45))
        last = str(self.n_max)
        self.rows(sc, 0.645, 0.385, [
            ("total = √(stat² + syst²)", RED),
            ("N = 400:", None, smoothstep(0.5, 0.58, u)),
            ("√(%s² + %g²) = %s" % (num(m["stat_at"]["400"], 1), self.model["scale_error"],
                                    num(m["total_at"]["400"])), None,
             smoothstep(0.5, 0.58, u)),
            ("N = %s:" % commas(self.n_max), None, smoothstep(0.66, 0.74, u)),
            ("√(%s² + %g²) = %s" % (num(m["stat_at"][last], 1), self.model["scale_error"],
                                    num(m["total_at"][last])), None,
             smoothstep(0.66, 0.74, u)),
            ("%d times the data:" % (self.n_max // 400), GREY, smoothstep(0.8, 0.88, u)),
            ("%s → %s, never below %g" % (num(m["total_at"]["400"]), num(m["total_at"][last]),
                                         self.model["scale_error"]), GREY,
             smoothstep(0.8, 0.88, u)),
            ("measured at %s: %s" % (commas(self.n_max), num(m["rms"][-1], 3)), RED,
             smoothstep(0.86, 0.94, u)),
        ], op, step=0.046)
        self.tag(sc, BOTH, op, "%s runs, a new shift (σ = %g) in each"
                 % (commas(m["toys"]), self.model["scale_error"]))

    def phase_4(self, sc, u, op):
        m = self.meta
        mu = self.model["mu"]
        grow = smoothstep(0.42, 0.58, u)
        half = float(self.stat[-1] + (self.total[-1] - self.stat[-1]) * grow)
        ink = tuple(np.asarray(BLUE) + (np.asarray(RED) - np.asarray(BLUE)) * grow)
        self.text(sc, 0.06, 0.315, "%d runs of %s events, each with its own shift"
                  % (len(self.means), commas(self.n_max)), op, MID, anchor="lm", strong=True)
        at = self.scale(sc, mu - 1.5, mu + 1.5, 0.08, 0.46, 0.80, op, (mu - 1, mu, mu + 1))
        self.seg(sc, (at(mu), 0.36), (at(mu), 0.80), GREEN, 0.45 * op, 2.4, 0.2)
        holds = 0
        shown = int(len(self.means) * smoothstep(0.02, 0.3, u))
        for i, v in enumerate(self.means[:shown]):
            y = 0.372 + 0.0104 * i
            ok = abs(v - mu) <= half
            holds += ok
            self.seg(sc, (at(max(v - half, mu - 1.5)), y), (at(min(v + half, mu + 1.5)), y),
                     ink if ok else DIM, op, 2.6, 0.8)
            self.spot(sc, at(v), y, 2.6, DIGIT_INK if ok else GREY, op, lift=1.4)
        self.text(sc, 0.27, 0.885, "mean ± %s: holds the truth in %d of %d"
                  % ("%s (%s)" % (num(half), "statistical only" if grow < 0.5 else "total"),
                     holds, shown), op, ink=dark(ink, 0.3))
        chart = LossChart(((0.60, 0.93), (0.40, 0.79)), (1.0, float(self.n_max)), (0.0, 1.0),
                          xlog=True)
        self.axes(sc, chart, op, "N: events averaged", "share of intervals holding the truth",
                  xticks=[(1, "1"), (100, "100"), (10000, "10,000")],
                  yticks=[(0, "0"), (0.5, "50%"), (S.ONE_SIGMA, "68%"), (1, "100%")])
        self.dash(sc, (chart.x0, chart.fy(S.ONE_SIGMA)), (chart.x1, chart.fy(S.ONE_SIGMA)),
                  AXIS, 0.45 * op, 18)
        go = smoothstep(0.1, 0.5, u)
        self.curve(sc, chart, self.ns, self.cover_exact, BLUE, op, go, 3.0)
        for i in range(int(len(self.ns) * go)):
            self.spot(sc, chart.fx(self.ns[i]), chart.fy(self.cover_stat[i]), 4.2,
                      dark(BLUE, 0.4), op, lift=1.5)
        go = smoothstep(0.55, 0.85, u)
        for i in range(int(len(self.ns) * go)):
            self.spot(sc, chart.fx(self.ns[i]), chart.fy(self.cover_total[i]), 4.2,
                      dark(RED, 0.2), op, lift=1.5)
        x = chart.x0 + 0.016
        self.text(sc, x, chart.fy(0.30), "statistical error only:", op, anchor="lm",
                  ink=dark(BLUE, 0.3))
        self.text(sc, x, chart.fy(0.19), "down to %s" % pct(m["cover_stat"][-1], 0), op,
                  anchor="lm", ink=dark(BLUE, 0.3))
        self.text(sc, x, chart.fy(0.87), "total error: stays near 68%",
                  op * smoothstep(0.6, 0.7, u), anchor="lm", ink=dark(RED, 0.3))
        self.tag(sc, BOTH, op, "points: %s runs; line: erf" % commas(m["toys"]))

    def phase_5(self, sc, u, op):
        m = self.meta
        c = m["control"]
        mu, ref = self.model["mu"], self.model["reference"]
        mean = m["mean_at"][str(self.n_max)]
        self.text(sc, 0.06, 0.315, "measure the shift where the answer is known", op, MID,
                  anchor="lm", strong=True)
        x0, x1 = 0.08, 0.50
        # the control: a reference whose true value is known
        on = op * smoothstep(0.02, 0.12, u)
        at = self.scale(sc, ref - 1.0, ref + 1.0, x0, x1, 0.47, on,
                        (ref - 1, ref - 0.5, ref, ref + 0.5, ref + 1), fmt="%.1f")
        self.text(sc, x0, 0.365, "control: %s events of a reference known to be %g"
                  % (commas(c["events"]), ref), on, anchor="lm", ink=dark(VIOLET, 0.3))
        self.seg(sc, (at(ref), 0.395), (at(ref), 0.47), GREEN, 0.45 * on, 2.6, 0.2)
        self.hbar_err(sc, at(c["mean"] - c["shift_error"]), at(c["mean"] + c["shift_error"]),
                      0.425, VIOLET, on, 3.2, dot=at(c["mean"]))
        g1 = smoothstep(0.16, 0.3, u)
        self.arrow(sc, (at(ref), 0.452), (at(ref + c["shift"] * g1), 0.452), AMBER, on * (
            g1 > 0.02), 1.0, 3.0)
        self.text(sc, x0, 0.555, "shift = %s − %g = %s ± %s"
                  % (num(c["mean"]), ref, signed(c["shift"]), num(c["shift_error"])),
                  op * smoothstep(0.24, 0.34, u), anchor="lm", ink=dark(AMBER, 0.3))
        # the measurement, corrected by that shift
        on = op * smoothstep(0.36, 0.46, u)
        at2 = self.scale(sc, mu - 1.0, mu + 1.0, x0, x1, 0.74, on,
                         (mu - 1, mu - 0.5, mu, mu + 0.5, mu + 1), fmt="%.1f")
        self.text(sc, x0, 0.625, "the measurement: %s events, mean %s"
                  % (commas(self.n_max), num(mean)), on, anchor="lm", ink=dark(BLUE, 0.3))
        self.seg(sc, (at2(mu), 0.655), (at2(mu), 0.74), GREEN, 0.45 * on, 2.6, 0.2)
        stat = m["stat_at"][str(self.n_max)]
        g2 = smoothstep(0.5, 0.66, u)
        self.hbar_err(sc, at2(mean - stat), at2(mean + stat), 0.682, BLUE, on * (1 - 0.6 * g2),
                      3.2, dot=at2(mean))
        self.arrow(sc, (at2(mean), 0.716), (at2(mean - c["shift"] * g2), 0.716), AMBER,
                   on * (g2 > 0.02), 1.0, 3.0)
        done = op * smoothstep(0.64, 0.72, u)
        self.hbar_err(sc, at2(c["corrected"] - c["corrected_error"]),
                      at2(c["corrected"] + c["corrected_error"]), 0.682, RED, done, 3.4,
                      dot=at2(c["corrected"]))
        self.rows(sc, x0, 0.825, [
            ("%s − %s = %s ± %s" % (num(mean), num(c["shift"]), num(c["corrected"]),
                                    num(c["corrected_error"])), RED),
            ("error: √(%s² + %s²) = %s, not %s"
             % (num(stat), num(c["shift_error"]), num(c["corrected_error"]),
                num(m["total_at"][str(self.n_max)])), None),
        ], done, step=0.046)
        chart = LossChart(((0.62, 0.93), (0.40, 0.79)), (4.0, float(self.n_max)), (0.1, 15.0),
                          xlog=True, ylog=True)
        on = op * smoothstep(0.5, 0.6, u)
        pays = float(m["control_pays_from"])
        self.axes(sc, chart, on, "N: events (control: N/%d)" % self.model["control_share"],
                  "total error",
                  xticks=[(10, "10"), (100, "100"), (pays, commas(pays)), (10000, "10,000")],
                  yticks=[(v, "%g" % v) for v in (0.1, 0.5, 1, 10)])
        go = smoothstep(0.56, 0.84, u)
        self.dash(sc, (chart.fx(pays), chart.y0), (chart.fx(pays), chart.y1), AXIS, 0.45 * on,
                  16)
        self.curve(sc, chart, self.nc, self.floor, RED, on, go, 3.0)
        self.curve(sc, chart, self.nc, self.ctrl, VIOLET, on, go, 3.2)
        for i in range(int(len(self.nc) * smoothstep(0.7, 0.92, u))):
            self.spot(sc, chart.fx(self.nc[i]), chart.fy(self.ctrl_rms[i]), 4.4,
                      dark(VIOLET, 0.4), on, lift=1.5)
        self.key(sc, chart.fx(5), chart.fy(0.3), [
            ("shift known to ± %g" % self.model["scale_error"], RED),
            ("shift measured", VIOLET)], on)
        self.tag(sc, BOTH, op, "points: %s runs" % commas(m["toys"]))


# ====================================================================== 135: profiling
FIXED, PROFILED, MARGINAL = RED, GREEN, VIOLET


class Profiling(Syst):
    """A count split into signal and an uncertain background; the likelihood over both
    as a ridge closed into a tilted valley; the cut at fixed background, the path along
    the valley floor and the average over the background as three curves; their three
    intervals; and how often each holds the truth."""

    TITLE = "Profiling a nuisance parameter"
    PHASES = (("the count", 10.0), ("the valley", 10.0), ("profile", 14.0),
              ("intervals", 9.0), ("coverage", 12.0))
    MAP = ((0.10, 0.44), (0.385, 0.80))
    NAMES = ("b fixed", "b profiled", "b marginalised")
    INKS3 = (FIXED, PROFILED, MARGINAL)

    def setup(self):
        p = self.study.panels
        m = self.meta = self.study.metadata
        self.n = int(m["count"])
        self.mu_map = np.asarray(p[0].data["mu"], float)
        self.b_map = np.asarray(p[0].data["b"], float)
        self.surface = np.asarray(p[0].data["values"], float)
        self.path_mu = np.asarray(p[0].data["path_mu"], float)
        self.path_b = np.asarray(p[0].data["path_b"], float)
        self.grid = np.asarray(p[1].data["x"], float)
        self.three = np.asarray(p[1].data["ys"], float)
        self.iv = np.asarray(p[2].data["intervals"], float)
        self.cover = np.asarray(p[3].data["values"], float)
        self.cover_exact = np.asarray(p[3].data["exact"], float)
        self.shown = np.asarray(p[3].data["shown"], float)
        self.s, self.b0, self.sb = (m["model"][k] for k in ("signal", "b_nominal", "b_error"))
        self.mu_true = m["model"]["mu"]
        mu, b = self.mu_map[None, :], self.b_map[:, None]
        lam = mu * self.s + b
        count = 2.0 * (lam - self.n * np.log(lam))
        side = ((b - self.b0) / self.sb) ** 2 + 0.0 * mu
        self.images = [self._image(np.exp(-(v - v.min()) / 2), ink)
                       for v, ink in ((count, BLUE), (side, AMBER), (self.surface, TEAL))]
        best = np.array([(self.n - self.b0) / self.s, self.b0])
        low = float(S.nll2(best[0], best[1], self.n))
        ring = S.contour(lambda q, r: S.nll2(q, r, self.n) - low, best, 1.0, (0.3, 5.0))
        self.ring = np.vstack([ring, ring[:1]])
        self.mc = LossChart(self.MAP, (self.mu_map[0], self.mu_map[-1]),
                            (self.b_map[0], self.b_map[-1]))

    @staticmethod
    def _image(rel, ink):
        v = (np.clip(rel, 0, 1) ** 0.8)[..., None]
        return (PAPER * (1 - v) + np.asarray(dark(ink, 0.1)) * v)[::-1].astype(np.float32)

    def subtitle_for(self, k):
        m = self.meta
        iv, hw = m["intervals"], m["half_width"]
        return [
            "count events: n = %d. Expected are μ × %g signal events plus b background "
            "ones, and b is only known as %g ± %g. μ is what we want; b is a nuisance "
            "parameter: needed, but not of interest" % (self.n, self.s, self.b0, self.sb),
            "the likelihood L(μ, b) is the chance of the data for each pair of values. "
            "The count fixes only the sum μ × %g + b: a ridge. What is known about b "
            "closes it into a tilted valley of −2 ln L" % self.s,
            "fix b at %g and −2 ln L rises steeply: a narrow curve. Profile instead, "
            "taking the best b at every μ along the valley floor, and it is wider. "
            "Averaging L over b (marginalising) gives nearly the same" % self.b0,
            "each interval ends where its curve has risen by 1. With b fixed, μ is %s to "
            "%s: half-width %s. Profiled, %s to %s: half-width %s. The true μ = %g is "
            "inside only the wider one"
            % (num(iv["fixed"][0]), num(iv["fixed"][1]), num(hw["fixed"]),
               num(iv["profiled"][0]), num(iv["profiled"][1]), num(hw["profiled"]),
               self.mu_true) if not iv["fixed"][0] <= self.mu_true <= iv["fixed"][1] else
            "each interval ends where its curve has risen by 1. With b fixed, μ is %s to "
            "%s: half-width %s. Profiled, %s to %s: half-width %s"
            % (num(iv["fixed"][0]), num(iv["fixed"][1]), num(hw["fixed"]),
               num(iv["profiled"][0]), num(iv["profiled"][1]), num(hw["profiled"])),
            "in %s pseudo-experiments, each with its own true background, the fixed "
            "interval holds μ = %g only %s of the time; the profiled one %s, the "
            "marginalised one %s. A ±1σ interval promises %s"
            % (commas(m["toys"]), self.mu_true, pct(self.cover[0]), pct(self.cover[1]),
               pct(self.cover[2]), pct(m["nominal"])),
        ][k]

    # ------------------------------------------------------------------ phase 1
    def phase_1(self, sc, u, op):
        m = self.meta
        n, s, b0, sb = self.n, self.s, self.b0, self.sb
        x0, per = 0.10, 0.0072                           # layout x per event
        b = b0 + sb * (-smoothstep(0.22, 0.4, u) + 2 * smoothstep(0.46, 0.7, u)
                       - smoothstep(0.76, 0.9, u))
        self.text(sc, 0.06, 0.315, "expected count = μ × %g signal + b background" % s, op,
                  MID, anchor="lm", strong=True)

        def bar(lo, hi, y, fill, text, border=None, ink=None):
            w = self.span((hi - lo) * per)[0]
            self.box(sc, x0 + per * (lo + hi) / 2, y, w, 34, fill, op, text, ink=ink,
                     border=border, radius=4)
        bar(0, n, 0.40, soft(GREY, 0.55), "counted: n = %d events" % n, GREY)
        bar(0, b, 0.50, soft(AMBER, 0.35), "background b = %.0f" % b, AMBER)
        bar(b, n, 0.50, soft(BLUE, 0.35), "signal μ × %g" % s, BLUE)
        for v in range(0, 111, 10):
            self.seg(sc, (x0 + per * v, 0.735), (x0 + per * v, 0.747), AXIS, 0.45 * op, 1.4)
        self.seg(sc, (x0, 0.735), (x0 + per * 110, 0.735), AXIS, 0.45 * op, 1.4)
        for v in (0, 20, 40, 60, 80, 100):
            self.text(sc, x0 + per * v, 0.771, "%d" % v, op)
        self.text(sc, x0 + per * 110 + 0.012, 0.771, "events", op, anchor="lm")
        # what is known about b: a bell on the same scale
        g = np.linspace(b0 - 3 * sb, b0 + 3 * sb, 61)
        bell = 0.735 - 0.115 * np.exp(-0.5 * ((g - b0) / sb) ** 2)
        self.polyline(sc, [self.P(x0 + per * a, float(y), lift=0.4) for a, y in zip(g, bell)],
                      3.0, AMBER, op, bias=0.3)
        xb = x0 + per * b
        self.seg(sc, (xb, 0.535), (xb, 0.735), dark(AMBER, 0.2), 0.45 * op, 2.4, 0.9)
        self.spot(sc, xb, 0.735 - 0.115 * np.exp(-0.5 * ((b - b0) / sb) ** 2), 6.0,
                  dark(AMBER, 0.15), op, lift=1.6, glow=0.3)
        self.text(sc, x0 + per * (b0 + 3 * sb) + 0.02, 0.60, "what is known about b:", op,
                  anchor="lm", ink=dark(AMBER, 0.3))
        self.text(sc, x0 + per * (b0 + 3 * sb) + 0.02, 0.645, "%g ± %g" % (b0, sb), op, MID,
                  anchor="lm", ink=dark(AMBER, 0.3))
        self.text(sc, 0.06, 0.835, "if b = %.0f:   μ = (%d − %.0f) / %g = %s"
                  % (b, n, b, s, num((n - round(b)) / s)), op, BIG, anchor="lm", strong=True)
        self.text(sc, 0.06, 0.885, "any b from %g to %g is plausible, so μ could be %s or %s"
                  % (b0 - sb, b0 + sb, num((n - b0 + sb) / s), num((n - b0 - sb) / s)),
                  op * smoothstep(0.5, 0.62, u), anchor="lm")
        self.tag(sc, TOYS, op, "run 1 of %s; true μ = %g, b = %s"
                 % (commas(m["toys"]), self.mu_true, num(m["b_true"], 1)))

    # ------------------------------------------------------------------ the map
    def _map(self, sc, op, mix):
        """The (mu, b) plane: ``mix`` weights the three images (count, side, both)."""
        c = self.mc
        centre = self.P((c.x0 + c.x1) / 2, (c.y0 + c.y1) / 2)
        w, h = self.span(c.x1 - c.x0, c.y1 - c.y0)
        self.rect(sc, centre, w + 6, h + 6, PAPER, op, border=AXIS, border_w=1.2, radius=3,
                  lift=0.1)
        for img, a in zip(self.images, mix):
            if a > 0.01:
                draw_image(self, sc, centre, img, w, h, op * a, grid=0)
        self.axes(sc, c, op, "signal strength μ", "background b", grid=False,
                  xticks=[(v, "%g" % v) for v in (0, 1, 2, 3)],
                  yticks=[(v, "%g" % v) for v in (20, 40, 60, 80)])

    def phase_2(self, sc, u, op):
        m = self.meta
        a = smoothstep(0.26, 0.36, u)
        b = smoothstep(0.52, 0.62, u)
        self._map(sc, op, (1 - a, a * (1 - b), b))
        self.text(sc, 0.06, 0.305, "how likely is the data, for every μ and b?", op, MID,
                  anchor="lm", strong=True)
        self.curve(sc, self.mc, self.ring[:, 0], self.ring[:, 1], DIGIT_INK, op,
                   smoothstep(0.66, 0.8, u), 2.6, lift=1.0)
        x = 0.52
        self.rows(sc, x, 0.385, [
            ("the count alone: every μ and b with", BLUE),
            ("μ × %g + b = %d fits equally: a ridge" % (self.s, self.n), BLUE),
        ], op, step=0.042)
        self.rows(sc, x, 0.495, [
            ("what is known about b alone:", AMBER),
            ("%g ± %g whatever μ is: a band" % (self.b0, self.sb), AMBER),
        ], op * a, step=0.042)
        self.rows(sc, x, 0.605, [
            ("likelihood L(μ, b): the two multiplied.", TEAL),
            ("A tilted valley of −2 ln L", TEAL),
        ], op * b, step=0.042)
        self.rows(sc, x, 0.715, [
            ("inside the line, −2 ln L is within 1", None),
            ("of its minimum at μ = %s, b = %g" % (num(m["mu_hat"][0]), self.b0), None),
        ], op * smoothstep(0.7, 0.8, u), step=0.042)
        self.tag(sc, EXACT, op, "the likelihood of n = %d" % self.n)

    def phase_3(self, sc, u, op):
        self._map(sc, op, (0, 0, 1))
        self.text(sc, 0.06, 0.305, "L(μ, b): darker is more likely", op, MID, anchor="lm",
                  strong=True)
        c = self.mc
        chart = LossChart(((0.56, 0.93), (0.47, 0.79)), (self.grid[0], self.grid[-1]),
                          (0.0, 4.0))
        self.axes(sc, chart, op, "signal strength μ", "−2 ln L, above its minimum",
                  xticks=[(v, "%g" % v) for v in (0, 1, 2, 3)],
                  yticks=[(v, "%d" % v) for v in range(5)])
        self.dash(sc, (chart.x0, chart.fy(1.0)), (chart.x1, chart.fy(1.0)), AXIS, 0.45 * op, 18)
        stages = ((0.04, 0.34), (0.38, 0.7), (0.76, 0.92))
        labels = ("b fixed at %g: a straight cut" % self.b0,
                  "b profiled: the best b at each μ", "b marginalised: L averaged over b")
        for k, ((t0, t1), ink) in enumerate(zip(stages, self.INKS3)):
            go = float(np.clip((u - t0) / (t1 - t0), 0, 1))
            self.text(sc, 0.56, 0.305 + 0.042 * k, labels[k], op * smoothstep(t0, t0 + 0.06, u),
                      anchor="lm", ink=dark(ink, 0.3))
            y = np.where(self.three[k] <= 4.0, self.three[k], np.nan)
            self.curve(sc, chart, self.grid, y, ink, op, go, 3.0 if k < 2 else 2.0,
                       lift=0.5 + 0.2 * k)
            if k == 2 or go <= 0:
                continue
            pb = np.full(len(self.path_mu), self.b0) if k == 0 else self.path_b
            ok = (pb >= self.b_map[0]) & (pb <= self.b_map[-1])
            end = self.curve(sc, c, self.path_mu, np.where(ok, pb, np.nan), ink, op, go, 3.2,
                             lift=1.2)
            if end is not None and go < 1:
                self.spot(sc, end[0], end[1], 6.5, ink, op, lift=1.8, glow=0.4)
                j = max(2, int(round(1 + (len(self.grid) - 1) * go))) - 1
                if np.isfinite(y[j]):
                    self.spot(sc, chart.fx(self.grid[j]), chart.fy(y[j]), 6.0, ink, op,
                              lift=1.8, glow=0.4)
        self.tag(sc, EXACT, op, "the likelihood of n = %d" % self.n)

    def phase_4(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.06, 0.315, "where each curve has risen by 1: the interval on μ", op,
                  MID, anchor="lm", strong=True)
        at = self.scale(sc, 0.0, 3.0, 0.27, 0.78, 0.70, op, (0, 0.5, 1, 1.5, 2, 2.5, 3))
        self.text(sc, 0.80, 0.70, "μ", op, MID, anchor="lm")
        self.seg(sc, (at(self.mu_true), 0.40), (at(self.mu_true), 0.70), DIGIT_INK, 0.45 * op,
                 2.4, 0.2)
        self.text(sc, at(self.mu_true), 0.375, "true μ = %g" % self.mu_true, op)
        for k, (name, ink) in enumerate(zip(self.NAMES, self.INKS3)):
            y = 0.455 + 0.075 * k
            on = op * smoothstep(0.05 + 0.2 * k, 0.15 + 0.2 * k, u)
            lo, hi, best = self.iv[k]
            self.text(sc, 0.25, y, name, on, anchor="rm", ink=dark(ink, 0.3))
            self.hbar_err(sc, at(lo), at(hi), y, ink, on, 3.6, dot=at(best), r=6.0)
            self.text(sc, at(hi) + 0.016, y, "%s to %s:  ± %s" % (num(lo), num(hi),
                                                                 num((hi - lo) / 2)),
                      on, anchor="lm", ink=dark(ink, 0.3))
        r = m["rule_of_thumb"]
        self.rows(sc, 0.06, 0.80, [
            ("counting error only:  √n / s = √%d / %g = %s" % (self.n, self.s,
                                                              num(r["fixed"])), FIXED,
             smoothstep(0.55, 0.65, u)),
            ("with the background's:  √(n + σ(b)²) / s = √(%d + %g²) / %g = %s"
             % (self.n, self.sb, self.s, num(r["profiled"])), PROFILED,
             smoothstep(0.7, 0.8, u)),
        ], op, step=0.046)
        self.tag(sc, EXACT, op, "n = %d" % self.n)

    def phase_5(self, sc, u, op):
        m = self.meta
        k = len(self.shown)
        self.text(sc, 0.06, 0.315, "%d of the runs: each its own b, n and interval" % k, op,
                  MID, anchor="lm", strong=True)
        at = self.scale(sc, -1.0, 3.5, 0.08, 0.46, 0.80, op, (-1, 0, 1, 2, 3))
        self.seg(sc, (at(self.mu_true), 0.36), (at(self.mu_true), 0.80), DIGIT_INK, 0.45 * op,
                 2.4, 0.2)
        wide = smoothstep(0.34, 0.5, u)
        shown = int(k * smoothstep(0.02, 0.28, u))
        hits = [0, 0]
        for i in range(shown):
            y = 0.372 + 0.0104 * i
            for j, ink, w in ((1, PROFILED, 2.0), (0, FIXED, 3.6)):
                lo, hi = (float(np.clip(v, -1.0, 3.5)) for v in self.shown[i, j, :2])
                ok = self.shown[i, j, 0] <= self.mu_true <= self.shown[i, j, 1]
                hits[j] += ok
                self.seg(sc, (at(lo), y), (at(hi), y), ink if ok or j else soft(FIXED, 0.5),
                         op * (wide if j else 1.0), w, 0.6 + 0.3 * (1 - j))
        self.text(sc, 0.27, 0.885, "true μ inside: fixed %d of %d%s"
                  % (hits[0], shown, ", profiled %d" % hits[1] if wide > 0.5 else ""), op)
        x, unit = 0.675, 310.0
        self.text(sc, 0.56, 0.39, "all %s runs: how often the interval" % commas(m["toys"]), op,
                  anchor="lm", strong=True)
        self.text(sc, 0.56, 0.432, "holds the true μ", op, anchor="lm", strong=True)
        nominal = x + self.frac(unit * m["nominal"])[0]
        on = op * smoothstep(0.4, 0.5, u)
        self.seg(sc, (nominal, 0.475), (nominal, 0.70), DIGIT_INK, 0.45 * on, 2.2, 0.2)
        self.text(sc, nominal, 0.735, "promised: %s" % pct(m["nominal"]), on)
        for j, (name, ink) in enumerate(zip(("fixed", "profiled", "marginalised"),
                                            self.INKS3)):
            y = 0.51 + 0.07 * j
            t0 = 0.42 + 0.14 * j
            self.hbar(sc, x, y, unit * self.cover[j], ink, op * smoothstep(t0, t0 + 0.08, u),
                      label=name, value=pct(self.cover[j]), u=smoothstep(t0, t0 + 0.16, u),
                      thick=22)
        self.text(sc, 0.56, 0.80, "exact, summed over every count:", op * smoothstep(
            0.84, 0.92, u), anchor="lm")
        self.text(sc, 0.56, 0.842, ", ".join(pct(v) for v in self.cover_exact),
                  op * smoothstep(0.84, 0.92, u), anchor="lm")
        self.tag(sc, BOTH, op, "true b drawn from %g ± %g" % (self.b0, self.sb))


# ====================================================================== 136: combining
ONE, TWO, SHARED, NAIVE, BEST = BLUE, GREEN, AMBER, RED, VIOLET


class Combining(Syst):
    """Two measurements averaged with weights 1/sigma^2; their errors split into an own
    part and a shared shift; the best linear combination for that covariance, with a
    negative weight, on pair after pair; the weights against the correlation; and the
    three estimates' spread and quoted error over many pairs."""

    TITLE = "Combining measurements"
    PHASES = (("two results", 10.0), ("shared shift", 11.0), ("best weights", 12.0),
              ("correlation", 10.0), ("many pairs", 12.0))
    PAIRS = 6

    def setup(self):
        p = self.study.panels
        m = self.meta = self.study.metadata
        self.xy = np.asarray(p[0].data["xy"], float)
        self.own = np.asarray(p[0].data["own"], float)
        self.shared = np.asarray(p[0].data["shared"], float)
        self.rhos = np.asarray(p[1].data["x"], float)
        self.w1, self.w2, self.err = (np.asarray(v, float) for v in p[1].data["ys"])
        self.spread = np.asarray(p[2].data["values"], float)
        self.hist = np.asarray(p[2].data["hist"], float)
        self.hist_range = [float(v) for v in p[2].data["hist_range"]]
        self.quoted = np.asarray(p[3].data["values"], float)
        self.cover = np.asarray(p[3].data["coverage"], float)
        self.sig = m["sigmas"]
        self.cov = np.asarray(m["covariance"], float)
        self.wn, self.wb = (np.asarray(m["weights"][k], float) for k in ("naive", "best"))
        self.truth = m["model"]["truth"]
        self.stat, self.sens = m["model"]["own_errors"], m["model"]["shared_shift"]

    def subtitle_for(self, k):
        m = self.meta
        x = m["first_pair"]
        q, sp, cv = m["quoted"], m["spread"], m["coverage"]
        return [
            "two measurements of one quantity: %s ± %s and %s ± %s. Weight each by 1/σ² "
            "and the average is %s ± %s, better than either. That is right when their "
            "errors are independent"
            % (num(x[0]), num(self.sig[0], 1), num(x[1]), num(self.sig[1], 1),
               num(m["first_combined"]["naive"]), num(q["naive"])),
            "here they are not: part of each error is one shared effect δ, which moves "
            "measurement 2 three times as far as measurement 1. Their covariance is %s, "
            "their correlation ρ = %s" % (num(self.cov[0, 1]), num(m["rho"])),
            "for that covariance the combination with the smallest error gives "
            "measurement 2 a negative weight, %s. The result lies outside both inputs: "
            "their difference reveals the shared shift, which is taken off"
            % num(self.wb[1], 3),
            "with errors %s and %s, the weight of measurement 2 falls as ρ grows: zero at "
            "ρ = %s (the ratio of the errors), negative beyond. There combining gains "
            "nothing; past it the error shrinks again"
            % (num(self.sig[0], 1), num(self.sig[1], 1), num(m["negative_from"])),
            "over %s pairs the best combination scatters least: %s. The 1/σ² average "
            "quotes ± %s but scatters by %s, worse than measurement 1 alone, and holds "
            "the truth only %s of the time"
            % (commas(m["toys"]), num(sp["best"]), num(q["naive"]), num(sp["naive"]),
               pct(cv["naive"])),
        ][k]

    def _truth(self, sc, at, y0, y1, op, label=True):
        self.seg(sc, (at(self.truth), y0), (at(self.truth), y1), DIGIT_INK, 0.45 * op, 2.2, 0.2)
        if label:
            self.text(sc, at(self.truth), y0 - 0.026, "true value: %g" % self.truth, op)

    def phase_1(self, sc, u, op):
        m = self.meta
        x = m["first_pair"]
        at = self.scale(sc, 6.0, 16.0, 0.08, 0.60, 0.67, op, range(6, 17, 2))
        self._truth(sc, at, 0.39, 0.67, op)
        self.text(sc, 0.06, 0.315, "one quantity, measured twice", op, MID, anchor="lm",
                  strong=True)
        rows = ((x[0], self.sig[0], ONE, "x₁ = %s ± %s", 0.0),
                (x[1], self.sig[1], TWO, "x₂ = %s ± %s", 0.1))
        for k, (v, s, ink, fmt, t0) in enumerate(rows):
            y = 0.45 + 0.065 * k
            on = op * smoothstep(t0, t0 + 0.1, u)
            self.hbar_err(sc, at(v - s), at(v + s), y, ink, on, 3.4, dot=at(v), r=6.0)
            self.text(sc, at(v + s) + 0.016, y, fmt % (num(v), num(s, 1)), on, anchor="lm",
                      ink=dark(ink, 0.3))
        avg, err = m["first_combined"]["naive"], m["quoted"]["naive"]
        on = op * smoothstep(0.62, 0.72, u)
        self.hbar_err(sc, at(avg - err), at(avg + err), 0.60, NAIVE, on, 3.8, dot=at(avg), r=6.5)
        self.text(sc, at(avg + err) + 0.016, 0.60, "average = %s ± %s" % (num(avg), num(err)),
                  on, anchor="lm", ink=dark(NAIVE, 0.3))
        raw = [1.0 / s ** 2 for s in self.sig]
        self.rows(sc, 0.665, 0.385, [
            ("weight = 1/σ²", None),
            ("1 / %s² = %s" % (num(self.sig[0], 1), num(raw[0], 3)), ONE,
             smoothstep(0.2, 0.28, u)),
            ("1 / %s² = %s" % (num(self.sig[1], 1), num(raw[1], 3)), TWO,
             smoothstep(0.28, 0.36, u)),
            ("as shares: %s and %s" % (num(self.wn[0], 3), num(self.wn[1], 3)), None,
             smoothstep(0.4, 0.48, u)),
            ("%s × %s + %s × %s" % (num(self.wn[0], 3), num(x[0]), num(self.wn[1], 3),
                                    num(x[1])), NAIVE, smoothstep(0.54, 0.62, u)),
            ("= %s" % num(avg), NAIVE, smoothstep(0.6, 0.68, u)),
            ("error: 1/√(%s + %s)" % (num(raw[0], 3), num(raw[1], 3)), NAIVE,
             smoothstep(0.7, 0.78, u)),
            ("= %s" % num(err), NAIVE, smoothstep(0.7, 0.78, u)),
        ], op, step=0.046)
        self.rows(sc, 0.06, 0.80, [
            ("right when the two errors are independent:", None),
            ("%s independent pairs scatter by %s" % (commas(m["toys"]),
                                                     num(m["independent"]["spread"], 3)),
             None),
        ], op * smoothstep(0.84, 0.92, u), step=0.046)
        self.tag(sc, BOTH, op, "pair 1 of %s, seed %d" % (commas(m["toys"]), m["seed"]))

    def phase_2(self, sc, u, op):
        m = self.meta
        k = min(int(u / 0.6 * 3), 2)
        f = float(np.clip(u / 0.6 * 3 - k, 0, 1)) if u < 0.6 else 1.0
        d = self.shared[k]
        g1, g2 = smoothstep(0.08, 0.45, f), smoothstep(0.5, 0.8, f)
        self.text(sc, 0.06, 0.315, "pair %d: the shared effect is δ = %s" % (k + 1, signed(d)),
                  op, MID, anchor="lm", strong=True)
        at = self.scale(sc, 4.0, 16.0, 0.08, 0.60, 0.60, op, range(4, 17, 2))
        self._truth(sc, at, 0.385, 0.60, op, label=False)
        names = ("measurement 1 = truth + own error (± %s) + %s × δ",
                 "measurement 2 = truth + own error (± %s) + %s × δ")
        for i, ink in enumerate((ONE, TWO)):
            y = 0.435 + 0.095 * i
            self.text(sc, 0.08, y - 0.043, names[i] % (num(self.stat[i], 1),
                                                       num(self.sens[i], 1)),
                      op, anchor="lm", ink=dark(ink, 0.3))
            mid = self.truth + self.sens[i] * d * g1
            self.arrow(sc, (at(self.truth), y), (at(mid), y), SHARED, op * (g1 > 0.03), 1.0, 3.6,
                       head=abs(at(mid) - at(self.truth)) > 0.012)
            end = mid + self.own[k, i] * g2
            self.arrow(sc, (at(mid), y + 0.018), (at(end), y + 0.018), ink, op * (g2 > 0.03),
                       1.0, 2.6, head=False)
            self.spot(sc, at(end), y + 0.018 * (g2 > 0.03), 6.0, ink, op, lift=1.6)
        s1, s2 = self.sig
        self.rows(sc, 0.08, 0.70, [
            ("σ₁ = √(%s² + %s²) = %s" % (num(self.stat[0], 1), num(self.sens[0], 1),
                                         num(s1, 1)), ONE),
            ("σ₂ = √(%s² + %s²) = %s" % (num(self.stat[1], 1), num(self.sens[1], 1),
                                         num(s2, 1)), TWO),
            ("covariance = %s × %s = %s" % (num(self.sens[0], 1), num(self.sens[1], 1),
                                            num(self.cov[0, 1])), SHARED,
             smoothstep(0.62, 0.7, u)),
            ("correlation ρ = %s / (%s × %s) = %s" % (num(self.cov[0, 1]), num(s1, 1),
                                                      num(s2, 1), num(m["rho"])), None,
             smoothstep(0.7, 0.78, u)),
        ], op, step=0.046)
        chart = LossChart(((0.69, 0.93), (0.42, 0.79)), (6.0, 14.0), (2.0, 18.0))
        self.axes(sc, chart, op, "x₁", "x₂", xticks=[(v, "%d" % v) for v in (6, 10, 14)],
                  yticks=[(v, "%d" % v) for v in (2, 6, 10, 14, 18)])
        count = int(3 + 297 * smoothstep(0.5, 0.9, u))
        self.text(sc, 0.81, 0.35, "%d of %s pairs" % (count, commas(m["toys"])), op,
                  strong=True)
        for i in range(count):
            a, b = self.xy[i]
            if 6.0 <= a <= 14.0 and 2.0 <= b <= 18.0:
                self.spot(sc, chart.fx(a), chart.fy(b), 2.6 if i > k else 5.0,
                          soft(TEAL, 0.2) if i > k else DIGIT_INK, 0.85 * op, lift=0.8)
        self.tag(sc, TOYS, op, "seed %d" % m["seed"])

    def phase_3(self, sc, u, op):
        m = self.meta
        k = min(int(np.clip((u - 0.2) / 0.78, 0, 0.999) * self.PAIRS), self.PAIRS - 1)
        x1, x2 = self.xy[k]
        naive, best = self.wn @ self.xy[k], self.wb @ self.xy[k]
        self.text(sc, 0.06, 0.315, "pair %d of %s" % (k + 1, commas(m["toys"])), op, MID,
                  anchor="lm", strong=True)
        at = self.scale(sc, 4.0, 18.0, 0.27, 0.93, 0.63, op, range(4, 19, 2))
        self._truth(sc, at, 0.37, 0.63, op, label=False)
        rows = (("measurement 1", x1, self.sig[0], ONE, 0.0),
                ("measurement 2", x2, self.sig[1], TWO, 0.0),
                ("weights 1/σ²", naive, m["quoted"]["naive"], NAIVE, 0.06),
                ("best weights", best, m["quoted"]["best"], BEST, 0.12))
        for i, (name, v, s, ink, t0) in enumerate(rows):
            y = 0.395 + 0.058 * i
            on = op * smoothstep(t0, t0 + 0.08, u)
            self.text(sc, 0.25, y, name, on, anchor="rm", ink=dark(ink, 0.3))
            self.hbar_err(sc, at(max(v - s, 4.0)), at(min(v + s, 18.0)), y, ink, on, 3.2,
                          dot=at(v), r=5.5)
        up = x2 > x1
        self.rows(sc, 0.06, 0.725, [
            ("smallest error:  w₁ = (σ₂² − cov) / (σ₁² + σ₂² − 2 cov) = (%s − %s) / (%s + "
             "%s − %s) = %s" % (num(self.cov[1, 1]), num(self.cov[0, 1]),
                                num(self.cov[0, 0], 0), num(self.cov[1, 1]),
                                num(2 * self.cov[0, 1]), num(self.wb[0], 3)), None),
            ("w₂ = 1 − w₁ = %s:    %s × %s − %s × %s = %s ± %s"
             % (num(self.wb[1], 3), num(self.wb[0], 3), num(x1), num(-self.wb[1], 3), num(x2),
                num(best), num(m["quoted"]["best"])), BEST),
            ("x₂ is %s x₁: the shared shift was probably %s, so the answer lies %s x₁"
             % (("above", "upward", "below") if up else ("below", "downward", "above")),
             SHARED, smoothstep(0.24, 0.32, u)),
        ], op, step=0.046, size=0.030)
        self.tag(sc, EXACT, op, "arithmetic on seeded pairs")

    def phase_4(self, sc, u, op):
        m = self.meta
        go = float(np.clip((u - 0.06) / 0.8, 0, 1))
        j = max(2, int(round(1 + (len(self.rhos) - 1) * go))) - 1
        zero, here = m["negative_from"], m["rho"]
        ticks = [(0, "0"), (zero, num(zero)), (here, num(here))]
        self.text(sc, 0.06, 0.315, "if the correlation were ρ = %s:  w₁ = %s,  w₂ = %s,  "
                  "error %s" % (num(self.rhos[j]), num(self.w1[j]), num(self.w2[j]),
                                num(self.err[j], 3)), op, MID, anchor="lm", strong=True)
        left = LossChart(((0.09, 0.47), (0.42, 0.79)), (0.0, 0.95), (-0.75, 1.75))
        self.axes(sc, left, op, "correlation ρ", "weights of the best combination",
                  xticks=ticks, yticks=[(v, num(v, 1)) for v in (-0.5, 0, 0.5, 1, 1.5)])
        self.seg(sc, (left.x0, left.fy(0)), (left.x1, left.fy(0)), AXIS, 0.45 * op, 1.6, 0.2)
        right = LossChart(((0.57, 0.93), (0.42, 0.79)), (0.0, 0.95), (0.0, 1.1))
        self.axes(sc, right, op, "correlation ρ", "error of the best combination",
                  xticks=ticks, yticks=[(v, "%g" % v) for v in (0, 0.5, 1.0)])
        self.dash(sc, (right.x0, right.fy(self.sig[0])), (right.x1, right.fy(self.sig[0])),
                  ONE, 0.45 * op, 18, 2.0)
        for chart in (left, right):
            for v in (zero, here):
                self.dash(sc, (chart.fx(v), chart.y0), (chart.fx(v), chart.y1), AXIS,
                          0.4 * op, 16)
        for chart, ys, ink in ((left, self.w1, ONE), (left, self.w2, TWO),
                               (right, self.err, BEST)):
            end = self.curve(sc, chart, self.rhos, ys, ink, op, go, 3.2)
            if end is not None:
                self.spot(sc, end[0], end[1], 5.5, ink, op, lift=1.6, glow=0.4)
        x = left.x0 + 0.014
        self.text(sc, x, left.fy(1.26), "weight of measurement 1", op, anchor="lm",
                  ink=dark(ONE, 0.3))
        self.text(sc, x, left.fy(-0.27), "weight of measurement 2", op, anchor="lm",
                  ink=dark(TWO, 0.3))
        x = right.x0 + 0.014
        self.text(sc, x, right.fy(0.36), "largest where w₂ = 0:", op * smoothstep(0.4, 0.5, u),
                  anchor="lm", ink=dark(BEST, 0.3))
        self.text(sc, x, right.fy(0.24), "measurement 1 alone, %s" % num(self.sig[0], 1),
                  op * smoothstep(0.4, 0.5, u), anchor="lm", ink=dark(BEST, 0.3))
        self.text(sc, x, right.fy(0.08), "this pair of experiments: ρ = %s" % num(here),
                  op * smoothstep(0.76, 0.86, u), anchor="lm")
        self.tag(sc, EXACT, op, "errors %s and %s" % (num(self.sig[0], 1),
                                                      num(self.sig[1], 1)))

    def phase_5(self, sc, u, op):
        m = self.meta
        names = ("measurement 1 alone", "weights 1/σ²", "best weights")
        inks = (ONE, NAIVE, BEST)
        lo, hi = self.hist_range
        x0, x1 = 0.08, 0.50
        bins = self.hist.shape[1]
        top = float(self.hist.max())
        bw = self.span((x1 - x0) / bins)[0]
        cols = (0.625, 0.75, 0.875)
        for c, head in zip(cols, (("scatter", "measured"), ("error", "quoted"),
                                  ("truth inside", "± quoted"))):
            self.text(sc, c, 0.325, head[0], op, strong=True)
            self.text(sc, c, 0.365, head[1], op, strong=True)

        def at(v):
            return x0 + (x1 - x0) * (v - lo) / (hi - lo)
        centres = lo + (np.arange(bins) + 0.5) * (hi - lo) / bins
        for k in range(3):
            base = 0.52 + 0.14 * k
            t0 = 0.04 + 0.18 * k
            on = op * smoothstep(t0, t0 + 0.08, u)
            grow = smoothstep(t0, t0 + 0.2, u)
            self.text(sc, x0, base - 0.105, names[k], on, anchor="lm", ink=dark(inks[k], 0.3))
            self.seg(sc, (x0, base), (x1, base), AXIS, 0.45 * on, 1.4, 0.2)
            for b in range(bins):
                h = 0.085 * self.hist[k, b] / top * grow
                if h < 0.002:
                    continue
                inside = abs(centres[b]) <= self.spread[k]
                self.rect(sc, self.P(at(centres[b]), base - h / 2), bw - 0.6,
                          self.span(0, h)[1], inks[k] if inside else soft(inks[k], 0.5), on,
                          radius=0.5, lift=0.4)
            q = self.quoted[k]
            self.hbar_err(sc, at(-q), at(q), base + 0.018, DIGIT_INK, on * smoothstep(
                t0 + 0.14, t0 + 0.22, u), 2.2, cap=4.0)
            show = op * smoothstep(t0 + 0.16, t0 + 0.26, u)
            bad = k == 1
            for c, s in zip(cols, (num(self.spread[k]), num(q), pct(self.cover[k]))):
                self.box(sc, c, base - 0.04, 92, 32, soft(inks[k], 0.7), show, s,
                         border=inks[k] if bad else None, radius=4, size=MID)
        self.text(sc, (x0 + x1) / 2, 0.86, "estimate − truth;  dark: within its scatter;  "
                  "bar: quoted error", op * smoothstep(0.5, 0.6, u))
        e = m["actual_exact"]
        self.text(sc, 0.06, 0.925, "exact scatter: %s, %s, %s"
                  % (num(e["alone"]), num(e["naive"]), num(e["best"])),
                  op * smoothstep(0.8, 0.9, u), anchor="lm", strong=True)
        self.tag(sc, BOTH, op, "%s pairs, seed %d" % (commas(m["toys"]), m["seed"]))


ANIMATIONS = {
    "error-ellipse": ErrorEllipse,
    "statistical-vs-systematic": StatVsSyst,
    "nuisance-parameters-profiling": Profiling,
    "combining-measurements": Combining,
}

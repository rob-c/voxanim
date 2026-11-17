"""Minimisation and chi-square fitting: animations 124-127 of the "next" catalogue (see
:mod:`voxanim.mech`).

* :class:`ChiSquareFit`: a line through six points; each residual a spring, their
  squared pulls summed into chi-square; the sum as a bowl over (a, b), its bottom the
  least-squares solution, the contour chi-square_min + 1 the errors;
* :class:`ChiSquarePerDof`: one dataset fitted with too simple, the right and too rich a
  model; the chi-square distribution built from pseudo-experiments; fit probabilities;
  what unneeded parameters and wrong error bars do;
* :class:`Minimisers`: gradient descent, Newton's method and the simplex walking on one
  chi-square surface with two bottoms, and the map of which start ends where;
* :class:`BinnedOrNot`: one sample of decay times fitted three ways, and the bias and
  spread of the fitted lifetime as the sample shrinks and the bins widen.

All four draw the experiments of :mod:`voxanim.study_stats_fit`; this module lays them
out. Every phase says, bottom right, what kind of evidence it shows: ``exact`` (a formula
evaluated) or ``pseudo-experiments`` (seeded draws from the stated model, with how many).
"""

from __future__ import annotations

import math

import numpy as np

from . import study_stats_fit as S
from .diagram import draw_image
from .easing import smoothstep
from .losschart import LossChart, sup
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIGIT_INK,
    DIM,
    GREEN,
    OCHRE,
    PAPER,
    RED,
    TEAL,
    VIOLET,
    WHITE,
    Mechanism,
    dark,
    soft,
)

BIG, HEAD, SMALL = 0.036, 0.032, 0.030
GREY = (0.52, 0.54, 0.56)
CHI = "χ²"


def sg(v, fmt="%.2f"):
    """A number with a real minus sign."""
    return (fmt % v).replace("-", "−")


def signed(v, fmt="%.2f"):
    return ("+" if v >= 0 else "−") + fmt % abs(v)


def commas(n):
    return format(int(n), ",")


def pct(v, digits=1):
    return "%.*f%%" % (digits, 100.0 * float(v))


def small(p):
    """A probability: plain when it is not tiny, else as m × 10^n."""
    if p >= 0.001:
        return "%.2f" % p if p >= 0.01 else "%.3f" % p
    e = int(math.floor(math.log10(p)))
    return "%.1f × 10%s" % (p / 10.0 ** e, sup(e))


def bands(values, lo, hi, ink, levels=10, power=0.5, log=False):
    """A surface as bands of colour, paper at ``lo`` deepening to ``ink`` at ``hi``: the
    edges between bands are its contours. Row 0 of ``values`` is drawn at the bottom."""
    v = np.asarray(values, float)
    if log:
        q = (np.log(np.maximum(v, lo)) - math.log(lo)) / (math.log(hi) - math.log(lo))
    else:
        q = (np.clip(v, lo, hi) - lo) / (hi - lo)
    q = np.clip(q, 0.0, 1.0) ** power
    q = np.minimum(np.floor(q * levels), levels - 1) / (levels - 1)
    rgb = PAPER * (1 - 0.82 * q[..., None]) + np.asarray(ink, np.float32) * 0.82 * q[..., None]
    return rgb[::-1].astype(np.float32)


def ellipse(centre, cov, delta=1.0, n=73):
    """Points of the contour chi-square_min + ``delta`` of a paraboloid with that
    covariance matrix."""
    t = np.linspace(0, 2 * np.pi, n)
    L = np.linalg.cholesky(np.asarray(cov, float))
    return np.asarray(centre, float) + math.sqrt(delta) * (L @ np.stack([np.cos(t),
                                                                          np.sin(t)])).T


class Fits(Mechanism):
    """What the four animations share: the evidence tag, data with error bars, curves,
    histograms and banded surfaces on a chart."""

    def tag(self, sc, kind, op):
        """The kind of evidence on screen, bottom right."""
        self.text(sc, 0.95, 0.925, kind, op, anchor="rm", ink=dark(TEAL, 0.25))

    def at(self, chart, x, y, lift=0.6):
        return self.P(chart.fx(float(x)), chart.fy(float(y)), lift=lift)

    def curve(self, sc, chart, xs, ys, ink, op, width=2.6, go=1.0, lift=0.6, clip=False):
        """``ys`` against ``xs`` on ``chart`` as far as fraction ``go`` of the points;
        ``clip`` leaves out what falls outside the chart."""
        if op <= 0.01 or go <= 0:
            return
        n = max(2, int(round(len(xs) * min(float(go), 1.0))))
        run = []
        (xa, xb), (ya, yb) = chart.x_range, chart.y_range
        for x, y in zip(xs[:n], ys[:n]):
            if clip and not (xa <= x <= xb and ya <= y <= yb):
                if len(run) > 1:
                    self.polyline(sc, run, width, ink, op, bias=0.3)
                run = []
                continue
            run.append(self.at(chart, x, y, lift))
        if len(run) > 1:
            self.polyline(sc, run, width, ink, op, bias=0.3)

    def dashed(self, sc, a, b, ink, op, n=14, width=1.4, lift=0.6):
        """A dashed line between two layout points."""
        if op <= 0.01:
            return
        a, b = self.P(*a, lift=lift), self.P(*b, lift=lift)
        for i in range(n):
            self.polyline(sc, [a + (b - a) * (i / n), a + (b - a) * ((i + 0.55) / n)], width,
                          ink, op, bias=0.3)

    def errbars(self, sc, chart, xs, ys, sig, op, ink=DIGIT_INK, r=4.0, upto=None):
        """Points with their error bars."""
        for i, (x, y, s) in enumerate(zip(xs, ys, np.broadcast_to(sig, np.shape(xs)))):
            if upto is not None and i >= upto:
                break
            a, b = self.at(chart, x, y - s, 0.9), self.at(chart, x, y + s, 0.9)
            self.polyline(sc, [a, b], 1.6, ink, op)
            for end in (a, b):
                self.polyline(sc, [self.shift(end, -4, 0), self.shift(end, 4, 0)], 1.6, ink, op)
            self.disk(sc, self.at(chart, x, y, 1.0), r, ink, op, lift=1.2, outline=0.0)

    def bar(self, sc, chart, x0, x1, y0, y1, ink, op, gap=1.0, lift=0.3):
        """A filled rectangle in chart coordinates."""
        fx0, fx1 = chart.fx(float(x0)), chart.fx(float(x1))
        fy0, fy1 = chart.fy(float(y0)), chart.fy(float(y1))
        w, h = self.span(abs(fx1 - fx0), abs(fy1 - fy0))
        if h < 0.4 or op <= 0.01:
            return
        self.rect(sc, self.P((fx0 + fx1) / 2, (fy0 + fy1) / 2), max(w - gap, 0.8), h, ink, op,
                  radius=min(2.0, h / 2), lift=lift)

    def hist(self, sc, chart, edges, heights, ink, op, grow=1.0, base=None):
        base = chart.y_range[0] if base is None else base
        for a, b, h in zip(edges[:-1], edges[1:], heights):
            if h > 0:
                self.bar(sc, chart, a, b, base, base + (h - base) * grow, ink, op)

    def spring(self, sc, a, b, ink, op, width=2.0, coils=5, amp=5.0):
        """A spring between two layout points (one above the other): the same number of
        coils however far it is stretched."""
        a, b = self.P(*a, lift=0.8), self.P(*b, lift=0.8)
        length = float(np.hypot(*(self.px(b) - self.px(a))))
        if length < 5.0:
            self.polyline(sc, [a, b], width, ink, op)
            return
        pts = [a]
        for k in range(1, 2 * coils + 1):
            mid = a + (b - a) * (0.12 + 0.76 * (k - 0.5) / (2 * coils))
            pts.append(self.shift(mid, amp if k % 2 else -amp, 0))
        self.polyline(sc, [*pts, b], width, ink, op, bias=0.3)

    def surface(self, sc, chart, tex, op):
        """A banded texture filling ``chart``'s box, in a thin frame."""
        w, h = self.span(chart.x1 - chart.x0, chart.y1 - chart.y0)
        c = self.P((chart.x0 + chart.x1) / 2, (chart.y0 + chart.y1) / 2)
        self.rect(sc, c, w + 4, h + 4, PAPER, op, border=AXIS, border_w=1.2, radius=2,
                  lift=0.1)
        draw_image(self, sc, c, tex, w, h, op, grid=0)

    def trail(self, sc, chart, pts, ink, op, width=2.6, lift=0.9, dots=True, r=3.4):
        """A path of (x, y) points over a surface."""
        world = [self.at(chart, x, y, lift) for x, y in pts]
        if len(world) > 1:
            self.polyline(sc, world, width, ink, op, bias=0.3)
        if dots:
            for p in world:
                self.disk(sc, p, r, ink, op, lift=1.3, outline=0.0)

    def marker(self, sc, chart, x, y, ink, op, r=6.5):
        """A ringed point on a surface."""
        self.disk(sc, self.at(chart, x, y, 1.2), r, ink, op, ring=WHITE, ring_w=2.0, lift=1.6,
                  outline=0.0)

    def key(self, sc, fx, fy, entries, op, step=0.046, size=SMALL, across=None):
        """A legend: a chip and a name per entry, stacked (or spaced ``across``)."""
        for n, (name, ink) in enumerate(entries):
            x = fx + (across[n] if across else 0.0)
            y = fy + (0.0 if across else n * step)
            self.box(sc, x, y, 14, 14, ink, op, radius=3)
            self.text(sc, x + 0.016, y, name, op, anchor="lm", ink=dark(ink, 0.3), size=size)

    def points(self, sc, chart, xs, ys, ink, op, go=1.0, width=2.6, r=4.2):
        """Measured points joined by straight lines, revealed left to right."""
        n = len(xs)
        shown = int(np.clip(go, 0, 1) * n + 1e-9)
        world = [self.at(chart, x, y, 0.8) for x, y in zip(xs[:shown], ys[:shown])]
        if len(world) > 1:
            self.polyline(sc, world, width, ink, op, bias=0.3)
        for p in world:
            self.disk(sc, p, r, ink, op, lift=1.3, outline=0.0)


# ====================================================================== 124: chi-square fit
class ChiSquareFit(Fits):
    """Six points and a trial line: each residual over its error is a pull, drawn as a
    spring, and the squared pulls add up to chi-square. Moving the line moves a point
    over the (a, b) plane, where chi-square is a bowl; its bottom is the least-squares
    solution, the contour one unit above the bottom gives the errors, and 10,000
    repeated experiments scatter as that contour says."""

    TITLE = "Chi-square as a landscape"
    PHASES = (("one term each", 12.0), ("the sum moves", 11.0), ("the bowl", 10.0),
              ("up by one", 11.0), ("repeat it", 11.0))
    LEFT = ((0.08, 0.48), (0.36, 0.80))
    RIGHT = ((0.585, 0.94), (0.37, 0.80))
    SCATTER = 700

    def setup(self):
        d = self.d = S.fit_data()
        m = self.meta = self.study.metadata
        self.x, self.sig, self.y = S.FIT_X, S.FIT_SIGMA, np.asarray(d["y"], float)
        self.best, self.cov = np.asarray(d["best"], float), np.asarray(d["cov"], float)
        self.sa, self.sb = m["errors"]
        self.path = np.asarray(d["path"], float)
        self.left = LossChart(self.LEFT, (0.0, 7.0), (0.0, 5.0))
        self.plane = LossChart(self.RIGHT, S.FIT_A, S.FIT_B)
        self.tex = bands(d["surface"], float(d["chi_min"]), float(np.max(d["surface"])), TEAL,
                         levels=12)
        # the bottom of the bowl, enlarged: bands one "sigma" wide
        k = 3.2
        self.za = (self.best[0] - k * self.sa, self.best[0] + k * self.sa)
        self.zb = (self.best[1] - k * self.sb, self.best[1] + k * self.sb)
        self.zoom = LossChart(self.RIGHT, self.za, self.zb)
        A, B = np.linspace(*self.za, 161), np.linspace(*self.zb, 161)
        near = S.line_chi2(A[None, :], B[:, None], self.x, self.y, self.sig) - d["chi_min"]
        self.ztex = bands(near, 0.0, 25.0, TEAL, levels=5)
        # the repeats, around the true values
        ta, tb = S.FIT_TRUE
        self.ra = (ta - 3.4 * self.sa, ta + 3.4 * self.sa)
        self.rb = (tb - 3.4 * self.sb, tb + 3.4 * self.sb)
        self.rep = LossChart(self.RIGHT, self.ra, self.rb)
        self.edges = np.linspace(*self.ra, 35)
        self.hist_a = np.histogram(d["toy_a"], self.edges, density=True)[0]
        mid = np.linspace(*self.ra, 121)
        self.gauss = (mid, np.exp(-0.5 * ((mid - ta) / self.sa) ** 2)
                      / (self.sa * math.sqrt(2 * math.pi)))
        self.profile = LossChart(self.LEFT, self.za, (0.0, 6.0))
        self.hchart = LossChart(self.LEFT, self.ra, (0.0, float(self.gauss[1].max()) * 1.15))

    def subtitle_for(self, k):
        m = self.meta
        return [
            "each point pulls on the line: its residual divided by its error σ. Squared and "
            "added up, the six pulls are %s, here %.1f for the trial line y = %.2f + %.2f x"
            % (CHI, m["start_chi"], m["start"][0], m["start"][1]),
            "every line y = a + b x is one point in the plane of intercept a and slope b. "
            "Moving the line moves the point, and %s falls from %.1f to %.2f"
            % (CHI, m["start_chi"], m["chi_min"]),
            "over that plane %s is a bowl. Its bottom is the best fit: a = %.3f, b = %.3f, "
            "the same line the least-squares formulas give without any search"
            % (CHI, m["best"][0], m["best"][1]),
            "the contour where %s is 1 above its minimum gives the errors: a = %.2f ± %.2f, "
            "b = %.3f ± %.3f. The same numbers come from the covariance-matrix formula"
            % (CHI, m["best"][0], m["contour_half_widths"][0], m["best"][1],
               m["contour_half_widths"][1]),
            "%s new datasets from the true line, each fitted: a comes out within its error "
            "of the truth %s of the time. Both a and b fall inside the contour only %s"
            % (commas(m["toys"]), pct(m["within_one_sigma"][0]), pct(m["inside_contour"])),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _data(self, sc, op, a, b, ink, springs=6, hot=None, truth=True, xlabel="x"):
        ch = self.left
        self.axes(sc, ch, op, xlabel, "", xticks=[(v, "%d" % v) for v in range(0, 8, 2)],
                  yticks=[(v, "%d" % v) for v in range(0, 6)])
        if truth:
            t0, t1 = S.FIT_TRUE
            self.curve(sc, ch, [0.0, 7.0], [t0, t0 + 7 * t1], GREY, 0.8 * op, width=1.3)
        self.curve(sc, ch, [0.0, 7.0], [a, a + 7 * b], ink, op, width=3.0)
        for i in range(min(springs, 6)):
            x = self.x[i]
            self.spring(sc, (ch.fx(x), ch.fy(self.y[i])), (ch.fx(x), ch.fy(a + b * x)),
                        AMBER if i == hot else RED, op, width=0.75 / self.sig[i])
        self.errbars(sc, ch, self.x, self.y, self.sig, op)

    def _plane(self, sc, op, chart, tex, note, ticks_a=None, ticks_b=None):
        self.surface(sc, chart, tex, op)
        self.axes(sc, chart, op, "intercept a", "slope b", grid=False, xticks=ticks_a,
                  yticks=ticks_b)
        self.text(sc, chart.x1, chart.y0 - 0.03, note, op, anchor="rm", size=SMALL)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        a, b = self.path[0]
        r, pull, sq = S.line_terms(a, b, self.x, self.y, self.sig)
        t = (u - 0.06) / 0.13
        i = int(np.clip(t, 0, 6))                       # points done so far
        f = t - i if 0 <= t < 6 else 1.0
        self.text(sc, 0.08, 0.315, "a trial line:  y = %.2f + %.2f x" % (a, b), op,
                  anchor="lm", ink=dark(BLUE, 0.25), size=HEAD)
        self._data(sc, op, a, b, BLUE, springs=i + (1 if 0 <= t < 6 else 0),
                   hot=i if 0 <= t < 6 else None)
        self.text(sc, 0.28, 0.905,
                  "springs: stiffer where the error is smaller;  grey: the true line", op,
                  size=SMALL)
        cols = (0.575, 0.675, 0.795, 0.905)
        for x, name in zip(cols, ("point", "residual", "÷ error σ", "squared")):
            self.text(sc, x, 0.36, name, op, size=SMALL, ink=GREY)
        for n in range(6):
            if n > i or (n == i and not 0 <= t < 6):
                continue
            y = 0.415 + 0.054 * n
            live = n == i
            ink = dark(AMBER, 0.35) if live else None
            on = [1.0, 1.0, 1.0] if not live else [smoothstep(0.05, 0.25, f),
                                                   smoothstep(0.3, 0.5, f),
                                                   smoothstep(0.55, 0.75, f)]
            self.text(sc, cols[0], y, "x = %d" % self.x[n], op, size=SMALL, ink=ink)
            self.text(sc, cols[1], y, signed(r[n]), op * on[0], size=SMALL, ink=ink)
            self.text(sc, cols[2], y, "÷ %.1f = %s" % (self.sig[n], signed(pull[n])),
                      op * on[1], size=SMALL, ink=ink)
            self.text(sc, cols[3], y, "%.2f" % sq[n], op * on[2], size=SMALL, strong=True)
        done = i if 0 <= t < 6 else (6 if t >= 6 else 0)
        total = float(sq[:done].sum()) + (float(sq[i]) * smoothstep(0.55, 0.75, f)
                                          if 0 <= t < 6 else 0.0)
        self.polyline(sc, [self.P(0.55, 0.742), self.P(0.94, 0.742)], 1.2, AXIS, 0.8 * op)
        self.text(sc, 0.795, 0.79, "%s = the sum =" % CHI, op, anchor="rm", size=BIG)
        self.text(sc, 0.905, 0.79, "%.2f" % total, op, strong=True, size=BIG)
        self.tag(sc, "exact; data: y = %g + %g x + noise, seed %d"
                 % (S.FIT_TRUE + (S.FIT_SEED,)), op)

    def _where(self, u):
        """The trial line at time ``u`` of phase 2: (a, b) and the segment it is on."""
        s = float(np.clip((u - 0.06) / 0.80, 0, 1)) * 3
        k = int(min(s, 2))
        f = smoothstep(0.08, 0.92, s - k)
        return self.path[k] + (self.path[k + 1] - self.path[k]) * f, k, s

    def phase_2(self, sc, u, op):
        (a, b), k, _ = self._where(u)
        chi = float(S.line_chi2(a, b, self.x, self.y, self.sig))
        self.text(sc, 0.08, 0.315, "y = %.2f + %.2f x" % (a, b), op, anchor="lm",
                  ink=dark(BLUE, 0.25), size=HEAD)
        self.text(sc, 0.48, 0.315, "%s = %.1f" % (CHI, chi), op, anchor="rm", strong=True,
                  size=BIG)
        self._data(sc, op, a, b, BLUE)
        self.text(sc, 0.28, 0.905,
                  ("tilt it: the best slope for this intercept",
                   "shift it: the best intercept for this slope",
                   "both together: down to the bottom")[k], op, size=SMALL)
        self._plane(sc, op, self.plane, self.tex, "each line is one point")
        seen = [self.path[j] for j in range(k + 1)] + [np.array([a, b])]
        self.trail(sc, self.plane, seen, DIGIT_INK, op, dots=False)
        self.marker(sc, self.plane, a, b, BLUE, op)
        self.tag(sc, "exact; darker bands: larger %s" % CHI, op)

    def _rings(self, sc, chart, op, deltas, centre=None, ink=DIGIT_INK, width=1.4, lift=0.9):
        for dl in deltas:
            pts = ellipse(self.best if centre is None else centre, self.cov, dl)
            self.curve(sc, chart, pts[:, 0], pts[:, 1], ink, op, width=width, lift=lift,
                       clip=True)

    def phase_3(self, sc, u, op):
        m = self.meta
        a, b = self.best
        self.text(sc, 0.08, 0.315, "the least-squares line", op, anchor="lm",
                  ink=dark(GREEN, 0.3), size=HEAD)
        self.text(sc, 0.48, 0.315, "%s = %.2f" % (CHI, m["chi_min"]), op, anchor="rm",
                  strong=True, size=BIG)
        self._data(sc, op, a, b, GREEN, xlabel="")
        show = smoothstep(0.35, 0.5, u)
        self.text(sc, 0.08, 0.868, "lowest point of a 121 × 121 grid:  a = %.3f,  b = %.3f"
                  % tuple(m["grid_minimum"]), op * show, anchor="lm", size=SMALL)
        self.text(sc, 0.08, 0.91, "least-squares formulas, exact:  a = %.3f,  b = %.3f"
                  % (a, b), op * smoothstep(0.55, 0.7, u), anchor="lm", size=SMALL,
                  ink=dark(GREEN, 0.3))
        self._plane(sc, op, self.plane, self.tex, "contours of equal %s" % CHI)
        deltas = (1, 10, 30, 60, 100, 150)
        n = int(np.clip((u - 0.08) / 0.4, 0, 1) * len(deltas) + 1e-9)
        self._rings(sc, self.plane, op, deltas[:n])
        ch = self.plane
        on = op * smoothstep(0.5, 0.62, u)
        self.dashed(sc, (ch.fx(a), ch.fy(b)), (ch.fx(a), ch.y1), GREEN, on, n=10, width=1.8)
        self.dashed(sc, (ch.fx(a), ch.fy(b)), (ch.x0, ch.fy(b)), GREEN, on, n=6, width=1.8)
        self.marker(sc, ch, a, b, GREEN, op)
        self.tag(sc, "exact", op)

    def phase_4(self, sc, u, op):
        m = self.meta
        a, b = self.best
        sa, sb = self.sa, self.sb
        # left: chi-square above its minimum along a, the slope re-fitted at each a
        ch = self.profile
        self.text(sc, 0.08, 0.315, "%s above its minimum, along a" % CHI, op, anchor="lm",
                  size=HEAD, strong=True)
        self.axes(sc, ch, op, "", "", xticks=[(a - sa, "%.2f" % (a - sa)), (a, "%.2f" % a),
                                               (a + sa, "%.2f" % (a + sa))],
                  yticks=[(v, "+%d" % v) for v in (0, 1, 2, 4, 6)])
        xs = np.linspace(*self.za, 81)
        self.curve(sc, ch, xs, ((xs - a) / sa) ** 2, TEAL, op, width=3.0,
                   go=smoothstep(0.05, 0.3, u), clip=True)
        on = op * smoothstep(0.32, 0.45, u)
        self.dashed(sc, (ch.x0, ch.fy(1)), (ch.x1, ch.fy(1)), RED, on, n=18, width=1.8)
        for x in (a - sa, a + sa):
            self.dashed(sc, (ch.fx(x), ch.fy(1)), (ch.fx(x), ch.y1), RED, on, n=6, width=1.8)
            self.disk(sc, self.at(ch, x, 1.0, 1.2), 5.0, RED, on, lift=1.4, outline=0.0)
        self.text(sc, ch.fx(a), ch.fy(1) - 0.035, "± %.3f" % m["contour_half_widths"][0], on,
                  ink=dark(RED, 0.3), size=HEAD)
        self.text(sc, 0.08, 0.868, "from the contour:  ± %.3f in a,  ± %.4f in b"
                  % tuple(m["contour_half_widths"]), op * smoothstep(0.55, 0.68, u),
                  anchor="lm", size=SMALL, ink=dark(RED, 0.3))
        self.text(sc, 0.08, 0.91, "covariance formula:  ± %.3f in a,  ± %.4f in b"
                  % (sa, sb), op * smoothstep(0.7, 0.83, u), anchor="lm", size=SMALL)
        # right: the bottom of the bowl, enlarged
        zm = self.zoom
        self._plane(sc, op, zm, self.ztex, "the bottom, enlarged",
                    ticks_a=[(a - sa, "%.2f" % (a - sa)), (a, "%.2f" % a),
                             (a + sa, "%.2f" % (a + sa))],
                    ticks_b=[(b - sb, "%.2f" % (b - sb)), (b, "%.2f" % b),
                             (b + sb, "%.2f" % (b + sb))])
        self._rings(sc, zm, op * smoothstep(0.2, 0.35, u), (1,), ink=RED, width=3.0)
        rho = m["correlation"]
        for x, yb in ((a - sa, b + rho * sb), (a + sa, b - rho * sb)):
            self.dashed(sc, (zm.fx(x), zm.fy(yb)), (zm.fx(x), zm.y1), RED, on, n=9, width=1.6)
        for xa, y in ((a + rho * sa, b + sb), (a - rho * sa, b - sb)):
            self.dashed(sc, (zm.fx(xa), zm.fy(y)), (zm.x0, zm.fy(y)), RED, on, n=9, width=1.6)
        self.marker(sc, zm, a, b, GREEN, op)
        self.box(sc, zm.fx(a + 1.8 * sa), zm.fy(b + 2.4 * sb), 112, 26, PAPER,
                 op * smoothstep(0.2, 0.35, u), "%s min + 1" % CHI, ink=dark(RED, 0.35),
                 border=RED, size=SMALL)
        self.tag(sc, "exact; tilted: a and b are correlated, ρ = %s" % sg(rho), op)

    def phase_5(self, sc, u, op):
        m, d = self.meta, self.d
        ta, tb = S.FIT_TRUE
        sa = self.sa
        go = smoothstep(0.05, 0.6, u)
        # left: the fitted intercepts of all repeats
        ch = self.hchart
        self.text(sc, 0.08, 0.315, "fitted a of %s repeats; true a = %g"
                  % (commas(m["toys"]), ta), op, anchor="lm", size=HEAD, strong=True)
        self.axes(sc, ch, op, "", "", yticks=[],
                  xticks=[(ta - sa, "%.2f" % (ta - sa)), (ta, "%g" % ta),
                          (ta + sa, "%.2f" % (ta + sa))], grid=False)
        self.hist(sc, ch, self.edges, self.hist_a, soft(TEAL, 0.25), op, grow=go)
        self.curve(sc, ch, *self.gauss, DIGIT_INK, op * smoothstep(0.6, 0.72, u), width=2.2)
        for x in (ta - sa, ta + sa):
            self.dashed(sc, (ch.fx(x), ch.y0 + 0.02), (ch.fx(x), ch.y1), RED,
                        op * smoothstep(0.6, 0.72, u), n=12, width=1.8)
        self.text(sc, 0.08, 0.868, "spread of a:  %.3f measured,  %.3f from the formula (curve)"
                  % (m["toy_sd"][0], sa), op * smoothstep(0.62, 0.75, u), anchor="lm",
                  size=SMALL)
        self.text(sc, 0.08, 0.91, "within ± σ of the truth:  %s  (exact %s)"
                  % (pct(m["within_one_sigma"][0]), pct(m["within_exact"])),
                  op * smoothstep(0.72, 0.85, u), anchor="lm", size=SMALL, ink=dark(RED, 0.3))
        # right: every repeat's (a, b), and the contour's shape around the truth
        rp = self.rep
        self.surface(sc, rp, np.broadcast_to(PAPER, (2, 2, 3)).astype(np.float32), op)
        self.axes(sc, rp, op, "intercept a", "slope b", grid=False,
                  xticks=[(ta - 2 * sa, "%.1f" % (ta - 2 * sa)), (ta, "%g" % ta),
                          (ta + 2 * sa, "%.1f" % (ta + 2 * sa))],
                  yticks=[(tb - 2 * self.sb, "%.2f" % (tb - 2 * self.sb)), (tb, "%g" % tb),
                          (tb + 2 * self.sb, "%.2f" % (tb + 2 * self.sb))])
        self.text(sc, rp.x1, rp.y0 - 0.03, "one dot per repeat", op, anchor="rm", size=SMALL)
        shown = int(go * self.SCATTER)
        (xa, xb), (ya, yb) = rp.x_range, rp.y_range
        for x, y, q in zip(d["toy_a"][:shown], d["toy_b"][:shown], d["toy_q"][:shown]):
            if xa < x < xb and ya < y < yb:
                self.disk(sc, self.at(rp, x, y, 0.8), 2.3, RED if q <= 1 else TEAL, 0.85 * op,
                          lift=1.0, outline=0.0)
        self._rings(sc, rp, op * smoothstep(0.6, 0.72, u), (1,), centre=(ta, tb),
                    ink=DIGIT_INK, width=2.8, lift=2.4)
        self.box(sc, rp.x0 + 0.18, rp.fy(tb - 2.75 * self.sb), 280, 26, PAPER,
                 op * smoothstep(0.75, 0.88, u), "inside the contour: %s  (exact %s)"
                 % (pct(m["inside_contour"]), pct(m["inside_contour_exact"])),
                 ink=dark(RED, 0.35), border=DIM, size=SMALL)
        self.tag(sc, "%s: %s from the true line" % (S.TOYS, commas(m["toys"])), op)


# ====================================================================== 125: chi-square / dof
MODEL_INK = (RED, GREEN, VIOLET)
MODEL_NAME = ("straight line", "parabola", "degree 7")
SCALE_INK = (AMBER, GREEN, BLUE)
SCALE_NAME = ("errors claimed half as large", "errors right", "errors claimed twice as large")


class ChiSquarePerDof(Fits):
    """Twelve points from a parabola fitted with a line, a parabola and a polynomial of
    degree 7; the chi-square of the right model over 20,000 pseudo-experiments piling up
    under the exact chi-square curve; each fit's probability as the area beyond its
    chi-square; what each extra parameter buys; and what wrong error bars do."""

    TITLE = "What a good chi-square looks like"
    PHASES = (("models", 11.0), ("repeats", 12.0), ("p-value", 11.0), ("parameters", 10.0),
              ("errors", 11.0))
    DATA = ((0.09, 0.48), (0.36, 0.80))
    HIST = ((0.59, 0.93), (0.37, 0.80))
    STAGES = (100, 1000, S.DOF_TOYS)

    def setup(self):
        d = self.d = S.dof_data()
        m = self.meta = self.study.metadata
        p = self.study.panels
        self.x, self.y = S.DOF_X, np.asarray(d["y"], float)
        self.fine = np.asarray(d["fine"], float)
        self.curves = np.asarray(p[0].data["ys"], float)
        self.chi = [float(v) for v in p[0].data["chi2"]]
        self.dof = [int(v) for v in p[0].data["dof"]]
        self.prob = [float(v) for v in p[0].data["probability"]]
        self.data = LossChart(self.DATA, (0.0, 6.5), (1.4, 5.2))
        toy = np.asarray(d["toy_chi"], float)
        self.right = toy[2]
        # phase 2: the histogram as it fills
        self.wide = np.linspace(0.0, 30.0, 31)
        self.first = np.searchsorted(self.wide, self.right[:S.DOF_SHOWN], side="right") - 1
        self.centres = np.asarray(p[1].data["x"], float)
        self.narrow = np.linspace(0.0, 30.0, 61)
        self.stage = [np.histogram(self.right[:n], self.narrow, density=True)[0]
                      for n in self.STAGES]
        self.exact9 = S.chi2_pdf(self.centres, self.dof[1])
        self.hist_chart = LossChart(self.HIST, (0.0, 30.0), (0.0, 0.125))
        # phase 3: each model's own chart
        self.tails = []
        for k, deg in enumerate(S.DOF_DEGREES):
            hi = 60.0 if k == 0 else 30.0
            edges = np.linspace(0.0, hi, 41)
            xs = np.linspace(0.0, hi, 161)
            pdf = S.chi2_pdf(xs, self.dof[k])
            x0 = 0.08 + 0.305 * k
            self.tails.append(dict(
                chart=LossChart(((x0, x0 + 0.25), (0.42, 0.74)), (0.0, hi),
                                (0.0, float(pdf.max()) * 1.2)),
                edges=edges, hist=np.histogram(toy[deg], edges, density=True)[0], xs=xs,
                pdf=pdf))
        # phase 4
        self.params = np.asarray(p[2].data["x"], float)
        self.chi_all, self.mean_all = (np.asarray(v, float) for v in p[2].data["ys"])
        self.gain = LossChart(((0.10, 0.50), (0.38, 0.80)), (0.5, 9.5), (0.0, 58.0))
        self.mini = LossChart(((0.62, 0.93), (0.36, 0.60)), (0.0, 6.5), (1.4, 5.2))
        self.truth = S.dof_truth(self.fine)
        # phase 5: chi-square per degree of freedom, on a logarithmic axis
        k9 = self.dof[1]
        self.ledges = np.exp(np.linspace(math.log(0.05), math.log(20.0), 53))
        width = math.log(self.ledges[1] / self.ledges[0])
        self.lhist = [np.histogram(self.right / s ** 2 / k9, self.ledges)[0]
                      / (len(self.right) * width) for s in S.DOF_SCALES]
        lx = np.exp(np.linspace(math.log(0.05), math.log(20.0), 161))
        self.lexact = [(lx, lx * k9 * s ** 2 * S.chi2_pdf(lx * k9 * s ** 2, k9))
                       for s in S.DOF_SCALES]
        self.log_chart = LossChart(((0.60, 0.93), (0.45, 0.78)), (0.05, 20.0),
                                   (0.0, float(self.lexact[1][1].max()) * 1.2), xlog=True)
        self.minis = [LossChart(((0.07, 0.255), (0.36 + 0.165 * r, 0.49 + 0.165 * r)),
                                (0.0, 6.5), (1.0, 5.6)) for r in range(3)]
        self.m = m

    def subtitle_for(self, k):
        m = self.meta
        c, n, pr = self.chi, self.dof, self.prob
        return [
            "twelve points fitted three ways. Every fitted parameter uses up one point, so "
            "the degrees of freedom are points minus parameters. %s is %.1f for the line, "
            "%.1f for the parabola, %.1f for degree 7" % (CHI, c[0], c[1], c[2]),
            "what should %s be? Draw new data from the true parabola %s times and fit each: "
            "the values pile up under the exact %s curve for %d degrees of freedom, and "
            "their mean is %.2f" % (CHI, commas(m["toys"]), CHI, n[1], m["toy_mean"]["2"]),
            "the fit probability p is the area of the curve beyond the %s observed: %s for "
            "the line, which is ruled out; %.2f for the parabola and %.2f for degree 7, "
            "both fine" % (CHI, small(pr[0]), pr[1], pr[2]),
            "the parameter the data need lowers %s by %.0f on average; each one after it "
            "by only %.2f, which is noise. And the degree-7 curve ends %.2f σ from the "
            "truth, the parabola %.2f"
            % (CHI, m["gain_line_to_parabola"], m["gain_per_extra_parameter"],
               m["distance_from_truth"]["7"], m["distance_from_truth"]["2"]),
            "error bars half their true size make %s per degree of freedom %.1f on average "
            "instead of %.1f; twice the size, %.2f. One dataset can stray: this one gives "
            "%.2f with the right errors"
            % (CHI, m["reduced_mean"][0], m["reduced_mean"][1], m["reduced_mean"][2],
               m["reduced_this"][1]),
        ][k]

    def _points(self, sc, chart, op, y=None, sig=S.DOF_SIGMA, r=3.8, axes=True):
        if axes:
            self.axes(sc, chart, op, "x", "", xticks=[(v, "%d" % v) for v in range(0, 7, 2)],
                      yticks=[(v, "%d" % v) for v in range(2, 6)])
        self.errbars(sc, chart, self.x, self.y if y is None else y, sig, op, r=r)

    def phase_1(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.09, 0.315, "one dataset: 12 points, each with error σ = %g"
                  % S.DOF_SIGMA, op, anchor="lm", size=HEAD, strong=True)
        cols = (0.555, 0.725, 0.80, 0.868, 0.928)
        for x, name in zip(cols[1:], ("parameters", "dof", CHI, "÷ dof")):
            self.text(sc, x, 0.37, name, op, size=SMALL, ink=GREY)
        for k in range(3):
            go = smoothstep(0.06 + 0.27 * k, 0.24 + 0.27 * k, u)
            self.curve(sc, self.data, self.fine, self.curves[k], MODEL_INK[k], op, width=3.0,
                       go=go, clip=True)
            on = op * smoothstep(0.16 + 0.27 * k, 0.28 + 0.27 * k, u)
            y = 0.445 + 0.085 * k
            pars = S.DOF_DEGREES[k] + 1
            self.box(sc, cols[0], y, 14, 14, MODEL_INK[k], on, radius=3)
            self.text(sc, cols[0] + 0.016, y, MODEL_NAME[k], on, anchor="lm",
                      ink=dark(MODEL_INK[k], 0.3), size=SMALL)
            self.text(sc, cols[1], y, "%d" % pars, on, size=SMALL)
            self.text(sc, cols[2], y, "%d" % self.dof[k], on, size=SMALL)
            self.text(sc, cols[3], y, "%.1f" % self.chi[k], on, size=SMALL, strong=True)
            self.text(sc, cols[4], y, "%.2f" % (self.chi[k] / self.dof[k]), on, size=SMALL)
        self._points(sc, self.data, op)
        end = op * smoothstep(0.82, 0.94, u)
        self.text(sc, 0.745, 0.725, "dof = points − fitted parameters", end, size=HEAD,
                  strong=True)
        self.text(sc, 0.745, 0.775, "for the parabola:  %d − 3 = %d"
                  % (m["points"], self.dof[1]), end, size=SMALL, ink=dark(GREEN, 0.3))
        self.tag(sc, "exact; data: y = 2 + 1.1x − 0.15x² + noise, seed %d" % S.DOF_SEED, op)

    def phase_2(self, sc, u, op):
        m, d = self.meta, self.d
        shown = S.DOF_SHOWN
        t = float(np.clip((u - 0.04) / 0.46, 0, 1)) * shown
        k = int(min(t, shown - 1))
        f = t - k if t < shown else 1.0
        self.text(sc, 0.09, 0.315, "new data %d, fitted" % (k + 1), op, anchor="lm",
                  size=HEAD, strong=True)
        self.text(sc, 0.48, 0.315, "%s = %.1f" % (CHI, self.right[k]), op, anchor="rm",
                  size=HEAD, ink=dark(GREEN, 0.3))
        self.curve(sc, self.data, self.fine, self.truth, GREY, 0.8 * op, width=1.3)
        fit = np.polyval(np.polyfit(self.x, d["toys_y"][k], 2), self.fine)
        self.curve(sc, self.data, self.fine, fit, GREEN, op, width=2.6, clip=True)
        self._points(sc, self.data, op, y=d["toys_y"][k])
        self.text(sc, 0.09, 0.905, "grey: the true parabola", op, anchor="lm", size=SMALL)
        ch = self.hist_chart
        self.axes(sc, ch, op, "%s of the parabola fit" % CHI, "",
                  xticks=[(v, "%d" % v) for v in (0, 9, 20, 30)], yticks=[], grid=False)
        stage = int(np.searchsorted((0.5, 0.62, 0.74), u, side="right"))
        if stage == 0:
            n = k + (1 if f > 0.75 else 0) if t < shown else shown
            counts = np.bincount(self.first[:n], minlength=30)[:30] / float(shown)
            self.hist(sc, ch, self.wide, counts, soft(GREEN, 0.2), op)
            if t < shown:
                b = int(np.clip(self.first[k], 0, 29))
                self.pulse(sc, (0.47, 0.335), (ch.fx(b + 0.5), ch.fy(counts[b]) - 0.01),
                           f / 0.75, GREEN, op, bend=0.05, r=5.5)
            done = n
        else:
            self.hist(sc, ch, self.narrow, self.stage[stage - 1], soft(GREEN, 0.2), op)
            done = self.STAGES[stage - 1]
        self.text(sc, ch.x1, ch.y0 - 0.03, "%s pseudo-experiments" % commas(done), op,
                  anchor="rm", size=HEAD, strong=True)
        end = smoothstep(0.78, 0.92, u)
        self.curve(sc, ch, self.centres, self.exact9, DIGIT_INK, op, width=2.4, go=end)
        on = op * smoothstep(0.88, 0.96, u)
        self.text(sc, ch.fx(21.0), ch.fy(0.115), "the exact %s curve, dof = %d"
                  % (CHI, self.dof[1]), on, size=SMALL)
        self.text(sc, ch.fx(21.0), ch.fy(0.099), "mean of the %s:  %.2f"
                  % (commas(m["toys"]), m["toy_mean"]["2"]), on, size=SMALL,
                  ink=dark(GREEN, 0.3))
        self.tag(sc, "%s from the true parabola; curve: exact" % S.TOYS, op)

    def phase_3(self, sc, u, op):
        for k, t in enumerate(self.tails):
            ch, ink = t["chart"], MODEL_INK[k]
            on = op * smoothstep(0.03 + 0.2 * k, 0.15 + 0.2 * k, u)
            self.text(sc, (ch.x0 + ch.x1) / 2, 0.325, MODEL_NAME[k], on, size=HEAD,
                      ink=dark(ink, 0.3))
            self.text(sc, (ch.x0 + ch.x1) / 2, 0.37, "%d degrees of freedom" % self.dof[k],
                      on, size=SMALL)
            hi = ch.x_range[1]
            self.axes(sc, ch, on, "", "", yticks=[], grid=False,
                      xticks=[(v, "%d" % v) for v in (0, self.dof[k], hi / 2, hi)
                              if v != self.dof[k] or abs(hi / 2 - v) > 4])
            self.hist(sc, ch, t["edges"], t["hist"], soft(ink, 0.45), on)
            self.curve(sc, ch, t["xs"], t["pdf"], DIGIT_INK, on, width=2.2, clip=True)
            seen = self.chi[k]
            mark = on * smoothstep(0.2 + 0.2 * k, 0.3 + 0.2 * k, u)
            tail = t["xs"] >= seen
            for x in t["xs"][tail][::2]:
                self.polyline(sc, [self.at(ch, x, 0.0, 0.9),
                                   self.at(ch, x, float(np.interp(x, t["xs"], t["pdf"])),
                                           0.9)], 2.4, ink, 0.7 * mark)
            self.dashed(sc, (ch.fx(seen), ch.y0 + 0.01), (ch.fx(seen), ch.y1), dark(ink, 0.3),
                        mark, n=10, width=2.2)
            self.text(sc, (ch.x0 + ch.x1) / 2, 0.835, "observed %s = %.1f" % (CHI, seen), mark,
                      size=SMALL)
            self.text(sc, (ch.x0 + ch.x1) / 2, 0.877, "p = %s" % small(self.prob[k]), mark,
                      size=HEAD, strong=True)
        self.tag(sc, "exact curves; bars: %s %s" % (commas(self.meta["toys"]), S.TOYS), op)

    def phase_4(self, sc, u, op):
        m = self.meta
        ch = self.gain
        self.text(sc, 0.10, 0.325, "%s against the number of parameters" % CHI, op,
                  anchor="lm", size=HEAD, strong=True)
        self.axes(sc, ch, op, "fitted parameters", "",
                  xticks=[(v, "%d" % v) for v in range(1, 10)],
                  yticks=[(v, "%d" % v) for v in (0, 9, 20, 40)])
        go = smoothstep(0.05, 0.45, u)
        self.points(sc, ch, self.params, self.mean_all, TEAL, op, go=go, width=3.0)
        for x, y in zip(self.params, self.chi_all):
            if y <= ch.y_range[1]:
                self.disk(sc, self.at(ch, x, y, 1.0), 4.0, DIGIT_INK, op * go, lift=1.4,
                          outline=0.0)
        self.key(sc, 0.30, 0.42, (("mean of %s repeats" % commas(m["toys"]), TEAL),
                                  ("this dataset", DIGIT_INK)), op)
        on = op * smoothstep(0.45, 0.58, u)
        self.text(sc, ch.fx(2.75), ch.fy(27.0), "−%.1f" % m["gain_line_to_parabola"], on,
                  anchor="lm", size=HEAD, ink=dark(GREEN, 0.3))
        self.text(sc, ch.fx(5.2), ch.fy(13.5), "then −%.2f each" %
                  m["gain_per_extra_parameter"], on, anchor="lm", size=HEAD,
                  ink=dark(VIOLET, 0.3))
        # right: where the fitted curve ends up
        mn = self.mini
        self.text(sc, 0.62, 0.325, "the fitted curves and the true one", op, anchor="lm",
                  size=SMALL, strong=True)
        self.curve(sc, mn, self.fine, self.truth, DIGIT_INK, op, width=1.6)
        self.curve(sc, mn, self.fine, self.curves[1], GREEN, op, width=2.6, clip=True)
        self.curve(sc, mn, self.fine, self.curves[2], VIOLET, op, width=2.6, clip=True,
                   go=smoothstep(0.1, 0.4, u))
        self.errbars(sc, mn, self.x, self.y, S.DOF_SIGMA, 0.7 * op, r=2.8, ink=GREY)
        bars = op * smoothstep(0.6, 0.72, u)
        self.text(sc, 0.62, 0.665, "typical distance from the truth, in σ", bars, anchor="lm",
                  size=SMALL)
        for r, (deg, ink) in enumerate(((2, GREEN), (7, VIOLET))):
            v = m["distance_from_truth"][str(deg)]
            self.hbar(sc, 0.69, 0.725 + 0.06 * r, 150 * v, ink, bars,
                      label="%d par." % (deg + 1),
                      value="%.2f  (exact %.2f)" % (v, m["distance_exact"][str(deg)]),
                      u=smoothstep(0.6, 0.85, u))
        self.tag(sc, "%s: %s from the true parabola" % (S.TOYS, commas(m["toys"])), op)

    def phase_5(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.08, 0.315, "the same points, three claims about their errors", op,
                  anchor="lm", size=HEAD, strong=True)
        ch = self.log_chart
        self.text(sc, ch.x0, 0.365, "%s ÷ dof over %s repeats, and its mean"
                  % (CHI, commas(m["toys"])), op, anchor="lm", size=SMALL, strong=True)
        self.axes(sc, ch, op, "%s per degree of freedom" % CHI, "", yticks=[], grid=False,
                  xticks=[(v, "%g" % v) for v in (0.1, 0.25, 1, 4, 10)])
        self.dashed(sc, (ch.fx(1.0), ch.y0 + 0.012), (ch.fx(1.0), ch.y1), AXIS, op, n=14)
        for r, s in enumerate(S.DOF_SCALES):
            order = (1, 0, 2).index(r)                    # the right errors first
            on = op * smoothstep(0.05 + 0.25 * order, 0.2 + 0.25 * order, u)
            ink = SCALE_INK[r]
            mn = self.minis[r]
            self.curve(sc, mn, self.fine, self.curves[1], GREEN, on, width=2.0, clip=True)
            self.errbars(sc, mn, self.x, self.y, S.DOF_SIGMA * s, on, r=3.2,
                         ink=dark(ink, 0.3))
            y = (mn.y0 + mn.y1) / 2
            self.text(sc, 0.275, y - 0.042, SCALE_NAME[r], on, anchor="lm", size=SMALL,
                      ink=dark(ink, 0.35))
            self.text(sc, 0.275, y, "this data: %s ÷ dof = %.2f" % (CHI, m["reduced_this"][r]),
                      on, anchor="lm", size=SMALL)
            share, word = ((m["p_below_5pc"][r], "p < 5%") if r < 2
                           else (m["p_above_95pc"][r], "p > 95%"))
            self.text(sc, 0.275, y + 0.042, "%s in %s of repeats" % (word, pct(share, 0)), on,
                      anchor="lm", size=SMALL)
            self.hist(sc, ch, self.ledges, self.lhist[r], soft(ink, 0.3), on)
            self.curve(sc, ch, *self.lexact[r], dark(ink, 0.25), on, width=2.2, clip=True)
            self.text(sc, ch.fx(m["reduced_mean"][r]), ch.y0 - 0.025,
                      "%.2f" % m["reduced_mean"][r], on, ink=dark(ink, 0.35), size=SMALL)
        self.tag(sc, "%s: %s from the true parabola; curves: exact"
                 % (S.TOYS, commas(m["toys"])), op)


# ====================================================================== 126: minimisers
METHOD_INK = (BLUE, RED, GREEN, VIOLET)
END_INK = (TEAL, AMBER, DIM)


class Minimisers(Fits):
    """A peak's position and width fitted to counts that hold a second, smaller peak:
    chi-square has two bottoms. Gradient descent, Newton's method and the simplex walk
    it from two starts, every evaluation counted; then every start on a grid is
    coloured by where each method ends."""

    TITLE = "How a minimiser walks"
    PHASES = (("two bottoms", 10.0), ("gradient descent", 11.0), ("Newton", 11.0),
              ("simplex", 10.0), ("basins", 13.0))
    DATA = ((0.08, 0.43), (0.37, 0.66))
    MAP = ((0.53, 0.93), (0.37, 0.80))
    VIEW = ((2.5, 10.5), (0.3, 1.7))

    def setup(self):
        d = self.d = S.min_data()
        self.meta = m = self.study.metadata
        self.y, self.err = np.asarray(d["y"], float), np.asarray(d["err"], float)
        self.minima = np.asarray(d["minima"], float)
        self.data = LossChart(self.DATA, (0.0, 12.0), (0.0, 135.0))
        # the part of the surface the walks cross (the basins are mapped over all of it)
        self.map = LossChart(self.MAP, *self.VIEW)
        mu, sg_ = (np.linspace(*r, n) for r, n in zip(self.VIEW, (241, 141)))
        surf = S.peak_chi2(mu[None, :], sg_[:, None], self.y, self.err)
        self.tex = bands(surf, float(surf.min()), float(surf.max()), OCHRE, levels=11,
                         power=1.0, log=True)
        self.fine = np.linspace(0.0, 12.0, 161)
        self.paths = [[np.asarray(d["path_%d_%d" % (n, i)], float) for i in range(4)]
                      for n in range(2)]
        self.calls_at = [[np.asarray(d["at_%d_%d" % (n, i)], int) for i in range(4)]
                         for n in range(2)]
        self.tri = [np.asarray(d["triangles_%d" % n], float) for n in range(2)]
        self.ends = np.asarray(d["ends"], int)
        self.end_tex = [np.asarray(END_INK, np.float32)[self.ends[i]][::-1]
                        for i in range(4)]
        self.share = np.asarray(self.study.panels[3].data["share"], float)
        self.median = np.asarray(self.study.panels[3].data["median_evaluations"], float)
        self.evals, self.steps, self.end = (np.asarray(m[k], int) for k in
                                            ("demo_evaluations", "demo_steps", "demo_end"))

    def subtitle_for(self, k):
        m = self.meta
        best, other = self.minima[0], self.minima[1]
        ev, st = self.evals, self.steps
        return [
            "the model is one peak: fit its position μ and width σ. The data hold a second, "
            "smaller peak, so %s has two bottoms: %.0f with the peak in place, %.0f with "
            "it sitting on the neighbour" % (CHI, best[0], other[0]),
            "gradient descent measures the slope and steps straight downhill: %d steps and "
            "%d evaluations of %s from the first start. From the second it rolls just as "
            "surely into the wrong bottom" % (st[0][0], ev[0][0], CHI),
            "Newton's method also measures the curvature and jumps to where a bowl of that "
            "shape would bottom out: %d steps. From the second start the surface is a "
            "dome, and the jump leaves the map" % st[0][1],
            "the simplex needs no slopes: a triangle flips its worst corner over, "
            "stretching and shrinking as it goes. %d evaluations from the first start; "
            "from the second it too ends in the wrong bottom" % ev[0][2],
            "%s starts each. The best fit is reached from %s by gradient descent, %s by "
            "Newton, %s by the simplex, %s by a variable-metric method, the kind MINUIT "
            "runs. None can tell it is in the wrong bottom"
            % ((commas(m["starts"]),) + tuple(pct(self.share[i, 0], 0) for i in range(4))),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _fit(self, sc, op, mu, sigma, ink):
        ch = self.data
        self.axes(sc, ch, op, "", "", xticks=[(v, "%d" % v) for v in range(0, 13, 4)],
                  yticks=[(v, "%d" % v) for v in (0, 50, 100)])
        self.curve(sc, ch, self.fine, np.minimum(S.peak_model(self.fine, mu, sigma), 135.0),
                   ink, op, width=3.0)
        self.errbars(sc, ch, S.MIN_X, self.y, self.err, op, r=2.6)

    def _map(self, sc, op, note=""):
        ch = self.map
        self.surface(sc, ch, self.tex, op)
        self.axes(sc, ch, op, "position μ", "width σ", grid=False,
                  xticks=[(v, "%d" % v) for v in (4, 6, 8, 10)],
                  yticks=[(v, "%g" % v) for v in (0.5, 1, 1.5)])
        if note:
            self.text(sc, ch.x1, ch.y0 - 0.03, note, op, anchor="rm", size=SMALL)

    def _bottoms(self, sc, op):
        for k, ink in ((0, TEAL), (1, AMBER)):
            self.disk(sc, self.at(self.map, self.minima[k, 1], self.minima[k, 2], 1.0), 7.5,
                      WHITE, op, ring=ink, ring_w=3.0, lift=1.2, outline=0.0)

    def _starts(self, sc, op):
        for n, (mu, sg_) in enumerate(S.MIN_STARTS):
            side = (1, -1)[n]                          # clear of the paths that leave it
            self.box(sc, self.map.fx(mu) + 0.03 * side, self.map.fy(sg_) + 0.045 * side, 22, 22,
                     PAPER, op, "12"[n], border=AXIS, size=SMALL, radius=4)

    def _walk(self, sc, u, op, i, heading, extra):
        """Phases 2 and 3: method ``i`` from the first start, then from the second."""
        ink = METHOD_INK[i]
        self.text(sc, 0.08, 0.315, heading, op, anchor="lm", size=HEAD, ink=dark(ink, 0.3))
        self._map(sc, op, "start 1, then start 2")
        self._bottoms(sc, op)
        self._starts(sc, op)
        n = 0 if u < 0.54 else 1
        t = np.clip((u - (0.06, 0.60)[n]) / (0.40, 0.32)[n], 0, 1)
        path = self.paths[n][i]
        s = float(t) * (len(path) - 1)
        k = int(min(s, len(path) - 2))
        f = smoothstep(0.1, 0.9, s - k)
        now = path[k] + (path[k + 1] - path[k]) * f
        if n == 1:
            self.trail(sc, self.map, self.paths[0][i], ink, 0.55 * op, r=2.6)
        self.trail(sc, self.map, [*path[:k + 1], now], ink, op)
        (xa, xb), (ya, yb) = self.VIEW
        done = k + (1 if s - k > 0.9 or t >= 1 else 0)
        at = path[done]
        if xa <= now[0] <= xb and ya <= now[1] <= yb:
            self.marker(sc, self.map, now[0], now[1], ink, op)
        if S.MIN_BOX[1][0] <= at[1] <= S.MIN_BOX[1][1]:
            self._fit(sc, op, now[0], max(now[1], 0.05), ink)
            chi = "there %s = %.1f" % (CHI, float(S.peak_chi2(at[0], at[1], self.y, self.err)))
        else:
            self._fit(sc, op, path[k][0], path[k][1], ink)
            chi = "a width of %s: off the map, no fit" % sg(at[1])
        self.text(sc, 0.08, 0.74, "start %d:  step %d of %d" % (n + 1, done, len(path) - 1),
                  op, anchor="lm", size=HEAD, strong=True)
        self.text(sc, 0.08, 0.79, "evaluations of %s so far:  %d"
                  % (CHI, self.calls_at[n][i][done]), op, anchor="lm", size=SMALL)
        self.text(sc, 0.08, 0.835, chi, op, anchor="lm", size=SMALL, ink=dark(ink, 0.3))
        self.text(sc, 0.08, 0.885, extra[n], op, anchor="lm", size=SMALL)
        self.tag(sc, "exact; data: 24 Poisson counts, seed %d" % S.MIN_SEED, op)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        best, other = self.minima[0], self.minima[1]
        f = smoothstep(0.38, 0.72, u)
        mu = best[1] + (other[1] - best[1]) * f
        sigma = best[2] + (other[2] - best[2]) * f
        ink = TEAL if f < 0.5 else AMBER
        self.text(sc, 0.08, 0.315, "counts in 24 bins, and a one-peak model", op, anchor="lm",
                  size=HEAD, strong=True)
        self._fit(sc, op, mu, sigma, ink)
        self.text(sc, 0.08, 0.74, "fitted: position μ = %.2f, width σ = %.2f" % (mu, sigma),
                  op, anchor="lm", size=HEAD, ink=dark(ink, 0.3))
        self.text(sc, 0.08, 0.79, "%s = %.0f" % (CHI, float(S.peak_chi2(mu, sigma, self.y,
                                                                         self.err))),
                  op, anchor="lm", size=BIG, strong=True)
        self.text(sc, 0.08, 0.84, "known: background %g, peak height %g"
                  % (S.MIN_BKG, S.MIN_HEIGHT), op, anchor="lm", size=SMALL)
        self.text(sc, 0.08, 0.885, "not in the model: the small peak at %g" % S.MIN_SECOND[1],
                  op, anchor="lm", size=SMALL)
        self._map(sc, op, "%s for every (μ, σ); darker: larger" % CHI)
        self._bottoms(sc, op)
        self.trail(sc, self.map, [best[1:], (mu, sigma)], DIGIT_INK, op * (f > 0.01),
                   dots=False)
        self.marker(sc, self.map, mu, sigma, ink, op)
        for k, word in ((0, "best fit"), (1, "another bottom")):
            on = op * (smoothstep(0.08, 0.2, u) if k == 0 else smoothstep(0.72, 0.84, u))
            self.text(sc, self.map.fx(self.minima[k, 1]), self.map.fy(self.minima[k, 2]) - 0.05,
                      "%s: %.0f" % (word, self.minima[k, 0]), on, size=SMALL, strong=True)
        self.tag(sc, "exact; data: 24 Poisson counts, seed %d" % S.MIN_SEED, op)

    def phase_2(self, sc, u, op):
        g = self.meta["start_gradient"]
        self._walk(sc, u, op, 0, "gradient descent: straight downhill",
                   ("slope at start 1:  %s per unit of μ,  %s per unit of σ"
                    % (signed(g[0], "%.0f"), signed(g[1], "%.0f")),
                    "start 2 lies on the other side of the ridge"))

    def phase_3(self, sc, u, op):
        c = self.meta["start_curvature"]
        self._walk(sc, u, op, 1, "Newton: slope and curvature, then jump",
                   ("curvatures at start 1:  %s and %s, a bowl"
                    % (signed(c[0][0], "%.0f"), signed(c[0][1], "%.0f")),
                    "curvatures at start 2:  %s and %s, a dome"
                    % (signed(c[1][0], "%.0f"), signed(c[1][1], "%.0f"))))

    def phase_4(self, sc, u, op):
        ink = METHOD_INK[2]
        self.text(sc, 0.08, 0.315, "simplex: a triangle that flips and shrinks", op,
                  anchor="lm", size=HEAD, ink=dark(ink, 0.3))
        self._map(sc, op, "start 1, then start 2")
        self._bottoms(sc, op)
        self._starts(sc, op)
        n = 0 if u < 0.54 else 1
        t = float(np.clip((u - (0.05, 0.59)[n]) / (0.43, 0.34)[n], 0, 1))
        tri = self.tri[n]
        k = int(min(t ** 2.2 * len(tri), len(tri) - 1))   # the early, large triangles longer
        if n == 1:
            self.trail(sc, self.map, self.paths[0][2], ink, 0.55 * op, r=2.6)
        for back in (3, 2, 1, 0):
            if k - back < 0:
                continue
            corners = tri[k - back]
            loop = [self.at(self.map, x, y, 0.9) for x, y in [*corners, corners[0]]]
            self.polyline(sc, loop, 3.4 if back == 0 else 2.0, dark(ink, 0.25) if back == 0
                          else ink, op * (1.0, 0.65, 0.45, 0.3)[back], bias=0.3)
        now = tri[k][0]                                  # the best corner
        seen = [self.paths[n][2][0]]
        for q in self.paths[n][2][1:]:
            if any(np.array_equal(q, c[0]) for c in tri[:k + 1]):
                seen.append(q)
        self.trail(sc, self.map, seen, ink, op, dots=False)
        self.marker(sc, self.map, now[0], now[1], ink, op)
        self._fit(sc, op, now[0], now[1], ink)
        self.text(sc, 0.08, 0.74, "start %d:  triangle %d of %d" % (n + 1, k + 1, len(tri)),
                  op, anchor="lm", size=HEAD, strong=True)
        self.text(sc, 0.08, 0.79, "best corner:  %s = %.1f"
                  % (CHI, float(S.peak_chi2(now[0], now[1], self.y, self.err))), op,
                  anchor="lm", size=SMALL, ink=dark(ink, 0.3))
        self.text(sc, 0.08, 0.835, "%d evaluations of %s in all, no slopes"
                  % (self.evals[n][2], CHI), op * smoothstep(0.85, 1.0, t), anchor="lm",
                  size=SMALL)
        self.text(sc, 0.08, 0.885, ("reflect the worst corner through the other two",
                                    "the same rule, the other bottom")[n], op, anchor="lm",
                  size=SMALL)
        self.tag(sc, "exact; data: 24 Poisson counts, seed %d" % S.MIN_SEED, op)

    def phase_5(self, sc, u, op):
        m = self.meta
        self.key(sc, 0.07, 0.315, (("ends at the best fit", TEAL), ("the other bottom", AMBER),
                                   ("an edge, or off the map", GREY)), op,
                 across=(0.0, 0.205, 0.39))
        self.text(sc, 0.94, 0.315, "a cell: one start (μ →, σ ↑)", op, anchor="rm",
                  size=SMALL)
        names = ("gradient descent", "Newton, unguarded", "simplex", "variable metric")
        for i in range(4):
            x = 0.165 + 0.225 * i
            on = op * smoothstep(0.04 + 0.13 * i, 0.16 + 0.13 * i, u)
            self.text(sc, x, 0.37, names[i], on, size=HEAD, ink=dark(METHOD_INK[i], 0.3))
            c = self.P(x, 0.545)
            self.rect(sc, c, 184, 150, PAPER, on, border=METHOD_INK[i], border_w=2.0, radius=3,
                      lift=0.1)
            draw_image(self, sc, c, self.end_tex[i], 178, 144, on, grid=0)
            self.text(sc, x, 0.735, "best fit from %s" % pct(self.share[i, 0], 0), on,
                      size=SMALL, ink=dark(TEAL, 0.35))
            self.text(sc, x, 0.775, "other bottom %s" % pct(self.share[i, 1], 0), on,
                      size=SMALL, ink=dark(AMBER, 0.4))
            self.text(sc, x, 0.815, "median %d evaluations" % self.median[i], on, size=SMALL)
        end = op * smoothstep(0.7, 0.82, u)
        self.text(sc, 0.5, 0.875, "MINUIT's MIGRAD is a refined variable-metric method; its "
                  "SIMPLEX is this triangle method", end, size=SMALL, strong=True)
        self.tag(sc, "exact; %s starts per method, every evaluation counted"
                 % commas(m["starts"]), op)


# ====================================================================== 127: binned or not
FIT_INK = (RED, BLUE, GREEN)
FIT_NAME = ("binned %s" % CHI, "binned likelihood", "unbinned likelihood")


class BinnedOrNot(Fits):
    """120 decay times as ticks and as counts in ten bins; the chi-square's sqrt(n)
    errors, which make a low count pull harder than a high one and drop empty bins; the
    lifetime each fit gives over 4,000 samples; and bias and spread as the sample
    shrinks and as the bins widen."""

    TITLE = "Binned or unbinned?"
    PHASES = (("sample", 10.0), ("√n errors", 12.0), ("repeats", 11.0),
              ("fewer events", 11.0), ("wider bins", 10.0))
    HISTO = ((0.09, 0.58), (0.43, 0.80))
    LEFT = ((0.10, 0.47), (0.41, 0.78))
    RIGHT = ((0.57, 0.93), (0.41, 0.78))

    def setup(self):
        d = self.d = S.bin_data()
        m = self.meta = self.study.metadata
        self.demo = m["demo"]
        self.times = np.asarray(d["demo_times"], float)
        self.order = np.random.default_rng(S.BIN_SEED).permutation(len(self.times))
        self.counts = np.asarray(d["demo_counts"], float)
        self.edges = np.linspace(0.0, S.WINDOW, S.DEMO_BINS + 1)
        self.width = S.WINDOW / S.DEMO_BINS
        self.histo = LossChart(self.HISTO, (0.0, S.WINDOW), (0.0, 60.0))
        self.t = np.linspace(0.0, S.WINDOW, 121)
        self.fitted = [float(v) for v in d["demo_fit"]]
        self.expected = np.asarray(self.study.panels[0].data["expected"], float)
        self.true_curve = S.DEMO_N * self.width * S.decay_pdf(self.t, S.TAU)
        p = self.study.panels
        self.est_edges = np.asarray(p[1].data["edges"], float)
        self.est_hist = np.asarray(p[1].data["ys"], float)
        top = float(self.est_hist.max()) * 1.1
        self.rows = [LossChart(((0.09, 0.60), (0.36 + 0.155 * r, 0.47 + 0.155 * r)),
                               (float(self.est_edges[0]), float(self.est_edges[-1])),
                               (0.0, top)) for r in range(3)]
        self.sizes = np.asarray(p[2].data["x"], float)
        self.size_bias = 100 * np.asarray(p[2].data["ys"], float) / S.TAU
        self.size_ratio = np.asarray(p[2].data["spread"], float) / p[2].data["exact_spread"]
        self.bins = np.asarray(p[3].data["x"], float)
        self.bins_bias = 100 * np.asarray(p[3].data["ys"], float) / S.TAU
        self.bins_ratio = np.asarray(p[3].data["spread"], float) / p[3].data["exact_spread"]

    def subtitle_for(self, k):
        m, demo = self.meta, self.demo
        sb = self.size_bias
        i30, i250 = S.SIZES.index(30), S.SIZES.index(250)
        b2 = S.BINNINGS.index(2)
        return [
            "%d decay times with true lifetime τ = %g μs. A binned fit sees only the ten "
            "counts; an unbinned likelihood uses every time. On this sample: %.2f from "
            "%s, %.2f and %.2f from the likelihoods"
            % (demo["events"], m["tau"], demo["fitted"][0], CHI, demo["fitted"][1],
               demo["fitted"][2]),
            "%s gives each bin the error √n of its own count. A count that came out low "
            "then has a small error and pulls hard; an empty bin has none and drops out. "
            "A likelihood asks how probable each count is" % CHI,
            "%s samples of %d decays. The likelihood fits average %s%% from the true τ; "
            "%s averages %s%%, a third of its own spread, an offset that more samples of "
            "this size never average away"
            % (commas(m["toys"]), demo["events"], signed(100 * demo["bias"][2] / S.TAU, "%.1f"),
               CHI, signed(100 * demo["bias"][0] / S.TAU, "%.1f")),
            "with fewer events per bin %s goes wrong: %s%% at 30 events, where empty bins "
            "drop out, %s%% at 250. The likelihoods stay within %.1f%% from 60 events up, "
            "close to the smallest spread possible"
            % (CHI, signed(sb[0, i30], "%.0f"), signed(sb[0, i250], "%.1f"),
               float(np.abs(sb[1:, 1:]).max())),
            "wide bins give %s enough events and its bias fades, but binning throws "
            "information away: with two bins both binned fits spread %.2f times wider "
            "than the unbinned likelihood" % (CHI, self.bins_ratio[1, b2]
                                              / self.bins_ratio[2, b2]),
        ][k]

    # ------------------------------------------------------------------ phases
    def _histogram(self, sc, op, counts, errors=0.0):
        ch = self.histo
        self.axes(sc, ch, op, "decay time t (μs)", "", grid=False,
                  xticks=[(v, "%d" % v) for v in range(0, 11, 2)],
                  yticks=[(v, "%d" % v) for v in (0, 20, 40, 60)])
        self.hist(sc, ch, self.edges, counts, soft(TEAL, 0.45), op)
        if errors <= 0:
            self.curve(sc, ch, self.t, self.true_curve, GREY, 0.9 * op, width=1.5)
        if errors > 0:
            mids = (self.edges[:-1] + self.edges[1:]) / 2
            full = self.counts > 0
            self.errbars(sc, ch, mids[full], self.counts[full], np.sqrt(self.counts[full]),
                         op * errors, r=3.2)
            for x, n in zip(mids[full], self.counts[full]):
                self.text(sc, ch.fx(x), ch.fy(n + math.sqrt(n)) - 0.028, "±%.1f" % math.sqrt(n),
                          op * errors, size=SMALL, ink=dark(RED, 0.3))

    def phase_1(self, sc, u, op):
        n = len(self.times)
        shown = int(np.clip((u - 0.05) / 0.5, 0, 1) * n + 1e-9)
        ch = self.histo
        self.text(sc, 0.09, 0.315, "%d of %d decay times, drawn with τ = %g μs"
                  % (shown, n, S.TAU), op, anchor="lm", size=HEAD, strong=True)
        picked = self.times[self.order[:shown]]
        for t in picked:
            x = ch.fx(t)
            self.polyline(sc, [self.P(x, 0.36), self.P(x, 0.395)], 1.6, DIGIT_INK, 0.75 * op)
        counts = np.histogram(picked, self.edges)[0]
        self._histogram(sc, op, counts)
        self.text(sc, ch.fx(6.2), ch.fy(50.0), "ten bins: only the counts remain", op,
                  size=SMALL)
        self.text(sc, ch.fx(6.2), ch.fy(41.0), "grey: the true curve", op, size=SMALL, ink=GREY)
        what = ("every time:  ln L = sum of ln f(t; τ)",
                "ten counts, each Poisson around the curve",
                "ten counts, each with error √n")
        for r, k in enumerate((2, 1, 0)):
            on = op * smoothstep(0.15 + 0.12 * r, 0.27 + 0.12 * r, u)
            y = 0.40 + 0.125 * r
            self.box(sc, 0.635, y, 14, 14, FIT_INK[k], on, radius=3)
            self.text(sc, 0.651, y, FIT_NAME[k], on, anchor="lm", size=HEAD,
                      ink=dark(FIT_INK[k], 0.3))
            self.text(sc, 0.635, y + 0.043, what[r], on, anchor="lm", size=SMALL)
            self.text(sc, 0.94, y, "τ = %.2f" % self.fitted[k],
                      op * smoothstep(0.66 + 0.07 * r, 0.76 + 0.07 * r, u), anchor="rm",
                      size=HEAD, strong=True)
        self.text(sc, 0.94, 0.80, "true τ = %.2f" % S.TAU, op * smoothstep(0.88, 0.96, u),
                  anchor="rm", size=HEAD, ink=GREY)
        self.tag(sc, "exact fits of one sample; seed %d" % S.BIN_SEED, op)

    def phase_2(self, sc, u, op):
        ch = self.histo
        self.text(sc, 0.09, 0.315, "%s: each bin's error is √n of its own count" % CHI, op,
                  anchor="lm", size=HEAD, ink=dark(RED, 0.3))
        self._histogram(sc, op, self.counts, errors=smoothstep(0.04, 0.18, u))
        empty = np.flatnonzero(self.counts == 0)
        note = op * smoothstep(0.2, 0.32, u)
        for b in empty:
            x = ch.fx((self.edges[b] + self.edges[b + 1]) / 2)
            self.rect(sc, self.P(x, ch.y1 - 0.02), 30, 16, AMBER, note, border=AMBER,
                      border_w=2.2, radius=4, lift=0.9, fill_op=0.0)
        if len(empty):
            x = ch.fx((self.edges[empty[0]] + self.edges[empty[0] + 1]) / 2)
            self.text(sc, min(x, 0.47), ch.y1 - 0.15, "n = 0: no error, left out", note,
                      size=SMALL, ink=dark(AMBER, 0.4))
            self.arrow(sc, (x, ch.y1 - 0.13), (x, ch.y1 - 0.045), AMBER, note, 1.0, 2.0)
        mu0 = float(self.expected[empty[0]]) if len(empty) else 1.0
        p0 = math.exp(-round(mu0, 1))
        low = (2 - 4) ** 2 / 2
        high = (6 - 4) ** 2 / 6
        p2, p6 = (math.exp(-4) * 4 ** n / math.factorial(n) for n in (2, 6))
        rows = (
            (0.30, 0.40, "a bin where the curve says 4:", HEAD, None, True),
            (0.40, 0.455, "n = 2:   (2 − 4)² ÷ 2  =  %.2f" % low, HEAD, dark(RED, 0.3), False),
            (0.48, 0.505, "n = 6:   (6 − 4)² ÷ 6  =  %.2f" % high, HEAD, dark(RED, 0.3),
             False),
            (0.56, 0.56, "the same miss, %.0f times the cost when low:" % (low / high),
             SMALL, None, False),
            (0.56, 0.60, "the fit settles on a curve that is too low", SMALL, None, False),
            (0.64, 0.675, "the likelihood: how probable is n?", HEAD, dark(BLUE, 0.3), False),
            (0.70, 0.725, "P(2 | 4) = %.3f,   P(6 | 4) = %.3f" % (p2, p6), SMALL, None, False),
            (0.76, 0.78, "P(0 | %.1f) = %.2f: the empty bin still counts" % (mu0, p0), SMALL,
             dark(AMBER, 0.4), False),
        )
        for start, y, s, size, ink, strong in rows:
            self.text(sc, 0.63, y, s, op * smoothstep(start, start + 0.1, u), anchor="lm",
                      size=size, ink=ink, strong=strong)
        self.tag(sc, "exact; one sample, seed %d" % S.BIN_SEED, op)

    def phase_3(self, sc, u, op):
        demo, m = self.demo, self.meta
        self.text(sc, 0.09, 0.315, "fitted τ of %s samples of %d decays"
                  % (commas(m["toys"]), demo["events"]), op, anchor="lm", size=HEAD,
                  strong=True)
        for r, k in enumerate((2, 1, 0)):
            ch, ink = self.rows[r], FIT_INK[k]
            go = smoothstep(0.05 + 0.2 * r, 0.3 + 0.2 * r, u)
            ticks = [(v, "%g" % v) for v in (1.5, 2.2, 3.0)] if r == 2 else []
            self.axes(sc, ch, op, "", "", xticks=ticks, yticks=[], grid=False)
            self.hist(sc, ch, self.est_edges, self.est_hist[k], soft(ink, 0.3), op, grow=go)
            self.dashed(sc, (ch.fx(S.TAU), ch.y0), (ch.fx(S.TAU), ch.y1), DIGIT_INK, op, n=6,
                        width=1.8)
            mean = S.TAU + demo["bias"][k]
            on = op * smoothstep(0.25 + 0.2 * r, 0.35 + 0.2 * r, u)
            self.polyline(sc, [self.P(ch.fx(mean), ch.y0, lift=1.0),
                               self.P(ch.fx(mean), ch.y1, lift=1.0)], 3.0, dark(ink, 0.3), on)
            y = (ch.y0 + ch.y1) / 2
            self.text(sc, 0.64, y - 0.04, FIT_NAME[k], on, anchor="lm", size=HEAD,
                      ink=dark(ink, 0.3))
            self.text(sc, 0.64, y + 0.002, "mean %.3f:  %s%%"
                      % (mean, signed(100 * demo["bias"][k] / S.TAU, "%.1f")), on, anchor="lm",
                      size=SMALL, strong=True)
            self.text(sc, 0.64, y + 0.042, "spread %.3f" % demo["spread"][k], on, anchor="lm",
                      size=SMALL)
        self.text(sc, 0.09, 0.875, "dashed: the true τ = %g μs;  solid: the mean" % S.TAU, op,
                  anchor="lm", size=SMALL)
        self.text(sc, 0.64, 0.835, "smallest spread possible: %.3f" % demo["exact_spread"],
                  op * smoothstep(0.8, 0.92, u), anchor="lm", size=SMALL, ink=GREY)
        self.tag(sc, "%s: %s samples from τ = %g μs, ten bins"
                 % (S.TOYS, commas(m["toys"]), S.TAU), op)

    def _two(self, sc, u, op, xs, bias, ratio, ticks, xlabel, bticks, rticks, pad, log):
        self.key(sc, 0.10, 0.315, list(zip(FIT_NAME, FIT_INK)), op, across=(0.0, 0.17, 0.39))
        for box, vals, title, yt, zero, fmt, pd in (
                (self.LEFT, bias, "bias: mean fitted τ − true τ", bticks, 0.0, "%+d%%", pad[0]),
                (self.RIGHT, ratio, "spread ÷ smallest possible", rticks, 1.0, "%.1f",
                 pad[1])):
            yr = (yt[0] - pd, yt[-1] + pd)
            ch = LossChart(box, (float(min(xs)), float(max(xs))), yr, xlog=log)
            self.text(sc, ch.x0, 0.365, title, op, anchor="lm", size=SMALL, strong=True)
            self.axes(sc, ch, op, xlabel, "", xticks=ticks,
                      yticks=[(v, sg(v, fmt)) for v in yt])
            self.dashed(sc, (ch.x0, ch.fy(zero)), (ch.x1, ch.fy(zero)), DIGIT_INK, op, n=22)
            for k in (2, 1, 0):
                self.points(sc, ch, xs, np.clip(vals[k], *yr), FIT_INK[k], op,
                            go=smoothstep(0.05, 0.6, u), width=2.8 if k == 0 else 2.2,
                            r=4.4 if k == 0 else 3.4)

    def phase_4(self, sc, u, op):
        self._two(sc, u, op, self.sizes, self.size_bias, self.size_ratio,
                  [(v, commas(v)) for v in (30, 120, 500, 2000)], "decays in the sample",
                  (-5, 0, 5, 10), (1.0, 1.5, 2.0), (1.0, 0.1), True)
        self.tag(sc, "%s: %s per point, τ = %g μs; bound: exact"
                 % (S.TOYS, commas(self.meta["toys"]), S.TAU), op)

    def phase_5(self, sc, u, op):
        xs = np.arange(len(self.bins), dtype=float)
        self._two(sc, u, op, xs, self.bins_bias, self.bins_ratio,
                  [(i, "%d" % b) for i, b in enumerate(self.bins)],
                  "bins across the window, %d decays" % S.SIZE_FOR_BINS, (-4, -2, 0, 2),
                  (1.0, 1.1, 1.2, 1.3, 1.4), (1.0, 0.06), False)
        self.tag(sc, "%s: %s per point, τ = %g μs; bound: exact"
                 % (S.TOYS, commas(self.meta["toys"]), S.TAU), op)


ANIMATIONS = {
    "chi-square-fit": ChiSquareFit,
    "chi-square-per-dof": ChiSquarePerDof,
    "minimisers-on-chi-square": Minimisers,
    "binned-vs-unbinned-fits": BinnedOrNot,
}

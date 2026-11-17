"""Machine learning the way a physics analysis uses it: animations 150-153 of the "next"
catalogue (see :mod:`voxanim.mech`).

* :class:`NetworkRegression`: a network with one linear output unit giving back the
  angle a digit was turned by; its resolution and bias;
* :class:`ClassifierScore`: a classifier's score as a variable to cut on and to fit;
* :class:`BoostedRegressionTrees`: trees fitted one after another to what is left;
* :class:`OvertrainingCheck`: the scores of training events over those of held-out ones.

They draw the experiments of :mod:`voxanim.study_phys_apps`. No physics dataset is used:
real handwritten digits stand in for events, which the first phase of each says, with
the physics counterpart. Every phase carries, bottom right, what kind of evidence it
shows (:data:`MEASURED` or :data:`TOYS`).
"""

from __future__ import annotations

import numpy as np

from . import study_phys_apps as S
from .easing import smoothstep
from .losschart import LossChart
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
    Mechanism,
    dark,
    soft,
)

MEASURED = "measured on MNIST"
TOYS = "pseudo-experiments from held-out MNIST"
STAND_IN = "no physics data here: real handwritten digits stand in for events"
BIG, HEAD = 0.036, 0.033
GREY = (0.50, 0.52, 0.54)


def deg(v, sign=True):
    """An angle as a label."""
    return (("%+.1f" if sign else "%g") % float(v)).replace("-", "−") + "°"


def plain(v):
    """A number inside a formula: a negative one in brackets."""
    return "%.1f" % v if v >= 0 else "(−%.1f)" % -v


def pct(v, digits=0):
    return "%.*f%%" % (digits, 100.0 * float(v))


def commas(n):
    return format(int(round(float(n))), ",")


STRETCH = 12.0


def stretch(score):
    """A score on the axis that spreads out the piles at 0 and 1: ln(score / (1 - score)),
    the output unit's value before the sigmoid, kept within plus and minus ``STRETCH``.
    It keeps the order of the scores, so cumulative curves and their gaps are unchanged."""
    p = np.clip(np.asarray(score, float), 1e-300, 1.0)
    with np.errstate(divide="ignore"):
        return np.clip(np.log(p) - np.log1p(-p), -STRETCH, STRETCH)


def between(a, k, w):
    """Row ``k + w`` of ``a``, interpolated between rows ``k`` and ``k + 1``."""
    a = np.asarray(a, float)
    return a[k] + (a[min(k + 1, len(a) - 1)] - a[k]) * w


class Phys(Mechanism):
    """What the four animations share: the evidence tag, charts with bars, points and
    staircases, and legends."""

    def tag(self, sc, kind, op):
        """The kind of evidence on screen, bottom right."""
        self.text(sc, 0.95, 0.925, kind, op, anchor="rm", ink=dark(TEAL, 0.25))

    def stand_in(self, sc, op):
        """The line every first phase carries: what stands in for what."""
        self.text(sc, 0.05, 0.925, STAND_IN, op, anchor="lm", ink=dark(OCHRE, 0.35))

    def pane(self, box, xr, yr, xlog=False, ylog=False):
        return LossChart(box, (float(xr[0]), float(xr[1])), (float(yr[0]), float(yr[1])),
                         xlog=xlog, ylog=ylog)

    def path(self, sc, ch, xs, ys, ink, op, go=1.0, width=2.6, dot=False):
        """``ys`` against ``xs`` through every point, drawn as far as fraction ``go`` of
        them. Returns the layout point it ends at."""
        if op <= 0.01 or go <= 0 or len(xs) < 2:
            return None
        s = min(float(go), 1.0) * (len(xs) - 1)
        k = int(s)
        pts = [(ch.fx(float(xs[i])), ch.fy(float(ys[i]))) for i in range(k + 1)]
        if k < len(xs) - 1:
            a, b = pts[-1], (ch.fx(float(xs[k + 1])), ch.fy(float(ys[k + 1])))
            pts.append((a[0] + (b[0] - a[0]) * (s - k), a[1] + (b[1] - a[1]) * (s - k)))
        if len(pts) > 1:
            self.polyline(sc, [self.P(x, y, 0.9) for x, y in pts], width, ink, op, bias=0.3)
        if dot:
            self.disk(sc, self.P(*pts[-1]), 5.0, ink, op, glow=0.4, lift=1.4, outline=0.0)
        return pts[-1]

    def dashes(self, sc, a, b, ink, op, n=12, width=1.8):
        """A dashed line between two layout points."""
        if op <= 0.01:
            return
        (ax, ay), (bx, by) = a, b
        for i in range(n):
            u0, u1 = i / n, (i + 0.55) / n
            sc.line(self.P(ax + (bx - ax) * u0, ay + (by - ay) * u0, 0.8),
                    self.P(ax + (bx - ax) * u1, ay + (by - ay) * u1, 0.8), width=width,
                    color=ink, opacity=op, bias=0.3)

    def bars(self, sc, ch, edges, values, ink, op, fill=0.6, base=None, grow=1.0):
        """A histogram: a bar per bin from ``base`` (the bottom of the chart) up by
        ``values`` (on a log axis: up to ``values``)."""
        if op <= 0.01:
            return
        lo = ch.y_range[0]
        for i, v in enumerate(values):
            b0 = lo if base is None else float(base[i])
            if ch.ylog:
                if v <= lo:
                    continue
                top = lo * (float(v) / lo) ** grow
            else:
                top = b0 + float(v) * grow
            y_top, y_bot = ch.fy(top), ch.fy(b0)
            if y_bot - y_top < 0.0015:
                continue
            x0, x1 = ch.fx(float(edges[i])), ch.fx(float(edges[i + 1]))
            w, h = self.span(x1 - x0, y_bot - y_top)
            self.rect(sc, self.P((x0 + x1) / 2, (y_top + y_bot) / 2), max(w - 1.0, 1.0), h, ink,
                      op, radius=1.0, lift=0.4, fill_op=fill)

    def dots(self, sc, ch, xs, ys, ink, op, r=3.0, lift=1.1):
        """Points, those outside the chart left out."""
        if op <= 0.01:
            return
        (x0, x1), (y0, y1) = ch.x_range, ch.y_range
        for x, y in zip(xs, ys):
            if x0 <= x <= x1 and y0 <= y <= y1:
                self.disk(sc, self.P(ch.fx(float(x)), ch.fy(float(y))), r, ink, op, lift=lift)

    def key(self, sc, fx, fy, entries, op, step=0.044, size=0.030):
        """A legend: a colour chip and a name per entry, stacked down from ``(fx, fy)``."""
        for n, (name, ink) in enumerate(entries):
            self.box(sc, fx, fy + n * step, 14, 14, ink, op, radius=3)
            self.text(sc, fx + 0.016, fy + n * step, name, op, anchor="lm", ink=dark(ink, 0.3),
                      size=size)

    def vline(self, sc, ch, x, ink, op, width=2.4, dashed=False):
        a, b = (ch.fx(float(x)), ch.y1), (ch.fx(float(x)), ch.y0)
        if dashed:
            self.dashes(sc, a, b, ink, op, width=width)
        else:
            self.polyline(sc, [self.P(*a, 1.0), self.P(*b, 1.0)], width, ink, op, bias=0.3)

    def hline(self, sc, ch, y, ink, op, width=1.6, dashed=True):
        a, b = (ch.x0, ch.fy(float(y))), (ch.x1, ch.fy(float(y)))
        if dashed:
            self.dashes(sc, a, b, ink, op, n=18, width=width)
        else:
            self.polyline(sc, [self.P(*a, 0.6), self.P(*b, 0.6)], width, ink, op, bias=0.3)


# ====================================================================== 150: regression
class NetworkRegression(Phys):
    """Real digits turned by a known angle; a network whose single output unit squashes
    nothing gives the angle back; the squared error of one digit in numbers; predicted
    against true closing in on the diagonal; the residuals narrowing to the resolution;
    and resolution and bias against the true angle."""

    TITLE = "Regression with a network"
    PHASES = (("the task", 9.0), ("one output", 13.0), ("training", 13.0),
              ("residuals", 11.0), ("by true value", 10.0))
    PRED, TRUE, RES, BIAS, TRAIN = BLUE, DIGIT_INK, RED, AMBER, VIOLET
    COLUMNS = (0.44, 0.58, 0.72, 0.86)

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.updates = np.asarray(p0["x"], float)
        self.held, self.train = (np.asarray(v, float) for v in p0["ys"])
        self.per_seed = np.asarray(p0["per_seed"], float)
        self.truth = np.asarray(p1["xy"], float)[:, 0]
        self.preds = np.asarray(p1["snapshots"], float)
        self.images, self.originals = np.asarray(p1["images"]), np.asarray(p1["originals"])
        self.hist, self.edges = np.asarray(p2["values"], float), np.asarray(p2["edges"], float)
        self.width, self.mean = np.asarray(p2["width"], float), np.asarray(p2["mean"], float)
        self.centres = np.asarray(p3["x"], float)
        self.res_bins, self.bias_bins = np.asarray(p3["width"]), np.asarray(p3["bias"])
        self.event = int(m["example"]["event"])
        te = S.reg_heldout()
        self.ev_image, self.ev_cells = te["image"][self.event], te["x"][self.event]
        self.ex = [S.net_example(S.net_start(), self.event),
                   S.net_example(S.net_weights(), self.event)]

    def subtitle_for(self, k):
        m, ex = self.meta, self.ex[1]
        w, e = m["resolution"], m["edge_bias"]
        return [
            "handwritten digits stand in for events: each is turned by a known angle and "
            "read out as 49 cells. A network must give the angle back, as an analysis "
            "regresses a particle's energy from detector cells",
            "the output unit is a plain weighted sum, with no squashing, so any number "
            "can come out. For this digit, turned by y = %s, it ends at ŷ = %s: the "
            "squared error (ŷ − y)² is %.1f" % (deg(ex["truth"]), deg(ex["prediction"]),
                                                ex["loss"]),
            "each update lowers the squared error a little, and the points close in on "
            "the diagonal, where predicted equals true. On %s held-out digits the error "
            "falls from %.1f° to %.1f° (mean of %d seeds)"
            % (commas(S.REG_TEST), m["before_training"]["rms"], m["held_rms"],
               len(m["seeds"])),
            "predicted minus true for every held-out digit: the width σ of this "
            "distribution is the resolution, its mean the bias. σ = %.1f° (%.1f to %.1f "
            "over %d seeds); always answering 0° would give %.1f°"
            % (w["mean"], w["min"], w["max"], len(m["seeds"]), m["guess_zero_rms"]),
            "against the true angle the resolution hardly changes, but the bias does: "
            "%s at −40° and %s at +40°. Near the ends of its range a regression pulls "
            "its answers towards the middle" % (deg(e["lowest_bin"]), deg(e["highest_bin"])),
        ][k]

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        self.lines(sc, 0.05, 0.36, [("a real handwritten digit", None)], op, size=HEAD)
        self.lines(sc, 0.05, 0.555, [("turned anticlockwise by", None),
                                     ("a known angle y: the target", None)],
                   op * smoothstep(0.12, 0.25, u), size=HEAD, step=0.05)
        self.lines(sc, 0.05, 0.77, [("read out as 7 × 7 cells:", None),
                                    ("49 numbers go in", None)],
                   op * smoothstep(0.5, 0.62, u), size=HEAD, step=0.05)
        turn = smoothstep(0.15, 0.5, u)
        for n, fx in enumerate(self.COLUMNS):
            angle = float(self.truth[n])
            self.digit(sc, self.originals[n], fx, 0.36, 80, op)
            now = S.rotate(self.originals[n], angle * turn)[0]
            show = op * smoothstep(0.12, 0.2, u)
            self.digit(sc, now, fx, 0.58, 80, show, frame=soft(self.PRED, 0.6))
            self.text(sc, fx, 0.469, "y = " + deg(angle * turn), show, strong=True, size=0.031)
            read = op * smoothstep(0.52 + 0.04 * n, 0.64 + 0.04 * n, u)
            self.arrow(sc, (fx, 0.668), (fx, 0.708), AXIS, read, width=2.0)
            self.grid(sc, S.cells(now, S.REG_BLOCK).reshape(7, 7), fx, 0.795, 80, 80, read,
                      vmax=1.0, hot=self.PRED)
        self.stand_in(sc, op)
        self.tag(sc, MEASURED, op)

    def _terms(self, sc, ex, op):
        """The largest contributions to the output unit, as signed bars, and their sum."""
        terms = ex["terms"]
        order = np.argsort(terms)
        top = [int(k) for k in (*order[::-1][:3], *order[:3][::-1])]
        scale = 0.105 / max(float(np.abs(terms).max()), 1e-9)
        zero = 0.275
        self.text(sc, 0.05, 0.525, "its 64 inputs: output × weight, in degrees", op,
                  anchor="lm", size=0.030)
        self.polyline(sc, [self.P(zero, 0.548), self.P(zero, 0.795)], 1.2, AXIS, 0.8 * op)
        for row, k in enumerate(top):
            y = 0.572 + 0.041 * row
            v = float(terms[k])
            length = float(np.clip(v * scale, -0.2, 0.2))
            ink = self.RES if v < 0 else self.PRED
            w = max(self.span(abs(length))[0], 1.0)
            self.rect(sc, self.P(zero + length / 2, y), w, 15, ink, op, radius=2)
            self.text(sc, 0.058, y, "unit %d" % (k + 1), op, anchor="lm", size=0.029)
            self.text(sc, 0.47, y, deg(v), op, anchor="rm", ink=dark(ink, 0.3), size=0.029)
        rest = float(terms.sum() - terms[top].sum() + ex["bias"])
        self.text(sc, 0.058, 0.828, "the other %d units and the bias" % (len(terms) - 6), op,
                  anchor="lm", size=0.029)
        self.text(sc, 0.47, 0.828, deg(rest), op, anchor="rm", size=0.029)
        self.polyline(sc, [self.P(0.34, 0.852), self.P(0.47, 0.852)], 1.4, AXIS, op)
        self.text(sc, 0.058, 0.88, "their sum, the prediction", op, anchor="lm", strong=True,
                  size=0.030)
        self.text(sc, 0.47, 0.88, "ŷ = " + deg(ex["prediction"]), op, anchor="rm",
                  ink=dark(self.PRED, 0.3), size=0.031)

    def _loss(self, sc, ex, op, u0, u, label):
        x = 0.58
        up = "raising" if ex["slope"] < 0 else "lowering"
        rows = [
            (label, True, 0.031),
            ("true angle  y = %s" % deg(ex["truth"]), False, 0.031),
            ("loss  L = (ŷ − y)²", True, HEAD),
            ("= (%s − %s)² = %.1f" % (plain(ex["prediction"]), plain(ex["truth"]), ex["loss"]),
             False, HEAD),
            ("slope  dL/dŷ = 2 (ŷ − y) = %s" % deg(ex["slope"]).rstrip("°"), False, 0.031),
            ("so %s ŷ lowers the loss" % up, False, 0.031),
        ]
        for n, (row, strong, size) in enumerate(rows):
            self.text(sc, x, 0.535 + 0.056 * n, row, op * smoothstep(u0 + 0.045 * n,
                                                                    u0 + 0.045 * n + 0.04, u),
                      anchor="lm", strong=strong, size=size)

    def phase_2(self, sc, u, op):
        y = 0.395
        self.digit(sc, self.ev_image, 0.09, y, 70, op, frame=soft(self.PRED, 0.6))
        self.grid(sc, self.ev_cells.reshape(7, 7), 0.205, y, 70, 70, op, vmax=1.0, hot=self.PRED)
        self.box(sc, 0.335, y, 108, 40, soft(GREEN, 0.6), op, "128 units", border=GREEN)
        self.box(sc, 0.475, y, 100, 40, soft(GREEN, 0.6), op, "64 units", border=GREEN)
        self.disk(sc, self.P(0.595, y), 15.0, self.PRED, op, lift=1.0)
        for a, b in ((0.132, 0.162), (0.248, 0.274), (0.396, 0.418), (0.532, 0.572)):
            self.arrow(sc, (a, y), (b, y), AXIS, op, width=2.2)
        self.text(sc, 0.335, y + 0.062, "each: max(0, sum)", op, size=0.029)
        self.text(sc, 0.622, y - 0.062, "output unit: a sum", op, anchor="rm", size=0.029,
                  ink=dark(self.PRED, 0.3))
        after = smoothstep(0.55, 0.60, u)
        fade = (1 - smoothstep(0.50, 0.55, u)) if u < 0.55 else after
        ex = self.ex[1 if u >= 0.55 else 0]
        s = (u * 6) % 1.0 if u < 0.5 else ((u - 0.55) * 6) % 1.0
        self.pulse(sc, (0.132, y), (0.572, y), s, AMBER, op, r=6)
        self.text(sc, 0.64, y - 0.022, "ŷ = " + deg(ex["prediction"]), op * fade, anchor="lm",
                  ink=dark(self.PRED, 0.3), size=0.044)
        self.text(sc, 0.64, y + 0.034, "no squashing: any angle can come out", op, anchor="lm",
                  size=0.029)
        self._terms(sc, ex, op * fade)
        if u < 0.55:
            self._loss(sc, ex, op * fade, 0.10, u, "before any training")
        else:
            self._loss(sc, ex, op * fade, 0.60, u,
                       "after %s updates" % commas(self.updates[-1]))
        self.tag(sc, MEASURED, op)

    def _snap(self, u, lead=0.85):
        s = min(u / lead, 1.0) * (len(self.updates) - 1)
        k = min(int(s), len(self.updates) - 2)
        return k, s - k

    def phase_3(self, sc, u, op):
        k, w = self._snap(u)
        dx, dy = self.frac(130, 130)
        ch = self.pane(((0.235 - dx, 0.235 + dx), (0.585 - dy, 0.585 + dy)), (-55, 55), (-55, 55))
        ticks = [(v, "%d" % v) for v in (-45, 0, 45)]
        self.axes(sc, ch, op, "true angle y (degrees)", "predicted ŷ, %d held-out digits"
                  % len(self.truth), ticks, ticks)
        self.polyline(sc, [self.P(ch.fx(-55), ch.fy(-55), 0.6), self.P(ch.fx(55), ch.fy(55), 0.6)],
                      2.0, self.TRUE, 0.7 * op)
        now = between(self.preds, k, w)
        self.dots(sc, ch, self.truth, now, self.PRED, op, r=2.7)
        e = self.event
        if abs(now[e]) <= 55:
            at = ch.fx(self.truth[e])
            self.polyline(sc, [self.P(at, ch.fy(self.truth[e]), 1.2),
                               self.P(at, ch.fy(now[e]), 1.2)], 2.6, self.RES, op)
            self.disk(sc, self.P(at, ch.fy(now[e])), 5.0, self.PRED, op, ring=AMBER, lift=1.5)
        self.text(sc, 0.235 + dx + 0.012, ch.fy(50), "ŷ = y", op, anchor="lm", size=0.029)
        chart = self.pane(((0.56, 0.93), (0.37, 0.70)), (1, self.updates[-1]), (8, 34), xlog=True)
        self.axes(sc, chart, op, "updates", "root-mean-square error (degrees)",
                  [(v, commas(v)) for v in (1, 10, 100, 1000)],
                  [(v, "%d" % v) for v in (10, 20, 30)])
        go = (k + w - 1) / (len(self.updates) - 2)
        xs = self.updates[1:]
        for row in self.per_seed:
            self.path(sc, chart, xs, row[1:], soft(self.PRED, 0.45), op, go, width=1.2)
        self.path(sc, chart, xs, self.train[1:], self.TRAIN, op, go, width=2.2)
        self.path(sc, chart, xs, self.held[1:], self.PRED, op, go, width=3.0, dot=True)
        self.key(sc, 0.76, 0.40, [("held-out digits", self.PRED), ("training digits", self.TRAIN)],
                 op)
        count = between(self.updates, k, w)
        self.text(sc, 0.56, 0.835, "after %s updates" % commas(count), op, anchor="lm",
                  strong=True, size=BIG)
        self.text(sc, 0.56, 0.885, "held-out error %.1f°" % between(self.held, k, w), op,
                  anchor="lm", ink=dark(self.PRED, 0.3), size=HEAD)
        self.tag(sc, MEASURED, op)

    def phase_4(self, sc, u, op):
        k, w = self._snap(u, 0.7)
        hist = between(self.hist, k, w)
        ch = self.pane(((0.09, 0.58), (0.36, 0.78)), (-60, 60), (0, float(self.hist.max()) * 1.12))
        self.axes(sc, ch, op, "predicted minus true, ŷ − y (degrees)", "share of held-out digits",
                  [(v, "%d" % v) for v in (-60, -30, 0, 30, 60)],
                  [(v, pct(v)) for v in (0.05, 0.10)])
        self.bars(sc, ch, self.edges, hist, self.RES, op, fill=0.75)
        mean, sigma = between(self.mean, k, w), between(self.width, k, w)
        self.vline(sc, ch, mean, self.BIAS, op, width=2.8)
        top = ch.y0 + 0.022
        a, b = ch.fx(mean - sigma), ch.fx(mean + sigma)
        self.polyline(sc, [self.P(a, top + 0.012, 1.0), self.P(a, top, 1.0), self.P(b, top, 1.0),
                           self.P(b, top + 0.012, 1.0)], 2.6, dark(self.RES, 0.25), op)
        self.text(sc, b + 0.012, top + 0.004, "σ each side", op, anchor="lm",
                  ink=dark(self.RES, 0.3), size=0.029)
        x = 0.645
        self.text(sc, x, 0.37, "after %s updates" % commas(between(self.updates, k, w)), op,
                  anchor="lm", strong=True, size=BIG)
        self.text(sc, x, 0.45, "resolution  σ = %.1f°" % sigma, op, anchor="lm",
                  ink=dark(self.RES, 0.3), size=BIG)
        self.text(sc, x, 0.50, "the width of the distribution", op, anchor="lm", size=0.029)
        self.text(sc, x, 0.58, "bias = %s" % deg(mean), op, anchor="lm",
                  ink=dark(self.BIAS, 0.35), size=BIG)
        self.text(sc, x, 0.63, "its mean: zero if unbiased", op, anchor="lm", size=0.029)
        end = op * smoothstep(0.74, 0.84, u)
        r = self.meta["resolution"]
        self.lines(sc, x, 0.73, ["%d seeds: σ = %.1f°" % (len(self.meta["seeds"]), r["mean"]),
                                 "(from %.1f° to %.1f°)" % (r["min"], r["max"]),
                                 "always answering 0°: %.1f°" % self.meta["guess_zero_rms"]],
                   end, size=0.031, step=0.046)
        self.tag(sc, MEASURED, op)

    def phase_5(self, sc, u, op):
        ch = self.pane(((0.09, 0.58), (0.36, 0.78)), (-45, 45), (-14, 14))
        self.axes(sc, ch, op, "true angle y (degrees)", "degrees",
                  [(v, "%d" % v) for v in (-40, -20, 0, 20, 40)],
                  [(v, "%d" % v) for v in (-10, -5, 0, 5, 10)])
        self.hline(sc, ch, 0.0, AXIS, op, dashed=False, width=1.4)
        go_r, go_b = smoothstep(0.05, 0.35, u), smoothstep(0.38, 0.7, u)
        for row in self.res_bins:
            self.dots(sc, ch, self.centres, row, soft(self.RES, 0.35), op * go_r, r=2.6)
        self.path(sc, ch, self.centres, self.res_bins.mean(0), self.RES, op, go_r, width=3.0)
        for row in self.bias_bins:
            self.dots(sc, ch, self.centres, row, soft(self.BIAS, 0.3), op * go_b, r=2.6)
        self.path(sc, ch, self.centres, self.bias_bins.mean(0), self.BIAS, op, go_b, width=3.0)
        x = 0.645
        e = self.meta["edge_bias"]
        mean = self.res_bins.mean(0)
        self.text(sc, x, 0.37, "resolution σ", op * go_r, anchor="lm", ink=dark(self.RES, 0.3),
                  size=BIG)
        self.text(sc, x, 0.42, "%.1f° to %.1f° across the range" % (mean.min(), mean.max()),
                  op * go_r, anchor="lm", size=0.030)
        self.text(sc, x, 0.51, "bias", op * go_b, anchor="lm", ink=dark(self.BIAS, 0.35),
                  size=BIG)
        self.lines(sc, x, 0.56, ["%s where y is near −40°" % deg(e["lowest_bin"]),
                                  "%s in the middle" % deg(e["middle_bin"]),
                                  "%s where y is near +40°" % deg(e["highest_bin"])],
                   op * go_b, size=0.030, step=0.044)
        self.lines(sc, x, 0.735, ["answers are pulled towards", "the middle of the range",
                                  "(dots: each of %d seeds)" % len(self.res_bins)],
                   op * smoothstep(0.7, 0.82, u), size=0.030, step=0.044)
        self.tag(sc, MEASURED, op)


# ====================================================================== 151: the score
class ClassifierScore(Phys):
    """The 4s as signal and the 9s as background: the score a network gives held-out
    digits; a cut on it and what the cut keeps; the ROC curve the cut traces; the best
    cut once background is fifty times more common; and the signal fraction of a
    mixture fitted from the two score shapes."""

    TITLE = "A classifier's score as a variable"
    PHASES = (("two classes", 8.0), ("the score", 8.0), ("a cut", 12.0), ("ROC", 8.0),
              ("1 : 50", 11.0), ("fit the shape", 12.0))
    SIG, BKG, CUT, FIT = RED, BLUE, AMBER, GREEN
    BOX = ((0.09, 0.58), (0.37, 0.78))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.hist, self.edges = np.asarray(p0["values"], float), np.asarray(p0["edges"], float)
        self.images, self.scores = np.asarray(p0["images"]), np.asarray(p0["example_scores"])
        self.cuts = np.asarray(p1["x"], float)
        self.eff_s, self.eff_b = np.asarray(p1["per_seed_signal"]), np.asarray(
            p1["per_seed_background"])
        self.purity, self.z, self.purity_even, self.z_even = np.asarray(p2["ys"], float)
        self.n_s, self.n_b = m["heldout"]["signal"], m["heldout"]["background"]
        self.best = [int(np.argmax(self.z_even)), int(np.argmax(self.z))]
        self.half = int(np.flatnonzero(self.cuts == 0.5)[0])
        k = S.FIT_SHOWN
        self.f_true = S.FIT_FRACTIONS[k]
        self.counts = np.asarray(p3["first"], float)[k]
        self.shape_s, self.shape_b = np.asarray(p3["signal_shape"]), np.asarray(
            p3["background_shape"])
        self.fits = np.asarray(p3["fitted_shown"], float)
        self.fit, self.fit_err = float(self.fits[0]), float(np.asarray(p3["error_shown"])[0])
        self.fgrid = np.linspace(self.fit - 3.5 * self.fit_err, self.fit + 3.5 * self.fit_err, 61)
        lnl = S.log_likelihood(self.counts, self.shape_s, self.shape_b, self.fgrid)
        self.lnl = lnl - S.log_likelihood(self.counts, self.shape_s, self.shape_b,
                                          [self.fit])[0]
        self.toys = (float(p3["values"][k]), float(p3["spread"][k]), float(p3["error"][k]))
        self.fit_edges = np.linspace(self.f_true - 0.07, self.f_true + 0.07, 29)
        self.fit_hist = np.histogram(self.fits, self.fit_edges)[0] / len(self.fits)
        self.k = k

    def subtitle_for(self, k):
        m = self.meta
        c, b, f = m["cut_at_half"], m["best_cut"], m["fit"]
        return [
            "handwritten digits stand in for events: the 4s are the signal, the 9s the "
            "background, a pair easily confused. A network reads 16 cells and gives a "
            "score, as an analysis scores collisions as signal-like or not",
            "the score of %s held-out 4s and %s held-out 9s, which the network never "
            "saw: signal piles up near 1 and background near 0, but each has a tail "
            "under the other" % (commas(self.n_s), commas(self.n_b)),
            "a cut keeps events scoring above it. At 0.5 it keeps %s of the signal "
            "(efficiency) and rejects %s of the background; with equal numbers of each, "
            "%s of what is kept is signal (purity)"
            % (pct(c["signal_efficiency"][0]), pct(c["background_rejection"][0]),
               pct(c["purity_even"][0])),
            "sliding the cut from 1 down to 0 traces the ROC curve: signal kept against "
            "background kept. The area under it is %.3f (%.3f to %.3f over %d seeds); "
            "0.5 would be guessing" % (m["auc"]["mean"], m["auc"]["min"], m["auc"]["max"],
                                       len(m["seeds"])),
            "with %d background events for each signal event, the cut at 0.5 is only "
            "%s pure. The cut with the highest s / √(s + b) moves up to %.2f: %s pure, "
            "and it keeps %s of the signal"
            % (m["ratio"], pct(c["purity_imbalanced"][0]), b["imbalanced"][0],
               pct(b["purity_imbalanced"][0]), pct(b["signal_efficiency_imbalanced"][0])),
            "no cut: fit the mixture's score histogram as f × signal shape + (1 − f) × "
            "background shape. %s samples with true f = %.2f give %.3f ± %.3f; the "
            "likelihood's own error is %.3f"
            % (commas(f["toys"]), self.f_true, *self.toys),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _scores(self, sc, op, box=None, grow=1.0, title=True):
        ch = self.pane(box or self.BOX, (0, 1), (0, 0.48))
        self.axes(sc, ch, op, "score" if title else "", "share of each class",
                  [(v, "%g" % v) for v in (0, 0.25, 0.5, 0.75, 1)],
                  [(v, pct(v)) for v in (0.2, 0.4)])
        self.bars(sc, ch, self.edges, self.hist[1], self.BKG, op, fill=0.62, grow=grow)
        self.bars(sc, ch, self.edges, self.hist[0], self.SIG, op, fill=0.62, grow=grow)
        return ch

    def _cut(self, sc, ch, cut, op, label=True):
        self.vline(sc, ch, cut, self.CUT, op, width=3.2)
        x0, x1 = ch.fx(cut), ch.x1
        w, h = self.span(x1 - x0, ch.y1 - ch.y0)
        if w > 1:
            self.rect(sc, self.P((x0 + x1) / 2, (ch.y0 + ch.y1) / 2), w, h, self.CUT, op,
                      radius=1.0, lift=0.2, fill_op=0.13)
        if label:
            right = cut < 0.62
            self.text(sc, x0 + (0.01 if right else -0.01), ch.y0 + 0.03, "cut at %.2f" % cut,
                      op, anchor="lm" if right else "rm", ink=dark(self.CUT, 0.45), size=0.030)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        groups = (("signal: the 4s", self.SIG, 0.06), ("background: the 9s", self.BKG, 0.54))
        for g, (name, ink, x0) in enumerate(groups):
            self.text(sc, x0 + 0.20, 0.32, name, op, ink=dark(ink, 0.3), size=BIG)
            for n in range(4):
                fx = x0 + 0.05 + 0.10 * n
                self.digit(sc, self.images[g, n], fx, 0.435, 78, op, frame=soft(ink, 0.5))
                read = op * smoothstep(0.15 + 0.03 * n, 0.3 + 0.03 * n, u)
                self.arrow(sc, (fx, 0.52), (fx, 0.545), AXIS, read, width=2.0)
                self.grid(sc, S.cells(self.images[g, n], S.CLS_BLOCK).reshape(4, 4), fx, 0.63,
                          78, 78, read, vmax=1.0, hot=ink)
                out = op * smoothstep(0.5 + 0.03 * n, 0.62 + 0.03 * n, u)
                self.arrow(sc, (fx, 0.715), (fx, 0.745), AXIS, out, width=2.0)
                self.text(sc, fx, 0.785, "%.2f" % self.scores[g, n], out, ink=dark(ink, 0.3),
                          size=BIG)
        self.text(sc, 0.5, 0.855, "each read out as 4 × 4 cells; a network turns the 16 "
                  "numbers into one score between 0 and 1", op * smoothstep(0.45, 0.6, u),
                  size=0.031)
        self.stand_in(sc, op)
        self.tag(sc, MEASURED, op)

    def phase_2(self, sc, u, op):
        ch = self._scores(sc, op, ((0.10, 0.66), (0.36, 0.78)), grow=smoothstep(0.05, 0.6, u))
        del ch
        x = 0.71
        self.key(sc, x, 0.39, [("signal: %s held-out 4s" % commas(self.n_s), self.SIG),
                               ("background: %s 9s" % commas(self.n_b), self.BKG)], op,
                 step=0.05)
        self.lines(sc, x - 0.008, 0.53, ["score near 1: looks like a 4", "score near 0: looks "
                                         "like a 9", "in between: it cannot tell"],
                   op * smoothstep(0.3, 0.5, u), size=0.030, step=0.046)
        self.lines(sc, x - 0.008, 0.72, ["one number per event,", "to cut on or to fit"],
                   op * smoothstep(0.5, 0.7, u), size=0.031, step=0.046)
        self.tag(sc, MEASURED, op)

    def _numbers(self, sc, i, op, show=(1, 1, 1)):
        """What the cut ``CUTS[i]`` keeps, in events."""
        x = 0.635
        es, eb = float(self.eff_s[0, i]), float(self.eff_b[0, i])
        ks, kb = es * self.n_s, eb * self.n_b
        self.text(sc, x, 0.37, "keep events with score > %.2f" % self.cuts[i], op, anchor="lm",
                  strong=True, size=HEAD)
        self.lines(sc, x, 0.45, [("signal efficiency", dark(self.SIG, 0.3)),
                                 ("= %s of %s kept = %s" % (commas(ks), commas(self.n_s),
                                                            pct(es)), None)],
                   op * show[0], size=0.031, step=0.044)
        self.lines(sc, x, 0.58, [("background rejection", dark(self.BKG, 0.3)),
                                 ("= %s of %s removed = %s" % (commas(self.n_b - kb),
                                                               commas(self.n_b), pct(1 - eb)),
                                  None)],
                   op * show[1], size=0.031, step=0.044)
        s, b = 1000 * es, 1000 * eb
        self.lines(sc, x, 0.71, [("purity  s / (s + b), 1,000 of each", dark(self.CUT, 0.45)),
                                 ("= %.0f / (%.0f + %.0f) = %s" % (s, s, b, pct(s / max(s + b,
                                                                                    1e-9))),
                                  None)],
                   op * show[2], size=0.031, step=0.044)

    def phase_3(self, sc, u, op):
        ch = self._scores(sc, op)
        if u < 0.5:
            cut = 0.5
        else:                                           # up to 0.9, down to 0.2, back
            s = (u - 0.5) / 0.45
            cut = float(np.interp(s, [0, 0.3, 0.75, 1.0], [0.5, 0.9, 0.2, 0.5]))
        i = int(round(min(max(cut, 0.0), 1.0) * 100))
        self._cut(sc, ch, self.cuts[i], op * smoothstep(0.03, 0.1, u))
        self._numbers(sc, i, op, [smoothstep(a, a + 0.08, u) for a in (0.1, 0.2, 0.3)])
        self.tag(sc, MEASURED, op)

    def phase_4(self, sc, u, op):
        s = smoothstep(0.05, 0.75, u)
        i = int(round((1 - s) * 100))
        ch = self._scores(sc, op, ((0.08, 0.44), (0.40, 0.72)))
        self._cut(sc, ch, self.cuts[i], op)
        dx, dy = self.frac(112, 112)
        roc = self.pane(((0.70 - dx, 0.70 + dx), (0.575 - dy, 0.575 + dy)), (0, 1), (0, 1))
        ticks = [(v, "%g" % v) for v in (0, 0.5, 1)]
        self.axes(sc, roc, op, "share of background kept", "share of signal kept", ticks, ticks)
        self.dashes(sc, (roc.x0, roc.y1), (roc.x1, roc.y0), GREY, 0.7 * op, n=16)
        seeds = op * smoothstep(0.78, 0.9, u)
        for k in range(1, len(self.eff_s)):
            self.path(sc, roc, self.eff_b[k][::-1], self.eff_s[k][::-1], soft(self.FIT, 0.35),
                      seeds, width=1.4)
        xs, ys = self.eff_b[0][::-1], self.eff_s[0][::-1]
        self.path(sc, roc, xs, ys, self.FIT, op, s, width=3.2)
        at = (roc.fx(self.eff_b[0, i]), roc.fy(self.eff_s[0, i]))
        self.disk(sc, self.P(*at), 6.0, self.CUT, op, glow=0.4, lift=1.5)
        self.text(sc, 0.26, 0.845, "kept: %s of the signal, %s of the background"
                  % (pct(self.eff_s[0, i]), pct(self.eff_b[0, i])), op, strong=True, size=0.031)
        a = self.meta["auc"]
        self.text(sc, 0.70 + dx + 0.015, 0.50, "area", seeds, anchor="lm", size=0.030)
        self.text(sc, 0.70 + dx + 0.015, 0.545, "%.3f" % a["mean"], seeds, anchor="lm",
                  strong=True, size=BIG)
        self.text(sc, 0.70 + dx + 0.015, 0.62, "guessing:", op, anchor="lm", size=0.029)
        self.text(sc, 0.70 + dx + 0.015, 0.66, "the diagonal", op, anchor="lm", size=0.029)
        self.tag(sc, MEASURED, op)

    def phase_5(self, sc, u, op):
        ratio = self.meta["ratio"]
        ch = self.pane(((0.09, 0.47), (0.39, 0.74)), (0, 1), (3, 60000), ylog=True)
        self.axes(sc, ch, op, "score", "events (log scale)",
                  [(v, "%g" % v) for v in (0, 0.5, 1)],
                  [(v, commas(v)) for v in (10, 100, 1000, 10000)])
        self.bars(sc, ch, self.edges, self.hist[1] * 1000 * ratio, self.BKG, op, fill=0.62)
        self.bars(sc, ch, self.edges, self.hist[0] * 1000, self.SIG, op, fill=0.7)
        move = smoothstep(0.35, 0.7, u)
        a, b = self.best
        i = int(round(a + (b - a) * move))
        self._cut(sc, ch, self.cuts[i], op)
        self.text(sc, 0.28, 0.32, "1,000 signal events among %s background"
                  % commas(1000 * ratio), op, strong=True, size=0.031)
        zc = self.pane(((0.60, 0.93), (0.37, 0.66)), (0, 1), (0, 32))
        self.axes(sc, zc, op, "cut", "Z = s / √(s + b)", [(v, "%g" % v) for v in (0, 0.5, 1)],
                  [(v, "%d" % v) for v in (0, 10, 20, 30)])
        self.path(sc, zc, self.cuts[:-1], self.z_even[:-1], GREY, op, smoothstep(0.02, 0.3, u),
                  width=2.4)
        self.path(sc, zc, self.cuts[:-1], self.z[:-1], self.CUT, op, smoothstep(0.1, 0.4, u),
                  width=3.2)
        self.disk(sc, self.P(zc.fx(self.cuts[i]), zc.fy(self.z[i])), 6.0, self.CUT, op, glow=0.4,
                  lift=1.5)
        self.text(sc, zc.fx(0.30), zc.fy(self.z_even[30]) - 0.03, "1 : 1", op, ink=GREY,
                  size=0.029)
        self.text(sc, zc.fx(0.45), zc.fy(self.z[45]) + 0.04, "1 : %d" % ratio, op,
                  ink=dark(self.CUT, 0.4), size=0.029)
        es, eb = float(self.eff_s[0, i]), float(self.eff_b[0, i])
        s, b = 1000 * es, 1000 * ratio * eb
        self.text(sc, 0.60, 0.80, "cut at %.2f:  s = %s,  b = %s" % (self.cuts[i], commas(s),
                                                                     commas(b)), op,
                  anchor="lm", strong=True, size=0.031)
        self.text(sc, 0.60, 0.85, "purity %s,   Z = %.1f" % (pct(s / max(s + b, 1e-9)),
                                                            self.z[i]), op, anchor="lm",
                  ink=dark(self.CUT, 0.45), size=HEAD)
        self.tag(sc, MEASURED, op)

    def phase_6(self, sc, u, op):
        n = S.FIT_EVENTS
        edges = np.linspace(0, 1, S.FIT_BINS + 1)
        mid = (edges[:-1] + edges[1:]) / 2
        ch = self.pane(((0.08, 0.36), (0.42, 0.74)), (0, 1), (0, float(self.counts.max()) * 1.15))
        self.axes(sc, ch, op, "score", "events in one sample of %s" % commas(n),
                  [(v, "%g" % v) for v in (0, 0.5, 1)], [(v, "%d" % v) for v in (100, 200, 300)])
        fit = op * smoothstep(0.12, 0.3, u)
        f = 0.5 + (self.fit - 0.5) * smoothstep(0.12, 0.42, u)       # the fit settling
        self.bars(sc, ch, edges, (1 - f) * n * self.shape_b, self.BKG, fit, fill=0.62)
        self.bars(sc, ch, edges, f * n * self.shape_s, self.SIG, fit, fill=0.62,
                  base=(1 - f) * n * self.shape_b)
        for x, c in zip(mid, self.counts):
            e = np.sqrt(c)
            self.polyline(sc, [self.P(ch.fx(x), ch.fy(max(c - e, 0)), 1.2),
                               self.P(ch.fx(x), ch.fy(c + e), 1.2)], 1.6, DIGIT_INK, op)
        self.dots(sc, ch, mid, self.counts, DIGIT_INK, op, r=3.0, lift=1.4)
        self.text(sc, 0.22, 0.335, "points: a mixture with f = %.2f" % self.f_true, op, size=0.030)
        self.text(sc, 0.22, 0.875, "bars: the two shapes, f = %.3f" % f, fit, size=0.030)
        like = op * smoothstep(0.38, 0.5, u)
        lc = self.pane(((0.455, 0.655), (0.42, 0.74)), (self.fgrid[0], self.fgrid[-1]), (-6.5, 0.6))
        ticks = [(v, "%.2f" % v) for v in (self.fit - 2 * self.fit_err, self.fit,
                                           self.fit + 2 * self.fit_err)]
        self.axes(sc, lc, like, "f", "ln L − ln L(max)", ticks,
                  [(v, "%g" % v) for v in (-6, -4, -2, 0)])
        self.path(sc, lc, self.fgrid, self.lnl, self.FIT, like, smoothstep(0.42, 0.62, u),
                  width=3.0)
        mark = op * smoothstep(0.6, 0.68, u)
        self.hline(sc, lc, -0.5, GREY, mark)
        self.vline(sc, lc, self.fit, self.FIT, mark, width=2.0, dashed=True)
        self.text(sc, 0.555, 0.335, "f = %.3f ± %.3f" % (self.fit, self.fit_err), mark,
                  ink=dark(self.FIT, 0.3), size=0.031)
        self.text(sc, 0.555, 0.875, "± 1σ: where ln L is ½ lower", mark, size=0.029)
        toys = op * smoothstep(0.66, 0.76, u)
        tc = self.pane(((0.745, 0.945), (0.42, 0.74)), (self.fit_edges[0], self.fit_edges[-1]),
                       (0, float(self.fit_hist.max()) * 1.15))
        self.axes(sc, tc, toys, "fitted f", "share of samples",
                  [(v, "%.2f" % v) for v in (self.f_true - 0.05, self.f_true, self.f_true + 0.05)],
                  [(v, pct(v)) for v in (0.05, 0.10)])
        self.bars(sc, tc, self.fit_edges, self.fit_hist, self.FIT, toys, fill=0.7,
                  grow=smoothstep(0.68, 0.9, u))
        self.vline(sc, tc, self.f_true, DIGIT_INK, toys, width=2.2)
        m = self.meta["fit"]
        self.text(sc, 0.845, 0.335, "%s samples" % commas(m["toys"]), toys, size=0.031)
        self.text(sc, 0.845, 0.875, "black: true f", toys, size=0.029)
        self.tag(sc, TOYS, op)


# ====================================================================== 152: boosted trees
class BoostedRegressionTrees(Phys):
    """One real feature, the tilt of a digit's ink, against the angle it was turned by;
    a first tree of four steps; the residuals it leaves and the second tree fitted to
    them; the sum of fifty; then all 49 cells: held-out error against the number of
    trees for several learning rates and depths, and against the network."""

    TITLE = "Boosted trees for regression"
    PHASES = (("one feature", 8.0), ("a tree", 11.0), ("residuals", 12.0), ("fifty", 9.0),
              ("49 cells", 10.0), ("the network", 8.0))
    SUM, TREE, RES, HELD, TRAIN, NET = BLUE, AMBER, RED, BLUE, VIOLET, GREEN
    RATE_INKS = (TEAL, BLUE, VIOLET, RED)
    DEPTH_INKS = (OCHRE, RED, MAGENTA)
    BOX = ((0.08, 0.58), (0.385, 0.78))
    XR, YR = (-90, 90), (-52, 52)

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2 = (p.data for p in s.panels[:3])
        self.x, self.y = np.asarray(p0["xy"], float).T
        self.grid_x = np.asarray(p0["grid"], float)
        self.steps, self.alone = np.asarray(p0["steps"], float), np.asarray(p0["alone"], float)
        self.residuals = np.asarray(p0["residuals"], float)
        self.start = float(p0["start"])
        self.cuts, self.leaves = np.asarray(p0["cuts"], float), np.asarray(p0["leaves"], float)
        self.one_train, self.one_held = np.asarray(p0["train_rms"]), np.asarray(p0["held_rms"])
        self.images = np.asarray(p0["images"])
        self.rate = m["one_feature"]["rate"]
        root = self.cuts[0, 0]                     # the first split by itself: two means
        r0 = self.y - self.start
        self.root_steps = self.start + np.where(self.grid_x < root, r0[self.x < root].mean(),
                                                r0[self.x >= root].mean())
        self.count = np.asarray(p1["x"], float)
        self.curves = np.asarray(p2["all_curves"], float)          # rate, depth, trees
        self.table = np.asarray(p2["grid"], float)
        self.lowest_at = np.asarray(p2["lowest_at"]).reshape(self.table.shape)
        self.lowest = np.asarray(p2["lowest"], float).reshape(self.table.shape)
        self.rates, self.depths = m["rates"], m["depths"]
        self.main = self.depths.index(m["main"]["depth"])
        self.guess = float(self.curves[0, 0, 0])

    def subtitle_for(self, k):
        m = self.meta
        one, best, net = m["one_feature"], m["best"], m["network"]
        full = self.rates.index(1.0)
        deep = len(self.depths) - 1
        return [
            "handwritten digits stand in for events, as in the network's regression: "
            "each is turned by a known angle. Start with one measured number per digit, "
            "the tilt of its ink, as an analysis starts from one detector variable",
            "a regression tree asks yes-or-no questions of the tilt and answers with the "
            "mean of the training digits in each leaf: a staircase. Only %d%% of it is "
            "added, which is the learning rate" % round(100 * one["rate"]),
            "what the sum still gets wrong are the residuals. The second tree is fitted "
            "to them, not to the angles, and %d%% of it is added: the training error "
            "goes from %.1f° to %.1f° to %.1f°"
            % (round(100 * one["rate"]), one["train_rms"][0], one["train_rms"][1],
               one["train_rms"][2]),
            "fifty small trees make a fine staircase. Training error %.1f°; on %s "
            "held-out digits %.1f°, lowest after %d trees (%.1f°): past that the steps "
            "follow these %d digits and not the rule"
            % (one["train_rms"][-1], commas(S.REG_TEST), one["held_rms"][-1],
               one["held_lowest_at"], one["held_lowest"], one["events"]),
            "all 49 cells and %s training digits: small learning rates are still "
            "improving after %d trees. At rate 1, depth %d, the held-out error is lowest "
            "after %d trees and then climbs: overfitting"
            % (commas(S.REG_TRAIN), S.BRT_TREES, self.depths[deep],
               self.lowest_at[full, deep]),
            "the best setting, rate %g and depth %d, reaches %.1f° on held-out digits; "
            "the network, on the same digits, %.1f° (%d seeds each). One feature alone "
            "gave %.1f°; always answering the mean, %.1f°"
            % (best["rate"], best["depth"], best["held"], net["held"], len(m["seeds"]),
               one["held_lowest"], self.guess),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _plane(self, sc, op, box=None, faint=False, ylabel="true angle y"):
        ch = self.pane(box or self.BOX, self.XR, self.YR)
        self.axes(sc, ch, op, "tilt of the ink (degrees)", ylabel,
                  [(v, "%d" % v) for v in (-90, -45, 0, 45, 90)],
                  [(v, "%d" % v) for v in (-45, 0, 45)])
        return ch

    def _events(self, sc, ch, ys, op, ink=DIM, upto=None):
        self.dots(sc, ch, self.x[:upto], ys[:upto], ink, op, r=2.5, lift=0.7)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        self.text(sc, 0.05, 0.315, "one number per digit: the tilt of its ink", op, anchor="lm",
                  strong=True, size=HEAD)
        for n, img in enumerate(self.images):
            fx = 0.10 + 0.125 * n
            self.digit(sc, img, fx, 0.47, 96, op)
            a = np.deg2rad(self.x[n])
            draw = smoothstep(0.1 + 0.08 * n, 0.3 + 0.08 * n, u)
            dx, dy = self.frac(-np.sin(a) * 50, -np.cos(a) * 50)
            self.polyline(sc, [self.P(fx - dx * draw, 0.47 - dy * draw, 1.2),
                               self.P(fx + dx * draw, 0.47 + dy * draw, 1.2)], 3.0, self.TREE,
                          op * (draw > 0))
            self.text(sc, fx, 0.60, "tilt " + deg(self.x[n]), op * draw,
                      ink=dark(self.TREE, 0.4), size=0.030)
            self.text(sc, fx, 0.645, "turned " + deg(self.y[n]), op * draw, strong=True,
                      size=0.030)
        self.lines(sc, 0.05, 0.73, ["the long axis of the ink, from its second moments",
                                    "it follows the angle the digit was turned by,",
                                    "loosely: people slant their digits too"],
                   op * smoothstep(0.4, 0.55, u), size=0.030, step=0.044)
        ch = self._plane(sc, op, ((0.56, 0.94), (0.385, 0.78)), ylabel="true angle y, %d "
                         "training digits" % len(self.x))
        shown = int(len(self.x) * smoothstep(0.3, 0.85, u))
        self._events(sc, ch, self.y, op, GREY, max(shown, len(self.images)))
        for n in range(len(self.images)):
            self.disk(sc, self.P(ch.fx(self.x[n]), ch.fy(self.y[n])), 4.6, self.TREE, op,
                      ring=DIGIT_INK, ring_w=1.4, lift=1.5)
        self.stand_in(sc, op)
        self.tag(sc, MEASURED, op)

    def _tree(self, sc, op, k, grow):
        """Tree ``k`` as its three questions and four leaves; ``grow``: 0..2 levels."""
        cuts, leaves = self.cuts[k], self.leaves[k]
        top, kids, leaf_y = (0.79, 0.40), ((0.705, 0.52), (0.875, 0.52)), 0.645
        self.box(sc, *top, 132, 34, soft(self.TREE, 0.55), op * min(grow, 1.0),
                 "tilt < %s?" % deg(cuts[0], False), border=self.TREE, size=0.030)
        second = op * float(np.clip(grow - 1.0, 0.0, 1.0))
        for side, (fx, fy) in enumerate(kids):
            self.arrow(sc, (top[0] + (-0.03 if side == 0 else 0.03), top[1] + 0.035),
                       (fx, fy - 0.036), AXIS, second, width=2.0)
            self.text(sc, fx + (-0.035 if side == 0 else 0.035), 0.458, "yes" if side == 0
                      else "no", second, size=0.029)
            self.box(sc, fx, fy, 116, 34, soft(self.TREE, 0.55), second,
                     "tilt < %s?" % deg(cuts[1 + side], False), border=self.TREE, size=0.030)
            for half in range(2):
                lx = fx + (-0.043 if half == 0 else 0.043)
                v = leaves[2 * side + half]
                self.arrow(sc, (fx + (-0.02 if half == 0 else 0.02), fy + 0.035),
                           (lx, leaf_y - 0.034), AXIS, second, width=1.8)
                self.box(sc, lx, leaf_y, 68, 32, soft(self.RES if v < 0 else self.SUM, 0.5),
                         second, deg(v), border=AXIS, size=0.029)

    def phase_2(self, sc, u, op):
        ch = self._plane(sc, op)
        self._events(sc, ch, self.y, op)
        self.hline(sc, ch, self.start, self.SUM, op * (1 - smoothstep(0.62, 0.7, u)), width=2.0)
        whole = self.start + self.alone[0]
        one = smoothstep(0.1, 0.2, u)
        two = smoothstep(0.3, 0.42, u)
        shrink = smoothstep(0.62, 0.85, u)
        level = self.root_steps + (whole - self.root_steps) * two
        self.vline(sc, ch, self.cuts[0, 0], self.TREE, 0.8 * op * one, width=1.8, dashed=True)
        for c in self.cuts[0, 1:]:
            self.vline(sc, ch, c, self.TREE, 0.6 * op * two, width=1.4, dashed=True)
        self.path(sc, ch, self.grid_x, level, self.TREE, op * one * (1 - 0.65 * shrink),
                  width=3.0)
        scale = 1.0 + (self.rate - 1.0) * shrink
        self.path(sc, ch, self.grid_x, self.start + self.alone[0] * scale, self.SUM,
                  op * smoothstep(0.6, 0.66, u), width=3.4)
        self._tree(sc, op, 0, one + two)
        self.text(sc, 0.33, 0.315, "start: the mean of the angles, %s" % deg(self.start), op,
                  ink=dark(self.SUM, 0.3), size=0.031)
        self.lines(sc, 0.66, 0.73, [("a leaf answers with the mean of what", None),
                                    ("its digits still lack", None)],
                   op * two, size=0.030, step=0.042)
        self.text(sc, 0.66, 0.84, "sum = start + %g × tree 1" % self.rate,
                  op * smoothstep(0.62, 0.72, u), anchor="lm", ink=dark(self.SUM, 0.3),
                  size=HEAD)
        self.tag(sc, MEASURED, op)

    def phase_3(self, sc, u, op):
        left = ((0.07, 0.46), (0.385, 0.76))
        right = ((0.57, 0.95), (0.385, 0.76))
        add = smoothstep(0.62, 0.88, u)
        ch = self._plane(sc, op, left)
        self._events(sc, ch, self.y, op)
        self.path(sc, ch, self.grid_x, between(self.steps, 1, add), self.SUM, op, width=3.2)
        rc = self._plane(sc, op, right, ylabel="residual: true angle − sum so far")
        self.hline(sc, rc, 0.0, AXIS, op, dashed=False, width=1.2)
        shown = int(len(self.x) * smoothstep(0.03, 0.28, u))
        self._events(sc, rc, between(self.residuals, 1, add), op, soft(self.RES, 0.2), shown)
        tree = op * smoothstep(0.32, 0.42, u)
        for c in self.cuts[1]:
            self.vline(sc, rc, c, self.TREE, 0.6 * tree, width=1.4, dashed=True)
        self.path(sc, rc, self.grid_x, self.alone[1] * (1 - add), self.TREE, tree,
                  smoothstep(0.34, 0.56, u), width=3.2)
        self.arrow(sc, (0.55, 0.50), (0.475, 0.50), self.TREE, op * smoothstep(0.58, 0.66, u),
                   width=3.0)
        self.text(sc, 0.512, 0.455, "+ %g ×" % self.rate, op * smoothstep(0.58, 0.66, u),
                  ink=dark(self.TREE, 0.4), size=0.031)
        self.text(sc, 0.265, 0.315, "the sum of %d tree%s" % (1 + (add > 0.5), "s" *
                                                             (add > 0.5)), op,
                  ink=dark(self.SUM, 0.3), size=0.031)
        self.text(sc, 0.76, 0.315, "tree 2 is fitted to the residuals", tree,
                  ink=dark(self.TREE, 0.4), size=0.031)
        e = between(self.one_train, 1, add)
        self.text(sc, 0.07, 0.88, "training error %.1f°  (it was %.1f° before any tree)"
                  % (e, self.one_train[0]), op, anchor="lm", strong=True, size=0.031)
        self.tag(sc, MEASURED, op)

    def phase_4(self, sc, u, op):
        last = len(self.steps) - 1
        m = 2.0 * (last / 2.0) ** smoothstep(0.05, 0.8, u)
        k, w = min(int(m), last - 1), m - min(int(m), last - 1)
        ch = self._plane(sc, op)
        self._events(sc, ch, self.y, op)
        self.path(sc, ch, self.grid_x, between(self.steps, k, w), self.SUM, op, width=3.2)
        self.text(sc, 0.33, 0.315, "the sum of %d trees" % int(round(m)), op,
                  ink=dark(self.SUM, 0.3), size=BIG)
        count = np.arange(last + 1)
        chart = self.pane(((0.66, 0.94), (0.385, 0.68)), (0, last), (19, 27))
        self.axes(sc, chart, op, "trees", "error (degrees)", [(v, "%d" % v) for v in
                                                              (0, 10, 20, 30, 40, 50)],
                  [(v, "%d" % v) for v in (20, 22, 24, 26)])
        self.path(sc, chart, count, self.one_train, self.TRAIN, op, m / last, width=2.6)
        self.path(sc, chart, count, self.one_held, self.HELD, op, m / last, width=3.2, dot=True)
        best = int(self.one_held.argmin())
        self.vline(sc, chart, best, self.HELD, 0.7 * op * (m >= best), width=1.6, dashed=True)
        self.key(sc, 0.67, 0.81, [("held-out: %.1f°" % between(self.one_held, k, w), self.HELD),
                                  ("training: %.1f°" % between(self.one_train, k, w),
                                   self.TRAIN)], op, step=0.046, size=0.031)
        self.tag(sc, MEASURED, op)

    def phase_5(self, sc, u, op):
        go = smoothstep(0.05, 0.6, u)
        yr = (11, 27)
        ticks = [(v, "%d" % v) for v in (0, 100, 200, 300)]
        yt = [(v, "%d" % v) for v in (12, 16, 20, 24)]
        left = self.pane(((0.08, 0.44), (0.39, 0.76)), (0, self.count[-1]), yr)
        self.axes(sc, left, op, "trees", "held-out error (degrees)", ticks, yt, grid=False)
        for i, ink in enumerate(self.RATE_INKS):
            self.path(sc, left, self.count, self.curves[i, self.main], ink, op, go, width=2.8)
        self.text(sc, 0.26, 0.315, "depth %d, four learning rates" % self.depths[self.main], op,
                  strong=True, size=0.031)
        self.key(sc, 0.30, 0.42, [("rate %g" % r, ink) for r, ink in zip(self.rates,
                                                                        self.RATE_INKS)], op,
                 step=0.04)
        full = self.rates.index(1.0)
        right = self.pane(((0.57, 0.93), (0.39, 0.76)), (0, self.count[-1]), yr)
        self.axes(sc, right, op, "trees", "held-out error (degrees)", ticks, yt, grid=False)
        for j, ink in enumerate(self.DEPTH_INKS):
            ys = self.curves[full, j]
            self.path(sc, right, self.count, ys, ink, op, go, width=2.8)
            at = int(self.lowest_at[full, j])
            if at < self.count[-1] - 20:
                self.disk(sc, self.P(right.fx(at), right.fy(ys[at])), 5.0, ink,
                          op * smoothstep(0.6, 0.7, u), ring=DIGIT_INK, ring_w=1.4, lift=1.5)
        self.text(sc, 0.75, 0.315, "rate 1 (each tree added whole), three depths", op,
                  strong=True, size=0.031)
        self.key(sc, 0.79, 0.42, [("depth %d" % d, ink) for d, ink in zip(self.depths,
                                                                         self.DEPTH_INKS)], op,
                 step=0.04)
        deep = len(self.depths) - 1
        self.text(sc, 0.08, 0.875, "ringed: the lowest held-out error. Rate 1, depth %d: "
                  "%.1f° after %d trees, %.1f° after %d"
                  % (self.depths[deep], self.lowest[full, deep], self.lowest_at[full, deep],
                     self.table[full, deep], self.count[-1]),
                  op * smoothstep(0.62, 0.74, u), anchor="lm", size=0.030)
        self.tag(sc, MEASURED, op)

    def phase_6(self, sc, u, op):
        best = self.meta["best"]
        self.text(sc, 0.27, 0.33, "held-out error after %d trees (degrees)" % self.count[-1], op,
                  strong=True, size=0.031)
        x0, y0, dx, dy = 0.23, 0.46, 0.105, 0.085
        lo, hi = float(self.table.min()), float(self.table.max())
        for j, depth in enumerate(self.depths):
            self.text(sc, x0 + dx * j, y0 - 0.065, "depth %d" % depth, op, size=0.030)
        for i, rate in enumerate(self.rates):
            self.text(sc, x0 - 0.07, y0 + dy * i, "rate %g" % rate, op, anchor="rm", size=0.030)
            for j, depth in enumerate(self.depths):
                v = float(self.table[i, j])
                show = op * smoothstep(0.05 + 0.04 * (i + j), 0.15 + 0.04 * (i + j), u)
                won = rate == best["rate"] and depth == best["depth"]
                self.box(sc, x0 + dx * j, y0 + dy * i, 88, 38,
                         soft(self.SUM, 0.25 + 0.7 * (v - lo) / (hi - lo)), show, "%.1f" % v,
                         border=AMBER if won else AXIS, size=0.032)
                if won:
                    self.rect(sc, self.P(x0 + dx * j, y0 + dy * i), 94, 44, AMBER, show,
                              border=AMBER, border_w=2.6, radius=7, lift=0.9, fill_op=0.0)
        self.text(sc, 0.27, 0.80, "darker: lower error; ringed: the best", op, size=0.029)
        rows = (("always the mean", GREY, self.guess),
                ("one feature, trees", self.TREE, self.meta["one_feature"]["held_lowest"]),
                ("49 cells, best trees", self.SUM, best["held"]),
                ("49 cells, network", self.NET, self.meta["network"]["held"]))
        self.text(sc, 0.76, 0.33, "held-out error, the same digits", op, strong=True, size=0.031)
        full = self.span(0.20)[0]
        for r, (name, ink, v) in enumerate(rows):
            self.hbar(sc, 0.725, 0.43 + 0.085 * r, full * v / self.guess, ink, op, name,
                      "%.1f°" % v, u=(u - 0.35 - 0.08 * r) / 0.2, thick=24.0)
        net = self.meta["network"]
        self.lines(sc, 0.59, 0.79, ["over %d seeds: trees %.1f° to %.1f°,"
                                    % (len(best["per_seed"]), min(best["per_seed"]),
                                       max(best["per_seed"])),
                                    "network %.1f° to %.1f°" % (min(net["per_seed"]),
                                                                max(net["per_seed"]))],
                   op * smoothstep(0.7, 0.8, u), size=0.030, step=0.044)
        self.tag(sc, MEASURED, op)


# ====================================================================== 153: overtraining
class OvertrainingCheck(Phys):
    """The check itself, on a large training sample where training and held-out scores
    agree; a small sample and a large network, whose training scores pile up at 0 and 1;
    the Kolmogorov-Smirnov distance as the largest gap between two cumulative curves;
    distance and ROC-area gap against the sample size; early stopping and weight decay."""

    TITLE = "The overtraining check"
    PHASES = (("the check", 10.0), ("small sample", 13.0), ("a distance", 11.0),
              ("sample size", 10.0), ("remedies", 12.0))
    SIG, BKG = RED, BLUE
    ROW_INKS = (AMBER, GREEN, VIOLET)
    BOX = ((0.09, 0.60), (0.385, 0.78))
    FLOOR = 0.0015

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.final = np.asarray(p0["values"], float).reshape(2, 2, 2, -1)
        self.during = np.asarray(p0["during"], float)             # snap, class, set, bin
        self.edges = np.linspace(0, 1, self.final.shape[-1] + 1)
        self.ex_train = np.asarray(p0["example_train"], float)
        self.ex_held = np.asarray(p0["example_held"], float)
        self.updates = np.asarray(p1["x"], float)
        self.ks_t, self.gap_t, self.ks_big, self.gap_big = np.asarray(p1["ys"], float)
        self.sizes = np.asarray(p2["x"], float)
        self.ks, self.ks_seeds = np.asarray(p2["ys"], float), np.asarray(p2["per_seed"], float)
        self.critical = np.asarray(p2["critical"], float)
        self.gap, self.gap_seeds = np.asarray(p3["ys"], float), np.asarray(p3["per_seed"])
        self.held_auc = np.asarray(p3["held"], float)
        self.small, self.large = m["small"], m["large"]
        self.a, self.b = (list(m["sizes"]).index(v) for v in (self.small, self.large))

    def subtitle_for(self, k):
        m = self.meta
        a, b = self.a, self.b
        ks, gap, auc = m["ks"], m["auc_gap"], m["heldout_auc"]
        end, stop, decay = m["rows"]
        ex = m["example"]
        return [
            "handwritten digits stand in for events (4s: signal, 9s: background). The "
            "check every analysis shows: the score of the training events, as points, "
            "over the score of held-out events, as bars. Here they agree",
            "a large network and only %d training digits a class: as training goes on "
            "every training score moves to 0 or 1, the held-out scores do not, and the "
            "points leave the bars. The network has learnt its sample" % self.small,
            "a number for it: the Kolmogorov-Smirnov distance D, the largest gap "
            "between the two cumulative curves. For these 4s D = %.2f; two samples of "
            "one distribution would be %.2f apart or more only 5%% of the time"
            % (ex["distance"], m["critical"][a]),
            "against the size of the training sample, D falls from %.2f at %d a class "
            "to %.2f at %s, and the gap in ROC area from %.3f to %.3f (mean of %d "
            "seeds). More training events is the first remedy"
            % (ks[end]["mean"][0], m["sizes"][0], ks[end]["mean"][b], commas(self.large),
               gap[end]["mean"][0], gap[end]["mean"][b], len(m["seeds"])),
            "at %d a class, stopping when a validation loss turns up takes D from %.2f "
            "to %.2f; weight decay to %.2f. But with %s a class weight decay costs "
            "held-out ROC area: %.3f against %.3f"
            % (self.small, ks[end]["mean"][a], ks[stop]["mean"][a], ks[decay]["mean"][a],
               commas(self.large), auc[decay]["mean"][b], auc[end]["mean"][b]),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _overlay(self, sc, hist, n, op, points=1.0, box=None):
        """``hist[class, set, bin]``: held-out events as bars, training events as points
        with the error bar of their count (``n`` training events a class)."""
        ch = self.pane(box or self.BOX, (0, 1), (self.FLOOR, 1.6), ylog=True)
        self.axes(sc, ch, op, "score", "share of the class (log scale)",
                  [(v, "%g" % v) for v in (0, 0.25, 0.5, 0.75, 1)],
                  [(v, pct(v, 1 if v < 0.01 else 0)) for v in (0.01, 0.1, 1.0)])
        mid = (self.edges[:-1] + self.edges[1:]) / 2
        half = (mid[1] - mid[0]) * 0.17
        for c, ink in ((1, self.BKG), (0, self.SIG)):
            self.bars(sc, ch, self.edges, hist[c, 1], ink, op, fill=0.5)
        for c, ink in ((1, self.BKG), (0, self.SIG)):
            shift = half if c == 0 else -half
            for x, v in zip(mid, hist[c, 0]):
                if v < self.FLOOR:
                    continue
                e = np.sqrt(v / n)
                fx = ch.fx(x + shift)
                self.polyline(sc, [self.P(fx, ch.fy(max(v - e, self.FLOOR)), 1.3),
                                   self.P(fx, ch.fy(v + e), 1.3)], 1.8, dark(ink, 0.35),
                              op * points)
                self.disk(sc, self.P(fx, ch.fy(v)), 4.2, dark(ink, 0.25), op * points, lift=1.5)
        return ch

    def _legend(self, sc, op, x=0.66, y=0.39):
        self.key(sc, x, y, [("signal: the 4s", self.SIG), ("background: the 9s", self.BKG)], op,
                 step=0.046, size=0.031)
        self.rect(sc, self.P(x, y + 0.105), 16, 22, GREY, op, radius=1.0, fill_op=0.55)
        self.text(sc, x + 0.016, y + 0.105, "bars: held-out events", op, anchor="lm", size=0.031)
        self.disk(sc, self.P(x, y + 0.155), 4.6, DIGIT_INK, op)
        self.text(sc, x + 0.016, y + 0.155, "points: training events", op, anchor="lm",
                  size=0.031)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        self._overlay(sc, self.final[1], self.large, op, smoothstep(0.3, 0.5, u))
        self._legend(sc, op)
        self.text(sc, 0.345, 0.315, "%s training digits a class" % commas(self.large), op,
                  strong=True, size=0.031)
        b = self.b
        self.lines(sc, 0.652, 0.64, ["points on bars: the network", "treats digits it has "
                                     "not seen", "as it treats its own",
                                     "D = %.2f, ROC-area gap %.3f"
                                     % (self.ks[0, b], self.gap[0, b])],
                   op * smoothstep(0.55, 0.7, u), size=0.030, step=0.044)
        self.stand_in(sc, op)
        self.tag(sc, MEASURED, op)

    def phase_2(self, sc, u, op):
        s = smoothstep(0.03, 0.82, u) * (len(self.updates) - 1)
        k = min(int(s), len(self.updates) - 2)
        w = s - k
        self._overlay(sc, between(self.during, k, w), self.small, op)
        self.text(sc, 0.345, 0.315, "%d training digits a class, after %s updates"
                  % (self.small, commas(between(self.updates, k, w))), op, strong=True,
                  size=0.031)
        chart = self.pane(((0.69, 0.94), (0.42, 0.68)), (self.updates[0], self.updates[-1]),
                          (0, 0.27), xlog=True)
        self.axes(sc, chart, op, "updates", "",
                  [(v, commas(v)) for v in (10, 100, 1000)],
                  [(v, "%g" % v) for v in (0.1, 0.2)])
        go = s / (len(self.updates) - 1)
        self.path(sc, chart, self.updates, self.gap_t, VIOLET, op, go, width=2.8, dot=True)
        self.path(sc, chart, self.updates, self.ks_t, AMBER, op, go, width=3.0, dot=True)
        self.key(sc, 0.70, 0.80, [("distance D: %.2f" % between(self.ks_t, k, w), AMBER),
                                  ("ROC-area gap: %.3f" % between(self.gap_t, k, w), VIOLET)],
                 op, step=0.046, size=0.031)
        self.text(sc, 0.815, 0.375, "training against held-out", op, size=0.029)
        self.tag(sc, MEASURED, op)

    def phase_3(self, sc, u, op):
        ex = self.meta["example"]
        z_held, z_train = stretch(self.ex_held), stretch(self.ex_train)
        ch = self.pane(self.BOX, (-STRETCH, STRETCH), (0, 1.02))
        self.axes(sc, ch, op, "score (axis stretched near 0 and 1)",
                  "share of the 4s scoring below",
                  [(float(stretch(v)), "%g" % v) for v in (0.0001, 0.01, 0.5, 0.99, 0.9999)],
                  [(v, pct(v)) for v in (0, 0.5, 1)])
        grid = np.linspace(-STRETCH, STRETCH, 241)
        held = np.searchsorted(z_held, grid, side="right") / len(z_held)
        train = np.searchsorted(z_train, grid, side="right") / len(z_train)
        self.path(sc, ch, grid, held, soft(self.SIG, 0.25), op, smoothstep(0.03, 0.3, u),
                  width=4.0)
        go = smoothstep(0.2, 0.47, u)
        self.path(sc, ch, grid, train, dark(self.SIG, 0.45), op, go, width=2.6)
        n = len(z_train)                               # a step up for every training event
        shown = z_train <= -STRETCH + 2 * STRETCH * go
        self.dots(sc, ch, z_train[shown], (np.arange(n)[shown] + 1) / n, dark(self.SIG, 0.45),
                  op, r=2.4, lift=1.3)
        lo = float(np.searchsorted(self.ex_train, ex["at"], side="right") / len(self.ex_train))
        hi = float(np.searchsorted(self.ex_held, ex["at"], side="right") / len(self.ex_held))
        gap = op * smoothstep(0.5, 0.6, u)
        fx = ch.fx(float(stretch(ex["at"])))
        self.polyline(sc, [self.P(fx, ch.fy(lo), 1.4), self.P(fx, ch.fy(hi), 1.4)], 4.5, AMBER,
                      gap)
        self.text(sc, fx - 0.014, ch.fy(0.38), "D = %.2f" % ex["distance"], gap, anchor="rm",
                  ink=dark(AMBER, 0.45), size=BIG)
        self.text(sc, 0.345, 0.315, "the %d training 4s against %s held-out 4s"
                  % (len(self.ex_train), commas(len(self.ex_held))), op, strong=True, size=0.031)
        x = 0.655
        self.key(sc, x, 0.40, [("held-out 4s", soft(self.SIG, 0.25)),
                               ("training 4s", dark(self.SIG, 0.45))], op, step=0.046,
                 size=0.031)
        self.lines(sc, x - 0.008, 0.52, ["%s of the held-out 4s score" % pct(hi),
                                         "below %.4f; %s of the" % (ex["at"], pct(lo)),
                                         "training 4s do: a gap of %.2f" % ex["distance"]],
                   gap, size=0.030, step=0.044)
        p = ex["p_value"]
        chance = "below one in a million" if p < 1e-6 else "%.2g" % p
        self.lines(sc, x - 0.008, 0.68, ["D is the largest such gap", "the chance of one this "
                                         "large for", "two samples of one distribution:",
                                         "p = " + chance],
                   op * smoothstep(0.66, 0.78, u), size=0.030, step=0.044)
        self.tag(sc, MEASURED, op)

    def _against_size(self, sc, op, rows, go, level=1.0):
        """Distance (left) and ROC-area gap (right) against the training-sample size."""
        xt = [(v, commas(v)) for v in (25, 100, 500, 2000)]
        left = self.pane(((0.09, 0.46), (0.39, 0.76)), (self.sizes[0], self.sizes[-1]), (0, 0.44),
                         xlog=True)
        self.axes(sc, left, op, "training digits a class", "distance D",
                  xt, [(v, "%g" % v) for v in (0.1, 0.2, 0.3, 0.4)])
        right = self.pane(((0.58, 0.94), (0.39, 0.76)), (self.sizes[0], self.sizes[-1]),
                          (-0.01, 0.16), xlog=True)
        self.axes(sc, right, op, "training digits a class", "ROC area: training − held-out",
                  xt, [(v, "%g" % v) for v in (0.05, 0.10, 0.15)])
        self.path(sc, left, self.sizes, self.critical, GREY, op * level, width=2.0)
        for r in rows:
            ink = self.ROW_INKS[r]
            g = go[r]
            for chart, mean, seeds in ((left, self.ks[r], self.ks_seeds[r]),
                                       (right, self.gap[r], self.gap_seeds[r])):
                for i, x in enumerate(self.sizes):
                    fx = chart.fx(x)
                    self.polyline(sc, [self.P(fx, chart.fy(seeds[i].min()), 0.8),
                                       self.P(fx, chart.fy(seeds[i].max()), 0.8)], 2.0,
                                  soft(ink, 0.3), op * (g >= i / (len(self.sizes) - 1)))
                self.path(sc, chart, self.sizes, mean, ink, op, g, width=3.0, dot=True)
        return left, right

    def phase_4(self, sc, u, op):
        go = smoothstep(0.08, 0.7, u)
        self._against_size(sc, op, (0,), {0: go})
        self.box(sc, 0.10, 0.88, 14, 14, GREY, op, radius=3)
        self.text(sc, 0.116, 0.88, "grey: two samples of one distribution are this far apart, "
                  "or more, 5% of the time", op, anchor="lm", size=0.029)
        self.text(sc, 0.5, 0.315, "trained to the end (%s updates): mean and range of %d seeds"
                  % (commas(self.updates[-1]), self.ks_seeds.shape[-1]), op, strong=True,
                  size=0.031)
        self.tag(sc, MEASURED, op)

    def phase_5(self, sc, u, op):
        go = {0: 1.0, 1: smoothstep(0.08, 0.4, u), 2: smoothstep(0.42, 0.74, u)}
        self._against_size(sc, op, (0, 1, 2), go, level=0.8)
        names = ("trained to the end", "stopped early", "weight decay")
        for r, name in enumerate(names):
            show = op * (1.0 if r == 0 else smoothstep(0.0, 0.1, go[r]))
            x = 0.12 + 0.27 * r
            self.box(sc, x, 0.315, 14, 14, self.ROW_INKS[r], show, radius=3)
            self.text(sc, x + 0.016, 0.315, name, show, anchor="lm",
                      ink=dark(self.ROW_INKS[r], 0.35), size=0.031)
        b = self.b
        self.text(sc, 0.09, 0.88, "held-out ROC area at %s a class:  %.3f,  %.3f stopped "
                  "early,  %.3f with weight decay"
                  % (commas(self.large), self.held_auc[0, b], self.held_auc[1, b],
                     self.held_auc[2, b]), op * smoothstep(0.78, 0.88, u), anchor="lm",
                  size=0.029)
        self.tag(sc, MEASURED, op)


ANIMATIONS = {
    "network-regression": NetworkRegression,
    "classifier-score-as-variable": ClassifierScore,
    "boosted-regression-trees": BoostedRegressionTrees,
    "overtraining-check": OvertrainingCheck,
}

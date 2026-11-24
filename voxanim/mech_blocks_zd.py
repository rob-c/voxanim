"""Animations 269-270 of "Blocks and code", "Minuit and SGD" (see :mod:`voxanim.blocks`), drawn
from :mod:`voxanim.study_blocks_zd`: one small fit (``examples/minuit_vs_sgd.py``), found by
Minuit and by SGD.

* :class:`Feels` (``minuit-feels-its-way``): the 40 made-up points and the model's curve; the
  space of the model's three numbers (a, tau, b) as three axes in perspective, the start in
  it; every call MIGRAD made, in its order: before each step it asks about points a little
  up and down each axis (shown in a magnified inset round the point it asks from, each probe
  marked lower, higher or flat), then steps, sometimes too far, coming back; the path to the
  minimum, the curve over the data improving with each kept step; HESSE's 16 calls and the
  error bars.
* :class:`Versus` (``minuit-and-sgd``): the same space with Minuit's few kept steps; SGD's
  batch of 8 points and its first small step; its 3,000 steps as a drunken walk; its last
  1,000 positions wandering round the minimum (magnified), their average beside Minuit's
  minimum and its error bars; the counts of single-point evaluations.

The alphabet: the data are blue dots; the model's curve and Minuit's points and path dark ink;
SGD's walk teal; chi-square (the loss) violet; a probe that lowered chi-square green, raised it
red, changed it by less than 0.0001 grey (the word is written beside the number too).
The 3-D space is projected by hand (:class:`Space`): a fixed perspective view, a on the axis
running left to right, b up, tau going back.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .blocks import (
    DATA,
    HEAD,
    INK,
    LOSS,
    QUIET,
    SIZE,
    Blocks,
    Code,
    exact,
    measured,
)
from .easing import ease_in_out, smoothstep
from .losschart import LossChart
from .mech import AXIS, GREEN, GRID, PAPER, RED, TEAL, dark

ON = "made-up data"
LOSS_INK = dark(LOSS, 0.3)
PATH = INK                       # Minuit's points and path, and the model's curve
WALK = TEAL                      # SGD's walk
WALK_INK = dark(TEAL, 0.35)
FLAT_INK = (0.58, 0.60, 0.62)
CHANGE_INK = dict(lower=GREEN, higher=RED, flat=FLAT_INK)
CHANGE_TEXT = dict(lower=dark(GREEN, 0.3), higher=dark(RED, 0.2), flat=QUIET)
# which way a probe moves along each axis, (down, up) its parameter: a runs left to right, tau
# back into the picture, b up
WAY = (("left", "right"), ("forward", "back"), ("down", "up"))

# ---------------------------------------------------------------------- the layout
SPACE_BOX = (0.075, 0.285, 0.43, 0.87)    # the 3-D space: x0, x1, top, bottom
COL_X = 0.308                             # the right column of the picture
COL_W = 0.475 - COL_X                     # and its width
COUNT_Y = (0.385, 0.427, 0.468)           # the counter's three lines
CHART_BOX = ((0.33, 0.47), (0.505, 0.625))
INSET = (0.392, 0.768)                    # the magnified inset's centre
INSET_W, INSET_H = 150.0, 114.0           # its size, px
INSET_R = 38.0                            # its largest probe, px from the centre
AXIS_LEN = 39.0                           # the inset's axis lines, px each way
ERR_R = 38.0                              # the longest error bar in the inset, px each way
ROW_Y = 0.902                             # the row of values under the space


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on(u, a, op, width=0.05):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


class Space:
    """The space of (a, tau, b) drawn in perspective: a runs left to right, b up, tau back
    (a fixed view from the front, a little to the left and above). :meth:`at` gives a
    point's layout ``(fx, fy)``; :meth:`axis_dir` an axis's direction on screen there."""

    LO = np.array([0.0, 0.0, 0.0])
    HI = np.array([9.0, 3.5, 2.2])
    DIMS = np.array([1.35, 1.3, 1.2])         # the box's world size along a, tau, b
    YAW, PITCH, DIST = np.radians(30.0), np.radians(22.0), 4.2

    def __init__(self, box=SPACE_BOX, width=960.0, height=540.0):
        self.w, self.h = width, height
        x0, x1, top, bottom = box
        corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float)
        raw = np.array([self._raw(self.LO + c * (self.HI - self.LO)) for c in corners])
        lo, hi = raw.min(0), raw.max(0)
        bw, bh = (x1 - x0) * width, (bottom - top) * height
        self.k = min(bw / (hi[0] - lo[0]), bh / (hi[1] - lo[1]))
        self.cx = (x0 + x1) / 2 * width - self.k * (lo[0] + hi[0]) / 2
        self.cy = (top + bottom) / 2 * height + self.k * (lo[1] + hi[1]) / 2

    def _raw(self, p):
        n = (np.asarray(p, float) - self.LO) / (self.HI - self.LO) - 0.5
        X, Z, Y = n * self.DIMS
        cy, sy, cp, sp = np.cos(self.YAW), np.sin(self.YAW), np.cos(self.PITCH), np.sin(self.PITCH)
        x1, z1 = X * cy + Z * sy, -X * sy + Z * cy
        y2, z2 = Y * cp + z1 * sp, -Y * sp + z1 * cp
        s = self.DIST / (self.DIST + z2)
        return np.array([x1 * s, y2 * s])

    def px(self, p):
        r = self._raw(p)
        return np.array([self.cx + self.k * r[0], self.cy - self.k * r[1]])

    def at(self, p):
        """Layout ``(fx, fy)`` of parameters ``p = (a, tau, b)``."""
        q = self.px(p)
        return float(q[0] / self.w), float(q[1] / self.h)

    def axis_dir(self, p, axis):
        """The unit screen direction (px, y down) of parameter ``axis`` growing at ``p``."""
        e = np.zeros(3)
        e[axis] = (self.HI[axis] - self.LO[axis]) * 1e-3
        d = self.px(np.asarray(p, float) + e) - self.px(p)
        return d / np.hypot(*d)

    def floor(self, p):
        """``p`` dropped to b = 0 (its shadow on the floor)."""
        return (p[0], p[1], 0.0)


class Fit(Blocks):
    """What the two share: the space, the data chart, the inset, the counter, timing."""

    STATION = None

    def setup(self):
        self.meta = self.study.metadata
        self.space = Space()
        self.starts = np.cumsum([0.0] + [s for _n, s in self.PHASES])
        x, y = np.asarray(self.meta["x"]), np.asarray(self.meta["y"])
        self.data_x, self.data_y = x, y
        self.grid = np.asarray(self.meta["grid"])

    def station_for(self, k):
        return None

    # ------------------------------------------------------------------ time
    def time_of(self, k, u):
        return float(self.starts[k] + (self.starts[k + 1] - self.starts[k]) * u)

    def since(self, t0, width=0.4):
        """0 before time ``t0``, 1 from ``t0 + width`` (eased)."""
        return smoothstep(t0, t0 + width, getattr(self, "_t", 0.0))

    def at_u(self, k, u, width=0.4):
        return self.since(self.time_of(k, u), width)

    def prog(self, k, u0, u1):
        t0, t1 = self.time_of(k, u0), self.time_of(k, u1)
        return ease_in_out(float(np.clip((getattr(self, "_t", 0.0) - t0) / (t1 - t0), 0, 1)))

    # ------------------------------------------------------------------ the space
    def lifted(self, fx, fy):
        return self.P(float(fx), float(fy), self.ARROW_LIFT)

    def guide(self, sc, pts, ink, op, width=1.6):
        """A polyline through layout points, lifted in front of the shapes."""
        if op > 0.01 and len(pts) > 1:
            self.polyline(sc, [self.lifted(*p) for p in pts], width, ink, op, bias=0.3)

    def dashed(self, sc, a, b, ink, op, n=7, width=1.2):
        if op <= 0.01:
            return
        a, b = np.asarray(a, float), np.asarray(b, float)
        for i in range(n):
            u0, u1 = i / n, (i + 0.5) / n
            self.guide(sc, [a + (b - a) * u0, a + (b - a) * u1], ink, op, width)

    def draw_space(self, sc, op):
        """The box of the three axes: its edges faint, the three axes from the corner where
        a, tau and b are smallest, each named at its end, with a few ticks."""
        if op <= 0.01:
            return
        S = self.space
        lo, hi = S.LO, S.HI
        for i in range(3):
            for c in ((0, 0), (0, 1), (1, 0), (1, 1)):
                a, b = np.zeros(3), np.zeros(3)
                j, k = [n for n in range(3) if n != i]
                for v in (a, b):
                    v[j], v[k] = (lo[j], hi[j])[c[0]], (lo[k], hi[k])[c[1]]
                a[i], b[i] = lo[i], hi[i]
                if c == (0, 0):
                    continue
                self.polyline(sc, [self.P(*S.at(a)), self.P(*S.at(b))], 1.0, GRID, 0.45 * op,
                              bias=0.0)
        names = self.meta["names"]
        ticks = ((0, 5), (2, 1), (2, 2))
        for i in range(3):
            end = lo.copy()
            end[i] = hi[i] * 1.06
            self.arrow(sc, S.at(lo), S.at(end), AXIS, 0.8 * op, width=1.4, lift=0.0)
            d = S.axis_dir(end, i)
            ex, ey = S.at(end)
            # beyond the end; tau's to its left, clear of the paths that rise past it
            along = 13.0 if i != 1 else 4.0
            side = np.array([-d[1], d[0]]) * (12.0 if i == 1 else 0.0)
            side = side if side[0] <= 0 else -side
            dx, dy = self.frac(*(d * along + side))
            self.text(sc, ex + dx, ey + dy, names[i], op, SIZE, strong=True)
        for i, v in ticks:
            p = lo.copy()
            p[i] = v
            fx, fy = S.at(p)
            off = {0: (0.0, 14.0), 1: (14.0, 6.0), 2: (-12.0, 0.0)}[i]
            ox, oy = self.frac(*off)
            self.polyline(sc, [self.P(fx, fy), self.P(fx - ox * 0.35, fy - oy * 0.35)], 1.0,
                          AXIS, 0.6 * op, bias=0.0)
            self.text(sc, fx + ox, fy + oy, "%g" % v, op, SIZE, ink=QUIET)

    def point(self, sc, p, ink, op, r=4.5, hollow=False, lift=1.5):
        """A point of the space: a dot, its shadow on the floor and a faint line down to it."""
        if op <= 0.01:
            return
        S = self.space
        fx, fy = S.at(p)
        if p[2] > 0.02:
            sx, sy = S.at(S.floor(p))
            self.dashed(sc, (fx, fy), (sx, sy), AXIS, 0.4 * op, n=5, width=1.0)
            self.disk(sc, self.P(sx, sy), 1.8, AXIS, 0.5 * op, lift=1.38, outline=0.0)
        if hollow:
            self.rect(sc, self.P(fx, fy), 2 * r, 2 * r, PAPER, op, border=ink, border_w=2.0,
                      radius=r, lift=lift)
        else:
            self.disk(sc, self.P(fx, fy), r, ink, op, lift=lift, outline=0.15)

    def ring(self, sc, fx, fy, op, r=11.0):
        """The ring round the point the inset magnifies."""
        if op > 0.01:
            self.rect(sc, self.P(fx, fy), 2 * r, 2 * r, INK, op, border=INK, border_w=1.6,
                      radius=r, lift=1.45, fill_op=0.0)

    # ------------------------------------------------------------------ the data chart
    def data_chart(self):
        return LossChart(CHART_BOX, (0.0, 10.0), (0.0, 8.0))

    def draw_chart(self, sc, op, ys=None, ink=PATH, picked=None, curve_op=1.0):
        """The 40 points (blue dots; ``picked``: these ringed, the rest dimmed) and the model's
        curve ``ys`` over them (clipped to the chart)."""
        if op <= 0.01:
            return
        ch = self.data_chart()
        self.axes(sc, ch, op, xticks=[], yticks=[], grid=False)
        if ys is not None and curve_op > 0.01:
            self.clipped(sc, ch, self.grid, ys, ink, op * curve_op)
        for i, (x, y) in enumerate(zip(self.data_x, self.data_y, strict=True)):
            dim = picked is not None and i not in picked
            a = op * (0.3 if dim else 1.0)
            self.disk(sc, self.P(ch.fx(x), ch.fy(y)), 2.4, DATA, a, lift=1.5, outline=0.1)
            if picked is not None and not dim:
                self.rect(sc, self.P(ch.fx(x), ch.fy(y)), 11.0, 11.0, INK, op, border=INK,
                          border_w=1.5, radius=5.5, lift=1.55, fill_op=0.0)

    def clipped(self, sc, ch, xs, ys, ink, op, width=2.0):
        """``ys`` over ``xs`` on chart ``ch``, the parts outside its y range left out."""
        lo, hi = ch.y_range
        run = []
        for x, y in zip(xs, ys, strict=True):
            if lo <= y <= hi:
                run.append((ch.fx(x), ch.fy(y)))
            else:
                self.guide(sc, run, ink, op, width)
                run = []
        self.guide(sc, run, ink, op, width)

    # ------------------------------------------------------------------ the inset
    def inset_box(self, sc, op, src=None, label="magnified"):
        """The inset's frame, its label, and a faint line from the point it magnifies."""
        if op <= 0.01:
            return
        cx, cy = INSET
        self.rect(sc, self.P(cx, cy), INSET_W, INSET_H, PAPER, op, border=AXIS, border_w=1.2,
                  radius=6, lift=0.2)
        hw, hh = self.frac(INSET_W / 2, INSET_H / 2)
        self.text(sc, cx - hw + self.frac(2.0)[0], cy - hh - self.frac(0, 12.0)[1], label, op,
                  SIZE, "lm", ink=QUIET)
        if src is not None:
            self.ring(sc, *src, op)
            r = self.frac(11.0, 11.0)
            a = np.array(src) + np.array([r[0] * 0.7, r[1] * 0.7])
            b = np.array([cx - hw, cy - hh * 0.2])
            self.polyline(sc, [self.P(*a, 0.0), self.P(*b, 0.0)], 1.0, AXIS, 0.4 * op, bias=0.0)

    def inset_axes(self, sc, base, op, length=AXIS_LEN):
        """The three axes' directions through the inset's centre, faint, named at their ends."""
        if op <= 0.01:
            return
        cx, cy = INSET
        for i in range(3):
            d = self.space.axis_dir(base, i)
            a = self.frac(*(-d * length))
            b = self.frac(*(d * length))
            self.polyline(sc, [self.P(cx + a[0], cy + a[1], 0.3), self.P(cx + b[0], cy + b[1],
                                                                          0.3)],
                          1.0, AXIS, 0.35 * op, bias=0.0)
            e = self.frac(*(d * (length + 9.0)))
            self.text(sc, cx + e[0], cy + e[1], self.meta["names"][i], op, SIZE, ink=QUIET)

    def offset(self, base, d, scale):
        """Layout offset in the inset of a move ``d`` from ``base``, magnified ``scale`` px per
        unit of the parameters, along the axes' screen directions."""
        v = sum(float(d[i]) * self.space.axis_dir(base, i) for i in range(3)) * scale
        return self.frac(*v)

    def scale_for(self, base, moves, r=INSET_R):
        """Px per unit so that the longest of ``moves`` reaches ``r`` px."""
        longest = max(float(np.hypot(*sum(float(d[i]) * self.space.axis_dir(base, i)
                                          for i in range(3)))) for d in moves)
        return r / max(longest, 1e-12)

    def error_bars(self, sc, base, op, r=ERR_R, width=2.4):
        """HESSE's errors as bars along each axis through the inset's centre, magnified so the
        longest reaches ``r`` px, with end caps; returns the magnification (px per unit)."""
        errs = self.meta["errors"]
        scale = self.scale_for(base, [np.eye(3)[i] * errs[i] for i in range(3)], r=r)
        if op <= 0.01:
            return scale
        cx, cy = INSET
        for i in range(3):
            dx, dy = self.offset(base, np.eye(3)[i] * errs[i], scale)
            self.guide(sc, [(cx - dx, cy - dy), (cx + dx, cy + dy)], PATH, op, width)
            n = np.array(self.span(-dy, dx))
            tx, ty = self.frac(*(n / max(float(np.hypot(*n)), 1e-9) * 5.0))
            for s in (-1, 1):
                self.guide(sc, [(cx + s * dx - tx, cy + s * dy - ty),
                                (cx + s * dx + tx, cy + s * dy + ty)], PATH, op, width - 0.4)
        return scale

    # ------------------------------------------------------------------ text
    def lines_at(self, sc, rows, op):
        """The counter: up to three ``(text, size, ink)`` rows down the right column."""
        for (text, size, ink), y in zip(rows, COUNT_Y, strict=False):
            if text:
                self.text(sc, COL_X, y, text, op, size, "lm", ink=ink)

    def step_indicator(self, sc, current, phases):
        """The phase under way by name, the others by number (beside the title)."""
        items = ["%d %s" % (n + 1, name) if n == current else "%d" % (n + 1)
                 for n, name in enumerate(phases)]
        size, gap = 0.030, 0.016
        widths = [layout.text_width(s, size) for s in items]
        x = (1 - layout.SAFE) - (sum(widths) + gap * (len(items) - 1)) - 0.004
        for n, (s, w) in enumerate(zip(items, widths, strict=True)):
            ink = (0.12, 0.36, 0.66) if n == current else layout.INK_SOFT if n < current \
                else (0.56, 0.58, 0.60)
            sc.text(x, layout.SAFE + 0.012, s, size, "regular", ink, 1.0, "la")
            x += w + gap


# ====================================================================== 269
class Feels(Fit):
    """The data and the space; the first probes; the first step, too far and back; probe and
    step again; the minimum; HESSE's error bars."""

    TITLE = "How Minuit feels its way"
    PHASES = (("the fit", 8.0), ("feeling", 13.0), ("a step", 9.0), ("again", 14.0),
              ("the minimum", 7.0), ("error bars", 9.0))
    LAST = "Binned or unbinned?"
    NEXT = "Minuit and SGD"

    def setup(self):
        super().setup()
        m = self.meta
        tl = m["timeline"]
        self.tl = tl
        # when each call is asked: (phase, u), in MIGRAD's order
        when = {0: (0, 0.5)}
        for j in range(12):
            when[1 + j] = (1, 0.1 + 0.064 * j)
        when[13], when[14] = (2, 0.12), (2, 0.55)
        u = 0.03
        for i in range(15, 49):
            when[i] = (3, u)
            nxt = tl[i + 1]["kind"] if i + 1 < len(tl) else ""
            u += 0.045 if tl[i]["kind"] == "try" or nxt == "try" else 0.021
        for j, i in enumerate(range(49, 65)):
            when[i] = (4, 0.03 + 0.018 * j)
        for j, i in enumerate(range(65, 81)):
            when[i] = (5, 0.04 + 0.022 * j)
        self.call_t = np.array([self.time_of(*when[i]) for i in range(len(tl))])
        # the inset's magnification for each round: its largest move reaches INSET_R px
        self.round_scale = {}
        for r in range(m["rounds"]):
            near = [e for e in tl if e["round"] == r and e["near"]]
            if near:
                base = tl[near[0]["base"]]["point"]
                self.round_scale[r] = self.scale_for(base, [e["d"] for e in near])
        self.CODE = (
            Code(("mn-model", "mn-chi2"), run=((0.0, 1), (0.4, 3), (0.62, 4))),
            Code("mn-migrad", run=((0.0, 0), (0.08, 2))),
            Code("mn-migrad", run=((0.0, 2),)),
            Code("mn-migrad", run=((0.0, 2),)),
            Code("mn-migrad", run=((0.0, 2), (0.5, 3)), out=((0.5, "%d calls" % m[
                "migrad_calls"]),)),
            Code("mn-migrad", run=((0.0, 4),), out=((0.75, m["printed"]),)))
        self.EVIDENCE = (
            measured("chi-square at the start", ON),
            measured("iminuit %s's own calls 1-12" % m["iminuit"], ON),
            measured("calls 13 and 14, in iminuit's order", ON),
            measured("calls 15-48: probes, tries, kept", ON),
            measured("65 calls; EDM %s" % m["edm_text"], ON),
            measured("HESSE's 16 calls; the errors", ON))

    def heading_for(self, k, u):
        return ("a fit with 3 numbers: one point in 3-D",
                "before a step, it asks around",
                "a step: too far, then back",
                "ask around, step, again",
                "the minimum",
                "HESSE: how sure, along each axis")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "40 made-up points; the model has 3 numbers, a, τ and b. Each guess is a point in "
            "3-D, and chi-square scores it: %s at the start." % m["start_text"],
            "MIGRAD, Minuit's minimiser, first asks about points a little up and down each "
            "axis: 12 calls. Green: chi-square went down.",
            "Then it steps. The first try goes too far (chi-square %s); it comes back to %s "
            "and keeps that point." % (m["overshoot_text"], m["back_text"]),
            "Round each kept point it asks the 3 axes again, then steps: the curve closes "
            "in on the data. Grey: flat.",
            "It stops when its EDM, its estimate of how much lower chi-square could go, is "
            "tiny: %d calls, %d probes, %d tries, %d kept." % (m["migrad_calls"], m["probes"],
                                                              m["tries"], m["kept"]),
            "HESSE asks %d more points round the minimum, for the curvature: the errors are "
            "±%s, ±%s and ±%s." % (m["hesse"], *m["errors_text"]),
        ][k]

    # ------------------------------------------------------------------ the state at t
    def latest(self):
        """The index of the last call asked by now (-1 before the start)."""
        return int(np.searchsorted(self.call_t, getattr(self, "_t", 0.0), side="right")) - 1

    def base_of(self, n):
        """The point the calls are asked from once call ``n`` is the latest."""
        return self.tl[n]["base"] if n >= 0 else 0

    def steady(self, sc, k, u, op):
        m, tl, S = self.meta, self.tl, self.space
        n = self.latest()
        # the space and its box
        sp = op * self.at_u(0, 0.25, 0.6)
        self.draw_space(sc, sp)
        # the data and the model's curve, which follows the start and each try
        ch_op = op * self.at_u(0, 0.04, 0.6)
        ys = None
        if n >= 0:
            shown = [i for i in range(n + 1) if tl[i]["kind"] in ("start", "try")]
            now = shown[-1]
            ys = np.asarray(m["curves"][now])
            if len(shown) > 1:
                prev = np.asarray(m["curves"][shown[-2]])
                g = self.since(self.call_t[now], 0.6)
                ys = prev + (ys - prev) * ease_in_out(g)
        self.draw_chart(sc, ch_op, ys, curve_op=self.at_u(0, 0.3, 0.5) if k == 0 else 1.0)
        if n < 0:
            return
        # the path: the kept points so far, joined
        base = self.base_of(n)
        kept = [i for i in m["kept_calls"] if i <= base]
        pts = [S.at(tl[i]["point"]) for i in kept]
        self.guide(sc, pts, PATH, op, width=2.0)
        for j, i in enumerate(kept):
            last = j == len(kept) - 1
            self.point(sc, tl[i]["point"], PATH, op, r=5.0 if last else 3.8)
        # the tries: each drawn as an arrow from its base when asked, kept or dropped after
        for e in tl[:n + 1]:
            if e["kind"] != "try" or e["near"]:
                continue
            i = e["i"]
            a = S.at(tl[e["base"]]["point"])
            b = S.at(e["point"])
            g = self.since(self.call_t[i], 0.7)
            later = n > i
            ink = CHANGE_INK[e["change"]]
            if not later:
                self.arrow(sc, a, b, ink, op, u=g, width=2.0)
                self.point(sc, e["point"], ink, op * g, r=4.5)
            elif not e["kept"]:
                self.dashed(sc, a, b, RED, 0.4 * op, n=9)
                self.point(sc, e["point"], RED, 0.6 * op, r=3.6, hollow=True)
        # the start's and the first step's numbers
        start = S.at(tl[0]["point"])
        self.text(sc, start[0] + 0.012, start[1] + 0.035, m["start_text"],
                  op * self.since(self.call_t[0], 0.4) * (1 - self.at_u(3, 0.03)), SIZE, "lm",
                  ink=LOSS_INK)
        if n >= 13:
            ov = S.at(tl[13]["point"])
            a = op * self.since(self.call_t[13] + 0.5) * (1 - self.at_u(3, 0.03))
            self.text(sc, ov[0] - 0.008, ov[1] - 0.032, "too far: " + m["overshoot_text"], a,
                      SIZE, "rm", ink=dark(RED, 0.2))
        if n >= 14:
            bk = S.at(tl[14]["point"])
            a = op * self.since(self.call_t[14] + 0.5) * (1 - self.at_u(3, 0.03))
            self.text(sc, bk[0] + 0.012, bk[1] + 0.03, m["back_text"], a, SIZE, "lm",
                      ink=LOSS_INK)
        # the inset: the calls of this round round the point they are asked from
        self.draw_inset(sc, k, n, op)
        # the counter: this call, what it moved, its chi-square and the change
        self.counter(sc, n, op * self.since(self.call_t[0], 0.4))
        # under the space: the kept chi-squares, then the values
        if k >= 4:
            a = op * self.at_u(4, 0.4)
            vals = m["values_text"]
            errs = m["errors_text"]
            hb = self.at_u(5, 0.6)
            for i, x in enumerate((0.065, 0.19, 0.315)):
                txt = "%s = %s" % (m["names"][i], vals[i])
                self.text(sc, x, ROW_Y, txt, a, SIZE, "lm", strong=True)
                w = layout.text_width(txt, SIZE) + 0.007
                self.text(sc, x + w, ROW_Y, "± " + errs[i], op * hb, SIZE, "lm", ink=QUIET)

    def counter(self, sc, n, op):
        if op <= 0.01 or n < 0:
            return
        m, e = self.meta, self.tl[n]
        names = m["names"]
        if e["kind"] == "start":
            head = "the start"
        elif e["kind"] == "probe":
            head = "call %d: %s %s" % (n, names[e["axis"]], WAY[e["axis"]][e["sign"] > 0])
            sized = "%s %s" % (head, m["step_text"][n].lstrip("+−"))
            if layout.text_width(sized, SIZE) <= COL_W:
                head = sized
        elif e["kind"] == "hesse":
            head = "call %d: HESSE" % n
        else:
            head = "call %d: a step" % n if not e["near"] else "call %d: two axes" % n
        if self.since(self.call_t[-1] + 0.6) > 0.5:
            head = "%d + %d calls" % (m["migrad_calls"], m["hesse"])
        elif n == 64 and self._t > self.call_t[64] + 0.8:
            head = "%d calls" % m["migrad_calls"]
        rows = [(head, SIZE, INK), ("χ² " + m["chi2_text"][n], HEAD, LOSS_INK)]
        if head == "%d calls" % m["migrad_calls"]:
            rows.append(("EDM " + m["edm_text"], SIZE, INK))
        elif e["kind"] != "start":
            rows.append(("%s  %s" % (m["delta_text"][n], e["change"])
                         if e["change"] != "flat" else "flat: under 0.0001", SIZE,
                         CHANGE_TEXT[e["change"]]))
        self.lines_at(sc, rows, op)

    def draw_inset(self, sc, k, n, op):
        tl, S = self.tl, self.space
        a = op * self.since(self.call_t[1] - 0.6, 0.5)
        if a <= 0.01:
            return
        base_i = self.base_of(n)
        # the ring sits on the point asked from; while a try is under way, still on its base
        src = S.at(tl[base_i]["point"])
        bars = self.at_u(5, 0.45, 0.5)
        self.inset_box(sc, a, src, "magnified")
        cx, cy = INSET
        base = tl[base_i]["point"]
        self.inset_axes(sc, base, a)
        rnd = tl[n]["round"]
        scale = self.round_scale.get(rnd)
        probes = a * (1.0 - bars)
        if scale is not None and probes > 0.01:
            for e in tl[:n + 1]:
                if e["round"] != rnd or not e["near"]:
                    continue
                g = probes * self.since(self.call_t[e["i"]], 0.2)
                dx, dy = self.offset(base, e["d"], scale)
                ink = CHANGE_INK[e["change"]]
                newest = e["i"] == n
                if abs(dx) + abs(dy) > 1e-9:
                    self.guide(sc, [(cx, cy), (cx + dx, cy + dy)], ink, g, width=2.4 if newest
                               else 1.8)
                self.disk(sc, self.P(cx + dx, cy + dy), 4.2 if newest else 3.2, ink, g,
                          lift=1.6, outline=0.1)
        self.disk(sc, self.P(cx, cy), 4.5, PATH, a, lift=1.7, outline=0.1)
        # HESSE's answer: the error bars, magnified less
        if bars > 0.01:
            self.error_bars(sc, base, a * bars)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        pass

    def phase_2(self, sc, u, op):
        pass

    def phase_3(self, sc, u, op):
        pass

    def phase_4(self, sc, u, op):
        pass

    def phase_5(self, sc, u, op):
        pass

    def phase_6(self, sc, u, op):
        pass


# ====================================================================== 270
class Versus(Fit):
    """Minuit's few steps; SGD's batch and first step; the walk; the wandering at the end,
    magnified, beside Minuit's minimum; the counts."""

    TITLE = "Minuit and SGD"
    PHASES = (("Minuit", 9.0), ("a batch", 12.0), ("the walk", 14.0), ("the end", 13.0),
              ("the counts", 12.0))
    LAST = "How Minuit feels its way"
    NEXT = "Error bars on a network"
    AVERAGE_U = 0.45            # phase 4: the average of the last 1,000 is drawn, and counted
    FIRST_U = 0.6               # phase 2: SGD's first step
    FIRST = 10                  # phase 2 takes the first 10 steps; phase 3 the rest
    ZOOM = (0.55, 0.7)          # phase 4: the inset magnifies less, from the walk to the errors

    def setup(self):
        super().setup()
        m = self.meta
        self.path = np.asarray(m["path"])
        self.walk_idx = np.asarray(m["walk_idx"])
        self.walk_pts = np.array([self.space.at(self.path[i]) for i in self.walk_idx])
        self.tail = np.asarray(m["tail"])
        self.CODE = (
            Code("mn-migrad", run=((0.0, 2),), out=((0.1, m["printed"]),)),
            Code("mn-sgd", run=((0.0, 0), (0.35, 2), (0.6, 3))),
            Code("mn-sgd", run=((0.0, 3),)),
            Code("mn-sgd", run=((0.0, 3),)),
            Code("mn-sgd", run=((0.0, 0),)))
        self.EVIDENCE = (
            measured("iminuit %s's own calls" % m["iminuit"], ON),
            measured("SGD's first step, seed %d" % m["seed"], ON),
            measured("one SGD run of %s steps" % format(m["steps"], ","), ON),
            measured("SGD's last 1,000 positions", ON),
            exact("3,000 × 8 = 24,000; 65 × 40 = 2,600"))

    def heading_for(self, k, u):
        return ("Minuit: a handful of careful steps",
                "SGD: 8 points, one small step",
                "3,000 small steps: a drunken walk",
                "at the end it wanders round the minimum",
                "points the model ran on, each way")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "Last time: Minuit used all 40 points for every call, %d calls in all, and kept "
            "%d steps to the minimum." % (m["migrad_calls"], m["kept"]),
            "SGD (stochastic gradient descent): each step looks at a batch of %d of the 40 "
            "points and moves the learning rate, %g, times their slope." % (m["batch"], m["lr"]),
            "%s such steps from the same start: each batch pulls a different way, so the "
            "path wobbles, but it gets there." % format(m["steps"], ","),
            "Magnified: its last 1,000 positions wander (spread about %s); their average has "
            "chi-square %s, Minuit's %s." % (m["spread_text"], m["average_chi2_text"],
                                            m["minuit_chi2_text"]),
            "The model ran on one point %s times for SGD, %s for MIGRAD. A small fit is "
            "Minuit's, with error bars; SGD pays off on huge data." % (
                m["sgd_evaluations_text"], m["minuit_evaluations_text"]),
        ][k]

    def walk_upto(self):
        """How many of SGD's steps are taken by now: none, then the first (phase 2), then
        steps 2 to 10 one by one; in phase 3 on to 200 slowly, then the rest."""
        k, u = self.where(getattr(self, "_t", 0.0))
        if k == 0:
            return 0
        if k == 1:
            if u < self.FIRST_U:
                return 0
            return 1 + int(round((self.FIRST - 1) * self.prog(1, self.FIRST_U + 0.12, 0.95)))
        if k > 2:
            return self.meta["steps"]
        k2 = self.prog(2, 0.4, 0.9)
        if k2 <= 0:
            return int(round(self.FIRST + (200 - self.FIRST) * self.prog(2, 0.05, 0.4)))
        return int(round(200 + (self.meta["steps"] - 200) * k2))

    def steady(self, sc, k, u, op):
        m, S = self.meta, self.space
        self.draw_space(sc, op)
        # Minuit's kept path and minimum, all through
        pts = [S.at(p) for p in m["kept_points"]]
        mp = op * (self.at_u(0, 0.1, 0.8) if k == 0 else 1.0)
        self.guide(sc, pts, PATH, mp, width=2.0)
        for j, p in enumerate(m["kept_points"]):
            self.point(sc, p, PATH, mp, r=5.0 if j == len(pts) - 1 else 3.8)
        # SGD's walk so far: the first step an arrow, then the path through every step
        upto = self.walk_upto()
        if upto > 0:
            if k == 1:
                # the first step is too short for an arrow's head: a line, and the dot moving
                g = self.prog(1, self.FIRST_U, self.FIRST_U + 0.1)
                self.guide(sc, [S.at(p) for p in self.path[:upto + 1]], WALK, op * g,
                           width=1.8)
            else:
                sel = self.walk_idx <= upto
                self.guide(sc, [tuple(p) for p in self.walk_pts[sel]], WALK, op * 0.9,
                           width=1.4)
            self.point(sc, self.path[upto], WALK, op * (g if k == 1 else 1.0), r=4.5, lift=1.6)
        # the counter, its rows swapped in place (never blank): Minuit's result; SGD's step
        # and chi-square on all 40 points; the average of its last 1,000 against Minuit's
        if k == 0:
            rows = [("%d calls, %d kept" % (m["migrad_calls"], m["kept"]), SIZE, INK),
                    ("χ² %.2f" % m["fval"], HEAD, LOSS_INK), ("Minuit", SIZE, INK)]
            self.lines_at(sc, rows, op * self.at_u(0, 0.2))
        elif k in (1, 2) or (k == 3 and u < self.AVERAGE_U):
            step = self.walk_upto()
            p = self.path[step]
            chi = float(np.sum(((self.data_y - (p[0] * np.exp(-self.data_x / p[1]) + p[2]))
                                / m["sigma"]) ** 2))
            rows = [("step %s of %s" % (format(step, ","), format(m["steps"], ",")), SIZE, INK),
                    ("χ² %s" % _chi(chi), HEAD, LOSS_INK), ("on all 40 points", SIZE, QUIET)]
            self.lines_at(sc, rows, op)
        else:
            rows = [("last 1,000, averaged", SIZE, WALK_INK),
                    ("χ² %s" % m["average_chi2_text"], HEAD, LOSS_INK),
                    ("Minuit: %s" % m["minuit_chi2_text"], SIZE, INK)]
            self.lines_at(sc, rows, op)
        # the data chart: phases 1-3
        c = op * self.kept(0, 2)
        if c > 0.01:
            if k == 0:
                ys = np.asarray(m["curve_minuit"])
                ink = PATH
            else:
                p = self.path[self.walk_upto()]
                ys = p[0] * np.exp(-self.grid / p[1]) + p[2]
                ink = WALK
            picked = m["first"]["idx"] if k == 1 else None
            pk = self.prog(1, 0.06, 0.15) if k == 1 else 0.0
            self.draw_chart(sc, c, ys, ink=ink, picked=picked if pk > 0.5 else None)

    def phase_1(self, sc, u, op):
        pass

    def phase_2(self, sc, u, op):
        m = self.meta
        cx, cy = INSET
        self.text(sc, COL_X, 0.70, "batch: %d of 40 points" % m["batch"], on(u, 0.12, op),
                  SIZE, "lm", strong=True)
        self.text(sc, COL_X, 0.75, "learning rate: %g" % m["lr"], on(u, 0.38, op), SIZE, "lm",
                  ink=dark(TEAL, 0.35))
        self.text(sc, COL_X, 0.80, "step: rate × slope", on(u, 0.6, op), SIZE, "lm",
                  ink=WALK_INK)
        del cx, cy

    def phase_3(self, sc, u, op):
        pass

    def phase_4(self, sc, u, op):
        m, S = self.meta, self.space
        base = np.asarray(m["values"])
        self.inset_box(sc, on(u, 0.03, op), S.at(base), "magnified")
        cx, cy = INSET
        a = on(u, 0.06, op)
        self.inset_axes(sc, base, a)
        # first magnified to fit the wandering, then less, to Minuit's error bars round it
        near = self.scale_for(base, self.tail - base)
        far = self.scale_for(base, [np.eye(3)[i] * m["errors"][i] for i in range(3)], r=ERR_R)
        scale = near * (far / near) ** eased(u, self.ZOOM[0], self.ZOOM[1])
        self.error_bars(sc, base, a * 0.85 * eased(u, self.ZOOM[1] - 0.02, self.ZOOM[1] + 0.08),
                        width=2.0)
        self.disk(sc, self.P(cx, cy), 4.5, PATH, a, lift=1.85, outline=0.1, ring=PAPER,
                  ring_w=1.5)
        # SGD's last 1,000 positions, wandering
        n = int(len(self.tail) * eased(u, 0.1, 0.42))
        if n > 1:
            pts = [(cx + o[0], cy + o[1]) for o in
                   (self.offset(base, p - base, scale) for p in self.tail[:n:2])]
            self.guide(sc, pts, WALK, on(u, 0.1, op) * 0.8, width=1.1)
        # their average
        dx, dy = self.offset(base, np.asarray(m["sgd_average"]) - base, scale)
        self.disk(sc, self.P(cx + dx, cy + dy), 4.5, WALK, on(u, self.AVERAGE_U, op), lift=1.8,
                  outline=0.2, ring=PAPER, ring_w=1.5)
        b = on(u, 0.8, op)
        self.text(sc, 0.065, ROW_Y, "SGD's spread ≈ %s" % m["spread_text"], b, SIZE, "lm",
                  ink=WALK_INK)
        self.text(sc, 0.065 + 0.17, ROW_Y, "Minuit's errors ≈ 0.2", b, SIZE, "lm", ink=INK)

    def phase_5(self, sc, u, op):
        m = self.meta
        x0 = COL_X
        top = 0.56
        full = 92.0
        rows = (("SGD   3,000 × 8", m["sgd_evaluations"], m["sgd_evaluations_text"], WALK),
                ("MIGRAD   65 × 40", m["minuit_evaluations"], m["minuit_evaluations_text"],
                 PATH))
        for j, (label, v, text, ink) in enumerate(rows):
            a = on(u, 0.08 + 0.22 * j, op)
            y = top + 0.03 + j * 0.105
            self.text(sc, x0, y, label, a, SIZE, "lm", ink=WALK_INK if j == 0 else INK)
            length = full * v / m["sgd_evaluations"]
            self.hbar(sc, x0, y + 0.045, length, ink, a, value=text,
                      u=eased(u, 0.08 + 0.22 * j, 0.24 + 0.22 * j), thick=12.0)
        self.text(sc, 0.065, ROW_Y, "small fit: Minuit · huge data: SGD", on(u, 0.5, op), SIZE,
                  "lm", strong=True)


def _chi(v):
    v = float(v)
    if v >= 1000:
        return f"{v:,.0f}"
    if v >= 100:
        return "%.0f" % v
    return "%.2f" % v


ANIMATIONS = {"minuit-feels-its-way": Feels, "minuit-and-sgd": Versus}

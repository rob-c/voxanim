"""Randomness and entropy: animations 110-114 of the "next" catalogue (see :mod:`voxanim.mech`).

How a generator makes numbers, what a seed does, how even numbers become draws from a
distribution, where unpredictability comes from, and how entropy is estimated. Every
state, output and count on screen is computed by :mod:`voxanim.study_random`, whose small
generators are written out in integer arithmetic; this module only lays them out. Each
phase says, bottom right, what kind of evidence it shows: exact arithmetic, statistics
of a generator's output, real text, or a measurement of the machine that built it.
"""

from __future__ import annotations

import numpy as np

from . import layout
from . import study_random as SR
from .diagram import draw_image
from .easing import smoothstep
from .losschart import pow10, sup
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
    WHITE,
    Mechanism,
    dark,
    soft,
)
from .mech_curios import progress

EXACT = "exact arithmetic"
STATS = "statistics of the generators' outputs"
LOCAL = "measured on this machine: different on every other"
TEXT_REAL = "real text: Tiny Shakespeare"
BIG, HEAD = 0.036, 0.034
STRONG = layout.STRONG_INK


def _rgb(c):
    return np.asarray(c, np.float32)


def commas(n):
    """An integer with thousands separators."""
    return format(int(n), ",")


def two_to(k):
    """``2^k`` written with a superscript."""
    return "2" + sup(int(k))


def bit_list(value, n):
    """The ``n`` bits of an integer, highest first."""
    return [(int(value) >> (n - 1 - k)) & 1 for k in range(n)]


class Random(Mechanism):
    """What the five animations share: bit rows, rastered point clouds, the evidence tag."""

    def tag(self, sc, kind, op):
        """The kind of evidence on screen, bottom right."""
        self.text(sc, 0.95, 0.925, kind, op, anchor="rm", ink=dark(TEAL, 0.25))

    def cells(self, sc, values, fx, fy, w, h, op, ink=BLUE, off=PAPER, grid=0.35, frame=True):
        """A matrix of 0..1 values as cells from ``off`` to ``ink``, ``w x h`` px centred
        at ``(fx, fy)``. Returns ``cell(row, col) -> (fx, fy)``."""
        v = np.atleast_2d(np.asarray(values, np.float32))[..., None]
        tex = _rgb(off) * (1 - v) + _rgb(ink) * v
        c = self.P(float(fx), float(fy))
        if frame:
            self.rect(sc, c, w + 6, h + 6, PAPER, op, border=AXIS, border_w=1.2, radius=4,
                      lift=0.1)
        draw_image(self, sc, c, tex, w, h, op, grid=grid)
        rows, cols = v.shape[:2]
        dw, dh = self.frac(w, h)

        def cell(r, k):
            return fx - dw / 2 + (k + 0.5) * dw / cols, fy - dh / 2 + (r + 0.5) * dh / rows
        return cell

    def picture(self, sc, tex, box, op, frame=AXIS):
        """A texture filling the layout box ``((x0, x1), (top, bottom))``."""
        (x0, x1), (y0, y1) = box
        w, h = self.span(x1 - x0, y1 - y0)
        c = self.P((x0 + x1) / 2, (y0 + y1) / 2)
        if frame is not None:
            self.rect(sc, c, w + 6, h + 6, PAPER, op, border=frame, border_w=1.2, radius=4,
                      lift=0.1)
        draw_image(self, sc, c, np.asarray(tex, np.float32), w, h, op, grid=0)

    def ring(self, sc, fx, fy, w, h, ink, op, width=2.4):
        """An outline ``w x h`` px centred at ``(fx, fy)``."""
        self.rect(sc, self.P(float(fx), float(fy)), w, h, ink, op, border=ink, border_w=width,
                  radius=3, lift=0.8, fill_op=0.0)

    def path(self, sc, chart, xs, ys, ink, op, width=2.2):
        """A polyline through chart coordinates."""
        pts = [self.P(chart.fx(float(a)), chart.fy(float(b))) for a, b in zip(xs, ys)]
        self.polyline(sc, pts, width, ink, op, bias=0.3)

    def vbars(self, sc, values, box, op, ink, vmax=None, gap=0.2):
        """Vertical bars filling the layout box from its bottom edge."""
        (x0, x1), (y0, y1) = box
        v = np.asarray(values, float)
        vmax = float(v.max()) if vmax is None else vmax
        step = (x1 - x0) / len(v)
        wpx = self.span(step * (1 - gap))[0]
        for k, val in enumerate(v):
            hgt = (y1 - y0) * float(val) / vmax
            if hgt <= 0:
                continue
            self.rect(sc, self.P(x0 + (k + 0.5) * step, y1 - hgt / 2), wpx,
                      max(self.span(0, hgt)[1], 0.6), ink, op, radius=min(2.0, wpx / 2),
                      lift=0.4)


def raster(x, y, xr, yr, shape, colours, back=PAPER, fat=1):
    """Points ``(x, y)`` drawn into a ``shape`` (rows, cols) texture: ``xr`` and ``yr``
    are the ranges shown, ``colours`` one colour or one per point, y upwards."""
    rows, cols = shape
    tex = np.empty((rows, cols, 3), np.float32)
    tex[:] = _rgb(back)
    x, y = np.asarray(x, float), np.asarray(y, float)
    col = ((x - xr[0]) / (xr[1] - xr[0]) * cols).astype(int)
    row = ((1.0 - (y - yr[0]) / (yr[1] - yr[0])) * rows).astype(int)
    ok = (col >= 0) & (col < cols - fat + 1) & (row >= 0) & (row < rows - fat + 1)
    colours = np.asarray(colours, np.float32)
    paint = colours[ok] if colours.ndim == 2 else colours
    for a in range(fat):
        for b in range(fat):
            tex[row[ok] + a, col[ok] + b] = paint
    return tex


def grey(b, cols):
    """Bytes as a texture of grey cells, ``cols`` to a row: 0 is paper, 255 near black."""
    v = (np.asarray(b, np.float32) / 255.0).reshape(-1, cols)[..., None]
    return _rgb(PAPER) * (1 - v) + _rgb((0.13, 0.15, 0.17)) * v


# ====================================================================== 110: generators
GEN_INK = (RED, AMBER, VIOLET, BLUE, GREEN)      # RANDU, 32-bit LCG, xorshift32, PCG32, numpy


class PseudoRandomGenerators(Random):
    """A 4-bit linear congruential generator stepped in real arithmetic round its whole
    cycle; a 32-bit one whose low bits repeat at once; RANDU's triples turned until
    their 15 planes show; xorshift and PCG's output function bit by bit; the scores."""

    TITLE = "Pseudo-random numbers"
    PHASES = (("one step", 9.0), ("the cycle", 9.0), ("low bits", 10.0), ("planes", 11.0),
              ("scramble", 10.0), ("scores", 9.0))
    RING = (0.745, 0.605, 112.0)                    # centre and radius (px) of the big ring

    def setup(self):
        d = self.d = SR.prg_data()
        s = self.study
        self.cycle = [int(v) for v in s.panels[0].data["ys"][0]]
        self.short = [int(v) for v in s.panels[0].data["short_cycle"]]
        self.bit_period = [int(v) for v in s.panels[1].data["low_bit_period"]]
        self.periods = s.metadata["periods"]
        self.stats = np.asarray(s.panels[2].data["values"], float)        # (generator, stat)
        self.ideal = float(s.panels[2].data["ideal_cells"])
        self.planes = len(s.panels[3].data["planes"])
        self.rows = [int(v) for v in d["lcg_rows"][:12]]
        self.side = SR.PRG_SIDE
        # the view in which RANDU's planes are edge on, and the axis the view turns about
        n = np.array([9.0, -6.0, 1.0])
        self.normal = n / np.linalg.norm(n)
        a = np.array([0.0, 1.0, 6.0])
        self.across = a / np.linalg.norm(a)
        self.depth = np.cross(self.across, self.normal)
        self.triples = [np.stack([p[:-2], p[1:-1], p[2:]], 1) - 0.5
                        for p in (d["randu_points"], d["good_points"])]

    def subtitle_for(self, k):
        st = self.stats
        return [
            "a generator is a small state and a rule: here 4 bits, and x becomes 5x + 3 with "
            "only the low 4 bits kept. Each number it gives is its state, read off",
            "the rule can only walk a fixed path: this one visits all 16 states and is back "
            "where it began, period 16. Change + 3 to + 4 and it is back after %d steps" % (
                len(self.short) - 1),
            "the same rule with 32 bits has period %s, but its lowest bit just alternates "
            "and each bit above takes twice as long: the low byte comes round every %d "
            "outputs" % (commas(2 ** 32), self.bit_period[7]),
            "RANDU, x times 65539 with 31 bits kept: each number looks fine, but every three "
            "in a row lie on one of %d planes, and reach %.0f%% of a cube's %s cells" % (
                self.planes, 100 * st[0, 1], commas(self.side ** 3)),
            "better scrambles: xorshift folds the state onto itself with shifts and XORs; "
            "PCG keeps the simple rule on 64 bits and reads it through a shift and a turn",
            "mean of three seeds: all five have serial correlation near 0, which sees "
            "nothing; RANDU fills %.0f%% of the cells, the rest %.0f%%; only the plain "
            "rule's low bit repeats" % (100 * st[0, 1], 100 * st[1:, 1].mean()),
        ][k]

    # ------------------------------------------------------------------ drawing
    def node(self, v, centre=None):
        """Layout position of state ``v`` on a ring of 16."""
        fx, fy, r = centre or self.RING
        dx, dy = self.frac(r * np.sin(2 * np.pi * v / 16), -r * np.cos(2 * np.pi * v / 16))
        return fx + dx, fy + dy

    def wheel(self, sc, path, steps, op, centre=None, ink=AMBER, size=24.0, now=True):
        """The 16 states on a ring, the first ``steps`` (fractional) steps of ``path``
        drawn as arrows, the current state lit."""
        done = int(steps)
        seen = set(path[:done + 1])
        here = path[min(done, len(path) - 1)]
        for v in range(16):
            x, y = self.node(v, centre)
            lit = now and v == here
            fill = ink if lit else (soft(ink, 0.6) if v in seen else PAPER)
            self.box(sc, x, y, size, size, fill, op, str(v), ink=WHITE if lit else None,
                     border=ink if v in seen else AXIS, radius=size / 2)
        for k in range(min(done + 1, len(path) - 1)):
            a = np.array(self.span(*self.node(path[k], centre)))
            b = np.array(self.span(*self.node(path[k + 1], centre)))
            way = (b - a) / max(float(np.hypot(*(b - a))), 1e-6) * size * 0.62
            go = 1.0 if k < done else steps - done
            self.arrow(sc, self.frac(*(a + way)), self.frac(*(b - way)), ink,
                       op * (0.55 if k < done else 1.0), go, width=1.8)

    def phase_1(self, sc, u, op):
        path = self.cycle
        steps = min(u / 0.9, 1.0) * 4.0
        done = min(int(steps), 3)
        x = path[done]
        self.text(sc, 0.06, 0.315, "the state: 4 bits", op, anchor="lm", strong=True, size=HEAD)
        self.cells(sc, [bit_list(x, 4)], 0.125, 0.40, 136, 34, op, ink=AMBER)
        self.text(sc, 0.215, 0.40, "= %d" % x, op, anchor="lm", strong=True, size=BIG)
        self.text(sc, 0.06, 0.49, "the rule: x becomes 5 × x + 3, low 4 bits kept", op,
                  anchor="lm", size=0.032)
        for k in range(4):
            a, b = path[k], path[k + 1]
            show = op * smoothstep(k, k + 0.35, steps)
            y = 0.59 + 0.078 * k
            self.text(sc, 0.06, y, "5 × %d + 3 = %d" % (a, 5 * a + 3), show, anchor="lm",
                      size=BIG)
            self.text(sc, 0.255, y, "→  %d" % b, show * smoothstep(k + 0.3, k + 0.6, steps),
                      anchor="lm", size=BIG, strong=True)
            self.text(sc, 0.345, y, "output %d/16 = %.4f" % (b, b / 16.0),
                      show * smoothstep(k + 0.5, k + 0.8, steps), anchor="lm",
                      ink=dark(AMBER, 0.3))
        self.wheel(sc, path, steps, op)
        self.text(sc, self.RING[0], 0.315, "all 16 possible states", op, strong=True, size=HEAD)
        self.tag(sc, EXACT, op)

    def phase_2(self, sc, u, op):
        path = self.cycle
        steps = 4.0 + min(u / 0.55, 1.0) * 12.0
        done = int(steps)
        self.wheel(sc, path, steps, op, now=done < 16)
        self.text(sc, self.RING[0], 0.315, "5 × x + 3:  step %d of 16" % min(done, 16), op,
                  strong=True, size=HEAD)
        self.text(sc, 0.06, 0.315, "states visited, in order", op, anchor="lm", strong=True,
                  size=HEAD)
        for k in range(17):
            show = op * smoothstep(k - 0.4, k, steps)
            x, y = 0.085 + 0.047 * (k % 9), 0.385 + 0.068 * (k // 9)
            back = k == 16
            self.box(sc, x, y, 34, 28, AMBER if back or k == 0 else soft(AMBER, 0.6), show,
                     str(path[k]), ink=WHITE if back or k == 0 else None, border=AMBER)
        self.text(sc, 0.06, 0.525, "back at 1 after 16 steps: the period", op * smoothstep(
            15.6, 16.0, steps), anchor="lm", size=0.032)
        late = smoothstep(0.58, 0.68, u)
        small = (0.135, 0.745, 62.0)
        self.wheel(sc, self.short, (len(self.short) - 1) * smoothstep(0.66, 0.92, u), op * late,
                   centre=small, ink=RED, size=17.0, now=False)
        self.text(sc, 0.245, 0.69, "5 × x + 4 instead:", op * late, anchor="lm", size=0.032)
        self.text(sc, 0.245, 0.745, "1 → 9 → 1", op * late, anchor="lm", size=BIG, strong=True)
        self.text(sc, 0.245, 0.80, "period %d" % (len(self.short) - 1), op * smoothstep(
            0.85, 0.95, u), anchor="lm", size=0.032, ink=dark(RED))
        self.tag(sc, EXACT, op)

    def phase_3(self, sc, u, op):
        rows = self.rows
        n = len(rows)
        shown = int(np.ceil(min(u / 0.5, 1.0) * n))
        bits_ = np.array([bit_list(v, 32) for v in rows], np.float32)
        bits_[shown:] = 0.0
        self.text(sc, 0.27, 0.315, "x becomes 1664525 × x + 1013904223, 32 bits kept", op,
                  size=0.031)
        cell = self.cells(sc, bits_, 0.27, 0.59, 384, 168, op, ink=AMBER)
        x0, y0 = cell(0, 28)
        x1, y1 = cell(n - 1, 31)
        self.ring(sc, (x0 + x1) / 2, (y0 + y1) / 2, 4 * 12 + 4, 168 + 6, RED, op * smoothstep(
            0.45, 0.6, u))
        self.text(sc, 0.27, 0.368, "its first %d outputs, one per row, as 32 bits" % n, op)
        last = rows[max(shown, 1) - 1]
        self.text(sc, 0.27, 0.815, "output %d:  %s" % (max(shown, 1), commas(last)), op,
                  size=0.034, strong=True)
        low = "".join(str(v & 1) for v in rows[:shown])
        self.text(sc, 0.27, 0.87, "lowest bit, row by row:  " + " ".join(low), op * smoothstep(
            0.5, 0.62, u), ink=dark(RED))
        ks = np.arange(16)
        chart = self.chart(((0.63, 0.94), (0.40, 0.74)), [0, 15], [1, 2 ** 17], ylog=True)
        self.axes(sc, chart, op, "bit (0 is the lowest)", "outputs before the bit repeats",
                  xticks=[(v, "%d" % v) for v in (0, 5, 10, 15)],
                  yticks=[(2.0 ** v, two_to(v)) for v in (1, 6, 11, 16)])
        go = smoothstep(0.55, 0.9, u)
        for k in ks[:int(np.ceil(go * 16))]:
            self.disk(sc, self.P(chart.fx(k), chart.fy(self.bit_period[k])), 4.6, RED, op,
                      lift=1.2, outline=0.0)
        self.text(sc, 0.785, 0.315, "measured on %s outputs" % commas(1 << 18), op)
        self.tag(sc, EXACT, op)

    def cloud(self, sc, pts, angle, box, ink, op):
        up = np.cos(angle) * self.normal + np.sin(angle) * self.depth
        tex = raster(pts @ self.across, pts @ up, (-0.87, 0.87), (-0.87, 0.87), (132, 132),
                     ink)
        self.picture(sc, tex, box, op)

    def phase_4(self, sc, u, op):
        turn = 1.0 - smoothstep(0.12, 0.62, u)
        angle = np.radians(38.0) * turn
        self.cloud(sc, self.triples[0], angle, ((0.07, 0.335), (0.375, 0.845)), RED, op)
        self.cloud(sc, self.triples[1], angle, ((0.385, 0.65), (0.375, 0.845)), BLUE, op)
        self.text(sc, 0.2025, 0.315, "RANDU: 6,000 triples", op, strong=True, size=HEAD)
        self.text(sc, 0.5175, 0.315, "PCG32: 6,000 triples", op, strong=True, size=HEAD)
        self.text(sc, 0.36, 0.885, "three outputs in a row = one point in a cube; the cube "
                  "is turned %d°" % round(np.degrees(angle)), op)
        late = op * smoothstep(0.6, 0.72, u)
        self.lines(sc, 0.69, 0.40, [
            ("seen edge on:", None),
            ("%d planes" % self.planes, dark(RED)),
            ("hold every triple", None)], late, size=BIG, step=0.055)
        last = op * smoothstep(0.76, 0.88, u)
        self.lines(sc, 0.69, 0.62, [
            "%s triples into" % commas(SR.PRG_N),
            "%s cells:" % commas(self.side ** 3)], last, step=0.045)
        full = self.span(0.19)[0]
        self.hbar(sc, 0.69, 0.735, full * self.stats[0, 1], RED, last, "",
                  "%.0f%%" % (100 * self.stats[0, 1]), (u - 0.76) / 0.12, thick=18.0, track=full)
        self.hbar(sc, 0.69, 0.80, full * self.stats[3, 1], BLUE, last, "",
                  "%.0f%%" % (100 * self.stats[3, 1]), (u - 0.80) / 0.12, thick=18.0, track=full)
        self.tag(sc, STATS, op)

    def phase_5(self, sc, u, op):
        xs = [int(v) for v in self.d["xs_demo"]]
        names = ("the state", "shifted left 13", "XOR", "shifted right 17", "XOR",
                 "shifted left 5", "XOR: new state")
        inks = (VIOLET, DIM, VIOLET, DIM, VIOLET, DIM, VIOLET)
        self.text(sc, 0.06, 0.315, "xorshift32: one step", op, anchor="lm", strong=True,
                  size=HEAD)
        for k, (name, v) in enumerate(zip(names, xs)):
            show = op * smoothstep(0.04 + 0.075 * k, 0.10 + 0.075 * k, u)
            y = 0.385 + 0.064 * k
            ink = inks[k] if k % 2 == 0 else soft(VIOLET, 0.35)
            self.cells(sc, [bit_list(v, 32)], 0.365, y, 240, 20, show, ink=ink, frame=False)
            self.text(sc, 0.06, y, name, show, anchor="lm", strong=k in (0, 6))
        self.text(sc, 0.06, 0.845, "no multiplying; period %s − 1" % two_to(32), op * smoothstep(
            0.55, 0.65, u), anchor="lm")
        old, mixed, rot, word = (int(v) for v in self.d["pcg_demo"])
        self.text(sc, 0.56, 0.315, "PCG32: reading the state", op, anchor="lm", strong=True,
                  size=HEAD)
        cell = self.cells(sc, [bit_list(old, 64)], 0.75, 0.40, 336, 20, op, ink=BLUE,
                          frame=False)
        self.text(sc, 0.75, 0.455, "64 bits of state, stepped by a plain multiply-and-add",
                  op)
        show = op * smoothstep(0.5, 0.6, u)
        x0 = cell(0, 0)[0]
        x4 = cell(0, 4)[0]
        self.ring(sc, (x0 + x4) / 2, 0.40, 5 * 5.25 + 4, 26, AMBER, show)
        self.text(sc, 0.56, 0.535, "top 5 bits = %d: how far to turn" % rot, show, anchor="lm",
                  ink=dark(AMBER, 0.3))
        show = op * smoothstep(0.6, 0.7, u)
        self.cells(sc, [bit_list(mixed, 32)], 0.755, 0.61, 288, 20, show, ink=soft(BLUE, 0.35),
                   frame=False)
        self.text(sc, 0.755, 0.665, "32 bits from the top half, XORed with a shift", show)
        show = op * smoothstep(0.72, 0.82, u)
        self.cells(sc, [bit_list(word, 32)], 0.755, 0.745, 288, 20, show, ink=BLUE, frame=False)
        self.text(sc, 0.755, 0.80, "turned by %d places: the output, %s" % (rot, commas(word)),
                  show, strong=True)
        self.text(sc, 0.56, 0.855, "period %s" % two_to(64), op * smoothstep(0.82, 0.9, u),
                  anchor="lm")
        self.tag(sc, EXACT, op)

    def phase_6(self, sc, u, op):
        st = self.stats
        p = self.periods
        period = [two_to(p[k]["period"][2:4]) + (" − 1" if "-" in p[k]["period"] else "")
                  for k in ("randu", "lcg32", "xorshift32", "pcg32")] + ["not proved here"]
        low = []
        for g in range(5):
            if st[g, 3] == 0:
                low.append("never changes")
            elif st[g, 2] > 0:
                low.append("repeats every %d" % st[g, 2])
            else:
                low.append("no repeat found")
        cols = (0.06, 0.245, 0.395, 0.60, 0.865)
        heads = ("generator", "period", "lowest bit", "3-D cells reached", "serial corr.")
        for x, head in zip(cols, heads):
            self.text(sc, x, 0.325, head, op, anchor="lm", strong=True)
        full = self.span(0.18)[0]
        for g, name in enumerate(SR.PRG_NAMES):
            y = 0.405 + 0.083 * g
            show = op * smoothstep(0.05 + 0.1 * g, 0.15 + 0.1 * g, u)
            ink = GEN_INK[g]
            self.box(sc, cols[0] + 0.008, y, 14, 14, ink, show, radius=3)
            self.text(sc, cols[0] + 0.026, y, name, show, anchor="lm", ink=dark(ink, 0.3),
                      size=0.032)
            self.text(sc, cols[1], y, period[g], show, anchor="lm")
            self.text(sc, cols[2], y, low[g], show, anchor="lm",
                      ink=dark(RED) if g < 2 else None)
            self.hbar(sc, cols[3], y, full * st[g, 1], ink, show, "", "%.0f%%" % (100 * st[g, 1]),
                      (u - 0.05 - 0.1 * g) / 0.15, thick=18.0, track=full)
            self.text(sc, cols[4], y, "%+.4f" % st[g, 0], show, anchor="lm")
        self.text(sc, 0.06, 0.845, "an ideal source would reach %.0f%% of the cells; the "
                  "periods are proved, the rest measured" % (100 * self.ideal), op * smoothstep(
                      0.6, 0.75, u), anchor="lm")
        self.tag(sc, STATS, op)


# ====================================================================== 111: seeds
SEED_INK = (AMBER, BLUE, GREEN)        # the LCG seeded directly, PCG32 seeded by hash, numpy
STAGES = ("the seed", "add a constant", "XOR with a shift", "multiply", "XOR with a shift",
          "multiply", "XOR with a shift")


def unit4(x):
    """A 32-bit output as a number in [0, 1), four decimals."""
    return "%.4f" % (int(x) / 4294967296.0)


class SeedsAndStreams(Random):
    """One seed twice and its neighbour; three neighbouring seeds in a generator that
    uses the seed as its state; the exact relation between their streams; a seed hashed
    step by step while its neighbour's bits part from it; four workers sharing a seed."""

    TITLE = "Seeds and streams"
    PHASES = (("same seed", 10.0), ("neighbours", 11.0), ("related", 11.0), ("hashing", 11.0),
              ("in parallel", 12.0))

    def setup(self):
        d = self.d = SR.seeds_data()
        s = self.study
        self.equal = np.asarray(s.panels[0].data["values"], float)
        self.first = (np.asarray(s.panels[1].data["xy"])[:, 1],
                      np.asarray(s.panels[1].data["hashed"]))
        self.corr = np.asarray(s.panels[1].data["correlation"], float)
        self.gap = np.asarray(s.panels[1].data["adjacent_gap"], float)
        self.agree = (np.asarray(s.panels[2].data["values"]),
                      np.asarray(s.panels[2].data["hashed"]))
        self.relation = np.asarray(s.panels[2].data["relation"], float)
        self.pearson = np.asarray(s.panels[2].data["pearson"], float)
        self.rms = np.asarray(s.panels[3].data["values"], float)   # same, jumped, spawned, one
        self.aval = np.asarray(d["avalanche"], float)
        self.stage = [SR.splitmix64_stages(k) for k in (1, 2)]
        self.differ = [bin(a ^ b).count("1") for a, b in zip(*self.stage)]
        self.lcg = d["streams"][0]
        self.n = SR.SEED_N

    def subtitle_for(self, k):
        n = commas(self.n)
        return [
            "a seed is where the generator starts. Seed 42 twice gives the same stream, bit "
            "for bit: %s of %s outputs equal. Seed 43 gives another: %d equal, %.1f%% of "
            "bits agreeing" % (commas(self.equal[0] * self.n), n, self.equal[1] * self.n,
                               100 * self.equal[2]),
            "a weak generator uses the seed as its state. Seeds 1, 2, 3 then start almost "
            "alike: first outputs %.5f apart, and across %d seeds they follow the seed "
            "(correlation %.3f)" % (self.gap[0], SR.SEED_FIRST, self.corr[0]),
            "and the three streams stay tied: twice the second minus the first gives the "
            "third, for %.0f%% of %s outputs. The usual correlation is %.3f at most and "
            "sees none of it" % (100 * self.relation[0], n, np.abs(self.pearson[0]).max()),
            "so a seed is hashed first. Step by step the bits of seeds 1 and 2 part until "
            "%d of 64 differ; over 1,000 seeds one changed bit changes %.1f on average" % (
                self.differ[-1], self.aval[0]),
            "four workers, 1,000 darts each, estimate pi. Sharing one seed they throw the "
            "same darts: error %.3f, as one worker's. Streams jumped apart or spawned: "
            "%.3f and %.3f" % (self.rms[0], self.rms[1], self.rms[2]),
        ][k]

    # ------------------------------------------------------------------ drawing
    def stream_row(self, sc, y, seed, values, ink, op, go, x0=0.42, x1=0.95, name="PCG32",
                   w=74.0, mark=()):
        """``seed -> generator -> outputs``, the outputs appearing as ``go`` runs 0..1."""
        self.box(sc, 0.10, y, 84, 32, soft(ink, 0.5), op, "seed %d" % seed, border=ink,
                 size=0.031)
        self.arrow(sc, (0.15, y), (0.205, y), ink, op, smoothstep(0.0, 0.15, go), 2.2)
        self.box(sc, 0.27, y, 104, 36, ink, op, name, ink=WHITE, size=0.032)
        self.arrow(sc, (0.33, y), (x0 - 0.005, y), ink, op, smoothstep(0.1, 0.25, go), 2.2)
        n = len(values)
        xs = [x0 + (x1 - x0) * (k + 0.5) / n for k in range(n)]
        for k, (x, v) in enumerate(zip(xs, values)):
            show = op * smoothstep(0.2 + 0.7 * k / n, 0.2 + 0.7 * (k + 0.6) / n, go)
            self.box(sc, x, y, w, 30, soft(ink, 0.6), show, unit4(v), border=ink, size=0.031)
            if k in mark:
                self.ring(sc, x, y, w + 8, 38, RED, show)
        self.pulse(sc, (0.15, y), (xs[0], y), go / 0.35, ink, op)
        return xs

    def phase_1(self, sc, u, op):
        rows = self.d["same"]
        self.text(sc, 0.685, 0.315, "outputs, as numbers between 0 and 1", op, strong=True,
                  size=HEAD)
        for r, (seed, ink) in enumerate(((42, BLUE), (42, BLUE), (43, VIOLET))):
            go = (u - 0.03 - 0.22 * r) / 0.26
            self.stream_row(sc, 0.41 + 0.125 * r, seed, rows[r][:6], ink, op * smoothstep(
                -0.2, 0.0, go), go)
        self.text(sc, 0.06, 0.805, "seed 42 again: %s of %s outputs the same, bit for bit" % (
            commas(self.equal[0] * self.n), commas(self.n)), op * smoothstep(0.5, 0.6, u),
            anchor="lm", size=0.032, ink=dark(BLUE, 0.3))
        self.text(sc, 0.06, 0.865, "seed 43: %d the same; %.1f%% of their bits agree, as two "
                  "coins would" % (self.equal[1] * self.n, 100 * self.equal[2]),
                  op * smoothstep(0.75, 0.85, u), anchor="lm", size=0.032, ink=dark(VIOLET, 0.3))
        self.tag(sc, STATS, op)

    def phase_2(self, sc, u, op):
        self.text(sc, 0.06, 0.315, "the seed used as the state", op, anchor="lm", strong=True,
                  size=HEAD)
        for r in range(3):
            go = (u - 0.03 - 0.12 * r) / 0.22
            self.stream_row(sc, 0.41 + 0.105 * r, r + 1, self.lcg[r][:2], AMBER, op * smoothstep(
                -0.2, 0.0, go), go, x0=0.35, x1=0.53, name="LCG", w=70.0, mark=(0,))
        self.lines(sc, 0.06, 0.735, [
            ("first outputs: %s, %s, %s" % tuple(unit4(self.lcg[r][0]) for r in range(3)),
             dark(RED)),
            "the next ones already look unrelated"], op * smoothstep(0.42, 0.52, u),
            size=0.032, step=0.052)
        box = ((0.62, 0.94), (0.39, 0.68))
        chart = self.chart(box, [1, SR.SEED_FIRST], [0.0, 1.0], pad=0.0)
        self.axes(sc, chart, op, "seed", "first output",
                  xticks=[(v, "%d" % v) for v in (1, 64, 128, 192, 256)],
                  yticks=[(v, "%g" % v) for v in (0, 0.5, 1)], grid=False)
        seeds = np.arange(1, SR.SEED_FIRST + 1)
        n = int(smoothstep(0.3, 0.6, u) * len(seeds))
        m = int(smoothstep(0.62, 0.9, u) * len(seeds))
        xs = np.concatenate([seeds[:m], seeds[:n]])
        ys = np.concatenate([self.first[1][:m], self.first[0][:n]])
        col = np.concatenate([np.tile(_rgb(BLUE), (m, 1)), np.tile(_rgb(AMBER), (n, 1))])
        self.picture(sc, raster(xs, ys, (1, SR.SEED_FIRST + 1), (0.0, 1.0), (108, 196), col,
                                fat=2), box, op, frame=None)
        self.text(sc, 0.60, 0.815, "LCG: follows the seed, correlation %.3f" % self.corr[0],
                  op * smoothstep(0.5, 0.6, u), anchor="lm", ink=dark(AMBER, 0.3))
        self.text(sc, 0.60, 0.86, "PCG32, seed hashed: %.3f" % self.corr[1],
                  op * smoothstep(0.8, 0.9, u), anchor="lm", ink=dark(BLUE, 0.2))
        self.tag(sc, STATS, op)

    def phase_3(self, sc, u, op):
        x = self.lcg
        cols = (0.30, 0.445, 0.59)
        self.text(sc, 0.06, 0.315, "the LCG's streams from seeds 1, 2, 3", op, anchor="lm",
                  strong=True, size=HEAD)
        for r in range(3):
            y = 0.385 + 0.058 * r
            self.text(sc, 0.06, y, "seed %d" % (r + 1), op, anchor="lm", ink=dark(AMBER, 0.3),
                      size=0.032)
            for k, cx in enumerate(cols):
                self.text(sc, cx, y, commas(x[r][k]), op, size=0.032)
        show = op * smoothstep(0.12, 0.25, u)
        y = 0.385 + 0.058 * 3 + 0.012
        self.text(sc, 0.06, y, "2 × second − first", show, anchor="lm", strong=True, size=0.032)
        for k, cx in enumerate(cols):
            guess = (2 * int(x[1][k]) - int(x[0][k])) % (1 << 32)
            at = op * smoothstep(0.15 + 0.1 * k, 0.25 + 0.1 * k, u)
            self.text(sc, cx, y, commas(guess), at, size=0.032, strong=True)
            self.ring(sc, cx, 0.385 + 0.058 * 2, 124, 26, RED, at)
        self.lines(sc, 0.685, 0.385, [
            "the third stream,", "computed from the", "other two: right for",
            ("%s of %s" % (commas(self.relation[0] * self.n), commas(self.n)), dark(RED))],
            op * smoothstep(0.42, 0.52, u), size=0.032, step=0.05)
        late = op * smoothstep(0.55, 0.68, u)
        self.text(sc, 0.06, 0.655, "how often each of the 32 bits is the same in two streams "
                  "(lowest bit on the right)", late, anchor="lm", strong=True)
        for g, (cx, name) in enumerate(((0.285, "LCG, seed as the state"),
                                        (0.745, "PCG32, seed hashed"))):
            v = (np.asarray(self.agree[g])[:, ::-1] - 0.5) * 2
            cell = self.grid(sc, v, cx, 0.765, 300, 72, late, vmax=1.0)
            self.text(sc, cx, 0.875, name, late, ink=dark(SEED_INK[g], 0.3))
            if g == 0:
                for r, (i, j) in enumerate(SR.SEED_PAIRS):
                    self.text(sc, 0.103, cell(r, 0)[1], "%d, %d" % (i + 1, j + 1), late,
                              anchor="rm")
        self.text(sc, 0.515, 0.74, "red: always", late, ink=dark(RED))
        self.text(sc, 0.515, 0.79, "blue: never", late, ink=dark(BLUE))
        self.tag(sc, STATS, op)

    def phase_4(self, sc, u, op):
        a, b = self.stage
        self.text(sc, 0.405, 0.315, "seed 1, step by step", op, strong=True, size=HEAD)
        self.text(sc, 0.735, 0.315, "bits where seed 2 differs", op, strong=True, size=HEAD)
        for k, name in enumerate(STAGES):
            show = op * smoothstep(0.03 + 0.1 * k, 0.11 + 0.1 * k, u)
            y = 0.385 + 0.062 * k
            self.text(sc, 0.06, y, name, show, anchor="lm", strong=k in (0, 6))
            self.cells(sc, [bit_list(a[k], 64)], 0.405, y, 256, 20, show, ink=BLUE, frame=False,
                       grid=0.25)
            self.cells(sc, [bit_list(a[k] ^ b[k], 64)], 0.735, y, 256, 20, show, ink=RED,
                       frame=False, grid=0.25)
            self.text(sc, 0.895, y, "%d" % self.differ[k], show, anchor="lm", strong=True,
                      size=0.034, ink=dark(RED))
        late = op * smoothstep(0.78, 0.9, u)
        self.text(sc, 0.06, 0.845, "over 1,000 seeds, one bit changed: %.1f of 64 bits change "
                  "on average (least %d, most %d)" % tuple(self.aval), late, anchor="lm",
                  size=0.031)
        self.tag(sc, EXACT, op)

    def board(self, sc, darts, fx, fy, size, ink, op):
        inside = (darts ** 2).sum(1) < 1.0
        col = np.where(inside[:, None], _rgb(ink), _rgb(DIM) * 0.9)
        dw, dh = self.frac(size, size)
        self.picture(sc, raster(darts[:, 0], darts[:, 1], (0, 1), (0, 1), (46, 46), col),
                     ((fx - dw / 2, fx + dw / 2), (fy - dh / 2, fy + dh / 2)), op)

    def phase_5(self, sc, u, op):
        darts = self.d["darts"]
        est = self.d["pi"]
        n = int(smoothstep(0.02, 0.4, u) * darts.shape[1])
        groups = ((0.065, "one seed for all four", RED, 0), (0.545, "one stream, jumped apart",
                                                           BLUE, 1))
        for x0, name, ink, mode in groups:
            self.text(sc, x0 + 0.20, 0.315, name, op, strong=True, size=HEAD, ink=dark(ink, 0.3))
            for w in range(4):
                x = x0 + 0.05 + 0.1 * w
                self.board(sc, darts[w if mode else 0][:max(n, 1)], x, 0.445, 78, ink, op)
                self.text(sc, x, 0.565, "%.3f" % est[mode, 0, w], op * smoothstep(0.4, 0.5, u))
            self.text(sc, x0 + 0.20, 0.615, "their mean: %.3f" % est[mode, 0].mean(),
                      op * smoothstep(0.48, 0.58, u), strong=True)
        late = smoothstep(0.55, 0.65, u)
        self.text(sc, 0.06, 0.685, "error of the answer over %d runs (root mean square)"
                  % SR.PAR_TRIALS, op * late, anchor="lm", strong=True)
        rows = (("one worker", self.rms[3], DIM), ("four, one seed", self.rms[0], RED),
                ("four, one stream jumped apart", self.rms[1], BLUE),
                ("four, spawned (numpy)", self.rms[2], GREEN))
        full = self.span(0.42)[0]
        for k, (name, v, ink) in enumerate(rows):
            go = (u - 0.6 - 0.07 * k) / 0.14
            self.hbar(sc, 0.40, 0.74 + 0.048 * k, full * v / 0.06, ink, op * smoothstep(0, 0.3, go),
                      name, "%.3f" % v, go, thick=17.0)
        self.tag(sc, STATS, op)


# ====================================================================== 112: sampling
TOKEN_INK = (RED, BLUE, GREEN, AMBER, VIOLET)
WRITTEN = "written-out probabilities; the draws are real"


class SamplingADistribution(Random):
    """Even numbers dropped onto probabilities laid end to end; points thrown into a
    box and kept under a curve; pairs of even numbers carried to a bell; the error of
    n draws falling as n grows; and the pairs a bad generator's low byte never makes."""

    TITLE = "Sampling a distribution"
    PHASES = (("pick a token", 11.0), ("rejection", 11.0), ("to a bell", 11.0),
              ("how fast", 11.0), ("a bad source", 11.0))
    BAR = (0.10, 0.90)

    def setup(self):
        d = self.d = SR.sampling_data()
        s = self.study
        self.p = np.asarray(s.panels[0].data["probability"], float)
        self.share = np.asarray(s.panels[0].data["values"], float)
        self.grid_n = np.asarray(s.panels[3].data["x"], float)
        self.err, self.pair_err, self.pair_bad = np.asarray(s.panels[3].data["ys"], float)
        self.band = np.asarray(s.panels[3].data["single_band"], float)
        self.pair_ctl = np.asarray(s.panels[3].data["pairs_pcg_low_byte"], float).mean(0)
        self.slope = float(s.metadata["error_slope"])
        self.kept = float(s.metadata["rejection"]["kept"])
        self.bm = np.asarray(s.panels[2].data["stats"], float)
        self.period = int(s.metadata["lcg_low_byte_token_period"])
        idx = SR.inverse_cdf(SR.unit(SR.pcg32_seeded(7, SR.SAMP_N)), self.p)
        self.prefix = np.array([np.bincount(idx[:int(n)], minlength=5) for n in self.grid_n])
        self.one_err = np.array([SR.total_variation(c, self.p) for c in self.prefix])
        expect = np.outer(self.p, self.p)
        self.ratio = [np.asarray(d[k], float) / np.asarray(d[k], float).sum() / expect - 1.0
                      for k in ("pair_good", "pair_lcg")]
        self.never = int((np.asarray(d["pair_lcg"]) == 0).sum())

    def subtitle_for(self, k):
        g = self.grid_n
        return [
            "to pick the next word, lay the probabilities end to end from 0 to 1 and drop an "
            "even random number on them: it lands on 'mat' 40 times in 100. Shares of %s "
            "draws: %.3f, %.3f, %.3f, %.3f, %.3f" % ((commas(SR.SAMP_N),) + tuple(self.share)),
            "for a curve with no such table: throw points evenly into a box and keep those "
            "under the curve. Their positions follow it; %.1f%% were kept, the share of the "
            "box under the curve" % (100 * self.kept),
            "for a bell curve, pairs of even numbers are moved: one sets the distance from "
            "the centre, the other the angle. Of %s values, %.1f%% lie within one "
            "spread of the centre" % (commas(2 * SR.SAMP_N), 100 * self.bm[2]),
            "the shares of n draws are never exact. Over %d repeats the gap falls from %.3f "
            "at %d draws to %.4f at %s: four times the draws, half the error" % (
                SR.SAMP_REPEATS, self.err[0], g[0], self.err[-1], commas(g[-1])),
            "a bad source: the low byte of the 32-bit generator. Its tokens repeat every "
            "%d draws; %d of the 25 pairs never occur, and the error of the pairs stays "
            "at %.3f while numpy's falls to %.3f" % (self.period, self.never, self.pair_bad[-1],
                                                    self.pair_err[-1]),
        ][k]

    # ------------------------------------------------------------------ drawing
    def at(self, v):
        """Layout x of the number ``v`` in [0, 1] on the probability bar."""
        return self.BAR[0] + (self.BAR[1] - self.BAR[0]) * float(v)

    def phase_1(self, sc, u, op):
        d = self.d
        edges = np.concatenate([[0.0], np.cumsum(self.p)])
        y = 0.475
        self.text(sc, 0.06, 0.315, "\"the cat sat on the ...\": five next words and their "
                  "probabilities", op, anchor="lm", strong=True)
        for k, name in enumerate(SR.TOKENS):
            mid = self.at((edges[k] + edges[k + 1]) / 2)
            w = self.span(self.at(edges[k + 1]) - self.at(edges[k]))[0] - 3
            self.box(sc, mid, y, w, 44, soft(TOKEN_INK[k], 0.45), op, "%.2f" % self.p[k],
                     border=TOKEN_INK[k], size=0.031)
            self.text(sc, mid, y - 0.075, name, op, ink=dark(TOKEN_INK[k], 0.3), size=0.032)
        self.text(sc, self.at(0) - 0.012, y, "0", op, anchor="rm")
        self.text(sc, self.at(1) + 0.012, y, "1", op, anchor="lm")
        draws = len(d["token_u"]) // 2
        t = min(u / 0.8, 1.0) * draws
        now = min(int(t), draws - 1)
        within = t - now if t < draws else 1.0
        for k in range(now + (1 if within > 0.6 or t >= draws else 0)):
            ink = TOKEN_INK[int(d["token_idx"][k])]
            self.disk(sc, self.P(self.at(d["token_u"][k]), y + 0.062), 3.6, ink, op, lift=1.0,
                      outline=0.0)
            self.box(sc, 0.135 + 0.067 * k, 0.70, 56, 30, soft(ink, 0.5), op,
                     SR.TOKENS[int(d["token_idx"][k])], border=ink, size=0.03)
        if t < draws:
            v = float(d["token_u"][now])
            ink = TOKEN_INK[int(d["token_idx"][now])]
            x = self.at(v)
            fall = smoothstep(0.1, 0.6, within)
            self.disk(sc, self.P(x, 0.36 + (y - 0.36) * fall), 6.5, ink, op, glow=0.4, lift=1.8,
                      outline=0.0)
        else:
            v = float(d["token_u"][draws - 1])
        self.text(sc, 0.06, 0.625, "draw %d: the number %.4f" % (now + 1, v), op, anchor="lm",
                  size=0.032)
        late = op * smoothstep(0.84, 0.94, u)
        self.text(sc, 0.06, 0.80, "shares of %s draws:" % commas(SR.SAMP_N), late, anchor="lm")
        for k, name in enumerate(SR.TOKENS):
            self.text(sc, 0.33 + 0.125 * k, 0.80, "%s %.3f" % (name, self.share[k]), late,
                      ink=dark(TOKEN_INK[k], 0.3))
        self.tag(sc, WRITTEN, op)

    def phase_2(self, sc, u, op):
        d = self.d
        box = ((0.09, 0.50), (0.38, 0.78))
        chart = self.chart(box, [0, 1], [0, SR.BOUND], pad=0.0)
        n = int(smoothstep(0.03, 0.7, u) * SR.SAMP_SHOWN)
        keep = d["rej_keep"][:n]
        col = np.where(keep[:, None], _rgb(GREEN), _rgb(soft(RED, 0.25)))
        self.picture(sc, raster(d["rej_x"][:n], d["rej_y"][:n], (0, 1), (0, SR.BOUND),
                                (118, 204), col), box, op)
        xs = np.linspace(0, 1, 60)
        self.path(sc, chart, xs, SR.target(xs), dark(GREEN, 0.45), op, 2.6)
        self.axes(sc, chart, op, "", "", xticks=[(v, "%g" % v) for v in (0, 0.5, 1)],
                  yticks=[], grid=False)
        self.text(sc, 0.295, 0.325, "points thrown evenly into the box", op, strong=True,
                  size=HEAD)
        self.text(sc, 0.295, 0.875, "%s thrown: %s kept (green), %s dropped" % (
            commas(n), commas(keep.sum()), commas(n - keep.sum())), op, size=0.031)
        late = op * smoothstep(0.45, 0.6, u)
        box2 = ((0.60, 0.94), (0.38, 0.66))
        right = self.chart(box2, [0, 1], [0, SR.BOUND], pad=0.0)
        self.vbars(sc, d["rej_hist"], box2, late, soft(GREEN, 0.25), vmax=SR.BOUND, gap=0.1)
        self.path(sc, right, xs, SR.target(xs), dark(GREEN, 0.45), late, 2.6)
        self.axes(sc, right, late, "", "", xticks=[(v, "%g" % v) for v in (0, 0.5, 1)],
                  yticks=[], grid=False)
        self.text(sc, 0.77, 0.325, "where the kept ones are", late, strong=True, size=HEAD)
        self.lines(sc, 0.60, 0.775, [
            "of %s thrown, %.2f%% kept;" % (commas(SR.SAMP_N), 100 * self.kept),
            "the curve covers 9/16 = 56.25% of the box"], op * smoothstep(0.7, 0.82, u),
            step=0.045)
        self.tag(sc, STATS, op)

    def phase_3(self, sc, u, op):
        d = self.d
        box = ((0.06, 0.64), (0.37, 0.82))
        w, h = self.span(0.58, 0.45)
        wide = w / h
        uu, zz = np.asarray(d["bm_u"], float), np.asarray(d["bm_z"], float)
        n = len(uu)
        start = 0.08 + 0.5 * np.arange(n) / n
        s = np.clip((u - start) / 0.22, 0.0, 1.0)
        s = (s * s * (3 - 2 * s))[:, None]
        a = np.stack([0.04 + 0.9 * uu[:, 0], 0.05 + 0.9 * uu[:, 1]], 1)
        centre = wide - 0.52
        b = np.stack([centre + 0.115 * zz[:, 0], 0.5 + 0.115 * zz[:, 1]], 1)
        pos = a * (1 - s) + b * s
        col = np.where((uu[:, 0] < 0.5)[:, None], _rgb(GREEN), _rgb(VIOLET))
        self.picture(sc, raster(pos[:, 0], pos[:, 1], (0, wide), (0, 1), (118, int(118 * wide)),
                                col), box, op)
        (bx0, bx1), (by0, by1) = box
        self.ring(sc, bx0 + (bx1 - bx0) * 0.49 / wide, (by0 + by1) / 2, 0.92 * h, 0.92 * h,
                  AXIS, op, width=1.2)                          # the unit square they start in
        self.text(sc, 0.19, 0.325, "%s pairs of even numbers" % commas(n), op, strong=True,
                  size=HEAD)
        self.text(sc, 0.49, 0.325, "the same pairs, moved", op * smoothstep(0.3, 0.45, u),
                  strong=True, size=HEAD)
        self.lines(sc, 0.06, 0.868, [
            "the first number of a pair sets the distance from the centre,",
            "the second the angle; green: the first number is below a half"], op, step=0.04)
        late = op * smoothstep(0.62, 0.75, u)
        box2 = ((0.70, 0.95), (0.40, 0.64))
        mids = (SR.BELL_EDGES[:-1] + SR.BELL_EDGES[1:]) / 2
        chart = self.chart(box2, [-4, 4], [0, 0.42], pad=0.0)
        self.vbars(sc, d["bm_hist"], box2, late, soft(VIOLET, 0.3), vmax=0.42, gap=0.1)
        self.path(sc, chart, mids, np.exp(-mids ** 2 / 2) / np.sqrt(2 * np.pi), dark(VIOLET, 0.4),
                  late, 2.4)
        self.axes(sc, chart, late, "", "", xticks=[(v, "%d" % v) for v in (-4, -2, 0, 2, 4)],
                  yticks=[], grid=False)
        self.text(sc, 0.825, 0.345, "%s values" % commas(2 * SR.SAMP_N), late, strong=True)
        self.lines(sc, 0.70, 0.75, [
            "mean %.3f, spread %.3f" % (self.bm[0], self.bm[1]),
            "%.1f%% within one spread" % (100 * self.bm[2]),
            "(a bell curve: 68.3%)"], late, step=0.045)
        self.tag(sc, STATS, op)

    def phase_4(self, sc, u, op):
        g = self.grid_n
        k = min(int(smoothstep(0.03, 0.85, u) * len(g)), len(g) - 1)
        n = int(g[k])
        share = self.prefix[k] / float(n)
        box = ((0.08, 0.43), (0.42, 0.78))
        (x0, x1), (y0, y1) = box
        step = (x1 - x0) / 5
        self.text(sc, 0.255, 0.325, "%s draws" % commas(n), op, strong=True, size=BIG)
        for t, name in enumerate(SR.TOKENS):
            cx = x0 + (t + 0.5) * step
            hgt = (y1 - y0) * share[t] / 0.6
            self.rect(sc, self.P(cx, y1 - hgt / 2), self.span(step * 0.62)[0],
                      max(self.span(0, hgt)[1], 0.8), TOKEN_INK[t], op, radius=2, lift=0.4)
            ty = y1 - (y1 - y0) * self.p[t] / 0.6
            self.polyline(sc, [self.P(cx - step * 0.42, ty), self.P(cx + step * 0.42, ty)], 2.6,
                          (0.14, 0.16, 0.18), op, bias=0.6)
            self.text(sc, cx, 0.825, name, op, ink=dark(TOKEN_INK[t], 0.3))
        self.text(sc, 0.255, 0.38, "bars: shares so far;  lines: the probabilities", op)
        self.text(sc, 0.255, 0.885, "error: half the summed gaps = %.4f" % self.one_err[k], op,
                  strong=True)
        chart = self.chart(((0.57, 0.94), (0.40, 0.72)), g, [7e-4, 0.5], xlog=True, ylog=True)
        self.axes(sc, chart, op, "draws", "error, mean of %d repeats" % SR.SAMP_REPEATS,
                  xticks=[(v, pow10(v)) for v in (10, 100, 1000, 1e4, 1e5)],
                  yticks=[(v, "%g" % v) for v in (0.001, 0.01, 0.1)])
        go = (k + 1) / float(len(g))
        ref = self.err[0] * np.sqrt(g[0] / g)
        progress(self, sc, chart, g, ref, AXIS, op, 1.0, 1.2, xlog=True, ylog=True, dot=False)
        for row in self.band:
            progress(self, sc, chart, g, row, soft(BLUE, 0.5), op, go, 1.4, xlog=True, ylog=True,
                     dot=False)
        progress(self, sc, chart, g, self.err, BLUE, op, go, 3.0, xlog=True, ylog=True)
        self.disk(sc, self.P(chart.fx(n), chart.fy(max(self.one_err[k], 1e-3))), 4.5, AMBER, op,
                  lift=1.6, outline=0.0)
        self.lines(sc, 0.57, 0.84, ["grey: one over the square root of the draws",
                                    ("orange: this one run", dark(AMBER, 0.3))], op, step=0.042)
        self.tag(sc, STATS, op)

    def phase_5(self, sc, u, op):
        g = self.grid_n
        self.text(sc, 0.06, 0.315, "which word follows which: more often (red) or less (blue) "
                  "than it should", op, anchor="lm", strong=True)
        for t, (cx, name, ink) in enumerate(((0.235, "numpy's generator", GREEN),
                                             (0.47, "the bad low byte", AMBER))):
            show = op * smoothstep(0.05 + 0.25 * t, 0.2 + 0.25 * t, u)
            cell = self.grid(sc, self.ratio[t], cx, 0.60, 180, 180, show, vmax=1.0,
                             reveal=smoothstep(0.05 + 0.25 * t, 0.35 + 0.25 * t, u))
            self.text(sc, cx, 0.385, name, show, ink=dark(ink, 0.3), size=0.032)
            if t == 0:
                for r, tok in enumerate(SR.TOKENS):
                    self.text(sc, 0.118, cell(r, 0)[1], tok, op, anchor="rm",
                              ink=dark(TOKEN_INK[r], 0.3))
        self.text(sc, 0.06, 0.84, "rows: a word; columns: the next, in the same order", op,
                  anchor="lm")
        self.text(sc, 0.06, 0.885, "deepest blue: never. %d of 25 pairs never occur" % self.never,
                  op * smoothstep(0.55, 0.65, u), anchor="lm", ink=dark(BLUE))
        chart = self.chart(((0.66, 0.94), (0.40, 0.66)), g, [3e-3, 0.7], xlog=True, ylog=True)
        self.axes(sc, chart, op, "draws", "error of the pairs",
                  xticks=[(v, pow10(v)) for v in (10, 1000, 1e5)],
                  yticks=[(v, "%g" % v) for v in (0.01, 0.1)])
        go = (u - 0.45) / 0.4
        progress(self, sc, chart, g, self.pair_ctl, soft(BLUE, 0.3), op, go, 1.6, xlog=True,
                 ylog=True, dot=False)
        progress(self, sc, chart, g, self.pair_err, GREEN, op, go, 2.8, xlog=True, ylog=True)
        progress(self, sc, chart, g, self.pair_bad, AMBER, op, go, 2.8, xlog=True, ylog=True)
        self.lines(sc, 0.66, 0.785, [
            ("numpy", dark(GREEN, 0.3)), ("the LCG's low byte: stuck", dark(AMBER, 0.3)),
            ("PCG32's low byte", dark(BLUE, 0.2))], op * smoothstep(0.5, 0.6, u), step=0.04)
        self.tag(sc, STATS, op)


# ====================================================================== 113: sources
SRC_INK = (MAGENTA, TEAL, AMBER, VIOLET)      # timer jitter, operating system, counter, clock


class EntropySources(Random):
    """A seed stretched into a long stream that holds no more than the seed; this
    machine's timer, system bytes, a counter and the clock, estimated; a biased coin
    cleaned pair by pair; coins mixed by XOR and a pool hashed; the whole chain."""

    TITLE = "Where randomness comes from"
    PHASES = (("stretching", 9.0), ("sources", 12.0), ("cleaning", 11.0), ("mixing", 11.0),
              ("seeding", 10.0))

    def setup(self):
        self.d = SR.sources_data()
        s = self.study
        self.loc = SR.local_data()
        self.srcs = SR.local_sources()
        self.measures = np.asarray(s.panels[0].data["values"], float)      # (source, 3)
        self.vn = np.asarray(s.panels[1].data["values"], float)            # before, after, kept
        self.coin = np.asarray(s.panels[1].data["coin_entropy"], float).mean(0)
        self.vn_local = np.asarray(s.panels[1].data["local_timer_bit"], float)
        self.xor = np.asarray(s.panels[2].data["ys"], float)               # measured, exact
        self.hash_changed = np.asarray(s.panels[2].data["hash_bits_changed"], float)
        self.distinct = [int(v) for v in s.panels[3].data["values"]]
        self.bits = [int(v) for v in s.panels[3].data["seed_bits"]]
        self.stretched = np.asarray(s.panels[3].data["stretched_estimates"], float)
        self.machine = s.metadata["machine"]
        self.pool, self.seed = SR.pool_seed(self.srcs)
        self.n = SR.N_BYTES

    def subtitle_for(self, k):
        m = self.measures
        return [
            "a generator only stretches its seed. From a 16-bit seed come %s bytes that "
            "look perfect, %.2f bits per byte by the usual estimate, yet only 65,536 "
            "different streams exist: 16 bits in all" % (commas(self.n), self.stretched[0]),
            "so the seed must come from outside. On this machine: the low byte of the gaps "
            "between timer readings scores %.1f bits per byte, the system's bytes %.2f, a "
            "counter %.0f until its order is looked at" % (m[0, 0], m[1, 0], m[2, 0]),
            "a raw source is lopsided. Von Neumann's trick reads bits in pairs: 01 gives 0, "
            "10 gives 1, the rest is dropped. Ones go from %.0f%% to %.1f%%, and %.2f bits "
            "are kept per bit collected" % (100 * self.vn[0], 100 * self.vn[1], self.vn[2]),
            "sources are mixed so that one weak one does no harm: XOR of %d lopsided coins "
            "shows 1 %.1f%% of the time; a hash of the pool changes %.0f of its 256 bits "
            "when one input bit does" % (len(self.xor[0]), 100 * self.xor[0, -1],
                                         self.hash_changed[0]),
            "the chain: gather, clean, mix, seed, stretch. Trying every seed of %d bits "
            "gives exactly %s different streams: more output never holds more than "
            "went in" % (self.bits[-1], commas(self.distinct[-1])),
        ][k]

    # ------------------------------------------------------------------ drawing
    def phase_1(self, sc, u, op):
        seed = 12345
        self.text(sc, 0.06, 0.315, "a seed of 16 bits", op, anchor="lm", strong=True, size=HEAD)
        self.cells(sc, [bit_list(seed, 16)], 0.17, 0.42, 208, 26, op, ink=AMBER)
        self.text(sc, 0.17, 0.495, "= %s, one of 65,536" % commas(seed), op)
        self.arrow(sc, (0.17, 0.535), (0.17, 0.60), AMBER, op, smoothstep(0.05, 0.2, u), 2.4)
        self.box(sc, 0.17, 0.66, 150, 46, BLUE, op, "generator", ink=WHITE, size=BIG)
        self.arrow(sc, (0.26, 0.66), (0.375, 0.66), BLUE, op, smoothstep(0.15, 0.3, u), 2.4)
        self.pulse(sc, (0.26, 0.66), (0.375, 0.66), (u - 0.15) / 0.25, BLUE, op)
        rows = int(np.ceil(smoothstep(0.25, 0.8, u) * 64))
        tex = grey(self.d["stretched_bytes"], 64)
        tex[rows:] = _rgb(PAPER) * 0.97
        dw, dh = self.frac(220, 220)
        self.picture(sc, tex, ((0.39, 0.39 + dw), (0.37, 0.37 + dh)), op)
        self.text(sc, 0.515, 0.325, "its bytes: dark = high", op)
        out = int(self.n * smoothstep(0.25, 0.8, u))
        self.lines(sc, 0.69, 0.40, [
            ("bytes out so far", None), (commas(out), STRONG)], op, size=0.032, step=0.05)
        late = op * smoothstep(0.7, 0.82, u)
        self.lines(sc, 0.69, 0.545, [
            ("estimated from them:", None),
            ("%.2f bits per byte" % self.stretched[0], dark(BLUE, 0.2)),
            ("(Shannon, single bytes)", None)], late, size=0.032, step=0.05)
        last = op * smoothstep(0.82, 0.94, u)
        self.lines(sc, 0.69, 0.74, [
            ("unpredictability in it:", None), ("16 bits, in all", dark(RED))], last, size=0.032,
            step=0.05)
        self.tag(sc, STATS, op)

    def phase_2(self, sc, u, op):
        gaps = [int(v) for v in self.loc["deltas"][:8]]
        self.text(sc, 0.06, 0.315, "gaps between timer readings, in nanoseconds:", op,
                  anchor="lm", strong=True)
        for k, gap in enumerate(gaps):
            show = op * smoothstep(0.02 + 0.03 * k, 0.06 + 0.03 * k, u)
            self.box(sc, 0.515 + 0.058 * k, 0.315, 48, 28, soft(MAGENTA, 0.6), show, str(gap),
                     border=MAGENTA, size=0.03)
        heads = (("Shannon entropy", "of single bytes"), ("min-entropy", "of single bytes"),
                 ("Shannon entropy", "given the byte before"))
        starts = (0.425, 0.60, 0.775)
        for x, (a, b) in zip(starts, heads):
            self.text(sc, x, 0.39, a, op, anchor="lm", strong=True)
            self.text(sc, x, 0.43, b, op, anchor="lm")
        full = self.span(0.10)[0]
        for r, name in enumerate(SR.SRC_NAMES):
            y = 0.515 + 0.088 * r
            show = op * smoothstep(0.2 + 0.12 * r, 0.3 + 0.12 * r, u)
            ink = SRC_INK[r]
            self.text(sc, 0.06, y, name, show, anchor="lm", ink=dark(ink, 0.3), size=0.031)
            self.picture(sc, grey(self.srcs[r][:240], 60), ((0.275, 0.40), (y - 0.03, y + 0.03)),
                         show)
            for c, x in enumerate(starts):
                v = self.measures[r, c]
                self.hbar(sc, x, y, full * v / 8.0, ink, show, "", "%.2f" % v,
                          (u - 0.2 - 0.12 * r - 0.03 * c) / 0.12, thick=16.0, track=full)
        self.text(sc, 0.06, 0.875, "bits per byte, from %s bytes of each (%s)" % (
            commas(self.n), self.machine), op, anchor="lm")
        self.tag(sc, LOCAL, op)

    def phase_3(self, sc, u, op):
        bits_in = [int(v) for v in self.d["vn_in"]]
        self.text(sc, 0.06, 0.315, "a coin that shows 1 four times in five, read two at a time",
                  op, anchor="lm", strong=True)
        t = min(u / 0.5, 1.0) * 16
        kept = 0
        for k in range(16):
            a, b = bits_in[2 * k], bits_in[2 * k + 1]
            x = 0.105 + 0.0535 * k
            self.cells(sc, [[a, b]], x, 0.395, 40, 22, op, ink=AMBER)
            done = smoothstep(k, k + 0.7, t)
            if a != b:
                kept += done > 0.5
                self.arrow(sc, (x, 0.43), (x, 0.475), GREEN, op, done, 2.0)
                self.box(sc, x, 0.51, 24, 24, GREEN if a else PAPER, op * done, str(a),
                         ink=WHITE if a else None, border=GREEN)
            else:
                self.text(sc, x, 0.47, "×", op * done, ink=dark(RED), size=0.034)
        self.text(sc, 0.06, 0.585, "%d bits read, %d kept" % (2 * min(int(t + 0.3), 16), kept), op,
                  anchor="lm", size=0.032)
        late = smoothstep(0.5, 0.6, u)
        full = self.span(0.20)[0]
        rows = (("ones before", self.vn[0], AMBER, "%.1f%%" % (100 * self.vn[0])),
                ("ones after", self.vn[1], GREEN, "%.1f%%" % (100 * self.vn[1])),
                ("kept per bit", self.vn[2], TEAL, "%.3f" % self.vn[2]))
        for k, (name, v, ink, say) in enumerate(rows):
            go = (u - 0.52 - 0.07 * k) / 0.15
            self.hbar(sc, 0.19, 0.665 + 0.06 * k, full * v, ink, op * late, name, say, go,
                      thick=18.0, track=full)
        self.text(sc, 0.06, 0.855, "Shannon entropy of such a coin: %.2f bits a toss before, "
                  "%.2f after" % tuple(self.coin), op * smoothstep(0.72, 0.82, u), anchor="lm")
        v = self.vn_local
        self.lines(sc, 0.565, 0.60, [
            ("this machine's timer bit:", dark(MAGENTA, 0.3)),
            ("ones %.1f%% before, %.1f%% after" % (100 * v[0], 100 * v[1]), None),
            ("%.3f kept per bit" % v[2], None),
            ("but each bit leans on the last:", None),
            ("correlation %.2f before, %.2f after" % (v[3], v[4]), None)],
            op * smoothstep(0.8, 0.92, u), step=0.044)
        self.tag(sc, STATS + "; right: local", op)

    def phase_4(self, sc, u, op):
        ks = np.arange(1, len(self.xor[0]) + 1)
        self.text(sc, 0.265, 0.325, "XOR of lopsided coins", op, strong=True, size=HEAD)
        chart = self.chart(((0.10, 0.44), (0.41, 0.72)), ks, [0.2, 0.9], pad=0.0)
        self.axes(sc, chart, op, "coins XORed together", "share of ones",
                  xticks=[(v, "%d" % v) for v in ks], yticks=[(v, "%g" % v) for v in
                                                              (0.2, 0.5, 0.8)])
        chart.hline(sc, self.P, 0.5, op)
        go = smoothstep(0.05, 0.45, u)
        progress(self, sc, chart, ks, self.xor[1], AXIS, op, go, 1.4, dot=False)
        for k in range(int(np.ceil(go * len(ks)))):
            self.disk(sc, self.P(chart.fx(ks[k]), chart.fy(self.xor[0, k])), 5.5, AMBER, op,
                      lift=1.4, outline=0.0)
        shown = max(int(np.ceil(go * len(ks))), 1)
        self.text(sc, 0.265, 0.855, "%d coins: 1 shows %.1f%% of the time (line: exact)" % (
            shown, 100 * self.xor[0, shown - 1]), op)
        self.text(sc, 0.75, 0.325, "a pool of this machine's sources, hashed", op, strong=True,
                  size=HEAD)
        names = ("timer", "system", "counter", "clock")
        for r, name in enumerate(names):
            y = 0.385 + 0.042 * r
            show = op * smoothstep(0.4 + 0.04 * r, 0.46 + 0.04 * r, u)
            self.picture(sc, grey(self.srcs[r][:32], 32), ((0.64, 0.90), (y - 0.014, y + 0.014)),
                         show)
            self.text(sc, 0.625, y, name, show, anchor="rm", ink=dark(SRC_INK[r], 0.3))
        down = smoothstep(0.58, 0.68, u)
        self.arrow(sc, (0.77, 0.537), (0.77, 0.572), TEAL, op, down, 2.4)
        self.box(sc, 0.77, 0.612, 120, 34, TEAL, op * smoothstep(0.5, 0.6, u), "SHA-256",
                 ink=WHITE, size=0.032)
        self.arrow(sc, (0.77, 0.652), (0.77, 0.687), TEAL, op, smoothstep(0.66, 0.76, u), 2.4)
        seed_bits = np.unpackbits(np.frombuffer(self.seed, np.uint8)).reshape(4, 64)
        self.cells(sc, seed_bits, 0.77, 0.742, 288, 44, op * smoothstep(0.72, 0.82, u), ink=TEAL,
                   grid=0.2)
        self.lines(sc, 0.77, 0.83, ["a seed of 256 bits", "one input bit changed: %.0f of them "
                                    "change" % self.hash_changed[0]],
                   op * smoothstep(0.82, 0.92, u), step=0.042, anchor="mm")
        self.tag(sc, STATS + "; right: local", op)

    def phase_5(self, sc, u, op):
        boxes = ((0.12, 118, "sources", MAGENTA), (0.315, 150, "clean and mix", TEAL),
                 (0.50, 96, "seed", AMBER), (0.675, 124, "generator", BLUE),
                 (0.86, 104, "stream", VIOLET))
        notes = ("a little, slowly", "biased in, even out", "256 bits", "stretches it",
                 "as long as you like")
        for k, ((x, w, name, ink), note) in enumerate(zip(boxes, notes)):
            show = op * smoothstep(0.03 + 0.08 * k, 0.1 + 0.08 * k, u)
            self.box(sc, x, 0.405, w, 44, ink, show, name, ink=WHITE, size=0.034)
            self.text(sc, x, 0.49, note, show)
            if k:
                px, pw = boxes[k - 1][0], boxes[k - 1][1]
                a = (px + self.frac(pw / 2 + 4)[0], 0.405)
                b = (x - self.frac(w / 2 + 6)[0], 0.405)
                self.arrow(sc, a, b, AXIS, op, smoothstep(0.05 + 0.08 * k, 0.12 + 0.08 * k, u), 2.4)
                self.pulse(sc, a, b, (u - 0.1 - 0.08 * k) / 0.2, ink, op)
        late = smoothstep(0.45, 0.55, u)
        self.text(sc, 0.06, 0.575, "every seed of k bits tried: how many different streams "
                  "come out", op * late, anchor="lm", strong=True)
        full = self.span(0.40)[0]
        for k, (bits_, count) in enumerate(zip(self.bits, self.distinct)):
            go = (u - 0.5 - 0.06 * k) / 0.14
            self.hbar(sc, 0.22, 0.64 + 0.05 * k, full * np.log2(count) / 20.0, BLUE,
                      op * smoothstep(0, 0.3, go), "%d-bit seeds" % bits_, "%s streams (%s)" % (
                          commas(count), two_to(bits_)), go, thick=17.0)
        self.tag(sc, EXACT, op)


# ====================================================================== 114: measuring
ENT_INK = (GREEN, TEAL, BLUE, AMBER, MAGENTA, VIOLET)
ENT_TWO = (("English", "text"), ("the text,", "compressed"), ("a 32-bit", "generator"),
           ("a bad one's", "low byte"), ("the system's", "bytes"), ("a counter", ""))
ENT_ONE = ("English text", "compressed", "generator", "bad low byte", "the system's", "counter")
ENT_KIND = ("real text", "real text", "seed: 32 bits", "an LCG's bits 0-7", "local", "0, 1, 2, ...")
MIXED = "real text, generators' outputs, and this machine's bytes (local)"


def pair_texture(counts, ink):
    """A 256 x 256 table of pair counts as a 64 x 64 texture: a cell is inked by the
    logarithm of the largest count in its 4 x 4 block."""
    c = np.asarray(counts, np.float64).reshape(64, 4, 64, 4).max((1, 3))
    v = (np.log1p(c) / max(np.log1p(c.max()), 1e-9))[..., None].astype(np.float32)
    return _rgb(PAPER) * (1 - v) + _rgb(dark(ink, 0.25)) * v


class MeasuringEntropy(Random):
    """Six byte sources; the single-byte estimate and what it misses; its bias in small
    samples; pairs of bytes, which expose the counter and the bad low byte; three tests;
    and the generator that passes everything, its seed found by trying them all."""

    TITLE = "Measuring entropy"
    PHASES = (("six sources", 8.0), ("count bytes", 10.0), ("small samples", 9.0),
              ("pairs", 10.0), ("tests", 9.0), ("the seed", 11.0))

    def setup(self):
        d = self.d = SR.entropy_data()
        s = self.study
        self.m = np.asarray(s.panels[0].data["values"], float)            # (source, measure)
        self.ideal = np.asarray(s.panels[0].data["ideal"], float)
        self.grid_n = np.asarray(s.panels[1].data["x"], float)
        self.bias = np.asarray(s.panels[1].data["ys"], float)             # uniform, text shares
        self.true = np.asarray(s.panels[1].data["true"], float)
        self.tests = np.asarray(s.panels[2].data["values"], float)
        self.left = [int(v) for v in s.panels[3].data["values"]]
        self.outputs = [int(v) for v in s.panels[3].data["outputs"]]
        self.found = int(s.panels[3].data["recovered"][0])
        self.n = SR.N_BYTES
        hist = np.asarray(d["hist"][0], float)
        self.space = float(hist.max() / hist.sum())
        sample = np.random.default_rng(0).integers(0, 256, int(self.grid_n[-1]), dtype=np.uint8)
        self.sample = sample
        self.est = [SR.byte_entropy(sample[:int(n)]) for n in self.grid_n]
        self.textures = [pair_texture(d["pairs"][k], ENT_INK[k]) for k in range(6)]

    def subtitle_for(self, k):
        m, t = self.m, self.tests
        return [
            "how unpredictable is a source? Six are compared, %s bytes of each: English "
            "text, the same text compressed, a generator, a bad generator's low byte, the "
            "system's random bytes, a counter" % commas(self.n),
            "first estimate: the Shannon entropy of single bytes, from how often each "
            "occurs. Text scores %.2f bits per byte, the others %.2f or more: the counter "
            "and the bad low byte score a full %.0f" % (m[0, 0], m[1:, 0].min() - 0.005,
                                                       m[5, 0]),
            "the estimate comes from a sample, and a small one cannot show every value: "
            "from %d even bytes it says %.2f where the truth is 8, and %.2f from %s" % (
                self.grid_n[2], self.bias[0, 2], self.bias[0, 10], commas(self.grid_n[10])),
            "given the byte before, the text falls to %.2f bits and the counter and the bad "
            "low byte to %.0f: their next byte is certain. A compressor's output size is a "
            "ceiling: %.2f for the text" % (m[0, 2], m[5, 2], m[0, 4]),
            "three tests ask whether equal, unconnected bytes would look like this: only "
            "the generator and the system's bytes pass all three. The generator scores "
            "%.1f, %.1f and %.1f" % tuple(t[2]),
            "yet the generator has a 32-bit seed. Trying all %s seeds against its first "
            "outputs leaves %s after one and %d after two: the whole stream holds 32 "
            "bits, whatever its bytes score" % (commas(self.left[0]), commas(self.left[1]),
                                                self.left[2]),
        ][k]

    # ------------------------------------------------------------------ drawing
    def phase_1(self, sc, u, op):
        self.text(sc, 0.50, 0.325, "the first 4,096 bytes of each, one per cell: dark = high",
                  op, strong=True)
        dw, dh = self.frac(104, 104)
        for k in range(6):
            x = 0.125 + 0.15 * k
            rows = int(np.ceil(smoothstep(0.03 + 0.08 * k, 0.4 + 0.08 * k, u) * 64))
            tex = grey(self.d["strips"][k], 64)
            tex[rows:] = _rgb(PAPER) * 0.97
            self.picture(sc, tex, ((x - dw / 2, x + dw / 2), (0.51 - dh / 2, 0.51 + dh / 2)), op,
                         frame=ENT_INK[k])
            for j, line in enumerate(ENT_TWO[k]):
                self.text(sc, x, 0.675 + 0.045 * j, line, op, ink=dark(ENT_INK[k], 0.3),
                          size=0.031)
            self.text(sc, x, 0.79, ENT_KIND[k], op * smoothstep(0.5, 0.65, u))
        self.tag(sc, MIXED, op)

    def phase_2(self, sc, u, op):
        hist = np.asarray(self.d["hist"][0], float)
        box = ((0.07, 0.47), (0.385, 0.62))
        rows = 64
        level = (hist / hist.max() * rows * smoothstep(0.02, 0.3, u))[None, :]
        fill = (rows - np.arange(rows))[:, None] <= level
        tex = np.where(fill[..., None], _rgb(GREEN), _rgb(PAPER))
        self.picture(sc, tex, box, op)
        self.text(sc, 0.27, 0.325, "how often each of the 256 byte values occurs in the text",
                  op, strong=True)
        self.text(sc, 0.07, 0.665, "0", op, anchor="lm")
        self.text(sc, 0.47, 0.665, "255", op, anchor="rm")
        self.lines(sc, 0.07, 0.725, [
            ("Shannon entropy of single bytes: %.2f bits per byte" % self.m[0, 0], STRONG),
            ("min-entropy: %.2f. It looks only at the commonest," % self.m[0, 1], None),
            ("the space: %.1f%% of the bytes" % (100 * self.space), None)],
            op * smoothstep(0.3, 0.42, u), step=0.046)
        full = self.span(0.14)[0]
        self.text(sc, 0.575, 0.325, "Shannon (dark), min-entropy (pale)", op, anchor="lm",
                  strong=True)
        for k, name in enumerate(ENT_ONE):
            y = 0.405 + 0.078 * k
            go = (u - 0.35 - 0.07 * k) / 0.15
            show = op * smoothstep(0, 0.3, go)
            ink = ENT_INK[k]
            self.hbar(sc, 0.70, y - 0.013, full * self.m[k, 0] / 8.0, ink, show, "", "", go,
                      thick=12.0, track=full)
            self.hbar(sc, 0.70, y + 0.019, full * self.m[k, 1] / 8.0, soft(ink, 0.45), show, "",
                      "", go, thick=8.0)
            self.text(sc, 0.69, y, name, show, anchor="rm", ink=dark(ink, 0.3))
            self.text(sc, 0.855, y, "%.2f, %.2f" % (self.m[k, 0], self.m[k, 1]), show,
                      anchor="lm", strong=True)
        self.tag(sc, MIXED, op)

    def phase_3(self, sc, u, op):
        g = self.grid_n
        k = min(int(smoothstep(0.03, 0.8, u) * len(g)), len(g) - 1)
        n = int(g[k])
        seen = np.bincount(self.sample[:n], minlength=256) > 0
        self.text(sc, 0.20, 0.325, "%s even random bytes" % commas(n), op, strong=True, size=BIG)
        self.cells(sc, seen.reshape(16, 16), 0.20, 0.61, 184, 184, op, ink=BLUE)
        self.text(sc, 0.20, 0.38, "values seen so far: %d of 256" % seen.sum(), op)
        self.text(sc, 0.20, 0.85, "estimate: %.2f bits per byte" % self.est[k], op, strong=True,
                  size=0.034)
        self.text(sc, 0.20, 0.895, "the truth: 8", op)
        chart = self.chart(((0.50, 0.94), (0.40, 0.74)), g, [3.0, 8.4], pad=0.0, xlog=True)
        self.axes(sc, chart, op, "bytes in the sample", "estimated Shannon entropy, mean of %d "
                  "samples" % SR.BIAS_REPEATS,
                  xticks=[(v, commas(v)) for v in (16, 256, 4096, 65536)],
                  yticks=[(v, "%d" % v) for v in (4, 6, 8)])
        for v in self.true:
            chart.hline(sc, self.P, float(v), op)
        go = (k + 1) / float(len(g))
        progress(self, sc, chart, g, self.bias[1], GREEN, op, go, 2.6, xlog=True)
        progress(self, sc, chart, g, self.bias[0], BLUE, op, go, 2.6, xlog=True)
        self.text(sc, 0.93, chart.fy(8.0) + 0.04, "even bytes: truly 8", op, anchor="rm",
                  ink=dark(BLUE, 0.2))
        self.text(sc, 0.93, chart.fy(self.true[1]) - 0.04, "the text's shares: truly %.2f"
                  % self.true[1], op, anchor="rm", ink=dark(GREEN, 0.3))
        self.tag(sc, STATS, op)

    def phase_4(self, sc, u, op):
        m = self.m
        xs = [0.335 + 0.118 * k for k in range(6)]
        dw, dh = self.frac(96, 96)
        self.lines(sc, 0.06, 0.455, ["each dot: a byte", "(across) and the", "one after (down)"],
                   op, step=0.044)
        for k, x in enumerate(xs):
            self.text(sc, x, 0.325, ENT_ONE[k], op, ink=dark(ENT_INK[k], 0.3))
            self.picture(sc, self.textures[k], ((x - dw / 2, x + dw / 2),
                                                (0.47 - dh / 2, 0.47 + dh / 2)), op,
                         frame=ENT_INK[k])
        rows = (("Shannon, single bytes", 0), ("... given the byte before", 2),
                ("... given the two before", 3), ("size after zlib: a ceiling", 4))
        for r, (name, col) in enumerate(rows):
            y = 0.625 + 0.052 * r
            show = op * smoothstep(0.1 + 0.15 * r, 0.2 + 0.15 * r, u)
            self.text(sc, 0.06, y, name, show, anchor="lm", strong=r == 1)
            for k, x in enumerate(xs):
                fell = m[k, col] < 1.0
                self.text(sc, x, y, "%.2f" % m[k, col], show, size=0.032,
                          ink=dark(RED) if fell else None, strong=not fell and r == 1)
        self.text(sc, 0.06, 0.845, "an ideal source scores %.2f, %.2f and %.2f here: %s bytes "
                  "are far too few to count 16.7 million triples" % (
                      self.ideal[0], self.ideal[2], self.ideal[3], commas(self.n)),
                  op * smoothstep(0.7, 0.82, u), anchor="lm")
        self.tag(sc, MIXED, op)

    def phase_5(self, sc, u, op):
        heads = ("chi-square of the counts", "serial correlation", "runs")
        cols = (0.40, 0.62, 0.82)
        for x, head in zip(cols, heads):
            self.text(sc, x, 0.325, head, op, strong=True)
        for k, name in enumerate(ENT_ONE):
            y = 0.40 + 0.071 * k
            show = op * smoothstep(0.04 + 0.09 * k, 0.12 + 0.09 * k, u)
            self.box(sc, 0.068, y, 14, 14, ENT_INK[k], show, radius=3)
            self.text(sc, 0.086, y, name, show, anchor="lm", ink=dark(ENT_INK[k], 0.3),
                      size=0.032)
            for x, z in zip(cols, self.tests[k]):
                ok = abs(z) < 3.0
                say = "%+.1f" % z if abs(z) < 1000 else ("+" if z > 0 else "−") + commas(abs(z))
                self.box(sc, x, y, 136, 28, soft(GREEN if ok else RED, 0.55), show, say,
                         border=GREEN if ok else RED, size=0.031)
        self.text(sc, 0.06, 0.845, "each number: how many spreads from what chance gives. "
                  "Beyond 3 either way fails (red); too even fails too", op * smoothstep(
                      0.6, 0.72, u), anchor="lm")
        self.tag(sc, MIXED, op)

    def phase_6(self, sc, u, op):
        self.text(sc, 0.06, 0.315, "the generator's first outputs, 16 bits each:", op,
                  anchor="lm", strong=True)
        for k, v in enumerate(self.outputs):
            self.box(sc, 0.46 + 0.085 * k, 0.315, 70, 30, soft(BLUE, 0.55), op * smoothstep(
                0.02 + 0.1 * k, 0.1 + 0.1 * k, u), commas(v), border=BLUE, size=0.031)
        names = ("seeds possible", "after 1 output", "after 2", "after 3")
        full = self.span(0.26)[0]
        for k, (name, count) in enumerate(zip(names, self.left)):
            go = (u - 0.1 - 0.13 * k) / 0.16
            self.hbar(sc, 0.235, 0.415 + 0.07 * k, max(full * np.log2(count) / 32.0, 3.0), BLUE,
                      op * smoothstep(0, 0.3, go), name, commas(count), go, thick=20.0,
                      track=full)
        late = op * smoothstep(0.6, 0.72, u)
        self.text(sc, 0.06, 0.715, "the seed: %s, found by trying every one of %s" % (
            commas(self.found), two_to(32)), late, anchor="lm", size=0.032, strong=True)
        self.text(sc, 0.06, 0.77, "with it, every byte the generator will ever give is known",
                  late, anchor="lm")
        self.lines(sc, 0.69, 0.415, [
            ("its bytes score", None),
            ("%.2f bits per byte" % self.m[2, 0], dark(BLUE, 0.2)),
            ("and pass every test", None)], op * smoothstep(0.35, 0.47, u), size=0.032,
            step=0.05)
        self.lines(sc, 0.69, 0.60, [
            ("the stream holds", None), ("32 bits", dark(RED)),
            ("however long it is", None)], op * smoothstep(0.75, 0.87, u), size=0.032, step=0.05)
        self.tag(sc, EXACT, op)


ANIMATIONS = {
    "pseudo-random-generators": PseudoRandomGenerators,
    "seeds-and-streams": SeedsAndStreams,
    "sampling-a-distribution": SamplingADistribution,
    "entropy-sources": EntropySources,
    "measuring-entropy": MeasuringEntropy,
}

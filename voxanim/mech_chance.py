"""Chance with structure, and what a seed decides: animations 115-117 of the "next"
catalogue (see :mod:`voxanim.mech`).

* :class:`MarkovChains`: a chain counted from real text, walked, and measured;
* :class:`SeedsTraining`: one MNIST network trained from many seeds;
* :class:`SeedsText`: one prompt sampled by GPT-2 from many seeds.

The three draw the experiments of :mod:`voxanim.study_chance`, which records every
array shown here (counts, walks, runs, sampled tokens): nothing is computed in this
module but layout, and the counting of the 25 character pairs that phase 1 of the
first animation counts on screen.

Wherever a draw is random, the seed that fixes it is on screen, and the draw is shown
as what it is: a uniform number landing on a bar of chances.
"""

from __future__ import annotations

import numpy as np

from . import study_chance as S
from . import theme
from .diagram import draw_image
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIGIT_INK,
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

BIG = 0.034                                   # headings over a drawing
GREY = (0.56, 0.58, 0.60)


def show(c):
    """A character as something visible."""
    return "␣" if c == " " else "↵" if c == "\n" else c


def visible(text):
    return str(text).replace("\n", "↵")


def pct(v, digits=1):
    return "%.*f%%" % (digits, 100.0 * float(v))


class Chance(Mechanism):
    """What the three animations share: curves through measured points, legends, a
    bar of chances with the uniform numbers that pick from it, and dot plots."""

    def trace(self, sc, chart, xs, ys, ink, op, go=1.0, width=2.6, dots=True, r=4.0):
        """``ys`` against ``xs`` through every measured point, drawn as far as fraction
        ``go`` of the axis."""
        if op <= 0.01 or go <= 0:
            return
        fx = [chart.fx(float(x)) for x in xs]
        fy = [chart.fy(float(y)) for y in ys]
        upto = fx[0] + (fx[-1] - fx[0]) * min(float(go), 1.0)
        pts = []
        for i in range(len(fx)):
            if fx[i] <= upto + 1e-9:
                pts.append((fx[i], fy[i]))
                if dots:
                    self.disk(sc, self.P(fx[i], fy[i]), r, ink, op, lift=1.4, outline=0.0)
            else:
                a = (upto - fx[i - 1]) / (fx[i] - fx[i - 1])
                pts.append((upto, fy[i - 1] + a * (fy[i] - fy[i - 1])))
                break
        if len(pts) > 1:
            self.polyline(sc, [self.P(x, y) for x, y in pts], width, ink, op, bias=0.3)

    def key(self, sc, fx, fy, entries, op, step=0.046, size=0.031):
        """A legend: a colour chip and a name per entry, stacked down from ``(fx, fy)``."""
        for n, (name, ink) in enumerate(entries):
            y = fy + n * step
            self.box(sc, fx, y, 14, 14, ink, op, radius=3)
            self.text(sc, fx + 0.016, y, name, op, anchor="lm", ink=dark(ink, 0.3), size=size)

    def strip(self, sc, values, fx, fy, w, h, op, ink, vmax):
        """A distribution as one row of cells, darker where the chance is larger."""
        self.grid(sc, np.asarray(values, float)[None], fx, fy, w, h, op, vmax=vmax, hot=ink,
                  lines=0.0)

    def chances(self, sc, q, x0, x1, fy, h, op, inks=None, names=(), parts=40, size=0.030):
        """A distribution as a bar from ``x0`` to ``x1``: one segment per outcome, as wide
        as its chance (the first ``parts`` outcomes; the rest as one pale block). Returns
        ``at(u)``: the layout x of a uniform number ``u``."""
        q = np.asarray(q, float)
        cum = np.concatenate([[0.0], np.cumsum(q)])
        total, wide = cum[-1], x1 - x0
        self.rect(sc, self.P((x0 + x1) / 2, fy), self.span(wide)[0] + 6, h + 6, PAPER, op,
                  border=AXIS, border_w=1.2, radius=4, lift=0.1)
        for i in range(min(parts, len(q))):
            a, b = cum[i] / total, cum[i + 1] / total
            px = self.span((b - a) * wide)[0]
            if px < 0.6:
                continue
            ink = inks[i] if inks is not None else soft(BLUE, 0.30 + 0.28 * (i % 2))
            self.rect(sc, self.P(x0 + wide * (a + b) / 2, fy), max(px - 1.0, 0.6), h, ink, op,
                      radius=min(3.0, px / 3), lift=0.3)
            if i < len(names) and names[i] and px > 9.0 * len(names[i]) + 10:
                self.text(sc, x0 + wide * (a + b) / 2, fy, names[i], op, size=size, strong=True)
        if len(q) > parts and cum[parts] < total * 0.999:
            a = cum[parts] / total
            self.rect(sc, self.P(x0 + wide * (a + 1) / 2, fy), self.span((1 - a) * wide)[0] - 1,
                      h, DIM, op, radius=3, lift=0.3)
        return lambda u: x0 + wide * float(u)

    def pin(self, sc, fx, fy, h, ink, op, drop=0.03):
        """A uniform number on a bar of chances: a faint line through the bar at ``fx``,
        and a marker under it."""
        dy = self.frac(0, h / 2 + 3)[1]
        self.polyline(sc, [self.P(fx, fy - dy, lift=0.9), self.P(fx, fy + dy, lift=0.9)],
                      2.2, dark(ink, 0.25), 0.45 * op)
        self.polyline(sc, [self.P(fx, fy + dy, lift=0.9), self.P(fx, fy + dy + drop, lift=0.9)],
                      2.4, dark(ink, 0.25), op)
        self.disk(sc, self.P(fx, fy + dy + drop), 5.0, ink, op, lift=1.5, outline=0.1)

    def dots(self, sc, values, chart_x, fy, ink, op, shown=None, r=3.6, bin_px=7.6, up=True):
        """A dot per value on a horizontal axis (``chart_x``: value -> layout x), stacked
        where they fall in the same ``bin_px``-wide bin; the first ``shown`` only."""
        stack = {}
        for n, v in enumerate(values):
            if shown is not None and n >= shown:
                break
            x = self.span(chart_x(float(v)))[0]
            b = int(round(x / bin_px))
            k = stack[b] = stack.get(b, 0) + 1
            dy = self.frac(0, (k - 1) * (2 * r + 0.6))[1] * (-1 if up else 1)
            self.disk(sc, self.P(self.frac(b * bin_px)[0], fy + dy), r, ink, op, lift=1.2,
                      outline=0.12)

    def rug(self, sc, values, chart_x, fy, ink, op, shown=None, h=15.0):
        """A tick per value on a horizontal axis, and under them a bar from one standard
        deviation below their mean to one above (once all are shown)."""
        values = np.asarray(values, float)
        dy = self.frac(0, h / 2)[1]
        for n, v in enumerate(values):
            if shown is not None and n >= shown:
                break
            x = chart_x(float(v))
            self.polyline(sc, [self.P(x, fy - dy), self.P(x, fy + dy)], 2.2, ink, 0.75 * op)
        if shown is None or shown >= len(values):
            mean, sd = values.mean(), values.std(ddof=1)
            y = fy + dy + 0.008
            self.polyline(sc, [self.P(chart_x(mean - sd), y, lift=1.2),
                               self.P(chart_x(mean + sd), y, lift=1.2)], 3.2, dark(ink, 0.35),
                          op)

    def scale(self, sc, lo, hi, x0, x1, fy, op, ticks, fmt="%.1f%%", label=""):
        """A horizontal axis from ``lo`` to ``hi`` between ``x0`` and ``x1``; returns the
        value -> layout x map."""
        def at(v):
            return x0 + (x1 - x0) * (float(v) - lo) / (hi - lo)
        self.polyline(sc, [self.P(x0, fy), self.P(x1, fy)], 1.2, AXIS, 0.8 * op)
        for v in ticks:
            self.polyline(sc, [self.P(at(v), fy), self.P(at(v), fy + 0.012)], 1.2, AXIS,
                          0.8 * op)
            self.text(sc, at(v), fy + 0.036, fmt % (100 * v), op, size=0.029)
        if label:
            self.text(sc, (x0 + x1) / 2, fy + 0.078, label, op, size=0.030)
        return at


# ====================================================================== 115: Markov chains
INK4 = (RED, BLUE, GREEN, AMBER)              # vowel, consonant, space, mark
SIGN4 = ("V", "C", "␣", "!")
SHORT4 = ("vowel", "cons.", "space", "mark")
NODES = ((0.29, 0.41), (0.135, 0.62), (0.445, 0.62), (0.29, 0.83))
LOOPS = (((0.0, -0.066), (0.05, -0.075)), ((-0.032, -0.066), (-0.032, -0.122)),
         ((0.032, -0.066), (0.032, -0.122)), ((0.0, 0.066), (0.05, 0.075)))
NODE_W, NODE_H = 98.0, 36.0
HOPS = 22                                     # hops of the walk drawn one by one
START_INKS = (MAGENTA, BLUE, GREEN, AMBER)


class MarkovChains(Chance):
    """Counting which kind of character follows which in real text; the counts as a
    matrix of chances; a seeded walker hopping between the four states with its share
    of time settling on the matrix's fixed vector; the 65-character chain forgetting
    its start; its entropy rate against single characters; and chains of longer
    memory writing better text until their contexts are too rare."""

    TITLE = "A Markov chain from real text"
    PHASES = (("counting", 10.0), ("the walk", 13.0), ("forgetting", 10.0), ("entropy", 9.0),
              ("order", 13.0))
    ORDERS_SHOWN = (0, 1, 2, 3, 5)

    def setup(self):
        d = self.d = S.markov_data()
        s = self.study
        m = self.meta = s.metadata
        self.chars = str(d["alphabet"])
        self.P4 = np.asarray(s.panels[0].data["values"], float)
        self.C4 = np.asarray(s.panels[0].data["counts"], float)
        self.pi4 = np.asarray(s.panels[0].data["stationary"], float)
        self.head = str(d["head"])
        kinds = [int(d["classes"][self.chars.index(c)]) for c in self.head]
        self.head_kinds = kinds
        counts = np.zeros((len(kinds), 4, 4))
        for i in range(1, len(kinds)):                 # the pairs counted on screen
            counts[i] = counts[i - 1]
            counts[i, kinds[i - 1], kinds[i]] += 1
        self.head_counts = counts
        self.path = np.asarray(d["walk_path"], int)
        self.us = np.asarray(d["walk_us"], float)
        self.walk_at = [int(v) for v in s.panels[1].data["walk_at"]]
        self.walk_shares = np.asarray(s.panels[1].data["walk_shares"], float)
        self.walk_off = float(np.abs(self.walk_shares[-1] - self.pi4).max())
        self.forget = np.asarray(s.panels[1].data["ys"], float)
        self.dist = np.asarray(s.panels[1].data["distributions"], float)
        self.pi = np.asarray(d["pi"], float)
        self.rank = np.argsort(-self.pi, kind="stable")
        self.row_bits = np.asarray(s.panels[2].data["row_bits"], float)
        self.bits = [float(v) for v in s.panels[2].data["values"]]
        self.orders = [int(v) for v in s.panels[3].data["x"]]
        self.held, self.fitted = (np.asarray(v, float) for v in s.panels[3].data["ys"])
        self.contexts = [int(v) for v in s.panels[3].data["contexts"]]
        self.unseen = np.asarray(s.panels[3].data["unseen"], float)
        self.best = int(m["best_order"])
        self.texts = [visible(m["generated"][str(k)])[:46] for k in self.orders]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "a Markov chain needs one table: how often each state follows each. Counted "
            "over the %s character pairs of the training text, each row divided by its "
            "total is a row of chances" % f"{m['split_sizes'][0] - 1:,}",
            "the walker's next hop depends only on where it stands: seed %d gives one "
            "number per hop, and the number picks from that state's row. After %s hops its "
            "time in each state is within %.1f points of exact"
            % (m["seed"], f"{self.walk_at[-1]:,}", 100 * self.walk_off),
            "with all %d characters as states the chain forgets where it began. After q "
            "comes u, almost surely; yet %d steps on, every start is within %.3f of one "
            "fixed vector (exact, by powers of the matrix)"
            % (m["characters"], self.forget.shape[1] - 1, self.forget[:, -1].max()),
            "Shannon entropy, in bits per character: %.2f if all %d were equally likely, "
            "%.2f for single characters by frequency, %.2f once the current character is "
            "known. That last is the chain's entropy rate"
            % (self.bits[0], m["characters"], self.bits[1], self.bits[2]),
            "longer memory writes better text. Held-out cross-entropy falls from %.2f bits "
            "a character (order 0) to %.2f at order %d, then rises: at order %d, %d%% of "
            "held-out contexts never occurred in training"
            % (self.held[0], self.held[self.best], self.best, self.orders[-1],
               round(100 * self.unseen[-1])),
        ][k]

    # ------------------------------------------------------------------ phase 1
    def _table(self, sc, values, x0, fy, op, fmt, shade=None, row_names=True, ring=None):
        """A 4 x 4 table of numbers with the kinds down the side and across the top."""
        dx, dy = 0.078, 0.075
        for c in range(4):
            self.text(sc, x0 + dx * c, fy - 0.058, SHORT4[c], op, size=0.029,
                      ink=dark(INK4[c], 0.3))
        for r in range(4):
            if row_names:
                self.text(sc, x0 - 0.047, fy + dy * r, S.CLASS_NAMES[r], op, anchor="rm",
                          size=0.029, ink=dark(INK4[r], 0.3))
            for c in range(4):
                fill = PAPER if shade is None else soft(INK4[c], 1.0 - 0.8 * shade[r, c])
                self.box(sc, x0 + dx * c, fy + dy * r, 62, 32, fill, op, fmt(values[r, c]),
                         border=DIM if shade is None else INK4[c], size=0.029, radius=4)
        if ring is not None:
            self.rect(sc, self.P(x0 + dx * ring[1], fy + dy * ring[0]), 68, 38, AMBER, op,
                      border=AMBER, border_w=2.6, radius=5, lift=0.9, fill_op=0.0)

    def phase_1(self, sc, u, op):
        head, kinds = self.head, self.head_kinds
        n = len(head)
        self.text(sc, 0.5, 0.315, "real text; every character is a vowel, a consonant, a "
                  "space or a mark", op, strong=True, size=BIG)
        xs = [0.06 + 0.88 * (i + 0.5) / n for i in range(n)]
        for i, c in enumerate(head):
            self.box(sc, xs[i], 0.395, 25, 32, soft(INK4[kinds[i]], 0.5), op, show(c),
                     ink=dark(INK4[kinds[i]], 0.5), border=INK4[kinds[i]], size=0.032,
                     radius=4)
        counting = u < 0.52
        k = int(np.clip((u - 0.05) / 0.42, 0, 1) * (n - 1) + 1e-9)
        if counting and k > 0:
            j = k - 1                                  # the pair just counted
            self.rect(sc, self.P((xs[j] + xs[j + 1]) / 2, 0.395), 60, 42, AMBER, op,
                      border=AMBER, border_w=2.6, radius=5, lift=0.9, fill_op=0.0)
        if counting:
            self.text(sc, 0.297, 0.49, "pairs counted so far: %d" % k, op, strong=True,
                      size=0.032)
            pair = (kinds[k - 1], kinds[k]) if k else None
            self._table(sc, self.head_counts[k], 0.18, 0.60, op, lambda v: "%d" % v, ring=pair)
            if pair:
                on = op * (1 - smoothstep(0.46, 0.52, u))
                self.text(sc, 0.76, 0.49, "the pair just read", on, strong=True, size=0.032)
                for n, x in enumerate((0.70, 0.82)):
                    kind = pair[n]
                    self.box(sc, x, 0.62, 58, 58, soft(INK4[kind], 0.5), on,
                             show(head[k - 1 + n]), ink=dark(INK4[kind], 0.5),
                             border=INK4[kind], size=0.05)
                    self.text(sc, x, 0.715, S.CLASS_NAMES[kind], on, ink=dark(INK4[kind], 0.3),
                              size=0.031)
                self.arrow(sc, (0.737, 0.62), (0.783, 0.62), AXIS, on, 1.0, 2.6)
                self.text(sc, 0.76, 0.80, "add 1 where that row meets that column", on,
                          size=0.030)
        else:
            self.text(sc, 0.297, 0.49, "all %s pairs of the text" % f"{int(self.C4.sum()):,}",
                      op, strong=True, size=0.032)
            self._table(sc, self.C4, 0.18, 0.60, op, lambda v: f"{int(v):,}")
        self.text(sc, 0.297, 0.905, "row: this kind;  column: the kind that follows", op,
                  size=0.030)
        show_p = op * smoothstep(0.64, 0.76, u)
        self.arrow(sc, (0.468, 0.70), (0.548, 0.70), AXIS, op, smoothstep(0.58, 0.68, u), 2.6)
        self.text(sc, 0.508, 0.745, "each row", show_p, size=0.029)
        self.text(sc, 0.508, 0.785, "÷ its total", show_p, size=0.029)
        self.text(sc, 0.802, 0.49, "the transition matrix", show_p, strong=True, size=0.032)
        self._table(sc, self.P4, 0.685, 0.60, show_p, lambda v: "%.2f" % v,
                    shade=self.P4 / self.P4.max(), row_names=False)
        self.text(sc, 0.802, 0.905, "the chance of what comes next", show_p, size=0.030)

    # ------------------------------------------------------------------ phase 2
    def _bow(self, a, b, side=0.17, n=25):
        """Layout points of a curve from node ``a`` to node ``b`` that bulges to the left
        of its direction of travel, so the two directions of a pair do not coincide;
        the parts under the two node boxes are left out."""
        pa, pb = np.array(self.span(*NODES[a])), np.array(self.span(*NODES[b]))
        d = pb - pa
        mid = (pa + pb) / 2 + np.array([d[1], -d[0]]) * side
        s = np.linspace(0.0, 1.0, n)[:, None]
        pts = (1 - s) ** 2 * pa + 2 * s * (1 - s) * mid + s ** 2 * pb
        half = np.array([NODE_W / 2 + 5, NODE_H / 2 + 5])
        keep = [p for p in pts if (np.abs(p - pa) > half).any() and (np.abs(p - pb) > half).any()]
        return [self.frac(*p) for p in keep]

    def _loop(self, a, n=22):
        (dx, dy), _ = LOOPS[a]
        cx, cy = NODES[a][0] + dx, NODES[a][1] + dy
        rx, ry = self.frac(13.0, 13.0)
        ang = np.linspace(0, 2 * np.pi, n)
        return [(cx + rx * np.cos(t), cy + ry * np.sin(t)) for t in ang]

    def _along(self, pts, f):
        k = float(np.clip(f, 0, 1)) * (len(pts) - 1)
        i = int(min(k, len(pts) - 2))
        (ax, ay), (bx, by) = pts[i], pts[i + 1]
        return ax + (bx - ax) * (k - i), ay + (by - ay) * (k - i)

    def _graph(self, sc, op, now, label_op):
        """The four states and their sixteen arrows; the arrows out of state ``now``
        carry their chances."""
        for a in range(4):
            for b in range(4):
                p = self.P4[a, b]
                on = a == now
                ink = dark(INK4[a], 0.15) if on else GREY
                width = 1.2 + 4.0 * p
                pts = self._loop(a) if a == b else self._bow(a, b)
                world = [self.P(x, y, lift=0.6 if on else 0.2) for x, y in pts]
                self.polyline(sc, world, width, ink, op * (1.0 if on else 0.45))
                if a != b:
                    self._head(sc, world[-2], world[-1], ink, op * (1.0 if on else 0.45),
                               width)
                if on and label_op > 0.01:
                    if a == b:
                        lx, ly = NODES[a][0] + LOOPS[a][1][0], NODES[a][1] + LOOPS[a][1][1]
                    else:
                        pa, pb = (np.array(self.span(*NODES[k])) for k in (a, b))
                        d = (pb - pa) / np.hypot(*(pb - pa))
                        out = (pa + pb) / 2 + np.array([d[1], -d[0]]) * (
                            0.085 * np.hypot(*(pb - pa)) + 19.0)
                        lx, ly = self.frac(*out)
                    self.text(sc, lx, ly, "%.2f" % p, op * label_op, size=0.030,
                              ink=dark(INK4[b], 0.35))
        for a in range(4):
            x, y = NODES[a]
            self.box(sc, x, y, NODE_W, NODE_H, soft(INK4[a], 0.45 if a == now else 0.7), op,
                     S.CLASS_NAMES[a], ink=dark(INK4[a], 0.5), border=INK4[a], size=0.031,
                     glow=0.25 if a == now else 0.0)

    def phase_2(self, sc, u, op):
        h = float(np.clip((u - 0.03) / 0.75, 0, 1)) * HOPS
        i = int(min(h, HOPS - 1))
        f = h - i if h < HOPS else 1.0
        walking = h < HOPS
        a, b = int(self.path[i]), int(self.path[i + 1])
        now = a if walking else int(self.path[HOPS])
        self._graph(sc, op, now, 1.0)
        # the walker
        if walking:
            pts = self._loop(a) if a == b else [NODES[a], *self._bow(a, b), NODES[b]]
            x, y = self._along(pts, smoothstep(0.4, 0.92, f))
        else:
            x, y = NODES[now]
        if walking and f >= 0.4:
            # gone before it reaches the node, so it never sits on the state's name
            landing = 1.0 - smoothstep(0.8, 0.9, f)
            self.disk(sc, self.P(x, y, lift=1.8), 7.5, DIGIT_INK, op * landing, glow=0.3,
                      outline=0.0)
        else:
            self.rect(sc, self.P(x, y), NODE_W + 8, NODE_H + 8, DIGIT_INK, op,
                      border=DIGIT_INK, border_w=3.0, radius=8, lift=0.9, fill_op=0.0)
        # where the seed enters: a uniform number on the current state's row
        x0, x1 = 0.60, 0.935
        self.text(sc, x0, 0.315, "seed %d  →  one number u per hop" % self.meta["seed"], op,
                  anchor="lm", strong=True, size=0.032)
        self.text(sc, x0, 0.368, "the row of “%s”, as a bar" % S.CLASS_NAMES[now], op,
                  anchor="lm", ink=dark(INK4[now], 0.35), size=0.030)
        at = self.chances(sc, self.P4[now], x0, x1, 0.425, 26, op, inks=INK4, names=SIGN4)
        if walking:
            uu = float(self.us[i])
            self.pin(sc, at(uu), 0.425, 26, DIGIT_INK, op * smoothstep(0.02, 0.2, f))
            left = at(uu) > x1 - 0.11
            self.text(sc, at(uu) + (-0.014 if left else 0.014), 0.49, "u = %.2f" % uu,
                      op * smoothstep(0.02, 0.2, f), size=0.030, strong=True,
                      anchor="rm" if left else "lm")
        # the states visited
        done = i + (1 if f > 0.9 else 0) if walking else HOPS
        self.text(sc, x0, 0.56, "the states it visits", op, anchor="lm", size=0.030)
        for j in range(done):
            st = int(self.path[j + 1])
            self.box(sc, x0 + 0.014 + 0.0305 * (j % 11), 0.61 + 0.05 * (j // 11), 22, 21,
                     soft(INK4[st], 0.45), op, SIGN4[st], ink=dark(INK4[st], 0.5),
                     border=INK4[st], size=0.029, radius=3)
        # its share of time in each state, against the fixed vector
        if walking:
            steps = done
            share = (np.bincount(self.path[1:done + 1], minlength=4) / max(done, 1)
                     if done else np.zeros(4))
        else:
            stage = int(min((u - 0.78) / 0.22 * 3, 2))
            idx = (self.walk_at.index(100), self.walk_at.index(1000), len(self.walk_at) - 1)
            steps, share = self.walk_at[idx[stage]], self.walk_shares[idx[stage]]
        self.text(sc, x0, 0.715, "share of time after %s hops" % f"{steps:,}", op,
                  anchor="lm", size=0.030)
        for c in range(4):
            y = 0.76 + 0.038 * c
            self.hbar(sc, 0.705, y, 260 * min(share[c], 0.577), INK4[c], op,
                      label=S.CLASS_NAMES[c], thick=12, track=150)
            self.text(sc, 0.935, y, pct(share[c], 1 if not walking else 0), op, anchor="rm",
                      size=0.029, strong=True)
            tick = 0.705 + self.frac(260 * self.pi4[c])[0]
            self.polyline(sc, [self.P(tick, y - 0.019, lift=1.5), self.P(tick, y + 0.019,
                                                                         lift=1.5)],
                          2.4, DIGIT_INK, op)
        self.text(sc, x0, 0.925, "|  exact: the matrix's fixed vector", op, anchor="lm",
                  size=0.029, strong=True)

    # ------------------------------------------------------------------ phase 3
    def phase_3(self, sc, u, op):
        steps = self.dist.shape[1] - 1
        t = float(np.clip((u - 0.08) / 0.62, 0, 1)) * steps
        i = int(min(t, steps - 1))
        f = t - i
        now = int(round(t))
        self.text(sc, 0.5, 0.315, "65 states now. The chance of each character after %d "
                  "step%s, from four starts" % (now, "" if now == 1 else "s"), op,
                  strong=True, size=BIG)
        x, w = 0.59, 600
        for s, c in enumerate(S.MIX_STARTS):
            y = 0.374 + 0.046 * s
            p = self.dist[s, i] * (1 - f) + self.dist[s, i + 1] * f
            self.text(sc, 0.225, y, "start at %s" % show(c), op, anchor="rm",
                      ink=dark(START_INKS[s], 0.3), size=0.030)
            self.strip(sc, p[self.rank], x, y, w, 17, op, START_INKS[s], 0.22)
        self.text(sc, 0.225, 0.567, "the fixed vector", op, anchor="rm", ink=dark(TEAL, 0.3),
                  size=0.030)
        self.strip(sc, self.pi[self.rank], x, 0.567, w, 17, op, TEAL, 0.22)
        first = "  ".join(show(self.chars[k]) for k in self.rank[:12])
        self.text(sc, x, 0.617, "one cell per character, most frequent first:  %s  …" % first,
                  op, size=0.029)
        xs = np.arange(steps + 1)
        chart = self.chart(((0.245, 0.70), (0.705, 0.875)), xs, [1e-4, 1.0], ylog=True)
        self.axes(sc, chart, op, "steps", "distance to the fixed vector",
                  xticks=[(v, str(v)) for v in (0, 3, 6, 9, 12)],
                  yticks=[(1.0, "1"), (0.01, "0.01"), (0.0001, "0.0001")])
        for s in range(len(S.MIX_STARTS)):
            self.trace(sc, chart, xs, np.maximum(self.forget[s], 1e-4), START_INKS[s], op,
                       t / steps, width=2.4, r=3.4)
        self.lines(sc, 0.74, 0.725, ["exact: the start,", "times the matrix",
                                     "once per step"], op, size=0.030, step=0.044)

    # ------------------------------------------------------------------ phase 4
    SHOWN_AFTER = ("q", "t", "e", " ")

    def phase_4(self, sc, u, op):
        n = len(self.chars)
        x, w, bx, unit = 0.305, 420, 0.60, 34.0             # bars: 34 px per bit
        self.text(sc, bx, 0.325, "Shannon entropy, bits per character", op, anchor="lm",
                  strong=True, size=0.032)
        rows = (("%d characters, all equally likely" % n, np.full(n, 1.0 / n), GREY,
                 self.bits[0], 0.0),
                ("single characters, by frequency", self.pi[self.rank], TEAL, self.bits[1],
                 0.12))
        for k, (name, p, ink, bits, t0) in enumerate(rows):
            y = 0.415 + 0.13 * k
            on = op * smoothstep(t0, t0 + 0.1, u)
            self.text(sc, 0.062, y - 0.047, name, on, anchor="lm", ink=dark(ink, 0.3),
                      size=0.030)
            self.strip(sc, p, x, y, w, 20, on, ink, 0.22)
            self.hbar(sc, bx, y, unit * bits, ink, on, value="%.2f" % bits, thick=18,
                      u=smoothstep(t0, t0 + 0.2, u))
        turn = int(np.clip((u - 0.3) / 0.5, 0, 0.999) * len(self.SHOWN_AFTER))
        c = self.SHOWN_AFTER[turn]
        k = self.chars.index(c)
        on = op * smoothstep(0.26, 0.34, u)
        y = 0.675
        self.text(sc, 0.062, y - 0.047, "the next character, when this one is %s" % show(c),
                  on, anchor="lm", ink=dark(VIOLET, 0.3), size=0.030)
        self.strip(sc, np.asarray(self.d["P"], float)[k][self.rank], x, y, w, 20, on, VIOLET,
                   0.22)
        self.hbar(sc, bx, y, unit * self.row_bits[k], soft(VIOLET, 0.35), on,
                  value="%.2f after %s" % (abs(self.row_bits[k]), show(c)), thick=18)
        last = op * smoothstep(0.78, 0.9, u)
        self.text(sc, 0.062, 0.795, "averaged over this character, weighted by the fixed "
                  "vector:", last, anchor="lm", ink=dark(VIOLET, 0.3), size=0.030)
        self.hbar(sc, bx, 0.795, unit * self.bits[2], VIOLET, last,
                  value="%.2f: the entropy rate" % self.bits[2], thick=18)
        self.text(sc, 0.5, 0.895, "exact for the chain counted from the text; the text "
                  "itself remembers more than one character", last, size=0.030)

    # ------------------------------------------------------------------ phase 5
    def phase_5(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.14, 0.318, "written by chains of each order, all from seed %d"
                  % m["seed"], op, anchor="lm", strong=True, size=0.031)
        self.text(sc, 0.80, 0.318, "contexts seen", op, anchor="rm", strong=True, size=0.031)
        self.text(sc, 0.945, 0.318, "new to it", op, anchor="rm", strong=True, size=0.031)
        for n, k in enumerate(self.ORDERS_SHOWN):
            y = 0.368 + 0.046 * n
            on = op * smoothstep(0.03 + 0.07 * n, 0.09 + 0.07 * n, u)
            i = self.orders.index(k)
            chars = int(np.clip((u - 0.03 - 0.07 * n) / 0.3, 0, 1) * len(self.texts[i]))
            self.text(sc, 0.052, y, "order %d" % k, on, anchor="lm", ink=dark(BLUE, 0.3),
                      size=0.030)
            self.text(sc, 0.14, y, self.texts[i][:chars].strip(), on, anchor="lm", size=0.030,
                      strong=True)
            self.text(sc, 0.80, y, f"{self.contexts[i]:,}", on, anchor="rm", size=0.030)
            self.text(sc, 0.945, y, pct(self.unseen[i], 1), on, anchor="rm", size=0.030)
        xs = np.asarray(self.orders, float)
        chart = self.chart(((0.11, 0.59), (0.655, 0.86)), xs, [1.0, 5.0], pad=0.0)
        self.axes(sc, chart, op, "order: characters of memory", "bits per character",
                  xticks=[(v, str(int(v))) for v in xs],
                  yticks=[(v, str(v)) for v in (1, 2, 3, 4, 5)])
        go = (u - 0.42) / 0.3
        self.trace(sc, chart, xs, self.fitted, OCHRE, op, go)
        self.trace(sc, chart, xs, self.held, VIOLET, op, go)
        mark = op * smoothstep(0.74, 0.82, u)
        self.disk(sc, self.P(chart.fx(self.best), chart.fy(self.held[self.best])), 8.5, VIOLET,
                  mark, ring=DIGIT_INK, ring_w=2.0, lift=1.6, outline=0.0)
        self.key(sc, 0.645, 0.665, [("held-out text (measured)", VIOLET),
                                    ("the text it was counted from", OCHRE)], op)
        self.lines(sc, 0.645, 0.775, [
            ("best on held-out text: order %d, %.2f bits" % (self.best, self.held[self.best]),
             dark(VIOLET, 0.3)),
            ("on its own text the score only falls:", dark(OCHRE, 0.3)),
            ("an estimate biased low", dark(OCHRE, 0.3))], mark, size=0.030, step=0.044)


# ====================================================================== 116: seeds, training
SOURCE_INKS = (RED, BLUE, GREEN)              # initial weights, data order, dropout masks
ALL_INK = VIOLET
NARROW, WIDE = TEAL, MAGENTA                  # the two settings compared


class SeedsTraining(Chance):
    """One seed feeding three generators (weights, data order, dropout masks) and the
    same seed reproducing every weight; the same network from 64 seeds, its accuracy
    spread; which test digits the runs get wrong; one source varied at a time; and how
    often a few seeds name the wrong winner of two nearly equal settings."""

    TITLE = "One network, many seeds"
    PHASES = (("one seed", 11.0), ("many seeds", 10.0), ("mistakes", 11.0), ("sources", 12.0),
              ("comparing", 12.0))

    def setup(self):
        r = self.r = S.seeds_summary()
        d = self.d = r["d"]
        s = self.study
        self.meta = s.metadata
        self.updates = np.asarray(s.panels[0].data["x"], float)
        self.curves = np.asarray(s.panels[0].data["ys"], float)
        self.final = np.asarray(s.panels[0].data["final"], float)
        wrong = np.asarray(s.panels[1].data["values"], float)
        cols = wrong.shape[1] // 2 * 2
        v = wrong[:, :cols].reshape(len(wrong), -1, 2).mean(2)[..., None]
        self.wrong_tex = (PAPER * (1 - v) + np.asarray(RED) * v).astype(np.float32)
        self.spread = np.asarray(s.panels[2].data["per_base"], float)
        self.by_source = [np.asarray(s.panels[2].data[k], float)
                          for k in ("init", "order", "drop")]
        self.acc_a = np.asarray(s.panels[3].data["width_a"], float)
        self.acc_b = np.asarray(s.panels[3].data["width_b"], float)
        self.ns = [int(v) for v in s.panels[3].data["x"]]
        self.share = np.asarray(s.panels[3].data["ys"], float)[0]
        self.pairs = np.asarray(s.panels[3].data["pairs_shown"], int)
        self.weights = sum(a * b + b for a, b in ((784, S.WIDTH), (S.WIDTH, 10)))
        self.images, self.labels = d["test_images"], d["test_labels"]
        self.vmax = float(max(np.abs(d["init_0"]).max(), np.abs(d["init_1"]).max()))

    def subtitle_for(self, k):
        m = self.meta
        acc, mis, cmp = m["accuracy"], m["mistakes"], m["comparison"]
        sp = [100 * float(np.mean(m["spread_by_source"][name])) for name in S.SOURCES]
        return [
            "one seed feeds three generators: the starting weights, the order the digits "
            "are shown in, and the dropout masks. Another seed changes all three; the same "
            "seed gives each of the %s weights again, bit for bit" % f"{self.weights:,}",
            "the same network on the same data from %d seeds: held-out accuracy ends "
            "anywhere from %.2f%% to %.2f%% (mean %.2f%%, standard deviation %.2f points). "
            "One run is a draw, not the answer"
            % (len(self.final), 100 * acc["min"], 100 * acc["max"], 100 * acc["mean"],
               100 * acc["std"]),
            "each run gets about %d of 10,000 test digits wrong, but not the same ones: two "
            "runs share %d%% of their mistakes. %d digits defeat every run; %s are missed "
            "by at least one"
            % (round(mis["mean_per_run"]), round(100 * mis["overlap"]),
               mis["wrong_in_every_run"], f"{mis['wrong_in_some_run']:,}"),
            "change one source and hold the other two: the weights alone spread accuracy "
            "by %.2f points, the data order by %.2f, the dropout masks by %.2f; all three "
            "together by %.2f. The spreads do not add up"
            % (sp[0], sp[1], sp[2], 100 * m["spread_all"]),
            "%d hidden units against %d: over %d seeds each, really %.2f points apart. With "
            "one seed each the worse one comes out ahead %d%% of the time; it takes %d "
            "seeds each to get that under 5%%"
            % (cmp["widths"][0], cmp["widths"][1], len(self.acc_a), 100 * cmp["gap"],
               round(100 * cmp["wrong_winner"][0]), cmp["seeds_for_under_5_percent"]),
        ][k]

    # ------------------------------------------------------------------ phase 1
    def phase_1(self, sc, u, op):
        d = self.d
        part = 0 if u < 0.36 else 1 if u < 0.68 else 2
        w = (u - (0.0, 0.36, 0.68)[part]) / (0.36, 0.32, 0.32)[part]
        seed = (0, 1, 0)[part]
        self.text(sc, 0.095, 0.675, ("", "another seed", "seed 0 again")[part], op,
                  size=0.029, strong=True)
        self.box(sc, 0.095, 0.595, 86, 50, soft(AMBER, 0.4), op, "seed %d" % seed, border=AMBER,
                 size=0.038, glow=0.2)
        ys = (0.40, 0.595, 0.79)
        names = S.SOURCES
        for k in range(3):
            y = ys[k]
            grow = smoothstep(0.02 + 0.05 * k, 0.16 + 0.05 * k, w)
            self.arc(sc, (0.148, 0.595), (0.222, y), SOURCE_INKS[k], op, grow,
                     0.0 if k == 1 else 0.02 * (1 - k), 2.6)
            on = op * smoothstep(0.14 + 0.05 * k, 0.26 + 0.05 * k, w)
            if k == 0:
                self.grid(sc, d["init_%d" % seed], 0.34, y, 78, 78, on, vmax=self.vmax)
            elif k == 1:
                for j in range(6):
                    self.digit(sc, d["order_%d" % seed][j], 0.34 + (j - 2.5) * 0.0365, y, 30,
                               on)
            else:
                self.grid(sc, d["mask_%d" % seed].astype(float), 0.34, y, 78, 78, on, vmax=1.0,
                          hot=GREEN)
            self.text(sc, 0.462, y - 0.022, names[k], on, anchor="lm",
                      ink=dark(SOURCE_INKS[k], 0.3), size=0.032)
            self.text(sc, 0.462, y + 0.022, "generator [%d, %d]" % (seed, k), on, anchor="lm",
                      size=0.029)
            self.arc(sc, (0.625, y), (0.692, 0.595), SOURCE_INKS[k], op,
                     smoothstep(0.36, 0.5, w), 0.0, 2.2)
        self.box(sc, 0.79, 0.595, 160, 64, soft(VIOLET, 0.7), op, border=VIOLET)
        self.text(sc, 0.79, 0.573, "training", op, ink=dark(VIOLET, 0.45), size=0.032)
        self.text(sc, 0.79, 0.617, "%d updates" % self.updates[-1], op, ink=dark(VIOLET, 0.45),
                  size=0.029)
        self.text(sc, 0.79, 0.475, "784 → %d → 10" % S.WIDTH, op, size=0.030)
        self.text(sc, 0.79, 0.43, "the same network and data", op, size=0.030)
        self.arrow(sc, (0.79, 0.665), (0.79, 0.72), VIOLET, op, smoothstep(0.5, 0.6, w), 2.6)
        out = op * smoothstep(0.58, 0.7, w)
        self.text(sc, 0.79, 0.76, "held-out accuracy", out, size=0.030)
        self.text(sc, 0.79, 0.815, "%.2f%%" % (100 * self.final[seed]), out, size=0.05,
                  strong=True)
        if part == 2:
            self.text(sc, 0.5, 0.925, "trained twice from seed 0: all %s weights identical, "
                      "bit for bit" % f"{self.weights:,}", op * smoothstep(0.6, 0.75, w),
                      strong=True, size=0.031)

    # ------------------------------------------------------------------ phase 2
    def phase_2(self, sc, u, op):
        acc = self.meta["accuracy"]
        n = len(self.final)
        self.text(sc, 0.335, 0.315, "held-out accuracy during training, %d seeds" % n, op,
                  strong=True, size=BIG)
        xs = self.updates
        chart = self.chart(((0.11, 0.56), (0.395, 0.84)), xs, [0.86, 0.95], pad=0.0)
        self.axes(sc, chart, op, "updates", "",
                  xticks=[(v, str(int(v))) for v in (312, 624, 936)],
                  yticks=[(v, "%d%%" % round(100 * v)) for v in (0.86, 0.89, 0.92, 0.95)])
        go = u / 0.5
        lo, hi = int(np.argmin(self.final)), int(np.argmax(self.final))
        for k in range(n):
            if k not in (lo, hi):
                self.trace(sc, chart, xs, np.maximum(self.curves[k], 0.86), BLUE, 0.4 * op, go,
                           width=1.1, dots=False)
        for k, ink in ((lo, RED), (hi, GREEN)):
            self.trace(sc, chart, xs, np.maximum(self.curves[k], 0.86), ink, op, go, width=2.6,
                       dots=False)
        self.text(sc, 0.785, 0.315, "where the %d runs end" % n, op, strong=True, size=BIG)
        at = self.scale(sc, 0.934, 0.947, 0.625, 0.945, 0.80, op, (0.935, 0.94, 0.945),
                        label="held-out accuracy")
        shown = int(np.clip((u - 0.45) / 0.35, 0, 1) * n)
        self.dots(sc, self.final, at, 0.778, BLUE, op, shown=shown, r=4.6, bin_px=9.8)
        note = op * smoothstep(0.8, 0.9, u)
        self.lines(sc, 0.785, 0.395, [
            "mean %.2f%%" % (100 * acc["mean"]),
            "from %.2f%% to %.2f%%" % (100 * acc["min"], 100 * acc["max"]),
            "standard deviation %.2f points" % (100 * acc["std"])], note, size=0.031,
            step=0.045, anchor="mm")

    # ------------------------------------------------------------------ phase 3
    def phase_3(self, sc, u, op):
        mis, r = self.meta["mistakes"], self.r
        rows = len(self.wrong_tex)
        self.text(sc, 0.5, 0.315, "a row per seed, a column per test digit that some seed "
                  "gets wrong", op, strong=True, size=BIG)
        tex = self.wrong_tex.copy()
        shown = int(np.ceil(np.clip(u / 0.45, 0, 1) * rows))
        tex[shown:] = PAPER * 0.97
        self.rect(sc, self.P(0.5, 0.495), 768, 148, PAPER, op, border=AXIS, border_w=1.2,
                  radius=5, lift=0.1)
        draw_image(self, sc, self.P(0.5, 0.495), tex, 760, 140, op, grid=0)
        note = op * smoothstep(0.45, 0.58, u)
        self.text(sc, 0.06, 0.675, "←  wrong for all %d seeds: %d digits"
                  % (rows, mis["wrong_in_every_run"]), note, anchor="lm", ink=dark(RED, 0.3),
                  size=0.031)
        self.text(sc, 0.94, 0.675, "wrong for some seeds only: %s  →"
                  % f"{mis['wrong_in_some_run'] - mis['wrong_in_every_run']:,}", note,
                  anchor="rm", ink=dark(AMBER, 0.35), size=0.031)
        groups = ((0.075, r["always_ids"][:5], RED, "wrong every time", 0.5),
                  (0.40, r["half_ids"][:5], AMBER, "wrong for about half the seeds", 0.62))
        for x0, ids, ink, name, t0 in groups:
            on = op * smoothstep(t0, t0 + 0.12, u)
            self.text(sc, x0 - 0.025, 0.745, name, on, anchor="lm", ink=dark(ink, 0.3),
                      size=0.030)
            for j, k in enumerate(ids):
                self.digit(sc, self.images[int(k)], x0 + 0.058 * j, 0.845, 44, on, frame=ink)
        last = op * smoothstep(0.76, 0.88, u)
        self.lines(sc, 0.72, 0.745, [
            "mistakes per run: %d" % round(mis["mean_per_run"]),
            "two runs share %d%% of them" % round(100 * mis["overlap"]),
            "and disagree on %.1f%%" % (100 * mis["disagree"]),
            "of the test digits"], last, size=0.031, step=0.046)

    # ------------------------------------------------------------------ phase 4
    def phase_4(self, sc, u, op):
        heads = ("weights", "order", "dropout")
        cx = (0.085, 0.17, 0.255)
        for k in range(3):
            self.text(sc, cx[k], 0.325, heads[k], op, ink=dark(SOURCE_INKS[k], 0.3),
                      size=0.031)
        self.text(sc, 0.55, 0.325, "held-out accuracy of each run", op, strong=True,
                  size=0.031)
        self.text(sc, 0.945, 0.325, "standard deviation", op, anchor="rm", strong=True,
                  size=0.031)
        lo, hi = 0.934, 0.947
        at = self.scale(sc, lo, hi, 0.33, 0.75, 0.835, op, (0.935, 0.94, 0.945))
        rows = [(k, self.by_source[k], SOURCE_INKS[k]) for k in range(3)]
        for n, (k, runs, ink) in enumerate(rows):
            y = 0.42 + 0.105 * n
            t0 = 0.04 + 0.2 * n
            on = op * smoothstep(t0, t0 + 0.08, u)
            for j in range(3):
                varies = j == k
                self.box(sc, cx[j], y, 62, 30, soft(SOURCE_INKS[j], 0.35) if varies else PAPER,
                         on, "varies" if varies else "fixed",
                         ink=dark(SOURCE_INKS[j], 0.5) if varies else GREY,
                         border=SOURCE_INKS[j] if varies else DIM, size=0.029)
            shown = int(np.clip((u - t0 - 0.03) / 0.12, 0, 1) * runs.shape[1])
            for b in range(runs.shape[0]):
                self.rug(sc, np.clip(runs[b], lo, hi), at, y + (0.024 if b else -0.024), ink,
                         on, shown=shown, h=13.0)
            self.text(sc, 0.945, y, "%.2f and %.2f" % tuple(100 * self.spread[k]),
                      op * smoothstep(t0 + 0.12, t0 + 0.18, u), anchor="rm", strong=True,
                      size=0.031)
        y, t0 = 0.42 + 0.105 * 3, 0.66
        on = op * smoothstep(t0, t0 + 0.08, u)
        for j in range(3):
            self.box(sc, cx[j], y, 62, 30, soft(SOURCE_INKS[j], 0.35), on, "varies",
                     ink=dark(SOURCE_INKS[j], 0.5), border=SOURCE_INKS[j], size=0.029)
        shown = int(np.clip((u - t0 - 0.03) / 0.12, 0, 1) * len(self.final))
        self.rug(sc, np.clip(self.final, lo, hi), at, y, ALL_INK, on, shown=shown, h=20.0)
        self.text(sc, 0.945, y, "%.2f" % (100 * self.meta["spread_all"]),
                  op * smoothstep(t0 + 0.12, t0 + 0.18, u), anchor="rm", strong=True,
                  size=0.031)
        self.text(sc, 0.5, 0.925, "a tick per run; two rows: two choices of the fixed seeds; "
                  "the bar: mean ± one standard deviation (points)", op, size=0.029)

    # ------------------------------------------------------------------ phase 5
    def phase_5(self, sc, u, op):
        cmp = self.meta["comparison"]
        lo, hi = 0.934, 0.949
        self.text(sc, 0.06, 0.318, "two settings, %d seeds each" % len(self.acc_a), op,
                  anchor="lm", strong=True, size=BIG)
        at = self.scale(sc, lo, hi, 0.25, 0.66, 0.545, op, (0.935, 0.94, 0.945))
        rows = ((self.acc_a, NARROW, "%d hidden units" % cmp["widths"][0], 0.415),
                (self.acc_b, WIDE, "%d hidden units" % cmp["widths"][1], 0.515))
        for acc, ink, name, y in rows:
            self.text(sc, 0.06, y - 0.012, name, op, anchor="lm", ink=dark(ink, 0.3),
                      size=0.031)
            self.dots(sc, np.clip(acc, lo, hi), at, y, ink, op, r=3.0)
            mean = at(acc.mean())
            self.polyline(sc, [self.P(mean, y - 0.062, lift=1.6), self.P(mean, y + 0.012,
                                                                         lift=1.6)],
                          2.6, dark(ink, 0.35), op)
        self.text(sc, 0.69, 0.345, "the lines: each setting's mean", op, anchor="lm",
                  size=0.029)
        self.text(sc, 0.69, 0.387, "really %.2f points apart" % (100 * cmp["gap"]), op,
                  anchor="lm", strong=True, size=0.031)
        # one seed of each, drawn a few times
        k = int(np.clip((u - 0.1) / 0.4, 0, 0.999) * len(self.pairs))
        i, j = self.pairs[k]
        a, b = self.acc_a[i], self.acc_b[j]
        on = op * smoothstep(0.1, 0.16, u) * (1 - smoothstep(0.5, 0.56, u))
        for v, ink, y in ((a, NARROW, 0.415), (b, WIDE, 0.515)):
            self.disk(sc, self.P(at(float(np.clip(v, lo, hi))), y - 0.075, lift=1.8), 6.5, ink,
                      on, ring=DIGIT_INK, ring_w=1.8, glow=0.3, outline=0.0)
        wins = cmp["widths"][1] if b > a else cmp["widths"][0] if a > b else None
        right = wins == (cmp["widths"][1] if cmp["mean"][1] > cmp["mean"][0]
                         else cmp["widths"][0])
        self.text(sc, 0.69, 0.455, "one seed each: %.2f%%, %.2f%%" % (100 * a, 100 * b),
                  on, anchor="lm", size=0.029)
        self.text(sc, 0.69, 0.497, "a tie" if wins is None else "%d ahead%s"
                  % (wins, ": right" if right else ": the wrong winner"), on, anchor="lm",
                  ink=dark(GREEN if right else RED, 0.3), size=0.031)
        xs = np.asarray(self.ns, float)
        chart = self.chart(((0.12, 0.60), (0.685, 0.865)), xs, [0.0, 0.25], pad=0.0, xlog=True)
        self.axes(sc, chart, op, "seeds of each setting", "how often the worse setting wins",
                  xticks=[(v, str(int(v))) for v in (1, 2, 4, 8, 16, 32)],
                  yticks=[(v, "%d%%" % round(100 * v)) for v in (0.0, 0.1, 0.2)])
        chart.hline(sc, self.P, 0.05, op)
        self.trace(sc, chart, xs, self.share, RED, op, (u - 0.5) / 0.3)
        note = op * smoothstep(0.8, 0.9, u)
        self.lines(sc, 0.65, 0.715, [
            ("1 seed each: wrong %d%% of the time" % round(100 * self.share[0]),
             dark(RED, 0.3)),
            ("%d seeds each: under 5%% (dashed)" % cmp["seeds_for_under_5_percent"],
             dark(RED, 0.3)),
            "measured by redrawing from the",
            "%d runs of each" % len(self.acc_a)], note, size=0.030, step=0.044)


# ====================================================================== 117: seeds, sampling
SEED_INKS = (RED, BLUE, GREEN, AMBER, VIOLET, TEAL)
TREE_SETTING = 2                              # temperature 0.7: the tree drawn
TREE_DEPTH = 5
TEMP_INKS = (GREY, TEAL, GREEN, AMBER, RED)   # temperatures 0 to 1.3
ROWS_5 = (2, 3, 5, 6, 4, 7, 8)                # the settings compared in the last phase


class SeedsText(Chance):
    """GPT-2's next-token chances for a held-out prompt as a bar, and six seeds'
    uniform numbers landing on it; the six continuations as a tree; temperature
    reshaping the bar under the same numbers; agreement, distinct texts and GPT-2's
    own log-probability over twenty seeds and six prompts; and cutting the tail."""

    TITLE = "One prompt, many seeds"
    PHASES = (("one draw", 11.0), ("diverging", 12.0), ("temperature", 12.0),
              ("measured", 10.0), ("the tail", 11.0))
    BAR = (0.07, 0.93)

    def setup(self):
        d = self.d = S.sampling_data()
        s = self.study
        self.meta = s.metadata
        self.q = np.asarray(d["first_q"], float)             # (setting, 400 tokens)
        self.names = [str(w) for w in d["first_words"]]
        self.u = np.asarray(d["uniforms"], float)
        self.words = d["words"]                              # (setting, seed, token)
        self.texts = d["texts"]
        self.prompt = visible(str(d["prompt_text"][S.DRAWN_PROMPT]).split("\n")[-2]
                              + "\n" + str(d["prompt_text"][S.DRAWN_PROMPT]).split("\n")[-1])
        self.tokens = np.asarray(d["tokens"][S.DRAWN_PROMPT], int)
        self.agree = np.asarray(s.panels[0].data["ys"], float)
        self.temps = [float(v) for v in s.panels[1].data["x"]]
        self.distinct = np.asarray(s.panels[1].data["ys"], float)[0]
        self.logp = np.asarray(s.panels[2].data["values"], float)
        self.kinds, self.rare = np.asarray(s.panels[3].data["xy"], float).T
        self.kept = [int(v) for v in d["first_kept"]]
        # the tree of six seeds: rows in the order that keeps shared beginnings together
        six = self.tokens[TREE_SETTING, :6, :TREE_DEPTH]
        order = sorted(range(6), key=lambda b: tuple(six[b]))
        self.tree_row = {b: n for n, b in enumerate(order)}
        _, first = S.agreement(self.tokens[TREE_SETTING, :6])
        self.first6 = np.zeros((6, 6), int)
        self.first6[np.triu_indices(6, 1)] = first + 1
        self.first6 += self.first6.T
        self.first_tokens = len({int(self.tokens[3, b, 0]) for b in range(6)})

    def subtitle_for(self, k):
        m = self.meta
        pairs = self.first6[np.triu_indices(6, 1)]
        one = m["agreement_after_one_token"]
        return [
            "GPT-2 turns the prompt into a chance for each of its 50,257 tokens. The seed "
            "enters here: it gives a number from 0 to 1, and the number picks the token. "
            "One bar of chances, six seeds, %d different tokens" % self.first_tokens,
            "every later token takes the next number from the same seed. Once two texts "
            "part they do not meet again: here %d of the 15 pairs of seeds differ at the "
            "first token and all have parted by token %d"
            % (int((pairs == 1).sum()), int(pairs.max())),
            "temperature reshapes the chances before the number lands. At 0 the likeliest "
            "token takes the whole bar and every seed writes the same text; at 1.3 the "
            "unlikely tail swells and the same seeds scatter",
            "over %d seeds and %d held-out prompts, from temperature 0.4 to 1.3: pairs "
            "still together after one token %d%% → %d%%, distinct texts %.0f → %.0f of %d, "
            "GPT-2's own log-probability per token %.1f → %.1f"
            % (len(m["seeds"]), len(m["prompts"]), round(100 * one[1]), round(100 * one[4]),
               self.distinct[1], self.distinct[4], len(m["seeds"]), self.logp[1],
               self.logp[4]),
            "cutting the tail instead: sampling from the 40 likeliest tokens gives %.1f "
            "distinct texts of %d, yet only %d%% hold a token GPT-2 gave under 0.1%% "
            "(untouched: %d%%; smallest set holding 90%%: %d%%)"
            % (self.kinds[5], len(m["seeds"]), round(100 * self.rare[5]),
               round(100 * self.rare[3]), round(100 * self.rare[6])),
        ][k]

    def _bar(self, sc, q, fy, h, op, pins=6, pin_op=1.0, drop=0.03):
        x0, x1 = self.BAR
        at = self.chances(sc, q, x0, x1, fy, h, op, names=self.names)
        for b in range(pins):
            self.pin(sc, at(self.u[b, 0]), fy, h, SEED_INKS[b],
                     op * (pin_op if np.isscalar(pin_op) else pin_op[b]), drop=drop)
        return at

    # ------------------------------------------------------------------ phase 1
    def phase_1(self, sc, u, op):
        self.box(sc, 0.5, 0.335, 660, 34, PAPER, op, "… " + self.prompt, border=AXIS,
                 size=0.031)
        self.arrow(sc, (0.5, 0.372), (0.5, 0.398), VIOLET, op, smoothstep(0.02, 0.1, u), 2.6)
        self.box(sc, 0.5, 0.432, 100, 32, soft(VIOLET, 0.6), op, "GPT-2", border=VIOLET,
                 ink=dark(VIOLET, 0.5), size=0.032)
        self.arrow(sc, (0.5, 0.468), (0.5, 0.535), VIOLET, op, smoothstep(0.08, 0.18, u), 2.6)
        bar = op * smoothstep(0.14, 0.26, u)
        self.text(sc, 0.07, 0.498, "its chance for each next token", bar, anchor="lm",
                  strong=True, size=0.031)
        # the tail's block is the bar's empty track: pale on the light page, dark on the dark
        self.text(sc, 0.93, 0.498, "likeliest first; %s: thousands of unlikely ones"
                  % theme.word("pale", "dark"), bar, anchor="rm", size=0.029)
        pin_op = [smoothstep(0.3 + 0.08 * b, 0.36 + 0.08 * b, u) for b in range(6)]
        at = self._bar(sc, self.q[3], 0.585, 40, bar, pin_op=pin_op)
        for b in range(6):
            x, on = at(self.u[b, 0]), op * pin_op[b]
            self.text(sc, x, 0.70, "seed %d" % b, on, ink=dark(SEED_INKS[b], 0.3), size=0.030)
            self.text(sc, x, 0.74, "u = %.3f" % self.u[b, 0], on, size=0.029)
            self.box(sc, x, 0.796, 84, 30, soft(SEED_INKS[b], 0.5), on,
                     str(self.words[3, b, 0])[:9], ink=dark(SEED_INKS[b], 0.55),
                     border=SEED_INKS[b], size=0.031)
        self.text(sc, 0.5, 0.875, "the seed enters here: generator(seed) gives u, and the "
                  "token whose segment u lands in is the draw", op * smoothstep(0.75, 0.86, u),
                  strong=True, size=0.031)
        self.text(sc, 0.5, 0.918, "nothing else is random: the chances are the same for "
                  "every seed", op * smoothstep(0.8, 0.9, u), size=0.030)

    # ------------------------------------------------------------------ phase 2
    def phase_2(self, sc, u, op):
        temp = S.SETTINGS[TREE_SETTING][0]
        self.text(sc, 0.36, 0.315, "six seeds at temperature %.1f: each seed's numbers pick "
                  "its path" % temp, op, strong=True, size=0.032)
        six = self.tokens[TREE_SETTING, :6]
        rows = [0.405 + 0.085 * n for n in range(6)]
        root = (0.085, float(np.mean(rows)))
        self.box(sc, root[0], root[1], 62, 32, soft(DIGIT_INK, 0.75), op, "From", border=AXIS,
                 size=0.031)
        xs = [0.19 + 0.098 * t for t in range(TREE_DEPTH)]
        half = self.frac(41.0)[0]
        for t in range(TREE_DEPTH):
            on = op * smoothstep(0.04 + 0.13 * t, 0.12 + 0.13 * t, u)
            grow = smoothstep(0.02 + 0.13 * t, 0.1 + 0.13 * t, u)
            groups = {}
            for b in range(6):
                groups.setdefault(tuple(six[b, :t + 1]), []).append(b)
            for prefix, seeds in groups.items():
                y = float(np.mean([rows[self.tree_row[b]] for b in seeds]))
                if t == 0:
                    src = (root[0] + self.frac(33.0)[0], root[1])
                else:
                    parent = [b for b in range(6) if tuple(six[b, :t]) == prefix[:-1]]
                    src = (xs[t - 1] + half, float(np.mean([rows[self.tree_row[b]]
                                                            for b in parent])))
                one = len(seeds) == 1
                ink = SEED_INKS[seeds[0]] if one else DIGIT_INK
                self.arrow(sc, src, (xs[t] - half - 0.004, y), ink, op, grow,
                           2.2 if one else 3.4, head=False)
                word = str(self.words[TREE_SETTING, seeds[0], t])[:9]
                # text has a legibility floor, so a long token gets a wider box, not smaller type
                self.box(sc, xs[t], y, 80 if len(word) < 8 else 86, 30,
                         soft(ink, 0.5 if one else 0.8), on, word,
                         ink=dark(ink, 0.55) if one else None, border=ink, size=0.029)
        tag = op * smoothstep(0.6, 0.7, u)
        for b in range(6):
            y = rows[self.tree_row[b]]
            self.text(sc, xs[-1] + half + 0.012, y, "seed %d" % b, tag, anchor="lm",
                      ink=dark(SEED_INKS[b], 0.3), size=0.030)
        # where each pair first differs
        show_m = op * smoothstep(0.7, 0.82, u)
        cx, cy, side = 0.855, 0.66, 150.0
        self.text(sc, cx, 0.405, "the token at which", show_m, strong=True, size=0.030)
        self.text(sc, cx, 0.445, "each pair first differs", show_m, strong=True, size=0.030)
        cell = self.grid(sc, np.where(self.first6 > 0, 1.0 / np.maximum(self.first6, 1), 0.0),
                         cx, cy, side, side, show_m, vmax=1.6, hot=AMBER)
        step = self.frac(side / 6, side / 6)
        for a in range(6):
            ra, ka = self.tree_row[a], self.tree_row[a]
            self.disk(sc, self.P(cell(0, 0)[0] - step[0] * 0.5 - 0.014, cell(ra, 0)[1]), 5.0,
                      SEED_INKS[a], show_m, outline=0.1)
            self.disk(sc, self.P(cell(0, ka)[0], cell(0, 0)[1] - step[1] * 0.5 - 0.024), 5.0,
                      SEED_INKS[a], show_m, outline=0.1)
            for b in range(6):
                if a != b:
                    self.text(sc, *cell(ra, self.tree_row[b]), str(self.first6[a, b]), show_m,
                              size=0.029, strong=True)
        self.text(sc, cx, 0.875, "1: at the very first token", show_m, size=0.029)

    # ------------------------------------------------------------------ phase 3
    def phase_3(self, sc, u, op):
        n = S.N_TEMPS
        stage = int(min(u * n, n - 1))
        w = u * n - stage
        self.text(sc, 0.07, 0.325, "temperature", op, anchor="lm", strong=True, size=0.032)
        for a in range(n):
            on = a == stage
            self.box(sc, 0.27 + 0.085 * a, 0.325, 56, 30, soft(AMBER, 0.35) if on else PAPER,
                     op, "%.1f" % self.temps[a], border=AMBER if on else DIM, size=0.031,
                     glow=0.2 if on else 0.0)
        blend = smoothstep(0.0, 0.25, w) if stage else 1.0
        q = self.q[stage] * blend + self.q[max(stage - 1, 0)] * (1 - blend)
        self._bar(sc, q, 0.425, 36, op, drop=0.022)
        t = self.temps[stage]
        how = ("temperature 0: always the likeliest token" if t == 0 else
               "temperature 1: GPT-2's chances as they are" if t == 1 else
               "temperature %.1f: each chance raised to the power 1/%.1f, then rescaled"
               % (t, t))
        self.text(sc, 0.5, 0.525, how, op, size=0.031, strong=True)
        self.text(sc, 0.93, 0.325, "the six pins: the same six numbers", op, anchor="rm",
                  size=0.029)
        for b in range(5):
            y = 0.61 + 0.062 * b
            self.disk(sc, self.P(0.063, y), 6.0, SEED_INKS[b], op, outline=0.1)
            self.text(sc, 0.08, y, "seed %d" % b, op, anchor="lm", ink=dark(SEED_INKS[b], 0.3),
                      size=0.030)
            self.text(sc, 0.165, y, visible(self.texts[stage, b]).strip()[:74],
                      op * (smoothstep(0.1, 0.3, w) if stage else 1.0), anchor="lm",
                      size=0.030, strong=True)

    # ------------------------------------------------------------------ phase 4
    def phase_4(self, sc, u, op):
        m = self.meta
        n = S.N_TEMPS
        self.text(sc, 0.30, 0.318, "pairs of seeds still writing the same text", op,
                  strong=True, size=0.032)
        xs = np.arange(1, self.agree.shape[1] + 1)
        chart = self.chart(((0.10, 0.50), (0.40, 0.79)), xs, [0.0, 1.0], pad=0.0)
        self.axes(sc, chart, op, "tokens written", "",
                  xticks=[(v, str(v)) for v in (1, 3, 6, 9, 12)],
                  yticks=[(v, "%d%%" % round(100 * v)) for v in (0.0, 0.5, 1.0)])
        for a in range(n):
            self.trace(sc, chart, xs, self.agree[a], TEMP_INKS[a], op, (u - 0.04 * a) / 0.4,
                       width=2.6, r=3.2)
        self.text(sc, 0.30, 0.92, "%d seeds, %d held-out prompts, %d pairs each"
                  % (len(m["seeds"]), len(m["prompts"]), 190), op, size=0.029)
        blocks = (("distinct texts of %d" % len(m["seeds"]), self.distinct, 6.0, "%.1f", 0.375,
                   0.45),
                  ("GPT-2's log-probability per token", -self.logp, 14.0, "−%.2f", 0.655, 0.6))
        for name, values, unit, fmt, y0, t0 in blocks:
            on = op * smoothstep(t0, t0 + 0.1, u)
            self.text(sc, 0.60, y0, name, on, anchor="lm", strong=True, size=0.031)
            for a in range(n):
                y = y0 + 0.045 + 0.039 * a
                self.hbar(sc, 0.675, y, unit * values[a], TEMP_INKS[a], on,
                          label="T %.1f" % self.temps[a], value=fmt % values[a], thick=13,
                          u=smoothstep(t0, t0 + 0.25, u))

    # ------------------------------------------------------------------ phase 5
    def phase_5(self, sc, u, op):
        m = self.meta
        stage = 0 if u < 0.2 else 1 if u < 0.45 else 2
        setting = (3, 5, 6)[stage]
        x0, x1 = self.BAR
        p = self.q[3]
        cum = np.cumsum(p)
        heads = ("temperature 1, untouched: the %s block is the unlikely tail"
                 % theme.word("pale", "dark"),
                 "top 40: only the 40 likeliest tokens can be drawn",
                 "top-p 0.9: the likeliest tokens holding 90%% of the chance (%s here)"
                 % f"{self.kept[6]:,}")
        self.text(sc, 0.5, 0.318, heads[stage], op, strong=True, size=0.032)
        if stage == 1:
            w = smoothstep(0.2, 0.3, u)
            q = p * (1 - w) + self.q[5] * w
        else:
            q = p
        at = self.chances(sc, q, x0, x1, 0.40, 34, op, names=self.names)
        if stage == 2:
            cut = at(0.9)
            self.rect(sc, self.P((cut + x1) / 2, 0.40), self.span(x1 - cut)[0], 34, PAPER,
                      0.82 * op * smoothstep(0.47, 0.55, u), radius=2, lift=0.9)
            self.polyline(sc, [self.P(cut, 0.355, lift=1.2), self.P(cut, 0.445, lift=1.2)], 2.6,
                          DIGIT_INK, op)
        elif stage == 0:
            cut = at(cum[39])
            self.polyline(sc, [self.P(cut, 0.355, lift=1.2), self.P(cut, 0.445, lift=1.2)], 2.6,
                          DIGIT_INK, op * smoothstep(0.08, 0.14, u))
        for b in range(6):
            self.pin(sc, at(self.u[b, 0]), 0.40, 34, SEED_INKS[b], op, drop=0.02)
        # what each setting gives, over all seeds and prompts
        cols = (("distinct texts of %d" % len(m["seeds"]), self.kinds, 5.2, "%.1f", 0.27),
                ("texts with a token under 0.1%", 100 * self.rare, 1.25, "%.0f%%", 0.495),
                ("log-probability per token", -self.logp, 15.0, "−%.2f", 0.755))
        for name, _, _, _, x in cols:
            self.text(sc, x, 0.52, name, op, anchor="lm", strong=True, size=0.029)
        for n, a in enumerate(ROWS_5):
            y = 0.572 + 0.049 * n
            _, k, top_p = S.SETTINGS[a]
            ink = GREEN if k else VIOLET if top_p else BLUE
            on = op * (1.0 if a == setting else 0.75)
            self.text(sc, 0.052, y, S.setting_name(S.SETTINGS[a]), on, anchor="lm",
                      ink=dark(ink, 0.3), size=0.030, strong=a == setting)
            for _, values, unit, fmt, x in cols:
                self.hbar(sc, x, y, unit * values[a], ink, on, value=fmt % values[a],
                          thick=13, u=smoothstep(0.05, 0.3, u))
        self.text(sc, 0.5, 0.925, "the same seed and settings give the same text, token for "
                  "token, every time", op * smoothstep(0.8, 0.9, u), strong=True, size=0.030)


ANIMATIONS = {
    "markov-chains": MarkovChains,
    "seeds-and-training-outcomes": SeedsTraining,
    "seeds-and-sampled-text": SeedsText,
}

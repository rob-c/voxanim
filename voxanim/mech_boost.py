"""Boosting, and the algorithm XGBoost uses: animations 165-169 of the "next" catalogue
(see :mod:`voxanim.mech`).

* :class:`AdaBoost`: stumps fitted one after another to re-weighted events;
* :class:`BoostingSettings`: gradient boosting for classification, its learning rate,
  row subsampling, depth and early stopping;
* :class:`SplitGain`: the slope g and curvature h of the loss, a leaf's value and a
  split's gain;
* :class:`Histograms`: histogram split finding, the subtraction trick, missing values
  and the two ways to grow a tree;
* :class:`OneRound`: one boosting round end to end, the knobs, and the comparison with
  the other tree methods.

They draw the experiments of :mod:`voxanim.study_boost`. No physics dataset is used:
real handwritten digits stand in for events (4s for signal, 9s for background), which
the first phase of each says. Every phase carries, bottom right, what kind of evidence
it shows (one of :data:`TAGS`). The XGBoost animations draw this repository's own numpy
implementation of the published algorithm, and say so on screen (:data:`OWN`).
"""

from __future__ import annotations

import numpy as np

from . import study_boost as S
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
    dark,
    heat,
    soft,
)
from .mech_phys_apps import GREY, Phys

MEASURED = "measured on MNIST, held-out"
TRAINING = "measured on MNIST, training"
EXACT = "exact arithmetic on real events"
TIMED = "timed on this machine: it varies by machine"
TAGS = (MEASURED, TRAINING, EXACT, TIMED)
STAND_IN = "no physics data: handwritten 4s stand in for signal, 9s for background"
OWN = S.OWN
BIG, HEAD, SMALL = 0.036, 0.033, 0.030
SIG, BKG = BLUE, RED


def f2(v):
    """A number with two decimals and a proper minus sign."""
    return ("%.2f" % float(v)).replace("-", "−")


def f3(v):
    return ("%.3f" % float(v)).replace("-", "−")


def signed(v, digits=2):
    return ("%+.*f" % (digits, float(v))).replace("-", "−")


def pct(v, digits=1):
    return "%.*f%%" % (digits, 100.0 * float(v))


def stage(u, n):
    """Which of ``n`` equal parts of a phase ``u`` is in, and how far through it."""
    s = min(int(u * n), n - 1)
    return s, u * n - s


def tree_layout(left, right, root=0):
    """Where to draw each node of a tree: ``{node: (x, level)}`` with the leaves evenly
    spaced over 0..1 in order and every other node midway between its children."""
    out, leaves = {}, []

    def walk(k, level):
        if left[k] < 0:
            leaves.append(k)
            out[k] = (float(len(leaves) - 1), level)
        else:
            walk(int(left[k]), level + 1)
            walk(int(right[k]), level + 1)
            out[k] = ((out[int(left[k])][0] + out[int(right[k])][0]) / 2.0, level)
    walk(root, 0)
    n = len(leaves)
    return {k: ((x + 0.5) / n, level) for k, (x, level) in out.items()}


class Boost(Phys):
    """What the five animations share: the signal and background colours, the plane of
    two cells, a region drawn under it, a tree drawn as a tree, and the read-out."""

    def stand_in(self, sc, op):
        self.text(sc, 0.05, 0.925, STAND_IN, op, anchor="lm", ink=dark(OCHRE, 0.35))

    def own(self, sc, op, fx=0.05, fy=0.925):
        """The line the XGBoost animations carry: whose implementation this is."""
        self.text(sc, fx, fy, OWN, op, anchor="lm", ink=dark(OCHRE, 0.35))

    def ink_of(self, y):
        return SIG if y > 0.5 else BKG

    def plane(self, sc, box, lo, hi, op, names):
        ch = self.pane(box, (lo[0], hi[0]), (lo[1], hi[1]))
        ticks = [(v, "%g" % v) for v in (0.0, 0.25, 0.5, 0.75, 1.0)]
        self.axes(sc, ch, op, "cell %s: how much ink" % names[0], "cell %s" % names[1],
                  [t for t in ticks if t[0] <= hi[0]], [t for t in ticks if t[0] <= hi[1]],
                  grid=False)
        return ch

    def region(self, sc, ch, values, op, scale=None, strength=0.62):
        """``values`` (rows from the top) as colour under the chart: blue where the
        answer is signal, red where it is background, deeper the surer."""
        if op <= 0.01:
            return
        v = np.asarray(values, float)
        scale = scale or max(float(np.abs(v).max()), 1e-12)
        tex = PAPER + (heat(np.clip(v / scale, -1, 1), SIG, BKG) - PAPER) * strength
        w, h = self.span(ch.x1 - ch.x0, ch.y1 - ch.y0)
        draw_image(self, sc, self.P((ch.x0 + ch.x1) / 2, (ch.y0 + ch.y1) / 2), tex, w, h, op,
                   grid=0, lift=0.2)

    def shade(self, sc, ch, x0, x1, y0, y1, ink, op, fill=0.16):
        """A rectangle of the chart, in chart units, tinted."""
        a, b = ch.fx(x0), ch.fx(x1)
        c, d = ch.fy(y1), ch.fy(y0)
        w, h = self.span(b - a, d - c)
        if w > 0.5 and h > 0.5:
            self.rect(sc, self.P((a + b) / 2, (c + d) / 2), w, h, ink, op, radius=1.0,
                      lift=0.25, fill_op=fill)

    def readout(self, sc, img, fx, fy, size, op, marks=(), grid=True):
        """A digit with its 4 x 4 read-out drawn over it; ``marks``: ``(cell, ink)``
        cells outlined."""
        self.digit(sc, img, fx, fy, size, op)
        dx, dy = self.frac(size, size)
        if grid:
            for k in range(1, 4):
                t = -0.5 + k / 4.0
                self.polyline(sc, [self.P(fx + t * dx, fy - dy / 2, 0.6),
                                   self.P(fx + t * dx, fy + dy / 2, 0.6)], 1.0, AXIS, 0.6 * op)
                self.polyline(sc, [self.P(fx - dx / 2, fy + t * dy, 0.6),
                                   self.P(fx + dx / 2, fy + t * dy, 0.6)], 1.0, AXIS, 0.6 * op)
        for cell, ink in marks:
            r, c = divmod(int(cell), 4)
            self.rect(sc, self.P(fx + (c - 1.5) * dx / 4, fy + (r - 1.5) * dy / 4),
                      size / 4, size / 4, ink, op, border=ink, border_w=2.6, radius=2,
                      lift=0.9, fill_op=0.0)

    def tree(self, sc, left, right, box, op, text, fill, show=None, w=96.0, h=30.0,
             leaf_w=None, size=0.029, border=None, edge_op=None, sides=True):
        """A tree drawn as a tree in the layout box ``((x0, x1), (top, bottom))``.
        ``text(node)``, ``fill(node)``, ``show(node)`` (0..1), ``border(node)``. Returns
        ``{node: (fx, fy)}``."""
        at = tree_layout(left, right)
        levels = max(lv for _, lv in at.values())
        (x0, x1), (y0, y1) = box
        pos = {k: (x0 + (x1 - x0) * x, y0 + (y1 - y0) * (lv / max(levels, 1)))
               for k, (x, lv) in at.items()}
        gap = self.frac(0, h / 2 + 2)[1]
        for k, (fx, fy) in pos.items():
            o = op * (1.0 if show is None else float(show(k)))
            if o <= 0.01:
                continue
            leaf = left[k] < 0
            for side, child in enumerate(() if leaf else (int(left[k]), int(right[k]))):
                oc = op * (1.0 if show is None else float(show(child)))
                oc = oc if edge_op is None else oc * edge_op(k)
                cx, cy = pos[child]
                self.arrow(sc, (fx + (cx - fx) * 0.12, fy + gap), (cx, cy - gap), AXIS, oc,
                           width=1.8, head=False)
                if sides and k == 0:
                    self.text(sc, (fx + cx) / 2 + (-0.042 if side == 0 else 0.042),
                              (fy + cy) / 2 - 0.016, "yes" if side == 0 else "no", oc,
                              size=size)
            self.box(sc, fx, fy, (leaf_w or w) if leaf else w, h, fill(k), o, text(k),
                     border=AXIS if border is None else border(k), size=size)
        return pos

    def columns(self, sc, ch, xs, values, ink, op, width, grow=1.0, fill=0.9, base=0.0):
        """Bars up (or down, when negative) from ``base`` at the chart's ``xs``, each
        ``width`` px wide."""
        if op <= 0.01:
            return
        for x, v in zip(xs, values):
            a, b = ch.fy(base), ch.fy(base + float(v) * grow)
            h = self.span(0, abs(a - b))[1]
            if h < 0.6:
                continue
            self.rect(sc, self.P(ch.fx(float(x)), (a + b) / 2), width, h, ink, op, radius=1.5,
                      lift=0.4, fill_op=fill)

    def strip(self, sc, ch, x, values, ink, op, spread=0.16, r=3.4):
        """One category of a dot plot: a dot per seed around ``x`` and a tick at their
        mean."""
        values = np.asarray(values, float)
        n = len(values)
        for k, v in enumerate(values):
            dx = spread * ((k + 0.5) / n - 0.5) * 2
            self.disk(sc, self.P(ch.fx(x + dx), ch.fy(float(v))), r, ink, op, lift=1.1)
        m = float(values.mean())
        self.polyline(sc, [self.P(ch.fx(x - 0.3), ch.fy(m), 0.9),
                           self.P(ch.fx(x + 0.3), ch.fy(m), 0.9)], 3.0, dark(ink, 0.25), op)

    def cross(self, sc, fx, fy, r, ink, op, width=2.6):
        """A cross of half-width ``r`` px at ``(fx, fy)``."""
        dx, dy = self.frac(r, r)
        for sx in (1, -1):
            self.polyline(sc, [self.P(fx - dx, fy - sx * dy, 1.6), self.P(fx + dx, fy + sx * dy,
                                                                          1.6)], width, ink, op)


# ====================================================================== 165: AdaBoost
class AdaBoost(Boost):
    """Two cells of the digits as a plane of events; one stump and the events it gets
    wrong; their weights grow and the next stump is fitted to the weighted events; the
    weighted vote of 1, 2, 3, 10 and 100 stumps as a partition of the plane; then all
    sixteen cells: error against rounds, the margins, and the heaviest events."""

    TITLE = "AdaBoost: weights on the mistakes"
    PHASES = (("events", 8.0), ("a stump", 10.0), ("weights", 12.0), ("the vote", 10.0),
              ("16 cells", 10.0), ("heavy", 9.0))
    STUMP = (AMBER, VIOLET, TEAL)
    TRAIN, HELD = VIOLET, GREEN
    BOX = ((0.09, 0.47), (0.41, 0.80))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.xy, self.cls = np.asarray(p0["xy"], float), np.asarray(p0["classes"], float)
        self.n = len(self.cls)
        self.lo, self.hi = np.asarray(p0["lo"], float), np.asarray(p0["hi"], float)
        self.names = m["two"]["cells"]
        self.pair = [("ABCD".index(c[0])) * 4 + int(c[1]) - 1 for c in self.names]
        self.first = m["two"]["first"]
        self.feature = np.asarray(p0["feature"])
        self.threshold = np.asarray(p0["threshold"], float)
        self.sign = np.asarray(p0["sign"], float)
        self.weights = np.asarray(p0["weights"], float) * self.n       # 1 = the average
        self.wrong = np.asarray(p0["wrong"], bool)
        self.planes = np.asarray(p0["planes"], float)
        self.images = np.asarray(p0["images"], float)
        self.image_cells = np.asarray(p0["image_cells"], float)
        self.two_train, self.two_held = m["two"]["train_error"], m["two"]["held_error"]
        self.shown = m["two"]["shown"]
        self.rounds = np.asarray(p1["x"], float)
        self.train, self.held = np.asarray(p1["ys"], float)
        self.per_seed = np.asarray(p1["per_seed"], float)
        self.top = np.asarray(p1["top_share"], float)
        self.margin_counts = np.asarray(p2["counts"], float)[0]
        self.margin_edges = np.r_[-1.0, np.asarray(p2["x"], float)]
        self.heavy = np.asarray(p3["values"], float)
        self.heavy_w = np.asarray(p3["weights"], float)
        self.heavy_label = list(p3["labels"])

    def subtitle_for(self, k):
        m = self.meta
        two, f = m["two"], self.first
        mg, w = m["margin"], m["weights"]
        return [
            "handwritten 4s stand in for signal events and 9s for background, as in an "
            "analysis. Each is read out as 16 cells; two of them, %s and %s, make a "
            "plane on which %d training events can be drawn"
            % (self.names[0], self.names[1], self.n),
            "a stump is a tree with one question. The best one gets %d of the %d events "
            "wrong: err = %.3f. Its say in the final vote is α = ½ ln((1 − err) / err) = "
            "%.2f; a stump no better than a coin would get 0"
            % (f[0]["wrong"], self.n, f[0]["err"], f[0]["alpha"]),
            "the events it got wrong are made heavier (%.2f times the average), the "
            "others lighter (%.2f). The next stump is fitted to the weighted events: "
            "its weighted err is %.3f, so α = %.2f. And so on"
            % (two["heaviest_after_first"], two["lightest_after_first"], f[1]["err"],
               f[1]["alpha"]),
            "the answer is the sign of the sum of every stump's α times its answer. On "
            "these two cells the held-out error is %s with 1 stump, %s with 10, %s with "
            "100: the partition gets finer, but no better"
            % (pct(two["held_error"][0]), pct(two["held_error"][3]),
               pct(two["held_error"][4])),
            "all 16 cells, %d training events, %d seeds: after %d stumps the training "
            "error is %s, the held-out error %s. The margin, how far the vote is from a "
            "tie, stays small: half the events are within %.2f of one"
            % (2 * S.TRAIN, len(m["seeds"]), m["rounds"], pct(m["train_error"]["last"]),
               pct(m["held_error"]["last"]), mg["median"][-1]),
            "the weight piles up on events the stumps keep getting wrong: the heaviest "
            "1%% carry %s of it (%s at the start), the heaviest of all %.0f to %.0f "
            "times the average. A mislabelled event would end up here too"
            % (pct(w["top_share_end"], 0), pct(w["top_share_start"], 0),
               min(w["heaviest"]), max(w["heaviest"])),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _plane(self, sc, op, box=None):
        return self.plane(sc, box or self.BOX, self.lo, self.hi, op, self.names)

    def _events(self, sc, ch, weights, op, ring=None, base=3.3):
        for i in range(self.n):
            r = base * float(np.sqrt(weights[i]))
            ringed = ring is not None and ring[i] > 0.01
            self.disk(sc, self.P(ch.fx(self.xy[i, 0]), ch.fy(self.xy[i, 1])), r,
                      self.ink_of(self.cls[i]), op, ring=DIGIT_INK if ringed else None,
                      ring_w=1.6, ring_op=float(ring[i]) if ringed else 1.0, lift=1.0)

    def _stump_line(self, sc, ch, k, op, tint=True, grow=1.0):
        """Stump ``k``'s threshold on the plane, and the answer on each side of it."""
        ink, t = self.STUMP[k % 3], float(self.threshold[k])
        along_x = self.feature[k] == 0
        above = SIG if self.sign[k] > 0 else BKG
        below = BKG if self.sign[k] > 0 else SIG
        lo, hi = self.lo, self.hi
        if along_x:
            a, b = (ch.fx(t), ch.y1), (ch.fx(t), ch.y1 + (ch.y0 - ch.y1) * grow)
            if tint:
                self.shade(sc, ch, lo[0], t, lo[1], hi[1], below, op)
                self.shade(sc, ch, t, hi[0], lo[1], hi[1], above, op)
        else:
            a, b = (ch.x0, ch.fy(t)), (ch.x0 + (ch.x1 - ch.x0) * grow, ch.fy(t))
            if tint:
                self.shade(sc, ch, lo[0], hi[0], lo[1], t, below, op)
                self.shade(sc, ch, lo[0], hi[0], t, hi[1], above, op)
        self.polyline(sc, [self.P(*a, 1.3), self.P(*b, 1.3)], 3.2, ink, op, bias=0.3)

    def _stump_tree(self, sc, k, fx, fy, op, grow=1.0):
        f = self.first[k]
        ink = self.STUMP[k % 3]
        yes, no = ("background", "signal") if f["signal_above"] else ("signal", "background")
        left, right = np.array([1, -1, -1]), np.array([2, -1, -1])
        text = {0: "%s < %s?" % (f["cell"], f2(f["threshold"])), 1: yes, 2: no}
        fills = {0: soft(ink, 0.5), 1: soft(SIG if yes == "signal" else BKG, 0.55),
                 2: soft(SIG if no == "signal" else BKG, 0.55)}
        self.tree(sc, left, right, ((fx - 0.13, fx + 0.13), (fy, fy + 0.13)), op, text.get,
                  fills.get, show=lambda n: 1.0 if n == 0 else grow, w=124, h=32, leaf_w=112,
                  size=SMALL, border=lambda n: ink if n == 0 else AXIS)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        self.text(sc, 0.05, 0.315, "two kinds of event, read out as 4 × 4 cells", op,
                  anchor="lm", strong=True, size=HEAD)
        marks = [(self.pair[0], self.STUMP[1]), (self.pair[1], self.STUMP[0])]
        for n, (i, name, ink) in enumerate(((0, "signal: a 4", SIG), (2, "background: a 9",
                                                                     BKG))):
            fx = 0.125 + 0.17 * n
            self.readout(sc, self.images[i], fx, 0.51, 100, op,
                         marks if u > 0.25 else (), grid=u > 0.1)
            self.text(sc, fx, 0.375, name, op, ink=dark(ink, 0.25), size=SMALL)
            vals = self.image_cells[i]
            show = op * smoothstep(0.3, 0.42, u)
            self.text(sc, fx, 0.655, "%s = %s" % (self.names[0], f2(vals[0])), show,
                      ink=dark(self.STUMP[1], 0.3), size=SMALL)
            self.text(sc, fx, 0.70, "%s = %s" % (self.names[1], f2(vals[1])), show,
                      ink=dark(self.STUMP[0], 0.3), size=SMALL)
        self.lines(sc, 0.05, 0.775, ["a cell's reading: the mean ink of its 7 × 7 pixels",
                                     "every event starts with the same weight, 1/%d"
                                     % self.n], op * smoothstep(0.45, 0.6, u), size=SMALL,
                   step=0.044)
        ch = self._plane(sc, op, ((0.55, 0.93), (0.41, 0.80)))
        count = int(self.n * smoothstep(0.35, 0.9, u))
        order = np.argsort(np.arange(self.n) % (self.n // 2), kind="stable")   # 4, 9, 4, 9 ...
        seen = np.zeros(self.n)
        seen[order[:max(count, 4)]] = 1.0
        for i in np.flatnonzero(seen):
            self.disk(sc, self.P(ch.fx(self.xy[i, 0]), ch.fy(self.xy[i, 1])), 3.3,
                      self.ink_of(self.cls[i]), op, lift=1.0)
        self.key(sc, 0.80, 0.335, [("signal (4s)", SIG), ("background (9s)", BKG)], op,
                 step=0.042)
        self.stand_in(sc, op)
        self.tag(sc, TRAINING, op)

    def phase_2(self, sc, u, op):
        f = self.first[0]
        ch = self._plane(sc, op)
        line = smoothstep(0.08, 0.25, u)
        self._stump_line(sc, ch, 0, op * line, grow=line)
        ring = self.wrong[0] * smoothstep(0.38, 0.5, u)
        self._events(sc, ch, self.weights[0], op, ring)
        self.text(sc, 0.05, 0.315, "one question splits the plane in two", op, anchor="lm",
                  strong=True, size=HEAD)
        self._stump_tree(sc, 0, 0.74, 0.36, op * line, smoothstep(0.2, 0.32, u))
        y = 0.60
        a = op * smoothstep(0.4, 0.52, u)
        self.text(sc, 0.56, y, "ringed: the %d events it gets wrong" % f["wrong"], a,
                  anchor="lm", size=SMALL)
        self.text(sc, 0.56, y + 0.055, "err = %d × 1/%d = %s" % (f["wrong"], self.n,
                                                                f3(f["err"])), a,
                  anchor="lm", strong=True, size=HEAD)
        b = op * smoothstep(0.6, 0.72, u)
        self.text(sc, 0.56, y + 0.125, "its say:  α = ½ ln((1 − err) / err)", b, anchor="lm",
                  size=SMALL)
        self.text(sc, 0.56, y + 0.18, "= ½ ln(%s / %s) = %s" % (f3(1 - f["err"]),
                                                                f3(f["err"]),
                                                                f2(f["alpha"])), b,
                  anchor="lm", ink=dark(self.STUMP[0], 0.35), size=HEAD)
        self.tag(sc, EXACT, op)

    def phase_3(self, sc, u, op):
        ch = self._plane(sc, op)
        # three steps: weights after stump 1, stump 2 fitted, weights after stump 2,
        # stump 3 fitted
        w1 = smoothstep(0.06, 0.24, u)
        s2 = smoothstep(0.34, 0.46, u)
        w2 = smoothstep(0.56, 0.70, u)
        s3 = smoothstep(0.78, 0.88, u)
        weights = (self.weights[0] + (self.weights[1] - self.weights[0]) * w1
                   + (self.weights[2] - self.weights[1]) * w2)
        self._stump_line(sc, ch, 0, op * (1 - s2), tint=False)
        self._stump_line(sc, ch, 1, op * s2 * (1 - s3), tint=True, grow=s2)
        self._stump_line(sc, ch, 2, op * s3, tint=True, grow=s3)
        ring = self.wrong[0] * (1 - s2) + self.wrong[1] * s2 * (1 - s3) + self.wrong[2] * s3
        self._events(sc, ch, weights, op, ring)
        head = ("stump 1's mistakes grow, the rest shrink" if u < 0.34 else
                "stump 2 is fitted to the weighted events" if u < 0.56 else
                "its mistakes grow in turn" if u < 0.78 else "and stump 3 is fitted")
        self.text(sc, 0.05, 0.315, head, op, anchor="lm", strong=True, size=HEAD)
        e = np.exp(self.first[0]["alpha"])
        self.lines(sc, 0.55, 0.385, [
            "an event's size is its weight",
            "wrong: weight × e^α = × %s" % f2(e),
            "right: weight × e^−α = × %s" % f2(1 / e),
            "then all rescaled to sum to 1"], op * w1, size=SMALL, step=0.043)
        for k, show in ((0, 1.0), (1, s2), (2, s3)):
            f = self.first[k]
            y = 0.60 + 0.075 * k
            ink = self.STUMP[k]
            o = op * show
            self.box(sc, 0.59, y, 74, 30, soft(ink, 0.45), o, "stump %d" % (k + 1), border=ink,
                     size=SMALL)
            self.text(sc, 0.64, y, "%s %s %s: signal" % (f["cell"], ">" if f["signal_above"]
                                                         else "<", f2(f["threshold"])), o,
                      anchor="lm", size=SMALL)
            self.text(sc, 0.95, y, "err %s  α %s" % (f3(f["err"]), f2(f["alpha"])), o,
                      anchor="rm", ink=dark(ink, 0.35), size=SMALL)
        self.text(sc, 0.55, 0.835, "err: the weight of the events a stump gets wrong",
                  op * s2, anchor="lm", size=SMALL)
        self.tag(sc, EXACT, op)

    def phase_4(self, sc, u, op):
        n = len(self.shown)
        s, w = stage(smoothstep(0.02, 0.92, u), n)
        ch = self._plane(sc, op)
        scale = 0.6
        norm = [float(np.abs(p).max()) for p in self.planes]
        self.region(sc, ch, self.planes[s] / norm[s], op, scale)
        if s + 1 < n:
            self.region(sc, ch, self.planes[s + 1] / norm[s + 1],
                        op * smoothstep(0.75, 1.0, w), scale)
        for i in range(self.n):
            self.disk(sc, self.P(ch.fx(self.xy[i, 0]), ch.fy(self.xy[i, 1])), 2.4,
                      self.ink_of(self.cls[i]), op, lift=1.0)
        self.text(sc, 0.05, 0.315, "the weighted vote of %d stump%s" % (
            self.shown[s], "" if self.shown[s] == 1 else "s"), op, anchor="lm", strong=True,
            size=BIG)
        self.text(sc, 0.56, 0.385, "answer = sign of  Σ α × (stump's answer)", op, anchor="lm",
                  size=SMALL)
        self.text(sc, 0.665, 0.455, "stumps", op, anchor="rm", size=SMALL)
        self.text(sc, 0.76, 0.455, "training", op, ink=dark(self.TRAIN, 0.3), size=SMALL)
        self.text(sc, 0.88, 0.455, "held-out", op, ink=dark(self.HELD, 0.3), size=SMALL)
        for k in range(n):
            y = 0.515 + 0.06 * k
            o = op * (1.0 if k <= s else 0.0)
            if k == s:
                self.rect(sc, self.P(0.755, y), 330, 30, AMBER, o, radius=5, lift=0.2,
                          fill_op=0.25)
            self.text(sc, 0.665, y, "%d" % self.shown[k], o, anchor="rm", strong=True,
                      size=HEAD)
            self.text(sc, 0.76, y, pct(self.two_train[k]), o, ink=dark(self.TRAIN, 0.3),
                      size=HEAD)
            self.text(sc, 0.88, y, pct(self.two_held[k]), o, ink=dark(self.HELD, 0.3),
                      size=HEAD)
        self.text(sc, 0.56, 0.845, "error: the share of events on the wrong side", op,
                  anchor="lm", size=SMALL)
        self.tag(sc, MEASURED, op)

    def phase_5(self, sc, u, op):
        go = smoothstep(0.05, 0.55, u)
        last = self.rounds[-1]
        ch = self.pane(((0.09, 0.45), (0.41, 0.79)), (1, last), (0.10, 0.34), xlog=True)
        self.axes(sc, ch, op, "stumps", "error",
                  [(v, "%d" % v) for v in (1, 10, 100, int(last))],
                  [(v, "%d%%" % round(100 * v)) for v in (0.1, 0.2, 0.3)])
        x = self.rounds[1:]
        for seed in range(self.per_seed.shape[1]):
            self.path(sc, ch, x, self.per_seed[0, seed, 1:], soft(self.TRAIN, 0.45), op, go,
                      width=1.2)
            self.path(sc, ch, x, self.per_seed[1, seed, 1:], soft(self.HELD, 0.45), op, go,
                      width=1.2)
        self.path(sc, ch, x, self.train[1:], self.TRAIN, op, go, width=3.0)
        self.path(sc, ch, x, self.held[1:], self.HELD, op, go, width=3.0, dot=True)
        self.text(sc, 0.05, 0.315, "all 16 cells: error against stumps", op, anchor="lm",
                  strong=True, size=HEAD)
        self.key(sc, 0.27, 0.44, [("held-out %s" % pct(self.held[-1]), self.HELD),
                                  ("training %s" % pct(self.train[-1]), self.TRAIN)], op,
                 step=0.044, size=SMALL)
        self.text(sc, 0.05, 0.925, "thin lines: the %d seeds" % self.per_seed.shape[1], op,
                  anchor="lm", size=SMALL)
        self.text(sc, 0.57, 0.315, "margin of training events: label × vote / Σ α", op,
                  anchor="lm", strong=True, size=0.031)
        at = self.meta["margin"]["at"]
        rows = (1, 2, 3)
        for n, r in enumerate(rows):
            top = 0.375 + 0.15 * n
            c = self.margin_counts[r] / self.margin_counts[r].sum()
            mc = self.pane(((0.60, 0.91), (top, top + 0.115)), (-1, 1), (0, 0.16))
            self.axes(sc, mc, op, "", "", [(v, "%g" % v) for v in (-1, 0, 1)] if n == 2
                      else [], [], grid=False)
            grow = smoothstep(0.25 + 0.15 * n, 0.45 + 0.15 * n, u)
            edges = self.margin_edges
            neg = edges[1:] <= 1e-9
            self.bars(sc, mc, edges, np.where(neg, c, 0.0), BKG, op, fill=0.85, grow=grow)
            self.bars(sc, mc, edges, np.where(neg, 0.0, c), self.TRAIN, op, fill=0.85,
                      grow=grow)
            self.vline(sc, mc, 0.0, DIGIT_INK, 0.7 * op, width=1.4)
            self.text(sc, 0.925, top + 0.03, "%d" % at[r], op, anchor="lm", strong=True,
                      size=SMALL)
        self.text(sc, 0.925, 0.36, "stumps", op, anchor="lm", size=SMALL)
        self.text(sc, 0.60, 0.875, "wrong", op, anchor="lm", ink=dark(BKG, 0.25), size=SMALL)
        self.text(sc, 0.91, 0.875, "right, by this much", op, anchor="rm",
                  ink=dark(self.TRAIN, 0.3), size=SMALL)
        self.tag(sc, MEASURED, op)

    def phase_6(self, sc, u, op):
        go = smoothstep(0.05, 0.5, u)
        last = self.rounds[-1]
        ch = self.pane(((0.09, 0.43), (0.41, 0.78)), (0, last), (0.0, 0.11))
        self.axes(sc, ch, op, "stumps", "share of all weight on the heaviest 1%",
                  [(v, "%d" % v) for v in (0, 100, 200, 300, 400) if v <= last],
                  [(v, "%d%%" % round(100 * v)) for v in (0.0, 0.05, 0.10)], grid=False)
        for seed in range(len(self.top)):
            self.path(sc, ch, self.rounds, self.top[seed], soft(AMBER, 0.4), op, go, width=1.2)
        self.path(sc, ch, self.rounds, self.top.mean(0), AMBER, op, go, width=3.2, dot=True)
        self.hline(sc, ch, 0.01, GREY, op)
        self.text(sc, 0.42, ch.fy(0.01) - 0.022, "equal weights: 1%", op, anchor="rm",
                  size=SMALL)
        self.text(sc, 0.05, 0.315, "where the weight goes: the 8 heaviest training events "
                  "after %d stumps" % int(last), op, anchor="lm", strong=True, size=HEAD)
        for k in range(len(self.heavy)):
            r, c = divmod(k, 4)
            fx, fy = 0.575 + 0.11 * c, 0.46 + 0.215 * r
            show = op * smoothstep(0.3 + 0.05 * k, 0.4 + 0.05 * k, u)
            ink = SIG if self.heavy_label[k] == str(S.SIGNAL) else BKG
            self.digit(sc, self.heavy[k], fx, fy, 78, show, frame=ink)
            self.text(sc, fx, fy + 0.105, "a %s, × %.0f" % (self.heavy_label[k],
                                                           self.heavy_w[k]), show,
                      ink=dark(ink, 0.25), size=SMALL)
        self.text(sc, 0.05, 0.925, "frame: the true class; ×: its weight over the average",
                  op * smoothstep(0.5, 0.6, u), anchor="lm", size=SMALL)
        self.tag(sc, TRAINING, op)


# ====================================================================== 166: settings
def question(cell, threshold):
    """A node's question: is the reading of ``cell`` below ``threshold``?"""
    t = float(threshold)
    return "%s < %s?" % (cell, ("%.3f" if t < 0.01 else "%.2f") % t)


class BoostingSettings(Boost):
    """Events on the sigmoid of their score, each pulled towards its label by y - p; a
    small tree fitted to the pulls; the step it makes; then held-out loss against trees
    for four learning rates, the share of rows and the depth, and early stopping."""

    TITLE = "Gradient boosting: the settings"
    PHASES = (("pulls", 9.0), ("a tree", 10.0), ("step", 8.0), ("rate", 10.0),
              ("rows, depth", 10.0), ("stopping", 10.0))
    TREE, HELD, TRAIN, VALID = AMBER, GREEN, VIOLET, MAGENTA
    RATE_INKS = (RED, AMBER, BLUE, TEAL)
    BOX = ((0.09, 0.52), (0.405, 0.80))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.before = np.asarray(p0["xy"], float)[:, 0]
        self.cls = np.asarray(p0["classes"], float)
        self.after = np.asarray(p0["after"], float)
        self.leaf = np.asarray(p0["leaf"], int)
        self.t = m["one"]["tree"]
        self.pull = m["one"]["mean_pull"]
        self.rate = m["one"]["rate"]
        self.reach = float(np.ceil(max(np.abs(self.before).max(), np.abs(self.after).max())
                                   * 2 + 0.5) / 2)
        self.rounds = np.asarray(p1["x"], float)
        self.curves = np.asarray(p1["ys"], float)
        self.low, self.low_at = np.asarray(p1["lowest"]), np.asarray(p1["lowest_at"])
        self.rates = list(m["rates"])
        self.per_seed = np.asarray(p2["per_seed"], float)
        self.n_sub = len(m["subsamples"])
        self.three = np.asarray(p3["ys"], float)
        self.early = m["early_stopping"]

    def subtitle_for(self, k):
        m = self.meta
        one, e = m["one"], self.early["seed0"]
        r = m["rates"]
        names = list(r)
        sub, dep = m["subsamples"], m["depths"]
        sn, dn = list(sub), list(dep)
        return [
            "handwritten 4s stand in for signal and 9s for background, as in an "
            "analysis. After %d trees each event has a score and so a probability p. "
            "The log-loss pulls each one towards its label with strength y − p"
            % one["after_trees"],
            "the next tree is fitted to the pulls, not to the labels: its questions "
            "sort events with similar pulls into the same leaf, and a leaf's value is "
            "the sum of its pulls over the sum of p(1 − p)",
            "every score moves by the learning rate, here %g, times the value of its "
            "leaf. The training loss falls from %.3f to %.3f, the held-out loss from "
            "%.3f to %.3f; then the pulls are worked out again"
            % (one["rate"], one["train_loss"][0], one["train_loss"][1], one["held_loss"][0],
               one["held_loss"][1]),
            "a small learning rate needs more trees and ends lower: lowest held-out "
            "loss %.3f after %d trees at rate %s, %.3f after %d at %s, %.3f after %d at "
            "%s (%d seeds). Past its lowest point each curve climbs"
            % (r[names[0]]["lowest"]["mean"], round(r[names[0]]["lowest_at"]),
               names[0][5:], r[names[2]]["lowest"]["mean"],
               round(r[names[2]]["lowest_at"]), names[2][5:],
               r[names[3]]["lowest"]["mean"], round(r[names[3]]["lowest_at"]),
               names[3][5:], len(m["seeds"])),
            "a random half of the rows for each tree: %.3f against %.3f with all of "
            "them. Depth is how many cells one tree can combine: depth 1, one cell at "
            "a time, stops at %.3f; depth %s is best here, %.3f"
            % (sub[sn[1]]["lowest"]["mean"], sub[sn[0]]["lowest"]["mean"],
               dep[dn[0]]["lowest"]["mean"],
               min(dn, key=lambda n: dep[n]["lowest"]["mean"])[6:],
               min(v["lowest"]["mean"] for v in dep.values())),
            "early stopping: watch the loss on validation events the trees never saw "
            "and keep the trees up to its lowest point, here %d. Held-out loss there "
            "%.3f; its own lowest %.3f; after all %d trees %.3f"
            % (e["chosen"], e["held_there"], e["held_lowest"], S.GB_ROUNDS, e["held_end"]),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _curve(self, sc, op, box=None):
        ch = self.pane(box or self.BOX, (-self.reach, self.reach), (0.0, 1.0))
        self.axes(sc, ch, op, "score F (the sum of the trees so far)", "probability p",
                  [(v, "%g" % v) for v in (-4, -3, -2, -1, 0, 1, 2, 3, 4)
                   if abs(v) < self.reach],
                  [(v, "%g" % v) for v in (0, 0.5, 1)], grid=False)
        xs = np.linspace(-self.reach, self.reach, 60)
        self.path(sc, ch, xs, S.sigmoid(xs), GREY, op, width=2.2)
        self.hline(sc, ch, 1.0, soft(SIG, 0.3), op, width=1.4)
        return ch

    def _dots(self, sc, ch, scores, op, pulls=0.0, r=3.6):
        for f, y in zip(scores, self.cls):
            p = float(S.sigmoid(f))
            at = (ch.fx(float(f)), ch.fy(p))
            if pulls > 0.01:
                self.arrow(sc, at, (at[0], ch.fy(p + (y - p) * pulls)), self.ink_of(y),
                           0.75 * op, width=1.3, head=False)
            self.disk(sc, self.P(*at), r, self.ink_of(y), op, lift=1.2)

    def _tree(self, sc, op, grow, values=1.0):
        t = self.t
        left, right = np.array(t["left"]), np.array(t["right"])

        def text(k):
            if left[k] >= 0:
                return question(t["cell"][k], t["threshold"][k])
            return signed(t["value"][k]) if values > 0.5 else "?"

        def fill(k):
            if left[k] >= 0:
                return soft(self.TREE, 0.5)
            return soft(SIG if t["value"][k] > 0 else BKG, 0.5) if values > 0.5 else PAPER
        level = {0: 0, 1: 1, 2: 1}
        pos = self.tree(sc, left, right, ((0.635, 0.945), (0.40, 0.60)), op, text, fill,
                        show=lambda k: float(np.clip(grow - level.get(k, 2), 0, 1)),
                        w=112, h=30, leaf_w=66,
                        size=0.029, border=lambda k: self.TREE if left[k] >= 0 else AXIS)
        return pos, left

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        ch = self._curve(sc, op)
        shown = smoothstep(0.1, 0.35, u)
        if shown > 0.02:
            self._dots(sc, ch, self.before, op * shown, smoothstep(0.42, 0.7, u))
        self.text(sc, 0.05, 0.315, "after %d trees: every event has a score"
                  % self.meta["one"]["after_trees"], op, anchor="lm", strong=True, size=HEAD)
        self.text(sc, ch.x1 + 0.01, ch.fy(1.0), "y = 1: signal", op, anchor="lm",
                  ink=dark(SIG, 0.25), size=SMALL)
        self.text(sc, ch.x1 + 0.01, ch.fy(0.0) - 0.012, "y = 0: background", op, anchor="lm",
                  ink=dark(BKG, 0.25), size=SMALL)
        self.lines(sc, 0.60, 0.47, [
            "p = 1 / (1 + e^−F)",
            "log-loss of an event:",
            "−ln p if signal, −ln(1 − p) if not"], op * shown, size=SMALL, step=0.044)
        show = op * smoothstep(0.45, 0.6, u)
        self.text(sc, 0.60, 0.635, "the pull on an event:  y − p", show, anchor="lm",
                  strong=True, size=HEAD)
        self.lines(sc, 0.60, 0.685, ["minus the slope of its loss in F:",
                                    "long for events the trees get wrong"], show,
                   size=SMALL, step=0.044)
        self.stand_in(sc, op)
        self.tag(sc, TRAINING, op)

    def phase_2(self, sc, u, op):
        ch = self._curve(sc, op)
        self._dots(sc, ch, self.before, op, 1.0)
        self.text(sc, 0.05, 0.315, "a small tree is fitted to the pulls", op, anchor="lm",
                  strong=True, size=HEAD)
        grow = 3.0 * smoothstep(0.05, 0.5, u)
        values = smoothstep(0.62, 0.75, u)
        pos, left = self._tree(sc, op, grow, values)
        t = self.t
        leaves = [k for k in sorted(pos, key=lambda k: pos[k][0]) if left[k] < 0]
        counts = op * float(np.clip(grow - 2, 0, 1))
        self.text(sc, 0.625, 0.665, "events", counts, anchor="rm", size=0.029)
        self.text(sc, 0.625, 0.71, "mean pull", counts, anchor="rm", size=0.029)
        self.text(sc, 0.625, 0.60, "value", counts * values, anchor="rm",
                  ink=dark(self.TREE, 0.4), size=0.029)
        for k in leaves:
            fx = pos[k][0]
            self.text(sc, fx, 0.665, "%d" % t["N"][k], counts, size=0.029)
            self.text(sc, fx, 0.71, signed(self.pull[k]), counts, size=0.029)
        self.lines(sc, 0.56, 0.775, [
            "value = Σ pulls / Σ p(1 − p)  in the leaf:",
            "the step that lowers its events' loss most"], op * values, size=SMALL,
            step=0.044)
        self.tag(sc, TRAINING, op)

    def phase_3(self, sc, u, op):
        ch = self._curve(sc, op)
        move = smoothstep(0.2, 0.7, u)
        for f in self.before:
            self.disk(sc, self.P(ch.fx(float(f)), ch.fy(float(S.sigmoid(f)))), 2.4, DIM,
                      op * move, lift=0.8)
        self._dots(sc, ch, self.before + (self.after - self.before) * move, op)
        self.text(sc, 0.05, 0.315, "every score takes a small step", op, anchor="lm",
                  strong=True, size=HEAD)
        self._tree(sc, 0.9 * op, 3.0)
        k = int(np.argmax(np.abs(self.after - self.before)))
        w = (self.after[k] - self.before[k]) / self.rate
        self.text(sc, 0.56, 0.675, "new score = score + %g × value of its leaf" % self.rate,
                  op, anchor="lm", strong=True, size=0.031)
        self.text(sc, 0.56, 0.725, "one event:  %s + %g × %s = %s"
                  % (f2(self.before[k]), self.rate, f2(w) if w >= 0 else "(%s)" % f2(w),
                     f2(self.after[k])), op * smoothstep(0.15, 0.3, u), anchor="lm",
                  size=SMALL)
        one = self.meta["one"]
        self.text(sc, 0.56, 0.80, "training loss %s → %s" % (f3(one["train_loss"][0]),
                                                              f3(one["train_loss"][1])),
                  op * smoothstep(0.6, 0.75, u), anchor="lm", ink=dark(self.TRAIN, 0.3),
                  size=0.031)
        self.text(sc, 0.56, 0.85, "held-out loss %s → %s" % (f3(one["held_loss"][0]),
                                                              f3(one["held_loss"][1])),
                  op * smoothstep(0.6, 0.75, u), anchor="lm", ink=dark(self.HELD, 0.3),
                  size=0.031)
        self.tag(sc, TRAINING, op)

    def phase_4(self, sc, u, op):
        go = smoothstep(0.05, 0.7, u)
        last = self.rounds[-1]
        ch = self.pane(((0.09, 0.60), (0.405, 0.80)), (1, last), (0.24, 0.70), xlog=True)
        self.axes(sc, ch, op, "trees", "held-out loss",
                  [(v, "%d" % v) for v in (1, 10, 100, 1000)],
                  [(v, "%g" % v) for v in (0.3, 0.4, 0.5, 0.6, 0.7)])
        x = self.rounds[1:]
        self.text(sc, 0.05, 0.315, "four learning rates, the same trees otherwise", op,
                  anchor="lm", strong=True, size=HEAD)
        for i, (name, ink) in enumerate(zip(self.rates, self.RATE_INKS)):
            ys = np.minimum(self.curves[i, 1:], 0.70)
            self.path(sc, ch, x, ys, ink, op, go, width=2.8)
            row = self.meta["rates"][name]
            at = max(int(round(row["lowest_at"])), 1)
            ring = op * smoothstep(0.7, 0.8, u)
            self.disk(sc, self.P(ch.fx(at), ch.fy(self.curves[i, at])), 5.0, ink, ring,
                      ring=DIGIT_INK, ring_w=1.4, lift=1.5)
            y = 0.45 + 0.085 * i
            self.box(sc, 0.655, y, 14, 14, ink, op, radius=3)
            self.text(sc, 0.672, y, name, op, anchor="lm", ink=dark(ink, 0.3), size=0.031)
            self.text(sc, 0.672, y + 0.04, "lowest %s, after %d trees"
                      % (f3(row["lowest"]["mean"]), at), ring, anchor="lm", size=SMALL)
        self.text(sc, 0.05, 0.925, "mean of %d seeds; ringed: the lowest point"
                  % len(self.meta["seeds"]), op, anchor="lm", size=SMALL)
        self.tag(sc, MEASURED, op)

    def phase_5(self, sc, u, op):
        m = self.meta
        yr = (0.24, 0.36)
        yt = [(v, "%.2f" % v) for v in (0.26, 0.28, 0.30, 0.32, 0.34)]
        groups = ((list(m["subsamples"]), 0, ((0.10, 0.44), (0.405, 0.76)), TEAL,
                   "rows used by each tree", [n.split(" ")[0] for n in m["subsamples"]]),
                  (list(m["depths"]), self.n_sub, ((0.58, 0.94), (0.405, 0.76)), VIOLET,
                   "depth of each tree", [n[6:] for n in m["depths"]]))
        for g, (names, first, box, ink, label, ticks) in enumerate(groups):
            n = len(names)
            ch = self.pane(box, (-0.6, n - 0.4), yr)
            self.axes(sc, ch, op, label, "lowest held-out loss" if g == 0 else "",
                      [(k, t) for k, t in enumerate(ticks)], yt, grid=False)
            for k in range(n):
                show = op * smoothstep(0.1 + 0.08 * k + 0.05 * g, 0.2 + 0.08 * k + 0.05 * g, u)
                v = np.minimum(self.per_seed[first + k], yr[1])
                self.strip(sc, ch, k, v, ink, show)
                mean = float(self.per_seed[first + k].mean())
                self.text(sc, ch.fx(k), ch.fy(float(v.max())) - 0.032, f3(mean), show,
                          ink=dark(ink, 0.35), size=0.029)
        self.text(sc, 0.05, 0.315, "one setting changed at a time, from rate %g, depth %d, "
                  "all rows" % (m["base"]["rate"], m["base"]["depth"]), op, anchor="lm",
                  strong=True, size=HEAD)
        self.text(sc, 0.05, 0.925, "a dot per seed, the bar and the number their mean", op,
                  anchor="lm", size=SMALL)
        self.tag(sc, MEASURED, op)

    def phase_6(self, sc, u, op):
        go = smoothstep(0.05, 0.55, u)
        e = self.early["seed0"]
        upto = 1000
        ch = self.pane(((0.09, 0.58), (0.405, 0.80)), (0, upto), (0.0, 0.5))
        self.axes(sc, ch, op, "trees", "loss",
                  [(v, "%d" % v) for v in (0, 250, 500, 750, 1000)],
                  [(v, "%g" % v) for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5)])
        x = self.rounds[:upto + 1]
        inks = (self.TRAIN, self.VALID, self.HELD)
        for ys, ink in zip(self.three, inks):
            self.path(sc, ch, x, np.minimum(ys[:upto + 1], 0.5), ink, op, go, width=2.8)
        mark = op * smoothstep(0.55, 0.68, u)
        self.vline(sc, ch, e["chosen"], self.VALID, mark, width=2.2, dashed=True)
        self.vline(sc, ch, e["stopped_at"], GREY, mark, width=1.6, dashed=True)
        self.disk(sc, self.P(ch.fx(e["chosen"]), ch.fy(self.three[1][e["chosen"]])), 5.5,
                  self.VALID, mark, ring=DIGIT_INK, ring_w=1.4, lift=1.5)
        self.text(sc, 0.05, 0.315, "rate %g: three sets of events, one run" % self.early["rate"],
                  op, anchor="lm", strong=True, size=HEAD)
        self.key(sc, 0.64, 0.43, [("training events", self.TRAIN),
                                  ("validation events", self.VALID),
                                  ("held-out events", self.HELD)], op, step=0.046, size=0.031)
        self.lines(sc, 0.63, 0.60, [
            ("validation is lowest after %d trees" % e["chosen"], dark(self.VALID, 0.3)),
            ("%d trees later without a new low:" % self.early["patience"], None),
            ("stop, and keep the first %d" % e["chosen"], None)], mark, size=SMALL, step=0.044)
        self.lines(sc, 0.63, 0.76, [
            ("held-out loss with those: %s" % f3(e["held_there"]), dark(self.HELD, 0.3)),
            ("with all %d trees: %s" % (S.GB_ROUNDS, f3(e["held_end"])), None)],
            op * smoothstep(0.7, 0.82, u), size=0.031, step=0.048)
        self.tag(sc, MEASURED, op)


# ====================================================================== 167: the gain
class SplitGain(Boost):
    """One event's loss with its tangent and parabola, g and h on every event; a leaf's
    best value as the bottom of a parabola; a threshold sliding through a handful of
    events with GL, HL, GR, HR and the gain; a small real node under lambda, gamma and
    min_child_weight; a node where first and second order disagree; and whether it
    matters for the held-out loss."""

    TITLE = "How XGBoost scores a split"
    PHASES = (("g and h", 10.0), ("a leaf", 10.0), ("a split", 12.0), ("restraints", 11.0),
              ("2nd order", 9.0), ("matters?", 9.0))
    LOSS, SLOPE, CURVE = DIGIT_INK, TEAL, AMBER
    LAM_INKS = (TEAL, BLUE, VIOLET)
    KIND_INKS = (GREY, AMBER, BLUE)

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.w = np.asarray(p0["x"], float)
        self.one = np.asarray(p0["ys"], float)
        self.ev = m["one_event"]
        self.hand = m["handful"]
        self.hx, self.hg = np.asarray(p1["xy"], float).T
        self.hy, self.hh = np.asarray(p1["classes"], float), np.asarray(p1["h"], float)
        self.hp = S.sigmoid(np.asarray(p1["score"], float))
        self.exact, self.approx = np.asarray(p1["exact"], float), np.asarray(p1["approx"])
        self.gain = np.asarray(p1["gain"], float)
        self.GL, self.HL = np.asarray(p1["GL"], float), np.asarray(p1["HL"], float)
        self.node = m["node"]
        self.nx, self.ngain = np.asarray(p2["x"], float), np.asarray(p2["ys"], float)
        self.nHL = np.asarray(p2["HL"], float)
        self.rounds = np.asarray(p3["x"], float)
        self.cmp = np.asarray(p3["ys"], float)
        self.picked = np.asarray(p3["picked"], float)
        self.share = np.asarray(p3["share"], float)
        self.dis = m["disagree"]

    def subtitle_for(self, k):
        m = self.meta
        e, hd, nd, sw, di, cp = (m["one_event"], m["handful"], m["node"], m["sweeps"],
                                 m["disagree"], m["compare"])
        names = cp["names"]
        return [
            "digits stand in for events in an analysis: 4s signal, 9s background. At an "
            "event's current score its loss has a slope g = p − y and a curvature "
            "h = p(1 − p): this 4 has p = %.2f, so g = %s and h = %.2f"
            % (e["p"], f2(e["g"]), e["h"]),
            "adding w to every score in a leaf changes the loss by about "
            "G w + ½(H + λ)w², G and H being the sums of g and h. That parabola is "
            "lowest at w = −G / (H + λ) = %s, where it reaches −G² / 2(H + λ) = %s"
            % (f2(hd["value"]), f3(hd["score"])),
            "a split gives each side its own parabola. Its gain is how much lower the "
            "two bottoms reach than the one before: best here after event %d, gain "
            "%.2f; a gain can be negative, since λ charges for every leaf value"
            % (hd["best"] + 1, max(hd["gain"])),
            "on a real node of %d events: λ shrinks leaf values (%s to %s on the left); "
            "γ = %g refuses the split, whose gain is only %.2f; min child weight %g "
            "forbids sides with too little curvature, %d of %d thresholds"
            % (nd["events"], f2(nd["left_value"][0]), f2(nd["left_value"][2]), nd["gamma"],
               nd["best_gain"][1], nd["mcw"], nd["thresholds"] - nd["allowed_by_mcw"],
               nd["thresholds"]),
            "plain gradient boosting picks splits by least squares on g, as if every h "
            "were 1. In the first round all h are equal and the two agree; by rounds "
            "31 to 100 they pick different splits at %s of the nodes"
            % pct(di["share_by_round"]["rounds_31_to_100"], 0),
            "does it matter here? Held-out loss %.3f for plain boosting, %.3f for "
            "second-order splits (%d seeds, differences %s to %s): no. With λ = 1 as "
            "well, %.3f; and γ made the trees smaller, not better"
            % (cp["held"][names[0]]["mean"], cp["held"][names[1]]["mean"], len(m["seeds"]),
               signed(cp["plain_minus_second"]["min"], 3),
               signed(cp["plain_minus_second"]["max"], 3), cp["held"][names[2]]["mean"]),
        ][k] if sw else ""

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        e = self.ev
        ch = self.pane(((0.09, 0.47), (0.405, 0.80)), (-3, 3), (-1.6, 3.0))
        self.axes(sc, ch, op, "change of this event's score", "change of its loss",
                  [(v, "%d" % v) for v in (-3, -2, -1, 0, 1, 2, 3)],
                  [(v, "%d" % v) for v in (-1, 0, 1, 2, 3)], grid=False)
        self.path(sc, ch, self.w, self.one[0], self.LOSS, op, smoothstep(0.03, 0.2, u),
                  width=3.2)
        a, b = smoothstep(0.16, 0.24, u), smoothstep(0.28, 0.36, u)
        self.path(sc, ch, self.w, self.one[1], self.SLOPE, op * a, width=2.4)
        self.path(sc, ch, self.w, np.minimum(self.one[2], 3.0), self.CURVE, op * b, width=2.8)
        self.disk(sc, self.P(ch.fx(0), ch.fy(0)), 5.0, SIG, op, ring=DIGIT_INK, ring_w=1.4,
                  lift=1.5)
        self.text(sc, 0.05, 0.315, "one event (a 4 with p = %s) and its log-loss"
                  % f2(e["p"]), op, anchor="lm", strong=True, size=HEAD)
        self.text(sc, 0.30, 0.44, "slope  g = p − y = %s" % f2(e["g"]), op * a, anchor="lm",
                  ink=dark(self.SLOPE, 0.3), size=0.031)
        self.text(sc, 0.30, 0.49, "curvature  h = p(1 − p) = %s" % f2(e["h"]), op * b,
                  anchor="lm", ink=dark(self.CURVE, 0.4), size=0.031)
        # the two numbers on every event
        show = smoothstep(0.38, 0.46, u)
        self.text(sc, 0.75, 0.375, "and for every event, two numbers", op * show, size=0.031)
        cols = (0.60, 0.70, 0.80, 0.90)
        for x, name in zip(cols, ("event", "p", "g", "h")):
            self.text(sc, x, 0.425, name, op * show, strong=True, size=SMALL)
        for i in range(len(self.hg)):
            y = 0.47 + 0.044 * i
            o = op * smoothstep(0.38 + 0.035 * i, 0.46 + 0.035 * i, u)
            ink = self.ink_of(self.hy[i])
            self.disk(sc, self.P(cols[0] - 0.02, y), 5.0, ink, o, lift=1.0)
            self.text(sc, cols[0] + 0.012, y, "a %d" % (S.SIGNAL if self.hy[i] > 0.5 else
                                                        S.BACKGROUND), o, ink=dark(ink, 0.25),
                      size=SMALL)
            self.text(sc, cols[1], y, f2(self.hp[i]), o, size=SMALL)
            self.text(sc, cols[2], y, signed(self.hg[i]), o, ink=dark(self.SLOPE, 0.3),
                      size=SMALL)
            self.text(sc, cols[3], y, f2(self.hh[i]), o, ink=dark(self.CURVE, 0.4), size=SMALL)
        self.stand_in(sc, op)
        self.tag(sc, EXACT, op)

    def phase_2(self, sc, u, op):
        hd = self.hand
        keep = (self.w >= -0.8) & (self.w <= 1.2)
        ch = self.pane(((0.09, 0.49), (0.405, 0.80)), (-0.8, 1.2), (-0.15, 0.9))
        self.axes(sc, ch, op, "w: the value added to all %d scores" % hd["events"],
                  "change of their summed loss (+ ½λw²)",
                  [(v, "%g" % v) for v in (-0.5, 0, 0.5, 1.0)],
                  [(v, "%g" % v) for v in (0, 0.25, 0.5, 0.75)])
        w = self.w[keep]
        self.path(sc, ch, w, np.minimum(self.exact[keep], 0.9), self.LOSS, op,
                  smoothstep(0.03, 0.2, u), width=3.2)
        para = smoothstep(0.3, 0.45, u)
        self.path(sc, ch, w, np.minimum(self.approx[keep], 0.9), self.CURVE, op * para,
                  width=2.8)
        low = op * smoothstep(0.55, 0.68, u)
        bottom = (ch.fx(hd["value"]), ch.fy(hd["score"]))
        self.dashes(sc, (bottom[0], ch.fy(0.0)), bottom, self.CURVE, low, n=5)
        self.disk(sc, self.P(*bottom), 5.5, self.CURVE, low, ring=DIGIT_INK, ring_w=1.4,
                  lift=1.5)
        self.text(sc, 0.05, 0.315, "all %d events in one leaf: which value w is best?"
                  % hd["events"], op, anchor="lm", strong=True, size=HEAD)
        self.key(sc, 0.20, 0.44, [("the true change", self.LOSS),
                                  ("G w + ½(H + λ)w²", self.CURVE)], op * para, step=0.044,
                 size=SMALL)
        x = 0.57
        self.rect(sc, self.P(0.725, 0.385), self.span(0.31)[0], 30, PAPER, op, border=AXIS,
                  radius=8, lift=0.1)
        for i, y in enumerate(self.hy):                    # the leaf and the events in it
            self.disk(sc, self.P(0.60 + 0.036 * i, 0.385), 6.0, self.ink_of(y), op, lift=1.0)
        self.text(sc, x, 0.445, "G = Σ g = %s" % f2(hd["G"]), op, anchor="lm",
                  ink=dark(self.SLOPE, 0.3), size=HEAD)
        self.text(sc, x, 0.49, "H = Σ h = %s" % f2(hd["H"]), op, anchor="lm",
                  ink=dark(self.CURVE, 0.4), size=HEAD)
        self.text(sc, x, 0.535, "λ = %g: a charge on large leaf values" % hd["lam"], op,
                  anchor="lm", size=SMALL)
        self.text(sc, x, 0.605, "lowest at  w = −G / (H + λ)", low, anchor="lm", size=0.031)
        self.text(sc, x, 0.65, "= %s / %s = %s" % (f2(-hd["G"]), f2(hd["H"] + hd["lam"]),
                                                   f2(hd["value"])), low, anchor="lm",
                  strong=True, size=HEAD)
        deep = op * smoothstep(0.72, 0.84, u)
        self.text(sc, x, 0.725, "where it reaches  −G² / 2(H + λ)", deep, anchor="lm",
                  size=0.031)
        self.text(sc, x, 0.77, "= %s: the leaf's score" % f3(hd["score"]), deep, anchor="lm",
                  strong=True, size=HEAD)
        self.text(sc, x, 0.835, "pulled both ways, they gain little", deep, anchor="lm",
                  size=SMALL)
        self.own(sc, op)
        self.tag(sc, EXACT, op)

    def phase_3(self, sc, u, op):
        hd = self.hand
        n = len(self.hg)
        best = hd["best"]
        slide = smoothstep(0.08, 0.62, u) * (n - 1)            # gaps passed, 0..7
        settle = smoothstep(0.66, 0.74, u)
        k = int(min(slide, n - 2)) if settle < 0.5 else best    # the gap the cut is in
        xs = [0.125 + 0.064 * i for i in range(n)]
        cut = xs[k] + 0.032
        self.text(sc, 0.05, 0.315, "the same events in the order of cell %s; slide a cut"
                  % hd["cell"], op, anchor="lm", strong=True, size=HEAD)
        self.rect(sc, self.P((xs[0] - 0.03 + cut) / 2, 0.435),
                  self.span(cut - xs[0] + 0.03)[0], 96, SIG, op, radius=6, lift=0.1,
                  fill_op=0.10)
        self.rect(sc, self.P((cut + xs[-1] + 0.03) / 2, 0.435),
                  self.span(xs[-1] + 0.03 - cut)[0], 96, AMBER, op, radius=6, lift=0.1,
                  fill_op=0.12)
        self.polyline(sc, [self.P(cut, 0.355, 1.2), self.P(cut, 0.62, 1.2)], 3.0, DIGIT_INK, op)
        for row, name in ((0.43, "g"), (0.475, "h")):
            self.text(sc, 0.085, row, name, op, strong=True, size=SMALL)
        for i in range(n):
            self.disk(sc, self.P(xs[i], 0.385), 6.0, self.ink_of(self.hy[i]), op, lift=1.0)
            self.text(sc, xs[i], 0.43, signed(self.hg[i]), op, ink=dark(self.SLOPE, 0.3),
                      size=0.029)
            self.text(sc, xs[i], 0.475, f2(self.hh[i]), op, ink=dark(self.CURVE, 0.4),
                      size=0.029)
        # the gain of every cut passed so far, as a bar under its gap
        top = float(self.gain.max())
        self.polyline(sc, [self.P(xs[0], 0.60, 0.4), self.P(xs[-1], 0.60, 0.4)], 1.2, AXIS, op)
        self.text(sc, 0.085, 0.565, "gain", op, strong=True, size=SMALL)
        for j in range(n - 1):
            if j > slide and settle < 0.5:
                continue
            h = 44.0 * self.gain[j] / top
            gx = xs[j] + 0.032
            ink = AMBER if j == best else GREY
            dy = self.frac(0, h / 2)[1]
            if abs(h) > 0.8:
                self.rect(sc, self.P(gx, 0.60 - dy), 22, abs(h), ink, op, radius=2, lift=0.5)
        GL, HL = float(self.GL[k]), float(self.HL[k])
        GR, HR = hd["G"] - GL, hd["H"] - HL
        x = 0.66
        self.text(sc, x, 0.385, "left of the cut", op, anchor="lm", ink=dark(SIG, 0.2),
                  size=SMALL)
        self.text(sc, x, 0.43, "GL = %s   HL = %s" % (f2(GL), f2(HL)), op, anchor="lm",
                  strong=True, size=0.032)
        self.text(sc, x, 0.49, "right of the cut", op, anchor="lm", ink=dark(AMBER, 0.4),
                  size=SMALL)
        self.text(sc, x, 0.535, "GR = %s   HR = %s" % (f2(GR), f2(HR)), op, anchor="lm",
                  strong=True, size=0.032)
        self.text(sc, x, 0.60, "gain = %s" % f2(self.gain[k]), op, anchor="lm",
                  ink=dark(AMBER, 0.4) if k == best else None, size=BIG)
        lam = hd["lam"]
        a = op * smoothstep(0.72, 0.82, u)
        self.text(sc, 0.09, 0.70, "gain = ½ [ GL² / (HL + λ) + GR² / (HR + λ) − G² / (H + λ) ] "
                  "− γ", a, anchor="lm", size=0.032)
        bl, br = self.GL[best], hd["G"] - self.GL[best]
        hl, hr = self.HL[best], hd["H"] - self.HL[best]
        terms = (bl * bl / (hl + lam), br * br / (hr + lam), hd["G"] ** 2 / (hd["H"] + lam))
        b = op * smoothstep(0.8, 0.9, u)
        self.text(sc, 0.09, 0.76, "= ½ [ %s² / %s + %s² / %s − %s² / %s ] − 0"
                  % (f2(abs(bl)), f2(hl + lam), f2(br), f2(hr + lam), f2(abs(hd["G"])),
                     f2(hd["H"] + lam)), b, anchor="lm", size=0.032)
        self.text(sc, 0.09, 0.82, "= ½ [ %s + %s − %s ] = %s"
                  % (f3(terms[0]), f3(terms[1]), f3(terms[2]), f2(self.gain[best])), b,
                  anchor="lm", strong=True, size=0.032)
        self.own(sc, op)
        self.tag(sc, EXACT, op)

    def phase_4(self, sc, u, op):
        nd = self.node
        lo, hi = float(self.nx[0]), float(self.nx[-1])
        floor = float(np.floor(self.ngain.min() * 5) / 5)
        ch = self.pane(((0.09, 0.52), (0.405, 0.79)), (lo, hi), (floor, 1.1))
        self.axes(sc, ch, op, "cut on cell %s" % nd["cell"], "gain of the split",
                  [(v, "%g" % v) for v in (0.2, 0.3, 0.4, 0.5)],
                  [(v, "%g" % v) for v in (-1.0, -0.5, 0, 0.5, 1.0) if v >= floor])
        self.text(sc, 0.05, 0.315, "a real node of %d events, and three ways to restrain it"
                  % nd["events"], op, anchor="lm", strong=True, size=HEAD)
        self.hline(sc, ch, 0.0, AXIS, op, dashed=False, width=1.0)
        show = [smoothstep(0.03 + 0.1 * i, 0.13 + 0.1 * i, u) for i in range(3)]
        for i, ink in enumerate(self.LAM_INKS):
            self.path(sc, ch, self.nx, self.ngain[i], ink, op * show[i], width=2.8)
        x = 0.585
        self.text(sc, x, 0.385, "λ shrinks the leaf values:  left, right", op * show[0],
                  anchor="lm", size=SMALL)
        for i, (lam, ink) in enumerate(zip(nd["lams"], self.LAM_INKS)):
            y = 0.43 + 0.043 * i
            self.box(sc, x + 0.008, y, 14, 14, ink, op * show[i], radius=3)
            self.text(sc, x + 0.026, y, "λ = %g:   %s,  %s" % (
                lam, signed(nd["left_value"][i]), signed(nd["right_value"][i])),
                op * show[i], anchor="lm", ink=dark(ink, 0.3), size=0.031)
        g = op * smoothstep(0.42, 0.54, u)
        self.hline(sc, ch, nd["gamma"], RED, g, width=2.2)
        self.text(sc, ch.x1 - 0.005, ch.fy(nd["gamma"]) - 0.024, "γ = %g" % nd["gamma"], g,
                  anchor="rm", ink=dark(RED, 0.2), size=SMALL)
        sw = self.meta["sweeps"]["gamma"]
        half = sw["values"].index(0.5)
        self.lines(sc, x, 0.585, [("γ: a split must gain more than γ", dark(RED, 0.2)),
                                  ("none here does: the node stays a leaf", None),
                                  ("measured: %.1f leaves a tree at γ = 0.5, not %.1f"
                                   % (sw["leaves"][half], sw["leaves"][0]), None)], g,
                   size=SMALL, step=0.043)
        w = op * smoothstep(0.66, 0.78, u)
        ok = (self.nHL >= nd["mcw"]) & (nd["H"] - self.nHL >= nd["mcw"])
        step = (hi - lo) / (len(self.nx) - 1) / 2
        for v, allowed in zip(self.nx, ok):
            if not allowed:
                self.shade(sc, ch, max(v - step, lo), min(v + step, hi), floor, 1.1, GREY, w,
                           fill=0.32)
        self.lines(sc, x, 0.745, [("min child weight %g: each side needs" % nd["mcw"],
                                  dark(GREY, 0.3)),
                                 ("Σ h of at least that; greyed cuts would", None),
                                 ("leave a side with almost no events", None)], w,
                   size=SMALL, step=0.043)
        self.own(sc, op)
        self.tag(sc, TRAINING, op)

    def phase_5(self, sc, u, op):
        di = self.dis
        self.text(sc, 0.05, 0.315, "one node of %d events (round %d), two candidate splits"
                  % (di["events"], di["round"]), op, anchor="lm", strong=True, size=HEAD)
        cols = (0.335, 0.475)
        names = ("%s < %s" % (di["second"]["cell"], f2(di["second"]["threshold"])),
                 "%s < %s" % (di["plain"]["cell"], f2(di["plain"]["threshold"])))
        for x, name in zip(cols, names):
            self.box(sc, x, 0.40, 120, 32, soft(AMBER, 0.5), op, name, border=AMBER,
                     size=SMALL)
        rows = (("second order:  ½ Σ G² / H", di["second_scores"], BLUE),
                ("least squares:  ½ Σ G² / N", di["plain_scores"], GREY))
        for r, (name, scores, ink) in enumerate(rows):
            y = 0.48 + 0.075 * r
            o = op * smoothstep(0.1 + 0.2 * r, 0.25 + 0.2 * r, u)
            self.text(sc, 0.26, y, name, o, anchor="rm", ink=dark(ink, 0.3), size=SMALL)
            win = int(np.argmax(scores))
            for c, x in enumerate(cols):
                self.box(sc, x, y, 96, 34, soft(ink, 0.35 if c == win else 0.75), o,
                         f2(scores[c]), border=AMBER if c == win else AXIS, size=0.032)
                if c == win:
                    self.rect(sc, self.P(x, y), 102, 40, AMBER, o, border=AMBER, border_w=2.6,
                              radius=7, lift=0.9, fill_op=0.0)
        v = op * smoothstep(0.5, 0.62, u)
        self.lines(sc, 0.05, 0.655, [
            "each picks its own; the leaf values differ more:",
            "mean pull  −G / N = %s and %s" % (signed(di["mean_pull"][0]),
                                               signed(di["mean_pull"][1])),
            "Newton value  −G / H = %s and %s" % (signed(di["newton"][0]),
                                                  signed(di["newton"][1]))], v,
            size=SMALL, step=0.044)
        ch = self.pane(((0.63, 0.94), (0.405, 0.76)), (1, len(self.share)), (0.0, 0.8))
        self.axes(sc, ch, op, "round", "nodes where the two disagree",
                  [(k, "%d" % k) for k in (1, 25, 50, 75, 100)],
                  [(k, "%d%%" % round(100 * k)) for k in (0, 0.25, 0.5, 0.75)])
        x = np.arange(1, len(self.share) + 1)
        width = max(self.span(ch.x1 - ch.x0)[0] / len(x) - 0.6, 1.0)
        self.columns(sc, ch, x, self.share, BLUE, op, width,
                     grow=smoothstep(0.3, 0.7, u), fill=0.85)
        self.own(sc, op)
        self.tag(sc, TRAINING, op)

    def phase_6(self, sc, u, op):
        cp = self.meta["compare"]
        go = smoothstep(0.05, 0.55, u)
        upto = 600
        ch = self.pane(((0.09, 0.52), (0.405, 0.79)), (20, upto), (0.25, 0.45), xlog=True)
        self.axes(sc, ch, op, "trees", "held-out loss",
                  [(v, "%d" % v) for v in (50, 100, 200, 600)],
                  [(v, "%.2f" % v) for v in (0.25, 0.30, 0.35, 0.40, 0.45)])
        x = self.rounds[20:upto + 1]
        for ys, ink in zip(self.cmp, self.KIND_INKS):
            self.path(sc, ch, x, np.minimum(ys[20:upto + 1], 0.45), ink, op, go, width=2.8)
        self.text(sc, 0.05, 0.315, "the same events, rate %g, depth %d, three ways"
                  % (cp["rate"], cp["depth"]), op, anchor="lm", strong=True, size=HEAD)
        names = ("plain: least-squares splits", "second-order splits, λ = 0",
                 "second-order splits, λ = 1")
        pc = self.pane(((0.63, 0.93), (0.405, 0.63)), (-0.6, 2.6), (0.25, 0.28))
        self.axes(sc, pc, op, "", "held-out loss where validation is lowest",
                  [], [(v, "%.2f" % v) for v in (0.25, 0.26, 0.27, 0.28)])
        for i, (name, ink) in enumerate(zip(names, self.KIND_INKS)):
            show = op * smoothstep(0.5 + 0.08 * i, 0.62 + 0.08 * i, u)
            self.strip(sc, pc, i, self.picked[i], ink, show)
            y = 0.70 + 0.046 * i
            self.box(sc, 0.60, y, 14, 14, ink, op, radius=3)
            self.text(sc, 0.617, y, "%s:  %s" % (name, f3(self.picked[i].mean())), op,
                      anchor="lm", ink=dark(ink, 0.35), size=SMALL)
        self.text(sc, 0.05, 0.925, "curves: mean of %d seeds; dots: each seed"
                  % len(self.meta["seeds"]), op, anchor="lm", size=SMALL)
        self.tag(sc, MEASURED, op)


# ====================================================================== 168: histograms
def commas(n):
    return format(int(round(float(n))), ",")


class Histograms(Boost):
    """A dozen events sorted and every gap tried; events dropped into quantile bins, the
    G and H histograms and the scan of their boundaries; a child's histogram as the
    parent's minus its sibling's; a split that tries the missing readings on each side;
    the four ways of handling them, measured; two trees growing level-wise and
    leaf-wise; and time and held-out loss against the number of bins."""

    TITLE = "XGBoost's histograms"
    PHASES = (("sort", 7.5), ("bins", 10.5), ("subtract", 8.0), ("missing", 9.0),
              ("its side", 8.0), ("growth", 8.5), ("speed", 9.5))
    G, H, GAIN = TEAL, AMBER, VIOLET
    WAY_INKS = (GREEN, BLUE, RED, GREY)

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.cell = m["drawn"]["cell"]
        self.few_x, self.few_y = np.asarray(p0["few_x"], float), np.asarray(p0["few_y"])
        self.rank = np.argsort(np.argsort(self.few_x, kind="stable"), kind="stable")
        self.edges = np.asarray(p0["edges"], float)
        self.drop_y, self.drop_bin = np.asarray(p0["drop_y"]), np.asarray(p0["drop_bin"], int)
        self.drop_g, self.drop_h = np.asarray(p0["drop_g"], float), np.asarray(p0["drop_h"])
        self.scan = np.asarray(p0["gain"], float)
        self.sub = [np.asarray(p0[k], float) for k in ("sub_parent", "sub_right", "sub_left")]
        self.sub_n = [int(p0["sub_n"][k]) for k in (0, 2, 1)]
        self.miss_hist = np.asarray(p1["miss_hist"], float)
        self.miss_gain = np.asarray(p1["miss_gain"], float)
        self.miss_sample = np.asarray(p1["miss_sample"], float)
        self.miss = m["missing"]
        self.loss = np.asarray(p1["per_seed"], float).reshape(2, 4, -1)
        self.bins = np.asarray(p2["x"], float)
        self.by_bins, self.ms = np.asarray(p2["ys"], float)
        self.exact = np.asarray(p2["exact"], float)
        self.trees = [{k: np.asarray(p3["%s_%s" % (which, k)]) for k in
                       ("left", "right", "order", "level")} for which in ("level", "leaf")]
        self.growth = m["growth"]
        self.timing = m["timing"]

    def subtitle_for(self, k):
        m = self.meta
        dr, su, mi, gr, ti = (m["drawn"], m["subtraction"], m["missing"], m["growth"],
                              m["timing"])
        held = mi["held"]
        b = m["by_bins"]
        return [
            "digits stand in for an analysis's events: 4s signal, 9s background. The "
            "exact way to find a split sorts a node's events by each cell and tries "
            "every gap: per tree, %s readings sorted and %s gaps scored"
            % (commas(ti["sorted_per_tree"]), commas(ti["scored_per_tree"]["exact"])),
            "the histogram way cuts each cell into bins once, at its quantiles. A node "
            "then only adds each event's g and h into its bin and scans the bin "
            "boundaries: %s cuts scored per tree with %d bins"
            % (commas(ti["scored_per_tree"]["%d bins" % S.BINS]), S.BINS),
            "a node's events are its two children's, so a child's histogram is the "
            "parent's minus its sibling's. Only the smaller child is filled: %s "
            "readings put into bins per tree and not %s"
            % (commas(ti["added_per_tree"]["%d bins" % S.BINS]),
               commas(ti["added_without_subtraction"])),
            "readings can be missing; here %d%% were removed at random. They get a "
            "slot of their own, and each split tries them on the left, then on the "
            "right, and keeps the better: here gain %.2f against %.2f"
            % (round(100 * S.HIST_RANDOM), mi["drawn"]["gain_left"],
               mi["drawn"]["gain_right"]),
            "measured: with random holes the learnt side gives held-out loss %.3f, "
            "always left %.3f, filling in the mean %.3f. When low readings go "
            "unreported, always left is right, %.3f; learning it costs little, %.3f"
            % (held["random"]["learnt"]["mean"], held["random"]["left"]["mean"],
               held["random"]["mean"]["mean"], held["floor"]["left"]["mean"],
               held["floor"]["learnt"]["mean"]),
            "XGBoost grows a tree level by level; splitting the leaf with the largest "
            "gain first gives a lopsided tree with the same %d leaves. Held-out loss "
            "%.3f and %.3f over %d seeds: no difference to speak of here"
            % (gr["leaves"], gr["level"]["mean"], gr["leaf"]["mean"], len(m["seeds"])),
            "%d bins lose nothing against the exact method, %.3f and %.3f held-out "
            "loss, and this numpy code finds a tree's splits %.0f times faster. The "
            "times are this machine's and vary; the counts of operations do not"
            % (S.BINS, b["%d bins" % S.BINS]["mean"], b["exact"]["mean"],
               ti["speedup_at_32"]),
        ][k] if dr and su else ""

    # ------------------------------------------------------------------ shared drawing
    def _hist(self, sc, box, values, ink, op, scale, grow=1.0, signed_=True, width=None):
        """A small histogram of ``values`` (one bar per bin) in the layout ``box``."""
        n = len(values)
        ch = self.pane(box, (-0.5, n - 0.5), (-scale if signed_ else 0.0, scale))
        self.polyline(sc, [self.P(ch.x0, ch.fy(0.0), 0.4), self.P(ch.x1, ch.fy(0.0), 0.4)],
                      1.2, AXIS, op)
        width = width or self.span((ch.x1 - ch.x0) / n)[0] - 5.0
        self.columns(sc, ch, np.arange(n), values, ink, op, width, grow)
        return ch

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        n = len(self.few_x)
        self.text(sc, 0.05, 0.315, "the exact way: sort the events by a cell, try every gap",
                  op, anchor="lm", strong=True, size=HEAD)
        move = smoothstep(0.18, 0.5, u)
        xs = [0.10 + 0.0727 * k for k in range(n)]
        for i in range(n):
            a, b = xs[i], xs[int(self.rank[i])]
            fx = a + (b - a) * move
            fy = 0.47 - 0.07 * np.sin(np.pi * move) * (1 if i % 2 else -1) * (a != b)
            self.disk(sc, self.P(fx, fy), 8.0, self.ink_of(self.few_y[i]), op, lift=1.0)
            still = float(np.clip(1.0 - 40.0 * move * (1.0 - move), 0.0, 1.0))
            self.text(sc, fx, fy + 0.042, f2(self.few_x[i]), op * still, size=0.029)
        self.text(sc, 0.05, 0.385, "%d events and their reading of cell %s" % (n, self.cell),
                  op, anchor="lm", size=SMALL)
        gaps = smoothstep(0.55, 0.85, u) * (n - 1)
        for k in range(n - 1):
            o = op * float(np.clip(gaps - k, 0, 1))
            fx = (xs[k] + xs[k + 1]) / 2
            self.polyline(sc, [self.P(fx, 0.435, 1.2), self.P(fx, 0.505, 1.2)], 2.4, self.GAIN,
                          o)
        done = op * smoothstep(0.55, 0.65, u)
        self.text(sc, 0.05, 0.60, "%d events: %d gaps to score, for one cell of one node"
                  % (n, n - 1), done, anchor="lm", size=0.031, ink=dark(self.GAIN, 0.3))
        ti = self.timing
        self.lines(sc, 0.05, 0.675, [
            "a real node here: %s events × 16 cells, sorted again at every node"
            % commas(2 * S.TRAIN),
            "per tree of depth %d: %s readings sorted, %s gaps scored"
            % (S.HIST_SET["depth"], commas(ti["sorted_per_tree"]),
               commas(ti["scored_per_tree"]["exact"]))],
            op * smoothstep(0.75, 0.88, u), size=0.031, step=0.05)
        self.stand_in(sc, op)
        self.tag(sc, EXACT, op)

    def phase_2(self, sc, u, op):
        nb = len(self.edges) + 1
        n = len(self.drop_bin)
        x0, dx = 0.115, 0.058
        top, floor = 0.385, 0.565
        self.text(sc, 0.05, 0.315, "the histogram way: bins made once, then only sums", op,
                  anchor="lm", strong=True, size=HEAD)
        for k in range(nb + 1):                           # the walls of the bins
            fx = x0 + dx * k
            self.polyline(sc, [self.P(fx, top, 0.4), self.P(fx, floor, 0.4)], 1.4, AXIS, op)
            if 0 < k < nb:
                self.text(sc, fx, floor + 0.027, f2(self.edges[k - 1]), op, size=0.029)
        self.polyline(sc, [self.P(x0, floor, 0.4), self.P(x0 + dx * nb, floor, 0.4)], 1.4,
                      AXIS, op)
        self.text(sc, x0 - 0.008, floor + 0.027, self.cell, op, anchor="rm", strong=True,
                  size=0.029)
        fallen = smoothstep(0.04, 0.5, u) * n
        height = np.zeros(nb, int)
        G, H = np.zeros(nb), np.zeros(nb)
        for i in range(n):
            if i >= fallen:
                break
            b = self.drop_bin[i]
            w = float(np.clip(fallen - i, 0, 1))
            col, row = divmod(height[b], 4)               # four to a row within a bin
            fx = x0 + dx * b + 0.011 + 0.012 * row
            rest = floor - 0.016 - 0.024 * col
            fy = top - 0.02 + (rest - top + 0.02) * smoothstep(0.0, 1.0, w)
            self.disk(sc, self.P(fx, fy), 4.6, self.ink_of(self.drop_y[i]), op, lift=1.0)
            height[b] += 1
            G[b] += self.drop_g[i] * w
            H[b] += self.drop_h[i] * w
        xs = [x0 + dx * (k + 0.5) for k in range(nb)]
        for row, (name, vals, ink, base, scale) in enumerate((
                ("G: Σ g", G, self.G, 0.685, 1.5), ("H: Σ h", H, self.H, 0.83, 1.8))):
            self.text(sc, x0 - 0.008, base - 0.025 * row, name, op, anchor="rm",
                      ink=dark(ink, 0.35), size=0.029)
            self.polyline(sc, [self.P(x0, base, 0.4), self.P(x0 + dx * nb, base, 0.4)], 1.2,
                          AXIS, op)
            for fx, v in zip(xs, vals):
                h = 40.0 * v / scale
                if abs(h) > 0.8:
                    self.rect(sc, self.P(fx, base - self.frac(0, h / 2)[1]), 40, abs(h), ink,
                              op, radius=2, lift=0.5)
        # the scan of the seven boundaries
        scan = smoothstep(0.58, 0.9, u) * (nb - 1)
        best = int(self.scan.argmax())
        k = int(min(scan, nb - 2))
        if scan > 0.01:
            fx = x0 + dx * (k + 1 if scan < nb - 1.01 else best + 1)
            for a, b in ((top - 0.01, floor + 0.004), (floor + 0.05, 0.845)):
                self.polyline(sc, [self.P(fx, a, 1.3), self.P(fx, b, 1.3)], 3.0, self.GAIN, op)
        x = 0.635
        self.text(sc, x, 0.385, "then scan the %d boundaries:" % (nb - 1),
                  op * smoothstep(0.55, 0.62, u), anchor="lm", size=SMALL)
        full = self.span(0.13)[0]
        for j in range(nb - 1):
            o = op * float(np.clip(scan - j + 0.5, 0, 1))
            y = 0.44 + 0.05 * j
            self.text(sc, x + 0.10, y, "cut at %s" % f2(self.edges[j]), o, anchor="rm",
                      size=0.029)
            self.hbar(sc, x + 0.115, y, full * self.scan[j] / self.scan.max(),
                      self.GAIN if j == best else soft(self.GAIN, 0.4), o,
                      value="gain %s" % f2(self.scan[j]), thick=16.0)
        self.text(sc, x, 0.81, "sums to the left of a cut: running totals",
                  op * smoothstep(0.7, 0.8, u), anchor="lm", size=SMALL)
        self.own(sc, op)
        self.tag(sc, EXACT, op)

    def phase_3(self, sc, u, op):
        su = self.meta["subtraction"]
        self.text(sc, 0.05, 0.315, "the subtraction trick: fill the smaller child only", op,
                  anchor="lm", strong=True, size=HEAD)
        boxes = (((0.07, 0.30), (0.45, 0.67)), ((0.39, 0.62), (0.45, 0.67)),
                 ((0.71, 0.94), (0.45, 0.67)))
        names = ("the node: %s events" % commas(self.sub_n[0]),
                 "smaller child: %s, filled" % commas(self.sub_n[1]),
                 "its sibling: %s, not visited" % commas(self.sub_n[2]))
        inks = (self.G, self.G, self.GAIN)
        scale = float(np.abs(self.sub[0][0]).max()) * 1.1
        shows = (1.0, smoothstep(0.1, 0.3, u), smoothstep(0.45, 0.7, u))
        for box, name, ink, hist, show in zip(boxes, names, inks, self.sub, shows):
            self._hist(sc, box, hist[0], ink, op * show, scale)
            self.text(sc, (box[0][0] + box[0][1]) / 2, 0.40, name, op * show,
                      ink=dark(ink, 0.3), size=SMALL)
        self.text(sc, 0.345, 0.56, "−", op * shows[1], strong=True, size=0.06)
        self.text(sc, 0.665, 0.56, "=", op * shows[2], strong=True, size=0.06)
        self.text(sc, 0.05, 0.72, "G in the 8 bins of cell %s; the node was split on %s < %s"
                  % (self.cell, su["root_cell"], f2(su["root_threshold"])), op, anchor="lm",
                  size=SMALL)
        a, b, c = (h[0][0] for h in self.sub)
        self.text(sc, 0.05, 0.775, "first bin:  %s − (%s) = %s   and the same for H, in "
                  "every bin of every cell" % (f2(a), f2(b), f2(c)), op * shows[2],
                  anchor="lm", strong=True, size=0.031)
        ti = self.timing
        self.text(sc, 0.05, 0.835, "readings put into bins per tree: %s, not %s"
                  % (commas(ti["added_per_tree"]["%d bins" % S.BINS]),
                     commas(ti["added_without_subtraction"])),
                  op * smoothstep(0.75, 0.88, u), anchor="lm", size=0.031)
        self.own(sc, op)
        self.tag(sc, EXACT, op)

    def phase_4(self, sc, u, op):
        mi = self.miss["drawn"]
        self.text(sc, 0.05, 0.315, "missing readings: try them on each side of the cut", op,
                  anchor="lm", strong=True, size=HEAD)
        rows, cols = self.miss_sample.shape
        for r in range(rows):
            for c in range(cols):
                v = self.miss_sample[r, c]
                fx, fy = 0.085 + 0.052 * c, 0.435 + 0.052 * r
                gone = v < 0
                hide = smoothstep(0.05, 0.25, u) if gone else 0.0
                self.box(sc, fx, fy, 44, 24, soft(GREY, 0.75 - 0.35 * hide) if gone else PAPER,
                         op, "?" if hide > 0.5 else ("" if gone else f2(v)), size=0.029,
                         radius=3, border=AXIS)
        self.text(sc, 0.06, 0.385, "6 events × 8 of their cells", op, anchor="lm", size=SMALL)
        self.lines(sc, 0.06, 0.765, ["the rule here: each reading removed",
                                     "with probability %g" % S.HIST_RANDOM], op, size=SMALL,
                   step=0.043)
        # the G histogram of the cell, the missing events in a slot of their own
        G = self.miss_hist[0]
        nb = len(G) - 1
        scale = float(np.abs(G).max()) * 1.15
        box = ((0.53, 0.82), (0.47, 0.65))
        ch = self._hist(sc, box, G[:-1], self.G, op, scale)
        mb = self.pane(((0.87, 0.92), (0.47, 0.65)), (-0.5, 0.5), (-scale, scale))
        self.polyline(sc, [self.P(mb.x0, mb.fy(0), 0.4), self.P(mb.x1, mb.fy(0), 0.4)], 1.2,
                      AXIS, op)
        self.columns(sc, mb, [0], [G[-1]], GREY, op, 30)
        self.text(sc, 0.53, 0.385, "G by bin of cell %s, and for the %d missing"
                  % (self.cell, mi["missing"]), op, anchor="lm", size=0.029)
        cut = ch.fx(int(np.argmax(self.miss_gain.max(1))) + 0.5)
        self.polyline(sc, [self.P(cut, 0.475, 1.2), self.P(cut, 0.655, 1.2)], 3.0, self.GAIN,
                      op)
        self.text(sc, cut, 0.685, "cut at %s" % f2(mi["threshold"]), op,
                  ink=dark(self.GAIN, 0.3), size=0.029)
        self.text(sc, 0.895, 0.685, "missing", op, ink=dark(GREY, 0.3), size=0.029)
        left, right = smoothstep(0.3, 0.45, u), smoothstep(0.52, 0.67, u)
        self.arc(sc, (0.895, 0.475), (ch.fx(2.5), 0.465), BLUE, op * left, left, bend=0.06)
        self.arc(sc, (0.87, 0.63), (ch.fx(nb - 1) + 0.02, 0.63), RED, op * right, right,
                 bend=0.0)
        best = smoothstep(0.75, 0.85, u)
        for k, (name, gain, ink, show) in enumerate((
                ("missing go left:", mi["gain_left"], BLUE, left),
                ("missing go right:", mi["gain_right"], RED, right))):
            y = 0.755 + 0.055 * k
            won = (k == 0) == mi["sent_left"]
            self.text(sc, 0.53, y, "%s  gain %s" % (name, f2(gain)), op * show, anchor="lm",
                      ink=dark(ink, 0.25), size=0.032)
            if won:
                self.text(sc, 0.80, y, "kept: the default side", op * best, anchor="lm",
                          strong=True, size=SMALL)
        self.own(sc, op)
        self.tag(sc, TRAINING, op)

    def phase_5(self, sc, u, op):
        mi = self.miss
        self.text(sc, 0.05, 0.315, "four ways to handle the holes, two rules for making them",
                  op, anchor="lm", strong=True, size=HEAD)
        yr = (0.35, 0.46)
        heads = ("%d%% of readings removed at random" % round(100 * S.HIST_RANDOM),
                 "readings below %g not reported" % S.HIST_FLOOR)
        for g, rule in enumerate(S.HIST_RULES):
            box = ((0.10 + 0.47 * g, 0.46 + 0.47 * g), (0.40, 0.72))
            ch = self.pane(box, (-0.6, 3.6), yr)
            self.axes(sc, ch, op, heads[g], "held-out loss" if g == 0 else "",
                      [(k, w) for k, w in enumerate(("learnt side", "all left", "all right",
                                                     "fill mean"))],
                      [(v, "%.2f" % v) for v in (0.36, 0.38, 0.40, 0.42, 0.44)], grid=False)
            for k, ink in enumerate(self.WAY_INKS):
                show = op * smoothstep(0.08 + 0.1 * k + 0.05 * g, 0.2 + 0.1 * k + 0.05 * g, u)
                v = self.loss[g, k]
                self.strip(sc, ch, k, v, ink, show)
                self.text(sc, ch.fx(k), ch.fy(float(v.max())) - 0.032, f3(v.mean()), show,
                          ink=dark(ink, 0.35), size=0.029)
            self.text(sc, (box[0][0] + box[0][1]) / 2, 0.855,
                      "learnt: %s of the splits send them left" % pct(mi["sent_left"][rule], 0),
                      op * smoothstep(0.6, 0.72, u), ink=dark(GREEN, 0.3), size=SMALL)
        self.text(sc, 0.05, 0.925, "a dot per seed; with nothing missing: %s"
                  % f3(mi["complete"]["mean"]), op, anchor="lm", size=SMALL)
        self.tag(sc, MEASURED, op)

    def phase_6(self, sc, u, op):
        gr = self.growth
        self.text(sc, 0.05, 0.315, "two ways to grow a tree of %d leaves; the numbers: the "
                  "order of the splits" % gr["leaves"], op, anchor="lm", strong=True, size=HEAD)
        step = smoothstep(0.05, 0.75, u) * (gr["leaves"] - 1) + 0.001
        frames = (((0.06, 0.47), (0.44, 0.44 + 0.085 * gr["level_depth"])),
                  ((0.53, 0.94), (0.44, 0.44 + 0.062 * gr["leaf_depth"])))
        heads = ("level by level (XGBoost's default)", "best leaf first")
        for t, box, head, key in zip(self.trees, frames, heads, ("level", "leaf")):
            left, right, order = t["left"], t["right"], t["order"]
            parent = {int(c): k for k in range(len(left)) if left[k] >= 0
                      for c in (left[k], right[k])}

            def show(k, order=order, parent=parent):
                if k == 0:
                    return 1.0
                return float(np.clip(step - order[parent[k]], 0, 1))

            def split(k, order=order):
                return order[k] >= 0 and step > order[k]

            self.tree(sc, left, right, box, op,
                      lambda k, order=order, split=split: "%d" % (order[k] + 1) if split(k)
                      else "",
                      lambda k, split=split: soft(AMBER, 0.35) if split(k) else soft(GREY, 0.6),
                      show=show, w=34, h=24, leaf_w=34, size=0.029, sides=False,
                      border=lambda k, split=split: AMBER if split(k) else AXIS)
            cx = (box[0][0] + box[0][1]) / 2
            self.text(sc, cx, 0.385, head, op, strong=True, size=SMALL)
            self.text(sc, cx, 0.875, "held-out loss %s (%s to %s)" % (
                f3(gr[key]["mean"]), f3(gr[key]["min"]), f3(gr[key]["max"])),
                op * smoothstep(0.78, 0.9, u), ink=dark(GREEN, 0.3), size=0.031)
        self.own(sc, op)
        self.tag(sc, MEASURED, op)

    def phase_7(self, sc, u, op):
        ti = self.timing
        go = smoothstep(0.05, 0.5, u)
        ticks = [(v, "%d" % v) for v in (2, 8, 32, 128)]
        self.text(sc, 0.05, 0.315, "how fast, how good: against the number of bins", op,
                  anchor="lm", strong=True, size=HEAD)
        right = self.pane(((0.59, 0.93), (0.42, 0.72)), (2, 256), (0.24, 0.48), xlog=True)
        left = self.pane(((0.10, 0.44), (0.42, 0.72)), (2, 256), (0.0, max(self.exact[1],
                                                                          self.ms.max()) * 1.1),
                         xlog=True)
        self.axes(sc, left, op, "bins", "split finding per tree (ms), this machine", ticks,
                  grid=False)
        self.path(sc, left, self.bins, self.ms, BLUE, op, go, width=3.0)
        for x, v, w in zip(self.bins, self.ms, self.by_bins):
            seen = op * float(left.fx(x) <= left.x0 + (left.x1 - left.x0) * go + 1e-9)
            self.disk(sc, self.P(left.fx(x), left.fy(v)), 4.6, BLUE, seen, lift=1.2)
            self.disk(sc, self.P(right.fx(x), right.fy(w)), 4.6, GREEN, seen, lift=1.2)
        self.hline(sc, left, self.exact[1], RED, op, width=2.2)
        self.axes(sc, right, op, "bins", "held-out loss, %d seeds" % len(self.meta["seeds"]),
                  ticks, [(v, "%.2f" % v) for v in (0.25, 0.30, 0.35, 0.40, 0.45)])
        self.path(sc, right, self.bins, self.by_bins, GREEN, op, go, width=3.0)
        self.hline(sc, right, self.exact[0], RED, op, width=2.2)
        name = "%d bins" % S.BINS
        k = list(self.bins).index(S.BINS)
        done = op * smoothstep(0.5, 0.6, u)
        # the numbers sit under the charts, where no timing can push them into a curve
        self.text(sc, 0.10, 0.84, "%s: %.2f ms" % (name, self.ms[k]), done, anchor="lm",
                  ink=dark(BLUE, 0.3), size=SMALL)
        self.text(sc, 0.27, 0.84, "exact (dashed): %.2f ms" % self.exact[1], done, anchor="lm",
                  ink=dark(RED, 0.2), size=SMALL)
        self.text(sc, 0.59, 0.84, "%s: %s" % (name, f3(self.by_bins[k])), done, anchor="lm",
                  ink=dark(GREEN, 0.3), size=SMALL)
        self.text(sc, 0.76, 0.84, "exact (dashed): %s" % f3(self.exact[0]), done, anchor="lm",
                  ink=dark(RED, 0.2), size=SMALL)
        self.lines(sc, 0.05, 0.885, [
            "counted, the same on any machine: cuts scored per tree %s (%s) against %s "
            "(exact)" % (commas(ti["scored_per_tree"][name]), name,
                         commas(ti["scored_per_tree"]["exact"]))],
            op * smoothstep(0.55, 0.7, u), size=SMALL, step=0.044)
        self.own(sc, op)
        self.tag(sc, TIMED, op)


# ====================================================================== 169: one round
class OneRound(Boost):
    """Eight events through one round: their g and h, the root's histograms and best
    split, the tree with its weakest split pruned by gamma and its leaf weights, the
    step and the new predictions; then many rounds, the knobs one at a time, and five
    ways to use trees on the same events."""

    TITLE = "One round of XGBoost"
    PHASES = (("g and h", 8.5), ("split", 8.5), ("tree", 9.5), ("step", 8.5),
              ("rounds", 8.0), ("knobs", 9.5), ("rivals", 8.5))
    G, H, TREE = TEAL, AMBER, AMBER
    STAGES = ("predictions p", "g and h", "histograms", "best split", "tree, leaf weights",
              "step")
    RIVAL_INKS = (OCHRE, GREEN, VIOLET, GREY, BLUE)

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.images = np.asarray(p0["values"], float)
        self.labels = list(p0["labels"])
        self.table = np.asarray(p0["table"], float)        # round, field, event
        self.round = m["round"]
        keep = np.asarray(p0["root_hist"])[2] > 0
        self.root = np.asarray(p0["root_hist"], float)[:, keep]
        self.root_gain = np.asarray(p0["root_gain"], float)[:int(keep.sum()) - 1]
        self.by_cell = np.asarray(p0["best_by_feature"], float)
        self.t = m["round"]["tree"]
        self.kept = np.asarray(p0["kept"], bool)
        self.rounds = np.asarray(p1["x"], float)
        self.many = np.asarray(p1["ys"], float)
        self.knob_end = np.asarray(p2["values"], float)
        self.knob_stop = np.asarray(p2["stopped"], float)
        self.knob_names = list(p2["labels"])
        self.err, self.auc = np.asarray(p3["per_seed"], float), np.asarray(p3["auc_per_seed"])
        self.rivals = list(p3["labels"])

    def subtitle_for(self, k):
        m = self.meta
        r, mn, lo, kn, rv = m["round"], m["many"], m["loose"], m["knobs"], m["rivals"]
        gap = m["boosting_gap"]["gradient_minus_xgboost_error"]
        return [
            "handwritten 4s stand in for an analysis's signal events, 9s for "
            "background. A round starts from the current predictions: before any tree "
            "p = 0.5 for all, so g = p − y is ±0.5 and h = p(1 − p) is 0.25",
            "g and h are added into the bins of every cell, and every bin boundary is "
            "scored: %d cuts in 16 cells. The best is %s < %s, with gain %.0f; the "
            "events go left or right of it and each side is searched in turn"
            % (r["candidates"], r["root_cell"], ("%.3f" % r["root_threshold"]),
               r["root_gain"]),
            "the tree grows to the depth limit, %d. Then γ = %g prunes from the bottom: "
            "a split that gains only %.1f is undone, leaving %d leaves. Each leaf gets "
            "the weight −G / (H + λ) from the sums of its events"
            % (r["depth"], r["gamma"], r["pruned_gain"][0], r["leaves_after"]),
            "every event's score moves by the learning rate, %g, times the weight of "
            "its leaf; the training loss falls from %.3f to %.3f. Round two starts from "
            "the new p, g and h, and takes it to %.3f"
            % (r["rate"], r["train_loss"][0], r["train_loss"][1], r["train_loss"][2]),
            "%d rounds at rate %g: the held-out loss is lowest, %.3f, after %d. With "
            "half the rows and half the cells for each tree the lowest is %.3f: "
            "subsampling does not help on these events"
            % (mn["rounds"], mn["rate"], mn["held_lowest"]["all"],
               mn["held_lowest_at"]["all"], mn["held_lowest"]["sampled"]),
            "from a loose setting, held-out loss %.2f after %d rounds, one knob turned "
            "at a time: depth %d brings it to %.2f, λ = %g to %.2f, a learning rate of %g "
            "to %.2f. Half the rows per tree does nothing by itself: %.2f"
            % (lo["end"]["mean"], lo["rounds"], kn["max depth"]["value"],
               kn["max depth"]["end"]["mean"], kn["lambda"]["value"],
               kn["lambda"]["end"]["mean"], kn["learning rate"]["value"],
               kn["learning rate"]["end"]["mean"], kn["subsample"]["end"]["mean"]),
            "held-out error over %d seeds: one tree %s, a forest %s, AdaBoost %s, "
            "gradient boosting %s, this XGBoost-style code %s. The last gap is small, "
            "%s to %s, but there in every seed"
            % (len(m["seeds"]), pct(rv["one tree"]["error"]["mean"]),
               pct(rv["forest"]["error"]["mean"]), pct(rv["AdaBoost"]["error"]["mean"]),
               pct(rv["gradient boosting"]["error"]["mean"]),
               pct(rv["XGBoost-style"]["error"]["mean"]), pct(gap["min"]), pct(gap["max"])),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _pipe(self, sc, op, lit):
        """The round as a pipeline, the stages in ``lit`` alight."""
        from . import layout

        widths = [layout.text_width(s, 0.029) + 0.022 for s in self.STAGES]
        gap = (0.90 - sum(widths)) / (len(widths) - 1)
        x = 0.05
        for k, (name, w) in enumerate(zip(self.STAGES, widths)):
            on = k in lit
            self.box(sc, x + w / 2, 0.32, self.span(w)[0], 26, soft(AMBER, 0.35) if on else
                     PAPER, op * (1.0 if on else 0.75), name, border=AMBER if on else AXIS,
                     size=0.029, radius=5)
            if k:
                self.arrow(sc, (x - gap + 0.004, 0.32), (x - 0.004, 0.32), AXIS, op, width=1.6)
            x += w + gap

    def _events(self, sc, op, rows, formulas=()):
        """The eight events as columns: picture, then a row of numbers per entry of
        ``rows`` (``(name, values, ink, opacity)``)."""
        xs = [0.215 + 0.083 * k for k in range(len(self.labels))]
        self.text(sc, 0.165, 0.445, "8 events", op, anchor="rm", strong=True, size=SMALL)
        for fx, img, label in zip(xs, self.images, self.labels):
            ink = SIG if label == str(S.SIGNAL) else BKG
            self.digit(sc, img, fx, 0.445, 50, op, frame=ink)
        for r, (name, values, ink, show) in enumerate(rows):
            y = 0.545 + 0.047 * r
            o = op * show
            self.text(sc, 0.165, y, name, o, anchor="rm", ink=ink, strong=ink is None,
                      size=SMALL)
            for fx, v in zip(xs, values):
                self.text(sc, fx, y, v, o, ink=ink, size=SMALL)
            if r < len(formulas) and formulas[r]:
                self.text(sc, xs[-1] + 0.05, y, formulas[r], o, anchor="lm", size=0.029)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        self._pipe(sc, op, (0,) if u < 0.4 else (0, 1))
        t = self.table[0]
        a, b, c = (smoothstep(x, x + 0.12, u) for x in (0.12, 0.42, 0.6))
        self._events(sc, op, (
            ("label y", ["%d" % (lab == str(S.SIGNAL)) for lab in self.labels], None, 1.0),
            ("score F", [f2(v) for v in t[0]], None, a),
            ("p", [f2(v) for v in t[1]], None, a),
            ("g", [signed(v) for v in t[2]], dark(self.G, 0.3), b),
            ("h", [f2(v) for v in t[3]], dark(self.H, 0.4), c)),
            ("", "", "1/(1+e^−F)", "p − y", "p(1 − p)"))
        self.text(sc, 0.05, 0.815, "8 of the %s training events, before any tree"
                  % commas(2 * S.TRAIN), op, anchor="lm", size=SMALL)
        self.text(sc, 0.05, 0.865, "g: which way each score should move, and how hard.   "
                  "h: how fast its loss bends", op * b, anchor="lm", size=SMALL)
        self.stand_in(sc, op)
        self.tag(sc, EXACT, op)

    def phase_2(self, sc, u, op):
        r = self.round
        self._pipe(sc, op, (2,) if u < 0.5 else (2, 3))
        nb = self.root.shape[1]
        grow = smoothstep(0.05, 0.35, u)
        gs = float(np.abs(self.root[0]).max()) * 1.1
        hs = float(self.root[1].max()) * 1.1
        left = ((0.13, 0.47), (0.43, 0.61))
        for box, vals, ink, scale, name, sign in (
                (left, self.root[0], self.G, gs, "G: Σ g", True),
                (((0.13, 0.47), (0.66, 0.80)), self.root[1], self.H, hs, "H: Σ h", False)):
            n = len(vals)
            ch = self.pane(box, (-0.5, n - 0.5), (-scale if sign else 0.0, scale))
            self.polyline(sc, [self.P(ch.x0, ch.fy(0.0), 0.4), self.P(ch.x1, ch.fy(0.0), 0.4)],
                          1.2, AXIS, op)
            self.columns(sc, ch, np.arange(n), vals, ink, op, 30, grow)
            self.text(sc, 0.12, ch.fy(0.0), name, op, anchor="rm", ink=dark(ink, 0.35),
                      size=SMALL)
        self.text(sc, 0.30, 0.385, "all %s events in the %d bins of cell %s"
                  % (commas(2 * S.TRAIN), nb, r["root_cell"]), op, size=SMALL)
        cut = smoothstep(0.4, 0.5, u)
        fx = ch.fx(0.5)
        self.polyline(sc, [self.P(fx, 0.42, 1.2), self.P(fx, 0.81, 1.2)], 3.0, VIOLET, op * cut)
        self.text(sc, 0.30, 0.85, "the best cut of this cell: after its first bin, gain %.0f"
                  % r["root_gain"], op * cut, ink=dark(VIOLET, 0.3), size=SMALL)
        # the best gain of every cell
        bc = self.pane(((0.60, 0.94), (0.44, 0.74)), (-0.6, 15.6), (0.0, self.by_cell.max()
                                                                      * 1.12))
        self.axes(sc, bc, op, "", "best gain in each of the 16 cells", [],
                  [(v, "%d" % v) for v in (0, 50, 100, 150)])
        show = smoothstep(0.5, 0.8, u)
        best = int(self.by_cell.argmax())
        for j in range(16):
            ink = VIOLET if j == best else soft(VIOLET, 0.45)
            self.columns(sc, bc, [j], [self.by_cell[j]], ink, op, 13, show)
            if j % 4 == 0 and abs(j - best) > 1:
                self.text(sc, bc.fx(j), bc.y1 + 0.028, S.cell_name(j), op, size=0.029)
        self.text(sc, bc.fx(best), bc.y1 + 0.028, S.cell_name(best), op, strong=True,
                  size=0.029)
        self.text(sc, 0.77, 0.85, "the best of all: %s < %.3f" % (r["root_cell"],
                                                                 r["root_threshold"]),
                  op * smoothstep(0.75, 0.85, u), strong=True, size=0.031)
        self.own(sc, op)
        self.tag(sc, TRAINING, op)

    def phase_3(self, sc, u, op):
        r, t = self.round, self.t
        self._pipe(sc, op, (4,))
        left, right = np.array(t["left"]), np.array(t["right"])
        level = {}
        todo = [(0, 0)]
        while todo:
            k, lv = todo.pop()
            level[k] = lv
            if left[k] >= 0:
                todo += [(int(left[k]), lv + 1), (int(right[k]), lv + 1)]
        grow = smoothstep(0.03, 0.4, u) * 4.2
        gains = smoothstep(0.46, 0.54, u) * (1 - smoothstep(0.86, 0.92, u))
        cut = smoothstep(0.66, 0.8, u)
        pruned = set(r["pruned_nodes"])
        gone = {int(c) for k in pruned for c in (left[k], right[k])}

        def text(k):
            if left[k] < 0 or (k in pruned and cut > 0.5):
                return signed(t["value"][k])
            if gains > 0.5:
                return "gain %s" % ("%.0f" % t["gain"][k] if t["gain"][k] >= 10 else
                                    "%.1f" % t["gain"][k])
            return question(t["cell"][k], t["threshold"][k])

        def fill(k):
            if left[k] < 0 or (k in pruned and cut > 0.5):
                return soft(SIG if t["value"][k] > 0 else BKG, 0.5)
            if k in pruned and gains > 0.5:
                return soft(RED, 0.3)
            return soft(self.TREE, 0.5)

        def show(k):
            base = float(np.clip(grow - level[k], 0, 1))
            return base * (1 - cut) if k in gone else base
        self.tree(sc, left, right, ((0.07, 0.93), (0.40, 0.67)), op, text, fill, show=show,
                  w=104, h=28, leaf_w=72, size=0.029,
                  border=lambda k: AXIS if left[k] < 0 else (RED if k in pruned and gains > 0.5
                                                             else self.TREE))
        self.text(sc, 0.05, 0.745, "grown to depth %d: every split is the best cut of its "
                  "node's own histograms" % r["depth"], op, anchor="lm", size=SMALL)
        k = r["pruned_nodes"][0]
        self.text(sc, 0.05, 0.795, "γ = %g: a split at the bottom that gains less (%.1f) is "
                  "undone" % (r["gamma"], t["gain"][k]), op * smoothstep(0.56, 0.66, u),
                  anchor="lm", ink=dark(RED, 0.2), size=SMALL)
        leaf = max((j for j in range(len(left)) if left[j] < 0 and j not in gone),
                   key=lambda j: t["N"][j])
        self.text(sc, 0.05, 0.845, "leaf weight = −G / (H + λ):  the leaf of %d events has "
                  "G = %s, H = %s, so %s" % (t["N"][leaf], f2(t["G"][leaf]), f2(t["H"][leaf]),
                                            signed(t["value"][leaf])),
                  op * smoothstep(0.8, 0.9, u), anchor="lm", strong=True, size=SMALL)
        self.own(sc, op)
        self.tag(sc, TRAINING, op)

    def phase_4(self, sc, u, op):
        r = self.round
        self._pipe(sc, op, (5,) if u < 0.62 else (0, 1))
        t0, t1 = self.table[0], self.table[1]
        a, b, c, d = (smoothstep(x, x + 0.1, u) for x in (0.05, 0.22, 0.42, 0.64))
        self._events(sc, op, (
            ("score F", [f2(v) for v in t0[0]], None, 1.0),
            ("leaf weight w", [signed(v) for v in t0[4]], dark(self.TREE, 0.4), a),
            ("new F", [signed(v) for v in t1[0]], None, b),
            ("new p", [f2(v) for v in t1[1]], None, c),
            ("new g", [signed(v) for v in t1[2]], dark(self.G, 0.3), d),
            ("new h", [f2(v) for v in t1[3]], dark(self.H, 0.4), d)),
            ("", "", "F + %g × w" % r["rate"], "", "", ""))
        self.text(sc, 0.05, 0.865, "training loss %s → %s, and → %s after round two, which "
                  "starts from the new g and h" % (f3(r["train_loss"][0]),
                                                   f3(r["train_loss"][1]),
                                                   f3(r["train_loss"][2])),
                  op * d, anchor="lm", strong=True, size=SMALL)
        self.own(sc, op)
        self.tag(sc, EXACT, op)

    def phase_5(self, sc, u, op):
        mn = self.meta["many"]
        go = smoothstep(0.05, 0.6, u)
        ch = self.pane(((0.09, 0.56), (0.405, 0.80)), (0, self.rounds[-1]), (0.0, 0.7))
        self.axes(sc, ch, op, "rounds", "loss",
                  [(v, "%d" % v) for v in (0, 200, 400, 600)],
                  [(v, "%g" % v) for v in (0, 0.2, 0.4, 0.6)])
        inks = (VIOLET, GREEN, soft(VIOLET, 0.45), TEAL)
        for ys, ink, w in zip(self.many, inks, (2.8, 3.0, 2.0, 2.4)):
            self.path(sc, ch, self.rounds, ys, ink, op, go, width=w)
        self.text(sc, 0.05, 0.315, "round after round: rate %g, depth %d, λ = %g"
                  % (mn["rate"], mn["depth"], mn["lam"]), op, anchor="lm", strong=True,
                  size=HEAD)
        self.key(sc, 0.61, 0.43, [("training, every row and cell", VIOLET),
                                  ("held-out, every row and cell", GREEN),
                                  ("training, half of each per tree", soft(VIOLET, 0.45)),
                                  ("held-out, half of each per tree", TEAL)], op, step=0.046,
                 size=SMALL)
        show = op * smoothstep(0.6, 0.75, u)
        self.lines(sc, 0.60, 0.655, [
            ("lowest held-out loss:", None),
            ("%s after %d rounds, all rows and cells" % (f3(mn["held_lowest"]["all"]),
                                                         mn["held_lowest_at"]["all"]),
             dark(GREEN, 0.3)),
            ("%s after %d, half of each" % (f3(mn["held_lowest"]["sampled"]),
                                           mn["held_lowest_at"]["sampled"]),
             dark(TEAL, 0.3))], show, size=SMALL, step=0.046)
        self.text(sc, 0.05, 0.925, "mean of %d seeds" % len(self.meta["seeds"]), op,
                  anchor="lm", size=SMALL)
        self.tag(sc, MEASURED, op)

    def phase_6(self, sc, u, op):
        m = self.meta
        lo, kn = m["loose"], m["knobs"]
        self.text(sc, 0.05, 0.315, "the knobs, one at a time: held-out loss after %d rounds"
                  % lo["rounds"], op, anchor="lm", strong=True, size=HEAD)
        cols = (0.05, 0.235, 0.385, 0.66)
        for x, name in zip(cols, ("knob", "loose → turned", "what it restrains",
                                  "after %d rounds" % lo["rounds"])):
            self.text(sc, x, 0.375, name, op, anchor="lm", strong=True, size=0.029)
        self.text(sc, 0.95, 0.375, "stopped early", op, anchor="rm", strong=True, size=0.029)
        full = self.span(0.17)[0] / float(self.knob_end.max())
        rows = [("all loose", "", "nothing", self.knob_end[0], self.knob_stop[0], GREY)]
        for i, name in enumerate(self.knob_names[1:], start=1):
            k = kn[name]
            rows.append((name, "%g → %g" % (k["loose"], k["value"]), k["restrains"],
                         self.knob_end[i], self.knob_stop[i], BLUE))
        for r, (name, change, holds, end, stop, ink) in enumerate(rows):
            y = 0.43 + 0.052 * r
            o = op * smoothstep(0.05 + 0.07 * r, 0.15 + 0.07 * r, u)
            self.text(sc, cols[0], y, name, o, anchor="lm", strong=True, size=SMALL)
            self.text(sc, cols[1], y, change, o, anchor="lm", size=SMALL)
            self.text(sc, cols[2], y, holds, o, anchor="lm", size=0.029)
            self.hbar(sc, cols[3], y, full * end, ink, o, value=f3(end), thick=15.0)
            self.text(sc, 0.95, y, f3(stop), o, anchor="rm", ink=dark(GREEN, 0.3), size=SMALL)
        self.text(sc, 0.05, 0.875, "stopped early: at the round with the lowest validation "
                  "loss", op, anchor="lm", size=0.029)
        self.own(sc, op)
        self.tag(sc, MEASURED, op)

    def phase_7(self, sc, u, op):
        rv = self.meta["rivals"]
        self.text(sc, 0.05, 0.315, "five ways to use trees: the same events, the same "
                  "held-out digits", op, anchor="lm", strong=True, size=HEAD)
        self.text(sc, 0.245, 0.385, "held-out error", op, anchor="lm", strong=True, size=SMALL)
        self.text(sc, 0.60, 0.385, "over the seeds", op, anchor="lm", strong=True, size=SMALL)
        self.text(sc, 0.80, 0.385, "ROC area", op, anchor="lm", strong=True, size=SMALL)
        full = self.span(0.25)[0] / 0.22
        names = ("one tree, depth tuned", "forest-style averaging", "AdaBoost, stumps",
                 "gradient boosting", "XGBoost-style")
        for r, (key, name, ink) in enumerate(zip(self.rivals, names, self.RIVAL_INKS)):
            y = 0.455 + 0.078 * r
            o = op * smoothstep(0.05 + 0.1 * r, 0.2 + 0.1 * r, u)
            e, a = rv[key]["error"], rv[key]["auc"]
            self.hbar(sc, 0.245, y, full * e["mean"], ink, o, label=name, value=pct(e["mean"]),
                      u=(u - 0.05 - 0.1 * r) / 0.2, thick=22.0)
            self.text(sc, 0.60, y, "%s to %s" % (pct(e["min"]), pct(e["max"])), o, anchor="lm",
                      size=SMALL)
            self.text(sc, 0.80, y, "%s (%s to %s)" % (f3(a["mean"]), f3(a["min"]),
                                                      f3(a["max"])), o, anchor="lm",
                      ink=dark(ink, 0.35), size=SMALL)
        self.text(sc, 0.05, 0.875, "every choice (depth, number of trees) made on validation "
                  "events; %d seeds" % len(self.meta["seeds"]), op, anchor="lm", size=0.029)
        self.own(sc, op)
        self.tag(sc, MEASURED, op)


ANIMATIONS = {
    "adaboost": AdaBoost,
    "boosting-shrinkage-and-early-stopping": BoostingSettings,
    "xgboost-split-gain": SplitGain,
    "xgboost-histograms-and-missing-values": Histograms,
    "xgboost-one-round": OneRound,
}

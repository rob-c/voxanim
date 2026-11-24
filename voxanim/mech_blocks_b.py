"""Animations 188-190 of the "Blocks and code" series (see :mod:`voxanim.blocks`):
``nn.Linear``, ``nn.ReLU`` and ``nn.Sequential``, drawn from :mod:`voxanim.study_blocks_b`
(the course's model after one epoch, and the first batch through it).

* :class:`ALinear` (``blocks-a-linear-layer``): the first digit as a strip of 784; the
  layer's 64 x 784 weights as a grid of orange tiles and its 64 biases as a thin strip; one
  row lit: three of its products read out, all 784 added up as a marker sweeps along, the
  bias added; then the 64 outputs, and ``[64, 784] -> [64, 64]``.
* :class:`ReLU` (``blocks-relu``): those 64 numbers; eight read out, then each negative
  one turns into a grey 0; the shape stays and nothing is learned; the whole batch, and the
  share of numbers set to 0 there and over all 60,000 digits.
* :class:`Stack` (``blocks-a-model-is-a-stack``): the six blocks stacked one by one as
  the code's lines run; each one's output shape; ``print(model)``'s real text beside them
  (the blocks stretch to their lines); and each block's count of learned numbers, 52,650
  in all, as "Counting the weights: a DNN and a CNN" counts them.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). What stays across phases is drawn in
``steady``, so nothing blanks between them.
"""

from __future__ import annotations

import numpy as np

from .blocks import (
    BIAS,
    BIG,
    DATA,
    DATA_HI,
    DATA_LO,
    HEAD,
    INK,
    SIZE,
    WEIGHT,
    ZERO,
    Blocks,
    Code,
    commas,
    exact,
    measured,
    shades,
    val,
)
from .diagram import draw_image, shrink_columns
from .easing import ease_in_out, smoothstep
from .mech import AXIS, PAPER, dark, soft

DATA_INK = dark(DATA, 0.3)
WEIGHT_INK = dark(WEIGHT, 0.35)
ZERO_INK = dark(ZERO, 0.45)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def pct(v):
    return "%d%%" % round(100 * float(v))


# ====================================================================== 188
class ALinear(Blocks):
    """The input strip; the grid of weights and the bias strip; one output; all 64."""

    TITLE = "nn.Linear: a grid of weights"
    PHASES = (("the input", 10.0), ("the weights", 13.0), ("one output", 20.0),
              ("64 outputs", 12.0))
    LAST = "Flatten: [64, 784], no weights"
    NEXT = "nn.ReLU: negatives become 0"
    STATION = ("forward",) * 4
    X0, X1 = 0.115, 0.425              # the strips' and the grid's columns
    XY, WY = 0.395, 0.437              # the input strip, and the lit row's copy
    STRIP_H = 14.0
    GY, GH = 0.585, 120.0              # the grid: centre, height (px)
    BX, BW = 0.447, 9.0                # the bias strip
    OY = 0.80                          # the outputs' row
    PAIR_AT = (0.10, 0.2, 0.3)         # phase 3: when each product is read out
    SWEEP = (0.45, 0.6)                # phase 3: the marker runs along the 784 products
    BIAS_AT, OUT_AT = 0.66, 0.78

    def setup(self):
        m = self.meta = self.study.metadata
        s = self.study
        self.x = np.asarray(s.panels[0].data["values"][0], float)
        self.W = np.asarray(s.panels[1].data["values"], float)
        self.b = np.asarray(s.panels[1].data["bias"], float)
        self.run = np.asarray(s.panels[2].data["ys"][0], float)
        self.z = np.asarray(s.panels[3].data["values"], float)
        self.j = m["output"]
        # the shading's ends: the 1st and 99th percentiles, so that the grid's texture
        # shows (the few weights beyond them are drawn as paper or full ink)
        self.wlo, self.whi = (float(v) for v in np.percentile(self.W, [1, 99]))
        self.blo, self.bhi = float(self.b.min()), float(self.b.max())
        self.zlo, self.zhi = m["out_range"]
        w = self.span(self.X1 - self.X0)[0]
        self.gw = w
        self.grid_rgb = shrink_columns(shades(self.W, WEIGHT, self.wlo, self.whi), int(w))
        line = m["linear_line"]
        out = "%s → %s" % (m["batch_in_text"], m["batch_out_text"])
        self.CODE = (Code("model", run=((0.0, line - 1), (0.35, line))),
                     Code("model", run=((0.0, line),)),
                     Code("model", run=((0.0, line),)),
                     Code("model", run=((0.0, line),), out=((0.45, out),)))
        self.EVIDENCE = (exact("the first digit of the batch, flattened"),
                         measured("the weights after one epoch of SGD"),
                         exact("784 products added, plus the bias"),
                         exact("the first batch through nn.Linear"))

    def heading_for(self, k, u):
        j = self.j
        if k == 2:
            return ("one output: row %d" % j if u < 0.1 else "multiply the matching numbers"
                    if u < self.SWEEP[0] else "add them all up" if u < self.BIAS_AT
                    else "add the bias")
        return ("in: one digit, 784 numbers", "nn.Linear(784, 64): a grid of weights", "",
                "every row gives one output: 64")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "x is the first digit of the batch, a %d, flattened: %d numbers in one strip."
            % (m["label"], m["inputs"]),
            "nn.Linear(784, 64) holds 64 rows of 784 weights (%s) and a strip of 64 biases."
            % commas(m["weights"]),
            "Row %d: each number of x times its weight, added up (%s), plus the bias (%s), "
            "makes %s." % (self.j, val(m["added"]), val(m["bias"]), val(m["out"])),
            "Each of the 64 rows gives one output, so the batch %s comes out as %s."
            % (m["batch_in_text"], m["batch_out_text"]),
        ][k]

    # ------------------------------------------------------------------ geometry
    def col_x(self, i):
        """The layout x of column ``i`` of the strips and the grid."""
        return self.X0 + (self.X1 - self.X0) * (i + 0.5) / len(self.x)

    def row_y(self, r):
        dh = self.frac(0, self.GH)[1]
        return self.GY - dh / 2 + (r + 0.5) * dh / len(self.W)

    def ring_row(self, sc, r, op, ink=INK):
        self.rect(sc, self.P((self.X0 + self.X1) / 2, self.row_y(r)), self.gw + 6,
                  self.GH / len(self.W) + 5, ink, op, border=ink, border_w=2.0, radius=2,
                  lift=1.2, fill_op=0.0)

    def tick(self, sc, fx, top, bottom, op):
        """A thin outline round one column of the input strip and the lit row: the two
        numbers that are multiplied."""
        h = self.span(0, bottom - top)[1]
        self.rect(sc, self.P(fx, (top + bottom) / 2), 5.0, h, INK, op, border=INK,
                  border_w=1.8, radius=1.5, lift=1.4, fill_op=0.0)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        cx, w = (self.X0 + self.X1) / 2, self.gw
        self.card(sc, cx, self.XY, w + 10, self.STRIP_H + 12, op, lift=0.1)
        self.strip(sc, self.x, cx, self.XY, w, self.STRIP_H, op, lo=DATA_LO, hi=DATA_HI)
        self.text(sc, self.X0 - 0.014, self.XY, "x", op, HEAD, "rm", ink=DATA_INK)
        g = op * self.kept(1)
        if g <= 0.01:
            return
        reveal = smoothstep(0.02, 0.3, u) if k == 1 else 1.0
        img = self.grid_rgb.copy()
        rows = int(np.ceil(len(img) * reveal - 1e-9))
        img[rows:] = PAPER
        self.rect(sc, self.P(cx, self.GY), w + 8, self.GH + 8, soft(WEIGHT, 0.85), g,
                  border=WEIGHT, border_w=1.6, radius=4, lift=0.1)
        draw_image(self, sc, self.P(cx, self.GY), img, w, self.GH, g, grid=0.0, lift=0.3)
        b = g * (smoothstep(0.4, 0.48, u) if k == 1 else 1.0)
        self.tiles(sc, self.b[:, None], self.BX, self.GY, self.BW, self.GH, b, ink=BIAS,
                   lo=self.blo, hi=self.bhi, grid=0.0)
        top = self.GY + self.frac(0, self.GH / 2)[1]
        self.text(sc, self.BX, top + 0.03, "bias", b, SIZE, ink=WEIGHT_INK)
        d = g * (smoothstep(0.28, 0.35, u) if k == 1 else 1.0)
        dh = self.frac(0, self.GH)[1]
        self.dims(sc, (self.X0 - 0.012, self.GY - dh / 2), (self.X0 - 0.012, self.GY + dh / 2),
                  "64", d, side=-1, ink=WEIGHT_INK)

    # ------------------------------------------------------------------ 1: the input
    def phase_1(self, sc, u, op):
        on = op * smoothstep(0.3, 0.38, u)
        self.big(sc, self.X0, 0.82, "784", on, label="numbers in x", size=BIG)
        d = op * smoothstep(0.55, 0.62, u)
        self.text(sc, self.X0 + 0.075, 0.82, "= 28 × 28", d, HEAD, "lm")

    # ------------------------------------------------------------------ 2: the weights
    def phase_2(self, sc, u, op):
        m = self.meta
        a = op * smoothstep(0.5, 0.58, u)
        self.big(sc, self.X0, 0.825, commas(m["weights"]), a, label="784 × 64 weights", size=BIG)
        b = op * smoothstep(0.66, 0.72, u)
        self.text(sc, 0.3, 0.825, "+ 64 biases", b, HEAD, "lm")

    # ------------------------------------------------------------------ 3: one output
    def phase_3(self, sc, u, op):
        m = self.meta
        j = self.j
        cx = (self.X0 + self.X1) / 2
        lit = op * smoothstep(0.02, 0.06, u)
        self.ring_row(sc, j, lit)
        # the lit row, copied up under x so that matching numbers line up
        on = op * smoothstep(0.05, 0.1, u)
        self.card(sc, cx, self.WY, self.gw + 10, self.STRIP_H + 12, on, lift=0.1,
                  fill=soft(WEIGHT, 0.85))
        self.strip(sc, self.W[j], cx, self.WY, self.gw, self.STRIP_H, on, ink=WEIGHT,
                   lo=self.wlo, hi=self.whi)
        self.text(sc, self.X0 - 0.014, self.WY, "row %d" % j, on, SIZE, "rm", ink=WEIGHT_INK)
        self.arrow(sc, (self.X1 + 0.004, self.row_y(j)), (self.X1 + 0.004, self.WY + 0.016),
                   AXIS, on * 0.8, u=eased(u, 0.05, 0.1), width=1.4)
        # three matching pairs, one after another
        top, bottom = self.XY - self.frac(0, 11)[1], self.WY + self.frac(0, 11)[1]
        fade = 1.0 - smoothstep(self.SWEEP[0] - 0.03, self.SWEEP[0], u)
        for n, (p, at) in enumerate(zip(m["pairs"], self.PAIR_AT)):
            a = op * smoothstep(at, at + 0.04, u)
            now = n == len(self.PAIR_AT) - 1 or u < self.PAIR_AT[n + 1]
            self.tick(sc, self.col_x(p["at"]), top, bottom, a * fade * (1.0 if now else 0.4))
            self.text(sc, 0.06, 0.745 + 0.045 * n, "%s × %s = %s" % (
                val(p["x"]), val(p["w"], 3), val(p["product"])), a, SIZE, "lm")
        # the sweep: every product added, left to right
        s0, s1 = self.SWEEP
        if u >= s0 - 0.02:
            i = int(round(eased(u, s0, s1) * (len(self.x) - 1)))
            sweep = op * smoothstep(s0 - 0.02, s0, u) * (1.0 - smoothstep(s1, s1 + 0.03, u))
            self.tick(sc, self.col_x(i), top, bottom, sweep)
            a = op * smoothstep(s0 - 0.02, s0, u)
            self.text(sc, 0.27, 0.745, "added: %s" % val(self.run[i]), a, HEAD, "lm")
        b = op * smoothstep(self.BIAS_AT, self.BIAS_AT + 0.04, u)
        self.rect(sc, self.P(self.BX, self.row_y(j)), self.BW + 8, self.GH / len(self.W) + 6,
                  INK, b, border=INK, border_w=2.0, radius=2, lift=1.2, fill_op=0.0)
        self.text(sc, 0.27, 0.792, "+ bias: %s" % val(m["bias"]), b, HEAD, "lm")
        c = op * smoothstep(self.OUT_AT, self.OUT_AT + 0.04, u)
        self.text(sc, 0.27, 0.855, "= %s" % val(m["out"]), c, BIG, "lm", strong=True)

    # ------------------------------------------------------------------ 4: 64 outputs
    def phase_4(self, sc, u, op):
        m = self.meta
        n = len(self.z)
        filled = int(np.clip(round(eased(u, 0.06, 0.42) * n), 1, n))
        moving = u < 0.45
        self.ring_row(sc, filled - 1 if moving else self.j,
                      op * (1.0 if moving else smoothstep(0.45, 0.5, u)))
        cx = (self.X0 + self.X1) / 2
        rgb = shades(self.z[None], DATA, self.zlo, self.zhi)
        rgb[:, filled:] = PAPER
        self.card(sc, cx, self.OY, self.gw + 10, self.STRIP_H + 12, op, lift=0.1)
        draw_image(self, sc, self.P(cx, self.OY), rgb, self.gw, self.STRIP_H, op, grid=0.3)
        self.text(sc, self.X0 - 0.014, self.OY, "out", op, SIZE, "rm", ink=DATA_INK)
        tile = self.gw / n
        jx = self.X0 + (self.X1 - self.X0) * (self.j + 0.5) / n
        lit = op * (filled > self.j)
        self.rect(sc, self.P(jx, self.OY), tile + 5, self.STRIP_H + 5, INK, lit, border=INK,
                  border_w=2.2, radius=2, lift=1.2, fill_op=0.0)
        self.text(sc, jx, self.OY - 0.042, val(m["out"]), lit, SIZE)
        self.running(sc, 0.3, self.OY - 0.042, [(0.0, "%d of %d" % (filled, n))], 0.0,
                     op * (1 - smoothstep(0.45, 0.5, u)), size=SIZE)
        b = op * smoothstep(0.5, 0.56, u)
        self.text(sc, self.X0, 0.865, "the batch: %s" % m["batch_out_text"], b, HEAD, "lm",
                  strong=True)


# ====================================================================== 189
class ReLU(Blocks):
    """The 64 numbers; negatives become 0; same shape, no weights; the whole batch."""

    TITLE = "nn.ReLU: negatives become 0"
    PHASES = (("64 numbers", 11.0), ("negatives to 0", 18.0), ("no weights", 11.0),
              ("the batch", 14.0))
    LAST = "nn.Linear: 784 numbers in, 64 out"
    NEXT = "A model is a stack of blocks"
    STATION = ("forward",) * 4
    X0, X1 = 0.075, 0.455
    BEFORE_Y, AFTER_Y = 0.42, 0.75
    ROW_H = 16.0
    BOX_Y, OUT_Y = 0.53, 0.64
    BOX_W, BOX_H = 42.0, 28.0
    GY, GH = 0.575, 100.0
    STEP = (0.06, 0.07)                # phase 2: when the first box turns, and the gap

    def setup(self):
        m = self.meta = self.study.metadata
        s = self.study
        self.z = np.asarray(s.panels[0].data["values"], float)
        self.h = np.asarray(s.panels[1].data["values"], float)
        self.batch = np.asarray(s.panels[2].data["values"], float)
        self.batch_zero = np.asarray(s.panels[2].data["zero"], bool)
        self.lo, self.hi = float(self.z.min()), float(self.z.max())
        self.win = list(m["window"])
        self.w = self.span(self.X1 - self.X0)[0]
        # phase 2: the eight boxes one after another, then the rest of the row
        a, gap = self.STEP
        self.turn = {i: a + gap * n for n, i in enumerate(self.win)}
        rest = [i for i in range(len(self.z)) if i not in self.turn]
        for n, i in enumerate(rest):
            self.turn[i] = 0.68 + 0.2 * n / len(rest)
        line = m["relu_line"]
        out = "%s → %s" % (m["shape_text"], m["shape_text"])
        self.CODE = (Code("model", run=((0.0, line - 1), (0.5, line))),
                     Code("model", run=((0.0, line),)),
                     Code("model", run=((0.0, line),), out=((0.35, out),)),
                     Code("model", run=((0.0, line),), out=((0.0, out),)))
        self.EVIDENCE = (exact("the trained model, on the first digit"),
                         exact("every negative number set to 0"),
                         exact("nn.ReLU has no parameters"),
                         measured("the first batch, and 60,000 digits"))

    def heading_for(self, k, u):
        return ("64 numbers out of nn.Linear", "below 0 becomes 0; the rest stay",
                "the same shape, and no weights",
                "the whole batch: %s" % self.meta["shape_text"])[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "These are the %d numbers nn.Linear gave the first digit; %d of them are below 0."
            % (m["size"], m["zeroed"]),
            "nn.ReLU() keeps every positive number and turns every negative one into 0: "
            "%d of %d here." % (m["zeroed"], m["size"]),
            "The shape stays %s, and nn.ReLU learns no numbers: it is a rule, not a grid."
            % m["shape_text"],
            "In the first batch %s of %s numbers become 0 (%s); over all 60,000 digits, %s."
            % (commas(m["batch_zeroed"]), commas(m["batch_numbers"]), pct(m["batch_share"]),
               pct(m["train_share"])),
        ][k]

    # ------------------------------------------------------------------ geometry
    def tile_x(self, i):
        return self.X0 + (self.X1 - self.X0) * (i + 0.5) / len(self.z)

    def box_x(self, n):
        span = self.frac(len(self.win) * (self.BOX_W + 4))[0]
        left = (self.X0 + self.X1) / 2 - span / 2
        return left + self.frac((n + 0.5) * (self.BOX_W + 4))[0]

    def done(self, k, u):
        """Which of the 64 have been through ReLU (the after row's filled tiles)."""
        if k < 1:
            return np.zeros(len(self.z), bool)
        if k > 1:
            return np.ones(len(self.z), bool)
        return np.array([u >= self.turn[i] for i in range(len(self.z))])

    def row(self, sc, values, fy, op, zero=None, shown=None):
        cx = (self.X0 + self.X1) / 2
        self.card(sc, cx, fy, self.w + 10, self.ROW_H + 12, op, lift=0.1)
        rgb = shades(values[None], DATA, self.lo, self.hi,
                     None if zero is None else zero[None])
        if shown is not None:
            rgb[:, ~shown] = PAPER
        draw_image(self, sc, self.P(cx, fy), rgb, self.w, self.ROW_H, op, grid=0.3)

    def ink_on(self, v):
        """Text on a tile of value ``v``: dark on the pale ones, paper on the deep ones."""
        return PAPER if (v - self.lo) / (self.hi - self.lo) > 0.62 else INK

    def ring(self, sc, fy, op):
        a, b = self.tile_x(self.win[0]), self.tile_x(self.win[-1])
        tile = self.w / len(self.z)
        self.rect(sc, self.P((a + b) / 2, fy), self.span(b - a)[0] + tile + 5, self.ROW_H + 6,
                  INK, op, border=INK, border_w=2.2, radius=3, lift=1.2, fill_op=0.0)
        return a - self.frac(tile / 2 + 2.5)[0], b + self.frac(tile / 2 + 2.5)[0]

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        self.row(sc, self.z, self.BEFORE_Y, op)
        self.text(sc, self.X0, self.BEFORE_Y - 0.05, "out of nn.Linear", op, SIZE, "lm",
                  ink=DATA_INK)
        after = op * self.kept(1)
        done = self.done(k, u)
        self.row(sc, self.h, self.AFTER_Y, after, zero=self.z <= 0, shown=done)
        self.text(sc, self.X0, self.AFTER_Y - 0.05, "out of nn.ReLU", after, SIZE, "lm",
                  ink=DATA_INK)
        # the eight read out, before and after, through phases 1 and 2
        boxes = op * self.kept(0, 1)
        if boxes > 0.01:
            self.window(sc, k, u, boxes, done)
        # how many became 0, through phases 2 and 3
        count = op * self.kept(1, 2)
        n = int((done & (self.z <= 0)).sum())
        self.running(sc, self.X0, 0.845, [(0.0, "%d of %d" % (n, len(self.z)))], 0.0, count,
                     label="became 0", size=HEAD)

    def window(self, sc, k, u, op, done):
        on = op * (smoothstep(0.2, 0.27, u) if k == 0 else 1.0)
        left, right = self.ring(sc, self.BEFORE_Y, on)
        boxes = op * (smoothstep(0.3, 0.37, u) if k == 0 else 1.0)
        y = self.BEFORE_Y + self.frac(0, self.ROW_H / 2 + 3)[1]
        bl = self.box_x(0) - self.frac(self.BOX_W / 2)[0]
        br = self.box_x(len(self.win) - 1) + self.frac(self.BOX_W / 2)[0]
        bt = self.BOX_Y - self.frac(0, self.BOX_H / 2)[1]
        for a, b in ((left, bl), (right, br)):
            self.arrow(sc, (a, y), (b, bt), AXIS, boxes * 0.7, head=False, width=1.3)
        for n, i in enumerate(self.win):
            v = self.z[i]
            fill = tuple(float(c) for c in shades([v], DATA, self.lo, self.hi)[0])
            bx = self.box_x(n)
            self.box(sc, bx, self.BOX_Y, self.BOX_W, self.BOX_H, fill, boxes, border=DATA,
                     radius=4, size=SIZE, text=val(v),
                     ink=self.ink_on(v))
            if not done[i]:
                continue
            t = 1.0 if k > 1 else smoothstep(self.turn[i], self.turn[i] + 0.03, u)
            a = op * t
            self.arrow(sc, (bx, self.BOX_Y + self.frac(0, self.BOX_H / 2 + 2)[1]),
                       (bx, self.OUT_Y - self.frac(0, self.BOX_H / 2 + 2)[1]), AXIS, a,
                       width=1.6)
            if v > 0:
                self.box(sc, bx, self.OUT_Y, self.BOX_W, self.BOX_H, fill, a, border=DATA,
                         radius=4, size=SIZE, text=val(v),
                         ink=self.ink_on(v))
            else:
                self.box(sc, bx, self.OUT_Y, self.BOX_W, self.BOX_H, ZERO, a, border=ZERO_INK,
                         radius=4, size=SIZE, text="0", ink=INK)

    # ------------------------------------------------------------------ 1: 64 numbers
    def phase_1(self, sc, u, op):
        a = op * smoothstep(0.5, 0.57, u)
        neg = int((self.z <= 0).sum())
        self.text(sc, self.X0, 0.72, "%d of them below 0" % neg, a, HEAD, "lm")

    # ------------------------------------------------------------------ 2: negatives to 0
    def phase_2(self, sc, u, op):
        # the after row's window ringed once its first tile is filled
        self.ring(sc, self.AFTER_Y, op * smoothstep(self.STEP[0], self.STEP[0] + 0.04, u))

    # ------------------------------------------------------------------ 3: no weights
    def phase_3(self, sc, u, op):
        tile = self.w / len(self.z)
        half = self.frac(tile / 2)[0]
        a = op * smoothstep(0.08, 0.15, u)
        for fy in (self.BEFORE_Y, self.AFTER_Y):
            y = fy + self.frac(0, self.ROW_H / 2 + 10)[1]
            self.dims(sc, (self.tile_x(0) - half, y), (self.tile_x(63) + half, y), "64", a,
                      side=1, gap=4)
        b = op * smoothstep(0.35, 0.42, u)
        self.big(sc, self.X0 + 0.2, 0.6, "0", b, label="numbers it learns", size=BIG)

    # ------------------------------------------------------------------ 4: the batch
    def phase_4(self, sc, u, op):
        m = self.meta
        cx = (self.X0 + self.X1) / 2
        reveal = smoothstep(0.02, 0.45, u)
        rows = int(np.ceil(len(self.batch) * reveal - 1e-9))
        rgb = shades(self.batch, DATA, self.lo, self.hi, self.batch_zero)
        rgb[rows:] = PAPER
        self.rect(sc, self.P(cx, self.GY), self.w + 8, self.GH + 8, soft(DATA, 0.82), op,
                  border=DATA, border_w=1.6, radius=4, lift=0.1)
        draw_image(self, sc, self.P(cx, self.GY), rgb, self.w, self.GH, op, grid=0.0)
        n = int(self.batch_zero[:rows].sum())
        self.running(sc, self.X0, 0.855, [(0.0, "%s of %s" % (commas(n), commas(
            m["batch_numbers"])))], 0.0, op, label="became 0", size=HEAD)
        b = op * smoothstep(0.5, 0.56, u)
        self.text(sc, 0.26, 0.855, "= " + pct(m["batch_share"]), b, BIG, "lm", strong=True)
        c = op * smoothstep(0.68, 0.74, u)
        self.text(sc, self.X0, 0.9, "all 60,000 digits: %s" % pct(m["train_share"]), c, SIZE,
                  "lm")


# ====================================================================== 190
class Stack(Blocks):
    """The blocks one by one; their shapes; print(model); their counts."""

    TITLE = "A model is a stack of blocks"
    PHASES = (("six blocks", 16.0), ("the shapes", 12.0), ("print(model)", 14.0),
              ("its weights", 15.0))
    LAST = "nn.ReLU: negatives become 0"
    NEXT = "nn.Conv2d: one small stamp"
    STATION = ("forward",) * 4
    BX, BW, BH = 0.087, 52.0, 24.0     # the blocks' column, as a stack
    ROW0, ROW_STEP = 0.40, 0.062
    NX, SX = 0.135, 0.33               # names, and the shapes / counts column
    PX, PBW = 0.072, 26.0              # beside print(model): the column, narrower
    P0, P_STEP = 0.372, 0.041         # print(model)'s lines
    APPEAR = tuple(0.05 + 0.1 * k for k in range(6))     # phase 1
    SHOW = tuple(0.08 + 0.1 * k for k in range(6))        # phase 2: each shape
    COUNT_AT = tuple(0.05 + 0.065 * k for k in range(6))  # phase 4: each count

    def setup(self):
        m = self.meta = self.study.metadata
        s = self.study
        self.kinds = m["kinds"]
        grids = [np.asarray(s.panels[2].data[k], float) for k in ("values", "second", "third")]
        self.tex = {}
        for k, w in zip((1, 3, 5), grids):
            lo, hi = float(w.min()), float(w.max())
            self.tex[k] = shrink_columns(shades(w, WEIGHT, lo, hi), int(self.BW))
        lines = m["printed_lines"]
        blocks = m["printed_blocks"]
        self.print_y = [self.P0 + self.P_STEP * i for i in range(len(lines))]
        self.entry = {}
        for k in range(6):
            ys = [y for y, b in zip(self.print_y, blocks) if b == k]
            self.entry[k] = (float(np.mean(ys)), len(ys))
        at = m["lines"]
        run1 = ((0.0, None),) + tuple((a, at[k]) for k, a in enumerate(self.APPEAR)) + \
            ((0.68, m["sequential_line"]),)
        run2 = ((0.0, at[0]),) + tuple((a, at[k]) for k, a in enumerate(self.SHOW)) + \
            ((0.68, m["sequential_line"]),)
        self.CODE = (Code("model", run=run1), Code("model", run=run2),
                     Code("print-model", run=((0.0, None), (0.08, 0))),
                     Code("count", run=((0.0, 1), (0.5, 2)),
                          out=((0.56, m["printed_sum"]),)))
        self.EVIDENCE = (exact("the six blocks of the course's model"),
                         exact("the shapes of the first batch"),
                         exact("what print(model) writes"),
                         exact("every weight and bias, counted"))

    def heading_for(self, k, u):
        m = self.meta
        if k == 3:
            return "only the Linear blocks learn numbers"
        return ("nn.Sequential: one block after another",
                "%s in, %s out" % (m["input_text"], m["shape_texts"][-1]),
                "print(model) lists the same blocks", "")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "nn.Sequential puts six blocks one after another: Flatten, then Linear and ReLU "
            "twice, then Linear.",
            "Each block hands the next a new shape, from %s down to %s."
            % (m["input_text"], m["shape_texts"][-1]),
            "print(model) writes the same six blocks, numbered 0 to 5, with their sizes.",
            "Only the three Linear blocks learn numbers: %s = %s."
            % (" + ".join(commas(c) for c in m["counts"] if c), commas(m["total"])),
        ][k]

    # ------------------------------------------------------------------ geometry
    def squeeze(self, k, u):
        """0: the blocks as a stack; 1: stretched beside print(model)'s lines."""
        if k != 2:
            return 0.0
        return eased(u, 0.0, 0.1) * (1.0 - eased(u, 0.88, 0.97))

    def block_at(self, i, s):
        """Block ``i``'s centre and size (layout x, y; px w, h) at squeeze ``s``."""
        y0, w0, h0 = self.ROW0 + self.ROW_STEP * i, self.BW, self.BH
        ey, n = self.entry[i]
        y1, w1, h1 = ey, self.PBW, self.span(0, self.P_STEP * n)[1] - 5.0
        return (self.BX + (self.PX - self.BX) * s, y0 + (y1 - y0) * s, w0 + (w1 - w0) * s,
                h0 + (h1 - h0) * s)

    def draw_block(self, sc, i, s, op):
        fx, fy, w, h = self.block_at(i, s)
        kind = self.kinds[i]
        c = self.P(fx, fy)
        if kind == "Linear":
            self.rect(sc, c, w + 6, h + 6, soft(WEIGHT, 0.85), op, border=WEIGHT, border_w=1.8,
                      radius=4, lift=0.2)
            draw_image(self, sc, c, self.tex[i], w, h, op, grid=0.0, lift=0.3)
        elif kind == "ReLU":
            self.rect(sc, c, w + 6, h + 6, ZERO, op, border=ZERO_INK, border_w=1.8, radius=4,
                      lift=0.2)
        else:
            self.card(sc, fx, fy, w + 6, h + 6, op, lift=0.2)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        s = self.squeeze(k, u)
        names = op * (1.0 - smoothstep(0.0, 0.3, s))
        shown = [op * (smoothstep(a, a + 0.04, u) if k == 0 else 1.0) for a in self.APPEAR]
        # the data's path through the stack, behind the blocks
        if shown[-1] > 0.01 or k > 0:
            _x, top, _w, _h = self.block_at(0, s)
            _x, bottom, _w, h = self.block_at(5, s)
            fx = self.BX + (self.PX - self.BX) * s
            self.arrow(sc, (fx, top), (fx, bottom + self.frac(0, h / 2 + 14)[1]), AXIS,
                       min(shown[-1], op), width=1.6, lift=0.0)
        for i in range(6):
            if shown[i] <= 0.01:
                continue
            self.draw_block(sc, i, s, shown[i])
            _fx, fy, _w, _h = self.block_at(i, 0.0)
            self.text(sc, self.NX, fy, m["names"][i], min(shown[i], names), SIZE, "lm")
        # nn.Sequential: one outline round them all
        seq = op * (smoothstep(0.68, 0.72, u) if k == 0 else 1.0)
        _x, top, _w, h0 = self.block_at(0, s)
        _x, bottom, _w, h1 = self.block_at(5, s)
        w = self.BW + (self.PBW - self.BW) * s + 22
        pad = 9.0 + 8.0 * s
        dy = self.frac(0, h0 / 2 + pad)[1], self.frac(0, h1 / 2 + pad)[1]
        cy = (top - dy[0] + bottom + dy[1]) / 2
        hh = self.span(0, bottom + dy[1] - top + dy[0])[1]
        fx = self.BX + (self.PX - self.BX) * s
        self.rect(sc, self.P(fx, cy), w, hh, INK, seq, border=INK, border_w=2.0, radius=8,
                  lift=0.9, fill_op=0.0)

    # ------------------------------------------------------------------ 1: six blocks
    def phase_1(self, sc, u, op):
        a = op * smoothstep(0.72, 0.77, u)
        self.text(sc, self.NX, 0.81, "one model: 6 blocks", a, HEAD, "lm", strong=True)

    # ------------------------------------------------------------------ 2: the shapes
    def phase_2(self, sc, u, op):
        m = self.meta
        for i, at in enumerate(self.SHOW):
            a = op * smoothstep(at, at + 0.04, u)
            _fx, fy, _w, _h = self.block_at(i, 0.0)
            self.text(sc, self.SX, fy, m["shape_texts"][i], a, SIZE, "lm", ink=DATA_INK)
        b = op * smoothstep(0.68, 0.73, u)
        self.big(sc, self.NX, 0.84, " → ".join(str(v) for v in m["sizes"]), b,
                 label="numbers per digit", size=HEAD)

    # ------------------------------------------------------------------ 3: print(model)
    def phase_3(self, sc, u, op):
        lines = self.meta["printed_lines"]
        fx = self.PX + self.frac(self.PBW / 2 + 22)[0]
        for i, (s, y) in enumerate(zip(lines, self.print_y)):
            a = op * smoothstep(0.12 + 0.02 * i, 0.16 + 0.02 * i, u) * (
                1.0 - smoothstep(0.83, 0.87, u))
            self.label(sc, self.P(fx, y), s, SIZE, anchor="lm", opacity=a, ink=INK, kind="mono")

    # ------------------------------------------------------------------ 4: its weights
    def phase_4(self, sc, u, op):
        m = self.meta
        total = 0
        steps = [(0.0, "0")]
        for i, at in enumerate(self.COUNT_AT):
            a = op * smoothstep(at, at + 0.04, u)
            _fx, fy, _w, _h = self.block_at(i, 0.0)
            c = m["counts"][i]
            self.text(sc, self.SX, fy, commas(c), a, SIZE if c == 0 else HEAD, "lm",
                      ink=ZERO_INK if c == 0 else WEIGHT_INK, strong=False)
            total += c
            steps.append((at, commas(total)))
        self.running(sc, self.NX, 0.835, steps, u, op * smoothstep(0.03, 0.07, u),
                     label="numbers the model learns", size=BIG)
        c = op * smoothstep(0.62, 0.68, u)
        self.text(sc, 0.055, 0.895, "as in “%s”" % m["counted"], c, SIZE, "lm")


ANIMATIONS = {"blocks-a-linear-layer": ALinear, "blocks-relu": ReLU,
              "blocks-a-model-is-a-stack": Stack}

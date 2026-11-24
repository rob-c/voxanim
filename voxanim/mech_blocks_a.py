"""Animations 185-187, the first of the "Blocks and code" series (see :mod:`voxanim.blocks`):
a tensor, a batch and ``nn.Flatten``, drawn from :mod:`voxanim.study_blocks_a`.

* :class:`ATensor` (``blocks-a-tensor``): ``x, y = train_set[0]``: one digit as a blue card
  of 28 x 28 tiles, three of its rows read out number by number, its shape ``[1, 28, 28]``
  and its 784 numbers.
* :class:`ABatch` (``blocks-a-batch``): the loader's first batch: 64 digit cards slide one
  behind the other into one block, their labels into a row, the shape ``[64, 1, 28, 28]``
  and ``[64]``, and how many of each digit the shuffled batch holds.
* :class:`Flatten` (``blocks-flatten``): ``nn.Flatten()``: the square's rows copied one by
  one into a strip of 784, where pixel (row r, column c) lands (``28 r + c``), and the
  whole batch as ``[64, 784]``, with no weights.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). Everything that stays across phases is drawn
in ``steady``, so nothing blanks between them.
"""

from __future__ import annotations

import numpy as np

from . import study_course as course
from .blocks import (
    BIG,
    DATA,
    DATA_HI,
    DATA_LO,
    HEAD,
    HUGE,
    INK,
    SIZE,
    Blocks,
    Code,
    commas,
    exact,
    shades,
    shape,
    torch_size,
    val,
)
from .easing import ease_in_out, smoothstep
from .mech import AXIS, PAPER, dark, soft

LO, HI = DATA_LO, DATA_HI       # paper and full ink, as Normalize leaves them
DATA_INK = dark(DATA, 0.3)
RIGHT = 0.31                    # the picture's right-hand column of words and numbers


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


class Digits(Blocks):
    """What the three share (the digit card and its tiles are the series', in
    :class:`~voxanim.blocks.Blocks`)."""


# ====================================================================== 185
class ATensor(Digits):
    """One digit, its numbers, its shape."""

    TITLE = "A tensor is a block of numbers"
    PHASES = (("one digit", 12.0), ("numbers", 15.0), ("its shape", 14.0),
              ("a tensor", 11.0))
    NEXT = "64 digits in one block"
    STATION = ("data",) * 4
    EVIDENCE = (exact("train_set[0], as the script loads it"),) * 4
    CARD = (0.19, 0.585)
    TILE = 6.25

    def setup(self):
        m = self.meta = self.study.metadata
        self.img = np.asarray(self.study.panels[0].data["values"][0], float)
        self.patch = np.asarray(m["patch"], float)
        self.zoom = tuple(m["zoom"])
        self.cell = self.digit_cells(*self.CARD, self.TILE)
        self.CODE = (Code("one-digit", run=((0.0, None), (0.3, 0))),
                     Code("one-digit", run=((0.0, 0),)),
                     Code("one-digit", run=((0.0, 0), (0.08, 1)), out=((0.2, m["printed"]),)),
                     Code("one-digit", run=((0.0, 1),), out=((0.0, m["printed"]),)))

    def heading_for(self, k, u):
        return ("one digit from the training set", "every tile is one number",
                "its shape: " + self.meta["shape_text"],
                "a tensor: numbers in a block, with a shape")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "x is the first training digit: 28 rows of 28 tiles. y is its label, %d."
            % m["label"],
            "Each tile is one number: %s where the paper is, up to %s where the ink is darkest."
            % (val(m["paper"]), val(m["high"])),
            "Its shape is %s: 1 channel, 28 rows, 28 columns, so %s numbers."
            % (m["shape_text"], m["product"]),
            "A tensor is a block of numbers with a shape. This one holds %s." % commas(
                m["numbers"]),
        ][k]

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        cx, cy = self.CARD
        self.digit_card(sc, self.img, cx, cy, self.TILE, op)
        side = 28 * self.TILE
        dw, dh = self.frac(side, side)
        top, left = cy + self.frac(0, 3.0)[1] - dh / 2, cx - dw / 2
        self.text(sc, left, top - 0.05, "x", op * smoothstep(0.25, 0.32, u) if k == 0 else op,
                  HEAD, "lm", ink=DATA_INK)
        if k < 2:
            return
        a = op if k > 2 else op * smoothstep(0.12, 0.2, u)
        rows = op if k > 2 else op * smoothstep(0.38, 0.45, u)
        cols = op if k > 2 else op * smoothstep(0.5, 0.57, u)
        self.dims(sc, (left - 0.018, top), (left - 0.018, top + dh), "28", rows, side=-1)
        self.dims(sc, (left, top + dh + 0.03), (left + dw, top + dh + 0.03), "28", cols)
        self.text(sc, RIGHT, 0.45, self.meta["shape_text"], a, HEAD, "lm", strong=True)
        for i, (s, on) in enumerate((("1 channel", a), ("28 rows", rows),
                                     ("28 columns", cols))):
            self.text(sc, RIGHT, 0.51 + 0.045 * i, s, on, SIZE, "lm")
        total = op if k > 2 else op * smoothstep(0.68, 0.75, u)
        self.big(sc, RIGHT, 0.75, commas(self.meta["numbers"]), total, label="1 × 28 × 28 =",
                 size=BIG)

    # ------------------------------------------------------------------ 1: one digit
    def phase_1(self, sc, u, op):
        on = op * smoothstep(0.45, 0.52, u)
        self.big(sc, RIGHT, 0.62, str(self.meta["label"]), on, label="y: its label",
                 size=HUGE, ink=DATA_INK)

    # ------------------------------------------------------------------ 2: numbers
    def phase_2(self, sc, u, op):
        r0, c0 = self.zoom
        n = len(self.patch)
        ring = op * smoothstep(0.05, 0.12, u)
        _x0, y0, x1, y1 = self.outline(sc, self.cell, r0, c0, n, n, self.TILE, ring)
        # the window, magnified: one box per tile, its number in it
        bw, bh = 46.0, 32.0
        mx, my = RIGHT + self.frac(n * bw / 2)[0], 0.55
        dw, dh = self.frac(n * bw, n * bh)
        grow = op * smoothstep(0.15, 0.25, u)
        for a, b in ((x1, y0), (x1, y1)):
            self.arrow(sc, (a, b), (mx - dw / 2, my - dh / 2 if b == y0 else my + dh / 2),
                       AXIS, grow * 0.8, head=False, width=1.4)
        fills = shades(self.patch, DATA, LO, HI)
        for i in range(n):
            for j in range(n):
                # one row of numbers at a time
                on = op * smoothstep(0.25 + 0.14 * i, 0.3 + 0.14 * i, u)
                cx = mx - dw / 2 + (j + 0.5) * dw / n
                cy = my - dh / 2 + (i + 0.5) * dh / n
                v = self.patch[i, j]
                fill = tuple(float(c) for c in fills[i, j])
                self.box(sc, cx, cy, bw - 3, bh - 3, fill, on, border=DATA, radius=3,
                         size=SIZE, text=val(v), ink=INK if v < (LO + HI) / 2 else PAPER)
        key = op * smoothstep(0.75, 0.8, u)
        self.text(sc, RIGHT, my + dh / 2 + 0.05, "%s: paper" % val(LO), key, SIZE, "lm")
        self.text(sc, RIGHT, my + dh / 2 + 0.095, "%s: darkest ink" % val(HI), key, SIZE, "lm")

    # ------------------------------------------------------------------ 3: its shape
    def phase_3(self, sc, u, op):
        """The block is outlined while its shape is measured; the shape, the counts and the
        total stay into the next phase: they are drawn in ``steady``."""
        self.card_outline(sc, op * smoothstep(0.05, 0.12, u) * (1 - smoothstep(0.3, 0.38, u)))

    def card_outline(self, sc, op):
        cx, cy = self.CARD
        side = 28 * self.TILE
        self.rect(sc, self.P(cx, cy), side + 24, side + 30, INK, op, border=INK, border_w=2.4,
                  radius=9, lift=1.0, fill_op=0.0)

    # ------------------------------------------------------------------ 4: a tensor
    def phase_4(self, sc, u, op):
        cx, _cy = self.CARD
        a = op * smoothstep(0.08, 0.16, u)
        self.card_outline(sc, a)
        self.text(sc, cx, 0.875, "the numbers", a, SIZE, strong=True)
        b = op * smoothstep(0.35, 0.43, u)
        w = self.span(0.13)[0]
        self.rect(sc, self.P(RIGHT + 0.06, 0.45), w, 34, INK, b, border=INK, border_w=2.4,
                  radius=6, lift=1.0, fill_op=0.0)
        self.text(sc, RIGHT + 0.06, 0.395, "the shape", b, SIZE, strong=True)


# ====================================================================== 186
class ABatch(Digits):
    """One digit; 64 cards into one block; their labels; the shape; what is in it."""

    TITLE = "A batch: many digits in one block"
    PHASES = (("one digit", 10.0), ("64 digits", 16.0), ("64 labels", 11.0),
              ("its shape", 12.0), ("a mix", 9.0))
    LAST = "one digit is a block [1, 28, 28]"
    NEXT = "Flatten: a square becomes a strip"
    STATION = ("data",) * 5
    FRONT = (0.155, 0.615)
    CARD_PX, TILE = 100.0, 3.2
    DEPTH = 1.4                       # px between one card and the next behind it
    SRC = (0.425, 0.44)
    PILE = 64.0                       # the training set's cards, px
    SHOWN = 10                        # labels read out in boxes
    ROW_Y, BOX_Y = 0.868, 0.805
    ROW = (0.06, 0.465)

    def setup(self):
        m = self.meta = self.study.metadata
        imgs = np.asarray(self.study.panels[0].data["values"], float)
        self.imgs = (imgs - course.MEAN) / course.STD            # the normalised tiles
        self.labels = np.asarray(m["labels"])
        self.counts = np.asarray(m["counts"])
        n = len(self.labels)
        # the cards' arrivals in phase 2: three slowly, then the rest as a stream
        slow = [0.14 + 0.12 * i for i in range(3)]
        fast = [0.47 + 0.43 * ease_in_out(float(v)) ** 0.8 for v in np.linspace(0, 1, n - 4)]
        self.arrive = [0.0] + slow + fast                       # card 0 is there already
        self.travel = [0.0] + [0.1] * 3 + [0.025] * (n - 4)
        self.fill = list(0.06 + 0.5 * np.linspace(0, 1, n) ** 1.6)   # labels, phase 3
        self.CODE = (Code("loader", run=((0.0, 0), (0.45, 1))),
                     Code("loader", run=((0.0, 1),)),
                     Code("loader", run=((0.0, 1),)),
                     Code("one-batch", run=((0.0, 0), (0.25, 1)),
                          out=((0.35, "%s\n%s" % (torch_size(m["shape"]),
                                                   torch_size(m["label_shape"]))),)),
                     Code("loader", run=((0.0, 2),)))
        self.EVIDENCE = (exact("the first batch: 64 shuffled digits"),) * 4 + \
            (exact("digits counted in the first batch"),)

    def heading_for(self, k, u):
        return ("batch_size=64: take 64 at once", "64 digits slide into one block",
                "and their 64 labels, in a row", "one block: " + self.meta["shape_text"],
                "shuffle=True: every digit, mixed")[k]

    def subtitle_for(self, k):
        m = self.meta
        c = m["counts"]
        return [
            "One digit is a block of shape %s. The loader takes %d of them at a time."
            % (m["one_shape"], m["batch"]),
            "The first batch: %d digits, one behind the other, make one block." % m["batch"],
            "Their %d labels make a row: %s ..." % (m["batch"], ", ".join(
                str(v) for v in m["labels"][:self.SHOWN])),
            "x has shape %s and y has shape %s: %s numbers in one block."
            % (m["shape_text"], m["label_shape_text"], m["product"]),
            "This shuffled batch holds every digit: %d ones, but only %d sixes and %d eights."
            % (c[1], c[6], c[8]),
        ][k]

    # ------------------------------------------------------------------ the stack
    def card_at(self, i, u_phase2, k):
        """Where card ``i`` is: ``(fx, fy, on, scale)``: its slot once it has arrived, on its
        way from the training set before (growing from the pile's size), nowhere before it
        leaves."""
        fx, fy = self.FRONT
        dx, dy = self.frac(self.DEPTH * i, -self.DEPTH * i)
        slot = (fx + dx, fy + dy)
        if k >= 2 or i == 0:
            return slot[0], slot[1], 1.0, 1.0
        if k < 1:
            return slot[0], slot[1], 0.0, 1.0
        a = self.arrive[i]
        s = eased(u_phase2, a - self.travel[i], a)
        if u_phase2 < a - self.travel[i]:
            return slot[0], slot[1], 0.0, 1.0
        sx, sy = self.SRC
        small = self.PILE / (28 * self.TILE + 10)
        return sx + (slot[0] - sx) * s, sy + (slot[1] - sy) * s, 1.0, small + (1 - small) * s

    def draw_stack(self, sc, k, u, op):
        """The cards, back to front: each one's lift orders it."""
        side = 28 * self.TILE
        for i in reversed(range(len(self.labels))):
            fx, fy, on, z = self.card_at(i, u if k == 1 else 1.0, k)
            if on <= 0:
                continue
            lift = 0.9 - 0.012 * i
            self.card(sc, fx, fy, (side + 10) * z, (side + 16) * z, op, lift=lift)
            self.tiles(sc, self.imgs[i], fx, fy + self.frac(0, 2.5 * z)[1], side * z, side * z,
                       op, lo=LO, hi=HI, lift=lift + 0.005, grid=0.0)

    def count_in(self, k, u):
        if k == 0:
            return 1
        if k > 1:
            return len(self.labels)
        return 1 + sum(u >= a for a in self.arrive[1:])

    def steady(self, sc, k, u, op):
        self.draw_stack(sc, k, u, op)
        # the training set the cards come from
        src = op * self.kept(0, 1)
        sx, sy = self.SRC
        for i in reversed(range(5)):
            dx, dy = self.frac(2.0 * i, -2.0 * i)
            self.card(sc, sx + dx, sy + dy, self.PILE, self.PILE + 6, src, lift=0.1 - 0.01 * i)
        self.text(sc, sx, sy + 0.095, "train_set", src, SIZE, ink=DATA_INK)
        # how many are in the block, all along
        n = self.count_in(k, u)
        self.running(sc, RIGHT, 0.72, [(0.0, str(n))], 0.0, op * self.kept(1, 3),
                     label="digits", size=BIG)
        # the labels' row, once it is there
        if k >= 2:
            self.label_row(sc, 1.0 if k > 2 else u, op)
        side = 28 * self.TILE
        fx, fy = self.FRONT
        self.text(sc, fx, fy + self.frac(0, side / 2 + 26)[1], self.meta["one_shape"],
                  op * self.kept(0, 0), SIZE, ink=DATA_INK)

    def label_row(self, sc, u, op):
        """The labels: a row of 64 tiles under the block, the first ten read out."""
        n = len(self.labels)
        x0, x1 = self.ROW
        filled = sum(u >= f for f in self.fill)
        w = self.span(x1 - x0)[0]
        cell = self.tiles(sc, self.labels[None, :].astype(float), (x0 + x1) / 2, self.ROW_Y, w,
                          14.0, op, lo=0, hi=9, reveal=1.0, grid=0.3)
        if filled < n:          # the tiles not yet filled, covered
            a = x0 + (x1 - x0) * filled / n
            cw = self.span(x1 - a)[0]
            self.rect(sc, self.P((a + x1) / 2, self.ROW_Y), cw + 1, 16.0, PAPER, op, radius=0,
                      lift=0.6, border=soft(DATA, 0.4), border_w=1.0)
        bw = 30.0
        for i in range(self.SHOWN):
            on = op * (filled > i)
            bx = x0 + self.frac(bw / 2 + i * (bw + 3))[0]
            self.box(sc, bx, self.BOX_Y, bw, 26.0, soft(DATA, 0.82), on, border=DATA,
                     text=str(self.labels[i]), size=SIZE, radius=4)
            tx, _ty = cell(0, i)
            self.arrow(sc, (bx, self.BOX_Y + self.frac(0, 14)[1]),
                       (tx, self.ROW_Y - self.frac(0, 8)[1]), AXIS, on * 0.8, head=False,
                       width=1.2)
        more = x0 + self.frac(self.SHOWN * (bw + 3) + 8)[0]
        self.text(sc, more, self.BOX_Y, "y, all %d" % n if filled >= n else "y",
                  op * (filled > self.SHOWN), SIZE, "lm", ink=DATA_INK)

    # ------------------------------------------------------------------ 1: one digit
    def phase_1(self, sc, u, op):
        # the first card is already in its place; the arrow says where the others go
        on = op * smoothstep(0.5, 0.58, u)
        sx, sy = self.SRC
        self.arc(sc, (sx - 0.05, sy - 0.02), (0.29, 0.42), AXIS, on, eased(u, 0.5, 0.7),
                 bend=0.04, width=1.8)
        self.text(sc, RIGHT, 0.62, "64 at a time", on, SIZE, "lm", strong=True)

    # ------------------------------------------------------------------ 2: 64 digits
    def phase_2(self, sc, u, op):
        sx, sy = self.SRC
        self.arc(sc, (sx - 0.05, sy - 0.02), (0.29, 0.42), AXIS, op * 0.6,
                 bend=0.04, width=1.8)

    # ------------------------------------------------------------------ 3: 64 labels
    def phase_3(self, sc, u, op):
        filled = sum(u >= f for f in self.fill)
        self.running(sc, RIGHT, 0.52, [(0.0, str(filled))], 0.0, op * smoothstep(0.04, 0.08, u),
                     label="labels", size=BIG)

    # ------------------------------------------------------------------ 4: its shape
    def phase_4(self, sc, u, op):
        m = self.meta
        side = 28 * self.TILE
        fx, fy = self.FRONT
        dw, dh = self.frac(side, side)
        top = fy + self.frac(0, 2.5)[1] - dh / 2
        left = fx - dw / 2
        d = op * smoothstep(0.05, 0.12, u)
        r = op * smoothstep(0.15, 0.22, u)
        c = op * smoothstep(0.25, 0.32, u)
        back = self.frac(self.DEPTH * 63, -self.DEPTH * 63)
        ex, ey = left + dw + self.frac(12)[0], top + dh + self.frac(0, 10)[1]
        self.dims(sc, (ex, ey), (ex + back[0], ey + back[1]), "64", d, side=1, gap=12)
        self.dims(sc, (left - 0.016, top), (left - 0.016, top + dh), "28", r, side=-1)
        self.dims(sc, (left, top + dh + 0.025), (left + dw, top + dh + 0.025), "28", c,
                  side=1, gap=8)
        s = op * smoothstep(0.4, 0.47, u)
        self.text(sc, RIGHT, 0.43, m["shape_text"], s, HEAD, "lm", strong=True)
        p = op * smoothstep(0.6, 0.67, u)
        self.big(sc, RIGHT, 0.57, commas(m["numbers"]), p, label="64 × 784 =", size=BIG)

    # ------------------------------------------------------------------ 5: a mix
    def phase_5(self, sc, u, op):
        top = max(self.counts)
        x = RIGHT + 0.02
        for d in range(10):
            on = op * smoothstep(0.05 + 0.03 * d, 0.09 + 0.03 * d, u)
            y = 0.385 + 0.038 * d
            self.text(sc, RIGHT, y, str(d), on, SIZE, "lm")
            self.hbar(sc, x + 0.006, y, 70.0 * self.counts[d] / top, DATA, on,
                      value=str(self.counts[d]), thick=14.0)


# ====================================================================== 187
class Flatten(Digits):
    """The square; its rows copied into a strip; where each tile goes; the batch."""

    TITLE = "Flatten: a square becomes a strip"
    PHASES = (("the square", 10.0), ("unroll", 18.0), ("where it goes", 12.0),
              ("the batch", 12.0))
    LAST = "a batch: [64, 1, 28, 28]"
    NEXT = "nn.Linear: a grid of weights"
    STATION = ("forward",) * 4
    SQ = (0.16, 0.555)
    TILE = 6.0
    STRIP = (0.06, 0.465)
    STRIP_Y, STRIP_H = 0.868, 18.0
    MAG_Y = 0.785

    def setup(self):
        m = self.meta = self.study.metadata
        self.img = np.asarray(self.study.panels[0].data["values"][0], float)
        self.flat = np.asarray(self.study.panels[1].data["values"][0], float)
        self.batch = np.asarray(self.study.panels[2].data["values"], float)
        self.pixel = tuple(m["pixel"])
        self.cell = self.digit_cells(*self.SQ, self.TILE)
        line = m["flatten_line"]
        out = "%s → %s" % (m["batch_in_text"], m["batch_out_text"])
        self.CODE = (Code("model", run=((0.0, 0), (0.35, line))),
                     Code("model", run=((0.0, line),)),
                     Code("model", run=((0.0, line),)),
                     Code("model", run=((0.0, line),), out=((0.25, out),)))
        self.EVIDENCE = (exact("the same 784 numbers, row after row"),) * 3 + \
            (exact("the first batch, flattened"),)
        # phase 2: when each row leaves the square, and how long it takes to reach the
        # strip: row 0 slowly, the blank rows 1-4 quicker, the first rows with ink (5, 6)
        # slowly again, then the rest as a stream
        self.leave = [0.04] + [0.18 + 0.035 * i for i in range(4)] + [0.34, 0.52] + \
            [0.66 + 0.22 * i / 20 for i in range(21)]
        self.trip = [0.1] + [0.04] * 4 + [0.1, 0.1] + [0.04] * 21
        self.slow = (0, 5, 6)

    def heading_for(self, k, u):
        return ("the model's first block: nn.Flatten()", "row by row, into one strip",
                "each tile has a place in the strip",
                "the batch: %s, no weights" % self.meta["batch_out_text"])[k]

    def subtitle_for(self, k):
        m = self.meta
        r, c = m["pixel"]
        return [
            "nn.Flatten() is the first block of the model. Its input: a 28 × 28 digit.",
            "It copies the rows one after another: 28 rows of 28 make one strip of %d."
            % m["length"],
            "The tile in row %d, column %d goes to place %s. Its number, %s, stays the same."
            % (r, c, m["place_sum"], val(m["value"])),
            "Every digit of the batch becomes a strip: %s becomes %s. Flatten learns %d "
            "numbers." % (m["batch_in_text"], m["batch_out_text"], m["weights"]),
        ][k]

    # ------------------------------------------------------------------ geometry
    def seg(self):
        """The layout width of one row's piece of the strip."""
        return (self.STRIP[1] - self.STRIP[0]) / 28

    def rows_in(self, k, u):
        """How many rows have reached the strip."""
        if k < 1:
            return 0
        if k > 1:
            return 28
        return sum(u >= a + t for a, t in zip(self.leave, self.trip))

    def steady(self, sc, k, u, op):
        sq = op * self.kept(0, 2)
        cx, cy = self.SQ
        self.digit_card(sc, self.img, cx, cy, self.TILE, sq)
        # the strip, as far as it has come
        st = op * self.kept(1)
        x0, x1 = self.STRIP
        w = self.span(x1 - x0)[0]
        self.card(sc, (x0 + x1) / 2, self.STRIP_Y, w + 12, self.STRIP_H + 16, st, lift=0.1)
        n = self.rows_in(k, u)
        if n:
            part = self.seg() * n
            self.strip(sc, self.flat[:28 * n], x0 + part / 2, self.STRIP_Y,
                       self.span(part)[0], self.STRIP_H, st, lo=LO, hi=HI, lift=0.2)
        if n == 28:
            name = "a strip of %d: %s" % (self.meta["length"], shape([self.meta["length"]]))
        else:
            name = "a strip of %d" % (28 * n) if n else "a strip"
        self.text(sc, x0, self.STRIP_Y + 0.058, name, st, SIZE, "lm", ink=DATA_INK)

    # ------------------------------------------------------------------ 1: the square
    def phase_1(self, sc, u, op):
        on = op * smoothstep(0.45, 0.52, u)
        side = 28 * self.TILE
        cx, cy = self.SQ
        dw, dh = self.frac(side, side)
        top = cy + self.frac(0, 3.0)[1] - dh / 2
        self.dims(sc, (cx + dw / 2 + 0.018, top), (cx + dw / 2 + 0.018, top + dh), "28 rows",
                  on, side=1)
        self.dims(sc, (cx - dw / 2, top + dh + 0.035), (cx + dw / 2, top + dh + 0.035),
                  "28 columns", on * smoothstep(0.6, 0.67, u), side=1, gap=8)

    # ------------------------------------------------------------------ 2: unroll
    def phase_2(self, sc, u, op):
        side = 28 * self.TILE
        cx, cy = self.SQ
        dh = self.frac(0, side)[1]
        top = cy + self.frac(0, 3.0)[1] - dh / 2
        x0 = self.STRIP[0]
        sw = self.seg()
        for r, (a, trip) in enumerate(zip(self.leave, self.trip)):
            if not a <= u <= a + trip + (0.06 if r in self.slow else 0.0):
                continue
            s = eased(u, a, a + trip)
            sy = top + (r + 0.5) * dh / 28
            ex, ey = x0 + (r + 0.5) * sw, self.STRIP_Y
            fx, fy = cx + (ex - cx) * s, sy + (ey - sy) * s
            w = side + (self.span(sw)[0] - side) * s
            h = self.TILE + (self.STRIP_H - self.TILE) * s
            row = self.img[r][None, :]
            self.tiles(sc, row, fx, fy, w, h, op, lo=LO, hi=HI, lift=1.2, grid=0.0)
            self.rect(sc, self.P(fx, fy), w + 4, h + 4, INK, op, border=INK, border_w=2.0,
                      radius=2, lift=1.3, fill_op=0.0)
            # the row it came from, outlined on the square
            self.rect(sc, self.P(cx, sy), side + 4, self.TILE + 4, INK, op * (1 - s), border=INK,
                      border_w=2.0, radius=2, lift=1.0, fill_op=0.0)
            if r in self.slow:
                self.text(sc, RIGHT + 0.03, 0.43, "row %d" % r, op * (1 - smoothstep(
                    a + trip + 0.03, a + trip + 0.06, u)), HEAD, "lm", strong=True)
        n = self.rows_in(1, u)
        self.running(sc, RIGHT + 0.03, 0.62, [(0.0, str(28 * n))], 0.0, op,
                     label="numbers in it", size=BIG)
        self.text(sc, RIGHT + 0.03, 0.70, "28 × 28 = 784", op * smoothstep(0.9, 0.94, u),
                  SIZE, "lm")

    # ------------------------------------------------------------------ 3: where it goes
    def phase_3(self, sc, u, op):
        m = self.meta
        r, c = self.pixel
        ring = op * smoothstep(0.05, 0.12, u)
        self.outline(sc, self.cell, r, c, 1, 1, self.TILE, ring, lift=1.1)
        # row r of the strip, magnified to the strip's width
        x0, x1 = self.STRIP
        sw = self.seg()
        a, b = x0 + r * sw, x0 + (r + 1) * sw
        mag = op * smoothstep(0.18, 0.26, u)
        w = self.span(x1 - x0)[0]
        cell = self.tiles(sc, self.img[r][None, :], (x0 + x1) / 2, self.MAG_Y, w, 14.0, mag,
                          lo=LO, hi=HI, grid=0.35)
        top = self.STRIP_Y - self.frac(0, self.STRIP_H / 2 + 2)[1]
        self.arrow(sc, (a, top), (x0, self.MAG_Y + 0.016), AXIS, mag, head=False, width=1.4)
        self.arrow(sc, (b, top), (x1, self.MAG_Y + 0.016), AXIS, mag, head=False, width=1.4)
        self.rect(sc, self.P((a + b) / 2, self.STRIP_Y), self.span(sw)[0] + 4,
                  self.STRIP_H + 4, INK, mag, border=INK, border_w=2.0, radius=2, lift=1.2,
                  fill_op=0.0)
        hit = op * smoothstep(0.32, 0.38, u)
        px, py = cell(0, c)
        self.rect(sc, self.P(px, py), w / 28 + 7, 22.0, INK, hit,
                  border=INK, border_w=2.4, radius=2, lift=1.2, fill_op=0.0)
        self.text(sc, RIGHT + 0.03, 0.43, "row %d, column %d" % (r, c), ring, SIZE, "lm")
        self.text(sc, RIGHT + 0.03, 0.49, "%d × 28 + %d" % (r, c), hit, HEAD, "lm")
        self.big(sc, RIGHT + 0.03, 0.575, "= %d" % m["place"], op * smoothstep(0.45, 0.52, u),
                 size=BIG)
        self.text(sc, RIGHT + 0.03, 0.66, "same number: %s" % val(m["value"]),
                  op * smoothstep(0.62, 0.68, u), SIZE, "lm")

    # ------------------------------------------------------------------ 4: the batch
    def phase_4(self, sc, u, op):
        m = self.meta
        x0, x1 = self.STRIP
        w = self.span(x1 - x0)[0]
        on = op * smoothstep(0.05, 0.14, u)
        reveal = smoothstep(0.08, 0.3, u)
        self.card(sc, (x0 + x1) / 2, 0.52, w + 12, 146.0, on, lift=0.1)
        rgb = shades(self.batch, DATA, LO, HI)
        rows = max(1, int(np.ceil(len(rgb) * reveal)))
        img = rgb.copy()
        img[rows:] = PAPER
        from .diagram import draw_image, shrink_columns
        draw_image(self, sc, self.P((x0 + x1) / 2, 0.525), shrink_columns(img, int(w)), w, 128.0,
                   on, grid=0.0)
        self.text(sc, x0, 0.36, "64 strips, one per digit", on, SIZE, "lm", ink=DATA_INK)
        self.text(sc, x1, 0.36, m["batch_out_text"], op * smoothstep(0.3, 0.36, u), HEAD, "rm",
                  strong=True)
        z = op * smoothstep(0.5, 0.57, u)
        self.big(sc, x0, 0.75, "0", z, label="numbers Flatten learns", size=BIG)


ANIMATIONS = {"blocks-a-tensor": ATensor, "blocks-a-batch": ABatch, "blocks-flatten": Flatten}

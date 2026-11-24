"""Animations 227-229 of "Blocks and code: padding" (see :mod:`voxanim.blocks`), drawn from
:mod:`voxanim.study_blocks_o`.

* :class:`Modes` (``padding-modes``): the 8 x 8 piece of a real digit that region
  ``pad-modes`` pads, then ``F.pad``'s four modes one per step, each a 12 x 12 block whose
  border fills in round the piece: "constant" with 0 (a faint frame of ink: more than the paper,
  which is -0.42), then reflect, replicate and circular, each with arrows from the two tiles that
  two border tiles of one row copy, and that row's numbers under the block.
* :class:`Deep` (``padding-deep-stacks``): twelve 3 x 3 stamps as squares to scale, one per
  layer: without padding they shrink, 28 to 4; with ``padding=1`` they keep 28. Then the
  padded CNN of region ``pad-model``: the side walked 28 -> 28 -> 14 -> 14 -> 7 with one real
  map of each stage, 16 maps of 7 x 7 = 784 numbers into the last layer, beside the 400 of
  the CNN without padding, to scale.
* :class:`InTheCnn` (``padding-in-the-cnn``): the CNN without and with padding, one run each:
  the numbers each block learns (the stamps the same, the last layer bigger), held-out
  accuracy, digits missed, and digits moved towards an edge, which both get wrong alike. A
  small table keeps the numbers as they come.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). What stays across phases is drawn in
``steady``, so nothing blanks between them.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .blocks import (
    DATA,
    DATA_HI,
    DATA_LO,
    HEAD,
    INK,
    QUIET,
    SIZE,
    WEIGHT,
    Blocks,
    Code,
    exact,
    measured,
    shades,
)
from .diagram import draw_image
from .easing import smoothstep
from .losschart import LossChart
from .mech import AXIS, PAPER, dark, soft
from .mech_blocks_k import Cnn, at, eased
from .mech_blocks_p import HATCH, hatched

DATA_INK = dark(DATA, 0.3)
WEIGHT_INK = dark(WEIGHT, 0.35)
PLAIN_FILL = soft(DATA, 0.55)           # without padding: pale blue; with: full blue
PADDED_FILL = DATA
PLAIN_INK = dark(DATA, 0.05)
PADDED_INK = dark(DATA, 0.45)


# ====================================================================== 227
class Modes(Blocks):
    """The piece; then constant, reflect, replicate and circular round it."""

    TITLE = "What goes in the border"
    PHASES = (("the piece", 10.0), ("constant", 12.0), ("reflect", 11.5), ("replicate", 11.0),
              ("circular", 11.5))
    LAST = "Same and valid"
    NEXT = "Deep stacks need padding"
    STATION = ("data",) + ("forward",) * 4
    DIG, DT = (0.118, 0.5), 2.4         # the digit card and its tiles (px)
    GRID, GT = (0.335, 0.59), 15.0      # the padded block's centre and its tiles (px)
    TEX = 6                             # texels per tile of the striped border
    KEY_Y = 0.645                       # the striped tile that says "padding"
    READ_Y = 0.825                      # the traced row's numbers, under the block
    CALL_Y = (0.705, 0.765)             # constant: the ring, the paper
    BORDER_AT = 0.06                    # constant: the border fades in
    MORPH = (0.03, 0.12)                # the next modes: the border turns into the next
    ARCS = ((0.16, 0.3), (0.36, 0.5))   # the two arrows, one after the other
    READ_AT = 0.56

    def setup(self):
        m = self.meta = self.study.metadata
        self.img = np.asarray(m["image"], float)
        self.part = np.asarray(m["part"], float)
        self.blocks = [np.asarray(m["blocks"][k], float) for k in m["modes"]]
        self.n, self.w, self.side = m["side"], m["width"], m["padded"]
        self.pad_mask = np.ones((self.side, self.side), bool)
        self.pad_mask[self.w:self.w + self.n, self.w:self.w + self.n] = False
        pr = m["printed"]
        self.CODE = (Code("pad-modes", run=((0.0, None), (0.08, 0)),
                          out=((0.3, m["part_text"]),)),) + tuple(
            Code("pad-modes", run=((0.0, 1), (0.1, 3), (0.62, 4)),
                 out=((0.65, pr[mode]),)) for mode in m["modes"])
        self.EVIDENCE = (exact("x[0], training digit 0, normalised"),) + tuple(
            exact("what F.pad's %s mode puts there" % mode) for mode in m["modes"])

    def heading_for(self, k, u):
        m = self.meta
        return ("an 8 × 8 piece of a real %d" % m["label"],
                "constant: 0 all round",
                "reflect: a mirror at the edge",
                "replicate: the edge, repeated",
                "circular: the other side comes round")[k]

    def subtitle_for(self, k):
        m = self.meta
        c = m["copied"]
        return [
            "The piece: 8 × 8 numbers of a real %d, %s of x[0]." % (m["label"], m["box_text"]),
            "F.pad adds 2 on every side: 12 × 12. \"constant\" fills with 0, but a blank "
            "pixel after Normalize is %s." % m["bg_text"],
            "reflect mirrors the piece at its edge: the border copies %s and %s from just "
            "inside, the edge itself left out." % tuple(c["reflect"]),
            "replicate repeats the edge outwards: both border tiles copy the edge tile, "
            "%s." % c["replicate"][0],
            "circular wraps round: the left border copies the right side of the piece, "
            "%s and %s." % (c["circular"][1], c["circular"][0]),
        ][k]

    # ------------------------------------------------------------------ geometry
    def cell(self):
        """``cell(row, col)`` of the 12 x 12 block (the piece is rows and columns 2-9)."""
        gx, gy = self.GRID
        s = self.side * self.GT
        return self.cells(gx, gy, s, s, self.side, self.side)

    def border_on(self, k, u):
        return at(k, u, (1, self.BORDER_AT))

    def border(self, k, u):
        """The 12 x 12 values on screen: the current mode's, turning from the last one's."""
        if k <= 1:
            return self.blocks[0]
        s = eased(u, *self.MORPH)
        return self.blocks[k - 2] + (self.blocks[k - 1] - self.blocks[k - 2]) * s

    def ring(self, sc, cell, r, c, op, ink=INK, lift=1.1):
        x, y = cell(r, c)
        self.rect(sc, self.P(x, y), self.GT + 3, self.GT + 3, ink, op, border=ink,
                  border_w=2.2, radius=2, lift=lift, fill_op=0.0)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        dx, dy = self.DIG
        cell = self.digit_card(sc, self.img, dx, dy, self.DT, op)
        r0, _r1, c0, _c1 = m["box"]
        _left, top, right, bottom = self.outline(sc, cell, r0, c0, self.n, self.n, self.DT, op,
                                                ink=INK, lift=1.0)
        self.text(sc, dx, dy + 0.105, "x[0], a %d" % m["label"], op, SIZE)
        gx, gy = self.GRID
        on = op * self.border_on(k, u)
        frac_on = self.border_on(k, u)
        side = (self.n + (self.side - self.n) * frac_on) * self.GT
        # the card round the block, growing with the border
        # (on paper, not the card's pale blue, so the ring of 0s shows against it)
        self.card(sc, gx, gy - self.frac(0, 3.0)[1], side + 14, side + 20, op, lift=0.2,
                  fill=PAPER)
        if on > 0.01:
            s = self.side * self.GT
            # the 2-wide border striped grey over its real values: the series' mark for
            # padding (group p's), baked into the texture so both themes match
            tex = hatched(shades(self.border(k, u), DATA, DATA_LO, DATA_HI), self.pad_mask,
                          self.TEX, self.TEX, 6, border=0.18)
            draw_image(self, sc, self.P(gx, gy), tex, s, s, on, grid=0.0, lift=0.3)
            # what the stripes mean: a striped tile and its word
            fx, fy, px = 0.068, self.KEY_Y, 16.0
            key = hatched(np.full((1, 1, 3), PAPER, np.float32), np.ones((1, 1), bool),
                          self.TEX * 2, self.TEX * 2, 6)
            self.rect(sc, self.P(fx, fy), px + 4, px + 4, PAPER, on, border=HATCH,
                      border_w=1.2, radius=3, lift=0.25)
            draw_image(self, sc, self.P(fx, fy), key, px, px, on, grid=0.0, lift=0.3)
            self.text(sc, fx + self.frac(px / 2 + 8)[0], fy, "padding", on, SIZE, "lm")
        s = self.n * self.GT
        self.tiles(sc, self.part, gx, gy, s, s, op, lo=DATA_LO, hi=DATA_HI, lift=0.4,
                   grid=0.25)
        if on > 0.01:              # the piece's edge, inside the border
            self.outline(sc, self.cell(), self.w, self.w, self.n, self.n, self.GT, on * 0.7,
                         ink=DATA_INK, lift=1.0)
        # the arrow from the piece in the digit to the piece, big
        a = op * at(k, u, (0, 0.25))
        x1 = gx - self.frac(side / 2 + 12)[0]
        self.arrow(sc, (right + 0.006, (top + bottom) / 2), (x1, gy), AXIS, a,
                   eased(u, 0.25, 0.35) if k == 0 else 1.0, width=1.6)
        # its size, over it
        size = "%d × %d" % ((self.side,) * 2 if frac_on > 0.5 else (self.n,) * 2)
        self.text(sc, gx, gy - self.frac(0, side / 2 + 26)[1], size, op, SIZE, strong=True)

    # ------------------------------------------------------------------ 1: the piece
    def phase_1(self, sc, u, op):
        pass

    # ------------------------------------------------------------------ 2: constant
    def phase_2(self, sc, u, op):
        m = self.meta
        cell = self.cell()
        x0 = 0.06
        for (r, c), y, s, when in ((m["ring_tile"], self.CALL_Y[0], "the ring: 0", 0.2),
                                   (m["paper_tile"], self.CALL_Y[1],
                                    "the paper: %s" % m["bg_text"], 0.36)):
            a = op * smoothstep(when, when + 0.05, u)
            self.text(sc, x0, y, s, a, SIZE, "lm", ink=INK if "ring" in s else DATA_INK)
            self.ring(sc, cell, r, c, a)
            tx, ty = cell(r, c)
            end = x0 + self.text_width(s) + 0.008
            self.arrow(sc, (end, y), (tx - self.frac(self.GT / 2 + 3)[0], ty), INK, a,
                       eased(u, when, when + 0.06), width=1.6)
        self.readout(sc, u, op, "constant")

    def text_width(self, s):
        return layout.text_width(s, SIZE)

    def readout(self, sc, u, op, mode):
        """The traced row, outlined, and its numbers under the block."""
        m = self.meta
        cell = self.cell()
        a = op * smoothstep(self.READ_AT, self.READ_AT + 0.05, u)
        self.outline(sc, cell, m["row"], 0, 1, self.side, self.GT, a, ink=QUIET, lift=0.9)
        self.text(sc, self.GRID[0], self.READ_Y, m["readout"][mode], a, SIZE)

    # ------------------------------------------------------------------ 3-5: copies
    def copies(self, sc, u, op, mode):
        cell = self.cell()
        half = self.frac(0, self.GT / 2 + 2)[1]
        for (src, dst), (a0, a1) in zip(self.meta["arrows"][mode], self.ARCS):
            a = op * smoothstep(a0, a0 + 0.04, u)
            self.ring(sc, cell, *src, a, ink=DATA_INK)
            self.ring(sc, cell, *dst, a)
            (sx, sy), (tx, ty) = cell(*src), cell(*dst)
            reach = abs(src[1] - dst[1])
            bend = 0.03 + 0.01 * reach
            # a pale halo under the dark line, so it shows over dark tiles too
            self.arc(sc, (sx, sy - half), (tx, ty - half), PAPER, a, eased(u, a0, a1),
                     bend=bend, width=5.5, lift=self.ARROW_LIFT - 0.02)
            self.arc(sc, (sx, sy - half), (tx, ty - half), INK, a, eased(u, a0, a1),
                     bend=bend, width=2.2)
        self.readout(sc, u, op, mode)

    def phase_3(self, sc, u, op):
        self.copies(sc, u, op, "reflect")

    def phase_4(self, sc, u, op):
        self.copies(sc, u, op, "replicate")

    def phase_5(self, sc, u, op):
        self.copies(sc, u, op, "circular")


# ====================================================================== 228
class Deep(Cnn):
    """Twelve stamps without and with padding; the padded CNN's walk; 16 x 7 x 7 = 784."""

    TITLE = "Deep stacks need padding"
    PHASES = (("no padding", 14.0), ("padding=1", 13.0), ("the padded CNN", 15.0),
              ("784", 13.0))
    LAST = "What goes in the border"
    NEXT = "The CNN with padding, measured"
    STATION = ("forward",) * 4
    T, PAD = 2.5, 3.0                   # px per pixel, to scale; the cards' margin
    X0, X1 = 0.065, 0.47
    ROWS = (0.505, 0.755)               # the two decks; the walk is on the first
    LABEL_Y = (0.4, 0.65)
    FAN = 22.0                          # px between one square of a deck and the next
    APPEAR = (tuple(0.08 + 0.055 * i for i in range(12)),
              tuple(0.1 + 0.05 * i for i in range(12)))
    NEW = (0.1, 0.3, 0.5, 0.7)          # phase 3: each square of the walk
    STEP_Y = 0.695                      # phase 3: the step under way
    GX, GY, GSTEP = 0.08, (0.665, 0.72), 0.03     # phase 4: 16 small maps
    MAPS_AT, SUM_AT, STRIP_AT, PLAIN_AT = 0.05, 0.3, 0.45, 0.62
    STRIP_Y, STRIP_W = (0.8, 0.858), 0.3

    def setup(self):
        m = self.meta = self.study.metadata
        d = self.study.panels[2].data
        self.maps = [np.asarray(d[k], float) for k in ("values", "conv1", "pool1", "conv2",
                                                        "pool2")]
        self.maps16 = np.asarray(self.study.panels[3].data["values"], float)
        self.plain_flat = np.asarray(m["plain_flat"], float)
        self.walk = m["side_walk"]
        self.scale = [(DATA_LO, DATA_HI)] + [(0.0, float(v.max())) for v in self.maps[1:]]
        sizes = [self.card_size(n, self.T, self.PAD) for n in self.walk]
        gap = (self.span(self.X1 - self.X0)[0] - sum(w for w, _h in sizes)) / (len(sizes) - 1)
        self.cx, x = [], self.X0
        for w, _h in sizes:
            self.cx.append(x + self.frac(w / 2)[0])
            x += self.frac(w + gap)[0]
        self.sizes = sizes
        tr = m["transitions"]
        a = self.APPEAR[0]
        self.CODE = (
            Code("pad-deep", run=((0.0, 0), (0.04, 1), (a[0], 2), (0.76, 3)),
                 out=((0.79, m["printed_deep"]),)),
            Code("pad-one", run=((0.0, 0), (0.45, 1)), out=((0.7, m["printed_one"]),)),
            Code("pad-model", run=((0.0, 0),) + tuple((self.NEW[j], j + 1) for j in range(4)),
                 out=tuple((self.NEW[j], tr[j]) for j in range(4))),
            Code("pad-model", run=((0.0, 4), (self.STRIP_AT, 5), (self.PLAIN_AT, 6)),
                 out=((0.0, tr[3]), (self.STRIP_AT, tr[4]), (self.PLAIN_AT, tr[5]))))
        self.EVIDENCE = (exact("12 layers, each side worked out"),) * 2 + (
            exact("the side of each block's maps"), exact("16 × 7 × 7, counted"))

    def heading_for(self, k, u):
        m = self.meta
        return ("no padding: each stamp loses 2",
                "padding=1: each stamp keeps 28",
                "the padded CNN: %s" % m["walk_text"],
                "%s numbers in" % m["in_features_sum"])[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "Without padding each 3 × 3 stamp loses 2, so %d in a row take %d down to %d."
            % (m["layers"], m["valid"][0], m["valid"][-1]),
            "With padding=1 each stamp keeps the size, so %d in a row still give %d × %d."
            % (m["layers"], m["same"][-1], m["same"][-1]),
            "The course's CNN with padding=1: the stamps keep the side, the pools halve it: "
            "%s." % ", ".join(str(v) for v in m["side_walk"]),
            "%d maps of %d × %d are %d numbers, so the last layer is nn.Linear(16 * 7 * 7, "
            "10); without padding it read %d." % (m["maps"], m["side"], m["side"],
                                                  m["in_features"], m["plain_in_features"]),
        ][k]

    # ------------------------------------------------------------------ the decks
    def deck_x(self, i):
        """The centre of square ``i`` of a deck (0: the digit)."""
        return self.cx[0] + self.frac(self.FAN * i)[0]

    def deck(self, sc, row, sides, k, u, op):
        """One deck: square ``i`` (``sides[i]`` to scale) appears at its time, behind the
        one before, fanned out to its right; the digit is square 0, in front."""
        y = self.ROWS[row]
        for i in range(1, len(sides)):
            a = op * at(k, u, (row, self.APPEAR[row][i - 1]))
            if a <= 0.01:
                continue
            n = sides[i]
            self.map_card(sc, np.full((n, n), 0.3), self.deck_x(i), y, self.T, a, 0.0, 1.0,
                          pad=self.PAD, lift=0.3 - 0.01 * i)

    def counter(self, row, sides, k, u):
        """The deck's last square so far: ``"layer i: n × n"``."""
        shown = 0
        for i in range(1, len(sides)):
            if at(k, u, (row, self.APPEAR[row][i - 1])) > 0.5:
                shown = i
        n = sides[shown]
        return "%d × %d" % (n, n) if shown == 0 else "layer %d: %d × %d" % (shown, n, n)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        # the digit: the first square of the top deck and of the walk, all through
        self.map_card(sc, self.maps[0], self.cx[0], self.ROWS[0], self.T, op, DATA_LO, DATA_HI,
                      pad=self.PAD, lift=0.35)
        under = self.ROWS[0] + self.frac(0, self.sizes[0][1] / 2 + 14)[1]
        self.text(sc, self.cx[0], under, str(self.walk[0]), op, HEAD, strong=True)
        decks = op * self.kept(0, 1)
        if decks > 0.01:
            self.text(sc, self.X0, self.LABEL_Y[0], "no padding", decks, SIZE, "lm")
            self.text(sc, self.X1, self.LABEL_Y[0], self.counter(0, m["valid"], k, u), decks,
                      HEAD, "rm", strong=True)
            self.deck(sc, 0, m["valid"], k, u, decks)
            low = decks * at(k, u, (1, 0.04))
            if low > 0.01:
                self.text(sc, self.X0, self.LABEL_Y[1], "padding=1", low, SIZE, "lm")
                self.text(sc, self.X1, self.LABEL_Y[1], self.counter(1, m["same"], k, u), low,
                          HEAD, "rm", strong=True)
                self.map_card(sc, self.maps[0], self.cx[0], self.ROWS[1], self.T, low, DATA_LO,
                              DATA_HI, pad=self.PAD, lift=0.35)
                self.deck(sc, 1, m["same"], k, u, low)
        walk = op * self.kept(2, 3)
        if walk > 0.01:
            self.walk_row(sc, k, u, walk)

    def walk_row(self, sc, k, u, op):
        m = self.meta
        y = self.ROWS[0]
        under = y + self.frac(0, self.sizes[0][1] / 2 + 14)[1]
        over = y - self.frac(0, self.sizes[0][1] / 2 + 10)[1]
        for j in range(1, len(self.walk)):
            a = op * at(k, u, (2, self.NEW[j - 1]))
            if a <= 0.01:
                continue
            lo, hi = self.scale[j]
            self.map_card(sc, self.maps[j], self.cx[j], y, self.T, a, lo, hi, pad=self.PAD)
            self.text(sc, self.cx[j], under, str(self.walk[j]), a, HEAD, strong=True)
            x0 = self.cx[j - 1] + self.frac(self.sizes[j - 1][0] / 2 + 3)[0]
            x1 = self.cx[j] - self.frac(self.sizes[j][0] / 2 + 3)[0]
            self.arrow(sc, (x0, y), (x1, y), AXIS, a, width=1.6)
            # what the block did to the side, over the row (the gaps are narrow)
            self.text(sc, (x0 + x1) / 2, over, m["steps"][j - 1]["op"], a, SIZE, ink=DATA_INK)

    # ------------------------------------------------------------------ 1, 2: the decks
    def phase_1(self, sc, u, op):
        pass

    def phase_2(self, sc, u, op):
        pass

    # ------------------------------------------------------------------ 3: the padded CNN
    def phase_3(self, sc, u, op):
        m = self.meta
        j = sum(u >= a for a in self.NEW)
        if j == 0:
            big, small = "28 × 28", "one digit in"
        else:
            big, small = m["steps"][j - 1]["sum"], m["steps"][j - 1]["why"]
        on = op * smoothstep(0.03, 0.08, u)
        self.big(sc, self.X0, self.STEP_Y, big, on)
        self.text(sc, self.X0, self.STEP_Y + 0.06, small, on, SIZE, "lm")

    # ------------------------------------------------------------------ 4: 784
    def phase_4(self, sc, u, op):
        m = self.meta
        ring = op * smoothstep(0.02, 0.06, u)
        n = self.walk[-1]
        cell = self.cells(self.cx[-1], self.ROWS[0] + self.frac(0, 3.0)[1], n * self.T,
                          n * self.T, n, n)
        self.outline(sc, cell, 0, 0, n, n, self.T, ring, ink=INK)
        hi = float(self.maps16.max())
        for i, v in enumerate(self.maps16):
            a = op * smoothstep(self.MAPS_AT + 0.012 * i, self.MAPS_AT + 0.012 * i + 0.04, u)
            self.map_card(sc, v, self.GX + self.GSTEP * (i % 8), self.GY[i // 8], self.T, a,
                          0.0, hi, pad=3.0)
        b = op * smoothstep(self.SUM_AT, self.SUM_AT + 0.05, u)
        self.text(sc, self.X1, (self.GY[0] + self.GY[1]) / 2, m["in_features_sum"], b, HEAD,
                  "rm", strong=True)
        # the numbers into the last layer, to scale: 784 with padding, 400 without
        for y, count, label, when, ink in (
                (self.STRIP_Y[0], m["in_features"], "%d" % m["in_features"], self.STRIP_AT,
                 DATA_INK),
                (self.STRIP_Y[1], m["plain_in_features"],
                 "%d without padding" % m["plain_in_features"], self.PLAIN_AT, QUIET)):
            a = op * smoothstep(when, when + 0.05, u)
            w = self.STRIP_W * count / m["in_features"]
            self.card(sc, self.X0 + w / 2, y, self.span(w)[0] + 10, 22.0, a, lift=0.1)
            vals, top = ((self.maps16.ravel(), hi) if count == m["in_features"]
                         else (self.plain_flat, float(self.plain_flat.max())))
            self.strip(sc, vals, self.X0 + w / 2, y + self.frac(0, 2)[1], self.span(w)[0], 10.0,
                       a, lo=0.0, hi=top)
            self.text(sc, self.X0 + w + 0.015, y, label, a, SIZE if count != m["in_features"]
                      else HEAD, "lm", ink=ink, strong=count == m["in_features"])


# ====================================================================== 229
class InTheCnn(Blocks):
    """The two CNNs' sizes, held-out curves, misses, and moved digits."""

    TITLE = "The CNN with padding, measured"
    PHASES = (("two CNNs", 12.0), ("accuracy", 13.0), ("missed", 11.5), ("moved digits", 10.0),
              ("both fall", 12.5))
    LAST = "Deep stacks need padding"
    NEXT = "A batch is one block"
    STATION = ("forward", "step", "forward", "data", "forward")
    COLS = (0.3, 0.415)                 # the table's two columns
    LABEL_X = 0.065
    ROWS = (0.385, 0.428, 0.469, 0.51, 0.551)   # header, numbers, accuracy, missed, moved 6 px
    BLOCK_Y = (0.6, 0.645, 0.715, 0.775)         # phase 1: each block's numbers; Linear's inputs
    BOX = ((0.12, 0.45), (0.62, 0.84))          # the held-out chart, and the moved one
    BARS = (0.105, 0.455, 0.845, 0.21)          # the per-digit bars: x0, x1, baseline, tallest
    CARDS = (0.112, 0.217, 0.322, 0.427)        # phase 4: the moved digit
    CARD_Y, TILE = 0.715, 2.7

    def setup(self):
        m = self.meta = self.study.metadata
        self.CODE = (
            Code("pad-model", run=((0.0, 0), (0.1, 1), (0.25, 3), (0.5, 6))),
            Code("cnn-train", pick=tuple(range(8)), run=((0.0, 0), (0.08, 1), (0.16, 7))),
            Code("cnn-evaluate", run=((0.0, 0), (0.12, 5), (0.3, 7)),
                 out=((0.35, "no padding %s, padding=1 %s" % (m["printed_plain"],
                                                              m["printed_padded"])),)),
            Code("cnn-evaluate", run=((0.0, 3), (0.3, 4))),
            Code("cnn-evaluate", run=((0.0, 5), (0.5, 7))))
        self.EVIDENCE = (exact("every weight and bias, counted"),
                         measured("10,000 held-out digits, one run each"),
                         measured("10,000 held-out digits, one run each"),
                         exact("test digit %d, moved; the gap is paper" % m["moved_digit"]),
                         measured("moved digits, 4 edges, one run each"))
        self.steps = np.asarray(m["eval_steps"], float)
        ink = np.asarray(m["moved_images"][0], float) - m["bg"]
        self.guide = int(round(float((ink.sum(0) * np.arange(ink.shape[1])).sum() / ink.sum())))
        self.chart = LossChart(self.BOX, (0.0, float(m["steps"])), (0.0, 1.0))
        self.moved_chart = LossChart(self.BOX, (0.0, float(m["shifts"][-1])), (0.0, 1.0))

    def heading_for(self, k, u):
        return ("the same stamps, a bigger last layer",
                "held-out accuracy, on the same steps",
                "fewer missed, but not padding alone",
                "digits moved towards an edge",
                "moved digits: both fall alike")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "Padding adds nothing to the stamps: all %s more numbers are in the last layer, "
            "which reads %d, not %d." % (m["extra_text"], m["padded_in"], m["plain_in"]),
            "Both trained one epoch on the same batches, one run each: %s with padding, %s "
            "without." % (m["padded_accuracy_text"], m["plain_accuracy_text"]),
            "It misses %d digits against %d. But its last layer is bigger too, so the gain is "
            "not padding's alone." % (m["padded_missed"], m["plain_missed"]),
            "Now the test digits are moved 1 to 6 pixels towards an edge, the gap filled "
            "with paper.",
            "Moved 6 pixels, both fall alike: %s without padding, %s with. Padding does not "
            "help here." % (m["plain_shifted_text"][-1], m["padded_shifted_text"][-1]),
        ][k]

    # ------------------------------------------------------------------ what stays
    def at(self, k, u):
        """1 once phase ``k`` has reached ``u`` (fading in over a short while)."""
        a, b = self.spans[k]
        t0 = a + (b - a) * u
        return smoothstep(t0, t0 + 0.5, self._t)

    def T(self, k, a, b):
        s, e = self.spans[k]
        return s + (e - s) * a, s + (e - s) * b

    def steady(self, sc, k, u, op):
        """The numbers as they come: one row per measurement, never blank once shown."""
        m = self.meta
        y0, yn, ya, ym, y6 = self.ROWS
        self.text(sc, self.LABEL_X, y0, "one run each", op, SIZE, "lm", ink=QUIET)
        for x, name, fill, ink in ((self.COLS[0], "no padding", PLAIN_FILL, PLAIN_INK),
                                   (self.COLS[1], "padding=1", PADDED_FILL, PADDED_INK)):
            sx = x - layout.text_width(name, SIZE) / 2 - self.frac(10.0)[0]
            self.rect(sc, self.P(sx, y0), 10.0, 10.0, fill, op, radius=2, lift=0.3)
            self.text(sc, x, y0, name, op, SIZE, ink=ink, strong=True)
        rows = (
            (yn, "numbers", m["plain_params_text"], m["padded_params_text"], self.at(0, 0.62)),
            (ya, "accuracy", m["plain_accuracy_text"], m["padded_accuracy_text"],
             self.at(1, 0.62)),
            (ym, "missed", str(m["plain_missed"]), str(m["padded_missed"]), self.at(2, 0.15)),
            (y6, "moved 6 px", m["plain_shifted_text"][-1], m["padded_shifted_text"][-1],
             self.at(4, 0.62)),
        )
        for y, label, a, b, on in rows:
            self.text(sc, self.LABEL_X, y, label, op * on, SIZE, "lm")
            self.text(sc, self.COLS[0], y, a, op * on, SIZE, "mm", ink=PLAIN_INK)
            self.text(sc, self.COLS[1], y, b, op * on, SIZE, "mm", ink=PADDED_INK)

    # ------------------------------------------------------------------ 1: two CNNs
    def phase_1(self, sc, u, op):
        m = self.meta
        names = ("Conv2d(1, 8)", "Conv2d(8, 16)", "Linear")
        for i, name in enumerate(names):
            a = op * smoothstep(0.1 + 0.12 * i, 0.15 + 0.12 * i, u)
            y = self.BLOCK_Y[i]
            last = i == len(names) - 1
            self.text(sc, self.LABEL_X, y, name, a, SIZE, "lm")
            self.text(sc, self.COLS[0], y, m["plain_counts_text"][i], a, SIZE, ink=PLAIN_INK)
            self.text(sc, self.COLS[1], y, m["padded_counts_text"][i], a,
                      HEAD if last else SIZE, ink=WEIGHT_INK if last else PADDED_INK,
                      strong=last)
        a = op * smoothstep(0.45, 0.5, u)
        y = self.BLOCK_Y[3]
        self.text(sc, self.LABEL_X, y, "Linear reads", a, SIZE, "lm")
        self.text(sc, self.COLS[0], y, str(m["plain_in"]), a, SIZE, ink=PLAIN_INK)
        self.text(sc, self.COLS[1], y, str(m["padded_in"]), a, HEAD, ink=WEIGHT_INK,
                  strong=True)
        self.rect(sc, self.P((self.LABEL_X + self.COLS[1] + 0.04) / 2,
                             (self.BLOCK_Y[2] + self.BLOCK_Y[3]) / 2),
                  self.span(self.COLS[1] + 0.06 - self.LABEL_X)[0] + 8.0,
                  self.span(0, self.BLOCK_Y[3] - self.BLOCK_Y[2])[1] + 28.0, WEIGHT, a,
                  border=WEIGHT_INK, border_w=2.0, radius=6, lift=0.9, fill_op=0.0)

    # ------------------------------------------------------------------ 2: accuracy
    def phase_2(self, sc, u, op):
        m = self.meta
        c = self.chart
        on = op * smoothstep(0.03, 0.09, u)
        self.axes(sc, c, on, ylabel="held-out accuracy", xticks=[(0, "0"), (938, "938")],
                  yticks=[(0, "0%"), (0.5, "50%"), (1, "100%")], grid=False)
        draw = eased(u, 0.1, 0.55)
        self.plot(sc, c, self.steps, m["plain_curve"], PLAIN_FILL, on, draw, width=1.6,
                  dot=False)
        self.plot(sc, c, self.steps, m["padded_curve"], PADDED_FILL, on, draw, width=1.6,
                  dot=False)
        lab = on * smoothstep(0.56, 0.62, u)
        self.key(sc, c.x1, "rm", lab, ("no padding %s" % m["plain_accuracy_text"],
                                       "padding=1 %s" % m["padded_accuracy_text"]))

    def key(self, sc, x, anchor, op, texts, ys=(0.735, 0.78)):
        """The two curves' end values, in the chart's empty corner, each in its colour."""
        for y, s, ink in zip(ys, texts, (PLAIN_INK, PADDED_INK)):
            self.text(sc, x, y, s, op, SIZE, anchor, ink=ink)

    # ------------------------------------------------------------------ 3: missed
    def bar_x(self, d, which):
        x0, x1, _base, _top = self.BARS
        w = (x1 - x0) / 10
        return x0 + w * (d + 0.5) + (-1 if which == 0 else 1) * self.frac(7.5)[0]

    def phase_3(self, sc, u, op):
        m = self.meta
        x0, x1, base, tall = self.BARS
        top = max(max(m["per_digit_plain"]), max(m["per_digit_padded"]))
        on = op * smoothstep(0.04, 0.1, u)
        grow = eased(u, 0.12, 0.32)
        self.text(sc, x0, base - tall - 0.03, "missed, by digit", on, SIZE, "lm")
        self.rect(sc, self.P((x0 + x1) / 2, base + 0.002), self.span(x1 - x0)[0], 1.4, QUIET,
                  on, radius=0.5)
        for d in range(10):
            for which, (vals, fill) in enumerate(((m["per_digit_plain"], PLAIN_FILL),
                                                  (m["per_digit_padded"], PADDED_FILL))):
                h = self.span(0, tall * vals[d] / top * grow)[1]
                if h > 0.5:
                    self.rect(sc, self.P(self.bar_x(d, which), base - self.frac(0, h / 2)[1]),
                              13.0, h, fill, on, radius=min(2.0, h / 2), lift=0.3)
            self.text(sc, (self.bar_x(d, 0) + self.bar_x(d, 1)) / 2, base + 0.03, str(d), on,
                      SIZE)
        # the digit they differ on most, labelled
        d = int(np.argmax(np.abs(np.subtract(m["per_digit_plain"], m["per_digit_padded"]))))
        a = on * smoothstep(0.42, 0.5, u)
        half = self.frac(6.5)[0]
        pair = (m["per_digit_plain"][d], m["per_digit_padded"][d])
        for which, ink in ((0, PLAIN_INK), (1, PADDED_INK)):
            v = pair[which]
            x, anchor = self.bar_x(d, which), "mm"
            if v < pair[1 - which]:
                x, anchor = (x + half, "rm") if which == 0 else (x - half, "lm")
            self.text(sc, x, base - tall * v / top - 0.022, str(v), a, SIZE, anchor, ink=ink)

    # ------------------------------------------------------------------ 4: moved digits
    def phase_4(self, sc, u, op):
        m = self.meta
        guide = self.guide
        for i, (img, s, fx) in enumerate(zip(m["moved_images"], m["moved"], self.CARDS)):
            on = op * smoothstep(0.08 + 0.1 * i, 0.14 + 0.1 * i, u)
            cell = self.digit_card(sc, np.asarray(img), fx, self.CARD_Y, self.TILE, on)
            self.text(sc, fx, self.CARD_Y + 0.115, "%d px" % s, on, HEAD, strong=True)
            # a line where the unmoved digit's ink is centred, the same on every card
            (x, y0), (_x, y1) = cell(0, guide), cell(27, guide)
            half = self.frac(0, self.TILE / 2)[1]
            self.polyline(sc, [self.P(x, y0 - half, self.ARROW_LIFT),
                               self.P(x, y1 + half, self.ARROW_LIFT)], 1.4, QUIET,
                          on * smoothstep(0.45, 0.5, u), bias=0.3)
        a = op * smoothstep(0.55, 0.6, u)
        self.text(sc, self.LABEL_X, 0.875, "and the same up, down and left", a, SIZE, "lm",
                  ink=QUIET)

    # ------------------------------------------------------------------ 5: both fall
    def phase_5(self, sc, u, op):
        m = self.meta
        c = self.moved_chart
        on = op * smoothstep(0.03, 0.09, u)
        self.axes(sc, c, on, xlabel="pixels moved",
                  xticks=[(s, "%d" % s) for s in (0, 2, 4, 6)],
                  yticks=[(0, "0%"), (0.5, "50%"), (1, "100%")], grid=False)
        draw = eased(u, 0.1, 0.5)
        xs = np.asarray(m["shifts"], float)
        self.plot(sc, c, xs, m["plain_shifted"], PLAIN_FILL, on, draw, width=1.8)
        self.plot(sc, c, xs, m["padded_shifted"], PADDED_FILL, on, draw, width=1.8)
        lab = on * smoothstep(0.55, 0.62, u)
        self.key(sc, c.x0 + 0.015, "lm", lab, ("no padding %s at 6" % m["plain_shifted_text"][-1],
                                               "padding=1 %s at 6" % m["padded_shifted_text"][-1]),
                 ys=(0.745, 0.79))


ANIMATIONS = {"padding-modes": Modes, "padding-deep-stacks": Deep,
              "padding-in-the-cnn": InTheCnn}

"""Animations 191-193 of the "Blocks and code" series (see :mod:`voxanim.blocks`): three
blocks the digit classifier does not use, drawn from :mod:`voxanim.study_blocks_c`.

* :class:`Conv2d` (``blocks-conv2d``): ``nn.Conv2d(1, 4, 3)``: one 3 x 3 stamp of orange
  weights (untrained: as a fresh ``nn.Conv2d`` starts, seeded), one output worked out (9
  pixels times 9 weights, added, plus the bias), the stamp sliding over all 676 places
  and filling a sheet, then 4 stamps making 4 sheets; 40 numbers learned.
* :class:`MaxPool2d` (``blocks-pooling``): ``nn.MaxPool2d(2)`` on the first of those
  sheets: one 2 x 2 square keeps its biggest number, every square in turn fills a 13 x 13
  sheet, the shapes ``[64, 4, 26, 26] -> [64, 4, 13, 13]``, and nothing learned.
* :class:`Dropout` (``blocks-dropout``): ``nn.Dropout(0.2)`` on 64 hidden numbers of the
  course's dropout run: in training 1 in 5 switched off (grey, ringed), the rest made 1.25
  times bigger, a new set every step; in ``eval()`` none; and the held-out accuracy with
  and without ``model.eval()``.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). What stays across phases is drawn in
``steady``, so nothing blanks between them.
"""

from __future__ import annotations

import numpy as np

from .blocks import (
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
from .diagram import draw_image
from .easing import ease_in_out, smoothstep
from .mech import AXIS, PAPER, dark, soft

DATA_INK = dark(DATA, 0.3)
WEIGHT_INK = dark(WEIGHT, 0.35)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


class Parts(Blocks):
    """What the three share: a sheet of tiles that fills in place by place, and a small
    grid of numbers in boxes."""

    def sheet(self, sc, values, fx, fy, tile, op, shown=None, lo=None, hi=None, lift=0.2,
              label=None):
        """A data card of ``tile`` px tiles centred at ``(fx, fy)``; ``shown``: a mask of the
        tiles filled so far (the others are paper). Returns ``cell(row, col)``."""
        values = np.asarray(values, float)
        rows, cols = values.shape
        w, h = cols * tile, rows * tile
        cy = fy + self.frac(0, 3.0)[1]
        cell = self.cells(fx, cy, w, h, rows, cols)
        if op <= 0.01:
            return cell
        self.card(sc, fx, fy, w + 14, h + 20, op, lift=lift)
        rgb = shades(values, DATA, lo, hi)
        if shown is not None:
            rgb = np.where(np.asarray(shown, bool)[..., None], rgb,
                           np.asarray(PAPER, np.float32))
        draw_image(self, sc, self.P(fx, cy), rgb, w, h, op, grid=0.18 if tile >= 4 else 0.0,
                   lift=lift + 0.004)
        if label:
            self.text(sc, fx, fy + self.frac(0, h / 2 + 24)[1], label, op, SIZE, ink=DATA_INK)
        return cell

    def numbers(self, sc, values, fx, fy, bw, bh, op, ink=DATA, lo=None, hi=None,
                ring=None, places=2):
        """A small grid of numbers in boxes shaded as tiles (deeper: bigger), ``ring``: the
        ``(row, col)`` of one box outlined. Returns ``cell(row, col)``."""
        v = np.atleast_2d(np.asarray(values, float))
        rows, cols = v.shape
        cell = self.cells(fx, fy, cols * bw, rows * bh, rows, cols)
        if op <= 0.01:
            return cell
        fills = shades(v, ink, lo, hi)
        lo = float(v.min()) if lo is None else lo
        hi = float(v.max()) if hi is None else hi
        for i in range(rows):
            for j in range(cols):
                x, y = cell(i, j)
                deep = (v[i, j] - lo) / max(hi - lo, 1e-9) > 0.8
                self.box(sc, x, y, bw - 3, bh - 3, tuple(float(c) for c in fills[i, j]), op,
                         border=ink, radius=3, size=SIZE, text=val(v[i, j], places),
                         ink=PAPER if deep and ink is not WEIGHT else INK)
        if ring is not None:
            x, y = cell(*ring)
            self.rect(sc, self.P(x, y), bw + 3, bh + 3, INK, op, border=INK, border_w=2.6,
                      radius=4, lift=1.2, fill_op=0.0)
        return cell

    def frame(self, sc, cell, r, c, n, tile, op, ink=INK, width=2.4, lift=1.1):
        """An outline round the ``n x n`` tiles from ``(r, c)`` of a grid."""
        if op <= 0.01:
            return
        (x0, y0), (x1, y1) = cell(r, c), cell(r + n - 1, c + n - 1)
        self.rect(sc, self.P((x0 + x1) / 2, (y0 + y1) / 2), n * tile + 4, n * tile + 4, ink,
                  op, border=ink, border_w=width, radius=2, lift=lift, fill_op=0.0)


# ====================================================================== 191
class Conv2d(Parts):
    """The digit and one stamp; one place worked out; every place; 4 stamps, 4 sheets
    and the 40 numbers they learn."""

    TITLE = "nn.Conv2d: one small stamp, used everywhere"
    PHASES = (("stamp", 11.0), ("one place", 15.0), ("slide", 15.0), ("4 sheets", 18.0))
    LAST = "a model is a stack of blocks"
    NEXT = "MaxPool: keep the biggest"
    STATION = ("forward",) * 4
    DIG, DT = (0.13, 0.5), 5.0              # the digit card and its tiles (px)
    SH, STL = (0.375, 0.5), 5.0             # the sheet
    SMALL = 2.35                            # tile of the 4 small sheets in phase 4
    SLOTS = ((0.32, 0.425), (0.43, 0.425), (0.32, 0.6), (0.43, 0.6))
    BAND = 0.80                             # the row of numbers under the pictures
    PATCH_X, STAMP_X, RES_X = 0.13, 0.31, 0.395
    BW, BH = 48.0, 26.0                     # a number's box
    APPEAR = (0.0, 0.12, 0.22, 0.32)        # phase 4: stamp and sheet k appear
    COUNT = 0.45                            # phase 4: the count

    def setup(self):
        m = self.meta = self.study.metadata
        self.img = np.asarray(self.study.panels[0].data["values"][0], float)
        self.W = np.asarray(m["weight"], float)[:, 0]
        self.b = np.asarray(m["bias"], float)
        self.maps = np.asarray(self.study.panels[2].data["values"], float)
        self.lo, self.hi = float(self.maps.min()), float(self.maps.max())
        self.at = tuple(m["at"])
        self.bound = 1.0 / 3.0                 # nn.Conv2d(1, 4, 3) starts in ±1/3
        self.places = m["places"]
        self.side = m["side_out"]
        out = "%s → %s" % (m["shape_in_text"], m["shape_out_text"])
        self.CODE = (Code("conv", run=((0.0, None), (0.25, 0))),
                     Code("conv", run=((0.0, 0), (0.3, 1))),
                     Code("conv", run=((0.0, 1),)),
                     Code("conv", run=((0.0, 1), (self.COUNT, 0)), out=((0.42, out),)))
        self.EVIDENCE = (exact("a real digit, untrained stamps"),) * 4

    def heading_for(self, k, u):
        if k == 3 and u >= self.COUNT:
            return "40 numbers, used everywhere"
        return ("nn.Conv2d(1, 4, 3): a 3 × 3 stamp", "one place: 9 numbers times 9",
                "the same stamp at every place", "4 stamps make 4 sheets")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "nn.Conv2d(1, 4, 3) makes 4 stamps of 3 × 3 weights. They start as random numbers.",
            "Over 9 pixels: multiply each by its weight, add them up, add the bias: %s."
            % val(m["output"]),
            "The stamp slides over the digit: %s places, the same 9 numbers every time."
            % m["places_sum"],
            "4 stamps make 4 sheets, %s. They learn %s numbers in all." % (
                m["shape_out_text"], m["learned_sum"]),
        ][k]

    # ------------------------------------------------------------------ where things are
    def digit_cell(self):
        return self.digit_cells(*self.DIG, self.DT)

    def sheet_cell(self):
        return self.cells(self.SH[0], self.SH[1] + self.frac(0, 3.0)[1], self.side * self.STL,
                          self.side * self.STL, self.side, self.side)

    def scan(self, u):
        """Phase 3: how far the stamp has gone, in places (a float, 0..676): row 0 slowly,
        the other rows quicker."""
        if u < 0.4:
            return self.side * eased(u, 0.06, 0.38) if u >= 0.06 else 0.0
        return self.side + (self.places - self.side) * float(np.clip((u - 0.42) / 0.43, 0, 1))

    def filled(self, k, u):
        """Which tiles of sheet 1 are filled."""
        shown = np.zeros((self.side, self.side), bool)
        if k >= 3:
            shown[:] = True
            return shown
        if k == 2:
            n = int(np.ceil(self.scan(u))) if u > 0.06 else 0
            shown.reshape(-1)[:n] = True
        if k == 2 or (k == 1 and u >= 0.78):
            shown[self.at] = True
        return shown

    def sheet_place(self, k, u):
        """Sheet 1's centre and tile: big in phases 2-3, moving to its small slot in 4."""
        (x0, y0), (x1, y1) = self.SH, self.SLOTS[0]
        s = 0.0 if k < 3 else eased(u, 0.02, 0.1)
        return x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, self.STL + (self.SMALL - self.STL) * s

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        self.digit_card(sc, self.img, *self.DIG, self.DT, op)
        # the stamp, its numbers and its bias: phases 1-3
        st = op * self.kept(0, 2)
        on = st if k > 0 else st * smoothstep(0.2, 0.28, u)
        self.numbers(sc, self.W[0], self.STAMP_X, self.BAND, self.BW, self.BH, on, ink=WEIGHT,
                     lo=-self.bound, hi=self.bound)
        self.text(sc, self.STAMP_X, self.BAND - 0.105, "stamp 1 of 4", on, SIZE,
                  ink=WEIGHT_INK)
        bias = st if k > 0 else st * smoothstep(0.45, 0.52, u)
        self.text(sc, self.STAMP_X, self.BAND + 0.095, "bias %s" % val(self.b[0]), bias, SIZE,
                  ink=WEIGHT_INK)
        # the sheet: from phase 2
        fx, fy, tile = self.sheet_place(k, u)
        self.sheet(sc, self.maps[0], fx, fy, tile, op * self.kept(1), shown=self.filled(k, u),
                   lo=self.lo, hi=self.hi, label="sheet 1" if k < 3 else None)

    # ------------------------------------------------------------------ 1: a stamp
    def phase_1(self, sc, u, op):
        on = op * smoothstep(0.62, 0.7, u)
        self.text(sc, 0.06, self.BAND - 0.022, "9 weights,", on, SIZE, "lm")
        self.text(sc, 0.06, self.BAND + 0.022, "random to start", on, SIZE, "lm")

    # ------------------------------------------------------------------ 2: one place
    def phase_2(self, sc, u, op):
        m = self.meta
        r, c = self.at
        cell = self.digit_cell()
        ring = op * smoothstep(0.04, 0.1, u)
        self.frame(sc, cell, r, c, 3, self.DT, ring, ink=WEIGHT, width=2.8)
        patch = np.asarray(m["patch"], float)
        grow = op * smoothstep(0.12, 0.2, u)
        (x0, _y0), (x1, y1) = cell(r, c), cell(r + 2, c + 2)
        pw, ph = self.frac(3 * self.BW, 3 * self.BH)
        for a, b in ((x0 - self.frac(4)[0], self.PATCH_X - pw / 2),
                     (x1 + self.frac(4)[0], self.PATCH_X + pw / 2)):
            self.arrow(sc, (a, y1 + self.frac(0, 6)[1]), (b, self.BAND - ph / 2), AXIS,
                       grow * 0.8, head=False, width=1.3)
        self.numbers(sc, patch, self.PATCH_X, self.BAND, self.BW, self.BH, grow, lo=DATA_LO,
                     hi=DATA_HI)
        self.text(sc, self.PATCH_X, self.BAND + 0.095, "9 pixels", grow, SIZE, ink=DATA_INK)
        times = op * smoothstep(0.28, 0.34, u)
        self.text(sc, (self.PATCH_X + self.STAMP_X) / 2, self.BAND, "×", times, HEAD)
        self.rect(sc, self.P(self.STAMP_X, self.BAND), 3 * self.BW + 8, 3 * self.BH + 8, INK,
                  times, border=INK, border_w=2.2, radius=5, lift=1.1, fill_op=0.0)
        steps = [(0.0, ""), (0.42, val(m["products_sum"])), (0.58, val(m["output"]))]
        lab = "add up" if u < 0.58 else "+ bias"
        a = op * smoothstep(0.42, 0.48, u)
        self.running(sc, self.RES_X, self.BAND + 0.02, steps, u, a, label=lab, size=HEAD)
        # the answer flies to its tile in the sheet
        out = self.sheet_cell()(r, c)
        s = eased(u, 0.66, 0.78)
        if 0.66 <= u <= 0.8:
            sx, sy = self.RES_X + 0.03, self.BAND + 0.02
            self.box(sc, sx + (out[0] - sx) * s, sy + (out[1] - sy) * s, 14, 14,
                     soft(DATA, 0.5), op, border=INK)
        self.frame(sc, self.sheet_cell(), r, c, 1, self.STL, op * smoothstep(0.78, 0.84, u),
                   width=2.2)

    # ------------------------------------------------------------------ 3: every place
    def phase_3(self, sc, u, op):
        pos = self.scan(u)
        i = min(int(pos), self.places - 1)
        r, c = divmod(i, self.side)
        on = op * smoothstep(0.03, 0.08, u) * (1 - smoothstep(0.9, 0.95, u))
        self.frame(sc, self.digit_cell(), r, c, 3, self.DT, on, ink=WEIGHT, width=2.8)
        self.frame(sc, self.sheet_cell(), r, c, 1, self.STL, on, width=2.0)
        n = int(np.ceil(pos)) if u > 0.06 else 0
        self.running(sc, self.RES_X, self.BAND + 0.02, [(0.0, commas(n))], 0.0, op,
                     label="places", size=HEAD)
        self.text(sc, self.RES_X, self.BAND + 0.075, self.meta["places_sum"],
                  op * smoothstep(0.88, 0.93, u), SIZE, "lm")

    # ------------------------------------------------------------------ 4: 4 sheets
    def phase_4(self, sc, u, op):
        m = self.meta
        for i in range(4):
            on = op * smoothstep(self.APPEAR[i], self.APPEAR[i] + 0.05, u)
            if i:
                self.sheet(sc, self.maps[i], *self.SLOTS[i], self.SMALL, on, lo=self.lo,
                           hi=self.hi)
            x = 0.075 + 0.07 * i
            self.tiles(sc, self.W[i], x, self.BAND, 36.0, 36.0, on, ink=WEIGHT,
                       lo=-self.bound, hi=self.bound, grid=0.35)
            self.rect(sc, self.P(x, self.BAND), 40.0, 40.0, WEIGHT, on, border=WEIGHT,
                      border_w=1.6, radius=3, lift=0.5, fill_op=0.0)
            self.text(sc, x, self.BAND + 0.06, str(i + 1), on, SIZE, ink=WEIGHT_INK)
            sx, sy = self.SLOTS[i]
            self.text(sc, sx - 0.05, sy, str(i + 1), on, SIZE, ink=DATA_INK)
        self.text(sc, 0.075, self.BAND - 0.07, "4 stamps", op * smoothstep(0.36, 0.42, u),
                  SIZE, "lm", ink=WEIGHT_INK)
        a = op * smoothstep(self.COUNT + 0.02, self.COUNT + 0.08, u)
        self.big(sc, 0.36, self.BAND + 0.035, str(m["learned"]), a,
                 label="4 × 3 × 3 + 4 =", size=BIG)
        b = op * smoothstep(self.COUNT + 0.2, self.COUNT + 0.26, u)
        self.text(sc, 0.06, self.BAND + 0.1, "the same %d at all %d places" % (
            m["learned"], m["places"]), b, SIZE, "lm")


# ====================================================================== 192
class MaxPool2d(Parts):
    """A sheet; one 2 x 2 square; every square; all 4 sheets, and nothing learned."""

    TITLE = "nn.MaxPool2d: keep the biggest"
    PHASES = (("a sheet", 10.0), ("one square", 15.0), ("every square", 15.0),
              ("4 sheets", 18.0))
    LAST = "4 stamps made 4 sheets"
    NEXT = "Dropout: switched off at random"
    STATION = ("forward",) * 4
    SH, ST = (0.15, 0.53), 5.5              # the conv sheet
    OUT, OT = (0.39, 0.74), 5.5             # the pooled sheet
    MAG = (0.36, 0.45)                      # the magnified square
    BW, BH = 46.0, 30.0
    COUNT = 0.45                            # phase 4: nothing learned

    def setup(self):
        m = self.meta = self.study.metadata
        self.sheet0 = np.asarray(self.study.panels[0].data["values"][0], float)
        self.small = np.asarray(self.study.panels[2].data["values"], float)
        self.lo, self.hi = float(self.sheet0.min()), float(self.sheet0.max())
        self.at = tuple(m["at"])
        self.n = m["side_out"]
        out = "%s → %s" % (m["shape_in_text"], m["shape_out_text"])
        self.CODE = (Code("pool", run=((0.0, None), (0.3, 0))),
                     Code("pool", run=((0.0, 0), (0.3, 1))),
                     Code("pool", run=((0.0, 1),)),
                     Code("pool", run=((0.0, 1), (self.COUNT, 0)), out=((0.2, out),)))
        self.EVIDENCE = (exact("a real digit's first sheet"),) * 3 + \
            (exact("numbers counted per digit"),)

    def heading_for(self, k, u):
        if k == 3 and u >= self.COUNT:
            return "no weights: it only picks"
        return ("a sheet from nn.Conv2d", "each 2 × 2 square: keep the biggest",
                "every square, in turn", "all 4 sheets: a quarter left")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "One sheet that nn.Conv2d made from the digit: 26 × 26 numbers.",
            "nn.MaxPool2d(2) looks at one 2 × 2 square and keeps the biggest number: %s."
            % val(m["biggest"]),
            "Square after square: %s squares make a sheet of 13 × 13." % m["squares_sum"],
            "%s becomes %s: a quarter of the numbers, and %d learned."
            % (m["shape_in_text"], m["shape_out_text"], m["weights"]),
        ][k]

    def big_cell(self):
        return self.cells(self.SH[0], self.SH[1] + self.frac(0, 3.0)[1], 26 * self.ST,
                          26 * self.ST, 26, 26)

    def out_cell(self):
        return self.cells(self.OUT[0], self.OUT[1] + self.frac(0, 3.0)[1], 13 * self.OT,
                          13 * self.OT, 13, 13)

    def scan(self, u):
        """Phase 3: squares done (a float 0..169): the first row slowly, then quicker."""
        if u < 0.4:
            return self.n * eased(u, 0.06, 0.38) if u >= 0.06 else 0.0
        return self.n + (self.n * self.n - self.n) * float(np.clip((u - 0.42) / 0.43, 0, 1))

    def filled(self, k, u):
        shown = np.zeros((self.n, self.n), bool)
        if k >= 3:
            shown[:] = True
            return shown
        if k == 2:
            done = int(np.ceil(self.scan(u))) if u > 0.06 else 0
            shown.reshape(-1)[:done] = True
        if k == 2 or (k == 1 and u >= 0.72):
            shown[self.at[0] // 2, self.at[1] // 2] = True
        return shown

    def steady(self, sc, k, u, op):
        if k >= 3:          # the other three sheets, behind
            on = op if k > 3 else op * smoothstep(0.05, 0.12, u)
            for i in (3, 2, 1):
                d = self.frac(8.0 * i, -5.0 * i)
                self.card(sc, self.SH[0] + d[0], self.SH[1] + d[1], 26 * self.ST + 14,
                          26 * self.ST + 20, on, lift=0.1 - 0.01 * i)
                d = self.frac(6.0 * i, -4.0 * i)
                self.card(sc, self.OUT[0] + d[0], self.OUT[1] + d[1], 13 * self.OT + 14,
                          13 * self.OT + 20, on, lift=0.1 - 0.01 * i)
        self.sheet(sc, self.sheet0, *self.SH, self.ST, op, lo=self.lo, hi=self.hi,
                   label="26 × 26")
        self.sheet(sc, self.small[0], *self.OUT, self.OT, op * self.kept(1),
                   shown=self.filled(k, u), lo=self.lo, hi=self.hi,
                   label="13 × 13" if k >= 2 else None)

    # ------------------------------------------------------------------ 1: a sheet
    def phase_1(self, sc, u, op):
        on = op * smoothstep(0.45, 0.52, u)
        cell = self.big_cell()
        # the sheet cut into 2 x 2 squares: the first of them outlined
        for j in range(4):
            a = on * smoothstep(0.5 + 0.06 * j, 0.55 + 0.06 * j, u)
            self.frame(sc, cell, 0, 2 * j, 2, self.ST, a, width=1.6)
        self.text(sc, 0.3, 0.45, "squares of 2 × 2,", on, SIZE, "lm")
        self.text(sc, 0.3, 0.495, "side by side", on, SIZE, "lm")

    # ------------------------------------------------------------------ 2: one square
    def phase_2(self, sc, u, op):
        m = self.meta
        r, c = self.at
        ring = op * smoothstep(0.05, 0.12, u)
        cell = self.big_cell()
        self.frame(sc, cell, r, c, 2, self.ST, ring, width=2.6)
        grow = op * smoothstep(0.15, 0.24, u)
        (_x0, y0), (x1, y1) = cell(r, c), cell(r + 1, c + 1)
        mw, mh = self.frac(2 * self.BW, 2 * self.BH)
        mx, my = self.MAG
        for b, e in ((y0, my - mh / 2), (y1, my + mh / 2)):
            self.arrow(sc, (x1 + self.frac(5)[0], b), (mx - mw / 2, e), AXIS, grow * 0.8,
                       head=False, width=1.3)
        win = tuple(m["winner"]) if u >= 0.4 else None
        self.numbers(sc, m["square"], mx, my, self.BW, self.BH, grow, lo=self.lo, hi=self.hi,
                     ring=win)
        keep = op * smoothstep(0.42, 0.48, u)
        self.text(sc, mx, my + mh / 2 + 0.04, "biggest: %s" % val(m["biggest"]), keep, SIZE,
                  strong=True)
        s = eased(u, 0.55, 0.7)
        out = self.out_cell()(r // 2, c // 2)
        if 0.55 <= u <= 0.72:
            wx, wy = self.cells(mx, my, 2 * self.BW, 2 * self.BH, 2, 2)(*m["winner"])
            self.box(sc, wx + (out[0] - wx) * s, wy + (out[1] - wy) * s, 14, 14,
                     soft(DATA, 0.5), op, border=INK)
        self.frame(sc, self.out_cell(), r // 2, c // 2, 1, self.OT,
                   op * smoothstep(0.7, 0.76, u), width=2.2)
        self.text(sc, 0.06, 0.85, "4 numbers → 1", op * smoothstep(0.8, 0.86, u), HEAD, "lm",
                  strong=True)

    # ------------------------------------------------------------------ 3: every square
    def phase_3(self, sc, u, op):
        pos = self.scan(u)
        i = min(int(pos), self.n * self.n - 1)
        r, c = divmod(i, self.n)
        on = op * smoothstep(0.03, 0.08, u) * (1 - smoothstep(0.9, 0.95, u))
        self.frame(sc, self.big_cell(), 2 * r, 2 * c, 2, self.ST, on, width=2.4)
        self.frame(sc, self.out_cell(), r, c, 1, self.OT, on, width=2.0)
        n = int(np.ceil(pos)) if u > 0.06 else 0
        self.running(sc, 0.3, 0.47, [(0.0, str(n))], 0.0, op, label="squares", size=BIG)
        self.text(sc, 0.3, 0.53, self.meta["squares_sum"], op * smoothstep(0.88, 0.93, u),
                  SIZE, "lm")

    # ------------------------------------------------------------------ 4: 4 sheets
    def phase_4(self, sc, u, op):
        m = self.meta
        a = op * smoothstep(0.1, 0.16, u)
        self.text(sc, 0.3, 0.42, "numbers per digit", a, SIZE, "lm")
        self.text(sc, 0.3, 0.47, commas(m["per_digit_in"]), a, HEAD, "lm", strong=True)
        b = op * smoothstep(0.2, 0.26, u)
        self.text(sc, 0.3, 0.525, "→ %d" % m["per_digit_out"], b, HEAD, "lm", strong=True)
        c = op * smoothstep(0.3, 0.36, u)
        self.text(sc, 0.3, 0.58, "a quarter", c, SIZE, "lm")
        z = op * smoothstep(self.COUNT + 0.02, self.COUNT + 0.09, u)
        self.big(sc, 0.06, 0.86, str(m["weights"]), z, label="numbers it learns", size=BIG)


# ====================================================================== 193
class Dropout(Parts):
    """64 hidden numbers; training switches some off; a new set each step; testing
    switches none; the accuracy with and without model.eval()."""

    TITLE = "nn.Dropout: switched off at random"
    PHASES = (("64 numbers", 10.0), ("training", 15.0), ("each step", 13.0),
              ("testing", 11.0), ("model.eval()", 11.0))
    LAST = "MaxPool kept the biggest"
    NEXT = "BatchNorm and LayerNorm"
    STATION = ("forward",) * 5
    CARD, TILE = (0.15, 0.56), 19.0
    SWITCH = (0.37, 0.43)
    RIGHT = 0.31

    def setup(self):
        m = self.meta = self.study.metadata
        g = m["grid"]
        self.h = np.asarray(m["h"], float).reshape(g, g)
        self.steps = m["steps"]
        self.hi = float(self.h.max()) * m["scale"]
        off = []
        for s in self.steps:
            mask = np.zeros(g * g, bool)
            mask[s["off"]] = True
            off.append(mask.reshape(g, g))
        self.off = off
        n0 = len(self.steps[0]["off"])
        # phase 2: the switches go off one by one, a few slowly, then the rest
        self.when = [0.3 + 0.07 * i for i in range(3)] + list(np.linspace(0.5, 0.62, n0 - 3))
        self.CODE = (Code("dropout", run=((0.0, None), (0.3, 0))),
                     Code("dropout", run=((0.0, 1), (0.25, 2))),
                     Code("dropout", run=((0.0, 2),)),
                     Code("dropout", run=((0.0, 3), (0.3, 4))),
                     Code(("train-mode", "evaluate"), pick=(0, 1),
                          run=((0.0, None), (0.15, 0), (0.5, 1))))
        tag = exact("the dropout run's switches")
        self.EVIDENCE = (exact("the first digit's 64 numbers"), tag, tag,
                         exact("nothing switched off"),
                         measured("10,000 held-out digits"))

    def heading_for(self, k, u):
        return ("h: 64 hidden numbers", "training: 1 in 5 switched off",
                "a new random set each step", "testing: nothing switched off",
                "forget model.eval(): fewer right")[k]

    def subtitle_for(self, k):
        m = self.meta
        c = [s["count"] for s in self.steps]
        return [
            "h: one digit's 64 hidden numbers after nn.ReLU. nn.Dropout(0.2) comes next.",
            "In training it switches %d of the 64 off at random; the rest grow %s."
            % (c[0], m["scale_text"]),
            "Every step picks a new set: %d, then %d off. Over the epoch: %.1f%% off."
            % (c[1], c[2], 100 * m["epoch_share"]),
            "In eval() nothing is switched off: drop(h) gives back h, the same 64 numbers.",
            "Left in training mode, the model gets %.2f%% right; with model.eval(), %.2f%%."
            % (100 * m["accuracy_train_mode"], 100 * m["accuracy_eval"]),
        ][k]

    # ------------------------------------------------------------------ state
    def mode(self, k, u):
        """The switch: 0 train, 1 eval (to eval in phase 4; in phase 5 back to train, the
        mistake, then to eval)."""
        if k < 3:
            return 0.0
        if k == 3:
            return eased(u, 0.05, 0.2)
        return 1.0 - eased(u, 0.05, 0.15) + eased(u, 0.48, 0.58)

    def state(self, k, u):
        """``(values, off mask, count off)`` on the card now."""
        g = self.meta["grid"]
        none = np.zeros((g, g), bool)
        if k == 0 or (k == 1 and u < self.when[0]):
            return self.h, none, 0
        if k == 1:
            n = sum(u >= w for w in self.when)
            off = np.zeros(g * g, bool)
            off[self.steps[0]["off"][:n]] = True
            off = off.reshape(g, g)
            scaled = u >= 0.7
            v = np.where(off, 0.0, self.h * (self.meta["scale"] if scaled else 1.0))
            return v, off, n
        if k == 2:
            i = 0 if u < 0.15 else (1 if u < 0.45 else 2)
            off = self.off[i]
            return np.where(off, 0.0, self.h * self.meta["scale"]), off, int(off.sum())
        if k == 3 and u < 0.35:
            off = self.off[2]
            return np.where(off, 0.0, self.h * self.meta["scale"]), off, int(off.sum())
        return self.h, none, 0

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        v, off, n = self.state(k, u)
        cx, cy = self.CARD
        g = self.meta["grid"]
        on = op * self.kept(0, 3)
        side = g * self.TILE
        self.card(sc, cx, cy, side + 14, side + 20, on)
        ring = [tuple(int(i) for i in rc) for rc in np.argwhere(off)]
        cell = self.tiles(sc, v, cx, cy + self.frac(0, 3.0)[1], side, side, on, lo=0.0,
                          hi=self.hi, zero=off, ring=ring, ring_ink=INK, grid=0.25)
        self.cell = cell
        self.text(sc, cx, cy - self.frac(0, side / 2 + 26)[1], "h: 64 numbers", on, SIZE,
                  ink=DATA_INK)
        # the switch, from phase 2
        self.switch(sc, self.mode(k, u), op * self.kept(1))
        # how many are off, all along phases 2-4
        self.running(sc, self.RIGHT, 0.6, [(0.0, "%d of 64" % n)], 0.0, op * self.kept(1, 3),
                     label="switched off", size=HEAD)

    def switch(self, sc, s, op):
        """A switch: the knob at ``s`` (0: train, 1: eval)."""
        if op <= 0.01:
            return
        x, y = self.SWITCH
        c = self.P(x, y)
        self.rect(sc, c, 58.0, 24.0, soft(AXIS, 0.75), op, border=AXIS, border_w=1.4,
                  radius=12, lift=0.3)
        kx = -15.0 + 30.0 * s
        self.disk(sc, self.shift(c, kx, 0), 10.0, INK, op, lift=0.6, outline=0.0)
        self.text(sc, x - 0.045, y, "train", op * (1 - 0.6 * s), SIZE, "rm",
                  strong=s < 0.5)
        self.text(sc, x + 0.045, y, "eval", op * (0.4 + 0.6 * s), SIZE, "lm",
                  strong=s >= 0.5)

    # ------------------------------------------------------------------ 1: 64 numbers
    def phase_1(self, sc, u, op):
        m = self.meta
        on = op * smoothstep(0.4, 0.47, u)
        self.big(sc, self.RIGHT, 0.5, "p = %.1f" % m["p"], on, label="nn.Dropout(0.2)",
                 size=BIG)
        self.text(sc, self.RIGHT, 0.58, "1 in 5", op * smoothstep(0.6, 0.67, u), HEAD, "lm",
                  strong=True)

    # ------------------------------------------------------------------ 2: training
    def phase_2(self, sc, u, op):
        m = self.meta
        r, c = divmod(m["example"], m["grid"])
        a = op * smoothstep(0.7, 0.76, u)
        x, y = self.cell(r, c)
        self.rect(sc, self.P(x, y), self.TILE + 6, self.TILE + 6, DATA, a, border=DATA_INK,
                  border_w=3.0, radius=3, lift=1.2, fill_op=0.0)
        self.text(sc, self.RIGHT, 0.72, "kept: %s %s" % (val(m["example_in"]), m["scale_text"]),
                  a, SIZE, "lm")
        self.text(sc, self.RIGHT, 0.765, "= %s" % val(m["example_out"]), a, SIZE, "lm",
                  strong=True)

    # ------------------------------------------------------------------ 3: each step
    def phase_3(self, sc, u, op):
        i = 0 if u < 0.15 else (1 if u < 0.45 else 2)
        self.running(sc, self.RIGHT, 0.73, [(0.0, "step %d" % self.steps[i]["step"])], 0.0,
                     op * smoothstep(0.02, 0.06, u), size=HEAD)
        on = op * smoothstep(0.78, 0.84, u)
        self.text(sc, self.RIGHT, 0.79, "over the epoch:", on, SIZE, "lm")
        self.text(sc, self.RIGHT, 0.835, "%.1f%% off" % (100 * self.meta["epoch_share"]), on,
                  SIZE, "lm", strong=True)

    # ------------------------------------------------------------------ 4: testing
    def phase_4(self, sc, u, op):
        on = op * smoothstep(0.5, 0.57, u)
        self.text(sc, self.RIGHT, 0.72, "same = h", on, HEAD, "lm", strong=True)

    # ------------------------------------------------------------------ 5: model.eval()
    def phase_5(self, sc, u, op):
        m = self.meta
        rows = (("still in train()", m["accuracy_train_mode"], m["right_train_mode"], 0.15),
                ("model.eval()", m["accuracy_eval"], m["right_eval"], 0.5))
        x0, full = 0.07, 260.0
        for i, (name, acc, right, at) in enumerate(rows):
            on = op * smoothstep(at, at + 0.07, u)
            y = 0.55 + 0.15 * i
            self.text(sc, x0, y - 0.045, name, on, SIZE, "lm", strong=True)
            self.hbar(sc, x0, y, full * acc, DATA if i else ZERO, on,
                      value="%.2f%%" % (100 * acc), u=eased(u, at, at + 0.12), thick=22.0,
                      track=full)
            self.text(sc, x0, y + 0.045, "%s of %s right" % (commas(right), commas(m["test"])),
                      on, SIZE, "lm", ink=DATA_INK)
        more = m["right_eval"] - m["right_train_mode"]
        self.text(sc, x0, 0.85, "%d more right with model.eval()" % more,
                  op * smoothstep(0.68, 0.74, u), SIZE, "lm", strong=True)


ANIMATIONS = {"blocks-conv2d": Conv2d, "blocks-pooling": MaxPool2d, "blocks-dropout": Dropout}

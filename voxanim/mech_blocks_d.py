"""Animations 194-196 of the "Blocks and code" series (see :mod:`voxanim.blocks`): the
normalisation blocks, the embedding table and attention, drawn from
:mod:`voxanim.study_blocks_d`.

* :class:`Normalisation` (``blocks-normalisation``): the course model's hidden block for
  the first batch, ``[64, 64]``; ``nn.BatchNorm1d(64)`` evens out each column (down the
  batch, the arrow points down), ``nn.LayerNorm(64)`` each row (across the features, the
  arrow points across): one column and one row followed, then all of them; the two results
  side by side.
* :class:`Embedding` (``blocks-embedding``): ``nn.Embedding(10, 16)`` as a table of orange
  tiles; a label picks a row, which is copied out; the first batch's 64 labels make a
  ``[64, 16]`` block; a one-hot row times the table gives the same row with 160
  multiplications, the lookup with none.
* :class:`Attention` (``blocks-attention``): ``nn.MultiheadAttention(16, 2)`` on 5 patches
  of a real digit: the tokens, three weight grids making Q, K and V, each head's grid of
  scores and its softmax rows, and the output, a mix of V's rows.

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
    exact,
    measured,
    val,
)
from .easing import ease_in_out, smoothstep
from .mech import AXIS, PAPER, dark

DATA_INK = dark(DATA, 0.3)
WEIGHT_INK = dark(WEIGHT, 0.35)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on(u, a, op, width=0.06):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


class Parts(Blocks):
    """What the three share."""

    def frame(self, sc, fx, fy, w, h, op, ink=INK, lift=1.0, width=2.2):
        """An outline ``w x h`` px round ``(fx, fy)``."""
        if op <= 0.01:
            return
        self.rect(sc, self.P(fx, fy), w, h, ink, op, border=ink, border_w=width, radius=3,
                  lift=lift, fill_op=0.0)

    def veil(self, sc, fx, fy, w, h, op, lift=0.5):
        """Paper laid over ``w x h`` px round ``(fx, fy)``: what is under it dims."""
        if op <= 0.01:
            return
        self.rect(sc, self.P(fx, fy), w, h, PAPER, op, radius=0, lift=lift)


# ====================================================================== 194
NORM = ("batchnorm", "layernorm")


class Normalisation(Parts):
    """The block h; BatchNorm down each column; LayerNorm across each row; both."""

    TITLE = "BatchNorm and LayerNorm"
    PHASES = (("the block h", 11.0), ("BatchNorm", 16.0), ("LayerNorm", 16.0),
              ("two directions", 12.0))
    LAST = "nn.Dropout: switched off at random"
    NEXT = "nn.Embedding: a table you look up"
    STATION = ("forward",) * 4
    BLOCK = (0.20, 0.60)
    TILE = 3.0
    SMALL = 2.2                       # px per tile side by side, phase 4
    BN_AT, LN_AT = (0.15, 0.61), (0.34, 0.61)
    RIGHT = 0.335
    ONE = 0.35                        # u: the followed column (row) is normalised
    SWEEP = (0.5, 0.86)               # u: then all the others, one after the other
    STRIP_GAP = 13.0                  # px between the block and its means

    def setup(self):
        m = self.meta = self.study.metadata
        p = self.study.panels
        self.h = np.asarray(p[0].data["values"], float)
        self.bn = np.asarray(p[1].data["values"], float)
        self.ln = np.asarray(p[2].data["values"], float)
        self.lo, self.hi = m["shade"]
        self.col, self.row = m["col"], m["row"]
        self.n = self.h.shape[0]
        out = m["printed"]
        self.CODE = (Code(NORM, run=((0.0, None),)),
                     Code(NORM, run=((0.0, 0), (self.ONE - 0.05, 1)), out=((0.9, out),)),
                     Code(NORM, run=((0.0, 2), (self.ONE - 0.05, 3)), out=((0.9, out),)),
                     Code(NORM, run=((0.0, 1), (0.45, 3))))
        self.EVIDENCE = (measured("the first batch's hidden layer"),
                         measured("each column, down the batch"),
                         measured("each row, across the features"),
                         measured("the means after each"))

    def heading_for(self, k, u):
        return ("h: the hidden numbers of 64 digits", "BatchNorm: each column, down the batch",
                "LayerNorm: the same h, each row, across", "the same block, two directions")[k]

    def subtitle_for(self, k):
        m = self.meta
        c, r = m["col_before"], m["row_before"]
        return [
            "h is the trained model's hidden layer for the first batch: 64 digits, 64 numbers "
            "each.",
            "BatchNorm1d works down each column: column %d's mean %s and spread %s become 0 "
            "and 1." % (m["col"], val(c[0]), val(c[1])),
            "LayerNorm works across each row: row %d's mean %s and spread %s become 0 and 1."
            % (m["row"], val(r[0]), val(r[1])),
            "After BatchNorm every column's mean is 0, after LayerNorm every row's. The other "
            "means are not.",
        ][k]

    # ------------------------------------------------------------------ the block's state
    def done(self, u, count):
        """How many columns (rows) the sweep has reached at ``u``."""
        return int(np.floor(count * eased(u, *self.SWEEP) + 1e-9))

    def values(self, k, u):
        """The block as it is at ``u`` of phase ``k``."""
        if k == 0:
            return self.h
        if k == 1:
            v = self.h.copy()
            n = self.done(u, self.n)
            v[:, :n] = self.bn[:, :n]
            if u >= self.ONE:
                v[:, self.col] = self.bn[:, self.col]
            return v
        if k == 2:
            v = self.h.copy()
            n = self.done(u, self.n)
            v[:n] = self.ln[:n]
            if u >= self.ONE:
                v[self.row] = self.ln[self.row]
            return v
        return self.ln

    def side(self):
        return self.n * self.TILE

    def edges(self, at=None, tile=None):
        """``(left, top, right, bottom)`` of the block at ``at`` with ``tile`` px tiles."""
        fx, fy = self.BLOCK if at is None else at
        dw, dh = self.frac(self.n * (tile or self.TILE), self.n * (tile or self.TILE))
        return fx - dw / 2, fy - dh / 2, fx + dw / 2, fy + dh / 2

    def column_x(self, c, at=None, tile=None):
        left, _t, right, _b = self.edges(at, tile)
        return left + (c + 0.5) * (right - left) / self.n

    def row_y(self, r, at=None, tile=None):
        _l, top, _r, bottom = self.edges(at, tile)
        return top + (r + 0.5) * (bottom - top) / self.n

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        if k < 3:
            fx, fy = self.BLOCK
            tile = self.TILE
            if k == 2 and u < 0.12:           # back to h: BatchNorm's result fades off it
                self.tiles(sc, self.bn, fx, fy, self.side(), self.side(),
                           op * (1 - smoothstep(0.02, 0.1, u)), lo=self.lo, hi=self.hi,
                           grid=0.0, lift=0.35)
            self.tiles(sc, self.values(k, u), fx, fy, self.side(), self.side(), op, lo=self.lo,
                       hi=self.hi, grid=0.0, lift=0.3)
        else:                                 # LayerNorm's result moves aside, smaller
            s = eased(u, 0.03, 0.18)
            fx = self.BLOCK[0] + (self.LN_AT[0] - self.BLOCK[0]) * s
            fy = self.BLOCK[1] + (self.LN_AT[1] - self.BLOCK[1]) * s
            tile = self.TILE + (self.SMALL - self.TILE) * s
            self.tiles(sc, self.ln, fx, fy, self.n * tile, self.n * tile, op, lo=self.lo,
                       hi=self.hi, grid=0.0, lift=0.3)
        left, top, right, bottom = self.edges((fx, fy), tile)
        w, h = self.span(right - left, bottom - top)
        self.rect(sc, self.P(fx, fy), w + 6, h + 6, DATA, op, border=DATA, border_w=1.6,
                  radius=3, lift=0.2, fill_op=0.0)
        # the column means under the block (phases 1 and 2)
        cm = op * self.kept(0, 1)
        if cm > 0.01:
            v = self.values(k, u).mean(0)
            y = bottom + self.frac(0, self.STRIP_GAP)[1]
            self.tiles(sc, v[None, :], fx, y, w, 10.0, cm, lo=self.lo, hi=self.hi, grid=0.0)
            self.rect(sc, self.P(fx, y), w + 4, 14.0, DATA, cm, border=DATA, border_w=1.2,
                      radius=2, lift=0.25, fill_op=0.0)
            self.text(sc, left, y + 0.042, "column means", cm, SIZE, "lm", ink=DATA_INK)
        # the row means beside it (phase 3)
        rm = op * self.kept(2, 2)
        if rm > 0.01:
            v = self.values(k, u).mean(1)
            x = right + self.frac(self.STRIP_GAP)[0]
            self.tiles(sc, v[:, None], x, fy, 10.0, h, rm, lo=self.lo, hi=self.hi, grid=0.0)
            self.rect(sc, self.P(x, fy), 14.0, h + 4, DATA, rm, border=DATA, border_w=1.2,
                      radius=2, lift=0.25, fill_op=0.0)
            self.text(sc, x, bottom + 0.042, "row means", rm, SIZE, "mm", ink=DATA_INK)

    # ------------------------------------------------------------------ 1: the block h
    def phase_1(self, sc, u, op):
        left, top, right, bottom = self.edges()
        d = on(u, 0.08, op)
        self.dims(sc, (left - 0.016, top), (left - 0.016, bottom), "64", d, side=-1)
        self.dims(sc, (left, top - 0.025), (right, top - 0.025), "64", d, side=-1, gap=6)
        self.text(sc, self.RIGHT, 0.45, self.meta["shape_text"], on(u, 0.12, op), HEAD, "lm",
                  strong=True)
        r = on(u, 0.3, op)
        self.frame(sc, (left + right) / 2, self.row_y(self.row), self.span(right - left)[0] + 4,
                   self.TILE + 4, r)
        self.text(sc, self.RIGHT, 0.53, "a row: a digit", r, SIZE, "lm")
        c = on(u, 0.5, op)
        self.frame(sc, self.column_x(self.col), (top + bottom) / 2, self.TILE + 4,
                   self.span(0, bottom - top)[1] + 4, c)
        self.text(sc, self.RIGHT, 0.58, "a column: a feature", c, SIZE, "lm")

    # ------------------------------------------------------------------ 2: BatchNorm
    def phase_2(self, sc, u, op):
        m = self.meta
        left, top, _right, bottom = self.edges()
        x = self.column_x(self.col)
        a = on(u, 0.06, op)
        self.frame(sc, x, (top + bottom) / 2, self.TILE + 4, self.span(0, bottom - top)[1] + 4, a)
        self.arrow(sc, (left - 0.022, top), (left - 0.022, bottom), INK, a,
                   u=eased(u, 0.06, 0.16), width=2.6)
        self.text(sc, self.RIGHT, 0.45, "column %d" % m["col"], on(u, 0.12, op), HEAD, "lm",
                  strong=True)
        before, after = m["col_before"], m["col_after"]
        n = on(u, 0.18, op)
        self.running(sc, self.RIGHT, 0.56, [(0.0, val(before[0])), (self.ONE, val(after[0]))],
                     u, n, label="mean", size=BIG)
        self.running(sc, self.RIGHT, 0.69, [(0.0, val(before[1])), (self.ONE, val(after[1]))],
                     u, n, label="spread", size=BIG)
        self.text(sc, self.RIGHT, 0.77, "all %d columns" % self.n, on(u, self.SWEEP[1], op),
                  SIZE, "lm", strong=True)

    # ------------------------------------------------------------------ 3: LayerNorm
    def phase_3(self, sc, u, op):
        m = self.meta
        left, top, right, _bottom = self.edges()
        y = self.row_y(self.row)
        a = on(u, 0.12, op)
        self.frame(sc, (left + right) / 2, y, self.span(right - left)[0] + 4, self.TILE + 4, a)
        self.arrow(sc, (left, top - 0.035), (right, top - 0.035), INK, a,
                   u=eased(u, 0.12, 0.22), width=2.6)
        self.text(sc, self.RIGHT + 0.03, 0.45, "row %d" % m["row"], on(u, 0.16, op), HEAD, "lm",
                  strong=True)
        before, after = m["row_before"], m["row_after"]
        n = on(u, 0.2, op)
        x = self.RIGHT + 0.03
        self.running(sc, x, 0.56, [(0.0, val(before[0])), (self.ONE, val(after[0]))],
                     u, n, label="mean", size=BIG)
        self.running(sc, x, 0.69, [(0.0, val(before[1])), (self.ONE, val(after[1]))],
                     u, n, label="spread", size=BIG)
        self.text(sc, x, 0.77, "all %d rows" % self.n, on(u, self.SWEEP[1], op), SIZE, "lm",
                  strong=True)

    # ------------------------------------------------------------------ 4: two directions
    def phase_4(self, sc, u, op):
        m = self.meta
        b = on(u, 0.22, op)
        fx, fy = self.BN_AT
        side = self.n * self.SMALL
        self.tiles(sc, self.bn, fx, fy, side, side, b, lo=self.lo, hi=self.hi, grid=0.0)
        bl, bt, _br, bb = self.edges(self.BN_AT, self.SMALL)
        self.rect(sc, self.P(fx, fy), side + 6, side + 6, DATA, b, border=DATA, border_w=1.6,
                  radius=3, lift=0.2, fill_op=0.0)
        self.arrow(sc, (bl - 0.02, bt), (bl - 0.02, bb), INK, b, u=eased(u, 0.24, 0.32),
                   width=2.6)
        self.text(sc, fx, bt - 0.075, "BatchNorm", b, SIZE, strong=True)
        ll, lt, lr, lb = self.edges(self.LN_AT, self.SMALL)
        c = on(u, 0.42, op)
        self.arrow(sc, (ll, lt - 0.03), (lr, lt - 0.03), INK, c, u=eased(u, 0.42, 0.5), width=2.6)
        self.text(sc, self.LN_AT[0], lt - 0.075, "LayerNorm", c, SIZE, strong=True)
        d = on(u, 0.6, op)
        self.text(sc, bl, bb + 0.045, "column means: 0", d, SIZE, "lm", strong=True)
        self.text(sc, ll, lb + 0.045, "row means: 0", d, SIZE, "lm", strong=True)
        e = on(u, 0.75, op)
        lo, hi = m["bn_row_means_range"]
        self.text(sc, bl, bb + 0.085, "row means:", e, SIZE, "lm")
        self.text(sc, bl, bb + 0.125, "%s to %s" % (val(lo), val(hi)), e, SIZE, "lm")
        lo, hi = m["ln_col_means_range"]
        self.text(sc, ll, lb + 0.085, "column means:", e, SIZE, "lm")
        self.text(sc, ll, lb + 0.125, "%s to %s" % (val(lo), val(hi)), e, SIZE, "lm")


# ====================================================================== 195
class Embedding(Parts):
    """The table; one label, one row; the batch; one-hot times the table."""

    TITLE = "nn.Embedding: a table you look up"
    PHASES = (("the table", 12.0), ("one lookup", 16.0), ("the batch", 14.0),
              ("not a product", 13.0))
    LAST = "BatchNorm and LayerNorm"
    NEXT = "nn.MultiheadAttention as blocks"
    STATION = ("forward",) * 4
    TABLE = (0.215, 0.55)
    TILE = 16.0
    STRIP_Y = 0.765
    RIGHT = 0.365
    BOX = (0.41, 0.47)                # the label's box, phase 2
    SECOND = 0.55                     # u: the second label
    LABELS_X, OUT_X = 0.37, 0.43      # phase 3: the labels' column and the block
    OUT_TOP, OUT_TILE = 0.43, (5.0, 3.0)
    FILL = (0.12, 0.7)                # u: phase 3's rows arrive

    def setup(self):
        m = self.meta = self.study.metadata
        p = self.study.panels
        self.table = np.asarray(p[0].data["values"], float)
        self.vectors = np.asarray(p[1].data["values"], float)
        self.labels = np.asarray(m["labels"])
        self.lo, self.hi = m["low"], m["high"]
        self.rows, self.width = self.table.shape
        self.cell = self.cells(*self.TABLE, self.width * self.TILE, self.rows * self.TILE,
                               self.rows, self.width)
        self.hot = m["second"]                                  # the one-hot row, phase 4
        self.CODE = (Code("embedding", run=((0.0, 0),)),
                     Code("embedding", run=((0.0, 0), (0.12, 1))),
                     Code("embedding", run=((0.0, 1),), out=((0.78, m["printed"]),)),
                     Code("embedding", run=((0.0, 1),)))
        self.EVIDENCE = (exact("10 rows of 16, random (seeded)"),
                         exact("the row, copied: no arithmetic"),
                         exact("the first batch's 64 labels"),
                         exact("one-hot × table = the same row"))

    def heading_for(self, k, u):
        return ("a table: one row per label", "a label picks its row",
                "64 labels, 64 rows: " + self.meta["out_text"],
                "a lookup, not a multiplication")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "nn.Embedding(10, 16) is a table of 10 rows of 16 numbers, started at random "
            "(seeded here).",
            "Label %d picks row %d and label %d row %d. The row is copied out: nothing is "
            "multiplied." % (m["first"], m["first"], m["second"], m["second"]),
            "The batch's %d labels pick %d rows: %s becomes %s. Equal labels give equal rows."
            % (m["lookups"], m["lookups"], m["labels_text"], m["out_text"]),
            "A one-hot row times the table gives row %d exactly, with %d multiplications, %d "
            "of them by 0." % (m["second"], m["multiplications_one"], m["by_zero"]),
        ][k]

    # ------------------------------------------------------------------ geometry
    def edges(self):
        fx, fy = self.TABLE
        dw, dh = self.frac(self.width * self.TILE, self.rows * self.TILE)
        return fx - dw / 2, fy - dh / 2, fx + dw / 2, fy + dh / 2

    def row_y(self, r):
        return self.cell(r, 0)[1]

    def row_tiles(self, sc, values, fy, op, lift=0.6, zero=None):
        fx = self.TABLE[0]
        w = self.width * self.TILE
        self.tiles(sc, np.atleast_2d(values), fx, fy, w, self.TILE, op, ink=WEIGHT, lo=self.lo,
                   hi=self.hi, lift=lift, zero=zero)
        self.frame(sc, fx, fy, w + 4, self.TILE + 4, op, ink=WEIGHT_INK, lift=lift + 0.1,
                   width=1.6)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        left, top, right, bottom = self.edges()
        zero = None
        if k == 3:
            g = smoothstep(0.24, 0.3, u)
            if g > 0.5:
                zero = np.ones_like(self.table, bool)
                zero[self.hot] = False
        self.tiles(sc, self.table, *self.TABLE, self.width * self.TILE, self.rows * self.TILE,
                   op, ink=WEIGHT, lo=self.lo, hi=self.hi, zero=zero, grid=0.25)
        w, h = self.span(right - left, bottom - top)
        self.rect(sc, self.P(*self.TABLE), w + 8, h + 8, WEIGHT_INK, op, border=WEIGHT_INK,
                  border_w=1.6, radius=4, lift=0.2, fill_op=0.0)
        for r in range(self.rows):
            self.text(sc, left - 0.014, self.row_y(r), str(r), op, SIZE, "rm")

    # ------------------------------------------------------------------ 1: the table
    def phase_1(self, sc, u, op):
        m = self.meta
        left, _top, right, bottom = self.edges()
        d = on(u, 0.1, op)
        self.dims(sc, (left, bottom + 0.03), (right, bottom + 0.03), str(m["width"]), d, side=1,
                  gap=6)
        self.text(sc, self.RIGHT, 0.43, "%d rows of %d" % (m["rows"], m["width"]), d, SIZE,
                  "lm", strong=True)
        self.big(sc, self.RIGHT, 0.56, str(m["numbers"]), on(u, 0.3, op),
                 label="numbers", size=BIG, ink=WEIGHT_INK)
        self.text(sc, self.RIGHT, 0.62, "it learns", on(u, 0.3, op), SIZE, "lm")
        self.text(sc, self.RIGHT, 0.7, "random at first", on(u, 0.55, op), SIZE, "lm")
        self.text(sc, self.RIGHT, 0.745, "(seeded)", on(u, 0.55, op), SIZE, "lm")

    # ------------------------------------------------------------------ 2: one lookup
    def lookup(self, sc, u, op, label, start):
        """Label ``label``'s box, its arrow to the row, the row ringed and copied down."""
        _left, _top, right, _bottom = self.edges()
        y = self.row_y(label)
        bx, by = self.BOX
        a = on(u, start + 0.08, op)
        self.arrow(sc, (bx - self.frac(32)[0], by), (right + 0.012, y), AXIS, a,
                   u=eased(u, start + 0.08, start + 0.16), width=2.0)
        self.frame(sc, self.TABLE[0], y, self.width * self.TILE + 6, self.TILE + 6,
                   on(u, start + 0.14, op))
        s = eased(u, start + 0.2, start + 0.3)
        c = on(u, start + 0.2, op)
        self.row_tiles(sc, self.table[label], y + (self.STRIP_Y - y) * s, c)
        return c * smoothstep(start + 0.3, start + 0.34, u)

    def phase_2(self, sc, u, op):
        m = self.meta
        first, second = m["first"], m["second"]
        bx, by = self.BOX
        box = on(u, 0.04, op)
        label = first if u < self.SECOND else second
        self.box(sc, bx, by, 58, 28, PAPER, box, text="y = %d" % label, border=DATA,
                 size=SIZE, ink=DATA_INK)
        if u < self.SECOND:
            shown = self.lookup(sc, u, op * (1 - smoothstep(self.SECOND - 0.04, self.SECOND, u)),
                                first, 0.0)
            row = self.table[first]
        else:
            # the first row stays down there until the second one lands on it
            s2 = self.SECOND
            keep = 1 - smoothstep(s2 + 0.28, s2 + 0.31, u)
            self.row_tiles(sc, self.table[first], self.STRIP_Y, op * keep, lift=0.5)
            shown = self.lookup(sc, u, op, second, s2 - 0.06)
            row = self.table[second] if shown > 0.01 else self.table[first]
            shown = max(shown, op * keep)
        nums = "  ".join(val(v) for v in row[:4]) + "  …"
        self.text(sc, self.edges()[0], self.STRIP_Y + 0.05, nums, shown, SIZE, "lm",
                  ink=WEIGHT_INK)
        self.big(sc, self.RIGHT, 0.83, "0", on(u, 0.4, op), label="multiplications", size=BIG)

    # ------------------------------------------------------------------ 3: the batch
    def out_cells(self):
        tw, th = self.OUT_TILE
        n = len(self.labels)
        h = self.frac(0, n * th)[1]
        return self.OUT_TOP, self.OUT_TOP + h, tw, th

    def phase_3(self, sc, u, op):
        m = self.meta
        top, bottom, tw, th = self.out_cells()
        n = len(self.labels)
        cy = (top + bottom) / 2
        a = on(u, 0.04, op)
        filled = int(np.floor(n * eased(u, *self.FILL) + 1e-9))
        # the labels, a column of blue tiles, all there from the start
        self.tiles(sc, self.labels[:, None].astype(float), self.LABELS_X, cy, 10.0, n * th, a,
                   lo=0, hi=9, grid=0.0)
        self.text(sc, self.LABELS_X, top - 0.025, "y", a, SIZE, ink=DATA_INK)
        # the block of rows, filled from the top
        w = self.width * tw
        self.rect(sc, self.P(self.OUT_X, cy), w + 6, n * th + 6, WEIGHT_INK, a,
                  border=WEIGHT_INK, border_w=1.4, radius=3, lift=0.2, fill_op=0.0)
        if filled:
            part = self.frac(0, filled * th)[1]
            self.tiles(sc, self.vectors[:filled], self.OUT_X, top + part / 2, w, filled * th, a,
                       ink=WEIGHT, lo=self.lo, hi=self.hi, grid=0.0)
        self.text(sc, self.OUT_X, top - 0.025, "embed(y)", a, SIZE, ink=WEIGHT_INK)
        self.running(sc, self.edges()[0], 0.83, [(0.0, str(filled))], 0.0, a, label="lookups",
                     size=BIG)
        s = on(u, 0.78, op)
        self.text(sc, self.edges()[2], 0.83, "%s → %s" % (m["labels_text"], m["out_text"]), s,
                  SIZE, "rm", strong=True)
        # labels 7, 8 and 9 of this batch are all 1: three equal rows
        run = self.same_run()
        if run is not None:
            i, j = run
            r = on(u, 0.86, op)
            ys = top + self.frac(0, (i + j) / 2 * th)[1]
            self.frame(sc, (self.LABELS_X + self.OUT_X + self.frac(w / 2)[0]) / 2 - 0.006, ys,
                       self.span(self.OUT_X + self.frac(w / 2)[0] - self.LABELS_X)[0] + 18,
                       (j - i) * th + 4, r, width=1.8)

    def same_run(self):
        """The first run of three equal labels in a row: ``(first, last + 1)``."""
        y = self.labels
        for i in range(len(y) - 2):
            if y[i] == y[i + 1] == y[i + 2]:
                return i, i + 3
        return None

    # ------------------------------------------------------------------ 4: not a product
    def phase_4(self, sc, u, op):
        m = self.meta
        left, top, right, _bottom = self.edges()
        hot = np.zeros((self.rows, 1))
        hot[self.hot] = 1.0
        a = on(u, 0.05, op)
        x = right + 0.035
        self.tiles(sc, hot, x, self.TABLE[1], self.TILE, self.rows * self.TILE, a, lo=0, hi=1,
                   grid=0.4)
        self.text(sc, x - 0.015, top - 0.025, "one-hot of %d" % self.hot, a, SIZE, "lm",
                  ink=DATA_INK)
        self.text(sc, x + 0.02, self.row_y(self.hot), "1", a, SIZE, "lm", strong=True)
        self.text(sc, x + 0.02, self.row_y(0), "0", a, SIZE, "lm")
        g = on(u, 0.24, op)
        self.frame(sc, self.TABLE[0], self.row_y(self.hot), self.width * self.TILE + 6,
                   self.TILE + 6, g)
        self.text(sc, x + 0.04, self.row_y(4), "× 0", g, SIZE, "lm", ink=dark(ZERO, 0.4))
        s = eased(u, 0.42, 0.52)
        c = on(u, 0.42, op)
        y = self.row_y(self.hot)
        self.row_tiles(sc, self.table[self.hot], y + (self.STRIP_Y - y) * s, c)
        self.text(sc, right + 0.02, self.STRIP_Y, "row %d" % self.hot,
                  c * smoothstep(0.52, 0.56, u), SIZE, "lm", strong=True)
        self.big(sc, left, 0.875, str(m["multiplications_one"]), on(u, 0.6, op),
                 label="one-hot × table", size=BIG, ink=WEIGHT_INK)
        self.big(sc, left + 0.19, 0.875, "0", on(u, 0.76, op), label="lookup", size=BIG)


# ====================================================================== 196
ATTN = "attention"


class Attention(Parts):
    """The tokens; Q, K and V; the scores; the output."""

    TITLE = "nn.MultiheadAttention as blocks"
    PHASES = (("5 tokens", 12.0), ("Q, K and V", 14.0), ("the scores", 17.0),
              ("the output", 14.0))
    LAST = "nn.Embedding: a table you look up"
    NEXT = "Logits and softmax"
    STATION = ("forward",) * 4
    DIGIT = (0.15, 0.60)
    DIGIT_TILE = 4.0
    X1, X1_TILE = (0.355, 0.60), 9.0  # the tokens, phase 1
    X2, TILE = (0.13, 0.62), 7.0      # the tokens from phase 2, and Q, K, V
    QKV_X, QKV_Y = 0.39, (0.47, 0.625, 0.78)
    W_X, W_TILE = 0.27, 3.0
    G = ((0.115, 0.56), (0.245, 0.56))   # the two heads' grids
    G_TILE = 20.0
    READ = 0.715                      # the readouts' y, phases 3 and 4
    SOFT = 0.42                       # u: softmax, phase 3

    def setup(self):
        m = self.meta = self.study.metadata
        self.img = np.asarray(m["digit_image"], float)
        self.x = np.asarray(m["x"], float)
        self.q, self.k, self.v = (np.asarray(m[n], float) for n in ("q", "k", "v"))
        self.raw = np.asarray(m["raw"], float)
        self.probs = np.asarray(m["probs"], float)
        self.out = np.asarray(m["out"], float)
        qkv = np.concatenate([self.q, self.k, self.v, self.out])
        self.lo, self.hi = float(qkv.min()), float(qkv.max())
        self.T, self.D = self.x.shape
        self.hd = m["head_dim"]
        self.qi, self.ki = m["query"], m["key"]
        self.dcell = self.digit_cells(*self.DIGIT, self.DIGIT_TILE)
        w = np.asarray(m["in_proj_weight"], float)
        self.weights = [w[i * self.D:(i + 1) * self.D] for i in range(3)]
        self.w_lo, self.w_hi = float(w.min()), float(w.max())
        self.w_out = np.asarray(m["out_proj_weight"], float)
        self.CODE = (Code(ATTN, run=((0.0, 0),)),
                     Code(ATTN, run=((0.0, 1), (0.4, 3))),
                     Code(ATTN, run=((0.0, 3),), out=((0.84, "scores: " + m["printed"][1]),)),
                     Code(ATTN, run=((0.0, 3),), out=((0.5, "out: " + m["printed"][0]),)))
        self.EVIDENCE = (measured("5 patches of a real digit"),
                         exact("untrained weights, seeded"),
                         exact("each row of scores adds up to 1"),
                         exact("each output row mixes V's rows"))

    def heading_for(self, k, u):
        return ("x: 5 patches of a digit, not randn", "three grids make Q, K and V",
                "Q against K: a grid of scores", "out: V's rows, mixed by the scores")[k]

    def subtitle_for(self, k):
        m = self.meta
        p = m["query_row"]
        return [
            "Where the code makes up random numbers, each digit gives 5 patches of 4 × 4: 5 "
            "tokens of 16.",
            "Three grids of 16 × 16 weights turn the tokens into Q, K and V, each cut into %d "
            "heads of %d." % (m["heads"], m["head_dim"]),
            "Head 1 scores token %d against token %d as %s ÷ %s = %s; softmax makes each row "
            "add up to 1." % (m["query"] + 1, m["key"] + 1, val(m["dot"]), val(m["scale"]),
                              val(m["raw_cell"])),
            "Each output row mixes V's rows by its scores: in head 1, token %d takes %s of token "
            "%d's."
            % (m["query"] + 1, val(p[m["key"]]), m["key"] + 1),
        ][k]

    # ------------------------------------------------------------------ geometry
    def block(self, sc, values, at, tile, op, ink=DATA, lo=None, hi=None, ring=(), split=False,
              lift=0.3):
        """A ``T x 16`` block of tiles centred at ``at``; ``split``: a line between the two
        heads' halves. Returns ``cell``."""
        a = np.atleast_2d(values)
        rows, cols = a.shape
        w, h = cols * tile, rows * tile
        lo = self.lo if lo is None else lo
        hi = self.hi if hi is None else hi
        cell = self.tiles(sc, a, at[0], at[1], w, h, op, ink=ink, lo=lo, hi=hi, ring=ring,
                          grid=0.3, lift=lift)
        if op > 0.01:
            self.rect(sc, self.P(*at), w + 6, h + 6, dark(ink, 0.2), op, border=dark(ink, 0.2),
                      border_w=1.5, radius=3, lift=lift - 0.1, fill_op=0.0)
            if split:
                x = at[0]
                dy = self.frac(0, h / 2 + 3)[1]
                self.polyline(sc, [self.P(x, at[1] - dy), self.P(x, at[1] + dy)], 2.4, INK, split,
                              bias=lift + 0.5)
        return cell

    def half(self, at, tile, head, rows=None, row=None):
        """``(fx, fy, w px, h px)`` of head ``head``'s half of a block at ``at`` (one row:
        ``row``)."""
        rows = self.T if rows is None else rows
        dw = self.frac(self.hd * tile)[0]
        fx = at[0] - dw / 2 if head == 0 else at[0] + dw / 2
        if row is None:
            return fx, at[1], self.hd * tile + 4, rows * tile + 4
        fy = at[1] + self.frac(0, (row - (rows - 1) / 2) * tile)[1]
        return fx, fy, self.hd * tile + 4, tile + 4

    def x_at(self, k, u):
        """Where the tokens are and their tile size."""
        if k == 0:
            return self.X1, self.X1_TILE
        if k > 1:
            return self.X2, self.TILE
        s = eased(u, 0.03, 0.16)
        return ((self.X1[0] + (self.X2[0] - self.X1[0]) * s,
                 self.X1[1] + (self.X2[1] - self.X1[1]) * s),
                self.X1_TILE + (self.TILE - self.X1_TILE) * s)

    def grid_values(self, head, k, u):
        if k == 2 and head == 0 and u < self.SOFT:
            return self.raw[0], float(self.raw.min()), float(self.raw.max())
        return self.probs[head], 0.0, float(self.probs.max())

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        # the digit (phase 1)
        d = op * self.kept(0, 0)
        if d > 0.01:
            self.digit_card(sc, self.img, *self.DIGIT, self.DIGIT_TILE, d)
        # the tokens (phases 1 and 2)
        t = op * self.kept(0, 1)
        if t > 0.01:
            at, tile = self.x_at(k, u)
            self.block(sc, self.x, at, tile, t, lo=DATA_LO, hi=DATA_HI)
            self.text(sc, at[0] - self.frac(self.D * tile / 2 + 8)[0], at[1] - self.frac(
                0, self.T * tile / 2 + 14)[1], "x", t, SIZE, "lm", ink=DATA_INK)
        # Q, K (phases 2 and 3) and V (2 to 4)
        for i, name in enumerate("QKV"):
            s = op * (self.kept(1, 2) if i < 2 else self.kept(1, 3))
            if k == 1:
                s *= smoothstep(0.3 + 0.1 * i, 0.36 + 0.1 * i, u)
            if s <= 0.01:
                continue
            at = (self.QKV_X, self.QKV_Y[i])
            split = s * (smoothstep(0.7, 0.76, u) if k == 1 else 1.0)
            self.block(sc, (self.q, self.k, self.v)[i], at, self.TILE, s, split=split)
            self.text(sc, at[0] - self.frac(self.D * self.TILE / 2 + 3)[0],
                      at[1] - self.frac(0, self.T * self.TILE / 2 + 14)[1], name, s, SIZE, "lm",
                      strong=True)
        # the two heads' grids (phases 3 and 4)
        g = op * self.kept(2, 3)
        if g > 0.01:
            for head in range(m["heads"]):
                shown = g if k == 3 or head == 0 else g * smoothstep(0.62, 0.68, u)
                values, lo, hi = self.grid_values(head, k, u)
                at = self.G[head]
                side = self.T * self.G_TILE
                self.block(sc, values, at, self.G_TILE, shown, lo=lo, hi=hi)
                self.text(sc, at[0], at[1] - self.frac(0, side / 2 + 16)[1], "head %d" % (head + 1),
                          shown, SIZE, strong=True)

    # ------------------------------------------------------------------ 1: 5 tokens
    def phase_1(self, sc, u, op):
        m = self.meta
        p = m["patch"]
        at, tile = self.X1, self.X1_TILE
        top = at[1] - self.frac(0, self.T * tile / 2)[1]
        left = at[0] - self.frac(self.D * tile / 2)[0]
        for i, (r, c) in enumerate(m["corners"]):
            a = on(u, 0.12 + 0.1 * i, op)
            box = self.outline(sc, self.dcell, r, c, p, p, self.DIGIT_TILE, a, lift=1.1)
            ry = top + self.frac(0, (i + 0.5) * tile)[1]
            self.arrow(sc, ((box[0] + box[2]) / 2, box[1]), (left - 0.008, ry), AXIS, a * 0.8,
                       u=eased(u, 0.12 + 0.1 * i, 0.18 + 0.1 * i), width=1.4, head=True)
        s = on(u, 0.66, op)
        self.text(sc, at[0], top - 0.06, m["x_text"], s, HEAD, strong=True)
        self.text(sc, at[0], at[1] + self.frac(0, self.T * tile / 2)[1] + 0.045,
                  "5 tokens of 4 × 4 = 16", on(u, 0.5, op), SIZE)
        self.text(sc, self.DIGIT[0] - 0.08, 0.82, "a %d from the first batch" % m["label"],
                  on(u, 0.05, op), SIZE, "lm", ink=DATA_INK)

    # ------------------------------------------------------------------ 2: Q, K and V
    def phase_2(self, sc, u, op):
        right = self.X2[0] + self.frac(self.D * self.TILE / 2 + 4)[0]
        side = 16 * self.W_TILE
        for i in range(3):
            a = on(u, 0.24 + 0.1 * i, op)
            y = self.QKV_Y[i]
            self.tiles(sc, self.weights[i], self.W_X, y, side, side, a, ink=WEIGHT,
                       lo=self.w_lo, hi=self.w_hi, grid=0.0, lift=0.3)
            self.rect(sc, self.P(self.W_X, y), side + 4, side + 4, WEIGHT_INK, a,
                      border=WEIGHT_INK, border_w=1.4, radius=2, lift=0.25, fill_op=0.0)
            self.arrow(sc, (right + 0.005, self.X2[1]), (self.W_X - self.frac(side / 2 + 5)[0], y),
                       AXIS, a * 0.8, u=eased(u, 0.24 + 0.1 * i, 0.3 + 0.1 * i), width=1.4)
            self.arrow(sc, (self.W_X + self.frac(side / 2 + 4)[0], y),
                       (self.QKV_X - self.frac(self.D * self.TILE / 2 + 6)[0], y), AXIS, a * 0.8,
                       u=eased(u, 0.27 + 0.1 * i, 0.33 + 0.1 * i), width=1.4)
        self.text(sc, self.W_X, 0.86, "3 grids of 16 × 16", on(u, 0.55, op), SIZE,
                  ink=WEIGHT_INK)
        self.text(sc, self.QKV_X + self.frac(self.D * self.TILE / 2)[0], 0.86,
                  "2 heads: 8 + 8", on(u, 0.72, op), SIZE, "rm")

    # ------------------------------------------------------------------ 3: the scores
    def phase_3(self, sc, u, op):
        m = self.meta
        hd = self.hd
        a = on(u, 0.12, op)
        # head 1's halves of Q and K
        for i in range(2):
            fx, fy, w, h = self.half((self.QKV_X, self.QKV_Y[i]), self.TILE, 0)
            self.frame(sc, fx, fy, w, h, a * (1 - smoothstep(0.6, 0.66, u)), lift=1.2)
        # one score: a query row against a key row
        c = on(u, 0.2, op) * (1 - smoothstep(self.SOFT - 0.04, self.SOFT, u))
        for i, r in ((0, self.qi), (1, self.ki)):
            fx, fy, w, h = self.half((self.QKV_X, self.QKV_Y[i]), self.TILE, 0, row=r)
            self.frame(sc, fx, fy, w, h, c, lift=1.3, width=2.6)
        gx, gy = self.G[0]
        cell = self.cells(gx, gy, self.T * self.G_TILE, self.T * self.G_TILE, self.T, self.T)
        x, y = cell(self.qi, self.ki)
        self.frame(sc, x, y, self.G_TILE + 4, self.G_TILE + 4, on(u, 0.2, op) * (
            1 - smoothstep(self.SOFT - 0.04, self.SOFT, u)), lift=1.3, width=2.6)
        self.text(sc, 0.063, self.READ, "q · k ÷ √%d = %s ÷ %s = %s" % (
            hd, val(m["dot"]), val(m["scale"]), val(m["raw_cell"])), c, SIZE, "lm")
        # softmax: the row, then every row adds up to 1
        s = on(u, self.SOFT, op)
        _x0, y = cell(self.qi, 0)
        self.frame(sc, gx, y, self.T * self.G_TILE + 4, self.G_TILE + 4, s, lift=1.3, width=2.6)
        p = m["query_row"]
        self.text(sc, 0.063, self.READ, "row %d: " % (self.qi + 1) + "  ".join(val(v) for v in p),
                  s, SIZE, "lm")
        self.text(sc, 0.063, self.READ + 0.045, "adds up to %s" % val(sum(p)),
                  on(u, self.SOFT + 0.08, op), SIZE, "lm", strong=True)
        self.text(sc, 0.063, self.READ + 0.09, "scores: the 2 grids, averaged",
                  on(u, 0.8, op), SIZE, "lm")

    # ------------------------------------------------------------------ 4: the output
    def phase_4(self, sc, u, op):
        m = self.meta
        p = np.asarray(m["query_row"])
        a = on(u, 0.06, op)
        gx, gy = self.G[0]
        cell = self.cells(gx, gy, self.T * self.G_TILE, self.T * self.G_TILE, self.T, self.T)
        _x0, y = cell(self.qi, 0)
        self.frame(sc, gx, y, self.T * self.G_TILE + 4, self.G_TILE + 4, a, lift=1.3, width=2.6)
        # V's rows, dimmed by how little of each the row takes (head 1's half)
        vat = (self.QKV_X, self.QKV_Y[2])
        b = on(u, 0.14, op)
        for r in range(self.T):
            fx, fy, w, h = self.half(vat, self.TILE, 0, row=r)
            self.veil(sc, fx, fy, w - 4, h - 4, b * 0.75 * (1 - p[r] / p.max()), lift=0.6)
        fx, fy, w, h = self.half(vat, self.TILE, 0, row=self.ki)
        self.frame(sc, fx, fy, w, h, b, lift=1.2, width=2.6)
        terms = " + ".join("%s V%d" % (val(p[j]), j + 1) for j in range(2)) + " + …"
        self.text(sc, 0.063, self.READ, "head 1: " + terms, b, SIZE, "lm")
        # the output, where Q was
        oat = (self.QKV_X, self.QKV_Y[0])
        o = on(u, 0.36, op)
        self.block(sc, self.out, oat, self.TILE, o)
        rest = o * (1 - smoothstep(0.55, 0.62, u))         # row by row: the followed one first
        for r in range(self.T):
            if r != self.qi:
                fx, fy, w, h = self.half(oat, self.TILE, 0, row=r)
                self.veil(sc, oat[0], fy, self.D * self.TILE, self.TILE, rest, lift=0.6)
        self.text(sc, oat[0] - self.frac(self.D * self.TILE / 2 + 3)[0],
                  oat[1] - self.frac(0, self.T * self.TILE / 2 + 14)[1], "out", o, SIZE, "lm",
                  strong=True)
        # out_proj: one more grid of 16 x 16 weights, where K was
        pat = (self.QKV_X, self.QKV_Y[1])
        side = self.D * self.W_TILE
        self.tiles(sc, self.w_out, *pat, side, side, o, ink=WEIGHT, lo=self.w_lo, hi=self.w_hi,
                   grid=0.0, lift=0.3)
        self.rect(sc, self.P(*pat), side + 4, side + 4, WEIGHT_INK, o, border=WEIGHT_INK,
                  border_w=1.4, radius=2, lift=0.25, fill_op=0.0)
        self.text(sc, pat[0] + self.frac(side / 2 + 8)[0], pat[1], "out_proj", o, SIZE, "lm",
                  ink=WEIGHT_INK)
        half = self.frac(0, self.T * self.TILE / 2 + 6)[1]
        self.arrow(sc, (vat[0], vat[1] - half), (pat[0], pat[1] + self.frac(0, side / 2 + 5)[1]),
                   AXIS, o * 0.8, u=eased(u, 0.36, 0.42), width=1.6)
        self.arrow(sc, (pat[0], pat[1] - self.frac(0, side / 2 + 3)[1]), (oat[0], oat[1] + half),
                   AXIS, o * 0.8, u=eased(u, 0.42, 0.48), width=1.6)
        lk = on(u, 0.74, op)
        self.text(sc, 0.063, 0.84, "the count: “Counting the weights: a transformer”", lk,
                  SIZE, "lm")
        self.text(sc, 0.063, 0.88, "a real one: “Inside GPT-2”", on(u, 0.82, op), SIZE, "lm")


ANIMATIONS = {"blocks-normalisation": Normalisation, "blocks-embedding": Embedding,
              "blocks-attention": Attention}

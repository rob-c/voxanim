"""Animations 234-235, the first two of "Blocks and code: vector databases" (see
:mod:`voxanim.blocks`), drawn from :mod:`voxanim.study_blocks_r`.

* :class:`Training` (``vector-db-training``): a pair, two drawings of one digit, stays on the
  left all through; the same encoder turns each into 32 numbers of length 1, and their dot
  product is one cell of the [64, 64] grid of every pair in the batch, whose diagonal (the
  right pairs) is the target; the loss in its purple meter. The grid at steps 0, 100 and
  937, the pair beside it each step's own; the right pairs' mean against the other 63's
  (0.59 against 0.49, then 0.93 against 0.36). Last, step 937's grid sorted by digit: the
  same digit off the diagonal scores 0.92, the in-batch negatives that are not really
  negative.
* :class:`StoreSearch` (``vector-db-store-and-search``): the encoder turns the 60,000
  training digits into the store, a block [60000, 32] drawn as its first rows and its last,
  with an id and a label beside each row; test digit 0, a 7, through the same encoder to a
  query; a band runs down the rows as each gets its dot product, 60,000 in all; the top 5
  with their pictures (looked up by id), labels and scores; the quality over all 10,000
  test digits against raw pixels and an untrained encoder.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). What stays across phases is drawn in
``steady``, so nothing blanks between them.
"""

from __future__ import annotations

import numpy as np

from .blocks import (
    DATA,
    HEAD,
    INK,
    LOSS,
    QUIET,
    SIZE,
    WEIGHT,
    Blocks,
    Code,
    measured,
)
from .easing import ease_in_out, smoothstep
from .mech import AXIS, dark, soft

DATA_INK = dark(DATA, 0.3)
LOSS_INK = dark(LOSS, 0.25)
WEIGHT_INK = dark(WEIGHT, 0.35)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on(u, a, op, width=0.04):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


class Vectors(Blocks):
    """What the two share: the encoder as an orange block, and a vector of 32 as a strip of
    tiles."""

    def encoder(self, sc, fx, fy, w, h, op):
        """The encoder: an orange block (its weights), "encoder" written in it."""
        if op <= 0.01:
            return
        self.rect(sc, self.P(fx, fy), w, h, soft(WEIGHT, 0.6), op, border=WEIGHT, border_w=2.0,
                  radius=6, lift=0.2)
        self.text(sc, fx, fy, "encoder", op, SIZE, ink=WEIGHT_INK)

    def vector(self, sc, values, fx, fy, tile, op, lo, hi, h=12.0, lift=0.3):
        """32 numbers as a row of ``tile`` px tiles centred at ``(fx, fy)``, outlined."""
        if op <= 0.01:
            return
        n = len(values)
        self.tiles(sc, np.asarray(values, float)[None], fx, fy, n * tile, h, op, lo=lo, hi=hi,
                   lift=lift, grid=0.25)
        self.rect(sc, self.P(fx, fy), n * tile + 3, h + 3, DATA_INK, op, border=DATA_INK,
                  border_w=1.2, radius=2, lift=lift - 0.02, fill_op=0.0)

    def picture(self, sc, img, fx, fy, tile, op, lift=0.2):
        """A digit on a data card."""
        if op > 0.01:
            self.digit_card(sc, np.asarray(img, float), fx, fy, tile, op, lift=lift)

    def vdots(self, sc, fx, fy, op):
        """Three dots one above the other: rows left out."""
        if op <= 0.01:
            return
        for dy in (-5.0, 0.0, 5.0):
            self.disk(sc, self.shift(self.P(fx, fy), 0, dy), 1.6, QUIET, op, lift=0.3,
                      outline=0.0)

    def card_h(self, tile):
        """A :meth:`~voxanim.blocks.Blocks.digit_card`'s height, layout."""
        return self.frac(0, 28 * tile + 20)[1]


# ====================================================================== 234
class Training(Vectors):
    """A pair; the encoder; the grid and its diagonal; the grid as training goes; sorted by
    digit."""

    TITLE = "Training vectors for search"
    PHASES = (("a pair", 11.0), ("the encoder", 12.0), ("the grid", 13.0),
              ("training", 13.0), ("by digit", 11.0))
    LAST = "How wide should an embedding be?"
    NEXT = "Store, then search"
    STATION = ("data", "forward", "loss", "step", "loss")
    CX, CY, CT = 0.098, (0.475, 0.715), 2.0     # the pair's cards: x, the two y, tile px
    BOX = (0.205, 0.595, 72.0, 165.0)           # phase 2: the encoder (x, y, w px, h px)
    SX, ST = 0.37, 5.0                          # phase 2: the strips' centre x, tile px
    GX, GY, CELL = 0.29, 0.615, 3.4             # the grid's centre and px per cell
    MX, MY, MH = 0.447, 0.62, 140.0             # the loss meter
    LOSS_TOP = 4.5                              # the meter's full height
    SWAP = ((0.28, 0.34), (0.58, 0.64))         # phase 4: to step 100, to step 937
    SORT = (0.08, 0.2)                          # phase 5: the grid sorted by digit
    LOW = (0.852, 0.888)                        # the two lines under the grid

    def setup(self):
        m = self.meta = self.study.metadata
        self.snap = {int(k): v for k, v in m["snap"].items()}
        self.steps = tuple(int(s) for s in m["snaps"])
        self.grids = {s: np.asarray(self.snap[s]["grid"], float) for s in self.steps}
        last = self.steps[-1]
        self.sorted = np.asarray(self.snap[last]["sorted"], float)
        self.a, self.b = np.asarray(m["a"], float), np.asarray(m["b"], float)
        self.v_hi = float(max(np.abs(self.a).max(), np.abs(self.b).max()))
        side = len(self.grids[0]) * self.CELL
        self.side = side
        self.gw, self.gh = self.frac(side, side)
        self.gx0, self.gy0 = self.GX - self.gw / 2, self.GY - self.gh / 2
        self.CODE = (
            Code("vdb-loss", run=((0.0, None), (0.3, 1))),
            Code(("vdb-encoder", "vdb-loss"), pick=tuple(range(8)),
                 run=((0.0, 0), (0.12, 2), (0.2, 3), (0.32, 5), (0.45, 7))),
            Code("vdb-loss", run=((0.0, 3), (0.45, 4), (0.62, 5))),
            Code("vdb-step", run=((0.0, 1), (0.12, 2))),
            Code("vdb-loss", run=((0.0, 4),)))
        self.EVIDENCE = (measured("step 0's first pair, two real 5s"),
                         measured("the encoder before training"),
                         measured("step 0's 64 pairs, before training"),
                         measured("steps 0, 100 and 937 of one run"),
                         measured("step 937's 64 pairs, sorted by digit"))

    def heading_for(self, k, u):
        return ("two drawings of one digit",
                "each becomes 32 numbers",
                "every pair's dot product: the diagonal is right",
                "training: the diagonal brightens",
                "sorted by digit: the same digit is close too")[k]

    def subtitle_for(self, k):
        m = self.meta
        s0, s1, s2 = (self.snap[s] for s in self.steps)
        return [
            "x[0] is a %d from the batch, p[0] another %d drawn at random: the labels only "
            "pick the pairs." % (m["pair_label"], m["pair_label"]),
            "The same encoder turns each into 32 numbers of length 1; before training their "
            "dot product is %s." % m["dot_text"],
            "All 64 × 64 dot products of the batch; the loss wants the diagonal biggest in "
            "each row: %s, near ln 64 = %s." % (s0["loss_text"], m["chance_text"]),
            "Each step has its own 64 pairs. The right pairs score %s at step 0, %s at 100, "
            "%s at 937; the other 63 fall to %s." % (s0["pos_text"], s1["pos_text"],
                                                     s2["pos_text"], s2["neg_text"]),
            "Sorted by digit: the same digit off the diagonal scores %s, though the loss "
            "counted it wrong; other digits %s." % (s2["same_text"], s2["other_text"]),
        ][k]

    # ------------------------------------------------------------------ steps 0, 100, 937
    def levels(self, k, u):
        """How far phase 4 has swapped to steps 100 and 937: ``(s1, s2)``."""
        if k < 3:
            return 0.0, 0.0
        if k > 3:
            return 1.0, 1.0
        return tuple(smoothstep(a, b, u) for a, b in self.SWAP)

    def layers(self, k, u):
        """``[(step, opacity)]``: each step's picture drawn over the one before."""
        s1, s2 = self.levels(k, u)
        out = []
        for step, under, over in zip(self.steps, (1.0, s1, s2), (s1, s2, 0.0), strict=True):
            if under > 0.01 and over < 0.999:
                out.append((step, under))
        return out

    def pick(self, k, u, key):
        """The step's value of ``key`` nearest in time (the swap is quick)."""
        s1, s2 = self.levels(k, u)
        step = self.steps[2] if s2 >= 0.5 else self.steps[1] if s1 >= 0.5 else self.steps[0]
        return self.snap[step][key]

    # ------------------------------------------------------------------ geometry
    def cell(self, r, c):
        return (self.gx0 + (c + 0.5) * self.gw / 64, self.gy0 + (r + 0.5) * self.gh / 64)

    def diagonal(self, sc, op):
        """Two violet lines either side of the diagonal: the target."""
        if op <= 0.01:
            return
        d = self.gw / 64 * 2.0
        dy = self.gh / 64 * 2.0
        x0, y0, x1, y1 = self.gx0, self.gy0, self.gx0 + self.gw, self.gy0 + self.gh
        for (ax, ay), (bx, by) in (((x0 + d, y0), (x1, y1 - dy)), ((x0, y0 + dy), (x1 - d, y1))):
            self.polyline(sc, [self.P(ax, ay, self.ARROW_LIFT), self.P(bx, by, self.ARROW_LIFT)],
                          1.6, LOSS_INK, op, bias=0.4)

    def grid(self, sc, values, op, reveal=1.0, lift=0.3):
        self.tiles(sc, values, self.GX, self.GY, self.side, self.side, op, lo=0.0, hi=1.0,
                   reveal=reveal, lift=lift, grid=0.0)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        # the pair: step 0's, then each step's own
        for i, (step, a) in enumerate(self.layers(k, u)):
            s = self.snap[step]
            for img, y in ((s["x"], self.CY[0]), (s["p"], self.CY[1])):
                self.picture(sc, img, self.CX, y, self.CT, op * a, lift=0.2 + 0.01 * i)
        label = self.pick(k, u, "label")
        for name, y in (("x[0]", self.CY[0]), ("p[0]", self.CY[1])):
            self.text(sc, self.CX, y + self.card_h(self.CT) / 2 + 0.024,
                      "%s: a %d" % (name, label), op, SIZE, ink=DATA_INK)
        if k < 2:
            return
        # the grid, from phase 3 on: step 0's rows filling in, then each step's
        g = op * self.kept(2)
        reveal = eased(u, 0.05, 0.3) if k == 2 else 1.0
        for i, (step, a) in enumerate(self.layers(k, u)):
            self.grid(sc, self.grids[step], g * a, reveal, lift=0.3 + 0.01 * i)
        if k == 4:
            self.grid(sc, self.sorted, g * smoothstep(*self.SORT, u), lift=0.34)
            self.blocks(sc, g * smoothstep(self.SORT[1], self.SORT[1] + 0.06, u))
        self.rect(sc, self.P(self.GX, self.GY), self.side + 5, self.side + 5, DATA_INK, g,
                  border=DATA_INK, border_w=1.2, radius=2, lift=0.25, fill_op=0.0)
        self.diagonal(sc, g * (on(u, 0.45, 1.0) if k == 2 else
                               1.0 - 0.5 * smoothstep(*self.SORT, u) if k == 4 else 1.0))
        # the loss meter
        a = g * (on(u, 0.62, 1.0) if k == 2 else 1.0)
        if a > 0.01:
            s1, s2 = self.levels(k, u)
            l0, l1, l2 = (self.snap[s]["loss"] for s in self.steps)
            loss = l0 + (l1 - l0) * s1 + (l2 - l1) * s2
            self.meter(sc, self.MX, self.MY, loss / self.LOSS_TOP, a, h=self.MH,
                       label=self.pick(k, u, "loss_text"))
            self.text(sc, self.MX, self.MY - self.frac(0, self.MH / 2 + 14)[1], "loss", a,
                      SIZE, ink=LOSS_INK)

    def blocks(self, sc, op):
        """Phase 5: a square round each digit's rows and columns, its digit beside it."""
        if op <= 0.01:
            return
        counts = self.snap[self.steps[-1]]["counts"]
        start = 0
        for d, n in enumerate(counts):
            if n:
                (x0, y0), (x1, y1) = self.cell(start, start), self.cell(start + n - 1,
                                                                        start + n - 1)
                w, h = self.span(x1 - x0, y1 - y0)
                self.rect(sc, self.P((x0 + x1) / 2, (y0 + y1) / 2), w + self.CELL + 2,
                          h + self.CELL + 2, INK, op, border=INK, border_w=1.3, radius=1,
                          lift=1.2, fill_op=0.0)
                self.text(sc, self.gx0 - 0.007, (y0 + y1) / 2, str(d), op, SIZE, "rm",
                          ink=DATA_INK)
            start += n

    # ------------------------------------------------------------------ 1: a pair
    def phase_1(self, sc, u, op):
        m = self.meta
        x = self.CX + 0.065
        self.text(sc, x, (self.CY[0] + self.CY[1]) / 2 - 0.02, "a pair: the same digit",
                  on(u, 0.12, op), HEAD, "lm", strong=True)
        self.text(sc, x, (self.CY[0] + self.CY[1]) / 2 + 0.03,
                  "two drawings of a %d" % m["pair_label"], on(u, 0.3, op), SIZE, "lm",
                  ink=DATA_INK)
        self.text(sc, x, (self.CY[0] + self.CY[1]) / 2 + 0.07, "labels only pick the pair",
                  on(u, 0.55, op), SIZE, "lm", ink=QUIET)

    # ------------------------------------------------------------------ 2: the encoder
    def phase_2(self, sc, u, op):
        m = self.meta
        bx, by, bw, bh = self.BOX
        half = self.frac(bw / 2)[0]
        cw = self.frac(28 * self.CT / 2 + 7)[0]
        box = on(u, 0.08, op)
        self.encoder(sc, bx, by, bw, bh, box)
        strips = (self.a, self.b)
        names = ("a[0]", "b[0]")
        sw = self.frac(32 * self.ST)[0]
        for i, y in enumerate(self.CY):
            t = 0.3 + 0.12 * i
            self.arrow(sc, (self.CX + cw + 0.006, y), (bx - half - 0.006, y), AXIS,
                       on(u, 0.12 + 0.1 * i, op), width=1.6)
            self.arrow(sc, (bx + half + 0.006, y), (self.SX - sw / 2 - 0.008, y), AXIS,
                       on(u, t, op), width=1.6)
            self.vector(sc, strips[i], self.SX, y, self.ST, on(u, t + 0.02, op), -self.v_hi,
                        self.v_hi)
            self.text(sc, self.SX - sw / 2, y - 0.04, "%s: 32 numbers" % names[i],
                      on(u, t + 0.04, op), SIZE, "lm", ink=DATA_INK)
        self.text(sc, self.SX, by, "a[0] · b[0] = %s" % m["dot_text"], on(u, 0.6, op), HEAD,
                  strong=True)

    # ------------------------------------------------------------------ 3: the grid
    def phase_3(self, sc, u, op):
        m = self.meta
        a = on(u, 0.32, op)
        x, y = self.cell(0, 0)
        self.rect(sc, self.P(x, y), self.CELL + 5, self.CELL + 5, INK, a, border=INK,
                  border_w=2.0, radius=1, lift=1.25, fill_op=0.0)
        self.text(sc, self.gx0, self.gy0 - 0.025, "a[0] · b[0] = %s" % m["dot_text"], a, SIZE,
                  "lm")
        self.text(sc, self.gx0, self.LOW[0], "the diagonal: the right pairs", on(u, 0.45, op),
                  SIZE, "lm", ink=LOSS_INK)
        self.text(sc, self.gx0, self.LOW[1], "64 digits × 64 partners", on(u, 0.12, op), SIZE,
                  "lm", ink=QUIET)

    # ------------------------------------------------------------------ 4: training
    def phase_4(self, sc, u, op):
        a = on(u, 0.04, op)
        self.text(sc, self.gx0, self.gy0 - 0.028, "step %d" % self.step_at(u), a, HEAD, "lm",
                  strong=True)
        self.text(sc, self.gx0, self.LOW[0], "right pairs: %s" % self.pick(3, u, "pos_text"),
                  on(u, 0.1, op), SIZE, "lm", ink=LOSS_INK)
        self.text(sc, self.gx0, self.LOW[1], "the other 63: %s" % self.pick(3, u, "neg_text"),
                  on(u, 0.14, op), SIZE, "lm", ink=DATA_INK)

    def step_at(self, u):
        s1, s2 = self.levels(3, u)
        return self.steps[2] if s2 >= 0.5 else self.steps[1] if s1 >= 0.5 else self.steps[0]

    # ------------------------------------------------------------------ 5: by digit
    def phase_5(self, sc, u, op):
        s = self.snap[self.steps[-1]]
        self.text(sc, self.gx0, self.gy0 - 0.028, "step %d" % self.steps[-1],
                  op * (1.0 - smoothstep(self.SORT[0] - 0.03, self.SORT[0], u)), HEAD, "lm",
                  strong=True)
        self.text(sc, self.gx0, self.gy0 - 0.028, "step %d, sorted by digit" % self.steps[-1],
                  on(u, self.SORT[0], op), SIZE, "lm")
        self.text(sc, self.gx0, self.LOW[0], "right pairs %s · same digit %s"
                  % (s["pos_text"], s["same_text"]), on(u, 0.3, op), SIZE, "lm", strong=True)
        self.text(sc, self.gx0, self.LOW[1], "other digits: %s" % s["other_text"],
                  on(u, 0.4, op), SIZE, "lm", ink=DATA_INK)


# ====================================================================== 235
class StoreSearch(Vectors):
    """The store; a query; 60,000 dot products; the top 5; the quality over every test
    digit."""

    TITLE = "Store, then search"
    PHASES = (("the store", 13.0), ("a query", 11.0), ("60,000 scores", 12.0),
              ("the top 5", 13.0), ("quality", 11.0))
    LAST = "Training vectors for search"
    NEXT = "An index: look in fewer places"
    STATION = ("forward", "forward", None, None, None)
    LX, CARD_Y, CT = 0.118, 0.455, 2.0          # the left column: a digit card
    ENC_Y, ENC_W, ENC_H = 0.64, 84.0, 28.0      # ... the encoder
    Q_Y, QT = 0.73, 3.5                         # ... the query's 32 numbers
    TX0 = 0.215                                 # the table's left edge
    ID_X, LAB_X, VEC_X, SC_X = 0.262, 0.29, 0.36, 0.472     # its columns
    VT = 3.0                                    # px per number of a stored vector
    HEAD_Y, ROW0, ROW = 0.4, 0.44, 0.034        # its header and rows
    TOP0, TOP_ROW, TOP_T = 0.464, 0.088, 0.95   # phase 4: the top 5's rows, picture tile
    APPEAR = tuple(0.15 + 0.05 * i for i in range(6))      # phase 1: each row
    SWEEP = (0.1, 0.42)                         # phase 3: the band runs down the rows
    QUAL_ROWS = (("trained", "32, trained"), ("pixels", "784 pixels"),
                 ("untrained", "32, untrained"))

    def bar(self, sc, x0, y, w, h, ink, op, lift=0.3):
        """A flat bar from layout ``x0``, ``w`` layout wide, ``h`` px tall."""
        if op > 0.01 and w > 0:
            self.rect(sc, self.P(x0 + w / 2, y), self.span(w)[0], h, ink, op, radius=1,
                      lift=lift)

    def setup(self):
        m = self.meta = self.study.metadata
        self.rows = list(m["rows"])
        self.vecs = np.asarray(m["row_vectors"], float)
        self.q = np.asarray(m["q"], float)
        self.lo, self.hi = float(m["vec_lo"]), float(m["vec_hi"])
        n = len(self.rows)
        # rows 0..4, a gap for three dots, then the last row
        self.row_y = [self.ROW0 + self.ROW * i for i in range(n - 1)] + \
            [self.ROW0 + self.ROW * n]
        self.dots_y = self.ROW0 + self.ROW * (n - 1) - 0.004
        search = "vdb-search"
        self.CODE = (
            Code("vdb-store", run=((0.0, 0), (0.1, 1), (0.15, 2), (0.5, 3), (0.62, 4))),
            Code(search, run=((0.0, 1), (0.42, 2))),
            Code(search, run=((0.0, 3),)),
            Code(search, run=((0.0, 4), (0.3, 5), (0.5, 6)), out=((0.55, m["printed_wrapped"]),)),
            Code(search, run=((0.0, 3),)))
        self.EVIDENCE = (measured("the trained encoder, 60,000 digits"),
                         measured("test digit 0, the trained encoder"),
                         measured("test digit 0's 60,000 scores"),
                         measured("the 5 highest of 60,000 scores"),
                         measured("all 10,000 test digits as queries"))

    def heading_for(self, k, u):
        return ("the store: a vector per digit",
                "a query: a new digit, the same encoder",
                "a dot product with every row",
                "the 5 highest scores",
                "32 numbers do as well as 784")[k]

    def subtitle_for(self, k):
        m = self.meta
        q = m["quality_texts"]
        return [
            "The trained encoder turns the %s training digits into a block %s; each row "
            "keeps its id and label beside it." % (m["stored_text"], m["store_shape_text"]),
            "A test digit, a %d, goes through the same encoder: the query q is 32 numbers "
            "of length 1." % m["query_label"],
            "store @ q[0] is %s dot products, one score per stored digit: the %ds score %s on "
            "average, the rest %s." % (m["stored_text"], m["query_label"],
                                       "%.2f" % m["same_label_mean"],
                                       "%.2f" % m["other_label_mean"]),
            "topk(5) keeps the 5 highest: all %ds, each %s; the ids give back their pictures."
            % (m["query_label"], m["top_score_texts"][0]),
            "Over all %s test digits, %s of the 10 nearest share the query's label; raw "
            "pixels %s, untrained %s." % (m["tests_text"], q["trained"]["top10"],
                                          q["pixels"]["top10"], q["untrained"]["top10"]),
        ][k]

    # ------------------------------------------------------------------ the left column
    def steady(self, sc, k, u, op):
        m = self.meta
        ch = self.card_h(self.CT)
        # the encoder, all through
        self.encoder(sc, self.LX, self.ENC_Y, self.ENC_W, self.ENC_H, op)
        self.arrow(sc, (self.LX, self.CARD_Y + ch / 2 + 0.05),
                   (self.LX, self.ENC_Y - self.frac(0, self.ENC_H / 2 + 4)[1]), AXIS, op,
                   width=1.6)
        # the test digit and its query, from phase 2 on
        if k >= 1:
            a = op * (on(u, 0.05, 1.0) if k == 1 else 1.0)
            self.picture(sc, m["query_image"], self.LX, self.CARD_Y, self.CT, a)
            self.text(sc, self.LX, self.CARD_Y + ch / 2 + 0.03, "XT[0]: a %d" % m["query_label"],
                      a, SIZE, ink=DATA_INK)
            b = op * (on(u, 0.35, 1.0) if k == 1 else 1.0)
            self.arrow(sc, (self.LX, self.ENC_Y + self.frac(0, self.ENC_H / 2 + 4)[1]),
                       (self.LX, self.Q_Y - 0.022), AXIS, b, width=1.6)
            c = op * (on(u, 0.42, 1.0) if k == 1 else 1.0)
            self.vector(sc, self.q, self.LX, self.Q_Y, self.QT, c, self.lo, self.hi)
            self.text(sc, self.LX, self.Q_Y + 0.04, "q: 32 numbers", c, SIZE, ink=DATA_INK)
        # the store's table, phases 1 to 3
        t = op * self.kept(0, 2)
        if t > 0.01:
            self.table(sc, k, u, t)

    # ------------------------------------------------------------------ the store
    def table(self, sc, k, u, op):
        m = self.meta
        rows = op
        ids = op * (on(u, 0.5, 1.0) if k == 0 else 1.0)
        labels = op * (on(u, 0.62, 1.0) if k == 0 else 1.0)
        head = op * (on(u, 0.15, 1.0) if k == 0 else 1.0)
        self.text(sc, self.VEC_X, self.HEAD_Y, "vector", head, SIZE, ink=QUIET)
        self.text(sc, self.ID_X, self.HEAD_Y, "id", ids, SIZE, "rm", ink=QUIET)
        self.text(sc, self.LAB_X, self.HEAD_Y, "label", labels, SIZE, ink=QUIET)
        for i, (row, y) in enumerate(zip(self.rows, self.row_y, strict=True)):
            a = rows * (on(u, self.APPEAR[i], 1.0) if k == 0 else 1.0)
            self.vector(sc, self.vecs[i], self.VEC_X, y, self.VT, a, self.lo, self.hi, h=11.0)
            self.text(sc, self.ID_X, y, str(row), min(a, ids), SIZE, "rm", ink=INK)
            self.text(sc, self.LAB_X, y, str(m["row_labels"][i]), min(a, labels), SIZE,
                      ink=DATA_INK)
        d = rows * (on(u, self.APPEAR[-1], 1.0) if k == 0 else 1.0)
        for x, a in ((self.ID_X - 0.01, min(d, ids)), (self.LAB_X, min(d, labels)),
                     (self.VEC_X, d)):
            self.vdots(sc, x, self.dots_y, a)
        cap = op * (on(u, 0.3, 1.0) if k == 0 else 1.0)
        self.text(sc, self.TX0, 0.69, "store: %s" % m["store_shape_text"], cap, SIZE, "lm",
                  ink=DATA_INK)
        self.text(sc, self.TX0, 0.727, "ids and labels beside it", labels, SIZE, "lm",
                  ink=QUIET)
        if k == 0:
            self.arrow(sc, (self.LX + self.frac(self.ENC_W / 2 + 4)[0], self.ENC_Y),
                       (self.TX0 - 0.006, self.ENC_Y), AXIS, on(u, 0.1, op), width=1.6)
        if k == 2:
            self.scores(sc, u, op)

    def scores(self, sc, u, op):
        """Phase 3: a band runs down the rows, each row's score appearing as it passes; an
        arrow from the query follows it."""
        m = self.meta
        self.text(sc, self.SC_X, self.HEAD_Y, "score", on(u, 0.05, op), SIZE, "rm", ink=QUIET)
        n = len(self.rows)
        pos = eased(u, *self.SWEEP) * (n - 1)                  # 0 .. n-1, row by row
        band = op * smoothstep(self.SWEEP[0] - 0.04, self.SWEEP[0], u) * \
            (1.0 - smoothstep(self.SWEEP[1] + 0.03, self.SWEEP[1] + 0.08, u))
        y = float(np.interp(pos, np.arange(n), self.row_y))
        if band > 0.01:
            w = self.span(self.SC_X - self.TX0 + 0.012)[0]
            self.rect(sc, self.P((self.TX0 + self.SC_X) / 2, y), w, 18.0, soft(DATA, 0.7), band,
                      radius=4, lift=0.12)
            self.arrow(sc, (self.LX + self.frac(32 * self.QT / 2 + 6)[0], self.Q_Y),
                       (self.TX0 - 0.008, y), DATA_INK, band, width=1.4)
        for i, ry in enumerate(self.row_y):
            a = op * smoothstep(i - 0.3, i + 0.05, pos) if u < self.SWEEP[1] + 0.01 else op
            self.text(sc, self.SC_X, ry, m["row_score_texts"][i], a, SIZE, "rm", ink=INK)
        self.vdots(sc, self.SC_X - 0.02, self.dots_y, op * smoothstep(n - 1.6, n - 1.2, pos))
        self.text(sc, self.TX0, 0.79, "%s dot products" % m["stored_text"], on(u, 0.5, op),
                  HEAD, "lm", strong=True)
        self.text(sc, self.TX0, 0.835, "%ds: %.2f · others: %.2f"
                  % (m["query_label"], m["same_label_mean"], m["other_label_mean"]),
                  on(u, 0.62, op), SIZE, "lm", ink=DATA_INK)

    # ------------------------------------------------------------------ 1-3: in steady
    def phase_1(self, sc, u, op):
        """The training digits going in: a stack of cards over the encoder."""
        m = self.meta
        if op <= 0.01:
            return
        a = op * (1.0 - smoothstep(0.9, 1.0, u))
        w = 28 * self.CT + 14
        h = 28 * self.CT + 20
        self.stack(sc, self.LX - self.frac(6)[0], self.CARD_Y + self.frac(0, 6)[1], w, h, 4, a)
        self.picture(sc, m["first_image"], self.LX, self.CARD_Y, self.CT, a, lift=0.25)
        ch = self.card_h(self.CT)
        self.text(sc, self.LX, self.CARD_Y + ch / 2 + 0.03, "X: %s digits" % m["stored_text"],
                  a, SIZE, ink=DATA_INK)

    def phase_2(self, sc, u, op):
        pass

    def phase_3(self, sc, u, op):
        pass

    # ------------------------------------------------------------------ 4: the top 5
    def phase_4(self, sc, u, op):
        m = self.meta
        head = on(u, 0.06, op)
        self.text(sc, self.ID_X, self.HEAD_Y, "id", head, SIZE, "rm", ink=QUIET)
        self.text(sc, self.LAB_X, self.HEAD_Y, "label", head, SIZE, ink=QUIET)
        self.text(sc, self.VEC_X, self.HEAD_Y, "X[id]", on(u, 0.3, op), SIZE, ink=QUIET)
        self.text(sc, self.SC_X, self.HEAD_Y, "score", head, SIZE, "rm", ink=QUIET)
        for i in range(len(m["top"])):
            y = self.TOP0 + self.TOP_ROW * i
            a = on(u, 0.08 + 0.035 * i, op)
            self.text(sc, self.ID_X, y, str(m["top"][i]), a, SIZE, "rm", ink=INK)
            self.text(sc, self.LAB_X, y, str(m["top_labels"][i]), a, SIZE, ink=DATA_INK)
            self.text(sc, self.SC_X, y, m["top_score_texts"][i], a, SIZE, "rm", ink=INK)
            self.picture(sc, m["top_images"][i], self.VEC_X, y, self.TOP_T,
                         on(u, 0.3 + 0.03 * i, op))
        self.text(sc, 0.06, 0.835, "5 of 5: a %d" % m["query_label"], on(u, 0.62, op), HEAD,
                  "lm", strong=True)

    # ------------------------------------------------------------------ 5: quality
    def phase_5(self, sc, u, op):
        m = self.meta
        q = m["quality_texts"]
        a = on(u, 0.05, op)
        self.text(sc, self.TX0, self.HEAD_Y, "%s test digits as queries" % m["tests_text"], a,
                  SIZE, "lm")
        self.text(sc, self.TX0, self.HEAD_Y + 0.036, "same label as the query:", a, SIZE, "lm",
                  ink=QUIET)
        y0 = 0.5
        self.text(sc, 0.395, y0, "top 1", a, SIZE, "rm", ink=QUIET)
        self.text(sc, self.SC_X, y0, "top 10", a, SIZE, "rm", ink=QUIET)
        for i, (key, name) in enumerate(self.QUAL_ROWS):
            y = y0 + 0.05 * (i + 1)
            b = on(u, 0.15 + 0.1 * i, op)
            strong = key == "trained"
            self.text(sc, self.TX0, y, name, b, SIZE, "lm", ink=INK if strong else DATA_INK)
            self.text(sc, 0.395, y, q[key]["top1"], b, SIZE, "rm", ink=INK if strong else None)
            self.text(sc, self.SC_X, y, q[key]["top10"], b, SIZE, "rm",
                      ink=INK if strong else None)
            full = self.SC_X - self.TX0
            self.bar(sc, self.TX0, y + 0.024, full, 4.0, soft(DATA, 0.6), b)
            self.bar(sc, self.TX0, y + 0.024, full * m["quality"][key]["top10"], 4.0, DATA, b,
                     lift=0.32)
        self.text(sc, self.TX0, 0.75, "32 numbers instead of 784", on(u, 0.5, op), HEAD, "lm",
                  strong=True)


ANIMATIONS = {"vector-db-training": Training, "vector-db-store-and-search": StoreSearch}

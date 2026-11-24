"""Animations 232-233, the last two of the padding part of "Blocks and code" (see
:mod:`voxanim.blocks`), drawn from :mod:`voxanim.study_blocks_q`.

* :class:`Masks` (``padding-masks``): the first eight lines of Tiny Shakespeare padded to
  ``[8, 50]`` stay at the top all through, the pads striped grey as in the rest of the
  padding series (:func:`~voxanim.mech_blocks_p.hatched`). ``padding_idx=0`` makes the 46
  pads of "All:" into vectors of zeros; the plain mean divides by 50 and comes out 4 / 50
  = 0.08 of the true mean; every line's bar, plain then masked; the attention weights of
  each line's first character with ``key_padding_mask``, exactly 0 on every pad (the other
  weights are random, and say so); ``ignore_index=0``: 163 targets counted, 237 skipped.
* :class:`SortByLength` (``padding-sort-by-length``): real batches of 64 lines as rows of
  bars (blue: the line, grey stripes: its padding to the batch's longest), in file order,
  shuffled and sorted by length, with the share of padding over all 513 batches kept in a small
  chart at the bottom (38.0%, 38.7%, 0.16%); then six sorted batches shuffled as whole
  batches, each keeping its padding.

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
    Blocks,
    Code,
    exact,
    shades,
)
from .diagram import draw_image
from .easing import ease_in_out, smoothstep
from .mech import PAPER, TRACK, dark
from .mech_blocks_p import HATCH, hatched

DATA_INK = dark(DATA, 0.3)
LOSS_INK = dark(LOSS, 0.25)
PAD_INK = dark(HATCH, 0.2)        # words about the padding, in the stripes' grey


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on(u, a, op, width=0.05):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


class Padded(Blocks):
    """What the two share."""

    def at(self, k, u):
        """1 once phase ``k`` has reached ``u`` (fading in over a short while)."""
        a, b = self.spans[k]
        t0 = a + (b - a) * u
        return smoothstep(t0, t0 + 0.5, self._t)

    def bar(self, sc, x0, y, w, h, ink, op, lift=0.3):
        """A flat bar from layout ``x0``, ``w`` layout wide, ``h`` px tall, centred on ``y``."""
        if op <= 0.01 or w <= 0:
            return
        self.rect(sc, self.P(x0 + w / 2, y), max(self.span(w)[0], 1.2), h, ink, op, radius=0,
                  lift=lift)

    def striped(self, sc, rgb, pad, fx, fy, w, h, op, sy, sx, period=5, lift=0.3,
                border=None):
        """Tiles of colours ``rgb`` (rows, cols, 3) as a ``w x h`` px image centred at
        ``(fx, fy)``, the pads (``pad``) striped grey as in the rest of the padding series
        (:func:`~voxanim.mech_blocks_p.hatched`, ``sy x sx`` texels a tile)."""
        if op <= 0.01:
            return
        tex = hatched(rgb, pad, int(sy), int(sx), period, 2, border=border)
        draw_image(self, sc, self.P(fx, fy), tex, w, h, op, grid=0.0, lift=lift)

    def pad_bar(self, sc, x0, y, w, h, op, lift=0.3):
        """A bar of padding: paper with grey diagonal stripes, from layout ``x0``, ``w``
        layout wide, ``h`` px tall."""
        if op <= 0.01 or w <= 0:
            return
        px = max(self.span(w)[0], 1.2)
        n = max(int(round(px)), 1)
        self.striped(sc, np.full((1, n, 3), PAPER, np.float32), np.ones((1, n), bool),
                     x0 + w / 2, y, px, h, op, round(h), 1, lift=lift)


def padded_rgb(values, pad, ink=DATA, lo=0.0, hi=1.0):
    """Tile colours: ``values`` shaded in ``ink``, the pads paper (striped later)."""
    rgb = shades(values, ink, lo, hi)
    rgb[np.asarray(pad, bool)] = PAPER
    return rgb


# ====================================================================== 232
class Masks(Padded):
    """The padded block; the pads' zero vectors; plain against masked means; the attention
    weights on the pads; the targets the loss counts."""

    TITLE = "Masks: ignore the padding"
    PHASES = (("zero vectors", 10.0), ("plain mean", 12.0), ("masked mean", 13.0),
              ("attention", 14.0), ("the loss", 10.0))
    LAST = "Padding lines of text"
    NEXT = "Sort by length, pad less"
    STATION = ("forward", "forward", "forward", "forward", "loss")
    X0 = 0.075                      # the left edge of the block and of everything under it
    TILE, ROW = 6.5, 8.0            # px per character, per line
    BLOCK_TOP = 0.375               # the [8, 50] block's top
    LOW = 0.565                     # the captions of what is under it
    VEC_H = 3.5                     # vecs[2]: 16 rows of 3.5 px
    BAR_X1 = 0.375                  # phase 3: a bar's full length (the true mean) ends here
    BAR_Y, BAR_STEP = 0.602, 0.036

    def setup(self):
        m = self.meta = self.study.metadata
        self.n, self.L = m["shape"]
        self.mask = np.asarray(m["mask"], bool)
        self.vecs = np.asarray(m["follow_vecs"], float).T           # [16, 50]
        real = self.vecs[:, :m["follow_length"]]
        self.v_lo, self.v_hi = float(real.min()), float(real.max())
        w = np.asarray(m["first_weights"], float)
        self.w_shade = w / w.max(1, keepdims=True)                  # each row its own scale
        self.width = self.frac(self.L * self.TILE)[0]
        self.cx = self.X0 + self.width / 2
        self.block_h = self.n * self.ROW
        self.block_y = self.BLOCK_TOP + self.frac(0, self.block_h / 2)[1]
        self.under = self.LOW + 0.02            # the top of a block drawn under the caption
        self.vec_h = self.vecs.shape[0] * self.VEC_H
        self.vec_y = self.under + self.frac(0, self.vec_h / 2)[1]
        self.w_y = self.under + self.frac(0, self.block_h / 2)[1]
        self.CODE = (
            Code("pad-masked-mean", run=((0.0, 0), (0.35, 2))),
            Code("pad-masked-mean", run=((0.0, 2), (0.15, 3))),
            Code("pad-masked-mean", run=((0.0, 3), (0.42, 4), (0.5, 5), (0.58, 6))),
            Code("pad-attention", run=((0.0, 0), (0.2, 2), (0.3, 3))),
            Code("pad-loss", run=((0.0, 0), (0.25, 1))))
        self.EVIDENCE = (exact("%d of %d real; a pad's vector is 0" % (m["real"], m["cells"])),
                         exact("%s, whatever the weights" % m["follow_ratio_text"]),
                         exact("length ÷ %d, whatever the weights" % self.L),
                         exact("the zeros; the other weights random"),
                         exact("%d targets of %d, counted" % (m["real"], m["cells"])))

    def heading_for(self, k, u):
        return ("padding_idx=0: each pad's vector is 0",
                "the plain mean divides by %d" % self.L,
                "the masked mean divides by the length",
                "key_padding_mask: no weight on a pad",
                "ignore_index=0: pad targets don't count")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "nn.Embedding with padding_idx=0 turns every pad into 16 zeros: here the %d pads "
            "of the line \"%s\"." % (m["follow_pads"], m["follow_text"]),
            "vecs.mean(1) adds %d real vectors and %d zeros, then divides by %d: exactly %s "
            "of the true mean." % (m["follow_length"], m["follow_pads"], self.L,
                                   m["follow_ratio_text"]),
            "Each plain mean is the true mean × length ÷ %d. The masked mean divides by the "
            "length: it is right." % self.L,
            "With key_padding_mask, each line's first character gives every pad a weight of "
            "exactly 0. The other weights here are random.",
            "With ignore_index=0, a loss over targets padded like this counts the %d real "
            "ones and skips the %d pads." % (m["loss_counted"], m["loss_skipped"]),
        ][k]

    # ------------------------------------------------------------------ geometry
    def ring(self, sc, fy, row, op):
        """An outline round row ``row`` of an ``[8, 50]`` block centred at ``fy``."""
        if op <= 0.01:
            return
        y = fy + self.frac(0, (row + 0.5) * self.ROW - self.block_h / 2)[1]
        self.rect(sc, self.P(self.cx, y), self.L * self.TILE + 5, self.ROW + 4, INK, op,
                  border=INK, border_w=2.2, radius=2, lift=1.0, fill_op=0.0)

    def block(self, sc, values, fy, op, ink=DATA, lift=0.3):
        """``[8, 50]`` tiles centred at ``fy``, the pads striped."""
        self.striped(sc, padded_rgb(values, ~self.mask, ink), ~self.mask, self.cx, fy,
                     self.L * self.TILE, self.block_h, op, self.ROW, round(self.TILE),
                     lift=lift, border=0.22)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        real = np.where(self.mask, 0.8, 0.0)
        # the block: real characters blue, pads striped
        self.block(sc, real, self.block_y, op)
        self.rect(sc, self.P(self.cx, self.block_y), self.L * self.TILE + 5, self.block_h + 5,
                  DATA_INK, op, border=DATA_INK, border_w=1.3, radius=2, lift=0.25, fill_op=0.0)
        # in the last phase the real cells turn violet: the targets the loss counts
        if k == 4:
            self.block(sc, real, self.block_y, op * smoothstep(0.1, 0.2, u), ink=LOSS,
                       lift=0.32)
        self.text(sc, self.X0, self.BLOCK_TOP + self.frac(0, self.block_h)[1] + 0.028,
                  "%s: %d real, %d pads" % (m["shape_text"], m["real"], m["pads"]), op, SIZE,
                  "lm", ink=DATA_INK)
        # the line followed through the means, ringed (phases 1 and 2)
        r = op * self.kept(0, 1) * (self.at(0, 0.1) if k == 0 else 1.0)
        self.ring(sc, self.block_y, m["follow"], r)
        # vecs[2] (phases 1 and 2)
        v = op * self.kept(0, 1)
        if v > 0.01:
            self.follow_vectors(sc, v * (self.at(0, 0.1) if k == 0 else 1.0), k)

    def follow_vectors(self, sc, op, k):
        m = self.meta
        n = m["follow_length"]
        zero = np.zeros_like(self.vecs, bool)
        zero[:, n:] = True
        self.striped(sc, padded_rgb(self.vecs, zero, lo=self.v_lo, hi=self.v_hi), zero, self.cx,
                     self.vec_y, self.L * self.TILE, self.vec_h, op, round(self.VEC_H),
                     round(self.TILE))
        self.rect(sc, self.P(self.cx, self.vec_y), self.L * self.TILE + 5, self.vec_h + 5,
                  DATA_INK, op, border=DATA_INK, border_w=1.3, radius=2, lift=0.25, fill_op=0.0)
        bottom = self.under + self.frac(0, self.vec_h)[1]
        self.text(sc, self.X0, self.LOW, 'vecs[%d], "%s": %d vectors of %d'
                  % (m["follow"], m["follow_text"], self.L, m["dim"]), op, SIZE, "lm",
                  ink=DATA_INK)
        a = op * (self.at(0, 0.3) if k == 0 else 1.0)
        self.text(sc, self.X0, bottom + 0.028, "%d real, random" % n, a, SIZE, "lm",
                  ink=DATA_INK)
        b = op * (self.at(0, 0.4) if k == 0 else 1.0)
        self.text(sc, self.X0 + self.width, bottom + 0.028, "%d pads: all 0" % m["follow_pads"],
                  b, SIZE, "rm", ink=PAD_INK)

    # ------------------------------------------------------------------ 1: zero vectors
    def phase_1(self, sc, u, op):
        """The block, the ring and vecs[2], in ``steady``."""

    # ------------------------------------------------------------------ 2: the plain mean
    def phase_2(self, sc, u, op):
        m = self.meta
        n = m["follow_length"]
        y = 0.765
        self.text(sc, self.X0, y, "plain mean: the sum ÷ %d" % self.L, on(u, 0.15, op), SIZE,
                  "lm")
        self.text(sc, self.X0, y + 0.04, "true mean: the sum ÷ %d" % n, on(u, 0.3, op), SIZE,
                  "lm")
        self.text(sc, self.X0, y + 0.09, "%s of the true mean" % m["follow_ratio_text"],
                  on(u, 0.48, op), HEAD, "lm", strong=True)

    # ------------------------------------------------------------------ 3: every line
    def phase_3(self, sc, u, op):
        m = self.meta
        a = on(u, 0.06, op)
        lx = self.X0 + 0.03                   # the bars start here; the lengths end here
        full = self.BAR_X1 - lx
        masked = eased(u, 0.5, 0.62)
        head = self.LOW
        self.text(sc, lx - 0.008, head, "length", a, SIZE, "rm", ink=QUIET)
        self.text(sc, lx, head, "plain mean ÷ true mean", a * (1 - smoothstep(0.47, 0.5, u)),
                  SIZE, "lm")
        self.text(sc, lx, head, "masked mean ÷ true mean", on(u, 0.5, op), SIZE, "lm",
                  strong=True)
        grow = eased(u, 0.1, 0.3)
        for r, (n, ratio) in enumerate(zip(m["lengths"], m["ratio"])):
            y = self.BAR_Y + self.BAR_STEP * r
            self.text(sc, lx - 0.008, y, str(n), a, SIZE, "rm", ink=DATA_INK)
            self.bar(sc, lx, y, full, 13.0, TRACK, a * 0.7, lift=0.25)
            w = full * (ratio + (1.0 - ratio) * masked) * grow
            self.bar(sc, lx, y, w, 13.0, DATA, a, lift=0.3)
            self.text(sc, self.BAR_X1 + 0.01, y, m["ratio_texts"][r],
                      on(u, 0.3, op) * (1 - smoothstep(0.47, 0.5, u)), SIZE, "lm", ink=DATA_INK)
            self.text(sc, self.BAR_X1 + 0.01, y, "1.00", on(u, 0.62, op), SIZE, "lm",
                      ink=DATA_INK)

    # ------------------------------------------------------------------ 4: attention
    def phase_4(self, sc, u, op):
        m = self.meta
        a = on(u, 0.08, op)
        self.block(sc, self.w_shade, self.w_y, a)
        self.text(sc, self.X0, self.LOW, "w[:, 0]: each first character's weights", a, SIZE,
                  "lm", ink=DATA_INK)
        r = on(u, 0.3, op)
        self.ring(sc, self.w_y, m["shown_row"], r)
        self.ring(sc, self.block_y, m["shown_row"], r)
        y = self.under + self.frac(0, self.block_h)[1] + 0.04
        self.text(sc, self.X0, y, "w[%d, 0]: %d weights, adding to %s"
                  % (m["shown_row"], m["shown_length"], m["shown_sum_text"]), on(u, 0.35, op),
                  SIZE, "lm")
        self.text(sc, self.X0, y + 0.048, "its %d pads: exactly 0" % m["shown_pads"],
                  on(u, 0.45, op), HEAD, "lm", strong=True)
        self.text(sc, self.X0, y + 0.096, "the weights are random; the zeros are not",
                  on(u, 0.6, op), SIZE, "lm", ink=QUIET)

    # ------------------------------------------------------------------ 5: the loss
    def phase_5(self, sc, u, op):
        m = self.meta
        a = on(u, 0.15, op)
        y = self.LOW + 0.055
        counted = self.width * m["loss_counted"] / m["cells"]
        grow = eased(u, 0.15, 0.3)
        self.bar(sc, self.X0, y, counted * grow, 18.0, LOSS, a)
        self.pad_bar(sc, self.X0 + counted, y, (self.width - counted) * grow, 18.0, a)
        b = on(u, 0.28, op)
        self.text(sc, self.X0, self.LOW, "%d counted" % m["loss_counted"], b, SIZE, "lm",
                  ink=LOSS_INK)
        self.text(sc, self.X0 + self.width, self.LOW, "%d skipped" % m["loss_skipped"], b,
                  SIZE, "rm", ink=PAD_INK)
        self.text(sc, self.X0, y + 0.08, "the loss: ÷ %d, not ÷ %d"
                  % (m["loss_counted"], m["cells"]), on(u, 0.4, op), HEAD, "lm", strong=True)


# ====================================================================== 233
ORDERS = ("file", "shuffled", "sorted")
ORDER_NAMES = dict(file="file order", shuffled="shuffled", sorted="sorted")


class SortByLength(Padded):
    """Real batches as rows of bars in three orders, the padding over all of them, and
    whole batches shuffled."""

    TITLE = "Sort by length, pad less"
    PHASES = (("file order", 13.0), ("shuffled", 11.0), ("sorted", 14.0),
              ("whole batches", 13.0))
    LAST = "Masks: ignore the padding"
    NEXT = "Automatic differentiation"
    STATION = ("data",) * 4
    LEFTS = (0.065, 0.27)           # the two batches' left edges
    CHAR, ROW = 2.6, 2.2            # px per character, per line
    TOP = 0.405                     # the batches' top
    LABEL_Y, SHARE_Y = 0.385, 0.69
    TALLY_Y, TALLY_STEP = 0.74, 0.04
    TALLY_X = 0.17                  # the tally's bars start here
    TALLY_W = 0.2                   # ... and 40% is this long
    MINI_Y, MINI_CHAR, MINI_ROW = 0.49, 0.85, 1.2
    SLOTS = (0.065, 0.465)          # phase 4: the six slots span this

    def setup(self):
        m = self.meta = self.study.metadata
        self.CODE = (
            Code("pad-sort", pick=(1, 2, 3), run=((0.0, 0), (0.12, 1), (0.3, 2))),
            Code("pad-sort", pick=(1, 2, 3), run=((0.0, 0), (0.3, 2))),
            Code("pad-sort", run=((0.0, 0), (0.3, 1), (0.42, 3))),
            Code("pad-sort", run=((0.0, 1),)))
        self.EVIDENCE = (exact("counted over all %s lines" % m["lines_text"]),
                         exact("counted, one seeded shuffle"),
                         exact("counted over all %s lines" % m["lines_text"]),
                         exact("each batch keeps its own pads"))
        self.tally_at = ((0, 0.55), (1, 0.55), (2, 0.6))

    def heading_for(self, k, u):
        return ("64 lines at a time, in file order",
                "shuffled first: no better",
                "sorted by length: almost no padding",
                "then shuffle whole batches")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "In batches of 64, in file order, the %s lines need %s cells: %s of them are "
            "padding." % (m["lines_text"], m["cells_text"]["file"], m["share_text"]["file"]),
            "Shuffling the lines first does not help: %s of the cells are padding."
            % m["share_text"]["shuffled"],
            "lines.sort(key=len) puts lines of one length together: %s pads in %s cells, %s."
            % (m["padding_text"]["sorted"], m["cells_text"]["sorted"],
               m["share_text"]["sorted"]),
            "Sorting puts similar lines together, so in practice whole batches are shuffled; "
            "each keeps its padding.",
        ][k]

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        """The share of padding over all the batches, one bar per order, each filled in
        once its phase has counted it."""
        m = self.meta
        self.text(sc, self.LEFTS[0], self.TALLY_Y, "padding, all %d batches" % m["batches"],
                  op, SIZE, "lm")
        for i, (name, (kk, uu)) in enumerate(zip(ORDERS, self.tally_at)):
            y = self.TALLY_Y + self.TALLY_STEP * (i + 1)
            shown = op * self.at(kk, uu)
            self.text(sc, self.LEFTS[0], y, ORDER_NAMES[name], op * (0.45 + 0.55 * shown),
                      SIZE, "lm", ink=None if shown > 0.5 else QUIET)
            self.bar(sc, self.TALLY_X, y, self.TALLY_W, 12.0, TRACK, op * 0.5, lift=0.25)
            if shown > 0.01:
                w = self.TALLY_W * m["share"][name] / 0.4
                self.pad_bar(sc, self.TALLY_X, y, w, 12.0, shown, lift=0.3)
                self.text(sc, self.TALLY_X + max(w, 0.002) + 0.01, y, m["share_text"][name],
                          shown, SIZE, "lm", ink=INK)

    # ------------------------------------------------------------------ the batches
    def rows_of_bars(self, sc, rows, left, u, op, grow=(0.08, 0.25), pads=(0.3, 0.36),
                     char=None, row=None, top=None):
        """A batch as rows of bars from ``left``: blue, each line's length; then, fading in
        over them, the batch as one block with its padding up to the longest striped.
        Returns the block's ``(width, height)`` px."""
        char = self.CHAR if char is None else char
        row = self.ROW if row is None else row
        top = self.TOP if top is None else top
        lengths = rows["lengths"]
        longest = rows["longest"]
        g = eased(u, *grow)
        p = op * smoothstep(*pads, u)
        for i, n in enumerate(lengths):
            if (i + 0.5) / len(lengths) > g:
                break
            y = top + self.frac(0, (i + 0.5) * row)[1]
            self.bar(sc, left, y, self.frac(n * char)[0], row, DATA, op, lift=0.3)
        if p > 0.01:
            pad = np.arange(longest)[None, :] >= np.asarray(lengths)[:, None]
            w, h = longest * char, len(lengths) * row
            self.striped(sc, padded_rgb(np.full(pad.shape, 0.8), pad), pad,
                         left + self.frac(w / 2)[0], top + self.frac(0, h / 2)[1], w, h, p,
                         2, 3, lift=0.31)
        return longest * char, len(lengths) * row

    def two_batches(self, sc, u, op, which):
        m = self.meta
        for j, (rows, left) in enumerate(zip(m["rows"][which], self.LEFTS)):
            a = on(u, 0.06 + 0.1 * j, op)
            self.text(sc, left, self.LABEL_Y, "batch %d · longest %d"
                      % (rows["index"] + 1, rows["longest"]), a, SIZE, "lm")
            self.rows_of_bars(sc, rows, left, u, a, grow=(0.08 + 0.1 * j, 0.22 + 0.1 * j),
                              pads=(0.34 + 0.04 * j, 0.4 + 0.04 * j))
            self.text(sc, left, self.SHARE_Y, "%s pads" % m["row_share_text"][which][j],
                      on(u, 0.42 + 0.04 * j, op), SIZE, "lm", ink=PAD_INK)

    def phase_1(self, sc, u, op):
        self.two_batches(sc, u, op, "file")

    def phase_2(self, sc, u, op):
        self.two_batches(sc, u, op, "shuffled")

    def phase_3(self, sc, u, op):
        self.two_batches(sc, u, op, "sorted")

    # ------------------------------------------------------------------ 4: whole batches
    def slot_x(self, i):
        x0, x1 = self.SLOTS
        return x0 + (x1 - x0) * (i + 0.5) / len(self.meta["mini"])

    def mini(self, sc, rows, fx, fy, op, lift):
        """A sorted batch as a small block centred at ``(fx, fy)``: its lines blue, its
        padding striped."""
        lengths, longest = rows["lengths"], rows["longest"]
        pad = np.arange(longest)[None, :] >= np.asarray(lengths)[:, None]
        self.striped(sc, padded_rgb(np.full(pad.shape, 0.8), pad), pad, fx, fy,
                     longest * self.MINI_CHAR, len(lengths) * self.MINI_ROW, op, 1, 1, 4,
                     lift=lift)

    def phase_4(self, sc, u, op):
        m = self.meta
        order = m["mini_order"]
        move = eased(u, 0.38, 0.56)
        bump = float(np.sin(np.pi * move))
        still = 1.0 - smoothstep(0.33, 0.37, u) + smoothstep(0.57, 0.61, u)    # labels
        a = on(u, 0.06, op)
        self.text(sc, self.LEFTS[0], self.LABEL_Y, "%d sorted batches, short to long"
                  % len(m["mini"]), a * (1 - smoothstep(0.35, 0.38, u)), SIZE, "lm")
        self.text(sc, self.LEFTS[0], self.LABEL_Y, "the same batches, shuffled",
                  on(u, 0.38, op), SIZE, "lm")
        h = len(m["mini"][0]["lengths"]) * self.MINI_ROW
        below = self.MINI_Y + self.frac(0, h / 2)[1] + 0.028
        for i, rows in enumerate(m["mini"]):
            x0, x1 = self.slot_x(i), self.slot_x(order[i])
            x = x0 + (x1 - x0) * move
            y = self.MINI_Y - 0.02 * bump * np.sign(x1 - x0)    # right goes up, left down
            b = on(u, 0.08 + 0.03 * i, op)
            self.mini(sc, rows, x, y, b, lift=0.3 + 0.01 * i)
            self.text(sc, x, below, str(rows["longest"]), b * still, SIZE, ink=DATA_INK)
        self.text(sc, self.LEFTS[0], below + 0.04, "the longest line in each", on(u, 0.2, op),
                  SIZE, "lm", ink=QUIET)
        self.text(sc, self.LEFTS[0], below + 0.085, "padding: still %s"
                  % m["share_text"]["sorted"], on(u, 0.6, op), SIZE, "lm", strong=True)


ANIMATIONS = {"padding-masks": Masks, "padding-sort-by-length": SortByLength}

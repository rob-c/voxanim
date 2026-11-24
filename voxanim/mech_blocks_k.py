"""Animations 215-217, the first three of "Blocks and code: a CNN" (see :mod:`voxanim.blocks`):
the course's program with its model changed to a small CNN, drawn from
:mod:`voxanim.study_blocks_k`.

* :class:`TheModel` (``blocks-cnn-the-model``): the eight blocks stacked one by one as the
  model lines run; each one's output shape on the first batch; ``print(model)``'s real
  text in two halves, each beside the blocks it lists; and each block's count of learned
  numbers, 80 + 1,168 + 4,010 = 5,258, against the MLP's 52,650.
* :class:`Shapes` (``blocks-cnn-shapes``): where the 400 of ``nn.Linear(16 * 5 * 5, 10)``
  comes from: the side of the maps walked down as squares that shrink, to scale (a 3 x 3
  stamp slides across and fits 2 fewer places; a 2 x 2 pool halves, dropping the odd last
  row and column), 16 maps of 5 x 5 unrolled into 400 numbers; then ``nn.Linear(784,
  10)`` and torch's own error, in plain words: 400 numbers arrive, the layer expects 784.
* :class:`Forward` (``blocks-cnn-forward``): ``train_set[0]``, a 5, through the trained
  CNN: 8 maps, pooled, 16 maps, pooled, 400 numbers in a strip, 10 scores; the 5 wins.

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
from .diagram import draw_image, shrink_columns
from .easing import ease_in_out, smoothstep
from .mech import AXIS, PAPER, dark, soft

DATA_INK = dark(DATA, 0.3)
WEIGHT_INK = dark(WEIGHT, 0.35)
ZERO_INK = dark(ZERO, 0.45)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def at(k, u, when):
    """0 before ``when`` = ``(phase, u0)``, fading up to 1 just after, 1 from then on."""
    pk, pu = when
    if k != pk:
        return 1.0 if k > pk else 0.0
    return smoothstep(pu, pu + 0.04, u)


class Cnn(Blocks):
    """What the three share: a map of numbers as a small data card."""

    def map_card(self, sc, values, fx, fy, tile, op, lo=0.0, hi=None, pad=5.0, lift=0.2,
                 cover=None):
        """A square of numbers as ``tile`` px tiles on a data card centred at ``(fx, fy)``;
        ``cover``: a mask of tiles drawn faint (left out). Returns ``cell(row, col)`` of its
        tiles."""
        v = np.asarray(values, float)
        rows, cols = v.shape
        w, h = cols * tile, rows * tile
        cy = fy + self.frac(0, 3.0)[1]
        cell = self.cells(fx, cy, w, h, rows, cols)
        if op <= 0.01:
            return cell
        self.card(sc, fx, fy, w + 2 * pad, h + 2 * pad + 6, op, lift=lift)
        rgb = shades(v, DATA, lo, hi)
        if cover is not None:
            faint = PAPER + (rgb - PAPER) * 0.18
            rgb = np.where(np.asarray(cover, bool)[..., None], faint, rgb).astype(np.float32)
        draw_image(self, sc, self.P(fx, cy), rgb, w, h, op, grid=0.18 if tile >= 6 else 0.0,
                   lift=lift + 0.004)
        return cell

    def card_size(self, side, tile, pad=5.0):
        """A :meth:`map_card`'s width and height (px)."""
        return side * tile + 2 * pad, side * tile + 2 * pad + 6


# ====================================================================== 215
class TheModel(Cnn):
    """The blocks one by one; their shapes; print(model); their counts."""

    TITLE = "The CNN, block by block"
    PHASES = (("eight blocks", 15.0), ("the shapes", 13.0), ("print(model)", 16.0),
              ("its weights", 14.0))
    LAST = "Common mistakes"
    NEXT = "Where the 400 comes from"
    STATION = ("forward",) * 4
    BX, BW, BH = 0.087, 52.0, 20.0     # the blocks' column, as a stack
    ROW0, ROW_STEP = 0.395, 0.05
    NX, SX = 0.135, 0.47               # names, and the right edge of the shapes / counts
    PX, PBW = 0.068, 22.0              # beside print(model): the column, narrower
    P0, P_STEP, PTX = 0.375, 0.041, 0.095        # print(model)'s lines
    APPEAR = tuple(0.04 + 0.075 * k for k in range(8))     # phase 1: each block
    SEQ_AT = 0.68                                          # phase 1: nn.Sequential
    SHOW = tuple(0.06 + 0.07 * k for k in range(8))        # phase 2: each shape
    PAGES = ((0.06, 0.5), (0.54, 0.92))                    # phase 3: the two halves
    COUNT_AT = tuple(0.05 + 0.06 * k for k in range(8))    # phase 4: each count
    MLP_AT = 0.62

    def setup(self):
        m = self.meta = self.study.metadata
        s = self.study
        self.kinds = m["kinds"]
        grids = {0: np.asarray(s.panels[2].data["values"], float),
                 3: np.asarray(s.panels[2].data["second"], float),
                 7: np.asarray(s.panels[2].data["third"], float)}
        self.tex = {}
        for k, w in grids.items():
            lo, hi = (float(v) for v in np.percentile(w, [1, 99]))
            self.tex[k] = shrink_columns(shades(w, WEIGHT, lo, hi), int(self.BW))
        at_ = m["lines"]
        run1 = ((0.0, None),) + tuple((a, at_[k]) for k, a in enumerate(self.APPEAR)) + \
            ((self.SEQ_AT, m["sequential_line"]),)
        run2 = ((0.0, at_[0]),) + tuple((a, at_[k]) for k, a in enumerate(self.SHOW)) + \
            ((0.7, m["sequential_line"]),)
        run4 = ((0.0, None),) + tuple((a, at_[k]) for k, a in enumerate(self.COUNT_AT)
                                      if m["counts"][k]) + ((0.55, None),)
        self.CODE = (Code("cnn-model", run=run1), Code("cnn-model", run=run2),
                     Code("cnn-print-model", run=((0.0, None), (0.04, 0))),
                     Code("cnn-model", run=run4))
        self.EVIDENCE = (exact("the eight blocks of the CNN"),
                         exact("the shapes of the first batch"),
                         exact("what print(model) writes"),
                         exact("every weight and bias, counted"))

    def heading_for(self, k, u):
        m = self.meta
        return ("nn.Sequential: one block after another",
                "%s in, %s out" % (m["input_text"], m["shape_texts"][-1]),
                "print(model) lists the same blocks",
                "only Conv2d and Linear learn numbers")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "nn.Sequential puts eight blocks in a row: Conv2d, ReLU and MaxPool2d twice, "
            "then Flatten and Linear.",
            "Each block hands the next a new shape, from %s down to %s."
            % (m["input_text"], m["shape_texts"][-1]),
            "print(model) writes the same eight blocks, numbered 0 to 7, with their settings.",
            "Only the two Conv2d blocks and the Linear learn numbers: %s = %s, against the "
            "MLP's %s." % (" + ".join(commas(c) for c in m["counts"] if c), commas(m["total"]),
                           commas(m["mlp_total"])),
        ][k]

    # ------------------------------------------------------------------ geometry
    def squeeze(self, k, u):
        """0: the blocks as a stack; 1: a narrow column beside print(model)'s lines."""
        if k != 2:
            return 0.0
        return eased(u, 0.0, 0.05) * (1.0 - eased(u, 0.94, 0.99))

    def block_at(self, i, s):
        """Block ``i``'s centre and size (layout x, y; px w, h) at squeeze ``s``."""
        return (self.BX + (self.PX - self.BX) * s, self.ROW0 + self.ROW_STEP * i,
                self.BW + (self.PBW - self.BW) * s, self.BH)

    def lit(self, i, k, u):
        """Phase 3: the blocks of the half of print(model) on screen at full strength."""
        if k != 2:
            return 1.0
        (a0, a1), (b0, b1) = self.PAGES
        first = i in self.meta["page_blocks"][0]
        on1 = smoothstep(a0, a0 + 0.03, u) * (1 - smoothstep(a1 - 0.02, a1, u))
        on2 = smoothstep(b0, b0 + 0.03, u) * (1 - smoothstep(b1 - 0.02, b1, u))
        page = on1 if first else on2
        other = on2 if first else on1
        return 1.0 - 0.65 * other * (1 - page)

    def draw_block(self, sc, i, s, op):
        fx, fy, w, h = self.block_at(i, s)
        kind = self.kinds[i]
        c = self.P(fx, fy)
        if kind in ("Conv2d", "Linear"):
            self.rect(sc, c, w + 6, h + 6, soft(WEIGHT, 0.85), op, border=WEIGHT, border_w=1.8,
                      radius=4, lift=0.2)
            draw_image(self, sc, c, self.tex[i], w, h, op, grid=0.0, lift=0.3)
        elif kind == "ReLU":
            self.rect(sc, c, w + 6, h + 6, ZERO, op, border=ZERO_INK, border_w=1.8, radius=4,
                      lift=0.2)
        elif kind == "MaxPool2d":            # a data card holding a 2 x 2 square
            self.card(sc, fx, fy, w + 6, h + 6, op, lift=0.2)
            self.tiles(sc, [[0.35, 1.0], [0.15, 0.55]], fx, fy + self.frac(0, 2.0)[1], 12.0,
                       12.0, op, lo=0.0, hi=1.0, grid=0.3, lift=0.25)
        else:                                # Flatten: a data card holding a strip
            self.card(sc, fx, fy, w + 6, h + 6, op, lift=0.2)
            self.tiles(sc, np.linspace(0.15, 1.0, 10)[None], fx, fy + self.frac(0, 2.0)[1],
                       w - 8.0, 4.0, op, lo=0.0, hi=1.0, grid=0.0, lift=0.25)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        s = self.squeeze(k, u)
        names = op * (1.0 - smoothstep(0.0, 0.3, s))
        shown = [op * at(k, u, (0, a)) for a in self.APPEAR]
        if shown[-1] > 0.01:             # the data's path through the stack, behind it
            fx, top, _w, _h = self.block_at(0, s)
            _fx, bottom, _w, h = self.block_at(7, s)
            self.arrow(sc, (fx, top), (fx, bottom + self.frac(0, h / 2 + 12)[1]), AXIS,
                       shown[-1], width=1.6)
        for i in range(8):
            if shown[i] <= 0.01:
                continue
            self.draw_block(sc, i, s, shown[i] * self.lit(i, k, u))
            _fx, fy, _w, _h = self.block_at(i, 0.0)
            self.text(sc, self.NX, fy, m["names"][i], min(shown[i], names), SIZE, "lm")
        # nn.Sequential: one outline round them all
        seq = op * at(k, u, (0, self.SEQ_AT))
        fx, top, w, h = self.block_at(0, s)
        _fx, bottom, _w, _h = self.block_at(7, s)
        pad = self.frac(0, h / 2 + 8.0)[1]
        hh = self.span(0, bottom - top + 2 * pad)[1]
        self.rect(sc, self.P(fx, (top + bottom) / 2), w + 20 - 6 * s, hh, INK, seq, border=INK,
                  border_w=2.0, radius=8, lift=0.9, fill_op=0.0)

    # ------------------------------------------------------------------ 1: eight blocks
    def phase_1(self, sc, u, op):
        a = op * smoothstep(0.72, 0.77, u)
        self.text(sc, self.NX, 0.875, "one model: 8 blocks", a, HEAD, "lm", strong=True)

    # ------------------------------------------------------------------ 2: the shapes
    def phase_2(self, sc, u, op):
        m = self.meta
        for i, a in enumerate(self.SHOW):
            on = op * smoothstep(a, a + 0.04, u)
            _fx, fy, _w, _h = self.block_at(i, 0.0)
            self.text(sc, self.SX, fy, m["shape_texts"][i], on, SIZE, "rm", ink=DATA_INK)
        b = op * smoothstep(0.7, 0.75, u)
        self.big(sc, self.NX, 0.875, " → ".join(str(v) for v in m["side_walk"]), b,
                 label="the side of the maps", size=HEAD)

    # ------------------------------------------------------------------ 3: print(model)
    def phase_3(self, sc, u, op):
        for page, (a, b) in zip(self.meta["pages"], self.PAGES):
            for i, s in enumerate(page):
                on = op * smoothstep(a + 0.012 * i, a + 0.012 * i + 0.03, u) * (
                    1.0 - smoothstep(b - 0.02, b, u))
                self.label(sc, self.P(self.PTX, self.P0 + self.P_STEP * i), s, SIZE,
                           anchor="lm", opacity=on, ink=INK, kind="mono")

    # ------------------------------------------------------------------ 4: its weights
    def phase_4(self, sc, u, op):
        m = self.meta
        total = 0
        steps = [(0.0, "0")]
        for i, a in enumerate(self.COUNT_AT):
            on = op * smoothstep(a, a + 0.04, u)
            _fx, fy, _w, _h = self.block_at(i, 0.0)
            c = m["counts"][i]
            self.text(sc, self.SX, fy, commas(c), on, SIZE if c == 0 else HEAD, "rm",
                      ink=ZERO_INK if c == 0 else WEIGHT_INK)
            total += c
            steps.append((a, commas(total)))
        self.running(sc, self.NX, 0.875, steps, u, op * smoothstep(0.03, 0.07, u),
                     label="numbers the CNN learns", size=BIG)
        c = op * smoothstep(self.MLP_AT, self.MLP_AT + 0.05, u)
        self.text(sc, self.SX, 0.875, "the MLP: %s" % commas(m["mlp_total"]), c, HEAD, "rm")


# ====================================================================== 216
class Shapes(Cnn):
    """The side walked down, square by square; 16 x 5 x 5 = 400; 784 there instead."""

    TITLE = "Where the 400 comes from"
    PHASES = (("the digit", 10.0), ("stamp, pool", 14.0), ("again", 14.0),
              ("16 maps", 11.0), ("784 instead", 12.0))
    LAST = "The CNN, block by block"
    NEXT = "One digit through the CNN"
    STATION = ("data",) + ("forward",) * 4
    T = 2.7                              # px per tile: every square to the same scale
    PAD = 5.0
    X0, X1 = 0.06, 0.47                  # the row of squares
    YB = 0.585                           # the squares' bottom edge
    SLIDE = (0.06, 0.26)                 # phases 2 and 3: the stamp slides across
    NEW = (0.3, 0.64)                    # ... the next square appears (stamp, pool)
    POOL = (0.52, 0.6)                   # ... the 2 x 2 square slides along
    STEP_Y = 0.71                        # the arithmetic of the step under way
    # phase 4: 16 small maps, then the strip of 400 in nn.Linear's slot
    GX, GY, GSTEP = 0.0825, (0.685, 0.74), 0.025
    SY, SW = 0.825, 0.2                  # the strip: y, width (frame) for 400 numbers
    MAPS_AT, SUM_AT, STRIP_AT, SLOT_AT = 0.05, 0.32, 0.45, 0.62
    GROW, WRONG_AT, ERROR_AT = (0.12, 0.3), 0.32, 0.45

    def setup(self):
        m = self.meta = self.study.metadata
        d = self.study.panels[1].data
        self.maps = [np.asarray(d[k], float) for k in ("values", "conv1", "pool1", "conv2",
                                                        "pool2")]
        self.maps16 = np.asarray(self.study.panels[2].data["values"], float)
        self.walk = m["side_walk"]
        # one shading scale per stage: the digit as Normalize leaves it, each map 0 to its
        # stage's biggest
        self.scale = [(DATA_LO, DATA_HI)] + [(0.0, float(v.max())) for v in self.maps[1:]]
        sizes = [self.card_size(n, self.T, self.PAD) for n in self.walk]
        gap = (self.span(self.X1 - self.X0)[0] - sum(w for w, _h in sizes)) / (len(sizes) - 1)
        self.cx, x = [], self.X0
        for w, _h in sizes:
            self.cx.append(x + self.frac(w / 2)[0])
            x += self.frac(w + gap)[0]
        self.cy = [self.YB - self.frac(0, h / 2)[1] for _w, h in sizes]
        self.sizes = sizes
        tr = m["transitions"]
        self.CODE = (Code("cnn-one-batch", run=((0.0, 0), (0.2, 1)),
                          out=((0.3, m["batch_printed"]),)),
                     Code("cnn-model", run=((0.0, 1), (self.POOL[0], 2)),
                          out=((self.NEW[0], tr[0]), (self.NEW[1], tr[1]))),
                     Code("cnn-model", run=((0.0, 3), (self.POOL[0], 4)),
                          out=((0.0, tr[1]), (self.NEW[0], tr[2]), (self.NEW[1], tr[3]))),
                     Code("cnn-model", run=((0.0, 3), (self.STRIP_AT, 5), (self.SLOT_AT, 6)),
                          out=((0.0, tr[3]), (self.STRIP_AT, tr[4]), (self.SLOT_AT, tr[5]))),
                     Code("cnn-mistake-in-features", run=((0.0, 5), (0.08, 6))))
        self.EVIDENCE = (exact("the shape of the first batch"),) + \
            (exact("the side of each block's maps"),) * 2 + \
            (exact("16 × 5 × 5, counted"), exact("torch's own error"))

    def heading_for(self, k, u):
        if k in (1, 2):
            if u < self.POOL[0]:
                return "the 3 × 3 stamp loses 2" if k == 1 else "again: the stamp loses 2"
            return "pooling halves the side" if k == 1 else "pooling halves, rounding down"
        if k == 4:
            return "nn.Linear(784, 10): torch stops"
        return ("each digit: a 28 × 28 square", "", "",
                "16 maps of 5 × 5: 400 numbers", "")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "Each digit comes in as one square map of 28 × 28 numbers: the batch is %s."
            % m["shape_texts"][0],
            "A 3 × 3 stamp fits 26 places across 28, so the side loses 2; pooling 2 × 2 "
            "squares halves it to 13.",
            "Again: 13 − 2 = 11; then 11 ÷ 2 is 5.5, and pooling drops the odd last row and "
            "column: 5.",
            "%d maps of 5 × 5 are %s numbers per digit: the 400 of nn.Linear(16 * 5 * 5, 10)."
            % (m["maps"], m["in_features_sum"]),
            "With nn.Linear(%d, 10), torch stops: %d numbers arrive, but the layer expects %d."
            % (m["wants"], m["got"], m["wants"]),
        ][k]

    # ------------------------------------------------------------------ geometry
    APPEARS = ((0, 0.0), (1, 0.3), (1, 0.64), (2, 0.3), (2, 0.64))

    def cell(self, j):
        n = self.walk[j]
        return self.cells(self.cx[j], self.cy[j] + self.frac(0, 3.0)[1], n * self.T,
                          n * self.T, n, n)

    def step_text(self, k, u):
        """The arithmetic of the step under way, swapped in place: ``(big, small)``."""
        m = self.meta
        if k == 0 or (k == 1 and u < self.SLIDE[0]):
            return "28 × 28", "784 numbers: one map"
        s = m["steps"][2 * (k - 1):2 * k]
        if u < self.NEW[0]:
            n = self.places(k, u) + 1
            return str(n), "place" if n == 1 else "places across"
        if u < self.POOL[0]:
            return s[0]["sum"], "a 3 × 3 stamp loses 2"
        if k == 1:
            return s[1]["sum"], "2 × 2 squares become 1"
        return s[1]["sum"], "the odd row is dropped"

    def places(self, k, u):
        """Phases 2 and 3: where the sliding stamp is (column 0 .. side - 3)."""
        side = self.walk[2 * (k - 1)]
        return int(round(eased(u, *self.SLIDE) * (side - 3)))

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        dropped = op * at(k, u, (2, self.POOL[0] + 0.02))
        for j, when in enumerate(self.APPEARS):
            a = op * at(k, u, when)
            if a <= 0.01:
                continue
            cover = None
            if j == 3 and dropped > 0.5:
                cover = np.zeros((self.walk[3],) * 2, bool)
                cover[-1, :] = cover[:, -1] = True
            lo, hi = self.scale[j]
            self.map_card(sc, self.maps[j], self.cx[j], self.cy[j], self.T, a, lo, hi,
                          pad=self.PAD, cover=cover)
            self.text(sc, self.cx[j], self.YB + 0.03, str(self.walk[j]), a, HEAD, strong=True)
            if j == 0:
                continue
            # the arrow from the square before, and what it did to the side
            x0 = self.cx[j - 1] + self.frac(self.sizes[j - 1][0] / 2 + 4)[0]
            x1 = self.cx[j] - self.frac(self.sizes[j][0] / 2 + 4)[0]
            y = self.YB - 0.022
            self.arrow(sc, (x0, y), (x1, y), AXIS, a, width=1.6)
            self.text(sc, (x0 + x1) / 2, y - 0.035, m["steps"][j - 1]["op"], a, SIZE,
                      ink=DATA_INK)
        if k == 2 and dropped > 0.01:     # the 10 x 10 that pooling reads
            n = self.walk[3] - 1
            self.outline(sc, self.cell(3), 0, 0, n, n, self.T, dropped * op, ink=INK)
        # the arithmetic of the step under way, swapped in place
        on = op * self.kept(0, 2)
        if on > 0.01:
            big, small = self.step_text(k, u)
            self.running(sc, self.X0, self.STEP_Y, [(0.0, big)], 0.0, on, size=BIG)
            self.text(sc, self.X0, self.STEP_Y + 0.06, small, on, SIZE, "lm")
        # the strip of 400 in nn.Linear's slot (phases 4 and 5)
        keep = op * self.kept(3)
        if keep > 0.01:
            self.slot(sc, k, u, keep)

    def slot(self, sc, k, u, op):
        m = self.meta
        strip = op * at(k, u, (3, self.STRIP_AT))
        w = self.span(self.SW)[0]
        cx = self.X0 + 0.01 + self.SW / 2
        self.card(sc, cx, self.SY, w + 10, 26.0, strip, lift=0.1)
        self.strip(sc, self.study.panels[2].data["values"], cx, self.SY + self.frac(0, 2)[1],
                   w, 12.0, strip, lo=0.0, hi=float(self.maps16.max()))
        slot = op * at(k, u, (3, self.SLOT_AT))
        grow = eased(u, *self.GROW) if k == 4 else 0.0
        full = self.SW * (1 + (m["wants"] / m["in_features"] - 1) * grow)
        left = self.X0 + 0.01 - self.frac(9.0)[0]
        sw = self.span(full)[0] + 18.0
        self.rect(sc, self.P(left + self.frac(sw / 2)[0], self.SY), sw, 38.0, WEIGHT, slot,
                  border=WEIGHT_INK, border_w=2.0, radius=5, lift=0.6, fill_op=0.0)

    # ------------------------------------------------------------------ 1: the digit
    def phase_1(self, sc, u, op):
        pass

    # ------------------------------------------------------------------ 2, 3: a stamp, a pool
    def walk_phase(self, sc, k, u, op):
        j = 2 * (k - 1)                      # the square the stamp slides over
        cell = self.cell(j)
        s0 = self.SLIDE[0]
        on = op * smoothstep(s0, s0 + 0.03, u) * (1 - smoothstep(self.NEW[0] + 0.1,
                                                                 self.NEW[0] + 0.14, u))
        self.outline(sc, cell, 0, self.places(k, u), 3, 3, self.T, on, ink=WEIGHT_INK)
        # pooling: a 2 x 2 square slides along the new square's top row
        p0, p1 = self.POOL
        sq = op * smoothstep(p0, p0 + 0.03, u) * (1 - smoothstep(self.NEW[1] + 0.1,
                                                                 self.NEW[1] + 0.14, u))
        n = self.walk[j + 1] // 2
        c = 2 * int(round(eased(u, p0 + 0.02, p1) * (n - 1)))
        self.outline(sc, self.cell(j + 1), 0, c, 2, 2, self.T, sq, ink=INK)

    def phase_2(self, sc, u, op):
        self.walk_phase(sc, 1, u, op)

    def phase_3(self, sc, u, op):
        self.walk_phase(sc, 2, u, op)

    # ------------------------------------------------------------------ 4: 16 x 5 x 5
    def phase_4(self, sc, u, op):
        m = self.meta
        ring = op * smoothstep(0.02, 0.06, u)
        self.outline(sc, self.cell(4), 0, 0, 5, 5, self.T, ring, ink=INK)
        hi = float(self.maps16.max())
        for i, v in enumerate(self.maps16):
            a = op * smoothstep(self.MAPS_AT + 0.012 * i, self.MAPS_AT + 0.012 * i + 0.04, u)
            self.map_card(sc, v, self.GX + self.GSTEP * (i % 8), self.GY[i // 8], self.T, a,
                          0.0, hi, pad=3.0)
        b = op * smoothstep(self.SUM_AT, self.SUM_AT + 0.05, u)
        self.text(sc, 0.3, 0.71, m["in_features_sum"], b, HEAD, "lm", strong=True)
        c = op * smoothstep(self.SLOT_AT, self.SLOT_AT + 0.05, u)
        self.text(sc, 0.3, self.SY, "in_features = %d" % m["in_features"], c, SIZE, "lm",
                  ink=WEIGHT_INK)

    # ------------------------------------------------------------------ 5: 784 instead
    def phase_5(self, sc, u, op):
        m = self.meta
        a = op * smoothstep(0.04, 0.09, u)
        under = self.SY + 0.05
        self.text(sc, self.X0 + 0.01, under, "%d arrive" % m["got"], a, SIZE, "lm",
                  ink=DATA_INK)
        b = op * smoothstep(self.WRONG_AT, self.WRONG_AT + 0.05, u)
        right = self.X0 + 0.01 + self.SW * m["wants"] / m["in_features"]
        self.text(sc, right, under, "%d expected" % m["wants"], b, SIZE, "rm", ink=WEIGHT_INK)
        c = op * smoothstep(self.ERROR_AT, self.ERROR_AT + 0.05, u)
        for i, s in enumerate(m["error_lines"]):
            self.label(sc, self.P(self.X0 + 0.01, 0.665 + 0.041 * i), s, SIZE, anchor="lm",
                       opacity=c, ink=INK, kind="mono")


# ====================================================================== 217
class Forward(Cnn):
    """The digit; 8 maps; pooled; 16 maps, pooled; 400 numbers and 10 scores."""

    TITLE = "One digit through the CNN"
    PHASES = (("the digit", 9.0), ("8 maps", 12.0), ("pooled", 10.0), ("16 maps", 13.0),
              ("10 scores", 13.0))
    LAST = "Where the 400 comes from"
    NEXT = "A shared stamp's gradient"
    STATION = ("data",) + ("forward",) * 4
    DIG, DT = (0.105, 0.475), 2.2           # the digit card and its tiles (px)
    MT = 2.0                                # px per tile of every map
    COLS8 = (0.228, 0.298, 0.368, 0.438)
    COLS16 = tuple(0.214 + 0.034 * i for i in range(8))
    ROWS = (0.43, 0.565)
    APPEAR8 = tuple(0.08 + 0.055 * i for i in range(8))     # phase 2
    SHRINK = (0.1, 0.3)                     # phases 3 and 4 (from 0.5): pooling
    FADE8 = (0.03, 0.12)                    # phase 4: the 8 pooled maps give way
    APPEAR16 = tuple(0.1 + 0.018 * i for i in range(16))
    POOL16 = 0.5
    STRIP = (0.05, 0.3)                     # phase 5: the 400 numbers line up
    SY = 0.705
    SCORES_AT, WIN_AT = 0.42, 0.6
    BOX_Y, BOX_W = 0.8, 30.0

    def setup(self):
        m = self.meta = self.study.metadata
        s = self.study
        self.x = np.asarray(s.panels[0].data["values"], float)
        self.h1 = np.asarray(s.panels[1].data["values"], float)
        self.p1 = np.asarray(s.panels[1].data["pooled"], float)
        self.h2 = np.asarray(s.panels[2].data["values"], float)
        self.p2 = np.asarray(s.panels[2].data["pooled"], float)
        self.flat = np.asarray(s.panels[3].data["flat"], float)
        self.scores = np.asarray(s.panels[3].data["values"], float)
        self.hi1 = float(self.h1.max())
        self.hi2 = float(self.h2.max())
        st = m["shape_texts"]
        tr = ["%s → %s" % (st[a], st[b]) for a, b in
              (("x", "h1"), ("h1", "p1"), ("p1", "h2"), ("h2", "p2"), ("p2", "flat"),
               ("flat", "logits"))]
        self.CODE = (Code("cnn-one-digit", run=((0.0, 0), (0.3, 1)),
                          out=((0.38, m["printed"]),)),
                     Code("cnn-model", run=((0.0, 1),), out=((0.08, tr[0]),)),
                     Code("cnn-model", run=((0.0, 2),), out=((self.SHRINK[0], tr[1]),)),
                     Code("cnn-model", run=((0.0, 3), (self.POOL16, 4)),
                          out=((0.1, tr[2]), (self.POOL16, tr[3]))),
                     Code("cnn-model", run=((0.0, 5), (self.SCORES_AT, 6)),
                          out=((0.05, tr[4]), (self.SCORES_AT, tr[5]))))
        self.EVIDENCE = (measured("train_set[0], a real digit"),) + \
            (measured("the CNN after one epoch, train_set[0]"),) * 4

    def heading_for(self, k, u):
        m = self.meta
        if k == 3:
            return "16 stamps make 16 maps" if u < self.POOL16 else "pooled: 16 maps of 5 × 5"
        if k == 4:
            return "400 numbers in a row, then 10 scores"
        return ("one digit: train_set[0], a %d" % m["label"], "8 stamps make 8 maps",
                "pooling: each map a quarter the size")[k]

    def subtitle_for(self, k):
        m = self.meta
        st = m["shape_texts"]
        return [
            "train_set[0] is a %d: one digit, fed to the trained CNN as a batch of one, %s."
            % (m["label"], st["x"]),
            "nn.Conv2d(1, 8, 3) and ReLU make 8 maps of 26 × 26; ReLU sets %d%% of their "
            "numbers to 0." % round(100 * m["zero_share"]["h1"]),
            "nn.MaxPool2d(2) keeps the biggest of each 2 × 2 square: 8 maps of 13 × 13, a "
            "quarter of the numbers.",
            "16 stamps, each reading all 8 maps, make 16 maps of 11 × 11; pooled, they are "
            "5 × 5.",
            "Flatten lines up the 16 maps as 400 numbers; nn.Linear makes 10 scores, and "
            "the %d's is the biggest: %s (the %d next, %s)."
            % (m["label"], val(m["score"]), m["second"], val(m["second_score"])),
        ][k]

    # ------------------------------------------------------------------ the maps
    def maps8(self, sc, s, op):
        """The 8 first maps, pooled by ``s`` (0: 26 x 26; 1: 13 x 13)."""
        for i in range(8):
            fx, fy = self.COLS8[i % 4], self.ROWS[i // 4]
            if s < 1.0:
                tile = self.MT * (1.0 - 0.5 * s)
                self.map_card(sc, self.h1[i], fx, fy, tile, op, 0.0, self.hi1, pad=4.0)
            else:
                self.map_card(sc, self.p1[i], fx, fy, self.MT, op, 0.0, self.hi1, pad=4.0)

    def maps16(self, sc, s, ops):
        """The 16 second maps, pooled by ``s`` (0: 11 x 11; 1: 5 x 5); ``ops``: each one's
        opacity."""
        for i in range(16):
            fx, fy = self.COLS16[i % 8], self.ROWS[i // 8]
            if s < 1.0:
                tile = self.MT * (1.0 - 0.5 * s)
                self.map_card(sc, self.h2[i], fx, fy, tile, ops[i], 0.0, self.hi2, pad=4.0)
            else:
                self.map_card(sc, self.p2[i], fx, fy, self.MT, ops[i], 0.0, self.hi2, pad=4.0)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        dx, dy = self.DIG
        self.map_card(sc, self.x, dx, dy, self.DT, op, DATA_LO, DATA_HI)
        _w, h = self.card_size(28, self.DT)
        self.text(sc, dx, dy + self.frac(0, h / 2 + 16)[1], "y = %d" % m["label"], op, HEAD)
        # the 8 first maps: phases 2 and 3, giving way in phase 4
        if k in (1, 2, 3):
            if k == 1:
                for i in range(8):
                    a = op * smoothstep(self.APPEAR8[i], self.APPEAR8[i] + 0.04, u)
                    if a > 0.01:
                        self.map_card(sc, self.h1[i], self.COLS8[i % 4], self.ROWS[i // 4],
                                      self.MT, a, 0.0, self.hi1, pad=4.0)
            else:
                s = eased(u, *self.SHRINK) if k == 2 else 1.0
                a = op * (1.0 - smoothstep(*self.FADE8, u)) if k == 3 else op
                self.maps8(sc, s, a)
        if k >= 1:
            a = op * smoothstep(0.03, 0.08, u) if k == 1 else op
            x0 = dx + self.frac(self.card_size(28, self.DT)[0] / 2 + 5)[0]
            self.arrow(sc, (x0, dy), (0.188, dy), AXIS, a, width=1.6)
        # the 16 second maps: phase 4, and pooled through phase 5
        if k >= 3:
            ops = [op * smoothstep(a, a + 0.04, u) if k == 3 else op for a in self.APPEAR16]
            s = eased(u, self.POOL16 + self.SHRINK[0] * 0.5, self.POOL16 + 0.2) \
                if k == 3 else 1.0
            self.maps16(sc, s, ops)

    # ------------------------------------------------------------------ 1: the digit
    def phase_1(self, sc, u, op):
        a = op * smoothstep(0.45, 0.52, u)
        dx, dy = self.DIG
        w, _h = self.card_size(28, self.DT)
        self.text(sc, dx + self.frac(w / 2 + 14)[0], dy, "28 × 28 numbers", a, SIZE, "lm")

    def phase_2(self, sc, u, op):
        pass

    def phase_3(self, sc, u, op):
        pass

    def phase_4(self, sc, u, op):
        pass

    # ------------------------------------------------------------------ 5: 10 scores
    def phase_5(self, sc, u, op):
        m = self.meta
        x0, x1 = 0.07, 0.465
        w = self.span(x1 - x0)[0]
        cx = (x0 + x1) / 2
        reveal = eased(u, *self.STRIP)
        on = op * smoothstep(self.STRIP[0] - 0.03, self.STRIP[0], u)
        rgb = shrink_columns(shades(self.flat[None], DATA, 0.0, self.hi2), int(w))
        n = int(np.ceil(rgb.shape[1] * reveal - 1e-9))
        rgb[:, n:] = PAPER
        self.text(sc, x0, self.SY - 0.042, "%d numbers" % len(self.flat), on, SIZE, "lm",
                  ink=DATA_INK)
        self.card(sc, cx, self.SY, w + 10, 26.0, on, lift=0.1)
        draw_image(self, sc, self.P(cx, self.SY + self.frac(0, 2)[1]), rgb, w, 12.0, on,
                   grid=0.0)
        # the 10 scores, each a box shaded by its size, the digit it is for inside
        lo, hi = float(self.scores.min()), float(self.scores.max())
        b = op * smoothstep(self.SCORES_AT, self.SCORES_AT + 0.05, u)
        win = op * smoothstep(self.WIN_AT, self.WIN_AT + 0.05, u)
        step = (x1 - x0) / 10
        for d, v in enumerate(self.scores):
            fx = x0 + step * (d + 0.5)
            fill = tuple(float(c) for c in shades([v], DATA, lo, hi)[0])
            deep = (v - lo) / (hi - lo) > 0.62
            self.box(sc, fx, self.BOX_Y, self.BOX_W, 28.0, fill, b, text=str(d),
                     ink=PAPER if deep else INK, border=DATA, size=SIZE, radius=4)
            if d in (m["label"], m["second"]):
                self.text(sc, fx, self.BOX_Y + 0.05, val(v), win, SIZE,
                          ink=INK if d == m["label"] else DATA_INK)
        fx = x0 + step * (m["label"] + 0.5)
        self.rect(sc, self.P(fx, self.BOX_Y), self.BOX_W + 6, 34.0, INK, win, border=INK,
                  border_w=2.4, radius=5, lift=1.2, fill_op=0.0)
        self.text(sc, x1, self.BOX_Y + 0.07, "the %d wins" % m["label"], win, HEAD, "rm",
                  strong=True)


ANIMATIONS = {"blocks-cnn-the-model": TheModel, "blocks-cnn-shapes": Shapes,
              "blocks-cnn-forward": Forward}

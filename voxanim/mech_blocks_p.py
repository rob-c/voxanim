"""Animations 230-231 of "Blocks and code: padding" (see :mod:`voxanim.blocks`): padding as the
way to make one batch, one block of numbers, out of things that differ in size, drawn from
:mod:`voxanim.study_blocks_p`.

* :class:`ABatchIsABlock` (``padding-a-batch-is-a-block``): the first four training digits
  cut to their ink, drawn to scale as four cards of four sizes; ``torch.stack`` stops with
  PyTorch's own error, in plain words (entry 1 is 20 x 17, not 20 x 20: there would be
  holes); each piece padded on the right and at the bottom with the paper's value, -0.42,
  to 28 x 28, the padding striped so that it can be seen; then the four stack into one
  block, ``[4, 1, 28, 28]``, of which 1,716 of 3,136 numbers are padding.
* :class:`PaddingLinesOfText` (``padding-lines-of-text``): the first eight lines of Tiny
  Shakespeare, each character swapped for its id, as rows of tiles of eight lengths; padded
  with id 0 to the longest, 50; then the mask ``batch != 0``, row by row: 163 of 400 cells
  real, 59% padding.

Padding is marked the same way in both: grey stripes over the tiles it fills (shape backs
up colour). The stripes are baked into the tiles' picture, so they look the same in both
themes. The picture is on the left; the code, the loop map and the evidence tag are the
series' furniture (:class:`~voxanim.blocks.Blocks`); what stays across phases is drawn in
``steady``, so nothing blanks between them.
"""

from __future__ import annotations

import numpy as np

from .blocks import (
    BIG,
    DATA,
    DATA_HI,
    DATA_LO,
    GRAD,
    HEAD,
    INK,
    QUIET,
    SIZE,
    ZERO,
    Blocks,
    Code,
    commas,
    exact,
    measured,
    shades,
)
from .diagram import draw_image
from .easing import ease_in_out, smoothstep
from .mech import PAPER, dark, soft

DATA_INK = dark(DATA, 0.3)
GRAD_INK = dark(GRAD, 0.25)
HATCH = np.array((0.47, 0.49, 0.52), np.float32)      # the stripes that mark padding


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def at(k, u, when, width=0.04):
    """0 before ``when`` = ``(phase, u0)``, fading up to 1 just after, 1 from then on."""
    pk, pu = when
    if k != pk:
        return 1.0 if k > pk else 0.0
    return smoothstep(pu, pu + width, u)


def hatched(rgb, pad, sy, sx, period, width=2, strength=0.72, border=None):
    """Tile colours ``rgb`` (rows, cols, 3) as a texture of ``sy x sx`` texels per tile,
    the tiles marked by ``pad`` striped with grey diagonal lines (``period`` texels apart,
    ``width`` wide, running on across tiles); ``border``: the last column of texels of each
    tile darkened by that much (the tiles' edges)."""
    rgb = np.asarray(rgb, np.float32)
    tex = np.repeat(np.repeat(rgb, sy, 0), sx, 1)
    if border:
        edge = (np.arange(tex.shape[1]) % sx) == sx - 1
        tex[:, edge] *= 1.0 - border
    if pad is not None and np.any(pad):
        m = np.repeat(np.repeat(np.asarray(pad, bool), sy, 0), sx, 1)
        y, x = np.indices(m.shape)
        sel = m & (((x + y) % period) < width)
        tex[sel] = tex[sel] * (1.0 - strength) + HATCH * strength
    return tex


class Padding(Blocks):
    """What the two share: the striped swatch that says what the stripes mean."""

    def swatch(self, sc, fx, fy, text, op, fill=PAPER, px=18.0):
        """A small striped tile at ``(fx, fy)`` and ``text`` after it."""
        if op <= 0.01:
            return
        tex = hatched(np.full((1, 1, 3), fill, np.float32), np.ones((1, 1), bool), 18, 18, 6)
        c = self.P(fx, fy)
        self.rect(sc, c, px + 4, px + 4, PAPER, op, border=QUIET, border_w=1.2, radius=3,
                  lift=0.25)
        draw_image(self, sc, c, tex, px, px, op, grid=0.0, lift=0.3)
        self.text(sc, fx + self.frac(px / 2 + 9)[0], fy, text, op, SIZE, "lm")

    def chip(self, sc, fx, fy, fill, text, op, px=18.0, border=None):
        """A small plain tile of ``fill`` and ``text`` after it (a legend)."""
        if op <= 0.01:
            return
        self.rect(sc, self.P(fx, fy), px, px, fill, op, border=border, border_w=1.4, radius=3,
                  lift=0.3)
        self.text(sc, fx + self.frac(px / 2 + 9)[0], fy, text, op, SIZE, "lm")


# ====================================================================== 230
class ABatchIsABlock(Padding):
    """Four digits cut to their ink; torch.stack's error; padded to 28 x 28; one block."""

    TITLE = "A batch is one block"
    PHASES = (("cut to the ink", 13.0), ("stack fails", 12.0), ("pad to 28 × 28", 15.0),
              ("one block", 18.0))
    LAST = "The CNN with padding, measured"
    NEXT = "Padding lines of text"
    STATION = ("data",) * 4
    T = 2.8                                      # px per pixel: every card to the same scale
    CX = (0.1075, 0.2125, 0.3175, 0.4225)        # the four slots' centres
    TOP = 0.405                                  # the tiles' top edge in every slot
    SIZE_Y, NOTE_Y = 0.597, 0.638                # the two labels under each slot
    LOW = (0.70, 0.75, 0.80)                     # the lines of the picture's lower part
    # phase 1: the ink's box outlined; the card shrinks to it; the piece slides to the corner
    BOX_AT, SHRINK, SLIDE, SIZES_AT = 0.2, (0.38, 0.5), (0.58, 0.68), 0.72
    RING_AT, HOLES_AT = 0.3, 0.45                # phase 2
    LEGEND_AT = 0.05                             # phase 3: the stripes explained
    GROW = ((0.22, 0.32), (0.36, 0.46), (0.5, 0.6), (0.64, 0.74))
    STACK = ((0.0, 0.0), (0.05, 0.12), (0.16, 0.23), (0.27, 0.34))   # phase 4: card i slides
    STEP = 8.0                                   # px between the cards of the stack
    BLOCK_AT, LESSON_AT = 0.38, 0.48
    BLOCK_X = 0.215                              # phase 4: the block's shape, beside it

    def setup(self):
        m = self.meta = self.study.metadata
        s = self.study
        self.full = [np.asarray(v, float) for v in s.panels[0].data["values"]]
        self.padded = [np.asarray(v, float) for v in s.panels[2].data["values"]]
        self.shapes = [(sh[1], sh[2]) for sh in m["shapes"]]
        self.boxes = m["boxes"]
        self.side = m["side"]
        self.CODE = (
            Code("pad-cut", run=((0.0, 0), (0.7, 1)),
                 out=((self.SIZES_AT, "\n".join(m["printed_lines"])),)),
            Code("pad-cut", run=((0.0, 1), (0.06, 2)),
                 out=((0.15, "\n".join(m["error_lines"])),)),
            Code("pad-to-block", run=((0.0, 0), (0.16, 4), (self.GROW[0][0], 5)),
                 out=((0.04, m["bg_code"]),)),
            Code("pad-to-block", run=((0.0, 5), (0.04, 6)),
                 out=((self.BLOCK_AT, m["printed_block"]),)))
        self.EVIDENCE = (measured("the first 4 training digits, cut"),
                         exact("torch's own error"),
                         exact("the padding, counted cell by cell"),
                         exact("4 × 28 × 28 = 3,136 numbers"))

    def heading_for(self, k, u):
        if k == 3:
            return ("now torch.stack makes one block" if u < self.LESSON_AT
                    else "any input: one full block, no holes")
        return ("four digits, cut to their ink", "torch.stack stops: the sizes differ",
                "pad each one to 28 × 28")[k]

    def subtitle_for(self, k):
        m = self.meta
        sizes = m["sizes"]
        return [
            "Cut to their ink, the first four training digits have four sizes: %s, %s, %s "
            "and %s." % tuple(sizes),
            "torch.stack needs every piece the same size: entry %d is %s, not %s, so it "
            "stops with an error." % (m["other"], sizes[m["other"]], sizes[0]),
            "pad_to adds the paper's value, %s, on the right and at the bottom, until each "
            "digit is %d × %d." % (m["bg_text"], self.side, self.side),
            "Now torch.stack makes one block, %s, with no holes: %s of its %s numbers are "
            "padding." % (m["block_text"], commas(m["padding"]), commas(m["cells"])),
        ][k]

    # ------------------------------------------------------------------ geometry
    def left(self, i):
        """Slot ``i``'s left tile edge (layout x)."""
        return self.CX[i] - self.frac(self.side * self.T / 2)[0]

    def piece(self, sc, img, pad, left, top, op, lift=0.2):
        """``img`` (rows x cols) as a data card, its top-left tile at ``(left, top)``; the
        tiles marked by ``pad`` striped."""
        if op <= 0.01 or img.size == 0:
            return
        rows, cols = img.shape
        w, h = cols * self.T, rows * self.T
        fx = left + self.frac(w / 2)[0]
        fy = top + self.frac(0, h / 2)[1]
        self.card(sc, fx, fy - self.frac(0, 3.0)[1], w + 14, h + 20, op, lift=lift)
        tex = hatched(shades(img, DATA, DATA_LO, DATA_HI), pad, 3, 3, 6)
        draw_image(self, sc, self.P(fx, fy), tex, w, h, op, grid=0.0, lift=lift + 0.004)

    def state(self, i, k, u):
        """What card ``i`` shows: ``(img, pad, left, top)``."""
        r0, r1, c0, c1 = self.boxes[i]
        h, w = self.shapes[i]
        L, top = self.left(i), self.TOP
        T = self.T
        if k == 0:
            s, g = eased(u, *self.SHRINK), eased(u, *self.SLIDE)
            ra, rb = round(r0 * s), round(self.side + (r1 + 1 - self.side) * s)
            ca, cb = round(c0 * s), round(self.side + (c1 + 1 - self.side) * s)
            if g <= 0.0:
                return (self.full[i][ra:rb, ca:cb], None, L + self.frac(ca * T)[0],
                        top + self.frac(0, ra * T)[1])
            return (self.full[i][r0:r1 + 1, c0:c1 + 1], None,
                    L + self.frac(c0 * T * (1 - g))[0], top + self.frac(0, r0 * T * (1 - g))[1])
        pad = np.ones((self.side, self.side), bool)
        pad[:h, :w] = False
        if k == 1:
            return self.padded[i][:h, :w], None, L, top
        if k == 2:
            g = eased(u, *self.GROW[i])
            rows = h + round((self.side - h) * g)
            cols = w + round((self.side - w) * g)
            return self.padded[i][:rows, :cols], pad[:rows, :cols], L, top
        a, b = self.STACK[i]
        g = eased(u, a, b) if b > a else 1.0
        x1 = self.left(0) + self.frac(self.STEP * i)[0]
        y1 = top + self.frac(0, self.STEP * i)[1]
        return self.padded[i], pad, L + (x1 - L) * g, top + (y1 - top) * g

    def size_label(self, i, k, u):
        h, w = self.shapes[i]
        if k == 0:
            return "%d × %d" % (self.side, self.side) if u < self.SIZES_AT else \
                self.meta["sizes"][i]
        if k == 2 and u >= self.GROW[i][1]:
            return "%d × %d" % (self.side, self.side)
        return "%d × %d" % (h, w)

    def note_label(self, i, k, u):
        if k == 1:
            return "entry %d" % i
        if k == 2:
            if u >= self.GROW[i][1]:
                return "+%d pads" % self.meta["pads"][i]
            return "entry %d" % i
        return ""

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        for i in range(4):
            img, pad, left, top = self.state(i, k, u)
            self.piece(sc, img, pad, left, top, op, lift=0.2 + 0.02 * i)
        # the labels under each slot, until the cards leave for the stack
        under = op * (1.0 - at(k, u, (3, 0.02)))
        for i in range(4):
            self.text(sc, self.CX[i], self.SIZE_Y, self.size_label(i, k, u), under, HEAD,
                      strong=True)
            note = self.note_label(i, k, u)
            if note:
                fade = at(k, u, (1, 0.04)) if k == 1 else 1.0
                ink = QUIET if note.startswith("+") else None
                self.text(sc, self.CX[i], self.NOTE_Y, note, under * fade, SIZE, ink=ink)
        # what the stripes mean, from phase 3 to the end
        legend = op * self.kept(2) * at(k, u, (2, self.LEGEND_AT))
        self.swatch(sc, self.left(0), self.LOW[0], "padding: %s, the paper's value"
                    % self.meta["bg_text"], legend)

    # ------------------------------------------------------------------ 1: cut to the ink
    def phase_1(self, sc, u, op):
        box = op * smoothstep(self.BOX_AT, self.BOX_AT + 0.04, u) * \
            (1.0 - smoothstep(self.SHRINK[1] + 0.02, self.SHRINK[1] + 0.06, u))
        if box > 0.01:
            for i in range(4):
                r0, r1, c0, c1 = self.boxes[i]
                cell = self.cells(self.CX[i], self.TOP + self.frac(0, self.side * self.T / 2)[1],
                                  self.side * self.T, self.side * self.T, self.side, self.side)
                self.outline(sc, cell, r0, c0, r1 - r0 + 1, c1 - c0 + 1, self.T, box, ink=INK)
        note = op * smoothstep(self.BOX_AT, self.BOX_AT + 0.04, u)
        self.text(sc, self.left(0), self.LOW[0], "cut: the box round the ink", note, SIZE,
                  "lm")

    # ------------------------------------------------------------------ 2: stack fails
    def phase_2(self, sc, u, op):
        m = self.meta
        o = m["other"]
        ring = op * smoothstep(self.RING_AT, self.RING_AT + 0.04, u)
        for i, ink in ((0, INK), (o, GRAD_INK)):
            h, w = self.shapes[i]
            fx = self.left(i) + self.frac(w * self.T / 2)[0]
            fy = self.TOP + self.frac(0, h * self.T / 2 - 3.0)[1]
            self.rect(sc, self.P(fx, fy), w * self.T + 24, h * self.T + 30, ink, ring,
                      border=ink, border_w=2.4, radius=9, lift=1.0, fill_op=0.0)
        holes = op * smoothstep(self.HOLES_AT, self.HOLES_AT + 0.04, u)
        h0, w0 = self.shapes[0]
        for i in range(4):
            h, w = self.shapes[i]
            if (h, w) == (h0, w0):
                continue
            fx = self.left(i) + self.frac((w + (w0 - w) / 2) * self.T + 1.5)[0]
            fy = self.TOP + self.frac(0, h0 * self.T / 2)[1]
            self.rect(sc, self.P(fx, fy), (w0 - w) * self.T + 3, h0 * self.T, soft(GRAD, 0.55),
                      holes, border=GRAD_INK, border_w=1.6, radius=2, lift=0.6, fill_op=0.8)
        x = self.left(0)
        a = op * smoothstep(0.15, 0.19, u)
        self.text(sc, x, self.LOW[0], "torch.stack needs one size", a, SIZE, "lm",
                  strong=True)
        self.text(sc, x, self.LOW[1], "entry %d: %s, not %s" % (o, m["sizes"][o], m["sizes"][0]),
                  ring, SIZE, "lm", ink=GRAD_INK)
        self.chip(sc, x + self.frac(9.0)[0], self.LOW[2], soft(GRAD, 0.55),
                  "red: holes, no numbers there", holes, border=GRAD_INK)

    # ------------------------------------------------------------------ 3: pad to 28 x 28
    def phase_3(self, sc, u, op):
        pass

    # ------------------------------------------------------------------ 4: one block
    def phase_4(self, sc, u, op):
        m = self.meta
        steps = [(0.0, "%s of %s" % (commas(m["running_pads"][0]), commas(m["running_cells"][0])))]
        for i in range(1, 4):
            steps.append((self.STACK[i][1], "%s of %s" % (commas(m["running_pads"][i]),
                                                          commas(m["running_cells"][i]))))
        a = op * smoothstep(0.04, 0.08, u)
        x = self.left(0) - self.frac(11.0)[0]           # under the swatch's left edge
        self.running(sc, x, self.LOW[2] + 0.012, steps, u, a, label="numbers that are padding",
                     size=HEAD)
        c = op * smoothstep(self.BLOCK_AT + 0.06, self.BLOCK_AT + 0.1, u)
        self.text(sc, x, self.LOW[2] + 0.06, "%d%% padding" % round(100 * m["padding_share"]),
                  c, SIZE, "lm", ink=QUIET)
        b = op * smoothstep(self.BLOCK_AT, self.BLOCK_AT + 0.04, u)
        self.big(sc, self.BLOCK_X, 0.445, m["block_text"], b, label="one block", size=BIG)
        self.text(sc, self.BLOCK_X, 0.505, "%s numbers" % commas(m["cells"]), b, SIZE, "lm")


# ====================================================================== 231
class PaddingLinesOfText(Padding):
    """Eight lines; one id per character; padded with 0 to 50; the mask."""

    TITLE = "Padding lines of text"
    PHASES = (("eight lines", 12.0), ("one id each", 13.0), ("padded to 50", 14.0),
              ("the mask", 14.0))
    LAST = "A batch is one block"
    NEXT = "Masks: ignore the padding"
    STATION = ("data",) * 4
    TW, TH = 7.0, 15.0                 # px per character tile
    X0 = 0.092                         # the rows' left edge
    Y0, ROW = 0.40, 21.0               # the first row's centre; px from row to row
    NUM_X = 0.077                      # the lengths, right-aligned before the rows
    LOW_Y = 0.80                       # the big number under the rows
    RIGHT = 0.3                        # the notes beside it
    LENGTHS_AT, RANGE_AT = 0.35, 0.6   # phase 1
    SWITCH = tuple(0.06 + 0.055 * i for i in range(8))        # phase 2: text -> tiles
    IDS_AT = 0.56
    LONGEST_AT = 0.04                  # phase 3
    PAD = tuple(0.24 + 0.045 * i for i in range(8))
    BLOCK_AT = 0.64
    MASK = tuple(0.08 + 0.055 * i for i in range(8))          # phase 4: batch -> mask
    SHARE_AT = 0.6

    def setup(self):
        m = self.meta = self.study.metadata
        self.ids = np.asarray(self.study.panels[1].data["values"], float)
        self.mask = np.asarray(self.study.panels[2].data["values"], bool)
        self.lengths = m["lengths"]
        self.longest = m["longest"]
        self.rows = len(self.lengths)
        self.top_id = m["top_id"]
        self.CODE = (
            Code("pad-text", run=((0.0, 0), (0.3, 3), (0.42, 4)),
                 out=((self.LENGTHS_AT, m["lengths_out"]),)),
            Code(("pad-text", "pad-sequences"), pick=(1, 2, 5, 6),
                 run=((0.0, 0), (self.SWITCH[0], 1), (self.IDS_AT, 2)),
                 out=((0.02, m["vocab_out"]), (self.IDS_AT, m["seq_out"]))),
            Code("pad-sequences", run=((0.0, 2),),
                 out=((self.BLOCK_AT, m["printed_shape"]),)),
            Code("pad-sequences", run=((0.0, 2), (0.04, 4)),
                 out=((self.SHARE_AT, "\n".join(m["printed_lines"])),)))
        self.EVIDENCE = (exact("the first 8 lines of Tiny Shakespeare"),
                         exact("each character's id"),
                         exact("the pads, counted"),
                         exact("%d of %d cells, counted" % (m["real"], m["cells"])))

    def heading_for(self, k, u):
        if k == 2:
            return ("the longest line: %d characters" % self.longest if u < self.PAD[0]
                    else "pad with 0 up to %d" % self.longest)
        return ("eight lines, eight lengths", "each character becomes its id",
                "", "the mask: True where a character is real")[k]

    def subtitle_for(self, k):
        m = self.meta
        n = m["lengths"]
        return [
            "The first eight lines of Tiny Shakespeare are %s and %d characters long."
            % (", ".join(str(v) for v in n[:-1]), n[-1]),
            "Each character is swapped for its id, a number from 1 to %d; id 0 is kept for "
            "padding." % m["top_id"],
            "pad_sequence pads every line with id 0 up to the longest, %d: one block of %s."
            % (self.longest, m["batch_text"]),
            "batch != 0 marks the real characters: %d of the %d cells, so %d%% of the block "
            "is padding." % (m["real"], m["cells"], round(100 * m["padding_share"])),
        ][k]

    # ------------------------------------------------------------------ geometry
    def row_y(self, i):
        return self.Y0 + self.frac(0, self.ROW * i)[1]

    def row_tiles(self, sc, i, cols, op, mask=False):
        """Row ``i``'s first ``cols`` tiles: the ids (``mask``: True / False)."""
        if op <= 0.01 or cols <= 0:
            return
        n = self.lengths[i]
        pad = np.arange(cols) >= n
        if mask:
            rgb = np.where(pad[:, None], np.asarray(ZERO, np.float32),
                           shades([1.0], DATA, 0.0, 1.0)[0])[None]
        else:
            rgb = shades(self.ids[i, :cols], DATA, -30.0, float(self.top_id))
            rgb[pad] = PAPER
            rgb = rgb[None]
        tex = hatched(rgb, pad[None], int(self.TH), int(self.TW), 5, 2, border=0.22)
        w = cols * self.TW
        fx = self.X0 + self.frac(w / 2)[0]
        draw_image(self, sc, self.P(fx, self.row_y(i)), tex, w, self.TH, op, grid=0.0,
                   lift=0.3)

    def block_box(self, sc, op):
        """An outline round the whole padded block."""
        w = self.longest * self.TW
        h = self.ROW * (self.rows - 1) + self.TH
        fx = self.X0 + self.frac(w / 2)[0]
        fy = self.Y0 + self.frac(0, h / 2 - self.TH / 2)[1]
        self.rect(sc, self.P(fx, fy), w + 6, h + 6, INK, op, border=INK, border_w=2.0,
                  radius=3, lift=1.0, fill_op=0.0)

    # ------------------------------------------------------------------ what stays
    def page(self, sc, op):
        """The data card the eight rows sit on, as wide as the longest."""
        w = self.longest * self.TW
        h = self.ROW * (self.rows - 1) + self.TH
        fx = self.X0 + self.frac(w / 2)[0]
        fy = self.Y0 + self.frac(0, h / 2 - self.TH / 2 - 2.0)[1]
        self.card(sc, fx, fy, w + 14, h + 18, op, lift=0.2)

    def steady(self, sc, k, u, op):
        self.page(sc, op)
        # the lines as text, until each row turns into tiles in phase 2
        if k <= 1:
            for i, line in enumerate(self.meta["text"]):
                a = op if k == 0 else op * (1.0 - smoothstep(self.SWITCH[i],
                                                             self.SWITCH[i] + 0.025, u))
                self.text(sc, self.X0, self.row_y(i), line, a, SIZE, "lm")
        # the lengths, from phase 1 on
        num = op * at(k, u, (0, self.LENGTHS_AT))
        for i, n in enumerate(self.lengths):
            self.text(sc, self.NUM_X, self.row_y(i), str(n), num, SIZE, "rm", ink=DATA_INK)
        # the rows of tiles: the real characters from phase 2, the pads from phase 3, the
        # mask in phase 4
        for i, n in enumerate(self.lengths):
            if k == 0:
                continue
            on = op * (at(k, u, (1, self.SWITCH[i] + 0.03), 0.03) if k == 1 else 1.0)
            cols = n
            if k >= 2:
                cols = n + round((self.longest - n) * (eased(u, self.PAD[i], self.PAD[i] + 0.04)
                                                       if k == 2 else 1.0))
            m = at(k, u, (3, self.MASK[i]), 0.03) if k == 3 else 0.0
            self.row_tiles(sc, i, cols, on * (1.0 - m))
            self.row_tiles(sc, i, cols, on * m, mask=True)
        if k >= 2:
            self.block_box(sc, op * at(k, u, (2, self.BLOCK_AT)))

    # ------------------------------------------------------------------ 1: eight lines
    def phase_1(self, sc, u, op):
        a = op * smoothstep(self.RANGE_AT, self.RANGE_AT + 0.04, u)
        self.big(sc, self.X0, self.LOW_Y, "%d to %d" % (min(self.lengths), self.longest), a,
                 label="characters per line", size=BIG)

    # ------------------------------------------------------------------ 2: one id each
    def phase_2(self, sc, u, op):
        m = self.meta
        b = op * smoothstep(self.IDS_AT, self.IDS_AT + 0.04, u)
        n = len(m["first"])
        w = n * self.TW
        self.rect(sc, self.P(self.X0 + self.frac(w / 2)[0], self.row_y(0)), w + 7, self.TH + 7,
                  INK, b, border=INK, border_w=2.2, radius=3, lift=1.0, fill_op=0.0)
        self.text(sc, self.X0, self.LOW_Y - 0.05, "the first line:", b, SIZE, "lm")
        self.text(sc, self.X0, self.LOW_Y, "   ".join("%s → %d" % (ch, i) for ch, i in m["first"]),
                  b, HEAD, "lm", strong=True)
        c = op * smoothstep(0.1, 0.14, u)
        self.text(sc, self.X0, self.LOW_Y + 0.05, "id 0 is kept for padding", c, SIZE, "lm",
                  ink=QUIET)

    # ------------------------------------------------------------------ 3: padded to 50
    def phase_3(self, sc, u, op):
        m = self.meta
        r = m["longest_row"]
        ring = op * smoothstep(self.LONGEST_AT, self.LONGEST_AT + 0.04, u) * \
            (1.0 - smoothstep(self.BLOCK_AT - 0.04, self.BLOCK_AT, u))
        w = self.longest * self.TW
        self.rect(sc, self.P(self.X0 + self.frac(w / 2)[0], self.row_y(r)), w + 8, self.TH + 8,
                  INK, ring, border=INK, border_w=2.2, radius=4, lift=1.0, fill_op=0.0)
        a = op * smoothstep(self.LONGEST_AT, self.LONGEST_AT + 0.04, u)
        steps = [(0.0, str(self.longest)), (self.BLOCK_AT, m["batch_text"])]
        label = "the longest" if u < self.BLOCK_AT else "one block"
        self.running(sc, self.X0, self.LOW_Y, steps, u, a, label=label, size=BIG)
        b = op * smoothstep(self.PAD[0], self.PAD[0] + 0.04, u)
        self.swatch(sc, self.RIGHT, self.LOW_Y - 0.035, "a pad: id 0", b)
        c = op * smoothstep(self.BLOCK_AT, self.BLOCK_AT + 0.04, u)
        self.text(sc, self.RIGHT - self.frac(9.0)[0], self.LOW_Y + 0.025, "%d pads" % m["padding"],
                  c, SIZE, "lm", ink=QUIET)

    # ------------------------------------------------------------------ 4: the mask
    def phase_4(self, sc, u, op):
        m = self.meta
        steps = [(0.0, "0")] + [(a + 0.015, str(v)) for a, v in zip(self.MASK, m["running_real"])]
        steps.append((self.SHARE_AT, "%d of %d" % (m["real"], m["cells"])))
        a = op * smoothstep(0.03, 0.07, u)
        self.running(sc, self.X0, self.LOW_Y, steps, u, a, label="real characters", size=BIG)
        b = op * smoothstep(self.MASK[0], self.MASK[0] + 0.04, u)
        self.chip(sc, self.RIGHT, self.LOW_Y - 0.035, shades([1.0], DATA, 0.0, 1.0)[0],
                  "True: a character", b)
        self.swatch(sc, self.RIGHT, self.LOW_Y + 0.01, "False: a pad", b, fill=ZERO)
        c = op * smoothstep(self.SHARE_AT, self.SHARE_AT + 0.04, u)
        self.text(sc, self.RIGHT - self.frac(9.0)[0], self.LOW_Y + 0.06,
                  "%d%% padding" % round(100 * m["padding_share"]), c, HEAD, "lm", strong=True)


ANIMATIONS = {"padding-a-batch-is-a-block": ABatchIsABlock,
              "padding-lines-of-text": PaddingLinesOfText}

"""Animation 271 of "Blocks and code" (see :mod:`voxanim.blocks`), "Resizing pictures": a
conv2d whose kernel is bigger than the picture, with padding, makes the picture bigger,
drawn from :mod:`voxanim.study_blocks_ze`. It is about the idea, not the program: there is
no code panel and no training-loop map (nothing is trained), and the matrix takes that
space.

:class:`BigKernel` (``resample-big-kernel``): photo 0 shrunk to 6 x 6 and a 9 x 9 bell
(its middle row 0.001 ... 0.075 ... 0.001, all 81 adding up to 1), bigger than the
picture; padding 8 puts the picture in a 22 x 22 frame of zeros (black: a pixel of 0) and
the kernel fits in 6 + 2 x 8 - 9 + 1 = 14 places a side, 196 output pixels; the middle one
worked out (all 36 pixels under the kernel, its other 45 numbers on zeros: 0.61); then the
matrix M (196 x 36), the brightest pixel's column (25) lifted out and reshaped to 14 x 14:
the kernel stamped where that pixel sits, 81 outputs from one pixel; the 36 stamps added
up, each times its pixel, building the output; V (36 x 196), the same kernel without
padding, shrinking 14 x 14 to 6 x 6, each row an average of 81, and M exactly V turned on
its side; each column of M adds up to 1, so the total stays 22.99.

The photo is drawn grey (0 black, 1 white; people by number only); the output is shaded on
its own scale, 0 black to its largest value (0.62) white, and says so; kernel and matrix
numbers are orange tiles.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .blocks import BIG, HEAD, INK, QUIET, SIZE, WEIGHT, exact
from .diagram import draw_image
from .easing import smoothstep
from .mech import AXIS, PAPER
from .mech_blocks_t import gray
from .mech_blocks_zb import DATA_INK, KERNEL_FILL, WEIGHT_INK, Resize, during, eased, on


def strip_tex(values, lo, hi):
    """A vector as a column of grey (``lo`` black, ``hi`` white): ``(n, 1, 3)``."""
    v = (np.ravel(np.asarray(values, np.float32)) - lo) / max(hi - lo, 1e-12)
    return gray(v[:, None])


class BigKernel(Resize):
    """The kernel; padding 8; one pixel; the matrix M; the stamps added up; the mirror of
    shrinking; the total kept."""

    TITLE = "Bigger: a kernel larger than the picture"
    PHASES = (("kernel", 8.5), ("padding 8", 9.5), ("one pixel", 7.0), ("matrix M", 10.0),
              ("stamps added", 9.0), ("mirror", 8.0), ("total kept", 9.0))
    LAST = "Bigger: spread, then blur"
    NEXT = "A CNN training step"
    STATION = None
    # phase 1: the picture, the kernel, its middle row
    PIC1, PX1 = (0.14, 0.6), 14.0
    KER1 = (0.34, 0.6)
    ROWB, RB = (0.72, 0.6), (48.0, 24.0)
    # phases 2-3: the frame of zeros and the output beside it
    FRAME, PX2 = (0.2, 0.64), 11.0
    OUT2 = (0.53, 0.64)
    WORDS_X = 0.65
    # phases 4-7: the picture, a column as 14 x 14, M @ x = out
    PIC4, PX4 = (0.12, 0.5), 14.0
    STAMP, SPX = (0.33, 0.645), 10.0
    MAT, MW, MH = (0.575, 0.615), 108.0, 245.0
    XS, OS, SW = (0.665, 0.615), (0.712, 0.615), 10.0
    OUT4, OPX = (0.845, 0.615), 10.0
    VM = (0.33, 0.5)

    def setup(self):
        m = self.meta = self.study.metadata
        self.x = np.asarray(m["x"], np.float64)
        self.k = np.asarray(m["kernel"], np.float64)
        self.M = np.asarray(m["M"], np.float64)
        self.V = np.asarray(m["V"], np.float64)
        self.out = np.asarray(m["out"], np.float64)
        self.padded = np.asarray(m["padded"], np.float64)
        self.omax = float(m["out_max"])
        # orange tiles: paper at 0, full orange at 0.6 of the kernel's largest number (so the
        # bell's lower slopes show)
        self.kmax = 0.6 * float(m["kernel_max"])
        self.n, self.side, self.pad = int(m["small"]), int(m["side"]), int(m["pad"])
        n = self.n * self.n
        # the output after the first j stamps, each times its pixel (j = 0 .. 36)
        xs = self.x.ravel()
        self.partial = [(self.M[:, :j] @ xs[:j]).reshape(self.side, self.side)
                        for j in range(n + 1)]
        self.EVIDENCE = (exact("the kernel, a binomial row times itself"),
                         exact("conv2d with padding 8 on photo 0's 6 × 6"),
                         exact("output pixel (7, 7) of photo 0's 6 × 6"),
                         exact("the matrix M of conv2d, 196 × 36"),
                         exact("M @ x against conv2d: 0 apart"),
                         exact("M against V turned on its side: 0 apart"),
                         exact("the totals of photo 0's 6 × 6 and its 14 × 14"))

    # ------------------------------------------------------------------ no code, no map
    def code_panel(self, sc, t, op):
        """No code: the matrix takes the panel's place."""

    def loop_map(self, sc, t, op, centre=None):
        """Nothing is trained: no loop map."""

    def step_indicator(self, sc, current, phases):
        """The phase under way by name, the others by number (beside the long title), as
        in :mod:`voxanim.mech_blocks_zd`."""
        items = ["%d %s" % (n + 1, name) if n == current else "%d" % (n + 1)
                 for n, name in enumerate(phases)]
        size, gap = 0.030, 0.016
        widths = [layout.text_width(s, size) for s in items]
        x = (1 - layout.SAFE) - (sum(widths) + gap * (len(items) - 1)) - 0.004
        for n, (s, w) in enumerate(zip(items, widths, strict=True)):
            ink = (0.12, 0.36, 0.66) if n == current else layout.INK_SOFT if n < current \
                else (0.56, 0.58, 0.60)
            sc.text(x, layout.SAFE + 0.012, s, size, "regular", ink, 1.0, "la")
            x += w + gap

    def heading_for(self, k, u):
        return ("a 9 × 9 kernel, bigger than the picture",
                "padding 8: a frame of zeros round it",
                "one output pixel: a weighted sum",
                "the same thing as a matrix M",
                "the output: the stamps added up",
                "the mirror of shrinking",
                "the total is kept, only spread out")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "Photo 0, shrunk to 6 × 6, and a 9 × 9 kernel shaped like a bell: the kernel is "
            "bigger than the picture.",
            "Padding 8 puts the picture in a frame of zeros, and the kernel fits %s places "
            "a side. The output has its own shading." % m["side"],
            "In the middle all 36 pixels lie under the kernel; its other 45 numbers land on "
            "zeros. Multiplied and added: %s." % m["worked_text"],
            "Flattened, conv2d is M @ x. Column %d of M is the kernel stamped where pixel %d "
            "sits: one pixel spreads into %d." % (m["column"], m["column"], m["per_column"]),
            "The output is the 36 stamps added up, each times its pixel: a smooth picture "
            "the 36 numbers never showed.",
            "Without padding, V shrinks 14 × 14 to 6 × 6, each pixel an average of 81. M is "
            "exactly V turned on its side.",
            "Each column of M adds up to 1, so the total stays %s: nothing new, only spread "
            "out." % m["total_out_text"],
        ][k]

    # ------------------------------------------------------------------ where things are
    def photo_at(self, k, u):
        """The 6 x 6 picture's centre and screen px per pixel."""
        if k == 0:
            return self.PIC1, self.PX1
        if k == 1:
            s = eased(u, 0.02, 0.14)
            a, b, pa, pb = self.PIC1, self.FRAME, self.PX1, self.PX2
        elif k == 2:
            return self.FRAME, self.PX2
        elif k == 3:
            s = eased(u, 0.02, 0.14)
            a, b, pa, pb = self.FRAME, self.PIC4, self.PX2, self.PX4
        else:
            return self.PIC4, self.PX4
        return (a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s), pa + (pb - pa) * s

    def grid(self, centre, px, side):
        """``cell(row, col)`` of a ``side x side`` picture on a data card at ``centre``."""
        return self.picture_cell((side, side), *centre, px)

    def photo_cell(self, k, u):
        (fx, fy), px = self.photo_at(k, u)
        return self.grid((fx, fy), px, self.n)

    def frame_cell(self):
        return self.grid(self.FRAME, self.PX2, self.n + 2 * self.pad)

    def out2_cell(self):
        return self.grid(self.OUT2, self.PX2, self.side)

    def mat_cell(self):
        return self.cells(*self.MAT, self.MW, self.MH, self.side ** 2, self.n ** 2)

    def mw(self):
        return self.MW / self.n ** 2, self.MH / self.side ** 2

    def scan(self, u):
        """Phase 2: how many output pixels are done (a float, 0..196): row 0 slowly, the
        other rows quicker."""
        if u < 0.26:
            return 0.0
        if u < 0.52:
            return self.side * eased(u, 0.26, 0.5)
        return self.side + (self.side ** 2 - self.side) * float(np.clip((u - 0.53) / 0.32,
                                                                       0.0, 1.0))

    def added(self, u):
        """Phase 5: how many stamps are added (0..36): three slowly, then the rest."""
        if u < 0.05:
            return 0
        if u < 0.4:
            return 1 + int(u >= 0.18) + int(u >= 0.3)
        return min(36, 3 + int(np.floor(33 * float(np.clip((u - 0.4) / 0.38, 0.0, 1.0)))))

    # ------------------------------------------------------------------ drawing helpers
    def image(self, sc, tex, centre, px, op, card=1.0, lift=0.2, grid=0.0):
        """A grey picture (``tex`` rows x cols x 3) on a data card (``card``: its opacity)."""
        rows, cols = tex.shape[:2]
        w, h = cols * px, rows * px
        fx, fy = centre
        if op <= 0.01:
            return
        if card > 0.01:
            self.card(sc, fx, fy, w + 12, h + 18, op * card, lift=lift)
        draw_image(self, sc, self.P(fx, fy + self.frac(0, 3.0)[1]), np.asarray(tex, np.float32),
                   w, h, op, grid=grid, lift=lift + 0.004)

    def out_tex(self, img, shown=None):
        tex = gray(np.asarray(img) / self.omax)
        if shown is not None:
            tex = np.where(np.asarray(shown, bool)[..., None], tex,
                           np.asarray(PAPER, np.float32))
        return tex

    def ring(self, sc, cell, r, c, rows, cols, w, h, op, ink=INK, width=2.2):
        self.frame(sc, cell, r, c, rows, cols, w, h, op, ink=ink, width=width)

    def matrix_m(self, sc, op, col=None, col_op=0.0):
        """M as orange tiles (196 x 36), its border, label, and column ``col`` outlined."""
        if op <= 0.01:
            return
        fx, fy = self.MAT
        self.rect(sc, self.P(fx, fy), self.MW + 8, self.MH + 8, PAPER, op, border=WEIGHT,
                  border_w=1.6, radius=3, lift=0.2)
        self.tiles(sc, self.M, fx, fy, self.MW, self.MH, op, ink=WEIGHT, lo=0.0, hi=self.kmax,
                   grid=0.0)
        self.text(sc, fx, fy - self.frac(0, self.MH / 2)[1] - 0.04, "M: 196 × 36", op, SIZE,
                  ink=WEIGHT_INK)
        if col is not None:
            cw, ch = self.mw()
            self.ring(sc, self.mat_cell(), 0, col, self.side ** 2, 1, cw, ch, op * col_op,
                      width=2.0)

    def stamp(self, sc, col, op, label, ring=True):
        """Column ``col`` of M reshaped to 14 x 14: orange tiles, the 9 x 9 it fills
        outlined."""
        if op <= 0.01:
            return
        s = self.SPX * self.side
        fx, fy = self.STAMP
        self.rect(sc, self.P(fx, fy), s + 8, s + 8, PAPER, op, border=WEIGHT, border_w=1.6,
                  radius=3, lift=0.2)
        cell = self.tiles(sc, self.M[:, col].reshape(self.side, self.side), fx, fy, s, s, op,
                          ink=WEIGHT, lo=0.0, hi=self.kmax, grid=0.2)
        if ring:
            i, j = divmod(col, self.n)
            self.ring(sc, cell, i, j, 9, 9, self.SPX, self.SPX, op, ink=WEIGHT_INK, width=1.8)
        self.text(sc, fx, fy - self.frac(0, s / 2)[1] - 0.035, label, op, SIZE,
                  ink=WEIGHT_INK)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        # the 6 x 6 picture: all through, moving into the frame and out again
        frame_on = (self.at(1, 0.16) if k == 1 else 1.0) if k in (1, 2) else \
            (self.leaving(3, k, u) if k == 3 else 0.0)
        centre, px = self.photo_at(k, u)
        self.image(sc, gray(self.x), centre, px, op, card=1.0 - frame_on, lift=0.26)
        lab_op = op * (1.0 if k == 0 else (1.0 - smoothstep(0.0, 0.03, u)) if k == 1 else
                       0.0 if k == 2 else self.at(3, 0.14) if k == 3 else 1.0)
        if lab_op > 0.01:
            # over the picture (in phase 1 level with the kernel's label, clear of the
            # kernel's outline laid over the picture)
            top = centre[1] - self.frac(0, px * self.n / 2 + 9)[1] - (0.058 if k == 0 else 0.03)
            self.text(sc, centre[0], top, "photo 0, 6 × 6", lab_op, SIZE, ink=DATA_INK)
        # the frame of zeros: phases 2-3
        fa = op * frame_on
        if fa > 0.01:
            self.image(sc, gray(self.padded), self.FRAME, self.PX2, fa, lift=0.2)
            side = self.PX2 * (self.n + 2 * self.pad)
            self.text(sc, self.FRAME[0], self.FRAME[1] - self.frac(0, side / 2 + 9)[1] - 0.025,
                      "22 × 22: zeros round photo 0", fa, SIZE, ink=DATA_INK)
            self.ring(sc, self.frame_cell(), self.pad, self.pad, self.n, self.n, self.PX2,
                      self.PX2, fa * (self.at(1, 0.2) if k == 1 else 1.0), ink=DATA_INK,
                      width=1.6)
        # the output beside it: phases 2-3
        oa = op * self.kept(1, 2) * (self.at(1, 0.22) if k == 1 else 1.0) * \
            self.leaving(3, k, u)
        if oa > 0.01:
            shown = np.ones((self.side, self.side), bool)
            if k == 1:
                shown[:] = False
                shown.reshape(-1)[:int(np.ceil(self.scan(u)))] = True
            self.image(sc, self.out_tex(self.out, shown), self.OUT2, self.PX2, oa)
            s = self.PX2 * self.side
            self.text(sc, self.OUT2[0], self.OUT2[1] - self.frac(0, s / 2 + 9)[1] - 0.025,
                      "14 × 14", oa, SIZE, ink=DATA_INK)
            self.text(sc, self.OUT2[0], self.OUT2[1] + self.frac(0, s / 2 + 9)[1] + 0.03,
                      "shaded 0 to %s" % m["out_max_text"], oa, SIZE, ink=QUIET)
            self.arrow(sc, (self.FRAME[0] + 0.14, self.FRAME[1]),
                       (self.OUT2[0] - 0.093, self.OUT2[1]), AXIS, oa, width=2.0)
        # M and x: phases 4-7
        ma = op * self.kept(3) * (self.at(3, 0.16) if k == 3 else 1.0)
        if ma > 0.01:
            col, col_op = None, 0.0
            if k == 3:
                col, col_op = m["column"], self.at(3, 0.42)
            elif k == 4:
                j = self.added(u)
                col, col_op = (m["column"], 1.0 - self.at(4, 0.03)) if j == 0 else \
                    (j - 1, during(u, 0.05, 0.84, 1.0, 0.02))
            elif k == 5:
                col, col_op = m["column"], self.at(5, 0.3)
            self.matrix_m(sc, ma, col, col_op)
            xa = op * self.kept(3) * (self.at(3, 0.24) if k == 3 else 1.0)
            xh = 3.0 * self.n ** 2
            draw_image(self, sc, self.P(*self.XS), strip_tex(self.x, 0.0, 1.0), self.SW, xh,
                       xa, grid=0.0, lift=0.3)
            self.rect(sc, self.P(*self.XS), self.SW + 4, xh + 4, DATA_INK, xa,
                      border=DATA_INK, border_w=1.4, radius=2, lift=0.32, fill_op=0.0)
            self.text(sc, self.XS[0], self.XS[1] - self.frac(0, xh / 2)[1] - 0.03, "x: 36", xa,
                      SIZE, ink=DATA_INK)
            self.text(sc, (self.MAT[0] + self.XS[0]) / 2 + 0.017, self.MAT[1], "@", xa, HEAD,
                      strong=True)
            self.text(sc, (self.XS[0] + self.OS[0]) / 2, self.MAT[1], "=", xa, HEAD,
                      strong=True)
            if col is not None and col_op > 0.01:
                xc = self.cells(*self.XS, self.SW, xh, self.n ** 2, 1)
                self.ring(sc, xc, col, 0, 1, 1, self.SW, 3.0, xa * col_op, width=1.8)
            # the output's slot: empty in phase 4, filled from phase 5
            j = 0 if k == 3 else (self.added(u) if k == 4 else self.n ** 2)
            img = self.partial[j]
            self.rect(sc, self.P(*self.OS), self.SW + 4, self.MH + 4, DATA_INK, xa,
                      border=DATA_INK, border_w=1.4, radius=2, lift=0.32, fill_op=0.0)
            self.text(sc, self.OS[0], self.MAT[1] - self.frac(0, self.MH / 2)[1] - 0.04, "196",
                      xa, SIZE, ink=DATA_INK)
            if k >= 4:
                draw_image(self, sc, self.P(*self.OS), strip_tex(img, 0.0, self.omax), self.SW,
                           self.MH, xa, grid=0.0, lift=0.3)
        # the output, 14 x 14: phases 5-7
        pa = op * self.kept(4) * (self.at(4, 0.03) if k == 4 else 1.0)
        if pa > 0.01:
            j = self.added(u) if k == 4 else self.n ** 2
            self.image(sc, self.out_tex(self.partial[j]), self.OUT4, self.OPX, pa)
            s = self.OPX * self.side
            self.text(sc, self.OUT4[0], self.OUT4[1] - self.frac(0, s / 2 + 9)[1] - 0.03,
                      "out: 14 × 14", pa, SIZE, ink=DATA_INK)
            self.text(sc, self.OUT4[0], self.OUT4[1] + self.frac(0, s / 2 + 9)[1] + 0.028,
                      "shaded 0 to %s" % m["out_max_text"], pa, SIZE, ink=QUIET)
        # a column of M as 14 x 14: phases 4-5
        sa = op * self.kept(3, 4) * (self.at(3, 0.55) if k == 3 else 1.0) * \
            self.leaving(5, k, u)
        if sa > 0.01:
            if k == 4 and self.added(u) > 0:
                self.stamp(sc, self.added(u) - 1, sa, "each column, as 14 × 14")
            else:
                self.stamp(sc, m["column"], sa, "column %d, as 14 × 14" % m["column"])

    # ------------------------------------------------------------------ 1: the kernel
    def phase_1(self, sc, u, op):
        m = self.meta
        ka = on(u, 0.08, op)
        s = self.PX1 * 9
        kc = self.tiles(sc, self.k, *self.KER1, s, s, ka, ink=WEIGHT, lo=0.0, hi=self.kmax,
                        grid=0.25)
        top = self.KER1[1] - self.frac(0, s / 2)[1] - 0.035
        self.text(sc, self.KER1[0], top, "kernel, 9 × 9", ka, SIZE, ink=WEIGHT_INK)
        # its middle row, and the numbers in it
        ra = on(u, 0.3, op)
        self.ring(sc, kc, 4, 0, 1, 9, self.PX1, self.PX1, ra, width=2.0)
        bw, bh = self.RB
        cell = self.boxes(sc, [m["middle_row_text"]], *self.ROWB, bw, bh, ra, fill=KERNEL_FILL,
                          border=WEIGHT)
        self.text(sc, self.ROWB[0], self.ROWB[1] - 0.06, "its middle row", ra, SIZE,
                  ink=WEIGHT_INK)
        src = self.corners(kc, 4, 0, 1, 9, self.PX1, self.PX1)
        dst = self.corners(cell, 0, 0, 1, 9, bw, bh)
        self.guide(sc, (src[2], src[1]), (dst[0], dst[1]), AXIS, ra * 0.8)
        self.guide(sc, (src[2], src[3]), (dst[0], dst[3]), AXIS, ra * 0.8)
        # bigger than the picture: the kernel's outline laid over it
        fa = on(u, 0.5, op)
        pc = self.photo_cell(0, u)
        self.frame(sc, pc, -1.5, -1.5, 9, 9, self.PX1, self.PX1, fa, ink=WEIGHT, width=2.4)
        self.text(sc, self.PIC1[0], 0.8, "it does not fit inside", fa, SIZE, ink=INK)
        ta = on(u, 0.68, op)
        self.text(sc, 0.52, 0.76, "all 81 add up to", ta, SIZE, "lm")
        self.text(sc, 0.655, 0.76, "%g" % m["kernel_sum"], ta, BIG, "lm", ink=INK)

    # ------------------------------------------------------------------ 2: padding 8
    def phase_2(self, sc, u, op):
        m = self.meta
        pos = self.scan(u)
        r, c = divmod(min(int(pos), self.side ** 2 - 1), self.side)
        a = during(u, 0.26, 0.93, op)
        self.frame(sc, self.frame_cell(), r, c, 9, 9, self.PX2, self.PX2, a, ink=WEIGHT,
                   width=2.6)
        self.frame(sc, self.out2_cell(), r, c, 1, 1, self.PX2, self.PX2, a, ink=INK, width=2.0)
        e = on(u, 0.32, op)
        x = self.WORDS_X
        self.text(sc, x, 0.5, "places a side", e, SIZE, "lm")
        self.text(sc, x, 0.555, m["equation"], e, HEAD, "lm", strong=True)
        done = int(np.ceil(pos)) if u > 0.26 else 0
        self.text(sc, x, 0.68, "output pixels", on(u, 0.26, op), SIZE, "lm")
        self.text(sc, x, 0.735, "%d" % done, on(u, 0.26, op), BIG, "lm", ink=INK)

    # ------------------------------------------------------------------ 3: one pixel
    def phase_3(self, sc, u, op):
        m = self.meta
        i, j = m["worked_at"]
        wa = on(u, 0.06, op)
        self.frame(sc, self.frame_cell(), i, j, 9, 9, self.PX2, self.PX2, wa, ink=WEIGHT,
                   width=3.0)
        pa = on(u, 0.2, op)
        self.mark(sc, self.corners(self.frame_cell(), self.pad, self.pad, self.n, self.n,
                                   self.PX2, self.PX2), pa)
        x = self.WORDS_X
        self.text(sc, x, 0.5, "%d pixels × kernel numbers" % m["under"], pa, SIZE, "lm")
        self.text(sc, x, 0.55, "%d kernel numbers × 0" % m["on_zeros"], on(u, 0.32, op), SIZE,
                  "lm", ink=QUIET)
        ad = on(u, 0.46, op)
        self.text(sc, x, 0.63, "added up", ad, SIZE, "lm")
        self.text(sc, x, 0.69, m["worked_text"], ad, BIG, "lm", ink=INK)
        ra = on(u, 0.56, op)
        box = self.corners(self.out2_cell(), i, j, 1, 1, self.PX2, self.PX2)
        self.mark(sc, box, ra)
        oy = (box[1] + box[3]) / 2
        self.guide(sc, (x - 0.006, 0.69), (box[2] + 0.004, oy), INK, ra, width=1.8)

    # ------------------------------------------------------------------ 4: the matrix M
    def phase_4(self, sc, u, op):
        m = self.meta
        i, j = m["column_at"]
        ca = on(u, 0.42, op)
        self.ring(sc, self.photo_cell(3, u), i, j, 1, 1, self.PX4, self.PX4, ca, width=2.4)
        self.text(sc, self.PIC4[0], self.PIC4[1] + 0.12, "pixel %d: %s" % (
            m["column"], m["column_value_text"]), ca, SIZE, ink=INK)
        # the column lifted out
        sa = on(u, 0.55, op)
        cx = self.mat_cell()(0, m["column"])[0]
        s = self.SPX * self.side
        self.guide(sc, (cx, self.STAMP[1]), (self.STAMP[0] + self.frac(s / 2 + 4)[0],
                                             self.STAMP[1]), AXIS, sa * 0.8)
        self.text(sc, self.STAMP[0], self.STAMP[1] + self.frac(0, s / 2)[1] + 0.035,
                  "%d outputs from one pixel" % m["per_column"], on(u, 0.66, op), SIZE, ink=INK)

    # ------------------------------------------------------------------ 5: stamps added
    def phase_5(self, sc, u, op):
        j = self.added(u)
        a = op if j > 0 else 0.0
        ring = during(u, 0.05, 0.84, op, 0.02)
        if j > 0:
            i, c = divmod(j - 1, self.n)
            self.ring(sc, self.photo_cell(4, u), i, c, 1, 1, self.PX4, self.PX4, ring, width=2.4)
        self.text(sc, 0.06, 0.7, "stamps added", a, SIZE, "lm")
        self.text(sc, 0.06, 0.755, "%d of 36" % j, a, BIG, "lm", ink=INK)
        s = self.SPX * self.side
        below = self.STAMP[1] + self.frac(0, s / 2)[1] + 0.035
        if j > 0:
            self.running(sc, self.STAMP[0], below,
                         ((0.0, "× pixel 0: %.2f" % self.x.ravel()[0]),
                          (0.18, "× pixel 1: %.2f" % self.x.ravel()[1]),
                          (0.3, "× pixel 2: %.2f" % self.x.ravel()[2]),
                          (0.4, "each × its pixel")), u, op, size=SIZE, anchor="mm", ink=INK)
        self.text(sc, 0.06, 0.86, "smooth values the 36 never showed", on(u, 0.84, op), SIZE,
                  "lm", ink=INK)

    # ------------------------------------------------------------------ 6: the mirror
    def phase_6(self, sc, u, op):
        m = self.meta
        va = on(u, 0.06, op)
        w, h = 1.25 * self.side ** 2, 3.0 * self.n ** 2
        self.rect(sc, self.P(*self.VM), w + 8, h + 8, PAPER, va, border=WEIGHT, border_w=1.6,
                  radius=3, lift=0.2)
        vc = self.tiles(sc, self.V, *self.VM, w, h, va, ink=WEIGHT, lo=0.0, hi=self.kmax,
                        grid=0.0)
        self.text(sc, self.VM[0], self.VM[1] - self.frac(0, h / 2)[1] - 0.035,
                  "V: 36 × 196, no padding", va, SIZE, ink=WEIGHT_INK)
        ra = on(u, 0.3, op)
        self.ring(sc, vc, m["column"], 0, 1, self.side ** 2, 1.25, 3.0, ra, width=2.0)
        self.text(sc, 0.06, 0.68, "each row: an average of 81", on(u, 0.16, op), SIZE, "lm")
        self.text(sc, 0.06, 0.73, "row %d of V = column %d of M" % (m["column"], m["column"]),
                  ra, SIZE, "lm")
        ea = on(u, 0.5, op)
        self.text(sc, 0.06, 0.81, "M = V turned on its side", ea, HEAD, "lm", strong=True)
        self.text(sc, 0.06, 0.865, "%g apart" % m["transpose"], ea, SIZE, "lm", ink=QUIET)

    # ------------------------------------------------------------------ 7: the total kept
    def phase_7(self, sc, u, op):
        m = self.meta
        a = on(u, 0.05, op)
        self.text(sc, self.PIC4[0], 0.65, "total", a, SIZE, ink=DATA_INK)
        self.text(sc, self.PIC4[0], 0.705, m["total_in_text"], a, BIG, ink=INK)
        b = on(u, 0.15, op)
        self.text(sc, self.OUT4[0] - 0.035, 0.85, "total", b, SIZE, "rm", ink=DATA_INK)
        self.text(sc, self.OUT4[0] - 0.025, 0.85, m["total_out_text"], b, BIG, "lm", ink=INK)
        self.text(sc, 0.06, 0.79, "each column of M adds up to 1", on(u, 0.25, op), SIZE, "lm",
                  ink=WEIGHT_INK)
        self.text(sc, 0.06, 0.845, "nothing new, only spread out", on(u, 0.35, op), SIZE,
                  "lm", ink=INK)


ANIMATIONS = {"resample-big-kernel": BigKernel}

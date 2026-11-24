"""Animations 263-264 of "Blocks and code" (see :mod:`voxanim.blocks`), "Resizing pictures":
making a picture smaller and bigger with fixed convolution kernels (``examples/resample.py``),
drawn from :mod:`voxanim.study_blocks_zb`. Nothing is learned, so the loop map lights no
station.

* :class:`Smaller` (``resample-down``): the kernel built as a multiplication table, the row
  ``[1, 2, 1] / 4`` times the same column, 9 sixteenths adding up to 16/16 = 1; the kernel on
  photo 0's 9 pixels at an eye's edge (picked by a rule), multiplied and added: 0.53, one
  pixel of the 32 x 32 picture; then the kernel jumping 2 pixels at a time until 1,024 pixels
  are filled; the matrix D, row by row (the kernel moved 2 places a row; row 0's first
  number falls past the edge), and all 32 x 64 of it; D @ plate (32 x 64) then @ D.T (32 x 32),
  the same numbers as conv2d (torch prints 1.7881e-07: rounding); then the zone plate:
  where its rings are too fine for 32 x 32 (outside the circle) skipping leaves false rings
  of RMS 0.354, blurring first 0.079.
* :class:`Bigger` (``resample-up``): photo 0's 32 x 32 picture spread to 64 x 64 with a zero
  between its pixels (a close-up of numbers); the kernel x 4 fills each gap: between two,
  (0.64 + 0.53) / 2 = 0.59; among four, their average, 0.54; the whole picture turns smooth,
  bilinear. Nearest copies each pixel into a 2 x 2 block. A flat picture of ones comes back
  1 inside, 0.5 on the last row and column, 0.25 at the corner, so the photo's last row is
  dim (0.28 on average) until ``big / got`` (0.57). Over the 400 photos, down then up, the
  mean squared difference: inside the border bilinear 0.00129, nearest 0.00302; the whole
  picture 0.00343, 0.00199 with the edge made even; and skip then bilinear 0.00092, closer
  still: blur buys no false rings (0.354 against 0.079 on the zone plate), not a closer copy.

Photos are drawn as the grey photographs they are, on the blue data card (people by number
only); kernel and matrix numbers are orange number tiles, pixel numbers blue ones, zeros grey.
"""

from __future__ import annotations

import numpy as np

from .blocks import (
    BIG,
    DATA,
    HEAD,
    INK,
    LOSS,
    QUIET,
    SIZE,
    WEIGHT,
    ZERO,
    Blocks,
    Code,
    exact,
    measured,
)
from .diagram import draw_image
from .easing import ease_in_out, smoothstep
from .mech import AXIS, PAPER, dark, soft
from .mech_blocks_t import gray
from .study_blocks_zb import ON

DATA_INK = dark(DATA, 0.3)
WEIGHT_INK = dark(WEIGHT, 0.35)
LOSS_INK = dark(LOSS, 0.3)
PIXEL_FILL = soft(DATA, 0.72)
KERNEL_FILL = soft(WEIGHT, 0.6)
ZERO_FILL = soft(ZERO, 0.45)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on(u, a, op, width=0.05):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


def during(u, a, b, op, width=0.05):
    """``op`` from ``u = a`` to ``b``, faded in and out."""
    return op * smoothstep(a, a + width, u) * (1.0 - smoothstep(b - width, b, u))


# 264, phases 5-6: the height of each bar
BAR_ROWS = dict(inner_bilinear=0.6, inner_nearest=0.645, inner_skip=0.69, full_bilinear=0.79,
                full_even=0.835)


class Resize(Blocks):
    """What the two share: pictures on data cards, grids of number tiles, frames and guides
    in front."""

    STATION = None                  # nothing is trained

    def at(self, k, u):
        """1 once phase ``k`` has reached ``u`` (fading in over half a second); 0 before."""
        a, b = self.spans[k]
        t0 = a + (b - a) * u
        return smoothstep(t0, t0 + 0.5, self._t)

    @staticmethod
    def leaving(phase, k, u, width=0.06):
        """For :meth:`steady`: 1, fading out over the first ``width`` of ``phase`` (so a shape
        stays on screen while the next step fades in), 0 after."""
        if k < phase:
            return 1.0
        return 1.0 - smoothstep(0.0, width, u) if k == phase else 0.0

    # ------------------------------------------------------------------ pictures
    def picture_cell(self, shape, fx, fy, px):
        rows, cols = shape
        return self.cells(fx, fy + self.frac(0, 3.0)[1], cols * px, rows * px, rows, cols)

    def picture(self, sc, img, fx, fy, px, op, shown=None, lift=0.2):
        """A grey picture (0 dark, 1 light), ``px`` screen px per pixel, on a data card;
        ``shown``: a mask of the pixels filled so far (the rest are paper). Returns
        ``cell(row, col)``."""
        img = np.asarray(img, np.float32)
        rows, cols = img.shape
        cell = self.picture_cell(img.shape, fx, fy, px)
        if op <= 0.01:
            return cell
        w, h = cols * px, rows * px
        self.card(sc, fx, fy, w + 12, h + 18, op, lift=lift)
        tex = gray(img)
        if shown is not None:
            tex = np.where(np.asarray(shown, bool)[..., None], tex,
                           np.asarray(PAPER, np.float32))
        draw_image(self, sc, self.P(fx, fy + self.frac(0, 3.0)[1]), tex.astype(np.float32),
                   w, h, op, grid=0.0, lift=lift + 0.004)
        return cell

    def boxes(self, sc, texts, fx, fy, bw, bh, op, fill=PIXEL_FILL, border=DATA, zero=None,
              each=None, ring=(), ring_ink=INK):
        """A grid of numbers in tiles ``bw x bh`` px centred at ``(fx, fy)``; ``zero``: a
        mask of grey tiles (a 0); ``each``: per-tile opacity; ``ring``: tiles outlined.
        Returns ``cell(row, col)``."""
        rows, cols = len(texts), len(texts[0])
        cell = self.cells(fx, fy, cols * bw, rows * bh, rows, cols)
        if op <= 0.01:
            return cell
        for i in range(rows):
            for j in range(cols):
                a = op * (1.0 if each is None else float(each[i][j]))
                if a <= 0.01:
                    continue
                x, y = cell(i, j)
                z = zero is not None and bool(zero[i][j])
                self.box(sc, x, y, bw - 3, bh - 3, ZERO_FILL if z else fill, a,
                         text=texts[i][j], border=AXIS if z else border, radius=3, size=SIZE,
                         ink=INK)
        for i, j in ring:
            x, y = cell(i, j)
            self.rect(sc, self.P(x, y), bw + 2, bh + 2, ring_ink, op, border=ring_ink,
                      border_w=2.4, radius=4, lift=1.2, fill_op=0.0)
        return cell

    def frame(self, sc, cell, r, c, rows, cols, w, h, op, ink=WEIGHT, width=2.4, lift=1.1):
        """An outline round ``rows x cols`` cells from ``(r, c)`` of a grid whose cells are
        ``w x h`` px (``r``, ``c`` may lie past its edge)."""
        if op <= 0.01:
            return
        (x0, y0), (x1, y1) = cell(r, c), cell(r + rows - 1, c + cols - 1)
        self.rect(sc, self.P((x0 + x1) / 2, (y0 + y1) / 2), cols * w + 4, rows * h + 4, ink,
                  op, border=ink, border_w=width, radius=2, lift=lift, fill_op=0.0)

    def corners(self, cell, r, c, rows, cols, w, h):
        """Layout ``(left, top, right, bottom)`` of those cells."""
        (x0, y0), (x1, y1) = cell(r, c), cell(r + rows - 1, c + cols - 1)
        dx, dy = self.frac(w / 2, h / 2)
        return x0 - dx, y0 - dy, x1 + dx, y1 + dy

    def guide(self, sc, a, b, ink, op, width=1.3):
        """A straight guide line in front of the blocks."""
        if op <= 0.01:
            return
        lift = self.ARROW_LIFT
        self.polyline(sc, [self.P(a[0], a[1], lift), self.P(b[0], b[1], lift)], width, ink,
                      op, bias=0.5)

    def zoom_guides(self, sc, src, dst, op):
        """Two guides from a small outline's bottom corners to a close-up's top corners."""
        self.guide(sc, (src[0], src[3]), (dst[0], dst[1]), AXIS, op * 0.8)
        self.guide(sc, (src[2], src[3]), (dst[2], dst[1]), AXIS, op * 0.8)

    def mark(self, sc, box, op):
        """An outline round layout ``box`` (left, top, right, bottom) that shows on dark and
        light pixels alike: a pale band under a dark line, in front."""
        if op <= 0.01:
            return
        w, h = self.span(box[2] - box[0], box[3] - box[1])
        c = self.P((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        self.rect(sc, c, w + 7, h + 7, PAPER, op, border=PAPER, border_w=4.0, radius=3,
                  lift=1.05, fill_op=0.0)
        self.rect(sc, c, w + 5, h + 5, INK, op, border=INK, border_w=1.8, radius=2, lift=1.1,
                  fill_op=0.0)

    def circle(self, sc, fx, fy, r, ink, op, width=3.0):
        """A circle of radius ``r`` px round layout ``(fx, fy)``, in front."""
        if op <= 0.01:
            return
        pts = []
        for a in np.linspace(0.0, 2 * np.pi, 73):
            dx, dy = self.frac(r * np.cos(a), r * np.sin(a))
            pts.append(self.P(fx + dx, fy + dy, self.ARROW_LIFT))
        halo = [p - self.view * 0.02 for p in pts]
        self.polyline(sc, halo, width + 3.0, PAPER, op, bias=0.45)
        self.polyline(sc, pts, width, ink, op, bias=0.5)


# ====================================================================== 263
class Smaller(Resize):
    """The kernel; one pixel worked out on photo 0; every second place; the matrix D;
    D @ plate @ D.T; false rings on the zone plate."""

    TITLE = "Smaller: blur, then skip"
    PHASES = (("kernel", 9.0), ("one pixel", 11.0), ("every 2nd", 8.0), ("matrix D", 11.0),
              ("D X D.T", 9.0), ("false rings", 13.0))
    LAST = "Edge detection with Sobel filters"
    NEXT = "Bigger: spread, then blur"
    # phase 1: the multiplication table
    KX, KY, KB = (0.225, 0.285, 0.345), (0.6, 0.665, 0.73), (54.0, 31.0)
    ROW_Y, COL_X = 0.5, 0.135
    # phases 2-3: the photo, the small picture, and the band of numbers under them
    PH, PPX = (0.14, 0.53), 2.0
    SM, SPX = (0.37, 0.53), 2.0
    LABEL_Y = 0.372                 # the pictures' labels, over them
    BAND, BB = 0.81, (46.0, 25.0)
    PATCH_X, KER_X, RES_X = 0.135, 0.305, 0.385
    # phase 4: D's corner and the whole of D
    DC, DB = (0.295, 0.53), (38.0, 27.0)
    DFULL, DPX = (0.285, 0.79), 2.6
    # phases 5-6: the zone plate and what becomes of it
    PLATE, LPX = (0.11, 0.52), 1.6
    DX_AT, FIN_AT, CONV_AT = (0.265, 0.52), (0.42, 0.52), (0.42, 0.77)
    SKIP_AT, BLUR_AT, BPX = (0.265, 0.52), (0.415, 0.52), 3.2

    def setup(self):
        m = self.meta = self.study.metadata
        self.photo = np.asarray(m["photo"], np.float32)
        self.small = np.asarray(m["small"], np.float32)
        self.D = np.asarray(m["D"], np.float32)
        self.plate = np.asarray(m["plate"], np.float32)
        i, j = m["at"]
        self.CODE = (
            Code("rs-kernel", run=((0.0, 0), (0.3, 1), (0.72, 2))),
            Code("rs-down", run=((0.0, 0), (0.12, 1))),
            Code("rs-down", run=((0.0, 1), (0.08, 2))),
            Code("rs-matrix", run=((0.0, 0), (0.1, 1), (0.2, 2), (0.3, 3), (0.42, 4),
                                   (0.52, 5))),
            Code("rs-same", run=((0.0, 0), (0.6, 1)), out=((0.66, m["printed_same"]),)),
            Code("rs-down", run=((0.0, 3), (0.42, 1))))
        self.EVIDENCE = (exact("the kernel, [1, 2, 1] / 4 times itself"),
                         exact("photo 0's 9 pixels at (%d, %d)" % (2 * i, 2 * j)),
                         exact("conv2d with stride 2 on photo 0"),
                         exact("the matrix D, 32 × 64"),
                         exact("D @ plate @ D.T against conv2d, float32"),
                         exact("the zone plate, %d pixels too fine" % m["fine_pixels"]))

    def heading_for(self, k, u):
        return ("a 3 × 3 kernel that adds up to 1",
                "one new pixel: a weighted average of 9",
                "stride 2: the kernel jumps 2 pixels",
                "the same thing as a matrix D",
                "down, then across: D @ plate @ D.T",
                "skip alone makes false rings")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "[1, 2, 1] / 4 times itself gives 9 sixteenths. They add up to 1, so a flat "
            "grey stays the same grey.",
            "Photo 0's 9 pixels times the kernel's 9 numbers, added up: %s, one pixel of the "
            "smaller picture." % m["output_text"],
            "The kernel jumps 2 pixels at a time, so 64 × 64 pixels become 32 × 32 = %s."
            % m["outputs_text"],
            "Each row of D is the kernel moved 2 places on. Row 0's first number falls past "
            "the edge.",
            "D shrinks the plate down, D.T across: conv2d's numbers, 1.8e-7 apart, which is "
            "float32 rounding.",
            "Outside the circle the rings are too fine for 32 × 32: a true copy is plain "
            "grey. Skip leaves %s, blur first %s." % (m["rms_skip_text"], m["rms_blur_text"]),
        ][k]

    # ------------------------------------------------------------------ where things are
    def photo_cell(self):
        return self.picture_cell((64, 64), *self.PH, self.PPX)

    def small_cell(self):
        return self.picture_cell((32, 32), *self.SM, self.SPX)

    def scan(self, u):
        """Phase 3: how many output pixels are done (a float, 0..1,024): row 0 slowly, the
        other rows quicker."""
        if u < 0.06:
            return 0.0
        if u < 0.4:
            return 32 * eased(u, 0.06, 0.38)
        return 32 + (1024 - 32) * float(np.clip((u - 0.42) / 0.4, 0.0, 1.0))

    def filled(self, k, u):
        shown = np.zeros((32, 32), bool)
        if k >= 3:
            shown[:] = True
        if k == 2:
            shown.reshape(-1)[:int(np.ceil(self.scan(u)))] = True
        if k == 2 or (k == 1 and u >= 0.74):
            shown[tuple(self.meta["at"])] = True
        return shown

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        # the kernel: the table of phase 1 moving down to the band of phases 2-3
        ka = op * self.kept(0, 3) * self.leaving(3, k, u)
        if ka > 0.01:
            s = 0.0 if k == 0 else (eased(u, 0.0, 0.12) if k == 1 else 1.0)
            cx = self.KX[1] + (self.KER_X - self.KX[1]) * s
            cy = self.KY[1] + (self.BAND - self.KY[1]) * s
            bw = self.KB[0] + (self.BB[0] - self.KB[0]) * s
            bh = self.KB[1] + (self.BB[1] - self.KB[1]) * s
            each = None
            if k == 0:
                each = [[smoothstep(0.3 + 0.035 * (3 * i + j), 0.34 + 0.035 * (3 * i + j), u)
                         for j in range(3)] for i in range(3)]
            self.boxes(sc, m["kernel_text"], cx, cy, bw, bh, ka, fill=KERNEL_FILL,
                       border=WEIGHT, each=each)
            lab = ka * (smoothstep(0.13, 0.18, u) if k == 1 else (1.0 if k == 2 else 0.0))
            self.text(sc, self.KER_X, self.BAND - 0.085, "kernel", lab, SIZE, ink=WEIGHT_INK)
        # the photo and the small picture: phases 2-3
        pa = op * self.kept(1, 3) * self.leaving(3, k, u)
        if pa > 0.01:
            self.picture(sc, self.photo, *self.PH, self.PPX, pa)
            self.text(sc, self.PH[0], self.LABEL_Y, "photo 0, 64 × 64", pa, SIZE,
                      ink=DATA_INK)
            self.picture(sc, self.small, *self.SM, self.SPX, pa, shown=self.filled(k, u))
            self.text(sc, self.SM[0], self.LABEL_Y, "32 × 32", pa, SIZE, ink=DATA_INK)
            self.arrow(sc, (self.PH[0] + 0.085, self.PH[1]), (self.SM[0] - 0.045, self.SM[1]),
                       AXIS, pa, width=2.0)
        # all of D: from late in phase 4 into the start of phase 5
        if k in (3, 4):
            self.full_d(sc, op * (self.at(3, 0.76) if k == 3 else 1.0) * self.leaving(4, k, u,
                                                                                       0.1))
        # the zone plate: phases 5-6
        za = op * self.kept(4, 5)
        if za > 0.01:
            self.picture(sc, self.plate, *self.PLATE, self.LPX, za)
            self.text(sc, self.PLATE[0], self.PLATE[1] + 0.125, "zone plate", za, SIZE,
                      ink=DATA_INK)

    # ------------------------------------------------------------------ 1: the kernel
    def phase_1(self, sc, u, op):
        m = self.meta
        row = on(u, 0.06, op)
        col = on(u, 0.16, op)
        for j, x in enumerate(self.KX):
            self.box(sc, x, self.ROW_Y, self.KB[0] - 3, self.KB[1] - 3, KERNEL_FILL, row,
                     text=m["row_text"][j], border=WEIGHT, radius=3, size=SIZE, ink=INK)
        for i, y in enumerate(self.KY):
            self.box(sc, self.COL_X, y, self.KB[0] - 3, self.KB[1] - 3, KERNEL_FILL, col,
                     text=m["row_text"][i], border=WEIGHT, radius=3, size=SIZE, ink=INK)
        self.text(sc, self.KX[1], self.ROW_Y - 0.06, "[1, 2, 1] / 4", row, SIZE,
                  ink=WEIGHT_INK)
        self.text(sc, self.COL_X, self.ROW_Y, "×", col, HEAD, strong=True)
        # the first tile of the table: its row and column ringed
        ring = during(u, 0.3, 0.5, op)
        for fx, fy in ((self.KX[0], self.ROW_Y), (self.COL_X, self.KY[0]),
                       (self.KX[0], self.KY[0])):
            self.rect(sc, self.P(fx, fy), self.KB[0] + 1, self.KB[1] + 1, INK, ring, border=INK,
                      border_w=2.4, radius=4, lift=1.2, fill_op=0.0)
        tot = on(u, 0.7, op)
        self.text(sc, 0.06, 0.835, "adds up to", tot, SIZE, "lm")
        self.text(sc, 0.175, 0.835, "%d/16 = %g" % (m["total16"], m["total"]), tot, BIG, "lm",
                  ink=INK)

    # ------------------------------------------------------------------ 2: one pixel
    def phase_2(self, sc, u, op):
        m = self.meta
        i, j = m["at"]
        r0, c0 = 2 * i - 1, 2 * j - 1
        pc = self.photo_cell()
        ring = on(u, 0.06, op)
        self.frame(sc, pc, r0, c0, 3, 3, self.PPX, self.PPX, ring, width=2.6)
        bw, bh = self.BB
        grow = on(u, 0.16, op)
        cell = self.boxes(sc, m["patch_text"], self.PATCH_X, self.BAND, bw, bh, grow)
        src = self.corners(pc, r0, c0, 3, 3, self.PPX, self.PPX)
        dst = self.corners(cell, 0, 0, 3, 3, bw, bh)
        self.zoom_guides(sc, src, dst, grow)
        self.text(sc, self.PATCH_X, self.BAND - 0.085, "9 pixels", grow, SIZE, ink=DATA_INK)
        times = on(u, 0.32, op)
        self.text(sc, (self.PATCH_X + self.KER_X) / 2, self.BAND, "×", times, HEAD,
                  strong=True)
        add = on(u, 0.46, op)
        self.text(sc, self.RES_X, self.BAND, "=", add, HEAD, "lm", strong=True)
        # the answer, then it flies to its pixel of the small picture
        s = eased(u, 0.62, 0.74)
        x0, y0 = self.RES_X + 0.05, self.BAND
        if u < 0.62:
            self.text(sc, x0, y0, m["output_text"], add, HEAD, strong=True)
        else:
            self.text(sc, x0, y0, m["output_text"], op, HEAD, strong=True)
            x1, y1 = self.small_cell()(i, j)
            if u < 0.75:
                self.box(sc, x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, 12, 12,
                         tuple(float(v) for v in gray(np.array([[m["output"]]]))[0, 0]), op,
                         border=INK, radius=2)
        self.frame(sc, self.small_cell(), i, j, 1, 1, self.SPX, self.SPX, on(u, 0.74, op),
                   ink=INK, width=2.2)

    # ------------------------------------------------------------------ 3: every 2nd
    def phase_3(self, sc, u, op):
        pos = self.scan(u)
        n = min(int(pos), 1023)
        i, j = divmod(n, 32)
        a = during(u, 0.04, 0.88, op)
        self.frame(sc, self.photo_cell(), 2 * i - 1, 2 * j - 1, 3, 3, self.PPX, self.PPX, a,
                   width=2.6)
        self.frame(sc, self.small_cell(), i, j, 1, 1, self.SPX, self.SPX, a, ink=INK,
                   width=2.0)
        done = int(np.ceil(pos)) if u > 0.06 else 0
        self.text(sc, 0.06, self.BAND - 0.03, "new pixels", op, SIZE, "lm")
        self.text(sc, 0.06, self.BAND + 0.025, f"{done:,}", op, BIG, "lm", ink=INK)
        self.text(sc, self.RES_X, self.BAND, "2 at a time", on(u, 0.1, op), SIZE, "lm",
                  ink=QUIET)

    # ------------------------------------------------------------------ 4: the matrix D
    def phase_4(self, sc, u, op):
        m = self.meta
        bw, bh = self.DB
        texts = m["D_corner_text"]
        vals = np.asarray(m["D_corner"])
        each = [[smoothstep(0.06 + 0.13 * r, 0.1 + 0.13 * r, u)] * 9 for r in range(4)]
        cell = self.boxes(sc, texts, *self.DC, bw, bh, op, fill=KERNEL_FILL, border=WEIGHT,
                          zero=(vals == 0).tolist(), each=each)
        for r in range(4):
            x, y = cell(r, 0)
            self.text(sc, 0.06, y, "row %d" % r, op * each[r][0], SIZE, "lm", ink=QUIET)
        top = self.DC[1] - self.frac(0, 2 * bh)[1] - 0.025
        for c in range(9):
            self.text(sc, cell(0, c)[0], top, str(c), on(u, 0.04, op), SIZE, ink=QUIET)
        self.text(sc, 0.06, top, "pixel", on(u, 0.04, op), SIZE, "lm", ink=QUIET)
        # the middle number of each row: 2 places on, each row
        st = on(u, 0.62, op)
        mids = [cell(r, 2 * r) for r in range(4)]
        for x, y in mids:
            self.rect(sc, self.P(x, y), bw + 1, bh + 1, INK, st, border=INK, border_w=2.4,
                      radius=4, lift=1.2, fill_op=0.0)
        self.text(sc, 0.06, 0.665, "each row: 2 places on", st, SIZE, "lm")

    def full_d(self, sc, op):
        """All of D, its corner outlined (phase 4, and fading as phase 5 starts)."""
        if op <= 0.01:
            return
        w, h = 64 * self.DPX, 32 * self.DPX
        self.rect(sc, self.P(*self.DFULL), w + 8, h + 8, PAPER, op, border=WEIGHT,
                  border_w=1.6, radius=3, lift=0.2)
        dcell = self.tiles(sc, self.D, *self.DFULL, w, h, op, ink=WEIGHT, lo=0.0, hi=0.5,
                           grid=0.0)
        self.frame(sc, dcell, 0, 0, 4, 9, self.DPX, self.DPX, op, ink=INK, width=1.6)
        self.text(sc, 0.06, self.DFULL[1], "D: 32 × 64", op, SIZE, "lm", ink=WEIGHT_INK)

    # ------------------------------------------------------------------ 5: D @ X @ D.T
    def phase_5(self, sc, u, op):
        m = self.meta
        p = self.LPX
        a1 = on(u, 0.12, op)
        self.arrow(sc, (self.PLATE[0] + 0.064, self.DX_AT[1]), (self.DX_AT[0] - 0.062,
                                                                self.DX_AT[1]), AXIS, a1,
                   width=2.0)
        self.text(sc, (self.PLATE[0] + self.DX_AT[0]) / 2 + 0.001, self.DX_AT[1] - 0.05, "D @",
                  a1, SIZE, ink=WEIGHT_INK)
        self.picture(sc, m["plate_dx"], *self.DX_AT, p, a1)
        self.text(sc, self.DX_AT[0], self.DX_AT[1] + 0.085, "32 × 64", a1, SIZE, ink=DATA_INK)
        a2 = on(u, 0.32, op)
        self.arrow(sc, (self.DX_AT[0] + 0.064, self.DX_AT[1]), (self.FIN_AT[0] - 0.036,
                                                                self.FIN_AT[1]), AXIS, a2,
                   width=2.0)
        self.text(sc, (self.DX_AT[0] + self.FIN_AT[0]) / 2 + 0.012, self.DX_AT[1] - 0.05,
                  "@ D.T", a2, SIZE, ink=WEIGHT_INK)
        self.picture(sc, m["by_matrix"], *self.FIN_AT, p, a2)
        self.text(sc, self.FIN_AT[0], self.FIN_AT[1] + 0.085, "32 × 32", a2, SIZE,
                  ink=DATA_INK)
        a3 = on(u, 0.55, op)
        self.text(sc, self.FIN_AT[0], (self.FIN_AT[1] + self.CONV_AT[1]) / 2 + 0.02, "=", a3,
                  HEAD, strong=True)
        self.picture(sc, m["plate_small"], *self.CONV_AT, p, a3)
        self.text(sc, self.CONV_AT[0] - 0.05, self.CONV_AT[1] - 0.02, "conv2d", a3, HEAD, "rm",
                  strong=True)
        self.text(sc, self.CONV_AT[0] - 0.05, self.CONV_AT[1] + 0.03, "stride 2", a3, SIZE,
                  "rm", ink=QUIET)
        a4 = on(u, 0.7, op)
        self.text(sc, 0.06, 0.85, "1.8e-7 apart: rounding", a4, SIZE, "lm")

    # ------------------------------------------------------------------ 6: false rings
    def phase_6(self, sc, u, op):
        m = self.meta
        r = m["fine_radius"] * self.LPX
        sk = on(u, 0.04, op)
        bl = on(u, 0.45, op)
        skc = self.picture(sc, m["plate_skip"], *self.SKIP_AT, self.BPX, sk)
        blc = self.picture(sc, m["plate_blur"], *self.BLUR_AT, self.BPX, bl)
        ring = on(u, 0.18, op)
        pc = self.picture_cell((64, 64), *self.PLATE, self.LPX)
        self.circle(sc, *pc(32, 32), r, LOSS_INK, ring)
        self.circle(sc, *skc(16, 16), r, LOSS_INK, ring * sk)
        self.circle(sc, *blc(16, 16), r, LOSS_INK, ring * bl)
        ly = self.SKIP_AT[1] + 0.125
        self.text(sc, self.SKIP_AT[0], ly, "skip only", sk, SIZE, ink=DATA_INK)
        self.text(sc, self.BLUR_AT[0], ly, "blur, then skip", bl, SIZE, ink=DATA_INK)
        ny = ly + 0.07
        self.text(sc, 0.06, ny, "false rings", on(u, 0.3, op), SIZE, "lm")
        self.text(sc, self.SKIP_AT[0], ny, m["rms_skip_text"], on(u, 0.3, op), HEAD,
                  strong=True)
        self.text(sc, self.BLUR_AT[0], ny, m["rms_blur_text"], on(u, 0.6, op), HEAD,
                  strong=True)
        self.text(sc, 0.06, ny + 0.075, "outside the circle: too fine for 32 × 32",
                  on(u, 0.72, op), SIZE, "lm", ink=QUIET)


# ====================================================================== 264
class Bigger(Resize):
    """Spread out; the gaps filled; nearest; the edge; 400 round trips; closer is not the
    point."""

    TITLE = "Bigger: spread, then blur"
    PHASES = (("spread", 9.0), ("fill the gaps", 13.0), ("nearest", 8.0), ("the edge", 11.0),
              ("400 photos", 10.0), ("closer?", 10.0))
    LAST = "Smaller: blur, then skip"
    NEXT = "A CNN training step"
    IN, IPX = (0.105, 0.53), 2.4
    OUT, LEFT, OPX = (0.35, 0.55), (0.14, 0.55), 2.2
    LABEL_Y = 0.385
    GRID, GB = (0.16, 0.805), (40.0, 22.0)
    CORNER_B = (44.0, 24.0)
    KER, KB = (0.4, 0.77), (36.0, 20.0)
    EQ = (0.275, 0.87)
    # phases 5-6: the bars
    BAR_X, BAR_W, BAR_MAX = 0.215, 0.18, 0.0035
    GROUPS = ((0.555, "inside the 1-pixel border"), (0.745, "the whole picture"))
    THUMB_Y, TPX = 0.445, 1.0

    def setup(self):
        m = self.meta = self.study.metadata
        self.small = np.asarray(m["small"], np.float32)
        self.sp = np.asarray(m["spread"], np.float32)
        self.bil = np.asarray(m["big"], np.float32)
        self.blocks = np.asarray(m["blocks"], np.float32)
        self.got = np.asarray(m["got"], np.float32)
        self.fixed = np.asarray(m["fixed"], np.float32)
        trip = Code(("rs-down", "rs-up"), pick=(1, 2, 3, 5, 6, 7, 8, 9),
                    run=((0.0, 0), (0.3, 3), (0.5, 6), (0.7, 3)))
        self.CODE = (
            Code("rs-up", run=((0.0, 0), (0.15, 1), (0.3, 2))),
            Code("rs-up", run=((0.0, 2), (0.6, 1))),
            Code("rs-up", run=((0.0, 4), (0.4, 5))),
            Code("rs-edge", run=((0.0, 0), (0.08, 1), (0.6, 2))),
            trip,
            Code(("rs-down", "rs-up"), pick=(1, 2, 3, 5, 6, 7, 8, 9),
                 run=((0.0, 2), (0.3, 3))))
        self.EVIDENCE = (exact("photo 0's 32 × 32, spread to 64 × 64"),
                         exact("the kernel × 4 on photo 0"),
                         exact("each pixel copied 2 × 2"),
                         exact("what a flat picture of ones receives"),
                         measured("400 photos, down then up", on=ON),
                         measured("400 photos, skip against blur", on=ON))

    def heading_for(self, k, u):
        return ("spread out: a zero between the pixels",
                "the kernel × 4 fills each gap",
                "nearest: each pixel copied, 2 × 2",
                "the edge gets only half",
                "400 photos: down, then up",
                "skip lands closer; blur avoids false rings")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "Photo 0's 32 × 32 picture, spread to 64 × 64: each pixel kept, with a zero "
            "after it.",
            "The kernel × 4 slides over it: a pixel stays itself, a gap gets the average of "
            "its neighbours. That is bilinear.",
            "Nearest copies each pixel into a 2 × 2 block, so the bigger picture comes out "
            "blocky.",
            "The last row and column have a neighbour past the edge, so they get half: "
            "big / got doubles them.",
            "Mean squared difference from the photo, pixels 0 to 1: inside the border, "
            "bilinear %s, nearest %s." % (m["inner_bilinear_text"], m["inner_nearest_text"]),
            "Skip then bilinear lands closer, %s: blur throws detail away. What it buys is "
            "no false patterns." % m["inner_skip_bilinear_text"],
        ][k]

    # ------------------------------------------------------------------ where things are
    def out_cell(self):
        return self.picture_cell((64, 64), *self.OUT, self.OPX)

    def left_cell(self):
        return self.picture_cell((64, 64), *self.LEFT, self.OPX)

    def close_box(self, cell):
        """The close-up's outline (rows 16-20, columns 44-48) on a 64 x 64 picture."""
        r0, c0 = self.meta["close_rows"][0], self.meta["close_cols"][0]
        return self.corners(cell, r0, c0, 5, 5, self.OPX, self.OPX)

    def out_image(self, k, u):
        """What the picture on the right shows: spread, filling in row by row, bilinear,
        then the edge made even."""
        if k == 0:
            return self.sp
        if k == 1:
            rows = int(round(64 * eased(u, 0.62, 0.86)))
            img = self.sp.copy()
            img[:rows] = self.bil[:rows]
            return img
        if (k == 3 and u >= 0.62) or k >= 4:
            return self.fixed
        return self.bil

    def grid_state(self, k, u):
        """The close-up's numbers and which tiles are grey zeros."""
        m = self.meta
        zero = (np.asarray(m["spread_close"]) == 0)
        if k == 0:
            return m["spread_close_text"], zero
        if k == 1:
            filled = np.zeros((5, 5), bool)
            g2, g4 = self.gap(m["gap2"]), self.gap(m["gap4"])
            if u >= 0.24:
                filled[g2] = True
            if u >= 0.46:
                filled[g4] = True
            if u >= 0.62:
                rows = int(round(5 * eased(u, 0.62, 0.86)))
                filled[:rows] = True
            texts = [[m["big_close_text"][i][j] if filled[i, j] else m["spread_close_text"][i][j]
                      for j in range(5)] for i in range(5)]
            return texts, zero & ~filled
        if u < 0.22:                  # phase 3: bilinear's numbers until nearest's are shown
            return m["big_close_text"], np.zeros((5, 5), bool)
        return m["blocks_close_text"], np.zeros((5, 5), bool)

    def gap(self, rc):
        r0, c0 = self.meta["close_rows"][0], self.meta["close_cols"][0]
        return rc[0] - r0, rc[1] - c0

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        # the 32 x 32 picture: phases 1-2
        ia = op * self.kept(0, 1)
        if ia > 0.01:
            self.picture(sc, self.small, *self.IN, self.IPX, ia)
            self.text(sc, self.IN[0], self.IN[1] - 0.115, "32 × 32", ia, SIZE, ink=DATA_INK)
            self.arrow(sc, (self.IN[0] + 0.055, self.IN[1]), (self.OUT[0] - 0.085, self.IN[1]),
                       AXIS, ia * (self.at(0, 0.1) if k == 0 else 1.0), width=2.0)
        # the 64 x 64 picture on the right: phases 1-4
        oa = op * self.kept(0, 4) * (self.at(0, 0.14) if k == 0 else 1.0) * \
            self.leaving(4, k, u)
        if oa > 0.01:
            oc = self.picture(sc, self.out_image(k, u), *self.OUT, self.OPX, oa)
            lab = ((0.0, "64 × 64"),) if k == 0 else \
                ((0.0, "64 × 64"), (0.86, "bilinear")) if k == 1 else \
                ((0.0, "bilinear"),) if k == 2 else \
                ((0.0, "bilinear"), (0.62, "big / got")) if k == 3 else ((0.0, "big / got"),)
            self.running(sc, self.OUT[0], self.LABEL_Y, lab, u, oa, size=SIZE, anchor="mm",
                         ink=DATA_INK)
            if k <= 2:
                ca = oa * (self.at(0, 0.34) if k == 0 else 1.0)
                self.mark(sc, self.close_box(oc), ca)
        # the close-up's numbers: phases 1-3
        ga = op * self.kept(0, 2) * (self.at(0, 0.34) if k == 0 else 1.0)
        if ga > 0.01:
            texts, zero = self.grid_state(k, u)
            cell = self.boxes(sc, texts, *self.GRID, *self.GB, ga, zero=zero.tolist())
            src = self.close_box(self.out_cell() if k < 2 else self.left_cell())
            dst = self.corners(cell, 0, 0, 5, 5, *self.GB)
            # from phase 3 the close-up is of the nearest picture, which fades in there
            self.zoom_guides(sc, src, dst, ga * (self.at(2, 0.16) if k == 2 else 1.0))
        # the bars: phases 5-6
        ba = op * self.kept(4, 5)
        if ba > 0.01:
            self.bars(sc, k, u, ba)

    # ------------------------------------------------------------------ the bars
    def bar(self, sc, key, label, value, text, op, ink=LOSS, lit=0.0):
        y = BAR_ROWS[key]
        self.text(sc, 0.06, y, label, op, SIZE, "lm", ink=INK if lit > 0.5 else None)
        w = self.BAR_W * value / self.BAR_MAX
        hp = 16.0
        self.rect(sc, self.P(self.BAR_X + w / 2, y), max(self.span(w)[0], 1.5), hp, ink, op,
                  radius=2, lift=0.3)
        self.text(sc, self.BAR_X + w + 0.008, y, text, op, SIZE, "lm", ink=INK)
        if lit > 0.01:
            self.rect(sc, self.P(self.BAR_X + w / 2, y), self.span(w)[0] + 8, hp + 8, INK,
                      op * lit, border=INK, border_w=2.2, radius=4, lift=1.2, fill_op=0.0)

    def bars(self, sc, k, u, op):
        m = self.meta

        def when(t):
            return op * (self.at(4, t) if k == 4 else 1.0)
        for (y, title), t in zip(self.GROUPS, (0.14, 0.52), strict=True):
            self.text(sc, 0.06, y, title, when(t), SIZE, "lm", ink=QUIET)
        self.bar(sc, "inner_bilinear", "blur, bilinear", m["inner_bilinear"],
                 m["inner_bilinear_text"], when(0.18))
        self.bar(sc, "inner_nearest", "blur, nearest", m["inner_nearest"],
                 m["inner_nearest_text"], when(0.34))
        self.bar(sc, "full_bilinear", "blur, bilinear", m["full_bilinear"],
                 m["full_bilinear_text"], when(0.56))
        self.bar(sc, "full_even", "blur, even edge", m["full_even"], m["full_even_text"],
                 when(0.72))

    # ------------------------------------------------------------------ 1: spread
    def phase_1(self, sc, u, op):
        m = self.meta
        a = on(u, 0.5, op)
        self.text(sc, self.EQ[0], self.EQ[1] - 0.05, "%s pixels kept" % "{:,}".format(m["kept"]),
                  a, SIZE, "lm")
        self.text(sc, self.EQ[0], self.EQ[1], "%s zeros added" % "{:,}".format(m["zeros"]),
                  on(u, 0.6, op), SIZE, "lm")

    # ------------------------------------------------------------------ 2: fill the gaps
    def phase_2(self, sc, u, op):
        m = self.meta
        bw, bh = self.KB
        ka = on(u, 0.04, op)
        self.boxes(sc, m["kernel4_text"], *self.KER, bw, bh, ka, fill=KERNEL_FILL,
                   border=WEIGHT)
        self.text(sc, self.KER[0] - 0.065, self.KER[1], "4 × k", ka, SIZE, "rm",
                  ink=WEIGHT_INK)
        cell = self.cells(*self.GRID, 5 * self.GB[0], 5 * self.GB[1], 5, 5)
        g2, g4 = self.gap(m["gap2"]), self.gap(m["gap4"])
        f1 = during(u, 0.12, 0.38, op)
        self.frame(sc, cell, g2[0] - 1, g2[1] - 1, 3, 3, *self.GB, f1, width=2.6)
        f2 = during(u, 0.38, 0.6, op)
        self.frame(sc, cell, g4[0] - 1, g4[1] - 1, 3, 3, *self.GB, f2, width=2.6)
        a, b = m["two_text"]
        steps = ((0.0, ""), (0.16, "(%s + %s) / 2 = %s" % (a, b, m["gap2_text"])),
                 (0.42, "average of 4 = %s" % m["gap4_text"]),
                 (0.62, "each gap: an average"))
        self.running(sc, *self.EQ, steps, u, on(u, 0.16, op), size=SIZE, ink=INK)

    # ------------------------------------------------------------------ 3: nearest
    def phase_3(self, sc, u, op):
        a = on(u, 0.08, op)
        self.picture(sc, self.blocks, *self.LEFT, self.OPX, a)
        self.text(sc, self.LEFT[0], self.LABEL_Y, "nearest", a, SIZE, ink=DATA_INK)
        lc = self.left_cell()
        self.mark(sc, self.close_box(lc), a)
        cell = self.cells(*self.GRID, 5 * self.GB[0], 5 * self.GB[1], 5, 5)
        b = on(u, 0.3, op)
        for r in (0, 2, 4):
            for c in (0, 2, 4):
                rows, cols = min(2, 5 - r), min(2, 5 - c)
                self.frame(sc, cell, r, c, rows, cols, *self.GB, b, ink=DATA_INK, width=2.2)
        self.text(sc, self.EQ[0], self.EQ[1], "copies: 2 × 2 blocks", on(u, 0.4, op), SIZE,
                  "lm", ink=INK)

    # ------------------------------------------------------------------ 4: the edge
    def phase_4(self, sc, u, op):
        m = self.meta
        a = on(u, 0.06, op)
        gc = self.picture(sc, self.got, *self.LEFT, self.OPX, a)
        self.text(sc, self.LEFT[0], self.LABEL_Y, "got = up(ones)", a, SIZE, ink=DATA_INK)
        b = on(u, 0.2, op)
        src = self.corners(gc, 61, 61, 3, 3, self.OPX, self.OPX)
        self.mark(sc, src, b)
        bw, bh = self.CORNER_B
        cell = self.boxes(sc, m["got_corner_text"], self.GRID[0], self.GRID[1], bw, bh, b,
                          ring=((0, 2), (1, 2), (2, 0), (2, 1)), ring_ink=INK)
        self.zoom_guides(sc, src, self.corners(cell, 0, 0, 3, 3, bw, bh), b)
        # the photo's last row and column, outlined
        oc = self.out_cell()
        e = on(u, 0.36, op)
        right = self.corners(oc, 0, 63, 64, 1, self.OPX, self.OPX)
        bottom = self.corners(oc, 63, 0, 1, 64, self.OPX, self.OPX)
        for x0, y0, x1, y1 in (right, bottom):
            w, h = self.span(x1 - x0, y1 - y0)
            self.rect(sc, self.P((x0 + x1) / 2, (y0 + y1) / 2), w + 5, h + 5, INK, e,
                      border=INK, border_w=2.0, radius=2, lift=1.1, fill_op=0.0)
        self.text(sc, 0.27, 0.77, "last row, on average", e, SIZE, "lm")
        self.running(sc, 0.27, 0.83, ((0.0, m["last_row_text"]),
                                      (0.62, "%s → %s" % (m["last_row_text"],
                                                          m["last_row_fixed_text"]))),
                     u, e, size=HEAD, ink=INK)

    # ------------------------------------------------------------------ 5: 400 photos
    def phase_5(self, sc, u, op):
        m = self.meta
        a = on(u, 0.08, op)
        y = self.THUMB_Y
        p = self.TPX
        self.picture(sc, m["photo"], 0.1, y, p, a)
        self.arrow(sc, (0.142, y), (0.168, y), AXIS, a, width=1.8)
        self.picture(sc, self.small, 0.19, y, p, a)
        self.arrow(sc, (0.212, y), (0.238, y), AXIS, a, width=1.8)
        self.picture(sc, self.bil, 0.28, y, p, a)
        self.text(sc, 0.33, y, "× %d photos" % m["photos"], a, SIZE, "lm")

    # ------------------------------------------------------------------ 6: closer?
    def phase_6(self, sc, u, op):
        m = self.meta
        a = on(u, 0.1, op)
        self.bar(sc, "inner_skip", "skip, bilinear", m["inner_skip_bilinear"],
                 m["inner_skip_bilinear_text"], a, ink=soft(LOSS, 0.35),
                 lit=smoothstep(0.24, 0.3, u))
        # what blur buys: the zone plate's false rings, from last time
        b = on(u, 0.45, op)
        y = self.THUMB_Y
        p = 1.6
        self.picture(sc, m["plate_skip"], 0.1, y, p, b)
        self.text(sc, 0.14, y - 0.025, "skip", b, SIZE, "lm", ink=DATA_INK)
        self.text(sc, 0.14, y + 0.02, "rings " + m["rms_skip_text"], b, SIZE, "lm", ink=INK)
        c = on(u, 0.58, op)
        self.picture(sc, m["plate_blur"], 0.29, y, p, c)
        self.text(sc, 0.33, y - 0.025, "blur", c, SIZE, "lm", ink=DATA_INK)
        self.text(sc, 0.33, y + 0.02, "rings " + m["rms_blur_text"], c, SIZE, "lm", ink=INK)


ANIMATIONS = {"resample-down": Smaller, "resample-up": Bigger}

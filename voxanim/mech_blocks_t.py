"""Animations 238-240, the first three of "Blocks and code: finding faces" (see
:mod:`voxanim.blocks`), drawn from :mod:`voxanim.study_blocks_t`. All three follow one scene,
the first of the 200 held-out scenes, drawn on the left as the photo it is (grey, on a blue
data card), with what each animation reads off it drawn over it as guides, in front.

* :class:`ABox` (``faces-a-box``): the scene we made (2 faces pasted onto paper, 2
  not-faces); one face's box as the script writes it, centre and size (27.5, 61.5, 19, 19);
  the same box as corners after ``box_convert``, (18, 52, 37, 71); the trained detector's box
  for that face, magnified beside the scene; and their IoU worked out as areas, 308 ÷ 436 =
  0.71.
* :class:`TheGrid` (``faces-the-grid``): the 12 x 12 grid of 8 x 8-pixel cells over the scene;
  the face's centre falls in row 7, column 3; that cell magnified, the centre 0.44 across and
  0.69 down it; the size as log(19 / 16) = 0.17; ``targets``: the two face cells hold 1 and
  their numbers, the other 142 cells say 0, the not-faces' too.
* :class:`TheDetector` (``faces-the-detector``): the scene through ``block`` three times, the
  trained detector's real maps as stacks of cards with their shapes, [1, 8, 48, 48] ->
  [1, 16, 24, 24] -> [1, 32, 12, 12]; the 1 x 1 convolution to [1, 5, 12, 12] and the first
  number laid over the scene (0.97 in the face's cell); 6,053 learned numbers; what one cell
  can see, a square growing 3, 4, 8, 10, 18, 22 pixels across.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). What stays across phases is drawn in
``steady``, so nothing blanks between them.

Measured phases carry the series' tag for this data, "measured on Olivetti faces: ..."
(:func:`~voxanim.blocks.measured` with ``on=FACES``); arithmetic carries "exact arithmetic".
"""

from __future__ import annotations

import numpy as np

from . import layout
from .blocks import (
    BIG,
    DATA,
    HEAD,
    INK,
    QUIET,
    SIZE,
    WEIGHT,
    ZERO,
    Blocks,
    Code,
    exact,
    measured,
    shades,
)
from .diagram import draw_image
from .easing import ease_in_out, smoothstep
from .mech import AXIS, PAPER, dark, soft

DATA_INK = dark(DATA, 0.3)
WEIGHT_INK = dark(WEIGHT, 0.25)
TRUTH = dark(DATA, 0.15)            # a true box: the data's own (we pasted the face)
GUESS = WEIGHT                      # the detector's box: what the model drew
NOT = ZERO                          # a not-face's square
FACES = "Olivetti faces"            # what the measured tags say they were measured on


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on(u, a, op, width=0.04):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


def gray(img):
    """A photo as grey RGB (an image is data: it looks the same in both themes)."""
    g = np.clip(np.asarray(img, np.float32), 0.0, 1.0)
    return np.repeat(g[..., None], 3, -1)


class Faces(Blocks):
    """What the three share: the scene on a data card, and guides drawn over it in front."""

    SC = (0.165, 0.59)              # the scene card's centre
    PX = 2.0                        # screen px per scene pixel
    SIDE = 96
    RX = 0.295                      # the right-hand column of words and numbers
    BOX_W = 2.4                     # a box's line, px

    def read_scene(self):
        m = self.meta
        self.canvas = np.asarray(m["canvas"], float)
        self.squares = [tuple(q) for q in m.get("squares", ())]

    # ------------------------------------------------------------------ geometry
    def img_centre(self):
        return self.SC[0], self.SC[1] + self.frac(0, 3.0)[1]

    def pt(self, x, y):
        """Scene pixel coordinates (x right, y down; 0..96) as layout coordinates."""
        cx, cy = self.img_centre()
        fx, fy = self.frac(self.PX, self.PX)
        return cx + (x - self.SIDE / 2) * fx, cy + (y - self.SIDE / 2) * fy

    def card_box(self):
        """The scene card's layout extent: ``(left, top, right, bottom)``."""
        w, h = self.frac(self.SIDE * self.PX + 14, self.SIDE * self.PX + 20)
        return self.SC[0] - w / 2, self.SC[1] - h / 2, self.SC[0] + w / 2, self.SC[1] + h / 2

    # ------------------------------------------------------------------ drawing
    def scene_card(self, sc, op):
        if op <= 0.01:
            return
        side = self.SIDE * self.PX
        self.card(sc, self.SC[0], self.SC[1], side + 14, side + 20, op)
        draw_image(self, sc, self.P(*self.img_centre()), gray(self.canvas), side, side, op,
                   grid=0.0, lift=0.204)

    def frame(self, sc, corners, ink, op, width=None, pt=None, scale=None, lift=1.25):
        """A box over the scene from its corners ``(x1, y1, x2, y2)`` (scene pixels): an
        outline in front of the photo."""
        if op <= 0.01:
            return
        pt, scale = pt or self.pt, scale or self.PX
        x1, y1, x2, y2 = corners
        self.rect(sc, self.P(*pt((x1 + x2) / 2, (y1 + y2) / 2)), (x2 - x1) * scale,
                  (y2 - y1) * scale, ink, op, border=ink, border_w=width or self.BOX_W,
                  radius=1.5, lift=lift, fill_op=0.0)

    def fill(self, sc, corners, ink, op, pt=None, scale=None, lift=1.15):
        """A translucent patch over the scene (scene pixels)."""
        if op <= 0.01:
            return
        pt, scale = pt or self.pt, scale or self.PX
        x1, y1, x2, y2 = corners
        self.rect(sc, self.P(*pt((x1 + x2) / 2, (y1 + y2) / 2)), (x2 - x1) * scale,
                  (y2 - y1) * scale, ink, op, radius=0.5, lift=lift)

    def spot(self, sc, x, y, ink, op, r=3.6, pt=None):
        """A dot at scene pixel ``(x, y)``, in front."""
        if op > 0.01:
            self.disk(sc, self.P(*(pt or self.pt)(x, y)), r, ink, op, lift=1.5, outline=0.15)

    def guide(self, sc, a, b, ink, op, width=1.6):
        """A straight guide from layout ``a`` to ``b``, lifted in front of the blocks."""
        if op > 0.01:
            self.polyline(sc, [self.P(*a, self.ARROW_LIFT), self.P(*b, self.ARROW_LIFT)], width,
                          ink, op, bias=0.4)

    def span_mark(self, sc, a, b, ink, op, pt=None):
        """A measured span over the scene from pixel ``a`` to ``b``, with end ticks (no
        words on the photo: the number is in the column, in the same ink)."""
        if op <= 0.01:
            return
        pt = pt or self.pt
        pa, pb = pt(*a), pt(*b)
        self.guide(sc, pa, pb, ink, op, 2.0)
        d = np.array([pb[0] - pa[0], pb[1] - pa[1]])
        n = np.array([-d[1], d[0]]) / max(float(np.hypot(*d)), 1e-9)
        tx, ty = self.frac(5.0, 5.0)
        for p in (pa, pb):
            self.guide(sc, (p[0] + n[0] * tx, p[1] + n[1] * ty), (p[0] - n[0] * tx,
                                                                   p[1] - n[1] * ty), ink, op,
                       2.0)

    def grid_lines(self, sc, op, n=12, ink=None, pt=None, x0=0.0, y0=0.0, size=96.0):
        """``n`` x ``n`` cells over a square of the scene, as lines in front."""
        if op <= 0.01:
            return
        pt = pt or self.pt
        ink = PAPER if ink is None else ink
        for i in range(n + 1):
            v = size * i / n
            self.guide(sc, pt(x0 + v, y0), pt(x0 + v, y0 + size), ink, op, 1.1)
            self.guide(sc, pt(x0, y0 + v), pt(x0 + size, y0 + v), ink, op, 1.1)

    def swatch(self, sc, fx, fy, ink, op, filled=False):
        """A small square of a box's colour before a line of the column."""
        if op > 0.01:
            self.rect(sc, self.P(fx, fy), 13.0, 13.0, ink, op, border=ink, border_w=2.2,
                      radius=1.5, lift=0.5, fill_op=0.45 if filled else 0.0)


# ====================================================================== 238
class ABox(Faces):
    """A made scene; one face's box as centre and size; as corners; the detector's box; IoU."""

    TITLE = "A box is four numbers"
    PHASES = (("a made scene", 11.0), ("centre and size", 13.0), ("corners", 12.0),
              ("the detector's box", 11.0), ("IoU", 13.0))
    LAST = "Style transfer with Gram matrices"
    NEXT = "A grid of guesses"
    STATION = ("data", "data", "data", "forward", None)
    SC = (0.165, 0.565)
    Z0, ZN, ZPX = (14.0, 46.0), 28.0, 5.0       # phases 4-5: the magnified corner of the scene
    ZC = (0.385, 0.565)                         # its card's centre
    LX, LOW = 0.058, (0.8, 0.843, 0.886)        # phases 4-5: lines under the pictures

    def setup(self):
        m = self.meta = self.study.metadata
        self.read_scene()
        self.face = int(m["face"])
        self.real = [list(map(float, r)) for r in m["real"]]
        self.truth = self.real[self.face]
        self.guess = [float(v) for v in m["guess"]]
        self.inter = [float(v) for v in m["inter"]]
        paste = "fc-paste"
        self.CODE = (
            Code(paste, run=((0.0, None), (0.12, 0), (0.45, 2))),
            Code(paste, run=((0.0, 3),), out=((0.62, m["box_text"]),)),
            Code("fc-iou", run=((0.0, 0), (0.2, 1)), out=((0.75, m["printed_real"]),)),
            Code("fc-nms", run=((0.0, 0), (0.25, 2))),
            Code("fc-iou", run=((0.0, 2),), out=((0.72, m["printed_iou"]),)))
        pasted = exact("we pasted the faces, so the boxes are exact")
        self.EVIDENCE = (pasted, pasted, pasted,
                         measured("the trained detector, held-out", on=FACES),
                         measured("true box against the detector's", on=FACES))

    def heading_for(self, k, u):
        return ("a scene we made: faces pasted on paper",
                "a box: its centre and its size",
                "the same box as two corners",
                "the detector's box for the same face",
                "IoU: the overlap ÷ both together")[k]

    def subtitle_for(self, k):
        m = self.meta
        x, y, s = m["paste"]
        a = m["area_texts"]
        return [
            "A scene we made: %d faces pasted onto paper, and %d not-faces: photos shuffled "
            "in 8 × 8 blocks." % (m["faces"], m["nots"]),
            "This face was pasted at x %d, y %d, %d pixels wide: its box is centre %s and "
            "size %d × %d." % (x, y, s, m["centre_text"], s, s),
            "box_convert writes the same box as its two corners, x1 = cx − w / 2 and "
            "x2 = cx + w / 2: %s." % m["corners_text"],
            "The trained detector's box for this face, %s sure: a little off the true "
            "box." % m["guess_score_text"],
            "IoU = overlap ÷ together = %s ÷ %s = %s; 1 is the same box. Right boxes average "
            "%s over %d held-out scenes." % (a["inter"], a["union"], m["iou_text"],
                                             m["mean_iou_text"], m["test_scenes"]),
        ][k]

    # ------------------------------------------------------------------ the magnifier
    def zpt(self, x, y):
        """Scene pixel ``(x, y)`` in the magnified corner (phases 4-5)."""
        fx, fy = self.frac(self.ZPX, self.ZPX)
        half = self.ZN / 2
        return (self.ZC[0] + (x - self.Z0[0] - half) * fx,
                self.ZC[1] + 0.0055 + (y - self.Z0[1] - half) * fy)

    def magnifier(self, sc, op):
        """The corner of the scene round the face, 5 x bigger, on a card; guides from the
        region on the scene to it."""
        if op <= 0.01:
            return
        x0, y0 = self.Z0
        n = int(self.ZN)
        side = self.ZN * self.ZPX
        crop = self.canvas[int(y0):int(y0) + n, int(x0):int(x0) + n]
        self.card(sc, self.ZC[0], self.ZC[1], side + 14, side + 20, op)
        draw_image(self, sc, self.P(*self.zpt(x0 + self.ZN / 2, y0 + self.ZN / 2)), gray(crop),
                   side, side, op, grid=0.0, lift=0.204)
        self.frame(sc, (x0, y0, x0 + self.ZN, y0 + self.ZN), QUIET, op, width=1.4)
        for (ax, ay), (bx, by) in (((x0 + self.ZN, y0), (x0, y0)),
                                   ((x0 + self.ZN, y0 + self.ZN), (x0, y0 + self.ZN))):
            self.guide(sc, self.pt(ax, ay), self.zpt(bx, by), QUIET, op * 0.8, 1.2)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        self.scene_card(sc, op)
        # the faces' true boxes from phase 1 on, the followed one stronger from phase 2
        for i, box in enumerate(self.real):
            a = op * (on(u, 0.3, 1.0) if k == 0 else 1.0)
            mine = i == self.face
            if k >= 1 and not mine:
                a *= 0.45 if k > 1 else 1.0 - 0.55 * smoothstep(0.0, 0.1, u)
            self.frame(sc, box, TRUTH, a)
        # the not-faces' squares: phase 1 only
        nots = op * self.kept(0, 0) * smoothstep(0.5, 0.55, u if k == 0 else 1.0)
        for x, y, s in self.squares[self.meta["faces"]:]:
            self.frame(sc, (x, y, x + s, y + s), NOT, nots, width=1.8)
        # the centre (phase 2 on) and the corners (phase 3 on)
        x1, y1, x2, y2 = self.truth
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if k >= 1:
            self.spot(sc, cx, cy, TRUTH, op * (on(u, 0.12, 1.0) if k == 1 else 1.0))
        if k >= 2:
            c = op * (on(u, 0.3, 1.0) if k == 2 else 1.0)
            self.spot(sc, x1, y1, TRUTH, c * (on(u, 0.3, 1.0) if k == 2 else 1.0), r=3.0)
            self.spot(sc, x2, y2, TRUTH, c * (on(u, 0.52, 1.0) if k == 2 else 1.0), r=3.0)
        # the detector's box, phase 4 on, and the magnifier
        if k >= 3:
            g = op * (on(u, 0.3, 1.0) if k == 3 else 1.0)
            self.frame(sc, self.guess, GUESS, g)
            z = op * (on(u, 0.08, 1.0) if k == 3 else 1.0)
            self.magnifier(sc, z)
            self.frame(sc, self.truth, TRUTH, z, pt=self.zpt, scale=self.ZPX)
            self.frame(sc, self.guess, GUESS, g, pt=self.zpt, scale=self.ZPX)

    # ------------------------------------------------------------------ 1: a made scene
    def phase_1(self, sc, u, op):
        m = self.meta
        x = self.RX + 0.03
        self.text(sc, self.RX, 0.45, "%d × %d pixels" % (m["side"], m["side"]), on(u, 0.12, op),
                  HEAD, "lm", strong=True)
        f = on(u, 0.3, op)
        self.swatch(sc, self.RX + 0.008, 0.54, TRUTH, f)
        self.text(sc, x, 0.54, "%d faces, pasted" % m["faces"], f, SIZE, "lm", ink=DATA_INK)
        n = on(u, 0.5, op)
        self.swatch(sc, self.RX + 0.008, 0.6, NOT, n)
        self.text(sc, x, 0.6, "%d not-faces:" % m["nots"], n, SIZE, "lm")
        self.text(sc, x, 0.64, "photos, shuffled", n, SIZE, "lm")
        self.text(sc, self.RX, 0.74, "we know every box", on(u, 0.7, op), SIZE, "lm",
                  strong=True)

    # ------------------------------------------------------------------ 2: centre and size
    def phase_2(self, sc, u, op):
        m = self.meta
        x1, y1, x2, y2 = self.truth
        rows = tuple(zip(m["parts_text"], (0.2, 0.3, 0.42, 0.52), strict=True))
        for i, (s, at) in enumerate(rows):
            self.text(sc, self.RX, 0.44 + 0.062 * i, s, on(u, at, op), HEAD, "lm",
                      ink=DATA_INK if i < 2 else INK)
        # the spans on the scene, as each number arrives
        self.span_mark(sc, (x1, y1 - 3.5), (x2, y1 - 3.5), INK, on(u, 0.42, op))
        self.span_mark(sc, (x2 + 3.5, y1), (x2 + 3.5, y2), INK, on(u, 0.52, op))
        self.text(sc, self.RX, 0.72, "the centre, then the size", on(u, 0.66, op), SIZE, "lm",
                  ink=QUIET)

    # ------------------------------------------------------------------ 3: corners
    def phase_3(self, sc, u, op):
        m = self.meta
        for i, s in enumerate(m["corner_sums"]):
            self.text(sc, self.RX, 0.44 + 0.045 * i, s, on(u, 0.3 + 0.08 * (i % 2) +
                                                         0.22 * (i // 2), op), SIZE, "lm")
        self.text(sc, self.RX, 0.66, "corners:", on(u, 0.72, op), SIZE, "lm", ink=QUIET)
        self.text(sc, self.RX, 0.71, m["corners_text"], on(u, 0.72, op), HEAD, "lm",
                  strong=True)
        self.text(sc, self.RX, 0.77, "the same box", on(u, 0.82, op), SIZE, "lm",
                  ink=DATA_INK)

    # ------------------------------------------------------------------ 4: the detector's box
    def phase_4(self, sc, u, op):
        m = self.meta
        x, (y0, y1, y2) = self.LX, self.LOW
        t = on(u, 0.15, op)
        self.swatch(sc, x + 0.008, y0, TRUTH, t)
        self.text(sc, x + 0.03, y0, "true box: %s" % m["corners_text"], t, SIZE, "lm",
                  ink=DATA_INK)
        g = on(u, 0.3, op)
        self.swatch(sc, x + 0.008, y1, GUESS, g)
        self.text(sc, x + 0.03, y1, "the detector's box: %s" % m["guess_text"], g, SIZE, "lm",
                  ink=WEIGHT_INK)
        self.text(sc, x + 0.03, y2, "it is %s sure of a face" % m["guess_score_text"],
                  on(u, 0.5, op), SIZE, "lm", ink=QUIET)

    # ------------------------------------------------------------------ 5: IoU
    def phase_5(self, sc, u, op):
        m = self.meta
        a = m["area_texts"]
        x, (y0, y1, y2) = self.LX, self.LOW
        # the overlap, filled, on the scene and in the magnifier
        o = on(u, 0.1, op)
        self.fill(sc, self.inter, DATA, o * 0.45, pt=self.zpt, scale=self.ZPX)
        self.fill(sc, self.inter, DATA, o * 0.45)
        self.swatch(sc, x + 0.008, y0, DATA, o, filled=True)
        self.text(sc, x + 0.03, y0, "overlap: %s" % a["inter"], o, SIZE, "lm", ink=DATA_INK)
        self.text(sc, x + 0.03, y1, "together: %s" % m["union_sum"], on(u, 0.3, op), SIZE, "lm")
        self.text(sc, x + 0.03, y2 + 0.004, "IoU = %s" % m["iou_sum"], on(u, 0.5, op), HEAD,
                  "lm", strong=True)


# ====================================================================== 239
class TheGrid(Faces):
    """The grid; the centre's cell; where in the cell; how big; every other cell says 0."""

    TITLE = "A grid of guesses"
    PHASES = (("12 × 12 cells", 11.0), ("one cell", 12.0), ("where in it", 12.0),
              ("how big", 11.0), ("every cell", 14.0))
    LAST = "A box is four numbers"
    NEXT = "From pixels to a grid"
    STATION = ("data",) * 5
    SC, PX, RX = (0.18, 0.58), 2.3, 0.325       # a bigger scene: its cells are 18 px
    MC, MPX = (0.4, 0.5), 14.0                  # phase 3: the cell, magnified
    HC, HT = (0.4, 0.5), 9.0                    # phase 5: has, 12 x 12 tiles
    ROW = (0.058, 0.19, 0.052, 0.848)           # phase 5: the face's cell's 5 numbers

    def setup(self):
        m = self.meta = self.study.metadata
        self.read_scene()
        self.squares = []
        self.fbox = [float(v) for v in m["box"]]
        cx, cy, w, h = self.fbox
        self.truth = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]
        self.r, self.c = int(m["row"]), int(m["col"])
        self.has = np.asarray(m["has"], float)
        off, tg = "fc-offsets", "fc-targets"
        self.CODE = (
            Code(tg, run=((0.0, None), (0.25, 1), (0.6, 2))),
            Code(off, run=((0.0, 0), (0.3, 1))),
            Code(off, run=((0.0, 1), (0.2, 2))),
            Code(off, run=((0.0, 3), (0.35, 4), (0.55, 5), (0.7, 6)),
                 out=((0.72, m["printed_offsets_wrapped"]),)),
            Code(tg, run=((0.0, 3), (0.15, 4), (0.3, 5), (0.5, 6), (0.75, 7))))
        self.EVIDENCE = (exact("96 ÷ 8: twelve cells across"),) + \
            (exact("offsets() on the pasted box"),) * 3 + \
            (exact("targets() on the scene's two boxes"),)

    def heading_for(self, k, u):
        return ("the answer is a 12 × 12 grid",
                "the cell the face's centre falls in",
                "where the centre is in that cell",
                "how big: log(size ÷ 16)",
                "every other cell says 0")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "The detector answers on a grid: %s cells across and down, each 8 × 8 pixels."
            % m["grid_sum"],
            "The face's centre %s falls in row %d (%s) and column %d (%s)."
            % (m["centre_text"], m["row"], m["row_div"], m["col"], m["col_div"]),
            "That cell says where in it the centre is, from 0 to 1: %s across and %s down."
            % (m["dx_text"], m["dy_text"]),
            "And how big the box is, as log(size ÷ 16): %d pixels gives %s; 16 to 32 "
            "give 0 to %s." % (m["size"], m["lw_text"], m["size_logs_text"][1]),
            "targets() puts 1 and the 4 numbers in each face's cell; the other %d cells say "
            "0, the not-faces' too." % m["has_zeros"],
        ][k]

    # ------------------------------------------------------------------ geometry
    def cell_corners(self, r, c):
        s = 8.0
        return (c * s, r * s, c * s + s, r * s + s)

    def mpt(self, x, y):
        """Scene pixel ``(x, y)`` in the magnified cell (phase 3)."""
        c0, r0 = self.c * 8.0, self.r * 8.0
        fx, fy = self.frac(self.MPX, self.MPX)
        return self.MC[0] + (x - c0 - 4.0) * fx, self.MC[1] + 0.0055 + (y - r0 - 4.0) * fy

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        self.scene_card(sc, op)
        g = op * (on(u, 0.08, 1.0) if k == 0 else 1.0)
        self.grid_lines(sc, g * 0.6)
        # the face's box, faint, and its centre, from phase 2 on
        if k >= 1:
            a = op * (on(u, 0.05, 1.0) if k == 1 else 1.0)
            self.frame(sc, self.truth, TRUTH, a * 0.55, width=1.8)
            self.spot(sc, self.fbox[0], self.fbox[1], TRUTH, op * (on(u, 0.12, 1.0) if k == 1
                                                                 else 1.0))
            ring = op * (on(u, 0.62, 1.0) if k == 1 else 1.0)
            self.frame(sc, self.cell_corners(self.r, self.c), INK, ring, width=2.6, lift=1.3)

    # ------------------------------------------------------------------ 1: the grid
    def phase_1(self, sc, u, op):
        m = self.meta
        self.text(sc, self.RX, 0.45, "%d × %d cells" % (m["grid"], m["grid"]), on(u, 0.15, op),
                  HEAD, "lm", strong=True)
        self.text(sc, self.RX, 0.51, "each %d × %d pixels" % (m["cell"], m["cell"]),
                  on(u, 0.35, op), SIZE, "lm")
        self.text(sc, self.RX, 0.56, m["grid_sum"], on(u, 0.5, op), SIZE, "lm")
        a = on(u, 0.62, op)
        self.frame(sc, self.cell_corners(0, 0), INK, a, width=2.4, lift=1.3)
        self.text(sc, self.RX, 0.65, "one cell, ringed", a, SIZE, "lm", ink=QUIET)

    # ------------------------------------------------------------------ 2: one cell
    def phase_2(self, sc, u, op):
        m = self.meta
        self.text(sc, self.RX, 0.44, "centre %s" % m["centre_text"], on(u, 0.12, op), SIZE, "lm",
                  ink=DATA_INK)
        # the row band, then the column band, then the cell
        rb = op * smoothstep(0.25, 0.3, u) * (1 - smoothstep(0.6, 0.66, u))
        self.fill(sc, (0, self.r * 8, 96, self.r * 8 + 8), DATA, rb * 0.3)
        cb = op * smoothstep(0.42, 0.47, u) * (1 - smoothstep(0.6, 0.66, u))
        self.fill(sc, (self.c * 8, 0, self.c * 8 + 8, 96), DATA, cb * 0.3)
        self.text(sc, self.RX, 0.5, m["row_div"], on(u, 0.25, op), SIZE, "lm")
        self.text(sc, self.RX, 0.55, "row %d" % m["row"], on(u, 0.3, op), HEAD, "lm",
                  strong=True)
        self.text(sc, self.RX, 0.62, m["col_div"], on(u, 0.42, op), SIZE, "lm")
        self.text(sc, self.RX, 0.67, "column %d" % m["col"], on(u, 0.47, op), HEAD, "lm",
                  strong=True)
        self.text(sc, self.RX, 0.75, "its cell, ringed", on(u, 0.62, op), SIZE, "lm",
                  ink=QUIET)

    # ------------------------------------------------------------------ 3: where in it
    def magnified(self, sc, op):
        if op <= 0.01:
            return
        c0, r0 = self.c * 8, self.r * 8
        side = 8 * self.MPX
        self.card(sc, self.MC[0], self.MC[1], side + 14, side + 20, op)
        draw_image(self, sc, self.P(*self.mpt(c0 + 4.0, r0 + 4.0)),
                   gray(self.canvas[r0:r0 + 8, c0:c0 + 8]), side, side, op, grid=0.0,
                   lift=0.204)
        self.frame(sc, (c0, r0, c0 + 8, r0 + 8), INK, op, pt=self.mpt, scale=self.MPX,
                   width=2.6)
        for (ax, ay), (bx, by) in (((c0 + 8, r0), (c0, r0)), ((c0 + 8, r0 + 8), (c0, r0 + 8))):
            self.guide(sc, self.pt(ax, ay), self.mpt(bx, by), QUIET, op * 0.8, 1.2)

    def phase_3(self, sc, u, op):
        m = self.meta
        c0, r0 = self.c * 8, self.r * 8
        cx, cy = self.fbox[0], self.fbox[1]
        self.magnified(sc, on(u, 0.05, op))
        self.spot(sc, cx, cy, TRUTH, on(u, 0.15, op), r=5.0, pt=self.mpt)
        a = on(u, 0.3, op)
        self.span_mark(sc, (c0, r0 + 9.1), (cx, r0 + 9.1), DATA_INK, a, pt=self.mpt)
        b = on(u, 0.5, op)
        self.span_mark(sc, (c0 - 1.1, r0), (c0 - 1.1, cy), DATA_INK, b, pt=self.mpt)
        y = self.MC[1] + self.frac(0, 8 * self.MPX / 2 + 30)[1]
        self.text(sc, self.RX, y, "across: %s" % m["dx_text"], a, HEAD, "lm", ink=DATA_INK)
        self.text(sc, self.RX, y + 0.055, "down: %s" % m["dy_text"], b, HEAD, "lm",
                  ink=DATA_INK)
        self.text(sc, self.RX, y + 0.11, "0 to 1 inside the cell", on(u, 0.7, op), SIZE, "lm",
                  ink=QUIET)

    # ------------------------------------------------------------------ 4: how big
    def phase_4(self, sc, u, op):
        m = self.meta
        x1, y1, x2, _y2 = self.truth
        a = on(u, 0.1, op)
        self.frame(sc, self.truth, TRUTH, a)
        self.span_mark(sc, (x1, y1 - 3.5), (x2, y1 - 3.5), INK, a)
        self.text(sc, self.RX, 0.44, "w = h = %d pixels" % m["size"], a, SIZE, "lm")
        self.text(sc, self.RX, 0.5, m["log_sum"], on(u, 0.3, op), HEAD, "lm", strong=True)
        s = on(u, 0.55, op)
        lo, hi = m["size_range"]
        self.text(sc, self.RX, 0.6, "sizes %d to %d:" % (lo, hi), s, SIZE, "lm", ink=QUIET)
        self.text(sc, self.RX, 0.645, "%d → %s" % (lo, m["size_logs_text"][0]), s, SIZE, "lm")
        self.text(sc, self.RX, 0.69, "%d → %s" % (hi, m["size_logs_text"][1]), s, SIZE, "lm")
        self.text(sc, self.RX, 0.77, "small numbers either way", on(u, 0.75, op), SIZE, "lm",
                  ink=QUIET)

    # ------------------------------------------------------------------ 5: every cell
    def phase_5(self, sc, u, op):
        m = self.meta
        side = 12 * self.HT
        a = on(u, 0.05, op)
        lit = smoothstep(0.3, 0.36, u)
        shown = self.has * lit
        self.card(sc, self.HC[0], self.HC[1], side + 12, side + 18, a)
        self.tiles(sc, shown, self.HC[0], self.HC[1] + self.frac(0, 3)[1], side, side, a,
                   lo=0.0, hi=1.0, grid=0.35)
        self.text(sc, self.HC[0], self.HC[1] - self.frac(0, side / 2 + 24)[1], "has",
                  a, SIZE, ink=DATA_INK)
        for r, c in m["face_cells"]:
            self.fill(sc, self.cell_corners(r, c), DATA, op * lit * 0.5)
        y = self.HC[1] + self.frac(0, side / 2 + 30)[1]
        self.text(sc, self.RX, y, "%d cells say 1" % m["has_ones"], op * lit, SIZE,
                  "lm", ink=DATA_INK)
        z = on(u, 0.6, op)
        self.text(sc, self.RX, y + 0.045, "%d cells say 0" % m["has_zeros"], z, SIZE, "lm",
                  strong=True)
        n = on(u, 0.78, op)
        for x, yy, s in self.meta_nots():
            self.frame(sc, (x, yy, x + s, yy + s), NOT, n, width=1.8)
        self.text(sc, self.RX, y + 0.09, "not-faces too", n, SIZE, "lm", ink=QUIET)
        # the face's cell: its 5 numbers in a row under the scene
        x0, x1, step, ry = self.ROW
        f = on(u, 0.42, op)
        self.text(sc, x0, ry, "its cell:", f, SIZE, "lm", ink=DATA_INK)
        for i, (v, name) in enumerate(zip(m["five_text"], ("has", "dx", "dy", "lw", "lh"),
                                          strict=True)):
            b = on(u, 0.42 + 0.04 * i, op)
            fx = x1 + step * i
            self.box(sc, fx, ry, 44.0, 26.0, soft(DATA, 0.35 if i == 0 else 0.75), b,
                     text=v, ink=INK, border=DATA, size=SIZE, radius=4)
            self.text(sc, fx, ry + 0.042, name, b, SIZE, ink=QUIET)

    def meta_nots(self):
        return [tuple(q) for q in self.meta["not_squares"]]


# ====================================================================== 240
class TheDetector(Faces):
    """Three blocks with their real maps; the 1 x 1 convolution; 6,053 numbers; what one
    cell sees."""

    TITLE = "From pixels to a grid"
    PHASES = (("one block", 12.0), ("three blocks", 12.0), ("5 numbers a cell", 13.0),
              ("6,053 numbers", 10.0), ("what a cell sees", 12.0))
    LAST = "A grid of guesses"
    NEXT = "Two losses: is it there, and where"
    STATION = ("forward",) * 5
    SC = (0.165, 0.555)
    STX = 0.33                                  # the stacks' centre x
    LBX = 0.372                                 # their shapes, left edge
    TOP, BOTTOM = 0.362, 0.898                  # the stacks' column, top to bottom
    TILE = (1.0, 1.4, 2.0, 2.0)                 # px per map pixel
    BEHIND = (4, 5, 6, 3)                       # cards drawn behind the front map
    DEPTH = 2.0                                 # px between them
    LOW = 0.835                                 # the rows under the scene
    BIGY = 0.878                                # a number to look at, under the scene
    APPEAR = ((0, 0.4), (1, 0.12), (1, 0.55), (2, 0.1))     # each stack: (phase, u)
    GROW = (0.1, 0.22, 0.34, 0.46, 0.58, 0.7)   # phase 5: the field's steps

    def setup(self):
        m = self.meta = self.study.metadata
        self.read_scene()
        self.maps = [np.asarray(v, float) for v in m["maps"]]
        self.sure = np.asarray(m["sure"], float)
        self.r, self.c = (int(v) for v in m["face_cell"])
        # the four stacks spread down the column with equal gaps between them
        ext = [self.stack_size(i)[1] + self.DEPTH * self.BEHIND[i] for i in range(4)]
        gap = ((self.BOTTOM - self.TOP) / self.frac(0, 1.0)[1] - sum(ext)) / 3
        y, self.STY = self.TOP, []
        for i, e in enumerate(ext):
            top = y + self.frac(0, self.DEPTH * self.BEHIND[i])[1]
            self.STY.append(top + self.frac(0, self.stack_size(i)[1] / 2)[1])
            y += self.frac(0, e + gap)[1]
        st = m["steps_text"]
        self.CODE = (
            Code("fc-block", run=((0.0, 0), (0.15, 1), (0.3, 3), (0.5, 2)),
                 out=((0.42, st[0]),)),
            Code("fc-detector", run=((0.0, 1), (0.5, 2)),
                 out=((0.12, st[1]), (0.55, st[2]))),
            Code("fc-detector", run=((0.0, 3),), out=((0.1, st[3]),)),
            Code("fc-detector", run=((0.0, 1), (0.42, 2), (0.6, 3))),
            Code("fc-block", run=tuple((a, 1 + i % 2) for i, a in enumerate(self.GROW))))
        on_scene = measured("the trained detector, held-out", on=FACES)
        self.EVIDENCE = (on_scene, on_scene, on_scene,
                         exact("weights and biases, counted"),
                         exact("3 × 3 stamps and 2 × 2 pools: 22 pixels"))

    def heading_for(self, k, u):
        return ("block(): look closer, then halve",
                "three blocks: 96 → 48 → 24 → 12",
                "a 1 × 1 convolution: 5 numbers a cell",
                "%s numbers to learn" % self.meta["total_text"],
                "what one cell can see")[k]

    def subtitle_for(self, k):
        m = self.meta
        c = m["count_texts"]
        return [
            "block() reads the scene with 8 stamps of 3 × 3, keeps what is above 0, then "
            "halves it: 96 becomes 48.",
            "Two more blocks: 16 maps of 24 × 24, then 32 maps of 12 × 12, one map pixel per "
            "cell of the grid.",
            "A 1 × 1 convolution turns each cell's 32 numbers into 5. The first says a face: "
            "%s in this face's cell." % m["cell_sure_text"],
            "The detector learns %s + %s + %s + %s = %s numbers: the stamps' weights and "
            "biases." % (c[0], c[1], c[2], c[3], m["total_text"]),
            "Each 3 × 3 stamp and each halving widens what a cell sees: %s, then %d × %d "
            "pixels." % (", ".join(str(v) for v in m["reach_sizes"][:5]), m["field_side"],
                         m["field_side"]),
        ][k]

    # ------------------------------------------------------------------ the stacks
    def stack_size(self, i):
        """The front card of stack ``i``: ``(w, h)`` px."""
        side = len(self.maps[min(i, 2)]) * self.TILE[i]
        return side + 10, side + 14

    def draw_stack(self, sc, i, op):
        """Stack ``i`` (0-2: the blocks' maps; 3: the 5 numbers' maps): cards behind, the
        real front map on the front card."""
        if op <= 0.01:
            return
        fx, fy = self.STX, self.STY[i]
        w, h = self.stack_size(i)
        for j in reversed(range(1, self.BEHIND[i] + 1)):
            dx, dy = self.frac(self.DEPTH * j, -self.DEPTH * j)
            self.card(sc, fx + dx, fy + dy, w, h, op, lift=0.2 - 0.002 * j)
        self.card(sc, fx, fy, w, h, op, lift=0.21)
        if i < 3:
            v, lo, hi = self.maps[i], 0.0, max(float(self.maps[i].max()), 1e-6)
        else:
            v, lo, hi = self.sure, 0.0, 1.0
        side = len(v) * self.TILE[i]
        draw_image(self, sc, self.P(fx, fy + self.frac(0, 2.0)[1]), shades(v, DATA, lo, hi),
                   side, side, op, grid=0.0, lift=0.215)

    def cell_on_stack(self, i, r, c):
        """The layout centre of map pixel ``(r, c)`` on stack ``i``'s front card."""
        n = len(self.maps[min(i, 2)])
        t = self.frac(self.TILE[i], self.TILE[i])
        x0 = self.STX - n * t[0] / 2
        y0 = self.STY[i] + self.frac(0, 2.0)[1] - n * t[1] / 2
        return x0 + (c + 0.5) * t[0], y0 + (r + 0.5) * t[1]

    def shown(self, i, k, u):
        """How far stack ``i`` has appeared."""
        pk, pu = self.APPEAR[i]
        if k != pk:
            return 1.0 if k > pk else 0.0
        return smoothstep(pu, pu + 0.06, u)

    def steady(self, sc, k, u, op):
        m = self.meta
        self.scene_card(sc, op)
        self.text(sc, self.SC[0], self.card_box()[3] + 0.024, m["shape_texts"][0],
                  op, SIZE, ink=DATA_INK)
        _l, _t, right, _b = self.card_box()
        for i in range(4):
            a = op * self.shown(i, k, u)
            if a <= 0.01:
                continue
            self.draw_stack(sc, i, a)
            fy = self.STY[i]
            self.text(sc, self.LBX, fy - 0.012, m["shape_texts"][i + 1], a, SIZE, "lm",
                      ink=DATA_INK)
            w, h = self.stack_size(i)
            if i == 0:
                self.arrow(sc, (right + 0.004, fy), (self.STX - self.frac(w / 2 + 6)[0], fy),
                           AXIS, a, width=1.6)
            else:
                _pw, ph = self.stack_size(i - 1)
                top = self.STY[i - 1] + self.frac(0, ph / 2 + 3)[1]
                bottom = fy - self.frac(0, h / 2 + self.DEPTH * self.BEHIND[i] + 3)[1]
                self.arrow(sc, (self.STX - 0.02, top), (self.STX - 0.02, bottom), AXIS, a,
                           width=1.6)
            # the learned numbers of the layer that makes it, phase 4 on
            if k >= 3:
                b = op * (on(u, (0.1, 0.25, 0.42, 0.6)[i], 1.0) if k == 3 else 1.0)
                self.text(sc, self.LBX, fy + 0.025, "%s numbers" % m["count_texts"][i], b, SIZE,
                          "lm", ink=WEIGHT_INK)
        # the first of the 5 numbers laid over the scene, phase 3 on
        if k >= 2:
            a = op * (on(u, 0.45, 1.0) if k == 2 else 1.0)
            if k == 4:
                a *= 1.0 - 0.6 * smoothstep(0.02, 0.1, u)
            self.overlay(sc, a)

    def overlay(self, sc, op):
        """How sure each cell is of a face, as translucent blue over its 8 x 8 pixels."""
        if op <= 0.01:
            return
        for r in range(12):
            for c in range(12):
                p = float(self.sure[r, c])
                if p > 0.02:
                    self.fill(sc, (c * 8, r * 8, c * 8 + 8, r * 8 + 8), DATA, op * 0.62 * p)
        self.grid_lines(sc, op * 0.45)

    # ------------------------------------------------------------------ 1: one block
    def phase_1(self, sc, u, op):
        y = self.LOW
        x = 0.058
        parts = (("conv 3 × 3", 0.12, WEIGHT_INK), ("ReLU", 0.22, INK),
                 ("max 2 × 2", 0.32, DATA_INK))
        for i, (s, at, ink) in enumerate(parts):
            a = on(u, at, op)
            if i:
                self.text(sc, x, y, "→", a, SIZE, "lm", ink=QUIET)
                x += layout.text_width("→", SIZE) + 0.012
            self.text(sc, x, y, s, a, SIZE, "lm", ink=ink)
            x += layout.text_width(s, SIZE) + 0.012
        self.text(sc, 0.058, y + 0.045, "8 stamps: 8 maps, half as wide", on(u, 0.5, op), SIZE,
                  "lm", ink=QUIET)

    # ------------------------------------------------------------------ 2: three blocks
    def phase_2(self, sc, u, op):
        self.text(sc, 0.058, self.LOW, "each block halves the side", on(u, 0.2, op), SIZE, "lm")
        self.text(sc, 0.058, self.LOW + 0.045, "and makes more maps", on(u, 0.62, op), SIZE,
                  "lm", ink=QUIET)

    # ------------------------------------------------------------------ 3: 5 numbers a cell
    def phase_3(self, sc, u, op):
        m = self.meta
        r, c = self.r, self.c
        a = on(u, 0.05, op)
        # the cell on the scene, on the 32 maps and on the 5 maps
        self.frame(sc, (c * 8, r * 8, c * 8 + 8, r * 8 + 8), INK, a, width=2.4, lift=1.3)
        for i in (2, 3):
            fx, fy = self.cell_on_stack(i, r, c)
            t = self.TILE[i]
            self.rect(sc, self.P(fx, fy), t + 4, t + 4, INK, a if i == 2 else on(u, 0.2, op),
                      border=INK, border_w=2.0, radius=1, lift=1.3, fill_op=0.0)
        self.text(sc, 0.058, self.LOW, "this cell: 32 numbers → 5", on(u, 0.12, op), SIZE, "lm")
        self.text(sc, 0.058, self.LOW + 0.045, "a face here? %s" % m["cell_sure_text"],
                  on(u, 0.5, op), SIZE, "lm", strong=True)

    # ------------------------------------------------------------------ 4: 6,053 numbers
    def phase_4(self, sc, u, op):
        m = self.meta
        self.big(sc, 0.058, self.BIGY, m["total_text"], on(u, 0.75, op),
                 label="numbers to learn", size=BIG)

    # ------------------------------------------------------------------ 5: what a cell sees
    def phase_5(self, sc, u, op):
        m = self.meta
        r, c = self.r, self.c
        sizes = m["reach_sizes"][:6]
        a = on(u, 0.04, op)
        self.fill(sc, (c * 8, r * 8, c * 8 + 8, r * 8 + 8), DATA, a * 0.5)
        self.frame(sc, (c * 8, r * 8, c * 8 + 8, r * 8 + 8), INK, a, width=2.0, lift=1.3)
        # the field grows step by step, centred on the cell's centre
        n = sum(u >= g for g in self.GROW)
        if n:
            s = sizes[n - 1]
            cx, cy = c * 8 + 4, r * 8 + 4
            self.frame(sc, (cx - s / 2, cy - s / 2, cx + s / 2, cy + s / 2), WEIGHT_INK, a,
                       width=2.4, lift=1.32)
        steps = [(0.0, "1 pixel")] + [(g, "%d × %d pixels" % (s, s))
                                       for g, s in zip(self.GROW, sizes, strict=True)]
        self.running(sc, 0.058, self.BIGY, steps, u, a, label="one cell sees",
                     size=HEAD, ink=WEIGHT_INK)


ANIMATIONS = {"faces-a-box": ABox, "faces-the-grid": TheGrid, "faces-the-detector": TheDetector}

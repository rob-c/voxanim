"""Animations 241-244, the second half of "Finding faces: bounding boxes" in "Blocks and code"
(see :mod:`voxanim.blocks`), drawn from :mod:`voxanim.study_blocks_u`.

* :class:`TwoLosses` (``faces-two-losses``): training step 0's 16 made scenes turn into 16
  grids of 12 x 12 cells, 38 of 2,304 holding a face (1.6%); "there" counts every cell, face
  cells 10 times (14.4% of what it counts); "where" only the 38 face cells, one worked; the
  loss 0.853 + 5 x 0.438 = 3.044; the same run without the weight finds 72 of 412 unseen
  faces (AP 0.47) against 348 (AP 0.92).
* :class:`Training` (``faces-training-the-detector``): the two losses over 6,000 steps; the
  first test scene's grid of "a face here?" at steps 0, 100, 500, 2,000 and 6,000, and the
  boxes it keeps.
* :class:`OneBox` (``faces-nms``): the same scene, 4 cells above 0.5, decoded to boxes, three
  on one face; NMS at IoU 0.3 keeps 2 boxes for 2 faces.
* :class:`PrecisionRecall` (``faces-precision-and-recall``): matching at IoU 0.5 on a scene;
  348 of 412 faces found and 64 wrong boxes over 200 scenes; precision against recall as the
  confidence falls, AP 0.92, and the unweighted run's curve, AP 0.47.

A photo or a scene is drawn as it is, grey, on the series' blue data card, as in the other
face animations (:func:`voxanim.mech_blocks_t.gray`); grids and boxes keep their colours. Boxes
on a scene are guides, drawn in front of its tiles. Decoding, NMS and evaluation are not
training: the loop map lights no station there, and "forward" where a scene is run through
the detector.
"""

from __future__ import annotations

from typing import ClassVar

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
    commas,
    measured,
)
from .diagram import draw_image
from .easing import ease_in_out, smoothstep
from .mech import AXIS, TRACK, dark, soft
from .mech_blocks_t import gray

DATA_INK = dark(DATA, 0.3)
LOSS_INK = dark(LOSS, 0.3)
WHERE = dark(AXIS, 0.2)         # 242: the "where" curve, beside "there" in the loss's violet
SIDE = 96                       # a scene's side, in pixels
GRID = 12                       # cells across


def faces(what):
    """An evidence tag: measured on the Olivetti faces."""
    return measured(what, on="Olivetti faces")


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on(u, a, op, width=0.05):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


class Scenes(Blocks):
    """What the four share: scenes, boxes over them, grids of cells."""

    def at(self, k, u):
        """1 once phase ``k`` has reached ``u`` (fading in over a short while); 0 before
        phase ``k``."""
        a, b = self.spans[k]
        t0 = a + (b - a) * u
        return smoothstep(t0, t0 + 0.5, getattr(self, "_t", 0.0))

    def shown(self, k, kk, uu, u):
        """1 from ``u = uu`` of phase ``kk`` on (for something kept across phases)."""
        if k > kk:
            return 1.0
        if k < kk:
            return 0.0
        return self.at(kk, uu)

    # ------------------------------------------------------------------ a scene
    def place(self, fx, fy, side_px):
        """``(x, y) pixels of the scene -> layout (fx, fy)`` for a scene drawn ``side_px``
        px wide centred at ``(fx, fy)``."""
        w, h = self.frac(side_px, side_px)
        x0, y0 = fx - w / 2, fy - h / 2
        return lambda x, y: (x0 + w * x / SIDE, y0 + h * y / SIDE)

    def photo(self, sc, img, fx, fy, side_px, op, card=True, lift=0.2):
        """A photo or a scene (values 0..1) as it is, grey, centred at ``(fx, fy)`` on the
        series' blue data card (as groups t and w draw them); ``card=False``: a thin blue
        frame instead, for small photos side by side."""
        if op <= 0.01:
            return
        c = self.P(float(fx), float(fy))
        if card:            # the card's band is on top: the photo sits 3 px below its centre
            self.card(sc, fx, fy - self.frac(0, 3.0)[1], side_px + 14, side_px + 20, op,
                      lift=lift)
        else:
            self.rect(sc, c, side_px + 4, side_px + 4, soft(DATA, 0.82), op, border=DATA,
                      border_w=1.5, radius=2, lift=lift)
        draw_image(self, sc, c, gray(img), side_px, side_px, op, grid=0.0, lift=lift + 0.004)

    def box(self, sc, at, corners, ink, op, width=2.4):
        """A box (corners in scene pixels) outlined over the scene, in front of it."""
        if op <= 0.01:
            return
        x1, y1, x2, y2 = (float(v) for v in corners)
        (a, b), (c, d) = at(x1, y1), at(x2, y2)
        w, h = self.span(c - a, d - b)
        self.rect(sc, self.P((a + c) / 2, (b + d) / 2), w, h, ink, op, border=ink,
                  border_w=width, radius=2, lift=self.ARROW_LIFT, fill_op=0.0)

    def cell_lines(self, sc, at, op, ink=QUIET):
        """The 12 x 12 grid's lines over a scene."""
        if op <= 0.01:
            return
        for i in range(GRID + 1):
            v = i * SIDE / GRID
            self.polyline(sc, [self.P(*at(v, 0), self.ARROW_LIFT), self.P(*at(v, SIDE),
                                                                          self.ARROW_LIFT)],
                          1.0, ink, op * 0.55, bias=0.4)
            self.polyline(sc, [self.P(*at(0, v), self.ARROW_LIFT), self.P(*at(SIDE, v),
                                                                          self.ARROW_LIFT)],
                          1.0, ink, op * 0.55, bias=0.4)

    def cell_fill(self, sc, at, r, c, ink, op):
        """Cell ``(r, c)`` of the grid over a scene, tinted."""
        if op <= 0.01:
            return
        s = SIDE / GRID
        (a, b), (x, y) = at(c * s, r * s), at((c + 1) * s, (r + 1) * s)
        w, h = self.span(x - a, y - b)
        self.rect(sc, self.P((a + x) / 2, (b + y) / 2), w, h, ink, op, border=INK,
                  border_w=2.0, radius=1, lift=self.ARROW_LIFT - 0.05, fill_op=0.45)

    def hbar(self, sc, x0, y, w, h, ink, op, lift=0.3):
        """A flat bar from layout ``x0``, ``w`` layout wide, ``h`` px tall, centred on ``y``."""
        if op <= 0.01 or w <= 0:
            return
        self.rect(sc, self.P(x0 + w / 2, y), max(self.span(w)[0], 1.2), h, ink, op, radius=0,
                  lift=lift)


# ====================================================================== 241
class TwoLosses(Scenes):
    """2,304 cells, 38 faces; there over every cell, faces x 10; where on face cells only;
    without the weight the detector misses most faces."""

    TITLE = "Two losses: is it there, and where"
    PHASES = (("2,304 cells", 13.0), ("there", 14.0), ("where", 13.0), ("why the weight", 14.0))
    LAST = "From pixels to a grid"
    NEXT = "Training the detector"
    STATION = ("loss", "loss", "loss", None)
    BLOCK = (0.165, 0.625)          # the 4 x 4 block of scenes / grids: its centre
    TILE, GAP = 46.0, 6.0           # px
    RX = 0.295                      # the right-hand column
    METER = (0.452, 0.735)
    METER_H = 110.0
    METER_TOP = 3.5                 # the meter's full scale

    def setup(self):
        m = self.meta = self.study.metadata
        self.has = np.asarray(m["has"], float)
        self.thumbs = [np.asarray(t, float) for t in m["thumbs"]]
        n = int(round(np.sqrt(m["scenes"])))
        pitch = self.TILE + self.GAP
        dx, dy = self.frac(pitch, pitch)
        cx, cy = self.BLOCK
        self.centres = [(cx + (j % n - (n - 1) / 2) * dx, cy + (j // n - (n - 1) / 2) * dy)
                        for j in range(m["scenes"])]
        c = m["cell"]
        self.CODE = (
            Code("fc-loss", run=((0.0, 0),), out=((0.5, "obj: %s" % m["out_shape"]),)),
            Code(("fc-weight", "fc-loss"), pick=(0, 1, 2, 3), run=((0.0, 3), (0.32, 0),
                                                                   (0.62, 3)),
                 out=((0.7, "there: %s" % m["there_text"]),)),
            Code("fc-loss", pick=(3, 4, 5, 6, 7), run=((0.0, 1), (0.28, 3), (0.62, 4)),
                 out=((0.4, "where: %s" % m["where_text"]),
                      (0.68, "where: %s\nloss: %s" % (m["where_text"], m["loss_text"])))),
            Code(("fc-weight", "fc-loss"), pick=(0, 3), run=((0.0, 0),)))
        self.EVIDENCE = (faces("step 0's 16 scenes, counted"),
                         faces("step 0's batch, starting weights"),
                         faces("step 0's batch, starting weights"),
                         faces("200 unseen scenes, two runs"))
        self.cell_at = (c["scene"], c["row"], c["col"])

    def heading_for(self, k, u):
        return ("most cells hold no face",
                "there: every cell, faces count 10 times",
                "where: the box, on face cells only",
                "without the weight, faces are missed")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "Step 0's 16 made scenes have %s cells; only %d hold a face's centre, %s."
            % (m["cells_text"], m["face_cells"], m["face_share_text"]),
            "'there' scores every cell; face cells count %s times, so faces are %s of what it "
            "counts, not %s." % (m["weight_text"], m["weighted_share_text"],
                                 m["face_share_text"]),
            "'where' scores the box on the %d face cells only; at the start the loss is "
            "%s + 5 × %s = %s." % (m["face_cells"], m["there_text"], m["where_text"],
                                   m["loss_text"]),
            "Trained without the weight, it finds %d of %d unseen faces (AP %s); with it, %d "
            "(AP %s)." % (m["found_unweighted"], m["faces"], m["ap_unweighted_text"],
                          m["found"], m["ap_text"]),
        ][k]

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        self.block(sc, k, u, op)
        meter = op * self.kept(1, 2)
        if meter > 0.01:
            self.loss_meter(sc, k, u, meter)

    def block(self, sc, k, u, op):
        """The 16 scenes, turning into their grids of cells; in phase 3 the empty cells
        greyed (not counted)."""
        m = self.meta
        a = op * (self.at(0, 0.03) if k == 0 else 1.0)
        turn = eased(u, 0.36, 0.48) if k == 0 else 1.0
        grey = {2: eased(u, 0.06, 0.2), 3: 1.0 - eased(u, 0.02, 0.08)}.get(k, 0.0)
        cx, cy = self.BLOCK
        top = cy - self.frac(0, 2 * (self.TILE + self.GAP))[1]
        self.text(sc, cx, top - 0.024, "step 0: %d made scenes" % m["scenes"], a, SIZE)
        for j, (x, y) in enumerate(self.centres):
            if turn < 1.0:
                self.photo(sc, self.thumbs[j], x, y, self.TILE, a * (1.0 - turn), card=False)
            if turn > 0.0:
                h = self.has[j]
                zero = (h < 0.5) if grey > 0.5 else None
                self.tiles(sc, h, x, y, self.TILE, self.TILE, a * turn, lo=0.0, hi=1.0,
                           zero=zero, grid=0.0, lift=0.32)
        # phase 3: the cell worked out, ringed
        if k == 2:
            s, r, c = self.cell_at
            cell = self.cells(*self.centres[s], self.TILE, self.TILE, GRID, GRID)
            x, y = cell(r, c)
            ring = on(u, 0.3, op)
            if ring > 0.01:
                w = self.TILE / GRID + 5
                self.rect(sc, self.P(x, y), w, w, INK, ring, border=INK, border_w=2.2,
                          radius=2, lift=self.ARROW_LIFT, fill_op=0.0)
        bottom = cy + self.frac(0, 2 * (self.TILE + self.GAP))[1]
        b = op * (self.at(0, 0.64) if k == 0 else 1.0)
        self.text(sc, cx, bottom + 0.012, "%d of %s cells hold a face" % (m["face_cells"],
                                                                         m["cells_text"]),
                  b, SIZE, ink=DATA_INK)

    def loss_meter(self, sc, k, u, op):
        m = self.meta
        a = op * (self.at(1, 0.66) if k == 1 else 1.0)
        v = m["there"]
        if k == 2:
            v = m["there"] + (m["loss"] - m["there"]) * eased(u, 0.62, 0.74)
        label = m["there_text"] if (k == 1 or u < 0.68) else m["loss_text"]
        x, y = self.METER
        self.meter(sc, x, y, v / self.METER_TOP, a, h=self.METER_H, label=label)
        name = "loss" if k == 2 and u >= 0.68 else "there"
        self.text(sc, x, y - self.frac(0, self.METER_H / 2 + 16)[1], name, a, SIZE,
                  ink=LOSS_INK)

    # ------------------------------------------------------------------ 1: the cells
    def phase_1(self, sc, u, op):
        m = self.meta
        x = self.RX
        self.big(sc, x, 0.5, m["cells_text"], on(u, 0.52, op), label="16 × 144 cells")
        self.big(sc, x, 0.64, str(m["face_cells"]), on(u, 0.62, op), label="hold a face",
                 ink=DATA_INK)
        self.big(sc, x, 0.78, m["face_share_text"], on(u, 0.74, op), label="of the cells",
                 ink=DATA_INK)

    # ------------------------------------------------------------------ 2: there
    def phase_2(self, sc, u, op):
        m = self.meta
        x, w = self.RX, 0.13
        self.text(sc, x, 0.43, "what 'there' counts", on(u, 0.04, op), SIZE, "lm")
        grow = eased(u, 0.36, 0.5)
        share = m["face_share"] + (m["weighted_share"] - m["face_share"]) * grow
        a = on(u, 0.1, op)
        y = 0.49
        self.hbar(sc, x, y, w, 16.0, TRACK, a * 0.8, lift=0.25)
        self.hbar(sc, x, y, w * share, 16.0, DATA, a)
        self.running(sc, x, y + 0.05, [(0.0, "faces: %d of %s" % (m["face_cells"],
                                                               m["cells_text"])),
                                       (0.42, "faces × %s: %s of %s" % (
                                           m["weight_text"], m["weighted_face_text"],
                                           m["weighted_total_text"]))],
                     u, a, size=SIZE, ink=DATA_INK)
        self.running(sc, x, y + 0.1, [(0.0, m["face_share_text"]),
                                      (0.42, m["weighted_share_text"])], u, a, size=HEAD)
        self.text(sc, x, 0.7, "there = %s" % m["there_text"], on(u, 0.7, op), HEAD, "lm",
                  ink=LOSS_INK)
        self.text(sc, x, 0.745, "no weight: %s" % m["there_unweighted_text"], on(u, 0.78, op),
                  SIZE, "lm", ink=QUIET)

    # ------------------------------------------------------------------ 3: where
    def phase_3(self, sc, u, op):
        m = self.meta
        c = m["cell"]
        x = self.RX
        a = on(u, 0.3, op)
        self.text(sc, x, 0.43, "scene %d, cell (%d, %d)" % (c["scene"], c["row"], c["col"]),
                  a, SIZE, "lm")
        self.text(sc, x, 0.47, "want " + " ".join(c["want_text"]), on(u, 0.36, op), SIZE,
                  "lm", ink=DATA_INK)
        self.text(sc, x, 0.51, "says " + " ".join(c["said_text"]), on(u, 0.42, op), SIZE,
                  "lm")
        self.text(sc, x, 0.55, "miss = %s" % c["miss_text"], on(u, 0.48, op), SIZE, "lm")
        self.text(sc, x, 0.65, "where: mean of %d" % m["face_cells"], on(u, 0.56, op), SIZE,
                  "lm")
        self.text(sc, x, 0.69, "= %s" % m["where_text"], on(u, 0.56, op), HEAD, "lm",
                  ink=INK)
        self.text(sc, x, 0.77, "%s + 5 × %s" % (m["there_text"], m["where_text"]),
                  on(u, 0.66, op), SIZE, "lm", ink=LOSS_INK)
        self.text(sc, x, 0.815, "= %s" % m["loss_text"], on(u, 0.7, op), HEAD, "lm",
                  ink=LOSS_INK)
        self.text(sc, self.BLOCK[0], 0.875, "grey: not counted in where", on(u, 0.1, op),
                  SIZE, ink=QUIET)

    # ------------------------------------------------------------------ 4: why the weight
    def phase_4(self, sc, u, op):
        m = self.meta
        x, x0, full = self.RX, self.RX + 0.085, 0.075
        self.text(sc, x, 0.43, "AP, %d unseen scenes" % m["test_scenes"], on(u, 0.08, op),
                  SIZE, "lm")
        rows = (("faces × 10", m["ap"], m["ap_text"], 0.12, DATA, 0.475),
                ("no weight", m["ap_unweighted"], m["ap_unweighted_text"], 0.3, AXIS, 0.515),
                ("faces × 10", m["found"] / m["faces"], str(m["found"]), 0.5, DATA, 0.645),
                ("no weight", m["found_unweighted"] / m["faces"], str(m["found_unweighted"]),
                 0.62, AXIS, 0.685))
        for name, v, text, at, ink, y in rows:
            a = on(u, at, op)
            self.text(sc, x, y, name, a, SIZE, "lm")
            self.hbar(sc, x0, y, full, 14.0, TRACK, a * 0.6, lift=0.25)
            self.hbar(sc, x0, y, full * v * eased(u, at, at + 0.1), 14.0, ink, a)
            self.text(sc, x0 + full * v + 0.008, y, text, a, SIZE, "lm", ink=INK)
        self.text(sc, x, 0.6, "found at 0.5, of %d" % m["faces"], on(u, 0.44, op), SIZE, "lm")
        self.text(sc, x, 0.77, "one run each,", on(u, 0.78, op), SIZE, "lm", ink=QUIET)
        self.text(sc, x, 0.805, "same start, same scenes", on(u, 0.78, op), SIZE, "lm",
                  ink=QUIET)


# ====================================================================== 242
class Training(Scenes):
    """The two losses over 6,000 steps; a held-out scene's grid at five steps."""

    TITLE = "Training the detector"
    PHASES = (("the losses", 14.0), ("steps 0 and 100", 13.0), ("500 and 2,000", 13.0),
              ("step 6,000", 13.0))
    LAST = "Two losses: is it there, and where"
    NEXT = "One box per face"
    STATION = ("loss", "forward", "forward", "forward")
    CHART = ((0.11, 0.43), (0.41, 0.555))
    SCENE = (0.125, 0.78)
    SCENE_PX = 118.0
    PGRID = (0.275, 0.78)
    TEXT_X = 0.36
    SWAP = 0.5                      # where in phases 2 and 3 the second step comes in
    # which snapshot each phase shows, before and after SWAP
    SHOWS: ClassVar[dict] = {1: (0, 1), 2: (2, 3), 3: (4, 4)}

    def setup(self):
        m = self.meta = self.study.metadata
        self.canvas = np.asarray(m["canvas"], float)
        self.snaps = m["snaps"]
        self.ps = [np.asarray(s["p"], float) for s in self.snaps]
        self.chart_ = self.chart(self.CHART, [0.0, m["steps"]], [0.0, 0.9], pad=0.0)
        lines = ("fc-train", "fc-decode")
        pick = (0, 1, 2, 3, 4, 5, 6)
        s = self.snaps
        self.CODE = (
            Code(lines, pick=pick, run=((0.0, 0), (0.12, 1), (0.24, 3), (0.36, 4))),
            Code(lines, pick=pick, run=((0.0, 0), (0.2, 5)),
                 out=((0.1, "above 0.5: %d" % s[0]["above"]),
                      (self.SWAP, "above 0.5: %d" % s[1]["above"]))),
            Code(lines, pick=pick, run=((0.0, 5), (self.SWAP, 6)),
                 out=((0.0, "above 0.5: %d" % s[2]["above"]),
                      (self.SWAP, "above 0.5: %d" % s[3]["above"]))),
            Code(lines, pick=pick, run=((0.0, 5), (0.3, 6)),
                 out=((0.0, "above 0.5: %d" % s[4]["above"]),)))
        self.EVIDENCE = (faces("one run, 6,000 steps"),
                         faces("test scene 0, unseen people"),
                         faces("test scene 0, unseen people"),
                         faces("test scene 0, unseen people"))

    def heading_for(self, k, u):
        return ("both losses fall",
                "an unseen scene: is a face here?",
                "the faces light up",
                "two faces, two boxes")[k]

    def subtitle_for(self, k):
        m = self.meta
        s = self.snaps
        return [
            "Over %s steps of 16 new scenes, 'there' falls from %s to %s and 'where' from %s "
            "to %s." % (m["steps_text"], m["there_first_text"], m["there_last_text"],
                        m["where_first_text"], m["where_last_text"]),
            "On a scene of unseen people, every cell says %s at step 0; by step 100 no cell "
            "is above 0.5." % s[0]["p_max_text"],
            "At step 500 the surest cell says only %s; by 2,000 the faces pass 0.5, with %d "
            "boxes for %d faces." % (s[2]["p_max_text"], s[3]["boxes"], m["faces"]),
            "At step 6,000 the two faces get the scene's only boxes, %s and %s; the "
            "not-faces get none." % tuple(s[4]["score_texts"]),
        ][k]

    # ------------------------------------------------------------------ which step
    def snap_at(self, k, u):
        """``(index before, index after, how far)`` of the snapshot shown."""
        i, j = self.SHOWS[k]
        return i, j, eased(u, self.SWAP, self.SWAP + 0.06)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        self.curves(sc, k, u, op)
        keep = op * self.kept(1, 3)
        if keep > 0.01:
            self.held_out(sc, k, u, keep)

    def curves(self, sc, k, u, op):
        m = self.meta
        ch = self.chart_
        frame = op * (self.at(0, 0.03) if k == 0 else 1.0)
        self.axes(sc, ch, frame, xticks=[(0, "0"), (3000, "3,000"), (6000, "6,000")],
                  yticks=[(0.0, "0"), (0.4, "0.4"), (0.8, "0.8")], grid=False)
        self.text(sc, ch.x0 - 0.045, ch.y0 - 0.025, "loss", frame, SIZE, "lm")
        self.text(sc, ch.x1 + 0.008, ch.y1 - 0.018, "step", frame, SIZE, "lm")
        draw = eased(u, 0.1, 0.55) if k == 0 else 1.0
        xs = np.asarray(m["curve_x"])
        # labels: the start beside the curve, the end above where both lie flat
        for name, ink, at, start, end in (("there", LOSS, 0.6, (250, 0.85), 0.5),
                                          ("where", WHERE, 0.68, (1100, 0.44), 0.26)):
            ys = np.asarray(m[name])
            self.plot(sc, ch, xs, ys, ink, frame, draw, dot=True)
            lab = op * (self.at(0, at) if k == 0 else 1.0)
            self.text(sc, ch.fx(start[0]), ch.fy(start[1]), "%s %s" % (
                name, m[name + "_first_text"]), lab, SIZE, "lm", ink=dark(ink, 0.3))
            self.text(sc, ch.x1 - 0.01, ch.fy(end), "%s %s" % (name, m[name + "_last_text"]), lab,
                      SIZE, "rm", ink=dark(ink, 0.3))
        if k >= 1:                                       # where the grid below is
            i, j, s = self.snap_at(k, u)
            step = self.snaps[i]["step"] + (self.snaps[j]["step"] - self.snaps[i]["step"]) * s
            x = ch.fx(step)
            a = op * (self.at(1, 0.04) if k == 1 else 1.0)
            self.polyline(sc, [self.P(x, ch.y1 - 0.03, self.ARROW_LIFT),
                               self.P(x, ch.y1, self.ARROW_LIFT)], 2.4, INK, a, bias=0.5)
            self.disk(sc, self.P(x, ch.y1 - 0.03), 4.0, INK, a, lift=1.45, outline=0.0)

    def held_out(self, sc, k, u, op):
        m = self.meta
        a = op * (self.at(1, 0.03) if k == 1 else 1.0)
        fx, fy = self.SCENE
        self.text(sc, fx, fy - self.frac(0, self.SCENE_PX / 2)[1] - 0.045, "test scene %d"
                  % m["scene"], a, SIZE)
        self.photo(sc, self.canvas, fx, fy, self.SCENE_PX, a)
        at = self.place(fx, fy, self.SCENE_PX)
        i, j, s = self.snap_at(k, u)
        p = self.ps[i] * (1 - s) + self.ps[j] * s
        gx, gy = self.PGRID
        b = op * (self.at(1, 0.12) if k == 1 else 1.0)
        self.text(sc, gx, gy - self.frac(0, self.SCENE_PX / 2)[1] - 0.045, "a face here?", b,
                  SIZE)
        shown = self.snaps[j] if s > 0.5 else self.snaps[i]
        ring = []
        if shown["above"] <= 12:
            ring = [tuple(int(v) for v in np.argwhere(np.asarray(shown["p"]) > 0.5)[q])
                    for q in range(shown["above"])]
        self.tiles(sc, p, gx, gy, self.SCENE_PX, self.SCENE_PX, b, lo=0.0, hi=1.0, grid=0.25,
                   ring=ring)
        # the boxes kept, for the step shown
        for snap, w in ((self.snaps[i], 1 - s), (self.snaps[j], s)):
            if snap["step"] == 0 or w <= 0.01:
                continue
            for corners, right in zip(snap["found"], snap["right"], strict=True):
                self.box(sc, at, corners, INK if right else AXIS, b * w)
        # the step and what it says
        x = self.TEXT_X
        self.running(sc, x, gy - 0.05, [(0.0, "step " + self.snaps[i]["step_text"]),
                                        (self.SWAP, "step " + self.snaps[j]["step_text"])],
                     u if s < 1 else 1.0, b, size=HEAD)
        self.text(sc, x, gy, "above 0.5: %d" % shown["above"], b, SIZE, "lm", ink=DATA_INK)
        say = ("boxes: %d" % shown["boxes"]) if shown["step"] else "highest %s" % shown[
            "p_max_text"]
        self.text(sc, x, gy + 0.04, say, b, SIZE, "lm")

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.07, 0.69, "after step 0: means of %d steps" % m["window"],
                  on(u, 0.6, op), SIZE, "lm", ink=QUIET)
        self.text(sc, 0.07, 0.75, "loss = there + 5 × where", on(u, 0.7, op), SIZE, "lm")
        self.text(sc, 0.07, 0.81, "%s at the start, %s at the end" % (m["loss_first_text"],
                                                                      m["loss_last_text"]),
                  on(u, 0.76, op), SIZE, "lm", ink=LOSS_INK)

    def phase_2(self, sc, u, op):
        """The scene and its grid are in ``steady``."""

    def phase_3(self, sc, u, op):
        s = self.snaps[3]
        if s["boxes"] > s["right_count"]:
            self.text(sc, self.TEXT_X, self.PGRID[1] + 0.08, "%d too many"
                      % (s["boxes"] - s["right_count"]), on(u, self.SWAP + 0.15, op), SIZE,
                      "lm", ink=QUIET)

    def phase_4(self, sc, u, op):
        s = self.snaps[4]
        self.text(sc, self.TEXT_X, self.PGRID[1] + 0.08, "%d right" % s["right_count"],
                  on(u, 0.2, op), SIZE, "lm", ink=DATA_INK)


# ====================================================================== 243
class OneBox(Scenes):
    """4 cells above 0.5; decoded to boxes; three on one face; NMS keeps one per face."""

    TITLE = "One box per face"
    PHASES = (("cells above 0.5", 12.0), ("decode", 14.0), ("duplicates", 13.0),
              ("nms at 0.3", 14.0))
    LAST = "Training the detector"
    NEXT = "Is the detector any good?"
    STATION = ("forward", None, None, None)
    SCENE = (0.15, 0.635)
    SCENE_PX = 176.0
    RX = 0.27
    ROW_Y, ROW_STEP = 0.465, 0.042
    LOWER = 0.67                    # the right-hand column's lower part
    BOX_AT = (0.42, 0.5, 0.58, 0.66)   # phase 2: each box drawn, in score order
    PAIR_AT = (0.3, 0.55)           # phase 3: each duplicate lit beside the surest
    DROP_AT = (0.3, 0.45)           # phase 4: each dropped box goes

    def setup(self):
        m = self.meta = self.study.metadata
        self.canvas = np.asarray(m["canvas"], float)
        self.order = m["order"]
        self.top = self.order[0]
        self.dups = [j for j in self.order if j in m["claimed"] and j != self.top]
        self.dropped = [j for j in self.order if not m["fate"][j]["kept"]]
        self.CODE = (
            Code("fc-decode", run=((0.0, 0), (0.35, 1)),
                 out=((0.55, "%d cells above 0.5" % m["n_before"]),)),
            Code("fc-decode", run=((0.0, 2), (0.14, 3), (0.24, 4), (0.32, 5), (0.4, 6))),
            Code("fc-nms", run=((0.0, 0),)),
            Code("fc-nms", run=((0.0, 1), (0.62, 2)),
                 out=((0.62, "%d boxes kept of %d" % (m["n_after"], m["n_before"])),)))
        self.EVIDENCE = (faces("test scene %d, trained detector" % m["scene"]),
                         faces("test scene %d, trained detector" % m["scene"]),
                         exact_iou(m),
                         exact_iou(m))

    def heading_for(self, k, u):
        return ("cells that say: a face is here",
                "each cell's numbers become a box",
                "three boxes on one face",
                "keep the surest, drop its overlaps")[k]

    def subtitle_for(self, k):
        m = self.meta
        w = m["worked"]
        o = m["iou"]
        return [
            "The trained detector on the first test scene: %d of its 144 cells are above 0.5."
            % m["n_before"],
            "Cell (%d, %d) becomes a box %s wide, centred at (%s, %s) in pixels; then the "
            "others." % (w["row"], w["col"], w["w_text"], w["cx_text"], w["cy_text"]),
            "Three boxes sit on one face: they overlap the surest, %s, by %s." % (
                m["score_texts"][self.top],
                " and ".join("%.2f" % o[self.top][j] for j in self.dups)),
            "NMS keeps the surest box and drops any box overlapping it by more than %s: %d "
            "boxes for %d faces." % (m["nms_iou_text"], m["n_after"], m["faces"]),
        ][k]

    # ------------------------------------------------------------------ how a box looks
    def drawn(self, j, k, u):
        """How far box ``j`` is drawn (0..1), and how lit (0..1)."""
        m = self.meta
        if k == 0:
            return 0.0, 0.0
        rank = self.order.index(j)
        if k == 1:
            return smoothstep(self.BOX_AT[rank], self.BOX_AT[rank] + 0.04, u), \
                1.0 if j == self.top else 0.0
        if k == 2:
            lit = 1.0 if j == self.top else 0.0
            for at, d in zip(self.PAIR_AT, self.dups, strict=False):
                if j == d:
                    lit = smoothstep(at, at + 0.04, u)
            return 1.0, lit
        if m["fate"][j]["kept"]:
            return 1.0, 1.0
        at = self.DROP_AT[self.dropped.index(j)] if j in self.dropped else 0.3
        return 1.0 - smoothstep(at, at + 0.08, u), 0.0

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        a = op * (self.at(0, 0.03) if k == 0 else 1.0)
        fx, fy = self.SCENE
        half = self.frac(0, self.SCENE_PX / 2)[1]
        self.text(sc, fx, fy - half - 0.045, "test scene %d" % m["scene"], a, SIZE)
        self.photo(sc, self.canvas, fx, fy, self.SCENE_PX, a)
        at = self.place(fx, fy, self.SCENE_PX)
        grid = op * (self.at(0, 0.12) if k == 0 else 1.0)
        self.cell_lines(sc, at, grid * (1.0 - 0.6 * (k >= 2)))
        for rank, j in enumerate(self.order):
            r, c = m["cells"][j]
            lit = op * (self.at(0, 0.4 + 0.06 * rank) if k == 0 else 1.0)
            gone = 0.0 if k < 2 else (self.at(2, 0.0) if k == 2 else 1.0)
            self.cell_fill(sc, at, r, c, DATA, lit * (1.0 - gone))
        for j in self.order:
            d, lit = self.drawn(j, k, u)
            if d > 0.01:
                self.box(sc, at, m["corners"][j], INK if lit > 0.5 else DATA_INK, op * d,
                         width=2.8 if lit > 0.5 else 1.6)
        self.rows(sc, k, u, op)

    def rows(self, sc, k, u, op):
        """One row per cell above 0.5, surest first: its score, and from phase 3 its IoU
        with the surest, and from phase 4 whether it is kept."""
        m = self.meta
        x = self.RX
        a = op * (self.at(0, 0.4) if k == 0 else 1.0)
        self.text(sc, x, self.ROW_Y - self.ROW_STEP, "score · cell", a, SIZE, "lm", ink=QUIET)
        if k >= 2:
            self.text(sc, x + 0.1, self.ROW_Y - self.ROW_STEP, "IoU",
                      op * (self.at(2, 0.12) if k == 2 else 1.0), SIZE, "lm", ink=QUIET)
        for rank, j in enumerate(self.order):
            y = self.ROW_Y + self.ROW_STEP * rank
            r, c = m["cells"][j]
            b = op * (self.at(0, 0.4 + 0.06 * rank) if k == 0 else 1.0)
            d, lit = self.drawn(j, k, u)
            gone = k == 3 and not m["fate"][j]["kept"] and d < 0.5
            ink = QUIET if gone else (INK if lit > 0.5 else None)
            self.text(sc, x, y, "%s · (%d, %d)" % (m["score_texts"][j], r, c), b, SIZE, "lm",
                      ink=ink)
            if k >= 2 and j in self.dups:
                i = self.dups.index(j)
                c2 = op * (self.at(2, self.PAIR_AT[i]) if k == 2 else 1.0)
                self.text(sc, x + 0.1, y, "%.2f" % m["iou"][self.top][j], c2, SIZE, "lm",
                          ink=QUIET if gone else DATA_INK)
            if k == 3:
                kept = m["fate"][j]["kept"]
                at = 0.12 if kept else self.DROP_AT[self.dropped.index(j)]
                self.text(sc, x + 0.145, y, "kept" if kept else "dropped",
                          on(u, at, op), SIZE, "lm", ink=INK if kept else QUIET)

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        self.text(sc, self.SCENE[0], 0.875, "lit: above 0.5", on(u, 0.6, op), SIZE,
                  ink=DATA_INK)

    def phase_2(self, sc, u, op):
        w = self.meta["worked"]
        x, y = self.RX, self.LOWER
        self.text(sc, x, y, "sigmoid: %s, %s" % (w["sx_text"], w["sy_text"]), on(u, 0.06, op),
                  SIZE, "lm")
        self.text(sc, x, y + 0.042, "cx = (%d + %s) × 8 = %s" % (w["col"], w["sx_text"],
                                                                 w["cx_text"]),
                  on(u, 0.14, op), SIZE, "lm")
        self.text(sc, x, y + 0.084, "cy = (%d + %s) × 8 = %s" % (w["row"], w["sy_text"],
                                                                 w["cy_text"]),
                  on(u, 0.22, op), SIZE, "lm")
        self.text(sc, x, y + 0.126, "w = 16 × e^%.2f = %s" % (w["lw"], w["w_text"]),
                  on(u, 0.3, op), SIZE, "lm")
        self.text(sc, self.SCENE[0], 0.875, "%d boxes, in pixels" % self.meta["n_before"],
                  on(u, 0.7, op), SIZE, ink=DATA_INK)

    def phase_3(self, sc, u, op):
        m = self.meta
        x, y = self.RX, self.LOWER
        self.text(sc, x, y, "IoU with the surest, %s" % m["score_texts"][self.top],
                  on(u, 0.12, op), SIZE, "lm")
        self.text(sc, x, y + 0.05, "%d boxes, one face" % (len(self.dups) + 1),
                  on(u, 0.7, op), HEAD, "lm", strong=True)

    def phase_4(self, sc, u, op):
        m = self.meta
        x, y = self.RX, self.LOWER
        self.text(sc, x, y, "drop if IoU > %s" % m["nms_iou_text"], on(u, 0.06, op), SIZE,
                  "lm")
        self.text(sc, x, y + 0.05, "%d boxes, %d faces" % (m["n_after"], m["faces"]),
                  on(u, 0.6, op), HEAD, "lm", strong=True)
        self.text(sc, x, y + 0.1, "200 scenes: %s → %s boxes" % (
            commas(m["total_before"]), commas(m["total_after"])),
                  on(u, 0.76, op), SIZE, "lm", ink=QUIET)


def exact_iou(m):
    return faces("test scene %d; IoU, exact" % m["scene"])


# ====================================================================== 244
class PrecisionRecall(Scenes):
    """Matching at IoU 0.5; precision and recall at 0.5; the curve as the confidence falls;
    AP; the unweighted run beside it."""

    TITLE = "Is the detector any good?"
    PHASES = (("match at 0.5", 14.0), ("200 scenes", 13.0), ("the curve", 15.0),
              ("no weight", 13.0))
    LAST = "One box per face"
    NEXT = "A face is 4,096 numbers"
    STATION = (None, None, None, None)
    SCENE = (0.16, 0.635)
    SCENE_PX = 180.0
    RX = 0.3
    CHART = ((0.13, 0.44), (0.43, 0.76))
    SWEEP = (0.08, 0.6)             # phase 3: the confidence falls over this part
    AREA_AT = 0.68

    def setup(self):
        m = self.meta = self.study.metadata
        self.canvas = np.asarray(m["canvas"], float)
        self.w, self.uw = m["weighted"], m["unweighted"]
        self.chart_ = self.chart(self.CHART, [0.0, 1.0], [0.0, 1.0], pad=0.0)
        self.CODE = (
            Code("fc-iou", run=((0.0, 0), (0.2, 1), (0.4, 2))),
            Code("fc-decode", pick=(0, 1), run=((0.0, 1),)),
            Code("fc-decode", pick=(0, 1), run=((0.0, 1),)),
            Code(("fc-weight", "fc-decode"), pick=(0, 1, 2), run=((0.0, 0), (0.5, 2))))
        n = m["scenes"]
        self.EVIDENCE = (faces("test scene %d, unseen people" % m["scene"]),
                         faces("%d unseen scenes, counted" % n),
                         faces("%d unseen scenes, one run" % n),
                         faces("%d unseen scenes, two runs" % n))

    def heading_for(self, k, u):
        return ("a box is right if it overlaps a face",
                "how many found, how many wrong",
                "lower the bar: more found, more wrong",
                "without the weight")[k]

    def subtitle_for(self, k):
        m, w, uw = self.meta, self.w, self.uw
        return [
            "On test scene %d a box is right if it overlaps a face by 0.5 or more: %d right, "
            "%d on no face." % (m["scene"], m["right_here"], m["wrong_here"]),
            "Over %d unseen scenes at 0.5, %d of %d faces are found (recall) and %d of %d "
            "boxes are right (precision)." % (m["scenes"], w["found"], w["faces"], w["found"],
                                               w["boxes05"]),
            "Taking less sure boxes finds more faces but adds wrong ones; the area under the "
            "curve is AP, %s." % w["ap_text"],
            "Trained without the weight, it finds at most %s of the faces: AP %s, and %d "
            "found at 0.5." % (uw["last_recall_text"], uw["ap_text"], uw["found"]),
        ][k]

    # ------------------------------------------------------------------ handing over
    HAND = (0.02, 0.1)              # the last phase's picture leaves as the next comes

    def steady(self, sc, k, u, op):
        """Each picture stays into the start of the next phase, so no boundary is blank:
        the scene (phase 1) fades as the counts (2) come in beside it; the counts leave
        before the curves (3 and 4) come in where they were."""
        a, b = self.HAND
        leave = 1.0 - eased(u, a, b)
        if k == 0:
            self.matching(sc, u, op)
        elif k == 1:
            self.matching(sc, 1.0, op * leave)
            self.counts(sc, max(0.0, (u - a) / (1.0 - a)), op)
        elif k == 2:
            self.counts(sc, 1.0, op * leave)
            self.curves(sc, k, max(0.0, (u - b) / (1.0 - b)), op * smoothstep(b, b + 0.04, u))
        else:
            self.curves(sc, k, u, op)

    # ------------------------------------------------------------------ phase 1: matching
    def matching(self, sc, u, op):
        m = self.meta
        fx, fy = self.SCENE
        half = self.frac(0, self.SCENE_PX / 2)[1]
        a = on(u, 0.02, op)
        self.text(sc, fx, fy - half - 0.045, "test scene %d" % m["scene"], a, SIZE)
        self.photo(sc, self.canvas, fx, fy, self.SCENE_PX, a)
        at = self.place(fx, fy, self.SCENE_PX)
        t = on(u, 0.12, op)
        for corners in m["truth"]:
            self.box(sc, at, corners, DATA, t, width=1.4)
        x = self.RX
        self.text(sc, x, 0.45, "score · IoU", on(u, 0.3, op), SIZE, "lm", ink=QUIET)
        for i, (corners, right) in enumerate(zip(m["found"], m["right"], strict=True)):
            at_u = 0.3 + 0.1 * i
            b = on(u, at_u, op)
            self.box(sc, at, corners, INK if right else AXIS, b, width=2.6)
            y = 0.495 + 0.045 * i
            self.text(sc, x, y, "%s · %s" % (m["score_texts"][i], m["iou_texts"][i]), b, SIZE,
                      "lm")
            self.text(sc, x + 0.115, y, "right" if right else "wrong", on(u, at_u + 0.05, op),
                      SIZE, "lm", ink=DATA_INK if right else QUIET, strong=right)
        self.text(sc, fx, 0.875, "thin: true face · thick: found", on(u, 0.2, op), SIZE,
                  ink=QUIET)
        self.text(sc, x, 0.7, "right: IoU ≥ 0.5", on(u, 0.75, op), HEAD, "lm", strong=True)

    # ------------------------------------------------------------------ phase 2: counts
    def counts(self, sc, u, op):
        w = self.w
        x0, lab, full = 0.25, 0.06, 0.18
        self.text(sc, lab, 0.43, "at 0.5, over %d unseen scenes" % self.meta["scenes"],
                  on(u, 0.1, op), SIZE, "lm")
        rows = (("faces", "%d" % w["faces"], 1.0, DATA, 0.08),
                ("boxes", "%d" % w["boxes05"], w["boxes05"] / w["faces"], AXIS, 0.16),
                ("right", "%d" % w["found"], w["found"] / w["faces"], DATA, 0.26),
                ("wrong", "%d" % w["wrong"], w["wrong"] / w["faces"], AXIS, 0.34))
        for i, (name, text, frac, ink, at) in enumerate(rows):
            y = 0.48 + 0.045 * i
            a = on(u, at, op)
            self.text(sc, lab, y, name, a, SIZE, "lm")
            self.hbar(sc, x0 - 0.1, y, full * frac * eased(u, at, at + 0.08), 14.0, ink, a)
            self.text(sc, x0 - 0.1 + full * frac + 0.01, y, text, a, SIZE, "lm", ink=INK)
        y = 0.7
        self.text(sc, lab, y, "recall: %d of %d faces found" % (w["found"], w["faces"]),
                  on(u, 0.5, op), SIZE, "lm", ink=DATA_INK)
        self.text(sc, 0.46, y, w["recall05_text"], on(u, 0.5, op), HEAD, "rm", strong=True)
        self.text(sc, lab, y + 0.06, "precision: %d of %d boxes right" % (w["found"],
                                                                         w["boxes05"]),
                  on(u, 0.64, op), SIZE, "lm", ink=DATA_INK)
        self.text(sc, 0.46, y + 0.06, w["precision05_text"], on(u, 0.64, op), HEAD, "rm",
                  strong=True)

    # ------------------------------------------------------------------ the curves
    def curve(self, sc, d, n, ink, op, width=2.2):
        """Precision against recall for the first ``n`` boxes, surest first."""
        ch = self.chart_
        r, p = np.asarray(d["recall"]), np.asarray(d["precision"])
        n = int(np.clip(n, 0, len(r)))
        if n < 2 or op <= 0.01:
            return
        step = max(1, n // 240)
        idx = list(range(0, n, step)) + [n - 1]
        pts = [self.P(ch.fx(r[i]), ch.fy(p[i]), 1.0) for i in idx]
        self.polyline(sc, pts, width, ink, op, bias=0.5)

    def upto(self, d, keep):
        """How many boxes are at confidence ``keep`` or above."""
        return int(np.searchsorted(-np.asarray(d["scores"]), -keep, side="right"))

    def curves(self, sc, k, u, op):
        w = self.w
        ch = self.chart_
        frame = op
        self.axes(sc, ch, frame, xticks=[(0.0, "0%"), (0.5, "50%"), (1.0, "100%")],
                  yticks=[(0.5, "50%"), (1.0, "100%")], grid=False)
        self.text(sc, ch.x0 + 0.01, ch.y0 - 0.04, "precision: boxes right", frame, SIZE, "lm")
        self.text(sc, (ch.x0 + ch.x1) / 2, ch.y1 + 0.075, "recall: faces found", frame, SIZE)
        # the confidence falls from 1 to 0.05
        s0, s1 = self.SWEEP
        fall = eased(u, s0, s1) if k == 2 else 1.0
        keep = 1.0 - (1.0 - 0.05) * fall
        n = self.upto(w, keep) if fall < 1.0 else len(w["recall"])
        # the area under it: AP
        area = op * (smoothstep(self.AREA_AT, self.AREA_AT + 0.05, u) if k == 2 else 1.0)
        if area > 0.01:
            r, p = np.asarray(w["recall"]), np.asarray(w["precision"])
            cols = 60
            for c in range(cols):
                a, b = c / cols, (c + 1) / cols
                if a >= r[-1]:
                    break
                i = min(int(np.searchsorted(r, min(b, r[-1]), side="left")), len(r) - 1)
                top = float(p[i])
                x0 = ch.fx(a)
                wpx = self.span(ch.fx(min(b, r[-1])) - x0)[0]
                y0, y1 = ch.fy(top), ch.fy(0.0)
                hpx = self.span(0, y1 - y0)[1]
                self.rect(sc, self.P(x0 + self.frac(wpx / 2)[0], (y0 + y1) / 2), wpx + 0.6, hpx,
                          DATA, area * 0.22, radius=0, lift=0.2)
            self.text(sc, ch.fx(0.42), ch.fy(0.35), "AP = area = %s" % w["ap_text"], area, HEAD,
                      strong=True)
        self.curve(sc, w, n, DATA, frame)
        if k == 2 and 0.0 < fall < 1.0 and n > 0:
            i = n - 1
            self.disk(sc, self.P(ch.fx(w["recall"][i]), ch.fy(w["precision"][i])), 5.5, DATA,
                      frame, lift=1.4, outline=0.0, glow=0.4)
        # the readout under the chart, swapped in place as the confidence falls
        marks = w["marks"]
        steps = [(0.0, "from 1.0 down")]
        for mk in marks:
            f = (1.0 - mk["keep"]) / (1.0 - 0.05)
            at = s0 + (s1 - s0) * self._inverse_ease(f)
            steps.append((at, "from %s: found %s, right %s" % (mk["keep_text"],
                                                               mk["recall_text"],
                                                               mk["precision_text"])))
        if k == 2:
            self.running(sc, 0.06, 0.875, steps, u, frame, size=SIZE, ink=DATA_INK)
        # 0.5 marked
        m5 = next(mk for mk in marks if mk["keep"] == 0.5)
        at5 = s0 + (s1 - s0) * self._inverse_ease(0.5 / 0.95)
        g = op * (smoothstep(at5, at5 + 0.04, u) if k == 2 else 1.0)
        p5 = (ch.fx(m5["recall"]), ch.fy(m5["precision"]))
        self.disk(sc, self.P(*p5), 6.0, DATA, g, ring=INK, ring_w=2.0, lift=1.45, outline=0.0)
        self.text(sc, p5[0] - 0.012, p5[1] + 0.035, "0.5", g, SIZE, "rm", ink=DATA_INK)
        if k == 3:
            self.unweighted(sc, u, op)

    @staticmethod
    def _inverse_ease(f):
        """The ``x`` where :func:`ease_in_out` reaches ``f`` (by bisection)."""
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = (lo + hi) / 2
            if ease_in_out(mid) < f:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def unweighted(self, sc, u, op):
        uw = self.uw
        ch = self.chart_
        draw = eased(u, 0.1, 0.45)
        self.curve(sc, uw, int(len(uw["recall"]) * draw), AXIS, on(u, 0.08, op), width=2.4)
        end = (ch.fx(uw["recall"][-1]), ch.fy(uw["precision"][-1]))
        a = on(u, 0.48, op)
        self.text(sc, end[0] + 0.012, end[1] - 0.03, "no weight", a, SIZE, "lm", ink=QUIET)
        self.text(sc, end[0] + 0.012, end[1] + 0.005, "AP %s" % uw["ap_text"], a, SIZE, "lm",
                  ink=INK)
        m5 = next(mk for mk in uw["marks"] if mk["keep"] == 0.5)
        b = on(u, 0.6, op)
        p5 = (ch.fx(m5["recall"]), ch.fy(m5["precision"]))
        self.disk(sc, self.P(*p5), 6.0, AXIS, b, ring=INK, ring_w=2.0, lift=1.45, outline=0.0)
        self.text(sc, 0.06, 0.875, "at 0.5: %d found with the weight, %d without"
                  % (self.w["found"], uw["found"]), on(u, 0.66, op), SIZE, "lm", ink=INK)

    def phase_1(self, sc, u, op):
        """The scene and its boxes are in ``steady`` (:meth:`matching`)."""

    def phase_2(self, sc, u, op):
        """The counts are in ``steady`` (:meth:`counts`)."""

    def phase_3(self, sc, u, op):
        """The curve is in ``steady``."""

    def phase_4(self, sc, u, op):
        """The second curve is in ``steady``."""


ANIMATIONS = {"faces-two-losses": TwoLosses, "faces-training-the-detector": Training,
              "faces-nms": OneBox, "faces-precision-and-recall": PrecisionRecall}

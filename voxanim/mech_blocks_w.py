"""Animations 249-251, the last three of "Recognising faces" in "Blocks and code" (see
:mod:`voxanim.blocks`), drawn from :mod:`voxanim.study_blocks_w`.

* :class:`Identification` (``faces-identification``): one photo each of people 30-34
  enrolled, a row that stays at the top all through; a new photo's distance to each of
  them as five bars under it, with the threshold chosen on the training people across
  them: photo 305's nearest is person 30 at 0.18, under it; photo 301's nearest is person
  31 at 0.81, over it, so "unsure" (it is person 30). Then all 45 new photos, right or
  wrong, 36 of 45, the same as raw pixels; then the 50 photos of five strangers, sorted by
  their distance to the nearest enrolled face, 6 under the threshold: let in.
* :class:`Pipeline` (``faces-the-whole-pipeline``): the first seeded scene with three faces
  (picked by that rule), which stays on the left all through; the detector's three boxes;
  each face cut out and resized to 32 x 32; its 32 numbers, the nearest enrolled person and
  the distance (person 34's is over the threshold: unsure); then the true boxes in grey
  beside the detector's, and the rates over 200 scenes: 341 of 391 found, named right
  from the detector's box 67% (of those found), from the true box 82%.
* :class:`Limits` (``faces-what-40-people-cannot-show``): each unseen person's share of
  same-person pairs rejected (0 to 73%) and of other pairs accepted (0.2% to 3.8%), by
  number only; the whole dataset as 40 x 10 tiles; what it cannot measure, in words.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). Matching faces is not training: the loop map
lights "forward" only where a model runs (the encoder, the detector), and nothing else.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .blocks import DATA, HEAD, INK, QUIET, SIZE, Blocks, Code
from .diagram import draw_image
from .easing import ease_in_out, smoothstep
from .mech import AXIS, DIM, PAPER, dark, soft
from .study_blocks_w import about, measured

DATA_INK = dark(DATA, 0.3)
PALE = soft(DATA, 0.55)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on(u, a, op, width=0.05):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


def gray(img):
    """A photo in 0..1 as an RGB texture, exactly its grey levels."""
    a = np.clip(np.asarray(img, np.float32), 0.0, 1.0)
    return np.repeat(a[..., None], 3, axis=2)


class Faces(Blocks):
    """What the three share."""

    def at(self, k, u):
        """1 once phase ``k`` has reached ``u`` (fading in over half a second); 0 before."""
        a, b = self.spans[k]
        t0 = a + (b - a) * u
        return smoothstep(t0, t0 + 0.5, self._t)

    def photo(self, sc, img, fx, fy, side, op, ink=DATA, lift=0.2, border_w=2.0):
        """A photo, as it is (grey), ``side`` px square in a blue data frame."""
        if op <= 0.01:
            return
        c = self.P(float(fx), float(fy))
        self.rect(sc, c, side + 8, side + 8, soft(ink, 0.82), op, border=ink,
                  border_w=border_w, radius=4, lift=lift)
        draw_image(self, sc, c, gray(img), side, side, op, grid=0.0, lift=lift + 0.01)

    def column(self, sc, fx, base, h, w, ink, op, lift=0.3):
        """A bar standing on layout ``base``, ``h`` layout tall, ``w`` px wide."""
        if op <= 0.01:
            return
        hp = max(self.span(0, h)[1], 1.2)
        self.rect(sc, self.P(fx, base - self.frac(0, hp / 2)[1]), w, hp, ink, op, radius=0,
                  lift=lift)

    def bar(self, sc, x0, y, w, h, ink, op, lift=0.3):
        """A flat bar from layout ``x0``, ``w`` layout wide, ``h`` px tall, centred on ``y``."""
        if op <= 0.01 or w <= 0:
            return
        self.rect(sc, self.P(x0 + w / 2, y), max(self.span(w)[0], 1.2), h, ink, op, radius=0,
                  lift=lift)

    def guide(self, sc, a, b, ink, op, width=1.8):
        """A straight guide line in front of the blocks (no arrowhead)."""
        if op <= 0.01:
            return
        lift = self.ARROW_LIFT
        self.polyline(sc, [self.P(a[0], a[1], lift), self.P(b[0], b[1], lift)], width, ink, op,
                      bias=0.5)

    def dashed(self, sc, a, b, ink, op, width=1.6, dash=0.012):
        """A dashed guide from layout ``a`` to ``b``."""
        if op <= 0.01:
            return
        n = max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / dash))
        for i in range(0, n, 2):
            s0, s1 = i / n, min((i + 1) / n, 1.0)
            self.guide(sc, (a[0] + (b[0] - a[0]) * s0, a[1] + (b[1] - a[1]) * s0),
                       (a[0] + (b[0] - a[0]) * s1, a[1] + (b[1] - a[1]) * s1), ink, op, width)


# ====================================================================== 249
class Identification(Faces):
    """The gallery; one new photo named after the nearest; one too far: unsure; all 45;
    the strangers."""

    TITLE = "Who is it?"
    PHASES = (("enrol", 11.0), ("nearest", 12.0), ("unsure", 12.0), ("all 45", 12.0),
              ("strangers", 12.0))
    LAST = "Same person or not?"
    NEXT = "Find, cut out, compare"
    STATION = ("forward", None, None, None, None)
    GX = (0.215, 0.27, 0.325, 0.38, 0.435)     # the five enrolled photos
    GY, SIDE = 0.45, 38.0
    LABEL_Y = 0.37
    STRIP_Y = (0.565, 0.84)                      # phase 1: each enrolled photo's 32 numbers
    PROBE = (0.1, 0.45)
    BASE, TOP = 0.86, 0.6                        # the bars: d = 0 here, D_MAX at TOP
    D_MAX = 1.8
    BAR_W = 22.0
    LINE_X = (0.18, 0.47)                        # the threshold across the bars
    LEFT = 0.06                                  # what is said on the left
    TILE_Y = (0.575, 0.84)                       # phase 4: nine tiles under each photo
    STRANGE_X = (0.19, 0.465)                    # phase 5: fifty sorted bars

    def setup(self):
        m = self.meta = self.study.metadata
        a, b = m["sure_probe"], m["unsure_probe"]
        self.CODE = (
            Code(("fc-verify", "fc-identify"), pick=(0, 6, 7, 8, 9, 10, 4),
                 run=((0.0, 1), (0.45, 0))),
            Code(("fc-verify", "fc-identify"), pick=(0, 6, 7, 8, 9, 10, 4),
                 run=((0.0, 2), (0.12, 3), (0.3, 4), (0.55, 5))),
            Code(("fc-verify", "fc-identify"), pick=(0, 6, 7, 8, 9, 10, 4),
                 run=((0.0, 4), (0.45, 6))),
            Code(("fc-verify", "fc-identify"), pick=(0, 6, 7, 8, 9, 10, 4),
                 run=((0.0, 5),)),
            Code("fc-strangers", run=((0.0, 0), (0.18, 1), (0.36, 2), (0.6, 3)),
                 out=((0.66, "let in: tensor(%d)" % m["let_in"]),)))
        self.EVIDENCE = (measured("people 30-34, one photo each"),
                         measured("photo %d against the five" % a["photo"]),
                         measured("photo %d against the five" % b["photo"]),
                         measured("45 new photos of people 30-34"),
                         measured("50 photos of people 35-39"))

    def heading_for(self, k, u):
        return ("enrol one photo of each person",
                "a new photo: which enrolled face is nearest?",
                "nearest, but too far: unsure",
                "all 45 new photos",
                "strangers: people never enrolled")[k]

    def subtitle_for(self, k):
        m = self.meta
        a, b = m["sure_probe"], m["unsure_probe"]
        return [
            "One photo each of people 30 to 34 is enrolled. The encoder never trained on "
            "these people.",
            "Photo %d is %s from person %d's enrolled photo, under the threshold of %s: "
            "named person %d." % (a["photo"], a["nearest_text"], a["guess"],
                                  m["threshold_text"], a["guess"]),
            "Photo %d's nearest is person %d, at %s: over the threshold, so unsure. It is "
            "person %d." % (b["photo"], b["guess"], b["nearest_text"], b["person"]),
            "Right for %d of %d new photos, and raw pixels also get %d: the encoder helps "
            "verification, not this." % (m["identified"], m["probes"], m["pixels_identified"]),
            "%d photos of people 35 to 39, never enrolled: %d are under the threshold and "
            "would be let in." % (m["stranger_count"], m["let_in"]),
        ][k]

    # ------------------------------------------------------------------ geometry
    def y_of(self, d):
        return self.BASE - (self.BASE - self.TOP) * float(d) / self.D_MAX

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        # the gallery: the five enrolled photos, one at a time, then all through
        self.text(sc, self.GX[0] - 0.03, self.LABEL_Y, "enrolled people",
                  op * (self.at(0, 0.04) if k == 0 else 1.0), SIZE, "lm")
        for i, (x, img, p) in enumerate(zip(self.GX, m["gallery_images"], m["gallery_people"],
                                            strict=True)):
            a = op * (self.at(0, 0.1 + 0.08 * i) if k == 0 else 1.0)
            lit = self.lit(i, k, u)
            self.photo(sc, img, x, self.GY, self.SIDE, a, border_w=2.0 + 1.6 * lit)
            self.text(sc, x, self.GY + 0.06, str(p), a, SIZE, ink=INK if lit > 0.5 else None)
        # the threshold, across the bars (phases 2-3 and 5)
        t = op * max(self.kept(1, 2), self.kept(4, 4))
        if t > 0.01:
            y = self.y_of(m["threshold"])
            self.dashed(sc, (self.LINE_X[0], y), (self.LINE_X[1], y), AXIS, t)
            self.text(sc, self.LEFT, y, "threshold %s" % m["threshold_text"], t, SIZE, "lm",
                      ink=INK)
        # one new photo's five distances (phases 2-3), the bars moving from one to the next
        b = op * self.kept(1, 2)
        if b > 0.01:
            self.distances(sc, k, u, b)

    def lit(self, i, k, u):
        """How lit enrolled photo ``i`` is: the nearest to the new photo of phases 2-3."""
        m = self.meta
        if k == 1:
            return smoothstep(0.55, 0.6, u) * (i == m["sure_probe"]["guess"] - 30)
        if k == 2:
            return smoothstep(0.3, 0.35, u) * (i == m["unsure_probe"]["guess"] - 30)
        return 0.0

    def distances(self, sc, k, u, op):
        m = self.meta
        a, b = m["sure_probe"], m["unsure_probe"]
        grow = eased(u, 0.32, 0.5) if k == 1 else 1.0
        swap = eased(u, 0.06, 0.24) if k == 2 else 0.0
        appear = op * (self.at(1, 0.3) if k == 1 else 1.0)
        # the new photo, the one of phase 2 turning into the one of phase 3
        for p, w in ((a, 1.0 - swap), (b, swap)):
            if w > 0.01:
                self.photo(sc, p["image"], *self.PROBE, 44.0, op * w)
        self.text(sc, self.PROBE[0], self.LABEL_Y, "new photo", op, SIZE)
        self.running(sc, self.PROBE[0], self.PROBE[1] + 0.075,
                     ((0.0, "photo %d" % a["photo"]), (1.0, "photo %d" % b["photo"])),
                     1.0 if (k == 2 and u > 0.15) else 0.0, op, size=SIZE, anchor="mm")
        for i, x in enumerate(self.GX):
            d = (a["d"][i] + (b["d"][i] - a["d"][i]) * swap) * grow
            near = (a if k == 1 else b)["guess"] - 30
            lit = self.lit(i, k, u)
            sure = (a if k == 1 else b)["sure"]
            ink = (DATA if sure else PALE) if i == near else DIM
            ink = tuple(float(c) for c in np.asarray(DIM) + (np.asarray(ink) - DIM) * lit)
            self.column(sc, x, self.BASE, (self.BASE - self.TOP) * d / self.D_MAX, self.BAR_W,
                        ink, appear)
            txt = ((0.0, a["d_text"][i]), (0.15, b["d_text"][i])) if k == 2 else \
                ((0.0, a["d_text"][i]),)
            self.running(sc, x, self.y_of(d) - 0.025, txt, u if k == 2 else 0.0,
                         appear * (1.0 if k == 2 else smoothstep(0.42, 0.5, u)), size=SIZE,
                         anchor="mm", ink=INK if lit > 0.5 else None)
        self.text(sc, self.GX[0] - 0.035, self.BASE + 0.025, "distance", appear, SIZE, "rm",
                  ink=QUIET)
        self.guide(sc, (self.LINE_X[0], self.BASE), (self.LINE_X[1], self.BASE), AXIS,
                   appear, width=1.2)

    # ------------------------------------------------------------------ 1
    def phase_1(self, sc, u, op):
        """The gallery is in ``steady``; here, each enrolled photo's 32 numbers."""
        m = self.meta
        y0, y1 = self.STRIP_Y
        h = self.span(0, y1 - y0)[1]
        for i, (x, q) in enumerate(zip(self.GX, m["gallery_vectors"], strict=True)):
            a = on(u, 0.48 + 0.06 * i, op)
            self.tiles(sc, np.asarray(q)[:, None], x, (y0 + y1) / 2, 12.0, h, a, lo=-0.5,
                       hi=0.5, grid=0.0)
        self.text(sc, self.LEFT, 0.65, "%d numbers" % m["dims"], on(u, 0.48, op), HEAD, "lm",
                  strong=True)
        self.text(sc, self.LEFT, 0.7, "each", on(u, 0.48, op), HEAD, "lm", strong=True)

    # ------------------------------------------------------------------ 2-3
    def phase_2(self, sc, u, op):
        m = self.meta
        a = m["sure_probe"]
        self.text(sc, self.LEFT, 0.6, "nearest:", on(u, 0.6, op), SIZE, "lm")
        self.text(sc, self.LEFT, 0.65, "person %d" % a["guess"], on(u, 0.66, op), HEAD, "lm",
                  strong=True)

    def phase_3(self, sc, u, op):
        m = self.meta
        b = m["unsure_probe"]
        self.text(sc, self.LEFT, 0.6, "nearest: %d" % b["guess"], on(u, 0.35, op), SIZE, "lm")
        self.text(sc, self.LEFT, 0.65, "unsure", on(u, 0.5, op), HEAD, "lm", strong=True)
        self.text(sc, self.LEFT, 0.82, "it is person %d" % b["person"], on(u, 0.72, op), SIZE,
                  "lm", ink=QUIET)

    # ------------------------------------------------------------------ 4: all 45
    def phase_4(self, sc, u, op):
        m = self.meta
        y0, y1 = self.TILE_Y
        step = (y1 - y0) / 8
        for p, x in enumerate(self.GX):
            for j in range(9):
                i = p * 9 + j
                a = on(u, 0.06 + 0.035 * p + 0.004 * j, op)
                if a <= 0.01:
                    continue
                c = self.P(x, y0 + step * j)
                if m["right"][i]:
                    self.rect(sc, c, 15.0, 15.0, DATA, a, border=DATA_INK, border_w=1.0,
                              radius=2, lift=0.3)
                else:
                    self.rect(sc, c, 15.0, 15.0, PAPER, a, border=AXIS, border_w=1.8,
                              radius=2, lift=0.3, fill_op=0.0)
        k = on(u, 0.3, op)
        self.text(sc, self.LEFT, 0.6, "nearest is right", k, SIZE, "lm")
        self.text(sc, self.LEFT, 0.65, "%d of %d" % (m["identified"], m["probes"]), k, HEAD, "lm",
                  strong=True)
        self.rect(sc, self.P(self.LEFT + 0.008, 0.7), 12.0, 12.0, DATA, k,
                  border=DATA_INK, border_w=1.0, radius=2, lift=0.3)
        self.text(sc, self.LEFT + 0.022, 0.7, "right", k, SIZE, "lm", ink=QUIET)
        self.rect(sc, self.P(self.LEFT + 0.008, 0.74), 12.0, 12.0, PAPER, k, border=AXIS,
                  border_w=1.8, radius=2, lift=0.3, fill_op=0.0)
        self.text(sc, self.LEFT + 0.022, 0.74, "wrong", k, SIZE, "lm", ink=QUIET)
        p = on(u, 0.55, op)
        self.text(sc, self.LEFT, 0.8, "raw pixels", p, SIZE, "lm")
        self.text(sc, self.LEFT, 0.845, "%d of %d" % (m["pixels_identified"], m["probes"]), p,
                  HEAD, "lm", strong=True)

    # ------------------------------------------------------------------ 5: strangers
    def phase_5(self, sc, u, op):
        m = self.meta
        near = m["stranger_sorted"]
        n = len(near)
        x0, x1 = self.STRANGE_X
        w = self.span((x1 - x0) / n)[0] - 1.6
        let = m["let_in"]
        grow = eased(u, 0.1, 0.32)
        lit = smoothstep(0.42, 0.48, u)
        for i, d in enumerate(near):
            x = x0 + (x1 - x0) * (i + 0.5) / n
            ink = DATA if i < let else DIM
            ink = tuple(float(c) for c in np.asarray(DIM) + (np.asarray(ink) - DIM) * lit)
            self.column(sc, x, self.BASE, (self.BASE - self.TOP) * d * grow / self.D_MAX, w,
                        ink, on(u, 0.08, op))
        self.guide(sc, (self.LINE_X[0], self.BASE), (self.LINE_X[1], self.BASE), AXIS,
                   on(u, 0.08, op), width=1.2)
        self.text(sc, x0, self.BASE + 0.025, "one bar per photo, nearest first",
                  on(u, 0.12, op), SIZE, "lm", ink=QUIET)
        xl = x0 + (x1 - x0) * let / n
        b = on(u, 0.48, op)
        self.text(sc, x0, self.y_of(m["threshold"]) - 0.06,
                  "%d of %d let in" % (let, m["stranger_count"]), b, SIZE, "lm", ink=DATA_INK)
        self.guide(sc, (x0, self.y_of(m["threshold"]) - 0.035),
                   (xl, self.y_of(m["threshold"]) - 0.035), DATA_INK, b, width=2.0)
        self.text(sc, self.LEFT, 0.6, "people %d-%d" % (m["strangers_people"][0],
                                                        m["strangers_people"][-1]),
                  on(u, 0.04, op), SIZE, "lm")
        self.text(sc, self.LEFT, 0.64, "never enrolled", on(u, 0.04, op), SIZE, "lm",
                  ink=QUIET)


# ====================================================================== 250
class Pipeline(Faces):
    """The scene; the detector's boxes; each face cut out; its 32 numbers and the nearest
    enrolled person; the rates over 200 scenes."""

    TITLE = "Find, cut out, compare"
    PHASES = (("the scene", 11.0), ("find", 12.0), ("cut out", 12.0), ("compare", 13.0),
              ("200 scenes", 12.0))
    LAST = "Who is it?"
    NEXT = "What 40 people can't tell you"
    STATION = (None, "forward", None, "forward", None)
    SCENE = (0.155, 0.6)             # the scene's centre
    SCALE = 1.9                      # px per scene pixel
    CROP_X, CROP_SIDE = 0.315, 40.0
    ROWS = (0.46, 0.6, 0.74)         # the three faces' rows on the right
    STRIP = (0.35, 0.47)             # their 32 numbers
    RIGHT = 0.29                     # phase 5's numbers

    def setup(self):
        m = self.meta = self.study.metadata
        # the faces from top to bottom of the scene, as the rows on the right
        self.order = sorted(range(len(m["faces"])), key=lambda i: m["faces"][i]["box"][1])
        pipe = Code("fc-pipeline")
        self.CODE = (
            Code("fc-paste", run=((0.0, 0), (0.3, 2), (0.6, 3))),
            Code(("fc-decode", "fc-nms"), pick=(0, 1, 6, 7, 8, 9),
                 run=((0.0, 0), (0.2, 1), (0.35, 2), (0.5, 3), (0.62, 4), (0.75, 5))),
            Code(pipe.region, run=((0.0, 0), (0.25, 1), (0.5, 2))),
            Code(pipe.region, run=((0.0, 4), (0.4, 5), (0.6, 6))),
            Code(pipe.region, run=((0.0, 1),)))
        self.EVIDENCE = (measured("the first scene with 3 faces"),
                         measured("the trained detector, this scene"),
                         measured("the detector's 3 boxes, cut out"),
                         measured("the trained encoder, 3 faces"),
                         measured("200 scenes of people 30-34"))

    def heading_for(self, k, u):
        return ("a new scene of enrolled people",
                "find: the detector's boxes",
                "cut out each face, 32 × 32",
                "compare: 32 numbers, the nearest person",
                "200 scenes: how often it works")[k]

    def subtitle_for(self, k):
        m = self.meta
        f = [m["faces"][i] for i in self.order]
        unsure = next(x for x in f if not x["sure"])
        return [
            "Faces pasted on paper, so the true boxes are exact. Picked by a rule: the first "
            "scene with %d faces." % m["faces_in_scene"],
            "The detector finds all %d faces, sure of each: %s." % (
                len(f), ", ".join(x["score_text"] for x in f)),
            "Each box is cut out of the scene and resized to 32 × 32, the size the encoder "
            "trained on.",
            "All %d are named right; person %d's distance, %s, is over the threshold: unsure."
            % (m["named_right"], unsure["person"], unsure["distance_text"]),
            "Over %d scenes: named right from the detector's boxes %s of the faces found, from "
            "the true boxes %s." % (m["scenes"], m["det_text"], m["true_text"]),
        ][k]

    # ------------------------------------------------------------------ geometry
    def spot(self, x, y):
        """Layout point of scene pixel ``(x, y)``."""
        half = 48.0 * self.SCALE
        dx, dy = self.frac(x * self.SCALE - half, y * self.SCALE - half)
        return self.SCENE[0] + dx, self.SCENE[1] + dy

    def outline(self, sc, box, ink, op, width=2.2):
        """A box (corners in scene pixels) drawn in front of the scene."""
        if op <= 0.01:
            return
        (x0, y0), (x1, y1) = self.spot(box[0], box[1]), self.spot(box[2], box[3])
        w, h = self.span(x1 - x0, y1 - y0)
        self.rect(sc, self.P((x0 + x1) / 2, (y0 + y1) / 2), w, h, ink, op, border=ink,
                  border_w=width, radius=1, lift=self.ARROW_LIFT, fill_op=0.0)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        side = 96 * self.SCALE
        self.rect(sc, self.P(*self.SCENE), side + 8, side + 8, soft(DATA, 0.82), op,
                  border=DATA, border_w=2.0, radius=4, lift=0.2)
        draw_image(self, sc, self.P(*self.SCENE), gray(m["canvas"]), side, side, op, grid=0.0,
                   lift=0.21)
        below = self.SCENE[1] + self.frac(0, side / 2 + 22)[1]
        self.text(sc, self.SCENE[0], 0.385, "faces pasted on paper", op, SIZE)
        self.running(sc, self.SCENE[0], below, ((0, "the first scene with 3 faces"),
                                                (4, "grey: the true boxes")),
                     k, op, size=SIZE, anchor="mm", ink=QUIET)
        # the detector's boxes, from phase 2 on
        boxes = op * self.kept(1)
        if boxes > 0.01:
            for n, i in enumerate(self.order):
                f = m["faces"][i]
                a = boxes * (self.at(1, 0.3 + 0.12 * n) if k == 1 else 1.0)
                self.outline(sc, f["box"], INK, a)
                x, y = self.spot((f["box"][0] + f["box"][2]) / 2, f["box"][1])
                self.text(sc, x, y - 0.018, f["score_text"], a * self.kept(1, 2), SIZE,
                          ink=INK)
                self.text(sc, x, y - 0.018, str(f["person"]),
                          boxes * self.kept(3) * (self.at(3, 0.55 + 0.1 * n) if k == 3 else 1.0),
                          SIZE, ink=INK)
        # the rows on the right: each face cut out (phase 3) and compared (phase 4)
        rows = op * self.kept(2, 3)
        if rows > 0.01:
            self.rows(sc, k, u, rows)

    def rows(self, sc, k, u, op):
        m = self.meta
        self.text(sc, self.CROP_X, 0.385, "32 × 32", op * (self.at(2, 0.1) if k == 2 else 1.0),
                  SIZE)
        if k >= 3:
            self.text(sc, self.STRIP[0], 0.385, "person · distance", op * self.at(3, 0.5),
                      SIZE, "lm")
        for n, i in enumerate(self.order):
            f = m["faces"][i]
            y = self.ROWS[n]
            a = op * (self.at(2, 0.15 + 0.2 * n) if k == 2 else 1.0)
            if a <= 0.01:
                continue
            # the crop flies from its box to its row, growing to the size shown
            fly = eased(u, 0.15 + 0.2 * n, 0.28 + 0.2 * n) if k == 2 else 1.0
            bx, by = self.spot((f["box"][0] + f["box"][2]) / 2, (f["box"][1] + f["box"][3]) / 2)
            side0 = (f["box"][2] - f["box"][0]) * self.SCALE
            self.photo(sc, f["crop"], bx + (self.CROP_X - bx) * fly, by + (y - by) * fly,
                       side0 + (self.CROP_SIDE - side0) * fly, a, lift=0.2 + 1.3 * (fly < 1))
            if k < 3:
                continue
            s = op * self.at(3, 0.1 + 0.15 * n)
            w = self.span(self.STRIP[1] - self.STRIP[0])[0]
            q = np.atleast_2d(f["q"])
            self.tiles(sc, q, (self.STRIP[0] + self.STRIP[1]) / 2, y - 0.02, w, 10.0, s,
                       lo=-0.5, hi=0.5, grid=0.0)
            t = op * self.at(3, 0.55 + 0.1 * n)
            self.text(sc, self.STRIP[0], y + 0.022, "%d · %s" % (f["person"], f["distance_text"]),
                      t, SIZE, "lm", ink=INK)
            if not f["sure"]:
                self.text(sc, self.STRIP[0], y + 0.06, "unsure", t, SIZE, "lm", strong=True)

    # ------------------------------------------------------------------ 1-4
    def phase_1(self, sc, u, op):
        m = self.meta
        self.text(sc, self.RIGHT, 0.5, "people", on(u, 0.4, op), SIZE, "lm")
        self.text(sc, self.RIGHT, 0.55, "%d-%d" % (m["gallery_people"][0], m["gallery_people"][-1]),
                  on(u, 0.4, op), HEAD, "lm", strong=True)
        self.text(sc, self.RIGHT, 0.62, "not their", on(u, 0.6, op), SIZE, "lm", ink=QUIET)
        self.text(sc, self.RIGHT, 0.66, "enrolled photos", on(u, 0.6, op), SIZE, "lm", ink=QUIET)

    def phase_2(self, sc, u, op):
        m = self.meta
        self.text(sc, self.RIGHT, 0.5, "boxes over 0.5", on(u, 0.12, op), SIZE, "lm")
        self.running(sc, self.RIGHT, 0.55, ((0.0, "%d" % m["before"]),
                                            (0.62, "%d after nms" % len(m["faces"]))),
                     u, on(u, 0.12, op), size=HEAD, anchor="lm")

    def phase_3(self, sc, u, op):
        """The cut-outs are in ``steady``."""
        self.text(sc, self.STRIP[0], self.ROWS[0], "each box", on(u, 0.3, op), SIZE, "lm",
                  ink=QUIET)
        self.text(sc, self.STRIP[0], self.ROWS[0] + 0.04, "resized", on(u, 0.3, op), SIZE, "lm",
                  ink=QUIET)

    def phase_4(self, sc, u, op):
        """The numbers and names are in ``steady``."""
        self.text(sc, self.STRIP[0], 0.83, "threshold %s" % self.meta["threshold_text"],
                  on(u, 0.6, op), SIZE, "lm", ink=QUIET)

    # ------------------------------------------------------------------ 5: 200 scenes
    def phase_5(self, sc, u, op):
        m = self.meta
        a = on(u, 0.05, op)
        for f in m["faces"]:
            tb = f["true_box"]
            self.outline(sc, tb, AXIS, a, width=1.6)
        x = self.RIGHT
        b = on(u, 0.12, op)
        self.text(sc, x, 0.4, "%d scenes · %s faces" % (m["scenes"], m["all_faces_text"]), b,
                  SIZE, "lm")
        self.text(sc, x, 0.45, "found %d" % m["found_count"], b, HEAD, "lm", strong=True)
        c = on(u, 0.3, op)
        self.text(sc, x, 0.52, "named right:", c, SIZE, "lm")
        self.text(sc, x, 0.565, "from the detector's box", c, SIZE, "lm", ink=QUIET)
        self.text(sc, x, 0.61, "%s" % m["det_text"], c, HEAD, "lm", strong=True)
        self.text(sc, x + 0.06, 0.61, "%d of %d" % (m["right_detected"], m["found_count"]), c,
                  SIZE, "lm", ink=QUIET)
        d = on(u, 0.48, op)
        self.text(sc, x, 0.665, "from the true box", d, SIZE, "lm", ink=QUIET)
        self.text(sc, x, 0.71, "%s" % m["true_text"], d, HEAD, "lm", strong=True)
        self.text(sc, x + 0.06, 0.71, "%d of %d" % (m["right_true"], m["all_faces"]), d, SIZE,
                  "lm", ink=QUIET)
        e = on(u, 0.66, op)
        self.text(sc, x, 0.78, "boxes overlap", e, SIZE, "lm")
        self.text(sc, x, 0.82, "the truth by %s" % m["mean_iou_text"], e, SIZE, "lm")


# ====================================================================== 251
class Limits(Faces):
    """Errors person by person; the whole dataset; what it cannot measure."""

    TITLE = "What 40 people can't tell you"
    PHASES = (("rejects", 13.0), ("accepts", 12.0), ("the data", 13.0), ("can't show", 15.0))
    LAST = "Find, cut out, compare"
    NEXT = "Regression with a network"
    STATION = (None, None, None, None)
    XS = (0.1, 0.47)                 # the ten people's bars
    R_BASE, R_TOP = 0.6, 0.45        # false rejects: 0 here, 80% at the top
    A_BASE, A_TOP = 0.86, 0.72       # false accepts: 0 here, 5% at the top
    R_MAX, A_MAX = 0.8, 0.05
    GRID = (0.06, 0.47, 0.695, 0.86)   # phases 3-4: 40 x 10 tiles, one column per person
    TOP_Y = 0.42                       # phase 4: what it cannot measure

    def setup(self):
        m = self.meta = self.study.metadata
        n = len(m["people"])
        x0, x1 = self.XS
        self.xs = [x0 + (x1 - x0) * (i + 0.5) / n for i in range(n)]
        self.CODE = (
            Code("fc-verify", run=((0.0, 2), (0.25, 3), (0.45, 4))),
            Code("fc-verify", run=((0.0, 4),)),
            Code("fc-data", run=((0.0, 0), (0.3, 3), (0.45, 4))),
            Code("fc-data", run=((0.0, 0),)))
        self.EVIDENCE = (measured("450 same-person pairs, 30-39"),
                         measured("4,500 pairs of two people"),
                         about("40 people, one lab, 1992-94"),
                         about("40 people, one lab, 1992-94"))

    def heading_for(self, k, u):
        return ("one threshold, person by person",
                "accepted as someone else",
                "all the data there is",
                "what this data cannot measure")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "The same threshold rejects %s of one person's own pairs and %s of another's: "
            "%s over all ten." % (m["worst_text"], "0%", m["frr_all_text"]),
            "Pairs of two people wrongly accepted: from %s to %s per person, %s over all."
            % (m["far_lo_text"], m["far_hi_text"], m["far_all_text"]),
            "%d people, %d photos each, taken at one lab in %s for research: %d of them "
            "tested." % (m["people_all"], m["photos_each"], m["years"].replace("-", "–"),
                         m["test_people"]),
            "Real systems are tested on far larger, more varied sets, and still err unevenly. "
            "This small set can't show how.",
        ][k]

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        # the ten unseen people's rejected pairs (phases 1-3), their numbers under the bars
        keep = op * self.kept(0, 2)
        if keep > 0.01:
            for x, p in zip(self.xs, m["people"], strict=True):
                self.text(sc, x, self.R_BASE + 0.025, str(p), keep, SIZE)
            self.chart_bars(sc, m["frr"], [str(v) for v in m["rejects"]], self.R_BASE,
                            self.R_TOP, self.R_MAX, keep,
                            eased(u, 0.12, 0.35) if k == 0 else 1.0,
                            "own pairs rejected, of %d each" % m["same_each"],
                            "all: " + m["frr_all_text"],
                            keep * (self.at(0, 0.55) if k == 0 else 1.0))
        two = op * self.kept(1, 1)
        if two > 0.01:
            self.chart_bars(sc, m["far"], [str(v) for v in m["accepts"]], self.A_BASE,
                            self.A_TOP, self.A_MAX, two,
                            eased(u, 0.1, 0.35) if k == 1 else 1.0,
                            "pairs with others accepted, of %d each" % m["other_each"],
                            "all: " + m["far_all_text"], two * self.at(1, 0.5))
        # the whole dataset (phases 3-4)
        d = op * self.kept(2, 3)
        if d > 0.01:
            self.dataset(sc, k, u, d)

    def chart_bars(self, sc, values, texts, base, top, vmax, op, grow, label, total, t_op):
        """Ten bars, one per person, ``values`` (shares) scaled to ``vmax`` at ``top``, each
        labelled with its count; ``total`` (the share over all ten) at the right."""
        y = top - 0.045
        self.text(sc, self.XS[0] - 0.04, y, label, op, SIZE, "lm")
        self.text(sc, self.XS[1], y, total, t_op, SIZE, "rm", strong=True)
        w = self.span((self.XS[1] - self.XS[0]) / len(values))[0] - 12.0
        for x, v, s in zip(self.xs, values, texts, strict=True):
            h = (base - top) * v / vmax * grow
            self.column(sc, x, base, h, w, DATA, op)
            self.text(sc, x, base - h - 0.02, s, op * smoothstep(0.85, 1.0, grow), SIZE)
        self.guide(sc, (self.XS[0] - 0.01, base), (self.XS[1] + 0.005, base), AXIS, op,
                   width=1.2)

    def dataset(self, sc, k, u, op):
        m = self.meta
        x0, x1, y0, y1 = self.GRID
        a = op * (self.at(2, 0.04) if k == 2 else 1.0)
        rows = np.array([[1.0 if p < m["train_people"] else 0.45] * m["photos_each"]
                         for p in range(m["people_all"])]).T
        w, h = self.span(x1 - x0, y1 - y0)
        self.tiles(sc, rows, (x0 + x1) / 2, (y0 + y1) / 2, w, h, a, lo=0.0, hi=1.0, grid=0.35)
        self.text(sc, x0, y0 - 0.03, "%d people × %d photos" % (m["people_all"], m["photos_each"]),
                  a, SIZE, "lm")
        self.text(sc, x1, y0 - 0.03, "one lab · %s" % m["years"].replace("-", "–"),
                  op * (self.at(2, 0.3) if k == 2 else 1.0), SIZE, "rm", strong=True)
        split = x0 + (x1 - x0) * m["train_people"] / m["people_all"]
        last = m["people_all"] - 1
        self.text(sc, x0, y1 + 0.028, "trained on: 0-%d" % (m["train_people"] - 1), a, SIZE, "lm",
                  ink=DATA_INK)
        self.text(sc, x1, y1 + 0.028, "tested: %d-%d" % (m["train_people"], last), a, SIZE, "rm",
                  ink=QUIET)
        self.guide(sc, (split, y0 - 0.008), (split, y1 + 0.008), INK, a, width=1.6)

    # ------------------------------------------------------------------ 1-2
    def phase_1(self, sc, u, op):
        """The bars are in ``steady``; here, the extremes."""
        m = self.meta
        i = m["people"].index(m["worst"])
        self.text(sc, self.xs[i], self.R_BASE + 0.06, m["worst_text"], on(u, 0.6, op), SIZE,
                  ink=DATA_INK)

    def phase_2(self, sc, u, op):
        """Both sets of bars are in ``steady``."""

    # ------------------------------------------------------------------ 3-4
    def phase_3(self, sc, u, op):
        """The bars stay and the whole dataset comes in under them, in ``steady``."""

    def phase_4(self, sc, u, op):
        m = self.meta
        y = self.TOP_Y
        self.text(sc, self.GRID[0], y, "how errors differ across", on(u, 0.05, op), SIZE, "lm")
        widths = [layout.text_width(s, SIZE) + 0.022 for s in m["cannot"]]
        gap = (self.GRID[1] - self.GRID[0] - sum(widths)) / (len(widths) - 1)
        x = self.GRID[0]
        for i, (w, s) in enumerate(zip(widths, m["cannot"], strict=True)):
            a = on(u, 0.12 + 0.07 * i, op)
            self.rect(sc, self.P(x + w / 2, y + 0.055), self.span(w)[0], 26.0, PAPER, a,
                      border=AXIS, border_w=1.4, radius=12, lift=0.3)
            self.text(sc, x + w / 2, y + 0.055, s, a, SIZE, ink=INK)
            x += w + gap
        self.text(sc, self.GRID[0], y + 0.115, "that it never saw: not measured",
                  on(u, 0.5, op), SIZE, "lm", strong=True)
        self.text(sc, self.GRID[0], y + 0.17, "results here do not transfer", on(u, 0.7, op),
                  SIZE, "lm", ink=QUIET)


ANIMATIONS = {"faces-identification": Identification, "faces-the-whole-pipeline": Pipeline,
              "faces-what-40-people-cannot-show": Limits}

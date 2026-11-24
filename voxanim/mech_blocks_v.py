"""Animations 245-248, the first four of "Recognising faces" in "Blocks and code" (see
:mod:`voxanim.blocks`), drawn from :mod:`voxanim.study_blocks_v`.

* :class:`FaceNumbers` (``faces-a-face-is-numbers``): photo 300 (person 30's first) as
  64 x 64 = 4,096 numbers, nine of them written out; the 400 photos as a sheet of 40 people
  by 10 photos, people 0-29 marked for training and 30-39 as never trained on; two pairs
  compared pixel by pixel (person 30's first two photos 0.340 apart, persons 30 and 31 only
  0.336); then all 4,950 pairs of people 30-39 as two piles of distance, which overlap.
* :class:`Embedding` (``faces-an-embedding``): the encoder of region "fc-encoder" on photo
  300 at 32 x 32: three blocks (32 -> 16 -> 8 -> 4), flattened to 512, a linear layer to
  32 numbers of length 0.09, scaled to length 1; 22,304 numbers learned.
* :class:`Triplets` (``faces-triplets``): the first triplet of the first step (anchor and
  positive of person 14, negative of person 3) on a line of distance from the anchor:
  before training 0.04 and 0.09, short of the margin 0.5, loss 0.45; the loss and the share
  of triplets not yet 0.5 apart over 1,500 steps; after training 0.24 and 1.27, loss 0.
* :class:`Verification` (``faces-verification``): the 100 photos of people 30-39 make
  4,950 pairs; their distances through the encoder as two piles (one person above the
  line, two people below); the threshold 0.72 chosen on people 0-29: 86 false accepts and
  111 false rejects; the equal error rate 12.4% against 22.0% for raw pixels.

The people are named only by their number. Telling faces apart is not training except where
a photo goes through the encoder ("forward") and the triplet loss is computed ("loss").
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
    exact,
)
from .diagram import draw_image
from .easing import ease_in_out, smoothstep
from .mech import AXIS, PAPER, RED, TRACK, dark
from .mech_blocks_w import gray

DATA_INK = dark(DATA, 0.3)
OTHER = dark(AXIS, 0.05)        # two people: grey
OTHER_INK = dark(AXIS, 0.35)
LOSS_INK = dark(LOSS, 0.2)
WRONG = RED                     # a mistake (a false accept or reject)
WRONG_INK = dark(RED, 0.25)
TAG = "measured on Olivetti faces: "     # as group u (241-244) writes it


def faces(what):
    """An evidence tag: measured on the Olivetti faces."""
    return TAG + what


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on(u, a, op, width=0.05):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


def lerp(a, b, s):
    return a + (b - a) * s


def face_rgb(img, soften=0.0):
    """A photo's colours: its grey levels, as groups t and w draw photos (dark pixels dark;
    :func:`voxanim.mech_blocks_w.gray`); ``soften`` turns it that far towards paper (for
    numbers written over it)."""
    rgb = gray(img)
    return (rgb + (np.asarray(PAPER, np.float32) - rgb) * soften).astype(np.float32)


def mosaic(images, gap=1):
    """A ``rows x cols`` grid of square photos (``[rows, cols, s, s]``) as one texture, with
    ``gap`` texels of paper between them."""
    rows, cols, s = images.shape[0], images.shape[1], images.shape[2]
    out = np.empty((rows * (s + gap) - gap, cols * (s + gap) - gap, 3), np.float32)
    out[:] = np.asarray(PAPER, np.float32)
    for r in range(rows):
        for c in range(cols):
            out[r * (s + gap):r * (s + gap) + s, c * (s + gap):c * (s + gap) + s] = \
                face_rgb(images[r, c])
    return out


class Faces(Blocks):
    """What the four share: photos, sheets of photos, bars and two piles of distance."""

    def at(self, k, u):
        """1 once phase ``k`` has reached ``u`` (fading in over a short while); 0 before."""
        a, b = self.spans[k]
        t0 = a + (b - a) * u
        return smoothstep(t0, t0 + 0.5, self._t)

    def photo(self, sc, rgb, fx, fy, side, op, lift=0.2):
        """A photo (its colours, :func:`face_rgb`) ``side`` px square on a data card."""
        if op <= 0.01:
            return
        self.card(sc, fx, fy, side + 14, side + 20, op, lift=lift)
        draw_image(self, sc, self.P(fx, fy + self.frac(0, 3.0)[1]), rgb, side, side, op,
                   grid=0.0, lift=lift + 0.004)

    def photo_top(self, fy, side):
        """The layout y of the image's top edge in :meth:`photo` (its card is 3 px lower)."""
        return fy + self.frac(0, 3.0 - side / 2)[1]

    def sheet(self, sc, rgb, fx, fy, w, h, op, lift=0.2):
        """A sheet of photos (a :func:`mosaic`) ``w x h`` px on a data card."""
        if op <= 0.01:
            return
        self.card(sc, fx, fy, w + 12, h + 18, op, lift=lift)
        draw_image(self, sc, self.P(fx, fy + self.frac(0, 3.0)[1]), rgb, w, h, op, grid=0.0,
                   lift=lift + 0.004)

    def vbar(self, sc, fx, y0, y1, w, ink, op, lift=0.3):
        """A bar ``w`` px wide from layout ``y0`` to ``y1`` (either way round)."""
        if op <= 0.01 or abs(y1 - y0) * self.cam.height < 0.6:
            return
        h = max(abs(self.span(0, y1 - y0)[1]), 1.0)
        self.rect(sc, self.P(fx, (y0 + y1) / 2), w, h, ink, op, radius=0, lift=lift)

    def hbar_at(self, sc, x0, y, w, h, ink, op, lift=0.3):
        """A flat bar from layout ``x0``, ``w`` layout wide, ``h`` px tall, centred on ``y``."""
        if op <= 0.01 or w <= 0:
            return
        self.rect(sc, self.P(x0 + w / 2, y), max(self.span(w)[0], 1.2), h, ink, op, radius=0,
                  lift=lift)

    def vline(self, sc, fx, y0, y1, ink, op, width=2.0):
        """A guide line in front of the shapes."""
        if op > 0.01:
            self.polyline(sc, [self.P(fx, y0, self.ARROW_LIFT), self.P(fx, y1, self.ARROW_LIFT)],
                          width, ink, op, bias=0.5)

    def piles(self, sc, bins, up, down, x0, x1, base, tall, op, grow=1.0, ink_up=DATA,
              ink_down=OTHER, wrong_up=None, wrong_down=None, wrong=0.0):
        """Two histograms on one line of distance: ``up`` (shares) above ``base``, ``down``
        below it, both on one scale (the larger peak ``tall`` layout high); bars with
        ``wrong_up(centre)`` / ``wrong_down(centre)`` true turn red as ``wrong`` goes 0..1."""
        if op <= 0.01:
            return
        top = max(np.max(up), np.max(down))
        lo, hi = bins[0], bins[-1]
        n = len(bins) - 1
        w = self.span((x1 - x0) / n)[0] - 1.2
        for i in range(n):
            c = (bins[i] + bins[i + 1]) / 2
            fx = x0 + (x1 - x0) * (c - lo) / (hi - lo)
            for share, sign, ink, test in ((up[i], -1, ink_up, wrong_up),
                                           (down[i], 1, ink_down, wrong_down)):
                if share <= 0:
                    continue
                h = tall * share / top * grow
                g = wrong if test is not None and test(c) else 0.0
                col = tuple(float(a + (b - a) * g) for a, b in zip(ink, WRONG, strict=True))
                self.vbar(sc, fx, base, base + sign * max(h, self.frac(0, 1.0)[1]), w, col, op)
        self.polyline(sc, [self.P(x0, base, 0.35), self.P(x1, base, 0.35)], 1.4, AXIS, op,
                      bias=0.0)


# ====================================================================== 245
class FaceNumbers(Faces):
    """A photo as numbers; the 400 photos; two pairs, pixel by pixel; all pairs of the
    unseen people: the two kinds overlap."""

    TITLE = "A face is 4,096 numbers"
    PHASES = (("a photo", 13.0), ("400 photos", 13.0), ("two pairs", 14.0), ("all pairs", 14.0))
    LAST = "Is the detector any good?"
    NEXT = "A face as 32 numbers"
    STATION = ("data", "data", None, None)
    BIG = ((0.15, 0.555), 128.0)          # phase 1: the photo, centre and side (px)
    SMALL = ((0.10, 0.765), 48.0)         # from phase 2: the photo, small
    PATCH_AT = (0.375, 0.535)             # the 3 x 3 numbers, enlarged
    PATCH_TILE = 42.0
    SHEET_X0, SHEET_Y = 0.062, 0.49       # the 400 photos: left edge, centre
    THUMB, SPLIT = 9.5, 12.0              # px per photo; the gap after person 29
    PARTNERS = ((0.255, 0.715), (0.255, 0.845))
    PARTNER = 44.0
    HIST = (0.13, 0.46, 0.79, 0.09)       # x0, x1, base, tallest bar (layout)
    HIST_MAX = 0.6

    def setup(self):
        m = self.meta = self.study.metadata
        self.rgb = face_rgb(m["image"])
        self.patch_rgb = face_rgb(m["patch"], soften=0.55)
        th = np.asarray(m["thumbs"])
        n_train = m["train_people"]
        by = th.reshape(m["people"], m["per_person"], *th.shape[1:]).transpose(1, 0, 2, 3)
        self.train_rgb = mosaic(by[:, :n_train])
        self.test_rgb = mosaic(by[:, n_train:])
        self.n_train, self.n_test = n_train, m["people"] - n_train
        self.pair_rgb = [face_rgb(m["same_images"][1]), face_rgb(m["diff_images"][1])]
        self.CODE = (
            Code("fc-data", run=((0.0, 0), (0.3, 1))),
            Code("fc-data", run=((0.0, 2), (0.4, 3), (0.65, 4))),
            Code("fc-data", run=((0.0, 4),)),
            Code("fc-data", run=((0.0, 4),)))
        self.EVIDENCE = (faces("photo %d, %s numbers" % (m["photo"], m["numbers_text"])),
                         faces("400 photos, 40 people"),
                         faces("two pairs, pixel by pixel"),
                         faces("%s pairs of people 30-39" % m["pairs_text"]))
        self.HEADINGS = ("a photo is 64 × 64 numbers", "40 people, 10 photos each",
                         "how far apart, pixel by pixel?", "one person and two people overlap")

    def subtitle_for(self, k):
        m = self.meta
        return [
            "Photo %d, person %d's first, is 64 × 64 = %s numbers, each from 0 (dark) to 1 "
            "(light)." % (m["photo"], m["photo_person"], m["numbers_text"]),
            "400 photos: 40 people, 10 each. People 0–29 train the models; people 30–39 are "
            "only tested on.",
            "Pixel by pixel, person 30's first two photos are %s apart; persons 30 and 31 only "
            "%s." % (m["same_text"], m["diff_text"]),
            "Over all %s pairs, one person spans %s to %s and two people %s to %s: they "
            "overlap." % (m["pairs_text"], *m["same_range_text"], *m["diff_range_text"]),
        ][k]

    # ------------------------------------------------------------------ geometry
    def sheet_geometry(self):
        """``(train centre x, test centre x, width train, width test, height)`` in layout
        units and px."""
        wt, ws = self.n_train * self.THUMB, self.n_test * self.THUMB
        x0 = self.SHEET_X0
        cx_train = x0 + self.frac(wt / 2)[0]
        cx_test = x0 + self.frac(wt + self.SPLIT + ws / 2)[0]
        return cx_train, cx_test, wt, ws, self.THUMB * 10

    def where_photo(self, k, u):
        """The photo's centre and side: big in phase 1, moving to small early in phase 2."""
        (bx, by), bs = self.BIG
        (sx, sy), ss = self.SMALL
        s = 0.0 if k == 0 else (eased(u, 0.02, 0.16) if k == 1 else 1.0)
        return lerp(bx, sx, s), lerp(by, sy, s), lerp(bs, ss, s)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        keep = op * self.kept(0, 2)
        if keep > 0.01:
            a = keep * (self.at(0, 0.03) if k == 0 else 1.0)
            fx, fy, side = self.where_photo(k, u)
            self.photo(sc, self.rgb, fx, fy, side, a)
            if k == 0:
                self.text(sc, fx, fy + self.frac(0, side / 2 + 26)[1],
                          "photo %d · person %d" % (m["photo"], m["photo_person"]), a, SIZE)
            else:
                b = a * (smoothstep(0.14, 0.2, u) if k == 1 else 1.0)
                self.text(sc, fx, fy + self.frac(0, side / 2 + 24)[1], "person %d"
                          % m["photo_person"], b, SIZE)
        s = op * self.kept(1)
        if s > 0.01:
            self.photos(sc, k, u, s)

    def photos(self, sc, k, u, op):
        m = self.meta
        a = op * (self.at(1, 0.2) if k == 1 else 1.0)
        cx_train, cx_test, wt, ws, h = self.sheet_geometry()
        self.sheet(sc, self.train_rgb, cx_train, self.SHEET_Y, wt, h, a)
        self.sheet(sc, self.test_rgb, cx_test, self.SHEET_Y, ws, h, a)
        y = self.SHEET_Y + self.frac(0, h / 2 + 30)[1]
        b = op * (self.at(1, 0.45) if k == 1 else 1.0)
        self.text(sc, cx_train, y, "people 0–%d: training" % (m["train_people"] - 1), b, SIZE)
        c = op * (self.at(1, 0.62) if k == 1 else 1.0)
        self.text(sc, cx_test, y, "%d–%d: unseen" % (m["train_people"], m["people"] - 1), c,
                  SIZE, ink=DATA_INK)
        # photo 300's place: person 30, photo 0
        d = op * (self.at(1, 0.78) if k == 1 else 1.0) * self.kept(1, 2)
        if d > 0.01:
            x = cx_test - self.frac(ws / 2 - self.THUMB / 2)[0]
            yy = self.SHEET_Y + self.frac(0, 3.0 - h / 2 + self.THUMB / 2)[1]
            self.rect(sc, self.P(x, yy), self.THUMB + 5, self.THUMB + 5, INK, d, border=INK,
                      border_w=2.2, radius=2, lift=0.6, fill_op=0.0)

    # ------------------------------------------------------------------ 1: a photo
    def phase_1(self, sc, u, op):
        m = self.meta
        (fx, fy), side = self.BIG
        px = side / m["side"]
        r0, c0 = m["patch_at"]
        n = m["patch_side"]
        a = on(u, 0.22, op)
        # the nine pixels ringed on the photo, then enlarged with their numbers
        top = self.photo_top(fy, side)
        left = fx - self.frac(side / 2)[0]
        cx = left + self.frac((c0 + n / 2) * px)[0]
        cy = top + self.frac(0, (r0 + n / 2) * px)[1]
        self.rect(sc, self.P(cx, cy), n * px + 5, n * px + 5, INK, a, border=INK, border_w=2.0,
                  radius=2, lift=0.6, fill_op=0.0)
        tx, ty = self.PATCH_AT
        T = self.PATCH_TILE
        b = on(u, 0.3, op)
        self.arrow(sc, (cx + self.frac(n * px / 2 + 4)[0], cy),
                   (tx - self.frac(n * T / 2 + 8)[0], ty), AXIS, b, u=eased(u, 0.3, 0.36),
                   width=1.6)
        c = on(u, 0.36, op)
        draw_image(self, sc, self.P(tx, ty), self.patch_rgb, n * T, n * T, c, grid=0.35,
                   lift=0.3)
        for r in range(n):
            for k in range(n):
                x = tx + self.frac((k - (n - 1) / 2) * T)[0]
                y = ty + self.frac(0, (r - (n - 1) / 2) * T)[1]
                self.text(sc, x, y, m["patch_text"][r][k], on(u, 0.42 + 0.012 * (r * n + k), op),
                          SIZE, ink=INK)
        self.text(sc, tx, ty + self.frac(0, n * T / 2 + 20)[1], "9 of the %s"
                  % m["numbers_text"], on(u, 0.5, op), SIZE)
        self.text(sc, 0.06, 0.805, "64 × 64 = %s numbers" % m["numbers_text"], on(u, 0.62, op),
                  HEAD, "lm", strong=True)
        self.text(sc, 0.06, 0.855, "0 is dark, 1 is light", on(u, 0.7, op), SIZE, "lm",
                  ink=QUIET)

    # ------------------------------------------------------------------ 2: 400 photos
    def phase_2(self, sc, u, op):
        """The sheet of 400 photos is in ``steady``; here, what it is."""
        self.text(sc, 0.16, 0.765, "%s photos" % self.meta["photos"], on(u, 0.3, op), HEAD, "lm",
                  strong=True)
        self.text(sc, 0.16, 0.81, "a column: one person", on(u, 0.36, op), SIZE, "lm")

    # ------------------------------------------------------------------ 3: two pairs
    def phase_3(self, sc, u, op):
        m = self.meta
        (fx, fy), side = self.SMALL
        right = fx + self.frac(side / 2 + 9)[0]
        self.text(sc, 0.06, 0.925, "each photo scaled to length 1", on(u, 0.05, op), SIZE,
                  "lm", ink=QUIET)
        rows = ((m["same_people"][1], m["same_text"], "one person", 0.12),
                (m["diff_people"][1], m["diff_text"], "two people", 0.38))
        for (px, py), rgb, (who, d, kind, at) in zip(self.PARTNERS, self.pair_rgb, rows,
                                                     strict=True):
            a = on(u, at, op)
            self.photo(sc, rgb, px, py, self.PARTNER, a)
            self.arrow(sc, (right, fy + (py - fy) * 0.25),
                       (px - self.frac(self.PARTNER / 2 + 12)[0], py), AXIS, a,
                       u=eased(u, at, at + 0.06), width=1.6, head=False)
            b = on(u, at + 0.08, op)
            tx = px + self.frac(self.PARTNER / 2 + 16)[0]
            self.text(sc, tx, py - 0.018, "person %d: %s" % (who, d), b, HEAD, "lm",
                      strong=True)
            if kind == "one person":
                self.text(sc, tx, py + 0.024, kind, b, SIZE, "lm", ink=DATA_INK)
            else:
                self.running(sc, tx, py + 0.024, [(0.0, kind), (0.62, "two people, yet closer")],
                             u, b, size=SIZE, ink=OTHER_INK)

    # ------------------------------------------------------------------ 4: all pairs
    def phase_4(self, sc, u, op):
        m = self.meta
        x0, x1, base, tall = self.HIST
        bins = np.asarray(m["bins"])
        frame = on(u, 0.03, op)
        self.piles(sc, bins, np.asarray(m["share_same"]), np.asarray(m["share_diff"]), x0, x1,
                   base, tall, frame, grow=eased(u, 0.05, 0.3))
        a = on(u, 0.08, op)
        self.text(sc, 0.06, 0.675, "one person: %s pairs" % m["same_pairs_text"], a, SIZE, "lm",
                  ink=DATA_INK)
        self.text(sc, x1, 0.675, "two people: %s" % m["diff_pairs_text"], on(u, 0.14, op), SIZE,
                  "rm", ink=OTHER_INK)
        fx = lambda d: x0 + (x1 - x0) * d / self.HIST_MAX              # noqa: E731
        for v in (0.0, 0.2, 0.4, 0.6):
            self.text(sc, fx(v), base + tall + 0.03, "%g" % v, frame, SIZE, ink=QUIET)
        self.text(sc, 0.06, base, "distance", frame, SIZE, "lm", ink=QUIET)
        # where both kinds are: from the nearest two-people pair to the farthest one-person
        lo, hi = m["overlap"]
        b = on(u, 0.45, op)
        if b > 0.01:
            w = self.span(fx(hi) - fx(lo))[0]
            self.rect(sc, self.P((fx(lo) + fx(hi)) / 2, base), w, self.span(0, 2 * tall)[1] + 8,
                      TRACK, 0.35 * b, radius=3, lift=0.25)
            self.text(sc, fx(hi) + 0.01, base - tall * 0.78, "both kinds", b, SIZE, "lm",
                      strong=True)
            self.text(sc, fx(hi) + 0.01, base - tall * 0.36, "%s–%s" % m["overlap_text"], b,
                      SIZE, "lm")


# ====================================================================== 246
class Embedding(Faces):
    """Three blocks; flatten and a linear layer; length 1; the numbers learned."""

    TITLE = "A face as 32 numbers"
    PHASES = (("three blocks", 15.0), ("512, then 32", 13.0), ("length 1", 12.0),
              ("numbers learned", 13.0))
    LAST = "A face is 4,096 numbers"
    NEXT = "Triplets: same closer than different"
    STATION = ("forward", "forward", "forward", None)
    ROW_Y = 0.495
    XS = (0.105, 0.215, 0.315, 0.405)        # the photo and the three blocks' outputs
    TILES = (2.2, 3.0, 4.5, 7.0)             # px per tile
    SHAPE_Y, COUNT_Y = 0.61, 0.648
    FLAT_Y, OUT_Y = 0.71, 0.80
    STRIP_X = (0.16, 0.46)
    LIN_X = 0.31
    APPEAR = (0.03, 0.22, 0.42, 0.62)        # phase 1: each stage appears

    def setup(self):
        m = self.meta = self.study.metadata
        self.rgb = face_rgb(m["image"])
        self.lo, self.hi = -0.4, 0.4
        self.CODE = (
            Code("fc-encoder", run=((0.0, 0), (0.18, 1), (0.58, 2))),
            Code("fc-encoder", run=((0.0, 3), (0.4, 4))),
            Code(("fc-encoder", "fc-triplet"), pick=(0, 1, 2, 3, 4, 5, 6), run=((0.0, 6),)),
            Code("fc-encoder", run=((0.0, 1), (0.18, 2), (0.36, 4))))
        c = [x["text"] for x in m["counts"]]
        self.EVIDENCE = (faces("photo %d, trained encoder" % m["photo"]),
                         faces("photo %d: 512, then 32" % m["photo"]),
                         faces("photo %d: length %s, then 1" % (m["photo"], m["length_text"])),
                         exact("%s + %s + %s + %s" % tuple(c)))
        self.HEADINGS = ("three blocks: 32 → 16 → 8 → 4", "flatten to 512, then 32 numbers",
                         "scaled to length 1", "%s numbers learned" % m["total_text"])

    def subtitle_for(self, k):
        m = self.meta
        c = [x["text"] for x in m["counts"]]
        return [
            "Each block looks closer, then halves the picture: photo %d goes from 32 × 32 to "
            "32 maps of 4 × 4." % m["photo"],
            "Flatten lays the 32 × 4 × 4 out as 512 numbers; one linear layer turns them into "
            "32.",
            "F.normalize divides by the length, %s here: every face becomes 32 numbers of "
            "length 1." % m["length_text"],
            "The blocks learn %s, %s and %s numbers, the linear layer %s: %s in all."
            % (*c, m["total_text"]),
        ][k]

    # ------------------------------------------------------------------ what stays
    def appear(self, k, u, i):
        """How far stage ``i`` of the row has appeared (it stays once it has)."""
        return self.at(0, self.APPEAR[i]) if k == 0 else 1.0

    def steady(self, sc, k, u, op):
        m = self.meta
        a0 = op * self.appear(k, u, 0)
        self.text(sc, 0.06, 0.39, "photo %d · person %d, 32 × 32" % (m["photo"], m["person"]),
                  a0, SIZE, "lm")
        prev = None
        for i, (x, tile) in enumerate(zip(self.XS, self.TILES, strict=True)):
            a = op * self.appear(k, u, i)
            if a <= 0.01:
                break
            if i == 0:
                side = 32 * tile
                self.photo(sc, self.rgb, x, self.ROW_Y, side, a)
                half = side / 2 + 7
            else:
                img = np.asarray(m["maps"][i - 1], float)
                side = img.shape[0] * tile
                self.stack(sc, x, self.ROW_Y, side + 14, side + 20, 4, a)
                self.tiles(sc, img, x, self.ROW_Y + self.frac(0, 3.0)[1], side, side, a,
                           lo=0.0, hi=float(img.max()) or 1.0, lift=0.204, grid=0.18)
                half = side / 2 + 7
                self.arrow(sc, (prev, self.ROW_Y), (x - self.frac(half + 4)[0], self.ROW_Y),
                           AXIS, a, u=eased(u, self.APPEAR[i] - 0.06, self.APPEAR[i])
                           if k == 0 else 1.0, width=1.6)
            prev = x + self.frac(half + (16 if i else 0) + 4)[0]
            self.text(sc, x, self.SHAPE_Y, m["shapes"][i], a, SIZE, ink=DATA_INK)
        f = op * self.kept(1)
        if f > 0.01:
            self.flat(sc, k, u, f)

    def flat(self, sc, k, u, op):
        """The 512 and the 32 (phase 2 on)."""
        m = self.meta
        x0, x1 = self.STRIP_X
        w = self.span(x1 - x0)[0]
        cx = (x0 + x1) / 2
        a = op * (self.at(1, 0.06) if k == 1 else 1.0)
        top = self.ROW_Y + self.frac(0, 50)[1]
        self.arrow(sc, (0.452, top), (0.452, self.FLAT_Y - 0.03), AXIS, a,
                   u=eased(u, 0.06, 0.14) if k == 1 else 1.0, width=1.6)
        flat = np.asarray(m["flat"], float)
        self.strip(sc, flat, cx, self.FLAT_Y, w, 16.0, a, lo=0.0, hi=float(flat.max()))
        self.text(sc, 0.06, self.FLAT_Y, m["flat_text"], a, HEAD, "lm", strong=True)
        b = op * (self.at(1, 0.4) if k == 1 else 1.0)
        self.arrow(sc, (self.LIN_X, self.FLAT_Y + 0.022), (self.LIN_X, self.OUT_Y - 0.028), AXIS,
                   b, u=eased(u, 0.4, 0.48) if k == 1 else 1.0, width=1.6)
        self.text(sc, self.LIN_X + 0.012, (self.FLAT_Y + self.OUT_Y) / 2, "Linear(512, 32)", b,
                  SIZE, "lm")
        # the 32 numbers: as they come out, then scaled to length 1 (phase 3)
        s = 0.0 if k < 2 else (eased(u, 0.2, 0.5) if k == 2 else 1.0)
        vals = lerp(np.asarray(m["e"], float), np.asarray(m["v"], float), s)
        self.tiles(sc, vals[None], cx, self.OUT_Y, w, 18.0, b, lo=self.lo, hi=self.hi,
                   grid=0.25)
        self.text(sc, 0.06, self.OUT_Y, str(m["dims"]), b, HEAD, "lm", strong=True)
        c = op * (self.at(1, 0.6) if k == 1 else 1.0)
        unit = k > 2 or (k == 2 and u >= 0.42)
        self.text(sc, x0, 0.86, "length %s" % (m["unit_length_text"] if unit else m["length_text"]),
                  c, HEAD, "lm", strong=True)

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        """The row of blocks is in ``steady``."""

    def phase_2(self, sc, u, op):
        """The 512 and the 32 are in ``steady``."""

    def phase_3(self, sc, u, op):
        self.text(sc, 0.31, 0.86, "÷ %s" % self.meta["length_text"], on(u, 0.12, op), HEAD,
                  "lm", ink=DATA_INK)

    def phase_4(self, sc, u, op):
        m = self.meta
        c = m["counts"]
        for i, at in zip(range(3), (0.08, 0.18, 0.28), strict=True):
            self.text(sc, self.XS[i + 1], self.COUNT_Y, c[i]["text"], on(u, at, op), SIZE,
                      ink=dark(WEIGHT, 0.3), strong=False)
        self.text(sc, self.LIN_X - 0.012, (self.FLAT_Y + self.OUT_Y) / 2, c[3]["text"],
                  on(u, 0.4, op), SIZE, "rm", ink=dark(WEIGHT, 0.3))
        self.text(sc, 0.46, 0.86, "= %s" % m["total_text"], on(u, 0.52, op), HEAD, "rm",
                  strong=True)


# ====================================================================== 247
class Triplets(Faces):
    """Anchor, positive, negative; the margin before training; the loss over training; the
    same triplet after."""

    TITLE = "Triplets: same closer than different"
    PHASES = (("anchor, positive, negative", 13.0), ("the margin", 15.0), ("training", 14.0),
              ("after", 13.0))
    LAST = "A face as 32 numbers"
    NEXT = "Same person or not?"
    STATION = ("forward", "loss", "loss", "forward")
    ROW_Y = 0.475
    XS = (0.11, 0.225, 0.34)
    TILE = 1.8
    LINE = (0.08, 0.45, 0.735)              # the line of distance: x at 0, x at MAX, y
    MAX = 1.4
    CHARTS = (((0.12, 0.26), (0.665, 0.835)), ((0.33, 0.46), (0.665, 0.835)))

    def setup(self):
        m = self.meta = self.study.metadata
        self.rgb = [face_rgb(img) for img in m["images"]]
        self.CODE = (
            Code("fc-triplet", run=((0.0, 0), (0.33, 1), (0.6, 2))),
            Code("fc-triplet", run=((0.0, 3), (0.45, 4))),
            Code("fc-triplet", run=((0.0, 3),)),
            Code("fc-triplet", run=((0.0, 0), (0.12, 1), (0.24, 2), (0.5, 3))))
        self.EVIDENCE = (faces("triplets(0), the first of %d" % m["batch"]),
                         faces("the triplet, before training"),
                         faces("%s steps, means of %d" % (m["steps_text"], m["mean_of"])),
                         faces("the triplet, after training"))
        self.HEADINGS = ("anchor, positive, negative", "the negative must be 0.5 further",
                         "%s steps of %d triplets" % (m["steps_text"], m["batch"]),
                         "after training: the same triplet")
        x = np.concatenate([[0.0], m["curve_steps"]])
        self.curves = (np.concatenate([[m["first_loss"]], m["loss_means"]]),
                       np.concatenate([[m["first_active"]], m["active_means"]]))
        self.cx = x
        self.charts = (self.chart(self.CHARTS[0], [0.0, m["steps"]], [0.0, 0.5], pad=0.0),
                       self.chart(self.CHARTS[1], [0.0, m["steps"]], [0.0, 1.0], pad=0.0))

    def subtitle_for(self, k):
        m = self.meta
        b, a = m["before"], m["after"]
        p = m["people"]
        return [
            "A triplet from the training people: two photos of person %d and one of person %d."
            % (p[0], p[2]),
            "Before training: positive %s away, negative %s. Not 0.5 further, so the loss is "
            "%s." % (b["ap_text"], b["an_text"], b["loss_text"]),
            "Over %s steps the loss falls, and the share of triplets not yet 0.5 apart drops "
            "from %s to %s." % (m["steps_text"], m["first_active_text"],
                                "%.0f%%" % (100 * m["active_means"][-1])),
            "After training: positive %s away, negative %s, more than 0.5 further. The loss is "
            "%s." % (a["ap_text"], a["an_text"], "0" if a["loss"] == 0 else a["loss_text"]),
        ][k]

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        side = 32 * self.TILE
        names = ("anchor", "positive", "negative")
        inks = (INK, DATA_INK, OTHER_INK)
        for i, (x, rgb) in enumerate(zip(self.XS, self.rgb, strict=True)):
            a = op * (self.at(0, 0.04 + 0.2 * i) if k == 0 else 1.0)
            self.photo(sc, rgb, x, self.ROW_Y, side, a)
            self.text(sc, x, self.ROW_Y - self.frac(0, side / 2 + 20)[1], names[i], a, SIZE,
                      ink=inks[i], strong=i == 0)
            self.text(sc, x, self.ROW_Y + self.frac(0, side / 2 + 26)[1], "person %d"
                      % m["people"][i], a, SIZE)

    def fx(self, d):
        x0, x1, _y = self.LINE
        return x0 + (x1 - x0) * min(d, self.MAX) / self.MAX

    def distances(self, sc, ap, an, op, labels=1.0):
        """The line of distance from the anchor: the positive and the negative on it, and
        the margin: 0.5 beyond the positive."""
        x0, x1, y = self.LINE
        m = self.meta
        self.polyline(sc, [self.P(x0, y, 0.35), self.P(x1, y, 0.35)], 1.6, AXIS, op, bias=0.0)
        for v in (0.0, 0.5, 1.0):
            self.text(sc, self.fx(v), y + 0.05, "%g" % v, op, SIZE, ink=QUIET)
        self.text(sc, x0 - 0.02, y - 0.075, "distance from the anchor", op, SIZE, "lm", ink=QUIET)
        # the margin: from the positive to 0.5 beyond it
        w = self.span(self.fx(ap + m["margin"]) - self.fx(ap))[0]
        self.rect(sc, self.P((self.fx(ap) + self.fx(ap + m["margin"])) / 2, y), w, 14.0,
                  TRACK, 0.6 * op, radius=3, lift=0.25)
        self.text(sc, self.fx(ap + m["margin"]), y - 0.035, "+ %s" % m["margin_text"],
                  op * labels, SIZE, ink=QUIET)
        self.disk(sc, self.P(x0, y), 5.5, INK, op, lift=1.4, outline=0.0)
        self.disk(sc, self.P(self.fx(ap), y), 6.5, DATA, op, lift=1.45, outline=0.0)
        self.disk(sc, self.P(self.fx(an), y), 6.5, OTHER, op, lift=1.42, outline=0.0)
        self.text(sc, self.fx(ap), y - 0.035, "p %s" % ("%.2f" % ap), op * labels, SIZE,
                  ink=DATA_INK)
        self.text(sc, self.fx(an) + 0.004, y + 0.03 if an - ap < 0.4 else y - 0.035,
                  "n %s" % ("%.2f" % an), op * labels, SIZE, "lm" if an - ap < 0.4 else "mm",
                  ink=OTHER_INK)

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        """The three photos are in ``steady``; here, the rule they are picked by."""
        m = self.meta
        self.text(sc, 0.06, 0.66, "same person: anchor, positive", on(u, 0.65, op), SIZE, "lm",
                  ink=DATA_INK)
        self.text(sc, 0.06, 0.705, "somebody else: negative", on(u, 0.72, op), SIZE, "lm",
                  ink=OTHER_INK)
        self.text(sc, 0.06, 0.78, "photos %d, %d and %d" % tuple(m["ids"]), on(u, 0.8, op),
                  SIZE, "lm", ink=QUIET)

    def phase_2(self, sc, u, op):
        b = self.meta["before"]
        self.distances(sc, b["ap"], b["an"], on(u, 0.05, op))
        self.text(sc, 0.06, 0.835, "loss = max(%s − %s + 0.5, 0)" % (b["ap_text"], b["an_text"]),
                  on(u, 0.4, op), HEAD, "lm")
        self.text(sc, 0.06, 0.88, "= %s" % b["loss_text"], on(u, 0.5, op), HEAD, "lm",
                  strong=True)

    def phase_3(self, sc, u, op):
        m = self.meta
        draw = eased(u, 0.12, 0.5)
        titles = ("loss", "not yet 0.5 apart")
        ends = ("%.4f" % self.curves[0][-1], "%.0f%%" % (100 * self.curves[1][-1]))
        starts = (m["first_loss_text"], m["first_active_text"])
        yt = ((0.0, "0"), (0.5, "0.5")), ((0.0, "0%"), (1.0, "100%"))
        for i, ch in enumerate(self.charts):
            a = on(u, 0.04 + 0.04 * i, op)
            self.text(sc, ch.x0, ch.y0 - 0.03, titles[i], a, SIZE, "lm",
                      ink=LOSS_INK if i == 0 else INK)
            self.axes(sc, ch, a, xticks=[(0.0, "0"), (m["steps"], m["steps_text"])],
                      yticks=list(yt[i]))
            ink = LOSS if i == 0 else DATA
            self.plot(sc, ch, self.cx, self.curves[i], ink, a, draw, dot=False)
            b = on(u, 0.12, op)
            self.text(sc, ch.fx(0) + 0.018, ch.fy(self.curves[i][0]) + 0.012, starts[i], b,
                      SIZE, "lm", strong=True)
            c = on(u, 0.52, op)
            self.text(sc, ch.x1, ch.fy(self.curves[i][-1]) - 0.03, ends[i], c, SIZE, "rm",
                      strong=True)
        self.text(sc, 0.06, 0.9, "step 0, then means of %d steps" % m["mean_of"],
                  on(u, 0.6, op), SIZE, "lm", ink=QUIET)

    def phase_4(self, sc, u, op):
        b, a = self.meta["before"], self.meta["after"]
        s = eased(u, 0.15, 0.45)
        ap, an = lerp(b["ap"], a["ap"], s), lerp(b["an"], a["an"], s)
        self.distances(sc, ap, an, on(u, 0.03, op), labels=1.0 if s in (0.0, 1.0) else 0.0)
        self.text(sc, 0.06, 0.835, "loss = max(%s − %s + 0.5, 0)" % (a["ap_text"], a["an_text"]),
                  on(u, 0.5, op), HEAD, "lm")
        self.text(sc, 0.06, 0.88, "= %s" % ("0" if a["loss"] == 0 else a["loss_text"]),
                  on(u, 0.58, op), HEAD, "lm", strong=True)


# ====================================================================== 248
EER_ROWS = (("pixels", "raw pixels"), ("untrained", "untrained"), ("trained", "trained"))


class Verification(Faces):
    """The pairs of the unseen photos; two piles of distance; the threshold from people
    0-29 and its mistakes; against raw pixels."""

    TITLE = "Same person or not?"
    PHASES = (("4,950 pairs", 12.0), ("two piles", 14.0), ("a threshold", 15.0),
              ("against pixels", 14.0))
    LAST = "Triplets: same closer than different"
    NEXT = "Who is it?"
    STATION = ("forward", None, None, None)
    SHEET = (0.14, 0.6, 13.0)               # the 100 photos: centre, px per photo
    HIST = (0.15, 0.46, 0.6, 0.11)          # x0, x1, base, tallest bar (layout)
    HIST_MAX = 2.0
    TALLY_Y, TALLY_STEP = 0.785, 0.036
    BAR_X, BAR_W = 0.18, 0.75               # the equal error rates: bars start, 100% length

    def setup(self):
        m = self.meta = self.study.metadata
        th = np.asarray(m["thumbs"])
        self.sheet_rgb = mosaic(th.reshape(10, 10, *th.shape[1:]))
        self.CODE = (
            Code("fc-verify", run=((0.0, 0), (0.35, 1), (0.65, 2))),
            Code("fc-verify", run=((0.0, 2), (0.3, 3))),
            Code("fc-verify", run=((0.0, 4), (0.55, 5)), out=((0.6, m["printed"]),)),
            Code("fc-verify", run=((0.0, 5),), out=((0.0, m["printed"]),)))
        self.EVIDENCE = (faces("100 photos of people 30-39"),
                         faces("%s pairs, people 30-39" % m["pairs_text"]),
                         faces("threshold set on people 0-29"),
                         faces("%s pairs, three ways" % m["pairs_text"]))
        self.HEADINGS = ("every pair of the unseen photos", "one person near, two people far",
                         "a threshold, chosen on people 0–29", "equal error rate, against pixels")

    def subtitle_for(self, k):
        m = self.meta
        e = m["eer_text"]
        return [
            "People 30–39 were never trained on: their 100 photos make %s pairs, %s of one "
            "person." % (m["pairs_text"], m["same_pairs_text"]),
            "Through the encoder, one person's pairs are mostly near and two people's far, "
            "but the piles still overlap.",
            "At %s, chosen on people 0–29: %d false accepts, %d false rejects; %s of %s "
            "right." % (m["threshold_text"], m["false_accept"], m["false_reject"],
                        m["right_text"], m["pairs_text"]),
            "Where both mistakes are equally common: %s for the encoder, %s for raw pixels."
            % (e["trained"], e["pixels"]),
        ][k]

    def fx(self, d):
        x0, x1, _b, _t = self.HIST
        return x0 + (x1 - x0) * min(d, self.HIST_MAX) / self.HIST_MAX

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        a = op * (self.at(0, 0.72) if k == 0 else 1.0)
        self.text(sc, 0.06, 0.4, "%s pairs: %s one person, %s two people"
                  % (m["pairs_text"], m["same_pairs_text"], m["diff_pairs_text"]), a, SIZE, "lm")
        # the 100 photos: through phase 1, gone early in phase 2 (before the piles grow)
        if k < 2:
            fx, fy, T = self.SHEET
            p = op * (self.at(0, 0.04) if k == 0 else 1.0 - smoothstep(0.0, 0.08, u))
            self.sheet(sc, self.sheet_rgb, fx, fy, 10 * T, 10 * T, p)
        h = op * self.kept(1)
        if h > 0.01:
            self.histogram(sc, k, u, h)

    def histogram(self, sc, k, u, op):
        m = self.meta
        x0, x1, base, tall = self.HIST
        grow = eased(u, 0.1, 0.32) if k == 1 else 1.0
        a = op * (self.at(1, 0.08) if k == 1 else 1.0)
        t = m["threshold"]
        wrong = 0.0 if k < 2 else (smoothstep(0.3, 0.45, u) if k == 2 else 1.0)
        self.piles(sc, np.asarray(m["bins"]), np.asarray(m["share_same"]),
                   np.asarray(m["share_diff"]), x0, x1, base, tall, a, grow=grow,
                   wrong_up=lambda c: c >= t, wrong_down=lambda c: c < t, wrong=wrong)
        self.text(sc, 0.06, base - tall * 0.5, "one person", a, SIZE, "lm", ink=DATA_INK)
        self.text(sc, 0.06, base + tall * 0.5, "two people", a, SIZE, "lm", ink=OTHER_INK)
        for v in (0.0, 0.5, 1.0, 1.5, 2.0):
            self.text(sc, self.fx(v), base + tall + 0.03, "%g" % v, a, SIZE, ink=QUIET)
        # the threshold, from phase 3 on
        b = op * (self.at(2, 0.06) if k == 2 else (1.0 if k > 2 else 0.0))
        if b > 0.01:
            xt = self.fx(t)
            self.vline(sc, xt, base - tall - 0.02, base + tall + 0.01, INK, b, width=2.2)
            self.text(sc, xt, base - tall - 0.045, m["threshold_text"], b, SIZE, strong=True)

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        m = self.meta
        fx, fy, T = self.SHEET
        self.text(sc, fx, fy + self.frac(0, 5 * T + 26)[1], "a row: one person", on(u, 0.12, op),
                  SIZE, ink=QUIET)
        self.text(sc, 0.255, 0.52, "100 × 99 ÷ 2 = %s" % m["pairs_text"], on(u, 0.4, op), HEAD,
                  "lm", strong=True)
        self.text(sc, 0.255, 0.585, "%s of one person" % m["same_pairs_text"], on(u, 0.55, op),
                  SIZE, "lm", ink=DATA_INK)
        self.text(sc, 0.255, 0.625, "%s of two people" % m["diff_pairs_text"], on(u, 0.6, op),
                  SIZE, "lm", ink=OTHER_INK)

    def phase_2(self, sc, u, op):
        """The two piles are in ``steady``."""
        self.text(sc, 0.46, 0.79, "each pile: shares of its pairs", on(u, 0.4, op), SIZE, "rm",
                  ink=QUIET)

    def phase_3(self, sc, u, op):
        m = self.meta
        y = self.TALLY_Y
        a = on(u, 0.42, op)
        self.text(sc, 0.06, y, "%d false accepts · %s of %s" % (m["false_accept"], m["far_text"],
                                                              m["diff_pairs_text"]),
                  a, SIZE, "lm", ink=WRONG_INK)
        b = on(u, 0.5, op)
        self.text(sc, 0.06, y + self.TALLY_STEP, "%d false rejects · %s of %s"
                  % (m["false_reject"], m["frr_text"], m["same_pairs_text"]), b, SIZE, "lm",
                  ink=WRONG_INK)
        c = on(u, 0.62, op)
        self.text(sc, 0.06, y + 2.2 * self.TALLY_STEP, "%s of %s right" % (m["right_text"],
                                                                         m["pairs_text"]),
                  c, HEAD, "lm", strong=True)

    def phase_4(self, sc, u, op):
        m = self.meta
        y0 = self.TALLY_Y
        for i, (key, name) in enumerate(EER_ROWS):
            y = y0 + self.TALLY_STEP * i
            a = on(u, 0.08 + 0.14 * i, op)
            self.text(sc, 0.06, y, name, a, SIZE, "lm", strong=key == "trained")
            self.hbar_at(sc, self.BAR_X, y, self.BAR_W * 0.3, 12.0, TRACK, a * 0.5, lift=0.25)
            w = self.BAR_W * m["eer"][key] * eased(u, 0.08 + 0.14 * i, 0.16 + 0.14 * i)
            self.hbar_at(sc, self.BAR_X, y, w, 12.0, DATA if key == "trained" else OTHER, a)
            self.text(sc, self.BAR_X + self.BAR_W * m["eer"][key] + 0.01, y,
                      m["eer_text"][key], a, SIZE, "lm", strong=key == "trained")
        mt = m["mistakes_text"]
        self.text(sc, 0.06, y0 + self.TALLY_STEP * 3.1, "threshold from people 0–29: %s "
                  "mistakes, pixels %s" % (mt["trained"], mt["pixels"]), on(u, 0.55, op), SIZE,
                  "lm", ink=QUIET)


ANIMATIONS = {"faces-a-face-is-numbers": FaceNumbers, "faces-an-embedding": Embedding,
              "faces-triplets": Triplets, "faces-verification": Verification}

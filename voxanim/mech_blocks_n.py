"""Animations 224-226, the first three of "Blocks and code: padding" (see
:mod:`voxanim.blocks`), drawn from :mod:`voxanim.study_blocks_n`.

* :class:`MapsShrink` (``padding-maps-shrink``): the course's first digit as tiles and a
  3 x 3 stamp (an orange outline) sliding over it: 26 places across 28, 26 down, 676 in all,
  so ``nn.Conv2d(1, 8, 3)`` gives maps of 26 x 26; then the coverage map, how many places
  read each pixel, filling as the stamp sweeps, and three pixels called out: a corner read
  once, an edge pixel 3 times, a middle one 9 (the stamp visits each of its places).
* :class:`ARing` (``padding-a-ring``): the same picture with ``padding=1``: a ring of tiles
  round the digit, 30 x 30; the ring is 0 and the paper -0.42 (a corner magnified as
  numbers); the stamp fits 28 places, so the maps keep 28 x 28; the coverage map again:
  corner 4, edge 6, middle 9: more at the border, still fewer than in the middle.
* :class:`SameAndValid` (``padding-same-and-valid``): a table filled row by row, k = 3, 5,
  7: the padding "same" needs, (k - 1) / 2, the side with "same" (28) and with "valid"
  (28 - (k - 1)), each row with its square to scale; then the rule.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). What stays across phases is drawn in
``steady``, so nothing blanks between them.
"""

from __future__ import annotations

from types import MappingProxyType

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
    Code,
    commas,
    exact,
    shades,
    val,
)
from .diagram import draw_image
from .easing import ease_in_out, smoothstep
from .mech import AXIS, PAPER, dark, soft
from .mech_blocks_c import Parts
from .mech_blocks_p import hatched
from .study_blocks_n import coverage_after

DATA_INK = dark(DATA, 0.3)
WEIGHT_INK = dark(WEIGHT, 0.35)
COUNT_INK = (0.40, 0.44, 0.50)        # the coverage map: paper (read 0 times) to slate (9)
COUNT_HI = 9.0


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def at(k, u, when):
    """0 before ``when`` = ``(phase, u0)``, fading up to 1 just after, 1 from then on."""
    pk, pu = when
    if k != pk:
        return 1.0 if k > pk else 0.0
    return smoothstep(pu, pu + 0.04, u)


class Pad(Parts):
    """What 224 and 225 share: the digit as tiles, a 3 x 3 stamp sliding over it, the count
    of places, and the coverage map with its corner, edge and middle called out."""

    T = 4.6                       # px per tile, digit and coverage map alike
    CY = 0.54                     # the cards' centre
    DX, MX = 0.13, 0.30           # the digit's card, the coverage map's card
    CX = 0.405                    # the callouts' column
    UNDER = 0.715                 # the labels under the cards
    BAND = 0.83                   # the running count
    SPOT_AT = ((0.04, 0.24), (0.3, 0.56), (0.62, 0.94))   # the last phase: each spot's turn

    # ------------------------------------------------------------------ setup
    def read(self):
        m = self.meta = self.study.metadata
        self.k, self.pad = m["k"], m["padding"]
        self.side = m["side"]
        self.n = m["across"]
        self.big_side = self.side + 2 * self.pad
        self.spots = [(name, m["spots"][name]) for name in ("corner", "edge", "middle")]

    # ------------------------------------------------------------------ geometry
    def pic_cell(self):
        """``cell(row, col)`` of the picture the stamp slides over (padded coordinates)."""
        n = self.big_side
        return self.cells(self.DX, self.CY + self.frac(0, 3.0)[1], n * self.T, n * self.T,
                          n, n)

    def map_cell(self):
        """``cell(row, col)`` of the coverage map (the real pixels)."""
        n = self.side
        return self.cells(self.MX, self.CY + self.frac(0, 3.0)[1], n * self.T, n * self.T,
                          n, n)

    def place(self, i):
        """The ``i``-th place of the stamp (reading order), top-left, padded coordinates."""
        return divmod(int(i), self.n)

    # ------------------------------------------------------------------ shapes
    def stamp(self, sc, cell, r, c, op, width=None):
        """The 3 x 3 stamp: an orange outline round tiles ``r..r+2`` x ``c..c+2``."""
        if op > 0.01:
            self.outline(sc, cell, r, c, self.k, self.k, self.T, op, ink=WEIGHT_INK, lift=1.1)

    def pixel_ring(self, sc, cell, r, c, op):
        if op > 0.01:
            x, y = cell(r, c)
            self.rect(sc, self.P(x, y), self.T + 5, self.T + 5, INK, op, border=INK,
                      border_w=2.2, radius=2, lift=1.2, fill_op=0.0)

    def count_card(self, sc, cov, op):
        """The coverage map: a grey card of tiles, paper (read 0 times) to slate (9)."""
        if op <= 0.01:
            return self.map_cell()
        w = self.side * self.T
        c = self.P(self.MX, self.CY)
        self.rect(sc, c, w + 14, w + 20, soft(AXIS, 0.84), op, border=AXIS, border_w=2.0,
                  radius=7, lift=0.2)
        self.rect(sc, self.shift(c, 0, -(w + 20) / 2 + 4.0), w + 8, 4.0, AXIS, op, radius=2,
                  lift=0.22)
        return self.tiles(sc, cov, self.MX, self.CY + self.frac(0, 3.0)[1], w, w, op,
                          ink=COUNT_INK, lo=0.0, hi=COUNT_HI, lift=0.204)

    def map_label(self, sc, op):
        self.text(sc, self.MX, self.UNDER, "times each pixel is read", op, SIZE,
                  ink=dark(AXIS, 0.3))

    def count(self, sc, steps, u, op, label):
        """The running count under the digit (never blank: swapped in place)."""
        self.running(sc, 0.06, self.BAND, steps, u, op, label=label, size=BIG)

    def small(self, sc, s, op):
        self.text(sc, 0.06, self.BAND + 0.055, s, op, SIZE, "lm")

    # ------------------------------------------------------------------ the sweep
    def sweep(self, u, a=0.06, b=0.84):
        """How many places the stamp has visited while it sweeps them all (0..n x n)."""
        total = self.n * self.n
        return int(np.ceil(total * float(np.clip((u - a) / (b - a), 0.0, 1.0))))

    def coverage(self, done):
        return coverage_after(done, self.side, self.k, self.pad).astype(float)

    def sweep_stamp(self, sc, u, op, a=0.06, b=0.84):
        """The stamp at the place it has reached, on the picture and on the map."""
        done = self.sweep(u, a, b)
        on = op * smoothstep(a - 0.03, a, u) * (1 - smoothstep(b + 0.02, b + 0.05, u))
        r, c = self.place(max(done - 1, 0))
        self.stamp(sc, self.pic_cell(), r, c, on)
        self.stamp(sc, self.map_cell(), r - self.pad, c - self.pad, on)
        return done

    # ------------------------------------------------------------------ corner, edge, middle
    def spot_state(self, i, u):
        """Spot ``i``'s turn in the last phase: ``(shown, visiting, count)``: whether it is
        called out yet, the place index the stamp is at (None outside its turn) and how many
        of its places have been counted so far."""
        a, b = self.SPOT_AT[i]
        places = self.spots[i][1]["places"]
        if u < a:
            return False, None, 0
        n = len(places)
        s = (u - a - 0.03) / (b - a - 0.06)
        if s < 0:
            return True, None, 0
        if s >= 1:
            return True, None, n
        j = min(int(s * n), n - 1)
        return True, j, j + 1

    def spots_phase(self, sc, u, op):
        """The stamp visits the places that read each spot, one spot at a time; each spot is
        ringed on the picture and on the map, and its count called out beside the map."""
        pc, mc = self.pic_cell(), self.map_cell()
        for i, (name, spot) in enumerate(self.spots):
            shown, j, n = self.spot_state(i, u)
            if not shown:
                continue
            a, b = self.SPOT_AT[i]
            on = op * smoothstep(a, a + 0.04, u)
            r, c = spot["pixel"]
            # on the picture, only during its turn
            turn = on * (1 - smoothstep(b, b + 0.03, u))
            self.pixel_ring(sc, pc, r + self.pad, c + self.pad, turn)
            if j is not None:
                pr, pcol = spot["places"][j]
                self.stamp(sc, pc, pr, pcol, turn)
            # on the map, and its callout, from then on
            self.callout(sc, mc, (r, c), name, str(n) if n else "", on)

    def callout(self, sc, cell, pixel, name, number, op):
        if op <= 0.01:
            return
        x, y = cell(*pixel)
        self.pixel_ring(sc, cell, pixel[0], pixel[1], op)
        up = self.ARROW_LIFT
        x0 = x + self.frac(self.T / 2 + 3.0)[0]
        self.polyline(sc, [self.P(x0, y, up), self.P(self.CX - 0.008, y, up)], 1.8, INK, op,
                      bias=0.3)
        self.text(sc, self.CX, y - 0.025, name, op, SIZE, "lm")
        if number:
            self.text(sc, self.CX, y + 0.02, number, op, HEAD, "lm", strong=True)


# ====================================================================== 224
class MapsShrink(Pad):
    """The digit and the stamp; 26 places across, 676 in all; the coverage map fills;
    corner 1, edge 3, middle 9."""

    TITLE = "Why maps shrink"
    PHASES = (("the stamp", 11.0), ("26 places", 15.0), ("times read", 14.0),
              ("the border", 16.0))
    LAST = "CNN mistakes"
    NEXT = "A ring of padding"
    STATION = ("forward",) * 4
    ACROSS = (0.08, 0.42)         # phase 2: across the top row
    DOWN = (0.52, 0.66)           # ... then down the right column
    SHAPE_AT = 0.72               # ... and the maps' shape

    def setup(self):
        self.read()
        m = self.meta
        self.img = np.asarray(self.study.panels[0].data["values"], float)
        self.full = np.asarray(self.study.panels[1].data["values"], float)
        change = m["shape_change"]
        self.CODE = (Code("pad-none", run=((0.0, None), (0.36, 0))),
                     Code("pad-none", run=((0.0, 0), (0.06, 1)),
                          out=((self.SHAPE_AT, change),)),
                     Code("pad-none", run=((0.0, 1),), out=((0.0, change),)),
                     Code("pad-none", run=((0.0, 1),), out=((0.0, change),)))
        tag = exact("a real digit, the places counted")
        self.EVIDENCE = (exact("a real digit, a 3 × 3 stamp"), tag, tag, tag)

    def heading_for(self, k, u):
        if k == 1:
            return "26 places across, not 28" if u < self.DOWN[0] else "26 × 26 places"
        return ("a 3 × 3 stamp on a 28 × 28 digit", "", "how often is each pixel read?",
                "the border is read less")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "x[0], the first training digit: 28 × 28 pixels. nn.Conv2d(1, 8, 3) slides a "
            "3 × 3 stamp over it.",
            "The stamp fits %d places across 28 and %d down: %s places, so each map is "
            "%d × %d." % (m["across"], m["across"], commas(m["places"]), m["across"],
                          m["across"]),
            "Count, for each pixel, how many of the %s places read it: darker means more."
            % commas(m["places"]),
            "A corner pixel is read by %d place, an edge pixel by %d, a middle pixel by %d."
            % (m["corner"], m["edge"], m["middle"]),
        ][k]

    # ------------------------------------------------------------------ state
    def across(self, k, u):
        """Phase 2: the stamp's place ``(row, col)`` and the places counted."""
        if u < self.DOWN[0]:
            c = int(round(eased(u, *self.ACROSS) * (self.n - 1)))
            return (0, c), c + 1
        r = int(round(eased(u, *self.DOWN) * (self.n - 1)))
        return (r, self.n - 1), None

    def count_now(self, k, u):
        """The running count under the digit: ``(steps, label, small line)``."""
        m = self.meta
        if k == 0:
            return [(0.0, "3 × 3")], "the stamp: 9 pixels", ""
        if k == 1:
            if u < self.ACROSS[0]:
                return [(0.0, "0")], "places across", ""
            if u < self.DOWN[0]:
                n = self.across(k, u)[1]
                small = m["across_sum"] if u >= self.ACROSS[1] + 0.02 else ""
                return [(0.0, str(n))], "places across", small
            return [(0.0, commas(m["places"]))], "places", m["places_sum"]
        if k == 2:
            done = self.sweep(u)
            return [(0.0, commas(done))], "places so far", ""
        return [(0.0, commas(m["places"]))], "places", m["places_sum"]

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        self.digit_card(sc, self.img, self.DX, self.CY, self.T, op)
        self.text(sc, self.DX, self.UNDER, "28 × 28", op, SIZE, ink=DATA_INK)
        # the running count
        steps, label, small = self.count_now(k, u)
        on = op if k > 0 else op * smoothstep(0.38, 0.44, u)
        self.count(sc, steps, 0.0, on, label)
        if small:
            self.small(sc, small, op)
        # the coverage map, from phase 3
        keep = op * self.kept(2)
        if keep > 0.01:
            cov = self.coverage(self.sweep(u)) if k == 2 else self.full
            self.count_card(sc, cov, keep)
            self.map_label(sc, keep)

    # ------------------------------------------------------------------ 1: the stamp
    def phase_1(self, sc, u, op):
        on = op * smoothstep(0.38, 0.44, u)
        self.stamp(sc, self.pic_cell(), 0, 0, on)
        a = op * smoothstep(0.55, 0.61, u)
        self.text(sc, 0.24, 0.44, "a 3 × 3 stamp:", a, SIZE, "lm")
        self.text(sc, 0.24, 0.48, "9 pixels at a time", a, SIZE, "lm")

    # ------------------------------------------------------------------ 2: 26 places
    def phase_2(self, sc, u, op):
        (r, c), _n = self.across(1, u)
        self.stamp(sc, self.pic_cell(), r, c, op)
        a = op * smoothstep(self.ACROSS[1] + 0.02, self.ACROSS[1] + 0.06, u)
        self.text(sc, 0.24, 0.44, "the last place", a, SIZE, "lm")
        self.text(sc, 0.24, 0.48, "ends at the edge", a, SIZE, "lm")
        b = op * smoothstep(self.SHAPE_AT, self.SHAPE_AT + 0.05, u)
        self.text(sc, 0.24, 0.57, "each place gives", b, SIZE, "lm")
        self.text(sc, 0.24, 0.61, "one number:", b, SIZE, "lm")
        self.text(sc, 0.24, 0.66, "maps of 26 × 26", b, HEAD, "lm", strong=True)

    # ------------------------------------------------------------------ 3: times read
    def phase_3(self, sc, u, op):
        self.sweep_stamp(sc, u, op)

    # ------------------------------------------------------------------ 4: the border
    def phase_4(self, sc, u, op):
        self.spots_phase(sc, u, op)


# ====================================================================== 225
class ARing(Pad):
    """The ring; 0 against -0.42; 28 places; the coverage map; corner 4, edge 6, middle 9."""

    TITLE = "A ring of padding"
    PHASES = (("the ring", 11.0), ("0, not −0.42", 13.0), ("28 places", 12.0),
              ("times read", 10.0), ("the border", 13.0))
    LAST = "Why maps shrink"
    NEXT = "Same and valid"
    STATION = ("forward",) * 5
    GROW = (0.42, 0.56)           # phase 1: the ring appears
    ZOOM = (0.318, 0.48)          # phase 2: the top-right corner as numbers
    BW, BH = 52.0, 26.0
    ACROSS = (0.08, 0.42)
    DOWN = (0.52, 0.66)
    SHAPE_AT = 0.72

    def setup(self):
        self.read()
        m = self.meta
        self.padded = np.asarray(self.study.panels[0].data["values"], float)
        self.img = self.padded[1:-1, 1:-1]
        self.full = np.asarray(self.study.panels[1].data["values"], float)
        change = m["shape_change"]
        self.CODE = (Code("pad-one", run=((0.0, None), (0.38, 0))),
                     Code("pad-one", run=((0.0, 0),)),
                     Code("pad-one", run=((0.0, 0), (0.06, 1)),
                          out=((self.SHAPE_AT, change),)),
                     Code("pad-one", run=((0.0, 1),), out=((0.0, change),)),
                     Code("pad-one", run=((0.0, 1),), out=((0.0, change),)))
        tag = exact("a real digit, the places counted")
        self.EVIDENCE = (exact("padding=1 on a real digit"),
                         exact("Normalize: (0 − 0.1307) ÷ 0.3081"), tag, tag, tag)

    def heading_for(self, k, u):
        if k == 2:
            return "28 places across: no loss" if u < self.DOWN[0] else "28 × 28 places"
        return ("padding=1: a ring round the digit", "the ring is 0, the paper −0.42", "",
                "how often is each pixel read?", "the border: more, still fewer")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "padding=1 adds a ring of zeros on every side: the 28 × 28 digit becomes "
            "%d × %d." % (m["big"], m["big"]),
            "After Normalize blank paper is %s, so a ring of 0 is not blank: it is a "
            "little ink." % val(m["paper_value"]),
            "On %d × %d the stamp fits %d places across and %d down: the maps stay "
            "%d × %d." % (m["big"], m["big"], m["across"], m["across"], m["across"],
                          m["across"]),
            "Count again how many of the %s places read each pixel: darker means more."
            % commas(m["places"]),
            "Now a corner pixel is read %d times and an edge pixel %d: more, but still "
            "fewer than %d." % (m["corner"], m["edge"], m["middle"]),
        ][k]

    # ------------------------------------------------------------------ state
    def grow(self, k, u):
        return 1.0 if k > 0 else eased(u, *self.GROW)

    def across(self, k, u):
        if u < self.DOWN[0]:
            c = int(round(eased(u, *self.ACROSS) * (self.n - 1)))
            return (0, c), c + 1
        r = int(round(eased(u, *self.DOWN) * (self.n - 1)))
        return (r, self.n - 1), None

    def count_now(self, k, u):
        m = self.meta
        if k == 0:
            return [(0.0, "%d × %d" % (m["big"], m["big"]))], "with the ring", m["big_sum"]
        if k == 1:
            return [(0.0, val(m["paper_value"]))], "blank paper after Normalize", \
                "(0 − 0.1307) ÷ 0.3081"
        if k == 2:
            if u < self.ACROSS[0]:
                return [(0.0, "0")], "places across", ""
            if u < self.DOWN[0]:
                n = self.across(k, u)[1]
                small = m["across_sum"] if u >= self.ACROSS[1] + 0.02 else ""
                return [(0.0, str(n))], "places across", small
            return [(0.0, commas(m["places"]))], "places", m["places_sum"]
        if k == 3:
            return [(0.0, commas(self.sweep(u)))], "places so far", ""
        return [(0.0, commas(m["places"]))], "places", m["places_sum"]

    # ------------------------------------------------------------------ the padded digit
    def padded_card(self, sc, s, ring, op):
        """The digit on a card that grows by a ring of tiles (``s``: 0..1), the ring's tiles
        (the real value 0, on the digit's shading, striped as padding) at opacity
        ``ring``."""
        T = self.T
        n = self.side + 2 * s
        self.card(sc, self.DX, self.CY, n * T + 14, n * T + 20, op)
        cy = self.CY + self.frac(0, 3.0)[1]
        self.tiles(sc, self.img, self.DX, cy, self.side * T, self.side * T, op, lo=DATA_LO,
                   hi=DATA_HI, lift=0.204)
        a = op * ring
        if a <= 0.01:
            return
        # the whole 30 x 30 over the digit (the same tiles inside, so it only adds the
        # ring): the ring shaded as 0 is on the digit's scale, striped grey as padding is
        # throughout the series (mech_blocks_p.hatched)
        big = self.big_side
        ring_mask = np.ones((big, big), bool)
        ring_mask[self.pad:-self.pad, self.pad:-self.pad] = False
        tex = hatched(shades(self.padded, DATA, DATA_LO, DATA_HI), ring_mask, 3, 3, 6)
        draw_image(self, sc, self.P(self.DX, cy), tex, big * T, big * T, a, grid=0.0,
                   lift=0.206)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        s = self.grow(k, u)
        self.padded_card(sc, s, s, op)
        label = "28 × 28" if s < 0.5 else "%d × %d" % (m["big"], m["big"])
        self.text(sc, self.DX, self.UNDER + self.frac(0, 2 * self.T * s)[1] / 2, label, op,
                  SIZE, ink=DATA_INK)
        steps, label, small = self.count_now(k, u)
        on = op if k > 0 else op * smoothstep(self.GROW[1], self.GROW[1] + 0.06, u)
        self.count(sc, steps, 0.0, on, label)
        if small:
            self.small(sc, small, on if k == 0 else op)
        # the corner magnified as numbers: phases 2 and 3
        keep = op * self.kept(1, 2)
        if keep > 0.01:
            on = keep if k > 1 else keep * smoothstep(0.08, 0.14, u)
            self.zoom(sc, on)
        # the coverage map, from phase 4
        keep = op * self.kept(3)
        if keep > 0.01:
            cov = self.coverage(self.sweep(u)) if k == 3 else self.full
            self.count_card(sc, cov, keep)
            self.map_label(sc, keep)

    def zoom(self, sc, op):
        """The padded digit's top-right 3 x 3, magnified as numbers, with lines to it."""
        m = self.meta
        cell = self.pic_cell()
        big = self.big_side
        _x0, y0, x1, y1 = self.outline(sc, cell, 0, big - 3, 3, 3, self.T, op, ink=INK,
                                      lift=1.15)
        zx, zy = self.ZOOM
        hw, hh = self.frac(3 * self.BW / 2 + 2, 3 * self.BH / 2 + 2)
        for a, b in (((x1, y0), (zx - hw, zy - hh)), ((x1, y1), (zx - hw, zy + hh))):
            self.arrow(sc, a, b, AXIS, op * 0.8, head=False, width=1.3)
        self.numbers(sc, m["corner3"], zx, zy, self.BW, self.BH, op, lo=DATA_LO, hi=DATA_HI)
        self.text(sc, zx - hw, zy + hh + 0.035, "0: the ring", op, SIZE, "lm", ink=DATA_INK)
        self.text(sc, zx - hw, zy + hh + 0.075, "%s: the paper" % val(m["paper_value"]), op,
                  SIZE, "lm", ink=DATA_INK)

    # ------------------------------------------------------------------ 1: the ring
    def phase_1(self, sc, u, op):
        a = op * smoothstep(self.GROW[1] + 0.04, self.GROW[1] + 0.1, u)
        self.text(sc, 0.245, 0.46, "a ring of 0s,", a, SIZE, "lm")
        self.text(sc, 0.245, 0.50, "1 tile wide,", a, SIZE, "lm")
        self.text(sc, 0.245, 0.54, "on every side", a, SIZE, "lm")

    # ------------------------------------------------------------------ 2: 0, not -0.42
    def phase_2(self, sc, u, op):
        a = op * smoothstep(0.55, 0.61, u)
        self.text(sc, 0.245, 0.685, "0 is more than −0.42:", a, SIZE, "lm", strong=True)
        self.text(sc, 0.245, 0.725, "a faint frame of ink", a, SIZE, "lm", strong=True)

    # ------------------------------------------------------------------ 3: 28 places
    def phase_3(self, sc, u, op):
        (r, c), _n = self.across(2, u)
        on = op * smoothstep(0.03, 0.07, u)
        self.stamp(sc, self.pic_cell(), r, c, on)
        b = op * smoothstep(self.SHAPE_AT, self.SHAPE_AT + 0.05, u)
        self.text(sc, 0.245, 0.685, "maps of 28 × 28:", b, SIZE, "lm")
        self.text(sc, 0.245, 0.725, "the size is kept", b, SIZE, "lm", strong=True)

    # ------------------------------------------------------------------ 4: times read
    def phase_4(self, sc, u, op):
        self.sweep_stamp(sc, u, op)

    # ------------------------------------------------------------------ 5: the border
    def phase_5(self, sc, u, op):
        self.spots_phase(sc, u, op)


# ====================================================================== 226
class SameAndValid(Parts):
    """A table filled row by row, k = 3, 5, 7, each row with a square to scale; the rule."""

    TITLE = "Same and valid"
    PHASES = (("k = 3", 16.0), ("k = 5", 13.0), ("k = 7", 13.0), ("the rule", 16.0))
    LAST = "A ring of padding"
    NEXT = "What goes in the border"
    STATION = ("forward",) * 4
    COLS = MappingProxyType(dict(k=0.075, pad=0.15, same=0.24, valid=0.32, square=0.42))
    HEAD_ROW = 0.39
    ROWS = (0.475, 0.6, 0.725)
    S = 2.15                      # px per pixel of the squares, to scale
    WORK = 0.85                   # the working line under the table
    # per row phase: k, padding, same, valid, the square, the valid square shrinks
    K_AT, PAD_AT, SAME_AT, VALID_AT, SQ_AT, SHRINK = 0.06, 0.2, 0.36, 0.56, 0.74, (0.8, 0.92)
    LEAD = 0.22                   # phase 1 waits this long (the "last time" heading) first
    RULE = 0.38                   # phase 4: from "same" to "valid"

    def setup(self):
        m = self.meta = self.study.metadata
        self.rows = m["rows"]
        self.side = m["side"]
        codes = []
        for i, r in enumerate(self.rows):
            p0, p1 = r["printed"]
            lead = self.LEAD if i == 0 else 0.0

            def t(v, lead=lead):           # a row phase's time, after phase 1's wait
                return lead + (1 - lead) * v
            start = ((0.0, None),) if i == 0 else ()
            codes.append(Code("pad-same",
                              run=start + ((t(self.K_AT), 0), (t(self.PAD_AT - 0.04), 1),
                                           (t(self.SAME_AT - 0.04), 4),
                                           (t(self.VALID_AT - 0.06), 3), (t(self.VALID_AT), 5)),
                              out=((0.0, ""), (t(self.SAME_AT), p0),
                                   (t(self.VALID_AT), "%s\n%s" % (p0, p1)))))
        last = self.rows[-1]["printed"]
        codes.append(Code("pad-same", run=((0.0, 2), (self.RULE, 3)),
                          out=((0.0, "%s\n%s" % tuple(last)),)))
        self.CODE = tuple(codes)
        tag = exact("the sides, counted")
        self.EVIDENCE = (tag,) * 3 + (exact("(k − 1) ÷ 2 and k − 1"),)

    def heading_for(self, k, u):
        if k < 3:
            kk = self.rows[k]["k"]
            return "k = %d: a %d × %d stamp" % (kk, kk, kk)
        return "same pads (k − 1) ÷ 2" if u < self.RULE else "valid, the default, loses k − 1"

    def subtitle_for(self, k):
        if k < 3:
            r = self.rows[k]
            if k == 0:
                return ('For a %d × %d stamp padding="same" pads %d and keeps %d; "valid" '
                        "pads nothing and gives %d." % (r["k"], r["k"], r["pad"], r["same"],
                                                        r["valid"]))
            return ('A %d × %d stamp: "same" pads %d and keeps %d; "valid" loses %d and '
                    "gives %d." % (r["k"], r["k"], r["pad"], r["same"], r["lost"],
                                   r["valid"]))
        return '"same" pads (k − 1) ÷ 2 on every side; "valid", the default, loses k − 1.'

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        c = self.COLS
        for key, name in (("k", "k"), ("pad", "padding"), ("same", "same"),
                          ("valid", "valid"), ("square", "to scale")):
            self.text(sc, c[key], self.HEAD_ROW, name, op, SIZE, strong=key != "square",
                      ink=None if key != "square" else dark(AXIS, 0.3))
        v = self.local(k, u)
        for i, r in enumerate(self.rows):
            self.row(sc, i, r, k, v, op)
        # the working line: phases 1-3
        keep = op * self.kept(0, 2)
        if keep > 0.01 and k < 3:
            label, text = self.work(k, v)
            on = keep * at(k, v, (k, self.PAD_AT))
            self.running(sc, 0.06, self.WORK, [(0.0, text)], 0.0, on, label=label, size=HEAD)

    def local(self, k, u):
        """``u`` of a row's phase as the row's timings read it (phase 1 starts later)."""
        if k == 0:
            return max(0.0, (u - self.LEAD) / (1 - self.LEAD))
        return u

    def work(self, k, u):
        r = self.rows[k]
        if u < self.VALID_AT - 0.06:
            return "same pads (k − 1) ÷ 2 on every side", r["pad_sum"]
        return "valid loses k − 1", r["valid_sum"]

    def row(self, sc, i, r, k, u, op):
        c = self.COLS
        y = self.ROWS[i]
        if k < i:
            return
        if k == i:                                  # the row under way: a band behind it
            band = op * smoothstep(0.02, 0.06, u)
            self.rect(sc, self.P(0.255, y), self.span(0.4)[0], 62.0, soft(AXIS, 0.86), band,
                      border=soft(AXIS, 0.5), border_w=1.0, radius=6, lift=0.05)
        for key, text, when in (("k", r["k"], self.K_AT), ("pad", r["pad"], self.PAD_AT),
                                ("same", r["same"], self.SAME_AT),
                                ("valid", r["valid"], self.VALID_AT)):
            a = op * at(k, u, (i, when))
            self.text(sc, c[key], y, str(text), a, HEAD, strong=True,
                      ink=DATA_INK if key in ("same", "valid") else None)
        # the square to scale: 28 (same) outlined, the valid square inside it
        a = op * at(k, u, (i, self.SQ_AT))
        if a <= 0.01:
            return
        s = 1.0 if k > i else eased(u, *self.SHRINK)
        outer = self.side * self.S
        inner = (self.side - (self.side - r["valid"]) * s) * self.S
        p = self.P(c["square"], y)
        self.rect(sc, p, outer, outer, PAPER, a, border=DATA_INK, border_w=1.6, radius=1,
                  lift=0.2)
        self.rect(sc, p, inner, inner, soft(DATA, 0.55), a, border=DATA, border_w=1.2,
                  radius=1, lift=0.3)

    # ------------------------------------------------------------------ 1-3: a row each
    def phase_1(self, sc, u, op):
        pass

    def phase_2(self, sc, u, op):
        pass

    def phase_3(self, sc, u, op):
        pass

    # ------------------------------------------------------------------ 4: the rule
    def phase_4(self, sc, u, op):
        c = self.COLS
        a = op * smoothstep(0.08, 0.14, u)
        self.text(sc, c["pad"], self.WORK, "(k − 1) ÷ 2", a, SIZE, strong=True)
        self.text(sc, c["same"], self.WORK, "28", a, SIZE, strong=True)
        b = op * smoothstep(self.RULE, self.RULE + 0.06, u)
        self.text(sc, c["valid"], self.WORK, "28 − (k − 1)", b, SIZE, strong=True)
        self.text(sc, c["k"], self.WORK, "rule:", a, SIZE)
        top, bottom = self.ROWS[0] - 0.05, self.WORK + 0.03
        for key, when, w in (("pad", 0.12, 96.0), ("valid", self.RULE + 0.04, 104.0)):
            ring = op * smoothstep(when, when + 0.06, u)
            self.rect(sc, self.P(c[key], (top + bottom) / 2), w,
                      self.span(0, bottom - top)[1], INK, ring, border=INK, border_w=2.0,
                      radius=8, lift=1.2, fill_op=0.0)


ANIMATIONS = {"padding-maps-shrink": MapsShrink, "padding-a-ring": ARing,
              "padding-same-and-valid": SameAndValid}

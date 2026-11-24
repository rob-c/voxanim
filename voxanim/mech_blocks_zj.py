"""Animation 279 of "Blocks and code", "Jets as sets of particles" (see :mod:`voxanim.blocks`),
drawn from :mod:`voxanim.study_blocks_zj`: made-up jets as sets of particles
(``examples/jets_as_sets.py``), read by an MLP as a list and by a Deep Sets network as a set.

* :class:`JetsAsSets` (``jets-as-sets``): two made-up jets in (Δη, Δφ), a dot per particle
  sized by its momentum fraction z, one prong (a toy QCD jet) and two prongs (a toy W); the
  two-prong one becomes its list, row by row, hardest first, padded with zeros (striped, as in
  the padding animations) to 20 rows, with its mask; the MLP reading the 60 numbers in a row;
  Deep Sets running phi on each real row, adding them, rho reading the sum; both answers for
  this jet beside them, with the held-out accuracy; the rows shuffled: the MLP's answer moves,
  Deep Sets' sum and answer do not; over 4,000 held-out jets, the accuracies sorted and
  shuffled and the answers that flip; the rows sorted again, and the two fixes.

The alphabet: the particles and every list of numbers are data (blue tiles; deeper: bigger);
padding is grey stripes; the networks are weights (orange); "you are here" stays off, as no
phase is a training step. The list stays on the left all through; the rows move when shuffled
and when sorted again.
"""

from __future__ import annotations

import numpy as np

from .blocks import (
    DATA,
    HEAD,
    INK,
    QUIET,
    SIZE,
    WEIGHT,
    Blocks,
    Code,
    about_data,
    exact,
    measured,
    shades,
)
from .diagram import draw_image
from .easing import ease_in_out, smoothstep
from .mech import AXIS, GRID, PAPER, RED, TRACK, dark, soft
from .mech_blocks_p import HATCH, hatched

ON = "made-up data"
DATA_INK = dark(DATA, 0.3)
NET = WEIGHT                     # the networks: orange
NET_INK = dark(WEIGHT, 0.35)
MOVED_INK = dark(RED, 0.2)       # an answer that moved
PAD_INK = dark(HATCH, 0.2)

# ---------------------------------------------------------------------- the layout
W_PX, H_PX = 960.0, 540.0
J_MAX = 20                       # the list's rows (study_jets.MAX; checked by the tests)
BX0 = 0.06                       # the list's left edge
TILE_W, ROW_H = 30.0, 11.0       # px per column and per row of the list
TOP = 0.405                      # the list's top
HEADS_Y = 0.383                  # the column names over it
MASK_GAP = 16.0                  # px between the list and its mask column
MASK_W = 10.0
STRIP_X = 0.215                  # the MLP's 60 numbers in a row: a column of 3 x 20 tiles
STRIP_W = 12.0
CARD = (0.29, 0.6)              # the MLP's card
CARD_W, CARD_H = 70.0, 100.0
PHI_X, PHI_W = 0.265, 96.0       # Deep Sets: phi of each row, 32 numbers
SUM_Y, SUM_H = 0.855, 12.0       # their sum
RHO = (0.35, 0.855)               # rho's box
SB_X = 0.36                      # the scoreboard: both answers for this jet
SB_Y = dict(head=0.405, mlp=0.45, ds=0.62)
JET = 160.0                      # a jet's (Δη, Δφ) panel, px
JET_Q, JET_W = (0.15, 0.585), (0.37, 0.585)
BAR_X, BAR_LEN = 0.312, 96.0     # the accuracy bars: start, px for 100%
TEXT_X = 0.205                   # the bars' and the fixes' words
BAR_Y = dict(mlp=0.405, mlp_sorted=0.45, mlp_shuffled=0.49, mlp_again=0.53, ds=0.59,
             ds_sorted=0.635, ds_shuffled=0.675)
FIX_Y = (0.735, 0.785, 0.83, 0.875)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on(u, a, op, width=0.05):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


class JetsAsSets(Blocks):
    """The jets; the list; the MLP; Deep Sets; one jet shuffled; 4,000 jets; two fixes."""

    TITLE = "Jets as sets of particles"
    PHASES = (("jets", 8.5), ("a list", 8.5), ("an MLP", 8.5), ("Deep Sets", 9.5),
              ("shuffled", 9.5), ("4,000 jets", 8.0), ("two fixes", 9.0))
    LAST = "The Asimov dataset"
    NEXT = "Newton's step and gradient descent"
    ROWS_U = (0.15, 0.55)        # phase 2: the real rows fill in, one after another
    SHUFFLE_U = (0.08, 0.36)     # phase 5: the rows move to their shuffled places
    SWAP_U = 0.45                # phase 5: the answers change
    BACK_U = (0.04, 0.22)        # phase 7: the rows move back
    AGAIN_U = (0.22, 0.32)       # phase 7: the MLP on the list sorted again
    FIX_U = (0.3, 0.42, 0.55)    # phase 7: the two fixes, then graph networks
    PHI_U = (0.06, 0.44)         # phase 4: phi runs down the 20 rows
    MASK_U = 0.48                # phase 4: the mask sets the pads' rows to 0
    STRIP_U = (0.05, 0.4)        # phase 3: the rows are copied into one column

    def setup(self):
        m = self.meta = self.study.metadata
        f = m["followed"]
        self.n = f["n"]
        self.rows = np.asarray(f["rows"], float)               # [20, 3], sorted
        self.mask = np.asarray(f["mask"], bool)
        self.perm = list(f["perm"])
        # where each sorted row goes when shuffled: the row now at slot s was row perm[s]
        self.slot_shuffled = np.zeros(self.n)
        for s, j in enumerate(self.perm):
            self.slot_shuffled[j] = s
        self.zmax = float(self.rows[:, 0].max())
        self.phi = np.asarray(m["phi_rows"], float)
        self.phi_pad = np.asarray(m["phi_pad"], float)
        self.phi_hi = float(max(self.phi.max(), self.phi_pad.max()))
        self.sum = np.asarray(m["phi_sum"], float)
        self.sum_s = np.asarray(m["phi_sum_shuffled"], float)
        self.card = np.asarray(m["mlp_first"], float)
        pt, rt = m["p_text"], m["right_text"]
        self.CODE = (
            Code("js-jet", run=((0.0, 1), (0.1, 2), (0.2, 6), (0.35, 4), (0.6, 7))),
            Code("js-pad", run=((0.0, 1), (0.08, 2), (self.ROWS_U[0], 3), (0.62, 4))),
            Code("js-mlp", run=((0.0, 0), (0.42, 1), (0.55, 3))),
            Code("js-sum", run=((0.0, 1), (self.MASK_U, 2), (0.58, 3))),
            Code("js-shuffle", run=((0.0, 3), (self.SHUFFLE_U[0], 4))),
            Code("js-measure", run=((0.0, 1), (0.1, 2), (0.35, 3), (0.65, 4))),
            Code("js-again", run=((0.0, 1), (self.BACK_U[0], 2), (self.AGAIN_U[0], 3)),
                 out=((self.AGAIN_U[0] + 0.03, "True"),)))
        self.EVIDENCE = (
            about_data("made-up jets, 2 of the %s held out" % format(m["n_test"], ",")),
            exact("%d real rows + %d pads = %d" % (self.n, J_MAX - self.n, J_MAX)),
            measured("held-out accuracy, one run", ON),
            measured("held-out accuracy, one run", ON),
            measured("this jet, shuffled once", ON),
            measured("%s held-out jets, shuffled" % format(m["n_test"], ","), ON),
            exact("sorted again: the same list, the same p"))
        self.texts = dict(pt=pt, rt=rt)

    def station_for(self, k):
        return None

    def heading_for(self, k, u):
        return ("made-up jets: sets of particles",
                "one jet as a list, padded to %d rows" % J_MAX,
                "an MLP reads the 60 numbers in one order",
                "Deep Sets: φ on each particle, add, ρ",
                "shuffle the rows: the same jet",
                "%s held-out jets, each shuffled" % format(self.meta["n_test"], ","),
                "two fixes: sort first, or build it in")[k]

    def subtitle_for(self, k):
        m = self.meta
        pt, rt = m["p_text"], m["right_text"]
        return [
            "Made-up jets: each particle has a momentum fraction z (the dot's size) and "
            "offsets Δη, Δφ from the jet's axis. One prong or two?",
            "As in the padding animations: %d particles become rows sorted by z, zeros "
            "pad them to %d rows, and a mask marks the real ones." % (self.n, J_MAX),
            "An MLP reads the 60 numbers in this one order: p, its chance of two prongs, is "
            "%s here; %s of the held-out jets right." % (pt["mlp"], rt["mlp"]),
            "Deep Sets runs one small network φ on each real row, adds them up, and ρ reads "
            "the sum: p = %s, and %s right." % (pt["ds"], rt["ds"]),
            "The same %d particles in a new order: the MLP's p falls from %s to %s, one of the "
            "%s jets that flip. Deep Sets: p stays %s." % (self.n, pt["mlp"], pt["mlp_shuffled"],
                                                          m["changed_text"], pt["ds"]),
            "Shuffled, the MLP gets %s right: %s of %s answers flip, though most move little. "
            "Deep Sets keeps %s: rounding moves it under %s." % (
                rt["mlp_shuffled"], m["changed_text"], format(m["n_test"], ","), rt["ds"],
                m["ds_bound_text"]),
            "Two fixes: sort before the MLP (%s again), or build the symmetry in. Graph "
            "networks also pass messages between particles." % rt["mlp_again"],
        ][k]

    # ------------------------------------------------------------------ time
    def at(self, k, u, width=0.4):
        """1 once phase ``k`` has reached ``u`` (fading in over ``width`` seconds)."""
        a, b = self.spans[k]
        t0 = a + (b - a) * u
        return smoothstep(t0, t0 + width, getattr(self, "_t", 0.0))

    def ease_in(self, k, u0, u1):
        a, b = self.spans[k]
        t0, t1 = a + (b - a) * u0, a + (b - a) * u1
        return ease_in_out(float(np.clip((getattr(self, "_t", 0.0) - t0) / (t1 - t0), 0, 1)))

    def slots(self):
        """Each real row's place in the list now (sorted row j: slot j, or its shuffled slot,
        or between them while it moves)."""
        k, _u = self.where(getattr(self, "_t", 0.0))
        sorted_ = np.arange(self.n, dtype=float)
        if k < 4:
            return sorted_
        if k == 4:
            g = self.ease_in(4, *self.SHUFFLE_U)
        elif k == 5:
            g = 1.0
        else:
            g = 1.0 - self.ease_in(6, *self.BACK_U)
        return sorted_ + (self.slot_shuffled - sorted_) * g

    # ------------------------------------------------------------------ geometry
    def col_x(self, c):
        return BX0 + (c + 0.5) * TILE_W / W_PX

    def mask_x(self):
        return BX0 + (3 * TILE_W + MASK_GAP + MASK_W / 2) / W_PX

    def row_y(self, slot):
        return TOP + (slot + 0.5) * ROW_H / H_PX

    def bottom(self):
        return TOP + J_MAX * ROW_H / H_PX

    def row_rgb(self, row):
        """One row's three tiles: z on 0..the jet's largest z, the offsets on -span..span."""
        z = shades(row[:1], DATA, 0.0, self.zmax)
        off = shades(row[1:], DATA, -self.meta["span"], self.meta["span"])
        return np.concatenate([z, off])[None]                  # [1, 3, 3]

    def image(self, sc, rgb, fx, fy, w, h, op, lift=0.3, grid=0.25):
        if op > 0.01:
            draw_image(self, sc, self.P(float(fx), float(fy)), np.asarray(rgb, np.float32), w,
                       h, op, grid=grid, lift=lift)

    def pads(self, sc, fx, top_slot, w, cols, op, lift=0.3):
        """Striped paper for the pad rows ``top_slot..19``, ``w`` px wide, ``cols`` tiles."""
        rows = J_MAX - top_slot
        if op <= 0.01 or rows <= 0:
            return
        tex = hatched(np.full((rows, cols, 3), PAPER, np.float32), np.ones((rows, cols), bool),
                      int(ROW_H), max(1, round(w / cols)), 5, border=0.12)
        y = TOP + (top_slot + rows / 2) * ROW_H / H_PX
        draw_image(self, sc, self.P(fx, y), tex, w, rows * ROW_H, op, grid=0.0, lift=lift)

    def lifted(self, fx, fy):
        return self.P(float(fx), float(fy), self.ARROW_LIFT)

    # ------------------------------------------------------------------ the jets
    def jet_panel(self, sc, centre, rows, n, op, caption, ring=None):
        """A jet in (Δη, Δφ): a dot per particle, its area by z; ``ring``: one dot ringed."""
        if op <= 0.01:
            return
        cx, cy = centre
        span = self.meta["span"]
        hw, hh = self.frac(JET / 2, JET / 2)
        self.rect(sc, self.P(cx, cy), JET, JET, PAPER, op, border=GRID, border_w=1.2,
                  radius=6, lift=0.2)
        self.polyline(sc, [self.P(cx - hw, cy, 0.25), self.P(cx + hw, cy, 0.25)], 1.0, AXIS,
                      0.45 * op, bias=0.0)
        self.polyline(sc, [self.P(cx, cy - hh, 0.25), self.P(cx, cy + hh, 0.25)], 1.0, AXIS,
                      0.45 * op, bias=0.0)
        self.text(sc, cx + hw - 0.004, cy - 0.022, "Δη", op, SIZE, "rm", ink=QUIET)
        self.text(sc, cx + 0.008, cy - hh + 0.022, "Δφ", op, SIZE, "lm", ink=QUIET)
        for j in range(n):
            z, de, dp = rows[j]
            fx, fy = cx + de / span * hw, cy - dp / span * hh
            r = 2.2 + 13.0 * np.sqrt(z)
            self.disk(sc, self.P(fx, fy), r, DATA, op, lift=1.5, outline=0.18)
            if ring == j:
                self.rect(sc, self.P(fx, fy), 2 * r + 8, 2 * r + 8, INK, op, border=INK,
                          border_w=2.0, radius=r + 4, lift=1.6, fill_op=0.0)
        self.text(sc, cx, cy + hh + 0.035, caption, op, SIZE, strong=True)
        self.text(sc, cx, cy + hh + 0.075, "%d particles" % n, op, SIZE, ink=QUIET)

    # ------------------------------------------------------------------ the list
    def draw_list(self, sc, op, reveal=None, ring=None):
        """The list: its real rows at their places now, the pads striped, the mask beside it;
        ``reveal``: how many real rows are filled in yet (phase 2; the others are still
        zeros, striped)."""
        if op <= 0.01:
            return
        w = 3 * TILE_W
        x0 = self.col_x(1)
        filled = self.n if reveal is None else reveal
        for s in range(int(np.floor(filled)), self.n):
            self.pads_one(sc, x0, s, w, 3, op)
        self.pads(sc, x0, self.n, w, 3, op)
        slots = self.slots()
        for j in range(min(int(np.ceil(filled)), self.n)):
            a = op * float(np.clip(filled - j, 0.0, 1.0))
            self.image(sc, self.row_rgb(self.rows[j]), x0, self.row_y(slots[j]), w, ROW_H, a,
                       lift=0.32)
        mid = TOP + J_MAX * ROW_H / H_PX / 2
        self.rect(sc, self.P(x0, mid), w + 5, J_MAX * ROW_H + 5, DATA_INK, op,
                  border=DATA_INK, border_w=1.3, radius=2, lift=0.25, fill_op=0.0)
        if ring is not None:
            self.rect(sc, self.P(x0, self.row_y(slots[ring])), w + 6, ROW_H + 5, INK, op,
                      border=INK, border_w=2.0, radius=2, lift=1.0, fill_op=0.0)
        for c, name in enumerate(("z", "Δη", "Δφ")):
            self.text(sc, self.col_x(c), HEADS_Y, name, op, SIZE, ink=DATA_INK)
        self.text(sc, BX0, self.bottom() + 0.03, "%d real, %d pads" % (self.n, J_MAX - self.n),
                  op, SIZE, "lm", ink=DATA_INK)

    def pads_one(self, sc, fx, slot, w, cols, op):
        tex = hatched(np.full((1, cols, 3), PAPER, np.float32), np.ones((1, cols), bool),
                      int(ROW_H), max(1, round(w / cols)), 5, border=0.12)
        draw_image(self, sc, self.P(fx, self.row_y(slot)), tex, w, ROW_H, op, grid=0.0,
                   lift=0.3)

    def draw_mask(self, sc, op):
        if op <= 0.01:
            return
        x = self.mask_x()
        for s in range(J_MAX):
            real = s < self.n
            self.rect(sc, self.P(x, self.row_y(s)), MASK_W - 2, ROW_H - 2,
                      DATA if real else PAPER, op,
                      border=dark(DATA, 0.45) if real else AXIS, border_w=1.0, radius=1.5,
                      lift=0.3)
        self.text(sc, x, HEADS_Y, "mask", op, SIZE, ink=DATA_INK)

    def strip(self, sc, op, reveal=1.0, ring=None):
        """The MLP's input: each row's 3 numbers one under another, 60 in a column, the
        rows at their places now; ``ring``: one row's 3 tiles outlined."""
        if op <= 0.01:
            return
        slots = self.slots()
        shown = J_MAX * reveal
        for j in range(self.n):
            if j >= shown:
                break
            rgb = np.transpose(self.row_rgb(self.rows[j]), (1, 0, 2))     # [3, 1, 3]
            self.image(sc, rgb, STRIP_X, self.row_y(slots[j]), STRIP_W, ROW_H, op, lift=0.32,
                       grid=0.0)
        rest = int(np.clip(np.ceil(shown) - self.n, 0, J_MAX - self.n))
        if rest > 0:
            tex = hatched(np.full((rest * 3, 1, 3), PAPER, np.float32),
                          np.ones((rest * 3, 1), bool), 3, int(STRIP_W), 5)
            y = TOP + (self.n + rest / 2) * ROW_H / H_PX
            draw_image(self, sc, self.P(STRIP_X, y), tex, STRIP_W, rest * ROW_H, op,
                       grid=0.0, lift=0.3)
        mid = TOP + J_MAX * ROW_H / H_PX / 2
        self.rect(sc, self.P(STRIP_X, mid), STRIP_W + 4, J_MAX * ROW_H + 4, DATA_INK, op,
                  border=DATA_INK, border_w=1.2, radius=2, lift=0.25, fill_op=0.0)
        if ring is not None:
            self.rect(sc, self.P(STRIP_X, self.row_y(slots[ring])), STRIP_W + 6, ROW_H + 5, INK,
                      op, border=INK, border_w=2.0, radius=2, lift=1.0, fill_op=0.0)
        self.text(sc, STRIP_X - 0.009, HEADS_Y, "60 in a row", op, SIZE, "lm", ink=DATA_INK)

    def sum_strip(self, sc, values, op, label="sum"):
        if op <= 0.01:
            return
        rgb = shades(np.asarray(values)[None], DATA, 0.0, float(np.max(self.sum)))
        self.image(sc, rgb, PHI_X, SUM_Y, PHI_W, SUM_H, op, lift=0.32, grid=0.0)
        self.rect(sc, self.P(PHI_X, SUM_Y), PHI_W + 4, SUM_H + 4, DATA_INK, op,
                  border=DATA_INK, border_w=1.2, radius=2, lift=0.25, fill_op=0.0)
        self.text(sc, PHI_X - self.frac(PHI_W / 2 + 8)[0], SUM_Y, label, op, SIZE, "rm",
                  ink=DATA_INK)

    def box(self, sc, fx, fy, w, h, text, op):
        """A network: an orange box with its name."""
        if op <= 0.01:
            return
        self.rect(sc, self.P(fx, fy), w, h, soft(NET, 0.55), op, border=NET_INK, border_w=1.6,
                  radius=6, lift=0.4)
        self.label(sc, self.P(fx, fy, 0.45), text, SIZE, opacity=op, ink=NET_INK)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        o = m["other"]
        # the two jets: the one-prong one in phase 1, the followed one in phases 1 and 2
        ring = None
        reveal = None
        if k == 1:
            reveal = self.n * self.ease_in(1, *self.ROWS_U)
            j = int(reveal)
            if 0.0 < reveal < self.n and self.at(1, self.ROWS_U[0], 0.2) > 0.5:
                ring = j
        elif k == 2:
            ring = self.strip_ring()
        q = op * self.kept(0, 0) * self.at(0, 0.04)
        self.jet_panel(sc, JET_Q, np.asarray(o["rows"]), o["n"], q, "one prong · toy QCD")
        w = op * self.kept(0, 1) * self.at(0, 0.3)
        self.jet_panel(sc, JET_W, self.rows, self.n, w, "two prongs · toy W", ring=ring)
        # the list, from phase 2 on
        a = op * self.kept(1) * (self.at(1, 0.06) if k == 1 else 1.0)
        self.draw_list(sc, a, reveal=reveal, ring=ring)
        if k == 2:
            ring = None
        self.draw_mask(sc, a * (self.at(1, 0.62) if k == 1 else 1.0))
        # the scoreboard: this jet's p from each model (phases 3-5)
        self.scoreboard(sc, k, u, op * self.kept(2, 4))
        # the accuracy bars (phases 6-7)
        self.bars(sc, k, u, op * self.kept(5, 6))

    def scoreboard(self, sc, k, u, op):
        """This jet's p from each model, with its held-out accuracy under it; in phase 5 the
        p swaps in place (never blank) and "was ..." takes the accuracy's place."""
        if op <= 0.01:
            return
        pt, rt = self.meta["p_text"], self.meta["right_text"]
        show = dict(mlp=self.at(2, 0.6) if k == 2 else 1.0,
                    ds=0.0 if k == 2 else (self.at(3, 0.74) if k == 3 else 1.0))
        acc = dict(mlp=self.at(2, 0.75) if k == 2 else 1.0,
                   ds=self.at(3, 0.84) if k == 3 else 1.0)
        g = self.at(4, self.SWAP_U, 0.3) if k == 4 else 0.0
        swap = k == 4 and u >= self.SWAP_U
        self.text(sc, SB_X, SB_Y["head"], "p: two prongs?", op * show["mlp"], SIZE, "lm",
                  ink=QUIET)
        for name, key in (("MLP", "mlp"), ("Deep Sets", "ds")):
            a, y = op * show[key], SB_Y[key]
            if a <= 0.01:
                continue
            moved = key == "mlp"
            dip = 1.0
            if k == 4 and moved:
                dip = min(1.0, abs(self._t - self.time_of(4, self.SWAP_U)) / 0.15)
            self.text(sc, SB_X, y, name, a, SIZE, "lm", strong=True)
            self.text(sc, SB_X, y + 0.045, "p = " + (pt[key + "_shuffled"] if swap else pt[key]),
                      a * dip, HEAD, "lm", ink=MOVED_INK if swap and moved else INK)
            self.text(sc, SB_X, y + 0.09, rt[key] + " right", a * acc[key] * (1.0 - g), SIZE,
                      "lm", ink=QUIET)
            self.text(sc, SB_X, y + 0.09, "was " + pt[key], a * g, SIZE, "lm",
                      ink=MOVED_INK if moved else QUIET)

    def time_of(self, k, u):
        a, b = self.spans[k]
        return a + (b - a) * u

    def bars(self, sc, k, u, op):
        if op <= 0.01:
            return
        m = self.meta
        right, rt = m["right"], m["right_text"]
        grow = self.ease_in(5, 0.08, 0.3) if k == 5 else 1.0
        rows = (("mlp", "MLP", None), ("mlp_sorted", "sorted", "mlp"),
                ("mlp_shuffled", "shuffled", "mlp_shuffled"),
                ("mlp_again", "sorted again", "mlp_again"),
                ("ds", "Deep Sets", None), ("ds_sorted", "sorted", "ds"),
                ("ds_shuffled", "shuffled", "ds_shuffled"))
        for key, label, value in rows:
            y = BAR_Y[key]
            a = op
            g = grow
            if key == "mlp_again":
                if k < 6:
                    continue
                a = op * self.at(6, self.AGAIN_U[0])
                g = self.ease_in(6, *self.AGAIN_U)
            if value is None:
                self.text(sc, TEXT_X, y, label, a, SIZE, "lm", strong=True)
                continue
            self.text(sc, TEXT_X, y, label, a, SIZE, "lm", ink=QUIET)
            at = self.P(BAR_X, y)
            self.rect(sc, self.shift(at, BAR_LEN / 2, 0), BAR_LEN, 12.0, TRACK, 0.5 * a,
                      radius=3, lift=0.25)
            length = max(BAR_LEN * right[value] * g, 0.5)
            ink = MOVED_INK if value == "mlp_shuffled" else DATA
            self.rect(sc, self.shift(at, length / 2, 0), length, 12.0, ink, a, radius=3,
                      lift=0.3)
            self.label(sc, self.shift(at, BAR_LEN + 8, 0), rt[value], SIZE, anchor="lm",
                       opacity=a * smoothstep(0.6, 0.9, g), ink=INK)
        f = op * (self.at(5, 0.4) if k == 5 else 1.0)
        self.text(sc, TEXT_X, FIX_Y[0], "flipped: MLP %s, Deep Sets %d"
                  % (m["changed_text"], m["changed_ds"]), f, SIZE, "lm", strong=True)

    # ------------------------------------------------------------------ 1: the jets
    def phase_1(self, sc, u, op):
        """The two jets, in ``steady``."""

    # ------------------------------------------------------------------ 2: the list
    def phase_2(self, sc, u, op):
        """The list fills in, in ``steady``."""

    # ------------------------------------------------------------------ 3: the MLP
    def strip_reveal(self):
        """Phase 3: how much of the MLP's column is filled in, row by row."""
        return self.ease_in(2, *self.STRIP_U)

    def strip_ring(self):
        """Phase 3: the real row being copied into the column now (None between)."""
        shown = J_MAX * self.strip_reveal()
        k, _u = self.where(getattr(self, "_t", 0.0))
        if k != 2 or not 0.0 < shown < self.n:
            return None
        return int(shown)

    def phase_3(self, sc, u, op):
        self.strip(sc, on(u, 0.04, op), reveal=self.strip_reveal(), ring=self.strip_ring())
        a = on(u, 0.42, op)
        cx, cy = CARD
        if a > 0.01:
            self.rect(sc, self.P(cx, cy), CARD_W, CARD_H, soft(NET, 0.8), a, border=NET,
                      border_w=2.0, radius=7, lift=0.2)
            self.tiles(sc, self.card, cx, cy, CARD_W - 14, CARD_H - 14, a, ink=NET,
                       lo=-float(np.abs(self.card).max()), hi=float(np.abs(self.card).max()),
                       grid=0.15, lift=0.25)
            hy = self.frac(0, CARD_H / 2)[1]
            self.text(sc, cx, cy - hy - 0.025, "MLP", a, SIZE, strong=True)
            self.text(sc, cx, cy + hy + 0.03, self.meta["count_text"]["mlp"] + " numbers", a,
                      SIZE, ink=NET_INK)
            sx = STRIP_X + self.frac(STRIP_W / 2 + 4)[0]
            self.arrow(sc, (sx, cy), (cx - self.frac(CARD_W / 2 + 4)[0], cy), AXIS, a,
                       width=1.6)
        b = on(u, 0.55, op)
        self.arrow(sc, (cx + self.frac(CARD_W / 2 + 4)[0], cy - 0.06),
                   (SB_X - 0.006, SB_Y["mlp"] + 0.045), AXIS, b, width=1.6)

    # ------------------------------------------------------------------ 4: Deep Sets
    def phase_4(self, sc, u, op):
        m = self.meta
        a = on(u, 0.04, op)
        # phi runs down all 20 rows, one at a time (a pad's row of zeros gives phi(0), not 0);
        # then the mask sets the pads' rows to 0 (striped)
        done = J_MAX * eased(u, *self.PHI_U)
        zeroed = on(u, self.MASK_U, op, 0.06)
        for j in range(J_MAX):
            g = float(np.clip(done - j, 0.0, 1.0))
            pad = j >= self.n
            if g <= 0.01 or (pad and zeroed >= op - 1e-6):
                continue
            rgb = shades((self.phi_pad if pad else self.phi[j])[None], DATA, 0.0, self.phi_hi)
            self.image(sc, rgb, PHI_X, self.row_y(j), PHI_W, ROW_H,
                       a * g * ((1.0 - zeroed / max(op, 1e-6)) if pad else 1.0), lift=0.32,
                       grid=0.0)
        self.pads(sc, PHI_X, self.n, PHI_W, 32, zeroed)
        mid = TOP + J_MAX * ROW_H / H_PX / 2
        self.rect(sc, self.P(PHI_X, mid), PHI_W + 4, J_MAX * ROW_H + 4, DATA_INK, a,
                  border=DATA_INK, border_w=1.2, radius=2, lift=0.25, fill_op=0.0)
        self.text(sc, PHI_X, HEADS_Y, "φ on each row", a, SIZE, ink=NET_INK)
        # the φ box, travelling down beside the row it is working on
        if 0.0 < done < J_MAX:
            y = self.row_y(float(np.clip(done - 0.5, 0.0, J_MAX - 1)))
            self.box(sc, PHI_X - self.frac(PHI_W / 2 + 15)[0], y, 20.0, 16.0, "φ", a)
        # the sum, then rho
        s = on(u, 0.58, op)
        self.arrow(sc, (PHI_X, self.bottom() + 0.008), (PHI_X, SUM_Y - 0.022), AXIS, s,
                   width=1.6)
        self.sum_strip(sc, self.sum, s)
        r = on(u, 0.65, op)
        self.arrow(sc, (PHI_X + self.frac(PHI_W / 2 + 4)[0], SUM_Y),
                   (RHO[0] - self.frac(19.0)[0], RHO[1]), AXIS, r, width=1.6)
        self.box(sc, RHO[0], RHO[1], 30.0, 24.0, "ρ", r)
        self.text(sc, RHO[0], RHO[1] + 0.042, m["count_text"]["ds"] + " numbers", r, SIZE,
                  ink=NET_INK)
        self.arrow(sc, (RHO[0] + self.frac(10.0)[0], RHO[1] - 0.027),
                   (SB_X - 0.006, SB_Y["ds"] + 0.05), AXIS, on(u, 0.7, op), width=1.6)

    # ------------------------------------------------------------------ 5: shuffled
    def phase_5(self, sc, u, op):
        self.strip(sc, on(u, 0.02, op))
        self.sum_strip(sc, self.sum_s if u >= self.SHUFFLE_U[1] else self.sum, on(u, 0.02, op))
        self.text(sc, PHI_X + self.frac(PHI_W / 2 + 8)[0], SUM_Y, "the same",
                  on(u, self.SWAP_U, op), SIZE, "lm", strong=True)

    # ------------------------------------------------------------------ 6: 4,000 jets
    def phase_6(self, sc, u, op):
        """The bars, in ``steady``."""

    # ------------------------------------------------------------------ 7: two fixes
    def phase_7(self, sc, u, op):
        m = self.meta
        self.text(sc, TEXT_X, FIX_Y[1], "1  sort first: " + m["right_text"]["mlp_again"],
                  on(u, self.FIX_U[0], op), SIZE, "lm", strong=True)
        self.text(sc, TEXT_X, FIX_Y[2], "2  build the symmetry in", on(u, self.FIX_U[1], op),
                  SIZE, "lm", strong=True)
        self.text(sc, TEXT_X, FIX_Y[3], "GNNs: messages between particles",
                  on(u, self.FIX_U[2], op), SIZE, "lm", ink=QUIET)


ANIMATIONS = {"jets-as-sets": JetsAsSets}

"""Animations 203-205 of the "Blocks and code" series (see :mod:`voxanim.blocks`):
``optimizer.zero_grad()``, ``optimizer.step()`` and the learning rate, drawn from
:mod:`voxanim.study_blocks_g`.

All three show the same block: the last layer's weight, ``model[5].weight`` (10 x 32
orange tiles), with its gradient as red tiles just behind it, and the same three tiles read
out number by number under it.

* :class:`ZeroGrad` (``blocks-zero-grad``): the red tiles the first batch's
  ``loss.backward()`` left; ``zero_grad()`` turns them grey (0), column by column; the
  second batch fills them again. Then the real run that forgets it: ``.grad`` holds both
  batches, then 100, and its size climbs to 37.6 while each batch's own stays below 4; its
  loss climbs back up, and it gets 10% of the test digits right against 95%.
* :class:`OptimizerStep` (``blocks-optimizer-step``): the optimizer as a green hand and
  ``lr`` as a dial; three tiles stepped one at a time, ``w - 0.1 x red = new``, with the
  real numbers; the hand sweeping every tile (320 here, 52,650 in the model); the same
  batch's loss before and after on two meters.
* :class:`LearningRate` (``blocks-learning-rate``): the dial turned to 0.001, 0.1 and 0.8
  from the same start; each run's loss curve drawn in its own row with its held-out
  accuracy (55%, 95%, 27%), then the three side by side.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). What stays across phases is drawn in
``steady``, so nothing blanks between them.
"""

from __future__ import annotations

import numpy as np

from .blocks import (
    BIG,
    GRAD,
    HEAD,
    INK,
    LOSS,
    OPTIM,
    RATE,
    SIZE,
    WEIGHT,
    ZERO,
    Blocks,
    Code,
    commas,
    exact,
    measured,
    val,
)
from .easing import ease_in_out, smoothstep
from .losschart import LossChart
from .mech import AXIS, dark, soft

GRID = (0.215, 0.49)            # the centre of model[5].weight's tiles
TILE = 9.0                      # px
ROWS, COLS = 10, 32
BEHIND = (10.0, 10.0)           # how far the red tiles sit behind the orange (px)
W_LIM = 1.0 / np.sqrt(COLS)     # the weights' starting range: +-1/sqrt(32)
BOX_W, BOX_H = 64.0, 28.0
WEIGHT_INK, GRAD_INK, LOSS_INK, RATE_INK = (dark(WEIGHT, 0.3), dark(GRAD, 0.25),
                                            dark(LOSS, 0.25), dark(RATE, 0.4))
LOW_Y = 0.86                    # the bottom row of numbers
STEP_LINES = (0, 1, 4, 5, 6, 7, 8, 9)       # ("optimizer", "step"): SGD(...), lr=0.1, the step


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def four(v):
    return val(v, 4)


class Layer(Blocks):
    """What the three share: ``model[5].weight`` with its red tiles behind, and the tiles
    read out under it."""

    BEHIND = BEHIND

    def grid_cells(self):
        return self.cells(*GRID, COLS * TILE, ROWS * TILE, ROWS, COLS)

    def layer(self, sc, W, op, grad=None, grad_op=1.0, zero=None, glim=1.0, label=True):
        """The weight block (orange, on top) and its gradient (red, behind; ``zero``: the
        tiles set to 0, grey)."""
        fx, fy = GRID
        w, h = COLS * TILE, ROWS * TILE
        if grad is not None and grad_op * op > 0.01:
            # shaded by size (deeper: a bigger gradient, either way), never paler than a
            # light red, so a red tile always reads as red and a 0 as grey
            dx, dy = self.frac(*self.BEHIND)
            self.tiles(sc, np.abs(grad), fx + dx, fy + dy, w, h, op * grad_op, ink=GRAD,
                       lo=-0.45 * glim, hi=glim, zero=zero, lift=0.15)
        self.tiles(sc, W, fx, fy, w, h, op, ink=WEIGHT, lo=-W_LIM, hi=W_LIM, lift=0.3)
        if label:
            x0 = fx - self.frac(w / 2)[0]
            self.text(sc, x0, 0.383, "model[5].weight: [10, 32]", op, SIZE, "lm",
                      ink=WEIGHT_INK)
            if grad is not None:
                x1 = fx + self.frac(w / 2 + self.BEHIND[0])[0]
                y1 = fy + self.frac(0, h / 2 + self.BEHIND[1] + 13)[1]
                self.text(sc, x1, y1, ".grad", op * grad_op, SIZE, "rm", ink=GRAD_INK)

    def rate_dial(self, sc, fx, fy, frac, op, r, label=None):
        """The learning rate's dial (blocks.Blocks.dial)."""
        self.dial(sc, fx, fy, frac, op, r=r, label=label)

    def ring(self, sc, rc, op):
        if op > 0.01:
            self.outline(sc, self.grid_cells(), rc[0], rc[1], 1, 1, TILE, op, lift=1.0)

    def readout(self, sc, rc, fy, text, op, ink=GRAD, fill=None):
        """A box under tile ``rc`` holding ``text``, joined to the tile by a thin line."""
        if op <= 0.01:
            return
        tx, ty = self.grid_cells()(*rc)
        top = fy - self.frac(0, BOX_H / 2)[1]
        self.arrow(sc, (tx, ty + self.frac(0, TILE / 2 + 2)[1]), (tx, top), AXIS, op * 0.8,
                   head=False, width=1.2)
        self.box(sc, tx, fy, BOX_W, BOX_H, soft(ink, 0.8) if fill is None else fill, op,
                 text=text, border=ink, size=SIZE, radius=4)


# ====================================================================== 203
class ZeroGrad(Layer):
    """The red tiles; zero_grad empties them; the next batch fills them; forgotten: they
    add up, pile up, and the model learns nothing."""

    TITLE = "optimizer.zero_grad()"
    PHASES = (("red tiles", 10.0), ("zero_grad", 10.0), ("the next batch", 10.0),
              ("forgotten", 10.0), ("it piles up", 10.0), ("the loss", 10.0))
    LAST = "loss.backward() makes red tiles"
    NEXT = "optimizer.step() moves the tiles"
    STATION = ("backward",) * 6
    BEHIND = (15.0, 15.0)           # the red tiles are the subject here: more of them shows
    BOX_Y = 0.70
    BAR_X, BAR_PX = 0.2, 110.0          # the size's bar: where it starts, px per unit
    CHART = ((0.11, 0.45), (0.415, 0.655))
    ZERO_AT = (0.2, 0.45)               # phase 2: the sweep that turns the red tiles grey
    FILL_AT = (0.15, 0.4)               # phase 3: the sweep that fills them again
    ADD_AT = (0.25, 0.45)               # phase 4: the second batch's red added to the first's
    DRAW_AT = (0.08, 0.85)              # phase 5: the curves drawn

    def setup(self):
        m = self.meta = self.study.metadata
        self.W = np.asarray(m["weight"], float)
        self.G0 = np.asarray(m["grad"], float)
        self.G1 = np.asarray(m["grad_next"], float)
        self.pile = np.asarray(m["grad_pile"], float)
        self.glim = float(max(np.abs(a).max() for a in (self.G0, self.G1, self.pile)))
        self.tiles_at = [tuple(t) for t in m["tiles"]]
        self.fresh = np.asarray(m["fresh_all"], float)
        self.used = np.asarray(m["used_all"], float)
        self.steps = np.arange(1, len(self.used) + 1, dtype=float)
        self.CODE = (Code("step", run=((0.0, 4), (0.75, 5))),
                     Code("step", run=((0.0, 0), (0.1, 3))),
                     Code("step", run=((0.0, 4),)),
                     Code("mistake-no-zero-grad", run=((0.0, 0), (0.12, 2))),
                     Code("mistake-no-zero-grad", run=((0.0, 2),)),
                     Code("mistake-no-zero-grad", run=((0.0, 3),)))
        self.EVIDENCE = (measured("the first batch's gradients"),
                         exact("zero_grad() sets every one to 0"),
                         measured("the second batch's gradients"),
                         exact("the two batches' gradients, added"),
                         measured("the run without zero_grad"),
                         measured("both runs, 10,000 test digits"))

    def heading_for(self, k, u):
        return ("backward() left red tiles", "zero_grad(): every red tile to 0",
                "the next backward() fills them", "forgot zero_grad(): they add up",
                "the red tiles pile up", "the model learns nothing")[k]

    def subtitle_for(self, k):
        m = self.meta
        at100 = m["batches"].index(100)
        return [
            "loss.backward() left a red gradient behind every weight. Their total size: %s."
            % val(m["length_first"]),
            "optimizer.zero_grad() sets all %s red tiles to 0 before the next backward."
            % commas(m["numbers"]),
            "The second batch's backward fills them with its own gradients: total size %s."
            % val(m["length_next"]),
            "Without zero_grad, backward adds to the old tiles: .grad holds both batches, "
            "size %s." % val(m["length_pile"]),
            "Forgotten, the pile keeps growing: after 100 batches .grad has size %s, this "
            "batch only %s." % (val(m["used"][at100]), val(m["fresh"][at100])),
            "Its loss climbs back up, and it gets %s of the %s test digits right, against %s."
            % (m["percent_bad"], commas(m["test"]), m["percent_sgd"]),
        ][k]

    # ------------------------------------------------------------------ the state
    def red(self, k, u):
        """The red tiles at ``(k, u)``: ``(values, zero mask)``."""
        zero = np.zeros((ROWS, COLS), bool)
        if k == 0:
            return self.G0, zero
        if k == 1:
            zero[:, :int(round(COLS * eased(u, *self.ZERO_AT)))] = True
            return self.G0, zero
        if k == 2:
            zero[:, int(round(COLS * eased(u, *self.FILL_AT))):] = True
            return self.G1, zero
        s = eased(u, *self.ADD_AT) if k == 3 else 1.0
        return self.G1 + s * self.G0, zero

    def tile_text(self, k, u, i):
        """Tile ``i``'s red number at ``(k, u)`` (its box never blanks)."""
        m = self.meta
        col = self.tiles_at[i][1]
        if k == 0:
            return four(m["g0"][i])
        if k == 1:
            return "0" if COLS * eased(u, *self.ZERO_AT) > col + 0.5 else four(m["g0"][i])
        if k == 2:
            return four(m["g1"][i]) if COLS * eased(u, *self.FILL_AT) > col + 0.5 else "0"
        if k == 3:
            return four(m["pile"][i]) if u >= self.ADD_AT[1] else four(m["g1"][i])
        return four(m["pile"][i])

    def size_now(self, k, u):
        """The size of .grad shown at ``(k, u)``: only real values, swapped when a sweep
        ends."""
        m = self.meta
        if k == 0:
            return m["length_first"]
        if k == 1:
            return 0.0 if u >= self.ZERO_AT[1] else m["length_first"]
        if k == 2:
            return m["length_next"] if u >= self.FILL_AT[1] else 0.0
        if k == 3:
            return m["length_pile"] if u >= self.ADD_AT[1] else m["length_next"]
        if k == 4:
            return float(self.used[self.shown_step(u) - 1])
        return float(self.used[-1])

    def drawn_to(self, u):
        """Phase 5: how many batches the curves have reached (the first 100 take most of
        the time: that is where the pile grows)."""
        s = eased(u, *self.DRAW_AT)
        return 2.0 + (len(self.used) - 2) * s ** 2.6

    def shown_step(self, u):
        """Phase 5: the batch count read out, the last of the study's read-outs reached."""
        n = self.drawn_to(u)
        return max(b for b in self.meta["batches"] if b <= n)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        grid = op * self.kept(0, 3)
        if grid > 0.01:
            values, zero = self.red(k, u)
            self.layer(sc, self.W, grid, values, zero=zero, glim=self.glim)
            on = grid * (smoothstep(0.2, 0.27, u) if k == 0 else 1.0)
            for i, rc in enumerate(self.tiles_at):
                self.ring(sc, rc, on)
                txt = self.tile_text(k, u, i)
                fill = soft(ZERO, 0.5) if txt == "0" else None
                self.readout(sc, rc, self.BOX_Y, txt, on, fill=fill)
            bar = grid * (smoothstep(0.45, 0.52, u) if k == 0 else 1.0)
            size = self.size_now(k, u)
            px = max(0.1, size * self.BAR_PX)
            at = self.P(self.BAR_X, LOW_Y)
            self.rect(sc, self.shift(at, px / 2, 0), px, 18.0, GRAD, bar, radius=4)
        num = op * self.kept(0, 4) * (smoothstep(0.45, 0.52, u) if k == 0 else 1.0)
        self.big(sc, 0.065, LOW_Y, val(self.size_now(k, u)), num, label="size of .grad",
                 ink=GRAD_INK)
        chart = op * self.kept(4, 5)
        if chart > 0.01:
            self.axes(sc, self.chart_for(4), chart, xlabel="batches",
                      xticks=[(0, "0"), (500, "500"), (938, "938")], yticks=[], grid=False)

    def chart_for(self, k):
        top = 40.0 if k == 4 else 3.0
        return LossChart(self.CHART, (0.0, float(len(self.used))), (0.0, top))

    # ------------------------------------------------------------------ 1: red tiles
    def phase_1(self, sc, u, op):
        pass

    # ------------------------------------------------------------------ 2: zero_grad
    def phase_2(self, sc, u, op):
        on = op * smoothstep(self.ZERO_AT[1] + 0.05, self.ZERO_AT[1] + 0.12, u)
        self.text(sc, self.BAR_X + 0.02, LOW_Y, "all %s set to 0" % commas(self.meta["numbers"]),
                  on, SIZE, "lm")

    # ------------------------------------------------------------------ 3: the next batch
    def phase_3(self, sc, u, op):
        pass

    # ------------------------------------------------------------------ 4: forgotten
    def phase_4(self, sc, u, op):
        on = op * smoothstep(self.ADD_AT[0], self.ADD_AT[0] + 0.07, u)
        x = self.grid_cells()(0, self.tiles_at[-1][1])[0] + self.frac(BOX_W / 2 + 8)[0]
        self.text(sc, x, self.BOX_Y, "batch 1 + batch 2", on, SIZE, "lm", ink=GRAD_INK)

    # ------------------------------------------------------------------ 5: it piles up
    def phase_5(self, sc, u, op):
        ch = self.chart_for(4)
        self.axes(sc, ch, op, ylabel="size", xticks=[], grid=False,
                  yticks=[(0, "0"), (20, "20"), (40, "40")])
        upto = (self.drawn_to(u) - 1) / (len(self.used) - 1)
        on = op * smoothstep(0.02, 0.08, u)
        self.plot(sc, ch, self.steps, self.fresh, soft(GRAD, 0.35), on, upto, width=1.4,
                  dot=False)
        self.plot(sc, ch, self.steps, self.used, GRAD, on, upto)
        late = op * smoothstep(self.DRAW_AT[1], self.DRAW_AT[1] + 0.05, u)
        x1 = self.CHART[0][1]
        self.text(sc, x1, ch.fy(self.used[-1]) - 0.035, "what .grad holds", late, SIZE, "rm",
                  ink=GRAD_INK)
        self.text(sc, x1, ch.fy(0) - 0.035, "this batch's own", late, SIZE, "rm",
                  ink=GRAD_INK)
        self.big(sc, 0.27, LOW_Y, commas(self.shown_step(u)), on, label="batches")

    # ------------------------------------------------------------------ 6: the loss
    def phase_6(self, sc, u, op):
        m = self.meta
        ch = self.chart_for(5)
        self.axes(sc, ch, op, ylabel="loss (mean of 25 steps)", xticks=[],
                  yticks=[(0, "0"), (1, "1"), (2, "2"), (3, "3")], grid=False)
        x = np.asarray(m["chunk_x"], float)
        draw = eased(u, 0.06, 0.5)
        on = op * smoothstep(0.02, 0.08, u)
        self.plot(sc, ch, x, m["loss_sgd"], AXIS, on, draw, width=1.4)
        self.plot(sc, ch, x, m["loss_bad"], LOSS, on, draw)
        late = op * smoothstep(0.5, 0.56, u)
        x1 = self.CHART[0][1]
        self.text(sc, x1, ch.fy(min(m["loss_bad"][-12:])) + 0.035, "without zero_grad", late,
                  SIZE, "rm", ink=LOSS_INK)
        self.text(sc, x1, ch.fy(m["loss_sgd"][-1]) - 0.04, "with zero_grad", late, SIZE, "rm")
        self.text(sc, 0.065, 0.755, "test digits right", op * smoothstep(0.6, 0.66, u), SIZE,
                  "lm", strong=True)
        self.big(sc, 0.065, LOW_Y, m["percent_sgd"], op * smoothstep(0.6, 0.66, u),
                 label="with")
        self.big(sc, 0.27, LOW_Y, m["percent_bad"], op * smoothstep(0.72, 0.78, u),
                 label="without", ink=LOSS_INK)


# ====================================================================== 204
class OptimizerStep(Layer):
    """The optimizer and lr; three tiles stepped; every tile at once; the same batch's
    loss before and after."""

    TITLE = "optimizer.step()"
    PHASES = (("the optimizer", 11.0), ("one tile", 16.0), ("every tile", 11.0),
              ("the same batch", 12.0))
    LAST = "zero_grad() empties the red tiles"
    NEXT = "the learning rate: how far to move"
    STATION = ("step", "step", "step", "loss")
    HAND = (0.425, 0.44)
    DIAL = (0.425, 0.575)
    ROW_Y = (0.685, 0.765, 0.845)
    ROW_AT = (0.06, 0.37, 0.66)         # phase 2: when each tile's row starts
    SWEEP = (0.12, 0.6)                 # phase 3: the hand passes over the tiles

    def setup(self):
        m = self.meta = self.study.metadata
        self.W = np.asarray(m["weight"], float)
        self.G = np.asarray(m["grad"], float)
        self.after = np.asarray(m["after"], float)
        self.glim = float(np.abs(self.G).max())
        self.tiles_at = [tuple(t) for t in m["tiles"]]
        both = ("optimizer", "step")
        self.CODE = (Code(both, pick=STEP_LINES, run=((0.0, 0), (0.4, 1))),
                     Code(both, pick=STEP_LINES, run=((0.0, 7),)),
                     Code(both, pick=STEP_LINES, run=((0.0, 7),)),
                     Code(both, pick=STEP_LINES, run=((0.0, 3), (0.2, 4))))
        self.EVIDENCE = (measured("the first batch's gradients"),
                         exact("w − 0.1 × gradient, tile by tile"),
                         exact("one rule for all 52,650 numbers"),
                         measured("the first batch, before and after"))

    def heading_for(self, k, u):
        return ("the optimizer: a hand and a dial", "each tile: minus 0.1 × its red",
                "every tile moves at once", "the same batch: a lower loss")[k]

    def subtitle_for(self, k):
        m = self.meta
        i = int(np.argmax(np.abs(m["g"])))
        return [
            "SGD holds every weight and its red gradient. Its learning rate, lr=%g, says how "
            "far to move." % m["lr"],
            "Each tile moves against its red tile, by 0.1 times it: %s − 0.1 × (%s) = %s."
            % (four(m["before_w"][i]), four(m["g"][i]), four(m["after_w"][i])),
            "The same rule moves every number at once: %d in this block, %s in the whole model."
            % (m["layer_numbers"], commas(m["numbers"])),
            "The same %d digits again: the loss was %s and is now %s, lower by %s."
            % (m["batch"], four(m["loss_before"]), four(m["loss_after"]), four(m["drop"])),
        ][k]

    # ------------------------------------------------------------------ the state
    def swept(self, k, u):
        """How many columns the hand has passed (their tiles hold the new numbers)."""
        if k < 2:
            return 0
        if k > 2:
            return COLS
        return int(round(COLS * eased(u, *self.SWEEP)))

    def weights_now(self, k, u):
        W = self.W.copy()
        n = self.swept(k, u)
        W[:, :n] = self.after[:, :n]
        if k == 1:                       # each read-out tile takes its new number
            for i, (r, c) in enumerate(self.tiles_at):
                if u >= self.ROW_AT[i] + 0.14:
                    W[r, c] = self.after[r, c]
        return W

    def hand_at(self, k, u):
        if k != 2:
            return self.HAND
        x0 = GRID[0] - self.frac(COLS * TILE / 2)[0]
        x1 = GRID[0] + self.frac(COLS * TILE / 2)[0]
        y = GRID[1] - self.frac(0, ROWS * TILE / 2 + 16)[1]
        a, b = self.SWEEP
        go = eased(u, a - 0.08, a)
        back = eased(u, b, b + 0.1)
        sx = x0 + (x1 - x0) * eased(u, a, b)
        fx = self.HAND[0] + (x0 - self.HAND[0]) * go
        fy = self.HAND[1] + (y - self.HAND[1]) * go
        if u >= a:
            fx, fy = sx, y
        if u >= b:
            fx = x1 + (self.HAND[0] - x1) * back
            fy = y + (self.HAND[1] - y) * back
        return fx, fy

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        self.layer(sc, self.weights_now(k, u), op, self.G, glim=self.glim)
        on = op * (smoothstep(0.15, 0.22, u) if k == 0 else 1.0)
        hx, hy = self.hand_at(k, u)
        self.hand(sc, hx, hy, on)
        self.text(sc, self.HAND[0], self.HAND[1] - 0.055, "SGD", on * (1.0 if k != 2 else
                  1.0 - smoothstep(0.0, 0.05, u) + smoothstep(0.75, 0.8, u)), SIZE,
                  ink=dark(OPTIM, 0.35))
        d = op * (smoothstep(0.4, 0.47, u) if k == 0 else 1.0)
        self.rate_dial(sc, *self.DIAL, 0.75, d, r=22.0, label="lr=%g" % self.meta["lr"])
        n = self.swept(k, u)
        if n > 0 and k == 2:                 # the tiles the hand has moved, outlined green
            x0 = GRID[0] - self.frac(COLS * TILE / 2)[0]
            w = n * TILE
            self.rect(sc, self.P(x0 + self.frac(w / 2)[0], GRID[1]), w + 4, ROWS * TILE + 4,
                      OPTIM, op * (1.0 - smoothstep(0.85, 0.95, u)), border=OPTIM,
                      border_w=2.4, radius=3, lift=1.0, fill_op=0.0)

    # ------------------------------------------------------------------ 1: the optimizer
    def phase_1(self, sc, u, op):
        self.big(sc, 0.065, LOW_Y, commas(self.meta["numbers"]), op * smoothstep(0.6, 0.66, u),
                 label="numbers SGD will move")

    # ------------------------------------------------------------------ 2: one tile
    def row(self, sc, i, fy, u, op):
        m = self.meta
        a = self.ROW_AT[i]
        w_on = op * smoothstep(a, a + 0.03, u)
        g_on = op * smoothstep(a + 0.06, a + 0.09, u)
        n_on = op * smoothstep(a + 0.14, a + 0.17, u)
        px = self.frac(1.0)[0]
        x0 = GRID[0] - self.frac(COLS * TILE / 2)[0]
        self.box(sc, x0 + 32 * px, fy, BOX_W, BOX_H, soft(WEIGHT, 0.8), w_on,
                 text=four(m["before_w"][i]), border=WEIGHT, size=SIZE, radius=4)
        self.text(sc, x0 + 100 * px, fy, "− 0.1 ×", g_on, SIZE)
        self.box(sc, x0 + 168 * px, fy, BOX_W, BOX_H, soft(GRAD, 0.8), g_on,
                 text=four(m["g"][i]), border=GRAD, size=SIZE, radius=4)
        self.text(sc, x0 + 214 * px, fy, "=", n_on, SIZE)
        self.box(sc, x0 + 260 * px, fy, BOX_W, BOX_H, soft(WEIGHT, 0.55), n_on,
                 text=four(m["after_w"][i]), border=dark(WEIGHT, 0.2), size=SIZE, radius=4)

    def phase_2(self, sc, u, op):
        for i, rc in enumerate(self.tiles_at):
            a = self.ROW_AT[i]
            if u < a:
                continue
            now = i == len(self.ROW_AT) - 1 or u < self.ROW_AT[i + 1]
            self.ring(sc, rc, op * smoothstep(a, a + 0.03, u) * (1.0 if now else 0.4))
            self.row(sc, i, self.ROW_Y[i], u, op)

    # ------------------------------------------------------------------ 3: every tile
    def phase_3(self, sc, u, op):
        m = self.meta
        n = self.swept(2, u)
        self.big(sc, 0.065, LOW_Y, commas(n * ROWS), op * smoothstep(0.04, 0.1, u),
                 label="moved in this block")
        self.big(sc, 0.27, LOW_Y, commas(m["numbers"]), op * smoothstep(0.7, 0.76, u),
                 label="in the whole model")

    # ------------------------------------------------------------------ 4: the same batch
    def phase_4(self, sc, u, op):
        m = self.meta
        top = 2.5
        b = op * smoothstep(0.08, 0.14, u)
        a = op * smoothstep(0.36, 0.42, u)
        for x, name, loss, on in ((0.1, "before", m["loss_before"], b),
                                  (0.21, "after", m["loss_after"], a)):
            self.text(sc, x, 0.64, name, on, SIZE)
            self.meter(sc, x, 0.755, loss / top, on, h=104.0, label=four(loss))
        self.big(sc, 0.27, 0.78, four(m["drop"]), op * smoothstep(0.6, 0.66, u),
                 label="lower by", ink=LOSS_INK)


# ====================================================================== 205
class LearningRate(Layer):
    """The dial; three runs from one start, one row each; the three side by side."""

    TITLE = "The learning rate"
    PHASES = (("same start", 11.0), ("lr=0.001", 12.0), ("lr=0.1", 12.0), ("lr=0.8", 12.0),
              ("side by side", 9.0))
    LAST = "optimizer.step(): w − 0.1 × red"
    NEXT = "Adam keeps two more blocks"
    STATION = ("step",) * 5
    DIAL = (0.1, 0.455)
    ROW_Y = (0.605, 0.715, 0.825)
    CHART_X = (0.17, 0.365)
    HALF = 0.04                       # a row's chart: half its height
    TOP = 4.0                         # its loss axis: 0 to 4
    WORDS = ("crawls", "learns", "bounces")
    TURN = (0.0, 0.12)                # phases 2-4: the needle turns
    DRAW = (0.2, 0.75)                # phases 2-4: the curve draws
    SHOW = (0.1, 0.4, 0.7)            # phase 5: the needle visits each rate

    def setup(self):
        m = self.meta = self.study.metadata
        self.x = np.r_[0.0, np.asarray(m["chunk_x"], float)]
        self.curves = [np.r_[m["first_loss"], np.asarray(c, float)] for c in m["curves"]]
        self.rates = list(m["rates"])
        both = ("optimizer", "step")
        run = ((0.0, 1), (0.25, 7))
        self.CODE = (Code(both, pick=STEP_LINES, run=((0.0, 0), (0.3, 1))),
                     Code(both, pick=STEP_LINES, run=run),
                     Code(both, pick=STEP_LINES, run=run),
                     Code(both, pick=STEP_LINES, run=run),
                     Code(both, pick=STEP_LINES, run=((0.0, 1),)))
        self.EVIDENCE = (measured("one start, the first batch's loss"),) + tuple(
            measured("lr=%s, 10,000 test digits" % t) for t in m["rate_text"]) + (
            measured("three runs, 10,000 test digits"),)

    def heading_for(self, k, u):
        return ("lr: how far each step moves", "lr=0.001: tiny steps, it crawls",
                "lr=0.1: the script's, it learns", "lr=0.8: too far, it bounces",
                "one number, three endings")[k]

    def subtitle_for(self, k):
        m = self.meta
        p = m["percent"]
        return [
            "The script sets lr=0.1. We train three times from the same start, first loss %s, "
            "changing only lr." % val(m["first_loss"]),
            "lr=0.001: each step moves a tile %s. After 938 steps the loss is still %s: %s "
            "right." % (m["moves_text"][0], val(m["last"][0]), p[0]),
            "lr=0.1, the script's: the loss falls to %s and it gets %s of the %s test digits "
            "right." % (val(m["last"][1]), p[1], commas(m["test"])),
            "lr=0.8: each step jumps too far. The loss bounces (one batch: %s) and only %s are "
            "right." % (val(m["highest"][2]), p[2]),
            "The same model, data and start: only the learning rate differs, and it decides "
            "%s, %s or %s." % tuple(p),
        ][k]

    # ------------------------------------------------------------------ the dial
    @staticmethod
    def frac_of(lr):
        """Where the needle points for ``lr``: a log scale from 0.0001 to 1."""
        return float(np.clip((np.log10(lr) + 4.0) / 4.0, 0.0, 1.0))

    def rate_at(self, k, u):
        """``(needle frac, index of the rate it reads)`` at ``(k, u)``."""
        if k == 0:
            return self.frac_of(self.rates[1]), 1
        # the reading changes when the needle is half way round
        if k <= 3:
            i, j = k - 1, (1 if k == 1 else k - 2)
            s = eased(u, *self.TURN)
        else:
            n = sum(u >= a for a in self.SHOW) - 1
            if n < 0:
                return self.frac_of(self.rates[2]), 2
            i, j = n, (n - 1 if n > 0 else 2)
            s = eased(u, self.SHOW[n], self.SHOW[n] + 0.1)
        before = self.frac_of(self.rates[j])
        return before + (self.frac_of(self.rates[i]) - before) * s, (i if s >= 0.5 else j)

    def chart(self, i):
        y = self.ROW_Y[i]
        return LossChart((self.CHART_X, (y - self.HALF, y + self.HALF)),
                         (0.0, float(self.x[-1])), (0.0, self.TOP))

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        frac, i = self.rate_at(k, u)
        self.rate_dial(sc, *self.DIAL, frac, op, r=30.0)
        self.running(sc, 0.17, 0.44, [(0.0, "lr=%s" % m["rate_text"][i])], 0.0, op,
                     size=HEAD, ink=RATE_INK)
        mv = op * (smoothstep(0.3, 0.36, u) if k == 0 else 1.0)
        self.text(sc, 0.17, 0.49, "one step moves a tile %s" % m["moves_text"][i], mv, SIZE,
                  "lm")
        frame = op * (smoothstep(0.45, 0.52, u) if k == 0 else 1.0)
        self.text(sc, self.CHART_X[0], 0.545, "loss", frame, SIZE, "lm")
        self.text(sc, 0.39, 0.545, "correct", frame, SIZE, "lm")
        for r in range(3):
            self.row(sc, r, k, u, op, frame)

    def row(self, sc, r, k, u, op, frame):
        """Row ``r``: its rate, its chart, its curve once drawn, its accuracy."""
        m = self.meta
        y = self.ROW_Y[r]
        ch = self.chart(r)
        lit = self.lit(r, k, u)
        self.text(sc, 0.06, y - 0.014, m["rate_text"][r], frame, SIZE, "lm", strong=True)
        self.axes(sc, ch, frame, xticks=[(0, "0"), (938, "938 steps")] if r == 2 else [],
                  yticks=[(0, "0"), (2, "2")], grid=False)
        ch.hline(sc, self.P, m["first_loss"], frame * 0.8)
        self.disk(sc, self.P(ch.fx(0.0), ch.fy(m["first_loss"])), 4.0, LOSS, frame, lift=1.2,
                  outline=0.0)
        drawn = 1.0 if k > r + 1 else (eased(u, *self.DRAW) if k == r + 1 else 0.0)
        if drawn > 0:
            self.plot(sc, ch, self.x, self.curves[r], LOSS, op * lit, drawn,
                      dot=k == r + 1)
        done = k > r + 1 or (k == r + 1 and u >= self.DRAW[1] + 0.05)
        on = op * lit * (smoothstep(self.DRAW[1] + 0.05, self.DRAW[1] + 0.11, u)
                         if k == r + 1 else float(done))
        self.text(sc, 0.06, y + 0.022, self.WORDS[r], on, SIZE, "lm", ink=RATE_INK)
        self.text(sc, 0.39, y, m["percent"][r], on, BIG, "lm", ink=INK)

    def lit(self, r, k, u):
        """How strongly row ``r`` is drawn: the rate under way at full strength, the
        others dimmed."""
        _f, i = self.rate_at(k, u)
        if k == 0:
            return 1.0
        return 1.0 if r == i else (0.7 if k == 4 else 0.45)

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        on = op * smoothstep(0.6, 0.66, u)
        ch = self.chart(0)
        self.text(sc, ch.fx(0) + 0.012, ch.fy(self.meta["first_loss"]) - 0.025,
                  "start: %s" % val(self.meta["first_loss"]), on, SIZE, "lm", ink=LOSS_INK)

    def phase_2(self, sc, u, op):
        pass

    def phase_3(self, sc, u, op):
        pass

    def phase_4(self, sc, u, op):
        pass

    def phase_5(self, sc, u, op):
        pass


ANIMATIONS = {"blocks-zero-grad": ZeroGrad, "blocks-optimizer-step": OptimizerStep,
              "blocks-learning-rate": LearningRate}

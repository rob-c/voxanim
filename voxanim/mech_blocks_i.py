"""Animations 209-211 of the "Blocks and code" series (see :mod:`voxanim.blocks`): testing
the trained model, saving and loading it, and one whole training step in slow motion, drawn
from :mod:`voxanim.study_blocks_i`.

* :class:`Evaluation` (``blocks-evaluation``): ``model.eval()`` as a switch from train to
  eval; ``with torch.no_grad():`` as the three weights with no red tiles behind them;
  the first ten held-out digits counted one by one into ``correct``; then the ten batches
  of 1,000 of ``test_loader``, ``correct`` growing to 9,494, and the accuracy, 0.9494.
* :class:`SaveAndLoad` (``blocks-save-and-load``): ``model.state_dict()`` as six labelled
  boxes of tiles, each with its key and shape; their numbers counted (52,650); packed into
  the file ``mnist.pt`` and put back by ``load_state_dict``; the same answers on 10,000 of
  10,000 held-out digits.
* :class:`OneStep` (``blocks-one-step-slow-motion``): the recorded second step of the run,
  one station of the loop map per phase: the batch, the shapes through the three weights,
  the loss meter, the red tiles (last step's taken away by ``zero_grad``, new ones made
  from the last layer back), and the green hand moving the weights, one of them read out,
  and the loss on the same batch after it.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). Everything that stays across phases is drawn
in ``steady``, so nothing blanks between them.
"""

from __future__ import annotations

import numpy as np

from . import study_course as course
from .blocks import (
    BIG,
    DATA,
    DATA_HI,
    DATA_LO,
    GRAD,
    HEAD,
    INK,
    LOSS,
    MAP,
    OPTIM,
    QUIET,
    SIZE,
    WEIGHT,
    Blocks,
    Code,
    commas,
    exact,
    measured,
    val,
)
from .easing import ease_in_out, smoothstep
from .mech import AXIS, GRID, PAPER, TRACK, dark, soft

LO, HI = DATA_LO, DATA_HI
DATA_INK = dark(DATA, 0.3)
WEIGHT_INK = dark(WEIGHT, 0.35)
GRAD_INK = dark(GRAD, 0.25)
LOSS_INK = dark(LOSS, 0.3)
OPTIM_INK = dark(OPTIM, 0.35)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on_at(u, a, w=0.06):
    """0 before ``a``, fading in over ``w``."""
    return smoothstep(a, a + w, u)


class Shared(Blocks):
    """Drawing helpers the three share."""

    def weight_block(self, sc, values, fx, fy, side, op, lo=None, hi=None, ring=(),
                     lift=0.3):
        """A weight as a square of orange tiles with a thin dark-orange frame."""
        if op <= 0.01:
            return self.cells(fx, fy, side, side, *np.shape(values))
        self.rect(sc, self.P(fx, fy), side + 6, side + 6, soft(WEIGHT, 0.8), op,
                  border=WEIGHT_INK, border_w=1.4, radius=3, lift=lift - 0.01)
        return self.tiles(sc, values, fx, fy, side, side, op, ink=WEIGHT, lo=lo, hi=hi,
                          ring=ring, lift=lift, grid=0.25)

    def tick(self, sc, fx, fy, op, s=1.0, ink=INK):
        """A tick: right."""
        if op <= 0.01:
            return
        c = self.P(fx, fy)
        self.polyline(sc, [self.shift(c, -7 * s, 0), self.shift(c, -2 * s, 6 * s),
                           self.shift(c, 8 * s, -7 * s)], 3.0, ink, op, bias=1.0)

    def cross(self, sc, fx, fy, op, s=1.0, ink=INK):
        """A cross: wrong."""
        if op <= 0.01:
            return
        c = self.P(fx, fy)
        for a, b in (((-6, -6), (6, 6)), ((-6, 6), (6, -6))):
            self.polyline(sc, [self.shift(c, a[0] * s, a[1] * s), self.shift(c, b[0] * s,
                                                                             b[1] * s)],
                          3.0, ink, op, bias=1.0)

    def ring(self, sc, fx, fy, w, h, op, ink=INK, lift=1.0, width=2.4, radius=5):
        if op > 0.01:
            self.rect(sc, self.P(fx, fy), w, h, ink, op, border=ink, border_w=width,
                      radius=radius, lift=lift, fill_op=0.0)


# ====================================================================== 209
class Evaluation(Shared):
    """Test mode; no gradients; ten digits counted one by one; all 10,000."""

    TITLE = "Testing the model"
    PHASES = (("test mode", 12.0), ("no gradients", 13.0), ("one by one", 17.0),
              ("all 10,000", 15.0))
    LAST = "the loss per step and per epoch"
    NEXT = "saving the model to a file"
    STATION = ("forward",) * 4
    COUNT = (0.375, 0.835)          # the running total "correct", from phase 1 on
    BLOCK_X, BLOCK_Y, BLOCK = (0.105, 0.195, 0.285), 0.63, 56.0
    CARD_TILE = 1.45
    COLS = (0.085, 0.145, 0.205, 0.265, 0.325)
    ROWS = (0.465, 0.665)
    ZERO_AT = 0.72                  # phase 1: correct = 0 runs
    BAR_X, BAR_Y, BAR_STEP, BAR_LEN = 0.09, 0.405, 0.036, 150.0

    def setup(self):
        m = self.meta = self.study.metadata
        self.digits = (np.asarray(self.study.panels[0].data["values"], float)
                       - course.MEAN) / course.STD
        P = course.params(course.EPOCH_STEPS)
        from .study_blocks_i import corner
        self.blocks = [corner(k, P[k]) for k in course.KEYS if k.endswith("weight")]
        self.keys = [k for k in course.KEYS if k.endswith("weight")]
        n = len(m["shown_labels"])
        # phase 3: digit i arrives at arrive[i]; its answer and mark a little after
        self.arrive = [0.05 + 0.075 * i for i in range(n)]
        self.marked = [a + 0.035 for a in self.arrive]
        # phase 4: batch i of test_loader counted at bars[i]
        self.bars = [0.06 + 0.052 * i for i in range(m["batches"])]
        printed = m["printed"]
        self.CODE = (Code("evaluate", run=((0.0, None), (0.22, 0), (self.ZERO_AT - 0.04, 1))),
                     Code("evaluate", run=((0.0, 2),)),
                     Code("evaluate", run=((0.0, 5), (0.1, 6))),
                     Code("evaluate", run=((0.0, 3), (0.1, 6), (0.62, 7)),
                          out=((0.66, printed),)))
        self.EVIDENCE = (exact("this model has no dropout"),
                         exact("no gradients under torch.no_grad()"),
                         measured("the first ten test digits"),
                         measured("all 10,000 test digits"))

    def heading_for(self, k, u):
        return ("model.eval(): switch to test mode", "torch.no_grad(): no red tiles",
                "count the right answers",
                "test_loader: 10 batches of 1,000")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "model.eval() switches the model to test mode. It has no dropout, so nothing "
            "in it changes.",
            "Under torch.no_grad() no gradients are made: 0 red tiles, where a training step "
            "makes %s." % commas(m["grads_training"]),
            "Test digit %d is a %d, but the model says %d. Of the first %d, %d are right."
            % (m["first_wrong"], m["first_wrong_label"], m["first_wrong_pred"],
               len(m["shown_labels"]), m["shown_correct"]),
            "All %s held-out digits: %s right, so the accuracy is %s."
            % (commas(m["test"]), commas(m["correct"]), m["accuracy_text"]),
        ][k]

    # ------------------------------------------------------------------ what stays
    def correct_at(self, k, u):
        if k < 2:
            return 0
        if k == 2:
            return sum(r for r, t in zip(self.meta["shown_right"], self.marked) if u >= t)
        done = sum(u >= t for t in self.bars)
        return self.meta["running"][done - 1] if done else self.meta["shown_correct"]

    def steady(self, sc, k, u, op):
        on = op if k > 0 else op * on_at(u, self.ZERO_AT)
        x, y = self.COUNT
        self.big(sc, x, y, commas(self.correct_at(k, u)), on, label="correct", size=BIG)
        # the model, from the switch through no_grad; it goes as the first digit comes
        model = {0: on_at(u, 0.03), 1: 1.0, 2: 1.0 - on_at(u, 0.0, 0.04)}.get(k, 0.0)
        self.model(sc, op * model)
        # the ten digits, counted; they go as the first batch's bar comes
        if k == 2:
            self.digits_counted(sc, u, op)
        elif k == 3:
            self.digits_counted(sc, 1.0, op * (1.0 - on_at(u, 0.0, 0.05)))

    def model(self, sc, op):
        """The trained model's three weights, in a row, with their names."""
        if op <= 0.01:
            return
        side = self.BLOCK
        for i, (bx, vals, key) in enumerate(zip(self.BLOCK_X, self.blocks, self.keys)):
            self.weight_block(sc, vals, bx, self.BLOCK_Y, side, op)
            self.text(sc, bx, self.BLOCK_Y + self.frac(0, side / 2 + 26)[1], key, op, SIZE,
                      ink=WEIGHT_INK)
            if i:
                x0 = self.BLOCK_X[i - 1] + self.frac(side / 2 + 6)[0]
                x1 = bx - self.frac(side / 2 + 6)[0]
                self.arrow(sc, (x0, self.BLOCK_Y), (x1, self.BLOCK_Y), AXIS, op * 0.8,
                           width=1.6)

    def digits_counted(self, sc, u, op):
        """The first ten test digits as they arrive, each with the model's answer and a
        tick or a cross."""
        m = self.meta
        tile = self.CARD_TILE
        side = 28 * tile
        for i, img in enumerate(self.digits):
            a = op * on_at(u, self.arrive[i], 0.04)
            if a <= 0.01:
                continue
            fx, fy = self.COLS[i % 5], self.ROWS[i // 5]
            self.digit_card(sc, img, fx, fy, tile, a)
            b = op * on_at(u, self.marked[i], 0.03)
            py = fy + self.frac(0, side / 2 + 30)[1]
            self.text(sc, fx - 0.012, py, str(m["shown_preds"][i]), b, HEAD, ink=INK,
                      strong=True)
            if m["shown_right"][i]:
                self.tick(sc, fx + 0.017, py, b, 0.9)
            else:
                self.cross(sc, fx + 0.017, py, b, 0.9)
                self.ring(sc, fx, fy, side + 22, side + 28, b, radius=8)

    # ------------------------------------------------------------------ 1: test mode
    def phase_1(self, sc, u, op):
        cx, cy, w, h = 0.25, 0.43, 150.0, 44.0
        dw = self.frac(w)[0]
        a = op * on_at(u, 0.04)
        self.rect(sc, self.P(cx, cy), w, h, TRACK, a, border=GRID, border_w=1.4,
                  radius=h / 2, lift=0.3)
        s = eased(u, 0.24, 0.36)
        kx = cx - dw / 4 + dw / 2 * s
        self.rect(sc, self.P(kx, cy), w / 2 - 6, h - 8, PAPER, a, border=INK, border_w=2.0,
                  radius=(h - 8) / 2, lift=0.5)
        self.label(sc, self.P(kx, cy), "train" if s < 0.5 else "eval", SIZE, opacity=a,
                   ink=INK)
        # the two lines that set it, from the script
        for i, (line, px) in enumerate(((self.meta["train_mode"], 0.125),
                                        (self.meta["eval_line"], 0.375))):
            lit = s if i else 1.0 - s
            self.label(sc, self.P(px, cy + 0.072), line, SIZE, anchor="mm",
                       opacity=a * (0.45 + 0.55 * lit), ink=INK if lit > 0.5 else None,
                       kind="mono")
        b = op * on_at(u, 0.45)
        self.text(sc, 0.055, 0.83, "no dropout here: nothing changes", b, SIZE, "lm")

    # ------------------------------------------------------------------ 2: no gradients
    def phase_2(self, sc, u, op):
        side = self.BLOCK
        dx, dy = self.frac(9.0, 9.0)
        for i, bx in enumerate(self.BLOCK_X):
            # where the red tiles would be: an empty dashed slot behind each weight
            self.slot(sc, bx + dx, self.BLOCK_Y + dy, side, side, op * on_at(u, 0.06 + 0.04 * i))
            # the forward pass lights each block in turn, and leaves no red behind
            lit = op * smoothstep(0.3 + 0.1 * i, 0.34 + 0.1 * i, u) * (
                1 - smoothstep(0.4 + 0.1 * i, 0.44 + 0.1 * i, u))
            self.ring(sc, bx, self.BLOCK_Y, side + 14, side + 14, lit, radius=4)
        z = op * on_at(u, 0.62)
        self.big(sc, 0.055, 0.845, "0", z, label="red tiles made", ink=GRAD_INK)
        t = op * on_at(u, 0.75)
        self.text(sc, 0.1, 0.845, "training makes %s" % commas(self.meta["grads_training"]),
                  t, SIZE, "lm")

    def slot(self, sc, fx, fy, w, h, op):
        """An empty place for red tiles: a dashed red outline."""
        if op <= 0.01:
            return
        x0, y0 = fx - self.frac(w / 2)[0], fy - self.frac(0, h / 2)[1]
        x1, y1 = fx + self.frac(w / 2)[0], fy + self.frac(0, h / 2)[1]
        dash = 6.0
        for (ax, ay), (bx, by) in (((x0, y0), (x1, y0)), ((x1, y0), (x1, y1)),
                                   ((x1, y1), (x0, y1)), ((x0, y1), (x0, y0))):
            L = max(abs(self.span(bx - ax)[0]), abs(self.span(0, by - ay)[1]))
            n = max(1, int(L / (2 * dash)))
            for j in range(n):
                f0, f1 = j / n, (j + 0.5) / n
                self.polyline(sc, [self.P(ax + (bx - ax) * f0, ay + (by - ay) * f0),
                                   self.P(ax + (bx - ax) * f1, ay + (by - ay) * f1)],
                              1.6, GRAD_INK, op, bias=0.05)

    # ------------------------------------------------------------------ 3: one by one
    def phase_3(self, sc, u, op):
        """The digits themselves are drawn in ``steady``: they stay until the first bar."""
        m = self.meta
        self.text(sc, 0.055, 0.385, "its answer, below each",
                  op * on_at(u, self.marked[0]), SIZE, "lm", ink=QUIET)
        w = op * on_at(u, self.marked[m["first_wrong"]] + 0.06)
        self.text(sc, 0.375, 0.665, "says %d, not %d" % (m["first_wrong_pred"],
                                                         m["first_wrong_label"]), w, SIZE, "lm")
        e = op * on_at(u, self.marked[-1] + 0.05)
        self.text(sc, 0.375, 0.465, "%d of %d" % (m["shown_correct"], len(m["shown_labels"])),
                  e, HEAD, "lm", strong=True)

    # ------------------------------------------------------------------ 4: all 10,000
    def phase_4(self, sc, u, op):
        m = self.meta
        L = self.BAR_LEN
        for i, (t, n) in enumerate(zip(self.bars, m["per_batch"])):
            a = op * on_at(u, t, 0.035)
            if a <= 0.01:
                continue
            y = self.BAR_Y + self.BAR_STEP * i
            at = self.P(self.BAR_X, y)
            self.rect(sc, self.shift(at, L / 2, 0), L, 14.0, TRACK, a * 0.7, radius=3, lift=0.2)
            w = L * n / m["test_batch"]
            self.rect(sc, self.shift(at, w / 2, 0), w, 14.0, DATA, a, radius=3, lift=0.3)
            self.label(sc, self.shift(at, -8, 0), str(i + 1), SIZE, anchor="rm", opacity=a)
            self.label(sc, self.shift(at, L + 8, 0), str(n), SIZE, anchor="lm", opacity=a,
                       ink=INK)
        self.text(sc, self.BAR_X + self.frac(L)[0], self.BAR_Y - 0.04, "of 1,000",
                  op * on_at(u, self.bars[0] + 0.03), SIZE, "rm", ink=QUIET)
        f = op * on_at(u, 0.66)
        self.text(sc, 0.055, 0.83, m["fraction"], f, HEAD, "lm", strong=True)


# ====================================================================== 210
class SaveAndLoad(Shared):
    """The state_dict as six labelled boxes; their numbers; into the file; back; the same."""

    TITLE = "Saving and loading"
    PHASES = (("state_dict", 14.0), ("its numbers", 10.0), ("torch.save", 11.0),
              ("torch.load", 11.0), ("the same", 11.0))
    LAST = "9,494 of 10,000 right"
    NEXT = "one training step, slowly"
    STATION = (None,) * 5
    COL_X = (0.12, 0.255)           # weight boxes, bias boxes
    ROW_Y = (0.443, 0.583, 0.723)
    BOX_W, BOX_H, TILE_SIDE = 124.0, 62.0, 40.0
    FILE = (0.405, 0.775)
    FILE_W, FILE_H = 104.0, 92.0
    SHRINK = 0.32
    TRIP = 0.12                     # a copy's trip to or from the file (phase fraction)
    FRESH_AT = 0.04                 # phase 4: the new model appears

    def setup(self):
        m = self.meta = self.study.metadata
        self.corners = {k: np.asarray(v, float) for k, v in m["corners"].items()}
        self.fresh_corners = {k: np.asarray(v, float) for k, v in m["fresh"].items()}
        self.appear = [0.06 + 0.1 * i for i in range(6)]        # phase 1, box by box
        self.go = [0.06 + 0.075 * i for i in range(6)]          # phase 3, into the file
        self.back = [0.2 + 0.07 * i for i in range(6)]          # phase 4, out of it
        self.CODE = (Code(("save", "load"), run=((0.0, None), (0.03, 0))),
                     Code(("save", "load"), run=((0.0, 0),)),
                     Code(("save", "load"), run=((0.0, 1),)),
                     Code(("save", "load"), run=((0.0, 2), (0.14, 3))),
                     Code(("save", "load"), run=((0.0, 3),)))
        self.EVIDENCE = (exact("the trained model's six tensors"),
                         exact("numbers counted in each tensor"),
                         exact("every number, written to the file"),
                         exact("every number back: largest change 0"),
                         measured("10,000 test digits, before and after"))

    def heading_for(self, k, u):
        return ("state_dict: every weight and bias, by name", "how many numbers in it",
                "torch.save: pack them into a file", "torch.load: put them back",
                "the same answers")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "model.state_dict() holds the trained model's 6 tensors, each with its name and "
            "its shape.",
            "Together they hold %s numbers: %s bytes, at 4 bytes a number."
            % (m["numbers_text"], m["bytes_text"]),
            "torch.save writes every one of them, with its name, into the file %s." % m["file"],
            "load_state_dict puts each tensor into a new model, by its name. The largest "
            "change is %s." % val(m["max_change"], 0),
            "The loaded model gives the same answer on %s of %s test digits: accuracy %.4f."
            % (commas(m["same"]), commas(m["test"]), m["accuracy_after"]),
        ][k]

    # ------------------------------------------------------------------ geometry
    def home(self, i):
        """Box ``i``'s place in the model: weights left, biases right, one row a layer."""
        return self.COL_X[i % 2], self.ROW_Y[i // 2]

    def in_file(self, i):
        """Box ``i`` packed in the file: small, in two columns of three."""
        fx, fy = self.FILE
        dx = self.frac((i % 2 - 0.5) * (self.BOX_W * self.SHRINK + 6))[0]
        dy = self.frac(0, (i // 2 - 1) * (self.BOX_H * self.SHRINK + 5) + 6)[1]
        return fx + dx, fy + dy

    def flying(self, i, k, u):
        """``(fx, fy, scale)`` of the copy of box ``i`` on its way to or from the file, or
        None when it is not travelling."""
        hx, hy = self.home(i)
        px, py = self.in_file(i)
        if k == 2 and self.go[i] <= u < self.go[i] + self.TRIP:
            s = eased(u, self.go[i], self.go[i] + self.TRIP)
        elif k == 3 and self.back[i] <= u < self.back[i] + self.TRIP:
            s = 1.0 - eased(u, self.back[i], self.back[i] + self.TRIP)
        else:
            return None
        return hx + (px - hx) * s, hy + (py - hy) * s, 1.0 + (self.SHRINK - 1.0) * s

    def packed(self, i, k, u):
        """True once box ``i``'s copy is in the file (it stays there)."""
        return k > 2 or (k == 2 and u >= self.go[i] + self.TRIP)

    def fresh(self, i, k, u):
        """How far box ``i`` of the model shows the new model's numbers (phase 4: a new
        model, until its tensor arrives from the file)."""
        if k != 3:
            return 0.0
        return on_at(u, self.FRESH_AT) * (1.0 - on_at(u, self.back[i] + self.TRIP, 0.04))

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        # the file, from torch.save on, and what is packed in it
        f = op * self.kept(2)
        if f > 0.01:
            self.file_box(sc, f * on_at(u, 0.0, 0.05) if k == 2 else f)
            for i, key in enumerate(m["keys"]):
                if self.packed(i, k, u):
                    self.tensor_box(sc, i, key, *self.in_file(i), self.SHRINK, f, named=False)
        # the model's six tensors, in their places all through
        for i, key in enumerate(m["keys"]):
            hx, hy = self.home(i)
            on = op * (on_at(u, self.appear[i]) if k == 0 else 1.0)
            new = self.fresh(i, k, u)
            self.tensor_box(sc, i, key, hx, hy, 1.0, on, fresh=new)
        # the copies on their way
        for i, key in enumerate(m["keys"]):
            at = self.flying(i, k, u)
            if at is not None:
                self.tensor_box(sc, i, key, *at, op, lift=1.4, named=False)
        # "model", "a new model" while it waits for its numbers, "model" again: the
        # label dips to nothing at each change, so the two never overlap
        name, dip = "model", 1.0
        if k == 3:
            c1, c2 = self.FRESH_AT, self.back[-1] + self.TRIP + 0.04
            name = "a new model" if c1 <= u < c2 else "model"
            dip = min(1.0, abs(u - c1) / 0.025, abs(u - c2) / 0.025)
        self.text(sc, 0.055, 0.362, name, op * dip, SIZE, "lm",
                  ink=INK if name != "model" else QUIET)
        # no station of the loop is lit: saving and loading come after it
        self.text(sc, MAP[0], MAP[1], "after training", op, SIZE, ink=INK)

    def tensor_box(self, sc, i, key, fx, fy, z, op, fresh=0.0, lift=0.6, named=True):
        """One tensor of the state_dict: its tiles, its name, its shape (``fresh``: how far
        its tiles show the new model's numbers instead)."""
        if op <= 0.01:
            return
        m = self.meta
        w, h = self.BOX_W * z, self.BOX_H * z
        lift = lift + 0.02 * i
        self.rect(sc, self.P(fx, fy), w, h, soft(WEIGHT, 0.86), op, border=WEIGHT_INK,
                  border_w=1.6, radius=6 * z, lift=lift)
        t = self.TILE_SIDE * z
        tx = fx - self.frac(w / 2 - 8 * z - t / 2)[0]
        for vals, on, dz in ((self.corners[key], op * (1 - fresh), 0.01),
                             (self.fresh_corners[key], op * fresh, 0.012)):
            if on <= 0.01:
                continue
            if vals.ndim == 2:
                self.tiles(sc, vals, tx, fy, t, t, on, ink=WEIGHT, lift=lift + dz, grid=0.2)
            else:
                self.tiles(sc, vals[:, None], tx, fy, t / 4, t, on, ink=WEIGHT, lift=lift + dz,
                           grid=0.2)
        if named:
            lx = tx + self.frac(t / 2 + 10)[0]
            self.text(sc, lx, fy - 0.022, key, op, SIZE, "lm", ink=INK)
            self.text(sc, lx, fy + 0.022, m["shape_texts"][i], op, SIZE, "lm", ink=WEIGHT_INK)

    def file_box(self, sc, op):
        fx, fy = self.FILE
        c = self.P(fx, fy)
        self.rect(sc, c, self.FILE_W, self.FILE_H, PAPER, op, border=INK, border_w=2.0,
                  radius=6, lift=0.4)
        self.rect(sc, self.shift(c, 0, -self.FILE_H / 2 + 5), self.FILE_W - 6, 6.0, AXIS, op,
                  radius=2, lift=0.42)
        self.text(sc, fx, fy + self.frac(0, self.FILE_H / 2 + 16)[1], self.meta["file"], op,
                  SIZE, ink=INK)

    # ------------------------------------------------------------------ 1: state_dict
    def phase_1(self, sc, u, op):
        on = op * on_at(u, self.appear[-1] + 0.1)
        self.text(sc, 0.345, 0.79, "6 tensors", on, SIZE, "lm")

    # ------------------------------------------------------------------ 2: its numbers
    def phase_2(self, sc, u, op):
        m = self.meta
        for r in range(3):
            a = op * on_at(u, 0.05 + 0.12 * r)
            y = self.ROW_Y[r]
            s = "%s + %s" % (m["size_texts"][2 * r], m["size_texts"][2 * r + 1])
            self.text(sc, 0.34, y, s, a, SIZE, "lm", ink=INK)
        t = op * on_at(u, 0.45)
        self.big(sc, 0.055, 0.862, m["numbers_text"], t, label="numbers", size=BIG)
        b = op * on_at(u, 0.62)
        self.text(sc, 0.205, 0.862, "× 4 bytes = %s bytes" % m["bytes_text"], b, SIZE, "lm")

    # ------------------------------------------------------------------ 3: torch.save
    def phase_3(self, sc, u, op):
        pass

    # ------------------------------------------------------------------ 4: torch.load
    def phase_4(self, sc, u, op):
        on = op * on_at(u, self.back[-1] + 0.18)
        self.big(sc, 0.055, 0.862, "0", on, label="largest change", size=BIG)

    # ------------------------------------------------------------------ 5: the same
    def phase_5(self, sc, u, op):
        m = self.meta
        a = op * on_at(u, 0.06)
        self.big(sc, 0.055, 0.862, commas(m["same"]), a, label="same answers", size=BIG)
        b = op * on_at(u, 0.22)
        self.text(sc, 0.19, 0.862, "of %s test digits" % commas(m["test"]), b, SIZE, "lm")


# ====================================================================== 211
class OneStep(Shared):
    """Data, forward, loss, backward, step: the five stations of one recorded step."""

    TITLE = "One training step, slowly"
    PHASES = (("data", 10.0), ("forward", 12.0), ("loss", 10.0), ("backward", 13.0),
              ("step", 13.0))
    LAST = "saved and loaded: the same answers"
    NEXT = "938 steps, fast"
    STATION = ("data", "forward", "loss", "backward", "step")
    Y = 0.50                        # the row of the machine
    DATA_X, TILE = 0.085, 1.7
    W_X, W_SIDE = (0.18, 0.255, 0.33), 44.0
    LOGIT_X, METER_X = 0.405, 0.455
    METER_H = 110.0
    METER_MAX = 2.5                 # the meter's full height, in loss

    def setup(self):
        m = self.meta = self.study.metadata
        b = course.batch(m["step"])
        self.first = b["x"][0, 0]
        self.corners = [np.asarray(m["corners"][k], float) for k in m["weights"]]
        self.grads = [np.asarray(m["grads"][k], float) for k in m["weights"]]
        self.old = [np.asarray(m["old_grads"][k], float) for k in m["weights"]]
        g = max(np.abs(x).max() for x in self.grads + self.old)
        self.glo, self.ghi = -g, g
        self.logits = np.asarray(m["logits"], float)
        run = ((0.0, 0),), ((0.0, 1),), ((0.0, 2),), ((0.0, 3), (0.34, 4)), ((0.0, 5),)
        self.CODE = tuple(Code("step", run=r) for r in run)
        what = "step %d of %d, recorded" % (m["number"], m["steps"])
        self.EVIDENCE = (measured(what),) * 5
        # backward: the red tiles appear from the last layer back to the first
        self.red_at = (0.66, 0.53, 0.40)

    def heading_for(self, k, u):
        if k == 3 and u < 0.34:
            return "zero_grad: empty last step's red tiles"
        return ("data: a batch of 64 digits", "forward: through every block",
                "loss: one number, how wrong", "backward: a red tile per weight",
                "step: every weight moves")[k]

    def subtitle_for(self, k):
        m = self.meta
        w = m["weight"]
        return [
            "Step %d of %d starts with a batch: x is %s, y is %s." % (
                m["number"], m["steps"], m["x_shape"], m["y_shape"]),
            "The batch goes through the model: %s becomes %s, then %s, then %s." % (
                m["shapes"][1], m["shapes"][2], m["shapes"][3], m["shapes"][4]),
            "The loss of this batch is %.2f, near ln 10 = %.2f: the model is still guessing."
            % (m["loss"], m["ln10"]),
            "zero_grad takes away last step's gradients; backward makes %s new ones."
            % commas(m["numbers"]),
            "Each weight moves by 0.1 × its gradient: %s becomes %s. On this batch the loss "
            "falls to %.2f."
            % (val(w["before"], 4), val(w["after"], 4), m["loss_after"]),
        ][k]

    # ------------------------------------------------------------------ the machine
    def meter_loss(self, k, u):
        m = self.meta
        if k < 2:
            return None
        if k == 2:
            return m["loss"] * eased(u, 0.12, 0.4)
        if k == 4:
            return m["loss"] + (m["loss_after"] - m["loss"]) * eased(u, 0.72, 0.82)
        return m["loss"]

    def red(self, i, k, u):
        """How much of block ``i``'s red tiles show: ``(old, new)``."""
        old = 1.0
        if k == 3:
            old = 1.0 - eased(u, 0.06 + 0.06 * (2 - i), 0.12 + 0.06 * (2 - i))
        elif k > 3:
            old = 0.0
        new = 0.0
        if k == 3:
            new = on_at(u, self.red_at[i], 0.05)
        elif k > 3:
            new = 1.0
        return old, new

    def steady(self, sc, k, u, op):
        m = self.meta
        y = self.Y
        # the batch
        self.stack(sc, self.DATA_X, y, 28 * self.TILE + 10, 28 * self.TILE + 16, 4, op,
                   step=(3.0, -3.0))
        self.tiles(sc, self.first, self.DATA_X, y + self.frac(0, 2.5)[1], 28 * self.TILE,
                   28 * self.TILE, op, lo=LO, hi=HI, lift=0.25, grid=0.0)
        self.text(sc, self.DATA_X, y + 0.1, "x, y", op, SIZE, ink=DATA_INK)
        # the three weights, their red tiles behind them
        side = self.W_SIDE
        for i, x in enumerate(self.W_X):
            old, new = self.red(i, k, u)
            if old > 0.01:
                self.grad_tiles(sc, self.old[i], x, y, side, side, op * old * 0.55,
                                lo=self.glo, hi=self.ghi)
            if new > 0.01:
                self.grad_tiles(sc, self.grads[i], x, y, side, side, op * new,
                                lo=self.glo, hi=self.ghi)
            self.weight_block(sc, self.corners[i], x, y, side, op, lift=0.4)
            self.text(sc, x, y + 0.1, m["weights"][i], op, SIZE, ink=WEIGHT_INK)
        # arrows: data -> weights -> scores -> loss
        xs = [self.DATA_X + 0.045] + [a for x in self.W_X for a in (x - 0.03, x + 0.03)] + [
            self.LOGIT_X - 0.016, self.LOGIT_X + 0.016, self.METER_X - 0.022]
        for a, b in zip(xs[0::2], xs[1::2]):
            self.arrow(sc, (a, y), (b, y), AXIS, op * 0.7, width=1.5)
        # the 10 scores of the first digit, once the forward pass has made them
        sc_on = op * (on_at(u, 0.62) if k == 1 else float(k > 1))
        self.text(sc, self.LOGIT_X, y + 0.1, "logits", op, SIZE, ink=DATA_INK)
        self.rect(sc, self.P(self.LOGIT_X, y), 18.0, 86.0, PAPER, op, border=GRID,
                  border_w=1.2, radius=3, lift=0.2)
        if sc_on > 0.01:
            top = [(m["first_pred"], 0)] if k == 1 and u >= 0.78 else []
            self.tiles(sc, self.logits[:, None], self.LOGIT_X, y, 12.0, 80.0, sc_on, ink=DATA,
                       lift=0.3, grid=0.3, ring=top)
        # the loss meter
        loss = self.meter_loss(k, u)
        frac = 0.0 if loss is None else loss / self.METER_MAX
        self.meter(sc, self.METER_X, y, frac, op, h=self.METER_H, w=24.0)
        if loss is not None:
            self.text(sc, self.METER_X, y + self.frac(0, self.METER_H / 2 + 16)[1],
                      "%.2f" % loss, op * on_at(u, 0.12) if k == 2 else op, SIZE,
                      ink=LOSS_INK)

    # ------------------------------------------------------------------ 1: data
    def phase_1(self, sc, u, op):
        m = self.meta
        a = op * on_at(u, 0.08)
        self.ring(sc, self.DATA_X + 0.004, self.Y - 0.004, 28 * self.TILE + 30,
                  28 * self.TILE + 36, a, radius=8)
        b = op * on_at(u, 0.2)
        self.text(sc, 0.055, 0.68, "x: %s" % m["x_shape"], b, SIZE, "lm", ink=DATA_INK)
        c = op * on_at(u, 0.4)
        self.text(sc, 0.055, 0.73, "y: %s" % m["y_shape"], c, SIZE, "lm", ink=DATA_INK)
        for i, v in enumerate(m["labels"][:8]):
            d = op * on_at(u, 0.5 + 0.03 * i, 0.04)
            self.box(sc, 0.07 + 0.037 * i, 0.80, 30.0, 26.0, soft(DATA, 0.82), d, border=DATA,
                     text=str(v), size=SIZE, radius=4)
        self.text(sc, 0.07 + 0.037 * 8 - 0.012, 0.80, "...", op * on_at(u, 0.78), SIZE, "lm")

    # ------------------------------------------------------------------ 2: forward
    def phase_2(self, sc, u, op):
        m = self.meta
        # the batch's shape after each block, as the pass reaches it
        xs = [self.DATA_X] + list(self.W_X)
        for i, (x, s) in enumerate(zip(xs, m["shapes"][1:])):
            t = 0.06 + 0.13 * i
            a = op * on_at(u, t)
            self.text(sc, x, 0.665, s, a, SIZE, ink=DATA_INK)
            lit = op * smoothstep(t, t + 0.04, u) * (1 - smoothstep(t + 0.11, t + 0.15, u))
            if i:
                self.ring(sc, x, self.Y, self.W_SIDE + 16, self.W_SIDE + 16, lit, radius=4)
        p = op * on_at(u, 0.68)
        self.text(sc, 0.055, 0.78, "first digit: a %d" % m["first_label"], p, SIZE, "lm")
        q = op * on_at(u, 0.78)
        self.text(sc, 0.055, 0.83, "highest score: %d" % m["first_pred"], q, SIZE, "lm")
        self.ring(sc, self.LOGIT_X, self.Y, 30.0, 98.0, op * on_at(u, 0.62), radius=5)

    # ------------------------------------------------------------------ 3: loss
    def phase_3(self, sc, u, op):
        m = self.meta
        self.ring(sc, self.METER_X, self.Y, 40.0, self.METER_H + 16, op * on_at(u, 0.05),
                  radius=8)
        a = op * on_at(u, 0.42)
        self.big(sc, 0.055, 0.80, "%.2f" % m["loss"], a, label="loss", ink=LOSS_INK)
        b = op * on_at(u, 0.6)
        self.text(sc, 0.2, 0.80, "ln 10 = %.2f" % m["ln10"], b, SIZE, "lm")
        self.text(sc, 0.2, 0.845, "a guess", b, SIZE, "lm")

    # ------------------------------------------------------------------ 4: backward
    def phase_4(self, sc, u, op):
        m = self.meta
        a = op * on_at(u, 0.04) * (1 - on_at(u, 0.3, 0.04))
        self.text(sc, 0.055, 0.70, "old red tiles: gone", a, SIZE, "lm")
        # from the loss back to the first weight
        b = op * on_at(u, 0.36)
        self.arc(sc, (self.METER_X, self.Y + 0.175), (self.W_X[0], self.Y + 0.175), GRAD_INK,
                 b * 0.85, eased(u, 0.38, 0.72), bend=-0.04, width=2.0)
        c = op * on_at(u, 0.76)
        self.big(sc, 0.055, 0.80, commas(m["numbers"]), c, label="red tiles", ink=GRAD_INK)

    # ------------------------------------------------------------------ 5: step
    def phase_5(self, sc, u, op):
        m = self.meta
        w = m["weight"]
        # the hand passes over the weights, one after the other
        s = eased(u, 0.06, 0.4)
        hx = self.W_X[0] + (self.W_X[-1] - self.W_X[0]) * s
        self.hand(sc, hx, self.Y - 0.115, op * (1 - on_at(u, 0.44, 0.04)), size=0.9)
        # one tile of 5.weight read out
        r, c = w["index"]
        cell = self.cells(self.W_X[2], self.Y, self.W_SIDE, self.W_SIDE, 8, 8)
        px, py = cell(r, c)
        ring = op * on_at(u, 0.42)
        self.ring(sc, px, py, 10.0, 10.0, ring, lift=1.2, width=2.0, radius=2)
        rows = (("weight", val(w["before"], 4), 0.46, INK),
                ("gradient", val(w["grad"], 4), 0.52, GRAD_INK),
                ("new weight", val(w["after"], 4), 0.6, INK))
        for i, (name, v, t, ink) in enumerate(rows):
            a = op * on_at(u, t)
            yy = 0.69 + 0.048 * i
            self.text(sc, 0.055, yy, name, a, SIZE, "lm")
            self.text(sc, 0.255, yy, v, a, SIZE, "rm", ink=ink)
        self.text(sc, 0.055, 0.69 + 0.048 * 3, "%s − 0.1 × (%s)" % (val(w["before"], 4),
                                                                   val(w["grad"], 4)),
                  op * on_at(u, 0.6), SIZE, "lm", ink=QUIET)
        e = op * on_at(u, 0.78)
        self.text(sc, 0.30, 0.69, "loss", e, SIZE, "lm")
        self.text(sc, 0.30, 0.737, "%.2f → %.2f" % (m["loss"], m["loss_after"]), e, SIZE, "lm",
                  ink=LOSS_INK)


ANIMATIONS = {"blocks-evaluation": Evaluation, "blocks-save-and-load": SaveAndLoad,
              "blocks-one-step-slow-motion": OneStep}

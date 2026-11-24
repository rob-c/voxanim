"""Animations 200-202 of the "Blocks and code" series (see :mod:`voxanim.blocks`): the
forward pass, the loss and ``loss.backward()``, drawn from :mod:`voxanim.study_blocks_f`
(the first batch, the untrained weights of step 0).

* :class:`ForwardPass` (``blocks-forward-pass``): ``logits = model(x)``: the model's six
  blocks stacked top to bottom, the batch reaching them one at a time; at each, the batch
  after it appears as a blue block of 64 rows, as wide as its numbers per digit, and its
  shape beside it: ``[64, 1, 28, 28]`` to ``[64, 784]``, ``[64, 64]`` (ReLU greys the
  negative half), ``[64, 32]``, ``[64, 10]``.
* :class:`TheLoss` (``blocks-the-loss``): ``loss = loss_fn(logits, y)``: one digit's
  chances and its loss, ``-ln`` of its right label's chance; one purple bar per digit, 64
  of them; their mean, 2.322, filling the meter; against ``ln 10 = 2.303``, a blind guess.
* :class:`Backward` (``blocks-backward``): ``loss.backward()``: the three Linear blocks'
  orange weights and bias strips; red tiles appear behind them from the loss up, the last
  layer first, the count of numbers with a gradient running to 52,650; then each ``.grad``
  slides out beside its weight: the same shape (``print(model[1].weight.grad.shape)``).

The three share one picture: the blocks of the model as a column of labelled boxes, top to
bottom in the order of ``nn.Sequential``; the forward pass goes down it, the backward pass
up. Everything that stays across phases is drawn in ``steady`` from absolute times, so
nothing blanks between them.
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
    SIZE,
    WEIGHT,
    ZERO,
    Blocks,
    Code,
    code_lines,
    exact,
    measured,
    shades,
)
from .diagram import draw_image, shrink_columns
from .easing import ease_in_out, smoothstep
from .mech import AXIS, PAPER, dark, soft

DATA_INK = dark(DATA, 0.3)
WEIGHT_INK = dark(WEIGHT, 0.35)
GRAD_INK = dark(GRAD, 0.25)
LOSS_INK = dark(LOSS, 0.3)
BX, BW, BH = 0.12, 124.0, 28.0          # the column of blocks: centre x, box w and h (px)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def _pick(steps, u):
    """The value of the last ``(u0, value)`` of ``steps`` with ``u0 <= u``."""
    out = steps[0][1]
    for at, v in steps:
        if u >= at:
            out = v
    return out


def shrink(tex, rows, cols):
    """``tex`` (r, c, 3) averaged down to at most ``rows x cols`` (see
    :func:`~voxanim.diagram.shrink_columns`)."""
    t = shrink_columns(tex, max(1, int(cols)))
    return shrink_columns(t.transpose(1, 0, 2), max(1, int(rows))).transpose(1, 0, 2)


class Column(Blocks):
    """What the three share: absolute times, the column of blocks, a block of numbers."""

    def at(self, k, u):
        """The time of ``u`` of phase ``k`` (from :attr:`PHASES`: usable in ``setup``)."""
        a = sum(seconds for _name, seconds in self.PHASES[:k])
        return a + self.PHASES[k][1] * u

    def since(self, t0, d=0.6):
        """0 before ``t0``, easing to 1 over ``d`` seconds."""
        return smoothstep(t0, t0 + d, getattr(self, "_t", 0.0))

    def block(self, sc, values, fx, fy, w, h, op, ink=DATA, lo=None, hi=None, zero=None,
              lift=0.3):
        """A 2-D array as one ``w x h`` px block centred at ``(fx, fy)``, its shades
        averaged down to the pixels there are (no cell lines)."""
        if op <= 0.01:
            return
        tex = shrink(shades(values, ink, lo, hi, zero), h, w)
        draw_image(self, sc, self.P(float(fx), float(fy)), tex, w, h, op, grid=0.0, lift=lift)

    def block_box(self, sc, kind, name, fy, op, fx=BX, w=BW, h=BH):
        """One block of the model as a labelled box: orange for a Linear (it holds
        weights), grey for ReLU (it makes zeros), plain for Flatten, a blue card for the
        data."""
        if op <= 0.01:
            return
        if kind == "input":
            self.card(sc, fx, fy, w, h + 4, op, lift=0.2)
            self.text(sc, fx, fy + self.frac(0, 2)[1], name, op, SIZE, ink=DATA_INK)
            return
        fill, border = {"linear": (soft(WEIGHT, 0.6), WEIGHT),
                        "relu": (soft(ZERO, 0.55), ZERO)}.get(kind, (PAPER, AXIS))
        self.rect(sc, self.P(fx, fy), w, h, fill, op, border=border, border_w=1.8, radius=6,
                  lift=0.2)
        self.text(sc, fx, fy, name, op, SIZE, ink=INK)

    def ring(self, sc, fx, fy, w, h, op):
        """The outline round what is under way."""
        self.rect(sc, self.P(fx, fy), w, h, INK, op, border=INK, border_w=2.4, radius=8,
                  lift=1.0, fill_op=0.0)


# ====================================================================== 200
class ForwardPass(Column):
    """The batch down the stack of blocks, its shape changing at each."""

    TITLE = "The forward pass"
    PHASES = (("the batch", 10.0), ("Flatten", 10.0), ("Linear", 10.0),
              ("ReLU, Linear, ReLU", 15.0), ("the logits", 12.0))
    LAST = "the loader hands out batches of 64"
    NEXT = "the loss: how wrong are the logits?"
    STATION = ("forward",) * 5
    ROW0, ROW_STEP = 0.392, 0.0775
    AX = 0.205                          # the blocks of numbers start here
    SX = 0.352                          # the shapes
    AH = 28.0
    # when the batch reaches each row: (phase, u)
    REACH = ((0, 0.3), (1, 0.15), (2, 0.12), (3, 0.05), (3, 0.38), (3, 0.71), (4, 0.1))

    def setup(self):
        m = self.meta = self.study.metadata
        p = self.study.panels
        self.digits = np.asarray(p[0].data["values"], float)
        flat = self.digits.reshape(len(self.digits), -1)
        z1, h1 = np.asarray(p[1].data["before"]), np.asarray(p[1].data["values"])
        z2, h2 = np.asarray(p[2].data["before"]), np.asarray(p[2].data["values"])
        logits = np.asarray(p[3].data["values"])
        # each row's numbers, and the scale they are shaded on (ReLU keeps its input's, so
        # that the tiles it leaves alone look the same, and the ones it zeroes turn grey)
        self.rows = [None, (flat, DATA_LO, DATA_HI, None),
                     (z1, z1.min(), z1.max(), None), (h1, z1.min(), z1.max(), h1 == 0),
                     (z2, z2.min(), z2.max(), None), (h2, z2.min(), z2.max(), h2 == 0),
                     (logits, logits.min(), logits.max(), None)]
        self.ys = [self.ROW0 + self.ROW_STEP * i for i in range(7)]
        self.reach = [self.at(k, u) for k, u in self.REACH]
        lines, ch = m["lines"], m["changes"]
        call = m["step_code"].index(m["call"])
        self.CODE = (
            Code("step", run=((0.0, None), (0.3, call))),
            Code("model", run=((0.0, 0), (0.15, lines[1])), out=((0.3, ch[0]),)),
            Code("model", run=((0.0, lines[2]),), out=((0.25, ch[1]),)),
            Code("model", run=((0.0, lines[3]), (0.33, lines[4]), (0.66, lines[5])),
                 out=((0.13, ch[2]), (0.33, ""), (0.46, ch[3]), (0.66, ""), (0.79, ch[4]))),
            Code("model", run=((0.0, lines[6]),), out=((0.22, ch[5]),)))
        self.EVIDENCE = (exact("the first batch, untrained weights"),) * 5

    def width(self, n):
        """A block's width (px) for ``n`` numbers per digit: wider for more (log scale)."""
        return 24.0 + 101.0 * np.log(n / 10.0) / np.log(78.4)

    def heading_for(self, k, u):
        n = self.meta["names"]
        if k == 3:
            return _pick(((0.0, "ReLU: negatives become 0"),
                          (0.33, "%s: 32 per digit" % n[4]),
                          (0.66, "ReLU again: negatives become 0")), u)
        return ["logits = model(x): the batch goes in",
                "%s: each digit becomes a strip" % n[1],
                "%s: 64 numbers per digit" % n[2],
                None,
                "the logits: 10 scores per digit"][k]

    def subtitle_for(self, k):
        m = self.meta
        s = m["shape_texts"]
        return [
            "logits = model(x) sends the whole batch, %d digits, through the model's six "
            "blocks in order." % m["batch"],
            "Flatten makes each digit one strip of 784 numbers: %s becomes %s." % (s[0], s[1]),
            "Linear(784, 64) turns each strip into 64 numbers: %s becomes %s." % (s[1], s[2]),
            "ReLU turns %s of them to 0 (grey), Linear(64, 32) makes %s, and ReLU zeroes %s."
            % (m["zeroed_text"][0], s[4], m["zeroed_text"][1]),
            "The last Linear gives 10 scores per digit, the logits: %s. Untrained, they are "
            "small." % s[6],
        ][k]

    # ------------------------------------------------------------------ what stays
    def lit_row(self):
        """The row under way, as a float (between rows while the ring slides)."""
        t = getattr(self, "_t", 0.0)
        i = sum(t >= r for r in self.reach) - 1
        if i < 0:
            return None
        if i == 0:
            return 0.0
        return i - 1 + ease_in_out(float(np.clip((t - self.reach[i]) / 0.6, 0.0, 1.0)))

    def steady(self, sc, k, u, op):
        m = self.meta
        ys = self.ys
        for i, y in enumerate(ys):
            self.block_box(sc, m["kinds"][i], "x: the batch" if i == 0 else m["names"][i], y,
                           op)
            if i:
                self.arrow(sc, (BX, ys[i - 1] + self.frac(0, BH / 2 + 2)[1]),
                           (BX, y - self.frac(0, BH / 2 + 1)[1]), AXIS, op * 0.8, width=1.4)
        # the input: the batch's first digits as a little stack of cards
        for j in reversed(range(3)):
            dx, dy = self.frac(5.0 * j, -4.0 * j)
            cx, cy = self.AX + self.frac(17)[0] + dx, ys[0] + dy
            self.card(sc, cx, cy, 32.0, 36.0, op, lift=0.3 - 0.01 * j)
            self.tiles(sc, self.digits[j], cx, cy + self.frac(0, 2)[1], 28.0, 28.0, op,
                       lo=DATA_LO, hi=DATA_HI, lift=0.305 - 0.01 * j, grid=0.0)
        self.text(sc, self.SX, ys[0], m["shape_texts"][0], op, SIZE, "lm", ink=DATA_INK)
        # each row's numbers and shape, once the batch has reached it
        for i in range(1, len(ys)):
            t0 = self.reach[i]
            a = op * self.since(t0 + 0.4)
            if a <= 0.01:
                continue
            v, lo, hi, zero = self.rows[i]
            w = self.width(v.shape[1])
            self.block(sc, v, self.AX + self.frac(w / 2)[0], ys[i], w, self.AH, a, lo=lo,
                       hi=hi, zero=zero)
            self.rect(sc, self.P(self.AX + self.frac(w / 2)[0], ys[i]), w + 4, self.AH + 4,
                      DATA, a, border=DATA, border_w=1.4, radius=3, lift=0.25, fill_op=0.0)
            self.text(sc, self.SX, ys[i], m["shape_texts"][i], op * self.since(t0 + 1.2), SIZE,
                      "lm", ink=DATA_INK)
        # the ring round the row under way
        r = self.lit_row()
        if r is not None:
            y = ys[0] + self.ROW_STEP * r
            on = op * self.since(self.reach[0], 0.5)
            x0, x1 = 0.052, self.SX - 0.008
            self.ring(sc, (x0 + x1) / 2, y, self.span(x1 - x0)[0], BH + 12, on)

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        # the batch, counted
        self.text(sc, self.SX, self.ys[0] + 0.045, "%d digits" % self.meta["batch"],
                  op * smoothstep(0.4, 0.47, u), SIZE, "lm")

    def phase_2(self, sc, u, op):
        self.text(sc, self.SX, self.ys[1] + 0.045, "784 = 28 × 28", op * smoothstep(0.4, 0.47, u),
                  SIZE, "lm")

    def phase_3(self, sc, u, op):
        self.text(sc, self.SX, self.ys[2] + 0.045, "64 per digit", op * smoothstep(0.4, 0.47, u),
                  SIZE, "lm")

    def phase_4(self, sc, u, op):
        for i, a, b in ((3, 0.12, 0.33), (5, 0.78, 1.01)):
            on = op * smoothstep(a, a + 0.04, u) * (1 - smoothstep(b - 0.04, b, u))
            self.text(sc, self.SX, self.ys[i] + 0.045, "%s are 0" % self.meta["zeroed_text"][
                (i - 3) // 2], on, SIZE, "lm", ink=dark(ZERO, 0.4))

    def phase_5(self, sc, u, op):
        y = self.ys[6]
        self.text(sc, self.SX + 0.065, y, "logits", op * smoothstep(0.3, 0.36, u), SIZE, "lm",
                  strong=True)
        # the shape at every block, outlined together
        on = op * smoothstep(0.5, 0.56, u)
        top, bottom = self.ys[0] - 0.03, y + 0.03
        self.rect(sc, self.P(self.SX + 0.052, (top + bottom) / 2), self.span(0.118)[0],
                  self.span(0, bottom - top)[1], INK, on, border=INK, border_w=2.0, radius=8,
                  lift=1.0, fill_op=0.0)


# ====================================================================== 201
class TheLoss(Column):
    """One digit's loss; one bar per digit; their mean on the meter; ln 10."""

    TITLE = "The loss"
    PHASES = (("one digit", 14.0), ("one bar per digit", 15.0), ("their mean", 13.0),
              ("about ln 10", 13.0))
    LAST = "logits = model(x): [64, 10]"
    NEXT = "loss.backward(): a gradient for every weight"
    STATION = ("loss",) * 4
    BASE = 0.862                        # the bars' baseline: the meter's inside bottom
    MX, MH, MW = 0.445, 150.0, 26.0
    TOP = 3.0                           # the meter's and the bars' full scale
    BARS = (0.064, 0.398)
    CARD, TILE = (0.105, 0.478), 3.0
    P0, P_STEP, P_BASE, P_SCALE = 0.228, 0.0195, 0.565, 680.0

    def setup(self):
        m = self.meta = self.study.metadata
        b = course.batch(m["step"])
        self.img = np.asarray(b["x"][m["digit"], 0], float)
        self.losses = np.asarray(m["losses"])
        self.chances = np.asarray(m["chances"])
        n = len(self.losses)
        # the bars' arrivals in phase 2: three slowly, then the rest as a stream
        self.arrive = [0.1 + 0.12 * i for i in range(3)] + list(
            0.48 + 0.42 * np.linspace(0, 1, n - 4) ** 0.9)
        self.first_bar = self.at(0, 0.58)
        self.fill_at = self.at(2, 0.55)
        self.my = self.BASE - self.frac(0, self.MH / 2 - 3)[1]
        out = "loss: %s" % m["loss_text"]
        self.CODE = (Code(("loss", "step"), run=((0.0, 0), (0.35, 3))),
                     Code(("loss", "step"), run=((0.0, 3),)),
                     Code(("loss", "step"), run=((0.0, 3),), out=((0.55, out),)),
                     Code(("loss", "step"), run=((0.0, 3),), out=((0.0, out),)))
        self.EVIDENCE = (measured("the first batch, untrained"),) * 3 + (
            exact("−ln(1/10) = ln 10 = %s" % m["ln10_text"]),)

    def heading_for(self, k, u):
        return ["one digit: its chance of being a %d" % self.meta["label"],
                "one loss for every digit in the batch",
                "the loss: the mean, one number",
                "untrained: about ln 10, a blind guess"][k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "This digit is a %d. The model gives 0 a chance of %s, so its loss is %s."
            % (m["label"], m["chance_text"], m["digit_loss_text"]),
            "Every digit gets a loss the same way: %d bars, from %.2f to %.2f."
            % (m["batch"], m["lowest"], m["highest"]),
            "The loss is their mean: %s. One number says how wrong the model is."
            % m["mean_text"],
            "Guessing 1 in 10 for every digit gives ln 10 = %s; the untrained model scores %s."
            % (m["ln10_text"], m["loss_text"]),
        ][k]

    # ------------------------------------------------------------------ geometry
    def bar_x(self, i):
        x0, x1 = self.BARS
        return x0 + (x1 - x0) * (i + 0.5) / len(self.losses)

    def bar_h(self, v):
        return float(v) / self.TOP * (self.MH - 6)

    def bars_in(self, k, u):
        if k < 1:
            return 1 if self.since(self.first_bar) > 0 else 0
        if k > 1:
            return len(self.losses)
        return 1 + sum(u >= a for a in self.arrive)

    def draw_bar(self, sc, i, grow, op):
        h = self.bar_h(self.losses[i]) * grow
        if h < 0.5 or op <= 0.01:
            return
        w = self.span((self.BARS[1] - self.BARS[0]) / len(self.losses))[0] * 0.72
        self.rect(sc, self.P(self.bar_x(i), self.BASE - self.frac(0, h / 2)[1]), w, h, LOSS, op,
                  radius=min(1.5, w / 2), lift=0.3)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        # the baseline and the bars so far
        x0, x1 = self.BARS
        self.polyline(sc, [self.P(x0 - 0.008, self.BASE + 0.003), self.P(x1 + 0.008,
                                                                         self.BASE + 0.003)],
                      1.4, AXIS, op, bias=0.2)
        n = self.bars_in(k, u)
        for i in range(n):
            if i == 0:
                grow = self.since(self.first_bar, 0.8)
            elif k == 1:
                grow = smoothstep(self.arrive[i - 1], self.arrive[i - 1] + (
                    0.04 if i < 4 else 0.015), u)
            else:
                grow = 1.0
            self.draw_bar(sc, i, grow, op)
        self.text(sc, x0 - 0.008, self.BASE + 0.033, "one bar per digit", op * self.kept(1),
                  SIZE, "lm", ink=LOSS_INK)
        # the meter: empty until the mean fills it
        fill = smoothstep(self.fill_at, self.fill_at + 1.2, getattr(self, "_t", 0.0))
        v = float(m["loss"]) * fill
        self.meter(sc, self.MX, self.my, v / self.TOP, op, h=self.MH, w=self.MW)
        self.text(sc, self.MX, self.my - self.frac(0, self.MH / 2 + 14)[1], "loss", op, SIZE,
                  ink=LOSS_INK)
        if fill > 0:
            self.text(sc, self.MX, self.BASE + 0.033, m["loss_text"], op * self.since(
                self.fill_at + 0.8), SIZE, strong=True)
        # the digit card, while the losses are worked out
        cx, cy = self.CARD
        on = op * self.kept(0, 1)
        self.digit_card(sc, self.img, cx, cy, self.TILE, on)
        self.text(sc, cx, cy + 0.12, "y = %d" % m["label"], on, SIZE, ink=DATA_INK)

    # ------------------------------------------------------------------ 1: one digit
    def phase_1(self, sc, u, op):
        m = self.meta
        right = m["label"]
        for d, p in enumerate(self.chances):
            on = op * smoothstep(0.12 + 0.015 * d, 0.17 + 0.015 * d, u)
            x = self.P0 + self.P_STEP * d
            h = p * self.P_SCALE
            self.rect(sc, self.P(x, self.P_BASE - self.frac(0, h / 2)[1]), 13.0, h,
                      soft(DATA, 0.25), on, border=DATA, border_w=1.0, radius=2, lift=0.3)
            self.text(sc, x, self.P_BASE + 0.024, str(d), on, SIZE)
        top = self.P_BASE - self.frac(0, self.chances[right] * self.P_SCALE)[1]
        hit = op * smoothstep(0.32, 0.38, u)
        x = self.P0 + self.P_STEP * right
        self.rect(sc, self.P(x, (top + self.P_BASE + 0.024) / 2), 19.0,
                  self.span(0, self.P_BASE + 0.045 - top)[1], INK, hit, border=INK,
                  border_w=2.2, radius=3, lift=1.0, fill_op=0.0)
        self.text(sc, x + 0.004, top - 0.036, m["chance_text"], hit, SIZE, strong=True)
        self.text(sc, self.P0 + self.P_STEP * 4.5, self.P_BASE + 0.06, "its chances, 0 to 9",
                  op * smoothstep(0.12, 0.18, u), SIZE, ink=DATA_INK)
        # its loss: the first bar
        h = self.bar_h(self.losses[0])
        bx = self.bar_x(0)
        self.text(sc, bx + 0.012, self.BASE - self.frac(0, h)[1] + 0.02, m["digit_loss_text"],
                  op * smoothstep(0.66, 0.72, u), HEAD, "lm", ink=LOSS_INK)

    # ------------------------------------------------------------------ 2: one bar per digit
    def phase_2(self, sc, u, op):
        n = self.bars_in(1, u)
        self.running(sc, 0.235, 0.49, [(0.0, "%d" % n)], 0.0, op * smoothstep(0.03, 0.08, u),
                     label="digits with a loss", size=BIG)
        self.text(sc, 0.235, 0.555, "of %d" % self.meta["batch"], op * smoothstep(0.03, 0.08, u),
                  SIZE, "lm")

    # ------------------------------------------------------------------ 3: their mean
    def phase_3(self, sc, u, op):
        m = self.meta
        a = op * smoothstep(0.08, 0.14, u)
        self.text(sc, 0.08, 0.43, "add the %d losses: %s" % (m["batch"], m["total_text"]), a,
                  SIZE, "lm")
        b = op * smoothstep(0.28, 0.34, u)
        self.big(sc, 0.08, 0.53, m["mean_text"], b, label="divide by %d" % m["batch"],
                 size=HEAD, ink=LOSS_INK)
        # the mean as a line across the bars, to the meter
        y = self.BASE - self.frac(0, self.bar_h(m["loss"]))[1]
        c = op * smoothstep(0.42, 0.46, u)
        self.arrow(sc, (self.BARS[0], y), (self.MX - 0.02, y), INK, c, eased(u, 0.42, 0.52),
                   width=1.8)

    # ------------------------------------------------------------------ 4: about ln 10
    def phase_4(self, sc, u, op):
        m = self.meta
        a = op * smoothstep(0.08, 0.14, u)
        self.text(sc, 0.08, 0.41, "a blind guess: 1 in 10 for each digit", a, SIZE, "lm")
        b = op * smoothstep(0.28, 0.34, u)
        self.text(sc, 0.08, 0.48, "−ln 0.1 = ln 10 = %s" % m["ln10_text"], b, HEAD, "lm",
                  strong=True)
        c = op * smoothstep(0.5, 0.56, u)
        self.text(sc, 0.08, 0.56, "this model: %s" % m["loss_text"], c, HEAD, "lm",
                  ink=LOSS_INK)
        # ln 10 marked on the meter
        y = self.BASE - self.frac(0, self.bar_h(m["ln10"]))[1]
        half = self.frac(self.MW / 2 + 6)[0]
        up = self.ARROW_LIFT                     # across the meter, in front of it
        self.polyline(sc, [self.P(self.MX - half, y, up), self.P(self.MX + half, y, up)], 2.4,
                      INK, b, bias=0.3)


# ====================================================================== 202
class Backward(Column):
    """Red tiles behind every weight, from the loss up; each .grad its weight's shape."""

    TITLE = "loss.backward()"
    PHASES = (("the loss", 10.0), ("the last layer", 13.0), ("back to the first", 14.0),
              ("the same shape", 16.0))
    LAST = "the loss: one number, 2.322"
    NEXT = "zero_grad(): empty the red tiles"
    STATION = ("backward",) * 4
    ROWS = (0.43, 0.585, 0.722)          # the three Linear blocks, first to last
    LOSS_Y = 0.85
    WX, GX = 0.205, 0.352                # weights' left edge; where each .grad slides to
    SIZES = ((110.0, 44.0), (64.0, 32.0), (48.0, 16.0))
    OFFSET = (7.0, 7.0)                  # red tiles: just behind, down and to the right
    # when loss.backward() reaches each layer, last first: (phase, u)
    REACH = ((2, (1, 0.2)), (1, (2, 0.12)), (0, (2, 0.6)))
    SLIDE = ((0, (3, 0.22)), (1, (3, 0.5)), (2, (3, 0.68)))

    def setup(self):
        m = self.meta = self.study.metadata
        self.layers = m["layers"]
        self.W = [np.asarray(p.data["weight"]) for p in self.study.panels[:3]]
        self.G = [np.asarray(p.data["values"]) for p in self.study.panels[:3]]
        self.Bw = [np.asarray(p.data["bias"]) for p in self.study.panels[:3]]
        self.Bg = [np.asarray(p.data["bias_grad"]) for p in self.study.panels[:3]]
        self.reach = {j: self.at(*v) for j, v in self.REACH}
        self.slide = {j: self.at(*v) for j, v in self.SLIDE}
        call = list(code_lines("step")).index(m["call"])
        self.CODE = (Code("step", run=((0.0, None), (0.3, call))),
                     Code("step", run=((0.0, call),)),
                     Code("step", run=((0.0, call),)),
                     Code("grad", run=((0.0, None), (0.08, 0)), out=((0.16, m["printed"]),)))
        self.EVIDENCE = (measured("the first batch, untrained"),) + (
            measured("loss.backward() on the first batch"),) * 2 + (
            exact(".grad has its weight's shape"),)

    def heading_for(self, k, u):
        return ["loss.backward(): start at the loss",
                "the last layer gets its red tiles first",
                _pick(((0.0, "then the layer before it"), (0.48, "and back to the first")), u),
                "each .grad: the same shape as its weight"][k]

    def subtitle_for(self, k):
        m = self.meta
        L = self.layers
        return [
            "The loss of the first batch is %s. loss.backward() starts from it and works back."
            % m["loss_text"],
            "First the last layer: a gradient for each of its %s numbers, weight %s and bias %s."
            % (L[2]["numbers"], L[2]["shape_text"], L[2]["bias_text"]),
            "Then %s and %s: %s gradients in all, one for every number." % (
                L[1]["shape_text"], L[0]["shape_text"], m["total_text"]),
            "Each .grad has its weight's shape: model[1].weight.grad is %s."
            % L[0]["shape_text"],
        ][k]

    # ------------------------------------------------------------------ geometry
    def wcentre(self, j):
        w, _h = self.SIZES[j]
        return self.WX + self.frac(w / 2)[0], self.ROWS[j]

    def grad_at(self, j):
        """Where layer ``j``'s red tiles are: behind the weight, or sliding out to beside
        it."""
        wx, wy = self.wcentre(j)
        dx, dy = self.frac(*self.OFFSET)
        w, _h = self.SIZES[j]
        gx = self.GX + self.frac(w / 2)[0]
        s = ease_in_out(float(np.clip((getattr(self, "_t", 0.0) - self.slide[j]) / 1.4, 0, 1)))
        return wx + dx + (gx - wx - dx) * s, wy + dy * (1 - s), s

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        t = getattr(self, "_t", 0.0)
        for j, L in enumerate(self.layers):
            y = self.ROWS[j]
            w, h = self.SIZES[j]
            self.block_box(sc, "linear", L["name"], y, op)
            wx, wy = self.wcentre(j)
            # its gradient, once loss.backward() has reached it
            g = op * self.since(self.reach[j] + 0.5, 0.8)
            gx, gy, s = self.grad_at(j)
            if g > 0.01:
                self.block(sc, self.G[j], gx, gy, w, h, g, ink=GRAD, lift=0.15)
                self.rect(sc, self.P(gx, gy), w + 3, h + 3, GRAD, g, border=GRAD, border_w=1.4,
                          radius=2, lift=0.14, fill_op=0.0)
                bx = self.WX + self.frac(w + 8.5)[0]
                dx, dy = self.frac(*self.OFFSET)
                self.block(sc, self.Bg[j][:, None], bx + dx, y + dy, 7.0, h, g, ink=GRAD,
                           lift=0.15)
            # the weight and its bias, in front
            self.block(sc, self.W[j], wx, wy, w, h, op, ink=WEIGHT, lift=0.3)
            self.rect(sc, self.P(wx, wy), w + 3, h + 3, WEIGHT, op, border=WEIGHT, border_w=1.4,
                      radius=2, lift=0.31, fill_op=0.0)
            self.block(sc, self.Bw[j][:, None], self.WX + self.frac(w + 8.5)[0], y, 7.0, h, op,
                       ink=WEIGHT, lift=0.3)
            under = y + self.frac(0, h / 2 + 16)[1]
            self.text(sc, self.WX, under, L["shape_text"], op, SIZE, "lm", ink=WEIGHT_INK)
            self.text(sc, self.GX, under, ".grad " + L["shape_text"],
                      op * self.since(self.reach[j] + 1.4), SIZE, "lm", ink=GRAD_INK)
            # once its .grad is out beside it: the same shape
            self.text(sc, self.GX - 0.03, under, "=", op * self.since(self.slide[j] + 1.4), SIZE,
                      strong=True)
            # the red arrow up to it
            below = self.LOSS_Y - 0.03 if j == 2 else self.ROWS[j + 1] - self.frac(
                0, BH / 2 + 2)[1]
            self.arrow(sc, (BX, below), (BX, y + self.frac(0, BH / 2 + 3)[1]), GRAD, op,
                       u=eased(t, self.reach[j] - 0.2, self.reach[j] + 0.5), width=2.2)
        # the loss
        self.meter(sc, 0.08, self.LOSS_Y, m["loss"] / 3.0, op, h=50.0, w=20.0)
        self.big(sc, 0.134, self.LOSS_Y + 0.008, m["loss_text"], op, label="loss", size=HEAD,
                 ink=LOSS_INK)
        # how many numbers have a gradient so far
        n = sum(t >= self.reach[j] + 0.5 for j in self.reach)
        steps = [(0.0, "0")] + [(1.0, m["running_text"][i]) for i in range(n)]
        self.running(sc, 0.27, self.LOSS_Y + 0.008, steps[-1:], 1.0,
                     op * self.since(self.reach[2] + 0.5), label="numbers with a .grad",
                     size=HEAD, ink=GRAD_INK)
        # the ring round the block under way
        lit = [j for j in self.reach if t >= self.reach[j]]
        if t >= self.at(0, 0.35) and k < 3:
            if lit:
                j = min(lit)
                y0 = self.LOSS_Y if j == 2 else self.ROWS[j + 1]
                s = ease_in_out(float(np.clip((t - self.reach[j]) / 0.6, 0, 1)))
                y = y0 + (self.ROWS[j] - y0) * s
            else:
                y = self.LOSS_Y
            self.ring(sc, BX, y, BW + 12, BH + 14 if y != self.LOSS_Y else 62.0,
                      op * self.since(self.at(0, 0.35), 0.5) * (1 - self.since(self.at(2, 0.98),
                                                                              0.5)))

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        self.text(sc, 0.205, self.LOSS_Y, "starts here", op * smoothstep(0.4, 0.46, u), SIZE,
                  "lm", strong=True)

    def phase_2(self, sc, u, op):
        pass

    def phase_3(self, sc, u, op):
        pass

    def phase_4(self, sc, u, op):
        pass


ANIMATIONS = {"blocks-forward-pass": ForwardPass, "blocks-the-loss": TheLoss,
              "blocks-backward": Backward}

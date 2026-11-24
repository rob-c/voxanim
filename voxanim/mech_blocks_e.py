"""Animations 197-199 of the "Blocks and code" series (see :mod:`voxanim.blocks`): logits
and softmax, a recap of every block so far, and the data pipeline, drawn from
:mod:`voxanim.study_blocks_e`.

* :class:`LogitsSoftmax` (``blocks-logits-and-softmax``): one held-out digit through the
  trained model: its 10 logits as bars either side of 0; softmax turns them into positive
  bars that add up to 1 (added up one at a time); ``nn.CrossEntropyLoss`` takes the
  logits and gives ``-ln`` of the right digit's share on the loss meter; and with a softmax
  in front of it the bars go flat and the meter cannot fall below 1.46.
* :class:`WhichIsWhich` (``blocks-which-is-which``): the eleven blocks of 185-197, two or
  three at a time: each block's call, its input shape and a "?" for its output shape and
  its weights; a pause to guess; then the answers one block at a time, with a tally of
  the blocks seen and the ones that learn kept along the bottom.
* :class:`DataPipeline` (``blocks-the-data-pipeline``): the four stations MNIST, ToTensor,
  Normalize and DataLoader along the top, the one under way lit; the training set; one 3 x 3
  window of the first digit as 0..255, then ÷ 255, then minus the mean and over the spread
  of every training pixel; the shuffled order of two epochs; and batches of 64 riding a
  conveyor belt into the model, 938 of them, the last of 32.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). What stays across phases is drawn in
``steady``, so nothing blanks between them.
"""

from __future__ import annotations

import math

import numpy as np

from . import layout
from . import study_course as course
from .blocks import (
    BIG,
    DATA,
    DATA_HI,
    DATA_LO,
    HEAD,
    INK,
    LOSS,
    QUIET,
    SIZE,
    WEIGHT,
    ZERO,
    Blocks,
    Code,
    commas,
    exact,
    measured,
    shades,
    val,
)
from .easing import ease_in_out, smoothstep
from .mech import AXIS, GRID, PAPER, TRACK, dark, soft

DATA_INK = dark(DATA, 0.3)
WEIGHT_INK = dark(WEIGHT, 0.3)
LOSS_INK = dark(LOSS, 0.3)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def mix(a, b, s):
    return a + (b - a) * s


# ====================================================================== 197
class LogitsSoftmax(Blocks):
    """The 10 logits; softmax; the bars add up to 1; the loss takes the logits; softmax
    twice."""

    TITLE = "Logits and softmax"
    PHASES = (("10 scores", 11.0), ("softmax", 11.0), ("add up to 1", 12.0), ("the loss", 12.0),
              ("softmax twice", 13.0))
    LAST = "attention: Q, K, V and a grid of scores"
    NEXT = "every block, side by side"
    STATION = ("forward",) * 3 + ("loss",) * 2
    CARD = (0.115, 0.468)
    TILE = 3.4
    ZX = 0.310                  # the bars' zero line
    LABEL_X = 0.235             # the digits 0-9
    VALUE_X = 0.475             # the values, right-aligned
    ROW0, STEP = 0.415, 0.048
    THICK = 15.0
    LOGIT_PX = 90.0             # the biggest logit's bar, px
    PROB_PX = 104.0             # a share of 1, px
    SLOT = (0.058, 0.775)       # the number under the card: 10, then the sum
    LOSS_SLOT = (0.105, 0.765)  # the loss, beside its meter
    METER = (0.075, 0.765)
    METER_H = 104.0

    def setup(self):
        m = self.meta = self.study.metadata
        self.img = np.asarray(self.study.panels[3].data["values"][0], float)
        self.logits = np.asarray(m["logits"])
        self.probs = np.asarray(m["probs"])
        self.twice = np.asarray(m["twice"])
        self.y = m["label"]
        self.cell = None
        # phase 3: the shares added in order of size, the biggest four one at a time
        self.adds = m["order"]
        self.add_at = [0.08, 0.25, 0.42, 0.59, 0.72]
        self.scale = self.LOGIT_PX / float(np.abs(self.logits).max())
        mistake = list(m["mistake"])
        soft_line = next(i for i, s in enumerate(mistake) if "Softmax" in s)
        self.CODE = (Code("model", run=((0.0, 0), (0.3, 6))),
                     Code("softmax", pick=(1,), run=((0.0, 0),)),
                     Code("softmax", pick=(1,), run=((0.0, 0),)),
                     Code(("loss", "step"), pick=(0, 2, 3),
                          run=((0.0, 0), (0.25, 1), (0.45, 2))),
                     Code("mistake-double-softmax", run=((0.0, 0), (0.12, soft_line))))
        self.EVIDENCE = (exact("the trained model's 10 logits"),
                         exact("softmax of the 10 logits"),
                         exact("the 10 shares added up"),
                         exact("−ln of the right digit's share"),
                         measured("the run trained with softmax twice"))

    def heading_for(self, k, u):
        return ("the last block gives 10 numbers: the logits",
                "softmax: every bar positive",
                "the bars add up to 1",
                "the loss takes the logits",
                "softmax twice: the bars go flat")[k]

    def subtitle_for(self, k):
        m = self.meta
        L, p = m["logits"], m["probs"]
        y, s = m["label"], m["second"]
        return [
            "For this test digit, a %d, the trained model's highest score is for %d: %s. "
            "Next comes %d: %s." % (y, m["prediction"], val(L[y]), s, val(L[s])),
            "Softmax makes every score positive and shares out 1: %s for %d, %s for %d."
            % (val(p[y]), y, val(p[s]), s),
            "Added up, the ten bars make %s: %s and the rest."
            % (val(sum(p)), " + ".join(val(p[d]) for d in m["order"][:4])),
            "CrossEntropyLoss does the softmax itself, then takes −ln %s = %s."
            % (val(p[y]), val(m["loss"])),
            "A softmax in front squashes the bars: the loss can't go below %s, and only "
            "%d%% of test digits come out right." % (val(m["floor"]),
                                                    round(100 * m["accuracy_twice"])),
        ][k]

    # ------------------------------------------------------------------ the bars
    def row_y(self, d):
        return self.ROW0 + self.STEP * d

    def stage(self, k, u):
        """``(lengths px, values text)`` of the bars at ``(k, u)``."""
        logit_px = self.logits * self.scale
        prob_px = self.probs * self.PROB_PX
        twice_px = self.twice * self.PROB_PX
        if k == 0:
            g = eased(u, 0.06, 0.24)
            return logit_px * g, [val(v) for v in self.logits]
        if k == 1:
            s = eased(u, 0.2, 0.45)
            text = [val(v) for v in (self.logits if s < 0.5 else self.probs)]
            return mix(logit_px, prob_px, s), text
        if k < 4:
            return prob_px, [val(v) for v in self.probs]
        s = eased(u, 0.15, 0.4)
        return mix(prob_px, twice_px, s), [val(v) for v in (self.probs if s < 0.5
                                                            else self.twice)]

    def draw_bars(self, sc, k, u, op):
        lengths, text = self.stage(k, u)
        names = ("logits", "logits" if u < 0.325 else "softmax", "softmax", "softmax",
                 "softmax" if u < 0.275 else "softmax twice")
        self.text(sc, self.VALUE_X, self.row_y(0) - 0.052, names[k], op, SIZE, "rm",
                  ink=DATA_INK)
        top, bottom = self.row_y(0) - 0.03, self.row_y(9) + 0.03
        self.polyline(sc, [self.P(self.ZX, top), self.P(self.ZX, bottom)], 1.6, AXIS, op,
                      bias=0.2)
        self.text(sc, self.ZX, top - 0.022, "0", op, SIZE, ink=QUIET)
        one = op * (smoothstep(0.3, 0.45, u) if k == 1 else float(k > 1))
        if one > 0.01:            # a share of 1 reaches here
            x1 = self.ZX + self.frac(self.PROB_PX)[0]
            self.polyline(sc, [self.P(x1, top), self.P(x1, bottom)], 1.0, GRID, one, bias=0.1)
        for d in range(10):
            y = self.row_y(d)
            right = d == self.y
            self.text(sc, self.LABEL_X, y, str(d), op, SIZE, strong=right)
            w = float(lengths[d])
            if abs(w) > 0.3:
                x = self.ZX + self.frac(w / 2)[0]
                self.rect(sc, self.P(x, y), abs(w), self.THICK, DATA, op, radius=3, lift=0.3)
            self.text(sc, self.VALUE_X, y, text[d], op, SIZE, "rm",
                      ink=INK if right else None)

    def ring_row(self, sc, d, op):
        """An outline round row ``d`` (its digit, its bar and its value)."""
        x0, x1 = self.LABEL_X - 0.016, self.VALUE_X + 0.006
        w = self.span(x1 - x0)[0]
        self.rect(sc, self.P((x0 + x1) / 2, self.row_y(d)), w, 24.0, INK, op, border=INK,
                  border_w=2.2, radius=5, lift=1.0, fill_op=0.0)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        cx, cy = self.CARD
        self.digit_card(sc, self.img, cx, cy, self.TILE, op, lo=0.0, hi=1.0)
        self.text(sc, cx, 0.615, "label: %d" % self.y, op, SIZE, ink=DATA_INK)
        self.draw_bars(sc, k, u, op)
        if k >= 3:              # the loss meter, from the loss on
            self.loss_meter(sc, k, u, op * self.kept(3))

    def loss_meter(self, sc, k, u, op):
        """The loss meter (full: ln 10, the loss of a model that knows nothing), and the
        loss as ``-ln`` of the right digit's share beside it."""
        m = self.meta
        full = math.log(10)
        s = eased(u, 0.45, 0.62) if k == 4 else 0.0
        loss = mix(m["loss"], m["loss_twice"], s)
        grow = 1.0 if k == 4 else eased(u, 0.5, 0.65)
        mx, my = self.METER
        self.meter(sc, mx, my, loss / full * grow, op, h=self.METER_H, w=22.0)
        self.text(sc, mx, my + self.frac(0, self.METER_H / 2 + 14)[1], "loss", op, SIZE,
                  ink=LOSS_INK)
        sx, sy = self.LOSS_SLOT
        twice = k == 4 and u >= 0.535
        share = m["twice_right"] if twice else m["prob_right"]
        text = val(m["loss_twice"] if twice else m["loss"]) if (k == 4 or u >= 0.5) else "?"
        on = op if k == 4 else op * smoothstep(0.4, 0.47, u)
        self.running(sc, sx, sy, [(0.0, text)], 0.0, on, label="−ln %s =" % val(share),
                     ink=LOSS_INK, size=BIG)

    # ------------------------------------------------------------------ 1: the logits
    def phase_1(self, sc, u, op):
        on = op * smoothstep(0.45, 0.52, u)
        self.ring_row(sc, self.meta["prediction"], on)
        self.big(sc, self.SLOT[0], self.SLOT[1], "10", op * smoothstep(0.25, 0.32, u),
                 label="numbers", size=BIG)

    # ------------------------------------------------------------------ 2: softmax
    def phase_2(self, sc, u, op):
        self.ring_row(sc, self.y, op * smoothstep(0.55, 0.62, u))
        self.text(sc, self.SLOT[0], self.SLOT[1] - 0.06, "all positive", op * smoothstep(
            0.6, 0.67, u), SIZE, "lm", strong=True)

    # ------------------------------------------------------------------ 3: add up to 1
    def phase_3(self, sc, u, op):
        p = self.probs
        steps, total = [(0.0, "0.00")], 0.0
        for i, at in enumerate(self.add_at):
            if i < 4:
                total += p[self.adds[i]]
            else:
                total = float(np.sum(p))
            steps.append((at, val(total)))
        # the bar being added, outlined
        now = sum(u >= a for a in self.add_at) - 1
        if 0 <= now < 4:
            a = self.add_at[now]
            b = self.add_at[now + 1]
            self.ring_row(sc, self.adds[now], op * smoothstep(a, a + 0.04, u) * (
                1 - smoothstep(b - 0.04, b, u)))
        self.running(sc, self.SLOT[0], self.SLOT[1], steps, u, op, label="added up",
                     size=BIG)
        self.text(sc, self.SLOT[0], self.SLOT[1] + 0.06, "= 1" if u >= self.add_at[-1] else "",
                  op * smoothstep(self.add_at[-1], self.add_at[-1] + 0.04, u), SIZE, "lm",
                  strong=True)

    # ------------------------------------------------------------------ 4: the loss
    def phase_4(self, sc, u, op):
        self.ring_row(sc, self.y, op * smoothstep(0.3, 0.37, u))

    # ------------------------------------------------------------------ 5: softmax twice
    def phase_5(self, sc, u, op):
        m = self.meta
        # the floor: the meter cannot go below it
        f = op * smoothstep(0.66, 0.72, u)
        mx, my = self.METER
        fy = my + self.frac(0, self.METER_H / 2 - 3 - (self.METER_H - 6) * m["floor"]
                            / math.log(10))[1]
        x0 = mx - self.frac(16)[0]
        x1 = mx + self.frac(16)[0]
        up = self.ARROW_LIFT                     # across the meter, in front of it
        self.polyline(sc, [self.P(x0, fy, up), self.P(x1, fy, up)], 2.4, INK, f, bias=0.3)
        self.text(sc, self.LOSS_SLOT[0], 0.835, "at least %s" % val(m["floor"]), f, SIZE, "lm",
                  strong=True)
        r = op * smoothstep(0.82, 0.88, u)
        self.text(sc, self.LABEL_X - 0.008, 0.888, "%d%% right, not %d%%" % (
            round(100 * m["accuracy_twice"]), round(100 * m["accuracy"])), r, SIZE, "lm",
            ink=INK)


# ====================================================================== 198
class WhichIsWhich(Blocks):
    """Eleven blocks, two or three at a time: a guess, then the answers."""

    TITLE = "Which block is which"
    PHASES = (("the model's", 14.0), ("pictures", 14.0), ("norms, table", 14.0),
              ("attention", 13.0))
    LAST = "logits and softmax"
    NEXT = "the data pipeline"
    STATION = ("forward",) * 4
    EVIDENCE = (exact("shapes and counts from the settings"),) * 4
    ROWS = (0.43, 0.565, 0.70)
    ICON_X, TEXT_X, COUNT_X = 0.08, 0.115, 0.475
    TALLY_Y = 0.862
    APPEAR = (0.03, 0.12, 0.21)              # each row appears
    REVEAL = (0.44, 0.6, 0.76)               # each row's answers
    REVEAL_LAST = (0.44, 0.58)               # phase 4: two rows
    END = 0.72                               # phase 4: the closing count
    GROUP_NAMES = ("the model's blocks", "blocks for pictures", "norms and a table",
                   "attention and softmax")

    def setup(self):
        m = self.meta = self.study.metadata
        self.blocks = m["blocks"]
        self.groups = m["groups"]
        # each block's line in its phase's code (the indices into Code.lines())
        self.CODE = (Code("model", pick=(1, 2, 3)),
                     Code(("conv", "pool", "dropout"), pick=(0, 1, 2, 3, 4, 6)),
                     Code(("batchnorm", "layernorm", "embedding")),
                     Code(("attention", "softmax")))
        runs = []
        for k, code in enumerate(self.CODE):
            lines = code.lines()
            at = [next(i for i, s in enumerate(lines) if s.strip() == self.blocks[b]["line"])
                  for b in self.groups[k]]
            appear, reveal = self.APPEAR, self.reveals(k)
            run = [(appear[i], at[i]) for i in range(len(at))]
            run += [(reveal[i], at[i]) for i in range(len(at))]
            runs.append(tuple(run))
        self.CODE = tuple(Code(c.region, c.pick, run=r) for c, r in zip(self.CODE, runs))

    def reveals(self, k):
        return self.REVEAL_LAST if k == len(self.PHASES) - 1 else self.REVEAL

    def heading_for(self, k, u):
        if u < self.reveals(k)[0]:
            return "guess: what comes out? how many weights?"
        return "answers: " + self.GROUP_NAMES[k]

    def subtitle_for(self, k):
        b = {x["name"]: x for x in self.blocks}
        m = self.meta
        return [
            "Flatten and ReLU learn nothing. nn.Linear(784, 64) learns %s = %s numbers."
            % (b["Linear"]["how"], commas(b["Linear"]["weights"])),
            "Conv2d(1, 4, 3) learns %s numbers and gives %s; pooling halves each side."
            % (b["Conv2d"]["weights"], b["Conv2d"]["shape_out_text"]),
            "Each norm learns %s = %s numbers; the embedding is a table of %s = %s."
            % (b["LayerNorm"]["how"], b["LayerNorm"]["weights"], b["Embedding"]["how"],
               b["Embedding"]["weights"]),
            "Attention learns %s numbers; softmax learns none. %d of the %d blocks learn."
            % (commas(b["MultiheadAttention"]["weights"]), m["learning"], m["total"]),
        ][k]

    # ------------------------------------------------------------------ one block
    def call_text(self, b):
        """The block's call as the picture names it: the code's own call, its keyword
        settings left out when it is too wide for the picture."""
        s = b["call"]
        room = self.COUNT_X - 0.07 - self.TEXT_X
        if layout.text_width(s, SIZE, "mono") > room:
            s = "%s(%s)" % (s.split("(")[0], ", ".join(str(a) for a in b["args"]))
        return s

    def row(self, sc, b, y, op, appear, reveal):
        """Block ``b`` as a row at ``y``: an icon, its call, its shapes and its weights;
        the output shape and the weights are "?" until ``reveal`` (0..1) passes 0.5."""
        if op <= 0.01:
            return
        on = op * appear
        w = self.span(self.COUNT_X + 0.008 - (self.ICON_X - 0.03))[0]
        cx = (self.COUNT_X + 0.008 + self.ICON_X - 0.03) / 2
        shown = reveal >= 0.5
        learns = b["learns"]
        border = (WEIGHT if learns else ZERO) if shown else GRID
        self.rect(sc, self.P(cx, y), w, 66.0, PAPER, on, border=border,
                  border_w=1.4 + 0.8 * shown, radius=8, lift=0.1)
        # the icon
        ix = self.ICON_X
        if shown and learns:
            pattern = np.array([[0.9, 0.4, 0.7], [0.3, 1.0, 0.5], [0.6, 0.2, 0.8]])
            self.tiles(sc, pattern, ix, y, 27.0, 27.0, on, ink=WEIGHT, lo=0.0, hi=1.0,
                       grid=0.4, lift=0.3)
        elif shown:
            self.rect(sc, self.P(ix, y), 27.0, 27.0, soft(ZERO, 0.6), on, border=ZERO,
                      border_w=1.4, radius=3, lift=0.3)
        else:
            self.box(sc, ix, y, 27.0, 27.0, TRACK, on, text="?", border=GRID, size=SIZE,
                     radius=3)
        self.label(sc, self.P(self.TEXT_X, y - 0.04), self.call_text(b), SIZE, anchor="lm",
                   opacity=on, ink=INK, kind="mono")
        out = b["shape_out_text"] if shown else "?"
        self.text(sc, self.TEXT_X, y, "%s → %s" % (b["shape_in_text"], out), on, SIZE, "lm",
                  ink=DATA_INK)
        how = ("weights: " + b["how"] + " =") if learns else "weights: none"
        self.text(sc, self.TEXT_X, y + 0.04, how if shown else "weights:", on, SIZE, "lm",
                  ink=WEIGHT_INK if shown and learns else None)
        self.text(sc, self.COUNT_X - 0.004, y + 0.012, commas(b["weights"]) if shown else "?",
                  on, HEAD, "rm", ink=WEIGHT_INK if shown and learns else INK)

    # ------------------------------------------------------------------ the tally
    def seen(self, k, u):
        """How many blocks have been answered by ``(k, u)``."""
        done = sum(len(g) for g in self.groups[:k])
        return done + sum(u >= r for r in self.reveals(k)[:len(self.groups[k])])

    def steady(self, sc, k, u, op):
        n = self.seen(k, u)
        size = 22.0
        x0 = 0.062
        step = self.frac(size + 5.0)[0]
        for i, b in enumerate(self.blocks):
            x = x0 + step * i
            if i < n and b["learns"]:
                pattern = np.array([[0.9, 0.4], [0.5, 1.0]])
                self.tiles(sc, pattern, x, self.TALLY_Y, size, size, op, ink=WEIGHT, lo=0.0,
                           hi=1.0, grid=0.4, lift=0.3)
            elif i < n:
                self.rect(sc, self.P(x, self.TALLY_Y), size, size, soft(ZERO, 0.6), op,
                          border=ZERO, border_w=1.2, radius=3, lift=0.3)
            else:
                self.rect(sc, self.P(x, self.TALLY_Y), size, size, PAPER, op, border=GRID,
                          border_w=1.2, radius=3, lift=0.3)
        learn = sum(b["learns"] for b in self.blocks[:n])
        tx = x0 + step * len(self.blocks) + self.frac(4.0)[0]
        self.text(sc, tx, self.TALLY_Y, "%d of %d learn" % (learn, n) if n else "", op, SIZE,
                  "lm", strong=True)

    def group(self, sc, k, u, op):
        reveal = self.reveals(k)
        for i, b in enumerate(self.groups[k]):
            a = smoothstep(self.APPEAR[i], self.APPEAR[i] + 0.04, u)
            r = smoothstep(reveal[i], reveal[i] + 0.02, u)
            self.row(sc, self.blocks[b], self.ROWS[i], op, a, r)

    def phase_1(self, sc, u, op):
        self.group(sc, 0, u, op)

    def phase_2(self, sc, u, op):
        self.group(sc, 1, u, op)

    def phase_3(self, sc, u, op):
        self.group(sc, 2, u, op)

    def phase_4(self, sc, u, op):
        self.group(sc, 3, u, op)
        m = self.meta
        on = op * smoothstep(self.END, self.END + 0.05, u)
        self.big(sc, self.TEXT_X, self.ROWS[2] + 0.02, "%d of %d" % (m["learning"], m["total"]),
                 on, label="blocks that learn weights", size=BIG)


# ====================================================================== 199
class DataPipeline(Blocks):
    """MNIST; ToTensor; Normalize; a shuffled order; batches on a belt."""

    TITLE = "The data pipeline"
    PHASES = (("the dataset", 11.0), ("ToTensor", 11.0), ("Normalize", 13.0),
              ("shuffle", 11.0), ("batches", 13.0))
    LAST = "every block, side by side"
    NEXT = "the forward pass"
    STATION = ("data",) * 5
    STRIP_Y = 0.395
    STAGES = ("MNIST", "ToTensor", "Normalize", "DataLoader")
    LIT = (0, 1, 2, 3, 3)
    CARD = (0.12, 0.655)
    TILE = 3.6
    WIN = (0.268, 0.62)          # the window's magnified boxes: centre
    BOX_W, BOX_H = 44.0, 30.0
    RIGHT = 0.355                # the right-hand column of numbers
    SHOWN = 6                    # thumbnails per epoch row
    BELT_Y = 0.80
    BELT = (0.065, 0.375)
    MODEL_X = 0.43
    SPACING = 0.078
    MOVE = (0.06, 0.7)           # phase 5: the belt runs, at one speed
    TRAVEL = 0.36                # how far it runs (layout x)

    def setup(self):
        m = self.meta = self.study.metadata
        self.img = np.asarray(self.study.panels[0].data["values"][0], float)
        self.raw = np.asarray(m["raw"])
        self.tensor = np.asarray(m["tensor"])
        self.norm = np.asarray(m["normalised"])
        self.cell = self.digit_cells(*self.CARD, self.TILE)
        px = course.pixels()[0]
        self.thumbs = [px[m["first"][:self.SHOWN]].reshape(-1, 28, 28),
                       px[m["again"][:self.SHOWN]].reshape(-1, 28, 28)]
        # each batch on the belt shows its first four digits, two by two
        self.belt = [np.block([[px[a].reshape(28, 28), px[b].reshape(28, 28)],
                               [px[c].reshape(28, 28), px[d].reshape(28, 28)]])
                     for a, b, c, d in m["belt_digits"]]
        lines = list(m["data"])
        at = {name: next(i for i, s in enumerate(lines) if name in s)
              for name in ("ToTensor", "Normalize", "train_set", "test_set")}
        self.CODE = (Code("data", run=((0.0, at["train_set"]), (0.5, at["test_set"]))),
                     Code("data", run=((0.0, at["ToTensor"]),)),
                     Code("data", run=((0.0, at["Normalize"]),)),
                     Code(("loader", "train"), pick=(0, 1, 2, 4, 5),
                          run=((0.0, 0), (0.3, 2))),
                     Code(("loader", "train"), pick=(0, 1, 2, 4, 5),
                          run=((0.0, 1), (0.1, 4))))
        self.EVIDENCE = (exact("the training and test digits counted"),
                         exact("train_set[0]'s pixels ÷ 255"),
                         exact("mean and spread of every pixel"),
                         exact("the shuffled order (seeded)"),
                         exact("60,000 digits in batches of 64"))
        # phase 5: the belt's items (three batches, a gap, the last) and when each enters
        self.items = [(str(b + 1), s, img) for b, s, img in zip(m["belt"][:-1],
                                                                 m["belt_sizes"][:-1],
                                                                 self.belt[:-1])]
        self.items += [("…", 0, None), (str(m["belt"][-1] + 1), m["belt_sizes"][-1],
                                        self.belt[-1])]

    def heading_for(self, k, u):
        return ("the dataset: digits and their labels",
                "ToTensor: 0..255 becomes 0..1",
                "Normalize: minus the mean, over the spread",
                "shuffle=True: a new order every epoch",
                "the loader: batches of 64 to the model")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "MNIST gives %s training digits and %s test digits, each with its label."
            % (commas(m["train"]), commas(m["test"])),
            "ToTensor divides every pixel by 255: %d becomes %.3f, paper stays 0."
            % (m["pixel_raw"], m["pixel_tensor"]),
            "Normalize subtracts %.4f and divides by %.4f: the mean and spread of all %s "
            "pixels." % (m["script_mean"], m["script_std"], commas(m["pixels"])),
            "shuffle=True mixes the %s digits again in every epoch: a new order each time."
            % commas(m["train"]),
            "The loader hands out %d batches: %s. That is one epoch."
            % (m["steps"], m["epoch_sum"]),
        ][k]

    # ------------------------------------------------------------------ the stations
    def stations(self, sc, k, u, op):
        n = len(self.STAGES)
        x0, x1 = 0.058, 0.475
        ws = [self.frac(layout.text_width(s, SIZE) * self.cam.width + 14.0)[0]
              for s in self.STAGES]
        gap = (x1 - x0 - sum(ws)) / (n - 1)
        lit = self.LIT[k]
        x = x0
        for i, (name, w) in enumerate(zip(self.STAGES, ws)):
            on = i == lit
            fill = soft(DATA, 0.55) if on else PAPER
            self.box(sc, x + w / 2, self.STRIP_Y, self.span(w)[0], 28.0, fill, op, text=name,
                     border=INK if on else GRID, size=SIZE, ink=INK if on else None,
                     radius=6)
            if i < n - 1:
                a = x + w + 0.004
                self.arrow(sc, (a, self.STRIP_Y), (a + gap - 0.008, self.STRIP_Y), AXIS, op,
                           width=1.5)
            x += w + gap

    def window_values(self, k, u):
        """The window's numbers at ``(k, u)``: 0..255, then ÷ 255, then normalised."""
        if k == 1 and u < 0.45:
            return [str(int(v)) for v in self.raw.ravel()], self.raw / 255.0
        if k == 1 or (k == 2 and u < 0.5):
            return ["%.2f" % v for v in self.tensor.ravel()], self.tensor
        return [val(v) for v in self.norm.ravel()], (self.norm - DATA_LO) / (DATA_HI - DATA_LO)

    def steady(self, sc, k, u, op):
        self.stations(sc, k, u, op)
        card = op * self.kept(0, 2)
        cx, cy = self.CARD
        self.digit_card(sc, self.img, cx, cy, self.TILE, card, lo=0.0, hi=1.0)
        self.text(sc, cx, cy + 0.14, "label: %d" % self.meta["label"], card, SIZE,
                  ink=DATA_INK)
        if k in (1, 2):
            self.window(sc, k, u, op * self.kept(1, 2))

    def window(self, sc, k, u, op):
        r0, c0 = self.meta["zoom"]
        n = len(self.raw)
        ring = op if k == 2 else op * smoothstep(0.04, 0.1, u)
        _x0, y0, x1, y1 = self.outline(sc, self.cell, r0, c0, n, n, self.TILE, ring)
        mx, my = self.WIN
        dw, dh = self.frac(n * self.BOX_W, n * self.BOX_H)
        grow = op if k == 2 else op * smoothstep(0.1, 0.18, u)
        for a, b, c in ((x1, y0, my - dh / 2), (x1, y1, my + dh / 2)):
            self.arrow(sc, (a, b), (mx - dw / 2, c), AXIS, grow * 0.8, head=False, width=1.3)
        text, shade = self.window_values(k, u)
        fills = shades(np.clip(shade, 0, 1), DATA, 0.0, 1.0)
        pi, pj = self.meta["pixel"]
        for i in range(n):
            for j in range(n):
                bx = mx - dw / 2 + (j + 0.5) * dw / n
                by = my - dh / 2 + (i + 0.5) * dh / n
                s = float(np.clip(shade[i, j], 0, 1))
                fill = tuple(float(c) for c in fills[i, j])
                self.box(sc, bx, by, self.BOX_W - 3, self.BOX_H - 3, fill, grow, border=DATA,
                         radius=3, size=SIZE, text=text[i * n + j],
                         ink=INK if s < 0.5 else PAPER)
                if (i, j) == (pi, pj) and k == 2:
                    self.rect(sc, self.P(bx, by), self.BOX_W + 2, self.BOX_H + 2, INK,
                              grow * smoothstep(0.62, 0.68, u), border=INK, border_w=2.4,
                              radius=4, lift=1.2, fill_op=0.0)

    # ------------------------------------------------------------------ 1: the dataset
    def phase_1(self, sc, u, op):
        m = self.meta
        a = op * smoothstep(0.15, 0.22, u)
        b = op * smoothstep(0.5, 0.57, u)
        self.big(sc, 0.25, 0.6, commas(m["train"]), a, label="train_set: to learn from",
                 size=BIG, ink=DATA_INK)
        self.big(sc, 0.25, 0.77, commas(m["test"]), b, label="test_set: for testing",
                 size=BIG, ink=DATA_INK)

    # ------------------------------------------------------------------ 2: ToTensor
    def phase_2(self, sc, u, op):
        mx, my = self.WIN
        dh = self.frac(0, 3 * self.BOX_H)[1]
        on = op * smoothstep(0.3, 0.37, u)
        self.text(sc, self.RIGHT + 0.01, my, "÷ 255", on, HEAD, "lm", strong=True)
        self.text(sc, mx, my + dh / 2 + 0.04, "0..255" if u < 0.45 else "0..1",
                  op * smoothstep(0.12, 0.18, u) * (1 - smoothstep(0.41, 0.45, u)
                                                    + smoothstep(0.46, 0.5, u)), SIZE,
                  ink=DATA_INK)

    # ------------------------------------------------------------------ 3: Normalize
    def phase_3(self, sc, u, op):
        m = self.meta
        a = op * smoothstep(0.06, 0.13, u)
        self.text(sc, self.RIGHT + 0.01, 0.47, "every pixel:", a, SIZE, "lm")
        self.text(sc, self.RIGHT + 0.01, 0.53, "mean %.4f" % m["mean"], a, SIZE, "lm",
                  strong=True)
        b = op * smoothstep(0.22, 0.29, u)
        self.text(sc, self.RIGHT + 0.01, 0.58, "spread %.4f" % m["std"], b, SIZE, "lm",
                  strong=True)
        _mx, my = self.WIN
        dh = self.frac(0, 3 * self.BOX_H)[1]
        c = op * smoothstep(0.66, 0.72, u)
        self.text(sc, 0.2, my + dh / 2 + 0.05, m["pixel_sum"], c, SIZE, "lm")
        self.text(sc, 0.2, my + dh / 2 + 0.1, "= %s" % val(m["pixel_norm"]), c, HEAD, "lm",
                  strong=True)

    # ------------------------------------------------------------------ 4: shuffle
    def phase_4(self, sc, u, op):
        m = self.meta
        size = 38.0
        x0 = 0.15
        step = self.frac(size + 14)[0]
        for e, (y, at) in enumerate(((0.52, 0.08), (0.73, 0.5))):
            on = op * smoothstep(at, at + 0.06, u)
            self.text(sc, 0.06, y, "epoch %d" % (e + 1), on, SIZE, "lm", strong=True)
            idx = m["first"] if e == 0 else m["again"]
            for i in range(self.SHOWN):
                x = x0 + step * i
                self.card(sc, x, y, size + 4, size + 8, on, lift=0.2)
                self.tiles(sc, self.thumbs[e][i], x, y + self.frac(0, 2.0)[1], size - 4,
                           size - 4, on, lo=0.0, hi=1.0, grid=0.0, lift=0.25)
                self.text(sc, x, y + 0.062, str(idx[i]), on, SIZE * 0.9, ink=DATA_INK)
        self.text(sc, 0.06, 0.88, "numbers: places in train_set", op * smoothstep(
            0.25, 0.31, u), SIZE, "lm", ink=QUIET)

    # ------------------------------------------------------------------ 5: batches
    def phase_5(self, sc, u, op):
        m = self.meta
        x0, x1 = self.BELT
        y = self.BELT_Y
        # the belt and its rollers
        w = self.span(x1 - x0 + 0.02)[0]
        self.rect(sc, self.P((x0 + x1) / 2, y + 0.045), w, 10.0, TRACK, op, border=AXIS,
                  border_w=1.2, radius=5, lift=0.1)
        for i in range(7):
            rx = x0 + (x1 - x0) * i / 6
            self.disk(sc, self.P(rx, y + 0.045), 3.2, AXIS, op, lift=0.15, outline=0.0)
        # the model the batches go into
        self.box(sc, self.MODEL_X, y, 62.0, 64.0, soft(WEIGHT, 0.8), op, text="model",
                 border=WEIGHT, size=SIZE, radius=8)
        travel = self.TRAVEL * float(np.clip((u - self.MOVE[0]) / (self.MOVE[1] - self.MOVE[0]),
                                             0.0, 1.0))
        for i, (name, n, img) in enumerate(self.items):
            x = self.BELT[1] - 0.025 - self.SPACING * i + travel
            if x > self.MODEL_X - 0.035:
                continue
            on = op * float(np.clip((x - x0 - 0.01) / 0.03, 0, 1)) * float(np.clip(
                (self.MODEL_X - 0.035 - x) / 0.03, 0, 1))
            if name == "…":
                self.text(sc, x, y, "…", on, HEAD, strong=True)
                continue
            s = 1.0 if n == course.BATCH else 0.75
            cw, ch = 50.0 * s, 56.0 * s
            self.stack(sc, x, y + self.frac(0, 4)[1], cw, ch, 3, on, step=(3.0, -3.0))
            self.tiles(sc, img, x, y + self.frac(0, 6)[1], cw - 10, cw - 10, on, lo=0.0, hi=1.0,
                       grid=0.0, lift=0.25)
            self.text(sc, x + self.frac(3)[0], y - self.frac(0, ch / 2 + 16)[1], str(n), on,
                      SIZE, ink=DATA_INK)
        steps = [(0.0, "0")]
        for i, (name, _n, _img) in enumerate(self.items):
            if name == "…":
                continue
            # the time item i enters the model
            gap = self.MODEL_X - 0.035 - (self.BELT[1] - 0.025 - self.SPACING * i)
            steps.append((self.MOVE[0] + (self.MOVE[1] - self.MOVE[0]) * gap / self.TRAVEL,
                          name))
        self.running(sc, 0.06, 0.55, steps, u, op, label="batches in", size=BIG)
        last = steps[-1][0]
        e = op * smoothstep(last + 0.03, last + 0.08, u)
        self.text(sc, 0.06, 0.655, m["epoch_sum"], e, SIZE, "lm")
        self.text(sc, 0.06, 0.70, "one epoch", e, SIZE, "lm", strong=True)


ANIMATIONS = {"blocks-logits-and-softmax": LogitsSoftmax, "blocks-which-is-which": WhichIsWhich,
              "blocks-the-data-pipeline": DataPipeline}

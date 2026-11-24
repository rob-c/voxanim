"""Animations 221-223, the last three of the CNN part of "Blocks and code" (see
:mod:`voxanim.blocks`), drawn from :mod:`voxanim.study_blocks_m`.

* :class:`AgainstTheMLP` (``blocks-cnn-against-the-mlp``): the course's MLP and the CNN after
  one epoch on the same batches, one run each: their numbers as two stacks of orange tiles
  to scale (52,650 against 5,258), the held-out accuracy of both every 25 steps, and the
  digits each misses, per digit; a small table keeps the numbers as they come.
* :class:`WhatItGetsWrong` (``blocks-cnn-what-it-gets-wrong``): the 364 test digits the CNN
  misses, per digit, then its three most common confusions, each with the first four
  such digits of the test set (not picked) and what it said.
* :class:`CnnMistakes` (``blocks-cnn-common-mistakes``): three broken CNN programs, each with
  torch's own error in the output slot and the shapes that explain it, the fix, and the
  three shapes to check.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). What stays across phases is drawn in
``steady``, so nothing blanks between them.
"""

from __future__ import annotations

import numpy as np

from .blocks import (
    DATA,
    GRAD,
    HEAD,
    INK,
    QUIET,
    SIZE,
    WEIGHT,
    Blocks,
    Code,
    commas,
    exact,
    measured,
    shape,
)
from .easing import ease_in_out, smoothstep
from .losschart import LossChart
from .mech import dark, soft

DATA_INK = dark(DATA, 0.3)
WEIGHT_INK = dark(WEIGHT, 0.35)
GRAD_INK = dark(GRAD, 0.25)
MLP_FILL = soft(DATA, 0.55)             # the MLP: pale blue; the CNN: full blue
CNN_FILL = DATA
MLP_INK = dark(DATA, 0.05)
CNN_INK = dark(DATA, 0.45)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


# ====================================================================== 221
class AgainstTheMLP(Blocks):
    """The two models' sizes, their held-out curves, the digits each misses."""

    TITLE = "The CNN against the MLP"
    PHASES = (("the MLP", 12.0), ("the CNN", 12.0), ("accuracy", 15.0), ("digits missed", 15.0))
    LAST = "the stamps, from random to trained"
    NEXT = "what the CNN gets wrong"
    STATION = ("forward", "forward", "step", "forward")
    COLS = (0.30, 0.41)                 # the table's two columns, and the stacks under them
    LABEL_X = 0.065
    ROWS = (0.385, 0.43, 0.472, 0.514)  # header, numbers, accuracy, missed
    BOTTOM = 0.865                      # the stacks stand on this line
    TILE = 7.0                          # px per tile of the stacks
    BOX = ((0.12, 0.45), (0.61, 0.835))     # the held-out chart
    BARS = (0.105, 0.455, 0.835, 0.225)     # the per-digit bars: x0, x1, baseline, tallest

    def setup(self):
        m = self.meta = self.study.metadata
        self.CODE = (
            Code("model", run=((0.0, 0), (0.3, 2), (0.45, 4), (0.6, 6))),
            Code("cnn-model", run=((0.0, 0), (0.25, 1), (0.4, 3), (0.55, 6))),
            Code("cnn-train", pick=tuple(range(8)), run=((0.0, 0), (0.08, 1), (0.16, 7))),
            Code("cnn-evaluate", run=((0.0, 0), (0.12, 5), (0.3, 7)),
                 out=((0.35, m["printed_cnn"]),)))
        self.EVIDENCE = (exact("the MLP's %s numbers, counted" % m["mlp_params_text"]),
                         exact("the CNN's %s numbers, counted" % m["cnn_params_text"]),
                         measured("10,000 held-out digits, one run each"),
                         measured("10,000 held-out digits, one run each"))
        self.steps = np.asarray(m["eval_steps"], float)
        self.acc_mlp = np.asarray(m["accuracy_mlp"])
        self.acc_cnn = np.asarray(m["accuracy_cnn"])
        self.chart = LossChart(self.BOX, (0.0, float(m["steps"])), (0.0, 1.0))

    def heading_for(self, k, u):
        m = self.meta
        return ("the MLP: %s numbers to learn" % m["mlp_params_text"],
                "the CNN: %s numbers, %s times fewer" % (m["cnn_params_text"], m["ratio_text"]),
                "held-out accuracy, on the same steps",
                "which digits each one misses")[k]

    def subtitle_for(self, k):
        m = self.meta
        d, c = m["per_digit_mlp"], m["per_digit_cnn"]
        return [
            "The course's MLP learns %s numbers. Each orange tile here stands for %d of them."
            % (m["mlp_params_text"], m["per_tile"]),
            "The CNN learns %s numbers, %s times fewer, to read the same digits."
            % (m["cnn_params_text"], m["ratio_text"]),
            "Both tested every 25 steps on the 10,000 held-out digits, one run each: the CNN "
            "ends at %s, the MLP at %s." % (m["cnn_accuracy_text"], m["mlp_accuracy_text"]),
            "The CNN misses %d digits, the MLP %d. The CNN misses fewer 5s (%d against %d) "
            "but more 8s (%d against %d)." % (m["cnn_missed"], m["mlp_missed"], c[5], d[5],
                                              c[8], d[8]),
        ][k]

    # ------------------------------------------------------------------ what stays
    def at(self, k, u):
        """1 once phase ``k`` has reached ``u`` (fading in over a short while)."""
        a, b = self.spans[k]
        t0 = a + (b - a) * u
        return smoothstep(t0, t0 + 0.5, self._t)

    def table(self, sc, op):
        """The numbers as they come: one row per phase, never blank once shown."""
        m = self.meta
        y0, yn, ya, ym = self.ROWS
        self.text(sc, self.LABEL_X, y0, "one run each", op, SIZE, "lm", ink=QUIET)
        for x, name, fill, ink, on in ((self.COLS[0], "MLP", MLP_FILL, MLP_INK, self.at(0, 0.1)),
                                       (self.COLS[1], "CNN", CNN_FILL, CNN_INK, self.at(1, 0.1))):
            self.rect(sc, self.P(x - 0.03, y0), 10.0, 10.0, fill, op * on, radius=2, lift=0.3)
            self.text(sc, x + 0.008, y0, name, op * on, SIZE, ink=ink, strong=True)
        rows = (
            (yn, "numbers", (m["mlp_params_text"], self.at(0, 0.2)),
             (m["cnn_params_text"], self.at(1, 0.2))),
            (ya, "accuracy", (m["mlp_accuracy_text"], self.at(2, 0.68)),
             (m["cnn_accuracy_text"], self.at(2, 0.68))),
            (ym, "missed", (commas(m["mlp_missed"]), self.at(3, 0.15)),
             (commas(m["cnn_missed"]), self.at(3, 0.15))),
        )
        for y, label, (a, on_a), (b, on_b) in rows:
            self.text(sc, self.LABEL_X, y, label, op * max(on_a, on_b), SIZE, "lm")
            self.text(sc, self.COLS[0], y, a, op * on_a, SIZE, "mm", ink=MLP_INK)
            self.text(sc, self.COLS[1], y, b, op * on_b, SIZE, "mm", ink=CNN_INK)

    def stack(self, sc, fx, tiles, op, grow=1.0):
        """``tiles = (whole, fraction)`` orange tiles, ten to a row, standing on
        :attr:`BOTTOM` and growing up from it as ``grow`` goes 0..1; the last tile is
        drawn as the fraction it holds (to scale)."""
        if op <= 0.01:
            return
        whole, frac = tiles
        cols = self.meta["stack_cols"]
        full, rest = divmod(whole, cols)
        t = self.TILE
        rows = grow * (full + (1 if rest or frac else 0))
        n = int(min(full, np.floor(rows + 1e-9)))
        x0 = fx - self.frac(cols * t / 2)[0]
        dy = self.frac(0, t)[1]
        if n > 0:
            self.tiles(sc, np.full((n, cols), 0.8), fx, self.BOTTOM - n * dy / 2, cols * t,
                       n * t, op, ink=WEIGHT, lo=0.0, hi=1.0, grid=0.35, lift=0.3)
        if rows > full + 0.5 and (rest or frac):
            y = self.BOTTOM - (full + 0.5) * dy
            if rest:
                self.tiles(sc, np.full((1, rest), 0.8), x0 + self.frac(rest * t / 2)[0], y,
                           rest * t, t, op, ink=WEIGHT, lo=0.0, hi=1.0, grid=0.35, lift=0.3)
            if frac * t > 0.5:
                w = frac * t
                self.rect(sc, self.P(x0 + self.frac(rest * t + w / 2)[0], y), w, t - 1.0,
                          WEIGHT, op * 0.75, radius=0.5, lift=0.3)

    def steady(self, sc, k, u, op):
        m = self.meta
        self.table(sc, op)
        # the two stacks, through the first two phases
        on = op * self.kept(0, 1)
        if on > 0.01:
            self.stack(sc, self.COLS[0], m["mlp_tiles"], on, eased(self._t, *self.T(0, 0.1, 0.4)))
            self.stack(sc, self.COLS[1], m["cnn_tiles"], on * self.at(1, 0.0),
                       eased(self._t, *self.T(1, 0.08, 0.2)))
            x, y = self.LABEL_X, self.BOTTOM - 0.012
            self.rect(sc, self.P(x + self.frac(self.TILE / 2)[0], y), self.TILE - 1,
                      self.TILE - 1, soft(WEIGHT, 0.25), on * self.at(0, 0.45), radius=0.5)
            self.text(sc, x + 0.016, y, "= %d numbers" % m["per_tile"], on * self.at(0, 0.45),
                      SIZE, "lm", ink=WEIGHT_INK)

    def T(self, k, a, b):
        """Times of ``u = a`` and ``u = b`` in phase ``k``."""
        s, e = self.spans[k]
        return s + (e - s) * a, s + (e - s) * b

    # ------------------------------------------------------------------ 1, 2: the stacks
    def phase_1(self, sc, u, op):
        """The MLP's stack and its row of the table, in ``steady``."""

    def phase_2(self, sc, u, op):
        m = self.meta
        on = op * smoothstep(0.35, 0.43, u)
        self.big(sc, self.LABEL_X, 0.67, "%s× fewer" % m["ratio_text"], on, ink=WEIGHT_INK,
                 size=HEAD)

    # ------------------------------------------------------------------ 3: accuracy
    def phase_3(self, sc, u, op):
        m = self.meta
        c = self.chart
        on = op * smoothstep(0.03, 0.09, u)
        self.axes(sc, c, on, ylabel="held-out accuracy", xticks=[(0, "0"), (938, "938")],
                  yticks=[(0, "0%"), (0.5, "50%"), (1, "100%")], grid=False)
        draw = eased(u, 0.1, 0.55)
        self.plot(sc, c, self.steps, self.acc_mlp, MLP_FILL, on, draw, width=1.6, dot=False)
        self.plot(sc, c, self.steps, self.acc_cnn, CNN_FILL, on, draw, width=1.6, dot=False)
        lab = on * smoothstep(0.56, 0.62, u)
        end = float(m["steps"])
        self.text(sc, c.x1, c.fy(self.acc_mlp[-1]) + 0.04, "MLP %s" % m["mlp_accuracy_text"],
                  lab, SIZE, "rm", ink=MLP_INK)
        self.text(sc, c.fx(end * 0.85), c.fy(self.acc_cnn[-1]) - 0.03,
                  "CNN %s" % m["cnn_accuracy_text"], lab, SIZE, "rm", ink=CNN_INK)

    # ------------------------------------------------------------------ 4: per digit
    def bar_x(self, d, which):
        x0, x1, _base, _top = self.BARS
        w = (x1 - x0) / 10
        return x0 + w * (d + 0.5) + (-1 if which == 0 else 1) * self.frac(7.5)[0]

    def phase_4(self, sc, u, op):
        m = self.meta
        x0, x1, base, tall = self.BARS
        top = max(max(m["per_digit_mlp"]), max(m["per_digit_cnn"]))
        on = op * smoothstep(0.04, 0.1, u)
        grow = eased(u, 0.12, 0.32)
        self.text(sc, x0, base - tall - 0.035, "missed, by digit", on, SIZE, "lm")
        w = x1 - x0
        self.rect(sc, self.P((x0 + x1) / 2, base + 0.002), self.span(w)[0], 1.4, QUIET, on,
                  radius=0.5)
        for d in range(10):
            for which, (vals, fill) in enumerate(((m["per_digit_mlp"], MLP_FILL),
                                                  (m["per_digit_cnn"], CNN_FILL))):
                h = self.span(0, tall * vals[d] / top * grow)[1]
                if h > 0.5:
                    self.rect(sc, self.P(self.bar_x(d, which), base - self.frac(0, h / 2)[1]),
                              13.0, h, fill, on, radius=min(2.0, h / 2), lift=0.3)
            self.text(sc, (self.bar_x(d, 0) + self.bar_x(d, 1)) / 2, base + 0.03, str(d), on,
                      SIZE)
        # the two digits where they differ most, labelled
        # (the shorter bar's number leans away from the taller one beside it)
        half = self.frac(6.5)[0]
        for d, a in ((5, smoothstep(0.42, 0.5, u)), (8, smoothstep(0.6, 0.68, u))):
            pair = (m["per_digit_mlp"][d], m["per_digit_cnn"][d])
            for which, ink in ((0, MLP_INK), (1, CNN_INK)):
                v = pair[which]
                y = base - tall * v / top - 0.022
                x, anchor = self.bar_x(d, which), "mm"
                if v < pair[1 - which]:
                    x, anchor = (x + half, "rm") if which == 0 else (x - half, "lm")
                self.text(sc, x, y, str(v), on * a, SIZE, anchor, ink=ink)


# ====================================================================== 222
class WhatItGetsWrong(Blocks):
    """The misses per digit, then three confusions with real examples of each."""

    TITLE = "What the CNN gets wrong"
    PHASES = (("364 missed", 12.0), ("8 as 9", 14.0), ("2 as 7", 13.0), ("8 as 0", 13.0))
    LAST = "the CNN against the MLP"
    NEXT = "three broken CNN programs"
    STATION = ("forward",) * 4
    BIG_AT = (0.065, 0.47)
    BARS = (0.205, 0.465, 0.55, 0.15)       # x0, x1, baseline, tallest
    CARDS = (0.112, 0.217, 0.322, 0.427)    # the four examples' centres
    CARD_Y, TILE = 0.72, 2.7
    LABEL_Y, NOTE_Y = 0.832, 0.877
    LABEL_X0 = 0.065                        # the note under the examples

    def setup(self):
        m = self.meta = self.study.metadata
        self.groups = m["groups"]
        out = ((0.0, m["printed"]),)
        self.CODE = (Code("cnn-evaluate", run=((0.0, 0), (0.15, 5), (0.35, 7)),
                          out=((0.4, m["printed"]),)),) + tuple(
            Code("cnn-evaluate", run=((0.0, 5),), out=out) for _g in self.groups)
        self.EVIDENCE = (measured("the %s it misses of 10,000" % m["missed_text"]),) + tuple(
            measured("the first 4 %ds it reads as %d" % (g["true"], g["said"]))
            for g in self.groups)
        self.first = m["first"]

    @staticmethod
    def article(d):
        return "an" if d == 8 else "a"

    def heading_for(self, k, u):
        m = self.meta
        if k == 0:
            return "the CNN misses %s of 10,000" % m["missed_text"]
        g = self.groups[k - 1]
        return "%s %d read as %s %d: %d times" % (self.article(g["true"]), g["true"],
                                                  self.article(g["said"]), g["said"],
                                                  g["count"])

    def subtitle_for(self, k):
        m = self.meta
        if k == 0:
            return ("After one epoch the CNN names %s of 10,000 test digits right. Of the %s "
                    "it misses, %d are %ds." % (commas(m["test"] - m["missed"]),
                                                m["missed_text"], m["worst_count"], m["worst"]))
        g = self.groups[k - 1]
        if k == 1:
            return ("Its most common mistake: %s %d read as %s %d. Of the first four, three "
                    "are plain 8s; the fourth could pass for a 9."
                    % (self.article(g["true"]), g["true"], self.article(g["said"]),
                       g["said"]))
        if k == 2:
            return ("Next, %s %d read as %s %d, %d times: the first four in the test set, not "
                    "picked." % (self.article(g["true"]), g["true"], self.article(g["said"]),
                                 g["said"], g["count"]))
        return ("Then %s %d read as %s %d, %d times. These three mistakes are %d of the %s."
                % (self.article(g["true"]), g["true"], self.article(g["said"]), g["said"],
                   g["count"], m["top_total"], m["missed_text"]))

    # ------------------------------------------------------------------ what stays
    def bar_x(self, d):
        x0, x1, _b, _t = self.BARS
        return x0 + (x1 - x0) * (d + 0.5) / 10

    def steady(self, sc, k, u, op):
        m = self.meta
        x, y = self.BIG_AT
        self.big(sc, x, y, m["missed_text"], op, label="missed", ink=DATA_INK)
        self.text(sc, x, y + 0.045, "of 10,000", op, SIZE, "lm", ink=QUIET)
        x0, x1, base, tall = self.BARS
        top = max(m["per_digit"])
        grow = eased(u, 0.1, 0.3) if k == 0 else 1.0
        g = self.groups[k - 1] if k > 0 else None
        # from the second phase on, the bars of the other digits step back
        a0 = self.spans[1][0]
        back = smoothstep(a0, a0 + 0.6, self._t)
        self.rect(sc, self.P((x0 + x1) / 2, base + 0.002), self.span(x1 - x0)[0], 1.4, QUIET,
                  op, radius=0.5)
        for d in range(10):
            v = m["per_digit"][d]
            h = self.span(0, tall * v / top * grow)[1]
            mine = g is not None and d == g["true"]
            dim = 1.0 - 0.6 * back * (not mine)
            fx = self.bar_x(d)
            if h > 0.5:
                self.rect(sc, self.P(fx, base - self.frac(0, h / 2)[1]), 15.0, h, CNN_FILL,
                          op * dim, radius=min(2.0, h / 2), lift=0.3)
            self.text(sc, fx, base + 0.028, str(d), op * dim, SIZE)
            if mine:
                # this confusion's part of the bar, darker, from the bottom
                hs = self.span(0, tall * g["count"] / top)[1]
                a = smoothstep(0.08, 0.16, u)
                self.rect(sc, self.P(fx, base - self.frac(0, hs / 2)[1]), 15.0, hs,
                          dark(DATA, 0.55), op * a, radius=1.5, lift=0.35)
                self.text(sc, fx, base - tall * v / top - 0.024,
                          "%d said %d" % (g["count"], g["said"]), op * a, SIZE,
                          ink=DATA_INK)
        # the values over the two tallest bars, in the first phase
        lab = op * smoothstep(0.32, 0.4, u) * (1.0 - back)
        for d in np.argsort(m["per_digit"])[-2:]:
            v = m["per_digit"][int(d)]
            self.text(sc, self.bar_x(int(d)), base - tall * v / top - 0.022, str(v), lab, SIZE,
                      ink=DATA_INK)

    # ------------------------------------------------------------------ the examples
    def cards(self, sc, u, op, images, trues, saids, ring=None):
        for i, (img, t, s, fx) in enumerate(zip(images, trues, saids, self.CARDS)):
            on = op * smoothstep(0.2 + 0.06 * i, 0.27 + 0.06 * i, u)
            self.digit_card(sc, np.asarray(img), fx, self.CARD_Y, self.TILE, on)
            self.text(sc, fx, self.LABEL_Y, "%d → %d" % (t, s), on, HEAD, strong=True)
            if ring is not None and i == ring:
                r = on * smoothstep(0.55, 0.62, u)
                side = 28 * self.TILE
                self.rect(sc, self.P(fx, self.CARD_Y), side + 24, side + 30, INK, r,
                          border=INK, border_w=2.4, radius=8, lift=1.0, fill_op=0.0)

    def phase_1(self, sc, u, op):
        f = self.first
        self.cards(sc, u, op, f["image"], f["true"], f["said"])
        self.text(sc, self.LABEL_X0, self.NOTE_Y, "the first four missed: true → said",
                  op * smoothstep(0.45, 0.52, u), SIZE, "lm", ink=QUIET)

    def group_phase(self, sc, u, op, g):
        ring = next((i for i, v in enumerate(g["looks_like"]) if v is not None), None)
        self.cards(sc, u, op, g["image"], [g["true"]] * len(g["index"]),
                   [g["said"]] * len(g["index"]), ring)
        note = op * smoothstep(0.45, 0.52, u)
        if ring is None:
            self.text(sc, self.LABEL_X0, self.NOTE_Y, "the first four in the test set",
                      note, SIZE, "lm", ink=QUIET)
        else:
            self.text(sc, self.LABEL_X0, self.NOTE_Y, "the first four in the test set",
                      note * (1 - smoothstep(0.55, 0.59, u)), SIZE, "lm", ink=QUIET)
            self.text(sc, self.LABEL_X0, self.NOTE_Y, "the 4th: a person might say %d too"
                      % g["looks_like"][ring], op * smoothstep(0.6, 0.64, u), SIZE, "lm",
                      ink=INK)

    def phase_2(self, sc, u, op):
        self.group_phase(sc, u, op, self.groups[0])

    def phase_3(self, sc, u, op):
        self.group_phase(sc, u, op, self.groups[1])

    def phase_4(self, sc, u, op):
        self.group_phase(sc, u, op, self.groups[2])


# ====================================================================== 223
class CnnMistakes(Blocks):
    """Three broken CNN programs, the fix, and the three shapes to check."""

    TITLE = "CNN mistakes"
    PHASES = (("in_features", 12.5), ("channels", 12.5), ("no batch", 12.5), ("the fix", 10.5),
              ("all three", 8.5))
    LAST = "what the CNN gets wrong"
    NEXT = "A dense layer learns to be local"
    STATION = ("forward",) * 5
    X0, X1 = 0.075, 0.455               # the picture's width, for the bars
    CHECK_Y = (0.76, 0.805, 0.85)       # the three lessons, kept to the end

    def setup(self):
        m = self.meta = self.study.metadata
        w = m["wrapped"]
        self.CODE = (
            Code("cnn-mistake-in-features", pick=(3, 4, 5, 6, 7), run=((0.0, 0), (0.12, 3)),
                 out=((0.3, "\n".join(w["cnn-mistake-in-features"])),)),
            Code("cnn-mistake-channels", run=((0.0, 0), (0.12, 1)),
                 out=((0.3, "\n".join(w["cnn-mistake-channels"])),)),
            Code("cnn-mistake-no-batch", run=((0.0, 0), (0.12, 1)),
                 out=((0.3, "\n".join(w["cnn-mistake-no-batch"])),)),
            Code("cnn-fix-no-batch", run=((0.0, 0), (0.15, 1)),
                 out=((0.35, m["fix"]["printed"]),)),
            Code("cnn-model", run=((0.0, 0), (0.15, 1), (0.45, 6))))
        self.EVIDENCE = (exact("the shapes, and torch's own error"),
                         exact("the shapes, and torch's own error"),
                         exact("the shapes, and torch's own error"),
                         exact("the shapes: [1, 1, 28, 28] → [1, 10]"),
                         exact("the shapes the model needs"))
        self.digit = np.asarray(m["digit"], float)
        self.checks = (
            "in_features: %s" % m["in_features"]["sum"],
            "in_channels: 1 for a grey digit",
            "keep the batch: x[0:1], not x[0]")
        self.check_at = ((0, 0.72), (1, 0.72), (3, 0.72))

    def heading_for(self, k, u):
        return ("nn.Linear(784, 10): the wrong in_features",
                "nn.Conv2d(3, 8, 3): the wrong channels",
                "x[0]: one digit without its batch",
                "the fix: x[0:1], a batch of one",
                "three shapes to check")[k]

    def subtitle_for(self, k):
        m = self.meta
        f, nb = m["in_features"], m["no_batch"]
        return [
            "After Flatten, %s numbers arrive for each digit, but nn.Linear(784, 10) expects "
            "784, so torch stops." % f["sum"],
            "nn.Conv2d(3, 8, 3) wants 3 channels, as in a colour picture. A grey digit has 1, "
            "so torch stops.",
            "x[0] drops the batch: the blocks read %s as one digit, and Flatten makes %d rows "
            "of %d, not one row of %d." % (nb["one_text"], nb["rows"], nb["row"], nb["wants"]),
            "x[0:1] keeps a batch of one, %s: one row of %d after Flatten, and %d scores out."
            % (m["fix"]["one_text"], m["fix"]["flat"][1], m["fix"]["out"][1]),
            "Three shapes to check: 1 channel in, %d into the Linear, and the batch in front."
            % f["right"],
        ][k]

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        """Three things to check, as three empty boxes from the start: each fills in, with
        its words, once its mistake has been shown."""
        for (kk, uu), y, s in zip(self.check_at, self.CHECK_Y, self.checks):
            a, b = self.spans[kk]
            t0 = a + (b - a) * uu
            on = op * smoothstep(t0, t0 + 0.5, self._t)
            at = self.P(self.X0 + 0.004, y)
            self.rect(sc, at, 10.0, 10.0, dark(DATA, 0.2), op, border=dark(DATA, 0.2),
                      border_w=1.4, radius=2, lift=0.28, fill_op=0.0)
            if on > 0.01:
                self.rect(sc, at, 10.0, 10.0, dark(DATA, 0.2), on, radius=2, lift=0.3)
                self.text(sc, self.X0 + 0.018, y, s, on, SIZE, "lm")

    # ------------------------------------------------------------------ 1: in_features
    def phase_1(self, sc, u, op):
        f = self.meta["in_features"]
        x0, x1 = self.X0, self.X1
        full = x1 - x0
        wants = op * smoothstep(0.08, 0.16, u)
        self.text(sc, x0, 0.41, "nn.Linear(784, 10) wants %d" % f["wants"], wants, SIZE, "lm")
        self.rect(sc, self.P((x0 + x1) / 2, 0.455), self.span(full)[0], 16.0, WEIGHT, wants,
                  border=WEIGHT_INK, border_w=1.4, radius=3, fill_op=0.0, lift=0.3)
        gets = op * smoothstep(0.3, 0.38, u)
        short = full * f["arrive"] / f["wants"]
        self.text(sc, x0, 0.53, "after Flatten: %s" % f["sum"], gets, SIZE, "lm")
        self.strip(sc, np.full(f["arrive"], 0.6), x0 + short / 2, 0.575, self.span(short)[0],
                   16.0, gets, lo=0.0, hi=1.0)
        rows = op * smoothstep(0.5, 0.58, u)
        self.text(sc, x0, 0.64, "%s: %d rows of %d" % (f["flat_text"], f["flat"][0],
                                                       f["flat"][1]), rows, SIZE, "lm",
                  ink=DATA_INK)

    # ------------------------------------------------------------------ 2: channels
    def phase_2(self, sc, u, op):
        c = self.meta["channels"]
        a = op * smoothstep(0.08, 0.16, u)
        fx, fy = 0.15, 0.53
        for i in reversed(range(c["wants"])):
            self.tiles(sc, np.full((3, 3), 0.55 + 0.1 * i), fx + self.frac(9.0 * i)[0],
                       fy - self.frac(0, 9.0 * i)[1], 54.0, 54.0, a, ink=WEIGHT, lo=0.0,
                       hi=1.0, grid=0.5, lift=0.3 + 0.01 * (2 - i))
        self.text(sc, fx + 0.01, 0.415, "a stamp: 3 × 3 × 3", a, SIZE)
        self.text(sc, fx + 0.01, 0.64, "wants %d channels" % c["wants"], a, SIZE,
                  ink=WEIGHT_INK)
        b = op * smoothstep(0.3, 0.38, u)
        dx = 0.365
        self.digit_card(sc, self.digit, dx, fy, 2.6, b)
        self.text(sc, dx, 0.415, "x: %s" % c["input_text"], b, SIZE)
        self.text(sc, dx, 0.64, "a grey digit: %d channel" % c["got"], b, SIZE, ink=DATA_INK)

    # ------------------------------------------------------------------ 3: no batch
    def one_digit(self, sc, op, label, note):
        fx, fy = 0.15, 0.52
        self.digit_card(sc, self.digit, fx, fy, 2.0, op)
        self.text(sc, fx, 0.405, label, op, SIZE)
        self.text(sc, fx, 0.635, note, op, SIZE, ink=DATA_INK)

    def phase_3(self, sc, u, op):
        nb = self.meta["no_batch"]
        a = op * smoothstep(0.08, 0.16, u)
        self.one_digit(sc, a, "x[0]: %s" % nb["one_text"], "no batch")
        b = op * smoothstep(0.32, 0.4, u)
        self.arrow(sc, (0.235, 0.52), (0.285, 0.52), INK, b, eased(u, 0.32, 0.4), width=1.8)
        fx = 0.375
        self.tiles(sc, np.full((nb["rows"], nb["row"]), 0.6), fx, 0.52, nb["row"] * 6.0,
                   nb["rows"] * 6.0, b, lo=0.0, hi=1.0, grid=0.35)
        self.text(sc, fx, 0.405, "Flatten: %s" % nb["flat_text"], b, SIZE)
        self.text(sc, fx, 0.635, "%d rows of %d" % (nb["rows"], nb["row"]), b, SIZE,
                  ink=DATA_INK)
        c = op * smoothstep(0.55, 0.63, u)
        self.text(sc, self.X0, 0.69, "nn.Linear wants rows of %d" % nb["wants"], c, SIZE, "lm",
                  ink=GRAD_INK)

    # ------------------------------------------------------------------ 4: the fix
    def phase_4(self, sc, u, op):
        fix = self.meta["fix"]
        a = op * smoothstep(0.08, 0.16, u)
        self.one_digit(sc, a, "x[0:1]: %s" % fix["one_text"], "a batch of one")
        b = op * smoothstep(0.3, 0.38, u)
        self.arrow(sc, (0.255, 0.5), (0.295, 0.5), INK, b, eased(u, 0.3, 0.38), width=1.8)
        x0, x1 = 0.31, 0.455
        self.text(sc, x0, 0.43, "Flatten: %s" % fix["flat_text"], b, SIZE, "lm")
        self.strip(sc, np.full(fix["flat"][1], 0.6), (x0 + x1) / 2, 0.475,
                   self.span(x1 - x0)[0], 14.0, b, lo=0.0, hi=1.0)
        c = op * smoothstep(0.5, 0.58, u)
        self.text(sc, x0, 0.555, "scores: %s" % fix["out_text"], c, SIZE, "lm")
        self.tiles(sc, np.full((1, fix["out"][1]), 0.6), x0 + self.frac(5 * 11.0)[0], 0.6,
                   110.0, 11.0, c, lo=0.0, hi=1.0, grid=0.4)

    # ------------------------------------------------------------------ 5: all three
    def phase_5(self, sc, u, op):
        m = self.meta
        rows = (("x", m["batch_text"]), ("after Flatten", shape(m["right_walk"][-1][1])),
                ("scores", shape((m["batch"][0], 10))))
        for i, (name, value) in enumerate(rows):
            on = op * smoothstep(0.08 + 0.12 * i, 0.15 + 0.12 * i, u)
            y = 0.44 + 0.07 * i
            self.text(sc, self.X0, y, name, on, SIZE, "lm")
            self.text(sc, self.X1, y, value, on, HEAD, "rm", strong=True)


ANIMATIONS = {"blocks-cnn-against-the-mlp": AgainstTheMLP,
              "blocks-cnn-what-it-gets-wrong": WhatItGetsWrong,
              "blocks-cnn-common-mistakes": CnnMistakes}

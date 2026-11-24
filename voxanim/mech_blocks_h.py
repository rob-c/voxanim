"""Animations 206-208 of the "Blocks and code" series (see :mod:`voxanim.blocks`): Adam,
the training loop, and epochs and batches, drawn from :mod:`voxanim.study_blocks_h`.

* :class:`AdamBlocks` (``blocks-adam``): ``Adam(model.parameters())`` adds two green
  blocks, ``m`` and ``v``, the shape of the weights' orange block and all 0 at the start;
  memory: 3 x 52,650 numbers; one weight of the last layer through Adam's second step with
  its real numbers, tile by tile (its red gradient, the new ``m`` and ``v``, Adam's
  corrected ratio, and the move the green hand makes); then the real Adam and SGD runs.
* :class:`TrainingLoop` (``blocks-the-training-loop``): the loop as a circular track, the
  five lines of a step as its stations (the series' loop map, large); one batch goes round
  once, slowly, each station saying what it makes; then 938 times, the step counter and the
  real loss curve drawing as it goes; and what ``print(epoch, loss.item())`` writes.
* :class:`EpochsBatches` (``blocks-epochs-and-batches``): 60,000 / 64 as 938 tiles, one
  per batch, the last one of 32; one epoch uses each digit once; ``shuffle=True`` starts
  each epoch with other digits; the loss per step against per epoch over three epochs.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). What stays across phases is drawn in
``steady``, so nothing blanks between them.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import layout
from .blocks import (
    BIG,
    DATA,
    GRAD,
    HEAD,
    INK,
    LOOP,
    LOSS,
    OPTIM,
    QUIET,
    SIZE,
    STATION_INK,
    WEIGHT,
    Blocks,
    Code,
    _merge,
    _Timeline,
    commas,
    exact,
    measured,
)
from .easing import ease_in_out, smoothstep
from .mech import AXIS, PAPER, dark, soft
from .study_blocks_h import FIRST_SHOWN

DATA_INK, LOSS_INK, OPTIM_INK = dark(DATA, 0.3), dark(LOSS, 0.25), dark(OPTIM, 0.35)
WEIGHT_INK, GRAD_INK = dark(WEIGHT, 0.35), dark(GRAD, 0.25)
SGD_INK = dark(AXIS, 0.3)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def mix(a, b, s):
    return a + (b - a) * s


def clip_start(ys, top):
    """The first index where ``ys`` is at most ``top``."""
    below = np.flatnonzero(np.asarray(ys) <= top)
    return int(below[0]) if len(below) else len(ys) - 1


class Charted(Blocks):
    """A loss chart in the picture, as the series draws one."""

    def loss_axes(self, sc, ch, op, xticks, yticks):
        self.axes(sc, ch, op, xticks=xticks, yticks=yticks)

    def curve(self, sc, ch, xs, ys, ink, op, upto, width=1.6, dot=True, every=1):
        """``ys`` against ``xs`` up to ``upto``, from where it first comes into the chart's
        range (a curve that starts above it is not drawn flat along the top)."""
        if op <= 0.01:
            return None
        xs, ys = np.asarray(xs, float), np.asarray(ys, float)
        first = clip_start(ys, ch.y_range[1])
        if upto < xs[first]:
            return None
        return ch.curve(sc, self.P, xs[first:], ys[first:], upto, ink, op, width=width,
                        dot=dot, every=every)

    def dashed(self, sc, fx, y_from, y_to, op, dashes=10):
        """A dashed vertical line at layout ``fx`` from ``y_from`` up to ``y_to``."""
        if op <= 0.01:
            return
        for i in range(dashes):
            a = y_from + (y_to - y_from) * i / dashes
            b = y_from + (y_to - y_from) * (i + 0.5) / dashes
            self.polyline(sc, [self.P(fx, a), self.P(fx, b)], 0.9, AXIS, 0.8 * op, bias=0.0)

    def off_top(self, sc, ch, ys, op, text):
        """``text`` (``↑ 2.32``) where a curve comes into the chart from above."""
        x = ch.fx(clip_start(ys, ch.y_range[1]))
        self.text(sc, x + 0.008, ch.y0 + 0.014, text, op, SIZE, "lm", ink=LOSS_INK)


# ====================================================================== 206
class AdamBlocks(Charted):
    """m and v appear; memory 3 x; one weight's m and v; its move; Adam against SGD."""

    TITLE = "Adam keeps two more blocks"
    PHASES = (("m and v", 11.0), ("memory", 10.0), ("one weight", 14.0), ("the move", 13.0),
              ("Adam and SGD", 12.0))
    LAST = "three learning rates"
    NEXT = "the training loop"
    STATION = ("step",) * 5
    # the blocks: large, one under another (phases 1-2), then small, side by side
    BIG_TILE, SMALL_TILE = 6.0, 3.6
    BIG_X, BIG_YS = 0.165, (0.455, 0.625, 0.795)
    SMALL_XS, SMALL_Y = (0.12, 0.265, 0.41), 0.432
    NAMES = ("weight", "m", "v")
    INKS = (WEIGHT, OPTIM, OPTIM)
    TAG_X, TOTAL_X = 0.282, 0.36
    # phases 3-4: one weight's rows
    ROW_YS = (0.525, 0.61, 0.695, 0.78)
    LETTER_X, BOX_X, SUM_X = 0.058, 0.132, 0.187
    BOX_W, BOX_H = 88.0, 28.0
    CHART = ((0.11, 0.45), (0.54, 0.83))
    YMAX = 0.9                        # the curves start at 2.32, off the top (labelled)

    def setup(self):
        m = self.meta = self.study.metadata
        self.T = m["text"]
        self.blocks = np.asarray(self.study.panels[0].data["values"], float)
        self.at = tuple(m["weight"])
        self.sgd = np.asarray(m["sgd_curve"])
        self.adam = np.asarray(m["adam_curve"])
        self.steps = np.arange(len(self.sgd), dtype=float)
        self.lower_from = m["adam_ahead_until"] + 1
        i = m["adam_line_index"]
        step = m["step"].index(m["step_line"])
        self.CODE = (Code("optimizer", run=((0.0, 0), (0.12, i))),
                     Code("optimizer", run=((0.0, i),)),
                     Code("step", run=((0.0, step),)),
                     Code("step", run=((0.0, step),)),
                     Code("optimizer", run=((0.0, 0), (0.42, i))))
        self.EVIDENCE = (exact("an m and a v for every weight"),
                         exact("3 × %s numbers" % commas(m["parameters"])),
                         measured("the Adam run, its second step"),
                         measured("the Adam run, its second step"),
                         measured("SGD and Adam from the same start"))
        self.chart_ = self.chart(self.CHART, [0, len(self.sgd) - 1], [0.0, self.YMAX], pad=0.0)

    def heading_for(self, k, u):
        return ("Adam: two more blocks, m and v", "memory: 3 × the numbers",
                "one weight, step %d: g, m and v" % self.meta["t"],
                "the move: about 0.001", "Adam against SGD: real runs")[k]

    def subtitle_for(self, k):
        m, T = self.meta, self.T
        return [
            "Adam keeps two more numbers, m and v, for every weight. Both start at 0.",
            "The model has %s numbers. With m and v, Adam keeps 3 × as many: %s."
            % (commas(m["parameters"]), commas(m["adam_numbers"])),
            "At step %d this weight's gradient is %s; m becomes %s and v becomes %s."
            % (m["t"], T["g"], T["m"], T["v"]),
            "Adam moves it by %s. The largest move of all %s is %s: the learning rate."
            % (T["move"], commas(m["parameters"]), T["largest"]),
            "From the same start, Adam is lower at first; from step %d SGD is. Tested: "
            "%.2f%% and %.2f%% right." % (self.lower_from, 100 * m["adam_accuracy"],
                                          100 * m["sgd_accuracy"]),
        ][k]

    # ------------------------------------------------------------------ the blocks
    def small(self, k, u):
        """0 while the blocks are large, 1 once they are small (moving at phase 3's start)."""
        if k < 2:
            return 0.0
        return eased(u, 0.0, 0.12) if k == 2 else 1.0

    def block_at(self, i, s):
        """``(fx, fy, tile)`` of block ``i`` when ``s`` of the way to small."""
        return (mix(self.BIG_X, self.SMALL_XS[i], s), mix(self.BIG_YS[i], self.SMALL_Y, s),
                mix(self.BIG_TILE, self.SMALL_TILE, s))

    def filled(self, k, u):
        """How far m and v have their values (after 2 steps) rather than 0."""
        if k < 2:
            return 0.0
        return smoothstep(0.14, 0.24, u) if k == 2 else 1.0

    def shown(self, i, k, u):
        """Block ``i`` appears in phase 1: the weights first, then m, then v."""
        if k > 0:
            return 1.0
        return (1.0, smoothstep(0.2, 0.27, u), smoothstep(0.42, 0.49, u))[i]

    def steady(self, sc, k, u, op):
        s = self.small(k, u)
        fill = self.filled(k, u)
        ring = op * (smoothstep(0.26, 0.32, u) if k == 2 else float(k == 3))
        for i in range(3):
            on = op * self.shown(i, k, u)
            if on <= 0.01:
                continue
            fx, fy, tile = self.block_at(i, s)
            vals = self.blocks[i]
            w, h = 32 * tile, 10 * tile
            self.rect(sc, self.P(fx, fy), w + 8, h + 8, PAPER, on, border=dark(self.INKS[i], 0.2),
                      border_w=1.6, radius=4, lift=0.1)
            lo, hi = float(vals.min()), float(vals.max())
            if i == 2:                  # v: a few big values; deeper still means bigger
                lo, hi = 0.0, float(np.quantile(vals, 0.9))
            if i == 0 or fill >= 1.0:
                cell = self.tiles(sc, vals, fx, fy, w, h, on, ink=self.INKS[i], lo=lo, hi=hi,
                                  grid=0.25)
            else:
                cell = self.tiles(sc, np.zeros_like(vals), fx, fy, w, h, on * (1 - fill),
                                  zero=np.ones_like(vals, bool), grid=0.25)
                if fill > 0:
                    self.tiles(sc, vals, fx, fy, w, h, on * fill, ink=self.INKS[i], lo=lo,
                               hi=hi, grid=0.25, lift=0.35)
            if ring > 0.01:
                r, c = self.at
                x, y = cell(r, c)
                self.rect(sc, self.P(x, y), tile + 5, tile + 5, INK, ring, border=INK,
                          border_w=2.2, radius=2, lift=0.8, fill_op=0.0)
            # the label: under each large block's left corner's top, over each small one
            big = op * self.kept(0, 1) * self.shown(i, k, u)
            name = "%s %s" % (self.NAMES[i], self.meta["layer_shape_text"])
            self.text(sc, self.BIG_X - 0.1, self.BIG_YS[i] - 0.087, name, big * (1 - s), SIZE,
                      "lm", ink=dark(self.INKS[i], 0.35))
            self.text(sc, self.SMALL_XS[i], self.SMALL_Y - 0.068, self.NAMES[i], on * s, SIZE,
                      ink=dark(self.INKS[i], 0.35))
        # the rows of the one weight, phases 3-4
        rows = op * self.kept(2, 3)
        if rows > 0.01:
            self.weight_rows(sc, k, u, rows)

    # ------------------------------------------------------------------ 1: m and v
    def phase_1(self, sc, u, op):
        on = op * smoothstep(0.62, 0.69, u)
        for y in self.BIG_YS[1:]:
            self.text(sc, self.TAG_X, y, "all 0", on, SIZE, "lm", ink=QUIET)
        self.text(sc, self.TAG_X, 0.38, "same shape", op * smoothstep(0.8, 0.86, u), SIZE, "lm",
                  strong=True)

    # ------------------------------------------------------------------ 2: memory
    def phase_2(self, sc, u, op):
        n = self.meta["parameters"]
        self.text(sc, self.TAG_X, 0.38, "whole model", op * smoothstep(0.04, 0.1, u), SIZE,
                  "lm", ink=QUIET)
        times = (0.1, 0.32, 0.54)
        for i, y in enumerate(self.BIG_YS):
            on = op * smoothstep(times[i], times[i] + 0.06, u)
            self.text(sc, self.TAG_X, y, commas(n), on, SIZE, "lm", ink=dark(self.INKS[i], 0.35),
                      strong=False)
        total = [(0.0, commas(n)), (times[1] + 0.03, commas(2 * n)),
                 (times[2] + 0.03, commas(3 * n))]
        self.running(sc, self.TOTAL_X, 0.56, total, u, op * smoothstep(0.1, 0.16, u),
                     label="numbers kept", size=BIG)
        self.text(sc, self.TOTAL_X, 0.64, "3 × the model", op * smoothstep(0.72, 0.78, u), HEAD,
                  "lm", strong=True)

    # ------------------------------------------------------------------ 3-4: one weight
    def row_box(self, sc, y, text, ink, op, behind=None):
        """One number as a tile with its value in it; ``behind``: a tile of that colour
        just behind it (the weight's red gradient)."""
        if op <= 0.01:
            return
        if behind is not None:
            self.rect(sc, self.P(self.BOX_X + self.frac(6.0)[0], y + self.frac(0, 6.0)[1]),
                      self.BOX_W, self.BOX_H, soft(behind, 0.55), op, border=behind,
                      border_w=1.4, radius=6, lift=0.2)
        c = self.P(self.BOX_X, y)
        self.rect(sc, c, self.BOX_W, self.BOX_H, soft(ink, 0.7), op, border=dark(ink, 0.15),
                  border_w=1.4, radius=6, lift=0.6)
        self.label(sc, c, text, SIZE, opacity=op, ink=INK)

    def weight_rows(self, sc, k, u, op):
        T = self.T
        y0, y1, y2, y3 = self.ROW_YS
        a3 = (lambda t: 1.0) if k > 2 else (lambda t: smoothstep(t, t + 0.06, u))
        a4 = (lambda t: 0.0) if k < 3 else (lambda t: smoothstep(t, t + 0.06, u))
        # the weight, its red gradient behind it; the hand moves it in phase 4
        w_on = op * a3(0.3)
        after = a4(0.66)
        self.text(sc, self.LETTER_X, y0, "w", w_on, HEAD, "lm", ink=WEIGHT_INK)
        self.row_box(sc, y0, T["w_after"] if after > 0.5 else T["w_before"], WEIGHT, w_on,
                     behind=GRAD)
        self.text(sc, self.SUM_X, y0, "g = %s" % T["g"], op * a3(0.36), SIZE, "lm", ink=GRAD_INK)
        # m: the formula, then the box; in phase 4 the formula gives way to the corrected m
        m_sum, m_box = a3(0.44), a3(0.52)
        # each formula gives way to the next: the old one out, then the new one in
        gone_m, fix_m = a4(0.05), a4(0.1)
        self.text(sc, self.LETTER_X, y1, "m", op * m_sum, HEAD, "lm", ink=OPTIM_INK)
        self.row_box(sc, y1, T["m"], OPTIM, op * m_box)
        self.text(sc, self.SUM_X, y1, T["m_sum"], op * m_sum * (1 - gone_m), SIZE, "lm")
        self.text(sc, self.SUM_X, y1, "m " + T["m_hat"], op * fix_m, SIZE, "lm")
        v_sum, v_box = a3(0.64), a3(0.74)
        gone_v, fix_v = a4(0.21), a4(0.26)
        self.text(sc, self.LETTER_X, y2, "v", op * v_sum, HEAD, "lm", ink=OPTIM_INK)
        self.row_box(sc, y2, T["v"], OPTIM, op * v_box)
        self.text(sc, self.SUM_X, y2, T["v_sum"], op * v_sum * (1 - gone_v), SIZE, "lm")
        self.text(sc, self.SUM_X, y2, T["root"], op * fix_v, SIZE, "lm")
        # phase 4: the move, made by the green hand
        mv = op * a4(0.42)
        self.hand(sc, self.LETTER_X + 0.008, y3, mv, size=0.8)
        self.row_box(sc, y3, T["move"], OPTIM, mv)
        self.text(sc, self.SUM_X, y3, T["move_sum"], mv, SIZE, "lm")

    def phase_3(self, sc, u, op):
        pass

    def phase_4(self, sc, u, op):
        on = op * smoothstep(0.8, 0.86, u)
        self.text(sc, self.SUM_X, 0.865, "largest move of all: %s" % self.T["largest"], on, SIZE,
                  "lm", strong=True)

    # ------------------------------------------------------------------ 5: Adam and SGD
    def phase_5(self, sc, u, op):
        ch = self.chart_
        on = op * smoothstep(0.02, 0.08, u)
        last = len(self.sgd) - 1
        self.loss_axes(sc, ch, on, xticks=[(0, "0"), (last + 1, str(last + 1))],
                       yticks=[(0, "0"), (0.4, "0.4"), (0.8, "0.8")])
        self.text(sc, ch.x0, ch.y0 - 0.025, "loss, mean of 50 steps", on, SIZE, "lm",
                  ink=LOSS_INK)
        self.off_top(sc, ch, self.sgd, op * smoothstep(0.1, 0.16, u),
                     "↑ from %.2f" % self.meta["first_loss"])
        sgd_u = smoothstep(0.08, 0.38, u) * last
        self.curve(sc, ch, self.steps, self.sgd, SGD_INK, on, sgd_u, dot=u < 0.42)
        self.text(sc, ch.x1, ch.y0 + 0.04, "SGD, lr 0.1", op * smoothstep(0.08, 0.14, u), SIZE,
                  "rm", ink=SGD_INK)
        ad = op * smoothstep(0.4, 0.46, u)
        self.curve(sc, ch, self.steps, self.adam, OPTIM, ad, smoothstep(0.42, 0.72, u) * last,
                   dot=u < 0.76)
        self.text(sc, ch.x1, ch.y0 + 0.085, "Adam, lr 0.001", ad, SIZE, "rm", ink=OPTIM_INK)
        mk = op * smoothstep(0.78, 0.84, u)
        self.dashed(sc, ch.fx(self.lower_from), ch.y1, ch.y0 + 0.035, mk)
        self.text(sc, ch.fx(self.lower_from), ch.y1 + 0.03, str(self.lower_from), mk, SIZE,
                  ink=INK)


# ====================================================================== 207
class TrainingLoop(Charted):
    """The track; one step, slowly; 938 times round; one epoch."""

    TITLE = "The training loop"
    PHASES = (("the track", 12.0), ("one step", 20.0), ("938 times", 15.0),
              ("one epoch", 11.0))
    LAST = "Adam keeps m and v"
    NEXT = "epochs and batches"
    CENTRE = (0.265, 0.5)
    HW, HH = 178.0, 56.0              # the track's half width and half height (px)
    READ_Y = 0.665
    CHART = ((0.11, 0.45), (0.72, 0.855))
    # phase 2: (station, line key, arrive, leave): the card waits at each
    STOPS = (("data", "data", 0.0, 0.13), ("forward", "forward", 0.2, 0.32),
             ("loss", "loss", 0.39, 0.5), ("backward", "zero", 0.6, 0.67),
             ("backward", "backward", 0.67, 0.78), ("step", "step", 0.84, 0.93))
    TICK_STEP = 0.9                    # phase 2: the counter turns to 1 here
    SLOW_LAPS = (0.08, 0.38)           # phase 3: laps 1-3 slowly, then fast
    FAST_END = 0.9

    def setup(self):
        m = self.meta = self.study.metadata
        self.losses = np.asarray(m["losses"])
        self.n = len(self.losses)
        self.lines = {s["code"]: s["line"] for s in m["stations"]}
        line = dict(data=self.lines["for x, y in loader:"], forward=self.lines[
            "logits = model(x)"], loss=self.lines["loss = loss_fn(logits, y)"],
            zero=self.lines["optimizer.zero_grad()"], backward=self.lines["loss.backward()"],
            step=self.lines["optimizer.step()"])
        self.line = line
        pick = tuple(m["pick"])
        run2 = tuple((a if i else 0.0, line[key]) for i, (_st, key, a, _b) in
                     enumerate(self.STOPS)) + ((0.95, line["data"]),)
        self.CODE = (Code("train", pick, run=((0.0, 0), (0.25, line["data"]))),
                     Code("train", pick, run=run2),
                     Code("train", pick, run=((0.0, line["data"]),)),
                     Code("train", pick, run=((0.0, 0), (0.3, len(pick) - 1)),
                          out=((0.4, m["printed"]),)))
        self.EVIDENCE = (exact("the five lines of one step"),
                         measured("the first step of the run"),
                         measured("the loss of every step, one epoch"),
                         measured("the last batch's loss"))
        self.chart_ = self.chart(self.CHART, [0, self.n], [0.0, 2.4], pad=0.0)
        L, R = self.HW - self.HH, self.HH
        self.L, self.R = L, R
        self.perim = 4 * L + 2 * np.pi * R
        self.pos = dict(data=(-L + 30, -R), forward=(0.0, -R), loss=(L - 30, -R),
                        backward=(L * 0.5, R), step=(-L * 0.5, R))
        self.vs = dict(data=30.0, forward=L, loss=2 * L - 30,
                       backward=2 * L + np.pi * R + 0.5 * L, step=2 * L + np.pi * R + 1.5 * L)

    def _timelines(self):
        super()._timelines()
        # the corner map follows the card round in phase 2
        ts = np.arange(0.0, self.duration + self.TICK, self.TICK)
        self._tl_station = _Timeline(_merge((float(t), self.station_at(float(t))) for t in ts))

    def station_at(self, t):
        k, u = self.where(t)
        if k == 1:
            return self.stop_at(u)[0]
        return ("data", "data", "data", "data")[k]

    def station_for(self, k):
        return "data"

    def stop_at(self, u):
        """Phase 2: ``(station, line key)`` of the stop the card is at or heading to."""
        for st, key, _a, b in self.STOPS:
            if u < b:
                return st, key
        return "data", "data"

    def heading_for(self, k, u):
        return ("for x, y in loader: once round per batch", "one step: five lines, in order",
                "938 batches: 938 times round", "for epoch: once through the data")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "The loop is a track: each batch goes once round it, through five lines of code.",
            "One step on the first batch: its logits, a loss of %.3f, %s gradients, a move."
            % (m["first_loss"], commas(m["gradients"])),
            "Round and round: after %d steps the loss of a batch is down from %.2f to %.2f."
            % (m["steps"], m["first_loss"], m["last_loss"]),
            "One epoch is %d steps. Then print shows the last batch's loss: %.4f."
            % (m["steps"], m["last_loss"]),
        ][k]

    # ------------------------------------------------------------------ the track
    def point(self, v):
        """The track at distance ``v`` px round it (0: its top-left corner, clockwise)."""
        L, R = self.L, self.R
        v = float(v) % self.perim
        if v < 2 * L:
            return -L + v, -R
        if v < 2 * L + np.pi * R:
            a = (v - 2 * L) / R - np.pi / 2
            return L + R * np.cos(a), R * np.sin(a)
        if v < 4 * L + np.pi * R:
            return L - (v - 2 * L - np.pi * R), R
        a = (v - 4 * L - np.pi * R) / R + np.pi / 2
        return -L + R * np.cos(a), R * np.sin(a)

    def runs(self, boxes):
        """The track as px offsets, in pieces that stop short of the stations' boxes."""
        if getattr(self, "_runs", (None,))[0] == boxes:
            return self._runs[1]
        out, run = [], []
        for v in np.linspace(0.0, self.perim, 501):
            p = self.point(v) if v < self.perim else self.point(0.0)
            if any(abs(p[0] - bx) < hw and abs(p[1] - by) < hh for bx, by, hw, hh in boxes):
                if len(run) > 1:
                    out.append(tuple(run))
                run = []
            else:
                run.append((float(p[0]), float(p[1])))
        if len(run) > 1:
            out.append(tuple(run))
        self._runs = (boxes, tuple(out))
        return self._runs[1]

    def boxes(self):
        out = {}
        for name, (dx, dy) in self.pos.items():
            w = layout.text_width(name, SIZE) * self.cam.width + 22.0
            out[name] = (dx, dy, w / 2 + 4.0, 17.0)
        return out

    def track(self, sc, op, lit=None, glow=1.0, busy=0.0):
        """The big track with its five stations; ``lit`` station filled; ``busy``: the
        track drawn in the data's blue (batches going round too fast to follow)."""
        c = self.P(*self.CENTRE)
        boxes = self.boxes()
        for run in self.runs(tuple(boxes[n] for n in self.pos)):
            pts = [self.shift(c, x, y) for x, y in run]
            self.polyline(sc, pts, 2.6, LOOP, op, bias=0.1)
            if busy > 0.01:
                self.polyline(sc, pts, 4.0, DATA, op * busy, bias=0.15)
        L, R = self.L, self.R
        self._head(sc, self.shift(c, L + R - 0.5, -3), self.shift(c, L + R, 2), LOOP, op, 2.6)
        self._head(sc, self.shift(c, -L - R + 0.5, 3), self.shift(c, -L - R, -2), LOOP, op, 2.6)
        for name, (dx, dy) in self.pos.items():
            g = glow if name == lit else 0.0
            ink = STATION_INK[name]
            w = boxes[name][2] * 2 - 8.0
            fill = tuple(float(v) for v in mix(np.asarray(PAPER), np.asarray(soft(ink, 0.45)), g))
            self.rect(sc, self.shift(c, dx, dy), w, 28.0, fill, op,
                      border=INK if g > 0.5 else dark(ink, 0.1), border_w=1.4 + 1.2 * g,
                      radius=14, lift=0.4)
            self.label(sc, self.shift(c, dx, dy), name, SIZE, opacity=op, ink=INK)

    def batch_card(self, sc, v, op):
        """The batch, a small blue card, on the track at ``v`` px round it."""
        if op <= 0.01:
            return
        c = self.P(*self.CENTRE)
        x, y = self.point(v)
        p = self.shift(c, x, y)
        self.rect(sc, p, 26.0, 20.0, soft(DATA, 0.55), op, border=dark(DATA, 0.2), border_w=2.0,
                  radius=4, lift=1.4)
        self.rect(sc, self.shift(p, 0, -7.0), 20.0, 3.0, DATA, op, radius=1.5, lift=1.45)

    # ------------------------------------------------------------------ the phases' clock
    def stop_v(self, station):
        """Where the card waits at ``station``: on the track just before its box."""
        return self.vs[station] - self.boxes()[station][2] - 18.0

    def card_v(self, k, u):
        """Where the card is (px round the track) and how much of it shows."""
        home = self.stop_v("data")
        if k == 0:
            return home, smoothstep(0.3, 0.37, u)
        if k == 1:
            stops = [(self.stop_v(st), a, b) for st, _key, a, b in self.STOPS]
            stops.append((home + self.perim, 1.0, 1.0))
            for (v0, _a0, b0), (v1, a1, _b1) in itertools.pairwise(stops):
                if u < b0:
                    return v0, 1.0
                if u < a1:
                    return mix(v0, v1, eased(u, b0, a1)), 1.0
            return home, 1.0
        if k == 2:
            a, b = self.SLOW_LAPS
            laps = np.clip((u - a) / (b - a), 0.0, 1.0) * 3
            return home + self.perim * laps, 1.0 - smoothstep(
                b - 0.01, b + 0.05, u) + smoothstep(self.FAST_END, self.FAST_END + 0.05, u)
        return home, 1.0

    def steps_done(self, k, u):
        """The step counter: how many steps are done."""
        if k == 0:
            return 0.0
        if k == 1:
            return float(u >= self.TICK_STEP)
        if k == 2:
            a, b = self.SLOW_LAPS
            if u < b:
                return 1.0 + np.floor(np.clip((u - a) / (b - a), 0.0, 1.0) * 3 * 1.0001)
            s = np.clip((u - b) / (self.FAST_END - b), 0.0, 1.0)
            return 4.0 + (self.n - 4) * s ** 2
        return float(self.n)

    def steady(self, sc, k, u, op):
        lit = None
        glow = 1.0
        if k == 1:
            lit = self.stop_at(u)[0]
        elif k == 0:
            lit, glow = "data", smoothstep(0.3, 0.37, u)
        busy = 0.0
        if k == 2:
            b = self.SLOW_LAPS[1]
            busy = smoothstep(b - 0.01, b + 0.05, u) * (1 - smoothstep(self.FAST_END,
                                                                       self.FAST_END + 0.05, u))
        self.track(sc, op, lit, glow, busy)
        v, shown = self.card_v(k, u)
        self.batch_card(sc, v, op * shown)
        # the step counter in the middle of the track
        done = self.steps_done(k, u)
        cx, cy = self.CENTRE
        on = op * (smoothstep(0.55, 0.62, u) if k == 0 else 1.0)
        self.big(sc, cx, cy + 0.025, commas(int(done)), on, label="steps done", size=BIG,
                 anchor="mm")
        # the loss chart: the losses of the steps done
        ch = self.chart_
        con = op * (smoothstep(0.75, 0.82, u) if k == 0 else 1.0)
        self.loss_axes(sc, ch, con, xticks=[(0, "0"), (self.n, str(self.n))],
                       yticks=[(0, "0"), (1, "1"), (2, "2")])
        self.text(sc, ch.x0, ch.y0 - 0.022, "loss", con, SIZE, "lm", ink=LOSS_INK)
        if done >= 1:
            self.curve(sc, ch, np.arange(self.n), self.losses, LOSS, op, min(done - 1, self.n - 1),
                       width=1.3, dot=True)

    # ------------------------------------------------------------------ 1: the track
    def phase_1(self, sc, u, op):
        on = op * smoothstep(0.3, 0.37, u)
        self.text(sc, self.CENTRE[0], self.READ_Y, "a batch of 64 digits", on, SIZE,
                  ink=DATA_INK)

    # ------------------------------------------------------------------ 2: one step
    def phase_2(self, sc, u, op):
        reads = self.meta["reads"]
        steps = [(a, reads[key]) for _st, key, a, _b in self.STOPS]
        station = self.stop_at(u)[0]
        self.running(sc, self.CENTRE[0], self.READ_Y, steps, u, op, size=HEAD, anchor="mm",
                     ink=dark(STATION_INK[station], 0.35))

    # ------------------------------------------------------------------ 3: 938 times
    def phase_3(self, sc, u, op):
        on = op * smoothstep(self.FAST_END, self.FAST_END + 0.05, u)
        self.text(sc, self.CENTRE[0], self.READ_Y, "loss: %.3f" % self.meta["last_loss"], on,
                  HEAD, ink=LOSS_INK)

    # ------------------------------------------------------------------ 4: one epoch
    def phase_4(self, sc, u, op):
        self.text(sc, self.CENTRE[0], self.READ_Y, "%d steps = 1 epoch" % self.n,
                  op * smoothstep(0.08, 0.14, u), HEAD, strong=True)
        ch = self.chart_
        on = op * smoothstep(0.45, 0.51, u)
        x, y = ch.fx(self.n - 1), ch.fy(self.losses[-1])
        self.rect(sc, self.P(x, y), 16.0, 16.0, INK, on, border=INK, border_w=2.0, radius=8,
                  lift=1.5, fill_op=0.0)
        self.text(sc, ch.x1, ch.y0 - 0.022, "last: %.4f" % self.meta["last_loss"], on, SIZE, "rm",
                  strong=True)


# ====================================================================== 208
class EpochsBatches(Charted):
    """60,000 / 64; one epoch; a new order; loss per step against per epoch."""

    TITLE = "Epochs and batches"
    PHASES = (("60,000 ÷ 64", 14.0), ("one epoch", 12.0), ("a new order", 13.0),
              ("per epoch", 16.0))
    LAST = "the loop: 938 times round"
    NEXT = "evaluation: held-out digits"
    STATION = ("data",) * 4
    COLS, ROWS, TILE = 47, 20, 6.0
    GRID_X, GRID_Y = 0.207, 0.487
    RIGHT = 0.37
    LINES_Y = (0.665, 0.735, 0.805)
    ROWS_Y = (0.67, 0.755, 0.84)
    THUMB, THUMB_X0, THUMB_DX = 1.1, 0.175, 0.043
    CHART = ((0.1, 0.45), (0.66, 0.855))
    YMAX = 1.0

    def setup(self):
        m = self.meta = self.study.metadata
        self.steps = m["steps"]
        self.loss = np.asarray(m["losses"])
        self.imgs = np.asarray(self.study.panels[1].data["values"], float).reshape(
            m["epochs"], FIRST_SHOWN, 28, 28)
        self.full_rows = self.steps // self.COLS            # 19 rows of 47
        self.rest = self.steps - self.full_rows * self.COLS  # and 45 in the last
        loader = ("loader",)
        bl = m["loader"].index(m["batch_line"])
        sh = m["loader"].index(m["shuffle_line"])
        pick = tuple(i for i, s in enumerate(m["loop"]) if ".to(device)" not in s)
        self.CODE = (Code(loader, run=((0.0, 0), (0.12, bl))),
                     Code("train", pick, run=((0.0, 1),)),
                     Code(loader, run=((0.0, sh),)),
                     Code("train", pick, run=((0.0, 0),)))
        self.EVIDENCE = (exact("%s ÷ %d" % (commas(m["train"]), m["batch"])),
                         exact("every digit once an epoch"),
                         exact("each epoch's first digits"),
                         measured("three epochs of the run"))
        n = len(self.loss)
        self.chart_ = self.chart(self.CHART, [0, n], [0.0, self.YMAX], pad=0.0)

    def heading_for(self, k, u):
        m = self.meta
        return ("batch_size=64: how many batches?", "one epoch: %d steps" % m["steps"],
                "shuffle=True: a new order", "loss per step, loss per epoch")[k]

    def subtitle_for(self, k):
        m = self.meta
        e = m["epoch_mean"]
        f = m["first"]
        return [
            "60,000 digits ÷ 64 = 937.5: so 937 batches of 64, and a last one of 32.",
            "One epoch is %d steps, one per batch: each of the 60,000 digits is used once."
            % m["steps"],
            "Each epoch the loader shuffles again: epoch 1 starts with digit %s, epoch 2 "
            "with %s." % (commas(f[0][0]), commas(f[1][0])),
            "Three epochs (the script runs one): per step the loss jumps; per epoch it falls, "
            "%.3f, %.3f, %.3f." % tuple(e),
        ][k]

    # ------------------------------------------------------------------ the grid
    def grid_cells(self):
        w, h = self.COLS * self.TILE, self.ROWS * self.TILE
        return self.cells(self.GRID_X, self.GRID_Y, w, h, self.ROWS, self.COLS), w, h

    def draw_grid(self, sc, used, op):
        """The 938 batches as tiles, row by row: ``used`` of them deep (seen this epoch)."""
        cell, w, h = self.grid_cells()
        vals = np.zeros(self.ROWS * self.COLS)
        vals[:self.steps] = 0.42
        vals[:int(used)] = 1.0
        vals = vals.reshape(self.ROWS, self.COLS)
        fr = self.full_rows
        top = vals[:fr]
        th = fr * self.TILE
        _fx, fy0 = cell(0, 0)
        dy = self.frac(0, self.TILE)[1]
        ytop = fy0 - dy / 2
        self.rect(sc, self.P(self.GRID_X, self.GRID_Y), w + 10, h + 10, PAPER, op,
                  border=dark(DATA, 0.2), border_w=1.4, radius=4, lift=0.1)
        self.tiles(sc, top, self.GRID_X, ytop + self.frac(0, th / 2)[1], w, th, op, lo=0, hi=1,
                   grid=0.35)
        # the last row: 45 tiles, the last of them half full (32 digits)
        x0 = self.GRID_X - self.frac(w / 2)[0]
        rw = self.rest * self.TILE
        last = vals[fr, :self.rest].copy()
        self.tiles(sc, last[None], x0 + self.frac(rw / 2)[0], ytop + self.frac(
            0, th + self.TILE / 2)[1], rw, self.TILE, op, lo=0, hi=1, grid=0.35)
        return cell

    def steady(self, sc, k, u, op):
        on = op * (smoothstep(0.06, 0.12, u) if k == 0 else 1.0)
        used = 0
        if k == 1:
            used = self.steps * eased(u, 0.12, 0.82)
        elif k > 1:
            used = self.steps
        cell = self.draw_grid(sc, used, on)
        ring = op * (smoothstep(0.62, 0.68, u) if k == 0 else 1.0) * self.kept(0, 1)
        r, c = self.full_rows, self.rest - 1
        x, y = cell(r, c)
        self.rect(sc, self.P(x, y), self.TILE + 6, self.TILE + 6, INK, ring, border=INK,
                  border_w=2.2, radius=2, lift=0.9, fill_op=0.0)
        self.text(sc, x + 0.008, y + 0.045, "the last: 32", ring, SIZE, "rm", strong=True)
        # what one tile is, all along
        key = op * (smoothstep(0.12, 0.18, u) if k == 0 else 1.0)
        self.text(sc, self.RIGHT + 0.01, 0.375, "one tile:", key, SIZE, "lm", ink=DATA_INK)
        self.text(sc, self.RIGHT + 0.01, 0.41, "64 digits", key, SIZE, "lm", ink=DATA_INK)
        # the steps of one epoch, all along
        total = op * (smoothstep(0.72, 0.78, u) if k == 0 else 1.0)
        self.big(sc, self.RIGHT + 0.01, 0.525, str(self.steps), total, label="steps", size=BIG)
        self.text(sc, self.RIGHT + 0.01, 0.58, "= 1 epoch",
                  op * self.kept(1, 2) * (smoothstep(0.86, 0.92, u) if k == 1 else 1.0), SIZE,
                  "lm", strong=True)
        self.text(sc, self.RIGHT + 0.01, 0.58, "× 3 = %s" % commas(self.meta["total_steps"]),
                  op * self.kept(3) * (smoothstep(0.62, 0.68, u) if k == 3 else 1.0), SIZE, "lm",
                  strong=True)

    # ------------------------------------------------------------------ 1: 60,000 ÷ 64
    def phase_1(self, sc, u, op):
        m = self.meta
        x0 = self.GRID_X - self.frac(self.COLS * self.TILE / 2)[0]
        rows = (m["division"], m["check"], "937 + 1 = %d steps" % m["steps"])
        for i, (s, y) in enumerate(zip(rows, self.LINES_Y)):
            on = op * smoothstep(0.24 + 0.2 * i, 0.3 + 0.2 * i, u)
            self.text(sc, x0, y, s, on, HEAD, "lm", strong=i == 2)

    # ------------------------------------------------------------------ 2: one epoch
    def phase_2(self, sc, u, op):
        x0 = self.GRID_X - self.frac(self.COLS * self.TILE / 2)[0]
        p = eased(u, 0.12, 0.82)
        on = op * smoothstep(0.04, 0.1, u)
        steps = int(round(self.steps * p))
        seen = min(steps * self.meta["batch"], self.meta["train"])
        self.big(sc, x0, 0.735, "%s" % commas(steps), on, label="steps", size=BIG)
        self.big(sc, x0 + 0.16, 0.735, commas(seen), on, label="digits used", size=BIG)
        self.text(sc, x0, 0.83, "each digit once", op * smoothstep(0.86, 0.92, u), HEAD, "lm",
                  strong=True)

    # ------------------------------------------------------------------ 3: a new order
    def phase_3(self, sc, u, op):
        for e, y in enumerate(self.ROWS_Y):
            on = op * smoothstep(0.06 + 0.24 * e, 0.12 + 0.24 * e, u)
            self.text(sc, 0.062, y, "epoch %d" % (e + 1), on, SIZE, "lm", strong=True)
            side = 28 * self.THUMB
            for j in range(FIRST_SHOWN):
                x = self.THUMB_X0 + self.THUMB_DX * j
                self.card(sc, x, y, side + 6, side + 8, on, lift=0.2)
                self.tiles(sc, self.imgs[e, j], x, y + self.frac(0, 1.5)[1], side, side, on,
                           lo=0.0, hi=1.0, grid=0.0, lift=0.25)

    # ------------------------------------------------------------------ 4: per epoch
    def phase_4(self, sc, u, op):
        ch = self.chart_
        n = len(self.loss)
        s = self.steps
        on = op * smoothstep(0.03, 0.09, u)
        self.loss_axes(sc, ch, on, xticks=[(0, "0"), (s, str(s)), (2 * s, commas(2 * s)),
                                           (3 * s, commas(3 * s))],
                       yticks=[(0, "0"), (0.4, "0.4"), (0.8, "0.8")])
        self.text(sc, 0.058, ch.y0 - 0.022, "loss", on, SIZE, "lm", ink=LOSS_INK)
        upto = eased(u, 0.08, 0.42) * (n - 1)
        self.curve(sc, ch, np.arange(n), self.loss, soft(LOSS, 0.45), on * 0.45, upto, width=0.7,
                   dot=False)
        clip = op * smoothstep(0.1, 0.16, u)
        self.off_top(sc, ch, self.loss, clip, "↑ from %.2f" % self.meta["first_loss"])
        means = self.meta["epoch_mean"]
        for e, mean in enumerate(means):
            a = 0.5 + 0.13 * e
            m_on = op * smoothstep(a, a + 0.06, u)
            if m_on <= 0.01:
                continue
            xa, xb, y = ch.fx(e * s), ch.fx((e + 1) * s), ch.fy(mean)
            self.polyline(sc, [self.P(xa, y), self.P(xb, y)], 6.0, dark(LOSS, 0.45), m_on,
                          bias=1.0)
            self.box(sc, (xa + xb) / 2, ch.y0 - 0.024, 50.0, 22.0, PAPER, m_on, text="%.3f" % mean,
                     border=LOSS_INK, size=SIZE)
            if e:
                ch.mark(sc, self.P, e * s, AXIS, m_on)


ANIMATIONS = {"blocks-adam": AdamBlocks, "blocks-the-training-loop": TrainingLoop,
              "blocks-epochs-and-batches": EpochsBatches}

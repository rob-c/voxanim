"""Animations 212-214, the last three of the "Blocks and code" series (see
:mod:`voxanim.blocks`): the whole picture, drawn from :mod:`voxanim.study_blocks_j`.

* :class:`OneEpochFast` (``blocks-one-epoch-fast``): the 938 steps of one epoch sped up:
  four of the first layer's units, as 28 x 28 orange tiles, turn from noise into strokes
  while the loss of each step falls and the held-out accuracy rises; then the same four
  before and after, and how alike next-door weights became.
* :class:`WholeProgram` (``blocks-the-whole-program``): the whole script, 63 lines, on a map
  beside the picture, the code panel scrolling through it a part at a time; each part of
  the picture appears and lights while its lines run, ``.to(device)`` included.
* :class:`CommonMistakes` (``blocks-common-mistakes``): four broken programs, each with its
  real run: ``zero_grad`` forgotten, ``model.eval()`` forgotten with dropout, softmax put in
  front of the loss, and no ``nn.Flatten`` with torch's own error in plain words.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). What stays across phases is drawn in
``steady``, so nothing blanks between them.
"""

from __future__ import annotations

import numpy as np

from . import layout
from . import study_course as course
from .blocks import (
    DATA,
    DATA_HI,
    DATA_LO,
    GRAD,
    HEAD,
    INK,
    LOOP,
    LOSS,
    OPTIM,
    QUIET,
    SIZE,
    WEIGHT,
    Blocks,
    Code,
    _merge,
    _pick,
    _Timeline,
    commas,
    exact,
    measured,
    val,
)
from .easing import ease_in_out, smoothstep
from .losschart import LossChart
from .mech import AXIS, PAPER, dark, soft
from .study_blocks_j import WINDOWS, pct

DATA_INK = dark(DATA, 0.3)
WEIGHT_INK = dark(WEIGHT, 0.35)
LOSS_INK = dark(LOSS, 0.25)
GRAD_INK = dark(GRAD, 0.25)
OPTIM_INK = dark(OPTIM, 0.35)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


class Whole(Blocks):
    """What the three share."""

    def chart(self, box, x_range, y_range):
        return LossChart(box, x_range, y_range)

    def unit(self, sc, w, fx, fy, tile, op, scale, lift=0.3):
        """One unit of the first layer: its 784 weights as a 28 x 28 picture of orange
        tiles (deeper: bigger; the same scale all through), in a thin frame."""
        side = 28 * tile
        self.rect(sc, self.P(fx, fy), side + 6, side + 6, PAPER, op, border=WEIGHT_INK,
                  border_w=1.4, radius=3, lift=lift - 0.05)
        self.tiles(sc, np.reshape(w, (28, 28)), fx, fy, side, side, op, ink=WEIGHT,
                   lo=-scale, hi=scale, grid=0.0, lift=lift)


# ====================================================================== 212
class OneEpochFast(Whole):
    """One epoch at speed; the strokes; the loss and the accuracy."""

    TITLE = "One epoch, fast"
    PHASES = (("one epoch", 11.0), ("sped up", 20.0), ("strokes", 13.0),
              ("the result", 12.0))
    LAST = "one step, in slow motion"
    NEXT = "A dense layer learns to be local"
    STATION = ("data", "step", "step", "loss")
    EVIDENCE = (measured("the course's run, before step 0"),
                measured("938 steps of the course's run"),
                measured("first-layer weights, steps 0 and 938"),
                measured("the loss, and 10,000 held-out digits"))
    TILE = 3.1
    ROW = (0.068, 0.455)            # the units' row: its left, its centre's y
    GAP = 10.0                      # px between two units
    LINE_Y = 0.585                  # the line under the units: what they are, the step
    LOSS_BOX = ((0.105, 0.245), (0.685, 0.865))
    ACC_BOX = ((0.335, 0.465), (0.685, 0.865))
    RUN = (0.04, 0.9)               # phase 2: the counter runs from 0 to 938 between these
    FLIP = (0.12, 0.2, 0.46, 0.54)  # phase 3: to step 0 and back

    def setup(self):
        m = self.meta = self.study.metadata
        self.keep = np.asarray(m["keep"])
        self.weights = np.asarray(m["weights"], float)
        self.alike = np.asarray(m["alike"])
        self.loss = np.asarray(m["loss"])
        self.evals = np.asarray(m["eval_steps"], float)
        self.acc = np.asarray(m["accuracy"])
        self.steps = m["steps"]
        self.scale = m["scale"]
        side = 28 * self.TILE
        x0, y = self.ROW
        dx = self.frac(side + self.GAP)[0]
        hx = self.frac(side / 2)[0]
        self.centres = [(x0 + hx + dx * i, y) for i in range(4)]
        self.right = x0 + dx * 3 + 2 * hx
        self.CODE = (Code("train", pick=tuple(range(8)), run=((0.0, 0), (0.45, 1))),
                     Code("train", pick=tuple(range(8)), run=((0.0, 1),)),
                     Code("train", pick=tuple(range(8)), run=((0.0, 7),)),
                     Code("train", pick=tuple(range(1, 9)), run=((0.0, 7),),
                          out=((0.2, m["printed"]),)))
        self.loss_chart = self.chart(self.LOSS_BOX, (0.0, float(self.steps)), (0.0, 2.5))
        self.acc_chart = self.chart(self.ACC_BOX, (0.0, float(self.steps)), (0.0, 1.0))

    def heading_for(self, k, u):
        if k == 2:
            a, _b, c, _d = self.FLIP
            return ("the weights turn into strokes", "at the start: random noise",
                    "after one epoch: strokes")[(u >= a) + (u >= c)]
        if k == 0:
            return ("one epoch: 938 steps, every digit once" if u < 0.62 else
                    "60,000 digits ÷ 64 → 938 steps")
        return ("", "938 steps, sped up", "", "the loss falls, the accuracy rises")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "One epoch is %d steps of %d digits: all %s once. The weights start as random "
            "noise." % (m["steps"], m["batch"], commas(m["train"])),
            "Sped up: in %d steps the loss falls from %s to %s and the held-out accuracy "
            "climbs to %s." % (m["steps"], val(m["first_loss"]), val(m["last_loss"]),
                               pct(m["accuracy_end"])),
            "Before and after: next-door weights went from unrelated (%s) to alike (%s)."
            % (val(m["alike_start"], 1), val(m["alike_end"], 1)),
            "After one epoch the last batch's loss is %s, and %s of 10,000 held-out digits "
            "are named right." % (val(m["last_loss"]), pct(m["accuracy_end"])),
        ][k]

    # ------------------------------------------------------------------ the clock
    def step_at(self, k, u):
        """The training step the clock has reached (0..938)."""
        if k == 0:
            return 0.0
        if k == 1:
            v = float(np.clip((u - self.RUN[0]) / (self.RUN[1] - self.RUN[0]), 0.0, 1.0))
            return self.steps * v * v
        return float(self.steps)

    def back(self, k, u):
        """Phase 3 shows the units at step 0 again for a while: how far (0..1)."""
        if k != 2:
            return 0.0
        a, b, c, d = self.FLIP
        return smoothstep(a, b, u) * (1.0 - smoothstep(c, d, u))

    def shown_step(self, k, u):
        """The step whose weights the units show (a kept step), as the counter reads it."""
        if self.back(k, u) > 0.5:
            return 0
        s = self.step_at(k, u)
        return int(self.keep[int(np.searchsorted(self.keep, s, side="right") - 1)])

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        s = self.step_at(k, u)
        i = int(np.searchsorted(self.keep, s, side="right") - 1)
        back = self.back(k, u)
        for j, (fx, fy) in enumerate(self.centres):
            self.unit(sc, self.weights[i, j], fx, fy, self.TILE, op * (1 - back), self.scale)
            if back > 0.01:
                self.unit(sc, self.weights[0, j], fx, fy, self.TILE, op * back, self.scale,
                          lift=0.5)
        x0 = self.ROW[0]
        # what the tiles are; from the strokes on, how alike next-door weights became
        what = op * (1.0 - (smoothstep(0.6, 0.66, u) if k == 2 else float(k > 2)))
        self.text(sc, x0, self.LINE_Y, "first layer: 4 of %d units" % m["layer_units"], what,
                  SIZE, "lm", ink=WEIGHT_INK)
        if k >= 2:
            a = op * (smoothstep(0.66, 0.72, u) if k == 2 else 1.0)
            self.text(sc, x0, self.LINE_Y, "next-door alike: %s → %s" % (
                val(m["alike_start"], 1), val(m["alike_end"], 1)), a, SIZE, "lm",
                ink=WEIGHT_INK)
        # the step counter, never blank
        self.text(sc, self.right, self.LINE_Y, "step %s of %d" % (
            commas(self.shown_step(k, u) if k != 1 else s), self.steps), op, HEAD, "rm",
            strong=True)
        # the two charts, drawn as far as the clock has come
        on = op * self.kept(0)
        upto = s / self.steps
        lc, ac = self.loss_chart, self.acc_chart
        self.axes(sc, lc, on, ylabel="loss", xticks=[(0, "0"), (938, "938")],
                  yticks=[(0, "0"), (1, "1"), (2, "2")], grid=False)
        self.axes(sc, ac, on, ylabel="held-out accuracy", xticks=[(0, "0"), (938, "938")],
                  yticks=[(0, "0%"), (1, "100%")], grid=False)
        self.plot(sc, lc, np.arange(len(self.loss)), self.loss, LOSS, on, upto, width=1.0,
                  dot=k == 1)
        self.plot(sc, ac, self.evals, self.acc, DATA, on, upto, width=1.3, dot=k == 1)
        # where they started, from the first phase on
        st = on * (smoothstep(0.5, 0.58, u) if k == 0 else 1.0)
        self.text(sc, lc.fx(0) + 0.008, lc.fy(self.loss[0]) - 0.002, val(m["first_loss"]), st,
                  SIZE, "lm", ink=LOSS_INK)
        self.text(sc, ac.fx(0) + 0.008, ac.fy(self.acc[0]) - 0.022, pct(m["accuracy_start"]),
                  st, SIZE, "lm", ink=DATA_INK)

    # ------------------------------------------------------------------ 1: one epoch
    def phase_1(self, sc, u, op):
        """The units at step 0, the counter and the empty charts, in ``steady``; the
        heading says how many steps an epoch is."""

    # ------------------------------------------------------------------ 2: sped up
    def phase_2(self, sc, u, op):
        """The counter, the units and the curves all run off one clock (``steady``)."""

    # ------------------------------------------------------------------ 3: strokes
    def phase_3(self, sc, u, op):
        """The units flip back to step 0 and forward again (``steady``); the heading says
        which is which."""

    # ------------------------------------------------------------------ 4: the result
    def phase_4(self, sc, u, op):
        m = self.meta
        lc, ac = self.loss_chart, self.acc_chart
        a = op * smoothstep(0.1, 0.18, u)
        self.text(sc, lc.x1, lc.fy(m["last_loss"]) - 0.035, val(m["last_loss"]), a, SIZE, "rm",
                  ink=LOSS_INK)
        self.text(sc, ac.x1, ac.fy(m["accuracy_end"]) + 0.035, pct(m["accuracy_end"]),
                  op * smoothstep(0.3, 0.38, u), SIZE, "rm", ink=DATA_INK)


# ====================================================================== 213
PART_INK = dict(imports=LOOP, device=dark(AXIS, 0.3), data=DATA, batches=DATA, model=WEIGHT,
                loss=LOSS, optimizer=OPTIM, train=LOOP, grad=GRAD, test=DATA, save=WEIGHT)


class WholeProgram(Whole):
    """The program's map, the panel scrolling down it, and the picture each part makes."""

    TITLE = "The whole program"
    PHASES = (("device", 6.5), ("data", 7.0), ("batches", 7.0), ("model", 7.5), ("loss", 6.5),
              ("train", 13.0), ("test", 7.0), ("save", 6.5))
    LAST = "one epoch, sped up: 938 steps"
    NEXT = "Training on a CPU and on a GPU"
    # the running line of each window, and when it gets there
    RUNS = (((0.0, 0), (0.3, 1), (0.55, 2)),
            ((0.0, 0), (0.35, 4), (0.65, 6)),
            ((0.0, 0), (0.3, 2), (0.6, 6)),
            ((0.0, 0), (0.55, 7)),
            ((0.0, 1), (0.4, 2), (0.75, 6)),
            ((0.0, 1), (0.15, 2), (0.3, 3), (0.43, 4), (0.56, 5), (0.66, 6), (0.78, 7)),
            ((0.0, 0), (0.25, 3), (0.4, 4), (0.5, 5), (0.75, 7)),
            ((0.0, 0), (0.3, 1), (0.6, 2), (0.8, 3)))
    # the loop map's station: per phase, or per running line of it
    STATIONS = (None, "data", "data", "forward", {1: "loss", 2: "step", 6: "step"},
                {1: "data", 2: "data", 3: "forward", 4: "loss", 5: "backward", 6: "backward",
                 7: "step"}, "forward", None)
    MINI = (0.062, 0.378, 0.858)    # the program's map: left, first and last line's y
    CHAR = 1.5                      # px per character on the map
    COL = 0.228                     # the data column's centre
    TRAIN_Y, STAGE_Y, TEST_Y = 0.43, 0.635, 0.80
    DEV = (0.3, 0.47, 0.56, 0.87)   # the device: left, right, top, bottom
    MODEL_OUT, MODEL_IN = (0.385, 0.44), (0.385, 0.655)
    SLOT = (0.335, 0.795)           # where a batch sits on the device
    METER, HAND = (0.325, 0.44), (0.44, 0.445)
    TILE = 4.0

    def setup(self):
        m = self.meta = self.study.metadata
        self.lines = m["program"]
        self.parts = [tuple(p) for p in m["parts"]]
        self.windows = [w["index"] for w in m["windows"]]
        self.CODE = tuple(Code(regs, pick=pick, run=run) for (regs, pick), run in
                          zip(WINDOWS, self.RUNS))
        self.CODE = tuple(c if k not in (2, 6) else Code(c.region, c.pick, c.run, out)
                          for k, (c, out) in enumerate(zip(self.CODE, (
                              None, None, ((0.12, m["printed_digit"]),), None, None, None,
                              ((0.8, m["printed_accuracy"]),), None))))
        self.part_of = {}
        for name, a, b in self.parts:
            for i in range(a, b + 1):
                self.part_of[i] = name
        self.tiles_ = [np.asarray(t, float) for t in m["model_tiles"]]
        self.grads_ = [np.asarray(t, float) for t in m["grad_tiles"]]
        self.bounds = m["model_bounds"]
        b = course.first_batch()
        self.front = np.asarray(b["x"][0, 0], float)
        self.test_front = np.asarray(course.data()[2][0].reshape(28, 28), float)
        self.EVIDENCE = (exact("the program's own lines"),
                         exact("60,000 + 10,000 digits, counted"),
                         exact("938 batches of 64 (the last: 32)"),
                         exact("the model's %s numbers, counted" % commas(m["parameters"])),
                         exact("the program's own lines"),
                         measured("the course's run: first and last step"),
                         measured("10,000 held-out digits"),
                         exact("%s numbers, saved and loaded" % commas(m["parameters"])))

    def T(self, k, u):
        """The time of ``u`` in phase ``k``."""
        a, b = self.spans[k]
        return a + (b - a) * u

    def heading_for(self, k, u):
        return ("where the numbers live: the device", "the data: digits and labels",
                "batches of 64", "the model, then .to(device)", "the loss and the optimizer",
                "the training loop: 938 times round", "the test: digits never trained on",
                "save the numbers, load them back")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "After the imports, device is the GPU (\"cuda\") if there is one, otherwise the "
            "CPU.",
            "The data: %s training digits and %s test digits, each turned into normalised "
            "numbers." % (commas(m["train"]), commas(m["test"])),
            "The loader hands out the training digits %d at a time, in a new order every "
            "epoch." % m["batch"],
            "The model is built block by block; .to(device) moves its %s numbers to the "
            "device." % commas(m["parameters"]),
            "The loss measures how wrong the model is; the optimizer will move its weights."
            ,
            "Each step moves a batch to the device, scores it and nudges the weights: "
            "loss %s, then %s after %d steps." % (val(m["first_loss"]), val(m["last_loss"]),
                                                 m["steps"]),
            "On the %s test digits it never trained on, it names %s right: %s."
            % (commas(m["test"]), commas(m["correct"]), m["printed_accuracy"]),
            "state_dict() holds all %s numbers; torch.save writes them to a file, and "
            "torch.load reads them back." % commas(m["parameters"]),
        ][k]

    # ------------------------------------------------------------------ the running line
    def run_line(self, k, u):
        """The running line of phase ``k``: its index in the window, and in the program."""
        line = _pick(self.RUNS[k], u)
        return line, self.windows[k][line]

    def station_at(self, t):
        k, u = self.where(t)
        s = self.STATIONS[k]
        if isinstance(s, dict):
            line, _i = self.run_line(k, u)
            return s.get(line, s[min(s)])
        return s

    def station_for(self, k):
        s = self.STATIONS[k]
        return s[min(s)] if isinstance(s, dict) else s

    def _timelines(self):
        super()._timelines()
        ts = np.arange(0.0, self.duration + self.TICK, self.TICK)
        self._tl_station = _Timeline(_merge((float(t), self.station_at(float(t))) for t in ts))

    def step_indicator(self, sc, current, phases):
        """Eight phases: the one under way by name, the others by number."""
        items = ["%d %s" % (n + 1, name) if n == current else "%d" % (n + 1)
                 for n, name in enumerate(phases)]
        size, gap = 0.030, 0.016
        widths = [layout.text_width(s, size) for s in items]
        x = (1 - layout.SAFE) - (sum(widths) + gap * (len(items) - 1)) - 0.004
        for n, (s, w) in enumerate(zip(items, widths)):
            col = (0.12, 0.36, 0.66) if n == current else layout.INK_SOFT if n < current \
                else (0.56, 0.58, 0.60)
            sc.text(x, layout.SAFE + 0.012, s, size, "regular", col, 1.0, "la")
            x += w + gap

    # ------------------------------------------------------------------ the map
    def line_y(self, i):
        _x0, y0, y1 = self.MINI
        return y0 + (y1 - y0) * i / (len(self.lines) - 1)

    def program_map(self, sc, k, u, op):
        """Every line of the program as a bar of its length, in its part's colour: lit once
        it has run; the panel's window bracketed; the running line in ink."""
        x0, _y0, _y1 = self.MINI
        _line, now = self.run_line(k, u)
        start = self.windows[k][0]
        for i, s in enumerate(self.lines):
            indent = len(s) - len(s.lstrip())
            n = len(s.strip())
            w = n * self.CHAR
            fx = x0 + self.frac(indent * self.CHAR + w / 2)[0]
            ink = PART_INK[self.part_of[i]]
            ran = i < start or i <= now
            fill = INK if i == now else ink if ran else soft(ink, 0.65)
            self.rect(sc, self.P(fx, self.line_y(i)), w, 3.0, fill, op, radius=1.4, lift=0.2)
        # the bracket: the lines in the panel, sliding from one window to the next
        cur = self.windows[k]
        prev = self.windows[k - 1] if k > 0 else cur
        s = eased(u, 0.0, 0.12)
        a = self.line_y(prev[0]) + (self.line_y(cur[0]) - self.line_y(prev[0])) * s
        b = self.line_y(prev[-1]) + (self.line_y(cur[-1]) - self.line_y(prev[-1])) * s
        pad = self.frac(0, 4.0)[1]
        wpx = 41 * self.CHAR + 10
        self.rect(sc, self.P(x0 + self.frac(wpx / 2 - 5)[0], (a + b) / 2), wpx,
                  self.span(0, b - a + 2 * pad)[1], INK, op, border=INK, border_w=1.6,
                  radius=3, lift=0.4, fill_op=0.0)
        self.text(sc, x0, 0.893, "all %d lines" % len(self.lines), op, SIZE, "lm", ink=QUIET)

    # ------------------------------------------------------------------ the picture
    def appear(self, k, u):
        t = self._t
        return smoothstep(self.T(k, u), self.T(k, u) + 0.5, t)

    def glow(self, k0, u0, k1, u1):
        """1 while the running line is between ``(k0, u0)`` and ``(k1, u1)``."""
        t = self._t
        a, b = self.T(k0, u0), self.T(k1, u1)
        return smoothstep(a, a + 0.3, t) * (1.0 - smoothstep(b - 0.3, b, t))

    def ring(self, sc, fx, fy, w, h, g, op):
        if g * op > 0.01:
            self.rect(sc, self.P(fx, fy), w, h, INK, op * g, border=INK, border_w=2.2,
                      radius=6, lift=1.2, fill_op=0.0)

    def pile(self, sc, fx, fy, op, name, count, front):
        self.stack(sc, fx, fy, 40.0, 46.0, 4, op, step=(3.0, -3.0))
        self.tiles(sc, front, fx, fy + self.frac(0, 2.0)[1], 30.0, 30.0, op, lo=DATA_LO,
                   hi=DATA_HI, lift=0.25, grid=0.0)
        self.text(sc, fx, fy + 0.07, name, op, SIZE, ink=DATA_INK)
        self.text(sc, fx, fy + 0.105, count, op, SIZE, ink=DATA_INK)

    def model_at(self, sc, fx, fy, op, built=(1.0, 1.0, 1.0), grads=0.0):
        """The model: its three weight grids side by side (their corners), the gradients'
        red tiles behind them when ``grads``."""
        widths = [t.shape[1] * self.TILE for t in self.tiles_]
        gap = 7.0
        total = sum(widths) + gap * 2
        x = fx - self.frac(total / 2)[0]
        for t, g, b, w, on in zip(self.tiles_, self.grads_, self.bounds, widths, built):
            cx = x + self.frac(w / 2)[0]
            h = t.shape[0] * self.TILE
            if grads > 0.01:
                lim = float(np.percentile(np.abs(g), 75))
                self.grad_tiles(sc, np.abs(g), cx, fy, w, h, op * on * grads, offset=(6, 6),
                                lo=-0.3 * lim, hi=lim)
            self.tiles(sc, t, cx, fy, w, h, op * on, ink=WEIGHT, lo=-b, hi=b, grid=0.0,
                       lift=0.35)
            x += self.frac(w + gap)[0]
        self.text(sc, fx, fy + 0.06, "model", op * min(built), SIZE, ink=WEIGHT_INK)

    def steady(self, sc, k, u, op):
        m = self.meta
        self.program_map(sc, k, u, op)
        t = self._t
        # 1: the device
        dev = op * self.appear(0, 0.55)
        x0, x1, top, bottom = self.DEV
        w, h = self.span(x1 - x0, bottom - top)
        self.rect(sc, self.P((x0 + x1) / 2, (top + bottom) / 2), w, h, soft(AXIS, 0.85), dev,
                  border=dark(AXIS, 0.3), border_w=1.6 + 1.2 * self.glow(0, 0.55, 1, 0.0),
                  radius=10, lift=0.05, fill_op=0.5)
        self.text(sc, x0 + 0.012, top + 0.03, "device: GPU or CPU", dev, SIZE, "lm",
                  ink=INK)
        # 2: the data
        tr = op * self.appear(1, 0.35)
        self.pile(sc, self.COL, self.TRAIN_Y, tr, "train_set", commas(m["train"]), self.front)
        self.ring(sc, self.COL, self.TRAIN_Y + 0.035, 70, 104, self.glow(1, 0.35, 1, 0.65), tr)
        te = op * self.appear(1, 0.65)
        self.pile(sc, self.COL, self.TEST_Y, te, "test_set", commas(m["test"]), self.test_front)
        self.ring(sc, self.COL, self.TEST_Y + 0.035, 70, 104, self.glow(1, 0.65, 2, 0.0), te)
        # 3: a batch, out of the training set; 6: onto the device; 7: gone, the test batch
        b_on = op * self.appear(2, 0.6) * (1.0 - smoothstep(self.T(6, 0.0), self.T(6, 0.1), t))
        go = smoothstep(self.T(2, 0.6), self.T(2, 0.75), t)
        to = smoothstep(self.T(5, 0.15), self.T(5, 0.27), t)
        bx = self.COL + (self.SLOT[0] - self.COL) * ease_in_out(to)
        by = self.TRAIN_Y + (self.STAGE_Y - self.TRAIN_Y) * ease_in_out(go)
        by = by + (self.SLOT[1] - self.STAGE_Y) * ease_in_out(to)
        if b_on > 0.01:
            self.stack(sc, bx, by, 34.0, 40.0, 5, b_on, step=(2.0, -2.0), lift=0.5)
            self.tiles(sc, self.front, bx, by + self.frac(0, 2.0)[1], 26.0, 26.0, b_on,
                       lo=DATA_LO, hi=DATA_HI, lift=0.55, grid=0.0)
            self.text(sc, bx + 0.035, by, "64", b_on, SIZE, "lm", ink=DATA_INK)
        tb = op * self.appear(6, 0.25) * (1.0 - smoothstep(self.T(7, 0.0), self.T(7, 0.1), t))
        if tb > 0.01:
            s = smoothstep(self.T(6, 0.4), self.T(6, 0.52), t)
            fx = self.COL + (self.SLOT[0] - self.COL) * ease_in_out(s)
            self.stack(sc, fx, self.TEST_Y - 0.002, 34.0, 40.0, 5, tb, step=(2.0, -2.0),
                       lift=0.5)
            self.tiles(sc, self.test_front, fx, self.TEST_Y - 0.002 + self.frac(0, 2.0)[1],
                       26.0, 26.0, tb, lo=DATA_LO, hi=DATA_HI, lift=0.55, grid=0.0)
        # 4: the model, built block by block, then moved onto the device
        built = tuple(self.appear(3, a) for a in (0.08, 0.2, 0.32))
        mv = ease_in_out(smoothstep(self.T(3, 0.58), self.T(3, 0.75), t))
        mx = self.MODEL_OUT[0] + (self.MODEL_IN[0] - self.MODEL_OUT[0]) * mv
        my = self.MODEL_OUT[1] + (self.MODEL_IN[1] - self.MODEL_OUT[1]) * mv
        grads = self.appear(5, 0.66) * (1.0 - smoothstep(self.T(6, 0.0), self.T(6, 0.1), t))
        if max(built) > 0.01:
            self.model_at(sc, mx, my, op, built, grads)
            g = max(self.glow(3, 0.0, 3, 0.75), self.glow(5, 0.3, 5, 0.43),
                    self.glow(6, 0.5, 6, 0.75), self.glow(7, 0.0, 7, 0.3),
                    self.glow(7, 0.8, 7, 1.0))
            self.ring(sc, mx, my + 0.02, 160, 62, g, op)
        # 5: the loss and the optimizer
        me = op * self.appear(4, 0.0)
        loss = m["first_loss"] if t < self.T(5, 0.9) else m["last_loss"]
        fill = self.appear(5, 0.43) * (1.0 - smoothstep(self.T(5, 0.9), self.T(5, 1.0), t))
        fill += smoothstep(self.T(5, 0.9), self.T(5, 1.0), t) * m["last_loss"] / 2.5 / max(
            m["first_loss"] / 2.5, 1e-9)
        shown = loss if t >= self.T(5, 0.43) else None
        self.meter(sc, *self.METER, fill * m["first_loss"] / 2.5, me, h=60.0, w=18.0)
        self.text(sc, self.METER[0], self.METER[1] + 0.075, "loss_fn", me, SIZE,
                  ink=LOSS_INK)
        if shown is not None:
            self.text(sc, self.METER[0] + 0.025, self.METER[1], val(shown), me, SIZE, "lm",
                      ink=LOSS_INK)
        self.ring(sc, self.METER[0], self.METER[1] + 0.02, 56, 92,
                  max(self.glow(4, 0.0, 4, 0.4), self.glow(5, 0.43, 5, 0.56)), me)
        ha = op * self.appear(4, 0.4)
        reach = smoothstep(self.T(5, 0.78), self.T(5, 0.84), t) * (
            1.0 - smoothstep(self.T(5, 0.88), self.T(5, 0.94), t))
        hx = self.HAND[0] + (self.MODEL_IN[0] + 0.05 - self.HAND[0]) * reach
        hy = self.HAND[1] + (self.MODEL_IN[1] - 0.06 - self.HAND[1]) * reach
        self.hand(sc, hx, hy, ha, size=0.9)
        self.text(sc, self.HAND[0], self.HAND[1] + 0.07, "optimizer", ha * (1 - reach), SIZE,
                  ink=OPTIM_INK)
        self.ring(sc, self.HAND[0], self.HAND[1] + 0.02, 80, 92,
                  max(self.glow(4, 0.4, 4, 0.75), self.glow(5, 0.56, 5, 0.66)), ha)

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        """The device box, in ``steady``."""

    def phase_2(self, sc, u, op):
        """The two piles of digits, in ``steady``."""

    def phase_3(self, sc, u, op):
        """A batch, in ``steady``."""

    def phase_4(self, sc, u, op):
        """The model, in ``steady``."""

    def phase_5(self, sc, u, op):
        """The loss and the optimizer, in ``steady``."""

    def phase_6(self, sc, u, op):
        on = op * smoothstep(0.9, 0.95, u)
        _x0, x1, _top, bottom = self.DEV
        self.text(sc, x1 - 0.012, bottom - 0.03, "× %d steps" % self.meta["steps"], on, SIZE,
                  "rm", ink=INK)

    def phase_7(self, sc, u, op):
        m = self.meta
        on = op * smoothstep(0.75, 0.82, u)
        _x0, x1, _top, bottom = self.DEV
        self.text(sc, x1 - 0.012, bottom - 0.03, "%s right" % commas(m["correct"]), on, SIZE,
                  "rm", ink=DATA_INK)

    def phase_8(self, sc, u, op):
        m = self.meta
        on = op * smoothstep(0.3, 0.38, u)
        fx, fy = self.COL, self.STAGE_Y
        self.rect(sc, self.P(fx, fy), 76, 44, soft(WEIGHT, 0.8), on, border=WEIGHT_INK,
                  border_w=1.6, radius=5, lift=0.3)
        self.text(sc, fx, fy - 0.012, "mnist.pt", on, SIZE, ink=WEIGHT_INK)
        self.text(sc, fx, fy + 0.022, commas(m["parameters"]), on, SIZE, ink=WEIGHT_INK)
        left = self.DEV[0] + 0.008
        self.arrow(sc, (left, fy - 0.015), (fx + 0.046, fy - 0.015), WEIGHT_INK, on,
                   eased(u, 0.3, 0.45), width=1.8)
        back = op * smoothstep(0.6, 0.68, u)
        self.arrow(sc, (fx + 0.046, fy + 0.02), (left, fy + 0.02), WEIGHT_INK, back,
                   eased(u, 0.6, 0.75), width=1.8)


# ====================================================================== 214
class CommonMistakes(Whole):
    """Four broken programs, then all four side by side."""

    TITLE = "Common mistakes"
    PHASES = (("no zero_grad", 12.5), ("no eval()", 12.5), ("softmax twice", 12.5),
              ("no Flatten", 12.5), ("all four", 8.0))
    LAST = "the whole program, part by part"
    NEXT = "The CNN, block by block"
    STATION = ("backward", "forward", "loss", "forward", "step")
    EVIDENCE = (measured("a run without optimizer.zero_grad()"),
                measured("the dropout model, tested two ways"),
                measured("a run with softmax before the loss"),
                exact("the shapes, and torch's own error"),
                measured("10,000 held-out digits, each run"))
    BOX = ((0.12, 0.44), (0.41, 0.63))      # the top picture's chart
    BAR_X, BAR_W = 0.215, 190.0             # the result bars: their left, 100% in px
    BAR_Y = (0.77, 0.835)

    def setup(self):
        m = self.meta = self.study.metadata
        ng, ne, ds, nf = m["no_zero_grad"], m["no_eval"], m["double_softmax"], m["no_flatten"]
        self.loss = np.asarray(m["loss"])
        self.used = np.asarray(ng["used"])
        self.fresh = np.asarray(ng["fresh"])
        self.ds_loss = np.asarray(ds["loss"])
        self.mask = np.asarray(ne["mask"], bool)
        err = nf["error"]
        cut = err.index(" shapes")
        cut2 = err.index(" (")
        self.error_lines = (err[:cut], err[cut + 1:cut2], err[cut2 + 1:])
        self.CODE = (
            Code("mistake-no-zero-grad", run=((0.0, 0), (0.15, 2))),
            Code("mistake-no-eval", run=((0.0, 0), (0.15, 2))),
            Code("mistake-double-softmax", run=((0.0, 0), (0.15, 5))),
            Code("mistake-no-flatten", run=((0.0, 0), (0.12, 1)),
                 out=((0.3, "\n".join(self.error_lines)),)),
            Code("step", run=((0.0, 3),)))
        self.grad_chart = self.chart(self.BOX, (0.0, 938.0), (0.0, 40.0))
        self.loss_chart = self.chart(self.BOX, (0.0, 938.0), (0.0, 2.5))
        right = m["right"]
        # the result bars of each phase: (label, accuracy or None: no result), twice
        self.rows = (
            (("with zero_grad", right), ("without", ng["accuracy"])),
            (("with eval()", ne["with_eval"]), ("without", ne["accuracy"])),
            (("one softmax", right), ("two", ds["accuracy"])),
            (("with Flatten", right), ("without", None)),
            (("the script", right), ("", None)),
        )

    def heading_for(self, k, u):
        return ("forgot optimizer.zero_grad()", "forgot model.eval() with dropout",
                "a softmax before the loss", "forgot nn.Flatten()",
                "four mistakes, measured")[k]

    def subtitle_for(self, k):
        m = self.meta
        ng, ne, ds, nf = m["no_zero_grad"], m["no_eval"], m["double_softmax"], m["no_flatten"]
        return [
            "Without zero_grad, .grad adds up every step: %s by the end, against %s for one "
            "batch. Accuracy falls to %s." % (val(ng["used"][-1], 1), val(ng["fresh"][-1]),
                                               ng["text"]),
            "Without model.eval(), dropout still switches 1 in 5 off while testing: %s "
            "instead of %s." % (ne["text"], ne["with_eval_text"]),
            "The loss already does a softmax. Done twice, the loss cannot go below %s, and "
            "accuracy is %s." % (ds["floor_text"], ds["text"]),
            "Without Flatten, nn.Linear(784, 64) gets rows of %d numbers, not %d, and "
            "stops with an error." % (nf["got"], nf["wants"]),
            "The script itself: %s. Each mistake costs accuracy, or stops the program."
            % m["right_text"],
        ][k]

    # ------------------------------------------------------------------ what stays
    def result_bars(self, sc, k, u, op):
        """Held-out accuracy, the right program against the broken one: the bars swap
        their numbers in place from one mistake to the next."""
        rows = self.rows[k]
        prev = self.rows[k - 1] if k > 0 else rows
        grow = eased(u, 0.0, 0.12) if k > 0 else 1.0
        self.text(sc, self.BAR_X, self.BAR_Y[0] - 0.05, "held-out accuracy", op, SIZE, "lm")
        for r, ((label, v), (_l0, v0)) in enumerate(zip(rows, prev)):
            y = self.BAR_Y[r]
            row_op = op if r == 0 or k < 4 else op * (1 - smoothstep(0.0, 0.1, u))
            if row_op <= 0.01:
                continue
            show = op * (smoothstep(0.45, 0.53, u) if (r == 1 and k < 4) else 1.0)
            ink = DATA if r == 0 else dark(DATA, 0.45)
            self.text(sc, self.BAR_X - 0.012, y, label, row_op * (show if r else 1.0), SIZE,
                      "rm")
            a = 0.0 if v0 is None else v0
            b = 0.0 if v is None else v
            length = self.BAR_W * (a + (b - a) * grow)
            self.rect(sc, self.P(self.BAR_X + self.frac(self.BAR_W / 2)[0], y), self.BAR_W,
                      16.0, soft(DATA, 0.85), row_op * 0.6, radius=4, lift=0.2)
            if length > 1.0:
                self.hbar(sc, self.BAR_X, y, length, ink, row_op * (show if r else 1.0),
                          value=pct(v) if v is not None else "", thick=16.0)
            elif v is None and label:
                self.text(sc, self.BAR_X + 0.01, y, "an error: no result", row_op * show,
                          SIZE, "lm", ink=GRAD_INK)

    def steady(self, sc, k, u, op):
        self.result_bars(sc, k, u, op)

    # ------------------------------------------------------------------ 1: zero_grad
    def phase_1(self, sc, u, op):
        c = self.grad_chart
        on = op * smoothstep(0.05, 0.12, u)
        self.axes(sc, c, on, ylabel="size of the gradient used",
                  xticks=[(0, "0"), (938, "938")], yticks=[(0, "0"), (20, "20"),
                                                                 (40, "40")], grid=False)
        draw = eased(u, 0.15, 0.45)
        steps = np.arange(len(self.used))
        self.plot(sc, c, steps, self.fresh, soft(GRAD, 0.45), on, draw, width=1.0, dot=False)
        self.plot(sc, c, steps, self.used, GRAD, on, draw, width=1.4, dot=False)
        lab = on * smoothstep(0.45, 0.52, u)
        self.text(sc, c.x1, c.fy(self.used[-1]) + 0.03, ".grad: %s" % val(self.used[-1], 1),
                  lab, SIZE, "rm", ink=GRAD_INK)
        self.text(sc, c.x1, c.fy(0) - 0.03, "one batch: %s" % val(self.fresh[-1]), lab, SIZE,
                  "rm", ink=GRAD_INK)

    # ------------------------------------------------------------------ 2: eval
    def phase_2(self, sc, u, op):
        ne = self.meta["no_eval"]
        n = len(self.mask)
        side = int(np.sqrt(n))
        vals = np.full((side, side), 0.7)
        w = 8 * 13.0
        for j, (fx, label, zero, note) in enumerate((
                (0.17, "model.eval()", None, "%d of %d on" % (n, n)),
                (0.37, "no eval()", self.mask.reshape(side, side),
                 "%d of %d off" % (ne["off"], n)))):
            on = op * smoothstep(0.08 + 0.2 * j, 0.16 + 0.2 * j, u)
            self.text(sc, fx, 0.405, label, on, SIZE, ink=INK)
            self.tiles(sc, vals, fx, 0.53, w, w, on, ink=DATA, lo=0.0, hi=1.0, zero=zero,
                       grid=0.3)
            self.text(sc, fx, 0.655, note, on, SIZE, ink=DATA_INK if j == 0 else INK)

    # ------------------------------------------------------------------ 3: softmax twice
    def phase_3(self, sc, u, op):
        ds = self.meta["double_softmax"]
        c = self.loss_chart
        on = op * smoothstep(0.05, 0.12, u)
        self.axes(sc, c, on, ylabel="loss", xticks=[(0, "0"), (938, "938")],
                  yticks=[(0, "0"), (1, "1"), (2, "2")], grid=False)
        steps = np.arange(len(self.loss))
        draw = eased(u, 0.15, 0.45)
        self.plot(sc, c, steps, self.loss, soft(LOSS, 0.4), on, draw, width=1.0, dot=False)
        self.plot(sc, c, steps, self.ds_loss, LOSS, on, draw, width=1.0, dot=False)
        fl = on * smoothstep(0.48, 0.56, u)
        c.hline(sc, self.P, ds["floor"], fl, dashes=16, width=1.4)
        self.text(sc, c.fx(120), c.fy(ds["floor"]) + 0.028,
                  "lowest possible: %s" % ds["floor_text"], fl, SIZE, "lm", ink=LOSS_INK)
        lab = on * smoothstep(0.45, 0.52, u)
        self.text(sc, c.x1, c.fy(0.95), "one softmax", lab, SIZE, "rm",
                  ink=dark(soft(LOSS, 0.3), 0.2))
        self.text(sc, c.x1, c.fy(ds["last_100"]) - 0.045, "softmax twice", lab, SIZE, "rm",
                  ink=LOSS_INK)

    # ------------------------------------------------------------------ 4: no Flatten
    def phase_4(self, sc, u, op):
        nf = self.meta["no_flatten"]
        x0, x1 = 0.08, 0.46
        full = x1 - x0
        w = self.span(full)[0]
        wants = op * smoothstep(0.08, 0.16, u)
        self.text(sc, x0, 0.42, "nn.Linear(784, 64) wants rows of 784", wants, SIZE, "lm")
        self.rect(sc, self.P((x0 + x1) / 2, 0.47), w, 16.0, WEIGHT, wants, border=WEIGHT_INK,
                  border_w=1.4, radius=3, fill_op=0.0, lift=0.3)
        gets = op * smoothstep(0.3, 0.38, u)
        short = full * nf["got"] / nf["wants"]
        self.text(sc, x0, 0.54, "without Flatten, a row is %d" % nf["got"], gets, SIZE, "lm")
        self.strip(sc, np.full(nf["got"], 0.6), x0 + short / 2, 0.59,
                   self.span(short)[0], 16.0, gets, lo=0.0, hi=1.0)
        rows = op * smoothstep(0.52, 0.6, u)
        self.text(sc, x0, 0.645, "%s rows of %d" % (nf["rows_sum"], nf["got"]), rows, SIZE,
                  "lm", ink=DATA_INK)

    # ------------------------------------------------------------------ 5: all four
    def phase_5(self, sc, u, op):
        m = self.meta
        items = (("zero_grad forgotten", m["no_zero_grad"]["text"]),
                 ("eval() forgotten", m["no_eval"]["text"]),
                 ("softmax twice", m["double_softmax"]["text"]),
                 ("no Flatten", "an error"))
        for i, (name, value) in enumerate(items):
            on = op * smoothstep(0.05 + 0.1 * i, 0.12 + 0.1 * i, u)
            y = 0.42 + 0.06 * i
            self.text(sc, 0.08, y, name, on, SIZE, "lm")
            self.text(sc, 0.44, y, value, on, SIZE, "rm", strong=True)


ANIMATIONS = {"blocks-one-epoch-fast": OneEpochFast, "blocks-the-whole-program": WholeProgram,
              "blocks-common-mistakes": CommonMistakes}

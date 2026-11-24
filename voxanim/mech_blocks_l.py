"""Animations 218-220 of the "Blocks and code" series (see :mod:`voxanim.blocks`), the CNN
part: drawn from :mod:`voxanim.study_blocks_l` (the CNN of ``examples/mnist_cnn.py``,
trained once in numpy: :mod:`voxanim.study_course_cnn`).

* :class:`StampGrad` (``blocks-cnn-backward``): ``loss.backward()`` puts red tiles behind
  the 8 orange stamps of the first ``nn.Conv2d``; ``.grad`` slides out beside them, the same
  shape ``[8, 1, 3, 3]`` (``print(model[0].weight.grad.shape)``); then one of its numbers,
  stamp 1's centre weight, is built from its parts: on one digit, three places worked out
  (pixel x red = part) and the other 673 swept; then one bar per digit, the 64 digits'
  sums, adding up to the number PyTorch leaves in ``.grad``.
* :class:`SameLoop` (``blocks-cnn-same-loop``): the MLP's model lines and the CNN's, block
  by block, with their counts; the six lines of a step, the same in both files; both loss
  curves on the same 938 batches (each point the mean of 25 steps, for both).
* :class:`StampsLearn` (``blocks-cnn-stamps-learn``): the 8 stamps at steps 0, 10, 100,
  500 and 938, a row each; then the 8 maps they make of one digit, a 5, with the stamps of
  step 0 and of step 938.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). What stays across phases is drawn in
``steady`` from absolute times, so nothing blanks between them.
"""

from __future__ import annotations

import numpy as np

from .blocks import (
    DATA,
    GRAD,
    HEAD,
    INK,
    LOOP,
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
    val,
)
from .diagram import draw_image
from .easing import ease_in_out, smoothstep
from .losschart import LossChart
from .mech import AXIS, PAPER, dark, soft

DATA_INK = dark(DATA, 0.3)
WEIGHT_INK = dark(WEIGHT, 0.35)
GRAD_INK = dark(GRAD, 0.25)
LOSS_INK = dark(LOSS, 0.3)
MLP_INK = dark(AXIS, 0.3)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def _pick(steps, u):
    """The value of the last ``(u0, value)`` of ``steps`` with ``u0 <= u``."""
    out = steps[0][1]
    for at, v in steps:
        if u >= at:
            out = v
    return out


class Timed(Blocks):
    """What the three share: absolute times, a labelled block box, an outline."""

    def at(self, k, u):
        """The time of ``u`` of phase ``k`` (from :attr:`PHASES`: usable in ``setup``)."""
        a = sum(seconds for _name, seconds in self.PHASES[:k])
        return a + self.PHASES[k][1] * u

    def since(self, t0, d=0.6):
        """0 before ``t0``, easing to 1 over ``d`` seconds."""
        return smoothstep(t0, t0 + d, getattr(self, "_t", 0.0))

    def ring(self, sc, fx, fy, w, h, op, ink=INK, width=2.4, radius=4, lift=1.2):
        if op <= 0.01:
            return
        self.rect(sc, self.P(fx, fy), w, h, ink, op, border=ink, border_w=width,
                  radius=radius, lift=lift, fill_op=0.0)

    def stamp(self, sc, values, fx, fy, side, op, ink=WEIGHT, lo=None, hi=None, lift=0.3,
              zero=None):
        """A 3 x 3 stamp as ``side`` px of tiles with a thin outline; returns ``cell``."""
        cell = self.tiles(sc, values, fx, fy, side, side, op, ink=ink, lo=lo, hi=hi,
                          lift=lift, grid=0.35, zero=zero)
        if op > 0.01:
            self.rect(sc, self.P(fx, fy), side + 3, side + 3, ink, op, border=ink,
                      border_w=1.4, radius=2, lift=lift + 0.01, fill_op=0.0)
        return cell

    def block_box(self, sc, name, fx, fy, w, h, op):
        """One block of a model as a labelled box: orange for one that holds weights
        (Linear, Conv2d), grey for ReLU (it makes zeros), plain for the others."""
        if op <= 0.01:
            return
        kind = name.split("(")[0]
        fill, border = {"Linear": (soft(WEIGHT, 0.6), WEIGHT),
                        "Conv2d": (soft(WEIGHT, 0.6), WEIGHT),
                        "ReLU": (soft(ZERO, 0.55), ZERO)}.get(kind, (PAPER, AXIS))
        self.rect(sc, self.P(fx, fy), w, h, fill, op, border=border, border_w=1.6, radius=6,
                  lift=0.2)
        self.text(sc, fx, fy, name, op, SIZE, ink=INK)


# ====================================================================== 218
class StampGrad(Timed):
    """Red tiles behind the stamps; .grad the stamps' shape; one number as a sum of parts."""

    TITLE = "A shared stamp's gradient"
    PHASES = (("loss.backward()", 11.0), ("the stamps' shape", 11.0), ("one place", 18.0),
              ("every digit", 16.0))
    LAST = "one digit through the CNN"
    NEXT = "the same training loop"
    STATION = ("backward",) * 4
    SX, PITCH, SIDE = 0.09, 0.048, 30.0      # the row of 8 stamps: first centre, pitch, px
    WROW, GROW2, GROW3 = 0.45, 0.61, 0.415   # weights; .grad after sliding out; from phase 3
    OFFSET = (6.0, 6.0)                      # red tiles: just behind, down and to the right
    DIG, DT = (0.115, 0.632), 4.0            # phase 3: the digit card and its tiles (px)
    RED = (0.285, 0.632)                     # its map's red
    RX = 0.37                                # the running numbers
    EX = ((0.1, 0.3), (0.3, 0.49), (0.49, 0.68))      # phase 3: the three worked places
    SWEEP = (0.7, 0.82)                     # phase 3: the other places
    BY, BARS, BAR_H = 0.672, (0.07, 0.44), 46.0        # phase 4: one bar per digit
    STREAM = (0.12, 0.6)                     # phase 4: the bars arrive
    MATCH = 0.7                              # phase 4: PyTorch's number

    def setup(self):
        m = self.meta = self.study.metadata
        p = self.study.panels
        self.img = np.asarray(p[0].data["values"][0], float)
        self.red = np.asarray(p[1].data["values"], float)
        self.sums = np.asarray(p[3].data["values"], float)
        self.W = np.asarray(p[0].data["weight"], float)[:, 0]
        self.G = np.asarray(p[0].data["grad"], float)[:, 0]
        self.wlim = float(np.abs(self.W).max())
        self.glim = float(np.abs(self.G).max())
        self.rlim = float(np.abs(self.red).max())
        self.red_at = self.at(0, 0.4)
        self.slide_at = self.at(1, 0.2)
        self.up_at = self.at(2, 0.02)
        # phase 4: the digit of phase 3 first, then the others in the batch's order
        d = m["digit"]
        self.bar_order = [d] + [i for i in range(len(self.sums)) if i != d]
        self.bar_running = np.cumsum(self.sums[self.bar_order])
        self.blim = float(np.abs(self.sums).max())
        call = m["step_code"].index(m["call"])
        out = ((0.0, m["printed"]),)
        self.CODE = (Code("cnn-step", run=((0.0, None), (0.3, call))),
                     Code("cnn-grad", run=((0.0, None), (0.1, 0)), out=((0.2, m["printed"]),)),
                     Code("cnn-grad", run=((0.0, 0),), out=out),
                     Code("cnn-grad", run=((0.0, 0),), out=out))
        self.EVIDENCE = (measured("loss.backward() on the first batch"),
                         exact(".grad has the stamps' shape"),
                         measured("one digit of the first batch"),
                         measured("64 × 676 parts, added up"))

    def heading_for(self, k, u):
        return ["loss.backward(): red tiles behind the stamps",
                ".grad: the stamps' shape",
                "one place: pixel × red = one part",
                _pick(((0.0, "every place, every digit: add them up"),
                       (self.MATCH, "the sum is PyTorch's .grad")), u)][k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "loss.backward() gives all %s numbers of the CNN a gradient: red tiles behind "
            "its 8 stamps too." % m["count_text"],
            "model[0].weight.grad is %s, the stamps' shape: one red number for each of the "
            "%d weights." % (m["shape_text"], m["weights"]),
            "Stamp 1's centre weight sat at %d places on this %d. Each adds pixel × red; "
            "%d of them are not 0." % (m["places"], m["label"], m["digit_nonzero"]),
            "%s parts: one sum per digit, and all of them add up to %s, PyTorch's number."
            % (m["parts_sum"], m["total_text"]),
        ][k]

    # ------------------------------------------------------------------ geometry
    def sx(self, j):
        return self.SX + self.PITCH * j

    def grad_xy(self, j):
        """Where stamp ``j``'s red tiles are: behind the weight, sliding out below it in
        phase 2, then up to the top row in phase 3."""
        t = getattr(self, "_t", 0.0)
        dx, dy = self.frac(*self.OFFSET)
        s = ease_in_out(float(np.clip((t - self.slide_at) / 1.4, 0, 1)))
        x = self.sx(j) + dx * (1 - s)
        y = self.WROW + dy + (self.GROW2 - self.WROW - dy) * s
        v = ease_in_out(float(np.clip((t - self.up_at) / 1.2, 0, 1)))
        return x, y + (self.GROW3 - y) * v

    def digit_cell(self):
        return self.digit_cells(*self.DIG, self.DT)

    def red_cell(self):
        n = self.red.shape[0]
        return self.cells(*self.RED, n * self.DT, n * self.DT, n, n)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        # the stamps' weights: phases 1-2
        w_on = op * self.kept(0, 1)
        for j in range(len(self.W)):
            self.stamp(sc, self.W[j], self.sx(j), self.WROW, self.SIDE, w_on, lo=-self.wlim,
                       hi=self.wlim, lift=0.3)
        self.text(sc, 0.066, self.WROW + 0.052, "weight " + m["shape_text"], w_on, SIZE, "lm",
                  ink=WEIGHT_INK)
        # their red tiles, from loss.backward() on
        g_on = op * self.since(self.red_at, 0.8)
        for j in range(len(self.G)):
            gx, gy = self.grad_xy(j)
            self.stamp(sc, self.G[j], gx, gy, self.SIDE, g_on, ink=GRAD, lo=-self.glim,
                       hi=self.glim, lift=0.15)
        gy = self.grad_xy(0)[1]
        self.text(sc, 0.066, gy + 0.052, ".grad " + m["shape_text"],
                  op * self.since(self.slide_at + 1.4), SIZE, "lm", ink=GRAD_INK)
        # the number followed, from phase 3: stamp 1's centre weight
        on = op * self.kept(2)
        if on > 0.01:
            gx, gy = self.grad_xy(m["stamp"])
            cell = self.cells(gx, gy, self.SIDE, self.SIDE, 3, 3)
            self.ring(sc, *cell(m["row"], m["col"]), 14.0, 14.0,
                      on * self.since(self.up_at + 1.2), width=2.6)
        # the loss and the count: phases 1-2
        lo = op * self.kept(0, 1)
        self.meter(sc, 0.075, 0.835, m["loss"] / 3.0, lo, h=50.0, w=20.0)
        self.big(sc, 0.11, 0.843, m["loss_text"], lo, label="loss", size=HEAD, ink=LOSS_INK)
        self.big(sc, 0.25, 0.843, m["count_text"], lo * self.since(self.red_at + 0.4),
                 label="numbers with a .grad", size=HEAD, ink=GRAD_INK)

    # ------------------------------------------------------------------ 1: loss.backward()
    def phase_1(self, sc, u, op):
        self.arrow(sc, (0.035, 0.80), (0.035, self.WROW + 0.01), GRAD, op,
                   u=eased(u, 0.25, 0.4), width=2.2)
        self.text(sc, 0.066, self.WROW - 0.055, "8 stamps", op * smoothstep(0.1, 0.16, u),
                  SIZE, "lm", ink=WEIGHT_INK)

    # ------------------------------------------------------------------ 2: the same shape
    def phase_2(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.066, 0.735, "one red number per weight: %d" % m["weights"],
                  op * smoothstep(0.45, 0.51, u), SIZE, "lm", strong=True)

    # ------------------------------------------------------------------ 3: one place
    def place_at(self, u):
        """``(index in the adding order, sweeping)`` at ``u`` of phase 3, or None."""
        m = self.meta
        if u < self.EX[0][0] + 0.02:
            return None
        if u < self.SWEEP[0]:
            return sum(u >= a for a, _b in self.EX) - 1, False
        n = 3 + int(round((m["places"] - 3) * eased(u, *self.SWEEP)))
        return max(n - 1, 2), True

    def phase_3(self, sc, u, op):
        m = self.meta
        on = op * smoothstep(0.075, 0.11, u)       # once the red tiles have moved up
        self.digit_card(sc, self.img, *self.DIG, self.DT, on)
        self.text(sc, self.DIG[0], self.DIG[1] + 0.143, "a %d, digit %d of %d" % (
            m["label"], m["digit"] + 1, m["batch"]), on, SIZE, ink=DATA_INK)
        # its map's red: grey where it is 0
        n = self.red.shape[0]
        side = n * self.DT
        rgb = shades(self.red, GRAD, -self.rlim, self.rlim, zero=self.red == 0)
        if on > 0.01:
            draw_image(self, sc, self.P(*self.RED), rgb, side, side, on, grid=0.0, lift=0.3)
            self.ring(sc, *self.RED, side + 4, side + 4, on, ink=GRAD, width=1.4, radius=2,
                      lift=0.31)
        self.text(sc, self.RED[0], self.DIG[1] + 0.143, "its map's red; grey: 0", on, SIZE,
                  ink=GRAD_INK)
        # the place under way
        at = self.place_at(u)
        if at is not None:
            i, sweeping = at
            r, c = m["order"][i]
            w_on = op * (1 - smoothstep(self.SWEEP[1] + 0.02, self.SWEEP[1] + 0.06, u))
            cell = self.digit_cell()
            (x0, y0), (x1, y1) = cell(r, c), cell(r + 2, c + 2)
            self.ring(sc, (x0 + x1) / 2, (y0 + y1) / 2, 3 * self.DT + 4, 3 * self.DT + 4,
                      w_on, ink=WEIGHT, width=2.4, radius=2)
            self.ring(sc, *cell(r + 1, c + 1), self.DT + 3, self.DT + 3, w_on, width=1.6,
                      radius=1, lift=1.3)
            self.ring(sc, *self.red_cell()(r, c), self.DT + 4, self.DT + 4, w_on, width=2.0,
                      radius=1)
            if not sweeping:
                ex = m["worked"][i]
                a = self.EX[0][0]
                t_on = op * smoothstep(a + 0.03, a + 0.07, u) * (
                    1 - smoothstep(self.SWEEP[0] - 0.03, self.SWEEP[0], u))
                self.text(sc, 0.066, 0.815, "pixel × red = part", t_on, SIZE, "lm")
                self.text(sc, 0.066, 0.855, ex["text"], t_on, HEAD, "lm", strong=True)
            # the running numbers
            self.running(sc, self.RX, 0.55, [(0.0, str(i + 1))], 0.0, op, label="parts added",
                         size=HEAD)
            self.running(sc, self.RX, 0.69, [(0.0, val5(m["running"][i]))], 0.0, op,
                         label="this digit", size=HEAD, ink=GRAD_INK)
        sw = op * smoothstep(self.SWEEP[0], self.SWEEP[0] + 0.04, u)
        self.text(sc, 0.066, 0.835, "the other %d places, the same way" % (m["places"] - 3),
                  sw * (1 - smoothstep(self.SWEEP[1] + 0.01, self.SWEEP[1] + 0.05, u)), SIZE,
                  "lm")
        self.text(sc, 0.066, 0.835, "%d places: %s" % (m["places"], m["digit_sum_text"]),
                  op * smoothstep(self.SWEEP[1] + 0.05, self.SWEEP[1] + 0.07, u), HEAD, "lm",
                  ink=GRAD_INK)

    # ------------------------------------------------------------------ 4: every digit
    def bars_in(self, u):
        n = len(self.sums)
        return 1 + int(round((n - 1) * eased(u, *self.STREAM)))

    def phase_4(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.066, 0.535, m["parts_sum"] + " parts", op * smoothstep(0.02, 0.06, u),
                  SIZE, "lm", strong=True)
        x0, x1 = self.BARS
        self.polyline(sc, [self.P(x0 - 0.006, self.BY), self.P(x1 + 0.006, self.BY)], 1.2, AXIS,
                      op, bias=0.2)
        n = self.bars_in(u)
        w = self.span((x1 - x0) / len(self.sums))[0] * 0.7
        for i, d in enumerate(self.bar_order[:n]):
            v = self.sums[d]
            h = max(abs(v) / self.blim * self.BAR_H, 0.6)
            x = x0 + (x1 - x0) * (d + 0.5) / len(self.sums)
            y = self.BY - np.sign(v) * self.frac(0, h / 2)[1]
            ring = i == 0 and u < self.STREAM[0] + 0.04
            self.rect(sc, self.P(x, y), w, h, GRAD, op, radius=min(1.5, w / 2), lift=0.3,
                      border=INK if ring else None, border_w=1.4)
        self.text(sc, x0, self.BY + 0.095, "one sum per digit", op * smoothstep(0.04, 0.08, u),
                  SIZE, "lm", ink=GRAD_INK)
        self.running(sc, 0.066, 0.86, [(0.0, val5(self.bar_running[n - 1]))], 0.0,
                     op * smoothstep(0.04, 0.08, u), label="all parts added", size=HEAD,
                     ink=GRAD_INK)
        b = op * smoothstep(self.MATCH, self.MATCH + 0.05, u)
        self.text(sc, 0.235, 0.86, "=", b, HEAD, strong=True)
        self.big(sc, 0.27, 0.86, m["grad_text"], b, label="PyTorch's .grad", size=HEAD,
                 ink=GRAD_INK)


def val5(v):
    s = "%.5f" % float(v)
    if float(s) == 0:
        s = s.lstrip("-")
    return s.replace("-", "−")


# ====================================================================== 219
class SameLoop(Timed):
    """The two models' lines; the same step; both losses on the same batches."""

    TITLE = "The same training loop"
    PHASES = (("the MLP", 11.0), ("the CNN", 12.0), ("the same step", 12.0),
              ("both losses", 15.0), ("one block changed", 9.0))
    LAST = "a shared stamp's gradient"
    NEXT = "the stamps learn"
    STATION = ("forward", "forward", "step", "loss", "loss")
    MX, CX = 0.125, 0.355                    # the two columns of blocks
    Y0, PITCH, BH = 0.43, 0.04, 19.0
    COUNT_Y = 0.762
    BAND = (0.06, 0.45, 0.853, 46.0)         # phase 3: the shared step (x0, x1, y, h px)
    CHART = ((0.11, 0.44), (0.43, 0.745))
    TOP = 2.5
    DRAW = (0.08, 0.75)                      # phase 4: the curves draw
    LEG = (0.25, 0.47)                       # the legend

    def setup(self):
        m = self.meta = self.study.metadata
        self.x = np.asarray(m["chunk_x"], float)
        self.curves = [np.asarray(m["mlp_curve"], float), np.asarray(m["cnn_curve"], float)]
        # each block's line in its region, and when the running line reaches it
        self.lines_of = {}
        for key, region in (("mlp", "mlp_model"), ("cnn", "cnn_model")):
            rows = []
            for i, line in enumerate(m[region][1:-1], 1):
                rows += [i] * line.count("nn.")
            self.lines_of[key] = rows
        self.run_mlp = tuple((0.08 + 0.09 * (i - 1), i) for i in range(1, 7)) + ((0.66, 7),)
        self.run_cnn = tuple((0.08 + 0.09 * (i - 1), i) for i in range(1, 7)) + ((0.66, 7),)
        self.CODE = (Code("model", run=((0.0, 0),) + self.run_mlp),
                     Code("cnn-model", run=((0.0, 0),) + self.run_cnn),
                     Code("cnn-step", run=tuple((0.1 + 0.12 * i, i) for i in range(6))),
                     Code("cnn-train", pick=tuple(range(8)), run=((0.0, 4),)),
                     Code("cnn-train", pick=tuple(range(8)), run=((0.0, 4),)))
        self.appear = {}
        for key, k, run in (("mlp", 0, self.run_mlp), ("cnn", 1, self.run_cnn)):
            run_at = {line: a for a, line in run}
            self.appear[key] = [self.at(k, run_at[ln]) + 0.3 for ln in self.lines_of[key]]
        self.EVIDENCE = (exact("%s numbers, counted" % m["mlp_count_text"]),
                         exact("%s numbers, counted" % m["cnn_count_text"]),
                         exact("%d of %d lines the same" % (m["same_lines"],
                                                           m["program_lines"])),
                         measured("both runs, the same %d batches" % m["steps"]),
                         measured("the last %d steps of each run" % m["last"]))

    def heading_for(self, k, u):
        return ["the MLP's model: six blocks",
                "the CNN's model: only these lines change",
                "the training lines: the same in both",
                "both trained on the same batches",
                "change the model, keep everything else"][k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "The course's program builds an MLP: six blocks and %s numbers to learn."
            % m["mlp_count_text"],
            "The CNN program changes only the model's lines: eight blocks, %s numbers to learn."
            % m["cnn_count_text"],
            "The %d lines of a step are the same in both files, and so are %d of all %d lines."
            % (m["step_lines"], m["same_lines"], m["program_lines"]),
            "Both learn from the same %d batches. Each point is the mean of %d steps' losses, "
            "for both." % (m["steps"], m["chunk"]),
            "Over the last %d steps the CNN's loss is %s and the MLP's %s: the same loop." % (
                m["last"], m["end_cnn_text"], m["end_mlp_text"]),
        ][k]

    # ------------------------------------------------------------------ geometry
    def chart(self):
        return LossChart(self.CHART, (0.0, float(self.meta["steps"])), (0.0, self.TOP))

    def column(self, sc, key, fx, title, count, op):
        m = self.meta
        names = m["mlp_blocks" if key == "mlp" else "cnn_blocks"]
        w = 120.0 if key == "mlp" else 156.0
        on = op * self.since(self.appear[key][0] - 0.6)
        self.text(sc, fx, 0.392, title, on, SIZE, strong=True)
        for i, name in enumerate(names):
            a = op * self.since(self.appear[key][i])
            y = self.Y0 + self.PITCH * i
            self.block_box(sc, name, fx, y, w, self.BH, a)
        last = self.Y0 + self.PITCH * (len(names) - 1)
        self.text(sc, fx, self.COUNT_Y, count + " numbers",
                  op * self.since(self.appear[key][-1] + 0.6), SIZE, ink=WEIGHT_INK)
        return last

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        on = op * self.kept(0, 2)
        self.column(sc, "mlp", self.MX, "the MLP", m["mlp_count_text"], on)
        self.column(sc, "cnn", self.CX, "the CNN", m["cnn_count_text"], on)
        # the shared step: from phase 3 on
        self.band(sc, op * self.kept(2))
        # the chart: phases 4-5
        c_on = op * self.kept(3)
        if c_on <= 0.01:
            return
        ch = self.chart()
        self.axes(sc, ch, c_on, xticks=[(0, "0"), (m["steps"], "%d steps" % m["steps"])],
                  yticks=[(0, "0"), (1, "1"), (2, "2")], grid=False)
        self.text(sc, ch.x0, ch.y0 - 0.022, "loss", c_on, SIZE, "lm", ink=LOSS_INK)
        t = getattr(self, "_t", 0.0)
        drawn = eased(t, self.at(3, self.DRAW[0]), self.at(3, self.DRAW[1]))
        for curve, ink in zip(self.curves, (MLP_INK, LOSS)):
            if drawn > 0:
                self.plot(sc, ch, self.x, curve, ink, c_on, drawn, dot=k == 3 and drawn < 1)
        lx, ly = self.LEG
        for i, (name, ink) in enumerate((("MLP", MLP_INK), ("CNN", LOSS))):
            y = ly + 0.045 * i
            self.polyline(sc, [self.P(lx, y), self.P(lx + 0.03, y)], 3.2, ink, c_on, bias=0.5)
            self.text(sc, lx + 0.04, y, name, c_on, SIZE, "lm", ink=ink if i else MLP_INK)
        self.text(sc, lx, ly + 0.09, "each point: mean of %d steps" % m["chunk"], c_on,
                  SIZE, "lm")

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        pass

    def phase_2(self, sc, u, op):
        pass

    def band(self, sc, op):
        """The step's lines, shared by both programs: a band under the picture."""
        if op <= 0.01:
            return
        m = self.meta
        x0, x1, y, h = self.BAND
        self.rect(sc, self.P((x0 + x1) / 2, y), self.span(x1 - x0)[0], h, soft(LOOP, 0.85),
                  op, border=LOOP, border_w=2.0, radius=14, lift=0.2)
        self.text(sc, (x0 + x1) / 2, y - 0.017, "step: the same %d lines in both files"
                  % m["step_lines"], op, SIZE, strong=True)
        self.text(sc, (x0 + x1) / 2, y + 0.019, "%d of %d lines the same" % (
            m["same_lines"], m["program_lines"]), op * self.since(self.at(2, 0.75)), SIZE)

    def phase_3(self, sc, u, op):
        pass

    def phase_4(self, sc, u, op):
        pass

    def phase_5(self, sc, u, op):
        m = self.meta
        lx, ly = self.LEG
        a = op * smoothstep(0.08, 0.14, u)
        self.text(sc, lx + 0.09, ly - 0.042, "last %d steps" % m["last"], a, SIZE, "mm")
        for i, (txt, ink) in enumerate(((m["end_mlp_text"], MLP_INK),
                                        (m["end_cnn_text"], LOSS_INK))):
            self.text(sc, lx + 0.09, ly + 0.045 * i, txt, a, SIZE, "lm", ink=ink, strong=i == 1)
        b = op * smoothstep(0.35, 0.41, u)
        self.text(sc, lx, ly + 0.135, "CNN lower: %d of %d points" % (m["cnn_lower"],
                                                                   m["chunks"]), b, SIZE, "lm")


# ====================================================================== 220
class StampsLearn(Timed):
    """The 8 stamps at five steps; the maps they make of one digit, before and after."""

    TITLE = "The stamps learn"
    PHASES = (("random stamps", 11.0), ("training", 17.0), ("maps, step 0", 12.0),
              ("maps, step 938", 14.0))
    LAST = "the same loop trains the CNN"
    NEXT = "the CNN against the MLP"
    STATION = ("forward", "step", "forward", "forward")
    LX = 0.085                               # the label column
    X0, PITCH = 0.155, 0.043                 # the 8 columns
    SIDE = 27.0                              # a stamp (px)
    ROWS = (0.435, 0.495, 0.555, 0.615, 0.675)     # steps 0, 10, 100, 500, 938
    MAP_Y = (0.53, 0.775)                    # the maps of step 0 and of step 938
    MT = 1.32                                # a map's tile (px)
    DT = 1.5                                 # the digit's tile beside them
    ROW_AT = (0.06, 0.28, 0.5, 0.72)         # phase 2: rows 10 ... 938 appear

    def setup(self):
        m = self.meta = self.study.metadata
        p = self.study.panels
        n = m["stamps"]
        self.S = np.asarray(p[0].data["values"], float).reshape(len(m["steps"]), n, 3, 3)
        self.maps = [np.asarray(p[2].data["values"], float),
                     np.asarray(p[3].data["values"], float)]
        self.img = np.asarray(p[2].data["digit"], float)
        self.lim = float(m["lim"])
        self.row_t = [self.at(0, 0.15)] + [self.at(1, a) for a in self.ROW_AT]
        self.map_t = [self.at(2, 0.15), self.at(3, 0.15)]
        conv = list(code_lines("cnn-model")).index(m["conv_line"])
        self.CODE = (Code("cnn-model", run=((0.0, None), (0.25, conv))),
                     Code("cnn-step", run=((0.0, 5),)),
                     Code("cnn-model", run=((0.0, conv),)),
                     Code("cnn-model", run=((0.0, conv),)))
        self.EVIDENCE = (exact("the stamps nn.Conv2d starts with"),
                         measured("the first layer's stamps, one epoch"),
                         exact("a real digit, the stamps of step 0"),
                         exact("a real digit, the stamps of step %d" % m["steps"][-1]))

    def heading_for(self, k, u):
        m = self.meta
        return ["8 stamps of 3 × 3, random to start",
                "each step moves them a little",
                "step 0: the maps they make of a %d" % m["label"],
                "step %d: the trained stamps' maps" % m["steps"][-1]][k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "nn.Conv2d(1, 8, 3) starts with 8 random stamps of 9 weights, none bigger than "
            "1/3.",
            "SGD moves every weight a little each step: on average %s after 10 steps, %s "
            "after %d." % (m["moved_text"][1], m["moved_text"][-1], m["steps"][-1]),
            "Untrained, the 8 stamps turn this %d into 8 maps, one per stamp." % m["label"],
            "Trained, the same 8 stamps make different maps of the same %d; these feed the "
            "next blocks." % m["label"],
        ][k]

    def cx(self, j):
        return self.X0 + self.PITCH * j

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        n = m["stamps"]
        head = op * self.since(self.row_t[0] - 0.5)
        self.text(sc, self.LX, 0.39, "step", head, SIZE, ink=WEIGHT_INK)
        for j in range(n):
            self.text(sc, self.cx(j), 0.39, str(j + 1), head, SIZE, ink=WEIGHT_INK)
        for r, step in enumerate(m["steps"]):
            keep = self.kept(0, 1) if 0 < r < len(m["steps"]) - 1 else 1.0
            on = op * keep * self.since(self.row_t[r])
            if on <= 0.01:
                continue
            y = self.ROWS[r]
            self.text(sc, self.LX, y, str(step), on, SIZE, strong=True)
            for j in range(n):
                self.stamp(sc, self.S[r, j], self.cx(j), y, self.SIDE, on, lo=-self.lim,
                           hi=self.lim)
        # how far they moved: phases 1-2
        mv = op * self.kept(0, 1) * self.since(self.row_t[1] + 0.6)
        done = sum(getattr(self, "_t", 0.0) >= t + 0.6 for t in self.row_t)
        self.running(sc, 0.066, 0.82, [(0.0, m["moved_text"][max(done - 1, 0)])], 0.0, mv,
                     label="average move from step 0", size=HEAD, ink=WEIGHT_INK)
        # the maps of one digit: step 0 from phase 3, step 938 from phase 4
        for i, t0 in enumerate(self.map_t):
            keep = self.kept(2 + i)
            y = self.MAP_Y[i]
            d_on = op * keep * self.since(t0 - 0.9)
            self.digit_card(sc, self.img, self.LX, y, self.DT, d_on)
            for j in range(n):
                a = op * keep * self.since(t0 + 0.35 * j)
                self.map_tile(sc, self.maps[i][j], self.cx(j), y, a)

    def map_tile(self, sc, values, fx, fy, op):
        if op <= 0.01:
            return
        side = values.shape[0] * self.MT
        self.tiles(sc, values, fx, fy, side, side, op, ink=DATA, lift=0.3, grid=0.0)
        self.ring(sc, fx, fy, side + 3, side + 3, op, ink=DATA, width=1.2, radius=2, lift=0.31)

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        # stamp 1's nine numbers
        on = op * smoothstep(0.3, 0.36, u)
        w = self.S[0, 0]
        cell = self.cells(0.2, 0.62, 3 * 50.0, 3 * 27.0, 3, 3)
        fills = shades(w, WEIGHT, -self.lim, self.lim)
        for r in range(3):
            for c in range(3):
                self.box(sc, *cell(r, c), 46.0, 23.0, tuple(float(v) for v in fills[r, c]),
                         on, text=val(w[r, c]), ink=INK, border=WEIGHT, radius=3, size=SIZE)
        self.text(sc, 0.2, 0.51, "stamp 1: 9 numbers", on, SIZE, ink=WEIGHT_INK)
        self.text(sc, 0.066, 0.82, "orange: deeper is bigger", op * smoothstep(0.5, 0.56, u),
                  SIZE, "lm", ink=WEIGHT_INK)

    def phase_2(self, sc, u, op):
        pass

    def phase_3(self, sc, u, op):
        self.text(sc, self.X0 - 0.02, 0.875, "one map per stamp", op * smoothstep(0.5, 0.56, u),
                  SIZE, "lm", ink=DATA_INK)

    def phase_4(self, sc, u, op):
        self.text(sc, self.X0 - 0.02, 0.875, "each column: one stamp, before and after",
                  op * smoothstep(0.55, 0.61, u), SIZE, "lm", ink=DATA_INK)


ANIMATIONS = {"blocks-cnn-backward": StampGrad, "blocks-cnn-same-loop": SameLoop,
              "blocks-cnn-stamps-learn": StampsLearn}

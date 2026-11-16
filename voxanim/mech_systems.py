"""Training at scale, how it is arranged: animations 86-90 of the "next 100".

Each draws the mechanism first and the measurement after it (see
:mod:`voxanim.mech`). Every number and every heatmap is read from the study
(:mod:`voxanim.study_systems`; :mod:`voxanim.next_systems` for ZeRO): the
ring's buffers after every pass, what a checkpointed backward pass holds at
every step and the timings under the roofline are recorded by the experiment,
and nothing is computed here but where to draw it.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from .diagram import draw_image
from .easing import ease_in_out, smoothstep
from .losschart import LossChart
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIM,
    GREEN,
    MAGENTA,
    PAPER,
    RED,
    TEAL,
    TRACK,
    VIOLET,
    Mechanism,
    dark,
    heat,
    nice_ticks,
    number,
    soft,
)

WHITE = (1.0, 1.0, 1.0)
GREY = (0.33, 0.35, 0.38)               # negative values, beside a device's colour for positive
DEVICES = (RED, BLUE, GREEN, AMBER)     # one colour per worker / device, in every phase


def lerp(a, b, t):
    return a + (b - a) * t


def mix(a, b, t):
    """Colour ``a`` moved a fraction ``t`` towards ``b``."""
    return tuple(float(v) for v in C.mix(a, b, float(t)))


def along(path, s):
    """The layout point a fraction ``s`` of the way along the polyline ``path``
    (lengths measured on the 16:9 frame)."""
    pts = np.asarray(path, float)
    seg = np.hypot(np.diff(pts[:, 0]) * 16, np.diff(pts[:, 1]) * 9)
    at = float(np.clip(s, 0, 1)) * seg.sum()
    for i, length in enumerate(seg):
        if at <= length or i == len(seg) - 1:
            f = at / max(length, 1e-12)
            return tuple(pts[i] + (pts[i + 1] - pts[i]) * min(f, 1.0))
        at -= length


class Systems(Mechanism):
    """What the five animations share: a heatmap that can travel, a legend swatch."""

    def heatmap(self, sc, values, fx, fy, w, h, op, vmax, hot=RED, cold=BLUE, lift=0.3,
                border=None, border_w=2.2):
        """``values`` as a ``w x h`` px heatmap centred at ``(fx, fy)``, in front
        of whatever has a smaller ``lift``; ``border``: an outline in that colour."""
        if op <= 0.01:
            return
        a = np.atleast_2d(np.asarray(values, float))
        rgb = heat(np.clip(a / max(vmax, 1e-12), -1, 1), hot, cold)
        c = self.P(float(fx), float(fy))
        draw_image(self, sc, c, rgb.astype(np.float32), w, h, op,
                   grid=0.16 if min(w / a.shape[1], h / a.shape[0]) >= 9 else 0, lift=lift)
        if border is not None:
            self.rect(sc, c, w + 5, h + 5, border, op, border=border, border_w=border_w,
                      radius=2, lift=lift + 0.4, fill_op=0.0)

    def swatch(self, sc, fx, fy, ink, name, op):
        """A legend entry: a square of ``ink`` and what it stands for."""
        self.box(sc, fx, fy, 16, 16, ink, op, radius=3)
        self.text(sc, fx + 0.016, fy, name, op, anchor="lm")

    def route(self, sc, path, ink, op, width=2.2, head=True):
        """A connection along the layout polyline ``path``, ending in an arrowhead."""
        if op <= 0.01:
            return
        pts = [self.P(float(x), float(y)) for x, y in path]
        self.polyline(sc, pts, width, ink, op, bias=0.3)
        if head:
            self._head(sc, pts[-2], pts[-1], ink, op, width)


# ====================================================================== 86
class DataParallelAllReduce(Systems):
    """Four workers with the same partly trained weights and different digits;
    their gradients cut into chunks that go round a ring, first summing, then
    spreading the sum; the average they end with beside the gradient of all
    64 digits on one machine; and the whole training run both ways."""

    TITLE = "Data parallelism and ring all-reduce"
    PHASES = (("workers", 10.0), ("reduce", 14.0), ("gather", 12.0), ("one gradient", 10.0),
              ("training", 9.0))
    XS = (0.17, 0.39, 0.61, 0.83)          # the four workers
    BAND = (104.0, 39.0)                   # one chunk of a gradient: 3 rows of 8 values, px
    Y0, STEP = 0.475, 0.085                # the first chunk's centre, and the next one's offset
    LANE = 0.853                           # where the ring closes, under the workers

    def setup(self):
        s = self.study
        first, reduce, gather, training = (p.data for p in s.panels)
        local = np.asarray(first["local"], float)                   # worker x 12 x 8
        n = self.workers = len(local)
        self.shape = local.shape[1:]
        chunk = local[0].size // n

        def chunks(states):
            """``(pass, worker, chunk, values)`` from the ring's flat buffers."""
            a = np.asarray(states, float)
            return a.reshape(a.shape[0], n, n, chunk)

        def sends(rows):
            return [[tuple(int(v) for v in send) for send in turn] for turn in np.asarray(rows)]

        self.d = dict(
            digits=np.asarray(first["digits"]), local=local,
            full=np.asarray(gather["whole"], float),
            reduce=chunks(reduce["states"]), counts=np.asarray(reduce["counts"]).astype(int),
            gather=chunks(gather["states"]), have=np.asarray(gather["have"]).astype(bool),
            reduce_sends=sends(reduce["sends"]), gather_sends=sends(gather["sends"]))
        d = self.d
        self.grad_bytes, self.ring_bytes = (int(v) for v in gather["bytes"])
        self.per_pass = self.grad_bytes // n                     # one chunk, in bytes
        self.vloc = float(np.abs(local).max())
        self.vsum = float(np.abs(d["reduce"]).max())
        self.err = float(np.abs(d["gather"][-1].reshape((n,) + self.shape) - d["full"]).max())
        off = ~np.eye(n, dtype=bool)
        self.alike = np.asarray(first["cosines"], float)[off]
        self.steps = np.asarray(training["x"], float)
        self.loss = np.asarray(training["ys"], float)            # ring, one machine
        self.accuracy = np.asarray(training["accuracy"], float)
        self.apart = np.asarray(training["weights_apart"], float)
        self.step_apart = float(training["step_apart"])
        self.shown = int(s.metadata["shown_step"])

    def subtitle_for(self, k):
        return [
            "four workers hold the same weights of an MNIST classifier, %d training steps in; "
            "each gets 16 different digits and computes its own gradient, and the four differ"
            % self.shown,
            "reduce: each gradient is cut into 4 chunks; in 3 passes round the ring every worker "
            "ends up holding one chunk summed over all 4 workers",
            "gather: the summed chunks, divided by 4, go round the ring 3 more times, until "
            "every worker holds the whole average",
            "the ring's average and the gradient of all 64 digits on one machine differ by at "
            "most %s; each worker sent %s bytes for a gradient of %d" % (
                number(self.err), f"{self.ring_bytes:,}", self.grad_bytes),
            "trained %d steps both ways from the same start, the two models stay the same: "
            "held-out loss %.2f to %.2f for both, weights never more than %s apart" % (
                len(self.steps) - 1, self.loss[0, 0], self.loss[0, -1],
                number(self.apart.max())),
        ][k]

    # ------------------------------------------------------------------ phase 1
    def phase_1(self, sc, u, op):
        d = self.d
        for w, x in enumerate(self.XS):
            ink = DEVICES[w]
            self.text(sc, x, 0.295, "worker %d" % (w + 1), op, ink=dark(ink), size=0.034)
            shown = smoothstep(0.04 + 0.05 * w, 0.30 + 0.05 * w, u) * 16
            for i in range(16):
                r, c = divmod(i, 4)
                self.digit(sc, d["digits"][16 * w + i], x + (c - 1.5) * 0.0255,
                           0.352 + r * 0.0455, 22, op * float(np.clip(shown - i, 0, 1)))
            a, b = (x, 0.585), (x, 0.612)
            go = smoothstep(0.38, 0.50, u)
            self.arrow(sc, a, b, ink, op * go, go)
            rows = 12 * smoothstep(0.50 + 0.03 * w, 0.74 + 0.03 * w, u)
            self.heatmap(sc, d["local"][w] * (np.arange(12)[:, None] < rows), x, 0.745, 96, 144,
                         op, self.vloc, border=ink)
        self.box(sc, 0.5, 0.555, self.span(0.86)[0], 26, dark(TEAL, 0.1), op * smoothstep(
            0.28, 0.40, u), "the same weights on every worker: 12 inputs x 8 outputs", ink=WHITE)
        self.text(sc, 0.5, 0.915, "each worker's own gradient: red raises a weight, blue lowers it",
                  op * smoothstep(0.80, 0.92, u), strong=True)

    # ------------------------------------------------------------------ phases 2 and 3
    def _pass(self, u):
        """``(pass under way, how far through it)``: three passes in a phase."""
        tt = float(np.clip((u - 0.10) / 0.74, 0.0, 1.0)) * 3
        turn = min(int(tt), 2)
        return turn, tt - turn

    def _band_y(self, chunk):
        return self.Y0 + chunk * self.STEP

    def _ring(self, sc, u, op, gather):
        d = self.d
        bw, bh = self.BAND
        turn, within = self._pass(u)
        started, done = u > 0.10, u >= 0.84
        landed = within >= 0.80
        state = turn + (1 if landed else 0)
        sends = (d["gather_sends"] if gather else d["reduce_sends"])[turn]
        moving = started and not landed
        half = self.frac(bw / 2)[0]
        tab = self.frac(17)[0]
        for w, x in enumerate(self.XS):
            self.text(sc, x, 0.405, "worker %d" % (w + 1), op, ink=dark(DEVICES[w]), size=0.034)
            for c in range(self.workers):
                y = self._band_y(c)
                sending = moving and (w, c, (w + 1) % self.workers) in sends
                arrived = (started and landed and not done
                           and ((w - 1) % self.workers, c, w) in sends)
                ring = AMBER if sending else GREEN if arrived else None
                strong = ring is not None
                if gather:
                    if d["have"][state][w, c]:
                        self.heatmap(sc, d["gather"][state][w, c].reshape(3, 8), x, y, bw, bh, op,
                                     self.vloc, border=AXIS if ring is None else ring,
                                     border_w=1.2 if ring is None else 2.6)
                    else:
                        self.box(sc, x, y, bw + 4, bh + 4, PAPER, op * 0.75, border=DIM, radius=2)
                    continue
                n = int(d["counts"][state][w, c])
                whole = n == self.workers
                fade = 0.3 if done and not whole else 1.0
                if ring is None:
                    ring, strong = (GREEN, True) if done and whole else (AXIS, False)
                self.heatmap(sc, d["reduce"][state][w, c].reshape(3, 8), x, y, bw, bh, op * fade,
                             self.vsum, border=ring, border_w=2.6 if strong else 1.2)
                self.box(sc, x - half - tab, y, 22, bh, mix(PAPER, GREEN, (n - 1) / 3.0),
                         op * fade, str(n), ink=WHITE if n > 2 else dark(GREEN, 0.5), radius=3)
        # the ring: neighbour to neighbour, and from the last worker back to the first
        last, first = self.XS[-1], self.XS[0]
        bottom = self._band_y(self.workers - 1) + self.frac(0, bh / 2 + 8)[1]
        self.route(sc, [(last, bottom), (last, self.LANE), (first, self.LANE), (first, bottom)],
                   AXIS, op * 0.8, 1.6)
        self.text(sc, 0.5, self.LANE + 0.06, "the ring closes: worker 4 sends to worker 1", op)
        if moving:
            go = ease_in_out(float(np.clip((within - 0.12) / 0.66, 0.0, 1.0)))
            source = d["gather" if gather else "reduce"][turn]
            for w, c, to in sends:
                y = self._band_y(c)
                if to > w:
                    path = [(self.XS[w], y), (self.XS[to], y)]
                else:
                    path = [(last, y), (last, self.LANE), (first, self.LANE), (first, y)]
                at = along(path, go)
                self.heatmap(sc, source[w, c].reshape(3, 8), at[0], at[1], bw, bh, op,
                             self.vloc if gather else self.vsum, lift=2.4, border=AMBER)
        passes = (state if started else 0) + (3 if gather else 0)
        if done:
            line = ("every worker now holds all 4 chunks of the average: the same gradient"
                    if gather else
                    "each worker now holds one chunk summed over all 4 workers (outlined)")
        elif not started:
            line = ("each worker divides its summed chunk by 4: that chunk of the average is final"
                    if gather else
                    "each gradient is cut into 4 chunks; the number counts the gradients "
                    "summed in one")
        else:
            line = "pass %d of 3: every worker hands one chunk to its neighbour, who %s" % (
                turn + 1, "keeps a copy" if gather else "adds it to its own")
        self.text(sc, 0.05, 0.315, line, op, anchor="lm", strong=True)
        self.text(sc, 0.95, 0.36, "sent so far by each worker: %s bytes" % f"{passes * self.per_pass:,}", op, anchor="rm", ink=dark(AMBER, 0.3))  # noqa: E501

    def phase_2(self, sc, u, op):
        self._ring(sc, u, op, gather=False)

    def phase_3(self, sc, u, op):
        self._ring(sc, u, op, gather=True)

    # ------------------------------------------------------------------ phase 4
    def phase_4(self, sc, u, op):
        d = self.d
        average = d["gather"][-1][0].reshape(self.shape)
        self.text(sc, 0.15, 0.315, "the ring's average", op, strong=True, size=0.034)
        self.heatmap(sc, average, 0.15, 0.60, 144, 216, op, self.vloc, border=AXIS, border_w=1.4)
        self.text(sc, 0.265, 0.60, "=", op * smoothstep(0.15, 0.3, u), strong=True, size=0.06)
        self.text(sc, 0.38, 0.315, "all 64 digits at once", op, strong=True, size=0.034)
        self.heatmap(sc, d["full"] * (np.arange(12)[:, None] < 12 * smoothstep(0.05, 0.3, u)),
                     0.38, 0.60, 144, 216, op, self.vloc, border=AXIS, border_w=1.4)
        self.text(sc, 0.265, 0.875, "largest difference between them: %s" % number(self.err),
                  op * smoothstep(0.3, 0.45, u), strong=True)
        x, full = 0.70, self.span(0.17)[0]
        self.text(sc, 0.56, 0.40, "bytes each worker sends", op, anchor="lm", strong=True,
                  size=0.034)
        for row, (name, ink, value) in enumerate((("one gradient", TEAL, self.grad_bytes),
                                                  ("ring all-reduce", AMBER, self.ring_bytes))):
            self.hbar(sc, x, 0.49 + row * 0.10, full * value / self.ring_bytes, ink, op, name,
                      f"{value:,}", (u - 0.35 - 0.12 * row) / 0.3, thick=34.0, track=full)
        p = self.workers
        self.lines(sc, 0.56, 0.70, [
            "3 chunks out to sum, 3 out to share: 2 x (%d - 1) / %d" % (p, p),
            "= %s gradients' worth; under 2 with any number" % number(2 * (p - 1) / p),
            "one SGD step with either gradient: the new",
            "weights differ by at most %s" % number(self.step_apart)],
            op * smoothstep(0.62, 0.8, u), size=0.032, step=0.047)

    # ------------------------------------------------------------------ phase 5
    def phase_5(self, sc, u, op):
        steps, (ring, one) = self.steps, self.loss
        chart = LossChart(((0.11, 0.56), (0.38, 0.80)), (steps[0], steps[-1]),
                          (0.0, float(np.ceil(self.loss.max() * 2) / 2)))
        self.axes(sc, chart, op, "training steps of 64 digits", "held-out loss")
        go = smoothstep(0.04, 0.70, u)
        self.plot(sc, chart, steps, one, soft(TEAL, 0.35), op, go, width=5.0, dot=False)
        self.plot(sc, chart, steps, ring, RED, op, go, width=1.8)
        chart.mark(sc, self.P, self.shown, AXIS, op * 0.8)
        self.text(sc, chart.fx(self.shown) + 0.008, 0.40, "the step drawn", op, anchor="lm",
                  ink=dark(AXIS))
        self.swatch(sc, 0.63, 0.40, RED, "4 workers and the ring", op)
        self.swatch(sc, 0.63, 0.45, soft(TEAL, 0.35), "1 machine, all 64 digits at once", op)
        at = int(round(go * (len(steps) - 1)))
        self.lines(sc, 0.62, 0.55, [
            ("after %d steps" % at, None),
            ("held-out loss  %.4f  and  %.4f" % (ring[at], one[at]), None),
            ("held-out accuracy  %.1f%%  and  %.1f%%" % (
                100 * self.accuracy[0, at], 100 * self.accuracy[1, at]), None),
            ("largest difference between their", None),
            ("weights so far: %s" % number(self.apart[:at + 1].max()), None)],
            op, size=0.032, step=0.052)
        self.text(sc, 0.5, 0.915, "averaging the workers' gradients is the big-batch gradient: "
                  "the same training, shared out", op * smoothstep(0.74, 0.86, u), strong=True,
                  size=0.032)


# ====================================================================== 87
class TensorAndPipelineParallel(Systems):
    """One layer of a trained MNIST classifier cut across four devices, by
    column and then by row, giving the same output; then its four layers as a
    pipeline, micro-batches moving down it, and the idle slots (the bubble)
    counted."""

    TITLE = "Tensor and pipeline parallelism"
    PHASES = (("columns", 12.0), ("rows", 12.5), ("pipeline", 14.0), ("bubble", 11.5))
    CELL = 10.5

    def setup(self):
        s = self.study
        columns, rows, pipe, bubble = (p.data for p in s.panels)
        d = self.d = {k: np.asarray(columns[k], float) for k in ("x", "w", "cols", "reference")}
        d["parts"] = np.asarray(rows["parts"], float)
        self.vx = float(np.abs(d["x"]).max())
        self.vw = float(np.abs(d["w"]).max())
        self.vy = float(np.abs(d["reference"]).max())
        self.err_cols = float(np.abs(np.concatenate(d["cols"], 1) - d["reference"]).max())
        self.err_rows = float(np.abs(d["parts"].sum(0) - d["reference"]).max())
        self.accuracy = np.asarray(rows["accuracy"], float)      # whole, by column, by row
        self.timeline = np.asarray(pipe["values"]).astype(int)
        self.pipe_gap = float(pipe["gap"])
        self.micro = np.asarray(bubble["x"], float)
        self.util = np.asarray(bubble["ys"], float)[0]
        self.stages = self.timeline.shape[0]
        self.batches = int(self.timeline.max())
        # the schedule the study ran for each number of micro-batches
        self.sweep = [np.asarray(t)[:, :int(m) + self.stages - 1].astype(int)
                      for m, t in zip(self.micro, bubble["timelines"])]

    def subtitle_for(self, k):
        busy, slots = int((self.timeline > 0).sum()), self.timeline.size
        return [
            "tensor parallelism, by column: each of 4 devices holds 4 columns of a trained "
            "layer's 12 x 16 weights and computes 4 of its 16 outputs for 16 MNIST digits",
            "by row: each device holds 3 rows of the weights and the 3 inputs they multiply; "
            "the 4 partial outputs add up to the same output (largest difference %s)" % number(
                self.err_rows),
            "pipeline parallelism: the classifier's 4 layers on 4 devices, %d micro-batches "
            "moving down them a step at a time; %d of %d slots are busy" % (
                self.batches, busy, slots),
            "the idle slots are the bubble: with 4 stages it is always 3 steps per stage, so "
            "more micro-batches keep the devices busier, %.0f%% with 1 and %.0f%% with 32" % (
                100 * self.util[0], 100 * self.util[-1]),
        ][k]

    # ------------------------------------------------------------------ the matrices
    def _blocks(self, sc, values, fx, fy, op, vmax, axis=None, active=None, reveal=None,
                cell=None):
        """A matrix at ``CELL`` px per value, cut into four blocks along
        ``axis`` (0: bands of rows, 1: of columns; None: whole), each in its
        device's colour; ``active``: the block at full strength; ``reveal``:
        per block, 0..1 of it shown. Returns the blocks' centres."""
        a = np.asarray(values, float)
        rows, cols = a.shape
        gap, cell = 5.0, cell or self.CELL
        if axis is None:
            self.heatmap(sc, a, fx, fy, cols * cell, rows * cell, op, vmax, VIOLET, GREY,
                         border=AXIS, border_w=1.2)
            return [(fx, fy)]
        centres = []
        for d, block in enumerate(np.split(a, 4, axis=axis)):
            r, c = block.shape
            w, h = c * cell, r * cell
            off = (d - 1.5) * ((w if axis == 1 else h) + gap)
            dx, dy = self.frac(off if axis == 1 else 0.0, off if axis == 0 else 0.0)
            shown = 1.0 if reveal is None else float(reveal[d])
            strength = 1.0 if active is None or active == d else 0.3
            if shown < 1.0:
                self.box(sc, fx + dx, fy + dy, w + 4, h + 4, PAPER, op * 0.7, border=DIM, radius=2)
            self.heatmap(sc, block, fx + dx, fy + dy, w, h, op * strength * shown, vmax,
                         DEVICES[d], GREY, border=DEVICES[d], border_w=2.6 if active == d else 1.4)
            centres.append((fx + dx, fy + dy))
        return centres

    def _turn(self, u, n=4, start=0.06, end=0.86):
        tt = float(np.clip((u - start) / (end - start), 0.0, 1.0)) * n
        turn = min(int(tt), n - 1)
        return turn, tt - turn, u >= end

    def phase_1(self, sc, u, op):
        d = self.d
        turn, within, done = self._turn(u)
        reveal = [1.0 if k < turn or done else smoothstep(0.45, 0.85, within) if k == turn else 0.0
                  for k in range(4)]
        y = 0.585
        self.text(sc, 0.13, 0.325, "16 digits x 12 inputs", op, strong=True)
        self._blocks(sc, d["x"], 0.13, y, op, self.vx, cell=12.0)
        self.text(sc, 0.25, y, "x", op, strong=True, size=0.05)
        self.text(sc, 0.44, 0.325, "weights, cut by column", op, strong=True)
        wc = self._blocks(sc, d["w"], 0.44, y, op, self.vw, 1, None if done else turn, cell=12.0)
        self.text(sc, 0.63, y, "=", op, strong=True, size=0.05)
        self.text(sc, 0.81, 0.325, "output, 16 x 16", op, strong=True)
        oc = self._blocks(sc, np.concatenate(d["cols"], 1), 0.81, y, op, self.vy, 1, None,
                          reveal=reveal, cell=12.0)
        for k in range(4):
            self.text(sc, wc[k][0], y - 0.17, "%d" % (k + 1), op, ink=dark(DEVICES[k]), size=0.034)
        if not done:
            a, b = (wc[turn][0], y + 0.15), (oc[turn][0], y + 0.195)
            self.arc(sc, a, b, DEVICES[turn], op, smoothstep(0.05, 0.45, within), -0.10, 2.6)
            self.pulse(sc, a, b, (within - 0.2) / 0.6, DEVICES[turn], op, -0.10)
            line = "device %d: its 4 columns of weights give columns %d-%d of the output" % (
                turn + 1, 4 * turn + 1, 4 * turn + 4)
        else:
            line = ("side by side, the four pieces are the whole output: difference %s; the "
                    "classifier is still %.1f%% right" % (number(self.err_cols),
                                                         100 * self.accuracy[1]))
        self.text(sc, 0.5, 0.915, line, op, strong=True, size=0.032)

    def phase_2(self, sc, u, op):
        d = self.d
        turn, within, done = self._turn(u)
        landed = within >= 0.75 or done
        y = 0.59
        self.text(sc, 0.125, 0.335, "inputs, by column", op, strong=True)
        xc = self._blocks(sc, d["x"], 0.125, y, op, self.vx, 1, None if done else turn)
        self.text(sc, 0.242, y, "x", op, strong=True, size=0.05)
        self.text(sc, 0.37, 0.335, "weights, by row", op, strong=True)
        self._blocks(sc, d["w"], 0.37, y, op, self.vw, 0, None if done else turn)
        self.text(sc, 0.485, y, "=", op, strong=True, size=0.05)
        side = 16 * 8.5
        if not done:
            ink = DEVICES[turn]
            self.text(sc, 0.60, 0.335, "device %d's part" % (turn + 1), op, ink=dark(ink),
                      size=0.032)
            go = ease_in_out(float(np.clip((within - 0.5) / 0.25, 0.0, 1.0)))
            self.heatmap(sc, d["parts"][turn], lerp(0.60, 0.83, go), y, side, side,
                         op * smoothstep(0.05, 0.35, within) * (1 - smoothstep(0.72, 0.78, within)),
                         self.vy, ink, GREY, lift=2.0, border=ink)
            self.text(sc, 0.715, y, "+", op, strong=True, size=0.05)
        count = 4 if done else turn + (1 if landed else 0)
        total = d["parts"][:count].sum(0) if count else np.zeros((16, 16))
        self.text(sc, 0.83, 0.335, "sum of %d part%s" % (count, "" if count == 1 else "s"), op,
                  strong=True)
        self.heatmap(sc, total, 0.83, y, 16 * self.CELL, 16 * self.CELL, op, self.vy, VIOLET, GREY,
                     border=AXIS, border_w=1.4)
        for k in range(4):
            self.text(sc, xc[k][0], y + 0.20, "%d" % (k + 1), op, ink=dark(DEVICES[k]), size=0.034)
        line = ("all four parts added: the same output as one machine, difference %s; "
                "still %.1f%% right" % (number(self.err_rows), 100 * self.accuracy[2]) if done else
            "device %d: its 3 inputs times its 3 rows of weights give a full-size part" % (
                turn + 1))
        self.text(sc, 0.5, 0.915, line, op, strong=True, size=0.032)

    # ------------------------------------------------------------------ the pipeline
    def phase_3(self, sc, u, op):
        t = self.timeline
        stages, slots = t.shape
        x0, dx, y0, dy = 0.235, 0.0655, 0.43, 0.098
        now = float(np.clip((u - 0.05) / 0.80, 0.0, 1.0)) * slots
        self.text(sc, 0.05, 0.335, "time step", op, anchor="lm")
        for k in range(slots):
            self.text(sc, x0 + k * dx, 0.335, "%d" % (k + 1), op,
                      strong=int(now) == k and now < slots)
        busy = idle = 0
        for s in range(stages):
            y = y0 + s * dy
            self.text(sc, 0.05, y, "device %d" % (s + 1), op, anchor="lm", ink=dark(DEVICES[s]),
                      size=0.034)
            for k in range(slots):
                shown = float(np.clip(now - k, 0.0, 1.0))
                x = x0 + k * dx
                self.box(sc, x, y, 56, 42, PAPER, op * 0.6, border=DIM, radius=5)
                if shown <= 0:
                    continue
                if t[s, k]:
                    busy += shown >= 1
                    self.box(sc, x, y, 56, 42, TEAL, op * smoothstep(0.0, 0.6, shown),
                             "%d" % t[s, k], ink=WHITE, size=0.036, radius=5,
                             glow=0.2 if shown < 1 else 0.0)
                    if s + 1 < stages and k + 1 < slots:
                        self.pulse(sc, (x, y), (x + dx, y + dy), float(now - k - 1), TEAL, op)
                else:
                    idle += shown >= 1
                    self.box(sc, x, y, 56, 42, DIM, op * smoothstep(0.0, 0.6, shown), radius=5)
        self.swatch(sc, 0.06, 0.845, TEAL, "a micro-batch (numbered) being worked on", op)
        self.swatch(sc, 0.06, 0.895, DIM, "idle: the bubble", op)
        total = busy + idle
        if total:
            self.text(sc, 0.95, 0.845, "busy so far: %d of %d slots" % (busy, total), op,
                      anchor="rm", strong=True, size=0.034)
        if u > 0.86:
            self.text(sc, 0.95, 0.895, "%d / (%d + %d - 1) = %.0f%% busy; outputs as one batch "
                      "to %s" % (self.batches, self.batches, stages,
                                 100.0 * busy / max(total, 1), number(self.pipe_gap)),
                      op * smoothstep(0.86, 0.93, u), anchor="rm", ink=dark(TEAL, 0.3),
                      size=0.032)

    def phase_4(self, sc, u, op):
        chart = LossChart(((0.11, 0.57), (0.40, 0.80)), (1.0, 32.0), (0.0, 1.0), xlog=True)
        self.axes(sc, chart, op, "micro-batches per batch", "share of slots that are busy",
                  xticks=[(m, "%d" % m) for m in self.micro],
                  yticks=[(v, "%d%%" % (100 * v)) for v in (0, 0.25, 0.5, 0.75, 1.0)])
        n = len(self.micro)
        at = min(int(float(np.clip((u - 0.04) / 0.80, 0.0, 1.0)) * n), n - 1)
        pts = [self.P(chart.fx(m), chart.fy(v)) for m, v in zip(self.micro, self.util)]
        self.polyline(sc, pts[:at + 1], 3.0, TEAL, op, bias=0.3)
        for k in range(at + 1):
            self.disk(sc, pts[k], 6.5 if k == at else 5.0, TEAL, op, glow=0.45 if k == at else 0.0,
                      lift=1.4, outline=0.0)
        m = int(self.micro[at])
        t = self.sweep[at]
        self.text(sc, 0.79, 0.385, "%d micro-batch%s" % (m, "" if m == 1 else "es"), op,
                  strong=True, size=0.036)
        w = 300.0
        h = 120.0
        self.heatmap(sc, (t > 0).astype(float), 0.79, 0.565, w, h, op, 1.0, TEAL, GREY,
                     border=AXIS, border_w=1.2)
        self.text(sc, 0.79, 0.565 + self.frac(0, h / 2)[1] + 0.04, "4 devices down, time across",
                  op)
        self.text(sc, 0.79, 0.80, "busy %d of %d slots: %.0f%%" % (
            int((t > 0).sum()), t.size, 100.0 * self.util[at]), op, strong=True, size=0.034)
        self.text(sc, 0.79, 0.86, "idle slots: always %d" % int((t == 0).sum()), op,
                  ink=dark(AXIS))


# ====================================================================== 88
class ZeroSharding(Systems):
    """What training a 7-billion-parameter model keeps in memory, drawn to
    scale on each of 8 devices, and the three ZeRO stages cutting it down:
    each device keeps one slice of the optimizer state, then of the
    gradients, then of the weights."""

    TITLE = "ZeRO: sharding the training state"
    PHASES = (("one device", 10.5), ("stage 1", 10.0), ("stage 2", 9.5), ("stage 3", 10.0),
              ("more devices", 10.5))
    BYTES = (2, 2, 12)                     # per parameter: 16-bit weight, 16-bit gradient,
    #                                        32-bit copy + Adam's two 32-bit running averages
    KINDS = (("weights", BLUE), ("gradients", RED), ("optimizer state", AMBER))
    ROWS = (("no ZeRO", MAGENTA), ("stage 1", VIOLET), ("stage 2", TEAL), ("stage 3", GREEN))
    PXGB, BASE = 2.2, 0.865
    COL0, COLDX, COLW = 0.075, 0.060, 42.0

    def setup(self):
        s = self.study
        self.parts = np.asarray(s.panels[1].data["values"], float)        # stage x kind, GB
        self.per_device = np.asarray(s.panels[2].data["values"], float)
        self.devices = int(round(self.parts[0, 2] / self.parts[1, 2]))
        self.params = self.parts[0, 0] / self.BYTES[0]                    # billions
        self.counts = np.asarray(s.panels[3].data["x"], float)
        self.scaling = np.asarray(s.panels[3].data["ys"], float)

    def subtitle_for(self, k):
        p, n = self.per_device, self.devices
        return [
            "training a %.0f-billion-parameter model keeps weights, gradients and the "
            "optimizer's state in memory: %.0f GB, and plain data parallelism puts all of it "
            "on every device" % (self.params, p[0]),
            "ZeRO stage 1: the optimizer state is cut into %d slices and each device keeps "
            "one; a device now holds %g GB instead of %.0f" % (n, p[1], p[0]),
            "stage 2: the gradients are sliced the same way, since a device only updates its "
            "own slice of the weights; %g GB per device" % p[2],
            "stage 3: the weights are sliced too, and a layer's weights are fetched from the "
            "other devices when it runs; %.0f GB per device, an %dth of %.0f" % (p[3], n, p[0]),
            "with every device holding everything, adding devices saves nothing; with stage 3 "
            "the memory per device halves each time the devices double",
        ][k]

    def _gb(self, gb):
        """A height in GB as a layout extent."""
        return self.frac(0, gb * self.PXGB)[1]

    def _block(self, sc, fx, bottom, gb, w, ink, op):
        """A block of ``gb`` standing on layout y ``bottom``."""
        h = gb * self.PXGB
        self.box(sc, fx, bottom - self._gb(gb) / 2, w, max(h - 1.0, 1.0), ink, op,
                 radius=min(3.0, h / 2))

    # ------------------------------------------------------------------ phase 1
    def phase_1(self, sc, u, op):
        x, w = 0.15, 132.0
        gb = self.parts[0]
        third = gb[2] / 3
        b = float(self.params)
        rows = (
            (gb[0], BLUE, "weights, 16-bit", "%.0fB x 2 bytes = %.0f GB" % (b, gb[0])),
            (gb[1], RED, "gradients, 16-bit", "%.0fB x 2 bytes = %.0f GB" % (b, gb[1])),
            (third, AMBER, "32-bit copy of the weights", "%.0fB x 4 bytes = %.0f GB" % (b, third)),
            (third, AMBER, "Adam's average of the gradients", "%.0fB x 4 bytes = %.0f GB" % (
                b, third)),
            (third, AMBER, "Adam's average of their squares", "%.0fB x 4 bytes = %.0f GB" % (
                b, third)),
        )
        bottom = self.BASE
        for k, (size, ink, name, sums) in enumerate(rows):
            a = smoothstep(0.04 + 0.13 * k, 0.15 + 0.13 * k, u)
            self._block(sc, x, bottom, size * a, w, ink, op * min(1.0, a * 4))
            y = bottom - self._gb(size) / 2
            self.text(sc, 0.245, y, name, op * a, anchor="lm", ink=dark(ink, 0.3), size=0.034)
            self.text(sc, 0.58, y, sums, op * a, anchor="lm", strong=True, size=0.034)
            bottom -= self._gb(size)
        self.text(sc, x, self.BASE + 0.04, "one device, to scale", op)
        a = smoothstep(0.72, 0.84, u)
        self.route(sc, [(0.845, self.BASE - self._gb(gb[0] + gb[1]) - 0.01),
                        (0.845, self.BASE - self._gb(gb.sum()) + 0.01)], AMBER, op * a, 2.4,
                   head=False)
        self.lines(sc, 0.86, self.BASE - self._gb(gb[0] + gb[1] + gb[2] / 2) - 0.04, [
            ("optimizer", dark(AMBER, 0.3)), ("state", dark(AMBER, 0.3)),
            ("%.0f GB" % gb[2], dark(AMBER, 0.3))], op * a, step=0.04)
        self.text(sc, 0.95, self.BASE + 0.04, "%.0f GB on one device; %d devices each holding "
                  "all of it: %.0f GB" % (gb.sum(), self.devices, gb.sum() * self.devices),
                  op * smoothstep(0.84, 0.94, u), anchor="rm", strong=True, size=0.032)

    # ------------------------------------------------------------------ phases 2-4
    def _progress(self, u):
        """``(sliced, others gone, collapsed)``, each 0..1, through a stage."""
        return (smoothstep(0.06, 0.26, u), smoothstep(0.32, 0.52, u),
                ease_in_out(smoothstep(0.56, 0.84, u)))

    def _held(self, stage, u):
        """GB of each kind on one device, ``u`` through ``stage`` (1-3)."""
        return lerp(self.parts[stage - 1], self.parts[stage], self._progress(u)[2])

    def _devices(self, sc, u, op, stage):
        n = self.devices
        cut = 3 - stage                                  # the kind being sliced in this stage
        a, b, c = self._progress(u)
        held = self._held(stage, u)
        for i in range(n):
            x = self.COL0 + i * self.COLDX
            bottom = self.BASE
            for k, (_name, ink) in enumerate(self.KINDS):
                piece = self.parts[0, k] / n
                if k != cut:
                    self._block(sc, x, bottom, self.parts[stage - 1, k], self.COLW, ink, op)
                    bottom -= self._gb(self.parts[stage - 1, k])
                    continue
                h = piece * self.PXGB
                for j in range(n):
                    mine = j == i
                    at = bottom - self._gb(piece * j * (1.0 - c if mine else 1.0))
                    self.box(sc, x, at - self._gb(piece) / 2, self.COLW,
                             max(h - (1.2 * a if h > 6 else 0.0), 1.0),
                             ink if mine else mix(ink, soft(ink, 0.72), a),
                             op * (1.0 if mine else 1.0 - b), radius=min(3.0, h / 2),
                             glow=0.25 * a * (1 - c) if mine else 0.0)
                bottom -= self._gb(held[k])
            self.text(sc, x, self.BASE + 0.04, "%d" % (i + 1), op)
        self.text(sc, 0.05, 0.30, "%d devices, drawn to scale" % n, op, anchor="lm", strong=True,
                  size=0.034)
        top = self.BASE - self._gb(held.sum())
        right = self.COL0 + (n - 1) * self.COLDX + self.frac(self.COLW / 2 + 4)[0]
        self.route(sc, [(right + 0.035, top), (right + 0.005, top)], AXIS, op, 2.0)
        self.text(sc, right + 0.042, top, "%g GB" % round(float(held.sum()), 1), op,
                  anchor="lm", strong=True, size=0.034)
        # what the picture measures
        for k, (name, ink) in enumerate(self.KINDS):
            self.swatch(sc, 0.68, 0.335 + 0.042 * k, ink, name, op)
        self.text(sc, 0.67, 0.49, "GB on each device", op, anchor="lm", strong=True, size=0.034)
        full = self.span(0.11)[0]
        for r, (name, ink) in enumerate(self.ROWS[:stage + 1]):
            if r == stage and c <= 0.0:
                break
            value = self.per_device[r]
            self.hbar(sc, 0.785, 0.565 + 0.068 * r, full * value / self.per_device[0], ink,
                      op, name, "%g" % value, 1.0 if r < stage else c, thick=26.0, track=full)
        self.lines(sc, 0.67, 0.832, [
            "stage %d: the %s" % (stage, self.KINDS[cut][0]),
            "cut into %d slices, one slice" % n, "kept on each device"], op * a, step=0.042)

    def phase_2(self, sc, u, op):
        self._devices(sc, u, op, 1)

    def phase_3(self, sc, u, op):
        self._devices(sc, u, op, 2)

    def phase_4(self, sc, u, op):
        self._devices(sc, u, op, 3)

    # ------------------------------------------------------------------ phase 5
    def phase_5(self, sc, u, op):
        counts, (flat, sharded) = self.counts, self.scaling
        chart = LossChart(((0.11, 0.57), (0.40, 0.80)), (counts[0], counts[-1]),
                          (0.0, 120.0), xlog=True)
        self.axes(sc, chart, op, "devices", "GB on each device",
                  xticks=[(m, "%d" % m) for m in counts],
                  yticks=[(v, "%d" % v) for v in (0, 40, 80, 120)])
        n = len(counts)
        at = min(int(float(np.clip((u - 0.04) / 0.80, 0.0, 1.0)) * n), n - 1)
        for ys, ink in ((flat, MAGENTA), (sharded, GREEN)):
            pts = [self.P(chart.fx(m), chart.fy(v)) for m, v in zip(counts, ys)]
            self.polyline(sc, pts[:at + 1], 3.0, ink, op, bias=0.3)
            for k in range(at + 1):
                self.disk(sc, pts[k], 6.5 if k == at else 5.0, ink, op,
                          glow=0.45 if k == at else 0.0, lift=1.4, outline=0.0)
        m = int(counts[at])
        self.text(sc, 0.05, 0.305, "%d device%s: what each one holds, to scale" % (
            m, "" if m == 1 else "s"), op, anchor="lm", strong=True, size=0.034)
        base = 0.855
        for x, name, ink, share in ((0.71, "no ZeRO", MAGENTA, 1.0),
                                    (0.87, "stage 3", GREEN, 1.0 / m)):
            bottom = base
            for k, (_kind, colour) in enumerate(self.KINDS):
                gb = self.parts[0, k] * share
                self._block(sc, x, bottom, gb, 84.0, colour, op)
                bottom -= self._gb(gb)
            total = float((flat if share == 1.0 else sharded)[at])
            self.text(sc, x, bottom - 0.035, "%g GB" % total, op, strong=True, size=0.034)
            self.text(sc, x, base + 0.04, name, op, ink=dark(ink, 0.3), size=0.032)


# ====================================================================== 89
KINDS = ("start", "forward", "recompute", "backward", "end")      # the study's step codes


def events_of(panel, spacing):
    """The study's record of one backward pass at ``spacing``, a step at a
    time: ``dict(kind, at, kept, temp, done, calls)`` with ``kind`` "forward" /
    "recompute" / "backward", ``at`` the activation computed or the layer whose
    gradient was taken, ``kept`` the stored activations in memory, ``temp``
    the ones computed to be used and dropped, ``done`` the layers whose
    gradient is known and ``calls`` the layer evaluations so far."""
    rows = np.asarray(panel["trace"]).astype(int)
    pick = rows[:, 0] == spacing
    return [dict(kind=KINDS[kind], at=int(at), calls=int(calls),
                 kept=set(np.flatnonzero(state == 1).tolist()),
                 temp=set(np.flatnonzero(state == 2).tolist()),
                 done=set(np.flatnonzero(done).tolist()))
            for (_k, kind, at, calls), state, done in zip(
                rows[pick], np.asarray(panel["held"])[pick], np.asarray(panel["done"])[pick])]


def held(event):
    """Activations in memory at ``event``: the stored ones and the passing ones."""
    return len(event["kept"]) + len(event["temp"])


def tiles(gradient):
    """A first-layer gradient ``(49 pixels, 16 units)`` as a 4 x 4 sheet of
    7 x 7 images, one per hidden unit."""
    return np.asarray(gradient, float).T.reshape(4, 4, 7, 7).transpose(0, 2, 1, 3).reshape(28, 28)


def span_text(values):
    """``[3, 4, 5, 6]`` as "3 to 6"; anything else listed."""
    v = list(values)
    if len(v) > 2 and v == list(range(v[0], v[-1] + 1)):
        return "%d to %d" % (v[0], v[-1])
    return " and ".join(str(k) for k in v)


class ActivationCheckpointing(Systems):
    """A 16-layer MNIST classifier: backpropagation storing every activation,
    then storing every fourth and recomputing each segment as the gradient
    reaches it; memory against layer runs at every spacing; and the two
    gradients compared."""

    TITLE = "Activation checkpointing"
    PHASES = (("store all", 10.5), ("every 4th", 17.0), ("spacing", 12.5), ("gradient", 10.5))
    X0, DX = 0.076, 0.053                   # activation 0's x, and the step to the next
    TOP, LOW, MARK = 0.44, 0.555, 0.64      # stored row, passing row, gradient marks

    def setup(self):
        s = self.study
        mem, work, grad, same = (p.data for p in s.panels)
        self.spacings = [int(v) for v in mem["spacings"]]
        self.mem = [int(v) for v in mem["values"]]
        self.work = [int(v) for v in work["values"]]
        self.errors = [float(v) for v in same["values"]]
        self.layers = np.asarray(mem["done"]).shape[1]
        self.runs = {k: events_of(mem, k) for k in self.spacings}
        self.plain = self.runs[1]                       # spacing 1: every activation stored
        self.k = int(s.metadata["sqrt_layers"])
        self.best = self.spacings.index(self.k)
        self.ck = self.runs[self.k]
        self.least = [k for k, m in zip(self.spacings, self.mem) if m == min(self.mem)]
        self.plain_grad = tiles(grad["plain"])
        self.ck_grad = tiles(np.asarray(grad["checkpointed"])[self.best])
        self.vg = float(np.abs(self.plain_grad).max())
        self.diff = max(self.errors)
        self.plain_peak, self.plain_calls = self.mem[0], self.work[0]
        self.digits = np.asarray(grad["digits"])
        self.accuracy = float(s.metadata["heldout_loss_accuracy_after"][1])

    def subtitle_for(self, k):
        n, best = self.layers, self.best
        return [
            "backpropagation needs every layer's output on the way back; storing them all for "
            "this %d-layer MNIST classifier keeps %d activations in memory" % (
                n, self.plain_peak),
            "checkpointing: keep only every %dth activation; on the way back, run each segment "
            "forward again from its checkpoint, use those activations, and drop them" % self.k,
            "wider spacing stores fewer checkpoints but recomputes longer segments: the most "
            "held at once falls from %d to %d at spacings %s, then rises, and layer runs climb "
            "from %d to %d" % (self.plain_peak, min(self.mem), span_text(self.least),
                               self.plain_calls, max(self.work)),
            "the gradient is the same (largest difference %s); at spacing %d memory fell from "
            "%d activations to %d, and the layers ran %d times instead of %d" % (
                number(self.diff), self.k, self.plain_peak, self.mem[best], self.work[best],
                self.plain_calls),
        ][k]

    # ------------------------------------------------------------------ the chain
    def _x(self, i):
        return self.X0 + i * self.DX

    def _row(self, i, event):
        return self.TOP if i in event["kept"] else self.LOW

    def _chain(self, sc, u, op, events, title, recompute):
        m = len(events)
        tt = float(np.clip((u - 0.03) / 0.90, 0.0, 1.0)) * (m - 1)
        at = min(int(tt), m - 1)
        within = tt - at
        e = events[at]
        nxt = events[min(at + 1, m - 1)]
        kept, temp, done = e["kept"], e["temp"], e["done"]
        self.text(sc, 0.05, 0.315, title, op, anchor="lm", strong=True, size=0.034)
        for i, img in enumerate(self.digits[:5]):
            self.digit(sc, img, 0.835 + i * 0.027, 0.315, 24, op)
        self.text(sc, 0.05, self.TOP - 0.068, "stored", op, anchor="lm", ink=dark(BLUE))
        if recompute:
            self.text(sc, 0.95, self.TOP - 0.068, "computed, used, then dropped (lower row)", op,
                      anchor="rm", ink=dark(GREEN, 0.3))
        for i in range(self.layers + 1):
            x = self._x(i)
            on = i in kept
            self.box(sc, x, self.TOP, 36, 44, BLUE if on else PAPER, op * (1.0 if on else 0.8),
                     "%d" % i, ink=WHITE if on else dark(DIM, 0.25), border=None if on else DIM,
                     size=0.032, radius=5)
            if i in temp:
                self.box(sc, x, self.LOW, 36, 44, GREEN, op, "%d" % i, ink=WHITE, size=0.032,
                         radius=5, glow=0.15)
            if i < self.layers:
                self.box(sc, x + self.DX / 2, self.MARK, 40, 12, RED if i in done else DIM,
                         op * (1.0 if i in done else 0.6), radius=3)
        # the step under way
        kind = nxt["kind"] if at + 1 < m else "end"
        j = nxt["at"]
        if kind in ("forward", "recompute"):
            a, b = (self._x(j - 1), self._row(j - 1, e)), (self._x(j), self._row(j, nxt))
            self.pulse(sc, a, b, within, AMBER, op, bend=0.09 if a[1] == b[1] == self.TOP else 0.0,
                       r=6.5)
        elif kind == "backward":
            self.pulse(sc, (self._x(j + 1), self.MARK), (self._x(j), self.MARK), within, RED, op,
                       r=6.5)
        self.text(sc, 0.05, self.MARK + 0.045, "gradient known for %d of %d layers" % (
            len(done), self.layers), op, anchor="lm", ink=dark(RED))
        say = {"forward": "forward: layer %d runs" % j,
               "recompute": "layer %d runs again, from the checkpoint" % j,
               "backward": "the gradient crosses layer %d" % (j + 1),
               "end": "every layer has its gradient"}.get(kind, "")
        self.text(sc, 0.95, self.MARK + 0.045, say, op, anchor="rm", strong=True, size=0.034)
        now = held(e)
        peak = max(held(v) for v in events[:at + 1])
        full = self.span(0.30)[0]
        self.hbar(sc, 0.31, 0.79, full * now / self.plain_peak, BLUE, op, "activations in memory",
                  "%d now, %d at the peak" % (now, peak), thick=26.0, track=full)
        self.hbar(sc, 0.31, 0.87, full * e["calls"] / max(self.work), GREEN, op,
                  "times a layer has run", "%d" % e["calls"], thick=26.0, track=full)

    def phase_1(self, sc, u, op):
        self._chain(sc, u, op, self.plain, "%d activations with %d layers between them: store "
                    "every one" % (self.layers + 1, self.layers), False)

    def phase_2(self, sc, u, op):
        self._chain(sc, u, op, self.ck, "the same network, storing only every %dth activation"
                    % self.k, True)

    # ------------------------------------------------------------------ phase 3
    def phase_3(self, sc, u, op):
        ks = self.spacings
        n = len(ks)
        self.text(sc, 0.05, 0.315, "one gradient taken at every spacing, and what it cost", op,
                  anchor="lm", strong=True, size=0.034)
        top = 4 * int(np.ceil((max(self.work) + 1) / 4))
        chart = LossChart(((0.10, 0.50), (0.40, 0.79)), (ks[0], ks[-1]), (0.0, float(top)))
        self.axes(sc, chart, op, "checkpoint spacing", "",
                  xticks=[(k, "%d" % k) for k in ks if k in (1, 2, 4, 8, 12, 16)],
                  yticks=[(v, "%d" % v) for v in range(0, top + 1, 8)])
        chart.mark(sc, self.P, self.k, AXIS, op)
        at = min(int(float(np.clip((u - 0.04) / 0.74, 0.0, 1.0)) * n), n - 1)
        for ys, ink in ((self.work, GREEN), (self.mem, BLUE)):
            pts = [self.P(chart.fx(k), chart.fy(v)) for k, v in zip(ks, ys)]
            self.polyline(sc, pts[:at + 1], 3.0, ink, op, bias=0.3)
            for j in range(at + 1):
                self.disk(sc, pts[j], 6.5 if j == at else 4.5, ink, op,
                          glow=0.45 if j == at else 0.0, lift=1.4, outline=0.0)
        k = ks[at]
        e = max(self.runs[k], key=held)
        x0, dx = 0.575, 0.0235
        self.text(sc, x0 - 0.012, 0.40, "every %d: the moment of peak memory" % k, op,
                  anchor="lm", strong=True, size=0.034)
        for i in range(self.layers + 1):
            x = x0 + i * dx
            on = i in e["kept"]
            self.box(sc, x, 0.47, 19, 24, BLUE if on else PAPER, op * (1.0 if on else 0.7),
                     border=None if on else DIM, radius=3)
            if i in e["temp"]:
                self.box(sc, x, 0.52, 19, 24, GREEN, op, radius=3)
        self.swatch(sc, x0 - 0.004, 0.60, BLUE, "held at once: %d = %d stored + %d recomputed" % (
            self.mem[at], len(e["kept"]), len(e["temp"])), op)
        self.swatch(sc, x0 - 0.004, 0.66, GREEN, "times a layer ran: %d = %d + %d again" % (
            self.work[at], self.layers, self.work[at] - self.layers), op)
        self.text(sc, x0 - 0.012, 0.74, "dashed: spacing %d, the square root of %d" % (
            self.k, self.layers), op, anchor="lm", ink=dark(AXIS))
        self.text(sc, 0.5, 0.915, "fewest held at once: %d, at spacings %s; every wider spacing "
                  "costs more layer runs" % (min(self.mem), span_text(self.least)),
                  op * smoothstep(0.80, 0.9, u), strong=True, size=0.032)

    # ------------------------------------------------------------------ phase 4
    def phase_4(self, sc, u, op):
        side = 176.0
        self.text(sc, 0.15, 0.315, "storing everything", op, strong=True, size=0.032)
        self.heatmap(sc, self.plain_grad, 0.15, 0.56, side, side, op, self.vg, border=AXIS,
                     border_w=1.4)
        self.text(sc, 0.27, 0.56, "=", op * smoothstep(0.2, 0.35, u), strong=True, size=0.06)
        self.text(sc, 0.39, 0.315, "checkpoint every %d" % self.k, op, strong=True, size=0.032)
        rows = len(self.ck_grad)
        self.heatmap(sc, self.ck_grad * (np.arange(rows)[:, None] < rows * smoothstep(
            0.05, 0.3, u)), 0.39, 0.56, side, side, op, self.vg, border=AXIS, border_w=1.4)
        self.lines(sc, 0.27, 0.80, ["first-layer gradient: a 7 x 7 tile for each of 16 units",
                                    "largest difference, over every weight: %s" % number(
                                        self.diff)],
                   op * smoothstep(0.3, 0.45, u), anchor="mm", step=0.045)
        full = self.span(0.15)[0]
        b = self.best
        for g, (title, plain, ck, y) in enumerate((
                ("activations in memory at the peak", self.plain_peak, self.mem[b], 0.36),
                ("times a layer runs forward", self.plain_calls, self.work[b], 0.64))):
            a = smoothstep(0.35 + 0.2 * g, 0.5 + 0.2 * g, u)
            self.text(sc, 0.56, y, title, op * a, anchor="lm", strong=True, size=0.032)
            top = max(plain, ck)
            for r, (name, ink, value) in enumerate((("store all", VIOLET, plain),
                                                    ("every %d" % self.k, TEAL, ck))):
                self.hbar(sc, 0.685, y + 0.07 + r * 0.065, full * value / top, ink, op * a, name,
                          "%d" % value, (u - 0.38 - 0.2 * g) / 0.2, thick=26.0, track=full)


# ====================================================================== 90
OP_INKS = (RED, BLUE, GREEN, AMBER)        # one colour per operation, in every phase


class Roofline(Systems):
    """Four operations counted (arithmetic done, bytes moved); the machine's
    two ceilings timed on their own, a large copy and in-cache arithmetic; the
    operations fed through a machine with those ceilings; and where each one,
    timed, lands under the roof the two ceilings draw."""

    TITLE = "The roofline performance model"
    PHASES = (("count", 10.5), ("ceilings", 10.5), ("feed", 11.0), ("roofline", 12.5),
              ("limits", 10.0))
    YS = (0.415, 0.545, 0.675, 0.805)

    def setup(self):
        s = self.study
        count, speed, roof, took = (p.data for p in s.panels)
        self.names = [str(v) for v in count["labels"]]
        self.intensity = np.asarray(count["values"], float)              # operations per byte
        self.ops = list(zip(self.names, OP_INKS, (str(v) for v in count["what"]),
                            np.asarray(count["flops"], float), np.asarray(count["bytes"], float)))
        self.speed = np.asarray(speed["values"], float)                  # billion a second
        self.low, self.high = np.asarray(speed["low"], float), np.asarray(speed["high"], float)
        self.ceiling = np.asarray(speed["ceiling"], float)
        self.reach = np.asarray(speed["reach"], float)
        self.xs = np.asarray(roof["x"], float)
        self.roof = np.asarray(roof["ys"], float)[0]
        self.bandwidth, self.bandwidth_low, self.bandwidth_high = (
            float(v) for v in roof["bandwidth"])                         # GB a second
        self.peak, self.peak_low, self.peak_high = (float(v) for v in roof["peak"])
        self.copy_rates = np.asarray(roof["copy_rates"], float)          # every timing
        self.peak_rates = np.asarray(roof["peak_samples"], float)
        self.copy_mb = int(roof["copy_mb"])
        self.peak_size = int(roof["peak_size"])
        self.repeats = len(self.copy_rates)
        self.ridge = self.peak / self.bandwidth
        self.millis = np.asarray(took["values"], float) / 1000.0

    def subtitle_for(self, k):
        i = self.intensity
        lo, hi = int(np.argmin(i)), int(np.argmax(i))
        shares = ", ".join("%s %.0f%%" % (self.names[j], 100 * self.reach[j])
                           for j in np.argsort(-self.reach))
        return [
            "count what an operation does: its arithmetic and the bytes it moves through "
            "memory; %s does %s operations per byte, %s only %s" % (
                self.names[hi], number(i[hi]), self.names[lo], number(i[lo])),
            "the machine's two ceilings, each timed %d times on its own: copying %d MB shows "
            "how fast memory delivers bytes, multiplying small matrices that stay in cache how "
            "fast arithmetic runs" % (self.repeats, self.copy_mb),
            "memory hands over at most %s GB a second here; an operation that does little "
            "with each byte leaves the arithmetic waiting, and one that does a lot makes the "
            "bytes queue" % number(self.bandwidth),
            "the roofline: speed is capped by bytes per second times operations per byte, "
            "until the arithmetic itself is the cap (%s billion a second); then each "
            "operation is timed and placed under it" % number(self.peak),
            "the roof is a ceiling, not a forecast: of the roof above it, timed in NumPy on "
            "this machine, each reaches: %s" % shares,
        ][k]

    # ------------------------------------------------------------------ phase 1
    def phase_1(self, sc, u, op):
        self.text(sc, 0.48, 0.325, "arithmetic", op, strong=True)
        self.text(sc, 0.59, 0.325, "bytes moved", op, strong=True)
        self.text(sc, 0.675, 0.325, "operations per byte", op, anchor="lm", strong=True)
        full = self.span(0.19)[0]
        top = float(self.intensity.max())
        for r, (name, ink, what, flops, nbytes) in enumerate(self.ops):
            a = smoothstep(0.03 + 0.2 * r, 0.13 + 0.2 * r, u)
            y = self.YS[r]
            self.box(sc, 0.115, y, 124, 46, ink, op * a, name, ink=WHITE, size=0.036)
            self.text(sc, 0.195, y, what, op * a, anchor="lm")
            b = smoothstep(0.08 + 0.2 * r, 0.16 + 0.2 * r, u)
            self.text(sc, 0.48, y, number(flops), op * b, strong=True, size=0.034)
            self.text(sc, 0.59, y, number(nbytes), op * b, strong=True, size=0.034)
            self.text(sc, 0.535, y, "/", op * b, size=0.034)
            self.hbar(sc, 0.675, y, full * self.intensity[r] / top, ink, op * b, "",
                      number(self.intensity[r]), (u - 0.12 - 0.2 * r) / 0.12, thick=30.0,
                      track=full)
        self.text(sc, 0.5, 0.905, "counts are exact: 64-bit numbers, 8 bytes each, every "
                  "matrix read or written once", op * smoothstep(0.85, 0.95, u))

    # ------------------------------------------------------------------ phase 2
    def _timings(self, sc, y, title, samples, median, ink, unit, op, u):
        """One ceiling: every timing as a dot on a scale from 0, and the median."""
        x0, x1 = 0.09, 0.70
        top = float(nice_ticks(0.0, float(samples.max()) * 1.02, 4)[-1])
        if top < samples.max():
            top = float(samples.max())
        self.text(sc, 0.05, y - 0.075, title, op, anchor="lm", strong=True, size=0.032)
        self.box(sc, (x0 + x1) / 2, y, self.span(x1 - x0)[0], 30, TRACK, op * 0.5, radius=4)
        for v in nice_ticks(0.0, top, 4):
            self.text(sc, lerp(x0, x1, v / top), y + 0.045, number(v), op, ink=dark(AXIS))
        shown = float(np.clip(u, 0.0, 1.0)) * len(samples)
        for j, v in enumerate(samples):
            a = float(np.clip(shown - j, 0.0, 1.0))
            self.disk(sc, self.P(lerp(x0, x1, v / top), y), 6.0, ink, op * a * 0.85,
                      glow=0.5 * (1 - a) if a > 0 else 0.0, lift=1.4, outline=0.0)
        done = smoothstep(0.92, 1.0, u)
        xm = lerp(x0, x1, median / top)
        self.polyline(sc, [self.P(xm, y - 0.038), self.P(xm, y + 0.028)], 3.4, dark(ink, 0.35),
                      op * done, bias=0.3)
        self.text(sc, 0.95, y - 0.02, "median %s" % number(median), op * done, anchor="rm",
                  strong=True, size=0.036)
        self.text(sc, 0.95, y + 0.03, "%s, %s to %s" % (unit, number(samples.min()),
                                                       number(samples.max())),
                  op * done, anchor="rm", ink=dark(AXIS))

    def phase_2(self, sc, u, op):
        self._timings(sc, 0.44, "memory: copy %d MB from one array to another, computing nothing"
                      % self.copy_mb, self.copy_rates, self.bandwidth, TEAL, "GB a second", op,
                      (u - 0.04) / 0.36)
        self._timings(sc, 0.70, "arithmetic: multiply two %d x %d matrices, which stay in cache, "
                      "over and over" % (self.peak_size, self.peak_size), self.peak_rates,
                      self.peak, VIOLET, "billion a second", op, (u - 0.44) / 0.36)
        self.text(sc, 0.5, 0.905, "each dot is one timing; the medians are the roof's two "
                  "ceilings, and neither comes from the four operations",
                  op * smoothstep(0.84, 0.94, u), strong=True, size=0.032)

    # ------------------------------------------------------------------ phase 3
    def phase_3(self, sc, u, op):
        x_mem, x_alu, x_bar = 0.265, 0.535, 0.61
        full = self.span(0.17)[0]
        self.text(sc, x_mem, 0.325, "memory", op, strong=True)
        self.text(sc, x_alu, 0.325, "arithmetic", op, strong=True)
        self.text(sc, x_bar, 0.325, "speed it can reach, of %s" % number(self.peak), op,
                  anchor="lm", strong=True)
        grow = smoothstep(0.25, 0.75, u)
        for r, (name, ink, _what, _f, _b) in enumerate(self.ops):
            y = self.YS[r]
            self.box(sc, 0.115, y, 124, 46, ink, op, name, ink=WHITE, size=0.036)
            self.box(sc, x_mem, y, 72, 46, soft(AXIS, 0.6), op, border=AXIS, radius=5)
            a, b = (x_mem + 0.045, y), (x_alu - 0.062, y)
            self.arrow(sc, a, b, AXIS, op * 0.8, 1.0, 1.6)
            busy = self.ceiling[r] / self.peak
            # bytes flow as fast as memory allows, or as fast as the arithmetic can take them
            pace = self.ceiling[r] / (self.bandwidth * self.intensity[r])
            for j in range(5):
                self.pulse(sc, a, b, (u * 9.0 * pace + j / 5.0) % 1.0, ink, op, r=5.0)
            self.box(sc, x_alu, y, 104, 46, mix(PAPER, ink, 0.25 + 0.75 * busy * grow), op,
                     "x %s" % number(self.intensity[r]), ink=WHITE if busy * grow > 0.5 else
                     dark(ink, 0.4), border=ink, size=0.034, radius=5)
            self.hbar(sc, x_bar, y, full * busy, ink, op, "", "%s%s" % (
                number(self.ceiling[r]), ", waiting" if busy < 0.999 else ", flat out"),
                grow, thick=30.0, track=full)
        self.text(sc, 0.5, 0.905, "each byte is used in that many operations; where the "
                  "arithmetic is flat out, the bytes queue", op * smoothstep(0.1, 0.25, u))

    # ------------------------------------------------------------------ phase 4
    def phase_4(self, sc, u, op):
        chart = LossChart(((0.11, 0.60), (0.37, 0.80)), (self.xs[0], self.xs[-1]), (0.2, 400.0),
                          xlog=True, ylog=True)
        self.axes(sc, chart, op, "operations per byte", "billion operations a second",
                  xticks=[(v, number(v)) for v in (0.01, 0.1, 1, 10, 100)],
                  yticks=[(v, number(v)) for v in (1, 10, 100)])
        m = max(2, int(len(self.xs) * smoothstep(0.03, 0.40, u)))
        pts = [self.P(chart.fx(x), chart.fy(y)) for x, y in zip(self.xs[:m], self.roof[:m])]
        self.polyline(sc, pts, 3.2, VIOLET, op, bias=0.3)
        a = smoothstep(0.40, 0.50, u)
        self.text(sc, chart.fx(0.05) + 0.012, chart.fy(self.bandwidth * 0.05) + 0.055,
                  "memory is the cap", op * a, anchor="lm", ink=dark(VIOLET))
        # above the flat of the roof, clear of the grid lines wherever the peak came out
        flat = chart.fy(self.peak)
        y = next(flat - d for d in (0.04, 0.065, 0.09, 0.115)
                 if all(abs(flat - d - chart.fy(v)) > 0.024 for v in (1, 10, 100)))
        self.text(sc, chart.fx(60.0), y, "arithmetic is the cap", op * a, anchor="rm",
                  ink=dark(VIOLET))
        self.text(sc, 0.66, 0.36, "measured, billion a second", op, anchor="lm", strong=True)
        for r, (name, ink, _what, _f, _b) in enumerate(self.ops):
            b = smoothstep(0.50 + 0.09 * r, 0.58 + 0.09 * r, u)
            x = chart.fx(self.intensity[r])
            # the slowest and the fastest of its timings
            self.polyline(sc, [self.P(x, chart.fy(self.low[r])), self.P(x, chart.fy(self.high[r]))],
                          2.2, dark(ink, 0.25), op * b, bias=0.3)
            self.disk(sc, self.P(x, chart.fy(self.speed[r])), 8.0, ink, op * b, glow=0.4,
                      lift=1.6, outline=0.0)
            y = 0.44 + 0.095 * r
            self.box(sc, 0.725, y, 124, 40, ink, op * b, name, ink=WHITE, size=0.034)
            self.text(sc, 0.805, y, "%s of %s" % (number(self.speed[r]), number(self.ceiling[r])),
                      op * b, anchor="lm", strong=True, size=0.034)
        self.lines(sc, 0.66, 0.825, ["of: the roof above it", "bar: its slowest to fastest timing"],
                   op * smoothstep(0.85, 0.95, u), step=0.042)

    # ------------------------------------------------------------------ phase 5
    def phase_5(self, sc, u, op):
        self.text(sc, 0.205, 0.325, "what caps it", op, anchor="lm", strong=True)
        self.text(sc, 0.36, 0.325, "share of its roof reached", op, anchor="lm", strong=True)
        self.text(sc, 0.95, 0.325, "time taken", op, anchor="rm", strong=True)
        full = self.span(0.20)[0]
        for r, (name, ink, _what, _f, _b) in enumerate(self.ops):
            a = smoothstep(0.03 + 0.12 * r, 0.13 + 0.12 * r, u)
            y = self.YS[r]
            self.box(sc, 0.115, y, 124, 46, ink, op * a, name, ink=WHITE, size=0.036)
            self.text(sc, 0.205, y, "memory" if self.intensity[r] < self.ridge else "arithmetic",
                      op * a, anchor="lm", size=0.034)
            self.hbar(sc, 0.36, y, full * min(self.reach[r], 1.0), ink, op * a, "", "%.0f%%" % (
                100 * self.reach[r]), (u - 0.1 - 0.12 * r) / 0.15, thick=30.0, track=full)
            self.text(sc, 0.95, y, "%s thousandths of a second" % number(self.millis[r]), op * a,
                      anchor="rm")
        over = [self.names[j] for j in range(len(self.names)) if self.reach[j] > 1.0]
        line = ("%s came out above the roof: the copy that sets it was timed at %s to %s GB a "
                "second" % (" and ".join(over), number(self.bandwidth_low),
                            number(self.bandwidth_high)) if over else
                "the roof was timed on its own, on one core; the cap changes at %s operations "
                "per byte" % number(self.ridge))
        self.text(sc, 0.5, 0.905, line, op * smoothstep(0.6, 0.75, u), strong=True, size=0.032)


ANIMATIONS = {
    "data-parallel-allreduce": DataParallelAllReduce,
    "tensor-and-pipeline-parallel": TensorAndPipelineParallel,
    "zero-sharding": ZeroSharding,
    "activation-checkpointing": ActivationCheckpointing,
    "roofline": Roofline,
}

"""Memory across a gap: a plain RNN against an LSTM on a task that needs it.

The task (:mod:`voxanim.memory`): say the first value of a sequence after 49
steps of noise.  Both networks (8 units each, trained the same way; the
LSTM's forget gate starts open) read the same sequence, drawn unrolled
through time as in :mod:`voxanim.recurrent`:

    input x_t            the cue at step 0, then noise
    RNN: hidden h        8 x 50
    LSTM: cell c, h      8 x 50 each

Five phases:

1. the task,
2. forward -- both run in step; the RNN answers about 0, the LSTM the cue,
3. memory -- the same sequence with the cue flipped: the bands switch to the
   difference between the two runs, which is what each state still knows
   about the cue.  The RNN's fades within a few steps; the LSTM's cell state
   keeps it,
4. backward -- the gradient of the final error flows back through time,
   drawn as its magnitude on a log scale: through tanh and Wh it shrinks
   every step and reaches the cue 10,000 times smaller; the cell state hands
   it back almost unchanged,
5. training -- the loss curves: the LSTM learns the task, the RNN stays at
   chance, and so does an LSTM whose forget gate starts half shut (the
   usual bias of 1): the gate is the memory only when it is open.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from .conv import Tempo
from .easing import ease_in_out, remap, smoothstep, soft_band
from .memory import CHANCE
from .netanim import NetAnimation, amax, div_colors, lit_colors, particle

RNN_INK = (0.75, 0.22, 0.22)                  # the RNN's curve
LSTM_INK = tuple(float(v) for v in C.INPUT)   # the LSTM's
LEAKY_INK = (0.36, 0.38, 0.42)                # the half-shut LSTM's
GRAD_LO, GRAD_HI = C.rgb("#3a3f44"), C.rgb("#c9fff6")   # gradient magnitude: dark .. lit
GRAD_RANGE = (-6.0, -1.0)                     # log10 |gradient| at the dark and the lit end
LOSS_RANGE = (-3.3, -0.3)                     # log10 loss at the bottom and top of the chart
SMOOTH = 20                                   # batches the drawn loss is averaged over
CHART = ((0.665, 0.93), (0.32, 0.78))         # the chart's box ((x0, x1), (top, bottom)), layout
NOTE_X = 0.635                                # frame x of the read-outs right of the bands
# the rows of sheets, top to bottom: key, label, rows, gap above (world units)
ROWS = (("x", "input: cue, then noise", 1, 0.0), ("rh", "RNN: hidden h", 8, 1.5),
        ("lc", "LSTM: cell state c", 8, 2.5), ("lh", "LSTM: hidden h", 8, 1.2))
STACK_AT = (0.39, 0.57)                       # frame position of the stack's centre
PITCH = (1.0, 1.0)                            # cell pitch (rows, steps): 50 steps must fit
FRAME = (0.96, 0.96, 0.96)


def sci(v):
    """``1e-5`` style: a magnitude for a read-out."""
    return ("%.0e" % v).replace("e-0", "e-").replace("e+0", "e")


class GatedMemory(NetAnimation):
    """The memory task through a trained RNN and a trained LSTM side by side.

    ``rnn`` and ``lstm`` are :class:`voxanim.memory.MemoryTrace` s; ``leaky``
    is the :class:`voxanim.memory.MemoryRun` of the LSTM whose forget gate
    starts half shut, for the training chart.
    """

    PHASES = ("the task", "forward", "memory", "backward", "training")

    def __init__(self, rnn, lstm, leaky):
        super().__init__("Memory across a gap: RNN vs LSTM", "", 8.5)
        self.rnn, self.lstm = rnn, lstm
        self.T, self.n = len(rnn.x), len(rnn.P["b"])
        self.hold = 2.5
        # the rows, stacked in world space so that a column is one thin box
        self.rows = {}
        centre = self.cam.frac(*STACK_AT)
        height = sum(rows + gap for _k, _l, rows, gap in ROWS)
        z = centre[2] + height / 2
        for key, label, rows, gap in ROWS:
            s = self.sheet((rows, self.T), STACK_AT, 1.0, key, pitch=PITCH)
            z -= gap + rows / 2
            c = centre.copy()
            c[2] = z
            s.block.place(c)
            z -= rows / 2
            self.rows[key] = (s, label)
        # per band (n, T): the run, the flipped run and the gradient reaching it
        self.vals = {"rh": rnn.f["h"][1:].T, "lc": lstm.f["c"][1:].T, "lh": lstm.f["h"][1:].T}
        self.vals2 = {"rh": rnn.f2["h"][1:].T, "lc": lstm.f2["c"][1:].T,
                      "lh": lstm.f2["h"][1:].T}
        self.grads = {"rh": rnn.d["dh"].T, "lc": lstm.d["dc"].T, "lh": lstm.d["dh"].T}
        self.at_cue = {"rnn": float(np.abs(rnn.d["dh"][0]).max()),
                       "lstm": float(np.abs(lstm.d["dc"][0]).max())}
        kernel = np.ones(SMOOTH) / SMOOTH
        self.curves = {k: np.convolve(r.losses, kernel, "valid")
                       for k, r in (("rnn", rnn), ("lstm", lstm), ("leaky", leaky))}

        # ---- timeline -----------------------------------------------------------------
        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("task", 3.2, 0.3)
        fw = Tempo(slow=0.7, fast=0.11, ramp_in=3.0, ramp_out=2.0).durations(self.T)
        self.f_starts, self.f_durs = t + np.concatenate([[0.0], np.cumsum(fw)[:-1]]), fw
        add("fwd", float(fw.sum()), 1.6)
        add("memory", 6.0, 0.5)
        bw = Tempo(slow=0.6, fast=0.1, ramp_in=3.0, ramp_out=2.0).durations(self.T)
        self.b_starts, self.b_durs = t + np.concatenate([[0.0], np.cumsum(bw)[:-1]]), bw
        add("bwd", float(bw.sum()), 1.6)
        add("train", 7.0, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["fwd"][0], Tm["memory"][0], Tm["bwd"][0], Tm["train"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _sweep(self, t, name, starts, durs):
        """A column sweep at time ``t``: ``(steps finished, current step, its
        progress)``; the current step is -1 before the sweep, ``T`` after."""
        if t < self.Tm[name][0]:
            return 0, -1, 0.0
        if t >= self.Tm[name][1]:
            return self.T, self.T, 0.0
        k = int(np.searchsorted(starts, t, side="right") - 1)
        return k, k, (t - starts[k]) / durs[k]

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        rnn, lstm = self.rnn, self.lstm
        ph = 0 if reset > 0 else self.step_of(t)
        self.subtitle = [
            "the task: remember the first value (%.2f), sit through %d steps of noise, then "
            "say it" % (rnn.cue, self.T - 1),
            "forward: both read the same sequence; the RNN answers %.2f, the LSTM %.2f "
            "(target %.2f)" % (rnn.y, lstm.y, rnn.cue),
            "memory: run it again with the cue flipped; the difference between the runs is "
            "what the state still knows",
            "backward: the gradient shrinks every tanh step; reaching the cue: %s in the RNN, "
            "%s in the LSTM's cell state" % (sci(self.at_cue["rnn"]), sci(self.at_cue["lstm"])),
            "training: %d steps each; the LSTM learns the task, the RNN stays at chance (%.2f), "
            "and so does an LSTM whose forget gate starts half shut" % (len(rnn.losses), CHANCE),
        ][ph]
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        task = self.progress("task", t)
        done_f, nf, uf = self._sweep(t, "fwd", self.f_starts, self.f_durs)
        done_b, nb, ub = self._sweep(t, "bwd", self.b_starts, self.b_durs)
        mem = smoothstep(0.0, 0.2, self.progress("memory", t))
        train = self.progress("train", t)
        notes = (1.0 - smoothstep(0.0, 0.12, train)) * back    # read-outs give way to the chart
        self._input(sc, t, task, nf, uf, mem, back)
        self._bands(sc, t, done_f, nf, uf, done_b, nb, ub, mem, back)
        if 0 <= nf < self.T or 0 <= nb < self.T:
            self._column_frame(sc, nf if nf < self.T else self.T - 1 - nb)
        self._answers(sc, t, done_f, mem, notes)
        self._gradient_note(sc, done_b, notes)
        self._chart(sc, train, back)
        self.step_indicator(sc, ph, self.PHASES)

    # -- the input row -------------------------------------------------------------------
    def _input(self, sc, t, task, nf, uf, mem, back):
        """The sequence: the cue (flipped in the memory phase), then noise."""
        s, label = self.rows["x"]
        x = np.array(self.rnn.x)
        x[0] = x[0] * (1 - 2 * mem)                        # the flipped cue takes over
        cols = div_colors(x, 1.0)
        s.fill(cols)
        col = np.arange(self.T)
        s.show(col < task * (self.T + 1) * 1.05 if t < self.Tm["fwd"][0] else np.ones(self.T, bool))
        s.cage = 1.0
        s.fade(back)
        s.block.white[:, :, 0] = ((col == nf) & (uf < 1)) * (1 - smoothstep(0.7, 1.0, uf)) * 0.6
        s.block.white[0, 0, 0] = max(s.block.white[0, 0, 0], 0.5 * soft_band(task, 0.05, 0.4, 0.1))
        self.caption_left(sc, s, label, back)

    # -- the bands ---------------------------------------------------------------------
    def _bands(self, sc, t, done_f, nf, uf, done_b, nb, ub, mem, back):
        """The hidden-state bands: the run, then what remains of the cue, then
        the gradient reaching each step."""
        T, n = self.T, self.n
        col = np.tile(np.arange(T), n)
        lo, hi = GRAD_RANGE
        for key in ("rh", "lc", "lh"):
            s, label = self.rows[key]
            v = self.vals[key]
            c = div_colors(v.ravel(), amax(v))
            if mem > 0:                                   # the difference between the runs
                dv = v - self.vals2[key]
                c = c + (div_colors(dv.ravel(), amax(dv)) - c) * mem
            if t >= self.Tm["bwd"][0]:                     # the gradient, column by column
                g = np.abs(self.grads[key]).ravel()
                m = np.clip((np.log10(g + 1e-12) - lo) / (hi - lo), 0, 1)
                gc = lit_colors(m, 1.0, GRAD_LO, GRAD_HI)
                got = col >= T - done_b
                if 0 <= nb < T and ub > 0.35:
                    got |= col == T - 1 - nb
                c[got] = gc[got]
            shown = col < done_f
            if 0 <= nf < T and uf >= 0.5:
                shown |= col == nf
            if t >= self.Tm["fwd"][1]:
                shown[:] = True
            s.fill(c)
            s.show(shown)
            s.cage = 1.0
            s.fade(back)
            s.block.white[:, :, 0] = (((col == nf) & (uf < 1)) * (1 - smoothstep(0.7, 1.0, uf))
                                      * 0.6).reshape(n, T)
            text = label
            if mem > 0.5 and t < self.Tm["bwd"][0]:          # the difference between the runs
                text = label.replace("hidden h", "h, difference").replace("cell state c",
                                                                           "c, difference")
            elif t >= self.Tm["bwd"][0]:
                text = label.replace("hidden h", "gradient at h").replace("cell state c",
                                                                           "gradient at c")
            self.caption_left(sc, s, text, back)
        # what flows: the input into both networks, the state along each
        xs, _ = self.rows["x"]
        if 0 <= nf < T and t < self.Tm["fwd"][1]:
            xin = xs.cell(nf)
            for key in ("rh", "lc"):
                s, _ = self.rows[key]
                particle(sc, xin, s.cell(n // 2 * T + nf), remap(uf, 0.0, 0.3), C.INPUT, 0.6,
                         lift=0.0)
                if nf > 0:
                    vmax = amax(self.vals[key])
                    for r in range(n):
                        colr = div_colors(self.vals[key][r, nf - 1:nf], vmax)[0]
                        particle(sc, s.cell(r * T + nf - 1), s.cell(r * T + nf),
                                 remap(uf, 0.05, 0.45), colr, 0.5, lift=0.4)
            sc.probe(("fwd",), (nf * 20 + uf * 20, 0, 0), tol=10.0)
        if 0 <= nb < T and self.Tm["bwd"][0] <= t < self.Tm["bwd"][1]:
            s_ = T - 1 - nb
            for key in ("rh", "lc"):
                s, _ = self.rows[key]
                g = np.abs(self.grads[key])
                m = np.clip((np.log10(g + 1e-12) - lo) / (hi - lo), 0, 1)
                if s_ < T - 1:
                    for r in range(n):
                        gcol = lit_colors(m[r, s_ + 1:s_ + 2], 1.0, GRAD_LO, GRAD_HI)[0]
                        particle(sc, s.cell(r * T + s_ + 1), s.cell(r * T + s_),
                                 remap(ub, 0.05, 0.45), gcol, 0.55, lift=-0.4)
            sc.probe(("bwd",), (nb * 20 + ub * 20, 0, 0), tol=10.0)

    def _column_frame(self, sc, col):
        """A white frame around time step ``col`` through every row: the step at work."""
        ss = [s for s, _l in self.rows.values()]
        lo = np.min([s.block.box((0, s.h), (col, col + 1), (0, 1))[0] for s in ss], 0)
        hi = np.max([s.block.box((0, s.h), (col, col + 1), (0, 1))[1] for s in ss], 0)
        sc.wire_box(lo, hi, inflate=0.18, color=FRAME, width=1.6, opacity=0.95, bias=0.1)

    # -- read-outs ----------------------------------------------------------------------
    def _answer_voxels(self, sc, key, y, target, op):
        """Right of band ``key``'s last column: the answer, and the target framed."""
        s, _ = self.rows[key]
        p = s.cell(self.n // 2 * self.T + self.T - 1)
        a = p + np.array([2.4, 0.0, 0.0])
        b = p + np.array([4.2, 0.0, 0.0])
        sc.voxel(a, div_colors([y], 1.0)[0], op, glow=0.3, shadow=False, size=0.95)
        sc.voxel(b, div_colors([target], 1.0)[0], 0.45 * op, shadow=False, size=0.95)
        sc.wire_box(b - 0.6, b + 0.6, color=FRAME, width=1.1, opacity=0.8 * op, bias=0.1)

    def _answers(self, sc, t, done_f, mem, op):
        """Each network's answer against the target, and the flipped run's."""
        show = smoothstep(self.T - 0.5, self.T, done_f + (0.5 if done_f == self.T - 1 else 0.0))
        show = 1.0 if t >= self.Tm["fwd"][1] else show
        op *= show * self._cap
        if op <= 0.01:
            return
        cue = self.rnn.cue
        for key, name, tr, fy in (("rh", "RNN", self.rnn, 0.345), ("lh", "LSTM", self.lstm, 0.72)):
            y = tr.y + (tr.y2 - tr.y) * mem
            target = cue * (1 - 2 * mem)
            self._answer_voxels(sc, key, y, target, op)
            sc.text(NOTE_X, fy, "%s answers %.2f" % (name, y), 0.03, "regular",
                    layout.STRONG_INK, op, "lm")
            sc.text(NOTE_X, fy + 0.04, "target %.2f" % target, 0.026, "regular",
                    layout.CAPTION_INK, op, "lm")
            if mem > 0:
                sc.text(NOTE_X, fy + 0.075, "with the cue %.2f it said %.2f" % (cue, tr.y), 0.024,
                        "regular", layout.NOTE_INK, op * mem, "lm")

    def _gradient_note(self, sc, done_b, op):
        """How much of the final error's gradient reaches the cue step."""
        op *= smoothstep(self.T - 1, self.T, done_b) * self._cap
        if op <= 0.01:
            return
        sc.text(NOTE_X, 0.86, "gradient reaching the cue", 0.026, "regular", layout.CAPTION_INK,
                op, "lm")
        sc.text(NOTE_X, 0.90, "RNN %s      LSTM %s (cell state)"
                % (sci(self.at_cue["rnn"]), sci(self.at_cue["lstm"])), 0.03, "regular",
                layout.STRONG_INK, op, "lm")

    # -- the training curves ----------------------------------------------------------------
    def _chart(self, sc, train, back):
        """Both networks' training loss (log-log), drawn as training runs."""
        op = smoothstep(0.14, 0.3, train) * back * self._cap     # once the read-outs have gone
        if op <= 0.01:
            return
        (x0, x1), (y0, y1) = CHART
        cam, txt = self.cam, self.fr
        N = len(self.curves["rnn"])
        first = SMOOTH                                     # the first averaged step
        k = max(first, int(round(N ** smoothstep(0.0, 0.9, train))))
        lo, hi = LOSS_RANGE

        def fx(step):
            return x0 + (x1 - x0) * np.log10(step) / np.log10(N)

        def pt(step, loss):
            fy = y1 - (y1 - y0) * (np.clip(np.log10(loss), lo, hi) - lo) / (hi - lo)
            return cam.frac(fx(step), fy)

        axis = (0.5, 0.52, 0.54)
        sc.line(cam.frac(x0, y1), cam.frac(x1, y1), width=1.0, color=axis, opacity=0.7 * op,
                bias=0.0)
        sc.line(cam.frac(x0, y1), cam.frac(x0, y0), width=1.0, color=axis, opacity=0.7 * op,
                bias=0.0)
        sc.text(*txt(x0, y0 - 0.014), "loss", 0.026, "regular", layout.CAPTION_INK, op, "lb")
        for step in (100, 1000):
            sc.text(*txt(fx(step), y1 + 0.01), str(step), 0.024, "regular", layout.NOTE_INK, op,
                    "mt")
        sc.text(*txt((x0 + x1) / 2, y1 + 0.046), "training steps (loss averaged over %d)" % SMOOTH,
                0.026, "regular", layout.CAPTION_INK, op, "mt")
        # chance: always answering 0
        a, b = pt(first, CHANCE), pt(N, CHANCE)
        for i in range(12):
            u0, u1 = i / 12, (i + 0.5) / 12
            sc.line(a + (b - a) * u0, a + (b - a) * u1, width=0.9, color=axis, opacity=0.6 * op,
                    bias=0.0)
        sc.text(*txt(x1 - 0.006, y1 - (y1 - y0) * (np.log10(CHANCE) - lo) / (hi - lo) - 0.008),
                "chance", 0.024, "regular", layout.NOTE_INK, op, "rb")
        steps = np.unique(np.round(np.logspace(np.log10(first), np.log10(N), 160)).astype(int))
        for key, ink in (("leaky", LEAKY_INK), ("rnn", RNN_INK), ("lstm", LSTM_INK)):
            curve, prev = self.curves[key], None
            for step in [*steps[steps < k], k]:
                p = pt(step, curve[step - 1])
                if prev is not None:
                    sc.line(prev, p, width=1.3, color=ink, opacity=0.9 * op, bias=0.0)
                prev = p
            sc.voxel(prev, ink, op, glow=0.3, shadow=False, reflect=False, size=0.6)
        # legend, lower left: the failures run along the top, the LSTM falls at the right
        legend = (("RNN", RNN_INK), ("LSTM, gate from 0.73", LEAKY_INK), ("LSTM", LSTM_INK))
        for n, (name, ink) in enumerate(legend):
            sc.text(*txt(x0 + 0.012, y0 + (y1 - y0) * (0.52 + 0.12 * n)), name, 0.026, "regular",
                    ink, op, "lm")
        sc.probe(("train",), (np.log10(k) * 40, 0, 0), tol=6.0)

"""Mamba: selection.  The block of :mod:`voxanim.mamba` on the memory task.

The task is :mod:`voxanim.gated_anim`'s: say the first value of a sequence
after 49 steps of noise.  The trained block (16 channels, 8 states each)
reads the same example sequence, drawn unrolled through time:

    input x_t         the cue, then noise
    step Δ_t          16 x 50: the input-dependent step, per channel
    kept exp(Δ_t A)   1 x 50: the fraction of the state that survives the step
    state             16 x 50: each channel's state read out, h_t . C_t
    output            1 x 50

Four phases:

1. the task,
2. selection -- the forward sweep: ``Δ_t`` opens at the cue (the block can
   tell the first step: the causal convolution's window is zero-padded
   there) and shuts on the noise, so the state written at the cue is kept
   across the gap and the answer is the cue,
3. scan -- the same states by the parallel (prefix) scan: ``log₂ T`` levels
   of combine steps, each drawn as arcs from ``t − s`` to ``t`` over the
   band while the band's cells turn into the partial results; after the
   last level they equal the sequential recurrence's to rounding,
4. training -- the loss against the RNN's and the LSTM's from
   :mod:`voxanim.memory`: the gated cell and the selective state learn the
   task, the plain recurrence stays at chance.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import color as C
from . import layout
from .conv import Tempo
from .easing import ease_in_out, ease_out, remap, smoothstep, soft_band
from .gated_anim import LSTM_INK, RNN_INK
from .mamba import scan_levels
from .memory import CHANCE
from .netanim import NetAnimation, amax, div_colors, lit_colors, particle
from .text import MINUS, num

MAMBA_INK = (0.56, 0.30, 0.72)                # the block's curve (violet)
LOSS_RANGE = (-4.3, -0.3)                     # log10 loss at the bottom and top of the chart
SMOOTH = 20                                   # batches the drawn loss is averaged over
CHART = ((0.665, 0.93), (0.32, 0.78))         # the chart's box ((x0, x1), (top, bottom)), layout
NOTE_X = 0.64                                 # frame x of the read-outs right of the bands
STACK_AT = (0.385, 0.57)                      # frame position of the stack's centre
PITCH = (1.0, 1.0)
PPU = 8.5
LEVEL_LIFT = 1.3                              # world units between the scan's arc levels
LEVEL_SECONDS = 1.8                           # one combine level of the scan
FRAME = (0.96, 0.96, 0.96)
KEPT_LO, KEPT_HI = C.rgb("#2b2f33"), C.rgb("#c9fff6")   # kept fraction: dark .. lit
KEPT_RANGE = (0.5, 1.0)                       # kept fraction at the dark and the lit end
DT_FLOOR = 1e-3                               # Δ drawn on a log scale from here to its maximum
COLOR_FLOOR = 0.03                            # the dark end of the ramps, never the OFF colour


def readout(h, Ct):
    """Each channel's state read out, ``h_t . C_t``: ``(T, d)`` from ``h`` ``(T, d, n)``
    and ``C_t`` ``(T, n)``."""
    return np.einsum("tdk,tk->td", h, Ct)


class MambaSelection(NetAnimation):
    """The memory example through the trained block; ``rnn`` and ``lstm`` are
    :class:`voxanim.memory.MemoryTrace` s, for the training chart."""

    PHASES = ("the task", "selection", "scan", "training")

    def __init__(self, tr, rnn, lstm):
        super().__init__("Mamba: selection", "", PPU)
        self.tr, self.rnn, self.lstm = tr, rnn, lstm
        f = tr.f
        self.T, self.d, self.n = f["dt"].shape[0], f["dt"].shape[1], f["h"].shape[2]
        self.hold = 2.5
        self.rows = {}
        rows_spec = (("x", "input: cue, then noise", 1, 0.0),
                     ("dt", "step Δ, per channel", self.d, 1.5),
                     ("kept", "state kept, exp(ΔA)", 1, 1.2),
                     ("state", "state read out h·C", self.d, 1.6),
                     ("y", "output", 1, 1.2))
        centre = self.cam.frac(*STACK_AT)
        height = sum(r + g for _k, _l, r, g in rows_spec)
        z = centre[2] + height / 2
        for key, label, rows, gap in rows_spec:
            s = self.sheet((rows, self.T), STACK_AT, 1.0, key, pitch=PITCH)
            z -= gap + rows / 2
            c = centre.copy()
            c[2] = z
            s.block.place(c)
            z -= rows / 2
            self.rows[key] = (s, label)
        # the values on the bands, (rows, T)
        self.dt = f["dt"].T
        self.kept = f["Abar"].mean((1, 2))[None]
        self.state = readout(f["h"][1:], f["Ct"]).T
        self.out = f["out"][None]
        self.levels = scan_levels(self.T)
        self.trail = [readout(tr.trail[k], f["Ct"]).T for k in range(len(self.levels))]
        self.scan_diff = float(np.abs(self.trail[-1] - self.state).max())
        self.vmax_state = amax(self.state[:, 1:])
        self.dt_cue, self.dt_noise = float(self.dt[:, 0].mean()), float(self.dt[:, 1:].mean())
        self.kept_cue, self.kept_noise = float(self.kept[0, 0]), float(self.kept[0, 1:].mean())
        kernel = np.ones(SMOOTH) / SMOOTH
        self.curves = {k: np.convolve(L, kernel, "valid")
                       for k, L in (("rnn", rnn.losses), ("lstm", lstm.losses),
                                    ("mamba", tr.losses))}

        # ---- timeline -----------------------------------------------------------------
        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("task", 3.2, 0.3)
        fw = Tempo(slow=0.7, fast=0.13, ramp_in=3.0, ramp_out=2.0).durations(self.T)
        self.f_starts, self.f_durs = t + np.concatenate([[0.0], np.cumsum(fw)[:-1]]), fw
        add("fwd", float(fw.sum()), 2.0)
        add("scan_in", 1.5, 0.0)
        add("scan", LEVEL_SECONDS * len(self.levels), 2.0)
        add("train", 8.5, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["fwd"][0], Tm["scan_in"][0], Tm["train"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _sweep(self, t):
        """The forward sweep: ``(steps finished, current step, its progress)``."""
        if t < self.Tm["fwd"][0]:
            return 0, -1, 0.0
        if t >= self.Tm["fwd"][1]:
            return self.T, self.T, 0.0
        k = int(np.searchsorted(self.f_starts, t, side="right") - 1)
        return k, k, (t - self.f_starts[k]) / self.f_durs[k]

    def _level(self, t):
        """The scan: ``(levels finished, current level, its progress)``; -1 before."""
        if t < self.Tm["scan"][0]:
            return 0, -1, 0.0
        if t >= self.Tm["scan"][1]:
            return len(self.levels), len(self.levels), 0.0
        u = (t - self.Tm["scan"][0]) / LEVEL_SECONDS
        return int(u), int(u), u - int(u)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        tr = self.tr
        ph = 0 if reset > 0 else self.step_of(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        task = self.progress("task", t)
        done, nf, uf = self._sweep(t)
        scan_in = self.progress("scan_in", t)
        done_l, nl, ul = self._level(t)
        train = self.progress("train", t)
        notes = (1.0 - smoothstep(0.0, 0.12, train)) * back
        self.subtitle = [
            "the task of gated-memory: remember the first value (%.2f), sit through %d steps "
            "of noise, then say it" % (tr.cue, self.T - 1),
            "selection: Δₜ = softplus(linear(xₜ)) is %.2f at the cue, %.3f on the noise: what is "
            "written at the cue is kept" % (self.dt_cue, self.dt_noise),
            "scan: the same recurrence as a prefix scan, %d levels for %d steps; the states agree "
            "to %s" % (len(self.levels), self.T, "%.0e" % self.scan_diff),
            "training: the LSTM and Mamba learn the task (%d and %d steps), the RNN stays at "
            "chance (%.2f)" % (len(self.lstm.losses), len(tr.losses), CHANCE),
        ][ph]
        in_scan = t >= self.Tm["scan_in"][0] and t < self.Tm["train"][0] - 0.3
        self._input(sc, t, task, nf, uf, back)
        self._bands(sc, done, nf, uf, done_l, nl, ul, in_scan, scan_in, back)
        if 0 <= nf < self.T:
            self._column_frame(sc, nf)
        if in_scan:
            self._scan_arcs(sc, done_l, nl, ul, scan_in, back)
        self._notes(sc, t, done, done_l, nl, notes, in_scan)
        self._chart(sc, train, back)
        self.step_indicator(sc, ph, self.PHASES)

    # -- the input row -------------------------------------------------------------------
    def _input(self, sc, t, task, nf, uf, back):
        """The sequence: the cue, then noise."""
        s, label = self.rows["x"]
        s.fill(div_colors(self.tr.x, 1.0))
        col = np.arange(self.T)
        shown = np.ones(self.T, bool)
        if t < self.Tm["fwd"][0]:
            shown = col < task * (self.T + 1) * 1.05
        s.show(shown)
        s.cage = 1.0
        s.fade(back)
        s.block.white[:, :, 0] = ((col == nf) & (uf < 1)) * (1 - smoothstep(0.7, 1.0, uf)) * 0.6
        s.block.white[0, 0, 0] = max(s.block.white[0, 0, 0], 0.5 * soft_band(task, 0.05, 0.4, 0.1))
        self.caption_left(sc, s, label, back)

    # -- the bands ---------------------------------------------------------------------
    def _band_colors(self, key, values):
        """Colours for band ``key``'s ``(rows, T)`` values."""
        if key == "dt":                        # Δ spans three decades: a log scale
            lo, hi = np.log10(DT_FLOOR), np.log10(amax(self.dt))
            m = (np.log10(np.maximum(values.ravel(), DT_FLOOR)) - lo) / (hi - lo)
            return lit_colors(np.maximum(m, COLOR_FLOOR), 1.0)
        if key == "kept":
            lo, hi = KEPT_RANGE
            m = (values.ravel() - lo) / (hi - lo)
            return lit_colors(np.maximum(m, COLOR_FLOOR), 1.0, KEPT_LO, KEPT_HI)
        if key == "state":                     # the cue step's transients are off the scale
            return div_colors(values.ravel(), self.vmax_state)
        return div_colors(values.ravel(), 1.0)

    def _bands(self, sc, done, nf, uf, done_l, nl, ul, in_scan, scan_in, back):
        """The step, the kept fraction, the state and the output, filled by the
        sweep; in the scan the state band shows the partial results level by level."""
        T, d = self.T, self.d
        col = np.arange(T)
        shown = col < done
        if 0 <= nf < T and uf >= 0.5:
            shown |= col == nf
        if done >= T:
            shown[:] = True
        values = {"dt": self.dt, "kept": self.kept, "state": self.state, "y": self.out}
        if in_scan:
            # the scan starts from the inputs alone (level 0) and refines level by level
            k = min(done_l, len(self.levels)) - 1
            partial = self.trail[k] if k >= 0 else self.trail[0] * 0 + self._level0()
            values = dict(values, state=partial)
        for key in ("dt", "kept", "state", "y"):
            s, label = self.rows[key]
            v = values[key]
            c = self._band_colors(key, v)
            s.fill(c)
            s.show(np.tile(shown, v.shape[0]))
            s.cage = 1.0
            fade = back
            if in_scan and key != "state":
                fade *= 1.0 - smoothstep(0.0, 0.5, scan_in)      # the others make room for the arcs
            s.fade(fade)
            s.block.white[:, :, 0] = (((col == nf) & (uf < 1)) * (1 - smoothstep(0.7, 1.0, uf))
                                      * 0.6)[None].repeat(v.shape[0], 0)
            if in_scan and key == "state" and 0 <= nl < len(self.levels):
                # the cells being combined at this level flash as the arcs land
                s_off = self.levels[nl]
                hit = (col >= s_off) * smoothstep(0.55, 0.75, ul) * (1 - smoothstep(0.85, 1.0, ul))
                s.block.white[:, :, 0] = (0.5 * hit)[None].repeat(d, 0)
            self.caption_left(sc, s, label, fade)
        # what flows during the sweep
        xs, _ = self.rows["x"]
        if 0 <= nf < T and done < T:
            dts, _ = self.rows["dt"]
            st, _ = self.rows["state"]
            particle(sc, xs.cell(nf), dts.cell(d // 2 * T + nf), remap(uf, 0.0, 0.3), C.INPUT,
                     0.6, lift=0.0)
            if nf > 0:
                for r in range(d):
                    colr = div_colors(self.state[r, nf - 1:nf], self.vmax_state)[0]
                    particle(sc, st.cell(r * T + nf - 1), st.cell(r * T + nf),
                             remap(uf, 0.05, 0.45), colr, 0.5, lift=0.4)
            sc.probe(("fwd",), (nf * 20 + uf * 20, 0, 0), tol=10.0)

    def _level0(self):
        """The scan's starting values: each step's own input term alone, read out."""
        f = self.tr.f
        inp = f["dt"][..., None] * f["Bt"][:, None, :] * f["v"][..., None]
        return readout(inp, f["Ct"]).T

    def _column_frame(self, sc, col):
        """A white frame around time step ``col`` through every row: the step at work."""
        ss = [s for s, _l in self.rows.values()]
        lo = np.min([s.block.box((0, s.h), (col, col + 1), (0, 1))[0] for s in ss], 0)
        hi = np.max([s.block.box((0, s.h), (col, col + 1), (0, 1))[1] for s in ss], 0)
        sc.wire_box(lo, hi, inflate=0.18, color=FRAME, width=1.6, opacity=0.95, bias=0.1)

    # -- the scan's tree -----------------------------------------------------------------
    def _scan_arcs(self, sc, done_l, nl, ul, scan_in, back):
        """Level ``s`` of the scan as arcs from column ``t − s`` to ``t`` over the state
        band, one tier per level; the current level's arcs grow in, finished ones stay faint."""
        s, _ = self.rows["state"]
        top = s.block.box((0, 1), (0, 1), (0, 1))[1][2]       # z of the band's top face
        base = s.cell(0)
        step = s.cell(1) - base
        op0 = back * smoothstep(0.3, 1.0, scan_in)
        for k, off in enumerate(self.levels):
            if k > nl:
                break
            if k == nl and nl < len(self.levels):
                grow = ease_out(remap(ul, 0.0, 0.6))
                op = op0
            else:
                grow, op = 1.0, op0 * 0.35
            z = top + LEVEL_LIFT * (k + 1)
            ink = C.INPUT if k == nl else (0.55, 0.57, 0.60)
            for t_ in range(off, self.T, max(1, off // 8)):   # fewer arcs at the higher levels
                a = base + step * (t_ - off)
                b = base + step * t_
                a[2] = b[2] = z
                self._arc(sc, a, b, grow, ink, op)
        if 0 <= nl < len(self.levels):
            sc.probe(("scan",), (ul * 60, 0, 0), tol=8.0)

    def _arc(self, sc, a, b, grow, ink, op, segments=6):
        """A low arc from ``a`` to ``b`` (both at the arc's base height), ``grow`` of it drawn."""
        if grow <= 0 or op <= 0.01:
            return
        lift = 0.35 * LEVEL_LIFT
        pts = []
        for i in range(segments + 1):
            u = i / segments * grow
            p = a + (b - a) * u
            p = p + np.array([0.0, 0.0, lift * 4 * u * (1 - u)])
            pts.append(p)
        for p, q in itertools.pairwise(pts):
            sc.line(p, q, width=1.1, color=ink, opacity=op, bias=0.3)

    # -- read-outs ----------------------------------------------------------------------
    def _notes(self, sc, t, done, done_l, nl, op, in_scan):
        """The step at the cue and on the noise, the answer, the scan's agreement."""
        tr = self.tr
        op *= self._cap
        if op <= 0.01:
            return
        ink, soft = layout.STRONG_INK, layout.CAPTION_INK
        show = 1.0 if t >= self.Tm["fwd"][1] else smoothstep(1, 3, done)
        if show > 0.01 and not in_scan:
            sc.text(NOTE_X, 0.33, "step Δ (mean over channels)", 0.026, "regular", soft, op * show,
                    "lm")
            sc.text(NOTE_X, 0.37, "at the cue %.2f,  on the noise %.3f" % (self.dt_cue,
                                                                             self.dt_noise),
                    0.03, "regular", ink, op * show, "lm")
            sc.text(NOTE_X, 0.43, "state kept per step, exp(ΔA)", 0.026, "regular", soft, op * show,
                    "lm")
            sc.text(NOTE_X, 0.47, "at the cue %.2f,  on the noise %.2f" % (self.kept_cue,
                                                                             self.kept_noise),
                    0.03, "regular", ink, op * show, "lm")
        ans = 1.0 if t >= self.Tm["fwd"][1] else smoothstep(self.T - 1, self.T, done)
        if ans > 0.01 and not in_scan:
            sc.text(NOTE_X, 0.73, "Mamba answers %s" % num(tr.y), 0.03, "regular", ink, op * ans,
                    "lm")
            sc.text(NOTE_X, 0.77, "target %s" % num(tr.cue), 0.026, "regular", soft, op * ans,
                    "lm")
        if in_scan:
            L = len(self.levels)
            sc.text(NOTE_X, 0.33, "parallel scan", 0.026, "regular", soft, op, "lm")
            sc.text(NOTE_X, 0.37, "level %d of %d" % (min(done_l + 1, L), L), 0.03, "regular", ink,
                    op, "lm")
            sc.text(NOTE_X, 0.43, "(a, b) then (a', b') = (a a', a' b + b')", 0.026, "regular",
                    soft, op, "lm")
            sc.text(NOTE_X, 0.47, "level s: step t combines with t %s %d" % (MINUS, self.levels[
                min(nl, L - 1)]) if nl >= 0 else "", 0.026, "regular", soft, op, "lm")
            sc.text(NOTE_X, 0.53, "%d sequential steps become %d levels" % (self.T, L), 0.026,
                    "regular", soft, op, "lm")
            if done_l >= L:
                sc.text(NOTE_X, 0.73, "same states as the recurrence", 0.026, "regular", soft, op,
                        "lm")
                sc.text(NOTE_X, 0.77, "largest difference %.0e" % self.scan_diff, 0.03, "regular",
                        ink, op, "lm")

    # -- the training curves ----------------------------------------------------------------
    def _chart(self, sc, train, back):
        """The three networks' training loss (log-log), drawn as training runs."""
        op = smoothstep(0.14, 0.3, train) * back * self._cap
        if op <= 0.01:
            return
        (x0, x1), (y0, y1) = CHART
        cam, txt = self.cam, self.fr
        N = max(len(c) for c in self.curves.values())
        first = SMOOTH
        k = max(first, int(round(N ** smoothstep(0.0, 0.9, train))))
        lo, hi = LOSS_RANGE

        def fx(step):
            return x0 + (x1 - x0) * np.log10(step) / np.log10(N)

        def pt(step, loss):
            fy = y1 - (y1 - y0) * (np.clip(np.log10(loss), lo, hi) - lo) / (hi - lo)
            return cam.frac(fx(step), fy)

        axis = (0.5, 0.52, 0.54)
        for a, b in ((cam.frac(x0, y1), cam.frac(x1, y1)), (cam.frac(x0, y1), cam.frac(x0, y0))):
            sc.line(a, b, width=1.0, color=axis, opacity=0.7 * op, bias=0.0)
        sc.text(*txt(x0, y0 - 0.014), "loss", 0.026, "regular", layout.CAPTION_INK, op, "lb")
        for step in (100, 1000):
            sc.text(*txt(fx(step), y1 + 0.01), str(step), 0.024, "regular", layout.NOTE_INK, op,
                    "mt")
        sc.text(*txt((x0 + x1) / 2, y1 + 0.046), "training steps (loss averaged over %d)" % SMOOTH,
                0.026, "regular", layout.CAPTION_INK, op, "mt")
        a, b = pt(first, CHANCE), pt(N, CHANCE)
        for i in range(12):
            u0, u1 = i / 12, (i + 0.5) / 12
            sc.line(a + (b - a) * u0, a + (b - a) * u1, width=0.9, color=axis, opacity=0.6 * op,
                    bias=0.0)
        sc.text(*txt(x1 - 0.006, y1 - (y1 - y0) * (np.log10(CHANCE) - lo) / (hi - lo) - 0.008),
                "chance", 0.024, "regular", layout.NOTE_INK, op, "rb")
        for key, ink in (("rnn", RNN_INK), ("lstm", LSTM_INK), ("mamba", MAMBA_INK)):
            curve = self.curves[key]
            end = min(k, len(curve))
            grid = np.logspace(np.log10(first), np.log10(end), 160)
            steps = np.unique(np.round(grid).astype(int))
            prev = None
            for step in steps:
                p = pt(step, curve[step - 1])
                if prev is not None:
                    sc.line(prev, p, width=1.3, color=ink, opacity=0.9 * op, bias=0.0)
                prev = p
            sc.voxel(prev, ink, op, glow=0.3, shadow=False, reflect=False, size=0.6)
        legend = (("RNN", RNN_INK), ("LSTM", LSTM_INK), ("Mamba", MAMBA_INK))
        for n, (name, ink) in enumerate(legend):
            sc.text(*txt(x0 + 0.012, y0 + (y1 - y0) * (0.40 + 0.12 * n)), name, 0.026, "regular",
                    ink, op, "lm")
        sc.probe(("train",), (np.log10(k) * 40, 0, 0), tol=6.0)

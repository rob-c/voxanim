"""A state space model: one recurrence, two views.

The diagonal SSM of :mod:`voxanim.ssm` (6 states, ``A_n = −(n + 1)``, step
``Δ = 0.1``) reads a sine wave that steps up half way (48 steps), drawn
unrolled through time like :mod:`voxanim.gated_anim`:

    kernel K          slides over the input in the convolution phase
    input x_t         the sine wave with a step
    state h           6 x 48
    output y          by the recurrence
    output y          by the convolution

Four phases:

1. the system -- the input, and the eigenvalues ``Ā_n = exp(Δ A_n)`` of the
   discretised state matrix: how much of each state survives a step,
2. recurrence -- ``h_t = Ā h_{t−1} + B̄ x_t`` step by step: the state band
   fills column by column, ``y_t = C . h_t`` under it,
3. convolution -- the kernel ``K_k = C . Ā^k B̄`` (it decays as the ``Ā_n`` do)
   slides over the input; ``y_t = Σ_k K_k x_{t−k}`` fills the second output
   row and matches the first to rounding (the difference is printed),
4. step Δ -- ``Δ`` falls to 0.02 and rises to 0.5: the kernel lengthens and
   shortens, the state keeps the past for longer or shorter, the output
   smooths or sharpens, all recomputed live.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import color as C
from . import layout, ssm
from .conv import Tempo
from .easing import ease_in_out, remap, smoothstep
from .netanim import NetAnimation, amax, div_colors, particle, screen_box
from .text import MINUS, TIMES, num, sub

# the rows of sheets, top to bottom: key, label, rows, gap above (world units)
ROWS = (("x", "input x", 1, 0.0), ("h", "state h", ssm.N_STATE, 1.6),
        ("y", "y by recurrence", 1, 1.6), ("yc", "y by convolution", 1, 1.8))
STACK_AT = (0.41, 0.60)                       # frame position of the stack's centre
PPU = 12.0                                    # pixels per voxel: 48 steps fill half the frame
KERNEL_LIFT = 3.2                             # the kernel row floats this far above the input
NOTE_X = 0.69                                 # frame x of the read-outs right of the bands
DT_PATH = ((0.0, ssm.DT), (0.35, 0.02), (0.75, 0.5), (1.0, ssm.DT))   # (progress, Δ)
FRAME = (0.96, 0.96, 0.96)
PITCH = (1.0, 1.0)


def sci(v):
    """``4×10⁻¹⁷`` style for a tiny magnitude."""
    if v == 0:
        return "0"
    e = int(np.floor(np.log10(v)))
    m = v / 10 ** e
    exp = ("⁻" if e < 0 else "") + str(abs(e)).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return "%.0f%s10%s" % (m, TIMES, exp)


class SSMRecurrence(NetAnimation):
    """The SSM run two ways on :func:`voxanim.ssm.example`, then with its step changed."""

    PHASES = ("the system", "recurrence", "convolution", "step Δ")

    def __init__(self, P=None, x=None):
        super().__init__("SSM: one recurrence, two views", "", PPU)
        self.P = ssm.init(np.random.default_rng(ssm.SEED)) if P is None else P
        self.x = ssm.example() if x is None else x
        self.T, self.n = len(self.x), len(self.P["A"])
        self.hold = 2.5
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
        f = ssm.recurrent(self.P, self.x[None])
        self.h0, self.y0 = f["h"][0, 1:].T, f["y"][0]          # (n, T), (T,)
        self.yc0 = ssm.convolve(self.P, self.x[None])[0]
        self.K0 = ssm.kernel(self.P, self.T)
        self.diff = float(np.abs(self.y0 - self.yc0).max())
        self.vmax_h = amax(self.h0)                               # one colour scale throughout
        self.mem0 = float(ssm.memory_steps(self.P)[0])

        # ---- timeline -----------------------------------------------------------------
        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("system", 4.0, 0.3)
        rec = Tempo(slow=0.6, fast=0.12, ramp_in=3.0, ramp_out=2.0).durations(self.T)
        self.r_starts, self.r_durs = t + np.concatenate([[0.0], np.cumsum(rec)[:-1]]), rec
        add("rec", float(rec.sum()), 1.2)
        add("kernel", 2.5, 0.2)
        cv = Tempo(slow=0.5, fast=0.12, ramp_in=3.0, ramp_out=2.0).durations(self.T)
        self.c_starts, self.c_durs = t + np.concatenate([[0.0], np.cumsum(cv)[:-1]]), cv
        add("conv", float(cv.sum()), 1.5)
        add("dt", 12.0, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["rec"][0], Tm["kernel"][0], Tm["dt"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _sweep(self, t, name, starts, durs):
        """A column sweep: ``(steps finished, current step, its progress)``;
        the current step is -1 before the sweep, ``T`` after."""
        if t < self.Tm[name][0]:
            return 0, -1, 0.0
        if t >= self.Tm[name][1]:
            return self.T, self.T, 0.0
        k = int(np.searchsorted(starts, t, side="right") - 1)
        return k, k, (t - starts[k]) / durs[k]

    def dt_at(self, u):
        """The step ``Δ`` at progress ``u`` of the last phase: a smooth walk along
        :data:`DT_PATH` in log space."""
        for (u0, d0), (u1, d1) in itertools.pairwise(DT_PATH):
            if u <= u1:
                a = ease_in_out(remap(u, u0, u1))
                return float(np.exp(np.log(d0) + (np.log(d1) - np.log(d0)) * a))
        return DT_PATH[-1][1]

    def state(self, dt):
        """``(P, h (n, T), y (T,), yc (T,), K (T,))`` of the model with step ``dt``."""
        P = ssm.with_dt(self.P, dt)
        f = ssm.recurrent(P, self.x[None])
        yc = ssm.convolve(P, self.x[None])[0]
        return P, f["h"][0, 1:].T, f["y"][0], yc, ssm.kernel(P, self.T)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        system = self.progress("system", t)
        done_r, nr, ur = self._sweep(t, "rec", self.r_starts, self.r_durs)
        kern = self.progress("kernel", t)
        done_c, nc, uc = self._sweep(t, "conv", self.c_starts, self.c_durs)
        udt = self.progress("dt", t)
        dt = self.dt_at(udt) if ph == 3 else ssm.DT
        P, h, y, yc, K = self.state(dt)
        Abar, _ = ssm.discretize(P)
        mem = ssm.memory_steps(P)
        self.subtitle = [
            "a linear system with a diagonal A, read every Δ = %.1f: Ā = exp(ΔA) is what "
            "survives a step" % ssm.DT,
            "recurrence: hₜ = Ā hₜ₋₁ + B̄ xₜ, yₜ = C·hₜ: one state update per step, "
            "the state carries the past",
            "convolution: yₜ = Σ Kₖ xₜ₋ₖ with Kₖ = C·Āᵏ B̄, the recurrence unrolled; the two "
            "outputs differ by %s" % sci(self.diff),
            "step Δ = %.2f: Ā = exp(ΔA) is %.2f .. %.2f, the slowest state remembers %.0f steps "
            "(%.0f at Δ = %.1f)" % (dt, Abar[0], Abar[-1], mem[0], self.mem0, ssm.DT),
        ][ph]
        self._input(sc, t, system, nr, ur, nc, uc, K, back)
        self._state(sc, h, y, yc, done_r, nr, ur, done_c, nc, uc, ph, back)
        if 0 <= nr < self.T:
            self._column_frame(sc, nr, ("x", "h", "y"))
        if 0 <= nc < self.T:
            self._column_frame(sc, nc, ("x", "yc"))
        if t >= self.Tm["kernel"][0]:
            self._kernel(sc, K, kern, nc, uc, done_c, back)
        self._eigen_note(sc, Abar, mem, dt, back, ph)
        self.step_indicator(sc, ph, self.PHASES)

    # -- the input row -------------------------------------------------------------------
    def _input(self, sc, t, system, nr, ur, nc, uc, K, back):
        """The sequence; the step at work is highlighted, the taps under the kernel lit."""
        s, label = self.rows["x"]
        s.fill(div_colors(self.x, 1.0))
        col = np.arange(self.T)
        shown = np.ones(self.T, bool)
        if t < self.Tm["rec"][0]:
            shown = col < system * (self.T + 1) * 1.05
        s.show(shown)
        s.cage = 1.0
        s.fade(back)
        white = ((col == nr) & (ur < 1)) * (1 - smoothstep(0.7, 1.0, ur)) * 0.6
        if 0 <= nc < self.T:                               # the taps the kernel weights
            k = nc - col
            w = np.where(k >= 0, np.abs(K[np.clip(k, 0, self.T - 1)]) / amax(K), 0.0)
            white = np.maximum(white, 0.55 * w * (1 - smoothstep(0.7, 1.0, uc)))
        s.block.white[:, :, 0] = white
        self.caption_left(sc, s, label, back)

    # -- the state band and the outputs ---------------------------------------------------
    def _state(self, sc, h, y, yc, done_r, nr, ur, done_c, nc, uc, ph, back):
        """The state and the two output rows, filled by their sweeps."""
        T = self.T
        col = np.arange(T)
        # the state band, with the recurrence's particles
        s, label = self.rows["h"]
        s.fill(div_colors(h.ravel(), self.vmax_h))
        shown = col < done_r
        if 0 <= nr < T and ur >= 0.5:
            shown |= col == nr
        if done_r >= T:
            shown[:] = True
        s.show(np.tile(shown, self.n))
        s.cage = 1.0
        s.fade(back)
        s.block.white[:, :, 0] = (((col == nr) & (ur < 1)) * (1 - smoothstep(0.7, 1.0, ur))
                                  * 0.6)[None].repeat(self.n, 0)
        self.caption_left(sc, s, label, back)
        xs, _ = self.rows["x"]
        if 0 <= nr < T and done_r < T:
            particle(sc, xs.cell(nr), s.cell(self.n // 2 * T + nr), remap(ur, 0.0, 0.3), C.INPUT,
                     0.6, lift=0.0)
            if nr > 0:
                for r in range(self.n):
                    colr = div_colors(h[r, nr - 1:nr], self.vmax_h)[0]
                    particle(sc, s.cell(r * T + nr - 1), s.cell(r * T + nr), remap(ur, 0.05, 0.45),
                             colr, 0.5, lift=0.4)
            sc.probe(("rec",), (nr * 20 + ur * 20, 0, 0), tol=10.0)
        # the recurrence's output
        vmax_y = amax(self.y0)
        sy, label = self.rows["y"]
        sy.fill(div_colors(y, vmax_y))
        sy.show(shown)
        sy.cage = 1.0
        sy.fade(back)
        self.caption_left(sc, sy, label, back)
        if 0 <= nr < T and done_r < T and ur >= 0.5:
            particle(sc, s.cell(self.n // 2 * T + nr), sy.cell(nr), remap(ur, 0.5, 0.9),
                     div_colors(y[nr:nr + 1], vmax_y)[0], 0.5, lift=0.0)
        # the convolution's output
        sc_, label = self.rows["yc"]
        sc_.fill(div_colors(yc, vmax_y))
        shown_c = col < done_c
        if 0 <= nc < T and uc >= 0.5:
            shown_c |= col == nc
        if done_c >= T:
            shown_c[:] = True
        sc_.show(shown_c)
        sc_.cage = 1.0
        sc_.fade(back * (1.0 if done_c > 0 or (0 <= nc < T) else 0.0))
        sc_.visible = done_c > 0 or 0 <= nc < T
        if sc_.visible:
            self.caption_left(sc, sc_, label, back)
        if 0 <= nc < T and done_c < T:
            particle(sc, xs.cell(nc), sc_.cell(nc), remap(uc, 0.1, 0.6),
                     div_colors(yc[nc:nc + 1], vmax_y)[0], 0.55, lift=-0.5)
            sc.probe(("conv",), (nc * 20 + uc * 20, 0, 0), tol=10.0)

    def _column_frame(self, sc, col, keys):
        """A white frame around time step ``col`` through the rows ``keys``."""
        ss = [self.rows[k][0] for k in keys]
        lo = np.min([s.block.box((0, s.h), (col, col + 1), (0, 1))[0] for s in ss], 0)
        hi = np.max([s.block.box((0, s.h), (col, col + 1), (0, 1))[1] for s in ss], 0)
        sc.wire_box(lo, hi, inflate=0.18, color=FRAME, width=1.6, opacity=0.95, bias=0.1)

    # -- the kernel ----------------------------------------------------------------------
    def _kernel(self, sc, K, kern, nc, uc, done_c, back):
        """``K_k`` drawn as a row of voxels above the input, ``K_0`` over ``x_t``: it
        appears (right-aligned, over the whole input) and slides with the sweep."""
        xs, _ = self.rows["x"]
        T = self.T
        if 0 <= nc < T:
            t_at = nc + ease_in_out(min(1.0, uc / 0.5)) if nc < T - 1 else nc
        else:
            t_at = float(T - 1)
        appear = smoothstep(0.0, 0.6, kern)
        cols = div_colors(K, amax(K))
        base = xs.cell(0)
        step = xs.cell(1) - base
        for k in range(T):
            pos_col = t_at - k
            if pos_col < -0.5:
                continue
            p = base + step * pos_col + np.array([0.0, 0.0, KERNEL_LIFT])
            op = appear * back * min(1.0, (pos_col + 0.5) / 1.0)
            sc.voxel(p, cols[k], op, glow=0.12, shadow=False, size=1.0)
        # the label sits a fixed height above "input x", however short the kernel's box
        lo, hi = xs.bbox()
        lift = np.array([0.0, 0.0, KERNEL_LIFT])
        self._label_left(sc, lo + lift, hi + lift, "kernel K", appear * back)
        if 0 <= nc < T:
            sc.probe(("kernel",), (t_at * 20, 0, 0), tol=10.0)

    def _label_left(self, sc, lo, hi, text, opacity):
        """A caption left of the world box ``lo``..``hi`` (as :meth:`caption_left`
        does for a sheet)."""
        opacity *= self._cap
        if opacity <= 0.01:
            return
        x0, y0, _x1, y1 = screen_box(self.cam, lo, hi)
        sc.text((x0 - 10) / self.cam.width, (y0 + y1) / 2 / self.cam.height, text, 0.027,
                "regular", layout.CAPTION_INK, opacity, "rm")

    # -- read-outs ----------------------------------------------------------------------
    def _eigen_note(self, sc, Abar, mem, dt, back, ph):
        """The eigenvalues ``Ā_n`` as a column of voxels with their values, and the
        memory of the slowest state."""
        op = back * self._cap
        if op <= 0.01:
            return
        s, _ = self.rows["h"]
        p0 = s.cell(self.T - 1) + np.array([3.2, 0.0, 0.0])
        for r in range(self.n):
            p = p0 - np.array([0.0, 0.0, r * 1.0])
            sc.voxel(p, div_colors([Abar[r]], 1.0)[0], op * Abar[r] ** 0.5 + 0.15 * op, glow=0.1,
                     shadow=False, size=0.9)
        ink = layout.STRONG_INK
        sc.text(NOTE_X, 0.33, "Δ = %.2f" % dt, 0.03, "regular", ink, op, "lm")
        sc.text(NOTE_X, 0.375, "A = %s1 … %s%d" % (MINUS, MINUS, self.n), 0.026, "regular",
                layout.CAPTION_INK, op, "lm")
        sc.text(NOTE_X, 0.43, "Ā = exp(ΔA)", 0.026, "regular", layout.CAPTION_INK, op, "lm")
        for r in range(self.n):
            sc.text(NOTE_X + 0.01, 0.475 + r * 0.04, "Ā%s = %s" % (sub(r + 1), num(Abar[r])), 0.026,
                    "regular", ink if r == 0 else layout.NOTE_INK, op, "lm")
        sc.text(NOTE_X, 0.745, "memory of state 1: %.0f steps" % mem[0], 0.026, "regular",
                layout.CAPTION_INK, op, "lm")
        if ph >= 2:
            sc.text(NOTE_X, 0.79, "max |y%s %s y%s| = %s" % (sub("rec"), MINUS, sub("conv"),
                                                              sci(self.diff)),
                    0.026, "regular", layout.CAPTION_INK, op, "lm")

"""Diffusion: noise and back.  A small DDPM (:mod:`voxanim.diffusion`) on
MNIST, in four phases:

    row A:  four 28 x 28 sheets                 chart (right)
    row B:  four more, below (reverse phase)

1. **forward** -- four test digits dissolve step by step, 200 steps of a
   random walk in pixel space, until they are indistinguishable noise.  The
   schedule is drawn alongside: the signal level sqrt(abar_t) falling and
   the noise level sqrt(1 - abar_t) rising, with a marker at the current
   step; the subtitle reads both off.
2. **the loss** -- what the model learns: one digit at step t = 60, the
   noise eps that was added, the denoiser's prediction eps^, and their
   difference, whose mean square is the loss.
3. **reverse** -- four chains from fixed-seed pure noise walk back, 200
   steps: the top row is the sample x_t, the bottom row the denoiser's
   estimate of the clean image x0^ at that step, sharpening as t falls.
4. **training** -- the loss curve of the recorded training run (log-log)
   draws itself; the samples stay on screen.

Every pixel comes from :func:`voxanim.diffusion.trace`: the recorded forward
walk, the worked example, the reverse chains and the per-step training loss
of the cached run.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from .conv import Tempo
from .diffusion import BATCH, TEST_N, to_image
from .easing import ease_in_out, remap, smoothstep
from .netanim import NetAnimation, div_colors, gray_colors
from .tensor import Block

SLOT_X = (0.19, 0.325, 0.46, 0.595)      # layout x of the four columns
ROW_Y = (0.43, 0.745)                    # layout y of row A and row B
SHEET_WIDTH = 0.105                      # on-screen width of a sheet (frame fraction)
CHART = ((0.735, 0.935), (0.29, 0.58))   # the chart's box ((x0, x1), (top, bottom)), layout
SIGNAL_INK = tuple(float(v) for v in C.INPUT)       # sqrt(abar): the signal level
NOISE_INK = (0.75, 0.22, 0.22)                      # sqrt(1 - abar): the noise level
LOSS_INK = (0.75, 0.22, 0.22)
AXIS_INK = (0.5, 0.52, 0.54)
EPS_MAX = 3.0                            # |eps| drawn at full colour (standard normal: 3 sigma)
SMOOTH = 20                              # training steps the drawn loss is averaged over
FADE = 0.5                               # seconds a phase's sheets take to fade out at its end
LEARN_BEATS = dict(xt=(0.0, 0.14), eps=(0.2, 0.34), hat=(0.44, 0.58), err=(0.68, 0.82))
ABAR = "ᾱ"                               # the cumulative product of the alphas
SIGNAL, NOISE = "√ᾱ", "√(1 − ᾱ)"         # the two levels of the schedule
EPS, X_T = "ε", "xₜ"


class DiffusionAnimation(NetAnimation):
    """The four phases of the module docstring on a
    :class:`voxanim.diffusion.DiffusionTrace`."""

    PHASES = ("forward", "the loss", "reverse", "training")

    def __init__(self, tr):
        super().__init__("Diffusion: noise and back", "", 9.0)
        self.tr = tr
        self.T = tr.T
        self.hold = 2.5
        self.k = len(tr.digits)
        scale = self._fit_scale(SHEET_WIDTH)
        self.rowA = [self.sheet((28, 28), (x, ROW_Y[0]), scale, "a%d" % i)
                     for i, x in enumerate(SLOT_X)]
        self.rowB = [self.sheet((28, 28), (x, ROW_Y[1]), scale, "b%d" % i)
                     for i, x in enumerate(SLOT_X)]
        # the loss curve, averaged, on a log step axis
        kernel = np.ones(SMOOTH) / SMOOTH
        self.curve = np.convolve(tr.losses, kernel, "valid")
        self.log_loss = (np.floor(np.log10(self.curve.min()) * 2) / 2,
                         np.ceil(np.log10(self.curve.max()) * 2) / 2)

        # ---- timeline -----------------------------------------------------------------
        Tm, t = {}, 0.0

        def add(name, d, gap=0.0):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("intro", 1.5)
        fw = Tempo(slow=0.8, fast=0.035, ramp_in=4.0, ramp_out=3.0).durations(self.T)
        self.f_starts, self.f_durs = t + np.concatenate([[0.0], np.cumsum(fw)[:-1]]), fw
        add("forward", float(fw.sum()), 1.6)
        add("learn", 8.5, 0.6)
        rv = Tempo(slow=0.6, fast=0.045, ramp_in=4.0, ramp_out=5.0).durations(self.T)
        self.r_starts, self.r_durs = t + np.concatenate([[0.0], np.cumsum(rv)[:-1]]), rv
        add("reverse", float(rv.sum()), 1.4)
        add("train", 6.5)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["learn"][0], Tm["reverse"][0], Tm["train"][0]]

    # ------------------------------------------------------------------ setup
    def _fit_scale(self, width):
        """Voxel scale at which a 28 x 28 sheet is ``width`` of the frame wide."""
        probe = Block((28, 28, 1), scale=1.0)
        lo, hi = probe.origin, probe.origin + probe.extent
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        s = self.cam.project(corners)
        return width / ((s[:, 0].max() - s[:, 0].min()) / self.cam.width)

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _sweep(self, t, name, starts, durs):
        """Step sweep at time ``t``: ``(steps done, progress through the current
        one)``; ``done`` is 0 before the sweep and ``T`` after it."""
        if t < self.Tm[name][0]:
            return 0, 0.0
        if t >= self.Tm[name][1]:
            return self.T, 0.0
        k = int(np.searchsorted(starts, t, side="right") - 1)
        return k, (t - starts[k]) / durs[k]

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        for s in self.rowA + self.rowB:
            s.visible = False
        if t < self.Tm["learn"][0]:
            subtitle = self._forward(sc, t, back)
        elif t < self.Tm["reverse"][0]:
            subtitle = self._learn(sc, t, back)
        else:
            subtitle = self._reverse(sc, t, back)
        if t < self.Tm["forward"][0] or reset > 0:          # the loop returns to the opening
            subtitle = ("four test digits; the forward process adds a little Gaussian noise "
                        "at each of %d steps until nothing of them is left" % self.T)
        self.subtitle = subtitle
        self.step_indicator(sc, ph, self.PHASES)

    # -- phase 1: the forward process ----------------------------------------------------
    def _forward(self, sc, t, back):
        """The four digits noised step by step, with the schedule chart."""
        tr = self.tr
        intro = smoothstep(0.0, 1.0, self.progress("intro", t))
        # the noise the digits became fades out before the next phase
        gone = 1.0 - smoothstep(self.Tm["learn"][0] - FADE, self.Tm["learn"][0], t)
        step, _u = self._sweep(t, "forward", self.f_starts, self.f_durs)
        for i, s in enumerate(self.rowA):
            s.visible = True
            s.fill(gray_colors(to_image(tr.walk[step, i])))
            s.cage = intro * gone * back
            s.fade(intro * gone * back)
            self.caption(sc, s, "a %d" % tr.labels[i], intro * gone * back)
        self._schedule_chart(sc, step, intro * back)
        if 0 < step < self.T:
            sc.probe(("fwd",), (step * 2.0, 0.0, 0.0), tol=6.0)
        ab = tr.abar[step]
        if step == 0:
            return ("four test digits; the forward process adds a little Gaussian noise "
                    "at each of %d steps until nothing of them is left" % self.T)
        if step < self.T:
            return ("forward, step t = %d of %d: %s = %s·x₀ + %s·%s, with %s = %.2f and "
                    "%s = %.2f" % (step, self.T, X_T, SIGNAL, NOISE, EPS, SIGNAL, np.sqrt(ab),
                                   NOISE, np.sqrt(1 - ab)))
        return ("after %d steps %s = %.0e: the four digits are indistinguishable from pure "
                "noise" % (self.T, ABAR, ab))

    # -- phase 2: what the model learns -------------------------------------------------
    def _learn(self, sc, t, back):
        """One digit at step t, the noise added, the prediction, the error."""
        tr = self.tr
        u = self.progress("learn", t)
        beats = {k: smoothstep(a, b, u) for k, (a, b) in LEARN_BEATS.items()}
        gone = 1.0 - smoothstep(self.Tm["learn"][1] - FADE, self.Tm["learn"][1], t)
        err = tr.example_eps_hat - tr.example_eps
        panels = [("xt", gray_colors(to_image(tr.example_x)), "%s (t = %d)" % (X_T, tr.example_t)),
                  ("eps", div_colors(tr.example_eps, EPS_MAX), "the noise %s" % EPS),
                  ("hat", div_colors(tr.example_eps_hat, EPS_MAX), "predicted %s" % EPS),
                  ("err", div_colors(err, EPS_MAX),
                   "difference\nmean square %.3f" % tr.example_loss)]
        for i, (key, cols, text) in enumerate(panels):
            s = self.rowA[i]
            s.visible = True
            op = beats[key] * gone * back
            s.fill(cols)
            s.cage = op
            s.fade(op)
            self.caption(sc, s, text, op, max_width=0.14)
        self._schedule_chart(sc, tr.example_t if beats["xt"] > 0.5 else 0, back)
        ab = tr.abar[tr.example_t]
        if u < LEARN_BEATS["hat"][0]:
            return ("what the model learns: take a digit at step t = %d (%s = %.2f) and the "
                    "noise %s that was added to it" % (tr.example_t, SIGNAL, np.sqrt(ab), EPS))
        if u < LEARN_BEATS["err"][0]:
            return ("the denoiser reads %s and t and predicts the noise; the ideal prediction "
                    "is %s itself" % (X_T, EPS))
        return ("the loss is the mean squared difference, %.3f on this image; averaged over "
                "%d test images at random steps it is %.3f"
                % (tr.example_loss, TEST_N, tr.test_loss))

    # -- phases 3 and 4: the reverse process, then the training curve ---------------------
    def _reverse(self, sc, t, back):
        """Four chains from noise (row A) and the x0 estimates (row B); then the
        loss curve."""
        tr = self.tr
        done, _u = self._sweep(t, "reverse", self.r_starts, self.r_durs)
        step = self.T - done                                   # the current x_t
        fade_in = smoothstep(0.0, 0.6, remap(t, self.Tm["learn"][1], self.Tm["reverse"][0]))
        train = self.progress("train", t)
        for i in range(self.k):
            a, b = self.rowA[i], self.rowB[i]
            a.visible = b.visible = True
            a.fill(gray_colors(to_image(tr.chain[step, i])))
            b.fill(gray_colors(to_image(tr.chain_x0[step, i])))
            for s in (a, b):
                s.cage = fade_in * back
                s.fade(fade_in * back)
        self.caption_left(sc, self.rowA[0], "sample %s" % X_T, fade_in * back)
        self.caption_left(sc, self.rowB[0], "x₀ estimate", fade_in * back)
        if 0 < step < self.T:
            sc.probe(("rev",), (step * 2.0, 0.0, 0.0), tol=6.0)
        if train > 0:
            self._loss_chart(sc, train, back)
        else:
            self._schedule_chart(sc, step, fade_in * back)
        if train > 0:
            n = len(tr.losses)
            first, last = float(tr.losses[0]), float(self.curve[-1])
            if train < 1:
                return ("training: %d steps of Adam on batches of %d (%d epochs); the loss "
                        "falls from %.2f at the first step to %.2f"
                        % (n, BATCH, n // tr.steps_per_epoch, first, last))
            return ("a 784-512-512-784 denoiser, %d epochs: loss %.2f → %.2f (test %.2f); the "
                    "samples are digit-like, not every one a clean digit"
                    % (n // tr.steps_per_epoch, first, last, tr.test_loss))
        if step == self.T:
            return ("reverse: start from pure noise (four seeds); the denoiser's first estimate "
                    "of the clean image x₀ underneath is noise too")
        if step > 0:
            ab = tr.abar[step]
            return ("reverse, step t = %d: x₀ estimate = (%s − %s·predicted %s) / %s; then "
                    "xₜ₋₁ is drawn a little closer to it; %s = %.2f"
                    % (step, X_T, NOISE, EPS, SIGNAL, SIGNAL, np.sqrt(ab)))
        return ("t = 0: four new digits from noise, none of them in the training set; the "
                "last steps only sharpen what the early steps decided")

    # -- the charts -----------------------------------------------------------------------
    def _frame_axes(self, sc, op):
        """The chart box's two axes."""
        (x0, x1), (y0, y1) = CHART
        cam = self.cam
        sc.line(cam.frac(x0, y1), cam.frac(x1, y1), width=1.0, color=AXIS_INK, opacity=0.7 * op,
                bias=0.0)
        sc.line(cam.frac(x0, y1), cam.frac(x0, y0), width=1.0, color=AXIS_INK, opacity=0.7 * op,
                bias=0.0)

    def _schedule_chart(self, sc, step, op):
        """sqrt(abar_t) and sqrt(1 - abar_t) against t, with a marker at ``step``."""
        op *= self._cap
        if op <= 0.01:
            return
        (x0, x1), (y0, y1) = CHART
        cam, txt, T = self.cam, self.fr, self.T
        self._frame_axes(sc, op)

        def pt(t, v):
            return cam.frac(x0 + (x1 - x0) * t / T, y1 - (y1 - y0) * v)

        ab = self.tr.abar
        ts = np.arange(0, T + 1, 2)
        for vals, ink in ((np.sqrt(ab), SIGNAL_INK), (np.sqrt(1 - ab), NOISE_INK)):
            prev = None
            for tt in ts:
                p = pt(tt, vals[tt])
                if prev is not None:
                    sc.line(prev, p, width=1.4, color=ink, opacity=0.9 * op, bias=0.0)
                prev = p
            sc.voxel(pt(step, vals[step]), ink, op, glow=0.35, shadow=False, reflect=False,
                     size=0.7)
        sc.line(pt(step, 0.0), pt(step, 1.0), width=1.0, color=AXIS_INK, opacity=0.5 * op,
                bias=0.0)
        for tt in (0, T // 2, T):
            sc.text(*txt(x0 + (x1 - x0) * tt / T, y1 + 0.012), str(tt), 0.024, "regular",
                    layout.NOTE_INK, op, "mt")
        sc.text(*txt((x0 + x1) / 2, y1 + 0.05), "step t", 0.026, "regular", layout.CAPTION_INK,
                op, "mt")
        sc.text(*txt(x0 - 0.01, y0), "1", 0.024, "regular", layout.NOTE_INK, op, "rm")
        sc.text(*txt(x0, y0 - 0.012), "the schedule", 0.026, "regular", layout.CAPTION_INK, op,
                "lb")
        legend = (("%s  signal level" % SIGNAL, SIGNAL_INK), ("%s  noise level" % NOISE, NOISE_INK))
        for n, (text, ink) in enumerate(legend):
            sc.text(*txt(x0, y1 + 0.10 + 0.045 * n), text, 0.026, "regular", ink, op, "lm")

    def _loss_chart(self, sc, train, back):
        """The training loss (averaged over SMOOTH steps) against the step, log-log,
        drawn as ``train`` runs 0 -> 1."""
        op = smoothstep(0.0, 0.15, train) * back * self._cap
        if op <= 0.01:
            return
        (x0, x1), (y0, y1) = CHART
        cam, txt = self.cam, self.fr
        N = len(self.curve)
        lo, hi = self.log_loss
        k = max(1, int(round(N ** smoothstep(0.0, 0.9, train))))

        def fx(step):
            return x0 + (x1 - x0) * np.log10(step) / np.log10(N)

        def pt(step, loss):
            fy = y1 - (y1 - y0) * (np.log10(loss) - lo) / (hi - lo)
            return cam.frac(fx(step), fy)

        self._frame_axes(sc, op)
        sc.text(*txt(x0, y0 - 0.012), "training loss", 0.026, "regular", layout.CAPTION_INK, op,
                "lb")
        for step in (1, 10, 100, 1000):
            sc.text(*txt(fx(step), y1 + 0.012), str(step), 0.024, "regular", layout.NOTE_INK, op,
                    "mt")
        sc.text(*txt((x0 + x1) / 2, y1 + 0.05), "training steps", 0.026, "regular",
                layout.CAPTION_INK, op, "mt")
        for v in np.arange(lo, hi + 1e-9, 0.5):
            if abs(v - round(v)) < 1e-9:
                sc.text(*txt(x0 - 0.01, y1 - (y1 - y0) * (v - lo) / (hi - lo)), "%g" % 10 ** v,
                        0.024, "regular", layout.NOTE_INK, op, "rm")
        steps = np.unique(np.round(np.logspace(0, np.log10(N), 160)).astype(int))
        prev = None
        for step in [*steps[steps < k], k]:
            p = pt(step, self.curve[step - 1])
            if prev is not None:
                sc.line(prev, p, width=1.3, color=LOSS_INK, opacity=0.9 * op, bias=0.0)
            prev = p
        sc.voxel(prev, LOSS_INK, op, glow=0.3, shadow=False, reflect=False, size=0.6)
        sc.probe(("train",), (np.log10(k) * 40, 0, 0), tol=6.0)

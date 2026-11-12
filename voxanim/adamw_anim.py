"""AdamW is not Adam + L2: the two decays on a real weight matrix.

Two ways to shrink the weights while Adam trains the MNIST MLP
(784-64-32-10, 3 epochs, :mod:`voxanim.mlp_runs`):

* **Adam + L2** adds ``lambda w`` to the gradient, so the decay goes through
  Adam's moments like everything else and the step it causes is
  ``eta lambda w / (sqrt(v) + eps)``: a weight whose gradient is large barely
  decays, one whose gradient has died decays at up to ``eta`` a step;
* **AdamW** subtracts ``eta wd w`` from the weight directly, outside the
  moments: every weight shrinks by the same fraction ``eta wd`` a step.

The animation shows the middle weight matrix ``W1`` (64 x 32) of each run as
a sheet: the weights themselves, then Adam's running RMS gradient
``sqrt(v)`` per weight, then the *effective decay per step* of each weight
relative to AdamW's uniform rate on a diverging scale (orange: decays
faster than under AdamW, blue: slower; the AdamW sheet is flat), and finally
the weight norms of the three runs (plain Adam, Adam + L2, AdamW) over
training, with the decay sheet following the run.  Every value is read from
the recorded runs; the ratio sheet spans three orders of magnitude.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from . import color as C
from . import layout
from . import mlp_runs as R
from .easing import ease_in_out, remap, smoothstep
from .landscape import LossChart
from .netanim import NetAnimation, amax, div_colors, lit_colors
from .optim import EPS, Adam, AdamW
from .schedule_anim import sci
from .text import TIMES, sub

LR = 1e-3                   # Adam's learning rate, all three runs
L2 = 1e-3                   # lambda: the L2 strength added to the gradient
WD = 0.1                    # AdamW's weight decay: eta * wd = 1e-4 a step
RATIO_MAX = 1.0             # log10 ratio drawn at full colour (10 x either way)
V_PERCENTILES = (2, 98)     # the colour range of the sqrt(v) sheet
RUNS = (("Adam", (0.42, 0.44, 0.47)), ("Adam + L2", C.rgb("#d84e4e")), ("AdamW", C.rgb("#1f6fb5")))
STACK_AT = (0.30, 0.60)     # layout fraction of the two sheets' centre
SHEET_SCALE = 0.4
GAP = 2.0                   # world units between the two sheets
TEXT_X = 0.615              # frame x of the right-hand column
CHARTS = (((0.675, 0.945), (0.27, 0.47)), ((0.675, 0.945), (0.62, 0.82)))   # ||W0||, ||W1||
LEGEND_Y = 0.535            # frame y of the runs' legend, between the charts


def runs():
    """The three recorded runs: ``{"Adam", "Adam + L2", "AdamW"}``."""
    return {
        "Adam": R.train("adam", dict(optimizer="adam", lr=LR), partial(Adam, lr=LR)),
        "Adam + L2": R.train("adam_l2", dict(optimizer="adam", lr=LR, l2=L2),
                             partial(Adam, lr=LR), decay=L2),
        "AdamW": R.train("adamw", dict(optimizer="adamw", lr=LR, wd=WD),
                         partial(AdamW, lr=LR, weight_decay=WD)),
    }


def effective_decay(v):
    """Adam + L2's decay fraction per step for each weight, given the root
    second moment ``v`` there: ``eta lambda / (sqrt(v) + eps)``."""
    return LR * L2 / (v + EPS)


class AdamWvsL2(NetAnimation):
    """``adamw-vs-l2``: the two decays on ``W1``, then the weight norms."""

    PHASES = ("two rules", "gradients", "decay", "weight norms")

    def __init__(self):
        super().__init__("AdamW is not Adam + L2", "", 11.0)
        self.runs = runs()
        l2, adamw = self.runs["Adam + L2"], self.runs["AdamW"]
        self.hold = 2.5
        # the two sheets, stacked in world space: Adam + L2 above AdamW
        rows, cols = l2.W.shape[2], l2.W.shape[1]          # 32 outputs x 64 inputs
        self.sheets_ = {}
        centre = self.cam.frac(*STACK_AT)
        height = 2 * rows * SHEET_SCALE + GAP
        z = centre[2] + height / 2
        for key in ("Adam + L2", "AdamW"):
            s = self.sheet((rows, cols), STACK_AT, SHEET_SCALE, key)
            c = centre.copy()
            c[2] = z - rows * SHEET_SCALE / 2
            s.block.place(c)
            z -= rows * SHEET_SCALE + GAP
            self.sheets_[key] = s
        # the numbers the sheets and subtitles use
        both = np.concatenate([l2.v[-1].ravel(), adamw.v[-1].ravel()])
        self.v_range = tuple(float(v) for v in np.percentile(both, V_PERCENTILES))
        self.v_min, self.v_max = float(both.min()), float(both.max())
        d = effective_decay(l2.v[-1])
        self.ratio = d / (LR * WD)
        self.pct = {p: float(np.percentile(self.ratio, p)) for p in (10, 50, 90)}
        self.ratio_max = float(self.ratio.max())
        self.steps = l2.steps
        self.charts = [LossChart(self, box, self.steps, lo, hi) for box, (lo, hi) in
                       zip(CHARTS, self._norm_ranges(), strict=True)]
        # ---- timeline
        Tm, t = {}, 0.3
        Tm["rules"] = (t, t + 9.0)
        t += 9.0 + 0.3
        Tm["grad"] = (t, t + 8.0)
        t += 8.0 + 0.3
        Tm["decay"] = (t, t + 14.0)
        t += 14.0 + 0.3
        Tm["norms"] = (t, t + 15.0)
        t += 15.0 + 1.5
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["grad"][0], Tm["decay"][0], Tm["norms"][0]]

    def _norm_ranges(self):
        """``[(lo, hi)]`` of the two norm charts (W0, W1), with a margin."""
        out = []
        for j in range(2):
            vals = np.concatenate([r.norms[:, j] for r in self.runs.values()])
            lo, hi = float(vals.min()), float(vals.max())
            out.append((lo - 0.08 * (hi - lo), hi + 0.08 * (hi - lo)))
        return out

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        rules = self.progress("rules", t)
        grad = self.progress("grad", t)
        decay = self.progress("decay", t)
        norms = self.progress("norms", t)
        intro = smoothstep(0.0, 0.5, rules) * back
        self.fill_sheets(t, grad, decay, norms, intro)
        self.formulas(sc, smoothstep(0.3, 0.8, rules) * (1 - smoothstep(0.0, 0.08, norms)) * back)
        self.norm_charts(sc, norms, back)
        self.subtitle = self.words(t, ph, rules, grad, decay, norms)
        self.step_indicator(sc, ph, self.PHASES)
        self.captions(sc, grad, decay, norms, intro)

    def snapshot(self, norms):
        """Index of the Adam + L2 snapshot the norms phase is at."""
        run = self.runs["Adam + L2"]
        step = norms * self.steps
        return int(np.searchsorted(run.snap_steps, step, side="right") - 1) if norms > 0 else -1

    def fill_sheets(self, t, grad, decay, norms, op):
        """What the sheets show now: weights, then sqrt(v), then the decay ratio."""
        lo, hi = np.log10(self.v_range[0]), np.log10(self.v_range[1])
        for key, s in self.sheets_.items():
            run = self.runs[key]
            W, v = run.W[-1], run.v[-1]
            if norms > 0 and key == "Adam + L2":          # the decay sheet follows the run
                i = max(0, self.snapshot(norms))
                W, v = run.W[i], run.v[i]
            if decay > 0:
                if key == "Adam + L2":
                    ratio = np.log10(effective_decay(v) / (LR * WD))
                    cols = div_colors(ratio.T.ravel(), RATIO_MAX)
                else:
                    cols = div_colors(np.zeros(W.size), RATIO_MAX)
                mix = smoothstep(0.0, 0.25, decay)
                prev = lit_colors(np.clip((np.log10(v.T.ravel()) - lo) / (hi - lo), 0, 1), 1.0)
                cols = prev + (cols - prev) * mix
            elif grad > 0:
                m = np.clip((np.log10(v.T.ravel()) - lo) / (hi - lo), 0.0, 1.0)
                cols = lit_colors(m, 1.0)
                prev = div_colors(W.T.ravel(), amax(W))
                cols = prev + (cols - prev) * smoothstep(0.0, 0.25, grad)
            else:
                cols = div_colors(W.T.ravel(), amax(W))
            s.fill(cols)
            s.cage = 1.0
            s.fade(op)

    def captions(self, sc, grad, decay, norms, op):
        """The sheet labels, left of each sheet."""
        what = "weights W%s (64 %s 32), after training" % (sub(1), TIMES)
        if decay > 0.25:
            what = "decay per step, relative to AdamW's"
        elif grad > 0.25:
            what = "RMS gradient per weight, √v"
        for key, s in self.sheets_.items():
            self.caption_left(sc, s, key, op)
        self.caption(sc, self.sheets_["AdamW"], what, op, size=0.027, max_width=0.5)
        if decay > 0.25:
            text = ("orange: faster than under AdamW, blue: slower; full colour is %d%s"
                    % (int(10 ** RATIO_MAX), TIMES))
            if norms > 0:
                i = max(0, self.snapshot(norms))
                text = "Adam + L2's decay at step %d" % self.runs["Adam + L2"].snap_steps[i]
            self.caption(sc, self.sheets_["AdamW"], text, op, size=0.026, offset=0.04,
                         max_width=0.6, ink=layout.NOTE_INK)

    def formulas(self, sc, op):
        """The two update rules, right of the sheets."""
        op *= self._cap
        if op <= 0.01:
            return
        lines = [
            ("Adam + L2  (λ = %g)" % L2, layout.STRONG_INK, "regular"),
            ("g = ∂L/∂w + λ w", layout.CAPTION_INK, "mono"),
            ("m, v ← running means of g, g²", layout.CAPTION_INK, "mono"),
            ("w ← w − η m/(√v + ε)", layout.CAPTION_INK, "mono"),
            ("the decay λw is divided by √v too", layout.NOTE_INK, "regular"),
            ("", layout.NOTE_INK, "regular"),
            ("AdamW  (wd = %g)" % WD, layout.STRONG_INK, "regular"),
            ("g = ∂L/∂w", layout.CAPTION_INK, "mono"),
            ("m, v ← running means of g, g²", layout.CAPTION_INK, "mono"),
            ("w ← w − η m/(√v + ε) − η wd w", layout.CAPTION_INK, "mono"),
            ("every weight shrinks by η wd = %s a step" % sci(LR * WD), layout.NOTE_INK,
             "regular"),
        ]
        y = 0.30
        for text, ink, kind in lines:
            if text:
                sc.text(TEXT_X, y, text, 0.027, kind, ink, op, "lm")
            y += 0.042 if text else 0.02

    def norm_charts(self, sc, norms, back):
        """``||W0||`` and ``||W1||`` of the three runs against the step, drawn as
        the norms phase runs."""
        op = smoothstep(0.1, 0.3, norms) * back
        if op <= 0.01:
            return
        upto = self.steps * smoothstep(0.2, 1.0, norms)
        for j, chart in enumerate(self.charts):
            chart.frame(sc, op, xlabel="step", ylabel="‖W%s‖" % sub(j))
            for name, col in RUNS:
                run = self.runs[name]
                chart.curve(sc, run.norms[:, j], upto, col, op, width=1.4)
        # legend between the charts
        x = TEXT_X
        for name, col in RUNS:
            sc.text(x, LEGEND_Y, name, 0.026, "regular", _ink(col), op * self._cap, "lm")
            x += layout.text_width(name, 0.026) + 0.022

    def words(self, t, ph, rules, grad, decay, norms):
        """The subtitle for the moment."""
        l2, adamw, plain = self.runs["Adam + L2"], self.runs["AdamW"], self.runs["Adam"]
        if ph == 0:
            if rules < 0.5 or t >= self.t_done + self.hold:
                return ("two ways to shrink the weights while Adam trains the MNIST MLP "
                        "(784-64-32-10, %d steps): L2 in the loss, or decay on the weight itself"
                        % self.steps)
            return ("W%s of each run after training: with the usual values (λ = %g, wd = %g) "
                    "the two look alike; the difference is in who gets decayed how much"
                    % (sub(1), L2, WD))
        if ph == 1:
            return ("Adam divides every step by √v, the running RMS of each weight's gradient: "
                    "mostly %.0e to %.0e here, but the quietest of the %d weights is at %.0e"
                    % (self.v_range[0], self.v_range[1], l2.W[-1].size, self.v_min))
        if ph == 2:
            if decay < 0.5:
                return ("Adam + L2: the decay λw is divided by √v too, so a weight's decay is "
                        "ηλ/√v: %.1f%s AdamW's rate where the gradient is large, %d%s where it "
                        "has died" % (self.pct[10], TIMES, int(round(self.ratio_max, -2)), TIMES))
            return ("AdamW: every weight shrinks by η%swd = %s a step whatever its gradient, "
                    "so its sheet is flat; the two decays agree only at the median (%.1f%s)"
                    % (TIMES, sci(LR * WD), self.pct[50], TIMES))
        if norms < 1:
            k = int(norms * self.steps)
            return ("the weight norms over training (step %d of %d): plain Adam lets them grow, "
                    "AdamW holds them back evenly, Adam + L2 collapses the quiet weights"
                    % (k, self.steps))
        return ("after %d steps: ‖W%s‖ %.1f, %.1f, %.1f and ‖W%s‖ %.1f, %.1f, %.1f (Adam, "
                "Adam + L2, AdamW); test accuracy %.1f%%, %.1f%%, %.1f%%"
                % (self.steps, sub(0), plain.norms[-1, 0], l2.norms[-1, 0], adamw.norms[-1, 0],
                   sub(1), plain.norms[-1, 1], l2.norms[-1, 1], adamw.norms[-1, 1],
                   100 * plain.acc, 100 * l2.acc, 100 * adamw.acc))


def _ink(col):
    """A readable text colour from a line colour."""
    return tuple(float(v) for v in np.asarray(col) * 0.85)

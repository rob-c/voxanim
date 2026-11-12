"""Three walks on the loss landscape of :mod:`voxanim.landscape`.

All three draw the same thing: the mean log-loss of two-weight logistic
regression on the MNIST zeros and ones as a field of voxel columns (one per
grid cell, as tall as the loss there, terraced like contour lines), a walker
for each optimiser (a glowing voxel floating at its true loss above the
surface, trailing its path), and the loss-against-step read-out beside it.
Every position, arrow, ellipse and number comes from a recorded run
(:func:`voxanim.landscape.descend`), cached with its settings.

**``loss-landscape-descent``** -- *A loss landscape.*  The landscape rises
from the floor (what the axes are, where the loss is high and low, where the
minimum is), then full-batch gradient descent walks down from a fixed start:
the gradient arrow at every step, the step ``-eta grad L`` visibly shrinking
as the gradient does, the loss falling.  The reference frame for the others.

**``drunken-walk``** -- *SGD: the drunken walk.*  The same start with
minibatch SGD: three walkers whose gradients come from 1, 8 and 64 random
examples a step, against the full-batch path.  A minibatch gradient is the
full gradient plus noise with covariance ``Sigma / B`` (``Sigma`` the
per-example gradient covariance), so the step's one-sigma ellipse is
``eta sqrt(Sigma / B)``: drawn around each walker, it shrinks by ``sqrt(8)``
from one walker to the next, and the paths stagger by exactly that much.
Then the learning rate too high (the walker never settles: its steps stay
long around the minimum) and too low (it stops short in the steps given).

**``optimisers``** -- *Optimisers on a narrow valley.*  The ill-conditioned
landscape (ink times 8): SGD, momentum, RMSProp, Adam and AdamW from the
same start.  SGD crosses the valley in one step, then crawls along it;
momentum builds speed and overshoots; RMSProp and Adam divide each weight's
step by the running size of its gradient (the raw gradient and the rescaled
step are drawn as two arrows, the running moments as bars), so both weights
move about ``eta`` a step and they reach the floor first; AdamW's decay
holds it short of the minimum.  Who arrives in how many steps is counted.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from . import color as C
from . import landscape as LS
from . import layout
from .conv import Tempo
from .easing import ease_in_out, ease_out, remap, smoothstep
from .netanim import NetAnimation
from .optim import SGD, Adam, AdamW, Momentum, RMSProp
from .text import MINUS, TIMES, num, sub

# ------------------------------------------------------------------ the runs

START = (2.5, -2.0)                 # where every walk on the round bowl starts
GD_LR, GD_STEPS = 3.0, 60           # full-batch gradient descent
BOX_LO, BOX_HI = (-7.0, -7.0), (3.0, 3.0)     # the drawn weight box, round bowl
BOX_CAP = 1.6                       # loss at the top of the tallest column
SGD_LR, SGD_STEPS = 3.0, 80         # the drunken walkers
BATCHES = (1, 8, 64)
LR_HIGH, LR_LOW = 8.0, 0.3          # the learning rate too high / too low
SEEDS = {"b1": 11, "b8": 12, "b64": 13, "high": 14, "low": 15}   # one generator per walker
ILL_LO, ILL_HI = (-5.0, -5.0), (3.0, 3.0)     # the drawn box of the narrow valley
ILL_CAP = 4.0
ILL_START = (0.3, 2.0)
ILL_STEPS = 150
ARRIVED = 0.01                      # "at the minimum": loss within this of the minimum
# optimiser name -> (class, keyword arguments); the learning rates were chosen
# so plain SGD is just stable across the valley (1.8 / largest curvature at
# the origin) and the adaptive ones take steps of about 0.1 a weight
OPTIMISERS = {
    "SGD": (SGD, dict(lr=0.11)),
    "momentum": (Momentum, dict(lr=0.03, beta=0.9)),
    "RMSProp": (RMSProp, dict(lr=0.1)),
    "Adam": (Adam, dict(lr=0.1)),
    "AdamW": (AdamW, dict(lr=0.1, weight_decay=0.1)),
}
OPT_COLORS = {
    "SGD": C.rgb("#d84e4e"), "momentum": C.rgb("#e0952e"), "RMSProp": C.rgb("#3d9e45"),
    "Adam": C.rgb("#1f6fb5"), "AdamW": C.rgb("#8952eb"),
}
FULL_INK = (0.22, 0.24, 0.26)       # the full-batch path
WALKER = LS.ARROW_STEP              # the single walker of the first animation
BATCH_COLORS = {1: C.rgb("#d84e4e"), 8: C.rgb("#e0952e"), 64: C.rgb("#1f6fb5")}
HIGH_INK, LOW_INK = C.rgb("#c0399b"), C.rgb("#5f9a45")


def gd_runs():
    """``{"gd": Run}``: full-batch gradient descent on the round bowl."""
    settings = dict(problem="mnist-0v1-ink-width", start=START, lr=GD_LR, steps=GD_STEPS,
                    optimizer="sgd-full-batch")

    def build():
        return {"gd": LS.descend(LS.problem(), partial(SGD, lr=GD_LR), START, GD_STEPS)}
    return LS.run_cache("landscape_gd", settings, build)


def sgd_runs():
    """The drunken walkers: ``{"full", "b1", "b8", "b64", "high", "low"}``."""
    settings = dict(problem="mnist-0v1-ink-width", start=START, lr=SGD_LR, steps=SGD_STEPS,
                    batches=BATCHES, lr_high=LR_HIGH, lr_low=LR_LOW, seeds=SEEDS,
                    sampling="with replacement")

    def build():
        prob = LS.problem()
        out = {"full": LS.descend(prob, partial(SGD, lr=SGD_LR), START, SGD_STEPS)}
        for b in BATCHES:
            out["b%d" % b] = LS.descend(prob, partial(SGD, lr=SGD_LR), START, SGD_STEPS,
                                        batch=b, rng=np.random.default_rng(SEEDS["b%d" % b]))
        for key, lr in (("high", LR_HIGH), ("low", LR_LOW)):
            out[key] = LS.descend(prob, partial(SGD, lr=lr), START, SGD_STEPS, batch=8,
                                  rng=np.random.default_rng(SEEDS[key]))
        return out
    return LS.run_cache("landscape_sgd", settings, build)


def optimiser_runs():
    """``{name: Run}`` of :data:`OPTIMISERS` on the narrow valley."""
    settings = dict(problem="mnist-0v1-ink-width", scale=LS.ILL_SCALE, start=ILL_START,
                    steps=ILL_STEPS, optimisers={k: [cls.__name__, kw]
                                                 for k, (cls, kw) in OPTIMISERS.items()})

    def build():
        ill = LS.problem().scaled(LS.ILL_SCALE)
        return {name: LS.descend(ill, partial(cls, **kw), ILL_START, ILL_STEPS)
                for name, (cls, kw) in OPTIMISERS.items()}
    return LS.run_cache("landscape_optimisers", settings, build)


# ------------------------------------------------------------------ shared drawing

LAND_AT = (0.35, 0.60)              # layout fraction of the landscape's centre
CHART = ((0.715, 0.945), (0.28, 0.56))      # the loss read-out's box
READ_X = 0.715                      # frame x of the read-outs under the chart
READ_Y = 0.63
MIN_FRAME = (0.98, 0.98, 0.97)


def wtxt(w):
    """``(w1, w2)`` as text: ``(−4.29, −1.77)``."""
    return "(%s, %s)" % (num(w[0]), num(w[1]))


class LandscapeAnimation(NetAnimation):
    """Base of the three: the landscape, its entrance, the chart, a walker."""

    PHASES = ()

    def __init__(self, title, prob, lo, hi, cap, steps, loss_hi, ppu=15.5):
        super().__init__(title, "", ppu, azimuth=33.0, elevation=34.0)
        self.prob = prob
        self.land = LS.Landscape(self, prob, lo, hi, cap, LAND_AT)
        self.wmin = prob.minimum()
        self.Lmin = prob.loss(self.wmin)
        self.chart = LS.LossChart(self, CHART, steps, self.Lmin - 0.03, loss_hi)
        self.hold = 2.5

    # -- timeline helpers --------------------------------------------------------
    def sweep(self, name, n, tempo):
        """Register a ``Tempo`` sweep of ``n`` steps as phase ``name`` starting at
        the current end of the timeline; returns its duration."""
        d = tempo.durations(n)
        starts = np.concatenate([[0.0], np.cumsum(d)[:-1]])
        self._sweeps = getattr(self, "_sweeps", {})
        self._sweeps[name] = (starts, d)
        return float(d.sum())

    def step_at(self, name, t):
        """Fractional step of sweep ``name`` at time ``t``: ``k + u`` with ``u``
        the eased progress through step ``k`` (0 before, ``n`` after)."""
        t0, t1 = self.Tm[name]
        starts, d = self._sweeps[name]
        if t < t0:
            return 0.0
        if t >= t1:
            return float(len(d))
        k = int(np.searchsorted(starts, t - t0, side="right") - 1)
        return k + ease_in_out(remap((t - t0 - starts[k]) / d[k], 0.0, 0.7))

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def lay(self, fx, fy):
        """Frame fraction -> layout fraction: the inverse of :meth:`fr`, for
        geometry that must line up with text placed in frame fractions."""
        k = self.cam.zoom
        return 0.5 + (fx - 0.5) / k, 0.5 + (fy - 0.5) / k

    # -- the landscape -----------------------------------------------------------
    def landscape(self, sc, t, back, reveal):
        """Columns, axes and the minimum's marker; ``reveal`` is the entrance."""
        self.land.show(LS.reveal_curve(reveal), back)
        self.land.axes(sc, smoothstep(0.5, 1.0, reveal) * back)

    def minimum_marker(self, sc, op):
        """A small frame on the surface at the minimum, labelled with its loss."""
        if op <= 0.01:
            return
        p = self.land.point(self.wmin, self.Lmin, lift=0.0)
        s = 0.45 * self.land.block.scale
        sc.wire_box(p - s, p + s, color=MIN_FRAME, width=1.3, opacity=0.95 * op, bias=0.1)

    def walker(self, sc, run, pos, color, op, key, trail=True, size=0.9, fade_len=None):
        """A run's walker at fractional step ``pos`` with its trail."""
        k = min(int(np.floor(pos)), run.steps)
        f = pos - k
        if k < run.steps and f > 0:
            w = run.w[k] + (run.w[k + 1] - run.w[k]) * f
            L = run.loss[k] + (run.loss[k + 1] - run.loss[k]) * f
        else:
            w, L = run.w[k], run.loss[k]
        if trail:
            self.land.trail(sc, run, pos, color, 0.9 * op, fade_len=fade_len)
        self.land.marker(sc, w, L, color, op, size=size, key=key)
        return w, L

    def chart_frame(self, sc, op, ticks=True):
        """The chart's axes and the dashed line at the minimum."""
        self.chart.frame(sc, op)
        self.chart.hline(sc, self.Lmin, 0.8 * op)
        if ticks:
            self.chart.label(sc, 0, self.Lmin, "min %.3f" % self.Lmin, layout.NOTE_INK, op,
                             dx=-0.006, anchor="rm", size=0.024)

    def readout(self, sc, lines, op, y=READ_Y, dy=0.045, ink=None):
        """Lines of text under the chart (the first in strong ink)."""
        op *= self._cap
        if op <= 0.01:
            return
        for n, line in enumerate(lines):
            if not line:
                continue
            col = ink or (layout.STRONG_INK if n == 0 else layout.CAPTION_INK)
            sc.text(READ_X, y + n * dy, line, 0.03 if n == 0 else 0.027, "regular", col, op, "lm")


# ------------------------------------------------------------------ 1. gradient descent


class LossLandscape(LandscapeAnimation):
    """``loss-landscape-descent``: the landscape, then gradient descent on it."""

    PHASES = ("landscape", "gradient", "descent", "arrived")

    def __init__(self):
        prob = LS.problem()
        self.run = gd_runs()["gd"]
        super().__init__("A loss landscape", prob, BOX_LO, BOX_HI, BOX_CAP, GD_STEPS,
                         float(self.run.loss[0]) + 0.1)
        self.g0 = self.run.g[0]
        self.step_len = np.linalg.norm(np.diff(self.run.w, axis=0), axis=1)
        self.arrived = self.run.first_within(self.Lmin, ARRIVED)
        self.acc = prob.accuracy(self.wmin)
        # ---- timeline
        Tm, t = {}, 0.3
        Tm["rise"] = (t, t + 3.5)
        t += 3.5 + 0.4
        Tm["survey"] = (t, t + 5.0)          # where the loss is high and low, the minimum
        t += 5.0 + 0.3
        Tm["start"] = (t, t + 1.6)           # the start point lands
        t += 1.6
        Tm["grad"] = (t, t + 2.6)            # the gradient arrow, the step it implies
        t += 2.6 + 0.6
        d = self.sweep("walk", GD_STEPS, Tempo(slow=1.3, fast=0.22, ramp_in=5.0, ramp_out=3.0))
        Tm["walk"] = (t, t + d)
        t += d + 0.4
        Tm["done"] = (t, t + 3.0)
        t += 3.0
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["start"][0], Tm["walk"][0], Tm["done"][0]]

    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        rise = self.progress("rise", t)
        survey = self.progress("survey", t)
        start = ease_out(self.progress("start", t))
        grad = self.progress("grad", t)
        pos = self.step_at("walk", t)
        done = self.progress("done", t)
        run = self.run
        self.landscape(sc, t, back, rise)
        self.minimum_marker(sc, smoothstep(0.45, 0.6, survey) * back)
        self.readout(sc, ["the minimum", "loss %.3f" % self.Lmin, "at w = %s" % wtxt(self.wmin),
                          "%.1f%% of the set right" % (100 * self.acc)],
                     smoothstep(0.5, 0.7, survey) * (1 - start) * back)
        # the walker, its trail, the arrow of the step about to be taken
        if start > 0:
            k = min(int(np.floor(pos)), run.steps)
            self.walker(sc, run, pos, WALKER, start * back, ("walker",))
            if k < run.steps and pos - k < 0.02 and (grad > 0 or pos > 0):
                # the gradient (uphill) and the step -eta grad (downhill)
                here = self.land.point(run.w[k], run.loss[k])
                step = self.land.step_vec(run.w[k], run.w[k + 1])
                show = smoothstep(0.0, 0.3, grad) if pos == 0 else 1.0
                if pos == 0:
                    # the gradient points uphill: the step reversed (drawn at the
                    # step's length; its true length is the step's over eta)
                    self.land.arrow(sc, here, -step, LS.ARROW_GRAD, show * back, width=1.4)
                self.land.arrow(sc, here, step, LS.ARROW_STEP, (smoothstep(0.45, 0.75, grad)
                                                                if pos == 0 else 1.0) * back)
            self.gradient_notes(sc, grad * (1.0 - smoothstep(0.0, 0.5, pos)) * back)
        chart = smoothstep(0.0, 0.4, start) * back
        self.chart_frame(sc, chart)
        self.chart.curve(sc, run.loss, pos, FULL_INK, chart)
        self.chart.label(sc, 0, run.loss[0], "start %.2f" % run.loss[0], layout.NOTE_INK,
                         chart * (1 - smoothstep(0.0, 0.1, pos / GD_STEPS)), dx=0.012,
                         anchor="lm", size=0.024)
        self.step_readout(sc, pos, chart * smoothstep(0.0, 0.2, pos + grad))
        self.subtitle = self.words(t, ph, survey, grad, pos, done)
        self.step_indicator(sc, ph, self.PHASES)

    def gradient_notes(self, sc, op):
        """The gradient and the step at the start, as numbers by the arrows."""
        op *= self._cap
        if op <= 0.01:
            return
        run = self.run
        p = self.land.point(run.w[0], run.loss[0])
        s = self.cam.project(p)
        x, y = s[0] / self.cam.width, s[1] / self.cam.height
        sc.text(x + 0.03, y - 0.075, "∂L/∂w = %s" % wtxt(self.g0), 0.027, "regular",
                _ink(LS.ARROW_GRAD), op, "lm")
        sc.text(x + 0.03, y - 0.04, "step = %s%g %s ∂L/∂w" % (MINUS, GD_LR, TIMES), 0.027,
                "regular", _ink(LS.ARROW_STEP), op, "lm")

    def step_readout(self, sc, pos, op):
        """Step, loss and the step's length, under the chart."""
        k = min(int(round(pos)), self.run.steps)
        L = self.run.loss[k]
        ln = self.step_len[min(k, GD_STEPS - 1)]
        self.readout(sc, ["step %d of %d" % (k, GD_STEPS), "loss %.3f" % L,
                          "|∂L/∂w| = %.3f" % (ln / GD_LR),
                          "|Δw| = %g %s %.3f = %.3f" % (GD_LR, TIMES, ln / GD_LR, ln)], op)

    def words(self, t, ph, survey, grad, pos, done):
        """The subtitle for the moment."""
        run = self.run
        if ph == 0 and (survey < 0.45 or t >= self.t_done + self.hold):
            return ("logistic regression on %s MNIST zeros and ones: p(one) = σ(w%s ink + w%s "
                    "width); the height is the mean log-loss"
                    % (_thousands(self.prob.n), sub(1), sub(2)))
        if ph == 0:
            return ("high where the weights call the digits wrong (ones are faint and thin, so "
                    "both weights must be negative); the minimum: loss %.3f, %.1f%% right"
                    % (self.Lmin, 100 * self.acc))
        if ph == 1:
            if grad < 0.45:
                return ("gradient descent starts at w = %s, loss %.2f; the gradient ∂L/∂w = %s "
                        "points uphill" % (wtxt(run.w[0]), run.loss[0], wtxt(self.g0)))
            return ("each step goes the other way, η = %g times as far as the gradient: Δw = %s%g "
                    "%s ∂L/∂w = %s" % (GD_LR, MINUS, GD_LR, TIMES, wtxt(run.w[1] - run.w[0])))
        if ph == 2:
            k = min(int(pos), GD_STEPS)
            if k < 12:
                return ("the step is always %g %s the gradient: long where the surface is "
                        "steep, shorter as it flattens (step %d moves %.2f)"
                        % (GD_LR, TIMES, k, self.step_len[min(k, GD_STEPS - 1)]))
            return ("the loss falls fast, then slowly: the floor of the bowl is nearly flat, "
                    "so the gradient and the steps are tiny (step %d moves %.3f)"
                    % (k, self.step_len[min(k, GD_STEPS - 1)]))
        return ("after %d steps: loss %.3f, %.3f above the minimum; it was within %.2f of it "
                "from step %d on" % (GD_STEPS, run.loss[-1], run.loss[-1] - self.Lmin, ARRIVED,
                                     self.arrived))


def _ink(col):
    """A readable text colour from a line colour (darkened a little)."""
    return tuple(float(v) for v in np.asarray(col) * 0.85)


def _thousands(n):
    """``12665`` -> ``12,665``."""
    return f"{int(n):,}"


# ------------------------------------------------------------------ 2. the drunken walk


class DrunkenWalk(LandscapeAnimation):
    """``drunken-walk``: minibatch SGD with 1, 8 and 64 examples a step against
    the full-batch path; the predicted spread; a learning rate too high, too low."""

    PHASES = ("landscape", "minibatches", "spread", "too high", "too low")

    def __init__(self):
        prob = LS.problem()
        self.runs = sgd_runs()
        super().__init__("SGD: the drunken walk", prob, BOX_LO, BOX_HI, BOX_CAP, SGD_STEPS,
                         float(self.runs["full"].loss[0]) + 0.1)
        # the predicted one-sigma spread of a step at the end of each walk
        self.sigma = {b: SGD_LR * np.sqrt(np.diag(prob.grad_cov(self.runs["b%d" % b].w[-1]) / b))
                      for b in BATCHES}
        last = SGD_STEPS // 2
        self.mean_step = {k: float(np.linalg.norm(np.diff(self.runs[k].w[last:], axis=0),
                                                  axis=1).mean()) for k in ("high", "low", "b8")}
        self.short = float(np.linalg.norm(self.runs["low"].w[-1] - self.wmin))
        # ---- timeline
        Tm, t = {}, 0.3
        Tm["rise"] = (t, t + 2.6)
        t += 2.6 + 0.2
        d = self.sweep("full", SGD_STEPS, Tempo(slow=0.2, fast=0.03, ramp_in=3.0, ramp_out=1.0))
        Tm["full"] = (t, t + d)
        t += d + 1.2
        d = self.sweep("walk", SGD_STEPS, Tempo(slow=1.0, fast=0.18, ramp_in=4.0, ramp_out=3.0))
        Tm["walk"] = (t, t + d)
        t += d + 0.4
        Tm["spread"] = (t, t + 7.0)
        t += 7.0 + 0.4
        d = self.sweep("high", SGD_STEPS, Tempo(slow=0.6, fast=0.1, ramp_in=4.0, ramp_out=3.0))
        Tm["high"] = (t, t + d)
        t += d + 2.0
        d = self.sweep("low", SGD_STEPS, Tempo(slow=0.5, fast=0.08, ramp_in=4.0, ramp_out=3.0))
        Tm["low"] = (t, t + d)
        t += d + 2.5
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["walk"][0], Tm["spread"][0], Tm["high"][0] - 0.4, Tm["low"][0] - 1.2]

    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        rise = self.progress("rise", t)
        full = self.step_at("full", t)
        pos = self.step_at("walk", t)
        spread = self.progress("spread", t)
        high = self.step_at("high", t)
        low = self.step_at("low", t)
        runs = self.runs
        self.landscape(sc, t, back, rise)
        self.minimum_marker(sc, smoothstep(0.6, 1.0, rise) * back)
        # the full-batch path, always there once drawn
        walkers = t < self.marks[3]
        tail = 1.0 if walkers else 0.55           # the reference fades back a little
        self.land.trail(sc, runs["full"], full, FULL_INK, 0.9 * tail * back, width=1.2)
        if full > 0 and walkers:
            self.land.marker(sc, *runs["full"].at(min(int(full), SGD_STEPS)), FULL_INK,
                             0.9 * back, size=0.7, glow=0.0)
        chart = smoothstep(0.1, 0.6, full / SGD_STEPS) * back
        self.chart_frame(sc, chart)
        self.chart.curve(sc, runs["full"].loss, full, FULL_INK, chart * tail, width=1.2,
                         marker=walkers)
        if walkers:
            self.minibatch_walkers(sc, pos, spread, back)
        else:
            self.rate_walkers(sc, high, low, back)
        self.subtitle = self.words(t, ph, pos, spread, high, low)
        self.step_indicator(sc, ph, self.PHASES)

    def minibatch_walkers(self, sc, pos, spread, back):
        """The three walkers, their ellipses and curves, and the legend."""
        runs = self.runs
        on = smoothstep(0.0, 0.3, pos + spread) * back
        for b in BATCHES:
            run, col = runs["b%d" % b], BATCH_COLORS[b]
            w, L = self.walker(sc, run, pos, col, on, ("b", b))
            # the one-sigma ellipse of the next step, live at the walker
            if pos > 0:
                cov = self.prob.grad_cov(w) / b * SGD_LR ** 2
                self.land.ellipse(sc, w, L, cov, col, 0.75 * on)
            self.chart.curve(sc, run.loss, pos, col, on, width=1.3)
        # legend under the chart
        y = READ_Y
        self.readout(sc, ["B = 1", "B = 8", "B = 64", "full batch (%s)" % _thousands(self.prob.n)],
                     on, y=y, dy=0.04, ink=layout.CAPTION_INK)
        for n, b in enumerate(BATCHES):
            sc.text(READ_X - 0.014, y + n * 0.04, "■", 0.03, "regular", BATCH_COLORS[b],
                    on * self._cap, "rm")
        sc.text(READ_X - 0.014, y + 3 * 0.04, "■", 0.03, "regular", FULL_INK, on * self._cap,
                "rm")
        # the spread, as numbers, once the walkers rest
        if spread > 0:
            op = smoothstep(0.0, 0.3, spread) * back * self._cap
            for n, b in enumerate(BATCHES):
                s1, s2 = self.sigma[b]
                sc.text(READ_X + 0.07, y + n * 0.04, "step σ = (%.2f, %.2f)" % (s1, s2),
                        0.026, "regular", _ink(BATCH_COLORS[b]), op, "lm")

    def rate_walkers(self, sc, high, low, back):
        """The too-high and too-low walkers, in turn, against the full path."""
        runs = self.runs
        for key, pos, col, lr in (("high", high, HIGH_INK, LR_HIGH), ("low", low, LOW_INK, LR_LOW)):
            if pos <= 0 and key == "low":
                continue
            on = back * (1.0 if key == "low" or low <= 0 else 0.45)
            w, _L = self.walker(sc, runs[key], pos, col, on, (key,))
            self.chart.curve(sc, runs[key].loss, pos, col, on, width=1.3)
            k = min(int(round(pos)), SGD_STEPS)
            if (key == "high" and low <= 0) or key == "low":
                self.readout(sc, ["η = %g, B = 8" % lr, "step %d of %d" % (k, SGD_STEPS),
                                  "loss %.3f" % runs[key].loss[k],
                                  "|w %s w*| = %.2f" % (MINUS, np.linalg.norm(w - self.wmin))],
                             on)
        self.readout(sc, ["", "", "", "", "full batch, η = %g: grey" % SGD_LR], back * 0.9,
                     ink=layout.NOTE_INK)

    def words(self, t, ph, pos, spread, high, low):
        """The subtitle for the moment."""
        runs = self.runs
        n = _thousands(self.prob.n)
        if ph == 0:
            if t < self.Tm["full"][0] + 0.3 or t >= self.t_done + self.hold:
                return ("the same landscape; full-batch gradient descent (grey) uses all %s "
                        "examples for every gradient: %d steps, loss %.2f to %.3f"
                        % (n, SGD_STEPS, runs["full"].loss[0], runs["full"].loss[-1]))
            return ("stochastic gradient descent uses a minibatch: B examples drawn at random, "
                    "a different draw every step; same start, same η = %g" % SGD_LR)
        if ph == 1:
            k = min(int(pos), SGD_STEPS)
            if k < 20:
                return ("B = 1: one example's gradient a step, so the walk staggers; B = 8 and "
                        "64 average more and stagger less (step %d)" % k)
            return ("on average a minibatch gradient is the full gradient, so all three still "
                    "walk down; step %d: loss %.3f, %.3f, %.3f (full batch %.3f)"
                    % (k, runs["b1"].loss[k], runs["b8"].loss[k], runs["b64"].loss[k],
                       runs["full"].loss[k]))
        if ph == 2:
            if spread < 0.5:
                return ("the stagger is predicted: a minibatch gradient is the full gradient plus "
                        "noise with covariance Σ/B, Σ the covariance of the %s examples' "
                        "gradients" % n)
            return ("the ellipses are one sigma of a step, η√(Σ/B): B = 8 is √8 = 2.8 times "
                    "tighter than B = 1, B = 64 another 2.8 times")
        if ph == 3:
            k = min(int(high), SGD_STEPS)
            if k < 15:
                return ("learning rate too high: η = %g, B = 8; the first step is %.1f long "
                        "and lands beyond the minimum; the next ones are big too"
                        % (LR_HIGH, np.linalg.norm(runs["high"].w[1] - runs["high"].w[0])))
            return ("the noise is η times the gradient's too, so the walker never settles: its "
                    "steps over the last %d stay %.2f long (η = %g: %.2f)"
                    % (SGD_STEPS // 2, self.mean_step["high"], SGD_LR, self.mean_step["b8"]))
        k = min(int(low), SGD_STEPS)
        if k >= SGD_STEPS - 5:
            return ("learning rate too low: η = %g, B = 8; steady, but after %d steps it is still "
                    "at loss %.2f, %.1f from the minimum: it stops short in the time given"
                    % (LR_LOW, SGD_STEPS, runs["low"].loss[-1], self.short))
        return ("learning rate too low: η = %g, B = 8; the steps are tiny (about %.2f), the loss "
                "creeps down (step %d: %.3f)" % (LR_LOW, self.mean_step["low"], k,
                                                 runs["low"].loss[k]))


# ------------------------------------------------------------------ 3. optimisers


class Optimisers(LandscapeAnimation):
    """``optimisers``: five optimisers from one start on the narrow valley."""

    PHASES = ("valley", "first step", "race", "arrivals")

    def __init__(self):
        prob = LS.problem().scaled(LS.ILL_SCALE)
        self.runs = optimiser_runs()
        super().__init__("Optimisers on a narrow valley", prob, ILL_LO, ILL_HI, ILL_CAP, ILL_STEPS,
                         1.2)
        self.names = list(OPTIMISERS)
        self.arrival = {k: self.runs[k].first_within(self.Lmin, ARRIVED) for k in self.names}
        self.curv = np.linalg.eigvalsh(prob.hessian(ILL_START))
        # momentum's overshoot: the first dip, then the rise after it
        L = self.runs["momentum"].loss
        rise = np.nonzero(np.diff(L) > 0)[0]
        self.dip = int(rise[0]) if len(rise) else 0
        self.peak = self.dip + int(np.argmax(L[self.dip:self.dip + 40]))
        self.sgd_lr = OPTIMISERS["SGD"][1]["lr"]
        # ---- timeline
        Tm, t = {}, 0.3
        Tm["rise"] = (t, t + 3.0)
        t += 3.0 + 3.2
        Tm["first"] = (t, t + 7.5)
        t += 7.5 + 0.5
        d = self.sweep("race", ILL_STEPS, Tempo(slow=1.1, fast=0.13, ramp_in=5.0, ramp_out=4.0))
        Tm["race"] = (t, t + d)
        t += d + 0.5
        Tm["table"] = (t, t + 6.0)
        t += 6.0
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["first"][0], Tm["race"][0], Tm["table"][0]]

    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        rise = self.progress("rise", t)
        first = self.progress("first", t)
        pos = self.step_at("race", t)
        table = self.progress("table", t)
        self.landscape(sc, t, back, rise)
        self.minimum_marker(sc, smoothstep(0.6, 1.0, rise) * back)
        on = smoothstep(0.0, 0.3, first) * back
        chart = smoothstep(0.0, 1.0, first) * back
        self.chart_frame(sc, chart)
        k = min(int(np.floor(pos)), ILL_STEPS)
        for name in self.names:
            run, col = self.runs[name], OPT_COLORS[name]
            self.walker(sc, run, pos, col, on, ("opt", name), size=0.85)
            self.chart.curve(sc, run.loss, pos, col, chart, width=1.3)
            if pos < ILL_STEPS and (first > 0.25 or pos > 0):
                # this step's move; for the first step, drawn while it is explained
                j = min(k, ILL_STEPS - 1)
                step = self.land.step_vec(run.w[j], run.w[j + 1])
                self.land.arrow(sc, self.land.point(run.w[j], run.loss[j]), step, col,
                                (smoothstep(0.25, 0.5, first) if pos == 0 else 0.8) * back,
                                width=1.6)
        self.legend(sc, on, k, table)
        self.moments(sc, k, smoothstep(0.5, 0.8, first) * (1 - smoothstep(0.0, 0.3, table)) * back)
        self.subtitle = self.words(t, ph, first, pos, table)
        self.step_indicator(sc, ph, self.PHASES)

    LEGEND_Y = 0.625                # one row of names under the chart
    TABLE_Y = 0.70                  # the arrivals table / the moments panel

    def along(self, name):
        """How far along the valley (in w2) a run got from the start."""
        return float(ILL_START[1] - self.runs[name].w[-1, 1])

    def legend(self, sc, op, k, table):
        """The names in their colours in a row under the chart; at the end, a
        table of who arrived when."""
        op *= self._cap
        if op <= 0.01:
            return
        x = 0.60
        for name in self.names:
            col = _ink(OPT_COLORS[name])
            sc.text(x, self.LEGEND_Y, name, 0.027, "regular", col, op, "lm")
            x += layout.text_width(name, 0.027) + 0.018
        op *= smoothstep(0.35, 0.6, table)           # once the moments panel has gone
        if op <= 0.01:
            return
        sc.text(READ_X, self.TABLE_Y, "within %.2f of the minimum" % ARRIVED, 0.027, "regular",
                layout.STRONG_INK, op, "lm")
        col = READ_X + max(layout.text_width(n, 0.026) for n in self.names) + 0.022
        for n, name in enumerate(self.names):
            a = self.arrival[name]
            text = "from step %d" % a if a >= 0 else "not in %d steps" % ILL_STEPS
            yy = self.TABLE_Y + 0.04 + n * 0.036
            sc.text(READ_X, yy, name, 0.026, "regular", _ink(OPT_COLORS[name]), op, "lm")
            sc.text(col, yy, text, 0.026, "regular", layout.CAPTION_INK, op, "lm")

    def moments(self, sc, k, op):
        """Adam's running moments for the two weights as bars: the raw gradient
        ``|g|``, the first moment ``|m|`` and the root second moment ``sqrt(v)``
        (both bias-corrected), and the step they make."""
        op *= self._cap
        if op <= 0.01:
            return
        run = self.runs["Adam"]
        j = max(0, min(k, ILL_STEPS) - 1)
        t = j + 1
        g = np.abs(run.g[j])
        mh = np.abs(run.m[j]) / (1 - 0.9 ** t)
        vh = np.sqrt(run.v[j] / (1 - 0.999 ** t))
        cam = self.cam
        x0, x1 = READ_X + 0.03, 0.945
        scale = float(np.abs(run.g).max())            # the longest bar: the largest gradient
        rows = (("gradient g", g, LS.ARROW_GRAD), ("m̂", mh, (0.42, 0.44, 0.47)),
                ("√v̂", vh, (0.66, 0.68, 0.71)))
        sc.text(READ_X, self.TABLE_Y, "Adam's moments, step %d" % t, 0.027, "regular",
                layout.STRONG_INK, op, "lm")
        x = READ_X
        for label, _val, col in rows:                 # the legend row, in the bars' colours
            sc.text(x, self.TABLE_Y + 0.036, label, 0.025, "regular", _ink(col), op, "lm")
            x += layout.text_width(label, 0.025) + 0.02
        for i in range(2):
            yy = self.TABLE_Y + 0.075 + i * 0.07
            sc.text(READ_X, yy + 0.018, "w%s" % sub(i + 1), 0.026, "regular",
                    layout.CAPTION_INK, op, "lm")
            for r, (_label, val, col) in enumerate(rows):
                yb = yy + r * 0.018
                frac = min(1.0, val[i] / scale)
                a = cam.frac(*self.lay(x0, yb))
                b = cam.frac(*self.lay(x0 + (x1 - x0) * frac, yb))
                sc.line(a, b, width=4.5, color=col, opacity=0.9 * op, bias=0.0)
        step = run.w[j + 1] - run.w[j]
        sc.text(READ_X, self.TABLE_Y + 0.075 + 2 * 0.07 + 0.005, "step = η m̂ / √v̂ = %s"
                % wtxt(step), 0.026, "regular", _ink(OPT_COLORS["Adam"]), op, "lm")

    def words(self, t, ph, first, pos, table):
        """The subtitle for the moment."""
        runs = self.runs
        if ph == 0:
            if t < self.Tm["rise"][1] or t >= self.t_done + self.hold:
                return ("the same data with ink %s %d: the loss changes %d%s faster along w%s "
                        "than along w%s, and the bowl becomes a narrow valley"
                        % (TIMES, int(LS.ILL_SCALE), int(LS.ILL_SCALE ** 2), TIMES, sub(1),
                           sub(2)))
            return ("curvature at the start: %.1f across the valley, %.2f along it (ratio %d); "
                    "plain SGD must keep η below 2/%.1f = %.2f to be stable"
                    % (self.curv[1], self.curv[0], int(self.curv[1] / self.curv[0]),
                       self.curv[1], 2 / self.curv[1]))
        if ph == 1:
            g = runs["SGD"].g[0]
            if first < 0.5:
                return ("five optimisers from w = %s; the gradient is almost all across the "
                        "valley: ∂L/∂w = %s, and SGD's step (η = %g) follows it"
                        % (wtxt(ILL_START), wtxt(g), self.sgd_lr))
            s = runs["Adam"].w[1] - runs["Adam"].w[0]
            return ("RMSProp and Adam divide each weight's step by the running size of its "
                    "gradient, so both weights move at once: Adam's first step is %s, η in each"
                    % wtxt(s))
        if ph == 2:
            k = min(int(pos), ILL_STEPS)
            if k < self.dip + 2:
                return ("SGD crosses the valley in one step (loss %.1f to %.2f), then creeps along "
                        "it; momentum (lr %g, β = 0.9) gathers speed"
                        % (runs["SGD"].loss[0], runs["SGD"].loss[1],
                           OPTIMISERS["momentum"][1]["lr"]))
            if k < self.peak + 12:
                return ("momentum overshoots: it ran down to loss %.2f at step %d, then up the "
                        "far wall to %.2f at step %d before turning back"
                        % (runs["momentum"].loss[self.dip], self.dip,
                           runs["momentum"].loss[self.peak], self.peak))
            if k < max(self.arrival["RMSProp"], 1) + 8:
                return ("the adaptive steps shrink as the gradients settle (√v̂ remembers "
                        "their size); step %d: RMSProp %.3f, Adam %.3f, SGD %.3f"
                        % (k, runs["RMSProp"].loss[k], runs["Adam"].loss[k], runs["SGD"].loss[k]))
            return ("RMSProp is within %.2f of the minimum from step %d; Adam's momentum makes it "
                    "overshoot too, AdamW's decay (%g) holds it %.1f short (step %d)"
                    % (ARRIVED, self.arrival["RMSProp"], OPTIMISERS["AdamW"][1]["weight_decay"],
                       np.linalg.norm(runs["AdamW"].w[-1] - self.wmin), k))
        return ("after %d steps: RMSProp arrived at step %d, AdamW at %d (its decay holds it %.1f "
                "short), Adam at %d; momentum is %.1f%s further along the valley than SGD, "
                "neither is there yet"
                % (ILL_STEPS, self.arrival["RMSProp"], self.arrival["AdamW"],
                   np.linalg.norm(runs["AdamW"].w[-1] - self.wmin), self.arrival["Adam"],
                   self.along("momentum") / self.along("SGD"), TIMES))

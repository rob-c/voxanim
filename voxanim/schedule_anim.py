"""Learning-rate schedules: four of them training the MNIST MLP, and the
same four driving Adam on the two-weight landscape, side by side.

The schedules (:func:`voxanim.mlp_runs.schedule_fn`): **constant**
``1e-3``; **step decay**, divided by 10 after each third of training;
**cosine with warmup**, a 5% linear warmup then a cosine down to 0; **one
cycle**, up to ``3e-3`` by 30% of training then a cosine down to ``3e-5``.
The MLP (784-64-32-10, Adam, 3 epochs of 128-image minibatches, 1407 steps)
is trained once with each; its minibatch loss (smoothed over 50 steps) is
drawn as training runs.  Beside it, Adam walks the loss landscape of
:mod:`voxanim.landscape` with the same four schedules squeezed into 150
steps (peak 0.1, or 0.3 for one cycle), the walkers synchronised to the
same fraction of training; they finish in the same order as the MLP runs.

Then warmup: Adam's first step is exactly ``eta`` in every weight, whatever
the gradient, because its bias-corrected ``m / sqrt(v)`` is ``+-1`` at step 1,
and the next few are nearly as long while ``v`` is still an estimate from a
handful of minibatches.  Two more walkers start together: Adam at a
constant 0.3 and Adam warming up to 0.3 over 8 steps; and the MLP numbers
for the same pair (``3e-3``, with and without a 70-step warmup) say how far
the weights moved in the first 20 steps.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from . import color as C
from . import landscape as LS
from . import layout
from . import mlp_runs as R
from .conv import Tempo
from .easing import ease_in_out, remap, smoothstep
from .optim import Adam
from .optim_anim import BOX_CAP, BOX_HI, BOX_LO, START, LandscapeAnimation, _ink, wtxt
from .text import TIMES

SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sci(v):
    """``3e-4`` as ``3×10⁻⁴`` (one significant digit)."""
    e = int(np.floor(np.log10(v)))
    m = v / 10 ** e
    mant = "" if abs(m - 1) < 1e-9 else "%g%s" % (round(m, 2), TIMES)
    return "%s10%s" % (mant, str(e).translate(SUPERSCRIPT))

EPOCHS, PEAK, ONECYCLE_PEAK = 3, 1e-3, 3e-3
SCHEDULES = (("constant", PEAK), ("step", PEAK), ("cosine", PEAK), ("onecycle", ONECYCLE_PEAK))
NAMES = dict(constant="constant", step="step decay", cosine="cosine + warmup", onecycle="one cycle")
COLORS = dict(constant=C.rgb("#d84e4e"), step=C.rgb("#e0952e"), cosine=C.rgb("#3d9e45"),
              onecycle=C.rgb("#1f6fb5"))
LAND_PEAK, LAND_ONECYCLE, LAND_STEPS = 0.1, 0.3, 150      # the landscape's version
WARM_PEAK = 0.3                                           # the warmup pair on the landscape
WARM_MLP_PEAK = 3e-3                                      # ... and on the MLP (one epoch each)
WARM_COLORS = (C.rgb("#c0399b"), C.rgb("#3d9e45"))        # no warmup, warmup
EARLY = 20                                                # steps the warmup comparison counts
SMOOTH = 50                                               # steps the drawn MLP loss averages over
LR_CHART = ((0.715, 0.945), (0.27, 0.42))
LOSS_CHART = ((0.715, 0.945), (0.52, 0.76))
LEGEND_Y = 0.845                # the legend: two rows of two names
WARM_X = 0.60                   # frame x of the warmup read-outs (under the rate chart)
LR_RANGE = (5e-6, 4e-3)         # the rate chart's (log) range; the cosine's tail is clipped


def mlp_runs():
    """``{schedule: MLPRun}`` of the four schedules."""
    out = {}
    for kind, peak in SCHEDULES:
        steps = EPOCHS * 469
        out[kind] = R.train("sched_" + kind, dict(optimizer="adam", schedule=kind, peak=peak),
                            partial(Adam, lr=peak), lr_at=R.schedule_fn(kind, steps, peak),
                            epochs=EPOCHS)
    return out


def warmup_runs():
    """The MLP for one epoch at ``3e-3``: ``("constant", "cosine")`` = without and
    with warmup."""
    steps = EPOCHS * 469           # the warmup is 5% of the full schedule: 70 steps
    return tuple(R.train(name, dict(optimizer="adam", schedule=kind, peak=WARM_MLP_PEAK),
                         partial(Adam, lr=WARM_MLP_PEAK),
                         lr_at=R.schedule_fn(kind, steps, WARM_MLP_PEAK), epochs=1)
                 for name, kind in (("const_hi", "constant"), ("warm_hi", "cosine")))


def landscape_runs():
    """Adam on the round bowl with the four schedules, plus the warmup pair."""
    settings = dict(problem="mnist-0v1-ink-width", start=START, steps=LAND_STEPS, optimizer="adam",
                    peak=LAND_PEAK, onecycle_peak=LAND_ONECYCLE, warm_peak=WARM_PEAK,
                    schedules=[k for k, _ in SCHEDULES])

    def build():
        prob = LS.problem()
        out = {}
        for kind, _peak in SCHEDULES:
            peak = LAND_ONECYCLE if kind == "onecycle" else LAND_PEAK
            out[kind] = LS.descend(prob, partial(Adam, lr=peak), START, LAND_STEPS,
                                   lr_at=R.schedule_fn(kind, LAND_STEPS, peak))
        for key, kind in (("nowarm", "constant"), ("warm", "cosine")):
            out[key] = LS.descend(prob, partial(Adam, lr=WARM_PEAK), START, LAND_STEPS,
                                  lr_at=R.schedule_fn(kind, LAND_STEPS, WARM_PEAK))
        return out
    return LS.run_cache("landscape_schedules", settings, build)


class LearningRateSchedules(LandscapeAnimation):
    """``learning-rate-schedule``: the schedules, the MLP and the landscape
    training in step, then warmup."""

    PHASES = ("schedules", "training", "warmup", "results")

    def __init__(self):
        prob = LS.problem()
        self.mlp = mlp_runs()
        self.warm = warmup_runs()
        self.land_runs = landscape_runs()
        self.steps = self.mlp["constant"].steps
        super().__init__("Learning-rate schedules", prob, BOX_LO, BOX_HI, BOX_CAP, LAND_STEPS,
                         float(self.land_runs["constant"].loss[0]) + 0.1)
        self.chart.xs, self.chart.ys = LOSS_CHART            # the landscape chart is not drawn
        self.curves = {k: R.smooth(r.losses, SMOOTH) for k, r in self.mlp.items()}
        self.lr_chart = LS.LossChart(self, LR_CHART, self.steps, *LR_RANGE, log=True)
        lo = min(c.min() for c in self.curves.values())
        hi = max(c.max() for c in self.curves.values())
        self.loss_chart = LS.LossChart(self, LOSS_CHART, self.steps, lo * 0.85, hi * 1.1, log=True)
        # the warmup pair's numbers
        self.moved = tuple(float(np.sqrt((r.norms[EARLY] ** 2).sum()) -
                                 np.sqrt((r.norms[0] ** 2).sum())) for r in self.warm)
        self.first = {k: self.land_runs[k].w[1] - self.land_runs[k].w[0]
                      for k in ("nowarm", "warm")}
        # ---- timeline
        Tm, t = {}, 0.3
        Tm["rise"] = (t, t + 2.5)
        Tm["sched"] = (t + 1.0, t + 10.0)                   # the four schedules, one by one
        t += 10.0 + 0.5
        Tm["train"] = (t, t + 22.0)
        t += 22.0 + 2.5
        d = self.sweep("early", EARLY, Tempo(slow=0.9, fast=0.25, ramp_in=4.0, ramp_out=2.0))
        Tm["early"] = (t, t + d)
        t += d + 3.0
        Tm["results"] = (t, t + 5.0)
        t += 5.0
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["train"][0], Tm["early"][0] - 1.0, Tm["results"][0]]

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        rise = self.progress("rise", t)
        sched = self.progress("sched", t)
        train = self.progress("train", t)
        early = self.step_at("early", t)
        warm_on = smoothstep(self.marks[2], self.marks[2] + 1.0, t)
        results = self.progress("results", t)
        self.landscape(sc, t, back, rise)
        self.minimum_marker(sc, smoothstep(0.6, 1.0, rise) * back)
        self.schedule_chart(sc, sched, (1 - smoothstep(0.0, 0.3, results)) * back)
        # the training block leaves before the warmup block arrives
        self.training(sc, train, (1 - smoothstep(0.0, 0.45, warm_on)) * back)
        self.warmup(sc, early, smoothstep(0.55, 1.0, warm_on) * back)
        self.legend(sc, smoothstep(0.0, 0.3, sched) * back, warm_on)
        self.table(sc, smoothstep(0.3, 0.7, results) * back)
        self.subtitle = self.words(t, ph, sched, train, early, results)
        self.step_indicator(sc, ph, self.PHASES)

    def schedule_chart(self, sc, sched, op):
        """The four schedules drawn one after another (log scale)."""
        op *= smoothstep(0.0, 0.1, sched)
        if op <= 0.01:
            return
        self.lr_chart.frame(sc, op, xlabel="step", ylabel="learning rate")
        for n, (kind, _peak) in enumerate(SCHEDULES):
            u = remap(sched, n / 4, (n + 0.85) / 4)
            self.lr_chart.curve(sc, self.mlp[kind].lrs, self.steps * u, COLORS[kind], op,
                                width=1.4, marker=u < 1)
        for v, text in ((1e-3, "10⁻³"), (1e-4, "10⁻⁴"), (1e-5, "10⁻⁵")):
            self.lr_chart.label(sc, 0, v, text, layout.NOTE_INK, op, dx=-0.006, anchor="rm",
                                size=0.024)

    def training(self, sc, train, op):
        """The MLP loss curves and the landscape walkers, in step."""
        op *= smoothstep(0.0, 0.05, train)
        if op <= 0.01:
            return
        u = smoothstep(0.0, 1.0, train) if train < 1 else 1.0
        self.loss_chart.frame(sc, op, xlabel="step", ylabel="training loss (smoothed)")
        for v, text in ((1.0, "1"), (0.3, "0.3"), (0.1, "0.1")):
            self.loss_chart.label(sc, 0, v, text, layout.NOTE_INK, op, dx=-0.006, anchor="rm",
                                  size=0.024)
        for kind, _peak in SCHEDULES:
            col = COLORS[kind]
            self.loss_chart.curve(sc, self.curves[kind], self.steps * u, col, op, width=1.4)
            self.walker(sc, self.land_runs[kind], LAND_STEPS * u, col, op, ("sched", kind),
                        size=0.85)
        k = int(u * self.steps)
        self.readout(sc, ["step %d of %d, epoch %d" % (k, self.steps, min(EPOCHS, k // 469 + 1))],
                     op, y=0.79)

    def warmup(self, sc, early, op):
        """Adam at 0.3 with and without warmup: the first steps on the landscape,
        and the MLP's numbers."""
        if op <= 0.01:
            return
        for key, col in (("nowarm", WARM_COLORS[0]), ("warm", WARM_COLORS[1])):
            run = self.land_runs[key]
            self.walker(sc, run, early, col, op, ("warm", key), size=0.9)
            k = min(int(np.floor(early)), LAND_STEPS - 1)
            if early < EARLY:
                step = self.land.step_vec(run.w[k], run.w[k + 1])
                self.land.arrow(sc, self.land.point(run.w[k], run.loss[k]), step, col, 0.9 * op,
                                width=1.6)
        k = min(int(early), EARLY)
        lines = ["step %d" % k,
                 "η = %.3f  vs  %.3f with warmup"
                 % (self.land_runs["nowarm"].lr[min(k, LAND_STEPS - 1)],
                    self.land_runs["warm"].lr[min(k, LAND_STEPS - 1)]),
                 "moved %.2f  vs  %.2f"
                 % (self.path_len("nowarm", k), self.path_len("warm", k))]
        self.readout(sc, lines, op, y=0.50, dy=0.04, x=WARM_X)
        mlp = ["the MLP at η = %s, first %d steps:" % (sci(WARM_MLP_PEAK), EARLY),
               "weights move %.2f without warmup" % self.moved[0],
               "%.2f with a %d-step warmup" % (self.moved[1], int(round(0.05 * self.steps)))]
        self.readout(sc, mlp, op * smoothstep(EARLY - 2, EARLY, early), y=0.66, dy=0.04,
                     ink=layout.CAPTION_INK, x=WARM_X)

    def path_len(self, key, k):
        """Length of the first ``k`` steps of a landscape run."""
        w = self.land_runs[key].w[:k + 1]
        return float(np.linalg.norm(np.diff(w, axis=0), axis=1).sum()) if k > 0 else 0.0

    def legend(self, sc, op, warm_on):
        """The schedules' names in their colours (the warmup pair's during warmup)."""
        op *= self._cap
        if op <= 0.01:
            return
        items = [(NAMES[k], COLORS[k]) for k, _ in SCHEDULES]
        if warm_on > 0.5:
            items = [("Adam, η = %g" % WARM_PEAK, WARM_COLORS[0]),
                     ("the same with warmup", WARM_COLORS[1])]
        for n, (name, col) in enumerate(items):
            x = 0.60 if n % 2 == 0 else 0.60 + 0.5 * (0.945 - 0.60)
            sc.text(x, LEGEND_Y + 0.04 * (n // 2), name, 0.027, "regular", _ink(col), op, "lm")

    def table(self, sc, op):
        """Final loss and test accuracy of the four MLP runs."""
        op *= self._cap
        if op <= 0.01:
            return
        x, y = 0.60, 0.27
        sc.text(x, y, "after %d epochs" % EPOCHS, 0.027, "regular", layout.STRONG_INK, op, "lm")
        sc.text(0.945, y, "loss   accuracy", 0.025, "regular", layout.NOTE_INK, op, "rm")
        order = sorted(SCHEDULES, key=lambda s: self.curves[s[0]][-1])
        for n, (kind, _peak) in enumerate(order):
            yy = y + 0.04 + n * 0.036
            sc.text(x, yy, NAMES[kind], 0.026, "regular", _ink(COLORS[kind]), op, "lm")
            sc.text(0.945, yy, "%.3f   %.1f%%" % (self.curves[kind][-1], 100 * self.mlp[kind].acc),
                    0.026, "mono", layout.CAPTION_INK, op, "rm")

    def words(self, t, ph, sched, train, early, results):
        """The subtitle for the moment."""
        c = self.curves
        if ph == 0:
            n = min(3, int(sched * 4))
            if sched <= 0 or t >= self.t_done + self.hold:
                return ("the learning rate need not stay put: four schedules for Adam training "
                        "the MNIST MLP (%d steps), and the same four on the two-weight landscape"
                        % self.steps)
            return [
                "constant: η = %s throughout" % sci(PEAK),
                "step decay: η = %s, divided by 10 after each third of training" % sci(PEAK),
                "cosine with warmup: a linear warmup over the first 5%% (%d steps) to %s, then a "
                "cosine down to 0" % (int(round(0.05 * self.steps)), sci(PEAK)),
                "one cycle: from %s up to %s at 30%% of training, then a cosine down to %s"
                % (sci(ONECYCLE_PEAK / 10), sci(ONECYCLE_PEAK), sci(ONECYCLE_PEAK / 100)),
            ][n]
        if ph == 1:
            k = int(smoothstep(0.0, 1.0, train) * self.steps)
            if train < 0.35:
                return ("training, in step: the MLP's loss on the right, Adam on the landscape "
                        "with the same schedules (peak 0.1, one cycle 0.3) on the left")
            if train < 1:
                return ("step %d: constant %.3f, step decay %.3f, cosine %.3f, one cycle %.3f; "
                        "the high one-cycle rate pays off once it comes down"
                        % (k, c["constant"][k - 1], c["step"][k - 1], c["cosine"][k - 1],
                           c["onecycle"][k - 1]))
            return ("after %d epochs: one cycle %.3f, constant %.3f, cosine %.3f, step decay "
                    "%.3f; dividing by 10 after a third was too early, and the landscape walkers "
                    "finish in the same order"
                    % (EPOCHS, c["onecycle"][-1], c["constant"][-1], c["cosine"][-1],
                       c["step"][-1]))
        if ph == 2:
            if early < 1:
                return ("warmup: Adam's first step is exactly η in every weight, whatever the "
                        "gradient (m/√v = ±1 at step 1): at η = %g that is %s"
                        % (WARM_PEAK, wtxt(self.first["nowarm"])))
            return ("the next steps are nearly as long while √v is still a guess from a few "
                    "gradients; with warmup the first rate is %.4f and the first step %s: small "
                    "until √v has settled"
                    % (self.land_runs["warm"].lr[0], wtxt(self.first["warm"])))
        return ("on the MLP (η = %s): in %d steps the weights move %.2f in norm without warmup "
                "and %.2f with; %s and %s after one epoch: the leap did no lasting harm here"
                % (sci(WARM_MLP_PEAK), EARLY, self.moved[0], self.moved[1],
                   "%.1f%%" % (100 * self.warm[0].acc), "%.1f%%" % (100 * self.warm[1].acc)))

"""Trimming a trained network: pruning by magnitude, the lottery ticket,
structured pruning and distillation -- four animations on the MNIST models of
:mod:`voxanim.mnist`, with every number computed by :mod:`voxanim.prune`.

**magnitude-pruning** -- the trained 784-64-32-10 MLP as sheets (six of layer
1's 64 templates, the 64 x 32 and 32 x 10 matrices) beside the histogram of
|w| over all 52,544 weights.  A threshold sweeps up the histogram and every
weight below it goes dark; at 50, 80, 90, 95 and 99% pruned the test accuracy
is read out without retraining (it collapses past 80%), then after 400 Adam
steps with the mask held (90% pruned recovers to 96%), then for iterative
pruning (prune, fine-tune, prune further), which keeps far more at 99%.

**lottery-ticket** -- the 20%-dense mask from pruning a trained network is
laid over that network's *original initialisation* and the sparse network is
retrained, against a random mask of the same per-layer density and the dense
network, all from the same initial weights on the same minibatches.  The
three test-error curves, then a second seed's; the subtitle reports whether
the ticket won.

**structured-pruning** -- the CNN's four filters, then the MLP's 64 hidden
units, ranked by activation energy (mean squared activation over the test
set) and removed one at a time, least active first: the feature maps and the
weight rows disappear, the accuracy is charted against the number removed,
with and without a brief fine-tune after each removal, and what is left after
half are gone is shown.

**distillation** -- a 784-256-256-10 teacher's output for one 3 as bars, at
temperature 1 and softened at temperature 4 (the "dark knowledge" in the wrong
classes: this 3 looks like an 8), then a 784-32-10 student trained three ways
on 3,000 images: hard labels, the teacher's soft targets, and both; the three
accuracy curves and the final numbers.

Every quantity is a pure function of time; the sheets, charts and captions
follow the house style of :mod:`voxanim.mlp` and :mod:`voxanim.norm_anim`.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import color as C
from . import layout
from .conv import Tempo
from .easing import ease_in_out, ease_out, remap, smoothstep, soft_band
from .losschart import LossChart
from .mnist import CNN_FILTERS, CONV_HW, SIZES, _cnn_forward, _cnn_params
from .netanim import (
    ACT_HI,
    ACT_LO,
    INK_HI,
    INK_LO,
    OFF,
    NetAnimation,
    amax,
    column_pitch,
    div_colors,
    lit_colors,
)
from .prune import (
    FILTER_FINETUNE,
    FINETUNE_STEPS,
    STUDENT_SIZES,
    TEACHER_SIZES,
    TEMPERATURE,
    TICKET_DENSITY,
    TRANSFER_N,
    UNIT_FINETUNE,
    all_weights,
    softmax,
    weight_keys,
)
from .sweep import Sweep
from .tensor import Block
from .text import TIMES

COLOR_QUANTILE = 0.995        # |w| at this quantile saturates a layer's colour scale
HIST_INK = C.rgb("#8a8f94")   # histogram bars (kept weights)
TUNED_INK = tuple(float(v) for v in C.INPUT)             # fine-tuned curves (teal)
PRUNED_INK = (0.75, 0.22, 0.22)                           # pruned, no retraining (red)
ITER_INK = tuple(float(v) for v in C.rgb("#8952eb"))      # iterative pruning (violet)
THRESHOLD_INK = (0.95, 0.95, 0.95)
PERCENT_TICKS = ((0.0, "0%"), (0.5, "50%"), (1.0, "100%"))


class Box:
    """A world box ``(lo, hi)`` that captions accept wherever they take a sheet
    (:meth:`bbox`) or a ``(lo, hi)`` pair (unpacking), so one caption can sit
    beside several sheets at once."""

    def __init__(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)

    def bbox(self, offsets=True):
        return self.lo, self.hi

    def __iter__(self):
        return iter((self.lo, self.hi))


def around(sheets):
    """The :class:`Box` enclosing ``sheets``."""
    los, his = zip(*[s.bbox() for s in sheets], strict=True)
    return Box(np.min(los, 0), np.max(his, 0))


def color_scale(W):
    """A layer's colour-scale maximum: the :data:`COLOR_QUANTILE` of ``|W|``
    (the odd large weight would otherwise wash the rest out)."""
    return float(np.quantile(np.abs(W), COLOR_QUANTILE)) or 1.0


def template_grid(anim, at, scale, rows, cols, gap=1.5, name="tpl"):
    """``rows x cols`` 28 x 28 sheets stacked in world space around layout
    point ``at`` (row-major), each ``gap`` cells apart."""
    sheets = []
    pitch = (28 + gap) * scale
    centre = anim.cam.frac(*at)
    for r in range(rows):
        for c in range(cols):
            s = anim.sheet((28, 28), at, scale, "%s%d" % (name, r * cols + c))
            s.block.place(centre + np.array([(c - (cols - 1) / 2) * pitch, 0.0,
                                             ((rows - 1) / 2 - r) * pitch]))
            sheets.append(s)
    return sheets


def dots(sc, place, chart, xs, ys, upto, ink, op, size=0.55):
    """A glowing voxel at each measured point ``(x, y)`` with ``x <= upto``."""
    for x, y in zip(xs, ys, strict=True):
        if x <= upto + 1e-9:
            sc.voxel(place(chart.fx(x), chart.fy(y)), ink, op, glow=0.3, shadow=False,
                     reflect=False, size=size)


def level_axis(f, levels):
    """Chart coordinate of a pruned fraction ``f``: the dense network at 0 and
    each of ``levels`` one unit further (the levels are spaced evenly; the
    tick labels say what each is), interpolated in between."""
    return float(np.interp(f, [0.0] + list(levels), np.arange(len(levels) + 1)))


def pct(v, digits=0):
    """``v`` in [0, 1] as a percentage string."""
    return "%.*f%%" % (digits, 100 * v)


# ================================================================== magnitude pruning

MP_PPU = 9.0
MP_TEMPLATES = (0, 1, 2, 3, 4, 5)                   # layer-1 neurons shown as templates
MP_TPL_AT, MP_TPL_SCALE = (0.20, 0.615), 0.45      # the 2 x 3 grid of templates
MP_AXIS_PAD = 0.25                                 # chart x before the dense point (levels)
MP_W1_AT, MP_W2_AT, MP_MAT_SCALE = (0.43, 0.615), (0.565, 0.615), 0.42
MP_HIST_AT = (0.79, 0.345)
HBINS, HROWS, HRANGE, HFULL = 20, 12, 0.4, 0.21    # bins over (0, 0.4); a bin of 21% fills
HSCALE, HPITCH = 0.85, (1.0, 1.15)
MP_CHART = ((0.66, 0.935), (0.61, 0.86))          # ((x0, x1), (top, bottom)), layout
MP_LEGEND = (0.67, 0.71)
MP_SHOW_LEVEL = 2                                  # the fine-tune the sheets show: 90% pruned
MP_LEVEL_SLOT = 3.0                                # seconds per level in the prune phase
MP_MOVE = 1.1                                      # seconds a threshold move takes
MP_TUNE = 3.5                                      # seconds the fine-tune crossfade takes
MP_CURVE = 5.0                                     # seconds a chart curve takes to draw


class MagnitudePruning(NetAnimation):
    """Pruning the trained MLP by weight magnitude (see the module docstring).

    ``sweep`` is :func:`voxanim.prune.prune_sweep`'s dict, ``P`` the trained
    network as ``{"W0", "b0", ...}``.
    """

    PHASES = ("trained", "threshold", "prune", "fine-tune", "iterative")

    def __init__(self, sweep, P):
        super().__init__("Pruning by magnitude", "", MP_PPU)
        self.hold = 2.5
        self.d, self.P = sweep, P
        self.levels = [float(v) for v in sweep["levels"]]
        self.total = int(sweep["total"])
        self.sorted_abs = np.sort(np.abs(all_weights(P)))
        self.acc_dense = float(sweep["acc_dense"])
        self.acc = {k: [float(v) for v in sweep["acc_" + k]] for k in ("pruned", "tuned", "iter")}
        self.tuned = {k: sweep["tuned%d_%s" % (MP_SHOW_LEVEL, k)] for k in weight_keys(P)}
        last = len(self.levels) - 1
        self.iter = {k: sweep["iter%d_%s" % (last, k)] for k in weight_keys(P)}
        self.vmax = {k: color_scale(P[k]) for k in weight_keys(P)}
        self.hist_frac = np.histogram(self.sorted_abs, HBINS, (0, HRANGE))[0] / self.total

        # ---- geometry
        self.tpls = template_grid(self, MP_TPL_AT, MP_TPL_SCALE, 2, 3)
        self.W1 = self.sheet(SIZES[1:3], MP_W1_AT, MP_MAT_SCALE, "W1")
        self.W2 = self.sheet(SIZES[2:4], MP_W2_AT, MP_MAT_SCALE, "W2")
        self.hist = self.sheet((HROWS, HBINS), MP_HIST_AT, HSCALE, "hist", pitch=HPITCH)
        self.chart = LossChart(MP_CHART, (-MP_AXIS_PAD, len(self.levels)), (0.0, 1.0))
        self.xs = [level_axis(f, self.levels) for f in [0.0] + self.levels]

        # ---- timeline
        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("trained", 5.0)
        add("sweep", 5.5, 0.5)
        add("prune", MP_LEVEL_SLOT * (len(self.levels) - 1), 0.5)
        add("tune", MP_MOVE + MP_TUNE + MP_CURVE, 0.5)
        add("iter", MP_MOVE + MP_CURVE)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["sweep"][0], Tm["prune"][0], Tm["tune"][0], Tm["iter"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ the schedule
    def fraction(self, t):
        """Fraction of the weights below the threshold at time ``t``."""
        Tm, L = self.Tm, self.levels
        if t < Tm["sweep"][0]:
            return 0.0
        if t < Tm["prune"][0]:
            return L[0] * ease_in_out(self.progress("sweep", t))
        if t < Tm["tune"][0]:
            u = (t - Tm["prune"][0]) / MP_LEVEL_SLOT
            n = min(int(u), len(L) - 2)
            move = ease_in_out(remap(u - n, 0.0, MP_MOVE / MP_LEVEL_SLOT))
            return L[n] + (L[n + 1] - L[n]) * move
        if t < Tm["iter"][0]:
            u = ease_in_out(remap(t, Tm["tune"][0], Tm["tune"][0] + MP_MOVE))
            return L[-1] + (L[MP_SHOW_LEVEL] - L[-1]) * u
        u = ease_in_out(remap(t, Tm["iter"][0], Tm["iter"][0] + MP_MOVE))
        return L[MP_SHOW_LEVEL] + (L[-1] - L[MP_SHOW_LEVEL]) * u

    def threshold(self, f):
        """The ``|w|`` below which a fraction ``f`` of the weights lie."""
        k = int(round(f * self.total))
        return float(self.sorted_abs[min(k, self.total - 1)]) if k > 0 else 0.0

    def tune_mix(self, t):
        """0..1: how far the sheets have crossfaded to the fine-tuned weights."""
        t0 = self.Tm["tune"][0] + MP_MOVE
        return ease_in_out(remap(t, t0, t0 + MP_TUNE))

    def iter_mix(self, t):
        """0..1: how far the sheets have crossfaded to the iterative chain's weights."""
        t0 = self.Tm["iter"][0]
        return ease_in_out(remap(t, t0, t0 + MP_MOVE))

    def shown(self, key, t):
        """``(values, mask)`` the sheet of ``key`` draws at time ``t``."""
        W = self.P[key]
        thr = self.threshold(self.fraction(t))
        mask = np.abs(W) >= thr if thr > 0 else np.ones(W.shape, bool)
        mix = self.tune_mix(t)
        values = W + (self.tuned[key] - W) * mix
        im = self.iter_mix(t)
        if im > 0:
            final = self.iter[key]
            values = values + (final - values) * im
            mask = final != 0 if im >= 1 else mask
        return values, mask

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        acc = self.acc
        self.subtitle = [
            "the trained 784-64-32-10 MLP: %s weights, %s on the test set; which of them "
            "matter?" % (f"{self.total:,}", pct(self.acc_dense, 1)),
            "a threshold on |w| sweeps up the histogram: every weight below it is set to 0 "
            "(the voxel goes dark)",
            "prune the smallest %s: without retraining the accuracy falls to %s"
            % (", ".join(pct(f) for f in self.levels[1:]),
               ", ".join(pct(a) for a in acc["pruned"][1:])),
            "fine-tune %d Adam steps with the mask held (pruned weights stay 0): %s pruned, "
            "%s back to %s" % (FINETUNE_STEPS, pct(self.levels[MP_SHOW_LEVEL]),
                                    pct(acc["pruned"][MP_SHOW_LEVEL]),
                                    pct(acc["tuned"][MP_SHOW_LEVEL])),
            "iterative: prune %s, fine-tune, prune to %s, fine-tune, %s: at %s pruned %s, "
            "against %s in one shot" % (pct(self.levels[0]), pct(self.levels[1]), "…",
                                        pct(self.levels[-1]), pct(acc["iter"][-1]),
                                        pct(acc["tuned"][-1])),
        ][ph]
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        intro = smoothstep(0.0, 0.6, t)
        op = intro * back
        self._sheets(sc, t, op)
        self._histogram(sc, t, op)
        self._chart(sc, t, op)
        self.step_indicator(sc, ph, self.PHASES)

    def _sheets(self, sc, t, op):
        """The templates and the two matrices, pruned weights dark."""
        W0, m0 = self.shown("W0", t)
        glint = 0.35 * soft_band(self.tune_mix(t), 0.05, 0.95, 0.05) * (1 - self.iter_mix(t))
        for s, j in zip(self.tpls, MP_TEMPLATES, strict=True):
            s.fill(div_colors(W0[:, j], self.vmax["W0"], mask=m0[:, j]))
            s.block.white[:, :, 0] = glint * m0[:, j].reshape(28, 28)
            s.cage = op
            s.fade(op)
        for s, key in ((self.W1, "W1"), (self.W2, "W2")):
            W, m = self.shown(key, t)
            s.fill(div_colors(W.ravel(), self.vmax[key], mask=m.ravel()))
            s.block.white[:, :, 0] = glint * m
            s.cage = op
            s.fade(op)
        self.caption(sc, around(self.tpls), "layer 1: %d of %d templates (%d weights each)"
                     % (len(MP_TEMPLATES), SIZES[1], SIZES[0]), op, above=True, max_width=0.4)
        self.caption(sc, self.W1, "layer 2  %d%s%d" % (SIZES[1], TIMES, SIZES[2]), op)
        self.caption(sc, self.W2, "layer 3  %d%s%d" % (SIZES[2], TIMES, SIZES[3]), op, above=True)
        im = self.iter_mix(t)
        if im > 0:
            kept = sum(int((self.iter[k] != 0).sum()) for k in weight_keys(self.P))
            self.caption(sc, around([self.W1, self.W2]), "what is left after iterative pruning "
                         "to %s: %s weights" % (pct(self.levels[-1]), f"{kept:,}"), op * im,
                         max_width=0.3, offset=0.045)

    def _histogram(self, sc, t, op):
        """|w| binned, the threshold line, the count below it."""
        f = self.fraction(t)
        thr = self.threshold(f)
        if self.iter_mix(t) >= 1:
            thr = self.threshold(self.levels[-1])
        present = np.zeros((HROWS, HBINS), bool)
        for b in range(HBINS):
            h = int(round(min(1.0, self.hist_frac[b] / HFULL) * HROWS))
            present[HROWS - h:, b] = True
        edges = np.linspace(0, HRANGE, HBINS + 1)
        below = (edges[1:] <= thr)                       # bins wholly under the threshold
        cols = np.where(below[:, None], OFF[None], HIST_INK[None])
        self.hist.fill(np.tile(cols, (HROWS, 1)))
        self.hist.show(present.ravel())
        self.hist.cage = op
        self.hist.fade(op)
        self.caption(sc, self.hist, "|w| of all %s weights" % f"{self.total:,}", op, above=True)
        lo, hi = self.hist.block.box((HROWS - 1, HROWS), (0, HBINS), (0, 1))
        for x, text, anchor in ((lo[0], "0", "rm"), (hi[0], "%g" % HRANGE, "lm")):
            p = self.cam.project([x, hi[1], (lo[2] + hi[2]) / 2])
            sc.text((p[0] + (-8 if anchor == "rm" else 8)) / self.cam.width,
                    p[1] / self.cam.height, text, layout.MIN_TEXT, "regular", layout.NOTE_INK,
                    op * self._cap, anchor)
        if thr > 0:
            # the threshold: a line up the histogram at |w| = thr
            u = min(thr / HRANGE, 1.0)
            x = lo[0] + (hi[0] - lo[0]) * u
            top = self.hist.block.box((0, 1), (0, 1), (0, 1))[1][2]
            y = (lo[1] + hi[1]) / 2
            sc.line([x, y, lo[2] - 0.4 * HSCALE], [x, y, top + 0.6 * HSCALE], width=1.6,
                    color=THRESHOLD_INK, opacity=0.95 * op, bias=0.3)
            k = int(round(f * self.total))
            if self.iter_mix(t) >= 1:
                k = self.total - sum(int((self.iter[c] != 0).sum()) for c in weight_keys(self.P))
            self.caption(sc, self.hist, "threshold |w| < %.3f" % thr, op, max_width=0.3)
            self.caption(sc, self.hist, "pruned %s of %s (%s)"
                         % (f"{k:,}", f"{self.total:,}", pct(k / self.total)), op, max_width=0.3,
                         offset=0.044)

    def _chart(self, sc, t, op):
        """Accuracy against the weights kept: pruned, fine-tuned, iterative."""
        Tm = self.Tm
        op *= smoothstep(Tm["sweep"][0] + 1.0, Tm["sweep"][0] + 2.0, t) * self._cap
        if op <= 0.01:
            return
        cam, place, ch = self.cam, self.cam.frac, self.chart
        ticks = [(0.0, "all")] + [(x, pct(1 - f)) for x, f in zip(self.xs[1:], self.levels,
                                                                strict=True)]
        ch.axes(sc, cam, place, op, "weights kept", "test accuracy", ticks, PERCENT_TICKS)
        upto = level_axis(self.fraction(t), self.levels) if t < Tm["tune"][0] else self.xs[-1]
        curves = [("no retraining", "pruned", PRUNED_INK, upto)]
        if t >= Tm["tune"][0]:
            t0 = Tm["tune"][0] + MP_MOVE + MP_TUNE
            curves.append(("fine-tuned", "tuned", TUNED_INK,
                           self.xs[-1] * smoothstep(0.0, 1.0, remap(t, t0, t0 + MP_CURVE))))
        if t >= Tm["iter"][0]:
            t0 = Tm["iter"][0] + MP_MOVE
            curves.append(("iterative", "iter", ITER_INK,
                           self.xs[-1] * smoothstep(0.0, 1.0, remap(t, t0, t0 + MP_CURVE))))
        legend = []
        for text, key, ink, u in curves:
            ys = [self.acc_dense] + self.acc[key]
            ch.curve(sc, place, self.xs, ys, u, ink, op, dot=False)
            dots(sc, place, ch, self.xs, ys, u, ink, op)
            legend.append((text, ink))
            if 0 < u < self.xs[-1]:
                sc.probe(("curve", key), (u * 10, 0, 0), tol=6.0)
        ch.legend(sc, cam, legend, op, MP_LEGEND)


# ================================================================== the lottery ticket

LT_PPU = 9.0
LT_TEMPLATES = (0, 1)                               # layer-1 neurons shown per row
LT_ROWS_AT, LT_SCALE, LT_GAP = (0.295, 0.60), 0.46, 2.0
LT_CHART = ((0.55, 0.935), (0.30, 0.84))
LT_LEGEND = (0.71, 0.34)
LT_ERR_RANGE = (0.02, 0.9)                          # test error, log scale, bottom to top
LT_ERR_TICKS = ((0.5, "50%"), (0.2, "20%"), (0.1, "10%"), (0.05, "5%"), (0.02, "2%"))
LT_INKS = {"ticket": TUNED_INK, "random": PRUNED_INK, "dense": (0.30, 0.32, 0.34)}
LT_NAMES = {"ticket": "the ticket", "random": "random mask", "dense": "dense"}
LT_MASK_HI = INK_HI
LT_TRAIN, LT_SEED2 = 13.0, 5.0                      # seconds: the training phase, the second seed


class LotteryTicket(NetAnimation):
    """The lottery ticket hypothesis at MNIST scale (see the module docstring).

    ``runs`` are :func:`voxanim.prune.lottery`'s dicts for two seeds; the
    sheets show the first.
    """

    PHASES = ("trained", "mask", "rewind", "random", "retrain")

    def __init__(self, runs):
        super().__init__("The lottery ticket", "", LT_PPU)
        self.hold = 2.5
        self.runs = runs
        d = runs[0]
        self.steps = d["steps"]
        self.trained, self.init = d["trained_W0"], d["init_W0"]
        self.mask, self.rand = d["mask_W0"], d["rand_W0"]
        self.final = {k: d["final_%s_W0" % k] for k in ("ticket", "random")}
        self.sorted_abs = np.sort(np.abs(np.concatenate(
            [d["trained_W%d" % k].ravel() for k in range(3)])))
        self.total = self.sorted_abs.size
        self.kept = sum(int(d["mask_" + k].sum()) for k in ("W0", "W1", "W2"))
        self.layer_density = [float(d["mask_W%d" % k].mean()) for k in range(3)]
        self.acc_final = [{k: float(r["acc_" + k][-1]) for k in LT_INKS} for r in runs]
        self.vmax_trained = color_scale(self.trained)
        self.vmax_init = color_scale(self.init)
        self.chart = LossChart(LT_CHART, (0, int(self.steps[-1])), LT_ERR_RANGE, ylog=True)

        # ---- geometry: three rows of templates, stacked in world space
        self.rows = []
        pitch = (28 + LT_GAP) * LT_SCALE
        centre = self.cam.frac(*LT_ROWS_AT)
        for r in range(3):
            row = []
            for c, _j in enumerate(LT_TEMPLATES):
                s = self.sheet((28, 28), LT_ROWS_AT, LT_SCALE, "row%d_%d" % (r, c))
                s.block.place(centre + np.array([(c - (len(LT_TEMPLATES) - 1) / 2) * pitch, 0.0,
                                                 (1 - r) * pitch]))
                row.append(s)
            self.rows.append(row)

        # ---- timeline
        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("trained", 4.5)
        add("mask", 5.0)
        add("rewind", 5.5)
        add("random", 4.5)
        add("train", LT_TRAIN)
        add("seed2", LT_SEED2)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["mask"][0], Tm["rewind"][0], Tm["random"][0], Tm["train"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        a0, a1 = self.acc_final
        self.subtitle = [
            "train the 784-64-32-10 MLP (seed 0), then prune %s of its weights by magnitude"
            % pct(1 - TICKET_DENSITY),
            "the mask: %s of the weights kept (%s of layer 1, %s of layer 2, %s of layer 3)"
            % (pct(TICKET_DENSITY), *(pct(v) for v in self.layer_density)),
            "rewind: the same mask on the network's original random initialisation; the sparse "
            "network is the ticket",
            "for comparison, the initialisation under a random mask of the same density in "
            "each layer",
            "all three retrained from the same init, %s steps: ticket %s, random mask %s, "
            "dense %s" % (f"{int(self.steps[-1]):,}", pct(a0["ticket"], 1),
                          pct(a0["random"], 1), pct(a0["dense"], 1)),
        ]
        if reset <= 0 and t >= self.Tm["seed2"][0]:
            win = a0["ticket"] > a0["random"] and a1["ticket"] > a1["random"]
            self.subtitle[4] = (
                "seed 1 agrees: ticket %s, random mask %s, dense %s; the ticket %s and learns "
                "fastest" % (pct(a1["ticket"], 1), pct(a1["random"], 1), pct(a1["dense"], 1),
                             "wins" if win else "does not win"))
        self.subtitle = self.subtitle[ph]
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        op = smoothstep(0.0, 0.6, t) * back
        self._rows(sc, t, op)
        self._chart(sc, t, op)
        self.step_indicator(sc, ph, self.PHASES)

    def _row_states(self, t):
        """Per row: ``(values, mask, white, opacity)`` at time ``t``."""
        Tm = self.Tm
        train = ease_in_out(remap(t, Tm["train"][0], Tm["train"][0] + LT_TRAIN * 0.9))
        # row 0: trained weights -> the fraction below the threshold goes dark -> a mask
        mk = self.progress("mask", t)
        f = (1 - TICKET_DENSITY) * ease_in_out(remap(mk, 0.0, 0.6))
        k = int(round(f * self.total))
        thr = float(self.sorted_abs[min(k, self.total - 1)]) if k > 0 else 0.0
        m0 = np.abs(self.trained) >= thr if thr > 0 else np.ones(self.trained.shape, bool)
        to_mask = smoothstep(0.7, 1.0, mk)
        if to_mask >= 1:
            m0 = self.mask
        row0 = (self.trained, m0, to_mask, 1.0)
        # row 1: the initialisation, then under the ticket's mask, then trained
        rw = self.progress("rewind", t)
        drop = smoothstep(0.35, 0.75, rw)
        m1 = self.mask if drop >= 1 else (np.ones(self.mask.shape, bool) if drop <= 0 else None)
        v1 = self.init + (self.final["ticket"] - self.init) * train
        row1 = (v1, m1, drop, smoothstep(0.0, 0.3, rw))
        # row 2: the initialisation under a random mask, then trained
        rd = self.progress("random", t)
        drop2 = smoothstep(0.35, 0.75, rd)
        m2 = self.rand if drop2 >= 1 else (np.ones(self.rand.shape, bool) if drop2 <= 0 else None)
        v2 = self.init + (self.final["random"] - self.init) * train
        row2 = (v2, m2, drop2, smoothstep(0.0, 0.3, rd))
        return [row0, row1, row2], train

    def _rows(self, sc, t, op):
        """The three rows of templates and their captions."""
        states, train = self._row_states(t)
        for r, (row, (values, mask, prog, vis)) in enumerate(zip(self.rows, states, strict=True)):
            vmax = self.vmax_trained if r == 0 else max(self.vmax_init, color_scale(values))
            for s, j in zip(row, LT_TEMPLATES, strict=True):
                w = values[:, j]
                if mask is None:
                    # the mask dropping onto the row, top to bottom
                    rows_ = np.repeat(np.arange(28), 28) / 27.0
                    full = self.mask if r == 1 else self.rand
                    m = np.where(rows_ < prog, full[:, j], True)
                else:
                    m = mask[:, j]
                col = div_colors(w, vmax, mask=m)
                if r == 0 and prog > 0:
                    # the survivors turn into a mask: white where kept
                    maskcol = np.where(m[:, None], LT_MASK_HI[None], OFF[None])
                    col = col + (maskcol - col) * prog
                s.fill(col)
                s.cage = op * vis
                s.fade(op * vis)
                s.visible = vis > 0.01
        tex = ["trained, seed 0" if self.progress("mask", t) < 0.8 else
               "mask, %s kept" % pct(TICKET_DENSITY),
               "initialisation" if self.progress("rewind", t) < 0.6 else "the ticket",
               "random mask"]
        if train > 0:
            tex[1], tex[2] = "the ticket", "random mask"
        for r, (row, (_v, _m, _p, vis)) in enumerate(zip(self.rows, states, strict=True)):
            if vis > 0.01:
                self.caption_left(sc, around(row), tex[r], op * vis)
        self.caption(sc, around(self.rows[0]), "layer 1: neurons %s of %d (784 weights each)"
                     % (" and ".join(str(j + 1) for j in LT_TEMPLATES), SIZES[1]), op, above=True,
                     max_width=0.4)

    def _chart(self, sc, t, op):
        """Test error against steps for the three networks, then the second seed's."""
        Tm = self.Tm
        op *= smoothstep(Tm["train"][0], Tm["train"][0] + 0.8, t) * self._cap
        if op <= 0.01:
            return
        cam, place, ch = self.cam, self.cam.frac, self.chart
        last = int(self.steps[-1])
        xt = [(s, f"{s:,}") for s in range(0, last + 1, 500)]
        ch.axes(sc, cam, place, op, "training steps", "test error", xt, LT_ERR_TICKS)
        upto = last * smoothstep(0.0, 0.9, self.progress("train", t))
        second = smoothstep(0.0, 0.4, self.progress("seed2", t))
        legend = []
        for key, ink in LT_INKS.items():
            for n, run in enumerate(self.runs):
                if n == 1 and second <= 0.01:
                    continue
                o = op if n == 0 else op * second * 0.75
                err = 1.0 - run["acc_" + key]
                ch.curve(sc, place, self.steps, err, upto if n == 0 else last, ink, o,
                         width=1.3 if n == 0 else 0.8, dot=n == 0 and upto < last)
            a = [self.acc_final[0][key]]
            if second > 0.01:
                a.append(self.acc_final[1][key])
            legend.append(("%s: %s" % (LT_NAMES[key], " / ".join(pct(v, 1) for v in a)), ink))
        ch.legend(sc, cam, legend, op * smoothstep(0.5, 0.9, self.progress("train", t)),
                  LT_LEGEND)
        if second > 0.01:
            fx, fy = LT_LEGEND
            sc.text(*self.fr(fx, fy + 3 * 0.042), "thin: seed 1", layout.MIN_TEXT, "regular",
                    layout.NOTE_INK, op * second, "lm")
        if 0 < upto < last:
            sc.probe(("train",), (upto * 0.02, 0, 0), tol=6.0)


# ================================================================== structured pruning

SP_PPU = 8.0
SP_IMAGE = 18                                       # test image: the same 3 as cnn-training-step
SP_ROW_Y = (0.335, 0.505, 0.675, 0.845)             # one row per filter, layout fractions
SP_X_AT, SP_KERNEL_X, SP_MAP_X = (0.08, 0.59), 0.20, 0.305
SP_KERNEL_SCALE, SP_MAP_SCALE = 2.0, 0.40
SP_ENERGY_AT, SP_W1_AT = (0.565, 0.36), (0.565, 0.72)    # the MLP: 8 x 8 energies, 64 x 32
SP_ENERGY_SCALE, SP_W1_SCALE = 1.0, 0.34
SP_CHART = ((0.69, 0.935), (0.32, 0.76))
SP_LEGEND = (0.70, 0.62)
SP_FILTER_SLOT = 3.0                                # seconds per filter removed
SP_UNIT_TEMPO = Tempo(slow=0.7, fast=0.11, ramp_in=4.0, ramp_out=2.0)
SP_HALF = 6.0                                       # seconds of the "half gone" phase
MAP_OFF = 0.12                                      # opacity a removed map fades to
DIM = 0.55                                          # opacity of the model not under discussion
ORDINALS = ("1st", "2nd", "3rd", "last")


class StructuredPruning(NetAnimation):
    """Removing whole filters and hidden units (see the module docstring).

    ``d`` is :func:`voxanim.prune.structured_sweep`'s dict, ``cnn`` the trained
    CNN as ``{"Wc", "bc", "Wd", "bd"}``, ``mlp`` the trained MLP as ``{"W0",
    ...}`` and ``x`` the test image shown ``(784,)``.
    """

    PHASES = ("filters", "cut filters", "units", "cut units", "half gone")

    def __init__(self, d, cnn, mlp, x):
        super().__init__("Removing whole neurons and filters", "", SP_PPU)
        self.hold = 2.5
        self.d, self.cnn, self.mlp, self.x = d, cnn, mlp, x
        self.f_energy, self.f_order = d["filter_energy"], [int(v) for v in d["filter_order"]]
        self.f_acc, self.f_tuned = d["filter_acc"], d["filter_acc_tuned"]
        self.u_energy, self.u_order = d["unit_energy"], [int(v) for v in d["unit_order"]]
        self.u_acc, self.u_tuned = d["unit_acc"], d["unit_acc_tuned"]
        self.width = len(self.u_energy)
        self.half_cnn = {k: d["cnn_half_" + k] for k in ("Wc", "bc", "Wd", "bd")}
        self.half_mlp = {k: d["mlp_half_" + k] for k in mlp}
        self.maps = _cnn_forward(_cnn_params(cnn), x[None])["a1"][0]             # 26, 26, F
        self.maps_half = _cnn_forward(_cnn_params(self.half_cnn), x[None])["a1"][0]
        self.amax_maps = amax(self.maps)
        self.wcmax = amax(cnn["Wc"])
        self.vmax_w1 = color_scale(mlp["W1"])

        # ---- geometry
        self.sx = self.sheet((28, 28), SP_X_AT, 0.8, "x")
        self.ks = [self.sheet((3, 3), (SP_KERNEL_X, y), SP_KERNEL_SCALE, "k%d" % f)
                   for f, y in enumerate(SP_ROW_Y)]
        self.ms = [self.sheet((CONV_HW, CONV_HW), (SP_MAP_X, y), SP_MAP_SCALE, "map%d" % f)
                   for f, y in enumerate(SP_ROW_Y)]
        self.energy = self.sheet((8, 8), SP_ENERGY_AT, SP_ENERGY_SCALE, "energy")
        self.W1 = self.sheet(SIZES[1:3], SP_W1_AT, SP_W1_SCALE, "W1")
        self.chart_f = LossChart(SP_CHART, (0, CNN_FILTERS - 1), (0.0, 1.0))
        self.chart_u = LossChart(SP_CHART, (0, self.width - 1), (0.0, 1.0))

        # ---- timeline
        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("filters", 6.0)
        add("cut_f", SP_FILTER_SLOT * (CNN_FILTERS - 1), 0.5)
        add("units", 5.0)
        self.sweep = Sweep(t, self.width - 1, SP_UNIT_TEMPO)
        add("cut_u", self.sweep.duration + 1.5, 0.5)
        add("half", SP_HALF)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["cut_f"][0], Tm["units"][0], Tm["cut_u"][0], Tm["half"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ the schedule
    def filters_removed(self, t):
        """``(n, u)``: filters removed so far and the progress of the current removal."""
        Tm = self.Tm
        if t < Tm["cut_f"][0]:
            return 0, 0.0
        if t >= Tm["half"][0]:
            return CNN_FILTERS // 2, 1.0
        u = (t - Tm["cut_f"][0]) / SP_FILTER_SLOT
        n = min(int(u), CNN_FILTERS - 2)
        return n + 1, ease_in_out(remap(u - n, 0.0, 0.35))

    def units_removed(self, t):
        """``(n, u)``: hidden units removed so far and the progress of the current removal."""
        if t >= self.Tm["half"][0]:
            return self.width // 2, 1.0
        done, cur, u = self.sweep.at(t)
        if cur < 0:
            return 0, 0.0
        return min(done + 1, self.width - 1), ease_in_out(remap(u, 0.0, 0.6))

    def half(self, t):
        """0..1: the crossfade to the fine-tuned halves."""
        return ease_in_out(remap(t, self.Tm["half"][0], self.Tm["half"][0] + 1.5))

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        fa, ft, ua, ut = self.f_acc, self.f_tuned, self.u_acc, self.u_tuned
        h = self.width // 2
        self.subtitle = [
            "the CNN's %d filters ranked by activation energy: the mean squared ReLU output over "
            "10,000 test images" % CNN_FILTERS,
            "least active filter out first: %s at once; %s after %d fine-tune steps each"
            % (", ".join(pct(a) for a in fa[1:]), ", ".join(pct(a) for a in ft[1:]),
               FILTER_FINETUNE),
            "the MLP's %d hidden units ranked the same way, energies %.2f to %.1f; layer 2 has "
            "a row per unit" % (self.width, self.u_energy.min(), self.u_energy.max()),
            "remove them one at a time, least active first: %d fine-tune steps after each cut "
            "keep the accuracy up" % UNIT_FINETUNE,
            "half gone, fine-tuned: %d filters %s (%s without), %d units %s (%s), from %s and %s"
            % (CNN_FILTERS // 2, pct(ft[CNN_FILTERS // 2], 1), pct(fa[CNN_FILTERS // 2]), h,
               pct(ut[h], 1), pct(ua[h]), pct(fa[0], 1), pct(ua[0], 1)),
        ][ph]
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        op = smoothstep(0.0, 0.6, t) * back
        self._cnn(sc, t, op)
        self._mlp(sc, t, op)
        self._chart(sc, t, op)
        self.step_indicator(sc, ph, self.PHASES)

    def _cnn(self, sc, t, op):
        """The image, the four kernels and feature maps, energies and removals."""
        Tm = self.Tm
        focus = 1.0 if t < Tm["units"][0] or t >= Tm["half"][0] else DIM
        hf = self.half(t)
        self.sx.fill(lit_colors(self.x, 1.0, INK_LO, INK_HI))
        self.sx.light(self.x, 1.0)
        self.sx.fade(op * focus)
        self.caption(sc, self.sx, "image x", op * focus)
        n, u = self.filters_removed(t)
        gone = {}
        for rank, f in enumerate(self.f_order):
            if rank < n - 1:
                gone[f] = 1.0
            elif rank == n - 1:
                gone[f] = u
        Wc = self.cnn["Wc"] + (self.half_cnn["Wc"] - self.cnn["Wc"]) * hf
        maps = self.maps + (self.maps_half - self.maps) * hf
        for f, (k, m) in enumerate(zip(self.ks, self.ms, strict=True)):
            g = gone.get(f, 0.0)
            keep = 1.0 - (1.0 - MAP_OFF) * g
            kcol = div_colors(Wc[..., f].ravel(), self.wcmax)
            k.fill(kcol + (OFF[None] - kcol) * g)
            k.cage = op * focus
            k.fade(op * focus * keep)
            a = maps[..., f].ravel()
            mcol = lit_colors(a, self.amax_maps)
            m.fill(mcol + (OFF[None] - mcol) * g)
            m.light(a, 1.0 - g)
            m.cage = op * focus
            m.fade(op * focus * keep)
            rank = self.f_order.index(f)
            if g >= 1:
                text = "removed %s" % ORDINALS[rank]
            elif t < Tm["filters"][0] + 2.0:
                text = "energy %.2f" % self.f_energy[f]
            else:
                text = "energy %.2f, %s" % (self.f_energy[f], ORDINALS[rank])
            self.caption_side(sc, m, text, op * focus * (1.0 if g < 1 else 0.7))
        self.caption(sc, around([self.ks[0], self.ms[0]]), "%d filters 3%s3, their maps after ReLU"
                     % (CNN_FILTERS, TIMES), op * focus, above=True, max_width=0.3)
        if n > 0 and t < Tm["units"][0]:
            self.caption(sc, self.ms[-1], "%d removed: %s at once, %s fine-tuned"
                         % (n, pct(self.f_acc[n]), pct(self.f_tuned[n])), op * focus,
                         max_width=0.35)

    def _mlp(self, sc, t, op):
        """The 64 units' energies as an 8 x 8 sheet and the layer-2 rows."""
        Tm = self.Tm
        vis = smoothstep(Tm["filters"][0], Tm["filters"][0] + 1.0, t)
        focus = 1.0 if t >= Tm["units"][0] else DIM
        hf = self.half(t)
        n, u = self.units_removed(t)
        gone = np.zeros(self.width)
        if n > 0:
            gone[self.u_order[:n - 1]] = 1.0
            gone[self.u_order[n - 1]] = u
        e = lit_colors(self.u_energy, amax(self.u_energy))
        self.energy.fill(e + (OFF[None] - e) * gone[:, None])
        self.energy.light(self.u_energy * (gone < 0.5))
        self.energy.cage = op * vis * focus
        self.energy.fade(op * vis * focus)
        W1 = self.mlp["W1"] + (self.half_mlp["W1"] - self.mlp["W1"]) * hf
        col = div_colors(W1.ravel(), self.vmax_w1)
        col = col + (OFF[None] - col) * np.repeat(gone, SIZES[2])[:, None]
        self.W1.fill(col)
        self.W1.cage = op * vis * focus
        self.W1.fade(op * vis * focus)
        o = op * vis * focus
        self.caption(sc, self.energy, "%d hidden units: energy" % self.width, o, above=True)
        self.caption(sc, self.W1, "layer 2  %d%s%d, a row per unit" % (SIZES[1], TIMES, SIZES[2]),
                     o, max_width=0.25)
        if n > 0:
            self.caption(sc, self.energy, "%d removed" % n, o)

    def _chart(self, sc, t, op):
        """Accuracy against filters removed (first act), then units removed."""
        Tm = self.Tm
        cnn_act = t < Tm["units"][0]
        o = op * self._cap * (smoothstep(Tm["cut_f"][0], Tm["cut_f"][0] + 0.8, t)
                              if cnn_act else smoothstep(Tm["units"][0], Tm["units"][0] + 0.8, t))
        if o <= 0.01:
            return
        cam, place = self.cam, self.cam.frac
        if cnn_act:
            ch, xs = self.chart_f, np.arange(CNN_FILTERS)
            n, _u = self.filters_removed(t)
            ticks = [(k, str(k)) for k in range(CNN_FILTERS)]
            label = "filters removed"
            series = ((self.f_acc, PRUNED_INK), (self.f_tuned, TUNED_INK))
        else:
            ch, xs = self.chart_u, np.arange(self.width)
            n, _u = self.units_removed(t)
            ticks = [(k, str(k)) for k in (0, 16, 32, 48, 63)]
            label = "units removed"
            series = ((self.u_acc, PRUNED_INK), (self.u_tuned, TUNED_INK))
        upto = float(n)
        ch.axes(sc, cam, place, o, label, "test accuracy", ticks, PERCENT_TICKS)
        for ys, ink in series:
            ch.curve(sc, place, xs, ys, upto, ink, o, dot=False)
            dots(sc, place, ch, xs, ys, upto, ink, o, size=0.55 if cnn_act else 0.35)
        if t >= Tm["half"][0]:
            ch.mark(sc, place, self.width // 2, layout.NOTE_INK, o)
        ch.legend(sc, cam, [("no retraining", PRUNED_INK), ("fine-tuned", TUNED_INK)], o,
                  SP_LEGEND)
        if 0 < upto < xs[-1]:
            sc.probe(("cut",), (upto, 0, 0), tol=6.0)


# ================================================================== distillation

DS_PPU = 9.0
DS_IMAGE = 18                                       # the 3 that looks like an 8
DS_X_AT, DS_X_SCALE = (0.105, 0.56), 0.62
DS_BARS_AT = ((0.315, 0.55), (0.56, 0.55))          # the T = 1 and T = T bars
DS_BAR_SCALE, DS_BAR_ROWS = 0.85, 20                # a bar of p = 1 is 20 cells tall
DS_COLUMN_PX = 17.0                                 # on-screen column spacing, reference px
DS_CHART = ((0.715, 0.935), (0.30, 0.80))
DS_LEGEND = (0.745, 0.34)
DS_ERR_RANGE = (0.04, 0.9)
DS_ERR_TICKS = ((0.5, "50%"), (0.2, "20%"), (0.1, "10%"), (0.05, "5%"))
DS_INKS = {"hard": PRUNED_INK, "soft": TUNED_INK, "both": ITER_INK}
DS_NAMES = {"hard": "hard labels", "soft": "soft targets, T = %g" % TEMPERATURE, "both": "both"}
DS_TOP = 3                                          # probabilities read out per bar block
DS_SOFTEN = 3.5                                     # seconds the temperature takes to rise
DS_CURVE = 6.0                                      # seconds a student's curve takes to draw


def n_weights(sizes):
    """Weights and biases of a dense network of layer ``sizes``."""
    return sum(a * b + b for a, b in itertools.pairwise(sizes))


class Distillation(NetAnimation):
    """A teacher's soft targets training a small student (see the module docstring).

    ``d`` is :func:`voxanim.prune.distil`'s dict (the 3,000-image transfer
    set), ``full`` the same on the whole training set, ``x`` the test image
    shown and ``label`` its digit.
    """

    PHASES = ("teacher", "soften", "student", "three ways", "result")

    def __init__(self, d, full, x, label):
        super().__init__("Distillation", "", DS_PPU)
        self.hold = 2.5
        self.d, self.full, self.x, self.label = d, full, x, label
        self.steps = d["steps"]
        self.z = np.asarray(d["teacher_logits_test"][DS_IMAGE], np.float64)
        self.acc_teacher = float(d["acc_teacher"])
        self.acc = {k: float(d["acc_" + k][-1]) for k in DS_INKS}
        self.acc_full = {k: float(full["acc_" + k][-1]) for k in DS_INKS}
        self.chart = LossChart(DS_CHART, (0, int(self.steps[-1])), DS_ERR_RANGE, ylog=True)

        # ---- geometry: the image and two bar blocks
        self.sx = self.sheet((28, 28), DS_X_AT, DS_X_SCALE, "x")
        pitch = (1, column_pitch(self.cam, DS_BAR_SCALE, DS_COLUMN_PX))
        self.bars = []
        for n, at in enumerate(DS_BARS_AT):
            b = Block((DS_BAR_ROWS, 10, 1), scale=DS_BAR_SCALE, pitch=pitch, name="bars%d" % n)
            b.place(self.cam.frac(*at))
            self.blocks_extra.append(b)
            self.bars.append(b)

        # ---- timeline
        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("teacher", 6.0)
        add("soften", DS_SOFTEN + 4.0)
        add("student", DS_CURVE + 2.0)
        add("three", DS_CURVE + 3.5)
        add("result", 6.0)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["soften"][0], Tm["student"][0], Tm["three"][0], Tm["result"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def temperature(self, t):
        """The temperature of the second bar block at time ``t`` (1 rising to T)."""
        u = ease_in_out(remap(t, self.Tm["soften"][0], self.Tm["soften"][0] + DS_SOFTEN))
        return 1.0 + (TEMPERATURE - 1.0) * u

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        p1, pT = softmax(self.z[None])[0], softmax(self.z[None] / TEMPERATURE)[0]
        top = [int(i) for i in np.argsort(-pT) if i != self.label][:2]
        n_t, n_s = n_weights(TEACHER_SIZES), n_weights(STUDENT_SIZES)
        self.subtitle = [
            "the teacher, %s (%s weights, %s), on this %d: its softmax puts %.2f on %d"
            % ("-".join(map(str, TEACHER_SIZES)), f"{n_t:,}", pct(self.acc_teacher, 1),
               self.label, p1[self.label], self.label),
            "softmax of logits / %g: the wrong classes show, %d gets %.2f, %d gets %.2f; this %d "
            "looks like an %d" % (TEMPERATURE, top[0], pT[top[0]], top[1], pT[top[1]],
                                  self.label, top[0]),
            "a %s student (%s weights, %d%s fewer) trained on %s images with hard labels: %s"
            % ("-".join(map(str, STUDENT_SIZES)), f"{n_s:,}", round(n_t / n_s), TIMES,
               f"{TRANSFER_N:,}", pct(self.acc["hard"], 1)),
            "the same student on the teacher's soft targets at T = %g: %s; on both (half "
            "each): %s" % (TEMPERATURE, pct(self.acc["soft"], 1), pct(self.acc["both"], 1)),
            "soft targets pay when data is scarce: on all 60,000 images hard labels get %s, soft "
            "%s" % (pct(self.acc_full["hard"], 1), pct(self.acc_full["soft"], 1)),
        ][ph]
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        op = smoothstep(0.0, 0.6, t) * back
        self._image(sc, op)
        self._bars(sc, t, op)
        self._chart(sc, t, op)
        self.step_indicator(sc, ph, self.PHASES)

    def _image(self, sc, op):
        """The test digit, lit where its pixels are > 0."""
        self.sx.fill(lit_colors(self.x, 1.0, INK_LO, INK_HI))
        self.sx.light(self.x, 1.0)
        self.sx.fade(op)
        self.caption(sc, self.sx, "test image, a %d" % self.label, op)

    def _bar_block(self, sc, b, p, grow, op, title, note):
        """One block of ten bars of ``p``, drawn ``grow`` of the way up, with
        the digits under the columns, ``title`` above and ``note`` below."""
        b.present[...] = False
        if grow <= 0 or op <= 0.01:
            return
        heights = np.clip(np.round(p * DS_BAR_ROWS).astype(int), 0, DS_BAR_ROWS)
        heights[p > 0.02] = np.maximum(heights[p > 0.02], 1)
        up = ease_out(grow) * DS_BAR_ROWS
        for c in range(10):
            n = int(min(heights[c], np.ceil(up)))
            b.present[DS_BAR_ROWS - n:, c, 0] = True
            true = c == self.label
            b.base[:, c, 0] = ACT_HI if true else ACT_LO
            b.glow[:, c, 0] = 0.25 if true else 0.0
        b.opacity *= op
        h = 0.5 * b.scale
        for c in range(10):
            q = b.cell_center(DS_BAR_ROWS - 1, c, 0)
            self.caption(sc, (q - h, q + h), str(c), op, size=0.029)
        lo, hi = b.box((0, DS_BAR_ROWS), (0, 10), (0, 1))
        self.caption(sc, (lo, hi), title, op, above=True)
        self.caption(sc, (lo, hi), note, op * smoothstep(0.6, 1.0, grow), offset=0.045,
                     max_width=0.2)

    def _bars(self, sc, t, op):
        """The teacher's output at T = 1 and at the rising temperature."""
        Tm = self.Tm
        g1 = ease_out(remap(t, Tm["teacher"][0] + 1.0, Tm["teacher"][0] + 2.5))
        T = self.temperature(t)
        g2 = smoothstep(0.0, 1.0, remap(t, Tm["soften"][0] - 0.1, Tm["soften"][0] + 0.9))
        p1 = softmax(self.z[None])[0]
        pT = softmax(self.z[None] / T)[0]
        order = [int(i) for i in np.argsort(-pT)[:DS_TOP]]
        self._bar_block(sc, self.bars[0], p1, g1, op, "teacher softmax, T = 1",
                        "  ".join("%d: %.2f" % (i, p1[i]) for i in np.argsort(-p1)[:2]))
        self._bar_block(sc, self.bars[1], pT, g2, op, "softened, T = %.1f" % T,
                        "  ".join("%d: %.2f" % (i, pT[i]) for i in order))
        if 0 < g1 < 1 or (Tm["soften"][0] <= t < Tm["soften"][0] + DS_SOFTEN):
            sc.probe(("bars",), (T * 5, g1 * 5, 0), tol=6.0)

    def _chart(self, sc, t, op):
        """Test error against steps for the three students."""
        Tm = self.Tm
        op *= smoothstep(Tm["student"][0], Tm["student"][0] + 0.8, t) * self._cap
        if op <= 0.01:
            return
        cam, place, ch = self.cam, self.cam.frac, self.chart
        last = int(self.steps[-1])
        xt = [(s, f"{s:,}") for s in range(0, last + 1, 1000)]
        ch.axes(sc, cam, place, op, "training steps", "test error", xt, DS_ERR_TICKS)
        ch.hline(sc, place, 1.0 - self.acc_teacher, op)
        sc.text(*self.fr(DS_CHART[0][1], ch.fy(1.0 - self.acc_teacher) - 0.012),
                "teacher %s" % pct(self.acc_teacher, 1), layout.MIN_TEXT, "regular",
                layout.NOTE_INK, op, "rb")
        starts = {"hard": Tm["student"][0] + 0.5, "soft": Tm["three"][0],
                  "both": Tm["three"][0] + 2.0}
        legend = []
        for key, ink in DS_INKS.items():
            upto = last * smoothstep(0.0, 1.0, remap(t, starts[key], starts[key] + DS_CURVE))
            if upto <= 0:
                continue
            ch.curve(sc, place, self.steps, 1.0 - self.d["acc_" + key], upto, ink, op,
                     dot=upto < last)
            legend.append(("%s: %s" % (DS_NAMES[key], pct(self.acc[key], 1)), ink))
            if 0 < upto < last:
                sc.probe(("curve", key), (upto * 0.01, 0, 0), tol=6.0)
        ch.legend(sc, cam, legend, op, DS_LEGEND, step=0.038)

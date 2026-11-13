"""Kinks and curves: what a ReLU network computes is piecewise linear, what a
GELU, SiLU or tanh network computes is smooth, and both facts can be measured
on the trained race networks (:mod:`voxanim.actfun_race`).

    784 -> 128 -> 64 -> 10 after 1,500 Adam steps, one activation each

1. **two digits** -- a 3 and an 8 from the test set and the straight line
   between them in pixel space, x(α) = (1 − α)·3 + α·8, with the blends at
   α = ¼, ½, ¾,
2. **the output** -- each network's logit for "8", z₈(α), along the line
   (401 points): all seven rise from a 3 to an 8,
3. **the slope** -- dz₈/dα: under ReLU a staircase with one step per hidden
   unit that switches on or off along the line (50 of them; 57 under Leaky
   ReLU), under GELU, SiLU and tanh a curve.  The size of the largest jump
   of the slope, relative to the mean slope, is read out: 13% and 20%
   against under 1%,
4. **the loss** -- the training loss of one minibatch of 64 along a random
   direction in weight space, θ + s·d, each layer of d scaled to that
   layer's own size.  Over a wide range (s = ±0.6) all seven slices are
   smooth bowls; zoomed in sixty-fold (±0.01) and differentiated, the slope
   of the ReLU and Leaky ReLU slices is a staircase (the loss is a chain of
   straight pieces) while GELU's, SiLU's and tanh's slopes are curves: slope
   jumps of 11% and 24% of the mean slope against under 1%.  That is the
   smoother ground a smooth activation gives the optimiser; at the scale of
   a whole step it hardly shows, which is why ReLU trains fine.

Every curve is the trained parameters applied to real inputs; the slices
are cached (:func:`loss_slices`) under the runs' settings.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .actfun import (
    ACTS,
    KINKED,
    RACE,
    RACE_SIZES,
    SEED,
    along,
    data,
    direction,
    kinks,
    line_between,
    loss_slice,
    race_runs,
    roughness,
)
from .cache import cached_npz
from .diagram import Diagram, draw_image
from .easing import ease_in_out, remap, smoothstep
from .losschart import LossChart

CLASS_A, CLASS_B = 3, 8                       # the two digits: the first of each in the test subset
LINE_N = 401                                  # points along the line
SLOPE_KEYS = ("relu", "leaky_relu", "gelu", "silu", "tanh")    # whose slopes are drawn
SLICE_N = 64                                  # training images the loss slice is measured on
WIDE, WIDE_N = 0.6, 121                       # the wide slice: s in [-WIDE, WIDE]
ZOOM, ZOOM_N = 0.01, 401                      # the zoomed slice (its slope is drawn)
IMG_Y, IMG_PX, BLEND_PX = 0.26, 52.0, 38.0    # the row of digits: layout y, px sizes
IMG_X = (0.10, 0.30, 0.50, 0.70, 0.90)        # α = 0, ¼, ½, ¾, 1
LOGIT_BOX = ((0.08, 0.95), (0.42, 0.61))
SLOPE_BOX = ((0.08, 0.95), (0.71, 0.92))
WIDE_BOX = ((0.08, 0.46), (0.40, 0.82))
ZOOM_BOX = ((0.60, 0.95), (0.40, 0.82))
CURSOR = (0.5, 0.52, 0.54)
ALPHA_TICKS = ((0.25, "α = ¼"), (0.5, "½"), (0.75, "¾"))
NAMES_Y = 0.95                                # the row of names under the loss charts
DIGITS_T, OUTPUT_T, SLOPE_T, LOSS_T = 5.0, 10.0, 10.0, 13.0


def loss_slices(runs, dat, cache_dir=None):
    """``{"s_wide", "s_zoom", "wide" (7, WIDE_N), "zoom" (7, ZOOM_N)}``: the
    training loss of the first :data:`SLICE_N` images along one seeded random
    direction per network (cached under the runs' settings)."""
    settings = dict(task="loss_slices", sizes=list(RACE_SIZES), train=dict(RACE),
                    acts=[a.key for a in ACTS], n=SLICE_N, wide=[WIDE, WIDE_N],
                    zoom=[ZOOM, ZOOM_N], seed=SEED, data=dat.settings())

    def build():
        x, y = dat.x_train[:SLICE_N], dat.onehot(dat.y_train[:SLICE_N])
        s_wide, s_zoom = np.linspace(-WIDE, WIDE, WIDE_N), np.linspace(-ZOOM, ZOOM, ZOOM_N)
        wide, zoom = [], []
        for r in runs:
            d = direction(r.P, np.random.default_rng(SEED))
            wide.append(loss_slice(r.spec, r.P, x, y, d, s_wide))
            zoom.append(loss_slice(r.spec, r.P, x, y, d, s_zoom))
        return dict(s_wide=s_wide, s_zoom=s_zoom, wide=np.array(wide), zoom=np.array(zoom))

    return cached_npz("actfun_slices", settings, build, cache_dir)


def slope_of(s, values):
    """``(midpoints, slopes)``: the slope of each row of ``values (n, len(s))``
    between consecutive samples, each row minus its own mean (the curves are
    compared for their shape, not their level)."""
    d = np.diff(values, axis=1) / np.diff(s)[None]
    return (s[1:] + s[:-1]) / 2, d - d.mean(1, keepdims=True)


def pct(v):
    """A fraction as a percentage, ``<1%`` when it rounds to nothing."""
    return "<1%" if v < 0.005 else "%d%%" % round(100 * v)


def first_of(dat, digit):
    """Index of the first test image of ``digit`` in the subset."""
    return int(np.nonzero(dat.y_test == digit)[0][0])


def rounded_range(lo, hi, step):
    """``(lo, hi)`` widened to multiples of ``step``."""
    return float(np.floor(lo / step) * step), float(np.ceil(hi / step) * step)


class ActivationSmoothness(Diagram):
    """The line between two digits, the outputs and slopes along it, the loss
    slices (see the module)."""

    PHASES = ("two digits", "the output", "the slope", "the loss")

    def __init__(self, runs=None, dat=None, slices=None):
        super().__init__("Kinks and curves", "")
        dat = data() if dat is None else dat
        self.runs = race_runs(dat=dat) if runs is None else runs
        self.hold = 2.5
        ia, ib = first_of(dat, CLASS_A), first_of(dat, CLASS_B)
        self.xa, self.xb = dat.x_test[ia], dat.x_test[ib]
        self.alpha = np.linspace(0.0, 1.0, LINE_N)
        line = line_between(self.xa, self.xb, LINE_N)
        self.z, self.p, self.kinks, self.rough = [], [], [], []
        for r in self.runs:
            z, p = along(r.spec, r.P, line, CLASS_B)
            self.z.append(z)
            self.p.append(p)
            self.kinks.append(kinks(r.spec, r.P, line))
            self.rough.append(roughness(z))
        self.slope = [np.gradient(z, self.alpha) for z in self.z]
        self.slices = loss_slices(self.runs, dat) if slices is None else slices
        self.s_mid, self.zoom_slope = slope_of(self.slices["s_zoom"], self.slices["zoom"])
        self.rough_zoom = [roughness(v) for v in self.slices["zoom"]]
        keys = [a.key for a in ACTS]
        self.shown = [keys.index(k) for k in SLOPE_KEYS]
        zr = rounded_range(min(z.min() for z in self.z), max(z.max() for z in self.z), 2.0)
        sr = rounded_range(min(self.slope[i].min() for i in self.shown),
                           max(self.slope[i].max() for i in self.shown), 2.0)
        self.logit_chart = LossChart(LOGIT_BOX, (0.0, 1.0), zr)
        self.slope_chart = LossChart(SLOPE_BOX, (0.0, 1.0), sr)
        wide = self.slices["wide"]
        self.wide_chart = LossChart(WIDE_BOX, (-WIDE, WIDE), (0.0, rounded_range(0, wide.max(),
                                                                                  0.1)[1]))
        zr2 = rounded_range(self.zoom_slope[self.shown].min(), self.zoom_slope[self.shown].max(),
                            0.001)
        self.zoom_chart = LossChart(ZOOM_BOX, (-ZOOM, ZOOM), zr2)

        Tm, t = {}, 0.3
        Tm["digits"] = (t, t + DIGITS_T)
        t += DIGITS_T
        Tm["output"] = (t, t + OUTPUT_T)
        t += OUTPUT_T + 1.0
        Tm["slope"] = (t, t + SLOPE_T)
        t += SLOPE_T + 1.5
        Tm["loss"] = (t, t + LOSS_T)
        t += LOSS_T
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["output"][0], Tm["slope"][0], Tm["loss"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        loss = self.progress("loss", t)
        swap = smoothstep(0.0, 0.3, remap(t, self.Tm["loss"][0] - 1.5, self.Tm["loss"][0] - 0.3))
        line_op = (1.0 - swap) * back
        self._digits(sc, t, line_op)
        self._output(sc, t, line_op)
        self._slope(sc, t, line_op)
        self._loss(sc, loss, swap * back)
        self.step_indicator(sc, ph, self.PHASES)
        self.subtitle = self._words(t, reset)

    def _tex(self, x):
        """A 784-vector as a dark-on-light ``(28, 28, 3)`` texture."""
        return 1.0 - np.repeat(np.asarray(x, np.float32).reshape(28, 28, 1), 3, 2)

    # -- the digits -------------------------------------------------------------------------
    def _digits(self, sc, t, op):
        op *= smoothstep(0.0, 0.5, t)
        if op <= 0.01:
            return
        u = self.progress("digits", t)
        alphas = (0.0, 0.25, 0.5, 0.75, 1.0)
        for k, (fx, al) in enumerate(zip(IMG_X, alphas)):
            end = al in (0.0, 1.0)
            o = op * (1.0 if end else smoothstep(0.2 + 0.15 * k, 0.4 + 0.15 * k, u))
            px = IMG_PX if end else BLEND_PX
            c = self.P(fx, IMG_Y)
            draw_image(self, sc, c, self._tex((1 - al) * self.xa + al * self.xb), px, px, o,
                       grid=0.0, lift=0.1)
            text = ("a %d, α = 0" % CLASS_A, "¼", "½", "¾", "an %d, α = 1" % CLASS_B)[k]
            self.label(sc, c, text, layout.MIN_TEXT, anchor="mt", dy=px / 2 + 4,
                       ink=layout.NOTE_INK if 0 < k < 4 else layout.CAPTION_INK, opacity=o)
        # the line through the row
        a, b = self.P(IMG_X[0] + 0.045, IMG_Y), self.P(IMG_X[-1] - 0.045, IMG_Y)
        line_u = smoothstep(0.1, 0.6, u)
        if line_u > 0:
            sc.line(a, a + (b - a) * line_u, width=1.2, color=CURSOR, opacity=0.6 * op, bias=0.0)

    # -- the output and its slope -----------------------------------------------------------
    def _cursor(self, sc, chart, alpha, op):
        a, b = self.P(chart.fx(alpha), chart.y1), self.P(chart.fx(alpha), chart.y0)
        sc.line(a, b, width=0.9, color=CURSOR, opacity=0.5 * op, bias=0.0)

    def _names(self, sc, idx, fy, op, left=None, head=""):
        """A row of the networks' names in their inks at layout ``fy``, centred,
        or starting at layout x ``left`` with the caption ``head`` first."""
        labels = [ACTS[i].label for i in idx]
        inks = [ACTS[i].ink for i in idx]
        if head:
            labels, inks = [head] + labels, [layout.CAPTION_INK] + inks
        widths = [layout.text_width(s, layout.MIN_TEXT) for s in labels]
        gap = 0.022
        total = sum(widths) + gap * (len(labels) - 1)
        x = 0.5 - total / 2 if left is None else self.fr(self.P(left, fy))[0]
        y = self.fr(self.P(0.5, fy))[1]
        for s, ink, w in zip(labels, inks, widths):
            sc.text(x, y, s, layout.MIN_TEXT, "regular", ink, op * self._cap, "lm")
            x += w + gap

    def _output(self, sc, t, op):
        u = smoothstep(0.0, 0.08, self.progress("output", t)) * op * self._cap
        if u <= 0.01:
            return
        upto = remap(self.progress("output", t), 0.05, 0.9)
        ch = self.logit_chart
        ch.axes(sc, self.cam, self.P, u, "", "", ALPHA_TICKS,
                [(v, "%g" % v) for v in np.arange(ch.y_range[0], ch.y_range[1] + 1e-9, 4.0)])
        for i, a in enumerate(ACTS):
            ch.curve(sc, self.P, self.alpha, self.z[i], upto, a.ink, u, width=1.4, every=2)
        if upto < 1.0:
            self._cursor(sc, ch, upto, u)
        self._names(sc, range(len(ACTS)), LOGIT_BOX[1][0] - 0.03, u, left=LOGIT_BOX[0][0],
                    head="z₈, the logit for '8':")
        sc.probe(("output",), (upto * 300, 0, 0), tol=6.0)

    def _slope(self, sc, t, op):
        u = smoothstep(0.0, 0.08, self.progress("slope", t)) * op * self._cap
        if u <= 0.01:
            return
        upto = remap(self.progress("slope", t), 0.05, 0.9)
        ch = self.slope_chart
        step = 2.0 if ch.y_range[1] - ch.y_range[0] <= 14 else 5.0
        ch.axes(sc, self.cam, self.P, u, "", "", ALPHA_TICKS,
                [(v, "%g" % v) for v in np.arange(ch.y_range[0], ch.y_range[1] + 1e-9, step)])
        for i in self.shown:
            ch.curve(sc, self.P, self.alpha, self.slope[i], upto, ACTS[i].ink, u, width=1.4,
                     every=1 if ACTS[i].key in KINKED else 2)
        if upto < 1.0:
            self._cursor(sc, ch, upto, u)
        self._names(sc, self.shown, SLOPE_BOX[1][0] - 0.03, u, left=SLOPE_BOX[0][0],
                    head="dz₈/dα, its slope along the line:")
        sc.probe(("slope",), (upto * 300, 0, 0), tol=6.0)

    # -- the loss slices ---------------------------------------------------------------------
    def _loss(self, sc, loss, op):
        op *= self._cap
        if op <= 0.01:
            return
        s_w = self.slices["s_wide"]
        up_w = -WIDE + 2 * WIDE * remap(loss, 0.0, 0.35)
        up_z = self.s_mid[0] + (self.s_mid[-1] - self.s_mid[0]) * remap(loss, 0.4, 0.8)
        wc, zc = self.wide_chart, self.zoom_chart
        wc.axes(sc, self.cam, self.P, op, "s (every layer moved by s of its size)",
                "loss of one minibatch", [(v, "%g" % v) for v in (-0.3, 0, 0.3)],
                [(v, "%g" % v) for v in np.arange(0, wc.y_range[1] + 1e-9, 0.1)])
        for i, a in enumerate(ACTS):
            wc.curve(sc, self.P, s_w, self.slices["wide"][i], up_w, a.ink, op, width=1.4)
        zop = op * smoothstep(0.38, 0.45, loss)
        zc.axes(sc, self.cam, self.P, zop, "s, zoomed in %d×" % round(WIDE / ZOOM),
                "slope of the loss, each minus its mean",
                [(v, "%g" % v) for v in (-0.005, 0, 0.005)],
                [(v, "%+.3f" % v if abs(v) > 1e-9 else "0") for v in
                 np.arange(-0.02, 0.0201, 0.002) if zc.y_range[0] <= v <= zc.y_range[1]])
        for i in self.shown:
            zc.curve(sc, self.P, self.s_mid, self.zoom_slope[i], up_z, ACTS[i].ink, zop,
                     width=1.4)
        self._names(sc, range(len(ACTS)), NAMES_Y, op)
        sc.probe(("loss",), (loss * 300, 0, 0), tol=6.0)

    # ------------------------------------------------------------------ words
    def _words(self, t, reset):
        T = self.Tm
        i = {a.key: n for n, a in enumerate(ACTS)}
        if reset > 0 or t < T["output"][0]:
            return ("the straight line between a %d and an %d in pixel space, x(α) = (1 − α)·%d + "
                    "α·%d: the same inputs for all seven trained networks"
                    % (CLASS_A, CLASS_B, CLASS_A, CLASS_B))
        if t < T["slope"][0]:
            zr = self.z[i["relu"]]
            return ("z₈, each network's logit for '%d', along the line: from %.1f to %.1f under "
                    "ReLU, where p(8) goes %.3f → %.3f; every curve rises, so look at the slope"
                    % (CLASS_B, zr[0], zr[-1], self.p[i["relu"]][0], self.p[i["relu"]][-1]))
        if t < T["loss"][0] - 1.5:
            return ("under ReLU the slope is a staircase: %d steps, one per hidden unit switching "
                    "along the line (Leaky ReLU %d); GELU, SiLU and tanh curve. Largest jump: %s "
                    "and %s of the mean slope against %s, %s, %s"
                    % (self.kinks[i["relu"]], self.kinks[i["leaky_relu"]],
                       pct(self.rough[i["relu"]]), pct(self.rough[i["leaky_relu"]]),
                       pct(self.rough[i["gelu"]]), pct(self.rough[i["silu"]]),
                       pct(self.rough[i["tanh"]])))
        rz = self.rough_zoom
        return ("the loss along a random weight direction: at ±%.1f all seven are smooth bowls; "
                "at ±%.2f the slope of ReLU's is a staircase (jumps %s of the mean slope, Leaky "
                "ReLU %s), GELU's and SiLU's a curve (%s, %s)"
                % (WIDE, ZOOM, pct(rz[i["relu"]]), pct(rz[i["leaky_relu"]]), pct(rz[i["gelu"]]),
                   pct(rz[i["silu"]])))

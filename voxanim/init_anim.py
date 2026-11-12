"""Initialisation: the same 20-layer tanh MLP under three choices of weight scale.

    784 -> 100 x 20 (tanh) -> 10, trained on 10,000 MNIST images by plain SGD

Three copies of the network differ only in the std of their weights: 0.01
(too small), 1 (too large) and Xavier's ``1 / sqrt(fan_in) = 0.1``.  Each is
a panel of 20 columns, one per layer:

1. **network** -- the three panels and what each init is,
2. **forward** -- the std of every layer's activations, as bars growing up
   from the baseline, on a log scale (one decade per row).  With std 0.01
   the activations shrink tenfold a layer, to 8×10⁻²¹ at layer 20; with std
   1, tanh saturates and every layer has std 1; Xavier keeps them near 0.2,
3. **backward** -- the gradient of the loss reaching each layer, as bars
   growing down from the baseline, also one decade per row.  It shrinks to
   2×10⁻²² at layer 1 for the small init, *grows* a million-fold for the
   large one (each layer multiplies by ~2), and stays flat for Xavier,
4. **histogram** -- layer 10's activations: a spike at 0, two spikes at ±1,
   or a bell,
5. **training** -- 400 steps of SGD at learning rate 0.1: only the Xavier
   network learns (test accuracy 12%, 10% and 89%).

Every number comes from recorded runs of :mod:`voxanim.deepnet`, measured
on the first minibatch (phases 2-4) and over the run (phase 5).  SGD, not
Adam, because Adam rescales gradients by their own size and so hides what
the bars show.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from .conv import Tempo
from .deepnet import Spec, subset, train
from .diagram import GRAD, GRAD_HI
from .easing import ease_in_out, remap, smoothstep
from .losschart import LossChart, pow10, sup
from .netanim import ACT_HI, ACT_LO, OFF, NetAnimation, div_colors

DEPTH, WIDTH = 20, 100
SIZES = (784,) + (WIDTH,) * DEPTH + (10,)
INITS = (("std 0.01", Spec(SIZES, "tanh", "normal", 0.01)),
         ("std 1", Spec(SIZES, "tanh", "normal", 1.0)),
         ("Xavier, std 0.1", Spec(SIZES, "tanh", "xavier")))
TRAIN = dict(steps=400, batch=64, lr=0.1, opt="sgd", every=20, probe_layer=9)
N_TRAIN, N_TEST = 10000, 2000
HIST_LAYER = TRAIN["probe_layer"] + 1          # the layer whose activations are binned (1-based)
INKS = ((0.36, 0.38, 0.42), (0.75, 0.22, 0.22), tuple(float(v) for v in C.INPUT))

HALF = 18                                      # rows per half panel: one decade each
ACT_RANGE = (-18.0, 0.0)                       # log10 activation std: baseline .. top row
GRAD_RANGE = (-14.0, 4.0)                      # log10 gradient RMS: baseline .. bottom row
BINS = DEPTH                                   # histogram bins over (-1, 1): one per column
HIST_FULL = 0.5                                # the fraction of values that fills a column
BAR = 0.55                                     # voxel size of the bars
PITCH = (1.0, 1.5)                             # rows touch; columns leave a gap
PANEL_X = (0.235, 0.53, 0.825)                 # frame x of the three panels
PANEL_Y = 0.615
HIST_ZERO = C.rgb("#6a7076")                   # histogram bins near 0: a grey that shows
CHART = ((0.26, 0.68), (0.34, 0.82))           # the training chart ((x0, x1), (top, bottom))
LEGEND = (0.715, 0.50)                         # the legend, right of the chart
SMOOTH = 8                                     # batches the drawn loss is averaged over
BASELINE = (0.62, 0.64, 0.66)


def runs(cache_dir=None):
    """The three recorded training runs (trained and cached on first use)."""
    data = subset(N_TRAIN, N_TEST, seed=0, cache_dir=cache_dir)
    kw = dict(TRAIN)
    if cache_dir:
        kw["cache_dir"] = cache_dir
    return [train(spec, data, **kw) for _name, spec in INITS]


def histogram(a, bins=BINS):
    """Fraction of the activations ``a`` in each of ``bins`` equal bins over (-1, 1)."""
    counts, _edges = np.histogram(np.asarray(a).ravel(), bins=bins, range=(-1.0, 1.0))
    return counts / max(1, counts.sum())


class Initialisation(NetAnimation):
    """Three initialisations of one deep network, side by side (see the module)."""

    PHASES = ("network", "forward", "backward", "histogram", "training")

    def __init__(self, runs):
        super().__init__("Initialisation", "", 14.0, azimuth=26.0, elevation=18.0)
        self.runs = runs
        self.hold = 2.5
        self.panels = [self.sheet((2 * HALF, DEPTH), (x, PANEL_Y), BAR, "panel%d" % i, pitch=PITCH)
                       for i, x in enumerate(PANEL_X)]
        self.act = [r.act_std[0] for r in runs]                 # (DEPTH,) at step 0
        self.grad = [r.grad_rms[0] for r in runs]
        self.hist = [histogram(r.probe["a"][0]) for r in runs]
        self.mean_abs = [float(np.abs(r.probe["a"][0]).mean()) for r in runs]
        kernel = np.ones(SMOOTH) / SMOOTH
        self.curves = [np.convolve(r.batch_loss, kernel, "valid") for r in runs]
        self.acc = [float(r.test_acc[-1]) for r in runs]
        self.chart = LossChart(CHART, (0, TRAIN["steps"]), (0.0, 3.0))

        # ---- timeline -----------------------------------------------------------------
        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("network", 3.5, 0.3)
        fw = Tempo(slow=0.55, fast=0.13, ramp_in=3.0, ramp_out=2.0).durations(DEPTH)
        self.f_starts, self.f_durs = t + np.concatenate([[0.0], np.cumsum(fw)[:-1]]), fw
        add("fwd", float(fw.sum()), 1.8)
        self.b_starts, self.b_durs = t + np.concatenate([[0.0], np.cumsum(fw)[:-1]]), fw
        add("bwd", float(fw.sum()), 1.8)
        add("hist", 7.0, 0.5)
        add("train", 10.0, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["fwd"][0], Tm["bwd"][0], Tm["hist"][0], Tm["train"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _sweep(self, t, name, starts, durs):
        """``(layers done, current layer, its progress)`` of a column sweep;
        the current layer is -1 before the sweep and ``DEPTH`` after it."""
        if t < self.Tm[name][0]:
            return 0, -1, 0.0
        if t >= self.Tm[name][1]:
            return DEPTH, DEPTH, 0.0
        k = int(np.searchsorted(starts, t, side="right") - 1)
        return k, k, (t - starts[k]) / durs[k]

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        a, g = self.act, self.grad
        self.subtitle = [
            "a %d-layer tanh MLP (784 → %d × %d → 10) initialised three ways: weights too "
            "small, too large, and Xavier's 1/√n" % (DEPTH, WIDTH, DEPTH),
            "forward: the std of each layer's activations (one decade per row); std 0.01 "
            "shrinks them to %s by layer %d, std 1 saturates tanh at ±1" % (pow10(a[0][-1]), DEPTH),
            "backward: the gradient reaching each layer; it fades to %s at layer 1 with std "
            "0.01, grows to %s with std 1, stays near %s with Xavier"
            % (pow10(g[0][0]), pow10(g[1][0]), pow10(g[2][0])),
            "layer %d's activations binned: all at 0 (nothing to learn from), all at ±1 "
            "(tanh saturated, slope ≈ 0), or a bell" % HIST_LAYER,
            "training: %d steps of SGD (learning rate %g) on %s images; only the Xavier network "
            "learns: test accuracy %d%%, %d%% and %d%%"
            % (TRAIN["steps"], TRAIN["lr"], f"{N_TRAIN:,}", *[round(100 * v) for v in self.acc]),
        ][ph]
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        intro = smoothstep(0.0, 0.6, t)
        done_f, nf, uf = self._sweep(t, "fwd", self.f_starts, self.f_durs)
        done_b, nb, ub = self._sweep(t, "bwd", self.b_starts, self.b_durs)
        hist = smoothstep(0.0, 0.12, self.progress("hist", t))
        train = self.progress("train", t)
        panels = (1.0 - smoothstep(0.0, 0.12, train)) * back * intro
        self._panels(sc, t, done_f, nf, uf, done_b, nb, ub, hist, panels)
        self._chart(sc, train, back)
        self.step_indicator(sc, ph, self.PHASES)

    # ------------------------------------------------------------------ the panels
    def _rows(self, value, lo_hi):
        """Rows a bar gets for ``value`` on the log scale ``lo_hi`` (0 .. HALF)."""
        lo, hi = lo_hi
        return int(round(np.clip((np.log10(max(value, 1e-300)) - lo) / (hi - lo), 0, 1) * HALF))

    def _bars(self, i, done_f, nf, uf, done_b, nb, ub):
        """``(colours (n, 3), present (n,), white (n,))`` of panel ``i``'s bars."""
        colors = np.tile(OFF, (2 * HALF, DEPTH, 1)).astype(np.float32)
        present = np.zeros((2 * HALF, DEPTH), bool)
        white = np.zeros((2 * HALF, DEPTH), np.float32)
        ramp = np.linspace(0.0, 1.0, HALF)[:, None]
        act_ramp = ACT_LO[None] + (ACT_HI - ACT_LO)[None] * ramp         # dim at the baseline
        grad_ramp = GRAD[None] + (GRAD_HI - GRAD)[None] * ramp
        for l in range(DEPTH):
            if l < done_f or (l == nf and uf >= 0.5):
                h = self._rows(self.act[i][l], ACT_RANGE)
                present[HALF - h:HALF, l] = True
                colors[HALF - h:HALF, l] = act_ramp[HALF - h:][::-1]
            if l == nf and uf < 1.0:
                white[:HALF, l] = 0.6 * (1.0 - smoothstep(0.7, 1.0, uf))
            lb = DEPTH - 1 - l                                     # backward: last layer first
            if lb < done_b or (lb == nb and ub >= 0.5):
                h = self._rows(self.grad[i][l], GRAD_RANGE)
                present[HALF:HALF + h, l] = True
                colors[HALF:HALF + h, l] = grad_ramp[HALF - h:]
            if lb == nb and ub < 1.0:
                white[HALF:, l] = 0.6 * (1.0 - smoothstep(0.7, 1.0, ub))
        return colors.reshape(-1, 3), present.ravel(), white

    def _histogram(self, i):
        """``(colours, present)`` of panel ``i`` showing layer HIST_LAYER's histogram."""
        colors = np.tile(OFF, (2 * HALF, DEPTH, 1)).astype(np.float32)
        present = np.zeros((2 * HALF, DEPTH), bool)
        centres = (np.arange(BINS) + 0.5) / BINS * 2.0 - 1.0
        k = np.sqrt(np.abs(centres))[:, None]              # grey at 0, full colour at ±1
        tint = HIST_ZERO[None] + (div_colors(centres, 1.0) - HIST_ZERO[None]) * k
        for b in range(BINS):
            h = int(round(min(1.0, self.hist[i][b] / HIST_FULL) * 2 * HALF))
            present[2 * HALF - h:, b] = True
            colors[2 * HALF - h:, b] = tint[b]
        return colors.reshape(-1, 3), present.ravel()

    def _panels(self, sc, t, done_f, nf, uf, done_b, nb, ub, hist, op):
        """Fill the three panels with the bars (or the histogram) and caption them."""
        bars_op = op * (1.0 - smoothstep(0.0, 0.5, hist))
        hist_op = op * smoothstep(0.5, 1.0, hist)
        for i, s in enumerate(self.panels):
            if hist < 0.5:
                colors, present, white = self._bars(i, done_f, nf, uf, done_b, nb, ub)
                s.fill(colors)
                s.show(present)
                s.block.white[:, :, 0] = white
                s.fade(bars_op)
            else:
                colors, present = self._histogram(i)
                s.fill(colors)
                s.show(present)
                s.fade(hist_op)
            s.cage = op
            if hist < 0.5:
                self._baseline(sc, s, bars_op)
            name = INITS[i][0]
            self.caption(sc, s, "weights: %s" % name, op, above=True, ink=INKS[i])
            self.caption(sc, s, self._readout(i, t, done_f, done_b, hist), op, size=0.028)
        if op > 0.01:
            self._axis_labels(sc, hist, bars_op, hist_op)

    def _baseline(self, sc, s, op):
        """The line the bars grow from: activations above it, gradients below."""
        if op <= 0.01:
            return
        lo, hi = s.block.box((HALF, HALF + 1), (0, DEPTH), (0, 1))
        z = hi[2]
        front = hi[1]
        sc.line([lo[0] - 0.3, front, z], [hi[0] + 0.3, front, z], width=1.2, color=BASELINE,
                opacity=op, bias=0.3)

    def _readout(self, i, t, done_f, done_b, hist):
        """What the caption under panel ``i`` says in the current phase."""
        if hist >= 0.5:
            return "layer %d: mean |a| = %.2f" % (HIST_LAYER, self.mean_abs[i])
        if done_b > 0:
            l = max(0, DEPTH - done_b)
            return "gradient at layer %d: %s" % (l + 1, pow10(self.grad[i][l]))
        if done_f > 0:
            return "layer %d: std %s" % (done_f, pow10(self.act[i][done_f - 1]))
        return "layers 1 → %d" % DEPTH

    def _axis_labels(self, sc, hist, bars_op, hist_op):
        """Scale labels left of the first panel; the histogram's range when it shows."""
        first = self.panels[0]

        def at(s, row, side, dy=0.0):
            """Frame position beside row ``row`` of panel ``s`` (``side`` -1 left, 1 right)."""
            lo, hi = s.block.box((row, row + 1), (0, DEPTH), (0, 1))
            p = self.cam.project([lo[0] if side < 0 else hi[0], hi[1], (lo[2] + hi[2]) / 2])
            return (p[0] + side * 10) / self.cam.width, (p[1] + dy) / self.cam.height

        if bars_op > 0.01:
            # the two scales meet at the baseline: one label just above it, one just below
            scale = ((0, "std 1", 0.0, "rm"), (HALF - 1, "10" + sup(int(ACT_RANGE[0])), -1.0, "rb"),
                     (HALF, "10" + sup(int(GRAD_RANGE[0])), 1.0, "rt"),
                     (2 * HALF - 1, "grad 10" + sup(int(GRAD_RANGE[1])), 0.0, "rm"))
            for row, text, dy, anchor in scale:
                x, y = at(first, row, -1, dy)
                sc.text(x, y, text, layout.MIN_TEXT, "regular", layout.NOTE_INK, bars_op, anchor)
        if hist_op > 0.01:
            for s in self.panels:
                for side, text in ((-1, "−1"), (1, "+1")):
                    x, y = at(s, 2 * HALF - 1, side)
                    sc.text(x, y, text, layout.MIN_TEXT, "regular", layout.NOTE_INK, hist_op,
                            "rm" if side < 0 else "lm")

    # ------------------------------------------------------------------ the chart
    def _chart(self, sc, train, back):
        """The three training-loss curves (batch loss, averaged over SMOOTH batches)."""
        op = smoothstep(0.12, 0.3, train) * back * self._cap
        if op <= 0.01:
            return
        cam, place = self.cam, self.cam.frac
        n = len(self.curves[0])
        upto = SMOOTH + (n - 1) * smoothstep(0.0, 0.92, train)
        self.chart.axes(sc, cam, place, op, "training steps (batch loss, averaged over %d)"
                        % SMOOTH, "loss", [(s, str(s)) for s in (0, 100, 200, 300, 400)],
                        [(v, "%g" % v) for v in (0, 1, 2, 3)])
        steps = np.arange(n) + SMOOTH
        for i in range(3):
            self.chart.curve(sc, place, steps, self.curves[i], upto, INKS[i], op)
        legend = [("%s: %d%% test accuracy" % (INITS[i][0], round(100 * self.acc[i])), INKS[i])
                  for i in range(3)]
        self.chart.legend(sc, cam, legend, op * smoothstep(0.85, 1.0, train), LEGEND)
        sc.probe(("train",), (upto * 0.5, 0, 0), tol=6.0)

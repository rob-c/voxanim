"""Gradients through each activation: an eight-layer MLP under the seven
activations, and how much of the loss's gradient reaches each layer.

    784 -> 64 x 8 -> 10, Xavier weights for sigmoid and tanh, He for the rest,
    1,000 Adam steps (learning rate 0.001, batches of 64) on 10,000 MNIST images

The grid is seven rows (the activations) by eight columns (the layers); a
cell's brightness is the RMS of the gradient reaching that layer's output on
a log colour scale, six decades from dark to lit:

1. **network** -- the grid, empty, and what the rows and columns are,
2. **gradient** -- the untrained network: the gradient flows back from
   layer 8 to layer 1, one column at a time, and the ratio layer 1 / layer 8
   is read out per row.  Sigmoid's shrinks about four-fold a layer (its slope
   is at most 0.25), to 3×10⁻⁵ of what left layer 8; ReLU's and Leaky
   ReLU's arrive almost unchanged (slope exactly 1 or 0, He weights); GELU's
   and SiLU's sit in between (slope ½ around 0, where most pre-activations
   are), ELU's grows,
3. **slopes** -- each activation's slope f′ drawn over the real histogram of
   layer 4's pre-activations on 256 test images, the bins on a flat part of
   the activation (|f′| < 0.05) tinted and their share read out,
4. **after training** -- the same grid and histograms after the 1,000
   steps: Adam has levelled the gradients, but the sigmoid network has only
   reached 42% test accuracy against 93–94% for the others.

Every number is from the recorded runs of :func:`voxanim.actfun.deep8_runs`;
the histograms are the runs' initial and final parameters applied to one
batch of the test subset.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from .actfun import ACTS, DEEP8, DEEP8_SIZES, data, deep8_runs, fractions, hidden
from .actfun_plots import Z_RANGE, ramp_colors, slope_plot
from .conv import Tempo
from .deepnet import FLAT, act_grad
from .diagram import GRAD_HI, Diagram
from .easing import ease_in_out, remap, smoothstep
from .losschart import pow10, sup
from .netanim import screen_box
from .sweep import Sweep

DEPTH = len(DEEP8_SIZES) - 2
N_ACT = len(ACTS)
HIST_LAYER = 3                                  # 0-based: layer 4's pre-activations are binned
HIST_N = 256                                    # test images the histograms are made of
BINS = 30                                       # over Z_RANGE
HIST_FULL = 0.35                                # a bin with this share fills the plot
GRAD_RANGE = (-8.0, -2.0)                       # log10 gradient RMS: dark .. lit
DARK = C.rgb("#3a3f44")
GRID_AT, GRID_SCALE, PITCH = (0.45, 0.40), 2.6, (1.2, 1.3)
NOTE_X = 0.78                                   # the per-row read-outs, layout x
PLOT_X0, PLOT_W, PLOT_GAP = 0.005, 0.125, 0.0155
PLOT_Y = (0.69, 0.87)
PLOT_DT = 1.15                                  # seconds per slope plot
FRAME = (0.96, 0.96, 0.96)
TRAIN_T = 12.0                                  # the after-training phase


def histogram(y):
    """Fractions of the values ``y`` in :data:`BINS` bins over :data:`Z_RANGE`
    (values beyond the range land in the end bins)."""
    lo, hi = Z_RANGE
    v = np.clip(np.asarray(y).ravel(), lo + 1e-9, hi - 1e-9)
    counts, _edges = np.histogram(v, BINS, Z_RANGE)
    return counts / max(1, v.size)


def flat_bins(key):
    """Which bins' centres lie where ``|f'| <`` :data:`~voxanim.deepnet.FLAT`."""
    centres = Z_RANGE[0] + (np.arange(BINS) + 0.5) / BINS * (Z_RANGE[1] - Z_RANGE[0])
    return np.abs(act_grad(key, centres)) < FLAT


class ActivationGradients(Diagram):
    """The gradient grid, the slopes over real histograms, then after training."""

    PHASES = ("network", "gradient", "slopes", "after training")

    def __init__(self, runs=None, x=None):
        super().__init__("Gradients through each activation", "")
        if runs is None:
            dat = data()
            runs, x = deep8_runs(dat=dat), dat.x_test[:HIST_N]
        self.runs = runs
        self.hold = 2.5
        self.grad0 = np.array([r.grad_rms[0] for r in runs])           # (7, 8) at step 0
        self.grad1 = np.array([r.grad_rms[-1] for r in runs])          # after training
        self.acc = np.array([float(r.test_acc[-1]) for r in runs])
        self.hist0, self.hist1, self.flat0, self.flat1 = [], [], [], []
        for r in runs:
            for P, hist, flat in ((r.P0, self.hist0, self.flat0), (r.P, self.hist1, self.flat1)):
                ys, _a = hidden(r.spec, P, x)
                hist.append(histogram(ys[HIST_LAYER]))
                flat.append(float(fractions(r.spec, P, x)[2][HIST_LAYER]))
        self.flat_bins = [flat_bins(a.key) for a in ACTS]
        self.grid = self.sheet((N_ACT, DEPTH), GRID_AT, GRID_SCALE, "grid", pitch=PITCH)

        Tm, t = {}, 0.3
        Tm["network"] = (t, t + 4.0)
        t += 4.0
        self.bwd = Sweep(t, DEPTH, Tempo(slow=1.0, fast=0.55, ramp_in=2.0, ramp_out=1.5))
        Tm["sweep"] = (t, self.bwd.t1)
        t = self.bwd.t1 + 2.5
        Tm["slopes"] = (t, t + N_ACT * PLOT_DT)
        t += N_ACT * PLOT_DT + 4.0
        Tm["train"] = (t, t + TRAIN_T)
        t += TRAIN_T
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["sweep"][0], Tm["slopes"][0], Tm["train"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def ratio(self, grad):
        """Layer 1's gradient as a fraction of layer 8's, per row."""
        return grad[:, 0] / grad[:, -1]

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        intro = smoothstep(0.0, 0.6, t)
        done, cur, u = self.bwd.at(t)
        trained = ease_in_out(remap(self.progress("train", t), 0.05, 0.4))
        self._grid(sc, t, done, cur, u, trained, back * intro)
        self._plots(sc, t, trained, back)
        self.step_indicator(sc, ph, self.PHASES)
        self.subtitle = self._words(t, done, cur, u, reset)

    # -- the grid -------------------------------------------------------------------------
    def _grid(self, sc, t, done, cur, u, trained, op):
        g = self.grid
        col = np.tile(np.arange(DEPTH), N_ACT)
        c0 = ramp_colors(self.grad0, GRAD_RANGE, DARK, GRAD_HI)
        c1 = ramp_colors(self.grad1, GRAD_RANGE, DARK, GRAD_HI)
        g.fill(c0 + (c1 - c0) * trained)
        shown = col >= DEPTH - done
        if 0 <= cur < DEPTH and u >= 0.5:
            shown |= col == DEPTH - 1 - cur
        if t >= self.Tm["sweep"][1]:
            shown[:] = True
        g.show(shown)
        g.cage = op
        g.fade(op)
        white = np.zeros((N_ACT, DEPTH), np.float32)
        if 0 <= cur < DEPTH:
            white[:, DEPTH - 1 - cur] = 0.6 * (1.0 - smoothstep(0.7, 1.0, u))
        g.block.white[:, :, 0] = white
        if op <= 0.01:
            return
        # row labels, column labels, read-outs
        for r, a in enumerate(ACTS):
            lo, hi = g.block.box((r, r + 1), (0, DEPTH), (0, 1))
            x0, y0, _x1, y1 = screen_box(self.cam, lo, hi)
            y = (y0 + y1) / 2 / self.cam.height
            sc.text((x0 - 8) / self.cam.width, y, a.label, layout.MIN_TEXT, "regular", a.ink,
                    op * self._cap, "rm")
            if shown.reshape(N_ACT, DEPTH)[r, 0]:
                grad = self.grad0 + (self.grad1 - self.grad0) * trained
                ratio = float(grad[r, 0] / grad[r, -1])
                text = "layer 1 / layer %d: %s" % (DEPTH, pow10(ratio))
                ink = layout.STRONG_INK if ratio < 1e-2 else layout.CAPTION_INK
                sc.text(self.fr(self.P(NOTE_X, 0.0))[0], y, text, layout.MIN_TEXT, "regular",
                        ink, op * self._cap, "lm")
        for l in range(DEPTH):
            lo, hi = g.block.box((0, N_ACT), (l, l + 1), (0, 1))
            x0, _y0, x1, y1 = screen_box(self.cam, lo, hi)
            sc.text((x0 + x1) / 2 / self.cam.width, (y1 + 6) / self.cam.height, "%d" % (l + 1),
                    layout.MIN_TEXT, "regular", layout.NOTE_INK, op * self._cap, "mt")
            if l == 0:
                sc.text((x0 - 10) / self.cam.width, (y1 + 6) / self.cam.height, "layer",
                        layout.MIN_TEXT, "regular", layout.NOTE_INK, op * self._cap, "rt")
        self.caption(sc, g, "the gradient reaching each layer, log colour scale (10%s dark, "
                     "10%s lit)" % (sup(int(GRAD_RANGE[0])), sup(int(GRAD_RANGE[1]))), op,
                     above=True,
                     size=layout.MIN_TEXT, max_width=0.6)
        if 0 <= cur < DEPTH and t < self.Tm["sweep"][1]:
            l = DEPTH - 1 - cur
            lo, hi = g.block.box((0, N_ACT), (l, l + 1), (0, 1))
            sc.wire_box(lo, hi, inflate=0.12 * GRID_SCALE, color=FRAME, width=1.6, opacity=0.95,
                        bias=0.1)
            sc.probe(("sweep",), (cur * 20 + u * 20, 0, 0), tol=10.0)

    # -- the slope plots ------------------------------------------------------------------
    def _plots(self, sc, t, trained, op):
        if t < self.Tm["slopes"][0] or op <= 0.01:
            return
        for k, a in enumerate(ACTS):
            u = remap(t, self.Tm["slopes"][0] + k * PLOT_DT,
                      self.Tm["slopes"][0] + (k + 1) * PLOT_DT)
            if u <= 0:
                continue
            x0 = PLOT_X0 + k * (PLOT_W + PLOT_GAP)
            box = ((x0, x0 + PLOT_W), PLOT_Y)
            hist = self.hist0[k] + (self.hist1[k] - self.hist0[k]) * trained
            flat = self.flat0[k] + (self.flat1[k] - self.flat0[k]) * trained
            slope_plot(self, sc, box, a, hist, self.flat_bins[k], op, u, HIST_FULL)
            self.label(sc, self.P((box[0][0] + box[0][1]) / 2, PLOT_Y[1]),
                       "%d%% flat" % round(100 * flat), layout.MIN_TEXT, anchor="mt", dy=28,
                       ink=layout.NOTE_INK, opacity=op * smoothstep(0.8, 1.0, u))
        self.label(sc, self.P(0.5, PLOT_Y[0]), "f′ over layer %d's pre-activations on %d test "
                   "images (tinted bins: |f′| < %g, a flat part)" % (HIST_LAYER + 1, HIST_N, FLAT),
                   layout.MIN_TEXT, anchor="mb", dy=-5, opacity=op)

    # ------------------------------------------------------------------ words
    def _words(self, t, done, cur, u, reset):
        T = self.Tm
        r0, r1 = self.ratio(self.grad0), self.ratio(self.grad1)
        i = {a.key: k for k, a in enumerate(ACTS)}
        if reset > 0 or t < T["sweep"][0]:
            return ("an %d-layer MLP (784 → %d × %d → 10) under each activation, untrained: how "
                    "much of the loss's gradient reaches each layer on its way back"
                    % (DEPTH, DEEP8_SIZES[1], DEPTH))
        if t < T["slopes"][0]:
            l = DEPTH - 1 - cur if 0 <= cur < DEPTH else 0
            if t >= T["sweep"][1]:
                return ("at layer 1 the sigmoid network's gradient is %s of layer %d's (about "
                        "%.1f× smaller a layer); ReLU's %s, GELU's %s, ELU's %s"
                        % (pow10(r0[i["sigmoid"]]), DEPTH, r0[i["sigmoid"]] ** (-1 / (DEPTH - 1)),
                           pow10(r0[i["relu"]]), pow10(r0[i["gelu"]]), pow10(r0[i["elu"]])))
            g = self.grad0[:, l]
            return ("back through layer %d: gradient RMS %s (sigmoid), %s (tanh), %s (ReLU), %s "
                    "(GELU), %s (ELU)" % (l + 1, pow10(g[i["sigmoid"]]), pow10(g[i["tanh"]]),
                                          pow10(g[i["relu"]]), pow10(g[i["gelu"]]),
                                          pow10(g[i["elu"]])))
        if t < T["train"][0]:
            return ("why: each layer multiplies the gradient by f′; sigmoid's is at most 0.25, "
                    "ReLU's exactly 1 or 0, GELU's and SiLU's about ½ near 0, where layer %d's "
                    "values sit" % (HIST_LAYER + 1))
        best = [k for k, a in enumerate(ACTS) if a.key != "sigmoid"]
        lo, hi = round(100 * self.acc[best].min()), round(100 * self.acc[best].max())
        rest = "%d%%" % lo if lo == hi else "%d–%d%%" % (lo, hi)
        return ("after %s Adam steps no gradient shrinks on the way back any more (sigmoid's "
                "layer 1 / layer %d: %s → %s): Adam rescales each weight's step. Test accuracy: "
                "sigmoid %d%%, the rest %s"
                % (f"{DEEP8['steps']:,}", DEPTH, pow10(r0[i["sigmoid"]]), pow10(r1[i["sigmoid"]]),
                   round(100 * self.acc[i["sigmoid"]]), rest))

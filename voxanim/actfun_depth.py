"""Activations and depth: the 20-layer network of the ``initialisation``
animation under each of the seven activations with its matching init, no
normalisation, trained 400 steps of SGD.

    784 -> 100 x 20 -> 10; Xavier weights for sigmoid and tanh, He for the rest;
    SGD at learning rate 0.1, batches of 64, on 10,000 MNIST images

The grid is seven rows (the activations) by twenty columns (the layers); a
cell's brightness is the std of that layer's activations on the first
minibatch, on a log colour scale:

1. **network** -- the grid and what its rows and columns are,
2. **forward** -- layer by layer: tanh's std settles near 0.15 and ReLU's
   and Leaky ReLU's near 0.25 (Xavier and He do their job); sigmoid's sits
   at 0.1 from the start (everything squashed towards ½); GELU's and SiLU's
   shrink by a third a layer, to 0.003 by layer 20 (He weights overshoot
   for a function whose slope is ½ near 0, where the values are); ELU's
   grow, to 1.7,
3. **training** -- the seven loss curves over 400 SGD steps: tanh reaches
   89% test accuracy, ReLU 73%, Leaky ReLU 57%; sigmoid, GELU and SiLU never
   leave chance; ELU's loss overflows at step 5,
4. **inside** -- layer 10's activations after training, binned: sigmoid's
   all near ½ and GELU's and SiLU's all near 0 (nothing downstream can tell
   the digits apart), ReLU's half zeros, tanh's a bell.  Every one of the
   stalled networks is what skip connections (``skip-connections``) and
   normalisation (``normalisation``) were invented for.

Every number is from the recorded runs of :func:`voxanim.actfun.deep20_runs`;
the histograms apply the final parameters to 256 test images.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from .actfun import ACTS, DEEP20, DEEP20_SIZES, data, deep20_runs, hidden
from .actfun_plots import PANEL, PANEL_EDGE, ramp_colors
from .conv import Tempo
from .diagram import Diagram
from .easing import ease_in_out, remap, smoothstep
from .losschart import LossChart, pow10, sup
from .netanim import ACT_HI, screen_box
from .sweep import Sweep

DEPTH, WIDTH = len(DEEP20_SIZES) - 2, DEEP20_SIZES[1]
N_ACT = len(ACTS)
HIST_LAYER = 9                                  # 0-based: layer 10 is binned
HIST_N = 256                                    # test images binned
HBINS, HRANGE, HFULL = 20, 2.0, 0.5             # bins over (-2, 2); a bin with half fills a panel
ACT_RANGE = (-4.0, 0.5)                         # log10 activation std: dark .. lit
DARK = C.rgb("#3a3f44")
GRID_AT, GRID_SCALE, PITCH = (0.44, 0.47), 2.0, (1.4, 1.15)
NOTE_X = 0.735                                  # the per-row read-outs
CHART = ((0.08, 0.60), (0.26, 0.54))            # the training chart
LEGEND = (0.65, 0.29)
SMOOTH = 8                                      # batches the drawn loss is averaged over
PLOT_X0, PLOT_W, PLOT_GAP = 0.005, 0.125, 0.0155
PLOT_Y = (0.68, 0.84)
FRAME = (0.96, 0.96, 0.96)
CHANCE = np.log(10.0)
TRAIN_T, INSIDE_T = 12.0, 11.0
HIST_INK = (0.55, 0.57, 0.60)


def histogram(a):
    """Fractions of the activations ``a`` in :data:`HBINS` bins over
    ``(-HRANGE, HRANGE)``, values beyond the range in the end bins; all zero
    when ``a`` is not finite (a diverged network)."""
    v = np.asarray(a, np.float64).ravel()
    if not np.isfinite(v).all():
        return np.zeros(HBINS)
    v = np.clip(v, -HRANGE + 1e-9, HRANGE - 1e-9)
    counts, _edges = np.histogram(v, HBINS, (-HRANGE, HRANGE))
    return counts / max(1, v.size)


def diverged(run):
    """The first training step whose minibatch loss was not finite, or None."""
    bad = np.nonzero(~np.isfinite(run.batch_loss))[0]
    return int(bad[0]) if len(bad) else None


class ActivationDepth(Diagram):
    """The activation-std grid, the training curves, layer 10 inside (see the module)."""

    PHASES = ("network", "forward", "training", "inside")

    def __init__(self, runs=None, x=None):
        super().__init__("Activations and depth", "")
        if runs is None:
            dat = data()
            runs, x = deep20_runs(dat=dat), dat.x_test[:HIST_N]
        self.runs = runs
        self.hold = 2.5
        self.act = np.array([r.act_std[0] for r in runs])             # (7, 20) at step 0
        self.acc = np.array([float(r.test_acc[-1]) for r in runs])
        self.diverged = [diverged(r) for r in runs]
        kernel = np.ones(SMOOTH) / SMOOTH
        self.curves = []                          # per run: (steps, drawn loss) up to divergence
        for r in runs:
            bl = r.batch_loss
            ok = np.isfinite(bl)
            n = int(np.argmin(ok)) if not ok.all() else len(bl)
            if n >= SMOOTH:
                self.curves.append((np.arange(n - SMOOTH + 1) + SMOOTH,
                                    np.convolve(bl[:n], kernel, "valid")))
            else:
                self.curves.append((np.arange(n), bl[:n]))
        self.hist, self.mean_abs = [], []
        for r in runs:
            _ys, As = hidden(r.spec, r.P, x)
            self.hist.append(histogram(As[HIST_LAYER]))
            self.mean_abs.append(float(np.abs(As[HIST_LAYER]).mean()))
        self.grid = self.sheet((N_ACT, DEPTH), GRID_AT, GRID_SCALE, "grid", pitch=PITCH)
        self.chart = LossChart(CHART, (0, DEEP20["steps"]), (0.0, 3.0))

        Tm, t = {}, 0.3
        Tm["network"] = (t, t + 4.5)
        t += 4.5
        self.fwd = Sweep(t, DEPTH, Tempo(slow=0.55, fast=0.14, ramp_in=3.0, ramp_out=2.0))
        Tm["sweep"] = (t, self.fwd.t1)
        t = self.fwd.t1 + 3.0
        Tm["train"] = (t, t + TRAIN_T)
        t += TRAIN_T + 0.5
        Tm["inside"] = (t, t + INSIDE_T)
        t += INSIDE_T
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["sweep"][0], Tm["train"][0], Tm["inside"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else self.step_of(t)
        intro = smoothstep(0.0, 0.6, t)
        done, cur, u = self.fwd.at(t)
        train = self.progress("train", t)
        inside = smoothstep(0.0, 0.1, self.progress("inside", t))
        grid_op = (1.0 - smoothstep(0.0, 0.12, train)) * back * intro
        self._grid(sc, t, done, cur, u, grid_op)
        self._chart(sc, train, back)
        self._inside(sc, t, inside * back)
        self.step_indicator(sc, ph, self.PHASES)
        self.subtitle = self._words(t, done, cur, train, reset)

    # -- the grid -------------------------------------------------------------------------
    def _grid(self, sc, t, done, cur, u, op):
        g = self.grid
        if op <= 0.01:
            g.visible = False
            return
        col = np.tile(np.arange(DEPTH), N_ACT)
        g.fill(ramp_colors(self.act, ACT_RANGE, DARK, ACT_HI))
        shown = col < done
        if 0 <= cur < DEPTH and u >= 0.5:
            shown |= col == cur
        if t >= self.Tm["sweep"][1]:
            shown[:] = True
        g.show(shown)
        g.cage = op
        g.fade(op)
        white = np.zeros((N_ACT, DEPTH), np.float32)
        if 0 <= cur < DEPTH:
            white[:, cur] = 0.6 * (1.0 - smoothstep(0.7, 1.0, u))
        g.block.white[:, :, 0] = white
        last = done if done < DEPTH else DEPTH          # the layer the read-outs quote
        fx = self.fr(self.P(NOTE_X, 0.0))[0]
        for r, a in enumerate(ACTS):
            lo, hi = g.block.box((r, r + 1), (0, DEPTH), (0, 1))
            x0, y0, _x1, y1 = screen_box(self.cam, lo, hi)
            y = (y0 + y1) / 2 / self.cam.height
            sc.text((x0 - 8) / self.cam.width, y, a.label, layout.MIN_TEXT, "regular", a.ink,
                    op * self._cap, "rm")
            if last > 0:
                sc.text(fx, y, "layer %d: std %s" % (last, pow10(self.act[r, last - 1])),
                        layout.MIN_TEXT, "regular", layout.CAPTION_INK, op * self._cap, "lm")
        for l in range(DEPTH):
            if l % 5 and l:
                continue
            lo, hi = g.block.box((0, N_ACT), (l, l + 1), (0, 1))
            x0, _y0, x1, y1 = screen_box(self.cam, lo, hi)
            sc.text((x0 + x1) / 2 / self.cam.width, (y1 + 6) / self.cam.height,
                    "layer 1" if l == 0 else "%d" % (l + 1), layout.MIN_TEXT, "regular",
                    layout.NOTE_INK, op * self._cap, "mt")
        self.caption(sc, g, "activation std per layer, first minibatch (log scale: 10%s dark, "
                     "%.0f lit)" % (sup(int(ACT_RANGE[0])), 10 ** ACT_RANGE[1]), op, above=True,
                     size=layout.MIN_TEXT, max_width=0.7)
        if 0 <= cur < DEPTH and t < self.Tm["sweep"][1]:
            lo, hi = g.block.box((0, N_ACT), (cur, cur + 1), (0, 1))
            sc.wire_box(lo, hi, inflate=0.12 * GRID_SCALE, color=FRAME, width=1.6, opacity=0.95,
                        bias=0.1)
            sc.probe(("sweep",), (cur * 20 + u * 20, 0, 0), tol=10.0)

    # -- the chart ------------------------------------------------------------------------
    def _chart(self, sc, train, back):
        op = smoothstep(0.12, 0.3, train) * back * self._cap
        if op <= 0.01:
            return
        place = self.P
        steps_total = DEEP20["steps"]
        upto = SMOOTH + (steps_total - SMOOTH) * smoothstep(0.0, 0.9, train)
        self.chart.axes(sc, self.cam, place, op, "SGD steps (batch loss, mean of %d)" % SMOOTH,
                        "loss", [(s, str(s)) for s in (0, 100, 200, 300, 400)],
                        [(v, "%g" % v) for v in (0, 1, 2, 3)])
        self.chart.hline(sc, place, CHANCE, op)
        legend = []
        for i, a in enumerate(ACTS):
            steps, c = self.curves[i]
            if len(c) >= 2:
                end = self.chart.curve(sc, place, steps, c, upto, a.ink, op, width=1.4)
                if self.diverged[i] is not None and end is not None and upto >= steps[-1]:
                    self.label(sc, end, "diverged", layout.MIN_TEXT, anchor="lm", dx=9, dy=12,
                               ink=a.ink, opacity=op)
            if self.diverged[i] is not None:
                legend.append(("%s: loss overflows at step %d" % (a.label, self.diverged[i]),
                               a.ink))
            else:
                legend.append(("%s: %d%% test accuracy" % (a.label, round(100 * self.acc[i])),
                               a.ink))
        self.chart.legend(sc, self.cam, legend, op * smoothstep(0.7, 1.0, train), LEGEND,
                          step=0.040)
        sc.probe(("train",), (upto * 0.5, 0, 0), tol=6.0)

    # -- inside the trained networks ---------------------------------------------------
    def _inside(self, sc, t, op):
        op *= self._cap
        if op <= 0.01:
            return
        u = self.progress("inside", t)
        edges = np.linspace(-HRANGE, HRANGE, HBINS + 1)
        for k, a in enumerate(ACTS):
            x0 = PLOT_X0 + k * (PLOT_W + PLOT_GAP)
            box = ((x0, x0 + PLOT_W), PLOT_Y)
            (bx0, bx1), (by0, by1) = box
            pa, pb = self.P(bx0, by0), self.P(bx1, by1)
            w, h = self.px(pb) - self.px(pa)
            self.rect(sc, (pa + pb) / 2, w, h, PANEL, op, border=PANEL_EDGE, border_w=1.0,
                      radius=4, lift=0.1)
            grow = smoothstep(0.1, 0.6, u)
            for b in range(HBINS):
                hh = min(1.0, self.hist[k][b] / HFULL) * grow
                if hh <= 0:
                    continue
                fx0 = bx0 + (bx1 - bx0) * (edges[b] + HRANGE) / (2 * HRANGE)
                fx1 = bx0 + (bx1 - bx0) * (edges[b + 1] + HRANGE) / (2 * HRANGE)
                p0, p1 = self.P(fx0, by1 - (by1 - by0) * hh), self.P(fx1, by1)
                bw, bh = self.px(p1) - self.px(p0)
                self.rect(sc, (p0 + p1) / 2, max(bw - 1.0, 1.0), bh, a.ink, 0.9 * op, radius=0.0,
                          lift=0.15)
            zero = self.P(bx0 + (bx1 - bx0) / 2, by1)
            sc.line(zero, self.P(bx0 + (bx1 - bx0) / 2, by0), width=0.8, color=HIST_INK,
                    opacity=0.5 * op, bias=0.05)
            self.label(sc, self.P((bx0 + bx1) / 2, by1), a.label, layout.MIN_TEXT, anchor="mt",
                       dy=6, ink=a.ink, opacity=op)
            note = ("diverged" if self.diverged[k] is not None
                    else "mean |a| %s" % pow10(self.mean_abs[k]))
            self.label(sc, self.P((bx0 + bx1) / 2, by1), note, layout.MIN_TEXT, anchor="mt",
                       dy=24, ink=layout.NOTE_INK, opacity=op * smoothstep(0.6, 0.9, u))
        self.label(sc, self.P(0.5, PLOT_Y[0]), "layer %d's activations after training, binned over "
                   "−%g to %g, on %d test images" % (HIST_LAYER + 1, HRANGE, HRANGE, HIST_N),
                   layout.MIN_TEXT, anchor="mb", dy=-5, opacity=op)

    # ------------------------------------------------------------------ words
    def _words(self, t, done, cur, train, reset):
        T = self.Tm
        i = {a.key: n for n, a in enumerate(ACTS)}
        if reset > 0 or t < T["sweep"][0]:
            return ("the %d-layer MLP of the initialisation animation (784 → %d × %d → 10) under "
                    "each activation with its matching init, no normalisation"
                    % (DEPTH, WIDTH, DEPTH))
        if t < T["train"][0]:
            l = min(max(done, 1), DEPTH)
            a = self.act[:, l - 1]
            if t >= T["sweep"][1]:
                return ("by layer %d: tanh %s and ReLU %s hold their scale; sigmoid's %s is "
                        "squashed; GELU's and SiLU's shrink to %s and %s; ELU's grow to %s"
                        % (DEPTH, pow10(a[i["tanh"]]), pow10(a[i["relu"]]), pow10(a[i["sigmoid"]]),
                           pow10(a[i["gelu"]]), pow10(a[i["silu"]]), pow10(a[i["elu"]])))
            return ("forward through the untrained network, layer %d: std %s (tanh), %s (ReLU), "
                    "%s (sigmoid), %s (GELU), %s (ELU)"
                    % (l, pow10(a[i["tanh"]]), pow10(a[i["relu"]]), pow10(a[i["sigmoid"]]),
                       pow10(a[i["gelu"]]), pow10(a[i["elu"]])))
        if t < T["inside"][0]:
            acc = self.acc
            stalled = [a.label for n, a in enumerate(ACTS) if acc[n] < 0.2
                       and self.diverged[n] is None]
            return ("%d SGD steps at learning rate %g: tanh reaches %d%% test accuracy, ReLU %d%%, "
                    "Leaky ReLU %d%%; %s stay at chance; ELU's loss overflows at step %d"
                    % (DEEP20["steps"], DEEP20["lr"], round(100 * acc[i["tanh"]]),
                       round(100 * acc[i["relu"]]), round(100 * acc[i["leaky_relu"]]),
                       ", ".join(stalled), self.diverged[i["elu"]]))
        return ("layer %d after training: sigmoid's activations all ≈ ½ and GELU's and SiLU's all "
                "≈ 0, so nothing downstream can tell the digits apart; skip connections or "
                "normalisation would carry all of them" % (HIST_LAYER + 1))

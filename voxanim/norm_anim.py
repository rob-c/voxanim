"""Batch norm and layer norm: one hidden layer of a deep MLP, mid-training.

    784 -> 24 x 12 (tanh) -> 10, Adam at learning rate 0.01 on 10,000 MNIST images

Three copies of the network are trained: plain, with batch norm and with
layer norm before every tanh.  The sheet in the middle is layer 6's
pre-activations ``z`` for one real minibatch (32 examples x 24 units) at
step 500 of the batch-norm network's run:

1. **minibatch** -- the sheet, and the histogram of its values,
2. **batch norm** -- down each column, over the 32 examples: the column's
   mean and std appear under it and its cells become ``(z - μ) / σ``,
3. **layer norm** -- the same layer of the layer-norm network: across each
   row, over the 24 units, the row's mean and std appear to its right and
   the row is standardised,
4. **scale & shift** -- the learned ``γ`` and ``β`` of that layer, one per
   unit, applied: ``y = γ ẑ + β``, which is what tanh sees,
5. **training** -- the three training curves: at this learning rate the
   plain network's loss climbs back to chance; batch norm reaches 93% test
   accuracy, layer norm 80%.  (At learning rate 0.003 all three learn: the
   normalisation buys the higher rate.)

Every number is from the recorded runs of :mod:`voxanim.deepnet` (the probe
layer's minibatch values and parameters at each recorded step); the
normalisation drawn is recomputed from the recorded ``z`` and agrees with
the recorded network output (a test checks it).
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from .conv import Tempo
from .deepnet import EPS, Spec, subset, train
from .easing import ease_in_out, remap, smoothstep
from .losschart import LossChart, pow10
from .netanim import NetAnimation, amax, div_colors, lit_colors
from .sweep import Sweep
from .text import num

DEPTH, WIDTH, BATCH = 12, 24, 32
SIZES = (784,) + (WIDTH,) * DEPTH + (10,)
NETS = (("plain", Spec(SIZES, "tanh", "xavier")),
        ("batch norm", Spec(SIZES, "tanh", "xavier", norm="batch")),
        ("layer norm", Spec(SIZES, "tanh", "xavier", norm="layer")))
PROBE = 5                                       # hidden layer recorded (0-based): layer 6
TRAIN = dict(steps=1500, batch=BATCH, lr=0.01, opt="adam", every=50, probe_layer=PROBE)
N_TRAIN, N_TEST = 10000, 2000
CHECK_STEP = 500                                # the recorded step the sheet shows
LN_INK = C.rgb("#8952eb")
INKS = ((0.75, 0.22, 0.22), tuple(float(v) for v in C.INPUT), tuple(float(v) for v in LN_INK))

Z_AT = (0.31, 0.585)                            # frame position of the sheet
PPU = 10.5
SCALE = 0.9
GAP = 1.4                                       # between the sheet and its statistics, cells
HIST_AT = ((0.745, 0.45), (0.745, 0.70))        # before / after histograms
HBINS, HROWS, HRANGE, HFULL = 16, 12, 4.0, 0.35  # bins over (-4, 4); a bin of 35% fills a column
HSCALE, HPITCH = 0.7, (1.0, 1.15)
CHART = ((0.56, 0.93), (0.33, 0.78))            # the training chart ((x0, x1), (top, bottom))
LEGEND = (0.12, 0.50)                           # where the sheet was: it has faded by then
SMOOTH = 20
CHANCE = np.log(10.0)
SWITCH = 0.8                                    # seconds the sheet takes to swap networks
SHIFT = 1.5                                     # seconds the scale-and-shift takes


class _Box:
    """A world box with the :meth:`~voxanim.netanim.Sheet.bbox` interface, so a
    caption can sit beside two sheets at once."""

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def bbox(self, offsets=True):
        return self.lo, self.hi


def runs(cache_dir=None):
    """The three recorded runs: plain, batch norm, layer norm."""
    data = subset(N_TRAIN, N_TEST, seed=0, cache_dir=cache_dir)
    kw = dict(TRAIN)
    if cache_dir:
        kw["cache_dir"] = cache_dir
    return [train(spec, data, **kw) for _name, spec in NETS]


def standardise(z, axis):
    """``(ẑ, μ, σ)`` of ``z`` over ``axis`` with the model's epsilon."""
    mu = z.mean(axis, keepdims=True)
    sd = np.sqrt(z.var(axis, keepdims=True) + EPS)
    return (z - mu) / sd, mu, sd


def mean_text(v):
    """A mean to two decimals, without a signed zero."""
    v = float(v)
    return num(0.0 if abs(v) < 0.005 else v)


def histogram(values):
    """Fraction of ``values`` in each of HBINS bins over (-HRANGE, HRANGE)."""
    counts, _ = np.histogram(np.asarray(values).ravel(), HBINS, (-HRANGE, HRANGE))
    return counts / max(1, len(np.asarray(values).ravel()))


class Normalisation(NetAnimation):
    """One layer's minibatch through batch norm and layer norm (see the module)."""

    PHASES = ("minibatch", "batch norm", "layer norm", "scale & shift", "training")

    def __init__(self, runs):
        super().__init__("Batch norm and layer norm", "", PPU)
        self.runs = runs
        self.hold = 2.5
        _plain, bn, ln = runs
        k = int(np.searchsorted(bn.steps, CHECK_STEP))
        self.step = int(bn.steps[k])
        # the batch-norm network's layer: standardised down the columns
        self.z_bn = np.asarray(bn.probe["z"][k], np.float64)
        self.zn_bn, self.mu_j, self.sd_j = standardise(self.z_bn, 0)
        # the layer-norm network's layer: across the rows, then scaled and shifted
        self.z_ln = np.asarray(ln.probe["z"][k], np.float64)
        self.zn_ln, self.mu_i, self.sd_i = standardise(self.z_ln, 1)
        self.gamma, self.beta = (np.asarray(ln.probe[n][k], np.float64) for n in ("g", "beta"))
        self.y_ln = self.gamma * self.zn_ln + self.beta
        self.zmax = amax(self.z_bn)
        kernel = np.ones(SMOOTH) / SMOOTH
        self.curves = [np.convolve(r.batch_loss, kernel, "valid") for r in runs]
        self.acc = [float(r.test_acc[-1]) for r in runs]
        self.chart = LossChart(CHART, (0, TRAIN["steps"]), (0.0, 3.0))

        # ---- geometry: the sheet, its column statistics below, row statistics right ----
        self.Z = self.sheet((BATCH, WIDTH), Z_AT, SCALE, "z")
        c = self.Z.center
        s = SCALE
        self.cols = [self.sheet((1, WIDTH), Z_AT, s, "col%d" % n) for n in range(2)]
        for n, sh in enumerate(self.cols):
            sh.block.place(c + np.array([0.0, 0.0, -(BATCH / 2 + GAP + 0.5 + n) * s]))
        self.rowstats = [self.sheet((BATCH, 1), Z_AT, s, "row%d" % n) for n in range(2)]
        for n, sh in enumerate(self.rowstats):
            sh.block.place(c + np.array([(WIDTH / 2 + GAP + 0.5 + n) * s, 0.0, 0.0]))
        self.hists = [self.sheet((HROWS, HBINS), at, HSCALE, "hist%d" % n, pitch=HPITCH)
                      for n, at in enumerate(HIST_AT)]

        # ---- timeline -----------------------------------------------------------------
        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("batch", 4.5, 0.3)
        self.bn_sweep = Sweep(t, WIDTH, Tempo(slow=0.5, fast=0.1, ramp_in=3.0, ramp_out=2.0))
        add("bn", self.bn_sweep.duration, 1.8)
        self.ln_sweep = Sweep(t + SWITCH, BATCH, Tempo(slow=0.45, fast=0.08, ramp_in=3.0,
                                                       ramp_out=2.0))
        add("ln", SWITCH + self.ln_sweep.duration, 1.8)
        add("shift", SHIFT + 3.0, 0.5)
        add("train", 10.0, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["bn"][0], Tm["ln"][0], Tm["shift"][0], Tm["train"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ what the sheet shows
    def _state(self, t):
        """``(raw, shown, done_rows, done_cols, swap)`` at time ``t``: the raw
        ``z`` on the sheet, the values drawn, how many rows / columns have been
        standardised, and the crossfade of the network swap (1 = steady)."""
        if t < self.Tm["ln"][0]:
            done, cur, u = self.bn_sweep.at(t)
            n = done + (1 if 0 <= cur < WIDTH and u >= 0.5 else 0)
            shown = np.array(self.z_bn)
            shown[:, :n] = self.zn_bn[:, :n]
            return self.z_bn, shown, 0, n, 1.0
        x = remap(t, self.Tm["ln"][0], self.Tm["ln"][0] + SWITCH)
        if x < 0.5:
            return self.z_bn, self.zn_bn, 0, WIDTH, 1.0 - 2 * x
        done, cur, u = self.ln_sweep.at(t)
        n = done + (1 if 0 <= cur < BATCH and u >= 0.5 else 0)
        shown = np.array(self.z_ln)
        shown[:n] = self.zn_ln[:n]
        if t >= self.Tm["shift"][0]:
            w = ease_in_out(remap(t, self.Tm["shift"][0], self.Tm["shift"][0] + SHIFT))
            shown = self.zn_ln + (self.y_ln - self.zn_ln) * w
        return self.z_ln, shown, n, 0, min(1.0, 2 * x - 1)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        self.subtitle = [
            "layer %d of a %d-layer tanh MLP (%d units) at step %d of training: one minibatch's "
            "pre-activations z, %d examples × %d units" % (PROBE + 1, DEPTH, WIDTH, self.step,
                                                           BATCH, WIDTH),
            "batch norm: down each column, over the %d examples: ẑ = (z − μ) / σ per unit; "
            "every unit now has mean 0 and variance 1 across the batch" % BATCH,
            "layer norm: the same layer of the network trained with it; across each row, over "
            "the %d units: μ and σ per example, no batch needed" % WIDTH,
            "then a learned scale and shift per unit, y = γ ẑ + β (γ from %.2f to %.2f, β from "
            "%s to %s): the network sets the range tanh sees"
            % (self.gamma.min(), self.gamma.max(), pow10(self.beta.min(), 2),
               pow10(self.beta.max(), 2)),
            "training at learning rate %g (Adam): the plain network's loss climbs back to chance; "
            "batch norm reaches %d%% test accuracy, layer norm %d%%"
            % (TRAIN["lr"], round(100 * self.acc[1]), round(100 * self.acc[2])),
        ][ph]
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        intro = smoothstep(0.0, 0.6, t)
        train = self.progress("train", t)
        sheet_op = (1.0 - smoothstep(0.0, 0.12, train)) * back * intro
        raw, shown, rows_done, cols_done, swap = self._state(t)
        self._sheet(sc, t, shown, rows_done, cols_done, swap, sheet_op)
        self._column_stats(sc, t, cols_done, swap, sheet_op)
        self._row_stats(sc, t, rows_done, swap, sheet_op)
        self._histograms(sc, t, raw, shown, rows_done, cols_done, sheet_op)
        self._chart(sc, train, back)
        self.step_indicator(sc, ph, self.PHASES)

    def _sheet(self, sc, t, shown, rows_done, cols_done, swap, op):
        """The minibatch sheet, with the row or column at work framed white."""
        Z = self.Z
        Z.fill(div_colors(shown.ravel(), self.zmax))
        Z.cage = op
        Z.fade(op * swap)
        white = np.zeros((BATCH, WIDTH), np.float32)
        if t < self.Tm["ln"][0]:
            _d, cur, u = self.bn_sweep.at(t)
            if 0 <= cur < WIDTH:
                white[:, cur] = 0.6 * (1 - smoothstep(0.7, 1.0, u))
        else:
            _d, cur, u = self.ln_sweep.at(t)
            if 0 <= cur < BATCH:
                white[cur] = 0.6 * (1 - smoothstep(0.7, 1.0, u))
        Z.block.white[:, :, 0] = white
        net = "batch-norm" if t < self.Tm["ln"][0] + SWITCH / 2 else "layer-norm"
        self.caption(sc, Z, "%d units of layer %d, %s network" % (WIDTH, PROBE + 1, net), op,
                     above=True)
        self.caption_left(sc, Z, "%d examples" % BATCH, op)

    def _column_stats(self, sc, t, cols_done, swap, op):
        """Under the sheet: μ and σ of each column (batch norm), later γ and β."""
        shift = t >= self.Tm["shift"][0]
        if shift:
            vis = smoothstep(0.0, 0.5, self.progress("shift", t))
            top, bottom = lit_colors(self.gamma, amax(self.gamma)), div_colors(self.beta,
                                                                              amax(self.beta))
            label = "γ, β per unit"
            mask = np.ones(WIDTH, bool)
        else:
            vis = 1.0 if t < self.Tm["ln"][0] else 1.0 - smoothstep(0.0, 0.5, remap(
                t, self.Tm["ln"][0], self.Tm["ln"][0] + SWITCH))
            top = div_colors(self.mu_j.ravel(), self.zmax)
            bottom = lit_colors(self.sd_j.ravel(), amax(self.sd_j))
            label = "μ, σ per unit"
            mask = np.arange(WIDTH) < cols_done
        for sh, colors in zip(self.cols, (top, bottom)):
            sh.fill(colors)
            sh.show(mask)
            sh.cage = op * vis if mask.any() else 0.0
            sh.fade(op * vis)
        if mask.any():
            lo, hi = self.cols[1].bbox()[0], self.cols[0].bbox()[1]
            self.caption_left(sc, _Box(lo, hi), label, op * vis)

    def _row_stats(self, sc, t, rows_done, swap, op):
        """Right of the sheet: μ and σ of each row (layer norm)."""
        mask = np.arange(BATCH) < rows_done
        vis = swap if t >= self.Tm["ln"][0] + SWITCH / 2 else 0.0
        for sh, values, kind in zip(self.rowstats, (self.mu_i, self.sd_i), ("mu", "sd")):
            colors = div_colors(values.ravel(), self.zmax) if kind == "mu" else \
                lit_colors(values.ravel(), amax(self.sd_i))
            sh.fill(colors)
            sh.show(mask)
            sh.cage = op * vis if mask.any() else 0.0
            sh.fade(op * vis)
        if mask.any() and vis > 0.01:
            lo = self.rowstats[0].bbox()[0]
            hi = self.rowstats[1].bbox()[1]
            self.caption_side(sc, (lo, hi), "μ, σ per example", op * vis)

    def _histograms(self, sc, t, raw, shown, rows_done, cols_done, op):
        """The values' distribution before and after: live from the sheet."""
        if rows_done:
            after = shown[:rows_done]
        elif cols_done:
            after = shown[:, :cols_done]
        else:
            after = None
        for sh, values, ink, name in ((self.hists[0], raw, C.NEUTRAL, "before"),
                                      (self.hists[1], after, C.INPUT, "after")):
            if values is None or values.size == 0:
                sh.visible = False
                continue
            frac = histogram(values)
            present = np.zeros((HROWS, HBINS), bool)
            for b in range(HBINS):
                h = int(round(min(1.0, frac[b] / HFULL) * HROWS))
                present[HROWS - h:, b] = True
            centres = (np.arange(HBINS) + 0.5) / HBINS * 2 * HRANGE - HRANGE
            tint = div_colors(np.repeat(centres, HROWS).reshape(HBINS, HROWS).T.ravel(), HRANGE)
            sh.fill(tint if name == "after" else np.tile(C.rgb("#8a8f94"), (HROWS * HBINS, 1)))
            sh.show(present.ravel())
            sh.cage = op
            sh.fade(op)
            label = name if name == "before" else ("after y = γ ẑ + β"
                                                   if t >= self.Tm["shift"][0] else "after")
            self.caption(sc, sh, "%s: mean %s, std %.2f" % (label, mean_text(values.mean()),
                                                               values.std()), op, size=0.028)
            _ = ink
        if op > 0.01:
            for sh in self.hists:
                if not sh.visible:
                    continue
                lo, hi = sh.block.box((HROWS - 1, HROWS), (0, HBINS), (0, 1))
                for x, text, anchor in ((lo[0], "−%g" % HRANGE, "rm"), (hi[0], "+%g" % HRANGE,
                                                                        "lm")):
                    p = self.cam.project([x, hi[1], (lo[2] + hi[2]) / 2])
                    sc.text((p[0] + (-8 if anchor == "rm" else 8)) / self.cam.width,
                            p[1] / self.cam.height, text, layout.MIN_TEXT, "regular",
                            layout.NOTE_INK, op * self._cap, anchor)

    # ------------------------------------------------------------------ the chart
    def _chart(self, sc, train, back):
        """The three training-loss curves."""
        op = smoothstep(0.12, 0.3, train) * back * self._cap
        if op <= 0.01:
            return
        cam, place = self.cam, self.cam.frac
        n = len(self.curves[0])
        upto = SMOOTH + (n - 1) * smoothstep(0.0, 0.92, train)
        ticks = [(s, f"{s:,}") for s in (0, 500, 1000, 1500)]
        self.chart.axes(sc, cam, place, op, "steps (batch loss, mean of %d)" % SMOOTH, "loss",
                        ticks, [(v, "%g" % v) for v in (0, 1, 2, 3)])
        self.chart.hline(sc, place, CHANCE, op)
        steps = np.arange(n) + SMOOTH
        for i in range(3):
            self.chart.curve(sc, place, steps, self.curves[i], upto, INKS[i], op, every=4)
        legend = [("%s: %d%% test accuracy" % (NETS[i][0], round(100 * self.acc[i])), INKS[i])
                  for i in range(3)]
        self.chart.legend(sc, cam, legend, op * smoothstep(0.85, 1.0, train), LEGEND)
        sc.probe(("train",), (upto * 0.1, 0, 0), tol=6.0)

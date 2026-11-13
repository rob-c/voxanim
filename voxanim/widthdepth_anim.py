"""Width or depth?  One parameter budget, about 50,000, spent three ways:

    wide     784 ->  64          -> 10     50,890 parameters
    medium   784 ->  52 x 4      -> 10     49,618
    narrow   784 ->  37 x 16     -> 10     50,515

ReLU, He weights, the same 1,500 Adam steps on the same 20,000 MNIST digits
(:func:`voxanim.capacity.width_depth_runs`).  Each network is a band of
units × layers, one above the other:

1. **networks** -- the three bands and their parameter counts,
2. **forward** -- one test digit's activations layer by layer through the
   untrained weights, with each layer's activation std over a minibatch,
3. **backward** -- the gradient of its loss flowing back, log colour scale:
   with He weights it reaches layer 1 of the 16-layer network at about
   two thirds of what left layer 16 (the bands of
   :mod:`voxanim.residual_anim`, where Xavier weights lose it entirely),
4. **training** -- the three test-accuracy curves; the bands now show each
   layer's activation std as it is recorded, growing with depth; the legend
   carries the wall-clock time of one training step, measured on the machine
   that built the cache (:func:`voxanim.capacity.step_times`),
5. **verdict** -- all three land within a point of each other; the deep
   narrow network costs several times the time a step for the same FLOPs
   (sixteen small matrix products instead of one): which wins depends on the
   data, the budget and the recipe, and on MNIST width is enough.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from .capacity import N_TRAIN, SHAPES, TRAIN, n_params
from .conv import Tempo
from .deepnet import backward, forward
from .easing import ease_in_out, remap, smoothstep
from .losschart import LossChart, pow10
from .netanim import ACT_HI, NetAnimation, lit_colors
from .residual_anim import ACT_RANGE, DARK, GRAD_HI, GRAD_RANGE, log_scale
from .sweep import Sweep

PPU = 10.5
AZIMUTH, ELEVATION = 14.0, 20.0               # a flatter view: 64-unit rows stay nearly level
EXAMPLE = 0                                   # the test digit of the subset drawn through
INKS = ((0.75, 0.22, 0.22), (0.36, 0.38, 0.42), tuple(float(v) for v in C.INPUT))
BAND_X = 0.345                                # layout x of the widest band's centre
BAND_Y = 0.54                                 # layout y of the stack's centre
BAND_W = 0.36                                 # on-screen width of the widest band
BAND_GAP = 2.5                                # rows between bands
NOTE_X = 0.575                                # frame x of the read-outs right of the bands
NOTE_STEP = 0.046                             # least frame y between two bands' read-outs
STD_RANGE = (-1.5, 0.75)                      # log10 activation std at the dark and the lit end
CHART = ((0.655, 0.95), (0.27, 0.70))         # the accuracy chart ((x0, x1), (top, bottom))
FRAME = (0.96, 0.96, 0.96)
SPAN = 13.0                                   # seconds the training phase takes


def example_through(run, x, label):
    """``(h, dh)``, each a list over hidden layers of ``(width,)`` arrays: one
    digit's activations and the gradient of its loss at each layer, through
    the run's initial weights ``P0``."""
    f = forward(run.spec, run.P0, x[None])
    _G, D = backward(run.spec, run.P0, f, np.eye(10)[[label]])
    L = run.spec.depth
    return [f["h"][l + 1][0] for l in range(L)], [D["dh"][l + 1][0] for l in range(L)]


class WidthVsDepth(NetAnimation):
    """Three networks of one parameter budget, as bands (see the module docstring).

    ``runs`` are the :class:`voxanim.deepnet.Run` s of :data:`voxanim.capacity.SHAPES`,
    ``data`` their :class:`voxanim.deepnet.Data`, ``times`` the seconds per
    training step of each.
    """

    PHASES = ("networks", "forward", "backward", "training", "verdict")

    def __init__(self, runs, data, times):
        super().__init__("Width or depth?", "", PPU, azimuth=AZIMUTH, elevation=ELEVATION)
        self.runs, self.times = runs, np.asarray(times, np.float64)
        self.hold = 2.5
        self.names = [name for name, _sizes in SHAPES]
        self.depths = [r.spec.depth for r in runs]
        self.widths = [r.spec.sizes[1] for r in runs]
        self.params = [n_params(r.spec.sizes) for r in runs]
        cam = self.cam
        # the bands, left-aligned and stacked: the widest sets the voxel scale
        wmax = max(self.widths)
        probe = self.sheet((1, wmax), (BAND_X, BAND_Y), 1.0, "probe")
        self.sheets.remove(probe)
        lo, hi = probe.bbox()
        scale = BAND_W / ((cam.project(hi)[0] - cam.project(lo)[0]) / cam.width)
        centre = cam.frac(BAND_X, BAND_Y)
        total = sum(self.depths) + BAND_GAP * (len(runs) - 1)
        z = centre[2] + total / 2 * scale
        self.bands = []
        for i, (d, w) in enumerate(zip(self.depths, self.widths)):
            s = self.sheet((d, w), (BAND_X, BAND_Y), scale, self.names[i])
            c = centre.copy()
            c[0] -= (wmax - w) / 2 * scale                 # left edges in line
            c[2] = z - d / 2 * scale
            s.block.place(c)
            z -= (d + BAND_GAP) * scale
            self.bands.append(s)
        x, label = data.x_test[EXAMPLE], int(data.y_test[EXAMPLE])
        self.h, self.dh = zip(*[example_through(r, x, label) for r in runs])
        self.act0 = [r.act_std[0] for r in runs]
        self.grad0 = [r.grad_rms[0] for r in runs]
        self.ratio = [float(g[0] / g[-1]) for g in self.grad0]
        self.acc = [float(r.test_acc[-1]) for r in runs]
        self.chart = LossChart(CHART, (1, TRAIN["steps"]), (0.0, 100.0), xlog=True)
        dmax = max(self.depths)

        # ---- timeline -----------------------------------------------------------------
        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("networks", 4.0, 0.3)
        tempo = Tempo(slow=0.6, fast=0.17, ramp_in=3.0, ramp_out=2.0)
        self.fwd = Sweep(t, dmax, tempo)
        add("fwd", self.fwd.duration, 1.8)
        self.bwd = Sweep(t, dmax, tempo)
        add("bwd", self.bwd.duration, 2.0)
        add("train", SPAN, 0.3)
        add("verdict", 6.0, 0.0)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["fwd"][0], Tm["bwd"][0], Tm["train"][0], Tm["verdict"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        intro = smoothstep(0.0, 0.6, t)
        done_f, nf, uf = self.fwd.at(t)
        done_b, nb, ub = self.bwd.at(t)
        train = self.progress("train", t)
        notes = (1.0 - smoothstep(0.0, 0.12, train)) * back
        self._bands(sc, t, done_f, nf, uf, done_b, nb, ub, train, back * intro)
        if 0 <= nf < max(self.depths) and t < self.Tm["fwd"][1]:
            self._layer_frame(sc, nf, forward_=True)
        elif 0 <= nb < max(self.depths) and self.Tm["bwd"][0] <= t < self.Tm["bwd"][1]:
            self._layer_frame(sc, nb, forward_=False)
        self._notes(sc, done_f, done_b, notes)
        self._chart(sc, train, back)
        self.subtitle = self._subtitle(ph)
        self.step_indicator(sc, ph, self.PHASES)

    def _subtitle(self, ph):
        a0 = np.concatenate(self.act0)
        ms = 1e3 * self.times
        spread = 100 * (max(self.acc) - min(self.acc))
        nets = ", ".join("784 → %s → 10 (%s)" % ("%d" % w if d == 1 else "%d × %d" % (w, d),
                                                  f"{p:,}")
                         for w, d, p in zip(self.widths, self.depths, self.params))
        return [
            "about 50,000 parameters three ways: %s; ReLU, He weights" % nets,
            "forward: one test digit through the untrained weights; over a minibatch every "
            "layer's activations have std between %.2f and %.2f (He init)" % (a0.min(), a0.max()),
            "backward: the gradient of its loss flowing back, log colour scale; it reaches layer 1 "
            "of the %d-layer network at %.2f× what left layer %d"
            % (self.depths[-1], self.ratio[-1], self.depths[-1]),
            "training: %s Adam steps (lr %g, batch %d) on %s digits; the bands show each layer's "
            "activation std, live; a step takes %.1f, %.1f and %.1f ms"
            % (f"{TRAIN['steps']:,}", TRAIN["lr"], TRAIN["batch"], f"{N_TRAIN:,}", *ms),
            "a tie within %.1f points, and the %d-layer network costs %.1f× the time a step: on "
            "MNIST width is enough; where features compose, depth pays — it depends"
            % (spread, self.depths[-1], self.times[-1] / self.times[0]),
        ][ph]

    # -- the bands --------------------------------------------------------------------------
    def _band_colors(self, i, t, done_f, nf, uf, done_b, nb, ub, train):
        """``(colours (n, 3), shown (n,))`` of band ``i`` at time ``t``."""
        d, w = self.depths[i], self.widths[i]
        run = self.runs[i]
        layer = np.repeat(np.arange(d), w)                      # row-major: layer of each cell
        h = np.array(self.h[i])                                 # (d, w)
        c = lit_colors(log_scale(h, ACT_RANGE).ravel(), 1.0, DARK, ACT_HI)
        if t >= self.Tm["bwd"][0]:
            gc = lit_colors(log_scale(np.array(self.dh[i]), GRAD_RANGE).ravel(), 1.0, DARK,
                            GRAD_HI)
            got = layer >= d - done_b                             # from the output back
            if 0 <= nb < d and ub > 0.35:
                got |= layer == d - 1 - nb
            c[got] = gc[got]
        if train > 0:
            # the recorded activation std of each layer, interpolated between snapshots
            pos = (len(run.steps) - 1) * smoothstep(0.0, 0.92, train)
            k = min(int(pos), len(run.steps) - 2)
            f = pos - k
            std = run.act_std[k] + (run.act_std[k + 1] - run.act_std[k]) * f
            sc = lit_colors(log_scale(std[layer], STD_RANGE), 1.0, DARK, ACT_HI)
            mix = smoothstep(0.0, 0.15, train)
            c = c + (sc - c) * mix
        shown = layer < done_f
        if 0 <= nf < d and uf >= 0.5:
            shown |= layer == nf
        if t >= self.Tm["fwd"][1]:
            shown[:] = True
        return c, shown

    def _bands(self, sc, t, done_f, nf, uf, done_b, nb, ub, train, op):
        """Fill the three bands and caption them."""
        for i, s in enumerate(self.bands):
            d, w = self.depths[i], self.widths[i]
            c, shown = self._band_colors(i, t, done_f, nf, uf, done_b, nb, ub, train)
            s.fill(c)
            s.show(shown if t >= self.Tm["fwd"][0] else np.ones(d * w, bool))
            if t < self.Tm["fwd"][0]:
                s.fill(np.tile(DARK, (d * w, 1)))
            s.cage = 1.0
            s.fade(op)
            self.caption_left(sc, s, "%s: %d × %d" % (self.names[i], w, d), op)
            # values in flight: into the layer being computed, or back to the previous one
            if 0 < nf < d and t < self.Tm["fwd"][1]:
                self._flight(sc, s, i, nf - 1, nf, remap(uf, 0.05, 0.45), ACT_RANGE, ACT_HI,
                             self.h[i], 0.4)
            l_b = d - 1 - nb
            if self.Tm["bwd"][0] <= t < self.Tm["bwd"][1] and 0 <= l_b < d - 1:
                self._flight(sc, s, i, l_b + 1, l_b, remap(ub, 0.05, 0.45), GRAD_RANGE, GRAD_HI,
                             self.dh[i], -0.4)
        if 0 <= nf < max(self.depths) and t < self.Tm["fwd"][1]:
            sc.probe(("fwd",), (nf * 20 + uf * 20, 0, 0), tol=10.0)
        if 0 <= nb < max(self.depths) and self.Tm["bwd"][0] <= t < self.Tm["bwd"][1]:
            sc.probe(("bwd",), (nb * 20 + ub * 20, 0, 0), tol=10.0)

    def _flight(self, sc, s, i, src, dst, u, rng, hi, values, lift):
        """Particles from layer ``src`` to layer ``dst`` of band ``i``, coloured by
        the source layer's values on the log scale ``rng``."""
        from .netanim import particle
        w = self.widths[i]
        m = log_scale(np.asarray(values[src]), rng)
        cols = lit_colors(m, 1.0, DARK, hi)
        for c in range(0, w, 2):                               # every other unit: enough to read
            particle(sc, s.cell(src * w + c), s.cell(dst * w + c), u, cols[c], 0.5, lift=lift)

    def _layer_frame(self, sc, step, forward_):
        """A white frame around the layer at work in every band deep enough to have it."""
        for i, s in enumerate(self.bands):
            d, w = self.depths[i], self.widths[i]
            layer = step if forward_ else d - 1 - step
            if not 0 <= layer < d:
                continue
            lo, hi = s.block.box((layer, layer + 1), (0, w), (0, 1))
            sc.wire_box(lo, hi, inflate=0.18, color=FRAME, width=1.6, opacity=0.95, bias=0.1)

    # -- read-outs --------------------------------------------------------------------------
    def _notes(self, sc, done_f, done_b, op):
        """Right of the bands, one line each: the activation std over the
        minibatch at the layer just computed, then the gradient reaching the
        layer just reached (and its ratio to the last layer's)."""
        op *= self._cap
        if op <= 0.01 or (done_f == 0 and done_b == 0):
            return
        ys = []
        for s in self.bands:
            lo, hi = s.bbox()
            ys.append(self.cam.project((lo + hi) / 2)[1] / self.cam.height)
        for i in range(1, len(ys)):                      # never closer than a line apart
            ys[i] = max(ys[i], ys[i - 1] + NOTE_STEP)
        head = ("gradient reaching the layer (× the last layer's)" if done_b > 0
                else "activation std over a minibatch (He init)")
        sc.text(NOTE_X, ys[0] - NOTE_STEP, head, 0.027, "regular", layout.CAPTION_INK, op, "lm")
        for i, y in enumerate(ys):
            d = self.depths[i]
            if done_b > 0:
                l = max(0, d - done_b)
                text = "%s, layer %d: %s (%.2f×)" % (self.names[i], l + 1, pow10(self.grad0[i][l]),
                                                      self.grad0[i][l] / self.grad0[i][-1])
            else:
                l = min(done_f, d)
                text = "%s, layer %d: std %.2f%s" % (self.names[i], l, self.act0[i][l - 1],
                                                     "" if l == done_f else " (its last)")
            sc.text(NOTE_X, y, text, 0.029, "regular", layout.STRONG_INK, op, "lm")

    # -- the accuracy curves ----------------------------------------------------------------
    def _chart(self, sc, train, back):
        """The three test-accuracy curves against the step (log axis), with the
        final accuracy and the measured time per step in the legend."""
        op = smoothstep(0.12, 0.3, train) * back * self._cap
        if op <= 0.01:
            return
        cam, place = self.cam, self.cam.frac
        run = self.runs[0]
        upto = max(1.0, float(run.steps[-1]) ** smoothstep(0.0, 0.92, train))
        ticks = [(s, f"{s:,}") for s in (1, 10, 100, 1000)]
        self.chart.axes(sc, cam, place, op, "training steps (log scale)", "test accuracy",
                        ticks, [(v, "%d%%" % v) for v in (0, 50, 100)])
        for i, r in enumerate(self.runs):
            self.chart.curve(sc, place, np.maximum(r.steps, 1), 100 * r.test_acc, upto, INKS[i],
                             op)
        legend = [("%s: %.1f%%, %.1f ms a step" % (self.names[i], 100 * self.acc[i],
                                                   1e3 * self.times[i]), INKS[i])
                  for i in range(len(self.runs))]
        self.chart.legend(sc, cam, legend, op * smoothstep(0.85, 1.0, train),
                          (CHART[0][0], CHART[1][1] + 0.10), anchor="lm")      # under the axis
        sc.probe(("train",), (np.log10(upto) * 30, 0, 0), tol=6.0)

"""Fully-connected network (MLP) choreography: information flowing forward
through an MNIST classifier, and gradients flowing back.

Visual language (shared by every animation here):

* every vector is a **sheet** of voxels facing the camera -- the 28x28 image,
  the 8x8 / 4x8 hidden layers, the 10 logits (:data:`SHAPES`);
* a value > 0 is **lit**: bright, glowing and popped towards the viewer; a 0 is
  dark and flush.  That is the input's ink, and it is ReLU: only neurons whose
  value is above zero light up;
* signed numbers (weights, products, gradients) use the diverging scale --
  orange positive, blue negative;
* a dense layer is a **template** per neuron: that neuron's incoming weights,
  laid out like its input.  The input multiplies into it (zero pixels go dark:
  they contribute nothing), and the sum flies to the neuron;
* backprop runs the other way: a neuron's error flies back to the template,
  which becomes its weight gradient (error x the forward activations, so only
  inputs that were > 0 light up) and pours into the gradient of the layer
  below; ReLU then blocks the gradient at neurons that were off.

All numbers come from :mod:`voxanim.mnist` (a real, trained network).
Like the convolution animations, everything is a pure function of time.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from .conv import Tempo
from .easing import ease_in_out, ease_out, remap, smoothstep, soft_band
from .mnist import SIZES
from .netanim import (
    ACT_HI,
    INK_HI,
    INK_LO,
    OFF,
    NetAnimation,
    OutputPanel,
    amax,
    bezier,
    div_colors,
    gray_colors,
    hash01,
    lit_colors,
    particle,
    strongest,
)
from .text import TIMES

SHAPES = [(28, 28), (8, 8), (4, 8), (1, 10)]     # how each layer of SIZES is drawn
assert [h * w for h, w in SHAPES] == list(SIZES), "SHAPES must match mnist.SIZES"
N_IN, N_H1, N_H2, N_OUT = SIZES

# ------------------------------------------------------------------ forward, one layer


VIEW = {   # per layer: ppu, sheet scales (input, template, output), anchors
    0: dict(ppu=7.6, scales=(1.0, 1.0, 2.6), at=((0.17, 0.52), (0.47, 0.52), (0.78, 0.55))),
    1: dict(ppu=20.0, scales=(1.0, 1.0, 1.0), at=((0.17, 0.52), (0.47, 0.52), (0.79, 0.55))),
    2: dict(ppu=17.0, scales=(1.0, 1.0, 1.0), at=((0.14, 0.55), (0.40, 0.55), (0.70, 0.56))),
}


class DenseForward(NetAnimation):
    """Layer ``k`` (0, 1, 2) of ``tr`` (a :class:`~voxanim.mnist.Trace`)
    computed neuron by neuron, then its activation function."""

    def __init__(self, tr, k):
        names = ["Layer 1: %d pixels into %d neurons" % (N_IN, N_H1),
                 "Layer 2: %d into %d neurons" % (N_H1, N_H2),
                 "Output layer: %d neurons into %d digits" % (N_H2, N_OUT)]
        subs = ["each neuron weighs every pixel; only pixels > 0 contribute, then ReLU "
                "keeps the positive sums",
                "the same again on the %d activations; neurons that were 0 add nothing" % N_H1,
                "%d sums (logits) become probabilities, compared with the one-hot label"
                % N_OUT]
        v = VIEW[k]
        super().__init__(names[k], subs[k], v["ppu"])
        self.tr, self.k = tr, k
        self.a = tr.acts[k]
        self.W, self.b, self.z = tr.Ws[k], tr.bs[k], tr.zs[k]
        si, st_, so = v["scales"]
        ai, at_, ao = v["at"]
        self.inp = self.sheet(SHAPES[k], ai, si, "input")
        self.tpl = self.sheet(SHAPES[k], at_, st_, "template")
        self.out_n = self.W.shape[1]
        self.panel = None
        if k == 2:
            self.panel = OutputPanel(self, tr, ao, so)
            self.out = self.panel.logits
        else:
            self.out = self.sheet(SHAPES[k + 1], ao, so, "output")

        self.intro = 3.2 if k == 0 else 0.8      # seconds before the first neuron (the image sweep)
        tempo = Tempo(slow=3.0, fast=0.11 if k == 0 else 0.16, ramp_in=4, ramp_out=1.5)
        if k == 2:
            tempo = Tempo(slow=3.0, fast=0.55, ramp_in=3, ramp_out=1.5)
        self.durs = tempo.durations(self.out_n)
        self.starts = self.intro + np.concatenate([[0.0], np.cumsum(self.durs)[:-1]])
        self.t_steps_end = self.starts[-1] + self.durs[-1]
        t = self.t_steps_end + 0.4
        if k < 2:
            self.t_relu = (t, t + 2.0)
            t += 2.4
        else:
            self.t_bars = (t, t + 1.6)
            self.t_hot = (t + 1.9, t + 2.7)
            self.t_loss = (t + 2.9, t + 3.6)
            t += 4.2
        self.finish(t)

        # Colours that do not change over time.
        self.zmax = amax(self.z)
        self.out_div = div_colors(self.z, self.zmax)
        self.out_lit = lit_colors(np.maximum(self.z, 0), amax(np.maximum(self.z, 0)))
        self.wcols = [div_colors(self.W[:, j], amax(self.W[:, j])) for j in range(self.out_n)]
        on = self.a > 0
        self.pcols = []
        for j in range(self.out_n):
            prod = self.a * self.W[:, j]
            self.pcols.append(div_colors(prod, amax(prod), mask=on))

    # -- the input sheet ---------------------------------------------------------
    def _input(self, sc, t, reset):
        """The input sheet; layer 1 first sweeps the image from grey to lit ink."""
        a = self.a
        if self.k == 0:
            # Plain greyscale first, then a sweep: ink lights up, zeros go dark.
            gray = gray_colors(a)
            lit = lit_colors(a, 1.0, INK_LO, INK_HI)
            rows = np.repeat(np.arange(self.inp.h), self.inp.w) / self.inp.h
            m = np.array([smoothstep(0.6 + 1.6 * r, 0.9 + 1.6 * r, t) for r in rows])
            m = m * (1 - smoothstep(0.2, 0.8, reset))
            self.inp.fill(gray + (lit - gray) * m[:, None])
            self.inp.light(a, 1.0)
            if self.inp.block.offset is not None:
                self.inp.block.offset[..., 1] *= m.reshape(self.inp.h, self.inp.w, 1)
            if 0.6 < t < 2.4:
                r = int(np.clip((t - 0.6) / 1.6 * self.inp.h, 0, self.inp.h - 1))
                lo, hi = self.inp.block.box((r, r + 1), (0, self.inp.w), (0, 1))
                sc.wire_box(lo, hi, inflate=0.05, color=(0.93, 0.93, 0.93), width=1.0,
                            opacity=0.8, bias=0.1)
        else:
            self.inp.fill(lit_colors(a, amax(a)))
            self.inp.light(a)

    def draw(self, sc, t, reset):
        """One frame: input, outputs landed so far, the current neuron's template."""
        k = self.k
        self._input(sc, t, reset)
        n = int(np.searchsorted(self.starts, t, side="right") - 1)
        landed = np.zeros(self.out_n, bool)
        glint = np.zeros(self.out_n)
        for j in range(self.out_n):
            land = self.starts[j] + 0.86 * self.durs[j]
            if t >= land:
                landed[j] = True
                glint[j] = 1.0 - ease_out(remap(t - land, 0.0, 0.5))

        # ---- output: pre-activations, then ReLU (or softmax) --------------------
        if k < 2:
            relu = ease_in_out(remap(t, *self.t_relu))
            order = np.arange(self.out_n) / self.out_n
            m = np.array([smoothstep(o * 0.7, o * 0.7 + 0.3, relu) for o in order])
            cols = self.out_div + (self.out_lit - self.out_div) * m[:, None]
            self.out.fill(cols)
            self.out.show(landed)
            self.out.block.white[:, :, 0] = glint.reshape(self.out.h, self.out.w)
            self.out.light(self.z * (m > 0.5))
            self.out.cage = 1.0
            self.out.fade(1.0 - ease_in_out(remap(reset, 0.0, 0.6)))
        else:
            tb = remap(t, *self.t_bars)
            th = remap(t, *self.t_hot)
            tl = remap(t, *self.t_loss)
            f = 1.0 - ease_in_out(remap(reset, 0.0, 0.6))
            self.panel.draw(sc, tb, th, tl, landed, self.out_div, fade=f)
            self.out.block.white[:, :, 0] = glint.reshape(1, 10)

        # ---- the template of the neuron being computed --------------------------
        tpl_on = smoothstep(self.intro - 0.5, self.intro, t) * \
            (1.0 - smoothstep(self.t_steps_end, self.t_steps_end + 0.5, t))
        if tpl_on <= 0.01 or n < 0:
            self.tpl.visible = False
        else:
            j = min(n, self.out_n - 1)
            u = (t - self.starts[j]) / self.durs[j]
            prev = self.pcols[j - 1] if j > 0 else self.wcols[0]
            c = prev + (self.wcols[j] - prev) * smoothstep(0.0, 0.16, u)
            rows = np.repeat(np.arange(self.tpl.h), self.tpl.w) / max(self.tpl.h - 1, 1)
            m = np.array([smoothstep(0.24 + 0.26 * r, 0.30 + 0.26 * r, u) for r in rows])
            c = c + (self.pcols[j] - c) * m[:, None]
            self.tpl.fill(c)
            self.tpl.fade(tpl_on)
            self.tpl.cage = tpl_on
            # the row being read lights up on the input
            if 0.22 < u < 0.6:
                r = int(np.clip((u - 0.22) / 0.30 * self.inp.h, 0, self.inp.h - 1))
                self.inp.block.white[r] = np.maximum(self.inp.block.white[r],
                                                     0.35 * (self.a.reshape(self.inp.h, -1)[r]
                                                             > 0)[:, None])
            # the sum flies to the neuron
            uu = remap(u, 0.58, 0.86)
            size = 0.75 * self.out.block.scale
            particle(sc, self.tpl.center, self.out.cell(j), uu, self.out_div[j], size,
                     lift=4.0 * self.out.block.scale)
            sc.probe(("sum",), bezier(self.tpl.center, self.out.cell(j), ease_in_out(uu),
                                      4.0), tol=6.0)
            r, c = divmod(j, self.out.w)
            lo, hi = self.out.block.box((r, r + 1), (c, c + 1), (0, 1))
            sc.wire_box(lo, hi, inflate=0.06 * self.out.block.scale,
                        color=(0.93, 0.93, 0.93), width=1.3, opacity=0.9 * tpl_on, bias=0.1)
            self._step_text(sc, j, u, tpl_on)
        self._captions(sc, t, reset)

    def _step_text(self, sc, j, u, op):
        """The template's caption: detailed while the steps are slow, a count once fast."""
        slow = self.durs[j] > 0.8
        if slow:
            if u < 0.24:
                s = "weights of neuron %d" % (j + 1)
            elif u < 0.58:
                s = "weight %s input (zeros add nothing)" % TIMES
            else:
                s = "sum + bias = %.2f" % self.z[j]
        else:
            s = "neuron %d of %d" % (j + 1, self.out_n)
        self.caption(sc, self.tpl, s, op)

    def _captions(self, sc, t, reset):
        """Captions of the input and output sheets (or the panel)."""
        k = self.k
        f = 1.0 - smoothstep(0.0, 0.5, reset)
        names = ["image x  (%d pixels)" % N_IN, "h1 = ReLU(layer 1)  (%d)" % N_H1,
                 "h2 = ReLU(layer 2)  (%d)" % N_H2]
        if k == 0 and t < 2.6:
            names[0] = "image x  (28%s28 pixels, 0 to 1)" % TIMES if t < 0.7 else \
                "only pixels > 0 light up"
        self.caption(sc, self.inp, names[k], f)
        if k < 2:
            relu = remap(t, *self.t_relu)
            if relu <= 0:
                s = "pre-activation z  (%d)" % self.out_n
            elif relu < 1:
                s = "ReLU: max(0, z), only positive values light up"
            else:
                s = names[k + 1]
            self.caption(sc, self.out, s, f)
        else:
            self.panel.captions(sc, remap(t, *self.t_bars), remap(t, *self.t_hot),
                                remap(t, *self.t_loss))

    def scene(self, t, probes_only=False):
        """The scene at ``t``, plus the output panel's bars for the last layer."""
        sc = super().scene(t, probes_only)
        if self.panel is not None and not probes_only:
            self.panel.emit(sc)
        return sc


# ------------------------------------------------------------------ backward, one layer


class DenseBackward(NetAnimation):
    """Back-prop through layer ``k``, neuron by neuron.

    Right: the layer's error dL/dz.  Middle: each neuron's weight gradient,
    error x forward input (only inputs that were > 0 light up).  Left: the
    forward input (top) and the gradient flowing on to the layer below
    (bottom), accumulating one neuron at a time; then ReLU's gate.
    """

    def __init__(self, tr, k):
        names = ["Backprop, layer 1: errors to weights and pixels",
                 "Backprop, layer 2: %d errors back to %d neurons" % (N_H2, N_H1),
                 "Backprop, output layer: p - y back to %d neurons" % N_H2]
        subs = ["weight gradients are error %s pixel: only pixels > 0 change their weights"
                % TIMES,
                "each error reaches back along its weights; neurons that were off get no "
                "gradient",
                "the error is p - y: push the true digit up, the others down"]
        super().__init__(names[k], subs[k], {0: 6.2, 1: 17.0, 2: 17.0}[k])
        self.tr, self.k = tr, k
        a = tr.acts[k]
        self.a = a
        self.W = tr.Ws[k]
        self.delta = tr.deltas[k]
        self.dW = tr.dWs[k]
        self.da = tr.dacts[k]
        in_shape, out_shape = SHAPES[k], SHAPES[k + 1]
        # screen anchors; the 28x28 image sits a little higher, and layer 1's
        # 8x8 error sheet is drawn 3x larger to match it
        pos = dict(act=(0.17, 0.33 if k == 0 else 0.36), grad=(0.17, 0.74), tpl=(0.50, 0.55),
                   err=(0.82, 0.55))
        so = 3.0 if k == 0 else 1.0
        self.act = self.sheet(in_shape, pos["act"], 1.0, "activations")
        self.grad = self.sheet(in_shape, pos["grad"], 1.0, "grad")
        self.tpl = self.sheet(in_shape, pos["tpl"], 1.0, "dW")
        self.err = self.sheet(out_shape, pos["err"], so, "delta")

        # Only neurons with a non-zero error take part (ReLU-off ones have none).
        self.active = np.nonzero(np.abs(self.delta) > 1e-12)[0]
        tempo = Tempo(slow=3.0, fast=0.12 if k == 0 else 0.22, ramp_in=4, ramp_out=1.5)
        if k == 2:
            tempo = Tempo(slow=3.0, fast=0.6, ramp_in=3, ramp_out=1.5)
        self.intro = 2.0
        self.durs = tempo.durations(len(self.active))
        self.starts = self.intro + np.concatenate([[0.0], np.cumsum(self.durs)[:-1]])
        t = self.starts[-1] + self.durs[-1] + 0.4
        self.t_gate = (t, t + 2.0) if k > 0 else None
        t += 2.4 if k > 0 else 0.6
        self.finish(t)

        self.dmax = amax(self.delta)
        self.err_cols = div_colors(self.delta, self.dmax, mask=np.abs(self.delta) > 1e-12)
        on = a > 0
        self.tcols = [div_colors(self.dW[:, j], amax(self.dW[:, j]), mask=on)
                      for j in range(self.W.shape[1])]
        # running sums of the gradient flowing down, on one colour scale
        contrib = self.W[:, self.active] * self.delta[self.active][None]
        self.partial = np.cumsum(contrib, axis=1)
        self.gmax = amax(self.da)
        if k > 0:
            self.gate = tr.zs[k - 1] > 0

    def draw(self, sc, t, reset):
        """One frame: inputs and errors, the gradient below so far, the current neuron."""
        f = 1.0 - ease_in_out(remap(reset, 0.0, 0.6))
        self._inputs_and_errors()
        n = int(np.searchsorted(self.starts, t, side="right") - 1)
        gate = self._gradient_below(t, n, f)
        self._current_neuron(sc, t, n)
        self._captions(sc, reset, gate)

    def _inputs_and_errors(self):
        """The forward activations (what the weight gradients multiply) and dL/dz."""
        if self.k == 0:
            self.act.fill(lit_colors(self.a, 1.0, INK_LO, INK_HI))
        else:
            self.act.fill(lit_colors(self.a, amax(self.a)))
        self.act.light(self.a)
        self.err.fill(self.err_cols)
        self.err.cage = 1.0

    def _gradient_below(self, t, n, f):
        """The gradient flowing down, accumulated over neurons ``0..n``, then ReLU's
        gate; returns the gate's progress (0..1)."""
        appear = smoothstep(0.2, 1.2, t)
        if n < 0:
            g = np.zeros_like(self.da)
        else:
            j = min(n, len(self.active) - 1)
            u = (t - self.starts[j]) / self.durs[j]
            prev = self.partial[:, j - 1] if j > 0 else np.zeros_like(self.da)
            g = prev + (self.partial[:, j] - prev) * smoothstep(0.62, 0.9, u)
        self.grad.cage = appear * f
        cols = div_colors(g, self.gmax, mask=np.abs(g) > 1e-12 * self.gmax)
        cols[np.abs(g) <= 1e-12 * self.gmax] = C.NEUTRAL
        gate = remap(t, *self.t_gate) if self.t_gate else 0.0
        if gate > 0:
            rows = np.repeat(np.arange(self.grad.h), self.grad.w) / self.grad.h
            m = np.array([smoothstep(r * 0.7, r * 0.7 + 0.3, gate) for r in rows])
            off = ~self.gate
            cols[off] = cols[off] + (OFF - cols[off]) * m[off][:, None]
        self.grad.fill(cols)
        self.grad.fade(f * appear)
        return gate

    def _current_neuron(self, sc, t, n):
        """Neuron ``active[n]``: its error flies to the template, which becomes its
        weight gradient, and its share pours into the gradient below."""
        t_last = self.starts[-1] + self.durs[-1]
        tpl_on = smoothstep(self.intro - 0.5, self.intro, t) * \
            (1.0 - smoothstep(t_last, t_last + 0.5, t))
        if n < 0 or tpl_on <= 0.01:
            self.tpl.visible = False
            return
        j = min(n, len(self.active) - 1)
        nj = self.active[j]
        u = (t - self.starts[j]) / self.durs[j]
        prev = self.tcols[self.active[j - 1]] if j > 0 else \
            np.tile(C.NEUTRAL, (self.tpl.n, 1))
        rows = np.repeat(np.arange(self.tpl.h), self.tpl.w) / max(self.tpl.h - 1, 1)
        m = np.array([smoothstep(0.22 + 0.3 * r, 0.30 + 0.3 * r, u) for r in rows])
        self.tpl.fill(prev + (self.tcols[nj] - prev) * m[:, None])
        self.tpl.fade(tpl_on)
        self.tpl.cage = tpl_on
        # the error flies from its neuron to the template...
        size = 0.7 * self.err.block.scale
        particle(sc, self.err.cell(nj), self.tpl.center, remap(u, 0.0, 0.25),
                 self.err_cols[nj], size, lift=3.0)
        # ...and its share pours into the gradient below
        particle(sc, self.tpl.center, self.grad.center, remap(u, 0.55, 0.85),
                 self.err_cols[nj], 0.9, lift=-2.0)
        sc.probe(("err",), self.err.cell(nj), tol=6.0)
        r, c = divmod(int(nj), self.err.w)
        lo, hi = self.err.block.box((r, r + 1), (c, c + 1), (0, 1))
        sc.wire_box(lo, hi, inflate=0.06 * self.err.block.scale, color=(0.93, 0.93, 0.93),
                    width=1.3, opacity=0.9 * tpl_on, bias=0.1)
        slow = self.durs[j] > 0.8
        if slow and u < 0.25:
            s = "error of neuron %d = %.3g" % (nj + 1, self.delta[nj])
        elif slow:
            s = "weight gradient = error %s input" % TIMES
        else:
            s = "neuron %d" % (nj + 1)
        self.caption(sc, self.tpl, s, tpl_on)

    def _captions(self, sc, reset, gate):
        """Captions of the activations, the errors and the gradient below."""
        k = self.k
        f = 1.0 - smoothstep(0.0, 0.5, reset)
        below = ["image x", "h1 (forward)", "h2 (forward)"][k]
        self.caption(sc, self.act, below + ": only values > 0 light up", f)
        errs = ["error at layer 1: dL/dz1", "error at layer 2: dL/dz2", "error: p - y"]
        self.caption(sc, self.err, errs[k], f)
        if k == 0:
            g = "dL/dx: how each pixel would change the loss"
        elif gate <= 0:
            g = "dL/dh%d  (sum of errors %s weights)" % (k, TIMES)
        elif gate < 1:
            g = "%s ReLU': no gradient where the neuron was off" % TIMES
        else:
            g = "error at layer %d, on to the layer below" % k
        self.caption(sc, self.grad, g, f)


# ------------------------------------------------------------------ whole network


FLOW_AT = dict(x=(0.14, 0.53), h1=(0.37, 0.53), h2=(0.555, 0.53), out=(0.74, 0.55))
FLOW_SCALE = dict(x=1.0, h1=2.2, h2=2.2, out=1.8)


class NetworkFlow(NetAnimation):
    """The whole network: values flowing forward, or gradients flowing back."""

    K = 140          # particles per layer (the largest contributions)

    def __init__(self, tr, backward=False, at=FLOW_AT, ppu=7.4, title=None, subtitle=None):
        if title is None:
            title = "Back-propagation: gradients flow back" if backward else \
                "Forward pass: information flows through the network"
        if subtitle is None:
            subtitle = ("the error p - y travels back along the weights; ReLU blocks it at "
                        "neurons that were off") if backward else \
                ("only pixels and neurons > 0 light up and send information on; "
                 "orange excites, blue inhibits")
        super().__init__(title, subtitle, ppu)
        self.tr, self.backward = tr, backward
        self.sx = self.sheet(SHAPES[0], at["x"], FLOW_SCALE["x"], "x")
        self.s1 = self.sheet(SHAPES[1], at["h1"], FLOW_SCALE["h1"], "h1")
        self.s2 = self.sheet(SHAPES[2], at["h2"], FLOW_SCALE["h2"], "h2")
        self.panel = OutputPanel(self, tr, at["out"], FLOW_SCALE["out"])
        self.layers = [self.sx, self.s1, self.s2, self.panel.logits]
        self.phase = 3.0
        t = 1.4
        self.t_layer = []
        for _ in range(3):
            self.t_layer.append((t, t + self.phase))
            t += self.phase + 0.4
        if backward:
            self.t_err = (0.4, 1.2)
            t += 0.4
        else:
            self.t_bars = (t, t + 1.4)
            self.t_hot = (t + 1.6, t + 2.3)
            self.t_loss = (t + 2.5, t + 3.1)
            t += 3.6
        self.finish(t)
        self.parts_fwd = [self._particles(k, False) for k in range(3)]
        self.parts_bwd = [self._particles(k, True) for k in range(3)]

    def _particles(self, k, backward):
        """Top contributions of layer ``k``: ``(i, j, value / max, start)``."""
        tr = self.tr
        W = tr.Ws[k]
        if not backward:
            c = tr.acts[k][:, None] * W                   # a_i w_ij
        else:
            gate = (tr.zs[k - 1] > 0) if k > 0 else np.ones(W.shape[0], bool)
            c = W * tr.deltas[k][None, :] * gate[:, None]  # w_ij delta_j (gated)
        out = []
        cmax = amax(c)
        for n in strongest(c, self.K):
            i, j = divmod(int(n), W.shape[1])
            v = c[i, j] / cmax
            src_row = (i // SHAPES[k][1]) / SHAPES[k][0] if not backward else \
                (j // SHAPES[k + 1][1]) / SHAPES[k + 1][0]
            start = 0.55 * src_row + 0.12 * hash01(n, k)
            out.append((i, j, v, start))
        return out

    # forward: layer k carries layer k -> k+1; backward: layer k+1 -> k
    def draw(self, sc, t, reset):
        """One frame of the forward (or backward) flow."""
        if not self.backward:
            self._draw_forward(sc, t, reset)
        else:
            # The backward pass starts from the forward result, so its reset
            # rewinds the gradients instead of fading everything out.
            self._draw_backward(sc, t, 1.0 - ease_in_out(reset))

    def _sheet_value_colors(self, k, prog):
        """Colours of layer k's sheet once its values have arrived (prog 0..1)."""
        tr = self.tr
        if k == 0:
            return lit_colors(tr.x, 1.0, INK_LO, INK_HI)
        z = tr.zs[k - 1]
        a = np.maximum(z, 0)
        div = div_colors(z, amax(z))
        lit = lit_colors(a, amax(a))
        return div + (lit - div) * smoothstep(0.75, 1.0, prog)

    def _draw_forward(self, sc, t, reset, t_layer=None):
        """Values flowing forward; ``t_layer[k]`` times layer ``k`` (default ``self.t_layer``)."""
        tr = self.tr
        t_layer = t_layer or self.t_layer
        f = 1.0 - ease_in_out(remap(reset, 0.0, 0.6))
        # input reveal
        reveal = smoothstep(0.2, 1.2, t) * (1.0 - ease_in_out(remap(reset, 0.3, 1.0)))
        g = gray_colors(tr.x)
        self.sx.fill(g + (self._sheet_value_colors(0, 1) - g) * reveal)
        self.sx.light(tr.x, reveal)
        for k in range(3):
            t0, t1 = t_layer[k]
            prog = remap(t, t0, t1)
            dst = self.layers[k + 1]
            src = self.layers[k]
            # destination cells appear as their particles arrive
            if k < 2:
                if prog <= 0:
                    dst.show(np.zeros(dst.n, bool))
                else:
                    order = np.arange(dst.n) / dst.n
                    dst.show(prog * 1.15 > 0.4 + 0.5 * order)
                    dst.fill(self._sheet_value_colors(k + 1, prog))
                    if prog > 0.85:
                        dst.light(tr.zs[k], smoothstep(0.85, 1.0, prog))
                dst.cage = 1.0
            for i, j, v, start in self.parts_fwd[k]:
                u = remap(prog, start, start + 0.35)
                col = C.POSITIVE if v > 0 else C.NEGATIVE
                particle(sc, src.cell(i), dst.cell(j), u, col, 0.55,
                         opacity=0.35 + 0.65 * abs(v), lift=3.0)
            if 0 < prog < 1:
                sc.probe(("flow", k), (prog * 40, 0, 0), tol=8.0)
        # output panel
        t_out0, t_out1 = t_layer[2]
        po = remap(t, t_out0, t_out1)
        order = np.arange(10) / 10
        mask = po * 1.15 > 0.4 + 0.5 * order
        self.panel.draw(sc, remap(t, *self.t_bars), remap(t, *self.t_hot),
                        remap(t, *self.t_loss), mask if po > 0 else np.zeros(10, bool),
                        div_colors(tr.zs[2], amax(tr.zs[2])), fade=f)
        for s in (self.s1, self.s2):
            s.fade(f)
        self.panel.emit(sc)
        # captions
        self._layer_captions(sc)
        self.panel.captions(sc, remap(t, *self.t_bars), remap(t, *self.t_hot),
                            remap(t, *self.t_loss))

    def _layer_captions(self, sc):
        """``image x (784)``, ``h1 (64)``, ``h2 (32)`` under the three sheets."""
        self.caption(sc, self.sx, "image x (%d)" % N_IN)
        self.caption(sc, self.s1, "h1 (%d)" % N_H1)
        self.caption(sc, self.s2, "h2 (%d)" % N_H2)

    def _draw_backward(self, sc, t, back, t_layer=None, t_err=None, layers=(2, 1, 0)):
        """``t_layer[n]`` times the n-th layer in ``layers`` (output side first)."""
        tr = self.tr
        t_layer = t_layer or self.t_layer
        t_err = t_err or self.t_err
        # the forward result is where we start: activations lit, bars, target
        self.sx.fill(lit_colors(tr.x, 1.0, INK_LO, INK_HI))
        self.sx.light(tr.x)
        for k, s in ((1, self.s1), (2, self.s2)):
            s.fill(lit_colors(tr.acts[k], amax(tr.acts[k])))
            s.light(tr.acts[k])
            s.cage = 1.0
        # error at the output: p - y
        e = remap(t, *t_err) * back
        d3 = tr.deltas[2]
        logits = div_colors(tr.zs[2], amax(tr.zs[2]))
        err = div_colors(d3, amax(d3))
        self.panel.draw(sc, 1.0, 1.0, 0.0, np.ones(10, bool),
                        logits + (err - logits) * ease_in_out(e))
        self.panel.emit(sc)
        # back through the layers: output -> h2 -> h1 -> x
        sheets = [self.sx, self.s1, self.s2, self.panel.logits]
        for step, k in enumerate(layers):
            t0, t1 = t_layer[step]
            prog = remap(t, t0, t1)
            src, dst = sheets[k + 1], sheets[k]
            for i, j, v, start in self.parts_bwd[k]:
                u = remap(prog, start, start + 0.35)
                col = C.POSITIVE if v > 0 else C.NEGATIVE
                particle(sc, src.cell(j), dst.cell(i), u, col, 0.55,
                         opacity=0.35 + 0.65 * abs(v), lift=3.0)
            if prog > 0.4:
                m = smoothstep(0.4, 1.0, prog) * back
                if k > 0:
                    gate = tr.zs[k - 1] > 0
                    d = tr.deltas[k - 1]
                    cur = lit_colors(tr.acts[k], amax(tr.acts[k]))
                    tgt = div_colors(d, amax(d), mask=gate)
                    dst.fill(cur + (tgt - cur) * m)
                    if dst.block.offset is not None:
                        dst.block.offset *= (1 - m)
                else:
                    gx = tr.dacts[0]
                    cur = lit_colors(tr.x, 1.0, INK_LO, INK_HI)
                    dst.fill(cur + (div_colors(gx, amax(gx)) - cur) * m)
                    if dst.block.offset is not None:
                        dst.block.offset *= (1 - m)
            if 0 < prog < 1:
                sc.probe(("flow", k), (prog * 40, 0, 0), tol=8.0)
        # captions
        names = {0: ("image x", "dL/dx (saliency)"), 1: ("h1", "error at h1"),
                 2: ("h2", "error at h2")}
        for k, s in ((0, self.sx), (1, self.s1), (2, self.s2)):
            done = k in layers and remap(t, *t_layer[layers.index(k)]) > 0.6
            self.caption(sc, s, names[k][1] if done else names[k][0])
        self.caption_side(sc, self.panel.logits,
                          "error p - y" if t > t_err[0] else "logits z")


# ------------------------------------------------------------------ a training step


TRAIN_AT = dict(x=(0.13, 0.40), h1=(0.355, 0.40), h2=(0.535, 0.40), out=(0.76, 0.47))


class TrainingStep(NetworkFlow):
    """One full SGD step on one image, start to finish:

    1. forward -- the image flows through; only values > 0 light up,
    2. loss -- softmax against the one-hot target,
    3. backward -- the error p - y flows back, gated by ReLU,
    4. update -- for one neuron per layer: its weights, the change
       ``-lr x error x input`` (only inputs > 0 change), the updated weights;
       every neuron with an error does the same,
    5. forward again -- the same image through the updated weights.
    """

    PHASES = ("forward", "loss", "backward", "update", "forward again")

    def __init__(self, tr, tr_after, lr):
        """``tr`` / ``tr_after``: :class:`~voxanim.mnist.Trace` before and after the step."""
        super().__init__(tr, False, at=TRAIN_AT, ppu=6.8,
                         title="One training step on one image", subtitle="")
        self.tr2, self.lr = tr_after, lr
        self.hold = 2.5
        self._make_timeline()
        self._make_templates()

    def _make_timeline(self):
        """Phase times in seconds (each written out so the numbers stay exact)."""
        t = 1.2                                     # after the image fades in
        # forward: 2.4 s per layer, 0.3 s apart
        self.t_fwd = [(t, t + 2.4), (t + 2.7, t + 5.1), (t + 5.4, t + 7.8)]
        t += 8.1
        # loss: the bars grow, the one-hot target, the loss text
        self.t_bars, self.t_hot, self.t_loss = (t, t + 1.2), (t + 1.4, t + 2.0), (t + 2.2, t + 2.8)
        t += 3.6
        self.t_err = (t, t + 0.8)                   # the logits turn into p - y
        t += 1.0
        # backward: output -> h2, h2 -> h1 (2.4 s each)
        self.t_bwd = [(t, t + 2.4), (t + 2.7, t + 5.1)]
        t += 5.6
        # update: one neuron per layer, output first, 3.4 s each
        self.t_upd = {2: (t, t + 3.4), 1: (t + 3.7, t + 7.1), 0: (t + 7.4, t + 10.8)}
        t += 11.2
        self.t_again = (t, t + 5.0)                 # forward again
        t += 5.4
        self.finish(t)
        self.marks = [0.0, self.t_bars[0], self.t_err[0], self.t_upd[2][0], self.t_again[0]]

    def _make_templates(self):
        """One neuron per layer to show the update on (the most-blamed hidden
        neurons and the true digit's output neuron), and its colours."""
        tr, lr = self.tr, self.lr
        self.focus = {0: int(np.argmax(np.abs(tr.deltas[0]))),
                      1: int(np.argmax(np.abs(tr.deltas[1]))), 2: tr.label}
        at = {0: (0.20, 0.76), 1: (0.43, 0.77), 2: (0.645, 0.78)}
        sc = {0: 0.72, 1: 1.45, 2: 1.45}
        self.tpl, self.tcol = {}, {}
        for k in (0, 1, 2):
            self.tpl[k] = self.sheet(SHAPES[k], at[k], sc[k], "w%d" % k)
            j = self.focus[k]
            W, dW = tr.Ws[k][:, j], tr.dWs[k][:, j]
            change = -lr * dW
            on = tr.acts[k] > 0
            self.tcol[k] = dict(
                w=div_colors(W, amax(W)), change=div_colors(change, amax(change), mask=on),
                after=div_colors(W + change, amax(W)), on=on,
                rel=float(np.abs(change).max() / amax(W)))
        self.out_sheets = {0: self.s1, 1: self.s2, 2: self.panel.logits}
        self.in_sheets = {0: self.sx, 1: self.s1, 2: self.s2}

    # -- which step are we in ------------------------------------------------------
    def step_of(self, t):
        """Index into PHASES of the step under way at ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def draw(self, sc, t, reset):
        """One frame of whichever step is under way at ``t``."""
        for s in self.tpl.values():
            s.visible = False
        ph = self.step_of(t)
        lr, tr, tr2 = self.lr, self.tr, self.tr2
        label_ph = 0 if reset > 0 else ph       # the reset rewinds to step 1
        self.subtitle = [
            "forward: the image flows through the network; only values > 0 light up",
            "loss: compare the softmax probabilities with the one-hot target",
            "backward: the error p - y flows back; ReLU blocks it where neurons were off",
            "update: W = W - lr %s error %s input  (lr = %g): only inputs > 0 change their "
            "weights" % (TIMES, TIMES, lr),
            "forward again: same image, updated weights: p(%d) rises from %.2f to %.2f"
            % (tr.label, tr.p[tr.label], tr2.p[tr.label]),
        ][label_ph]
        if ph <= 1:
            self._draw_forward(sc, t, 0.0, self.t_fwd)
        elif ph <= 3:
            self._draw_backward(sc, t, 1.0, self.t_bwd, self.t_err, layers=(2, 1))
            if ph == 3:
                self._draw_update(sc, t)
        else:
            self._draw_again(sc, t, reset)
        self.step_indicator(sc, label_ph, self.PHASES)

    # -- 4: the update ---------------------------------------------------------------
    def _draw_update(self, sc, t):
        """Step 4: weights, change ``-lr x error x input`` and result, one neuron per layer."""
        tr = self.tr
        names = {0: "into h1 neuron %d", 1: "into h2 neuron %d", 2: "into digit %d"}
        for k in (2, 1, 0):
            a0, a1 = self.t_upd[k]
            if t < a0 - 0.3:
                continue
            u = remap(t, a0, a1)
            tp, c, j = self.tpl[k], self.tcol[k], self.focus[k]
            tp.visible = True
            rows = np.repeat(np.arange(tp.h), tp.w) / max(tp.h - 1, 1)
            m1 = np.array([smoothstep(0.22 + 0.2 * r, 0.30 + 0.2 * r, u) for r in rows])
            m2 = np.array([smoothstep(0.66 + 0.12 * r, 0.72 + 0.12 * r, u) for r in rows])
            col = c["w"] + (c["change"] - c["w"]) * m1[:, None]
            col = col + (c["after"] - col) * m2[:, None]
            tp.fill(col)
            tp.fade(smoothstep(a0 - 0.3, a0, t))
            tp.cage = 1.0
            # glint where weights changed (only where the input was > 0)
            glint = soft_band(u, 0.70, 0.88, 0.06)
            if glint > 0:
                tp.block.white[:, :, 0] = (c["on"] * glint * 0.8).reshape(tp.h, tp.w)
            # error (from the neuron) and input (from the layer below) meet
            dst, src = self.out_sheets[k], self.in_sheets[k]
            err_col = div_colors(tr.deltas[k][j:j + 1], amax(tr.deltas[k]))[0]
            particle(sc, dst.cell(j), tp.center, remap(u, 0.02, 0.24), err_col,
                     0.8 * dst.block.scale, lift=2.0)
            particle(sc, src.center, tp.center, remap(u, 0.04, 0.24), ACT_HI,
                     0.8 * max(src.block.scale, 1.0), lift=1.0)
            if 0.0 < u < 1.0:
                r_, c_ = divmod(j, dst.w)
                lo, hi = dst.block.box((r_, r_ + 1), (c_, c_ + 1), (0, 1))
                sc.wire_box(lo, hi, inflate=0.08 * dst.block.scale, color=(0.95, 0.95, 0.95),
                            width=1.5, opacity=0.95, bias=0.1)
                sc.probe(("upd", k), (u * 30, 0, 0), tol=8.0)
            # every neuron with an error updates its weights at the same time
            pulse = soft_band(u, 0.72, 0.92, 0.06)
            if pulse > 0:
                hit = np.abs(tr.deltas[k]) > 0
                dst.block.white[:, :, 0] = np.maximum(dst.block.white[:, :, 0],
                                                      0.6 * pulse * hit.reshape(dst.h, dst.w))
            if u < 0.22:
                s = "weights " + names[k] % (j if k == 2 else j + 1)
            elif u < 0.66:
                s = "change (only inputs > 0)"
            else:
                s = "updated: at most %.2g%%" % (100 * c["rel"])
            self.caption(sc, tp, s, smoothstep(a0 - 0.3, a0, t))

    # -- 5: forward again -------------------------------------------------------------
    def _draw_again(self, sc, t, reset):
        """Step 5: the same image through the updated weights, then the loop reset."""
        tr, tr2 = self.tr, self.tr2
        m = remap(t, *self.t_again)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        # the input: lit, and back to plain greyscale for the loop
        g = gray_colors(tr.x)
        lit = lit_colors(tr.x, 1.0, INK_LO, INK_HI)
        rv = 1.0 - ease_in_out(remap(reset, 0.3, 1.0))
        self.sx.fill(g + (lit - g) * rv)
        self.sx.light(tr.x, rv)
        # hidden layers: from their errors to the new activations
        for k, s, lo_ in ((1, self.s1, 0.05), (2, self.s2, 0.30)):
            gate = tr.zs[k - 1] > 0
            d = tr.deltas[k - 1]
            err = div_colors(d, amax(d), mask=gate)
            new = lit_colors(tr2.acts[k], amax(tr2.acts[k]))
            w = smoothstep(lo_, lo_ + 0.25, m)
            s.fill(err + (new - err) * w)
            s.light(tr2.acts[k], w)
            s.cage = 1.0
            s.fade(back)
        for k, (lo_, hi_) in enumerate(((0.0, 0.3), (0.25, 0.55), (0.5, 0.75))):
            prog = remap(m, lo_, hi_)
            src, dst = self.layers[k], self.layers[k + 1]
            for i, j, v, start in self.parts_fwd[k][::2]:
                u = remap(prog, start, start + 0.35)
                col = C.POSITIVE if v > 0 else C.NEGATIVE
                particle(sc, src.cell(i), dst.cell(j), u, col, 0.55,
                         opacity=0.35 + 0.65 * abs(v), lift=3.0)
            if 0 < prog < 1:
                sc.probe(("again", k), (prog * 40, 0, 0), tol=8.0)
        # output: logits and bars move to the new probabilities
        w = smoothstep(0.6, 0.9, m)
        p = tr.p + (tr2.p - tr.p) * w
        loss = tr.loss + (tr2.loss - tr.loss) * w
        self.panel.set_probs(p, loss)
        d3 = tr.deltas[2]
        err = div_colors(d3, amax(d3))
        new = div_colors(tr2.zs[2], amax(tr2.zs[2]))
        logits = err + (new - err) * smoothstep(0.5, 0.7, m)
        hide = ease_in_out(remap(reset, 0.0, 0.7))
        self.panel.draw(sc, back, back, back * smoothstep(0.85, 1.0, m), np.ones(10, bool),
                        logits, fade=back)
        self.panel.emit(sc)
        self._layer_captions(sc)
        self.panel.captions(sc, back, back, back * smoothstep(0.85, 1.0, m))
        self.panel.note(sc, "before the step: p = %.2f, loss %.2f" % (tr.p[tr.label], tr.loss),
                        smoothstep(0.85, 1.0, m) * (1 - hide))

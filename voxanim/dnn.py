"""Evaluating a trained fully-connected network on one digit.

Any dense ReLU network (loaded with :mod:`voxanim.model_io`) is laid out left
to right: the 28 x 28 image, one sheet per hidden layer (100 neurons -> a
10 x 10 sheet, any width is factored into a near-square), and the output
panel.  Each layer is evaluated in three beats:

1. **one neuron, up close** -- the most active neuron's incoming weights,
   laid out like the layer below; they multiply the inputs (inputs that are 0
   contribute nothing and go dark) and the sum flies to the neuron,
2. **the whole layer** -- every neuron at once: the strongest weighted inputs
   flow across and the pre-activations fill in (orange +, blue -),
3. **ReLU** -- only values > 0 light up and pass on.

The output layer ends with softmax probabilities and the one-hot prediction.
Like everything else here, it is a pure function of time.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import model_io
from .easing import ease_in_out, remap, smoothstep
from .mlp import (
    INK_HI,
    INK_LO,
    NetAnimation,
    OutputPanel,
    X,
    amax,
    div_colors,
    gray_colors,
    lit_colors,
    particle,
)


def sheet_shape(n):
    """A near-square ``(rows, cols)`` for ``n`` neurons (100 -> 10 x 10)."""
    r = int(np.floor(np.sqrt(n)))
    while n % r:
        r -= 1
    return (r, n // r) if r > 1 else (1, n)


class _Eval:
    """What :class:`OutputPanel` needs: the one-hot is the *prediction*."""

    def __init__(self, p, label):
        self.p = p
        self.prediction = int(p.argmax())
        self.label = self.prediction
        self.true_label = label
        self.loss = float(-np.log(p[self.prediction] + 1e-12))


class EvalPanel(OutputPanel):
    """The output panel for evaluation: the one-hot row is the prediction."""

    SIDE_TEXTS = ("softmax p(digit)", "logits z", "one-hot")

    @property
    def hot_index(self):
        return self.tr.prediction

    def result_text(self):
        ev = self.tr
        truth = "" if ev.true_label is None else (
            ", correct" if ev.true_label == ev.prediction else ", label is %d" % ev.true_label)
        return "predicts %d  (p = %.2f%s)" % (ev.prediction, ev.p[ev.prediction], truth)

    def captions(self, sc, t_bars, t_hot, t_loss, logits_text="logits z",
                 hot_text="one-hot"):
        super().captions(sc, t_bars, t_hot, t_loss, logits_text, hot_text)


class ModelEvaluation(NetAnimation):
    def __init__(self, layers, image, label=None, preprocess="unit", name="model"):
        sizes = [layers[0][0].shape[0]] + [W.shape[1] for W, _b in layers]
        arch = " - ".join(str(n) for n in sizes)
        super().__init__("Evaluating a trained network", "", 10.0)
        self.arch = arch
        self.layers = layers
        self.x = np.asarray(image, np.float64).reshape(-1)
        x_in = model_io.PREPROCESS[preprocess](self.x)
        self.zs, self.acts, self.p = model_io.forward(layers, x_in)
        self.acts[0] = self.x                        # draw the image as pixels, 0..1
        self.ev = _Eval(self.p, label)
        L = len(layers)
        hidden = sizes[1:-1]
        self.hold = 3.0

        # ---- sheets, spread left to right ------------------------------------
        xs = np.linspace(0.28, 0.575, max(len(hidden), 1)) if hidden else []
        self.sheets = []
        self.img = self.sheet((28, 28), (0.11, 0.47), 0.5, "image")
        self.hid = []
        for k, (n, fx) in enumerate(zip(hidden, xs)):
            shp = sheet_shape(n)
            scale = min(1.35, 13.5 / max(shp))
            self.hid.append(self.sheet(shp, (fx, 0.47), scale, "h%d" % (k + 1)))
        self.panel = EvalPanel(self, self.ev, (0.83, 0.47), 1.0)
        self.layer_in = [self.img] + self.hid
        self.layer_out = self.hid + [self.panel.logits]
        # the spotlight template: one neuron's weights, shaped like its input
        self.tpl = []
        for k in range(L):
            src = self.layer_in[k]
            sc = 0.35 if k == 0 else 0.85
            self.tpl.append(self.sheet((src.h, src.w), (0.0, 0.0), sc, "tpl%d" % k))
        self.focus = []
        for k in range(L):
            z = self.zs[k]
            self.focus.append(int(self.p.argmax()) if k == L - 1 else int(np.argmax(z)))

        # ---- timeline -----------------------------------------------------------
        Tm, t = {}, 0.8
        for k in range(L):
            d1 = 3.2 if k == 0 else 2.4
            Tm["one%d" % k] = (t, t + d1)
            t += d1 + 0.2
            Tm["all%d" % k] = (t, t + 2.2)
            t += 2.4
            if k < L - 1:
                Tm["relu%d" % k] = (t, t + 1.0)
                t += 1.3
        Tm["bars"] = (t, t + 1.2)
        Tm["hot"] = (t + 1.4, t + 2.0)
        Tm["text"] = (t + 2.2, t + 2.8)
        self.Tm = Tm
        self.finish(t + 3.0)
        self.marks = [Tm["one%d" % k][0] for k in range(L)]
        self.phase_names = ["layer %d" % (k + 1) for k in range(L - 1)] + ["output"]

        # ---- particles: the strongest weighted inputs of each layer -----------------
        self.parts = []
        for k, (W, _b) in enumerate(layers):
            a = self.acts[k] if k else self.x
            c = a[:, None] * W
            flat = np.abs(c).ravel()
            K = min(120, int((flat > 0).sum()))
            idx = np.argpartition(-flat, K - 1)[:K]
            cmax = amax(c)
            self.parts.append([(divmod(int(m), W.shape[1]), c.ravel()[m] / cmax,
                                0.55 * ((m * 2654435761) % 997) / 997) for m in idx])

    def p_(self, name, t):
        return remap(t, *self.Tm[name])

    def step_of(self, t):
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        L = len(self.layers)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = max(0, self.step_of(t))
        k_now = 0 if reset > 0 else min(ph, L - 1)      # the loop rewinds to layer 1
        nn = self.layers[k_now][0].shape
        self.subtitle = ("network %s, ReLU\n%s: %d inputs into %d neurons, then %s"
                         % (self.arch, self.phase_names[k_now].capitalize(), nn[0], nn[1],
                            "ReLU: only values > 0 light up and pass on" if k_now < L - 1
                            else "softmax gives the probability of each digit"))
        # image
        rv = smoothstep(0.0, 0.8, t) * (1.0 - ease_in_out(remap(reset, 0.3, 1.0)))
        g = gray_colors(self.x)
        self.img.fill(g + (lit_colors(self.x, 1.0, INK_LO, INK_HI) - g) * rv)
        self.img.light(self.x, rv)
        self.caption(sc, self.img, "image 28%s28" % X)

        for k in range(L):
            self._layer(sc, t, k, back)
        self.step_indicator(sc, -1 if reset > 0 else k_now if t >= self.marks[0] else -1,
                            self.phase_names)

    def _layer(self, sc, t, k, back):
        L = len(self.layers)
        W, _b = self.layers[k]
        z = self.zs[k]
        src = self.layer_in[k]
        dst = self.layer_out[k]
        a_in = self.x if k == 0 else self.acts[k]
        one, al = self.p_("one%d" % k, t), self.p_("all%d" % k, t)
        relu = self.p_("relu%d" % k, t) if k < L - 1 else 0.0
        j = self.focus[k]
        # ---- 1. one neuron, up close ------------------------------------------------
        tp = self.tpl[k]
        on = smoothstep(0.0, 0.08, one) * (1 - smoothstep(0.9, 1.0, one))
        tp.visible = on > 0.01
        if tp.visible:
            mid = (src.center + dst.center) / 2
            # in the empty lower third, between the two layers, clear of their
            # captions: solve for the height that puts its centre at 76% down
            cam = self.cam
            want = (0.5 + (0.78 - 0.5) * cam.zoom) * cam.height
            p = np.array([mid[0], mid[1], 0.0])
            height = (want - cam.project(p)[1]) / cam.M[1, 2]
            tp.block.place(np.array([mid[0], mid[1], height]))
            w = W[:, j]
            prod = a_in * w
            wc = div_colors(w, amax(w))
            pc = div_colors(prod, amax(prod), mask=a_in > 0)
            rows = np.repeat(np.arange(tp.h), tp.w) / max(tp.h - 1, 1)
            m = np.array([smoothstep(0.30 + 0.25 * r, 0.36 + 0.25 * r, one) for r in rows])
            tp.fill(wc + (pc - wc) * m[:, None])
            tp.fade(on * back)
            tp.cage = on
            particle(sc, src.center, tp.center, remap(one, 0.18, 0.32), INK_HI,
                     0.8 * max(1.0, src.block.scale), lift=1.0)
            particle(sc, tp.center, dst.cell(j), remap(one, 0.66, 0.9),
                     div_colors(z[j:j + 1], amax(z))[0], 0.9 * dst.block.scale, lift=3.0)
            if 0 < one < 1:
                r_, c_ = divmod(j, dst.w)
                lo, hi = dst.block.box((r_, r_ + 1), (c_, c_ + 1), (0, 1))
                sc.wire_box(lo, hi, inflate=0.08 * dst.block.scale, color=(0.96, 0.96, 0.96),
                            width=1.5, opacity=0.95, bias=0.1)
                sc.probe(("one", k), (one * 30, 0, 0), tol=8.0)
            if one < 0.3:
                s = "neuron %d: its %d weights" % (j + 1, len(w))
            elif one < 0.66:
                s = "weights %s inputs (0 inputs add nothing)" % X
            else:
                s = "sum + bias = %.2f" % z[j]
            self.caption(sc, tp, s, on)
        # ---- 2 & 3: the whole layer, then ReLU (output: softmax) -----------------------
        if k == L - 1:
            self._output(sc, t, al, back)
        else:
            n = len(z)
            order = np.arange(n) / n
            got = al * 1.2 > 0.25 + 0.6 * order
            got[j] |= one >= 0.9
            div = div_colors(z, amax(z))
            lit = lit_colors(np.maximum(z, 0), amax(np.maximum(z, 0)))
            m = smoothstep(0.0, 1.0, relu)
            dst.fill(div + (lit - div) * m)
            dst.show(got)
            dst.light(z, smoothstep(0.5, 1.0, relu))
            dst.cage = 1.0
            dst.fade(back)
            label = "h%d  (%d%s%d)" % (k + 1, dst.h, X, dst.w)
            if relu > 0:
                label = "h%d: %d > 0" % (k + 1, int((z > 0).sum()))
            self.caption(sc, dst, label)
        for (i, jj), v, start in self.parts[k]:
            u = remap(al, start, start + 0.4)
            particle(sc, src.cell(i), dst.cell(jj), u, C.POSITIVE if v > 0 else C.NEGATIVE,
                     0.5, opacity=0.35 + 0.65 * abs(v), lift=2.5)
        if 0 < al < 1:
            sc.probe(("all", k), (al * 40, 0, 0), tol=8.0)

    def _output(self, sc, t, al, back):
        z = self.zs[-1]
        tb, th, tt = self.p_("bars", t), self.p_("hot", t), self.p_("text", t)
        mask = np.arange(10) < al * 11
        mask[self.focus[-1]] |= self.p_("one%d" % (len(self.layers) - 1), t) >= 0.9
        self.panel.draw(sc, tb * back, th * back, tt * back, mask, div_colors(z, amax(z)),
                        fade=back)
        self.panel.emit(sc)
        self.panel.captions(sc, tb * back, th * back, tt * back)


def from_file(path, digit=0, preprocess=None):
    """A :class:`ModelEvaluation` for test image ``digit`` and a model file."""
    from . import mnist
    layers, pre = model_io.load_dense(path)
    _a, _b, xte, yte = mnist.load()
    return ModelEvaluation(layers, xte[int(digit)], int(yte[int(digit)]),
                           preprocess or pre)

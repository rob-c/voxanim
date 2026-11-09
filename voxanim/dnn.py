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
from .netanim import (
    INK_HI,
    INK_LO,
    NetAnimation,
    OutputPanel,
    amax,
    div_colors,
    gray_colors,
    hash_mod,
    lit_colors,
    particle,
    strongest,
)
from .text import TIMES


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
        """The predicted digit."""
        return self.tr.prediction

    def result_text(self):
        """The prediction, its probability and whether it is right."""
        ev = self.tr
        truth = "" if ev.true_label is None else (
            ", correct" if ev.true_label == ev.prediction else ", label is %d" % ev.true_label)
        return "predicts %d  (p = %.2f%s)" % (ev.prediction, ev.p[ev.prediction], truth)


IMAGE_AT = (0.11, 0.47)            # screen anchors: the image, the output panel, and
PANEL_SCALE = 1.1                   # wide enough for ten digit labels with clear gaps
PANEL_AT = (0.83, 0.47)            # the hidden layers spread evenly between HIDDEN_X
HIDDEN_X = (0.28, 0.575)
SPOTLIGHT_Y = 0.78                 # the spotlight template's centre, fraction of the height
PARTICLES = 120                    # strongest weighted inputs drawn per layer


class ModelEvaluation(NetAnimation):
    """A dense ReLU network evaluating one image, layer by layer.

    ``layers`` is ``[(W (in, out), b), ...]`` (see :mod:`voxanim.model_io`),
    ``image`` the 784 pixels in [0, 1], ``label`` the true digit if known and
    ``preprocess`` a key of :data:`voxanim.model_io.PREPROCESS`.
    """

    hold = 3.0

    def __init__(self, layers, image, label=None, preprocess="unit"):
        sizes = [layers[0][0].shape[0]] + [W.shape[1] for W, _b in layers]
        super().__init__("Evaluating a trained network", "", 10.0)
        self.arch = " - ".join(str(n) for n in sizes)
        self.layers = layers
        self.x = np.asarray(image, np.float64).reshape(-1)
        x_in = model_io.PREPROCESS[preprocess](self.x)
        self.zs, self.acts, self.p = model_io.forward(layers, x_in)
        self.acts[0] = self.x                        # draw the image as pixels, 0..1
        self.ev = _Eval(self.p, label)
        L = len(layers)
        self._make_sheets(sizes[1:-1])
        self.focus = [int(self.p.argmax()) if k == L - 1 else int(np.argmax(self.zs[k]))
                      for k in range(L)]
        self._make_timeline()
        self.parts = [self._particles(k) for k in range(L)]

    def _make_sheets(self, hidden):
        """The image, one sheet per hidden layer, the panel, and the spotlight templates."""
        xs = np.linspace(*HIDDEN_X, max(len(hidden), 1)) if hidden else []
        self.img = self.sheet((28, 28), IMAGE_AT, 0.5, "image")
        self.hid = []
        for k, (n, fx) in enumerate(zip(hidden, xs, strict=False)):
            shp = sheet_shape(n)
            scale = min(1.35, 13.5 / max(shp))     # at most 13.5 units wide
            self.hid.append(self.sheet(shp, (fx, IMAGE_AT[1]), scale, "h%d" % (k + 1)))
        self.panel = EvalPanel(self, self.ev, PANEL_AT, PANEL_SCALE)
        self.layer_in = [self.img] + self.hid
        self.layer_out = self.hid + [self.panel.logits]
        # the spotlight template: one neuron's weights, shaped like its input
        self.tpl = []
        for k, src in enumerate(self.layer_in):
            sc = 0.35 if k == 0 else 0.85
            self.tpl.append(self.sheet((src.h, src.w), (0.0, 0.0), sc, "tpl%d" % k))

    def _make_timeline(self):
        """``self.Tm``: per layer ``one<k>`` (the spotlight; 3.2 s for the first
        layer, then 2.4 s), ``all<k>`` (2.2 s) and ``relu<k>`` (1 s); then the
        output's ``bars``, ``hot`` and ``text``."""
        L = len(self.layers)
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

    def _particles(self, k):
        """Layer ``k``'s strongest weighted inputs: ``((i, j), value / max, start)``."""
        W = self.layers[k][0]
        a = self.acts[k] if k else self.x
        c = a[:, None] * W
        cmax = amax(c)
        return [(divmod(int(m), W.shape[1]), c.ravel()[m] / cmax,
                 0.55 * hash_mod(m, 997) / 997) for m in strongest(c, PARTICLES)]

    def step_of(self, t):
        """Index of the layer being evaluated at ``t`` (-1 before the first)."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """One frame: the image, every layer, and the step indicator."""
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
        self.caption(sc, self.img, "image 28%s28" % TIMES)

        for k in range(L):
            self._layer(sc, t, k, back)
        self.step_indicator(sc, -1 if reset > 0 else k_now if t >= self.marks[0] else -1,
                            self.phase_names)

    def _layer(self, sc, t, k, back):
        """Layer ``k``: the spotlight neuron, then the whole layer (or the output)."""
        L = len(self.layers)
        one, al = self.progress("one%d" % k, t), self.progress("all%d" % k, t)
        self._spotlight(sc, k, one, back)
        if k == L - 1:
            self._output(sc, t, al, back)
        else:
            self._hidden(sc, k, one, al, self.progress("relu%d" % k, t), back)
        src, dst = self.layer_in[k], self.layer_out[k]
        for (i, jj), v, start in self.parts[k]:
            u = remap(al, start, start + 0.4)
            particle(sc, src.cell(i), dst.cell(jj), u, C.POSITIVE if v > 0 else C.NEGATIVE,
                     0.5, opacity=0.35 + 0.65 * abs(v), lift=2.5)
        if 0 < al < 1:
            sc.probe(("all", k), (al * 40, 0, 0), tol=8.0)

    def _place_template(self, tp, src, dst):
        """Put the template between ``src`` and ``dst``, in the empty lower third."""
        mid = (src.center + dst.center) / 2
        # clear of the layers' captions: solve for the height that puts its
        # centre SPOTLIGHT_Y of the way down the (zoomed) frame
        cam = self.cam
        want = (0.5 + (SPOTLIGHT_Y - 0.5) * cam.zoom) * cam.height
        p = np.array([mid[0], mid[1], 0.0])
        height = (want - cam.project(p)[1]) / cam.M[1, 2]
        tp.block.place(np.array([mid[0], mid[1], height]))

    def _spotlight(self, sc, k, one, back):
        """Beat 1: neuron ``focus[k]``'s weights times the inputs, summed (``one`` 0..1)."""
        W, z = self.layers[k][0], self.zs[k]
        src, dst, tp, j = self.layer_in[k], self.layer_out[k], self.tpl[k], self.focus[k]
        a_in = self.x if k == 0 else self.acts[k]
        on = smoothstep(0.0, 0.08, one) * (1 - smoothstep(0.9, 1.0, one))
        tp.visible = on > 0.01
        if not tp.visible:
            return
        self._place_template(tp, src, dst)
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
            s = "weights %s inputs (0 inputs add nothing)" % TIMES
        else:
            s = "sum + bias = %.2f" % z[j]
        self.caption(sc, tp, s, on)

    def _hidden(self, sc, k, one, al, relu, back):
        """Beats 2 and 3 of hidden layer ``k``: the pre-activations fill in, then ReLU."""
        z, dst = self.zs[k], self.layer_out[k]
        n = len(z)
        order = np.arange(n) / n
        got = al * 1.2 > 0.25 + 0.6 * order
        got[self.focus[k]] |= one >= 0.9
        div = div_colors(z, amax(z))
        lit = lit_colors(np.maximum(z, 0), amax(np.maximum(z, 0)))
        m = smoothstep(0.0, 1.0, relu)
        dst.fill(div + (lit - div) * m)
        dst.show(got)
        dst.light(z, smoothstep(0.5, 1.0, relu))
        dst.cage = 1.0
        dst.fade(back)
        label = "h%d  (%d%s%d)" % (k + 1, dst.h, TIMES, dst.w)
        if relu > 0:
            label = "h%d: %d > 0" % (k + 1, int((z > 0).sum()))
        self.caption(sc, dst, label)

    def _output(self, sc, t, al, back):
        """The output layer: the logits fill in, then softmax bars and the one-hot."""
        z = self.zs[-1]
        tb, th, tt = self.progress("bars", t), self.progress("hot", t), self.progress("text", t)
        mask = np.arange(10) < al * 11
        mask[self.focus[-1]] |= self.progress("one%d" % (len(self.layers) - 1), t) >= 0.9
        self.panel.draw(sc, tb * back, th * back, tt * back, mask, div_colors(z, amax(z)),
                        fade=back)
        self.panel.emit(sc)
        self.panel.captions(sc, tb * back, th * back, tt * back)


def from_file(path, index=0, preprocess=None):
    """A :class:`ModelEvaluation` of the model in ``path`` on MNIST test image ``index``.

    ``preprocess`` overrides the scaling recorded in the file.
    """
    from . import mnist
    layers, pre = model_io.load_dense(path)
    _a, _b, xte, yte = mnist.load()
    return ModelEvaluation(layers, xte[int(index)], int(yte[int(index)]), preprocess or pre)

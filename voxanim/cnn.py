"""One full training step of a small convolutional network on one MNIST image.

    x 28x28 -> conv 3x3 x 4 filters -> ReLU -> max-pool 2x2 -> dense 676 -> 10 -> softmax

Laid out as four rows, one per channel: kernel -> feature map (26x26) ->
pooled map (13x13, same footprint, bigger cells), all feeding the output panel
(logits, softmax bars, one-hot target, loss).  The five steps:

1. forward -- each filter slides over the image (the first one slowly), ReLU
   keeps values > 0, max-pool keeps the largest of each 2x2, the dense layer
   reads the lit pooled cells,
2. loss -- softmax against the one-hot label,
3. backward -- p - y flows back to the pooled maps; max-pool routes it only to
   each window's winner and ReLU only where the map was on (most cells get
   nothing); each kernel's gradient gathers error x input patch,
4. update -- the kernels and the dense weights into the true digit step
   against their gradients,
5. forward again -- the same image through the updated network.

Same visual language as :mod:`voxanim.mlp`: lit = value > 0, orange/blue =
signed.  All numbers come from :func:`voxanim.mnist.cnn_trace`.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from .conv import Tempo
from .easing import ease_in_out, remap, smoothstep, soft_band
from .mnist import CNN_FILTERS, CONV_HW, POOL_HW
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

F = CNN_FILTERS
N_CONV = CONV_HW * CONV_HW          # 676 positions of each filter
N_POOL = POOL_HW * POOL_HW          # 169 cells of each pooled map
ROW_Y = (0.315, 0.465, 0.615, 0.765)          # one row per channel, screen fractions
AT = dict(x=(0.105, 0.55), kernel=0.255, conv=0.40, pool=0.555, out=(0.765, 0.57))
KERNEL_PARTICLES = 24               # gradient particles per kernel (the largest)


class ConvNetTrainingStep(NetAnimation):
    """One SGD step of the CNN: ``tr`` and ``tr_after`` are
    :class:`~voxanim.mnist.CnnTrace` before and after the step, ``lr`` its rate."""

    PHASES = ("forward", "loss", "backward", "update", "forward again")
    K = 150                                   # dense-layer particles

    def __init__(self, tr, tr_after, lr):
        super().__init__("CNN: one training step", "", 6.5)
        self.tr, self.tr2, self.lr = tr, tr_after, lr
        self.hold = 2.5
        self.sx = self.sheet((28, 28), AT["x"], 1.0, "x")
        self.ks = [self.sheet((3, 3), (AT["kernel"], y), 2.3, "k%d" % f)
                   for f, y in enumerate(ROW_Y)]
        self.ms = [self.sheet((CONV_HW, CONV_HW), (AT["conv"], y), 0.42, "conv%d" % f)
                   for f, y in enumerate(ROW_Y)]
        self.qs = [self.sheet((POOL_HW, POOL_HW), (AT["pool"], y), 0.84, "pool%d" % f)
                   for f, y in enumerate(ROW_Y)]
        self.panel = OutputPanel(self, tr, AT["out"], 2.3)
        self._make_timeline()
        self._make_particles()

    def _make_timeline(self):
        """``self.Tm``: each phase ``(start, end)`` in seconds, back to back with gaps."""
        T, t = {}, 0.0

        def add(name, d, gap=0.25):
            nonlocal t
            T[name] = (t, t + d)
            t += d + gap

        add("reveal", 1.0, 0.2)
        # the first filter slowly at first, then the other three together
        durs = Tempo(slow=0.5, fast=0.0032, ramp_in=3.0, ramp_out=1.0).durations(N_CONV)
        self.conv_starts = np.concatenate([[0.0], np.cumsum(durs)[:-1]])
        add("conv0", float(durs.sum()))
        add("conv123", 1.6)
        add("relu", 1.0)
        add("pool", 1.6)
        add("dense", 2.4, 0.3)
        add("bars", 1.2, 0.2)
        add("hot", 0.6, 0.2)
        add("loss", 0.6, 0.6)
        add("err", 0.8)
        add("dense_back", 2.2)
        add("unpool", 2.2)
        add("kgrad", 2.2, 0.5)
        add("kupdate", 2.8)
        add("dupdate", 2.8, 0.4)
        add("again", 5.0, 0.4)
        self.Tm = T
        self.finish(t)
        self.conv_starts = self.conv_starts + T["conv0"][0]
        self.marks = [0.0, T["bars"][0], T["err"][0], T["kupdate"][0], T["again"][0]]

    def _make_particles(self):
        """Fixed colour scales and the particles of the dense layer and the kernels."""
        tr = self.tr
        fw, g = tr.fwd, tr.grad
        Wc, Wd = tr.params[0][0], tr.params[1][0]
        self.zmax = amax(fw["z1"])
        self.amax_ = amax(fw["a1"])
        self.wcmax = amax(Wc)
        self.parts_fwd = self._top(fw["pooled"].reshape(-1)[:, None] * Wd)
        self.parts_bwd = self._top(Wd * g["d2"][None, :])
        # kernel-gradient particles: the conv cells that actually received gradient
        self.kparts = []
        for f in range(F):
            d = g["d1"][..., f].ravel()
            nz = np.nonzero(d)[0]
            pick = nz[np.argsort(-np.abs(d[nz]))[:KERNEL_PARTICLES]]
            self.kparts.append([(int(n), float(np.sign(d[n]))) for n in pick])

    def _top(self, c):
        """Largest |contributions| ``c`` ``(676, 10)`` of the dense layer:
        ``(filter, pooled cell, logit, value / max, start)``."""
        cmax = amax(c)
        out = []
        for n in strongest(c, self.K):
            i, j = divmod(int(n), 10)
            r, col, f = np.unravel_index(i, (POOL_HW, POOL_HW, F))
            start = 0.6 * (f / F) + 0.15 * hash_mod(n, 1000) / 1000.0
            out.append((f, int(r * POOL_HW + col), j, c[i, j] / cmax, start))
        return out

    def step_of(self, t):
        """Index into :attr:`PHASES` of the step under way at ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """One frame: every row of the network, the output panel and the captions."""
        tr, tr2 = self.tr, self.tr2
        ph = self.step_of(t)
        label_ph = 0 if reset > 0 else ph
        self.subtitle = [
            "forward: each 3%s3 filter slides over the image; ReLU keeps values > 0; "
            "max-pool keeps the largest of each 2%s2" % (TIMES, TIMES),
            "loss: a dense layer turns the %d%s%d%s%d maps into 10 scores; softmax vs the "
            "one-hot label" % (POOL_HW, TIMES, POOL_HW, TIMES, F),
            "backward: p - y flows back; max-pool sends it only to each window's winner, "
            "ReLU only where the map was on",
            "update: the filters and dense weights step against their gradients, "
            "W = W - lr %s dW  (lr = %g)" % (TIMES, self.lr),
            "forward again: same image, updated network: p(%d) rises from %.2f to %.2f"
            % (tr.label, tr.p[tr.label], tr2.p[tr.label]),
        ][label_ph]
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))    # the loop rewind
        again = self.progress("again", t)

        self._input(sc, t, reset)
        self._kernels(sc, t, reset)
        self._conv_maps(sc, t, back, again)
        self._pooled(sc, t, back, again)
        self._output(sc, t, back, again)
        self._captions(sc, t, again)
        self.step_indicator(sc, label_ph, self.PHASES)

    # ------------------------------------------------------------------ pieces
    def _input(self, sc, t, reset):
        """The image, and the 3x3 window being read during the convolution."""
        x = self.tr.x
        rv = smoothstep(0.1, 1.0, t) * (1.0 - ease_in_out(remap(reset, 0.3, 1.0)))
        g = gray_colors(x)
        self.sx.fill(g + (lit_colors(x, 1.0, INK_LO, INK_HI) - g) * rv)
        self.sx.light(x, rv)
        n = self._conv_pos(t)
        if n is not None:
            i, j = divmod(n, CONV_HW)
            lo, hi = self.sx.block.box((i, i + 3), (j, j + 3), (0, 1))
            sc.wire_box(lo, hi, inflate=0.06, color=(0.95, 0.95, 0.95), width=1.4,
                        opacity=0.95, bias=0.1)
            sc.probe(("convpos",), lo, tol=10.0)

    def _conv_pos(self, t):
        """Window currently being read (None outside the conv phases)."""
        if self.Tm["conv0"][0] <= t < self.Tm["conv0"][1]:
            return int(np.searchsorted(self.conv_starts, t, side="right") - 1)
        if self.Tm["conv123"][0] <= t < self.Tm["conv123"][1]:
            return min(N_CONV - 1, int(self.progress("conv123", t) * N_CONV))
        return None

    def _landed(self, f, t):
        """How many positions of filter ``f``'s map have been computed."""
        if f == 0:
            if t >= self.Tm["conv0"][1]:
                return N_CONV
            return max(0, int(np.searchsorted(self.conv_starts, t, side="right")) - 1)
        return int(min(N_CONV, self.progress("conv123", t) * (N_CONV + 1)))

    def _kernels(self, sc, t, reset):
        """The four kernels: weights, gradient, change, updated; gradient particles."""
        tr, tr2, lr = self.tr, self.tr2, self.lr
        Wc, dWc = tr.params[0][0], tr.grad["dWc"]
        Wc2 = tr2.params[0][0]
        kg = self.progress("kgrad", t)
        ku = self.progress("kupdate", t)
        for f, ks in enumerate(self.ks):
            w = div_colors(Wc[..., f].ravel(), self.wcmax)
            gcol = div_colors(dWc[..., f].ravel(), amax(dWc[..., f]))
            change = -lr * dWc[..., f].ravel()
            ccol = div_colors(change, amax(change))
            w2 = div_colors(Wc2[..., f].ravel(), self.wcmax)
            col = w + (gcol - w) * smoothstep(0.45, 0.8, kg)          # gradient arrives
            col = col + (ccol - col) * smoothstep(0.0, 0.15, ku)       # as a change...
            col = col + (w2 - col) * smoothstep(0.55, 0.8, ku)         # ...applied
            col = col + (w - col) * ease_in_out(remap(reset, 0.2, 0.9))   # loop rewind
            ks.fill(col)
            glint = soft_band(ku, 0.58, 0.8, 0.06)
            ks.block.white[...] = 0.7 * glint
            active = (f == 0 and self._conv_pos(t) is not None
                      and t < self.Tm["conv0"][1]) or \
                (f > 0 and self.Tm["conv123"][0] <= t < self.Tm["conv123"][1])
            ks.cage = 1.0
            if active:
                lo, hi = ks.bbox()
                sc.wire_box(lo, hi, inflate=0.12, color=(0.95, 0.95, 0.95), width=1.6,
                            opacity=0.95, bias=0.1)
            # gradient gathering: cells that got gradient send it to the kernel
            if 0 < kg < 0.8:
                for n, s in self.kparts[f]:
                    u = remap(kg, 0.05 + 0.3 * ((n * 7919) % 97) / 97, 0.55)
                    particle(sc, self.ms[f].cell(n), ks.center, u,
                             C.POSITIVE if s > 0 else C.NEGATIVE, 0.42, lift=1.5)
                sc.probe(("kgrad",), (kg * 30, 0, 0), tol=8.0)

    def _conv_maps(self, sc, t, back, again):
        """The feature maps: conv, ReLU, the sparse gradient, then forward again."""
        tr, tr2 = self.tr, self.tr2
        fw, g = tr.fwd, tr.grad
        relu = self.progress("relu", t)
        unpool = self.progress("unpool", t)
        pool = self.progress("pool", t)
        for f, m in enumerate(self.ms):
            z = fw["z1"][..., f].ravel()
            a = fw["a1"][..., f].ravel()
            n = self._landed(f, t)
            present = np.arange(N_CONV) < n
            div = div_colors(z, self.zmax)
            lit = lit_colors(a, self.amax_)
            col = div + (lit - div) * smoothstep(0.0, 1.0, relu)
            # backward: only max-pool winners that were on receive gradient
            d1 = g["d1"][..., f].ravel()
            gcol = div_colors(d1, amax(g["d1"]), mask=d1 != 0)
            rows = np.repeat(np.arange(CONV_HW), CONV_HW) / CONV_HW
            mu = np.clip((unpool - rows * 0.6) / 0.4, 0, 1)
            col = col + (gcol - col) * mu[:, None]
            # forward again: the new activations
            a2 = tr2.fwd["a1"][..., f].ravel()
            ma = smoothstep(0.0, 0.25, again)
            col = col + (lit_colors(a2, self.amax_) - col) * ma
            m.fill(col)
            m.show(present)
            lit_amount = smoothstep(0.6, 1.0, relu) * (1 - smoothstep(0.0, 0.4, unpool)) + ma
            m.light(a2 if again > 0 else a, min(1.0, lit_amount))
            m.cage = 1.0
            m.fade(back)
            if 0 < pool < 1:                       # the 2x2 window being pooled
                k = min(N_POOL - 1, int(pool * N_POOL))
                r, c = divmod(k, POOL_HW)
                lo, hi = m.block.box((2 * r, 2 * r + 2), (2 * c, 2 * c + 2), (0, 1))
                sc.wire_box(lo, hi, inflate=0.04, color=(0.95, 0.95, 0.95), width=1.1,
                            opacity=0.9, bias=0.1)

    def _pooled(self, sc, t, back, again):
        """The pooled maps, the dense weights' update, and the dense-layer particles."""
        tr, tr2, lr = self.tr, self.tr2, self.lr
        fw, g = tr.fwd, tr.grad
        pool = self.progress("pool", t)
        dense = self.progress("dense", t)
        db = self.progress("dense_back", t)
        du = self.progress("dupdate", t)
        pmax = amax(fw["pooled"])
        Wd, dWd = tr.params[1][0], g["dWd"]
        y = tr.label
        wd = Wd[:, y].reshape(POOL_HW, POOL_HW, F)
        ch = (-lr * dWd[:, y]).reshape(POOL_HW, POOL_HW, F)
        for f, q in enumerate(self.qs):
            v = fw["pooled"][..., f].ravel()
            col = lit_colors(v, pmax)
            dp = g["dpool"][..., f].ravel()
            gcol = div_colors(dp, amax(g["dpool"]))
            col = col + (gcol - col) * smoothstep(0.45, 0.9, db)
            # the dense weights into the true digit: weights -> change -> updated
            w = div_colors(wd[..., f].ravel(), amax(wd))
            c = div_colors(ch[..., f].ravel(), amax(ch), mask=v > 0)
            w2 = div_colors((wd + ch)[..., f].ravel(), amax(wd))
            if du > 0:
                col = col + (w - col) * smoothstep(0.0, 0.15, du)
                col = col + (c - col) * smoothstep(0.25, 0.4, du)
                col = col + (w2 - col) * smoothstep(0.6, 0.75, du)
                col = col + (lit_colors(v, pmax) - col) * smoothstep(0.88, 1.0, du)
                q.block.white[:, :, 0] = (0.7 * soft_band(du, 0.62, 0.8, 0.05) *
                                          (v > 0)).reshape(POOL_HW, POOL_HW)
            v2 = tr2.fwd["pooled"][..., f].ravel()
            ma = smoothstep(0.25, 0.5, again)
            col = col + (lit_colors(v2, pmax) - col) * ma
            q.fill(col)
            rows = np.repeat(np.arange(POOL_HW), POOL_HW) / POOL_HW
            q.show(pool * 1.15 > rows)
            lit_on = (1 - smoothstep(0.3, 0.6, db)) if du <= 0 or du >= 1 else 0.0
            q.light(v2 if again > 0 else v, max(lit_on * (pool >= 1), ma))
            q.cage = 1.0
            q.fade(back)
        # the dense layer reads the lit pooled cells / sends errors back to them
        for f, n, j, val, start in self.parts_fwd:
            u = remap(dense, start, start + 0.35)
            particle(sc, self.qs[f].cell(n), self.panel.logits.cell(j), u,
                     C.POSITIVE if val > 0 else C.NEGATIVE, 0.6,
                     opacity=0.35 + 0.65 * abs(val), lift=3.0)
            u2 = remap(again, 0.45 + 0.3 * start, 0.45 + 0.3 * start + 0.2)
            particle(sc, self.qs[f].cell(n), self.panel.logits.cell(j), u2,
                     C.POSITIVE if val > 0 else C.NEGATIVE, 0.6,
                     opacity=0.35 + 0.65 * abs(val), lift=3.0)
        for f, n, j, val, start in self.parts_bwd:
            u = remap(db, start, start + 0.35)
            particle(sc, self.panel.logits.cell(j), self.qs[f].cell(n), u,
                     C.POSITIVE if val > 0 else C.NEGATIVE, 0.6,
                     opacity=0.35 + 0.65 * abs(val), lift=3.0)
        for key, v in (("dense", dense), ("db", db), ("again", again)):
            if 0 < v < 1:
                sc.probe(("flow", key), (v * 40, 0, 0), tol=8.0)

    def _output(self, sc, t, back, again):
        """Logits, error, the probabilities moving to their new values."""
        tr, tr2 = self.tr, self.tr2
        dense = self.progress("dense", t)
        order = np.arange(10) / 10
        mask = dense * 1.15 > 0.45 + 0.5 * order
        logits = div_colors(tr.fwd["z2"], amax(tr.fwd["z2"]))
        err = div_colors(tr.grad["d2"], amax(tr.grad["d2"]))
        col = logits + (err - logits) * ease_in_out(self.progress("err", t))
        new = div_colors(tr2.fwd["z2"], amax(tr2.fwd["z2"]))
        col = col + (new - col) * smoothstep(0.55, 0.75, again)
        w = smoothstep(0.7, 0.95, again)
        self.panel.set_probs(tr.p + (tr2.p - tr.p) * w, tr.loss + (tr2.loss - tr.loss) * w)
        tb = self.progress("bars", t) * back
        th = self.progress("hot", t) * back
        tl = (self.progress("loss", t) * (1 - self.progress("err", t)) +
              smoothstep(0.9, 1.0, again)) * back
        self.panel.draw(sc, tb, th, tl, mask, col, fade=back)
        self.panel.emit(sc)
        self.panel.captions(sc, tb, th, tl, "error p - y" if t >= self.Tm["err"][0] and
                            again <= 0.6 else "logits z")
        self.panel.note(sc, "before the step: p = %.2f, loss %.2f" % (tr.p[tr.label], tr.loss),
                        smoothstep(0.9, 1.0, again) * back)

    def _captions(self, sc, t, again):
        """Captions of the image, the kernels, the feature maps and the pooled maps."""
        T = self.Tm
        self.caption(sc, self.sx, "image x  28%s28" % TIMES)
        ktext = "%d filters 3%s3" % (F, TIMES)
        if t < T["relu"][0]:
            conv = "conv %d%s%d%s%d" % (CONV_HW, TIMES, CONV_HW, TIMES, F)
        elif t < T["unpool"][0] + 0.3 or again > 0:
            conv = "conv + ReLU"
        else:
            conv = "gradient reaches only max-pool winners that were on"
        self.caption(sc, self.ms[3], conv)
        if T["dupdate"][0] <= t < T["dupdate"][1]:
            pool = "dense weights into digit %d" % self.tr.label
            du = self.progress("dupdate", t)
            if 0.25 <= du < 0.6:
                pool = "change: only inputs > 0"
        elif T["dense_back"][0] + 1.0 <= t < T["again"][0]:
            pool = "gradient dL/dpool"
        else:
            pool = "max-pool %d%s%d%s%d" % (POOL_HW, TIMES, POOL_HW, TIMES, F)
        self.caption(sc, self.qs[0], pool, above=True)
        if T["kgrad"][0] <= t < T["kupdate"][1]:
            ku = self.progress("kupdate", t)
            if ku <= 0:
                s = "filter gradients"
            elif ku < 0.55:
                s = "change = -lr %s gradient" % TIMES
            else:
                rel = np.abs(self.lr * self.tr.grad["dWc"]).max() / self.wcmax
                s = "updated: at most %.2g%%" % (100 * rel)
            ktext = s
        self.caption(sc, self.ks[0], ktext, above=True)

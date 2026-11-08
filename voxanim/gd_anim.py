"""Gradient descent through a one-hidden-layer perceptron, learning XOR.

    x1 --\\    h1 --\\
          >-- h2 ---> y^ ---> loss  L = 1/2 (y^ - y)^2
    x2 --/    h3 --/

Two inputs, three sigmoid hidden neurons, one sigmoid output, squared error.
Each neuron shows its activation function: a small sigmoid plot with the
operating point z on it; on the way back the tangent at z is the
derivative sigma'(z) that the gradient is multiplied by.

1. **forward** -- one example, x = (1, 0) with target 1: the data (teal)
   flows left to right, each neuron computes z = w.x + b and a = sigma(z),
2. **loss** -- the output against the target,
3. **backward** -- the gradient (magenta) flows right to left: dL/dy^,
   times sigma' gives the output's delta; each weight's gradient is its
   delta times the value that came in; the hidden deltas follow.  Every
   neuron's ring lights up once its gradient has been computed and recorded,
4. **batch** -- the same for the other three examples; gradients average,
5. **update** -- w = w - eta * dL/dw: every weight steps downhill,
6. **train** -- 2,000 steps: data and gradients flowing in opposite
   directions, the loss falling, the four predictions becoming XOR.

All numbers are real (:func:`train`), plain full-batch gradient descent.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from . import color as C
from . import layout
from .diagram import DATA, DATA_HI, EDGE, GRAD, NODE, Diagram
from .easing import ease_in_out, remap, smoothstep
from .netanim import hash01

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float64)
Y = np.array([0, 1, 1, 0], np.float64)
EX = 2                                  # the worked example: x = (1, 0), y = 1
LR, STEPS, SEED = 2.0, 2000, 0
SUB = "₁₂₃"
MINUS = "−"


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def num(v, d=2):
    s = "%.*f" % (d, v)
    if s.startswith("-"):
        s = MINUS + s[1:]
    return "0" + s[1:] if s.startswith("-0") else s


def unpack(p):
    return p[:6].reshape(2, 3), p[6:9], p[9:12], p[12]


def forward(p, x):
    W, b, v, c = unpack(p)
    z = x @ W + b
    h = sig(z)
    o = h @ v + c
    return z, h, o, sig(o)


def grads(p, x, y):
    """Gradient of 1/2 (y^ - y)^2 for one example, and the deltas."""
    _W, _b, v, _c = unpack(p)
    _z, h, _o, yh = forward(p, x)
    dy = yh - y
    d_out = dy * yh * (1 - yh)
    dv = d_out * h
    dh = d_out * v
    dz = dh * h * (1 - h)
    dW = np.outer(x, dz)
    g = np.concatenate([dW.ravel(), dz, dv, [d_out]])
    return g, dict(dy=dy, d_out=d_out, dz=dz, dv=dv, dW=dW)


@cache
def train(seed=SEED, steps=STEPS, lr=LR):
    """Full-batch gradient descent on the mean loss.  ``(params per step, loss per step)``."""
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.normal(0, 1, 6), np.zeros(3), rng.normal(0, 1, 3), [0.0]])
    hist, losses = [p.copy()], []
    for _ in range(steps):
        yh = np.array([forward(p, x)[3] for x in X])
        losses.append(float(np.mean(0.5 * (yh - Y) ** 2)))
        G = np.mean([grads(p, x, y)[0] for x, y in zip(X, Y)], axis=0)
        p = p - lr * G
        hist.append(p.copy())
    yh = np.array([forward(p, x)[3] for x in X])
    losses.append(float(np.mean(0.5 * (yh - Y) ** 2)))
    return np.array(hist), np.array(losses)


# ------------------------------------------------------------------ layout (fractions)

IN_AT = [(0.05, 0.45), (0.05, 0.70)]
HID_AT = [(0.25, 0.32), (0.25, 0.57), (0.25, 0.82)]
OUT_AT = (0.45, 0.57)
LOSS_AT = (0.615, 0.57)
TABLE_AT = (0.80, 0.30)                 # top-left of the XOR table
CHART = (0.745, 0.67, 0.985, 0.96)      # loss chart: x0, y0, x1, y1
R = 22.0                                # neuron radius, px
ICON_W, ICON_H = 46.0, 26.0
WEIGHT_MAX = 6.0


def w_color(w):
    k = min(1.0, abs(w) / WEIGHT_MAX) ** 0.7
    base = C.POSITIVE if w >= 0 else C.NEGATIVE
    return C.mix(EDGE, C.mix(base, EDGE, 0.25), k)


class GradientDescent(Diagram):
    PHASES = ("forward", "loss", "backward", "batch", "update", "train")

    def __init__(self):
        super().__init__("Gradient descent", "")
        self.hist, self.losses = train()
        self.p0 = self.hist[0]
        self.p1 = self.hist[1]
        self.f0 = forward(self.p0, X[EX])
        self.g0, self.d0 = grads(self.p0, X[EX], Y[EX])
        self.G = np.mean([grads(self.p0, x, y)[0] for x, y in zip(X, Y)], axis=0)
        self.gall = [grads(self.p0, x, y)[0] for x, y in zip(X, Y)]
        self.inp = [self.P(*a) for a in IN_AT]
        self.hid = [self.P(*a) for a in HID_AT]
        self.out = self.P(*OUT_AT)
        self.loss = self.P(*LOSS_AT)
        self.hold, self.reset = 3.0, 1.5
        T = dict(intro=(0.2, 1.4), x=(1.4, 2.4), f1=(2.4, 4.2), hid=(4.2, 6.0),
                 f2=(6.0, 7.2), outn=(7.2, 8.4), f3=(8.4, 9.4), lossv=(9.4, 11.4),
                 b0=(11.6, 12.8), bout=(12.8, 15.0), gv=(15.0, 17.4), b1=(17.4, 18.8),
                 bhid=(18.8, 23.0), gw=(23.0, 25.8), batch=(26.2, 31.8),
                 update=(32.2, 36.2), train=(36.8, 55.0), final=(55.0, 56.0))
        self.Tm = T
        self.finish(56.4)

    def p_(self, name, t):
        return remap(t, *self.Tm[name])

    def step_of(self, t):
        for k, name in ((5, "train"), (4, "update"), (3, "batch"), (2, "b0"), (1, "f3")):
            if t >= self.Tm[name][0]:
                return k
        return 0

    # ------------------------------------------------------------------ geometry
    def ends(self, a, b, ra=R, rb=R):
        sa, sb = self.px(a), self.px(b)
        d = np.linalg.norm(sb - sa)
        u = (b - a) / d
        return a + u * ra, b - u * rb

    def edges(self):
        """``[(kind, i, j, src, dst)]``: w_ij (input i to hidden j), v_j (hidden j to out)."""
        out = []
        for i in range(2):
            for j in range(3):
                out.append(("w", i, j, self.inp[i], self.hid[j]))
        for j in range(3):
            out.append(("v", j, 0, self.hid[j], self.out))
        return out

    @staticmethod
    def w_of(p, kind, i, j):
        W, _b, v, _c = unpack(p)
        return W[i, j] if kind == "w" else v[i]

    @staticmethod
    def g_index(kind, i, j):
        return i * 3 + j if kind == "w" else 9 + i

    # ------------------------------------------------------------------ pieces
    def icon(self, sc, c, z, op, tangent=0.0):
        """The neuron's activation function: sigma over z in [-6, 6], the point
        at z, and (``tangent``) its slope."""
        if op <= 0.01:
            return
        top = self.shift(c, 0.0, -(R + 8 + ICON_H / 2))
        self.rect(sc, top, ICON_W + 8, ICON_H + 8, (0.95, 0.95, 0.94), op * 0.9,
                  border=(0.70, 0.71, 0.72), border_w=1.0, radius=4, lift=0.4)

        def pt(zz, s):
            return self.shift(top, (zz / 12.0) * ICON_W, (0.5 - s) * ICON_H) + self.view * 0.7
        zs = np.linspace(-6, 6, 25)
        pts = [pt(zz, sig(zz)) for zz in zs]
        self.polyline(sc, pts, width=1.4, color=(0.36, 0.38, 0.40), opacity=0.9 * op)
        zc = float(np.clip(z, -6, 6))
        p = pt(zc, sig(zc))
        if tangent > 0:
            s = sig(zc) * (1 - sig(zc))
            dz = 2.2
            a = pt(zc - dz, sig(zc) - s * dz)
            b = pt(zc + dz, sig(zc) + s * dz)
            sc.line(a, b, width=2.0, color=GRAD, opacity=op * tangent, bias=0.6)
        self.disk(sc, p, 3.2, DATA, op, lift=1.3, outline=0.0)

    def node(self, sc, c, fill, text, op, ring=0.0, glow=0.0, ink=None):
        self.disk(sc, c, R, fill, op, ring=GRAD, ring_w=3.0, ring_op=ring, glow=glow)
        self.label(sc, c, text, layout.MIN_TEXT, ink=ink or layout.STRONG_INK, opacity=op,
                   kind="mono")

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        T = self.Tm
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        ph = 0 if reset > 0 else self.step_of(t)
        intro = smoothstep(0.0, 1.0, self.p_("intro", t)) * back
        # which parameters are on show
        upd = ease_in_out(self.p_("update", t))
        tr = self.p_("train", t)
        step = int(round(STEPS * tr)) if tr > 0 else (1 if upd >= 1 else 0)
        if tr > 0:
            p = self.hist[step]
        else:
            p = self.p0 + (self.p1 - self.p0) * upd
        # the example on show: the worked one, the batch, then the worked one again
        bu = self.p_("batch", t)
        others = [k for k in range(4) if k != EX]
        ex = EX
        if 0 < bu < 1:
            ex = others[min(2, int(bu * 3))]
        x, y = X[ex], Y[ex]
        z, h, o, yh = forward(p, x)

        # progress markers
        fx = self.p_("x", t)
        f1, fh, f2, fo = (self.p_(k, t) for k in ("f1", "hid", "f2", "outn"))
        f3, fl = self.p_("f3", t), self.p_("lossv", t)
        b0, bo, gv, b1 = (self.p_(k, t) for k in ("b0", "bout", "gv", "b1"))
        bh, gw = self.p_("bhid", t), self.p_("gw", t)
        training = tr > 0
        lit_v, lit_w, lit_o, lit_h = gv, gw, bo, bh        # recorded: stays lit
        if 0 < bu < 1:                          # each other example: forward then back
            u = (bu * 3) % 1.0
            fx = f1 = fh = f2 = fo = f3 = fl = smoothstep(0.0, 0.45, u)
            b0 = bo = gv = b1 = bh = gw = smoothstep(0.5, 0.95, u)
        if training or upd > 0:
            fx = f1 = fh = f2 = fo = f3 = fl = 1.0
        # recorded gradients: worked example, then the running batch average
        g_rec = self.g0
        if bu > 0:
            n = 1 + min(3, int(bu * 3 + (1 if bu >= 1 else 0)))
            got = [self.gall[EX]] + [self.gall[k] for k in others][:n - 1]
            g_rec = np.mean(got, axis=0) * (4 / 4)
            if bu >= 1:
                g_rec = self.G
        rec = (1.0 - smoothstep(0.55, 0.95, upd)) * back
        if training:
            rec = 0.0

        # ---- edges
        gmax = np.abs(self.G).max() if bu > 0 else np.abs(self.g0).max()
        for kind, i, j, a, b in self.edges():
            w = self.w_of(p, kind, i, j)
            ea, eb = self.ends(a, b)
            width = 0.8 + 3.2 * min(1.0, abs(w) / WEIGHT_MAX) ** 0.7
            sc.line(ea, eb, width=width, color=w_color(w), opacity=0.45 * intro, bias=0.0)
            # the weight's gradient, once computed
            gi = self.g_index(kind, i, j)
            lit = (smoothstep(0.0, 0.25, remap(lit_v, i / 3, i / 3 + 0.3)) if kind == "v" else
                   smoothstep(0.0, 0.25, remap(lit_w, j / 3, j / 3 + 0.3)))
            g = g_rec[gi]
            if lit > 0 and rec > 0 and abs(g) > 1e-9:
                k = min(1.0, abs(g) / gmax) ** 0.5
                off = self.Mi @ np.array([0.0, -3.0])
                sc.line(ea + off, eb + off, width=1.0 + 3.0 * k, color=GRAD,
                        opacity=min(0.48, 0.2 + 0.4 * k) * lit * rec, bias=0.1)

        # ---- the data, flowing forward
        dkey = 0
        for kind, i, j, a, b in self.edges():
            ea, eb = self.ends(a, b)
            if kind == "w":
                u = remap(f1, 0.1 * i, 0.6 + 0.1 * i)
                val = x[i]
            else:
                u = remap(f2, 0.1 * j, 0.6 + 0.1 * j)
                val = h[i]
            if not training:
                self.dot(sc, ea, eb, u, DATA if val > 0 else C.mix(DATA, EDGE, 0.6),
                         4.2 if val > 0 else 2.6, back, key=("d", dkey))
            dkey += 1
        oa, ob = self.ends(self.out, self.loss, R, 58)
        if not training:
            self.dot(sc, oa, ob, f3, DATA, 4.2, back, key=("d", "loss"))

        # ---- gradients, flowing back
        if not training:
            self.dot(sc, ob, oa, b0, GRAD, 4.2, back, key=("g", "loss"))
            for j in range(3):
                ea, eb = self.ends(self.hid[j], self.out)
                self.dot(sc, eb, ea, remap(b1, 0.1 * j, 0.7 + 0.1 * j), GRAD, 4.2, back,
                         key=("g", j))
        else:
            self._train_flow(sc, t, back)

        # ---- nodes
        for i, c in enumerate(self.inp):
            on = smoothstep(0.0, 0.6, fx)
            fill = C.mix(NODE, DATA_HI if x[i] > 0 else C.mix(DATA_HI, NODE, 0.7), on)
            self.node(sc, c, fill, num(x[i], 0) if on > 0.5 else "", intro)
            self.label(sc, c, "x" + SUB[i], dy=-(R + 14), opacity=intro)
        d_out = self.d0["d_out"] if bu <= 0 else grads(p, x, y)[1]["d_out"]
        dz = self.d0["dz"] if bu <= 0 else grads(p, x, y)[1]["dz"]
        for j, c in enumerate(self.hid):
            on = smoothstep(0.0, 0.25, remap(fh, j / 3, j / 3 + 0.33))
            fill = C.mix(NODE, C.mix(NODE, DATA_HI, float(h[j]) * 0.9 + 0.1), on)
            ring = smoothstep(0.0, 0.2, remap(lit_h, j / 3, j / 3 + 0.33)) * rec
            tang = smoothstep(0.0, 0.2, remap(bh, j / 3, j / 3 + 0.33)) * \
                (1 - smoothstep(0.3, 0.6, remap(bh, j / 3, j / 3 + 0.33)) if bu <= 0 else 0.0)
            self.node(sc, c, fill, num(h[j]) if on > 0.5 else "", intro, ring=ring)
            self.icon(sc, c, z[j] if on > 0 else -6.0, intro, tangent=tang)
            self.label(sc, c, "h" + SUB[j], dx=-(R + 8), dy=-(R + 2), anchor="rm",
                       opacity=intro)
            if not training:
                self.label(sc, c, "z = " + num(z[j]), dy=R + 13, ink=_ink(DATA),
                           opacity=on * back, size=layout.MIN_TEXT)
                got = smoothstep(0.0, 0.2, remap(bh, j / 3, j / 3 + 0.33))  # this example's
                self.label(sc, c, "δ = " + num(dz[j], 3), dy=R + 31, ink=_ink(GRAD),
                           opacity=ring * got, size=layout.MIN_TEXT)
        on = smoothstep(0.0, 0.5, fo)
        fill = C.mix(NODE, C.mix(NODE, DATA_HI, float(yh) * 0.9 + 0.1), on)
        ring = smoothstep(0.3, 0.6, lit_o) * rec
        tang = smoothstep(0.1, 0.3, bo) * (1 - smoothstep(0.85, 1.0, bo)) if bu <= 0 else 0.0
        self.node(sc, self.out, fill, num(yh) if on > 0.5 else "", intro, ring=ring)
        self.icon(sc, self.out, float(o) if on > 0 else -6.0, intro, tangent=tang)
        self.label(sc, self.out, "ŷ", dx=R + 8, dy=-(R + 2), anchor="lm", opacity=intro)
        if not training:
            self.label(sc, self.out, "z = " + num(o), dy=R + 13, ink=_ink(DATA),
                       opacity=on * back, size=layout.MIN_TEXT)
            self.label(sc, self.out, "δ = " + num(d_out, 3), dy=R + 31, ink=_ink(GRAD),
                       opacity=ring * smoothstep(0.3, 0.6, bo), size=layout.MIN_TEXT)

        # ---- the loss
        L = 0.5 * (yh - y) ** 2
        lo = smoothstep(0.0, 0.3, fl)
        self.rect(sc, self.loss, 112, 64, (0.97, 0.97, 0.96), intro, border=(0.55, 0.57, 0.59),
                  border_w=1.2, radius=8, lift=0.8,
                  glow=0.0)
        self.label(sc, self.shift(self.loss, 0, -13), "L = ½(ŷ − y)²", layout.MIN_TEXT,
                   opacity=intro)
        if training:
            Lm = self.losses[min(step, STEPS)]
            self.label(sc, self.shift(self.loss, 0, 13), "mean " + num(Lm, 3), layout.MIN_TEXT,
                       ink=layout.STRONG_INK, opacity=intro)
        else:
            self.label(sc, self.shift(self.loss, 0, 13), "= " + num(L, 3), layout.MIN_TEXT,
                       ink=layout.STRONG_INK, opacity=lo * back)
        self.label(sc, self.loss, "target y = %d" % y, layout.MIN_TEXT, dy=48,
                   opacity=intro * (1 - smoothstep(0.0, 0.3, tr)))

        self._table(sc, p, ex, intro, training)
        self._chart(sc, step, tr, back)

        # ---- words
        self.step_indicator(sc, ph, self.PHASES)
        hh = forward(self.p0, X[EX])[1]
        dW = self.d0["dW"]
        if reset > 0 or t < T["x"][0]:
            sub = ("XOR: the output should be 1 when exactly one input is 1. "
                   "2 inputs, 3 hidden neurons, 1 output, sigmoid activations")
        elif t < T["hid"][0]:
            sub = ("forward: the data (teal) flows in. Example x = (1, 0), target 1. "
                   "Each neuron: z = w · x + b, then a = σ(z)")
        elif t < T["f3"][0]:
            sub = ("each neuron's activation function: the dot is z on the sigmoid; "
                   "h%s = σ(%s) = %s" % (SUB[0], num(self.f0[0][0]), num(self.f0[1][0])))
        elif t < T["b0"][0]:
            sub = ("the loss: L = ½(ŷ − y)² = ½(%s − 1)² = %s"
                   % (num(self.f0[3]), num(0.5 * (self.f0[3] - 1) ** 2, 3)))
        elif t < T["gv"][0]:
            sub = ("backward: the gradient (magenta) flows back from the loss\n"
                   "∂L/∂ŷ = ŷ − y = %s; times the slope σ′(z) = %s: the output's δ = %s"
                   % (num(self.d0["dy"]), num(self.f0[3] * (1 - self.f0[3]), 3),
                      num(self.d0["d_out"], 3)))
        elif t < T["b1"][0]:
            sub = ("each weight's gradient = δ × the value that came in: "
                   "∂L/∂v%s = %s × %s = %s" % (SUB[0], num(self.d0["d_out"], 3), num(hh[0]),
                                                num(self.d0["dv"][0], 3)))
        elif t < T["gw"][0]:
            sub = ("the hidden δs: back through each weight and the slope of each sigmoid: "
                   "δ%s = %s × %s × %s = %s"
                   % (SUB[0], num(self.d0["d_out"], 3), num(unpack(self.p0)[2][0]),
                      num(hh[0] * (1 - hh[0]), 3), num(self.d0["dz"][0], 3)))
        elif t < T["batch"][0]:
            sub = ("input weights: ∂L/∂w = δ × x. x%s = 0, so its weights get no gradient "
                   "from this example (∂L/∂w%s%s = %s)"
                   % (SUB[1], SUB[0], SUB[0], num(dW[0, 0], 4)))
        elif t < T["update"][0]:
            sub = ("the same for the other three examples; each weight's gradients are "
                   "averaged over the batch")
        elif t < T["train"][0]:
            W0, W1 = unpack(self.p0)[2][0], unpack(self.p1)[2][0]
            sub = ("update: every weight steps downhill, new w = w − η · ∂L/∂w with η = %g: "
                   "v%s = %s − %g × %s = %s"
                   % (LR, SUB[0], num(W0, 3), LR, num(self.G[9], 4), num(W1, 3)))
        elif tr < 1:
            sub = ("repeat: data flows forward, gradients flow back, weights step. "
                   "Step %d of %d" % (step, STEPS))
        else:
            yhs = [forward(self.hist[-1], xx)[3] for xx in X]
            sub = ("after %d steps the network computes XOR: ŷ = %s" %
                   (STEPS, ", ".join(num(v) for v in yhs)))
        self.subtitle = sub

    def _train_flow(self, sc, t, back):
        """Data forward and gradients back, at the same time, on every edge."""
        u0 = (t - self.Tm["train"][0])
        n = 0
        segs = [(self.ends(a, b), 0.9) for _k, _i, _j, a, b in self.edges()]
        segs.append((self.ends(self.out, self.loss, R, 58), 0.7))
        fade = smoothstep(0.0, 0.6, u0) * (1 - smoothstep(-0.8, 0.0, t - self.Tm["train"][1])) \
            * back
        for (ea, eb), speed in segs:
            for k in range(2):
                ph = (u0 * speed + k * 0.5 + hash01(n)) % 1.0
                self.dot(sc, ea, eb, ph, DATA, 3.8, fade, key=("tf", n, k), bend=4)
                ph = (u0 * speed + k * 0.5 + hash01(n, 7) + 0.25) % 1.0
                self.dot(sc, eb, ea, ph, GRAD, 3.8, fade, key=("tb", n, k), bend=-4)
            n += 1

    def _table(self, sc, p, ex, op, training):
        x0, y0 = TABLE_AT
        rows = 22.0 / (540 * 0.9)
        cols = [0.0, 0.045, 0.095, 0.15]
        self.label(sc, self.P(x0, y0), "XOR data", layout.MIN_TEXT, anchor="lm", opacity=op)
        for c, head in zip(cols, ("x" + SUB[0], "x" + SUB[1], "y", "ŷ")):
            self.label(sc, self.P(x0 + c, y0 + 1.4 * rows), head, layout.MIN_TEXT, anchor="lm",
                       opacity=op, ink=layout.NOTE_INK)
        for k in range(4):
            yy = y0 + (2.6 + k * 1.1) * rows
            yh = forward(p, X[k])[3]
            vals = ["%d" % X[k, 0], "%d" % X[k, 1], "%d" % Y[k], num(yh)]
            hot = (k == ex) and not training
            if hot:
                self.rect(sc, self.P(x0 + 0.083, yy), 178, 21, (1.0, 1.0, 1.0), op * 0.9,
                          border=DATA, border_w=1.2, radius=5, lift=0.2)
            for c, v in zip(cols, vals):
                ink = layout.STRONG_INK if c == cols[-1] else layout.CAPTION_INK
                self.label(sc, self.P(x0 + c, yy), v, layout.MIN_TEXT, anchor="lm", opacity=op,
                           ink=ink, kind="mono")

    def _chart(self, sc, step, tr, back):
        x0, y0, x1, y1 = CHART
        op = smoothstep(0.0, 0.08, tr) * back
        if op <= 0.01:
            return
        a, b = self.P(x0, y0), self.P(x1, y1)
        mid = (a + b) / 2
        w, h = (self.px(b) - self.px(a))
        self.rect(sc, mid, w, h, (0.96, 0.96, 0.95), op, border=(0.70, 0.71, 0.72),
                  border_w=1.0, radius=4, lift=0.1)
        Lmax = float(self.losses.max())

        def pt(s, L):
            return self.P(x0 + 0.01 + (x1 - x0 - 0.02) * s / STEPS,
                          y1 - 0.02 - (y1 - y0 - 0.1) * L / Lmax, 0.3)
        ks = np.unique(np.linspace(0, step, 90).astype(int))
        if len(ks) > 1:
            self.polyline(sc, [pt(k, self.losses[k]) for k in ks], width=2.0, color=GRAD,
                          opacity=op * 0.9)
        self.disk(sc, pt(step, self.losses[step]), 3.4, GRAD, op, lift=0.6, outline=0.0)
        self.label(sc, self.P(x0 + 0.012, y0 + 0.035), "loss", layout.MIN_TEXT, anchor="lm",
                   opacity=op, ink=layout.NOTE_INK)
        self.label(sc, self.P(x1 - 0.012, y0 + 0.035), "step %d" % step, layout.MIN_TEXT,
                   anchor="rm", opacity=op, ink=layout.NOTE_INK, kind="mono")


def _ink(c):
    """A text colour from a flow colour, dark enough to read on the backdrop."""
    return tuple(float(v) for v in C.darken(c, 0.25))

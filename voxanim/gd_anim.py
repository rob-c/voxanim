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

All numbers are real (:func:`train`): plain full-batch gradient descent,
seeded, with the 13 parameters packed in one vector (see :func:`unpack`).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from . import color as C
from . import layout
from .diagram import DATA, DATA_HI, EDGE, GRAD, NODE, Diagram
from .easing import ease_in_out, remap, smoothstep
from .netanim import hash01
from .text import num, sub

# ------------------------------------------------------------------ the model

XOR_X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], np.float64)
XOR_Y = np.array([0, 1, 1, 0], np.float64)
EX = 2                                  # the worked example: x = (1, 0), y = 1
LR, STEPS, SEED = 2.0, 2000, 0          # learning rate, steps, initialisation seed


def sigmoid(z):
    """The logistic function 1 / (1 + e^-z)."""
    return 1.0 / (1.0 + np.exp(-z))


def unpack(p):
    """The 13-vector of parameters as ``(W (2, 3), b (3,), v (3,), c)``:
    input-to-hidden weights and biases, hidden-to-output weights and bias."""
    return p[:6].reshape(2, 3), p[6:9], p[9:12], p[12]


def forward(p, x):
    """``(z, h, o, y^)``: hidden pre-activations and values, output pre-activation
    and value, for one input ``x`` (2,)."""
    W, b, v, c = unpack(p)
    z = x @ W + b
    h = sigmoid(z)
    o = h @ v + c
    return z, h, o, sigmoid(o)


def grads(p, x, y):
    """Gradient of 1/2 (y^ - y)^2 for one example, packed like ``p``, and the
    intermediate quantities the animation shows: ``dy`` = dL/dy^, ``d_out``
    and ``dz`` (the output's and hidden neurons' deltas), ``dv``, ``dW``."""
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


def mean_loss(p):
    """Mean of 1/2 (y^ - y)^2 over the four XOR examples."""
    yh = np.array([forward(p, x)[3] for x in XOR_X])
    return float(np.mean(0.5 * (yh - XOR_Y) ** 2))


@cache
def train(seed=SEED, steps=STEPS, lr=LR):
    """Full-batch gradient descent on the mean loss.

    Returns ``(params, losses)``: the parameters before every step and after
    the last, ``(steps + 1, 13)``, and the mean loss at each of them.
    """
    rng = np.random.default_rng(seed)
    p = np.concatenate([rng.normal(0, 1, 6), np.zeros(3), rng.normal(0, 1, 3), [0.0]])
    hist, losses = [p.copy()], []
    for _ in range(steps):
        losses.append(mean_loss(p))
        G = np.mean([grads(p, x, y)[0] for x, y in zip(XOR_X, XOR_Y)], axis=0)
        p = p - lr * G
        hist.append(p.copy())
    losses.append(mean_loss(p))
    return np.array(hist), np.array(losses)


# ------------------------------------------------------------------ layout (fractions)

IN_AT = [(0.05, 0.45), (0.05, 0.70)]
HID_AT = [(0.25, 0.32), (0.25, 0.57), (0.25, 0.82)]
OUT_AT = (0.45, 0.57)
LOSS_AT = (0.615, 0.57)
TABLE_AT = (0.80, 0.30)                 # top-left of the XOR table
CHART = (0.745, 0.67, 0.985, 0.96)      # loss chart: x0, y0, x1, y1
R = 22.0                                # neuron radius, px
LOSS_W, LOSS_H = 112.0, 64.0            # the loss box, px
LOSS_GAP = 58.0                         # where the output-to-loss edge stops, px
ICON_W, ICON_H = 46.0, 26.0             # each neuron's sigmoid plot, px
WEIGHT_MAX = 6.0                        # |w| drawn at full width and colour
PANEL = (0.95, 0.95, 0.94)              # light panels: icons, chart
PANEL_EDGE = (0.70, 0.71, 0.72)
CURVE = (0.36, 0.38, 0.40)

# When each beat happens (seconds).  f* forward, b* backward, g* weight gradients.
TIMELINE = dict(intro=(0.2, 1.4), x=(1.4, 2.4), f1=(2.4, 4.2), hid=(4.2, 6.0),
                f2=(6.0, 7.2), outn=(7.2, 8.4), f3=(8.4, 9.4), lossv=(9.4, 11.4),
                b0=(11.6, 12.8), bout=(12.8, 15.0), gv=(15.0, 17.4), b1=(17.4, 18.8),
                bhid=(18.8, 23.0), gw=(23.0, 25.8), batch=(26.2, 31.8),
                update=(32.2, 36.2), train=(36.8, 55.0), final=(55.0, 56.0))
T_DONE = 56.4


def w_color(w):
    """Edge colour for weight ``w``: grey when small, orange (+) / blue (-) when large."""
    k = min(1.0, abs(w) / WEIGHT_MAX) ** 0.7
    base = C.POSITIVE if w >= 0 else C.NEGATIVE
    return C.mix(EDGE, C.mix(base, EDGE, 0.25), k)


def _ink(c):
    """A text colour from a flow colour, dark enough to read on the backdrop."""
    return tuple(float(v) for v in C.darken(c, 0.25))


def _stagger(u, j, n=3, width=0.33):
    """Progress of item ``j`` of ``n`` when they start one after another."""
    return remap(u, j / n, j / n + width)


@dataclass
class Frame:
    """Everything one frame shows, worked out from ``t`` once (see ``_frame``).

    Progress values run 0 -> 1: ``fwd`` per forward beat and ``bwd`` per
    backward beat (keys as in :data:`TIMELINE`); ``lit`` is how far each
    *recorded* gradient has appeared (it stays lit through the batch).
    """

    back: float                 # 1, fading to 0 in the loop reset
    intro: float                # the diagram fading in
    p: np.ndarray               # the parameters on show
    step: int                   # training step on show
    tr: float                   # progress through the training phase
    upd: float                  # progress through the update
    bu: float                   # progress through the batch
    ex: int                     # the example on show (row of XOR_X)
    out: tuple                  # forward(p, x): z, h, o, y^
    d: dict                     # grads(p, x, y)[1] for the example on show
    fwd: dict
    bwd: dict
    lit: dict
    g_rec: np.ndarray           # the recorded gradient (one example, then the batch mean)
    rec: float                  # recorded gradients' visibility (they drain in the update)

    @property
    def training(self):
        """True in the training phase."""
        return self.tr > 0

    @property
    def x(self):
        """The input on show."""
        return XOR_X[self.ex]

    @property
    def y(self):
        """Its target."""
        return XOR_Y[self.ex]


class GradientDescent(Diagram):
    """The whole lesson: one example forward and back in detail, the batch, the
    update, then 2,000 steps of training (see the module docstring)."""

    PHASES = ("forward", "loss", "backward", "batch", "update", "train")

    def __init__(self):
        super().__init__("Gradient descent", "")
        self.hist, self.losses = train()
        self.p0, self.p1 = self.hist[0], self.hist[1]
        self.f0 = forward(self.p0, XOR_X[EX])
        self.g0, self.d0 = grads(self.p0, XOR_X[EX], XOR_Y[EX])
        self.gall = [grads(self.p0, x, y)[0] for x, y in zip(XOR_X, XOR_Y)]
        self.G = np.mean(self.gall, axis=0)
        self.inp = [self.P(*a) for a in IN_AT]
        self.hid = [self.P(*a) for a in HID_AT]
        self.out = self.P(*OUT_AT)
        self.loss = self.P(*LOSS_AT)
        self.hold, self.reset = 3.0, 1.5
        self.Tm = TIMELINE
        self.finish(T_DONE)


    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        for k, name in ((5, "train"), (4, "update"), (3, "batch"), (2, "b0"), (1, "f3")):
            if t >= self.Tm[name][0]:
                return k
        return 0

    # ------------------------------------------------------------------ geometry
    def edges(self):
        """``[(kind, i, j, src, dst)]``: w_ij (input i to hidden j), v_i (hidden i to out)."""
        out = [("w", i, j, self.inp[i], self.hid[j]) for i in range(2) for j in range(3)]
        return out + [("v", j, 0, self.hid[j], self.out) for j in range(3)]

    @staticmethod
    def w_of(p, kind, i, j):
        """The value of edge ``(kind, i, j)``'s weight in parameters ``p``."""
        W, _b, v, _c = unpack(p)
        return W[i, j] if kind == "w" else v[i]

    @staticmethod
    def g_index(kind, i, j):
        """Index of edge ``(kind, i, j)``'s weight in the packed parameter vector."""
        return i * 3 + j if kind == "w" else 9 + i

    # ------------------------------------------------------------------ the frame's state
    def _frame(self, t, reset):
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        intro = smoothstep(0.0, 1.0, self.progress("intro", t)) * back
        # which parameters are on show
        upd = ease_in_out(self.progress("update", t))
        tr = self.progress("train", t)
        step = int(round(STEPS * tr)) if tr > 0 else (1 if upd >= 1 else 0)
        p = self.hist[step] if tr > 0 else self.p0 + (self.p1 - self.p0) * upd
        # the example on show: the worked one, the batch, then the worked one again
        bu = self.progress("batch", t)
        others = [k for k in range(4) if k != EX]
        ex = others[min(2, int(bu * 3))] if 0 < bu < 1 else EX
        fwd = {k: self.progress(k, t) for k in ("x", "f1", "hid", "f2", "outn", "f3", "lossv")}
        bwd = {k: self.progress(k, t) for k in ("b0", "bout", "gv", "b1", "bhid", "gw")}
        lit = dict(v=bwd["gv"], w=bwd["gw"], o=bwd["bout"], h=bwd["bhid"])
        if 0 < bu < 1:                          # each other example: forward then back
            u = (bu * 3) % 1.0
            fwd = dict.fromkeys(fwd, smoothstep(0.0, 0.45, u))
            bwd = dict.fromkeys(bwd, smoothstep(0.5, 0.95, u))
        if tr > 0 or upd > 0:
            fwd = dict.fromkeys(fwd, 1.0)
        # recorded gradients: the worked example, then the running batch average
        g_rec = self.g0
        if bu > 0:
            n = 1 + min(3, int(bu * 3 + (1 if bu >= 1 else 0)))
            got = [self.gall[EX]] + [self.gall[k] for k in others][:n - 1]
            g_rec = self.G if bu >= 1 else np.mean(got, axis=0)
        rec = 0.0 if tr > 0 else (1.0 - smoothstep(0.55, 0.95, upd)) * back
        d = self.d0 if bu <= 0 else grads(p, XOR_X[ex], XOR_Y[ex])[1]
        return Frame(back, intro, p, step, tr, upd, bu, ex, forward(p, XOR_X[ex]), d, fwd,
                     bwd, lit, g_rec, rec)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        f = self._frame(t, reset)
        self._edges(sc, f)
        self._data_flow(sc, f)
        if f.training:
            self._train_flow(sc, t, f.back)
        else:
            self._grad_flow(sc, f)
        self._inputs(sc, f)
        self._hidden(sc, f)
        self._output(sc, f)
        self._loss_box(sc, f)
        self._table(sc, f.p, f.ex, f.intro, f.training)
        self._chart(sc, f.step, f.tr, f.back)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, f, reset)

    def _edges(self, sc, f):
        """Weights as edges (width and colour = size and sign) and, once
        computed, each weight's recorded gradient as a magenta band."""
        gmax = np.abs(self.G).max() if f.bu > 0 else np.abs(self.g0).max()
        for kind, i, j, a, b in self.edges():
            w = self.w_of(f.p, kind, i, j)
            ea, eb = self.ends(a, b, R, R)
            width = 0.8 + 3.2 * min(1.0, abs(w) / WEIGHT_MAX) ** 0.7
            sc.line(ea, eb, width=width, color=w_color(w), opacity=0.45 * f.intro, bias=0.0)
            lit = smoothstep(0.0, 0.25, _stagger(f.lit["v"], i, width=0.3) if kind == "v" else
                             _stagger(f.lit["w"], j, width=0.3))
            g = f.g_rec[self.g_index(kind, i, j)]
            if lit > 0 and f.rec > 0 and abs(g) > 1e-9:
                k = min(1.0, abs(g) / gmax) ** 0.5
                off = self.Mi @ np.array([0.0, -3.0])
                sc.line(ea + off, eb + off, width=1.0 + 3.0 * k, color=GRAD,
                        opacity=min(0.48, 0.2 + 0.4 * k) * lit * f.rec, bias=0.1)

    def _data_flow(self, sc, f):
        """Teal dots: the inputs into the hidden layer, the hidden values into the
        output, the output into the loss."""
        h = f.out[1]
        for n, (kind, i, j, a, b) in enumerate(self.edges()):
            ea, eb = self.ends(a, b, R, R)
            if kind == "w":
                u, val = remap(f.fwd["f1"], 0.1 * i, 0.6 + 0.1 * i), f.x[i]
            else:
                u, val = remap(f.fwd["f2"], 0.1 * j, 0.6 + 0.1 * j), h[i]
            if not f.training:
                self.dot(sc, ea, eb, u, DATA if val > 0 else C.mix(DATA, EDGE, 0.6),
                         4.2 if val > 0 else 2.6, f.back, key=("d", n))
        oa, ob = self.ends(self.out, self.loss, R, LOSS_GAP)
        if not f.training:
            self.dot(sc, oa, ob, f.fwd["f3"], DATA, 4.2, f.back, key=("d", "loss"))

    def _grad_flow(self, sc, f):
        """Magenta dots: the loss's gradient into the output, then the output's
        delta back along each hidden-to-output weight."""
        oa, ob = self.ends(self.out, self.loss, R, LOSS_GAP)
        self.dot(sc, ob, oa, f.bwd["b0"], GRAD, 4.2, f.back, key=("g", "loss"))
        for j in range(3):
            ea, eb = self.ends(self.hid[j], self.out, R, R)
            self.dot(sc, eb, ea, remap(f.bwd["b1"], 0.1 * j, 0.7 + 0.1 * j), GRAD, 4.2,
                     f.back, key=("g", j))

    def _train_flow(self, sc, t, back):
        """Data forward and gradients back, at the same time, on every edge."""
        u0 = t - self.Tm["train"][0]
        segs = [(self.ends(a, b, R, R), 0.9) for _k, _i, _j, a, b in self.edges()]
        segs.append((self.ends(self.out, self.loss, R, LOSS_GAP), 0.7))
        fade = smoothstep(0.0, 0.6, u0) * (1 - smoothstep(-0.8, 0.0, t - self.Tm["train"][1])) \
            * back
        for n, ((ea, eb), speed) in enumerate(segs):
            for k in range(2):
                ph = (u0 * speed + k * 0.5 + hash01(n)) % 1.0
                self.dot(sc, ea, eb, ph, DATA, 3.8, fade, key=("tf", n, k), bend=4)
                ph = (u0 * speed + k * 0.5 + hash01(n, 7) + 0.25) % 1.0
                self.dot(sc, eb, ea, ph, GRAD, 3.8, fade, key=("tb", n, k), bend=-4)

    # ------------------------------------------------------------------ nodes
    def icon(self, sc, c, z, op, tangent=0.0):
        """The neuron's activation function above it: sigma over z in [-6, 6],
        the operating point at ``z`` and, with ``tangent`` > 0, the slope there
        (the derivative the gradient is multiplied by)."""
        if op <= 0.01:
            return
        top = self.shift(c, 0.0, -(R + 8 + ICON_H / 2))
        self.rect(sc, top, ICON_W + 8, ICON_H + 8, PANEL, op * 0.9, border=PANEL_EDGE,
                  border_w=1.0, radius=4, lift=0.4)

        def pt(zz, s):
            return self.shift(top, (zz / 12.0) * ICON_W, (0.5 - s) * ICON_H) + self.view * 0.7
        pts = [pt(zz, sigmoid(zz)) for zz in np.linspace(-6, 6, 25)]
        self.polyline(sc, pts, width=1.4, color=CURVE, opacity=0.9 * op)
        zc = float(np.clip(z, -6, 6))
        if tangent > 0:
            s = sigmoid(zc) * (1 - sigmoid(zc))
            dz = 2.2
            a = pt(zc - dz, sigmoid(zc) - s * dz)
            b = pt(zc + dz, sigmoid(zc) + s * dz)
            sc.line(a, b, width=2.0, color=GRAD, opacity=op * tangent, bias=0.6)
        self.disk(sc, pt(zc, sigmoid(zc)), 3.2, DATA, op, lift=1.3, outline=0.0)

    def node(self, sc, c, fill, text, op, ring=0.0):
        """A neuron: a disk with its value inside and a magenta ring (opacity
        ``ring``) once its gradient is recorded."""
        self.disk(sc, c, R, fill, op, ring=GRAD, ring_w=3.0, ring_op=ring)
        self.label(sc, c, text, layout.MIN_TEXT, ink=layout.STRONG_INK, opacity=op,
                   kind="mono")

    def _inputs(self, sc, f):
        for i, c in enumerate(self.inp):
            on = smoothstep(0.0, 0.6, f.fwd["x"])
            fill = C.mix(NODE, DATA_HI if f.x[i] > 0 else C.mix(DATA_HI, NODE, 0.7), on)
            self.node(sc, c, fill, num(f.x[i], 0) if on > 0.5 else "", f.intro)
            self.label(sc, c, "x" + sub(i + 1), dy=-(R + 14), opacity=f.intro)

    def _hidden(self, sc, f):
        z, h = f.out[0], f.out[1]
        bh = f.bwd["bhid"]
        for j, c in enumerate(self.hid):
            on = smoothstep(0.0, 0.25, _stagger(f.fwd["hid"], j))
            fill = C.mix(NODE, C.mix(NODE, DATA_HI, float(h[j]) * 0.9 + 0.1), on)
            ring = smoothstep(0.0, 0.2, _stagger(f.lit["h"], j)) * f.rec
            here = _stagger(bh, j)                       # this neuron's backward beat
            tang = smoothstep(0.0, 0.2, here) * \
                (1 - smoothstep(0.3, 0.6, here) if f.bu <= 0 else 0.0)
            self.node(sc, c, fill, num(h[j]) if on > 0.5 else "", f.intro, ring=ring)
            self.icon(sc, c, z[j] if on > 0 else -6.0, f.intro, tangent=tang)
            self.label(sc, c, "h" + sub(j + 1), dx=-(R + 8), dy=-(R + 2), anchor="rm",
                       opacity=f.intro)
            if not f.training:
                self.label(sc, c, "z = " + num(z[j]), dy=R + 13, ink=_ink(DATA),
                           opacity=on * f.back, size=layout.MIN_TEXT)
                got = smoothstep(0.0, 0.2, here)         # this example's delta is known
                self.label(sc, c, "δ = " + num(f.d["dz"][j], 3), dy=R + 31, ink=_ink(GRAD),
                           opacity=ring * got, size=layout.MIN_TEXT)

    def _output(self, sc, f):
        o, yh = f.out[2], f.out[3]
        bo = f.bwd["bout"]
        on = smoothstep(0.0, 0.5, f.fwd["outn"])
        fill = C.mix(NODE, C.mix(NODE, DATA_HI, float(yh) * 0.9 + 0.1), on)
        ring = smoothstep(0.3, 0.6, f.lit["o"]) * f.rec
        tang = smoothstep(0.1, 0.3, bo) * (1 - smoothstep(0.85, 1.0, bo)) if f.bu <= 0 else 0.0
        self.node(sc, self.out, fill, num(yh) if on > 0.5 else "", f.intro, ring=ring)
        self.icon(sc, self.out, float(o) if on > 0 else -6.0, f.intro, tangent=tang)
        self.label(sc, self.out, "ŷ", dx=R + 8, dy=-(R + 2), anchor="lm", opacity=f.intro)
        if not f.training:
            self.label(sc, self.out, "z = " + num(o), dy=R + 13, ink=_ink(DATA),
                       opacity=on * f.back, size=layout.MIN_TEXT)
            self.label(sc, self.out, "δ = " + num(f.d["d_out"], 3), dy=R + 31, ink=_ink(GRAD),
                       opacity=ring * smoothstep(0.3, 0.6, bo), size=layout.MIN_TEXT)

    def _loss_box(self, sc, f):
        """The loss: the formula, its value for the example (or the mean while
        training) and the target."""
        L = 0.5 * (f.out[3] - f.y) ** 2
        self.rect(sc, self.loss, LOSS_W, LOSS_H, (0.97, 0.97, 0.96), f.intro,
                  border=(0.55, 0.57, 0.59), border_w=1.2, radius=8, lift=0.8)
        self.label(sc, self.shift(self.loss, 0, -13), "L = ½(ŷ − y)²", layout.MIN_TEXT,
                   opacity=f.intro)
        if f.training:
            self.label(sc, self.shift(self.loss, 0, 13),
                       "mean " + num(self.losses[min(f.step, STEPS)], 3), layout.MIN_TEXT,
                       ink=layout.STRONG_INK, opacity=f.intro)
        else:
            self.label(sc, self.shift(self.loss, 0, 13), "= " + num(L, 3), layout.MIN_TEXT,
                       ink=layout.STRONG_INK,
                       opacity=smoothstep(0.0, 0.3, f.fwd["lossv"]) * f.back)
        self.label(sc, self.loss, "target y = %d" % f.y, layout.MIN_TEXT, dy=48,
                   opacity=f.intro * (1 - smoothstep(0.0, 0.3, f.tr)))

    # ------------------------------------------------------------------ side panels
    def _table(self, sc, p, ex, op, training):
        """The four XOR examples and the network's current prediction for each;
        the example on show is outlined."""
        x0, y0 = TABLE_AT
        rows = 22.0 / (layout.REF_H * self.cam.zoom)            # one row, as a fraction
        cols = [0.0, 0.045, 0.095, 0.15]
        self.label(sc, self.P(x0, y0), "XOR data", layout.MIN_TEXT, anchor="lm", opacity=op)
        for c, head in zip(cols, ("x" + sub(1), "x" + sub(2), "y", "ŷ")):
            self.label(sc, self.P(x0 + c, y0 + 1.4 * rows), head, layout.MIN_TEXT, anchor="lm",
                       opacity=op, ink=layout.NOTE_INK)
        for k in range(4):
            yy = y0 + (2.6 + k * 1.1) * rows
            yh = forward(p, XOR_X[k])[3]
            vals = ["%d" % XOR_X[k, 0], "%d" % XOR_X[k, 1], "%d" % XOR_Y[k], num(yh)]
            if k == ex and not training:
                self.rect(sc, self.P(x0 + 0.083, yy), 178, 21, (1.0, 1.0, 1.0), op * 0.9,
                          border=DATA, border_w=1.2, radius=5, lift=0.2)
            for c, v in zip(cols, vals):
                ink = layout.STRONG_INK if c == cols[-1] else layout.CAPTION_INK
                self.label(sc, self.P(x0 + c, yy), v, layout.MIN_TEXT, anchor="lm", opacity=op,
                           ink=ink, kind="mono")

    def _chart(self, sc, step, tr, back):
        """The mean loss against the training step, drawn up to ``step``."""
        x0, y0, x1, y1 = CHART
        op = smoothstep(0.0, 0.08, tr) * back
        if op <= 0.01:
            return
        a, b = self.P(x0, y0), self.P(x1, y1)
        w, h = self.px(b) - self.px(a)
        self.rect(sc, (a + b) / 2, w, h, (0.96, 0.96, 0.95), op, border=PANEL_EDGE,
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

    # ------------------------------------------------------------------ words
    def _words(self, t, f, reset):
        """The subtitle: the first entry of :meth:`_script` whose beat has not
        started yet, or the last one."""
        if reset > 0 or t < self.Tm["x"][0]:
            return self._script(f)[0][1]
        for beat, text in self._script(f)[1:]:
            if beat is None or t < self.Tm[beat][0]:
                return text() if callable(text) else text
        raise AssertionError("unreachable: the script ends with a catch-all")

    def _script(self, f):
        """``[(shown until this beat starts, text)]`` in order; ``None`` = the rest."""
        d0, f0 = self.d0, self.f0
        h0 = f0[1]
        v0, v1 = unpack(self.p0)[2][0], unpack(self.p1)[2][0]
        return [
            (None, "XOR: the output should be 1 when exactly one input is 1. "
                   "2 inputs, 3 hidden neurons, 1 output, sigmoid activations"),
            ("hid", "forward: the data (teal) flows in. Example x = (1, 0), target 1. "
                    "Each neuron: z = w · x + b, then a = σ(z)"),
            ("f3", "each neuron's activation function: the dot is z on the sigmoid; "
                   "h%s = σ(%s) = %s" % (sub(1), num(f0[0][0]), num(h0[0]))),
            ("b0", "the loss: L = ½(ŷ − y)² = ½(%s − 1)² = %s"
                   % (num(f0[3]), num(0.5 * (f0[3] - 1) ** 2, 3))),
            ("gv", "backward: the gradient (magenta) flows back from the loss\n"
                   "∂L/∂ŷ = ŷ − y = %s; times the slope σ′(z) = %s: the output's δ = %s"
                   % (num(d0["dy"]), num(f0[3] * (1 - f0[3]), 3), num(d0["d_out"], 3))),
            ("b1", "each weight's gradient = δ × the value that came in: "
                   "∂L/∂v%s = %s × %s = %s" % (sub(1), num(d0["d_out"], 3), num(h0[0]),
                                                num(d0["dv"][0], 3))),
            ("gw", "the hidden δs: back through each weight and the slope of each sigmoid: "
                   "δ%s = %s × %s × %s = %s"
                   % (sub(1), num(d0["d_out"], 3), num(v0), num(h0[0] * (1 - h0[0]), 3),
                      num(d0["dz"][0], 3))),
            ("batch", "input weights: ∂L/∂w = δ × x. x%s = 0, so its weights get no gradient "
                      "from this example (∂L/∂w%s%s = %s)"
                      % (sub(2), sub(1), sub(1), num(d0["dW"][0, 0], 4))),
            ("update", "the same for the other three examples; each weight's gradients are "
                       "averaged over the batch"),
            ("train", "update: every weight steps downhill, new w = w − η · ∂L/∂w with "
                      "η = %g: v%s = %s − %g × %s = %s"
                      % (LR, sub(1), num(v0, 3), LR, num(self.G[9], 4), num(v1, 3))),
            (None, lambda: self._training_words(f)),
        ]

    def _training_words(self, f):
        if f.tr < 1:
            return ("repeat: data flows forward, gradients flow back, weights step. "
                    "Step %d of %d" % (f.step, STEPS))
        yhs = [forward(self.hist[-1], x)[3] for x in XOR_X]
        return ("after %d steps the network computes XOR: ŷ = %s"
                % (STEPS, ", ".join(num(v) for v in yhs)))

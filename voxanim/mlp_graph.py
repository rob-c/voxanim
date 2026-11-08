"""A multi-layer dense classifier on MNIST, drawn as the classic diagram:
neurons as nodes in columns, weights as the lines between them.

    image 28x28  ->  32 neurons  ->  16 neurons  ->  10 outputs  ->  softmax
    (784 values)     ReLU            ReLU            logits          p(digit)

A real network (:func:`voxanim.mnist.train`, 784-32-16-10, ~96% on the test
set).  Each line's opacity is how much it carried for this digit
(|weight x incoming value|; orange adds, blue subtracts), so only the paths
that matter show.  A neuron is dark when ReLU switched it off (z <= 0) and
lit by how large its value is otherwise.

1. one digit, layer by layer, with one neuron per layer worked out in
   numbers: z = sum w x + b, then ReLU,
2. softmax turns the 10 logits into probabilities; the largest is the answer,
3. more test digits go through, one after another, including a mistake.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from . import color as C
from . import layout
from .diagram import DATA, EDGE, NODE, Diagram, draw_image
from .easing import ease_in_out, remap, smoothstep
from .netanim import ACT_HI, ACT_LO, GRAY_HI, GRAY_LO, OFF, hash01

SIZES = (784, 32, 16, 10)
EPOCHS = 6
FIRST = 8                                   # a '5' (p = 0.89; 6 is next)
MORE = (0, 2, 3, 20, 4, 38, 11)             # 7 1 0 9 4 2 6 -- the 2 is read as a 3
IMG_AT, IMG_PX = (0.09, 0.60), 150.0
COL_X = (0.36, 0.56, 0.755)
SPAN = ((0.27, 0.98), (0.30, 0.95), (0.29, 0.96))
RADII = (4.8, 7.6, 12.5)
BAR_DX, BAR_LEN = 24.0, 112.0              # px right of an output node
TOP_IN = 6                                  # strongest pixel lines per layer-1 neuron
MINUS = "−"
DIM = C.rgb("#c4c8ca")                      # an unlikely output


def num(v, d=2):
    s = "%.*f" % (d, v)
    return MINUS + s[1:] if s.startswith("-") else s


@cache
def model():
    from . import mnist
    params, acc = mnist.train(sizes=SIZES, epochs=EPOCHS)
    _xtr, _ytr, xte, yte = mnist.load()
    return params, acc, xte, yte


def run(params, x):
    zs, acts = [], [x]
    a = x
    for n, (W, b) in enumerate(params):
        z = a @ W + b
        zs.append(z)
        a = np.maximum(z, 0) if n < len(params) - 1 else z
        acts.append(a)
    e = np.exp(zs[-1] - zs[-1].max())
    return zs, acts, e / e.sum()


class MLPClassifier(Diagram):
    PHASES = ("image", "layer 1", "layer 2", "output", "more digits")

    def __init__(self):
        super().__init__("A dense network classifies digits", "")
        self.params, self.acc, xte, yte = model()
        self.digits = [FIRST] + list(MORE)
        self.runs = []
        for i in self.digits:
            zs, acts, p = run(self.params, xte[i])
            self.runs.append(dict(x=xte[i], label=int(yte[i]), zs=zs, acts=acts, p=p,
                                  pred=int(p.argmax())))
        self.nodes = []
        for k in range(3):
            n = SIZES[k + 1]
            y0, y1 = SPAN[k]
            self.nodes.append([self.P(COL_X[k], y0 + (y1 - y0) * i / (n - 1)) for i in range(n)])
        self.img = self.P(*IMG_AT)
        self.pix = [self._pixel(i) for i in range(784)]
        # the neuron worked out in numbers, per layer: the strongest lit one
        r0 = self.runs[0]
        self.feat = [int(np.argmax(r0["acts"][1])), int(np.argmax(r0["acts"][2])),
                     r0["pred"]]
        self.hold, self.reset = 3.0, 1.2
        T = dict(intro=(0.2, 1.2), image=(1.0, 2.4), wires=(2.4, 4.0),
                 f1=(4.2, 6.6), l1=(6.6, 9.0), f2=(9.2, 11.4), l2=(11.4, 13.6),
                 l3=(13.8, 16.0), soft=(16.0, 18.0), verdict=(18.0, 19.4))
        t = 20.0
        self.quick = []
        for _ in MORE:
            self.quick.append(t)
            t += 2.7
        T["more"] = (20.0, t)
        self.Tm = T
        self.finish(t + 0.4)

    def _pixel(self, i):
        r, c = divmod(i, 28)
        s = IMG_PX / 28
        return self.shift(self.img, (c + 0.5) * s - IMG_PX / 2, (r + 0.5) * s - IMG_PX / 2)

    def p_(self, name, t):
        return remap(t, *self.Tm[name])

    def step_of(self, t):
        T = self.Tm
        for k, name in ((4, "more"), (3, "l3"), (2, "f2"), (1, "f1")):
            if t >= T[name][0]:
                return k
        return 0

    # ------------------------------------------------------------------ state
    def _progress(self, t):
        """(run index, image fade, per-layer progress, softmax, verdict)."""
        T = self.Tm
        if t < T["more"][0]:
            lay = [max(self.p_("f1", t) * 0.35, self.p_("l1", t)),
                   max(self.p_("f2", t) * 0.35, self.p_("l2", t)), self.p_("l3", t)]
            return 0, smoothstep(0.0, 1.0, self.p_("image", t)), lay, self.p_("soft", t), \
                self.p_("verdict", t)
        k = max(i for i, t0 in enumerate(self.quick) if t >= t0 or i == 0)
        u = t - self.quick[k]
        lay = [remap(u, 0.35, 0.95), remap(u, 0.95, 1.45), remap(u, 1.45, 1.85)]
        return k + 1, 1.0, lay, remap(u, 1.85, 2.15), remap(u, 2.1, 2.4)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        T = self.Tm
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        ph = 0 if reset > 0 else self.step_of(t)
        intro = smoothstep(0.0, 1.0, self.p_("intro", t)) * back
        k, img_on, lay, soft, verdict = self._progress(t)
        R = self.runs[k]
        # the previous digit fades out under the new one
        swap = 1.0
        if k > 0:
            swap = smoothstep(0.0, 0.3, t - self.quick[k - 1])
        img = R["x"] if swap >= 1 or k == 0 else \
            self.runs[k - 1]["x"] + (R["x"] - self.runs[k - 1]["x"]) * swap
        tex = GRAY_LO[None] + (GRAY_HI - GRAY_LO)[None] * np.asarray(img)[:, None]
        detail = k == 0
        wires = smoothstep(0.0, 1.0, self.p_("wires", t)) if detail else 1.0

        draw_image(self, sc, self.img, tex.reshape(28, 28, 3), IMG_PX, IMG_PX,
                   img_on * back, grid=0.25)
        self.rect(sc, self.img, IMG_PX + 8, IMG_PX + 8, (0.97, 0.97, 0.96), intro * 0.9,
                  border=(0.62, 0.63, 0.65), border_w=1.0, radius=5, lift=0.0)

        # ---- edges: how much each carried, |w * a|
        for L in range(3):
            W = self.params[L][0]
            a_in = R["acts"][L]
            c = a_in[:, None] * W
            u = lay[L]
            if u <= 0:
                continue
            cmax = float(np.abs(c).max()) or 1.0
            n_out = SIZES[L + 1]
            for j in range(n_out):
                sweep = smoothstep(0.0, 0.2, remap(u, 0.6 * j / n_out, 0.6 * j / n_out + 0.4))
                if detail and L < 2 and u < 0.35:     # only the worked neuron at first
                    sweep = smoothstep(0.0, 0.3, u / 0.35) if j == self.feat[L] else 0.0
                if sweep <= 0:
                    continue
                col = c[:, j]
                if L == 0:
                    idx = np.argsort(-np.abs(col))[:TOP_IN]
                else:
                    idx = np.nonzero(np.abs(col) > 0.06 * cmax)[0]
                for i in idx:
                    v = col[i] / cmax
                    if abs(v) < 0.02:
                        continue
                    src = self.pix[i] if L == 0 else self.nodes[L - 1][i]
                    dst = self.nodes[L][j]
                    ea, eb = self._ends(src, dst, 0.0 if L == 0 else RADII[L - 1], RADII[L])
                    op = (0.10 + 0.38 * abs(v) ** 0.6) * sweep * back
                    if detail and L < 2 and j == self.feat[L]:
                        op = min(0.48, op * 1.6)
                    sc.line(ea, eb, width=0.7 + 1.6 * abs(v) ** 0.6,
                            color=C.mix(EDGE, C.POSITIVE if v > 0 else C.NEGATIVE, 0.75),
                            opacity=op, bias=0.0)
            self._flow(sc, L, R, u, back, detail)

        # ---- faint wiring before the data arrives
        if wires > 0 and lay[0] <= 0:
            for j in range(0, SIZES[1]):
                for corner in ((-1, -1), (1, 1), (1, -1), (-1, 1)):
                    p = self.shift(self.img, corner[0] * IMG_PX / 2, corner[1] * IMG_PX / 2)
                    ea, eb = self._ends(p, self.nodes[0][j], 0.0, RADII[0])
                    sc.line(ea, eb, width=0.7, color=EDGE, opacity=0.22 * wires * back,
                            bias=0.0)

        # ---- nodes
        for L in range(3):
            vals = R["acts"][L + 1]
            vmax = float(np.abs(vals).max()) or 1.0
            n = SIZES[L + 1]
            for j in range(n):
                on = smoothstep(0.0, 0.15, remap(lay[L], 0.6 * j / n + 0.3, 0.6 * j / n + 0.45))
                if detail and L < 2 and j == self.feat[L]:
                    on = max(on, smoothstep(0.75, 1.0, lay[L] / 0.35))
                v = vals[j]
                if L == 2 and soft > 0:          # by probability: pale = unlikely
                    lit = C.mix(DIM, ACT_HI, R["p"][j] ** 0.5)
                elif v > 0:
                    lit = ACT_LO + (ACT_HI - ACT_LO) * (v / vmax) ** 0.6
                else:                            # ReLU off; a negative logit stays readable
                    lit = OFF if L < 2 else DIM
                fill = C.mix(NODE, lit, on)
                ring = None
                ring_op = 0.0
                if detail and L < 2 and j == self.feat[L]:
                    ring, ring_op = (0.12, 0.36, 0.66), smoothstep(0.0, 0.1, lay[L]) * \
                        (1 - smoothstep(0.0, 0.5, t - T["l%d" % (L + 1)][1]))
                if L == 2 and verdict > 0 and j == R["pred"]:
                    good = R["pred"] == R["label"]
                    ring, ring_op = ((0.20, 0.60, 0.30) if good else (0.80, 0.25, 0.20)), \
                        smoothstep(0.0, 0.4, verdict)
                self.disk(sc, self.nodes[L][j], RADII[L], fill, intro, ring=ring,
                          ring_w=2.4, ring_op=ring_op * back)
        for j in range(10):
            self.label(sc, self.nodes[2][j], str(j), layout.MIN_TEXT, opacity=intro,
                       ink=layout.STRONG_INK)

        # ---- softmax bars
        p = R["p"]
        if soft > 0:
            for j in range(10):
                y = self.nodes[2][j]
                a = self.shift(y, RADII[2] + BAR_DX, 0)
                b = self.shift(a, BAR_LEN * p[j] * ease_in_out(soft), 0)
                if p[j] * soft > 0.004:
                    sc.line(a, b, width=9.0, color=DATA if j == R["pred"] else
                            C.mix(DATA, EDGE, 0.55), opacity=0.95 * back, bias=0.1)
                if p[j] >= 0.005:
                    q = self.shift(a, BAR_LEN + 8, 0)
                    self.label(sc, q, num(p[j]), layout.MIN_TEXT, anchor="lm",
                               opacity=smoothstep(0.4, 1.0, soft) * back, kind="mono",
                               ink=layout.STRONG_INK if j == R["pred"] else layout.CAPTION_INK)
        self._captions(sc, t, k, intro, verdict, back)
        self.step_indicator(sc, ph, self.PHASES)
        self.subtitle = self._words(t, k, reset)

    def _ends(self, a, b, ra, rb):
        sa, sb = self.px(a), self.px(b)
        d = max(np.linalg.norm(sb - sa), 1e-6)
        return a + (b - a) * (ra / d), b - (b - a) * (rb / d)

    def _flow(self, sc, L, R, u, back, detail):
        """Teal dots carrying the strongest products into layer ``L``."""
        if not 0 < u < 1:
            return
        W = self.params[L][0]
        c = R["acts"][L][:, None] * W
        if detail and L < 2 and u < 0.35:
            j = self.feat[L]
            idx = np.argsort(-np.abs(c[:, j]))[:14]
            pairs = [(i, j) for i in idx]
            uu = u / 0.35
        else:
            flat = np.argsort(-np.abs(c).ravel())[:48]
            pairs = [divmod(int(m), c.shape[1]) for m in flat]
            uu = remap(u, 0.35, 1.0) if detail and L < 2 else u
        for n, (i, j) in enumerate(pairs):
            src = self.pix[i] if L == 0 else self.nodes[L - 1][i]
            s0 = 0.45 * hash01(n, L)
            self.dot(sc, src, self.nodes[L][j], remap(uu, s0, s0 + 0.55),
                     C.POSITIVE if c[i, j] > 0 else C.NEGATIVE, 2.8, back, key=("f", L, n))

    def _captions(self, sc, t, k, op, verdict, back):
        for L, text in enumerate(("32 neurons", "16 neurons", "10 outputs")):
            self.label(sc, self.nodes[L][0], text, layout.MIN_TEXT, anchor="mb",
                       dy=-(RADII[L] + 10), opacity=op)
        self.label(sc, self.img, "784 pixel values", layout.MIN_TEXT, anchor="mt",
                   dy=IMG_PX / 2 + 10, opacity=op)
        top = self.shift(self.nodes[2][0], RADII[2] + BAR_DX + BAR_LEN / 2 + 30, 0)
        self.label(sc, top, "softmax p", layout.MIN_TEXT, anchor="mb", dy=-(RADII[2] + 10),
                   opacity=op)
        R = self.runs[k]
        if verdict > 0:
            good = R["pred"] == R["label"]
            text = "%d" % R["pred"] + ("  correct" if good else "  (it is a %d)" % R["label"])
            self.label(sc, self.img, "reads: " + text, layout.CAPTION, anchor="mb",
                       dy=-(IMG_PX / 2 + 12), opacity=smoothstep(0.0, 0.5, verdict) * back,
                       ink=layout.STRONG_INK)

    def _words(self, t, k, reset):
        T = self.Tm
        r0 = self.runs[0]
        if reset > 0 or t < T["wires"][0]:
            return ("a trained network (784 → 32 → 16 → 10, %.1f%% on 10,000 test digits) "
                    "reads a handwritten digit" % (100 * self.acc)).replace("→", "-")
        if t < T["f1"][0]:
            return ("the 784 pixel values go in; every pixel connects to every one of the "
                    "32 neurons in layer 1: 784 × 32 weights")
        if t < T["f2"][0]:
            j = self.feat[0]
            z = r0["zs"][0][j]
            if t < T["l1"][0]:
                return ("one neuron: z = Σ wᵢxᵢ + b over all 784 pixels = %s; ReLU keeps it: "
                        "max(0, z) = %s. Lines show the strongest w × x" % (num(z), num(max(z, 0))))
            n_on = int((r0["acts"][1] > 0).sum())
            return ("all 32 at once: %d have z > 0 and light up; ReLU switches the other %d off "
                    "(dark)" % (n_on, 32 - n_on))
        if t < T["l3"][0]:
            j = self.feat[1]
            z = r0["zs"][1][j]
            if t < T["l2"][0]:
                return ("layer 2: each neuron sums the 32 values below, weighted: "
                        "z = %s, ReLU %s. Orange lines add, blue subtract" % (num(z),
                                                                            num(max(z, 0))))
            n_on = int((r0["acts"][2] > 0).sum())
            return "layer 2: %d of 16 neurons light up" % n_on
        if t < T["soft"][0]:
            zz = r0["zs"][2]
            return ("output: 10 scores (logits), one per digit; the largest is %s for '%d'"
                    % (num(zz.max()), int(zz.argmax())))
        if t < T["more"][0]:
            o = np.argsort(-r0["p"])
            return ("softmax: p = exp(z) / Σ exp(z), so the 10 add up to 1. "
                    "'%d': %s, '%d': %s" % (o[0], num(r0["p"][o[0]]), o[1], num(r0["p"][o[1]])))
        done = [self.runs[i] for i in range(1, k + 1)]
        right = sum(r["pred"] == r["label"] for r in done)
        R = self.runs[k]
        u = t - self.quick[k - 1]
        if u >= 2.1:
            right_now = right
            seen = len(done)
        else:
            right_now = right - (R["pred"] == R["label"])
            seen = len(done) - 1
        tally = ("; %d of %d right so far" % (right_now, seen)) if seen else ""
        return "more test digits, the same weights every time" + tally

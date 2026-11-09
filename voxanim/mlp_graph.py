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

from dataclasses import dataclass
from functools import cache

import numpy as np

from . import color as C
from . import layout
from .diagram import DATA, EDGE, NODE, Diagram, draw_image
from .easing import ease_in_out, remap, smoothstep
from .netanim import ACT_HI, ACT_LO, GRAY_HI, GRAY_LO, OFF, hash01
from .text import TIMES, num

SIZES = (784, 32, 16, 10)                   # the network
EPOCHS = 6
FIRST = 8                                   # test image shown in detail: a '5' (p = 0.89)
MORE = (0, 2, 3, 20, 4, 38, 11)             # then 7 1 0 9 4 2 6 -- the 2 is read as a 3

IMG_AT, IMG_PX = (0.09, 0.60), 150.0        # the input image: centre (fraction), size (px)
COL_X = (0.36, 0.56, 0.755)                 # x of each layer's column (fraction)
SPAN = ((0.27, 0.98), (0.30, 0.95), (0.29, 0.96))   # each column's top and bottom
RADII = (4.8, 7.6, 12.5)                    # node radius per layer, px
BAR_DX, BAR_LEN = 24.0, 112.0               # softmax bars: gap after the node, full length (px)
TOP_IN = 6                                  # strongest pixel lines per layer-1 neuron
DIM = C.rgb("#c4c8ca")                      # an unlikely output
FOCUS = (0.12, 0.36, 0.66)                  # ring of the neuron worked out in numbers
RIGHT, WRONG = (0.20, 0.60, 0.30), (0.80, 0.25, 0.20)

# The detailed first digit (seconds)...
TIMELINE = dict(intro=(0.2, 1.2), image=(1.0, 2.4), wires=(2.4, 4.0),
                f1=(4.2, 6.6), l1=(6.6, 9.0), f2=(9.2, 11.4), l2=(11.4, 13.6),
                l3=(13.8, 16.0), soft=(16.0, 18.0), verdict=(18.0, 19.4))
MORE_T0, MORE_STEP = 20.0, 2.7              # ...then one quick digit every 2.7 s,
QUICK = dict(l1=(0.35, 0.95), l2=(0.95, 1.45), l3=(1.45, 1.85),   # seconds into its slot
             soft=(1.85, 2.15), verdict=(2.1, 2.4))
FEATURE_SHARE = 0.35                        # first part of a detailed layer: one neuron only


@cache
def model():
    """``(params, test accuracy, x_test, y_test)`` for the trained 784-32-16-10 network."""
    from . import mnist
    params, acc = mnist.train(sizes=SIZES, epochs=EPOCHS)
    _xtr, _ytr, xte, yte = mnist.load()
    return params, acc, xte, yte


def run(params, x):
    """One image through the network: ``(zs, acts, p)``.

    ``zs[k]`` is layer k's pre-activation, ``acts[k]`` its input (``acts[0]`` is
    the image, ``acts[-1]`` the logits), ``p`` the softmax probabilities.
    """
    zs, acts = [], [x]
    a = x
    for n, (W, b) in enumerate(params):
        z = a @ W + b
        zs.append(z)
        a = np.maximum(z, 0) if n < len(params) - 1 else z
        acts.append(a)
    e = np.exp(zs[-1] - zs[-1].max())
    return zs, acts, e / e.sum()


@dataclass
class Frame:
    """What one frame shows (see :meth:`MLPClassifier._frame`)."""

    back: float                 # 1, fading to 0 in the loop reset
    intro: float                # the diagram fading in
    k: int                      # which digit (index into runs)
    image: np.ndarray           # the pixels on show (cross-fading between digits)
    image_on: float
    lay: list                   # progress 0..1 into each of the three layers
    soft: float                 # softmax bars growing
    verdict: float              # the answer appearing
    wires: float                # faint wiring before the first digit's data arrives

    @property
    def detail(self):
        """True for the first digit, which is worked through slowly."""
        return self.k == 0


class MLPClassifier(Diagram):
    """One digit through the network in detail, then seven more quickly."""

    PHASES = ("image", "layer 1", "layer 2", "output", "more digits")

    def __init__(self):
        super().__init__("A dense network classifies digits", "")
        self.params, self.acc, xte, yte = model()
        self.runs = []
        for i in (FIRST, *MORE):
            zs, acts, p = run(self.params, xte[i])
            self.runs.append(dict(x=xte[i], label=int(yte[i]), zs=zs, acts=acts, p=p,
                                  pred=int(p.argmax())))
        self.nodes = []
        for k in range(3):
            n = SIZES[k + 1]
            y0, y1 = SPAN[k]
            self.nodes.append([self.P(COL_X[k], y0 + (y1 - y0) * i / (n - 1)) for i in range(n)])
        self.img = self.P(*IMG_AT)
        self.pix = [self._pixel(i) for i in range(SIZES[0])]
        # the neuron worked out in numbers, per layer: the strongest lit one
        r0 = self.runs[0]
        self.feat = [int(np.argmax(r0["acts"][1])), int(np.argmax(r0["acts"][2])), r0["pred"]]
        self.hold, self.reset = 3.0, 1.2
        self.quick = [MORE_T0 + MORE_STEP * n for n in range(len(MORE))]
        self.Tm = dict(TIMELINE, more=(MORE_T0, MORE_T0 + MORE_STEP * len(MORE)))
        self.finish(self.Tm["more"][1] + 0.4)

    def _pixel(self, i):
        r, c = divmod(i, 28)
        s = IMG_PX / 28
        return self.shift(self.img, (c + 0.5) * s - IMG_PX / 2, (r + 0.5) * s - IMG_PX / 2)


    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        for k, name in ((4, "more"), (3, "l3"), (2, "f2"), (1, "f1")):
            if t >= self.Tm[name][0]:
                return k
        return 0

    # ------------------------------------------------------------------ state
    def _frame(self, t, reset):
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        intro = smoothstep(0.0, 1.0, self.progress("intro", t)) * back
        if t < self.Tm["more"][0]:
            k, image_on = 0, smoothstep(0.0, 1.0, self.progress("image", t))
            lay = [max(self.progress("f1", t) * FEATURE_SHARE, self.progress("l1", t)),
                   max(self.progress("f2", t) * FEATURE_SHARE, self.progress("l2", t)),
                   self.progress("l3", t)]
            soft, verdict = self.progress("soft", t), self.progress("verdict", t)
        else:
            n = max(i for i, t0 in enumerate(self.quick) if t >= t0 or i == 0)
            u = t - self.quick[n]
            k, image_on = n + 1, 1.0
            lay = [remap(u, *QUICK["l1"]), remap(u, *QUICK["l2"]), remap(u, *QUICK["l3"])]
            soft, verdict = remap(u, *QUICK["soft"]), remap(u, *QUICK["verdict"])
        image = self.runs[k]["x"]
        if k > 0:                               # the previous digit fades out under the new one
            swap = smoothstep(0.0, 0.3, t - self.quick[k - 1])
            if swap < 1:
                prev = self.runs[k - 1]["x"]
                image = prev + (image - prev) * swap
        wires = smoothstep(0.0, 1.0, self.progress("wires", t)) if k == 0 else 1.0
        return Frame(back, intro, k, image, image_on, lay, soft, verdict, wires)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        f = self._frame(t, reset)
        R = self.runs[f.k]
        self._image(sc, f)
        for L in range(3):
            if f.lay[L] > 0:
                self._edges(sc, L, R, f)
                self._flow(sc, L, R, f.lay[L], f.back, f.detail)
        if f.wires > 0 and f.lay[0] <= 0:
            self._wiring(sc, f)
        self._nodes(sc, t, R, f)
        if f.soft > 0:
            self._bars(sc, R, f)
        self._captions(sc, R, f)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, f.k, reset)

    def _image(self, sc, f):
        tex = GRAY_LO[None] + (GRAY_HI - GRAY_LO)[None] * np.asarray(f.image)[:, None]
        draw_image(self, sc, self.img, tex.reshape(28, 28, 3), IMG_PX, IMG_PX,
                   f.image_on * f.back, grid=0.25)
        self.rect(sc, self.img, IMG_PX + 8, IMG_PX + 8, (0.97, 0.97, 0.96), f.intro * 0.9,
                  border=(0.62, 0.63, 0.65), border_w=1.0, radius=5, lift=0.0)

    def _edges(self, sc, L, R, f):
        """The lines into layer ``L``, each as strong as |w x incoming value|,
        sweeping down the column as the layer computes."""
        c = R["acts"][L][:, None] * self.params[L][0]
        u = f.lay[L]
        cmax = float(np.abs(c).max()) or 1.0
        n_out = SIZES[L + 1]
        focus = f.detail and L < 2
        for j in range(n_out):
            sweep = smoothstep(0.0, 0.2, remap(u, 0.6 * j / n_out, 0.6 * j / n_out + 0.4))
            if focus and u < FEATURE_SHARE:          # only the worked neuron at first
                sweep = smoothstep(0.0, 0.3, u / FEATURE_SHARE) if j == self.feat[L] else 0.0
            if sweep <= 0:
                continue
            col = c[:, j]
            idx = np.argsort(-np.abs(col))[:TOP_IN] if L == 0 else \
                np.nonzero(np.abs(col) > 0.06 * cmax)[0]
            for i in idx:
                v = col[i] / cmax
                if abs(v) < 0.02:
                    continue
                src = self.pix[i] if L == 0 else self.nodes[L - 1][i]
                ea, eb = self._ends(src, self.nodes[L][j], 0.0 if L == 0 else RADII[L - 1],
                                    RADII[L])
                op = (0.10 + 0.38 * abs(v) ** 0.6) * sweep * f.back
                if focus and j == self.feat[L]:
                    op = min(0.48, op * 1.6)
                sc.line(ea, eb, width=0.7 + 1.6 * abs(v) ** 0.6,
                        color=C.mix(EDGE, C.POSITIVE if v > 0 else C.NEGATIVE, 0.75),
                        opacity=op, bias=0.0)

    def _wiring(self, sc, f):
        """Faint lines from the image to every layer-1 neuron before the data arrives."""
        for j in range(SIZES[1]):
            for corner in ((-1, -1), (1, 1), (1, -1), (-1, 1)):
                p = self.shift(self.img, corner[0] * IMG_PX / 2, corner[1] * IMG_PX / 2)
                ea, eb = self._ends(p, self.nodes[0][j], 0.0, RADII[0])
                sc.line(ea, eb, width=0.7, color=EDGE, opacity=0.22 * f.wires * f.back, bias=0.0)

    def _nodes(self, sc, t, R, f):
        """Every neuron, lit by its value (dark = switched off by ReLU); the
        outputs by probability once softmax has run, with the answer ringed."""
        for L in range(3):
            vals = R["acts"][L + 1]
            vmax = float(np.abs(vals).max()) or 1.0
            n = SIZES[L + 1]
            for j in range(n):
                on = smoothstep(0.0, 0.15, remap(f.lay[L], 0.6 * j / n + 0.3, 0.6 * j / n + 0.45))
                worked = f.detail and L < 2 and j == self.feat[L]
                if worked:
                    on = max(on, smoothstep(0.75, 1.0, f.lay[L] / FEATURE_SHARE))
                v = vals[j]
                if L == 2 and f.soft > 0:           # by probability: pale = unlikely
                    lit = C.mix(DIM, ACT_HI, R["p"][j] ** 0.5)
                elif v > 0:
                    lit = ACT_LO + (ACT_HI - ACT_LO) * (v / vmax) ** 0.6
                else:                               # ReLU off; a negative logit stays readable
                    lit = OFF if L < 2 else DIM
                ring, ring_op = None, 0.0
                if worked:
                    ring, ring_op = FOCUS, smoothstep(0.0, 0.1, f.lay[L]) * \
                        (1 - smoothstep(0.0, 0.5, t - self.Tm["l%d" % (L + 1)][1]))
                if L == 2 and f.verdict > 0 and j == R["pred"]:
                    ring = RIGHT if R["pred"] == R["label"] else WRONG
                    ring_op = smoothstep(0.0, 0.4, f.verdict)
                self.disk(sc, self.nodes[L][j], RADII[L], C.mix(NODE, lit, on), f.intro,
                          ring=ring, ring_w=2.4, ring_op=ring_op * f.back)
        for j in range(SIZES[-1]):
            self.label(sc, self.nodes[2][j], str(j), layout.MIN_TEXT, opacity=f.intro,
                       ink=layout.STRONG_INK)

    def _bars(self, sc, R, f):
        """Softmax probabilities as bars beside the outputs, with their values."""
        p = R["p"]
        for j in range(SIZES[-1]):
            a = self.shift(self.nodes[2][j], RADII[2] + BAR_DX, 0)
            b = self.shift(a, BAR_LEN * p[j] * ease_in_out(f.soft), 0)
            if p[j] * f.soft > 0.004:
                sc.line(a, b, width=9.0, color=DATA if j == R["pred"] else
                        C.mix(DATA, EDGE, 0.55), opacity=0.95 * f.back, bias=0.1)
            if p[j] >= 0.005:
                self.label(sc, self.shift(a, BAR_LEN + 8, 0), num(p[j]), layout.MIN_TEXT,
                           anchor="lm", opacity=smoothstep(0.4, 1.0, f.soft) * f.back,
                           kind="mono",
                           ink=layout.STRONG_INK if j == R["pred"] else layout.CAPTION_INK)

    def _ends(self, a, b, ra, rb):
        """Like :meth:`Diagram.ends`, but safe for zero-length segments."""
        sa, sb = self.px(a), self.px(b)
        d = max(np.linalg.norm(sb - sa), 1e-6)
        return a + (b - a) * (ra / d), b - (b - a) * (rb / d)

    def _flow(self, sc, L, R, u, back, detail):
        """Dots carrying the strongest products into layer ``L`` (orange adds,
        blue subtracts): for the worked neuron first, then for the whole layer."""
        if not 0 < u < 1:
            return
        c = R["acts"][L][:, None] * self.params[L][0]
        if detail and L < 2 and u < FEATURE_SHARE:
            j = self.feat[L]
            pairs = [(i, j) for i in np.argsort(-np.abs(c[:, j]))[:14]]
            uu = u / FEATURE_SHARE
        else:
            flat = np.argsort(-np.abs(c).ravel())[:48]
            pairs = [divmod(int(m), c.shape[1]) for m in flat]
            uu = remap(u, FEATURE_SHARE, 1.0) if detail and L < 2 else u
        for n, (i, j) in enumerate(pairs):
            src = self.pix[i] if L == 0 else self.nodes[L - 1][i]
            s0 = 0.45 * hash01(n, L)
            self.dot(sc, src, self.nodes[L][j], remap(uu, s0, s0 + 0.55),
                     C.POSITIVE if c[i, j] > 0 else C.NEGATIVE, 2.8, back, key=("f", L, n))

    def _captions(self, sc, R, f):
        names = ("%d neurons" % SIZES[1], "%d neurons" % SIZES[2], "%d outputs" % SIZES[3])
        for L, text in enumerate(names):
            self.label(sc, self.nodes[L][0], text, layout.MIN_TEXT, anchor="mb",
                       dy=-(RADII[L] + 10), opacity=f.intro)
        self.label(sc, self.img, "%d pixel values" % SIZES[0], layout.MIN_TEXT, anchor="mt",
                   dy=IMG_PX / 2 + 10, opacity=f.intro)
        top = self.shift(self.nodes[2][0], RADII[2] + BAR_DX + BAR_LEN / 2 + 30, 0)
        self.label(sc, top, "softmax p", layout.MIN_TEXT, anchor="mb", dy=-(RADII[2] + 10),
                   opacity=f.intro)
        if f.verdict > 0:
            text = "%d" % R["pred"] + ("  correct" if R["pred"] == R["label"] else
                                       "  (it is a %d)" % R["label"])
            self.label(sc, self.img, "reads: " + text, layout.CAPTION, anchor="mb",
                       dy=-(IMG_PX / 2 + 12), opacity=smoothstep(0.0, 0.5, f.verdict) * f.back,
                       ink=layout.STRONG_INK)

    # ------------------------------------------------------------------ words
    def _words(self, t, k, reset):
        """The subtitle at ``t``, with the real numbers for the first digit."""
        T, r0 = self.Tm, self.runs[0]
        n_in, n1, n2, n_out = SIZES
        if reset > 0 or t < T["wires"][0]:
            return ("a trained network (%s, %.1f%% on 10,000 test digits) reads a handwritten "
                    "digit" % (" - ".join(map(str, SIZES)), 100 * self.acc))
        if t < T["f1"][0]:
            return ("the %d pixel values go in; every pixel connects to every one of the %d "
                    "neurons in layer 1: %d %s %d weights" % (n_in, n1, n_in, TIMES, n1))
        if t < T["f2"][0]:
            if t < T["l1"][0]:
                z = r0["zs"][0][self.feat[0]]
                return ("one neuron: z = Σ wᵢxᵢ + b over all %d pixels = %s; ReLU keeps it: "
                        "max(0, z) = %s. Lines show the strongest w %s x"
                        % (n_in, num(z), num(max(z, 0)), TIMES))
            n_on = int((r0["acts"][1] > 0).sum())
            return ("all %d at once: %d have z > 0 and light up; ReLU switches the other %d off "
                    "(dark)" % (n1, n_on, n1 - n_on))
        if t < T["l3"][0]:
            if t < T["l2"][0]:
                z = r0["zs"][1][self.feat[1]]
                return ("layer 2: each neuron sums the %d values below, weighted: "
                        "z = %s, ReLU %s. Orange lines add, blue subtract"
                        % (n1, num(z), num(max(z, 0))))
            return "layer 2: %d of %d neurons light up" % (int((r0["acts"][2] > 0).sum()), n2)
        if t < T["soft"][0]:
            zz = r0["zs"][2]
            return ("output: %d scores (logits), one per digit; the largest is %s for '%d'"
                    % (n_out, num(zz.max()), int(zz.argmax())))
        if t < T["more"][0]:
            o = np.argsort(-r0["p"])
            return ("softmax: p = exp(z) / Σ exp(z), so the %d add up to 1. '%d': %s, '%d': %s"
                    % (n_out, o[0], num(r0["p"][o[0]]), o[1], num(r0["p"][o[1]])))
        return "more test digits, the same weights every time" + self._tally(t, k)

    def _tally(self, t, k):
        """'; 5 of 5 right so far', counting the current digit once it is answered."""
        done = self.runs[1:k + 1]
        if t - self.quick[k - 1] < QUICK["verdict"][0]:
            done = done[:-1]
        if not done:
            return ""
        right = sum(r["pred"] == r["label"] for r in done)
        return "; %d of %d right so far" % (right, len(done))

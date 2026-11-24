"""Animation 183 of the "next" catalogue (see :mod:`voxanim.mech`): counting the numbers a
dense network and a convolutional network learn, for students who need it slow.

:class:`CountingWeights`: one weight (one input, one output, one line between them) and a
bias; the DNN of "An MLP, step by step" counted one ``nn.Linear`` at a time, each drawn
as an input column, a grid of weights, an output row and a row of biases, with a running
total; the CNN of "A CNN training step": its four 3 × 3 filters (40 numbers, for a picture
of any size), then its dense layer, where nearly all of its numbers sit; and the two totals
side by side, with the line of PyTorch that counts them.

The same pattern every time, one thing counted at a time; the PyTorch line that makes each
layer is shown beside it (text only). One colour means one thing throughout: blue for
weights, amber for biases, green for the running total. Every number comes from
:mod:`voxanim.study_count_nets`, which checks each count against the models' own
parameter arrays. Every phase says, bottom right, what kind of evidence it shows.
"""

from __future__ import annotations

import numpy as np

from . import layout, mnist
from .easing import smoothstep
from .mech import AMBER, BLUE, GREEN, Mechanism, dark, soft
from .mech_phys_apps import GREY, commas

WEIGHT, BIAS, TOTAL = BLUE, AMBER, GREEN        # one colour, one meaning, all through
UNIT = GREY                                     # inputs and outputs: values, not learned
HUGE, BIG, HEAD, CODE, SMALL = 0.060, 0.048, 0.036, 0.031, 0.031
ONE = "exact arithmetic: one weight, one bias"
SMALL_LAYER = "exact arithmetic: a layer of 3 inputs and 2 outputs"
DNN = "exact arithmetic: the MLP of “An MLP, step by step”"
CNN = "exact arithmetic: the CNN of “A CNN training step”"
BOTH = "exact arithmetic: both models' parameter arrays counted"
RULE = "each output: one weight per input, plus one bias"
CODE_X, COUNT_X, CODE_Y, CODE_STEP = 0.03, 0.47, 0.36, 0.064
GRID_X, GRID_Y, CELL = 0.70, 0.45, 9.0          # the layer picture: grid's top left; cell px
SHOWN_DIGIT = 0                                 # the test picture the filters slide over


def is_evidence(text):
    """True for the tag every phase carries."""
    return text.startswith("exact arithmetic: ")


def cells(n):
    """How many blocks stand for ``n`` inputs or outputs in a layer picture: more for more,
    on a log scale (the labels give the exact count)."""
    return int(np.clip(round(1.6 * np.log2(max(n, 2))), 3, 15))


def product(*v):
    return " × ".join(commas(x) for x in v)


class CountingWeights(Mechanism):
    """One weight and one bias; the DNN layer by layer with its running total; the CNN's
    filters, the same for any picture; the CNN's dense layer; both totals and the line
    of code that counts them."""

    TITLE = "Counting the weights: a DNN and a CNN"
    PHASES = (("a weight", 9.0), ("the DNN", 18.0), ("filters", 12.0), ("CNN dense", 10.0),
              ("both", 11.0))
    # phase 2: (start, end) of each nn.Linear, then the zeros and the total
    DNN_SLOTS = ((0.0, 0.34), (0.34, 0.58), (0.58, 0.80))
    DNN_END = 0.80
    CNN_ZERO = 0.22                     # phase 4: the zeros, then the dense layer

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        self.one, self.toy = m["one"], m["toy"]
        self.dnn, self.cnn = m["dnn"]["layers"], m["cnn"]["layers"]
        self.dnn_dense = [i for i, x in enumerate(self.dnn) if x["kind"] == "Linear"]
        self.conv = self.cnn[0]
        self.cnn_dense = self.cnn[-1]
        f = np.abs(np.asarray(s.panels[2].data["values"], float))
        self.filters = 0.3 + 0.7 * f / f.max()          # learned numbers, all visibly blue
        self.sizes = m["cnn"]["any_size"]
        img = mnist.load()[2][SHOWN_DIGIT].reshape(28, 28)
        self.pictures = [np.kron(img, np.ones((k["side"] // 28,) * 2)) for k in self.sizes]

    # ------------------------------------------------------------------ words
    def subtitle_for(self, k):
        m = self.meta
        d, c = self.dnn, self.cnn
        L = [d[i] for i in self.dnn_dense]
        return [
            "a weight is one number the model learns, for one input and one output; each "
            "output also learns one bias",
            "the DNN: %s × %s + %s = %s, then %s, then %s. In all, %s numbers to learn"
            % (commas(L[0]["inputs"]), commas(L[0]["outputs"]), commas(L[0]["biases"]),
               commas(L[0]["total"]), commas(L[1]["total"]), commas(L[2]["total"]),
               commas(m["dnn"]["total"])),
            "a 3 × 3 filter is 9 numbers. %d filters and %d biases: %s numbers, for a picture "
            "of any size" % (self.conv["out_channels"], self.conv["biases"],
                             commas(self.conv["total"])),
            "the CNN's dense layer: %s × %s + %s = %s. With the filters, the CNN learns %s"
            % (commas(c[-1]["inputs"]), commas(c[-1]["outputs"]), commas(c[-1]["biases"]),
               commas(c[-1]["total"]), commas(m["cnn"]["total"])),
            "the DNN learns %s numbers, the CNN %s. In both, one dense layer holds most of them"
            % (commas(m["dnn"]["total"]), commas(m["cnn"]["total"])),
        ][k]

    # ------------------------------------------------------------------ text helpers
    def width(self, s, size, kind="regular"):
        """The layout width of ``s``."""
        return self.frac(layout.text_width(s, size, kind) * self.cam.width)[0]

    def parts(self, sc, fx, fy, segs, op, size=BIG, centre=False):
        """One line of ``(text, ink, kind)`` pieces side by side from ``fx`` (or centred),
        a space and a little more apart, so that no two pieces touch."""
        segs = [(s.strip(), ink, kind) for s, ink, kind in segs]
        ws = [self.width(s, size, kind) for s, _ink, kind in segs]
        gap = self.frac(layout.text_width(" ", size) * self.cam.width + 6.0)[0]
        x = fx - (sum(ws) + gap * (len(ws) - 1)) / 2 if centre else fx
        for (s, ink, kind), w in zip(segs, ws):
            self.label(sc, self.P(x, fy), s, size, anchor="lm", opacity=op, ink=ink, kind=kind)
            x += w + gap

    def total_sum(self, sc, fx, fy, layer, op, size=BIG):
        """``in × out + biases = total``: weights in blue, biases in amber."""
        if layer["kind"] == "Conv2d":
            w = product(layer["out_channels"], layer["in_channels"], layer["kernel"],
                        layer["kernel"])
        else:
            w = product(layer["inputs"], layer["outputs"])
        self.parts(sc, fx, fy, [(w, dark(WEIGHT, 0.25), "regular"),
                                ("+", None, "regular"),
                                (commas(layer["biases"]), dark(BIAS, 0.3), "regular"),
                                ("=", layout.STRONG_INK, "regular"),
                                (commas(layer["total"]), layout.STRONG_INK, "regular")],
                   op, size)

    def code_line(self, sc, fx, fy, s, op, on=1.0, size=CODE):
        """A line of PyTorch in the monospace font; ``on`` lights its box."""
        w = layout.text_width(s, size, "mono") * self.cam.width
        dx = self.frac(w)[0]
        if on > 0.01:
            self.box(sc, fx + dx / 2, fy, w + 16, 30, soft(WEIGHT, 0.8), op * on,
                     border=WEIGHT)
        self.label(sc, self.P(fx, fy), s, size, anchor="lm", opacity=op * (0.55 + 0.45 * on),
                   ink=layout.STRONG_INK if on > 0.5 else None, kind="mono")

    def code_block(self, sc, layers, op, lit, counted):
        """The model's code lines down the left, ``lit[i]`` (0..1) lighting line ``i`` and
        ``counted[i]`` (0..1) showing its count at the right."""
        for i, layer in enumerate(layers):
            fy = CODE_Y + CODE_STEP * i
            self.code_line(sc, CODE_X, fy, layer["code"], op, lit[i])
            n = layer["total"]
            self.text(sc, COUNT_X, fy, commas(n), op * counted[i], anchor="rm",
                      ink=layout.STRONG_INK if n else dark(GREY, 0.1), size=SMALL)

    def running(self, sc, value, op):
        """The running total, bottom left, in green."""
        self.text(sc, CODE_X, 0.80, "total so far", op, anchor="lm", size=SMALL)
        self.text(sc, CODE_X, 0.865, commas(value), op, anchor="lm", size=HUGE,
                  ink=dark(TOTAL, 0.3))

    def tag(self, sc, kind, op):
        self.text(sc, 0.95, 0.925, kind, op, anchor="rm", ink=dark(GREY, 0.25))

    # ------------------------------------------------------------------ pictures
    def strip(self, sc, n, fx, fy, op, ink, vertical, reveal=1.0):
        """``n`` blocks in a row (or a column) centred at ``(fx, fy)``, ``CELL`` px each."""
        shape = (n, 1) if vertical else (1, n)
        self.grid(sc, np.ones(shape), fx, fy, CELL * shape[1], CELL * shape[0], op,
                  reveal=reveal, vmax=1.0, hot=ink, frame=False, lines=0.28)

    def layer_picture(self, sc, layer, op, grow=1.0):
        """A dense layer as blocks: the inputs down the left, the outputs along the top,
        the grid of weights between them and the row of biases under it; ``grow`` fills
        the grid row by row."""
        r, k = cells(layer["inputs"]), cells(layer["outputs"])
        dw, dh = self.frac(CELL * k, CELL * r)
        gx, gy = GRID_X, GRID_Y
        gap_x, gap_y = self.frac(CELL / 2 + 6, CELL / 2 + 6)
        cx, cy = gx + dw / 2, gy + dh / 2
        self.strip(sc, r, gx - gap_x, cy, op, UNIT, True)
        self.strip(sc, k, cx, gy - gap_y, op, UNIT, False)
        self.grid(sc, np.ones((r, k)), cx, cy, CELL * k, CELL * r, op, reveal=grow, vmax=1.0,
                  hot=WEIGHT, frame=False, lines=0.28)
        on = op * smoothstep(0.9, 1.0, grow)
        self.strip(sc, k, cx, gy + dh + gap_y, on, BIAS, False)
        self.text(sc, gx - 2 * gap_x - 0.012, cy, "%s inputs" % commas(layer["inputs"]), op,
                  anchor="rm", size=SMALL)
        self.text(sc, cx, gy - 2 * gap_y - 0.03, "%s outputs" % commas(layer["outputs"]), op,
                  size=SMALL)
        self.text(sc, gx + dw + 0.02, cy, "weights", op * smoothstep(0.3, 0.6, grow),
                  anchor="lm", ink=dark(WEIGHT, 0.25), size=SMALL)
        self.text(sc, gx + dw + 0.02, gy + dh + gap_y, "%s biases" % commas(layer["biases"]),
                  on, anchor="lm", ink=dark(BIAS, 0.3), size=SMALL)
        return cy

    # ------------------------------------------------------------------ 1: one weight
    def node(self, sc, fx, fy, op):
        self.disk(sc, self.P(fx, fy), 15, soft(UNIT, 0.4), op, outline=0.3)

    def bias(self, sc, fx, fy, op):
        self.box(sc, fx, fy, 16, 16, BIAS, op, radius=3)

    def phase_1(self, sc, u, op):
        # the words change over cleanly; the drawings cross-fade, so the frame is never empty
        a, b = op * (1.0 - smoothstep(0.42, 0.47, u)), op * smoothstep(0.49, 0.54, u)
        ga, gb = op * (1.0 - smoothstep(0.42, 0.52, u)), op * smoothstep(0.46, 0.54, u)
        # one input, one output, one line: one weight; one bias
        xi, xo, y = 0.14, 0.40, 0.58
        self.text(sc, 0.05, 0.34, "a weight: one number the model learns", a, anchor="lm",
                  size=HEAD, strong=True)
        self.node(sc, xi, y, ga)
        self.node(sc, xo, y, ga)
        go = smoothstep(0.04, 0.14, u)
        self.arrow(sc, self.ends_at(xi, y, xo, y), self.ends_at(xo, y, xi, y), WEIGHT, ga, go,
                   width=4.0, head=False)
        self.text(sc, (xi + xo) / 2, y - 0.055, "1 weight", a * go, ink=dark(WEIGHT, 0.25),
                  size=HEAD)
        self.text(sc, xi, y + 0.07, "input", a, size=SMALL)
        self.text(sc, xo, y + 0.07, "output", a, size=SMALL)
        late = a * smoothstep(0.16, 0.22, u)
        self.bias(sc, xo + 0.045, y, ga * smoothstep(0.16, 0.22, u))
        self.text(sc, xo + 0.07, y, "1 bias", late, anchor="lm", ink=dark(BIAS, 0.3),
                  size=HEAD)
        code = a * smoothstep(0.24, 0.3, u)
        self.code_line(sc, 0.62, 0.50, self.one["code"], code)
        self.total_sum(sc, 0.62, 0.62, self.one, code)
        # three inputs, two outputs: six weights, two biases
        self.text(sc, 0.05, 0.34, RULE, b, anchor="lm", size=HEAD, strong=True)
        ins = [0.48 + 0.1 * i for i in range(self.toy["inputs"])]
        outs = [0.53 + 0.1 * i for i in range(self.toy["outputs"])]
        n = len(ins) * len(outs)
        drawn = smoothstep(0.55, 0.75, u) * n
        for j, yo in enumerate(outs):
            for i, yi in enumerate(ins):
                g = float(np.clip(drawn - (j * len(ins) + i), 0, 1))
                self.arrow(sc, self.ends_at(xi, yi, xo, yo), self.ends_at(xo, yo, xi, yi),
                           WEIGHT, gb, g, width=3.0, head=False)
        for yi in ins:
            self.node(sc, xi, yi, gb)
        on = b * smoothstep(0.76, 0.8, u)
        for yo in outs:
            self.node(sc, xo, yo, gb)
            self.bias(sc, xo + 0.045, yo, on)
        self.text(sc, xi, 0.77, "%d inputs" % self.toy["inputs"], b, size=SMALL)
        self.text(sc, xo, 0.77, "%d outputs" % self.toy["outputs"], b, size=SMALL)
        self.text(sc, xo + 0.07, 0.66, "%d biases" % self.toy["biases"], on, anchor="lm",
                  ink=dark(BIAS, 0.3), size=SMALL)
        self.text(sc, (xi + xo) / 2, 0.42, "%d weights" % self.toy["weights"],
                  b * smoothstep(0.72, 0.76, u), ink=dark(WEIGHT, 0.25), size=SMALL)
        code = b * smoothstep(0.8, 0.85, u)
        self.code_line(sc, 0.62, 0.50, self.toy["code"], code)
        self.total_sum(sc, 0.62, 0.62, self.toy, code)
        self.tag(sc, ONE, a)
        self.tag(sc, SMALL_LAYER, b)

    def ends_at(self, x0, y0, x1, y1, r=17.0):
        """The layout point ``r`` px from ``(x0, y0)`` towards ``(x1, y1)``."""
        d = np.array(self.span(x1 - x0, y1 - y0))
        d = d / np.hypot(*d) * r
        dx, dy = self.frac(*d)
        return x0 + dx, y0 + dy

    # ------------------------------------------------------------------ 2: the DNN
    def phase_2(self, sc, u, op):
        layers = self.dnn
        lit, counted = [0.0] * len(layers), [0.0] * len(layers)
        value = 0
        for j, (a, b) in enumerate(self.DNN_SLOTS):
            i = self.dnn_dense[j]
            layer = layers[i]
            fade = smoothstep(a, a + 0.03, u) * (1 - smoothstep(b - 0.03, b, u))
            lit[i] = fade
            shown = smoothstep(a + 0.12, a + 0.15, u)
            counted[i] = shown
            if shown > 0.5:
                value = layer["running"]
            on = op * fade
            if on > 0.01:
                self.text(sc, 0.52, 0.315, RULE, on, anchor="lm", size=SMALL, strong=True)
                self.layer_picture(sc, layer, on, smoothstep(a + 0.02, a + 0.11, u))
                self.total_sum(sc, 0.52, 0.83, layer, on * shown)
        # the layers with nothing to learn
        end = smoothstep(self.DNN_END, self.DNN_END + 0.04, u)
        for i, layer in enumerate(layers):
            if layer["total"] == 0:
                lit[i], counted[i] = end, end
            elif u >= self.DNN_END:
                counted[i] = 1.0
        on = op * end
        names = sorted({x["kind"] for x in layers if x["total"] == 0}, reverse=True)
        self.text(sc, 0.72, 0.42, " and ".join(names) + ":", on, size=HEAD)
        self.text(sc, 0.72, 0.52, "0", on, size=HUGE, ink=dark(GREY, 0.1))
        self.text(sc, 0.72, 0.61, "nothing to learn", on, size=HEAD)
        self.code_block(sc, layers, op, lit, counted)
        self.running(sc, value, op)
        self.tag(sc, DNN, op)

    # ------------------------------------------------------------------ 3: the filters
    def phase_3(self, sc, u, op):
        layers, c = self.cnn, self.conv
        lit = [1.0] + [0.0] * (len(layers) - 1)
        counted = [smoothstep(0.3, 0.34, u)] + [0.0] * (len(layers) - 1)
        self.code_block(sc, layers, op, lit, counted)
        self.running(sc, c["running"] if counted[0] > 0.5 else 0, op)
        a = op * (1.0 - smoothstep(0.52, 0.57, u))
        b = op * smoothstep(0.57, 0.62, u)
        # four filters of 3 x 3 weights, one bias each
        self.text(sc, 0.52, 0.315, "each filter: 3 × 3 weights, plus one bias", a,
                  anchor="lm", size=SMALL, strong=True)
        xs = [0.60 + 0.09 * f for f in range(len(self.filters))]
        self.text(sc, (xs[0] + xs[-1]) / 2, 0.39, "%d filters" % c["out_channels"], a,
                  size=SMALL)
        for f, fx in enumerate(xs):
            g = a * smoothstep(0.04 + 0.04 * f, 0.08 + 0.04 * f, u)
            self.grid(sc, self.filters[f], fx, 0.47, 42, 42, g, vmax=1.0, hot=WEIGHT,
                      cold=WEIGHT, frame=False, lines=0.28)
            self.bias(sc, fx, 0.555, a * smoothstep(0.22, 0.26, u))
        self.text(sc, (xs[0] + xs[-1]) / 2, 0.61, "%d biases" % c["biases"],
                  a * smoothstep(0.22, 0.26, u), ink=dark(BIAS, 0.3), size=SMALL)
        s = op * smoothstep(0.3, 0.34, u)
        self.total_sum(sc, 0.52, 0.76, c, s)
        self.text(sc, 0.52, 0.825, "1: a grey picture has one channel", a * smoothstep(
            0.36, 0.4, u), anchor="lm", size=SMALL)
        # the same filters on a bigger picture
        self.text(sc, 0.52, 0.315, "any picture size: the same %s numbers" % commas(c["total"]),
                  b, anchor="lm", size=SMALL, strong=True)
        pos = smoothstep(0.62, 0.98, u)
        for k, (size, fx) in enumerate(zip(self.sizes, (0.60, 0.82))):
            px = 120.0 * size["side"] / self.sizes[-1]["side"]
            fy = 0.62 - self.frac(0, px / 2)[1]
            img = self.pictures[k]
            self.digit(sc, img, fx, fy, px, b, frame=soft(UNIT, 0.5))
            # the 3 x 3 window, as big on both: it covers the same 9 pixels' worth
            n = size["map_side"]
            q, r = int((n - 1) * 0.5 * (1 - np.cos(2 * np.pi * pos))), int(0.45 * n)
            cell = px / size["side"]
            dw, dh = self.frac(px, px)
            wx = fx - dw / 2 + (q + 1.5) * dw / size["side"]
            wy = fy - dh / 2 + (r + 1.5) * dh / size["side"]
            self.rect(sc, self.P(wx, wy), 3 * cell + 3, 3 * cell + 3, WEIGHT, b,
                      border=WEIGHT, border_w=2.0, radius=1.5, lift=0.7, fill_op=0.0)
            self.parts(sc, fx, 0.665, [("%d × %d: " % (size["side"], size["side"]), None,
                                       "regular"),
                                      (commas(size["parameters"]), layout.STRONG_INK,
                                       "regular")], b, HEAD, centre=True)
        self.tag(sc, CNN, op)

    # ------------------------------------------------------------------ 4: the dense layer
    def phase_4(self, sc, u, op):
        layers, d = self.cnn, self.cnn_dense
        z = self.CNN_ZERO
        zero = smoothstep(0.0, 0.04, u) * (1 - smoothstep(z - 0.03, z, u))
        dense = smoothstep(z, z + 0.03, u)
        shown = smoothstep(z + 0.14, z + 0.17, u)
        lit = [0.0] + [zero] * (len(layers) - 2) + [dense]
        counted = [1.0] + [smoothstep(0.0, 0.04, u)] * (len(layers) - 2) + [shown]
        self.code_block(sc, layers, op, lit, counted)
        self.running(sc, d["running"] if shown > 0.5 else self.conv["running"], op)
        on = op * zero
        names = [x["kind"] for x in layers if x["total"] == 0]
        self.text(sc, 0.72, 0.42, ", ".join(names) + ":", on, size=HEAD)
        self.text(sc, 0.72, 0.52, "0", on, size=HUGE, ink=dark(GREY, 0.1))
        self.text(sc, 0.72, 0.61, "nothing to learn", on, size=HEAD)
        on = op * dense
        self.text(sc, 0.52, 0.315, RULE, on, anchor="lm", size=SMALL, strong=True)
        cy = self.layer_picture(sc, d, on, smoothstep(z + 0.02, z + 0.12, u))
        self.text(sc, GRID_X - 0.04, cy + 0.045, "(%s)" % product(d["maps"], d["map_side"],
                                                                 d["map_side"]),
                  on, anchor="rm", size=SMALL)
        self.total_sum(sc, 0.52, 0.83, d, on * shown)
        self.tag(sc, CNN, op)

    # ------------------------------------------------------------------ 5: both
    def phase_5(self, sc, u, op):
        m = self.meta
        track, x0 = 560.0, 0.13
        top = max(m["dnn"]["total"], m["cnn"]["total"])
        rows = (("DNN", self.dnn, m["dnn"], 0.36), ("CNN", self.cnn, m["cnn"], 0.55))
        for k, (name, layers, model, fy) in enumerate(rows):
            on = op * smoothstep(0.02 + 0.12 * k, 0.06 + 0.12 * k, u)
            grow = smoothstep(0.04 + 0.12 * k, 0.2 + 0.12 * k, u)
            self.text(sc, 0.05, fy, name, on, anchor="lm", size=HEAD, strong=True)
            x = x0
            big = model["biggest"]
            for i, layer in enumerate(layers):
                w = track * layer["total"] / top * grow
                if w < 0.5:
                    continue
                dx = self.frac(w)[0]
                ink = dark(TOTAL, 0.15) if i == big else soft(TOTAL, 0.45)
                self.rect(sc, self.P(x + dx / 2, fy), w, 30, ink, on, radius=min(3.0, w / 2))
                x += dx
            self.text(sc, x + 0.015, fy, commas(model["total"]),
                      on * smoothstep(0.9, 1.0, grow), anchor="lm", size=BIG,
                      ink=dark(TOTAL, 0.3))
            most = op * smoothstep(0.3 + 0.08 * k, 0.34 + 0.08 * k, u)
            layer = layers[big]
            self.parts(sc, x0, fy + 0.075, [("most: ", None, "regular"),
                                            (layer["code"], layout.STRONG_INK, "mono"),
                                            ("  " + commas(layer["total"]), layout.STRONG_INK,
                                             "regular")], most, SMALL)
        late = op * smoothstep(0.55, 0.6, u)
        self.code_line(sc, 0.05, 0.76, m["code"], late, 1.0)
        self.text(sc, 0.05, 0.835, "gives %s and %s" % (commas(m["dnn"]["total"]),
                                                     commas(m["cnn"]["total"])),
                  op * smoothstep(0.65, 0.7, u), anchor="lm", size=HEAD,
                  ink=dark(TOTAL, 0.3))
        self.tag(sc, BOTH, op)


ANIMATIONS = {"counting-weights-dnn-and-cnn": CountingWeights}

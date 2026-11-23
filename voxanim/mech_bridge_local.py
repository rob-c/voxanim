"""Animation 179 of the "next" catalogue (see :mod:`voxanim.mech`): the bridge from a
network's capacity (Part 3 ends on "The entropy of a network's weights") to convolutions
(Part 4 opens on "Convolution, 3×3 kernel").

:class:`DenseLayerLearnsLocal`: the first layer of the MNIST classifier of animation 174,
before and after training, as 28 × 28 pictures; how alike two of a unit's weights are
against how many pixels apart they sit; the weights a dense layer spends against a 3 × 3
convolution that keeps only near pixels and shares one filter everywhere; and the two
models, trained with the same budget, on held-out digits moved by 0 to 4 pixels.

It draws :mod:`voxanim.study_bridge_local`'s study; the pictures and the CNN's filters are
read from its cached parameters. Every phase says, bottom right, what kind of evidence it
shows: exact arithmetic on the layers' shapes, or a measurement on MNIST.
"""

from __future__ import annotations

import numpy as np

from . import mnist
from . import study_bridge_local as S
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    GREEN,
    OCHRE,
    RED,
    VIOLET,
    dark,
    soft,
)
from .mech_info_nets import Info, fixed
from .mech_phys_apps import GREY, commas, pct

LAYER = "measured on MNIST: the classifier's first layer"
PAIRS = "measured on MNIST: 64 units' weights, every pair"
COUNTED = "exact arithmetic: weights counted from the layers' shapes"
SHIFTED = "measured on MNIST: 10,000 held-out digits, moved 4 ways"
BOTH = "measured on MNIST: both models, held-out digits"
BIG, HEAD, SMALL, TEXT_S = 0.036, 0.033, 0.030, 0.029
BEFORE, AFTER = VIOLET, GREEN            # the weights before and after training
DENSE, CONV = BLUE, RED                  # the dense classifier; the convolution
UNITS = 4                                # hidden units drawn as pictures
SHOWN_DIGIT = 0                          # the held-out digit that is moved
ROW = 14                                 # the picture row the ringed pixel pair sits in
BARS_X = 0.43                            # where the weight-count bars start


def is_evidence(text):
    """True for the tag every phase carries."""
    return text.startswith(("exact arithmetic: ", "measured on MNIST"))


def times(a, b, v):
    """``a × b = v`` with thousands separated."""
    return "%s × %s = %s" % (commas(a), commas(b), commas(v))


class DenseLayerLearnsLocal(Info):
    """The classifier's first-layer weights as pictures, recalled from 174; their
    correlation against pixel distance, with a ringed pair moving apart on one unit; the
    weights of a dense layer, of a 3 × 3 convolution and of the layers between; accuracy
    on held-out digits moved 0-4 pixels; and the step to convolutions."""

    TITLE = "A dense layer learns to be local"
    PHASES = (("recall", 11.0), ("distance", 12.0), ("sharing", 12.0), ("shifted", 13.0),
              ("next", 9.0))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.pics = np.asarray(p0["values"], float)              # shown units, before then after
        self.n_shown = len(self.pics) // 2
        self.d = np.asarray(p1["x"], float)
        self.corr = np.asarray(p1["ys"], float)                  # (before, after)
        self.ladder = np.asarray(p2["values"], float)
        self.shifts = np.asarray(p3["x"], float)
        self.acc = np.asarray(p3["ys"], float)                   # (dense, cnn)
        self.counts = m["counts"]
        xt = mnist.load()[2]
        self.digit_img = xt[SHOWN_DIGIT]
        self.moved = [S.shift(xt[SHOWN_DIGIT:SHOWN_DIGIT + 1], int(k), (0, 1))[0]
                      for k in self.shifts]
        self.filters = np.asarray(S.cnn_params()["W1"], float).T.reshape(-1, 3, 3)
        # where the correlation has faded: from the first distance below a tenth of its
        # one-pixel value, the range of what is left
        after = self.corr[1]
        far = int(np.argmax(after < 0.1 * after[0]))
        self.far = int(self.d[far]), (float(after[far:].min()), float(after[far:].max()))
        lo = float(min(self.acc.min(), 0.9))
        self.acc_lo = np.floor(lo * 10) / 10

    # ------------------------------------------------------------------ words
    def corr_at(self, d, when=1):
        return fixed(self.corr[when][int(d) - 1])

    def subtitle_for(self, k):
        m = self.meta
        h = m["horizontal_neighbours"]
        c, a = self.counts, m["accuracy"]
        return [
            "from “The entropy of a network's weights”: trained, the classifier's "
            "first layer draws strokes and blobs, and neighbouring pixels' weights go from "
            "correlation %s to %s" % (fixed(h["before"]), fixed(h["after"])),
            "how alike two of a unit's weights are depends on how far apart they sit: %s one "
            "pixel apart, %s at three, and close to zero from %d pixels on"
            % (self.corr_at(1), self.corr_at(3), self.far[0]),
            "a dense unit pays for all %d pixels. A 3 × 3 convolution keeps the nine near ones "
            "and shares them at every position: %s weights for 64 maps, not %s"
            % (c["pixels"], commas(c["conv"]), commas(c["dense"])),
            "trained with the same budget, both lose held-out digits as they move, the CNN "
            "fewer: moved %d pixels, the dense classifier names %s right, the CNN %s"
            % (int(self.shifts[-1]), pct(a["dense"][-1]), pct(a["cnn"][-1])),
            "training made the dense layer local by itself; a convolution is that, built in "
            "and shared. Next: Convolution, 3×3 kernel",
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def picture(self, sc, img, fx, fy, size, op, ring=()):
        vmax = float(np.percentile(np.abs(img), 99))
        return self.grid(sc, img, fx, fy, size, size, op, vmax=vmax, ring=ring)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        m = self.meta
        h = m["horizontal_neighbours"]
        self.text(sc, 0.05, 0.315, "from: The entropy of a network's weights", op, anchor="lm",
                  ink=dark(OCHRE, 0.35), size=HEAD)
        size, xs = 70, [0.163 + 0.092 * i for i in range(UNITS)]
        for row, (fy, word) in enumerate(((0.44, "before"), (0.61, "after"))):
            on = op * (1.0 if row == 0 else smoothstep(0.12, 0.26, u))
            self.text(sc, 0.05, fy, word, on, anchor="lm", size=SMALL,
                      ink=dark((BEFORE, AFTER)[row], 0.3))
            for i, fx in enumerate(xs):
                self.picture(sc, self.pics[self.n_shown * row + i], fx, fy, size, on)
        self.text(sc, 0.05, 0.725, "four hidden units' 784 weights as 28 × 28 pictures", op,
                  anchor="lm", size=TEXT_S)
        self.text(sc, 0.05, 0.765, "red: a positive weight, blue: a negative one", op,
                  anchor="lm", size=TEXT_S)
        sizes = "-".join(str(v) for v in m["mlp_sizes"])
        self.reveal(sc, 0.56, 0.40, [
            ("the first layer of the %s classifier" % sizes, None, SMALL),
            ("%d units, each with one weight per pixel" % S.UNITS, None, SMALL),
            ("trained %d epochs on 60,000 digits" % m["epochs"], None, SMALL),
            ("before: noise. After: strokes and blobs", dark(AFTER, 0.35), SMALL)],
            u, op, at=0.2, every=0.07)
        late = op * smoothstep(0.55, 0.65, u)
        self.text(sc, 0.56, 0.62, "weights of neighbouring pixels:", late, anchor="lm",
                  size=SMALL)
        self.text(sc, 0.56, 0.675, "correlation %s → %s" % (fixed(h["before"]),
                                                           fixed(h["after"])), late,
                  anchor="lm", ink=dark(AFTER, 0.35), size=BIG)
        self.text(sc, 0.56, 0.73, "(one pixel to the right, as in 174)", late, anchor="lm",
                  size=TEXT_S)
        self.tag(sc, LAYER, op)

    def _sweep(self, u):
        """The distance shown at ``u``: 1 pixel, then out to the farthest and held."""
        return int(round(1 + (len(self.d) - 1) * smoothstep(0.3, 0.8, u)))

    def phase_2(self, sc, u, op):
        self.text(sc, 0.05, 0.315, "two weights of one unit, d pixels apart: how alike?", op,
                  anchor="lm", strong=True, size=HEAD)
        ch = self.pane(((0.10, 0.52), (0.40, 0.735)), (0.5, self.d[-1] + 0.5), (-0.1, 0.55))
        self.axes(sc, ch, op, "d: pixels apart, along a row or a column", "correlation",
                  [(v, "%d" % v) for v in (1, 3, 6, 9, 12)],
                  [(v, fixed(v, 1)) for v in (0.0, 0.2, 0.4)])
        self.hline(sc, ch, 0.0, AXIS, op, width=1.2)
        go0, go1 = smoothstep(0.03, 0.2, u), smoothstep(0.12, 0.3, u)
        self.path(sc, ch, self.d, self.corr[0], BEFORE, op, go0, width=2.2)
        self.path(sc, ch, self.d, self.corr[1], AFTER, op, go1, width=2.6)
        self.dots(sc, ch, self.d, self.corr[1], AFTER, op * go1, r=3.2)
        self.key(sc, 0.36, 0.42, [("before", BEFORE), ("after", AFTER)], op, size=TEXT_S,
                 step=0.04)
        d = self._sweep(u)
        on = op * smoothstep(0.25, 0.32, u)
        self.disk(sc, self.P(ch.fx(d), ch.fy(self.corr[1][d - 1])), 6.5, AMBER, on, glow=0.4,
                  lift=1.4)
        # one trained unit, a pair of its pixels d apart in one row
        c0 = (28 - int(self.d[-1])) // 2
        self.picture(sc, self.pics[self.n_shown], 0.73, 0.50, 150, op,
                     ring=((ROW, c0), (ROW, c0 + d)) if on > 0.01 else ())
        self.text(sc, 0.73, 0.685, "one trained unit, two of its pixels", op, size=TEXT_S)
        self.text(sc, 0.73, 0.735, "d = %d: correlation %s" % (d, self.corr_at(d)), on,
                  ink=dark(AMBER, 0.4), size=HEAD)
        late = op * smoothstep(0.82, 0.9, u)
        lo, hi = self.far[1]
        self.lines(sc, 0.58, 0.80, [
            "half the one-pixel value by %.1f pixels" % self.meta["half_distance"],
            ("from %d pixels apart: %s to %s" % (self.far[0], fixed(lo), fixed(hi)),
             dark(AFTER, 0.35))], late, size=TEXT_S, step=0.04)
        self.text(sc, 0.10, 0.87, "before training: %s to %s at every distance"
                  % (fixed(self.corr[0].min(), 3), fixed(self.corr[0].max(), 3)),
                  op * go0, anchor="lm", ink=dark(BEFORE, 0.3), size=TEXT_S)
        self.tag(sc, PAIRS, op)

    def _bar(self, sc, fy, n, ink, op, u, label):
        track = 300.0
        self.text(sc, BARS_X, fy, label, op, anchor="lm", size=SMALL, ink=dark(ink, 0.3))
        self.hbar(sc, BARS_X, fy + 0.042, track * np.log10(n) / 8.0, ink, op, u=u, thick=14.0,
                  track=track)

    def phase_3(self, sc, u, op):
        c = self.counts
        # a 3 x 3 window sliding over a held-out digit
        fx, fy, size = 0.19, 0.48, 150
        self.digit(sc, self.digit_img, fx, fy, size, op, frame=soft(CONV, 0.45))
        out = c["out_side"]
        pos = smoothstep(0.05, 0.95, u) * (out * out - 1)
        r, k = divmod(int(pos), out)
        dw, dh = self.frac(size, size)
        cx = fx - dw / 2 + (k + 1.5) * dw / 28
        cy = fy - dh / 2 + (r + 1.5) * dh / 28
        cell = size / 28
        self.rect(sc, self.P(cx, cy), 3 * cell + 2, 3 * cell + 2, AMBER, op, border=AMBER,
                  border_w=2.2, radius=2, lift=0.7, fill_op=0.0)
        self.lines(sc, 0.05, 0.69, [
            "one 3 × 3 filter at every position:",
            "%d × %d outputs a map; for 64 maps," % (out, out),
            ("%s numbers, each from 9 pixels" % commas(c["outputs"]), dark(CONV, 0.3))],
            op, size=TEXT_S, step=0.04)
        self.text(sc, BARS_X, 0.315, "weights to read one 28 × 28 picture (bars: log scale)",
                  op, anchor="lm", strong=True, size=SMALL)
        g = [smoothstep(0.04 + 0.1 * i, 0.2 + 0.1 * i, u) for i in range(2)]
        self._bar(sc, 0.37, c["dense"], DENSE, op, g[0],
                  "dense, 64 units: " + times(c["pixels"], c["units"], c["dense"]))
        self._bar(sc, 0.47, c["conv"], CONV, op * smoothstep(0.1, 0.16, u), g[1],
                  "3 × 3 conv, 64 maps: " + times(9, c["units"], c["conv"]))
        self.text(sc, BARS_X, 0.585, "%.0f times fewer weights, %s times the outputs"
                  % (self.meta["ratio"], commas(c["outputs"] / c["units"])),
                  op * smoothstep(0.3, 0.38, u), anchor="lm", ink=dark(CONV, 0.3), size=HEAD)
        late = op * smoothstep(0.5, 0.58, u)
        g = [smoothstep(0.52 + 0.12 * i, 0.68 + 0.12 * i, u) for i in range(2)]
        self._bar(sc, 0.65, c["dense_same_outputs"], GREY, late, g[0],
                  "dense, as many outputs: " + times(c["pixels"], c["outputs"],
                                                     c["dense_same_outputs"]))
        self._bar(sc, 0.75, c["local"], OCHRE, late * smoothstep(0.6, 0.66, u), g[1],
                  "local, not shared: " + times(9, c["outputs"], c["local"]))
        self.text(sc, BARS_X, 0.855, "local: 9 pixels, not 784. Shared: one filter, not %s"
                  % commas(c["out_side"] ** 2), op * smoothstep(0.75, 0.83, u), anchor="lm",
                  size=TEXT_S)
        self.tag(sc, COUNTED, op)

    def phase_4(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.05, 0.315, "held-out digit %d moved right by" % SHOWN_DIGIT, op,
                  anchor="lm", size=SMALL)
        for i, img in enumerate(self.moved):
            on = op * smoothstep(0.02 + 0.05 * i, 0.08 + 0.05 * i, u)
            x = 0.095 + 0.087 * i
            self.digit(sc, img, x, 0.405, 54, on, frame=soft(AMBER, 0.55))
            self.text(sc, x, 0.49, "%d px" % self.shifts[i], on, size=TEXT_S)
        ch = self.pane(((0.10, 0.48), (0.60, 0.82)), (-0.2, self.shifts[-1] + 0.2),
                       (self.acc_lo, 1.0))
        self.axes(sc, ch, op, "pixels moved (any of four ways)", "named right",
                  [(v, "%d" % v) for v in self.shifts],
                  [(v, pct(v)) for v in np.arange(1.0, self.acc_lo - 1e-9, -0.2)], grid=False)
        go = smoothstep(0.15, 0.6, u)
        for k, ink in enumerate((DENSE, CONV)):
            self.path(sc, ch, self.shifts, self.acc[k], ink, op, go, width=2.6, dot=True)
        self.key(sc, 0.14, 0.725, [("dense", DENSE), ("CNN", CONV)], op, size=TEXT_S, step=0.04)
        sizes = "-".join(str(v) for v in m["mlp_sizes"])
        f1, f2 = m["cnn_filters"]
        self.reveal(sc, 0.55, 0.35, [
            ("same 60,000 digits, %d epochs:" % m["epochs"], None, SMALL),
            ("%s updates of %d, Adam" % (commas(m["updates"]), m["batch"]), None, SMALL),
            ("dense %s: %s parameters" % (sizes, commas(m["mlp_parameters"])),
             dark(DENSE, 0.3), SMALL),
            ("CNN, 3 × 3, %d then %d maps: %s" % (f1, f2, commas(m["cnn_parameters"])),
             dark(CONV, 0.3), SMALL)], u, op, at=0.05, every=0.06)
        a = m["accuracy"]
        late = op * smoothstep(0.6, 0.68, u)
        self.text(sc, 0.55, 0.555, "in place: dense %s, CNN %s" % (pct(a["dense"][0], 1),
                                                                   pct(a["cnn"][0], 1)),
                  late, anchor="lm", size=SMALL)
        self.text(sc, 0.55, 0.61, "moved %d px: dense %s, CNN %s"
                  % (int(self.shifts[-1]), pct(a["dense"][-1]), pct(a["cnn"][-1])), late,
                  anchor="lm", size=HEAD, ink=dark(CONV, 0.3))
        eq = m["equivariance"]
        end = op * smoothstep(0.75, 0.83, u)
        self.lines(sc, 0.55, 0.69, [
            "moved %d px, %d digits' first-layer maps" % (eq["shift"], eq["digits"]),
            "are their maps moved %d px: largest" % eq["shift"],
            ("difference %s (values up to %.1f)" % (number_or_zero(eq["largest_difference"]),
                                                     eq["largest_value"]), dark(CONV, 0.3)),
            "ink still in the frame at %d px: %s" % (int(self.shifts[-1]),
                                                     pct(m["ink_kept"][-1], 1))],
            end, size=TEXT_S, step=0.04)
        self.tag(sc, SHIFTED, op)

    def phase_5(self, sc, u, op):
        m = self.meta
        c, a = self.counts, m["accuracy"]
        self.reveal(sc, 0.05, 0.33, [
            ("trained, a dense layer gives near pixels similar weights:", None, SMALL),
            ("correlation %s one pixel apart, %s to %s from %d"
             % (self.corr_at(1), fixed(self.far[1][0]), fixed(self.far[1][1]), self.far[0]),
             dark(AFTER, 0.35), SMALL),
            ("a 3 × 3 convolution keeps only the near ones, and shares them:", None, SMALL),
            ("%s weights for 64 maps, not %s" % (commas(c["conv"]), commas(c["dense"])),
             dark(CONV, 0.3), SMALL),
            ("moved %d pixels: dense %s, CNN %s" % (int(self.shifts[-1]), pct(a["dense"][-1]),
                                                   pct(a["cnn"][-1])), None, SMALL)],
            u, op, at=0.02, every=0.07, step=0.05)
        on = op * smoothstep(0.3, 0.4, u)
        self.text(sc, 0.05, 0.62, "the CNN's %d first-layer 3 × 3 filters, trained"
                  % len(self.filters), on, anchor="lm", size=TEXT_S)
        vmax = float(np.abs(self.filters).max())
        for i, f in enumerate(self.filters):
            self.grid(sc, f, 0.085 + 0.075 * i, 0.70, 48, 48, on, vmax=vmax)
        self.text(sc, 0.95, 0.70, "each one slid over", on, anchor="rm", size=TEXT_S)
        self.text(sc, 0.95, 0.74, "every position", on, anchor="rm", size=TEXT_S)
        self.text(sc, 0.05, 0.83, "next: Convolution, 3×3 kernel", op * smoothstep(0.55, 0.65, u),
                  anchor="lm", ink=dark(CONV, 0.3), size=BIG)
        self.tag(sc, BOTH, op)


def number_or_zero(v):
    """A difference as a label: ``0`` when exactly zero, else in scientific notation."""
    return "0" if float(v) == 0 else ("%.1e" % float(v)).replace("-", "−")


ANIMATIONS = {"dense-layer-learns-local": DenseLayerLearnsLocal}

"""Animation 182 of the "next" catalogue (see :mod:`voxanim.mech`): how many bits a
weight carries, the bridge from "The entropy of a network's weights" into "Quantization
and precision".

* :class:`BitsInAWeight`: one trained weight and the 32 bits float32 stores it in; the
  classifier's 52,544 weights rounded to each number format, the bits each stores against
  the entropy of the values they take there; held-out accuracy in each, and why plain
  FP4 collapses (nearly every weight rounds to 0); one block of 32 weights with and
  without a shared scale; integer grids of 2 to 8 bits, accuracy against the entropy
  left; then on to Quantization.

It draws :mod:`voxanim.study_bridge_bits`'s ``bits_in_a_weight`` study; the bit pattern of
the weight shown and the histogram are worked out here from the study's cached weights.
Every phase carries, bottom right, what kind of evidence it shows.
"""

from __future__ import annotations

import numpy as np

from . import layout
from . import study_bridge_bits as B
from .easing import smoothstep
from .losschart import sup
from .mech import AXIS, BLUE, DIM, GREEN, OCHRE, PAPER, RED, TEAL, VIOLET, dark, soft
from .mech_phys_apps import GREY, Phys, commas, pct

WEIGHTS = "measured on MNIST: the trained classifier's 52,544 weights"
ROUNDED = "measured on MNIST: the trained weights rounded to each format"
HELD = "measured on MNIST: 10,000 held-out digits"
EXACT = "exact arithmetic on one trained weight's 32 bits"
BIG, HEAD, SMALL, TEXT_S = 0.036, 0.033, 0.030, 0.029
STORED, CARRIED = DIM, BLUE              # bits a format stores; the entropy they carry
RIGHT, LOST, SCALED = GREEN, RED, VIOLET  # accuracy kept; collapsed; with a shared scale
SIGN, EXPO, FRAC = RED, OCHRE, TEAL      # the three fields of a float


def is_evidence(text):
    """True for the tag every phase carries."""
    return text.startswith(("exact arithmetic on ", "measured on MNIST"))


def float_bits(v):
    """``(sign, exponent, fraction)`` of ``v`` stored as float32, as strings of bits."""
    u = int(np.array(v, np.float32).view(np.uint32))
    s = format(u, "032b")
    return s[0], s[1:9], s[9:]


def minus(s):
    return str(s).replace("-", "−")


class BitsInAWeight(Phys):
    """A trained weight's 32 bits; the bits stored and carried in each format; accuracy
    in each and FP4's zeros; a block of 32 with a shared scale; integer grids."""

    TITLE = "How many bits a weight carries"
    PHASES = (("32 bits", 11.0), ("bits carried", 13.0), ("accuracy", 12.0),
              ("a shared scale", 12.0), ("how few", 11.0))
    ROWS = 0.385                         # the first row of the format tables
    STEP = 0.064
    TRACK = 300.0                        # px for 32 bits

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        self.names = list(B.FORMATS)
        self.f = m["formats"]
        self.ints = {int(k): v for k, v in m["ints"].items()}
        self.w = np.asarray(B.runs()["weights"], float)
        self.n = int(self.w.size)
        # the bell in bins of 174's width; the few weights beyond ±0.5 are not drawn
        self.edges = np.arange(-50, 51) * m["histogram_bin"]
        self.counts = np.histogram(self.w, self.edges)[0].astype(float)
        blk = s.panels[3].data
        self.block = np.asarray(blk["values"], float)           # trained, FP4, FP4 scaled
        self.codes = np.asarray(blk["codes"], float)
        self.scale = float(m["block_scale"])
        self.shown = float(self.block[0, 0])                    # the weight drawn in bits
        self.bits = float_bits(self.shown)
        _sign, expo, frac = self.bits
        self.e = int(expo, 2) - 127
        self.mant = 1.0 + int(frac, 2) / 2.0 ** 23
        self.fp4_zero_edge = float(m["fp4_grid"][1]) / 2      # nearer 0 than this: 0
        self.inside = float(np.mean(np.abs(self.w) <= self.fp4_zero_edge))
        self.base = m["float32"]
        self.fewest = int(m["fewest_int_bits"])
        below = m["below_int_bits"]
        self.below = None if below is None else int(below)

    # ------------------------------------------------------------------ words
    def subtitle_for(self, k):
        m, f = self.meta, self.f
        lo = self.ints[self.below] if self.below else None
        return [
            "the weights of 'The entropy of a network's weights' counted in bins of %g had "
            "%.2f bits each. Stored as float32, each takes 32"
            % (m["histogram_bin"], m["histogram_174"]),
            "rounded to each format, the values carry far fewer bits than are stored: FP8 "
            "stores 8 and its %d values carry %.2f" % (f["FP8"]["distinct"],
                                                       f["FP8"]["entropy"]),
            "FP16, BF16 and FP8 keep the accuracy (%s, %s); in plain FP4 %s of the weights "
            "round to 0 and it falls to %s" % (pct(f["BF16"]["accuracy"], 1),
                                               pct(f["FP8"]["accuracy"], 1),
                                               pct(f["FP4"]["zeros"], 1),
                                               pct(f["FP4"]["accuracy"], 1)),
            "one power-of-two scale for every %d weights keeps the small ones: %s zeros, "
            "%.2f bits carried, %s right" % (m["block"], pct(f["FP4 scaled"]["zeros"], 1),
                                             f["FP4 scaled"]["entropy"],
                                             pct(f["FP4 scaled"]["accuracy"], 1)),
            ("on integer grids accuracy holds down to %d bits, %.2f carried; at %d bits "
             "(%.2f carried) it is %s. Next: Quantization"
             % (self.fewest, m["fewest_entropy"], self.below, lo["entropy"],
                pct(lo["accuracy"], 1))) if lo else
            ("on integer grids accuracy holds down to %d bits, %.2f carried. Next: "
             "Quantization" % (self.fewest, m["fewest_entropy"])),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def histogram(self, sc, op, box, grow=1.0, ink=CARRIED, ylabel=True):
        top = float(self.counts.max()) * 1.08
        ch = self.pane(box, (self.edges[0], self.edges[-1]), (0.0, top))
        self.axes(sc, ch, op, "the value of a weight",
                  "weights in each bin of %g" % self.meta["histogram_bin"] if ylabel else "",
                  [(v, minus("%.1f" % v) if v else "0") for v in (-0.4, -0.2, 0.2, 0.4)]
                  + [(0.0, "0")],
                  [(v, commas(v)) for v in (0, 1000, 2000, 3000) if v < top], grid=False)
        self.bars(sc, ch, self.edges, self.counts, ink, op, fill=0.7, grow=grow)
        return ch

    def bar_row(self, sc, y, name, length, op, ink, value, u=1.0, track=None, under=None,
                label_ink=None):
        """A format's row: its name, a grey ``track`` (px), an ``under`` bar (px) in pale
        ``ink`` and a ``length`` bar (px) in ``ink``, and ``value`` after the track."""
        x0 = 0.165
        self.text(sc, 0.05, y, name, op, anchor="lm", size=SMALL, ink=label_ink)
        at = self.P(x0, y)
        if track:
            self.rect(sc, self.shift(at, track / 2, 0), track, 16, PAPER, 0.8 * op,
                      border=AXIS, border_w=1.0, radius=3, lift=0.2)
        if under:
            w = max(under * smoothstep(0.0, 1.0, u), 0.1)
            self.rect(sc, self.shift(at, w / 2, 0), w, 16, soft(ink, 0.6), op, radius=3,
                      lift=0.3)
        w = max(length * smoothstep(0.0, 1.0, u), 0.1)
        self.rect(sc, self.shift(at, w / 2, 0), w, 16, ink, op, radius=3, lift=0.4)
        end = x0 + self.frac(track or w)[0] + 0.012
        self.text(sc, end, y, value, op, anchor="lm", size=TEXT_S)

    def bitpx(self, bits):
        return self.TRACK * bits / 32.0

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.05, 0.315, "from: The entropy of a network's weights", op,
                  anchor="lm", strong=True, size=HEAD)
        self.histogram(sc, op, ((0.09, 0.45), (0.42, 0.70)), smoothstep(0.03, 0.3, u))
        self.text(sc, 0.05, 0.815, "%s weights of a %s classifier"
                  % (commas(self.n), "-".join(str(v) for v in m["sizes"])), op, anchor="lm",
                  size=SMALL)
        self.text(sc, 0.05, 0.857, "174: all its parameters, bins of %g: %.2f bits each"
                  % (m["histogram_bin"], m["histogram_174"]), op * smoothstep(0.1, 0.2, u),
                  anchor="lm", size=SMALL, ink=dark(CARRIED, 0.3))
        # one of them, as float32 stores it
        on = op * smoothstep(0.3, 0.4, u)
        self.text(sc, 0.53, 0.38, "one weight, W0[%d, %d] = %s" % (
            m["block_rows"][0], m["block_unit"], minus("%.8f" % self.shown)), on,
            anchor="lm", size=SMALL, strong=True)
        sign, expo, frac = self.bits
        x, y, free = 0.53, 0.47, 0.0
        for i, (bits, ink, what) in enumerate(((sign, SIGN, "sign"), (expo, EXPO, "exponent"),
                                               (frac, FRAC, "fraction"))):
            a = op * smoothstep(0.38 + 0.06 * i, 0.46 + 0.06 * i, u)
            wd = layout.text_width(bits, TEXT_S) + 0.012
            self.box(sc, x + wd / 2, y, self.span(wd)[0], 28, soft(ink, 0.6), a, bits,
                     border=ink, size=TEXT_S, ink=dark(ink, 0.45))
            word = "%s %d" % (what, len(bits))
            at = max(x, free)
            self.text(sc, at, y + 0.052, word, a, anchor="lm", size=TEXT_S, ink=dark(ink, 0.3))
            free = at + layout.text_width(word, TEXT_S) + 0.015
            x += wd + 0.006
        late = op * smoothstep(0.6, 0.68, u)
        self.text(sc, 0.53, 0.6, "= −1 × 2^(%d − 127) × %.8f" % (self.e + 127, self.mant)
                  if sign == "1" else "= 2^(%d − 127) × %.8f" % (self.e + 127, self.mant),
                  late, anchor="lm", size=SMALL)
        self.text(sc, 0.53, 0.69, "stored: 32 bits a weight,", late, anchor="lm", size=SMALL)
        self.text(sc, 0.53, 0.732, "%s × 32 = %s bits in all" % (commas(self.n),
                                                                 commas(32 * self.n)),
                  late, anchor="lm", size=SMALL)
        self.text(sc, 0.53, 0.82, "how many of the 32 does it carry?",
                  op * smoothstep(0.75, 0.83, u), anchor="lm", size=HEAD,
                  ink=dark(CARRIED, 0.3))
        self.tag(sc, WEIGHTS if u < 0.3 else EXACT, op)

    def phase_2(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.05, 0.315, "each format", op, anchor="lm", strong=True, size=SMALL)
        self.text(sc, 0.165, 0.315, "bits stored (pale) and carried: the entropy of the "
                  "values", op, anchor="lm", size=TEXT_S)
        for i, name in enumerate(self.names):
            f = self.f[name]
            go = smoothstep(0.05 + 0.07 * i, 0.25 + 0.07 * i, u)
            carried = ("≥ %.2f" % f["entropy"]) if name == "FP32" else "%.2f" % f["entropy"]
            stored = ("%g" % f["stored"])
            self.bar_row(sc, self.ROWS + self.STEP * i, name, self.bitpx(f["entropy"]),
                         op * smoothstep(0.0, 0.08, go + 0.01), CARRIED,
                         "%s of %s bits, %s values" % (carried, stored, commas(f["distinct"])),
                         go, track=self.bitpx(32), under=self.bitpx(f["stored"]))
        late = op * smoothstep(0.62, 0.7, u)
        self.text(sc, 0.05, 0.85, "FP32: %s of %s values differ; a count of %s can show at "
                  "most log₂ %s = %.2f bits" % (commas(self.f["FP32"]["distinct"]),
                                               commas(self.n), commas(self.n), commas(self.n),
                                               m["log2_weights"]), late, anchor="lm",
                  size=TEXT_S)
        self.tag(sc, ROUNDED, op)

    def phase_3(self, sc, u, op):
        self.text(sc, 0.05, 0.315, "test digits right", op, anchor="lm", strong=True,
                  size=SMALL)
        for i, name in enumerate(self.names):
            f = self.f[name]
            go = smoothstep(0.03 + 0.05 * i, 0.2 + 0.05 * i, u)
            bad = f["accuracy"] < self.base["accuracy"] - 0.1
            ink = LOST if bad else (SCALED if name.endswith("scaled") else RIGHT)
            self.bar_row(sc, self.ROWS + self.STEP * i, name, 200.0 * f["accuracy"],
                         op * smoothstep(0.0, 0.08, go + 0.01), ink, pct(f["accuracy"], 1),
                         go, track=200.0, label_ink=dark(LOST, 0.3) if bad else None)
        # why FP4 collapses: its grid against the pile of weights
        on = op * smoothstep(0.35, 0.45, u)
        box = ((0.60, 0.93), (0.42, 0.66))
        ch = self.histogram(sc, on, box, ink=GREY, ylabel=False)
        lo = ch.fx(-self.fp4_zero_edge)
        hi = ch.fx(self.fp4_zero_edge)
        top, bot = ch.y0, ch.y1
        band = op * smoothstep(0.45, 0.55, u)
        self.rect(sc, self.P((lo + hi) / 2, (top + bot) / 2), self.span(hi - lo)[0],
                  self.span(0, bot - top)[1], soft(LOST, 0.4), band, radius=2, lift=0.1,
                  fill_op=0.35)
        self.text(sc, 0.60, 0.322, "FP4's values: 0, ±0.5, ±1 … ±6", on, anchor="lm",
                  size=TEXT_S)
        self.text(sc, 0.60, 0.372, "the band rounds to 0: |w| ≤ %g" % self.fp4_zero_edge,
                  band, anchor="lm", size=TEXT_S, ink=dark(LOST, 0.3))
        late = op * smoothstep(0.6, 0.68, u)
        self.text(sc, 0.60, 0.80, "%s of weights fall in it" % pct(self.inside, 1), late,
                  anchor="lm", size=SMALL, ink=dark(LOST, 0.3))
        self.text(sc, 0.60, 0.845, "FP4 keeps %d values: %.2f bits" % (
            self.f["FP4"]["distinct"], self.f["FP4"]["entropy"]), late, anchor="lm",
            size=SMALL, ink=dark(LOST, 0.3))
        self.tag(sc, HELD, op)

    def phase_4(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.05, 0.315, "one block: pixels %d–%d into hidden unit %d"
                  % (m["block_rows"][0], m["block_rows"][1] - 1, m["block_unit"]), op,
                  anchor="lm", strong=True, size=SMALL)
        vmax = float(np.abs(self.block).max())
        rows = (("trained", 0, None), ("FP4", 1, LOST), ("FP4, scale %g" % self.scale, 2,
                                                          SCALED))
        for j, (name, k, ink) in enumerate(rows):
            on = op * smoothstep(0.05 + 0.15 * j, 0.15 + 0.15 * j, u)
            y = 0.39 + 0.09 * j
            self.text(sc, 0.05, y, name, on, anchor="lm", size=TEXT_S,
                      ink=None if ink is None else dark(ink, 0.3))
            self.grid(sc, self.block[k][None, :], 0.585, y, 32 * 17.0, 26, on, vmax=vmax)
            if k:
                zeros = int((self.block[k] == 0).sum())
                self.text(sc, 0.05, y + 0.04, "%d of 32 at 0" % zeros, on, anchor="lm",
                          size=TEXT_S, ink=dark(ink, 0.3))
        late = op * smoothstep(0.5, 0.58, u)
        big = float(np.abs(self.block[0]).max())
        self.text(sc, 0.05, 0.665, "the scale 2%s = %g puts the largest, %.3f, at FP4's top: "
                  "%.3f / %g = %.2f → %g" % (sup(int(np.log2(self.scale))), self.scale, big, big,
                                             self.scale, big / self.scale,
                                             float(np.abs(self.codes).max())),
                  late, anchor="lm", size=TEXT_S)
        f0, f1 = self.f["FP4"], self.f["FP4 scaled"]
        rows = (("weights at 0", pct(f0["zeros"], 1), pct(f1["zeros"], 1)),
                ("bits carried", "%.2f" % f0["entropy"], "%.2f" % f1["entropy"]),
                ("test digits right", pct(f0["accuracy"], 1), pct(f1["accuracy"], 1)))
        self.text(sc, 0.44, 0.725, "FP4", late, size=TEXT_S, ink=dark(LOST, 0.3))
        self.text(sc, 0.58, 0.725, "scaled", late, size=TEXT_S, ink=dark(SCALED, 0.3))
        for i, (name, a, b) in enumerate(rows):
            y = 0.77 + 0.045 * i
            on = op * smoothstep(0.58 + 0.06 * i, 0.64 + 0.06 * i, u)
            self.text(sc, 0.05, y, name, on, anchor="lm", size=TEXT_S)
            self.text(sc, 0.44, y, a, on, size=TEXT_S, ink=dark(LOST, 0.3))
            self.text(sc, 0.58, y, b, on, size=TEXT_S, ink=dark(SCALED, 0.3))
        cost = op * smoothstep(0.8, 0.88, u)
        self.text(sc, 0.66, 0.77, "cost: %d bits a block" % m["scale_bits"], cost,
                  anchor="lm", size=TEXT_S)
        self.text(sc, 0.66, 0.815, "= %.2f bits a weight" % (m["scale_bits"] / m["block"]),
                  cost, anchor="lm", size=TEXT_S)
        self.tag(sc, HELD if u > 0.55 else ROUNDED, op)

    def phase_5(self, sc, u, op):
        m = self.meta
        bits = sorted(self.ints)
        xs = [self.ints[b]["entropy"] for b in bits]
        ys = [self.ints[b]["accuracy"] for b in bits]
        ch = self.pane(((0.11, 0.55), (0.37, 0.69)), (0.0, 8.0), (0.0, 1.0))
        self.axes(sc, ch, op, "bits carried a weight (entropy of the values)",
                  "test digits right",
                  [(v, "%d" % v) for v in (0, 2, 4, 6, 8)],
                  [(v, pct(v)) for v in (0.0, 0.25, 0.5, 0.75, 1.0)])
        self.hline(sc, ch, self.base["accuracy"], GREY, op)
        go = smoothstep(0.05, 0.4, u)
        self.path(sc, ch, xs, ys, TEAL, op, go, width=2.2)
        for i, b in enumerate(bits):
            a = op * smoothstep(0.05 + 0.05 * i, 0.1 + 0.05 * i, u)
            self.disk(sc, self.P(ch.fx(xs[i]), ch.fy(ys[i])), 4.5, TEAL, a, lift=1.2,
                      outline=0.0)
            self.text(sc, ch.fx(xs[i]) + 0.006, ch.fy(ys[i]) + 0.035, "%d" % b, a,
                      anchor="lm", size=TEXT_S, ink=dark(TEAL, 0.3))
        pts = (("FP4", LOST), ("FP4 scaled", SCALED), ("FP8", BLUE))
        on = op * smoothstep(0.4, 0.5, u)
        for name, ink in pts:
            f = self.f[name]
            self.disk(sc, self.P(ch.fx(f["entropy"]), ch.fy(f["accuracy"])), 5.5, ink, on,
                      lift=1.3, outline=0.0)
        self.key(sc, 0.62, 0.36, [("integer grids, 2–8 bits", TEAL), ("FP4", LOST),
                                  ("FP4 scaled", SCALED), ("FP8", BLUE),
                                  ("float32: %s" % pct(self.base["accuracy"], 1), GREY)],
                 op, size=TEXT_S, step=0.042)
        lo = self.ints.get(self.below) if self.below else None
        late = op * smoothstep(0.5, 0.58, u)
        rows = ["holds to %d-bit integers: %.2f bits," % (self.fewest, m["fewest_entropy"]),
                "%s right" % pct(self.ints[self.fewest]["accuracy"], 1)]
        if lo:
            rows += ["%d bits: %.2f carried, %s" % (self.below, lo["entropy"],
                                                    pct(lo["accuracy"], 1))]
        self.lines(sc, 0.62, 0.6, rows, late, size=TEXT_S, step=0.04)
        nxt = op * smoothstep(0.72, 0.8, u)
        self.text(sc, 0.05, 0.85, "next: Quantization", nxt, anchor="lm", strong=True,
                  size=HEAD)
        self.text(sc, 0.34, 0.85, "then: each format's speed on a CPU and on a GPU", nxt,
                  anchor="lm", size=TEXT_S, ink=dark(TEAL, 0.3))
        self.tag(sc, HELD, op)


ANIMATIONS = {"bits-in-a-weight": BitsInAWeight}

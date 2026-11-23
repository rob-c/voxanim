"""Animations 174-175 of the "next" catalogue (see :mod:`voxanim.mech`): entropy in a
trained network.

* :class:`WeightEntropy`: every free parameter of a 784-64-32-10 classifier before and
  after training. As a pile of numbers their entropy goes *up* (they spread out and stay
  bell-shaped); what training wrote is in their arrangement: neighbouring pixels'
  weights agree, and each layer leans on fewer directions.
* :class:`LogitsLossAndNats`: the trained classifier's loss in statistical terms: one
  digit's logits as log-odds and its loss in nats and bits, the 10,000 test losses and
  their heavy tail, the mean as an estimate with an error, the likelihood it is the
  negative log of, and temperature, which moves the loss and not the accuracy.

They draw :mod:`voxanim.study_info`'s ``weight_entropy`` and ``logits_loss_and_nats``
studies; what is drawn beyond the study's metadata (histograms, singular values, the
neighbour scatter, cumulative code lengths) is computed here from its cached parameters
and test scores. Every phase carries, bottom right, what kind of evidence it shows.
"""

from __future__ import annotations

import numpy as np

from . import mnist
from . import study_info as S
from .diagram import draw_image
from .easing import smoothstep
from .losschart import sup
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIM,
    GREEN,
    MAGENTA,
    OCHRE,
    PAPER,
    RED,
    TEAL,
    VIOLET,
    dark,
    soft,
)
from .mech_phys_apps import GREY, Phys, commas, pct

RUNS = "measured on MNIST: three training runs"
SEED0 = "measured on MNIST: the seed-0 run's weights"
HELD = "measured on MNIST: 10,000 held-out digits"
HALVES = "measured on MNIST: two halves of the held-out digits"
EXACT = "exact arithmetic on one digit's measured logits"
BIG, HEAD, SMALL, TEXT_S = 0.036, 0.033, 0.030, 0.029
BEFORE, AFTER = VIOLET, GREEN            # the weights before and after training
NATS, BITS = MAGENTA, TEAL               # a loss in nats and in bits
MODEL, WRONG = GREEN, RED                # the classifier; the digits it names wrongly


def is_evidence(text):
    """True for the tag every phase carries."""
    return text.startswith(("exact arithmetic on ", "measured on MNIST"))


def fixed(v, digits=2):
    """``v`` to ``digits`` places with a typographic minus, never "−0.00"."""
    s = "%.*f" % (digits, float(v))
    if float(s) == 0:
        s = "%.*f" % (digits, 0.0)
    return s.replace("-", "−")


def span_of(values, fmt):
    """A range over seeds: ``4.57`` when they agree when printed, else ``4.56-4.57``."""
    lo, hi = fmt(min(values)), fmt(max(values))
    return lo if lo == hi else "%s–%s" % (lo, hi)


def density(a, b, lim, ink, bins=56):
    """The pairs ``(a, b)`` as a ``bins x bins`` picture: paper where there are none,
    ``ink`` deepening with how many fall in a cell (``a`` across, ``b`` up)."""
    h = np.histogram2d(np.asarray(b), np.asarray(a), bins=bins, range=[[-lim, lim], [-lim, lim]])[0]
    d = np.sqrt(h[::-1] / max(float(h.max()), 1.0))[..., None]
    return (PAPER + (np.asarray(ink) - PAPER) * d).astype(np.float32)


class Info(Phys):
    """What both animations share: a column of text lines appearing one by one."""

    def reveal(self, sc, fx, fy, rows, u, op, at=0.0, every=0.06, step=0.044):
        """Rows of ``(text, ink, size)`` stacked down from ``(fx, fy)``, the ``i``-th
        fading in at ``u = at + i * every``."""
        y = fy
        for i, (s, ink, size) in enumerate(rows):
            a = op * smoothstep(at + i * every, at + i * every + 0.05, u)
            self.text(sc, fx, y, s, a, anchor="lm", ink=ink, size=size)
            y += step
        return y


# ====================================================================== 174: the weights
class WeightEntropy(Info):
    """The network and its 52,650 numbers as a histogram before training; the
    histogram after, its entropy through training in three seeds and how much of the
    rise is spread; first-layer weights as pictures and neighbouring pixels plotted
    against each other; each layer's singular values; and which measure sees what."""

    TITLE = "The entropy of a network's weights"
    PHASES = (("the pile", 11.0), ("trained", 13.0), ("arrangement", 13.0),
              ("directions", 11.0), ("verdict", 10.0))
    HIST = ((0.55, 0.93), (0.38, 0.68))
    LIM = 0.3                            # the window of the neighbour pictures
    UNITS = 4                            # hidden units drawn as pictures

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.steps = np.asarray(p0["x"], float)
        self.hist_runs = np.asarray(p0["ys"], float)            # (seeds, snapshots)
        self.mi_runs = np.asarray(p1["ys"], float)
        self.spec_runs = np.asarray(p2["ys"], float)            # (layers, snapshots), seed 0
        self.pics = np.asarray(p3["values"], float)             # 8 before, 8 after
        P0, P1 = S.params("before"), S.params("after")
        v0, v1 = S.flat(P0), S.flat(P1)
        self.n = int(v0.size)
        self.edges = np.arange(-50, 51) * S.HIST_BIN
        self.counts = [np.histogram(v, self.edges)[0].astype(float) for v in (v0, v1)]
        self.H = [S.histogram_entropy(v) for v in (v0, v1)]                 # seed 0
        self.sd = [float(v.std()) for v in (v0, v1)]
        # the same piles rescaled to one spread: what is left of the rise is their shape
        self.shape_bits = S.histogram_entropy(v1 / v1.std()) - S.histogram_entropy(v0 / v0.std())
        self.pairs = [S.neighbour_pairs(P["W0"]) for P in (P0, P1)]
        self.clouds = [density(a, b, self.LIM, ink) for (a, b), ink in zip(self.pairs,
                                                                         (BEFORE, AFTER))]
        self.corr = [float(np.corrcoef(a, b)[0, 1]) for a, b in self.pairs]
        self.mi = [S.mutual_information(a, b) for a, b in self.pairs]
        self.shares = []                                        # per layer: (before, after)
        for k in range(len(S.SIZES) - 1):
            pair = []
            for P in (P0, P1):
                sv = np.linalg.svd(np.asarray(P["W%d" % k], float), compute_uv=False)
                pair.append(sv ** 2 / (sv ** 2).sum())
            self.shares.append(pair)
        never = S.weight_runs()["never_lit"]
        mask = np.zeros(784, bool)
        mask[never] = True
        self.never = mask.reshape(28, 28)
        self.never_n, self.never_same = m["never_lit"], m["never_lit_unchanged"]
        se = m["spectral_entropy"]
        self.spec_mean = {k: (float(np.mean(se[k]["before"])), float(np.mean(se[k]["after"])))
                          for k in se}

    # ------------------------------------------------------------------ words
    def acc(self, when, digits=0):
        return span_of(self.meta["accuracy"][when], lambda v: pct(v, digits))

    def hist_span(self, when):
        return span_of(self.meta["histogram_entropy"][when], lambda v: "%.2f" % v)

    def subtitle_for(self, k):
        m = self.meta
        grow = self.sd[1] / self.sd[0]
        w0 = self.spec_mean["W0"]
        return [
            "a classifier of handwritten digits has %s free parameters. Before training "
            "they are random draws: counted in bins of %g, the pile of numbers has an "
            "entropy of %.2f bits a parameter"
            % (commas(m["parameters"]), m["histogram_bin"], self.H[0]),
            "after %d epochs (%s right) the entropy of the pile has gone up, to %.2f bits, "
            "in all three seeds. The numbers spread out %.2f times, which alone adds "
            "%.2f bits" % (m["epochs"], self.acc("after", 1), self.H[1], grow,
                           np.log2(grow)),
            "the structure is in where each number sits. Hidden units draw strokes, and "
            "neighbouring pixels' weights agree: correlation %s to %s, mutual information "
            "%.3f to %.2f bits" % (fixed(self.corr[0]), fixed(self.corr[1]), self.mi[0],
                                   self.mi[1]),
            "each layer also leans on fewer directions. The entropy of the first layer's "
            "squared singular values falls from %.2f to %.2f bits: %.0f effective "
            "directions of %d, then %.0f" % (w0[0], w0[1], 2 ** w0[0], S.SIZES[1],
                                             2 ** w0[1]),
            "the entropy of the numbers goes up; the entropy of their arrangement goes "
            "down. As a pile, the trained weights look as random as before: what training "
            "wrote is in the arrangement",
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def histogram(self, sc, op, which=(0,), grow=(1.0, 1.0), box=None):
        top = max(float(self.counts[0].max()), float(self.counts[1].max())) * 1.06
        ch = self.pane(box or self.HIST, (self.edges[0], self.edges[-1]), (0.0, top))
        self.axes(sc, ch, op, "the value of a parameter",
                  "parameters in each bin of %g" % self.meta["histogram_bin"],
                  [(v, fixed(v, 1)) for v in (-0.4, -0.2, 0.0, 0.2, 0.4)],
                  [(v, commas(v)) for v in (0, 1000, 2000, 3000) if v < top], grid=False)
        for w in sorted(which, reverse=True):
            self.bars(sc, ch, self.edges, self.counts[w], (BEFORE, AFTER)[w], op,
                      fill=0.55 if w == 0 and len(which) > 1 else 0.75, grow=grow[w])
        return ch

    def picture(self, sc, img, fx, fy, size, op):
        vmax = float(np.percentile(np.abs(img), 99))
        self.grid(sc, img, fx, fy, size, size, op, vmax=vmax)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.05, 0.315, "a %s classifier of MNIST digits"
                  % "-".join(str(v) for v in m["sizes"]), op, anchor="lm", strong=True,
                  size=HEAD)
        names = ("pixels", "hidden", "hidden", "digits")
        xs = [0.085, 0.19, 0.295, 0.40]
        for i, (n, x) in enumerate(zip(S.SIZES, xs)):
            on = op * smoothstep(0.02 + 0.05 * i, 0.08 + 0.05 * i, u)
            self.box(sc, x, 0.39, 62, 30, soft(BEFORE, 0.62), on, str(n), border=BEFORE,
                     size=SMALL)
            self.text(sc, x, 0.44, names[i], on, size=TEXT_S)
            if i:
                self.arrow(sc, (xs[i - 1] + 0.038, 0.39), (x - 0.038, 0.39), AXIS, on,
                           width=1.8)
        rows = []
        for a, b in zip(S.SIZES[:-1], S.SIZES[1:]):
            rows.append(("%d × %d = %s weights, %d biases" % (a, b, commas(a * b), b), None,
                         SMALL))
        rows.append(("%s free parameters" % commas(m["parameters"]), dark(BEFORE, 0.3), BIG))
        self.reveal(sc, 0.05, 0.505, rows, u, op, at=0.12, every=0.05)
        late = op * smoothstep(0.42, 0.5, u)
        self.text(sc, 0.05, 0.725, "drawn at random, not yet trained:", late, anchor="lm",
                  size=SMALL)
        self.text(sc, 0.05, 0.767, "%s of test digits right, loss %s nats"
                  % (self.acc("before"), span_of(m["test_nats"]["before"],
                                                 lambda v: "%.2f" % v)),
                  late, anchor="lm", size=SMALL)
        self.text(sc, 0.05, 0.809, "(a uniform guess: ln 10 = %.2f)" % np.log(S.CLASSES),
                  late, anchor="lm", size=SMALL)
        self.histogram(sc, op, (0,), (smoothstep(0.2, 0.55, u), 0.0))
        end = op * smoothstep(0.62, 0.7, u)
        self.text(sc, 0.55, 0.795, "count the numbers in bins; p: a bin's share", end,
                  anchor="lm", size=SMALL)
        self.text(sc, 0.55, 0.845, "H = −Σ p log₂ p = %.2f bits a parameter" % self.H[0],
                  op * smoothstep(0.72, 0.8, u), anchor="lm", ink=dark(BEFORE, 0.3),
                  size=HEAD)
        self.tag(sc, SEED0, op)

    def phase_2(self, sc, u, op):
        m = self.meta
        go = smoothstep(0.05, 0.55, u)
        ch = self.pane(((0.09, 0.42), (0.38, 0.60)), (0.0, self.steps[-1]), (4.5, 5.3))
        self.axes(sc, ch, op, "training steps (%d epochs)" % m["epochs"],
                  "histogram entropy, bits a parameter",
                  [(v, commas(v)) for v in (0, 1000, 2000)],
                  [(v, "%.1f" % v) for v in (4.6, 4.8, 5.0, 5.2)])
        for k in range(len(self.hist_runs)):
            self.path(sc, ch, self.steps, self.hist_runs[k], soft(AFTER, 0.35 * k) if k else
                      AFTER, op, go, width=2.4 if k == 0 else 1.6, dot=k == 0)
        self.text(sc, 0.09, 0.715, "three seeds: %s → %s bits"
                  % (self.hist_span("before"), self.hist_span("after")), op, anchor="lm",
                  ink=dark(AFTER, 0.3), size=HEAD)
        grow = self.sd[1] / self.sd[0]
        self.reveal(sc, 0.05, 0.775, [
            ("spread (standard deviation) %.3f → %.3f: × %.2f" % (self.sd[0], self.sd[1],
                                                                 grow), None, SMALL),
            ("a pile %.2f times wider has log₂ %.2f = %.2f bits more" % (grow, grow,
                                                                        np.log2(grow)),
             None, SMALL),
            ("rescaled to one spread, the two differ by %.2f bits" % self.shape_bits,
             dark(AFTER, 0.35), SMALL)], u, op, at=0.55, every=0.08, step=0.043)
        both = smoothstep(0.1, 0.45, u)
        self.histogram(sc, op, (0, 1), (1.0, both))
        self.key(sc, 0.77, 0.43, [("before: %.2f bits" % self.H[0], BEFORE),
                                  ("after: %.2f bits" % self.H[1], AFTER)], op * both,
                 size=TEXT_S)
        self.text(sc, 0.55, 0.795, "after: wider, and still one bell", op * smoothstep(
            0.45, 0.55, u), anchor="lm", size=SMALL)
        self.tag(sc, RUNS, op)

    def phase_3(self, sc, u, op):
        self.text(sc, 0.05, 0.315, "four hidden units' %d weights as 28 × 28 pictures"
                  % S.SIZES[0], op, anchor="lm", strong=True, size=HEAD)
        size, xs = 70, [0.163 + 0.092 * i for i in range(self.UNITS)]
        for row, (fy, word) in enumerate(((0.43, "before"), (0.60, "after"))):
            on = op * (1.0 if row == 0 else smoothstep(0.08, 0.2, u))
            self.text(sc, 0.05, fy, word, on, anchor="lm", size=SMALL,
                      ink=dark((BEFORE, AFTER)[row], 0.3))
            for i, fx in enumerate(xs):
                self.picture(sc, self.pics[8 * row + i], fx, fy, size, on)
        self.text(sc, 0.05, 0.71, "red: a positive weight, blue: a negative one", op,
                  anchor="lm", size=TEXT_S)
        late = op * smoothstep(0.62, 0.72, u)
        tex = np.where(self.never[..., None], np.asarray(soft(OCHRE, 0.15)),
                       np.asarray(PAPER)).astype(np.float32)
        c = self.P(0.085, 0.815)
        self.rect(sc, c, 60, 60, PAPER, late, border=AXIS, border_w=1.2, radius=4, lift=0.1)
        draw_image(self, sc, c, tex, 54, 54, late, grid=0, keep_theme=False)
        self.lines(sc, 0.135, 0.79, [
            "%d pixels are dark in every training digit:" % self.never_n,
            "their weights get no gradient; %d of %d end" % (self.never_same, self.never_n),
            "exactly as drawn: the noise at the edges"], late, size=TEXT_S, step=0.04)
        # the neighbour pictures
        for w, fx in enumerate((0.60, 0.82)):
            on = op * smoothstep(0.22 + 0.14 * w, 0.32 + 0.14 * w, u)
            ink = (BEFORE, AFTER)[w]
            self.text(sc, fx, 0.36, ("before", "after")[w], on, ink=dark(ink, 0.3), size=SMALL)
            c = self.P(fx, 0.53)
            self.rect(sc, c, 152, 152, PAPER, on, border=AXIS, border_w=1.2, radius=4, lift=0.1)
            draw_image(self, sc, c, self.clouds[w], 144, 144, on, grid=0)
            self.text(sc, fx, 0.715, "correlation %s" % fixed(self.corr[w]), on, size=SMALL)
            self.text(sc, fx, 0.755, "information %s bits" % ("%.3f" % self.mi[w] if w == 0
                                                             else "%.2f" % self.mi[w]),
                      on, size=SMALL, ink=dark(ink, 0.3))
        on = op * smoothstep(0.22, 0.32, u)
        self.text(sc, 0.71, 0.81, "across: a pixel's weight; up: the weight", on, size=TEXT_S)
        self.text(sc, 0.71, 0.85, "of the pixel to its right, in the same unit", on,
                  size=TEXT_S)
        self.tag(sc, SEED0, op)

    def phase_4(self, sc, u, op):
        q0, q1 = self.shares[0]
        ranks = np.arange(1, len(q0) + 1, dtype=float)
        ch = self.pane(((0.09, 0.45), (0.40, 0.68)), (1, len(q0)), (0.0, 0.08))
        self.text(sc, 0.05, 0.315, "first layer: the squared singular values as shares", op,
                  anchor="lm", strong=True, size=HEAD)
        self.axes(sc, ch, op, "direction, largest first", "share of the layer's weight",
                  [(v, "%d" % v) for v in (1, 16, 32, 48, 64)],
                  [(v, "%.2f" % v) for v in (0.0, 0.02, 0.04, 0.06, 0.08)])
        self.path(sc, ch, ranks, q0, BEFORE, op, smoothstep(0.05, 0.3, u), width=2.4)
        self.path(sc, ch, ranks, q1, AFTER, op, smoothstep(0.2, 0.45, u), width=2.6)
        self.key(sc, 0.30, 0.45, [("before", BEFORE), ("after", AFTER)], op, size=TEXT_S)
        self.text(sc, 0.05, 0.80, "spectral entropy H = −Σ q log₂ q; 2^H: the", op,
                  anchor="lm", size=SMALL)
        self.text(sc, 0.05, 0.842, "effective number of directions (seed 0 drawn)", op,
                  anchor="lm", size=SMALL)
        self.text(sc, 0.56, 0.37, "each layer, mean of three seeds", op, anchor="lm",
                  strong=True, size=SMALL)
        names = ("first layer", "second layer", "last layer")
        for k in range(3):
            n = min(S.SIZES[k], S.SIZES[k + 1])
            hb, ha = self.spec_mean["W%d" % k]
            y = 0.445 + 0.15 * k
            on = op * smoothstep(0.3 + 0.12 * k, 0.4 + 0.12 * k, u)
            self.text(sc, 0.56, y, "%s, %d × %d: %.2f → %.2f bits"
                      % (names[k], S.SIZES[k], S.SIZES[k + 1], hb, ha), on, anchor="lm",
                      size=TEXT_S)
            track = 230.0
            for j, (h, ink) in enumerate(((hb, BEFORE), (ha, AFTER))):
                self.hbar(sc, 0.57, y + 0.045 + 0.04 * j, track * 2 ** h / n, ink, on,
                          value="%.0f of %d" % (2 ** h, n), u=smoothstep(0.3 + 0.12 * k,
                                                                        0.5 + 0.12 * k, u),
                          thick=13.0, track=track)
        self.tag(sc, RUNS, op)

    def phase_5(self, sc, u, op):
        m = self.meta

        def mean(name, when):
            return float(np.mean(m[name][when]))
        se = self.spec_mean
        rows = [
            ("entropy of the pile (bits)", "%.2f" % mean("histogram_entropy", "before"),
             "%.2f" % mean("histogram_entropy", "after"), "up: the numbers spread", GREY),
            ("neighbours' correlation", fixed(mean("neighbour_correlation", "before")),
             fixed(mean("neighbour_correlation", "after")), "structure", AFTER),
            ("neighbours' information (bits)", "%.3f" % mean("neighbour_information",
                                                              "before"),
             "%.2f" % mean("neighbour_information", "after"), "structure", AFTER),
            ("spectral entropy, first layer", "%.2f" % se["W0"][0], "%.2f" % se["W0"][1],
             "structure", AFTER),
            ("spectral entropy, second layer", "%.2f" % se["W1"][0], "%.2f" % se["W1"][1],
             "structure", AFTER),
            ("spectral entropy, last layer", "%.2f" % se["W2"][0], "%.2f" % se["W2"][1],
             "about the same", GREY),
            ("test digits right", self.acc("before"), self.acc("after", 1), "", None),
        ]
        self.text(sc, 0.05, 0.315, "mean of three seeds", op, anchor="lm", strong=True,
                  size=SMALL)
        self.text(sc, 0.52, 0.315, "before", op, ink=dark(BEFORE, 0.3), size=SMALL)
        self.text(sc, 0.655, 0.315, "after", op, ink=dark(AFTER, 0.3), size=SMALL)
        for i, (name, a, b, verdict, ink) in enumerate(rows):
            y = 0.37 + 0.064 * i
            on = op * smoothstep(0.04 + 0.07 * i, 0.1 + 0.07 * i, u)
            self.text(sc, 0.05, y, name, on, anchor="lm", size=SMALL)
            self.box(sc, 0.52, y, 112, 26, soft(BEFORE, 0.7), on, a, border=BEFORE,
                     size=TEXT_S)
            self.box(sc, 0.655, y, 112, 26, soft(AFTER, 0.7), on, b, border=AFTER,
                     size=TEXT_S)
            if verdict:
                self.box(sc, 0.745, y, 12, 12, ink, on, radius=3)
                self.text(sc, 0.76, y, verdict, on, anchor="lm", ink=dark(ink, 0.3),
                          size=SMALL)
        self.text(sc, 0.05, 0.85, "the numbers: entropy up. Their arrangement: entropy "
                  "down", op * smoothstep(0.6, 0.7, u), anchor="lm", ink=dark(AFTER, 0.35),
                  size=BIG)
        self.tag(sc, RUNS, op)


# ====================================================================== 175: the loss
def tenths(v):
    """``v`` to one place, with commas and a typographic minus: ``−1,128.0``."""
    return f"{float(v):,.1f}".replace("-", "−")


class LogitsLossAndNats(Info):
    """One digit's logits as log-odds and its loss in nats and bits; the 10,000 losses
    on a log scale with their mean, median and tail; the mean as an estimate against
    its baselines; the likelihood and the code length it is; temperature."""

    TITLE = "Logits, loss and nats"
    PHASES = (("one digit", 13.0), ("10,000 losses", 12.0), ("an estimate", 11.0),
              ("likelihood", 11.0), ("temperature", 11.0))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p1, p3 = s.panels[1].data, s.panels[3].data
        z, lp, yt, _prior = S.test_scores()
        self.loss = S.per_digit_loss(lp, yt)
        self.wrong = lp.argmax(1) != yt
        self.z = np.asarray(m["shown_logits"], float)
        self.p = np.asarray(m["shown_probabilities"], float)
        xt = mnist.load()[2]
        self.image = xt[m["shown_digit"]]
        self.worst_image = xt[m["worst_digit"]]
        # the histogram's bins, back from their geometric centres
        x = np.asarray(p1["x"], float)
        r = np.sqrt(x[1] / x[0])
        self.edges = np.append(x / r, x[-1] * r)
        self.counts = np.asarray(p1["ys"], float)[0]
        clipped = np.clip(self.loss, self.edges[0], self.edges[-1])
        self.counts_wrong = np.histogram(clipped[self.wrong], self.edges)[0].astype(float)
        self.counts_right = np.histogram(clipped[~self.wrong], self.edges)[0].astype(float)
        self.wrong_share = float(self.loss[self.wrong].sum() / self.loss.sum())
        self.sd = float(self.loss.std(ddof=1))
        # the code length of the labels, digit by digit
        bits = np.cumsum(self.loss) / np.log(2)
        self.upto = np.arange(0, m["n"] + 1, 25)
        self.cum_bits = np.append(0.0, bits)[self.upto]
        self.T = np.asarray(m["temperatures"], float)
        self.first, self.second = (np.asarray(v, float) for v in p3["ys"])
        h = m["half"]
        self.acc_T = np.array([float((np.argmax(z[h:] / t, 1) == yt[h:]).mean())
                               for t in self.T])
        self.p_T = np.array([np.exp(S.log_softmax(self.z / t))[m["shown_label"]]
                             for t in self.T])
        self.one = int(np.argmin(np.abs(self.T - 1.0)))
        self.best = int(np.argmin(self.first))

    def subtitle_for(self, k):
        m = self.meta
        a, b = m["shown_top2"]
        return [
            "test digit %d is a %d. Its logits, %.2f for a %d and %.2f for a %d, are log-odds "
            "up to a shared constant: their gap, %.2f, is ln(%.4f / %.4f). Its loss: %.3f nats"
            % (m["shown_digit"], m["shown_label"], self.z[a], a, self.z[b], b,
               m["shown_top2_gap"], self.p[a], self.p[b], m["shown_loss_nats"]),
            "over all %s test digits the mean loss is %.3f nats, but the median digit's is "
            "%.3f: the worst 100 carry %s of the total, the worst a %d called a %d"
            % (commas(m["n"]), m["mean_nats"], m["median_nats"],
               pct(m["worst_shares"]["100"]), m["worst_label"], m["worst_named"]),
            "the mean is an estimate from a sample: %.3f ± %.3f nats (%.3f bits). A uniform "
            "guess scores ln 10 = %.3f, the class shares %.3f; the perplexity is %.2f"
            % (m["mean_nats"], m["standard_error"], m["mean_bits"], m["uniform_nats"],
               m["prior_nats"], m["perplexity"]),
            "the mean loss is minus the log of the likelihood, the product of %s "
            "probabilities, over %s. In bits it is a code: %s bits for the labels, %s "
            "without the model" % (commas(m["n"]), commas(m["n"]), commas(m["total_bits"]),
                                   commas(m["raw_bits"])),
            "dividing the logits by a temperature T moves the loss, never the answer. Chosen "
            "on half the digits, the best T is %.2f: the trained model is already "
            "calibrated" % m["best_temperature"],
        ][k]

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        m = self.meta
        a, b = m["shown_top2"]
        self.digit(sc, self.image, 0.10, 0.44, 92, op, frame=soft(MODEL, 0.3))
        self.text(sc, 0.10, 0.575, "test digit %d: a %d" % (m["shown_digit"], m["shown_label"]),
                  op, size=TEXT_S)
        shift = -4.0 * np.sin(np.pi * smoothstep(0.30, 0.52, u))
        ch = self.pane(((0.22, 0.52), (0.36, 0.78)), (-0.5, 9.5), (-15.0, 12.0))
        self.axes(sc, ch, op, "digit", "logit", [(k, str(k)) for k in range(10)],
                  [(v, fixed(v, 0)) for v in (-10, -5, 0, 5, 10)], grid=False)
        self.hline(sc, ch, 0.0, AXIS, op, width=1.2, dashed=False)
        for k, v in enumerate(self.z + shift):
            ink = MODEL if k == a else (AMBER if k == b else DIM)
            y0, y1 = ch.fy(0.0), ch.fy(float(v))
            w, h = self.span(0.6 * (ch.x1 - ch.x0) / 10, abs(y1 - y0))
            self.rect(sc, self.P(ch.fx(k), (y0 + y1) / 2), w, max(h, 1.0), ink, op, radius=2,
                      lift=0.4)
            if k in (a, b):
                self.text(sc, ch.fx(k), y1 - 0.025, fixed(v), op, size=TEXT_S,
                          ink=dark(ink, 0.35))
        moved = op * smoothstep(0.33, 0.37, u) * (1 - smoothstep(0.52, 0.56, u))
        p = np.exp(S.log_softmax(self.z + shift))
        self.text(sc, 0.57, 0.53, "all ten logits %s %.2f: p(%d) still %.4f"
                  % ("−" if shift <= 0 else "+", abs(shift), a, p[a]), moved, anchor="lm",
                  size=SMALL, ink=dark(AMBER, 0.4))
        x = 0.57
        rows = [
            ("softmax: p(%d) = %.4f, p(%d) = %.4f" % (a, self.p[a], b, self.p[b]), None, SMALL),
            ("%.2f − %.2f = %.2f = ln(%.4f / %.4f)" % (self.z[a], self.z[b],
                                                       m["shown_top2_gap"], self.p[a],
                                                       self.p[b]), dark(AMBER, 0.4), SMALL),
            ("a logit gap is a log of odds", None, SMALL),
        ]
        self.reveal(sc, x, 0.38, rows, u, op, at=0.06, every=0.08)
        loss = m["shown_loss_nats"]
        rows = [
            ("loss = −ln p(true digit) = −ln %.4f" % self.p[a], None, SMALL),
            ("= %.3f nats = %.3f bits" % (loss, loss / np.log(2)), dark(NATS, 0.3), BIG),
            ("1 nat = 1 / ln 2 = %.3f bits" % (1 / np.log(2)), dark(BITS, 0.3), SMALL),
        ]
        self.reveal(sc, x, 0.62, rows, u, op, at=0.6, every=0.1, step=0.05)
        self.tag(sc, EXACT, op)

    def phase_2(self, sc, u, op):
        m = self.meta
        grow = smoothstep(0.03, 0.35, u)
        top = float(self.counts.max()) * 1.22
        ch = self.pane(((0.09, 0.60), (0.38, 0.74)), (self.edges[0], self.edges[-1]),
                       (0.0, top), xlog=True)
        self.axes(sc, ch, op, "the loss of a digit, −ln p (nats, log scale)",
                  "test digits in each bin", [(1e-6, "≤10" + sup(-6)), (1e-4, "10" + sup(-4)),
                                              (1e-2, "10" + sup(-2)), (1.0, "1"),
                                              (10.0, "10")],
                  [(v, commas(v)) for v in (500, 1000, 1500, 2000) if v < top])
        self.bars(sc, ch, self.edges, self.counts_right, MODEL, op, fill=0.7, grow=grow)
        self.bars(sc, ch, self.edges, self.counts_wrong, WRONG, op, fill=0.8,
                  base=self.counts_right * grow, grow=grow)
        lines = op * smoothstep(0.35, 0.45, u)
        for v, ink, word in ((m["median_nats"], BLUE, "median"), (m["mean_nats"], NATS, "mean")):
            self.vline(sc, ch, v, ink, lines, width=2.2, dashed=True)
            self.text(sc, ch.fx(v) + (-0.008 if word == "median" else 0.008), 0.405,
                      "%s %.3f" % (word, v), lines, ink=dark(ink, 0.3), size=TEXT_S,
                      anchor="rm" if word == "median" else "lm")
        self.key(sc, 0.10, 0.84, [("named right", MODEL), ("named wrongly (%s)"
                                                           % commas(self.wrong.sum()), WRONG)],
                 op, size=TEXT_S, step=0.04)
        self.text(sc, 0.36, 0.84, "the %s wrong carry %s of the loss" % (
            commas(self.wrong.sum()), pct(self.wrong_share)), op * smoothstep(0.4, 0.5, u),
            anchor="lm", size=TEXT_S, ink=dark(WRONG, 0.3))
        shares = m["worst_shares"]
        self.reveal(sc, 0.65, 0.40, [
            ("the share of the total loss", None, SMALL),
            ("worst 10 digits: %s" % pct(shares["10"]), None, SMALL),
            ("worst 100: %s" % pct(shares["100"]), dark(NATS, 0.3), HEAD),
            ("worst 1,000: %s" % pct(shares["1000"]), None, SMALL)], u, op, at=0.45,
            every=0.06)
        late = op * smoothstep(0.7, 0.8, u)
        self.digit(sc, self.worst_image, 0.695, 0.70, 62, late, frame=soft(WRONG, 0.3))
        self.lines(sc, 0.755, 0.68, ["the worst: a %d, called a %d" % (m["worst_label"],
                                                                     m["worst_named"]),
                                    "p = %s, %.2f nats" % (fixed(m["worst_probability"], 5),
                                                           m["worst_loss_nats"])],
                   late, size=TEXT_S, step=0.04)
        self.vline(sc, ch, m["worst_loss_nats"], WRONG, late, width=2.0)
        self.tag(sc, HELD, op)

    def phase_3(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.05, 0.315, "mean loss per digit, against two baselines", op,
                  anchor="lm", strong=True, size=HEAD)
        track, scale = 330.0, 2.5
        rows = (("a uniform guess", m["uniform_nats"], GREY),
                ("the class shares", m["prior_nats"], OCHRE),
                ("the classifier", m["mean_nats"], MODEL))
        for i, (name, v, ink) in enumerate(rows):
            g = smoothstep(0.04 + 0.1 * i, 0.2 + 0.1 * i, u)
            self.hbar(sc, 0.23, 0.39 + 0.065 * i, track * v / scale, ink, op, name,
                      "%.3f nats = %.3f bits" % (v, v / np.log(2)), u=g, track=track)
        self.text(sc, 0.23, 0.575, "uniform: ln 10 nats = log₂ 10 bits", op * smoothstep(
            0.3, 0.4, u), anchor="lm", size=TEXT_S)
        se = m["standard_error"]
        self.reveal(sc, 0.05, 0.665, [
            ("%s digits are a sample:" % commas(m["n"]), None, SMALL),
            ("standard deviation of a loss %.3f" % self.sd, None, SMALL),
            ("standard error %.3f / √%s = %.3f" % (self.sd, commas(m["n"]), se), None,
             SMALL),
            ("%.3f ± %.3f nats" % (m["mean_nats"], se), dark(NATS, 0.3), HEAD)], u, op,
            at=0.4, every=0.07)
        self.reveal(sc, 0.55, 0.665, [
            ("in bits: %.3f ± %.3f" % (m["mean_bits"], se / np.log(2)), dark(BITS, 0.3),
             SMALL),
            ("perplexity exp(%.3f) = %.2f" % (m["mean_nats"], m["perplexity"]), dark(NATS, 0.3),
             HEAD),
            ("as unsure as a fair pick among %.2f" % m["perplexity"], None, SMALL),
            ("digits; a uniform guess: %.0f" % np.exp(m["uniform_nats"]), None, SMALL)],
            u, op, at=0.66, every=0.06)
        self.tag(sc, HELD, op)

    def phase_4(self, sc, u, op):
        m = self.meta
        go = smoothstep(0.05, 0.55, u)
        n = m["n"]
        ch = self.pane(((0.10, 0.44), (0.38, 0.64)), (0, n), (0, 35000))
        self.axes(sc, ch, op, "test digits sent", "bits to send their labels",
                  [(v, commas(v)) for v in (0, 5000, 10000)],
                  [(v, commas(v)) for v in (0, 10000, 20000, 30000)], grid=False)
        self.path(sc, ch, [0, n], [0, m["raw_bits"]], GREY, op, go, width=2.4, dot=True)
        self.path(sc, ch, self.upto, self.cum_bits, MODEL, op, go, width=2.6, dot=True)
        k = int(round(go * (len(self.upto) - 1)))
        self.key(sc, 0.12, 0.405, [("without the model: %s" % commas(m["raw_bits"] * go), GREY),
                                    ("with it: %s" % commas(self.cum_bits[k]), MODEL)], op,
                 size=TEXT_S, step=0.04)
        ch2 = self.pane(((0.58, 0.93), (0.38, 0.64)), (0, n), (0, 1800))
        self.axes(sc, ch2, op, "test digits sent", "with the model, close up",
                  [(v, commas(v)) for v in (0, 5000, 10000)],
                  [(v, commas(v)) for v in (0, 500, 1000, 1500)])
        self.path(sc, ch2, self.upto, self.cum_bits, MODEL, op, go, width=2.6, dot=True)
        ll = m["log_likelihood"]
        self.reveal(sc, 0.05, 0.765, [
            ("likelihood: p₁ × p₂ × … × p₁₀₀₀₀ = exp(%s) ≈ 10%s"
             % (tenths(ll), sup(int(round(ll / np.log(10))))), None, SMALL),
            ("mean loss = −(log-likelihood) / %s = %s / %s = %.3f nats"
             % (commas(n), tenths(-ll), commas(n), m["mean_nats"]), dark(NATS, 0.3), SMALL),
            ("in bits: %s with the model against %s × log₂ 10 = %s without"
             % (commas(m["total_bits"]), commas(n), commas(m["raw_bits"])), dark(BITS, 0.3),
             SMALL)], u, op, at=0.55, every=0.1)
        self.tag(sc, HELD, op)

    def _sweep(self, u):
        """The temperature shown at ``u``: up from 1 to the largest, down to the smallest,
        back to 1, as an index into the study's temperatures."""
        last = len(self.T) - 1
        if u < 0.4:
            f = self.one + (last - self.one) * smoothstep(0.08, 0.4, u)
        elif u < 0.72:
            f = last - last * smoothstep(0.42, 0.72, u)
        else:
            f = self.one * smoothstep(0.74, 0.86, u)
        return int(round(f))

    def phase_5(self, sc, u, op):
        m = self.meta
        k = self._sweep(u)
        ticks = [(v, "%g" % v) for v in (0.25, 0.5, 1, 2, 4)]
        ch = self.pane(((0.10, 0.52), (0.38, 0.60)), (self.T[0], self.T[-1]), (0.0, 0.8),
                       xlog=True)
        self.axes(sc, ch, op, "", "mean loss, nats", ticks,
                  [(v, "%.1f" % v) for v in (0.2, 0.4, 0.6, 0.8)])
        self.path(sc, ch, self.T, self.first, soft(NATS, 0.5), op, width=1.6)
        self.path(sc, ch, self.T, self.second, NATS, op, width=2.6)
        ch2 = self.pane(((0.10, 0.52), (0.72, 0.80)), (self.T[0], self.T[-1]), (0.9, 1.0),
                        xlog=True)
        self.axes(sc, ch2, op, "temperature T (log scale)", "accuracy, second half", ticks,
                  [(0.95, "95%"), (1.0, "100%")])
        self.path(sc, ch2, self.T, self.acc_T, MODEL, op, width=2.6)
        for c, ys in ((ch, self.second), (ch2, self.acc_T)):
            self.disk(sc, self.P(c.fx(self.T[k]), c.fy(ys[k])), 6.0, AMBER, op, glow=0.4,
                      lift=1.4)
        x = ch.fx(self.T[self.best])
        self.dashes(sc, (x, ch.y1), (x, ch.fy(0.3)), AMBER, op * smoothstep(0.86, 0.92, u),
                    n=6, width=1.8)
        self.key(sc, 0.16, 0.405, [("first half: chooses T", soft(NATS, 0.5)),
                                  ("second half: scores it", NATS)], op, size=TEXT_S, step=0.04)
        self.text(sc, 0.58, 0.38, "logits ÷ T, then softmax", op, anchor="lm", strong=True,
                  size=SMALL)
        self.text(sc, 0.58, 0.46, "T = %.2f" % self.T[k], op, anchor="lm", ink=dark(AMBER, 0.4),
                  size=BIG)
        self.lines(sc, 0.58, 0.52, [
            ("second half: %.3f nats" % self.second[k], dark(NATS, 0.3)),
            ("accuracy %s" % pct(self.acc_T[k], 1), dark(MODEL, 0.3)),
            "digit %d: p(%d) = %.3f" % (m["shown_digit"], m["shown_label"], self.p_T[k])],
            op, size=SMALL, step=0.042)
        end = op * smoothstep(0.86, 0.92, u)
        self.lines(sc, 0.58, 0.70, [
            "dividing never changes the order,",
            "so never the answer",
            ("best on the first half: T = %.2f" % m["best_temperature"], dark(AMBER, 0.4)),
            "(the first half is harder: %.3f at T = 1)" % self.first[self.one]],
            end, size=SMALL, step=0.042)
        self.tag(sc, HALVES, op)


ANIMATIONS = {
    "weight-entropy": WeightEntropy,
    "logits-loss-and-nats": LogitsLossAndNats,
}

"""What a model was never shown, a generator that stops covering its data, and a
background with no formula: animations 154-156 of the "next" catalogue (see
:mod:`voxanim.mech`).

* :class:`AnomalyDetection`: an autoencoder trained without one digit class, and the
  rebuilding error as a score for what it never saw;
* :class:`ModeCollapse`: a GAN that settles on a few digits, counted by a separate
  classifier, and the same networks told the label;
* :class:`GPBackground`: a falling spectrum fitted by a fixed function and by a
  Gaussian process, with and without an injected resonance.

The three draw the experiments of :mod:`voxanim.study_phys_adv`, which records every
array shown here: nothing is computed in this module but layout, and the one pixel and
the one bin whose arithmetic is written out on screen. Every phase says, bottom right,
what kind of evidence it shows.
"""

from __future__ import annotations

import numpy as np

from . import study_phys_adv as S
from .easing import smoothstep
from .losschart import LossChart
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    GREEN,
    INKS,
    PAPER,
    RED,
    TEAL,
    VIOLET,
    dark,
    soft,
)
from .mech_chance import Chance

BIG = 0.034                                   # headings over a drawing
MID = 0.031
GREY = (0.50, 0.52, 0.54)
MNIST_TAG = "measured on MNIST, held-out digits"
MNIST_GAN_TAG = "measured on MNIST: samples read by a classifier"
PSEUDO = "pseudo-data: one seeded draw from the stated model"
EXACT = "exact: the expected counts, no fluctuation"
TOYS = "pseudo-experiments: %s seeded spectra"
TAG_STARTS = ("measured on MNIST", "pseudo-data", "exact", "pseudo-experiments")


def pct(v, digits=0):
    return "%.*f%%" % (digits, 100.0 * float(v))


def signed(v, digits=0):
    """A number with its sign, a true minus for negatives."""
    s = "%+.*f" % (digits, float(v))
    return s.replace("-", "−")


class PhysAdv(Chance):
    """What the three animations share: the evidence tag, bars, dots and bands on a
    chart, and a picture with a heat map beside it."""

    def tag(self, sc, kind, op):
        """The kind of evidence on screen, bottom right."""
        self.text(sc, 0.95, 0.925, kind, op, anchor="rm", ink=dark(TEAL, 0.25))

    def head(self, sc, s, op):
        self.text(sc, 0.5, 0.315, s, op, strong=True, size=BIG)

    def spots(self, sc, chart, xs, ys, ink, op, go=1.0, r=3.0):
        """A dot per point of ``ys`` against ``xs``, as far as fraction ``go`` of them."""
        n = int(np.ceil(len(xs) * float(np.clip(go, 0, 1)) - 1e-9))
        for x, y in zip(xs[:n], ys[:n]):
            self.disk(sc, self.P(chart.fx(float(x)), chart.fy(float(y))), r, ink, op, lift=1.2,
                      outline=0.0)

    def columns(self, sc, chart, xs, lo, hi, ink, op, wpx, go=1.0):
        """A bar per point from ``lo`` to ``hi`` (numbers or arrays, chart units)."""
        lo = np.broadcast_to(np.asarray(lo, float), np.shape(xs))
        hi = np.broadcast_to(np.asarray(hi, float), np.shape(xs))
        n = int(np.ceil(len(xs) * float(np.clip(go, 0, 1)) - 1e-9))
        for x, a, b in zip(xs[:n], lo[:n], hi[:n]):
            ya, yb = chart.fy(float(a)), chart.fy(float(b))
            h = max(self.span(0, abs(ya - yb))[1], 0.8)
            self.rect(sc, self.P(chart.fx(float(x)), (ya + yb) / 2), wpx, h, ink, op,
                      radius=min(1.5, wpx / 2), lift=0.3)

    def level(self, sc, chart, y, ink, op, width=1.6, x0=None, x1=None):
        """A horizontal line across ``chart`` at ``y``."""
        fy = chart.fy(float(y))
        a = chart.x0 if x0 is None else chart.fx(x0)
        b = chart.x1 if x1 is None else chart.fx(x1)
        self.polyline(sc, [self.P(a, fy), self.P(b, fy)], width, ink, op, bias=0.2)

    def upright(self, sc, chart, x, ink, op, width=1.8, y0=None, y1=None):
        """A vertical line on ``chart`` at ``x``."""
        fx = chart.fx(float(x))
        a = chart.y1 if y0 is None else chart.fy(y0)
        b = chart.y0 if y1 is None else chart.fy(y1)
        self.polyline(sc, [self.P(fx, a, lift=0.9), self.P(fx, b, lift=0.9)], width, ink, op,
                      bias=0.2)


# ====================================================================== 154: anomalies
class AnomalyDetection(PhysAdv):
    """Training digits with one class crossed out and the autoencoder they train; a seen
    and an unseen held-out digit going in and coming back, with their error maps; the
    two groups' error histograms, a sliding threshold and the ROC curve it traces; the
    area for every removed class and seed; and the area against the size of the code."""

    TITLE = "Anomaly detection with an autoencoder"
    PHASES = (("training", 10.0), ("error", 12.0), ("threshold", 12.0), ("classes", 11.0),
              ("code size", 10.0))
    SEEN, UNSEEN, CODE, MEAN = BLUE, RED, TEAL, VIOLET
    SHOWN_CODES = (2, 8, 32, 128)

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.left, self.code = int(m["featured"]), int(m["code_shown"])
        self.pics = np.asarray(p0["values"], float)
        self.errors = [float(v) for v in p0["errors"]]
        self.classes = [int(v) for v in p0["classes"]]
        self.train = np.asarray(p0["train_examples"], float)
        self.hard = np.asarray(p0["hardest_values"], float)
        self.hard_error = float(p0["hardest_error"])
        self.hard_seen = float(p0["hardest_seen_error"])
        self.maps = [(self.pics[1] - self.pics[0]) ** 2, (self.pics[3] - self.pics[2]) ** 2]
        self.worst = np.unravel_index(int(self.maps[1].argmax()), (28, 28))
        self.hist = np.asarray(p1["values"], float)
        self.edges = np.asarray(p1["edges"], float)
        self.thr = float(p1["threshold"])
        self.seen = np.sort(np.asarray(p1["seen"], float))
        self.unseen = np.sort(np.asarray(p1["unseen"], float))
        self.grid_x = np.asarray(p2["x"], float)
        self.roc = np.asarray(p2["seeds"], float)[self.left, 0]
        self.thr_grid = np.quantile(self.seen, 1.0 - self.grid_x)
        self.auc = np.asarray(p2["auc"], float)                   # [removed class, seed]
        self.codes = [int(v) for v in p3["x"]]
        self.area = np.asarray(p3["ys"], float)                   # [removed class, code]
        self.area_mean = np.asarray(p3["mean"], float)
        self.code_pics = np.asarray(p3["pictures"], float)
        self.code_in = np.asarray(p3["picture_in"], float)
        self.code_errors = np.asarray(p3["picture_errors"], float)
        self.group_mse = (np.asarray(p3["featured_seen_mse"], float),
                          np.asarray(p3["featured_unseen_mse"], float))

    def subtitle_for(self, k):
        m = self.meta
        by = m["auc_by_class"]
        easy, hard = m["easiest"], m["hardest"]
        why = ("under a coin toss, because a simple shape is easy to rebuild, seen or not"
               if by[str(hard)]["mean"] < 0.5 else "some digits are much harder to catch")
        main, last = self.codes.index(self.code), len(self.codes) - 1
        return [
            "an autoencoder squeezes a picture through a few numbers and rebuilds it. This "
            "one trains on real digits with every %d removed: the nine kept classes stand "
            "for the known processes, the %ds for new physics" % (self.left, self.left),
            "a held-out %d, a class it trained on, comes back close: mean squared pixel "
            "error %.3f. A held-out %d, the class it never saw, comes back as something "
            "else: %.3f. That error is the anomaly score"
            % (self.classes[0], self.errors[0], self.classes[1], self.errors[1]),
            "put a threshold on the error. Where %s of seen digits pass, %s of the unseen "
            "%ds are flagged. Sliding the threshold traces the ROC curve; the area under "
            "it is %.2f (1 is perfect, 0.5 a coin toss)"
            % (pct(1 - m["false_alarm"]), pct(m["flagged"]), self.left,
               self.auc[self.left, 0]),
            "it depends on what is missing. Over ten removed classes and three seeds the "
            "area runs from %.2f (the %ds) down to %.2f (the %ds): %s"
            % (by[str(easy)]["mean"], easy, by[str(hard)]["mean"], hard, why),
            "and on the code: with %d numbers the mean area is %.2f, with %d it is %.2f, "
            "because a roomy code rebuilds any digit, unseen ones included. Choosing the "
            "size takes anomalies to test on, which a real search lacks"
            % (self.code, self.area_mean[main], self.codes[last], self.area_mean[last]),
        ][k]

    # ------------------------------------------------------------------ phase 1
    def phase_1(self, sc, u, op):
        left = self.left
        self.head(sc, "the training digits, with every %d taken out" % left, op)
        gone = smoothstep(0.12, 0.24, u)
        for c in range(10):
            x = 0.14 + 0.08 * c
            out = c == left
            self.digit(sc, self.train[c], x, 0.425, 56, op * (1 - 0.6 * gone if out else 1.0),
                       frame=self.UNSEEN if out else self.SEEN)
            if out:
                for sx in (-1, 1):
                    a, b = self.frac(34 * sx, 34)
                    self.polyline(sc, [self.P(x - a, 0.425 - b, lift=1.0),
                                       self.P(x + a, 0.425 + b, lift=1.0)], 4.0, self.UNSEEN,
                                  op * gone)
        self.text(sc, 0.14 + 0.08 * left, 0.515, "never shown" if left in (0, 9) else "out",
                  op * gone, ink=dark(self.UNSEEN, 0.3), size=MID)
        self.text(sc, 0.14 + 0.08 * (5.0 if left < 5 else 4.0), 0.515,
                  "the nine kept classes: the known processes", op * gone,
                  ink=dark(self.SEEN, 0.3), size=MID)
        boxes = ((0.14, "784 pixels", 112, PAPER, AXIS), (0.32, "256", 64, PAPER, AXIS),
                 (0.50, "code: %d numbers" % self.code, 170, soft(self.CODE), self.CODE),
                 (0.68, "256", 64, PAPER, AXIS), (0.86, "784 pixels", 112, PAPER, AXIS))
        net = op * smoothstep(0.30, 0.40, u)
        for n, (x, name, w, fill, border) in enumerate(boxes):
            self.box(sc, x, 0.655, w, 44, fill, net, name, border=border, size=MID)
            if n:
                x0, w0 = boxes[n - 1][0], boxes[n - 1][2]
                a = (x0 + self.frac(w0 / 2 + 6)[0], 0.655)
                b = (x - self.frac(w / 2 + 6)[0], 0.655)
                self.arrow(sc, a, b, AXIS, net, smoothstep(0.34 + 0.05 * n, 0.44 + 0.05 * n, u),
                           2.4)
                for lap in range(2):
                    self.pulse(sc, a, b, (u - 0.50 - 0.2 * lap) / 0.08 - 0.9 * (n - 1) + 0.0,
                               self.CODE, net)
        self.text(sc, 0.14, 0.59, "a digit in", net, size=0.030)
        self.text(sc, 0.86, 0.59, "the digit rebuilt", net, size=0.030)
        self.text(sc, 0.50, 0.59, "the only narrow place", net, size=0.030,
                  ink=dark(self.CODE, 0.3))
        self.text(sc, 0.5, 0.755, "trained to return its own input.  A picture's score: the "
                  "mean of (out − in)² over its pixels", op * smoothstep(0.52, 0.62, u),
                  size=MID)
        self.text(sc, 0.5, 0.835, "a model-independent search does the same: trained on "
                  "known processes, it keeps what it rebuilds badly",
                  op * smoothstep(0.70, 0.80, u), size=MID, ink=dark(self.UNSEEN, 0.3))
        self.tag(sc, "measured on MNIST: the training digits", op)

    # ------------------------------------------------------------------ phase 2
    def phase_2(self, sc, u, op):
        self.head(sc, "a held-out digit goes in; what comes back is compared pixel by pixel",
                  op)
        for x, name in ((0.25, "in"), (0.43, "out"), (0.61, "(out − in)²")):
            self.text(sc, x, 0.372, name, op, size=MID)
        rows = ((0.505, self.SEEN, "a class it", "trained on", 0.02),
                (0.75, self.UNSEEN, "the class it", "never saw", 0.44))
        size = 92
        for j, (y, ink, a, b, t0) in enumerate(rows):
            def t(k, t0=t0):
                return op * smoothstep(t0 + 0.07 * k, t0 + 0.07 * k + 0.06, u)
            self.lines(sc, 0.055, y - 0.022, [(a, dark(ink, 0.3)), (b, dark(ink, 0.3))], t(0),
                       size=MID, step=0.044)
            self.digit(sc, self.pics[2 * j], 0.25, y, size, t(0), frame=ink)
            self.arrow(sc, (0.315, y), (0.365, y), AXIS, t(1),
                       smoothstep(t0 + 0.07, t0 + 0.16, u), 2.6)
            self.pulse(sc, (0.315, y), (0.365, y), (u - t0 - 0.07) / 0.09, self.CODE, op)
            self.digit(sc, self.pics[2 * j + 1], 0.43, y, size, t(2), frame=ink)
            cell = self.grid(sc, self.maps[j], 0.61, y, size, size, t(3), vmax=1.0, lines=0.0)
            self.text(sc, 0.70, y - 0.058, "error = %.3f" % self.errors[j], t(4), anchor="lm",
                      strong=True, size=BIG)
            self.text(sc, 0.70, y - 0.014, "the mean of the 784 squares", t(4), anchor="lm",
                      size=0.030)
            if j == 1:
                r, c = self.worst
                self.rect(sc, self.P(*cell(r, c)), 13, 13, AMBER, t(5), border=AMBER,
                          border_w=2.4, radius=3, lift=0.9, fill_op=0.0)
                a, b = self.pics[2][r, c], self.pics[3][r, c]
                self.text(sc, 0.70, y + 0.036, "ringed pixel: in %.2f, out %.2f" % (a, b), t(5),
                          anchor="lm", size=0.030, ink=dark(AMBER, 0.4))
                self.text(sc, 0.70, y + 0.078, "(%.2f − %.2f)² = %.2f" % (b, a, (b - a) ** 2),
                          t(5), anchor="lm", size=0.030, ink=dark(AMBER, 0.4))
        self.tag(sc, MNIST_TAG, op)

    # ------------------------------------------------------------------ phase 3
    def phase_3(self, sc, u, op):
        self.head(sc, "the error of every held-out digit, and a threshold that slides", op)
        e = self.edges
        mids = (e[:-1] + e[1:]) / 2
        chart = LossChart(((0.085, 0.52), (0.42, 0.76)), (float(e[0]), float(e[-1])),
                          (0.0, float(self.hist.max()) * 1.05))
        self.axes(sc, chart, op, "rebuilding error", "share of digits", yticks=[], grid=False)
        wpx = self.span(0.435 / len(mids))[0]
        self.columns(sc, chart, mids, 0.0, self.hist[0], soft(self.SEEN, 0.15), op, wpx * 0.86,
                     smoothstep(0.02, 0.10, u))
        self.columns(sc, chart, mids, 0.0, self.hist[1], self.UNSEEN, op, wpx * 0.42,
                     smoothstep(0.02, 0.10, u))
        self.key(sc, 0.24, 0.385, [("seen classes", self.SEEN)], op, size=0.030)
        self.key(sc, 0.38, 0.385, [("unseen: the %ds" % self.left, self.UNSEEN)], op,
                 size=0.030)
        # the false-alarm rate sweeps from 0 to 1, then settles where 95% of seen digits pass
        rest = float(self.meta["false_alarm"])
        sweep = smoothstep(0.14, 0.56, u)
        back = smoothstep(0.60, 0.72, u)
        rate = sweep + (rest - 1.0) * back
        thr = float(np.quantile(self.seen, 1.0 - rate)) if back < 1 else self.thr
        show = op * smoothstep(0.10, 0.14, u)
        self.upright(sc, chart, min(thr, e[-1]), AMBER, show, 3.0)
        n_seen, n_unseen = len(self.seen), len(self.unseen)
        a = n_seen - int(np.searchsorted(self.seen, thr, "right"))
        b = n_unseen - int(np.searchsorted(self.unseen, thr, "right"))
        self.text(sc, 0.05, 0.875, "above the threshold:  %s of %s seen (%s)  ·  %d of %d "
                  "unseen (%s)" % (f"{a:,}", f"{n_seen:,}", pct(a / n_seen), b, n_unseen,
                                   pct(b / n_unseen)), show, size=MID, anchor="lm",
                  ink=dark(AMBER, 0.5))
        roc = LossChart(((0.675, 0.925), (0.40, 0.76)), (0.0, 1.0), (0.0, 1.0))
        ticks = [(0.0, "0"), (0.5, "50%"), (1.0, "100%")]
        self.axes(sc, roc, show, "seen flagged", "unseen flagged", xticks=ticks, yticks=ticks)
        self.polyline(sc, [self.P(roc.fx(0), roc.fy(0)), self.P(roc.fx(1), roc.fy(1))], 1.4,
                      GREY, 0.7 * show)
        self.trace(sc, roc, self.grid_x, self.roc, self.UNSEEN, show, sweep, 3.0, dots=False)
        self.disk(sc, self.P(roc.fx(rate), roc.fy(b / n_unseen)), 6.5, AMBER, show, glow=0.4,
                  lift=1.6, outline=0.0)
        done = op * smoothstep(0.74, 0.84, u)
        self.text(sc, 0.845, 0.67, "area %.2f" % self.auc[self.left, 0], done, strong=True,
                  size=BIG)
        self.text(sc, 0.95, 0.875, "the diagonal: a coin toss", done, size=0.030, anchor="rm")
        self.tag(sc, MNIST_TAG, op)

    # ------------------------------------------------------------------ phase 4
    def phase_4(self, sc, u, op):
        m = self.meta
        hard = int(m["hardest"])
        self.head(sc, "each class removed in turn, three seeds each: area under the ROC "
                  "curve", op)
        chart = LossChart(((0.09, 0.60), (0.40, 0.80)), (-0.6, 9.6), (0.0, 1.0))
        self.axes(sc, chart, op, "the class removed", "",
                  xticks=[(c, str(c)) for c in range(10)],
                  yticks=[(0.0, "0"), (0.5, "0.5"), (1.0, "1")])
        chart.hline(sc, self.P, 0.5, op, dashes=24, width=1.4)
        self.text(sc, 0.607, chart.fy(0.5), "coin toss", op, anchor="lm", size=0.030)
        mean = self.auc.mean(1)
        for c in range(10):
            go = smoothstep(0.05 + 0.05 * c, 0.15 + 0.05 * c, u)
            ink = self.UNSEEN if mean[c] < 0.5 else self.MEAN
            self.columns(sc, chart, [c], 0.0, [mean[c] * go], soft(ink, 0.25), op, 30.0)
            for k, v in enumerate(self.auc[c]):
                self.disk(sc, self.P(chart.fx(c + 0.22 * (k - 1)), chart.fy(v * go)), 4.2,
                          dark(ink, 0.25), op * go, lift=1.4, outline=0.0)
        on = op * smoothstep(0.62, 0.72, u)
        x = 0.825
        self.text(sc, x, 0.40, "the hardest: an unseen %d" % hard, on, strong=True, size=MID)
        self.digit(sc, self.hard[0], x - 0.075, 0.535, 78, on, frame=self.UNSEEN)
        self.arrow(sc, (x - 0.025, 0.535), (x + 0.025, 0.535), AXIS, on, 1.0, 2.6)
        self.digit(sc, self.hard[1], x + 0.075, 0.535, 78, on, frame=self.UNSEEN)
        better = self.hard_error < self.hard_seen
        self.lines(sc, x, 0.675, [
            "it comes back with error %.3f," % self.hard_error,
            "%s the typical seen" % ("below" if better else "near"),
            "digit's %.3f" % self.hard_seen], on, size=MID, step=0.046, anchor="mm")
        self.tag(sc, MNIST_TAG, op)

    # ------------------------------------------------------------------ phase 5
    def phase_5(self, sc, u, op):
        self.head(sc, "the same test at seven code sizes", op)
        codes = self.codes
        low = float(np.floor(self.area.min() * 10) / 10)
        chart = LossChart(((0.09, 0.50), (0.40, 0.80)), (codes[0] * 0.85, codes[-1] * 1.18),
                          (low, 1.0), xlog=True)
        self.axes(sc, chart, op, "numbers in the code", "area under the ROC curve",
                  xticks=[(c, str(c)) for c in codes],
                  yticks=[(v, "%.1f" % v) for v in np.arange(low, 1.01, 0.2)])
        if low < 0.5:
            chart.hline(sc, self.P, 0.5, op, dashes=24, width=1.4)
        go = smoothstep(0.06, 0.50, u)
        for c in range(10):
            if c != self.left:
                self.trace(sc, chart, codes, self.area[c], GREY, 0.5 * op, go, 1.6, dots=False)
        self.trace(sc, chart, codes, self.area[self.left], self.UNSEEN, op, go, 2.4, r=3.4)
        self.trace(sc, chart, codes, self.area_mean, self.MEAN, op, go, 3.4, r=4.6)
        self.text(sc, 0.74, 0.385, "one unseen %d, rebuilt at four code sizes" % self.left, op,
                  size=MID, strong=True)
        xs = (0.585, 0.675, 0.755, 0.835, 0.915)
        self.digit(sc, self.code_in, xs[0], 0.50, 58, op, frame=self.UNSEEN)
        self.text(sc, xs[0], 0.585, "in", op, size=0.030)
        for n, code in enumerate(self.SHOWN_CODES):
            k = codes.index(code)
            on = op * smoothstep(0.12 + 0.12 * n, 0.22 + 0.12 * n, u)
            self.digit(sc, self.code_pics[k], xs[n + 1], 0.50, 58, on, frame=self.CODE)
            self.text(sc, xs[n + 1], 0.585, "code %d" % code, on, size=0.030,
                      ink=dark(self.CODE, 0.3))
            self.text(sc, xs[n + 1], 0.625, "%.3f" % self.code_errors[k], on, size=0.030)
        self.text(sc, 0.555, 0.625, "error", op * smoothstep(0.12, 0.22, u), size=0.030,
                  anchor="lm")
        note = op * smoothstep(0.62, 0.72, u)
        main, last = codes.index(self.code), len(codes) - 1
        seen, unseen = self.group_mse
        self.lines(sc, 0.75, 0.70, [
            ("mean error, seen / unseen %ds" % self.left, None),
            ("code %d:  %.3f / %.3f" % (self.code, seen[main], unseen[main]), None),
            ("code %d:  %.3f / %.3f" % (codes[last], seen[last], unseen[last]), None)],
            note, size=MID, step=0.046, anchor="mm")
        self.key(sc, 0.56, 0.855, [("removed class %d" % self.left, self.UNSEEN)], op,
                 size=0.030)
        self.key(sc, 0.76, 0.855, [("mean of ten", self.MEAN)], op, size=0.030)
        self.tag(sc, MNIST_TAG, op)


# ====================================================================== 155: mode collapse
BAL, HEAVY, BAL_L, HEAVY_L = range(4)


class ModeCollapse(PhysAdv):
    """A generator's samples read by a separate classifier into a histogram of digits,
    with its entropy; the same networks under a generator-heavy schedule, their samples
    and histogram followed through training as they settle on a digit; entropy against
    training for five seeds of each; the label given to both networks and samples asked
    for by digit; and the share read as asked, with the digits each setting covers."""

    TITLE = "Mode collapse in a GAN"
    PHASES = (("the count", 10.0), ("collapse", 13.0), ("five seeds", 11.0),
              ("the label", 12.0), ("on demand", 9.0))
    INK = (GREEN, RED, BLUE, VIOLET)
    SHORT = ("balanced", "generator-heavy", "balanced + label", "heavy + label")

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.names = list(m["settings"])
        self.pics = np.asarray(p0["history"], float)            # [setting, seed, snap, 20, 784]
        self.steps = [int(v) for v in p0["steps"]]
        self.read = np.asarray(p0["read"], int)
        self.asked = np.asarray(p0["asked"], int)
        self.real = np.asarray(p1["values"], float)[0]
        self.hist = np.asarray(p1["history"], float)            # [setting, seed, snap, digit]
        self.H = np.asarray(p2["seeds"], float)                 # [setting, seed, snap]
        self.H_mean = np.asarray(p2["ys"], float)
        self.covered = np.asarray(p2["covered"], int)
        self.top = np.asarray(p2["top"], int)
        self.hit = np.asarray(p3["ys"], float)                  # [setting, snap]
        self.vmax = float(max(0.3, self.hist[BAL, 0, -1].max(), self.real.max()))
        self.end = {k: [m[k][n]["mean"] for n in self.names]
                    for k in ("entropy_end", "covered_end", "asked_end")}
        self.collapsed = [int(m["collapsed"][n]) for n in self.names]
        self.n_seeds = len(m["seeds"])

    def subtitle_for(self, k):
        m = self.meta
        e, c, a = self.end["entropy_end"], self.end["covered_end"], self.end["asked_end"]
        last = self.hist[HEAVY, 0, -1]
        return [
            "a generator turns noise into pictures; nothing makes it cover every digit. "
            "To check, a separately trained classifier reads %s samples: this one "
            "makes all ten, entropy %.2f of a possible %.2f bits"
            % (f"{m['samples']:,}", self.H[BAL, 0, -1], m["max_entropy"]),
            "give the generator %d updates per discriminator update and it stops "
            "spreading out: it chases whatever fools the discriminator now. After %s "
            "updates %s of its samples are one digit: entropy %.2f bits"
            % (S.GAN_HEAVY, f"{m['steps']:,}", pct(last.max()), self.H[HEAVY, 0, -1]),
            "over %d seeds the balanced schedule ends at %.1f bits and the "
            "generator-heavy one at %.1f: %d of its %d runs collapse (under %g bits). "
            "Which digit a run lands on depends on its seed"
            % (self.n_seeds, e[BAL], e[HEAVY], self.collapsed[HEAVY], self.n_seeds,
               S.GAN_COLLAPSED),
            "conditioning: generator and discriminator are both told which digit. The "
            "discriminator now rejects a 3 offered as a 7, so every label needs its own "
            "digit. These samples were asked for 0 to 9 (first seed)",
            "the classifier reads %s of the labelled samples as the digit asked for, %s "
            "under the generator-heavy schedule. There the label lifts the digits covered "
            "from %.1f to %.1f of 10, and %d of %d runs collapse instead of %d"
            % (pct(a[BAL_L]), pct(a[HEAVY_L]), c[HEAVY], c[HEAVY_L], self.collapsed[HEAVY_L],
               self.n_seeds, self.collapsed[HEAVY]),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _samples(self, sc, pics, op, ink, shown=20):
        """Twenty samples as a 5 x 4 block on the left."""
        for n in range(min(shown, 20)):
            self.digit(sc, pics[n], 0.078 + 0.059 * (n % 5), 0.448 + 0.103 * (n // 5), 48, op,
                       frame=ink)

    def _reader(self, sc, op, go):
        self.arrow(sc, (0.352, 0.60), (0.392, 0.60), AXIS, op, go, 2.6)
        self.box(sc, 0.455, 0.60, 112, 44, PAPER, op, "classifier", border=AXIS, size=MID)
        self.arrow(sc, (0.518, 0.60), (0.558, 0.60), AXIS, op, go, 2.6)

    def _histogram(self, sc, shares, ink, op, go=1.0):
        chart = LossChart(((0.60, 0.93), (0.42, 0.74)), (-0.6, 9.6), (0.0, 1.0))
        self.axes(sc, chart, op, "read as the digit", "",
                  xticks=[(d, str(d)) for d in range(10)],
                  yticks=[(0.0, "0"), (0.5, "50%"), (1.0, "100%")])
        self.columns(sc, chart, np.arange(10), 0.0, np.asarray(shares) * go, ink, op, 20.0)
        for d in range(10):                                   # the real digits' shares
            self.level(sc, chart, self.real[d], dark(GREY, 0.4), op, 2.4, d - 0.42, d + 0.42)
        return chart

    # ------------------------------------------------------------------ phase 1
    def phase_1(self, sc, u, op):
        m = self.meta
        self.head(sc, "which digits does a generator make? A separate classifier counts", op)
        shares = self.hist[BAL, 0, -1]
        shown = int(np.clip(u / 0.2, 0, 1) * 20 + 1e-9)
        self._samples(sc, self.pics[BAL, 0, -1], op, self.INK[BAL], shown)
        self.text(sc, 0.196, 0.372, "samples of a trained generator", op, size=0.030)
        go = smoothstep(0.22, 0.34, u)
        self._reader(sc, op * smoothstep(0.2, 0.26, u), go)
        self.text(sc, 0.455, 0.665, "trained apart", op * go, size=0.030)
        self.text(sc, 0.455, 0.705, "on real digits", op * go, size=0.030)
        bars = smoothstep(0.34, 0.58, u)
        self._histogram(sc, shares, self.INK[BAL], op * smoothstep(0.30, 0.36, u), bars)
        self.text(sc, 0.765, 0.372, "share of samples; dark marks: the real digits", op * bars,
                  size=0.030)
        on = op * smoothstep(0.62, 0.72, u)
        self.text(sc, 0.05, 0.875, "entropy H = −Σ p·log₂ p = −(%.2f·log₂ %.2f + %.2f·log₂ "
                  "%.2f + …)  =  %.2f bits" % (shares[0], shares[0], shares[1], shares[1],
                                               self.H[BAL, 0, -1]), on, size=MID, anchor="lm",
                  strong=True)
        self.text(sc, 0.05, 0.925, "ten equal digits: %.2f bits;  one digit: 0"
                  % m["max_entropy"], op * smoothstep(0.76, 0.86, u), size=0.030, anchor="lm")
        self.tag(sc, MNIST_GAN_TAG.split(":")[0], op)

    # ------------------------------------------------------------------ phase 2
    def phase_2(self, sc, u, op):
        self.head(sc, "the same two networks, with %d generator updates for each "
                  "discriminator update" % S.GAN_HEAVY, op)
        last = len(self.steps) - 1
        i = int(round(smoothstep(0.0, 1.0, np.clip((u - 0.05) / 0.75, 0, 1)) * last))
        ink = self.INK[HEAVY]
        self._samples(sc, self.pics[HEAVY, 0, i], op, ink)
        self.text(sc, 0.196, 0.372, "after %s updates" % f"{self.steps[i]:,}", op, size=MID,
                  strong=True)
        self._reader(sc, op, 1.0)
        shares = self.hist[HEAVY, 0, i]
        self._histogram(sc, shares, ink, op)
        self.text(sc, 0.765, 0.372, "entropy %.2f bits   ·   %d of 10 digits"
                  % (abs(self.H[HEAVY, 0, i]), self.covered[HEAVY, 0, i]), op, size=MID,
                  strong=True)
        self.text(sc, 0.05, 0.872, "its commonest digit, reading by reading:", op, anchor="lm",
                  size=0.030)
        for j in range(i + 1):
            d = int(self.top[HEAVY, 0, j])
            self.box(sc, 0.405 + 0.0176 * j, 0.872, 15, 24, soft(INKS[d % 8], 0.45), op, str(d),
                     size=0.029, radius=3)
        self.tag(sc, MNIST_GAN_TAG, op)

    # ------------------------------------------------------------------ phase 3
    def phase_3(self, sc, u, op):
        m = self.meta
        self.head(sc, "entropy of the generated digits through training: %d seeds of each "
                  "schedule" % self.n_seeds, op)
        chart = LossChart(((0.09, 0.60), (0.40, 0.80)), (0.0, float(self.steps[-1])),
                          (0.0, 3.6))
        self.axes(sc, chart, op, "discriminator updates", "entropy (bits)",
                  yticks=[(v, "%d" % v) for v in (0, 1, 2, 3)])
        self.level(sc, chart, m["max_entropy"], dark(GREY, 0.4), op, 1.8)
        chart.hline(sc, self.P, S.GAN_COLLAPSED, op, dashes=30, width=1.4)
        for k, t0 in ((BAL, 0.05), (HEAVY, 0.36)):
            go = smoothstep(t0, t0 + 0.30, u)
            for seed in range(self.n_seeds):
                self.trace(sc, chart, self.steps, self.H[k, seed], soft(self.INK[k], 0.35), op,
                           go, 1.6, dots=False)
            self.trace(sc, chart, self.steps, self.H_mean[k], self.INK[k], op, go, 3.4,
                       dots=False)
        x = 0.645
        self.key(sc, x, 0.405, [("balanced: 1 to 1", self.INK[BAL]),
                                ("generator-heavy: %d to 1" % S.GAN_HEAVY, self.INK[HEAVY])],
                 op, size=0.030)
        self.lines(sc, x - 0.008, 0.515, ["grey line: ten equal digits, %.2f"
                                          % m["max_entropy"],
                                          "dashes: the collapse rule, %g bits" % S.GAN_COLLAPSED],
                   op, size=0.030, step=0.044)
        on = op * smoothstep(0.70, 0.80, u)
        self.lines(sc, x - 0.008, 0.612, [
            ("generator-heavy: %d of %d runs collapse" % (self.collapsed[HEAVY], self.n_seeds),
             dark(self.INK[HEAVY], 0.3)),
            ("balanced: %d of %d" % (self.collapsed[BAL], self.n_seeds),
             dark(self.INK[BAL], 0.3))], on, size=MID, step=0.044)
        # one sample of each generator-heavy run, followed along its curve
        run = op * smoothstep(0.34, 0.40, u)
        i = int(round(smoothstep(0.36, 0.66, u) * (len(self.steps) - 1)))
        self.text(sc, x - 0.008, 0.715, "each heavy run: a sample, its top digit",
                  run, size=0.030, anchor="lm")
        for seed in range(self.n_seeds):
            fx = 0.665 + 0.062 * seed
            self.digit(sc, self.pics[HEAVY, seed, i, 0], fx, 0.795, 44, run,
                       frame=self.INK[HEAVY])
            self.text(sc, fx, 0.872, str(self.top[HEAVY, seed, i]), run, size=MID, strong=True)
        self.tag(sc, MNIST_GAN_TAG, op)

    # ------------------------------------------------------------------ phase 4
    def phase_4(self, sc, u, op):
        self.head(sc, "tell both networks the digit: the label goes in beside the noise and "
                  "beside the picture", op)
        boxes = ((0.13, "noise + label", 150, AMBER), (0.325, "generator", 116, AXIS),
                 (0.53, "picture + label", 164, AMBER), (0.745, "discriminator", 142, AXIS))
        y = 0.40
        for n, (x, name, w, border) in enumerate(boxes):
            self.box(sc, x, y, w, 40, soft(AMBER, 0.6) if border is AMBER else PAPER, op, name,
                     border=border, size=MID)
            if n:
                x0, w0 = boxes[n - 1][0], boxes[n - 1][2]
                a = (x0 + self.frac(w0 / 2 + 5)[0], y)
                b = (x - self.frac(w / 2 + 5)[0], y)
                self.arrow(sc, a, b, AXIS, op, smoothstep(0.03 * n, 0.03 * n + 0.1, u), 2.4)
                self.pulse(sc, a, b, (u - 0.12) / 0.07 - (n - 1), AMBER, op)
        self.text(sc, 0.832, y, "→  real or made?", op, anchor="lm", size=0.030)
        self.text(sc, 0.118, 0.505, "asked for", op, anchor="rm", size=0.030)
        rows = ((BAL_L, 0.62, "balanced"), (HEAVY_L, 0.745, "generator-heavy"))
        for d in range(10):
            x = 0.165 + 0.079 * d
            self.box(sc, x, 0.505, 40, 32, soft(AMBER, 0.6), op, str(d), border=AMBER, size=BIG)
        for r, (k, yy, name) in enumerate(rows):
            self.text(sc, 0.118, yy - (0.0 if r == 0 else 0.022), name if r == 0 else
                      "generator-", op, anchor="rm", size=0.030, ink=dark(self.INK[k], 0.3))
            if r:
                self.text(sc, 0.118, yy + 0.022, "heavy", op, anchor="rm", size=0.030,
                          ink=dark(self.INK[k], 0.3))
            for d in range(10):
                on = op * smoothstep(0.2 + 0.25 * r + 0.02 * d, 0.26 + 0.25 * r + 0.02 * d, u)
                ok = self.read[k, d] == self.asked[d]
                self.digit(sc, self.pics[k, 0, -1, d], 0.165 + 0.079 * d, yy, 54, on,
                           frame=self.INK[k] if ok else GREY)
                if not ok:
                    self.text(sc, 0.165 + 0.079 * d + 0.039, yy + 0.035, str(self.read[k, d]),
                              on, size=0.029, ink=dark(RED, 0.2))
        a = self.end["asked_end"]
        self.text(sc, 0.05, 0.842, "grey frame: the classifier reads another digit (the small "
                  "red number)", op * smoothstep(0.6, 0.7, u), anchor="lm", size=0.030)
        self.text(sc, 0.05, 0.887, "read as asked, %s samples, mean of %d seeds:  %s and %s"
                  % (f"{self.meta['samples']:,}", self.n_seeds, pct(a[BAL_L]), pct(a[HEAVY_L])),
                  op * smoothstep(0.72, 0.82, u), anchor="lm", size=MID, strong=True)
        self.tag(sc, MNIST_GAN_TAG.split(":")[0], op)

    # ------------------------------------------------------------------ phase 5
    def phase_5(self, sc, u, op):
        self.head(sc, "does it draw what it is asked?  And how many digits does it cover?", op)
        chart = LossChart(((0.09, 0.47), (0.44, 0.80)), (0.0, float(self.steps[-1])), (0.0, 1.0))
        self.axes(sc, chart, op, "discriminator updates", "",
                  yticks=[(v, pct(v)) for v in (0, 0.5, 1.0)])
        self.text(sc, 0.28, 0.395, "read as the digit asked for", op, size=MID, strong=True)
        go = smoothstep(0.05, 0.40, u)
        for k in (BAL, HEAVY, BAL_L, HEAVY_L):
            self.trace(sc, chart, self.steps, self.hit[k], self.INK[k], op, go,
                       3.2 if k >= BAL_L else 1.8, dots=False)
        self.text(sc, 0.465, chart.fy(0.1) - 0.035, "no label: chance", op, anchor="rm",
                  size=0.030)
        self.text(sc, 0.775, 0.395, "digits covered at the end, of 10", op, size=MID,
                  strong=True)
        c = self.end["covered_end"]
        for k in range(4):
            y = 0.475 + 0.082 * k
            on = op * smoothstep(0.20 + 0.07 * k, 0.28 + 0.07 * k, u)
            self.hbar(sc, 0.735, y, 15.0 * c[k], self.INK[k], on, self.SHORT[k], "%.1f" % c[k],
                      smoothstep(0.22 + 0.07 * k, 0.40 + 0.07 * k, u), thick=22.0)
        self.text(sc, 0.74, 0.815, "runs collapsed, of %d:  %s" % (
            self.n_seeds, "  ·  ".join(str(v) for v in self.collapsed)),
            op * smoothstep(0.66, 0.76, u), size=MID)
        self.text(sc, 0.74, 0.858, "(same order)", op * smoothstep(0.66, 0.76, u), size=0.030)
        self.tag(sc, MNIST_GAN_TAG, op)


# ====================================================================== 156: a GP background
class GPBackground(PhysAdv):
    """A spectrum from a stated model and the exponential fitted to it; the wave the fit
    leaves, bin by bin and as a spurious signal across the spectrum; a Gaussian process
    as a sliding weighted average that follows the wave and gives a band; a resonance
    injected and what three backgrounds leave of it; and, over thousands of spectra,
    spurious signal and share recovered against the kernel's length scale."""

    TITLE = "A Gaussian-process background"
    PHASES = (("spectrum", 10.0), ("wave", 11.0), ("process", 12.0), ("catch", 12.0),
              ("trade", 11.0))
    EXP, GP, SHORT, SIGNAL = RED, BLUE, AMBER, GREEN

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.m = np.asarray(p0["x"], float)
        self.data, self.truth, self.fit, self.bkg = (np.asarray(v, float) for v in p0["ys"])
        self.band = np.asarray(p0["band"], float)
        self.pull_exp, self.pull_gp = (np.asarray(v, float) for v in p1["ys"])
        self.expected = np.asarray(p1["expected"], float)
        self.H = np.asarray(p1["smoother_good"], float)
        self.H_short = np.asarray(p1["smoother_short"], float)
        self.sd = np.asarray(p1["band"], float)
        self.centre = int(p1["centre"])
        self.gp_mean = self.H @ self.pull_exp               # what the process takes of the pulls
        self.with_signal = np.asarray(p0["with_signal"], float)
        self.backgrounds = np.asarray(p1["backgrounds"], float)
        self.signal = np.asarray(p1["signal"], float)
        self.ells = np.asarray(p2["x"], float)
        self.spurious, self.share = (np.asarray(v, float) for v in p2["ys"])
        self.spurious_exact, self.share_exact = (np.asarray(v, float) for v in p2["exact"])
        self.scan_x = np.asarray(p3["x"], float)
        err = np.asarray(p3["error"], float)
        self.scan = np.asarray(p3["ys"], float) / err       # in units of the window's error
        self.scan_exact = np.asarray(p3["exact"], float) / err
        self.toys = f"{m['toys']:,}"
        self.good, self.short = m["ell_good"], m["ell_short"]
        self.stiff = int(np.abs(self.spurious).argmax())

    def subtitle_for(self, k):
        m = self.meta
        e, g, sh = m["exponential"], m["gp_good"], m["gp_short"]
        return [
            "a mass spectrum from a stated model: two falling processes, %s events in 80 "
            "bins. The usual background model is one fixed function, fitted to the "
            "counts; here an exponential, with two parameters"
            % f"{round(m['total_events'], -3):,.0f}",
            "the exponential cannot bend as the truth does, so it leaves a wave. With no "
            "signal anywhere, a window at 300 GeV averages %s events against the fit; "
            "near 470 GeV, %.1f errors too many: a spurious signal"
            % (signed(e["spurious"]), self.scan_exact[0].max()),
            "a Gaussian process assumes only smoothness: bins closer than its length "
            "scale ℓ = %g GeV move together. It follows the wave, with a band: the "
            "spurious signal at 300 GeV goes from %s to %s events"
            % (self.good, signed(e["spurious"]), signed(g["spurious"], 1)),
            "the catch: now a resonance is injected, %d events of width %g GeV. A process "
            "with ℓ = %g GeV is free to follow the bump as well, and takes it into the "
            "background; with ℓ = %g GeV it takes only part"
            % (S.GP_S, S.GP_SIGMA, self.short, self.good),
            "over %s spectra the process keeps %s of the signal at ℓ = %g GeV and %s at "
            "%g; the exponential keeps %s. Stiffer, at %g GeV, it cannot follow the "
            "background either: %s spurious events"
            % (self.toys, pct(sh["share"]), self.short, pct(g["share"]), self.good,
               pct(e["share"]), self.ells[self.stiff], signed(self.spurious[self.stiff])),
        ][k]

    # ------------------------------------------------------------------ shared charts
    def _mass_ticks(self, lo=100, hi=500, step=100):
        return [(v, "%d" % v) for v in range(lo, hi + 1, step)]

    def _pull_chart(self, box):
        top = float(np.ceil(max(self.pull_exp.max(), self.expected.max())))
        low = float(np.floor(min(self.pull_exp.min(), self.expected.min())))
        return LossChart(box, (100.0, 500.0), (low, top))

    def _scan_chart(self, sc, box, op):
        lim = float(np.ceil(np.abs(self.scan_exact).max()))
        chart = LossChart(box, (100.0, 500.0), (-lim, lim))
        self.axes(sc, chart, op, "window centre (GeV)", "", xticks=self._mass_ticks(200, 400, 100),
                  yticks=[(v, signed(v) if v else "0") for v in (-lim, -lim / 2, 0, lim / 2,
                                                                 lim)])
        self.level(sc, chart, 0.0, AXIS, 0.8 * op, 1.2)
        (x0, x1), (y0, _y1) = box
        self.text(sc, (x0 + x1) / 2, y0 - 0.085, "events above background in a 20 GeV", op,
                  size=0.030)
        self.text(sc, (x0 + x1) / 2, y0 - 0.045, "window ÷ the window's error, no signal", op,
                  size=0.030)
        return chart

    # ------------------------------------------------------------------ phase 1
    def phase_1(self, sc, u, op):
        self.head(sc, "a falling mass spectrum: truth, pseudo-data, and one fixed function",
                  op)
        chart = LossChart(((0.105, 0.60), (0.40, 0.80)), (100.0, 500.0), (4.0, 15000.0),
                          ylog=True)
        self.axes(sc, chart, op, "mass m (GeV)", "events per 5 GeV bin",
                  xticks=self._mass_ticks(),
                  yticks=[(10, "10"), (100, "100"), (1000, "1,000"), (10000, "10,000")])
        self.trace(sc, chart, self.m, self.truth, GREY, op, smoothstep(0.04, 0.26, u), 3.0,
                   dots=False)
        self.spots(sc, chart, self.m, self.data, dark(GREY, 0.55), op,
                   smoothstep(0.28, 0.52, u), 2.6)
        fit = op * smoothstep(0.56, 0.62, u)
        self.trace(sc, chart, self.m, self.fit, self.EXP, fit, smoothstep(0.58, 0.82, u), 2.8,
                   dots=False)
        (n1, t1), (n2, t2) = S.GP_TRUTH
        x = 0.655
        self.lines(sc, x, 0.40, [("the truth: two processes", dark(GREY, 0.5)),
                                 ("  %d · exp(−(m − 100) / %d)" % (n1, t1), dark(GREY, 0.5)),
                                 ("+ %d · exp(−(m − 100) / %d)" % (n2, t2), dark(GREY, 0.5))],
                   op * smoothstep(0.04, 0.14, u), size=MID, step=0.046)
        self.lines(sc, x, 0.565, ["pseudo-data: Poisson counts", "about the truth, seed %d"
                                  % self.meta["seed"]],
                   op * smoothstep(0.28, 0.38, u), size=MID, step=0.046)
        self.lines(sc, x, 0.685, [("the fixed function:", dark(self.EXP, 0.3)),
                                  ("A · exp(−k · m), fitted by", dark(self.EXP, 0.3)),
                                  ("maximum likelihood", dark(self.EXP, 0.3))],
                   fit, size=MID, step=0.046)
        self.tag(sc, PSEUDO, op)

    # ------------------------------------------------------------------ phase 2
    def _pulls(self, sc, chart, op, go=1.0, ink=None):
        self.level(sc, chart, 0.0, AXIS, 0.8 * op, 1.2)
        self.spots(sc, chart, self.m, self.pull_exp, soft(self.EXP, 0.25) if ink is None else ink,
                   op, go, 2.8)

    def phase_2(self, sc, u, op):
        self.head(sc, "what the fit leaves in each bin, in units of the bin's error: the pull",
                  op)
        c = self.centre
        n, f = self.data[c], self.fit[c]
        self.text(sc, 0.31, 0.385, "bin at %.1f GeV:  N = %d,  fit = %.1f" % (self.m[c], n, f),
                  op, size=MID)
        self.text(sc, 0.31, 0.428, "pull = (N − fit) / √fit = %s" % signed((n - f) / np.sqrt(f),
                                                                          2), op, size=MID,
                  strong=True)
        chart = self._pull_chart(((0.09, 0.53), (0.475, 0.77)))
        lo, hi = chart.y_range
        self.axes(sc, chart, op, "mass m (GeV)", "", xticks=self._mass_ticks(),
                  yticks=[(v, signed(v) if v else "0") for v in (-8.0, -4.0, 0.0, 4.0, 8.0)
                          if lo <= v <= hi])
        self._pulls(sc, chart, op, smoothstep(0.05, 0.35, u))
        ring = op * smoothstep(0.10, 0.2, u) * (1 - smoothstep(0.5, 0.6, u))
        self.disk(sc, self.P(chart.fx(self.m[c]), chart.fy(self.pull_exp[c])), 7.5, AMBER, ring,
                  lift=1.0, outline=0.0)
        wave = op * smoothstep(0.38, 0.44, u)
        self.trace(sc, chart, self.m, self.expected, self.EXP, wave, smoothstep(0.40, 0.62, u),
                   3.0, dots=False)
        self.text(sc, 0.31, 0.885, "line: the pull expected with no fluctuation", wave,
                  size=0.030, ink=dark(self.EXP, 0.3))
        on = op * smoothstep(0.62, 0.70, u)
        scan = self._scan_chart(sc, ((0.65, 0.93), (0.49, 0.80)), on)
        self.trace(sc, scan, self.scan_x, self.scan_exact[0], self.EXP, on,
                   smoothstep(0.66, 0.88, u), 3.0, dots=False)
        self.spots(sc, scan, self.scan_x, self.scan[0], dark(self.EXP, 0.25), on,
                   smoothstep(0.66, 0.88, u), 3.0)
        self.tag(sc, "pseudo-data; lines exact; right: %s pseudo-experiments" % self.toys, op)

    # ------------------------------------------------------------------ phase 3
    def phase_3(self, sc, u, op):
        self.head(sc, "a Gaussian process: each bin's value is a weighted average of the "
                  "pulls around it", op)
        chart = self._pull_chart(((0.09, 0.53), (0.405, 0.62)))
        lo, hi = chart.y_range
        self.axes(sc, chart, op, "", "pull", xticks=[],
                  yticks=[(v, signed(v) if v else "0") for v in (-4.0, 0.0, 4.0) if lo <= v <= hi])
        self._pulls(sc, chart, op)
        go = smoothstep(0.08, 0.60, u)
        k = int(round(go * (len(self.m) - 1)))
        self.columns(sc, chart, self.m, self.gp_mean - self.sd, self.gp_mean + self.sd,
                     soft(self.GP, 0.55), 0.8 * op, 4.2, go)
        self.trace(sc, chart, self.m, self.gp_mean, self.GP, op, go, 3.0, dots=False)
        weights = LossChart(((0.09, 0.53), (0.69, 0.80)), (100.0, 500.0),
                            (0.0, float(self.H.max()) * 1.05))
        self.axes(sc, weights, op, "mass m (GeV)", "weights, ℓ = %g GeV" % self.good,
                  xticks=self._mass_ticks(), yticks=[], grid=False)
        self.columns(sc, weights, self.m, 0.0, np.maximum(self.H[k], 0.0), self.GP, op, 3.6)
        moving = op * (1 - smoothstep(0.62, 0.68, u))
        self.upright(sc, chart, self.m[k], AMBER, 0.8 * moving)
        self.disk(sc, self.P(chart.fx(self.m[k]), chart.fy(self.gp_mean[k])), 5.5, self.GP,
                  moving, glow=0.4, lift=1.5, outline=0.0)
        on = op * smoothstep(0.64, 0.72, u)
        scan = self._scan_chart(sc, ((0.65, 0.93), (0.49, 0.80)), op)
        self.trace(sc, scan, self.scan_x, self.scan_exact[0], soft(self.EXP, 0.35), op, 1.0, 2.4,
                   dots=False)
        self.trace(sc, scan, self.scan_x, self.scan_exact[1], self.GP, on,
                   smoothstep(0.66, 0.88, u), 3.0, dots=False)
        self.spots(sc, scan, self.scan_x, self.scan[1], dark(self.GP, 0.25), on,
                   smoothstep(0.66, 0.88, u), 3.0)
        self.tag(sc, "pseudo-data; lines exact; right: %s pseudo-experiments" % self.toys, op)

    # ------------------------------------------------------------------ phase 4
    def phase_4(self, sc, u, op):
        self.head(sc, "inject a resonance: %d events at %g GeV. What does each background "
                  "leave?" % (S.GP_S, S.GP_M0), op)
        near = np.abs(self.m - S.GP_M0) <= 57.5
        x = self.m[near]
        left = self.with_signal[None, near] - self.backgrounds[:, near]
        top = float(max(left.max(), self.signal.max())) * 1.08
        low = float(left.min()) * 1.08
        d = self.meta["drawn"]
        cols = (("exponential fit", self.EXP, d["exponential"], None),
                ("process, ℓ = %g GeV" % self.good, self.GP, d["good"], self.H),
                ("process, ℓ = %g GeV" % self.short, self.SHORT, d["short"], self.H_short))
        wmax = float(self.H_short[self.centre].max())
        for j, (name, ink, kept, H) in enumerate(cols):
            x0 = 0.085 + 0.305 * j
            on = op * smoothstep(0.04 + 0.2 * j, 0.12 + 0.2 * j, u)
            go = smoothstep(0.06 + 0.2 * j, 0.24 + 0.2 * j, u)
            self.text(sc, x0 + 0.12, 0.385, name, on, size=MID, ink=dark(ink, 0.3))
            chart = LossChart(((x0, x0 + 0.24), (0.42, 0.64)), (238.0, 362.0), (low, top))
            self.axes(sc, chart, on, "", "",
                      xticks=[(v, "%d" % v) for v in (250, 300, 350)],
                      yticks=[(v, signed(v) if v else "0") for v in (-30.0, 0.0, 30.0, 60.0)
                              if low <= v <= top] if j == 0 else [], grid=j == 0)
            self.level(sc, chart, 0.0, AXIS, 0.8 * on, 1.2)
            self.columns(sc, chart, x, 0.0, left[j], soft(ink, 0.2), on, 7.0, go)
            self.trace(sc, chart, x, self.signal[near], self.SIGNAL, on, 1.0, 2.6, dots=False)
            self.text(sc, x0 + 0.12, 0.715, "left in the window: %s" % signed(kept), on * go,
                      size=MID, strong=True)
            if H is None:
                self.lines(sc, x0 + 0.12, 0.775, ["two parameters:", "it cannot bend to a bump"],
                           on, size=0.030, anchor="mm")
            else:
                w = LossChart(((x0, x0 + 0.24), (0.755, 0.835)), (238.0, 362.0), (0.0, wmax))
                self.columns(sc, w, x, 0.0, np.maximum(H[self.centre, near], 0.0), ink, on, 7.0)
                self.level(sc, w, 0.0, AXIS, 0.8 * on, 1.2)
        self.text(sc, 0.05, 0.878, "bars: events minus background.  Green: the resonance.  "
                  "Bottom: each process's weights for the bin at 300 GeV",
                  op * smoothstep(0.3, 0.4, u), size=0.030, anchor="lm")
        self.tag(sc, PSEUDO.split(":")[0], op)

    # ------------------------------------------------------------------ phase 5
    def phase_5(self, sc, u, op):
        self.head(sc, "%s spectra at each of %d length scales ℓ" % (self.toys, len(self.ells)),
                  op)
        e = self.meta["exponential"]
        ticks = [(v, "%g" % v) for v in (5, 10, 20, 40, 80, 160, 320)]
        span = (float(self.ells[0]) * 0.9, float(self.ells[-1]) * 1.1)
        lim = float(np.ceil(np.abs(np.r_[self.spurious, e["spurious"]]).max() / 10) * 10)
        boxes = (((0.10, 0.47), (0.43, 0.77)), ((0.585, 0.945), (0.43, 0.77)))
        names = ("spurious signal at 300 GeV (events)", "share of the injected signal kept")
        ranges = ((-lim, lim), (0.0, 1.0))
        yticks = ([(v, signed(v) if v else "0") for v in (-lim, -lim / 2, 0, lim / 2, lim)],
                  [(v, pct(v)) for v in (0, 0.25, 0.5, 0.75, 1.0)])
        rows = ((self.spurious, self.spurious_exact, e["spurious"]),
                (self.share, self.share_exact, e["share"]))
        for j in range(2):
            on = op * smoothstep(0.04 + 0.34 * j, 0.12 + 0.34 * j, u)
            go = smoothstep(0.08 + 0.34 * j, 0.36 + 0.34 * j, u)
            (x0, x1), (y0, _y1) = boxes[j]
            chart = LossChart(boxes[j], span, ranges[j], xlog=True)
            self.axes(sc, chart, on, "length scale ℓ (GeV)", "", xticks=ticks, yticks=yticks[j])
            self.text(sc, (x0 + x1) / 2, y0 - 0.04, names[j], on, size=MID, strong=True)
            if j == 0:
                self.level(sc, chart, 0.0, AXIS, 0.8 * on, 1.2)
            toys, exact, fixed = rows[j]
            self.level(sc, chart, fixed, self.EXP, on, 2.6)
            self.upright(sc, chart, self.good, soft(self.GP, 0.5), 0.9 * on, 1.6)
            self.trace(sc, chart, self.ells, exact, self.GP, on, go, 2.8, dots=False)
            n = int(np.ceil(len(self.ells) * go - 1e-9))
            for k in range(n):
                self.disk(sc, self.P(chart.fx(self.ells[k]), chart.fy(toys[k])), 4.6,
                          dark(self.GP, 0.2), on, lift=1.4, outline=0.0)
        self.key(sc, 0.115, 0.472, [("exponential fit", self.EXP)], op, size=0.030)
        self.key(sc, 0.115, 0.557, [("Gaussian process", self.GP)], op, size=0.030)
        self.text(sc, 0.05, 0.882, "thin line: ℓ = %g GeV, the length scale with the largest "
                  "evidence; the one drawn before" % self.good, op,
                  size=0.030, anchor="lm")
        self.tag(sc, (TOYS % self.toys) + "; lines exact", op)


ANIMATIONS = {
    "autoencoder-anomaly-detection": AnomalyDetection,
    "gan-mode-collapse": ModeCollapse,
    "gp-background-fit": GPBackground,
}

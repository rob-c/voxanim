"""An anomaly search with two scores: animations 157-158 of the "next" catalogue (see
:mod:`voxanim.mech`).

* :class:`LossAndLatent`: one picture through an autoencoder gives two scores - how badly
  it is rebuilt (loss space) and how far its code lies from the training codes (latent
  space) - and an anomaly can show in both, in one or in neither;
* :class:`TwoScoreFlow`: the whole procedure, from training on normal data to a decision:
  a density fitted to the codes, thresholds from normal data, and three ways of combining
  the two scores.

Both draw the experiments of :mod:`voxanim.study_anomaly`, which records every array
shown here: nothing is computed in this module but layout. They continue
``autoencoder-anomaly-detection`` (:mod:`voxanim.mech_phys_adv`) and keep its colours:
blue for the kept classes, red for the removed one. Every phase says, bottom right, what
kind of evidence it shows.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import study_anomaly as S
from .easing import smoothstep
from .losschart import LossChart
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    GREEN,
    MAGENTA,
    PAPER,
    RED,
    TEAL,
    TRACK,
    VIOLET,
    dark,
    soft,
)
from .mech_phys_adv import BIG, GREY, MID, MNIST_TAG, PhysAdv, pct

SMALL = 0.030
TAG_STARTS = ("measured on MNIST",)
NORMAL_TAG = "measured on MNIST · thresholds from normal digits only"
FIT_TAG = "measured on MNIST · fits and thresholds use normal digits only"


def bump(w, a=0.10, b=0.92):
    """Opacity of something shown for one stage: in at its start, out at its end."""
    return smoothstep(0.0, a, w) * (1.0 - smoothstep(b, 1.0, w))


class Anomaly(PhysAdv):
    """What the two animations share: the colours, histograms of two groups, a row of
    stacked shares."""

    SEEN, UNSEEN, SPOILED = BLUE, RED, MAGENTA
    LOSS, CODE, BOTH = AMBER, TEAL, VIOLET

    def hist(self, sc, box, edges, hists, inks, op, go=1.0, xlabel=""):
        """Histograms of several groups over the same bins; returns the chart."""
        e = np.asarray(edges, float)
        mids = (e[:-1] + e[1:]) / 2
        chart = LossChart(box, (float(e[0]), float(e[-1])),
                          (0.0, float(np.max(hists)) * 1.08))
        self.axes(sc, chart, op, xlabel, "share of digits", yticks=[], grid=False)
        wpx = self.span((box[0][1] - box[0][0]) / len(mids))[0]
        for n, (h, ink) in enumerate(zip(hists, inks)):
            self.columns(sc, chart, mids, 0.0, h, soft(ink, 0.15) if n == 0 else ink, op,
                         wpx * (0.86 if n == 0 else 0.42), go)
        return chart

    def stack(self, sc, fx, fy, length, shares, inks, op, go=1.0, thick=18.0):
        """Shares as one bar growing right from ``(fx, fy)``; ``length`` px is 100%."""
        at = self.P(float(fx), float(fy))
        self.rect(sc, self.shift(at, length / 2, 0), length, thick, TRACK, 0.5 * op, radius=3,
                  lift=0.3)
        x = 0.0
        for share, ink in zip(shares, inks):
            w = float(share) * length * float(go)
            if w > 0.6:
                self.rect(sc, self.shift(at, x + w / 2, 0), w, thick, ink, op,
                          radius=min(3.0, w / 2), lift=0.5)
            x += w


# ====================================================================== 157: two spaces
class LossAndLatent(Anomaly):
    """One held-out digit through the autoencoder and its two scores; every held-out
    digit in the two spaces of a 2-number code, where the unseen class lands among the
    training codes; the two scores' histograms at 8 numbers; a real digit for each of the
    four cases; the two scores for every removed class; and eight ways of spoiling a
    digit, by where they are caught."""

    TITLE = "Two places an anomaly can show"
    PHASES = (("scores", 10.0), ("spaces", 9.5), ("8 numbers", 10.0), ("cases", 11.0),
              ("classes", 10.0), ("spoiled", 9.0))
    CASE_INKS = (VIOLET, AMBER, TEAL)

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = self.panels = [p.data for p in s.panels]
        self.left = int(m["featured"])
        self.drawn, self.main = (int(v) for v in m["codes"])
        self.pics = np.asarray(p0["values"], float)
        self.cloud = np.asarray(p1["xy"], float)[:600]
        self.unseen = np.asarray(p1["unseen"], float)[:220]
        self.inverted = np.asarray(p1["inverted"], float)[:120]
        self.quad = np.asarray(p2["values"], float)               # [kind, case]
        self.by_class = np.asarray(p2["by_class"], float).mean(1)  # [class, case]
        self.auc = np.asarray(p3["ys"], float)                    # [score, class]
        self.auc_seeds = np.asarray(p3["seeds"], float)

    def subtitle_for(self, k):
        m, left = self.meta, self.left
        _p0, p1, p2, _ = self.panels
        by = m["auc_by_class"][str(left)]
        q = m["quadrants"]
        f2, f8 = np.asarray(p1["flagged"]), np.asarray(p2["flagged"])
        rescued = by["knn"]["mean"] > 0.5
        half = (q["lower half erased"]["code only"] + q["lower half erased"]["both"],
                q["lower half erased"]["loss only"] + q["lower half erased"]["both"])
        return [
            "an autoencoder trained on real digits with every %d removed. Each picture gets "
            "two scores: how badly it is rebuilt (loss space), and how far its code lies "
            "from the codes of the training digits (latent space)" % left,
            "with a code of %d numbers the unseen %ds land among the training codes: %s of "
            "them are above the loss threshold and %s above the distance threshold. "
            "Neither space shows them" % (self.drawn, left, pct(f2[0, 1]), pct(f2[1, 1])),
            "with %d numbers the unseen %ds are still rebuilt better than kept digits (area "
            "%.2f), but their codes lie a little apart: area %.2f for the distance, and %s "
            "of them above its threshold"
            % (self.main, left, p2["run_auc"][0], p2["run_auc"][1], pct(f8[1, 1])),
            "two cuts, each passing 95%% of normal digits, make four cases. Inverted digits "
            "are caught by both (%s), noisy ones by the loss alone (%s); dimmed ones "
            "(%s) and most unseen %ds (%s) by neither"
            % (pct(q["inverted"]["both"]), pct(q["pixel noise"]["loss only"]),
               pct(q["dimmed to 40%"]["neither"]), left,
               pct(m["quadrants_featured"]["neither"]["mean"])),
            "over ten removed classes and three seeds: the loss has area %.2f, the distance "
            "%.2f. For the %ds it is %.2f against %.2f: the code %s what the loss misses, "
            "though at 5%% false alarms it catches only %s of them"
            % (m["auc_loss"]["mean"], m["auc_knn"]["mean"], left, by["loss"]["mean"],
               by["knn"]["mean"], "sees" if rescued else "does not see",
               pct(m["quadrants_featured"]["code only"]["mean"])),
            "spoiled digits mostly live in loss space: noise, a shift, a turn. Erasing half "
            "a digit shows %s in the code (%s) as in the loss (%s). Dimmed and averaged "
            "digits show in neither"
            % ("more often" if half[0] > half[1] + 0.05 else "about as often", pct(half[0]),
               pct(half[1])),
        ][k]

    # ------------------------------------------------------------------ phase 1
    def phase_1(self, sc, u, op):
        p0 = self.panels[0]
        self.head(sc, "one held-out digit in: a picture comes back, and on the way a code",
                  op)
        stage = int(min(u * 3, 2))
        w = u * 3 - stage
        on = op * bump(w)
        ink = (self.SEEN, self.UNSEEN, self.SPOILED)[stage]
        who = ("a kept class (an %d)" % int(p0["classes"][0]),
               "the removed class (a %d)" % self.left, "a kept digit, inverted")[stage]
        y = 0.475
        self.digit(sc, self.pics[2 * stage], 0.12, y, 92, on, frame=ink)
        self.text(sc, 0.12, 0.60, who, on, size=SMALL, ink=dark(ink, 0.3))
        boxes = ((0.275, "encoder", 92, PAPER, AXIS),
                 (0.455, "code", 150, soft(self.CODE), self.CODE),
                 (0.635, "decoder", 92, PAPER, AXIS))
        prev = 0.12 + self.frac(56)[0]
        for n, (x, name, wpx, fill, border) in enumerate(boxes):
            a, b = (prev, y), (x - self.frac(wpx / 2 + 6)[0], y)
            self.arrow(sc, a, b, AXIS, op, 1.0, 2.4)
            self.pulse(sc, a, b, (w - 0.08 - 0.10 * n) / 0.12, ink, op)
            self.box(sc, x, y, wpx, 48, fill, op, name if n != 1 else None, border=border,
                     size=MID)
            prev = x + self.frac(wpx / 2 + 6)[0]
        a, b = (prev, y), (0.79 - self.frac(56)[0], y)
        self.arrow(sc, a, b, AXIS, op, 1.0, 2.4)
        self.pulse(sc, a, b, (w - 0.38) / 0.12, ink, op)
        code = np.asarray(p0["codes"], float)[stage]
        shown = on * smoothstep(0.20, 0.30, w)
        self.text(sc, 0.455, y, ("(%.2f, %.2f)" % tuple(code)).replace("-", "−"), shown,
                  size=MID, strong=True)
        self.text(sc, 0.455, y - 0.075, "the code: %d numbers" % self.drawn, op, size=SMALL,
                  ink=dark(self.CODE, 0.3))
        back = on * smoothstep(0.48, 0.58, w)
        self.digit(sc, self.pics[2 * stage + 1], 0.79, y, 92, back, frame=ink)
        self.text(sc, 0.79, 0.60, "rebuilt", back, size=SMALL)
        self.text(sc, 0.08, 0.675, "A  loss space: the mean of (out − in)² over the 784 "
                  "pixels", op, anchor="lm", size=MID, ink=dark(self.LOSS, 0.45))
        self.text(sc, 0.92, 0.675, "= %.3f" % p0["errors"][stage], back, anchor="rm",
                  size=BIG, strong=True)
        self.text(sc, 0.08, 0.735, "B  latent space: distance from the code to its %dth "
                  "nearest training code" % S.KNN, op, anchor="lm", size=MID,
                  ink=dark(self.CODE, 0.3))
        self.text(sc, 0.92, 0.735, "= %.3f" % p0["knn"][stage], shown, anchor="rm", size=BIG,
                  strong=True)
        note = op * smoothstep(0.10, 0.20, u)
        self.text(sc, 0.5, 0.815, "real digits stand in for events, the removed %ds (or a "
                  "spoiled digit) for new physics:" % self.left, note, size=SMALL)
        self.text(sc, 0.5, 0.86, "the counterpart is a model-independent search, trained "
                  "on known processes only", note, size=SMALL)
        self.tag(sc, MNIST_TAG, op)

    # ------------------------------------------------------------------ phase 2
    def phase_2(self, sc, u, op):
        p1 = self.panels[1]
        left = self.left
        self.head(sc, "every held-out digit in the two spaces, for a code of %d numbers"
                  % self.drawn, op)
        thr = np.asarray(p1["thresholds"], float)
        flagged = np.asarray(p1["flagged"], float)
        h = np.asarray(p1["loss_hist"], float)
        chart = self.hist(sc, ((0.08, 0.44), (0.43, 0.73)), p1["loss_edges"], h[:2],
                          (self.SEEN, self.UNSEEN), op, smoothstep(0.02, 0.14, u),
                          "A  reconstruction error")
        cutop = op * smoothstep(0.16, 0.24, u)
        self.upright(sc, chart, thr[0], dark(self.LOSS, 0.1), cutop, 3.0)
        self.text(sc, 0.44, 0.385, "loss space", op, size=MID, ink=dark(self.LOSS, 0.45),
                  anchor="rm")
        self.lines(sc, 0.26, 0.84, [
            "above the threshold: %s of unseen %ds," % (pct(flagged[0, 1]), left),
            "%s of inverted digits" % pct(flagged[0, 2])], cutop, size=SMALL, step=0.042,
            anchor="mm")
        box = np.asarray(p1["box"], float)
        cl = LossChart(((0.58, 0.92), (0.43, 0.73)), (box[0], box[1]), (box[2], box[3]))
        self.axes(sc, cl, op, "code number 1", "code number 2", xticks=[], yticks=[],
                  grid=False)
        self.text(sc, 0.92, 0.385, "latent space", op, size=MID, ink=dark(self.CODE, 0.3),
                  anchor="rm")
        self.spots(sc, cl, self.cloud[:, 0], self.cloud[:, 1], soft(self.SEEN, 0.35), op,
                   smoothstep(0.02, 0.20, u), r=2.0)
        self.spots(sc, cl, self.unseen[:, 0], self.unseen[:, 1], self.UNSEEN, op,
                   smoothstep(0.26, 0.50, u), r=2.8)
        self.spots(sc, cl, self.inverted[:, 0], self.inverted[:, 1], self.SPOILED, op,
                   smoothstep(0.52, 0.70, u), r=2.8)
        self.lines(sc, 0.75, 0.84, [
            "farther than %.2f from the %dth nearest code:" % (thr[1], S.KNN),
            "%s of unseen %ds, %s of inverted digits"
            % (pct(flagged[1, 1]), left, pct(flagged[1, 2]))],
            op * smoothstep(0.52, 0.62, u), size=SMALL, step=0.042, anchor="mm")
        self.key(sc, 0.465, 0.47, [("kept", soft(self.SEEN, 0.2)),
                                   ("unseen %ds" % left, self.UNSEEN),
                                   ("inverted", self.SPOILED)], op, size=SMALL)
        self.tag(sc, NORMAL_TAG, op)

    # ------------------------------------------------------------------ phase 3
    def phase_3(self, sc, u, op):
        p2 = self.panels[2]
        left = self.left
        self.head(sc, "the same two scores with a code of %d numbers (it cannot be drawn: "
                  "its scores can)" % self.main, op)
        edges, hist = np.asarray(p2["edges"], float), np.asarray(p2["hist"], float)
        thr, flagged = np.asarray(p2["thresholds"], float), np.asarray(p2["flagged"], float)
        med, area = np.asarray(p2["medians"], float), np.asarray(p2["run_auc"], float)
        specs = ((0.08, "A  reconstruction error", self.LOSS, "%.3f", 0.02),
                 (0.56, "B  distance to the %dth nearest training code" % S.KNN, self.CODE,
                  "%.2f", 0.36))
        for k, (x0, name, ink, fmt, t0) in enumerate(specs):
            chart = self.hist(sc, ((x0, x0 + 0.36), (0.42, 0.66)), edges[k], hist[k],
                              (self.SEEN, self.UNSEEN), op, smoothstep(t0, t0 + 0.14, u), name)
            on = op * smoothstep(t0 + 0.16, t0 + 0.24, u)
            self.upright(sc, chart, thr[k], dark(ink, 0.1), on, 3.0)
            self.lines(sc, x0 + 0.18, 0.775, [
                "typical kept digit " + fmt % med[k, 0] + ", typical unseen %d " % left
                + fmt % med[k, 1],
                "above the threshold: %s of unseen %ds  ·  area %.2f"
                % (pct(flagged[k, 1]), left, area[k])], on, size=SMALL, step=0.044,
                anchor="mm")
        self.key(sc, 0.47, 0.44, [("kept", self.SEEN), ("unseen", self.UNSEEN)], op,
                 size=SMALL)
        self.text(sc, 0.5, 0.875, "each threshold: the value 95% of held-out kept digits "
                  "stay under", op * smoothstep(0.70, 0.80, u), size=SMALL)
        self.tag(sc, NORMAL_TAG, op)

    # ------------------------------------------------------------------ phase 4
    def phase_4(self, sc, u, op):
        p2 = self.panels[2]
        left = self.left
        thr = np.asarray(p2["thresholds"], float)
        self.head(sc, "loss above %.3f?  distance above %.2f?  Four cases, each a real digit"
                  % (thr[0], thr[1]), op)
        pics = [np.asarray(v, float) for v in p2["examples"]]
        names = list(p2["example_names"])
        scores = [tuple(v) for v in np.asarray(p2["example_scores"], float)]
        shares = [float(v) for v in p2["example_share"]]
        if p2["own_found"]:                           # the removed class's own "code only"
            pics[2], names[2] = np.asarray(p2["own"], float), "unseen %d" % left
            scores[2] = tuple(np.asarray(p2["own_scores"], float))
            shares[2] = float(np.asarray(p2["run"], float)[0, 2])
        titles = ("caught by both", "by the loss only", "by the code only", "by neither")
        inks = (self.BOTH, self.LOSS, self.CODE, GREY)
        for q in range(4):
            x = 0.155 + 0.23 * q
            on = op * smoothstep(0.04 + 0.16 * q, 0.14 + 0.16 * q, u)
            self.text(sc, x, 0.39, titles[q], on, size=MID, ink=dark(inks[q], 0.35))
            self.digit(sc, pics[q], x, 0.505, 84, on, frame=inks[q])
            a, b = scores[q]
            self.lines(sc, x, 0.63, [
                names[q],
                ("loss %.3f %s" % (a, "> cut" if a > thr[0] else "< cut"),
                 dark(self.LOSS, 0.45)),
                ("distance %.2f %s" % (b, "> cut" if b > thr[1] else "< cut"),
                 dark(self.CODE, 0.3)),
                "%s of its kind here" % pct(shares[q])], on, size=SMALL, step=0.044, anchor="mm")
        self.text(sc, 0.5, 0.855, "over three seeds, %s of the unseen %ds are caught by "
                  "neither" % (pct(self.meta["quadrants_featured"]["neither"]["mean"]), left),
                  op * smoothstep(0.72, 0.82, u), size=MID, strong=True)
        self.tag(sc, NORMAL_TAG, op)

    # ------------------------------------------------------------------ phase 5
    def phase_5(self, sc, u, op):
        m, left = self.meta, self.left
        self.head(sc, "each class removed in turn, three seeds: the two scores alone, and "
                  "what two cuts catch", op)
        ticks = [(c, str(c)) for c in range(10)]
        top = LossChart(((0.09, 0.60), (0.385, 0.60)), (-0.6, 9.6), (0.0, 1.0))
        self.axes(sc, top, op, "", "area under the ROC curve", xticks=[],
                  yticks=[(0.0, "0"), (0.5, "0.5"), (1.0, "1")])
        top.hline(sc, self.P, 0.5, op, dashes=24, width=1.4)
        inks = (self.LOSS, self.CODE, GREY)
        for k in (2, 0, 1):
            go = smoothstep(0.04 + 0.12 * k, 0.24 + 0.12 * k, u)
            self.trace(sc, top, range(10), self.auc[k], inks[k], op, go, 2.6, r=4.0)
        low = LossChart(((0.09, 0.60), (0.67, 0.83)), (-0.6, 9.6), (0.0, 1.0))
        self.axes(sc, low, op, "the class removed", "share caught at 5% false alarms each",
                  xticks=ticks, yticks=[(0.0, "0"), (1.0, "100%")])
        grow = smoothstep(0.46, 0.66, u)
        for c in range(10):
            base = 0.0
            for q in (1, 0, 2):                       # loss only, both, code only
                v = self.by_class[c, q] * grow
                self.columns(sc, low, [c], [base], [base + v], self.CASE_INKS[q], op, 24.0)
                base += v
        self.key(sc, 0.655, 0.40, [("A  the loss", self.LOSS),
                                   ("B  distance to %dth code" % S.KNN, self.CODE),
                                   ("Mahalanobis distance", GREY)], op, size=SMALL)
        by = m["auc_by_class"][str(left)]
        note = op * smoothstep(0.40, 0.50, u)
        self.lines(sc, 0.64, 0.565, [
            "mean area: %.2f, %.2f, %.2f" % (m["auc_loss"]["mean"], m["auc_knn"]["mean"],
                                           m["auc_gauss"]["mean"]),
            "the %ds: %.2f, %.2f, %.2f" % (left, by["loss"]["mean"], by["knn"]["mean"],
                                         by["gauss"]["mean"])], note, size=SMALL,
            step=0.044)
        self.key(sc, 0.655, 0.69, [("caught by the loss only", self.LOSS),
                                   ("by both", self.BOTH), ("by the code only", self.CODE)],
                 op * grow, size=SMALL)
        self.lines(sc, 0.64, 0.835, ["B was picked over Mahalanobis", "after seeing these areas"],
                   op * smoothstep(0.70, 0.80, u), size=SMALL, step=0.042)
        self.tag(sc, NORMAL_TAG, op)

    # ------------------------------------------------------------------ phase 6
    def phase_6(self, sc, u, op):
        p2 = self.panels[2]
        self.head(sc, "one kept digit spoiled eight ways: where is each kind caught?", op)
        names = list(p2["labels"])
        pics = np.asarray(p2["kind_pictures"], float)
        order = [0] + [1 + int(j) for j in np.argsort(-self.quad[1:, :3].sum(1), kind="stable")]
        length = self.span(0.40)[0]
        for row, j in enumerate(order):
            y = 0.385 + 0.056 * row
            go = smoothstep(0.04 + 0.07 * row, 0.16 + 0.07 * row, u)
            on = op * smoothstep(0.02 + 0.07 * row, 0.10 + 0.07 * row, u)
            if j:
                self.digit(sc, pics[j], 0.077, y, 29, on)
            name = "the unseen class (mean of ten)" if j == 0 else names[j]
            self.text(sc, 0.10, y, name, on, anchor="lm", size=SMALL,
                      ink=dark(self.UNSEEN, 0.3) if j == 0 else None)
            q = self.quad[j]
            self.stack(sc, 0.40, y, length, (q[1], q[0], q[2]), self.CASE_INKS[1:2]
                       + self.CASE_INKS[0:1] + self.CASE_INKS[2:3], on, go)
            self.text(sc, 0.815, y, "neither: %s" % pct(q[3]), on * go, anchor="lm",
                      size=SMALL)
        self.key(sc, 0.075, 0.925, [("loss only", self.LOSS)], op, size=SMALL)
        self.key(sc, 0.215, 0.925, [("both", self.BOTH)], op, size=SMALL)
        self.key(sc, 0.31, 0.925, [("code only", self.CODE)], op, size=SMALL)
        self.tag(sc, MNIST_TAG, op)


# ====================================================================== 158: the procedure
class TwoScoreFlow(Anomaly):
    """The procedure as five steps with one held-out event followed through them; three
    densities fitted to the same training codes; every held-out digit in the plane of
    the two scores with the three ways of cutting it; what each rule catches and what
    two cuts cost in false alarms; and the same rules at four code sizes."""

    TITLE = "An anomaly search with two scores"
    PHASES = (("procedure", 12.0), ("density", 12.0), ("plane", 12.0), ("rules", 12.0),
              ("code size", 10.0))
    INKS5 = (AMBER, TEAL, GREEN, VIOLET, MAGENTA)     # loss, code, either, sum, ellipse
    SHORT = ("loss alone", "code alone", "either cut", "sum", "ellipse")
    RANGE = (-3.5, 6.5)

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        _p0, p1, _p2, p3 = self.panels = [p.data for p in s.panels]
        self.left, self.main = int(m["featured"]), int(m["code_shown"])
        self.chosen = int(p1["chosen"])
        self.dens_name = ("the Gaussian", "the mixture", "the flow")[self.chosen]
        nll = np.asarray(p1["values"], float)
        self.maps = np.exp(-(nll - nll.min((1, 2), keepdims=True)))[:, ::-1]
        self.cloud = np.asarray(p1["cloud"], float)[:120]
        self.caught = np.asarray(p3["caught"], float)             # [alpha, class, seed, rule]
        self.a5 = [float(a) for a in p3["alphas"]].index(m["alphas"][-1])
        self.codes = [int(v) for v in p3["x"]]
        self.by_code = np.asarray(p3["ys"], float)                # [rule, code]

    def subtitle_for(self, k):
        m, left = self.meta, self.left
        p0, p1, _p2, _p3 = self.panels
        nll = np.asarray(p1["normal_nll"], float).mean((0, 1))
        area = np.asarray(p1["auc"], float).mean((0, 1))
        c5 = m["caught"][str(m["alphas"][-1])]
        raw = m["or_uncorrected"][str(m["alphas"][-1])]
        one = self.caught[self.a5, left].mean(0)
        run = np.asarray(self.panels[2]["caught"], float)
        codes, by = self.codes, self.by_code
        main, last = codes.index(self.main), len(codes) - 1
        better = by[1, main + 1:].min() > by[0, main + 1:].max()
        return [
            "train on normal digits only, fit a density p to their codes, score each new "
            "event twice, cut both scores where normal digits allow. A typical unseen %d "
            "(loss %.3f, −ln p %.1f) %s"
            % (left, p0["event"][0], p0["event"][1],
               "is flagged" if any(p0["decisions"]) else "passes every cut"),
            "three densities for the codes. Held-out normal codes have −ln p %.1f under one "
            "Gaussian, %.1f under a mixture, %.1f under a flow: %s is used. As anomaly "
            "scores their areas are %.2f, %.2f and %.2f"
            % (nll[0], nll[1], nll[2], self.dens_name, area[0], area[1], area[2]),
            "each digit is a point: its loss across, its code's −ln p up. Either cut flags "
            "the corner's outside, the sum a diagonal, the ellipse everything far from the "
            "normal cloud. Of these unseen %ds they catch %s, %s and %s"
            % (left, pct(run[2]), pct(run[3]), pct(run[4])),
            "at 5%% false alarms the loss alone catches %s of anomalies, either cut %s, the "
            "sum %s: no gain on average, but the %ds go from %s to %s. Two 5%% cuts flag "
            "%s of normal digits, so each is set at %s"
            % (pct(c5["loss"]["mean"]), pct(c5["either"]["mean"]), pct(c5["sum"]["mean"]),
               left, pct(one[0]), pct(one[2]), pct(raw["false_alarms"], 1),
               pct(raw["fair_rate"], 1)),
            "a roomy code rebuilds everything: the loss's area falls from %.2f at %d numbers "
            "to %.2f at %d. The code's density goes from %.2f to %.2f at %d (%.2f at %d): %s"
            % (by[0, main], self.main, by[0, last], codes[last], by[1, main],
               by[1, main + 1], codes[main + 1], by[1, last], codes[last],
               "the anomaly moves into the code" if better
               else "the code does not make up for the loss"),
        ][k]

    # ------------------------------------------------------------------ phase 1
    def phase_1(self, sc, u, op):
        p0 = self.panels[0]
        left = self.left
        counts = [int(v) for v in p0["counts"]]
        self.head(sc, "the procedure in five steps, followed by one held-out event", op)
        steps = (("1  train on", "normal digits only"), ("2  fit a density p", "to their codes"),
                 ("3  score a new", "event twice"), ("4  thresholds from", "normal digits"),
                 ("5  combine,", "decide"))
        now = int(min(u * 8.0, 4))
        for n, (a, b) in enumerate(steps):
            x = 0.14 + 0.18 * n
            on = n <= now
            ink = self.CODE if n == now else AXIS
            self.box(sc, x, 0.425, 148, 62, soft(self.CODE, 0.7) if n == now else PAPER,
                     op * (1.0 if on else 0.4), border=ink)
            self.lines(sc, x, 0.405, [a, b], op * (1.0 if on else 0.4), size=SMALL, step=0.04,
                       anchor="mm")
            if n:
                self.arrow(sc, (x - 0.18 + self.frac(76)[0], 0.425),
                           (x - self.frac(78)[0], 0.425), AXIS, op, 1.0, 2.2)
        t = [op * smoothstep(0.02 + 0.125 * n, 0.08 + 0.125 * n, u) for n in range(5)]
        self.lines(sc, 0.06, 0.545, [
            "%s training digits of nine classes;" % f"{counts[0]:,}",
            "no %d among them" % left], t[0], size=SMALL, step=0.04)
        self.text(sc, 0.06, 0.645, "p: %s on their codes"
                  % ("one Gaussian", "a mixture of Gaussians",
                     "a normalizing flow")[self.chosen], t[1], size=SMALL, anchor="lm")
        y = 0.755
        self.digit(sc, p0["values"][0], 0.105, y, 76, t[2], frame=self.UNSEEN)
        self.arrow(sc, (0.155, y), (0.195, y), AXIS, t[2], 1.0, 2.4)
        self.pulse(sc, (0.155, y), (0.195, y), (u - 0.27) / 0.08, self.UNSEEN, op)
        self.digit(sc, p0["values"][1], 0.245, y, 76, t[2], frame=self.UNSEEN)
        self.text(sc, 0.105, 0.86, "an unseen %d" % left, t[2], size=SMALL,
                  ink=dark(self.UNSEEN, 0.3))
        self.text(sc, 0.245, 0.86, "rebuilt", t[2], size=SMALL)
        event, cuts = np.asarray(p0["event"], float), np.asarray(p0["cuts"], float)
        rows = (("A  loss", "%.3f", self.LOSS, 0.45), ("B  −ln p(code)", "%.1f", self.CODE, 0.3))
        for k, (name, fmt, ink, d) in enumerate(rows):
            yy = 0.565 + 0.062 * k
            self.text(sc, 0.44, yy, name + " = " + fmt % event[k], t[2], anchor="lm", size=MID,
                      ink=dark(ink, d))
            over = event[k] > cuts[k]
            self.text(sc, 0.665, yy, ("> " if over else "< ") + "threshold " + fmt % cuts[k]
                      + (":  flagged" if over else ":  passes"), t[3], anchor="lm", size=MID,
                      strong=True)
        self.text(sc, 0.44, 0.70, "thresholds: what 95%% of %s held-out normal digits stay "
                  "under" % f"{counts[1]:,}", t[3], anchor="lm", size=SMALL)
        dec = [bool(v) for v in p0["decisions"]]
        self.lines(sc, 0.44, 0.775, [
            "either cut: %s  ·  sum: %s  ·  ellipse: %s"
            % tuple("flagged" if v else "passes" for v in dec[2:]),
            "real digits stand in for events, the unseen %ds for new physics:" % left,
            "a model-independent search, trained on known processes"], t[4], size=SMALL,
            step=0.042)
        self.tag(sc, FIT_TAG, op)

    # ------------------------------------------------------------------ phase 2
    def phase_2(self, sc, u, op):
        p1 = self.panels[1]
        self.head(sc, "step 2: three densities fitted to the same training codes (drawn "
                  "for a code of 2 numbers)", op)
        box = np.asarray(p1["box"], float)
        nll = np.asarray(p1["normal_nll"], float).mean((0, 1))
        area = np.asarray(p1["auc"], float).mean((0, 1))
        names = ("one Gaussian", "a mixture of %d Gaussians" % S.MIX,
                 "a normalizing flow")
        size = 160
        dw, dh = self.frac(size, size)
        for k in range(3):
            x, y = 0.19 + 0.31 * k, 0.565
            on = op * smoothstep(0.03 + 0.2 * k, 0.13 + 0.2 * k, u)
            self.text(sc, x, 0.368, names[k], on, size=MID, strong=k == self.chosen)
            self.grid(sc, self.maps[k], x, y, size, size, on, vmax=1.0, hot=self.CODE,
                      lines=0.0)
            fx = x - dw / 2 + (self.cloud[:, 0] - box[0]) / (box[1] - box[0]) * dw
            fy = y + dh / 2 - (self.cloud[:, 1] - box[2]) / (box[3] - box[2]) * dh
            for a, b in zip(fx, fy):
                self.disk(sc, self.P(float(a), float(b)), 1.5, dark(self.SEEN, 0.3), on,
                          lift=1.2, outline=0.0)
            num = op * smoothstep(0.62, 0.70, u)
            self.lines(sc, x, 0.77, [
                "−ln p of normal codes: %.2f" % nll[k],
                "area as a score: %.2f" % area[k]], num, size=SMALL, step=0.042, anchor="mm")
        best = int(p1["best_by_auc"])
        self.text(sc, 0.5, 0.865, "%s fits held-out normal codes best, so it is used; no "
                  "anomaly was looked at.  %s"
                  % (self.dens_name.capitalize(),
                     "It is also the best score." if best == self.chosen else
                     "The best score is %s." % ("the Gaussian", "the mixture",
                                                "the flow")[best]),
                  op * smoothstep(0.76, 0.86, u), size=SMALL)
        self.tag(sc, FIT_TAG, op)

    # ------------------------------------------------------------------ phase 3
    def _clip(self, a, b):
        """The part of segment ``a``-``b`` inside the plane's range (None if none)."""
        lo, hi = self.RANGE
        t0, t1 = 0.0, 1.0
        for k in (0, 1):
            d = b[k] - a[k]
            for edge, sign in ((lo, -1.0), (hi, 1.0)):
                p, q = sign * d, sign * (edge - a[k])
                if abs(p) < 1e-12:
                    if q < 0:
                        return None
                elif p < 0:
                    t0 = max(t0, q / p)
                else:
                    t1 = min(t1, q / p)
        if t0 >= t1:
            return None
        return tuple((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])) for t in (t0, t1))

    def _path(self, sc, chart, pts, ink, op, go=1.0, width=3.0):
        int(np.ceil((len(pts) - 1) * float(np.clip(go, 0, 1))))
        for a, b in itertools.pairwise(pts):
            seg = self._clip(a, b)
            if seg:
                self.polyline(sc, [self.P(chart.fx(x), chart.fy(y), lift=0.9) for x, y in seg],
                              width, ink, op, bias=0.3)

    def phase_3(self, sc, u, op):
        p2 = self.panels[2]
        self.head(sc, "steps 3 to 5: every held-out digit as a point, and three ways to "
                  "cut the plane", op)
        lo, hi = self.RANGE
        sets = ((0.09, self.left, p2["normal"], p2["anomalies"], p2["cuts"], p2["cov"],
                 p2["caught"]),
                (0.57, 0, p2["other_normal"], p2["other_anomalies"], p2["other_cuts"],
                 p2["other_cov"], p2["other_caught"]))
        ticks = [(v, str(v)) for v in (-2, 0, 2, 4, 6)]
        assert hi > 6
        ang = np.linspace(0.0, 2 * np.pi, 73)
        for x0, left, normal, anom, cuts, cov, caught in sets:
            chart = LossChart(((x0, x0 + 0.34), (0.39, 0.69)), (lo, hi), (lo, hi))
            self.axes(sc, chart, op, "A  ln loss, in σ of normal digits", "B  −ln p, in σ",
                      xticks=ticks, yticks=ticks, grid=False)
            for pts, ink, r, n, t0 in ((normal, soft(self.SEEN, 0.25), 2.2, 260, 0.02),
                                       (anom, self.UNSEEN, 2.4, 200, 0.10)):
                pts = np.asarray(pts, float)[:n]
                ok = (pts > lo).all(1) & (pts < hi).all(1)
                self.spots(sc, chart, pts[ok, 0], pts[ok, 1], ink, op,
                           smoothstep(t0, t0 + 0.10, u), r=r)
            c = [float(v) for v in cuts]
            L = np.linalg.cholesky(np.asarray(cov, float))
            ring = (np.sqrt(c[5]) * np.column_stack([np.cos(ang), np.sin(ang)])) @ L.T
            self._path(sc, chart, [(c[2], lo), (c[2], c[3]), (lo, c[3])], self.INKS5[2], op,
                       smoothstep(0.24, 0.36, u))
            self._path(sc, chart, [(c[4] - hi, hi), (c[4] - lo, lo)], self.INKS5[3], op,
                       smoothstep(0.42, 0.54, u))
            self._path(sc, chart, [tuple(v) for v in ring], self.INKS5[4], op,
                       smoothstep(0.60, 0.72, u), 2.6)
            self.text(sc, x0 + 0.34, 0.372, "the %ds removed" % left, op, size=MID,
                      anchor="rm", ink=dark(self.UNSEEN, 0.3))
            got = [float(v) for v in caught]
            self.lines(sc, x0 + 0.17, 0.80, [
                "caught: either cut %s, sum %s, ellipse %s"
                % (pct(got[2]), pct(got[3]), pct(got[4])),
                "(loss alone %s, code alone %s)" % (pct(got[0]), pct(got[1]))],
                op * smoothstep(0.74, 0.82, u), size=SMALL, step=0.038, anchor="mm")
        for n, (name, k) in enumerate((("either cut", 2), ("the sum", 3), ("the ellipse", 4))):
            self.key(sc, 0.09 + 0.15 * n, 0.884, [(name, self.INKS5[k])],
                     op * smoothstep(0.24 + 0.18 * n, 0.30 + 0.18 * n, u), size=SMALL)
        self.tag(sc, FIT_TAG, op)

    # ------------------------------------------------------------------ phase 4
    def phase_4(self, sc, u, op):
        m, left = self.meta, self.left
        self.head(sc, "ten removed classes, three seeds: the share of anomalies each rule "
                  "catches", op)
        mean = self.caught.mean((1, 2))                           # [alpha, rule]
        one = self.caught[self.a5, left].mean(0)
        area = m["auc"]
        full = self.span(0.26)[0]                                 # px for a share of 60%
        self.text(sc, 0.505, 0.375, "area", op, size=SMALL)
        self.text(sc, 0.59, 0.375, "the %ds" % left, op, size=SMALL, ink=dark(self.UNSEEN, 0.3))
        for k in range(5):
            y = 0.425 + 0.073 * k
            go = smoothstep(0.04 + 0.08 * k, 0.20 + 0.08 * k, u)
            ink = self.INKS5[k]
            self.hbar(sc, 0.185, y, mean[self.a5, k] / 0.6 * full, ink, op, self.SHORT[k],
                      pct(mean[self.a5, k]), go, thick=17.0, track=full)
            self.rect(sc, self.shift(self.P(0.185, y + 0.026), mean[0, k] / 0.6 * full * go / 2,
                                     0), max(mean[0, k] / 0.6 * full * go, 0.5), 5.0,
                      dark(ink, 0.35), op, radius=2)
            self.text(sc, 0.505, y, "%.2f" % area[S.METHODS[k]]["mean"], op * go, size=SMALL)
            self.text(sc, 0.59, y, pct(one[k]), op * go, size=SMALL,
                      ink=dark(self.UNSEEN, 0.3))
        self.lines(sc, 0.075, 0.80, [
            "bar: caught at 5% false alarms  ·  thin bar: at 1%",
            "area: under the ROC curve  ·  the %ds: caught at 5%%, class %d removed"
            % (left, left)], op, size=SMALL, step=0.042)
        raw = m["or_uncorrected"][str(m["alphas"][-1])]
        got = m["false_alarms"][str(m["alphas"][-1])]["either"]
        on = [op * smoothstep(0.50 + 0.1 * n, 0.58 + 0.1 * n, u) for n in range(4)]
        x = 0.795
        self.text(sc, x, 0.385, "the cost of a second cut", on[0], size=MID, strong=True)
        self.lines(sc, x, 0.445, ["two cuts at 5% each flag"], on[0], size=SMALL, anchor="mm")
        self.text(sc, x, 0.505, "%s of normal digits" % pct(raw["false_alarms"], 1), on[1],
                  size=BIG, ink=dark(RED, 0.25))
        self.text(sc, x, 0.56, "1 − 0.95² = %s if independent" % pct(raw["if_independent"], 2),
                  on[1], size=SMALL)
        self.text(sc, x, 0.635, "set each at %s instead:" % pct(raw["fair_rate"], 1), on[2],
                  size=SMALL)
        self.text(sc, x, 0.69, "%s in total" % pct(got, 1), on[2], size=BIG,
                  ink=dark(GREEN, 0.3))
        self.text(sc, x, 0.745, "(on the other half of the normal digits)", on[2],
                  size=SMALL)
        self.tag(sc, FIT_TAG, op)

    # ------------------------------------------------------------------ phase 5
    def phase_5(self, sc, u, op):
        p3 = self.panels[3]
        codes, by = self.codes, self.by_code
        self.head(sc, "the same rules at four code sizes", op)
        knn = np.asarray(p3["knn"], float)
        low = float(np.floor(min(by[:4].min(), knn.min()) * 10) / 10)
        chart = LossChart(((0.09, 0.52), (0.40, 0.80)), (codes[0] * 0.8, codes[-1] * 1.25),
                          (low, 1.0), xlog=True)
        self.axes(sc, chart, op, "numbers in the code", "area under the ROC curve",
                  xticks=[(c, str(c)) for c in codes],
                  yticks=[(v, "%.1f" % v) for v in np.arange(low, 1.01, 0.2)])
        if low < 0.5:
            chart.hline(sc, self.P, 0.5, op, dashes=24, width=1.4)
        go = smoothstep(0.06, 0.50, u)
        self.trace(sc, chart, codes, knn, GREY, op, go, 2.0, r=3.2)
        for k in (3, 2, 1, 0):
            self.trace(sc, chart, codes, by[k], self.INKS5[k], op, go, 3.0, r=4.4)
        self.key(sc, 0.585, 0.40, [(self.SHORT[k] + (": density of the code" if k == 1 else ""),
                                    self.INKS5[k]) for k in range(4)]
                 + [("code alone: %dth-neighbour distance" % S.KNN, GREY)], op, size=SMALL)
        main = codes.index(self.main)
        ones = np.asarray(p3["seeds"], float)[:, self.left, :, 1].mean(1)
        note = op * smoothstep(0.56, 0.66, u)
        self.lines(sc, 0.575, 0.65, [
            "area at %s numbers:" % ", ".join(str(c) for c in codes[main:]),
            "loss alone  " + ", ".join("%.2f" % v for v in by[0, main:]),
            "code alone  " + ", ".join("%.2f" % v for v in by[1, main:]),
            "either cut  " + ", ".join("%.2f" % v for v in by[2, main:]),
            "neighbour distance  " + ", ".join("%.2f" % v for v in knn[main:]),
            ("the %ds, code alone  " % self.left + ", ".join("%.2f" % v for v in ones[main:]),
             dark(self.UNSEEN, 0.3))], note, size=SMALL, step=0.043)
        self.tag(sc, MNIST_TAG, op)


ANIMATIONS = {
    "anomaly-loss-and-latent": LossAndLatent,
    "anomaly-two-score-flow": TwoScoreFlow,
}

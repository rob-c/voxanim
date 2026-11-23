"""Animation 177 of the "next" catalogue (see :mod:`voxanim.mech`): every loss is a
likelihood, the bridge from fitting in Part 1 (chi-square, the likelihood curve) to
training classifiers in Part 2 and the networks after it.

:class:`EveryLossIsALikelihood` draws :mod:`voxanim.study_bridge_loss`'s study: a
straight line through binned ink measurements of MNIST digits, its chi-square and its
likelihood over the slope; the width of the likelihood against the scatter of the slope
in 120 disjoint samples; the same line by least squares on the digits themselves; a
softmax regression's cross-entropy as the log of a product of probabilities; a logistic
regression's binary cross-entropy; and the table of losses with the distribution each
assumes. Every phase carries, bottom right, what kind of evidence it shows.
"""

from __future__ import annotations

import numpy as np

from . import mnist
from . import study_bridge_loss as S
from .easing import smoothstep
from .losschart import sup
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    GREEN,
    OCHRE,
    RED,
    TEAL,
    VIOLET,
    dark,
    nice_ticks,
    number,
    soft,
)
from .mech_phys_apps import Phys, commas, pct
from .study_stats_likelihood import normal_pdf

BINNED = "measured on MNIST: 500 training digits, binned"
SAMPLED = "measured on MNIST: 120 disjoint samples of 500 digits"
DIGITS = "measured on MNIST: 500 digits, 10,000 held out"
TEN = "measured on MNIST: 10,000 held-out digits"
TWO = "measured on MNIST: 1,991 held-out 4s and 9s"
TABLE = "measured on MNIST: every number from the runs above"
BIG, HEAD, SMALL, TEXT_S = 0.036, 0.033, 0.030, 0.029
CHI, LIK = BLUE, GREEN                   # chi-square; the likelihood
LSQ = AMBER                              # least squares on the digits
CAT, BER = VIOLET, TEAL                  # the ten-class and the two-class models
FROM = dark(OCHRE, 0.35)                 # where the bridge comes from and goes to


def is_evidence(text):
    """True for the tag every phase carries."""
    return text.startswith(("exact arithmetic on ", "measured on MNIST"))


def signed(v, digits=2):
    """``v`` to ``digits`` places with a typographic minus: ``−83.11``."""
    return ("%.*f" % (digits, float(v))).replace("-", "−")


def thousands(v, digits=1):
    """``v`` with commas and a typographic minus: ``−2,747.3``."""
    return f"{float(v):,.{digits}f}".replace("-", "−")


class EveryLossIsALikelihood(Phys):
    """Chi-square against the likelihood over one slope; the likelihood's width against
    120 repeats; least squares on single digits; cross-entropy of ten classes as a product
    of probabilities; binary cross-entropy of two; the table of losses."""

    TITLE = "Every loss is a likelihood"
    PHASES = (("chi-square", 11.0), ("its width", 9.0), ("one σ", 9.0),
              ("ten digits", 11.0), ("two digits", 9.0), ("the losses", 9.0))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        self.g, self.one = m["gauss"], m["one_sigma"]
        self.ten, self.two = m["softmax"], m["logistic"]
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.b = np.asarray(p0["x"], float)
        self.chi2, self.m2ll = (np.asarray(v, float) for v in p0["ys"])
        self.lik = np.exp(-(self.m2ll - self.m2ll.min()) / 2)          # L / L at its peak
        self.bx, self.by, self.bs = (np.asarray(p0[k], float) for k in ("bins_x", "bins_y",
                                                                       "bins_sigma"))
        self.counts = np.asarray(p1["values"], float)
        self.edges = np.asarray(p1["edges"], float)
        self.L, self.R = np.asarray(p2["xy"], float).T
        self.hist = np.asarray(p3["x"], float), np.asarray(p3["ys"], float)
        sm, lg = S.softmax_run(), S.logistic_run()
        W = np.asarray(sm["W"], float)
        self.templates = W.T.reshape(10, 28, 28)
        self.tvmax = float(np.percentile(np.abs(W), 99.5))
        self.w2 = np.asarray(lg["w"], float).reshape(28, 28)
        self.w2max = float(np.percentile(np.abs(self.w2), 99.5))
        z, t = np.asarray(lg["test_logits"], float), np.asarray(lg["test_t"], float)
        self.zedges = np.linspace(-16.0, 16.0, 41)
        clip = np.clip(z, self.zedges[0] + 1e-9, self.zedges[-1] - 1e-9)
        self.z4 = np.histogram(clip[t == 0], self.zedges)[0].astype(float)
        self.z9 = np.histogram(clip[t == 1], self.zedges)[0].astype(float)
        self.image = mnist.load()[2][self.ten["shown_digit"]]
        g = self.g
        # the Gaussian the curvature predicts for the 120 slopes
        width = self.edges[1] - self.edges[0]
        self.gx = np.linspace(self.edges[0], self.edges[-1], 121)
        self.gy = g["samples"] * width * normal_pdf(self.gx, g["mean_b"], g["mean_sigma_b"])
        # -ln L above its lowest point, near the peak
        self.dnll = (self.m2ll - self.m2ll.min()) / 2

    # ------------------------------------------------------------------ words
    def subtitle_for(self, k):
        g, ten, two = self.g, self.ten, self.two
        return [
            "from Part 1: a line fitted to measured ink by the lowest chi-square. With "
            "Gaussian errors −2 ln L is chi-square plus a constant, so the same slope, "
            "%.3f, has the highest likelihood" % g["b"],
            "the curvature of −ln L at its peak gives the slope an error of ±%.3f. The "
            "slopes of %d disjoint samples of %d digits scatter by %.3f"
            % (g["curvature_sigma"], g["samples"], g["n_digits"], g["spread"]),
            "with one σ for every digit, chi-square is N / σ² times the mean squared error: "
            "least squares is maximum likelihood for Gaussian errors",
            "one of ten digits: a softmax regression's held-out cross-entropy, %.3f nats, "
            "is minus the log of the product of the probabilities it gave the right "
            "digits, over %s" % (ten["cross_entropy"], commas(ten["n_test"])),
            "one of two, 4 against 9: the binary cross-entropy of a logistic regression is "
            "minus the log of a Bernoulli likelihood, %.3f nats a held-out digit"
            % two["bce"],
            "each loss is minus the log-likelihood of the distribution it assumes. Next: "
            "the same cross-entropy trains every network on this page",
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def ink_axes(self, sc, ch, op, xlabel=True):
        self.axes(sc, ch, op, "ink of the left half (mean pixel)" if xlabel else "",
                  "ink of the right half", [(v, "%.2f" % v) for v in (0.05, 0.10, 0.15, 0.20,
                                                                       0.25)],
                  [(v, "%.2f" % v) for v in (0.05, 0.10, 0.15, 0.20, 0.25)])

    def error_bars(self, sc, ch, op, grow=1.0):
        for x, y, s in zip(self.bx, self.by, self.bs):
            a, b = ch.fy(y - s * grow), ch.fy(y + s * grow)
            fx = ch.fx(x)
            self.polyline(sc, [self.P(fx, a, 0.9), self.P(fx, b, 0.9)], 1.8, dark(CHI, 0.2),
                          op, bias=0.3)
            for yy in (a, b):
                self.polyline(sc, [self.P(fx - 0.005, yy, 0.9), self.P(fx + 0.005, yy, 0.9)],
                              1.6, dark(CHI, 0.2), op, bias=0.3)
            self.disk(sc, self.P(fx, ch.fy(y)), 4.2, CHI, op, lift=1.2)

    def line(self, sc, ch, a, b, ink, op, go=1.0, width=2.4, dashed=False):
        x0, x1 = ch.x_range
        x1 = x0 + (x1 - x0) * go
        p, q = (ch.fx(x0), ch.fy(a + b * x0)), (ch.fx(x1), ch.fy(a + b * x1))
        if dashed:
            self.dashes(sc, p, q, ink, op, n=16, width=width)
        else:
            self.polyline(sc, [self.P(*p, 0.8), self.P(*q, 0.8)], width, ink, op, bias=0.3)

    def reveal(self, sc, fx, fy, rows, u, op, at=0.0, every=0.06, step=0.044):
        """Rows of ``(text, ink, size)`` stacked down from ``(fx, fy)``, the ``i``-th
        fading in at ``u = at + i * every``."""
        y = fy
        for i, (s, ink, size) in enumerate(rows):
            a = op * smoothstep(at + i * every, at + i * every + 0.05, u)
            self.text(sc, fx, y, s, a, anchor="lm", ink=ink, size=size)
            y += step
        return y

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        g = self.g
        self.text(sc, 0.05, 0.315, "from: Chi-square as a landscape · The likelihood curve",
                  op, anchor="lm", ink=FROM, size=SMALL)
        ch = self.pane(((0.10, 0.45), (0.40, 0.71)), (0.02, 0.27), (0.03, 0.27))
        self.ink_axes(sc, ch, op)
        self.error_bars(sc, ch, op, smoothstep(0.02, 0.15, u))
        self.line(sc, ch, g["a"], g["b"], CHI, op * smoothstep(0.12, 0.2, u),
                  smoothstep(0.12, 0.3, u))
        self.text(sc, 0.05, 0.83, "%d training digits in %d bins: mean ± its error"
                  % (g["digits_used"], g["points"]), op, anchor="lm", size=TEXT_S)
        # chi-square and the likelihood over the slope
        b0, b1 = float(self.b[0]), float(self.b[-1])
        bt = [(v, "%.2f" % v) for v in nice_ticks(b0, b1, 3)]
        go = smoothstep(0.25, 0.55, u)
        top = float(self.chi2.max())
        c1 = self.pane(((0.60, 0.93), (0.36, 0.47)), (b0, b1), (0.0, top * 1.05))
        self.axes(sc, c1, op, "", "chi-square", [],
                  [(v, number(v)) for v in nice_ticks(0.0, top, 2)], grid=False)
        self.path(sc, c1, self.b, self.chi2, CHI, op, go, width=2.4)
        c2 = self.pane(((0.60, 0.93), (0.56, 0.70)), (b0, b1), (0.0, 1.08))
        self.axes(sc, c2, op, "the slope b (a at its best)", "L / L max", bt,
                  [(0.0, "0"), (1.0, "1")], grid=False)
        self.path(sc, c2, self.b, self.lik, LIK, op, go, width=2.4)
        mark = op * smoothstep(0.55, 0.62, u)
        x = c1.fx(g["b"])
        self.dashes(sc, (x, c1.y0), (x, c2.y1), AMBER, mark, n=16, width=1.6)
        self.text(sc, 0.545, 0.81, "lowest chi-square: b = %.3f" % g["b_chi2_min"], mark,
                  anchor="lm", size=TEXT_S, ink=dark(CHI, 0.3))
        self.text(sc, 0.545, 0.857, "highest likelihood: b = %.3f" % g["b_likelihood_max"],
                  mark, anchor="lm", size=TEXT_S, ink=dark(LIK, 0.3))
        late = op * smoothstep(0.7, 0.78, u)
        self.text(sc, 0.05, 0.878, "−2 ln L = χ² + Σ ln 2πσᵢ² = χ² %s %.2f"
                  % ("−" if g["constant"] < 0 else "+", abs(g["constant"])), late,
                  anchor="lm", size=SMALL, ink=dark(LIK, 0.3))
        self.tag(sc, BINNED, op)

    def phase_2(self, sc, u, op):
        g = self.g
        self.text(sc, 0.05, 0.315, "from: Errors from the likelihood", op, anchor="lm",
                  ink=FROM, size=SMALL)
        b0, b1 = float(self.b[0]), float(self.b[-1])
        ch = self.pane(((0.10, 0.44), (0.40, 0.66)), (b0, b1), (0.0, 4.0))
        self.axes(sc, ch, op, "the slope b", "−ln L above its lowest",
                  [(v, "%.2f" % v) for v in nice_ticks(b0, b1, 3)],
                  [(v, "%g" % v) for v in (0, 0.5, 2, 4)], grid=False)
        keep = self.dnll <= 4.0
        self.path(sc, ch, self.b[keep], self.dnll[keep], LIK, op, smoothstep(0.02, 0.25, u),
                  width=2.4)
        half = op * smoothstep(0.25, 0.32, u)
        self.hline(sc, ch, 0.5, AMBER, half, width=1.6)
        for v in (g["b"] - g["curvature_sigma"], g["b"] + g["curvature_sigma"]):
            self.dashes(sc, (ch.fx(v), ch.fy(0.5)), (ch.fx(v), ch.y1), AMBER, half, n=6,
                        width=1.6)
        self.reveal(sc, 0.05, 0.795, [
            ("curvature of −ln L at b = %.3f: %s" % (g["b"], commas(g["curvature"])), None,
             SMALL),
            ("error 1 / √%s = ±%.3f, where −ln L is ½ up" % (commas(g["curvature"]),
                                                            g["curvature_sigma"]),
             dark(LIK, 0.3), SMALL)], u, op, at=0.3, every=0.08)
        # the slopes of the disjoint samples
        top = max(float(self.counts.max()), float(self.gy.max())) * 1.15
        c2 = self.pane(((0.58, 0.93), (0.40, 0.66)), (self.edges[0], self.edges[-1]),
                       (0.0, top))
        self.axes(sc, c2, op, "the fitted slope", "samples",
                  [(v, "%.2f" % v) for v in nice_ticks(self.edges[0], self.edges[-1], 3)],
                  [(v, "%d" % v) for v in nice_ticks(0.0, top, 3)], grid=False)
        grow = smoothstep(0.4, 0.7, u)
        self.bars(sc, c2, self.edges, self.counts, LSQ, op, fill=0.7, grow=grow)
        self.path(sc, c2, self.gx, self.gy, LIK, op * smoothstep(0.65, 0.72, u), width=2.2)
        self.reveal(sc, 0.55, 0.77, [
            ("%d disjoint samples of %d digits:" % (g["samples"], g["n_digits"]), None, SMALL),
            ("the slopes scatter by %.3f ± %.3f" % (g["spread"], g["spread_error"]),
             dark(LSQ, 0.4), SMALL),
            ("green: the curvature's errors, mean %.3f" % g["mean_sigma_b"], dark(LIK, 0.3),
             SMALL)], u, op, at=0.55, every=0.08)
        self.tag(sc, SAMPLED, op)

    def phase_3(self, sc, u, op):
        g, one = self.g, self.one
        ch = self.pane(((0.10, 0.45), (0.36, 0.70)), (0.0, 0.34), (0.0, 0.34))
        self.axes(sc, ch, op, "ink of the left half (mean pixel)", "ink of the right half",
                  [(v, "%.1f" % v) for v in (0.1, 0.2, 0.3)],
                  [(v, "%.1f" % v) for v in (0.0, 0.1, 0.2, 0.3)], grid=False)
        n = int(len(self.L) * smoothstep(0.02, 0.3, u))
        self.dots(sc, ch, self.L[:n], self.R[:n], soft(BLUE, 0.3), op, r=2.2, lift=0.8)
        self.line(sc, ch, one["a"], one["b"], LSQ, op * smoothstep(0.3, 0.36, u),
                  smoothstep(0.3, 0.45, u), width=2.6)
        self.line(sc, ch, g["a"], g["b"], CHI, op * smoothstep(0.45, 0.5, u), dashed=True,
                  width=1.8)
        self.key(sc, 0.06, 0.815, [("least squares, every digit: b = %.3f" % one["b"], LSQ),
                                  ("chi-square, binned: b = %.3f ± %.3f"
                                   % (g["b"], g["sigma_b"]), CHI)],
                 op * smoothstep(0.45, 0.5, u), size=TEXT_S, step=0.04)
        self.reveal(sc, 0.53, 0.36, [
            ("one σ for every digit:", None, SMALL),
            ("−2 ln L = N ln 2πσ² + Σ rᵢ² / σ²", None, SMALL),
            ("Σ rᵢ² / σ² = χ² = N · MSE / σ²", dark(CHI, 0.3), HEAD),
            ("whatever σ is, the lowest mean squared", None, SMALL),
            ("error is the highest likelihood", None, SMALL)], u, op, at=0.08, every=0.06)
        self.reveal(sc, 0.53, 0.62, [
            ("MSE %.6f on these %d digits," % (one["mse_train"], one["n"]), None, SMALL),
            ("%.6f on %s held out" % (one["mse_test"], commas(one["n_test"])),
             dark(LSQ, 0.4), SMALL),
            ("the best σ is √MSE = %.4f," % one["sigma"], None, SMALL),
            ("and then χ² = N = %s" % commas(one["chi2"]), dark(CHI, 0.3), SMALL)], u, op,
            at=0.5, every=0.07)
        self.tag(sc, DIGITS, op)

    def phase_4(self, sc, u, op):
        ten = self.ten
        self.text(sc, 0.05, 0.315, "a softmax regression, 784 → 10: one weight per pixel "
                  "and digit", op, anchor="lm", strong=True, size=SMALL)
        size = 38
        for k in range(10):
            on = op * smoothstep(0.02 + 0.02 * k, 0.08 + 0.02 * k, u)
            fx = 0.075 + 0.047 * k
            self.grid(sc, self.templates[k], fx, 0.39, size, size, on, vmax=self.tvmax,
                      frame=False)
            self.text(sc, fx, 0.445, str(k), on, size=TEXT_S)
        steps, ys = self.hist
        top = ten["uniform"] * 1.08
        ch = self.pane(((0.10, 0.50), (0.51, 0.74)), (0.0, steps[-1]), (0.0, top))
        self.axes(sc, ch, op, "training steps of %d digits (%d epochs)"
                  % (ten["batch"], ten["epochs"]), "cross-entropy, nats",
                  [(v, commas(v)) for v in (0, 1000, 2000, 3000)],
                  [(v, "%.1f" % v) for v in (0.0, 0.5, 1.0, 1.5, 2.0)], grid=False)
        go = smoothstep(0.1, 0.45, u)
        self.path(sc, ch, steps, ys[0], soft(CAT, 0.5), op, go, width=1.8)
        self.path(sc, ch, steps, ys[1], CAT, op, go, width=2.6, dot=True)
        self.key(sc, 0.29, 0.56, [("training", soft(CAT, 0.5)), ("held out", CAT)], op,
                 size=TEXT_S, step=0.04)
        self.text(sc, 0.05, 0.85, "starts at ln 10 = %.3f: every digit 1 in 10"
                  % ten["uniform"], op * smoothstep(0.1, 0.2, u), anchor="lm", size=TEXT_S)
        # one digit, then all of them
        lbl = ten["shown_label"]
        p = ten["shown_probabilities"][lbl]
        on = op * smoothstep(0.35, 0.42, u)
        self.digit(sc, self.image, 0.60, 0.40, 54, on, frame=soft(CAT, 0.3))
        self.lines(sc, 0.645, 0.38, ["test digit %d, a %d:" % (ten["shown_digit"], lbl),
                                     ("p(%d) = %.4f" % (lbl, p), dark(CAT, 0.3))], on,
                   size=SMALL, step=0.042)
        ll = ten["log_likelihood"]
        self.reveal(sc, 0.56, 0.52, [
            ("the likelihood of the %s test labels:" % commas(ten["n_test"]), None, SMALL),
            ("Π p(right digit) ≈ 10%s" % sup(int(round(ten["log10_product"]))),
             dark(CAT, 0.3), HEAD),
            ("ln L = Σ ln p = %s" % thousands(ll), None, SMALL),
            ("−ln L / %s = %.3f nats" % (commas(ten["n_test"]), ten["cross_entropy"]),
             dark(CAT, 0.3), HEAD),
            ("= the held-out cross-entropy;", None, SMALL),
            ("%s of the test digits right" % pct(ten["accuracy"], 1), None, SMALL)],
            u, op, at=0.48, every=0.06)
        self.tag(sc, TEN, op)

    def phase_5(self, sc, u, op):
        two = self.two
        self.text(sc, 0.05, 0.315, "a logistic regression, 784 → 1: is it a 9 or a 4?", op,
                  anchor="lm", strong=True, size=SMALL)
        self.grid(sc, self.w2, 0.12, 0.50, 120, 120, op, vmax=self.w2max)
        self.lines(sc, 0.05, 0.665, ["its 784 weights:", ("red: towards a 9", dark(RED, 0.2)),
                                     ("blue: towards a 4", dark(BLUE, 0.2))], op,
                   size=TEXT_S, step=0.04)
        top = max(float(self.z4.max()), float(self.z9.max())) * 1.15
        ch = self.pane(((0.27, 0.56), (0.40, 0.68)), (self.zedges[0], self.zedges[-1]),
                       (0.0, top))
        self.axes(sc, ch, op, "z = w · x + c, held out", "digits",
                  [(v, signed(v, 0)) for v in (-15, -10, -5, 0, 5, 10, 15)],
                  [(v, "%d" % v) for v in nice_ticks(0.0, top, 3)], grid=False)
        grow = smoothstep(0.05, 0.35, u)
        self.bars(sc, ch, self.zedges, self.z4, BLUE, op, fill=0.55, grow=grow)
        self.bars(sc, ch, self.zedges, self.z9, RED, op, fill=0.55, grow=grow)
        self.vline(sc, ch, 0.0, AXIS, op, width=1.4, dashed=True)
        self.key(sc, 0.29, 0.80, [("the %s held-out 4s" % commas(two["n_test"] - two["nines"]),
                                   BLUE), ("the %s held-out 9s" % commas(two["nines"]), RED)],
                 op, size=TEXT_S, step=0.04)
        self.reveal(sc, 0.60, 0.37, [
            ("p(9) = σ(z), y = 1 for a 9", None, SMALL),
            ("Bernoulli: L = Π p^y (1 − p)^(1−y)", None, SMALL),
            ("ln L = %s over %s digits" % (thousands(two["log_likelihood"]),
                                          commas(two["n_test"])), None, SMALL),
            ("−ln L / %s = %.3f nats" % (commas(two["n_test"]), two["bce"]), dark(BER, 0.3),
             HEAD),
            ("= the binary cross-entropy", dark(BER, 0.3), SMALL),
            ("%s right; a coin: ln 2 = %.3f" % (pct(two["accuracy"], 1), two["uniform"]), None,
             SMALL)], u, op, at=0.3, every=0.06)
        self.reveal(sc, 0.60, 0.665, [
            ("two classes of a softmax: σ(z) = softmax(0, z)", None, TEXT_S),
            ("the ten-class model on 4 and 9 alone,", None, TEXT_S),
            ("σ(z₉ − z₄): %.3f nats, %s right" % (two["ten_class_bce"],
                                                 pct(two["ten_class_accuracy"], 1)),
             dark(CAT, 0.3), TEXT_S)], u, op, at=0.68, every=0.06, step=0.04)
        self.tag(sc, TWO, op)

    def phase_6(self, sc, u, op):
        g, one, ten, two = self.g, self.one, self.ten, self.two
        cols = (0.05, 0.29, 0.56, 0.78)
        heads = ("the errors are", "−ln L of one point", "the loss", "measured here")
        for x, h in zip(cols, heads):
            self.text(sc, x, 0.33, h, op, anchor="lm", strong=True, size=SMALL)
        rows = (
            ("Gaussian, σᵢ each", "½ ((y − f) / σᵢ)² + const", "chi-square",
             "%.1f, %d d.o.f." % (g["chi2_min"], g["dof"]), CHI),
            ("Gaussian, one σ", "(y − f)² / 2σ² + const", "squared error",
             "MSE %.6f held out" % one["mse_test"], LSQ),
            ("Bernoulli", "−y ln p − (1−y) ln(1−p)", "binary cross-entropy",
             "%.3f nats held out" % two["bce"], BER),
            ("categorical", "−ln p(the right class)", "cross-entropy",
             "%.3f nats held out" % ten["cross_entropy"], CAT),
        )
        for i, (dist, term, loss, value, ink) in enumerate(rows):
            y = 0.41 + 0.085 * i
            on = op * smoothstep(0.04 + 0.1 * i, 0.12 + 0.1 * i, u)
            self.box(sc, 0.5, y, 880, 50, soft(ink, 0.82), on * 0.6, radius=6)
            self.box(sc, cols[0] - 0.012, y, 8, 34, ink, on, radius=3)
            self.text(sc, cols[0], y, dist, on, anchor="lm", size=SMALL, ink=dark(ink, 0.3))
            self.text(sc, cols[1], y, term, on, anchor="lm", size=TEXT_S)
            self.text(sc, cols[2], y, loss, on, anchor="lm", size=SMALL, strong=True)
            self.text(sc, cols[3], y, value, on, anchor="lm", size=TEXT_S)
        self.text(sc, 0.05, 0.79, "minimise the loss = maximise the likelihood", op *
                  smoothstep(0.45, 0.52, u), anchor="lm", size=HEAD)
        self.text(sc, 0.05, 0.85, "next: the same cross-entropy trains every network on "
                  "this page", op * smoothstep(0.6, 0.68, u), anchor="lm", ink=FROM, size=BIG)
        self.tag(sc, TABLE, op)


ANIMATIONS = {"every-loss-is-a-likelihood": EveryLossIsALikelihood}

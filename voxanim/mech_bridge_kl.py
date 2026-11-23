"""Animation 176 of the "next" catalogue (see :mod:`voxanim.mech`): the bridge from a
classifier's loss ("Logits, loss and nats", 175) to the KL divergence that comes back
later on the page (label smoothing, distillation, the KL term of a VAE, t-SNE, RLHF).

:class:`KLDivergence`: one test digit's loss recalled as a cross-entropy, split into the
target's own entropy and KL(p || q); the split class by class for a label-smoothed
target; three targets for the same model over the 10,000 test digits, with the untrained
network for comparison; KL(p || q) against KL(q || p); one Gaussian fitted to a
two-humped distribution of real digits by forward and by reverse KL; and where KL comes
back.

It draws :mod:`voxanim.study_bridge_kl`'s study; the histogram of the two humps is
counted here from its cached projections. Every phase says, bottom right, what kind of
evidence it shows: exact arithmetic on one digit's measured probabilities, or a
measurement on MNIST.
"""

from __future__ import annotations

import numpy as np

from . import mnist
from . import study_bridge_kl as S
from .easing import smoothstep
from .mech import (
    AMBER,
    BLUE,
    DIGIT_INK,
    GREEN,
    MAGENTA,
    OCHRE,
    RED,
    TEAL,
    VIOLET,
    dark,
    soft,
)
from .mech_info_nets import Info
from .mech_phys_apps import GREY, commas, pct

EXACT = "exact arithmetic on one digit's measured probabilities"
HELD = "measured on MNIST: 10,000 held-out digits"
HUMPS = "measured on MNIST: %s training 0s and 1s"
BIG, HEAD, SMALL, TEXT_S = 0.036, 0.033, 0.030, 0.029
ENT, KLI, CE = VIOLET, AMBER, MAGENTA       # H(p), KL(p || q), the cross-entropy
MODEL = GREEN                               # the classifier q
FWD, REV = BLUE, RED                        # KL(p || q) and KL(q || p); their two fits
ONES, ZEROS = OCHRE, TEAL                   # the two humps' digits
NAMES = {"one-hot": "one-hot labels", "smoothed": "smoothed, ε = 0.1",
         "teacher": "teacher, T = 2"}


def is_evidence(text):
    """True for the tag every phase carries."""
    return text.startswith(("exact arithmetic on ", "measured on MNIST"))


def nats(v, digits=3):
    """A value in nats with a typographic minus."""
    return ("%.*f" % (digits, float(v))).replace("-", "−")


def tiny(v):
    """A small probability as ``1.0e−10`` (or plainly when it is not small)."""
    v = float(v)
    if v >= 0.001:
        return "%.4f" % v
    m, e = ("%.1e" % v).split("e")
    return "%s×10%s" % (m, "".join("⁰¹²³⁴⁵⁶⁷⁸⁹"[int(c)] if c.isdigit() else "⁻"
                                   for c in str(int(e))))


class KLDivergence(Info):
    """175's loss of one digit as a cross-entropy and its split; the split term by term;
    three targets over the test set; which way round; two humps and one bell; next."""

    TITLE = "Cross-entropy is entropy plus KL"
    PHASES = (("recall", 10.0), ("split", 11.0), ("targets", 10.5), ("reversed", 9.0),
              ("fits", 11.0), ("next", 8.0))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        self.rows = m["rows"]
        self.shown = m["shown"]
        self.hm = m["humps"]
        d0, d3 = s.panels[0].data, s.panels[3].data
        self.q = np.asarray(d0["q"], float)
        self.p = np.asarray(d0["p"], float)
        self.h_terms = np.asarray(d0["entropy"], float)
        self.kl_terms = np.asarray(d0["kl"], float)
        self.ce_terms = np.asarray(d0["values"], float)
        self.image = mnist.load()[2][self.shown["digit"]]
        self.x = np.asarray(d3["x"], float)
        self.curves = np.asarray(d3["ys"], float)           # p, forward fit, reverse fit
        self.direction = np.asarray(d3["direction"], float)
        hb = S.humps()
        v, y = hb["v"], hb["y"]
        lo = float(self.hm["class_mean"]["1"]) - 4.0
        hi = float(self.hm["class_mean"]["0"]) + 6.0
        self.xr = (lo, hi)
        self.edges = np.linspace(lo, hi, 61)
        width = self.edges[1] - self.edges[0]
        self.counts = {c: np.histogram(v[y == c], self.edges)[0] / (len(v) * width)
                       for c in (1, 0)}
        self.n_humps = int(len(v))

    def subtitle_for(self, k):
        m, r, sh, hm = self.meta, self.rows, self.shown, self.hm
        f = hm["forward"]
        return [
            "from Logits, loss and nats: test digit %d's loss, −ln %.4f = %.3f nats, is a "
            "cross-entropy. It splits into the target's own entropy, 0 for a one-hot label, "
            "and the KL divergence" % (sh["digit"], self.q[sh["label"]], sh["onehot_loss"]),
            "with a label-smoothed target the split is %.3f + %.3f = %.3f nats, class by class."
            " %s of it is the %.2f on each wrong digit, which q gave almost nothing"
            % (sh["entropy"], sh["kl"], sh["cross_entropy"], pct(sh["wrong_share"]),
               self.p[0 if sh["label"] else 1]),
            "over the %s test digits the parts add up exactly for every target. H(p) belongs "
            "to the target, so training can change only the KL: for smoothed targets "
            "%.2f → %.2f" % (commas(m["n"]), r["smoothed"]["untrained_kl"],
                             r["smoothed"]["kl"]),
            "KL is not symmetric: forward KL(p‖q) weighs the log ratio by p, reverse KL(q‖p) "
            "by q. For the teacher %.2f against %.2f; for one-hot labels the reverse is infinite"
            % (r["teacher"]["kl"], r["teacher"]["reverse_kl"]),
            "one bell fitted to two humps of real digits: forward KL covers both, centred "
            "where p is %s of its peak; reverse KL sits on the 1s alone"
            % pct(f["p_at_centre"]),
            "next on the page, KL is what label smoothing, distillation and a VAE's KL term "
            "measure, each in one direction; t-SNE and RLHF's penalty use it too",
        ][k]

    # ------------------------------------------------------------------ helpers
    def stacked(self, sc, fx, fy, h, kl, scale, op, grow=1.0, thick=22.0):
        """A horizontal bar of ``h`` (entropy) then ``kl``, ``scale`` px a nat; returns
        the layout x where it ends."""
        a = self.P(float(fx), float(fy))
        wh = max(h * scale * grow, 0.0)
        wk = max(kl * scale * smoothstep(0.3, 1.0, grow), 0.0)
        if wh > 0.5:
            self.rect(sc, self.shift(a, wh / 2, 0), wh, thick, ENT, op, radius=3, lift=0.3)
        if wk > 0.5:
            self.rect(sc, self.shift(a, wh + wk / 2, 0), wk, thick, KLI, op, radius=3,
                      lift=0.3)
        return fx + self.frac(wh + wk)[0]

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        sh, r = self.shown, self.rows
        lab = sh["label"]
        self.text(sc, 0.05, 0.315, "from: Logits, loss and nats", op, anchor="lm",
                  strong=True, size=HEAD)
        self.digit(sc, self.image, 0.10, 0.48, 84, op, frame=soft(MODEL, 0.3))
        self.text(sc, 0.10, 0.60, "test digit %d: a %d" % (sh["digit"], lab), op, size=TEXT_S)
        ch = self.pane(((0.21, 0.47), (0.40, 0.72)), (-0.5, 9.5), (0.0, 1.0))
        self.axes(sc, ch, op, "digit", "probability", [(k, str(k)) for k in range(10)],
                  [(v, "%g" % v) for v in (0, 0.5, 1)], grid=False)
        grow = smoothstep(0.03, 0.25, u)
        bw = self.span(0.55 * (ch.x1 - ch.x0) / 10)[0]
        for k in range(10):
            y0, y1 = ch.fy(0.0), ch.fy(float(self.q[k]) * grow)
            h = self.span(0, y0 - y1)[1]
            if h > 0.5:
                self.rect(sc, self.P(ch.fx(k), (y0 + y1) / 2), bw, h, MODEL, op, radius=2,
                          lift=0.4)
        target = op * smoothstep(0.2, 0.3, u)
        y0, y1 = ch.fy(0.0), ch.fy(1.0)
        self.rect(sc, self.P(ch.fx(lab), (y0 + y1) / 2), bw + 10, self.span(0, y0 - y1)[1] + 4,
                  dark(GREY, 0.2), target, border=dark(GREY, 0.2), border_w=2.0, radius=3,
                  lift=0.7, fill_op=0.0)
        self.key(sc, 0.23, 0.835, [("q: the classifier's probabilities", MODEL),
                                  ("p: the one-hot target", dark(GREY, 0.2))], op,
                 size=TEXT_S, step=0.04)
        x = 0.53
        self.reveal(sc, x, 0.38, [
            ("loss = −ln q(%d) = −ln %.4f = %.3f nats" % (lab, self.q[lab], sh["onehot_loss"]),
             None, SMALL),
            ("a cross-entropy: CE(p, q) = −Σ p ln q", dark(CE, 0.3), SMALL),
            ("= H(p) + KL(p‖q)", dark(CE, 0.3), BIG)], u, op, at=0.12, every=0.09,
            step=0.05)
        self.reveal(sc, x, 0.56, [
            ("H(p) = −Σ p ln p: the target's own entropy", dark(ENT, 0.3), TEXT_S),
            ("KL(p‖q) = Σ p ln(p / q): how far q is from p", dark(KLI, 0.4), TEXT_S),
            ("one-hot p: H = 0, so CE = KL = %.3f" % sh["onehot_loss"], None, SMALL),
            ("all test digits: %.3f nats, all of it KL" % r["one-hot"]["kl"], None, SMALL)],
            u, op, at=0.45, every=0.09, step=0.048)
        self.tag(sc, EXACT, op)

    def phase_2(self, sc, u, op):
        sh = self.shown
        lab = sh["label"]
        other = 0 if lab else 1
        self.text(sc, 0.05, 0.315, "label-smoothed target, ε = 0.1: p = %.2f on the %d, %.2f "
                  "on each other digit" % (self.p[lab], lab, self.p[other]), op, anchor="lm",
                  strong=True, size=SMALL)
        top = float(np.max(self.h_terms + np.maximum(self.kl_terms, 0))) * 1.15
        ch = self.pane(((0.10, 0.56), (0.40, 0.76)), (-0.5, 9.5), (0.0, top))
        self.axes(sc, ch, op, "digit", "nats", [(k, str(k)) for k in range(10)],
                  [(v, "%.1f" % v) for v in (0.0, 0.1, 0.2) if v < top], grid=True)
        bw = self.span(0.62 * (ch.x1 - ch.x0) / 10)[0]
        for k in range(10):
            g = smoothstep(0.04 + 0.02 * k, 0.24 + 0.02 * k, u)
            h, kl = self.h_terms[k] * g, self.kl_terms[k] * smoothstep(0.25, 0.45, u)
            yb, yh = ch.fy(0.0), ch.fy(float(h))
            hh = self.span(0, yb - yh)[1]
            if hh > 0.5:
                self.rect(sc, self.P(ch.fx(k), (yb + yh) / 2), bw, hh, ENT, op, radius=2,
                          lift=0.3)
            yk = ch.fy(float(h + kl))
            hk = self.span(0, abs(yh - yk))[1]
            if hk > 0.5:
                self.rect(sc, self.P(ch.fx(k), (yh + yk) / 2), bw * (1 if kl > 0 else 0.55),
                          hk, KLI, op, radius=2, lift=0.5, fill_op=1.0 if kl > 0 else 0.85)
            cap = op * smoothstep(0.45, 0.55, u)
            yc = ch.fy(float(self.ce_terms[k]))
            self.polyline(sc, [self.P(ch.fx(k) - 0.022, yc, 0.9), self.P(ch.fx(k) + 0.022, yc,
                                                                          0.9)],
                          2.6, CE, cap, bias=0.3)
        self.key(sc, 0.10, 0.845, [("−p ln p: entropy term", ENT),
                                   ("p ln(p/q): KL term", KLI)], op, size=TEXT_S, step=0.04)
        self.key(sc, 0.36, 0.845, [("−p ln q: their sum", CE)], op * smoothstep(0.45, 0.55, u),
                 size=TEXT_S, step=0.04)
        self.text(sc, 0.36, 0.885, "a KL term is below 0 where q > p",
                  op * smoothstep(0.35, 0.45, u), anchor="lm", size=TEXT_S, ink=dark(KLI, 0.45))
        x = 0.60
        self.reveal(sc, x, 0.40, [
            ("H(p) = Σ −p ln p = %.3f" % sh["entropy"], dark(ENT, 0.3), SMALL),
            ("KL(p‖q) = Σ p ln(p/q) = %.3f" % sh["kl"], dark(KLI, 0.45), SMALL),
            ("CE = Σ −p ln q = %.3f" % sh["cross_entropy"], dark(CE, 0.3), SMALL),
            ("= %.3f + %.3f" % (sh["entropy"], sh["kl"]), dark(CE, 0.3), BIG)], u, op,
            at=0.5, every=0.06, step=0.05)
        k6 = int(np.argmin(self.q))
        self.reveal(sc, x, 0.64, [
            ("q(%d) = %s, so −%.2f ln q = %.3f" % (k6, tiny(self.q[k6]), self.p[k6],
                                                   self.ce_terms[k6]), None, TEXT_S),
            ("%s of the CE is the %.2f on wrong digits:" % (pct(sh["wrong_share"]),
                                                            self.p[other]), None, TEXT_S),
            ("q gave them almost nothing", None, TEXT_S)], u, op, at=0.72, every=0.06,
            step=0.044)
        self.tag(sc, EXACT, op)

    def phase_3(self, sc, u, op):
        r = self.rows
        self.text(sc, 0.05, 0.315, "three targets p, one model q: mean over %s test digits"
                  % commas(self.meta["n"]), op, anchor="lm", strong=True, size=HEAD)
        x0, scale = 0.25, 125.0
        late = op * smoothstep(0.55, 0.65, u)
        for i, t in enumerate(S.TARGETS):
            y = 0.41 + 0.10 * i
            row = r[t]
            g = smoothstep(0.05 + 0.12 * i, 0.35 + 0.12 * i, u)
            self.text(sc, x0 - 0.015, y, NAMES[t], op, anchor="rm", size=SMALL)
            # the untrained network: the same H, a longer KL, drawn as a track behind
            w0 = row["untrained_cross_entropy"] * scale
            self.rect(sc, self.P(x0 + self.frac(w0)[0] / 2, y), w0, 22.0, GREY, late,
                      radius=3, lift=0.1, fill_op=0.25)
            self.stacked(sc, x0, y, row["entropy"], row["kl"], scale, op, grow=g)
            end = x0 + self.frac(max(w0, row["cross_entropy"] * scale))[0] + 0.012
            shown = op * smoothstep(0.3 + 0.12 * i, 0.4 + 0.12 * i, u)
            self.text(sc, end, y - 0.016, "%.3f + %.3f = %.3f" % (
                row["entropy"], row["kl"], row["cross_entropy"]), shown, anchor="lm",
                size=TEXT_S, ink=dark(CE, 0.3))
            self.text(sc, end, y + 0.022, "untrained: %.3f" % row["untrained_cross_entropy"],
                      late, anchor="lm", size=0.026, ink=dark(GREY, 0.2))
        self.key(sc, 0.10, 0.72, [("H(p): the target's own entropy", ENT),
                                  ("KL(p‖q): the part training can change", KLI),
                                  ("the same network before training", soft(GREY, 0.3))], op,
                 size=TEXT_S, step=0.046)
        self.reveal(sc, 0.58, 0.72, [
            ("training moved only the KL part:", None, TEXT_S),
            ("one-hot  %.2f → %.3f" % (r["one-hot"]["untrained_kl"], r["one-hot"]["kl"]),
             dark(KLI, 0.45), TEXT_S),
            ("smoothed  %.2f → %.3f" % (r["smoothed"]["untrained_kl"], r["smoothed"]["kl"]),
             dark(KLI, 0.45), TEXT_S),
            ("teacher  %.2f → %.3f" % (r["teacher"]["untrained_kl"], r["teacher"]["kl"]),
             dark(KLI, 0.45), TEXT_S)], u, op, at=0.62, every=0.06, step=0.04)
        self.text(sc, 0.05, 0.885, "teacher: the same network from seed %d, logits ÷ %g"
                  % (self.meta["teacher_seed"], self.meta["teacher_temperature"]),
                  op * smoothstep(0.4, 0.5, u), anchor="lm", size=TEXT_S)
        self.tag(sc, HELD, op)

    def phase_4(self, sc, u, op):
        r = self.rows
        self.text(sc, 0.05, 0.315, "which way round: KL(p‖q) against KL(q‖p)", op,
                  anchor="lm", strong=True, size=HEAD)
        x0, scale, room = 0.25, 300.0, 330.0
        for i, t in enumerate(S.TARGETS):
            y = 0.40 + 0.115 * i
            row = r[t]
            self.text(sc, x0 - 0.015, y + 0.02, NAMES[t], op, anchor="rm", size=SMALL)
            g = smoothstep(0.05 + 0.1 * i, 0.3 + 0.1 * i, u)
            self.hbar(sc, x0, y, row["kl"] * scale, FWD, op, "", "%.3f" % row["kl"], u=g,
                      thick=15.0)
            g2 = smoothstep(0.2 + 0.1 * i, 0.45 + 0.1 * i, u)
            if row["reverse_kl"] is None:
                self.hbar(sc, x0, y + 0.04, room, REV, op, "", "∞", u=g2, thick=15.0)
            else:
                self.hbar(sc, x0, y + 0.04, row["reverse_kl"] * scale, REV, op, "",
                          "%.3f" % row["reverse_kl"], u=g2, thick=15.0)
        self.key(sc, 0.25, 0.74, [("forward, KL(p‖q) = Σ p ln(p/q): weighed by p", FWD),
                                  ("reverse, KL(q‖p) = Σ q ln(q/p): weighed by q", REV)], op,
                 size=TEXT_S, step=0.048)
        self.reveal(sc, 0.25, 0.85, [
            ("one-hot: q > 0 on every wrong digit where p = 0, and ln(q/0) = ∞",
             dark(REV, 0.3), TEXT_S)], u, op, at=0.55)
        sh = self.shown
        self.reveal(sc, 0.70, 0.40, [
            ("test digit %d, smoothed:" % sh["digit"], None, TEXT_S),
            ("KL(p‖q) = %.3f" % sh["kl"], dark(FWD, 0.3), TEXT_S),
            ("KL(q‖p) = %.3f" % sh["reverse_kl"], dark(REV, 0.3), TEXT_S)], u, op, at=0.65,
            every=0.06, step=0.048)
        self.tag(sc, HELD, op)

    def phase_5(self, sc, u, op):
        hm = self.hm
        f, b = hm["forward"], hm["reverse"]
        top = float(max(self.curves.max(), max(c.max() for c in self.counts.values()))) * 1.1
        ch = self.pane(((0.08, 0.58), (0.36, 0.76)), self.xr, (0.0, top))
        ticks = [(v, nats(v, 0)) for v in range(int(np.ceil(self.xr[0] / 2) * 2),
                                                int(self.xr[1]) + 1, 2)]
        self.axes(sc, ch, op, "a digit along (average 0 − average 1)", "density", ticks,
                  [(0, "0")], grid=False)
        grow = smoothstep(0.02, 0.2, u)
        self.bars(sc, ch, self.edges, self.counts[1], ONES, op, fill=0.45, grow=grow)
        self.bars(sc, ch, self.edges, self.counts[0], ZEROS, op, fill=0.45, grow=grow)
        inside = (self.x >= self.xr[0]) & (self.x <= self.xr[1])
        xs = self.x[inside]
        self.path(sc, ch, xs, self.curves[0][inside], DIGIT_INK, op * smoothstep(0.1, 0.2, u),
                  width=1.8)
        gf, gr = smoothstep(0.25, 0.45, u), smoothstep(0.5, 0.7, u)
        self.path(sc, ch, xs, self.curves[1][inside], FWD, op, go=gf, width=2.8)
        self.path(sc, ch, xs, self.curves[2][inside], REV, op, go=gr, width=2.8)
        self.grid(sc, self.direction, 0.535, 0.42, 46, 46, op, lines=0.0)
        self.text(sc, 0.535, 0.49, "avg 0 − avg 1", op, size=0.026)
        self.key(sc, 0.25, 0.39, [("training 1s (%s)" % commas(hm["n"]["1"]), ONES),
                                  ("training 0s (%s)" % commas(hm["n"]["0"]), ZEROS),
                                  ("p: both, smoothed", DIGIT_INK)], op, size=TEXT_S,
                 step=0.04)
        x = 0.62
        self.reveal(sc, x, 0.36, [
            ("minimise KL(p‖q): cover p", dark(FWD, 0.3), SMALL),
            ("centre %s, width %.2f" % (nats(f["mu"], 2), f["sd"]), None, TEXT_S),
            ("centred where p is %s of its peak" % pct(f["p_at_centre"]), None, TEXT_S)],
            u, op, at=0.28, every=0.05, step=0.042)
        self.reveal(sc, x, 0.52, [
            ("minimise KL(q‖p): stay inside p", dark(REV, 0.3), SMALL),
            ("centre %s, width %.2f: on the 1s" % (nats(b["mu"], 2), b["sd"]), None, TEXT_S),
            ("its mass on the 0s: %s" % pct(b["mass_right"]), None, TEXT_S)],
            u, op, at=0.53, every=0.05, step=0.042)
        self.reveal(sc, x, 0.68, [
            ("each fit scored both ways (nats):", None, TEXT_S),
            ("forward fit: KL(p‖q) %.2f, KL(q‖p) %.1f" % (f["kl_pq"], f["kl_qp"]),
             dark(FWD, 0.3), TEXT_S),
            ("reverse fit: KL(p‖q) %.1f, KL(q‖p) %.2f" % (b["kl_pq"], b["kl_qp"]),
             dark(REV, 0.3), TEXT_S)], u, op, at=0.75, every=0.05, step=0.048)
        self.tag(sc, HUMPS % commas(self.n_humps), op)

    def phase_6(self, sc, u, op):
        r = self.rows
        self.text(sc, 0.05, 0.315, "next: where KL comes back on the page", op, anchor="lm",
                  strong=True, size=HEAD)
        x0, scale = 0.06, 110.0
        for i, t in enumerate(S.TARGETS):
            y = 0.44 + 0.09 * i
            self.text(sc, x0, y - 0.036, NAMES[t], op, anchor="lm", size=0.026)
            end = self.stacked(sc, x0, y, r[t]["entropy"], r[t]["kl"], scale, op, thick=14.0)
            self.text(sc, end + 0.008, y, "%.3f" % r[t]["cross_entropy"], op, anchor="lm",
                      size=0.026, ink=dark(CE, 0.3))
        self.key(sc, x0, 0.72, [("H(p)", ENT), ("KL(p‖q)", KLI)], op, size=0.026, step=0.048)
        rows = (
            ("Dropout, noise, max-norm, label smoothing",
             "a smoothed target: CE = %.3f + KL, never below H(p)" % r["smoothed"]["entropy"],
             KLI),
            ("Distillation", "KL(soft teacher‖student): %.3f here at T = %g"
             % (r["teacher"]["kl"], self.meta["teacher_temperature"]), KLI),
            ("The KL term of a VAE", "KL(encoder's bell‖prior): the fitted bell first", REV),
            ("t-SNE and UMAP · RLHF with PPO · DPO",
             "KL(P‖Q) for t-SNE; KL(policy‖reference) for RLHF", GREY))
        for i, (title, line, ink) in enumerate(rows):
            a = op * smoothstep(0.08 + 0.12 * i, 0.16 + 0.12 * i, u)
            y = 0.40 + 0.115 * i
            self.text(sc, 0.36, y, title, a, anchor="lm", strong=True, size=SMALL)
            self.text(sc, 0.36, y + 0.042, line, a, anchor="lm", size=TEXT_S, ink=dark(ink, 0.4))
        self.tag(sc, HELD, op)


ANIMATIONS = {"kl-divergence": KLDivergence}

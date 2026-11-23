"""Animation 181 of the "next" catalogue (see :mod:`voxanim.mech`): the bridge from
training a classifier on labels (Part 5) to learning from reward (Part 6).

:class:`LabelsAgainstRewards`: the same 784-64-10 classifier trained from labels and from
a right-or-wrong reward. One real training digit's ten probabilities and the two
gradients on its logits, ``p - onehot(y)`` against ``(r - b)(p - onehot(a))``; many draws
of the reward's, whose average is exactly ``p(y)`` times the label's; the reward's entropy
through training, at most 1 bit against the ``log2 10`` a label can carry; held-out
accuracy of both in three seeds, and the updates each needs; each digit's accuracy, with
the digits a rewarded run stopped naming. It closes on the next animation, Q-learning in
a gridworld.

It draws :mod:`voxanim.study_bridge_reward`'s study: every number on screen is the
study's. Every phase carries, bottom right, what kind of evidence it shows.
"""

from __future__ import annotations

import numpy as np

from . import study_bridge_reward as S
from .easing import smoothstep
from .mech import AMBER, AXIS, BLUE, DIM, GREEN, OCHRE, RED, VIOLET, dark, soft
from .mech_info_nets import Info, fixed, span_of
from .mech_phys_apps import GREY, commas, pct

EXACT = "exact arithmetic on one digit's measured probabilities"
RUNS = "measured on MNIST: three seeds, 10,000 held-out digits"
TRAIN = "measured on MNIST: the rewards of three training runs"
ONE_RUN = "measured on MNIST: one seed, 10,000 held-out digits"
BIG, HEAD, SMALL, TEXT_S = 0.036, 0.033, 0.030, 0.029
LABEL, REWARD = BLUE, VIOLET             # trained from labels; from rewards
RIGHT, WRONG = GREEN, RED                # a draw that names the label; one that does not
FROM = "A dense network classifies digits"
NEXT = "Q-learning in a gridworld"


def is_evidence(text):
    """True for the tag every phase carries."""
    return text.startswith(("exact arithmetic on ", "measured on MNIST"))




def stall_rows(stalls, most=4):
    """The longest ``most`` stalls as ``(digit, since, until)``, longest first."""
    order = sorted(stalls, key=lambda d: (d["since"] - d["until"], d["digit"]))
    return [(d["digit"], d["since"], d["until"]) for d in order[:most]]


class LabelsAgainstRewards(Info):
    """One digit and its two gradients; many draws and their exact average; the
    reward's entropy through training; held-out accuracy and the gap; each digit's
    accuracy and the digits a rewarded run stopped naming; next, Q-learning."""

    TITLE = "Labels against rewards"
    PHASES = (("two signals", 12.0), ("on average", 12.0), ("bits", 11.0),
              ("held out", 12.0), ("never named", 12.0))
    COLUMNS_TO, EVERY = 3000, 50         # the per-digit pictures: updates 0..3000 in steps

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (pnl.data for pnl in s.panels)
        sh = self.shown = m["shown"]
        self.p = np.asarray(sh["probabilities"], float)
        self.y = int(sh["label"])
        self.b = float(sh["baseline"])
        self.image = np.asarray(p0["image"], float)
        self.g_label = np.asarray(sh["label_gradient"], float)
        self.g_draws = np.asarray(sh["draw_gradients"], float)
        self.draws = [int(v) for v in sh["draws"]]
        self.g_mean = np.asarray(sh["mean_gradient"], float)
        self.g_exp = np.asarray(sh["expected_gradient"], float)
        # the two draws phase 1 follows: the first right one and the first wrong one
        right = [i for i, a in enumerate(self.draws) if a == self.y]
        wrong = [i for i, a in enumerate(self.draws) if a != self.y]
        self.follow = (right[:1] + wrong[:1]) or [0]
        info = m["information"]
        self.ends = np.asarray(info["ends"], float)
        self.H = np.asarray(p2["ys"], float)                 # (seeds, windows)
        self.rate = np.asarray(p2["rate"], float)
        self.probes = np.asarray(p3["x"], float)
        n = len(m["seeds"])
        acc = np.asarray(p3["ys"], float)
        self.acc_label, self.acc_reward = acc[:n], acc[n:]
        # each digit's accuracy at even steps of updates: the last probe at or before each
        values = np.asarray(p1["values"], float)              # rewards' 10 rows, labels'
        self.cols = np.arange(0, self.COLUMNS_TO + 1, self.EVERY)
        at = np.searchsorted(self.probes, self.cols, side="right") - 1
        self.cls_reward, self.cls_label = values[:10][:, at], values[10:][:, at]
        self.seed = m["stalled_seed"]
        self.stalls = stall_rows(m["stalls"]["reward"][m["seeds"].index(self.seed)])

    # ------------------------------------------------------------------ words
    def reach(self, level, kind):
        return self.meta["reach"]["%.2f" % level][kind]

    def reach_span(self, level, kind):
        v = self.reach(level, kind)
        hit = [x for x in v if x is not None]
        if not hit:
            return "never"
        missed = len(v) - len(hit)
        return span_of(hit, commas) + (" (%d never)" % missed if missed else "")

    def gap_span(self):
        g = [v for v in self.meta["gap_90"] if v is not None]
        return span_of(g, lambda v: "%.1f" % v) if g else ""

    def final(self, kind):
        return span_of(self.meta["final_accuracy"][kind], lambda v: pct(v, 1))

    def stall_words(self):
        if not self.stalls:
            return "no rewarded run stopped naming a digit"
        d, a, b = self.stalls[0]
        return "in seed %d no %ds were named from update %s to %s" % (self.seed, d, commas(a),
                                                                     commas(b))

    def subtitle_for(self, k):
        m, sh = self.meta, self.shown
        info = m["information"]
        gap = self.gap_span()
        held = ("on held-out digits the reward needs %s times the updates to reach 90%%"
                % gap if gap else "on held-out digits the reward never reaches 90%")
        return [
            "the same classifier, two ways to learn. A label names the digit: gradient "
            "p − onehot(y). A reward only says whether a guess drawn from p was right: "
            "(r − b)(p − onehot(a)), the label never shown",
            "a right guess gives the label's gradient times (1 − b); a wrong one only "
            "pushes that guess down. On average the two point the same way, exactly: the "
            "reward's is p(y) = %.2f times the label's" % sh["p_label"],
            "a label can carry log₂ 10 = %.2f bits. A right/wrong reward carries at most "
            "1: measured, %s bits at its peak, and %s once %s of guesses are right"
            % (info["label_bits_max"], span_of(info["peak"], lambda v: "%.2f" % v),
               span_of(info["final_entropy"], lambda v: "%.2f" % v),
               span_of(info["final_rate"], lambda v: pct(v))),
            "%s, and after %s updates it is at %s against the labels' %s"
            % (held, commas(m["updates"]), self.final("reward"), self.final("label")),
            "%s. A digit never guessed is never rewarded: its push p(y) is near 0. "
            "Next: rewards that come late, in a gridworld" % self.stall_words(),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def tenbars(self, sc, box, values, lim, op, inks, grow=1.0, mark=(), ylabel="",
                ticks=None, fmt=fixed):
        """Ten vertical bars from zero (or the bottom) in the layout box, ``lim``: the y
        range; ``mark``: the digits whose value is written at their bar's end."""
        ch = self.pane(box, (-0.5, 9.5), lim)
        self.axes(sc, ch, op, "", ylabel, [(k, str(k)) for k in range(10)], ticks,
                  grid=False)
        if lim[0] < 0 < lim[1]:
            self.hline(sc, ch, 0.0, AXIS, op, width=1.2, dashed=False)
        base = max(lim[0], 0.0)
        for k, v in enumerate(values):
            v = float(v) * grow
            y0, y1 = ch.fy(base), ch.fy(v)
            w, h = self.span(0.62 * (ch.x1 - ch.x0) / 10, abs(y1 - y0))
            ink = inks[k] if isinstance(inks, list) else inks
            self.rect(sc, self.P(ch.fx(k), (y0 + y1) / 2), w, max(h, 1.0), ink, op,
                      radius=2, lift=0.4)
            if k in mark and grow > 0.5:
                off = -0.03 if v >= 0 else 0.03
                self.text(sc, ch.fx(k), y1 + off, fmt(v), op, size=TEXT_S,
                          ink=dark(ink, 0.35))
        return ch

    @staticmethod
    def grad_inks(values, ink):
        return [ink if abs(v) > 1e-9 else DIM for v in values]

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        sh = self.shown
        self.text(sc, 0.05, 0.305, "from: %s, trained on a label for every digit" % FROM, op,
                  anchor="lm", ink=dark(OCHRE, 0.35), size=SMALL)
        self.digit(sc, self.image, 0.09, 0.415, 62, op, frame=soft(GREY, 0.3))
        self.lines(sc, 0.145, 0.385, ["training digit %s: a %d" % (commas(sh["index"]), self.y),
                                      "the network after %d" % sh["update"],
                                      "rewarded updates (seed 0)"], op, size=TEXT_S,
                   step=0.04)
        inks = [GREY] * 10
        inks[self.y] = RIGHT
        self.tenbars(sc, ((0.07, 0.31), (0.60, 0.82)), self.p, (0.0, 1.0), op, inks,
                     grow=smoothstep(0.02, 0.15, u), mark=(self.y,), ylabel="p",
                     ticks=[(0, "0"), (0.5, "0.5"), (1, "1")], fmt=lambda v: "%.2f" % v)
        self.text(sc, 0.19, 0.88, "the softmax p: ten probabilities", op, size=TEXT_S)
        # labels
        on = op * smoothstep(0.12, 0.22, u)
        self.text(sc, 0.38, 0.37, "labels: the label says %d" % self.y, on, anchor="lm",
                  ink=dark(LABEL, 0.3), size=HEAD)
        ticks = [(-1, "−1"), (0, "0"), (1, "1")]
        self.tenbars(sc, ((0.41, 0.63), (0.48, 0.82)), self.g_label, (-1.0, 1.0), on,
                     self.grad_inks(self.g_label, LABEL), grow=smoothstep(0.15, 0.3, u),
                     mark=(self.y,), ticks=ticks)
        self.text(sc, 0.52, 0.88, "gradient p − onehot(%d)" % self.y, on, size=TEXT_S,
                  ink=dark(LABEL, 0.3))
        # rewards: a right draw, then a wrong one
        on = op * smoothstep(0.32, 0.42, u)
        second = len(self.follow) > 1 and u >= 0.7
        i = self.follow[1 if second else 0]
        a = self.draws[i]
        r = int(a == self.y)
        dip = 1 - smoothstep(0.62, 0.68, u) * (1 - smoothstep(0.7, 0.76, u)) \
            if len(self.follow) > 1 else 1.0
        self.text(sc, 0.69, 0.37, "rewards: draw a guess a from p", on, anchor="lm",
                  ink=dark(REWARD, 0.3), size=HEAD)
        ink = RIGHT if r else WRONG
        self.text(sc, 0.69, 0.42, "a = %d: %s, r = %d; b = %.2f" % (
            a, "right" if r else "wrong", r, self.b), on * dip, anchor="lm",
            ink=dark(ink, 0.3), size=TEXT_S)
        g = self.g_draws[i]
        self.tenbars(sc, ((0.72, 0.94), (0.48, 0.82)), g, (-1.0, 1.0), on * dip,
                     self.grad_inks(g, REWARD), grow=smoothstep(0.42, 0.55, u),
                     mark=(self.y, a), ticks=ticks)
        self.text(sc, 0.83, 0.88, "(r − b)(p − onehot(a))", on, size=TEXT_S,
                  ink=dark(REWARD, 0.3))
        self.tag(sc, EXACT, op)

    def phase_2(self, sc, u, op):
        sh = self.shown
        n = len(self.draws)
        self.text(sc, 0.05, 0.305, "%d draws of a guess for the same digit (a %d)"
                  % (n, self.y), op, anchor="lm", strong=True, size=SMALL)
        lim = float(np.abs(self.g_draws).max())
        w, h = 190.0, 20.0 * n
        grow = smoothstep(0.03, 0.35, u)
        cell = self.grid(sc, self.g_draws, 0.17, 0.375 + self.frac(0, h)[1] / 2 + 0.012, w,
                         h, op, reveal=grow, vmax=lim, hot=RED, cold=BLUE)
        for k in range(10):
            self.text(sc, cell(0, k)[0], 0.355, str(k), op, size=TEXT_S)
        for j in range(int(np.ceil(n * grow - 1e-9))):
            a = self.draws[j]
            right = a == self.y
            x, yy = cell(j, 9)
            self.text(sc, x + 0.03, yy, "a = %d, %s" % (a, "right" if right else "wrong"),
                      op, anchor="lm", size=0.022, ink=dark(RIGHT if right else WRONG, 0.3))
        self.lines(sc, 0.05, 0.845, ["a row: (r − b)(p − onehot(a)), b = %.2f" % self.b,
                                     "blue: that logit is pushed up; red: down"], op,
                   size=TEXT_S, step=0.04)
        # the average against the label's
        on = op * smoothstep(0.36, 0.46, u)
        self.text(sc, 0.47, 0.36, "mean of %s draws" % commas(sh["many"]), on, anchor="lm",
                  ink=dark(REWARD, 0.3), size=SMALL)
        self.text(sc, 0.73, 0.36, "p(%d) × the label's" % self.y, on, anchor="lm",
                  ink=dark(LABEL, 0.3), size=SMALL)
        ticks = [(-0.3, "−0.3"), (0, "0"), (0.3, "0.3")]
        lim2 = (-0.45, 0.3)
        self.tenbars(sc, ((0.50, 0.68), (0.41, 0.63)), self.g_mean, lim2, on,
                     self.grad_inks(self.g_mean, REWARD), mark=(self.y,), ticks=ticks)
        self.tenbars(sc, ((0.76, 0.94), (0.41, 0.63)), self.g_exp, lim2, on,
                     self.grad_inks(self.g_exp, LABEL), mark=(self.y,), ticks=ticks)
        self.reveal(sc, 0.47, 0.715, [
            ("E[(r − b)(p − onehot(a))] = p(y)(p − onehot(y))", None, SMALL),
            ("= %.3f × the label's gradient, for any b" % sh["p_label"], dark(LABEL, 0.3),
             SMALL),
            ("cosine with the label's: 1 if right, %.2f if wrong" % sh["wrong_cosine"],
             None, SMALL),
            ("p(y) near 0: no push where it is wrong", dark(WRONG, 0.3), SMALL)],
            u, op, at=0.5, every=0.08, step=0.042)
        self.tag(sc, EXACT, op)

    def phase_3(self, sc, u, op):
        m = self.meta
        info = m["information"]
        go = smoothstep(0.05, 0.5, u)
        ch = self.pane(((0.10, 0.52), (0.36, 0.70)), (self.ends[0], self.ends[-1]), (0.0, 3.6),
                       xlog=True)
        self.axes(sc, ch, op, "updates (log scale)", "bits per digit",
                  [(v, commas(v)) for v in (100, 1000, 10000)],
                  [(v, "%g" % v) for v in (1, 2, 3)])
        self.hline(sc, ch, info["label_bits_max"], LABEL, op, width=2.0, dashed=False)
        self.text(sc, ch.x1, ch.fy(info["label_bits_max"]) - 0.03,
                  "a label: log₂ 10 = %.2f bits" % info["label_bits_max"], op, anchor="rm",
                  ink=dark(LABEL, 0.3), size=TEXT_S)
        self.hline(sc, ch, 1.0, GREY, op, width=1.6, dashed=True)
        self.text(sc, ch.fx(150), ch.fy(1.0) - 0.032, "a right/wrong reward: at most 1 bit",
                  op, anchor="lm", ink=dark(GREY, 0.3), size=TEXT_S)
        for k in range(len(self.H)):
            self.path(sc, ch, self.ends, self.H[k], soft(REWARD, 0.3 * k) if k else REWARD,
                      op, go, width=2.4 if k == 0 else 1.6, dot=k == 0)
        self.lines(sc, 0.05, 0.835, [
            "H(reward) = −q log₂ q − (1 − q) log₂(1 − q), q: the share right",
            "each point: %d updates of %d digits; seed 0 dark" % (info["window"], m["batch"])],
            op, size=TEXT_S, step=0.04)
        # the binary entropy, with the run moving along it
        ch2 = self.pane(((0.64, 0.92), (0.38, 0.60)), (0.0, 1.0), (0.0, 1.05))
        self.axes(sc, ch2, op, "q: share of guesses right", "H(reward), bits",
                  [(0, "0"), (0.5, "50%"), (1, "100%")], [(0, "0"), (0.5, "0.5"), (1, "1")])
        q = np.linspace(0, 1, 101)
        self.path(sc, ch2, q, S.binary_entropy(q), GREY, op, width=1.6)
        k = int(round(go * (len(self.ends) - 1)))
        qn, hn = self.rate[0, k], self.H[0, k]
        self.disk(sc, self.P(ch2.fx(qn), ch2.fy(hn)), 6.0, REWARD, op, glow=0.4, lift=1.4)
        self.text(sc, 0.60, 0.705, "seed 0, update %s: %s right, %.2f bits"
                  % (commas(self.ends[k]), pct(qn), hn), op, anchor="lm", size=TEXT_S,
                  ink=dark(REWARD, 0.3))
        # all the bits of a run: what its labels could carry, what its rewards did
        late = op * smoothstep(0.3, 0.4, u)
        grow = smoothstep(0.32, 0.5, u)
        self.text(sc, 0.60, 0.755, "bits in the %s digits of a run" % commas(m["examples"]),
                  late, anchor="lm", size=TEXT_S, strong=True)
        track, most = 170.0, info["label_total_bits"]
        self.hbar(sc, 0.70, 0.80, track, LABEL, late, "labels", "%.2fM" % (most / 1e6),
                  u=grow, thick=14.0)
        self.hbar(sc, 0.70, 0.845, track * max(info["reward_total_bits"]) / most, REWARD, late,
                  "rewards", "%sM" % span_of(info["reward_total_bits"],
                                             lambda v: "%.2f" % (v / 1e6)), u=grow, thick=14.0)
        self.tag(sc, TRAIN, op)

    def phase_4(self, sc, u, op):
        m = self.meta
        go = smoothstep(0.04, 0.55, u)
        x = self.probes[1:]
        ch = self.pane(((0.10, 0.54), (0.36, 0.78)), (x[0], x[-1]), (0.0, 1.0), xlog=True)
        self.axes(sc, ch, op, "updates of %d digits (log scale)" % m["batch"],
                  "held-out accuracy", [(v, commas(v)) for v in (10, 100, 1000, 10000)],
                  [(v, pct(v)) for v in (0.5, 0.9, 1.0)])
        self.hline(sc, ch, 0.9, GREY, op * smoothstep(0.5, 0.6, u), width=1.4, dashed=True)
        for k in range(len(self.acc_label)):
            for runs, ink in ((self.acc_label, LABEL), (self.acc_reward, REWARD)):
                self.path(sc, ch, x, runs[k][1:], soft(ink, 0.3 * k) if k else ink, op, go,
                          width=2.4 if k == 0 else 1.6, dot=k == 0)
        self.key(sc, 0.36, 0.665, [("labels, three seeds", LABEL),
                                   ("rewards, three seeds", REWARD)], op, size=TEXT_S,
                 step=0.04)
        self.text(sc, 0.60, 0.36, "updates to reach", op, anchor="lm", strong=True,
                  size=SMALL)
        self.text(sc, 0.785, 0.36, "labels", op, ink=dark(LABEL, 0.3), size=SMALL)
        self.text(sc, 0.895, 0.36, "rewards", op, ink=dark(REWARD, 0.3), size=SMALL)
        for i, level in enumerate(m["levels"]):
            on = op * smoothstep(0.5 + 0.08 * i, 0.58 + 0.08 * i, u)
            yy = 0.415 + 0.05 * i
            self.text(sc, 0.60, yy, pct(level), on, anchor="lm", size=SMALL)
            self.text(sc, 0.785, yy, self.reach_span(level, "label"), on, size=TEXT_S)
            self.text(sc, 0.895, yy, self.reach_span(level, "reward"), on, size=TEXT_S)
        late = op * smoothstep(0.76, 0.84, u)
        gap = self.gap_span()
        self.lines(sc, 0.60, 0.60, [
            ("to 90%%: %s times the updates" % gap if gap else "rewards never reach 90%",
             dark(REWARD, 0.3)),
            "after %s updates:" % commas(m["updates"]),
            "rewards %s, labels %s" % (self.final("reward"), self.final("label")),
            "a batch's reward gradient against its",
            "labels': cosine %s on average" % span_of(m["batch_cosine"]["mean"],
                                                     lambda v: "%.2f" % v)],
            late, size=TEXT_S, step=0.04)
        self.tag(sc, RUNS, op)

    def phase_5(self, sc, u, op):
        m = self.meta
        w, h = 170.0, 236.0
        nrow = len(self.cols)
        self.text(sc, 0.05, 0.305, "seed %d: each digit's held-out accuracy, updates 0 to %s"
                  % (self.seed, commas(self.COLUMNS_TO)), op, anchor="lm", strong=True,
                  size=SMALL)
        lost = {d for d, _, _ in self.stalls}
        top = 0.415
        fy = top + self.frac(0, h)[1] / 2
        for col, (vals, ink, word, fx) in enumerate((
                (self.cls_reward, REWARD, "rewards", 0.19),
                (self.cls_label, LABEL, "labels", 0.405))):
            on = op * (1.0 if col == 0 else smoothstep(0.1, 0.2, u))
            self.text(sc, fx, 0.355, word, on, ink=dark(ink, 0.3), size=SMALL)
            cell = self.grid(sc, vals.T, fx, fy, w, h, on, vmax=1.0, hot=GREEN)
            for k in range(10):
                self.text(sc, cell(0, k)[0], top - 0.022, str(k), on, size=0.022,
                          ink=dark(AMBER, 0.4) if (k in lost and col == 0) else None)
            if col == 0:
                ring = op * smoothstep(0.25, 0.35, u)
                rh = self.frac(0, h / nrow)[1]
                for d, a, b in self.stalls:
                    r0 = int(np.searchsorted(self.cols, a))
                    r1 = max(int(np.searchsorted(self.cols, b, side="right")) - 1, r0)
                    x0, y0 = cell(r0, d)
                    _, y1 = cell(r1, d)
                    _, ph = self.span(0, y1 - y0 + rh)
                    self.rect(sc, self.P(x0, (y0 + y1) / 2), w / 10 + 2, ph + 2, AMBER, ring,
                              border=AMBER, border_w=2.0, radius=2, lift=0.7, fill_op=0.0)
                for v in (0, 1000, 2000, 3000):
                    yy = cell(int(np.searchsorted(self.cols, v)), 0)[1]
                    self.text(sc, fx - self.frac(w / 2)[0] - 0.012, yy, commas(v), op,
                              anchor="rm", size=0.022)
        self.text(sc, 0.30, top + self.frac(0, h)[1] + 0.035, "down: updates; green: named "
                  "right", op, size=0.024)
        rows = []
        if self.stalls:
            rows.append(("rewards: digits no guess named", dark(AMBER, 0.4), SMALL))
            for d, a, b in self.stalls:
                rows.append(("%ds: from update %s to %s" % (d, commas(a), commas(b)),
                             dark(AMBER, 0.4), SMALL))
        else:
            rows.append(("no rewarded run lost a digit", dark(AMBER, 0.4), SMALL))
        rows += [("a digit never guessed is never rewarded:", None, SMALL),
                 ("its push p(y)(p − onehot(y)) is near 0", None, SMALL),
                 ("labels: every digit named, every seed" if not any(
                     m["stalls"]["label"]) else "labels lost a digit too", dark(LABEL, 0.3),
                  SMALL),
                 ("rewards lost a digit in %d of 3 seeds" % m["stalled_runs"], None, SMALL)]
        self.reveal(sc, 0.54, 0.36, rows, u, op, at=0.2, every=0.06, step=0.044)
        end = op * smoothstep(0.7, 0.8, u)
        self.text(sc, 0.54, 0.775, "next: %s" % NEXT, end, anchor="lm", strong=True, size=HEAD)
        self.lines(sc, 0.54, 0.822, ["there only rewards teach, and they come",
                                    "late, after many moves"], end, size=TEXT_S, step=0.04)
        self.tag(sc, ONE_RUN, op)


ANIMATIONS = {"labels-against-rewards": LabelsAgainstRewards}

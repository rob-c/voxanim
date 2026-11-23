"""Animation 180 of the "next" catalogue (see :mod:`voxanim.mech`): the bridge from a
counted Markov chain to a recurrent network, on Tiny Shakespeare.

* :class:`MarkovChainToRnn`: tables of counts of every order, scored on the text they
  were counted from and on held-out text; one held-out position whose long contexts were
  never counted; a small GRU trained on the same text, its held-out bits against the best
  table; where the two differ (held-out positions by how often their context was
  counted) and how big each is; and bits per character as perplexity.

It draws :mod:`voxanim.study_bridge_text`'s study: every number on screen is the study's.
Every phase says, bottom right, what kind of evidence it shows.
"""

from __future__ import annotations

import numpy as np

from .easing import smoothstep
from .losschart import sup
from .mech import AMBER, AXIS, BLUE, GREEN, TEAL, VIOLET, dark, soft
from .mech_phys_apps import GREY, Phys, commas, pct

BIG, HEAD, SMALL, TEXT_S = 0.036, 0.033, 0.030, 0.029
TABLE, RNN = AMBER, BLUE                   # the counted tables; the recurrent network
TRAIN_INK = soft(AMBER, 0.45)              # a table scored on its own training text
RNN_TRAIN = soft(BLUE, 0.5)
HELD = "measured on Tiny Shakespeare: %s held-out characters"
TRAINED = "measured on Tiny Shakespeare: the text each table was counted from"
EXACT = "exact arithmetic on the measured held-out bits"
FROM = "from: Measuring entropy, and A Markov chain from real text"
NEXT = "next: An RNN training step"


def is_evidence(text):
    """True for the tag every phase carries."""
    return text.startswith(("exact arithmetic on ", "measured on Tiny Shakespeare"))


def shown(s):
    """Characters as drawn: a space as ␣, a line break as ↵."""
    return str(s).replace(" ", "␣").replace("\n", "↵")


def bits(v):
    return "%.2f" % float(v)


class MarkovChainToRnn(Phys):
    """Counted contexts by order, on their own text and on held-out text; one held-out
    position's contexts; a GRU's held-out bits through training against the best table;
    bits by how often the context was counted, and the size of each model; perplexity."""

    TITLE = "From a Markov chain to an RNN"
    PHASES = (("counts", 11.0), ("held out", 12.0), ("a state", 13.0), ("why", 12.0),
              ("perplexity", 10.0))
    CHART = ((0.58, 0.93), (0.37, 0.72))
    STRIP = 18                           # held-out characters drawn before the guess

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.orders = np.asarray(p0["x"], float)
        self.held, self.train = (np.asarray(v, float) for v in p0["ys"])
        self.unseen = np.asarray(p0["unseen"], float)
        self.steps = np.asarray(p1["x"], float)
        self.rnn_held, self.rnn_train = (np.asarray(v, float) for v in p1["ys"])
        self.buckets = np.asarray(p2["values"], float)
        self.bucket_names = list(p2["labels"])
        self.perp = np.asarray(p3["values"], float)
        self.perp_bits = np.asarray(p3["bits"], float)
        self.perp_labels = list(p3["labels"])
        self.k = int(m["best_order"])
        self.best = float(m["best_bits"])
        self.top = int(self.orders[-1])
        self.ex = m["example"]
        self.n_held = commas(m["scored_positions"])
        # the example's contexts drawn: short, around the best order, and the longest
        self.example_orders = sorted({1, max(self.k - 1, 1), self.k, self.k + 1, self.top})

    # ------------------------------------------------------------------ words
    def subtitle_for(self, k):
        m = self.meta
        r = m["rnn"]
        return [
            "a chain of order k counts what followed every k characters of the training "
            "text. Scored on that same text, more context always looks better: %s bits per "
            "character at order 0, %s at order %d"
            % (bits(self.train[0]), bits(self.train[-1]), self.top),
            "on held-out text the same tables are best at order %d, %s bits, then get worse: "
            "at order %d, %s of held-out contexts were never counted, and the table knows "
            "nothing about them" % (self.k, bits(self.best), self.top, pct(self.unseen[-1])),
            "a recurrent network reads one character at a time and carries a state of %d "
            "numbers. Trained on the same text, its held-out bits fall below the best "
            "table's, to %s" % (r["hidden"], bits(m["rnn_final"])),
            "the table needs the exact context; the network shares %s parameters across "
            "all of them. Where the order-%d context was never counted: %s bits for the "
            "table, %s for the network" % (commas(r["parameters"]), self.k,
                                          bits(m["bucket_table_bits"][0]),
                                          bits(m["bucket_rnn_bits"][0])),
            "two to the bits is the perplexity: as hard as choosing among that many equally "
            "likely characters. Next: how one training step of such a network works",
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def order_chart(self, sc, op, go_train=1.0, go_held=0.0):
        ch = self.pane(self.CHART, (0, self.top), (0.0, 6.5))
        self.axes(sc, ch, op, "order k: characters of context",
                  "bits per character",
                  [(v, "%d" % v) for v in range(0, self.top + 1, 2)],
                  [(v, "%d" % v) for v in (0, 2, 4, 6)])
        self.path(sc, ch, self.orders, self.train, TRAIN_INK, op, go_train, width=2.4,
                  dot=go_train < 1)
        for i in range(int(go_train * (len(self.orders) - 1) + 1e-9) + 1):
            self.disk(sc, self.P(ch.fx(self.orders[i]), ch.fy(self.train[i])), 3.6,
                      TRAIN_INK, op, lift=1.2, outline=0.0)
        if go_held > 0:
            self.path(sc, ch, self.orders, self.held, TABLE, op, go_held, width=2.8,
                      dot=go_held < 1)
            for i in range(int(go_held * (len(self.orders) - 1) + 1e-9) + 1):
                self.disk(sc, self.P(ch.fx(self.orders[i]), ch.fy(self.held[i])), 4.2,
                          TABLE, op, lift=1.3, outline=0.0)
        return ch

    def strip(self, sc, fx, fy, op, k, width=0.47):
        """The held-out characters before the example position, the last ``k`` lit, then
        the character to guess."""
        chars = list(self.ex["before"][-self.STRIP:]) + ["?"]
        n = len(chars)
        step = width / n
        w = self.span(step)[0] - 3
        for i, c in enumerate(chars):
            x = fx + (i + 0.5) * step
            if i == n - 1:
                self.box(sc, x, fy, w, 26, soft(GREEN, 0.6), op, "?", border=GREEN,
                         size=TEXT_S)
                continue
            lit = i >= n - 1 - k
            self.box(sc, x, fy, w, 26, soft(TABLE, 0.55) if lit else soft(GREY, 0.8), op,
                     shown(c), border=TABLE if lit else None, size=TEXT_S,
                     ink=dark(TABLE, 0.45) if lit else None)

    def pair_key(self, sc, a, b, op):
        """A legend of two entries side by side under the chart."""
        self.key(sc, 0.58, 0.845, [a], op, size=TEXT_S)
        self.key(sc, 0.74, 0.845, [b], op, size=TEXT_S)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.05, 0.30, FROM, op, anchor="lm", ink=dark(TEAL, 0.25), size=SMALL)
        kk = int(np.clip(np.floor(u * 6), 0, self.k))
        self.text(sc, 0.05, 0.37, "order %d: the last %d characters pick a row of counts"
                  % (kk, kk), op, anchor="lm", strong=True, size=SMALL)
        self.strip(sc, 0.05, 0.43, op, kk)
        ctx = m["contexts"]
        rows = [("training text: %s characters" % commas(m["split_sizes"][0]), None, SMALL)]
        for o in (1, 2, self.k, self.top):
            rows.append(("order %d: %s contexts counted" % (o, commas(ctx[o])), None,
                         TEXT_S))
        y = 0.51
        for i, (s, ink, size) in enumerate(rows):
            a = op * smoothstep(0.15 + 0.07 * i, 0.22 + 0.07 * i, u)
            self.text(sc, 0.05, y, s, a, anchor="lm", ink=ink, size=size)
            y += 0.046
        go = smoothstep(0.25, 0.85, u)
        ch = self.order_chart(sc, op, go)
        end = op * smoothstep(0.85, 0.92, u)
        self.text(sc, ch.fx(0) + 0.01, ch.fy(self.train[0]) - 0.04, bits(self.train[0]), end,
                  anchor="lm", ink=dark(TABLE, 0.45), size=TEXT_S)
        self.text(sc, ch.fx(self.top), ch.fy(self.train[-1]) - 0.04, bits(self.train[-1]),
                  end, anchor="mm", ink=dark(TABLE, 0.45), size=TEXT_S)
        self.key(sc, 0.58, 0.845, [("scored on the text it was counted from", TRAIN_INK)],
                 op * smoothstep(0.3, 0.4, u), size=TEXT_S)
        self.tag(sc, TRAINED, op)

    def phase_2(self, sc, u, op):
        m, ex = self.meta, self.ex
        self.text(sc, 0.05, 0.30, "one held-out character to guess: %s" % shown(ex["next"]),
                  op, anchor="lm", strong=True, size=SMALL)
        self.strip(sc, 0.05, 0.365, op, self.top)
        self.text(sc, 0.05, 0.43, "its context", op, anchor="lm", size=TEXT_S, ink=GREY)
        self.text(sc, 0.27, 0.43, "counted", op, anchor="mm", size=TEXT_S, ink=GREY)
        self.text(sc, 0.40, 0.43, "table: bits", op, anchor="mm", size=TEXT_S, ink=GREY)
        for i, o in enumerate(self.example_orders):
            y = 0.48 + 0.05 * i
            a = op * smoothstep(0.08 + 0.08 * i, 0.15 + 0.08 * i, u)
            c = ex["counts"][o]
            ink = dark(TABLE, 0.45) if c else dark((0.8, 0.25, 0.25), 0.1)
            self.text(sc, 0.05, y, "%d: %s" % (o, shown(ex["contexts"][o])), a, anchor="lm",
                      size=TEXT_S)
            self.text(sc, 0.27, y, commas(c) if c else "never", a, anchor="mm", size=TEXT_S,
                      ink=ink)
            self.text(sc, 0.40, y, bits(ex["table_bits"][o]), a, anchor="mm", size=TEXT_S,
                      ink=ink)
        late = op * smoothstep(0.55, 0.62, u)
        self.lines(sc, 0.05, 0.75, [
            "never counted: all %d characters equally likely" % m["characters"],
            ("held-out contexts never counted:", dark(TABLE, 0.45)),
            ("%s at order %d, %s at order %d" % (pct(self.unseen[self.k]), self.k,
                                                pct(self.unseen[-1]), self.top),
             dark(TABLE, 0.45))], late, size=TEXT_S, step=0.042)
        go = smoothstep(0.15, 0.7, u)
        ch = self.order_chart(sc, op, 1.0, go)
        best = op * smoothstep(0.7, 0.78, u)
        self.text(sc, ch.fx(self.k), ch.fy(self.best) - 0.05, "order %d: %s" % (
            self.k, bits(self.best)), best, ink=dark(TABLE, 0.45), size=TEXT_S)
        self.pair_key(sc, ("held out", TABLE), ("on its own text", TRAIN_INK),
                      op * smoothstep(0.15, 0.25, u))
        self.tag(sc, HELD % self.n_held, op)

    def phase_3(self, sc, u, op):
        m = self.meta
        r = m["rnn"]
        self.text(sc, 0.05, 0.30, "a GRU: one character in, a state carried on", op,
                  anchor="lm", strong=True, size=SMALL)
        # the cell, its input, its output and the state it keeps
        self.box(sc, 0.10, 0.50, 62, 32, soft(GREEN, 0.6), op, "x", border=GREEN,
                 size=TEXT_S)
        self.text(sc, 0.10, 0.555, "a character", op, size=TEXT_S)
        self.box(sc, 0.25, 0.50, 120, 56, soft(RNN, 0.6), op, "state h: %d" % r["hidden"],
                 border=RNN, size=TEXT_S, ink=dark(RNN, 0.4))
        self.box(sc, 0.41, 0.50, 72, 32, soft(GREEN, 0.6), op, "p(next)", border=GREEN,
                 size=TEXT_S)
        self.arrow(sc, (0.135, 0.50), (0.185, 0.50), AXIS, op, width=1.8)
        self.arrow(sc, (0.315, 0.50), (0.365, 0.50), AXIS, op, width=1.8)
        self.arc(sc, (0.29, 0.445), (0.21, 0.445), RNN, op, u=smoothstep(0.05, 0.2, u),
                 bend=0.07, width=2.0)
        self.text(sc, 0.25, 0.36, "to the next character", op, size=TEXT_S,
                  ink=dark(RNN, 0.4))
        self.lines(sc, 0.05, 0.63, [
            ("%s parameters, the same at every step" % commas(r["parameters"]),
             dark(RNN, 0.4)),
            "%s updates × %d streams × %d characters" % (commas(r["steps"]), r["streams"],
                                                         r["window"]),
            "(%.1f passes through the training text)" % r["passes"],
            "truncated BPTT, Adam, seed %d" % r["seed"]],
            op * smoothstep(0.1, 0.2, u), size=TEXT_S, step=0.042)
        go = smoothstep(0.2, 0.85, u)
        ch = self.pane(self.CHART, (0, self.steps[-1]), (2.0, 6.5))
        self.axes(sc, ch, op, "updates", "bits per character",
                  [(v, commas(v)) for v in (0, 400, 800, 1200) if v <= self.steps[-1]],
                  [(v, "%d" % v) for v in (2, 3, 4, 5, 6)])
        self.hline(sc, ch, self.best, TABLE, op, width=1.8)
        self.text(sc, ch.x1, ch.fy(self.best) - 0.03, "best table, order %d: %s"
                  % (self.k, bits(self.best)), op, anchor="rm", ink=dark(TABLE, 0.45),
                  size=TEXT_S)
        self.path(sc, ch, self.steps, self.rnn_train, RNN_TRAIN, op, go, width=2.0)
        self.path(sc, ch, self.steps, self.rnn_held, RNN, op, go, width=2.8, dot=True)
        end = op * smoothstep(0.85, 0.92, u)
        self.text(sc, 0.05, 0.83, "held out after %s updates: %s bits"
                  % (commas(self.steps[-1]), bits(m["rnn_final"])), end, anchor="lm",
                  ink=dark(RNN, 0.4), size=SMALL)
        self.pair_key(sc, ("held out", RNN), ("training batches", RNN_TRAIN),
                      op * smoothstep(0.2, 0.3, u))
        self.tag(sc, HELD % self.n_held, op)

    def phase_4(self, sc, u, op):
        m = self.meta
        r = m["rnn"]
        self.text(sc, 0.05, 0.30, "what each model keeps", op, anchor="lm", strong=True,
                  size=SMALL)
        self.lines(sc, 0.05, 0.36, [
            ("order-%d table: %s counts" % (self.k, commas(m["best_pairs"])),
             dark(TABLE, 0.45)),
            "one per (context, next) it saw, of 65%s = %s" % (sup(self.k + 1),
                                                              commas(m["possible_cells"])),
            ("GRU: %s parameters" % commas(r["parameters"]), dark(RNN, 0.4)),
            "the same ones for every context"], op, size=TEXT_S, step=0.044)
        late = op * smoothstep(0.55, 0.65, u)
        self.lines(sc, 0.05, 0.60, [
            "an unseen context: the table has",
            "no row, the network still has a state",
            ("training %s → held out %s (table)" % (bits(m["best_training_bits"]),
                                                    bits(self.best)), dark(TABLE, 0.45)),
            ("training %s → held out %s (GRU)" % (bits(m["rnn_training_bits"]),
                                                  bits(m["rnn_final"])), dark(RNN, 0.4))],
            late, size=TEXT_S, step=0.044)
        x = 0.62
        self.text(sc, x - 0.06, 0.30, "held-out bits, by how often the order-%d" % self.k,
                  op, anchor="lm", size=TEXT_S, strong=True)
        self.text(sc, x - 0.06, 0.34, "context was counted in training", op, anchor="lm",
                  size=TEXT_S, strong=True)
        top = max(float(np.nanmax(self.buckets[:, 1:])), 1.0)
        full = self.span(0.17)[0]
        for i, name in enumerate(self.bucket_names):
            y = 0.43 + 0.11 * i
            a = op * smoothstep(0.08 + 0.1 * i, 0.16 + 0.1 * i, u)
            g = smoothstep(0.08 + 0.1 * i, 0.3 + 0.1 * i, u)
            share, tb, rb = self.buckets[i]
            self.text(sc, x - 0.06, y - 0.021, name if name == "never" else
                      name.replace("-", "–") + " times", a, anchor="lm", size=TEXT_S)
            self.text(sc, x - 0.06, y + 0.021, "%s of positions" % pct(share), a, anchor="lm",
                      size=TEXT_S, ink=GREY)
            self.hbar(sc, x + 0.09, y - 0.021, full * tb / top, TABLE, a, value=bits(tb),
                      u=g, thick=13.0)
            self.hbar(sc, x + 0.09, y + 0.021, full * rb / top, RNN, a, value=bits(rb),
                      u=g, thick=13.0)
        self.pair_key(sc, ("table", TABLE), ("GRU", RNN), op)
        self.tag(sc, HELD % self.n_held, op)

    def phase_5(self, sc, u, op):
        self.text(sc, 0.05, 0.30, "perplexity = 2^bits: how many equally likely characters",
                  op, anchor="lm", strong=True, size=SMALL)
        inks = [GREY, soft(TABLE, 0.3), soft(TABLE, 0.15), TABLE, RNN]
        full = self.span(0.30)[0]
        for i, (name, p, b) in enumerate(zip(self.perp_labels, self.perp, self.perp_bits)):
            y = 0.38 + 0.075 * i
            a = op * smoothstep(0.05 + 0.08 * i, 0.12 + 0.08 * i, u)
            self.hbar(sc, 0.25, y, full * p / self.perp[0], inks[i], a, name,
                      "2%s = %.1f" % ("^" + bits(b), p), u=smoothstep(0.05 + 0.08 * i,
                                                                    0.3 + 0.08 * i, u),
                      thick=18.0)
        late = op * smoothstep(0.45, 0.55, u)
        self.lines(sc, 0.05, 0.78, [
            "lower is better: the GRU is as unsure as a choice",
            "among %.1f characters, the best table %.1f" % (self.perp[-1], self.perp[-2])],
            late, size=TEXT_S, step=0.042)
        end = op * smoothstep(0.35, 0.45, u)
        self.text(sc, 0.93, 0.62, NEXT, end, anchor="rm", ink=dark(GREEN, 0.35), size=BIG)
        self.text(sc, 0.93, 0.69, "perplexity returns in Training language models", end,
                  anchor="rm", ink=dark(VIOLET, 0.3), size=TEXT_S)
        self.tag(sc, EXACT, op)


ANIMATIONS = {"markov-chain-to-rnn": MarkovChainToRnn}

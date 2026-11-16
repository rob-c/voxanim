"""Training a language model: animations 21-28 of the "next 100" (see :mod:`voxanim.mech`).

Each draws the mechanism first and the measurement after it. The measurements
are the study's (:mod:`voxanim.next_language`, and :mod:`voxanim.study_training`
for the three experiments that were redesigned); what a mechanism needs beyond
them (a rollout, per-image predictions, a tokenisation, a batch's make-up) is
recomputed here from the same data, seeds and cached checkpoints.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import realdata
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIM,
    GREEN,
    GRID,
    PAPER,
    RED,
    TEAL,
    VIOLET,
    WHITE,
    Mechanism,
    dark,
    number,
    soft,
)

INK = (0.16, 0.17, 0.19)


def glyph(ch):
    """A character as it is shown in a token box (a space as an open box)."""
    return "␣" if ch == " " else "↵" if ch == "\n" else ch


def glyphs(ids):
    """Tiny Shakespeare character ids as on-screen glyphs."""
    return [glyph(ch) for ch in realdata.decode(ids)]


def ramp(u, a, b):
    """0 before ``a``, 1 after ``b``, smooth between."""
    return float(smoothstep(a, b, u))


def turn_of(u, n, span=1.0):
    """Which of ``n`` things has its turn at ``u`` (taking ``span`` of the
    phase between them), and how far through that turn it is."""
    k = min(max(u, 0.0) / span, 1.0) * n
    i = min(int(k), n - 1)
    return i, min(k - i, 1.0)


# ---------------------------------------------------------------------- 21


@lru_cache(maxsize=1)
def teacher_rollout(example=1):
    """The study's trained model on one held-out excerpt: the recorded
    characters, its prediction at every position given the recorded text
    before it (teacher forcing), and what it writes when it reads its own
    output instead (greedy, from the first recorded character)."""
    from . import next_language as nl
    from . import tinylm as lm

    seq = nl._text_windows(256, 16)
    c = lm.Config(vocab=nl._corpus()["vocab"], width=24, heads=4, kv_heads=4)
    run = lm.train(c, seq[:, :-1], seq[:, 1:], steps=350, seed=nl.SEED, name="real-teacher-v8")
    test = nl._text_windows(96, 16, "test", nl.SEED + 1)
    row = test[example:example + 1]
    forced = lm.forward(run["P"], row[:, :-1], c)[0].argmax(-1)
    free = row[:, :1].copy()
    for _ in range(15):
        free = np.c_[free, lm.forward(run["P"], free, c)[:, -1].argmax(-1)]
    return dict(recorded=row[0], forced=forced, free=free[0])


class TeacherForcing(Mechanism):
    """One sequence as many predictions: inputs and the targets shifted by
    one, the causal mask that keeps each target hidden, then the same model
    reading its own output instead of the recorded text."""

    TITLE = "Teacher forcing"
    PHASES = (("shifted targets", 11.0), ("the mask", 12.0), ("training", 10.0),
              ("on its own", 13.0), ("measured", 10.0))
    N = 8
    ROW = (0.05, 0.62)
    TOKEN = dict(w=60.0, h=46.0, size=0.040)  # noqa: RUF012

    def setup(self):
        s = self.study
        pair = np.asarray(s.panels[0].data["values"], int)
        self.x = glyphs(pair[0][: self.N])
        self.y = glyphs(pair[1][: self.N])
        self.mask = np.asarray(s.panels[1].data["values"], float)[: self.N, : self.N]
        self.steps = np.asarray(s.panels[2].data["x"], float) + 1
        self.loss = np.asarray(s.panels[2].data["ys"], float)[0]
        self.by_pos = np.asarray(s.panels[3].data["ys"], float)
        self.length = self.by_pos.shape[1]
        self.forced_loss = float(self.by_pos[0, 1:].mean())
        self.free_loss = float(self.by_pos[1, 1:].mean())
        r = teacher_rollout()
        self.recorded = glyphs(r["recorded"])
        self.forced = glyphs(r["forced"])
        self.free = glyphs(r["free"])
        self.forced_hits = int((r["forced"][:-1] == r["recorded"][1:-1]).sum())
        self.free_hits = int((r["free"][1:] == r["recorded"][1:-1]).sum())

    def subtitle_for(self, k):
        return [
            "the targets are the same text moved one place left: every character is asked for "
            "the one that follows it, so %d characters give %d predictions" % (self.N, self.N),
            "the causal mask keeps it honest: each character reads only itself and the ones "
            "before it, so the character it must predict is never in view",
            "all positions are trained in one pass, always reading the recorded text: %d updates "
            "take the loss from %.2f to %.2f nats per character" % (
                len(self.loss), self.loss[0], self.loss[-1]),
            "when it generates, the model reads its own output instead; here it matches the "
            "recorded text at %d of 15 places, against %d of 15 when reading the recording" % (
                self.free_hits, self.forced_hits),
            "scored against the recorded continuation on 96 held-out excerpts: %.2f nats reading "
            "the recording, %.2f reading its own output" % (self.forced_loss, self.free_loss),
        ][k]

    def phase_1(self, sc, u, op):
        x0, x1 = self.ROW
        top, low = 0.45, 0.71
        n = self.N
        pitch = (x1 - x0) / n
        xs = self.tokens(sc, self.x, x0, x1, top, op, **self.TOKEN)
        self.text(sc, x0 + 0.005, top - 0.085, "what the model reads", op, anchor="lm",
                  strong=True, size=0.034)
        slide = ramp(u, 0.22, 0.48)
        # the same characters again, sliding one place to the left: the first
        # leaves, the next character of the text arrives
        every = [self.x[0]] + self.y
        for j, ch in enumerate(every):
            x = x0 + pitch * (j + 0.5 - slide)
            fade = (1.0 - slide) if j == 0 else slide if j == n else 1.0
            self.box(sc, x, low, 60, 46, soft(GREEN), op * fade, ch, ink=dark(GREEN, 0.35),
                     border=GREEN, size=0.040)
        self.text(sc, x0 + 0.005, low + 0.085,
                  "the same text again" if slide < 0.5 else "what it must predict: the next character",  # noqa: E501
                  op, anchor="lm", strong=True, size=0.034)
        for i in range(n):
            a = ramp(u, 0.52 + 0.04 * i, 0.60 + 0.04 * i)
            self.arrow(sc, (xs[i], top + 0.052), (xs[i], low - 0.052), GREEN, op, a, 2.6)
            self.pulse(sc, (xs[i], top + 0.052), (xs[i], low - 0.052),
                       (u - 0.56 - 0.04 * i) / 0.10, GREEN, op)
        done = sum(u > 0.60 + 0.04 * i for i in range(n))
        self.lines(sc, 0.68, 0.53, [("one sequence", None), ("%d predictions" % done, dark(GREEN))],
                   op * ramp(u, 0.5, 0.56), size=0.040, step=0.07)

    def phase_2(self, sc, u, op):
        x0, x1 = self.ROW
        y, n = 0.60, self.N
        turn, within = turn_of(u, n, 0.94)
        blocked = [j for j in range(n) if not self.mask[turn, j]]
        plain = ["" if j == turn or j in blocked else ch for j, ch in enumerate(self.x)]
        xs = self.tokens(sc, plain, x0, x1, y, op, **self.TOKEN)
        self.box(sc, xs[turn], y, 60, 46, BLUE, op, self.x[turn], ink=WHITE, size=0.040, glow=0.15)
        for j in blocked:
            self.box(sc, xs[j], y, 60, 46, DIM, op * 0.85, self.x[j], size=0.040)
        for j in range(turn):
            bend = 0.07 + 0.034 * (turn - j)
            a, b = (xs[j], y - 0.055), (xs[turn], y - 0.055)
            self.arc(sc, a, b, BLUE, op, ramp(within, 0.0, 0.45), bend, 2.4)
            self.pulse(sc, a, b, (within - 0.3) / 0.5, BLUE, op, bend)
        # its target: the next character, which the mask keeps out of its sight
        show = ramp(within, 0.45, 0.7)
        self.arrow(sc, (xs[turn], y + 0.052), (xs[turn], 0.745), GREEN, op * show, 1.0, 2.6)
        self.box(sc, xs[turn], 0.795, 60, 46, soft(GREEN), op * show, self.y[turn],
                 ink=dark(GREEN, 0.35), border=GREEN, size=0.040)
        self.text(sc, x0 + 0.005, 0.90, "%s reads %d of the %d characters and must predict %s" % (
            self.x[turn], int(self.mask[turn].sum()), n, self.y[turn]), op, anchor="lm",
            strong=True, size=0.034)
        cell = self.grid(sc, self.mask, 0.81, 0.60, 236, 236, op, reveal=(turn + 1) / n,
                         ring=[(turn, turn)], hot=BLUE)
        self.text(sc, 0.81, 0.31, "who may read whom", op, strong=True, size=0.034)
        self.text(sc, cell(turn, 0)[0] - 0.035, cell(turn, 0)[1], self.x[turn], op,
                  ink=dark(BLUE), size=0.034)

    def phase_3(self, sc, u, op):
        x0, x1 = self.ROW
        xs = self.tokens(sc, self.x, x0, x1, 0.335, op, w=52.0, h=36.0, size=0.036)
        for i, x in enumerate(xs):
            self.box(sc, x, 0.445, 52, 36, soft(GREEN), op, self.y[i], ink=dark(GREEN, 0.35),
                     border=GREEN, size=0.036)
            self.arrow(sc, (x, 0.372), (x, 0.408), GREEN, op, 1.0, 2.2)
            self.pulse(sc, (x, 0.365), (x, 0.415), (u * 6) % 1.0, GREEN, op, r=3.5)
        self.lines(sc, 0.66, 0.335, ["read", ("predict, all at once", dark(GREEN))], op,
                   size=0.034, step=0.11)
        chart = self.chart(((0.10, 0.62), (0.57, 0.84)), self.steps, self.loss)
        self.axes(sc, chart, op, "training update", "loss (nats per character)")
        self.plot(sc, chart, self.steps, self.loss, GREEN, op, u / 0.85)
        now = int(min(u / 0.85, 1.0) * (len(self.steps) - 1))
        self.lines(sc, 0.70, 0.64, [("update %d" % self.steps[now], None),
                                    ("loss  %.2f" % self.loss[now], dark(GREEN))],
                   op, step=0.07, size=0.040)

    def phase_4(self, sc, u, op):
        n = 16
        x0, x1 = 0.05, 0.95
        tok = dict(w=46.0, h=38.0, size=0.036)
        ya, yb, yc = 0.355, 0.50, 0.70
        xs = self.tokens(sc, self.recorded[:n], x0, x1, ya, op, **tok)
        self.text(sc, x0 + 0.004, ya - 0.062, "the recorded text (held out)", op, anchor="lm",
                  strong=True)
        # teacher forcing: every prediction made from the recorded text, at once
        a = ramp(u, 0.04, 0.22)
        self.text(sc, x0 + 0.004, yb + 0.066, "reading the recording: each guess starts from the "
                  "true text", op * a, anchor="lm", ink=dark(GREEN))
        for i in range(n - 1):
            hit = self.forced[i] == self.recorded[i + 1]
            self.arrow(sc, (xs[i] + 0.012, ya + 0.043), (xs[i + 1] - 0.012, yb - 0.043), GREEN,
                       op, a, 2.0)
            self.box(sc, xs[i + 1], yb, 46, 38, GREEN if hit else soft(GREEN, 0.8),
                     op * ramp(u, 0.14, 0.26), self.forced[i], ink=WHITE if hit else dark(GREEN, 0.35),  # noqa: E501
                     border=GREEN, size=0.036)
        # free running: one character at a time, each read back in
        k, within = turn_of((u - 0.34) / 0.56, n - 1)
        started = u > 0.34
        self.text(sc, x0 + 0.004, yc - 0.062, "reading its own output: one wrong guess becomes "
                  "the next input", op * ramp(u, 0.30, 0.36), anchor="lm", ink=dark(RED))
        self.box(sc, xs[0], yc, 46, 38, PAPER, op * ramp(u, 0.30, 0.36), self.free[0],
                 border=AXIS, size=0.036)
        for i in range(1, n):
            if not started or i - 1 > k:
                break
            grown = 1.0 if i - 1 < k else ramp(within, 0.35, 0.7)
            hit = self.free[i] == self.recorded[i]
            src, dst = (xs[i - 1], yc + 0.043), (xs[i], yc + 0.043)
            self.arc(sc, src, dst, RED, op, 1.0 if i - 1 < k else ramp(within, 0.0, 0.5), -0.08, 2.0)  # noqa: E501
            if i - 1 == k:
                self.pulse(sc, src, dst, within / 0.6, RED, op, -0.08)
            self.box(sc, xs[i], yc, 46, 38, RED if hit else soft(RED, 0.8), op * grown,
                     self.free[i], ink=WHITE if hit else dark(RED, 0.35), border=RED, size=0.036)
        self.text(sc, x0 + 0.004, 0.875, "a filled box matches the recorded text:  %d of 15 "
                  "reading the recording,  %d of 15 on its own" % (self.forced_hits, self.free_hits),  # noqa: E501
                  op * ramp(u, 0.88, 0.95), anchor="lm", strong=True)

    def phase_5(self, sc, u, op):
        pos = np.arange(self.length) + 1.0
        chart = self.chart(((0.10, 0.62), (0.36, 0.80)), pos, np.r_[self.by_pos.ravel(), 2.5])
        self.axes(sc, chart, op, "character of the excerpt", "loss on the recorded next character (nats)",  # noqa: E501
                  xticks=[(v, "%d" % v) for v in (1, 4, 8, 12, 16)])
        self.plot(sc, chart, pos, self.by_pos[0], GREEN, op, u / 0.6, width=2.4)
        self.plot(sc, chart, pos, self.by_pos[1], RED, op, u / 0.6, width=2.4)
        self.lines(sc, 0.68, 0.44, [("reading its own output", dark(RED)),
                                    ("%.2f nats" % self.free_loss, dark(RED))],
                   op, size=0.036, step=0.055)
        self.lines(sc, 0.68, 0.66, [("reading the recording", dark(GREEN)),
                                    ("%.2f nats" % self.forced_loss, dark(GREEN))],
                   op, size=0.036, step=0.055)
        self.text(sc, 0.10, 0.925, "training never showed it its own mistakes; another "
                  "continuation can still be good English", op * ramp(u, 0.6, 0.8), anchor="lm")


# ---------------------------------------------------------------------- 22


@lru_cache(maxsize=1)
def perplexity_tokens(n=9):
    """The first ``n`` GPT-2 tokens of the study's first held-out excerpt (the
    first is only read; the rest are the ones it scores), as written."""
    from . import next_language as nl

    tok = nl._tok()
    ids = tok.encode(nl._corpus()["test"][:100])[:n]
    return [tok.decode([i]) for i in ids]


def one_in(p):
    """A probability as odds: ``1 in 2.7``, ``1 in 145M``."""
    return "1 in %s" % number(1.0 / p)


class PerplexityExplained(Mechanism):
    """GPT-2's surprise at each real token, the average turned into bits and
    into perplexity (a pick among that many equally likely tokens), and the
    same tokens scored by a unigram and a bigram count."""

    TITLE = "Perplexity"
    PHASES = (("surprise", 14.0), ("perplexity", 12.0), ("three models", 13.0),
              ("compared", 10.0))
    MODELS = (("unigram", AMBER), ("bigram", VIOLET), ("GPT-2", BLUE))

    def setup(self):
        s = self.study
        self.words = ["".join(glyph(ch) for ch in w) for w in perplexity_tokens()]
        self.surprise = np.asarray(s.panels[0].data["values"], float)
        self.mean = [float(v) for v in s.panels[1].data["values"]]
        self.ppl = [float(v) for v in s.panels[2].data["values"]]
        self.scored = int(s.metadata["settings"]["scored_tokens"])
        self.counted = int(s.metadata["settings"]["baseline_train_tokens"])
        from . import gpt2

        self.vocab = int(gpt2.N_VOCAB)

    def subtitle_for(self, k):
        uni, bi, gpt = self.ppl
        return [
            "GPT-2 reads the text so far and gives every possible next token a chance; its "
            "surprise at the token that really came is minus the log of that chance, in nats",
            "over %d held-out tokens the average surprise is %.2f nats, or %.2f bits; perplexity "
            "is e to that power: %.0f, a blind pick among %.0f equally likely tokens" % (
                self.scored, self.mean[2], self.mean[2] / np.log(2), gpt, gpt),
            "the same %d tokens scored three ways: a unigram count has perplexity %s, a bigram "
            "count %s, GPT-2 %.0f" % (self.scored, f"{uni:,.0f}", f"{bi:,.0f}", gpt),
            "counted from only %s tokens the bigram table rarely knows the pair, and lands near "
            "a blind pick among all %s tokens; GPT-2 narrows it to %.0f" % (
                f"{self.counted:,}", f"{self.vocab:,}", gpt),
        ][k]

    def phase_1(self, sc, u, op):
        n = len(self.words)
        x0, x1, y = 0.05, 0.95, 0.83
        turn, within = turn_of(u, n - 1, 0.93)
        target = turn + 1
        xs = [x0 + (x1 - x0) * (i + 0.5) / n for i in range(n)]
        for i, word in enumerate(self.words):
            if i == target:
                self.box(sc, xs[i], y, 80, 44, BLUE, op, word, ink=WHITE, size=0.036, glow=0.15)
            elif i < target:
                self.box(sc, xs[i], y, 80, 44, PAPER, op, word, border=AXIS, size=0.036)
            else:
                self.box(sc, xs[i], y, 80, 44, DIM, op * 0.7, word, size=0.036)
        base, full = y - 0.058, 0.36          # a bar's foot; its height in layout units at 19 nats
        for i in range(1, target + 1):
            value = self.surprise[i - 1]
            grow = 1.0 if i < target else ramp(within, 0.35, 0.75)
            h = full * value / 19.0 * grow
            hp = max(self.span(0, h)[1], 0.5)
            self.rect(sc, self.P(xs[i], base - h / 2), 44, hp, BLUE, op * (1.0 if i == target else 0.6),  # noqa: E501
                      radius=3)
            if grow > 0.5:
                self.text(sc, xs[i], base - h - 0.026, "%.1f" % value, op * ramp(grow, 0.5, 1.0),
                          size=0.034, strong=True)
        for j in range(target):
            bend = 0.05 + 0.012 * (target - j)
            a, b = (xs[j], y + 0.05), (xs[target], y + 0.05)
            self.arc(sc, a, b, BLUE, op * 0.8, ramp(within, 0.0, 0.3), -bend, 1.8)
            self.pulse(sc, a, b, (within - 0.1) / 0.35, BLUE, op, -bend)
        p = float(np.exp(-self.surprise[turn]))
        self.lines(sc, x0, 0.305, [
            ("GPT-2 gave  %s  a chance of %s" % (self.words[target], one_in(p)), None),
            ("surprise = ln(%s) = %.2f nats" % (number(1.0 / p), self.surprise[turn]), dark(BLUE))],
            op * ramp(within, 0.25, 0.4), size=0.034, step=0.052)

    def phase_2(self, sc, u, op):
        nats = self.mean[2] * ramp(u, 0.08, 0.72)
        k = float(np.exp(nats))
        full = self.span(0.36)[0]
        x = 0.07
        self.text(sc, x, 0.32, "average surprise over %d tokens" % self.scored, op, anchor="lm",
                  strong=True, size=0.034)
        self.hbar(sc, x, 0.40, full * nats / 8.0, BLUE, op, "", "%.2f nats" % nats, thick=30.0,
                  track=full)
        self.text(sc, x, 0.50, "the same in yes-or-no questions", op, anchor="lm", strong=True,
                  size=0.034)
        self.hbar(sc, x, 0.58, full * nats / 8.0, TEAL, op, "", "%.2f bits" % (nats / np.log(2)),
                  thick=30.0, track=full)
        self.text(sc, x, 0.70, "perplexity = e to the power of the nats", op, anchor="lm",
                  strong=True, size=0.034)
        self.text(sc, x, 0.79, "%.0f" % k, op, anchor="lm", ink=dark(BLUE), size=0.075)
        self.text(sc, x, 0.875, "equally likely tokens to pick from", op, anchor="lm")
        # that many cells, one of them the right token
        rows, cols = 12, 18
        cells = np.zeros(rows * cols)
        shown = int(round(k))
        cells[:shown] = 0.35
        cells[int(u * 37) * 53 % max(shown, 1)] = 1.0
        self.grid(sc, cells.reshape(rows, cols), 0.755, 0.60, 330, 220, op, vmax=1.0, hot=BLUE)
        self.text(sc, 0.755, 0.33, "one cell per equally likely choice", op, strong=True, size=0.034)  # noqa: E501
        self.text(sc, 0.755, 0.875, "every extra 0.69 nats (one bit) doubles them", op)

    def _reads(self, sc, x, y, kind, ink, op):
        """Which of the earlier tokens a model uses to predict the last one."""
        n = len(self.words)
        for j in range(n):
            used = j == n - 1 or kind == 2 or (kind == 1 and j == n - 2)
            fill = INK if j == n - 1 else ink if used else DIM
            self.box(sc, x + (j - (n - 1) / 2) * 0.0215, y, 15, 22, fill, op, radius=3)

    def phase_3(self, sc, u, op):
        what = ("how common the token is", "the one token before it", "everything before it")
        scale = 1.5                               # px per unit of the square root of a perplexity
        for i, ((name, ink), ppl, nats) in enumerate(zip(self.MODELS, self.ppl, self.mean)):
            a = ramp(u, 0.05 + 0.24 * i, 0.12 + 0.24 * i)
            x = (0.19, 0.50, 0.81)[i]
            self.text(sc, x, 0.31, name, op * a, ink=dark(ink), size=0.040)
            self._reads(sc, x, 0.375, i, ink, op * a)
            self.text(sc, x, 0.43, what[i], op * a)
            side = max(np.sqrt(ppl) * scale * ramp(u, 0.10 + 0.24 * i, 0.30 + 0.24 * i), 0.5)
            self.rect(sc, self.P(x, 0.645), side, side, ink, op * a, radius=2)
            b = ramp(u, 0.26 + 0.24 * i, 0.32 + 0.24 * i)
            self.text(sc, x, 0.855, "%.2f nats" % nats, op * a * b, strong=True, size=0.034)
            self.text(sc, x, 0.905, "a pick among %s" % f"{ppl:,.0f}", op * a * b,
                      ink=dark(ink), size=0.034)

    def phase_4(self, sc, u, op):
        rows = [("knows nothing", AXIS, float(self.vocab)),
                ("bigram", VIOLET, self.ppl[1]), ("unigram", AMBER, self.ppl[0]),
                ("GPT-2", BLUE, self.ppl[2])]
        full = self.span(0.50)[0]
        top = np.log(self.vocab)
        self.text(sc, 0.24, 0.34, "bar length: average surprise in nats.   label: perplexity", op,
                  anchor="lm", strong=True)
        for i, (name, ink, ppl) in enumerate(rows):
            self.hbar(sc, 0.24, 0.44 + 0.115 * i, full * np.log(ppl) / top, ink, op, name,
                      "%s  (%.2f nats)" % (f"{ppl:,.0f}", np.log(ppl)),
                      (u - 0.08 * i) / 0.4, thick=36.0, track=full)
        self.text(sc, 0.24, 0.90, "the top bar is exact: ln of GPT-2's %s tokens, each equally "
                  "likely" % f"{self.vocab:,}", op * ramp(u, 0.5, 0.7), anchor="lm")


# ---------------------------------------------------------------------- 23


@lru_cache(maxsize=1)
def packing_example(pair=1):
    """What the packing study draws from: its twelve source lines, and one of
    the pairs of held-out excerpts it packs into a single 16-character row."""
    from . import next_language as nl

    lines = [line for line in nl._corpus()["train"].splitlines() if 3 <= len(line) <= 80][:12]
    test = nl._text_windows(64, 8, "test", nl.SEED + 1)[:, :-1]
    return dict(lines=lines, a=test[pair], b=test[32 + pair])


def packed_masks(n=8):
    """Who may read whom when two ``n``-character excerpts share a row:
    with nothing done (causal across both), and what of that crosses from
    the second excerpt into the first."""
    doc = np.repeat([0, 1], n)
    causal = np.tril(np.ones((2 * n, 2 * n), bool))
    leak = causal & (doc[:, None] != doc[None, :])
    return causal, leak


class PackingAndPadding(Mechanism):
    """A batch of real lines padded to the longest, two excerpts packed into
    one row instead, the second reading into the first, and the two resets
    (attention mask, positions) that stop it."""

    TITLE = "Packing and padding"
    PHASES = (("padding", 12.0), ("packing", 9.0), ("the leak", 12.0), ("the resets", 11.0),
              ("measured", 10.0))
    A_INK, B_INK, PAD, LEAK, MAY = BLUE, GREEN, AMBER, RED, TEAL
    ROW_Y = 0.42

    def setup(self):
        s = self.study
        self.lengths = np.asarray(s.panels[0].data["values"], int)
        self.used, self.wasted = (int(v) for v in s.panels[1].data["values"])
        self.reset_mask = np.asarray(s.panels[2].data["values"], float) > 0.5
        self.losses = [float(v) for v in s.panels[3].data["values"]]
        ex = packing_example()
        self.source = ex["lines"]
        self.a, self.b = glyphs(ex["a"]), glyphs(ex["b"])
        self.causal, self.leak = packed_masks(len(self.a))
        self.n = 2 * len(self.a)

    def subtitle_for(self, k):
        ind, plain, reset = self.losses
        return [
            "twelve real lines in one batch, each padded to the longest: %d characters of text "
            "and %d cells of padding, %.0f%% of the batch computed for nothing" % (
                self.used, self.wasted, 100.0 * self.wasted / (self.used + self.wasted)),
            "packing puts excerpts end to end in one row instead: two 8-character excerpts "
            "fill all 16 cells, with no padding at all",
            "but nothing now separates them: the second excerpt reads the first, through %d "
            "of the %d allowed pairs, and its positions run on from 8" % (
                int(self.leak.sum()), int(self.causal.sum())),
            "two resets fix it: the mask stops attention at the boundary, leaving %d pairs, "
            "and the position count starts again at 0" % int(self.reset_mask.sum()),
            "on 64 held-out excerpts: %.3f nats one at a time, %.3f packed with no resets, "
            "%.3f packed with both resets, the same as one at a time" % (ind, plain, reset),
        ][k]

    def phase_1(self, sc, u, op):
        rows, longest = len(self.lengths), int(self.lengths.max())
        turn, _ = turn_of(u, rows, 0.42)
        sweep = ramp(u, 0.5, 0.8) * longest
        col = np.arange(longest)[None, :]
        text = col < self.lengths[:, None]
        v = np.where(text, 1.0, np.where(col < sweep, -0.75, 0.0))
        w, h = 690.0, 204.0
        cx, cy = 0.50, 0.565
        self.grid(sc, v, cx, cy, w, h, op, reveal=(turn + 1) / rows, vmax=1.0, hot=self.A_INK,
                  cold=self.PAD)
        dw, dh = self.frac(w, h)
        left, _top = cx - dw / 2, cy - dh / 2
        line = self.source[turn]
        if u < 0.47:
            self.text(sc, left, 0.315, "line %d:  %s" % (turn + 1, line), op, anchor="lm",
                      strong=True, size=0.032)
            self.text(sc, left + dw, 0.315, "%d characters" % len(line), op, anchor="rm",
                      ink=dark(self.A_INK), size=0.032)
        else:
            self.text(sc, left, 0.315, "every line is padded to the longest, %d characters"
                      % longest, op, anchor="lm", strong=True, size=0.032)
        shown = int(self.lengths[: turn + 1].sum())
        pad = int((v < 0).sum())
        self.text(sc, left, 0.845, "%d characters of text" % shown, op, anchor="lm",
                  ink=dark(self.A_INK), size=0.038)
        self.text(sc, left + dw, 0.845, "%d padding cells" % pad, op * ramp(u, 0.5, 0.55),
                  anchor="rm", ink=dark(self.PAD, 0.3), size=0.038)
        self.text(sc, left, 0.91, "one row per line, one cell per character", op, anchor="lm")

    def _packed(self, sc, op, y=None, positions=None, pos_ink=None, blank=None):
        """The two excerpts in one row of 16; returns the cells' x."""
        y = self.ROW_Y if y is None else y
        half = self.n // 2
        words = ["" if i == blank else ch for i, ch in enumerate(self.a + self.b)]
        xs = self.tokens(sc, words, 0.05, 0.95, y, op,
                         inks=[self.A_INK] * half + [self.B_INK] * half, w=46.0, h=42.0, size=0.038)
        if positions is not None:
            for i, x in enumerate(xs):
                self.text(sc, x, y + 0.068, positions[i], op, size=0.030,
                          ink=None if pos_ink is None else pos_ink[i])
        return xs

    def phase_2(self, sc, u, op):
        half = self.n // 2
        slide = ramp(u, 0.25, 0.65)
        pitch = 0.9 / self.n
        ya, yb = 0.46, 0.68
        for row, (chars, ink) in enumerate(((self.a, self.A_INK), (self.b, self.B_INK))):
            for i, ch in enumerate(chars):
                x = 0.05 + pitch * (i + 0.5 + (half * slide if row else 0.0))
                y = ya if row == 0 else yb + (ya - yb) * slide
                self.box(sc, x, y, 46, 42, soft(ink), op, ch, ink=dark(ink, 0.35), border=ink,
                         size=0.038)
        self.text(sc, 0.05, ya - 0.085, "held-out excerpt A", op, anchor="lm", ink=dark(self.A_INK),
                  size=0.034)
        self.text(sc, 0.05 + pitch * half * slide, yb + (ya - yb) * slide + 0.085,
                  "held-out excerpt B", op, anchor="lm", ink=dark(self.B_INK), size=0.034)
        done = ramp(u, 0.68, 0.8)
        self.lines(sc, 0.05, 0.70, ["one row of 16 cells, every one of them real text",
                                    "padded to a batch of two rows they would use the same 16: "
                                    "packing pays when lengths differ"],
                   op * done, size=0.032, step=0.055)

    def _reading(self, sc, u, op, mask, leak):
        half = self.n // 2
        turn, within = turn_of(u, half, 0.94)
        q = half + turn
        positions = ["%d" % (i if leak is not None or i < half else i - half) for i in range(self.n)]  # noqa: E501
        inks = [dark(self.LEAK) if (leak is not None and i >= half) else None for i in range(self.n)]  # noqa: E501
        xs = self._packed(sc, op, positions=positions, pos_ink=inks, blank=q)
        y = self.ROW_Y
        self.box(sc, xs[q], y, 46, 42, self.B_INK, op, self.b[turn], ink=WHITE, size=0.038, glow=0.15)  # noqa: E501
        for j in range(q):
            if not mask[q, j]:
                continue
            crossing = leak is not None and leak[q, j]
            ink = self.LEAK if crossing else self.MAY
            bend = 0.04 + 0.0095 * (q - j)
            a, b = (xs[j], y - 0.05), (xs[q], y - 0.05)
            self.arc(sc, a, b, ink, op, ramp(within, 0.0, 0.45), bend, 2.2)
            self.pulse(sc, a, b, (within - 0.3) / 0.55, ink, op, bend)
        # the boundary between the excerpts
        bx = (xs[half - 1] + xs[half]) / 2
        self.polyline(sc, [self.P(bx, y - 0.06), self.P(bx, y + 0.085)], 2.2,
                      INK if leak is None else DIM, op)
        values = mask.astype(float) - (2.0 * leak if leak is not None else 0.0)
        side = 196.0
        cell = self.grid(sc, values, 0.81, 0.735, side, side, op, vmax=1.0, ring=[(q, q)],
                         hot=self.MAY, cold=self.LEAK, lines=0.10)
        own = int(mask[q, half:].sum())
        other = int(mask[q, :half].sum())
        self.lines(sc, 0.05, 0.60, [
            ("who may read whom, both excerpts", None),
            ("%s (in B) reads %d of its own excerpt" % (self.b[turn], own), dark(self.MAY)),
            ("and %d of excerpt A" % other, dark(self.LEAK) if other else None),
            ("its position: %s" % positions[q], dark(self.LEAK) if leak is not None else None)],
            op, size=0.034, step=0.062)
        return cell

    def phase_3(self, sc, u, op):
        self._reading(sc, u, op, self.causal, self.leak)

    def phase_4(self, sc, u, op):
        self._reading(sc, u, op, self.reset_mask, None)

    def phase_5(self, sc, u, op):
        half = self.n // 2
        thumbs = (np.tril(np.ones((half, half))), self.causal.astype(float) - 2.0 * self.leak,
                  self.reset_mask.astype(float))
        names = ("one excerpt at a time", "packed, no resets", "packed, both resets")
        inks = (self.A_INK, self.LEAK, self.MAY)
        full = self.span(0.40)[0]
        self.text(sc, 0.40, 0.31, "loss on 64 held-out excerpts (nats)", op, anchor="lm",
                  strong=True)
        for i, (name, ink, value, thumb) in enumerate(zip(names, inks, self.losses, thumbs)):
            y = 0.44 + 0.17 * i
            a = ramp(u, 0.05 + 0.2 * i, 0.15 + 0.2 * i)
            self.grid(sc, thumb, 0.10, y, 70, 70, op * a, vmax=1.0, hot=self.MAY, cold=self.LEAK,
                      lines=0.0)
            self.text(sc, 0.155, y, name, op * a, anchor="lm", ink=dark(ink), size=0.034)
            more = value - self.losses[0]
            self.hbar(sc, 0.40, y, full * value / 3.8, ink, op * a, "",
                      "%.3f" % value + ("  (+%.3f)" % more if more > 5e-4 else ""),
                      (u - 0.1 - 0.2 * i) / 0.2, thick=34.0, track=full)
        self.text(sc, 0.40, 0.905, "the resets give back the one-at-a-time loss exactly "
                  "(difference %.0e)" % abs(self.losses[2] - self.losses[0]),
                  op * ramp(u, 0.65, 0.8), anchor="lm")


# ---------------------------------------------------------------------- 24


def clip_ratios(norms, clip=1.0):
    """What clipping at ``clip`` multiplies each gradient by, given its length
    before clipping (the rule the study's training loop applies)."""
    return np.minimum(1.0, clip / np.maximum(np.asarray(norms, float), 1e-12))


class WarmupAndClipping(Mechanism):
    """The learning rate as the size of every update (fixed, or ramped up over
    the first 80), the gradient's length cut back to 1, then five seeds of each
    recipe: at the fixed rate none leaves the plateau of guessing
    by letter frequency, with warm-up all do."""

    TITLE = "Warm-up and gradient clipping"
    PHASES = (("the rate", 12.0), ("clipping", 12.0), ("five seeds each", 14.0),
              ("measured", 12.0))
    RUNS = (("fixed rate", RED), ("warm-up", BLUE), ("warm-up + clipping", GREEN),
            ("fixed, half the rate", AMBER))

    def setup(self):
        p = self.study.panels
        self.steps = np.asarray(p[0].data["x"], float)
        self.rate = np.asarray(p[0].data["ys"], float)            # fixed, warm-up, half
        self.held_at = np.asarray(p[1].data["x"], float)
        self.held = np.asarray(p[1].data["each"], float)          # arm, seed, checkpoint
        self.plateau = float(p[1].data["unigram"])
        self.norm = np.asarray(p[2].data["ys"], float)[2]         # warm-up + clipping, seed 0
        self.ratio = np.asarray(p[2].data["ratio"], float)[0]
        self.best = np.asarray(p[3].data["each"], float)          # arm, seed
        self.mean = [float(v) for v in p[3].data["values"]]
        self.learned = [int(v) for v in p[3].data["learned"]]
        self.seeds = self.best.shape[1]
        self.clipped = int((self.ratio < 1.0).sum())
        self.ramp_len = int(np.argmax(self.rate[1] >= self.rate[1].max()) + 1)
        self.peak = float(self.rate.max())
        self.layers = int(self.study.metadata["settings"]["layers"])
        self.stuck = float(self.held[0, :, -1].mean())

    def below(self, arm, k):
        """How many of an arm's seeds are under the plateau at checkpoint ``k``
        (by the study's rule: 0.2 nats under the letter-frequency loss)."""
        return int((self.held[arm, :, k] < self.plateau - 0.2).sum())

    def verdict(self):
        """What the half-rate runs say about the warmed-up ones."""
        gap = self.mean[1] - self.mean[3]
        if abs(gap) < 0.05:
            return "warm-up makes the higher rate usable, not better"
        return "and warm-up at the full rate ends %.2f nats %s" % (
            abs(gap), "lower" if gap < 0 else "higher")

    def subtitle_for(self, k):
        n, seeds = len(self.steps), self.seeds
        return [
            "the learning rate sets how far each update moves the weights: one run uses %.3f "
            "from the first update, warm-up climbs to it over the first %d" % (
                self.peak, self.ramp_len),
            "clipping measures each gradient's length and scales it back to 1 when it is longer: "
            "in this run that happened at %d of the %d updates" % (self.clipped, n),
            "%d seeds of each recipe, a %d-layer model, loss on held-out text: at the fixed "
            "rate %d of %d get under the %.2f nats of guessing by letter frequency, with warm-up "
            "%d of %d" % (seeds, self.layers, self.learned[0], seeds, self.plateau,
                          self.learned[1], seeds),
            "each run at its best held-out point: fixed rate %.2f, warm-up %.2f, warm-up and "
            "clipping %.2f nats. A fixed rate of half the size also trains (%.2f): %s" % (
                self.mean[0], self.mean[1], self.mean[2], self.mean[3], self.verdict()),
        ][k]

    def _now(self, u, span=0.86):
        """The update a phase has reached (0-based), and that as a fraction of the run."""
        f = min(max(u, 0.0) / span, 1.0)
        return int(round(f * (len(self.steps) - 1))), f

    def phase_1(self, sc, u, op):
        # time runs slowly through the ramp, then quickly to the end
        f = min(u / 0.86, 1.0)
        f = f * 0.5 if f < 0.7 else 0.35 + (f - 0.7) / 0.3 * 0.65
        i = int(round(f * (len(self.steps) - 1)))
        full = self.span(0.20)[0]
        n = self.layers
        self.text(sc, 0.05, 0.31, "update %d of %d" % (i + 1, len(self.steps)), op, anchor="lm",
                  strong=True, size=0.038)
        self.text(sc, 0.27, 0.31, "learning rate", op, anchor="lm")
        self.text(sc, 0.79, 0.31, "the model: %d layers" % n, op)
        pitch = 0.30 / n
        for row, (name, ink) in enumerate(self.RUNS[:2]):
            y = 0.385 + 0.095 * row
            lr = self.rate[row][i]
            self.text(sc, 0.05, y, name, op, anchor="lm", ink=dark(ink), size=0.036)
            self.hbar(sc, 0.27, y, full * lr / self.peak, ink, op, "", "%.4f" % lr, thick=30.0,
                      track=full)
            self.arrow(sc, (0.565, y), (0.625, y), ink, op, 1.0, 2.4)
            for j in range(n):
                x = 0.64 + pitch * (j + 0.5)
                self.box(sc, x, y, self.span(pitch)[0] - 8, 34, soft(ink, 0.75), op, border=ink)
                # each update nudges every layer: a bigger rate, a bigger nudge
                self.disk(sc, self.P(x, y), 2.0 + 9.0 * lr / self.peak, ink,
                          op * (0.45 + 0.55 * abs(np.sin(np.pi * (u * 14 + 0.2 * j)))), lift=1.2,
                          outline=0.0)
        chart = self.chart(((0.27, 0.86), (0.62, 0.84)), self.steps, np.r_[0.0, self.rate.ravel()],
                           pad=0.08)
        self.axes(sc, chart, op, "training update", "learning rate",
                  yticks=[(v, "%.3f" % v) for v in (0.0, self.peak / 2, self.peak)])
        for row, (_name, ink) in enumerate(self.RUNS[:2]):
            self.plot(sc, chart, self.steps, self.rate[row], ink, op, f, width=2.4)
        self.lines(sc, 0.05, 0.68, ["the size of", "each nudge", "to the weights"], op, step=0.045)

    def phase_2(self, sc, u, op):
        i, f = self._now(u)
        norm, ratio = self.norm[i], self.ratio[i]
        unit = self.span(0.20)[0]                 # the length of a gradient of norm 1, in px
        x0, y = 0.20, 0.40
        self.text(sc, 0.05, 0.315, "update %d: the gradient's length is %.2f" % (i + 1, norm), op,
                  anchor="lm", strong=True, size=0.036)
        kept = min(norm, 1.0)
        self.text(sc, x0 - 0.012, y, "gradient", op, anchor="rm", ink=dark(GREEN))
        self.rect(sc, self.shift(self.P(x0, y), unit * kept / 2, 0), unit * kept, 26, GREEN, op,
                  radius=3)
        if norm > 1.0:
            over = unit * (min(norm, 3.2) - 1.0)
            self.rect(sc, self.shift(self.P(x0, y), unit + over / 2, 0), over, 26, soft(RED, 0.35),
                      op, radius=3)
        wall = x0 + self.frac(unit)[0]
        self.polyline(sc, [self.P(wall, y - 0.055), self.P(wall, y + 0.055)], 2.6, INK, op)
        self.text(sc, wall, y + 0.082, "the clip: length 1", op, size=0.030)
        self.text(sc, 0.95, y + 0.082, "cut off: scaled by %.2f" % ratio if norm > 1.0
                  else "short enough: left alone", op, anchor="rm",
                  ink=dark(RED) if norm > 1.0 else dark(GREEN), size=0.034)
        chart = self.chart(((0.10, 0.62), (0.60, 0.84)), self.steps, np.r_[0.0, self.norm],
                           pad=0.04)
        self.axes(sc, chart, op, "training update", "gradient length before clipping")
        chart.hline(sc, self.P, 1.0, op, dashes=24, width=1.4)
        self.plot(sc, chart, self.steps, self.norm, GREEN, op, f, width=1.6)
        done = int((self.ratio[: i + 1] < 1.0).sum())
        self.lines(sc, 0.68, 0.66, [("clipped so far", None),
                                    ("%d of %d updates" % (done, i + 1), dark(RED))],
                   op, size=0.038, step=0.065)

    def phase_3(self, sc, u, op):
        last = len(self.held_at) - 1
        f = min(u / 0.86, 1.0)
        k = max(1, int(round(f * last)))
        # the first checkpoint (before any update, 4.6 nats) is left off so the rest can be read
        xs = self.held_at[1:]
        chart = self.chart(((0.10, 0.60), (0.36, 0.82)), xs,
                           np.r_[self.held[:3, :, 1:].ravel(), self.plateau + 0.3])
        self.axes(sc, chart, op, "training update", "loss on held-out text (nats per character)")
        chart.hline(sc, self.P, self.plateau, op, dashes=30, width=1.6)
        for arm in (0, 1, 2):
            ink = self.RUNS[arm][1]
            for seed in range(self.seeds):
                self.plot(sc, chart, xs[:k], self.held[arm, seed, 1:k + 1], ink, op, 1.0,
                          width=1.2, dot=k > 1)
        self.text(sc, 0.66, 0.36, "update %d" % self.held_at[k], op, anchor="lm", strong=True,
                  size=0.038)
        for arm in (0, 1, 2):
            name, ink = self.RUNS[arm]
            y = 0.455 + 0.125 * arm
            self.text(sc, 0.66, y, name, op, anchor="lm", ink=dark(ink), size=0.034)
            self.text(sc, 0.66, y + 0.048, "%d of %d seeds learning" % (
                self.below(arm, k), self.seeds), op, anchor="lm")
        self.lines(sc, 0.66, 0.83, ["dashed: guessing by letter", "frequency, %.2f nats"
                                    % self.plateau], op, step=0.042)
        self.text(sc, 0.10, 0.925, "one line per seed; batches are random windows of the "
                  "training text, the loss is on text kept apart", op * ramp(u, 0.5, 0.65),
                  anchor="lm")

    def phase_4(self, sc, u, op):
        x0, x1 = 0.33, 0.80
        lo, hi = 2.2, 3.6                         # the loss axis, in nats

        def at(value):
            return x0 + (x1 - x0) * (value - lo) / (hi - lo)
        self.text(sc, x0, 0.315, "best held-out loss of each run (a dot) and their mean (the bar)",
                  op, anchor="lm", strong=True)
        px = at(self.plateau)
        self.polyline(sc, [self.P(px, 0.35), self.P(px, 0.81)], 1.6, AXIS, op * 0.8)
        self.text(sc, px + 0.01, 0.835, "letter frequency: %.2f" % self.plateau, op, anchor="lm",
                  size=0.030)
        for v in (2.4, 2.8, 3.2):
            self.polyline(sc, [self.P(at(v), 0.35), self.P(at(v), 0.81)], 1.0, GRID, op * 0.55)
            self.text(sc, at(v), 0.835, "%.1f" % v, op, size=0.028)
        for row, (name, ink) in enumerate(self.RUNS):
            y = 0.40 + 0.115 * row
            a = (u - 0.1 * row) / 0.3
            b = ramp(a, 0.3, 1.0)
            self.text(sc, x0 - 0.02, y - 0.018, name, op, anchor="rm", ink=dark(ink), size=0.034)
            self.text(sc, x0 - 0.02, y + 0.026, "%d of %d learned" % (self.learned[row], self.seeds),  # noqa: E501
                      op * b, anchor="rm")
            self.polyline(sc, [self.P(x0, y), self.P(x1, y)], 1.2, AXIS, op * 0.6)
            m = at(self.mean[row])
            self.rect(sc, self.P(m, y), 5, 38, ink, op * b, radius=2)
            for v in self.best[row]:
                self.disk(sc, self.P(at(v), y), 6.0, ink, op * b, lift=1.4, outline=0.3)
            self.text(sc, x1 + 0.02, y, "%.2f" % self.mean[row], op * b, anchor="lm", strong=True,
                      size=0.038)
        self.text(sc, 0.05, 0.925, "the fixed-rate runs did not blow up: they settled on the "
                  "plateau at %.2f and stayed" % self.stuck, op * ramp(u, 0.6, 0.8), anchor="lm")


# ---------------------------------------------------------------------- 25


class Grokking(Mechanism):
    """A network started from weights three times the usual size learns 1,000
    MNIST digits by heart while the test digits stay wrong, then - as weight
    decay shrinks the weights - gets the test digits right too. The same run
    from the usual size shows no such wait."""

    TITLE = "Grokking: the test catches up late"
    PHASES = (("the task", 9.0), ("by heart", 12.0), ("the long wait", 15.0),
              ("the control", 9.0), ("measured", 11.0))
    TRAIN, TEST = BLUE, AMBER
    LARGE, DEFAULT = 0, 1
    TICKS = (1, 10, 100, 1000, 10000)

    def setup(self):
        from . import study_training as st

        s = self.study
        p = s.panels
        self.digits = np.asarray(p[0].data["values"], float)
        self.labels = list(p[0].data["labels"])
        self.steps = np.asarray(p[1].data["x"], float)
        self.train = np.asarray(p[1].data["train"], float)          # arm, seed, checkpoint
        self.test = np.asarray(p[1].data["test"], float)
        self.norm = np.asarray(p[2].data["each"], float)
        self.fitted = np.asarray(p[3].data["fitted"], float)        # arm, seed
        self.generalised = np.asarray(p[3].data["generalised"], float)
        self.test_then = np.asarray(p[3].data["test_then"], float)
        settings = s.metadata["settings"]
        self.sizes = [int(v) for v in settings["sizes"]]
        self.scale = float(settings["scales"][0])
        self.levels = float(settings["fitted_at"]), float(settings["generalised_at"])
        r = st.grokking_runs()
        self.train_ok, self.test_ok = r["train_ok"], r["test_ok"]   # arm, checkpoint, image
        self.n_train, self.n_test = self.train_ok.shape[2], self.test_ok.shape[2]
        # seed 0 of the large start, whose images the walls show
        self.fit_k = int(np.searchsorted(self.steps, self.fitted[0, 0]))
        self.late = float(self.generalised[0].mean() / self.fitted[0].mean())

    def subtitle_for(self, k):
        k0 = self.fit_k
        big, plain = self.LARGE, self.DEFAULT
        return [
            "grokking: a network fits its training examples first and gets new ones right only "
            "much later. The recipe here: %s MNIST digits, weights started %g times larger than "
            "usual, and weight decay" % (f"{self.n_train:,}", self.scale),
            "one cell per image, lit when its digit is named: by update %s the network has %d of "
            "the %s training images right, but only %d of the %s test images" % (
                f"{int(self.steps[k0]):,}", self.train_ok[0, k0].sum(),
                f"{self.n_train:,}", self.test_ok[0, k0].sum(),
                f"{self.n_test:,}"),
            "training goes on with nothing left to fit: weight decay shrinks the weights from "
            "%.0f to %.0f, and test accuracy climbs from %.0f%% to %.0f%%, passing %.0f%% only at "
            "update %s" % (
                self.norm[big, 0, k0], self.norm[big, 0, -1], 100 * self.test[big, 0, k0],
                100 * self.test[big, 0, -1], 100 * self.levels[1],
                f"{int(self.generalised[big, 0]):,}"),
            "the control starts from the usual weight size, everything else the same: its test "
            "accuracy passes %.0f%% near update %.0f, before training reaches %.0f%% near update "
            "%.0f. No wait" % (100 * self.levels[1], self.generalised[plain].mean(),
                               100 * self.levels[0], self.fitted[plain].mean()),
            "three seeds each. Large start: training at %.0f%% by update %s, test at %.0f%% by "
            "update %s, %.0f times later. Default start: the test gets there first" % (
                100 * self.levels[0], f"{self.fitted[big].mean():,.0f}",
                100 * self.levels[1], f"{self.generalised[big].mean():,.0f}", self.late),
        ][k]

    def _walls(self, sc, op, k, places, cell, short=False, arm=0):
        """Both walls of cells (seed 0) at checkpoint ``k``, with their counts above."""
        for (x, y, cols), ok, ink, name, px in zip(
                places, (self.train_ok[arm], self.test_ok[arm]), (self.TRAIN, self.TEST),
                ("training images", "test images"), cell):
            rows = ok.shape[1] // cols
            v = np.where(ok[k], 1.0, 0.0).reshape(rows, cols)
            w, h = cols * px, rows * px
            self.grid(sc, v, x, y, w, h, op, vmax=1.0, hot=ink, lines=0.0)
            dh = self.frac(0, h)[1]
            count = "%s of %s" % (f"{int(ok[k].sum()):,}", f"{ok.shape[1]:,}")
            self.text(sc, x, y - dh / 2 - 0.04, count if short else "%s %s right" % (count, name),
                      op, ink=dark(ink, 0.3), size=0.034)
            if short:
                self.text(sc, x, y - dh / 2 - 0.085, name.split()[0], op, ink=dark(ink, 0.3))

    def _chart(self, sc, op):
        chart = self.chart(((0.10, 0.56), (0.36, 0.76)), np.r_[1.0, self.steps[-1]],
                           np.r_[0.0, 1.0], pad=0.03, xlog=True)
        self.axes(sc, chart, op, "training update (each step to the right is 10 times more)",
                  "share of images right",
                  xticks=[(v, f"{v:,}") for v in self.TICKS],
                  yticks=[(v, "%d%%" % (100 * v)) for v in (0.0, 0.25, 0.5, 0.75, 1.0)])
        return chart

    def _curves(self, sc, chart, arm, k, op, width=1.4, pale=0.0):
        """An arm's three seeds, training and test, up to checkpoint ``k`` (from update 1)."""
        for values, ink in ((self.train, self.TRAIN), (self.test, self.TEST)):
            ink = soft(ink, pale) if pale else ink
            for seed in range(values.shape[1]):
                self.plot(sc, chart, self.steps[1:k + 1], values[arm, seed, 1:k + 1], ink, op,
                          1.0, width=width, dot=not pale)

    def phase_1(self, sc, u, op):
        n = len(self.digits)
        for i in range(n):
            a = ramp(u, 0.04 + 0.06 * i, 0.12 + 0.06 * i)
            x = 0.10 + 0.094 * i
            self.digit(sc, self.digits[i], x, 0.45, 72, op * a, frame=self.TRAIN)
            self.text(sc, x, 0.57, self.labels[i], op * a, strong=True, size=0.038)
        self.text(sc, 0.05, 0.31, "MNIST digits, shrunk to 14 x 14 pixels", op, anchor="lm",
                  strong=True, size=0.034)
        b = ramp(u, 0.42, 0.52)
        self.arrow(sc, (0.625, 0.45), (0.66, 0.45), INK, op * b, ramp(u, 0.42, 0.55), 2.6)
        heights = (150, 150, 150, 46)
        for j, (size, hpx) in enumerate(zip(self.sizes, heights)):
            x = 0.70 + 0.075 * j
            self.box(sc, x, 0.45, 50, hpx, soft(VIOLET, 0.7), op * b, "%d" % size, border=VIOLET,
                     size=0.036, ink=dark(VIOLET, 0.3))
            if j:
                self.arrow(sc, (x - 0.046, 0.45), (x - 0.029, 0.45), VIOLET, op * b, 1.0, 2.2)
        self.text(sc, 0.81, 0.65, "pixels, two hidden layers, digits", op * b)
        self.pulse(sc, (0.625, 0.45), (0.66, 0.45), (u * 5) % 1.0, self.TRAIN, op * b)
        c = ramp(u, 0.6, 0.72)
        self.lines(sc, 0.05, 0.74, [
            ("%s images to train on, shown again and again" % f"{self.n_train:,}",
             dark(self.TRAIN)),
            ("%s test images it never trains on" % f"{self.n_test:,}",
             dark(self.TEST, 0.3)),
            ("every weight starts %g times larger than usual, and weight decay pulls it back"
             % self.scale, dark(VIOLET))],
            op * c, size=0.034, step=0.07)

    def phase_2(self, sc, u, op):
        k, _ = turn_of(u, self.fit_k + 1, 0.9)
        self._walls(sc, op, k, ((0.235, 0.63, 40), (0.715, 0.63, 50)), (7.0, 6.0))
        self.text(sc, 0.05, 0.925, "update %s" % f"{int(self.steps[k]):,}", op,
                  anchor="lm", strong=True, size=0.040)
        self.text(sc, 0.95, 0.925, "a lit cell: that image's digit is named correctly", op,
                  anchor="rm")

    def phase_3(self, sc, u, op):
        last = len(self.steps) - 1
        k = max(1, int(round(min(u / 0.88, 1.0) * last)))
        chart = self._chart(sc, op)
        self._curves(sc, chart, self.LARGE, k, op)
        self._walls(sc, op, k, ((0.675, 0.52, 40), (0.86, 0.52, 50)), (3.6, 3.2), short=True)
        self.lines(sc, 0.10, 0.905, [("update %s:  training %.1f%%   test %.1f%%" % (
            f"{int(self.steps[k]):,}", 100 * self.train[0, :, k].mean(),
            100 * self.test[0, :, k].mean()), None)], op, size=0.036)
        full = self.span(0.26)[0]
        norm = self.norm[0, :, k].mean()
        self.text(sc, 0.60, 0.80, "size of the weights", op, anchor="lm")
        self.hbar(sc, 0.60, 0.86, full * norm / self.norm.max(), VIOLET, op, "", "%.0f" % norm,
                  thick=26.0, track=full)

    def phase_4(self, sc, u, op):
        last = len(self.steps) - 1
        k = max(1, int(round(min(u / 0.8, 1.0) * last)))
        chart = self._chart(sc, op)
        self._curves(sc, chart, self.LARGE, last, op, width=1.0, pale=0.6)
        self._curves(sc, chart, self.DEFAULT, k, op)
        self.lines(sc, 0.62, 0.36, [
            ("default start", None),
            ("training", dark(self.TRAIN)),
            ("test", dark(self.TEST, 0.3)),
            ("the pale curves: the", None),
            ("%g times larger start" % self.scale, None)], op, size=0.034, step=0.052)
        self.lines(sc, 0.62, 0.68, [
            ("same images, same batches,", None), ("same weight decay", None)], op,
            step=0.045)
        self.text(sc, 0.10, 0.905, "update %s:  training %.1f%%   test %.1f%%" % (
            f"{int(self.steps[k]):,}", 100 * self.train[1, :, k].mean(),
            100 * self.test[1, :, k].mean()), op, anchor="lm", size=0.036)

    def phase_5(self, sc, u, op):
        x0 = 0.44
        full = self.span(0.40)[0]
        top = np.log10(self.steps[-1])

        def at(value):
            return x0 + self.frac(full * np.log10(max(value, 1.0)) / top)[0]
        self.text(sc, x0, 0.315, "when each accuracy arrives (three seeds and their "
                  "mean)", op, anchor="lm", strong=True)
        rows = ((self.LARGE, "%g times larger start" % self.scale),
                (self.DEFAULT, "default start"))
        for i, (arm, name) in enumerate(rows):
            y0 = 0.42 + 0.20 * i
            self.text(sc, 0.05, y0 + 0.035, name, op, anchor="lm", strong=True, size=0.034)
            for j, (values, ink, what) in enumerate((
                    (self.fitted[arm], self.TRAIN, "training %.0f%%" % (100 * self.levels[0])),
                    (self.generalised[arm], self.TEST, "test %.0f%%" % (100 * self.levels[1])))):
                y = y0 + 0.07 * j
                a = (u - 0.1 * (2 * i + j)) / 0.3
                self.text(sc, x0 - 0.012, y, what, op, anchor="rm", ink=dark(ink, 0.3))
                self.hbar(sc, x0, y, full * np.log10(values.mean()) / top, soft(ink, 0.3), op, "",
                          "", a, thick=24.0, track=full)
                for v in values:
                    self.disk(sc, self.P(at(v), y), 4.5, ink, op * ramp(a, 0.7, 1.0), lift=1.4,
                              outline=0.3)
                self.text(sc, x0 + self.frac(full)[0] + 0.012, y,
                          f"{values.mean():,.0f}", op * ramp(a, 0.6, 1.0), anchor="lm",
                          strong=True, size=0.034)
        for v in self.TICKS:
            self.text(sc, at(v), 0.80, f"{v:,}", op, size=0.028)
        end = [100 * self.test[arm, :, -1].mean() for arm in (0, 1)]
        self.text(sc, 0.05, 0.875, "after %s updates the test images right: %.1f%% from the "
                  "large start, %.1f%% from the default" % (
                      f"{int(self.steps[-1]):,}", end[0], end[1]),
                  op * ramp(u, 0.6, 0.75), anchor="lm")
        self.text(sc, 0.05, 0.925, "the late rise is real, but it does not end higher than "
                  "an ordinary start" if end[0] <= end[1] + 0.005 else
                  "the late rise ends above the ordinary start", op * ramp(u, 0.7, 0.85),
                  anchor="lm")


# ---------------------------------------------------------------------- 26


def stretches(best):
    """Runs of equal values in ``best`` as ``(value, first index, last index)``."""
    out = []
    for i, v in enumerate(best):
        if out and out[-1][0] == v:
            out[-1][2] = i
        else:
            out.append([v, i, i])
    return [tuple(s) for s in out]


class ScalingLaws(Mechanism):
    """Four model sizes, each trained to a ladder of compute budgets: what one
    budget buys each size, the grid of runs, loss against compute with a power
    law fitted to the cheap budgets and tested on the dear ones, and the best
    size at each budget."""

    TITLE = "Scaling laws, from a ladder of budgets"
    PHASES = (("one budget", 11.0), ("the runs", 14.0), ("against compute", 14.0),
              ("the best size", 12.0))
    INKS = (BLUE, GREEN, AMBER, RED)             # one per model size
    X0, Y0, DX, DY = 0.305, 0.455, 0.0685, 0.105   # the grid: first cell's centre, pitch
    CELL = (58.0, 44.0)
    SHOWN = 4                                    # the budget whose shapes phase 1 draws

    def setup(self):
        s = self.study
        p = s.panels
        self.loss = np.asarray(p[0].data["values"], float)        # width, budget (0: not run)
        self.ran = np.asarray(p[0].data["ran"], bool)
        self.compute = np.asarray(p[0].data["compute"], float)
        self.characters = np.asarray(p[0].data["characters"], float)
        self.updates = np.asarray(p[0].data["updates"], int)
        self.weights = [int(v) for v in p[0].data["weights"]]
        self.widths = [int(v) for v in s.metadata["settings"]["widths"]]
        self.seeds = int(s.metadata["settings"]["seeds"])
        self.fit_levels = int(s.metadata["settings"]["fit_levels"])
        self.runs = int(self.ran.sum())
        self.front_x = np.asarray(p[3].data["x"], float)
        self.front_y = np.asarray(p[3].data["ys"], float)[0]
        self.best = [int(v) for v in p[3].data["best"]]
        self.predicted = np.exp(np.asarray(p[2].data["ys"], float)[1])
        self.slope = float(p[2].data["slope"])
        self.error = np.asarray(p[2].data["held_out_error"], float)
        self.budget = ["%dx" % 2 ** j for j in range(self.loss.shape[1])]

    def best_by_budget(self):
        """The frontier in words: which width is best over which budgets."""
        parts = []
        for width, a, b in stretches([self.widths[i] for i in self.best]):
            span = self.budget[a] if a == b else "%s to %s" % (self.budget[a], self.budget[b])
            parts.append("width %d at %s" % (width, span))
        return ", ".join(parts)

    def subtitle_for(self, k):
        j = self.SHOWN
        widest = len(self.widths) - 1
        return [
            "compute is about weights times characters read, so one budget buys a small model "
            "far more text: at the same budget width %d reads %s characters and width %d only "
            "%s" % (self.widths[0], f"{self.characters[0, j]:,.0f}", self.widths[-1],
                    f"{self.characters[-1, j]:,.0f}"),
            "%d runs, each the mean of %d seeds: %d model sizes trained to compute budgets that "
            "double, random windows of the text. Held-out loss falls from %.2f to %.2f nats" % (
                self.runs, self.seeds, len(self.widths), self.loss.max(),
                self.loss[self.ran].min()),
            "loss falls as compute grows, but the line bends: a power law fitted to the %d "
            "cheapest budgets (slope %.2f) predicts %.2f nats at the dearest, and the run there "
            "measured %.2f" % (self.fit_levels, self.slope, self.predicted[-1],
                               self.front_y[-1]),
            "the best size at each budget, of those run: %s. %s" % (
                self.best_by_budget(),
                "The widest model is never the best use of the compute" if widest not in self.best
                else "The best size grows with the budget"),
        ][k]

    def _cell(self, r, c):
        return self.X0 + self.DX * c, self.Y0 + self.DY * r

    def _headers(self, sc, op):
        self.text(sc, 0.05, 0.325, "model", op, anchor="lm", strong=True)
        self.text(sc, self.X0 - 0.035, 0.325, "compute budget (1x is the smallest):", op,
                  anchor="lm", strong=True)
        for r, width in enumerate(self.widths):
            _x, y = self._cell(r, 0)
            side = 0.115 * np.sqrt(self.weights[r])
            self.box(sc, 0.075, y, side, side, self.INKS[r], op, radius=3)
            self.text(sc, 0.115, y - 0.019, "width %d" % width, op, anchor="lm",
                      ink=dark(self.INKS[r]), size=0.032)
            self.text(sc, 0.115, y + 0.021, "%s weights" % f"{self.weights[r]:,}", op,
                      anchor="lm")
        for c, name in enumerate(self.budget):
            x, _ = self._cell(0, c)
            self.text(sc, x, 0.375, name, op, strong=True, size=0.030)

    def _cells(self, sc, op, filled, ring=()):
        lo, hi = self.loss[self.ran].min(), self.loss.max()
        w, h = self.CELL
        for r in range(self.loss.shape[0]):
            for c in range(self.loss.shape[1]):
                x, y = self._cell(r, c)
                if not self.ran[r, c]:
                    self.box(sc, x, y, w, h, DIM, op * 0.35, radius=4)
                    continue
                a = filled(r, c)
                q = (self.loss[r, c] - lo) / (hi - lo)
                fill = tuple(PAPER + (np.asarray(soft(self.INKS[r], 0.25)) - PAPER) * (1 - q) * a)
                on = (r, c) in ring
                self.box(sc, x, y, w, h, fill, op, "%.2f" % self.loss[r, c] if a > 0.5 else None,
                         border=INK if on else AXIS, size=0.030, radius=4)
                if on:
                    self.rect(sc, self.P(x, y), w + 5, h + 5, INK, op, border=INK, border_w=3.0,
                              radius=6, lift=0.7, fill_op=0.0)

    def phase_1(self, sc, u, op):
        j = self.SHOWN
        # one budget as four shapes of the same area: weights (height) x characters (width)
        chars, weights = self.characters[:, j], np.asarray(self.weights, float)
        wide = self.span(0.36)[0] / chars.max()
        tall = 100.0 / weights.max()
        x0, y = 0.30, 0.36
        self.text(sc, 0.05, 0.31, "one budget of compute, spent four ways", op, anchor="lm",
                  strong=True, size=0.034)
        for r, width in enumerate(self.widths):
            a = ramp(u, 0.06 + 0.17 * r, 0.20 + 0.17 * r)
            h, w = max(weights[r] * tall, 2.5), chars[r] * wide * a
            dh = max(self.frac(0, h)[1], 0.035)
            yc = y + dh / 2
            self.text(sc, 0.05, yc, "width %d" % width, op * a, anchor="lm",
                      ink=dark(self.INKS[r]), size=0.034)
            self.text(sc, 0.285, yc, "%s weights" % f"{self.weights[r]:,}", op * a,
                      anchor="rm")
            self.rect(sc, self.shift(self.P(x0, yc), w / 2, 0), max(w, 0.5), h, self.INKS[r],
                      op * a, radius=2)
            self.text(sc, x0 + self.frac(w)[0] + 0.012, yc, "%s characters, %d updates" % (
                f"{chars[r]:,.0f}", self.updates[r, j]), op * ramp(a, 0.7, 1.0),
                anchor="lm")
            y += dh + 0.035
        b = ramp(u, 0.78, 0.9)
        self.lines(sc, 0.05, 0.875, [
            "height: weights.  length: characters read.  the four areas are the same compute",
            "the ladder has %d such budgets, each double the last" % len(self.budget)],
            op * b, step=0.048)

    def phase_2(self, sc, u, op):
        order = [(r, c) for c in range(self.loss.shape[1]) for r in range(self.loss.shape[0])
                 if self.ran[r, c]]
        turn, within = turn_of(u, len(order), 0.9)
        r0, c0 = order[turn]
        rank = {cell: i for i, cell in enumerate(order)}
        self._headers(sc, op)

        def filled(r, c):
            i = rank[(r, c)]
            return 1.0 if i < turn else ramp(within, 0.4, 0.9) if i == turn else 0.0
        self._cells(sc, op, filled)
        x, y = self._cell(r0, c0)
        if u < 0.9:
            self.pulse(sc, (0.22, y), (x, y), within / 0.6, self.INKS[r0], op, r=6.0)
        self.text(sc, 0.05, 0.925, "width %d at budget %s:  %d updates, %s characters,  held-out "
                  "loss %.2f nats" % (self.widths[r0], self.budget[c0], self.updates[r0, c0],
                                      f"{self.characters[r0, c0]:,.0f}",
                                      self.loss[r0, c0]), op, anchor="lm", strong=True,
                  size=0.034)

    def phase_3(self, sc, u, op):
        k = self.fit_levels
        # room around the points: the axes are logarithmic, which chart() does not pad
        chart = self.chart(((0.12, 0.62), (0.34, 0.80)),
                           np.r_[self.front_x[0] * 0.75, self.front_x[-1] * 1.4],
                           np.r_[min(self.predicted.min(), self.front_y.min()) * 0.95,
                                 self.loss.max() * 1.04], xlog=True, ylog=True)
        self.axes(sc, chart, op, "estimated compute to train (operations, log scale)",
                  "held-out loss (nats, log scale)",
                  xticks=[(v, number(v)) for v in (1e9, 1e10, 1e11)],
                  yticks=[(v, "%g" % v) for v in (2, 2.5, 3, 3.5)])
        rows, _cols = self.loss.shape
        for r in range(rows):
            at = np.flatnonzero(self.ran[r])
            a = ramp(u, 0.03 + 0.09 * r, 0.25 + 0.09 * r)
            n = max(1, int(round(a * len(at))))
            if a > 0:
                self.plot(sc, chart, self.compute[r, at[:n]], self.loss[r, at[:n]], self.INKS[r],
                          op, 1.0, width=1.3, dot=False)
                for c in at[:n]:
                    self.disk(sc, self.P(chart.fx(self.compute[r, c]), chart.fy(self.loss[r, c])),
                              4.5, self.INKS[r], op, lift=1.2, outline=0.2)
        f = ramp(u, 0.48, 0.62)
        if f > 0:
            a = (chart.fx(self.front_x[0]), chart.fy(self.predicted[0]))
            b = (chart.fx(self.front_x[k - 1]), chart.fy(self.predicted[k - 1]))
            self.arrow(sc, a, b, INK, op, f, 2.8, head=False)
        g = ramp(u, 0.66, 0.8)
        if g > 0:
            # the same line carried on past the budgets it was fitted to
            a = np.array([chart.fx(self.front_x[k - 1]), chart.fy(self.predicted[k - 1])])
            b = np.array([chart.fx(self.front_x[-1]), chart.fy(self.predicted[-1])])
            dashes = 9
            for i in range(int(dashes * g + 1e-9)):
                p, q = a + (b - a) * (i / dashes), a + (b - a) * ((i + 0.55) / dashes)
                self.polyline(sc, [self.P(*p), self.P(*q)], 2.8, INK, op)
            if g >= 1.0:
                top = self.best[-1]
                self.disk(sc, self.P(*b), 6.0, PAPER, op, ring=INK, lift=1.3, outline=0.0)
                self.disk(sc, self.P(chart.fx(self.front_x[-1]), chart.fy(self.front_y[-1])),
                          7.5, self.INKS[top], op, glow=0.5, lift=1.5, outline=0.3)
        x = 0.68
        self.text(sc, x, 0.33, "model width", op, anchor="lm", strong=True)
        for r in range(rows):
            self.box(sc, x + 0.012, 0.385 + 0.048 * r, 16, 16, self.INKS[r], op, radius=8)
            self.text(sc, x + 0.035, 0.385 + 0.048 * r, "%d" % self.widths[r], op, anchor="lm",
                      ink=dark(self.INKS[r]), size=0.032)
        self.lines(sc, x, 0.61, [("power law fitted to the", None),
                                 ("%d cheapest budgets" % k, None)], op * f, size=0.032,
                   step=0.046)
        self.lines(sc, x, 0.72, [("carried on (dashed), it says", None),
                                 ("%.2f at the top budget" % self.predicted[-1], None),
                                 ("measured there: %.2f" % self.front_y[-1],
                                  dark(self.INKS[self.best[-1]]))],
                   op * g, size=0.032, step=0.046)

    def phase_4(self, sc, u, op):
        cols = self.loss.shape[1]
        turn, _ = turn_of(u, cols, 0.85)
        self._headers(sc, op)
        ring = [(self.best[c], c) for c in range(turn + 1)]
        self._cells(sc, op, lambda r, c: 1.0, ring=ring)
        c = turn
        others = ",  ".join("width %d: %.2f" % (self.widths[r], self.loss[r, c])
                            for r in range(len(self.widths))
                            if self.ran[r, c] and r != self.best[c])
        self.text(sc, 0.05, 0.925, "at budget %s the best is width %d:  %.2f nats%s" % (
            self.budget[c], self.widths[self.best[c]], self.loss[self.best[c], c],
            "   (%s)" % others if others else ""), op, anchor="lm", strong=True, size=0.034)


# ---------------------------------------------------------------------- 27


def mix_counts(mode, step, batch=24, steps=320):
    """How many of a batch's ``batch`` excerpts come from source A at update
    ``step``, by the study's rule for each recipe."""
    ratio = 0.9 if mode == "90/10" else 0.5 if mode == "50/50" else 0.9 - 0.8 * step / steps
    return int(round(batch * ratio))


class CurriculumAndDataMixing(Mechanism):
    """Two halves of Tiny Shakespeare as two sources, three recipes for
    filling each batch of 24 from them, and what each recipe does to the loss
    on either source's held-out text."""

    TITLE = "Data mixing and curriculum"
    PHASES = (("two sources", 9.0), ("the mix", 13.0), ("source A", 12.0), ("source B", 12.0),
              ("both, equally", 10.0))
    A_INK, B_INK = BLUE, AMBER
    RECIPES = (("90/10", RED), ("50/50", GREEN), ("curriculum", VIOLET))
    BATCH = 24

    def setup(self):
        s = self.study
        pair = np.asarray(s.panels[0].data["values"], int)
        self.a_text, self.b_text = glyphs(pair[0][:12]), glyphs(pair[1][:12])
        self.steps = np.asarray(s.panels[1].data["x"], float)
        self.on_a = np.asarray(s.panels[1].data["ys"], float)
        self.on_b = np.asarray(s.panels[2].data["ys"], float)
        self.mean = [float(v) for v in s.panels[3].data["values"]]
        self.total = int(self.steps[-1])

    def subtitle_for(self, k):
        last = self.total
        return [
            "two sources of text: the first and second halves of Tiny Shakespeare, each with "
            "its own last fifth held out for testing",
            "every batch has %d excerpts, and the recipe says where they come from: always %d "
            "from A, always %d, or a curriculum that slides from %d down to %d" % (
                self.BATCH, mix_counts("90/10", 0), mix_counts("50/50", 0),
                mix_counts("curriculum", 0), mix_counts("curriculum", last)),
            "held-out loss on source A after %d updates: %.2f for 90/10, %.2f for 50/50, %.2f "
            "for the curriculum; the recipe that shows A most ends worst on it" % (
                last, *self.on_a[:, -1]),
            "on source B: %.2f for 90/10, %.2f for 50/50, %.2f for the curriculum; 90/10 saw "
            "the fewest B excerpts and its loss turns back up" % tuple(self.on_b[:, -1]),
            "averaging the two sources equally: 90/10 %.2f, 50/50 %.2f, curriculum %.2f nats; "
            "the even mix is best in this run" % tuple(self.mean),
        ][k]

    def _strip(self, sc, x0, y, mode, step, op, cell=22.0, gap=2.0, h=None):
        """A batch as ``BATCH`` cells, A's first; returns the layout x after it."""
        n = mix_counts(mode, step, self.BATCH, self.total)
        dx = self.frac(cell + gap)[0]
        for i in range(self.BATCH):
            self.box(sc, x0 + dx * (i + 0.5), y, cell, cell if h is None else h,
                     self.A_INK if i < n else self.B_INK, op, radius=3)
        return x0 + dx * self.BATCH, n

    def phase_1(self, sc, u, op):
        full = self.span(0.90)[0]
        y = 0.37
        for half, (ink, name) in enumerate(((self.A_INK, "source A: the first half"),
                                            (self.B_INK, "source B: the second half"))):
            a = ramp(u, 0.05 + 0.3 * half, 0.2 + 0.3 * half)
            x0 = 0.05 + 0.45 * half
            train, held = full / 2 * 0.8, full / 2 * 0.2
            self.rect(sc, self.shift(self.P(x0, y), train / 2, 0), train - 3, 34, ink, op * a,
                      radius=4)
            self.rect(sc, self.shift(self.P(x0, y), train + held / 2, 0), held - 3, 34,
                      soft(ink, 0.6), op * a, border=ink, radius=4)
            self.text(sc, x0 + 0.004, y - 0.065, name, op * a, anchor="lm", ink=dark(ink, 0.3),
                      size=0.034)
            self.text(sc, x0 + 0.45 * 0.9, y + 0.062, "held out", op * a, ink=dark(ink, 0.3))
            self.text(sc, x0 + 0.004, y + 0.062, "trained on", op * a, anchor="lm",
                      ink=dark(ink, 0.3))
            b = ramp(u, 0.25 + 0.3 * half, 0.4 + 0.3 * half)
            yy = 0.62 + 0.16 * half
            self.text(sc, 0.05, yy - 0.068, "an excerpt of %s" % name.split(":")[0], op * b,
                      anchor="lm", ink=dark(ink, 0.3))
            self.tokens(sc, self.a_text if half == 0 else self.b_text, 0.05, 0.71, yy, op * b,
                        inks=[ink], w=44.0, h=42.0, size=0.038)
        self.text(sc, 0.05, 0.91, "the same play collection, but different plays and speakers "
                  "in each half", op * ramp(u, 0.7, 0.85), anchor="lm")

    def phase_2(self, sc, u, op):
        step = int(round(min(u / 0.88, 1.0) * self.total))
        self.text(sc, 0.05, 0.315, "update %d of %d" % (step, self.total), op, anchor="lm",
                  strong=True, size=0.038)
        self.box(sc, 0.52, 0.315, 150, 30, self.A_INK, op, "source A", ink=WHITE, size=0.032)
        self.box(sc, 0.80, 0.315, 150, 30, self.B_INK, op, "source B", ink=WHITE, size=0.032)
        for row, (mode, ink) in enumerate(self.RECIPES):
            y = 0.46 + 0.15 * row
            end, n = self._strip(sc, 0.27, y, mode, step, op)
            self.text(sc, 0.05, y - 0.022, mode, op, anchor="lm", ink=dark(ink), size=0.038)
            self.text(sc, 0.05, y + 0.028, "%d from A, %d from B" % (n, self.BATCH - n), op,
                      anchor="lm")
            dx = (end - 0.27) / self.BATCH
            beat = (u * 9 + 0.3 * row) % 1.0
            self.pulse(sc, (0.52, 0.35), (0.27 + dx * n / 2, y - 0.03), beat, self.A_INK, op)
            self.pulse(sc, (0.80, 0.35), (0.27 + dx * (n + self.BATCH) / 2, y - 0.03), beat,
                       self.B_INK, op)
        self.text(sc, 0.27, 0.90, "one batch of %d excerpts per update, drawn afresh each time"
                  % self.BATCH, op, anchor="lm")

    def _measured(self, sc, u, op, curves, source, ink):
        f = min(u / 0.86, 1.0)
        k = int(round(f * (len(self.steps) - 1)))
        step = int(self.steps[k])
        self.text(sc, 0.05, 0.315, "update %d" % step, op, anchor="lm", strong=True, size=0.036)
        for row, (mode, rink) in enumerate(self.RECIPES):
            y = 0.385 + 0.058 * row
            self.text(sc, 0.05, y, mode, op, anchor="lm", ink=dark(rink), size=0.032)
            self._strip(sc, 0.20, y, mode, step, op, cell=15.0, gap=1.5)
            self.text(sc, 0.95, y, "loss on %s  %.2f" % (source, curves[row][k]), op, anchor="rm",
                      ink=dark(rink), size=0.032)
        first = 2                                  # the fall from 4.7 in the first 16 updates is off the scale  # noqa: E501
        xs = self.steps[first:]
        chart = self.chart(((0.12, 0.90), (0.61, 0.845)), xs, np.r_[self.on_a, self.on_b][:, first:])  # noqa: E501
        self.axes(sc, chart, op, "training update",
                  "loss on held-out text of source %s (nats per character)" % source)
        upto = (step - xs[0]) / (xs[-1] - xs[0])
        for row, (_mode, rink) in enumerate(self.RECIPES):
            self.plot(sc, chart, xs, curves[row][first:], rink, op, upto, width=2.2)
        self.box(sc, 0.93, 0.315, 30, 30, ink, op, source, ink=WHITE, size=0.034)

    def phase_3(self, sc, u, op):
        self._measured(sc, u, op, self.on_a, "A", self.A_INK)

    def phase_4(self, sc, u, op):
        self._measured(sc, u, op, self.on_b, "B", self.B_INK)

    def phase_5(self, sc, u, op):
        full = self.span(0.34)[0]
        step = int(round(min(u / 0.8, 1.0) * self.total))
        self.text(sc, 0.52, 0.33, "held-out loss, both sources weighted equally (nats)", op,
                  anchor="lm", strong=True)
        self.text(sc, 0.05, 0.33, "the batch at update %d" % step, op, anchor="lm", strong=True)
        for row, ((mode, ink), value) in enumerate(zip(self.RECIPES, self.mean)):
            y = 0.46 + 0.15 * row
            self.text(sc, 0.05, y - 0.05, mode, op, anchor="lm", ink=dark(ink), size=0.034)
            self._strip(sc, 0.05, y, mode, step, op, cell=14.0, gap=1.5, h=24.0)
            self.hbar(sc, 0.52, y, full * value / 3.4, ink, op, "", "%.2f" % value,
                      (u - 0.12 * row) / 0.35, thick=34.0, track=full)
        low = int(np.argmin(self.on_a[0]))
        self.text(sc, 0.05, 0.90, "90/10's loss on source A was lowest at update %d (%.2f) and "
                  "had climbed back to %.2f by the end" % (self.steps[low], self.on_a[0][low],
                                                           self.on_a[0][-1]),
                  op * ramp(u, 0.6, 0.8), anchor="lm")


# ---------------------------------------------------------------------- 28


@lru_cache(maxsize=1)
def bpe_demo(start=9, length=22):
    """The tokenizer study's BPE again: the 96 merges it learns from its
    training text (the first 16 and 48 are its smaller vocabularies), and a
    phrase of its held-out passage segmented after every merge."""
    from . import next_language as nl
    from . import toycorpus

    train = nl._corpus()["train"][:18000]
    test = nl._corpus()["test"][:4000]
    rules = toycorpus.bpe_train(train, 96)
    phrase = test[start:start + length]
    return dict(rules=rules, phrase=phrase, test=test, alphabet=len(set(train)),
                segments=[toycorpus.bpe_apply(phrase, rules[:k]) for k in range(len(rules) + 1)])


def show(token):
    """A token of several characters as it is drawn."""
    return "".join(glyph(ch) for ch in token)


class TokenizerEffects(Mechanism):
    """One phrase as characters and as bytes, BPE merging its commonest pairs
    one rule at a time, the shorter sequences and bigger tables that buys,
    and the loss per character of a model trained on each."""

    TITLE = "What the tokenizer changes"
    PHASES = (("characters, bytes", 9.0), ("merging pairs", 16.0), ("shorter, bigger", 12.0),
              ("per character", 12.0))
    INKS = (BLUE, TEAL, GREEN, AMBER, VIOLET)    # characters, BPE16, BPE48, BPE96, bytes
    MERGES = (0, 16, 48, 96)
    X0, X1 = 0.05, 0.95

    def setup(self):
        s = self.study
        self.names = list(s.panels[0].data["labels"])
        self.tokens_n = [int(v) for v in s.panels[0].data["values"]]
        self.table = [int(v) for v in s.panels[1].data["values"]]
        self.per_char = [float(v) for v in s.panels[2].data["values"]]
        self.curves = np.asarray(s.panels[3].data["ys"], float)
        self.steps = np.asarray(s.panels[3].data["x"], float) + 1
        self.chars = int(s.metadata["settings"]["test_characters"])
        self.width = 16                            # the study's embedding width
        d = bpe_demo()
        self.rules, self.phrase, self.segments = d["rules"], d["phrase"], d["segments"]
        self.alphabet = d["alphabet"]
        self.best = int(np.argmin(self.per_char))

    def subtitle_for(self, k):
        n = len(self.phrase)
        return [
            "the simplest tokens are single characters, or the bytes that store them; for this "
            "English text they are the same %d pieces, drawn from %d symbols" % (n, self.alphabet),
            "byte-pair encoding joins the commonest neighbouring pair into one new token, again "
            "and again: after 16, 48 and 96 merges this phrase is %d, %d and %d tokens" % tuple(
                len(self.segments[m]) for m in self.MERGES[1:]),
            "a held-out passage of %s characters shrinks to %s tokens with 96 merges, while "
            "the embedding table grows from %d rows to %d" % (
                f"{self.chars:,}", f"{self.tokens_n[3]:,}",
                self.table[0] // self.width, self.table[3] // self.width),
            "a token's loss is only fair once shared out per character: %.2f nats for "
            "characters and bytes, %.2f, %.2f and %.2f for BPE with 16, 48 and 96 merges" % (
                self.per_char[0], *self.per_char[1:4]),
        ][k]

    def _segments(self, sc, tokens, y, op, h=42.0, size=0.036, pitch=None, x0=None, text=True,
                  ink_of=None):
        """A segmentation drawn to scale: each token as wide as its characters."""
        n = len(self.phrase)
        x0 = self.X0 if x0 is None else x0
        pitch = (self.X1 - self.X0) / n if pitch is None else pitch
        px = self.span(pitch)[0]
        at = 0
        for tok in tokens:
            x = x0 + pitch * (at + len(tok) / 2)
            ink = None if ink_of is None else ink_of(tok)
            self.box(sc, x, y, px * len(tok) - 4, h, PAPER if ink is None else soft(ink), op,
                     show(tok) if text else None, ink=None if ink is None else dark(ink, 0.35),
                     border=AXIS if ink is None else ink, size=size, radius=5 if text else 3)
            at += len(tok)

    def _ink_of(self, tok):
        """A merged token's colour: the smallest of the three vocabularies that has it."""
        if len(tok) == 1:
            return None
        made = [a + b for a, b in self.rules].index(tok)
        return self.INKS[1] if made < 16 else self.INKS[2] if made < 48 else self.INKS[3]

    def phase_1(self, sc, u, op):
        n = len(self.phrase)
        pitch = (self.X1 - self.X0) / n
        self.text(sc, self.X0, 0.33, "%d characters" % n, op, anchor="lm", ink=dark(self.INKS[0]),
                  size=0.036)
        for i, ch in enumerate(self.phrase):
            x = self.X0 + pitch * (i + 0.5)
            self.box(sc, x, 0.42, 34, 44, soft(self.INKS[0]), op, glyph(ch),
                     ink=dark(self.INKS[0], 0.35), border=self.INKS[0], size=0.038)
            a = ramp(u, 0.15 + 0.02 * i, 0.25 + 0.02 * i)
            self.arrow(sc, (x, 0.47), (x, 0.565), self.INKS[4], op * a, a, 2.0)
            self.box(sc, x, 0.62, 34, 44, soft(self.INKS[4]), op * a, "%d" % ord(ch),
                     ink=dark(self.INKS[4], 0.35), border=self.INKS[4], size=0.029)
        b = ramp(u, 0.3, 0.45)
        self.text(sc, self.X0, 0.71, "%d bytes: the number each character is stored as" % n, op * b,
                  anchor="lm", ink=dark(self.INKS[4]), size=0.036)
        self.text(sc, self.X0, 0.86, "many small pieces, and a small table: one row for each of "
                  "the %d symbols, plus one for unknowns" % self.alphabet, op * ramp(u, 0.7, 0.85),
                  anchor="lm")

    def phase_2(self, sc, u, op):
        # the first 16 merges slowly, the rest faster
        f = min(u / 0.9, 1.0)
        merges = f / 0.45 * 16 if f < 0.45 else 16 + (f - 0.45) / 0.55 * 80
        k = min(int(merges), len(self.rules) - 1)
        within = min(merges - k, 1.0) if merges < len(self.rules) else 1.0
        a, b = self.rules[k]
        y = 0.385
        self.text(sc, self.X0, 0.315, "merge %d: the commonest neighbouring pair becomes one token"
                  % (k + 1), op, anchor="lm", strong=True, size=0.034)
        gap = 0.006 + 0.04 * (1.0 - ramp(within, 0.2, 0.7))
        ink = self.INKS[1] if k < 16 else self.INKS[2] if k < 48 else self.INKS[3]
        wa, wb, wm = 30 + 16 * len(a), 30 + 16 * len(b), 40 + 16 * len(a + b)
        fa, fb, fm = (self.frac(w)[0] for w in (wa, wb, wm))
        x = self.X0 + 0.01
        self.box(sc, x + fa / 2, y + 0.02, wa, 44, soft(ink), op, show(a), border=ink,
                 ink=dark(ink, 0.35), size=0.038)
        self.box(sc, x + fa + gap + fb / 2, y + 0.02, wb, 44, soft(ink), op, show(b), border=ink,
                 ink=dark(ink, 0.35), size=0.038)
        x += fa + fb + 0.065
        self.arrow(sc, (x, y + 0.02), (x + 0.06, y + 0.02), ink, op, 1.0, 2.6)
        self.box(sc, x + 0.08 + fm / 2, y + 0.02, wm, 44, ink, op * ramp(within, 0.5, 0.8),
                 show(a + b), ink=WHITE, size=0.038)
        self.text(sc, 0.95, 0.52, "vocabulary: %d symbols + %d merges" % (self.alphabet, k + 1),
                  op, anchor="rm", size=0.034)
        done = k + (1 if within > 0.6 else 0)
        seg = self.segments[done]
        self._segments(sc, seg, 0.60, op, ink_of=self._ink_of)
        self.text(sc, self.X0, 0.52, "the same phrase now: %d tokens" % len(seg), op, anchor="lm",
                  strong=True, size=0.034)
        # how far through the 96 merges, with the three vocabularies the study trains marked
        x0, x1, yb = 0.12, 0.90, 0.80
        full = self.span(x1 - x0)[0]
        self.hbar(sc, x0, yb, full * (k + within) / len(self.rules), ink, op, "", "", thick=14.0,
                  track=full)
        for m, mink, name in zip(self.MERGES[1:], self.INKS[1:4], self.names[1:4]):
            x = x0 + (x1 - x0) * m / len(self.rules)
            self.polyline(sc, [self.P(x, yb - 0.03), self.P(x, yb + 0.03)], 2.4, mink, op)
            self.text(sc, x, yb + 0.065, "%s: %d tokens" % (name, len(self.segments[m])),
                      op * (1.0 if done >= m else 0.4), ink=dark(mink), size=0.030,
                      anchor="mm" if m < 96 else "rm")

    def phase_3(self, sc, u, op):
        heads = ((0.05, "tokenizer"), (0.20, "the phrase"), (0.49, "tokens in the passage"),
                 (0.76, "embedding table"))
        for x, name in heads:
            self.text(sc, x, 0.315, name, op, anchor="lm", strong=True)
        bar = self.span(0.17)[0]
        tab = self.span(0.10)[0]
        segs = [list(self.phrase)] + [self.segments[m] for m in self.MERGES[1:]] + [list(self.phrase)]  # noqa: E501
        for row, (name, ink) in enumerate(zip(self.names, self.INKS)):
            y = 0.405 + 0.112 * row
            a = ramp(u, 0.04 + 0.12 * row, 0.14 + 0.12 * row)
            self.text(sc, 0.05, y, name, op * a, anchor="lm", ink=dark(ink), size=0.034)
            self._segments(sc, segs[row], y, op * a, h=26.0, pitch=0.0118, x0=0.20, text=False,
                           ink_of=lambda tok, ink=ink: ink)
            g = (u - 0.08 - 0.12 * row) / 0.2
            self.hbar(sc, 0.49, y, bar * self.tokens_n[row] / max(self.tokens_n), ink, op * a, "",
                      f"{self.tokens_n[row]:,}", g, thick=26.0, track=bar)
            rows = self.table[row] // self.width
            self.hbar(sc, 0.76, y, tab * self.table[row] / max(self.table), ink, op * a, "",
                      "%d rows" % rows, g, thick=26.0, track=tab)
        self.text(sc, 0.05, 0.925, "each table row holds %d numbers; a model must learn every "
                  "row from the tokens it sees" % self.width, op * ramp(u, 0.7, 0.85), anchor="lm")

    def phase_4(self, sc, u, op):
        f = min(u / 0.5, 1.0)
        chart = self.chart(((0.10, 0.52), (0.36, 0.80)), self.steps, self.curves)
        self.axes(sc, chart, op, "training update", "training loss per token (nats)")
        for row in (4, 0, 1, 2, 3):
            self.plot(sc, chart, self.steps, self.curves[row], self.INKS[row], op, f, width=1.5,
                      dot=row != 4)
        self.text(sc, 0.10, 0.925, "per token the bigger vocabularies look worse: each of their "
                  "tokens is more text to guess", op * ramp(u, 0.3, 0.45), anchor="lm")
        full = self.span(0.20)[0]
        b = ramp(u, 0.5, 0.6)
        self.text(sc, 0.60, 0.33, "held-out loss per character", op * b, anchor="lm", strong=True)
        for row, (name, ink) in enumerate(zip(self.names, self.INKS)):
            y = 0.41 + 0.092 * row
            self.hbar(sc, 0.70, y, full * self.per_char[row] / 3.0, ink, op * b, name,
                      "%.3f" % self.per_char[row] + ("  lowest" if row == self.best else ""),
                      (u - 0.5 - 0.05 * row) / 0.25, thick=28.0, track=full)


ANIMATIONS = {
    "teacher-forcing": TeacherForcing,
    "perplexity-explained": PerplexityExplained,
    "packing-and-padding": PackingAndPadding,
    "warmup-and-clipping": WarmupAndClipping,
    "grokking": Grokking,
    "scaling-laws": ScalingLaws,
    "curriculum-and-data-mixing": CurriculumAndDataMixing,
    "tokenizer-effects": TokenizerEffects,
}

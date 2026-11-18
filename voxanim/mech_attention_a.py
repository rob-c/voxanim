"""Attention, how it is wired: animations 1-6 of the "next 100".

Each draws the mechanism first and the measurement after it; the numbers are
the study's (:mod:`voxanim.next_attention`, and :mod:`voxanim.study_attention_a` for
the two experiments that were redesigned: NumPy transformers trained on Tiny
Shakespeare, or the pinned GPT-2).
"""

from __future__ import annotations

import numpy as np

from . import next_attention as NA
from . import realdata, theme
from . import study_attention_a as SA
from . import tinylm as T
from .cache import cached_npz
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIGIT_INK,
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
    heat,
    nice_ticks,
    number,
    soft,
)

INK_BOX = (0.16, 0.17, 0.19)        # a hidden slot, or the token whose turn it is


def show(ch):
    """A character as it is drawn: a space as an open box, a newline as a hook."""
    return "␣" if ch == " " else "↵" if ch == "\n" else ch


def configs_of(study):
    """The model configurations a study trained, in its order."""
    return [T.Config(**run["settings"]["config"]) for run in study.metadata["runs"]]


def commas(v):
    return f"{int(round(float(v))):,}"


def window(n=8, seed=73):
    """``n`` characters of held-out Tiny Shakespeare, as they are shown on
    screen (a space as an open box)."""
    ids = realdata.text_sequences(n=1, length=n, split="test", seed=seed)[0]
    return [("␣" if ch == " " else "↵" if ch == "\n" else ch) for ch in realdata.decode(ids)]


class CausalVsBidirectional(Mechanism):
    """The causal mask against the full one: who may read whom, drawn as arcs
    over the same eight characters and as the two mask matrices, then one
    model trained each way and both measured on held-out text."""

    TITLE = "Causal and bidirectional attention"
    PHASES = (("causal", 13.0), ("bidirectional", 11.0), ("training", 13.0), ("held out", 11.0))
    N = 8
    ROW = (0.05, 0.62, 0.64)            # the token row: x0, x1, y
    TOKEN = dict(w=60.0, h=46.0, size=0.040)  # noqa: RUF012
    MASK_AT = (0.81, 0.60, 236.0)       # the mask matrix: x, y, side in px

    def setup(self):
        s = self.study
        self.chars = window(self.N)
        self.causal = np.tril(np.ones((self.N, self.N)))
        self.full = np.ones((self.N, self.N))
        self.steps = np.asarray(s.panels[1].data["x"], float)
        self.traces = np.asarray(s.panels[1].data["ys"], float)
        self.heldout = [float(v) for v in s.metadata["heldout_loss"]]
        self.updates = int(s.metadata["steps"])

    def subtitle_for(self, k):
        lo, hi = self.heldout
        return [
            "causal attention: each character reads itself and the ones before it, never the ones "
            "after; the mask is the lower triangle, %d of %d pairs" % (
                int(self.causal.sum()), self.N * self.N),
            "bidirectional attention: every character reads every other, in both directions; "
            "all %d pairs are allowed" % (self.N * self.N),
            "one model of each kind, %d updates on the same Tiny Shakespeare windows, each asked "
            "to restore one hidden character" % self.updates,
            "on held-out text the causal model's loss is %.2f nats and the bidirectional one's "
            "%.2f: the hidden character's right-hand neighbours were worth %.2f" % (
                lo, hi, lo - hi),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _reading(self, sc, u, op, mask, ink, both):
        """Queries taking their turn left to right: arcs from every character
        the mask lets them read, and the matching row of the mask lighting up."""
        x0, x1, y = self.ROW
        turn = min(int(u * 1.08 * self.N), self.N - 1)        # the query whose turn it is
        within = min(u * 1.08 * self.N - turn, 1.0)
        blocked = [j for j in range(self.N) if not mask[turn, j]]
        plain = ["" if j == turn or j in blocked else ch for j, ch in enumerate(self.chars)]
        xs = self.tokens(sc, plain, x0, x1, y, op, **self.TOKEN)
        self.box(sc, xs[turn], y, 60, 46, ink, op, self.chars[turn], ink=(1, 1, 1), size=0.040,
                 glow=0.15)
        for j in range(self.N):
            if j == turn or not mask[turn, j]:
                continue
            reach = abs(turn - j)
            bend = (0.07 + 0.034 * reach) * (1 if j < turn or not both else -1)
            a = (xs[j], y - 0.055 if bend > 0 else y + 0.055)
            b = (xs[turn], y - 0.055 if bend > 0 else y + 0.055)
            self.arc(sc, a, b, ink, op, smoothstep(0.0, 0.5, within), bend, 2.4)
            self.pulse(sc, a, b, (within - 0.35) / 0.6, ink, op, bend)
        for j in blocked:
            self.box(sc, xs[j], y, 60, 46, DIM, op * 0.85, self.chars[j], size=0.040)
        self.text(sc, x0 + 0.005, 0.90, "%s is reading %d of the %d characters" % (
            self.chars[turn], int(mask[turn].sum()), self.N), op, anchor="lm", strong=True,
            size=0.034)
        mx, my, side = self.MASK_AT
        cell = self.grid(sc, mask, mx, my, side, side, op, reveal=(turn + 1) / self.N,
                         ring=[(turn, turn)], hot=ink, mask=True)
        self.text(sc, mx, my - 0.29, "who may read whom", op, strong=True, size=0.034)
        self.text(sc, mx, my + 0.285, "row: the reader   column: the one read", op)
        self.text(sc, cell(turn, 0)[0] - 0.035, cell(turn, 0)[1], self.chars[turn], op,
                  ink=dark(ink), size=0.034)

    def phase_1(self, sc, u, op):
        self._reading(sc, u, op, self.causal, RED, both=False)

    def phase_2(self, sc, u, op):
        self._reading(sc, u, op, self.full, BLUE, both=True)

    def _task(self, sc, op, y=0.34):
        """The task both models train on: one character hidden."""
        x0, x1, _ = self.ROW
        hidden = self.N // 2
        shown = list(self.chars)
        shown[hidden] = ""
        xs = self.tokens(sc, shown, x0, x1, y, op, w=52.0, h=38.0, size=0.036)
        self.box(sc, xs[hidden], y, 52, 38, (0.16, 0.17, 0.19), op, "?", ink=(1, 1, 1), size=0.036)
        self.text(sc, x1 + 0.03, y, "the task: restore the hidden character", op, anchor="lm",
                  strong=True)
        # what each kind of model may use to do it
        for row, (name, ink, left_only) in enumerate((("causal", RED, True),
                                                      ("bidirectional", BLUE, False))):
            yy = y + 0.07 + row * 0.04
            self.text(sc, x1 + 0.03, yy, "what the %s model reads" % name, op, anchor="lm",
                      ink=dark(ink))
            for j, x in enumerate(xs):
                if j != hidden and (j < hidden or not left_only):
                    self.box(sc, x, yy, 44, 12, ink, op, radius=3)
        return xs, hidden

    def phase_3(self, sc, u, op):
        self._task(sc, op)
        chart = self.chart(((0.10, 0.62), (0.53, 0.84)), self.steps, self.traces)
        self.axes(sc, chart, op, "training update", "loss (nats)")
        for trace, ink in zip(self.traces, (RED, BLUE)):
            self.plot(sc, chart, self.steps, trace, ink, op, u / 0.85)
        now = int(min(u / 0.85, 1.0) * (len(self.steps) - 1))
        self.lines(sc, 0.72, 0.62, [("update %d" % self.steps[now], None),
                                    ("causal  %.2f" % self.traces[0][now], dark(RED)),
                                    ("bidirectional  %.2f" % self.traces[1][now], dark(BLUE))],
                   op, step=0.06, size=0.036)

    def phase_4(self, sc, u, op):
        self._task(sc, op)
        worst = max(self.heldout)
        full = self.span(0.42)[0]
        for row, (name, ink, value) in enumerate(zip(("causal", "bidirectional"), (RED, BLUE),
                                                     self.heldout)):
            self.hbar(sc, 0.22, 0.63 + row * 0.10, full * value / worst, ink, op, name,
                      "%.2f nats" % value, u / 0.5, thick=34.0, track=full)
        self.text(sc, 0.22, 0.55, "loss on held-out text (lower is better)", op, anchor="lm",
                  strong=True)
        gain = self.heldout[0] - self.heldout[1]
        self.lines(sc, 0.22, 0.85, [
            "reading both sides: %s nats %s" % (number(abs(gain)), "better" if gain > 0 else "worse"),  # noqa: E501
            "a next-character model cannot train this way: it would see its answer"],
            op * smoothstep(0.5, 0.75, u), size=0.032, step=0.046)


class PrefixLM(Mechanism):
    """The prefix mask, between the causal and the full one: the prompt reads
    itself both ways, the reply only leftwards. Then the three masks side by
    side, a trained model of each writing a reply to the same held-out prompt,
    and how many reply characters each gets right."""

    TITLE = "The prefix mask"
    PHASES = (("prefix mask", 14.0), ("three masks", 10.0), ("writing the reply", 13.0),
              ("training", 9.0), ("held out", 10.0))
    NAMES = SA.MASKS
    INK = (RED, VIOLET, BLUE)
    ROW = (0.04, 0.62, 0.66)
    MASK_AT = (0.81, 0.60, 240.0)
    START = "→"                         # the start-of-reply mark, token 0 in the model
    EXAMPLE = 0                         # the first held-out prompt the study scores

    def setup(self):
        m = self.study.metadata
        self.K = K = int(m["prompt"])
        self.N = 2 * K
        self.configs = [T.Config(**c) for c in m["configs"]]
        self.masks = [T.attention_mask(self.N, c).astype(float) for c in self.configs]
        self.pairs = [int(mask.sum()) for mask in self.masks]
        runs = SA.prefix_runs()
        prompts, _x, y, _w = SA.prefix_heldout()
        self.tested = len(prompts)
        letters = lambda ids: [show(c) for c in realdata.decode(ids)]  # noqa: E731
        self.prompt = letters(prompts[self.EXAMPLE])
        self.target = letters(y[self.EXAMPLE, K:])
        # what each mask's first model (seed 0) writes for that prompt
        self.written = [letters(runs[name][0]["written"][self.EXAMPLE]) for name in self.NAMES]
        # the row a model reads in training: the prompt, the start mark, the reply so far
        self.chars = self.prompt + [self.START] + self.target[:-1]
        self.updates = np.asarray(m["updates"], float)
        self.curves = np.asarray(m["written_accuracy_curves"], float)   # (mask, seed, check)
        self.accuracy = np.asarray(m["written_accuracy"], float)        # (mask, seed)
        self.loss = np.asarray(m["heldout_loss"], float)
        self.steps = int(m["steps"])
        self.seeds = self.accuracy.shape[1]
        causal, prefix = self.accuracy[0], self.accuracy[1]
        self.edge = round(100 * float(prefix.mean()), 1) - round(100 * float(causal.mean()), 1)
        self.clear = bool(prefix.min() > causal.max() or causal.min() > prefix.max())

    def verdict(self):
        """What the causal and prefix runs support, in words."""
        if not self.clear:
            return "the seeds overlap: no measurable gain from the prefix mask"
        return "every %s seed is ahead" % ("prefix" if self.edge > 0 else "causal")

    def subtitle_for(self, k):
        c, p, b = (100 * self.accuracy.mean(1)).tolist()
        n, K = self.N, self.K
        return [
            "the prefix mask: the %d prompt characters read each other in both directions; "
            "the %d reply slots read only what is to their left" % (K, n - K),
            "it sits between the other two: the causal mask allows %d of the %d pairs, the "
            "prefix mask %d, the bidirectional mask all %d" % (
                self.pairs[0], n * n, self.pairs[1], self.pairs[2]),
            "the task: write the prompt's characters in order. one trained model per mask "
            "writes its reply to a held-out prompt, each character it writes fed back to it",
            "%d two-block models per mask, %d updates each on fresh Shakespeare windows. "
            "measured as they train: the share of held-out reply characters written right" % (
                self.seeds, self.steps),
            "replies to %d held-out prompts: causal %.1f%% of characters right, prefix %.1f%%, "
            "bidirectional %.1f%%. %s. a reply trained reading ahead cannot be written" % (
                self.tested, c, p, b, "causal and prefix: the seeds overlap" if not self.clear
                else self.verdict()),
        ][k]

    def phase_1(self, sc, u, op):
        mask, ink, N, K = self.masks[1], VIOLET, self.N, self.K
        x0, x1, y = self.ROW
        box = dict(w=30.0, h=44.0, size=0.034)
        turn = min(int(u * 1.06 * N), N - 1)
        within = min(u * 1.06 * N - turn, 1.0)
        blocked = [j for j in range(N) if not mask[turn, j]]
        plain = ["" if j == turn or j in blocked else ch for j, ch in enumerate(self.chars)]
        xs = self.tokens(sc, plain, x0, x1, y, op, **box)
        self.box(sc, xs[turn], y, 30, 44, ink, op, self.chars[turn], ink=WHITE, size=0.034,
                 glow=0.15)
        for j in range(N):
            if j == turn or not mask[turn, j]:
                continue
            reach = abs(turn - j)
            up = j < turn
            bend = min(0.07 + 0.03 * reach, 0.30) * (1 if up else -1)
            a = (xs[j], y - 0.055 if up else y + 0.055)
            b = (xs[turn], y - 0.055 if up else y + 0.055)
            self.arc(sc, a, b, ink, op, smoothstep(0.0, 0.5, within), bend, 2.0)
            self.pulse(sc, a, b, (within - 0.35) / 0.6, ink, op, bend)
        for j in blocked:
            self.box(sc, xs[j], y, 30, 44, DIM, op * 0.85, self.chars[j], size=0.034)
        for name, a, b in (("the prompt", 0, K - 1), ("the reply", K, N - 1)):
            self.text(sc, (xs[a] + xs[b]) / 2, 0.325, name, op, strong=True, size=0.034)
            self.polyline(sc, [self.P(xs[a] - 0.014, 0.362), self.P(xs[b] + 0.014, 0.362)], 2.0,
                          AXIS, op)
        where = ("the whole prompt, both ways" if turn < K else "everything up to itself")
        self.text(sc, x0 + 0.005, 0.90, "%s reads %d of the %d: %s" % (
            self.chars[turn], int(mask[turn].sum()), N, where), op, anchor="lm", strong=True,
            size=0.034)
        mx, my, side = self.MASK_AT
        cell = self.grid(sc, mask, mx, my, side, side, op, reveal=(turn + 1) / N,
                         ring=[(turn, turn)], hot=ink, mask=True)
        self.text(sc, mx, my - 0.29, "who may read whom", op, strong=True, size=0.034)
        self.text(sc, mx, my + 0.285, "row: the reader   column: the one read", op)
        self.text(sc, cell(turn, 0)[0] - 0.03, cell(turn, 0)[1], self.chars[turn], op,
                  ink=dark(ink), size=0.034)

    def phase_2(self, sc, u, op):
        causal, prefix, full = self.masks
        N, K, side = self.N, self.K, 200.0
        g1, g2 = smoothstep(0.30, 0.48, u), smoothstep(0.60, 0.78, u)
        mid = causal + g1 * (prefix - causal)
        shown = [causal, mid, mid + g2 * (full - prefix)]
        notes = ["the lower triangle",
                 "+%d: the prompt reads rightwards" % (self.pairs[1] - self.pairs[0]),
                 "+%d: the reply reads ahead too" % (self.pairs[2] - self.pairs[1])]
        more = (0, (self.pairs[1] - self.pairs[0]) * (g1 > 0.5),
                (self.pairs[2] - self.pairs[1]) * (g2 > 0.5))
        count = [self.pairs[0] + sum(more[:i + 1]) for i in range(3)]
        for i, cx in enumerate((0.20, 0.50, 0.80)):
            ink = self.INK[i]
            self.text(sc, cx, 0.325, self.NAMES[i], op, ink=dark(ink), size=0.038)
            self.text(sc, cx, 0.375, "%d of %d pairs" % (count[i], N * N), op,
                      strong=True)
            cell = self.grid(sc, shown[i], cx, 0.62, side, side, op, reveal=u / 0.2, vmax=1.0,
                             hot=ink, mask=True)
            if i >= 1 and g1 > 0:
                bx, by = cell((K - 1) / 2, (K - 1) / 2)
                self.rect(sc, self.P(bx, by), side * K / N + 3, side * K / N + 3, AMBER, op * g1,
                          border=AMBER, border_w=2.4, radius=2, lift=0.7, fill_op=0.0)
            self.text(sc, cx, 0.86, notes[i], op * (1.0, g1, g2)[i])
        for x, g in ((0.35, g1), (0.65, g2)):
            self.arrow(sc, (x - 0.025, 0.62), (x + 0.025, 0.62), AXIS, op * g, 1.0, 2.6)

    def _task(self, sc, op, y=0.34):
        """The task every model trains on: a held-out prompt and its reply."""
        K = self.K
        xs = self.tokens(sc, self.prompt, 0.05, 0.33, y, op, w=30.0, h=36.0, size=0.034)
        self.arrow(sc, (xs[-1] + 0.03, y), (xs[-1] + 0.07, y), AXIS, op, 1.0, 2.6)
        self.tokens(sc, self.target, 0.43, 0.71, y, op, inks=[GREEN], w=30.0, h=36.0, size=0.034)
        self.text(sc, 0.73, y, "the reply: the prompt's", op, anchor="lm", strong=True)
        self.text(sc, 0.73, y + 0.042, "%d characters in order" % K, op, anchor="lm", strong=True)

    def phase_3(self, sc, u, op):
        K = self.K
        self._task(sc, op, 0.33)
        done = u * 1.5 * K                              # reply characters written so far
        x0, x1 = 0.43, 0.71
        for row, (name, ink, wrote) in enumerate(zip(self.NAMES, self.INK, self.written)):
            y = 0.50 + row * 0.115
            self.text(sc, 0.40, y, name + " writes", op, anchor="rm", ink=dark(ink), size=0.036)
            n = int(min(done, K))
            right = [a == b for a, b in zip(wrote, self.target)]
            xs = self.tokens(sc, [""] * K, x0, x1, y, op, w=30.0, h=40.0)
            for j in range(n):
                ok = right[j]
                self.box(sc, xs[j], y, 30, 40, soft(ink) if ok else INK_BOX, op, wrote[j],
                         ink=dark(ink, 0.35) if ok else WHITE, border=ink, size=0.036)
            if n < K and done > 0:
                self.box(sc, xs[n], y, 30, 40, ink, op * smoothstep(0.0, 0.6, done - n), "",
                         glow=0.2)
            if n == K:
                self.text(sc, 0.73, y, "%d of %d right" % (sum(right), K), op, anchor="lm",
                          strong=True, size=0.034)
        late = op * smoothstep(0.70, 0.82, u)
        self.lines(sc, 0.05, 0.875, [
            "In training the bidirectional reply could read the slot ahead of it: its own answer.",
            "Writing alone, that slot is still empty. (A %s box is a wrong character.)"
            % theme.word("dark", "pale")],
            late, step=0.044)

    def phase_4(self, sc, u, op):
        self._task(sc, op)
        chart = self.chart(((0.10, 0.60), (0.50, 0.84)), self.updates, [0.0, 1.0], pad=0.0)
        self.axes(sc, chart, op, "training update", "characters right",
                  yticks=[(v, "%d%%" % (100 * v)) for v in (0, 0.25, 0.5, 0.75, 1.0)])
        go = u / 0.85
        mean = self.curves.mean(1)
        for i, ink in enumerate(self.INK):
            for trace in self.curves[i]:
                self.plot(sc, chart, self.updates, trace, ink, op * 0.45, go, width=1.0, dot=False)
            self.plot(sc, chart, self.updates, mean[i], ink, op, go, width=2.6)
        now = int(min(go, 1.0) * (len(self.updates) - 1))
        self.lines(sc, 0.66, 0.56, [
            ("update %d" % self.updates[now], None),
            ("causal  %.1f%%" % (100 * mean[0][now]), dark(RED)),
            ("prefix  %.1f%%" % (100 * mean[1][now]), dark(VIOLET)),
            ("bidirectional  %.1f%%" % (100 * mean[2][now]), dark(BLUE))], op, step=0.06,
            size=0.036)
        self.text(sc, 0.66, 0.84, "thin lines: each of %d seeds" % self.seeds, op, anchor="lm")

    def phase_5(self, sc, u, op):
        self._task(sc, op)
        full = self.span(0.36)[0]
        self.text(sc, 0.22, 0.455, "reply characters right, %d held-out prompts, written alone" %
                  self.tested, op, anchor="lm", strong=True)
        for row, (name, ink, acc) in enumerate(zip(self.NAMES, self.INK, self.accuracy)):
            self.hbar(sc, 0.22, 0.535 + row * 0.085, full * acc.mean(), ink, op, name,
                      "%.1f%%  (seeds %.1f to %.1f)" % (
                          100 * acc.mean(), 100 * acc.min(), 100 * acc.max()),
                      u / 0.5, thick=30.0, track=full)
        self.lines(sc, 0.10, 0.80, [
            "prefix against causal: %+.1f points; %s" % (self.edge, self.verdict()),
            "bidirectional: loss %.3f nats when shown the true reply (causal %.3f): there it "
            "reads its own answer" % (self.loss[2].mean(), self.loss[0].mean())],
            op * smoothstep(0.5, 0.75, u), size=0.030, step=0.048, width=0.80)


def cross_attention_runs(study, checkpoints=()):
    """The study's trained encoder-decoder (from its cache) and, for each count
    in ``checkpoints``, the same run stopped after that many updates: the
    seed and minibatch order are the same, so each is that run on its way."""
    cfg = configs_of(study)[0]
    source, x, target = NA.denoising_data(192, "train", 11)
    test = NA.denoising_data(64, "test", 73)
    provenance = {**NA.task_provenance("denoise", train_seed=11),
                  "evaluation_arrays": [T.array_identity(a) for a in test]}
    steps = len(study.panels[2].data["x"])

    def run(n):
        name = "realtext-cross-attention" if n == steps else "mech-cross-attention-%d" % n
        return T.train(cfg, x, target, context=source, steps=n, name=name,
                       provenance=provenance)["P"]
    return cfg, test, run(steps), {n: run(n) for n in checkpoints}


class CrossAttention(Mechanism):
    """A decoder reading an encoder's row: the two kinds of attention, then one
    decoder position at a time sending its query up and getting values back in
    the measured proportions, the link being trained, and what it restores."""

    TITLE = "Cross-attention"
    PHASES = (("two rows", 10.0), ("one query", 17.0), ("training", 10.0), ("held out", 11.0))
    EXAMPLE = 2                         # the held-out span drawn
    CHECKPOINTS = (0, 20, 60, 120, 200)
    XS = (0.30, 0.66)
    ENC, DEC, OUT = 0.41, 0.70, 0.865
    TOKEN = dict(w=50.0, h=44.0, size=0.038)  # noqa: RUF012

    def setup(self):
        s = self.study
        e = self.EXAMPLE
        cfg, (source, x, target), P, stages = cross_attention_runs(s, self.CHECKPOINTS)
        logits, rec = T.forward(P, x, cfg, context=source, keep=True)
        pred = logits.argmax(-1)
        self.model_attention = rec["attention"][1]          # (span, head, decoder, encoder)
        self.attn = self.model_attention[e, 0]
        self.blank = source[e] == 0
        self.target = [show(c) for c in realdata.decode(target[e])]
        self.written = [show(c) for c in realdata.decode(pred[e])]
        self.right = pred[e] == target[e]
        self.N = len(self.target)
        hit, hidden = pred == target, source == 0
        self.accuracy = float(s.metadata["accuracy"])
        self.chance = float(s.panels[3].data["values"][1])
        self.seen_accuracy = float(hit[~hidden].mean())
        self.blank_accuracy = float(hit[hidden].mean())
        self.held = len(x)
        self.loss = np.asarray(s.panels[2].data["ys"], float)[0]
        self.steps = np.asarray(s.panels[2].data["x"], float)
        self.stages = [(n, T.forward(Q, x[e:e + 1], cfg, context=source[e:e + 1], keep=True)[1][
            "attention"][1][0, 0]) for n, Q in stages.items()] + [(len(self.steps), self.attn)]

    def subtitle_for(self, k):
        top = int(self.attn[2].argmax())
        return [
            "a decoder has two kinds of attention. self-attention reads its own row, leftwards; "
            "cross-attention reads every character of the row the encoder was given",
            "each decoder position sends a query to the encoder's %d keys; how well each key "
            "matches decides how much of that value comes back. here position 3 takes %.0f%% "
            "from encoder position %d" % (self.N, 100 * self.attn[2, top], top + 1),
            "%d updates on Shakespeare spans with two characters blanked: the loss falls from "
            "%.2f to %.2f nats, and the attention moves away from an even spread" % (
                len(self.steps), self.loss[0], self.loss[-1]),
            "on %d held-out spans it writes %.1f%% of the characters correctly (guessing: %.1f%%): "
            "%.0f%% of those it could see in the encoder's row, %.0f%% of the blanked ones" % (
                self.held, 100 * self.accuracy, 100 * self.chance, 100 * self.seen_accuracy,
                100 * self.blank_accuracy),
        ][k]

    def _rows(self, sc, op, query=None, written=0):
        """The encoder's row, the decoder's, and what the decoder has written."""
        x0, x1 = self.XS
        enc = ["" if b else ch for ch, b in zip(self.target, self.blank)]
        xs = self.tokens(sc, enc, x0, x1, self.ENC, op, **self.TOKEN)
        for j in np.flatnonzero(self.blank):
            self.box(sc, xs[j], self.ENC, 50, 44, INK_BOX, op, "?", ink=WHITE, size=0.038)
        dec = [">"] + self.target[:-1]
        self.tokens(sc, ["" if j == query else ch for j, ch in enumerate(dec)], x0, x1, self.DEC,
                    op, **self.TOKEN)
        if query is not None:
            self.box(sc, xs[query], self.DEC, 50, 44, INK_BOX, op, dec[query], ink=WHITE,
                     size=0.038, glow=0.15)
        for j in range(self.N):
            if j < written:
                ink = GREEN if self.right[j] else RED
                self.box(sc, xs[j], self.OUT, 50, 40, soft(ink), op, self.written[j],
                         ink=dark(ink, 0.35), border=ink, size=0.038)
            else:
                self.box(sc, xs[j], self.OUT, 50, 40, PAPER, op * 0.4, border=AXIS)
        for y, a, b in ((self.ENC, "the encoder reads", "the damaged text"),
                        (self.DEC, "the decoder reads", "the text so far"),
                        (self.OUT, "and writes", "the next character")):
            self.text(sc, 0.05, y - 0.021, a, op, anchor="lm", strong=True)
            self.text(sc, 0.05, y + 0.021, b, op, anchor="lm")
        return xs

    def phase_1(self, sc, u, op):
        q = 3
        xs = self._rows(sc, op, query=q)
        top, foot = self.DEC - 0.052, self.ENC + 0.052
        for j in range(q):
            a, b = (xs[j], top), (xs[q], top)
            bend = 0.05 + 0.03 * (q - j)
            self.arc(sc, a, b, TEAL, op, smoothstep(0.05, 0.25, u), bend, 2.6)
            self.pulse(sc, a, b, (u - 0.18) / 0.22, TEAL, op, bend)
        cross = smoothstep(0.50, 0.70, u)
        for j in range(self.N):
            a, b = (xs[j], foot), (xs[q], top)
            self.arrow(sc, a, b, AMBER, op, cross, 2.6)
            self.pulse(sc, a, b, (u - 0.66) / 0.24, AMBER, op)
        x = 0.71
        self.text(sc, x, 0.585, "self-attention", op, anchor="lm", ink=dark(TEAL), size=0.036)
        self.lines(sc, x, 0.635, ["its own row, only the characters before it"], op, width=0.24)
        show_cross = op * smoothstep(0.45, 0.6, u)
        self.text(sc, x, 0.35, "cross-attention", show_cross, anchor="lm", ink=dark(AMBER, 0.3),
                  size=0.036)
        self.lines(sc, x, 0.40, ["the other row: all %d characters, wherever they are" % self.N],
                   show_cross, width=0.24)

    def phase_2(self, sc, u, op):
        N = self.N
        turn = min(int(u * 1.04 * N), N - 1)
        within = min(u * 1.04 * N - turn, 1.0)
        xs = self._rows(sc, op, query=turn, written=turn + (1 if within > 0.78 else 0))
        top, foot = self.DEC - 0.052, self.ENC + 0.052
        w = self.attn[turn]
        for j in range(N):
            dn, up = (xs[turn], top), (xs[j], foot)
            self.arc(sc, dn, up, soft(AMBER, 0.3), op, smoothstep(0.0, 0.22, within), 0.0, 1.2,
                     head=False)
            self.pulse(sc, dn, up, within / 0.26, INK_BOX, op, r=3.0)
            self.arrow(sc, up, dn, AMBER, op, smoothstep(0.30, 0.52, within), 1.0 + 9.0 * w[j])
            self.pulse(sc, up, dn, (within - 0.40) / 0.36, AMBER, op, r=2.5 + 7.0 * w[j])
            self.text(sc, xs[j], self.ENC - 0.078, "%.2f" % w[j],
                      op * smoothstep(0.30, 0.45, within), ink=dark(AMBER, 0.35), size=0.032)
        if within > 0.78:
            self.arrow(sc, (xs[turn], self.DEC + 0.05), (xs[turn], self.OUT - 0.045), INK_BOX, op,
                       1.0, 2.0)
        mx, my, side = 0.84, 0.57, 196.0
        cell = self.grid(sc, self.attn, mx, my, side, side, op, reveal=(turn + 1) / N,
                         ring=[(turn, int(w.argmax()))], ring_ink=DIGIT_INK, hot=AMBER)
        self.text(sc, mx, my - 0.235, "how much of each", op, strong=True, size=0.034)
        self.text(sc, mx, my + 0.225, "row: decoder position", op)
        self.text(sc, mx, my + 0.265, "column: encoder position", op)
        self.text(sc, cell(turn, 0)[0] - 0.042, cell(turn, 0)[1], "%d" % (turn + 1), op,
                  ink=dark(AMBER, 0.35), size=0.032)

    def phase_3(self, sc, u, op):
        go = u / 0.85
        chart = self.chart(((0.09, 0.55), (0.45, 0.80)), self.steps, self.loss)
        self.axes(sc, chart, op, "training update", "loss (nats)")
        self.plot(sc, chart, self.steps, self.loss, AMBER, op, go, width=2.2)
        now = int(min(go, 1.0) * (len(self.steps) - 1))
        done = now + 1 if go >= 1.0 else now
        self.text(sc, 0.09, 0.335, "update %d: loss %.2f nats" % (now, self.loss[now]), op,
                  anchor="lm", strong=True, size=0.036)
        count, attn = [st for st in self.stages if st[0] <= done][-1]
        mx, my, side = 0.79, 0.60, 216.0
        self.grid(sc, attn, mx, my, side, side, op, hot=AMBER, vmax=float(self.attn.max()))
        self.text(sc, mx, my - 0.255, "its attention after %d updates" % count, op, strong=True,
                  size=0.034)
        self.text(sc, mx, my + 0.245, "row: decoder position", op)
        self.text(sc, mx, my + 0.285, "column: encoder position", op)

    def phase_4(self, sc, u, op):
        x0, x1 = 0.17, 0.44
        rows = (("damaged", 0.37), ("original", 0.48), ("written", 0.59))
        enc = ["" if b else ch for ch, b in zip(self.target, self.blank)]
        xs = self.tokens(sc, enc, x0, x1, rows[0][1], op, w=38.0, h=36.0, size=0.036)
        for j in np.flatnonzero(self.blank):
            self.box(sc, xs[j], rows[0][1], 38, 36, INK_BOX, op, "?", ink=WHITE, size=0.036)
        self.tokens(sc, self.target, x0, x1, rows[1][1], op, w=38.0, h=36.0, size=0.036)
        for j in range(self.N):
            ink = GREEN if self.right[j] else RED
            self.box(sc, xs[j], rows[2][1], 38, 36, soft(ink), op * smoothstep(
                j * 0.05, j * 0.05 + 0.08, u), self.written[j], ink=dark(ink, 0.35), border=ink,
                size=0.036)
        for name, y in rows:
            self.text(sc, 0.05, y, name, op, anchor="lm")
        self.lines(sc, 0.05, 0.70, ["one of the %d held-out spans:" % self.held,
                                    ("green: written correctly", dark(GREEN)),
                                    ("red: not", dark(RED))], op, step=0.045)
        full = self.span(0.19)[0]
        fx = 0.70
        self.text(sc, fx, 0.335, "characters right", op, anchor="lm", strong=True, size=0.034)
        bars = (("every position", self.accuracy, AMBER),
                ("seen by the encoder", self.seen_accuracy, GREEN),
                ("blanked for it", self.blank_accuracy, VIOLET),
                ("guessing evenly", self.chance, DIM))
        for row, (name, value, ink) in enumerate(bars):
            self.hbar(sc, fx, 0.42 + row * 0.095, full * value, ink, op, name,
                      "%.1f%%" % (100 * value), (u - 0.1 * row) / 0.4, thick=30.0, track=full)
        self.lines(sc, 0.50, 0.83, [
            "teacher-forced: it is always shown the true text so far",
            "restoring damaged text; not translation, not free writing"],
            op * smoothstep(0.55, 0.75, u), step=0.045)


class MultiHeadVsSingle(Mechanism):
    """The same 64 numbers as one head or as eight: where the one head of a
    trained model looks, where each of the eight looks, how far back each of
    them reads on held-out text, and what both models learned."""

    TITLE = "One head or eight"
    PHASES = (("one head", 11.0), ("eight heads", 14.0), ("where they look", 12.0),
              ("training", 9.0), ("held out", 10.0))
    ROW = (0.04, 0.96, 0.34)
    TOKEN = dict(w=46.0, h=44.0, size=0.036)  # noqa: RUF012
    ONE, EIGHT = RED, BLUE
    COLUMNS = ("itself", "1 back", "2 back", "3 back", "4 back", "further")

    def setup(self):
        m = self.study.metadata
        runs = SA.head_runs()
        one, eight = runs[SA.HEAD_COUNTS[0]][0], runs[SA.HEAD_COUNTS[1]][0]     # seed 0 of each
        self.width, self.heads = one["config"].width, eight["config"].heads
        xt, _yt = SA.head_heldout()
        self.tested = len(xt)
        self.chars = [show(c) for c in realdata.decode(xt[0])]
        self.N = len(self.chars)
        self.one = one["attention"][0, 0]           # the drawn window: (reader, read)
        self.eight = eight["attention"][0]          # (head, reader, read)
        self.offsets = np.vstack([one["offsets"], eight["offsets"]])    # one head, then eight
        self.even = SA.even_spread()
        self.updates = np.asarray(m["updates"], float)
        self.curves = np.asarray(m["heldout_loss_curves"], float)       # (model, seed, check)
        self.loss = np.asarray(m["heldout_loss"], float)                # (model, seed)
        self.seeds = self.loss.shape[1]
        self.steps = int(m["steps"])
        self.parameters = [int(v) for v in m["parameters"]]
        self.guess = float(m["guessing_evenly"])
        self.gain = float(self.loss[0].mean() - self.loss[1].mean())
        self.clear = bool(self.loss[1].max() < self.loss[0].min())
        # the heads' lowest and highest share on the character before, in each seed's model
        self.before = [(100 * min(v), 100 * max(v)) for v in m["previous_character_share"][1]]

    def subtitle_for(self, k):
        before = 100 * self.offsets[:, 1]
        return [
            "an attention head turns each character into a pattern of attention over the "
            "characters before it. one head, %d numbers wide, makes one pattern each" % self.width,
            "split the same %d numbers into %d heads of %d and every character gets %d patterns "
            "at once; in this trained model they point at different characters" % (
                self.width, self.heads, self.width // self.heads, self.heads),
            "over %d held-out windows the one head puts %.0f%% of its attention on the character "
            "before. the eight heads differ: from %.0f%% to %.0f%% there (other seeds: %s)" % (
                self.tested, before[0], before[1:].min(), before[1:].max(),
                ", ".join("%.0f%% to %.0f%%" % pair for pair in self.before[1:])),
            "both models have %s parameters in one block and train for %d updates on fresh "
            "windows of Shakespeare, predicting the next character; %d seeds each" % (
                commas(self.parameters[0]), self.steps, self.seeds),
            "held-out loss, mean of %d seeds: one head %.3f nats, eight heads %.3f. eight "
            "patterns at once are worth %.3f nats here%s" % (
                self.seeds, self.loss[0].mean(), self.loss[1].mean(), self.gain,
                ": every eight-head model is ahead" if self.clear else "; the seeds overlap"),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _reader(self, sc, op, q, weights, ink, u=1.0):
        """The window with character ``q`` reading: an arc from each earlier
        character, as thick as its share of the attention."""
        x0, x1, y = self.ROW
        tw, th, size = self.TOKEN["w"], self.TOKEN["h"], self.TOKEN["size"]
        xs = self.tokens(sc, ["" if j == q else ch for j, ch in enumerate(self.chars)], x0, x1, y,
                         op, active=range(q + 1), **self.TOKEN)
        self.box(sc, xs[q], y, tw, th, ink, op, self.chars[q], ink=WHITE, size=size, glow=0.15)
        for j in range(q):
            w = float(weights[j])
            if w < 0.03:
                continue
            bend = -min(0.06 + 0.014 * (q - j), 0.17)
            a, b = (xs[j], y + 0.055), (xs[q], y + 0.055)
            self.arc(sc, a, b, ink, op, smoothstep(0.0, 0.45, u), bend, 1.0 + 8.0 * w)
            self.pulse(sc, a, b, (u - 0.3) / 0.6, ink, op, bend, r=2.5 + 6.0 * w)
        return xs

    def _said(self, q, weights):
        top = int(np.argmax(weights[:q + 1]))
        return "%s puts %.0f%% of its attention on %s" % (
            self.chars[q], 100 * weights[top],
            "itself" if top == q else "the %s %d back" % (self.chars[top], q - top))

    def _strip(self, sc, x, y, groups, ink, op, cell=9.0, gap=0.0):
        """The model's ``width`` numbers as cells, cut into ``groups`` heads."""
        per = self.width // groups
        dx, dg = self.frac(cell)[0], self.frac(gap)[0]
        for i in range(self.width):
            self.box(sc, x + (i + 0.5) * dx + (i // per) * dg, y, cell - 2, 20, ink, op, radius=2)
        return x + self.width * dx + (groups - 1) * dg

    def phase_1(self, sc, u, op):
        N = self.N
        q = 1 + min(int(u * 1.05 * (N - 1)), N - 2)
        within = min(u * 1.05 * (N - 1) - (q - 1), 1.0)
        self._reader(sc, op, q, self.one[q], self.ONE, within)
        mx, my, side = 0.20, 0.735, 184.0
        cell = self.grid(sc, self.one, mx, my, side, side, op, reveal=(q + 1) / N,
                         ring=[(q, int(np.argmax(self.one[q])))], ring_ink=DIGIT_INK,
                         hot=self.ONE, vmax=1.0)
        self.text(sc, cell(q, 0)[0] - 0.03, cell(q, 0)[1], self.chars[q], op,
                  ink=dark(self.ONE), size=0.034)
        x = 0.40
        self.text(sc, x, 0.585, "one head: all %d numbers work as one" % self.width, op,
                  anchor="lm", strong=True, size=0.036)
        self._strip(sc, x, 0.655, 1, self.ONE, op, cell=8.0)
        self.lines(sc, x, 0.735, [
            (self._said(q, self.one[q]), dark(self.ONE)),
            "the map: a row for each reader, a column for each character read",
            "a deeper cell: more of that reader's attention"], op, step=0.05, size=0.031)

    def phase_2(self, sc, u, op):
        H, N = self.heads, self.N
        head = min(int(u * 1.04 * H), H - 1)
        within = min(u * 1.04 * H - head, 1.0)
        q = N - 1
        self._reader(sc, op, q, self.eight[head, q], self.EIGHT, within)
        x0, x1, _ = self.ROW
        pitch = (x1 - x0) / H
        side = 88.0
        for h in range(H):
            cx = x0 + (h + 0.5) * pitch
            on = h == head
            self.text(sc, cx, 0.572, "head %d" % (h + 1), op, strong=on)
            self.grid(sc, self.eight[h], cx, 0.70, side, side, op,
                      ring=[(q, int(np.argmax(self.eight[h, q])))] if on else (),
                      ring_ink=DIGIT_INK, hot=self.EIGHT, vmax=1.0, lines=0)
            if on:
                self.rect(sc, self.P(cx, 0.70), side + 12, side + 12, self.EIGHT, op,
                          border=self.EIGHT, border_w=2.6, radius=6, lift=0.7, fill_op=0.0)
            per = self.width // H
            for i in range(per):
                self.box(sc, cx + (i - (per - 1) / 2) * self.frac(11)[0], 0.845, 9, 16,
                         self.EIGHT, op * (1.0 if on else 0.45), radius=2)
        self.text(sc, x0, 0.915, "head %d: %s" % (head + 1, self._said(q, self.eight[head, q])),
                  op, anchor="lm", strong=True, size=0.034)
        self.text(sc, x1, 0.915, "%d heads of %d numbers" % (H, self.width // H), op,
                  anchor="rm")

    def phase_3(self, sc, u, op):
        """How far back each head reads, as a table: a row a head, a column a distance."""
        _N, q = self.N, self.N - 1
        x0, x1, y = self.ROW
        tw, th, size = self.TOKEN["w"], self.TOKEN["h"], self.TOKEN["size"]
        xs = self.tokens(sc, ["" if j == q else ch for j, ch in enumerate(self.chars)], x0, x1, y,
                         op, **self.TOKEN)
        self.box(sc, xs[q], y, tw, th, INK_BOX, op, self.chars[q], ink=WHITE, size=size)
        for d, name in enumerate(("itself", "1 back", "2", "3", "4")):
            self.text(sc, xs[q - d], y + 0.07, name, op, size=0.026)
        self.text(sc, (xs[0] + xs[q - 5]) / 2, y + 0.07, "further back", op, size=0.026)
        self.polyline(sc, [self.P(xs[0] - 0.02, y + 0.046), self.P(xs[q - 5] + 0.02, y + 0.046)],
                      1.6, AXIS, op)
        names = ["one head"] + ["head %d" % (h + 1) for h in range(self.heads)]
        cx0, pitch, top, step = 0.20, 0.082, 0.505, 0.0465
        for k, name in enumerate(self.COLUMNS):
            self.text(sc, cx0 + k * pitch, top - 0.04, name, op, strong=True, size=0.028)
        peak = float(self.offsets.max())
        for r, (name, row) in enumerate(zip(names, self.offsets)):
            ink = self.ONE if r == 0 else self.EIGHT
            yy = top + r * step + (0.012 if r else 0.0)
            seen = op * smoothstep(0.0, 1.0, (u - 0.05 - 0.07 * r) / 0.12)
            self.text(sc, cx0 - 0.055, yy, name, seen, anchor="rm", ink=dark(ink), size=0.030)
            best = int(np.argmax(row))
            for k, v in enumerate(row):
                fill = tuple(float(c) for c in heat(v / peak, ink))
                self.box(sc, cx0 + k * pitch, yy, 72, 21, fill, seen, "%.0f%%" % (100 * v),
                         ink=WHITE if v / peak > 0.6 else None, size=0.028, radius=3,
                         border=DIGIT_INK if k == best else None)
        before, own = 100 * self.offsets[1:, 1], 100 * self.offsets[1:, 0]
        even = 100 * self.even
        self.lines(sc, 0.70, 0.52, [
            ("the one head must be one mixture", dark(self.ONE)),
            ("the eight (seed 0): eight mixtures", dark(self.EIGHT)),
            "on the reader itself: %.0f%% to %.0f%%" % (own.min(), own.max()),
            "on the one before: %.0f%% to %.0f%%" % (before.min(), before.max()),
            "an even spread would put %.0f%% on each of these and %.0f%% further back" % (
                even[0], even[-1]),
            "outlined: where a row looks most"], op * smoothstep(0.55, 0.7, u), step=0.046,
            width=0.27)

    def _two(self, sc, op, y=0.33):
        """The two models as things: the same numbers, cut differently."""
        per = self.width // self.heads
        for row, (groups, ink, name) in enumerate((
                (1, self.ONE, "one head, %d numbers wide" % self.width),
                (self.heads, self.EIGHT, "%d heads, %d numbers each" % (self.heads, per)))):
            self._strip(sc, 0.05, y + row * 0.065, groups, ink, op, cell=8.0,
                        gap=4.0 * (groups > 1))
            self.text(sc, 0.05 + self.frac(64 * 8.0 + 7 * 4.0)[0] + 0.02, y + row * 0.065, name,
                      op, anchor="lm", ink=dark(ink))

    def phase_4(self, sc, u, op):
        self._two(sc, op)
        chart = self.chart(((0.10, 0.60), (0.53, 0.84)), self.updates, self.curves)
        self.axes(sc, chart, op, "training update", "held-out loss (nats)")
        go = u / 0.85
        mean = self.curves.mean(1)
        for i, ink in enumerate((self.ONE, self.EIGHT)):
            for trace in self.curves[i]:
                self.plot(sc, chart, self.updates, trace, ink, op * 0.45, go, width=1.0, dot=False)
            self.plot(sc, chart, self.updates, mean[i], ink, op, go, width=2.6)
        now = int(min(go, 1.0) * (len(self.updates) - 1))
        self.lines(sc, 0.67, 0.58, [("update %d" % self.updates[now], None),
                                    ("one head  %.3f" % mean[0][now], dark(self.ONE)),
                                    ("eight heads  %.3f" % mean[1][now], dark(self.EIGHT))],
                   op, step=0.06, size=0.036)
        self.lines(sc, 0.67, 0.79, ["%s parameters each" % commas(self.parameters[0]),
                                    "thin lines: each of %d seeds" % self.seeds], op, step=0.045)

    def phase_5(self, sc, u, op):
        """Every seed's held-out loss on one scale: a dot a model, a bar the mean."""
        self._two(sc, op)
        lo, hi = float(self.loss.min()), float(self.loss.max())
        pad = max(0.35 * (hi - lo), 0.005)
        lo, hi = lo - pad, hi + pad
        x0, x1 = 0.24, 0.80
        fx = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)  # noqa: E731
        self.text(sc, x0, 0.47, "next-character loss on %d held-out windows (lower is better)" %
                  self.tested, op, anchor="lm", strong=True)
        base = 0.76
        self.polyline(sc, [self.P(x0, base), self.P(x1, base)], 1.6, AXIS, op)
        ticks = [v for v in nice_ticks(lo, hi, 4) if lo <= v <= hi]
        for v in ticks:
            self.polyline(sc, [self.P(fx(v), 0.51), self.P(fx(v), base + 0.012)], 1.0, GRID, op)
            self.text(sc, fx(v), base + 0.04, number(v), op)
        self.text(sc, x1 + 0.02, base + 0.04, "nats", op, anchor="lm")
        grow = smoothstep(0.05, 0.45, u)
        for r, (name, ink, row) in enumerate(zip(("one head", "eight heads"),
                                                 (self.ONE, self.EIGHT), self.loss)):
            y = 0.575 + r * 0.105
            self.text(sc, x0 - 0.02, y, name, op, anchor="rm", ink=dark(ink), size=0.034)
            self.polyline(sc, [self.P(fx(row.min()), y), self.P(fx(row.max()), y)], 5.0,
                          soft(ink, 0.3), op * grow)
            for v in row:
                self.disk(sc, self.P(fx(v), y), 7.0, ink, op * grow, lift=1.2, outline=0.0)
            mx = fx(row.mean())
            self.polyline(sc, [self.P(mx, y - 0.035), self.P(mx, y + 0.035)], 3.0, dark(ink),
                          op * grow)
            self.text(sc, fx(row.max()) + 0.02, y, "mean %.3f" % row.mean(), op * grow,
                      anchor="lm", strong=True)
        self.lines(sc, 0.10, 0.865, [
            "eight heads: %.3f nats lower on average; %s" % (
                self.gain, "every eight-head model is below every one-head model" if self.clear
                else "the two groups of seeds overlap"),
            "the scale is zoomed: guessing evenly costs %.2f nats. no window was trained on "
            "twice" % self.guess], op * smoothstep(0.5, 0.75, u), size=0.031, step=0.046)


def si_bytes(v):
    """A byte count in decimal units (1 GB = 10^9 bytes)."""
    for scale, unit in ((1e9, "GB"), (1e6, "MB"), (1e3, "kB")):
        if v >= scale:
            return "%.3g %s" % (v / scale, unit)
    return "%d bytes" % v


class GroupedQueryAttention(Mechanism):
    """Query heads sharing key/value heads: the wiring of the three variants,
    one token's cache drawn to scale, that cache over a long context, and
    what sharing cost three small trained models."""

    TITLE = "Grouped-query attention"
    PHASES = (("who shares", 13.0), ("one token", 12.0), ("long context", 10.0),
              ("training", 10.0), ("held out", 10.0))
    NAMES = ("multi-head", "grouped-query", "multi-query")
    INK = (RED, GREEN, BLUE)
    # the model the study's memory count is for
    QUERY_HEADS, KV_HEADS, LAYERS, HEAD_WIDTH, BYTES = 8, (8, 2, 1), 32, 128, 2
    CONTEXT, SLAB = 8192, 1024

    def setup(self):
        s = self.study
        self.per_token = [int(v) for v in s.panels[0].data["values"]]
        self.steps = np.asarray(s.panels[1].data["x"], float)
        self.traces = np.asarray(s.panels[1].data["ys"], float)
        self.heldout = [float(v) for v in s.metadata["heldout_loss"]]
        self.parameters = [int(v) for v in s.panels[3].data["values"]]
        self.updates = int(s.metadata["steps"])
        configs = configs_of(s)
        self.small_q = configs[0].heads
        self.small_kv = [c.kv_heads for c in configs]

    def subtitle_for(self, k):
        b = self.per_token
        return [
            "every attention head has its own query (Q). in multi-head attention each also has "
            "its own keys and values (K V); the other two make query heads share them",
            "keys and values are what a model keeps in memory for every token it has read. with "
            "%d, %d and %d key/value heads that is %s, %s and %s bytes per token (exact count)" % (
                *self.KV_HEADS, commas(b[0]), commas(b[1]), commas(b[2])),
            "over a context of %s tokens: %s, %s and %s. sharing key/value heads is how a long "
            "context fits in memory" % (commas(self.CONTEXT), *(
                si_bytes(v * self.CONTEXT) for v in b)),
            "does sharing hurt? three small models with %d query heads and %d, %d and %d "
            "key/value heads, %d updates each on the same Shakespeare windows" % (
                self.small_q, *self.small_kv, self.updates),
            "held-out loss %.3f, %.3f and %.3f nats: in this small run sharing cost nothing, and "
            "the models that share have fewer parameters" % tuple(self.heldout),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _wiring(self, sc, cx, yq, ykv, q, kv, ink, op, pitch=0.032, w=22.0, h=26.0, beat=None,
                letters=True):
        """``q`` query heads over ``kv`` key/value heads, each query head joined
        to the one it reads; ``beat`` (0..1) sends keys and values up the lines."""
        gap = self.frac(0, h / 2)[1]
        for g in range(kv):
            members = [i for i in range(q) if i * kv // q == g]
            gx = cx + (np.mean(members) - (q - 1) / 2) * pitch
            self.box(sc, gx, ykv, self.span(pitch * len(members))[0] - (self.span(pitch)[0] - w), h,
                     ink, op, "K V" if letters and len(members) > 1 else None, ink=WHITE,
                     radius=4)
            for i in members:
                qx = cx + (i - (q - 1) / 2) * pitch
                a, b = (gx if kv < q else qx, ykv - gap - 0.008), (qx, yq + gap + 0.008)
                self.arrow(sc, a, b, ink, op, 1.0, 1.8, head=False)
                if beat is not None:
                    self.pulse(sc, a, b, beat, ink, op, r=3.6)
        for i in range(q):
            self.box(sc, cx + (i - (q - 1) / 2) * pitch, yq, w, h, PAPER, op,
                     "Q" if letters else None, border=AXIS, radius=4)

    def phase_1(self, sc, u, op):
        Q = self.QUERY_HEADS
        notes = ("each query head has its own", "four query heads share each",
                 "all eight share one")
        for i, cx in enumerate((0.19, 0.50, 0.81)):
            a = op * smoothstep(0.02 + 0.27 * i, 0.12 + 0.27 * i, u)
            kv = self.KV_HEADS[i]
            self.text(sc, cx, 0.32, self.NAMES[i], a, ink=dark(self.INK[i]), size=0.040)
            self.text(sc, cx, 0.405, "%d query heads" % Q, a)
            self._wiring(sc, cx, 0.475, 0.735, Q, kv, self.INK[i], a,
                         beat=(u * 6.0 - 0.27 * 6.0 * i - 0.5) % 1.0)
            self.text(sc, cx, 0.815, "%d key/value head%s" % (kv, "s" if kv > 1 else ""), a,
                      strong=True, size=0.034)
            self.text(sc, cx, 0.875, notes[i] if Q // kv in (1, 4, 8) else "", a)

    def phase_2(self, sc, u, op):
        L, per = self.LAYERS, 2 * self.HEAD_WIDTH * self.BYTES      # one head, one layer
        cell_w, cell_h = 20.0, 6.0
        height = L * cell_h
        shown = int(np.clip(u / 0.6, 0, 1) * L + 1e-9)
        self.text(sc, 0.50, 0.37, "per token: %d layers x key/value heads x (%d key numbers + "
                  "%d value numbers) x %d bytes" % (L, self.HEAD_WIDTH, self.HEAD_WIDTH,
                                                    self.BYTES), op)
        for i, cx in enumerate((0.24, 0.53, 0.80)):
            kv, ink = self.KV_HEADS[i], self.INK[i]
            self.text(sc, cx, 0.315, self.NAMES[i], op, ink=dark(ink), size=0.038)
            width = kv * cell_w
            self.grid(sc, np.ones((L, kv)), cx, 0.62, width, height, op, reveal=shown / L,
                      vmax=1.0, hot=ink, lines=0, mask=True)
            half = self.frac(width / 2, height / 2)
            for k in range(1, kv):
                x = cx - half[0] + self.frac(k * cell_w)[0]
                filled = height * shown / L
                self.rect(sc, self.P(x, 0.62 - half[1] + self.frac(0, filled / 2)[1]), 1.6,
                          max(filled, 0.1), PAPER, op, radius=0, lift=0.9)
            if i == 0:
                self.text(sc, cx - half[0] - 0.016, 0.62, "%d layers" % L, op, anchor="rm")
            self.text(sc, cx, 0.875, "%s bytes" % commas(shown * kv * per), op, strong=True,
                      size=0.036)
            self.text(sc, cx, 0.925, "%d key/value head%s%s" % (
                kv, "s" if kv > 1 else "", "" if i == 0 else ": %d times less" % (
                    self.per_token[0] // self.per_token[i])), op)

    def phase_3(self, sc, u, op):
        go = smoothstep(0.0, 1.0, min(u / 0.75, 1.0))
        tokens = int(round(self.CONTEXT * go))
        self.text(sc, 0.05, 0.33, "context: %s tokens" % commas(tokens), op, anchor="lm",
                  strong=True, size=0.042)
        self.text(sc, 0.05, 0.39, "every token read keeps its keys and values in memory", op,
                  anchor="lm")
        fx, full, thick = 0.20, self.span(0.58)[0], 44.0
        most = self.per_token[0] * self.CONTEXT
        for i, (name, ink, b) in enumerate(zip(self.NAMES, self.INK, self.per_token)):
            y = 0.50 + i * 0.125
            length = full * b * tokens / most
            self.hbar(sc, fx, y, length, ink, op, name, "", 1.0, thick=thick, track=full)
            for k in range(1, tokens // self.SLAB + 1):
                x = fx + self.frac(full * b * k * self.SLAB / most)[0]
                if k * self.SLAB < tokens:
                    self.rect(sc, self.P(x, y), 1.8, thick, PAPER, op, radius=0, lift=0.9)
            self.text(sc, fx + self.frac(length)[0] + 0.012, y, si_bytes(b * tokens), op,
                      anchor="lm", strong=True, size=0.034)
        self.text(sc, 0.05, 0.895, "exact count: %d layers, heads %d numbers wide, %d bytes a "
                  "number; a slab is %s tokens" % (self.LAYERS, self.HEAD_WIDTH, self.BYTES,
                                                   commas(self.SLAB)), op, anchor="lm")

    def _small(self, sc, op):
        """The three trained models: the same wiring, four query heads."""
        for i, cx in enumerate((0.66, 0.79, 0.92)):
            self._wiring(sc, cx, 0.325, 0.43, self.small_q, self.small_kv[i], self.INK[i], op,
                         pitch=0.022, w=15.0, h=18.0, letters=False)
            self.text(sc, cx, 0.50, self.NAMES[i], op, ink=dark(self.INK[i]))

    def phase_4(self, sc, u, op):
        self._small(sc, op)
        chart = self.chart(((0.08, 0.54), (0.36, 0.82)), self.steps, self.traces)
        self.axes(sc, chart, op, "training update", "loss (nats)")
        go = u / 0.85
        for trace, ink in zip(self.traces, self.INK):
            self.plot(sc, chart, self.steps, trace, ink, op, go)
        now = int(min(go, 1.0) * (len(self.steps) - 1))
        self.lines(sc, 0.62, 0.60, [("update %d" % self.steps[now], None)] + [
            ("%s  %.2f" % (name, trace[now]), dark(ink))
            for name, trace, ink in zip(self.NAMES, self.traces, self.INK)],
            op, step=0.06, size=0.036)

    def phase_5(self, sc, u, op):
        self._small(sc, op)
        fx, full = 0.20, self.span(0.24)[0]
        worst = max(self.heldout)
        self.text(sc, 0.05, 0.36, "loss on held-out text (lower is better)", op, anchor="lm",
                  strong=True)
        for i, (name, ink, v) in enumerate(zip(self.NAMES, self.INK, self.heldout)):
            self.hbar(sc, fx, 0.45 + i * 0.09, full * v / worst, ink, op, name, "%.3f nats" % v,
                      u / 0.4, thick=30.0, track=full)
        self.lines(sc, 0.62, 0.60, [("parameters to learn", None)] + [
            ("%s  %s" % (name, commas(n)), dark(ink))
            for name, n, ink in zip(self.NAMES, self.parameters, self.INK)],
            op * smoothstep(0.35, 0.55, u), step=0.06, size=0.036)
        spread = max(self.heldout) - min(self.heldout)
        self.lines(sc, 0.05, 0.78, [
            "the three finish within %.2f nats of each other" % spread,
            "one small %d-update run: not a ranking of the three" % self.updates],
            op * smoothstep(0.5, 0.7, u), step=0.048, size=0.032)


KV_TEXT = "Good morrow, neighbour Baptista."     # from the study's held-out passage
KV_SHOWN = 16                                       # numbers of each key and value drawn


def kv_cache_demo():
    """The pinned GPT-2 reading :data:`KV_TEXT` twice: one full forward pass,
    and token by token with the key/value cache. Per token: what each predicts
    next, the largest difference between their logits, and the first
    :data:`KV_SHOWN` numbers of its layer-1 key and value from both passes."""
    from . import gpt2, gpt2_hooks

    tok = gpt2.Tokenizer()
    ids = [int(i) for i in tok.encode(KV_TEXT)]

    def build():
        full = gpt2_hooks.forward(ids)
        cache, rows = None, []
        for token in ids:
            r = gpt2_hooks.forward([token], cache=cache)
            rows.append(r["logits"][0])
            cache = r["cache"]
        rows = np.array(rows)
        out = {"full_next": full["logits"].argmax(1), "cached_next": rows.argmax(1),
               "diff": np.abs(full["logits"] - rows).max(1),
               "peak": np.abs(full["logits"]).max()}
        for name, source in (("full", full["cache"]), ("cached", cache)):
            # layer 1: (head, token, 64) -> each token's 768 numbers, head after head
            for j, part in enumerate(("key", "value")):
                flat = np.asarray(source[0][j]).transpose(1, 0, 2).reshape(len(ids), -1)
                out["%s_%s" % (name, part)] = flat[:, :KV_SHOWN]
        return out

    a = cached_npz("mech-kv-cache-demo", {"ids": ids, "version": 1, "shown": KV_SHOWN,
                                          "model_revision": gpt2.REVISION}, build)
    return dict(a, ids=np.array(ids), tokens=[tok.decode([i]) for i in ids],
                next=[tok.decode([int(i)]) for i in a["full_next"]],
                cached=[tok.decode([int(i)]) for i in a["cached_next"]])


def token_label(s):
    """A GPT-2 token as a short label (its leading space dropped)."""
    s = s.replace("\n", "↵").strip()
    return s if s else "␣"


class KVCache(Mechanism):
    """Decoding with and without the key/value cache: the table of keys and
    values recomputed whole at every step, then gaining one row a step; the
    two giving the same predictions; the work as a triangle against a
    diagonal; and the memory the cache costs."""

    TITLE = "The key-value cache"
    PHASES = (("no cache", 13.0), ("with a cache", 13.0), ("same answer", 10.0),
              ("the work", 10.0), ("the memory", 10.0))
    OFF, ON = RED, BLUE
    ROW0, STEP = 0.365, 0.058
    LAYERS, WIDTH = 12, 768

    def setup(self):
        s = self.study
        d = kv_cache_demo()
        self.demo = d
        self.tokens_shown = [token_label(t) for t in d["tokens"]]
        self.next_full = [token_label(t) for t in d["next"]]
        self.next_cached = [token_label(t) for t in d["cached"]]
        self.agree = int(np.sum(d["full_next"] == d["cached_next"]))
        self.T = len(self.tokens_shown)
        self.key, self.value = d["full_key"], d["full_value"]
        self.diff = np.asarray(s.panels[0].data["ys"], float)[0]
        self.n = np.asarray(s.panels[1].data["x"], float)
        self.work = np.asarray(s.panels[1].data["ys"], float)       # per step: no cache, cache
        self.total = [int(v) for v in s.panels[3].data["values"]]
        self.memory = np.asarray(s.panels[2].data["ys"], float)[0]
        self.per_token = int(self.memory[0])

    def subtitle_for(self, k):
        T, N = self.T, len(self.n)
        return [
            "GPT-2 writes one token at a time, and every token it has read needs a key and a "
            "value. without a cache each step computes them all again: %d rows of work for "
            "these %d tokens" % (T * (T + 1) // 2, T),
            "a token's key and value never change once it has been read. so keep them: with the "
            "cache each step computes one new row, %d in all" % T,
            "both ways give the same next token at %d of %d steps; over the study's %d tokens "
            "their logits differ by at most %.6f" % (self.agree, T, N, self.diff.max()),
            "over %d tokens, recomputing is a triangle of %d key-and-value computations; with "
            "the cache it is the diagonal: %d (exact count)" % (N, self.total[0], self.total[1]),
            "the price is memory: %s bytes for every token kept (%d layers, a key and a value "
            "of %d numbers each), %s bytes after %d tokens" % (
                commas(self.per_token), self.LAYERS, self.WIDTH, commas(self.memory[-1]), N),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _y(self, r):
        return self.ROW0 + r * self.STEP

    def _decode(self, sc, u, op, cached):
        """The table of keys and values at the step ``u`` has reached."""
        T, ink = self.T, self.ON if cached else self.OFF
        step = min(int(u * 1.06 * T), T - 1)
        within = min(u * 1.06 * T - step, 1.0)
        tx, kx, vx, wide = 0.12, 0.365, 0.635, 208.0
        for x, name in ((tx, "token read"), (kx, "its key"), (vx, "its value")):
            self.text(sc, x, 0.305, name, op, strong=True)
        self.text(sc, 0.785, 0.305, "at this step", op, anchor="lm", strong=True)
        for r in range(step + 1):
            y = self._y(r)
            new = r == step
            fresh = new or not cached                       # computed at this step
            self.box(sc, tx, y, 118, 24, PAPER, op, self.tokens_shown[r], border=AXIS,
                     size=0.030)
            a = op * (0.15 + 0.85 * smoothstep(0.08, 0.45, within)) if fresh else op
            for x, values in ((kx, self.key), (vx, self.value)):
                self.grid(sc, values[r][None], x, y, wide, 22, a, vmax=float(np.abs(values).max()),
                          frame=False, hot=AMBER, cold=VIOLET)
            if fresh:
                mid = (kx + vx) / 2
                self.rect(sc, self.P(mid, y), self.span(vx - kx)[0] + wide + 10, 28, ink, op,
                          border=ink, border_w=2.2, radius=4, lift=0.7, fill_op=0.0)
                self.pulse(sc, (tx + 0.07, y), (vx + 0.12, y), within / 0.5, ink, op)
            self.text(sc, 0.785, y, ("computed" if new else "computed again") if fresh
                      else "kept in the cache", op, anchor="lm",
                      ink=dark(ink) if fresh else None)
        done = step + 1
        now, so_far = (1, done) if cached else (done, done * (done + 1) // 2)
        self.text(sc, 0.05, 0.85, "step %d: %d row%s computed, %d so far" % (
            done, now, "" if now == 1 else "s", so_far), op, anchor="lm", strong=True, size=0.036)
        self.text(sc, 0.05, 0.905, "shown: the first %d of each %d numbers, in layer 1 of %d" % (
            KV_SHOWN, self.WIDTH, self.LAYERS), op, anchor="lm")

    def phase_1(self, sc, u, op):
        self._decode(sc, u, op, cached=False)

    def phase_2(self, sc, u, op):
        self._decode(sc, u, op, cached=True)

    def phase_3(self, sc, u, op):
        T = self.T
        cols = ((0.11, "token read", None), (0.265, "recomputing: next", self.OFF),
                (0.42, "cache: next", self.ON))
        for x, name, ink in cols:
            self.text(sc, x, 0.305, name, op, ink=None if ink is None else dark(ink),
                      strong=ink is None)
        shown = min(int(u / 0.6 * T) + 1, T)
        for r in range(shown):
            y = self._y(r)
            self.box(sc, cols[0][0], y, 112, 24, PAPER, op, self.tokens_shown[r], border=AXIS,
                     size=0.030)
            for (x, _name, ink), word in zip(cols[1:], (self.next_full[r], self.next_cached[r])):
                self.box(sc, x, y, 112, 24, soft(ink), op, word, ink=dark(ink, 0.4), border=ink,
                         size=0.030)
        self.text(sc, 0.05, 0.86, "the same next token at %d of %d steps" % (
            int(np.sum(self.demo["full_next"][:shown] == self.demo["cached_next"][:shown])),
            shown), op, anchor="lm", strong=True, size=0.034)
        micro = self.diff * 1e6
        xs = np.arange(1, len(micro) + 1, dtype=float)
        chart = self.chart(((0.60, 0.95), (0.40, 0.70)), xs, np.append(micro, 0.0))
        self.axes(sc, chart, op, "token", "largest logit difference (millionths)")
        self.plot(sc, chart, xs, micro, self.ON, op, u / 0.7, width=2.2)
        self.lines(sc, 0.57, 0.845, [
            "the study's %d tokens: at most %.6f apart" % (len(micro), self.diff.max()),
            "on logits as large as %.0f" % float(self.demo["peak"])], op, step=0.046)

    def phase_4(self, sc, u, op):
        N = len(self.n)
        step = int(np.clip(u / 0.75, 0, 1) * N + 1e-9)
        tri = np.tril(np.ones((N, N)))
        side = 234.0
        for cx, name, ink, cells, count in (
                (0.25, "no cache", self.OFF, tri, self.work[0][:step].sum()),
                (0.75, "with the cache", self.ON, np.eye(N), self.work[1][:step].sum())):
            self.text(sc, cx, 0.315, name, op, ink=dark(ink), size=0.038)
            self.grid(sc, cells, cx, 0.625, side, side, op, reveal=step / N, vmax=1.0, hot=ink,
                      lines=0.3, mask=True)
            self.text(sc, cx, 0.915, "%d computed" % count, op, strong=True, size=0.036)
        self.text(sc, 0.50, 0.47, "step %d" % step, op, strong=True, size=0.042)
        self.text(sc, 0.50, 0.53, "of %d" % N, op)
        self.lines(sc, 0.50, 0.66, ["a row: one step", "a cell: one token's", "key and value,",
                                    "computed"], op, anchor="mm", step=0.042)

    def phase_5(self, sc, u, op):
        N, L = len(self.n), self.LAYERS
        kept = int(np.clip(u / 0.75, 0, 1) * N + 1e-9)
        cells = np.zeros((L, N))
        cells[:, :kept] = 1.0
        mx, my, w, h = 0.21, 0.63, 260.0, 192.0
        self.text(sc, mx, 0.315, "the cache", op, ink=dark(self.ON), size=0.038)
        self.text(sc, mx, 0.365, "a column for every token kept", op)
        self.grid(sc, cells, mx, my, w, h, op, vmax=1.0, hot=self.ON, lines=0.3, mask=True)
        self.text(sc, mx, 0.875, "a row for each of the %d layers" % L, op)
        self.text(sc, mx, 0.92, "a cell: a key and a value, %s bytes" % commas(
            self.per_token / L), op)
        chart = self.chart(((0.52, 0.95), (0.44, 0.76)), self.n, np.append(self.memory, 0.0))
        self.axes(sc, chart, op, "tokens kept", "bytes in the cache")
        self.plot(sc, chart, self.n, self.memory, self.ON, op, (kept - 1) / max(N - 1, 1),
                  width=2.4)
        self.text(sc, 0.52, 0.335, "%d tokens: %s bytes" % (
            kept, commas(self.memory[kept - 1] if kept else 0)), op, anchor="lm", strong=True,
            size=0.036)


ANIMATIONS = {
    "causal-vs-bidirectional": CausalVsBidirectional,
    "prefix-lm-attention": PrefixLM,
    "cross-attention": CrossAttention,
    "multi-head-vs-single": MultiHeadVsSingle,
    "grouped-query-attention": GroupedQueryAttention,
    "kv-cache": KVCache,
}

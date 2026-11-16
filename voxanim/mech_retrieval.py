"""Retrieval and embeddings: animations 81-85 of the "next 100".

Each draws the mechanism first and the measurement after it. The curves,
scores and final numbers are the study's: for 81, 83 and 84 the redesigned
experiments of :mod:`voxanim.study_retrieval` (sixty questions put to the
pinned GPT-2 over a store of Tiny Shakespeare; ReLU contrastive encoders
trained on MNIST views from three seeds), for 82 and 85 the first ones
(:mod:`voxanim.next_systems_real`). What the studies did not keep as panels
is read from what they saved (their run evidence) or recomputed here with
their own code: the search graph and every step of a search through it, the
similarity of one real batch of views at each temperature, the first training
batch as the encoder ranked it before and after training, the neighbour each
embedding width retrieves. The tests check that these reproduce the studies'
numbers.
"""

from __future__ import annotations

import heapq
import re
from functools import lru_cache

import numpy as np

from . import layout
from . import next_systems as S
from . import realdata as R
from . import study_retrieval as SR
from .camera import REF_H, REF_W
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    GREEN,
    GRID,
    OCHRE,
    PAPER,
    RED,
    TEAL,
    TEXT,
    TRACK,
    VIOLET,
    WHITE,
    Mechanism,
    dark,
    soft,
)

INKY = (0.16, 0.17, 0.19)           # a box holding the model, or something hidden
WORD = re.compile(r"[a-z]+")


def evidence(study):
    """What the study's run saved: ``(settings, arrays)``; the arrays are its
    weights (``parameter_*``) and its recorded traces (``history_*``)."""
    record = study.metadata["training_provenance"]
    record = record[0] if isinstance(record, (list, tuple)) else record
    with np.load(record["artifact"], allow_pickle=False) as f:
        return record["settings"], {k: f[k] for k in f.files}


def seg(u, a, b):
    """Progress 0..1 of ``u`` through ``a..b``."""
    return float(np.clip((u - a) / (b - a), 0.0, 1.0))


def unit(z):
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-10)


def log_sweep(xs, g):
    """A cursor ``g`` (0..1) of the way along a logarithmic axis over ``xs``:
    ``(u, at)``, the fraction of the linear range :meth:`Mechanism.plot`
    draws to, and the index of the last of ``xs`` it has reached."""
    xs = np.asarray(xs, float)
    x = float(np.exp(np.log(xs[0]) + np.clip(g, 0.0, 1.0) * (np.log(xs[-1]) - np.log(xs[0]))))
    return (x - xs[0]) / (xs[-1] - xs[0]), int(np.searchsorted(xs, x * (1 + 1e-9), "right") - 1)


def clip(s, size, width=None):
    """``s``, or as many of its words as fit ``width`` (a fraction of the
    frame) followed by an ellipsis."""
    if width is None or layout.text_width(s, size) <= width:
        return s
    words = s.split(" ")
    while len(words) > 1 and layout.text_width(" ".join(words) + " ...", size) > width:
        words.pop()
    return " ".join(words) + " ..."


def percent(v, digits=1):
    return "%.*f%%" % (digits, 100 * float(v))


class _Kit(Mechanism):
    """Drawing shared by this section's animations."""

    def vbar(self, sc, fx, fy, height, width, ink, op, value="", size=TEXT):
        """A vertical bar standing on ``(fx, fy)``, ``height`` px tall, its
        ``value`` above it."""
        h = max(float(height), 0.6)
        base = self.P(float(fx), float(fy))
        self.rect(sc, self.shift(base, 0, -h / 2), width, h, ink, op, radius=min(3.0, h / 2))
        if value != "":
            self.label(sc, self.shift(base, 0, -h - 11), str(value), size, opacity=op,
                       ink=layout.STRONG_INK)

    def marked(self, sc, fx, fy, s, op, size, marks=(), ink=AMBER, mark_op=1.0, width=None):
        """``s`` from ``(fx, fy)`` rightwards, the words in ``marks``
        (lower case) on a highlight; cut short (and marked so) past
        ``width``, a fraction of the frame."""
        s = clip(s, size, width)
        if mark_op > 0.01:
            for m in re.finditer(r"[A-Za-z]+", s):
                if m.group().lower() in marks:
                    before = layout.text_width(s[:m.start()], size) * REF_W
                    w = layout.text_width(m.group(), size) * REF_W
                    at = self.shift(self.P(float(fx), float(fy)), before + w / 2, 0)
                    self.rect(sc, at, w + 6, size * REF_H * 1.3, soft(ink, 0.3), op * mark_op,
                              radius=4, lift=0.6)
        self.text(sc, fx, fy, s, op, size, "lm", strong=True)

    def chip(self, sc, fx, fy, w, h, fill, op, text=None, ink=None, border=None, size=TEXT):
        """A :meth:`box` lying on top of another box."""
        c = self.P(float(fx), float(fy))
        self.rect(sc, c, w, h, fill, op, border=border, radius=5.0, lift=0.9)
        if text is not None:
            self.label(sc, c, str(text), size, opacity=op,
                       ink=layout.STRONG_INK if ink is None else ink)

    def frame(self, sc, fx, fy, w, h, op, border=AXIS, fill=PAPER, border_w=1.2):
        self.rect(sc, self.P(float(fx), float(fy)), w, h, fill, op, border=border,
                  border_w=border_w, radius=5, lift=0.1)


# ====================================================================== 81
tfidf_table = SR.tfidf_table
tfidf_vector = SR.tfidf_vector


class RetrievalAugmentedGeneration(_Kit):
    """A cut-off line of Shakespeare as the question: GPT-2's guesses alone,
    the best-matching lines on file scored against it word by word, the best
    one put in the prompt and the true next word's probability measured
    again; the same with the question's own line taken off file; then sixty
    such questions, two retrievers, with and without their own lines."""

    TITLE = "Retrieval-augmented generation"
    PHASES = (("question", 10.0), ("retrieve", 13.0), ("in the prompt", 12.0),
              ("off file", 10.0), ("all asked", 13.0))
    Q = (0.05, 0.281, 0.40)             # the question card: x0, x1, y
    CARDS = (0.35, 0.80)                # the passages' cards: x0, x1
    ROWS = (0.365, 0.128)               # their first y and step
    SIZE = 0.029
    SHOWN = 5                           # lines on file drawn: the best matches
    # the measured prompts, as rows of the last phase: condition, label, ink
    TABLE = (("words_in", "own line on file: found by words", GREEN),
             ("embedding_in", "own line on file: found by embedding", GREEN),
             ("words_out", "own line removed: best by words", AMBER),
             ("embedding_out", "own line removed: best by embedding", AMBER),
             ("random", "a random line from the file", VIOLET),
             ("alone", "no line: the question alone", BLUE))

    def setup(self):
        from . import gpt2

        s = self.study
        settings, arrays = evidence(s)
        self.store = list(settings["store"])
        self.asked = [int(i) for i in settings["asked"]]
        self.query = s.metadata["questions"][0]
        self.source = self.asked[0]
        every = np.asarray(arrays["history_word_scores"][0], float)
        ranked = np.argsort(-every, kind="stable")
        # the five best matches of the whole file, in the order they are on file
        self.on_file = sorted(int(j) for j in ranked[:self.SHOWN])
        self.passages = [self.store[j] for j in self.on_file]
        self.scores = every[self.on_file]
        self.order = np.argsort(-self.scores, kind="stable")
        picks = dict(zip(SR.CONDITIONS, (int(j) for j in arrays["history_pick"][0])))
        self.best = self.on_file.index(picks["words_in"])
        self.second = self.on_file.index(picks["words_out"])
        self.found = picks["words_in"] == self.source
        self.prob = np.asarray(s.panels[2].data["values"], float)      # question, prompt
        one = dict(zip(SR.CONDITIONS, self.prob[0]))
        self.probs = [float(one[c]) for c in ("alone", "words_in", "words_out")]
        tok = gpt2.Tokenizer()
        ids = [int(i) for i in arrays["history_example_candidates"]]
        answer = int(arrays["history_answer"][0])
        tops = np.asarray(arrays["history_example_tops"], float)[[0, 2, 4]]
        self.answer = tok.decode([answer]).strip()
        rank = sorted(range(len(ids)), key=lambda j: (ids[j] != answer, -tops[0, j]))
        self.cands = [tok.decode([ids[j]]).strip() for j in rank]      # the answer first
        self.tops = tops[:, rank]
        words, idf = tfidf_table()
        index = {w: i for i, w in enumerate(words)}
        self.qwords = WORD.findall(self.query.lower())
        self.qidf = [float(idf[index[w]]) for w in self.qwords]
        self.shared = [set(self.qwords) & set(WORD.findall(p.lower())) & set(index)
                       for p in self.passages]
        # the measurement: every question under every prompt
        alone = self.prob[:, 0]
        self.mean = dict(zip(SR.CONDITIONS, self.prob.mean(0)))
        self.helped = {c: int((self.prob[:, k] > alone).sum())
                       for k, c in enumerate(SR.CONDITIONS)}
        self.hurt = {c: int((self.prob[:, k] < alone).sum())
                     for k, c in enumerate(SR.CONDITIONS)}
        self.first = [int(round(float(v) * len(self.asked)))
                      for v in s.panels[3].data["values"]]

    def subtitle_for(self, k):
        best, second = self.scores[self.order[0]], self.scores[self.order[1]]
        alone, top, off = self.probs
        n, m = len(self.asked), self.mean
        out = max(self.helped["words_out"], self.helped["embedding_out"])
        return [
            "a line of Shakespeare is cut off and GPT-2 must give the next word, \"%s\": on its "
            "own it gives that word %s; %d held-out lines are on file, here the %d that will "
            "match best" % (self.answer, percent(alone), len(self.store), self.SHOWN),
            "each line on file is scored by the words it shares with the question, rare words "
            "counting more: %s scores %.2f, the next %.2f" % (
                "the line the question was cut from" if self.found else "the best line",
                best, second),
            "the best-scoring line goes into the prompt ahead of the question, and GPT-2 now "
            "gives \"%s\" %s instead of %s%s" % (
                self.answer, percent(top), percent(alone),
                ": the answer is there to be copied" if self.found else ""),
            "take the question's own line off file and the best match left only shares some "
            "words with it: \"%s\" gets %s, against %s alone; retrieval can pass on only what "
            "the file holds" % (self.answer, percent(off), percent(alone)),
            "%d questions: with its own line on file the true next word goes from %s to %s "
            "(found by words) or %s (by embedding); with it removed, %s and %s, helping %s "
            "than a random line" % (
                n, percent(m["alone"]), percent(m["words_in"]), percent(m["embedding_in"]),
                percent(m["words_out"]), percent(m["embedding_out"]),
                "more often" if out > self.helped["random"] else "no more often"),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _question(self, sc, op, marks=(), mark_op=0.0):
        x0, x1, y = self.Q
        w = self.span(x1 - x0)[0]
        self.box(sc, (x0 + x1) / 2, y, w, 44, soft(BLUE, 0.78), op, border=BLUE)
        self.marked(sc, x0 + 0.012, y, self.query, op, 0.032, marks, mark_op=mark_op)
        self.box(sc, x1 + 0.03, y, 40, 44, INKY, op, "?", ink=WHITE, size=0.036)

    def _passages(self, sc, op, shown=5.0, marks=None, scored=0.0, chosen=0.0):
        """The lines on file that are drawn: ``shown`` of them present, the
        first ``scored`` of them with their score bars, the top one coloured
        as ``chosen`` rises."""
        x0, x1 = self.CARDS
        y0, step = self.ROWS
        w = self.span(x1 - x0)[0]
        full = 80.0
        for i, line in enumerate(self.passages):
            a = op * seg(shown, i, i + 0.6)
            y = y0 + i * step
            fill, border = PAPER, AXIS
            if i == self.order[0] and chosen > 0:
                fill, border = soft(GREEN, 1 - 0.3 * chosen), GREEN
            self.box(sc, (x0 + x1) / 2, y, w, 40, fill, a, border=border)
            live = marks is not None and marks[0] == i
            self.marked(sc, x0 + 0.012, y, line, a, self.SIZE,
                        self.shared[i] if live else (), mark_op=marks[1] if live else 0.0,
                        width=self.frac(w - 20)[0])
            grow = seg(scored, i, i + 1)
            if grow > 0:
                ink = GREEN if i == self.order[0] and chosen > 0.5 else AMBER
                self.hbar(sc, x1 + 0.018, y, full * self.scores[i], ink, a,
                          value="%.2f" % self.scores[i], u=grow / 0.6, thick=22.0, track=full)

    def _guesses(self, sc, fx, ys, row, op, ink, u, full, top=1.0, rows=None):
        """GPT-2's probabilities for the candidate next words under prompt
        ``row`` (0 alone, 1 with the best line, 2 with the best line left
        once the question's own is off file), as bars ``full`` px long at
        probability ``top``."""
        for n, j in enumerate(range(len(self.cands)) if rows is None else rows):
            p = self.tops[row, j]
            self.hbar(sc, fx, ys[n], full * p / top, ink, op, "\"%s\"" % self.cands[j],
                      percent(p), u, thick=18.0)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        x0, x1, y = self.Q
        self.text(sc, x0, y - 0.085, "the question: what comes next?", op, anchor="lm",
                  strong=True, size=0.032)
        self._question(sc, op)
        # GPT-2 answering on its own
        a = op * smoothstep(0.10, 0.2, u)
        mid = (x0 + x1) / 2
        self.arrow(sc, (mid, y + 0.06), (mid, y + 0.125), AXIS, a, seg(u, 0.1, 0.2))
        self.box(sc, mid, y + 0.175, 84, 38, INKY, a, "GPT-2", ink=WHITE, size=0.032)
        b = op * smoothstep(0.2, 0.3, u)
        rows = sorted(range(4), key=lambda j: -self.tops[0, j])
        ys = [y + 0.265 + n * 0.058 for n in range(len(rows))]
        self._guesses(sc, x0 + 0.085, ys, 0, b, BLUE, seg(u, 0.22, 0.45), 130.0,
                      top=float(self.tops[0].max()), rows=rows)
        self.text(sc, x0, ys[-1] + 0.07, "the true next word is \"%s\"" % self.answer,
                  b * smoothstep(0.42, 0.5, u), anchor="lm", ink=dark(BLUE))
        # the lines on file
        c = op * smoothstep(0.5, 0.6, u)
        self.text(sc, self.CARDS[0], 0.295, "on file: %d held-out lines of Shakespeare; %d of "
                  "them" % (len(self.store), self.SHOWN), c, anchor="lm", strong=True,
                  size=0.032)
        self._passages(sc, op, shown=seg(u, 0.52, 0.9) * 5.6)

    def phase_2(self, sc, u, op):
        x0, x1, y = self.Q
        n = len(self.passages)
        turn = min(int(u / 0.84 * n), n - 1)
        within = seg(u, turn * 0.84 / n, (turn + 1) * 0.84 / n)
        done = u >= 0.84
        mark = 0.0 if done else smoothstep(0.0, 0.2, within) * (1 - smoothstep(0.85, 1.0, within))
        self.text(sc, x0, y - 0.085, "the question", op, anchor="lm", strong=True, size=0.032)
        self._question(sc, op, self.shared[turn], mark)
        # the question as weighted words
        self.text(sc, x0, y + 0.115, "its words, weighted by rarity", op, anchor="lm")
        peak = max(self.qidf)
        for j, (word, w) in enumerate(zip(self.qwords, self.qidf)):
            fx = x0 + 0.03 + j * 0.059
            live = (not done) and word in self.shared[turn]
            self.vbar(sc, fx, 0.845, 78.0 * w / peak, 30.0, AMBER if live and mark > 0.3 else BLUE,
                      op * (1.0 if live or mark < 0.3 else 0.45), "%.1f" % w)
            self.text(sc, fx, 0.885, word, op)
        # each line scored in turn
        self.text(sc, self.CARDS[0], 0.295, "words shared with each line", op, anchor="lm",
                  strong=True, size=0.032)
        self.text(sc, self.CARDS[1] + 0.018, 0.295, "match", op, anchor="lm", strong=True,
                  size=0.032)
        self._passages(sc, op, marks=(turn, mark),
                       scored=n if done else turn + smoothstep(0.45, 0.9, within),
                       chosen=smoothstep(0.86, 0.94, u))
        if not done:
            a = (x1 + 0.055, y)
            b = (self.CARDS[0] - 0.006, self.ROWS[0] + turn * self.ROWS[1])
            bend = 0.03 if turn else 0.0
            self.arc(sc, a, b, AMBER, op * mark, smoothstep(0.0, 0.35, within), bend, 2.4)
            self.pulse(sc, a, b, (within - 0.2) / 0.5, AMBER, op, bend)

    def _prompt(self, sc, op, y, passage=None, ink=GREEN, enter=1.0, answer=False, x0=0.05,
                x1=0.50):
        """A prompt as GPT-2 receives it: the passage (if any) on the line
        above the question."""
        w = self.span(x1 - x0)[0]
        two = passage is not None
        self.box(sc, (x0 + x1) / 2, y, w, 84 if two else 44, PAPER, op, border=AXIS)
        if two:
            slide = (1 - smoothstep(0.0, 1.0, enter)) * 0.25
            self.chip(sc, (x0 + x1) / 2 + slide, y - 0.04, w - 12, 32, soft(ink, 0.7),
                      op * enter, border=ink)
            self.text(sc, x0 + 0.016 + slide, y - 0.04,
                      clip(self.passages[passage], self.SIZE, self.frac(w - 28)[0]), op * enter,
                      self.SIZE, "lm", strong=True)
        qy = y + 0.04 if two else y
        self.chip(sc, x0 + 0.008 + self.frac(95)[0], qy, 190, 32, soft(BLUE, 0.78), op,
                  border=BLUE)
        self.text(sc, x0 + 0.02, qy, self.query, op, self.SIZE, "lm", strong=True)
        ax = x0 + 0.008 + self.frac(190 + 30)[0]
        if answer:
            self.chip(sc, ax, qy, 46, 32, GREEN, op, self.answer, ink=WHITE, size=self.SIZE)
        else:
            self.chip(sc, ax, qy, 34, 32, INKY, op, "?", ink=WHITE, size=0.032)

    def phase_3(self, sc, u, op):
        enter = seg(u, 0.30, 0.48)
        self.text(sc, 0.05, 0.33, "the prompt GPT-2 reads", op, anchor="lm", strong=True,
                  size=0.032)
        self._prompt(sc, op, 0.47, self.best, GREEN, enter, answer=self.found and u > 0.8)
        self.lines(sc, 0.05, 0.63, [("retrieved: the best-matching line", dark(GREEN)),
                                    ("the question, as before", dark(BLUE))],
                   op * smoothstep(0.34, 0.5, u), step=0.05)
        # the measured change, side by side
        c = op * smoothstep(0.62, 0.7, u)
        self.text(sc, 0.05, 0.775, "chance of the true next word", c, anchor="lm", strong=True)
        self.hbar(sc, 0.20, 0.84, 230.0 * self.probs[0], BLUE, c, "alone",
                  percent(self.probs[0]), thick=22.0, track=230.0)
        self.hbar(sc, 0.20, 0.905, 230.0 * self.probs[1], GREEN, c, "with the line",
                  percent(self.probs[1]), seg(u, 0.66, 0.85), thick=22.0, track=230.0)
        self.arrow(sc, (0.51, 0.47), (0.565, 0.47), AXIS, op)
        self.box(sc, 0.612, 0.47, 70, 38, INKY, op, "GPT-2", ink=WHITE, size=0.030)
        self.arrow(sc, (0.612, 0.515), (0.612, 0.55), AXIS, op, head=False)
        # its next-word probabilities, before and after the passage arrives
        after = u >= 0.56
        show = smoothstep(0.56, 0.62, u) if after else 1 - smoothstep(0.50, 0.56, u)
        self.text(sc, 0.56, 0.33, "next word, " + ("with the retrieved line" if after
                                                    else "the question alone"),
                  op * show, anchor="lm", strong=True, size=0.032)
        ys = [0.585 + n * 0.048 for n in range(len(self.cands))]
        if after:
            self._guesses(sc, 0.66, ys, 1, op * show, GREEN, seg(u, 0.58, 0.8), 200.0)
        else:
            self._guesses(sc, 0.66, ys, 0, op * show, BLUE, seg(u, 0.06, 0.26), 200.0)

    def phase_4(self, sc, u, op):
        rows = (("the question alone", None, BLUE),
                ("its own line is on file, and is retrieved", self.best, GREEN),
                ("that line taken off file: the best match left", self.second, AMBER))
        full = 190.0
        self.text(sc, 0.60, 0.295, "chance GPT-2 gives \"%s\"" % self.answer, op, anchor="lm",
                  strong=True, size=0.032)
        y = 0.327
        for n, ((name, passage, ink), p) in enumerate(zip(rows, self.probs)):
            a = op * smoothstep(n * 0.2, n * 0.2 + 0.12, u)
            half = 0.045 if passage is None else 0.087
            self.text(sc, 0.05, y - 0.022, name, a, anchor="lm", ink=dark(ink), size=0.030)
            mid = y + 0.012 + half
            self._prompt(sc, a, mid, passage, ink, answer=n == 1 and self.found and u > 0.5,
                         x1=0.49)
            self.arrow(sc, (0.50, mid), (0.585, mid), AXIS, a,
                       seg(u, n * 0.2 + 0.08, n * 0.2 + 0.2))
            self.hbar(sc, 0.60, mid, full * p, ink, a, value=percent(p),
                      u=seg(u, n * 0.2 + 0.15, n * 0.2 + 0.35), thick=30.0, track=full)
            y = mid + half + 0.058

    def phase_5(self, sc, u, op):
        n = len(self.asked)
        self.text(sc, 0.05, 0.30, "%d cut-off lines asked, one line put ahead of each" % n, op,
                  anchor="lm", strong=True, size=0.032)
        bx, sx = 0.435, 0.715                # the two columns of bars
        full, split = 170.0, 150.0
        self.text(sc, bx, 0.355, "true next word, mean", op, anchor="lm", strong=True)
        self.text(sc, sx, 0.355, "helped", op, anchor="lm", ink=dark(GREEN))
        self.text(sc, sx + self.frac(split)[0], 0.355, "hurt", op, anchor="rm", ink=dark(RED))
        for r, (key, name, ink) in enumerate(self.TABLE):
            y = 0.42 + r * 0.076
            a = op * smoothstep(0.04 + 0.11 * r, 0.12 + 0.11 * r, u)
            grow = seg(u, 0.08 + 0.11 * r, 0.26 + 0.11 * r)
            self.text(sc, bx - 0.012, y, name, a, anchor="rm", ink=dark(ink))
            self.hbar(sc, bx, y, full * self.mean[key], ink, a, value=percent(self.mean[key]),
                      u=grow, thick=22.0, track=full)
            if key == "alone":
                continue
            at = self.P(sx, y)
            good, bad = self.helped[key] / n, self.hurt[key] / n
            self.rect(sc, self.shift(at, split / 2, 0), split, 22.0, TRACK, 0.5 * a, radius=4,
                      lift=0.3)
            g = max(split * good * smoothstep(0.0, 1.0, grow), 0.1)
            b = max(split * bad * smoothstep(0.0, 1.0, grow), 0.1)
            self.rect(sc, self.shift(at, g / 2, 0), g, 22.0, GREEN, a, radius=min(4.0, g / 2))
            self.rect(sc, self.shift(at, split - b / 2, 0), b, 22.0, RED, a,
                      radius=min(4.0, b / 2))
            self.text(sc, sx + self.frac(split + 8)[0], y, "%d : %d" % (
                self.helped[key], self.hurt[key]), a * seg(grow, 0.6, 1.0), anchor="lm",
                strong=True)
        self.text(sc, 0.05, 0.915, "embedding: GPT-2's own 768 numbers for a line.  Own line "
                  "found first: by words %d of %d, by embedding %d" % (
                      self.first[0], n, self.first[1]),
                  op * smoothstep(0.7, 0.8, u), anchor="lm")


# ====================================================================== 82
def search_trace(graph, q, ef):
    """:meth:`voxanim.next_systems.HNSW.query`, one distance at a time.

    Returns ``(events, answer, checked)``: the same answer and count of
    distance computations as the study's search, and every step as a dict with
    ``kind`` (``enter``; ``look``: a distance computed; ``hop``: one computed
    and found nearer, so taken; ``drop``: down a layer; ``expand``: a
    base-layer candidate's links opened), ``layer``,
    ``node``, ``src`` (the node whose links are being read), ``d`` (squared
    distance to the query), ``n`` (distances computed so far) and ``best``
    (the nearest node found so far)."""
    x = graph.x
    order, known = [], set()
    events = []

    def dist(i):
        if i not in known:
            known.add(i)
            order.append(i)
        return float(np.sum((x[i] - q) ** 2))

    def note(kind, layer, node, src, d, best):
        events.append(dict(kind=kind, layer=layer, node=int(node), src=int(src), d=float(d),
                           n=len(order), best=int(best)))

    current = graph.entry
    current_d = dist(current)
    top = len(graph.edges) - 1
    note("enter", top, current, current, current_d, current)
    for layer in range(top, 0, -1):
        links = graph.edges[layer]
        changed = True
        while changed:
            changed = False
            owner = current
            for j in sorted(links[owner]):
                d = dist(j)
                if d < current_d:
                    current, current_d, changed = j, d, True
                    note("hop", layer, j, owner, d, current)
                else:
                    note("look", layer, j, owner, d, current)
        note("drop", layer - 1, current, current, current_d, current)
    candidates = [(current_d, current)]
    best = [(-current_d, current)]
    seen = {current}
    while candidates:
        d, i = heapq.heappop(candidates)
        if len(best) >= ef and d > -best[0][0]:
            break
        note("expand", 0, i, i, d, min((-b, k) for b, k in best)[1])
        for j in sorted(graph.edges[0][i]):
            if j in seen:
                continue
            seen.add(j)
            dj = dist(j)
            if len(best) < ef or dj < -best[0][0]:
                heapq.heappush(candidates, (dj, j))
                heapq.heappush(best, (-dj, j))
                if len(best) > ef:
                    heapq.heappop(best)
            note("look", 0, j, i, dj, min((-b, k) for b, k in best)[1])
    answer = min((-d, i) for d, i in best)[1]
    return events, int(answer), len(order)


@lru_cache(maxsize=1)
def hnsw_index(artifact):
    """The study's index again, from the encoder it saved: the 256 training
    embeddings, the 64 held-out queries, the graph over the embeddings and
    the flat projection the study draws them in."""
    from .next_systems_real import digits

    with np.load(artifact, allow_pickle=False) as f:
        w = f["parameter_encoder"]
    x, y, xt, yt, _ = digits()
    u, q = unit(x[:256] @ w), unit(xt[:64] @ w)
    graph = S.HNSW(u)
    mean = u.mean(0)
    _, _, vt = np.linalg.svd(u - mean, full_matrices=False)
    return dict(u=u, q=q, graph=graph, xy=(u - mean) @ vt[:2].T, qxy=(q - mean) @ vt[:2].T,
                y=y[:256], yt=yt[:64],
                images=R.mnist_split(1024, "train", side=28, seed=17)[0][:256],
                queries=R.mnist_split(384, "test", side=28, seed=73)[0][:64],
                exact=((q[:, None] - u[None]) ** 2).sum(-1).argmin(1))


class VectorSearchHNSW(_Kit):
    """A layered graph over 256 image embeddings: one search followed hop by
    hop down the layers, the same search widened, then recall against the
    number of distances computed."""

    TITLE = "Vector search: a layered graph"
    PHASES = (("the index", 11.0), ("descend", 18.0), ("search wider", 13.0),
              ("recall and work", 12.0))
    QUERY = 2                            # the held-out image searched for
    WIDTHS = (1, 8, 32)
    BIG = (0.49, 0.615, 400.0, 262.0)    # the map of one layer: x, y, w, h px
    RINGS = (0.5, 1.0, 1.5)              # distances from the query drawn as rings
    STACK = (0.135, 0.385, 0.135)        # the layers as a stack: x, top y, step
    SIDE = 0.78                          # the read-out column's left edge

    def setup(self):
        s = self.study
        ix = hnsw_index(s.metadata["training_provenance"]["artifact"])
        self.ix = ix
        self.graph = ix["graph"]
        self.levels = np.asarray(s.panels[0].data["classes"], int)
        # two layouts of the same points: the study's flat projection, and one
        # around the query, each point at its true distance from it, on its own
        # spoke (the spokes in the order the points lie around the projection's
        # centre, so that similar digits stay side by side)
        xy = np.asarray(s.panels[0].data["xy"], float)
        self.flat = xy / np.abs(xy).max(0)
        q = ix["q"][self.QUERY]
        self.dist = np.sqrt(((ix["u"] - q) ** 2).sum(1))
        self.reach = float(self.dist.max()) * 1.02
        spoke = np.argsort(np.argsort(np.arctan2(xy[:, 1], xy[:, 0])))
        angle = 2 * np.pi * spoke / len(xy)
        self.around = (self.dist / self.reach)[:, None] * np.c_[np.cos(angle), np.sin(angle)]
        self.efs = [int(v) for v in s.panels[3].data["x"]]
        self.recall = np.asarray(s.panels[3].data["ys"][0], float)
        self.work = np.asarray(s.panels[3].data["ys"][1], float)
        self.traces = {ef: search_trace(self.graph, ix["q"][self.QUERY], ef) for ef in self.efs}
        self.exact = int(ix["exact"][self.QUERY])
        self.top = len(self.graph.edges) - 1
        self.members = [sorted(g) for g in self.graph.edges]
        self.links = [sorted({(min(i, j), max(i, j)) for i, near in g.items() for j in near})
                      for g in self.graph.edges]

    def subtitle_for(self, k):
        n = len(self.flat)
        _e1, a1, c1 = self.traces[1]
        _e8, a8, c8 = self.traces[8]
        i8 = self.efs.index(8)
        return [
            "%d training digits, each a point (its 12-number embedding, flattened here) linked "
            "to its nearest neighbours; %d of them are also in layer 1, %d in layer 2 and %d in "
            "layer 3" % (n, len(self.members[1]), len(self.members[2]), len(self.members[3])),
            "a new digit is searched for from the top layer: hop to any linked point that is "
            "nearer, drop a layer when none is; it stops after %d of the %d distances" % (c1, n),
            "that greedy search %s; keeping the 8 best candidates open in the bottom layer "
            "computes %d distances and %s" % (
                "found the true nearest" if a1 == self.exact else "stopped short of the true "
                "nearest", c8, "finds it" if a8 == self.exact else "still misses it"),
            "over %d held-out digits: width 1 finds the true nearest %s of the time for %s of "
            "the distances, width 8 finds it %s for %s" % (
                len(self.ix["q"]), percent(self.recall[0]), percent(self.work[0]),
                percent(self.recall[i8]), percent(self.work[i8])),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _at(self, box, p, lift=0.0):
        """World point of projected point ``p`` (in -1..1) in panel ``box``."""
        fx, fy, w, h = box
        dx, dy = self.frac((w - 44) / 2, (h - 44) / 2)
        return self.P(float(fx + p[0] * dx), float(fy - p[1] * dy), lift)

    def _map(self, sc, layer, op, box=None, checked=(), best=None, look=None, live=False,
             reveal=1.0, title=True, raised=0, query=True, thumbs=True, mix=1.0):
        """Layer ``layer`` as a map: its links and points and the query;
        ``checked`` points in blue, the ``best`` so far in red, ``look`` =
        ``(src, node)`` the link being read now; points of level ``raised``
        and above stand out (0: none). ``mix``: 0 the flat projection, 1 the
        layout around the query."""
        box = self.BIG if box is None else box
        xy = self.flat * (1 - mix) + self.around * mix
        fx, fy, w, h = box
        self.frame(sc, fx, fy, w, h, op, border=AMBER if live else AXIS,
                   border_w=2.4 if live else 1.2)
        members = self.members[layer]
        if title:
            n = len(members)
            self.text(sc, fx - self.frac(w / 2)[0], fy - self.frac(0, h / 2)[1] - 0.028,
                      title if isinstance(title, str) else "layer %d: %s" % (
                          layer, "all %d digits and their links" % n if layer == 0
                          else "%d of the digits, linked" % n),
                      op, anchor="lm", strong=True, size=0.032)
        if query and mix >= 1.0:
            turn = np.linspace(0, 2 * np.pi, 49)
            for d in self.RINGS:
                ring = d / self.reach * np.c_[np.cos(turn), np.sin(turn)]
                self.polyline(sc, [self._at(box, q, 0.3) for q in ring], 1.2, AMBER, 0.4 * op)
        thumbs = thumbs and layer >= 2
        r = 2.5 if layer == 0 else 5.0
        shown = set(members[:int(np.ceil(len(members) * reveal))])
        for i, j in self.links[layer]:
            if i in shown and j in shown and mix in (0.0, 1.0):
                sc.line(self._at(box, xy[i], 0.5), self._at(box, xy[j], 0.5),
                        width=0.8 if layer == 0 else 1.6, color=AXIS,
                        opacity=(0.13 if layer == 0 else 0.45) * op, bias=0.1)
        if look is not None:
            sc.line(self._at(box, xy[look[0]], 0.8), self._at(box, xy[look[1]], 0.8),
                    width=3.0, color=BLUE, opacity=op, bias=0.3)
        checked = set(checked)
        for i in members:
            if i not in shown:
                continue
            at = self._at(box, xy[i])
            state = RED if i == best else BLUE if i in checked else None
            if thumbs:
                self.digit(sc, self.ix["images"][i], *self._fr(at), 26 if state is None else 30,
                           op, frame=AXIS if state is None else state)
            elif state is RED:
                self.disk(sc, at, r + 3.0, RED, op, lift=1.5, glow=0.3, outline=0.0, ring=WHITE,
                          ring_w=1.2)
            elif state is BLUE:
                self.disk(sc, at, r + 1.0, BLUE, op, lift=1.2, outline=0.1)
            elif raised and self.levels[i] >= raised:
                self.disk(sc, at, r + 2.2, VIOLET, op, lift=1.1, outline=0.1)
            else:
                self.disk(sc, at, r, (0.60, 0.62, 0.65), op, lift=1.0, outline=0.1)
        if look is not None and not thumbs:
            self.disk(sc, self._at(box, xy[look[1]]), r + 2.5, BLUE, op, lift=1.3,
                      glow=0.4, outline=0.0)
        if query:
            at = self._at(box, (0.0, 0.0))
            if thumbs:
                self.digit(sc, self.ix["queries"][self.QUERY], *self._fr(at), 30, op,
                           frame=AMBER)
            else:
                self.rect(sc, at, 2 * r + 8, 2 * r + 8, AMBER, op, border=WHITE, border_w=1.4,
                          radius=2.0, lift=1.8, glow=0.25)

    def _fr(self, p):
        """Layout fraction of world point ``p``."""
        x, y = self.fr(p)
        k = self.cam.zoom
        return 0.5 + (x - 0.5) / k, 0.5 + (y - 0.5) / k

    def _stack(self, sc, op, live=None, built=None, dropped=None, pulse=0.0):
        """The layers as a stack of boxes, the sparsest on top; ``live``: the
        layer the search is in; ``built``: how many layers exist yet (from the
        bottom); ``dropped``: arrows down to this layer are drawn."""
        x, y0, step = self.STACK
        self.text(sc, x, y0 - 0.085, "the layers", op, strong=True, size=0.032)
        for layer in range(self.top, -1, -1):
            a = op
            if built is not None:
                a *= seg(built, layer, layer + 0.6)
            y = y0 + (self.top - layer) * step
            on = live == layer
            ink = AMBER if on else VIOLET if live is None and layer else None
            self.box(sc, x, y, 100 + 14 * (self.top - layer), 40,
                     PAPER if ink is None else soft(ink, 0.6), a * (1.0 if on or live is None
                                                                    else 0.55),
                     "layer %d:  %d" % (layer, len(self.members[layer])),
                     border=AXIS if ink is None else ink, size=0.032)
            if dropped is not None and layer > dropped:
                self.arrow(sc, (x, y + 0.048), (x, y + step - 0.048), RED, op, width=2.0)
            if dropped is not None and layer == dropped + 1:
                self.pulse(sc, (x, y + 0.03), (x, y + step - 0.03), pulse, RED, op, r=6.0)

    def _readout(self, sc, op, best, n, d=None, head=None, found=None, note=None):
        """The right-hand column: the query image, the nearest found so far
        and the count of distances computed."""
        x = self.SIDE
        self.text(sc, x, 0.30, "searching for", op, anchor="lm", ink=dark(AMBER))
        self.digit(sc, self.ix["queries"][self.QUERY], x + 0.035, 0.385, 50, op, frame=AMBER)
        if head:
            self.text(sc, x + 0.08, 0.36, head[0], op, anchor="lm", strong=True)
            self.text(sc, x + 0.08, 0.41, head[1], op, anchor="lm", strong=True, size=0.036)
        self.text(sc, x, 0.485, "nearest so far" if found is None else found, op,
                  anchor="lm", ink=dark(RED))
        if best is not None:
            self.digit(sc, self.ix["images"][best], x + 0.035, 0.57, 50, op, frame=RED)
        if d is not None:
            self.text(sc, x + 0.08, 0.545, "distance", op, anchor="lm")
            self.text(sc, x + 0.08, 0.595, "%.2f" % np.sqrt(d), op, anchor="lm", strong=True,
                      size=0.036)
        if note:
            self.text(sc, x, 0.675, note[0], op, anchor="lm", ink=note[1])
        self.text(sc, x, 0.755, "distances computed", op, anchor="lm", ink=dark(BLUE))
        self.text(sc, x, 0.815, "%d of %d" % (n, len(self.flat)), op, anchor="lm",
                  strong=True, size=0.040)
        full = self.span(0.17)[0]
        self.hbar(sc, x, 0.875, full * n / len(self.flat), BLUE, op, thick=12.0, track=full)

    def _state(self, ef, upto):
        """The search of width ``ef`` after its first ``upto`` steps: the
        step, the points whose distance has been computed, and the squared
        distance of the nearest so far."""
        events = self.traces[ef][0]
        k = int(np.clip(upto, 0, len(events) - 1))
        e = events[k]
        checked = {ev["node"] for ev in events[:k + 1] if ev["kind"] in ("enter", "look", "hop")}
        d = float(np.sum((self.ix["u"][e["best"]] - self.ix["q"][self.QUERY]) ** 2))
        return e, checked, d

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        built = seg(u, 0.42, 0.9) * (self.top + 0.6)
        raised = 0
        for layer in range(1, self.top + 1):
            if built >= layer + 0.3:
                raised = layer
        self._map(sc, 0, op, reveal=seg(u, 0.0, 0.25), raised=raised, query=False, mix=0.0)
        self._stack(sc, op * smoothstep(0.36, 0.44, u), built=built + 1.0)
        # what a point is: three of the digits behind them
        x = self.SIDE
        self.text(sc, x, 0.30, "each point is a digit", op, anchor="lm", strong=True)
        for n, i in enumerate(self.members[2][:3]):
            y = 0.41 + n * 0.155
            a = op * smoothstep(0.05 + 0.06 * n, 0.15 + 0.06 * n, u)
            self.digit(sc, self.ix["images"][i], x + 0.085, y, 58, a, frame=AXIS)
            self.arrow(sc, (x + 0.04, y), self._fr(self._at(self.BIG, self.flat[i])), AXIS,
                       a * 0.8, seg(u, 0.1 + 0.06 * n, 0.25 + 0.06 * n), width=1.4)
        if raised:
            self.lines(sc, x, 0.84, [("in violet: also in", dark(VIOLET)),
                                     ("layer %d%s" % (raised, " and up" if raised < self.top
                                                      else ""), dark(VIOLET))], op, step=0.045)

    def phase_2(self, sc, u, op):
        events = self.traces[1][0]
        caption = "rings: true distance %s from the new digit, at the centre" % ", ".join(
            "%g" % d for d in self.RINGS)
        if u < 0.1:
            # the same points, moved to their true distance from the new digit
            mix = smoothstep(0.02, 0.085, u)
            self._map(sc, 0, op, mix=mix, query=mix >= 1.0,
                      title="the same %d points, placed around a new digit" % len(self.flat))
            self._stack(sc, op)
            self.text(sc, self.BIG[0], 0.925, caption, op * mix)
            self._readout(sc, op * smoothstep(0.0, 0.03, u), None, 0)
            return
        self.text(sc, self.BIG[0], 0.925, caption, op)
        t = min((u - 0.1) / 0.84, 1.0) * len(events)
        k = min(int(t), len(events) - 1)
        within = min(t - k, 1.0)
        e, checked, d = self._state(1, k)
        look = (e["src"], e["node"]) if e["kind"] in ("look", "hop") else None
        self._map(sc, e["layer"], op, checked=checked, best=e["best"], live=True, look=look)
        self._stack(sc, op, live=e["layer"], dropped=e["layer"],
                    pulse=within if e["kind"] == "drop" else 0.0)
        note = None
        if u > 0.94:
            note = ("no link is nearer: stop", dark(RED))
        elif e["kind"] == "hop":
            note = ("this one: %.2f, nearer" % np.sqrt(e["d"]), dark(RED))
        elif e["kind"] == "look":
            note = ("this one: %.2f, farther" % np.sqrt(e["d"]), dark(BLUE))
        elif e["kind"] == "drop":
            note = ("none nearer: drop a layer", dark(RED))
        self._readout(sc, op, e["best"], e["n"], d, head=("in layer", str(e["layer"])),
                      note=note)

    def phase_3(self, sc, u, op):
        n = len(self.WIDTHS)
        turn = min(int(u * n), n - 1)
        within = seg(u, turn / n, (turn + 1) / n)
        ef = self.WIDTHS[turn]
        events, answer, _count = self.traces[ef]
        e, checked, d = self._state(ef, int(min(within / 0.7, 1.0) * (len(events) - 1) + 1e-9))
        look = (e["src"], e["node"]) if e["layer"] == 0 and e["kind"] == "look" and \
            within < 0.7 else None
        self._map(sc, 0, op, checked=checked, best=e["best"], live=True, look=look)
        self._stack(sc, op, live=0, dropped=0)
        self.rect(sc, self._at(self.BIG, self.around[self.exact]), 19, 19, GREEN, op,
                  border=GREEN, border_w=2.4, radius=9.5, lift=1.7, fill_op=0.0)
        self.text(sc, self.BIG[0], 0.925, "circled in green: the true nearest of all %d" % len(
            self.flat), op, ink=dark(GREEN))
        verdict = note = None
        if within >= 0.7:
            hit = answer == self.exact
            verdict = "found" if hit else "found (not the nearest)"
            note = ("the true nearest", dark(GREEN)) if hit else ("a nearer one was missed",
                                                                  dark(RED))
        self._readout(sc, op, e["best"], e["n"], d, head=("width", str(ef)), found=verdict,
                      note=note)

    def phase_4(self, sc, u, op):
        # the same search at the width under the cursor
        grow, at = log_sweep(self.efs, seg(u, 0.1, 0.85))
        ef = self.efs[at]
        events, answer, count = self.traces[ef]
        _e, checked, _d = self._state(ef, len(events) - 1)
        self._map(sc, 0, op, box=(0.225, 0.615, 310.0, 230.0), checked=checked, best=answer,
                  title=False)
        self.text(sc, 0.05, 0.31, "width %d: this search computed %d" % (ef, count), op,
                  anchor="lm", strong=True, size=0.032)
        self.text(sc, 0.05, 0.895, "width: candidates kept open at once", op, anchor="lm")
        chart = self.chart(((0.52, 0.93), (0.41, 0.80)), self.efs, [0.0, 1.0], pad=0.04,
                           xlog=True)
        self.axes(sc, chart, op, "search width", "",
                  xticks=[(v, str(v)) for v in self.efs],
                  yticks=[(v, percent(v, 0)) for v in (0, 0.25, 0.5, 0.75, 1.0)])
        self.plot(sc, chart, self.efs, self.recall, GREEN, op, grow, width=2.6)
        self.plot(sc, chart, self.efs, self.work, BLUE, op, grow, width=2.6)
        self.text(sc, 0.52, 0.30, "true nearest found: %s of %d searches" % (
            percent(self.recall[at]), len(self.ix["q"])), op, anchor="lm", ink=dark(GREEN),
            size=0.032)
        self.text(sc, 0.52, 0.35, "distances computed: %s of the %d" % (
            percent(self.work[at]), len(self.flat)), op, anchor="lm", ink=dark(BLUE), size=0.032)


# ====================================================================== 83, 84
TEMPERATURE_INKS = (BLUE, GREEN, RED)      # the three temperatures drawn: 0.01, 0.1, 2


def encoder(arrays, name):
    """One of the encoders a contrastive study saved."""
    return {k: arrays["parameter_%s_%s" % (name, k)] for k in ("W1", "b1", "W2", "b2")}


def view_pairs(weights, digits=(0, 1, 2, 3, 4, 6, 7, 9), seed=17):
    """Two views (the studies' own: shifted, pooled, noise added) of the first
    test image of each of ``digits``, and the cosine of every first view with
    every second under the encoder ``weights``: ``(ids, a, b, za, zb, sim)``."""
    d = SR.digits()
    ids = np.array([int(np.flatnonzero(d["yt"] == k)[0]) for k in digits])
    rng = np.random.default_rng(seed)
    a, b = SR.views(d["xt"][ids], rng), SR.views(d["xt"][ids], rng)
    za, zb = SR.embed(weights, a).astype(float), SR.embed(weights, b).astype(float)
    return ids, a, b, za, zb, za @ zb.T


def shares(sim, temperature):
    """Each row of similarities as InfoNCE turns it into probabilities."""
    return SR.softmax(np.asarray(sim, float) / temperature)


def push(sim, temperature, row):
    """Of the push on the wrong matches of ``row`` (their shares), the part
    that lands on the most similar one."""
    p = np.delete(shares(sim, temperature)[row], row)
    return float(p.max() / max(p.sum(), 1e-300))


def pair_cosines(weights, bins):
    """How alike the encoder ``weights`` finds two test digits: the cosines
    of every pair of the study's test embeddings, as two histograms over
    -1..1 (each summing to 1): pairs of different digits, pairs of the same
    digit. A flat picture of 32 dimensions cannot show crowding (a cloud
    spread over many directions flattens to a blob at the centre); the
    cosines can."""
    d = SR.digits()
    z = SR.embed(weights, SR.pool(d["xt"])).astype(float)
    c = z @ z.T
    same = d["yt"][:, None] == d["yt"][None]
    other = ~np.eye(len(z), dtype=bool)
    return tuple(np.histogram(c[m], bins=bins, range=(-1, 1))[0] / m.sum()
                 for m in (~same, same & other))


class InfoNCETemperature(_Kit):
    """The temperature of the contrastive loss: eight real pairs of views
    scored against each other, the scores turned into shares at a sliding
    temperature, then the encoders the study trained at eight temperatures:
    where they put the test digits, how spread and how aligned, and how well
    the nearest neighbour names a held-out digit."""

    TITLE = "Temperature in contrastive learning"
    PHASES = (("pairs", 10.0), ("temperature", 14.0), ("embeddings", 10.0),
              ("sweep", 11.0), ("held out", 10.0))
    SHOWN = (0.01, 0.1, 2.0)             # the temperatures whose encoders are drawn
    ROW = 3                              # the row followed: the view of a 3
    MAT = (0.775, 0.635, 224.0)          # the 8 x 8 matrix: x, y, side px
    BINS = 20                            # of cosine, from -1 to 1

    def setup(self):
        s = self.study
        settings, arrays = evidence(s)
        d = SR.digits()
        self.sweep = [float(t) for t in settings["temperatures"]]
        self.at = [self.sweep.index(t) for t in self.SHOWN]
        self.temps = list(self.SHOWN)
        self.updates, self.n_seeds = int(settings["updates"]), len(settings["seeds"])
        self.encoders = [encoder(arrays, "t%d" % i) for i in self.at]
        # mean, lowest and highest seed at each temperature of the sweep
        self.align, self.unif, self.acc = (np.asarray(s.panels[k].data["ys"], float)
                                           for k in (1, 2, 3))
        self.seeds = {k: np.asarray(s.metadata[k], float)       # temperature, seed
                      for k in ("accuracy", "alignment", "uniformity")}
        self.best = int(np.argmax(self.acc[0]))
        self.untrained = float(s.metadata["untrained"]["accuracy"])
        self.pixels = float(s.metadata["pixel_accuracy"])
        self.ids, self.a, self.b, self.za, self.zb, self.sim = view_pairs(self.encoders[1])
        self.images = d["xt"][self.ids]
        off = ~np.eye(len(self.ids), dtype=bool)
        self.partner, self.other = float(np.diag(self.sim).mean()), float(self.sim[off].mean())
        self.tests = len(d["yt"])
        self.pairs = [pair_cosines(w, self.BINS) for w in self.encoders]

    def share(self, temperature):
        return float(shares(self.sim, temperature)[self.ROW, self.ROW])

    def subtitle_for(self, k):
        t, sweep = self.temps, self.sweep
        lo, mid, hi = self.at
        spread = int(np.argmin(self.unif[0]))
        gap = 100 * float((self.acc[2] - self.acc[1]).max())
        ends = "falling to %s at %g and %s at %g" % (
            percent(self.acc[0, 0]), sweep[0], percent(self.acc[0, -1]), sweep[-1]) \
            if 0 < self.best < len(sweep) - 1 else "against %s at %g" % (
                percent(self.acc[0, 0 if self.best else -1]), sweep[0 if self.best else -1])
        return [
            "eight test digits, each seen twice: shifted a little, with fresh noise; an "
            "encoder turns every view into 32 numbers and each first view is scored against "
            "each second: partners average %.2f, the rest %.2f" % (self.partner, self.other),
            "each score is divided by the temperature, then the row is shared out. Of the "
            "push on this 3's wrong matches, the closest look-alike takes %s at temperature "
            "%g, %s at %g and %s at %g" % (
                percent(push(self.sim, t[2], self.ROW), 0), t[2],
                percent(push(self.sim, t[1], self.ROW), 0), t[1],
                percent(push(self.sim, t[0], self.ROW), 0), t[0]),
            "an encoder trained %s updates at each temperature, and how alike it finds two "
            "of %s test digits: crowding (lower is more spread) ends at %.2f at temperature "
            "%g, %.2f at %g and %.2f at %g" % (
                f"{self.updates:,}", f"{self.tests:,}", self.unif[0, lo], t[0],
                self.unif[0, mid], t[1], self.unif[0, hi], t[2]),
            "eight temperatures, %d seeds each: the test digits are most spread at %g "
            "(crowding %.2f) and more crowded towards both ends, %.2f at %g and %.2f at %g%s" % (
                self.n_seeds, sweep[spread], self.unif[0, spread], self.unif[0, 0], sweep[0],
                self.unif[0, -1], sweep[-1],
                "; there, two views of a digit also sit closest" if max(
                    self.align[0, 0], self.align[0, -1]) < self.align[0, spread] else ""),
            "naming each test digit by its nearest training digit is right %s of the time at "
            "temperature %g, the best, %s; the %d seeds are never more than %.1f points "
            "apart" % (percent(self.acc[0, self.best]), sweep[self.best], ends, self.n_seeds,
                       gap),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _matrix(self, sc, op, values, title, reveal=1.0, hot=None, vmax=1.0, ring=()):
        mx, my, side = self.MAT
        n = len(self.ids)
        cell = self.grid(sc, values, mx, my, side, side, op, reveal=reveal, vmax=vmax,
                         ring=ring, ring_ink=GREEN, hot=TEAL if hot is None else hot,
                         cold=OCHRE)
        dx, dy = self.frac(side / 2, side / 2)
        self.text(sc, mx - dx - 0.045, my - dy - 0.092, title, op, anchor="lm", strong=True)
        for j in range(n):
            self.digit(sc, self.images[j], cell(0, j)[0], my - dy - 0.038, 24, op)
            self.digit(sc, self.images[j], mx - dx - 0.025, cell(j, 0)[1], 24, op)
        return cell

    def _row_bars(self, sc, fx, fy, p, ink, op, step, width, height, values=False):
        """One row of shares as vertical bars standing on ``fy``."""
        for j, v in enumerate(p):
            on = j == self.ROW
            self.vbar(sc, fx + j * step, fy, height * v, width, ink if on else soft(ink, 0.45),
                      op, percent(v, 0) if values else "")

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        n = len(self.ids)
        t = min(u / 0.9, 1.0) * n
        turn = min(int(t), n - 1)
        within = min(t - turn, 1.0)
        show = smoothstep(0.0, 0.15, within) * (1 - smoothstep(0.9, 1.0, within)) \
            if u < 0.9 else 1.0
        a = op * show
        # one digit, its two views, their two vectors
        self.text(sc, 0.05, 0.305, "a test digit", op, anchor="lm", strong=True)
        self.digit(sc, self.images[turn], 0.095, 0.51, 70, a, frame=AXIS)
        peak = float(np.abs(np.r_[self.za, self.zb]).max())
        for row, (views, z, name, y) in enumerate(((self.a, self.za, "first view", 0.40),
                                                   (self.b, self.zb, "second view", 0.62))):
            self.arrow(sc, (0.142, 0.51), (0.19, y), AXIS, a, width=1.6)
            self.digit(sc, views[turn], 0.228, y, 54, a, frame=AXIS)
            self.text(sc, 0.228, y - 0.085 if row == 0 else y + 0.085, name, op)
            self.arrow(sc, (0.265, y), (0.295, 0.51 + (y - 0.51) * 0.25), AXIS, a, width=1.6)
            self.arrow(sc, (0.385, 0.51 + (y - 0.51) * 0.25), (0.41, y), AXIS, a, width=1.6)
            self.grid(sc, z[turn][None, :], 0.485, y, 128, 16, a, vmax=peak, hot=TEAL,
                      cold=OCHRE)
            self.text(sc, 0.485, y - 0.05 if row == 0 else y + 0.05, "32 numbers", op)
        self.box(sc, 0.34, 0.51, 76, 40, INKY, op, "encoder", ink=WHITE, size=0.030)
        self.text(sc, 0.485, 0.51, "cosine %.2f" % self.sim[turn, turn], a, strong=True,
                  size=0.034)
        self.lines(sc, 0.05, 0.82, ["a view: moved up to 3 pixels, shrunk, noise added",
                                    "partners: the two views of one digit"], op, step=0.05)
        # every first view against every second view
        cell = self._matrix(sc, op, self.sim, "each first view against each second",
                            reveal=(turn + smoothstep(0.3, 0.7, within)) / n if u < 0.9 else 1.0,
                            ring=[(j, j) for j in range(turn + 1 if u < 0.9 else n)])
        if u < 0.9:
            self.pulse(sc, (0.56, 0.51), cell(turn, 0), (within - 0.25) / 0.5, GREEN, op, 0.04)

    def _tau(self, u):
        """The temperature at ``u`` of the phase: held at the three drawn
        values, sliding between them."""
        hi, mid, lo = np.log(self.temps[2]), np.log(self.temps[1]), np.log(self.temps[0])
        if u < 0.5:
            return float(np.exp(hi + (mid - hi) * smoothstep(0.18, 0.38, u)))
        return float(np.exp(mid + (lo - mid) * smoothstep(0.58, 0.78, u)))

    def _ink(self, tau):
        t = self.temps
        return TEMPERATURE_INKS[0 if tau < np.sqrt(t[0] * t[1]) else 1 if tau < np.sqrt(
            t[1] * t[2]) else 2]

    def phase_2(self, sc, u, op):
        tau = self._tau(u)
        ink = self._ink(tau)
        p = shares(self.sim, tau)
        # the temperature, as a slider
        x0, x1, y = 0.25, 0.54, 0.325
        self.text(sc, 0.05, y, "temperature %.2f" % tau, op, anchor="lm", strong=True, size=0.036,
                  ink=dark(ink))
        self.polyline(sc, [self.P(x0, y), self.P(x1, y)], 3.0, AXIS, 0.45 * op)
        lo, hi = np.log(self.temps[0]), np.log(self.temps[2])
        for t, c in zip(self.temps, TEMPERATURE_INKS):
            fx = x0 + (x1 - x0) * (hi - np.log(t)) / (hi - lo)
            self.disk(sc, self.P(fx, y), 4.0, c, op, lift=1.0, outline=0.0)
            self.text(sc, fx, y + 0.045, "%g" % t, op, ink=dark(c))
        fx = x0 + (x1 - x0) * (hi - np.log(tau)) / (hi - lo)
        self.disk(sc, self.P(float(fx), y), 9.0, ink, op, lift=1.4, glow=0.3, outline=0.0,
                  ring=WHITE, ring_w=1.5)
        # one row of the matrix, as bars
        self.text(sc, 0.05, 0.435, "how the first view of the 3 shares 100%", op, anchor="lm")
        self._row_bars(sc, 0.09, 0.775, p[self.ROW], ink, op, 0.06, 34.0, 112.0, values=True)
        for j in range(len(self.ids)):
            self.digit(sc, self.images[j], 0.09 + j * 0.06, 0.82, 26, op,
                       frame=GREEN if j == self.ROW else None)
        loss = float(-np.log(np.diag(p)).mean())
        self.text(sc, 0.05, 0.885, "of the push on the wrong ones, the closest takes %s" %
                  percent(push(self.sim, tau, self.ROW), 0), op, anchor="lm", strong=True)
        self.text(sc, 0.05, 0.932, "loss, the mean of minus log partner's share: %.2f" % loss,
                  op, anchor="lm")
        self._matrix(sc, op, p, "every row shared out the same way", hot=ink,
                     ring=[(self.ROW, self.ROW)])

    def phase_3(self, sc, u, op):
        self.text(sc, 0.05, 0.30, "how alike each encoder finds two test digits: the cosine "
                  "of their 32 numbers", op, anchor="lm", strong=True)
        wide, tall = 216.0, 118.0
        peak = max(float(h.max()) for pair in self.pairs for h in pair)
        cell = wide / self.BINS
        for i, (t, ink, (apart, same)) in enumerate(zip(self.temps, TEMPERATURE_INKS,
                                                        self.pairs)):
            a = op * smoothstep(i * 0.12, i * 0.12 + 0.15, u)
            grow = smoothstep(0.0, 1.0, seg(u, i * 0.12 + 0.05, i * 0.12 + 0.35))
            fx, base = 0.20 + i * 0.30, 0.685
            left = self.P(fx, base)
            self.text(sc, fx, 0.355, "temperature %g" % t, a, ink=dark(ink), size=0.034)
            for b, v in enumerate(apart):
                h = max(tall * grow * v / peak, 0.6)
                self.rect(sc, self.shift(left, (b + 0.5 - self.BINS / 2) * cell, -h / 2),
                          cell - 1.5, h, ink, a, radius=min(2.0, h / 2))
            self.polyline(sc, [self.shift(left, (b + 0.5 - self.BINS / 2) * cell,
                                          -tall * grow * v / peak)
                               for b, v in enumerate(same)], 2.6, INKY, a, bias=0.4)
            self.polyline(sc, [self.shift(left, -wide / 2, 0), self.shift(left, wide / 2, 0)],
                          1.4, AXIS, a)
            for v in (-1, 0, 1):
                self.text(sc, fx + v * self.frac(wide / 2)[0], base + 0.035, "%d" % v, a)
            b = a * smoothstep(0.45, 0.6, u)
            j = self.at[i]
            self.text(sc, fx, 0.775, "crowding %.2f" % self.unif[0, j], b, strong=True)
            self.text(sc, fx, 0.818, "above 0.5: %s of the bars" % percent(
                apart[self.BINS * 3 // 4:].sum(), 0), b)
            self.text(sc, fx, 0.861, "partners %.2f apart" % self.align[0, j], b)
        self.text(sc, 0.05, 0.915, "bars: two different digits.  Line: two of the same digit.  "
                  "Apart: squared distance of two views", op, anchor="lm")

    def _swept(self, sc, op, box, key, mean, ink, grow, reached, yticks=None):
        """One measure against temperature: every seed a point, their mean a
        line, drawn as far as the cursor."""
        chart = self.chart(box, self.sweep, self.seeds[key], pad=0.1, xlog=True)
        self.axes(sc, chart, op, "temperature", "", xticks=[(t, "%g" % t) for t in self.sweep],
                  yticks=yticks)
        for j, row in enumerate(self.seeds[key][:reached + 1]):
            for v in row:
                self.disk(sc, self.P(chart.fx(self.sweep[j]), chart.fy(float(v))), 3.2,
                          soft(ink, 0.25), op, lift=1.2, outline=0.15)
        self.plot(sc, chart, self.sweep, mean, ink, op, grow, width=2.4)
        return chart

    def phase_4(self, sc, u, op):
        grow, reached = log_sweep(self.sweep, seg(u, 0.08, 0.8))
        t = self.sweep[reached]
        self.text(sc, 0.05, 0.305, "crowding of the test digits", op, anchor="lm", strong=True,
                  ink=dark(VIOLET))
        self.text(sc, 0.05, 0.35, "lower: spread more evenly over the sphere", op, anchor="lm")
        self._swept(sc, op, ((0.10, 0.47), (0.41, 0.80)), "uniformity", self.unif[0], VIOLET,
                    grow, reached)
        self.text(sc, 0.53, 0.305, "two views of one digit: how far apart", op, anchor="lm",
                  strong=True, ink=dark(TEAL))
        self.text(sc, 0.53, 0.35, "lower: partners pulled closer", op, anchor="lm")
        self._swept(sc, op, ((0.58, 0.95), (0.41, 0.80)), "alignment", self.align[0], TEAL,
                    grow, reached)
        self.text(sc, 0.05, 0.93, "temperature %g:  crowding %.2f,  squared distance between "
                  "views %.2f   (points: the %d seeds)" % (
                      t, self.unif[0, reached], self.align[0, reached], self.n_seeds), op,
                  anchor="lm", strong=True)

    def phase_5(self, sc, u, op):
        grow, reached = log_sweep(self.sweep, seg(u, 0.05, 0.7))
        self.text(sc, 0.05, 0.305, "test digits whose nearest training digit is the same "
                  "digit", op, anchor="lm", strong=True)
        lo = min(float(self.seeds["accuracy"].min()), self.untrained)
        ticks = [v / 100 for v in range(int(np.floor(lo * 20) * 5), 101, 5)
                 if v / 100 <= float(self.seeds["accuracy"].max()) + 0.04]
        chart = self._swept(sc, op, ((0.10, 0.55), (0.37, 0.80)), "accuracy", self.acc[0], GREEN,
                            grow, reached, yticks=[(v, percent(v, 0)) for v in ticks])
        y = chart.fy(self.untrained)
        self.polyline(sc, [self.P(chart.x0, y), self.P(chart.x1, y)], 1.6, AXIS, 0.8 * op)
        self.text(sc, 0.10, 0.93, "grey line: the encoder before any training, %s" %
                  percent(self.untrained), op, anchor="lm")
        self.text(sc, 0.64, 0.355, "temperature", op, anchor="lm", strong=True)
        self.text(sc, 0.80, 0.355, "mean", op, anchor="rm", strong=True)
        self.text(sc, 0.95, 0.355, "seeds", op, anchor="rm", strong=True)
        for n, t in enumerate(self.sweep):
            a = op * (1.0 if n <= reached else 0.0)
            y = 0.415 + n * 0.058
            ink = dark(GREEN) if n == self.best else None
            self.text(sc, 0.64, y, "%g" % t, a, anchor="lm", ink=ink)
            self.text(sc, 0.80, y, percent(self.acc[0, n]), a, anchor="rm", ink=ink,
                      strong=ink is None)
            self.text(sc, 0.95, y, "%.0f to %.0f" % (100 * self.acc[1, n], 100 * self.acc[2, n]),
                      a, anchor="rm", ink=ink)
        self.text(sc, 0.95, 0.93, "by raw pixels: %s" % percent(self.pixels), op, anchor="rm")


# ====================================================================== 84
class HardNegatives(_Kit):
    """Which other images a contrastive loss pushes an anchor away from:
    eight drawn by lot, or the eight the encoder itself finds most similar.
    The same batch mined before and after training, how many of the mined
    are really the anchor's own digit, and the encoders that result."""

    TITLE = "Hard negatives"
    PHASES = (("a batch", 10.0), ("mine the hardest", 11.0), ("false negatives", 12.0),
              ("training", 10.0), ("held out", 11.0))
    ANCHOR = 0                           # the anchor's place in the batch
    SHOWN = 12                           # of its ranking, the first so many are drawn
    INKS = (BLUE, RED, AMBER, GREEN)     # by lot, hardest, after skipping, labels used
    XS = (0.31, 0.056)                   # the ranked row: first x, step

    def setup(self):
        s = self.study
        settings, arrays = evidence(s)
        d = SR.digits()
        ix, self.a, self.b = SR.first_batch(int(settings["seeds"][0]))
        self.labels, self.images = d["y"][ix], d["x"][ix]
        self.digit_of = int(self.labels[self.ANCHOR])
        self.updates, self.k, self.skip = (int(settings[k]) for k in ("updates", "negatives",
                                                                      "skip"))
        self.n_seeds = len(settings["seeds"])
        self.modes = list(settings["modes"])
        self.names = ["%d by lot" % self.k, "the %d most similar" % self.k,
                      "most similar after skipping %d" % self.skip,
                      "most similar of other digits"]
        # the first batch as the encoder ranked it before training and after
        self.rank, self.cos, self.mined_false = {}, {}, {}
        for at in (0, self.updates):
            w = encoder(arrays, "hardest%d" % at)
            u, v = SR.embed(w, self.a), SR.embed(w, self.b)
            row = (u @ v.T).astype(float)[self.ANCHOR]
            row[self.ANCHOR] = -np.inf
            order = np.argsort(-row, kind="stable")[:self.SHOWN]
            self.rank[at], self.cos[at] = [int(j) for j in order], row[order]
            self.mined_false[at] = SR.choose(u, v, "hardest", None, self.labels)[1]
            if at == 0:
                lots = np.random.default_rng(10_000 + int(settings["seeds"][0]))
                mask = SR.choose(u, v, "random", lots, self.labels)[0][self.ANCHOR]
                self.lot = [int(j) for j in np.flatnonzero(mask) if j != self.ANCHOR]
        self.same = {at: [int(self.labels[j]) == self.digit_of for j in order]
                     for at, order in self.rank.items()}
        # the measurements
        self.windows = np.asarray(s.panels[0].data["x"], float)
        self.false = np.asarray(s.panels[0].data["ys"], float)           # mode, window
        share = s.metadata["false_negative_share"]
        self.first = [float(np.mean(share["first_100_updates"][m])) for m in self.modes]
        self.last = [float(np.mean(share["last_100_updates"][m])) for m in self.modes]
        self.steps = np.asarray(s.panels[1].data["x"], float)
        self.trace = np.asarray(s.panels[1].data["ys"], float)           # mode, time
        self.seeds = np.asarray(s.panels[2].data["values"], float)       # mode, seed
        self.acc = self.seeds.mean(1)
        self.tests = int(settings["test"])

    def verdict(self):
        """What the accuracies say, in words that hold whichever way they fell."""
        lot, hard, _skip, clean = self.seeds
        if hard.min() > lot.max():
            mined = "mining beats the lot in every seed"
        elif hard.mean() > lot.mean():
            mined = "mining beats the lot on average, not in every seed"
        else:
            mined = "mining did not beat the lot"
        if clean.min() > hard.max():
            return mined + ", and its false negatives hold it back"
        return mined + ", and removing its false negatives changed little"

    def subtitle_for(self, k):
        n = len(self.labels) - 1
        return [
            "contrastive training pulls an image towards its own second view and pushes it "
            "from other images in the batch: here from %d of the other %d, and the plain way "
            "to pick them is by lot" % (self.k, n),
            "hard negatives are mined, not labelled: the encoder scores the anchor against the "
            "rest of the batch and the %d it finds most similar become the negatives, the "
            "look-alikes it has not yet told apart" % self.k,
            "but the most similar images are often the anchor's own digit, false negatives "
            "pushed apart wrongly: %s of the mined in the first 100 updates and %s in the last "
            "100, against %s of those drawn by lot" % (
                percent(self.first[1], 0), percent(self.last[1], 0), percent(self.last[0], 0)),
            "four encoders per seed, same images in the same order, %s updates: the nearest "
            "training digit names a test digit right %s of the time by lot, %s mined, %s mined "
            "after skipping the top %d" % (
                f"{self.updates:,}", percent(self.acc[0]), percent(self.acc[1]),
                percent(self.acc[2]), self.skip),
            "mean of %d seeds: %s by lot, %s mined, and %s when the labels are used to take "
            "the anchor's own digit out of the mined: %s" % (
                self.n_seeds, percent(self.acc[0]), percent(self.acc[1]), percent(self.acc[3]),
                self.verdict()),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _anchor(self, sc, op):
        self.text(sc, 0.10, 0.305, "anchor", op, strong=True)
        self.digit(sc, self.a[self.ANCHOR], 0.10, 0.40, 62, op, frame=INKY)
        self.text(sc, 0.215, 0.305, "its pair", op, strong=True, ink=dark(GREEN))
        self.digit(sc, self.b[self.ANCHOR], 0.215, 0.40, 62, op, frame=GREEN)
        self.arrow(sc, (0.142, 0.40), (0.173, 0.40), GREEN, op, width=2.4)

    def _ranked(self, sc, at, fy, size, op, shown=None, taken=0.0, truth=0.0, x0=None,
                step=None, count=None):
        """The rest of the batch as the encoder of update ``at`` ranks it
        against the anchor, most similar first; ``taken`` of the first
        :attr:`k` framed as its negatives; ``truth`` shows which are the
        anchor's own digit."""
        x0 = self.XS[0] if x0 is None else x0
        step = self.XS[1] if step is None else step
        for n, j in enumerate(self.rank[at][:count]):
            a = op if shown is None else op * seg(shown, n, n + 1)
            frame = AXIS
            if n < self.k and taken > n / self.k:
                frame = GREEN if truth > 0.5 and self.same[at][n] else RED
            self.digit(sc, self.b[j], x0 + n * step, fy, size, a * (1.0 if n < self.k or
                                                                    taken < 1.0 else 0.4),
                       frame=frame)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        self._anchor(sc, op)
        self.lines(sc, 0.05, 0.495, [("one digit seen twice:", dark(GREEN)),
                                     ("pulled together", dark(GREEN))], op, step=0.042)
        n = len(self.labels)
        self.text(sc, 0.31, 0.305, "pushed away from: %d of the other %d" % (self.k, n - 1),
                  op, anchor="lm", strong=True)
        drawn = seg(u, 0.45, 0.85) * len(self.lot)
        for m, j in enumerate(self.lot):
            self.digit(sc, self.b[j], 0.335 + m * 0.068, 0.40, 50, op * seg(drawn, m, m + 1),
                       frame=BLUE)
        self.text(sc, 0.31, 0.495, "random negatives: the %d are drawn by lot" % self.k,
                  op * smoothstep(0.45, 0.55, u), anchor="lm", ink=dark(BLUE))
        # the whole batch the anchor sits in
        a = op * smoothstep(0.0, 0.1, u)
        self.text(sc, 0.05, 0.585, "the batch: %d training digits" % n, a, anchor="lm",
                  strong=True)
        self.text(sc, 0.95, 0.585, "blue: drawn by lot for this anchor", a, anchor="rm",
                  ink=dark(BLUE))
        for j in range(n):
            m = self.lot.index(j) if j in self.lot else -1
            frame = INKY if j == self.ANCHOR else BLUE if m >= 0 and drawn > m else None
            self.digit(sc, self.images[j], 0.064 + (j % 32) * 0.0281, 0.655 + (j // 32) * 0.073,
                       22, a * seg(u * 6, j / n, j / n + 0.3), frame=frame)

    def phase_2(self, sc, u, op):
        self._anchor(sc, op)
        x0, step = self.XS
        self.text(sc, x0 - 0.027, 0.305, "the rest of the batch, most like the anchor first",
                  op, anchor="lm", strong=True)
        shown = seg(u, 0.05, 0.4) * self.SHOWN
        taken = seg(u, 0.5, 0.75)
        self._ranked(sc, 0, 0.40, 44, op, shown=shown, taken=taken)
        self.text(sc, 0.05, 0.60, "its cosine with the anchor,", op, anchor="lm")
        self.text(sc, 0.05, 0.645, "by the encoder as it is now", op, anchor="lm")
        for n, c in enumerate(self.cos[0]):
            grow = seg(shown, n, n + 1)
            ink = RED if n < self.k and taken > n / self.k else soft(RED, 0.5)
            self.vbar(sc, x0 + n * step, 0.66, 78.0 * c * smoothstep(0.0, 1.0, grow), 30.0, ink,
                      op, "%.2f" % c if grow > 0.5 else "")
        a = op * smoothstep(0.5, 0.6, u)
        end = x0 + (self.k - 1) * step
        self.polyline(sc, [self.P(x0 - 0.022, 0.70), self.P(end + 0.022, 0.70)], 3.0, RED, a)
        self.text(sc, (x0 + end) / 2, 0.745, "the %d most similar: its hard negatives" % self.k,
                  a, ink=dark(RED), size=0.032)
        self.lines(sc, 0.05, 0.84, ["no label is consulted: the encoder's own similarity does "
                                    "the choosing,", "afresh at every update, as it would in "
                                    "practice"], op * smoothstep(0.7, 0.8, u), step=0.05)

    def phase_3(self, sc, u, op):
        late = self.updates
        truth = smoothstep(0.12, 0.2, u)
        x0, step = 0.335, 0.049
        self.text(sc, 0.05, 0.305, "anchor: a %d" % self.digit_of, op, anchor="lm", strong=True)
        self.digit(sc, self.a[self.ANCHOR], 0.10, 0.435, 62, op, frame=INKY)
        self.text(sc, x0 - 0.022, 0.305, "its %d mined negatives; green: also a %d" % (
            self.k, self.digit_of), op, anchor="lm", strong=True)
        for row, (at, name) in enumerate(((0, "before training"),
                                          (late, "after %s updates" % f"{late:,}"))):
            y = 0.385 + row * 0.115
            a = op * (1.0 if row == 0 else smoothstep(0.25, 0.35, u))
            self.text(sc, x0 - 0.03, y, name, a, anchor="rm", ink=dark(RED))
            self._ranked(sc, at, y, 36, a, taken=1.0, truth=truth, x0=x0, step=step,
                         count=self.k)
            self.text(sc, x0 + self.k * step - 0.01, y, "%d of %d; this batch %s" % (
                sum(self.same[at][:self.k]), self.k, percent(self.mined_false[at], 0)),
                a * truth, anchor="lm", strong=True, ink=dark(GREEN))
        # the share over the whole run, every anchor of every batch
        shown = (1, 2, 0)
        ys = self.false[list(shown)]
        chart = self.chart(((0.10, 0.50), (0.655, 0.86)), np.r_[0.0, self.windows],
                           [0.0, float(ys.max())], pad=0.08)
        b = op * smoothstep(0.35, 0.45, u)
        self.text(sc, 0.05, 0.60, "mined negatives that are the anchor's own digit, all "
                  "batches", b, anchor="lm", strong=True)
        top = float(np.ceil(ys.max() * 10) / 10)
        self.axes(sc, chart, b, "training update", "",
                  yticks=[(v, percent(v, 0)) for v in np.arange(0.0, top + 0.01, 0.2)])
        grow = seg(u, 0.45, 0.85)
        for m in shown:
            self.plot(sc, chart, self.windows, self.false[m], self.INKS[m], b,
                      grow, width=2.4)
        for n, m in enumerate(shown):
            y = 0.69 + n * 0.062
            self.text(sc, 0.56, y, self.names[m], b, anchor="lm", ink=dark(self.INKS[m]))
            self.text(sc, 0.95, y, "%s, then %s" % (percent(self.first[m], 0),
                                                    percent(self.last[m], 0)), b, anchor="rm",
                      strong=True)
        self.text(sc, 0.56, 0.69 + 3 * 0.062, "first and last 100 updates, %d seeds" %
                  self.n_seeds, b, anchor="lm")

    def phase_4(self, sc, u, op):
        chart = self.chart(((0.10, 0.54), (0.36, 0.82)), self.steps, self.trace, pad=0.06)
        self.text(sc, 0.05, 0.305, "test digits named right by their nearest training digit",
                  op, anchor="lm", strong=True)
        lo, hi = float(self.trace.min()), float(self.trace.max())
        self.axes(sc, chart, op, "training update", "",
                  yticks=[(v / 100, "%d%%" % v) for v in range(0, 101, 5)
                          if lo - 0.01 <= v / 100 <= hi + 0.01])
        grow = u / 0.8
        for trace, ink in zip(self.trace, self.INKS):
            self.plot(sc, chart, self.steps, trace, ink, op, grow, width=2.4)
        now = int(min(grow, 1.0) * (len(self.steps) - 1))
        self.text(sc, 0.60, 0.37, "negatives for each anchor", op, anchor="lm", strong=True)
        for m, ink in enumerate(self.INKS):
            y = 0.445 + m * 0.095
            self.text(sc, 0.60, y, self.names[m], op, anchor="lm", ink=dark(ink))
            self.text(sc, 0.95, y, percent(self.trace[m, now]), op, anchor="rm", strong=True,
                      size=0.034)
            if m == 3:
                self.text(sc, 0.60, y + 0.042, "(a control: it needs the labels)", op,
                          anchor="lm")
        self.text(sc, 0.60, 0.895, "update %d of %d, mean of %d seeds" % (
            self.steps[now], self.steps[-1], self.n_seeds), op, anchor="lm")

    def phase_5(self, sc, u, op):
        self.text(sc, 0.05, 0.305, "held-out digits named right, after %s updates: each seed "
                  "a point, the tick their mean" % f"{self.updates:,}", op, anchor="lm",
                  strong=True)
        lo = np.floor(float(self.seeds.min()) * 20) / 20
        hi = np.ceil(float(self.seeds.max()) * 20) / 20
        x0, x1 = 0.40, 0.86

        def at(v):
            return x0 + (x1 - x0) * (float(v) - lo) / (hi - lo)
        ticks = np.arange(lo, hi + 1e-9, 0.05)
        for v in ticks:
            self.polyline(sc, [self.P(at(v), 0.36), self.P(at(v), 0.815)], 1.0, GRID, 0.6 * op)
            self.text(sc, at(v), 0.845, percent(v, 0), op)
        for m, (ink, name) in enumerate(zip(self.INKS, self.names)):
            y = 0.415 + m * 0.108
            a = op * smoothstep(0.05 + 0.15 * m, 0.15 + 0.15 * m, u)
            grow = smoothstep(0.0, 1.0, seg(u, 0.1 + 0.15 * m, 0.35 + 0.15 * m))
            self.text(sc, x0 - 0.02, y - (0.02 if m == 3 else 0.0), name, a, anchor="rm",
                      ink=dark(ink))
            if m == 3:
                self.text(sc, x0 - 0.02, y + 0.024, "(control: uses labels)", a, anchor="rm")
            self.polyline(sc, [self.P(at(self.seeds[m].min()), y),
                               self.P(at(self.seeds[m].max()), y)], 3.0, soft(ink, 0.3), a * grow)
            for v in self.seeds[m]:
                self.disk(sc, self.P(at(v), y), 5.5, ink, a * grow, lift=1.2, outline=0.15,
                          ring=WHITE, ring_w=1.0)
            self.rect(sc, self.P(at(self.acc[m]), y), 4.0, 30.0, dark(ink), a * grow, radius=1.5,
                      lift=1.6)
            self.text(sc, 0.95, y, percent(self.acc[m]), a * grow, anchor="rm", strong=True,
                      size=0.034)
        gain, more = self.acc[1] - self.acc[0], self.acc[3] - self.acc[1]
        self.text(sc, 0.05, 0.915, "mined against by lot: %+.1f points.  The anchor's own digit "
                  "taken out of the mined: %+.1f more" % (100 * gain, 100 * more),
                  op * smoothstep(0.7, 0.82, u), anchor="lm", strong=True)


# ====================================================================== 85
def neighbours(w, x, xt):
    """Index of the training image nearest (by cosine) each test image under
    encoder ``w``, as the study's accuracy takes it."""
    return (unit(xt @ w) @ unit(x @ w).T).argmax(1)


class DimensionalityOfEmbeddings(_Kit):
    """One digit encoded at eight widths, what a search retrieves at each,
    the accuracy against width, and one wide vector cut short."""

    TITLE = "How wide should an embedding be?"
    PHASES = (("as numbers", 11.0), ("search", 15.0), ("accuracy", 10.0), ("cut short", 15.0))
    SHOWN = 2                            # the test digit whose vectors are drawn
    QUERIES = 8
    ROWS = (2, 4, 8, 16, 64, 256)        # the widths whose searches are shown
    SEPARATE, NESTED = BLUE, AMBER

    def setup(self):
        from .next_systems_real import digits

        s = self.study
        _settings, arrays = evidence(s)
        x, y, xt, yt, _ = digits()
        self.y, self.yt = y, yt
        self.train = R.mnist_split(1024, "train", side=28, seed=17)[0]
        self.test = R.mnist_split(384, "test", side=28, seed=73)[0]
        self.dims = [int(v) for v in s.panels[1].data["x"]]
        self.acc = np.asarray(s.panels[1].data["ys"][0], float)
        self.nested = np.asarray(s.panels[2].data["ys"][1], float)
        self.bytes = [int(v) for v in s.panels[3].data["values"]]
        self.updates = int(s.metadata["training_provenance"]["settings"]["updates"])
        self.extra = int(s.metadata["training_provenance"]["settings"]["nested_updates"])
        weights = {d: arrays["parameter_width%d" % d] for d in self.dims}
        wide = arrays["parameter_nested"]
        self.near = {d: neighbours(weights[d], x, xt) for d in self.dims}
        self.near_cut = {d: neighbours(wide[:, :d], x, xt) for d in self.dims}
        one = xt[self.SHOWN:self.SHOWN + 1]
        self.vectors = {d: unit(one @ weights[d])[0] for d in self.dims}
        self.wide = (one @ wide)[0]

    def subtitle_for(self, k):
        d, acc = self.dims, self.acc
        best = int(np.argmax(acc))
        i16 = d.index(16)
        return [
            "an encoder turns a digit into a list of numbers, its embedding; eight encoders are "
            "trained the same way (%d updates) to give 2, 4, 8 and so on up to %d numbers, 4 "
            "bytes each" % (self.updates, d[-1]),
            "each test digit looks up its nearest training digit by embedding: with 2 numbers "
            "the match is the right digit %s of the time, with 16 %s, with %d %s" % (
                percent(acc[0]), percent(acc[i16]), d[-1], percent(acc[-1])),
            "accuracy climbs steeply to 16 numbers and then levels off: %d numbers score %s, "
            "%d numbers (%s bytes instead of %s) score %s" % (
                d[best], percent(acc[best]), d[-1], f"{self.bytes[-1]:,}",
                f"{self.bytes[best]:,}", percent(acc[-1])),
            "a nested model: one %d-number encoder given %d more updates so its first few "
            "numbers work alone; cut to 16 it scores %s, against %s for the encoder built at "
            "16" % (d[-1], self.extra, percent(self.nested[i16]), percent(acc[i16])),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _vector(self, sc, v, fx, fy, cell, op, keep=None, hot=None):
        """A vector as cells, 32 to a row, its top-left corner at ``(fx, fy)``;
        only the first ``keep`` cells are filled. Returns its height (layout)."""
        n = len(v)
        cols = min(n, 32)
        rows = int(np.ceil(n / cols))
        a = np.zeros(rows * cols)
        a[:n] = v / max(float(np.abs(v).max()), 1e-12)
        if keep is not None:
            a[keep:] = 0.0
        w, h = cols * cell, rows * cell
        dx, dy = self.frac(w, h)
        self.grid(sc, a.reshape(rows, cols), fx + dx / 2, fy + dy / 2, w, h, op, vmax=1.0,
                  frame=False, hot=self.SEPARATE if hot is None else hot, cold=OCHRE)
        return dy

    def _verdict(self, sc, image, fx, fy, size, op, right):
        self.digit(sc, image, fx, fy, size, op, frame=GREEN if right else RED)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        self.text(sc, 0.05, 0.305, "a test digit", op, anchor="lm", strong=True)
        self.digit(sc, self.test[self.SHOWN], 0.10, 0.55, 76, op, frame=AXIS)
        self.arrow(sc, (0.152, 0.55), (0.185, 0.55), AXIS, op)
        self.box(sc, 0.235, 0.55, 78, 40, INKY, op, "encoder", ink=WHITE, size=0.030)
        self.text(sc, 0.235, 0.63, "one per width", op)
        self.text(sc, 0.44, 0.305, "its embedding at each width", op, anchor="lm", strong=True)
        self.text(sc, 0.95, 0.305, "storage", op, anchor="rm", strong=True)
        cell = 9.5
        y = 0.35
        for n, d in enumerate(self.dims):
            a = op * smoothstep(0.08 + 0.09 * n, 0.16 + 0.09 * n, u)
            rows = int(np.ceil(d / 32))
            mid = y + self.frac(0, rows * cell / 2)[1]
            self.text(sc, 0.425, mid, "%d numbers" % d, a, anchor="rm")
            self.pulse(sc, (0.285, 0.55), (0.33, mid), seg(u, 0.04 + 0.09 * n, 0.14 + 0.09 * n),
                       self.SEPARATE, op)
            y += self._vector(sc, self.vectors[d], 0.44, y, cell, a) + self.frac(0, 13.0)[1]
            self.text(sc, 0.95, mid, "%s bytes" % f"{self.bytes[n]:,}", a, anchor="rm",
                      strong=True)

    def phase_2(self, sc, u, op):
        q = self.QUERIES
        xs = [0.275 + j * 0.062 for j in range(q)]
        self.text(sc, 0.05, 0.345, "%d test digits" % q, op, anchor="lm", strong=True)
        for j, fx in enumerate(xs):
            self.digit(sc, self.test[j], fx, 0.345, 40, op, frame=AXIS)
        self.text(sc, 0.95, 0.30, "right digit, all %d tests" % len(self.yt), op, anchor="rm",
                  strong=True)
        self.text(sc, 0.05, 0.405, "nearest training digit by", op, anchor="lm")
        n = len(self.ROWS)
        t = min(u / 0.9, 1.0) * n
        full = 100.0
        for r, d in enumerate(self.ROWS):
            a = op * seg(t, r, r + 0.35)
            if a <= 0.01:
                continue
            y = 0.47 + r * 0.079
            self.text(sc, 0.05, y, "%d numbers" % d, a, anchor="lm", strong=True)
            for j, fx in enumerate(xs):
                i = int(self.near[d][j])
                self._verdict(sc, self.train[i], fx, y, 30, a * seg(t, r + 0.04 * j,
                                                                    r + 0.04 * j + 0.3),
                              self.y[i] == self.yt[j])
            v = self.acc[self.dims.index(d)]
            self.hbar(sc, 0.775, y, full * v, self.SEPARATE, a, value=percent(v),
                      u=seg(t, r + 0.3, r + 0.8), thick=22.0, track=full)
        self.text(sc, 0.25, 0.94, "green frame: the right digit", op, anchor="lm",
                  ink=dark(GREEN))
        self.text(sc, 0.52, 0.94, "red frame: another digit", op, anchor="lm", ink=dark(RED))

    def _curve(self, sc, op, box, u, nested=False, at=None):
        chart = self.chart(box, self.dims, [0.0, 1.0], pad=0.03, xlog=True)
        self.axes(sc, chart, op, "numbers in the embedding", "",
                  xticks=[(v, str(v)) for v in self.dims],
                  yticks=[(v, percent(v, 0)) for v in (0, 0.25, 0.5, 0.75, 1.0)])
        self.plot(sc, chart, self.dims, self.acc, self.SEPARATE, op, 1.0 if nested else u,
                  width=2.6, dot=not nested)
        if nested:
            self.plot(sc, chart, self.dims, self.nested, self.NESTED, op, u, width=2.6)
        return chart

    def phase_3(self, sc, u, op):
        grow, reached = log_sweep(self.dims, seg(u, 0.05, 0.7))
        self.text(sc, 0.10, 0.305, "right digit retrieved, %d test digits" % len(self.yt), op,
                  anchor="lm", strong=True)
        self._curve(sc, op, ((0.10, 0.56), (0.37, 0.82)), grow)
        self.text(sc, 0.64, 0.305, "numbers", op, anchor="lm", strong=True)
        self.text(sc, 0.80, 0.305, "bytes", op, anchor="rm", strong=True)
        self.text(sc, 0.95, 0.305, "right", op, anchor="rm", strong=True)
        best = int(np.argmax(self.acc))
        for n, d in enumerate(self.dims):
            a = op * (1.0 if n <= reached else 0.0)
            y = 0.375 + n * 0.066
            ink = dark(self.SEPARATE) if n == best else None
            self._vector(sc, np.ones(min(n + 1, 8)), 0.585, y - 0.012, 5.0, a)
            self.text(sc, 0.64, y, str(d), a, anchor="lm", ink=ink)
            self.text(sc, 0.80, y, f"{self.bytes[n]:,}", a, anchor="rm", ink=ink)
            self.text(sc, 0.95, y, percent(self.acc[n]), a, anchor="rm", ink=ink,
                      strong=ink is None)

    def phase_4(self, sc, u, op):
        grow, at = log_sweep(self.dims, seg(u, 0.12, 0.88))
        d = self.dims[at]
        q = self.QUERIES
        self.text(sc, 0.05, 0.305, "one %d-number embedding, cut to its first %d" % (
            self.dims[-1], d), op, anchor="lm", strong=True)
        self._vector(sc, self.wide, 0.05, 0.345, 10.0, op, keep=d, hot=self.NESTED)
        xs = [0.075 + j * 0.0463 for j in range(q)]
        self.text(sc, 0.05, 0.56, "test digits, and the nearest by those %d" % d, op,
                  anchor="lm")
        for j, fx in enumerate(xs):
            self.digit(sc, self.test[j], fx, 0.625, 30, op, frame=AXIS)
            i = int(self.near_cut[d][j])
            self._verdict(sc, self.train[i], fx, 0.715, 30, op, self.y[i] == self.yt[j])
        self.text(sc, 0.05, 0.805, "cut to %d: %s right" % (d, percent(self.nested[at])), op,
                  anchor="lm", ink=dark(self.NESTED), size=0.034)
        self.text(sc, 0.05, 0.865, "built at %d: %s right" % (d, percent(self.acc[at])), op,
                  anchor="lm", ink=dark(self.SEPARATE), size=0.034)
        self.text(sc, 0.05, 0.925, "the nested model had %d extra updates" % self.extra, op,
                  anchor="lm")
        self._curve(sc, op, ((0.53, 0.95), (0.36, 0.80)), grow, nested=True)
        self.text(sc, 0.53, 0.305, "one model cut short", op, anchor="lm", ink=dark(self.NESTED))
        self.text(sc, 0.95, 0.305, "a model per width", op, anchor="rm",
                  ink=dark(self.SEPARATE))


ANIMATIONS = {
    "retrieval-augmented-generation": RetrievalAugmentedGeneration,
    "vector-search-hnsw": VectorSearchHNSW,
    "infonce-temperature": InfoNCETemperature,
    "hard-negatives": HardNegatives,
    "dimensionality-of-embeddings": DimensionalityOfEmbeddings,
}

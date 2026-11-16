"""Attention, made cheaper and kept stable: animations 7-12 of the "next 100".

Each draws the mechanism first and the measurement after it (see
:mod:`voxanim.mech`). The numbers are the study's (:mod:`voxanim.next_attention`,
and for 7 and 8 their redesigned experiments in :mod:`voxanim.study_attention_b`:
NumPy transformers trained on Tiny Shakespeare, exact counts, seeded
arithmetic checks, or the pinned GPT-2); what a mechanism needs beyond them
(a mask, the per-tile sums, one head's attention) is computed here from the
same arrays and functions.
"""

from __future__ import annotations

import numpy as np

from . import realdata
from . import tinylm as T
from .easing import smoothstep
from .layout import MIN_TEXT as TEXT
from .layout import STRONG_INK
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIGIT_INK,
    DIM,
    GREEN,
    OCHRE,
    PAPER,
    RED,
    TEAL,
    VIOLET,
    Mechanism,
    dark,
    number,
    soft,
)
from .next_attention import _flash, sparse_mask

INK_TEXT = (1.0, 1.0, 1.0)


def window(n=8, seed=73):
    """``n`` characters of held-out Tiny Shakespeare as they are shown on screen
    (the same window as :func:`voxanim.mech_attention_a.window`; kept here so
    this module does not depend on that one loading)."""
    ids = realdata.text_sequences(n=1, length=n, split="test", seed=seed)[0]
    return [("␣" if ch == " " else "↵" if ch == "\n" else ch) for ch in realdata.decode(ids)]


def glyphs(ids):
    """Character ids as they are shown on screen: a space and a line break made visible."""
    return [("␣" if ch == " " else "↵" if ch == "\n" else ch) for ch in realdata.decode(ids)]


def commas(v):
    """A count in full: 65,536."""
    return f"{int(round(float(v))):,}"


def stage(u, n, stretch=1.0):
    """``u`` (0..1) cut into ``n`` turns: ``(turn, how far through it)``; a
    ``stretch`` above 1 leaves a hold on the last turn's end."""
    x = min(max(u, 0.0) * stretch, 1.0) * n
    k = min(int(x), n - 1)
    return k, min(x - k, 1.0)


def training(m, sc, u, op, box, steps, traces, names, inks, at, step=0.055):
    """The models' training traces drawn together in ``box``, with the
    update reached and each loss there listed at ``at``."""
    chart = m.chart(box, steps, traces)
    m.axes(sc, chart, op, "training update", "loss (nats)")
    for trace, ink in zip(traces, inks):
        m.plot(sc, chart, steps, trace, ink, op, u / 0.85)
    now = int(min(u / 0.85, 1.0) * (len(steps) - 1))
    m.lines(sc, at[0], at[1], [("update %d" % steps[now], None)] + [
        ("%s  %.2f" % (name, trace[now]), dark(ink))
        for name, trace, ink in zip(names, traces, inks)], op, step=step, size=0.034)


# ---------------------------------------------------------------------- 7
def reach_after(n, width, layers):
    """Which positions each position can have heard from after 1..``layers``
    stacked window layers: the band mask composed with itself (exact)."""
    band = T.attention_mask(n, T.Config(mask="window", window=width))
    out, reach = [], np.eye(n, dtype=bool)
    for _ in range(layers):
        reach = (reach.astype(int) @ band.astype(int)) > 0
        out.append(reach.copy())
    return band, out


class SlidingWindow(Mechanism):
    """A band mask four characters wide and the cone of what three stacked
    layers of it reach; then the test of that reach: models of one, two and
    three layers asked to name the character d places back in real text, at
    each depth's limit and one place past it."""

    TITLE = "Sliding-window attention"
    PHASES = (("window", 9.0), ("stacking", 14.0), ("the task", 13.0), ("training", 8.0),
              ("held out", 12.0))
    N, W = 12, 4
    INK = (RED, BLUE, GREEN)
    NAMES = ("1 layer", "2 layers", "3 layers")
    ROW = (0.05, 0.62, 0.64)
    MASK_AT = (0.81, 0.60, 236.0)
    ASKED = (3, 6, 9)                     # the distances walked through: each depth's limit

    def setup(self):
        s, m = self.study, self.study.metadata
        self.ids = [int(v) for v in m["example"]["window"]]    # one held-out window
        self.window = glyphs(self.ids)
        self.chars = self.window[-self.N:]
        self.band, self.reach = reach_after(self.N, self.W, 3)
        self.reached = [int(r[-1].sum()) for r in self.reach]      # by the last character
        self.delays = [int(d) for d in m["delays"]]
        self.back = [int(v) for v in m["reach_back"]]
        self.acc = np.asarray(m["heldout_accuracy"], float)        # depth, distance, seed
        self.covers = np.asarray(m["reach_covers"], bool)
        self.said = np.asarray(m["example"]["said"], int)          # depth, distance, position
        self.steps = np.asarray(s.panels[1].data["x"], float)
        self.traces = np.asarray(s.panels[1].data["ys"], float)
        self.traced = int(s.panels[1].data["delay"])
        self.updates = int(m["steps"])
        self.seeds = len(m["seeds"])
        self.inside, self.outside = self.acc[self.covers], self.acc[~self.covers]

    def subtitle_for(self, k):
        end = self.traces[:, -20:].mean(1)
        return [
            "sliding-window attention: each character reads only itself and the %d before it; "
            "the mask is a band, %d of %d pairs" % (
                self.W - 1, int(self.band.sum()), self.N * self.N),
            "stacked layers pass on what they read, so the reach widens by %d each layer: "
            "%d, %d, then %d of the %d characters (exact)" % (
                self.W - 1, *self.reached, self.N),
            "the test, on %d characters of Tiny Shakespeare: name the character that stood d "
            "places back. 1, 2 and 3 window layers hear %d, %d and %d back" % (
                len(self.window), *self.back),
            "one model per depth and distance, %d updates, %d seeds. Asked for the character "
            "%d back, the loss ends at %.2f, %.2f and %.2f nats for 1, 2, 3 layers" % (
                self.updates, self.seeds, self.traced, *end),
            "held-out text, every seed: %.0f to %.0f%% right where the stack's reach covers "
            "the distance, %.0f to %.0f%% where it does not" % (
                100 * self.inside.min(), 100 * self.inside.max(),
                100 * self.outside.min(), 100 * self.outside.max()),
        ][k]

    def phase_1(self, sc, u, op):
        x0, x1, y = self.ROW
        turn, within = stage(u, self.N, 1.06)
        mask = self.band
        shown = ["" if j == turn or not mask[turn, j] else ch for j, ch in enumerate(self.chars)]
        xs = self.tokens(sc, shown, x0, x1, y, op, w=40.0, h=46.0, size=0.038)
        self.box(sc, xs[turn], y, 40, 46, RED, op, self.chars[turn], ink=INK_TEXT, size=0.038,
                 glow=0.15)
        for j in range(self.N):
            if j == turn:
                continue
            if not mask[turn, j]:
                self.box(sc, xs[j], y, 40, 46, DIM, op * 0.85, self.chars[j], size=0.038)
                continue
            bend = 0.10 + 0.05 * (turn - j)
            a, b = (xs[j], y - 0.055), (xs[turn], y - 0.055)
            self.arc(sc, a, b, RED, op, smoothstep(0.0, 0.5, within), bend, 2.4)
            self.pulse(sc, a, b, (within - 0.35) / 0.6, RED, op, bend)
        self.text(sc, x0 + 0.005, 0.80, "%s reads %d of the %d characters" % (
            self.chars[turn], int(mask[turn].sum()), self.N), op, anchor="lm", strong=True,
            size=0.034)
        self.text(sc, x0 + 0.005, 0.86, "a window %d wide: itself and the %d before it" % (
            self.W, self.W - 1), op, anchor="lm")
        mx, my, side = self.MASK_AT
        cell = self.grid(sc, mask, mx, my, side, side, op, reveal=(turn + 1) / self.N,
                         ring=[(turn, turn)], hot=RED)
        self.text(sc, mx, my - 0.29, "who may read whom", op, strong=True, size=0.034)
        self.text(sc, mx, my + 0.285, "row: the reader   column: the one read", op)
        self.text(sc, cell(turn, 0)[0] - 0.035, cell(turn, 0)[1], self.chars[turn], op,
                  ink=dark(RED), size=0.034)

    def phase_2(self, sc, u, op):
        """The cone: the last character's output after L layers, and every
        connection that feeds it, layer by layer down to the text."""
        L, within = stage(u, 3, 1.04)
        L += 1
        ink = self.INK[L - 1]
        n, hop = self.N, self.W - 1
        x0, x1 = 0.15, 0.66
        ys = [0.87, 0.71, 0.55, 0.39]                     # the text, then layers 1..3
        xs = [x0 + (x1 - x0) * (i + 0.5) / n for i in range(n)]

        def first(level):                                  # leftmost position in the cone
            return n - 1 - (L - level) * hop
        slot = 0.62 / L                                    # each level's connections take a turn
        down = [smoothstep(i * slot, (i + 1) * slot, within) for i in range(L)]
        for level in range(1, 4):
            self.text(sc, 0.05, ys[level], "layer %d" % level, op * (1.0 if level <= L else 0.45),
                      anchor="lm", ink=dark(self.INK[level - 1]) if level <= L else None)
            for i in range(n):
                inside = level <= L and i >= first(level)
                lit = inside and (level == L or down[L - level - 1] > 0.98)
                self.box(sc, xs[i], ys[level], 30, 22, soft(ink, 0.25) if lit else PAPER,
                         op * (1.0 if inside else 0.35), border=ink if lit else AXIS, radius=4)
        self.text(sc, 0.05, ys[0], "the text", op, anchor="lm")
        for i, level in enumerate(range(L, 0, -1)):       # the top level's connections first
            if down[i] <= 0.0:
                continue
            for q in range(max(first(level), 0), n):
                for j in range(max(q - hop, 0), q + 1):
                    self.arrow(sc, (xs[q], ys[level] + 0.026), (xs[j], ys[level - 1] - 0.03),
                               ink, op * 0.8, down[i], 1.5, head=False)
        need = [max(1, int(np.ceil((n - 1 - j) / hop))) for j in range(n)]   # layers to reach j
        landed = smoothstep(0.62, 0.75, within)
        for j, ch in enumerate(self.chars):
            reached = need[j] <= L and (need[j] < L or landed > 0.5)
            if reached:
                self.box(sc, xs[j], ys[0], 36, 38, self.INK[need[j] - 1], op, ch, ink=INK_TEXT,
                         size=0.036)
            else:
                self.box(sc, xs[j], ys[0], 36, 38, DIM, op * 0.85, ch, size=0.036)
        self.box(sc, xs[-1], ys[L], 30, 22, ink, op, glow=0.2, radius=4)
        mx, side = 0.835, 200.0
        self.grid(sc, self.reach[L - 1], mx, 0.56, side, side, op, hot=ink,
                  ring=[(n - 1, j) for j in range(max(first(0), 0), n)])
        self.text(sc, mx, 0.33, "reach after %s" % self.NAMES[L - 1], op, strong=True,
                  size=0.034)
        self.text(sc, mx, 0.80, "1 + %d × %d = %d characters" % (L, hop, 1 + L * hop), op,
                  ink=dark(ink), size=0.034)
        self.text(sc, mx, 0.86, "the last one hears %d of %d" % (self.reached[L - 1], n), op)

    def phase_3(self, sc, u, op):
        """One held-out window, the character each position is asked for, and
        what the first seed's model of each depth answered there."""
        turn, within = stage(u, len(self.ASKED), 1.03)
        d = self.ASKED[turn]
        col, n = self.delays.index(d), len(self.window)
        x0, x1 = 0.225, 0.955
        ys = (0.385, 0.485, 0.60, 0.69, 0.78)
        self.text(sc, 0.045, 0.30, "name the character %d back" % d, op, anchor="lm",
                  strong=True, size=0.038)
        xs = self.tokens(sc, self.window, x0, x1, ys[0], op, w=30.0, h=30.0, size=0.032)
        self.text(sc, 0.045, ys[0], "held-out text", op, anchor="lm")
        self.text(sc, 0.045, ys[1], "the answer", op, anchor="lm", ink=dark(AMBER, 0.35))
        moved = smoothstep(0.0, 0.28, within)
        for i in range(d, n):
            if i in (d, n - 1):                            # the shift, shown at its two ends
                self.arrow(sc, (xs[i - d], ys[0] + 0.034), (xs[i], ys[1] - 0.034), AMBER,
                           op * 0.9, moved, 2.0)
            self.box(sc, xs[i], ys[1], 30, 30, soft(AMBER, 0.45), op * moved, self.window[i - d],
                     border=AMBER, size=0.032)
        told = float(np.clip((within - 0.28) / 0.4, 0.0, 1.0))
        upto = d + int(np.ceil((n - d) * told - 1e-9))
        for row, (name, ink, back) in enumerate(zip(self.NAMES, self.INK, self.back)):
            y = ys[2 + row]
            said = self.said[row][col]
            right = [int(said[i]) == self.ids[i - d] for i in range(n)]
            words = glyphs(said)
            for i in range(d, upto):
                if right[i]:
                    self.box(sc, xs[i], y, 30, 30, ink, op, words[i], ink=INK_TEXT, size=0.032)
                else:
                    self.box(sc, xs[i], y, 30, 30, PAPER, op, words[i], ink=AXIS, border=AXIS,
                             size=0.032)
            self.text(sc, 0.045, y - 0.02, "%s: hears %d" % (name, back), op, anchor="lm",
                      ink=dark(ink))
            if upto > d:
                self.text(sc, 0.045, y + 0.02, "%d of %d right" % (sum(right[d:upto]), upto - d),
                          op, anchor="lm", strong=True)
        self.text(sc, 0.045, 0.875, "what each depth's model answered on this window "
                  "(filled: right)", op, anchor="lm")

    def _reach(self, sc, op, y=0.33):
        """The end of the window, the character asked for at its last position,
        and how much of it each model's last position can have heard from."""
        x0, x1 = 0.05, 0.58
        target = self.N - 1 - self.traced
        xs = self.tokens(sc, ["" if j == target else ch for j, ch in enumerate(self.chars)],
                         x0, x1, y, op, w=40.0, h=34.0, size=0.034)
        self.box(sc, xs[target], y, 40, 34, soft(AMBER, 0.45), op, self.chars[target],
                 border=AMBER, size=0.034)
        self.text(sc, x1 + 0.025, y, "asked: the character %d back" % self.traced, op,
                  anchor="lm", ink=dark(AMBER, 0.35))
        for row, (name, ink, count) in enumerate(zip(self.NAMES, self.INK, self.reached)):
            yy = y + 0.065 + row * 0.04
            self.text(sc, x1 + 0.025, yy, "%s: its last position hears %d" % (name, count), op,
                      anchor="lm", ink=dark(ink))
            for x in xs[self.N - count:]:
                self.box(sc, x, yy, 36, 11, ink, op, radius=3)

    def phase_4(self, sc, u, op):
        self._reach(sc, op)
        training(self, sc, u, op, ((0.10, 0.60), (0.59, 0.86)), self.steps, self.traces,
                 self.NAMES, self.INK, (0.70, 0.63))
        self.text(sc, 0.70, 0.86, "mean of %d seeds" % self.seeds, op, anchor="lm")

    def phase_5(self, sc, u, op):
        """Depth by distance: the share of right answers on held-out text, and
        outlined the cells whose distance the composed mask reaches."""
        cols = len(self.delays)
        cx = [0.345 + 0.108 * j for j in range(cols)]
        ry = (0.465, 0.565, 0.665)
        mean = self.acc.mean(2)
        self.text(sc, 0.05, 0.30, "right answers on held-out text (mean of %d seeds)" %
                  self.seeds, op, anchor="lm", strong=True)
        self.text(sc, 0.05, 0.385, "asked for the character", op, anchor="lm")
        for j, d in enumerate(self.delays):
            self.text(sc, cx[j], 0.385, "%d back" % d, op, strong=True)
        for r, (name, ink, back) in enumerate(zip(self.NAMES, self.INK, self.back)):
            self.text(sc, 0.05, ry[r] - 0.02, name, op, anchor="lm", ink=dark(ink), size=0.034)
            self.text(sc, 0.05, ry[r] + 0.022, "hears %d back" % back, op, anchor="lm")
            for j in range(cols):
                show = op * smoothstep(0.55 * j / cols, 0.55 * (j + 1) / cols, u)
                v = float(mean[r, j])
                self.box(sc, cx[j], ry[r], 84, 40, soft(ink, 1.0 - v), show,
                         "%.0f%%" % (100 * v), ink=INK_TEXT if v > 0.6 else None, border=AXIS,
                         size=0.036)
                if self.covers[r, j]:
                    self.rect(sc, self.P(cx[j], ry[r]), 94, 50, ink, show, border=dark(ink, 0.3),
                              border_w=2.6, radius=9, fill_op=0.0)
        self.lines(sc, 0.05, 0.765, [
            "outlined: the distance is within what the stacked windows reach (exact)",
            ("within reach, every model and seed: %.0f to %.0f%% right" % (
                100 * self.inside.min(), 100 * self.inside.max()), dark(GREEN)),
            ("beyond it: %.0f to %.0f%% - a guess from the characters it does hear" % (
                100 * self.outside.min(), 100 * self.outside.max()), dark(RED)),
        ], op * smoothstep(0.55, 0.75, u), size=0.031, step=0.045)


# ---------------------------------------------------------------------- 8
class SparsePatterns(Mechanism):
    """Five ways to wire a causal mask, each drawn as who the reader reads
    and as its matrix; what each costs at long lengths; then the test: over
    32 characters of real text, which of them can carry the character 10
    back to where it is asked for in two hops, and what each model scores."""

    TITLE = "Sparse attention patterns"
    PHASES = (("patterns", 17.0), ("cost", 8.0), ("two hops", 15.0), ("training", 7.0),
              ("held out", 11.0))
    N = 12
    KINDS = ("dense", "strided", "block", "global", "random")
    INK = (RED, BLUE, GREEN, AMBER, VIOLET)
    RULE = ("every character before it",
            "the last 4, and every 4th before that",
            "only its own block of 4",
            "the last 4, and the first character",
            "the last 4, the first, and 2 picked at random")
    QUERIES = (7, 11)
    ROW = (0.05, 0.62, 0.62)
    MASK_AT = (0.81, 0.60, 236.0)
    READER = 25                           # the position whose route is drawn in "two hops"

    def setup(self):
        s, m = self.study, self.study.metadata
        self.n, self.delay = int(m["length"]), int(m["delay"])
        self.ids = [int(v) for v in m["example"]["window"]]        # one held-out window
        self.window = glyphs(self.ids)
        self.chars = self.window[:self.N]
        self.masks = [sparse_mask(self.N, kind) for kind in self.KINDS]     # the rules, small
        self.pairs = [int(v.sum()) for v in self.masks]
        self.full = [sparse_mask(self.n, kind) for kind in self.KINDS]      # as trained
        self.full_pairs = [int(v) for v in m["pairs"]]
        self.asked = [int(v) for v in m["asked_positions"]]
        self.reach = np.asarray(m["reachable"], bool)              # pattern, asked position
        self.acc = np.asarray(m["heldout_accuracy"], float)        # pattern, seed
        self.said = np.asarray(m["example"]["said"], int)          # pattern, position
        self.ns = np.asarray(s.panels[3].data["x"], float)
        self.counts = np.asarray(s.panels[3].data["ys"], float)
        self.steps = np.asarray(s.panels[1].data["x"], float)
        self.traces = np.asarray(s.panels[1].data["ys"], float)
        self.updates, self.seeds = int(m["steps"]), len(m["seeds"])
        self.layers = int(m["layers"])
        within = np.asarray(m["right_where_reachable"], float).sum()
        beyond = np.asarray(m["right_where_unreachable"], float).sum()
        self.right_within = within / (self.seeds * sum(m["positions_reachable"]))
        self.right_beyond = beyond / (self.seeds * sum(m["positions_unreachable"]))

    def subtitle_for(self, k):
        mean = 100 * self.acc.mean(1)
        return [
            "five ways to let a character read fewer of the others: dense allows all %d pairs "
            "of a %d-character window, the others %s and %d" % (
                self.pairs[0], self.N, ", ".join(str(p) for p in self.pairs[1:-1]),
                self.pairs[-1]),
            "counted exactly at %s characters: dense needs %s pairs, strided %s, and block, "
            "global and random only %s to %s" % (
                commas(self.ns[-1]), number(self.counts[0, -1]), number(self.counts[1, -1]),
                number(self.counts[2:, -1].min()), number(self.counts[2:, -1].max())),
            "the test, on %d characters of Tiny Shakespeare with %d layers: name the character "
            "%d back. Two hops of the mask connect %s of the %d askers (exact)" % (
                self.n, self.layers, self.delay,
                ", ".join(str(int(r.sum())) for r in self.reach), len(self.asked)),
            "one %d-layer model per pattern and seed, %d updates each on the same windows; "
            "the training loss is the mean of %d seeds" % (
                self.layers, self.updates, self.seeds),
            "held-out text, mean of %d seeds: %s" % (self.seeds, ", ".join(
                "%s %.0f%% right with %d pairs" % (kind, value, pairs)
                for kind, value, pairs in zip(self.KINDS, mean, self.full_pairs))),
        ][k]

    def phase_1(self, sc, u, op):
        k, within = stage(u, len(self.KINDS), 1.02)
        q_i, part = stage(within, len(self.QUERIES), 1.05)
        turn, mask, ink = self.QUERIES[q_i], self.masks[k], self.INK[k]
        x0, x1, y = self.ROW
        shown = ["" if j == turn or not mask[turn, j] else ch for j, ch in enumerate(self.chars)]
        xs = self.tokens(sc, shown, x0, x1, y, op, w=40.0, h=46.0, size=0.038)
        self.box(sc, xs[turn], y, 40, 46, ink, op, self.chars[turn], ink=INK_TEXT, size=0.038,
                 glow=0.15)
        for j in range(self.N):
            if j == turn:
                continue
            if not mask[turn, j]:
                self.box(sc, xs[j], y, 40, 46, DIM, op * 0.85, self.chars[j], size=0.038)
                continue
            bend = 0.08 + 0.036 * (turn - j)
            a, b = (xs[j], y - 0.055), (xs[turn], y - 0.055)
            self.arc(sc, a, b, ink, op, smoothstep(0.0, 0.45, part), bend, 2.4)
            self.pulse(sc, a, b, (part - 0.3) / 0.6, ink, op, bend)
        self.text(sc, x0 + 0.005, 0.76, self.KINDS[k], op, anchor="lm", ink=dark(ink),
                  size=0.044)
        self.text(sc, x0 + 0.005, 0.825, "each character reads " + self.RULE[k], op,
                  anchor="lm", strong=True, size=0.032)
        self.text(sc, x0 + 0.005, 0.88, "%s reads %d of the %d characters up to it" % (
            self.chars[turn], int(mask[turn].sum()), turn + 1), op, anchor="lm")
        mx, my, side = self.MASK_AT
        cell = self.grid(sc, mask, mx, my, side, side, op, reveal=(turn + 1) / self.N,
                         ring=[(turn, turn)], hot=ink)
        self.text(sc, mx, my - 0.29, "who may read whom", op, strong=True, size=0.034)
        self.text(sc, mx, my + 0.285, "%d pairs allowed, of dense's %d" % (
            self.pairs[k], self.pairs[0]), op)
        self.text(sc, cell(turn, 0)[0] - 0.035, cell(turn, 0)[1], self.chars[turn], op,
                  ink=dark(ink), size=0.034)

    def _legend(self, sc, op, values, x=0.66, y0=0.37, step=0.118, side=46.0):
        """The five masks in a column, each with its name and a value."""
        for row, (kind, ink, mask, value) in enumerate(zip(self.KINDS, self.INK, self.masks,
                                                           values)):
            y = y0 + row * step
            self.grid(sc, mask, x, y, side, side, op, hot=ink)
            self.text(sc, x + 0.045, y - 0.022, kind, op, anchor="lm", ink=dark(ink), size=0.034)
            self.text(sc, x + 0.045, y + 0.024, value, op, anchor="lm", strong=True)

    def phase_2(self, sc, u, op):
        chart = self.chart(((0.11, 0.57), (0.37, 0.84)), self.ns, self.counts, xlog=True,
                           ylog=True)
        self.axes(sc, chart, op, "characters in the sequence", "pairs allowed",
                  xticks=[(v, commas(v)) for v in self.ns[::2]],
                  yticks=[(v, number(v)) for v in (1e5, 1e7, 1e9)])
        grow = u / 0.8
        for row, ink in zip(self.counts, self.INK):
            self.plot(sc, chart, self.ns, row, ink, op, grow)
        now = int(round(min(grow, 1.0) * (len(self.ns) - 1)))
        self._legend(sc, op, ["%s pairs" % number(row[now]) for row in self.counts])
        self.text(sc, 0.635, 0.92, "at %s characters (exact counts)" % commas(self.ns[now]), op,
                  anchor="lm")

    def route(self, k, q):
        """How the character ``delay`` back can get to position ``q`` under
        pattern ``k``: ``[target]`` if read directly, ``[relay, target]`` if a
        position ``q`` reads has read it one layer earlier, ``[]`` if neither."""
        mask, target = self.full[k], q - self.delay
        if mask[q, target]:
            return [target]
        relays = [j for j in range(target, q + 1) if mask[q, j] and mask[j, target]]
        return [relays[0], target] if relays else []

    WHY = {  # noqa: RUF012
        "direct": "it reads that character itself: one hop",
        "relay": "it reads a character that read it one layer earlier: two hops",
        "none": "nothing it reads has read that character: no route in two layers",
    }

    def phase_3(self, sc, u, op):
        """One held-out window of 32 characters under each pattern: what the
        reader reads, the route (if any) from the character 10 back, which
        positions have such a route, and what the first seed's model answered."""
        k, within = stage(u, len(self.KINDS), 1.02)
        mask, ink, n, d = self.full[k], self.INK[k], self.n, self.delay
        q, target = self.READER, self.READER - self.delay
        x0, x1, y = 0.04, 0.96, 0.455
        slot = (x1 - x0) / n
        xs = [x0 + slot * (i + 0.5) for i in range(n)]
        hops = self.route(k, q)
        kind = "direct" if len(hops) == 1 else "relay" if hops else "none"
        self.text(sc, x0, 0.30, self.KINDS[k], op, anchor="lm", ink=dark(ink), size=0.044)
        self.text(sc, x1, 0.30, "%d pairs of dense's %d" % (
            self.full_pairs[k], self.full_pairs[0]), op, anchor="rm", strong=True)
        for i, ch in enumerate(self.window):
            if i == q:
                self.box(sc, xs[i], y, 24, 32, ink, op, ch, ink=INK_TEXT, size=0.031, radius=4)
            elif i == target:
                self.box(sc, xs[i], y, 24, 32, soft(AMBER, 0.35), op, ch, border=AMBER,
                         size=0.031, radius=4)
            elif mask[q, i]:
                self.box(sc, xs[i], y, 24, 32, soft(ink, 0.5), op, ch, border=ink, size=0.031,
                         radius=4)
            else:
                self.box(sc, xs[i], y, 24, 32, DIM, op * 0.8, ch, size=0.031, radius=4)
        top = y - 0.04
        if kind == "direct":
            self.arc(sc, (xs[target], top), (xs[q], top), ink, op, smoothstep(0.05, 0.4, within),
                     0.13, 2.4)
            self.pulse(sc, (xs[target], top), (xs[q], top), (within - 0.3) / 0.5, ink, op, 0.13)
        elif kind == "relay":
            relay = hops[0]
            self.arc(sc, (xs[target], top), (xs[relay], top), dark(ink, 0.35), op,
                     smoothstep(0.05, 0.3, within), 0.07, 2.4)
            self.arc(sc, (xs[relay], top), (xs[q], top), ink, op,
                     smoothstep(0.3, 0.55, within), 0.13, 2.4)
            self.pulse(sc, (xs[target], top), (xs[relay], top), (within - 0.1) / 0.3,
                       dark(ink, 0.35), op, 0.07)
            self.pulse(sc, (xs[relay], top), (xs[q], top), (within - 0.4) / 0.35, ink, op, 0.13)
        self.text(sc, x0, 0.545, "position %d is asked for the character %d back: %s" % (
            q + 1, d, self.WHY[kind]), op, anchor="lm", strong=True)
        # which askers have a route, then what the model answered
        ys, ya = 0.625, 0.745
        lit = int(self.reach[k].sum())
        self.text(sc, x0, ys - 0.038, "askers that two hops of this mask connect to their "
                  "character (exact): %d of %d" % (lit, len(self.asked)), op, anchor="lm")
        said, words = self.said[k], glyphs(self.said[k])
        right = 0
        for r, i in enumerate(self.asked):
            self.box(sc, xs[i], ys, 24, 14, ink if self.reach[k][r] else DIM, op, radius=3)
            ok = int(said[i]) == self.ids[i - d]
            right += ok
            if ok:
                self.box(sc, xs[i], ya, 24, 30, ink, op, words[i], ink=INK_TEXT, size=0.031,
                         radius=4)
            else:
                self.box(sc, xs[i], ya, 24, 30, PAPER, op, words[i], ink=AXIS, border=AXIS,
                         size=0.031, radius=4)
        self.text(sc, x0, ya - 0.05, "this pattern's trained model on this held-out window "
                  "(filled: right): %d of %d" % (right, len(self.asked)), op, anchor="lm")
        self.text(sc, x0, 0.86, "%d characters, %d layers; the first %d positions have no "
                  "character %d back and are not asked" % (n, self.layers, d, d), op,
                  anchor="lm")

    def phase_4(self, sc, u, op):
        chart = self.chart(((0.11, 0.57), (0.37, 0.84)), self.steps, self.traces)
        self.axes(sc, chart, op, "training update", "loss (nats)")
        for trace, ink in zip(self.traces, self.INK):
            self.plot(sc, chart, self.steps, trace, ink, op, u / 0.85)
        now = int(min(u / 0.85, 1.0) * (len(self.steps) - 1))
        self._legend(sc, op, ["loss %.2f" % trace[now] for trace in self.traces])
        self.text(sc, 0.635, 0.92, "update %d of %d (mean of %d seeds)" % (
            self.steps[now], self.updates, self.seeds), op, anchor="lm")

    def phase_5(self, sc, u, op):
        """Right answers on held-out text per pattern (bar: mean; dots: each
        seed), against the share of askers its two hops connect (the tick)."""
        full = self.span(0.36)[0]
        x = 0.25
        self.text(sc, x, 0.305, "right answers on held-out text (mean of %d seeds, dots: "
                  "each seed)" % self.seeds, op, anchor="lm", strong=True)
        grow = u / 0.45
        for row, (kind, ink, mask, acc, pairs, reach) in enumerate(zip(
                self.KINDS, self.INK, self.full, self.acc, self.full_pairs, self.reach)):
            y = 0.385 + row * 0.092
            self.grid(sc, mask, 0.105, y, 36.0, 36.0, op, hot=ink)
            self.hbar(sc, x, y, full * float(acc.mean()), ink, op, kind, "", grow, thick=28.0,
                      track=full)
            at = self.P(x, y)
            self.label(sc, self.shift(at, full + 14, 0), "%.0f%% · %d pairs" % (
                100 * acc.mean(), pairs), TEXT, anchor="lm", opacity=op, ink=STRONG_INK)
            mark = self.shift(at, full * float(reach.mean()), 0)
            self.polyline(sc, [self.shift(mark, 0, -19), self.shift(mark, 0, 19)], 2.6,
                          DIGIT_INK, op * 0.45, bias=0.6)
            for value in acc:
                self.disk(sc, self.shift(at, full * float(value), 0), 3.4, dark(ink, 0.5),
                          op * smoothstep(0.45, 0.6, u), lift=1.8, outline=0.0)
        self.lines(sc, x, 0.85, [
            "tick: the share of askers two hops of the mask connect to their character (exact)",
            "where they connect, the models are right %.0f%% of the time; where they do "
            "not, %.0f%%" % (100 * self.right_within, 100 * self.right_beyond),
        ], op * smoothstep(0.5, 0.75, u), size=0.031, step=0.045)


# ---------------------------------------------------------------------- 9
def phi(z):
    """The ELU+1 feature map of the study's kernel attention: always positive."""
    return np.where(z > 0, z + 1, np.exp(np.minimum(z, 0)))


def kernel_example():
    """The study's check (seed 8: 16 queries, keys and values, 8 wide), with
    what the drawing needs kept: both sets of weights, and the running sums
    after each key."""
    rng = np.random.default_rng(8)
    q, k, v = [rng.normal(size=(16, 8)) for _ in range(3)]
    fq, fk = phi(q), phi(k)
    a = fq @ fk.T
    sm = np.exp(q @ k.T / np.sqrt(8))
    sm /= sm.sum(1, keepdims=True)
    S = np.cumsum(fk[:, :, None] * v[:, None, :], axis=0)     # S[t]: keys 0..t folded in
    z = np.cumsum(fk, axis=0)
    return dict(fq=fq, fk=fk, v=v, kernel=a / a.sum(1, keepdims=True), softmax=sm,
                pair=(a @ v) / a.sum(1, keepdims=True), S=S, z=z,
                rec=(fq @ S[-1]) / (fq @ z[-1])[:, None], soft_out=sm @ v)


def vbars(m, sc, x0, x1, base, values, unit, ink, op, inks=None, gap=0.25, lift=0.4):
    """Vertical bars spread from ``x0`` to ``x1``, each ``value * unit`` (a
    layout fraction) up from ``base`` (down when negative). Returns their x."""
    n = len(values)
    pitch = (x1 - x0) / n
    w = m.span(pitch)[0] * (1 - gap)
    xs = [x0 + pitch * (i + 0.5) for i in range(n)]
    for i, (x, value) in enumerate(zip(xs, values)):
        h = float(value) * unit
        hp = max(abs(m.span(0, h)[1]), 1.5)
        m.rect(sc, m.P(x, base - h / 2), w, hp, ink if inks is None else inks[i], op,
               radius=min(2.0, hp / 2), lift=lift)
    return xs


class LinearAttention(Mechanism):
    """Softmax attention scoring every query against every key; the kernel
    form folding the keys into one running sum that every query then reads;
    how far its weights are from softmax's; and both trained on real text."""

    TITLE = "Linear attention"
    PHASES = (("every pair", 11.0), ("running sum", 18.0), ("not softmax", 11.0),
              ("real text", 13.0))
    INK = (RED, BLUE)
    NAMES = ("softmax", "kernel")
    DATA = dict(hot=TEAL, cold=OCHRE)             # signed numbers that belong to neither model  # noqa: E501, RUF012

    def setup(self):
        s = self.study
        self.x = kernel_example()
        self.n = len(self.x["v"])
        self.identity, self.gap = [float(v) for v in s.panels[0].data["values"]]
        self.steps = np.asarray(s.panels[1].data["x"], float)
        self.traces = np.asarray(s.panels[1].data["ys"], float)
        self.heldout = [float(v) for v in s.metadata["heldout_loss"]]
        self.updates = int(s.metadata["steps"])

    def subtitle_for(self, k):
        n = self.n
        lo, hi = self.heldout
        return [
            "softmax attention: every query scores every key, so %d tokens cost %d × %d = %d "
            "scores before any value is mixed" % (n, n, n, n * n),
            "kernel attention: keys, made positive by a feature map, are added once into a "
            "running sum that every query reads: %d steps, not %d scores" % (2 * n, n * n),
            "the running sum matches pair-by-pair kernel attention to %.1e, but its outputs "
            "differ from softmax's by up to %.2f: it is a different attention" % (
                self.identity, self.gap),
            "both trained for %d updates on the same Tiny Shakespeare windows: held-out loss "
            "%.2f nats for softmax, %.2f for the kernel" % (self.updates, lo, hi),
        ][k]

    def phase_1(self, sc, u, op):
        n, x = self.n, self.x
        turn, within = stage(u, n, 1.08)
        y, h = 0.60, 230.0
        cell = self.grid(sc, x["softmax"], 0.19, y, h, h, op, reveal=(turn + 1) / n, hot=RED,
                         ring=[(turn, j) for j in range(n)])
        self.grid(sc, x["v"], 0.42, y, h / 2, h, op, **self.DATA)
        out = self.grid(sc, x["soft_out"], 0.59, y, h / 2, h, op, reveal=(turn + 1) / n,
                        ring=[(turn, j) for j in range(8)], **self.DATA)
        for fx, s in ((0.19, "attention weights"), (0.42, "values"), (0.59, "outputs")):
            self.text(sc, fx, 0.32, s, op, strong=True, size=0.032)
        self.text(sc, 0.345, y, "×", op, size=0.05)
        self.text(sc, 0.508, y, "=", op, size=0.05)
        self.text(sc, 0.19, 0.885, "row: a query   column: a key", op)
        yy = cell(turn, 0)[1]
        self.pulse(sc, (0.31, yy), (out(turn, 0)[0] - 0.01, yy), within, RED, op)
        self.lines(sc, 0.70, 0.46, [
            ("query %d of %d" % (turn + 1, n), dark(RED)),
            ("%d scores so far" % ((turn + 1) * n), None)], op, size=0.036, step=0.06)
        self.lines(sc, 0.70, 0.62, [
            "a softmax needs every score in its row before it can weigh one value",
            "%d tokens: %d scores" % (n, n * n)], op, width=0.27, step=0.045)

    def phase_2(self, sc, u, op):
        n, x = self.n, self.x
        folding = u < 0.5
        j, a = stage(u / 0.5, n, 1.03) if folding else (n - 1, 1.0)
        i, b = stage((u - 0.5) / 0.45, n, 1.0) if not folding else (-1, 0.0)
        y, h, w = 0.565, 224.0, 96.0
        keys = self.grid(sc, x["fk"], 0.095, y, w, h, op, hot=BLUE,
                         ring=[(j, c) for c in range(8)] if folding else ())
        self.grid(sc, x["v"], 0.225, y, w, h, op,
                         ring=[(j, c) for c in range(8)] if folding else (), **self.DATA)
        side = 168.0
        self.grid(sc, x["S"][j], 0.435, y - 0.03, side, side, op, vmax=float(abs(x["S"]).max()),
                  **self.DATA)
        self.grid(sc, x["z"][j][None], 0.435, y + 0.175, side, side / 8, op,
                  vmax=float(x["z"].max()), hot=BLUE)
        queries = self.grid(sc, x["fq"], 0.655, y, w, h, op, hot=BLUE,
                            ring=[(i, c) for c in range(8)] if i >= 0 else ())
        out = self.grid(sc, x["rec"], 0.85, y, w, h, op, reveal=(i + 1) / n,
                        ring=[(i, c) for c in range(8)] if i >= 0 else (), **self.DATA)
        for fx, s in ((0.095, "keys"), (0.225, "values"), (0.435, "running sum"),
                      (0.655, "queries"), (0.85, "outputs")):
            self.text(sc, fx, 0.30, s, op, strong=True, size=0.032)
        left, right = 0.435 - self.frac(side / 2)[0], 0.435 + self.frac(side / 2)[0]
        if folding:
            yy = keys(j, 0)[1]
            self.arrow(sc, (0.275, yy), (left - 0.012, y - 0.03), AMBER, op, a / 0.4, 2.0)
            self.pulse(sc, (0.275, yy), (left - 0.012, y - 0.03), (a - 0.1) / 0.8, AMBER, op)
        else:
            yy = queries(i, 0)[1]
            self.arrow(sc, (right + 0.012, y - 0.03), (0.60, yy), AMBER, op, b / 0.3, 2.0)
            self.pulse(sc, (right + 0.012, y - 0.03), (0.60, yy), b / 0.5, AMBER, op)
            self.arrow(sc, (0.71, yy), (out(i, 0)[0] - 0.012, yy), AMBER, op, (b - 0.3) / 0.3,
                       2.0)
            self.pulse(sc, (0.71, yy), (out(i, 0)[0] - 0.012, yy), (b - 0.45) / 0.5, AMBER, op)
        self.text(sc, 0.05, 0.855,
                  "key %d of %d: it and its value are added to the running sum" % (j + 1, n)
                  if folding else
                  "query %d of %d reads the finished sum and gets its output" % (i + 1, n), op,
                  anchor="lm", strong=True, size=0.034)
        done = (j + 1) if folding else n + i + 1
        self.text(sc, 0.05, 0.91, "%d steps so far; softmax would be at %d of its %d scores" % (
            done, (0 if folding else (i + 1) * n), n * n), op, anchor="lm")

    def phase_3(self, sc, u, op):
        x = self.x
        qi, _within = stage(u, 4, 1.0)
        top = float(max(x["softmax"][:4].max(), x["kernel"][:4].max()))
        for row, (name, key, ink) in enumerate((("softmax", "softmax", RED),
                                                ("kernel", "kernel", BLUE))):
            base = 0.53 + row * 0.30
            vbars(self, sc, 0.07, 0.60, base, x[key][qi], 0.17 / top, ink, op)
            self.polyline(sc, [self.P(0.07, base), self.P(0.60, base)], 1.2, AXIS, op)
            self.text(sc, 0.07, base - 0.215, "%s weights of query %d on the %d keys" % (
                name, qi + 1, self.n), op, anchor="lm", ink=dark(ink), size=0.032)
            self.text(sc, 0.63, base - 0.06, "largest  %.2f" % x[key][qi].max(), op, anchor="lm",
                      ink=dark(ink), size=0.034)
        self.lines(sc, 0.63, 0.57, [
            "running sum against pair-by-pair kernel: %.1e apart, which is rounding" %
            self.identity], op, width=0.31, step=0.042)
        self.lines(sc, 0.63, 0.86, [
            "kernel outputs against softmax outputs: up to %.2f apart" % self.gap],
            op, width=0.31, step=0.042)

    def phase_4(self, sc, u, op):
        training(self, sc, u / 0.6, op, ((0.10, 0.55), (0.42, 0.84)), self.steps, self.traces,
                 self.NAMES, self.INK, (0.63, 0.36))
        show = smoothstep(0.55, 0.65, u)
        worst = max(self.heldout)
        full = self.span(0.20)[0]
        self.text(sc, 0.63, 0.58, "loss on held-out text", op * show, anchor="lm", strong=True)
        for row, (name, ink, value) in enumerate(zip(self.NAMES, self.INK, self.heldout)):
            self.hbar(sc, 0.71, 0.65 + row * 0.075, full * value / worst, ink, op * show, name,
                      "%.2f" % value, (u - 0.55) / 0.2, thick=28.0, track=full)
        self.lines(sc, 0.63, 0.82, [
            "the kernel model is %.2f nats %s here; no speed was measured" % (
                abs(self.heldout[0] - self.heldout[1]),
                "better" if self.heldout[1] < self.heldout[0] else "worse")],
            op * smoothstep(0.75, 0.9, u), width=0.31, step=0.042)
        self.text(sc, 0.10, 0.335, "one small transformer with each kind of attention", op,
                  anchor="lm", strong=True)


# ---------------------------------------------------------------------- 10
def flash_example(tile=8):
    """The study's arrays (seed 4: 32 queries, keys and values, 16 wide):
    dense softmax attention, the tiled result, and query 1's running maximum
    and sum after each tile of ``tile`` keys with what changed them."""
    rng = np.random.default_rng(4)
    q, k, v = [rng.normal(size=(32, 16)) for _ in range(3)]
    scores = q @ k.T / 4
    a = np.exp(scores - scores.max(1, keepdims=True))
    a /= a.sum(1, keepdims=True)
    tiled, hist = _flash(q, k, v, tile)
    rows, m, total = [], -np.inf, 0.0
    for start in range(0, len(k), tile):
        part = scores[0, start:start + tile]
        new = max(m, float(part.max()))
        factor, add = float(np.exp(m - new)), float(np.exp(part - new).sum())
        total = total * factor + add
        rows.append((float(part.max()), new, factor, add, total))
        m = new
    return dict(scores=scores, weights=a, dense=a @ v, tiled=tiled, rows=np.array(rows),
                hist=hist, tile=tile)


def size(v):
    """A byte count: in full below a million."""
    return commas(v) if v < 1e6 else number(v)


class FlashTiling(Mechanism):
    """The online softmax: one query's scores taken eight keys at a time,
    with a running maximum and a running sum that is rescaled whenever the
    maximum rises; the result against the all-at-once one; the scratch
    memory each needs."""

    TITLE = "Flash attention: the online softmax"
    PHASES = (("all at once", 10.0), ("tile by tile", 23.0), ("same answer", 9.0),
              ("memory", 12.0))
    BARS = (0.07, 0.93)

    def setup(self):
        s = self.study
        self.x = flash_example()
        self.n = len(self.x["scores"])
        self.tiles = len(self.x["rows"])
        self.error = float(s.panels[2].data["values"][0])
        self.ns = np.asarray(s.panels[3].data["x"], float)
        self.bytes = np.asarray(s.panels[3].data["ys"], float)

    def subtitle_for(self, k):
        n, rows = self.n, self.x["rows"]
        return [
            "the plain way: score every query against every key, %d × %d = %s numbers held in "
            "memory, then take each row's softmax" % (n, n, commas(n * n)),
            "the online softmax takes %d keys at a time and keeps two numbers, the largest score "
            "so far and a sum that is rescaled when that rises" % self.x["tile"],
            "after the last tile the running sum is %.2f, as it is all at once; over all %d "
            "queries the two outputs differ by at most %.1e" % (rows[-1, 4], n, self.error),
            "scratch memory at %s tokens: %s bytes for every score, %s for one tile of scores, "
            "%d times less (exact arithmetic)" % (
                commas(self.ns[-1]), size(self.bytes[0, -1]), size(self.bytes[1, -1]),
                round(self.bytes[0, -1] / self.bytes[1, -1])),
        ][k]

    def phase_1(self, sc, u, op):
        n, x = self.n, self.x
        s = x["scores"][0]
        top = int(np.argmax(s))
        shown = int(np.ceil(min(u / 0.45, 1.0) * n))
        unit = 0.15 / float(abs(s).max())
        inks = [AMBER if i == top and shown == n else RED for i in range(n)]
        vals = np.where(np.arange(n) < shown, s, 0.0)
        vbars(self, sc, 0.06, 0.60, 0.53, vals, unit, RED, op, inks=inks)
        self.polyline(sc, [self.P(0.06, 0.53), self.P(0.60, 0.53)], 1.2, AXIS, op)
        self.text(sc, 0.06, 0.33, "query 1 scored against each of the %d keys" % n, op,
                  anchor="lm", strong=True, size=0.034)
        self.lines(sc, 0.06, 0.78, [
            ("its softmax needs the largest score: %.2f" % s.max(), dark(AMBER, 0.3)),
            ("and the sum of exp(score − largest) over all %d: %.2f" % (n, x["rows"][-1, 4]),
             None)], op * smoothstep(0.45, 0.6, u), size=0.032, step=0.05)
        mx, side = 0.80, 236.0
        self.grid(sc, x["weights"], mx, 0.60, side, side, op, hot=RED,
                  reveal=smoothstep(0.3, 0.95, u))
        self.text(sc, mx, 0.32, "all %s scores, held at once" % commas(n * n), op, strong=True,
                  size=0.032)
        self.text(sc, mx, 0.89, "row: a query   column: a key", op)

    def phase_2(self, sc, u, op):
        n, x, tile = self.n, self.x, self.x["tile"]
        s, rows = x["scores"][0], x["rows"]
        k, w = stage(u, self.tiles, 1.06)
        x0, x1 = self.BARS
        base, unit = 0.44, 0.105 / float(abs(s).max())
        pitch = (x1 - x0) / n
        inks, vals = [], []
        for i in range(n):
            t = i // tile
            here = int(np.argmax(s[t * tile:(t + 1) * tile])) + t * tile == i
            if t < k:
                inks.append(soft(BLUE, 0.7))
                vals.append(s[i])
            elif t == k:
                inks.append(AMBER if here and w > 0.25 else BLUE)
                vals.append(s[i] * smoothstep(0.0, 0.18, w))
            else:
                inks.append(DIM)
                vals.append(0.0)
        for t in range(self.tiles):
            cx = x0 + pitch * (t + 0.5) * tile
            self.box(sc, cx, base, self.span(pitch * tile)[0] - 6, 132, PAPER,
                     op * (1.0 if t == k else 0.5), border=BLUE if t == k else AXIS, radius=5,
                     fill_op=0.5 if t <= k else 0.15)
            self.text(sc, cx, 0.60, "tile %d" % (t + 1) if t != k else
                      "tile %d: in memory now" % (t + 1), op * (1.0 if t <= k else 0.5),
                      ink=dark(BLUE) if t == k else None)
        vbars(self, sc, x0, x1, base, vals, unit, BLUE, op, inks=inks, gap=0.4, lift=0.9)
        self.text(sc, x0, 0.275, "query 1's scores, %d keys at a time" % tile, op, anchor="lm",
                  strong=True, size=0.032)
        m_old = rows[k - 1, 1] if k else None
        m_now = rows[k, 1] if w > 0.4 else m_old
        if m_now is not None:
            end = x0 + pitch * tile * (k + (1 if w > 0.4 else 0))
            yy = base - m_now * unit
            self.polyline(sc, [self.P(x0, yy), self.P(end, yy)], 1.6, AMBER, op, bias=1.0)
        cols = (0.36, 0.50, 0.64, 0.77, 0.89)
        heads = ("largest score in it", "running max", "rescale old sum", "add this tile",
                 "running sum")
        self.text(sc, x0, 0.665, "tile", op, anchor="lm")
        for cx, head in zip(cols, heads):
            self.text(sc, cx, 0.665, head, op)
        for r in range(k + 1):
            y = 0.72 + r * 0.05
            big, m, factor, add, total = rows[r]
            cells = ("%.2f" % big, "%.2f" % m,
                     "nothing yet" if r == 0 else "× %.3f" % factor, "+ %.2f" % add,
                     "%.2f" % total)
            self.text(sc, x0, y, "%d: keys %d-%d" % (r + 1, r * tile + 1, (r + 1) * tile), op,
                      anchor="lm", ink=dark(BLUE) if r == k else None, size=0.032)
            for c, (cx, cell) in enumerate(zip(cols, cells)):
                seen = 1.0 if r < k else smoothstep(0.25 + 0.14 * c, 0.33 + 0.14 * c, w)
                rose = r > 0 and c == 1 and rows[r, 1] > rows[r - 1, 1]
                self.text(sc, cx, y, cell, op * seen, size=0.032, strong=(c == 4),
                          ink=dark(AMBER, 0.3) if (c == 0 or rose) and c < 2 else None)
        done = smoothstep(0.9, 1.0, w) if k == self.tiles - 1 else 0.0
        self.text(sc, x0, 0.93, "the weights are exp(score − %.2f) ÷ %.2f: the softmax, and "
                  "no more than %d scores were ever in memory" % (rows[-1, 1], rows[-1, 4], tile),
                  op * done, anchor="lm", strong=True)

    def phase_3(self, sc, u, op):
        x, n = self.x, self.n
        reveal = smoothstep(0.0, 0.7, u)
        h = 236.0
        vmax = float(abs(x["dense"]).max())
        for fx, key, name, ink in ((0.17, "dense", "all at once", RED),
                                   (0.43, "tiled", "tile by tile", BLUE)):
            self.grid(sc, x[key], fx, 0.60, 150.0, h, op, reveal=reveal, vmax=vmax,
                      hot=TEAL, cold=OCHRE)
            self.text(sc, fx, 0.32, name, op, ink=dark(ink), size=0.034)
        self.text(sc, 0.30, 0.60, "=", op, size=0.06)
        self.text(sc, 0.30, 0.89, "the outputs: one row for each of the %d queries" % n, op)
        self.lines(sc, 0.60, 0.46, [
            ("largest difference", None),
            ("%.1e" % self.error, None)], op * smoothstep(0.6, 0.8, u), size=0.044, step=0.08)
        self.lines(sc, 0.60, 0.66, [
            "two units in the last place a 64-bit number holds: rounding, not approximation"
            if abs(self.error / np.finfo(float).eps - 2) < 1e-9 else
            "the size of 64-bit rounding: not an approximation"],
            op * smoothstep(0.6, 0.8, u), width=0.33, step=0.045)

    def phase_4(self, sc, u, op):
        x, n, tile = self.x, self.n, self.x["tile"]
        side = 190.0
        self.grid(sc, x["weights"], 0.15, 0.565, side, side, op, hot=RED)
        self.grid(sc, x["weights"][:, :tile], 0.335, 0.565, side * tile / n, side, op, hot=BLUE)
        self.text(sc, 0.15, 0.325, "every score: n × n", op, ink=dark(RED), size=0.032)
        self.text(sc, 0.35, 0.325, "one tile: n × %d" % tile, op, ink=dark(BLUE), size=0.032)
        chart = self.chart(((0.49, 0.94), (0.37, 0.80)), self.ns, self.bytes, xlog=True,
                           ylog=True)
        self.axes(sc, chart, op, "tokens (n)", "scratch bytes",
                  xticks=[(v, commas(v)) for v in self.ns[[0, 2, -1]]],
                  yticks=[(v, size(v)) for v in (1e4, 1e5, 1e6, 1e7)])
        grow = u / 0.8
        for row, ink in zip(self.bytes, (RED, BLUE)):
            self.plot(sc, chart, self.ns, row, ink, op, grow)
        now = int(round(min(grow, 1.0) * (len(self.ns) - 1)))
        self.lines(sc, 0.05, 0.81, [
            ("at %s tokens" % commas(self.ns[now]), None),
            ("every score: %s bytes" % size(self.bytes[0, now]), dark(RED)),
            ("one tile: %s bytes, %d times less" % (
                size(self.bytes[1, now]), round(self.bytes[0, now] / self.bytes[1, now])),
             dark(BLUE))], op, size=0.032, step=0.048)


# ---------------------------------------------------------------------- 11
def sink_example():
    """The study's 64-token stream of held-out Shakespeare read once by the
    pinned GPT-2 with nothing evicted: the tokens as text, and every head's
    attention, ``(layer, head, reader, read)``."""
    from . import gpt2, gpt2_hooks

    tok = gpt2.Tokenizer()
    stream = tok.encode(realdata.text("test")[:520])[:64]
    attn = gpt2_hooks.forward(stream[:-1], record=True)["attn"]
    return [tok.decode([t]) for t in stream], attn


def cache_slots(policy, i, window=32, sinks=4):
    """The positions in the cache while token ``i`` is read, as the study's
    streaming loop keeps them: everything (0), the newest ``window`` (1), or
    the first ``sinks`` and the newest ``window - sinks`` (2)."""
    if policy == 0 or i < window:
        return list(range(i + 1))
    if policy == 1:
        return list(range(i - window + 1, i + 1))
    return list(range(sinks)) + list(range(i - (window - sinks) + 1, i + 1))


class AttentionSinks(Mechanism):
    """GPT-2's heads sending their attention to the first token; what a
    32-slot cache loses when it evicts that token and keeps when it does
    not; the perplexity each cache policy gives on the same text."""

    TITLE = "Attention sinks"
    PHASES = (("one head", 13.0), ("every head", 11.0), ("eviction", 17.0), ("perplexity", 11.0))
    SHOWN = 9                                     # tokens drawn as text in the first phase
    NAMES = ("full cache", "last 32", "4 sinks + last 28")
    INK = (BLUE, RED, GREEN)
    SKIP = 3            # running perplexity is drawn from its fourth point: the first three
    #                     average one to three tokens and reach 1e10

    def setup(self):
        s = self.study
        self.mass = np.asarray(s.panels[0].data["values"], float)
        words, attn = sink_example()
        self.words = [w.replace("\n", "↵").strip() or "␣" for w in words]
        self.attn = attn
        self.reader = self.SHOWN - 1
        # three heads to watch: the first one, the one nearest an even split between the first
        # token and the rest for this reader, and the strongest sink of the study's average
        on_first = attn[:, :, self.reader, 0]
        picks = (0, int(np.argmin(abs(on_first - 0.5))), int(np.argmax(self.mass)))
        self.heads = [tuple(int(v) for v in np.unravel_index(i, self.mass.shape)) for i in picks]
        self.steps = np.asarray(s.panels[1].data["x"], float)
        self.running = np.asarray(s.panels[1].data["ys"], float)
        self.perplexity = [float(v) for v in s.metadata["perplexity"]]
        self.window = int(s.metadata["window"])
        self.slots = [int(v) for v in s.panels[3].data["values"]]
        self.last = int(self.steps[-1])
        self.over_half = int((self.mass > 0.5).sum())

    def share(self, head, j=0):
        """The share of the reader's attention that ``head`` puts on token ``j``."""
        return float(self.attn[head[0], head[1], self.reader, j])

    def subtitle_for(self, k):
        full, last, sinks = self.perplexity
        late = len(self.steps)
        return [
            "GPT-2 reading Shakespeare: in many of its heads the token being read sends most of "
            "its attention back to the very first token; in layer %d, head %d, %d%% of it" % (
                self.heads[2][0] + 1, self.heads[2][1] + 1,
                round(100 * self.share(self.heads[2]))),
            "averaged over the last %d tokens of the stream, %d of GPT-2's %d heads put more "
            "than half of their attention on the first token" % (
                late, self.over_half, self.mass.size),
            "a cache of %d slots has to evict: dropping the oldest loses the first token, "
            "keeping the first 4 and the newest %d does not" % (self.window, self.window - 4),
            "perplexity on the last %d tokens (lower is better): %.0f with everything cached, "
            "%s with the first tokens evicted, %.0f with 4 of them kept" % (
                late, full, commas(last), sinks),
        ][k]

    def phase_1(self, sc, u, op):
        k, within = stage(u, len(self.heads), 1.0)
        head = self.heads[k]
        n, r, y = self.SHOWN, self.reader, 0.76
        weights = self.attn[head[0], head[1], r, :n]
        xs = self.tokens(sc, [""] * n, 0.05, 0.95, y, op, w=80.0, h=46.0)
        for j, x in enumerate(xs):
            if j == 0:
                self.box(sc, x, y, 80, 46, AMBER, op, self.words[j], ink=INK_TEXT, size=0.038,
                         glow=0.15)
            elif j == r:
                self.box(sc, x, y, 80, 46, VIOLET, op, self.words[j], ink=INK_TEXT, size=0.038)
            else:
                self.box(sc, x, y, 80, 46, PAPER, op, self.words[j], border=AXIS, size=0.038)
            self.text(sc, x, y + 0.075, "%d%%" % round(100 * weights[j]), op,
                      strong=weights[j] >= 0.1, size=0.032)
            if j < r and weights[j] >= 0.004:
                bend = 0.07 + 0.04 * (r - j)
                a, b = (x, y - 0.052), (xs[r], y - 0.052)
                self.arc(sc, a, b, VIOLET, op, smoothstep(0.0, 0.35, within), bend,
                         1.2 + 10.0 * float(weights[j]))
                self.pulse(sc, a, b, (within - 0.25) / 0.5, VIOLET, op, bend)
        self.text(sc, 0.05, 0.31, "layer %d, head %d" % (head[0] + 1, head[1] + 1), op,
                  anchor="lm", ink=dark(VIOLET), size=0.044)
        self.text(sc, 0.05, 0.37, "where “%s” looks: %d%% of this head's attention is on the "
                  "first token" % (self.words[r], round(100 * weights[0])), op, anchor="lm",
                  strong=True, size=0.032)
        self.text(sc, 0.05, 0.915, "the first %d tokens of the stream; ↵ is a line break" % n,
                  op, anchor="lm")

    def phase_2(self, sc, u, op):
        layers, heads = self.mass.shape
        reveal = min(u / 0.7, 1.0)
        row = min(int(reveal * layers), layers - 1)
        fx, fy, side = 0.30, 0.61, 250.0
        shown = [h for h in self.heads if h[0] <= row]
        cell = self.grid(sc, self.mass, fx, fy, side, side, op, reveal=(row + 1) / layers,
                         vmax=1.0, hot=VIOLET, ring=shown)
        self.text(sc, fx, 0.305, "share of attention on the first token", op, strong=True,
                  size=0.032)
        self.text(sc, cell(0, 0)[0] - 0.035, cell(0, 0)[1], "layer 1", op, anchor="rm")
        self.text(sc, cell(0, 0)[0] - 0.035, cell(layers - 1, 0)[1], "layer %d" % layers, op,
                  anchor="rm")
        self.text(sc, fx, 0.925, "%d heads in every layer" % heads, op)
        self.box(sc, 0.56, 0.39, 66, 44, AMBER, op, self.words[0], ink=INK_TEXT, size=0.034)
        self.text(sc, 0.61, 0.39, "the first token of the stream", op, anchor="lm", strong=True)
        self.lines(sc, 0.525, 0.50, [
            ("layer %d" % (row + 1), dark(VIOLET)),
            ("its heads give it %d%% on average" % round(100 * self.mass[row].mean()), None)],
            op, size=0.036, step=0.06)
        self.lines(sc, 0.525, 0.68, [
            "%d of the %d heads give it more than half" % (self.over_half, self.mass.size),
            "all heads together: %d%%" % round(100 * self.mass.mean()),
            "the three ringed heads are the ones just shown"],
            op * smoothstep(0.7, 0.85, u), step=0.05, size=0.032)

    def _strip(self, sc, policy, i, fx, fy, w, h, op):
        """The stream's positions as a row of cache slots: held, or evicted."""
        held = cache_slots(policy, i, self.window)
        values = np.zeros(self.last + 1)
        values[held] = 1.0
        if policy == 2 and i >= self.window:
            values[:4] = -1.0
        return self.grid(sc, values[None], fx, fy, w, h, op, vmax=1.0, hot=self.INK[policy],
                         cold=AMBER)

    def phase_3(self, sc, u, op):
        first = self.window - 8
        i = int(round(first + (self.last - first) * min(u / 0.9, 1.0)))
        cell = None
        for p, name in enumerate(self.NAMES):
            y = 0.365 + p * 0.062
            self.text(sc, 0.05, y, name, op, anchor="lm", ink=dark(self.INK[p]), size=0.032)
            cell = self._strip(sc, p, i, 0.60, y, 600.0, 18.0, op)
        self.text(sc, min(cell(0, i)[0], 0.89), 0.305, "reading token %d" % (i + 1), op,
                  strong=True)
        xs, ys = self.steps[self.SKIP:], self.running[:, self.SKIP:]
        chart = self.chart(((0.12, 0.60), (0.61, 0.86)), xs, ys, ylog=True)
        lo, hi = chart.y_range
        self.axes(sc, chart, op, "tokens read", "perplexity so far",
                  xticks=[(v, "%d" % (v + 1)) for v in xs[::9]],
                  yticks=[(v, commas(v)) for v in (100, 300, 1000, 3000, 10000) if lo <= v <= hi])
        grow = (i - xs[0]) / (xs[-1] - xs[0])
        if grow > 0:
            for row, ink in zip(ys, self.INK):
                self.plot(sc, chart, xs, row, ink, op, grow)
        if i < self.window:
            rows = [("the cache still has room:", None), ("nothing is evicted yet", None)]
        elif i < xs[0]:
            rows = [("the cache is full: “last %d”" % self.window, None),
                    ("has just dropped the first token", None)]
        else:
            at = int(i - self.steps[0])
            rows = [("perplexity since token %d" % (self.window + 1), None)] + [
                ("%s  %s" % (name, commas(self.running[p, at])), dark(self.INK[p]))
                for p, name in enumerate(self.NAMES)]
        self.lines(sc, 0.66, 0.64, rows, op, size=0.032, step=0.055)

    def phase_4(self, sc, u, op):
        worst = max(self.perplexity)
        full = self.span(0.44)[0]
        for p, (name, ink, value, slots) in enumerate(zip(self.NAMES, self.INK, self.perplexity,
                                                          self.slots)):
            y = 0.38 + p * 0.16
            self.text(sc, 0.05, y - 0.035, name, op, anchor="lm", ink=dark(ink), size=0.034)
            self._strip(sc, p, self.last, 0.523, y - 0.035, 420.0, 14.0, op)
            self.text(sc, 0.79, y - 0.035, "%d slots" % slots, op, anchor="lm")
            self.hbar(sc, 0.28, y + 0.03, full * value / worst, ink, op, "perplexity",
                      commas(value), u / 0.5, thick=26.0, track=full)
        gone = smoothstep(0.5, 0.75, u)
        self.lines(sc, 0.05, 0.855, [
            "the same %d slots, spent differently: %s without the first tokens, %.0f with them" %
            (self.window, commas(self.perplexity[1]), self.perplexity[2]),
            "one stream of %d tokens of Shakespeare: a demonstration, not a guarantee" %
            (self.last + 2)], op * gone, size=0.031, step=0.045)


# ---------------------------------------------------------------------- 12
def scale_example():
    """The study's random-vector experiment (seed 54: 128 queries against 32
    keys at each width ``d``) with the first query's scores and weights kept
    for drawing, and the scaled scores given the same measurements."""
    rng = np.random.default_rng(54)
    ds = np.array([8, 16, 32, 64, 128, 256, 512])
    keep = {key: [] for key in ("scores", "spread", "raw", "scaled", "h", "hs", "g", "gs")}
    for d in ds:
        q = rng.normal(size=(128, d))
        k = rng.normal(size=(32, d))
        s = q @ k.T
        p = np.exp(s - s.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        z = s / np.sqrt(d)
        p2 = np.exp(z - z.max(1, keepdims=True))
        p2 /= p2.sum(1, keepdims=True)
        for key, value in (("scores", s[0]), ("spread", s.std()), ("raw", p[0]),
                           ("scaled", p2[0]),
                           ("h", -(p * np.log(p + 1e-30)).sum(1).mean()),
                           ("hs", -(p2 * np.log(p2 + 1e-30)).sum(1).mean()),
                           ("g", (p * (1 - p)).sum(1).mean()),
                           ("gs", (p2 * (1 - p2)).sum(1).mean())):
            keep[key].append(value)
    return dict(d=ds, **{key: np.asarray(value) for key, value in keep.items()})


class ScoreScale(Mechanism):
    """Dot-product scores spreading as the vectors widen, the softmax they
    saturate, what dividing by the square root of the width restores, and a
    model trained with and without the division."""

    TITLE = "Why divide scores by √d?"
    PHASES = (("scores", 11.0), ("softmax", 13.0), ("entropy", 9.0), ("gradient", 10.0),
              ("real text", 12.0))
    INK = (BLUE, RED)                             # the study's order: scaled, unscaled
    NAMES = ("scaled", "unscaled")
    BARS = (0.07, 0.60)

    def setup(self):
        s = self.study
        self.x = scale_example()
        self.ds = np.asarray(s.panels[0].data["x"], float)
        self.h, self.hs = np.asarray(s.panels[0].data["ys"], float)
        self.g = np.asarray(s.panels[3].data["ys"], float)[0]
        self.gs = self.x["gs"]
        self.keys = self.x["raw"].shape[1]
        self.queries = 128
        self.steps = np.asarray(s.panels[1].data["x"], float)
        self.traces = np.asarray(s.panels[1].data["ys"], float)
        self.heldout = [float(v) for v in s.metadata["heldout_loss"]]
        self.updates = int(s.metadata["steps"])
        config = s.metadata["runs"][0]["settings"]["config"]
        self.head = int(config["width"]) // int(config["heads"])

    def subtitle_for(self, k):
        x, d = self.x, int(self.ds[-1])
        return [
            "a score is a dot product, a sum of d products, so random scores spread out like √d: "
            "%.1f at d = %d, %.1f at d = %d" % (x["spread"][0], self.ds[0], x["spread"][-1], d),
            "a softmax of widely spread scores puts nearly everything on one key; dividing the "
            "scores by √d first keeps the attention spread at every d",
            "entropy measures how spread out the attention is: unscaled it falls from %.2f to "
            "%.2f as d grows to %d, scaled it stays near %.1f" % (
                self.h[0], self.h[-1], d, self.hs.mean()),
            "learning signal passes through a weight p in proportion to p(1 − p); summed over "
            "the keys that is %.2f unscaled at d = %d, %.2f scaled" % (
                self.g[-1], d, self.gs[-1]),
            "trained on Tiny Shakespeare with heads %d wide: held-out loss %.2f nats with the "
            "division, %.2f without" % (self.head, self.heldout[0], self.heldout[1]),
        ][k]

    def phase_1(self, sc, u, op):
        x = self.x
        k, w = stage(u, len(self.ds), 1.03)
        d = int(self.ds[k])
        x0, x1 = self.BARS
        base, unit = 0.62, 0.28 / float(abs(x["scores"]).max())
        vbars(self, sc, x0, x1, base, x["scores"][k] * smoothstep(0.0, 0.3, w), unit, RED, op)
        self.polyline(sc, [self.P(x0, base), self.P(x1, base)], 1.2, AXIS, op)
        for sign in (1, -1):
            yy = base - sign * np.sqrt(d) * unit
            self.polyline(sc, [self.P(x0, yy), self.P(x1, yy)], 1.6, AMBER, op)
        self.text(sc, x1 + 0.012, base - np.sqrt(d) * unit - 0.016, "+√d", op, anchor="lm",
                  ink=dark(AMBER, 0.3))
        self.text(sc, x1 + 0.012, base + np.sqrt(d) * unit + 0.016, "−√d", op, anchor="lm",
                  ink=dark(AMBER, 0.3))
        self.text(sc, x0, 0.30, "one query's scores against %d keys (random vectors)" %
                  self.keys, op, anchor="lm", strong=True, size=0.032)
        self.text(sc, 0.72, 0.42, "d = %d" % d, op, anchor="lm", ink=dark(RED), size=0.05)
        self.lines(sc, 0.72, 0.52, [
            ("numbers in each vector", None),
            ("spread of the scores: %.1f" % x["spread"][k], dark(RED)),
            ("√d = %.1f" % np.sqrt(d), dark(AMBER, 0.3))], op, size=0.032, step=0.06)

    def _weights(self, sc, op, k, x0, x1, bases, height, labels=True, of=lambda p: p):
        """Query 1's attention over the keys at width ``ds[k]``: from the raw
        scores, and from the scores divided by the root of the width. ``of``
        turns each weight into what is drawn."""
        for base, key, ink, name in zip(bases, ("raw", "scaled"), (RED, BLUE),
                                        ("raw scores", "scores ÷ √d")):
            vbars(self, sc, x0, x1, base, of(self.x[key][k]), height, ink, op)
            self.polyline(sc, [self.P(x0, base), self.P(x1, base)], 1.2, AXIS, op)
            if labels:
                self.text(sc, x0, base - height - 0.04, "attention from the %s" % name, op,
                          anchor="lm", ink=dark(ink), size=0.032)

    def phase_2(self, sc, u, op):
        x = self.x
        k, _w = stage(u, len(self.ds), 1.03)
        d = int(self.ds[k])
        self._weights(sc, op, k, *self.BARS, (0.56, 0.86), 0.19)
        self.text(sc, 0.66, 0.33, "d = %d" % d, op, anchor="lm", size=0.05, strong=True)
        for base, key, ink, entropy in ((0.56, "raw", RED, self.h[k]),
                                        (0.86, "scaled", BLUE, self.hs[k])):
            self.lines(sc, 0.66, base - 0.13, [
                ("largest weight  %.2f" % x[key][k].max(), dark(ink)),
                ("entropy, all %d queries  %.2f" % (self.queries, entropy), dark(ink))],
                op, size=0.034, step=0.055)
        self.lines(sc, 0.66, 0.60, [
            "entropy: 0 when one key takes it all, %.2f when all %d share evenly" % (
                np.log(self.keys), self.keys)], op, width=0.28, step=0.04)

    def _beside(self, sc, op, u, what, height=0.15, of=lambda p: p):
        """The weights at the width a chart has been drawn to, beside it."""
        k = int(round(min(u / 0.8, 1.0) * (len(self.ds) - 1)))
        self._weights(sc, op, k, 0.64, 0.94, (0.58, 0.84), height, labels=False, of=of)
        self.text(sc, 0.64, 0.325, "%s at d = %d" % (what, self.ds[k]), op, anchor="lm",
                  strong=True)
        return k

    def phase_3(self, sc, u, op):
        even = float(np.log(self.keys))
        chart = self.chart(((0.10, 0.57), (0.37, 0.84)), self.ds,
                           np.concatenate([self.h, self.hs, [0.0, even]]), xlog=True)
        ticks = [(v, "%d" % v) for v in self.ds[::2]]
        self.axes(sc, chart, op, "d: numbers in each vector", "entropy of the attention",
                  xticks=ticks)
        chart.hline(sc, self.P, even, op)
        self.text(sc, 0.56, chart.fy(even) - 0.03, "even spread: %.2f" % even, op, anchor="rm")
        for ys, ink in ((self.h, RED), (self.hs, BLUE)):
            self.plot(sc, chart, self.ds, ys, ink, op, u / 0.8)
        k = self._beside(sc, op, u, "query 1's attention")
        own = [float(-(p * np.log(p + 1e-30)).sum()) for p in (self.x["raw"][k],
                                                               self.x["scaled"][k])]
        self.text(sc, 0.64, 0.37, "its entropy: %.2f unscaled, %.2f scaled" % tuple(own), op,
                  anchor="lm")

    def phase_4(self, sc, u, op):
        chart = self.chart(((0.10, 0.57), (0.37, 0.84)), self.ds,
                           np.concatenate([self.g, self.gs, [0.0, 1.0]]), xlog=True)
        self.axes(sc, chart, op, "d: numbers in each vector", "sum of p(1 − p) over the keys",
                  xticks=[(v, "%d" % v) for v in self.ds[::2]])
        for ys, ink in ((self.g, RED), (self.gs, BLUE)):
            self.plot(sc, chart, self.ds, ys, ink, op, u / 0.8)
        k = self._beside(sc, op, u, "query 1's p(1 − p)", 0.15 / 0.25, lambda p: p * (1 - p))
        self.text(sc, 0.64, 0.37, "its sum: %.2f unscaled, %.2f scaled" % (
            float((self.x["raw"][k] * (1 - self.x["raw"][k])).sum()),
            float((self.x["scaled"][k] * (1 - self.x["scaled"][k])).sum())), op, anchor="lm")
        self.text(sc, 0.64, 0.905, "a weight near 0 or 1 gives nearly 0", op, anchor="lm")

    def phase_5(self, sc, u, op):
        self.text(sc, 0.10, 0.335, "one small transformer trained each way; its heads are %d "
                  "wide, so √d = %s" % (self.head, number(np.sqrt(self.head))), op, anchor="lm",
                  strong=True)
        training(self, sc, u / 0.6, op, ((0.10, 0.55), (0.42, 0.84)), self.steps, self.traces,
                 self.NAMES, self.INK, (0.63, 0.40))
        show = smoothstep(0.55, 0.65, u)
        worst = max(self.heldout)
        full = self.span(0.19)[0]
        self.text(sc, 0.63, 0.60, "loss on held-out text", op * show, anchor="lm", strong=True)
        for row, (name, ink, value) in enumerate(zip(self.NAMES, self.INK, self.heldout)):
            self.hbar(sc, 0.725, 0.67 + row * 0.075, full * value / worst, ink, op * show, name,
                      "%.2f" % value, (u - 0.55) / 0.2, thick=28.0, track=full)
        gain = self.heldout[1] - self.heldout[0]
        self.lines(sc, 0.63, 0.84, [
            "dividing was %.2f nats %s; at d = %d it only divides the scores by %s" % (
                abs(gain), "better" if gain > 0 else "worse", self.head,
                number(np.sqrt(self.head)))],
            op * smoothstep(0.75, 0.9, u), width=0.31, step=0.042)


ANIMATIONS = {
    "sliding-window-attention": SlidingWindow,
    "sparse-attention-patterns": SparsePatterns,
    "linear-attention": LinearAttention,
    "flash-attention-tiling": FlashTiling,
    "attention-sinks": AttentionSinks,
    "attention-temperature": ScoreScale,
}

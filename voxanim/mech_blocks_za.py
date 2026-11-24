"""Animations 261-262, the last two of "Hosting a language model" in "Blocks and code" (see
:mod:`voxanim.blocks`), drawn from :mod:`voxanim.study_blocks_za`: what "thinking" and
"effort" are, on a model small enough to see (``examples/thinking.py``).

* :class:`ThinkingTokens` (``thinking-tokens``): a held-out bit string as a row of tokens
  (the effort, 12 bits, "="), which stays at the top all through. The model, one orange
  card, reads every token so far and writes one token a pass: at effort 0 the answer at
  once, a guess, wrong here; at effort 12 a step under each bit (the parity so far), each
  read back by the next pass, then the answer, right: 13 tokens, 13 passes. Over the 468
  strings it never saw: 465 right, against 215 (46%) with no steps. Then a large served
  model's tokens: the question, thinking, the answer; thinking is written, read back and
  counted like any token, though often not shown to the person.
* :class:`ThinkingEffort` (``thinking-effort``): what effort 0, 3 and 12 ask the model to
  write (``example`` of the script); one pass of the whole model per token; then right
  answers (up, blue) and tokens written (down, magenta) at every effort, two bars each:
  about chance up to 6 steps, 465 of 468 at 12; then what carries over to large served
  models, and what this toy cannot show.

Colours: the question's bits are data (blue tiles); the tokens the model writes are magenta;
the model is orange (its numbers are weights). Nothing here is training: the loop map lights
no station. A pass is counted, never timed.
"""

from __future__ import annotations

import numpy as np

from . import layout
from .blocks import (
    DATA,
    HEAD,
    INK,
    QUIET,
    SIZE,
    WEIGHT,
    Blocks,
    Code,
    about_data,
    exact,
    measured,
)
from .easing import ease_in_out, smoothstep
from .mech import AXIS, MAGENTA, PAPER, dark, soft

ON = "a tiny transformer"
DATA_INK = dark(DATA, 0.3)
WRITE = MAGENTA                      # tokens the model writes
WRITE_INK = dark(MAGENTA, 0.3)
MODEL_INK = dark(WEIGHT, 0.3)

COL0, COLS_W, NCOLS = 0.065, 0.4, 14   # the effort, 12 bits, "=": a column each
BOX = 23.0                           # px: a token's box


def col(j):
    """The centre x of column ``j`` (0: the effort; 1-12: the bits; 13: "=")."""
    return COL0 + COLS_W * (j + 0.5) / NCOLS


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def on(u, a, op, width=0.05):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


def written_col(effort, j):
    """The column of token ``j`` that the model writes at ``effort``: step ``j`` sits under
    the last bit it covers; the answer under "="."""
    if j >= effort:
        return NCOLS - 1
    return (j + 1) * (12 // effort)


class Thinking(Blocks):
    """What the two share: tokens, the model card, a laptop, a server."""

    def step_indicator(self, sc, current, phases):
        """The phase under way by name, the others by number: the titles are long."""
        items = ["%d %s" % (n + 1, name) if n == current else "%d" % (n + 1)
                 for n, name in enumerate(phases)]
        size, gap = 0.030, 0.016
        widths = [layout.text_width(s, size) for s in items]
        x = (1 - layout.SAFE) - (sum(widths) + gap * (len(items) - 1)) - 0.004
        for n, (s, w) in enumerate(zip(items, widths, strict=True)):
            ink = (0.12, 0.36, 0.66) if n == current else layout.INK_SOFT if n < current \
                else (0.56, 0.58, 0.60)
            sc.text(x, layout.SAFE + 0.012, s, size, "regular", ink, 1.0, "la")
            x += w + gap

    def at(self, k, u):
        """1 once phase ``k`` has reached ``u`` (fading in over half a second); 0 before."""
        a, b = self.spans[k]
        t0 = a + (b - a) * u
        return smoothstep(t0, t0 + 0.5, self._t)

    def token(self, sc, fx, fy, text, kind, op, w=BOX, h=BOX, ring=False):
        """One token's box: ``kind`` "bit" (blue), "effort" (grey, bold), "equals",
        "step" (magenta), "answer" (magenta, a strong border), "thinking" (pale magenta)."""
        if op <= 0.01:
            return
        if kind == "bit":
            fill, border, ink = soft(DATA, 0.8), DATA, DATA_INK
        elif kind in ("effort", "equals"):
            fill, border, ink = PAPER, AXIS, INK
        elif kind == "thinking":
            fill, border, ink = soft(WRITE, 0.86), soft(WRITE, 0.35), WRITE_INK
        else:
            fill, border, ink = soft(WRITE, 0.72), WRITE, WRITE_INK
        bw = 2.6 if kind == "answer" or ring else 1.5
        self.rect(sc, self.P(fx, fy), w, h, fill, op, border=INK if ring else border,
                  border_w=bw, radius=4, lift=0.3)
        if text:
            self.text(sc, fx, fy, text, op, SIZE, ink=ink)

    def model_card(self, sc, fx, fy, w, h, op, label="the model", sub=None, tiles=None):
        """The model: an orange card (its numbers are weights), named, with a strip of
        its real numbers along the bottom."""
        if op <= 0.01:
            return
        c = self.P(fx, fy)
        self.rect(sc, c, w, h, soft(WEIGHT, 0.82), op, border=WEIGHT, border_w=2.0, radius=7,
                  lift=0.2)
        dy = self.frac(0, h)[1]
        if label:
            self.text(sc, fx, fy - dy * 0.27, label, op, SIZE, strong=True)
        if sub:
            self.text(sc, fx, fy + dy * 0.02, sub, op, SIZE, ink=MODEL_INK)
        if tiles is not None:
            self.tiles(sc, tiles, fx, fy + dy * 0.3, w - 40, 10.0, op, ink=WEIGHT,
                       lo=0.0, hi=float(np.abs(tiles).max()), grid=0.0, lift=0.24)

    def mini_model(self, sc, fx, fy, op, w=18.0, h=20.0):
        """One pass: a small orange card, the same model run once."""
        if op <= 0.01:
            return
        c = self.P(fx, fy)
        self.rect(sc, c, w, h, soft(WEIGHT, 0.7), op, border=WEIGHT, border_w=1.6, radius=3,
                  lift=0.25)
        for dy in (-4.0, 0.0, 4.0):
            self.rect(sc, self.shift(c, 0, dy), w - 8, 1.6, MODEL_INK, op * 0.7, radius=0.5,
                      lift=0.27)

    def laptop(self, sc, fx, fy, op, w=70.0, h=46.0):
        """A laptop: a screen on a thin base. Returns the screen's layout half-size."""
        if op <= 0.01:
            return
        c = self.P(fx, fy)
        self.rect(sc, c, w, h, PAPER, op, border=INK, border_w=2.0, radius=4, lift=0.2)
        self.rect(sc, self.shift(c, 0, h / 2 + 4), w + 18, 6, INK, op, radius=3, lift=0.2)

    def server(self, sc, fx, fy, op, w=56.0, h=74.0):
        """A server: a tall box of a computer, with three drive bays."""
        if op <= 0.01:
            return
        c = self.P(fx, fy)
        self.rect(sc, c, w, h, soft(AXIS, 0.8), op, border=INK, border_w=2.0, radius=5,
                  lift=0.2)
        for dy in (-22.0, -8.0, 6.0):
            self.rect(sc, self.shift(c, 0, dy), w - 14, 8, PAPER, op, border=AXIS,
                      border_w=1.0, radius=2, lift=0.22)
        self.disk(sc, self.shift(c, w / 2 - 12, h / 2 - 12), 3.0, soft(DATA, 0.2), op,
                  lift=0.24, outline=0.0)


# ====================================================================== 261
class ThinkingTokens(Thinking):
    """The question as tokens; the model writes a token a pass, each read back; effort 0 a
    guess, effort 12 right; then the same in a large served model."""

    TITLE = "Thinking: tokens it writes for itself"
    PHASES = (("the question", 11.0), ("no steps", 11.0), ("12 steps", 14.0),
              ("the answer", 13.0), ("large models", 12.0))
    LAST = "A coding agent's loop"
    NEXT = "Effort: how much thinking is allowed"
    STATION = (None, None, None, None, None)
    LABEL_Y, Q_Y, W_Y = 0.39, 0.435, 0.495
    MODEL = (0.2, 0.665)
    MODEL_W, MODEL_H = 192.0, 66.0
    READ_X, WRITE_X = 0.14, 0.26
    ARROW_Y = (0.526, 0.6)              # the two arrows' ends, the rows' side and the card's
    COUNT_X = 0.325
    LOW = (0.775, 0.815, 0.852, 0.888)  # the lines under the model
    EFFORT_SWAP = (2, 0.03, 0.09)       # phase, u: the effort token turns from 0 to 12

    def setup(self):
        m = self.meta = self.study.metadata
        self.bits = m["bits"]
        self.deep, self.none = m["deep"], m["none"]
        # the passes, in order: (phase, u from, u to, effort, j: the token written)
        p = [(1, 0.2, 0.52, 0, 0)]
        p += [(2, 0.13 + 0.2 * j, 0.31 + 0.2 * j, 12, j) for j in range(4)]
        p += [(3, 0.03 + 0.062 * (j - 4), 0.086 + 0.062 * (j - 4), 12, j) for j in range(4, 13)]
        self.passes = p
        from . import study_thinking as T
        w = T.params()["head.weight"][:2]
        self.weights = np.abs(np.asarray(w, float))[:, :40]
        self.CODE = (
            Code("th-generate", run=((0.0, 0),)),
            Code("th-generate", run=((0.0, 0), (0.18, 1), (0.22, 2), (0.36, 3), (0.46, 4),
                                     (0.6, None))),
            Code("th-generate", run=((0.0, 0), (0.1, 1), (0.13, 2), (0.22, 3), (0.27, 4),
                                     (0.33, 2), (0.42, 3), (0.47, 4), (0.53, 2), (0.62, 3),
                                     (0.67, 4), (0.73, 2), (0.82, 3), (0.87, 4))),
            Code("th-generate", run=((0.0, 2), (0.6, 4))),
            Code("th-generate", run=((0.0, 4),)))
        i = m["index"]
        self.EVIDENCE = (about_data("held-out string %d of 468, never trained on" % i),
                         measured("string %d at effort 0" % i, on=ON),
                         measured("string %d at effort 12" % i, on=ON),
                         measured("string %d, then all 468" % i, on=ON),
                         about_data("large served models, described"))

    def heading_for(self, k, u):
        return ("a question, as tokens",
                "effort 0: the answer at once",
                "effort 12: it writes its steps first",
                "each step read back: then the answer",
                "large models think in tokens too")[k]

    def subtitle_for(self, k):
        m = self.meta
        return [
            "The tiny model is asked for the parity of 12 bits it never saw in training: 1 "
            "if the count of ones is odd.",
            "With effort 0 it must answer at once: one pass through the model, one token. "
            "It writes %d: wrong." % self.none["tokens"][-1],
            "With effort 12 it first writes a step under each bit, the parity so far. Each "
            "pass reads every token before it.",
            "Its 12 steps and its answer are right: 13 tokens, 13 passes. Over 468 new "
            "strings, %d right; with no steps, %d (%s)." % (m["right12"], m["right0"],
                                                           m["pct0"]),
            "A large served model's thinking is the same kind of thing: tokens written "
            "before the answer, read back, and counted.",
        ][k]

    # ------------------------------------------------------------------ the passes
    def pass_now(self, k, u):
        """``(index, f)`` of the pass under way or last made by ``(k, u)`` (``f`` 0..1 is
        how far it has gone; past its end, > 1), or None before the first."""
        last = None
        for n, (pk, a, b, _e, _j) in enumerate(self.passes):
            if pk < k or (pk == k and u >= a):
                f = (u - a) / (b - a) if pk == k else 2.0
                last = (n, f)
        return last

    def shown(self, effort, j, k, u):
        """How visible written token ``j`` of ``effort`` is at ``(k, u)``."""
        for pk, a, b, e, jj in self.passes:
            if e == effort and jj == j:
                if k < pk:
                    return 0.0
                f = (u - a) / (b - a) if k == pk else 2.0
                op = smoothstep(0.82, 0.95, f)
                if effort == 0:                      # gone once effort 12 begins
                    if k > 2:
                        return 0.0
                    if k == 2:
                        op *= 1.0 - smoothstep(0.0, 0.04, u)
                return op
        return 0.0

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        a = op * (self.at(0, 0.03) if k == 0 else 1.0)
        self.text(sc, COL0, self.LABEL_Y, "effort", a, SIZE, "lm", ink=QUIET)
        self.text(sc, (col(1) + col(12)) / 2 + 0.012, self.LABEL_Y,
                  "12 bits it never saw", a, SIZE, ink=DATA_INK)
        # the effort token turns from 0 to 12 when effort 12 begins
        pk, a0, a1 = self.EFFORT_SWAP
        mid = (a0 + a1) / 2                 # out, then in: never both at once
        out = 0.0 if k < pk else (smoothstep(a0, mid, u) if k == pk else 1.0)
        into = 1.0 if k > pk else (smoothstep(mid, a1, u) if k == pk else 0.0)
        self.token(sc, col(0), self.Q_Y, "0", "effort", a * (1.0 - out))
        self.token(sc, col(0), self.Q_Y, "12", "effort", a * into)
        ones = k == 0 and u > 0.45
        for j, b in enumerate(self.bits):
            self.token(sc, col(j + 1), self.Q_Y, str(b), "bit",
                       op * (self.at(0, 0.06 + 0.012 * j) if k == 0 else 1.0),
                       ring=ones and b == 1)
        self.token(sc, col(13), self.Q_Y, "=", "equals",
                   op * (self.at(0, 0.24) if k == 0 else 1.0))
        # what the model has written
        self.token(sc, col(13), self.W_Y, str(self.none["tokens"][0]), "answer",
                   op * self.shown(0, 0, k, u))
        for j, v in enumerate(self.deep["tokens"]):
            kind = "answer" if j == 12 else "step"
            self.token(sc, col(written_col(12, j)), self.W_Y, str(v), kind,
                       op * self.shown(12, j, k, u))
        keep = op * self.kept(1, 3)
        if keep > 0.01:
            self.machine(sc, k, u, keep)

    def machine(self, sc, k, u, op):
        """The model card, the arrows in and out, the pulses of the pass under way, and
        the count of passes."""
        m = self.meta
        a = op * (self.at(1, 0.04) if k == 1 else 1.0)
        mx, my = self.MODEL
        self.model_card(sc, mx, my, self.MODEL_W, self.MODEL_H, a,
                        sub="%s numbers" % m["parameters_text"], tiles=self.weights)
        y0, y1 = self.ARROW_Y
        b = op * (self.at(1, 0.1) if k == 1 else 1.0)
        self.arrow(sc, (self.READ_X, y0), (self.READ_X, y1), AXIS, b, width=1.8)
        self.arrow(sc, (self.WRITE_X, y1), (self.WRITE_X, y0), WRITE, b, width=1.8)
        self.text(sc, self.READ_X - 0.01, (y0 + y1) / 2, "reads", b, SIZE, "rm")
        self.text(sc, self.WRITE_X + 0.01, (y0 + y1) / 2, "writes", b, SIZE, "lm",
                  ink=WRITE_INK)
        now = self.pass_now(k, u)
        if now is None:
            return
        n, f = now
        _pk, _a, _b, effort, j = self.passes[n]
        if 0.0 < f < 1.0:
            self.dot(sc, self.P(self.READ_X, y0), self.P(self.READ_X, y1),
                     float(np.clip(f / 0.4, 0, 1)), AXIS, r=4.5, opacity=op)
            self.dot(sc, self.P(self.WRITE_X, y1), self.P(self.WRITE_X, y0),
                     float(np.clip((f - 0.42) / 0.26, 0, 1)), WRITE, r=4.5, opacity=op)
            self.dot(sc, self.P(self.WRITE_X, y0), self.P(col(written_col(effort, j)), self.W_Y),
                     float(np.clip((f - 0.68) / 0.22, 0, 1)), WRITE, r=4.5, opacity=op)
        c = op * self.at(1, self.passes[0][1])
        self.text(sc, self.COUNT_X, 0.64, "pass %d" % (j + 1), c, HEAD, "lm", strong=True)
        self.text(sc, self.COUNT_X, 0.68, "reads %d tokens" % (14 + j), c, SIZE, "lm")

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        m = self.meta
        self.text(sc, COL0, 0.56, "answer: 1 if the count of 1s is odd", on(u, 0.3, op),
                  SIZE, "lm")
        self.text(sc, COL0, 0.61, "this string: %d ones, odd: %d" % (m["ones"], m["answer"])
                  if m["answer"] else "this string: %d ones, even: 0" % m["ones"],
                  on(u, 0.5, op), HEAD, "lm", strong=True)
        # the row the model's own tokens will fill, empty for now
        c = on(u, 0.68, op)
        self.rect(sc, self.P((col(1) + col(13)) / 2, self.W_Y), self.span(col(13) - col(1))[0]
                  + BOX + 6, BOX + 6, PAPER, c * 0.6, border=soft(WRITE, 0.3), border_w=1.4,
                  radius=6, lift=0.15, fill_op=0.0)
        self.text(sc, col(1) - 0.005, self.W_Y, "the tokens it writes will go here", c, SIZE,
                  "lm", ink=WRITE_INK)

    def phase_2(self, sc, u, op):
        m = self.meta
        got = self.none["tokens"][-1]
        self.text(sc, COL0, self.LOW[0], "answer %d: wrong" % got, on(u, 0.56, op), HEAD,
                  "lm", strong=True)
        self.text(sc, COL0, self.LOW[1], "%d ones is %s: %d" % (m["ones"], "odd" if
                  m["answer"] else "even", m["answer"]), on(u, 0.64, op), SIZE, "lm")
        self.text(sc, COL0, self.LOW[2], "1 token written · 1 pass", on(u, 0.72, op), SIZE,
                  "lm", ink=WRITE_INK)

    def phase_3(self, sc, u, op):
        self.text(sc, COL0, self.LOW[0], "a step: 1 if the ones so far are odd",
                  on(u, 0.3, op), SIZE, "lm", ink=WRITE_INK)
        self.text(sc, COL0, self.LOW[1], "each step is read back next pass", on(u, 0.55, op),
                  SIZE, "lm")

    def phase_4(self, sc, u, op):
        m = self.meta
        self.text(sc, COL0, self.LOW[0], "answer %d: right" % self.deep["tokens"][-1],
                  on(u, 0.6, op), HEAD, "lm", strong=True)
        self.text(sc, COL0, self.LOW[1], "13 tokens written · 13 passes", on(u, 0.65, op),
                  SIZE, "lm", ink=WRITE_INK)
        self.text(sc, COL0, self.LOW[2], "all 468 new strings: 12 steps, %d right"
                  % m["right12"], on(u, 0.74, op), SIZE, "lm")
        self.text(sc, COL0, self.LOW[3], "no steps: %d right (%s, chance)"
                  % (m["right0"], m["pct0"]), on(u, 0.8, op), SIZE, "lm")

    def phase_5(self, sc, u, op):
        a = on(u, 0.04, op)
        self.text(sc, COL0, 0.585, "a large served model", a, SIZE, "lm", strong=True)
        y, pitch, w = 0.635, 0.026, 19.0
        groups = (("question", "bit", 4, 0.1), ("thinking", "thinking", 8, 0.22),
                  ("answer", "answer", 3, 0.36))
        x = COL0 + pitch / 2
        for name, kind, n, at in groups:
            g = on(u, at, op)
            xs = [x + pitch * i for i in range(n)]
            for fx in xs:
                self.token(sc, fx, y, "", kind, g, w=w, h=w)
            self.text(sc, (xs[0] + xs[-1]) / 2, y + 0.045, name, g, SIZE,
                      ink=WRITE_INK if kind != "bit" else DATA_INK)
            x = xs[-1] + pitch * 1.4
        think = (COL0 + pitch / 2 + pitch * 5.4, COL0 + pitch / 2 + pitch * 12.4)
        self.text(sc, (think[0] + think[1]) / 2, 0.725, "written, read back, counted",
                  on(u, 0.48, op), SIZE, strong=True)
        # what the person sees: often only the question and the answer
        b = on(u, 0.62, op)
        lx, ly = 0.115, 0.82
        self.laptop(sc, lx, ly, b)
        for i, kind in enumerate(("bit", "bit", "answer")):
            self.token(sc, lx - 0.022 + 0.022 * i, ly, "", kind, b, w=14.0, h=14.0)
        self.text(sc, 0.18, ly - 0.018, "thinking: often", b, SIZE, "lm")
        self.text(sc, 0.18, ly + 0.022, "not shown to you", b, SIZE, "lm")


# ====================================================================== 262
class ThinkingEffort(Thinking):
    """What effort 0, 3 and 12 ask for; a pass per token; right answers and tokens at
    every effort; what carries over to large served models."""

    TITLE = "Effort: how much thinking is allowed"
    PHASES = (("what effort asks", 11.0), ("a pass a token", 11.0), ("every effort", 14.0),
              ("the cliff", 9.0), ("large models", 15.0))
    LAST = "Thinking: tokens it writes for itself"
    NEXT = "The cost of context: attention, RNN, SSM"
    STATION = (None, None, None, None, None)
    LABEL_Y, Q_Y = 0.39, 0.435
    ROWS_Y = (0.5, 0.555, 0.61)          # efforts 0, 3, 12
    ROWS_AT = (0.1, 0.36, 0.62)
    PASS_Y = (0.69, 0.735, 0.78)
    PASS_AT = (0.12, 0.3, 0.48)
    BASE, UP, DOWN = 0.665, 0.205, 0.12  # the chart: its baseline, and the bars' full lengths
    BAR_X0, BAR_DX = 0.15, 0.052
    GROUP_AT = (0.1, 0.18, 0.26, 0.34, 0.42, 0.5, 0.6)

    def setup(self):
        m = self.meta = self.study.metadata
        self.bits = m["bits"]
        self.rows = m["shown"]
        self.res = m["results"]
        lines = ((0.0, 0), (0.08, 1), (0.36, 3), (0.44, 4), (0.52, 5), (0.7, 6))
        self.CODE = (
            Code("th-example", run=lines),
            Code("th-generate", run=((0.0, 1), (0.5, 2))),
            Code("th-generate", run=((0.0, 1),)),
            Code("th-generate", run=((0.0, 1),)),
            Code("th-generate", run=((0.0, 1),)))
        self.EVIDENCE = (exact("what example() asks at efforts 0, 3, 12"),
                         exact("effort k: k + 1 tokens, k + 1 passes"),
                         measured("468 held-out strings", on=ON),
                         measured("one run, 468 strings", on=ON),
                         about_data("large served models, described"))

    def heading_for(self, k, u):
        return ("effort: how many steps it may write",
                "one pass through the model per token",
                "right answers and tokens, every effort",
                "about chance, then right",
                "large models: the same, and not")[k]

    def subtitle_for(self, k):
        m = self.meta
        r = self.res
        return [
            "The effort token says how many steps the model may write before its answer: "
            "0, 1, 2, 3, 4, 6 or 12.",
            "Each token written is one more pass through the whole model, all %s numbers: "
            "effort k costs k + 1 passes." % m["parameters_text"],
            "On 468 strings it never saw: right answers go up, tokens written go down. Half "
            "right, %d, is chance." % m["chance"],
            "Up to 6 steps it stays near chance, %d to %d right. One bit a step: %d. One "
            "run: the cliff is this model's." % (m["low"], m["high"], r[-1]["right"]),
            "A coding agent can ask for more or less effort per request. More thinking costs "
            "more tokens; how much it helps varies.",
        ][k]

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        keep = op * self.until(1, k, u)
        if keep > 0.01:
            self.asks(sc, k, u, keep)
        chart = op * self.until(3, k, u) * (k >= 2)
        if chart > 0.01:
            self.chart_bars(sc, k, u, chart)

    def until(self, last, k, u):
        """1 through phase ``last``, then gone in the first moment of the next phase, just
        before what replaces it fades in (so the picture never stands empty)."""
        if k <= last:
            return 1.0
        return 1.0 - smoothstep(0.0, 0.02, u) if k == last + 1 else 0.0

    def asks(self, sc, k, u, op):
        """The question's bits, and under them what efforts 0, 3 and 12 write."""
        a = op * (self.at(0, 0.02) if k == 0 else 1.0)
        self.text(sc, COL0, self.LABEL_Y, "effort", a, SIZE, "lm", ink=QUIET)
        self.text(sc, (col(1) + col(12)) / 2 + 0.012, self.LABEL_Y, "the same 12 bits", a,
                  SIZE, ink=DATA_INK)
        for j, b in enumerate(self.bits):
            self.token(sc, col(j + 1), self.Q_Y, str(b), "bit", a)
        self.token(sc, col(13), self.Q_Y, "=", "equals", a)
        for row, y, at in zip(self.rows, self.ROWS_Y, self.ROWS_AT, strict=True):
            e = row["effort"]
            r = op * (self.at(0, at) if k == 0 else 1.0)
            self.token(sc, col(0), y, str(e), "effort", r)
            for j, v in enumerate(row["truth"]):
                self.token(sc, col(written_col(e, j)), y, str(v),
                           "answer" if j == e else "step", r)

    def chart_bars(self, sc, k, u, op):
        """Right answers up (blue) and tokens written down (magenta), one effort each."""
        m = self.meta
        held = m["held"]
        most = m["most_tokens"]
        a = op * (self.at(2, 0.012) if k == 2 else 1.0)
        # the legend, the baseline, the chance line
        self.rect(sc, self.P(COL0 + 0.008, 0.395), 14, 14, DATA, a, radius=3, lift=0.3)
        self.text(sc, COL0 + 0.022, 0.395, "right, of 468", a, SIZE, "lm", ink=DATA_INK)
        self.rect(sc, self.P(0.27, 0.395), 14, 14, WRITE, a, radius=3, lift=0.3)
        self.text(sc, 0.284, 0.395, "tokens written", a, SIZE, "lm", ink=WRITE_INK)
        x0, x1 = self.BAR_X0 - 0.03, self.BAR_X0 + self.BAR_DX * 6 + 0.03
        self.polyline(sc, [self.P(x0, self.BASE, 1.3), self.P(x1, self.BASE, 1.3)], 1.4, AXIS,
                      a, bias=0.4)
        # chance: a short dashed tick at the axis's left end (the bars sit on it)
        cy = self.BASE - self.UP * m["chance"] / held
        for xs in np.arange(x0 - 0.002, x0 + 0.028, 0.012):
            self.polyline(sc, [self.P(xs, cy, 1.3), self.P(xs + 0.007, cy, 1.3)], 1.6, AXIS,
                          a, bias=0.4)
        self.text(sc, x0 - 0.008, cy, "chance", a, SIZE, "rm", ink=QUIET)
        self.text(sc, COL0, 0.85, "steps", a, SIZE, "lm", ink=QUIET)
        for i, r in enumerate(self.res):
            g = op * (self.at(2, self.GROUP_AT[i]) if k == 2 else 1.0)
            if g <= 0.01:
                continue
            grow = eased(self._t, *self._span(2, self.GROUP_AT[i])) if k == 2 else 1.0
            x = self.BAR_X0 + self.BAR_DX * i
            hi = self.UP * r["right"] / held * grow
            lo = self.DOWN * r["tokens"] / most * grow
            dim = k == 3 and r["effort"] != 12
            ink = DATA if not dim else soft(DATA, 0.45)
            self.rect(sc, self.P(x, self.BASE - hi / 2), 24, self.span(0, hi)[1], ink, g,
                      radius=0, lift=0.3)
            self.rect(sc, self.P(x, self.BASE + lo / 2), 24, self.span(0, lo)[1], WRITE, g,
                      radius=0, lift=0.3)
            self.text(sc, x, self.BASE - hi - 0.02, str(r["right"]), g, SIZE, ink=DATA_INK)
            self.text(sc, x, self.BASE + lo + 0.022, str(r["tokens"]), g, SIZE, ink=WRITE_INK)
            self.text(sc, x, 0.85, str(r["effort"]), g, SIZE, strong=True)
            # phase 4: the bits each step must take in
            c = op * (self.at(3, 0.3) if k == 3 else 0.0 if k < 3 else 1.0)
            self.text(sc, x, 0.888, str(r["bits_per_step"]), c, SIZE,
                      ink=INK if r["effort"] == 12 else QUIET, strong=r["effort"] == 12)
        c = op * (self.at(3, 0.3) if k == 3 else 0.0 if k < 3 else 1.0)
        self.text(sc, COL0, 0.888, "bits each", c, SIZE, "lm", ink=QUIET)

    def _span(self, k, at, width=0.06):
        a, b = self.spans[k]
        return a + (b - a) * at, a + (b - a) * (at + width)

    # ------------------------------------------------------------------ the phases
    def phase_1(self, sc, u, op):
        for row, at, y in zip(self.rows, self.ROWS_AT, (0.69, 0.735, 0.78), strict=True):
            e = row["effort"]
            n = len(row["truth"])
            if e == 0:
                s = "0 steps: 1 token, the answer"
            else:
                s = "%d steps of %d bit%s: %d tokens" % (e, 12 // e, "" if e == 12 else "s",
                                                         n)
            self.text(sc, COL0, y, s, on(u, at + 0.04, op), SIZE, "lm", ink=WRITE_INK)
        self.text(sc, COL0, 0.835, "a step: the parity so far", on(u, 0.8, op), SIZE, "lm")

    def phase_2(self, sc, u, op):
        m = self.meta
        for row, y, at in zip(self.rows, self.PASS_Y, self.PASS_AT, strict=True):
            e = row["effort"]
            n = len(row["truth"])
            self.text(sc, col(0), y, str(e), on(u, at, op), SIZE, strong=True)
            for j in range(n):
                g = on(u, at + 0.012 * j, op, 0.03)
                self.mini_model(sc, col(written_col(e, j)), y, g)
        self.text(sc, COL0, 0.835, "a pass: all %s numbers, once" % m["parameters_text"],
                  on(u, 0.62, op), SIZE, "lm", ink=MODEL_INK)
        self.text(sc, COL0, 0.875, "passes: " + ", ".join(str(len(r["truth"]))
                                                         for r in self.rows),
                  on(u, 0.72, op), SIZE, "lm", strong=True)

    def phase_3(self, sc, u, op):
        """The bars, in ``steady``."""

    def phase_4(self, sc, u, op):
        m = self.meta
        x0, x1 = self.BAR_X0 - 0.015, self.BAR_X0 + self.BAR_DX * 5 + 0.015
        y = 0.455
        a = on(u, 0.1, op)
        self.polyline(sc, [self.P(x0, y + 0.02, 1.3), self.P(x0, y, 1.3), self.P(x1, y, 1.3),
                           self.P(x1, y + 0.02, 1.3)], 1.6, INK, a, bias=0.4)
        self.text(sc, (x0 + x1) / 2, y - 0.025, "up to 6 steps: %d to %d" % (m["low"],
                                                                         m["high"]),
                  a, SIZE, strong=True)
        b = on(u, 0.45, op)
        x = self.BAR_X0 + self.BAR_DX * 6
        top = self.BASE - self.UP * self.res[-1]["right"] / m["held"]
        self.rect(sc, self.P(x, (top + self.BASE) / 2 - 0.012), 38,
                  self.span(0, self.BASE - top + 0.06)[1], INK, b, border=INK, border_w=2.2,
                  radius=4, lift=0.9, fill_op=0.0)

    def phase_5(self, sc, u, op):
        a = on(u, 0.03, op)
        lx, ly, sx = 0.1, 0.42, 0.43
        self.laptop(sc, lx, ly, a)
        self.text(sc, lx, ly + 0.08, "coding agent", a, SIZE)
        self.server(sc, sx, ly, a, h=64.0)
        self.text(sc, sx, ly + 0.09, "the service", a, SIZE)
        b = on(u, 0.1, op)
        self.arrow(sc, (lx + 0.055, ly + 0.01), (sx - 0.04, ly + 0.01), AXIS, b, width=1.8,
                   u=eased(u, 0.1, 0.16))
        # the request asks for an effort: less, then more
        less = 1.0 - smoothstep(0.23, 0.255, u)     # out, then in
        more = smoothstep(0.255, 0.28, u)
        cx = (lx + sx) / 2 + 0.008
        self.rect(sc, self.P(cx, ly - 0.035), 116, 26, PAPER, b, border=AXIS, border_w=1.4,
                  radius=4, lift=0.3)
        self.text(sc, cx, ly - 0.035, "effort: less", b * less, SIZE)
        self.text(sc, cx, ly - 0.035, "effort: more", b * more, SIZE, strong=True)
        # the thinking tokens it writes: few, then more
        ty = 0.555
        self.text(sc, COL0, ty, "thinking tokens", b, SIZE, "lm", ink=WRITE_INK)
        n = 3 + int(round(7 * more))
        for i in range(10):
            g = b * (1.0 if i < 3 else more) * (1.0 if i < n else 0.0)
            self.token(sc, 0.226 + 0.0235 * i, ty, "", "thinking", g, w=17.0, h=17.0)
        self.text(sc, COL0, 0.625, "the same in large models", on(u, 0.38, op), SIZE, "lm",
                  strong=True)
        rows = (("more tokens: more passes, time and cost", 0.44),
                ("helps when each step is one it can do", 0.52))
        for i, (s, at) in enumerate(rows):
            self.text(sc, COL0, 0.662 + 0.037 * i, s, on(u, at, op), SIZE, "lm")
        self.text(sc, COL0, 0.77, "what this toy cannot show", on(u, 0.64, op), SIZE, "lm",
                  strong=True)
        rows = (("by how much it helps", 0.7), ("that more always helps", 0.76))
        for i, (s, at) in enumerate(rows):
            self.text(sc, COL0, 0.807 + 0.037 * i, s, on(u, at, op), SIZE, "lm", ink=QUIET)


ANIMATIONS = {"thinking-tokens": ThinkingTokens, "thinking-effort": ThinkingEffort}

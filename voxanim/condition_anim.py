"""The prompt steers the distribution: one target token watched while the
context around the same question grows, on the real GPT-2 small.

    Name the capital of each country.          <- instruction      (stage 1)
    The capital of Japan is Tokyo.             <- example          (stage 2)
    The capital of Egypt is Cairo.             <- example          (stage 3)
    The capital of China is Beijing.           <- example          (stage 4)
    The capital of France is ___               <- the question     (stage 0: alone)

1. **bare prompt** -- the question alone: the next-token distribution is
   flat and "the" is on top; the target " Paris" has a few per cent,
2. **instruction** -- a line saying what the task is, prepended: a little
   more probability on Paris, still not the top token,
3. **examples** -- one, two, three worked examples (few-shot): the
   distribution reshapes around city names and Paris rises to the top;
   a curve tracks p(Paris) against the number of context tokens,
4. **elsewhere** -- a different instruction and different examples
   ("The capital of Japan is in Asia.") around the *same* question: now
   " in" takes almost all the probability and Paris nearly none.  The
   examples set the form of the answer, not just the fact,
5. **attention** -- back to the few-shot context: the last layer's attention
   from the last token to every context token, as a strip over the tokens
   (real weights).  The mean over the 12 heads parks most of its weight on
   the first token (the "attention sink" every GPT-2 head has); the head
   that reads the context most broadly is shown on its own.

Every number is GPT-2 small's (:mod:`voxanim.gpt2_probe`), computed once
and cached (:func:`trace`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import color as C
from . import gpt2_probe as P
from . import layout
from .cache import cached_npz
from .diagram import EDGE, Diagram
from .easing import ease_in_out, ease_out, remap, smoothstep
from .prob_bars import BAR, MARK, draw_bars, shown
from .text import num

# ------------------------------------------------------------------ the contexts

QUESTION = "The capital of France is"
TARGET = " Paris"
INSTRUCTION = "Name the capital of each country.\n"
EXAMPLES = ("The capital of Japan is Tokyo.\n", "The capital of Egypt is Cairo.\n",
            "The capital of China is Beijing.\n")
# the other conditioning: the same question after examples of another form
OTHER_INSTRUCTION = "Name the continent of each capital.\n"
OTHER_EXAMPLES = ("The capital of Japan is in Asia.\n", "The capital of Egypt is in Africa.\n",
                  "The capital of China is in Asia.\n")
TOP_K = 6
N_STAGES = 2 + len(EXAMPLES)            # bare, instruction, then one stage per example


def pieces(stage):
    """The lines of the context at ``stage`` (0 = the bare question), in order."""
    if stage == 0:
        return [QUESTION]
    return [INSTRUCTION] + list(EXAMPLES[:stage - 1]) + [QUESTION]


def other_pieces():
    """The lines of the other conditioning (instruction, examples, question)."""
    return [OTHER_INSTRUCTION] + list(OTHER_EXAMPLES) + [QUESTION]


@dataclass
class Stage:
    """One context and the distribution after it."""

    lines: list                 # the context as lines of (token string) lists
    n_tokens: int
    top: list                   # [(token string, probability)], most probable first
    p_target: float
    entropy: float              # bits

    @property
    def tokens(self):
        """Every token string in order."""
        return [t for line in self.lines for t in line]


def _split_lines(pieces_):
    """Tokenize the context and group the token strings by line (each piece
    of the context is one line; its newline token ends the line)."""
    ids = P.encode("".join(pieces_))
    toks = [P.decode([i]) for i in ids]
    lines, cur = [], []
    for s in toks:
        cur.append(s)
        if s.endswith("\n"):
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    return ids, lines


def _probe(pieces_):
    """Arrays for one context: its token ids, the top-k, p(target), the entropy,
    and the last layer's attention from the last token to every token."""
    ids, _lines = _split_lines(pieces_)
    logits, attn = P.forward(ids, attention=True)
    p = P.softmax(logits[-1])
    top = np.argsort(-p)[:TOP_K]
    return dict(ids=np.array(ids), top_ids=top, top_p=p[top],
                p_target=np.array(p[P.token_id(TARGET)]), entropy=np.array(P.entropy(p)),
                attn=attn[:, -1, :])


def trace(cache_dir=None):
    """``(stages, other, attn)``: the :class:`Stage` for each few-shot stage, the
    other conditioning's :class:`Stage`, and the last layer's attention
    ``(N_HEAD, T)`` from the last token of the final few-shot context.
    Cached by :func:`voxanim.cache.cached_npz`."""
    settings = dict(question=QUESTION, target=TARGET, instruction=INSTRUCTION, examples=EXAMPLES,
                    other=(OTHER_INSTRUCTION, OTHER_EXAMPLES), top_k=TOP_K)

    def build():
        out = {}
        for s in range(N_STAGES):
            for k, v in _probe(pieces(s)).items():
                out["s%d_%s" % (s, k)] = v
        for k, v in _probe(other_pieces()).items():
            out["other_" + k] = v
        return out

    d = cached_npz("gpt2-conditioning", settings, build, cache_dir)

    def stage(prefix, pieces_):
        g = {k[len(prefix):]: v for k, v in d.items() if k.startswith(prefix)}
        _ids, lines = _split_lines(pieces_)
        top = [(P.decode([i]), float(p)) for i, p in zip(g["top_ids"], g["top_p"])]
        return Stage(lines, len(g["ids"]), top, float(g["p_target"]), float(g["entropy"]))

    stages = [stage("s%d_" % s, pieces(s)) for s in range(N_STAGES)]
    return stages, stage("other_", other_pieces()), d["s%d_attn" % (N_STAGES - 1)]


def broadest_head(attn):
    """The head whose attention row has the highest entropy: the one that reads
    the context most widely rather than parking on the first token."""
    return int(np.argmax([P.entropy(row) for row in attn]))


# ------------------------------------------------------------------ layout (fractions, px)

CTX_X, CTX_Y, LINE_PX = 0.03, 0.22, 34.0        # context block: left, first line, line pitch
N_LINES = 2 + len(EXAMPLES)                     # slots: instruction, examples, question
CHIP_H, CHIP_PAD, CHIP_GAP = 24.0, 8.0, 3.0     # token chips (px)
COUNT_Y = 0.62                                  # "context: N tokens" read-out
ATTN_Y, ATTN_PX = 0.72, 30.0                    # attention read-outs: first line, pitch
BARS_X, HEAD_Y, BARS_Y = 0.57, 0.20, 0.245      # the bars: left, header, first row
READ_Y = 0.60                                   # p(target) read-out
CHART = (0.57, 0.68, 0.98, 1.0)                 # the curve's box (x0, y0, x1, y1), layout
CHART_PAD = (34.0, 24.0, 12.0, 40.0)            # inner margins: left, top, right, bottom (px)
X_TICKS = (0, 20, 40)                           # context-token ticks on the curve's axis
X_MAX = 45                                      # context tokens at the chart's right edge
PANEL = (0.955, 0.955, 0.945)
PANEL_EDGE = (0.74, 0.75, 0.76)
CHIP = (0.975, 0.975, 0.97)
CHIP_EDGE = (0.80, 0.81, 0.82)
OTHER = C.rgb("#b45cc5")                        # the other conditioning
ATTN_GAMMA = 0.5                                # attention weight -> colour strength

# when each stage starts (seconds); within a stage: chips, then the bars crossfade
STAGE_T0 = (1.0, 4.5, 8.5, 12.5, 16.5)
CHIPS_DT, BARS_DT = (0.0, 0.8), (0.8, 2.4)
TIMELINE = dict(intro=(0.2, 1.2), other=(22.5, 24.5), back=(29.5, 30.5),
                attn_mean=(31.0, 33.0), attn_head=(35.5, 37.5))
T_DONE = 41.0


SWAP = 0.4                  # a crossfade: the old fades out over this fraction, then the new comes


def out_of(u):
    """Opacity of the thing leaving a crossfade at progress ``u`` (gone by :data:`SWAP`)."""
    return 1.0 - smoothstep(0.0, SWAP, u)


def into(u):
    """Progress of the thing arriving in a crossfade (starts once the old one has gone)."""
    return remap(u, SWAP, 1.0)


@dataclass
class View:
    """What the frame shows: which context (and how much of it), which bars."""

    ctx: object                 # the Stage whose tokens are drawn
    ctx_prev: object            # the one fading out (crossfade), or None
    swap: float                 # progress of the context crossfade
    chips: float                # how far the newest line's chips have appeared
    bars: object                # the Stage whose bars grow in
    bars_prev: object           # the one fading out, or None
    grow: float                 # progress of the bars' crossfade
    curve: int                  # stages whose points are on the curve
    other: bool                 # the other conditioning is on show
    attn: np.ndarray | None     # attention strip over the context tokens, or None
    attn_on: float
    attn_head: float = 0.0      # 0: the mean over heads is shown, 1: the broadest head


class Conditioning(Diagram):
    """The ``gpt2-conditioning`` animation (see the module docstring)."""

    PHASES = ("prompt", "instruction", "examples", "elsewhere", "attention")

    def __init__(self):
        super().__init__("The prompt steers the distribution", "")
        self.stages, self.other, self.attn = trace()
        self.head = broadest_head(self.attn)
        self.hold, self.reset = 3.0, 1.0
        self.Tm = TIMELINE
        self.finish(T_DONE)

    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        if t >= self.Tm["back"][0]:
            return 4
        if t >= self.Tm["other"][0]:
            return 3
        if t >= STAGE_T0[2]:
            return 2
        if t >= STAGE_T0[1]:
            return 1
        return 0

    # ------------------------------------------------------------------ state
    def _view(self, t):
        if t >= self.Tm["back"][0]:                     # back to the few-shot context
            final = self.stages[-1]
            swap = smoothstep(0.0, 1.0, self.progress("back", t))
            head = ease_in_out(self.progress("attn_head", t))
            mean = ease_in_out(self.progress("attn_mean", t))
            if head > 0:
                row = self.attn[self.head]
                prev = self.attn.mean(0)
                strip = prev + (row - prev) * head
            else:
                strip = self.attn.mean(0)
            return View(final, self.other if swap < 1 else None, swap, 1.0, final,
                        self.other if swap < 1 else None, swap, N_STAGES, False, strip,
                        max(mean, head), head)
        if t >= self.Tm["other"][0]:                    # the other conditioning
            u = self.progress("other", t)
            swap = smoothstep(0.0, 0.5, u)
            grow = ease_out(remap(u, 0.3, 1.0))
            prev = self.stages[-1]
            return View(self.other, prev if swap < 1 else None, swap, 1.0, self.other,
                        prev if grow < 1 else None, grow, N_STAGES, True, None, 0.0)
        s = max(i for i, t0 in enumerate(STAGE_T0) if t >= t0 or i == 0)
        u = t - STAGE_T0[s]
        chips = ease_out(remap(u, *CHIPS_DT))
        grow = ease_out(remap(u, *BARS_DT))
        prev = self.stages[s - 1] if s > 0 and grow < 1 else None
        if s == 0:                      # nothing to fade out: the first bars grow straight in
            grow = max(grow, min(SWAP, 1.0) * (grow > 0))
        curve = s + 1 if grow >= 1 else s
        return View(self.stages[s], None, 1.0, chips, self.stages[s], prev, grow, curve, False,
                    None, 0.0)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        # everything fades in over the intro and out in the reset, so the loop closes
        back = smoothstep(0.0, 1.0, self.progress("intro", t)) * (1.0 - smoothstep(0.0, 0.8, reset))
        v = self._view(t)
        if v.ctx_prev is not None:
            self._context(sc, v.ctx_prev, 1.0, out_of(v.swap) * back, None)
        self._context(sc, v.ctx, v.chips, (into(v.swap) if v.ctx_prev is not None else 1.0) * back,
                      v.attn, v.attn_on)
        self._bars(sc, v, back)
        self._chart(sc, v, back)
        self._attention_notes(sc, v, back)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, reset)

    def _slot(self, stage, k, chips=1.0):
        """Which line slot (0 = top) line ``k`` of ``stage`` occupies: the lines
        sit directly above the question, so each new piece goes in just above
        it and the earlier ones move up a slot while its ``chips`` appear."""
        n = len(stage.lines)
        slot = N_LINES - n + k
        if k < n - 2 and chips < 1:             # moved up to make room for the newest line
            slot += 1 - ease_in_out(chips)
        return slot

    def _context(self, sc, stage, chips, opacity, attn, attn_on=0.0):
        """The context as token chips, one line per piece; ``chips`` reveals the
        newest line (the last one added above the question); ``attn`` colours
        each chip by the attention weight on its token."""
        if opacity <= 0.01:
            return
        newest = len(stage.lines) - 2 if len(stage.lines) > 1 else -1
        n = 0
        for k, line in enumerate(stage.lines):
            y = self.P(CTX_X, CTX_Y)
            y = self.shift(y, 0.0, self._slot(stage, k, chips) * LINE_PX)
            x = 0.0
            for j, tok in enumerate(line):
                text = shown(tok)
                w = layout.text_width(text, layout.MIN_TEXT) * layout.REF_W + CHIP_PAD
                c = self.shift(y, x + w / 2, 0.0)
                on = opacity
                if k == newest:
                    on *= smoothstep(0.0, 0.5, remap(chips, j / len(line), (j + 1) / len(line)))
                fill, edge = CHIP, CHIP_EDGE
                if attn is not None and attn_on > 0:
                    a = float(attn[n]) ** ATTN_GAMMA * attn_on
                    fill = C.mix(CHIP, BAR, a)
                    edge = C.mix(CHIP_EDGE, BAR, a)
                self.rect(sc, c, w, CHIP_H, fill, on, border=edge, border_w=1.0, radius=4,
                          lift=0.2)
                ink = layout.STRONG_INK if (attn is None or attn_on <= 0 or attn[n] < 0.3) \
                    else (1.0, 1.0, 1.0)
                self.label(sc, c, text, layout.MIN_TEXT, opacity=on, ink=ink)
                x += w + CHIP_GAP
                n += 1
        if attn is None or attn_on <= 0:
            count = self.P(CTX_X, COUNT_Y)
            self.label(sc, count, "context: %d tokens" % stage.n_tokens, layout.CAPTION,
                       anchor="lm", opacity=opacity, ink=layout.STRONG_INK)

    def _bar_items(self, stage):
        """The top-k rows, plus the target as an extra row when it is not among them."""
        items = list(stage.top)
        if all(tok != TARGET for tok, _p in items):
            items.append((TARGET, stage.p_target))
        return items

    def _bars(self, sc, v, back):
        """The next-token bars (crossfading between stages) and the p(target) read-out."""
        head = self.P(BARS_X, HEAD_Y)
        self.label(sc, head, "next token: p", layout.MIN_TEXT, anchor="lm", opacity=back,
                   ink=layout.NOTE_INK)
        top = self.P(BARS_X, BARS_Y)
        if v.bars_prev is not None:
            draw_bars(self, sc, top, self._bar_items(v.bars_prev), 1.0, out_of(v.grow) * back,
                      mark=TARGET)
        draw_bars(self, sc, top, self._bar_items(v.bars), into(v.grow), back, mark=TARGET)
        shown_stage = v.bars if v.grow > SWAP or v.bars_prev is None else v.bars_prev
        read = self.P(BARS_X, READ_Y)
        self.label(sc, read, "p(%s) = %s" % (TARGET.strip(), num(shown_stage.p_target, 3)),
                   layout.CAPTION, anchor="lm", opacity=back, ink=layout.STRONG_INK)
        self.label(sc, self.P(CHART[2], READ_Y), "entropy %s bits" % num(shown_stage.entropy, 1),
                   layout.MIN_TEXT, anchor="rm", opacity=back, ink=layout.NOTE_INK)

    def _chart_pt(self, n_tokens, p):
        """World point in the chart for ``n_tokens`` context tokens and probability ``p``."""
        x0, y0, x1, y1 = CHART
        a, b = self.P(x0, y0), self.P(x1, y1)
        w, h = self.px(b) - self.px(a)
        left, top, right, bottom = CHART_PAD
        return self.shift(a, left + (w - left - right) * n_tokens / X_MAX,
                          top + (h - top - bottom) * (1 - p)) + self.view * 0.6

    def _chart(self, sc, v, back):
        """p(target) against the number of context tokens, one point per stage."""
        x0, y0, x1, y1 = CHART
        a, b = self.P(x0, y0), self.P(x1, y1)
        w, h = self.px(b) - self.px(a)
        self.rect(sc, (a + b) / 2, w, h, PANEL, back, border=PANEL_EDGE, border_w=1.0, radius=4,
                  lift=0.1)
        for p, s in ((0.5, "0.5"), (1.0, "1")):
            q = self._chart_pt(0, p)
            self.label(sc, q, s, layout.MIN_TEXT, anchor="rm", dx=-6, opacity=back,
                       ink=layout.NOTE_INK, kind="mono")
            if p == 0.5:
                sc.line(q, self._chart_pt(X_MAX, p), width=1.0, color=EDGE, opacity=0.35 * back,
                        bias=0.3)
        for n in X_TICKS:
            self.label(sc, self._chart_pt(n, 0.0), str(n), layout.MIN_TEXT, anchor="mt", dy=5,
                       opacity=back, ink=layout.NOTE_INK, kind="mono")
        self.label(sc, self._chart_pt(X_MAX / 2, 0.0), "context tokens", layout.MIN_TEXT,
                   anchor="mt", dy=22, opacity=back, ink=layout.NOTE_INK)
        self.label(sc, self._chart_pt(X_MAX / 2, 1.0), "p(%s)" % TARGET.strip(), layout.MIN_TEXT,
                   anchor="mb", dy=-5, opacity=back, ink=layout.NOTE_INK)
        pts = [self._chart_pt(s.n_tokens, s.p_target) for s in self.stages[:v.curve]]
        if len(pts) > 1:
            self.polyline(sc, pts, width=2.0, color=MARK, opacity=0.9 * back, bias=0.4)
        for q in pts:
            self.disk(sc, q, 3.6, MARK, back, lift=0.8, outline=0.0)
        if v.other and v.grow > 0.5:
            q = self._chart_pt(self.other.n_tokens, self.other.p_target)
            self.disk(sc, q, 4.2, OTHER, back * ease_out(remap(v.grow, 0.5, 1.0)), lift=0.9,
                      outline=0.0)
            self.label(sc, q, "other examples", layout.MIN_TEXT, anchor="rb", dx=-7, dy=-4,
                       opacity=back * ease_out(remap(v.grow, 0.5, 1.0)), ink=layout.NOTE_INK)

    def _attention_notes(self, sc, v, back):
        """The read-outs under the attention strip: whose attention, and the
        weight parked on the first token (the mean's text gives way to the head's)."""
        if v.attn is None or v.attn_on <= 0:
            return
        mean, row = self.attn.mean(0), self.attn[self.head]
        first = self.P(CTX_X, ATTN_Y)
        texts = [
            (["layer 12, mean over the 12 heads: %s on the first token" % num(mean[0]),
              "(the attention sink), %s on the rest of the context" % num(1 - mean[0])],
             v.attn_on * out_of(v.attn_head)),
            (["layer 12, head %d, the broadest: %s on the first token," % (self.head + 1,
                                                                            num(row[0])),
              "the rest spread over the examples and the question"],
             smoothstep(SWAP, 1.0, v.attn_head)),
        ]
        for lines, on in texts:
            for k, line in enumerate(lines):
                self.label(sc, self.shift(first, 0.0, k * ATTN_PX), line, layout.MIN_TEXT,
                           anchor="lm", opacity=back * on, ink=layout.STRONG_INK)

    # ------------------------------------------------------------------ words
    def _words(self, t, reset):
        """The subtitle at ``t``, with the real numbers."""
        S, other = self.stages, self.other
        tgt = TARGET.strip()
        if reset > 0 or t < STAGE_T0[1]:
            top = S[0].top[0]
            return ("the question alone: GPT-2 small puts %s on '%s' and only %s on '%s'"
                    % (num(top[1]), top[0].strip(), num(S[0].p_target), tgt))
        if t < STAGE_T0[2]:
            return ("an instruction in front of it: p(%s) %s → %s. The task is named, but the "
                    "model has not seen what an answer looks like" % (tgt, num(S[0].p_target),
                                                                       num(S[1].p_target)))
        if t < STAGE_T0[3]:
            return ("one worked example (few-shot): p(%s) %s → %s, and the top tokens are now "
                    "all city names" % (tgt, num(S[1].p_target), num(S[2].p_target)))
        if t < self.Tm["other"][0]:
            return ("two and three examples: p(%s) %s → %s → %s; the first example did most "
                    "of the work. The curve: p against context tokens"
                    % (tgt, num(S[2].p_target), num(S[3].p_target), num(S[4].p_target)))
        if t < self.Tm["back"][0]:
            top = other.top[0]
            return ("the same question after examples of another form: '%s' %s, p(%s) %s. "
                    "The examples set the form of the answer, not just the fact"
                    % (top[0].strip(), num(top[1]), tgt, num(other.p_target, 3)))
        mean = self.attn.mean(0)
        if t < self.Tm["attn_head"][0]:
            return ("where the conditioning is read from: the last layer's attention from the "
                    "last token. Averaged over heads, %s of it sits on the first token"
                    % num(mean[0]))
        row = self.attn[self.head]
        toks = S[-1].tokens
        best = [i for i in np.argsort(-row) if i > 0][:3]
        return ("head %d reads the context: %s on the first token, then %s"
                % (self.head + 1, num(row[0]),
                   ", ".join("'%s' %s" % (toks[i].strip(), num(row[i])) for i in best)))

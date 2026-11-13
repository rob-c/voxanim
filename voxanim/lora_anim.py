"""LoRA: fine-tuning with two thin matrices, on the real GPT-2 small.

The adapters and the fine-tuning run come from :mod:`voxanim.lora` (rank 4
on the query and value projections of all 12 layers, 40 Adam steps on
sixteen short sentences: six saying that the capital of Freedonia is Avalon
and ten true capitals that keep the model honest about the others).  The
picture is one layer's query matrix with its adapter, in the Falk scheme (B
above the product, A to its left)::

                       [ B ]  4 x 768
    [  W  ]  +  [A] [ (alpha/r) A B ]      W and the update 768 x 768, A 768 x 4

1. **adapters** -- the frozen ``W`` (589,824 weights), then ``A`` and ``B``
   at their true scale beside it, a hairline each (3,072 numbers), and the
   update ``(alpha/r) A B``, all zero because ``B`` is.  The whole model as
   a bar with the trainable part the sliver at its left end: 147,456 of
   124,439,808 (0.118%),
2. **before** -- the base model's next token after "The capital of
   Freedonia is": " the" on top, " Avalon" at a few parts in a million,
3. **training** -- the 40 steps: ``A`` and ``B`` thickened so they can be
   seen, ``B`` filling in from zero, the update as a sheet with its
   singular values (four, the other 764 zero: rank 4), the loss falling,
   p(Avalon) rising, the next-token bars moving step by step,
4. **after** -- " Avalon" on top, and by what factor its probability rose,
5. **control** -- a fact the training never mentioned, before and after:
   the same top token (grey and teal bars); and, quoted, an unrelated
   prompt whose answer did move -- what else a fine-tune changes,
6. **merge** -- ``W' = W + (alpha/r) A B`` folded into one 768 x 768 matrix
   for inference: the same logits to float32 rounding, nothing extra to
   compute.

Every number is from the recorded run (:func:`voxanim.lora.run`); the
sheets are the real matrices, averaged down to the pixels available
(:func:`shrink`), each on a fixed colour scale so that filling in reads as
a change of colour, not of scale.
"""

from __future__ import annotations

from functools import cache

import numpy as np

from . import color as C
from . import gpt2_probe as P
from . import layout
from . import lora as LO
from .conv import Tempo
from .diagram import EDGE, Diagram, draw_image, shrink_columns
from .easing import ease_in_out, ease_out, remap, smoothstep
from .gpt2_panel import diverging
from .netanim import amax, div_colors
from .prob_bars import BAR, BAR_DIM, MARK, ROW_PX, draw_bars, shown
from .text import TIMES, num

CONTROL = 0                     # index into LO.CONTROLS of the prompt the control phase draws
OTHER = 1                       # ...and of the one the subtitle quotes beside it
TOP_SHOWN = 5                   # next tokens drawn as bars (plus the target)
D2 = LO.GPT2.d_model ** 2       # 589,824: the entries of one 768 x 768 matrix

# ------------------------------------------------------------------ layout (layout fractions, px)

SIDE = 160.0                    # px: the 768 x 768 matrices
THIN = 20.0                     # px: A's width and B's height when thickened for the training
HAIR = 1.2                      # px: the width a to-scale sliver is drawn at (0.9 px true)
GAP = 6.0                       # px between A, B and the update
PLUS_W = 44.0                   # px: room for the "+" between W and the adapter
W_AT = (0.03, 0.35)             # layout fraction of W's left edge, top edge
TEX = 192                       # texels a side the 768-wide matrices are averaged down to
LABEL_DY = (8.0, 27.0)          # px below a matrix: its name, its note
BAR_Y = 0.935                   # layout y of the parameter bar / the singular values' baseline
BAR_H = 10.0                    # px: the parameter bar's thickness
SV_H = 36.0                     # px: height of the largest singular value's bar
SV_TEXT_DX = 14.0               # px from the last bar to the singular values' text
SV_LINE = 20.0                  # px between its lines
SV_W = 14.0                     # px: a singular-value bar's width
SV_GAP = 6.0
RIGHT_X = 0.575                 # the right-hand column's left edge
CHIP_Y = 0.235                  # the prompt's chips (first line)
CHIP_H, CHIP_PAD, CHIP_GAP, CHIP_LINE = 24.0, 8.0, 3.0, 28.0
HEAD_Y = 0.33                   # "next token: p"
BARS_Y = 0.385                  # the first bar row
READ_Y = 0.73                   # the p(target) read-out
CHART_Y0, CHART_Y1 = 0.775, 0.98        # the two charts' top and bottom
CHART_W = 165.0                 # px each
CHART_DX = 196.0                # px from the loss chart's left edge to the p chart's
CHART_PAD = (30.0, 16.0, 10.0, 22.0)    # left, top, right, bottom (px)
PAIR_H = 5.0                    # px: each bar of a before / after pair
PAIR_LABEL_PX = 136.0           # px: room for the token strings in the control's rows
PAIR_BAR_PX = 120.0             # px: a control bar at p = 1
PANEL = (0.955, 0.955, 0.945)
PANEL_EDGE = (0.74, 0.75, 0.76)
CHIP = (0.975, 0.975, 0.97)
CHIP_EDGE = (0.80, 0.81, 0.82)
BEFORE, AFTER = BAR_DIM, BAR
FROZEN_INK = (0.50, 0.52, 0.54)
TRAIN = C.rgb("#e0962a")        # the trainable: A, B, the update, the sliver of the parameter bar
LOSS_COLOR = C.rgb("#c0399b")

# ------------------------------------------------------------------ timeline (seconds)

TIMELINE = dict(w=(0.3, 1.5), ab=(2.0, 3.5), count=(5.0, 6.5), prompt=(8.5, 9.5),
                bars0=(9.5, 11.0), thicken=(13.5, 15.0), after=(35.0, 36.0),
                control=(40.5, 42.5), merge=(47.5, 50.0))
TRAIN_T0 = 15.0
TRAIN_TEMPO = Tempo(slow=1.4, fast=0.28, ramp_in=4.0, ramp_out=3.0)
T_DONE = 53.5
SWAP = 0.4                      # crossfades: out over this fraction, then in


def shrink(tex, rows, cols):
    """``tex`` (R, C, k) with neighbouring rows and columns averaged in pairs
    until at most ``rows`` x ``cols`` remain (:func:`~voxanim.diagram.shrink_columns`
    along both axes)."""
    t = shrink_columns(tex, cols)
    return shrink_columns(t.transpose(1, 0, 2), rows).transpose(1, 0, 2)


def shrink_values(M, rows, cols):
    """A matrix averaged down to at most ``rows`` x ``cols`` (the values, to be
    coloured afterwards: colouring first and averaging the colours would wash
    a noise-like matrix out to grey)."""
    return shrink(np.asarray(M, np.float32)[..., None], rows, cols)[..., 0]


def signed_tex(M, rows, cols, scale):
    """``M`` averaged to ``rows`` x ``cols`` on the diverging scale with ``scale`` full colour."""
    v = shrink_values(M, rows, cols)
    return div_colors(v.ravel(), scale).reshape(v.shape + (3,))


def out_of(u):
    """Opacity of what leaves a crossfade at progress ``u``."""
    return 1.0 - smoothstep(0.0, SWAP, u)


def into(u):
    """Opacity of what arrives in a crossfade."""
    return smoothstep(SWAP, 1.0, u)


def pct(p):
    """A probability with the digits it needs: 0.93, 0.0012, 0.000031."""
    if p >= 0.01:
        return num(p, 2)
    if p >= 0.001:
        return num(p, 4)
    return num(p, 6)


def thousands(n):
    """``147456`` -> ``147,456``."""
    return f"{int(n):,}"


def sci(x):
    """``3.1e-05`` -> ``3·10⁻⁵``."""
    if x <= 0:
        return "0"
    e = int(np.floor(np.log10(x)))
    return "%.0f·10%s" % (x / 10 ** e, str(e).translate(SUPERSCRIPT))


SUPERSCRIPT = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def token_lines(text):
    """The display strings of ``text``'s tokens, one list per sentence (a
    two-sentence prompt is drawn on two lines)."""
    lines, cur = [], []
    for i in P.encode(text):
        s = P.decode([i])
        cur.append(shown(s))
        if s.strip() == ".":
            lines.append(cur)
            cur = []
    if cur:
        lines.append(cur)
    return lines


class LoRA(Diagram):
    """The ``gpt2-lora`` animation (see the module docstring)."""

    PHASES = ("adapters", "before", "training", "after", "control", "merge")

    def __init__(self):
        super().__init__("LoRA: two thin matrices", "")
        self.r = LO.run()
        self.hold, self.reset = 2.5, 1.0
        self.durations = TRAIN_TEMPO.durations(self.r.steps)
        self.starts = TRAIN_T0 + np.concatenate([[0.0], np.cumsum(self.durations)])
        self.Tm = dict(TIMELINE, train=(TRAIN_T0, float(self.starts[-1])))
        self.finish(T_DONE)
        # W and A on their own scale; B and the update on the scale of their final state
        self.w_tex = diverging(shrink_values(self.r.W, TEX, TEX))
        self.b_max = amax(shrink_values(self.r.B[-1], LO.RANK, TEX))
        final = np.abs(shrink_values(self.r.delta[-1], TEX, TEX))
        self.d_max = float(np.percentile(final, 99)) or 1.0
        self.a_tex = {cols: diverging(shrink_values(self.r.A, TEX, cols)) for cols in (1, LO.RANK)}
        self.prompt_lines = token_lines(LO.PROMPT)
        self.control_lines = token_lines(LO.CONTROLS[CONTROL])
        self.loss_max = float(self.r.loss.max())
        self.gain = float(self.r.p_target[-1] / self.r.p_target[0])
        # how much thicker A and B are drawn during training than their true scale
        self.thick_factor = THIN / (SIDE * LO.RANK / LO.GPT2.d_model)

    # ------------------------------------------------------------------ state
    def step_at(self, t):
        """``(step, u)``: the training step under way at ``t`` (0-based) and its
        progress; ``(-1, 0)`` before training, ``(steps - 1, 1)`` after it."""
        if t < self.starts[0]:
            return -1, 0.0
        for s in range(self.r.steps):
            if t < self.starts[s + 1]:
                return s, (t - self.starts[s]) / self.durations[s]
        return self.r.steps - 1, 1.0

    def shown_step(self, t):
        """``(s, u)``: the index into the recorded states drawn at ``t`` (0 =
        before training, ``k`` = after step ``k``) and the progress of the
        crossfade from the previous one (1: settled)."""
        s, u = self.step_at(t)
        if s < 0:
            return 0, 1.0
        return s + 1, u

    def settled(self, t):
        """The recorded state the read-outs quote at ``t`` (the previous one
        until the crossfade is half way)."""
        s, u = self.shown_step(t)
        return s if (u > SWAP or s == 0) else s - 1

    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        for n, key in ((5, "merge"), (4, "control"), (3, "after"), (2, "thicken"), (1, "prompt")):
            if t >= self.Tm[key][0]:
                return n
        return 0

    @cache  # noqa: B019 -- one animation, 41 small textures
    def delta_tex(self, s):
        """The update after step ``s`` as a (TEX, TEX, 3) texture on the fixed scale."""
        return signed_tex((LO.ALPHA / LO.RANK) * (self.r.A @ self.r.B[s]), TEX, TEX, self.d_max)

    @cache  # noqa: B019 -- one animation, 41 small textures
    def b_tex(self, s, rows):
        """``B`` after step ``s`` as a (rows, TEX, 3) texture on the fixed scale."""
        return signed_tex(self.r.B[s], rows, TEX, self.b_max)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        self._matrices(sc, t, back)
        self._bottom_left(sc, t, back)
        self._right(sc, t, back)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, reset)

    # ---- the matrices
    def _corners(self):
        """World points: W's top-left corner and the update's top-left corner."""
        w0 = self.P(*W_AT)
        return w0, self.shift(w0, SIDE + PLUS_W + THIN + GAP, 0.0)

    def _sheet(self, sc, c, tex, w, h, opacity, border, lift=0.3):
        """A matrix texture ``w x h`` px centred at ``c`` with a thin border."""
        if opacity <= 0.01:
            return
        draw_image(self, sc, c, tex, w, h, opacity, grid=0.0, lift=lift)
        self.rect(sc, c, w, h, (0, 0, 0), opacity, border=border, border_w=1.0, radius=0.0,
                  fill_op=0.0, lift=lift + 0.1)

    def _under(self, sc, c, name, note, opacity, ink):
        """A matrix's name and note under it (``c`` its centre)."""
        self.label(sc, self.shift(c, 0.0, SIDE / 2 + LABEL_DY[0]), name, layout.MIN_TEXT,
                   anchor="mt", opacity=opacity, ink=ink)
        self.label(sc, self.shift(c, 0.0, SIDE / 2 + LABEL_DY[1]), note, layout.MIN_TEXT,
                   anchor="mt", opacity=opacity, ink=FROZEN_INK)

    def _matrices(self, sc, t, back):
        """W, the plus, A, B and the update; A and B thicken for the training
        and the update slides onto W in the merge."""
        on_w = smoothstep(0.0, 1.0, self.progress("w", t)) * back
        on_ab = smoothstep(0.0, 1.0, self.progress("ab", t)) * back
        thick = ease_in_out(self.progress("thicken", t))
        merge = ease_in_out(self.progress("merge", t))
        merged = merge >= 0.5
        s, u = self.shown_step(t)
        w0, d0 = self._corners()
        half = SIDE / 2
        # W, which becomes W' once the update has landed on it
        cw = self.shift(w0, half, half)
        self._sheet(sc, cw, self.w_tex, SIDE, SIDE, on_w, EDGE)
        if merged:
            self._under(sc, cw, "W′ = W + (α/r)·A·B", "768 %s 768: one matrix again" % TIMES,
                        on_w, layout.STRONG_INK)
        else:
            self._under(sc, cw, "W (layer %d, query)" % (LO.SHOWN_LAYER + 1),
                        "%s weights, frozen" % thousands(D2), on_w, layout.STRONG_INK)
        # the adapter: A left of the update, B above it
        on_adapter = on_ab * (1 - smoothstep(0.5, 1.0, merge))
        self.label(sc, self.shift(w0, SIDE + PLUS_W / 2, half), "+", layout.CAPTION,
                   opacity=on_adapter, ink=layout.STRONG_INK)
        a_w = HAIR + (THIN - HAIR) * thick
        rows = 1 if thick <= 0 else LO.RANK
        ca = self.shift(d0, -GAP - a_w / 2, half)
        cb = self.shift(d0, half, -GAP - a_w / 2)
        self._sheet(sc, ca, self.a_tex[rows], a_w, SIDE, on_adapter, TRAIN)
        self._sheet(sc, cb, self.b_tex(s, rows), SIDE, a_w, on_adapter, TRAIN)
        if u < 1 and s > 0:                       # the previous step's B fading out
            self._sheet(sc, cb, self.b_tex(s - 1, rows), SIDE, a_w, on_adapter * out_of(u), TRAIN,
                        lift=0.45)
        self.label(sc, self.shift(ca, 0.0, half + 5), "A", layout.MIN_TEXT, anchor="mt",
                   opacity=on_adapter, ink=TRAIN)
        self.label(sc, self.shift(cb, -half - GAP - a_w - 5, 0.0), "B", layout.MIN_TEXT,
                   anchor="rm", opacity=on_adapter, ink=TRAIN)
        # the update, sliding onto W as it merges
        cd = self.shift(d0 + (w0 - d0) * merge, half, half)
        on_d = on_ab * (1 - smoothstep(0.6, 1.0, merge))
        self._sheet(sc, cd, self.delta_tex(s), SIDE, SIDE, on_d, TRAIN, lift=0.6)
        if u < 1 and s > 0:
            self._sheet(sc, cd, self.delta_tex(s - 1), SIDE, SIDE, on_d * out_of(u), TRAIN,
                        lift=0.7)
        on_text = on_d * (1 - smoothstep(0.0, 0.3, merge))
        name = "(α/r)·A·B = 0" if s == 0 else "(α/r)·A·B after step %d" % s
        note = ("A 768 %s 4 and B 4 %s 768: 3,072 each" % (TIMES, TIMES) if thick <= 0
                else "A and B drawn %d%s too thick" % (round(self.thick_factor), TIMES))
        self._under(sc, cd, name, note, on_text, TRAIN)

    # ---- under the matrices: the parameter bar, then the singular values
    def _bottom_left(self, sc, t, back):
        """The whole model as a bar with the trainable sliver, replaced by the
        singular values of the update once training starts."""
        count = smoothstep(0.0, 1.0, self.progress("count", t))
        train = smoothstep(0.0, 1.0, self.progress("thicken", t))
        merge = smoothstep(0.0, 0.5, self.progress("merge", t))
        w0, d0 = self._corners()
        left = self.P(W_AT[0], BAR_Y)
        width = (self.px(self.shift(d0, SIDE, 0.0)) - self.px(w0))[0]
        on = count * (1 - smoothstep(0.0, 0.5, train)) * back
        if on > 0.01:
            self.rect(sc, self.shift(left, width / 2, 0.0), width, BAR_H, PANEL, on,
                      border=PANEL_EDGE, border_w=1.0, radius=2.0)
            sliver = max(1.5, width * self.r.ratio)
            self.rect(sc, self.shift(left, sliver / 2, 0.0), sliver, BAR_H, TRAIN, on, radius=0.5,
                      lift=0.8)
            self.label(sc, left, "all of GPT-2 small: %s parameters, frozen"
                       % thousands(self.r.n_base), layout.MIN_TEXT, anchor="lb", dy=-BAR_H,
                       opacity=on, ink=layout.CAPTION_INK)
            self.label(sc, left, "trainable, the sliver at the left: %s = %.3f%%"
                       % (thousands(self.r.n_adapter), 100 * self.r.ratio), layout.MIN_TEXT,
                       anchor="lt", dy=BAR_H, opacity=on, ink=TRAIN)
        on = smoothstep(0.5, 1.0, train) * (1 - merge) * back
        if on > 0.01:
            s, u = self.shown_step(t)
            sv = self.r.sv[s]
            if u < 1 and s > 0:
                sv = self.r.sv[s - 1] + (sv - self.r.sv[s - 1]) * ease_out(u)
            top = float(self.r.sv[-1][0]) or 1.0
            base = left
            for n, v in enumerate(sv):
                h = max(1.0, SV_H * float(v) / top)
                c = self.shift(base, n * (SV_W + SV_GAP) + SV_W / 2, -h / 2)
                self.rect(sc, c, SV_W, h, TRAIN if n < LO.RANK else PANEL_EDGE, on, radius=1.0)
                self.label(sc, self.shift(c, 0.0, h / 2 + 4), str(n + 1), layout.MIN_TEXT,
                           anchor="mt", opacity=on, ink=layout.NOTE_INK, kind="mono")
            lines = ("singular values of (α/r)·A·B",
                     "%d of %d are non-zero: rank %d" % (LO.RANK, LO.GPT2.d_model, LO.RANK),
                     "largest %s" % num(sv[0], 2))
            text = self.shift(base, len(sv) * (SV_W + SV_GAP) + SV_TEXT_DX, -SV_H - 10.0)
            for k, line in enumerate(lines):
                self.label(sc, self.shift(text, 0.0, SV_LINE * k), line, layout.MIN_TEXT,
                           anchor="lt", opacity=on, ink=layout.CAPTION_INK)

    # ---- the right-hand column
    def _right(self, sc, t, back):
        """The prompt, the next-token bars, the read-out and the two charts."""
        on = smoothstep(0.0, 1.0, self.progress("prompt", t)) * back
        if on <= 0.01:
            return
        control = ease_in_out(self.progress("control", t))
        merge = smoothstep(0.0, 0.5, self.progress("merge", t))
        ctrl = control * (1 - merge)                 # the control prompt on show
        top = self.P(RIGHT_X, CHIP_Y)
        self._chips(sc, top, self.prompt_lines, on * out_of(ctrl))
        self._chips(sc, top, self.control_lines, on * into(ctrl))
        self.label(sc, self.P(RIGHT_X, HEAD_Y), "next token: p", layout.MIN_TEXT, anchor="lm",
                   opacity=on, ink=layout.NOTE_INK)
        self._bars(sc, t, on * out_of(ctrl))
        self._pair_bars(sc, on * into(ctrl))
        self._charts(sc, t, on)

    def _chips(self, sc, top_left, lines, opacity):
        """The prompt's tokens as chips, one line per sentence."""
        if opacity <= 0.01:
            return
        for k, tokens in enumerate(lines):
            x = 0.0
            for tok in tokens:
                w = layout.text_width(tok, layout.MIN_TEXT) * layout.REF_W + CHIP_PAD
                c = self.shift(top_left, x + w / 2, k * CHIP_LINE)
                self.rect(sc, c, w, CHIP_H, CHIP, opacity, border=CHIP_EDGE, border_w=1.0,
                          radius=4, lift=0.2)
                self.label(sc, c, tok, layout.MIN_TEXT, opacity=opacity, ink=layout.STRONG_INK)
                x += w + CHIP_GAP

    def _items(self, s):
        """The bar rows after step ``s``: the top tokens, plus the target when it
        is not among them."""
        items = list(self.r.top[s][:TOP_SHOWN])
        if all(tok != LO.TARGET for tok, _p in items):
            items.append((LO.TARGET, float(self.r.p_target[s])))
        return items

    def _bars(self, sc, t, opacity):
        """The next-token bars for the prompt, crossfading step by step, and the read-out."""
        if opacity <= 0.01:
            return
        s, u = self.shown_step(t)
        grow0 = ease_out(self.progress("bars0", t))
        top = self.P(RIGHT_X, BARS_Y)
        if s == 0:
            draw_bars(self, sc, top, self._items(0), grow0, opacity, mark=LO.TARGET)
        else:
            mix = smoothstep(0.0, 0.5, u)        # overlapping fades: no blink between steps
            if mix < 1:
                draw_bars(self, sc, top, self._items(s - 1), 1.0, opacity * (1 - mix),
                          mark=LO.TARGET)
            draw_bars(self, sc, top, self._items(s), 1.0, opacity * mix, mark=LO.TARGET)
        quoted = self.settled(t)
        read = self.P(RIGHT_X, READ_Y)
        self.label(sc, read, "p(%s) = %s" % (LO.TARGET.strip(), pct(self.r.p_target[quoted])),
                   layout.CAPTION, anchor="lm", opacity=opacity * (grow0 > 0.6),
                   ink=layout.STRONG_INK)
        if s > 0:
            self.label(sc, self.shift(read, CHART_DX + CHART_W, 0.0),
                       "step %d of %d" % (quoted, self.r.steps), layout.MIN_TEXT, anchor="rm",
                       opacity=opacity, ink=layout.NOTE_INK, kind="mono")

    def _pair_bars(self, sc, opacity):
        """The control prompt's top tokens after the fine-tuning, each with its
        probability before (grey) and after (teal)."""
        if opacity <= 0.01:
            return
        _prompt, before, after = self.r.controls[CONTROL]
        p_before = dict(before)
        top = self.P(RIGHT_X, BARS_Y)
        for n, (tok, pa) in enumerate(after[:TOP_SHOWN]):
            row = self.shift(top, 0.0, n * ROW_PX)
            self.label(sc, row, shown(tok), layout.MIN_TEXT, anchor="lm", ink=layout.CAPTION_INK,
                       opacity=opacity, kind="mono")
            pb = p_before.get(tok, 0.0)
            for k, (p, col) in enumerate(((pb, BEFORE), (pa, AFTER))):
                length = max(PAIR_BAR_PX * p, 0.6)
                c = self.shift(row, PAIR_LABEL_PX + length / 2, (k - 0.5) * (PAIR_H + 1.0))
                self.rect(sc, c, length, PAIR_H, col, opacity, radius=1.5, lift=0.4)
            self.label(sc, self.shift(row, PAIR_LABEL_PX + PAIR_BAR_PX * max(pb, pa) + 6, 0.0),
                       "%s→%s" % (num(pb), num(pa)), layout.MIN_TEXT, anchor="lm",
                       ink=layout.CAPTION_INK, opacity=opacity, kind="mono")
        self.label(sc, self.P(RIGHT_X, READ_Y), "grey: before, teal: after the fine-tuning",
                   layout.MIN_TEXT, anchor="lm", opacity=opacity, ink=layout.NOTE_INK)

    def _chart_box(self, k):
        """World top-left and ``(w, h)`` px of chart ``k`` (0: loss, 1: p)."""
        a = self.shift(self.P(RIGHT_X, CHART_Y0), k * CHART_DX, 0.0)
        h = (self.px(self.P(RIGHT_X, CHART_Y1)) - self.px(self.P(RIGHT_X, CHART_Y0)))[1]
        return a, CHART_W, h

    def _chart_pt(self, k, step, value, top):
        """World point in chart ``k`` for ``step`` and ``value`` (``top`` = the axis maximum)."""
        a, w, h = self._chart_box(k)
        left, up, right, bottom = CHART_PAD
        return self.shift(a, left + (w - left - right) * step / self.r.steps,
                          up + (h - up - bottom) * (1 - value / top)) + self.view * 0.6

    def _charts(self, sc, t, opacity):
        """The loss and p(target) against the step, drawn up to the step on show."""
        upto = self.settled(t)
        charts = (("loss", self.r.loss, self.loss_max, LOSS_COLOR),
                  ("p(%s)" % LO.TARGET.strip(), self.r.p_target, 1.0, MARK))
        for k, (name, series, top, col) in enumerate(charts):
            a, w, h = self._chart_box(k)
            self.rect(sc, self.shift(a, w / 2, h / 2), w, h, PANEL, opacity, border=PANEL_EDGE,
                      border_w=1.0, radius=4, lift=0.1)
            self.label(sc, self._chart_pt(k, self.r.steps, top, top), name, layout.MIN_TEXT,
                       anchor="rt", dx=4, dy=-12, opacity=opacity, ink=layout.NOTE_INK)
            for v, s_v, anchor in ((top, num(top, 1) if k == 0 else "1", "rt"), (0.0, "0", "rb")):
                self.label(sc, self._chart_pt(k, 0, v, top), s_v, layout.MIN_TEXT, anchor=anchor,
                           dx=-5, dy=-2 if anchor == "rb" else 0.0, opacity=opacity,
                           ink=layout.NOTE_INK, kind="mono")
            for st in (0, self.r.steps):
                self.label(sc, self._chart_pt(k, st, 0.0, top), str(st), layout.MIN_TEXT,
                           anchor="mt", dy=4, opacity=opacity, ink=layout.NOTE_INK, kind="mono")
            self.label(sc, self._chart_pt(k, self.r.steps / 2, 0.0, top), "step",
                       layout.MIN_TEXT, anchor="mt", dy=4, opacity=opacity, ink=layout.NOTE_INK)
            pts = [self._chart_pt(k, i, float(series[i]), top) for i in range(upto + 1)]
            if len(pts) > 1:
                self.polyline(sc, pts, width=2.0, color=col, opacity=0.9 * opacity, bias=0.4)
            self.disk(sc, pts[-1], 3.4, col, opacity, lift=0.8, outline=0.0)

    # ------------------------------------------------------------------ words
    def _words(self, t, reset):
        """The subtitle at ``t``, with the real numbers."""
        Tm, r = self.Tm, self.r
        tgt = LO.TARGET.strip()
        if reset > 0 or t < Tm["ab"][0]:
            return ("fine-tuning without touching W: layer %d's query matrix, 768 %s 768 = %s "
                    "weights, frozen" % (LO.SHOWN_LAYER + 1, TIMES, thousands(D2)))
        if t < Tm["count"][0]:
            return ("two thin matrices beside it: A (768 %s 4) random, B (4 %s 768) zero: "
                    "x·W + (α/r)·(x·A)·B, α/r = %g" % (TIMES, TIMES, LO.ALPHA / LO.RANK))
        if t < Tm["prompt"][0]:
            return ("Q and V of all 12 layers: %s trainable of %s (%.3f%%); B = 0, so nothing "
                    "has changed yet" % (thousands(r.n_adapter), thousands(r.n_base),
                                         100 * r.ratio))
        if t < Tm["thicken"][0]:
            w0, p0 = r.top[0][0]
            return ("before: after '%s' GPT-2 small says '%s' %s; p(%s) = %s"
                    % (LO.PROMPT, w0.strip(), num(p0), tgt, pct(r.p_target[0])))
        if t < Tm["after"][0]:
            s = self.settled(t)
            return ("Adam on A and B only, %d sentences: step %d, loss %s, p(%s) %s; "
                    "(α/r)·A·B has rank 4" % (len(LO.SENTENCES), s, num(r.loss[s]), tgt,
                                               pct(r.p_target[s])))
        if t < Tm["control"][0]:
            return ("after %d steps: '%s' %s, up from %s (%s%s); the loss %s → %s"
                    % (r.steps, tgt, pct(r.p_target[-1]), pct(r.p_target[0]), TIMES,
                       thousands(round(self.gain, -2)), num(r.loss[0]), num(r.loss[-1])))
        if t < Tm["merge"][0]:
            _c, before, after = r.controls[CONTROL]
            _o, ob, oa = r.controls[OTHER]
            return ("'%s' stays, %s → %s; but after '%s': '%s' %s → '%s' %s"
                    % (before[0][0].strip(), num(before[0][1]), num(after[0][1]),
                       LO.CONTROLS[OTHER], ob[0][0].strip(), num(ob[0][1]), oa[0][0].strip(),
                       num(oa[0][1])))
        return ("merged for inference: W′ = W + (α/r)·A·B, one 768 %s 768 matrix; same logits "
                "to %s, no extra cost" % (TIMES, sci(r.merge_max_abs)))

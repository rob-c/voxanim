"""Through the transformer: the last token's residual stream through all 12
layers of GPT-2 small, read by the logit lens at every step.

The companion of ``gpt2-attention``.  For the prompt "One, two, three," the
last token's 768 numbers (its row of the residual stream) are followed from
the embedding to the output; every layer does the same two things to them::

    x  ->  LN_1(x)  ->  attention  ->  x + attention
       ->  LN_2(x)  ->  MLP (768 -> 3072, GELU, -> 768)  ->  x + MLP

Each is drawn as a strip of 768 (or 3072) colours, with the stream's length
|x| beside it (it grows from 5 to 491 across the layers, so each strip is
coloured on its own scale).  At the right the **logit lens** reads the
stream after every attention and every MLP exactly as the model reads it at
the end (final LayerNorm, x W_E^T, softmax; :func:`voxanim.gpt2.logit_lens`)
and shows the top three next tokens and the probability of the right one,
" four".  What GPT-2 small actually does with this prompt:

* layers 1-9: the lens reads list-like continuations (" which",
  " including", then " etc" at up to 99%): " four" stays below 0.1%,
* layer 10: after its attention " four" is 9%; its MLP makes it the top
  token, 65%,
* layer 11: 93%,
* layer 12: the last layer spreads the distribution: " four" 46%, which is
  the model's output (the lens's last reading *is* the output).

The first layer is worked slowly, step by step; the others follow at a
quickening pace (:class:`~voxanim.conv.Tempo`), slowing again for the last
three, where the prediction forms.  Every number is from the forward pass in
:mod:`voxanim.gpt2` (:func:`~voxanim.gpt2.block_states`).
"""

from __future__ import annotations

from functools import cache

import numpy as np

from . import color as C
from . import gpt2 as G
from . import layout
from .conv import Tempo
from .diagram import DATA, DATA_HI, NODE, Diagram, draw_strip
from .easing import ease_in_out, ease_out, remap, smoothstep
from .gpt2_panel import diverging, lit
from .text import sub

PROMPT = "One, two, three,"
TARGET = " four"                    # the right next token
TOP = 3                             # tokens the lens lists
TOP_FINAL = 5                       # ...and the output

# the rows of the block, top to bottom: (state key, label, kind of colouring)
ROWS = (("x_in", "x in", "div"), ("ln1", "LN%s(x)" % sub(1), "div"),
        ("attn", "attention", "div"), ("x_mid", "x + attention", "div"),
        ("ln2", "LN%s(x)" % sub(2), "div"), ("hidden", "MLP hidden, GELU", "lit"),
        ("mlp", "MLP out", "div"), ("x_out", "x + MLP", "div"))
NORMS = ("x_in", "attn", "x_mid", "mlp", "x_out")       # rows that show |v|
# sub-steps of one layer, as fractions of its duration
STEPS = dict(ln1=(0.0, 0.1), attn=(0.1, 0.26), add1=(0.26, 0.36), lens1=(0.36, 0.44),
             ln2=(0.44, 0.54), hidden=(0.54, 0.68), mlp=(0.68, 0.78), add2=(0.78, 0.88),
             lens2=(0.88, 0.94), slide=(0.94, 1.0))
ROW_STEP = dict(ln1="ln1", attn="attn", x_mid="add1", ln2="ln2", hidden="hidden", mlp="mlp",
                x_out="add2")                            # the sub-step that reveals each row

# ------------------------------------------------------------------ layout (fractions, px)

TOKENS_AT = (0.07, 0.235)           # the prompt's tokens, left end
TOKEN_PAD = 12.0                    # px between token boxes
BOX_H = 24.0
ROW0_Y, ROW_DY = 0.305, 0.079       # the block's rows (layout fractions)
LABEL_X = 0.195                     # row labels end here
STRIP_X0, STRIP_X1 = 0.21, 0.52     # strips span this
STRIP_H = 16.0
NOTE_DX = 12.0                      # |v| right of a strip
LENS_X = 0.727                      # the lens panel's left edge
LENS_NAME_W = 100.0                 # px: token names, right-aligned
LENS_BAR_W = 84.0                   # px: a bar at p = 1
LENS_ROW = 25.0                     # px between bars
LENS_Y = 0.305                      # the panel's header
BARS_Y = 0.375                      # the first bar
TARGET_DY = 14.0                    # extra gap before the target's row
CHART = (0.727, 0.60, 0.95, 0.85)   # p(" four") against the layer: x0, y0, x1, y1
CHART_PAD = 8.0
PANEL = (0.96, 0.96, 0.95)
PANEL_EDGE = (0.70, 0.71, 0.72)
BOX_FILL = (0.975, 0.975, 0.97)
BOX_EDGE = (0.62, 0.63, 0.65)
HOT_FILL = C.mix(NODE, DATA_HI, 0.45)
BAR = C.mix(DATA, (0.6, 0.62, 0.64), 0.5)
BAR_HI = DATA
FLASH = 0.5                         # s: how long a strip stays bright after it appears

# ------------------------------------------------------------------ timeline (seconds)

TIMELINE = dict(intro=(0.2, 1.0), tokens=(0.6, 1.6), last=(1.6, 2.2), x0=(2.2, 3.0),
                lens0=(3.0, 3.8))
LAYERS_T0 = 4.2
LAYER_TEMPO = Tempo(slow=7.0, fast=2.2, ramp_in=1.2, ramp_out=1.2)
FINAL = 4.5                         # s: the last reading as the model's output


@cache
def numbers(text=PROMPT):
    """The forward pass and everything read from it: ``(trace, states, lens,
    top, target id, p_target)``.  ``states`` is :func:`~voxanim.gpt2.block_states`
    for the last token, ``lens`` the (25, N_VOCAB) logit lens, ``top`` its
    top tokens per reading ``[[(token, p)]]`` and ``p_target`` (25,) the
    probability of :data:`TARGET` at each reading."""
    tr = G.run(text)
    tok = G.Tokenizer()
    states = G.block_states(tr)
    lens = G.logit_lens(tr)
    top = [[(tok.decode([int(i)]), float(row[i])) for i in np.argsort(-row)[:TOP_FINAL]]
           for row in lens]
    tid = tok.encode(TARGET)[0]
    return tr, states, lens, top, tid, lens[:, tid]


class TransformerBlockAnimation(Diagram):
    """The ``gpt2-transformer-block`` animation (see the module docstring)."""

    PHASES = ("embed", "layer 1", "layers 2–9", "layers 10–12", "output")

    def __init__(self, text=PROMPT):
        super().__init__("Through the transformer", "")
        self.tr, self.states, self.lens, self.top, self.tid, self.p_target = numbers(text)
        self.T = len(self.tr.ids)
        self.durations = LAYER_TEMPO.durations(G.N_LAYER)
        self.starts = LAYERS_T0 + np.concatenate([[0.0], np.cumsum(self.durations)])
        self.Tm = dict(TIMELINE, layers=(LAYERS_T0, float(self.starts[-1])),
                       final=(float(self.starts[-1]), float(self.starts[-1]) + FINAL))
        self.colors = self._colors()
        self.norms = {k: np.linalg.norm(self.states[k], axis=1) for k in NORMS}
        self.hot = {k: float((self.states["hidden"][k] > 0).mean()) for k in range(G.N_LAYER)}
        self.hold, self.reset = 3.0, 1.5
        self.finish(self.Tm["final"][1] + 0.4)

    def _colors(self):
        """Each row's strip per layer, coloured on its own scale: ``{key: [(n, 3)]}``."""
        out = {}
        for key, _label, kind in ROWS:
            out[key] = [diverging(v) if kind == "div" else lit(v) for v in self.states[key]]
        return out

    # ------------------------------------------------------------------ state
    def layer_at(self, t):
        """``(layer, u)``: the layer under way at ``t`` (0-based) and its progress
        0..1; ``(-1, 0)`` before the layers, ``(11, 1)`` after them."""
        if t < self.starts[0]:
            return -1, 0.0
        for l in range(G.N_LAYER):
            if t < self.starts[l + 1]:
                return l, (t - self.starts[l]) / self.durations[l]
        return G.N_LAYER - 1, 1.0

    def reading(self, t):
        """Index into the lens (0 = embedding, 2l+1 after layer l+1's attention,
        2l+2 after its MLP) of the reading shown at ``t``, and how long ago it changed."""
        l, u = self.layer_at(t)
        if l < 0:
            return 0, t - self.Tm["lens0"][0]
        s = 2 * l
        since = t - self.starts[l]
        if u >= STEPS["lens1"][0]:
            s, since = 2 * l + 1, t - (self.starts[l] + STEPS["lens1"][0] * self.durations[l])
        if u >= STEPS["lens2"][0]:
            s, since = 2 * l + 2, t - (self.starts[l] + STEPS["lens2"][0] * self.durations[l])
        return s, since

    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        if t >= self.Tm["final"][0]:
            return 4
        l, _u = self.layer_at(t)
        return 0 if l < 0 else (1 if l == 0 else (2 if l < 9 else 3))

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        self._tokens(sc, t, back)
        self._block(sc, t, back)
        self._lens(sc, t, back)
        self._chart(sc, t, back)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, reset)

    def _tokens(self, sc, t, back):
        """The prompt's tokens as boxes, the last one highlighted."""
        on = smoothstep(0.0, 1.0, self.progress("tokens", t)) * back
        if on <= 0.01:
            return
        hot = smoothstep(0.0, 1.0, self.progress("last", t))
        x = 0.0
        p0 = self.P(*TOKENS_AT)
        for i, tokn in enumerate(self.tr.tokens):
            w = layout.text_width(tokn, layout.MIN_TEXT, "mono") * self.cam.width + 14
            c = self.shift(p0, x + w / 2, 0.0)
            last = i == self.T - 1
            self.rect(sc, c, w, BOX_H, C.mix(BOX_FILL, HOT_FILL, hot if last else 0.0), on,
                      border=C.mix(BOX_EDGE, DATA, hot if last else 0.0), border_w=1.2,
                      radius=5, lift=0.2)
            self.label(sc, c, tokn, layout.MIN_TEXT, opacity=on, kind="mono",
                       ink=layout.STRONG_INK)
            x += w + TOKEN_PAD
        self.label(sc, self.shift(p0, x + 4, 0.0), "the last token's %d numbers, position %d"
                   % (G.D_MODEL, self.T - 1), layout.MIN_TEXT, anchor="lm", opacity=on * hot,
                   ink=layout.NOTE_INK)

    def _row_c(self, n):
        """World centre of row ``n``'s strip."""
        return self.P((STRIP_X0 + STRIP_X1) / 2, ROW0_Y + n * ROW_DY)

    @property
    def strip_w(self):
        """Width of a strip in px."""
        return (self.px(self.P(STRIP_X1, 0.5)) - self.px(self.P(STRIP_X0, 0.5)))[0]

    def _strip(self, sc, n, colors, op, wipe=1.0, flash=0.0, label=None, note=None, dy=0.0):
        """Row ``n``: its label, strip (revealed left to right by ``wipe``) and note."""
        if op <= 0.01 or wipe <= 0:
            return
        c = self.shift(self._row_c(n), 0.0, dy)
        w = self.strip_w
        k = max(1, int(round(len(colors) * wipe)))
        cw = self.shift(c, -w / 2 + w * wipe / 2, 0.0)
        draw_strip(self, sc, cw, colors[:k], w * wipe, STRIP_H, op)
        if flash > 0:
            self.rect(sc, cw, w * wipe, STRIP_H, (1.0, 1.0, 1.0), op * 0.35 * flash, radius=1,
                      lift=0.5)
        if label:
            self.label(sc, self.P(LABEL_X, 0.5), label, layout.MIN_TEXT, anchor="rm",
                       dy=(self.px(c) - self.px(self.P(LABEL_X, 0.5)))[1], opacity=op)
        if note:
            self.label(sc, c, note, layout.MIN_TEXT, anchor="lm", dx=w / 2 + NOTE_DX,
                       opacity=op, ink=layout.CAPTION_INK, kind="mono")

    def _block(self, sc, t, back):
        """The eight rows of the layer under way (the embedding before the layers)."""
        l, u = self.layer_at(t)
        if l < 0:
            x0 = ease_out(self.progress("x0", t))
            if x0 > 0:
                self._strip(sc, 0, self.colors["x_in"][0], back, x0, label="x = wte[%d] + wpe[%d]"
                            % (self.tr.ids[-1], self.T - 1),
                            note="|x| = %.1f" % self.norms["x_in"][0])
            return
        d = self.durations[l]
        slide = ease_in_out(remap(u, *STEPS["slide"])) if l < G.N_LAYER - 1 else 0.0
        others = 1 - smoothstep(0.0, 0.6, slide)
        layer = "layer %d" % (l + 1)
        for n, (key, label, _kind) in enumerate(ROWS):
            if key == "x_in":
                op, wipe, flash, dy = back * others, 1.0, 0.0, 0.0
                label = "x in (%s)" % layer
            elif key == "x_out":
                v = remap(u, *STEPS["add2"])
                op, wipe = back * (v > 0), ease_out(v)
                flash = max(0.0, 1 - (u - STEPS["add2"][1]) * d / FLASH) if u > STEPS["add2"][1] \
                    else 1.0
                dy = -(self.px(self._row_c(n)) - self.px(self._row_c(0)))[1] * slide
                label = "x out (%s)" % layer
            else:
                a, b = STEPS[ROW_STEP[key]]
                v = remap(u, a, b)
                op, wipe = back * others * (v > 0), ease_out(v)
                flash = max(0.0, 1 - (u - b) * d / FLASH) if u > b else 1.0
                dy = 0.0
            note = None
            if key in NORMS:
                note = "|%s| = %.1f" % ("x" if key.startswith("x") else "v", self.norms[key][l])
            elif key == "hidden":
                note = "%d: %.0f%% > 0" % (G.D_MLP, 100 * self.hot[l])
            if key == "x_out" and slide > 0:        # sliding up: no note, no label
                note = label = None
            self._strip(sc, n, self.colors[key][l], op, wipe, flash, label, note, dy)
            if key in ("x_mid", "x_out") and wipe > 0 and slide <= 0:
                self.label(sc, self._row_c(n), "+", layout.CAPTION, anchor="lm",
                           dx=-self.strip_w / 2 + 2, dy=-ROW_DY * 0.5 * self.cam.height *
                           self.cam.zoom, opacity=op, ink=layout.STRONG_INK)

    def _lens(self, sc, t, back):
        """The logit lens panel: the top tokens of the current reading and p(" four")."""
        on = smoothstep(0.0, 0.6, self.progress("lens0", t)) * back
        if on <= 0.01:
            return
        s, since = self.reading(t)
        final = t >= self.Tm["final"][0]
        flash = max(0.0, 1 - since / FLASH)
        grow = ease_out(min(1.0, since / 0.5))
        top = self.top[s][:TOP_FINAL if final else TOP]
        head = self.P(LENS_X, LENS_Y)
        if final:
            what = "the output (after layer %d)" % G.N_LAYER
        elif s == 0:
            what = "logit lens: the embedding"
        else:
            what = "after layer %d's %s" % ((s + 1) // 2, "attention" if s % 2 else "MLP")
        self.label(sc, head, what, layout.MIN_TEXT, anchor="lm", opacity=on,
                   ink=layout.NOTE_INK)
        p0 = self.P(LENS_X, BARS_Y)
        rows = [(tokn, p, tokn == TARGET) for tokn, p in top]
        if not any(hit for _t, _p, hit in rows):
            rows.append((TARGET, float(self.p_target[s]), True))
        for n, (tokn, p, hit) in enumerate(rows):
            dy = n * LENS_ROW + (TARGET_DY if n >= len(top) else 0.0)
            c = self.shift(p0, LENS_NAME_W, dy)
            self.label(sc, c, '"%s"' % tokn, layout.MIN_TEXT, anchor="rm", dx=-8, opacity=on,
                       kind="mono", ink=layout.STRONG_INK if hit else layout.CAPTION_INK)
            w = max(1.5, LENS_BAR_W * p * grow)
            self.rect(sc, self.shift(c, w / 2, 0.0), w, LENS_ROW * 0.55,
                      C.mix(BAR_HI if hit else BAR, (1.0, 1.0, 1.0), 0.4 * flash), on,
                      radius=2, lift=0.3)
            self.label(sc, c, self._pct(p), layout.MIN_TEXT, anchor="lm", dx=w + 7,
                       opacity=on, kind="mono", ink=layout.STRONG_INK if hit else
                       layout.CAPTION_INK)

    @staticmethod
    def _pct(p):
        """A probability as a percentage with the digits it needs: 46%, 0.04%, 0.0003%."""
        if p >= 0.1:
            return "%.0f%%" % (100 * p)
        if p >= 0.0005:
            return "%.2f%%" % (100 * p)
        if p >= 0.0000005:
            return "%.4f%%" % (100 * p)
        return "<0.0001%"

    def _chart(self, sc, t, back):
        """p(" four") after every attention and MLP, against the layer."""
        on = smoothstep(0.0, 0.6, self.progress("lens0", t)) * back
        if on <= 0.01:
            return
        s, _since = self.reading(t)
        x0, y0, x1, y1 = CHART
        a, b = self.P(x0, y0), self.P(x1, y1)
        w, h = self.px(b) - self.px(a)
        self.rect(sc, (a + b) / 2, w, h, PANEL, on, border=PANEL_EDGE, border_w=1.0, radius=4,
                  lift=0.1)
        n_s = 2 * G.N_LAYER

        def pt(i, p):
            fx = CHART_PAD + (w - 2 * CHART_PAD) * i / n_s
            fy = CHART_PAD + (h - 2 * CHART_PAD) * (1 - p)
            return self.shift(a, fx, fy) + self.view * 0.3

        pts = [pt(i, self.p_target[i]) for i in range(s + 1)]
        if len(pts) > 1:
            self.polyline(sc, pts, width=2.0, color=DATA, opacity=0.95 * on)
        self.disk(sc, pts[-1], 3.4, DATA, on, lift=0.6, outline=0.0)
        self.label(sc, pt(0, 1.0), 'p(" four")', layout.MIN_TEXT, anchor="lt", dx=4, dy=-2,
                   opacity=on, ink=layout.NOTE_INK)
        for v in (1.0, 0.0):
            self.label(sc, pt(0, v), "%d" % v, layout.MIN_TEXT, anchor="rm", dx=-10, opacity=on,
                       ink=layout.NOTE_INK, kind="mono")
        for l in (0, 4, 8, 12):
            self.label(sc, pt(2 * l, 0.0), "%d" % l, layout.MIN_TEXT, anchor="mt", dy=10,
                       opacity=on, ink=layout.NOTE_INK, kind="mono")
        self.label(sc, pt(n_s / 2, 0.0), "layer", layout.MIN_TEXT, anchor="mt", dy=26,
                   opacity=on, ink=layout.NOTE_INK)

    # ------------------------------------------------------------------ words
    def _words(self, t, reset):
        """The subtitle at ``t``: what the step under way does, with its numbers."""
        Tm = self.Tm
        if reset > 0 or t < Tm["last"][0]:
            return ('"%s" is %d tokens; follow the last one through the %d layers: its %d '
                    'numbers' % (PROMPT, self.T, G.N_LAYER, G.D_MODEL))
        if t < Tm["lens0"][0]:
            return ("x = its token embedding + its position embedding: %d numbers, |x| = %.1f"
                    % (G.D_MODEL, self.norms["x_in"][0]))
        if t < Tm["final"][0]:
            l, u = self.layer_at(t)
            if l < 0:
                return self._lens_words(0, "logit lens: read x like the output (final "
                                           "LayerNorm, × W_E^T, softmax):")
            return self._layer_words(l, u)
        return ('after layer %d the lens is the model\'s own output: final LayerNorm, × W_E^T, '
                'softmax: "%s" %s' % (G.N_LAYER, self.top[-1][0][0], self._pct(self.top[-1][0][1])))

    def _lens_words(self, s, lead):
        """``lead`` followed by reading ``s``'s top token and p(" four") (or the
        runner-up when " four" is the top token)."""
        (w0, p0), (w1, p1) = self.top[s][:2]
        if w0 == TARGET:
            return '%s "%s" %s, then "%s" %s' % (lead, w0, self._pct(p0), w1, self._pct(p1))
        return '%s "%s" %s; "%s" %s' % (lead, w0, self._pct(p0), TARGET,
                                        self._pct(self.p_target[s]))

    def _layer_words(self, l, u):
        """The subtitle in layer ``l`` at progress ``u``: step by step for the
        first layer, one line per layer afterwards."""
        L = l + 1
        if l == 0:
            if u < STEPS["attn"][0]:
                return ("layer 1: LayerNorm scales the %d numbers to mean 0 and spread 1, then "
                        "a learned gain and bias per number" % G.D_MODEL)
            if u < STEPS["add1"][0]:
                return ("attention: %d heads read the other tokens' streams; projected by W_O: "
                        "%d numbers, |v| = %.1f" % (G.N_HEAD, G.D_MODEL, self.norms["attn"][0]))
            if u < STEPS["ln2"][0]:
                return self._lens_words(1, "added to the stream: x + attention. The lens reads")
            if u < STEPS["add2"][0]:
                return ("LayerNorm again, then the MLP on this token alone: %d → %d, GELU "
                        "(%.0f%% of them > 0) → %d"
                        % (G.D_MODEL, G.D_MLP, 100 * self.hot[0], G.D_MODEL))
            return self._lens_words(2, "added to the stream: x + MLP. After layer 1 the lens "
                                       "reads")
        if u < STEPS["lens1"][0]:
            return ("layer %d: LayerNorm, attention, add; LayerNorm, MLP (%.0f%% of %d > 0), "
                    "add. |x| = %.0f in" % (L, 100 * self.hot[l], G.D_MLP, self.norms["x_in"][l]))
        if u < STEPS["lens2"][0]:
            return self._lens_words(2 * l + 1, "layer %d, after attention:" % L)
        return self._lens_words(2 * l + 2, "layer %d, after the MLP (|x| = %.0f out):"
                                % (L, self.norms["x_out"][l]))

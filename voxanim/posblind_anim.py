"""Attention alone cannot see word order: GPT-2's first layer without positions.

A real model, run four ways: GPT-2 small's first attention layer
(:func:`voxanim.posenc.gpt2_first_layer`, all 12 heads, the residual stream
after it, ``x + attn(ln_1(x))``) on three orderings of the same five words,
:data:`voxanim.posenc.ORDERS` -- "the dog bit the man", "the man bit the dog",
"bit the man the dog" -- with its position table ``wpe`` and its causal mask
each switched off or on.  The vector followed is " dog"'s output.  Five phases:

1. **a set** -- the words, head 0's 5 × 5 attention weights (row: the word
   reading, column: the word read) and the 12 heads as thumbnails, with no
   position table and no mask: each word's output is its input plus a
   weighted sum ``Σⱼ aᵢⱼ vⱼ`` over the words, which has no order,
2. **shuffle** -- the words move into the other two orders and the weights'
   rows and columns move with them, unchanged; " dog"'s three output vectors
   are drawn as strips and their differences are blank: the largest
   difference (printed) is float rounding,
3. **positions** -- GPT-2's learned position vectors ``wpe[0 .. 4]`` added to
   the words: the weights change with the order and " dog" differs (the
   cosines with the first order printed),
4. **the mask** -- no positions, but the causal mask: a word sees itself and
   the words before it.  " dog" is last in orders 2 and 3, sees the same
   five words and comes out the same; second in order 1 it sees "the dog",
5. **both** -- positions and mask, as GPT-2 runs: all three differ.

Every number on screen (weights, cosines, differences) is computed at build
time from the GPT-2 weights (:mod:`voxanim.gpt2`, cached).
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout, posenc
from .diagram import EDGE, Diagram, draw_image, draw_strip
from .easing import ease_in_out, remap, smoothstep
from .netanim import div_colors

HEAD = 0                             # the head drawn large
WORD = " dog"                        # the word whose output vector is followed
CONDS = {"none": (False, False), "pos": (True, False), "mask": (False, True),
         "both": (True, True)}       # (position table, causal mask)
PHASE_COND = ("none", "none", "pos", "mask", "both")
PAIRS = ((1, 0), (2, 0), (2, 1))     # the differences drawn (orders, 0-based)
CELL = 44.0                          # px: one attention weight
MATRIX_AT = (0.29, 0.60)
ROW_ASIDE, COL_ASIDE = 44.0, 24.0    # px: how far moving words step aside
THUMB, THUMB_GAP = 40.0, 12.0        # px: the 12 heads' thumbnails
PANEL_X = 0.56                       # layout x of the right panel's left edge
VEC_W, DIFF_W = 300.0, 176.0         # px: the strips' widths
LOW, HIGH = C.rgb("#f3f3ef"), C.rgb("#1d6f69")      # attention weight 0 .. 1
MASKED = C.rgb("#c4c5c3")
ON_INK, OFF_INK = C.rgb("#1f9e94"), C.rgb("#9a9c9e")
DOG_INK = C.rgb("#d84e4e")


def sci(x):
    """``x`` as ``5×10⁻¹⁵`` (one significant digit)."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = round(x / 10.0 ** e)
    if m == 10:
        m, e = 1, e + 1
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return "%d×10%s" % (m, str(e).translate(sup))


def word_ids(tokens):
    """Name each token by word and occurrence (" the" twice: "the1", "the2"), so a
    word can be followed from one order to the next."""
    seen, out = {}, []
    for tk in tokens:
        seen[tk] = seen.get(tk, 0) + 1
        out.append("%s%d" % (tk.strip(), seen[tk]))
    return out


def cosine(a, b):
    """Cosine similarity of two vectors."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def weight_colors(v):
    """Attention weights in [0, 1] -> colours, light (0) to dark teal (1)."""
    a = np.clip(np.asarray(v, np.float64), 0, 1) ** 0.7
    return (LOW[None] + (HIGH - LOW)[None] * a[..., None]).astype(np.float32)


class PositionBlindness(Diagram):
    """GPT-2's first attention layer on three orders of one sentence, without and with
    its position table and causal mask."""

    PHASES = ("a set", "shuffle", "positions", "the mask", "both")

    def __init__(self):
        super().__init__("Attention without positions", "", 10.0)
        self.hold = 2.5
        self.att, self.vec, self.tokens = {}, {}, []
        for name, (pos, causal) in CONDS.items():
            A, V = [], []
            for text in posenc.ORDERS:
                toks, a, out = posenc.gpt2_first_layer(text, pos, causal)
                A.append(a)
                V.append(out[toks.index(WORD)])
                if name == "none":
                    self.tokens.append(toks)
            self.att[name], self.vec[name] = np.array(A), np.array(V)
        self.ids = [word_ids(tk) for tk in self.tokens]
        self.words = self.ids[0]                    # the words, in the first order
        self.slot = np.array([[ids.index(w) for w in self.words] for ids in self.ids])
        self.text = {w: tk.strip() for w, tk in zip(self.ids[0], self.tokens[0])}
        self.dog = self.words.index(WORD.strip() + "1")
        n = len(self.words)
        # the head-0 weights per condition and order, indexed by word (reader, read)
        self.W = {c: np.array([[[self.att[c][k, HEAD, self.slot[k, i], self.slot[k, j]]
                                 for j in range(n)] for i in range(n)] for k in range(3)])
                  for c in CONDS}
        self.masked = np.array([[[self.slot[k, j] > self.slot[k, i] for j in range(n)]
                                 for i in range(n)] for k in range(3)], float)
        self.diff = {c: [(float(np.abs(self.vec[c][a] - self.vec[c][b]).max()),
                          cosine(self.vec[c][a], self.vec[c][b])) for a, b in PAIRS]
                     for c in CONDS}
        self.vmax = float(np.percentile(np.abs(self.vec["none"]), 95))
        pdiff = np.abs(self.vec["pos"][1:] - self.vec["pos"][:1])
        self.dvmax = float(np.percentile(pdiff, 99))

        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("intro", 1.5, 0.0)
        add("set", 7.0, 1.0)
        for ph in "bcde":
            if ph != "b":
                add(ph + "_on", 1.4, 0.8)
            fast = 0.8 if ph == "e" else 1.2
            add(ph + "_12", 1.3, fast)
            add(ph + "_23", 1.3, fast)
            add(ph + "_31", 1.3, 0.4)
            add(ph + "_res", 3.6 if ph == "e" else 3.2, 1.0)
        self.Tm = Tm
        self.finish(t)
        self.starts = [0.0, Tm["b_12"][0] - 0.6, Tm["c_on"][0] - 0.6, Tm["d_on"][0] - 0.6,
                       Tm["e_on"][0] - 0.6]
        self.marks = self.starts

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _phase_ops(self, t, back):
        """Opacity of each phase's panel: one at a time, with short crossfades."""
        st = self.starts + [None]
        op = []
        for n in range(len(self.PHASES)):
            a = 1.0 if n == 0 else smoothstep(st[n] + 0.1, st[n] + 0.5, t)
            b = 1.0 if st[n + 1] is None else 1.0 - smoothstep(st[n + 1] - 0.5, st[n + 1], t)
            op.append(a * b * back)
        return op

    # ------------------------------------------------------------------ the state
    def _order(self, t):
        """The order on screen: ``(from, to, u)`` with ``u`` the eased move 0..1."""
        cur, nxt, u = 0, 0, 0.0
        for ph in "bcde":
            for name, (a, b) in (("_12", (0, 1)), ("_23", (1, 2)), ("_31", (2, 0))):
                t0, t1 = self.Tm[ph + name]
                if t >= t1:
                    cur, nxt, u = b, b, 0.0
                elif t > t0:
                    return a, b, ease_in_out(remap(t, t0, t1))
        return cur, nxt, u

    def _cond(self, t):
        """The condition on screen: ``(from, to, u)``."""
        cur = "none"
        for ph, c in zip("cde", ("pos", "mask", "both")):
            t0, t1 = self.Tm[ph + "_on"]
            if t >= t1:
                cur = c
            elif t > t0:
                return cur, c, ease_in_out(remap(t, t0, t1))
        return cur, cur, 0.0

    def _seen(self, t, ph):
        """How far each of the three orders has been shown in phase ``ph`` (0..1)."""
        if ph == "b":
            first = smoothstep(self.starts[1], self.starts[1] + 0.5, t)
        else:
            first = 1.0
        return (first, smoothstep(*self.Tm[ph + "_12"], t), smoothstep(*self.Tm[ph + "_23"], t))

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ops = self._phase_ops(t, back)
        show = smoothstep(0.0, 1.0, self.progress("intro", t)) * back
        dn, dp, dm, db = (self.diff[c] for c in ("none", "pos", "mask", "both"))
        self.subtitle = [
            "GPT-2's first attention layer with no position table and no causal mask: each "
            "word's output adds a weighted sum Σⱼ aᵢⱼ vⱼ over the words, a sum over a set",
            "shuffle the words: head %d's weights move with them, row and column, unchanged; "
            "' dog' comes out the same in all three orders, largest difference %s (rounding)"
            % (HEAD, sci(max(d for d, _ in dn))),
            "add GPT-2's learned position vectors wpe[0 .. 4]: the weights now change with the "
            "order, and ' dog' differs, cosine %.3f and %.3f with the first order"
            % (dp[0][1], dp[1][1]),
            "the causal mask alone leaks order too: last in orders 2 and 3, ' dog' sees the "
            "same five words (difference %s); second in order 1 it sees two (cosine %.3f)"
            % (sci(dm[2][0]), dm[0][1]),
            "both, as GPT-2 runs: all three differ (cosines %.3f, %.3f, %.3f); a transformer "
            "needs positions to tell 'dog bit man' from 'man bit dog'"
            % (db[0][1], db[1][1], db[2][1]),
        ][ph]
        self._matrix(sc, t, show)
        self._switches(sc, t, show)
        if ops[0] > 0.01:
            self._heads(sc, t, ops[0] * show)
        for n, key in enumerate("bcde"):
            if ops[n + 1] > 0.01:
                self._vectors(sc, t, key, PHASE_COND[n + 1], ops[n + 1])
        notes = ("GPT-2 small's first attention layer, run without its position table and "
                 "without its causal mask",
                 "GPT-2 small's first attention layer, run without its position table and "
                 "without its causal mask",
                 "the same layer with its position table: x = token vector + wpe[position], "
                 "still no mask",
                 "no position table, but the causal mask: each word sees itself and the words "
                 "before it",
                 "position table and causal mask both on: the layer as GPT-2 runs it")
        for n, text in enumerate(notes):
            if n == 1:
                continue
            o = ops[n] + (ops[1] if n == 0 else 0.0)
            self.label(sc, self.P(0.5, 0.925), text, 0.027, opacity=o * show,
                       ink=layout.STRONG_INK)
        self.step_indicator(sc, ph, self.PHASES)

    # -- the attention matrix (every phase) ----------------------------------------------------
    def _matrix(self, sc, t, op):
        """Head 0's weights for the order and condition on screen; cells follow their words."""
        if op <= 0.01:
            return
        a, b, u = self._order(t)
        ca, cb, cu = self._cond(t)
        W = ((1 - cu) * ((1 - u) * self.W[ca][a] + u * self.W[ca][b])
             + cu * ((1 - u) * self.W[cb][a] + u * self.W[cb][b]))
        mask_on = (1 - cu) * CONDS[ca][1] + cu * CONDS[cb][1]
        M = ((1 - u) * self.masked[a] + u * self.masked[b]) * mask_on
        slot = (1 - u) * self.slot[a] + u * self.slot[b]
        n = len(self.words)
        c = self.P(*MATRIX_AT)

        def at(row, col):
            return self.shift(c, (col - (n - 1) / 2) * CELL, (row - (n - 1) / 2) * CELL)

        cols = weight_colors(W)
        # in a move the numbers hide and the words step aside of each other: those moving
        # right (down) go up (left), those moving left (up) down (further left)
        moving = a != b
        steady = smoothstep(0.9, 0.98, abs(2 * u - 1)) if moving else 1.0
        aside = float(np.clip(1.6 * np.sin(np.pi * u), 0, 1)) if moving else 0.0
        way = np.sign(self.slot[b] - self.slot[a])
        for i in range(n):
            for j in range(n):
                p = at(slot[i], slot[j])
                fill = cols[i, j] * (1 - M[i, j]) + MASKED * M[i, j]
                size = (CELL - 3) * (1 - 0.35 * aside)       # smaller in flight: less overlap
                self.rect(sc, p, size, size, fill, op, radius=3.0, lift=0.4)
                ink = (0.97, 0.97, 0.95) if W[i, j] > 0.42 else layout.STRONG_INK
                self.label(sc, p, "%.2f" % W[i, j], 0.026, ink=ink,
                           opacity=op * steady * (1 - M[i, j]))
        for i, w in enumerate(self.words):
            dog = i == self.dog
            ink = tuple(float(v) for v in DOG_INK) if dog else layout.CAPTION_INK
            dx = -aside * (ROW_ASIDE if way[i] > 0 else 2 * ROW_ASIDE) if way[i] else 0.0
            self.label(sc, self.shift(at(slot[i], 0), -CELL / 2 - 8 + dx, 0), self.text[w],
                       0.028, anchor="rm", opacity=op, ink=ink)
            dy = -aside * COL_ASIDE * way[i]
            self.label(sc, self.shift(at(0, slot[i]), 0, -CELL / 2 - 6 + dy), self.text[w],
                       0.028, anchor="mb", opacity=op, ink=ink)
        # " dog"'s row, outlined
        d = at(slot[self.dog], (n - 1) / 2)
        self.rect(sc, d, n * CELL + 4, CELL + 4, DOG_INK, op, border=DOG_INK, border_w=2.0,
                  radius=5.0, lift=0.8, fill_op=0.0)
        self.label(sc, self.shift(c, 0, n * CELL / 2 + 10), "head %d of %d: row = the word "
                   "reading, column = the word read" % (HEAD, self.att["none"].shape[1]), 0.026,
                   anchor="mt", opacity=op, ink=layout.NOTE_INK)
        sc.probe(("dog-row",), d, tol=6.0)
        sc.probe(("order",), (u * 40, cu * 40, 0), tol=6.0)

    def _switches(self, sc, t, op):
        """Two pills: the position table and the causal mask, each on or off."""
        if op <= 0.01:
            return
        ca, cb, cu = self._cond(t)
        for k, (x, name) in enumerate(((0.12, "position table"), (0.31, "causal mask"))):
            on = (1 - cu) * CONDS[ca][k] + cu * CONDS[cb][k]
            fill = C.mix(OFF_INK, ON_INK, on)
            p = self.P(x, 0.25)
            self.rect(sc, p, 150, 24, C.lighten(fill, 0.78), op, border=fill, border_w=1.5,
                      radius=12.0, lift=0.4)
            word = "on" if on > 0.5 else "off"
            self.label(sc, p, "%s: %s" % (name, word), 0.026, opacity=op,
                       ink=layout.STRONG_INK)

    # -- phase 1: the heads ---------------------------------------------------------------------
    def _heads(self, sc, t, op):
        """The 12 heads' weights as thumbnails, and the sum that reads them."""
        s = self.progress("set", t)
        A = self.att["none"][0]
        x0 = self.P(PANEL_X, 0.40)
        self.label(sc, self.P(PANEL_X, 0.32), "the layer's 12 heads, each its own 5 × 5 weights",
                   0.027, anchor="lm", opacity=op, ink=layout.STRONG_INK)
        for h in range(A.shape[0]):
            r, k = divmod(h, 6)
            c = self.shift(x0, THUMB / 2 + k * (THUMB + THUMB_GAP), r * (THUMB + THUMB_GAP))
            o = op * smoothstep(0.02 * h, 0.02 * h + 0.1, s)
            draw_image(self, sc, c, weight_colors(A[h]), THUMB, THUMB, o, grid=0.0)
            if h == HEAD:
                self.rect(sc, c, THUMB + 6, THUMB + 6, DOG_INK, o, border=EDGE, border_w=1.5,
                          radius=3.0, lift=0.6, fill_op=0.0)
        y = 0.40 + 2 * (THUMB + THUMB_GAP) / 540 / layout.ZOOM + 0.02
        lines = (("the output for ' dog' = its input + Σⱼ a(dog, j) · vⱼ", layout.STRONG_INK),
                 ("(summed over the heads, then projected)", layout.NOTE_INK),
                 ("a weighted sum over the set of words:", layout.CAPTION_INK),
                 ("the same terms in any order give the same sum", layout.CAPTION_INK))
        for n, (text, ink) in enumerate(lines):
            o = op * smoothstep(0.35 + 0.12 * n, 0.47 + 0.12 * n, s)
            self.label(sc, self.P(PANEL_X, y + 0.05 * n + (0.02 if n > 1 else 0.0)), text,
                       0.027, anchor="lm", opacity=o, ink=ink)

    # -- phases 2-5: " dog"'s vectors -----------------------------------------------------------
    def _vectors(self, sc, t, ph, cond, op):
        """" dog"'s output in each order as a strip, and the three differences."""
        seen = self._seen(t, ph)
        res = self.progress(ph + "_res", t)
        V = self.vec[cond]
        self.label(sc, self.P(PANEL_X, 0.31), "' dog' after the layer, 768 numbers", 0.027,
                   anchor="lm", opacity=op, ink=layout.STRONG_INK)
        for k in range(3):
            o = op * seen[k]
            y = 0.37 + 0.105 * k
            self.label(sc, self.P(PANEL_X, y), "%d   %s" % (k + 1, posenc.ORDERS[k].strip()),
                       0.026, anchor="lm", opacity=o)
            c = self.shift(self.P(PANEL_X, y + 0.045), VEC_W / 2, 0)
            draw_strip(self, sc, c, div_colors(V[k], self.vmax), VEC_W, 16, o, lift=0.5)
        o = op * smoothstep(0.0, 0.25, res)
        self.label(sc, self.P(PANEL_X, 0.70), "differences (on a scale %.0f× finer)"
                   % (self.vmax / self.dvmax), 0.026, anchor="lm", opacity=o,
                   ink=layout.NOTE_INK)
        for n, ((a, b), (dmax, cs)) in enumerate(zip(PAIRS, self.diff[cond])):
            on = op * smoothstep(0.1 + 0.2 * n, 0.3 + 0.2 * n, res)
            y = 0.75 + 0.048 * n
            p = self.P(PANEL_X, y)
            self.label(sc, p, "%d − %d" % (a + 1, b + 1), 0.026, anchor="lm", opacity=on)
            c = self.shift(p, 44 + DIFF_W / 2, 0)
            draw_strip(self, sc, c, div_colors(V[a] - V[b], self.dvmax), DIFF_W, 14, on,
                       lift=0.5)
            text = ("the same (%s)" % sci(dmax)) if dmax < 1e-9 else "cosine %.3f" % cs
            self.label(sc, self.shift(p, 52 + DIFF_W, 0), text, 0.026, anchor="lm", opacity=on,
                       ink=layout.STRONG_INK)

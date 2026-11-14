"""A token's embedding is a row of a table.

The table is the input table ``W_in`` (39 words × 16) of the word2vec study
:mod:`voxanim.w2v`, trained on its small generated corpus.  Four phases:

1. **lookup** -- the 39-word vocabulary, numbered in sorted order, and the
   table drawn as a heatmap (orange +, blue −); the corpus's first five-word
   sentence, "the sheep chases the cat", tokenised to its ids, and each id's
   row sliding out of the table to make the sentence's 5 × 16 matrix ("the"
   twice: the same row twice),
2. **one-hot** -- the same row as a product: a one-hot vector ``x`` (39 long,
   a single 1 at "sheep"'s id) times the table, ``x · W_in = Σᵢ xᵢ W_in[i]``:
   38 rows are multiplied by 0, one by 1.  The product is computed with numpy
   and compared with the row: the largest difference is printed (it is 0),
3. **cost** -- the work per token on a log scale: a lookup copies 16 numbers;
   the product does 39 × 16 multiply-adds, all but 16 of them by 0.  At
   GPT-2 small's size (``gpt2.N_VOCAB`` × ``gpt2.D_MODEL`` = 50,257 × 768,
   constants only, nothing downloaded) the lookup copies 768 numbers and the
   product does 38,597,376 multiply-adds,
4. **training** -- one SGNS update, the same one the ``word2vec-training``
   animation shows: centre "baker", context "bread" (they occur together in
   the corpus), 5 noise words drawn with seed 0 from the study's noise
   distribution, at the **exact** training state of snapshot 6 (step 1,403;
   the learning rate the schedule had there, 0.45).  The study stores
   ``W_in`` per snapshot but only the final ``W_out``;
   :func:`voxanim.w2v_anim.replay` reruns the study's training loop to that
   step and checks its ``W_in`` against the stored snapshot bit for bit, so
   ``W_out`` is the real one there, not the final one standing in.  The
   step is :func:`w2v.sgns_step`; both whole tables are updated, and the rows
   that changed are counted by comparing them before and after: 1 row of
   ``W_in`` (the centre) and 6 of ``W_out`` (the context and the noise words),
   every other row bit-for-bit unchanged.  The update itself, ``ΔW``, is
   drawn on its own scale, then the tables after; and the pair's loss before
   and after.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import gpt2, layout, w2v
from .diagram import DATA, EDGE, GRAD, Diagram, draw_image
from .easing import ease_in_out, ease_out, remap, smoothstep
from .netanim import amax, div_colors
from .text import TIMES, num, sub
from .w2v_anim import training_step

LOOKUP_LEN = 5                         # the sentence: the corpus's first of this many words
ONE_HOT_WORD = "sheep"
CELL = 8.0                             # px: a cell of the 39 × 16 tables
TABLE_AT = (0.47, 0.30)                # the table's top centre (phases 1 and 2)
OUT_X, OUT_Y, OUT_DY = 0.80, 0.52, 0.056   # the sentence matrix: centre x, first row, pitch
STRIP_W, STRIP_H = 224.0, 16.0
VOCAB_COLS, VOCAB_X, VOCAB_Y, VOCAB_DX, VOCAB_DY = 3, 0.0, 0.345, 0.108, 0.047
TOKEN_X, TOKEN_DX, TOKEN_Y = 0.635, 0.078, 0.31
FADED = C.rgb("#e4e5e3")               # a row multiplied by 0
TRAIN_AT = (0.08, 0.40, 0.30)          # phase 4: W_in's and W_out's centre x, their top
LABEL_PITCH = 21.0                     # px between row labels


def spread(ys, pitch, lo, hi):
    """Positions as close to ``ys`` (sorted) as possible, at least ``pitch``
    apart, inside ``[lo, hi]``: labels for rows that may be neighbours."""
    out = []
    for y in ys:
        out.append(max(y, out[-1] + pitch) if out else max(y, lo))
    over = out[-1] - hi if out else 0.0
    if over > 0:
        out = [y - over for y in out]
        for n in range(len(out) - 2, -1, -1):
            out[n] = min(out[n], out[n + 1] - pitch)
    return out


class EmbeddingLookup(Diagram):
    """An embedding as a table lookup, the same as one-hot times the table, and a
    training step that touches only the rows used (see the module docstring)."""

    PHASES = ("lookup", "one-hot", "cost", "training")

    def __init__(self):
        super().__init__("A token's embedding is a row of a table", "", 10.0)
        self.hold = 2.5
        run = self.run = w2v.train()
        self.words = run.words
        index = run.index
        self.W = run.W_in
        self.V, self.D = self.W.shape
        self.vmax = amax(self.W)
        self.tex = div_colors(self.W.ravel(), self.vmax).reshape(self.V, self.D, 3)
        sents = w2v.corpus()
        self.sentence = next(s for s in sents if len(s) == LOOKUP_LEN)
        self.ids = [index[w] for w in self.sentence]
        self.matrix = self.W[self.ids]                       # (5, 16): the lookup
        # one-hot times the table, against the row
        self.hot = index[ONE_HOT_WORD]
        x = np.zeros(self.V)
        x[self.hot] = 1.0
        self.product = x @ self.W
        self.diff = float(np.abs(self.product - self.W[self.hot]).max())
        # the cost
        self.copy_small, self.mac_small = self.D, self.V * self.D
        self.copy_gpt2, self.mac_gpt2 = gpt2.D_MODEL, gpt2.N_VOCAB * gpt2.D_MODEL
        # the training step: whole tables before and after
        st = self.st = training_step()
        self.changed_in = np.flatnonzero(np.any(st["W_in1"] != st["W_in0"], axis=1))
        self.changed_out = np.flatnonzero(np.any(st["W_out1"] != st["W_out0"], axis=1))
        self.unchanged = 2 * self.V - len(self.changed_in) - len(self.changed_out)
        both = np.concatenate([st["W_in0"], st["W_out0"], st["W_in1"], st["W_out1"]])
        tmax = amax(both)
        self.tr_tex = {k: div_colors(st[k].ravel(), tmax).reshape(self.V, self.D, 3)
                       for k in ("W_in0", "W_out0", "W_in1", "W_out1")}
        d_in, d_out = st["W_in1"] - st["W_in0"], st["W_out1"] - st["W_out0"]
        dmax = max(amax(d_in), amax(d_out))
        self.d_tex = {"in": div_colors(d_in.ravel(), dmax).reshape(self.V, self.D, 3),
                      "out": div_colors(d_out.ravel(), dmax).reshape(self.V, self.D, 3)}
        self.d_rest = max(float(np.abs(np.delete(d_in, self.changed_in, 0)).max()),
                          float(np.abs(np.delete(d_out, self.changed_out, 0)).max()))

        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("vocab", 2.5, 0.2)
        add("tokens", 4.0, 0.2)
        add("slide", 5.5, 2.0)
        add("onehot", 2.5, 0.3)
        add("multiply", 3.0, 0.3)
        add("sum", 2.5, 3.0)
        add("cost", 8.0, 3.0)
        add("tables", 1.5, 0.3)
        add("pick", 2.5, 0.5)
        add("grad", 3.0, 1.5)
        add("after", 3.0, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["onehot"][0] - 0.5, Tm["cost"][0] - 0.5, Tm["tables"][0] - 0.5]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _phase_ops(self, t, back):
        """Opacity of each phase's drawing: one at a time, with short crossfades.  The
        table of phases 1 and 2 stays put across their change (``table``)."""
        keys = ("lookup", "onehot", "cost", "train")
        ends = self.marks[1:]
        op = {}
        for n, key in enumerate(keys):
            a = 1.0 if n == 0 else smoothstep(ends[n - 1] + 0.1, ends[n - 1] + 0.5, t)
            b = 1.0 if n == len(keys) - 1 else 1.0 - smoothstep(ends[n] - 0.6, ends[n], t)
            op[key] = a * b * back
        op["table"] = (1.0 - smoothstep(ends[1] - 0.6, ends[1], t)) * back
        return op

    # ------------------------------------------------------------------ geometry
    def _row_at(self, top, i, cx=0.0):
        """World centre of row ``i`` of a 39-row table whose top centre is ``top``."""
        return self.shift(top, cx, (i + 0.5) * CELL)

    def _table(self, sc, top, tex, op, title=None):
        """A 39 × 16 table hanging from ``top`` (its top centre)."""
        h, w = self.V * CELL, self.D * CELL
        draw_image(self, sc, self.shift(top, 0, h / 2), tex, w, h, op, grid=0.25)
        if title:
            self.label(sc, top, title, 0.027, anchor="mb", dy=-8, opacity=op,
                       ink=layout.STRONG_INK)

    def _ring(self, sc, c, w, h, ink, op):
        """An outline around a row or an entry."""
        self.rect(sc, c, w, h, ink, op, border=ink, border_w=1.6, radius=2, lift=0.6,
                  fill_op=0.0)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ops = self._phase_ops(t, back)
        st = self.st
        ids = " ".join(str(i) for i in self.ids)
        self.subtitle = [
            "\"%s\": %d tokens, ids %s; each id picks its row of the table W_in (%d %s %d), "
            "giving the sentence a %d %s %d matrix" % (
                " ".join(self.sentence), len(self.ids), ids, self.V, TIMES, self.D,
                len(self.ids), TIMES, self.D),
            "the same row as a product: a one-hot x (%d long, a single 1 at %d) times W_in "
            "gives W_in[%d]; the largest difference is %g" % (self.V, self.hot, self.hot,
                                                             self.diff),
            "per token a lookup copies %d numbers, the product does %d %s %d = %s multiply-adds; "
            "at GPT-2 small's %s %s %d: %d against %.1f million" % (
                self.copy_small, self.V, TIMES, self.D, format(self.mac_small, ","),
                format(gpt2.N_VOCAB, ","), TIMES, gpt2.D_MODEL, self.copy_gpt2,
                self.mac_gpt2 / 1e6),
            "one SGNS step on (%s, %s) at training step %s: %d row of W_in and %d of W_out "
            "change, the other %d not at all; the pair's loss %.2f → %.2f" % (
                st["centre"], st["context"], format(st["step"], ","), len(self.changed_in),
                len(self.changed_out), self.unchanged, st["loss0"], st["loss1"]),
        ][ph]
        if ops["table"] > 0.01:
            self._shared_table(sc, t, ops["table"])
        if ops["lookup"] > 0.01:
            self._lookup(sc, t, ops["lookup"])
        if ops["onehot"] > 0.01:
            self._onehot(sc, t, ops["onehot"])
        if ops["cost"] > 0.01:
            self._cost(sc, t, ops["cost"])
        if ops["train"] > 0.01:
            self._training(sc, t, ops["train"])
        self.step_indicator(sc, ph, self.PHASES)

    # -- phases 1 and 2: the table ------------------------------------------------------------
    def _shared_table(self, sc, t, op):
        """The trained table, faded row by row while the one-hot multiplies it."""
        on = op * smoothstep(0.0, 0.6, self.progress("vocab", t))
        m = smoothstep(0.0, 0.6, self.progress("multiply", t)) * (
            1.0 - smoothstep(0.0, 1.0, self.progress("cost", t)))
        tex = self.tex
        if m > 0:
            tex = tex.copy()
            keep = np.ones(self.V, bool)
            keep[self.hot] = False
            tex[keep] = tex[keep] + (FADED - tex[keep]) * np.float32(m)
        top = self.P(*TABLE_AT)
        self._table(sc, top, tex, on, "W_in: %d %s %d" % (self.V, TIMES, self.D))
        for i in (0, 10, 20, 30, self.V - 1):
            self.label(sc, self._row_at(top, i, -self.D * CELL / 2), str(i), 0.025,
                       anchor="rm", dx=-6, opacity=on * (1 - smoothstep(0.0, 0.5, self.progress(
                           "onehot", t))), ink=layout.NOTE_INK, kind="mono")

    # -- phase 1: the lookup ------------------------------------------------------------------
    def _vocab_at(self, i):
        """Where entry ``i`` of the numbered vocabulary sits."""
        col, row = divmod(i, (self.V + VOCAB_COLS - 1) // VOCAB_COLS)
        return VOCAB_X + VOCAB_DX * col, VOCAB_Y + VOCAB_DY * row

    def _lookup(self, sc, t, op):
        """The vocabulary, the tokens and their ids, the rows sliding out."""
        voc = self.progress("vocab", t)
        tok = self.progress("tokens", t)
        sl = self.progress("slide", t)
        on = op * smoothstep(0.0, 0.5, voc)
        self.label(sc, self.P(VOCAB_X, 0.285), "the vocabulary: %d words, numbered" % self.V,
                   0.027, anchor="lm", opacity=on, ink=layout.STRONG_INK)
        n = len(self.ids)
        found = [smoothstep(0.15 + 0.16 * k, 0.27 + 0.16 * k, tok) for k in range(n)]
        hot = {}
        for k, i in enumerate(self.ids):
            hot[i] = max(hot.get(i, 0.0), found[k] * (1.0 - smoothstep(0.85, 1.0, sl)))
        for i, w in enumerate(self.words):
            fx, fy = self._vocab_at(i)
            self.label(sc, self.P(fx + 0.028, fy), str(i), 0.025, anchor="rm", opacity=on,
                       kind="mono", ink=layout.NOTE_INK)
            self.label(sc, self.P(fx + 0.036, fy), w, 0.026, anchor="lm", opacity=on,
                       ink=layout.STRONG_INK if hot.get(i, 0) > 0.5 else layout.CAPTION_INK)
            if hot.get(i, 0) > 0.01:
                self._ring(sc, self.P(fx + 0.047, fy), 86, 20, DATA, op * hot[i])
        # the sentence and its ids
        show = op * smoothstep(0.0, 0.12, tok)
        self.label(sc, self.P(TOKEN_X + TOKEN_DX * (n - 1) / 2, 0.255), "a sentence of the "
                   "corpus, tokenised", 0.026, opacity=show, ink=layout.NOTE_INK)
        for k, (w, i) in enumerate(zip(self.sentence, self.ids)):
            x = TOKEN_X + TOKEN_DX * k
            self.label(sc, self.P(x, TOKEN_Y), w, 0.029, opacity=show, ink=layout.STRONG_INK)
            self.label(sc, self.P(x, TOKEN_Y + 0.055), str(i), 0.028, opacity=op * found[k],
                       kind="mono", ink=tuple(float(v) for v in C.darken(DATA, 0.25)))
        # rows highlighted in the table, then sliding out
        top = self.P(*TABLE_AT)
        tw = self.D * CELL
        for k, i in enumerate(self.ids):
            src = self._row_at(top, i)
            u = ease_in_out(remap(sl, 0.06 + 0.15 * k, 0.36 + 0.15 * k))
            ring = found[k] * (1.0 - smoothstep(0.85, 1.0, sl))
            if ring > 0.01:
                self._ring(sc, src, tw + 6, CELL + 4, DATA, op * ring)
            if u <= 0:
                continue
            dst = self.P(OUT_X, OUT_Y + OUT_DY * k)
            c = src + (dst - src) * u
            w = tw + (STRIP_W - tw) * u
            h = CELL + (STRIP_H - CELL) * u
            draw_image(self, sc, c, self.tex[i][None], w, h, op, grid=0.25, lift=0.5)
            sc.probe(("row", k), c, tol=5.0)
            if u >= 1:
                self.label(sc, dst, "%s  %d" % (self.sentence[k], i), 0.026, anchor="rm",
                           dx=-(STRIP_W / 2 + 10), opacity=op, ink=layout.CAPTION_INK)
        done = op * smoothstep(0.8, 0.95, sl)
        y_end = OUT_Y + OUT_DY * (n - 1)
        self.label(sc, self.P(OUT_X, y_end + 0.07), "the sentence: a %d %s %d matrix" % (
            n, TIMES, self.D), 0.027, opacity=done, ink=layout.STRONG_INK)
        self.label(sc, self.P(OUT_X, y_end + 0.12), "no arithmetic: each row is a copy", 0.026,
                   opacity=done, ink=layout.NOTE_INK)
        if self.ids.count(self.ids[0]) > 1:
            self.label(sc, self.P(OUT_X, y_end + 0.165), "\"%s\" twice: the same row twice"
                       % self.sentence[0], 0.026, opacity=done, ink=layout.NOTE_INK)

    # -- phase 2: one-hot times the table --------------------------------------------------------
    def _onehot(self, sc, t, op):
        """x as a column beside the table's rows; 38 rows times 0, one times 1; the sum."""
        oh = self.progress("onehot", t)
        mu = self.progress("multiply", t)
        su = self.progress("sum", t)
        top = self.P(*TABLE_AT)
        tw = self.D * CELL
        xcol = self.shift(top, -tw / 2 - 26, 0)
        show = op * smoothstep(0.0, 0.4, oh)
        tex = np.tile(C.rgb("#f2f2ef"), (self.V, 1, 1))
        tex[self.hot, 0] = C.darken(DATA, 0.1)
        h = self.V * CELL
        draw_image(self, sc, self.shift(xcol, 0, h / 2), tex, CELL, h, show, grid=0.25)
        self.label(sc, xcol, "x", 0.029, anchor="mb", dy=-8, opacity=show,
                   ink=layout.STRONG_INK)
        hot_row = self._row_at(xcol, self.hot)
        self.label(sc, hot_row, "1 at %d (%s)" % (self.hot, ONE_HOT_WORD), 0.026, anchor="rm",
                   dx=-14, opacity=show, ink=layout.STRONG_INK)
        self.label(sc, self._row_at(xcol, 12), "0 everywhere else", 0.026, anchor="rm",
                   dx=-14, opacity=show, ink=layout.NOTE_INK)
        # the sums, written out at the left
        formula = "x · W_in = x%s W_in[0] + … + x%s W_in[%d]" % (sub(0), sub(self.V - 1),
                                                                 self.V - 1)
        self.label(sc, self.P(0.0, 0.33), formula, 0.029, anchor="lm", opacity=show,
                   ink=layout.STRONG_INK)
        mon = op * smoothstep(0.0, 0.3, mu)
        self.label(sc, self.P(0.0, 0.40), "%d rows times 0" % (self.V - 1), 0.026,
                   anchor="lm", opacity=mon, ink=layout.CAPTION_INK)
        self.label(sc, self.P(0.0, 0.45), "1 row times 1", 0.026, anchor="lm", opacity=mon,
                   ink=layout.CAPTION_INK)
        # the surviving row slides out as the result
        u = ease_in_out(remap(su, 0.0, 0.55))
        src = self._row_at(top, self.hot)
        dst = self.P(OUT_X, 0.50)
        if u > 0:
            c = src + (dst - src) * u
            w = tw + (STRIP_W - tw) * u
            hh = CELL + (STRIP_H - CELL) * u
            draw_image(self, sc, c, self.tex[self.hot][None], w, hh, op, grid=0.25, lift=0.5)
            sc.probe(("hot",), c, tol=5.0)
        res = op * smoothstep(0.5, 0.7, su)
        self.label(sc, dst, "x · W_in", 0.027, anchor="mb", dy=-STRIP_H / 2 - 6, opacity=res,
                   ink=layout.STRONG_INK)
        row = self.P(OUT_X, 0.64)
        tex_row = div_colors(self.W[self.hot], self.vmax).reshape(1, self.D, 3)
        draw_image(self, sc, row, tex_row, STRIP_W, STRIP_H, res, grid=0.25)
        self.label(sc, row, "the lookup, W_in[%d]" % self.hot, 0.027, anchor="mb",
                   dy=-STRIP_H / 2 - 6, opacity=res, ink=layout.STRONG_INK)
        eq = op * smoothstep(0.7, 0.9, su)
        self.label(sc, self.P(OUT_X, 0.74), "largest difference: %g" % self.diff, 0.029,
                   opacity=eq, ink=layout.STRONG_INK)
        self.label(sc, self.P(OUT_X, 0.795), "the same 16 numbers, bit for bit", 0.026,
                   opacity=eq, ink=layout.NOTE_INK)

    # -- phase 3: the cost --------------------------------------------------------------------
    def _cost(self, sc, t, op):
        """Work per token, lookup against product, here and at GPT-2's size, on a log scale."""
        u = self.progress("cost", t)
        x0, x1 = 0.30, 0.97
        top = 10 ** 8
        rows = [
            ("this table, %d %s %d" % (self.V, TIMES, self.D), None, None),
            ("lookup: copy %d numbers" % self.copy_small, self.copy_small, DATA),
            ("one-hot × table: %s multiply-adds" % format(self.mac_small, ","),
             self.mac_small, GRAD),
            ("GPT-2 small, %s %s %d" % (format(gpt2.N_VOCAB, ","), TIMES, gpt2.D_MODEL),
             None, None),
            ("lookup: copy %d numbers" % self.copy_gpt2, self.copy_gpt2, DATA),
            ("one-hot × table: %s" % format(self.mac_gpt2, ","), self.mac_gpt2, GRAD),
        ]
        ys = (0.30, 0.38, 0.46, 0.57, 0.65, 0.73)
        self.label(sc, self.P(x0, 0.255), "work per token, log scale", 0.026, anchor="lm",
                   opacity=op, ink=layout.NOTE_INK)
        bar_n = 0
        for (text, value, ink), y in zip(rows, ys):
            if value is None:
                self.label(sc, self.P(0.0, y), text, 0.028, anchor="lm", opacity=op,
                           ink=layout.STRONG_INK)
                continue
            g = ease_out(remap(u, 0.05 + 0.15 * bar_n, 0.25 + 0.15 * bar_n))
            bar_n += 1
            self.label(sc, self.P(0.0, y), text, 0.026, anchor="lm", opacity=op,
                       ink=layout.CAPTION_INK)
            frac = np.log10(value) / np.log10(top) * g
            a = self.P(x0, y)
            full = float((self.px(self.P(x1, y)) - self.px(a))[0])
            w = max(2.0, full * frac)
            self.rect(sc, self.shift(a, w / 2, 0), w, 14, ink, op, radius=2, lift=0.3)
            sc.probe(("bar", bar_n), (w, 0, 0), tol=4.0)
        # the log axis
        ya = 0.80
        a, b = self.P(x0, ya), self.P(x1, ya)
        sc.line(a, b, width=1.0, color=EDGE, opacity=0.7 * op, bias=0.2)
        for e in range(0, 9, 2):
            p = self.P(x0 + (x1 - x0) * e / 8, ya)
            sc.line(p, self.shift(p, 0, 5), width=1.0, color=EDGE, opacity=0.7 * op, bias=0.2)
            self.label(sc, p, "1" if e == 0 else "10%s" % "⁰¹²³⁴⁵⁶⁷⁸"[e], 0.025, anchor="mt",
                       dy=8, opacity=op, ink=layout.NOTE_INK)
        late = op * smoothstep(0.75, 0.9, u)
        zeros_small = self.mac_small - self.copy_small
        zeros_gpt2 = self.mac_gpt2 - self.copy_gpt2
        self.label(sc, self.P(0.5, 0.88), "the product's extra work is all multiplication by 0: "
                   "%s of %s here, %s of %s in GPT-2" % (
                       format(zeros_small, ","), format(self.mac_small, ","),
                       format(zeros_gpt2, ","), format(self.mac_gpt2, ",")), 0.026,
                   opacity=late, ink=layout.CAPTION_INK)
        self.label(sc, self.P(0.5, 0.93), "so models store the table and index it", 0.026,
                   opacity=late, ink=layout.NOTE_INK)

    # -- phase 4: a training step -------------------------------------------------------------
    def _training(self, sc, t, op):
        """W_in and W_out at step 1,403: the rows the step uses, the update, the tables after."""
        st = self.st
        tb = smoothstep(0.0, 1.0, self.progress("tables", t))
        pk = self.progress("pick", t)
        gr = self.progress("grad", t)
        af = self.progress("after", t)
        on = op * tb
        x_in, x_out, y_top = TRAIN_AT
        tops = {"in": self.P(x_in, y_top), "out": self.P(x_out, y_top)}
        # before → ΔW → after, crossfading
        g = smoothstep(0.0, 0.3, gr)
        a = smoothstep(0.0, 0.4, af)
        for key, k0, k1 in (("in", "W_in0", "W_in1"), ("out", "W_out0", "W_out1")):
            tex = self.tr_tex[k0]
            if g > 0:
                tex = tex + (self.d_tex[key] - tex) * np.float32(g)
            if a > 0:
                tex = tex + (self.tr_tex[k1] - tex) * np.float32(a)
            name = "W_%s" % key
            title = name if g < 0.5 else ("Δ" + name if a < 0.5 else name + " after")
            self._table(sc, tops[key], tex, on, title)
        # the rows the step uses, labelled beside their tables
        tw = self.D * CELL
        pick = op * smoothstep(0.0, 0.3, pk)
        groups = (("in", [(st["c"], st["centre"], "centre")]),
                  ("out", [(st["o"], st["context"], "context")] +
                   [(int(i), self.words[int(i)], "noise") for i in st["neg"]]))
        lo = self.px(tops["in"])[1]
        hi = lo + self.V * CELL
        for key, items in groups:
            seen, uniq = set(), []
            for i, w, role in items:
                if i not in seen:
                    seen.add(i)
                    uniq.append((i, w, role))
            uniq.sort()
            rows_px = [self.px(self._row_at(tops[key], i))[1] for i, _w, _r in uniq]
            at = spread(rows_px, LABEL_PITCH, lo + 6, hi - 6)
            for (i, w, role), ry, ly in zip(uniq, rows_px, at):
                ink = DATA if role != "noise" else GRAD
                self._ring(sc, self._row_at(tops[key], i), tw + 6, CELL + 4, ink, pick)
                edge = self._row_at(tops[key], i, tw / 2 + 4)
                lab = self.shift(edge, 22, ly - ry)
                sc.line(self.shift(edge, 0, 0), self.shift(lab, -3, 0), width=1.0, color=ink,
                        opacity=pick, bias=0.6)
                self.label(sc, lab, "%s (%s)" % (w, role), 0.026, anchor="lm", opacity=pick,
                           ink=tuple(float(v) for v in C.darken(ink, 0.25)))
        # the numbers, at the right
        xr = 0.68
        self.label(sc, self.P(xr, 0.33), "training step %s" % format(st["step"], ","), 0.027,
                   anchor="lm", opacity=on, ink=layout.STRONG_INK)
        self.label(sc, self.P(xr, 0.38), "one pair, %s → %s," % (st["centre"], st["context"]),
                   0.027, anchor="lm", opacity=on, ink=layout.CAPTION_INK)
        self.label(sc, self.P(xr, 0.43), "and %d noise words" % len(st["negatives"]), 0.027,
                   anchor="lm", opacity=on, ink=layout.CAPTION_INK)
        gon = op * smoothstep(0.2, 0.5, gr)
        self.label(sc, self.P(xr, 0.52), "the update ΔW is nonzero in", 0.027, anchor="lm",
                   opacity=gon, ink=layout.CAPTION_INK)
        self.label(sc, self.P(xr, 0.57), "%d row of W_in, %d rows of W_out" % (
            len(self.changed_in), len(self.changed_out)), 0.027, anchor="lm", opacity=gon,
            ink=layout.STRONG_INK)
        self.label(sc, self.P(xr, 0.62), "the other %d rows: change %g" % (
            self.unchanged, self.d_rest), 0.027, anchor="lm", opacity=gon,
            ink=layout.CAPTION_INK)
        self.label(sc, self.P(xr, 0.67), "(ΔW on its own colour scale)", 0.025, anchor="lm",
                   opacity=gon * (1 - a), ink=layout.NOTE_INK)
        aon = op * smoothstep(0.3, 0.6, af)
        self.label(sc, self.P(xr, 0.73), "this pair's loss", 0.027, anchor="lm", opacity=aon,
                   ink=layout.CAPTION_INK)
        self.label(sc, self.P(xr, 0.78), "%s → %s" % (num(st["loss0"], 3), num(st["loss1"], 3)),
                   0.029, anchor="lm", opacity=aon, ink=layout.STRONG_INK, kind="mono")
        self.label(sc, self.P(xr, 0.86), "a batch of 128 pairs touches only", 0.026,
                   anchor="lm", opacity=aon, ink=layout.NOTE_INK)
        self.label(sc, self.P(xr, 0.905), "the rows of the words it contains", 0.026,
                   anchor="lm", opacity=aon, ink=layout.NOTE_INK)
        sc.probe(("train",), (gr * 40 + af * 40, 0, 0), tol=4.0)

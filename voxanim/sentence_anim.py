"""Sentence embeddings: a sentence as the mean of its word vectors, and SIF.

The word vectors are the 16-number rows of the word2vec model trained on the
toy corpus of :mod:`voxanim.w2v` (no downloaded data); the six sentences are
:data:`voxanim.w2v.SENTENCES`, embedded by :func:`voxanim.w2v.sentence_vectors`
and compared by :func:`voxanim.w2v.cosine_matrix`.  Five phases:

1. **mean** -- "the farmer eats cheese in the village": its seven word
   vectors drawn as strips of 16 cells, dropped one by one into their running
   mean; column 1 of the mean written out as the sum of the column over 7,
2. **cosine** -- the 6 × 6 cosine similarity of the six sentence means, with
   the numbers: the two eating sentences 0.976, a dog chase against the
   doctor's shopping 0.390, every pair at least that alike,
3. **order** -- "the dog chases the cat" and "the cat chases the dog": the
   same five words, so the same mean (the two means differ only by rounding,
   2 × 10⁻¹⁶) and cosine 1: mean pooling is a bag of words,
4. **weights** -- SIF (Arora et al., 2017): each word weighted by
   ``a / (a + p(w))``, ``a = 0.001``, ``p(w)`` its share of the corpus ("the",
   28% of it, gets 0.0035, "cheese" 0.080); the weighted mean, then its part
   along the first principal direction of 2,000 corpus sentences removed,
5. **SIF** -- the cosine matrix again with SIF: the eating pair 0.847, the dog
   chase against the doctor −0.805; the two dog/cat sentences are still 1.000
   (SIF is still an average); trained sentence encoders read word order.

Every number is computed at build time from the cached word2vec run.
"""

from __future__ import annotations

import numpy as np

from . import layout, w2v
from .diagram import Diagram, draw_image
from .easing import ease_in_out, ease_out, remap, smoothstep
from .netanim import div_colors
from .text import MINUS, num

SENTENCES = w2v.SENTENCES
POOL = 2                            # the sentence pooled in phase 1 (and weighted in 4)
ORDER = (0, 1)                      # the two dog/cat sentences
PAIR_HI = (2, 3)                    # the two eating sentences
PAIR_LO = (0, 5)                    # a dog chase against the doctor's shopping
PAIR_MID = (0, 4)                   # a dog chase against the horse running
COLUMN = 0                          # the column of phase 1 written out
STRIP_CELL = 22.0                   # px per cell of a 16-number strip (phase 1)
ROW_DY = 0.058                      # layout rows of phase 1
MAT_AT, CELL_W, CELL_H = (0.75, 0.60), 50.0, 40.0
LIST_X, LIST_Y, LIST_DY = 0.04, 0.36, 0.062
NEUTRAL_INK = (0.50, 0.52, 0.54)


def _strip_tex(v, vmax):
    """A vector as a one-row texture on the diverging scale."""
    return div_colors(v, vmax).reshape(1, -1, 3)


def _sci(x):
    """``x`` as ``2.2 × 10⁻¹⁶`` (or ``0``)."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10.0 ** e
    sup = str(e).replace("-", "⁻").translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return "%.1f × 10%s" % (m, sup)


class SentenceEmbeddings(Diagram):
    """Mean pooling and SIF over the word2vec vectors of six sentences."""

    PHASES = ("mean", "cosine", "order", "weights", "SIF")

    def __init__(self):
        super().__init__("Sentence embeddings", "", 10.0)
        self.hold = 2.5
        run = w2v.train()
        self.run = run
        idx = run.index
        self.words = SENTENCES[POOL].split()
        ids = [idx[w] for w in self.words]
        self.wvec = run.W_in[ids]                                 # (7, 16)
        self.vmax = float(np.abs(self.wvec).max())
        self.X = w2v.sentence_vectors(run)
        self.Xs = w2v.sentence_vectors(run, sif=True)
        self.S = w2v.cosine_matrix(self.X)
        self.Ss = w2v.cosine_matrix(self.Xs)
        a, b = ORDER
        self.order_diff = float(np.abs(self.X[a] - self.X[b]).max())
        self.order_words = [SENTENCES[a].split(), SENTENCES[b].split()]
        self.order_vecs = [run.W_in[[idx[w] for w in ws]] for ws in self.order_words]
        # the running mean of phase 1: the mean of the first k words
        self.running = np.cumsum(self.wvec, 0) / np.arange(1, len(ids) + 1)[:, None]
        # SIF, step by step, for the pooled sentence
        self.p = run.p[ids]
        self.wt = w2v.SIF_A / (w2v.SIF_A + self.p)
        self.weighted = (self.wt[:, None] * self.wvec).sum(0) / len(ids)
        self.sif = self.Xs[POOL]
        self.removed = self.weighted - self.sif                   # its part along u
        self.removed_frac = float(np.linalg.norm(self.removed) / np.linalg.norm(self.weighted))
        iu = np.triu_indices(len(SENTENCES), 1)
        off = [(i, j) for i, j in zip(*iu) if (i, j) != ORDER]
        self.plain_min = float(min(self.S[i, j] for i, j in off))
        self.sif_neg = sum(self.Ss[i, j] < 0 for i, j in off)
        self.n_off = len(off)
        the = self.words.index("the")
        self.the_i = the
        self.content_i = int(np.argmax(self.wt))

        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("rows", 4.0, 0.0)
        add("pool", 5.0, 0.3)
        add("column", 3.0, 2.2)
        add("cosine", 9.5, 1.0)
        add("order", 9.5, 1.0)
        add("weights", 10.5, 1.0)
        add("sif", 10.5, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.starts = [Tm[k][0] for k in ("cosine", "order", "weights", "sif")]
        self.marks = [0.0] + [s - 0.5 for s in self.starts]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _phase_ops(self, t, back):
        """Opacity of each phase's drawing: one at a time, with short crossfades."""
        ops = []
        for k in range(len(self.PHASES)):
            op = 1.0 if k == 0 else smoothstep(self.starts[k - 1] - 0.4, self.starts[k - 1], t)
            if k < len(self.starts):
                end = self.starts[k] - 0.5
                op *= 1.0 - smoothstep(end - 0.6, end, t)
            ops.append(op * back)
        return ops

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ops = self._phase_ops(t, back)
        i, j = PAIR_HI
        k, m = PAIR_LO
        n_words = len(self.words)
        self.subtitle = [
            "a sentence as one vector: the mean of its %d word vectors, %d numbers each "
            "(word2vec on a toy corpus)" % (n_words, w2v.DIM),
            "cosine similarity of the six means: the two eating sentences %.3f, a dog chase "
            "and the doctor's shopping %.3f" % (self.S[i, j], self.S[k, m]),
            "the same five words in another order give the same mean: cosine %.6f. Mean "
            "pooling is a bag of words" % self.S[ORDER],
            "SIF weights each word by a / (a + p(w)): \"the\", %d%% of the corpus, gets %.4f; "
            "\"%s\" gets %.3f" % (round(100 * self.p[self.the_i]), self.wt[self.the_i],
                                  self.words[self.content_i], self.wt[self.content_i]),
            "with SIF the eating pair keeps %.3f, the dog chase and the doctor fall to %s; "
            "the reordered pair is still %.3f" % (self.Ss[i, j], num(self.Ss[k, m], 3),
                                                  self.Ss[ORDER]),
        ][ph]
        draws = (self._mean, self._cosine, self._order, self._weights, self._sif)
        for fn, op in zip(draws, ops):
            if op > 0.01:
                fn(sc, t, op)
        self.step_indicator(sc, ph, self.PHASES)

    # -- phase 1: the mean of the word vectors -----------------------------------------------
    def _row_at(self, r):
        return 0.31 + ROW_DY * r

    def _mean(self, sc, t, op):
        """The pooled sentence's word strips, dropped into their running mean."""
        rows = self.progress("rows", t)
        pool = self.progress("pool", t)
        col = self.progress("column", t)
        n = len(self.words)
        w = STRIP_CELL * w2v.DIM
        x_mid = 0.53
        mean_y = self._row_at(n) + 0.075
        self.label(sc, self.P(x_mid, 0.265), "\"%s\"" % SENTENCES[POOL], 0.029, opacity=op,
                   ink=layout.STRONG_INK)
        for r, word in enumerate(self.words):
            show = smoothstep(r / n * 0.8, r / n * 0.8 + 0.25, rows)
            c = self.P(x_mid, self._row_at(r))
            draw_image(self, sc, c, _strip_tex(self.wvec[r], self.vmax), w, 16, op * show,
                       grid=0.35)
            self.label(sc, self.shift(c, -w / 2 - 10, 0), word, 0.027, anchor="rm",
                       opacity=op * show)
            # a faint copy falls into the mean, one word after another
            u = remap(pool, r / n * 0.85, r / n * 0.85 + 0.2)
            if 0.0 < u < 1.0:
                y = self._row_at(r) + (mean_y - self._row_at(r)) * ease_in_out(u)
                fade = min(1.0, (1.0 - u) / 0.3)
                draw_image(self, sc, self.P(x_mid, y), _strip_tex(self.wvec[r], self.vmax),
                           w, 16, 0.6 * op * fade, grid=0.0, lift=0.6)
                sc.probe(("drop", r), self.P(x_mid, y), tol=6.0)
        got = sum(remap(pool, r / n * 0.85, r / n * 0.85 + 0.2) >= 1.0 for r in range(n))
        c = self.P(x_mid, mean_y)
        ink_in = smoothstep(0.0, 0.08, pool)
        if got:
            draw_image(self, sc, c, _strip_tex(self.running[got - 1], self.vmax), w, 20,
                       op * ink_in, grid=0.35)
        else:
            self.rect(sc, c, w, 20, (0.93, 0.93, 0.92), op * ink_in, border=NEUTRAL_INK,
                      radius=2.0)
        label = "mean" if got == n else "mean of %d" % got
        self.label(sc, self.shift(c, -w / 2 - 10, 0), label, 0.028, anchor="rm",
                   opacity=op * ink_in, ink=layout.STRONG_INK)
        self.label(sc, self.shift(c, w / 2 + 10, 0), "1 × %d" % w2v.DIM, 0.026, anchor="lm",
                   opacity=op * ink_in, ink=layout.NOTE_INK)
        # column 1 written out
        show = smoothstep(0.0, 0.3, col)
        if show > 0.01:
            xl = -w / 2 + STRIP_CELL * COLUMN
            top = self.P(x_mid, self._row_at(0))
            bot = self.P(x_mid, mean_y)
            pts = [self.shift(top, xl, -11), self.shift(top, xl + STRIP_CELL, -11),
                   self.shift(bot, xl + STRIP_CELL, 13), self.shift(bot, xl, 13),
                   self.shift(top, xl, -11)]
            self.polyline(sc, pts, 1.6, layout.STRONG_INK, op * show)
            terms = " + ".join(num(v) for v in self.wvec[:, COLUMN]).replace("+ " + MINUS,
                                                                             MINUS + " ")
            self.label(sc, self.P(x_mid, mean_y + 0.075), "column %d: (%s) / %d = %s" % (
                COLUMN + 1, terms, n, num(self.running[-1, COLUMN])), 0.027, opacity=op * show,
                ink=layout.STRONG_INK)
        self.label(sc, self.P(0.885, 0.40), "orange +", 0.026, opacity=op * smoothstep(
            0.5, 1.0, rows), ink=layout.NOTE_INK)
        self.label(sc, self.P(0.885, 0.445), "blue %s" % MINUS, 0.026, opacity=op * smoothstep(
            0.5, 1.0, rows), ink=layout.NOTE_INK)

    # -- the matrix, shared by phases 2 and 5 ------------------------------------------------
    def _sentence_list(self, sc, op, mark=()):
        """The six sentences, numbered, at the left; ``mark`` rows in strong ink."""
        self.label(sc, self.P(LIST_X, LIST_Y - 0.065), "the six sentences", 0.027,
                   anchor="lm", opacity=op, ink=layout.NOTE_INK)
        for r, s in enumerate(SENTENCES):
            ink = layout.STRONG_INK if r in mark else layout.CAPTION_INK
            self.label(sc, self.P(LIST_X, LIST_Y + LIST_DY * r), "%d   %s" % (r + 1, s), 0.028,
                       anchor="lm", opacity=op, ink=ink)

    def _cell(self, r, c):
        n = len(SENTENCES)
        return self.shift(self.P(*MAT_AT), (c - (n - 1) / 2) * CELL_W, (r - (n - 1) / 2) * CELL_H)

    def _matrix(self, sc, S, op, reveal=1.0, numbers=True, mark=()):
        """The 6 × 6 cosine matrix, cell colours on the diverging scale and the numbers;
        ``reveal`` shows the rows one by one; ``mark`` cells are framed."""
        n = len(SENTENCES)
        shown = int(np.ceil(reveal * n - 1e-9))
        tex = div_colors(S.ravel(), 1.15).reshape(n, n, 3)
        c = self.P(*MAT_AT)
        W, H = CELL_W * n, CELL_H * n
        if shown < n:
            tex = tex.copy()
            tex[shown:] = (0.875, 0.875, 0.87)
        draw_image(self, sc, c, tex, W, H, op, grid=0.35)
        for r in range(n):
            self.label(sc, self.shift(self._cell(r, 0), -CELL_W / 2 - 10, 0), str(r + 1), 0.027,
                       anchor="rm", opacity=op, ink=layout.NOTE_INK)
            self.label(sc, self.shift(self._cell(0, r), 0, -CELL_H / 2 - 6), str(r + 1), 0.027,
                       anchor="mb", opacity=op, ink=layout.NOTE_INK)
        for r in range(min(shown, n) if numbers else 0):
            for q in range(n):
                self.label(sc, self._cell(r, q), num(S[r, q]), 0.028, opacity=op,
                           ink=layout.STRONG_INK)
        for (r, q), m_op in mark:
            if m_op <= 0.01:
                continue
            for a, b in ((r, q), (q, r)):
                self.rect(sc, self._cell(a, b), CELL_W - 3, CELL_H - 3, (1.0, 1.0, 1.0),
                          op * m_op, border=layout.STRONG_INK, border_w=2.2, radius=3.0,
                          lift=0.6, fill_op=0.0)

    def _cosine(self, sc, t, op):
        """Phase 2: the plain means' cosine matrix, filling row by row."""
        u = self.progress("cosine", t)
        reveal = ease_out(remap(u, 0.05, 0.5))
        hi = smoothstep(0.55, 0.65, u)
        lo = smoothstep(0.72, 0.82, u)
        mark = []
        if hi > 0.01:
            mark.append((PAIR_HI, hi))
        if lo > 0.01:
            mark.append((PAIR_LO, lo))
        self._sentence_list(sc, op, mark=PAIR_HI if u < 0.72 else PAIR_HI + PAIR_LO)
        self._matrix(sc, self.S, op, reveal, mark=mark)
        self.label(sc, self.P(LIST_X, 0.86), "plain means: every pair at least %.2f alike" %
                   self.plain_min, 0.027, anchor="lm", opacity=op * smoothstep(0.85, 0.95, u),
                   ink=layout.NOTE_INK)
        self.label(sc, self.shift(self.P(*MAT_AT), 0, CELL_H * 3 + 10),
                   "cosine of two means a, b: a · b / (|a| |b|)", 0.027, anchor="mt", opacity=op)

    # -- phase 3: order blindness -----------------------------------------------------------
    def _order(self, sc, t, op):
        """The two dog/cat sentences: their word strips (one pair swapping places) and the
        two means, equal."""
        u = self.progress("order", t)
        cw = 15.0
        w = cw * w2v.DIM
        xs = (0.28, 0.76)
        y0, dy = 0.37, 0.065
        swap = ease_in_out(remap(u, 0.12, 0.35))
        means = smoothstep(0.42, 0.52, u)
        same = smoothstep(0.62, 0.72, u)
        a, b = ORDER
        for side, (x, words, vecs) in enumerate(zip(xs, self.order_words, self.order_vecs)):
            title = SENTENCES[a] if (side == 0 or swap < 0.5) else SENTENCES[b]
            o_title = 1.0 if side == 0 else abs(1.0 - 2.0 * swap) if 0 < swap < 1 else 1.0
            self.label(sc, self.P(x, 0.275), "\"%s\"" % title, 0.029, opacity=op * o_title,
                       ink=layout.STRONG_INK)
            # side 1 starts in side 0's order; the second and last rows trade places
            for r in range(len(words)):
                if side == 0:
                    yr, vec, word = r, vecs[r], words[r]
                else:
                    src = self.order_words[0].index(words[r]) if words[r] != "the" else r
                    yr = src + (r - src) * swap
                    vec, word = vecs[r], words[r]
                c = self.P(x, y0 + dy * yr)
                draw_image(self, sc, c, _strip_tex(vec, self.vmax), w, 15, op, grid=0.35,
                           lift=0.3 + 0.2 * (side == 1 and words[r] in ("dog", "cat")))
                moving = side == 1 and words[r] in ("dog", "cat")
                # the moving rows' names fade while they pass the others'
                o_word = 1.0 - smoothstep(0.0, 0.08, swap) * (1.0 - smoothstep(0.92, 1.0, swap))
                self.label(sc, self.shift(c, -w / 2 - 8, 0), word, 0.027, anchor="rm",
                           opacity=op * (o_word if moving else 1.0))
                if side == 1 and 0 < swap < 1 and words[r] in ("dog", "cat"):
                    sc.probe(("swap", r), c, tol=6.0)
            cm = self.P(x, y0 + dy * 5 + 0.05)
            draw_image(self, sc, cm, _strip_tex(self.X[ORDER[side]], self.vmax), w, 20,
                       op * means, grid=0.35)
            self.label(sc, self.shift(cm, -w / 2 - 8, 0), "mean", 0.028, anchor="rm",
                       opacity=op * means, ink=layout.STRONG_INK)
        self.label(sc, self.P(0.52, y0 + dy * 5 + 0.05), "=", 0.031, opacity=op * same,
                   ink=layout.STRONG_INK)
        self.label(sc, self.P(0.5, 0.83), "the largest difference between the two means: %s, "
                   "floating-point rounding" % _sci(self.order_diff), 0.027, opacity=op * same,
                   ink=layout.STRONG_INK)
        self.label(sc, self.P(0.5, 0.88), "a mean adds the same numbers in any order: "
                   "{the, the, dog, chases, cat} either way", 0.027,
                   opacity=op * smoothstep(0.75, 0.85, u))
        self.label(sc, self.P(0.5, 0.93), "who chases whom is gone before any comparison", 0.026,
                   opacity=op * smoothstep(0.85, 0.95, u), ink=layout.NOTE_INK)

    # -- phase 4: SIF's weights -------------------------------------------------------------
    def _weights(self, sc, t, op):
        """Each word's corpus share p(w) and SIF weight as bars; the weighted mean, its part
        along the common direction, and what is left."""
        u = self.progress("weights", t)
        y0, dy = 0.385, 0.062
        x_word, x_p, x_w = 0.14, 0.16, 0.43
        bar_px = 130.0
        pmax, wmax = float(self.p.max()), float(self.wt.max())
        bars = ease_out(remap(u, 0.05, 0.4))
        self.label(sc, self.P(x_p, 0.275), "p(w): share of the corpus", 0.027, anchor="lm",
                   opacity=op, ink=layout.NOTE_INK)
        self.label(sc, self.P(x_w, 0.275), "weight a / (a + p(w))", 0.027, anchor="lm",
                   opacity=op, ink=layout.NOTE_INK)
        self.label(sc, self.P(x_w, 0.315), "a = %g" % w2v.SIF_A, 0.026, anchor="lm",
                   opacity=op, ink=layout.NOTE_INK)
        for r, word in enumerate(self.words):
            y = y0 + dy * r
            strong = r == self.the_i or word == "the"
            ink = layout.STRONG_INK if strong else layout.CAPTION_INK
            self.label(sc, self.P(x_word, y), word, 0.028, anchor="rm", opacity=op, ink=ink)
            for x, v, vmax, fmt, col in ((x_p, self.p[r], pmax, "%.3f", (0.55, 0.57, 0.60)),
                                         (x_w, self.wt[r], wmax, "%.4f", (0.27, 0.62, 0.58))):
                L = max(1.5, bar_px * v / vmax * bars)
                c = self.shift(self.P(x, y), L / 2, 0)
                self.rect(sc, c, L, 14, col, op, radius=2.0)
                self.label(sc, self.shift(self.P(x, y), L + 8, 0), fmt % v, 0.026, anchor="lm",
                           opacity=op * smoothstep(0.3, 0.45, u), ink=ink)
        # right: the weighted mean, the common direction's part, and what is left
        x = 0.83
        cw = 13.0
        w = cw * w2v.DIM
        steps = [("weighted mean", self.weighted, 0.45),
                 ("minus its part along u", self.removed, 0.62),
                 ("= the SIF vector", self.sif, 0.78)]
        vmax = float(np.abs(self.weighted).max())
        for k, (name, v, at) in enumerate(steps):
            show = smoothstep(at, at + 0.1, u)
            yy = 0.33 + 0.16 * k
            c = self.P(x, yy + 0.045)
            self.label(sc, self.P(x, yy), name, 0.027, opacity=op * show,
                       ink=layout.STRONG_INK if k == 2 else layout.CAPTION_INK)
            draw_image(self, sc, c, _strip_tex(v, vmax), w, 16, op * show, grid=0.35)
        self.label(sc, self.P(x, 0.82), "u: the first principal direction", 0.026,
                   opacity=op * smoothstep(0.62, 0.72, u), ink=layout.NOTE_INK)
        self.label(sc, self.P(x, 0.86), "of 2,000 corpus sentences' means;", 0.026,
                   opacity=op * smoothstep(0.62, 0.72, u), ink=layout.NOTE_INK)
        self.label(sc, self.P(x, 0.90), "here %d%% of the vector's length" % round(
            100 * self.removed_frac), 0.026, opacity=op * smoothstep(0.62, 0.72, u),
            ink=layout.NOTE_INK)
        self.label(sc, self.P(0.30, 0.86), "frequent words count for less: \"the\" %.0f× less "
                   "than \"%s\"" % (self.wt[self.content_i] / self.wt[self.the_i],
                                    self.words[self.content_i]), 0.027, opacity=op *
                   smoothstep(0.45, 0.55, u), ink=layout.STRONG_INK)

    # -- phase 5: the SIF matrix -------------------------------------------------------------
    def _sif(self, sc, t, op):
        """The cosine matrix turning from the plain means' to SIF's."""
        u = self.progress("sif", t)
        k = smoothstep(0.08, 0.3, u)
        tex_S = self.S + (self.Ss - self.S) * k
        mark = [(PAIR_HI, smoothstep(0.35, 0.45, u)), (PAIR_LO, smoothstep(0.35, 0.45, u)),
                (ORDER, smoothstep(0.55, 0.65, u))]
        self._sentence_list(sc, op, mark=PAIR_HI + PAIR_LO)
        # the colours move smoothly; the numbers cross-fade between the two real matrices
        self._matrix(sc, tex_S, op, 1.0, numbers=False, mark=mark)
        n = len(SENTENCES)
        for r in range(n):
            for q in range(n):
                # one after the other, never both at once
                for vals, o in ((self.S, 1.0 - smoothstep(0.0, 0.45, k)),
                                (self.Ss, smoothstep(0.55, 1.0, k))):
                    if o > 0.01:
                        self.label(sc, self._cell(r, q), num(vals[r, q]), 0.028,
                                   opacity=op * o, ink=layout.STRONG_INK)
        self.label(sc, self.shift(self.P(*MAT_AT), 0, CELL_H * 3 + 10),
                   "plain means" if k < 0.5 else "SIF vectors", 0.027, anchor="mt",
                   opacity=op * abs(2 * k - 1))
        i, j = PAIR_HI
        a, b = PAIR_LO
        moved = [
            "%d and %d: %.3f → %.3f" % (i + 1, j + 1, self.S[i, j], self.Ss[i, j]),
            "%d and %d: %.3f → %s" % (a + 1, b + 1, self.S[a, b], num(self.Ss[a, b], 3)),
            "%d and %d, the same words: %.3f → %.3f" % (ORDER[0] + 1, ORDER[1] + 1,
                                                         self.S[ORDER], self.Ss[ORDER]),
            "%d of the other %d pairs now below 0" % (self.sif_neg, self.n_off),
        ]
        for r, line in enumerate(moved):
            self.label(sc, self.P(LIST_X, 0.74 + 0.05 * r), line, 0.027, anchor="lm",
                       opacity=op * smoothstep(0.38 + 0.1 * r, 0.48 + 0.1 * r, u),
                       ink=layout.STRONG_INK if r < 3 else layout.NOTE_INK)
        self.label(sc, self.P(0.5, 0.95), "SIF still averages, so word order is still lost; "
                   "trained sentence encoders (transformers) read it", 0.027,
                   opacity=op * smoothstep(0.8, 0.9, u), ink=layout.STRONG_INK)
        sc.probe(("sif",), (k * 60, 0, 0), tol=6.0)

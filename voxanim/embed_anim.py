"""What GPT-2's embeddings know: a token's vector, its nearest neighbours,
an analogy, and the position embeddings.

Everything is read from the real tables of GPT-2 small (:mod:`voxanim.gpt2`):
``wte`` (50,257 tokens x 768) and ``wpe`` (1,024 positions x 768), learned
with the rest of the model.

1. **table** -- the token table, 50,257 rows x 768 numbers (drawn from
   every 98th row),
2. **vector** -- one row, " king" (id 5,822), pulled out as a strip of 768
   numbers,
3. **neighbours** -- cosine similarity of that row with every other: the
   nearest are " kings" (0.73), " King" (0.70), " queen" (0.66), ...,
   listed with their strips and numbers,
4. **analogy** -- king - man + woman: the strips are added and subtracted
   number by number and the result's nearest rows looked up.  GPT-2's
   embeddings give " king" itself first (0.78), then " queen" (0.71): the
   famous analogy holds, roughly,
5. **positions** -- the position table as a 1,024 x 768 image; with each
   column scaled to its own spread, every dimension is seen to drift
   slowly along the position,
6. **distance** -- the cosine similarity of positions i and i + d: a 1,024 x
   1,024 matrix banded along its diagonals, and the mean against d:
   0.997 one apart, 0.47 at 100, zero at 164, lowest (-0.69) at 780.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from . import gpt2 as G
from . import layout
from .diagram import DATA, Diagram, draw_image, draw_strip
from .easing import ease_in_out, ease_out, remap, smoothstep
from .gpt2_panel import diverging, mip
from .text import MINUS, num

WORD = " king"
ANALOGY = (" king", " man", " woman")       # a - b + c
N_NEAR = 8                                  # neighbours listed
N_ANALOGY = 5                               # nearest rows to the analogy's result
ROW_STEP = 98                               # the table is drawn from every 98th row
N_SHOWN = 5                                 # numbers of the vector written out

# ------------------------------------------------------------------ layout (fractions, px)

TABLE_AT = (0.10, 0.26)                    # top-left of the token table
TABLE_W, TABLE_H = 105.0, 320.0             # px
STRIP_AT = (0.575, 0.30)                    # centre of " king"'s strip
STRIP_W, STRIP_H = 340.0, 16.0
NAME_DX = 14.0                              # token names left of their strips, cosines right
LIST_DY = 72.0                              # the neighbour list starts this far under the strip
LIST_ROW = 24.0                             # px per neighbour
VALUES_DY = 22.0                            # the written-out numbers, under the strip
READOUT_DY = 30.0                           # the sweep's read-out, under the chart's axis labels
POS_AT = (0.215, 0.60)                      # centre of the position table image
POS_W, POS_H = 210.0, 280.0                 # px: 1,024 rows x 768 columns, 4:3
SIM_AT = (0.535, 0.60)                      # centre of the 1,024 x 1,024 similarity matrix
SIM_W = 220.0
CHART = (0.72, 0.42, 0.955, 0.80)           # the curve: x0, y0, x1, y1 (layout fractions)
CHART_PAD = 10.0                            # px inside the chart's frame
PANEL = (0.96, 0.96, 0.95)
PANEL_EDGE = (0.70, 0.71, 0.72)
CURVE = (0.36, 0.38, 0.40)
MARK = (0.95, 0.95, 0.92)                   # the diagonal picked out on the matrix
D_SWEEP_POWER = 2.2                         # the distance sweep spends most of its time small

# ------------------------------------------------------------------ timeline (seconds)

TIMELINE = dict(intro=(0.2, 1.0), table=(0.8, 3.0), pick=(3.4, 4.4), pull=(4.4, 6.0),
                values=(6.0, 7.6), near=(8.2, 18.0), fade_near=(19.6, 20.4),
                terms=(20.6, 24.0), result=(24.0, 26.0), anear=(26.4, 32.0),
                fade_all=(34.0, 35.0), pos=(35.2, 38.8), sim=(39.6, 42.0),
                curve=(42.0, 44.6), sweep=(45.0, 55.0))
T_DONE = 55.6


def cosine_rows(E, v):
    """Cosine similarity of vector ``v`` with every row of ``E``."""
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    return En @ (v / np.linalg.norm(v))


def nearest(E, v, k, tok):
    """The ``k`` rows of ``E`` nearest to ``v`` by cosine: ``[(token, cosine)]``."""
    s = cosine_rows(E, v)
    return [(tok.decode([int(i)]), float(s[i])) for i in np.argsort(-s)[:k]]


def position_similarity(P):
    """``(S, curve)``: the cosine similarity of every pair of positions
    (1024 x 1024) and its mean over the pairs at each distance ``d`` (1024,)."""
    Pn = P / np.linalg.norm(P, axis=1, keepdims=True)
    S = Pn @ Pn.T
    return S, np.array([np.mean(np.diag(S, d)) for d in range(len(P))])


@dataclass
class Facts:
    """Everything the animation reads from the two tables (see :func:`facts`)."""

    word_id: int
    word_vec: np.ndarray
    near: list                  # [(token, cosine)] nearest to WORD, WORD itself first
    terms: list                 # [(token, id, vector)] of the analogy's three words
    result: np.ndarray          # a - b + c
    anear: list                 # [(token, cosine)] nearest to the result
    strips: dict                # token -> (768, 3) colours, one scale for all
    table: np.ndarray           # the token table sampled, as colours
    pos_img: np.ndarray         # the position table, each column scaled to its spread
    sim: np.ndarray             # (1024, 1024) cosine similarities as colours
    curve: np.ndarray           # mean cosine against distance, (1024,)


@cache
def facts():
    """Read the tables once: the neighbours, the analogy, the position structure."""
    tok = G.Tokenizer()
    W = G.weights()
    E, P = W["wte.weight"], W["wpe.weight"]
    wid = tok.encode(WORD)[0]
    near = nearest(E, E[wid], N_NEAR + 1, tok)
    terms = [(w, tok.encode(w)[0], E[tok.encode(w)[0]]) for w in ANALOGY]
    result = terms[0][2] - terms[1][2] + terms[2][2]
    anear = nearest(E, result, N_ANALOGY, tok)
    names = [w for w, _c in near] + [w for w, _i, _v in terms] + [w for w, _c in anear]
    vecs = [E[tok.encode(w)[0]] for w in names] + [result]
    cols = diverging(np.array(vecs))
    strips = dict(zip(names, cols))
    strips["result"] = cols[-1]
    S, curve = position_similarity(P)
    pos_img = diverging((P - P.mean(0)) / P.std(0))
    return Facts(wid, E[wid], near, terms, result, anear, strips, diverging(E[::ROW_STEP]),
                 pos_img, diverging(S), curve)


class EmbeddingAnimation(Diagram):
    """The ``gpt2-embeddings`` animation (see the module docstring)."""

    PHASES = ("table", "vector", "neighbours", "analogy", "positions", "distance")

    def __init__(self):
        super().__init__("What embeddings know", "")
        self.f = facts()
        self.Tm = dict(TIMELINE)
        self.hold, self.reset = 3.0, 1.5
        # textures sized once to the pixels they are drawn at
        self.table_tex = mip(self.f.table, 1, 3)              # 257 x 96
        self.pos_tex = mip(self.f.pos_img, 2, 2)              # 256 x 192
        self.sim_tex = mip(self.f.sim, 2, 2)                  # 256 x 256
        c = self.f.curve
        self.d_zero = int(np.argmax(c < 0))                   # first distance with cosine < 0
        self.d_min = int(np.argmin(c))
        self.finish(T_DONE)

    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        for k, name in ((5, "sim"), (4, "fade_all"), (3, "fade_near"), (2, "near"), (1, "pick")):
            if t >= self.Tm[name][0]:
                return k
        return 0

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        first = (1 - smoothstep(*self.Tm["fade_all"], t)) * back      # the token-table half
        self._table(sc, t, first)
        self._vector(sc, t, first)
        self._neighbours(sc, t, first)
        self._analogy(sc, t, first)
        self._positions(sc, t, back)
        self._similarity(sc, t, back)
        self._chart(sc, t, back)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, reset)

    def _table(self, sc, t, op):
        """The token table (every 98th row) with the chosen row picked out."""
        on = smoothstep(0.0, 1.0, self.progress("table", t)) * op
        if on <= 0.01:
            return
        x0, y0 = TABLE_AT
        c = self.shift(self.P(x0, y0), TABLE_W / 2, TABLE_H / 2)
        draw_image(self, sc, c, self.table_tex, TABLE_W, TABLE_H, on, grid=0.0)
        self.label(sc, self.P(x0, y0), "%s × %d" % (format(G.N_VOCAB, ","), G.D_MODEL),
                   layout.MIN_TEXT, anchor="lb", dy=-6, opacity=on)
        pick = smoothstep(0.0, 1.0, self.progress("pick", t))
        if pick > 0:
            y = self.f.word_id / G.N_VOCAB * TABLE_H
            row = self.shift(self.P(x0, y0), TABLE_W / 2, y)
            self.rect(sc, row, TABLE_W + 6, 5, DATA, pick * op, border=DATA, border_w=1.0,
                      radius=1, lift=0.6, fill_op=0.35)
            self.label(sc, row, "row %s" % format(self.f.word_id, ","), layout.MIN_TEXT,
                       anchor="rm", dx=-(TABLE_W / 2 + 8), opacity=pick * op,
                       ink=layout.CAPTION_INK)

    def _vector(self, sc, t, op):
        """The chosen row flies out of the table and becomes a strip; its first
        numbers are written out."""
        pull = ease_in_out(self.progress("pull", t))
        if pull <= 0 or op <= 0.01:
            return
        x0, y0 = TABLE_AT
        src = self.shift(self.P(x0, y0), TABLE_W / 2, self.f.word_id / G.N_VOCAB * TABLE_H)
        dst = self.P(*STRIP_AT)
        c = src + (dst - src) * pull
        w = TABLE_W + (STRIP_W - TABLE_W) * pull
        h = 5 + (STRIP_H - 5) * pull
        draw_strip(self, sc, c, self.f.strips[WORD], w, h, op, lift=0.5)
        if pull >= 1:
            self.label(sc, dst, '"%s"' % WORD, layout.MIN_TEXT, anchor="rm",
                       dx=-(STRIP_W / 2 + NAME_DX), opacity=op, kind="mono",
                       ink=layout.STRONG_INK)
            v = ease_out(self.progress("values", t))
            if v > 0:
                shown = "  ".join(num(x, 3) for x in self.f.word_vec[:N_SHOWN])
                text = "%d numbers:  %s  …" % (G.D_MODEL, shown)
                self.label(sc, dst, text, layout.MIN_TEXT, anchor="mt", dy=VALUES_DY - 8,
                           opacity=op * v * (1 - smoothstep(*self.Tm["fade_near"], t)),
                           kind="mono", ink=layout.CAPTION_INK)

    def _neighbours(self, sc, t, op):
        """The nearest rows by cosine, one per second, with their strips and numbers."""
        u = self.progress("near", t)
        op *= 1 - smoothstep(*self.Tm["fade_near"], t)
        if u <= 0 or op <= 0.01:
            return
        top = self.P(*STRIP_AT)
        self.label(sc, top, "cosine similarity  a · b ÷ (|a| |b|)  with every row:",
                   layout.MIN_TEXT, anchor="lm", dx=-STRIP_W / 2, dy=LIST_DY - LIST_ROW,
                   opacity=op * smoothstep(0.0, 0.1, u), ink=layout.NOTE_INK)
        for n, (tokn, cosv) in enumerate(self.f.near[1:]):
            k = smoothstep(0.0, 0.5, remap(u, n / N_NEAR * 0.85, n / N_NEAR * 0.85 + 0.12))
            if k <= 0:
                break
            self._strip_row(sc, self.shift(top, 0.0, LIST_DY + n * LIST_ROW), tokn, cosv,
                            op * k)

    def _strip_row(self, sc, c, tokn, value, op, left=None, strip=None):
        """One row of a list: a token's name, its strip and a number at the right."""
        self.label(sc, c, '"%s"' % tokn if left is None else left, layout.MIN_TEXT, anchor="rm",
                   dx=-(STRIP_W / 2 + NAME_DX), opacity=op, kind="mono")
        draw_strip(self, sc, c, self.f.strips[tokn] if strip is None else strip, STRIP_W,
                   STRIP_H, op)
        if value is not None:
            self.label(sc, c, num(value), layout.MIN_TEXT, anchor="lm", dx=STRIP_W / 2 + NAME_DX,
                       opacity=op, kind="mono", ink=layout.STRONG_INK)

    def _analogy(self, sc, t, op):
        """king − man + woman: the three strips, the result, its nearest rows."""
        tm = self.progress("terms", t)
        if tm <= 0 or op <= 0.01:
            return
        top = self.P(*STRIP_AT)
        ops = (MINUS, "+", "=")
        rows = [(w, v) for w, _i, v in self.f.terms[1:]] + [("result", self.f.result)]
        res = ease_out(self.progress("result", t))
        for n, (w, _v) in enumerate(rows):
            if n < 2:
                k = smoothstep(0.0, 0.6, remap(tm, 0.4 * n, 0.4 * n + 0.6))
            else:
                k = res
            if k <= 0:
                break
            c = self.shift(top, 0.0, (n + 1) * (STRIP_H + 10))
            self._strip_row(sc, c, w if n < 2 else "result", None, op * k,
                            left='%s "%s"' % (ops[n], w) if n < 2 else "%s result" % ops[n])
        an = self.progress("anear", t)
        if an <= 0:
            return
        c0 = self.shift(top, 0.0, 3 * (STRIP_H + 10) + LIST_DY - 6)
        self.label(sc, c0, "nearest rows to the result:", layout.MIN_TEXT, anchor="lm",
                   dx=-STRIP_W / 2, dy=-LIST_ROW, opacity=op * smoothstep(0.0, 0.15, an),
                   ink=layout.NOTE_INK)
        for n, (tokn, cosv) in enumerate(self.f.anear):
            k = smoothstep(0.0, 0.5, remap(an, n / N_ANALOGY * 0.8, n / N_ANALOGY * 0.8 + 0.15))
            if k <= 0:
                break
            self._strip_row(sc, self.shift(c0, 0.0, n * LIST_ROW), tokn, cosv, op * k)

    def _positions(self, sc, t, back):
        """The position table as an image, each column scaled to its own spread."""
        on = smoothstep(0.0, 0.5, self.progress("pos", t)) * back
        if on <= 0.01:
            return
        c = self.P(*POS_AT)
        draw_image(self, sc, c, self.pos_tex, POS_W, POS_H, on, grid=0.0)
        self.label(sc, c, "positions 0 … %d  (rows) × %d" % (G.N_CTX - 1, G.D_MODEL),
                   layout.MIN_TEXT, anchor="mb", dy=-(POS_H / 2 + 6), opacity=on)
        self.label(sc, c, "0", layout.MIN_TEXT, anchor="rm", dx=-(POS_W / 2 + 6),
                   dy=-POS_H / 2 + 8, opacity=on, ink=layout.NOTE_INK, kind="mono")
        self.label(sc, c, "%d" % (G.N_CTX - 1), layout.MIN_TEXT, anchor="rm",
                   dx=-(POS_W / 2 + 6), dy=POS_H / 2 - 8, opacity=on, ink=layout.NOTE_INK,
                   kind="mono")

    def _d_at(self, t):
        """The distance the sweep is at: 1 … 1023, slow at first."""
        u = self.progress("sweep", t)
        return max(1, int(round(1 + (G.N_CTX - 2) * u ** D_SWEEP_POWER)))

    def _similarity(self, sc, t, back):
        """The 1,024 x 1,024 cosine matrix, with the sweep's diagonal picked out."""
        on = smoothstep(0.0, 0.5, self.progress("sim", t)) * back
        if on <= 0.01:
            return
        c = self.P(*SIM_AT)
        draw_image(self, sc, c, self.sim_tex, SIM_W, SIM_W, on, grid=0.0)
        self.label(sc, c, "cosine of position i with position j", layout.MIN_TEXT, anchor="mb",
                   dy=-(SIM_W / 2 + 6), opacity=on)
        self.label(sc, c, "i", layout.MIN_TEXT, anchor="rm", dx=-(SIM_W / 2 + 6), opacity=on,
                   ink=layout.NOTE_INK)
        self.label(sc, c, "j", layout.MIN_TEXT, anchor="mt", dy=SIM_W / 2 + 4, opacity=on,
                   ink=layout.NOTE_INK)
        sw = self.progress("sweep", t)
        if 0 < sw < 1:
            d = self._d_at(t)
            k = SIM_W / G.N_CTX
            a = self.shift(c, -SIM_W / 2 + d * k, -SIM_W / 2) + self.view * 0.8
            b = self.shift(c, SIM_W / 2, SIM_W / 2 - d * k) + self.view * 0.8
            sc.line(a, b, width=1.6, color=MARK, opacity=0.9 * back, bias=0.5)
            sc.probe(("diag",), a, tol=3.0)

    def _chart(self, sc, t, back):
        """Mean cosine of positions i and i + d against d, with the sweep's point."""
        cu = self.progress("curve", t)
        on = smoothstep(0.0, 0.3, cu) * back
        if on <= 0.01:
            return
        x0, y0, x1, y1 = CHART
        a, b = self.P(x0, y0), self.P(x1, y1)
        w, h = self.px(b) - self.px(a)
        self.rect(sc, (a + b) / 2, w, h, PANEL, on, border=PANEL_EDGE, border_w=1.0, radius=4,
                  lift=0.1)
        curve = self.f.curve

        def pt(d, v):
            fx = (d / (G.N_CTX - 1)) * (w - 2 * CHART_PAD) + CHART_PAD
            fy = (1 - v) / 2 * (h - 2 * CHART_PAD) + CHART_PAD
            return self.shift(a, fx, fy) + self.view * 0.3

        zero = (pt(0, 0.0), pt(G.N_CTX - 1, 0.0))
        sc.line(zero[0], zero[1], width=1.0, color=PANEL_EDGE, opacity=on, bias=0.2)
        shown = int(round(ease_out(cu) * (G.N_CTX - 1)))
        ds = np.unique(np.linspace(0, shown, 160).astype(int))
        if len(ds) > 1:
            self.polyline(sc, [pt(d, curve[d]) for d in ds], width=2.0, color=DATA,
                          opacity=0.95 * on)
        for v in (1.0, 0.0, -1.0):
            self.label(sc, pt(0, v), num(v, 0), layout.MIN_TEXT, anchor="rm", dx=-10,
                       opacity=on, ink=layout.NOTE_INK, kind="mono")
        self.label(sc, pt(0, -1.0), "d = 0", layout.MIN_TEXT, anchor="lt", dy=10, opacity=on,
                   ink=layout.NOTE_INK)
        self.label(sc, pt(G.N_CTX - 1, -1.0), "%d" % (G.N_CTX - 1), layout.MIN_TEXT, anchor="rt",
                   dy=10, opacity=on, ink=layout.NOTE_INK, kind="mono")
        self.label(sc, pt(G.N_CTX - 1, 1.0), "mean cosine of i and i + d", layout.MIN_TEXT,
                   anchor="rb", dy=-8, opacity=on, ink=layout.NOTE_INK)
        sw = self.progress("sweep", t)
        if sw > 0:
            d = self._d_at(t) if sw < 1 else self.d_min
            self.disk(sc, pt(d, curve[d]), 3.6, DATA, back, lift=0.6, outline=0.0)
            self.label(sc, pt(G.N_CTX // 2, -1.0), "d = %d:  %s" % (d, num(curve[d])),
                       layout.MIN_TEXT, anchor="mt", dy=READOUT_DY, opacity=back,
                       ink=layout.STRONG_INK, kind="mono")

    # ------------------------------------------------------------------ words
    def _words(self, t, reset):
        """The subtitle at ``t``."""
        Tm, f = self.Tm, self.f
        n_v, d = format(G.N_VOCAB, ","), G.D_MODEL
        if reset > 0 or t < Tm["pick"][0]:
            return ("the token table: %d learned numbers per token; %s × %d = %.1f M of GPT-2's "
                    "124 M parameters" % (d, n_v, d, G.N_VOCAB * d / 1e6))
        if t < Tm["near"][0]:
            return ('"%s" is row %s: %d numbers, %s …' % (
                WORD, format(f.word_id, ","), d, ", ".join(num(x) for x in f.word_vec[:4])))
        if t < Tm["fade_near"][0]:
            near = f.near[1:4]
            return ("cosine similarity with every other row: the nearest to \"%s\" are %s"
                    % (WORD, ", ".join('"%s" %.2f' % (w, c) for w, c in near)))
        if t < Tm["anear"][0]:
            return ("an analogy: %s %s %s + %s, number by number, gives a new vector; which "
                    "rows is it nearest to?" % (ANALOGY[0].strip(), MINUS, ANALOGY[1].strip(),
                                                ANALOGY[2].strip()))
        if t < Tm["fade_all"][0]:
            (w0, c0), (w1, c1) = f.anear[:2]
            return ('nearest is "%s" itself (%.2f), then "%s" (%.2f): the famous analogy '
                    'holds, roughly' % (w0, c0, w1, c1))
        if t < Tm["sim"][0]:
            return ("position table %s × %d, also learned; each column scaled to its own "
                    "spread: slow waves down the rows" % (format(G.N_CTX, ","), d))
        if t < Tm["sweep"][0]:
            return ("the cosine similarity of every pair of positions: bands along the "
                    "diagonals, so it depends on their distance d")
        sw = self.progress("sweep", t)
        if sw < 1:
            dd = self._d_at(t)
            return ("positions i and i + d: at d = %d the mean cosine is %s. %s one apart, "
                    "%s at 100" % (dd, num(f.curve[dd]), num(f.curve[1], 3), num(f.curve[100])))
        return ("nearby positions have nearby vectors: %s one apart, %s at 100, zero at %d, "
                "lowest (%s) at %d" % (num(f.curve[1], 3), num(f.curve[100]), self.d_zero,
                                        num(f.curve[self.d_min]), self.d_min))

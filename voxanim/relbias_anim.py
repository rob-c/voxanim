"""T5's relative position bias, in a real trained model.

T5 (Raffel et al., 2020) adds no position vector to the tokens.  Instead each
attention head adds a learned number to the score of query ``i`` against key
``j``, looked up by the *offset* ``i − j``.  Offsets are sorted into 16
buckets (:func:`voxanim.posenc.t5_bucket`, ``max_distance`` 32): offsets 0 .. 7
get a bucket each, larger ones logarithmically wider buckets, and every
offset from 27 on shares the last.  The model is the study's T5 run
(:func:`voxanim.posenc.train`): one layer, 4 heads of width 8, trained on an
order-2 Markov language over 8 symbols at length 32.  Four phases:

1. **offsets** -- a sampled sequence; a query walks along it and every earlier
   key is labelled with its offset ``i − j``; beside it the 16 × 16 matrix of
   offsets, one diagonal per offset,
2. **buckets** -- the offsets 0 .. 31 coloured by bucket, with the bucket
   numbers and boundaries computed from ``t5_bucket``, then the whole line
   0 .. 127: everything past 26 in the last bucket,
3. **biases** -- the trained bias table ``P["bias"]`` (4 heads × 16 buckets),
   the values of buckets 0, 1 and 15 per head, and one head's 32 × 32 bias
   matrix: a band at offset 1, the previous token (the head is the one whose
   bias at offset 1 stands highest above its other buckets),
4. **longer** -- that head's attention on one test sequence at length 32
   and at length 128, and the test loss on length-128 sequences per range of
   positions against the best possible (:func:`voxanim.posenc.entropies`); and
   the share of that head's attention on the previous token before position
   32 and past 64 (64 sampled sequences), which says why the loss creeps up.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout, posenc
from .diagram import EDGE, Diagram, draw_image
from .easing import ease_in_out, ease_out, remap, smoothstep
from .netanim import OFF, div_colors, lit_colors
from .text import num

N_SEQ = 16                           # tokens in the offsets phase
SEQ_SEED = 7                         # the shown sequence
STATS_SEED = 11                      # sequences for the attention shares
N_STATS = 64
SYMBOLS = "abcdefgh"
QUERY_FROM = 3                       # the query walks from here to the end
SEQ_X = (0.08, 0.60)                 # token row, layout x
SEQ_Y = 0.40
TOKEN_PX = 22.0
OFFSET_AT, OFFSET_PX = (0.81, 0.56), 210.0
STRIP_X = (0.07, 0.93)               # bucket strips, layout x
STRIP_SHORT_Y, STRIP_LONG_Y = 0.42, 0.66
STRIP_H = 24.0
TABLE_AT, TABLE_W, TABLE_H = (0.34, 0.43), 400.0, 92.0
BIAS_AT, BIAS_PX = (0.80, 0.56), 220.0
MAP_PX = 210.0
MAP_AT = ((0.19, 0.55), (0.48, 0.55))
QUERY_INK = C.rgb("#d84e4e")
KEY_INK = C.rgb("#3d9e45")
BOX_INK = (0.12, 0.36, 0.66)
BUCKET_ANCHORS = [C.rgb(h) for h in ("#449696", "#5f9a45", "#a1814b", "#d84e4e", "#8952eb")]


def bucket_ranges(num_buckets=posenc.T5_BUCKETS, max_distance=posenc.T5_MAX, upto=4096):
    """``[(first, last), ...]`` offsets of each bucket (the last bucket's ``last`` is None)."""
    b = posenc.t5_bucket(np.arange(upto), num_buckets, max_distance)
    out = []
    for k in range(num_buckets):
        w = np.flatnonzero(b == k)
        out.append((int(w[0]), None if k == num_buckets - 1 else int(w[-1])))
    return out


def bucket_color(k, n=posenc.T5_BUCKETS, alternate=True):
    """A colour per bucket: a ramp through a few hues, alternate buckets lighter
    (``alternate`` False: the ramp alone, dark enough for text)."""
    u = k / (n - 1) * (len(BUCKET_ANCHORS) - 1)
    i = min(int(u), len(BUCKET_ANCHORS) - 2)
    c = C.mix(BUCKET_ANCHORS[i], BUCKET_ANCHORS[i + 1], u - i)
    return C.lighten(c, 0.35) if alternate and k % 2 else c


def offset_colors(n):
    """``(n, n, 3)``: the offset ``i − j`` on a light-to-dark ramp, OFF above the diagonal."""
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    rel = (i - j) / (n - 1)
    lo, hi = C.rgb("#eef3f2"), C.rgb("#2a6f6b")
    tex = lo[None, None] + (hi - lo)[None, None] * rel[..., None]
    tex[j > i] = OFF
    return tex.astype(np.float32)


def attention_tex(A):
    """``(n, n)`` attention weights as the house's lit ramp; masked cells OFF."""
    n = A.shape[0]
    tex = lit_colors(A.ravel(), 1.0).reshape(n, n, 3)
    tex[np.triu(np.ones((n, n), bool), 1)] = OFF
    return tex


class RelativePositionBias(Diagram):
    """T5's offset buckets and the per-head biases of the study's trained model."""

    PHASES = ("offsets", "buckets", "biases", "longer")

    def __init__(self):
        super().__init__("T5 relative position bias", "", 10.0)
        self.hold = 2.5
        run = posenc.train("t5")
        self.run = run
        self.bias = run.P["bias"]                                 # (heads, buckets)
        self.ranges = bucket_ranges()
        self.far = self.ranges[-1][0]                             # first offset of the last bucket
        nb = posenc.T5_BUCKETS
        # previous-token heads: bias highest at bucket 1; show the one that stands out most
        others = np.array([np.delete(self.bias[h], 1).max() for h in range(posenc.HEADS)])
        self.peak1 = [h for h in range(posenc.HEADS) if int(np.argmax(self.bias[h])) == 1]
        lead = self.bias[:, 1] - others
        self.head = int(np.argmax(lead))
        self.n_short = posenc.TRAIN_LEN
        self.n_long = posenc.TEST_LEN
        bk = posenc.t5_bucket(np.maximum(np.arange(self.n_short)[:, None]
                                         - np.arange(self.n_short)[None, :], 0), nb, posenc.T5_MAX)
        Bm = self.bias[self.head][bk]
        Bm[np.triu(np.ones_like(Bm, bool), 1)] = np.nan
        self.bias_mat = Bm
        rng = np.random.default_rng(SEQ_SEED)
        self.seq = posenc.sample(1, self.n_long, rng)[0]
        A_long = posenc.attention(run, self.seq)[self.head]
        A_short = posenc.attention(run, self.seq[:self.n_short])[self.head]
        self.tex_long = attention_tex(A_long)
        self.tex_short = attention_tex(A_short)
        # share of the head's attention on the previous token, by query range
        xs = posenc.sample(N_STATS, self.n_long, np.random.default_rng(STATS_SEED))
        prev = np.array([np.diagonal(posenc.attention(run, x)[self.head], -1) for x in xs])
        # prev[:, q - 1] is query q's weight on key q - 1
        self.prev_in = float(prev[:, 0:self.n_short - 1].mean())
        self.prev_far = float(prev[:, 2 * self.n_short - 1:].mean())
        tl = run.test_loss                         # tl[p - 1]: the loss of predicting position p
        n = self.n_short
        self.loss_ranges = [(2, n - 1), (n, 2 * n - 1), (2 * n, self.n_long - 1)]
        self.losses = [float(tl[a - 1:b].mean()) for a, b in self.loss_ranges]
        self.best, self.h1, self.h0 = posenc.entropies()
        self.lead_far = float(self.bias[self.head, 1] - self.bias[self.head, -1])

        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("seq_in", 2.0, 0.0)
        add("sweep", 9.0, 1.0)
        add("buckets", 12.0, 1.0)
        add("biases", 12.0, 1.0)
        add("long", 13.0, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["buckets"][0] - 0.5, Tm["biases"][0] - 0.5, Tm["long"][0] - 0.5]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _phase_ops(self, t, back):
        """Opacity of each phase's drawing: one at a time, with short crossfades."""
        Tm = self.Tm
        ends = {"offsets": Tm["buckets"][0] - 0.5, "buckets": Tm["biases"][0] - 0.5,
                "biases": Tm["long"][0] - 0.5}
        starts = {"buckets": Tm["buckets"][0] - 0.4, "biases": Tm["biases"][0] - 0.4,
                  "long": Tm["long"][0] - 0.4}
        op = {}
        op["offsets"] = 1.0 - smoothstep(ends["offsets"] - 0.6, ends["offsets"], t)
        for key in ("buckets", "biases"):
            op[key] = smoothstep(starts[key], starts[key] + 0.4, t) * (
                1.0 - smoothstep(ends[key] - 0.6, ends[key], t))
        op["long"] = smoothstep(starts["long"], starts["long"] + 0.4, t)
        return {k: v * back for k, v in op.items()}

    def _query(self, t, reset):
        """The query position during the sweep (an integer)."""
        if reset > 0:
            return QUERY_FROM
        u = ease_in_out(self.progress("sweep", t))
        return int(round(QUERY_FROM + u * (N_SEQ - 1 - QUERY_FROM)))

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ops = self._phase_ops(t, back)
        q = self._query(t, reset)
        b = self.bias
        lo1, hi1 = float(b[self.peak1, 1].min()) if self.peak1 else 0.0, float(b[:, 1].max())
        if len(self.peak1) == posenc.HEADS:
            who = "all %d heads peak at offset 1 (%s to %s)" % (
                posenc.HEADS, num(lo1), num(hi1))
        elif self.peak1:
            who = "%d of %d heads peak at offset 1" % (len(self.peak1), posenc.HEADS)
        else:
            who = "no head peaks at offset 1"
        l0, l1, l2 = self.losses
        self.subtitle = [
            "each head adds a learned bias to the score of query i against key j, chosen by "
            "the offset i − j alone: query %d sees offsets %d down to 0" % (q, q),
            "the offsets go into %d buckets: 0 to 7 get one each, then wider and wider up to "
            "%d; every offset from %d on shares bucket %d" % (
                posenc.T5_BUCKETS, posenc.T5_MAX, self.far, posenc.T5_BUCKETS - 1),
            "the trained biases: %s, the previous token, which an order-2 language needs; "
            "far buckets sit near %s" % (who, num(float(b[:, -1].mean()), 1)),
            "past 32 every offset lands in bucket %d, which was trained: at 4× length the loss "
            "goes %.2f, %.2f, %.2f over positions 2–31, 32–63, 64–127 (best possible %.2f)" % (
                posenc.T5_BUCKETS - 1, l0, l1, l2, self.best),
        ][ph]
        if ops["offsets"] > 0.01:
            self._offsets(sc, t, q, ops["offsets"])
        if ops["buckets"] > 0.01:
            self._buckets(sc, self.progress("buckets", t), ops["buckets"])
        if ops["biases"] > 0.01:
            self._biases(sc, self.progress("biases", t), ops["biases"])
        if ops["long"] > 0.01:
            self._long(sc, self.progress("long", t), ops["long"])
        self.step_indicator(sc, ph, self.PHASES)

    def _box(self, sc, c, w, h, ink, op, width=1.6):
        """An outline ``w x h`` px centred at ``c``, in front of the images."""
        c = np.asarray(c, np.float64) + self.view * 0.8
        pts = [self.shift(c, sx * w / 2, sy * h / 2)
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1))]
        self.polyline(sc, pts, width, ink, op, bias=0.8)

    # -- phase 1: offsets ------------------------------------------------------------------
    def _offsets(self, sc, t, q, op):
        """A row of tokens, the query and the offsets of the keys before it; the offset matrix."""
        show = smoothstep(0.0, 0.6, self.progress("seq_in", t))
        o = op * show
        x0, x1 = SEQ_X
        xs = [x0 + (x1 - x0) * k / (N_SEQ - 1) for k in range(N_SEQ)]
        self.label(sc, self.P((x0 + x1) / 2, 0.29), "a sequence of the language, one symbol "
                   "per position", 0.027, opacity=o)
        for k, x in enumerate(xs):
            c = self.P(x, SEQ_Y)
            self.label(sc, self.shift(c, 0, -TOKEN_PX / 2 - 6), str(k), 0.024, anchor="mb",
                       opacity=o, ink=layout.NOTE_INK)
            if k == q:
                fill, ink = C.lighten(QUERY_INK, 0.55), QUERY_INK
            elif k < q:
                fill, ink = C.lighten(KEY_INK, 0.6), KEY_INK
            else:
                fill, ink = C.rgb("#f2f2ef"), EDGE
            self.rect(sc, c, TOKEN_PX, TOKEN_PX, fill, o, border=ink, radius=4.0)
            self.label(sc, c, SYMBOLS[self.seq[k]], 0.027, opacity=o, ink=layout.STRONG_INK,
                       kind="mono")
            if k <= q:
                self.label(sc, self.shift(c, 0, TOKEN_PX / 2 + 8), str(q - k), 0.026,
                           anchor="mt", opacity=o, ink=tuple(float(v) for v in ink))
        self.label(sc, self.P(x0 - 0.03, SEQ_Y + 0.055), "i − j", 0.026, anchor="rm", opacity=o,
                   ink=layout.NOTE_INK)
        self.label(sc, self.P(x0 - 0.03, SEQ_Y - 0.045), "j", 0.026, anchor="rm", opacity=o,
                   ink=layout.NOTE_INK)
        sc.probe(("query",), (q * 30.0, 0, 0), tol=6.0)
        # notes under the row
        self.label(sc, self.P(x0, 0.56), "query i = %d (red) against every key j ≤ i (green)" % q,
                   0.027, anchor="lm", opacity=o, ink=layout.STRONG_INK)
        self.label(sc, self.P(x0, 0.61), "score(i, j) = q_i · k_j / √8 + bias_h[bucket(i − j)]",
                   0.027, anchor="lm", opacity=o)
        self.label(sc, self.P(x0, 0.66), "no position vector is added to the tokens", 0.026,
                   anchor="lm", opacity=o, ink=layout.NOTE_INK)
        self.label(sc, self.P(x0, 0.70), "the same offset gets the same bias wherever it sits",
                   0.026, anchor="lm", opacity=o, ink=layout.NOTE_INK)
        # the offset matrix, the query's row outlined
        c = self.P(*OFFSET_AT)
        draw_image(self, sc, c, offset_colors(N_SEQ), OFFSET_PX, OFFSET_PX, o, grid=0.35)
        cell = OFFSET_PX / N_SEQ
        row = self.shift(c, 0, (q + 0.5) * cell - OFFSET_PX / 2)
        self._box(sc, row, OFFSET_PX + 4, cell + 2, QUERY_INK, o)
        self.label(sc, self.shift(c, 0, -OFFSET_PX / 2 - 10), "the offset i − j, %d × %d" % (
            N_SEQ, N_SEQ), 0.027, anchor="mb", opacity=o, ink=layout.STRONG_INK)
        self.label(sc, self.shift(c, 0, OFFSET_PX / 2 + 8), "rows: query i, columns: key j",
                   0.025, anchor="mt", opacity=o, ink=layout.NOTE_INK)
        self.label(sc, self.shift(c, 0, OFFSET_PX / 2 + 28), "light 0 to dark %d; one offset "
                   "per diagonal" % (N_SEQ - 1), 0.025, anchor="mt", opacity=o,
                   ink=layout.NOTE_INK)

    # -- phase 2: buckets ------------------------------------------------------------------
    def _strip(self, sc, n, y, reveal, op):
        """Offsets 0 .. n − 1 as cells coloured by bucket; returns the x of offset ``k``'s
        left edge (layout fraction)."""
        x0, x1 = STRIP_X
        cw = (x1 - x0) / n
        shown = int(np.ceil(reveal * n))
        bk = posenc.t5_bucket(np.arange(n), posenc.T5_BUCKETS, posenc.T5_MAX)
        tex = np.array([bucket_color(int(k)) for k in bk], np.float32)[None]
        tex[:, shown:] = C.BACKGROUND
        c = self.P((x0 + x1) / 2, y)
        wpx = self.px(self.P(x1, y))[0] - self.px(self.P(x0, y))[0]
        draw_image(self, sc, c, tex, wpx, STRIP_H, op, grid=0.0)
        return lambda k: x0 + cw * k

    def _buckets(self, sc, u, op):
        """The offsets 0 .. 31 and 0 .. 127 coloured by bucket, with the bucket boundaries."""
        n_short, n_long = self.n_short, self.n_long
        r1 = ease_out(remap(u, 0.02, 0.45))
        r2 = ease_out(remap(u, 0.5, 0.75))
        y = STRIP_SHORT_Y
        self.label(sc, self.P(STRIP_X[0], y - 0.115), "offsets 0 .. %d (up to the training "
                   "length), coloured by bucket; bucket numbers above" % (n_short - 1), 0.027,
                   anchor="lm", opacity=op, ink=layout.STRONG_INK)
        xat = self._strip(sc, n_short, y, r1, op)
        cw = (STRIP_X[1] - STRIP_X[0]) / n_short
        h = STRIP_H / 540.0 / layout.ZOOM               # the strip's height in layout units
        for k, (a, b) in enumerate(self.ranges):
            b = n_short - 1 if b is None else b
            if a >= n_short:
                break
            vis = op * smoothstep(a, a + 1, r1 * n_short)
            mid = (xat(a) + xat(b + 1)) / 2
            self.label(sc, self.P(mid, y - h / 2 - 0.035), str(k), 0.026, opacity=vis,
                       ink=tuple(float(v) for v in C.darken(bucket_color(k, alternate=False), 0.3)))
            if b > a:
                # a bracket over the shared bucket
                p0 = self.P(xat(a) + 0.15 * cw, y - h / 2 - 0.012)
                p1 = self.P(xat(b + 1) - 0.15 * cw, y - h / 2 - 0.012)
                self.polyline(sc, [p0, p1], 1.4, EDGE, vis)
            self.label(sc, self.P(xat(a) + cw / 2, y + h / 2 + 0.03), str(a), 0.024, opacity=vis,
                       ink=layout.NOTE_INK)
        sc.probe(("strip",), (r1 * 300 + r2 * 300, 0, 0), tol=6.0)
        # the whole line 0 .. 127
        y2 = STRIP_LONG_Y
        o2 = op * smoothstep(0.48, 0.55, u)
        self.label(sc, self.P(STRIP_X[0], y2 - 0.075), "offsets 0 .. %d: four times the "
                   "training length" % (n_long - 1), 0.027, anchor="lm", opacity=o2,
                   ink=layout.STRONG_INK)
        xat2 = self._strip(sc, n_long, y2, r2, o2)
        for k in (0, n_short, 2 * n_short, 3 * n_short, n_long - 1):
            self.label(sc, self.P(xat2(k + 0.5), y2 + h / 2 + 0.03), str(k), 0.024, opacity=o2,
                       ink=layout.NOTE_INK)
        o3 = op * smoothstep(0.75, 0.85, u)
        far = self.far
        p0 = self.P(xat2(far), y2 + h / 2 + 0.07)
        p1 = self.P(xat2(n_long), y2 + h / 2 + 0.07)
        self.polyline(sc, [p0, p1], 1.4, EDGE, o3)
        self.label(sc, self.P((xat2(far) + xat2(n_long)) / 2, y2 + h / 2 + 0.105),
                   "%d to %d and beyond: all bucket %d" % (far, n_long - 1, len(self.ranges) - 1),
                   0.026, opacity=o3, ink=layout.STRONG_INK)
        nlog = sum(1 for a, b in self.ranges if b is not None and b > a)
        self.label(sc, self.P(STRIP_X[0], 0.86), "%d buckets of one offset, %d of 2 to 4 "
                   "offsets (log spacing up to %d), 1 for the rest" % (
                       posenc.T5_BUCKETS // 2, nlog, posenc.T5_MAX), 0.026, anchor="lm",
                   opacity=o3, ink=layout.NOTE_INK)

    # -- phase 3: biases -------------------------------------------------------------------
    def _biases(self, sc, u, op):
        """The 4 × 16 bias table, a few of its values, and one head's bias matrix."""
        b = self.bias
        H, nb = b.shape
        vmax = float(np.abs(b).max())
        reveal = ease_out(remap(u, 0.02, 0.35))
        c = self.P(*TABLE_AT)
        tex = div_colors(b.ravel(), vmax).reshape(H, nb, 3)
        tex[:, int(np.ceil(reveal * nb)):] = C.BACKGROUND
        draw_image(self, sc, c, tex, TABLE_W, TABLE_H, op, grid=0.35)
        cw, rh = TABLE_W / nb, TABLE_H / H
        self.label(sc, self.shift(c, 0, -TABLE_H / 2 - 30), "the trained bias, head × bucket "
                   "(orange +, blue −)", 0.027, anchor="mb", opacity=op, ink=layout.STRONG_INK)
        for k in range(nb):
            self.label(sc, self.shift(c, (k + 0.5) * cw - TABLE_W / 2, -TABLE_H / 2 - 5), str(k),
                       0.024, anchor="mb", opacity=op, ink=layout.NOTE_INK)
        for h in range(H):
            ink = layout.STRONG_INK if h == self.head else layout.CAPTION_INK
            self.label(sc, self.shift(c, -TABLE_W / 2 - 8, (h + 0.5) * rh - TABLE_H / 2),
                       "head %d" % h, 0.026, anchor="rm", opacity=op, ink=ink)
        hl = smoothstep(0.4, 0.5, u)
        row = self.shift(c, 0, (self.head + 0.5) * rh - TABLE_H / 2)
        self._box(sc, row, TABLE_W + 4, rh + 2, BOX_INK, op * hl, width=2.0)
        # a few values, read off the table
        o2 = op * smoothstep(0.3, 0.4, u)
        x0 = TABLE_AT[0] - 0.24
        cols = [TABLE_AT[0] + 0.05 + 0.07 * h for h in range(H)]
        y0 = 0.62
        for h, x in enumerate(cols):
            self.label(sc, self.P(x, y0), "head %d" % h, 0.025, anchor="mm", opacity=o2,
                       ink=layout.NOTE_INK)
        rows = [("bucket 0: itself", 0), ("bucket 1: offset 1", 1),
                ("bucket %d: %d and up" % (nb - 1, self.far), nb - 1)]
        for r, (name, k) in enumerate(rows):
            y = y0 + 0.055 * (r + 1)
            self.label(sc, self.P(x0, y), name, 0.026, anchor="lm", opacity=o2)
            for h, x in enumerate(cols):
                ink = layout.STRONG_INK if k == 1 else layout.CAPTION_INK
                self.label(sc, self.P(x, y), num(float(b[h, k])), 0.026, opacity=o2, ink=ink,
                           kind="mono")
        # the chosen head's bias matrix
        o3 = op * smoothstep(0.45, 0.55, u)
        n = self.n_short
        m = self.bias_mat
        tex = div_colors(np.nan_to_num(m).ravel(), vmax,
                         mask=~np.isnan(m).ravel()).reshape(n, n, 3)
        shown = int(np.ceil(ease_out(remap(u, 0.5, 0.8)) * n))
        tex[shown:] = C.BACKGROUND
        cm = self.P(*BIAS_AT)
        draw_image(self, sc, cm, tex, BIAS_PX, BIAS_PX, o3, grid=0.0)
        self.label(sc, self.shift(cm, 0, -BIAS_PX / 2 - 10), "head %d's bias, %d × %d" % (
            self.head, n, n), 0.027, anchor="mb", opacity=o3, ink=layout.STRONG_INK)
        self.label(sc, self.shift(cm, 0, BIAS_PX / 2 + 8), "rows: query i, columns: key j",
                   0.025, anchor="mt", opacity=o3, ink=layout.NOTE_INK)
        o4 = op * smoothstep(0.8, 0.9, u)
        self.label(sc, self.shift(cm, 0, BIAS_PX / 2 + 28), "orange line under the diagonal: "
                   "offset 1, %s" % num(float(b[self.head, 1])), 0.025, anchor="mt", opacity=o4,
                   ink=layout.STRONG_INK)
        self.label(sc, self.shift(cm, 0, BIAS_PX / 2 + 48), "a band per bucket, wider further out",
                   0.025, anchor="mt", opacity=o4, ink=layout.NOTE_INK)
        sc.probe(("bias",), (reveal * 300 + shown * 8, 0, 0), tol=6.0)

    # -- phase 4: four times the length ----------------------------------------------------
    def _long(self, sc, u, op):
        """The head's attention at length 32 and 128; the loss by range of positions."""
        o1 = op * smoothstep(0.0, 0.08, u)
        o2 = op * smoothstep(0.15, 0.25, u)
        for (at, tex, n), o in zip(((MAP_AT[0], self.tex_short, self.n_short),
                                    (MAP_AT[1], self.tex_long, self.n_long)), (o1, o2)):
            c = self.P(*at)
            draw_image(self, sc, c, tex, MAP_PX, MAP_PX, o, grid=0.0)
            self.label(sc, self.shift(c, 0, -MAP_PX / 2 - 10), "head %d, length %d" % (
                self.head, n), 0.028, anchor="mb", opacity=o, ink=layout.STRONG_INK)
        # the trained corner of the long map
        c = self.P(*MAP_AT[1])
        k = MAP_PX * self.n_short / self.n_long
        o3 = op * smoothstep(0.3, 0.4, u)
        self._box(sc, self.shift(c, k / 2 - MAP_PX / 2, k / 2 - MAP_PX / 2), k, k, BOX_INK, o3)
        self.label(sc, self.shift(c, 0, MAP_PX / 2 + 8), "boxed: the left map, exactly", 0.025,
                   anchor="mt", opacity=o3, ink=layout.NOTE_INK)
        self.label(sc, self.shift(self.P(*MAP_AT[0]), 0, MAP_PX / 2 + 8),
                   "bright line: previous token", 0.025, anchor="mt", opacity=o1,
                   ink=layout.NOTE_INK)
        o4 = op * smoothstep(0.45, 0.55, u)
        xc = self.fr(self.shift(self.P(*MAP_AT[0]), -MAP_PX / 2, 0))[0]
        xc = 0.5 + (xc - 0.5) / layout.ZOOM                 # the left map's edge, layout x
        self.label(sc, self.P(xc, 0.85), "head %d's weight on the previous token: %.2f before "
                   "position %d, %.2f past %d" % (self.head, self.prev_in, self.n_short,
                                                   self.prev_far, 2 * self.n_short), 0.026,
                   anchor="lm", opacity=o4, ink=layout.STRONG_INK)
        self.label(sc, self.P(xc, 0.895), "offset 1's bias leads the far bucket's by %.1f; "
                   "past %d, %d to %d keys share that bucket" % (
                       self.lead_far, 2 * self.n_short, 2 * self.n_short - self.far + 1,
                       self.n_long - self.far), 0.025, anchor="lm", opacity=o4,
                   ink=layout.NOTE_INK)
        # the loss
        o5 = op * smoothstep(0.55, 0.65, u)
        x0 = 0.68
        self.label(sc, self.P(x0, 0.36), "test loss, 512 sequences", 0.027, anchor="lm",
                   opacity=o5, ink=layout.STRONG_INK)
        self.label(sc, self.P(x0, 0.40), "of length %d (nats per symbol)" % self.n_long, 0.025,
                   anchor="lm", opacity=o5, ink=layout.NOTE_INK)
        rows = [("positions %d–%d" % r, "%.2f" % v) for r, v in zip(self.loss_ranges,
                                                                    self.losses)]
        for r, (name, val) in enumerate(rows):
            y = 0.47 + 0.055 * r
            o = op * smoothstep(0.6 + 0.06 * r, 0.68 + 0.06 * r, u)
            self.label(sc, self.P(x0, y), name, 0.026, anchor="lm", opacity=o)
            self.label(sc, self.P(x0 + 0.24, y), val, 0.027, anchor="rm", opacity=o,
                       ink=layout.STRONG_INK, kind="mono")
        o6 = op * smoothstep(0.8, 0.9, u)
        for r, (name, val) in enumerate((("best possible", self.best),
                                          ("current token only", self.h1))):
            y = 0.67 + 0.055 * r
            self.label(sc, self.P(x0, y), name, 0.026, anchor="lm", opacity=o6,
                       ink=layout.NOTE_INK)
            self.label(sc, self.P(x0 + 0.24, y), "%.2f" % val, 0.027, anchor="rm", opacity=o6,
                       ink=layout.NOTE_INK, kind="mono")

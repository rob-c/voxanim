"""ALiBi: attention with a linear penalty for distance, in a real trained model.

ALiBi (Press et al., 2022) adds nothing to the tokens.  Head ``h`` subtracts
``m_h · (i − j)`` from the score of query ``i`` against key ``j``, with fixed
slopes ``m_h = 2^(−8h/H)`` (:func:`voxanim.posenc.alibi_slopes`: 1/4, 1/16,
1/64, 1/256 for 4 heads); nothing about position is learned.  The model is
the study's ALiBi run (:func:`voxanim.posenc.train`): one layer, 4 heads of
width 8, trained on an order-2 Markov language over 8 symbols at length 32
and tested at length 128.  Four phases:

1. **penalty** -- the score formula and the four heads' penalty matrices
   ``−m_h (i − j)`` for 16 positions, each on its own colour scale with its
   corner value printed: one triangle, four steepnesses,
2. **distance** -- ``e^(−m d)``, the factor the penalty puts on a key ``d``
   back, for ``d`` = 0 .. 127, one line per head, with the distance at which
   each halves (``ln 2 / m``) and the training length marked,
3. **attention** -- the four heads' attention on one test sequence of length
   32, and per head the mean weight on the token itself, on the previous one,
   and the mean distance looked back (64 sampled sequences); against the T5
   model's heads' weight on the previous token; and how strongly the shallowest
   head prefers some symbols to others as keys (:func:`symbol_preference`:
   the bright columns of its map),
4. **longer** -- the per-position test loss on length-128 sequences of the
   ALiBi and RoPE models (a moving average of 5 positions), the best possible
   loss and the loss with no context at all (:func:`voxanim.posenc.entropies`),
   and the mean loss of ALiBi, RoPE and T5 over positions 2–31, 32–63, 64–127.
"""

from __future__ import annotations

import math

import numpy as np

from . import color as C
from . import layout, posenc
from .diagram import Diagram, draw_image
from .easing import ease_in_out, ease_out, remap, smoothstep
from .netanim import OFF, div_colors, lit_colors

N_PEN = 16                           # positions of the penalty matrices
SEQ_SEED = 5                         # the sequence whose attention is drawn
STATS_SEED = 11                      # sequences for the attention statistics
N_STATS = 64
SMOOTH = 5                           # moving-average width of the loss curves
HEAD_INKS = [C.rgb(h) for h in ("#d84e4e", "#a1814b", "#3d9e45", "#6370eb")]
ROPE_INK = (0.14, 0.16, 0.18)
REF_INK = (0.50, 0.52, 0.54)
AXIS = (0.5, 0.52, 0.54)
HEAD_X = (0.15, 0.38, 0.61, 0.84)
PEN_Y, PEN_PX = 0.60, 160.0
ATT_Y, ATT_PX = 0.56, 170.0
ATT_VMAX = 0.5                       # attention weight drawn at full brightness
DIST_PLOT = (0.10, 0.34, 0.60, 0.80)
LOSS_PLOT = (0.08, 0.32, 0.58, 0.80)
LOSS_LO = 0.6                        # the loss plot's floor: room under the best-possible line


def smooth(v, w=SMOOTH):
    """A centred moving average of width ``w`` (shorter windows at the ends)."""
    v = np.asarray(v, np.float64)
    k = w // 2
    return np.array([v[max(0, i - k):i + k + 1].mean() for i in range(len(v))])


def penalty(slope, n):
    """``(n, n)``: ``−m (i − j)`` below the diagonal, NaN above (masked)."""
    i, j = np.arange(n)[:, None], np.arange(n)[None, :]
    P = -slope * (i - j).astype(np.float64)
    P[j > i] = np.nan
    return P


def fraction(m):
    """``1/4`` for 0.25: a slope as a unit fraction."""
    return "1/%d" % round(1 / m)


def head_stats(kind, P, xs):
    """Per head: the mean weight on the token itself, on the previous token, and
    the mean distance looked back, over queries 2 .. n − 1 of the sequences ``xs``."""
    A = posenc.forward(kind, P, xs, keep=True)[1]["A"]           # (B, H, n, n)
    n = xs.shape[1]
    q = np.arange(2, n)
    own = A[:, :, q, q].mean((0, 2))
    prev = A[:, :, q, q - 1].mean((0, 2))
    dist = np.maximum(np.arange(n)[:, None] - np.arange(n)[None, :], 0)
    look = (A * dist).sum(-1)[:, :, 2:].mean((0, 2))
    return own, prev, look


def symbol_preference(A, xs, h, first=8):
    """How much more head ``h`` weights its favourite key symbol than its least
    favourite: the mean weight per key of each symbol, keys before the previous
    token, queries from ``first`` on; the ratio of the largest to the smallest."""
    n = xs.shape[1]
    tot, cnt = np.zeros(posenc.VOCAB), np.zeros(posenc.VOCAB)
    for q in range(first, n):
        keys = xs[:, :q - 1]
        np.add.at(tot, keys.ravel(), A[:, h, q, :q - 1].ravel() * q)
        np.add.at(cnt, keys.ravel(), 1)
    w = tot / np.maximum(cnt, 1)
    return float(w.max() / w.min())


class ALiBi(Diagram):
    """ALiBi's slopes and penalties, then the study's trained ALiBi model."""

    PHASES = ("penalty", "distance", "attention", "longer")

    def __init__(self):
        super().__init__("ALiBi: attention with linear biases", "", 10.0)
        self.hold = 2.5
        H = posenc.HEADS
        self.slopes = posenc.alibi_slopes(H)
        self.pen = [penalty(m, N_PEN) for m in self.slopes]
        self.half = math.log(2) / self.slopes                  # distance at which e^(−md) halves
        run = posenc.train("alibi")
        rope = posenc.train("rope")
        t5 = posenc.train("t5")
        n = posenc.TRAIN_LEN
        self.n, self.n_long = n, posenc.TEST_LEN
        x = posenc.sample(1, n, np.random.default_rng(SEQ_SEED))[0]
        A = posenc.attention(run, x)
        self.att_tex = []
        for h in range(H):
            tex = lit_colors(A[h].ravel(), ATT_VMAX).reshape(n, n, 3)
            tex[np.triu(np.ones((n, n), bool), 1)] = OFF
            self.att_tex.append(tex)
        xs = posenc.sample(N_STATS, n, np.random.default_rng(STATS_SEED))
        self.own, self.prev, self.look = head_stats("alibi", run.P, xs)
        A_all = posenc.forward("alibi", run.P, xs, keep=True)[1]["A"]
        self.pref = symbol_preference(A_all, xs, H - 1)
        _, self.t5_prev, _ = head_stats("t5", t5.P, xs)
        self.best, self.h1, self.h0 = posenc.entropies()
        self.ranges = [(2, n - 1), (n, 2 * n - 1), (2 * n, self.n_long - 1)]

        def by_range(tl):                  # tl[p − 1]: the loss of predicting position p
            return [float(tl[a - 1:b].mean()) for a, b in self.ranges]

        self.rows = [("ALiBi", by_range(run.test_loss)), ("RoPE", by_range(rope.test_loss)),
                     ("T5 bias", by_range(t5.test_loss))]
        # positions 2 .. 127 (the first prediction sees one symbol only)
        self.pos = np.arange(2, self.n_long)
        self.curves = [smooth(run.test_loss[1:]), smooth(rope.test_loss[1:])]

        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("pen_in", 2.0, 0.0)
        add("penalty", 10.0, 1.0)
        add("distance", 11.0, 1.0)
        add("attention", 12.0, 1.0)
        add("longer", 13.0, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["distance"][0] - 0.5, Tm["attention"][0] - 0.5,
                      Tm["longer"][0] - 0.5]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _phase_ops(self, t, back):
        """Opacity of each phase's drawing: one at a time, with short crossfades."""
        Tm = self.Tm
        ends = {"penalty": Tm["distance"][0] - 0.5, "distance": Tm["attention"][0] - 0.5,
                "attention": Tm["longer"][0] - 0.5}
        starts = {"distance": Tm["distance"][0] - 0.4, "attention": Tm["attention"][0] - 0.4,
                  "longer": Tm["longer"][0] - 0.4}
        op = {}
        op["penalty"] = 1.0 - smoothstep(ends["penalty"] - 0.6, ends["penalty"], t)
        for key in ("distance", "attention"):
            op[key] = smoothstep(starts[key], starts[key] + 0.4, t) * (
                1.0 - smoothstep(ends[key] - 0.6, ends[key], t))
        op["longer"] = smoothstep(starts["longer"], starts["longer"] + 0.4, t)
        return {k: v * back for k, v in op.items()}

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ops = self._phase_ops(t, back)
        m = self.slopes
        (al, ro, t5) = (r[1] for r in self.rows)
        self.subtitle = [
            "no position vectors at all: head h subtracts m_h · (i − j) from every score, "
            "with fixed slopes m = %s" % ", ".join(fraction(v) for v in m),
            "the penalty against distance: e^(−m·d) halves every %.1f tokens in the steepest "
            "head and every %.0f in the shallowest" % (self.half[0], self.half[-1]),
            "the trained heads: the steepest looks back %.1f tokens on average, about evenly on "
            "itself (%.2f) and the previous token (%.2f); T5's heads give it %.2f–%.2f" % (
                self.look[0], self.own[0], self.prev[0], self.t5_prev.min(), self.t5_prev.max()),
            "at 4× length ALiBi's loss only goes from %.2f to %.2f and stays; RoPE's climbs to "
            "%.2f; but in range ALiBi trails T5 (%.2f against %.2f)" % (
                al[0], max(al[1], al[2]), ro[2], al[0], t5[0]),
        ][ph]
        if ops["penalty"] > 0.01:
            self._penalty(sc, t, ops["penalty"])
        if ops["distance"] > 0.01:
            self._distance(sc, self.progress("distance", t), ops["distance"])
        if ops["attention"] > 0.01:
            self._attention(sc, self.progress("attention", t), ops["attention"])
        if ops["longer"] > 0.01:
            self._longer(sc, self.progress("longer", t), ops["longer"])
        self.step_indicator(sc, ph, self.PHASES)

    def _ink(self, h):
        return tuple(float(v) for v in HEAD_INKS[h])

    # -- phase 1: the penalty matrices ----------------------------------------------------
    def _penalty(self, sc, t, op):
        """The formula and the four heads' penalty triangles."""
        show = smoothstep(0.0, 0.6, self.progress("pen_in", t))
        u = self.progress("penalty", t)
        o = op * show
        self.label(sc, self.P(0.5, 0.26), "score(i, j) = q_i · k_j / √8 − m_h · (i − j)", 0.031,
                   opacity=o, ink=layout.STRONG_INK)
        self.label(sc, self.P(0.5, 0.31), "the tokens go in as they are: the penalty is the "
                   "only thing that knows where a key sits", 0.026, opacity=o)
        rows = int(np.ceil(ease_out(remap(u, 0.0, 0.5)) * N_PEN))
        for h, (x, P, m) in enumerate(zip(HEAD_X, self.pen, self.slopes)):
            oh = o * smoothstep(0.08 * h, 0.08 * h + 0.1, u)
            c = self.P(x, PEN_Y)
            mask = ~np.isnan(P)
            tex = div_colors(np.nan_to_num(P).ravel(), float(np.nanmax(-P)),
                             mask=mask.ravel()).reshape(N_PEN, N_PEN, 3)
            tex[rows:] = C.BACKGROUND
            draw_image(self, sc, c, tex, PEN_PX, PEN_PX, oh, grid=0.0)
            self.label(sc, self.shift(c, 0, -PEN_PX / 2 - 30), "head %d" % h, 0.028,
                       anchor="mb", opacity=oh, ink=self._ink(h))
            self.label(sc, self.shift(c, 0, -PEN_PX / 2 - 8), "m = %s" % fraction(m), 0.027,
                       anchor="mb", opacity=oh, ink=layout.STRONG_INK)
            corner = -m * (N_PEN - 1)
            self.label(sc, self.shift(c, 0, PEN_PX / 2 + 8), "corner %s" % (
                ("%.3g" % corner).replace("-", "−")), 0.026, anchor="mt", opacity=oh)
        o2 = o * smoothstep(0.6, 0.7, u)
        self.label(sc, self.P(0.5, 0.86), "the same triangle in every head, scaled by m (each "
                   "drawn on its own scale, grey 0 to blue at the corner)", 0.025,
                   opacity=o2, ink=layout.NOTE_INK)
        self.label(sc, self.P(0.5, 0.90), "rows: query i, columns: key j, %d positions; dark: "
                   "the causal mask" % N_PEN, 0.025, opacity=o2, ink=layout.NOTE_INK)
        sc.probe(("pen",), (rows * 20.0, 0, 0), tol=6.0)

    # -- phase 2: against distance --------------------------------------------------------
    def _distance(self, sc, u, op):
        """``e^(−m d)`` for d = 0 .. 127, one line per head."""
        x0, y0, x1, y1 = DIST_PLOT
        D = self.n_long - 1
        reveal = ease_out(remap(u, 0.05, 0.6))
        k = max(2, int(reveal * D) + 1)

        def pt(d, v):
            return self.P(x0 + (x1 - x0) * d / D, y1 - (y1 - y0) * v)

        self.polyline(sc, [self.P(x0, y1), self.P(x1, y1)], 1.0, AXIS, 0.7 * op)
        self.polyline(sc, [self.P(x0, y0), self.P(x0, y1)], 1.0, AXIS, 0.7 * op)
        for d in (0, 32, 64, 96, 127):
            self.label(sc, self.P(x0 + (x1 - x0) * d / D, y1 + 0.015), str(d), 0.024,
                       anchor="mt", opacity=op, ink=layout.NOTE_INK)
        for v in (0.5, 1.0):            # the 0 under the axis would meet the d = 0
            self.label(sc, self.P(x0 - 0.01, y1 - (y1 - y0) * v), "%g" % v, 0.024, anchor="rm",
                       opacity=op, ink=layout.NOTE_INK)
        self.label(sc, self.P((x0 + x1) / 2, y1 + 0.05), "distance d = i − j", 0.026,
                   anchor="mt", opacity=op)
        self.label(sc, self.P(x0, y0 - 0.03), "e^(−m·d): the factor on a key d back, all "
                   "else equal", 0.026, anchor="lb", opacity=op)
        # the training length
        xt = x0 + (x1 - x0) * self.n / D
        self.polyline(sc, [self.P(xt, y0), self.P(xt, y1)], 1.0, REF_INK, 0.6 * op)
        self.label(sc, self.P(xt + 0.008, y0 + 0.01), "trained up to %d" % self.n, 0.024,
                   anchor="lt", opacity=op, ink=layout.NOTE_INK)
        d = np.arange(k)
        for h, m in enumerate(self.slopes):
            pts = [pt(dd, v) for dd, v in zip(d, np.exp(-m * d))]
            self.polyline(sc, pts, 2.0, HEAD_INKS[h], op)
            self.disk(sc, pts[-1], 3.0, HEAD_INKS[h], op, lift=1.0, outline=0.0)
        sc.probe(("dist",), (k * 6.0, 0, 0), tol=6.0)
        o2 = op * smoothstep(0.2, 0.3, u)
        self.label(sc, self.P(0.66, 0.36), "halves every", 0.026, anchor="lm", opacity=o2,
                   ink=layout.NOTE_INK)
        self.label(sc, self.P(0.95, 0.36), "at d = %d" % self.n, 0.026, anchor="rm", opacity=o2,
                   ink=layout.NOTE_INK)
        for h, m in enumerate(self.slopes):
            y = 0.42 + 0.06 * h
            oh = op * smoothstep(0.25 + 0.08 * h, 0.33 + 0.08 * h, u)
            self.label(sc, self.P(0.66, y), "m = %s" % fraction(m), 0.026, anchor="lm",
                       opacity=oh, ink=self._ink(h))
            self.label(sc, self.P(0.83, y), "%.1f" % self.half[h], 0.026, anchor="rm",
                       opacity=oh, ink=layout.STRONG_INK, kind="mono")
            self.label(sc, self.P(0.95, y), "%.2g" % math.exp(-m * self.n), 0.026, anchor="rm",
                       opacity=oh, ink=layout.STRONG_INK, kind="mono")
        o3 = op * smoothstep(0.75, 0.85, u)
        self.label(sc, self.P(0.66, 0.70), "a steep head sees the last few", 0.025, anchor="lm",
                   opacity=o3, ink=layout.NOTE_INK)
        self.label(sc, self.P(0.66, 0.74), "tokens, a shallow one all of them;", 0.025,
                   anchor="lm", opacity=o3, ink=layout.NOTE_INK)
        self.label(sc, self.P(0.66, 0.78), "defined for every d, trained or not", 0.025,
                   anchor="lm", opacity=o3, ink=layout.NOTE_INK)

    # -- phase 3: the trained heads -------------------------------------------------------
    def _attention(self, sc, u, op):
        """The four heads' attention on one sequence, and what they put where."""
        n = self.n
        rows = int(np.ceil(ease_out(remap(u, 0.02, 0.4)) * n))
        for h, x in enumerate(HEAD_X):
            c = self.P(x, ATT_Y)
            tex = self.att_tex[h].copy()
            tex[rows:] = C.BACKGROUND
            draw_image(self, sc, c, tex, ATT_PX, ATT_PX, op, grid=0.0)
            self.label(sc, self.shift(c, 0, -ATT_PX / 2 - 8), "head %d, m = %s" % (
                h, fraction(self.slopes[h])), 0.027, anchor="mb", opacity=op, ink=self._ink(h))
            oh = op * smoothstep(0.4 + 0.06 * h, 0.48 + 0.06 * h, u)
            self.label(sc, self.shift(c, 0, ATT_PX / 2 + 8), "itself %.2f, previous %.2f" % (
                self.own[h], self.prev[h]), 0.025, anchor="mt", opacity=oh,
                ink=layout.STRONG_INK)
            self.label(sc, self.shift(c, 0, ATT_PX / 2 + 28), "looks back %.1f on average" %
                       self.look[h], 0.025, anchor="mt", opacity=oh, ink=layout.CAPTION_INK)
        self.label(sc, self.P(0.5, 0.25), "attention on one test sequence of %d (rows: query, "
                   "columns: key); the numbers are means over %d" % (n, N_STATS), 0.025,
                   opacity=op, ink=layout.NOTE_INK)
        self.label(sc, self.P(0.5, 0.29), "bright columns: keys chosen by symbol; head %d "
                   "weights one symbol %.1f× another" % (posenc.HEADS - 1, self.pref), 0.025,
                   opacity=op * smoothstep(0.5, 0.6, u), ink=layout.NOTE_INK)
        sc.probe(("att",), (rows * 10.0, 0, 0), tol=6.0)
        o2 = op * smoothstep(0.75, 0.85, u)
        self.label(sc, self.P(0.5, 0.85), "T5's learned biases point its heads at the previous "
                   "token: %.2f to %.2f of their weight" % (self.t5_prev.min(),
                                                             self.t5_prev.max()), 0.026,
                   opacity=o2, ink=layout.STRONG_INK)
        self.label(sc, self.P(0.5, 0.895), "ALiBi's penalty only falls with distance: it always "
                   "favours the token itself over the one before", 0.025, opacity=o2,
                   ink=layout.NOTE_INK)

    # -- phase 4: four times the length ---------------------------------------------------
    def _longer(self, sc, u, op):
        """Per-position loss on length-128 sequences: ALiBi and RoPE, with references."""
        x0, y0, x1, y1 = LOSS_PLOT
        lo = LOSS_LO
        hi = math.ceil(max(float(c.max()) for c in self.curves) * 2) / 2
        P0, P1 = 0, self.n_long - 1

        def pt(p, v):
            return self.P(x0 + (x1 - x0) * (p - P0) / (P1 - P0), y1 - (y1 - y0) * (v - lo) /
                          (hi - lo))

        self.polyline(sc, [self.P(x0, y1), self.P(x1, y1)], 1.0, AXIS, 0.7 * op)
        self.polyline(sc, [self.P(x0, y0), self.P(x0, y1)], 1.0, AXIS, 0.7 * op)
        for p in (32, 64, 96, 127):
            self.label(sc, self.P(x0 + (x1 - x0) * p / P1, y1 + 0.015), str(p), 0.024,
                       anchor="mt", opacity=op, ink=layout.NOTE_INK)
        for v in np.arange(1.0, hi + 1e-9, 0.5 if hi - lo <= 2 else 1.0):
            self.label(sc, pt(P0, v), "%g" % v, 0.024, anchor="rm", dx=-6, opacity=op,
                       ink=layout.NOTE_INK)
        self.label(sc, self.P((x0 + x1) / 2, y1 + 0.05), "position in a test sequence of %d"
                   % self.n_long, 0.026, anchor="mt", opacity=op)
        self.label(sc, self.P(x0, y0 - 0.03), "loss per position (nats per symbol)", 0.026,
                   anchor="lb", opacity=op)
        xt = x0 + (x1 - x0) * self.n / P1
        self.polyline(sc, [self.P(xt, y0), self.P(xt, y1)], 1.0, REF_INK, 0.6 * op)
        self.label(sc, self.P(xt + 0.008, y0 + 0.01), "trained up to %d" % self.n, 0.024,
                   anchor="lt", opacity=op, ink=layout.NOTE_INK)
        # references
        o_ref = op * smoothstep(0.05, 0.15, u)
        for v, name in ((self.best, "best possible %.2f" % self.best),
                        (self.h0, "no context %.2f" % self.h0)):
            self.polyline(sc, [pt(P0, v), pt(P1, v)], 1.0, REF_INK, 0.8 * o_ref)
            below = v < 1.5
            self.label(sc, pt(P1, v), name, 0.024, anchor="rt" if below else "rb",
                       dy=2 if below else -3, opacity=o_ref, ink=layout.NOTE_INK)
        reveal = ease_out(remap(u, 0.1, 0.6))
        k = max(2, int(reveal * len(self.pos)))
        o_end = op * smoothstep(0.6, 0.7, u)
        for curve, ink, w, name in zip(self.curves, (HEAD_INKS[3], ROPE_INK), (2.2, 1.8),
                                       ("ALiBi", "RoPE")):
            pts = [pt(p, v) for p, v in zip(self.pos[:k], curve[:k])]
            self.polyline(sc, pts, w, ink, op)
            self.disk(sc, pts[-1], 3.0, ink, op, lift=1.0, outline=0.0)
            self.label(sc, pts[-1], name, 0.026, anchor="rb", dy=-8, opacity=o_end,
                       ink=tuple(float(v) for v in ink))
        sc.probe(("loss",), (k * 6.0, 0, 0), tol=6.0)
        # the table of means
        o2 = op * smoothstep(0.55, 0.65, u)
        xs = (0.62, 0.77, 0.86, 0.95)
        self.label(sc, self.P(xs[0], 0.36), "mean loss, positions", 0.025, anchor="lm",
                   opacity=o2, ink=layout.NOTE_INK)
        for x, (a, b) in zip(xs[1:], self.ranges):
            self.label(sc, self.P(x, 0.41), "%d–%d" % (a, b), 0.025, anchor="rm", opacity=o2,
                       ink=layout.NOTE_INK)
        inks = (self._ink(3), ROPE_INK, layout.CAPTION_INK)
        for r, ((name, vals), ink) in enumerate(zip(self.rows, inks)):
            y = 0.47 + 0.055 * r
            orow = op * smoothstep(0.6 + 0.05 * r, 0.68 + 0.05 * r, u)
            self.label(sc, self.P(xs[0], y), name, 0.026, anchor="lm", opacity=orow, ink=ink)
            for x, v in zip(xs[1:], vals):
                self.label(sc, self.P(x, y), "%.2f" % v, 0.026, anchor="rm", opacity=orow,
                           ink=layout.STRONG_INK, kind="mono")
        o3 = op * smoothstep(0.85, 0.95, u)
        ro = self.rows[1][1]
        notes = ("ALiBi: flat past %d, but no head puts" % self.n,
                 "over %.2f on the previous token" % self.prev.max(),
                 "RoPE past %d: %.2f, %s the %.2f" % (
                     2 * self.n, ro[2], "above" if ro[2] > self.h0 else "below", self.h0),
                 "of a guess with no context at all")
        for r, text in enumerate(notes):
            self.label(sc, self.P(xs[0], 0.67 + 0.04 * r), text, 0.025, anchor="lm",
                       opacity=o3, ink=layout.NOTE_INK)

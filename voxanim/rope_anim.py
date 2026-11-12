"""Rotary position embeddings (RoPE), and GPT-2's learned absolute positions.

Maths, drawn with real numbers, plus one real matrix: GPT-2 small's position
embedding ``wpe`` (1024 × 768, :mod:`voxanim.gpt2`) for the comparison.

RoPE splits a query or key of width ``d`` into ``d / 2`` pairs and rotates
pair ``i`` at position ``m`` by the angle ``m θ_i``, ``θ_i = 10000^(−2i/d)``
(here ``d = 64``, GPT-2's head width, so ``θ_0 = 1`` and ``θ_31 ≈ 0.00075``):

    R(m) q = (q₂ᵢ cos mθᵢ − q₂ᵢ₊₁ sin mθᵢ,  q₂ᵢ sin mθᵢ + q₂ᵢ₊₁ cos mθᵢ)  for each pair i

A rotation of ``q`` by ``mθ`` dotted with a rotation of ``k`` by ``nθ`` is
``q`` dotted with ``k`` rotated by ``(n − m) θ``: the score depends on the
relative position only.  Four phases:

1. **pairs** -- a query and a key split into pairs; pairs 0, 4 and 8 drawn as
   arrows that turn as the position runs from 0 to 15, each at its own rate,
2. **relative** -- the score ``R(m) q · R(n) k`` for the pairs (3, 7) and
   (10, 14): the same number (printed to 10 decimals), while (3, 10) differs,
3. **distance** -- the score between a vector and a copy of itself ``Δ``
   positions away, per pair: ``|q_i|² cos(Δ θ_i)``, fast pairs oscillating,
   slow ones hardly moving; their sum decays and oscillates with distance,
4. **GPT-2** -- the cosine similarity between GPT-2's learned position
   vectors, ``wpe[m] · wpe[n] / |wpe[m]| |wpe[n]|``, for positions 0 .. 63
   (a real matrix): smooth and decaying, but tied to the absolute index and
   ending at row 1023; the rotary score at a distance is the same formula
   wherever the pair sits, including positions never seen in training.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import gpt2, layout
from .diagram import EDGE, Diagram, draw_image
from .easing import ease_in_out, ease_out, remap, smoothstep
from .netanim import gray_colors
from .text import num

D_HEAD = gpt2.D_HEAD                 # 64: the rotary width used here
BASE = 10000.0                       # RoPE's base
SHOWN_PAIRS = (0, 4, 8)              # the pairs drawn as turning arrows
N_POSITIONS = 16                     # positions the arrows run through
PAIRS_MN = ((3, 7), (10, 14), (3, 10))   # the relative-position check
N_DISTANCE = 64                      # distances of the decay plot and the GPT-2 matrix
N_GPT2 = 64                          # positions of the GPT-2 similarity matrix
SEED = 0
QUERY_INK, KEY_INK = C.rgb("#d84e4e"), C.rgb("#3d9e45")
PAIR_INKS = (C.rgb("#d84e4e"), C.rgb("#a1814b"), C.rgb("#6370eb"))
SUM_INK = (0.14, 0.16, 0.18)
ARROW_R = 46.0                       # px: the rotating arrows' length
DIAL_AT = ((0.17, 0.56), (0.42, 0.56), (0.67, 0.56))
PLOT = (0.10, 0.36, 0.62, 0.80)     # the distance plot's box: x0, y0, x1, y1 (layout)
MATRIX_AT, MATRIX_PX = (0.26, 0.60), 260.0
CURVE = (0.56, 0.36, 0.93, 0.75)    # the GPT-2 similarity curve's box


def frequencies(d=D_HEAD, base=BASE):
    """``θ_i = base^(−2i/d)`` for the ``d / 2`` pairs."""
    return base ** (-np.arange(d // 2) * 2.0 / d)


def rotate(x, m, theta):
    """``R(m) x``: every pair of ``x`` ``(d,)`` turned by ``m θ_i``."""
    x1, x2 = x[0::2], x[1::2]
    c, s = np.cos(m * theta), np.sin(m * theta)
    out = np.empty_like(x)
    out[0::2] = x1 * c - x2 * s
    out[1::2] = x1 * s + x2 * c
    return out


def score(q, k, m, n, theta):
    """``R(m) q · R(n) k``."""
    return float(rotate(q, m, theta) @ rotate(k, n, theta))


def self_score_by_pair(q, delta, theta):
    """``|q_i|² cos(Δ θ_i)`` per pair: the score of ``q`` against itself ``delta`` apart."""
    power = q[0::2] ** 2 + q[1::2] ** 2
    return power * np.cos(delta * theta)


def gpt2_similarity(n=N_GPT2):
    """Cosine similarity between GPT-2 small's first ``n`` position vectors, ``(n, n)``."""
    W = gpt2.weights()["wpe.weight"][:n].astype(np.float64)
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    return W @ W.T


class RotaryEmbeddings(Diagram):
    """RoPE on a seeded query and key of width 64, then GPT-2's position matrix."""

    PHASES = ("pairs", "relative", "distance", "GPT-2")

    def __init__(self):
        super().__init__("Rotary position embeddings", "", 10.0)
        self.hold = 2.5
        rng = np.random.default_rng(SEED)
        self.q = rng.normal(0, 1, D_HEAD) / np.sqrt(D_HEAD)
        self.k = rng.normal(0, 1, D_HEAD) / np.sqrt(D_HEAD)
        self.theta = frequencies()
        self.scores = [score(self.q, self.k, m, n, self.theta) for m, n in PAIRS_MN]
        self.sim = gpt2_similarity()
        self.by_pair = np.array([self_score_by_pair(self.q, dl, self.theta)
                                 for dl in range(N_DISTANCE)])      # (distance, pair)
        self.total = self.by_pair.sum(1)

        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("pairs_in", 2.5, 0.0)
        add("turn", 9.0, 1.0)
        add("relative", 9.0, 1.0)
        add("distance", 10.0, 1.0)
        add("gpt2", 10.0, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["relative"][0] - 0.5, Tm["distance"][0] - 0.5, Tm["gpt2"][0] - 0.5]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _phase_ops(self, t, back):
        """Opacity of each phase's drawing: one at a time, with short crossfades."""
        Tm = self.Tm
        ends = {"pairs": Tm["relative"][0] - 0.5, "relative": Tm["distance"][0] - 0.5,
                "distance": Tm["gpt2"][0] - 0.5}
        starts = {"relative": Tm["relative"][0] - 0.4, "distance": Tm["distance"][0] - 0.4,
                  "gpt2": Tm["gpt2"][0] - 0.4}
        op = {}
        op["pairs"] = 1.0 - smoothstep(ends["pairs"] - 0.6, ends["pairs"], t)
        for key in ("relative", "distance"):
            op[key] = smoothstep(starts[key], starts[key] + 0.4, t) * (
                1.0 - smoothstep(ends[key] - 0.6, ends[key], t))
        op["gpt2"] = smoothstep(starts["gpt2"], starts["gpt2"] + 0.4, t)
        return {k: v * back for k, v in op.items()}

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ops = self._phase_ops(t, back)
        turn = self.progress("turn", t)
        pos = ease_in_out(turn) * (N_POSITIONS - 1)
        rel = self.progress("relative", t)
        dist = self.progress("distance", t)
        g2 = self.progress("gpt2", t)
        th = self.theta
        s0, _s1, s2 = self.scores
        self.subtitle = [
            "position %d: pair i turns by %d·θᵢ, θᵢ = 10000^(−2i/64); pair 0 by %.1f rad, "
            "pair 4 by %.2f, pair 8 by %.3f" % (round(pos), round(pos), pos * th[0], pos * th[4],
                                                 pos * th[8]),
            "R(m)q · R(n)k depends on n − m only: positions (3, 7) and (10, 14) both give %.6f; "
            "(3, 10) gives %.6f" % (s0, s2),
            "a vector against itself Δ positions away: Σᵢ |qᵢ|² cos(Δθᵢ), fast pairs oscillate, "
            "slow ones hold, the sum decays",
            "GPT-2's learned positions: the cosine similarity of wpe rows %d apart is %.2f, "
            "%d apart %.2f; there is no row 1024" % (1, self._sim_at(1), 16, self._sim_at(16)),
        ][ph]
        if ops["pairs"] > 0.01:
            self._pairs(sc, t, pos, ops["pairs"])
        if ops["relative"] > 0.01:
            self._relative(sc, rel, ops["relative"])
        if ops["distance"] > 0.01:
            self._distance(sc, dist, ops["distance"])
        if ops["gpt2"] > 0.01:
            self._gpt2(sc, g2, ops["gpt2"])
        self.step_indicator(sc, ph, self.PHASES)

    def _sim_at(self, gap):
        """Mean cosine similarity of GPT-2 position vectors ``gap`` apart (first 64 rows)."""
        return float(np.mean(np.diag(self.sim, gap)))

    # -- phase 1: the pairs turning ----------------------------------------------------------
    def _pairs(self, sc, t, pos, op):
        """Three dials: the query's (red) and the key's (green) pair as arrows that turn by
        ``m θ_i`` as the position ``m`` runs."""
        show = smoothstep(0.0, 0.6, self.progress("pairs_in", t))
        self.label(sc, self.P(0.5, 0.26), "a query q and a key k, 64 numbers each, read as 32 "
                   "pairs (x, y); every pair is a little arrow", 0.027, opacity=op * show)
        for i, at in zip(SHOWN_PAIRS, DIAL_AT):
            c = self.P(*at)
            # a ring of segments, not a filled disk: billboards would hide the arrows
            ring = [self.shift(c, (ARROW_R + 6) * np.cos(a), (ARROW_R + 6) * np.sin(a))
                    for a in np.linspace(0, 2 * np.pi, 49)]
            self.polyline(sc, ring, 1.0, EDGE, 0.6 * op * show)
            self.disk(sc, c, 2.0, EDGE, op * show, lift=0.8, outline=0.0)
            for vec, ink in ((self.q, QUERY_INK), (self.k, KEY_INK)):
                x, y = vec[2 * i], vec[2 * i + 1]
                r = np.hypot(x, y)
                a0 = np.arctan2(y, x)
                a = a0 + pos * self.theta[i]
                scale = ARROW_R / max(np.hypot(*self.q[2 * i:2 * i + 2]),
                                      np.hypot(*self.k[2 * i:2 * i + 2]))
                tip = self.shift(c, r * scale * np.cos(a), -r * scale * np.sin(a))
                sc.line(c, tip, width=2.4, color=ink, opacity=op * show, bias=0.6)
                self.disk(sc, tip, 3.5, ink, op * show, lift=1.0, outline=0.0)
                # the start, faint
                tip0 = self.shift(c, r * scale * np.cos(a0), -r * scale * np.sin(a0))
                sc.line(c, tip0, width=1.0, color=ink, opacity=0.35 * op * show, bias=0.4)
            self.label(sc, self.shift(c, 0, -ARROW_R - 16), "pair %d: θ = %s" % (
                i, ("%.4g" % self.theta[i])), 0.027, anchor="mb", opacity=op * show,
                ink=layout.STRONG_INK)
            self.label(sc, self.shift(c, 0, ARROW_R + 14), "turned by %s rad" % num(
                pos * self.theta[i], 3), 0.026, anchor="mt", opacity=op * show)
        self.label(sc, self.P(0.17, 0.83), "position m = %d" % round(pos), 0.03, anchor="lm",
                   opacity=op * show, ink=layout.STRONG_INK)
        self.label(sc, self.P(0.17, 0.875), "q in red, k in green; the faint arrow is position 0",
                   0.025, anchor="lm", opacity=op * show, ink=layout.NOTE_INK)
        sc.probe(("turn",), (pos * 30, 0, 0), tol=6.0)

    # -- phase 2: relative position -----------------------------------------------------------
    def _relative(self, sc, rel, op):
        """Three rows: the positions of q and k, the gap, the score; the first two equal."""
        self.label(sc, self.P(0.5, 0.30), "score = R(m) q · R(n) k", 0.031, opacity=op,
                   ink=layout.STRONG_INK)
        self.label(sc, self.P(0.5, 0.355), "a rotation by mθ against a rotation by nθ is a "
                   "rotation by (n − m) θ: only the gap survives", 0.026, opacity=op)
        heads = ("q at m", "k at n", "n − m", "score")
        xs = (0.22, 0.38, 0.54, 0.74)
        for x, h in zip(xs, heads):
            self.label(sc, self.P(x, 0.45), h, 0.027, opacity=op, ink=layout.NOTE_INK)
        for r, ((m, n), s) in enumerate(zip(PAIRS_MN, self.scores)):
            show = smoothstep(0.1 + 0.25 * r, 0.25 + 0.25 * r, rel)
            y = 0.53 + 0.09 * r
            ink = layout.STRONG_INK if r < 2 else layout.CAPTION_INK
            for x, text in zip(xs, (str(m), str(n), str(n - m), "%.10f" % s)):
                self.label(sc, self.P(x, y), text, 0.03, opacity=op * show, ink=ink,
                           kind="mono" if x == xs[-1] else "regular")
        same = smoothstep(0.62, 0.75, rel)
        a, b = self.P(0.86, 0.53), self.P(0.86, 0.62)
        sc.line(a, b, width=1.6, color=C.INPUT, opacity=op * same, bias=0.3)
        self.label(sc, self.P(0.875, 0.575), "equal", 0.027, anchor="lm", opacity=op * same,
                   ink=tuple(float(v) for v in C.INPUT))
        self.label(sc, self.P(0.5, 0.80), "the third pair sits 7 apart, not 4: a different score",
                   0.026, opacity=op * smoothstep(0.85, 0.95, rel))

    # -- phase 3: decay with distance ---------------------------------------------------------
    def _distance(self, sc, dist, op):
        """Per-pair scores against distance for the three pairs, and the sum over all 32."""
        x0, y0, x1, y1 = PLOT
        n_show = max(1, int(ease_out(remap(dist, 0.05, 0.8)) * (N_DISTANCE - 1)) + 1)
        vmax = float(np.abs(self.total).max())
        mid = (y0 + y1) / 2

        def pt(dl, v):
            return self.P(x0 + (x1 - x0) * dl / (N_DISTANCE - 1), mid - (y1 - y0) / 2 * v / vmax)

        axis = (0.5, 0.52, 0.54)
        self.polyline(sc, [self.P(x0, mid), self.P(x1, mid)], 1.0, axis, 0.7 * op)
        self.polyline(sc, [self.P(x0, y0), self.P(x0, y1)], 1.0, axis, 0.7 * op)
        for dl in (0, 16, 32, 48):
            self.label(sc, self.P(x0 + (x1 - x0) * dl / (N_DISTANCE - 1), y1 + 0.015), str(dl),
                       0.024, anchor="mt", opacity=op, ink=layout.NOTE_INK)
        self.label(sc, self.P((x0 + x1) / 2, y1 + 0.05), "distance Δ between the two positions",
                   0.026, anchor="mt", opacity=op)
        self.label(sc, self.P(x0, y0 - 0.02), "score of q against itself, Δ apart", 0.026,
                   anchor="lb", opacity=op)
        pmax = float(np.abs(self.by_pair[:, list(SHOWN_PAIRS)]).max())
        for i, ink in zip(SHOWN_PAIRS, PAIR_INKS):
            pts = [pt(dl, self.by_pair[dl, i] / pmax * vmax * 0.5) for dl in range(n_show)]
            self.polyline(sc, pts, 1.3, ink, 0.85 * op)
        pts = [pt(dl, self.total[dl]) for dl in range(n_show)]
        self.polyline(sc, pts, 2.0, SUM_INK, op)
        self.disk(sc, pts[-1], 3.0, SUM_INK, op, lift=1.0, outline=0.0)
        legend = [("pair %d (θ = %.4g)" % (i, self.theta[i]), ink)
                  for i, ink in zip(SHOWN_PAIRS, PAIR_INKS)] + [("sum over all 32 pairs", SUM_INK)]
        for n, (name, ink) in enumerate(legend):
            self.label(sc, self.P(0.70, 0.40 + 0.055 * n), name, 0.026, anchor="lm", opacity=op,
                       ink=tuple(float(v) for v in ink))
        self.label(sc, self.P(0.70, 0.66), "|q|² = %.2f at Δ = 0" % self.total[0], 0.026,
                   anchor="lm", opacity=op, ink=layout.STRONG_INK)
        self.label(sc, self.P(0.70, 0.70), "at Δ = %d: %s" % (n_show - 1,
                                                               num(self.total[n_show - 1])),
                   0.026, anchor="lm", opacity=op, ink=layout.STRONG_INK)
        self.label(sc, self.P(0.70, 0.76), "the formula holds for any Δ,", 0.025, anchor="lm",
                   opacity=op * smoothstep(0.8, 0.95, dist), ink=layout.NOTE_INK)
        self.label(sc, self.P(0.70, 0.80), "trained on or not", 0.025, anchor="lm",
                   opacity=op * smoothstep(0.8, 0.95, dist), ink=layout.NOTE_INK)
        sc.probe(("dist",), (n_show * 8, 0, 0), tol=6.0)

    # -- phase 4: GPT-2's learned positions ------------------------------------------------------
    def _gpt2(self, sc, g2, op):
        """The 64 × 64 cosine-similarity matrix of wpe, and similarity against the gap."""
        n = N_GPT2
        reveal = ease_out(remap(g2, 0.05, 0.6))
        shown = max(1, int(reveal * n))
        sim = self.sim.copy()
        tex = gray_colors(np.clip(sim, 0, 1).ravel()).reshape(n, n, 3)
        tex[shown:, :] = C.BACKGROUND
        tex[:, shown:] = C.BACKGROUND
        c = self.P(*MATRIX_AT)
        draw_image(self, sc, c, tex, MATRIX_PX, MATRIX_PX, op, grid=0.0)
        self.label(sc, self.shift(c, 0, -MATRIX_PX / 2 - 12),
                   "cos(wpe[m], wpe[n]), positions 0 .. 63", 0.027, anchor="mb", opacity=op,
                   ink=layout.STRONG_INK)
        self.label(sc, self.shift(c, 0, MATRIX_PX / 2 + 10), "GPT-2 small's learned table: 1024 "
                   "rows × 768", 0.026, anchor="mt", opacity=op)
        self.label(sc, self.shift(c, 0, MATRIX_PX / 2 + 32), "white 1, black 0: bright near the "
                   "diagonal, fading with the gap", 0.025, anchor="mt", opacity=op,
                   ink=layout.NOTE_INK)
        # similarity against the gap, averaged over the first 64 rows
        x0, y0, x1, y1 = CURVE
        gaps = np.arange(1, n)
        vals = np.array([self._sim_at(g) for g in gaps])
        axis = (0.5, 0.52, 0.54)
        self.polyline(sc, [self.P(x0, y1), self.P(x1, y1)], 1.0, axis, 0.7 * op)
        self.polyline(sc, [self.P(x0, y0), self.P(x0, y1)], 1.0, axis, 0.7 * op)

        def pt(g, v):
            return self.P(x0 + (x1 - x0) * g / (n - 1), y1 - (y1 - y0) * np.clip(v, 0, 1))

        k = max(2, int(reveal * (n - 1)))
        self.polyline(sc, [pt(g, v) for g, v in zip(gaps[:k], vals[:k])], 2.0, SUM_INK, op)
        for g in (0, 16, 32, 48):
            self.label(sc, self.P(x0 + (x1 - x0) * g / (n - 1), y1 + 0.015), str(g), 0.024,
                       anchor="mt", opacity=op, ink=layout.NOTE_INK)
        self.label(sc, self.P((x0 + x1) / 2, y1 + 0.05), "gap n − m", 0.026, anchor="mt",
                   opacity=op)
        self.label(sc, self.P(x0, y0 - 0.02), "mean cosine similarity", 0.026, anchor="lb",
                   opacity=op)
        self.label(sc, self.P(x1, y0 + 0.02), "1 apart: %.2f" % vals[0], 0.026, anchor="rt",
                   opacity=op, ink=layout.STRONG_INK)
        self.label(sc, self.P(x1, y0 + 0.06), "16 apart: %.2f" % vals[15], 0.026, anchor="rt",
                   opacity=op, ink=layout.STRONG_INK)
        late = smoothstep(0.7, 0.9, g2)
        self.label(sc, self.P(0.745, 0.86), "learned: a row per absolute index, none past 1023",
                   0.025, opacity=op * late, ink=layout.NOTE_INK)
        self.label(sc, self.P(0.745, 0.90), "rotary: cos(Δθ) at any index, in or out of training",
                   0.025, opacity=op * late, ink=layout.NOTE_INK)
        sc.probe(("gpt2",), (shown * 4, 0, 0), tol=6.0)

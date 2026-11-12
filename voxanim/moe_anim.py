"""Mixture of experts: a router sending each token to two of eight small MLPs.

The layer of :mod:`voxanim.moe`, trained on next-character prediction in a
language with three regimes (digits count up, letters run along the
alphabet, punctuation switches between them), drawn as a flat diagram:

    text  3 4 5 . g h i :  7 ...        the tokens, one at a time
    token -> router -> p over 8 experts (bars) -> the top-2 light up and compute
          -> their outputs, weighted by the gate weights g₁, g₂ -> the next-character guess

Three phases:

1. **routing** -- each token of a sample text in turn: the router's eight
   probabilities as bars, the two largest chosen, the two experts computing
   while six stay idle, the outputs combined by the renormalised gate
   weights, the predicted next character against the real one,
2. **utilisation** -- how often each expert is chosen over 2,048 fresh tokens,
   for the model trained with the load-balancing loss and for one trained
   without it, each bar split by the regime of the tokens it took,
3. **the count** -- every token touches two experts: 2 × (16·16 + 16·16) =
   1,024 multiply-adds, while the layer holds eight experts' parameters,
   8 × (16·16 + 16 + 16·16 + 16) = 4,352: a quarter of the weights per token.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout, moe
from .conv import Tempo
from .diagram import EDGE, NODE, Diagram
from .easing import ease_in_out, ease_out, remap, smoothstep
from .text import TIMES

REGIME_INKS = (C.rgb("#429696"), C.rgb("#a1814b"), C.rgb("#8952eb"))   # digits, letters, punct
CHOSEN = C.rgb("#3fb2a8")                     # an expert at work
IDLE = C.rgb("#c9cbcd")                       # an expert left idle
TEXT_Y = 0.22                                 # the sample text's row
TEXT_X = (0.08, 0.62)                         # its span
ROUTER_AT = (0.14, 0.56)
BAR_X = (0.235, 0.365)                        # the probability bars grow from BAR_X[0]
EXPERT_X = 0.535
EXPERT_Y = (0.33, 0.82)                       # the eight experts, top to bottom
EXPERT_W, EXPERT_H = 86.0, 20.0               # px
COMBINE_AT = (0.69, 0.56)
OUT_X = 0.77                                  # the prediction's column
UTIL_CHARTS = ((0.10, 0.44), (0.56, 0.90))    # the two utilisation charts' x spans
UTIL_Y = (0.36, 0.76)                         # top and baseline
COUNT_AT = (0.5, 0.56)


class MixtureOfExperts(Diagram):
    """``balanced`` and ``plain`` are :class:`voxanim.moe.MoERun` s (with and
    without the balancing loss); the routing phase shows ``balanced``."""

    PHASES = ("routing", "utilisation", "the count")

    def __init__(self, balanced, plain):
        super().__init__("Mixture of experts", "", 10.0)
        self.hold = 2.5
        self.run, self.plain = balanced, plain
        rng = np.random.default_rng(moe.TEXT_SEED)
        self.ids = moe.sample_text(rng, moe.TEXT_LEN + 1)      # one extra: the last target
        self.f = moe.forward(balanced.P, self.ids[:-1], 0.0)
        self.E = moe.N_EXPERTS
        self.util = {"with": moe.utilisation(balanced.P), "without": moe.utilisation(plain.P)}
        self.compute, self.params = moe.compute_per_token(), moe.parameters()

        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        tok = Tempo(slow=2.2, fast=0.75, ramp_in=2.0, ramp_out=2.0).durations(moe.TEXT_LEN)
        self.k_starts, self.k_durs = t + np.concatenate([[0.0], np.cumsum(tok)[:-1]]), tok
        add("route", float(tok.sum()), 1.2)
        add("util", 8.0, 1.0)
        add("count", 6.0, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["util"][0] - 0.6, Tm["count"][0] - 0.5]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _token(self, t):
        """``(token index, its progress)`` of the routing sweep; -1 before, the last after."""
        if t < self.Tm["route"][0]:
            return -1, 0.0
        if t >= self.Tm["route"][1]:
            return moe.TEXT_LEN - 1, 1.0
        k = int(np.searchsorted(self.k_starts, t, side="right") - 1)
        return k, (t - self.k_starts[k]) / self.k_durs[k]

    # -- placement -----------------------------------------------------------------------
    def expert_at(self, e):
        """World point of expert ``e``'s box."""
        y0, y1 = EXPERT_Y
        return self.P(EXPERT_X, y0 + (y1 - y0) * e / (self.E - 1))

    def token_at(self, k):
        """World point of token ``k`` in the text row."""
        x0, x1 = TEXT_X
        return self.P(x0 + (x1 - x0) * k / (moe.TEXT_LEN - 1), TEXT_Y)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        k, u = self._token(t)
        util = self.progress("util", t)
        count = self.progress("count", t)
        route_op = (1.0 - smoothstep(self.Tm["util"][0] - 1.0, self.Tm["util"][0] - 0.4, t)) * back
        if reset > 0:                      # the reset returns to the opening frame: token 0 due
            k, u, route_op = 0, 0.0, 1.0 - back
        util_op = smoothstep(0.0, 0.08, util) * (1.0 - smoothstep(self.Tm["count"][0] - 0.8,
                                                                   self.Tm["count"][0] - 0.3, t))
        count_op = smoothstep(0.0, 0.1, count) * back
        w, wo = self.util["with"][0], self.util["without"][0]
        self.subtitle = [
            self._route_subtitle(k, u),
            "utilisation over 2,048 tokens: busiest / quietest expert %.1f%s with the balancing "
            "loss, %.1f%s without" % (w.max() / w.min(), TIMES, wo.max() / wo.min(), TIMES),
            "every token: 2 of %d experts, %s multiply-adds, while the layer holds %s expert "
            "parameters" % (self.E, f"{self.compute:,}", f"{self.params:,}"),
        ][ph]
        if route_op > 0.01:
            self._routing(sc, k, u, route_op)
        if util_op * back > 0.01:
            self._utilisation(sc, util, util_op * back)
        if count_op > 0.01:
            self._count(sc, count, count_op)
        self.step_indicator(sc, ph, self.PHASES)

    def _route_subtitle(self, k, u):
        """The routing phase's line: the token, its two experts and gate weights."""
        if k < 0:
            return "a router sends each token to two of eight small MLPs; the rest stay idle"
        f = self.f
        e1, e2 = f["top"][k]
        g1, g2 = f["g"][k]
        return ("token '%s' (%s): experts %d and %d, gate weights %.2f and %.2f; the rest of the "
                "%d idle" % (moe.VOCAB[self.ids[k]], moe.REGIMES[moe.regime_of(self.ids[k])],
                             e1 + 1, e2 + 1, g1, g2, self.E))

    # -- phase 1 --------------------------------------------------------------------------
    def _routing(self, sc, k, u, op):
        """The text, the router, its bars, the experts and the combined output for token ``k``."""
        f = self.f
        # the text row
        for j in range(moe.TEXT_LEN):
            ch = moe.VOCAB[self.ids[j]]
            ink = REGIME_INKS[moe.regime_of(self.ids[j])]
            strong = 1.0 if j == k else (0.55 if j < k else 0.4)
            self.label(sc, self.token_at(j), ch, 0.034, ink=ink, opacity=op * strong, kind="mono")
        self.label(sc, self.P(TEXT_X[0], TEXT_Y - 0.06), "the text, one token at a time",
                   0.027, anchor="lm", opacity=op)
        # the router
        router = self.P(*ROUTER_AT)
        self.rect(sc, router, 74, 30, NODE, op, border=EDGE)
        self.label(sc, router, "router", 0.028, opacity=op, ink=layout.STRONG_INK)
        self.label(sc, self.shift(router, 0, 26), "p = softmax(x W)", 0.025, anchor="mt",
                   opacity=op)
        # the experts
        for e in range(self.E):
            c = self.expert_at(e)
            chosen = k >= 0 and e in f["top"][k] and u >= 0.45
            fill = CHOSEN if chosen else IDLE
            self.rect(sc, c, EXPERT_W, EXPERT_H, fill, op, border=EDGE if not chosen else None)
            self.label(sc, self.shift(c, -EXPERT_W / 2 - 6, 0), "expert %d" % (e + 1), 0.025,
                       anchor="rm", opacity=op)
            if chosen:          # a bar of work crossing the box
                w = EXPERT_W * ease_out(remap(u, 0.5, 0.75))
                self.rect(sc, self.shift(c, -EXPERT_W / 2 + w / 2, 0), w, EXPERT_H, C.lighten(
                    CHOSEN, 0.45), op, lift=0.6, radius=3.0)
        self.label(sc, self.shift(self.expert_at(0), 0, -EXPERT_H / 2 - 14),
                   "8 experts (16 → 16 → 16)", 0.026, anchor="mb", opacity=op)
        # the combine node and the output column
        comb = self.P(*COMBINE_AT)
        self.disk(sc, comb, 15, NODE, op, outline=0.1)
        self.label(sc, comb, "Σ", 0.03, opacity=op, ink=layout.STRONG_INK)
        self.label(sc, self.shift(comb, 0, 22), "g₁ out₁ + g₂ out₂", 0.025, anchor="mt",
                   opacity=op)
        if k < 0:
            return
        e1, e2 = f["top"][k]
        p = f["p"][k]
        # the token flies in
        self.dot(sc, self.token_at(k), router, remap(u, 0.0, 0.2), REGIME_INKS[
            moe.regime_of(self.ids[k])], key=("tok",), lift=1.5)
        # the bars: the router's probabilities
        bars = ease_out(remap(u, 0.18, 0.4))
        x0, x1 = self.P(BAR_X[0], 0.5), self.P(BAR_X[1], 0.5)
        for e in range(self.E):
            c = self.expert_at(e)
            a = x0.copy()
            a[2] = c[2]
            b = x0 + (x1 - x0) * p[e] * bars
            b[2] = c[2]
            ink = CHOSEN if e in (e1, e2) and u >= 0.4 else (0.6, 0.62, 0.64)
            sc.line(a, b, width=7.0, color=ink, opacity=op, bias=0.2)
            if bars > 0.5:
                self.label(sc, b, "%.2f" % p[e], 0.024, anchor="lm", dx=5, opacity=op * bars)
        self.label(sc, self.shift(self.P(BAR_X[0], EXPERT_Y[0]), 0, -EXPERT_H / 2 - 14),
                   "router p", 0.026, anchor="lb", opacity=op)
        # the two chosen: lines from the router, dots to the experts
        for e in (e1, e2):
            self.dot(sc, router, self.expert_at(e), remap(u, 0.4, 0.55), CHOSEN, key=("to", e),
                     lift=1.5)
        # the outputs combine
        g = f["g"][k]
        for e, ge in zip((e1, e2), g):
            self.dot(sc, self.expert_at(e), comb, remap(u, 0.75, 0.9), CHOSEN, r=2.4 + 3.0 * ge,
                     key=("from", e), lift=1.5)
        if u >= 0.75:
            gl = smoothstep(0.75, 0.85, u)
            self.label(sc, self.shift(comb, 0, 42), "g = %.2f, %.2f" % (g[0], g[1]), 0.026,
                       anchor="mt", opacity=op * gl, ink=layout.STRONG_INK)
        # the prediction
        shown = smoothstep(0.88, 0.98, u)
        if shown > 0:
            q = f["q"][k]
            top3 = np.argsort(-q)[:3]
            nxt = self.ids[k + 1]
            base = self.P(OUT_X, 0.44)
            self.label(sc, base, "next character?", 0.026, anchor="lm", opacity=op * shown)
            for n, j in enumerate(top3):
                mark = "   ← it was" if j == nxt else ""
                self.label(sc, self.shift(base, 0, 26 + 22 * n), "'%s'  %.2f%s" % (
                    moe.VOCAB[j], q[j], mark), 0.027, anchor="lm", opacity=op * shown,
                    ink=layout.STRONG_INK if j == nxt else layout.CAPTION_INK)
            if nxt not in top3:
                self.label(sc, self.shift(base, 0, 26 + 22 * 3), "it was '%s' (%.2f)" % (
                    moe.VOCAB[nxt], q[nxt]), 0.027, anchor="lm", opacity=op * shown,
                    ink=layout.STRONG_INK)

    # -- phase 2 --------------------------------------------------------------------------
    def _utilisation(self, sc, util, op):
        """Two charts: load per expert, with and without the balancing loss, split by regime."""
        grow = ease_out(remap(util, 0.05, 0.5))
        y_top, y_base = UTIL_Y
        for (x0, x1), key, title in zip(UTIL_CHARTS, ("with", "without"),
                                         ("with the balancing loss", "without it")):
            fr, by = self.util[key]
            counts = by * fr[None] / np.maximum(by.sum(0), 1e-9)[None]   # fraction by regime
            counts = counts / counts.sum(0, keepdims=True) * fr[None]
            width = (x1 - x0) / self.E
            vmax = 0.30
            base = self.P(x0, y_base)
            end = self.P(x1, y_base)
            sc.line(base, end, width=1.0, color=(0.5, 0.52, 0.54), opacity=0.7 * op, bias=0.0)
            fair = self.P(x0, y_base - (y_base - y_top) * (1 / self.E) / vmax)
            fair2 = self.P(x1, y_base - (y_base - y_top) * (1 / self.E) / vmax)
            for i in range(10):
                a0, a1 = i / 10, (i + 0.5) / 10
                sc.line(fair + (fair2 - fair) * a0, fair + (fair2 - fair) * a1, width=0.9,
                        color=(0.5, 0.52, 0.54), opacity=0.6 * op, bias=0.0)
            self.label(sc, fair2, "1/8", 0.024, anchor="lm", dx=4, opacity=op, ink=layout.NOTE_INK)
            for e in range(self.E):
                cx = x0 + width * (e + 0.5)
                y = y_base
                for r in range(len(moe.REGIMES)):
                    h = (y_base - y_top) * counts[r, e] / vmax * grow
                    c = self.P(cx, y - h / 2)
                    px_w = width * self.cam.width * self.cam.zoom * 0.62
                    px_h = max(h * self.cam.height * self.cam.zoom, 0.0)
                    if px_h > 0.3:
                        self.rect(sc, c, px_w, px_h, REGIME_INKS[r], op, radius=0.0)
                    y -= h
                self.label(sc, self.P(cx, y_base + 0.012), str(e + 1), 0.024, anchor="mt",
                           opacity=op, ink=layout.NOTE_INK)
                if grow > 0.95:          # a row of percentages above the chart
                    self.label(sc, self.P(cx, y_top - 0.025), "%.0f%%" % (100 * fr[e]), 0.024,
                               anchor="mb", opacity=op * smoothstep(0.95, 1.0, grow))
            self.label(sc, self.P((x0 + x1) / 2, y_top - 0.105), title, 0.028, anchor="mm",
                       opacity=op, ink=layout.STRONG_INK)
            self.label(sc, self.P((x0 + x1) / 2, y_base + 0.05), "expert", 0.025, anchor="mt",
                       opacity=op)
        for r, name in enumerate(moe.REGIMES):
            p = self.P(0.36 + 0.11 * r, 0.90)
            self.rect(sc, p, 10, 10, REGIME_INKS[r], op, radius=2.0)
            self.label(sc, p, name, 0.025, anchor="lm", dx=9, opacity=op)

    # -- phase 3 --------------------------------------------------------------------------
    def _count(self, sc, count, op):
        """Eight blocks of parameters, two of them at work, and the arithmetic."""
        lit = smoothstep(0.25, 0.45, count)
        cx, cy = COUNT_AT
        per = self.params // self.E
        for e in range(self.E):
            c = self.P(cx - 0.42 + 0.12 * e, cy - 0.05)
            on = e in self.f["top"][-1]
            fill = CHOSEN if on else IDLE
            self.rect(sc, c, 70, 54, C.mix(IDLE, fill, lit) if on else IDLE, op, border=EDGE,
                      radius=5.0)
            self.label(sc, c, f"{per:,}", 0.027, opacity=op, ink=layout.STRONG_INK)
            self.label(sc, self.shift(c, 0, 14), "params", 0.024, opacity=op, ink=layout.NOTE_INK)
            self.label(sc, self.shift(c, 0, -40), "expert %d" % (e + 1), 0.025, anchor="mb",
                       opacity=op)
        self.label(sc, self.P(cx, cy + 0.10), "parameters held: %d experts %s %s = %s" % (
            self.E, TIMES, f"{per:,}", f"{self.params:,}"), 0.029, anchor="mm", opacity=op,
            ink=layout.STRONG_INK)
        self.label(sc, self.P(cx, cy + 0.17), "multiply-adds per token: 2 experts %s (%d·%d + "
                   "%d·%d) = %s, a quarter of the weights" % (TIMES, moe.D_MODEL, moe.D_HIDDEN,
                                                              moe.D_HIDDEN, moe.D_MODEL,
                                                              f"{self.compute:,}"),
                   0.029, anchor="mm", opacity=op * lit, ink=layout.STRONG_INK)
        self.label(sc, self.P(cx, cy + 0.24), "the layer's capacity grows with the experts; "
                   "the cost per token does not", 0.027, anchor="mm", opacity=op * lit)

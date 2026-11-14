"""The original transformer's sinusoidal position encoding (Vaswani et al., 2017).

Pure maths, drawn with real numbers (:mod:`voxanim.posenc`): the table
``PE[p, 2i] = sin(p ωᵢ)``, ``PE[p, 2i+1] = cos(p ωᵢ)``, ``ωᵢ = 10000^(−2i/d)``,
here with ``d = 64`` (32 pairs), is added to the token embedding,
``x = token + PE[p]``.  Four phases:

1. **waves** -- the sine and cosine columns of pairs 0, 2, 4, 8 and 16 over
   positions 0 .. 63, each pair at its own frequency: pair 0 repeats every
   2π ≈ 6.3 positions, pair 31 every ≈ 47,000 (printed from ``ωᵢ``),
2. **the table** -- the whole 64 × 64 table as a heatmap (rows: positions,
   columns: dimensions; the classic figure), a marker sweeping down the rows
   and the row under it drawn as 64 bars,
3. **a rotation** -- one pair drawn as a point (cos, sin) on a circle at
   position ``p`` and ``p + k``: every step turns it by ``ωᵢ``, so ``k`` steps
   turn it by ``k ωᵢ`` wherever ``p`` is.  All 32 pairs at once:
   ``PE[p + k] = PE[p] · M_k`` with ``M_k`` (:func:`posenc.shift_matrix`) block
   diagonal, drawn as a 64 × 64 heatmap; the largest error of that identity
   is printed for several ``p`` (float rounding),
4. **dot products** -- ``PE[p] · PE[p + k]`` against ``k`` for several ``p``:
   one curve, ``Σᵢ cos(k ωᵢ)``, decaying and wiggling.  Then the caveat: the
   one-layer model trained with this encoding (:func:`posenc.train`, the
   length-extrapolation study) still fails past its training length; its
   mean test loss at positions 2 .. 30 and at 63 .. 126 is read from
   ``posenc.train("sinusoidal").test_loss``.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout, posenc
from .diagram import EDGE, Diagram, draw_image
from .easing import ease_in_out, ease_out, remap, smoothstep
from .netanim import div_colors
from .text import sub

D = 64                               # the encoding's width: 32 (sin, cos) pairs
N = 64                               # positions drawn
WAVE_PAIRS = (0, 2, 4, 8, 16)        # the pairs drawn as waves
DIAL_PAIRS = (2, 6)                  # the pairs drawn as turning points
K = 4                                # the offset of the rotation phase
CHECK_P = (3, 17, 40, 90)            # positions of the numerical check
DOT_P = (10, 37, 80)                 # positions whose dot-product curves are overlaid
N_GAP = 128                          # gaps k of the dot-product curve
NEAR, FAR = (2, 31), (63, 127)       # test-loss positions: within / past the training length
INKS = [C.rgb(h) for h in ("#d84e4e", "#a1814b", "#3d9e45", "#6370eb", "#8952eb")]
SUM_INK = (0.14, 0.16, 0.18)
AXIS = (0.5, 0.52, 0.54)
WAVES = (0.10, 0.62, 0.29, 0.78)     # the waves' box: x0, x1, y0 (top row), y1 (bottom row)
TABLE_AT, TABLE_PX = (0.27, 0.585), 270.0
BARS = (0.58, 0.95, 0.47)            # the row's bar chart: x0, x1, y of its zero line
DIAL_AT = ((0.15, 0.46), (0.37, 0.46))
DIAL_R = 58.0                        # px
MK_AT, MK_PX = (0.76, 0.53), 230.0
PLOT = (0.10, 0.30, 0.66, 0.74)      # the dot-product plot: x0, y0, x1, y1


def grouped(n):
    """``n`` with thousands separators: ``47,117``."""
    return f"{int(round(n)):,}"


def sci(x, digits=1):
    """``x`` as ``8×10⁻¹⁶`` (``digits`` significant digits), ``0`` for zero."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = round(x / 10.0 ** e, digits - 1)
    if abs(m) >= 10:
        m, e = m / 10, e + 1
    sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return "%s×10%s" % (("%.*f" % (digits - 1, m)), str(e).translate(sup))


class SinusoidalPositions(Diagram):
    """The sinusoidal table, its frequencies, the shift as a rotation and the dot products."""

    PHASES = ("waves", "the table", "a rotation", "dot products")

    def __init__(self):
        super().__init__("Sinusoidal position encoding", "", 10.0)
        self.hold = 2.5
        self.w = posenc.frequencies(D)
        self.period = 2 * np.pi / self.w
        self.pe = posenc.sinusoidal(max(CHECK_P) + K + N_GAP + 1, D)
        self.M = posenc.shift_matrix(K, D)
        self.err = [float(np.abs(self.pe[p + K] - self.pe[p] @ self.M).max()) for p in CHECK_P]
        self.dots = np.array([[self.pe[p] @ self.pe[p + k] for k in range(N_GAP)] for p in DOT_P])
        self.formula = np.array([np.cos(k * self.w).sum() for k in range(N_GAP)])
        self.dot_err = float(np.abs(self.dots - self.formula[None]).max())
        run = posenc.train("sinusoidal")
        self.near = float(run.test_loss[slice(*NEAR)].mean())
        self.far = float(run.test_loss[slice(*FAR)].mean())
        self.best = posenc.entropies()[0]

        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("intro", 1.0, 0.0)
        add("waves", 10.0, 1.2)
        add("table", 10.0, 1.2)
        add("turn", 9.0, 0.5)
        add("matrix", 6.5, 1.2)
        add("dots", 8.0, 0.5)
        add("note", 4.5, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["table"][0] - 0.6, Tm["turn"][0] - 0.6, Tm["dots"][0] - 0.6]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _phase_ops(self, t, back):
        """Opacity of each phase's drawing: one at a time, with short crossfades."""
        st = self.marks + [None]
        op = []
        for n in range(len(self.PHASES)):
            a = 1.0 if n == 0 else smoothstep(st[n] + 0.1, st[n] + 0.5, t)
            b = 1.0 if st[n + 1] is None else 1.0 - smoothstep(st[n + 1] - 0.5, st[n + 1], t)
            op.append(a * b * back)
        return op

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ops = self._phase_ops(t, back)
        w, per = self.w, self.period
        self.subtitle = [
            "PE[p, 2i] = sin(p ωᵢ), PE[p, 2i+1] = cos(p ωᵢ), ωᵢ = 10000^(−2i/64): pair 0 "
            "repeats every %.1f positions, pair 31 every %s" % (per[0], grouped(per[-1])),
            "the table, one row per position: fast pairs on the left change at every step, "
            "slow ones on the right barely move (pair 31 turns %.5f rad a step)" % w[-1],
            "moving %d places turns pair i by %d·ωᵢ wherever p is, so PE[p + %d] = PE[p] · M%s "
            "for every p: largest error %s" % (K, K, K, sub(K), sci(max(self.err))),
            "PE[p] · PE[p + k] = Σᵢ cos(k ωᵢ) depends on the gap only: %d at k = 0, %.1f at "
            "k = 16, %.1f at k = 64, the same for every p" % (D // 2, self.formula[16],
                                                             self.formula[64]),
        ][ph]
        if ops[0] > 0.01:
            self._waves(sc, t, ops[0])
        if ops[1] > 0.01:
            self._table(sc, t, ops[1])
        if ops[2] > 0.01:
            self._rotation(sc, t, ops[2])
        if ops[3] > 0.01:
            self._dots(sc, t, ops[3])
        self.step_indicator(sc, ph, self.PHASES)

    # -- phase 1: the waves ----------------------------------------------------------------------
    def _waves(self, sc, t, op):
        """Each drawn pair's sin (solid) and cos (faint) column over positions 0 .. 63."""
        show = smoothstep(0.0, 1.0, self.progress("intro", t))
        wv = self.progress("waves", t)
        x0, x1, ya, yb = WAVES
        p_now = ease_in_out(remap(wv, 0.05, 0.75)) * (N - 1)
        n_rows = len(WAVE_PAIRS)
        dy = (yb - ya) / (n_rows - 1)
        amp = 0.36 * dy

        def pt(p, row, v):
            return self.P(x0 + (x1 - x0) * p / (N - 1), ya + dy * row - amp * v)

        ps = np.linspace(0, p_now, max(2, int(p_now * 4) + 2))
        for row, (i, ink) in enumerate(zip(WAVE_PAIRS, INKS)):
            o = op * show * smoothstep(0.0 + 0.04 * row, 0.08 + 0.04 * row, wv)
            self.polyline(sc, [pt(0, row, 0), pt(N - 1, row, 0)], 1.0, AXIS, 0.6 * o)
            sin = [pt(p, row, np.sin(p * self.w[i])) for p in ps]
            cos = [pt(p, row, np.cos(p * self.w[i])) for p in ps]
            self.polyline(sc, cos, 1.2, C.lighten(ink, 0.45), 0.8 * o)
            self.polyline(sc, sin, 2.0, ink, o)
            self.disk(sc, sin[-1], 3.2, ink, o, lift=1.0, outline=0.0)
            self.label(sc, self.P(x1 + 0.03, ya + dy * row - 0.018), "pair %d: dims %d, %d" % (
                i, 2 * i, 2 * i + 1), 0.026, anchor="lm", opacity=o,
                ink=tuple(float(v) for v in ink))
            self.label(sc, self.P(x1 + 0.03, ya + dy * row + 0.022), "ω = %.3g, period %s" % (
                self.w[i], ("%.1f" % self.period[i]) if self.period[i] < 100
                else grouped(self.period[i])), 0.026, anchor="lm", opacity=o)
        o = op * show
        cur = self.P(x0 + (x1 - x0) * p_now / (N - 1), ya - 0.06)
        self.polyline(sc, [cur, self.P(x0 + (x1 - x0) * p_now / (N - 1), yb + 0.05)], 1.0,
                      EDGE, 0.5 * o)
        self.label(sc, self.shift(cur, 0, -4), "p = %d" % round(p_now), 0.026, anchor="mb",
                   opacity=o, ink=layout.STRONG_INK)
        for p in (0, 16, 32, 48, 63):
            self.label(sc, self.P(x0 + (x1 - x0) * p / (N - 1), yb + 0.065), str(p), 0.024,
                       anchor="mt", opacity=o, ink=layout.NOTE_INK)
        self.label(sc, self.P((x0 + x1) / 2, yb + 0.11), "position p  ·  sin solid, cos faint",
                   0.026, anchor="mt", opacity=o)
        late = smoothstep(0.78, 0.9, wv)
        self.label(sc, self.P(x1 + 0.03, yb + 0.075), "… pair 31: ω = %s," % sci(self.w[-1], 2),
                   0.026, anchor="lm", opacity=o * late, ink=layout.NOTE_INK)
        self.label(sc, self.P(x1 + 0.03, yb + 0.115), "period %s positions" % grouped(
            self.period[-1]), 0.026, anchor="lm", opacity=o * late, ink=layout.NOTE_INK)
        sc.probe(("wave",), cur, tol=6.0)

    # -- phase 2: the table ----------------------------------------------------------------------
    def _table(self, sc, t, op):
        """The 64 × 64 table, revealed row by row, and the row under the marker as bars."""
        tb = self.progress("table", t)
        rows = ease_in_out(remap(tb, 0.05, 0.8)) * (N - 1)
        r = int(round(rows))
        tex = div_colors(self.pe[:N].ravel(), 1.0).reshape(N, N, 3)
        tex[r + 1:] = C.BACKGROUND
        c = self.P(*TABLE_AT)
        h = TABLE_PX / 2
        draw_image(self, sc, c, tex, TABLE_PX, TABLE_PX, op, grid=0.0)
        self.label(sc, self.shift(c, 0, -h - 10), "dimension 0 → 63", 0.026, anchor="mb",
                   opacity=op)
        self.label(sc, self.shift(c, 0, h + 10), "rows: positions 0 → 63, top to bottom", 0.026,
                   anchor="mt", opacity=op)
        self.label(sc, self.shift(c, 0, h + 32), "orange +1, blue −1, grey 0", 0.026,
                   anchor="mt", opacity=op, ink=layout.NOTE_INK)
        # the marker
        y = (rows + 0.5) / N * TABLE_PX - h
        a, b = self.shift(c, -h - 4, y), self.shift(c, h + 4, y)
        sc.line(a, b, width=1.8, color=SUM_INK, opacity=op, bias=0.6)
        self.label(sc, self.shift(c, -h - 8, y), "row %d" % r, 0.026, anchor="rm", opacity=op,
                   ink=layout.STRONG_INK)
        sc.probe(("row",), a, tol=6.0)
        # the row as bars
        x0, x1, yz = BARS
        o = op * smoothstep(0.02, 0.1, tb)
        self.label(sc, self.P(x0, yz - 0.16), "PE[%d]: the 64 numbers for position %d" % (r, r),
                   0.027, anchor="lm", opacity=o, ink=layout.STRONG_INK)
        self.polyline(sc, [self.P(x0, yz), self.P(x1, yz)], 1.0, AXIS, 0.7 * o)
        z = self.P(x0, yz)
        bw = (self.px(self.P(x1, yz))[0] - self.px(z)[0]) / D
        for j in range(D):
            v = self.pe[r, j]
            ink = C.POSITIVE if v >= 0 else C.NEGATIVE
            base = self.shift(z, (j + 0.5) * bw, 0)
            sc.line(base, self.shift(base, 0, -v * 52), width=bw * 0.7, color=ink, opacity=o,
                    bias=0.3)
        for j in (0, 32, 63):
            self.label(sc, self.shift(z, (j + 0.5) * bw, 62), str(j), 0.024, anchor="mt",
                       opacity=o, ink=layout.NOTE_INK)
        notes = ("left: pair 0, 1 rad a step",
                 "right: pair 31, %s rad a step" % sci(self.w[-1], 2),
                 "every row different: a vector per position,",
                 "added to the word's: x = token + PE[p]")
        for n, text in enumerate(notes):
            on = o * smoothstep(0.25 + 0.12 * n, 0.35 + 0.12 * n, tb)
            self.label(sc, self.P(x0, yz + 0.19 + 0.045 * n + (0.02 if n > 1 else 0)), text,
                       0.026, anchor="lm", opacity=on,
                       ink=layout.CAPTION_INK if n < 2 else layout.STRONG_INK)

    # -- phase 3: the rotation -------------------------------------------------------------------
    def _rotation(self, sc, t, op):
        """Two pairs as points on circles at p and p + k, then M_k and the check."""
        tu = self.progress("turn", t)
        mx = self.progress("matrix", t)
        p = ease_in_out(remap(tu, 0.05, 0.9)) * 30
        for (i, at), ink in zip(zip(DIAL_PAIRS, DIAL_AT), (INKS[1], INKS[3])):
            c = self.P(*at)
            w = self.w[i]
            ring = [self.shift(c, DIAL_R * np.cos(a), -DIAL_R * np.sin(a))
                    for a in np.linspace(0, 2 * np.pi, 61)]
            self.polyline(sc, ring, 1.0, EDGE, 0.6 * op)
            self.polyline(sc, [self.shift(c, -DIAL_R - 6, 0), self.shift(c, DIAL_R + 6, 0)],
                          0.8, AXIS, 0.5 * op)
            self.polyline(sc, [self.shift(c, 0, -DIAL_R - 6), self.shift(c, 0, DIAL_R + 6)],
                          0.8, AXIS, 0.5 * op)
            a0, a1 = p * w, (p + K) * w
            arc = [self.shift(c, (DIAL_R + 9) * np.cos(a), -(DIAL_R + 9) * np.sin(a))
                   for a in np.linspace(a0, a1, 24)]
            self.polyline(sc, arc, 2.2, ink, op)
            for a, fill, r in ((a0, ink, 5.0), (a1, C.lighten(ink, 0.55), 5.0)):
                q = self.shift(c, DIAL_R * np.cos(a), -DIAL_R * np.sin(a))
                sc.line(c, q, width=1.4, color=ink, opacity=0.7 * op, bias=0.4)
                self.disk(sc, q, r, fill, op, lift=1.0, outline=0.3)
            self.label(sc, self.shift(c, 0, -DIAL_R - 24), "pair %d: ω = %.3f" % (i, w), 0.027,
                       anchor="mb", opacity=op, ink=layout.STRONG_INK)
            self.label(sc, self.shift(c, 0, DIAL_R + 22), "%d steps: %.2f rad" % (K, K * w),
                       0.026, anchor="mt", opacity=op, ink=tuple(float(v) for v in ink))
            sc.probe(("dial", i), self.shift(c, DIAL_R * np.cos(a0), -DIAL_R * np.sin(a0)),
                     tol=6.0)
        mid = (DIAL_AT[0][0] + DIAL_AT[1][0]) / 2
        self.label(sc, self.P(mid, 0.71), "point = (cos pω, sin pω) at p = %d (dark) and "
                   "p + %d = %d (light)" % (round(p), K, round(p) + K), 0.026, opacity=op)
        # M_k and the check
        o = op * smoothstep(0.0, 0.15, mx)
        c = self.P(*MK_AT)
        reveal = ease_out(remap(mx, 0.0, 0.4))
        tex = div_colors(self.M.ravel(), 1.0).reshape(D, D, 3)
        tex[int(reveal * D):] = C.BACKGROUND
        draw_image(self, sc, c, tex, MK_PX, MK_PX, o, grid=0.0)
        self.label(sc, self.shift(c, 0, -MK_PX / 2 - 10), "M%s, 64 × 64: PE[p + %d] = PE[p]·M%s"
                   % (sub(K), K, sub(K)), 0.027, anchor="mb", opacity=o, ink=layout.STRONG_INK)
        b0 = self.M[:2, :2]
        self.label(sc, self.shift(c, 0, MK_PX / 2 + 10), "32 blocks of 2 × 2 on the diagonal, "
                   "0 elsewhere", 0.026, anchor="mt", opacity=o)
        self.label(sc, self.shift(c, 0, MK_PX / 2 + 54), "slow pairs hardly turn: their blocks "
                   "are nearly 1, 0, 0, 1", 0.026, anchor="mt", opacity=o, ink=layout.NOTE_INK)
        self.label(sc, self.shift(c, 0, MK_PX / 2 + 32), "block 0: [[%.2f, %.2f], [%.2f, %.2f]]"
                   % tuple(b0.ravel()), 0.026, anchor="mt", opacity=o, ink=layout.NOTE_INK)
        self.label(sc, self.P(0.05, 0.78), "max |PE[p + %d] − PE[p]·M%s| over the 64 numbers:"
                   % (K, sub(K)), 0.026, anchor="lm", opacity=o * smoothstep(0.35, 0.45, mx))
        for n, (pp, e) in enumerate(zip(CHECK_P, self.err)):
            on = o * smoothstep(0.42 + 0.1 * n, 0.52 + 0.1 * n, mx)
            x = 0.05 + 0.13 * n
            self.label(sc, self.P(x, 0.835), "p = %d" % pp, 0.026, anchor="lm", opacity=on,
                       ink=layout.NOTE_INK)
            self.label(sc, self.P(x, 0.88), sci(e), 0.027, anchor="lm", opacity=on,
                       ink=layout.STRONG_INK)

    # -- phase 4: dot products -------------------------------------------------------------------
    def _dots(self, sc, t, op):
        """PE[p] · PE[p + k] against k: one curve for every p; then the trained model's note."""
        dt = self.progress("dots", t)
        nt = self.progress("note", t)
        x0, y0, x1, y1 = PLOT
        top = D / 2

        def pt(k, v):
            return self.P(x0 + (x1 - x0) * k / (N_GAP - 1), y1 - (y1 - y0) * v / top)

        self.polyline(sc, [self.P(x0, y1), self.P(x1, y1)], 1.0, AXIS, 0.7 * op)
        self.polyline(sc, [self.P(x0, y0), self.P(x0, y1)], 1.0, AXIS, 0.7 * op)
        for k in (0, 32, 64, 96, 127):
            self.label(sc, self.P(x0 + (x1 - x0) * k / (N_GAP - 1), y1 + 0.015), str(k), 0.024,
                       anchor="mt", opacity=op, ink=layout.NOTE_INK)
        for v in (16, 32):
            self.label(sc, self.shift(pt(0, v), -8, 0), str(v), 0.024, anchor="rm", opacity=op,
                       ink=layout.NOTE_INK)
        self.label(sc, self.P((x0 + x1) / 2, y1 + 0.05), "gap k", 0.026, anchor="mt", opacity=op)
        self.label(sc, self.P(x0, y0 - 0.03), "PE[p] · PE[p + k]", 0.026, anchor="lb",
                   opacity=op)
        n_show = max(2, int(ease_out(remap(dt, 0.05, 0.7)) * (N_GAP - 1)) + 1)
        self.polyline(sc, [pt(k, self.formula[k]) for k in range(n_show)], 2.0, SUM_INK, op)
        for n, (pp, ink) in enumerate(zip(DOT_P, INKS[:3])):
            on = op * smoothstep(0.15 + 0.12 * n, 0.25 + 0.12 * n, dt)
            for k in range(n * 3, n_show, 9):
                self.disk(sc, pt(k, self.dots[n, k]), 3.0, ink, on, lift=1.0, outline=0.0)
            self.label(sc, self.P(0.71, 0.36 + 0.05 * n), "• p = %d" % pp, 0.026, anchor="lm",
                       opacity=on, ink=tuple(float(v) for v in ink))
        self.label(sc, self.P(0.71, 0.51), "— Σᵢ cos(k ωᵢ)", 0.026, anchor="lm", opacity=op,
                   ink=SUM_INK)
        late = smoothstep(0.55, 0.7, dt)
        self.label(sc, self.P(0.71, 0.58), "the dots sit on the line:", 0.026, anchor="lm",
                   opacity=op * late)
        self.label(sc, self.P(0.71, 0.62), "largest difference %s" % sci(self.dot_err), 0.026,
                   anchor="lm", opacity=op * late, ink=layout.STRONG_INK)
        self.label(sc, self.P(0.71, 0.68), "near positions alike,", 0.026, anchor="lm",
                   opacity=op * late)
        self.label(sc, self.P(0.71, 0.72), "far ones less, on the whole", 0.026, anchor="lm",
                   opacity=op * late)
        on = op * smoothstep(0.0, 0.2, nt)
        self.label(sc, self.P(0.5, 0.86), "a one-layer model trained with it on %d-token "
                   "sequences still fails beyond them:" % posenc.TRAIN_LEN, 0.027, opacity=on,
                   ink=layout.STRONG_INK)
        self.label(sc, self.P(0.5, 0.905), "test loss %.2f at positions %d–%d, %.2f at %d–%d "
                   "(the best possible is %.2f)" % (self.near, NEAR[0], NEAR[1] - 1, self.far,
                                                   FAR[0], FAR[1] - 1, self.best), 0.027,
                   opacity=on)
        sc.probe(("dots",), pt(n_show - 1, self.formula[n_show - 1]), tol=6.0)


"""RoPE past the length it was trained on: what goes wrong, and two fixes.

The model is :mod:`voxanim.posenc`'s RoPE transformer (one layer, 4 heads of
width 8, so 4 rotary pairs per head, ``θ_i = 10000^(−2i/8)`` = 1, 0.1, 0.01,
0.001), trained on sequences of 32 symbols of an order-2 Markov language and
tested, with no retraining, on sequences of 128.  RoPE's score between a
query and a key ``Δ`` apart depends on the angles ``Δ θ_i`` only (the
``gpt2-rotary-embeddings`` animation shows why); training only ever showed
it ``Δ < 32``.  Four phases:

1. **angles** -- one dial per pair: the arc of angles ``Δ θ_i`` training
   covered (``Δ ≤ 31``) and, as ``Δ`` runs to 127, the arc only the long test
   reaches.  The fast pair has turned 4.9 times in training and meets no new
   angle; the slow pairs are pushed past anything they saw,
2. **the failure** -- the RoPE model's attention (the head with the sharpest
   previous-token pattern) on one sequence of 128 (:func:`posenc.attention`):
   inside 32 a sharp previous-token diagonal; from position 39 the strongest
   key is more than 31 back, and past 64 the weight sits on a stripe of
   tokens about 51 back (both measured from the attention, not assumed).  The
   per-position test loss (512 sequences, ``Run.test_loss``) jumps with it,
3. **interpolation** -- every position scaled by 32/128 (Chen et al., 2023):
   all angles are back in range, but neighbours are now a quarter step apart.
   The previous token's angle in the fast pair drops from 1 rad to 0.25, and
   offset 4 now sits where offset 1 used to.  The previous-token weight
   smears out and the loss is worse everywhere, inside 32 too,
4. **NTK scaling** -- the base raised to ``10000 · 4^(8/6)``
   (:func:`posenc.ntk_base`): the fast pair unchanged, the slowest slowed by
   4.  A table gives each pair's reach at 128 against its reach in training;
   the loss holds to about 2× and fails at 4×.  The three loss curves are
   drawn against the best possible loss (:func:`posenc.entropies`).

Every number comes from :func:`posenc.train`, :func:`posenc.rope_extensions`
(cached here: it evaluates 512 sequences twice), :func:`posenc.attention` and
:func:`posenc.entropies`.
"""

from __future__ import annotations

import hashlib

import numpy as np

from . import color as C
from . import layout, posenc
from .cache import cached_npz
from .diagram import EDGE, Diagram, draw_image
from .easing import ease_in_out, ease_out, remap, smoothstep

SCALE = posenc.TEST_LEN / posenc.TRAIN_LEN           # 4: how much longer the test is
N_PAIRS = posenc.DH // 2                             # 4 rotary pairs per head
MAX_TRAIN_OFF = posenc.TRAIN_LEN - 1                 # 31: the largest offset training saw
MAX_TEST_OFF = posenc.TEST_LEN - 1                   # 127
# test_loss[i] is the loss of predicting symbol i + 1: windows by that symbol's position
WINDOWS = ((2, 31), (32, 63), (64, 127))
SEQ_SEED = 5                                         # the sequence whose attention is drawn
TRAIN_INK, NEW_INK = C.rgb("#3d9e45"), C.rgb("#d84e4e")
PLAIN_INK, INTERP_INK, NTK_INK = C.rgb("#d84e4e"), C.rgb("#a1814b"), C.rgb("#6370eb")
BEST_INK = (0.14, 0.16, 0.18)
AXIS = (0.5, 0.52, 0.54)
SHADE = C.rgb("#e9d6cf")
DIAL_X, DIAL_Y, DIAL_R = (0.16, 0.39, 0.62, 0.85), 0.47, 50.0
PLOT = (0.61, 0.33, 0.95, 0.76)                      # the loss plot: x0, y0, x1, y1 (layout)
MATRIX_AT, MATRIX_PX = (0.26, 0.575), 250.0
RULER = (0.07, 0.49, 0.37)                           # x0, x1, y of the two rulers
SMALL_DIAL_AT, SMALL_DIAL_R = (0.14, 0.73), 50.0
HEAT_LO, HEAT_HI = C.BACKGROUND, C.rgb("#1d2024")     # attention weight 0 and 1


def window_means(curve):
    """Mean of a per-position loss ``(TEST_LEN − 1,)`` over each of :data:`WINDOWS`."""
    return tuple(float(curve[a - 1:b].mean()) for a, b in WINDOWS)


def extensions(run, cache_dir=None):
    """:func:`posenc.rope_extensions`, cached (keyed by the trained weights)."""
    h = hashlib.sha256()
    for k in sorted(run.P):
        h.update(np.ascontiguousarray(run.P[k]).tobytes())
    settings = dict(weights=h.hexdigest()[:16], train_len=posenc.TRAIN_LEN,
                    test_len=posenc.TEST_LEN, n_eval=posenc.N_EVAL,
                    eval_seed=posenc.EVAL_SEED, v=1)
    d = cached_npz("rope-extensions", settings, lambda: posenc.rope_extensions(run), cache_dir)
    return {k: d[k] for k in ("plain", "interpolation", "ntk")}


def offset_profile(A, rows, max_off):
    """Mean attention weight at each offset ``0 .. max_off`` over query ``rows``."""
    return np.array([np.mean([A[i, i - d] for i in rows if i - d >= 0]) for d in
                     range(max_off + 1)])


class RopeContextExtension(Diagram):
    """The RoPE model of :mod:`voxanim.posenc` at 4× its training length."""

    PHASES = ("angles", "the failure", "interpolation", "NTK scaling")

    def __init__(self):
        super().__init__("RoPE past its training length", "", 10.0)
        self.hold = 2.5
        self.run = posenc.train("rope")
        self.h2, self.h1, _h0 = posenc.entropies()
        self.ext = extensions(self.run)
        self.means = {k: window_means(v) for k, v in self.ext.items()}
        self.theta = posenc.frequencies(posenc.DH)
        self.ntk_b = posenc.ntk_base(SCALE, posenc.DH)
        self.theta_ntk = posenc.frequencies(posenc.DH, self.ntk_b)
        # the attention of one test sequence, plain and interpolated
        x = posenc.sample(1, posenc.TEST_LEN, np.random.default_rng(SEQ_SEED))[0]
        A = posenc.attention(self.run, x)
        A_int = posenc.attention(self.run, x, posenc.Opts(rope_scale=1 / SCALE))
        inside = range(2, posenc.TRAIN_LEN)
        far = range(2 * posenc.TRAIN_LEN, posenc.TEST_LEN)
        prev = [np.mean([A[h, i, i - 1] for i in inside]) for h in range(posenc.HEADS)]
        self.head = int(np.argmax(prev))
        self.A = A[self.head]
        self.prev_in = float(prev[self.head])
        self.prev_far = float(np.mean([self.A[i, i - 1] for i in far]))
        prof = offset_profile(self.A, far, 2 * posenc.TRAIN_LEN)
        self.far_off = int(np.argmax(prof))
        self.far_w = float(prof[self.far_off])
        # the first query past 32 whose strongest key is more than 31 back
        rows = np.arange(posenc.TRAIN_LEN, posenc.TEST_LEN)
        jump = rows - self.A[rows].argmax(1) > MAX_TRAIN_OFF
        self.jump_at = int(rows[np.argmax(jump)])
        prof_i = offset_profile(A_int[self.head], inside, 6)
        self.int_prof = prof_i
        self.prev_int = float(prof_i[1])
        self.int_near = float(prof_i[1:5].sum())

        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        add("angles_in", 1.5, 0.0)
        add("sweep", 11.0, 1.0)
        add("fail", 12.0, 1.0)
        add("interp", 11.5, 1.0)
        add("ntk", 12.0, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["fail"][0] - 0.5, Tm["interp"][0] - 0.5, Tm["ntk"][0] - 0.5]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _phase_ops(self, t, back):
        """Opacity of each phase's drawing: one at a time, with short crossfades;
        the loss plot stays from the second phase on."""
        Tm = self.Tm
        ends = {"angles": Tm["fail"][0] - 0.5, "fail": Tm["interp"][0] - 0.5,
                "interp": Tm["ntk"][0] - 0.5}
        starts = {"fail": Tm["fail"][0] - 0.4, "interp": Tm["interp"][0] - 0.4,
                  "ntk": Tm["ntk"][0] - 0.4}
        op = {}
        op["angles"] = 1.0 - smoothstep(ends["angles"] - 0.6, ends["angles"], t)
        for key in ("fail", "interp"):
            op[key] = smoothstep(starts[key], starts[key] + 0.4, t) * (
                1.0 - smoothstep(ends[key] - 0.6, ends[key], t))
        op["ntk"] = smoothstep(starts["ntk"], starts["ntk"] + 0.4, t)
        op["plot"] = smoothstep(starts["fail"], starts["fail"] + 0.4, t)
        return {k: v * back for k, v in op.items()}

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ops = self._phase_ops(t, back)
        sweep = self.progress("sweep", t)
        delta = ease_in_out(sweep) * MAX_TEST_OFF if reset == 0 else 0.0
        fail = self.progress("fail", t)
        interp = self.progress("interp", t)
        ntk = self.progress("ntk", t)
        th, mp, mi, mn = self.theta, self.means["plain"], self.means["interpolation"], \
            self.means["ntk"]
        d = round(delta)
        self.subtitle = [
            "a query and a key Δ apart meet at the angle Δ·θᵢ in pair i; training saw Δ ≤ %d\n"
            "Δ = %d: pair 0 at %.1f rad, pair 1 at %.2f, pair 2 at %.3f, pair 3 at %.4f"
            % (MAX_TRAIN_OFF, d, d * th[0], d * th[1], d * th[2], d * th[3]),
            "head %d's previous-token weight falls from %.2f to %.2f: from position %d it "
            "looks over %d back, past 64 mostly %d back. Loss %.2f inside 32, %.2f at 32–63, "
            "%.2f at 64–127" % (self.head, self.prev_in, self.prev_far, self.jump_at,
                                MAX_TRAIN_OFF, self.far_off, mp[0], mp[1], mp[2]),
            "interpolation: positions × %d/%d, angles back in range (%.2f× training's), but "
            "the previous token is now %.2f rad away in pair 0, not %.0f. Loss %.2f inside 32, "
            "%.2f at 64–127" % (posenc.TRAIN_LEN, posenc.TEST_LEN,
                                MAX_TEST_OFF / SCALE / MAX_TRAIN_OFF, th[0] / SCALE, th[0],
                                mi[0], mi[2]),
            "NTK-aware: base %d instead of 10000; pair 0 unchanged, pair 3 slowed %.0f×\nLoss %.2f "
            "inside 32, %.2f at 32–63 (plain %.2f), but %.2f at 64–127"
            % (round(self.ntk_b), th[3] / self.theta_ntk[3], mn[0], mn[1], mp[1], mn[2]),
        ][ph]
        if ops["angles"] > 0.01:
            self._angles(sc, t, delta, ops["angles"])
        if ops["fail"] > 0.01:
            self._failure(sc, fail, ops["fail"])
        if ops["interp"] > 0.01:
            self._interp(sc, interp, ops["interp"])
        if ops["ntk"] > 0.01:
            self._ntk(sc, ntk, ops["ntk"])
        if ops["plot"] > 0.01:
            self._plot(sc, t, ops["plot"])
        self.step_indicator(sc, ph, self.PHASES)

    # -- helpers ------------------------------------------------------------------------------
    def _arc(self, sc, c, r, a0, a1, width, ink, op):
        """An arc of radius ``r`` px from angle ``a0`` to ``a1`` (radians, anticlockwise
        on screen)."""
        if a1 - a0 <= 1e-4 or op <= 0.01:
            return
        n = max(2, int((a1 - a0) / 0.05) + 1)
        pts = [self.shift(c, r * np.cos(a), -r * np.sin(a)) for a in np.linspace(a0, a1, n)]
        self.polyline(sc, pts, width, ink, op, bias=0.5)

    def _dashed(self, sc, a, b, ink, op, width=1.2, dash=6.0, bias=0.3):
        """A dashed line from ``a`` to ``b`` (dash length in px)."""
        n = max(1, int(np.linalg.norm(self.px(b) - self.px(a)) / dash))
        for k in range(0, n, 2):
            sc.line(a + (b - a) * k / n, a + (b - a) * min(k + 1, n) / n, width=width,
                    color=ink, opacity=op, bias=bias)

    def _covered(self, i, theta=None):
        """The largest angle pair ``i`` reached in training (``Δ ≤ 31``)."""
        th = self.theta if theta is None else theta
        return MAX_TRAIN_OFF * th[i]

    # -- phase 1: the angles each pair turns through --------------------------------------------
    def _angles(self, sc, t, delta, op):
        """Four dials: the trained arc in green, the arc only the long test reaches in red,
        and a pointer at ``Δ θ_i``."""
        show = smoothstep(0.0, 0.6, self.progress("angles_in", t)) * op
        tau = 2 * np.pi
        for i, x in enumerate(DIAL_X):
            c = self.P(x, DIAL_Y)
            th = self.theta[i]
            ring = [self.shift(c, DIAL_R * np.cos(a), DIAL_R * np.sin(a))
                    for a in np.linspace(0, tau, 61)]
            self.polyline(sc, ring, 1.0, EDGE, 0.5 * show)
            self.disk(sc, c, 2.0, EDGE, show, lift=0.8, outline=0.0)
            cov = self._covered(i)
            # what has been swept so far: the trained part, then the new part
            self._arc(sc, c, DIAL_R, 0.0, min(min(delta, MAX_TRAIN_OFF) * th, tau), 4.0,
                      TRAIN_INK, show)
            if delta > MAX_TRAIN_OFF and cov < tau:
                self._arc(sc, c, DIAL_R, cov, min(delta * th, tau), 4.0, NEW_INK, show)
            a = delta * th
            new = delta > MAX_TRAIN_OFF and (a % tau) > cov and cov < tau
            ink = NEW_INK if new else TRAIN_INK
            tip = self.shift(c, (DIAL_R - 6) * np.cos(a), -(DIAL_R - 6) * np.sin(a))
            sc.line(c, tip, width=2.2, color=ink, opacity=show, bias=0.6)
            self.disk(sc, tip, 3.2, ink, show, lift=1.0, outline=0.0)
            sc.probe(("tip", i), tip, tol=6.0)
            self.label(sc, self.shift(c, 0, -DIAL_R - 14), "pair %d: θ = %s" % (i, "%.4g" % th),
                       0.027, anchor="mb", opacity=show, ink=layout.STRONG_INK)
            lines = [("training: 0 – %s rad" % _rad(cov), TRAIN_INK),
                     ("at 128: 0 – %s rad" % _rad(MAX_TEST_OFF * th), NEW_INK)]
            if cov >= tau:
                lines.append(("%.1f turns: all round" % (cov / tau), layout.NOTE_INK))
            else:
                lines.append(("new: %s – %s rad" % (_rad(cov), _rad(min(MAX_TEST_OFF * th, tau))),
                              layout.NOTE_INK))
            for n, (text, ink) in enumerate(lines):
                reveal = 1.0 if n == 0 else smoothstep(MAX_TRAIN_OFF, MAX_TRAIN_OFF + 6, delta)
                self.label(sc, self.shift(c, 0, DIAL_R + 14 + 22 * n), text, 0.026,
                           anchor="mt", opacity=show * reveal,
                           ink=tuple(float(v) for v in ink))
        self.label(sc, self.P(0.06, 0.81), "Δ = %d" % round(delta), 0.031, anchor="lm",
                   opacity=show, ink=layout.STRONG_INK)
        self.label(sc, self.P(0.17, 0.81), "green: angles training reached (Δ ≤ %d); red: "
                   "angles only Δ up to %d reaches" % (MAX_TRAIN_OFF, MAX_TEST_OFF), 0.026,
                   anchor="lm", opacity=show, ink=layout.CAPTION_INK)
        self.label(sc, self.P(0.17, 0.86), "θᵢ = 10000^(−2i/8): heads of width 8 have 4 "
                   "pairs", 0.025, anchor="lm", opacity=show, ink=layout.NOTE_INK)

    # -- the loss plot (phases 2-4) ----------------------------------------------------------------
    def _plot(self, sc, t, op):
        """Per-position test loss of the plain, interpolated and NTK-scaled model."""
        x0, y0, x1, y1 = PLOT
        n = posenc.TEST_LEN - 1
        curves = [self.ext[k] for k in ("plain", "interpolation", "ntk")]
        ymax = float(np.ceil(max(float(c.max()) for c in curves) * 2) / 2) + 0.5
        ymin = 1.0

        def pt(pos, v):
            return self.P(x0 + (x1 - x0) * (pos - 1) / (n - 1),
                          y1 - (y1 - y0) * (np.clip(v, ymin, ymax) - ymin) / (ymax - ymin))

        # the region no training sequence reached
        A, B = pt(posenc.TRAIN_LEN - 0.5, ymax), pt(n, ymin)
        a, b = self.px(A), self.px(B)
        self.rect(sc, (A + B) / 2, b[0] - a[0], b[1] - a[1], SHADE, 0.55 * op, radius=0.0,
                  lift=-0.5)
        self.label(sc, pt((posenc.TRAIN_LEN + n) / 2, ymax), "never trained on", 0.025,
                   anchor="mt", dy=4, opacity=op, ink=layout.NOTE_INK)
        self.polyline(sc, [self.P(x0, y1), self.P(x1, y1)], 1.0, AXIS, 0.7 * op)
        self.polyline(sc, [self.P(x0, y0), self.P(x0, y1)], 1.0, AXIS, 0.7 * op)
        for v in np.arange(ymin, ymax - 0.25, 1.0):
            self.label(sc, pt(1, v), "%.0f" % v, 0.024, anchor="rm", dx=-6, opacity=op,
                       ink=layout.NOTE_INK)
        for pos in (32, 64, 96, 127):
            self.label(sc, pt(pos, ymin), str(pos), 0.024, anchor="mt", dy=5, opacity=op,
                       ink=layout.NOTE_INK)
        self.label(sc, self.P((x0 + x1) / 2, y1 + 0.05), "position of the predicted symbol",
                   0.026, anchor="mt", opacity=op)
        self.label(sc, self.P(x0, y0 - 0.02), "test loss per position, nats (%d sequences)"
                   % posenc.N_EVAL, 0.026, anchor="lb", opacity=op)
        self._dashed(sc, pt(1, self.h2), pt(n, self.h2), BEST_INK, 0.8 * op)
        self.label(sc, pt(n, self.h2), "best possible %.2f" % self.h2, 0.024, anchor="rb",
                   dy=-3, opacity=op, ink=BEST_INK)
        # the current-symbol-only line is labelled in the legend: curves cross it everywhere
        ink = layout.NOTE_INK
        self._dashed(sc, pt(1, self.h1), pt(n, self.h1), ink, 0.8 * op)
        self._dashed(sc, self.P(x0, y1 + 0.16), self.P(x0 + 0.025, y1 + 0.16), ink, op,
                     dash=4.0)
        self.label(sc, self.P(x0 + 0.032, y1 + 0.16), "knowing only the current symbol: %.2f"
                   % self.h1, 0.025, anchor="lm", opacity=op, ink=ink)
        # the curves: plain revealed in phase 2, the others in 3 and 4
        reveal = (ease_out(remap(self.progress("fail", t), 0.05, 0.6)),
                  ease_out(remap(self.progress("interp", t), 0.35, 0.8)),
                  ease_out(remap(self.progress("ntk", t), 0.35, 0.8)))
        inks = (PLAIN_INK, INTERP_INK, NTK_INK)
        names = ("plain RoPE", "interpolation", "NTK base")
        legend_x = x0
        for curve, r, ink, name in zip(curves, reveal, inks, names):
            if r <= 0:
                continue
            k = max(2, int(r * n))
            self.polyline(sc, [pt(p + 1, v) for p, v in enumerate(curve[:k])], 1.5, ink, op,
                          bias=0.6)
            self.disk(sc, pt(k, curve[k - 1]), 2.6, ink, op, lift=1.0, outline=0.0)
            sc.probe(("curve", name), pt(k, curve[k - 1]), tol=6.0)
            self.polyline(sc, [self.P(legend_x, y1 + 0.115), self.P(legend_x + 0.025, y1 + 0.115)],
                          2.0, ink, op * smoothstep(0.0, 0.1, r))
            self.label(sc, self.P(legend_x + 0.032, y1 + 0.115), name, 0.025, anchor="lm",
                       opacity=op * smoothstep(0.0, 0.1, r), ink=tuple(float(c) for c in ink))
            legend_x += 0.035 + layout.text_width(name, 0.025) / self.cam.zoom + 0.02

    # -- phase 2: the failure ----------------------------------------------------------------------
    def _failure(self, sc, fail, op):
        """The attention of one head on a sequence of 128, revealed row by row."""
        n = posenc.TEST_LEN
        reveal = ease_out(remap(fail, 0.05, 0.6))
        shown = max(1, int(reveal * n))
        w = np.sqrt(self.A)[..., None]
        tex = (HEAT_LO + (HEAT_HI - HEAT_LO) * w).astype(np.float32)
        tex[np.triu(np.ones((n, n), bool), 1)] = C.BACKGROUND
        tex[shown:] = C.BACKGROUND
        c = self.P(*MATRIX_AT)
        W = MATRIX_PX
        draw_image(self, sc, c, tex, W, W, op, grid=0.0, lift=0.0)

        def cell(i, j, lift=0.0):
            return self.shift(c, ((j + 0.5) / n - 0.5) * W, ((i + 0.5) / n - 0.5) * W) + \
                self.view * lift

        self.label(sc, self.shift(c, 0, -W / 2 - 10), "head %d's attention on one sequence of "
                   "%d" % (self.head, n), 0.027, anchor="mb", opacity=op, ink=layout.STRONG_INK)
        self.label(sc, self.shift(c, 0, W / 2 + 8), "row: the position reading; column: what it "
                   "attends to", 0.025, anchor="mt", opacity=op)
        self.label(sc, self.shift(c, 0, W / 2 + 30), "darker: more weight (shaded by √weight)",
                   0.025, anchor="mt",
                   opacity=op, ink=layout.NOTE_INK)
        # the training length, across the matrix
        L = posenc.TRAIN_LEN
        mark = smoothstep(0.25, 0.35, fail) * op
        a, b = cell(L - 0.5, -0.5, 1.0), cell(L - 0.5, L + 2, 1.0)
        self._dashed(sc, a, b, NEW_INK, mark, width=1.3)
        self.label(sc, b, "row %d: the training length" % L, 0.025, anchor="lm", dx=4,
                   opacity=mark, ink=tuple(float(v) for v in NEW_INK))
        late = smoothstep(0.62, 0.72, fail) * op
        self.label(sc, cell(6, 18), "previous token: %.2f" % self.prev_in, 0.025, anchor="lm",
                   opacity=smoothstep(0.2, 0.3, fail) * op, ink=layout.STRONG_INK)
        r = self.jump_at + 8
        at = cell(r, r + 16)
        self.label(sc, at, "from row %d: over %d back;" % (self.jump_at, MAX_TRAIN_OFF), 0.025,
                   anchor="lm", opacity=late, ink=layout.STRONG_INK)
        self.label(sc, at, "past 64: %d back, %.2f" % (self.far_off, self.far_w), 0.025,
                   anchor="lm", dy=20, opacity=late, ink=layout.STRONG_INK)
        # a pointer from the note to the stripe
        r2 = r + 30
        sc.line(self.shift(at, -4, 28) + self.view, cell(r2, r2 - self.far_off + 3, 1.0),
                width=1.2, color=NEW_INK, opacity=late)
        sc.probe(("rows",), (shown * 3, 0, 0), tol=6.0)

    # -- phase 3: position interpolation ----------------------------------------------------------
    def _interp(self, sc, prog, op):
        """Two rulers (positions squeezed by 4) and the fast pair's dial: offset 1 at 1 rad in
        training, at 0.25 rad now."""
        xa, xb, y = RULER
        yb = y + 0.13
        n = posenc.TEST_LEN
        s = 1 / SCALE

        def rx(p):
            return xa + (xb - xa) * p / (n - 1)

        squeeze = ease_in_out(remap(prog, 0.05, 0.35))
        for yy in (y, yb):
            self.polyline(sc, [self.P(xa, yy), self.P(xb, yy)], 1.2, AXIS, op)
        self.polyline(sc, [self.P(xa, yb), self.P(rx(MAX_TRAIN_OFF), yb)], 4.0, TRAIN_INK, op)
        self.label(sc, self.P(xa, y - 0.06), "test positions 0 .. %d" % (n - 1), 0.026,
                   anchor="lb", opacity=op)
        self.label(sc, self.P(rx(40), yb + 0.02), "RoPE now turns by position × %d/%d" % (
            posenc.TRAIN_LEN, n), 0.026, anchor="lt", opacity=op)
        self.label(sc, self.P(xa, yb + 0.02), "trained: 0 .. %d" % MAX_TRAIN_OFF, 0.026,
                   anchor="lt", opacity=op, ink=tuple(float(v) for v in TRAIN_INK))
        for p in range(0, n, 16):
            top = self.P(rx(p), y)
            bot = self.P(rx(p * (1 - squeeze + squeeze * s)), yb)
            sc.line(top, bot, width=1.0, color=EDGE, opacity=0.6 * op, bias=0.2)
            self.disk(sc, top, 2.4, EDGE, op, lift=0.8, outline=0.0)
            self.disk(sc, bot, 2.4, INTERP_INK, op, lift=0.8, outline=0.0)
            sc.probe(("squeeze", p), bot, tol=6.0)
        for p in (0, 32, 64, 96, 127):
            self.label(sc, self.P(rx(p), y), str(p), 0.024, anchor="mb", dy=-4, opacity=op,
                       ink=layout.NOTE_INK)
        # the fast pair's dial
        dial = smoothstep(0.4, 0.5, prog) * op
        c = self.P(*SMALL_DIAL_AT)
        R = SMALL_DIAL_R
        ring = [self.shift(c, R * np.cos(a), R * np.sin(a)) for a in np.linspace(0, 2 * np.pi, 61)]
        self.polyline(sc, ring, 1.0, EDGE, 0.5 * dial)
        self.disk(sc, c, 2.0, EDGE, dial, lift=0.8, outline=0.0)
        th = self.theta[0]
        for off in range(5):
            a = off * th
            tip = self.shift(c, R * np.cos(a), -R * np.sin(a))
            sc.line(c, tip, width=1.4, color=TRAIN_INK, opacity=0.8 * dial, bias=0.5)
            self.label(sc, self.shift(c, (R + 10) * np.cos(a), -(R + 10) * np.sin(a)), str(off),
                       0.024, opacity=dial, ink=tuple(float(v) for v in TRAIN_INK))
        new = smoothstep(0.55, 0.65, prog) * op
        a = th * s
        tip = self.shift(c, R * 1.25 * np.cos(a), -R * 1.25 * np.sin(a))
        sc.line(c, tip, width=2.2, color=NEW_INK, opacity=new, bias=0.6)
        self.disk(sc, tip, 3.0, NEW_INK, new, lift=1.0, outline=0.0)
        self.label(sc, tip, "1 now", 0.025, anchor="lm", dx=6, opacity=new,
                   ink=tuple(float(v) for v in NEW_INK))
        tx = 0.27
        self.label(sc, self.P(tx, 0.64), "pair 0 (θ = 1), offsets 0 – 4:", 0.025, anchor="lm",
                   opacity=dial, ink=layout.STRONG_INK)
        self.label(sc, self.P(tx, 0.68), "trained at 0, 1, 2, 3, 4 rad", 0.025,
                   anchor="lm", opacity=dial, ink=tuple(float(v) for v in TRAIN_INK))
        self.label(sc, self.P(tx, 0.72), "now offset 1 sits at %.2f rad," % (th * s), 0.025,
                   anchor="lm", opacity=new, ink=tuple(float(v) for v in NEW_INK))
        self.label(sc, self.P(tx, 0.76), "offset %d where 1 used to be" % SCALE, 0.025,
                   anchor="lm", opacity=new, ink=tuple(float(v) for v in NEW_INK))
        last = smoothstep(0.8, 0.9, prog) * op
        self.label(sc, self.P(0.05, 0.875), "head %d, previous token: %.2f before, %.2f now;" % (
            self.head, self.prev_in, self.prev_int), 0.025, anchor="lm", opacity=last,
            ink=layout.CAPTION_INK)
        self.label(sc, self.P(0.05, 0.925), "%.2f spread over the last four tokens" % (
            self.int_near), 0.025, anchor="lm", opacity=last, ink=layout.CAPTION_INK)

    # -- phase 4: NTK-aware scaling ---------------------------------------------------------------
    def _ntk(self, sc, prog, op):
        """A table: each pair's θ before and after, and how far its angles reach at 128 as a
        multiple of what training reached, for the three ways of running the model."""
        cols = (0.27, 0.35, 0.43, 0.51)
        top = 0.36
        row = 0.058
        self.label(sc, self.P(0.05, top), "pair", 0.026, anchor="lm", opacity=op,
                   ink=layout.NOTE_INK)
        for i, x in enumerate(cols):
            self.label(sc, self.P(x, top), str(i), 0.026, opacity=op, ink=layout.NOTE_INK)
        rows = [("θ, base 10000", self.theta, None),
                ("θ, base %d" % round(self.ntk_b), self.theta_ntk, None)]
        reach = MAX_TEST_OFF / MAX_TRAIN_OFF
        tau = 2 * np.pi
        fits = [("plain", np.full(N_PAIRS, reach), PLAIN_INK),
                ("interpolation", np.full(N_PAIRS, reach / SCALE), INTERP_INK),
                ("NTK base", reach * self.theta_ntk / self.theta, NTK_INK)]
        for r, (name, vals, _) in enumerate(rows):
            y = top + row * (r + 1)
            show = smoothstep(0.05 + 0.1 * r, 0.15 + 0.1 * r, prog) * op
            self.label(sc, self.P(0.05, y), name, 0.026, anchor="lm", opacity=show)
            for x, v in zip(cols, vals):
                self.label(sc, self.P(x, y), "%.2g" % v, 0.026, opacity=show,
                           ink=layout.STRONG_INK)
        y = top + row * 3 + 0.02
        head = smoothstep(0.3, 0.4, prog) * op
        self.label(sc, self.P(0.05, y), "largest angle at 128 ÷ largest in training", 0.025,
                   anchor="lm", opacity=head, ink=layout.NOTE_INK)
        for r, (name, ratio, ink) in enumerate(fits):
            y = top + row * (r + 4) + 0.02
            show = smoothstep(0.35 + 0.1 * r, 0.45 + 0.1 * r, prog) * op
            self.label(sc, self.P(0.05, y), name, 0.026, anchor="lm", opacity=show,
                       ink=tuple(float(v) for v in ink))
            for i, (x, v) in enumerate(zip(cols, ratio)):
                wraps = self._covered(i) >= tau          # the pair went all round in training
                ok = wraps or v <= 1.05
                cell_ink = TRAIN_INK if ok else NEW_INK
                self.label(sc, self.P(x, y), "%.2f" % v, 0.026, opacity=show,
                           ink=tuple(float(c) for c in cell_ink))
        note = smoothstep(0.7, 0.8, prog) * op
        self.label(sc, self.P(0.05, 0.785), "green: within what the pair saw (pair 0 went round "
                   "%.1f times)" % (self._covered(0) / tau), 0.025, anchor="lm", opacity=note,
                   ink=layout.NOTE_INK)
        self.label(sc, self.P(0.05, 0.825), "interpolation keeps every angle, but not offset 1 "
                   "(%.2f rad)" % (self.theta[0] / SCALE), 0.025, anchor="lm", opacity=note,
                   ink=layout.NOTE_INK)
        last = smoothstep(0.85, 0.95, prog) * op
        self.label(sc, self.P(0.05, 0.875), "real models are then fine-tuned briefly at the",
                   0.025, anchor="lm", opacity=last, ink=layout.CAPTION_INK)
        self.label(sc, self.P(0.05, 0.925), "new length (Chen et al., 2023); this one is not",
                   0.025, anchor="lm", opacity=last, ink=layout.CAPTION_INK)


def _rad(v):
    """An angle with about three significant figures."""
    return ("%.0f" if v >= 10 else "%.1f" if v >= 1 else "%.2f" if v >= 0.1 else
            "%.3f" if v >= 0.01 else "%.4f") % v

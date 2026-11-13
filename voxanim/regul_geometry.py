"""L1 against L2: why one penalty makes weights exactly zero and the other
only shrinks them, on a real classifier.

    a logistic regression of MNIST 3s against 8s on block averages of the
    digit (:func:`voxanim.regul.logistic_data`), fitted with an L1 or an L2
    penalty of growing strength (FISTA, warm-started; the intercept free)

1. **the balls** -- the two weights of the small model, w₆ and w₉, and the
   two blocks of the digit they read; the L1 ball ``|w₆| + |w₉| <= r`` (a
   diamond, its corners on the axes) and the L2 ball ``w₆² + w₉² <= r²``
   (a circle),
2. **the plane** -- the training loss over the two weights as contours.  As
   the strength λ grows the penalised optimum slides along its path, and the
   ball through it touches the lowest contour it can reach: the L2 circle
   touches it on its curve, so both weights shrink and neither reaches 0;
   the L1 diamond touches it at a corner, so w₆ becomes exactly 0 (and stays
   there) long before w₉ does,
3. **one weight** -- the same on one number: the best ``w`` for a pull
   ``½(w - v)²`` towards ``v`` plus the penalty.  L2 gives ``v / (1 + λ)``,
   a shrink; L1 gives the soft threshold, ``v`` moved λ towards 0 and stopped
   there, flat at 0 for ``|v| <= λ``.  The sixteen fitted weights of the full
   model ride on both curves,
4. **sixteen weights** -- the full model's lasso and ridge paths: the weights
   against λ (log), the sixteen as a 4 x 4 map of the digit blocks (exact
   zeros blank), how many are left and the test accuracy.

Every number is from :func:`voxanim.regul.plane` and :func:`voxanim.regul.paths`.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import layout
from .diagram import Diagram
from .easing import ease_in_out, remap, smoothstep
from .gen_common import draw_digit
from .losschart import LossChart, pow10, sup
from .regul import BLOCK, CLASSES, GRID_SIDE, PAIR, first_zero, logistic_data, paths, plane
from .regul_plots import L1_INK, L2_INK, NONE_INK, contour_segments, draw_mosaic, signed_texture
from .text import MINUS, num, sub

PANEL = (0.955, 0.955, 0.95)
PANEL_EDGE = (0.70, 0.71, 0.72)
AXIS = (0.50, 0.52, 0.54)
CONTOUR = (0.58, 0.60, 0.62)
STRONG = layout.STRONG_INK
PX_X, PX_Y = 960 * layout.ZOOM, 540 * layout.ZOOM          # reference px per layout unit
# the plane: two boxes with equal px per unit on both axes, so circles are round
UNIT_PX = 72.0
W_RANGE, V_RANGE = (-2.0, 0.45), (-0.3, 3.45)               # w6 (across) and w9 (up)
BOX_W = (W_RANGE[1] - W_RANGE[0]) * UNIT_PX / PX_X
BOX_H = (V_RANGE[1] - V_RANGE[0]) * UNIT_PX / PX_Y
BOX_TOP = 0.30
BOXES = {"l1": ((0.085, 0.085 + BOX_W), (BOX_TOP, BOX_TOP + BOX_H)),
         "l2": ((0.915 - BOX_W, 0.915), (BOX_TOP, BOX_TOP + BOX_H))}
INKS = {"l1": L1_INK, "l2": L2_INK}
NAMES = {"l1": "L1", "l2": "L2"}
DIGIT_PX = 84.0
DIGITS_Y = 0.40                                              # the two digits, layout y
CONTOUR_STEPS = (0.01, 0.03, 0.07, 0.13, 0.22, 0.35, 0.52)  # loss above the optimum
BALL_TINT = 0.16
TEX_PER_UNIT = 40                                           # the ball's texture resolution
# one weight
ONE_BOX = ((0.10, 0.50), (0.30, 0.86))
ONE_RANGE = (-3.0, 3.0)
ONE_LAM = 1.0
# sixteen weights
CHARTS = {"l1": ((0.085, 0.425), (0.31, 0.66)), "l2": ((0.575, 0.915), (0.31, 0.66))}
PATH_Y = (-1.6, 2.9)
MAP_CELL = 17.0
MAP_Y = 0.83
KEEP_WITHIN = 0.01             # "nearly as good": test accuracy within this of all sixteen


W1, W2 = ("w" + sub(i) for i in PAIR)


def soft(v, lam):
    """L1's proximal step: ``v`` moved ``lam`` towards 0, stopped at 0."""
    return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)


def lerp_path(path, u):
    """``(w, lam, i)`` of ``path`` at fraction ``u`` of its strengths (log-even):
    the weights interpolated between fitted strengths, ``i`` the fitted one
    at or below (what the read-outs quote)."""
    k = u * (len(path.lams) - 1)
    i = int(np.floor(k))
    j = min(i + 1, len(path.lams) - 1)
    a = k - i
    w = path.w[i] * (1 - a) + path.w[j] * a
    lam = float(np.exp(np.log(path.lams[i]) * (1 - a) + np.log(path.lams[j]) * a))
    return w, lam, i


def fmt_lam(v):
    """A strength to two significant figures: ``0.10``, ``0.0011``, ``1.8``, ``60``."""
    if v >= 10:
        return "%.0f" % v
    return "%#.2g" % v


def wfmt(v):
    """A weight: ``0`` when exactly zero, ``−1.29``, and ``−4×10⁻⁴`` when it is
    smaller than 0.01 but not zero (so a shrunk weight never reads as 0)."""
    if v == 0:
        return "0"
    if abs(v) < 0.01:
        s = pow10(abs(v))
        return (MINUS if v < 0 else "") + s
    return num(v, 2)


def pct(v):
    """``0.858`` as ``85.8%``."""
    return "%.1f%%" % (100 * v)


class L1VsL2(Diagram):
    """The balls, the plane, one weight and sixteen (see the module)."""

    PHASES = ("the balls", "the plane", "one weight", "sixteen weights")

    def __init__(self, pl=None, pa=None, dat=None):
        super().__init__("L1 against L2", "")
        self.pl = plane() if pl is None else pl
        self.pa = paths() if pa is None else pa
        dat = logistic_data() if dat is None else dat
        self.n_train = len(dat.y_train)
        self.digit3 = dat.images[int(np.argmax(dat.y_test == 0))]
        self.digit8 = dat.images[int(np.argmax(dat.y_test == 1))]
        self.hold = 2.5
        # the plane
        self.r0 = {"l1": float(np.abs(self.pl.w0).sum()), "l2": float(np.linalg.norm(self.pl.w0))}
        xs, ys = self.pl.axes
        self.xs, self.ys = xs, ys
        self.static = [contour_segments(self.pl.loss, xs, ys, self.pl.loss0 + d)
                       for d in CONTOUR_STEPS]
        self.zero_at = {k: [first_zero(getattr(self.pl, k), i) for i in range(2)]
                        for k in ("l1", "l2")}
        tx = np.linspace(*W_RANGE, int((W_RANGE[1] - W_RANGE[0]) * TEX_PER_UNIT))
        ty = np.linspace(V_RANGE[1], V_RANGE[0], int((V_RANGE[1] - V_RANGE[0]) * TEX_PER_UNIT))
        self.tex_x, self.tex_y = np.meshgrid(tx, ty)
        # one weight: the sixteen fitted weights
        self.v16 = self.pa.w0
        # sixteen weights
        self.charts = {k: LossChart(CHARTS[k], (getattr(self.pa, k).lams[0],
                                                getattr(self.pa, k).lams[-1]), PATH_Y, xlog=True)
                       for k in ("l1", "l2")}
        l1 = self.pa.l1
        ok = np.nonzero(l1.test_acc >= self.pa.acc0 - KEEP_WITHIN)[0]
        self.keep_i = int(ok.max())
        Tm, t = {}, 0.3
        Tm["balls"] = (t, t + 6.5)
        t += 6.5
        Tm["plane"] = (t, t + 17.0)
        Tm["sweep"] = (t + 2.6, t + 13.4)
        t += 17.0
        Tm["one"] = (t, t + 10.0)
        Tm["one_lam"] = (t + 1.5, t + 6.0)
        t += 10.0
        Tm["paths"] = (t, t + 17.0)
        Tm["path_sweep"] = (t + 1.0, t + 11.5)
        Tm["path_back"] = (t + 12.0, t + 14.0)
        t += 17.0
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["plane"][0], Tm["one"][0], Tm["paths"][0]]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        ph = 0 if reset > 0 else self.step_of(t)
        T = self.Tm
        def out_in(t0):
            # the old content leaves before the new one arrives (no crossfade)
            return (smoothstep(0.0, 1.0, remap(t, t0 - 0.6, t0 - 0.1)),
                    smoothstep(0.0, 1.0, remap(t, t0, t0 + 0.5)))
        one_out, one_in = out_in(T["one"][0])
        paths_out, paths_in = out_in(T["paths"][0])
        balls = smoothstep(0.0, 0.08, self.progress("balls", t))
        plane_op = balls * (1.0 - one_out) * back
        one_op = one_in * (1.0 - paths_out) * back
        paths_op = paths_in * back
        self._plane(sc, t, plane_op)
        self._digits(sc, t, plane_op)
        self._one(sc, t, one_op)
        self._paths(sc, t, paths_op)
        self.step_indicator(sc, ph, self.PHASES)
        self.subtitle = self._words(t, reset)

    # -- the plane --------------------------------------------------------------------
    def wp(self, key, w6, w9, lift=0.3):
        """World point of weights ``(w6, w9)`` in the plane box ``key``."""
        (x0, x1), (y0, y1) = BOXES[key]
        fx = x0 + (x1 - x0) * (w6 - W_RANGE[0]) / (W_RANGE[1] - W_RANGE[0])
        fy = y1 - (y1 - y0) * (w9 - V_RANGE[0]) / (V_RANGE[1] - V_RANGE[0])
        return self.P(fx, fy, lift)

    def _inside(self, w6, w9):
        return W_RANGE[0] <= w6 <= W_RANGE[1] and V_RANGE[0] <= w9 <= V_RANGE[1]

    def _ball_texture(self, key, r, ink):
        if key == "l1":
            inside = np.abs(self.tex_x) + np.abs(self.tex_y) <= r
        else:
            inside = np.hypot(self.tex_x, self.tex_y) <= r
        tex = np.empty(self.tex_x.shape + (3,), np.float32)
        tex[:] = np.asarray(PANEL, np.float32)
        tint = np.asarray(PANEL) * (1 - BALL_TINT) + np.asarray(ink) * BALL_TINT
        tex[inside] = np.asarray(tint, np.float32)
        return tex

    def _ball_outline(self, key, r, n=480):
        a = np.linspace(0.0, 2 * np.pi, n)
        if key == "l2":
            return r * np.cos(a), r * np.sin(a)
        c, s = np.cos(a), np.sin(a)
        k = r / (np.abs(c) + np.abs(s))
        return k * c, k * s

    def _clipped(self, sc, key, xs, ys, ink, width, op):
        prev = None
        for x, y in zip(xs, ys):
            p = (x, y) if self._inside(x, y) else None
            if p is not None and prev is not None:
                sc.line(self.wp(key, *prev, 0.5), self.wp(key, *p, 0.5), width=width, color=ink,
                        opacity=op, bias=0.0)
            prev = p

    def _segments(self, sc, key, segs, ink, width, op):
        for (a, b) in segs:
            if not (self._inside(*a) and self._inside(*b)):
                continue
            sc.line(self.wp(key, *a, 0.4), self.wp(key, *b, 0.4), width=width, color=ink,
                    opacity=op, bias=0.0)

    def plane_state(self, key, t):
        """``(w, lam, i, r)``: the penalised optimum, the strength, the fitted
        index at or below it and the ball's radius at time ``t``."""
        path = getattr(self.pl, key)
        u = ease_in_out(self.progress("sweep", t))
        if u <= 0:
            w = self.pl.w0
            return w, 0.0, -1, self._norm(key, w)
        w, lam, i = lerp_path(path, u)
        return w, lam, i, self._norm(key, w)

    @staticmethod
    def _norm(key, w):
        return float(np.abs(w).sum()) if key == "l1" else float(np.linalg.norm(w))

    def _plane(self, sc, t, op):
        op *= self._cap
        if op <= 0.01:
            return
        grow = smoothstep(0.0, 1.0, remap(self.progress("balls", t), 0.35, 0.75))
        contours = smoothstep(0.0, 1.0, remap(t, self.Tm["plane"][0], self.Tm["plane"][0] + 1.2))
        sweeping = self.progress("sweep", t) > 0
        for key in ("l1", "l2"):
            ink = INKS[key]
            (x0, x1), (y0, y1) = BOXES[key]
            if sweeping:
                w, _lam, i, r = self.plane_state(key, t)
            else:
                w, i, r = self.pl.w0, -1, self.r0[key] * grow
            a, b = self.P(x0, y0), self.P(x1, y1)
            wpx, hpx = self.px(b) - self.px(a)
            draw_mosaic(self, sc, (a + b) / 2, self._ball_texture(key, r, ink),
                        wpx / self.tex_x.shape[1], op, lift=0.1)
            self.rect(sc, (a + b) / 2, wpx + 2, hpx + 2, PANEL, op, border=PANEL_EDGE,
                      border_w=1.0, radius=3, lift=0.05, fill_op=0.0)
            # the axes through 0
            sc.line(self.wp(key, W_RANGE[0], 0.0, 0.2), self.wp(key, W_RANGE[1], 0.0, 0.2),
                    width=1.0, color=AXIS, opacity=0.8 * op, bias=0.0)
            sc.line(self.wp(key, 0.0, V_RANGE[0], 0.2), self.wp(key, 0.0, V_RANGE[1], 0.2),
                    width=1.0, color=AXIS, opacity=0.8 * op, bias=0.0)
            # the axes' names, outside the box: w6 runs across, w9 up
            self.label(sc, self.wp(key, W_RANGE[0], 0.0), W1 + " →", layout.MIN_TEXT,
                       anchor="rm", dx=-5, ink=layout.NOTE_INK, opacity=op)
            self.label(sc, self.wp(key, 0.0, V_RANGE[0]), W2 + " ↑", layout.MIN_TEXT,
                       anchor="mt", dy=4, ink=layout.NOTE_INK, opacity=op)
            # contours of the training loss
            if contours > 0.01:
                for segs in self.static:
                    self._segments(sc, key, segs, CONTOUR, 1.0, 0.45 * op * contours)
                self.disk(sc, self.wp(key, *self.pl.w0), 3.6, NONE_INK, op * contours, lift=1.0)
                self.label(sc, self.wp(key, *self.pl.w0), "no penalty", layout.MIN_TEXT,
                           anchor="rb", dx=-5, dy=-5, ink=NONE_INK,
                           opacity=op * contours * (1.0 - smoothstep(
                               0.0, 1.0, remap(t, self.Tm["sweep"][0] - 0.5, self.Tm["sweep"][0]))))
            # the ball's rim
            xs, ys = self._ball_outline(key, r)
            self._clipped(sc, key, xs, ys, ink, 1.6, 0.9 * op * min(1.0, 4 * grow + sweeping))
            if sweeping:
                path = getattr(self.pl, key)
                u = ease_in_out(self.progress("sweep", t))
                k = u * (len(path.lams) - 1)
                pts = [self.pl.w0] + [path.w[j] for j in range(int(np.floor(k)) + 1)] + [w]
                for p, q in itertools.pairwise(pts):
                    sc.line(self.wp(key, *p, 0.6), self.wp(key, *q, 0.6), width=2.0, color=ink,
                            opacity=0.9 * op, bias=0.0)
                level = float(np.interp(k, np.arange(len(path.lams)), path.train_loss))
                self._segments(sc, key, contour_segments(self.pl.loss, self.xs, self.ys, level),
                               ink, 1.4, 0.8 * op)
                self.disk(sc, self.wp(key, *w), 4.2, ink, op, lift=1.2)
                sc.probe(("plane", key), self.wp(key, *w), tol=6.0)
            # the panel's name and read-outs
            name = ("L1 penalty  λ(|%s| + |%s|)" % (W1, W2) if key == "l1"
                    else "L2 penalty  λ(%s² + %s²) / 2" % (W1, W2))
            self.label(sc, self.P((x0 + x1) / 2, y0), name, layout.MIN_TEXT, anchor="mb", dy=-6,
                       ink=ink, opacity=op)
            if sweeping:
                path = getattr(self.pl, key)
                ii = max(i, 0)
                lines = ("λ = %s:  %s = %s,  %s = %s" % (fmt_lam(path.lams[ii]), W1,
                                                         wfmt(path.w[ii, 0]), W2,
                                                         wfmt(path.w[ii, 1])),
                         "test accuracy %s" % pct(path.test_acc[ii]))
            else:
                lines = ("no penalty:  %s = %s,  %s = %s" % (W1, wfmt(self.pl.w0[0]), W2,
                                                             wfmt(self.pl.w0[1])),
                         "test accuracy %s" % pct(self.pl.acc0))
            for n, s in enumerate(lines):
                self.label(sc, self.P(x0, y1 + 0.075 + n * 0.045), s, layout.MIN_TEXT,
                           anchor="lm", ink=STRONG if n == 0 else layout.CAPTION_INK,
                           opacity=op * contours)

    def _digits(self, sc, t, op):
        op *= self._cap
        if op <= 0.01:
            return
        cx = 0.5
        for n, (img, lab) in enumerate(((self.digit3, CLASSES[0]), (self.digit8, CLASSES[1]))):
            c = self.P(cx, DIGITS_Y + n * 0.215)
            draw_digit(self, sc, c, img, DIGIT_PX, op)
            for f in PAIR:
                r, col = divmod(f, GRID_SIDE)
                s = DIGIT_PX * BLOCK / 28.0
                off = np.array([((col + 0.5) * BLOCK / 28.0 - 0.5) * DIGIT_PX,
                                ((r + 0.5) * BLOCK / 28.0 - 0.5) * DIGIT_PX])
                self.rect(sc, self.shift(c, *off), s, s, PANEL, op, border=STRONG, border_w=1.4,
                          radius=1.0, lift=0.5, fill_op=0.0)
            self.label(sc, c, "a %d" % lab, layout.MIN_TEXT, anchor="mt", dy=DIGIT_PX / 2 + 4,
                       ink=layout.NOTE_INK, opacity=op)
        c = self.P(cx, DIGITS_Y)
        s = DIGIT_PX * BLOCK / 28.0
        for f, name in zip(PAIR, (W1, W2)):
            r, col = divmod(f, GRID_SIDE)
            off = np.array([((col + 0.5) * BLOCK / 28.0 - 0.5) * DIGIT_PX,
                            ((r + 0.5) * BLOCK / 28.0 - 0.5) * DIGIT_PX])
            side = 1 if col >= GRID_SIDE / 2 else -1
            self.label(sc, self.shift(c, side * (DIGIT_PX / 2 + 6), off[1]), name,
                       layout.MIN_TEXT, anchor="lm" if side > 0 else "rm", ink=STRONG, opacity=op)

    # -- one weight ------------------------------------------------------------------
    def one_lam(self, t):
        """λ of the one-weight panel at ``t``."""
        return ONE_LAM * ease_in_out(self.progress("one_lam", t))

    def _op(self, v, w, lift=0.4):
        (x0, x1), (y0, y1) = ONE_BOX
        a, b = ONE_RANGE
        fx = x0 + (x1 - x0) * (v - a) / (b - a)
        fy = y1 - (y1 - y0) * (np.clip(w, a, b) - a) / (b - a)
        return self.P(fx, fy, lift)

    def _one(self, sc, t, op):
        op *= self._cap
        if op <= 0.01:
            return
        lam = self.one_lam(t)
        (x0, x1), (y0, y1) = ONE_BOX
        a0, b0 = self.P(x0, y0), self.P(x1, y1)
        wpx, hpx = self.px(b0) - self.px(a0)
        self.rect(sc, (a0 + b0) / 2, wpx, hpx, PANEL, op, border=PANEL_EDGE, border_w=1.0,
                  radius=4, lift=0.1)
        a, b = ONE_RANGE
        for p, q in (((a, 0.0), (b, 0.0)), ((0.0, a), (0.0, b))):
            sc.line(self._op(*p, 0.2), self._op(*q, 0.2), width=1.0, color=AXIS,
                    opacity=0.7 * op, bias=0.0)
        v = np.linspace(a, b, 241)
        self.polyline(sc, [self._op(x, x, 0.25) for x in (a, b)], 1.0, CONTOUR, 0.8 * op)
        self.polyline(sc, [self._op(x, x / (1 + lam)) for x in (a, b)], 2.0, L2_INK, op)
        self.polyline(sc, [self._op(x, y) for x, y in zip(v, soft(v, lam))], 2.0, L1_INK, op)
        self.label(sc, self.P(x1, y1), "v", layout.MIN_TEXT, anchor="rb", dx=-4, dy=-4,
                   ink=layout.NOTE_INK, opacity=op)
        self.label(sc, self.P((x0 + x1) / 2, y0), "the best w", layout.MIN_TEXT, anchor="lt",
                   dx=6, dy=4, ink=layout.NOTE_INK, opacity=op)
        # on the side of the axis the L1 curve leaves free
        self.label(sc, self._op(-lam, 0.0), "−λ", layout.MIN_TEXT, anchor="mb", dy=-6,
                   ink=L1_INK, opacity=op * smoothstep(0.25, 0.45, lam))
        self.label(sc, self._op(lam, 0.0), "λ", layout.MIN_TEXT, anchor="mt", dy=6,
                   ink=L1_INK, opacity=op * smoothstep(0.25, 0.45, lam))
        dots = smoothstep(0.0, 1.0, remap(t, self.Tm["one"][0] + 0.6, self.Tm["one"][0] + 1.4))
        for x in self.v16:
            self.disk(sc, self._op(x, x / (1 + lam), 0.8), 3.0, L2_INK, op * dots, lift=0.8)
            self.disk(sc, self._op(x, soft(x, lam), 0.9), 3.0, L1_INK, op * dots, lift=0.9)
        zeros = int((soft(self.v16, lam) == 0).sum())
        lines = [("pull towards v:  ½(w − v)² + penalty", STRONG),
                 ("L2:  w = v / (1 + λ),  a shrink", L2_INK),
                 ("L1:  w = sign(v) · max(|v| − λ, 0)", L1_INK),
                 ("λ = %.2f" % lam, layout.CAPTION_INK),
                 ("the %d fitted weights (dots):" % len(self.v16), layout.CAPTION_INK),
                 ("L1 sets %d to exactly 0" % zeros, L1_INK),
                 ("L2 scales all %d by %.2f" % (len(self.v16), 1 / (1 + lam)), L2_INK)]
        for n, (s, ink) in enumerate(lines):
            self.label(sc, self.P(0.56, 0.33 + n * 0.075), s, layout.MIN_TEXT, anchor="lm",
                       ink=ink, opacity=op)

    # -- sixteen weights -------------------------------------------------------------
    def path_state(self, key, t):
        """``(lam, i)``: the strength reached and the fitted index at or below it."""
        path = getattr(self.pa, key)
        u = ease_in_out(self.progress("path_sweep", t))
        back = ease_in_out(self.progress("path_back", t))
        u = u + (self.keep_i / (len(self.pa.l1.lams) - 1) - u) * back
        k = u * (len(path.lams) - 1)
        i = int(np.floor(k))
        j = min(i + 1, len(path.lams) - 1)
        lam = float(np.exp(np.log(path.lams[i]) + (k - i) * (np.log(path.lams[j])
                                                             - np.log(path.lams[i]))))
        return lam, i

    def _paths(self, sc, t, op):
        op *= self._cap
        if op <= 0.01:
            return
        for key in ("l1", "l2"):
            path, ch, ink = getattr(self.pa, key), self.charts[key], INKS[key]
            lam, i = self.path_state(key, t)
            lo, hi = path.lams[0], path.lams[-1]
            ticks = [(10.0 ** e, "10" + sup(e)) for e in range(int(np.ceil(np.log10(lo))),
                                                                int(np.floor(np.log10(hi))) + 1)]
            ch.axes(sc, self.cam, self.P, op, "strength λ (log)", "%s: the 16 weights"
                    % ("lasso (L1)" if key == "l1" else "ridge (L2)"), ticks,
                    [(v, "%g" % v) for v in (-1, 0, 1, 2)])
            ch.hline(sc, self.P, 0.0, op)
            drawn = path.lams[-1] if self.progress("path_back", t) > 0 else lam
            for f in range(path.w.shape[1]):
                ch.curve(sc, self.P, path.lams, path.w[:, f], drawn, ink, 0.8 * op, width=1.2,
                         dot=False)
            ch.mark(sc, self.P, lam, ink, op)
            # the sixteen as a map of the digit's blocks
            x0 = CHARTS[key][0][0]
            w = path.w[i]
            tex = signed_texture(w, 2.5, side=GRID_SIDE)
            c = self.P(x0 + 0.06, MAP_Y)
            draw_mosaic(self, sc, c, tex, MAP_CELL, op)
            self.rect(sc, c, MAP_CELL * GRID_SIDE + 2, MAP_CELL * GRID_SIDE + 2, PANEL, op,
                      border=PANEL_EDGE, border_w=1.0, radius=1.5, lift=0.05, fill_op=0.0)
            nz = int(path.nonzero()[i])
            lines = (("λ = %s:  %d of 16 weights ≠ 0" % (fmt_lam(path.lams[i]), nz), STRONG),
                     ("test accuracy %s" % pct(path.test_acc[i]), layout.CAPTION_INK))
            for n, (s, ink2) in enumerate(lines):
                self.label(sc, self.P(x0 + 0.13, MAP_Y - 0.03 + n * 0.055), s, layout.MIN_TEXT,
                           anchor="lm", ink=ink2, opacity=op)
            sc.probe(("paths", key), (np.log10(lam) * 100, 0, 0), tol=6.0)

    # ------------------------------------------------------------------ words
    def _words(self, t, reset):
        T = self.Tm
        pl, pa = self.pl, self.pa
        if reset > 0 or t < T["balls"][0] + 2.6:
            return ("a classifier of %ds against %ds reading two 7 × 7 blocks of the digit, one "
                    "weight each (%s, %s), fitted on MNIST's %s training %ds and %ds"
                    % (CLASSES + (W1, W2, f"{self.n_train:,}") + CLASSES))
        if t < T["plane"][0]:
            return ("a penalty limits the weights to a ball: |%s| + |%s| ≤ r for L1, a diamond "
                    "with its corners on the axes; %s² + %s² ≤ r² for L2, a circle"
                    % (W1, W2, W1, W2))
        if t < T["sweep"][0]:
            return ("the training loss over the two weights, in contours; with no penalty its "
                    "lowest point is at %s = %s, %s = %s (%s test accuracy)"
                    % (W1, wfmt(pl.w0[0]), W2, wfmt(pl.w0[1]), pct(pl.acc0)))
        if t < T["sweep"][1] - 4.0:
            return ("as λ grows the best point slides in: it is where the smallest ball reaches "
                    "the lowest contour, and the contour through it (coloured) just touches it")
        if t < T["one"][0]:
            z6 = self.zero_at["l1"][0]
            z9 = self.zero_at["l1"][1]
            w6_l2 = pl.l2.w[-1, 0]
            return ("L1: the contour meets the diamond at its corner, so %s = 0 exactly from λ = "
                    "%s (and %s from %s); L2 at λ = %s still has %s = %s: shrunk, never zero"
                    % (W1, fmt_lam(z6), W2, fmt_lam(z9), fmt_lam(pl.l2.lams[-1]), W1,
                       wfmt(w6_l2)))
        if t < T["paths"][0]:
            if t < T["one_lam"][1]:
                return ("one weight pulled towards a value v, plus the penalty: L2's best w is "
                        "v / (1 + λ); L1's is v moved λ towards 0 and stopped there")
            zeros = int((soft(self.v16, ONE_LAM) == 0).sum())
            return ("at λ = %g the soft threshold puts every |v| ≤ λ at exactly 0: %d of the %d "
                    "fitted weights; L2's line only scales them by %.2f"
                    % (ONE_LAM, zeros, len(self.v16), 1 / (1 + ONE_LAM)))
        if t < T["path_back"][0] + 0.5:
            seq = []
            for n in pa.l1.nonzero():
                if not seq or seq[-1] != int(n):
                    seq.append(int(n))
            return ("all 16 weights as λ grows: lasso (L1) drops them to 0 one by one (%s) while "
                    "ridge (L2) shrinks all 16 together and zeroes none"
                    % ", ".join(str(n) for n in seq[:7]) + ", …")
        k = self.keep_i
        return ("lasso at λ = %s keeps %d of the 16 weights for %s test accuracy (%s with all "
                "16): the blank blocks are the parts of the digit it no longer reads"
                % (fmt_lam(pa.l1.lams[k]), int(pa.l1.nonzero()[k]), pct(pa.l1.test_acc[k]),
                   pct(pa.acc0)))


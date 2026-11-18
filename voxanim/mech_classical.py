"""Classical machine learning: animations 67-74 of the "next 100" (see :mod:`voxanim.mech`).

Each draws the method at work on the real MNIST images its study measured -
the resampled fits, the folds, the margin, the trees, the ellipses, the
neighbours, the maps, the threshold - and then the numbers that came out.
Four studies are :mod:`voxanim.next_vision`'s; what one of those kept on disk
(its fitted models, its samples) is read back with :func:`kept`. The other
four (67, 71, 72, 73) were redesigned in :mod:`voxanim.study_classical`, whose
``*_data`` functions give their arrays. Anything a study did not keep (the
steps in between) is recomputed here by the study's own procedure and checked
against the study's result in ``tests/test_mech_classical.py``.
"""

from __future__ import annotations

import itertools
import json
from functools import cache

import numpy as np

from . import next_vision as nv
from . import study_classical as sx
from . import theme, toy2d
from .cache import cached_npz
from .diagram import draw_image
from .easing import smoothstep
from .losschart import LossChart, pow10
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIGIT_INK,
    DIM,
    GREEN,
    INKS,
    MAGENTA,
    OCHRE,
    PAPER,
    RED,
    TEAL,
    VIOLET,
    WHITE,
    Mechanism,
    dark,
    number,
    soft,
)

# The settings each study's cache file is keyed by (next_vision.py passes the same ones).
KEPT = {
    "cross_validation_mnist": dict(train=200, test=200, digits=[3, 8], outer_folds=5,
                                   inner_folds=4, ridge=0.03),
    "svm_mnist": dict(train=200, test=200, digits=[3, 8], cs=[0.15, 1.0, 10.0, 100.0],
                      smo_passes=8, linear_updates=600),
    "boosting_mnist": dict(train=500, test=300, trees=60, depth=3, rates=[0.12, 1.0],
                           target="bottom-half mean ink from top-half pixels"),
}


def _not_kept():
    raise RuntimeError("the study's arrays are not in the cache under these settings: "
                       "KEPT in mech_classical.py no longer matches next_vision.py")


@cache
def kept(key):
    """Everything the study ``key`` saved (its models, samples and measurements).
    The study has been built by the time its animation is, so this only reads."""
    return nv.cached(key, _not_kept, **KEPT[key])


def pixels(split, indices):
    """MNIST images (rows of 784 values in 0..1) by their index in ``split``."""
    x, _, xt, _ = nv.mnist_data()
    return (x if split == "train" else xt)[np.asarray(indices)]


def contour(f, xr, yr, n=56):
    """Where ``f(points) = 0`` inside the box, as line segments ``(m, 2, 2)``
    (marching squares on an ``n x n`` grid)."""
    gx, gy = np.linspace(xr[0], xr[1], n), np.linspace(yr[0], yr[1], n)
    X, Y = np.meshgrid(gx, gy)
    v = np.asarray(f(np.column_stack((X.ravel(), Y.ravel()))), float).reshape(n, n)
    out = []
    for i in range(n - 1):
        for j in range(n - 1):
            corners = ((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))
            cross = []
            for (r0, c0), (r1, c1) in zip(corners, corners[1:] + corners[:1]):
                a, b = v[r0, c0], v[r1, c1]
                if (a > 0) != (b > 0):
                    s = a / (a - b)
                    cross.append((gx[c0] + (gx[c1] - gx[c0]) * s, gy[r0] + (gy[r1] - gy[r0]) * s))
            for k in range(0, len(cross) - 1, 2):
                out.append((cross[k], cross[k + 1]))
    return np.array(out, float).reshape(-1, 2, 2)


class Pane(LossChart):
    """A plotting area: data coordinates to layout fractions, unclamped."""

    def __init__(self, box, xr, yr):
        super().__init__(box, tuple(map(float, xr)), tuple(map(float, yr)))

    def at(self, x, y):
        (a, b), (c, d) = self.x_range, self.y_range
        return (self.x0 + (self.x1 - self.x0) * (x - a) / (b - a),
                self.y1 - (self.y1 - self.y0) * (y - c) / (d - c))

    def holds(self, x, y, pad=0.0):
        (a, b), (c, d) = self.x_range, self.y_range
        return (a - pad * (b - a) <= x <= b + pad * (b - a)
                and c - pad * (d - c) <= y <= d + pad * (d - c))

    @property
    def centre(self):
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2


class Classical(Mechanism):
    """What these eight share: points, curves and outlines in a :class:`Pane`."""

    def square(self, cx, cy, side, xr, yr):
        """A ``side`` px square pane centred at layout ``(cx, cy)``."""
        dx, dy = self.frac(side / 2, side / 2)
        return Pane(((cx - dx, cx + dx), (cy - dy, cy + dy)), xr, yr)

    def backdrop(self, sc, pane, op, pad=8.0):
        """The paper a pane's points lie on."""
        w, h = self.span(pane.x1 - pane.x0, pane.y1 - pane.y0)
        self.rect(sc, self.P(*pane.centre), w + pad, h + pad, PAPER, op, border=AXIS,
                  border_w=1.2, radius=5, lift=0.1)

    def dots(self, sc, pane, pts, inks, op, r=3.2, upto=None, ring=None):
        """Points (data coordinates) as balls; ``inks`` one colour or one per
        point; only the first ``upto`` are drawn; ``ring``: a colour for a band."""
        if op <= 0.01:
            return
        pts = np.asarray(pts, float).reshape(-1, 2)
        one = np.ndim(inks[0]) == 0
        for i, (x, y) in enumerate(pts[:upto]):
            if pane.holds(x, y):
                self.disk(sc, self.P(*pane.at(x, y)), r, inks if one else inks[i], op, ring=ring,
                          ring_w=1.8, lift=1.0)

    def path(self, sc, pane, xs, ys, ink, op, width=2.0, u=1.0):
        """A curve in data coordinates, drawn as far as fraction ``u`` and only
        where it is inside the pane."""
        if op <= 0.01 or u <= 0:
            return
        m = max(2, int(np.ceil(len(xs) * min(u, 1.0))))
        prev = None
        for x, y in zip(xs[:m], ys[:m]):
            p = self.P(*pane.at(x, y), 0.9) if pane.holds(x, y) else None
            if prev is not None and p is not None:
                sc.line(prev, p, width=width, color=ink, opacity=op, bias=0.3)
            prev = p

    def outline(self, sc, pane, segments, ink, op, width=2.2, u=1.0):
        """Line segments ``(m, 2, 2)`` in data coordinates (a :func:`contour`)."""
        if op <= 0.01 or u <= 0:
            return
        for a, b in segments[:max(1, int(len(segments) * min(u, 1.0)))]:
            sc.line(self.P(*pane.at(*a), 0.9), self.P(*pane.at(*b), 0.9), width=width, color=ink,
                    opacity=op, bias=0.3)

    def tint(self, sc, fx, fy, w, h, ink, op, strength=0.22):
        """A wash of ``ink`` over a ``w x h`` px area centred at ``(fx, fy)``."""
        self.rect(sc, self.P(fx, fy), w, h, ink, op, radius=1.0, lift=0.8, fill_op=strength)

    def turn(self, u, n, lead=1.0):
        """Which of ``n`` equal turns ``u`` is in, and how far through it (0..1)."""
        s = min(max(u, 0.0) * lead, 1.0) * n
        k = min(int(s), n - 1)
        return k, min(s - k, 1.0)


# ====================================================================== 67 bias-variance
def small(v):
    """A squared error as a label: four decimals, three significant once it is large."""
    return "%.4f" % v if v < 0.0995 else "%.3f" % v


class BiasVariance(Classical):
    """Polynomials fitted to resampled measurements of where a digit's ink is:
    a stiff curve misses the hump every time (bias), a flexible one chases each
    sample's accidents (variance), and the error is least in between."""

    TITLE = "Bias and variance"
    PHASES = (("the task", 11.0), ("resample", 13.0), ("degrees", 17.0), ("the trade-off", 13.0))
    XR, YR = (-0.8, 27.8), (-0.22, 1.02)
    BOX = ((0.10, 0.58), (0.34, 0.83))
    FITS = 36                      # of each degree's 200, drawn
    SAMPLES = 6                    # resamples shown one at a time
    SHOWN = (1, 2, 4, 6, 8, 10)    # the degrees stepped through
    SCALE = 0.02                   # the squared error a full bar stands for
    TRUTH, FIT, MEAN, NOISE, TOTAL = DIGIT_INK, BLUE, RED, OCHRE, VIOLET

    def setup(self):
        d, s = sx.bias_variance_data(), self.study
        self.pool, self.sampled = d["pool"], d["sampled"]
        self.fine, self.curves, self.truth = d["fine"], d["curves"], d["truth"]
        self.degrees = np.asarray(s.panels[2].data["x"], int)
        self.bias, self.spread, noise, self.mse = (np.asarray(v, float)
                                                   for v in s.panels[2].data["ys"])
        self.noise = float(noise[0])
        self.best = int(np.argmin(self.mse))
        self.mean_curves = self.curves.mean(1)
        self.shown = [int(np.flatnonzero(self.degrees == v)[0]) for v in self.SHOWN]
        self.draws = self.sampled[self.best]            # [resample, (image, column), 40]
        many = self.draws[:12]
        self.cloud = np.column_stack((many[:, 1].ravel(),
                                      self.pool[many[:, 0].ravel(), many[:, 1].ravel()]))
        used = np.unique(self.draws[:self.SAMPLES, 0, :10])
        self.images = dict(zip(used.tolist(), pixels("train", used)))
        first = self.draws[0]
        order = np.argsort(first[1], kind="stable")[[5, 15, 24, 34]]
        self.examples = [(int(first[0, i]), int(first[1, i])) for i in order]
        self.images.update(zip([i for i, _ in self.examples],
                               pixels("train", [i for i, _ in self.examples])))
        self.pane = Pane(self.BOX, self.XR, self.YR)

    def subtitle_for(self, k):
        best, last = self.degrees[self.best], self.degrees[-1]
        return [
            "one measurement: pick a handwritten digit and one of its 28 columns, and measure "
            "how much of that column is inked. The task: predict the ink from where the column is",
            "draw %d measurements at random and fit a degree-%d curve through them; draw again "
            "and the curve comes out different. %d draws give %d curves" % (
                self.sampled.shape[-1], best, self.curves.shape[1], self.curves.shape[1]),
            "a straight line misses the hump on every draw: bias %s, falling to %s at degree %d. "
            "But flexible curves chase their own %d points: the spread grows from %s to %s" % (
                small(self.bias[0]), small(self.bias[self.best]), best, self.sampled.shape[-1],
                small(self.spread[0]), small(self.spread[-1])),
            "on 10,000 unseen images the squared error is exactly bias plus spread plus noise. "
            "It is lowest at degree %d: %s, against %s for the line and %s for degree %d" % (
                best, small(self.mse[self.best]), small(self.mse[0]), small(self.mse[-1]), last),
        ][k]

    def _axes(self, sc, pane, op):
        self.backdrop(sc, pane, op, pad=0.0)
        self.axes(sc, pane, op, "column of the image, left to right", "share of the column inked",
                  grid=False, xticks=[(v, str(v + 1)) for v in (0, 9, 18, 27)],
                  yticks=[(v, "%d%%" % (100 * v)) for v in (0, 0.5, 1.0)])

    def _strip(self, sc, fx, fy, size, column, op):
        """The measured column of an image drawn ``size`` px wide at ``(fx, fy)``."""
        dx = self.frac((column + 0.5) / 28.0 * size - size / 2)[0]
        self.tint(sc, fx + dx, fy, max(size / 28.0, 3.0), size, AMBER, op, strength=0.55)

    def _truth(self, sc, pane, op, u=1.0, width=3.2):
        self.path(sc, pane, np.arange(28.0), self.truth, self.TRUTH, op, width, u)

    def phase_1(self, sc, u, op):
        pane = Pane(((0.56, 0.93), (0.34, 0.83)), self.XR, self.YR)
        self._axes(sc, pane, op)
        self.dots(sc, pane, self.cloud, DIM, op, r=2.3,
                  upto=int(len(self.cloud) * smoothstep(0.48, 0.72, u)))
        mean = smoothstep(0.74, 0.92, u)
        self._truth(sc, pane, op, mean)
        self.lines(sc, 0.915, 0.385, ["%s: the average of" % theme.word("black", "white"),
                                      "10,000 unseen images"],
                   op * mean, anchor="rm", step=0.04, size=0.030)
        for n, (i, c) in enumerate(self.examples):
            fx, fy = 0.14 + 0.21 * (n % 2), 0.445 + 0.275 * (n // 2)
            self.digit(sc, self.images[i], fx, fy, 96, op)
            self._strip(sc, fx, fy, 96, c, op)
            ink = self.pool[i, c]
            self.text(sc, fx, fy + 0.118, "column %d: %.0f%% inked" % (c + 1, 100 * ink), op,
                      ink=dark(AMBER), size=0.032)
            at = pane.at(c, ink)
            start = 0.08 + 0.09 * n
            self.pulse(sc, (fx, fy), at, (u - start) / 0.2, AMBER, op, bend=0.05, r=6)
            if u > start + 0.18:
                self.disk(sc, self.P(*at), 5.5, AMBER, op, lift=1.3)

    def phase_2(self, sc, u, op):
        self._axes(sc, self.pane, op)
        self._truth(sc, self.pane, op * 0.55, width=2.4)
        k, w = self.turn(u, self.SAMPLES, 1.04)
        images, columns = self.draws[k]
        for j in range(k):                               # the fits already made
            self.path(sc, self.pane, self.fine, self.curves[self.best][j], self.FIT, op * 0.45,
                      1.6)
        self.dots(sc, self.pane, np.column_stack((columns, self.pool[images, columns])), AMBER,
                  op, r=3.8, upto=int(len(images) * smoothstep(0.0, 0.4, w)))
        self.path(sc, self.pane, self.fine, self.curves[self.best][k], self.FIT, op, 2.8,
                  smoothstep(0.42, 0.85, w))
        x = 0.66
        self.text(sc, x, 0.33, "sample %d of %d" % (k + 1, self.curves.shape[1]), op,
                  anchor="lm", strong=True, size=0.040)
        self.text(sc, x, 0.39, "%d measurements, drawn at random" % len(images), op, anchor="lm",
                  ink=dark(AMBER))
        shown = int(10 * smoothstep(0.0, 0.4, w) + 0.999)
        for n in range(shown):
            fx, fy = x + 0.025 + 0.057 * (n % 5), 0.49 + 0.105 * (n // 5)
            self.digit(sc, self.images[int(images[n])], fx, fy, 46, op)
            self._strip(sc, fx, fy, 46, int(columns[n]), op)
        self.text(sc, x, 0.70, "and %d more" % (len(images) - 10), op * smoothstep(0.3, 0.45, w),
                  anchor="lm")
        self.text(sc, x, 0.76, "the degree-%d curve through them" % self.degrees[self.best],
                  op * smoothstep(0.42, 0.6, w), anchor="lm", ink=dark(self.FIT), size=0.034)
        self.text(sc, x, 0.82, "%d curves so far" % (k + (w > 0.85)), op, anchor="lm")

    def phase_3(self, sc, u, op):
        self._axes(sc, self.pane, op)
        n, w = self.turn(u, len(self.shown), 1.03)
        k = self.shown[n]
        show = smoothstep(0.0, 0.3, w)
        for j in range(self.FITS):
            self.path(sc, self.pane, self.fine, self.curves[k][j], self.FIT, op * 0.42 * show,
                      1.3)
        self._truth(sc, self.pane, op)
        self.path(sc, self.pane, self.fine, self.mean_curves[k], self.MEAN, op * show, 3.2)
        x = 0.66
        self.text(sc, x, 0.34, "degree %d" % self.degrees[k], op, anchor="lm", strong=True,
                  size=0.046)
        full = self.span(0.17)[0]
        for row, (name, ink, v) in enumerate((
                ("bias: how far the mean fit misses", self.MEAN, self.bias[k]),
                ("variance: the spread of the fits", self.FIT, self.spread[k]))):
            y = 0.43 + 0.135 * row
            self.text(sc, x, y, name, op, anchor="lm", ink=dark(ink))
            over = v > self.SCALE
            self.hbar(sc, x, y + 0.055, full * min(v / self.SCALE, 1.0), ink, op,
                      value=small(v) + (", off the scale" if over else ""), u=show, thick=26.0,
                      track=full)
        self.lines(sc, x, 0.72, [("%s: the average of unseen images"
                                  % theme.word("black", "white"), None),
                                 ("thin: %d of the %d fits" % (self.FITS, self.curves.shape[1]),
                                  dark(self.FIT)),
                                 ("thick red: the mean of all %d" % self.curves.shape[1],
                                  dark(self.MEAN))], op, step=0.045, size=0.032)

    def phase_4(self, sc, u, op):
        chart = LossChart(((0.12, 0.58), (0.37, 0.80)), (1.0, float(self.degrees[-1])),
                          (1e-4, 0.3), ylog=True)
        self.axes(sc, chart, op, "degree of the polynomial",
                  "squared error on unseen images (each line: ten times more)",
                  xticks=[(d, str(d)) for d in self.degrees],
                  yticks=[(v, "%g" % v) for v in (1e-4, 1e-3, 1e-2, 1e-1)])
        s = u / 0.62
        flat = np.full(len(self.degrees), self.noise)
        for ys, ink, width in ((flat, self.NOISE, 1.8), (self.bias, self.MEAN, 2.4),
                               (self.spread, self.FIT, 2.4), (self.mse, self.TOTAL, 3.4)):
            self.plot(sc, chart, self.degrees, ys, ink, op, s, width=width)
        last = len(self.degrees) - 1
        k = int(np.clip(round(s * last), 0, last))
        x = 0.66
        self.text(sc, x, 0.35, "degree %d" % self.degrees[k], op, anchor="lm", strong=True,
                  size=0.046)
        self.lines(sc, x, 0.44, [("bias   %s" % small(self.bias[k]), dark(self.MEAN)),
                                 ("+ spread   %s" % small(self.spread[k]), dark(self.FIT)),
                                 ("+ noise   %s" % small(self.noise), dark(self.NOISE)),
                                 ("= error   %s" % small(self.mse[k]), dark(self.TOTAL))],
                   op, step=0.062, size=0.036)
        end = op * smoothstep(0.68, 0.82, u)
        if end > 0.01:
            at = chart.fx(self.degrees[self.best]), chart.fy(self.mse[self.best])
            self.disk(sc, self.P(*at), 8.0, self.TOTAL, end, ring=AMBER, ring_w=3.0, lift=1.6)
        self.lines(sc, x, 0.72, [
            "lowest at degree %d" % self.degrees[self.best],
            "stiffer: the bias costs more",
            "more flexible: the spread does",
            "noise: no curve removes it",
            "(red at %d: a mean of %d wild fits)" % (
                self.degrees[-1], self.curves.shape[1])], end, step=0.044, size=0.032)


# ====================================================================== 68 cross-validation
RIDGE = 0.03                       # the study's kernel-ridge penalty
ROLE_PAPER = np.array([PAPER, soft(AMBER, 0.35), soft(VIOLET, 0.45), DIM], float)
CLASS_INK = np.array([dark(TEAL, 0.25), dark(MAGENTA, 0.2)], float)


def ridge_fit(a, target, gamma):
    """The study's kernel-ridge classifier: dual weights for labels in {0, 1}."""
    return np.linalg.solve(toy2d.kernel(a, a, gamma) + RIDGE * np.eye(len(a)), 2 * target - 1)


def cv_correct(x, y, gammas, folds=5):
    """``[setting, fold, image]``: was the image, held out in that fold, classified right?"""
    out = np.zeros((len(gammas), folds, len(x)), bool)
    for g, gamma in enumerate(gammas):
        for fold in range(folds):
            valid = np.arange(len(x)) % folds == fold
            alpha = ridge_fit(x[~valid], y[~valid], gamma)
            out[g, fold, valid] = (toy2d.kernel(x[valid], x[~valid], gamma) @ alpha > 0) == y[valid]
    return out


def inner_scores(x, y, gammas, folds=5, inner=4):
    """``[outer fold, setting]``: mean accuracy over the inner folds of the
    images outside that outer fold (what nested selection chooses by)."""
    out = np.zeros((folds, len(gammas)))
    for fold in range(folds):
        idx = np.flatnonzero(np.arange(len(x)) % folds != fold)
        for g, gamma in enumerate(gammas):
            acc = []
            for k in range(inner):
                held = np.arange(len(idx)) % inner == k
                iv, it = idx[held], idx[~held]
                alpha = ridge_fit(x[it], y[it], gamma)
                acc.append(np.mean((toy2d.kernel(x[iv], x[it], gamma) @ alpha > 0) == y[iv]))
            out[fold, g] = np.mean(acc)
    return out


def thumbs(x, labels, roles, cols):
    """Pooled 7 x 7 images as one sheet of thumbnails, ``cols`` across: strokes in
    their class's ink, on paper tinted by each image's role (:data:`ROLE_PAPER`)."""
    n = len(x)
    v = np.clip(np.asarray(x, float).reshape(n, 7, 7) * 1.5, 0, 1)[..., None]
    cell = ROLE_PAPER[roles][:, None, None] * (1 - v) + CLASS_INK[labels][:, None, None] * v
    full = np.empty((n, 8, 8, 3))
    full[:] = 0.80                 # the gaps between thumbnails
    full[:, :7, :7] = cell
    return full.reshape(n // cols, cols, 8, 8, 3).transpose(0, 2, 1, 3, 4).reshape(
        n // cols * 8, cols * 8, 3)


class CrossValidation(Classical):
    """Five folds taking turns as the held-out fifth, every setting scored on
    every fold, the best-of-six shortcut against nested selection, and what
    200 untouched images then say."""

    TITLE = "Cross-validation"
    PHASES = (("folds", 9.0), ("scores", 14.0), ("the leak", 9.0), ("nested", 14.0),
              ("new images", 9.0))
    COLS = 20
    SHEET = (0.285, 0.59, 440.0, 220.0)

    def setup(self):
        d, s = kept("cross_validation_mnist"), self.study
        self.x, self.y, self.gammas = d["x"], d["y"], d["gammas"]
        self.scores = np.asarray(s.panels[2].data["values"], float)
        self.outer = np.asarray(s.panels[3].data["values"], float)
        self.chosen = d["selected"]
        self.test_accuracy = float(d["test_accuracy"])
        self.optimistic, self.nested = float(self.scores.max(0).mean()), float(self.outer.mean())
        self.fold = np.arange(len(self.x)) % 5
        self.correct = cv_correct(self.x, self.y, self.gammas)
        self.test_x, self.test_y = d["test_x"], d["test_y"]
        gamma = float(d["selected_gamma"])
        self.final = int(np.argmin(np.abs(self.gammas - gamma)))
        self.test_right = (toy2d.kernel(self.test_x, self.x, gamma) @ d["weights_alpha"] > 0) \
            == self.test_y

    def subtitle_for(self, k):
        def pc(v):
            return number(100 * v)
        return [
            "200 real images of threes and eights, dealt into five folds. Each fold takes a turn "
            "as the 40 held out; the model is fitted on the other 160",
            "the model has one setting to choose, six candidates. Each is fitted five times and "
            "scored on the fold it did not see: accuracies from %s%% to %s%%" % (
                pc(self.scores.min()), pc(self.scores.max())),
            "the shortcut: keep the best of the six scores on each fold. That averages %s%%, but "
            "the winner was picked on the very images that scored it" % pc(self.optimistic),
            "nested: lock one fold away, let the other four choose the setting, then score it on "
            "the locked fold. The five honest scores average %s%%" % pc(self.nested),
            "on 200 new images the chosen model is right %s%% of the time. The nested estimate "
            "said %s%%; the shortcut said %s%%, %s points more" % (
                pc(self.test_accuracy), pc(self.nested), pc(self.optimistic),
                number(100 * (self.optimistic - self.nested))),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _sheet(self, sc, op, roles, at=None, x=None, y=None, wrong=()):
        """The images as a sheet of thumbnails; ``wrong``: indices ringed in red."""
        fx, fy, w, h = at or self.SHEET
        x, y = (self.x, self.y) if x is None else (x, y)
        sheet = thumbs(x, y, roles, self.COLS).astype(np.float32)
        draw_image(self, sc, self.P(fx, fy), sheet, w, h, op, grid=0)
        dw, dh = self.frac(w, h)
        rows = len(x) // self.COLS
        for i in wrong:
            cx = fx - dw / 2 + (i % self.COLS * 8 + 3.5) / (self.COLS * 8) * dw
            cy = fy - dh / 2 + (i // self.COLS * 8 + 3.5) / (rows * 8) * dh
            self.rect(sc, self.P(cx, cy), w / self.COLS, h / rows, RED, op, border=RED,
                      border_w=2.4, radius=2, lift=0.9, fill_op=0.0)
        return fx - dw / 2, fy + dh / 2

    def _classes(self, sc, op, y=0.335, x=0.055):
        self.text(sc, x, y, "threes", op, anchor="lm", ink=dark(TEAL), size=0.034)
        self.text(sc, x + 0.085, y, "eights", op, anchor="lm", ink=dark(MAGENTA), size=0.034)

    def _matrix(self, sc, op, filled, ring=(), now=None):
        """The six-by-five table of scores, its first ``filled`` cells shown."""
        x0, y0, dx, dy = 0.635, 0.42, 0.058, 0.072
        self.text(sc, x0 - 0.05, y0 - dy, "setting", op, anchor="rm")
        for f in range(5):
            self.text(sc, x0 + f * dx, y0 - dy, "fold %d" % (f + 1) if f == 0 else str(f + 1), op)
        for g in range(6):
            self.text(sc, x0 - 0.05, y0 + g * dy, number(self.gammas[g]), op, anchor="rm",
                      strong=True, size=0.034)
            for f in range(5):
                n = g * 5 + f
                if n >= filled:
                    self.box(sc, x0 + f * dx, y0 + g * dy, 50, 31, PAPER, op * 0.5, border=DIM)
                    continue
                v = self.scores[g, f]
                span = max(self.scores.max() - self.scores.min(), 1e-9)
                fill = soft(GREEN, 0.85 - 0.6 * (v - self.scores.min()) / span)
                hot = (g, f) in ring or n == now
                self.box(sc, x0 + f * dx, y0 + g * dy, 50, 31, fill, op, number(100 * v),
                         border=AMBER if hot else AXIS, size=0.032)
                if hot:
                    self.rect(sc, self.P(x0 + f * dx, y0 + g * dy), 52, 33, AMBER, op,
                              border=AMBER, border_w=2.6, radius=6, lift=0.9, fill_op=0.0)
        return x0, y0, dx, dy

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        k, _ = self.turn(u, 5, 1.05)
        at = (0.36, 0.61, 540.0, 270.0)
        self._sheet(sc, op, (self.fold == k).astype(int), at)
        self._classes(sc, op, 0.31, 0.08)
        x = 0.70
        self.text(sc, x, 0.40, "fold %d of 5" % (k + 1), op, anchor="lm", strong=True, size=0.046)
        self.box(sc, x + 0.012, 0.50, 22, 22, soft(AMBER, 0.35), op, border=AMBER)
        self.text(sc, x + 0.035, 0.50, "40 held out", op, anchor="lm", ink=dark(AMBER),
                  size=0.036)
        self.box(sc, x + 0.012, 0.57, 22, 22, PAPER, op, border=AXIS)
        self.text(sc, x + 0.035, 0.57, "160 to fit on", op, anchor="lm", size=0.036)
        self.lines(sc, x, 0.68, ["every image is held", "out exactly once", "",
                                 "shown as the model sees", "them: 7 x 7 pixels"], op,
                   step=0.046, size=0.034)

    def phase_2(self, sc, u, op):
        q = u / 0.45 * 5 if u < 0.45 else 5 + (u - 0.45) / 0.5 * 25
        n = int(min(q, 29))
        g, f = divmod(n, 5)
        wrong = np.flatnonzero((self.fold == f) & ~self.correct[g, f])
        left, bottom = self._sheet(sc, op, (self.fold == f).astype(int), wrong=wrong)
        self._classes(sc, op)
        right = int(self.correct[g, f].sum())
        self.lines(sc, left, bottom + 0.05, [
            ("setting %s, fold %d held out" % (number(self.gammas[g]), f + 1), None),
            ("%d of 40 right: %s%%" % (right, number(100 * right / 40)), dark(GREEN))],
            op, step=0.052, size=0.036)
        self._matrix(sc, op, min(int(q) + 1, 30) if q < 30 else 30, now=n if q < 30 else None)

    def phase_3(self, sc, u, op):
        best = [(int(np.argmax(self.scores[:, f])), f) for f in range(5)]
        shown = int(np.clip(u / 0.5 * 5 + 0.5, 0, 5))
        x0, y0, dx, dy = self._matrix(sc, op, 30, ring=best[:shown])
        for f in range(shown):
            self.text(sc, x0 + f * dx, y0 + 6 * dy + 0.01, number(100 * self.scores[:, f].max()),
                      op, ink=dark(AMBER), size=0.034)
        self.text(sc, x0 - 0.05, y0 + 6 * dy + 0.01, "best", op, anchor="rm", ink=dark(AMBER))
        self.lines(sc, 0.06, 0.38, ["the tempting shortcut", ], op, size=0.040, step=0.06)
        self.lines(sc, 0.06, 0.46, ["on each fold, report the best", "of the six settings"], op,
                   step=0.05, size=0.036)
        late = op * smoothstep(0.5, 0.65, u)
        self.text(sc, 0.06, 0.62, "average of the best: %s%%" % number(100 * self.optimistic),
                  late, anchor="lm", ink=dark(AMBER), size=0.042)
        self.lines(sc, 0.06, 0.72, ["those 40 images chose the winner", "and then graded it:",
                                    "the score has seen the answers"],
                   op * smoothstep(0.65, 0.8, u), step=0.048, size=0.034)

    def phase_4(self, sc, u, op):
        k, w = self.turn(u, 5, 1.18)
        inner = min(int(w / 0.6 * 4), 3)
        roles = (self.fold == k).astype(int)
        idx = np.flatnonzero(self.fold != k)
        if w < 0.6:
            roles[idx[np.arange(len(idx)) % 4 == inner]] = 2
        left, bottom = self._sheet(sc, op, roles)
        self._classes(sc, op)
        self.lines(sc, left, bottom + 0.05, [
            ("fold %d locked away" % (k + 1), dark(AMBER)),
            ("inner fold %d of 4 scores the six settings" % (inner + 1) if w < 0.6 else
             "the chosen setting is scored on the locked fold", dark(VIOLET) if w < 0.6 else None)],
            op, step=0.052, size=0.034)
        x = 0.56
        self.text(sc, x, 0.335, "locked fold", op, anchor="lm")
        self.text(sc, x + 0.15, 0.335, "chosen", op, anchor="lm", ink=dark(VIOLET))
        self.text(sc, x + 0.27, 0.335, "its score", op, anchor="lm", ink=dark(AMBER))
        for r in range(k + 1):
            y = 0.41 + r * 0.072
            done = r < k or w >= 0.6
            self.text(sc, x, y, "fold %d" % (r + 1), op, anchor="lm", strong=True, size=0.036)
            if done:
                self.text(sc, x + 0.15, y, number(self.chosen[r]), op, anchor="lm",
                          ink=dark(VIOLET), size=0.036)
            if r < k or w >= 0.8:
                self.hbar(sc, x + 0.27, y, self.span(0.10)[0] * self.outer[r], AMBER, op,
                          value="%s%%" % number(100 * self.outer[r]), thick=24.0)
        self.text(sc, x, 0.80, "average of the five: %s%%" % number(100 * self.nested),
                  op * smoothstep(0.84, 0.89, u), anchor="lm", strong=True, size=0.040)

    def phase_5(self, sc, u, op):
        wrong = np.flatnonzero(~self.test_right)
        shown = int(len(wrong) * smoothstep(0.1, 0.45, u))
        left, bottom = self._sheet(sc, op, np.zeros(len(self.test_x), int), x=self.test_x,
                                   y=self.test_y, wrong=wrong[:shown])
        self.text(sc, left, 0.335, "200 new images, never used above", op, anchor="lm",
                  strong=True, size=0.034)
        self.lines(sc, left, bottom + 0.05, [
            ("setting %s, fitted on all 200 old ones" % number(self.gammas[self.final]), None),
            ("%d wrong, %d right" % (len(wrong), int(self.test_right.sum())), dark(RED))],
            op, step=0.052, size=0.034)
        x, full = 0.56, self.span(0.30)[0]
        rows = (("best of six, per fold", AMBER, self.optimistic),
                ("nested estimate", VIOLET, self.nested),
                ("the new images", GREEN, self.test_accuracy))
        for r, (name, ink, v) in enumerate(rows):
            y = 0.40 + r * 0.14
            self.text(sc, x, y - 0.048, name, op, anchor="lm", ink=dark(ink), size=0.034)
            self.hbar(sc, x, y, full * (v - 0.85) / 0.15, ink, op, value="%s%%" % number(100 * v),
                      u=(u - 0.1 * r) / 0.4, thick=28.0, track=full)
        self.text(sc, x, 0.80, "bars start at 85%", op, anchor="lm")


# ====================================================================== 69 svm-and-kernels
def linear_svm_path(x, y, c=10.0, steps=600):
    """:func:`toy2d.linear_svm` again, keeping ``(w, b)`` after every update."""
    p = dict(w=np.zeros(x.shape[1]), b=np.array(0.0))
    opt = toy2d.Adam(p, 0.01)
    path = []
    for _ in range(steps):
        _, g = toy2d.linear_svm_loss(p, x, 2 * y - 1, c)
        opt.step(p, g)
        path.append(np.r_[p["w"], p["b"]])
    return np.array(path)


class SvmAndKernels(Classical):
    """A soft-margin line settling between real threes and eights, the same
    points lifted by their squared radius so a flat cut becomes a circle, and
    the RBF kernel's boundary as C changes."""

    TITLE = "SVMs and kernels"
    PHASES = (("points", 8.0), ("the street", 13.0), ("lift", 14.0), ("kernel", 13.0),
              ("test", 9.0))
    R = (-2.5, 2.5)
    CLASS = (BLUE, RED)                # threes, eights
    LINEAR, LIFTED, RBF, HOLD = OCHRE, VIOLET, GREEN, AMBER
    ZS = 0.36                          # how tall the lift is drawn

    def setup(self):
        d, s = kept("svm_mnist"), self.study
        self.x, self.y = d["x"], d["y"]
        self.inks = [self.CLASS[int(v)] for v in self.y]
        self.test, self.test_y = d["data_test"], (d["data_test_labels"] == 8).astype(int)
        self.images = pixels("train", d["data_train_indices"])
        self.trail = linear_svm_path(self.x, self.y)
        self.w, self.b = d["linear_w"], float(d["linear_b"])
        self.lw, self.lb = d["lifted_w"], float(d["lifted_b"])
        self.cs = KEPT["svm_mnist"]["cs"]
        self.support = np.asarray(s.panels[3].data["values"], int)
        self.acc = d["acc"]
        self.linear_acc = float(s.metadata["linear_accuracy"])
        self.lifted_acc = float(s.metadata["lifted_accuracy"])
        self.models = [dict(x=d["rbf%d_x" % i], y=d["rbf%d_y" % i], a=d["rbf%d_a" % i],
                            b=float(d["rbf%d_b" % i]), gamma=float(d["rbf%d_gamma" % i]))
                       for i in range(4)]
        self.rbf_edges = [contour(lambda q, m=m: toy2d.svm_predict(m, q), self.R, self.R, 64)
                          for m in self.models]
        self.lifted_edge = contour(self.lifted_score, self.R, self.R, 90)
        margin = (2 * self.y - 1) * (self.x @ self.w + self.b)
        self.in_street, self.wrong_side = margin < 1, margin < 0
        self.examples = [int(i) for c in (0, 1) for i in np.flatnonzero(self.y == c)[[2, 9, 17]]]

    def lifted_score(self, q):
        return q @ self.lw[:2] + self.lw[2] * (q * q).sum(1) + self.lb

    def subtitle_for(self, k):
        return [
            "200 real images of threes and eights, each squeezed to two numbers so it can be "
            "drawn as a point. The two kinds overlap: no line separates them cleanly",
            "a support vector machine looks for the widest street between the kinds. Here %d of "
            "the 200 points end up in the street or across it, and only they set the line" % (
                int(self.in_street.sum())),
            "add a third number to every point, its squared distance from the centre. A flat cut "
            "through the lifted points comes back down as a circle",
            "a kernel does such lifting without writing it out, and the boundary can curl. "
            "C sets what a misplaced point costs: %d points hold the boundary at C = %s, %d at %s"
            % (self.support[0], number(self.cs[0]), self.support[2], number(self.cs[2])),
            "on 200 new images: the line is right %s%% of the time, the circle %s%%, the kernel "
            "boundary %s%%" % (number(100 * self.linear_acc), number(100 * self.lifted_acc),
                               number(100 * self.acc[2])),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _plane(self, sc, op, cx=0.29, cy=0.64, side=296.0, pts=None, inks=None, r=3.6, upto=None):
        pane = self.square(cx, cy, side, self.R, self.R)
        self.backdrop(sc, pane, op, pad=0.0)
        self.dots(sc, pane, self.x if pts is None else pts, self.inks if inks is None else inks,
                  op, r=r, upto=upto)
        return pane

    def _legend(self, sc, x, y, op):
        self.disk(sc, self.P(x, y), 6, BLUE, op)
        self.text(sc, x + 0.015, y, "a three", op, anchor="lm", ink=dark(BLUE), size=0.034)
        self.disk(sc, self.P(x + 0.11, y), 6, RED, op)
        self.text(sc, x + 0.125, y, "an eight", op, anchor="lm", ink=dark(RED), size=0.034)

    def _line(self, sc, pane, w, b, level, ink, op, width):
        """The straight line ``w.x + b = level`` across the pane."""
        n2 = float(w @ w)
        if n2 < 1e-9:
            return
        p0, along = w * (level - b) / n2, np.array([-w[1], w[0]]) / np.sqrt(n2)
        t = np.linspace(-4.5, 4.5, 140)
        self.path(sc, pane, p0[0] + along[0] * t, p0[1] + along[1] * t, ink, op, width)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        pane = self._plane(sc, op, 0.70, 0.64, 296.0,
                           upto=int(len(self.x) * smoothstep(0.55, 0.95, u)))
        self._legend(sc, 0.55, 0.30, op)
        for n, i in enumerate(self.examples):
            fx, fy = 0.11 + 0.135 * (n % 3), 0.46 + 0.29 * (n // 3)
            ink = self.CLASS[int(self.y[i])]
            self.digit(sc, self.images[i], fx, fy, 96, op, frame=ink)
            self.text(sc, fx, fy + 0.125, "(%.1f, %.1f)" % tuple(self.x[i]), op, ink=dark(ink),
                      size=0.032)
            at = pane.at(*self.x[i])
            start = 0.08 + 0.07 * n
            self.pulse(sc, (fx, fy), at, (u - start) / 0.22, ink, op, bend=0.06, r=6)
            if u > start + 0.2:
                self.disk(sc, self.P(*at), 6.5, ink, op, ring=AMBER, ring_w=2.0, lift=1.4)

    def phase_2(self, sc, u, op):
        pane = self._plane(sc, op)
        done = smoothstep(0.02, 0.62, u) ** 1.6
        step = int(np.clip(done * len(self.trail), 6, len(self.trail))) - 1
        w, b = self.trail[step][:2], float(self.trail[step][2])
        for level in (-1.0, 1.0):
            self._line(sc, pane, w, b, level, self.HOLD, op * 0.9, 1.8)
        self._line(sc, pane, w, b, 0.0, self.LINEAR, op, 3.4)
        held = op * smoothstep(0.64, 0.74, u)
        if held > 0.01:
            m = (2 * self.y - 1) * (self.x @ w + b)
            self.dots(sc, pane, self.x[m < 1], [k for k, v in zip(self.inks, m < 1) if v], held,
                      r=3.6, ring=self.HOLD)
        self._legend(sc, 0.14, 0.30, op)
        x = 0.54
        self.text(sc, x, 0.34, "update %d of %d" % (step + 1, len(self.trail)), op, anchor="lm",
                  strong=True, size=0.040)
        width = 2 / max(float(np.linalg.norm(w)), 1e-9)
        self.lines(sc, x, 0.43, [("the line", dark(self.LINEAR)),
                                 ("the street's edges, %.2f apart" % width, dark(self.HOLD))],
                   op, step=0.052, size=0.036)
        self.lines(sc, x, 0.58, [
            ("%d points are in the street or" % int(self.in_street.sum()), None),
            ("across it (ringed): move one of", None),
            ("these and the line moves", None),
            ("%d of them are on the wrong side" % int(self.wrong_side.sum()), None),
            ("the other %d could vanish unnoticed" % int((~self.in_street).sum()), None)],
            held, step=0.050, size=0.034)

    def _view(self, pts, tilt):
        """3-D points as the tilted camera sees them (``tilt`` 0: from straight above)."""
        e, a = np.radians(90 - 63 * tilt), np.radians(24 * tilt)
        x, y, z = np.asarray(pts, float).T
        depth = x * np.sin(a) + y * np.cos(a)
        return np.column_stack((x * np.cos(a) - y * np.sin(a),
                                depth * np.sin(e) + z * self.ZS * np.cos(e))), depth

    def phase_3(self, sc, u, op):
        tilt = float(smoothstep(0.10, 0.36, u) - smoothstep(0.74, 0.94, u))
        pane = Pane(((0.05, 0.60), (0.30, 0.91)), (-4.4, 4.4),
                    (-2.75 + 1.35 * tilt, 2.75 + 1.35 * tilt))
        r2 = (self.x ** 2).sum(1)
        lifted = np.column_stack((self.x, r2 * tilt))
        top, depth = self._view(lifted, tilt)
        ground, _ = self._view(np.column_stack((self.x, 0 * r2)), tilt)
        c = self.R[1]
        corners, _ = self._view([(-c, -c, 0), (c, -c, 0), (c, c, 0), (-c, c, 0), (-c, -c, 0)], tilt)
        self.path(sc, pane, corners[:, 0], corners[:, 1], AXIS, op, 1.4)
        cut = op * smoothstep(0.42, 0.52, u)
        if tilt > 0.02:
            for g, t in zip(ground, top):
                sc.line(self.P(*pane.at(*g), 0.6), self.P(*pane.at(*t), 0.6), width=1.0,
                        color=DIM, opacity=op * 0.8 * tilt, bias=0.2)
            # the flat cut, as a mesh where it passes among the points
            grid, fine = np.linspace(-c, c, 9), np.linspace(-c, c, 40)
            for const in grid:
                fixed = np.full_like(fine, const)
                for xs, ys in ((fixed, fine), (fine, fixed)):
                    zs = -(self.lb + self.lw[0] * xs + self.lw[1] * ys) / self.lw[2]
                    ok = (zs >= 0) & (zs <= 7.0)
                    v, _ = self._view(np.column_stack((xs, ys, zs * tilt)), tilt)
                    v[~ok] = 99.0
                    self.path(sc, pane, v[:, 0], v[:, 1], self.LIFTED, cut * 0.8 * tilt, 1.3)
        for i in np.argsort(-depth):
            self.disk(sc, self.P(*pane.at(*top[i])), 3.8, self.inks[i], op, lift=1.0)
        if cut > 0.01:                 # where the cut meets the lifted points' surface
            for a, b in self.lifted_edge:
                va, _ = self._view([(a[0], a[1], (a @ a) * tilt)], tilt)
                vb, _ = self._view([(b[0], b[1], (b @ b) * tilt)], tilt)
                sc.line(self.P(*pane.at(*va[0]), 1.2), self.P(*pane.at(*vb[0]), 1.2), width=3.4,
                        color=self.LIFTED, opacity=cut, bias=0.3)
        x = 0.64
        self.lines(sc, x, 0.34, ["a third number for each point", ], op, size=0.036, step=0.05)
        self.text(sc, x, 0.40, "height = x² + y²", op, anchor="lm", strong=True, size=0.042)
        self.lines(sc, x, 0.48, ["points far from the centre", "rise the most"],
                   op * smoothstep(0.12, 0.3, u), step=0.048, size=0.034)
        self.lines(sc, x, 0.61, [("a flat cut through the lifted", dark(self.LIFTED)),
                                 ("points, fitted as before", dark(self.LIFTED))],
                   cut, step=0.048, size=0.034)
        self.lines(sc, x, 0.74, ["seen from above again,", "the flat cut is a circle"],
                   op * smoothstep(0.78, 0.92, u), step=0.048, size=0.034)

    def phase_4(self, sc, u, op):
        k, w = self.turn(u, 4, 1.03)
        pane = self._plane(sc, op, r=3.0)
        self.outline(sc, pane, self.rbf_edges[k], self.RBF, op, 3.0)
        holds = self.models[k]["a"] > 1e-6
        self.dots(sc, pane, self.x[holds], [c for c, h in zip(self.inks, holds) if h],
                  op * smoothstep(0.15, 0.4, w), r=3.0, ring=self.HOLD)
        self._legend(sc, 0.14, 0.30, op)
        x = 0.54
        self.text(sc, x, 0.335, "C", op, anchor="lm", strong=True, size=0.036)
        self.text(sc, x + 0.07, 0.335, "points holding the boundary", op, anchor="lm",
                  ink=dark(self.HOLD))
        full = self.span(0.20)[0]
        for r in range(k + 1):
            y = 0.415 + r * 0.085
            self.text(sc, x, y, number(self.cs[r]), op, anchor="lm", strong=True, size=0.040)
            self.hbar(sc, x + 0.07, y, full * self.support[r] / 200.0, self.HOLD, op,
                      value="%d of 200" % self.support[r], u=1.0 if r < k else w / 0.4,
                      thick=28.0, track=full)
        self.lines(sc, x, 0.77, [("the kernel boundary at C = %s" % number(self.cs[k]),
                                  dark(self.RBF)),
                                 ("small C forgives misplaced points;", None),
                                 ("large C bends to fit each one", None)],
                   op, step=0.048, size=0.034)

    def phase_5(self, sc, u, op):
        pane = self._plane(sc, op, pts=self.test, inks=[self.CLASS[int(v)] for v in self.test_y],
                           r=3.0)
        self._line(sc, pane, self.w, self.b, 0.0, self.LINEAR, op, 3.0)
        self.outline(sc, pane, self.lifted_edge, self.LIFTED, op * smoothstep(0.15, 0.3, u), 3.0)
        self.outline(sc, pane, self.rbf_edges[2], self.RBF, op * smoothstep(0.3, 0.45, u), 3.0)
        self.text(sc, 0.135, 0.30, "200 new images", op, anchor="lm", strong=True, size=0.034)
        self._legend(sc, 0.32, 0.30, op)
        x, full = 0.54, self.span(0.30)[0]
        rows = (("the line", self.LINEAR, self.linear_acc),
                ("the circle (lifted)", self.LIFTED, self.lifted_acc),
                ("the kernel, C = %s" % number(self.cs[2]), self.RBF, float(self.acc[2])))
        self.text(sc, x, 0.335, "right on the new images", op, anchor="lm", strong=True,
                  size=0.036)
        for r, (name, ink, v) in enumerate(rows):
            y = 0.47 + r * 0.14
            self.text(sc, x, y - 0.05, name, op, anchor="lm", ink=dark(ink), size=0.034)
            self.hbar(sc, x, y, full * v, ink, op, value="%s%%" % number(100 * v),
                      u=(u - 0.15 * r) / 0.3, thick=28.0, track=full)


# ====================================================================== 70 gradient-boosting
def as_tree(node):
    """A tree as :func:`toy2d.tree_fit` returns it, from its JSON form."""
    if not isinstance(node, list):
        return float(node)
    return (int(node[0]), float(node[1]), as_tree(node[2]), as_tree(node[3]))


def running(trees, x, start, rate):
    """Predictions for ``x`` after 0, 1, ... trees: ``[trees + 1, images]``."""
    out = [np.full(len(x), float(start))]
    for tree in trees:
        out.append(out[-1] + rate * toy2d.tree_predict(tree, x))
    return np.array(out)


def spots(tree, x0, x1, y0, dy, path=()):
    """Where to draw each node: ``{path: (fx, fy, node)}`` (path: 0 left, 1 right)."""
    out = {path: ((x0 + x1) / 2, y0, tree)}
    if not isinstance(tree, float):
        mid = (x0 + x1) / 2
        out.update(spots(tree[2], x0, mid, y0 + dy, dy, path + (0,)))
        out.update(spots(tree[3], mid, x1, y0 + dy, dy, path + (1,)))
    return out


def route(tree, row):
    """The path one image (its 49 inputs) takes down ``tree``."""
    path = ()
    while not isinstance(tree, float):
        go = int(row[tree[0]] > tree[1])
        path, tree = path + (go,), tree[2 + go]
    return path


class GradientBoosting(Classical):
    """Depth-three trees added one at a time, each fitted to what the ones
    before left wrong; then the same trees at full strength, and averaged
    instead of chained."""

    TITLE = "Gradient boosting"
    PHASES = (("the task", 9.0), ("a tree", 16.0), ("sixty", 11.0), ("the rate", 10.0),
              ("bagging", 11.0))
    SLOW, FAST, BAG, LEFT = BLUE, AMBER, GREEN, RED
    SHOWN = 30

    def setup(self):
        d, s = kept("boosting_mnist"), self.study
        self.x, self.y, self.xt, self.yt = d["train_x"], d["train_y"], d["test_x"], d["test_y"]
        self.start = float(d["initial_prediction"])
        self.trees = [as_tree(t) for t in json.loads(str(d["trees"]))]
        self.rate = KEPT["boosting_mnist"]["rates"][0]
        self.fit = running(self.trees, self.x, self.start, self.rate)
        self.held = running(self.trees, self.xt, self.start, self.rate)
        self.held_fast = running([as_tree(t) for t in json.loads(str(d["fast_trees"]))], self.xt,
                                 self.start, 1.0)
        self.train_mse, self.test_mse = (np.asarray(v, float) for v in s.panels[1].data["ys"])
        self.bag_mse = np.asarray(s.panels[2].data["ys"][1], float)
        self.fast_mse = d["fast_loss"]
        self.final = [float(v) for v in s.panels[3].data["values"]]     # slow, fast, bagging
        self.count = np.arange(1, len(self.trees) + 1)
        self.pick = np.argsort(self.y)[np.linspace(4, len(self.y) - 5, self.SHOWN).astype(int)]
        self.pick_t = np.argsort(self.yt)[np.linspace(3, len(self.yt) - 4, self.SHOWN).astype(int)]
        self.images = pixels("test", d["test_indices"][:5]).reshape(-1, 28, 28)
        self.yr = (0.02, 0.31)

    def subtitle_for(self, k):
        return [
            "the task: from the top half of a handwritten digit, predict how much ink is in the "
            "bottom half. 500 images to learn from, 300 held out",
            "start by guessing the average for everyone. Fit a small tree to what is still wrong, "
            "add %.0f%% of its correction, and repeat on what is wrong now" % (100 * self.rate),
            "sixty trees, each fixing a little of what is left: the error on the training images "
            "falls from %.5f to %.5f, on held-out images to %.5f" % (
                self.train_mse[0], self.train_mse[-1], self.test_mse[-1]),
            "adding each tree at full strength instead of %.0f%% learns faster at first but ends "
            "worse on held-out images: %.5f against %.5f" % (
                100 * self.rate, self.final[1], self.final[0]),
            "bagging grows the trees separately, each on a random resample, and averages them: "
            "%.5f held-out error after sixty, against boosting's %.5f" % (
                self.final[2], self.final[0]),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _sticks(self, sc, op, truth, preds, box=((0.57, 0.93), (0.37, 0.77)), title=""):
        """Images in order of their true value: the truth, each model's
        prediction ``[(values, ink)]``, and the gap between them."""
        pane = Pane(box, (-1, len(truth)), self.yr)
        self.axes(sc, pane, op, title, "ink in the bottom half", xticks=[])
        for i, t in enumerate(truth):
            for values, ink in preds:
                sc.line(self.P(*pane.at(i, t), 0.5), self.P(*pane.at(i, values[i]), 0.5),
                        width=2.0, color=self.LEFT if len(preds) == 1 else ink, opacity=op * 0.8,
                        bias=0.3)
                self.disk(sc, self.P(*pane.at(i, values[i])), 3.6, ink, op, lift=1.2)
            self.disk(sc, self.P(*pane.at(i, t)), 3.2, DIGIT_INK, op, lift=1.0)
        return pane

    def _tree(self, sc, op, tree, lit=()):
        """One fitted tree: each split as the top-half cell it tests, each leaf
        as its correction. ``lit``: a path to highlight. Returns the node spots."""
        at = spots(tree, 0.05, 0.50, 0.385, 0.135)
        for path, (fx, fy, node) in at.items():
            if path:
                px, py, _ = at[path[:-1]]
                on = lit[:len(path)] == path
                sc.line(self.P(px, py + 0.03), self.P(fx, fy - 0.035), width=3.0 if on else 1.6,
                        color=AMBER if on else AXIS, opacity=op, bias=0.2)
            if isinstance(node, float):
                self.box(sc, fx, fy, 47, 30, soft(RED if node > 0 else BLUE, 0.45), op,
                         "%+.2f" % node, border=AMBER if lit == path else AXIS)
            else:
                cells = np.zeros(49)
                cells[node[0]] = 1.0
                self.grid(sc, cells.reshape(7, 7), fx - 0.034, fy, 49, 28, op, hot=VIOLET,
                          vmax=0.6)
                limit = ("≤ %.3f" if node[1] < 0.01 else "≤ %.2f") % node[1]
                self.text(sc, fx - 0.002, fy, limit, op, anchor="lm")
        return at

    def _icon(self, sc, fx, fy, ink, op):
        """A small tree."""
        for dx in (-0.012, 0.012):
            sc.line(self.P(fx, fy - 0.018), self.P(fx + dx, fy + 0.018), width=2.4, color=ink,
                    opacity=op, bias=0.2)
            self.disk(sc, self.P(fx + dx, fy + 0.018), 4.0, ink, op)
        self.disk(sc, self.P(fx, fy - 0.018), 4.0, ink, op)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        for n, img in enumerate(self.images):
            fx = 0.14 + 0.18 * n
            self.digit(sc, img, fx, 0.455, 112, op)
            dy = self.frac(0, 28)[1]
            self.tint(sc, fx, 0.455 - dy, 112, 56, VIOLET, op, 0.16)
            self.tint(sc, fx, 0.455 + dy, 112, 56, self.LEFT, op * smoothstep(0.25, 0.4, u), 0.22)
            seen = op * smoothstep(0.1 + 0.06 * n, 0.3 + 0.06 * n, u)
            self.grid(sc, self.xt[n].reshape(7, 7), fx, 0.665, 112, 56, seen, hot=VIOLET, vmax=1.0)
            self.text(sc, fx, 0.80, "ink below: %.2f" % self.yt[n],
                      op * smoothstep(0.45 + 0.05 * n, 0.55 + 0.05 * n, u), ink=dark(self.LEFT),
                      size=0.036)
        self.text(sc, 0.05, 0.31, "given: the top half", op, anchor="lm", ink=dark(VIOLET),
                  size=0.034)
        self.text(sc, 0.95, 0.31, "to predict: the ink in the bottom half", op, anchor="rm",
                  ink=dark(self.LEFT), size=0.034)
        self.text(sc, 0.5, 0.88, "the trees see the top half as 7 x 7 averages: 49 numbers",
                  op * smoothstep(0.3, 0.5, u), ink=dark(VIOLET), size=0.034)

    def phase_2(self, sc, u, op):
        k, w = self.turn(u, 4, 1.03)
        tree = self.trees[k]
        who = int(self.pick[(7 * k + 5) % self.SHOWN])
        slot = int(np.flatnonzero(self.pick == who)[0])
        path = route(tree, self.x[who])
        grown = smoothstep(0.0, 0.2, w)
        at = self._tree(sc, op * grown, tree, lit=path if w > 0.3 else ())
        move = smoothstep(0.62, 0.95, w)
        now = self.fit[k] + (self.fit[k + 1] - self.fit[k]) * move
        pane = self._sticks(sc, op, self.y[self.pick], [(now[self.pick], self.SLOW)],
                            title="%d of the 500 training images" % self.SHOWN)
        # one image goes down the tree, and its leaf's correction out to its prediction
        legs = [at[path[:n]][:2] for n in range(len(path) + 1)]
        s = (w - 0.26) / 0.24 * len(path)
        for n in range(len(path)):
            self.pulse(sc, legs[n], legs[n + 1], s - n, AMBER, op, r=6)
        self.pulse(sc, legs[-1], pane.at(slot, now[who]), (w - 0.5) / 0.14, AMBER, op,
                   bend=0.10, r=6)
        self.disk(sc, self.P(*pane.at(slot, now[who])), 6.0, self.SLOW, op, ring=AMBER, lift=1.4)
        self.text(sc, 0.05, 0.31, "tree %d of %d, fitted to the red gaps" % (
            k + 1, len(self.trees)), op, anchor="lm", strong=True, size=0.036)
        self.lines(sc, 0.57, 0.90, [("%s: the truth" % theme.word("black", "white"), None)],
                   op)
        self.text(sc, 0.72, 0.90, "blue: predicted", op, anchor="lm", ink=dark(self.SLOW))
        self.text(sc, 0.86, 0.90, "red: wrong", op, anchor="lm", ink=dark(self.LEFT))
        self.text(sc, 0.05, 0.90, "a split tests one cell of the top half; a leaf is a correction",
                  op, anchor="lm")

    def phase_3(self, sc, u, op):
        s = min(u / 0.85, 1.0)
        n = int(round(s * len(self.trees)))
        self._sticks(sc, op, self.y[self.pick], [(self.fit[n][self.pick], self.SLOW)],
                     title="%d of the 500 training images" % self.SHOWN)
        chart = self.chart(((0.10, 0.48), (0.37, 0.80)), self.count, np.r_[0.0, self.test_mse])
        self.axes(sc, chart, op, "trees added", "squared error")
        self.plot(sc, chart, self.count, self.train_mse, VIOLET, op, s, width=2.4)
        self.plot(sc, chart, self.count, self.test_mse, self.SLOW, op, s, width=2.8)
        self.text(sc, 0.30, 0.425, "held-out images", op, anchor="lm", ink=dark(self.SLOW))
        self.text(sc, 0.30, 0.467, "training images", op, anchor="lm", ink=dark(VIOLET))
        self.text(sc, 0.57, 0.89, "%d trees" % n, op, anchor="lm", strong=True, size=0.040)
        if n:
            self.text(sc, 0.70, 0.89, "training error %.5f" % self.train_mse[n - 1], op,
                      anchor="lm", ink=dark(VIOLET))

    def phase_4(self, sc, u, op):
        s = min(u / 0.85, 1.0)
        n = int(round(s * len(self.trees)))
        self._sticks(sc, op, self.yt[self.pick_t], [(self.held_fast[n][self.pick_t], self.FAST),
                                                     (self.held[n][self.pick_t], self.SLOW)],
                     title="%d of the 300 held-out images" % self.SHOWN)
        chart = self.chart(((0.10, 0.48), (0.37, 0.80)), self.count,
                           np.r_[0.0, self.test_mse, self.fast_mse])
        self.axes(sc, chart, op, "trees added", "squared error, held-out images")
        self.plot(sc, chart, self.count, self.fast_mse, self.FAST, op, s, width=2.8)
        self.plot(sc, chart, self.count, self.test_mse, self.SLOW, op, s, width=2.8)
        self.text(sc, 0.57, 0.89, "%d trees" % n, op, anchor="lm", strong=True, size=0.040)
        if n:
            self.text(sc, 0.70, 0.875, "%.0f%% of each: %.5f" % (
                100 * self.rate, self.test_mse[n - 1]), op, anchor="lm", ink=dark(self.SLOW))
            self.text(sc, 0.70, 0.918, "all of each: %.5f" % self.fast_mse[n - 1], op,
                      anchor="lm", ink=dark(self.FAST))

    def phase_5(self, sc, u, op):
        s = min(u / 0.6, 1.0)
        chart = self.chart(((0.10, 0.48), (0.37, 0.80)), self.count,
                           np.r_[0.0, self.test_mse, self.fast_mse, self.bag_mse])
        self.axes(sc, chart, op, "trees", "squared error, held-out images")
        for ys, ink in ((self.fast_mse, self.FAST), (self.bag_mse, self.BAG),
                        (self.test_mse, self.SLOW)):
            self.plot(sc, chart, self.count, ys, ink, op, s, width=2.8)
        x = 0.57
        self.text(sc, x, 0.33, "boosting: a chain, each fixes the last", op, anchor="lm",
                  ink=dark(self.SLOW), size=0.034)
        self.text(sc, x, 0.52, "bagging: side by side, then averaged", op, anchor="lm",
                  ink=dark(self.BAG), size=0.034)
        for n in range(5):
            fx = x + 0.025 + 0.062 * n
            self._icon(sc, fx, 0.41, self.SLOW, op)
            if n:
                self.arrow(sc, (fx - 0.044, 0.41), (fx - 0.02, 0.41), self.SLOW, op, width=2.0)
            self._icon(sc, fx, 0.59, self.BAG, op)
            self.arrow(sc, (fx, 0.615), (x + 0.15, 0.665), self.BAG, op * 0.8, width=1.6,
                       head=False)
        full = self.span(0.20)[0]
        top = max(self.final)
        rows = (("boosting, %.0f%%" % (100 * self.rate), self.SLOW, self.final[0]),
                ("boosting, 100%", self.FAST, self.final[1]),
                ("bagging", self.BAG, self.final[2]))
        for r, (name, ink, v) in enumerate(rows):
            self.hbar(sc, x + 0.14, 0.735 + r * 0.068, full * v / top, ink, op, name,
                      "%.5f" % v, u=(u - 0.45 - 0.06 * r) / 0.2, thick=24.0)


# ====================================================================== 71 gaussian-mixture-em
def ellipse(mu, cov, sigmas=2.0, n=40):
    """The ``sigmas`` contour of a 2-D Gaussian as ``(n + 1, 2)`` points."""
    values, vectors = np.linalg.eigh(cov)
    theta = np.linspace(0, 2 * np.pi, n + 1)
    unit = np.column_stack((np.cos(theta), np.sin(theta)))
    return mu + unit @ (vectors * np.sqrt(np.maximum(values, 0))).T * sigmas


class GaussianMixtureEM(Classical):
    """Four Gaussian shapes fitted to real digit images by EM: points shared
    out among the shapes, the shapes moving to their share, the likelihood
    rising every round; then a start that gets stuck, how often 300 starts do,
    and which way of starting helps."""

    TITLE = "Gaussian mixture, by EM"
    PHASES = (("the points", 8.0), ("E, then M", 15.0), ("100 rounds", 10.0),
              ("a bad start", 11.0), ("300 starts", 12.0))
    XR, YR = (-2.0, 3.8), (-2.5, 2.3)
    BOX = ((0.06, 0.46), (0.33, 0.91))
    FIRST, SECOND = DIGIT_INK, MAGENTA
    SHOWN = 400
    KINDS = ("random points", "points far apart (k-means++)", "a k-means solution")

    def setup(self):
        d, s = sx.gaussian_mixture_data(), self.study
        self.x, self.k = d["train"], int(d["starts"].shape[2])
        self.summary = sx.mixture_summary(d)
        self.chosen = [int(self.summary[key][0]) for key in ("best_start", "worst_start")]
        self.runs = [sx.em_history(self.x, d["starts"][0, i]) for i in self.chosen]
        inks = np.array(INKS[:self.k], float)
        for run in self.runs:
            run["inks"] = run["r"][:, :self.SHOWN] @ inks
            run["shapes"] = np.array([[ellipse(m, c) for m, c in zip(mus, covs)]
                                      for mus, covs in zip(run["mu"], run["cov"])])
        self.loss = [np.asarray(v, float) for v in s.panels[1].data["ys"]]
        self.held = [float(d["test_ll"][0, i]) for i in self.chosen]
        self.sorted = np.asarray(s.panels[2].data["ys"], float)       # [kind, start]
        self.reached = np.asarray(s.panels[3].data["values"], float)
        self.bad = np.asarray(self.summary["bad"], float)
        # what became of every start: 0 the best fit, 1 another fit, 2 clearly worse held out
        best = d["train_ll"] > d["train_ll"].max() - sx.NEAR_BEST
        worse = d["test_ll"] < self.summary["usual"] - sx.CLEARLY_WORSE
        self.fate = np.sort(np.where(best, 0, np.where(worse, 2, 1)), axis=1)
        self.rounds = np.arange(1, len(self.loss[0]) + 1)
        self.images = pixels("train", d["train_indices"][:self.SHOWN])
        labels = d["train_labels"][:self.SHOWN]
        self.examples = [int(np.flatnonzero(labels == c)[n])
                         for c, n in ((0, 1), (1, 1), (6, 1), (7, 1), (0, 2), (7, 2))]
        r0 = self.runs[0]["r"][0][:self.SHOWN]
        self.who = int(np.argmax(-(r0 * np.log(r0 + 1e-12)).sum(1)))
        self.pane = Pane(self.BOX, self.XR, self.YR)

    def subtitle_for(self, k):
        a, b = self.loss
        pc = ["%.0f" % (100 * v) for v in self.reached]
        bad = ["%.0f" % (100 * v) for v in self.bad]
        return [
            "%s real images of handwritten zeros, ones, sixes and sevens, each squeezed to two "
            "numbers and drawn as a point. No labels are given: the task is to describe the "
            "cloud with four blobs" % f"{len(self.x):,}",
            "E-step: share every point out among the four blobs, by how well each explains it. "
            "M-step: move and reshape each blob to fit its share. Then again",
            "every round the blobs explain the points a little better, never worse: the "
            "log-likelihood climbs from %.2f to %.2f in %d rounds" % (a[0], a[-1], len(a)),
            "the same method from other starting points can get stuck. This start ends at "
            "log-likelihood %.2f, not %.2f; on 600 held-out images, %.2f against %.2f" % (
                b[-1], a[-1], self.held[1], self.held[0]),
            "of %d random starts %s%% reach the best fit and %s%% end clearly worse. Seeding the "
            "centres far apart barely changes that (%s%%); starting from a k-means solution "
            "gives %s%%, and %s" % (
                self.sorted.shape[1], pc[0], bad[0], pc[1], pc[2],
                "none end clearly worse" if self.bad[2] == 0 else "%s%% end clearly worse"
                % bad[2]),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _cloud(self, sc, op, pane, inks, upto=None):
        self.backdrop(sc, pane, op, pad=0.0)
        self.dots(sc, pane, self.x[:self.SHOWN], inks, op, r=3.1, upto=upto)

    def _blobs(self, sc, op, pane, run, state, width=2.8, ink=None):
        """The blobs at a (fractional) state between rounds."""
        i = int(min(state, len(run["shapes"]) - 2))
        f = min(state - i, 1.0)
        shapes = run["shapes"][i] * (1 - f) + run["shapes"][i + 1] * f
        centres = run["mu"][i] * (1 - f) + run["mu"][i + 1] * f
        for j, (shape, c) in enumerate(zip(shapes, centres)):
            self.path(sc, pane, shape[:, 0], shape[:, 1], INKS[j] if ink is None else ink, op,
                      width)
            if ink is None and pane.holds(*c):
                self.disk(sc, self.P(*pane.at(*c)), 5.5, INKS[j], op, ring=WHITE, ring_w=1.6,
                          lift=1.5)
        return centres

    def _curves(self, sc, op, upto):
        """Log-likelihood by round; ``upto``: rounds drawn of each run (None: not drawn)."""
        chart = self.chart(((0.58, 0.93), (0.44, 0.77)), self.rounds, np.r_[self.loss])
        self.axes(sc, chart, op, "round", "log-likelihood of the points")
        for ys, ink, n in zip(self.loss, (self.FIRST, self.SECOND), upto):
            if n is not None:
                self.plot(sc, chart, self.rounds, ys, ink, op, (n - 1) / (len(ys) - 1), width=2.8)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        pane = Pane(((0.53, 0.93), (0.33, 0.91)), self.XR, self.YR)
        self._cloud(sc, op, pane, DIM, int(self.SHOWN * smoothstep(0.5, 0.9, u)))
        for n, i in enumerate(self.examples):
            fx, fy = 0.12 + 0.135 * (n % 3), 0.455 + 0.285 * (n // 3)
            self.digit(sc, self.images[i], fx, fy, 96, op)
            self.text(sc, fx, fy + 0.118, "(%.1f, %.1f)" % tuple(self.x[i]), op, size=0.032)
            at = pane.at(*self.x[i])
            start = 0.06 + 0.07 * n
            self.pulse(sc, (fx, fy), at, (u - start) / 0.22, AMBER, op, bend=0.06, r=6)
            if u > start + 0.2:
                self.disk(sc, self.P(*at), 5.5, AMBER, op, lift=1.4)

    def phase_2(self, sc, u, op):
        run = self.runs[0]
        k, w = self.turn(u, 3, 1.03)
        share = smoothstep(0.05, 0.38, w)
        before = run["inks"][k - 1] if k else np.tile(DIM, (self.SHOWN, 1))
        self._cloud(sc, op, self.pane, before * (1 - share) + run["inks"][k] * share)
        centres = self._blobs(sc, op, self.pane, run, k + smoothstep(0.55, 0.95, w))
        r, at = run["r"][k][self.who], self.pane.at(*self.x[self.who])
        e_step = w < 0.5
        for j in range(self.k):
            if r[j] > 0.02 and e_step:
                sc.line(self.P(*self.pane.at(*centres[j]), 0.95), self.P(*at, 0.95),
                        width=1.0 + 6.0 * r[j], color=INKS[j], opacity=op * share, bias=0.3)
        self.disk(sc, self.P(*at), 6.0, run["inks"][k][self.who], op, ring=DIGIT_INK, ring_w=2.2,
                  lift=1.6)
        x = 0.56
        self.text(sc, x, 0.335, "round %d of %d" % (k + 1, len(self.rounds)), op, anchor="lm",
                  strong=True, size=0.040)
        self.lines(sc, x, 0.41, ["E-step: share out each point" if e_step else
                                 "M-step: each blob moves to its share"], op, size=0.036)
        self.text(sc, x, 0.50, "the ringed point's shares", op, anchor="lm")
        full = self.span(0.26)[0]
        for j in range(self.k):
            self.hbar(sc, x + 0.005, 0.575 + j * 0.075, full * r[j], INKS[j], op,
                      value="%.0f%%" % (100 * r[j]), u=share if k == 0 else 1.0, thick=26.0,
                      track=full)

    def phase_3(self, sc, u, op):
        run = self.runs[0]
        state = 3 + (len(self.rounds) - 3) * smoothstep(0.0, 0.9, u) ** 2
        self._cloud(sc, op, self.pane, run["inks"][min(int(state), len(self.rounds) - 1)])
        self._blobs(sc, op, self.pane, run, state)
        self._curves(sc, op, (max(state, 1.0), None))
        n = int(state)
        self.text(sc, 0.58, 0.33, "round %d of %d" % (n, len(self.rounds)), op, anchor="lm",
                  strong=True, size=0.040)
        self.text(sc, 0.58, 0.90, "log-likelihood %.3f" % self.loss[0][n - 1], op, anchor="lm",
                  size=0.036)

    def phase_4(self, sc, u, op):
        run = self.runs[1]
        state = len(self.rounds) * smoothstep(0.0, 0.6, u) ** 2
        n = int(np.clip(state, 1, len(self.rounds)))
        self._cloud(sc, op, self.pane, run["inks"][n - 1])
        ended = op * smoothstep(0.64, 0.76, u)
        self._blobs(sc, ended * 0.85, self.pane, self.runs[0], len(self.rounds), 1.7, self.FIRST)
        self._blobs(sc, op, self.pane, run, state)
        self._curves(sc, op, (len(self.rounds), max(state, 1.0)))
        self.text(sc, 0.58, 0.33, "another random start, round %d" % n, op, anchor="lm",
                  ink=dark(self.SECOND), size=0.040)
        self.lines(sc, 0.58, 0.885, [("thin %s: the blobs of the best fit"
                                      % theme.word("black", "white"), None),
                                    ("same points, same method, a worse answer", None)],
                   ended, step=0.045)

    def phase_5(self, sc, u, op):
        cols, cell = 50, 7.0
        rows = self.fate.shape[1] // cols
        w, h = cols * cell, rows * cell
        paint = np.array([GREEN, OCHRE, RED], float)
        count = self.fate.shape[1]
        for row, name in enumerate(self.KINDS):
            y = 0.42 + 0.165 * row
            show = op * smoothstep(0.18 * row, 0.18 * row + 0.12, u)
            done = int(count * smoothstep(0.18 * row + 0.02, 0.18 * row + 0.2, u))
            rgb = np.tile(np.asarray(PAPER, float) * 0.94, (count, 1))
            rgb[:done] = paint[self.fate[row][:done]]
            self.text(sc, 0.07, y - 0.062, "%d starts from %s" % (count, name), show,
                      anchor="lm", strong=True, size=0.032)
            draw_image(self, sc, self.P(0.07 + self.frac(w / 2)[0], y), rgb.reshape(
                cols, rows, 3).transpose(1, 0, 2).astype(np.float32), w, h, show, grid=0.16)
            late = show * smoothstep(0.18 * row + 0.18, 0.18 * row + 0.26, u)
            self.text(sc, 0.50, y - 0.022, "%.0f%% reach the best fit" % (
                100 * self.reached[row]), late, anchor="lm", ink=dark(GREEN), size=0.038)
            self.text(sc, 0.50, y + 0.026, "%.0f%% end clearly worse" % (100 * self.bad[row]),
                      late, anchor="lm", ink=dark(RED), size=0.034)
        y = 0.895
        for x, ink, name in ((0.07, GREEN, "one square, one start: it ended at the best fit"),
                             (0.47, OCHRE, "at another fit"),
                             (0.63, RED, "clearly worse on held-out images")):
            self.box(sc, x + 0.008, y, 13, 13, ink, op)
            self.text(sc, x + 0.024, y, name, op, anchor="lm", size=0.030)


# ====================================================================== 72 curse-of-dimensionality
ball_in_box = sx.ball_in_box


class CurseOfDimensionality(Classical):
    """Real images described by their 49 pixels plus more and more coordinates
    that say nothing about them: the distances bunching up, the exact
    ball-in-a-box volumes, the nearest neighbour turning into a stranger, and
    how little more data buys back."""

    TITLE = "The curse of dimensionality"
    PHASES = (("distances", 10.0), ("add noise", 14.0), ("ball in box", 8.0),
              ("neighbours", 12.0), ("more data", 11.0))
    NEAR, FAR, NEW, REAL, NOISE = TEAL, MAGENTA, AMBER, GREEN, VIOLET
    BALL = (2, 4, 8, 16, 32, 49)
    SHOWN = 1000                       # training images the distances are taken to
    LEVEL = 2                          # the noise level shown beside the real coordinates

    def setup(self):
        d, s = sx.curse_data(), self.study
        self.dims = np.asarray(s.panels[0].data["x"], int)
        self.ratio = np.asarray(s.panels[0].data["ys"][0], float)
        self.accuracy, self.acc_lo, self.acc_hi = (np.asarray(v, float)
                                                   for v in s.panels[1].data["ys"])
        self.table = np.asarray(s.panels[2].data["values"], float)     # [level, size]
        self.sizes = np.asarray(sx.CURSE["sizes"], int)
        self.pca_dims = np.asarray(s.panels[3].data["x"], int)
        self.pca_accuracy = np.asarray(s.panels[3].data["ys"][0], float)
        self.hist, self.nearest, self.farthest = d["hist"], d["nearest"], d["farthest"]
        self.near_d, self.far_d = d["near_distance"], d["far_distance"]
        self.tiles = d["noise_tiles"]                                   # [block, image, 49]
        self.labels, self.test_labels = d["train_labels"], d["test_labels"]
        self.volume = np.array([ball_in_box(v) for v in self.BALL])
        self.images = pixels("train", np.arange(self.SHOWN))
        self.test_images = pixels("test", np.arange(16))
        self.pooled = sx.downsample(self.test_images)
        train = sx.downsample(self.images)
        self.rug = np.sqrt(np.maximum(sx.squared_distances(self.pooled[:4], train), 0))

    def subtitle_for(self, k):
        def pc(v):
            return "%.0f" % (100 * v)
        acc, table = self.accuracy, self.table
        enough = np.flatnonzero(table[1] >= table[0, 0])
        if len(enough):
            more = ("%s images described by the real coordinates alone do as well (%s%%) as %s "
                    "images with 49 shuffled coordinates added (%s%%)" % (
                        f"{self.sizes[0]:,}", pc(table[0, 0]),
                        f"{self.sizes[enough[0]]:,}", pc(table[1, enough[0]])))
        else:
            more = ("%s images described by the real coordinates alone are right %s%% of the "
                    "time; with 49 shuffled coordinates added, even %s images reach only %s%%" % (
                        f"{self.sizes[0]:,}", pc(table[0, 0]),
                        f"{self.sizes[-1]:,}", pc(table[1, -1])))
        return [
            "each image is described by 49 numbers, the ink in each cell of a 7 by 7 grid. Among "
            "1,000 training images a new image's nearest is far closer than its farthest: "
            "%s%% as far, on average" % pc(self.ratio[0]),
            "now add coordinates that say nothing about the image: real pixel values, dealt to "
            "the images at random. Distances bunch up: with %s added, the nearest image "
            "is %s%% as far as the farthest" % (
                f"{self.dims[-1] - 49:,}", pc(self.ratio[-1])),
            "exact geometry: the biggest ball that fits in a box fills %s%% of it in 2 "
            "dimensions, %s%% in 8, almost none in 49. The room is all in the corners" % (
                number(100 * self.volume[0]), number(100 * self.volume[2])),
            "by the 49 real coordinates the nearest image is the same digit %s%% of the time. "
            "With 49 shuffled ones added, %s%%; with %d, %s%%; with %s, %s%%. Guessing gets "
            "10%%" % (pc(acc[0]), pc(acc[1]), self.dims[self.LEVEL] - 49, pc(acc[self.LEVEL]),
                      f"{self.dims[-1] - 49:,}", pc(acc[-1])),
            "more data buys it back only slowly: " + more,
        ][k]

    def _framed(self, sc, img, fx, fy, size, ink, op, label="", note=""):
        self.digit(sc, img, fx, fy, size, op, frame=ink)
        if label:
            self.text(sc, fx, fy - self.frac(0, size / 2)[1] - 0.04, label, op, ink=dark(ink),
                      size=0.034)
        if note:
            self.text(sc, fx, fy + self.frac(0, size / 2)[1] + 0.04, note, op, size=0.034)

    def phase_1(self, sc, u, op):
        k, w = self.turn(u, 4, 1.12)
        near, far = int(self.nearest[0][k]), int(self.farthest[k])
        reach = smoothstep(0.1, 0.5, w)
        self._framed(sc, self.test_images[k], 0.13, 0.47, 100, self.NEW, op, "a new image")
        self.grid(sc, self.pooled[k].reshape(7, 7), 0.33, 0.47, 100, 100, op, hot=DIGIT_INK,
                  vmax=1.0)
        self.text(sc, 0.33, 0.335, "as 49 numbers", op, size=0.034)
        self.arrow(sc, (0.20, 0.47), (0.26, 0.47), AXIS, op)
        self._framed(sc, self.images[near], 0.60, 0.47, 100, self.NEAR, op * reach, "nearest",
                     "%.2f away" % self.near_d[k])
        self._framed(sc, self.images[far], 0.82, 0.47, 100, self.FAR, op * reach, "farthest",
                     "%.2f away" % self.far_d[k])
        # every training image's distance from the new one, as a mark on a line
        x0, x1, y = 0.10, 0.90, 0.745
        sc.line(self.P(x0, y, 0.5), self.P(x1, y, 0.5), width=1.2, color=AXIS, opacity=op,
                bias=0.2)
        scale = (x1 - x0) / float(self.far_d[k])
        for v in self.rug[k][:int(self.SHOWN * reach):3]:
            fx = x0 + scale * v
            sc.line(self.P(fx, y - 0.022, 0.6), self.P(fx, y + 0.022, 0.6), width=1.0,
                    color=BLUE, opacity=op * 0.55, bias=0.2)
        for v, ink in ((self.near_d[k], self.NEAR), (self.far_d[k], self.FAR)):
            fx = x0 + scale * v
            sc.line(self.P(fx, y - 0.04, 0.9), self.P(fx, y + 0.04, 0.9), width=3.4, color=ink,
                    opacity=op * reach, bias=0.3)
        self.text(sc, x0, y + 0.065, "0", op, size=0.032)
        self.text(sc, 0.5, y + 0.065, "distance from the new image to each of 1,000 training "
                  "images", op, size=0.032)
        self.text(sc, 0.10, 0.885, "nearest ÷ farthest = %s%%" % number(
            100 * self.near_d[k] / self.far_d[k]), op * smoothstep(0.5, 0.65, w), anchor="lm",
            strong=True, size=0.040)
        self.text(sc, 0.90, 0.885, "averaged over 1,000 new images: %s%%" % number(
            100 * self.ratio[0]), op * smoothstep(0.9, 0.97, u), anchor="rm", size=0.036)

    def phase_2(self, sc, u, op):
        k, w = self.turn(u, len(self.dims), 1.03)
        f = smoothstep(0.0, 0.45, w) if k else 1.0
        hist = self.hist[k] if not k else self.hist[k - 1] * (1 - f) + self.hist[k] * f
        ratio = self.ratio[k] if not k else self.ratio[k - 1] * (1 - f) + self.ratio[k] * f
        bins = len(hist)
        chart = LossChart(((0.08, 0.50), (0.42, 0.80)), (0.0, 1.0), (0.0, float(self.hist.max())))
        self.axes(sc, chart, op, "distance to a training image, as a share of the farthest",
                  "", xticks=[(v, "%d%%" % (100 * v)) for v in (0, 0.25, 0.5, 0.75, 1.0)],
                  yticks=[])
        bw = self.span(0.42 / bins)[0]
        for b, v in enumerate(hist):
            h = self.span(0, 0.38 * v / self.hist.max())[1]
            x = chart.fx((b + 0.5) / bins)
            self.rect(sc, self.P(x, 0.80 - self.frac(0, h / 2)[1]), bw - 1.5, max(h, 0.5), BLUE,
                      op, radius=1.5)
        for v, ink, name in ((ratio, self.NEAR, "the nearest"), (1.0, self.FAR, "the farthest")):
            x = chart.fx(v)
            sc.line(self.P(x, 0.80, 0.9), self.P(x, 0.405, 0.9), width=3.0, color=ink, opacity=op,
                    bias=0.3)
            self.text(sc, x + (0.008 if v == 1.0 else -0.008), 0.43, name, op, ink=dark(ink),
                      anchor="lm" if v == 1.0 else "rm")
        self.text(sc, 0.08, 0.325, "nearest ÷ farthest: %.0f%%" % (100 * self.ratio[k]), op,
                  anchor="lm", ink=dark(self.NEAR), size=0.040)
        x = 0.58
        added = int(self.dims[k]) - 49
        self.text(sc, x, 0.325, "49 real + %s shuffled coordinates" % f"{added:,}", op,
                  anchor="lm", strong=True, size=0.036)
        self.grid(sc, self.pooled[0].reshape(7, 7), x + 0.045, 0.515, 78, 78, op, hot=DIGIT_INK,
                  vmax=1.0, ring_ink=self.REAL)
        self.text(sc, x + 0.045, 0.395, "real", op, ink=dark(self.REAL), size=0.032)
        tiles = min(added // 49, 3)
        grow = smoothstep(0.0, 0.35, w)
        for n in range(tiles):
            fx = x + 0.045 + 0.10 * (n + 1)
            self.grid(sc, self.tiles[n][0].reshape(7, 7), fx, 0.515, 78, 78,
                      op * (grow if k == (1 if n == 0 else 2) else 1.0), hot=self.NOISE,
                      vmax=1.0)
        if tiles:
            self.text(sc, x + 0.045 + 0.10 * (tiles + 1) / 2, 0.395, "shuffled", op,
                      ink=dark(self.NOISE), size=0.032)
        if added // 49 > 3:
            self.text(sc, x, 0.64, "and %d more blocks like these" % (added // 49 - 3), op,
                      anchor="lm", ink=dark(self.NOISE))
        self.lines(sc, x, 0.72, ["a shuffled value is a real pixel of some",
                                 "other image, dealt here at random: it says",
                                 "nothing about this image or its digit"], op, step=0.046,
                   size=0.032)

    def phase_3(self, sc, u, op):
        cx, cy, side = 0.21, 0.61, 250.0
        self.rect(sc, self.P(cx, cy), side, side, PAPER, op, border=AXIS, radius=2, lift=0.1)
        self.disk(sc, self.P(cx, cy), side / 2 - 1, soft(BLUE, 0.3), op, lift=0.3, outline=0.1)
        dx, dy = self.frac(side / 2, side / 2)
        for end, ink in (((cx + dx, cy), DIGIT_INK), ((cx + dx, cy - dy), RED)):
            sc.line(self.P(cx, cy, 0.9), self.P(*end, 0.9), width=2.6, color=ink, opacity=op,
                    bias=0.3)
        self.text(sc, cx + dx / 2, cy + 0.03, "0.5", op, strong=True)
        self.text(sc, cx + dx / 2 - 0.03, cy - dy / 2 - 0.02, "0.71", op, ink=dark(RED))
        self.text(sc, cx, cy + dy + 0.045, "2 dimensions: the ball fills %s%%" % number(
            100 * self.volume[0]), op, size=0.034)
        x, full = 0.50, self.span(0.20)[0]
        self.text(sc, x, 0.335, "dimensions", op, anchor="rm")
        self.text(sc, x + 0.02, 0.335, "share inside the ball", op, anchor="lm", ink=dark(BLUE))
        self.text(sc, 0.93, 0.335, "to a corner", op, anchor="rm", ink=dark(RED))
        for r, (d, v) in enumerate(zip(self.BALL, self.volume)):
            y = 0.415 + r * 0.078
            show = op * smoothstep(0.08 * r, 0.08 * r + 0.15, u)
            value = "%s%%" % number(100 * v) if v > 1e-4 else "1 part in %s" % pow10(1 / v)
            self.text(sc, x, y, str(d), show, anchor="rm", strong=True, size=0.040)
            self.hbar(sc, x + 0.02, y, full * v, BLUE, show, value=value, thick=26.0, track=full)
            self.text(sc, 0.93, y, "%.2f" % (np.sqrt(d) / 2), show, anchor="rm", ink=dark(RED),
                      size=0.036)
        self.text(sc, x + 0.02, 0.89, "exact formulas, for a box of side 1; not a measurement",
                  op, anchor="lm")

    def phase_4(self, sc, u, op):
        shown, right = 12, [0, 0]
        columns = (0, self.LEVEL)
        names = ("new", "49 real", "+%d" % (self.dims[self.LEVEL] - 49))
        for r in range(shown):
            x0, y = 0.075 + 0.235 * (r // 6), 0.40 + (r % 6) * 0.088
            show = op * smoothstep(0.03 * r, 0.03 * r + 0.1, u)
            if r % 6 == 0:
                for name, dx in zip(names, (0.0, 0.068, 0.136)):
                    self.text(sc, x0 + dx, 0.325, name, op)
            self.digit(sc, self.test_images[r], x0, y, 36, show, frame=self.NEW)
            for n, column in enumerate(columns):
                j = int(self.nearest[column][r])
                same = bool(self.labels[j] == self.test_labels[r])
                right[n] += same
                self.digit(sc, self.images[j], x0 + 0.068 * (n + 1), y, 36, show,
                           frame=GREEN if same else RED)
        self.text(sc, 0.05, 0.915, "green frame: the nearest image is the same digit "
                  "(%d of these %d, then %d)" % (right[0], shown, right[1]), op, anchor="lm",
                  size=0.030)
        chart = self.chart(((0.59, 0.93), (0.44, 0.80)), self.dims, [0.0, 1.0], xlog=True, pad=0.0)
        self.axes(sc, chart, op, "coordinates (the first 49 are real)", "",
                  xticks=[(d, f"{d:,}") for d in self.dims[[0, 2, 4, 5]]],
                  yticks=[(v, "%d%%" % (100 * v)) for v in (0.25, 0.5, 0.75, 1.0)])
        s = (u - 0.3) / 0.5
        for ys in (self.acc_lo, self.acc_hi):
            self.plot(sc, chart, self.dims, ys, BLUE, op * 0.45, s, width=1.0, dot=False)
        self.plot(sc, chart, self.dims, self.accuracy, BLUE, op, s, width=3.0)
        self.plot(sc, chart, self.dims, self.ratio, self.NEAR, op, s, width=3.0)
        y = chart.fy(0.1)
        for i in range(0, 24, 2):
            a, b = (chart.x0 + (chart.x1 - chart.x0) * v / 24 for v in (i, i + 1))
            sc.line(self.P(a, y, 0.6), self.P(b, y, 0.6), width=1.4, color=AXIS, opacity=op,
                    bias=0.2)
        self.text(sc, chart.x0 + 0.01, y - 0.028, "guessing", op, anchor="lm", size=0.030)
        self.text(sc, 0.57, 0.32, "nearest image is the same digit (5 shuffles)", op,
                  anchor="lm", ink=dark(BLUE), size=0.032)
        self.text(sc, 0.57, 0.372, "nearest ÷ farthest distance", op, anchor="lm",
                  ink=dark(self.NEAR), size=0.032)

    def phase_5(self, sc, u, op):
        chart = self.chart(((0.10, 0.52), (0.37, 0.80)), self.sizes, [0.0, 1.0], xlog=True,
                           pad=0.0)
        self.axes(sc, chart, op, "training images", "nearest image is the same digit",
                  xticks=[(v, f"{v:,}") for v in self.sizes[::2]],
                  yticks=[(v, "%d%%" % (100 * v)) for v in (0.25, 0.5, 0.75, 1.0)])
        inks = (self.REAL, BLUE, self.NOISE, MAGENTA, OCHRE, RED)
        at = [chart.fy(ys[-1]) for ys in self.table]      # the labels, kept from overlapping
        for level in range(1, len(at)):
            at[level] = max(at[level], at[level - 1] + 0.036)
        for level, (ys, ink) in enumerate(zip(self.table, inks)):
            s = (u - 0.06 * level) / 0.4
            self.plot(sc, chart, self.sizes, ys, ink, op, s, width=3.0 if level == 0 else 2.2)
            if s >= 1.0:
                self.text(sc, chart.x1 + 0.012, at[level],
                          "49 real" if level == 0 else "+%s" % f"{self.dims[level] - 49:,}", op, anchor="lm", ink=dark(ink), size=0.030)  # noqa: E501
        x = 0.66
        self.lines(sc, x, 0.335, ["on the curves: how many shuffled",
                                  "coordinates were added to the 49 real"], op, step=0.044,
                   size=0.030)
        late = op * smoothstep(0.55, 0.7, u)
        ref = self.table[0, 0]
        self.lines(sc, x, 0.46, [
            ("49 real, %d images: %.0f%%" % (self.sizes[0], 100 * ref), dark(self.REAL)),
            ("+49, %s images: %.0f%%" % (f"{self.sizes[-1]:,}",
                                         100 * self.table[1, -1]), dark(BLUE)),
            ("+%d, %s images: %.0f%%" % (self.dims[3] - 49, f"{self.sizes[-1]:,}",
                                         100 * self.table[3, -1]), dark(MAGENTA))],
            late, step=0.056, size=0.034)
        self.lines(sc, x, 0.66, [
            "it is the empty coordinates that hurt:",
            "by %d, %d and %d real PCA coordinates" % tuple(self.pca_dims[[0, 2, 5]]),
            "the nearest image is right %.0f%%, %.0f%%" % tuple(
                100 * self.pca_accuracy[[0, 2]]),
            "and %.0f%% of the time" % (100 * self.pca_accuracy[5])],
            op * smoothstep(0.75, 0.9, u), step=0.046, size=0.032)


# ====================================================================== 73 tsne-and-umap
TEN = INKS + [AXIS, DIGIT_INK]         # a colour per digit, 0 to 9
pair_distances, ranks, map_faithfulness = sx.pair_distances, sx.ranks, sx.map_faithfulness


def conditional_affinities(x, perplexity):
    """Row ``i``: how image ``i`` shares its attention among the others (each
    row sums to one), by the bandwidth search of :func:`next_vision.tsne_affinity`,
    which symmetrises exactly this matrix."""
    dist = ((x[:, None] - x[None]) ** 2).sum(-1)
    n, p = len(x), np.zeros_like(dist)
    for i in range(n):
        mask = np.arange(n) != i
        beta, lo, hi = 1.0, 0.0, np.inf
        for _ in range(50):
            values = -dist[i, mask] * beta
            values -= values.max()
            row = np.exp(values)
            row /= row.sum()
            entropy = -np.sum(row * np.log(row + 1e-300))
            if entropy > np.log(perplexity):
                lo = beta
                beta = beta * 2 if np.isinf(hi) else (beta + hi) / 2
            else:
                hi = beta
                beta = (beta + lo) / 2
        p[i, mask] = row
    return p


def tsne_path(x, perplexity, steps=400, seed=nv.SEED, every=5):
    """:func:`next_vision.tsne_fit` again, keeping the map every ``every`` steps
    (the first is the random start, the last the study's map)."""
    rng = np.random.default_rng(seed)
    p = nv.tsne_affinity(x, perplexity)
    z = rng.normal(0, 1e-3, (len(x), 2))
    velocity = np.zeros_like(z)
    frames = [z.copy()]
    for step in range(steps):
        dif = z[:, None] - z[None]
        inv = 1 / (1 + (dif ** 2).sum(-1))
        np.fill_diagonal(inv, 0)
        q = inv / inv.sum()
        pp = p * (4 if step < 80 else 1)
        gradient = 4 * np.sum(((pp - q) * inv)[:, :, None] * dif, axis=1)
        velocity = 0.7 * velocity - 50 * gradient
        z += velocity
        z -= z.mean(0)
        if (step + 1) % every == 0:
            frames.append(z.copy())
    return np.array(frames)


class TsneUmap(Classical):
    """240 real digit images mapped twice: by t-SNE (whom each image counts as a
    neighbour at three perplexities, the map forming, the three maps) and by
    UMAP (its graph of nearest images, pulled into a map), then both maps side
    by side with how much of the images' own distances each keeps."""

    TITLE = "t-SNE and UMAP"
    PHASES = (("neighbours", 10.0), ("t-SNE", 11.0), ("three maps", 9.0), ("UMAP", 13.0),
              ("side by side", 13.0))
    PERPLEXITIES = (5, 30, 100)
    WHO = 0                            # the image followed throughout
    NEAR = 12
    TSNE, UMAP = BLUE, GREEN

    def setup(self):
        d = sx.maps_data()
        self.x, self.labels = d["pixels"], d["labels"]
        self.maps = [np.asarray(z, float) for z in d["tsne"][:, 0]]
        self.losses = d["tsne_losses"][:, 0]
        self.umap, self.umap_frames, self.graph = d["umap"][1, 0], d["umap_frames"], d["graph"]
        self.neighbours = int(sx.MAPS["neighbours"][1])
        self.epochs, self.every = int(sx.MAPS["umap_epochs"]), int(sx.MAPS["every"])
        self.faith = d["faith"]                        # [method, setting, seed, measure]
        self.images = pixels("test", d["indices"])
        self.inks = [TEN[int(v)] for v in self.labels]
        self.dist = pair_distances(self.x)
        self.order = np.argsort(self.dist[self.WHO], kind="stable")[1:self.NEAR + 1]
        self.share = np.array([conditional_affinities(self.x, p)[self.WHO]
                               for p in self.PERPLEXITIES])
        settings = dict(perplexities=list(self.PERPLEXITIES), steps=400, every=5, seed=nv.SEED,
                        version=1)
        paths = cached_npz("mech_classical_tsne_paths", settings, lambda: {
            "p%d" % p: tsne_path(self.x, p) for p in self.PERPLEXITIES})
        self.frames = paths["p30"]
        ii, jj = np.nonzero(np.triu(self.graph, 1) > 0.5)
        self.strong = np.column_stack((ii, jj))        # the graph's stronger links
        self.linked = np.flatnonzero(self.graph[self.WHO] > 0)

    def _span(self, method, measure):
        """Mean, lowest and highest over the seeds, of the middle setting."""
        v = self.faith[method, 1, :, measure]
        return float(v.mean()), float(v.min()), float(v.max())

    def subtitle_for(self, k):
        many = [int((row > 0.01).sum()) for row in self.share]
        return [
            "t-SNE starts from who is near whom. Each image shares its attention among the "
            "others; the perplexity sets how many get a real share: here %d, %d, then %d" % (
                many[0], many[1], many[2]),
            "then 240 points start in a heap and move for 400 steps: images that count each "
            "other as neighbours pull together, all the rest push apart",
            "the same 240 images at perplexity 5, 30 and 100: three different maps. The ringed "
            "points are one image and its 12 nearest images",
            "UMAP keeps a graph instead: each image linked to its %d nearest, most strongly to "
            "the closest. For %d rounds linked points pull together, and points drawn at random "
            "are pushed apart" % (self.neighbours, self.epochs),
            "the same images, two methods, two maps. Of each image's 10 nearest images t-SNE "
            "keeps %.0f%% nearby and UMAP %.0f%%. Across all pairs, map distance follows real "
            "distance only loosely: %.2f and %.2f" % (
                100 * self._span(0, 0)[0], 100 * self._span(1, 0)[0], self._span(0, 1)[0],
                self._span(1, 1)[0]),
        ][k]

    def _key(self, sc, x, y, op):
        """The digits' colours."""
        for d in range(10):
            self.box(sc, x + 0.033 * d, y, 26, 26, TEN[d], op, str(d), ink=WHITE, size=0.032)

    def _map(self, sc, op, pane, z, r=3.6, ring=True):
        self.backdrop(sc, pane, op, pad=0.0)
        self.dots(sc, pane, z, self.inks, op, r=r)
        if ring:
            for j in self.order:
                self.disk(sc, self.P(*pane.at(*z[j])), r, self.inks[j], op, ring=DIGIT_INK,
                          ring_w=1.6, lift=1.3)
            self.disk(sc, self.P(*pane.at(*z[self.WHO])), r + 2.5, self.inks[self.WHO], op,
                      ring=AMBER, ring_w=3.0, lift=1.6)

    def _fitted(self, cx, cy, side, z):
        """A square pane holding the map ``z``."""
        lo, hi = z.min(0), z.max(0)
        half = float((hi - lo).max()) / 2 * 1.08
        (mx, my) = (lo + hi) / 2
        return self.square(cx, cy, side, (mx - half, mx + half), (my - half, my + half))

    def phase_1(self, sc, u, op):
        k, w = self.turn(u, 3, 1.03)
        f = smoothstep(0.0, 0.4, w) if k else 1.0
        share = self.share[k] if not k else self.share[k - 1] * (1 - f) + self.share[k] * f
        top = float(self.share[:, self.order].max())
        self.digit(sc, self.images[self.WHO], 0.13, 0.47, 120, op, frame=AMBER)
        self.text(sc, 0.13, 0.315, "one of the 240 images", op)
        self.text(sc, 0.05, 0.67, "perplexity %d" % self.PERPLEXITIES[k], op, anchor="lm",
                  strong=True, size=0.046)
        self.lines(sc, 0.05, 0.75, ["%d images get more than" % int((self.share[k] > 0.01).sum()),
                                    "1% of its attention"], op, step=0.048, size=0.034)
        base, tall = 0.71, 0.36
        for n, j in enumerate(self.order):
            fx = 0.345 + 0.05 * n
            self.digit(sc, self.images[j], fx, 0.79, 38, op, frame=TEN[int(self.labels[j])])
            h = self.span(0, tall * share[j] / top)[1]
            self.rect(sc, self.P(fx, base - self.frac(0, h / 2)[1]), 28, max(h, 0.6), BLUE, op,
                      radius=2.0)
            if n < 6 or share[j] > 0.02:
                self.text(sc, fx, base - self.frac(0, h)[1] - 0.028, "%.0f%%" % (100 * share[j]),
                          op)
        self.text(sc, 0.345 + 0.05 * 5.5, 0.885, "its 12 nearest images, nearest first; "
                  "the bars: its attention to each", op)

    def phase_2(self, sc, u, op):
        steps = (len(self.frames) - 1) * min(u / 0.88, 1.0)
        i = int(min(steps, len(self.frames) - 2))
        z = self.frames[i] + (self.frames[i + 1] - self.frames[i]) * (steps - i)
        r = float(np.abs(self.frames[-1]).max()) * 1.08
        pane = self.square(0.27, 0.62, 310.0, (-r, r), (-r, r))
        self._map(sc, op, pane, z)
        for j in self.order[:6]:
            sc.line(self.P(*pane.at(*z[self.WHO]), 0.9), self.P(*pane.at(*z[j]), 0.9), width=1.6,
                    color=AMBER, opacity=op * 0.8, bias=0.3)
        step = int(round(steps * 5))
        xs = np.arange(1, self.losses.shape[1] + 1)
        chart = self.chart(((0.58, 0.93), (0.44, 0.74)), xs, np.r_[0.0, self.losses[1]])
        self.axes(sc, chart, op, "step", "mismatch between map and neighbours")
        self.plot(sc, chart, xs, self.losses[1], BLUE, op, step / 400.0, width=2.8)
        self.text(sc, 0.58, 0.33, "t-SNE, perplexity 30, step %d of 400" % step, op, anchor="lm",
                  strong=True, size=0.036)
        self.text(sc, 0.58, 0.845, "each point is an image; its colour, the digit", op,
                  anchor="lm")
        self._key(sc, 0.595, 0.905, op)

    def phase_3(self, sc, u, op):
        for n, (p, z) in enumerate(zip(self.PERPLEXITIES, self.maps)):
            show = op * smoothstep(0.12 * n, 0.12 * n + 0.15, u)
            r = float(np.abs(z).max()) * 1.08
            pane = self.square(0.19 + 0.31 * n, 0.615, 244.0, (-r, r), (-r, r))
            self._map(sc, show, pane, z, r=3.0)
            self.text(sc, 0.19 + 0.31 * n, 0.32, "perplexity %d" % p, show, strong=True,
                      size=0.038)
            self.text(sc, 0.19 + 0.31 * n, 0.88, "%.0f units across" % (2 * np.abs(z).max()),
                      show)
        self.text(sc, 0.5, 0.928, "amber ring: the same image in each map; %s rings: its 12 "
                  "nearest images" % theme.word("black", "white"), op)

    def phase_4(self, sc, u, op):
        last = len(self.umap_frames) - 1
        steps = last * smoothstep(0.30, 0.90, u)
        i = int(min(steps, last - 1))
        z = self.umap_frames[i] + (self.umap_frames[i + 1] - self.umap_frames[i]) * (steps - i)
        # the view closes in as the map tightens: map units mean nothing by themselves
        pane = self._fitted(0.27, 0.62, 310.0, z)
        self.backdrop(sc, pane, op, pad=0.0)
        web = smoothstep(0.04, 0.26, u)
        for a, b in self.strong[:int(len(self.strong) * web)]:
            sc.line(self.P(*pane.at(*z[a]), 0.6), self.P(*pane.at(*z[b]), 0.6), width=1.0,
                    color=AXIS, opacity=op * 0.55, bias=0.2)
        for j in self.linked:
            sc.line(self.P(*pane.at(*z[self.WHO]), 0.9), self.P(*pane.at(*z[j]), 0.9),
                    width=1.0 + 3.0 * self.graph[self.WHO, j], color=AMBER, opacity=op * 0.9,
                    bias=0.3)
        self.dots(sc, pane, z, self.inks, op, r=3.4)
        self.disk(sc, self.P(*pane.at(*z[self.WHO])), 6.0, self.inks[self.WHO], op, ring=AMBER,
                  ring_w=3.0, lift=1.6)
        x = 0.58
        epoch = int(round(steps * self.every))
        self.text(sc, x, 0.33, "UMAP, %d neighbours, round %d of %d" % (
            self.neighbours, epoch, self.epochs), op, anchor="lm", strong=True, size=0.036)
        # the followed image's links, nearest first: the strength each is given
        near = [j for j in np.argsort(self.dist[self.WHO], kind="stable")[1:]
                if self.graph[self.WHO, j] > 0][:8]
        self.digit(sc, self.images[self.WHO], x + 0.03, 0.47, 50, op, frame=AMBER)
        base, tall = 0.51, 0.09
        for n, j in enumerate(near):
            fx = x + 0.10 + 0.034 * n
            v = float(self.graph[self.WHO, j])
            h = self.span(0, tall * v)[1]
            self.rect(sc, self.P(fx, base - self.frac(0, h / 2)[1]), 14, max(h, 0.6), AMBER, op,
                      radius=1.5)
            self.digit(sc, self.images[j], fx, base + 0.045, 24, op)
        self.text(sc, x + 0.10 + 0.034 * 3.5, 0.395, "its links: strength 1 for the nearest",
                  op, size=0.030)
        self.lines(sc, x, 0.635, [
            ("grey: the %d strongest links of the graph" % len(self.strong), None),
            ("amber: the links of one image", dark(AMBER)),
            ("the start: the graph's own two smoothest", None),
            ("directions; then each round every link pulls", None),
            ("its ends together and 5 points drawn at", None),
            ("random are pushed away", None)], op, step=0.041, size=0.030)
        self._key(sc, 0.595, 0.915, op * smoothstep(0.3, 0.4, u))

    def phase_5(self, sc, u, op):
        for n, (name, ink, z) in enumerate((("t-SNE, perplexity 30", self.TSNE, self.maps[1]),
                                            ("UMAP, %d neighbours" % self.neighbours, self.UMAP,
                                             self.umap))):
            show = op * smoothstep(0.1 * n, 0.1 * n + 0.15, u)
            pane = self._fitted(0.165 + 0.275 * n, 0.60, 236.0, z)
            self._map(sc, show, pane, z, r=3.0)
            self.text(sc, 0.165 + 0.275 * n, 0.32, name, show, ink=dark(ink), size=0.036)
        self.text(sc, 0.30, 0.875, "amber ring: the same image; %s rings: its 12 nearest"
                  % theme.word("black", "white"), op, size=0.030)
        x, full = 0.60, self.span(0.21)[0]
        rows = (("of the 10 nearest images, kept nearby", 0, "%.0f%%", 100.0),
                ("map distance against real distance", 1, "%.2f", 1.0))
        for r, (title, measure, form, times) in enumerate(rows):
            y = 0.35 + 0.265 * r
            late = (u - 0.25 - 0.18 * r) / 0.2
            self.text(sc, x, y, title, op * smoothstep(0.0, 0.3, late), anchor="lm", strong=True,
                      size=0.032)
            for m, (name, ink) in enumerate((("t-SNE", self.TSNE), ("UMAP", self.UMAP))):
                mean, lo, hi = self._span(m, measure)
                yy = y + 0.065 + 0.075 * m
                self.hbar(sc, x + 0.065, yy, full * mean, ink, op * smoothstep(0.0, 0.3, late),
                          label=name, value="%s  (%s to %s)" % (
                              form % (times * mean), form % (times * lo), form % (times * hi)),
                          u=late, thick=24.0, track=full)
        self.lines(sc, x, 0.87, ["in brackets: lowest and highest of 3 seeds",
                                 "1.00 would be a perfect ranking of distances"],
                   op * smoothstep(0.7, 0.85, u), step=0.042, size=0.030)


# ====================================================================== 74 roc-and-calibration
def seven_scores(n=2000):
    """The study's classifier on its ``n`` held-out images: the score it gives
    "this is a seven", and whether each image is one."""
    from . import mnist
    params, _ = mnist.train()
    _, _, x, y = nv.mnist_data()
    return mnist._forward(params, x[:n])[2][:, 7], y[:n] == 7


def precision_at(tpr, fpr, prevalence):
    """Exact: the share of positive calls that are right, for a classifier with
    these rates on data where ``prevalence`` of the cases are positive."""
    return tpr * prevalence / (tpr * prevalence + fpr * (1 - prevalence))


def reliability(prob, target, bins=10):
    """The study's calibration bins: ``(lo, hi, mean score, share positive, count)``
    for each occupied tenth of the score range."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for a, b in itertools.pairwise(edges):
        chosen = (prob >= a) & (prob <= b if b == 1 else prob < b)
        if chosen.any():
            out.append((a, b, float(prob[chosen].mean()), float(target[chosen].mean()),
                        int(chosen.sum())))
    return out


class RocAndCalibration(Classical):
    """A threshold sweeping across a real classifier's scores: the four counts
    changing, the ROC and precision-recall curves traced by the sweep, what a
    rarer positive does to precision, and whether a score of 15% means 15%."""

    TITLE = "ROC and calibration"
    PHASES = (("scores", 10.0), ("threshold", 15.0), ("two curves", 10.0), ("base rate", 9.0),
              ("calibration", 11.0))
    YES, NO, CUT = GREEN, VIOLET, AMBER
    Z, NBINS = 12.0, 24                 # the score axis: log-odds from -Z to Z
    STRIP = (0.07, 0.60, 0.50, 0.15)    # x0, x1, baseline y, tallest bar
    CAP = 0.16                          # the share of a class the tallest bar stands for

    def setup(self):
        s = self.study
        self.prob, self.seven = seven_scores()
        self.confusion = np.asarray(s.panels[0].data["values"], int)
        self.fpr = np.asarray(s.panels[1].data["x"], float)
        self.tpr = np.asarray(s.panels[1].data["ys"][0], float)
        self.precision = np.asarray(s.panels[2].data["ys"][0], float)
        self.auc = float(s.panels[1].notes[0].split()[-1])
        self.z = np.clip(np.log(np.maximum(self.prob, 1e-12) / np.maximum(1 - self.prob, 1e-12)),
                         -self.Z, self.Z)
        edges = np.linspace(-self.Z, self.Z, self.NBINS + 1)
        self.share = [np.histogram(self.z[m], edges)[0] / m.sum()
                      for m in (self.seven, ~self.seven)]
        self.bins = reliability(self.prob, self.seven)
        self.images = pixels("test", np.arange(len(self.prob)))
        order = np.argsort(self.prob)
        yes, no = order[self.seven[order]], order[~self.seven[order]]
        self.examples = [int(i) for i in (yes[-1], yes[len(yes) // 4], yes[0],
                                          no[-1], no[-12], no[len(no) // 2])]
        (tn, fp), (fn, tp) = self.confusion
        self.rates = tp / (tp + fn), fp / (fp + tn)
        self.prevalence = float(self.seven.mean())
        self.opened = 1                     # the calibration bin whose images are shown

    def subtitle_for(self, k):
        (tn, fp), (fn, tp) = self.confusion
        a, b, _conf, freq, count = self.bins[self.opened]
        return [
            "a digit classifier gives each of 2,000 unseen images a score: how sure it is that "
            "the image is a seven. %d of them are sevens; most scores sit at the far ends" % (
                int(self.seven.sum())),
            "a threshold turns scores into calls. At 50%% it catches %d sevens, misses %d, and "
            "raises %d false alarms among the %s other digits" % (
                tp, fn, fp, f"{tn + fp:,}"),
            "slide the threshold and plot sevens caught against false alarms: the ROC curve. "
            "The area under it is %.4f; 1 would be a perfect ranking" % self.auc,
            "precision is the share of calls that are right: %.1f%% here, where %.1f%% of images "
            "are sevens. The same classifier on rarer sevens would be wrong far more often" % (
                100 * tp / (tp + fp), 100 * self.prevalence),
            "calibration asks whether a score means what it says. Of the %d images scored "
            "between %.0f%% and %.0f%%, %.0f%% were sevens" % (count, 100 * a, 100 * b, 100 * freq),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _zx(self, z):
        x0, x1 = self.STRIP[:2]
        return x0 + (x1 - x0) * (z + self.Z) / (2 * self.Z)

    def _strip(self, sc, op, cut=None, grow=1.0):
        """Every image's score on one axis: sevens stacked up, the others
        down, each bar a share of its own group. ``cut``: the threshold (log-odds)."""
        x0, x1, base, tall = self.STRIP
        width = self.span((x1 - x0) / self.NBINS)[0] - 2.0
        for b in range(self.NBINS):
            zc = -self.Z + (b + 0.5) * 2 * self.Z / self.NBINS
            called = cut is None or zc >= cut
            for share, ink, sign in ((self.share[0][b], self.YES, -1),
                                     (self.share[1][b], self.NO, 1)):
                if share <= 0:
                    continue
                h = self.span(0, tall * min(share, self.CAP) / self.CAP)[1] * grow
                self.rect(sc, self.P(self._zx(zc), base + sign * (0.006 + self.frac(0, h / 2)[1])),
                          width, max(h, 1.0), ink if called else soft(ink, 0.6), op, radius=2.0)
                if share > self.CAP:
                    self.text(sc, self._zx(zc) + 0.02, base + sign * (tall + 0.038),
                              "%.0f%% of them" % (100 * share), op * grow, anchor="lm",
                              ink=dark(ink))
        sc.line(self.P(x0, base), self.P(x1, base), width=1.2, color=AXIS, opacity=op, bias=0.2)
        for pr, name in ((1e-4, "0.01%"), (0.01, "1%"), (0.5, "50%"), (0.99, "99%"),
                         (0.9999, "99.99%")):
            self.text(sc, self._zx(np.log(pr / (1 - pr))), base + tall + 0.085, name, op)
        self.text(sc, (x0 + x1) / 2, base + tall + 0.13, "the score: how sure it is a seven", op)
        if cut is not None:
            x = self._zx(cut)
            sc.line(self.P(x, base - tall - 0.03, 0.9), self.P(x, base + tall + 0.008, 0.9),
                    width=3.2, color=self.CUT, opacity=op, bias=0.3)
            self.text(sc, float(np.clip(x, x0 + 0.09, x1 - 0.09)), base - tall - 0.055,
                      "threshold %s" % self._pc(cut), op, ink=dark(self.CUT), size=0.034)

    @staticmethod
    def _pc(z):
        pr = 100.0 / (1.0 + np.exp(-z))
        return "%.0f%%" % pr if 1 <= pr <= 99 else "%.2g%%" % pr if pr < 1 else "%.4g%%" % pr

    def _counts(self, cut):
        """(caught, missed, false alarms, left alone) at a threshold in log-odds."""
        called = self.prob >= 1.0 / (1.0 + np.exp(-cut)) if abs(cut) < self.Z else \
            np.full(len(self.prob), cut < 0)
        return (int((called & self.seven).sum()), int((~called & self.seven).sum()),
                int((called & ~self.seven).sum()), int((~called & ~self.seven).sum()))

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        self._strip(sc, op, grow=smoothstep(0.45, 0.9, u))
        x0, x1, base, tall = self.STRIP
        self.text(sc, x0, base - tall - 0.04, "the %d sevens" % int(self.seven.sum()), op,
                  anchor="lm", ink=dark(self.YES), size=0.036)
        others = f"the {int((~self.seven).sum()):,} other digits"
        self.text(sc, x1, base + tall - 0.02, others,
                  op, anchor="rm", ink=dark(self.NO), size=0.036)
        for n, i in enumerate(self.examples):
            fx, fy = 0.70 + 0.10 * (n % 3), 0.43 + 0.27 * (n // 3)
            ink = self.YES if self.seven[i] else self.NO
            self.digit(sc, self.images[i], fx, fy, 72, op, frame=ink)
            self.text(sc, fx, fy + 0.105, self._pc(self.z[i]), op, ink=dark(ink), size=0.034)
            sign = -1 if self.seven[i] else 1
            self.pulse(sc, (fx, fy), (self._zx(self.z[i]), base + sign * 0.03),
                       (u - 0.08 - 0.06 * n) / 0.3, ink, op, bend=0.08 * -sign, r=6)
        self.text(sc, 0.80, 0.315, "six of the images, and their scores", op)

    def phase_2(self, sc, u, op):
        cut = self.Z * (1 - smoothstep(0.03, 0.42, u)) - self.Z * smoothstep(0.62, 0.95, u)
        self._strip(sc, op, cut)
        tp, fn, fp, tn = self._counts(cut)
        x0 = self.STRIP[0]
        for row, (name, ink, left, right) in enumerate((
                ("really a seven", self.YES, ("missed", fn), ("caught", tp)),
                ("really not", self.NO, ("left alone", tn), ("false alarm", fp)))):
            y = 0.848 + row * 0.066
            self.text(sc, x0, y, name, op, anchor="lm", ink=dark(ink), size=0.034)
            for (what, count), fx, solid in ((left, 0.305, False), (right, 0.495, True)):
                self.box(sc, fx, y, 168, 30, ink if solid else soft(ink, 0.6), op,
                         "%s  %s" % (what, f"{count:,}"),
                         ink=WHITE if solid else dark(ink, 0.35), size=0.032)
        pane = Pane(((0.70, 0.93), (0.36, 0.74)), (0, 1), (0, 1))
        self.backdrop(sc, pane, op, pad=0.0)
        self.axes(sc, pane, op, "false alarms", "sevens caught", grid=False,
                  xticks=[(0, "0"), (1, "all")], yticks=[(0, "0"), (1, "all")])
        self._roc(sc, pane, op, tp / (tp + fn), fp / (fp + tn))
        self.lines(sc, 0.70, 0.865, [("caught %.1f%% of sevens" % (100 * tp / (tp + fn)),
                                      dark(self.YES)),
                                     ("alarms on %.2f%% of the rest" % (100 * fp / (fp + tn)),
                                      dark(self.NO))], op, step=0.045)

    def _roc(self, sc, pane, op, tpr, fpr, every=6):
        """The ROC curve as far as the point ``(fpr, tpr)`` the threshold has reached."""
        k = int(np.sum((self.fpr <= fpr + 1e-12) & (self.tpr <= tpr + 1e-12)))
        xs, ys = np.r_[self.fpr[:k:every], fpr], np.r_[self.tpr[:k:every], tpr]
        self.path(sc, pane, xs, ys, BLUE, op, 3.0)
        self.disk(sc, self.P(*pane.at(fpr, tpr)), 6.0, BLUE, op, ring=self.CUT, ring_w=2.2,
                  lift=1.5)

    def phase_3(self, sc, u, op):
        k = int(np.clip(smoothstep(0.0, 0.8, u) * len(self.fpr), 2, len(self.fpr)))
        zoom = Pane(((0.11, 0.47), (0.41, 0.81)), (0, 0.05), (0.5, 1.0))
        self.backdrop(sc, zoom, op, pad=0.0)
        self.axes(sc, zoom, op, "false alarms, share of the other digits",
                  "sevens caught (from 50%)",
                  grid=False, xticks=[(v, "%d%%" % (100 * v)) for v in (0, 0.01, 0.03, 0.05)],
                  yticks=[(v, "%d%%" % (100 * v)) for v in (0.75, 1.0)])
        self.path(sc, zoom, self.fpr[:k], self.tpr[:k], BLUE, op, 3.0)
        self.text(sc, 0.45, 0.74, "the ROC curve's corner", op, anchor="rm", ink=dark(BLUE),
                  size=0.034)
        self.text(sc, 0.45, 0.685, "area under the whole curve: %.4f" % self.auc,
                  op * smoothstep(0.7, 0.85, u), anchor="rm", strong=True, size=0.034)
        pr = Pane(((0.58, 0.93), (0.41, 0.81)), (0, 1.0), (0.5, 1.0))
        self.backdrop(sc, pr, op, pad=0.0)
        self.axes(sc, pr, op, "sevens caught (recall)",
                  "calls that are right (precision, from 50%)",
                  grid=False, xticks=[(v, "%d%%" % (100 * v)) for v in (0, 0.5, 1.0)],
                  yticks=[(v, "%d%%" % (100 * v)) for v in (0.75, 1.0)])
        self.path(sc, pr, self.tpr[1:k:3], self.precision[1:k:3], TEAL, op, 3.0)
        self.text(sc, 0.60, 0.74, "the precision-recall curve", op, anchor="lm", ink=dark(TEAL),
                  size=0.034)
        self.text(sc, 0.5, 0.31, "both traced by the same sweep of the threshold, high to low",
                  op, size=0.034)

    def phase_4(self, sc, u, op):
        tpr, fpr = self.rates
        x, full = 0.36, self.span(0.44)[0]
        self.text(sc, 0.05, 0.33, "if sevens are", op, anchor="lm", strong=True, size=0.036)
        self.text(sc, x, 0.33, "of the images it calls a seven:", op, anchor="lm", strong=True,
                  size=0.036)
        rows = ((self.prevalence, "%.1f%% of images (measured)" % (100 * self.prevalence)),
                (0.01, "1% (same rates, arithmetic)"), (0.001, "0.1% (same rates, arithmetic)"))
        for r, (share, name) in enumerate(rows):
            y = 0.45 + r * 0.14
            show = op * smoothstep(0.12 * r + 0.05, 0.12 * r + 0.2, u)
            p = precision_at(tpr, fpr, share)
            self.lines(sc, 0.05, y, [name], show, size=0.032)
            w = full * p
            self.rect(sc, self.shift(self.P(x, y), w / 2, 0), w, 34, self.YES, show, radius=3)
            self.rect(sc, self.shift(self.P(x, y), w + (full - w) / 2, 0), full - w, 34, self.NO,
                      show, radius=3)
            self.text(sc, x + 0.008, y, "%.1f%% sevens" % (100 * p), show, anchor="lm", ink=WHITE,
                      size=0.034)
            self.text(sc, x + 0.445, y + 0.055, "%.1f%% false alarms" % (100 * (1 - p)), show,
                      anchor="rm", ink=dark(self.NO))
        self.text(sc, 0.05, 0.875, "at threshold 50%%: it catches %.1f%% of sevens and raises an "
                  "alarm on %.2f%% of everything else" % (100 * tpr, 100 * fpr), op, anchor="lm")

    def phase_5(self, sc, u, op):
        pane = Pane(((0.11, 0.45), (0.37, 0.80)), (0, 1), (0, 1))
        self.backdrop(sc, pane, op, pad=0.0)
        self.axes(sc, pane, op, "the score given", "share that were sevens", grid=False,
                  xticks=[(v, "%d%%" % (100 * v)) for v in (0, 0.5, 1.0)],
                  yticks=[(v, "%d%%" % (100 * v)) for v in (0.5, 1.0)])
        self.path(sc, pane, np.array([0.0, 1.0]), np.array([0.0, 1.0]), AXIS, op, 1.4)
        shown = int(np.clip(u / 0.5 * len(self.bins) + 0.5, 0, len(self.bins)))
        for n, (a, b, conf, freq, count) in enumerate(self.bins[:shown]):  # noqa: B007
            self.disk(sc, self.P(*pane.at(conf, freq)), 3.0 + 1.6 * np.log10(count) * 2.2, BLUE,
                      op * 0.9, ring=self.CUT if n == self.opened else None, ring_w=2.6, lift=1.3)
        self.text(sc, 0.44, 0.765, "on the line: an honest score", op, anchor="rm")
        self.text(sc, 0.28, 0.925, "bigger dot: more images with that score", op)
        a, b, conf, freq, count = self.bins[self.opened]
        members = np.flatnonzero((self.prob >= a) & (self.prob < b))
        x = 0.56
        self.text(sc, x, 0.335, "the %d images scored %.0f%% to %.0f%%" % (count, 100 * a, 100 * b),
                  op, anchor="lm", strong=True, size=0.036)
        for n, i in enumerate(members):
            ink = self.YES if self.seven[i] else self.NO
            self.digit(sc, self.images[i], x + 0.03 + 0.055 * (n % 7), 0.455 + 0.125 * (n // 7),
                       42, op * smoothstep(0.3 + 0.02 * n, 0.4 + 0.02 * n, u), frame=ink)
        late = op * smoothstep(0.65, 0.8, u)
        self.lines(sc, x, 0.70, [
            ("%d of them are sevens: %.0f%%" % (int(self.seven[members].sum()), 100 * freq),
             dark(self.YES)),
            ("their average score: %.0f%%" % (100 * conf), dark(BLUE)),
            ("%s of the 2,000 images score below" % f"{self.bins[0][4] + self.bins[-1][4]:,}", None),  # noqa: E501
            ("10% or above 90%: the middle of the", None),
            ("chart rests on a handful of images", None)],
            late, step=0.046, size=0.032)


ANIMATIONS = {
    "bias-variance": BiasVariance,
    "cross-validation": CrossValidation,
    "svm-and-kernels": SvmAndKernels,
    "gradient-boosting": GradientBoosting,
    "gaussian-mixture-em": GaussianMixtureEM,
    "curse-of-dimensionality": CurseOfDimensionality,
    "tsne-and-umap": TsneUmap,
    "roc-and-calibration": RocAndCalibration,
}

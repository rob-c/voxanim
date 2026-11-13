"""Sharp and flat minima: two minima of the MNIST MLP, found with minibatches
of 8 and of 1,024, and the loss landscape between and around them.

The network, 784-128-10 ReLU (He init), is trained by Adam on 10,000 MNIST
images for the same 40 epochs twice: with minibatches of 8 (50,000 steps,
lr 0.001) and of 1,024 (400 steps, lr 0.01, ten times the rate so that the
few steps reach a comparable training loss).  Both end near zero training
loss: two minima, 56 units apart in weight space.  Everything else is
measured on the trained weights (:mod:`voxanim.generalise`, cached):

* the **plane** through both minima: ``P_8 + u (P_1024 - P_8) + v d`` with
  ``d`` a random unit direction, the training loss (on 2,000 training
  images) on a 36 x 36 grid, drawn as the voxel height-field of
  :class:`voxanim.landscape.Landscape`, the two minima marked;
* the **line** between them (``v = 0``): the training and the test loss at
  31 points, a real one-dimensional slice -- the training loss rises to a
  small barrier between the two, the test loss does not rise;
* the loss **around each** along six random unit directions, averaged: the
  bowl's width, as the radius at which the training loss has risen by
  :data:`RISE` (0.01) -- the large-batch minimum is the sharper one;
* a **weight perturbation**: Gaussian noise of norm 8, 16 and 32 added to
  all the weights (five draws each): the test error of the sharp minimum
  climbs faster.

The test *loss* is lower at the sharp minimum even though its test *error*
is higher: the flat, small-batch network is the more confident of the two,
so its mistakes cost more cross-entropy.  The animation says so.

1. **two minima** -- the height-field rises; the two minima, their training
   loss and test error (3.5% for batch 8, 4.2% for batch 1,024),
2. **the line between** -- a walker crosses from one to the other, the chart
   draws the training and test loss along the way,
3. **around each** -- the two bowls, loss against distance,
4. **perturb** -- the test error after noise of each norm, both minima.

(The numbers are this machine's; CI retrains.)
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from . import color as C
from . import landscape as LS
from . import layout
from .cache import cached_npz
from .deepnet import Data, Spec, subset, train
from .easing import ease_in_out, ease_out, remap, smoothstep
from .gen_common import pct
from .generalise import (
    along_line,
    around,
    error_rate,
    loss_on,
    param_distance,
    perturbed,
    plane_grid,
    random_direction,
    weight_norm,
)
from .netanim import NetAnimation

N_TRAIN, N_TEST, SEED = 10000, 10000, 0
SIZES = (784, 128, 10)
EPOCHS = 40
SMALL = dict(batch=8, lr=0.001)                   # the small-batch run
LARGE = dict(batch=1024, lr=0.01)                 # the large-batch run
OPT = "adam"
N_SLICE = 2000                                    # training images the slices are measured on
LINE_U = (-0.25, 1.25, 31)                        # the line: from, to, points
RADIUS, N_RADII, N_DIRS, DIR_SEED = 48.0, 25, 6, 3   # the bowls: +-radius, points, directions
NORMS, N_TRIALS, NOISE_SEED = (8.0, 16.0, 32.0), 5, 5  # the perturbations
RISE = 0.01                                       # bowl width: where the loss has risen by this
GRID_N, GRID_U, GRID_V = 36, (-0.3, 1.3), (-RADIUS, RADIUS)
INK_SMALL, INK_LARGE = C.rgb("#1f6fb5"), C.rgb("#d84e4e")
TEST_INK = (0.40, 0.42, 0.46)
LAND_AT = (0.35, 0.60)
CHART = ((0.715, 0.945), (0.28, 0.56))
READ_X, READ_Y = 0.715, 0.63
CAP = 0.3                                         # loss at the top of the tallest column
WALKER = (0.98, 0.98, 0.97)


def dataset(cache_dir=None):
    """The 10,000-image problem, all 10,000 test digits."""
    raw = subset(N_TRAIN, N_TEST, None, SEED, cache_dir)
    return Data(raw.x_train, raw.y_train, raw.x_test, raw.y_test, 10)


def training_runs(data, cache_dir=None):
    """``(small, large)``: the two recorded runs, the same epochs each."""
    spec = Spec(SIZES, "relu", "he")
    out = []
    for cfg in (SMALL, LARGE):
        steps = EPOCHS * (-(-N_TRAIN // cfg["batch"]))
        kw = dict(cache_dir=cache_dir) if cache_dir else {}
        out.append(train(spec, data, steps, cfg["batch"], cfg["lr"], OPT, **kw))
    return tuple(out)


def measurements(data, small, large, cache_dir=None):
    """The slices, bowls and perturbations of the two minima (a dict of
    arrays, cached under the runs' settings)."""
    spec = small.spec
    settings = dict(sizes=list(SIZES), epochs=EPOCHS, small=SMALL, large=LARGE, opt=OPT,
                    n_slice=N_SLICE, line=LINE_U, radius=RADIUS, n_radii=N_RADII, dirs=N_DIRS,
                    dir_seed=DIR_SEED, norms=NORMS, trials=N_TRIALS, noise_seed=NOISE_SEED,
                    grid=[GRID_N, GRID_U, GRID_V], data=data.settings(),
                    checksum=float(sum(v.sum() for v in small.P.values()) +
                                   sum(v.sum() for v in large.P.values())))

    def build():
        Pa, Pb = small.P, large.P
        x, y = data.x_train[:N_SLICE], data.y_train[:N_SLICE]
        us = np.linspace(*LINE_U)
        radii = np.linspace(-RADIUS, RADIUS, N_RADII)
        rng = np.random.default_rng(DIR_SEED)
        d = random_direction(Pa, rng)
        gu, gv = np.linspace(*GRID_U, GRID_N), np.linspace(*GRID_V, GRID_N)
        prng = np.random.default_rng(NOISE_SEED)
        return dict(
            us=us, radii=radii,
            line_train=along_line(spec, Pa, Pb, us, x, y, 10),
            line_test=along_line(spec, Pa, Pb, us, data.x_test, data.y_test, 10),
            around_small=around(spec, Pa, np.random.default_rng(DIR_SEED + 1), radii, x, y, 10,
                                N_DIRS),
            around_large=around(spec, Pb, np.random.default_rng(DIR_SEED + 2), radii, x, y, 10,
                                N_DIRS),
            gu=gu, gv=gv, grid=plane_grid(spec, Pa, Pb, d, gu, gv, x, y, 10),
            perturbed=np.array([[perturbed(spec, P, prng, n, data.x_test, data.y_test, N_TRIALS)
                                 for n in NORMS] for P in (Pa, Pb)]),
            test_err=np.array([error_rate(spec, P, data.x_test, data.y_test) for P in (Pa, Pb)]),
            train_loss=np.array([loss_on(spec, P, data.x_train, data.y_train, 10)
                                 for P in (Pa, Pb)]),
            distance=np.array(param_distance(Pa, Pb)),
            norms=np.array([weight_norm(Pa), weight_norm(Pb)]))
    return cached_npz("minima", settings, build, cache_dir)


def runs(cache_dir=None):
    """``(data, small, large, measurements)``."""
    data = dataset(cache_dir)
    small, large = training_runs(data, cache_dir)
    return data, small, large, measurements(data, small, large, cache_dir)


def bowl_width(radii, profile, rise=RISE):
    """The smallest ``|r|`` at which ``profile`` (loss against ``radii``,
    averaged over directions) exceeds its centre value by ``rise``,
    interpolated between grid points; the largest radius when it never does."""
    c = int(np.argmin(np.abs(radii)))
    base = profile[c]
    best = float(abs(radii[-1]))
    for side in (slice(c, None, 1), slice(c, None, -1)):
        r, p = radii[side], profile[side] - base
        k = np.nonzero(p >= rise)[0]
        if len(k):
            k = k[0]
            frac = (rise - p[k - 1]) / max(p[k] - p[k - 1], 1e-12)
            best = min(best, abs(float(r[k - 1] + (r[k] - r[k - 1]) * frac)))
    return best


class Plane:
    """The 2-D slice as the ``prob`` a :class:`voxanim.landscape.Landscape`
    draws: ``grid`` returns the cached loss grid, ``loss`` interpolates it."""

    def __init__(self, gu, gv, L):
        self.gu, self.gv, self.L = gu, gv, L

    def grid(self, lo, hi, n):
        """``(us, vs, L)`` -- precomputed; the box asked for is the grid's."""
        return self.gu, self.gv, self.L

    def loss(self, w):
        """Bilinear interpolation of the grid at ``(u, v)``."""
        u = np.interp(w[0], self.gu, np.arange(len(self.gu)))
        v = np.interp(w[1], self.gv, np.arange(len(self.gv)))
        i, j = int(np.clip(u, 0, len(self.gu) - 2)), int(np.clip(v, 0, len(self.gv) - 2))
        a, b = u - i, v - j
        return float((1 - a) * (1 - b) * self.L[i, j] + a * (1 - b) * self.L[i + 1, j]
                     + (1 - a) * b * self.L[i, j + 1] + a * b * self.L[i + 1, j + 1])


@dataclass
class Frame:
    """What one instant shows."""

    back: float
    phase: int
    rise: float      # the landscape's entrance
    u: float         # the walker's position along the line (phase 2)
    grow: float      # bowls / bars growth


class SharpVsFlat(NetAnimation):
    """Two minima and the landscape between them (see the module docstring)."""

    PHASES = ("two minima", "the line between", "around each", "perturb")

    def __init__(self, data, small, large, m):
        super().__init__("Sharp and flat minima", "", 15.5, azimuth=33.0, elevation=34.0)
        self.data, self.small, self.large, self.m = data, small, large, m
        self.hold = 2.5
        self.plane = Plane(m["gu"], m["gv"], m["grid"])
        lo, hi = (m["gu"][0], m["gv"][0]), (m["gu"][-1], m["gv"][-1])
        self.land = LS.Landscape(self, self.plane, lo, hi, CAP, LAND_AT, n=GRID_N)
        self.prof = (m["around_small"].mean(0), m["around_large"].mean(0))
        self.width = tuple(bowl_width(m["radii"], p) for p in self.prof)
        line_hi = float(max(m["line_train"].max(), m["line_test"].max())) * 1.1
        self.line_chart = LS.LossChart(self, CHART, 1.0, 0.0, line_hi)
        bowl_hi = float(max(p.max() for p in self.prof)) * 1.1
        self.bowl_chart = LS.LossChart(self, CHART, 1.0, 0.0, bowl_hi)
        Tm, t = {}, 0.3
        Tm["rise"] = (t, t + 2.5)
        Tm["minima"] = (t + 2.0, t + 7.0)
        t += 7.0 + 0.3
        Tm["line"] = (t, t + 9.0)
        t += 9.0 + 2.0
        Tm["bowls"] = (t, t + 6.0)
        t += 6.0 + 2.0
        Tm["perturb"] = (t, t + 5.0)
        t += 5.0 + 2.0
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["line"][0], Tm["bowls"][0], Tm["perturb"][0]]

    # ------------------------------------------------------------------ state
    def _frame(self, t, reset):
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        ph = 0 if reset > 0 else int(np.searchsorted(self.marks, t, side="right") - 1)
        rise = self.progress("rise", t)
        u = ease_in_out(self.progress("line", t)) if ph == 1 else (1.0 if ph > 1 else 0.0)
        grow = ease_out(self.progress("bowls" if ph == 2 else "perturb", t)) if ph >= 2 else 0.0
        return Frame(back, ph, rise, u, grow)

    def line_point(self, u):
        """World point above the surface at ``(u, 0)``."""
        us, L = self.m["us"], self.m["line_train"]
        return self.land.point(np.array([u, 0.0]), float(np.interp(u, us, L)))

    # ------------------------------------------------------------------ drawing
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t``."""
        f = self._frame(t, reset)
        self.subtitle = self._words(f)
        back = f.back
        self.land.show(LS.reveal_curve(f.rise), back)
        self.land.axes(sc, smoothstep(0.5, 1.0, f.rise) * back, labels=False)
        self._axis_labels(sc, smoothstep(0.5, 1.0, f.rise) * back)
        on = smoothstep(0.0, 0.5, self.progress("minima", t)) * back
        self._minima(sc, on)
        if f.phase == 1:
            self._line(sc, f, back)
        elif f.phase == 2:
            self._bowls(sc, f, back)
        elif f.phase == 3:
            self._perturb(sc, f, back)
        else:
            self._readout(sc, on)
        self.step_indicator(sc, f.phase, self.PHASES)

    def _axis_labels(self, sc, op):
        """What the two axes of the plane are."""
        if op <= 0.01:
            return
        lo, hi = self.land.bbox()
        cam = self.cam
        p = cam.project(np.array([(lo[0] + hi[0]) / 2, hi[1] + 0.7 * self.land.block.scale,
                                  lo[2]]))
        sc.text(p[0] / cam.width, p[1] / cam.height + 0.03, "u: batch 8 (0) → batch 1,024 (1)",
                0.027, "regular", layout.CAPTION_INK, op * self._cap, "mt")
        q = cam.project(np.array([lo[0] - 0.7 * self.land.block.scale, (lo[1] + hi[1]) / 2,
                                  lo[2]]))
        sc.text(q[0] / cam.width - 0.02, q[1] / cam.height, "v: a random direction, ±%d"
                % RADIUS, 0.027, "regular", layout.CAPTION_INK, op * self._cap, "rm")

    def _minima(self, sc, op):
        """The two minima on the surface, labelled."""
        if op <= 0.01:
            return
        for u, ink, name in ((0.0, INK_SMALL, "batch 8"), (1.0, INK_LARGE, "batch 1,024")):
            L = float(np.interp(u, self.m["us"], self.m["line_train"]))
            self.land.marker(sc, np.array([u, 0.0]), L, ink, op, size=1.1, glow=0.4)
            p = self.land.point(np.array([u, 0.0]), L, lift=2.5)
            s = self.cam.project(p)
            sc.text(s[0] / self.cam.width, s[1] / self.cam.height - 0.01, name, 0.028,
                    "regular", ink, op * self._cap, "mb")

    def _readout(self, sc, op):
        """The two runs' numbers (phase 1)."""
        m = self.m
        lines = [("the two minima", layout.STRONG_INK),
                 ("batch 8: %s steps, training loss %.4f, test error %s"
                  % (f"{len(self.small.batch_loss):,}", m["train_loss"][0], pct(m["test_err"][0])),
                  INK_SMALL),
                 ("batch 1,024: %s steps, training loss %.4f, test error %s"
                  % (f"{len(self.large.batch_loss):,}", m["train_loss"][1], pct(m["test_err"][1])),
                  INK_LARGE),
                 ("%.0f apart in weight space (|W| %.0f and %.0f)"
                  % (m["distance"], m["norms"][0], m["norms"][1]), layout.CAPTION_INK)]
        self._lines(sc, lines, op, y=0.30)

    def _lines(self, sc, lines, op, y=READ_Y, dy=0.045, x=READ_X):
        op *= self._cap
        if op <= 0.01:
            return
        for n, (text, ink) in enumerate(lines):
            sc.text(x, y + n * dy, text, 0.03 if n == 0 else 0.027, "regular", ink, op, "lm")

    def _line(self, sc, f, back):
        """The walker along the line and the chart of both losses along it."""
        m = self.m
        us = m["us"]
        u = us[0] + (us[-1] - us[0]) * f.u
        # the path so far on the surface
        pts = [self.line_point(v) for v in np.linspace(us[0], u, 40)]
        for a, b in itertools.pairwise(pts):
            sc.line(a, b, width=1.6, color=WALKER, opacity=0.9 * back, bias=0.15)
        self.land.marker(sc, np.array([u, 0.0]), float(np.interp(u, us, m["line_train"])),
                         WALKER, back, size=0.8, key=("walker",))
        chart = self.line_chart
        chart.frame(sc, back, xlabel="u", ylabel="loss")
        k = (u - us[0]) / (us[-1] - us[0]) * (len(us) - 1)
        chart.curve(sc, m["line_train"], k, INK_SMALL, back)
        chart.curve(sc, m["line_test"], k, TEST_INK, back)
        for v, ink in ((0.0, INK_SMALL), (1.0, INK_LARGE)):
            j = (v - us[0]) / (us[-1] - us[0]) * (len(us) - 1)
            a = chart.pt(j, 0.0)
            b = chart.pt(j, chart.hi)
            sc.line(a, b, width=0.9, color=ink, opacity=0.5 * back, bias=0.0)
        i = int(round(k))
        lines = [("along the line, u = %.2f" % u, layout.STRONG_INK),
                 ("training loss %.3f" % m["line_train"][i], INK_SMALL),
                 ("test loss %.3f" % m["line_test"][i], TEST_INK),
                 ("the barrier between: training loss up to %.3f"
                  % m["line_train"][(us >= 0) & (us <= 1)].max(), layout.CAPTION_INK)]
        self._lines(sc, lines, back)

    def _bowls(self, sc, f, back):
        """Loss against distance along random directions, both minima."""
        m = self.m
        radii = m["radii"]
        chart = self.bowl_chart
        chart.frame(sc, back, xlabel="distance r", ylabel="training loss")
        k = f.grow * (len(radii) - 1)
        for prof, ink in zip(self.prof, (INK_SMALL, INK_LARGE)):
            chart.curve(sc, prof, k, ink, back, marker=False)
        # draw the same bowls as rings on the surface: the width of each
        for u, w, ink in ((0.0, self.width[0], INK_SMALL), (1.0, self.width[1], INK_LARGE)):
            L = float(np.interp(u, m["us"], m["line_train"]))
            c = self.land.point(np.array([u, 0.0]), L)
            kx = (GRID_N - 1) * self.land.block.scale / (GRID_U[1] - GRID_U[0])
            ky = (GRID_N - 1) * self.land.block.scale / (GRID_V[1] - GRID_V[0])
            th = np.linspace(0, 2 * np.pi, 37)
            ring = [c + np.array([w / m["distance"] * kx * np.cos(a) * f.grow,
                                  w * ky * np.sin(a) * f.grow, 0.0]) for a in th]
            for a, b in itertools.pairwise(ring):
                sc.line(a, b, width=1.2, color=ink, opacity=0.9 * back, bias=0.25)
        lines = [("the bowls: loss along %d random directions" % N_DIRS, layout.STRONG_INK),
                 ("batch 8: up %.2f within r = %.1f" % (RISE, self.width[0]), INK_SMALL),
                 ("batch 1,024: up %.2f within r = %.1f" % (RISE, self.width[1]), INK_LARGE),
                 ("%.1f× sharper" % (self.width[0] / self.width[1]), layout.CAPTION_INK)]
        self._lines(sc, lines, back * smoothstep(0.5, 1.0, f.grow))

    def _perturb(self, sc, f, back):
        """Test error after noise of each norm, as bars beside the chart box."""
        m = self.m
        (x0, x1), (y0, y1) = CHART
        base = y1
        hmax = (y1 - y0) * 0.9
        top = float(m["perturbed"].max()) * 1.15
        groups = len(NORMS) + 1
        cam = self.cam
        for g in range(groups):
            fx = x0 + (x1 - x0) * (g + 0.5) / groups
            for s, (ink, row) in enumerate(((INK_SMALL, 0), (INK_LARGE, 1))):
                err = m["test_err"][row] if g == 0 else m["perturbed"][row, g - 1]
                h = hmax * err / top * f.grow
                bx = fx + (s - 0.5) * 0.028
                a, b = cam.frac(bx, base), cam.frac(bx, base - h)
                sc.line(a, b, width=9.0, color=ink, opacity=0.95 * back, bias=0.2)
                tx, ty = self.fr(bx, base - h)
                sc.text(tx, ty - 0.012, pct(err), 0.026, "regular", ink,
                        back * self._cap * smoothstep(0.6, 1.0, f.grow), "mb")
            tx, ty = self.fr(fx, base)
            sc.text(tx, ty + 0.012, "no noise" if g == 0 else "norm %d" % NORMS[g - 1], 0.027,
                    "regular", layout.CAPTION_INK, back * self._cap, "mt")
        sc.line(cam.frac(x0, base), cam.frac(x1, base), width=1.0, color=LS.CHART_INK,
                opacity=0.8 * back, bias=0.0)
        tx, ty = self.fr(x0, y0 - 0.012)
        sc.text(tx, ty, "test error after weight noise", 0.027, "regular", layout.CAPTION_INK,
                back * self._cap, "lb")
        lines = [("noise of norm %d, %d draws each" % (NORMS[-1], N_TRIALS), layout.STRONG_INK),
                 ("batch 8: %s → %s" % (pct(m["test_err"][0]), pct(m["perturbed"][0, -1])),
                  INK_SMALL),
                 ("batch 1,024: %s → %s" % (pct(m["test_err"][1]), pct(m["perturbed"][1, -1])),
                  INK_LARGE)]
        self._lines(sc, lines, back * smoothstep(0.5, 1.0, f.grow), y=READ_Y + 0.06)

    # ------------------------------------------------------------------ words
    def _words(self, f):
        """The subtitle for frame ``f``."""
        m = self.m
        return [
            "a %d-%d-%d network trained twice for %d epochs on %s images: minibatches of %d "
            "(lr %g) and of %s (lr %g); the loss on the plane through both"
            % (*SIZES, EPOCHS, f"{N_TRAIN:,}", SMALL["batch"], SMALL["lr"],
               f"{LARGE['batch']:,}", LARGE["lr"]),
            "the straight line between them: the training loss climbs to %.3f in between, "
            "the test loss goes from %.2f to %.2f"
            % (m["line_train"][(m["us"] >= 0) & (m["us"] <= 1)].max(),
               m["line_test"][np.argmin(np.abs(m["us"]))], m["line_test"][np.argmin(
                   np.abs(m["us"] - 1))]),
            "along random directions the loss rises by %.2f within %.1f units of the batch-%s "
            "minimum, %.1f of the batch-8 one: the large batch found the sharp one"
            % (RISE, self.width[1], f"{LARGE['batch']:,}", self.width[0]),
            "noise of norm %d on the weights: test error %s → %s at the flat minimum, %s → %s at "
            "the sharp one" % (NORMS[-1], pct(m["test_err"][0]), pct(m["perturbed"][0, -1]),
                               pct(m["test_err"][1]), pct(m["perturbed"][1, -1])),
        ][f.phase]

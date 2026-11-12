"""Learning the hyperparameters: gradient ascent on the log marginal likelihood.

    plot: 12 noisy observations and the      the landscape log p(y) over
    posterior for the hyperparameters        (lengthscale, noise sd), a field of
    on show, changing in step                voxel columns; the walker on it with
                                             its gradient arrow
                                             bars: data fit, complexity, log p(y)

The RBF kernel's variance is fixed at 1; the lengthscale l and the noise sd
s_n are learned.  log p(y) = -1/2 y^T K^-1 y - 1/2 log|K| - n/2 log 2 pi: the
first term rewards fitting the data, the second punishes a flexible model
(a K with large determinant), so the optimum is a trade-off.

1. **landscape** -- log p(y) evaluated on a 24 x 24 grid of (l, s_n), both on
   log axes: the columns rise to their heights (floored 40 below the top),
2. **start** -- too smooth: l = 6, s_n = 1.5.  The posterior mean is nearly
   flat, the band wide; the data-fit bar is poor,
3. **ascent** -- gradient ascent in (log l, log s_n) with step 0.05 x the
   gradient, 40 steps (:func:`voxanim.gp.ascend`, cached): the arrow is the
   next step, the posterior tightens on the data as l shrinks and s_n falls,
   the fit bar improves while the complexity bar worsens,
4. **optimum** -- l = 1.8, s_n = 0.24, where the two trade off: log p(y) = -8.9.

Every frame's posterior and bars are computed for the hyperparameters on
show (interpolated geometrically between steps).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import color as C
from . import gp, layout
from .cache import cached_npz
from .conv import Tempo
from .easing import ease_in_out, ease_out, remap, smoothstep
from .gp_anim import MEAN, MEAN_INK, PANEL_EDGE, Beats, GPDiagram, log_lerp
from .netanim import div_colors
from .tensor import Block
from .text import num

XR, YR = (0.0, 10.0), (-3.0, 3.0)
YTICKS = (-2.0, 0.0, 2.0)
PLOT = (0.05, 0.31, 0.52, 0.93)
N_FINE = 121
N_OBS, NOISE_SD, DATA_SEED = 12, 0.3, 7         # the observations
VARIANCE = 1.0                                   # the kernel variance (fixed)
L_RANGE, N_RANGE = (0.1, 10.0), (0.03, 3.0)      # the landscape's axes (log scale)
N_L, N_N = 24, 24                                # columns per axis
LEVELS, DEPTH = 18, 40.0                         # column levels; log p(y) range they span
LAND_LO, LAND_HI = C.rgb("#1d4f4c"), C.rgb("#d2fff7")   # column colour: floor .. top
START = (6.0, 1.5)                               # (lengthscale, noise sd): too smooth
LR, STEPS = 0.05, 40                             # gradient ascent in log space
LAND_AT, LAND_SCALE = (0.75, 0.44), 0.80         # the landscape: layout position, voxel size
BARS = (0.60, 0.845, 0.955)                      # bars panel: x0, first row y, x1 (layout)
BAR_ROW, BAR_HALF, BAR_SPAN = 0.052, 0.085, 20.0 # row spacing; half-width (layout); its units
BAR_LABELS = (("data fit", "fit"), ("complexity", "complexity"), ("log p(y)", "value"))
ARROW_HEAD, ARROW_W = 7.0, 3.0                   # px
ARROW_PX, ARROW_MAX = 60.0, 80.0                 # the first gradient's arrow length; the longest
ARROW = C.POSITIVE                               # orange: distinct from the teal landscape
ARROW_BIAS = 12.0                                # world units: the arrow wins over the columns
ARROW_S, MOVE_MAX = 1.4, 1.2                     # the first arrow's showing; a move's longest
WALKER = C.rgb("#f6f3e6")


def truth(x):
    """The function the observations come from."""
    x = np.asarray(x, np.float64)
    return np.sin(x) + 0.3 * x - 1.5


def dataset(seed=DATA_SEED):
    """``(X, y)``: 12 observations at seeded random x, noise sd NOISE_SD."""
    rng = np.random.default_rng(seed)
    X = np.sort(rng.uniform(0.3, 9.7, N_OBS))
    return X, truth(X) + rng.normal(0.0, NOISE_SD, N_OBS)


def kernel(l):
    """The RBF kernel with lengthscale ``l`` (variance fixed)."""
    return gp.RBF(variance=VARIANCE, lengthscale=l)


def landscape():
    """The cached numbers: ``Z`` ``(N_L, N_N)`` log p(y) over the grid, and the
    ascent's ``path`` ``(STEPS + 1, 2)`` of (lengthscale, noise sd)."""
    X, y = dataset()

    def build():
        ls = np.exp(np.linspace(np.log(L_RANGE[0]), np.log(L_RANGE[1]), N_L))
        ns = np.exp(np.linspace(np.log(N_RANGE[0]), np.log(N_RANGE[1]), N_N))
        Z = np.array([[gp.log_marginal(kernel(l), X, y, n * n).value for n in ns] for l in ls])
        ks, nz, _evs = gp.ascend(kernel(START[0]), X, y, START[1] ** 2,
                                 ["lengthscale", "noise"], LR, STEPS)
        path = np.array([[k.p["lengthscale"], np.sqrt(n)] for k, n in zip(ks, nz)])
        return dict(Z=Z, path=path, ls=ls, ns=ns)
    settings = dict(n=N_OBS, noise=NOISE_SD, seed=DATA_SEED, variance=VARIANCE, l=L_RANGE,
                    sd=N_RANGE, grid=(N_L, N_N), start=START, lr=LR, steps=STEPS)
    return cached_npz("gp_hyper", settings, build)


@dataclass
class Frame:
    """What one frame shows (see :meth:`HyperAnimation._frame`)."""

    back: float                  # 1, fading to 0 in the loop reset
    intro: float                 # axes and data fading in
    rise: float                  # the landscape's columns rising (0..1)
    walker: float                # the walker's and posterior's visibility
    step: int                    # the step the walker has reached
    u: float                     # progress of its move to the next step
    arrow: float                 # the gradient arrow's visibility
    l: float                     # lengthscale on show
    sd: float                    # noise sd on show


class HyperAnimation(GPDiagram):
    """Gradient ascent on log p(y) over (lengthscale, noise) beside the posterior."""

    PHASES = ("landscape", "start", "ascent", "optimum")

    def __init__(self):
        super().__init__("Learning the hyperparameters", XR, YR, plot=PLOT, azimuth=33.0,
                         elevation=28.0)
        self.xs = np.linspace(*XR, N_FINE)
        self.X, self.y = dataset()
        land = landscape()
        self.Z, self.path = land["Z"], land["path"]
        self.zmax = float(self.Z.max())
        self.land = Block((LEVELS, N_L, N_N), scale=LAND_SCALE, name="landscape")
        self.land.place(self.cam.frac(*LAND_AT))
        self.blocks_extra.append(self.land)
        self._timeline()
        self.hold, self.reset = 2.5, 1.2
        self.finish(self.b.t)
        self.final = self.evidence(*self.path[-1])
        self.g0 = float(np.linalg.norm(self.log_gradient(*self.path[0])))

    def _timeline(self):
        b = Beats()
        b.add("intro", 1.0, 0.3)
        b.add("landscape", 3.2, 0.4)
        b.add("start", 3.0, 0.2)
        b.add("first", ARROW_S + MOVE_MAX + 1.0, 0.0)          # the first step, slowly
        d = Tempo(slow=1.0, fast=0.26, ramp_in=6.0, ramp_out=4.0).durations(STEPS - 1)
        # when each move (step k to k + 1) starts, and how long it takes
        self.step_t = np.concatenate([[b.Tm["first"][0] + ARROW_S], b.t + np.cumsum(d) - d[0]])
        self.move_d = np.minimum(np.diff(np.append(self.step_t, b.t + d.sum())), MOVE_MAX)
        b.add("ascent", float(d.sum()), 0.3)
        b.add("optimum", 3.5, 0.0)
        self.b, self.Tm = b, b.Tm

    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        for k, name in ((3, "optimum"), (2, "first"), (1, "start")):
            if t >= self.Tm[name][0]:
                return k
        return 0

    # ------------------------------------------------------------------ numbers
    def evidence(self, l, sd):
        """log p(y) and its parts at ``(l, sd)``."""
        return gp.log_marginal(kernel(l), self.X, self.y, sd * sd)

    def log_gradient(self, l, sd):
        """The gradient of log p(y) in the ascent's coordinates (log l, log sd^2),
        drawn on the landscape's axes (log l, log sd): a step is LR times this,
        so the arrow points where the walker goes."""
        g = self.evidence(l, sd).grad
        return np.array([g["lengthscale"] * l, 0.5 * g["noise"] * sd * sd])

    def level(self, z):
        """Column height, in levels (0..LEVELS), of a log p(y) value."""
        return LEVELS * float(np.clip(1.0 + (z - self.zmax) / DEPTH, 0.0, 1.0))

    def land_xy(self, l, sd):
        """Fractional column coordinates of ``(l, sd)`` on the landscape."""
        i = (np.log(l) - np.log(L_RANGE[0])) / (np.log(L_RANGE[1]) - np.log(L_RANGE[0]))
        j = (np.log(sd) - np.log(N_RANGE[0])) / (np.log(N_RANGE[1]) - np.log(N_RANGE[0]))
        return i * (N_L - 1), j * (N_N - 1)

    def land_point(self, l, sd, z, lift=0.0):
        """World point above column ``(l, sd)`` at height ``z`` log p(y)."""
        i, j = self.land_xy(l, sd)
        b = self.land
        s = b.scale
        return b.origin + np.array([(i + 0.5) * s, (j + 0.5) * s, self.level(z) * s + lift])

    # ------------------------------------------------------------------ state
    def _frame(self, t, reset):
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        intro = smoothstep(0.0, 1.0, self.progress("intro", t)) * back
        rise = ease_out(self.progress("landscape", t)) * back
        walker = smoothstep(0.0, 0.8, self.progress("start", t)) * back
        k = int(np.searchsorted(self.step_t, t, side="right")) - 1   # the move under way
        u = ease_in_out(remap(t, self.step_t[k], self.step_t[k] + self.move_d[k])) \
            if k >= 0 else 0.0
        step = max(k, 0) + (u >= 1.0)
        u = 0.0 if u >= 1.0 else u
        a, c = self.path[step], self.path[min(step + 1, STEPS)]
        l, sd = log_lerp(a[0], c[0], u), log_lerp(a[1], c[1], u)
        # the arrow of the next step shows while the walker stands still
        since = t - self.Tm["first"][0] if k < 0 else t - self.step_t[k] - self.move_d[k]
        arrow = smoothstep(0.0, 0.5, since) * (u <= 0.0) * (step < STEPS) * walker
        return Frame(back, intro, rise, walker, step, u, arrow, float(l), float(sd))

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        f = self._frame(t, reset)
        ev = self.evidence(f.l, f.sd)
        self.axes(sc, f.intro, yticks=YTICKS)
        self.points(sc, self.X, self.y, f.intro)
        self._posterior(sc, f)
        self._landscape(sc, f)
        self._walker(sc, f, ev)
        self._bars(sc, f, ev)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, f, ev, reset)

    def _posterior(self, sc, f):
        if f.walker <= 0.01:
            return
        post = gp.posterior(kernel(f.l), self.X, self.y, self.xs, f.sd ** 2)
        self.band(sc, self.xs, post.mu, post.sd, f.walker)
        self.curve(sc, self.xs, post.mu, MEAN, width=2.2, op=0.95 * f.walker)
        self.label(sc, self.pt(XR[1], post.mu[-1]), "μ*", layout.CAPTION, anchor="lm", dx=8,
                   opacity=0.8 * f.walker, ink=MEAN_INK, kind="mono")
        self.label(sc, self.pt(XR[0], YR[0]), "ℓ = %s   σₙ = %s" % (num(f.l), num(f.sd)),
                   layout.CAPTION, anchor="lb", dx=8, dy=-8, opacity=f.walker,
                   ink=layout.STRONG_INK)

    def _landscape(self, sc, f):
        """The columns: height and colour from log p(y); the floor is DEPTH below
        the top."""
        b = self.land
        b.reset()
        if f.rise <= 0.01:
            b.present[...] = False
            return
        h = np.array([[self.level(z) for z in row] for row in self.Z]) * f.rise
        levels = np.arange(LEVELS)[::-1]                    # row 0 is the top level
        b.present[...] = levels[:, None, None] < np.ceil(h)[None]
        a = (h / LEVELS) ** 2                                # the ridge bright, the floor dark
        b.base[...] = (LAND_LO + (LAND_HI - LAND_LO) * a.ravel()[:, None]).reshape(1, N_L, N_N, 3)
        b.opacity *= f.back
        lo, hi = b.origin, b.origin + b.extent
        self.caption(sc, (lo, hi), "log p(y) over lengthscale ℓ (%s → %s, to the right)\n"
                     "and noise σₙ (%s → %s, towards you)"
                     % (num(L_RANGE[0], 1), num(L_RANGE[1], 0), num(N_RANGE[0], 2),
                        num(N_RANGE[1], 0)), f.rise, size=layout.MIN_TEXT, max_width=0.40)

    def _walker(self, sc, f, ev):
        """The walker on the landscape and the arrow of its next step."""
        if f.walker <= 0.01:
            return
        p = self.land_point(f.l, f.sd, ev.value, lift=0.6 * LAND_SCALE)
        sc.voxel(p, WALKER, f.walker, glow=0.5, shadow=False, reflect=False,
                 size=0.9 * LAND_SCALE)
        sc.probe("walker", p, tol=4.0)
        if f.arrow > 0.01 and f.step < STEPS:
            g = self.log_gradient(f.l, f.sd)
            px = min(ARROW_MAX, ARROW_PX * float(np.linalg.norm(g)) / self.g0)
            # the arrow points along the gradient in the landscape's log coordinates
            # and its tip rests on the surface there (uphill: into taller columns)
            d = np.array([g[0] / np.log(L_RANGE[1] / L_RANGE[0]) * N_L,
                          g[1] / np.log(N_RANGE[1] / N_RANGE[0]) * N_N])
            d = d / max(np.linalg.norm(d), 1e-9)
            on_screen = np.linalg.norm(self.Mr @ np.array([d[0], d[1], 0.0])) * LAND_SCALE
            d = d * px / max(on_screen, 1e-9)              # columns, for ``px`` on screen
            l2 = f.l * (L_RANGE[1] / L_RANGE[0]) ** (d[0] / N_L)
            sd2 = f.sd * (N_RANGE[1] / N_RANGE[0]) ** (d[1] / N_N)
            q = self.land_point(l2, sd2, self.evidence(l2, sd2).value, lift=0.6 * LAND_SCALE)
            self._arrow(sc, p, q, f.arrow)

    def _arrow(self, sc, a, b, op):
        """An arrow from ``a`` to ``b`` (world points), head in the view plane."""
        sc.line(a, b, width=ARROW_W, color=ARROW, opacity=op, bias=ARROW_BIAS)
        sa, sb = self.px(a), self.px(b)
        d = sb - sa
        n = np.linalg.norm(d)
        if n < 1e-6:
            return
        d = d / n
        side = np.array([-d[1], d[0]])
        for s in (1, -1):
            tip = b + self.Mi @ (-d * ARROW_HEAD + side * s * ARROW_HEAD * 0.6)
            sc.line(b, tip, width=ARROW_W, color=ARROW, opacity=op, bias=ARROW_BIAS)

    def _bars(self, sc, f, ev):
        """Signed bars of the data-fit term, the complexity term and their sum."""
        op = f.walker
        if op <= 0.01:
            return
        x0, y0, x1 = BARS
        xm = (x0 + x1) / 2 + 0.03
        for r, (label, attr) in enumerate(BAR_LABELS):
            y = y0 + r * BAR_ROW
            v = getattr(ev, attr)
            self.label(sc, self.P(x0, y), label, layout.MIN_TEXT, anchor="lm", opacity=op,
                       ink=layout.STRONG_INK if attr == "value" else layout.CAPTION_INK)
            self.label(sc, self.P(x1, y), num(v, 1), layout.MIN_TEXT, anchor="rm", opacity=op,
                       ink=layout.STRONG_INK if attr == "value" else layout.CAPTION_INK,
                       kind="mono")
            a, c = self.P(xm - BAR_HALF, y), self.P(xm + BAR_HALF, y)
            w = (self.px(c) - self.px(a))[0]
            self.rect(sc, (a + c) / 2, w, 9, (1, 1, 1), op * 0.8, border=PANEL_EDGE,
                      border_w=0.8, radius=2, lift=0.1, fill_op=0.6)
            k = float(np.clip(v / BAR_SPAN, -1.0, 1.0))
            if abs(k) > 1e-3:
                centre = self.P(xm + BAR_HALF * k / 2, y)
                self.rect(sc, centre, abs(w * k / 2), 7, div_colors([v], BAR_SPAN)[0], op,
                          radius=1, lift=0.3)
            sc.line(self.P(xm, y, 0.4) + self.Mi @ np.array([0, -6.0]),
                    self.P(xm, y, 0.4) + self.Mi @ np.array([0, 6.0]), width=1.0,
                    color=PANEL_EDGE, opacity=op, bias=0.0)

    # ------------------------------------------------------------------ words
    def _words(self, t, f, ev, reset):
        """The subtitle: what is happening now, with the numbers."""
        if reset > 0 or t < self.Tm["landscape"][0]:
            return ("%d noisy observations. An RBF kernel with variance 1: which lengthscale ℓ "
                    "and noise σₙ? Maximise log p(y), the marginal likelihood" % N_OBS)
        if t < self.Tm["start"][0]:
            return ("log p(y) for every (ℓ, σₙ): a landscape with one ridge; its top is at "
                    "ℓ ≈ %s, σₙ ≈ %s (log p(y) = %s)"
                    % (num(self.path[-1, 0], 1), num(self.path[-1, 1], 2), num(self.zmax, 1)))
        if t < self.Tm["first"][0]:
            return ("start too smooth, ℓ = %s, σₙ = %s: a big noise explains anything, so the "
                    "data-fit term %s is easy, but the complexity term %s pays for it"
                    % (num(START[0], 0), num(START[1], 1), num(ev.fit, 1),
                       num(ev.complexity, 1)))
        if t < self.Tm["ascent"][0]:
            return ("the arrow: the gradient of log p(y) in (log ℓ, log σₙ²). Each step moves "
                    "%s × the gradient, uphill" % num(LR, 2))
        if t < self.Tm["optimum"][0]:
            return ("step %d: ℓ = %s, σₙ = %s. As σₙ falls the complexity term rises to %s while "
                    "the data fit drops to %s; their sum log p(y) = %s climbs"
                    % (f.step, num(f.l), num(f.sd), num(ev.complexity, 1), num(ev.fit, 1),
                       num(ev.value, 1)))
        return ("optimum after %d steps: ℓ = %s, σₙ = %s. log p(y) = %s = %s %s %s − %s: "
                "neither term alone, the trade-off between them"
                % (STEPS, num(self.final_l), num(self.final_sd), num(self.final.value, 1),
                   num(self.final.fit, 1), "+" if self.final.complexity >= 0 else "−",
                   num(abs(self.final.complexity), 1), num(-self.final.const, 1)))

    @property
    def final_l(self):
        """The lengthscale the ascent ends at."""
        return float(self.path[-1, 0])

    @property
    def final_sd(self):
        """The noise sd the ascent ends at."""
        return float(self.path[-1, 1])

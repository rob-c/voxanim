"""A Gaussian process prior: the kernel, its matrix, sample functions.

    plot of f(x) on [0, 10]            the kernel matrix K over 24 grid points
    (the kernel bump, samples,         (a sheet of voxels, orange = large)
     the mean and the +-2 sd band)     read-outs: lengthscale, variance

1. **kernel** -- the RBF kernel k(x, x') = s^2 exp(-(x - x')^2 / 2 l^2)
   drawn as a function of x for x' = 5: how strongly f(x) is tied to f(x'),
2. **matrix** -- x' sweeps along the axis and each row of K = k(x_i, x_j)
   over 24 grid points fills in as it passes: a band along the diagonal,
3. **samples** -- functions drawn from N(0, K) over a fine grid, one after
   another (the same seeded draws throughout), then the mean (0) and the
   band of +-2 standard deviations that 95% of samples stay inside,
4. **lengthscale** -- l goes 1 -> 0.3 -> 3 -> 1: the diagonal band narrows
   and widens, the samples wiggle and smooth out (each is ``L z`` for the
   same ``z``, so they move continuously),
5. **variance** -- s^2 goes 1 -> 2.25 -> 0.25 -> 1: the band and the samples
   scale, the matrix brightens and dims.

Every value is computed by :mod:`voxanim.gp` for the hyperparameters on
show in that frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import gp, layout
from .conv import Tempo
from .easing import ease_in_out, remap, smoothstep
from .gp_anim import FOCUS, MEAN, MEAN_INK, SAMPLES, Beats, GPDiagram, log_lerp
from .text import num

XR, YR = (0.0, 10.0), (-3.5, 3.5)           # the plot's ranges
YTICKS = (-3.0, 0.0, 3.0)
N_FINE = 121                                 # grid for the curves
N_GRID = 24                                  # grid for the kernel matrix
X_MARK = 5.0                                 # x' of the kernel bump in the first phase
L0, VAR0 = 1.0, 1.0                          # the starting hyperparameters
L_PATH = (0.3, 3.0, 1.0)                     # the lengthscales visited, in order
VAR_PATH = (2.25, 0.25, 1.0)                 # the variances visited, in order
N_SAMPLES = 3
SEED = 3
SHEET_AT, SHEET_SCALE = (0.80, 0.57), 0.62   # the kernel matrix: layout position, voxel size
MOVE, HOLD = 3.4, 1.4                        # seconds per hyperparameter move and hold after
DRAW_S, GAP_S = 1.3, 1.6                     # a sample's drawing time; the next starts after
BAND_S = 1.6                                 # the mean and band fading in


@dataclass
class Frame:
    """What one frame shows (see :meth:`PriorAnimation._frame`)."""

    back: float                  # 1, fading to 0 in the loop reset
    intro: float                 # the axes fading in
    l: float                     # lengthscale on show
    var: float                   # variance on show
    bump: float                  # the kernel-function curve's visibility
    grow: float                  # how far the bump has grown out from x' (0..1)
    x_mark: float                # x' of the kernel bump
    started: bool                # the matrix phase has begun (cells may show)
    rows: int                    # rows of the matrix filled in
    row_u: float                 # progress through the current row
    samples: list                # per sample: how much of it is drawn (0..1)
    band: float                  # the mean and band's visibility


class PriorAnimation(GPDiagram):
    """The prior: kernel, matrix, samples, then the hyperparameters move."""

    PHASES = ("kernel", "matrix", "samples", "lengthscale", "variance")

    def __init__(self):
        super().__init__("A Gaussian process prior", XR, YR)
        self.xs = np.linspace(*XR, N_FINE)
        self.grid = np.linspace(*XR, N_GRID)
        self.z = np.random.default_rng(SEED).standard_normal((N_SAMPLES, N_FINE))
        self.K = self.sheet((N_GRID, N_GRID), SHEET_AT, SHEET_SCALE, "K")
        self._timeline()
        self.hold, self.reset = 2.5, 1.2
        self.finish(self.b.t)

    def _timeline(self):
        b = Beats()
        b.add("intro", 1.0, 0.0)
        b.add("kernel", 3.6, 0.4)
        self.row_d = Tempo(slow=0.55, fast=0.13, ramp_in=4.0, ramp_out=3.0).durations(N_GRID)
        self.row_t = b.t + np.concatenate([[0.0], np.cumsum(self.row_d)])
        b.add("matrix", float(self.row_d.sum()), 0.6)
        self.sample_t = [b.t + k * GAP_S for k in range(N_SAMPLES)]
        b.add("samples", GAP_S * (N_SAMPLES - 1) + DRAW_S, 0.2)
        b.add("band", BAND_S, 0.5)
        self.l_moves = self._moves(b, "lengthscale", L0, L_PATH)
        self.var_moves = self._moves(b, "variance", VAR0, VAR_PATH)
        self.b, self.Tm = b, b.Tm

    @staticmethod
    def _moves(b, name, start, path):
        """Schedule the hyperparameter moves of phase ``name``: ``[(t0, t1, from,
        to)]``, each move followed by a hold."""
        moves, v = [], start
        t = b.t
        for target in path:
            moves.append((t, t + MOVE, v, target))
            t += MOVE + HOLD
            v = target
        b.add(name, t - b.t, 0.4)
        return moves

    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        for k, name in ((4, "variance"), (3, "lengthscale"), (2, "samples"), (1, "matrix")):
            if t >= self.Tm[name][0]:
                return k
        return 0

    # ------------------------------------------------------------------ state
    @staticmethod
    def _value(t, moves, start):
        """The hyperparameter at ``t`` along its ``moves`` (geometric interpolation)."""
        v = start
        for t0, t1, a, c in moves:
            if t >= t0:
                v = log_lerp(a, c, ease_in_out(remap(t, t0, t1)))
        return v

    def kernel(self, l, var):
        """The RBF kernel with the hyperparameters on show."""
        return gp.RBF(variance=var, lengthscale=l)

    def _frame(self, t, reset):
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        intro = smoothstep(0.0, 1.0, self.progress("intro", t)) * back
        l = self._value(t, self.l_moves, L0)
        var = self._value(t, self.var_moves, VAR0)
        # the kernel bump: grows from x' = 5 in the first phase, then follows x'
        ku = self.progress("kernel", t)
        rows = int(np.searchsorted(self.row_t[1:], t, side="right"))
        rows = min(rows, N_GRID)
        in_matrix = self.Tm["matrix"][0] <= t < self.Tm["samples"][0] - 0.3
        row_u = remap(t, self.row_t[rows], self.row_t[rows + 1]) if rows < N_GRID else 1.0
        if in_matrix and rows < N_GRID:
            x_mark = float(self.grid[rows]) if rows == 0 else \
                float(self.grid[rows - 1] + (self.grid[rows] - self.grid[rows - 1]) * row_u)
        else:
            x_mark = X_MARK
        bump = smoothstep(0.0, 0.3, ku) * (1.0 - smoothstep(-0.6, 0.0, t - self.Tm["samples"][0]
                                                             + 0.6)) * back
        grow = ease_in_out(ku)
        samples = [smoothstep(0.0, 1.0, remap(t, t0, t0 + DRAW_S)) for t0 in self.sample_t]
        band = smoothstep(0.0, 1.0, self.progress("band", t)) * back
        return Frame(back, intro, l, var, bump, grow, x_mark, t >= self.Tm["matrix"][0], rows,
                     row_u, samples, band)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        f = self._frame(t, reset)
        k = self.kernel(f.l, f.var)
        self.axes(sc, f.intro, yticks=YTICKS)
        self._bump(sc, f, k)
        self._matrix(sc, f, k)
        self._samples(sc, f, k)
        self._readouts(sc, f)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, f, reset)

    def _bump(self, sc, f, k):
        """k(x, x') as a function of x, for the x' in play."""
        if f.bump <= 0.01:
            return
        half = (XR[1] - XR[0]) * f.grow
        xs = self.xs[np.abs(self.xs - f.x_mark) <= half + 1e-9]
        if len(xs) < 2:
            xs = np.array([f.x_mark - 0.01, f.x_mark + 0.01])
        ks = k(xs, np.array([f.x_mark]))[:, 0]
        self.curve(sc, xs, ks, FOCUS, width=2.2, op=0.95 * f.bump)
        self.vline(sc, f.x_mark, 0.5 * f.bump, FOCUS, width=1.0)
        self.label(sc, self.pt(f.x_mark, f.var), "k(x, x′)", layout.MIN_TEXT, anchor="lb",
                   dx=8, dy=-4, opacity=f.bump, ink=FOCUS)
        right = f.x_mark < 0.8 * XR[1]                  # the label keeps clear of the corner
        self.label(sc, self.pt(f.x_mark, YR[0]), "x′ = " + num(f.x_mark, 1), layout.MIN_TEXT,
                   anchor="lb" if right else "rb", dx=6 if right else -6, dy=-5, opacity=f.bump,
                   ink=FOCUS)

    def _matrix(self, sc, f, k):
        """The kernel matrix over the grid, filled in row by row, then live."""
        K = k(self.grid, self.grid)
        n = N_GRID
        rows = np.arange(n)[:, None]
        cols = np.arange(n)[None, :]
        if f.rows < n and f.band <= 0:
            done = rows < f.rows
            # the current row fills left to right as x' passes
            cur = (rows == f.rows) & (cols <= int(np.floor(f.row_u * n)))
            mask = (done | cur) if f.started else np.zeros((n, n), bool)
            glow = np.broadcast_to(np.where(rows == f.rows, 0.25, 0.0), (n, n))
        else:
            mask = np.ones((n, n), bool)
            glow = np.zeros((n, n))
        self.heat(self.K, K, max(VAR_PATH), mask & (f.back > 0.01), glow.ravel())
        self.K.fade(f.back)
        self.K.visible = bool(mask.any())
        if self.K.visible:
            self.K.cage = 0.6 * f.back
            self.sheet_label(sc, self.K, "K = k(x%s, x%s), %d grid points"
                             % ("ᵢ", "ⱼ", n), f.back, above=True)

    def _samples(self, sc, f, k):
        """Sample paths, the zero mean and the +-2 sd band."""
        if not any(u > 0 for u in f.samples) and f.band <= 0:
            return
        pri = gp.prior(k, self.xs)
        paths = pri.samples(self.z)
        sd = pri.sd
        self.band(sc, self.xs, pri.mu, sd, f.band)
        if f.band > 0:
            self.curve(sc, self.xs, pri.mu, MEAN, width=2.2, op=0.9 * f.band)
            # labels outside the panel, where no sample path crosses them
            self.label(sc, self.pt(XR[1], 2 * sd[-1]), "+2σ", layout.MIN_TEXT, anchor="lm",
                       dx=8, opacity=f.band, ink=MEAN_INK)
            self.label(sc, self.pt(XR[1], 0.0), "mean 0", layout.MIN_TEXT, anchor="lm", dx=8,
                       opacity=f.band, ink=MEAN_INK)
        for j, (u, path) in enumerate(zip(f.samples, paths)):
            if u > 0:
                self.curve(sc, self.xs, path, SAMPLES[j], width=1.8, op=0.9 * f.back,
                           upto=u if u < 1 else None)

    def _readouts(self, sc, f):
        """The hyperparameters under the matrix."""
        op = f.intro
        text = "lengthscale ℓ = %s    variance σ² = %s" % (num(f.l), num(f.var))
        self.sheet_label(sc, self.K, text, op, size=layout.CAPTION)

    # ------------------------------------------------------------------ words
    def _words(self, t, f, reset):
        """The subtitle: what is happening now, with the numbers."""
        if reset > 0 or t < self.Tm["kernel"][0] + 0.2:
            return ("a prior over functions: any set of values f(x) is jointly Gaussian, "
                    "with covariance k(x, x′)")
        if t < self.Tm["matrix"][0]:
            return ("RBF kernel k(x, x′) = σ² exp(−(x − x′)² / 2ℓ²), here for x′ = 5: "
                    "nearby x are strongly tied, far ones not at all")
        if t < self.Tm["samples"][0]:
            return ("the kernel matrix K: entry (i, j) = k(xᵢ, xⱼ) over %d grid points, "
                    "x′ = %s is row %d" % (N_GRID, num(f.x_mark, 1), min(f.rows + 1, N_GRID)))
        if t < self.Tm["band"][0]:
            n = sum(u > 0 for u in f.samples)
            return ("sample functions f = L z, with L the Cholesky factor of K and z standard "
                    "normal draws: %d of %d" % (n, N_SAMPLES))
        if t < self.Tm["lengthscale"][0]:
            return ("the mean is 0 everywhere; 95%% of samples lie within ±2σ = ±%s"
                    % num(2 * np.sqrt(f.var), 1))
        if t < self.Tm["variance"][0]:
            how = "wiggly" if f.l < 0.6 else ("smooth" if f.l > 1.8 else "moderate")
            return ("lengthscale ℓ = %s: how far correlation reaches. Short ℓ, %s samples; "
                    "the diagonal band narrows" % (num(f.l), how)
                    if f.l < 0.9 else
                    "lengthscale ℓ = %s: long ℓ, %s samples; the diagonal band widens"
                    % (num(f.l), how))
        return ("variance σ² = %s: the scale of the functions. The band is ±2σ = ±%s; "
                "the matrix brightens with σ²" % (num(f.var), num(2 * np.sqrt(f.var), 1)))

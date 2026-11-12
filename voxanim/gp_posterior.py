"""Conditioning a Gaussian process on data: the posterior, point by point.

    plot: the true function f (dashed),      the joint covariance of the data and
    the observations, the posterior mean,    the grid, as one sheet of voxels:
    the +-2 sd band, three sample paths          [ K    k*  ]
                                                 [ k*^T k** ]
                                             and the formula under it

1. **prior** -- before any data: mean 0, the band +-2, sample paths,
2. **observe** -- six noise-free observations of f(x) = sin x + 0.3 x - 1.5
   arrive one at a time.  After each, the posterior mean and band update:
   the band collapses to nothing at the data, the mean passes through it,
   the sample paths (the same seeded draws throughout) are pinned down.
   The matrix grows a row and a column per observation.
   Every frame is an exact posterior: a new point enters with a huge
   observation variance (no influence) that shrinks to its real value, so
   the band morphs honestly instead of being cross-faded,
3. **formula** -- mu* = k*^T K^-1 y and sd*^2 = k** - k*^T K^-1 k*: the
   pieces light up in turn, each block of the matrix with its symbol, the
   mean with mu*, the band with sd*,
4. **noise** -- the same data with observation noise sd 0.3 (the points move
   to their noisy values, the model is told the noise): the band no longer
   touches the points and the mean no longer passes through them.

Every number is computed by :func:`voxanim.gp.posterior`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import gp, layout
from .easing import ease_in_out, remap, smoothstep
from .gp_anim import FOCUS, MEAN, MEAN_INK, SAMPLES, TRUTH, Beats, GPDiagram, log_lerp
from .netanim import screen_box
from .text import num

XR, YR = (0.0, 10.0), (-3.0, 3.0)
YTICKS = (-2.0, 0.0, 2.0)
N_FINE, N_GRID = 121, 24
X_OBS = (4.1, 7.3, 1.2, 8.6, 2.5, 5.6)        # the observations, in order of arrival
LENGTHSCALE, VARIANCE = 1.5, 1.0             # the RBF kernel (fixed in this animation)
NOISE_FREE = 1e-6                            # observation variance of an exact observation
NOISE_SD = 0.3                               # the noise in the last phase
NOISE_FAR = 1e3                              # the variance a point enters with (no influence)
N_SAMPLES, SEED = 3, 11
SHEET_AT, SHEET_SCALE = (0.805, 0.56), 0.56  # the joint matrix: layout position, voxel size
POP_S, MORPH_S, SETTLE_S = 0.4, 1.3, 0.8     # per observation: appear, condition, hold
LIGHT_S = 1.4                                # each piece of the formula lights for this long
NOISE_MOVE_S = 2.4                           # the noise rising to its value
FORMULA = (                                  # the two lines: (text, tag) pieces
    (("μ* = ", None), ("k*ᵀ", "ks"), (" ", None), ("K⁻¹", "K"), (" y", "y")),
    (("σ*² = ", None), ("k**", "kss"), (" − ", None), ("k*ᵀ", "ks"), (" ", None), ("K⁻¹", "K"),
     (" ", None), ("k*", "ks")),
)
LIGHT_ORDER = ("K", "ks", "mu", "kss", "sd")  # what lights up, in order
FORMULA_DROP = 0.075                         # frame fraction from the matrix's foot to line 1
UNDERLINE_DROP = 0.72                        # the underline, in text sizes below the baseline


def truth(x):
    """The true function the observations come from."""
    x = np.asarray(x, np.float64)
    return np.sin(x) + 0.3 * x - 1.5


@dataclass
class Frame:
    """What one frame shows (see :meth:`PosteriorAnimation._frame`)."""

    back: float                  # 1, fading to 0 in the loop reset
    intro: float                 # axes and truth fading in
    noise: np.ndarray            # observation variance per point (NOISE_FAR: not yet in)
    seen: np.ndarray             # per point: how visible it is (0..1)
    y: np.ndarray                # the observed values on show (exact, sliding to noisy)
    current: int                 # the observation in play (-1: none)
    light: str | None            # the piece of the formula lit up
    light_u: float               # progress of that lighting
    noisy: float                 # progress of the noise phase
    formula: float               # the formula lines' visibility


class PosteriorAnimation(GPDiagram):
    """Six observations, one at a time, then the formula, then noise."""

    PHASES = ("prior", "observe", "formula", "noise")

    def __init__(self):
        super().__init__("Conditioning on data", XR, YR)
        self.xs = np.linspace(*XR, N_FINE)
        self.grid = np.linspace(*XR, N_GRID)
        self.X = np.array(X_OBS, np.float64)
        self.order = np.argsort(self.X)              # sorted position of each observation
        self.y_exact = truth(self.X)
        rng = np.random.default_rng(SEED)
        self.z = rng.standard_normal((N_SAMPLES, N_FINE))
        self.eps = rng.normal(0.0, NOISE_SD, len(self.X))
        self.k = gp.RBF(variance=VARIANCE, lengthscale=LENGTHSCALE)
        n = len(self.X) + N_GRID
        self.J = self.sheet((n, n), SHEET_AT, SHEET_SCALE, "joint")
        self._timeline()
        self.hold, self.reset = 2.5, 1.2
        self.finish(self.b.t)

    def _timeline(self):
        b = Beats()
        b.add("intro", 1.0, 0.0)
        b.add("prior", 3.0, 0.3)
        per = POP_S + MORPH_S + SETTLE_S
        self.obs_t = [b.t + k * per for k in range(len(self.X))]
        b.add("observe", per * len(self.X), 0.4)
        self.light_t = [b.t + k * LIGHT_S for k in range(len(LIGHT_ORDER))]
        b.add("formula", LIGHT_S * len(LIGHT_ORDER) + 0.6, 0.4)
        b.add("noise", NOISE_MOVE_S + 3.2, 0.0)
        self.b, self.Tm = b, b.Tm

    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        for k, name in ((3, "noise"), (2, "formula"), (1, "observe")):
            if t >= self.Tm[name][0]:
                return k
        return 0

    # ------------------------------------------------------------------ state
    def _frame(self, t, reset):
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        intro = smoothstep(0.0, 1.0, self.progress("intro", t)) * back
        noisy = ease_in_out(remap(t, self.Tm["noise"][0], self.Tm["noise"][0] + NOISE_MOVE_S))
        target = log_lerp(NOISE_FREE, NOISE_SD ** 2, noisy)
        noise = np.full(len(self.X), NOISE_FAR)
        seen = np.zeros(len(self.X))
        current = -1
        for i, t0 in enumerate(self.obs_t):
            if t < t0:
                break
            current = i
            seen[i] = smoothstep(0.0, POP_S, t - t0)
            u = ease_in_out(remap(t, t0 + POP_S, t0 + POP_S + MORPH_S))
            noise[i] = log_lerp(NOISE_FAR, target, u)
        if t >= self.Tm["observe"][1]:
            current = -1
        y = self.y_exact + self.eps * noisy
        light, light_u = None, 0.0
        for name, t0 in zip(LIGHT_ORDER, self.light_t):
            if t0 <= t < t0 + LIGHT_S:
                light, light_u = name, (t - t0) / LIGHT_S
        formula = smoothstep(0.0, 0.6, t - self.Tm["formula"][0]) * intro
        return Frame(back, intro, noise, seen * back, y, current, light, light_u, noisy, formula)

    def posterior(self, f, xs):
        """The exact posterior of the points on show, on the grid ``xs``."""
        on = f.noise < NOISE_FAR
        return gp.posterior(self.k, self.X[on], f.y[on], xs, f.noise[on])

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        f = self._frame(t, reset)
        post = self.posterior(f, self.xs)
        self.axes(sc, f.intro, yticks=YTICKS)
        self.curve(sc, self.xs, truth(self.xs), TRUTH, width=1.6, op=0.9 * f.intro, dashed=True)
        self._posterior(sc, f, post)
        self._observations(sc, f)
        self._matrix(sc, f)
        self._formula(sc, f)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, f, post, reset)

    def _posterior(self, sc, f, post):
        """Band, sample paths and mean; the mean and band pulse when the
        formula names them."""
        sd = post.sd
        mu_lit = smoothstep(0.0, 0.3, f.light_u) * (1 - smoothstep(0.7, 1.0, f.light_u)) \
            if f.light == "mu" else 0.0
        sd_lit = smoothstep(0.0, 0.3, f.light_u) * (1 - smoothstep(0.7, 1.0, f.light_u)) \
            if f.light == "sd" else 0.0
        self.band(sc, self.xs, post.mu, sd, f.intro)
        for j, path in enumerate(post.samples(self.z)):
            self.curve(sc, self.xs, path, SAMPLES[j], width=1.5, op=0.75 * f.intro)
        self.curve(sc, self.xs, post.mu, MEAN, width=2.2 + 1.4 * mu_lit, op=0.95 * f.intro)
        if sd_lit > 0:
            for sign in (-1, 1):
                self.curve(sc, self.xs, post.mu + sign * 2 * sd, MEAN, width=1.6,
                           op=0.9 * sd_lit * f.intro)
        self.label(sc, self.pt(XR[1], post.mu[-1]), "μ*", layout.CAPTION, anchor="lm", dx=8,
                   opacity=f.intro * (0.6 + 0.4 * mu_lit), ink=MEAN_INK, kind="mono")
        self.label(sc, self.pt(XR[1], post.mu[-1] + 2 * sd[-1]), "+2σ*", layout.MIN_TEXT,
                   anchor="lm", dx=8, opacity=f.intro * (0.6 + 0.4 * sd_lit), ink=MEAN_INK,
                   kind="mono")

    def _observations(self, sc, f):
        """The data points; the one arriving is ringed."""
        self.points(sc, self.X, f.y, 1.0, each=f.seen)
        if f.current >= 0:
            i = f.current
            self.marker(sc, self.X[i], f.y[i], f.seen[i] * f.back)

    def _joint(self, f):
        """The joint covariance ``[[K, k*], [k*^T, k**]]`` over the sorted data and
        the coarse grid, and the mask of the rows/columns of points already in."""
        n, m = len(self.X), N_GRID
        Xs = self.X[self.order]
        on = f.noise[self.order] < NOISE_FAR
        J = np.zeros((n + m, n + m))
        J[:n, :n] = gp.gram(self.k, Xs, np.minimum(f.noise[self.order], 1.0))
        J[:n, n:] = self.k(Xs, self.grid)
        J[n:, :n] = J[:n, n:].T
        J[n:, n:] = self.k(self.grid, self.grid)
        keep = np.concatenate([on, np.ones(m, bool)])
        return J, keep[:, None] & keep[None, :]

    def _matrix(self, sc, f):
        """The joint covariance as one sheet; the block the formula names pops out."""
        J, mask = self._joint(f)
        n, m = len(self.X), N_GRID
        blocks = {"K": (slice(0, n), slice(0, n)), "ks": (slice(0, n), slice(n, None)),
                  "kss": (slice(n, None), slice(n, None))}
        lit = np.zeros_like(J)
        if f.light in blocks:
            amount = smoothstep(0.0, 0.25, f.light_u) * (1 - smoothstep(0.75, 1.0, f.light_u))
            lit[blocks[f.light]] = amount
            if f.light == "ks":
                lit[n:, :n] = amount                   # k*^T is the same numbers
        show = mask if f.intro > 0.01 else np.zeros_like(mask)
        self.heat(self.J, J, VARIANCE, show, 0.3 * lit.ravel())
        self.J.light(lit.ravel(), float(lit.max()))
        self.J.fade(f.back)
        self.J.cage = 0.6 * f.intro                 # (0 at the loop's seam: no cells either)
        self._block_labels(sc, f, n, m)

    def _block_labels(self, sc, f, n, m):
        """K, k*, k*^T and k** beside their blocks (mono: the superscript T); the
        data blocks' names arrive with the first observation."""
        op = f.intro
        data_op = op * float(f.seen.max())
        b = self.J.block
        s = SHEET_SCALE

        def over(c0, c1, text, tag, o, above=True):
            lo = b.cell_center(0 if above else n + m - 1, c0, 0)
            hi = b.cell_center(0 if above else n + m - 1, c1 - 1, 0)
            box = (np.minimum(lo, hi) - 0.5 * s, np.maximum(lo, hi) + 0.5 * s)
            self.caption(sc, box, text, o, size=layout.MIN_TEXT, above=above,
                         ink=self._tag_ink(f, tag))
        over(0, n, "K", "K", data_op)
        over(n, n + m, "k*", "ks", data_op)
        over(n, n + m, "k**", "kss", op, above=False)
        # k*^T left of the lower block, at its middle row
        x0 = screen_box(self.cam, *self.J.bbox())[0]
        y = self.cam.project(b.cell_center(n + m // 2, 0, 0))[1]
        sc.text((x0 - 8) / self.cam.width, y / self.cam.height, "k*ᵀ", layout.MIN_TEXT, "mono",
                self._tag_ink(f, "ks"), data_op * self._cap, "rm")

    @staticmethod
    def _tag_ink(f, tag):
        """Blue while the formula piece ``tag`` is lit, else the strong ink."""
        return FOCUS if f.light == tag else layout.STRONG_INK

    def _formula(self, sc, f):
        """The two formula lines under the matrix; the piece in play is underlined
        (one text per line: pieces drawn separately would collide)."""
        op = f.formula
        if op <= 0.01:
            return
        size = layout.CAPTION
        cx = self.fr(self.J.center)[0]
        top = screen_box(self.cam, *self.J.bbox())[3] / self.cam.height + FORMULA_DROP
        for row, pieces in enumerate(FORMULA):
            widths = [layout.text_width(s, size, "mono") for s, _tag in pieces]
            x0 = cx - sum(widths) / 2
            y = top + row * size * 1.5
            sc.text(x0, y, "".join(s for s, _tag in pieces), size, "mono", layout.STRONG_INK,
                    op * self._cap, "lm")
            x = x0
            for (_s, tag), w in zip(pieces, widths):
                lit = tag is not None and (tag == f.light or (tag == "y" and f.light == "mu"))
                if lit:
                    self._underline(sc, x, y + size * UNDERLINE_DROP, w, op)
                x += w

    def _underline(self, sc, x, y, w, op):
        """A short blue bar under a piece of the formula (frame fractions)."""
        a = self.cam.unproject(x * self.cam.width, y * self.cam.height) + self.view * 2.0
        b = self.cam.unproject((x + w) * self.cam.width, y * self.cam.height) + self.view * 2.0
        sc.line(a, b, width=2.2, color=FOCUS, opacity=op * self._cap, bias=0.0)

    # ------------------------------------------------------------------ words
    def _words(self, t, f, post, reset):
        """The subtitle: what is happening now, with the numbers."""
        if reset > 0 or t < self.Tm["observe"][0]:
            return ("the prior: RBF kernel, ℓ = %s, σ² = %s. Mean 0, band ±2σ. The dashed "
                    "curve is the truth f(x) = sin x + 0.3x − 1.5, unknown to the model"
                    % (num(LENGTHSCALE, 1), num(VARIANCE, 1)))
        if t < self.Tm["formula"][0]:
            i = f.current if f.current >= 0 else len(self.X) - 1
            n_in = int((f.seen > 0).sum())
            return ("observation %d at x = %s: y = f(x) = %s. The posterior mean passes through "
                    "it and the band closes there; %d point%s in"
                    % (i + 1, num(self.X[i], 1), num(f.y[i]), n_in, "" if n_in == 1 else "s"))
        if t < self.Tm["noise"][0]:
            return {
                None: "μ* and σ*² from K, k* and k**: every posterior on this page is computed "
                      "this way",
                "K": "K = k(X, X): the %d × %d covariance of the data; its inverse weights "
                     "the observations" % (len(self.X), len(self.X)),
                "ks": "k* = k(X, x*): how each data point is tied to every x on the grid",
                "mu": "the mean μ* = k*·(K⁻¹y): a weighted sum of the %d observations, weights "
                      "K⁻¹y" % len(self.X),
                "kss": "k** = k(x*, x*): the prior variance on the grid, before any data",
                "sd": "the variance σ*² = k** minus what the data explains: 0 at the data, "
                      "the prior's %s far from it" % num(VARIANCE, 1),
            }[f.light]
        return ("with observation noise σₙ = %s (points moved to noisy values, K gets σₙ² on "
                "its diagonal): the band stays open at the data, the mean need not pass through"
                % num(NOISE_SD, 1))


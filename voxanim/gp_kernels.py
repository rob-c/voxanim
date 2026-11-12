"""Kernels are assumptions: one small data set, five kernels, five posteriors.

    plot: 7 noisy observations of f(x) = sin x + 0.3 x - 1.5 on x in [0.8, 5.4],
    the posterior mean, the +-2 sd band and two sample paths on [0, 10]
    (so most of the plot is extrapolation: what each kernel assumes)
                                       right: the kernel's matrix over the grid,
                                       its hyperparameters, and a table of
                                       log p(y) for every kernel seen so far

One segment per kernel: **RBF**, **Matérn 3/2**, **periodic** (period 4
assumed), **linear** (through x = 0) and **periodic + RBF**.  Each kernel's
variance(s) and lengthscale(s) are fitted first by maximising the log
marginal likelihood (:func:`voxanim.gp.ascend`, 600 steps, cached); the
noise variance 0.01 is given.  log p(y) is the read-out: it ranks the
assumptions on this data (the sum first, the straight line last).

Between segments the posterior morphs through exact posteriors of the kernel
``(1 - u) k_old + u k_new`` (:func:`voxanim.gp.blend`), with the same seeded
draws for the sample paths, so nothing is cross-faded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import gp, layout
from .cache import cached_npz
from .easing import ease_in_out, remap, smoothstep
from .gp_anim import FOCUS, MEAN, MEAN_INK, SAMPLES, Beats, GPDiagram
from .text import num

XR, YR = (0.0, 10.0), (-3.0, 3.0)
YTICKS = (-2.0, 0.0, 2.0)
N_FINE, N_GRID = 121, 24
X_OBS = (0.8, 1.6, 2.3, 3.1, 3.9, 4.6, 5.4)
NOISE_SD, DATA_SEED = 0.1, 5                 # the observations' noise (given to the model)
PERIOD, OFFSET = 4.0, 0.0                    # the assumptions that are not fitted
FIT_LR, FIT_STEPS = 0.01, 600                # gradient ascent on log p(y), in log space
N_SAMPLES, SEED = 2, 21
SHEET_AT, SHEET_SCALE = (0.805, 0.45), 0.50  # the kernel matrix: layout position, voxel size
TABLE_AT, ROW = (0.685, 0.70), 0.046         # the log p(y) table: top-left, row spacing
TABLE_RIGHT = 0.935
SEGMENT, MORPH = 7.4, 1.0                    # seconds per kernel; the morph at its start
COMPARE_S = 3.5                              # the closing comparison

# (phase name, display name, starting kernel, the hyperparameters that are fitted)
KERNELS = (
    ("RBF", "RBF", lambda: gp.RBF(variance=1.0, lengthscale=1.0), ("variance", "lengthscale")),
    ("Matérn", "Matérn 3/2", lambda: gp.Matern32(variance=1.0, lengthscale=1.0),
     ("variance", "lengthscale")),
    ("periodic", "periodic", lambda: gp.Periodic(variance=1.0, lengthscale=1.0, period=PERIOD),
     ("variance", "lengthscale")),
    ("linear", "linear", lambda: gp.Linear(variance=0.1, offset=OFFSET), ("variance",)),
    ("sum", "periodic + RBF",
     lambda: gp.Periodic(variance=1.0, lengthscale=1.0, period=PERIOD)
     + gp.RBF(variance=1.0, lengthscale=1.0),
     ("1.variance", "1.lengthscale", "2.variance", "2.lengthscale")),
)


def truth(x):
    """The function the observations come from."""
    x = np.asarray(x, np.float64)
    return np.sin(x) + 0.3 * x - 1.5


def dataset(seed=DATA_SEED):
    """``(X, y)``: the observations, ``truth(X)`` plus seeded noise of sd NOISE_SD."""
    X = np.array(X_OBS, np.float64)
    return X, truth(X) + np.random.default_rng(seed).normal(0.0, NOISE_SD, len(X))


def fit_all():
    """Every kernel of :data:`KERNELS` fitted to the data: ``[(kernel, Evidence)]``
    in that order.  The fitted hyperparameters are cached."""
    X, y = dataset()
    noise = NOISE_SD ** 2

    def build():
        out = {}
        for i, (_name, _label, make, free) in enumerate(KERNELS):
            ks, _ns, _evs = gp.ascend(make(), X, y, noise, free, FIT_LR, FIT_STEPS)
            out["k%d" % i] = np.array(list(ks[-1].params.values()))
        return out
    settings = dict(x=X_OBS, noise=NOISE_SD, seed=DATA_SEED, period=PERIOD, offset=OFFSET,
                    lr=FIT_LR, steps=FIT_STEPS,
                    kernels=[(n, list(f)) for n, _l, _m, f in KERNELS])
    fitted = cached_npz("gp_kernels", settings, build)
    out = []
    for i, (_name, _label, make, _free) in enumerate(KERNELS):
        k0 = make()
        k = k0.set_params(dict(zip(k0.params, fitted["k%d" % i].tolist())))
        out.append((k, gp.log_marginal(k, X, y, noise)))
    return out


@dataclass
class Frame:
    """What one frame shows (see :meth:`KernelsAnimation._frame`)."""

    back: float                  # 1, fading to 0 in the loop reset
    intro: float                 # axes and data fading in
    k: int                       # the kernel segment on show (-1: none yet)
    u: float                     # progress of the morph into it (1: settled)
    compare: float               # the closing comparison's progress


class KernelsAnimation(GPDiagram):
    """Five kernels on the same data, one after another, with log p(y) of each."""

    PHASES = tuple(k[0] for k in KERNELS)

    def __init__(self):
        super().__init__("Kernels are assumptions", XR, YR)
        self.xs = np.linspace(*XR, N_FINE)
        self.grid = np.linspace(*XR, N_GRID)
        self.X, self.y = dataset()
        self.noise = NOISE_SD ** 2
        self.fits = fit_all()
        self.z = np.random.default_rng(SEED).standard_normal((N_SAMPLES, N_FINE))
        self.K = self.sheet((N_GRID, N_GRID), SHEET_AT, SHEET_SCALE, "K")
        self._timeline()
        self.hold, self.reset = 2.5, 1.2
        self.finish(self.b.t)

    def _timeline(self):
        b = Beats()
        b.add("intro", 1.0, 0.3)
        self.seg_t = []
        for name, *_ in KERNELS:
            self.seg_t.append(b.t)
            b.add(name, SEGMENT, 0.0)
        b.add("compare", COMPARE_S, 0.0)
        self.b, self.Tm = b, b.Tm

    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        return max(0, int(np.searchsorted(self.seg_t, t, side="right")) - 1)

    # ------------------------------------------------------------------ state
    def _frame(self, t, reset):
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        intro = smoothstep(0.0, 1.0, self.progress("intro", t)) * back
        k = int(np.searchsorted(self.seg_t, t, side="right")) - 1
        u = ease_in_out(remap(t, self.seg_t[k], self.seg_t[k] + MORPH)) if k >= 0 else 0.0
        return Frame(back, intro, k, u, smoothstep(0.0, 0.6, self.progress("compare", t)))

    def kernel_at(self, f):
        """The kernel on show: the segment's, or the morph from the previous one."""
        new = self.fits[f.k][0]
        if f.k == 0 or f.u >= 1:
            return new
        return gp.blend(self.fits[f.k - 1][0], new, f.u)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        f = self._frame(t, reset)
        self.axes(sc, f.intro, yticks=YTICKS)
        self.points(sc, self.X, self.y, f.intro)
        if f.k >= 0:
            k = self.kernel_at(f)
            post = gp.posterior(k, self.X, self.y, self.xs, self.noise)
            op = f.back * (f.u if f.k == 0 else 1.0)
            self._posterior(sc, post, op)
            self._matrix(sc, k, self.fits[f.k][0], op)
        else:
            self.K.visible = False
        self._table(sc, f)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, f, reset)

    def _posterior(self, sc, post, op):
        sd = post.sd
        self.band(sc, self.xs, post.mu, sd, op)
        for j, path in enumerate(post.samples(self.z)):
            self.curve(sc, self.xs, path, SAMPLES[j], width=1.5, op=0.75 * op)
        self.curve(sc, self.xs, post.mu, MEAN, width=2.2, op=0.95 * op)
        self.label(sc, self.pt(XR[1], post.mu[-1]), "μ*", layout.CAPTION, anchor="lm", dx=8,
                   opacity=0.8 * op, ink=MEAN_INK, kind="mono")
        edge = (XR[1] - XR[0]) * 0.1
        self.label(sc, self.pt(self.X.max() + 0.5 * edge, YR[1]), "extrapolation →",
                   layout.MIN_TEXT, anchor="lt", dx=4, dy=6, opacity=0.9 * op,
                   ink=layout.NOTE_INK)

    def _matrix(self, sc, k, target, op):
        """The kernel matrix over the grid (of ``k``, which may be a morph) and
        the hyperparameters of the segment's kernel ``target`` under it."""
        K = k(self.grid, self.grid)
        vmax = max(float(np.abs(K).max()), 1e-9)
        self.heat(self.K, K, vmax)
        self.K.fade(op)
        self.K.cage = 0.6 * op
        self.sheet_label(sc, self.K, "k(xᵢ, xⱼ) on the grid", op, above=True)
        self.sheet_label(sc, self.K, self._params_text(target), op, size=layout.MIN_TEXT)

    @classmethod
    def _params_text(cls, k):
        """The hyperparameters of ``k``, one line (a sum: one line per part)."""
        if isinstance(k, gp.Sum):
            return "\n".join("%s: %s" % (part.NAME, cls._params_text(part))
                             for part in (k.k1, k.k2))
        short = {"variance": "σ²", "lengthscale": "ℓ", "period": "p", "offset": "c"}
        return "  ".join("%s = %s" % (short[name], num(v, 2)) for name, v in k.params.items())

    def _table(self, sc, f):
        """log p(y) of every kernel reached so far; the one on show in blue, the
        best marked at the end."""
        op = f.intro
        x0, y0 = TABLE_AT
        self.label(sc, self.P(x0, y0), "log p(y), hyperparameters fitted", layout.MIN_TEXT,
                   anchor="lm", opacity=op, ink=layout.NOTE_INK)
        best = int(np.argmax([ev.value for _k, ev in self.fits]))
        for i, (_name, label, _make, _free) in enumerate(KERNELS):
            shown = op * (1.0 if f.k > i else (f.u if f.k == i else 0.0))
            if shown <= 0.01:
                continue
            y = y0 + ROW * (i + 1)
            ink = FOCUS if (f.k == i and f.compare <= 0) or (f.compare > 0 and i == best) \
                else layout.CAPTION_INK
            if f.compare > 0 and i == best:                  # a frame around the winner
                c = self.P((x0 + TABLE_RIGHT) / 2, y)
                w = (TABLE_RIGHT - x0) * self.cam.width * self.cam.zoom + 16
                self.rect(sc, c, w, ROW * self.cam.height * self.cam.zoom - 2, (1, 1, 1),
                          shown * f.compare, border=FOCUS, border_w=1.4, radius=5, lift=0.2,
                          fill_op=0.0)
            self.label(sc, self.P(x0, y), label, layout.MIN_TEXT, anchor="lm", opacity=shown,
                       ink=ink)
            self.label(sc, self.P(TABLE_RIGHT, y), num(self.fits[i][1].value, 2),
                       layout.MIN_TEXT, anchor="rm", opacity=shown, ink=ink, kind="mono")

    # ------------------------------------------------------------------ words
    def _words(self, t, f, reset):
        """The subtitle: the kernel on show, its fitted numbers, what it assumes."""
        if reset > 0 or f.k < 0:
            return ("%d noisy observations (σₙ = %s) of f(x) = sin x + 0.3x − 1.5 on [%s, %s]. "
                    "The kernel decides what happens elsewhere"
                    % (len(self.X), num(NOISE_SD, 1), num(self.X.min(), 1),
                       num(self.X.max(), 1)))
        if f.compare > 0:
            order = sorted(range(len(KERNELS)), key=lambda i: -self.fits[i][1].value)
            return ("the marginal likelihood ranks the assumptions on this data: %s"
                    % ", ".join("%s %s" % (KERNELS[i][1], num(self.fits[i][1].value, 1))
                                for i in order))
        k, ev = self.fits[f.k]
        p = k.params
        name = KERNELS[f.k][0]
        if name == "RBF":
            return ("RBF, fitted ℓ = %s, σ² = %s: smooth paths; away from the data the mean "
                    "returns to 0 and the band to ±2σ. log p(y) = %s"
                    % (num(p["lengthscale"]), num(p["variance"]), num(ev.value)))
        if name == "Matérn":
            return ("Matérn 3/2, fitted ℓ = %s, σ² = %s: the same data, rougher paths (once "
                    "differentiable); the band opens sooner. log p(y) = %s"
                    % (num(p["lengthscale"]), num(p["variance"]), num(ev.value)))
        if name == "periodic":
            return ("periodic, period p = %s assumed, fitted ℓ = %s, σ² = %s: what it saw it "
                    "expects again, forever, as sure as at the data. log p(y) = %s"
                    % (num(PERIOD, 0), num(p["lengthscale"]), num(p["variance"]),
                       num(ev.value)))
        if name == "linear":
            return ("linear, slopes through x = %s, fitted σ² = %s: only straight lines; the "
                    "data cannot be fitted well, so log p(y) = %s"
                    % (num(OFFSET, 0), num(p["variance"]), num(ev.value)))
        return ("periodic + RBF: a repeating part (p = %s, ℓ = %s) plus a smooth drift "
                "(ℓ = %s): the best evidence here, log p(y) = %s"
                % (num(PERIOD, 0), num(p["1.lengthscale"]), num(p["2.lengthscale"]),
                   num(ev.value)))

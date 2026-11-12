"""Bayesian optimisation: find the maximum of an unknown function in few evaluations.

    top plot: the evaluations, the GP posterior (mean, +-2 sd band)   right: the
    bottom plot: the acquisition function, expected improvement,      evaluations
    with its maximum marked                                           and the best
                                                                      so far

The unknown function is g(x) = 1.4 exp(-(x - 2.2)^2 / 0.8) + 1.8 exp(-(x - 7.3)^2
/ 1.5) + 0.3 sin 2x on [0, 10]: two hills, the right one higher.  The model
is an RBF GP (variance 1, lengthscale 1), evaluations are exact.

1. **start** -- two evaluations, at x = 1.5 and 6.0, and the posterior,
2. **acquisition** -- expected improvement over the best value so far,
   EI(x) = E[max(f(x) - best, 0)], drawn below the plot; large where the
   mean is high (exploit) and where the band is wide (explore),
3. **rounds** -- eight times: the next evaluation is taken where EI is
   highest (a particle flies up to the function), the posterior updates
   (through exact posteriors: the new point's observation variance shrinks
   from huge to its real value), EI is recomputed.  Which rounds explore and
   which exploit is read off the picks: far from every evaluation (more than
   one lengthscale), or next to the best one,
4. **result** -- the true function is revealed with its maximum, next to the
   best evaluation found.

Every number comes from :func:`voxanim.gp.posterior` and
:func:`voxanim.gp.expected_improvement`; the rounds are cached.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import gp, layout
from .cache import cached_npz
from .easing import ease_in_out, ease_out, remap, smoothstep
from .gp_anim import ACQ, ACQ_INK, MEAN, MEAN_INK, OBS, TRUTH, Beats, GPDiagram, log_lerp
from .text import num

XR, YR = (0.0, 10.0), (-1.5, 3.0)
YTICKS = (-1.0, 0.0, 1.0, 2.0)
PLOT = (0.05, 0.30, 0.63, 0.70)                  # the function and the posterior
ACQ_PLOT = (0.05, 0.775, 0.63, 0.93)             # expected improvement under it
N_FINE = 241
X_START = (1.5, 6.0)                             # the two evaluations we begin with
LENGTHSCALE, VARIANCE = 1.0, 1.0                 # the RBF kernel (fixed)
NOISE = 1e-4                                     # evaluations are exact (a numerical floor)
NOISE_FAR = 1e3                                  # a point enters with this variance (no effect)
XI = 0.01                                        # EI's improvement margin
ROUNDS = 8
TABLE_AT, ROW, TABLE_RIGHT = (0.69, 0.33), 0.041, 0.955  # the evaluations table (layout)
ACQ_S, FLY_S, MORPH_S, SETTLE_S = 1.2, 0.9, 1.1, 0.7     # per round: EI, the flight, update, hold


def unknown(x):
    """The function being optimised (the model never sees this formula)."""
    x = np.asarray(x, np.float64)
    return (1.4 * np.exp(-((x - 2.2) ** 2) / 0.8) + 1.8 * np.exp(-((x - 7.3) ** 2) / 1.5)
            + 0.3 * np.sin(2 * x))


def kernel():
    """The GP's kernel."""
    return gp.RBF(variance=VARIANCE, lengthscale=LENGTHSCALE)


def run():
    """The optimisation: ``X`` and ``y`` of every evaluation (the two starting
    ones first), each round's pick being the grid point of highest EI.  Cached."""
    xs = np.linspace(*XR, N_FINE)

    def build():
        X, y = list(X_START), [float(unknown(x)) for x in X_START]
        for _ in range(ROUNDS):
            post = gp.posterior(kernel(), np.array(X), np.array(y), xs, NOISE)
            ei = gp.expected_improvement(post.mu, post.sd, max(y), XI)
            x_next = float(xs[int(np.argmax(ei))])
            X.append(x_next)
            y.append(float(unknown(x_next)))
        return dict(X=np.array(X), y=np.array(y))
    settings = dict(start=X_START, l=LENGTHSCALE, v=VARIANCE, noise=NOISE, xi=XI, rounds=ROUNDS,
                    grid=N_FINE, xr=XR, f="two bumps + 0.3 sin 2x")
    d = cached_npz("gp_bayesopt", settings, build)
    return d["X"], d["y"]


def explores(X, i, l=LENGTHSCALE):
    """True when evaluation ``i`` lies more than a lengthscale from every
    earlier one (it went where the model was unsure)."""
    return bool(np.min(np.abs(X[:i] - X[i])) > l)


@dataclass
class Frame:
    """What one frame shows (see :meth:`BayesOptAnimation._frame`)."""

    back: float                  # 1, fading to 0 in the loop reset
    intro: float                 # axes and the first evaluations fading in
    acq: float                   # the acquisition panel's visibility
    rnd: int                     # the round under way (0: none yet; 1..ROUNDS)
    stage: str                   # "acq" (EI drawn), "fly", "morph", "settle"
    u: float                     # progress through that stage
    noise: np.ndarray            # observation variance per evaluation (NOISE_FAR: not yet in)
    reveal: float                # the true function's visibility
    post: float                  # the posterior's visibility
    acq_draw: float              # how much of the first EI curve is drawn (0..1)


class BayesOptAnimation(GPDiagram):
    """Eight rounds of expected-improvement search on a hidden function."""

    PHASES = ("start", "acquisition", "rounds", "result")

    def __init__(self):
        super().__init__("Bayesian optimisation", XR, YR, plot=PLOT)
        self.xs = np.linspace(*XR, N_FINE)
        self.X, self.y = run()
        self.n0 = len(X_START)
        self.acq_plot = self.make_plot(ACQ_PLOT, XR, (0.0, 1.0))      # EI scaled to its max
        self.ei_max = self._ei_max()
        self.true_x = float(self.xs[int(np.argmax(unknown(self.xs)))])
        self.true_max = float(unknown(self.true_x))
        self._timeline()
        self.hold, self.reset = 2.5, 1.2
        self.finish(self.b.t)

    def _ei_max(self):
        """The largest EI over all rounds (the acquisition panel's scale)."""
        top = 0.0
        for r in range(ROUNDS):
            n = self.n0 + r
            post = gp.posterior(kernel(), self.X[:n], self.y[:n], self.xs, NOISE)
            top = max(top, float(gp.expected_improvement(post.mu, post.sd, self.y[:n].max(),
                                                         XI).max()))
        return top

    def _timeline(self):
        b = Beats()
        b.add("intro", 1.2, 0.3)
        b.add("start", 2.6, 0.2)
        b.add("acquisition", 3.0, 0.2)
        per = ACQ_S + FLY_S + MORPH_S + SETTLE_S
        self.round_t = [b.t + r * per for r in range(ROUNDS)]
        b.add("rounds", per * ROUNDS, 0.3)
        b.add("result", 4.5, 0.0)
        self.b, self.Tm = b, b.Tm

    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        for k, name in ((3, "result"), (2, "rounds"), (1, "acquisition")):
            if t >= self.Tm[name][0]:
                return k
        return 0

    # ------------------------------------------------------------------ state
    def _frame(self, t, reset):
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        intro = smoothstep(0.0, 1.0, self.progress("intro", t)) * back
        acq = smoothstep(0.0, 1.0, self.progress("acquisition", t)) * back
        noise = np.full(len(self.X), NOISE_FAR)
        noise[:self.n0] = NOISE
        rnd, stage, u = 0, "settle", 1.0
        for r, t0 in enumerate(self.round_t):
            if t < t0:
                break
            rnd = r + 1
            i = self.n0 + r
            s = t - t0
            if s < ACQ_S:
                stage, u = "acq", s / ACQ_S
            elif s < ACQ_S + FLY_S:
                stage, u = "fly", (s - ACQ_S) / FLY_S
            elif s < ACQ_S + FLY_S + MORPH_S:
                stage, u = "morph", (s - ACQ_S - FLY_S) / MORPH_S
                noise[i] = log_lerp(NOISE_FAR, NOISE, ease_in_out(u))
            else:
                stage, u = "settle", remap(s, ACQ_S + FLY_S + MORPH_S, ACQ_S + FLY_S + MORPH_S
                                           + SETTLE_S)
                noise[i] = NOISE
        reveal = smoothstep(0.0, 0.35, self.progress("result", t)) * back
        post = intro * smoothstep(0.0, 0.6, self.progress("start", t))
        acq_draw = ease_out(self.progress("acquisition", t) * 1.6)
        return Frame(back, intro, acq, rnd, stage, u, noise, reveal, post, acq_draw)

    def posterior(self, f):
        """The exact posterior of the evaluations on show."""
        on = f.noise < NOISE_FAR
        return gp.posterior(kernel(), self.X[on], self.y[on], self.xs, f.noise[on])

    def best(self, f):
        """Index of the best evaluation among those in (or landing)."""
        n = self.n0 + f.rnd - (1 if f.rnd > 0 and f.stage in ("acq", "fly") else 0)
        return int(np.argmax(self.y[:n]))

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        f = self._frame(t, reset)
        post = self.posterior(f)
        known = self.n0 + f.rnd - (1 if f.rnd > 0 and f.stage in ("acq", "fly") else 0)
        ei = gp.expected_improvement(post.mu, post.sd, self.y[:known].max(), XI)
        self.axes(sc, f.intro, ylabel="g(x)", yticks=YTICKS)
        self.axes(sc, f.acq, ylabel="", xlabel="", yticks=(), zero=False, plot=self.acq_plot)
        self._truth(sc, f)
        self._posterior(sc, f, post)
        self._acquisition(sc, f, ei)
        self._evaluations(sc, f)
        self._table(sc, f)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, f, reset)

    def _truth(self, sc, f):
        if f.reveal <= 0.01:
            return
        self.curve(sc, self.xs, unknown(self.xs), TRUTH, width=1.8, op=0.95 * f.reveal,
                   dashed=True)
        self.marker(sc, self.true_x, self.true_max, f.reveal, color=TRUTH, r=7.0)
        self.label(sc, self.pt(self.true_x, self.true_max), "true maximum", layout.MIN_TEXT,
                   anchor="mb", dy=-12, opacity=f.reveal, ink=TRUTH)

    def _posterior(self, sc, f, post):
        op = f.post
        self.band(sc, self.xs, post.mu, post.sd, op)
        self.curve(sc, self.xs, post.mu, MEAN, width=2.2, op=0.95 * op)
        self.label(sc, self.pt(XR[1], post.mu[-1]), "μ*", layout.CAPTION, anchor="lm", dx=8,
                   opacity=0.8 * op, ink=MEAN_INK, kind="mono")

    def _acquisition(self, sc, f, ei):
        """EI under the plot; its maximum marked and, in a round, the flight."""
        if f.acq <= 0.01:
            return
        y = ei / self.ei_max
        upto = None
        if f.rnd > 0 and f.stage == "acq":
            upto = ease_out(f.u)
        elif f.rnd == 0:
            upto = f.acq_draw
        self.fill(sc, self.xs, y, 0.35 * f.acq, ACQ, plot=self.acq_plot)
        self.curve(sc, self.xs, y, ACQ, width=2.0, op=0.95 * f.acq, plot=self.acq_plot,
                   upto=None if upto is None or upto >= 1 else upto)
        self.label(sc, self.pt(XR[1], 0.5, plot=self.acq_plot), "EI(x)", layout.CAPTION,
                   anchor="lm", dx=8, opacity=f.acq, ink=ACQ_INK, kind="mono")
        if f.rnd == 0 or f.stage == "acq":
            k = int(np.argmax(ei))
            if upto is None or upto >= 1:
                xk, yk = self.xs[k], y[k]
                self.marker(sc, xk, yk, f.acq, color=ACQ, r=5.0, plot=self.acq_plot)
                self.vline(sc, xk, 0.6 * f.acq, ACQ, plot=self.acq_plot)
                self.label(sc, self.pt(xk, yk, plot=self.acq_plot), "max", layout.MIN_TEXT,
                           anchor="lb" if xk < 8.5 else "rb", dx=10 if xk < 8.5 else -10,
                           dy=-4, opacity=f.acq, ink=ACQ_INK)
        if f.rnd > 0 and f.stage == "fly":
            i = self.n0 + f.rnd - 1
            a = self.pt(self.X[i], 0.0, plot=self.acq_plot)
            b = self.pt(self.X[i], self.y[i])
            self.vline(sc, self.X[i], 0.6 * (1 - f.u) * f.back, ACQ, plot=self.acq_plot)
            self.dot(sc, a, b, f.u, ACQ, r=4.5, key="fly")

    def _evaluations(self, sc, f):
        """The evaluations made so far; the latest ringed, the best in teal."""
        n_in = int((f.noise < NOISE_FAR).sum())
        landing = f.rnd > 0 and f.stage == "morph"
        shown = n_in - (1 if landing else 0)
        best = self.best(f)
        for i in range(shown):
            col = MEAN if i == best else OBS
            self.points(sc, [self.X[i]], [self.y[i]], f.intro, color=col,
                        r=5.0 if i == best else 4.2)
        if landing:
            i = n_in - 1
            on = smoothstep(0.0, 0.3, f.u) * f.back
            self.points(sc, [self.X[i]], [self.y[i]], on)
            self.marker(sc, self.X[i], self.y[i], on, color=ACQ, r=7.0)

    def _table(self, sc, f):
        """Every evaluation with its value; the best so far in teal."""
        op = f.intro
        x0, y0 = TABLE_AT
        self.label(sc, self.P(x0, y0), "evaluations", layout.MIN_TEXT, anchor="lm", opacity=op,
                   ink=layout.NOTE_INK)
        n_in = int((f.noise < NOISE_FAR).sum())
        best = self.best(f)
        for i in range(n_in):
            y = y0 + ROW * (i + 1)
            if i >= self.n0 and f.rnd == i - self.n0 + 1 and f.stage == "morph":
                on = op * smoothstep(0.0, 0.3, f.u)
            else:
                on = op
            name = "start" if i < self.n0 else "round %d" % (i - self.n0 + 1)
            if i >= self.n0:
                name += "  E" if explores(self.X, i) else "  x"
            ink = MEAN_INK if i == best else layout.CAPTION_INK
            self.label(sc, self.P(x0, y), name, layout.MIN_TEXT, anchor="lm", opacity=on,
                       ink=ink)
            self.label(sc, self.P(TABLE_RIGHT, y), "g(%s) = %s" % (num(self.X[i], 1),
                                                                   num(self.y[i])),
                       layout.MIN_TEXT, anchor="rm", opacity=on, ink=ink, kind="mono")
        y = y0 + ROW * (ROUNDS + self.n0 + 1.4)
        self.label(sc, self.P(x0, y), "best so far", layout.MIN_TEXT, anchor="lm", opacity=op,
                   ink=layout.NOTE_INK)
        self.label(sc, self.P(TABLE_RIGHT, y), "g(%s) = %s" % (num(self.X[best], 1),
                                                               num(self.y[best])),
                   layout.MIN_TEXT, anchor="rm", opacity=op, ink=MEAN_INK, kind="mono")
        for n, line in enumerate(("E: explore (> ℓ from every point)", "x: exploit (near one)")):
            self.label(sc, self.P(x0, y0 - ROW * (2.2 - n)), line, layout.MIN_TEXT, anchor="lm",
                       opacity=op * (f.rnd > 0), ink=layout.NOTE_INK)

    # ------------------------------------------------------------------ words
    def _words(self, t, f, reset):
        """The subtitle: what is happening now, with the numbers."""
        if reset > 0 or t < self.Tm["start"][0] + 0.3:
            return ("an unknown function g(x) on [0, 10], expensive to evaluate: find its "
                    "maximum in as few evaluations as possible")
        if t < self.Tm["acquisition"][0]:
            return ("start with %d evaluations, g(%s) = %s and g(%s) = %s, and a GP posterior "
                    "(RBF, ℓ = %s): where to look next?"
                    % (self.n0, num(self.X[0], 1), num(self.y[0]), num(self.X[1], 1),
                       num(self.y[1]), num(LENGTHSCALE, 0)))
        if f.rnd == 0:
            return ("the acquisition function, expected improvement EI(x) = E[max(f(x) − best, 0)] "
                    "under the posterior: high where the mean is high or the band is wide")
        if f.rnd > 0 and t < self.Tm["result"][0]:
            i = self.n0 + f.rnd - 1
            best = self.best(f)
            how = ("far from every evaluation: explore" if explores(self.X, i)
                   else "next to what it knows: exploit")
            if f.stage == "acq":
                return ("round %d: EI is highest at x = %s, %s"
                        % (f.rnd, num(self.X[i], 1), how))
            if f.stage == "fly":
                return "round %d: evaluate g(%s)" % (f.rnd, num(self.X[i], 1))
            return ("round %d: g(%s) = %s; the posterior updates and EI is recomputed. "
                    "Best so far g(%s) = %s"
                    % (f.rnd, num(self.X[i], 1), num(self.y[i]), num(self.X[best], 1),
                       num(self.y[best])))
        best = int(np.argmax(self.y))
        return ("after %d rounds the best evaluation is g(%s) = %s; the true maximum is "
                "g(%s) = %s. Explore and exploit, by turns"
                % (ROUNDS, num(self.X[best], 1), num(self.y[best]), num(self.true_x, 1),
                   num(self.true_max)))

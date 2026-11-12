"""A loss landscape you can walk on: logistic regression with two weights on
real MNIST data, its loss over a grid, and the voxel height-field that draws
it with optimiser trajectories on top.

**The model.**  Every MNIST training image of a 0 or a 1 (12,665 of them)
becomes two numbers: its *ink* (the mean pixel value) and its *width* (the
fraction of the 28 columns that contain any ink); ones are thin and faint,
zeros are wide and dark.  Both features are standardised over the set
(mean 0, standard deviation 1).  The model is logistic regression with no
intercept, ``p(one) = sigmoid(w1 ink + w2 width)``, and the loss is the mean
log-loss over the whole set -- a function of two numbers ``(w1, w2)``, so it
can be evaluated on a grid and drawn as a surface.  It is convex, its minimum
is at ``w* = (-4.29, -1.77)`` with loss 0.137 (97.3% of the set right), and
the bowl is only mildly elongated there (condition number 2.4): a genuine,
slightly curved bowl.  Everything is exact and vectorised, and the gradient
is checked against finite differences (``tests/test_landscape.py``).

**The narrow valley.**  :meth:`Problem.scaled` multiplies a feature by a
constant.  With ink times :data:`ILL_SCALE` (10) the weight on it has to be
10 times smaller and the curvature along it is 100 times larger: the same
data now gives a long, narrow valley (condition number 42 at the minimum, 284
at the origin), which is what the optimiser comparison runs on.

**Runs.**  :func:`descend` records an optimiser walking on a problem: the
weights before every step, the (full-batch) loss there, the gradient the
step used (a minibatch's, if ``batch`` is given) and the optimiser's running
moments, so the animations can draw the raw gradient, the step actually
taken and the moments as bars.

**Drawing.**  :class:`Landscape` is the shared renderer: the loss grid as a
height-field of voxel columns (one column per grid cell, as many voxels tall
as the loss there), the two weight axes, walkers (a glowing voxel at the
true loss above the surface, with its trail), arrows for gradients and steps,
the one-sigma ellipse of a minibatch gradient, and the synchronised
loss-against-step read-out beside it.  The animations in
:mod:`voxanim.optim_anim`, :mod:`voxanim.adamw_anim` and
:mod:`voxanim.schedule_anim` are built on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from . import color as C
from . import layout
from .cache import cached_npz
from .easing import smoothstep
from .netanim import screen_box
from .tensor import Block
from .text import num as _num
from .text import sub as _sub

# ------------------------------------------------------------------ the data

DIGITS = (0, 1)                   # the binary problem: is it a one?
FEATURE_NAMES = ("ink", "width")
ILL_SCALE = 10.0                  # ink times this: the ill-conditioned version
N_GRID = 40                       # grid cells along each weight axis


def features(x):
    """``(N, 2)`` raw features of images ``x`` ``(N, 784)`` in [0, 1]: the mean
    pixel value (ink) and the fraction of columns with any ink (width)."""
    img = np.asarray(x, np.float64).reshape(-1, 28, 28)
    ink = img.mean(axis=(1, 2))
    width = (img.sum(axis=1) > 0).sum(axis=1) / 28.0
    return np.stack([ink, width], axis=1)


@dataclass
class Problem:
    """Logistic regression (no intercept) on features ``F`` ``(N, 2)`` with
    labels ``y`` ``(N,)`` in {0, 1}: ``loss(w) = mean softplus(F w) - y F w``.

    ``scale`` records how the standardised features were multiplied (see
    :meth:`scaled`); the raw standardised features are ``F / scale``.
    """

    F: np.ndarray
    y: np.ndarray
    scale: tuple = (1.0, 1.0)

    @property
    def n(self):
        """Number of examples."""
        return len(self.y)

    def scaled(self, k1, k2=1.0):
        """The same data with the features multiplied by ``(k1, k2)``."""
        k = np.array([k1, k2], np.float64)
        return Problem(self.F * k, self.y, (self.scale[0] * k1, self.scale[1] * k2))

    def logits(self, w):
        """``F w`` for one weight vector ``(2,)`` -> ``(N,)``."""
        return self.F @ np.asarray(w, np.float64)

    def loss(self, W, idx=None):
        """Mean log-loss at weights ``W`` ``(..., 2)`` -> ``(...)``; over the
        examples ``idx`` if given (a minibatch), else all of them."""
        F, y = (self.F, self.y) if idx is None else (self.F[idx], self.y[idx])
        W = np.asarray(W, np.float64)
        z = W.reshape(-1, 2) @ F.T                       # (M, N)
        L = (np.logaddexp(0.0, z) - z * y[None, :]).mean(axis=1)
        return L.reshape(W.shape[:-1]) if W.ndim > 1 else float(L[0])

    def per_example_grads(self, w, idx=None):
        """``(n, 2)`` gradient of each example's log-loss at ``w``:
        ``(sigmoid(F w) - y) F``."""
        F, y = (self.F, self.y) if idx is None else (self.F[idx], self.y[idx])
        p = 1.0 / (1.0 + np.exp(-(F @ np.asarray(w, np.float64))))
        return (p - y)[:, None] * F

    def grad(self, w, idx=None):
        """Mean gradient at ``w`` over all examples, or over the minibatch ``idx``."""
        return self.per_example_grads(w, idx).mean(axis=0)

    def grad_cov(self, w):
        """``(2, 2)`` covariance of the per-example gradients at ``w``: a
        minibatch of ``B`` examples (drawn at random) has a gradient with
        covariance ``grad_cov / B`` about the full gradient."""
        g = self.per_example_grads(w)
        return np.cov(g, rowvar=False, bias=True)

    def hessian(self, w):
        """``(2, 2)`` Hessian of the mean loss at ``w``: ``F^T diag(p (1 - p)) F / N``."""
        p = 1.0 / (1.0 + np.exp(-self.logits(w)))
        s = p * (1.0 - p)
        return (self.F * s[:, None]).T @ self.F / self.n

    def accuracy(self, w):
        """Fraction of the set the weights classify right (threshold 0.5)."""
        return float(((self.logits(w) > 0) == (self.y > 0.5)).mean())

    def minimum(self, w0=(0.0, 0.0), iters=60):
        """The minimiser, by Newton's method from ``w0`` (the loss is convex)."""
        w = np.array(w0, np.float64)
        for _ in range(iters):
            w = w - np.linalg.solve(self.hessian(w), self.grad(w))
        return w

    def grid(self, lo, hi, n=N_GRID):
        """``(w1s, w2s, L)``: the loss on an ``n x n`` grid; ``lo`` / ``hi`` are
        ``(w1, w2)`` corners and ``L[i, j]`` is at ``(w1s[i], w2s[j])``."""
        w1s = np.linspace(lo[0], hi[0], n)
        w2s = np.linspace(lo[1], hi[1], n)
        W = np.stack(np.meshgrid(w1s, w2s, indexing="ij"), axis=-1)
        return w1s, w2s, self.loss(W)


@cache
def problem(cache_dir=None):
    """The standardised 0-vs-1 problem on the MNIST training set."""
    from .mnist import load
    xtr, ytr, _xte, _yte = load(cache_dir)
    keep = np.isin(ytr, DIGITS)
    f = features(xtr[keep])
    f = (f - f.mean(axis=0)) / f.std(axis=0)
    return Problem(f, (ytr[keep] == DIGITS[1]).astype(np.float64))


# ------------------------------------------------------------------ runs


@dataclass
class Run:
    """An optimiser's walk on a problem, step by step.

    ``w`` ``(steps + 1, 2)`` are the weights before each step and after the
    last, ``loss`` the full-batch loss at each of them; ``g`` ``(steps, 2)``
    is the gradient each step used (the minibatch's when there was one) and
    ``m`` / ``v`` the optimiser's first / second moment after each step
    (zeros for optimisers without them).  ``lr`` is the learning rate of
    each step.
    """

    w: np.ndarray
    loss: np.ndarray
    g: np.ndarray
    m: np.ndarray
    v: np.ndarray
    lr: np.ndarray

    @property
    def steps(self):
        """Number of steps taken."""
        return len(self.g)

    def at(self, k):
        """``(w, loss)`` before step ``k`` (``k = steps``: the end)."""
        return self.w[k], float(self.loss[k])

    def first_within(self, target, tol):
        """The first step whose loss is within ``tol`` of ``target`` (or -1)."""
        hit = np.nonzero(self.loss - target <= tol)[0]
        return int(hit[0]) if len(hit) else -1


def descend(prob, make_opt, w0, steps, batch=None, rng=None, lr_at=None, decay=0.0):
    """Record ``steps`` steps of an optimiser on ``prob`` from ``w0``.

    ``make_opt(params)`` builds the optimiser (any class of
    :mod:`voxanim.optim`); ``batch`` draws that many examples (with ``rng``)
    for each step's gradient; ``lr_at(k)`` gives step ``k``'s learning rate
    (a schedule); ``decay`` adds ``decay * w`` to every gradient (L2 in the
    loss).  Returns a :class:`Run`.
    """
    P = {"w": np.array(w0, np.float64)}
    opt = make_opt(P)
    ws, losses, gs, ms, vs, lrs = [P["w"].copy()], [prob.loss(P["w"])], [], [], [], []
    for k in range(steps):
        idx = None if batch is None else rng.integers(0, prob.n, batch)
        g = prob.grad(P["w"], idx) + decay * P["w"]
        lr = None if lr_at is None else float(lr_at(k))
        opt.step(P, {"w": g}, lr)
        gs.append(g)
        ms.append(np.array(getattr(opt, "m", {"w": np.zeros(2)})["w"]))
        vs.append(np.array(getattr(opt, "v", {"w": np.zeros(2)})["w"]))
        lrs.append(opt.lr if lr is None else lr)
        ws.append(P["w"].copy())
        losses.append(prob.loss(P["w"]))
    return Run(np.array(ws), np.array(losses), np.array(gs), np.array(ms), np.array(vs),
               np.array(lrs))


def run_cache(name, settings, build, cache_dir=None):
    """Record runs through :func:`voxanim.cache.cached_npz`: ``build()`` returns
    ``{run name: Run}``; the file stores every array under ``<name>.<field>``."""
    def arrays():
        out = {}
        for key, run in build().items():
            for field in ("w", "loss", "g", "m", "v", "lr"):
                out["%s.%s" % (key, field)] = getattr(run, field)
        return out
    d = cached_npz(name, settings, arrays, cache_dir)
    keys = sorted({k.split(".")[0] for k in d})
    return {key: Run(*(d["%s.%s" % (key, f)] for f in ("w", "loss", "g", "m", "v", "lr")))
            for key in keys}


# ------------------------------------------------------------------ drawing

SURF_LO = C.rgb("#2f6f6c")         # the lowest loss: deep teal
SURF_HI = C.rgb("#e9e4d4")         # the highest: pale sand
AXIS_INK = (0.46, 0.48, 0.50)
GRID_INK = (0.62, 0.64, 0.66)
ARROW_GRAD = C.rgb("#c0399b")      # the raw gradient (magenta, as in the other animations)
ARROW_STEP = C.rgb("#1f6fb5")      # the step actually taken (blue)
START = (0.98, 0.98, 0.97)         # the start marker
HEAD = 0.35                        # arrow head length, world units
LIFT = 0.55                        # how far above the surface walkers and lines float
CHART_INK = (0.5, 0.52, 0.54)


def tick(v):
    """An axis value as text: whole numbers plain, halves with one decimal."""
    v = float(v)
    return _num(v, 0) if abs(v - round(v)) < 1e-9 else _num(v, 1)


def surface_colors(h):
    """Column colours for heights ``h`` in [0, 1] (low teal, high sand)."""
    a = np.clip(np.asarray(h, np.float64), 0.0, 1.0)[:, None] ** 0.8
    return (SURF_LO[None] + (SURF_HI - SURF_LO)[None] * a).astype(np.float32)


class Landscape:
    """The loss grid of ``prob`` over the box ``lo .. hi`` as voxel columns.

    Column ``(i, j)`` stands at weights ``(w1s[i], w2s[j])`` and is
    ``1 + (levels - 1) * (clip(L, L_min, L_cap) - L_min) / (L_cap - L_min)``
    voxels tall, rounded: terraces of equal loss, like contour lines.  ``cap``
    is the loss the tallest column shows (higher losses are clipped, so the
    bowl's floor keeps its detail).  ``at`` is the layout fraction of the
    box's centre; ``scale`` the voxel size.
    """

    def __init__(self, anim, prob, lo, hi, cap, at, scale=0.45, levels=20, n=N_GRID):
        self.anim, self.prob = anim, prob
        self.lo, self.hi = np.asarray(lo, np.float64), np.asarray(hi, np.float64)
        self.cap, self.levels, self.n = float(cap), int(levels), int(n)
        self.w1s, self.w2s, self.L = prob.grid(lo, hi, n)
        self.L_min = float(self.L.min())
        # rows = height levels (row 0 on top), cols = w1, channels = w2
        self.block = Block((levels, n, n), scale=scale, name="landscape")
        self.block.place(anim.cam.frac(*at))
        anim.blocks_extra.append(self.block)
        self.heights = self.height(self.L)                         # (n, n) in voxels
        cols = surface_colors((self.heights.ravel() - 1) / max(1, levels - 1))
        self.block.base[...] = np.broadcast_to(cols.reshape(1, n, n, 3), (levels, n, n, 3))

    # -- geometry -----------------------------------------------------------
    def height(self, L):
        """Column height in voxels (1 .. levels) for loss ``L`` (array)."""
        a = (np.clip(L, self.L_min, self.cap) - self.L_min) / (self.cap - self.L_min)
        return np.rint(1 + (self.levels - 1) * a).astype(int)

    def z_of(self, L):
        """World z of the surface at loss ``L`` (continuous, not terraced)."""
        b = self.block
        a = (np.clip(L, self.L_min, self.cap) - self.L_min) / (self.cap - self.L_min)
        return b.origin[2] + (1 + (self.levels - 1) * a) * b.scale

    def xy(self, w):
        """World ``(x, y)`` of weights ``w`` ``(..., 2)``: w1 along x, w2 along y."""
        b = self.block
        w = np.asarray(w, np.float64)
        u = (w - self.lo) / (self.hi - self.lo) * (self.n - 1)
        x = b.origin[0] + (u[..., 0] + 0.5) * b.scale
        y = b.origin[1] + (u[..., 1] + 0.5) * b.scale
        return x, y

    def point(self, w, L=None, lift=LIFT):
        """World point above the surface at weights ``w`` (loss ``L``, or the
        model's loss there), ``lift`` voxels above the terrace."""
        x, y = self.xy(w)
        L = self.prob.loss(w) if L is None else L
        return np.array([x, y, self.z_of(L) + lift * self.block.scale])

    def inside(self, w):
        """True when weights ``w`` lie inside the drawn box."""
        return bool(np.all(w >= self.lo) and np.all(w <= self.hi))

    def bbox(self):
        """World box of the columns."""
        b = self.block
        return b.origin, b.origin + b.extent

    # -- per-frame -----------------------------------------------------------
    def show(self, reveal=1.0, opacity=1.0):
        """Set the columns for this frame: ``reveal`` in [0, 1] raises them
        from the floor (a sweep across w1 with the heights growing), ``opacity``
        fades the whole field."""
        b = self.block
        n, levels = self.n, self.levels
        h = self.heights.astype(np.float64)
        if reveal < 1.0:
            # the sweep: columns at the far end start later; each grows in
            i = np.arange(n)[:, None]
            start = 0.6 * i / (n - 1)
            u = np.clip((reveal * 1.6 - start) / 0.4, 0.0, 1.0)
            u = u * u * (3 - 2 * u)
            h = np.where(u > 0, np.maximum(1.0, np.rint(h * u)), 0.0)
        rows = levels - 1 - np.arange(levels)[:, None, None]          # height of row i
        b.present[...] = rows < h[None, :, :]
        b.opacity *= opacity

    def axes(self, sc, op, labels=True, size=0.027):
        """The two weight axes along the near edges -- w1 along the lower-left
        edge, w2 along the left edge -- with their names and three tick values
        each (ends and middle)."""
        if op <= 0.01:
            return
        a = self.anim
        b = self.block
        lo, hi = self.bbox()
        z = lo[2] - 0.05
        pad = 0.7 * b.scale
        p0 = np.array([lo[0], hi[1] + pad, z])          # w1: x from lo to hi, nearest y
        p1 = np.array([hi[0], hi[1] + pad, z])
        q0 = np.array([lo[0] - pad, lo[1], z])          # w2: y from lo to hi, nearest x
        q1 = np.array([lo[0] - pad, hi[1], z])
        sc.line(p0, p1, width=1.1, color=AXIS_INK, opacity=0.9 * op, bias=0.0)
        sc.line(q0, q1, width=1.1, color=AXIS_INK, opacity=0.9 * op, bias=0.0)
        if not labels:
            return
        cam = a.cam

        def put(p, text, anchor, dx=0.0, dy=0.0, ink=layout.CAPTION_INK):
            s = cam.project(p)
            sc.text(s[0] / cam.width + dx, s[1] / cam.height + dy, text, size, "regular", ink,
                    op * a._cap, anchor)

        for u, v in ((0.0, self.lo), (0.5, (self.lo + self.hi) / 2), (1.0, self.hi)):
            # w1 ticks under the lower edge; the first sits right of the corner,
            # out of the way of w2's last tick
            put(p0 + (p1 - p0) * u, tick(v[0]), "lt" if u == 0 else "mt", dy=0.008,
                ink=layout.NOTE_INK)
            put(q0 + (q1 - q0) * u, tick(v[1]), "rm", dx=-0.008, ink=layout.NOTE_INK)
        put((p0 + p1) / 2, "w%s (%s)" % (_sub(1), FEATURE_NAMES[0]), "mt", dy=0.045)
        put((q0 + q1) / 2, "w%s (%s)" % (_sub(2), FEATURE_NAMES[1]), "rm", dx=-0.04)

    def marker(self, sc, w, L, color, op, size=0.9, glow=0.35, key=None):
        """A walker: a glowing voxel above the surface at ``w``."""
        if op <= 0.01:
            return
        p = self.point(w, L)
        sc.voxel(p, color, op, glow=glow, shadow=False, reflect=False,
                 size=size * self.block.scale)
        if key is not None:
            sc.probe(key, p, tol=4.0)

    def trail(self, sc, run, upto, color, op, width=1.4, fade_len=None, lift=LIFT):
        """The path of ``run`` up to (fractional) step ``upto``: lines between
        the recorded points, the current segment partly drawn.  ``fade_len``
        makes older segments fainter (a window of that many steps)."""
        if op <= 0.01 or upto <= 0:
            return
        k = int(np.floor(upto))
        f = upto - k
        pts = [self.point(run.w[i], run.loss[i], lift) for i in range(0, min(k, run.steps) + 1)]
        if f > 0 and k < run.steps:
            w = run.w[k] + (run.w[k + 1] - run.w[k]) * f
            L = run.loss[k] + (run.loss[k + 1] - run.loss[k]) * f
            pts.append(self.point(w, L, lift))
        for i in range(len(pts) - 1):
            a = 1.0
            if fade_len:
                a = max(0.15, 1.0 - (len(pts) - 2 - i) / fade_len)
            sc.line(pts[i], pts[i + 1], width=width, color=color, opacity=op * a, bias=0.15)

    def arrow(self, sc, p, d, color, op, width=2.0, lift=0.0):
        """An arrow from world point ``p`` along the world vector ``d`` (in the
        horizontal plane), with a head."""
        if op <= 0.01:
            return
        d = np.asarray(d, np.float64)
        n = np.linalg.norm(d[:2])
        if n < 1e-6:
            return
        p = np.asarray(p, np.float64) + np.array([0.0, 0.0, lift])
        q = p + d
        sc.line(p, q, width=width, color=color, opacity=op, bias=0.3)
        u = d[:2] / n
        head = min(HEAD * self.block.scale * 2, 0.6 * n)
        for s in (1.0, -1.0):
            side = np.array([-u[1], u[0]]) * s
            back = -u * head + side * head * 0.5
            sc.line(q, q + np.array([back[0], back[1], 0.0]), width=width, color=color,
                    opacity=op, bias=0.3)

    def step_vec(self, w_from, w_to):
        """World vector between the surface points of two weights (horizontal)."""
        x0, y0 = self.xy(w_from)
        x1, y1 = self.xy(w_to)
        return np.array([x1 - x0, y1 - y0, 0.0])

    def ellipse(self, sc, w, L, cov, color, op, segments=36, width=1.3):
        """The one-sigma ellipse of a step with covariance ``cov`` ``(2, 2)`` in
        weight units, drawn on the surface around ``w``."""
        if op <= 0.01:
            return
        vals, vecs = np.linalg.eigh(cov)
        vals = np.sqrt(np.maximum(vals, 0.0))
        c = self.point(w, L)
        k = (self.n - 1) * self.block.scale / (self.hi - self.lo)      # world per weight unit
        th = np.linspace(0, 2 * np.pi, segments + 1)
        pts = []
        for a in th:
            d = vecs @ (vals * np.array([np.cos(a), np.sin(a)]))
            pts.append(c + np.array([d[0] * k[0], d[1] * k[1], 0.0]))
        for i in range(segments):
            sc.line(pts[i], pts[i + 1], width=width, color=color, opacity=op, bias=0.25)

    def caption_below(self, sc, text, op, size=0.027, offset=0.0):
        """A caption centred under the box (clear of the axis labels)."""
        self.anim.caption(sc, self.bbox(), text, op, size=size, offset=0.06 + offset,
                          max_width=0.5)


# ------------------------------------------------------------------ the loss read-out


class LossChart:
    """Loss against step, in a box of layout fractions ``((x0, x1), (top,
    bottom))``, with a vertical range ``(lo, hi)`` in loss and ``steps`` on
    the horizontal axis.  Curves are drawn up to a fractional step, with the
    current value written beside the box."""

    def __init__(self, anim, box, steps, lo, hi, log=False):
        self.anim, (self.xs, self.ys), self.steps = anim, box, steps
        self.lo, self.hi, self.log = lo, hi, log

    def pt(self, step, loss):
        """World point of ``(step, loss)`` in the chart's plane."""
        (x0, x1), (y0, y1) = self.xs, self.ys
        v = np.log10(max(loss, 1e-12)) if self.log else loss
        lo, hi = (np.log10(self.lo), np.log10(self.hi)) if self.log else (self.lo, self.hi)
        fy = y1 - (y1 - y0) * np.clip((v - lo) / (hi - lo), 0.0, 1.0)
        return self.anim.cam.frac(x0 + (x1 - x0) * min(1.0, step / self.steps), fy)

    def frame(self, sc, op, xlabel="step", ylabel="loss"):
        """The axes and their labels."""
        if op <= 0.01:
            return
        a = self.anim
        (x0, x1), (y0, y1) = self.xs, self.ys
        cam, txt = a.cam, a.fr
        sc.line(cam.frac(x0, y1), cam.frac(x1, y1), width=1.0, color=CHART_INK, opacity=0.8 * op,
                bias=0.0)
        sc.line(cam.frac(x0, y1), cam.frac(x0, y0), width=1.0, color=CHART_INK, opacity=0.8 * op,
                bias=0.0)
        c = op * a._cap
        sc.text(*txt(x0, y0 - 0.012), ylabel, 0.026, "regular", layout.CAPTION_INK, c, "lb")
        sc.text(*txt(x1, y1 + 0.012), xlabel, 0.026, "regular", layout.CAPTION_INK, c, "rt")

    def curve(self, sc, losses, upto, color, op, width=1.5, marker=True, dashed=False):
        """``losses`` (one per step, from step 0) drawn up to fractional step
        ``upto``; with ``marker`` a dot at the end."""
        if op <= 0.01 or upto <= 0:
            return
        k = int(np.floor(upto))
        n = min(k, len(losses) - 1)
        stride = max(1, n // 160)              # at most ~160 segments
        ks = list(range(0, n + 1, stride))
        if ks[-1] != n:
            ks.append(n)
        pts = [self.pt(i, losses[i]) for i in ks]
        if upto > n and n + 1 < len(losses):
            f = upto - n
            pts.append(self.pt(upto, losses[n] + (losses[n + 1] - losses[n]) * f))
        for i in range(len(pts) - 1):
            if dashed and i % 2:
                continue
            sc.line(pts[i], pts[i + 1], width=width, color=color, opacity=op, bias=0.05)
        if marker:
            sc.voxel(pts[-1], color, op, glow=0.3, shadow=False, reflect=False, size=0.55)

    def hline(self, sc, loss, op, color=CHART_INK, dashes=14):
        """A dashed horizontal line at ``loss`` (the minimum, say)."""
        if op <= 0.01:
            return
        a, b = self.pt(0, loss), self.pt(self.steps, loss)
        for i in range(dashes):
            u0, u1 = i / dashes, (i + 0.5) / dashes
            sc.line(a + (b - a) * u0, a + (b - a) * u1, width=0.9, color=color, opacity=op,
                    bias=0.0)

    def label(self, sc, step, loss, text, color, op, dx=0.008, dy=0.0, anchor="lm", size=0.026):
        """Text next to the chart point ``(step, loss)``."""
        if op <= 0.01:
            return
        s = self.anim.cam.project(self.pt(step, loss))
        sc.text(s[0] / self.anim.cam.width + dx, s[1] / self.anim.cam.height + dy, text, size,
                "regular", color, op * self.anim._cap, anchor)


def reveal_curve(u):
    """Smooth 0..1 for the landscape's entrance."""
    return smoothstep(0.0, 1.0, u)


def box_center_px(anim, box):
    """Screen centre, in frame fractions, of a world box."""
    x0, y0, x1, y1 = screen_box(anim.cam, *box)
    return (x0 + x1) / 2 / anim.cam.width, (y0 + y1) / 2 / anim.cam.height

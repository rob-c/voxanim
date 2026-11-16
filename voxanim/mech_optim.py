"""Optimisation, what a step does: animations 75-80 of the "next 100".

Each draws the mechanism first and the measurement after it. 75, 76 and 80 draw
the redesigned experiments of :mod:`voxanim.study_optim`, which record
everything shown (the weights after every update, the loss across the sharpest
valley, Adam's state). 77-79 draw the first studies
(:mod:`voxanim.next_systems`, :mod:`voxanim.next_systems_real`); what those did
not keep is recomputed here by running the study's own experiment again, step
for step (the tests check that the rerun reproduces the study's curves).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from . import next_systems as S
from . import realdata
from . import study_optim as O
from .cache import cached_npz
from .diagram import draw_image
from .easing import ease_in_out, smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIM,
    GREEN,
    MAGENTA,
    PAPER,
    RED,
    TEAL,
    VIOLET,
    Mechanism,
    dark,
    soft,
)


def _cached(name, build, **settings):
    settings = dict(version=1, source=realdata.source_hash(
        "mech_optim.py", "next_systems.py", "next_systems_real.py", "optim.py"), **settings)
    return {k: np.asarray(v) for k, v in cached_npz("mech-optim-" + name, settings, build).items()}


# ====================================================================== the reruns
@lru_cache(maxsize=1)
def ridge_problem():
    """The studies' ridge regression of one-hot MNIST labels (75 and 79):
    inputs with a bias column, targets, and the exact Hessian of the loss."""
    from .next_systems_real import digits

    x, y, xt, yt, _ = digits()
    a, at = np.c_[x, np.ones(len(x))], np.c_[xt, np.ones(len(xt))]
    target = np.eye(10)[y]
    h = a.T @ a / len(a) + 0.02 * np.eye(a.shape[1])
    return dict(a=a, at=at, target=target, yt=yt, h=h, x=x, y=y)


@lru_cache(maxsize=2)
def ridge_paths(newton):
    """:func:`voxanim.next_systems_real.precondition` again, keeping the weights
    before every update: ``loss`` (3, 81), ``accuracy`` (3, 81) and ``w``
    (3, 81, 50, 10) for gradient descent, Adam (``newton``) or diagonal
    scaling, and the full Hessian solve."""
    from .optim import Adam

    p = ridge_problem()
    a, at, target, yt, h = p["a"], p["at"], p["target"], p["yt"], p["h"]
    scale = 1 / np.sqrt(np.diag(h))
    diagonal_rate = 0.9 / np.linalg.eigvalsh(scale[:, None] * h * scale[None, :]).max()
    rate = 0.9 / np.linalg.eigvalsh(h).max()
    losses, tests, paths = [], [], []
    for method in range(3):
        w = np.zeros((a.shape[1], 10))
        parameters = {"w": w}
        opt = Adam(parameters, 0.08)
        ls, ev, ws = [], [], []
        for step in range(81):
            error = a @ w - target
            ls.append(float(np.mean(np.sum(error * error, axis=1)) / 2 + 0.01 * np.sum(w * w)))
            ev.append(float(np.mean((at @ w).argmax(1) == yt)))
            ws.append(np.array(w))
            if step == 80:
                break
            g = a.T @ error / len(a) + 0.02 * w
            if method == 0:
                w = w - rate * g
            elif method == 1:
                if newton:
                    parameters["w"] = w
                    opt.step(parameters, {"w": g})
                    w = parameters["w"]
                else:
                    w = w - diagonal_rate * g / np.diag(h)[:, None]
            else:
                w = w - np.linalg.solve(h, g)
        losses.append(ls)
        tests.append(ev)
        paths.append(ws)
    return dict(loss=np.array(losses), accuracy=np.array(tests), w=np.array(paths),
                rate=rate, diagonal_rate=diagonal_rate)


def batch_gradients():
    """:func:`voxanim.next_systems.batch_noise` again (the same batches, drawn
    from the same generator), keeping what it averaged: every sampled batch's
    gradient, as its two coordinates in the plane of the full gradient
    (``xy[..., 0]``, along it) and the direction in which single-image
    gradients vary most (``xy[..., 1]``). ``noise`` is the study's measure,
    ``inplane`` the part of it this plane shows, ``index`` the images of
    each size's first eight batches (padded with -1)."""
    def build():
        x, y, _xt, _yt = S._digit_data(512)
        P = S.mlp_init(49, 20, 10, np.random.default_rng(17))
        rng = np.random.default_rng(18)
        full = S._pack(S.mlp_loss(P, x, y)[1])
        each = np.array([S._pack(S.mlp_loss(P, x[i:i + 1], y[i:i + 1])[1]) for i in range(len(y))])
        e1 = full / np.linalg.norm(full)
        rest = each - full
        rest = rest - np.outer(rest @ e1, e1)
        e2 = np.linalg.svd(rest, full_matrices=False)[2][0]
        basis = np.stack([e1, e2])
        batches = np.array([1, 4, 16, 64, 256])
        noise, xy, inplane, index = [], [], [], []
        for b in batches:
            picks = rng.integers(0, len(y), (100, b))
            g = np.array([S._pack(S.mlp_loss(P, x[r], y[r])[1]) for r in picks])
            noise.append(np.mean(np.linalg.norm(g - full, axis=1) ** 2))
            xy.append(g @ basis.T)
            inplane.append(np.mean(np.sum(((g - full) @ basis.T) ** 2, axis=1)))
            pad = np.full((8, 256), -1)
            pad[:, :b] = picks[:8]
            index.append(pad)
            # the study's training run at this size draws from the same generator
            Q = {k: v.copy() for k, v in P.items()}
            for _ in range(100):
                ix = rng.integers(0, len(y), b)
                _, G = S.mlp_loss(Q, x[ix], y[ix])
                for k in Q:
                    Q[k] -= min(0.8, 0.05 * b) * G[k]
        return dict(batches=batches, noise=noise, xy=xy, inplane=inplane, index=index,
                    full=full @ basis.T, signal=full @ full)

    return _cached("batch-gradients", build)


def averaging_run():
    """:func:`voxanim.next_systems.weight_average` again, keeping the three
    sets of weights after every update (the study kept SGD's every fifth).
    ``xy`` (3, 401, 2): SGD, the exponential average and the running mean in
    the study's plane (the two directions SGD's recorded weights vary most
    in). ``late`` (3, 201, 2): updates 200-400 seen in the plane that holds
    all three final sets of weights, and ``field``, the held-out loss over
    that plane (at ``fx`` x ``fy``). ``picks``: the 32 images of each update."""
    def build():
        x, y, xt, yt = S._digit_data(640)
        rng = np.random.default_rng(17)
        P = S.mlp_init(49, 24, 10, rng)
        ema = {k: v.copy() for k, v in P.items()}
        swa = {k: v.copy() for k, v in P.items()}
        packs, picks, count = [], [], 0
        for i in range(401):
            packs.append([S._pack(Q) for Q in (P, ema, swa)])
            if i == 400:
                break
            ix = rng.integers(0, len(y), 32)
            picks.append(ix)
            _, G = S.mlp_loss(P, x[ix], y[ix])
            for k in P:
                P[k] -= 0.8 * G[k]
                ema[k] = 0.97 * ema[k] + 0.03 * P[k]
            if i >= 200:
                count += 1
                for k in P:
                    swa[k] += (P[k] - swa[k]) / count
            else:
                swa = {k: v.copy() for k, v in P.items()}
        W = np.array(packs).transpose(1, 0, 2)                 # kind, update, weight
        snaps = W[0, ::5]
        vt = np.linalg.svd(snaps - snaps.mean(0), full_matrices=False)[2]
        xy = (W - snaps.mean(0)) @ vt[:2].T
        origin = W[2, -1]
        e1 = W[0, -1] - origin
        e1 = e1 / np.linalg.norm(e1)
        e2 = W[1, -1] - origin
        e2 = e2 - (e2 @ e1) * e1
        e2 = e2 / np.linalg.norm(e2)
        basis = np.stack([e1, e2])
        late = (W[:, 200:] - origin) @ basis.T
        lo, hi = late.reshape(-1, 2).min(0), late.reshape(-1, 2).max(0)
        fx = np.linspace(lo[0] - 0.45, hi[0] + 0.45, 44)
        fy = np.linspace(lo[1] - 0.45, hi[1] + 0.45, 44)
        field = np.array([[S.mlp_loss(S._unpack(origin + a * e1 + b * e2, P), xt, yt,
                                      False)[0] for a in fx] for b in fy])
        final = [S.mlp_loss(S._unpack(w, P), xt, yt, False) for w in W[:, -1]]
        return dict(xy=xy, late=late, fx=fx, fy=fy, field=field, final=np.array(final),
                    picks=np.array(picks))

    return _cached("averaging-run", build, plane="three-finals")


# ====================================================================== shared drawing
def upsample(a, n):
    """A 2-D array resampled to ``n x n`` by linear interpolation."""
    a = np.asarray(a, float)
    r, c = np.linspace(0, a.shape[0] - 1, n), np.linspace(0, a.shape[1] - 1, n)
    rows = np.array([np.interp(c, np.arange(a.shape[1]), row) for row in a])
    return np.array([np.interp(r, np.arange(a.shape[0]), col) for col in rows.T]).T


def terrain(t, ink, levels=9):
    """A height map as a texture: ``t`` in 0..1 (0 the lowest) from paper up
    to a soft ``ink``, with a contour line wherever it crosses a level."""
    t = np.clip(np.asarray(t, float), 0.0, 1.0)
    tex = PAPER + (np.asarray(soft(ink, 0.30), float) - PAPER) * t[..., None]
    q = np.floor(t * levels - 1e-9)
    edge = np.zeros(t.shape, bool)
    edge[:, 1:] |= q[:, 1:] != q[:, :-1]
    edge[1:, :] |= q[1:, :] != q[:-1, :]
    edge[:, :-1] |= edge[:, 1:]
    edge[:-1, :] |= edge[1:, :]
    tex[edge] = tex[edge] * 0.55 + np.asarray(dark(ink, 0.25), float) * 0.45
    return tex.astype(np.float32)


class Map:
    """A rectangle of the frame showing ``xr`` x ``yr`` of some plane (y up):
    ``m(x, y)`` is the layout point of a plane point."""

    def __init__(self, box, xr, yr):
        (self.x0, self.x1), (self.y0, self.y1) = box
        self.xr, self.yr = xr, yr

    def __call__(self, x, y):
        return (self.x0 + (self.x1 - self.x0) * (x - self.xr[0]) / (self.xr[1] - self.xr[0]),
                self.y1 - (self.y1 - self.y0) * (y - self.yr[0]) / (self.yr[1] - self.yr[0]))

    @property
    def centre(self):
        return (self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2

    def lattice(self, n):
        """The plane points under an ``n x n`` texture (row 0 at the top)."""
        xs = self.xr[0] + (np.arange(n) + 0.5) / n * (self.xr[1] - self.xr[0])
        ys = self.yr[1] - (np.arange(n) + 0.5) / n * (self.yr[1] - self.yr[0])
        return np.meshgrid(xs, ys)


class Optim(Mechanism):
    """What these six share: a plane drawn as terrain, and paths over it."""

    def arc(self, sc, a, b, ink, op, u=1.0, bend=0.08, width=2.2, head=True):
        """:meth:`Mechanism.arc`, lifted clear of a terrain drawn under it."""
        if op <= 0.01 or u <= 0.0:
            return
        x, y = self._curve_points(a, b, bend)
        n = max(2, int(np.ceil(len(x) * min(u, 1.0))))
        pts = [self.P(float(px), float(py), 1.2) for px, py in zip(x[:n], y[:n])]
        self.polyline(sc, pts, width, ink, op, bias=0.3)
        if head:
            self._head(sc, pts[-2], pts[-1], ink, op, width)

    def land(self, sc, m, tex, op):
        """Texture ``tex`` filling the map ``m``, framed."""
        w, h = self.span(m.x1 - m.x0, m.y1 - m.y0)
        c = self.P(*m.centre)
        self.rect(sc, c, w + 8, h + 8, PAPER, op, border=AXIS, border_w=1.2, radius=5, lift=0.1)
        draw_image(self, sc, c, tex, w, h, op, grid=0)

    def path(self, sc, pts, ink, op, upto=None, width=2.6):
        """The polyline through layout points ``pts`` as far as the (fractional)
        index ``upto``; returns the point reached."""
        pts = np.asarray(pts, float)
        upto = len(pts) - 1 if upto is None else float(np.clip(upto, 0, len(pts) - 1))
        k = int(upto)
        f = upto - k
        end = pts[k] if k >= len(pts) - 1 else pts[k] + (pts[k + 1] - pts[k]) * f
        if op > 0.01 and upto > 0:
            line = [self.P(float(x), float(y), 1.2) for x, y in pts[:k + 1]]
            line.append(self.P(float(end[0]), float(end[1]), 1.2))
            self.polyline(sc, line, width, ink, op, bias=0.3)
        return float(end[0]), float(end[1])

    def dot(self, sc, at, ink, op, r=6.5, ring=False):
        """A lit point at layout ``at``; ``ring``: an open circle instead."""
        if ring:
            self.rect(sc, self.P(*at), 2 * r, 2 * r, ink, op, border=ink, border_w=2.4, radius=r,
                      lift=0.9, fill_op=0.0)
        else:
            self.disk(sc, self.P(*at), r, ink, op, glow=0.45, lift=1.5, outline=0.0)

    def dashes(self, sc, a, b, ink, op, n=14, width=1.6):
        """A dashed straight line between layout points."""
        a, b = np.asarray(a, float), np.asarray(b, float)
        for i in range(n):
            p, q = a + (b - a) * (i / n), a + (b - a) * ((i + 0.55) / n)
            self.polyline(sc, [self.P(float(p[0]), float(p[1]), 1.2),
                               self.P(float(q[0]), float(q[1]), 1.2)], width, ink, op, bias=0.2)

    def bars(self, sc, fx, fy, rows, op, u, width=0.34, step=0.10, thick=30.0, scale=None):
        """Horizontal bars ``(label, ink, value, text)`` stacked from ``(fx, fy)``."""
        full = self.span(width)[0]
        top = scale if scale is not None else max(v for _, _, v, _ in rows)
        for i, (label, ink, value, text) in enumerate(rows):
            self.hbar(sc, fx, fy + i * step, full * value / top, ink, op, label, text,
                      u, thick=thick, track=full)


# ====================================================================== 75
class NewtonVsGradientDescent(Optim):
    """Gradient descent at a safe step, at one near its limit and at one past
    it, then Newton's step, on one bowl: the loss of the study's ridge
    classifier in the plane of its two steepest directions, each method's
    recorded weights walking over it; then the updates each needs."""

    TITLE = "Newton's step and gradient descent"
    PHASES = (("the bowl", 9.0), ("a safe step", 12.0), ("a larger step", 14.0),
              ("Newton", 11.0), ("the count", 11.0))
    NAMES = ("step 1 ÷ steepest", "step 1.9 ÷ steepest", "step 2.1 ÷ steepest", "Newton step")
    INKS = (RED, BLUE, MAGENTA, GREEN)
    XR, YR = (-0.42, 0.20), (-0.25, 0.31)          # shallow way across, steep way up

    def setup(self):
        s = self.study
        d = self.d = O.newton_data()
        self.h = np.asarray(s.panels[0].data["values"], float)
        self.loss = np.asarray(s.panels[1].data["ys"], float)
        self.cond = np.asarray(s.panels[2].data["x"], float)
        self.counts = np.asarray(s.panels[2].data["ys"], float).T      # ridge, method
        self.accuracy = np.asarray(s.panels[3].data["values"], float)
        self.main = list(O.NEWTON_RIDGES).index(s.metadata["ridge_drawn"])
        lam, vec = np.linalg.eigh(self.h)
        self.lam = float(lam[-1]), float(lam[-2])
        w = d["w"]
        best = w[3, -1]                                      # the Newton step's: the minimum
        coords = np.einsum("ik,mtkc->mtic", vec.T[[-1, -2]], w - best)
        share = (0.5 * np.array(self.lam)[:, None] * coords[0, 0] ** 2).sum(0)
        self.digit_class = int(share.argmax())
        c = coords[:, :, :, self.digit_class]                # method, update, (steep, shallow)
        self.route = np.stack([c[..., 1], c[..., 0]], -1)    # as (across, up)
        self.start = self.route[0, 0]
        self.slopes = self.lam[1] * self.start[0], self.lam[0] * self.start[1]
        self.floor = float(d["floor"])
        self.goal = float(d["goal"])
        self.digits = O.newton_problem()["x"][:6]
        self._tex = {}

    def subtitle_for(self, k):
        n = self.counts[self.main]
        return [
            "the loss of a digit classifier is a bowl over its weights; measured on 1,024 MNIST "
            "digits, its steepest direction curves %.0f times more than its shallowest: its "
            "condition number" % self.cond[self.main],
            "gradient descent with the largest step that does not overshoot the steep way: one "
            "update reaches its floor, then the same small step creeps along the shallow "
            "ways; %d updates to get within 1%% of the bottom" % n[0],
            "a step 1.9 times larger zig-zags across the steep way and goes faster along the "
            "shallow ones: %d updates; but at 2.1 times every hop lands higher, and the loss "
            "grows from %.2f to %s in 80 updates" % (n[1], self.loss[2, 0],
                                                      f"{self.loss[2, -1]:,.0f}"),
            "Newton's step divides the slope in each direction by the curvature there, which "
            "aims it at the bottom: one update takes the loss from %.3f to %.4f, the lowest "
            "possible" % (self.loss[3, 0], self.loss[3, 1]),
            "gradient descent's step is capped by the steepest direction, so its count grows "
            "with the condition number: %d updates where that is %.0f, %d where it is %s; "
            "Newton's step takes one at any" % (self.counts[-1, 0], self.cond[-1],
                                                self.counts[0, 0],
                                                f"{self.cond[0]:,.0f}"),
        ][k]

    # ------------------------------------------------------------------ the bowl
    def _map(self, x0):
        w, h = self.frac(556 * (self.XR[1] - self.XR[0]), 556 * (self.YR[1] - self.YR[0]))
        return Map(((x0, x0 + w), (0.305, 0.305 + h)), self.XR, self.YR)

    def _bowl(self, sc, m, op, labels=True):
        if "bowl" not in self._tex:
            x, y = m.lattice(400)
            f = 0.5 * (self.lam[1] * x * x + self.lam[0] * y * y)
            self._tex["bowl"] = terrain(np.sqrt(f / f.max()), AMBER, 11)
        self.land(sc, m, self._tex["bowl"], op)
        self.dot(sc, m(0, 0), dark(GREEN, 0.3), op, 7.5, ring=True)
        if labels:
            self.text(sc, m.x0 + 0.008, m.y0 + 0.03, "steep way: curvature %.2f" % self.lam[0],
                      op, anchor="lm")
            self.text(sc, m.x1 - 0.008, m.y1 - 0.03, "shallow way: %.2f" % self.lam[1], op,
                      anchor="rm")

    def _routes(self, m):
        return [np.array([m(x, y) for x, y in r]) for r in self.route]

    def _inside(self, i):
        """How many of method ``i``'s recorded weights lie on the map before one leaves."""
        x, y = self.route[i, :, 0], self.route[i, :, 1]
        out = (x < self.XR[0]) | (x > self.XR[1]) | (y < self.YR[0]) | (y > self.YR[1])
        return int(np.argmax(out)) if out.any() else len(x)

    def phase_1(self, sc, u, op):
        xs = [0.085 + 0.052 * i for i in range(6)]
        for i, x in enumerate(xs):
            self.digit(sc, self.digits[i], x, 0.345, 40, op * smoothstep(0.0, 0.15, u - 0.02 * i),
                       frame=DIM)
        self.text(sc, 0.39, 0.345, "1,024 digits", op, anchor="lm", strong=True, size=0.034)
        grow = smoothstep(0.1, 0.45, u)
        self.grid(sc, self.h, 0.215, 0.665, 216, 216, op, reveal=grow, hot=AMBER, lines=0,
                  vmax=0.45)
        self.lines(sc, 0.36, 0.56, ["curvature of the", "loss: how each pair",
                                    "of the 50 input", "weights bends it"], op * grow,
                   step=0.045, size=0.032)
        m = self._map(0.545)
        show = op * smoothstep(0.45, 0.7, u)
        self.arrow(sc, (0.36, 0.78), (m.x0 - 0.012, 0.78), AXIS, show, smoothstep(0.45, 0.7, u))
        self._bowl(sc, m, show)
        start = m(*self.start)
        self.dot(sc, start, AXIS, show, 6.0)
        self.text(sc, start[0] + 0.015, start[1] + 0.035, "start: all weights 0", show,
                  anchor="lm")
        self.text(sc, m(0, 0)[0] + 0.018, m(0, 0)[1] - 0.03, "lowest loss", show, anchor="lm",
                  ink=dark(GREEN, 0.3))

    def _readout(self, sc, x, y, op, k, which):
        self.text(sc, x, y, "update %d" % k, op, anchor="lm", strong=True, size=0.036)
        for row, i in enumerate(which):
            value = self.loss[i, k]
            self.text(sc, x, y + 0.055 * (row + 1), "%s   loss %s" % (
                self.NAMES[i], "%.4f" % value if value < 10 else f"{value:,.0f}"),
                op, anchor="lm", ink=dark(self.INKS[i]), size=0.034)

    def _walk(self, sc, pts, i, op, k, every=10, upto=81):
        end = self.path(sc, pts[:upto], self.INKS[i], op, min(k, upto - 1), 3.0)
        for j in range(min(int(k) + 1, upto)):
            if j < 6 or j % every == 0:
                self.dot(sc, tuple(pts[j]), self.INKS[i], op * 0.9, 3.6)
        self.dot(sc, end, self.INKS[i], op, 7.0)

    def _gap_chart(self, sc, x, op, which, top=0.585):
        """The loss above its minimum for the ``(method, updates drawn, opacity)`` given."""
        steps = np.arange(81)
        above = np.clip(self.loss - self.floor, 1e-4, 30.0)        # off the chart at 30
        chart = self.chart(((x + 0.065, 0.94), (top, 0.83)), steps, [1e-4, 30.0], ylog=True)
        self.axes(sc, chart, op, "update", "loss above the lowest possible",
                  yticks=[(v, t) for v, t in ((1e-4, "0.0001"), (1e-2, "0.01"), (1.0, "1"))])
        y = chart.fy(self.goal - self.floor)
        self.dashes(sc, (chart.x0, y), (chart.x1, y), dark(GREEN, 0.3), op * 0.9, 22, 1.6)
        self.text(sc, chart.x0 + 0.01, y + 0.03, "within 1% of the bottom", op, anchor="lm",
                  ink=dark(GREEN, 0.3))
        for i, k, show in which:
            n = int(np.argmax(above[i] >= 30.0)) if (above[i] >= 30.0).any() else 81
            self.plot(sc, chart, steps[:n], above[i][:n], self.INKS[i], op * show,
                      min(k / max(n - 1, 1), 1.0))

    def phase_2(self, sc, u, op):
        m = self._map(0.05)
        self._bowl(sc, m, op)
        s = min(u / 0.88, 1.0)
        k = 4 * s / 0.45 if s < 0.45 else 4 + 76 * ((s - 0.45) / 0.55) ** 1.6
        self._walk(sc, self._routes(m)[0], 0, op, k)
        x = 0.485
        self._readout(sc, x, 0.33, op, int(k), (0,))
        self.lines(sc, x, 0.45, ["step × steepest curvature = 1: it lands on the steep",
                                 "way's floor; along the shallow way it covers %.0f%% of"
                                 % (100 * self.lam[1] / self.lam[0]),
                                 "what is left each update, and less in the 48 others"],
                   op * smoothstep(0.1, 0.25, u), step=0.042)
        self._gap_chart(sc, x, op, ((0, k, 1.0),), top=0.62)

    def phase_3(self, sc, u, op):
        m = self._map(0.05)
        self._bowl(sc, m, op, labels=False)
        pts = self._routes(m)
        self.path(sc, pts[0], RED, op * 0.45, None, 2.0)
        k = 80 * min(u / 0.55, 1.0) ** 1.7
        self._walk(sc, pts[1], 1, op, k, every=100)
        # past the limit: the hops that stay on the map, then it has left
        late = smoothstep(0.62, 0.66, u)
        n = self._inside(2)
        j = (n - 1) * min(max(u - 0.64, 0.0) / 0.22, 1.0)
        if late > 0:
            self._walk(sc, pts[2], 2, op * late, j, every=1, upto=n)
        x = 0.485
        over = int(round(80 * min(max(u - 0.64, 0.0) / 0.3, 1.0)))
        if late <= 0:
            self._readout(sc, x, 0.33, op, int(k), (0, 1))
        else:
            self._readout(sc, x, 0.33, op, over, (0, 1, 2))
        self._gap_chart(sc, x, op, ((0, k, 1.0), (1, k, 1.0), (2, over, late)))

    def phase_4(self, sc, u, op):
        m = self._map(0.05)
        self._bowl(sc, m, op)
        pts = self._routes(m)
        self.path(sc, pts[0], RED, op * 0.45, None, 2.0)
        start, bottom = tuple(pts[3][0]), tuple(pts[3][1])
        down = smoothstep(0.05, 0.25, u)
        self.arrow(sc, start, tuple(pts[0][1] + (pts[0][1] - pts[0][0]) * 0.9), RED, op, down, 3.0)
        go = smoothstep(0.5, 0.72, u)
        self.arrow(sc, start, bottom, GREEN, op * smoothstep(0.42, 0.5, u), max(go, 0.02), 3.2)
        at = (start[0] + (bottom[0] - start[0]) * go, start[1] + (bottom[1] - start[1]) * go)
        self.dot(sc, at, GREEN, op, 7.5)
        x = 0.485
        self.text(sc, x, 0.33, "downhill is not the way to the bottom", op * down, anchor="lm",
                  ink=dark(RED), size=0.034)
        rows = op * smoothstep(0.22, 0.42, u)
        self.text(sc, x, 0.42, "slope ÷ curvature = distance to the bottom", rows, anchor="lm",
                  strong=True, size=0.034)
        for row, (name, slope, lam, dist) in enumerate((
                ("steep way", self.slopes[1], self.lam[0], self.start[1]),
                ("shallow way", self.slopes[0], self.lam[1], self.start[0]))):
            y = 0.50 + row * 0.065
            self.text(sc, x, y, name, rows, anchor="lm", size=0.034)
            self.text(sc, x + 0.14, y, "%.3f ÷ %.3f = %.3f" % (abs(slope), lam, abs(dist)), rows,
                      anchor="lm", size=0.034, ink=dark(GREEN, 0.3))
        done = op * smoothstep(0.72, 0.85, u)
        k = 1 if u > 0.72 else 0
        self.text(sc, x, 0.68, "update %d" % k, op * smoothstep(0.42, 0.5, u), anchor="lm",
                  strong=True, size=0.036)
        self.text(sc, x, 0.735, "Newton step  %.4f" % self.loss[3, k],
                  op * smoothstep(0.42, 0.5, u), anchor="lm", ink=dark(GREEN, 0.3), size=0.034)
        self.text(sc, x, 0.79, "gradient descent  %.4f" % self.loss[0, k],
                  op * smoothstep(0.42, 0.5, u), anchor="lm", ink=dark(RED), size=0.034)
        self.lines(sc, x, 0.86, ["one update lands on the lowest loss:",
                                 "this loss is an exact bowl, so the curvature tells all"], done,
                   step=0.045)

    def phase_5(self, sc, u, op):
        inks = (RED, BLUE, GREEN)
        names = ("gradient descent, step 1 ÷ steepest", "gradient descent, step 1.9 ÷ steepest",
                 "Newton step")
        chart = self.chart(((0.12, 0.56), (0.36, 0.80)), [2.0, 3000.0], [0.7, 1000.0],
                           xlog=True, ylog=True)
        self.axes(sc, chart, op, "condition number: steepest ÷ shallowest curvature",
                  "updates to get within 1%",
                  xticks=[(v, "%d" % v) for v in (3, 10, 100, 1000)],
                  yticks=[(v, "%d" % v) for v in (1, 10, 100, 1000)])
        order = np.argsort(self.cond)
        shown = len(order) * min(u / 0.5, 1.0)
        for i in (0, 1, 2):
            pts = [(chart.fx(self.cond[j]), chart.fy(self.counts[j, i])) for j in order]
            self.path(sc, pts, inks[i], op * 0.8, shown - 1 if shown >= 1 else 0, 2.2)
            for n, p in enumerate(pts):
                if n < shown:
                    self.dot(sc, p, inks[i], op, 6.0)
        x = 0.62
        j = self.main
        self.text(sc, x, 0.36, "this bowl: condition number %.0f" % self.cond[j], op,
                  anchor="lm", strong=True, size=0.034)
        show = op * smoothstep(0.45, 0.6, u)
        for i in range(3):
            y = 0.44 + i * 0.105
            self.text(sc, x, y, names[i], show, anchor="lm", ink=dark(inks[i], 0.25), size=0.032)
            self.text(sc, x, y + 0.045, "%d update%s,  %.1f%% of held-out digits right" % (
                self.counts[j, i], "" if self.counts[j, i] == 1 else "s",
                100 * self.accuracy[i]), show, anchor="lm")
        self.lines(sc, x, 0.78, ["the other bowls: the same digits with",
                                 "a weaker or stronger pull of the",
                                 "weights towards 0 (ridge)"],
                   op * smoothstep(0.6, 0.75, u), step=0.042)


# ====================================================================== 79
def coupled_pair():
    """The two pixel weights of the ridge classifier that the curvature
    matrix couples most (largest H_ab / sqrt(H_aa H_bb), bias excluded), and
    the loss over those two alone, for the digit class whose best weights lie
    furthest along the shallow direction:
    its 2 x 2 curvature ``h``, its minimum ``best`` (start: both 0), and the
    exact updates of plain gradient descent, diagonal scaling and the full
    solve on it, with the study's step-size rules (``paths``, 3 x 31 x 2)."""
    p = ridge_problem()
    h, a, target = p["h"], p["a"], p["target"]
    d = np.sqrt(np.diag(h))
    r = (h / d[:, None] / d[None, :])[:49, :49]
    np.fill_diagonal(r, 0.0)
    i, j = (int(v) for v in np.unravel_index(np.argmax(r), r.shape))
    h2 = h[np.ix_([i, j], [i, j])]
    b = (a.T @ target / len(a))[[i, j]]                       # 2 x 10
    best = np.linalg.solve(h2, b)
    _lam, vec = np.linalg.eigh(h2)
    c = int(np.argmax((vec[:, 0] @ best) ** 2))               # most to do along the valley
    best = best[:, c]
    scaled = h2 / np.sqrt(np.outer(np.diag(h2), np.diag(h2)))
    rates = 0.9 / np.linalg.eigvalsh(h2).max(), 0.9 / np.linalg.eigvalsh(scaled).max()
    paths = []
    for method in range(3):
        w, ws = np.zeros(2), []
        for _ in range(31):
            ws.append(w.copy())
            g = h2 @ (w - best)
            w = w - (rates[0] * g, rates[1] * g / np.diag(h2), np.linalg.solve(h2, g))[method]
        paths.append(ws)
    return dict(pixels=(i, j), h=h2, best=best, digit=c, paths=np.array(paths),
                rho=float(scaled[0, 1]))


def root(h):
    """The symmetric square root of a positive-definite matrix."""
    lam, vec = np.linalg.eigh(h)
    return (vec * np.sqrt(lam)) @ vec.T


class SecondOrderPreconditioners(Optim):
    """Preconditioning as a change of coordinates: the slanted valley of two
    coupled weights, squared up by diagonal scaling, made round by the full
    curvature matrix, beside the whole 50 x 50 matrix going the same way."""

    TITLE = "Preconditioning"
    PHASES = (("coupled weights", 10.0), ("plain steps", 10.0), ("diagonal scaling", 12.0),
              ("full matrix", 12.0), ("80 updates", 11.0))
    NAMES = ("plain", "diagonal", "full matrix")
    INKS = (RED, BLUE, GREEN)
    BOWL = ((0.05, 0.05 + 0.355), (0.305, 0.305 + 0.631))       # 309 px square

    def setup(self):
        s = self.study
        self.h = np.asarray(s.panels[0].data["values"], float)
        self.loss = np.asarray(s.panels[1].data["ys"], float)
        self.cond = np.asarray(s.panels[2].data["values"], float)
        self.accuracy = np.asarray(s.panels[3].data["values"], float)
        self.pair = coupled_pair()
        h2 = self.pair["h"]
        d = np.sqrt(np.diag(self.h))
        # the coordinates each method takes its plain gradient steps in, and the
        # whole curvature matrix seen in those coordinates
        self.maps = [np.eye(2), np.diag(np.sqrt(np.diag(h2))), root(h2)]
        self.mats = [self.h, self.h / np.outer(d, d), np.eye(len(self.h))]
        self.mean = ridge_problem()["x"].mean(0)
        self._tex = {}

    def subtitle_for(self, k):
        c = self.cond
        return [
            "two of a digit classifier's weights read pixels that are often lit together, so "
            "the loss over them is a slanted valley: neither can be set without the other",
            "plain gradient descent: quick across the valley, slow along it. Over all 50 "
            "inputs the steepest curvature is %.0f times the shallowest" % c[0],
            "diagonal scaling gives every weight its own step size: the axes are squared up "
            "but the slant stays, and the ratio only falls from %.0f to %.0f" % (c[0], c[1]),
            "the full curvature matrix also undoes the coupling between weights: the valley "
            "becomes a round bowl, the ratio %.0f, and one step reaches the bottom" % c[2],
            "the whole classifier after 80 updates: loss %.4f plain, %.4f diagonal, %.4f with "
            "the full matrix; held-out digits right %.1f%%, %.1f%%, %.1f%%" % (
                *self.loss[:, -1], *(100 * self.accuracy)),
        ][k]

    # ------------------------------------------------------------------ the valley
    def _valley(self, sc, op, t, labels=("weight of pixel A", "weight of pixel B")):
        """The two-weight loss in the coordinates ``t`` (a 2 x 2 map from the raw
        weights); returns ``place(w)``: the layout point of raw weights ``w``."""
        best, h2 = self.pair["best"], self.pair["h"]
        z0 = t @ (0 - best)
        half = 1.25 * float(np.abs(z0).max())
        m = Map(self.BOWL, (-half, half), (-half, half))
        key = np.round(t, 4).tobytes()
        if key not in self._tex:
            if len(self._tex) > 60:
                self._tex.clear()
            x, y = m.lattice(300)
            back = np.linalg.inv(t)
            zx, zy = back[0, 0] * x + back[0, 1] * y, back[1, 0] * x + back[1, 1] * y
            f = 0.5 * (h2[0, 0] * zx * zx + 2 * h2[0, 1] * zx * zy + h2[1, 1] * zy * zy)
            start = 0.5 * best @ h2 @ best
            self._tex[key] = terrain(np.sqrt(f / start) * 0.42, AMBER, 12)
        self.land(sc, m, self._tex[key], op)
        self.dot(sc, m(0, 0), dark(GREEN, 0.3), op, 7.5, ring=True)
        if labels:
            self.text(sc, m.x1 - 0.008, m.y1 - 0.03, labels[0] + "  →", op, anchor="rm")
            if labels[1]:
                self.text(sc, m.x0 + 0.008, m.y0 + 0.03, "↑  " + labels[1], op, anchor="lm")

        def place(w):
            z = t @ (np.asarray(w, float) - best)
            return m(z[0], z[1])
        return place

    def _matrix(self, sc, op, values, note, fx=0.615):
        i, j = self.pair["pixels"]
        cell = self.grid(sc, values, fx, 0.565, 200, 200, op, hot=AMBER, lines=0, vmax=1.0)
        for r, k in ((i, j), (j, i)):
            self.rect(sc, self.P(*cell(r, k)), 13, 13, VIOLET, op, border=VIOLET, border_w=2.2,
                      radius=2, lift=0.8, fill_op=0.0)
        self.text(sc, cell(i, j)[0] + 0.022, cell(i, j)[1], "A-B", op, anchor="lm",
                  ink=dark(VIOLET))
        self.text(sc, fx, 0.335, "curvature, all 50 inputs", op, strong=True)
        self.lines(sc, 0.50, 0.845, note, op, step=0.045)

    def _ratio(self, sc, op, upto, u=1.0, fx=0.775):
        """The conditioning bars for the first ``upto`` coordinate systems."""
        self.text(sc, fx, 0.40, "steepest ÷", op, anchor="lm")
        self.text(sc, fx, 0.44, "shallowest", op, anchor="lm")
        full = self.span(0.115)[0]
        for i in range(upto):
            grow = u if i == upto - 1 else 1.0
            y = 0.535 + i * 0.125
            self.text(sc, fx, y - 0.045, self.NAMES[i], op, anchor="lm", ink=dark(self.INKS[i]))
            self.hbar(sc, fx, y, full * self.cond[i] / self.cond[0], self.INKS[i], op,
                      value="%.0f" % self.cond[i], u=grow, thick=24.0, track=full)

    def _steps(self, sc, place, op, method, k):
        pts = np.array([place(w) for w in self.pair["paths"][method]])
        end = self.path(sc, pts, self.INKS[method], op, k, 3.0)
        for j in range(int(k) + 1):
            self.dot(sc, tuple(pts[j]), self.INKS[method], op * 0.9, 3.4)
        self.dot(sc, end, self.INKS[method], op, 7.0)

    def phase_1(self, sc, u, op):
        place = self._valley(sc, op * smoothstep(0.35, 0.6, u), self.maps[0])
        self.dot(sc, place((0, 0)), AXIS, op * smoothstep(0.35, 0.6, u), 6.0)
        i, j = self.pair["pixels"]
        fx, fy, size = 0.56, 0.47, 132.0
        self.digit(sc, self.mean / self.mean.max(), fx, fy, size, op, frame=DIM)
        dw, dh = self.frac(size, size)
        for name, pix in (("A", i), ("B", j)):
            x = fx - dw / 2 + (pix % 7 + 0.5) * dw / 7
            y = fy - dh / 2 + (pix // 7 + 0.5) * dh / 7
            self.rect(sc, self.P(x, y), size / 7 + 2, size / 7 + 2, VIOLET, op, border=VIOLET,
                      border_w=2.4, radius=2, lift=0.8, fill_op=0.0)
            self.text(sc, fx + dw / 2 + 0.025, y, name, op, ink=dark(VIOLET), size=0.036)
            self.dashes(sc, (x + dw / 14 + 0.004, y), (fx + dw / 2 + 0.012, y), VIOLET, op, 5)
        self.text(sc, fx, fy + dh / 2 + 0.04, "the average digit", op)
        self.lines(sc, 0.68, 0.36, ["pixels A and B are lit", "together in many digits:",
                                    "their weights' coupling", "is %.2f on a scale" % (
                                        self.pair["rho"]), "where 1 is total"], op, step=0.045,
                   size=0.032)
        show = op * smoothstep(0.55, 0.8, u)
        self.lines(sc, 0.49, 0.74, ["the loss over these two weights alone",
                                    "(fitting the digit %d): a slanted valley" % (
                                        self.pair["digit"]),
                                    "start: both weights 0; the ring: lowest loss"], show,
                   step=0.048, size=0.032)

    def phase_2(self, sc, u, op):
        place = self._valley(sc, op, self.maps[0])
        self._steps(sc, place, op, 0, 30 * min(u / 0.85, 1.0) ** 1.4)
        self._matrix(sc, op, self.mats[0], ["each square: how one pair of weights bends the",
                                            "loss together; off the diagonal: coupling"])
        self._ratio(sc, op, 1, u / 0.5)

    def _morph(self, a, b, f):
        return self.maps[a] + (self.maps[b] - self.maps[a]) * f, \
            self.mats[a] + (self.mats[b] - self.mats[a]) * f

    def phase_3(self, sc, u, op):
        f = ease_in_out(smoothstep(0.08, 0.42, u))
        t, mat = self._morph(0, 1, f)
        place = self._valley(sc, op, t, ("A, in its own units", "B, in its own units")
                             if f > 0.5 else ("weight of pixel A", "weight of pixel B"))
        self.path(sc, [place(w) for w in self.pair["paths"][0]], RED, op * 0.4 * (1 - f), None, 2.0)
        if u < 0.45:
            self.dot(sc, place((0, 0)), BLUE, op, 7.0)
        else:
            self._steps(sc, place, op, 1, 30 * min((u - 0.45) / 0.45, 1.0) ** 1.4)
        self._matrix(sc, op, mat, ["every row and column divided by its own diagonal",
                                   "entry: the diagonal is all 1, the coupling is still there"])
        self._ratio(sc, op, 2, (u - 0.2) / 0.4)

    def phase_4(self, sc, u, op):
        f = ease_in_out(smoothstep(0.08, 0.45, u))
        t, mat = self._morph(1, 2, f)
        place = self._valley(sc, op, t, ("the two weights, uncoupled", "")
                             if f > 0.5 else ("A, in its own units", "B, in its own units"))
        self.path(sc, [place(w) for w in self.pair["paths"][1]], BLUE, op * 0.4 * (1 - f), None,
                  2.0)
        go = smoothstep(0.55, 0.75, u)
        a, b = place((0, 0)), place(self.pair["best"])
        self.arrow(sc, a, b, GREEN, op * smoothstep(0.5, 0.55, u), max(go, 0.02), 3.2)
        self.dot(sc, (a[0] + (b[0] - a[0]) * go, a[1] + (b[1] - a[1]) * go), GREEN, op, 7.0)
        self._matrix(sc, op, mat, ["multiplied by its own inverse: nothing is left off",
                                   "the diagonal, so no weight pulls on another"])
        self._ratio(sc, op, 3, (u - 0.2) / 0.4)

    def phase_5(self, sc, u, op):
        steps = np.arange(81)
        chart = self.chart(((0.11, 0.56), (0.36, 0.82)), steps, self.loss)
        self.axes(sc, chart, op, "update", "training loss of the whole classifier")
        k = min(u / 0.7, 1.0)
        for i in (0, 1, 2):
            self.plot(sc, chart, steps, self.loss[i], self.INKS[i], op, k)
        now = int(k * 80)
        self.lines(sc, 0.32, 0.40, [("update %d" % now, None)] + [
            ("%s  %.4f" % (self.NAMES[i], self.loss[i, now]), dark(self.INKS[i]))
            for i in range(3)], op, step=0.05, size=0.034)
        self.text(sc, 0.73, 0.36, "held-out digits right", op, anchor="lm", strong=True,
                  size=0.034)
        self.bars(sc, 0.73, 0.47, [("", self.INKS[i], self.accuracy[i],
                                    "%.1f%%" % (100 * self.accuracy[i])) for i in range(3)],
                  op, (u - 0.3) / 0.4, width=0.15, step=0.115, scale=1.0, thick=30.0)
        for i in range(3):
            self.text(sc, 0.72, 0.47 + i * 0.115, self.NAMES[i], op, anchor="rm",
                      ink=dark(self.INKS[i]))
        self.lines(sc, 0.62, 0.84, ["a real network has millions of weights:",
                                    "its full matrix cannot be stored, so",
                                    "optimisers approximate it in blocks"],
                   op * smoothstep(0.6, 0.8, u), step=0.045)


# ====================================================================== 76
class EdgeOfStability(Optim):
    """Full-batch gradient descent seen across the sharpest valley of the
    loss: the valley narrows as training goes on until a step lands as high on
    the far wall as it started (curvature 2 / step size); there the weights
    swing out, the loss jumps and the curvature is thrown back. Then the loss
    and the curvature of the study's runs at four step sizes."""

    TITLE = "The edge of stability"
    PHASES = (("the sharpest valley", 9.0), ("it sharpens", 12.0), ("over the edge", 15.0),
              ("the loss", 9.0), ("every step size", 12.0))
    INKS = (TEAL, RED, AMBER, BLUE)        # step sizes 0.2, 0.3, 0.4, 0.5
    VALLEY = ((0.09, 0.58), (0.40, 0.84))
    CAPTION = "up: the loss.   across: the weights, moved along the sharpest direction"
    MAIN = O.EDGE_MAIN

    def setup(self):
        s = self.study
        d = self.d = O.edge_data()
        self.loss = np.asarray(s.panels[0].data["ys"], float)
        self.sharp = np.asarray(s.panels[1].data["ys"], float)
        self.at = np.asarray(s.panels[1].data["x"], float)
        self.limits = np.asarray(s.panels[1].data["limits"], float)
        self.window = np.asarray(s.panels[2].data["x"], int)
        self.w_loss, self.w_lam, self.w_off = np.asarray(s.panels[2].data["ys"], float)
        self.level = np.asarray(s.panels[3].data["each"], float)        # start, step size
        self.lrs = tuple(O.EDGE_STEPS)
        self.lr = self.lrs[self.MAIN]
        self.limit = float(self.limits[self.MAIN])
        self.first = int(d["first"])
        self.snaps = np.asarray(d["snaps"], int)[1:]         # from update 200: at rest across
        prof = np.asarray(d["profiles"], float)[1:]
        self.rise = prof - prof.min(1, keepdims=True)
        self.width = np.asarray(d["width"], float)
        self.snap_sharp = np.asarray(d["snap_sharp"], float)[1:]
        self.top = float(self.rise.max())
        # one step of the main size from the wall of the first valley drawn
        self.q0 = -0.22
        slope = float(np.interp(self.q0, self.width, np.gradient(self.rise[0], self.width)))
        self.q1 = self.q0 - self.lr * slope
        self.digits = d["digits"]
        self.rises = (np.diff(self.loss, axis=1) > 0).sum(1)
        self.peak = int(np.argmax(self.w_loss))
        self.accuracy = [float(np.mean(v)) for v in s.metadata["held_out_accuracy"]]
        bins = self.loss[:, 1:].reshape(4, -1, 30)
        self.binned = np.c_[self.loss[:, :1], bins.max(2)]
        self.bin_at = np.arange(self.binned.shape[1]) * 30

    def subtitle_for(self, k):
        lam = self.snap_sharp
        mean = self.level.mean(0)
        return [
            "a small network learning 256 MNIST digits: across the sharpest direction of its "
            "loss the surface is a valley, and a step of %g lands %g × its curvature (%.2f "
            "here) = %.2f of the way to the floor" % (self.lr, self.lr, lam[0],
                                                      self.lr * lam[0]),
            "as gradient descent goes on the valley narrows by itself: its curvature climbs "
            "from %.2f to the limit 2 ÷ %g = %.2f by update %s; there a step lands as high on "
            "the far wall as it started" % (self.sharp[self.MAIN, 0], self.lr, self.limit,
                                            f"{self.first:,}"),
            "past the limit every step lands farther from the floor than it started: the "
            "weights swing out, the loss jumps from %.3f to %.3f, and the swing carries them "
            "to where the curvature is %.2f, under the limit again" % (
                self.w_loss[0], self.w_loss.max(), self.w_lam.min()),
            "with step size %g the limit, %.0f, is never reached and the loss never rises; "
            "with %g it rises at %s of 6,000 updates and still ends lower: %.4f against "
            "%.4f" % (self.lrs[0], self.limits[0], self.lr,
                      f"{int(self.rises[self.MAIN]):,}",
                      self.loss[self.MAIN, -1], self.loss[0, -1]),
            "each step size levels off at its own limit: from update 4,000 on the curvature "
            "averages %.2f, %.2f and %.2f for step sizes %g, %g and %g, over three starts; "
            "the limits are %.2f, %.0f and %.0f" % (*mean[1:], *self.lrs[1:], *self.limits[1:]),
        ][k]

    # ------------------------------------------------------------------ the valley
    def _box(self, sc, op, xr, yr, caption=True):
        m = Map(self.VALLEY, xr, yr)
        self.rect(sc, self.P(*m.centre), *self.span(m.x1 - m.x0 + 0.02, m.y1 - m.y0 + 0.03),
                  PAPER, op, border=AXIS, border_w=1.2, radius=6, lift=0.1)
        if caption:
            self.text(sc, (m.x0 + m.x1) / 2, m.y1 + 0.05, self.CAPTION, op)
        return m

    def _wall(self, sc, m, xs, ys, ink, op, width=3.4, dashed=False):
        keep = np.asarray(ys) <= m.yr[1]
        pts = [m(x, y) for x, y in zip(np.asarray(xs)[keep], np.asarray(ys)[keep])]
        if dashed:
            for j in range(0, len(pts) - 1, 2):
                self.path(sc, pts[j:j + 2], ink, op, None, width)
        elif len(pts) > 1:
            self.path(sc, pts, ink, op, None, width)

    def _hop(self, sc, a, b, f, ink, op, bend=0.12, r=8.0):
        """A ball going from layout ``a`` to ``b`` over an arc; ``f`` in 0..1."""
        f = float(np.clip(f, 0.0, 1.0))
        if f > 0:
            self.arc(sc, a, b, ink, op * 0.9, ease_in_out(f), bend, 2.4, head=f >= 1.0)
        x, y = self._curve_points(a, b, bend, n=41)
        k = ease_in_out(f) * 40
        i = int(min(k, 39))
        at = float(x[i] + (x[i + 1] - x[i]) * (k - i)), float(y[i] + (y[i + 1] - y[i]) * (k - i))
        self.dot(sc, at, ink, op, r)
        return at

    def _floor(self, sc, m, op, label="floor"):
        x = m(0.0, 0.0)[0]
        self.dashes(sc, (x, m.y1 - 0.005), (x, m.y0 + 0.06), GREEN, op, 9)
        self.text(sc, x, m.y0 + 0.035, label, op, ink=dark(GREEN, 0.3))

    def _rest_map(self, sc, op):
        return self._box(sc, op, (float(self.width[0]), float(self.width[-1])),
                         (-0.10 * self.top, 1.04 * self.top))

    def _limit_wall(self, sc, m, op):
        fine = np.linspace(self.width[0], self.width[-1], 81)
        self._wall(sc, m, fine, 0.5 * self.limit * fine * fine, dark(RED, 0.1), op * 0.75, 1.6,
                   dashed=True)

    def phase_1(self, sc, u, op):
        for i in range(6):
            self.digit(sc, self.digits[i], 0.11 + 0.047 * i, 0.325, 36, op, frame=DIM)
        self.text(sc, 0.40, 0.325, "256 digits  →  a small network  →  its loss", op,
                  anchor="lm", strong=True, size=0.034)
        show = op * smoothstep(0.12, 0.3, u)
        m = self._rest_map(sc, show)
        self._wall(sc, m, self.width, self.rise[0], AXIS, show)
        self._floor(sc, m, show * smoothstep(0.3, 0.42, u))
        a = m(self.q0, float(np.interp(self.q0, self.width, self.rise[0])))
        b = m(self.q1, float(np.interp(self.q1, self.width, self.rise[0])))
        a, b = (a[0], a[1] - 0.022), (b[0], b[1] - 0.022)
        go = (u - 0.5) / 0.25
        if go <= 0:
            self.dot(sc, a, RED, show, 8.0)
        else:
            self._hop(sc, a, b, go, RED, show, 0.07)
        x = 0.655
        self.lines(sc, x, 0.44, ["after %d updates:" % self.snaps[0],
                                 "curvature %.2f" % self.snap_sharp[0]],
                   op * smoothstep(0.3, 0.42, u), step=0.052, size=0.034)
        self.lines(sc, x, 0.60, [("a step of %g lands" % self.lr, dark(RED)),
                                 ("%g × %.2f = %.2f" % (self.lr, self.snap_sharp[0],
                                                        self.lr * self.snap_sharp[0]), dark(RED)),
                                 ("of the way to the floor", dark(RED))],
                   op * smoothstep(0.45, 0.55, u), step=0.052, size=0.034)
        self.lines(sc, x, 0.80, ["more than 2 of the way", "would be higher up the",
                                 "far wall than it started"],
                   op * smoothstep(0.75, 0.88, u), step=0.045)

    def phase_2(self, sc, u, op):
        m = self._rest_map(sc, op)
        f = (len(self.snaps) - 1) * min(u / 0.85, 1.0)
        i = min(int(f), len(self.snaps) - 2)
        rise = self.rise[i] + (self.rise[i + 1] - self.rise[i]) * (f - i)
        self._limit_wall(sc, m, op)
        self._wall(sc, m, self.width, rise, AXIS, op)
        self.dot(sc, (m(0, 0)[0], m(0, 0)[1] - 0.022), RED, op, 8.0)
        self.text(sc, m.centre[0], m.y0 + 0.03, "dashed: curvature %.2f" % self.limit, op,
                  ink=dark(RED))
        now = int(round((self.snaps[i] + (self.snaps[i + 1] - self.snaps[i]) * (f - i)) / 20) * 20)
        k = now // 20
        x = 0.655
        self.text(sc, x, 0.34, "step size %g, update %d" % (self.lr, now), op, anchor="lm",
                  strong=True, size=0.036)
        self.lines(sc, x, 0.395, [("curvature %.2f" % self.sharp[self.MAIN, k], dark(RED)),
                                  "loss %.3f" % self.loss[self.MAIN, now]], op, step=0.05,
                   size=0.034)
        upto = self.at <= self.first + 60
        chart = self.chart(((x + 0.045, 0.94), (0.54, 0.80)), self.at[upto], [2.0, 7.4], pad=0.0)
        self.axes(sc, chart, op, "update", "curvature",
                  xticks=[(v, "%d" % v) for v in (0, 500, 1000, 1500)],
                  yticks=[(v, "%d" % v) for v in (2, 4, 6)])
        y = chart.fy(self.limit)
        self.dashes(sc, (chart.x0, y), (chart.x1, y), dark(RED, 0.1), op * 0.9, 18, 1.8)
        self.text(sc, chart.x0 + 0.01, y - 0.028, "limit: 2 ÷ %g = %.2f" % (self.lr, self.limit),
                  op, anchor="lm", ink=dark(RED))
        self.plot(sc, chart, self.at[upto], self.sharp[self.MAIN][upto], RED, op,
                  now / self.at[upto][-1])

    def phase_3(self, sc, u, op):
        d = self.d
        grid, lines = np.asarray(d["w_grid"], float), np.asarray(d["w_line"], float)
        lo, hi = float(lines.min()), float(lines[:, 4:-4].max())
        m = self._box(sc, op, (float(grid[0]), float(grid[-1])),
                      (lo - 0.10 * (hi - lo), hi + 0.05 * (hi - lo)))
        n = len(self.window) - 1
        slot = n * min(u / 0.93, 1.0) ** 0.7
        t = min(int(slot), n - 1)
        w = slot - t
        self._wall(sc, m, grid, lines[t if w < 0.5 else t + 1], AXIS, op)
        self._floor(sc, m, op)

        def spot(j):
            p = m(self.w_off[j], self.w_loss[j])
            return p[0], p[1] - 0.022
        a, b = spot(t), spot(t + 1)
        self._hop(sc, a, b, w, RED, op, 0.35 * abs(b[0] - a[0]), 8.0)
        now = t if w < 0.5 else t + 1
        times = self.lr * self.w_lam[now]
        over = times > 2.0
        x = 0.655
        self.text(sc, x, 0.34, "step size %g, update %d" % (self.lr, self.window[now]), op,
                  anchor="lm", strong=True, size=0.036)
        self.lines(sc, x, 0.395, [
            "curvature %.2f" % self.w_lam[now],
            ("a step lands %.2f of the way" % times, dark(RED) if over else dark(GREEN, 0.3)),
            ("over 2: each hop ends farther out" if over else
             "under 2: each hop ends closer in", dark(RED) if over else dark(GREEN, 0.3)),
        ], op, step=0.048)
        chart = self.chart(((x + 0.045, 0.94), (0.60, 0.80)), self.window, [0.0, 0.18], pad=0.0)
        self.axes(sc, chart, op, "update", "loss",
                  xticks=[(v, f"{v:,}") for v in self.window[13::20]],
                  yticks=[(v, "%g" % v) for v in (0.0, 0.05, 0.1, 0.15)])
        self.plot(sc, chart, self.window, self.w_loss, RED, op, slot / n)
        self.text(sc, 0.94, 0.56, "loss %.3f" % self.w_loss[now], op, anchor="rm", strong=True,
                  size=0.034)

    def phase_4(self, sc, u, op):
        chart = self.chart(((0.12, 0.64), (0.36, 0.82)), self.bin_at, [0.001, 1.0], ylog=True)
        self.axes(sc, chart, op, "full-batch update",
                  "training loss: the highest of each 30 updates",
                  xticks=[(v, f"{v:,}") for v in (0, 2000, 4000, 6000)],
                  yticks=[(v, t) for v, t in ((0.001, "0.001"), (0.01, "0.01"), (0.1, "0.1"),
                                              (1.0, "1"))])
        k = min(u / 0.8, 1.0)
        for i in (0, self.MAIN):
            self.plot(sc, chart, self.bin_at, self.binned[i], self.INKS[i], op, k, width=2.2)
        now = int(k * 6000)
        x = 0.70
        self.text(sc, x, 0.40, f"update {now:,}", op, anchor="lm", strong=True,
                  size=0.036)
        for row, i in enumerate((0, self.MAIN)):
            self.lines(sc, x, 0.47 + row * 0.16, [
                ("step size %g   loss %.4f" % (self.lrs[i], self.loss[i, now]),
                 dark(self.INKS[i], 0.25)),
                ("it rose at %s updates" % "{:,}".format(int(
                    (np.diff(self.loss[i, :now + 1]) > 0).sum())), None)],
                op, step=0.05, size=0.034)
        self.lines(sc, x, 0.80, ["swinging across the valley", "makes the loss jolt, yet the",
                                 "larger step still gets lower"],
                   op * smoothstep(0.6, 0.8, u), step=0.045)

    def phase_5(self, sc, u, op):
        chart = self.chart(((0.10, 0.60), (0.36, 0.82)), self.at, [2.0, 10.6], pad=0.0)
        self.axes(sc, chart, op, "full-batch update", "curvature of the sharpest direction",
                  xticks=[(v, f"{v:,}") for v in (0, 2000, 4000, 6000)],
                  yticks=[(v, "%d" % v) for v in (2, 4, 6, 8, 10)])
        k = min(u / 0.75, 1.0)
        for i in range(4):
            y = chart.fy(self.limits[i])
            self.dashes(sc, (chart.x0, y), (chart.x1, y), self.INKS[i], op * 0.9, 30, 1.8)
            self.plot(sc, chart, self.at, self.sharp[i], self.INKS[i], op, k, width=2.2)
        self.text(sc, chart.x0 + 0.01, chart.fy(10.0) + 0.03, "dashed: each step size's limit",
                  op, anchor="lm")
        x = 0.65
        self.text(sc, x, 0.36, "step size     limit     settles at", op, anchor="lm", strong=True)
        show = op * smoothstep(0.55, 0.75, u)
        mean, lo, hi = self.level.mean(0), self.level.min(0), self.level.max(0)
        for i in range(4):
            y = 0.43 + i * 0.075
            ink = dark(self.INKS[i], 0.25)
            self.text(sc, x + 0.02, y, "%g" % self.lrs[i], op, anchor="lm", ink=ink, size=0.034)
            self.text(sc, x + 0.115, y, "%.2f" % self.limits[i], op, anchor="lm", ink=ink,
                      size=0.034)
            self.text(sc, x + 0.20, y, "%.2f" % mean[i] if hi[i] - lo[i] < 0.05 else
                      "%.1f–%.1f" % (lo[i], hi[i]), show, anchor="lm", ink=ink, size=0.034)
        self.lines(sc, x, 0.75, ["mean from update 4,000 on, three starts",
                                 "step size %g stays under its limit" % self.lrs[0],
                                 "held-out digits right: %.0f%% at every" % (
                                     100 * np.mean(self.accuracy)),
                                 "step size (256 digits are memorised)"],
                   show, step=0.042)


# ====================================================================== 77
class BatchSizeAndNoise(Optim):
    """A batch's gradient as an arrow beside the true one: eight batches of
    four, then a hundred at each size closing in on it, the measured noise
    against 1/B, and the study's training runs at each size."""

    TITLE = "Batch size and gradient noise"
    PHASES = (("one batch", 12.0), ("bigger batches", 17.0), ("one over B", 11.0),
              ("training", 12.0))
    INKS = (RED, AMBER, GREEN, TEAL, BLUE)          # batches of 1, 4, 16, 64, 256
    PLANE = ((0.05, 0.60), (0.37, 0.86))
    XR, YR = (-2.15, 1.75), (-0.5, 1.45)            # the noisiest direction; along the truth

    def setup(self):
        s = self.study
        self.sizes = [int(b) for b in s.panels[0].data["x"]]
        self.noise = np.asarray(s.panels[1].data["ys"], float)[0]
        self.reference = np.asarray(s.panels[1].data["ys"], float)[1]
        self.loss = np.asarray(s.panels[2].data["ys"], float)
        self.signal = float(s.panels[3].data["values"][0])
        self.rates = [min(0.8, 0.05 * b) for b in self.sizes]
        self.grads = batch_gradients()
        from .mnist import load
        self.images = load()[0][:512]
        self.match = int(np.argmin(np.abs(self.noise - self.signal)))

    def subtitle_for(self, k):
        n, f = self.noise, self.loss[:, -1]
        return [
            "the true gradient is the average over all 512 training digits; a batch of %d "
            "gives an estimate, and every batch drawn points a different way" % self.sizes[1],
            "100 batches at each size: their squared distance from the true gradient "
            "averages %.2f with 1 digit, %.2f with 16, and %.3f with 256" % (n[0], n[2], n[4]),
            "four times the batch, a quarter of the noise, as 1/B predicts; at %d digits the "
            "noise (%.3f) has shrunk to the size of the true gradient itself (%.3f)" % (
                self.sizes[self.match], n[self.match], self.signal),
            "100 updates at each size, step size %.2f per image up to %.1f: held-out loss "
            "%.2f, %.2f, %.2f with 1, 4, 16 digits; after that four times the images buys "
            "little: %.2f at 64, %.2f at 256" % (self.rates[0], max(self.rates), *f),
        ][k]

    # ------------------------------------------------------------------ the plane of gradients
    def _plane(self, sc, op, zoom=1.0):
        m = Map(self.PLANE, (self.XR[0] / zoom, self.XR[1] / zoom),
                (self.YR[0] / zoom, self.YR[1] / zoom))
        self.rect(sc, self.P(*m.centre), *self.span(m.x1 - m.x0 + 0.015, m.y1 - m.y0 + 0.025),
                  PAPER, op, border=AXIS, border_w=1.2, radius=6, lift=0.1)
        self.dashes(sc, m(m.xr[0], 0), m(m.xr[1], 0), AXIS, op * 0.45, 40, 1.0)
        self.dashes(sc, m(0, m.yr[0]), m(0, m.yr[1]), AXIS, op * 0.45, 20, 1.0)
        self.text(sc, m.x0 + 0.012, m.y0 + 0.03, "violet arrow: the true gradient, from all 512 "
                  "digits", op, anchor="lm", ink=dark(VIOLET))
        return m

    def _truth(self, sc, m, op):
        tip = m(0.0, float(self.grads["full"][0]))
        self.arrow(sc, m(0, 0), tip, VIOLET, op, 1.0, 4.2)
        return tip

    @staticmethod
    def _fmt(v):
        return "%.2f" % v if v >= 1 else "%.3f" % v

    def _batch(self, sc, k, n, fx, fy, size, op):
        """Batch ``n`` of size index ``k`` as a square of its digits."""
        b = self.sizes[k]
        side = int(np.sqrt(b))
        picks = self.grads["index"][k][n][:b]
        tile = np.zeros((side * 28, side * 28))
        for j, i in enumerate(picks):
            r, c = divmod(j, side)
            tile[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = self.images[int(i)].reshape(28, 28)
        self.digit(sc, tile, fx, fy, size, op, frame=self.INKS[k])

    def phase_1(self, sc, u, op):
        m = self._plane(sc, op, 1.75)
        self.text(sc, m.x0 + 0.01, m.y0 - 0.04, "gradients, seen along two of their %s "
                  "directions" % "1,210", op, anchor="lm")
        k, n = 1, 8
        slot = min(u / 0.92, 0.9999) * n
        t, w = int(slot), slot - int(slot)
        xy = self.grads["xy"][k]
        for j in range(t):
            self.path(sc, [m(0, 0), m(xy[j, 1], xy[j, 0])], AMBER, op * 0.45, None, 1.6)
            self.dot(sc, m(xy[j, 1], xy[j, 0]), AMBER, op * 0.85, 4.5)
        self._truth(sc, m, op)
        at = (0.775, 0.50)
        self._batch(sc, k, t, at[0], at[1], 128, op * smoothstep(0.0, 0.12, w))
        self.text(sc, at[0], 0.345, "batch %d: %d digits" % (t + 1, self.sizes[k]), op,
                  strong=True, size=0.034)
        tip = m(xy[t, 1], xy[t, 0])
        go = smoothstep(0.2, 0.6, w)
        self.pulse(sc, (at[0] - 0.08, at[1]), m(0, 0), (w - 0.1) / 0.3, AMBER, op, 0.05, r=6.0)
        self.arrow(sc, m(0, 0), tip, AMBER, op * smoothstep(0.3, 0.4, w), max(go, 0.02), 3.4)
        self.lines(sc, 0.66, 0.70, ["its gradient: the average", "over these 4 digits only",
                                    ("distance² from the truth: %.2f" % float(
                                        np.sum((xy[t] - self.grads["full"]) ** 2)), dark(AMBER)),
                                    "(in this plane)"], op, step=0.048, size=0.032)

    def phase_2(self, sc, u, op):
        m = self._plane(sc, op)
        n = len(self.sizes)
        slot = min(u / 0.95, 0.9999) * n
        t, w = int(slot), slot - int(slot)
        for k in range(t + 1):
            xy = self.grads["xy"][k]
            shown = len(xy) if k < t else int(len(xy) * smoothstep(0.1, 0.7, w))
            dim = 0.22 if k < t else 1.0
            for j in range(shown):
                self.dot(sc, m(xy[j, 1], xy[j, 0]), self.INKS[k], op * dim, 3.6)
        self._truth(sc, m, op)
        self.text(sc, m.x0 + 0.01, m.y0 - 0.04, "100 batches of %d: where each one's gradient "
                  "points" % self.sizes[t], op, anchor="lm", strong=True, size=0.034)
        self._batch(sc, t, 0, 0.705, 0.455, 118, op * smoothstep(0.0, 0.12, w))
        self.text(sc, 0.80, 0.43, "one batch", op, anchor="lm")
        self.text(sc, 0.80, 0.48, "of %d" % self.sizes[t], op, anchor="lm", strong=True,
                  size=0.036)
        full = self.span(0.15)[0]
        self.text(sc, 0.645, 0.625, "noise: mean distance² from the truth", op, anchor="lm")
        for k in range(t + 1):
            grow = smoothstep(0.3, 0.8, w) if k == t else 1.0
            self.hbar(sc, 0.725, 0.68 + k * 0.054, full * self.noise[k] / self.noise[0],
                      self.INKS[k], op, "B = %d" % self.sizes[k], self._fmt(self.noise[k]), grow,
                      thick=20.0, track=full)

    def phase_3(self, sc, u, op):
        b = np.asarray(self.sizes, float)
        chart = self.chart(((0.12, 0.60), (0.36, 0.82)), b, [0.012, 9.0], xlog=True, ylog=True)
        self.axes(sc, chart, op, "digits in a batch (B)", "noise: mean distance² from the truth",
                  xticks=[(v, "%d" % v) for v in b],
                  yticks=[(v, t) for v, t in ((0.01, "0.01"), (0.1, "0.1"), (1.0, "1"))])
        y = chart.fy(self.signal)
        self.dashes(sc, (chart.x0, y), (chart.x1, y), VIOLET, op, 30, 1.8)
        self.text(sc, chart.x0 + 0.012, y + 0.035, "the true gradient's own length²: %.3f" % (
            self.signal), op, anchor="lm", ink=dark(VIOLET))
        a, c = (chart.fx(b[0]), chart.fy(self.reference[0])), (chart.fx(b[-1]),
                                                              chart.fy(self.reference[-1]))
        self.dashes(sc, a, c, AXIS, op, 30, 1.8)
        for i in range(len(b)):
            show = op * smoothstep(i / 5 * 0.6, i / 5 * 0.6 + 0.08, u)
            self.dot(sc, (chart.fx(b[i]), chart.fy(self.noise[i])), self.INKS[i], show, 8.0)
        self.text(sc, chart.x1 - 0.01, chart.y0 + 0.03, "dashed line: exactly 1/B", op,
                  anchor="rm")
        x = 0.675
        self.text(sc, x, 0.36, "B", op, anchor="lm", strong=True)
        self.text(sc, x + 0.075, 0.36, "measured", op, anchor="lm", strong=True)
        self.text(sc, x + 0.19, 0.36, "1/B", op, anchor="lm", strong=True)
        for i in range(len(b)):
            show = op * smoothstep(i / 5 * 0.6, i / 5 * 0.6 + 0.08, u)
            yy = 0.42 + i * 0.055
            self.text(sc, x, yy, "%d" % b[i], show, anchor="lm", ink=dark(self.INKS[i]),
                      size=0.034)
            self.text(sc, x + 0.075, yy, self._fmt(self.noise[i]), show, anchor="lm", size=0.034,
                      ink=dark(self.INKS[i]))
            self.text(sc, x + 0.19, yy, self._fmt(self.reference[i]), show, anchor="lm", size=0.034)
        self.lines(sc, x, 0.74, ["each batch averages B", "independent digits, so",
                                 "its error shrinks as 1/B"], op * smoothstep(0.6, 0.8, u),
                   step=0.045)

    def phase_4(self, sc, u, op):
        steps = np.arange(self.loss.shape[1])
        chart = self.chart(((0.12, 0.60), (0.36, 0.82)), steps, self.loss)
        self.axes(sc, chart, op, "update", "loss on held-out digits")
        k = min(u / 0.75, 1.0)
        for i in range(len(self.sizes)):
            self.plot(sc, chart, steps, self.loss[i], self.INKS[i], op, k, width=1.7)
        now = int(k * (len(steps) - 1))
        x = 0.655
        self.text(sc, x, 0.36, "update %d" % now, op, anchor="lm", strong=True, size=0.036)
        for col, head in ((0.0, "B"), (0.07, "step size"), (0.20, "loss")):
            self.text(sc, x + col, 0.425, head, op, anchor="lm")
        for i, b in enumerate(self.sizes):
            y = 0.48 + i * 0.055
            ink = dark(self.INKS[i])
            self.text(sc, x, y, "%d" % b, op, anchor="lm", ink=ink, size=0.034)
            self.text(sc, x + 0.07, y, "%.2f" % self.rates[i], op, anchor="lm", ink=ink,
                      size=0.034)
            self.text(sc, x + 0.20, y, "%.2f" % self.loss[i, now], op, anchor="lm", ink=ink,
                      size=0.034)
        self.lines(sc, x, 0.79, ["the step size differs too:", "less noise allows a larger",
                                 "step, until the cap at 0.8"], op * smoothstep(0.5, 0.75, u),
                   step=0.045)


# ====================================================================== 78
class WeightAveraging(Optim):
    """SGD's weights wandering, and two averages of them: the exponential one
    trailing the path, the running mean sitting in its middle, drawn over the
    held-out loss in the plane that holds all three final sets of weights."""

    TITLE = "Averaging the weights"
    PHASES = (("noisy steps", 10.0), ("a moving average", 13.0), ("the mean", 11.0),
              ("where they end", 10.0), ("held out", 11.0))
    NAMES = ("SGD", "moving average", "mean since 200")
    INKS = (RED, BLUE, GREEN)
    PLANE = ((0.05, 0.62), (0.37, 0.87))

    def setup(self):
        s = self.study
        self.route = np.asarray(s.panels[0].data["xy"], float)
        self.at = np.asarray(s.panels[1].data["x"], float)
        self.loss = np.asarray(s.panels[1].data["ys"], float)
        self.accuracy = np.asarray(s.panels[2].data["ys"], float)
        self.curvature = np.asarray(s.panels[3].data["values"], float)
        self.run = averaging_run()
        r = self.run
        self.map = Map(self.PLANE, (float(r["fx"][0]), float(r["fx"][-1])),
                       (float(r["fy"][0]), float(r["fy"][-1])))
        f = upsample(r["field"][::-1], 300)                  # row 0 at the top
        lo = float(f.min())
        self.tex = terrain(np.sqrt(np.clip((f - lo) / (0.85 - lo), 0, 1)), AMBER, 12)
        self.late = np.array([[self.map(x, y) for x, y in kind] for kind in r["late"]])
        from .mnist import load
        self.images = load()[0][:640]

    def subtitle_for(self, k):
        f, c = self.loss[:, -1], self.curvature
        return [
            "stochastic gradient descent: each update follows the gradient of 32 random "
            "digits, so the weights wander on their way; this is their path over 400 updates",
            "an exponential moving average is a second set of weights: after every update "
            "it moves 3% of the way to the new ones, so it trails the path and smooths it",
            "stochastic weight averaging keeps the plain mean of every set of weights since "
            "update 200: it sits in the middle of where the run has been",
            "after 400 updates, on 256 held-out digits: the last SGD weights have loss %.3f, "
            "the moving average %.3f and the mean %.3f; the paler the ground, the lower the "
            "loss" % tuple(f),
            "held-out loss through the run: the averages lose SGD's jitter. Curvature at the "
            "end, lower meaning a flatter spot: %.2f for SGD, %.2f and %.2f for the "
            "averages" % tuple(c),
        ][k]

    def _now(self, update):
        """Index of the study's last held-out evaluation at or before ``update``."""
        return int(np.searchsorted(self.at, update, side="right") - 1)

    def _ground(self, sc, op):
        self.land(sc, self.map, self.tex, op)
        m = self.map
        self.text(sc, m.x0 + 0.005, m.y0 - 0.04, "held-out loss over a plane of weights: paler "
                  "is lower", op, anchor="lm")

    def _trail(self, sc, kind, op, upto, width=2.6, r=7.0):
        end = self.path(sc, self.late[kind], self.INKS[kind], op, upto, width)
        self.dot(sc, end, self.INKS[kind], op, r)
        return end

    def _readout(self, sc, op, update, kinds, x=0.665, y=0.37):
        i = self._now(update)
        self.text(sc, x, y, "update %d" % update, op, anchor="lm", strong=True, size=0.036)
        self.text(sc, x, y + 0.05, "held-out loss at update %d" % self.at[i], op, anchor="lm")
        for row, k in enumerate(kinds):
            self.text(sc, x, y + 0.105 + row * 0.052, "%s  %.3f" % (self.NAMES[k], self.loss[k, i]),
                      op, anchor="lm", ink=dark(self.INKS[k]), size=0.034)

    def phase_1(self, sc, u, op):
        k = min(u / 0.9, 1.0) * 400
        now = min(int(k), 399)
        picks = self.run["picks"][now]
        tile = np.zeros((4 * 28, 8 * 28))
        for j, i in enumerate(picks):
            r, c = divmod(j, 8)
            tile[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = self.images[int(i)].reshape(28, 28)
        c = self.P(0.20, 0.49)
        self.rect(sc, c, 248, 128, RED, op, radius=5, lift=0.1)
        tex = (PAPER * (1 - tile[..., None]) + np.asarray((0.13, 0.15, 0.16)) * tile[..., None])
        draw_image(self, sc, c, tex.astype(np.float32), 240, 120, op, grid=0)
        self.text(sc, 0.20, 0.345, "update %d: these 32 digits" % (now + 1), op, strong=True,
                  size=0.034)
        self.lines(sc, 0.07, 0.69, ["their gradient moves every weight", "a little: one step. "
                                    "The next 32", "digits pull another way"], op, step=0.048,
                   size=0.032)
        box = ((0.42, 0.93), (0.36, 0.84))
        lo, hi = self.route.min(0), self.route.max(0)
        pad = 0.08 * (hi - lo)
        m = Map(box, (lo[0] - pad[0], hi[0] + pad[0]), (lo[1] - 3 * pad[1], hi[1] + 3 * pad[1]))
        self.rect(sc, self.P(*m.centre), *self.span(0.525, 0.50), PAPER, op, border=AXIS,
                  border_w=1.2, radius=6, lift=0.1)
        self.arrow(sc, (0.345, 0.49), (0.405, 0.49), AXIS, op)
        pts = np.array([m(x, y) for x, y in self.route])
        end = self.path(sc, pts, RED, op, k / 5.0, 2.8)
        for j in range(0, int(k / 5.0) + 1, 4):
            self.dot(sc, tuple(pts[j]), RED, op * 0.8, 3.2)
        self.dot(sc, end, RED, op, 7.0)
        self.text(sc, m.x0 + 0.01, m.y0 + 0.03, "the weights' path, in the two directions they "
                  "moved most", op, anchor="lm")
        self.text(sc, pts[0][0] + 0.02, pts[0][1], "start", op, anchor="lm")
        self.text(sc, (m.x0 + m.x1) / 2, 0.90, "the network's %s weights, squeezed onto a "
                  "plane" % "1,450", op)

    def phase_2(self, sc, u, op):
        self._ground(sc, op)
        k = min(u / 0.9, 1.0) * 200
        a = self._trail(sc, 0, op, k, 2.2)
        b = self._trail(sc, 1, op, k, 3.2)
        self.dashes(sc, b, a, BLUE, op * 0.9, 7, 1.8)
        self._readout(sc, op, 200 + int(k), (0, 1))
        self.lines(sc, 0.665, 0.64, [("after every update:", None),
                                     ("average = 97% old average", dark(BLUE)),
                                     ("  + 3% new weights", dark(BLUE)),
                                     "it follows, a few dozen", "updates behind"], op,
                   step=0.048, size=0.032)
        self.text(sc, 0.335, 0.925, "updates 200 to 400, as their shadow on the plane of the "
                  "three final models", op)

    def phase_3(self, sc, u, op):
        self._ground(sc, op)
        k = min(u / 0.9, 1.0) * 200
        mean = self.late[2][min(int(k), 200)]
        for j in range(0, int(k) + 1, 8):
            self.path(sc, [self.late[0][j], mean], GREEN, op * 0.35, None, 1.2)
            self.dot(sc, tuple(self.late[0][j]), RED, op * 0.8, 3.2)
        self._trail(sc, 0, op, k, 2.2)
        self._trail(sc, 2, op, k, 3.2)
        self._readout(sc, op, 200 + int(k), (0, 2))
        self.lines(sc, 0.665, 0.64, [("after every update:", None),
                                     ("mean of all %d sets of" % (int(k) + 1), dark(GREEN)),
                                     ("weights since update 200", dark(GREEN)),
                                     "every visit counts the same,", "old or new"], op,
                   step=0.048, size=0.032)
        self.text(sc, 0.335, 0.925, "updates 200 to 400, as their shadow on the plane of the "
                  "three final models", op)

    def phase_4(self, sc, u, op):
        self._ground(sc, op)
        for kind in range(3):
            self.path(sc, self.late[kind], self.INKS[kind], op * 0.55, None, 1.8)
            self.dot(sc, tuple(self.late[kind][-1]), self.INKS[kind], op, 8.0)
        x = 0.665
        self.text(sc, x, 0.37, "after update 400", op, anchor="lm", strong=True, size=0.036)
        self.text(sc, x, 0.425, "held-out loss (lower is better)", op, anchor="lm")
        worst = float(self.loss[:, -1].max())
        full = self.span(0.17)[0]
        for i in range(3):
            y = 0.52 + i * 0.125
            self.text(sc, x, y - 0.045, "%s: %.1f%% right" % (
                self.NAMES[i], 100 * self.accuracy[i, -1]), op, anchor="lm",
                ink=dark(self.INKS[i]))
            self.hbar(sc, x, y, full * self.loss[i, -1] / worst, self.INKS[i], op,
                      value="%.3f" % self.loss[i, -1], u=u / 0.5, thick=24.0, track=full)
        self.text(sc, 0.335, 0.925, "the three final models lie exactly in this plane",
                  op)

    def phase_5(self, sc, u, op):
        chart = self.chart(((0.11, 0.60), (0.36, 0.82)), self.at, [0.45, 1.25], pad=0.0)
        self.axes(sc, chart, op, "update", "loss on held-out digits")
        k = min(u / 0.7, 1.0)
        upto = self.at[0] + (self.at[-1] - self.at[0]) * k
        for i, first in ((1, int(np.argmax(self.loss[1] < 1.25))),
                         (0, int(np.argmax(self.loss[0] < 1.25))), (2, self._now(200))):
            # the mean is SGD itself until update 200; the first updates are above the chart
            xs, ys = self.at[first:], self.loss[i][first:]
            if upto > xs[0]:
                self.plot(sc, chart, xs, ys, self.INKS[i], op, (upto - xs[0]) / (xs[-1] - xs[0]))
        chart.mark(sc, self.P, 200, GREEN, op)
        self.text(sc, chart.fx(200) + 0.025, chart.y0 + 0.085, "the mean starts here", op,
                  anchor="lm", ink=dark(GREEN, 0.3))
        now = int(k * (len(self.at) - 1))
        x = 0.665
        self.lines(sc, x, 0.37, [("update %d" % self.at[now], None)] + [
            ("%s  %.3f" % (self.NAMES[i], self.loss[i, now]), dark(self.INKS[i]))
            for i in range(3)], op, step=0.052, size=0.034)
        show = op * smoothstep(0.55, 0.75, u)
        self.text(sc, x, 0.62, "curvature at the final weights", show, anchor="lm")
        full = self.span(0.17)[0]
        for i in range(3):
            self.hbar(sc, x, 0.68 + i * 0.062, full * self.curvature[i] / self.curvature.max(),
                      self.INKS[i], show, value="%.2f" % self.curvature[i],
                      u=(u - 0.55) / 0.25, thick=22.0, track=full)


# ====================================================================== 80
def glyphs(ids):
    """Characters as they are shown in a token box."""
    return [("␣" if ch == " " else "↵" if ch == "\n" else ch) for ch in realdata.decode(ids)]


def trailing(a, t, n=50):
    """The median of the ``n`` values of ``a`` before update ``t``: what is usual there."""
    return float(np.median(a[max(t - n, 0):t]))


class LossSpikes(Optim):
    """A loss spike that nothing sets off, seen from inside Adam: on fixed
    text the gradients stay small, Adam's memory of them fades and its step
    per unit of gradient grows, until one larger gradient is turned into a
    leap. Then the same start with a short memory, a larger epsilon and
    clipping, and all three starts of each."""

    TITLE = "A loss spike, inside Adam"
    PHASES = (("the run", 9.0), ("the memory fades", 12.0), ("the spike", 13.0),
              ("three remedies", 12.0), ("three starts", 10.0))
    NAMES = ("plain Adam", "short memory", "larger ε", "clipped")
    INKS = (RED, BLUE, AMBER, GREEN)
    LEAD, AFTER = 12, 48                  # the updates drawn before and after the peak

    def setup(self):
        s = self.study
        p = [panel.data for panel in s.panels]
        self.loss = np.asarray(p[0]["ys"], float)                     # setting, update
        self.norm = np.asarray(p[1]["ys"], float)
        self.ratio = np.asarray(p[2]["ys"], float)
        m = self.m = list(s.metadata["seeds"]).index(s.metadata["start_drawn"])
        self.moment = np.asarray(p[2]["moment"], float)[:, m]
        self.step = np.asarray(p[2]["step"], float)[:, m]
        self.worst = np.asarray(p[3]["each"], float)                  # setting, start
        self.count = np.asarray(p[3]["count"], int)
        self.tail = np.asarray(p[3]["tail"], float)
        self.held = np.asarray(p[3]["held"], float)
        self.at = int(s.metadata["spike_drawn_at"])
        self.q0, self.q1 = O.SPIKE_FROM, self.at - self.LEAD
        self.steps = np.arange(self.loss.shape[1])
        text = O.spike_text()[0]
        self.rows = [glyphs(r) for r in text[:3]]
        # what is usual at the start and at the end of the quiet stretch (plain Adam)
        self.usual = [dict(norm=trailing(self.norm[0], t), moment=float(self.moment[0, t]),
                           ratio=trailing(self.ratio[0], t), step=trailing(self.step[0], t))
                      for t in (self.q0, self.q1)]
        a, b = self.usual
        self.fade = a["moment"] / b["moment"]
        self.grow = b["ratio"] / a["ratio"]
        w = slice(self.q1, self.at + 1)
        self.kick = float(self.norm[0, w].max()) / b["norm"]
        self.leap = float(self.step[0, w].max()) / b["step"]
        self.spiked = (self.count > 0).sum(1)

    def subtitle_for(self, k):
        L = self.loss
        return [
            "a small transformer learns 64 windows of Shakespeare with the Adam optimiser, the "
            "same 64 at every update: in %d updates its loss falls from %.2f to %.3f, and "
            "nothing about the data ever changes" % (self.q0, L[0, 0], L[0, self.q0]),
            "the gradients stay small, so Adam's memory of them fades: from update %d to %s "
            "it shrinks %.1f times, and as Adam divides by it, the step it takes per unit of "
            "gradient grows %.1f times" % (self.q0, f"{self.q1:,}", self.fade, self.grow),
            "around update %s the gradient grows to %.0f times the usual, faster than Adam's "
            "long memory can follow: its step reaches %.0f times the usual and the loss leaps "
            "from %.3f to %.2f, on unchanged data" % (f"{self.at:,}", self.kick,
                                                      self.leap, L[0, self.q1], L[0, self.at]),
            "the same start with three remedies: a memory of 10 updates grows as soon as the "
            "gradients do and brakes (the loss never passes %.1f times its lowest); a larger ε "
            "(%.0f times) and clipping (%.0f times) still spike" % tuple(
                self.worst[1:, self.m]),
            "over three starts plain Adam spikes in %d, the short memory in none, the larger ε "
            "in %d and clipping in %d; the short memory's price is a slightly higher loss at "
            "the end, %.3f against %.3f" % (self.spiked[0], self.spiked[2], self.spiked[3],
                                           self.tail[1].mean(), self.tail[0].mean()),
        ][k]

    # ------------------------------------------------------------------ the text
    def _snippet(self, sc, row, fx0, fy, op, w=26.0, h=32.0, target=True):
        chars = self.rows[row]
        step = self.frac(w + 3.0)[0]
        for j, ch in enumerate(chars[:12]):
            self.box(sc, fx0 + (j + 0.5) * step, fy, w, h, soft(TEAL), op, ch,
                     ink=dark(TEAL, 0.4), border=TEAL, size=0.034)
        if target:
            x = fx0 + 12.9 * step
            self.arrow(sc, (fx0 + 12.05 * step, fy), (x - step * 0.55, fy), AXIS, op, 1.0, 1.8)
            self.box(sc, x, fy, w, h, TEAL, op, chars[12], ink=(1, 1, 1), size=0.034)

    def phase_1(self, sc, u, op):
        for row in range(3):
            self._snippet(sc, row, 0.05, 0.40 + row * 0.085, op)
        self.text(sc, 0.05, 0.335, "64 windows like these, the same every update", op,
                  anchor="lm", strong=True, size=0.034)
        self.lines(sc, 0.05, 0.67, ["the model reads 12 characters and", "is scored on the next "
                                    "one (dark box)", "", "Adam turns each gradient into a step"],
                   op, step=0.046, size=0.032)
        n = self.q0 + 1
        chart = self.chart(((0.59, 0.94), (0.37, 0.80)), self.steps[:n], self.loss[0, :n])
        self.axes(sc, chart, op, "update", "loss on these windows")
        k = min(u / 0.8, 1.0)
        self.plot(sc, chart, self.steps[:n], self.loss[0, :n], RED, op, k)
        now = int(k * (n - 1))
        self.text(sc, 0.93, 0.42, "update %d   loss %.2f" % (now, self.loss[0, now]), op,
                  anchor="rm", strong=True, size=0.036)

    def _three(self, sc, op, names, values, before, scales, ink, times, note):
        """Adam's three quantities as bars, each against its value ``before``."""
        full = self.span(0.27)[0]
        for row in range(3):
            y = 0.44 + row * 0.135
            self.text(sc, 0.05, y - 0.05, names[row], op, anchor="lm", size=0.032)
            label = ("%.2f" if values[row] < 10 else "%.1f") % values[row]
            if times:
                label += "   × %.1f" % (values[row] / before[row])
            self.hbar(sc, 0.05, y, full * min(values[row] / scales[row], 1.0), ink, op,
                      value=label, thick=26.0, track=full)
            x = 0.05 + self.frac(full * min(before[row] / scales[row], 1.0))[0]
            self.dashes(sc, (x, y + 0.03), (x, y + 0.05), AXIS, op, 1, 2.4)
        self.text(sc, 0.05, 0.82, "step = recent gradients ÷ memory", op, anchor="lm",
                  strong=True, size=0.032)
        self.text(sc, 0.05, 0.865, note, op, anchor="lm")

    def phase_2(self, sc, u, op):
        t = int(self.q0 + (self.q1 - self.q0) * min(u / 0.9, 1.0))
        a = self.usual[0]
        values = (trailing(self.norm[0], t), float(self.moment[0, t]), trailing(self.ratio[0], t))
        before = (a["norm"], a["moment"], a["ratio"])
        self._three(sc, op, ("gradient (median of 50 updates)", "Adam's memory of past "
                             "gradients", "step taken per unit of gradient (median of 50)"),
                    values, before,
                    (1.0, 1.1 * a["moment"], 1.5 * self.usual[1]["ratio"]), RED, t > self.q0,
                    "× : against update %d, the tick under each bar" % self.q0)
        xs = self.steps[self.q0:self.q1 + 1]
        chart = self.chart(((0.60, 0.94), (0.44, 0.78)), xs, [0.0, 0.4], pad=0.0)
        self.axes(sc, chart, op, "update", "loss")
        self.plot(sc, chart, xs, self.loss[0, self.q0:self.q1 + 1], RED, op,
                  (t - self.q0) / (self.q1 - self.q0))
        self.text(sc, 0.94, 0.36, "update %s   loss %.3f" % (f"{t:,}", self.loss[0, t]),
                  op, anchor="rm", strong=True, size=0.036)
        self.text(sc, 0.94, 0.925, "plain Adam: its memory averages about 1,000 updates", op,
                  anchor="rm", ink=dark(RED))

    def phase_3(self, sc, u, op):
        lo, hi = self.q1, self.at + self.AFTER
        t = int(lo + (hi - lo) * min(u / 0.92, 1.0))
        b = self.usual[1]
        values = (float(self.norm[0, t]), float(self.moment[0, t]), float(self.step[0, t]))
        before = (b["norm"], b["moment"], b["step"])
        w = slice(lo, hi + 1)
        scales = (float(self.norm[0, w].max()), float(self.moment[0, w].max()) * 1.05,
                  float(self.step[0, w].max()))
        self._three(sc, op, ("gradient: how hard the loss pulls", "Adam's memory of past "
                             "gradients", "the step Adam takes"), values, before, scales, RED,
                    True, "× : against the usual before it, the tick under each bar")
        xs = self.steps[lo - 8:hi + 1]
        ys = self.loss[0, lo - 8:hi + 1]
        chart = self.chart(((0.60, 0.94), (0.44, 0.78)), xs, [0.0, 1.05 * float(ys.max())],
                           pad=0.0)
        self.axes(sc, chart, op, "update", "loss")
        self.plot(sc, chart, xs, ys, RED, op, (t - xs[0]) / (xs[-1] - xs[0]))
        self.text(sc, 0.94, 0.36, "update %s   loss %.2f" % (f"{t:,}", self.loss[0, t]),
                  op, anchor="rm", strong=True, size=0.036)
        self.text(sc, 0.94, 0.925, "%d updates after the peak, Adam's steps are still %.0f "
                  "times the usual" % (self.AFTER, self.step[0, hi] / b["step"]),
                  op * smoothstep(0.85, 0.95, u), anchor="rm", ink=dark(RED))

    def _strip(self, sc, i, box, op, u, values=None):
        """Setting ``i``'s loss over the whole run (highest of each 15 updates)."""
        ys = self.loss[i] if values is None else values
        top = ys[:len(ys) // 15 * 15].reshape(-1, 15).max(1)
        xs = np.arange(len(top)) * 15
        chart = self.chart(box, xs, [0.08, 12.0], ylog=True)
        self.axes(sc, chart, op, "", "",
                  xticks=[(v, f"{v:,}") for v in (0, 1000, 2000, 3000)],
                  yticks=[(v, t) for v, t in ((0.1, "0.1"), (1.0, "1"), (10.0, "10"))])
        self.plot(sc, chart, xs, top, self.INKS[i], op, u, width=2.0, dot=False)
        return chart

    def phase_4(self, sc, u, op):
        detail = ("memory 1,000 updates", "memory 10 updates", "ε 0.001 in place of 0.00000001",
                  "gradients cut to length 1")
        for i in range(4):
            col, row = i % 2, i // 2
            x0, y0 = 0.10 + col * 0.45, 0.385 + row * 0.27
            show = op * smoothstep(0.0, 0.08, u - 0.16 * i)
            self._strip(sc, i, ((x0, x0 + 0.34), (y0, y0 + 0.15)), show,
                        min(max(u - 0.16 * i, 0.0) / 0.3, 1.0))
            self.text(sc, x0, y0 - 0.045, "%s: %s" % (self.NAMES[i], detail[i]), show,
                      anchor="lm", ink=dark(self.INKS[i], 0.25), size=0.032)
            self.text(sc, x0 + 0.34, y0 + 0.02, "worst: × %s" % (
                "%.1f" % self.worst[i, self.m] if self.worst[i, self.m] < 10 else
                "%.0f" % self.worst[i, self.m]), show * smoothstep(0.3, 0.4, u - 0.16 * i),
                anchor="rm", strong=True)
        self.text(sc, 0.5, 0.925, "loss at each update (up: 10 times higher per line);   worst: "
                  "the most it rises above its lowest so far", op)

    def phase_5(self, sc, u, op):
        self.text(sc, 0.08, 0.35, "the most the loss rises above its lowest so far, after "
                  "update %d: three starts of each" % self.q0, op, anchor="lm", strong=True,
                  size=0.034)
        full = self.span(0.36)[0]
        top = np.log10(300.0)
        x = 0.27
        for i in range(4):
            y0 = 0.435 + i * 0.118
            self.text(sc, x - 0.02, y0 + 0.027, self.NAMES[i], op, anchor="rm",
                      ink=dark(self.INKS[i], 0.25), size=0.034)
            for j in range(3):
                v = self.worst[i, j]
                y = y0 + j * 0.027
                self.hbar(sc, x, y, full * np.log10(max(v, 1.0)) / top + 2.0, self.INKS[i], op,
                          u=(u - 0.05 * i) / 0.4, thick=11.0, track=full)
            self.text(sc, x + 0.375, y0 + 0.027, ",  ".join(
                "× %.1f" % v if v < 10 else "× %.0f" % v for v in self.worst[i]),
                op * smoothstep(0.3, 0.45, u - 0.05 * i), anchor="lm", strong=True)
        for v in (1, 10, 100):
            fx = x + self.frac(full * np.log10(v) / top)[0]
            self.text(sc, fx, 0.40, "× %d" % v, op)
        show = op * smoothstep(0.5, 0.7, u)
        self.text(sc, 0.82, 0.40, "loss at the end*", show, anchor="lm")
        for i in range(4):
            self.text(sc, 0.82, 0.462 + i * 0.118, "%.3f" % self.tail[i].mean(), show,
                      anchor="lm", ink=dark(self.INKS[i], 0.25), size=0.034)
        self.text(sc, 0.94, 0.925, "a spike: more than × 2.    * mean of the last 500 updates; "
                  "on unseen text %.0f (the 64 windows are memorised)" % self.held.mean(),
                  show, anchor="rm")


ANIMATIONS = {
    "newton-vs-gradient-descent": NewtonVsGradientDescent,
    "edge-of-stability": EdgeOfStability,
    "batch-size-and-noise": BatchSizeAndNoise,
    "weight-averaging": WeightAveraging,
    "second-order-preconditioners": SecondOrderPreconditioners,
    "loss-spikes": LossSpikes,
}

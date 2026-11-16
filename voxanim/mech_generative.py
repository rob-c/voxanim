"""Generative models: animations 59-66 of the "next 100" (see :mod:`voxanim.mech`).

Each draws its mechanism first (points moving on a plane, a grid bending, a
noise ladder walked down, a pixel grid filling in) and the measurement after
it. The numbers are the study's: NumPy models trained on MNIST, in
:mod:`voxanim.study_generative` for the five experiments that were redesigned
(GAN, flow, guidance, codebook, pixels; they save everything drawn here) and
in :mod:`voxanim.next_vision` for the other three, whose sampler chains are
replayed here from the same seeds and saved weights. The tests hold what is
drawn to the studies' results.
"""

from __future__ import annotations

import itertools
from contextlib import contextmanager
from functools import cache
from unittest import mock

import numpy as np

from . import next_vision as nv
from . import next_vision_models as vm
from . import study_generative as sg
from . import toy2d as toy
from .cache import cached_npz
from .diagram import draw_image
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIM,
    GREEN,
    PAPER,
    RED,
    TEAL,
    TRACK,
    VIOLET,
    Mechanism,
    dark,
    heat,
    soft,
)

VERSION = 1                        # of the replays cached by this module
WHITE = (1.0, 1.0, 1.0)
INK = (0.16, 0.17, 0.19)


# ---------------------------------------------------------------------- the studies' arrays
@cache
def raw(name):
    """Every array study ``name`` saved (its samples, weights and data): the
    panels show only part of them. Read back from the study's own cache."""
    got = {}
    saved = nv.cached

    def keep(key, build, **settings):
        values = saved(key, build, **settings)
        got.update(values)
        return values
    with mock.patch.object(nv, "cached", keep):
        nv._BUILDERS[name]()
    return got


def weights(arrays, prefix="weights_"):
    """The parameters saved under ``prefix``, by their own names."""
    return {k[len(prefix):]: np.asarray(v) for k, v in arrays.items() if k.startswith(prefix)}


def replay(name, build, **settings):
    """A replay of a study's run, cached under this module's version and the study's."""
    return cached_npz("mech_generative_" + name,
                      dict(version=VERSION, study=nv.VERSION, seed=nv.SEED, **settings), build)


def mnist7(split="test", n=256):
    """The first ``n`` MNIST images of ``split`` at the studies' 7 x 7 (4 x 4 means)."""
    x, _y, xt, _yt = nv.mnist_data()
    return vm.downsample((xt if split == "test" else x)[:n])


# ---------------------------------------------------------------------- shared drawing
class Plane:
    """A square window on a 2-D space: data ``lim = (x0, x1, y0, y1)`` drawn
    ``side`` px wide, centred at layout ``(fx, fy)``, y upwards."""

    def __init__(self, m, fx, fy, side, lim):
        self.m, self.fx, self.fy, self.side, self.lim = m, fx, fy, side, lim
        self.dw, self.dh = m.frac(side, side)

    def at(self, p):
        x0, x1, y0, y1 = self.lim
        return (self.fx + ((p[0] - x0) / (x1 - x0) - 0.5) * self.dw,
                self.fy - ((p[1] - y0) / (y1 - y0) - 0.5) * self.dh)

    def inside(self, pts, pad=0.0):
        x0, x1, y0, y1 = self.lim
        p = np.asarray(pts, float)
        return ((p[..., 0] >= x0 + pad) & (p[..., 0] <= x1 - pad)
                & (p[..., 1] >= y0 + pad) & (p[..., 1] <= y1 - pad))

    def frame(self, sc, op, label=None, ink=None, below=None):
        m = self.m
        m.rect(sc, m.P(self.fx, self.fy), self.side + 8, self.side + 8, PAPER, op, border=AXIS,
               border_w=1.2, radius=5, lift=0.1)
        if label:
            m.text(sc, self.fx, self.fy - self.dh / 2 - 0.038, label, op, size=0.034,
                   strong=ink is None, ink=ink)
        if below:
            m.text(sc, self.fx, self.fy + self.dh / 2 + 0.038, below, op)

    def image(self, sc, rgb, op):
        """A ``(rows, cols, 3)`` picture filling the window (row 0 at the top)."""
        draw_image(self.m, sc, self.m.P(self.fx, self.fy), np.asarray(rgb, np.float32),
                   self.side, self.side, op, grid=0)

    def dots(self, sc, pts, ink, op, r=2.7, glow=0.0):
        m = self.m
        pts = np.asarray(pts, float)
        for p in pts[self.inside(pts)]:
            m.disk(sc, m.P(*self.at(p)), r, ink, op, glow=glow, lift=1.0, outline=0.12)

    def path(self, sc, pts, ink, op, width=1.5):
        """A polyline through ``pts``, drawn only where it is inside the window."""
        m = self.m
        pts = np.asarray(pts, float)
        ok = self.inside(pts)
        for i in range(len(pts) - 1):
            if ok[i] and ok[i + 1]:
                sc.line(m.P(*self.at(pts[i]), lift=0.7), m.P(*self.at(pts[i + 1]), lift=0.7),
                        width=width, color=ink, opacity=op, bias=0.3)

    def vector(self, sc, p, v, ink, op, width=1.6):
        """An arrow from data point ``p`` along data vector ``v``."""
        q = np.asarray(p, float) + np.asarray(v, float)
        if self.inside(p) and self.inside(q) and op > 0.01:
            m = self.m
            a, b = m.P(*self.at(p), lift=0.7), m.P(*self.at(q), lift=0.7)
            sc.line(a, b, width=width, color=ink, opacity=op, bias=0.3)
            m._head(sc, a, b, ink, op, width * 0.6)


class Generative(Mechanism):
    """What the eight animations share: a legend key, upright bars, rows of images."""

    _over = 0.0

    def P(self, fx, fy, lift=0.0):
        return super().P(fx, fy, lift + self._over)

    @contextmanager
    def over(self, lift=0.8):
        """Draw what is inside in front of a :class:`Plane`'s paper (arcs and
        arrows are otherwise drawn on the floor, behind it)."""
        self._over, old = lift, self._over
        try:
            yield
        finally:
            self._over = old

    def key(self, sc, fx, fy, ink, label, op, r=5.0, size=None):
        self.disk(sc, self.P(fx, fy), r, ink, op, outline=0.12)
        self.text(sc, fx + 0.016, fy, label, op, anchor="lm", **({"size": size} if size else {}))

    def vbar(self, sc, fx, base, w, h, ink, op, u=1.0):
        """An upright bar ``w`` px wide standing on layout y ``base``, ``h`` px
        tall at ``u`` = 1."""
        h = max(float(h) * smoothstep(0.0, 1.0, u), 0.6)
        self.rect(sc, self.shift(self.P(fx, base), 0, -h / 2), w, h, ink, op,
                  radius=min(3.0, h / 2))

    def pixels(self, sc, rgb, fx, fy, size, op, frame=AXIS, ring=(), ring_ink=AMBER):
        """A small image as a grid of square cells from the texture ``rgb``
        ``(rows, cols, 3)``, ``size`` px wide; ``ring``: ``(row, col)`` cells
        outlined. Returns ``cell(row, col) -> (fx, fy)``."""
        rgb = np.asarray(rgb, np.float32)
        rows, cols = rgb.shape[:2]
        c = self.P(float(fx), float(fy))
        if frame is not None:
            self.rect(sc, c, size + 10, size + 10, frame, op, radius=5, lift=0.1)
        draw_image(self, sc, c, rgb, size, size, op, grid=0.16)
        dw, dh = self.frac(size, size)

        def cell(r, k):
            return fx - dw / 2 + (k + 0.5) * dw / cols, fy - dh / 2 + (r + 0.5) * dh / rows
        for r, k in ring:
            self.rect(sc, self.P(*cell(r, k)), size / cols + 2, size / rows + 2, ring_ink, op,
                      border=ring_ink, border_w=2.6, radius=2, lift=0.8, fill_op=0.0)
        return cell

    def between(self, snaps, x):
        """``snaps`` (a recorded sequence) at fractional index ``x``."""
        x = float(np.clip(x, 0, len(snaps) - 1))
        i = min(int(x), len(snaps) - 2)
        f = x - i
        return snaps[i] * (1 - f) + snaps[i + 1] * f


def shade(v, ink=INK):
    """Pixel values in [0, 1] as a paper-to-``ink`` texture."""
    v = np.clip(np.asarray(v, float), 0, 1)[..., None]
    return PAPER * (1 - v) + np.asarray(ink) * v


def seg(u, a, b):
    """0..1 as ``u`` crosses ``a..b``."""
    return float(np.clip((u - a) / (b - a), 0.0, 1.0))


# ====================================================================== 59 gan-training
DIGIT_INKS = {0: BLUE, 1: TEAL, 4: VIOLET}      # the three real clusters


class GanTraining(Generative):
    """A generator and a discriminator on three clusters of real digits (MNIST
    0s, 1s and 4s at two PCA coordinates): the game, a plain GAN dropping and
    refinding a digit, the same with a discriminator that sees crowding, and
    how often five runs of each cover all three."""

    TITLE = "GAN training"
    PHASES = (("the game", 11.0), ("plain GAN", 15.0), ("sees crowding", 13.0),
              ("coverage", 13.0))
    LIM = (-3.0, 3.0, -3.0, 3.0)
    PLANE = (0.215, 0.615, 296.0)
    DRAWN = 240                        # of each run's 500 fixed-noise samples

    def setup(self):
        d = sg.arrays("gan-training")
        s = self.study
        self.real = np.asarray(s.panels[0].data["xy"], float)
        self.labels = np.asarray(s.panels[0].data["classes"])
        self.pixels = d["pixels"]
        self.steps = np.asarray(d["steps"])
        self.updates = int(self.steps[-1])
        self.fakes = d["fakes"]                       # kind, seed, checkpoint, point, xy
        self.surfaces = d["surfaces"]
        self.shares = np.asarray(d["shares"], float)
        self.covered, self.all_three, self.fewest = sg.gan_summary(d)
        self.late = self.covered.shape[-1] // 2
        self.seeds = len(sg.GAN_SEEDS)
        self.centres = {k: self.real[self.labels == k].mean(0) for k in sg.DIGITS}
        # three real digits to show as the points they are: one of each cluster
        self.shown = [int(np.flatnonzero(self.labels == k)[0]) for k in sg.DIGITS]

    def subtitle_for(self, k):
        def pct(v):
            return "%.0f%%" % (100 * v)

        return [
            "every MNIST 0, 1 and 4 is one point on this plane, in three clusters; a "
            "generator turns noise into points and a discriminator scores each point as "
            "real or fake",
            "plain GAN, %s updates: the generated points chase wherever the discriminator "
            "scores as real and keep abandoning a digit; in the second half all three are "
            "covered at only %s of the checkpoints" % (
                f"{self.updates:,}", pct(self.all_three[0, 0])),
            "a second GAN whose discriminator also sees how crowded a batch is: piling "
            "onto one cluster now gives the fakes away, and all three digits stay covered "
            "at %s of the late checkpoints" % pct(self.all_three[1, 0]),
            "five runs of each: the plain GAN holds all three digits at %s to %s of its "
            "late checkpoints, the crowd-aware one at %s to %s; at the very last update "
            "all ten happened to cover all three" % (
                pct(self.all_three[0].min()), pct(self.all_three[0].max()),
                pct(self.all_three[1].min()), pct(self.all_three[1].max())),
        ][k]

    # ------------------------------------------------------------------ drawing
    def _plane(self, at=None):
        fx, fy, side = at or self.PLANE
        return Plane(self, fx, fy, side, self.LIM)

    def _surface(self, pl, sc, score, op):
        """The discriminator's score over the plane: blue where it says real."""
        q = np.clip((np.asarray(score)[::-1] - 0.5) * 2.0, -1, 1)
        pl.image(sc, heat(q * 0.45, hot=BLUE, cold=RED), op)

    def _real(self, pl, sc, op, r=2.3, names=True):
        for digit, ink in DIGIT_INKS.items():
            pl.dots(sc, self.real[self.labels == digit], ink, op, r=r)
        if names:
            for digit, (dx, dy) in zip(sg.DIGITS, ((-0.9, 1.25), (0.75, 1.3), (-1.3, -0.75))):
                at = pl.at(self.centres[digit] + np.array([dx, dy]))
                self.text(sc, at[0], at[1], str(digit), op, size=0.05,
                          ink=dark(DIGIT_INKS[digit]))

    def phase_1(self, sc, u, op):
        pl = self._plane()
        pl.frame(sc, op, "the plane of digits")
        self._real(pl, sc, op * smoothstep(0.0, 0.25, u), r=2.6)
        fakes = self.fakes[0, 0, 0, :self.DRAWN]
        made = smoothstep(0.42, 0.62, u)
        pl.dots(sc, fakes, RED, op * made)
        x = 0.47
        # the generator
        y = 0.40
        self.box(sc, x + 0.26, y, 84, 44, PAPER, op, "noise", border=AXIS, size=0.034)
        self.box(sc, x + 0.065, y, 124, 44, RED, op, "generator", ink=WHITE, size=0.034)
        self.arrow(sc, (x + 0.212, y), (x + 0.135, y), RED, op, width=2.2)
        self.text(sc, x + 0.32, y, "in, fake points out", op, anchor="lm", ink=dark(RED),
                  size=0.034)
        w = seg(u, 0.38, 0.6)
        centre = pl.at(fakes.mean(0))
        with self.over():
            self.arc(sc, (x - 0.005, y), centre, RED, op * 0.8, w, bend=0.03, width=1.8)
        self.pulse(sc, (x - 0.005, y), centre, w, RED, op, bend=0.03)
        # real digits and the points they are
        y = 0.615
        for i, (k, digit) in enumerate(zip(self.shown, sg.DIGITS)):
            ink = DIGIT_INKS[digit]
            cx = x + 0.035 + i * 0.085
            self.digit(sc, self.pixels[k], cx, y, 52, op, frame=ink)
            a, b = (cx, y - 0.062), pl.at(self.real[k])
            w = seg(u, 0.05 + 0.08 * i, 0.3 + 0.08 * i)
            with self.over():
                self.arc(sc, a, b, ink, op * 0.8, w, bend=0.05, width=1.8)
            self.pulse(sc, a, b, w, ink, op, bend=0.05)
            self.disk(sc, self.P(*b), 5.5, ink, op * smoothstep(0.9, 1.0, w), glow=0.3,
                      ring=WHITE, ring_w=1.5)
        self.lines(sc, x + 0.26, y - 0.022, [("real: MNIST 0s, 1s and 4s", dark(BLUE)),
                                             "each one a point"], op, step=0.046, size=0.034)
        # the discriminator
        y = 0.815
        self.box(sc, x + 0.09, y, 152, 44, AMBER, op, "discriminator", ink=WHITE, size=0.034)
        self.text(sc, x + 0.185, y, "real or fake?", op, anchor="lm", size=0.034, strong=True)
        w = seg(u, 0.64, 0.84)
        for p, ink in ((self.real[self.shown[2]], DIGIT_INKS[sg.DIGITS[2]]),
                       (fakes.mean(0), RED)):
            a = pl.at(p)
            with self.over():
                self.arc(sc, a, (x + 0.01, y), ink, op * 0.8, w, bend=-0.07, width=1.8)
            self.pulse(sc, a, (x + 0.01, y), w, ink, op, bend=-0.07)
        self.lines(sc, x, 0.895, ["the generator never sees a digit:",
                                  "it learns only from the discriminator's score"],
                   op * smoothstep(0.82, 0.94, u), step=0.044)

    def _training(self, sc, u, op, kind, ink, name):
        pl = self._plane()
        prog = seg(u, 0.04, 0.9)
        at = prog * (len(self.steps) - 1)
        k = int(round(at))
        fakes = self.between(self.fakes[kind, 0, :, :self.DRAWN], at)
        if kind == 0:
            pl.frame(sc, op, "update %d" % self.steps[k],
                     below="shading: the discriminator's score")
            self._surface(pl, sc, self.between(self.surfaces[kind], at), op)
        else:
            pl.frame(sc, op, "update %d" % self.steps[k])
        self._real(pl, sc, op * 0.5, r=2.1)
        pl.dots(sc, fakes, ink, op, r=3.0)
        x = 0.47
        self.key(sc, x, 0.33, ink, "generated by the %s" % name, op)
        # where the generated points are landing, by digit
        share = self.between(self.shares[kind, 0], at)
        self.text(sc, x, 0.40, "share of generated points on each digit's cluster", op,
                  anchor="lm", strong=True)
        full = self.span(0.30)[0]
        for row, digit in enumerate(sg.DIGITS):
            y = 0.455 + row * 0.055
            lost = share[row] < sg.GAN_COVERED
            self.hbar(sc, x + 0.06, y, max(full * share[row] / 0.7, 1.0),
                      RED if lost else DIGIT_INKS[digit], op, "a %d" % digit,
                      "%d%%" % (100 * share[row]) + ("  dropped" if lost else ""),
                      thick=20.0, track=full)
        # digits covered, through training
        chart = self.chart(((x + 0.03, 0.945), (0.66, 0.85)), [0, self.updates], [0, 3],
                           pad=0.08)
        self.axes(sc, chart, op, "update", "digits covered",
                  yticks=[(v, str(v)) for v in range(4)])
        self.plot(sc, chart, self.steps, self.covered[kind, 0], ink, op, prog, width=2.2)

    def phase_2(self, sc, u, op):
        self._training(sc, u, op, 0, RED, "plain GAN")

    def phase_3(self, sc, u, op):
        self._training(sc, u, op, 1, GREEN, "crowd-aware GAN")

    ALL_THREE = soft(BLUE, 0.6)

    def _strips(self, sc, kind, fx, fy, w, h, op, reveal):
        """One row per run, one cell per checkpoint: how many digits it covered."""
        rows = self.covered[kind]
        tex = np.empty(rows.shape + (3,), np.float32)
        tex[rows == 3] = self.ALL_THREE
        tex[rows == 2] = np.asarray(AMBER)
        tex[rows <= 1] = np.asarray(INK)
        shown = int(round(reveal * rows.shape[1]))
        tex[:, shown:] = PAPER * 0.92
        c = self.P(fx, fy)
        self.rect(sc, c, w + 6, h + 6, PAPER, op, border=AXIS, border_w=1.0, radius=3, lift=0.1)
        draw_image(self, sc, c, tex, w, h, op, grid=0)
        dw, dh = self.frac(w, h)
        for row in range(len(rows)):
            y = fy - dh / 2 + (row + 0.5) * dh / len(rows)
            self.text(sc, fx + dw / 2 + 0.012, y, "%.0f%%" % (100 * self.all_three[kind, row]),
                      op * smoothstep(0.85, 1.0, reveal), anchor="lm", size=0.031, strong=True)
        return dw, dh

    def phase_4(self, sc, u, op):
        x, w, h = 0.36, 480.0, 96.0
        names = ("plain GAN", "crowd-aware GAN")
        for kind, fy in enumerate((0.45, 0.72)):
            dw, dh = self._strips(sc, kind, x, fy, w, h, op, seg(u, 0.05 + 0.3 * kind,
                                                                  0.4 + 0.3 * kind))
            self.text(sc, x - dw / 2, fy - dh / 2 - 0.035, "%s: five runs, update 0 to %d" % (
                names[kind], self.updates), op, anchor="lm", size=0.034,
                ink=dark((RED, GREEN)[kind]))
            with self.over(1.2):                       # where the counted half begins
                self.polyline(sc, [self.P(x, fy - dh / 2 - 0.008), self.P(x, fy + dh / 2 + 0.008)],
                              2.0, INK, op)
            self.text(sc, x + dw / 2 + 0.012, fy - dh / 2 - 0.035, "late", op, anchor="lm")
        self.text(sc, x + 0.008, 0.85, "counted from here: the second half", op, anchor="lm")
        y = 0.915
        for i, (label, fill) in enumerate((("all three digits covered", self.ALL_THREE),
                                           ("two", AMBER), ("one or none", INK))):
            fx = 0.12 + (0.0, 0.27, 0.36)[i]
            self.rect(sc, self.P(fx, y), 14, 14, fill, op, border=AXIS, radius=2)
            self.text(sc, fx + 0.014, y, label, op, anchor="lm")
        mean = self.all_three.mean(1)
        for kind, (fy, name, ink, when) in enumerate(((0.39, "plain", RED, 0.4),
                                                      (0.66, "crowd-aware", GREEN, 0.7))):
            self.lines(sc, 0.72, fy, [
                (name, dark(ink)), "all three at %.0f%%" % (100 * mean[kind]),
                "of late checkpoints", "fewest at once: %d" % self.fewest[kind].min()],
                op * smoothstep(when, when + 0.1, u), step=0.044, size=0.034)


# ====================================================================== 60 normalizing-flows
def flow_states(p, z):
    """``z`` on its way through the fitted flow's inverse, one coupling layer
    at a time: seven arrays, from the Gaussian side to the data side."""
    layers = len(p) // 4
    return np.array([np.asarray(z, float)] + [toy.flow_inverse(p, z, until=k)
                                               for k in range(layers - 1, -1, -1)])


def coupling_terms(p, layer, x):
    """What inverse coupling ``layer`` does to points ``x``: the coordinate it
    reads, the one it moves, and the shift ``t`` and stretch ``exp(-s)`` it
    applies (``new = (old - t) * stretch``)."""
    i, j = layer % 2, 1 - layer % 2
    h = np.tanh(x[:, i:i + 1] @ p["W%d" % layer] + p["b%d" % layer])
    st = h @ p["V%d" % layer] + p["c%d" % layer]
    return i, j, st[:, 1], np.exp(-0.7 * np.tanh(st[:, 0]))


class NormalizingFlows(Generative):
    """Six affine coupling layers between a Gaussian and three clusters of
    real digits (MNIST 0s, 1s and 4s at two PCA coordinates): one layer's
    move, all six bending the grid into the three arms, what the stretching
    does to density, and the exact held-out likelihood against a Gaussian's."""

    TITLE = "Normalizing flows"
    PHASES = (("two spaces", 9.0), ("one layer", 12.0), ("six layers", 12.0),
              ("stretch and density", 12.0), ("exact likelihood", 10.0))
    LIM = (-3.6, 3.6, -3.6, 3.6)

    def setup(self):
        d = sg.arrays("normalizing-flows")
        s = self.study
        self.p = weights(d)
        self.x = np.asarray(s.panels[0].data["xy"], float)
        self.labels = np.asarray(s.panels[0].data["classes"])
        self.z = np.asarray(d["z"], float)
        self.states = flow_states(self.p, self.z)
        self.mesh = np.array([flow_states(self.p, line) for line in sg.grid_lines()])
        self.encoded, self.logdet = np.asarray(d["encoded"]), np.asarray(d["logdet"])
        self.loss = np.asarray(s.panels[2].data["ys"][0], float)
        self.held = np.asarray(d["nll"], float)[:, 1]          # held out, one per seed
        self.error = float(d["error"])
        self.plain = float(d["bell"][1])                       # the bell with no flow, held out
        self.gaussian = float(d["fitted_gaussian"])            # the best single Gaussian
        # the first layer undone, on four samples that it moves far, spread over the plane
        self.layer = len(self.p) // 4 - 1
        _i, _j, self.shifts, self.stretches = coupling_terms(self.p, self.layer, self.z)
        move = np.abs(self.states[1] - self.states[0]).sum(1)
        inside = (np.abs(self.states[0]) < 2.2).all(1) & (np.abs(self.states[1]) < 3.2).all(1)
        order = np.argsort(-move * inside)
        picks = [int(order[0])]
        for k in order[1:]:
            if len(picks) == 4:
                break
            if all(abs(self.z[k, 0] - self.z[q, 0]) > 0.5 for q in picks):
                picks.append(int(k))
        self.picks = picks
        # one digit followed back to its partner: the one the flow stretches most
        self.one = int(np.argmax(self.logdet))
        zz = self.encoded[self.one]
        self.one_bell = float(-0.5 * (zz ** 2).sum() - np.log(2 * np.pi))

    def subtitle_for(self, k):
        one = self.picks[0]
        return [
            "a flow is a map you can run both ways: every point of a simple bell-shaped cloud "
            "has exactly one partner among these points, each a real MNIST 0, 1 or 4, and back",
            "one layer reads a point's height and uses it to shift and stretch the point "
            "sideways: here a shift of %.2f and a stretch of %.2f; knowing the height, it can "
            "be undone" % (self.shifts[one], self.stretches[one]),
            "six such layers, taking turns between sideways and up-down, bend the bell "
            "cloud and its grid into the three arms of the digit points",
            "where the flow stretches space the density thins by exactly that factor: a "
            "digit's log density is its partner's under the bell, %.2f, plus the log stretch, "
            "%.2f" % (self.one_bell, self.logdet[self.one]),
            "so the likelihood is exact: %.2f nats per held-out digit (three seeds: %.2f to "
            "%.2f) against %.2f for the best single Gaussian, which cannot have three arms; "
            "there and back again errs by %.0e" % (
                self.held.mean(), self.held.min(), self.held.max(), self.gaussian, self.error),
        ][k]

    # ------------------------------------------------------------------ drawing
    def _grid(self, pl, sc, lines, op, ink=AXIS):
        for line in lines:
            pl.path(sc, line, ink, op * 0.75, width=1.1)

    def phase_1(self, sc, u, op):
        left = Plane(self, 0.215, 0.615, 280.0, self.LIM)
        right = Plane(self, 0.785, 0.615, 280.0, self.LIM)
        left.frame(sc, op, "a simple bell-shaped cloud", ink=dark(BLUE))
        self._grid(left, sc, self.mesh[:, 0], op)
        left.dots(sc, self.z, BLUE, op * smoothstep(0.0, 0.2, u))
        right.frame(sc, op, "MNIST 0s, 1s and 4s as points", ink=dark(RED))
        right.dots(sc, self.x, RED, op * smoothstep(0.1, 0.3, u))
        self.box(sc, 0.5, 0.50, 150, 46, VIOLET, op, "the flow", ink=WHITE, size=0.038)
        self.text(sc, 0.5, 0.435, "six layers", op, size=0.034)
        w = seg(u, 0.3, 0.6)
        self.arrow(sc, (0.385, 0.57), (0.615, 0.57), BLUE, op, w, width=2.6)
        self.pulse(sc, (0.385, 0.57), (0.615, 0.57), seg(u, 0.3, 0.65), BLUE, op)
        self.text(sc, 0.5, 0.61, "generate", op * smoothstep(0.35, 0.5, u), ink=dark(BLUE),
                  size=0.034)
        w = seg(u, 0.55, 0.85)
        self.arrow(sc, (0.615, 0.70), (0.385, 0.70), RED, op, w, width=2.6)
        self.pulse(sc, (0.615, 0.70), (0.385, 0.70), seg(u, 0.55, 0.9), RED, op)
        self.text(sc, 0.5, 0.74, "measure", op * smoothstep(0.6, 0.75, u), ink=dark(RED),
                  size=0.034)
        self.lines(sc, 0.5, 0.82, ["the same map,", "run either way"],
                   op * smoothstep(0.75, 0.9, u), anchor="mm", step=0.044)

    def phase_2(self, sc, u, op):
        pl = Plane(self, 0.215, 0.615, 296.0, self.LIM)
        w = smoothstep(0.0, 1.0, seg(u, 0.3, 0.72))
        pl.frame(sc, op, "one layer: every point slides sideways")
        self._grid(pl, sc, self.mesh[:, 0] * (1 - w) + self.mesh[:, 1] * w, op)
        now = self.states[0] * (1 - w) + self.states[1] * w
        rest = np.ones(len(now), bool)
        rest[self.picks] = False
        pl.dots(sc, now[rest], BLUE, op * 0.7)
        for k in self.picks:
            pl.path(sc, [self.states[0][k], now[k]], AMBER, op, width=2.4)
            pl.dots(sc, [self.states[0][k]], DIM, op, r=4.0)
            pl.dots(sc, [now[k]], AMBER, op, r=5.0, glow=0.3)
        one = self.picks[0]
        old, new = self.states[0][one], self.states[1][one]
        x = 0.47
        self.text(sc, x, 0.335, "follow the ringed point", op, anchor="lm")
        self.disk(sc, self.P(*pl.at(now[one])), 6.5, AMBER, op, glow=0.3, ring=INK, ring_w=2.0,
                  lift=1.4)
        # the height passes through; a small network turns it into a shift and a stretch
        ya, yb = 0.42, 0.70
        self.box(sc, x + 0.055, ya, 96, 44, soft(TEAL), op, "height", border=TEAL, size=0.034)
        self.box(sc, x + 0.42, ya, 96, 44, soft(TEAL), op, "height", border=TEAL, size=0.034)
        self.arrow(sc, (x + 0.11, ya), (x + 0.365, ya), TEAL, op, width=2.2)
        self.text(sc, x + 0.24, ya - 0.035, "kept: %.2f" % old[1], op, ink=dark(TEAL))
        self.box(sc, x + 0.055, yb, 96, 44, soft(AMBER), op, "sideways", border=AMBER,
                 size=0.034)
        self.box(sc, x + 0.42, yb, 96, 44, soft(AMBER), op, "sideways", border=AMBER,
                 size=0.034)
        self.box(sc, x + 0.24, yb, 150, 44, AMBER, op, "shift, stretch", ink=WHITE, size=0.034)
        self.arrow(sc, (x + 0.11, yb), (x + 0.155, yb), AMBER, op, width=2.2)
        self.arrow(sc, (x + 0.325, yb), (x + 0.365, yb), AMBER, op, width=2.2)
        self.box(sc, x + 0.24, 0.56, 150, 40, VIOLET, op, "small network", ink=WHITE, size=0.034)
        v = seg(u, 0.08, 0.3)
        self.arrow(sc, (x + 0.16, ya + 0.045), (x + 0.21, 0.52), TEAL, op, v, width=2.0)
        self.arrow(sc, (x + 0.24, 0.60), (x + 0.24, yb - 0.05), VIOLET, op, seg(u, 0.2, 0.34),
                   width=2.0)
        self.pulse(sc, (x + 0.11, yb), (x + 0.365, yb), seg(u, 0.3, 0.72), AMBER, op)
        self.pulse(sc, (x + 0.11, ya), (x + 0.365, ya), seg(u, 0.3, 0.72), TEAL, op)
        shown = op * smoothstep(0.25, 0.4, u)
        self.lines(sc, x, 0.80, [
            ("shift %.2f, stretch %.2f (both set by the height)" % (
                self.shifts[one], self.stretches[one]), dark(VIOLET)),
            ("sideways: (%.2f %s %.2f) × %.2f = %.2f" % (
                old[0], "-" if self.shifts[one] >= 0 else "+", abs(self.shifts[one]),
                self.stretches[one], new[0]), dark(AMBER)),
            "its height is untouched, so the layer can be undone"],
            shown, step=0.046, size=0.031)

    def phase_3(self, sc, u, op):
        pl = Plane(self, 0.215, 0.615, 296.0, self.LIM)
        n = len(self.states) - 1
        at = seg(u, 0.05, 0.82) * n
        k = min(int(at), n - 1)
        w = smoothstep(0.15, 0.85, at - k)
        pl.frame(sc, op, "the cloud and its grid, layer by layer")
        self._grid(pl, sc, self.mesh[:, k] * (1 - w) + self.mesh[:, k + 1] * w, op)
        done = smoothstep(0.84, 0.94, u)
        pl.dots(sc, self.x, RED, op * 0.6 * done, r=2.2)
        pl.dots(sc, self.states[k] * (1 - w) + self.states[k + 1] * w, BLUE, op)
        x = 0.50
        for row in range(n):
            layer = n - 1 - row
            sideways = layer % 2 == 1
            y = 0.36 + row * 0.072
            active = row == k and u < 0.84
            past = row < k or u >= 0.84
            fill = (AMBER if sideways else TEAL)
            self.box(sc, x + 0.04, y, 64, 32, fill if (active or past) else soft(fill, 0.75), op,
                     str(row + 1), ink=WHITE if (active or past) else dark(fill), size=0.034,
                     glow=0.25 if active else 0.0)
            self.text(sc, x + 0.09, y, "slides points %s" % (
                "sideways" if sideways else "up and down"), op * (1.0 if active or past else 0.5),
                anchor="lm", size=0.034, strong=active)
        self.key(sc, x + 0.012, 0.83, BLUE, "240 points drawn from the bell", op)
        self.key(sc, x + 0.012, 0.875, RED, "240 real digits, for comparison", op * done)

    def phase_4(self, sc, u, op):
        data = Plane(self, 0.185, 0.60, 250.0, self.LIM)
        bell = Plane(self, 0.515, 0.60, 250.0, self.LIM)
        data.frame(sc, op, "digits, coloured by stretch", below="red: stretched most")
        bell.frame(sc, op, "their partners under the bell", ink=dark(BLUE))
        self._grid(bell, sc, self.mesh[:, 0], op * 0.6)
        top = float(np.abs(self.logdet).max())
        cols = heat(self.logdet / top, hot=RED, cold=TEAL)
        ok = data.inside(self.x)
        for p, c in zip(self.x[ok], cols[ok]):
            self.disk(sc, self.P(*data.at(p)), 3.2, c, op, lift=1.0, outline=0.25)
        sent = smoothstep(0.15, 0.45, u)
        bell.dots(sc, self.encoded, BLUE, op * sent)
        a, b = data.at(self.x[self.one]), bell.at(self.encoded[self.one])
        w = seg(u, 0.2, 0.5)
        self.disk(sc, self.P(*a), 6.0, AMBER, op, glow=0.3)
        with self.over():
            self.arc(sc, a, b, AMBER, op, w, bend=0.12, width=2.4)
        self.pulse(sc, a, b, w, AMBER, op, bend=0.12)
        self.disk(sc, self.P(*b), 6.0, AMBER, op * smoothstep(0.9, 1.0, w), glow=0.3)
        x = 0.70
        total = self.one_bell + self.logdet[self.one]
        self.text(sc, x, 0.36, "one digit's log density", op, anchor="lm", strong=True,
                  size=0.034)
        rows = [("its partner under the bell", "%.2f" % self.one_bell, BLUE, 0.45),
                ("+ log of the flow's stretch", "%+.2f" % self.logdet[self.one], RED, 0.58),
                ("= the digit's log density", "%.2f" % total, AMBER, 0.72)]
        for i, (label, value, ink, when) in enumerate(rows):
            y = 0.44 + i * 0.115
            show = op * smoothstep(when, when + 0.1, u)
            self.text(sc, x, y, label, show, anchor="lm", ink=dark(ink))
            self.text(sc, x, y + 0.048, value, show, anchor="lm", size=0.044, strong=True)
        self.text(sc, x, 0.80, "no sampling, no bound:", op * smoothstep(0.8, 0.9, u),
                  anchor="lm")
        self.text(sc, x, 0.842, "an exact number", op * smoothstep(0.8, 0.9, u), anchor="lm")

    def phase_5(self, sc, u, op):
        pl = Plane(self, 0.185, 0.60, 250.0, self.LIM)
        pl.frame(sc, op, "the fitted flow", below="grid and samples after six layers")
        self._grid(pl, sc, self.mesh[:, -1], op)
        pl.dots(sc, self.x, RED, op * 0.5, r=2.0)
        pl.dots(sc, self.states[-1], BLUE, op, r=2.4)
        steps = np.arange(1, len(self.loss) + 1)
        chart = self.chart(((0.44, 0.93), (0.36, 0.56)), steps, np.r_[self.loss, self.plain])
        self.axes(sc, chart, op, "training update", "nats per digit")
        self.plot(sc, chart, steps, self.loss, VIOLET, op, seg(u, 0.05, 0.5), width=1.4)
        full = self.span(0.25)[0]
        self.text(sc, 0.44, 0.675, "held-out negative log likelihood, lower is better", op,
                  anchor="lm", strong=True)
        rows = [("the bell alone, no flow", self.plain, DIM, "%.2f" % self.plain),
                ("the best single Gaussian", self.gaussian, AMBER, "%.2f" % self.gaussian)]
        rows += [("the flow, seed %d" % seed, value, VIOLET, "%.2f" % value)
                 for seed, value in zip(sg.FLOW_SEEDS, self.held)]
        for row, (label, value, ink, shown) in enumerate(rows):
            self.hbar(sc, 0.64, 0.725 + row * 0.047, full * value / 3.0, ink, op, label, shown,
                      seg(u, 0.4 + 0.08 * row, 0.6 + 0.08 * row), thick=17.0, track=full)


# ====================================================================== 61 flow-matching
def euler_path(model, start, steps):
    """``start`` carried along ``model``'s velocity field in ``steps`` equal
    Euler steps from time 0 to 1 (the study's sampler): ``(steps + 1, n, 2)``."""
    z, out = np.array(start, float), [np.array(start, float)]
    for k in range(steps):
        z = z + toy.forward(model, np.column_stack((z, np.full(len(z), k / steps))))[0] / steps
        out.append(z.copy())
    return np.array(out)


def velocity(model, points, t):
    """``model``'s velocity at ``points`` and time ``t``."""
    points = np.asarray(points, float)
    return toy.forward(model, np.column_stack((points, np.full(len(points), float(t)))))[0]


class FlowMatching(Generative):
    """Flow matching on MNIST's two main PCA coordinates: straight training
    lines between noise and digits, the velocity field they teach and its
    curved paths, the rectified field's straight ones, and what each costs in
    sampling steps."""

    TITLE = "Flow matching"
    PHASES = (("training lines", 12.0), ("learned flow", 13.0), ("rectified", 12.0),
              ("few steps", 13.0))
    LIM = (-3.2, 3.2, -3.2, 3.2)
    FEW = (1, 4, 10)

    def setup(self):
        d = raw("flow-matching")
        s = self.study
        self.models = [weights(d, "weights_"), weights(d, "rectified_weights_")]
        self.start = np.asarray(s.panels[0].data["xy"], float)
        self.target = np.asarray(d["target"], float)
        self.paths = [np.asarray(d["paths"], float), np.asarray(d["rect_paths"], float)]
        self.loss = np.asarray(s.panels[1].data["ys"][0], float)
        self.excess = [float(v) for v in s.panels[3].data["values"]]
        self.few_error = np.asarray(d["few_error"], float)
        self.few = [[euler_path(m, self.start, n) for n in self.FEW] for m in self.models]
        self.ode_steps = len(self.paths[0]) - 1
        # training pairs as the study draws them: a noise point and a digit, independently
        ok = (np.abs(self.start) < 2.6).all(1) & (np.abs(self.target) < 2.6).all(1)
        self.pairs = np.flatnonzero(ok)[:7]
        # the paths followed closely: the twelve the first flow carries farthest, and a
        # window that holds them under both flows
        reach = np.linalg.norm(self.paths[0][-1] - self.start, axis=1)
        self.followed = np.argsort(-reach)[:12]
        both = np.concatenate([q[:, self.followed].reshape(-1, 2) for q in self.paths])
        mid = (both.min(0) + both.max(0)) / 2
        half = float((both.max(0) - both.min(0)).max()) / 2 * 1.12
        self.close = (mid[0] - half, mid[0] + half, mid[1] - half, mid[1] + half)
        axis = np.linspace(-2.6, 2.6, 9)
        gx, gy = np.meshgrid(axis, axis)
        self.lattice = np.column_stack((gx.ravel(), gy.ravel()))
        self.loss_ends = (float(self.loss[:20].mean()), float(self.loss[-100:].mean()))

    def subtitle_for(self, k):
        e = self.few_error
        return [
            "training: join a noise point to a digit with a straight line and teach a network "
            "the line's direction at every point along it; its error falls from %.2f to %.2f"
            % self.loss_ends,
            "the trained network is a field of arrows; following it in %d small steps carries "
            "noise to the digits, along paths %.2f longer than a straight line would be" % (
                self.ode_steps, self.excess[0]),
            "rectifying: train again on each noise point and the place the first flow took it; "
            "the new paths are only %.3f longer than straight" % self.excess[1],
            "straight paths need few steps: in one step the first flow misses its own 40-step "
            "answer by %.2f (squared), the rectified one by %.3f" % (e[0, 0], e[1, 0]),
        ][k]

    # ------------------------------------------------------------------ drawing
    def phase_1(self, sc, u, op):
        pl = Plane(self, 0.215, 0.615, 296.0, self.LIM)
        pl.frame(sc, op, "noise and digits")
        pl.dots(sc, self.start, BLUE, op * 0.45, r=2.2)
        pl.dots(sc, self.target, RED, op * 0.45, r=2.2)
        grow = seg(u, 0.12, 0.36)
        t = seg(u, 0.4, 0.9)
        for k in self.pairs:
            a, b = self.start[k], self.target[k]
            pl.path(sc, [a, a + (b - a) * grow], AXIS, op, width=1.6)
            pl.dots(sc, [a], BLUE, op, r=4.5)
            pl.dots(sc, [b], RED, op * smoothstep(0.9, 1.0, grow), r=4.5)
            if t > 0.0:
                at = a + (b - a) * t
                pl.vector(sc, at, (b - a) * 0.22, AMBER, op * min(1.0, (1 - t) / 0.1 + 0.4),
                          width=2.4)
                pl.dots(sc, [at], AMBER, op, r=4.0, glow=0.3)
        x = 0.47
        self.key(sc, x, 0.335, BLUE, "noise", op, size=0.034)
        self.key(sc, x + 0.10, 0.335, RED, "MNIST digits as points", op, size=0.034)
        self.lines(sc, x - 0.008, 0.40, [
            "1  pair a noise point with a digit, at random",
            "2  join them with a straight line",
            ("3  anywhere on the line, the target is its direction", dark(AMBER))],
            op, step=0.047, size=0.032)
        y = 0.585
        self.box(sc, x + 0.065, y, 130, 40, PAPER, op, "place, time", border=AXIS, size=0.034)
        self.box(sc, x + 0.235, y, 110, 40, VIOLET, op, "network", ink=WHITE, size=0.034)
        self.box(sc, x + 0.40, y, 120, 40, soft(AMBER), op, "direction", border=AMBER,
                 size=0.034)
        self.arrow(sc, (x + 0.137, y), (x + 0.172, y), VIOLET, op)
        self.arrow(sc, (x + 0.297, y), (x + 0.332, y), VIOLET, op)
        steps = np.arange(1, len(self.loss) + 1)
        chart = self.chart(((x + 0.03, 0.94), (0.69, 0.86)), steps, self.loss)
        self.axes(sc, chart, op, "training update", "error")
        self.plot(sc, chart, steps, self.loss, VIOLET, op, seg(u, 0.3, 0.92), width=1.4)

    def _flow(self, sc, u, op, which, ink, label):
        """The model's field with all 180 points moving, and twelve paths
        against the straight line between their ends."""
        model, paths = self.models[which], self.paths[which]
        t = seg(u, 0.08, 0.84)
        at = t * self.ode_steps
        now = self.between(paths, at)
        left = Plane(self, 0.215, 0.615, 296.0, self.LIM)
        left.frame(sc, op, "time %.2f" % t)
        v = velocity(model, self.lattice, min(t, 0.999))
        for p, d in zip(self.lattice, v):
            left.vector(sc, p - d * 0.09, d * 0.18, soft(ink, 0.25), op * 0.9, width=1.5)
        left.dots(sc, self.target, RED, op * 0.35, r=2.0)
        left.dots(sc, now, BLUE, op, r=2.8)
        right = Plane(self, 0.64, 0.615, 296.0, self.close)
        right.frame(sc, op, label)
        n = int(at)
        for k in self.followed:
            trail = np.vstack((paths[:n + 1, k], now[k][None]))
            right.path(sc, [paths[0, k], now[k]], DIM, op, width=1.3)
            right.path(sc, trail, ink, op, width=2.2)
            right.dots(sc, [paths[0, k]], BLUE, op, r=3.4)
            right.dots(sc, [now[k]], ink, op, r=3.6)
        long = np.linalg.norm(np.diff(paths[:n + 1], axis=0), axis=-1).sum(0).mean() if n else 0.0
        straight = np.linalg.norm(paths[n] - paths[0], axis=-1).mean()
        x = 0.82
        self.lines(sc, x, 0.42, [("path so far", dark(ink)), ("straight line", None)], op,
                   step=0.12, size=0.032)
        self.text(sc, x, 0.47, "%.2f" % long, op, anchor="lm", size=0.044, strong=True)
        self.text(sc, x, 0.59, "%.2f" % straight, op, anchor="lm", size=0.044, strong=True)
        done = op * smoothstep(0.84, 0.92, u)
        self.text(sc, x, 0.69, "extra length", done, anchor="lm", ink=dark(ink), size=0.032)
        self.text(sc, x, 0.74, "%.3f" % self.excess[which], done, anchor="lm", size=0.044,
                  strong=True)
        self.text(sc, x, 0.80, "(mean of 180)", op, anchor="lm")

    def phase_2(self, sc, u, op):
        self._flow(sc, u, op, 0, AMBER, "close up: the 12 longest paths")

    def phase_3(self, sc, u, op):
        self._flow(sc, u, op, 1, GREEN, "the same 12, rectified")

    def phase_4(self, sc, u, op):
        which = min(int(seg(u, 0.0, 0.8) * 3), 2)
        within = seg(u, 0.0, 0.8) * 3 - which if u < 0.8 else 1.0
        steps = self.FEW[which]
        for m, (ink, name) in enumerate(((AMBER, "first flow"), (GREEN, "rectified flow"))):
            pl = Plane(self, 0.18 + m * 0.285, 0.59, 236.0, self.close)
            pl.frame(sc, op, "%s, %d step%s" % (name, steps, "" if steps == 1 else "s"),
                     ink=dark(ink))
            path = self.few[m][which]
            reach = min(within / 0.7, 1.0) * steps
            n = min(int(reach), steps - 1)
            f = reach - n if reach < steps else 1.0
            for k in self.followed:
                head = path[n, k] + (path[n + 1, k] - path[n, k]) * f
                pl.path(sc, np.vstack((path[:n + 1, k], head[None])), ink, op, width=2.2)
                pl.dots(sc, [path[0, k]], BLUE, op, r=3.2)
                pl.dots(sc, [head], ink, op, r=3.6)
                ref = self.paths[m][-1, k]
                if pl.inside(ref):
                    self.disk(sc, self.P(*pl.at(ref)), 5.0, PAPER, op, ring=INK, ring_w=1.6,
                              lift=0.9, outline=0.0)
        x = 0.74
        self.text(sc, x - 0.02, 0.345, "miss, against 40 steps", op, anchor="lm", strong=True,
                  size=0.034)
        full = self.span(0.15)[0]
        top = float(self.few_error.max())
        for row, n in enumerate(self.FEW):
            y = 0.425 + row * 0.15
            lit = 1.0 if row <= which else 0.3
            self.text(sc, x - 0.02, y, "%d step%s" % (n, "" if n == 1 else "s"), op * lit,
                      anchor="lm", size=0.034, strong=row == which)
            for m, ink in enumerate((AMBER, GREEN)):
                value = self.few_error[m, row]
                self.hbar(sc, x - 0.02, y + 0.045 + m * 0.042, max(full * value / top, 1.5),
                          ink, op * lit, "", "%.2g" % value, thick=15.0, track=full)
        self.text(sc, 0.05, 0.885, "ring: where 40 small steps of the same flow end", op,
                  anchor="lm")



# ====================================================================== 62 classifier-free-guidance
SCALES = sg.CFG_SCALES


class ClassifierFreeGuidance(Generative):
    """One denoiser asked twice (told the digit, told nothing), the dial that
    pushes past the difference, four dial settings denoising the same noise
    into the ten digits, and what a separate classifier reads in the results."""

    TITLE = "Classifier-free guidance"
    PHASES = (("two guesses", 12.0), ("the dial", 13.0), ("sampling", 14.0),
              ("measured", 13.0))
    AT = 20                          # the sampler step the first two phases look at (of 50)
    WHICH = 3                        # ... on the sample asked to be a 3, in the dial-1 chain
    SIDE = sg.CFG_SIDE

    def setup(self):
        d = sg.arrays("classifier-free-guidance")
        s = self.study
        self.scales = [int(v) for v in s.panels[3].data["x"]]
        self.match = np.asarray(d["match"], float)             # denoiser, dial
        self.variety = np.asarray(d["variety"], float)
        self.ink = np.asarray(d["ink"], float).mean(0)
        self.judge = float(d["judge_accuracy"])
        self.x0 = d["trace_x0"]                                # dial, step, digit, pixel
        self.steps = self.x0.shape[1]
        self.finals = np.asarray(d["finals"], float)
        self.per = self.finals.shape[1]
        one = self.scales.index(1)
        self.noisy = d["trace_x"][one, self.AT, self.WHICH]
        self.e0 = d["trace_e0"][one, self.AT, self.WHICH]
        self.ec = d["trace_ec"][one, self.AT, self.WHICH]
        self.gap = self.ec - self.e0
        self.top = float(np.abs(self.e0 + 3 * self.gap).max())
        self.ratio = float(np.linalg.norm(self.gap) / np.linalg.norm(self.e0))
        self.level = int(d["timesteps"][self.AT])
        self.levels = int(d["timesteps"][0])
        self.abar = float(d["abar"][self.level])

    def implied(self, eps):
        """The clean image a noise guess implies at the step looked at: what the dial
        changes."""
        return np.clip((self.noisy - np.sqrt(1 - self.abar) * eps) / np.sqrt(self.abar), -1, 1)

    def subtitle_for(self, k):
        lo, hi, mean = self.match.min(0), self.match.max(0), self.variety.mean(0)

        def span(k):
            a, b = "%d" % (round(1000 * lo[k]) // 10), "%d" % (round(1000 * hi[k]) // 10)
            return a + "%" if a == b else "%s-%s%%" % (a, b)
        return [
            "one denoiser is asked twice about the same noisy image: told the digit should "
            "be a %d, and told nothing; each time it guesses the noise to remove" % self.WHICH,
            "guidance starts from the told-nothing guess and adds the difference times a "
            "dial: 0 ignores the label, 1 is the plain told-the-digit guess, %d pushes far "
            "past it; here the difference is %.0f%% the size of a guess" % (
                max(self.scales), 100 * self.ratio),
            "the same starting noise denoised in %d steps at dial 0, 1, 3 and 8, asking for "
            "each digit 0 to 9: every picture is the model's current estimate of the clean "
            "image" % self.steps,
            "three denoisers, %d samples a dial: read as the digit asked for by %s at dial "
            "0, %s at 1 and %s at 3; variety within a digit falls %.3f to %.3f, then rises "
            "at 8 as strokes thicken" % (
                self.per, span(0), span(1), span(2), mean[0], mean[2]),
        ][k]

    # ------------------------------------------------------------------ drawing
    def _guess(self, sc, values, fx, fy, side, op, ink, label=None):
        """A noise guess as a heatmap framed in ``ink``."""
        self.rect(sc, self.P(fx, fy), side + 12, side + 12, ink, op, radius=6, lift=0.05)
        self.grid(sc, np.asarray(values).reshape(self.SIDE, self.SIDE), fx, fy, side, side, op,
                  vmax=self.top, frame=False, lines=0)
        if label:
            self.text(sc, fx, fy - self.frac(0, side / 2)[1] - 0.04, label, op, ink=dark(ink),
                      size=0.032)

    def phase_1(self, sc, u, op):
        x0, yc = 0.13, 0.63
        self.digit(sc, (self.noisy + 1) / 2, x0, yc, 150, op, frame=AXIS)
        self.text(sc, x0, yc - 0.19, "a noisy image", op, strong=True, size=0.034)
        self.text(sc, x0, yc + 0.185, "noise level %d of %d" % (self.level, self.levels), op)
        rows = ((0.47, RED, "told: a %d" % self.WHICH, self.ec, 0.1),
                (0.79, BLUE, "told nothing", self.e0, 0.35))
        for y, ink, told, guess, when in rows:
            w = seg(u, when, when + 0.22)
            self.box(sc, 0.36, y, 124, 46, VIOLET, op, "denoiser", ink=WHITE, size=0.034)
            self.box(sc, 0.36, y - 0.085, 124, 30, soft(ink), op, told, border=ink, size=0.031)
            self.arc(sc, (x0 + 0.09, yc), (0.292, y), AXIS, op, 1.0, bend=0.0, width=2.0)
            self.pulse(sc, (x0 + 0.09, yc), (0.292, y), seg(u, when, when + 0.12), ink, op)
            self.arrow(sc, (0.428, y), (0.49, y), ink, op, w, width=2.4)
            self.pulse(sc, (0.428, y), (0.49, y), seg(u, when + 0.1, when + 0.22), ink, op)
            self._guess(sc, guess, 0.57, y, 112, op * smoothstep(0.7, 1.0, w), ink,
                        "its guess of the noise")
            self.arrow(sc, (0.64, y), (0.775, yc + (y - yc) * 0.25), AMBER, op,
                       seg(u, 0.6, 0.75), width=2.2)
        shown = smoothstep(0.7, 0.85, u)
        self._guess(sc, self.gap, 0.85, yc, 118, op * shown, AMBER, "the difference")
        self.text(sc, 0.85, yc + 0.165, "what the label adds:", op * shown, ink=dark(AMBER),
                  size=0.032)
        self.text(sc, 0.85, yc + 0.21, "%.0f%% the size of a guess" % (100 * self.ratio),
                  op * shown, ink=dark(AMBER), size=0.032)

    def _dial(self, u):
        """The dial's value through phase 2: resting at each of the study's settings."""
        stops = self.scales
        x = seg(u, 0.06, 0.9) * len(stops)
        k = min(int(x), len(stops) - 1)
        f = smoothstep(0.55, 1.0, x - k)
        nxt = stops[min(k + 1, len(stops) - 1)]
        return stops[k] + (nxt - stops[k]) * f

    def phase_2(self, sc, u, op):
        w = self._dial(u)
        y = 0.455
        self._guess(sc, self.e0, 0.11, y, 112, op, BLUE, "told nothing")
        self.text(sc, 0.205, y, "+", op, size=0.06, strong=True)
        self.text(sc, 0.275, y, "%.1f ×" % w, op, size=0.05, ink=dark(AMBER))
        self._guess(sc, self.gap, 0.395, y, 112, op, AMBER, "the difference")
        self.text(sc, 0.49, y, "=", op, size=0.06, strong=True)
        used = self.e0 + w * self.gap
        self._guess(sc, used, 0.585, y, 112, op, GREEN, "the guess used")
        # what removing that guess leaves: the clean image the model now expects
        self.arrow(sc, (0.66, y), (0.745, y), GREEN, op, width=2.4)
        self.digit(sc, (self.implied(used) + 1) / 2, 0.84, y, 130, op, frame=GREEN)
        self.text(sc, 0.84, y - 0.175, "the clean image it implies", op, ink=dark(GREEN),
                  size=0.032)
        # the dial
        a, b, yd = 0.16, 0.84, 0.76
        top = float(max(self.scales))
        self.polyline(sc, [self.P(a, yd), self.P(b, yd)], 3.0, TRACK, op)
        notes = {0: "label ignored", 1: "plain: told the digit", 8: "far past it"}
        for stop in self.scales:
            fx = a + (b - a) * stop / top
            self.polyline(sc, [self.P(fx, yd - 0.018), self.P(fx, yd + 0.018)], 2.0, AXIS, op)
            self.text(sc, fx, yd + 0.05, str(stop), op, size=0.036, strong=abs(w - stop) < 0.05)
            if stop in notes:
                self.text(sc, fx + (0.012 if stop == 0 else -0.012 if stop == 1 else 0.0),
                          yd + 0.095, notes[stop], op * (1.0 if abs(w - stop) < 0.6 else 0.45),
                          anchor={0: "rm", 1: "lm"}.get(stop, "mm"))
        fx = a + (b - a) * w / top
        self.polyline(sc, [self.P(a, yd), self.P(fx, yd)], 3.4, GREEN, op)
        self.disk(sc, self.P(fx, yd), 9.0, GREEN, op, glow=0.35)
        self.text(sc, a - 0.02, yd, "dial", op, anchor="rm", size=0.034, strong=True)

    ROWS = (0.415, 0.56, 0.705, 0.85)

    def _samples(self, sc, op, at, x0, pitch, size, rows=None):
        """One sample per digit at each dial, at fractional step index ``at``."""
        rows = rows or self.ROWS
        for row, (y, w) in enumerate(zip(rows, self.scales)):
            self.text(sc, x0, y, "dial %d" % w, op, anchor="lm", size=0.034, strong=True)
            for col in range(10):
                img = (self.between(self.x0[row, :, col], at) + 1) / 2
                self.digit(sc, img, x0 + 0.105 + col * pitch, y, size, op,
                           frame=GREEN if w else BLUE)
        for col in range(10):
            self.text(sc, x0 + 0.105 + col * pitch, rows[0] - 0.09, str(col), op, size=0.034)
        self.text(sc, x0, rows[0] - 0.09, "asked for", op, anchor="lm")

    def phase_3(self, sc, u, op):
        prog = seg(u, 0.05, 0.8)
        self._samples(sc, op, prog * (self.steps - 1), 0.06, 0.066, 52.0)
        x = 0.80
        step = 1 + int(round(prog * (self.steps - 1)))
        self.text(sc, x, 0.40, "step %d of %d" % (step, self.steps), op, anchor="lm",
                  size=0.034, strong=True)
        full = self.span(0.13)[0]
        self.hbar(sc, x, 0.455, full * prog, GREEN, op, thick=12.0, track=full)
        self.lines(sc, x, 0.55, ["every row starts", "from the same", "noise; only the",
                                 "dial differs"], op, step=0.046, size=0.034)

    def _spread(self, sc, chart, xs, values, ink, op, shown):
        """Mean of the three denoisers at each dial, with their lowest and highest."""
        for k, stop in enumerate(xs):
            lo, hi = values[:, k].min(), values[:, k].max()
            fx = chart.fx(stop)
            self.polyline(sc, [self.P(fx, chart.fy(lo)), self.P(fx, chart.fy(hi))], 5.0,
                          soft(ink, 0.35), op * shown, bias=0.2)
            self.disk(sc, self.P(fx, chart.fy(values[:, k].mean())), 4.5, ink, op * shown,
                      lift=1.2)

    def phase_4(self, sc, u, op):
        rows = (0.43, 0.555, 0.68, 0.805)
        self._samples(sc, op, self.steps - 1, 0.04, 0.041, 34.0, rows)
        self.text(sc, 0.04, 0.90, "one of the %d samples per digit and dial" % (
            self.per // 10), op, anchor="lm")
        # two measurements against the dial, each on its own scale
        xs = np.asarray(self.scales, float)
        box = (0.635, 0.945)
        top = self.chart((box, (0.37, 0.54)), xs, [0.0, 1.0], pad=0.04)
        ticks = [(float(v), str(int(v))) for v in xs]
        self.axes(sc, top, op, "", "read as the digit asked for", xticks=ticks,
                  yticks=[(0, "0"), (0.5, "50%"), (1, "100%")])
        self.plot(sc, top, xs, self.match.mean(0), GREEN, op, seg(u, 0.05, 0.45), width=2.6)
        self._spread(sc, top, xs, self.match, GREEN, op, smoothstep(0.45, 0.5, u))
        low = self.chart((box, (0.70, 0.86)), xs, self.variety, pad=0.25)
        self.axes(sc, low, op, "dial", "variety within a digit", xticks=ticks)
        self.plot(sc, low, xs, self.variety.mean(0), VIOLET, op, seg(u, 0.4, 0.8), width=2.6)
        self._spread(sc, low, xs, self.variety, VIOLET, op, smoothstep(0.8, 0.85, u))


# ====================================================================== 63 ddpm-vs-ddim
DDIM_STEPS = (10, 25, 50, 200)


def sampler_replay():
    """The study's two samplers run again from its starting noise and weights,
    keeping every state: ``ddpm`` ``(201, 4, 784)`` and, per DDIM setting,
    ``ddim<n>`` ``(n + 1, 4, 784)`` with the clean-image estimates ``ddim<n>_x0``."""
    def build():
        from . import diffusion as df

        d = raw("ddpm-vs-ddim")
        p, sched = weights(d), df.schedule()
        abar = sched[2]
        rng = np.random.default_rng(nv.SEED)
        initial = rng.normal(size=(4, 784))          # the study's first draw
        out = {}
        for steps in DDIM_STEPS:
            x, xs, x0s = initial.copy(), [initial.copy()], []
            times = np.linspace(df.T_STEPS, 0, steps + 1).astype(int)
            for t, prev in itertools.pairwise(times):
                eps = df.forward(p, x, np.full(len(x), t))["eps"]
                x0 = df.predict_x0(x, eps, t, abar)
                x = np.sqrt(abar[prev]) * x0 + np.sqrt(1 - abar[prev]) * eps
                xs.append(x.copy())
                x0s.append(x0)
            out["ddim%d" % steps], out["ddim%d_x0" % steps] = np.array(xs), np.array(x0s)
        x, xs = initial.copy(), [initial.copy()]
        for t in range(df.T_STEPS, 0, -1):
            x, _, _ = df.reverse_step(p, x, t, rng.normal(size=x.shape), sched)
            xs.append(x.copy())
        out["ddpm"] = np.array(xs)
        return out
    return replay("samplers", build)


def picture(x):
    """A sampler state (signal in [-1, 1]) as pixels in [0, 1]."""
    return np.clip((np.asarray(x, float) + 1.0) / 2.0, 0.0, 1.0)


class DdpmVsDdim(Generative):
    """One trained MNIST denoiser sampled two ways from the same noise: DDPM
    down every one of 200 noise levels with fresh noise each time, DDIM in ten
    deterministic jumps; one pixel's path under each; and the study's
    confidence scores against the number of steps."""

    TITLE = "DDPM and DDIM sampling"
    PHASES = (("noise ladder", 9.0), ("DDPM", 12.0), ("DDIM", 12.0), ("one pixel", 10.0),
              ("steps and quality", 10.0))
    TRACK = (0.09, 0.91, 0.875)
    FEW = 10
    ONE = 1                               # the sample followed in phases 1 and 4

    def setup(self):
        from . import mnist

        s = self.study
        self.r = sampler_replay()
        self.levels = len(self.r["ddpm"]) - 1
        self.scores = [float(v) for v in s.panels[3].data["values"]]
        self.finals = [picture(self.r["ddim%d" % n][-1]) for n in DDIM_STEPS]
        self.ddpm_final = np.asarray(s.panels[0].data["values"], float).reshape(4, -1)
        classifier, _ = mnist.train()
        self.ddpm_score = float(mnist._forward(classifier, self.ddpm_final)[2].max(1).mean())
        few = self.r["ddim%d" % self.FEW]
        # the pixel followed: one both samplers end up inking, that started farthest from ink
        both = np.minimum(picture(self.r["ddpm"][-1, self.ONE]), picture(few[-1, self.ONE]))
        start = self.r["ddpm"][0, self.ONE]
        self.pixel = int(np.argmin(np.where(both > 0.7, start, np.inf)))
        self.trace = [self.r["ddpm"][:, self.ONE, self.pixel], few[:, self.ONE, self.pixel]]
        self.jump = self.levels // self.FEW

    def subtitle_for(self, k):
        return [
            "a diffusion model makes an image by walking down a ladder of %d noise levels, "
            "from pure noise to a clean digit; one trained denoiser, two ways down" % self.levels,
            "DDPM steps down every level: the denoiser guesses the noise, a little is removed, "
            "and fresh random noise is added; %d denoiser calls for one batch" % self.levels,
            "DDIM from the same starting noise: guess the noise, work out the clean image it "
            "implies, and jump %d levels re-using that guess; %d calls, nothing random" % (
                self.jump, self.FEW),
            "one pixel on the way down: DDPM jitters through %d random steps, DDIM takes %d "
            "straight strides; they end at %.2f and %.2f" % (
                self.levels, self.FEW, picture(self.trace[0][-1]), picture(self.trace[1][-1])),
            "a digit classifier's confidence in four samples each: %.2f after 10 DDIM steps, "
            "%.2f after 50, %.2f after 200, and %.2f for DDPM's 200; four samples is a rough "
            "guide only" % (self.scores[0], self.scores[2], self.scores[3], self.ddpm_score),
        ][k]

    # ------------------------------------------------------------------ drawing
    def _fx(self, level):
        a, b, _y = self.TRACK
        return a + (b - a) * (1.0 - level / self.levels)

    def _ladder(self, sc, op, marks=(), at=None, ink=RED, done=None):
        """The noise ladder, pure noise on the left: ``marks`` are the levels
        a sampler visits, ``at`` its position, ``done`` the level it has reached."""
        a, b, y = self.TRACK
        self.polyline(sc, [self.P(a, y), self.P(b, y)], 3.0, TRACK, op)
        if done is not None:
            self.polyline(sc, [self.P(a, y), self.P(self._fx(done), y)], 3.4, ink, op)
        for level in marks:
            fx = self._fx(level)
            self.polyline(sc, [self.P(fx, y - 0.014), self.P(fx, y + 0.014)], 1.6,
                          ink if done is not None and level >= done else AXIS, op)
        self.text(sc, a, y + 0.05, "level %d: pure noise" % self.levels, op, anchor="lm")
        self.text(sc, b, y + 0.05, "level 0: clean", op, anchor="rm")
        if at is not None:
            self.disk(sc, self.P(self._fx(at), y), 8.0, ink, op, glow=0.35, lift=1.4)

    def _row(self, sc, images, y, size, op, frame, x0=0.13, pitch=0.15):
        for i, img in enumerate(images):
            self.digit(sc, img, x0 + i * pitch, y, size, op, frame=frame)

    def _recipe(self, sc, op, rows, x=0.84, y=0.40, pitch=0.105):
        for i, (label, ink, lit) in enumerate(rows):
            self.box(sc, x, y + i * pitch, 190, 40, ink if lit else soft(ink, 0.7), op, label,
                     ink=WHITE if lit else dark(ink), size=0.031, glow=0.2 if lit else 0.0)
            if i:
                self.arrow(sc, (x, y + (i - 1) * pitch + 0.042), (x, y + i * pitch - 0.045),
                           AXIS, op, width=1.8)

    def phase_1(self, sc, u, op):
        chain = self.r["ddpm"][:, self.ONE]
        shown = (200, 150, 100, 50, 0)
        for i, level in enumerate(shown):
            fx = self._fx(level) + (0.045 if i == 0 else -0.045 if i == len(shown) - 1 else 0.0)
            w = smoothstep(0.0, 1.0, seg(u, 0.05 + 0.1 * i, 0.2 + 0.1 * i))
            self.digit(sc, picture(chain[self.levels - level]), fx, 0.50, 118, op * w, frame=AXIS)
            self.text(sc, fx, 0.355, "level %d" % level, op * w, size=0.034)
            self.arrow(sc, (fx, 0.63), (self._fx(level), 0.845), AXIS, op * w, width=1.5,
                       head=False)
        every = seg(u, 0.55, 0.75)
        self._ladder(sc, op, marks=range(0, self.levels + 1, 4 if every < 0.5 else 1))
        few = op * smoothstep(0.75, 0.9, u)
        y = self.TRACK[2]
        for level in range(0, self.levels + 1, self.jump):
            self.disk(sc, self.P(self._fx(level), y), 5.0, BLUE, few, lift=1.3)
        self.text(sc, (self._fx(150) + self._fx(100)) / 2, 0.725, "DDPM: every level",
                  op * smoothstep(0.55, 0.7, u), ink=dark(RED), size=0.032)
        self.text(sc, (self._fx(100) + self._fx(50)) / 2, 0.725,
                  "DDIM: %d of them" % (self.FEW + 1), few, ink=dark(BLUE), size=0.032)

    def phase_2(self, sc, u, op):
        prog = seg(u, 0.05, 0.88)
        step = int(round(prog * self.levels))
        level = self.levels - step
        self._row(sc, picture(self.r["ddpm"][step]), 0.50, 124, op, RED)
        self.text(sc, 0.355, 0.335, "four images, level %d" % level, op, strong=True, size=0.034)
        self.text(sc, 0.355, 0.685, "denoiser calls so far: %d" % step, op, size=0.034)
        beat = (prog * self.levels) % 1.0 if 0 < prog < 1 else 0.0
        self._recipe(sc, op, (("guess the noise", VIOLET, beat < 0.34),
                              ("remove a little", TEAL, 0.34 <= beat < 0.67),
                              ("add fresh noise", RED, beat >= 0.67)))
        self._ladder(sc, op, marks=range(0, self.levels + 1, 2), at=level, ink=RED, done=level)

    def phase_3(self, sc, u, op):
        chain, clean = self.r["ddim%d" % self.FEW], self.r["ddim%d_x0" % self.FEW]
        x = seg(u, 0.05, 0.9) * self.FEW
        k = min(int(x), self.FEW - 1)
        within = x - k if x < self.FEW else 1.0
        landed = within > 0.75
        state = k + 1 if landed else k
        move = smoothstep(0.45, 0.75, within)
        level = self.levels - self.jump * (k + move)
        self._row(sc, picture(chain[state]), 0.455, 96, op, BLUE)
        self.text(sc, 0.355, 0.31, "the same four, level %d" % (self.levels - self.jump * state),
                  op, strong=True, size=0.034)
        guess = op * smoothstep(0.15, 0.3, within) if not landed else op
        self._row(sc, picture(clean[k]), 0.70, 64, guess, TEAL)
        self.text(sc, 0.355, 0.60, "the clean image each guess implies", guess,
                  ink=dark(TEAL))
        self.text(sc, 0.84, 0.80, "denoiser calls: %d" % (k + 1 if x > 0 else 0), op,
                  size=0.034)
        self._recipe(sc, op, (("guess the noise", VIOLET, within < 0.2),
                              ("clean image implied", TEAL, 0.2 <= within < 0.45),
                              ("jump %d levels" % self.jump, BLUE, 0.45 <= within <= 0.75)))
        self.text(sc, 0.84, 0.72, "no fresh noise", op, ink=dark(BLUE), size=0.034)
        self._ladder(sc, op, marks=range(0, self.levels + 1, self.jump), at=level, ink=BLUE,
                     done=level)

    def phase_4(self, sc, u, op):
        x0, y0 = 0.17, 0.60
        w = seg(u, 0.05, 0.85)
        step = int(round(w * self.levels))
        self.digit(sc, picture(self.r["ddpm"][step, self.ONE]), x0, y0 - 0.115, 108, op,
                   frame=RED)
        hop = int(w * self.FEW)
        self.digit(sc, picture(self.r["ddim%d" % self.FEW][hop, self.ONE]), x0, y0 + 0.14, 108,
                   op, frame=BLUE)
        side = int(round(np.sqrt(self.r["ddpm"].shape[-1])))
        r, c = divmod(self.pixel, side)
        for yy in (y0 - 0.115, y0 + 0.14):
            dx, dy = self.frac((c + 0.5 - side / 2) * 108 / side, (r + 0.5 - side / 2) * 108 / side)
            self.rect(sc, self.P(x0 + dx, yy + dy), 12, 12, AMBER, op, border=AMBER,
                      border_w=2.2, radius=2, lift=0.8, fill_op=0.0)
        self.text(sc, x0, 0.31, "one pixel, ringed", op, ink=dark(AMBER), size=0.034)
        steps = np.arange(self.levels + 1)
        chart = self.chart(((0.36, 0.93), (0.36, 0.80)), steps, np.r_[self.trace[0], -1.0, 1.0])
        ticks = [(float(k), str(self.levels - k)) for k in range(0, self.levels + 1, 50)]
        self.axes(sc, chart, op, "noise level", "the pixel's value", xticks=ticks,
                  yticks=[(-1, "-1 paper"), (0, "0"), (1, "1 ink")])
        self.plot(sc, chart, steps, self.trace[0], RED, op, w, width=1.5)
        self.plot(sc, chart, steps[::self.jump], self.trace[1], BLUE, op, w, width=2.6)
        for k in range(hop + 1):
            self.disk(sc, self.P(chart.fx(k * self.jump), chart.fy(self.trace[1][k])), 4.5, BLUE,
                      op, lift=1.3)
        self.key(sc, 0.40, 0.905, RED, "DDPM, %d random steps" % self.levels, op)
        self.key(sc, 0.66, 0.905, BLUE, "DDIM, %d strides" % self.FEW, op)

    def phase_5(self, sc, u, op):
        rows = [("DDIM, %d steps" % n, BLUE, f, v) for n, f, v in zip(
            DDIM_STEPS, self.finals, self.scores)]
        rows.append(("DDPM, %d steps" % self.levels, RED, self.ddpm_final, self.ddpm_score))
        full = self.span(0.26)[0]
        self.text(sc, 0.62, 0.325, "classifier confidence", op, anchor="lm", strong=True,
                  size=0.034)
        self.text(sc, 0.37, 0.325, "the same four starting noises", op)
        for i, (label, ink, images, value) in enumerate(rows):
            y = 0.405 + i * 0.118
            self.text(sc, 0.06, y, label, op, anchor="lm", size=0.034, ink=dark(ink))
            self._row(sc, images, y, 44, op, ink, x0=0.265, pitch=0.07)
            self.hbar(sc, 0.62, y, full * value, ink, op, "", "%.2f" % value,
                      seg(u, 0.1 + 0.1 * i, 0.4 + 0.1 * i), thick=24.0, track=full)



# ====================================================================== 64 vae-posterior-collapse
def vae_probe(p, x):
    """The study's VAE ``p`` on images ``x`` with the code at its mean: the
    code ``mu``, the KL of each code number, the teacher-forced output, and the
    output again with every image given the next image's code instead."""
    mu = x @ p["E"] + p["eb"]
    lv = x @ p["L"] + p["lb"]
    earlier = x @ (p["AR"] * np.triu(np.ones_like(p["AR"]), 1)) + p["db"]
    out = vm.sigmoid(mu @ p["D"] + earlier)
    swapped = vm.sigmoid(np.roll(mu, 1, axis=0) @ p["D"] + earlier)
    return dict(mu=mu, kl=0.5 * (mu ** 2 + np.exp(lv) - 1 - lv), out=out, swapped=swapped)


class VaePosteriorCollapse(Generative):
    """A VAE whose decoder also reads the earlier pixels: the two routes to a
    pixel, the code draining under strong pressure, filling and partly
    surviving when the pressure is ramped, and what is left after training."""

    TITLE = "VAE posterior collapse"
    PHASES = (("two routes", 12.0), ("strong pressure", 13.0), ("ramped pressure", 13.0),
              ("what survives", 12.0))
    DIMS = (VIOLET, TEAL, AMBER, BLUE)            # the four code numbers
    RUNS = (("strong pressure", RED, "beta4_"), ("ramped pressure", GREEN, "annealed_"))
    RAMP = 900                                    # updates over which the second run's beta rises

    def setup(self):
        d = raw("vae-posterior-collapse")
        s = self.study
        self.kl = [np.asarray(s.panels[k].data["ys"], float) for k in (0, 1)]      # (4, steps)
        self.updates = self.kl[0].shape[1]
        self.left = [float(v) for v in s.panels[3].data["values"]]
        self.left_each = [k[:, -50:].mean(1) for k in self.kl]
        self.recon = np.asarray(s.panels[2].data["values"], float).reshape(2, 4, 49)
        self.images = mnist7("test", 256)
        self.probe = [vae_probe(weights(d, prefix), self.images) for _n, _i, prefix in self.RUNS]
        self.swap = [float(np.abs(q["swapped"] - q["out"]).mean()) for q in self.probe]
        self.top = float(max(k.max() for k in self.kl))
        self.beta = [np.full(self.updates, 4.0),
                     np.minimum(1.0, np.arange(self.updates) / self.RAMP)]
        self.peak = float(self.kl[1].max())

    def subtitle_for(self, k):
        keep = int(np.argmax(self.left_each[1]))
        return [
            "this autoencoder squeezes a digit into a code of four numbers; its decoder redraws "
            "each pixel from the code and from every pixel before it, so it could do without "
            "the code",
            "trained under strong pressure to keep the code simple (beta 4), the information "
            "in all four numbers drains away: %.3f nats each after %d updates" % (
                self.left[0], self.updates),
            "second run: the pressure starts at 0 and ramps to 1 over %d updates; the code "
            "fills to %.1f nats, then drains as the pressure rises, keeping %.2f in number %d"
            % (self.RAMP, self.peak, self.left_each[1][keep], keep + 1),
            "what is left: %.3f nats per number under strong pressure, %.3f with the ramp; "
            "the two runs differ in final pressure (4 and 1) as well as the ramp, so the "
            "ramp alone is not shown to be the cause" % (self.left[0], self.left[1]),
        ][k]

    # ------------------------------------------------------------------ drawing
    def phase_1(self, sc, u, op):
        x, out = self.images[0], self.recon[1][0]
        k = min(int(seg(u, 0.12, 0.95) * 49), 48)
        r, c = divmod(k, 7)
        seen = (np.arange(49) < k).reshape(7, 7)
        tex = shade(x.reshape(7, 7))
        tex[seen] = tex[seen] * 0.6 + np.asarray(BLUE) * 0.4
        self.pixels(sc, tex, 0.15, 0.62, 168, op, ring=[(r, c)])
        self.text(sc, 0.15, 0.40, "a digit", op, strong=True, size=0.034)
        self.text(sc, 0.15, 0.845, "blue: the pixels before", op, ink=dark(BLUE))
        self.text(sc, 0.15, 0.89, "the ringed one", op, ink=dark(BLUE))
        shown = np.where((np.arange(49) <= k).reshape(7, 7)[..., None], shade(out.reshape(7, 7)),
                         PAPER * 0.93)
        b = self.pixels(sc, shown, 0.85, 0.62, 168, op, ring=[(r, c)])
        self.text(sc, 0.85, 0.40, "redrawn, pixel by pixel", op, strong=True, size=0.034)
        # route 1: through the code
        y = 0.70
        self.box(sc, 0.345, y, 104, 44, PAPER, op, "encoder", border=AXIS, size=0.034)
        for i, ink in enumerate(self.DIMS):
            self.box(sc, 0.46 + i * 0.033, y, 26, 44, ink, op, radius=4)
        self.text(sc, 0.51, y + 0.075, "the code: four numbers", op, ink=dark(VIOLET))
        self.box(sc, 0.665, y, 104, 44, VIOLET, op, "decoder", ink=WHITE, size=0.034)
        self.arrow(sc, (0.248, y), (0.285, y), AXIS, op)
        self.arrow(sc, (0.402, y), (0.44, y), AXIS, op)
        self.arrow(sc, (0.578, y), (0.606, y), VIOLET, op)
        beat = (seg(u, 0.12, 0.95) * 49) % 1.0
        self.arrow(sc, (0.722, y), (b(r, c)[0] - 0.02, b(r, c)[1]), VIOLET, op, width=2.4)
        self.pulse(sc, (0.578, y), (0.722, y), beat, VIOLET, op)
        # route 2: the earlier pixels, straight to the decoder
        self.arc(sc, (0.20, 0.445), (0.665, y - 0.05), BLUE, op, 1.0, bend=0.20, width=2.6)
        self.pulse(sc, (0.20, 0.445), (0.665, y - 0.05), beat, BLUE, op, bend=0.20)
        self.text(sc, 0.47, 0.335, "route 2: the earlier pixels", op, ink=dark(BLUE), size=0.034)
        self.text(sc, 0.51, y - 0.085, "route 1: the code", op, ink=dark(VIOLET), size=0.034)

    def _run(self, sc, u, op, which):
        name, ink, _prefix = self.RUNS[which]
        prog = seg(u, 0.05, 0.88)
        now = min(int(prog * (self.updates - 1)), self.updates - 1)
        # the code as four gauges: how much each number carries
        base, tall = 0.765, 168.0
        self.text(sc, 0.205, 0.325, "information in the code", op, strong=True, size=0.034)
        for i, colour in enumerate(self.DIMS):
            fx = 0.10 + i * 0.07
            value = float(self.kl[which][i, max(now - 10, 0):now + 1].mean())
            _w, dh = self.frac(0, tall)
            self.rect(sc, self.P(fx, base - dh / 2), 40, tall, PAPER, op, border=AXIS,
                      border_w=1.2, radius=5, lift=0.1)
            self.vbar(sc, fx, base, 34.0, max(tall * value / self.top, 1.0), colour, op)
            self.text(sc, fx, base + 0.04, "%d" % (i + 1), op, size=0.034, ink=dark(colour))
            self.text(sc, fx, base - dh - 0.035, "%.2f" % value, op, size=0.031)
        self.text(sc, 0.205, base + 0.085, "nats in each of the four numbers", op)
        self.text(sc, 0.205, base + 0.135, "%s: update %d, beta %.2f" % (
            name, now + 1, self.beta[which][now]), op, ink=dark(ink), size=0.032)
        steps = np.arange(1, self.updates + 1)
        box = (0.50, 0.945)
        chart = self.chart((box, (0.35, 0.57)), steps, [0.0, self.top], pad=0.04)
        self.axes(sc, chart, op, "", "nats in each number")
        for i, colour in enumerate(self.DIMS):
            self.plot(sc, chart, steps, self.kl[which][i], colour, op, prog, width=1.5)
        low = self.chart((box, (0.69, 0.87)), steps, [0.0, 4.0], pad=0.08)
        self.axes(sc, low, op, "training update", "pressure (beta)",
                  yticks=[(0, "0"), (1, "1"), (4, "4")])
        self.plot(sc, low, steps, self.beta[which], ink, op, prog, width=2.6)

    def phase_2(self, sc, u, op):
        self._run(sc, u, op, 0)

    def phase_3(self, sc, u, op):
        self._run(sc, u, op, 1)

    def phase_4(self, sc, u, op):
        rows = (("the digit", AXIS, self.images[:4]),
                ("strong pressure", RED, self.recon[0]), ("ramped", GREEN, self.recon[1]))
        for i, (label, ink, images) in enumerate(rows):
            y = 0.40 + i * 0.125
            self.text(sc, 0.05, y, label, op, anchor="lm", ink=None if i == 0 else dark(ink),
                      size=0.032)
            for k, img in enumerate(images):
                self.digit(sc, img, 0.235 + k * 0.066, y, 46, op, frame=ink)
        self.text(sc, 0.05, 0.76, "redrawn with the true earlier pixels given", op, anchor="lm")
        self.lines(sc, 0.05, 0.82, [
            "swap in another digit's code and the output moves by",
            ("%.5f under strong pressure" % self.swap[0], dark(RED)),
            ("%.4f with the ramp" % self.swap[1], dark(GREEN))],
            op * smoothstep(0.55, 0.7, u), step=0.044, size=0.031)
        # information left in each code number, both runs
        x0, x1, base, tall = 0.58, 0.93, 0.80, 140.0
        top = float(max(e.max() for e in self.left_each))
        self.text(sc, x0 - 0.03, 0.335, "nats left in each code number", op, anchor="lm",
                  strong=True, size=0.034)
        self.polyline(sc, [self.P(x0 - 0.03, base), self.P(x1 + 0.01, base)], 1.2, AXIS, op)
        for i in range(4):
            cx = x0 + (x1 - x0) * (i + 0.5) / 4
            self.text(sc, cx, base + 0.04, "number %d" % (i + 1), op, ink=dark(self.DIMS[i]))
            for k, (_name, ink, _prefix) in enumerate(self.RUNS):
                value = float(self.left_each[k][i])
                fx = cx + (k - 0.5) * 0.034
                w = seg(u, 0.1 + 0.25 * k, 0.4 + 0.25 * k)
                self.vbar(sc, fx, base, 26.0, tall * value / top, ink, op, w)
            _w, dh = self.frac(0, tall * max(e[i] for e in self.left_each) / top)
            self.text(sc, cx, base - dh - (0.035 if i % 2 else 0.10), "%.2f | %.2f" % (
                self.left_each[0][i], self.left_each[1][i]), op * smoothstep(0.65, 0.8, u))
        self.key(sc, x0 - 0.02, 0.385, RED, "strong pressure (beta 4)", op)
        self.key(sc, x0 + 0.22, 0.385, GREEN, "ramped to 1", op)



# ====================================================================== 65 vq-vae-codebook
VQ_SIDE = 8                            # the 64 codes drawn as an 8 x 8 sheet of tiles


def code_sheet(tiles, dim=()):
    """The codebook as one picture: each code's decoded 4 x 4 patch as a tile, in an
    8 x 8 sheet with one texel between tiles; codes in ``dim`` are drawn blank grey."""
    n = sg.VQ_PATCH
    step = n + 1
    sheet = np.empty((VQ_SIDE * step + 1, VQ_SIDE * step + 1, 3), np.float32)
    sheet[:] = np.asarray(AXIS) * 0.9
    for k, tile in enumerate(np.asarray(tiles, float)):
        r, c = divmod(k, VQ_SIDE)
        sheet[1 + r * step:1 + r * step + n, 1 + c * step:1 + c * step + n] = (
            np.asarray(TRACK) if k in dim else shade(tile.reshape(n, n)))
    return sheet


def vq_slope(enc, dec, codes, patch):
    """One patch through the quantised autoencoder: the encoder's output ``h``, the
    chosen code ``k`` and its vector ``z``, the output, and the slope of the patch's own
    loss at the code ``dz`` (which the straight-through rule hands to ``h`` unchanged)."""
    x = np.asarray(patch, float)[None]
    h, k, _ = sg.vq_apply(enc, dec, codes, x)
    out, tape = toy.forward(dec, codes[k])
    _g, dz = toy.backward(dec, tape, (sg.sigmoid(out) - x) / x.size)
    return dict(h=h[0], k=int(k[0]), z=codes[k[0]], out=sg.sigmoid(out)[0], dz=dz[0])


class VqVaeCodebook(Generative):
    """Vector quantisation on 28 x 28 MNIST: each 4 x 4 patch encoded and
    snapped to the nearest of 64 codes, the digit as 49 code numbers, the
    gradient copied across the snap, the codebook dying without its remedies,
    and what both codebooks end up using."""

    TITLE = "VQ-VAE: a codebook"
    PHASES = (("patch to code", 13.0), ("49 numbers", 10.0), ("copy the slope", 11.0),
              ("codes that die", 13.0), ("usage", 11.0))
    GRID = sg.VQ_GRID
    SHEET = (0.80, 0.60, 246.0)        # where the codebook is drawn in phases 1 and 4

    def setup(self):
        d = sg.arrays("vq-vae-codebook")
        self.originals = np.asarray(d["original"], float)            # 12 held-out digits
        self.recon = [np.asarray(d["remedies_recon"], float), np.asarray(d["control_recon"],
                                                                       float)]
        self.ids = np.asarray(d["remedies_ids"], int)
        self.tiles = [np.asarray(d["remedies_tiles"], float), np.asarray(d["control_tiles"],
                                                                       float)]
        self.counts = np.asarray(d["counts"], int)                   # arm, seed, code
        self.used = (self.counts > 0).sum(-1)
        self.mse = np.asarray(d["mse"], float)
        self.in_use = np.asarray(d["in_use"], int)                   # arm, seed, checkpoint
        self.steps = np.asarray(d["steps"], float)
        self.updates = int(self.steps[-1])
        self.blank = float(d["blank_share"])
        self.codes = len(self.tiles[0])
        self.patches = sg.vq_patches(self.originals.reshape(len(self.originals), -1))
        # the first digit's inked patches, in reading order: the ones phase 1 walks through
        self.walk = [int(i) for i in np.flatnonzero(self.patches[0].max(1) > 0.3)]
        w = weights(d, "remedies_weights_")
        self.enc, self.dec = sg.vq_split(w)
        self.codebook = np.asarray(d["remedies_codes"], float)
        self.one = self.walk[len(self.walk) // 2]
        self.q = vq_slope(self.enc, self.dec, self.codebook, self.patches[0, self.one])
        self.dead = [{int(i) for i in np.flatnonzero(self.counts[k, 0] == 0)}
                     for k in range(2)]

    def subtitle_for(self, k):
        used = self.used
        return [
            "a digit is cut into %d patches of 4 x 4 pixels; an encoder turns each patch "
            "into %d numbers, and the patch is replaced by whichever of the %d codes in "
            "the codebook lies nearest" % (self.GRID ** 2, sg.VQ_LATENT, self.codes),
            "so the whole digit becomes %d code numbers; the decoder is told nothing else, "
            "and rebuilds each patch from its code: every tile in the codebook is what one "
            "code decodes to" % self.GRID ** 2,
            "a snap has no slope to learn from, so the slope of the output's error at the "
            "code, here (%.4f, %.4f, ...), is copied unchanged to the encoder's output"
            % tuple(self.q["dz"][:2]),
            "a code no patch chooses is never moved: started at random, far from the "
            "encoder's outputs, only %s of %d codes stay in use (three seeds); with the "
            "remedies, %s" % (" and ".join((", ".join(str(v) for v in used[1][:-1]),
                                            str(used[1][-1]))), self.codes,
                              "all %d do" % self.codes if (used[0] == self.codes).all()
                              else ", ".join(str(v) for v in used[0])),
            "on 1,000 held-out digits the remedied codebook uses all %d codes and errs by "
            "%.3f per pixel; without the remedies %d to %d codes do all the work and the "
            "error is %.3f to %.3f" % (self.codes, self.mse[0].mean(), used[1].min(),
                                       used[1].max(), self.mse[1].min(), self.mse[1].max()),
        ][k]

    # ------------------------------------------------------------------ drawing
    def _sheet(self, sc, op, arm=0, dim=(), ring=None, at=None, label=None):
        """The codebook as a sheet of tiles; returns ``centre(code) -> (fx, fy)``."""
        fx, fy, side = at or self.SHEET
        c = self.P(fx, fy)
        self.rect(sc, c, side + 8, side + 8, PAPER, op, border=AXIS, border_w=1.2, radius=5,
                  lift=0.1)
        draw_image(self, sc, c, code_sheet(self.tiles[arm], dim), side, side, op, grid=0)
        dw, dh = self.frac(side, side)
        texels = VQ_SIDE * (sg.VQ_PATCH + 1) + 1

        def centre(k):
            r, col = divmod(int(k), VQ_SIDE)
            return (fx - dw / 2 + dw * (col * 5 + 3) / texels,
                    fy - dh / 2 + dh * (r * 5 + 3) / texels)
        if ring is not None:
            tile = side * 5 / texels
            self.rect(sc, self.P(*centre(ring)), tile + 3, tile + 3, AMBER, op, border=AMBER,
                      border_w=3.0, radius=2, lift=0.9, fill_op=0.0)
        if label:
            self.text(sc, fx, fy - dh / 2 - 0.04, label, op, size=0.034, strong=True)
        return centre

    def _cut(self, sc, img, fx, fy, side, op, ring=None, frame=AXIS):
        """A 28 x 28 digit with its 7 x 7 patch grid; returns ``centre(patch)``."""
        self.digit(sc, img, fx, fy, side, op, frame=frame)
        dw, dh = self.frac(side, side)
        with self.over(0.5):
            for k in range(1, self.GRID):
                f = k / self.GRID
                self.polyline(sc, [self.P(fx - dw / 2 + dw * f, fy - dh / 2),
                                   self.P(fx - dw / 2 + dw * f, fy + dh / 2)], 0.8, TRACK, op)
                self.polyline(sc, [self.P(fx - dw / 2, fy - dh / 2 + dh * f),
                                   self.P(fx + dw / 2, fy - dh / 2 + dh * f)], 0.8, TRACK, op)

        def centre(i):
            r, col = divmod(int(i), self.GRID)
            return (fx - dw / 2 + dw * (col + 0.5) / self.GRID,
                    fy - dh / 2 + dh * (r + 0.5) / self.GRID)
        if ring is not None:
            self.rect(sc, self.P(*centre(ring)), side / self.GRID + 2, side / self.GRID + 2,
                      AMBER, op, border=AMBER, border_w=3.0, radius=2, lift=0.9, fill_op=0.0)
        return centre

    def _patch(self, sc, values, fx, fy, size, op, frame):
        self.pixels(sc, shade(np.asarray(values).reshape(sg.VQ_PATCH, sg.VQ_PATCH)), fx, fy,
                    size, op, frame=frame)

    def phase_1(self, sc, u, op):
        at = seg(u, 0.06, 0.94) * len(self.walk)
        i = min(int(at), len(self.walk) - 1)
        within = at - i
        patch = self.walk[i]
        code = int(self.ids[0].ravel()[patch])
        cut = self._cut(sc, self.originals[0], 0.155, 0.60, 224, op, ring=patch)
        self.text(sc, 0.155, 0.335, "a digit: %d patches" % self.GRID ** 2, op, strong=True,
                  size=0.034)
        centre = self._sheet(sc, op, ring=code if within > 0.55 else None,
                             label="the codebook: %d codes" % self.codes)
        # the patch, the encoder, its numbers, the nearest code
        y = 0.50
        self._patch(sc, self.patches[0, patch], 0.345, y, 60, op, AMBER)
        self.text(sc, 0.345, y - 0.105, "patch %d" % (patch + 1), op)
        self.arrow(sc, (0.385, y), (0.415, y), BLUE, op, width=2.2)
        self.box(sc, 0.47, y, 96, 44, BLUE, op, "encoder", ink=WHITE, size=0.034)
        self.arrow(sc, (0.47, y + 0.05), (0.47, y + 0.115), BLUE, op, width=2.2)
        self.box(sc, 0.47, y + 0.16, 150, 40, soft(BLUE), op, "%d numbers" % sg.VQ_LATENT,
                 border=BLUE, size=0.034)
        self.arrow(sc, (0.47, y + 0.205), (0.47, y + 0.265), AMBER, op, width=2.2)
        self.box(sc, 0.47, y + 0.31, 170, 40, soft(AMBER), op * smoothstep(0.4, 0.55, within),
                 "nearest: code %d" % code, border=AMBER, size=0.034)
        with self.over():
            self.arc(sc, cut(patch), (0.31, y), AMBER, op * 0.8, 1.0, bend=0.03, width=1.6)
            self.arc(sc, (0.56, y + 0.31), centre(code), AMBER,
                     op * smoothstep(0.5, 0.6, within), 1.0, bend=0.04, width=1.8)
        self.pulse(sc, (0.385, y), (0.47, y + 0.31), seg(within, 0.05, 0.5), BLUE, op,
                   bend=-0.06)
        self.text(sc, 0.345, 0.90, "patch %d of the %d with ink" % (i + 1, len(self.walk)), op,
                  anchor="lm")

    def phase_2(self, sc, u, op):
        y = 0.60
        self._cut(sc, self.originals[0], 0.115, y, 170, op)
        self.text(sc, 0.115, 0.36, "784 pixels", op, strong=True, size=0.034)
        # the grid of code numbers
        side = 210.0
        dw, dh = self.frac(side, side)
        fx = 0.375
        self.rect(sc, self.P(fx, y), side + 8, side + 8, soft(AMBER, 0.75), op, border=AMBER,
                  border_w=1.4, radius=5, lift=0.1)
        n = self.GRID ** 2
        told = int(seg(u, 0.05, 0.45) * n + 0.999)
        built = int(seg(u, 0.45, 0.9) * n + 0.999)
        for i in range(min(told, n)):
            r, c = divmod(i, self.GRID)
            self.text(sc, fx - dw / 2 + dw * (c + 0.5) / self.GRID,
                      y - dh / 2 + dh * (r + 0.5) / self.GRID, str(self.ids[0, r, c]), op,
                      ink=dark(AMBER, 0.35))
        self.text(sc, fx, 0.36, "%d code numbers" % n, op, ink=dark(AMBER), size=0.034)
        self.arrow(sc, (0.212, y), (0.255, y), BLUE, op, width=2.4)
        self.arrow(sc, (0.493, y), (0.513, y), VIOLET, op, width=2.4)
        self.box(sc, 0.562, y, 82, 44, VIOLET, op, "decoder", ink=WHITE, size=0.031)
        self.arrow(sc, (0.608, y), (0.626, y), VIOLET, op, width=2.4)
        # the rebuilt digit, a patch at a time
        rebuilt = sg.vq_patches(self.recon[0][:1].reshape(1, -1))[0].copy()
        rebuilt[built:] = 0.0
        self._cut(sc, sg.vq_unpatch(rebuilt[None])[0], 0.72, y, 170, op, frame=VIOLET)
        self.text(sc, 0.72, 0.36, "rebuilt from the codes", op, ink=dark(VIOLET), size=0.034)
        # three more held-out digits, in and out
        for row in range(3):
            fy = 0.455 + row * 0.145
            self.digit(sc, self.originals[row + 1], 0.872, fy, 42, op, frame=AXIS)
            self.digit(sc, self.recon[0][row + 1], 0.926, fy, 42, op * smoothstep(0.6, 0.8, u),
                       frame=VIOLET)
        self.text(sc, 0.899, 0.36, "in, out", op)
        self.text(sc, 0.05, 0.90, "each number names one of the %d tiles of the codebook" % (
            self.codes), op * smoothstep(0.5, 0.65, u), anchor="lm", size=0.032)

    def phase_3(self, sc, u, op):
        q = self.q
        def pair(v):
            return "(%.2f, %.2f, ...)" % tuple(v[:2])

        wide = 186
        y = 0.43
        self.text(sc, 0.05, 0.325, "forward", op, anchor="lm", strong=True, size=0.034)
        self._patch(sc, self.patches[0, self.one], 0.085, y, 52, op, AXIS)
        self.box(sc, 0.20, y, 104, 44, BLUE, op, "encoder", ink=WHITE, size=0.034)
        self.box(sc, 0.365, y, 164, 44, soft(BLUE), op, pair(q["h"]), border=BLUE, size=0.031)
        self.box(sc, 0.615, y, 164, 44, soft(AMBER), op, pair(q["z"]), border=AMBER,
                 size=0.031)
        self.box(sc, 0.78, y, 104, 44, VIOLET, op, "decoder", ink=WHITE, size=0.034)
        self._patch(sc, q["out"], 0.90, y, 52, op, AMBER)
        self.text(sc, 0.365, y - 0.07, "the patch's %d numbers" % sg.VQ_LATENT, op,
                  ink=dark(BLUE))
        self.text(sc, 0.615, y - 0.07, "code %d" % q["k"], op, ink=dark(AMBER))
        for a, b, ink in (((0.122, y), (0.14, y), AXIS), ((0.258, y), (0.275, y), BLUE),
                          ((0.705, y), (0.722, y), AMBER), ((0.838, y), (0.862, y), VIOLET)):
            self.arrow(sc, a, b, ink, op, width=2.0)
        # the snap, drawn as the step it is
        sx = 0.49
        self.polyline(sc, [self.P(sx - 0.03, y + 0.022), self.P(sx - 0.01, y + 0.022),
                           self.P(sx - 0.01, y), self.P(sx + 0.01, y), self.P(sx + 0.01, y - 0.022),
                           self.P(sx + 0.03, y - 0.022)], 2.6, INK, op)
        self.text(sc, sx, y - 0.07, "snap", op, strong=True)
        self.pulse(sc, (0.14, y), (0.86, y), seg(u, 0.02, 0.3), BLUE, op)
        # backward: the slope reaches the code, and is copied across the step
        yb = 0.70
        back = op * smoothstep(0.3, 0.4, u)
        self.text(sc, 0.05, 0.60, "backward", back, anchor="lm", strong=True, size=0.034)
        slope = "(%.4f, %.4f, ...)" % tuple(q["dz"][:2])
        self.box(sc, 0.615, yb, wide, 44, soft(RED), back, slope, border=RED, size=0.031)
        self.arc(sc, (0.78, y + 0.05), (0.69, yb - 0.045), RED, back, seg(u, 0.32, 0.45),
                 bend=-0.04, width=2.4)
        self.text(sc, 0.80, yb, "how the output's error", back, anchor="lm", ink=dark(RED))
        self.text(sc, 0.80, yb + 0.045, "changes with the code", back, anchor="lm",
                  ink=dark(RED))
        copy = seg(u, 0.5, 0.72)
        self.arc(sc, (0.512, yb), (0.468, yb), RED, back, copy, bend=0.0, width=2.6)
        self.pulse(sc, (0.512, yb), (0.468, yb), copy, RED, op)
        got = op * smoothstep(0.7, 0.8, u)
        self.box(sc, 0.365, yb, wide, 44, soft(RED), got, slope, border=RED, size=0.031)
        self.text(sc, 0.49, yb - 0.075, "copied unchanged", back, ink=dark(RED), size=0.032)
        self.arc(sc, (0.365, yb - 0.045), (0.215, y + 0.05), RED, got, seg(u, 0.74, 0.86),
                 bend=-0.03, width=2.4)
        self.lines(sc, 0.05, 0.83, [
            "the encoder learns as if the snap were not there;",
            "only a code that some patch chose is moved, toward the outputs that chose it"],
            op * smoothstep(0.84, 0.94, u), step=0.046, size=0.032)

    def phase_4(self, sc, u, op):
        prog = seg(u, 0.08, 0.8)
        steps = self.steps
        chart = self.chart(((0.09, 0.50), (0.37, 0.70)), steps, [0, self.codes], pad=0.04)
        self.axes(sc, chart, op, "training update", "codes in use, of %d" % self.codes,
                  yticks=[(v, str(v)) for v in (0, 16, 32, 48, 64)])
        for arm, ink in ((0, GREEN), (1, RED)):
            for seed in range(self.in_use.shape[1]):
                self.plot(sc, chart, steps, self.in_use[arm, seed], ink, op, prog, width=2.0,
                          dot=seed == 0)
        k = int(round(prog * (len(steps) - 1)))
        self.text(sc, 0.51, chart.fy(self.in_use[0, :, k].mean()), "with the remedies", op,
                  anchor="lm", ink=dark(GREEN), size=0.032)
        self.text(sc, 0.51, chart.fy(self.in_use[1, :, k].mean()) - 0.02, "without", op,
                  anchor="lm", ink=dark(RED), size=0.032)
        self.lines(sc, 0.06, 0.815, [
            ("remedies: start each code on an encoder output; move it to the mean of", None),
            ("the outputs that choose it; restart a code that goes unchosen", None),
            ("three seeds each, counted on 64 held-out digits", None)], op, step=0.042)
        done = smoothstep(0.78, 0.9, u)
        at = (0.83, 0.58, 190.0)
        self._sheet(sc, op, arm=1, dim=self.dead[1] if done > 0.5 else (), at=at,
                    label="without: the codebook")
        self.text(sc, at[0], 0.82, "%d of %d codes ever chosen" % (
            self.used[1, 0], self.codes), op * done, ink=dark(RED), size=0.034)
        self.text(sc, at[0], 0.865, "grey: never chosen", op * done)

    def _usage(self, sc, arm, fx0, fx1, base, tall, ink, op, u):
        """Held-out patches per code, most used first, on a log scale."""
        counts = np.sort(self.counts[arm, 0])[::-1]
        top = np.log10(self.counts.max() + 1.0)
        for k, count in enumerate(counts):
            cx = fx0 + (fx1 - fx0) * (k + 0.5) / len(counts)
            if count:
                self.vbar(sc, cx, base, 4.2, max(tall * np.log10(count + 1.0) / top, 2.0), ink,
                          op, u)
            else:
                self.rect(sc, self.P(cx, base - 0.006), 3.4, 3.4, DIM, op, radius=1)
        self.polyline(sc, [self.P(fx0 - 0.005, base), self.P(fx1 + 0.005, base)], 1.2, AXIS, op)

    def phase_5(self, sc, u, op):
        # the same held-out digits rebuilt by each codebook
        names = (("input", AXIS, None), ("with remedies", GREEN, 0), ("without", RED, 1))
        for row, (label, ink, arm) in enumerate(names):
            y = 0.42 + row * 0.165
            self.text(sc, 0.05, y, label, op, anchor="lm", ink=None if arm is None else dark(ink),
                      size=0.032)
            for col in range(4):
                img = self.originals[col] if arm is None else self.recon[arm][col]
                self.digit(sc, img, 0.235 + col * 0.082, y, 68, op, frame=ink)
        self.text(sc, 0.05, 0.895, "error per pixel: %.3f with, %.3f to %.3f without" % (
            self.mse[0].mean(), self.mse[1].min(), self.mse[1].max()), op, anchor="lm")
        x0, x1 = 0.60, 0.94
        self.text(sc, x0 - 0.01, 0.335, "held-out patches choosing each code", op, anchor="lm",
                  strong=True)
        for arm, (base, ink, name) in enumerate(((0.565, GREEN, "with the remedies"),
                                                 (0.82, RED, "without"))):
            self._usage(sc, arm, x0, x1, base, 78.0, ink, op, seg(u, 0.05 + 0.3 * arm,
                                                                  0.45 + 0.3 * arm))
            self.text(sc, x1, base - 0.165, "%s: %s of %d used" % (
                name, ", ".join(str(v) for v in self.used[arm]), self.codes), op, anchor="rm",
                ink=dark(ink), size=0.032)
        self.text(sc, x0 - 0.01, 0.865, "codes, most used first; log scale", op, anchor="lm")
        self.text(sc, x0 - 0.01, 0.905, "tallest: the blank patch, %.0f%% of all" % (
            100 * self.blank), op, anchor="lm")


# ====================================================================== 66 autoregressive-pixels
MASK_A = np.array([[1, 1, 1], [1, 0, 0], [0, 0, 0]])     # a first layer: not the pixel itself
MASK_B = np.array([[1, 1, 1], [1, 1, 0], [0, 0, 0]])     # later layers: their own position too


def pixel_chances(p, images, dilations=sg.PIX_DEEP):
    """The model's chance of ink for every pixel of ``images`` ``(n, 14, 14)``, each
    from the earlier pixels its masks let it read."""
    x = np.asarray(images, np.float32)[..., None]
    return sg.sigmoid(sg.pix_forward(p, x, dilations)[0])


class AutoregressivePixels(Generative):
    """An image as a sequence: the reading order, the masks that hide the
    pixel itself and everything after it, the blind spot a single stack of
    them leaves and the second stream that closes it, all 196 predictions
    trained at once, and digits sampled pixel by pixel."""

    TITLE = "Autoregressive pixels"
    PHASES = (("reading order", 8.0), ("the masks", 14.0), ("training", 9.0),
              ("sampling", 14.0), ("samples", 10.0))
    SIDE = sg.PIX_SIDE
    TARGET = sg.PIX_TARGET             # the pixel whose view phase 2 draws
    FOLLOWED = 0                       # the study's first sample, drawn pixel by pixel

    def setup(self):
        d = sg.arrays("autoregressive-pixels")
        s = self.study
        n = self.SIDE
        self.p = weights(d)
        self.loss = np.asarray(d["loss"], float)                    # seed, update: bits
        self.samples = np.asarray(d["samples"], float)
        self.chances = np.asarray(d["chances"], float)
        self.shallow = np.asarray(d["shallow_samples"], float)
        self.real = np.asarray(d["real"], float)
        self.real_chances = pixel_chances(self.p, self.real[:1])[0]
        self.held = np.asarray(d["held_bits"], float)               # (six, one) x seed
        self.independent = float(d["independent"])
        self.bars = [float(v) for v in s.panels[3].data["values"]]
        r, c = self.TARGET
        order = np.arange(n * n).reshape(n, n)
        self.order = order
        self.earlier = order < r * n + c
        depths = sg.PIX_DEEP
        self.views = [sg.plain_seen(depths[:k + 1])[r, c] for k in range(len(depths))]
        self.never = self.earlier & ~sg.plain_seen((1,) * (4 * n))[r, c]   # at any depth
        self.two = sg.pix_seen(depths)[r, c]
        self.closed = int((self.never & self.two).sum())

    def subtitle_for(self, k):
        n = self.SIDE * self.SIDE
        return [
            "an image read as a sequence: %d pixels, left to right, top to bottom; the "
            "model predicts each pixel from the ones before it, like the next word of a "
            "sentence" % n,
            "a masked filter hides the pixel itself and everything after it; stacking such "
            "layers widens the view, but %d of this pixel's %d earlier pixels are never "
            "read at any depth, so a second stream reads the rows above" % (
                int(self.never.sum()), int(self.earlier.sum())),
            "training shows it real digits and asks for all %d pixels at once; six layers "
            "reach %.3f bits per held-out pixel, against %.3f when each pixel is guessed "
            "on its own" % (n, self.held[0].mean(), self.independent),
            "sampling goes one pixel at a time: the model gives a chance of ink, a random "
            "draw decides, and the result becomes input for every pixel after it",
            "sampled digits beside real ones. Held out, three seeds each: %.3f bits per "
            "pixel with six layers, %.3f with one, %.3f with each pixel on its own" % (
                self.held[0].mean(), self.held[1].mean(), self.independent),
        ][k]

    # ------------------------------------------------------------------ drawing
    def _cells(self, values, tint=None, unknown=None):
        """A texture: ink for ``values``; ``tint``: ``[(boolean map, colour)]`` mixed in;
        ``unknown``: a boolean map drawn as blank grey."""
        tex = shade(values)
        if unknown is not None:
            tex[unknown] = PAPER * 0.86
        for where, colour in tint or ():
            tex[where] = tex[where] * 0.4 + np.asarray(colour) * 0.6
        return tex

    def phase_1(self, sc, u, op):
        img = self.real[0]
        side = self.SIDE
        n = side * side
        k = min(int(seg(u, 0.1, 0.92) * n), n - 1)
        r, c = divmod(k, side)
        cell = self.pixels(sc, self._cells(img, [(self.order < k, BLUE)], self.order > k), 0.19,
                           0.565, 210, op, ring=[(r, c)])
        self.text(sc, 0.19, 0.32, "a digit, %d x %d pixels" % (side, side), op, strong=True,
                  size=0.034)
        # the same pixels unrolled into the sequence they are read as
        x0, x1, y = 0.07, 0.93, 0.865
        flat = self._cells(img, [(self.order < k, BLUE)], self.order > k).reshape(1, n, 3)
        draw_image(self, sc, self.P((x0 + x1) / 2, y), flat, self.span(x1 - x0)[0], 20, op,
                   grid=0)
        fx = x0 + (x1 - x0) * (k + 0.5) / n
        self.rect(sc, self.P(fx, y), 6, 26, AMBER, op, border=AMBER, border_w=2.2, radius=1,
                  lift=0.8, fill_op=0.0)
        self.text(sc, x0, y + 0.05, "the same pixels as one sequence", op, anchor="lm")
        self.text(sc, x1, y + 0.05, "pixel %d of %d" % (k + 1, n), op, anchor="rm", size=0.034,
                  strong=True)
        with self.over():
            self.arc(sc, cell(r, c), (fx, y - 0.03), AMBER, op, 1.0, bend=-0.04, width=1.8)
        x = 0.42
        self.key(sc, x, 0.42, BLUE, "already read: may be used", op, size=0.034)
        self.rect(sc, self.P(x, 0.50), 14, 14, PAPER, op, border=AMBER, border_w=2.6, radius=2)
        self.text(sc, x + 0.016, 0.50, "being predicted now", op, anchor="lm", size=0.034)
        self.rect(sc, self.P(x, 0.58), 14, 14, PAPER * 0.86, op, border=AXIS, radius=2)
        self.text(sc, x + 0.016, 0.58, "not yet: must stay hidden", op, anchor="lm", size=0.034)

    def _mask(self, sc, mask, fx, fy, side, op, label):
        tex = np.where(mask[..., None] > 0, np.asarray(BLUE) * 0.75 + PAPER * 0.25, PAPER * 0.86)
        cell = self.pixels(sc, tex, fx, fy, side, op, ring=[(1, 1)])
        for (rr, cc), mark in np.ndenumerate(mask):
            self.text(sc, *cell(rr, cc), "1" if mark else "0", op, size=0.034,
                      ink=WHITE if mark else AXIS)
        self.text(sc, fx, fy - self.frac(0, side / 2)[1] - 0.04, label, op)

    def phase_2(self, sc, u, op):
        r, c = self.TARGET
        depths = len(self.views)
        stage = min(int(seg(u, 0.14, 0.56) * depths), depths - 1) if u >= 0.14 else 0
        blind, second = u > 0.6, u > 0.78
        view = self.views[stage] & self.earlier
        tint = [(view, BLUE)]
        if blind:
            tint.append((self.never, RED))
        if second:
            tint = [(self.two & self.earlier, GREEN)]
        self.pixels(sc, self._cells(self.real[0], tint, ~self.earlier), 0.19, 0.60, 224, op,
                    ring=[(r, c)])
        self.text(sc, 0.19, 0.335, "what the ringed pixel may read", op, strong=True, size=0.034)
        # the two masks of a stack
        self._mask(sc, MASK_A, 0.445, 0.50, 84, op, "first layer")
        self._mask(sc, MASK_B, 0.585, 0.50, 84, op, "later layers")
        self.text(sc, 0.515, 0.655, "1: read   0: hidden", op)
        self.text(sc, 0.515, 0.70, "ringed: the pixel's own place", op, ink=dark(AMBER))
        x = 0.70
        earlier = int(self.earlier.sum())
        if not second:
            self.text(sc, x, 0.40, "layers stacked", op, anchor="lm", size=0.034)
            self.text(sc, x + 0.17, 0.40, str(stage + 1), op, anchor="lm", size=0.05, strong=True)
            self.text(sc, x, 0.48, "earlier pixels read", op, anchor="lm", size=0.034,
                      ink=dark(BLUE))
            self.text(sc, x, 0.54, "%d of %d" % (int(view.sum()), earlier), op, anchor="lm",
                      size=0.05, strong=True)
            shown = op * smoothstep(0.6, 0.68, u)
            self.text(sc, x, 0.64, "never read, at any depth", shown, anchor="lm", size=0.034,
                      ink=dark(RED))
            self.text(sc, x, 0.70, "%d: the blind spot" % int(self.never.sum()), shown,
                      anchor="lm", size=0.05, strong=True)
        else:
            self.lines(sc, x, 0.40, [("with a second stream", dark(GREEN)),
                                     "that reads the rows above"], op, step=0.046, size=0.034)
            self.text(sc, x, 0.54, "%d of %d" % (int((self.two & self.earlier).sum()), earlier),
                      op, anchor="lm", size=0.05, strong=True)
            self.lines(sc, x, 0.64, ["read by six layers,", "with %d of the %d" % (
                self.closed, int(self.never.sum())), "blind-spot pixels"], op, step=0.046,
                size=0.034)
        self.text(sc, 0.42, 0.84, "this model: six layers, two streams, 32 channels", op,
                  anchor="lm")
        self.text(sc, 0.42, 0.885, "(the reach is exact arithmetic on the masks)", op,
                  anchor="lm")

    def phase_3(self, sc, u, op):
        img, chance = self.real[0], self.real_chances
        self.pixels(sc, shade(img), 0.13, 0.56, 150, op)
        self.text(sc, 0.13, 0.36, "a real digit", op, strong=True, size=0.034)
        n = self.SIDE ** 2
        k = int(seg(u, 0.1, 0.35) * n)
        self.pixels(sc, self._cells(chance, None, self.order >= k if k < n else None), 0.335,
                    0.56, 150, op, frame=VIOLET)
        self.text(sc, 0.335, 0.36, "chance of ink, each pixel", op, ink=dark(VIOLET), size=0.034)
        self.arrow(sc, (0.22, 0.56), (0.245, 0.56), VIOLET, op, width=2.4)
        self.lines(sc, 0.05, 0.775, ["every pixel is predicted in one pass,",
                                     "each from the earlier pixels only;",
                                     "the masks stop any of them seeing itself"], op,
                   step=0.046, size=0.032)
        steps = np.arange(1, self.loss.shape[1] + 1)
        smooth = np.array([np.convolve(row, np.ones(25) / 25, mode="same") for row in self.loss])
        chart = self.chart(((0.555, 0.945), (0.38, 0.78)), steps, [0.15, 0.5], pad=0.0)
        self.axes(sc, chart, op, "training update (three seeds, smoothed)", "bits per pixel")
        y = chart.fy(self.independent)
        self.polyline(sc, [self.P(chart.x0, y), self.P(chart.x1, y)], 1.6, AMBER, op)
        self.text(sc, chart.x1, y - 0.03, "each pixel on its own: %.3f" % self.independent, op,
                  anchor="rm", ink=dark(AMBER))
        for seed, row in enumerate(smooth):
            self.plot(sc, chart, steps[12:-12], np.minimum(row[12:-12], 0.5), VIOLET, op,
                      seg(u, 0.3, 0.9), width=1.5, dot=seed == 0)

    def _partial(self, which, k):
        """Sample ``which`` with its first ``k`` pixels drawn."""
        return self.samples[which], self.order >= k

    def phase_4(self, sc, u, op):
        side = self.SIDE
        n = side * side
        at = seg(u, 0.04, 0.92) * n
        k = min(int(at), n - 1)
        within = at - k if at < n else 1.0
        r, c = divmod(k, side)
        drawn = k + (1 if within > 0.6 else 0)
        values, unknown = self._partial(self.FOLLOWED, drawn)
        self.pixels(sc, self._cells(values, None, unknown), 0.19, 0.60, 224, op,
                    ring=[(r, c)] if at < n else ())
        self.text(sc, 0.19, 0.335, "a sample, pixel %d of %d" % (k + 1, n), op, strong=True,
                  size=0.034)
        chance = float(self.chances[self.FOLLOWED][r, c])
        ink = bool(self.samples[self.FOLLOWED][r, c])
        x = 0.42
        self.text(sc, x, 0.42, "the model's chance of ink", op, anchor="lm", size=0.034,
                  ink=dark(VIOLET))
        full = self.span(0.22)[0]
        self.hbar(sc, x, 0.485, max(full * chance, 2.0), VIOLET, op, "", "%.2f" % chance,
                  thick=26.0, track=full)
        self.text(sc, x, 0.585, "the random draw", op, anchor="lm", size=0.034)
        self.box(sc, x + 0.06, 0.655, 110, 44, INK if ink else PAPER,
                 op * smoothstep(0.45, 0.6, within), "ink" if ink else "blank",
                 ink=WHITE if ink else INK, border=AXIS, size=0.036)
        self.text(sc, x, 0.76, "each chance comes from every", op, anchor="lm")
        self.text(sc, x, 0.804, "pixel already drawn above and left", op, anchor="lm")
        others = [i for i in range(4) if i != self.FOLLOWED]
        self.text(sc, 0.835, 0.325, "three more, same pace", op)
        for row, which in enumerate(others):
            v, hidden = self._partial(which, drawn)
            self.pixels(sc, self._cells(v, None, hidden), 0.835, 0.44 + row * 0.17, 72, op)

    def phase_5(self, sc, u, op):
        rows = (("sampled, six layers", self.samples, VIOLET, 0.0),
                ("sampled, one layer", self.shallow, AMBER, 0.2),
                ("real digits", self.real, AXIS, 0.35))
        for row, (label, pics, ink, when) in enumerate(rows):
            y = 0.40 + row * 0.15
            shown = op * smoothstep(when, when + 0.15, u)
            self.text(sc, 0.05, y, label, shown, anchor="lm", size=0.032,
                      ink=None if ink is AXIS else dark(ink))
            for i in range(8):
                self.digit(sc, pics[i], 0.30 + i * 0.076, y, 56, shown, frame=ink)
        full = self.span(0.20)[0]
        labels = ("each pixel on its own", "one masked layer", "six masked layers")
        for i, (label, value, ink) in enumerate(zip(labels, self.bars, (DIM, AMBER, VIOLET))):
            self.hbar(sc, 0.30, 0.825 + i * 0.047, full * value / 0.4, ink, op, label,
                      "%.3f" % value, seg(u, 0.45 + 0.1 * i, 0.7 + 0.1 * i), thick=17.0,
                      track=full)
        self.lines(sc, 0.63, 0.85, [("held-out bits per pixel,", None), "lower is better"], op,
                   step=0.046, size=0.034)


ANIMATIONS = {
    "gan-training": GanTraining,
    "normalizing-flows": NormalizingFlows,
    "flow-matching": FlowMatching,
    "classifier-free-guidance": ClassifierFreeGuidance,
    "ddpm-vs-ddim": DdpmVsDdim,
    "vae-posterior-collapse": VaePosteriorCollapse,
    "vq-vae-codebook": VqVaeCodebook,
    "autoregressive-pixels": AutoregressivePixels,
}

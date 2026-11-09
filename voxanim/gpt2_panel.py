"""Matrices as flat textured panels, and the colour maps that fill them.

Used by :mod:`voxanim.gpt2_anim`.  A :class:`Panel` is an ``R x C`` texture on
a parallelogram in model space.  Its texture is mip-mapped *per axis*, so a
6 x 3072 matrix neither shimmers when small (columns averaged) nor blurs its
6 rows (rows kept).  Big cells get a faint grid.
"""

from __future__ import annotations

import math

import numpy as np

from . import color as C
from .netanim import ACT_HI, ACT_LO, OFF

DIAG = np.array([1.0, 0.0, -1.0])          # flipping about this swaps rows and columns


def bump(x, width):
    """Gaussian bump ``exp(-(x / width)^2)``: 1 at 0, ~0.37 at ``width``."""
    return math.exp(-(x / width) ** 2)


def rotate(v, axis, theta):
    """Rodrigues: rotate ``v`` (3,) about ``axis`` (3,) by ``theta`` radians."""
    k = np.asarray(axis, np.float64) / np.linalg.norm(axis)
    return v * math.cos(theta) + np.cross(k, v) * math.sin(theta) + \
        k * np.dot(k, v) * (1 - math.cos(theta))


# ------------------------------------------------------------------ colours


def diverging(a):
    """Signed values -> RGB float32 ``a.shape + (3,)``: NEGATIVE, NEUTRAL at 0, POSITIVE.

    Scaled so the 99th percentile of ``|a|`` is full colour (outliers clip),
    with a 0.75 gamma so small values still show.
    """
    a = np.asarray(a, np.float64)
    m = float(np.percentile(np.abs(a), 99)) or 1.0
    v = np.clip(a / m, -1, 1)
    t = (np.abs(v) ** 0.75)[..., None]
    tgt = np.where(v[..., None] >= 0, C.POSITIVE, C.NEGATIVE)
    return (C.NEUTRAL + (tgt - C.NEUTRAL) * t).astype(np.float32)


def lit(a, vmax=None):
    """Activations -> RGB float32: only values > 0 light up (ACT_LO to ACT_HI).

    ``vmax`` (default: the 99th percentile of the positive values) is full
    brightness; values at or below ``1e-3 * vmax`` are OFF.
    """
    a = np.asarray(a, np.float64)
    if vmax is None:
        vmax = float(np.percentile(a[a > 0], 99)) if (a > 0).any() else 1.0
    t = np.clip(a / vmax, 0, 1) ** 0.6
    out = ACT_LO + (ACT_HI - ACT_LO) * t[..., None]
    out[a <= 1e-3 * vmax] = OFF
    return out.astype(np.float32)


# ------------------------------------------------------------------ panels


def _halve(t, axis):
    """Average neighbouring pairs along ``axis`` (an odd last entry is repeated)."""
    if t.shape[axis] % 2:
        t = np.concatenate([t, np.take(t, [-1], axis)], axis)
    return (np.take(t, range(0, t.shape[axis], 2), axis) +
            np.take(t, range(1, t.shape[axis], 2), axis)) / 2


def mip(tex, lr, lc):
    """``tex`` (R, C, 3) averaged over ``2^lr`` rows and ``2^lc`` columns.

    An axis stops shrinking at length 1.
    """
    t = tex
    for axis, lev in ((0, lr), (1, lc)):
        for _ in range(lev):
            if t.shape[axis] < 2:
                break
            t = _halve(t, axis)
    return t


class Panel:
    """A matrix drawn as a textured parallelogram.

    ``colors`` is ``(R, C, 3)`` RGB.  ``origin`` is its top-left corner in
    model space; columns run along +x and rows down -z, each cell ``cell =
    (row height, column width)`` model units.  :meth:`draw` maps the panel
    through the animation's world zoom (``anim.Z``, ``anim.s``).
    """

    def __init__(self, colors, origin, cell=(1.0, 1.0)):
        self.tex = np.asarray(colors, np.float32)
        self.R, self.C = self.tex.shape[:2]
        self.origin = np.asarray(origin, np.float64)
        self.e1 = np.array([self.C * cell[1], 0.0, 0.0])      # along a row (model units)
        self.e2 = np.array([0.0, 0.0, -self.R * cell[0]])     # down a column
        self._mips = {}

    def box(self):
        """``(lo, hi)``: the model-space bounding box, each (3,)."""
        p = np.array([self.origin, self.origin + self.e1, self.origin + self.e2,
                      self.origin + self.e1 + self.e2])
        return p.min(0), p.max(0)

    def mip(self, lr, lc, tex=None):
        """The texture (or ``tex``) averaged over ``2^lr`` rows and ``2^lc`` columns.

        Levels of the panel's own texture are cached.
        """
        if tex is not None:
            return mip(tex, lr, lc)
        if (lr, lc) not in self._mips:
            self._mips[lr, lc] = mip(self.tex, lr, lc)
        return self._mips[lr, lc]

    def draw(self, anim, sc, opacity=1.0, highlight=None, tex=None, alpha=None,
             e1=None, e2=None, origin=None, lift=0.0, glow=0.0):
        """Add the panel to scene ``sc`` as one shaded quad.

        ``highlight`` (scalar or ``(R, C)``) blends towards white; ``alpha``
        ``(R, C)`` is per-cell opacity; ``tex`` replaces the colours this
        frame.  ``e1`` / ``e2`` / ``origin`` override the model-space edges
        and corner (to move or turn the panel); ``lift`` moves it towards
        the viewer (world units, after the zoom) so it draws on top;
        ``glow`` feeds the bloom.  Skipped when transparent, edge-on or off
        screen.
        """
        if opacity <= 0.01:
            return
        cam = anim.cam
        p0 = anim.Z(self.origin if origin is None else origin) + cam.view * lift
        a1 = (self.e1 if e1 is None else e1) * anim.s
        a2 = (self.e2 if e2 is None else e2) * anim.s
        sa, sb = cam.M @ a1, cam.M @ a2
        if abs(sa[0] * sb[1] - sa[1] * sb[0]) < 0.25:          # edge-on or sub-pixel
            return
        s0 = cam.project(p0)
        pts = np.array([s0, s0 + sa, s0 + sb, s0 + sa + sb])
        if pts[:, 0].max() < -4 or pts[:, 0].min() > cam.width + 4 or \
                pts[:, 1].max() < -4 or pts[:, 1].min() > cam.height + 4:
            return
        n = np.cross(a1, a2)
        d1 = np.cross(a2, n) / np.dot(a1, np.cross(a2, n))
        d2 = np.cross(n, a1) / np.dot(a2, np.cross(n, a1))
        sc.quad(p0, a1, a2, self._shader(p0, d1, d2, opacity, highlight, tex, alpha, glow))

    def _shader(self, p0, d1, d2, opacity, hl, tex, alpha, glow):
        """The quad's shader: world points (h, w, 3) -> rgb, alpha (and glow).

        ``d1`` / ``d2`` are the dual vectors that turn ``P - p0`` into the
        texture coordinates (u along a row, v down a column, both 0..1).
        """
        R, Cn = self.R, self.C

        def shader(P):
            d = P - p0
            u = np.clip(d @ d1, 0.0, 1 - 1e-9)
            v = np.clip(d @ d2, 0.0, 1 - 1e-9)
            # texels per pixel along each axis -> mip level per axis
            if u.shape[0] > 1 and u.shape[1] > 1:
                fu = Cn * math.hypot(u[0, 1] - u[0, 0], u[1, 0] - u[0, 0])
                fv = R * math.hypot(v[0, 1] - v[0, 0], v[1, 0] - v[0, 0])
            else:
                fu = fv = 0.0
            lc = int(math.log2(fu)) if fu > 1.5 else 0
            lr = int(math.log2(fv)) if fv > 1.5 else 0
            t = self.mip(lr, lc, tex)
            tr, tc = t.shape[:2]
            row = np.minimum((v * tr).astype(np.int32), tr - 1)
            col = np.minimum((u * tc).astype(np.int32), tc - 1)
            rgb = t[row, col]
            a = np.full(u.shape, opacity, np.float32)
            if fu > 0 and fu < 1 / 9 and fv < 1 / 9:           # big cells: show the grid
                gu, gv = (u * Cn) % 1.0, (v * R) % 1.0
                g = 0.045
                edge = (gu < g) | (gu > 1 - g) | (gv < g) | (gv > 1 - g)
                rgb = rgb * np.where(edge, 0.86, 1.0)[..., None].astype(np.float32)
            ri = np.minimum((v * R).astype(np.int32), R - 1)
            ci = np.minimum((u * Cn).astype(np.int32), Cn - 1)
            if hl is not None:
                h = hl[ri, ci] if np.ndim(hl) else np.float32(hl)
                rgb = rgb + (1.0 - rgb) * (0.7 * np.asarray(h, np.float32))[..., None]
            if alpha is not None:
                a = a * alpha[ri, ci]
            if glow > 0:
                return rgb.astype(np.float32), a, np.full(u.shape, glow, np.float32)
            return rgb.astype(np.float32), a

        return shader

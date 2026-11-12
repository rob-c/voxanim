"""Flat diagrams in the voxel scene: node-link networks, plots and trees.

Everything lies in one plane facing the camera (through the origin, normal to
the view direction), placed by *layout fractions* of the frame like the rest
of the package; so depth is exactly ``lift`` everywhere and draw order never
depends on where on screen something is.  Shapes are
camera-facing billboards (textured quads) sized in reference pixels, so a
neuron is a round disk at any resolution, with analytic anti-aliasing:

* :meth:`Diagram.disk` -- a shaded ball, optionally ringed,
* :meth:`Diagram.rect` -- a rounded rectangle with a border,
* :meth:`Diagram.dot` -- a small glowing particle (a value in flight).

``lift`` moves a shape towards the viewer: edges (lift 0) are drawn first,
then panels, then nodes, so nodes always sit on top of their edges.
Text goes through :meth:`Diagram.label`, at a world point.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from . import color as C
from . import layout
from .easing import ease_in_out
from .netanim import NetAnimation

DATA = C.rgb("#1f9e94")            # values flowing forward (teal)
DATA_HI = C.rgb("#8ff0e4")
GRAD = C.rgb("#c0399b")            # gradients flowing back (magenta)
GRAD_HI = C.rgb("#ffb3ea")
NODE = C.rgb("#f4f4f1")            # an empty node
EDGE = (0.42, 0.44, 0.46)


def _footprint(q):
    """Reference px per output px, from the shader's pixel grid."""
    if q.ndim == 3 and q.shape[0] > 1 and q.shape[1] > 1:
        return max(math.hypot(*(q[0, 1] - q[0, 0])), 1e-6)
    return 1.0


class Diagram(NetAnimation):
    """A :class:`NetAnimation` whose content is a flat diagram."""

    def __init__(self, title, subtitle, ppu=10.0, azimuth=10.0, elevation=12.0):
        super().__init__(title, subtitle, ppu, azimuth, elevation)
        cam = self.cam
        self.Mr = cam.M                              # world -> reference px
        self.Mi = np.linalg.pinv(self.Mr)            # px -> world, in the view plane
        self.view = cam.view

    def finish(self, t_done):
        """Set the duration; the floor sits far below, so a flat diagram casts no
        shadows or reflections."""
        self.t_done = t_done
        self.duration = t_done + self.hold + self.reset
        # far below: no floor shadows or reflections under a flat diagram
        self.floor_z = -400.0

    # ------------------------------------------------------------------ placement
    def P(self, fx, fy, lift=0.0):
        """World point (in the diagram plane) at layout fraction ``(fx, fy)``."""
        cam = self.cam
        k = cam.zoom
        s = np.array([(0.5 + (fx - 0.5) * k) * cam.width, (0.5 + (fy - 0.5) * k) * cam.height])
        return self.Mi @ (s - cam.offset) + self.view * lift

    def px(self, p):
        """Screen position of world point ``p``, in reference px."""
        return self.cam.project(p)

    def fr(self, p):
        """Screen position of world point ``p``, as fractions of the frame."""
        s = self.cam.project(p)
        return s[0] / self.cam.width, s[1] / self.cam.height

    def ends(self, a, b, ra, rb):
        """The segment from ``a`` to ``b`` shortened by ``ra`` / ``rb`` px at each
        end, so an edge stops at the rim of the nodes it joins."""
        sa, sb = self.px(a), self.px(b)
        u = (b - a) / np.linalg.norm(sb - sa)
        return a + u * ra, b - u * rb

    def shift(self, p, dx, dy):
        """``p`` moved by ``(dx, dy)`` screen px (in the view plane)."""
        return np.asarray(p, np.float64) + self.Mi @ np.array([dx, dy])

    # ------------------------------------------------------------------ shapes
    def _billboard(self, sc, c, hw, hh, shader):
        p0 = c + self.Mi @ np.array([-hw, -hh])
        sc.quad(p0, self.Mi @ np.array([2 * hw, 0.0]), self.Mi @ np.array([0.0, 2 * hh]),
                shader)

    def disk(self, sc, c, r, fill, opacity=1.0, ring=None, ring_w=2.5, ring_op=1.0,
             glow=0.0, lift=1.0, outline=0.22):
        """A shaded ball of radius ``r`` px; ``ring`` adds a coloured band
        just outside it (the 'gradient recorded' badge)."""
        if opacity <= 0.01:
            return
        c = np.asarray(c, np.float64) + self.view * lift
        fill = np.asarray(fill, np.float32)
        rim = fill * (1 - outline)
        ringc = None if ring is None else np.asarray(ring, np.float32)
        R = r + (ring_w + 1.0 if ring is not None else 0.0)
        M = self.Mr

        def shader(P):
            q = (P - c) @ M.T
            pf = _footprint(q)
            d = np.hypot(q[..., 0], q[..., 1])
            a = np.clip((r - d) / pf + 0.5, 0.0, 1.0)
            # soft light from the top left, a darker rim
            lit = np.clip((-q[..., 0] - 1.4 * q[..., 1]) / (2.2 * r), -1, 1)
            col = fill[None, None] * (1.0 + 0.10 * lit)[..., None]
            edge = np.clip((d - (r - 1.3)) / pf + 0.5, 0.0, 1.0)[..., None]
            col = col * (1 - edge) + rim * edge
            alpha = a * opacity
            if ringc is not None and ring_op > 0:
                band = np.clip(np.minimum(d - (r + 0.8), (r + 0.8 + ring_w) - d) / pf + 0.5,
                               0.0, 1.0) * ring_op
                col = col * (1 - band[..., None]) + ringc * band[..., None]
                alpha = np.maximum(alpha, band * opacity)
            return np.clip(col, 0, 1), alpha, glow
        self._billboard(sc, c, R + 1, R + 1, shader)

    def rect(self, sc, c, w, h, fill, opacity=1.0, border=None, border_w=1.5, radius=6.0,
             lift=0.5, glow=0.0, fill_op=1.0):
        """Rounded rectangle ``w x h`` px centred at ``c``."""
        if opacity <= 0.01:
            return
        c = np.asarray(c, np.float64) + self.view * lift
        fill = np.asarray(fill, np.float32)
        bc = None if border is None else np.asarray(border, np.float32)
        b = np.array([w / 2.0, h / 2.0])
        M = self.Mr

        def shader(P):
            q = np.abs((P - c) @ M.T)
            pf = _footprint((P - c) @ M.T)
            dq = q - (b - radius)
            out = np.hypot(np.maximum(dq[..., 0], 0), np.maximum(dq[..., 1], 0))
            d = out + np.minimum(np.maximum(dq[..., 0], dq[..., 1]), 0) - radius
            a = np.clip(0.5 - d / pf, 0.0, 1.0)
            col = np.broadcast_to(fill, q.shape[:-1] + (3,)).copy()
            alpha = a * fill_op
            if bc is not None:
                band = np.clip(0.5 - np.abs(d + border_w / 2) / pf + border_w / 2 / pf,
                               0.0, 1.0) * a
                col = col * (1 - band[..., None]) + bc * band[..., None]
                alpha = np.maximum(alpha, band)
            return col, alpha * opacity, glow
        self._billboard(sc, c, w / 2 + 1, h / 2 + 1, shader)

    def dot(self, sc, a, b, u, color, r=3.2, opacity=1.0, key=None, lift=1.6, bend=0.0):
        """A value flying from ``a`` to ``b`` (``u`` in 0..1)."""
        if not 0.0 < u < 1.0 or opacity <= 0.01:
            return
        e = ease_in_out(u)
        a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
        p = a + (b - a) * e
        if bend:
            p = self.shift(p, 0.0, -bend * 4 * e * (1 - e))
        fade = min(1.0, u / 0.12, (1 - u) / 0.12)
        self.disk(sc, p, r, color, opacity * fade, glow=0.6, lift=lift, outline=0.0)
        if key is not None:
            sc.probe(key, p, tol=6.0)

    def polyline(self, sc, pts, width=1.6, color=EDGE, opacity=1.0, bias=0.2):
        """Lines joining consecutive points (``width`` in px)."""
        for p, q in itertools.pairwise(pts):
            sc.line(p, q, width=width, color=color, opacity=opacity, bias=bias)

    # ------------------------------------------------------------------ text
    def label(self, sc, p, text, size=layout.CAPTION, anchor="mm", dx=0.0, dy=0.0,
              ink=None, opacity=1.0, kind="regular"):
        """Text centred (``anchor``) on world point ``p``, offset ``dx``, ``dy`` px;
        it fades with the caption fade of the loop."""
        opacity *= self._cap
        if opacity <= 0.01 or not text:
            return
        x, y = self.fr(p)
        sc.text(x + dx / self.cam.width, y + dy / self.cam.height, text,
                max(size, layout.MIN_TEXT), kind, layout.CAPTION_INK if ink is None else ink,
                opacity, anchor)


def image_shader(anim, c, tex, w, h, opacity=1.0, grid=0.0):
    """A ``(rows, cols, 3)`` texture as a crisp pixel grid in a ``w x h`` px box."""
    tex = np.asarray(tex, np.float32)
    rows, cols = tex.shape[:2]
    M = anim.Mr

    def shader(P):
        q = (P - c) @ M.T
        u = np.clip(q[..., 0] / w + 0.5, 0, 1 - 1e-9)
        v = np.clip(q[..., 1] / h + 0.5, 0, 1 - 1e-9)
        col = tex[(v * rows).astype(np.int32), (u * cols).astype(np.int32)]
        if grid > 0:
            pf = _footprint(q)
            fu, fv = (u * cols) % 1.0, (v * rows) % 1.0
            cw = w / cols / pf                       # cell size in output px
            if cw > 4:
                line = np.minimum(np.minimum(fu, 1 - fu), np.minimum(fv, 1 - fv)) * cw
                k = np.clip(1.2 - line, 0, 1)[..., None] * grid
                col = col * (1 - k) + 0.5 * col * k
        return col, np.full(q.shape[:-1], opacity, np.float32), 0.0
    return shader


def draw_image(anim, sc, c, tex, w, h, opacity=1.0, grid=0.35, lift=0.3):
    """Draw texture ``tex`` (rows, cols, 3) as a crisp ``w x h`` px image centred at
    ``c``; ``grid`` darkens the cell borders when cells are large enough to show."""
    if opacity <= 0.01:
        return
    c = np.asarray(c, np.float64) + anim.view * lift
    anim._billboard(sc, c, w / 2, h / 2, image_shader(anim, c, tex, w, h, opacity, grid))


def shrink_columns(tex, max_cols):
    """``tex`` (rows, cols, 3) with neighbouring columns averaged in pairs until
    there are at most ``max_cols``: a 768-wide vector drawn 400 px wide keeps
    every value's share of the colour instead of dropping every other one."""
    t = np.asarray(tex, np.float32)
    while t.shape[1] > max_cols and t.shape[1] > 1:
        if t.shape[1] % 2:
            t = np.concatenate([t, t[:, -1:]], 1)
        t = (t[:, 0::2] + t[:, 1::2]) / 2
    return t


def draw_strip(anim, sc, c, colors, w, h, opacity=1.0, lift=0.3):
    """A vector's ``(n, 3)`` colours (or a ``(rows, n, 3)`` stack of vectors) as
    a ``w x h`` px strip centred at ``c``, with the columns averaged down to
    the pixels available (see :func:`shrink_columns`); no cell grid."""
    tex = np.asarray(colors, np.float32)
    if tex.ndim == 2:
        tex = tex[None]
    draw_image(anim, sc, c, shrink_columns(tex, int(w)), w, h, opacity, grid=0.0, lift=lift)

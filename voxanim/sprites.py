"""Pre-rendered voxel sprites.

Under an orthographic camera every voxel of a given size looks identical, so
instead of rasterising cubes per frame we ray-march one bevelled cube (a
rounded-box SDF) at 8x resolution and box-filter it down into a small atlas of
sub-pixel *phases*.  Placing a voxel is then a single blit, and moving one by a
fraction of a pixel picks a different phase instead of snapping -- which is
what keeps slow camera-relative motion free of jitter.

Each phase stores four premultiplied channels:

    A  coverage (anti-aliased alpha)
    D  diffuse factor  -> pixel = D * base_colour + W * white
    W  white / sheen factor
    Z  depth relative to the voxel centre (for the z-buffer)

Because the shading is linear in the base colour, one atlas serves every
colour; tinting happens at blit time.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import os
from dataclasses import astuple, dataclass

import numpy as np

VERSION = 3


@dataclass(frozen=True)
class Look:
    """Shape and lighting rig of a voxel."""

    size: float = 0.92           # cube edge (pitch is 1.0, the rest is seam)
    radius: float = 0.075        # bevel radius
    # (diffuse, white) per face direction: +x, -x, +y, -y, +z, -z.
    # Visible faces are -x (left), +y (right) and +z (top); the -z entry is
    # what floor reflections show.
    faces: tuple = ((0.72, 0.00), (0.86, 0.15), (1.00, 0.00),
                    (0.72, 0.00), (1.06, -0.06), (0.50, 0.00))
    bevel_dark: float = 0.34     # how much the rounded edges darken (seams)
    bevel_sheen: float = 0.30    # light rim on edges that face the key light
    sheen_everywhere: bool = False   # rim light on every edge, not just lit ones


# What the glossy floor shows: a pale, desaturated copy with bright seams.
# Faces come out close to the floor's own tone, so mostly the seams read.
REFLECTION_LOOK = Look(faces=((0.12, 0.82),) * 6, bevel_dark=0.0, bevel_sheen=0.65,
                       sheen_everywhere=True)


def _sdf_round_box(p, b, r):
    q = np.abs(p) - b
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    return outside + inside - r


def _normals(p, b):
    q = np.abs(p) - b
    g = np.maximum(q, 0.0) * np.sign(p)
    n = np.linalg.norm(g, axis=-1, keepdims=True)
    # Points on a flat face (all q <= 0 except the dominant one) fall back to
    # the dominant axis.
    axis = np.argmax(q, axis=-1)
    flat = np.zeros_like(p)
    np.put_along_axis(flat, axis[..., None],
                      np.take_along_axis(np.sign(p), axis[..., None], -1), -1)
    return np.where(n > 1e-9, g / np.maximum(n, 1e-9), flat)


def _shade(n, look):
    f = np.asarray(look.faces, np.float64)
    w = np.stack([np.maximum(n[..., 0], 0) ** 2, np.maximum(-n[..., 0], 0) ** 2,
                  np.maximum(n[..., 1], 0) ** 2, np.maximum(-n[..., 1], 0) ** 2,
                  np.maximum(n[..., 2], 0) ** 2, np.maximum(-n[..., 2], 0) ** 2], -1)
    w = w / np.maximum(w.sum(-1, keepdims=True), 1e-9)
    diffuse = w @ f[:, 0]
    white = w @ f[:, 1]
    # Bevels: how far the normal leans away from its face.  Flat = 0.
    lean = 1.0 - np.abs(n).max(axis=-1)
    edge = np.clip(lean / 0.30, 0.0, 1.0) ** 0.8
    key = 1.0 if look.sheen_everywhere else np.clip(-n[..., 0] * 0.7 + n[..., 2] * 0.7, 0, 1)
    diffuse = diffuse * (1.0 - look.bevel_dark * edge)
    white = white + look.bevel_sheen * edge * key
    return diffuse, white


class SpriteAtlas:
    """All sub-pixel phases of one voxel look, for one camera."""

    def __init__(self, M, view, look=Look(), hires=8, phases=4, cache_dir=None):
        self.look = look
        self.phases = phases
        M = np.asarray(M, np.float64)
        view = np.asarray(view, np.float64)
        h = look.size / 2.0
        corners = np.array(list(itertools.product((-h, h), repeat=3)))
        sc = corners @ M.T
        self.ox = int(math.floor(sc[:, 0].min())) - 1
        self.oy = int(math.floor(sc[:, 1].min())) - 1
        self.w = int(math.ceil(sc[:, 0].max())) - self.ox + 2
        self.h = int(math.ceil(sc[:, 1].max())) - self.oy + 2

        key = hashlib.sha1(repr((VERSION, np.round(M, 6).tolist(),
                                 np.round(view, 6).tolist(), astuple(look),
                                 hires, phases)).encode()).hexdigest()[:16]
        path = os.path.join(cache_dir, "voxel_%s.npy" % key) if cache_dir else None
        if path and os.path.exists(path):
            self.data = np.load(path)
            return
        self.data = self._build(M, view, look, hires, phases)
        if path:
            os.makedirs(cache_dir, exist_ok=True)
            tmp = path + ".%d.tmp.npy" % os.getpid()
            np.save(tmp, self.data)
            os.replace(tmp, path)

    def _build(self, M, view, look, hires, phases):
        h, r = look.size / 2.0, look.radius
        b = h - r
        nx, ny = (self.w + 1) * hires, (self.h + 1) * hires
        X = (self.ox - 1) + (np.arange(nx) + 0.5) / hires
        Y = (self.oy - 1) + (np.arange(ny) + 0.5) / hires
        XX, YY = np.meshgrid(X, Y)
        P = np.linalg.pinv(M)                       # screen offset -> plane point
        p0 = np.stack([XX, YY], -1) @ P.T
        ro = p0 + view * (2.0 * h)
        rd = -view
        t = np.zeros(XX.shape)
        for _ in range(96):
            d = _sdf_round_box(ro + t[..., None] * rd, b, r)
            t = t + d
            if np.abs(d).max() < 1e-5 or (t > 8 * h).all():
                break
        p = ro + t[..., None] * rd
        hit = (np.abs(_sdf_round_box(p, b, r)) < 1e-3) & (t < 4.0 * h)
        n = _normals(p, b)
        diffuse, white = _shade(n, look)
        z = p @ view
        hitf = hit.astype(np.float64)

        out = np.zeros((phases, phases, 4, self.h, self.w), np.float32)
        step = hires // phases
        for ky in range(phases):
            for kx in range(phases):
                sx, sy = hires - kx * step, hires - ky * step
                sl = (slice(sy, sy + self.h * hires), slice(sx, sx + self.w * hires))

                def down(a, sl=sl):
                    return a[sl].reshape(self.h, hires, self.w, hires).mean(axis=(1, 3))

                A = down(hitf)
                Zs = down(hitf * z)
                Z = np.where(A > 0, Zs / np.maximum(A, 1e-9), -np.inf)
                out[ky, kx] = (A, down(hitf * diffuse), down(hitf * white), Z)
        return out

    def locate(self, sx, sy):
        """Placement for a voxel whose centre lands on screen ``(sx, sy)``:
        ``(x0, y0, ky, kx)`` -- top-left pixel and the phase to use
        (``self.data[ky, kx]`` is the ``(4, h, w)`` sprite)."""
        q = self.phases
        ix, iy = math.floor(sx), math.floor(sy)
        kx, ky = int(round((sx - ix) * q)), int(round((sy - iy) * q))
        if kx == q:
            ix, kx = ix + 1, 0
        if ky == q:
            iy, ky = iy + 1, 0
        return ix + self.ox, iy + self.oy, ky, kx

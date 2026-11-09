"""Float framebuffer with a z-buffer.

Opaque voxels are drawn back to front *and* depth-tested/written, so the
painter's order gives clean anti-aliased edges while the z-buffer settles
anything the sort gets wrong (moving kernels passing other objects, lines that
run through a volume, ...).  Translucent primitives (ghost voxels, glass,
lines) are depth-tested against the opaque layer but never write depth.

Colours are premultiplied float32 in ``[0, 1]``.  ``cov`` (accumulated alpha)
is only tracked for layers that are composited over something else afterwards:
the geometry layer over the floor plate, and the floor reflection into it.

``emit`` (optional) is a scalar layer of extra white *light*: glows, glints and
the multiply/accumulate wave write to it.  It is occluded like everything else
but never changes the colour layer, so SDR output ignores it entirely and HDR
output adds it as highlights above graphics white.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


class Framebuffer:
    """A transparent ``width x height`` layer: premultiplied ``rgb``, depth ``z``
    (larger is nearer, ``-inf`` where empty) and optionally ``cov`` and ``emit``."""

    def __init__(self, width, height, track_coverage=False, track_emission=False):
        self.w, self.h = int(width), int(height)
        self.rgb = np.zeros((self.h, self.w, 3), np.float32)
        self.z = np.full((self.h, self.w), -np.inf, np.float32)
        self.cov = np.zeros((self.h, self.w), np.float32) if track_coverage else None
        self.emit = np.zeros((self.h, self.w), np.float32) if track_emission else None

    # ------------------------------------------------------------ helpers
    def _clip(self, x0, y0, w, h):
        """Intersect a ``w x h`` box at ``(x0, y0)`` with the frame."""
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(self.w, x0 + w), min(self.h, y0 + h)
        if fx1 <= fx0 or fy1 <= fy0:
            return None
        return (slice(fy0, fy1), slice(fx0, fx1),
                slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0))

    def _composite(self, sl, premul, alpha, depth=None, zwrite=False, bias=0.0, emit=None):
        """Blend premultiplied ``premul``/``alpha`` into region ``sl``; ``emit``
        is this primitive's own (premultiplied) highlight light, if any."""
        fy, fx = sl
        if depth is not None:
            visible = depth + bias >= self.z[fy, fx]
            alpha = alpha * visible
            premul = premul * visible[..., None]
            if emit is not None:
                emit = emit * visible
        a = alpha[..., None]
        self.rgb[fy, fx] = premul + self.rgb[fy, fx] * (1.0 - a)
        if self.cov is not None:
            self.cov[fy, fx] = alpha + self.cov[fy, fx] * (1.0 - alpha)
        if self.emit is not None:
            e = self.emit[fy, fx] * (1.0 - alpha)
            self.emit[fy, fx] = e if emit is None else e + emit
        if zwrite and depth is not None:
            m = alpha > 0.5
            zr = self.z[fy, fx]
            zr[m] = depth[m]

    # ------------------------------------------------------------ voxels
    def blit(self, x0, y0, premul, A, Z, opacity, zc, zwrite):
        """Blit an already-tinted sprite (premultiplied ``(h, w, 3)``)."""
        clip = self._clip(x0, y0, A.shape[1], A.shape[0])
        if clip is None:
            return
        fy, fx, sy, sx = clip
        A = A[sy, sx]
        prem = premul[sy, sx]
        if opacity != 1.0:
            prem = prem * opacity
            A = A * opacity
        self._composite((fy, fx), prem, A, Z[sy, sx] + zc, zwrite)

    def sprite(self, x0, y0, data, base, white_rgb, opacity, zc, zwrite, white=0.0,
               glow=0.0):
        """Blit one voxel sprite phase (``data`` is ``(4, h, w)``)."""
        A, D, W, Z = data
        clip = self._clip(x0, y0, A.shape[1], A.shape[0])
        if clip is None:
            return
        fy, fx, sy, sx = clip
        A, D, W, Z = A[sy, sx], D[sy, sx], W[sy, sx], Z[sy, sx]
        if white > 0.0:
            # Turn towards a flat-lit white: keeps faint face separation.
            base = base * (1.0 - white) + np.float32(white)
            D = D * (1.0 - 0.55 * white) + A * (0.55 * white)
        premul = D[..., None] * base + W[..., None] * white_rgb
        if glow > 0.0:
            premul = premul + A[..., None] * np.float32(glow)
        premul = np.minimum(premul, A[..., None])
        alpha = A * opacity
        emit = None
        if self.emit is not None and (glow > 0.0 or white > 0.0):
            # HDR: glows shine, white flashes flash (SDR clips both at white).
            emit = A * np.float32(opacity * (glow * 2.5 + white * 0.6))
        self._composite((fy, fx), premul * opacity, alpha, Z + zc, zwrite, emit=emit)

    # ------------------------------------------------------------ quads
    def quad(self, cam, p0, e1, e2, shader, bias=0.0):
        """Shaded parallelogram ``p0 + u e1 + v e2`` (u, v in [0, 1]).

        ``shader(world_points) -> (rgb (..., 3), alpha (...)[, emit (...)])``:
        the optional third output is the quad's highlight strength, written to
        the emission layer (HDR only).  The quad is depth-tested but never
        writes depth.
        """
        p0, e1, e2 = (np.asarray(v, np.float64) for v in (p0, e1, e2))
        s0 = cam.project(p0)
        M = cam.M
        a, b = M @ e1, M @ e2
        corners = np.array([s0, s0 + a, s0 + b, s0 + a + b])
        x0, y0 = np.floor(corners.min(0)).astype(int) - 1
        x1, y1 = np.ceil(corners.max(0)).astype(int) + 1
        clip = self._clip(x0, y0, x1 - x0, y1 - y0)
        if clip is None:
            return
        fy, fx, _, _ = clip
        ys = np.arange(fy.start, fy.stop) + 0.5
        xs = np.arange(fx.start, fx.stop) + 0.5
        XX, YY = np.meshgrid(xs - s0[0], ys - s0[1])
        inv = np.linalg.inv(np.stack([a, b], 1))
        u = inv[0, 0] * XX + inv[0, 1] * YY
        v = inv[1, 0] * XX + inv[1, 1] * YY
        # Anti-aliased coverage from the distance (in pixels) to each edge.
        du = 1.0 / max(np.hypot(*inv[0]), 1e-9)
        dv = 1.0 / max(np.hypot(*inv[1]), 1e-9)
        cov = np.clip(np.minimum(np.minimum(u, 1 - u) * du, np.minimum(v, 1 - v) * dv)
                      + 0.5, 0.0, 1.0)
        if not cov.any():
            return
        pts = p0 + u[..., None] * e1 + v[..., None] * e2
        out = shader(pts)
        rgb, alpha = out[0], out[1]
        alpha = np.asarray(alpha, np.float32) * cov
        depth = (pts @ cam.view).astype(np.float32)
        premul = np.asarray(rgb, np.float32) * alpha[..., None]
        emit = None
        if self.emit is not None and len(out) > 2:
            emit = np.float32(out[2]) * alpha        # shader's highlight strength
        self._composite((fy, fx), premul, alpha, depth, bias=bias, emit=emit)

    # ------------------------------------------------------------ lines
    def line(self, cam, p0, p1, width, color, opacity=1.0, bias=0.0):
        """Anti-aliased, depth-tested segment between two world points.

        ``width`` is in this buffer's pixels; ``bias`` (world units, towards
        the viewer) lets a line lying on a surface win the depth test."""
        a, b = cam.project(p0), cam.project(p1)
        za, zb = cam.depth(p0), cam.depth(p1)
        r = width / 2.0 + 1.0
        x0, y0 = np.floor(np.minimum(a, b) - r).astype(int)
        x1, y1 = np.ceil(np.maximum(a, b) + r).astype(int)
        clip = self._clip(x0, y0, x1 - x0, y1 - y0)
        if clip is None:
            return
        fy, fx, _, _ = clip
        ys = np.arange(fy.start, fy.stop) + 0.5
        xs = np.arange(fx.start, fx.stop) + 0.5
        XX, YY = np.meshgrid(xs, ys)
        d = b - a
        L2 = float(d @ d)
        s = np.clip(((XX - a[0]) * d[0] + (YY - a[1]) * d[1]) / max(L2, 1e-9), 0, 1)
        dist = np.hypot(XX - (a[0] + s * d[0]), YY - (a[1] + s * d[1]))
        alpha = np.clip(width / 2.0 + 0.5 - dist, 0.0, 1.0).astype(np.float32) * opacity
        if not alpha.any():
            return
        depth = (za + s * (zb - za)).astype(np.float32)
        premul = alpha[..., None] * np.asarray(color, np.float32)
        self._composite((fy, fx), premul, alpha, depth, bias=bias)

    # ------------------------------------------------------------ resolve
    def downsample(self, k):
        """Colour layer box-filtered down by ``k`` (the supersampling resolve)."""
        return box_downsample(self.rgb, k)

    def downsample_cov(self, k):
        """Coverage layer box-filtered down by ``k``."""
        return box_downsample(self.cov, k)

    def downsample_emit(self, k):
        """Emission layer box-filtered down by ``k`` (``None`` if not tracked)."""
        return None if self.emit is None else box_downsample(self.emit, k)


# ---------------------------------------------------------------- utilities


def box_downsample(a, k):
    """Mean of each ``k x k`` block of an ``(h, w[, c])`` float32 array (rows
    and columns beyond a multiple of ``k`` are dropped).  ``k == 1`` returns
    ``a`` itself, not a copy."""
    if k == 1:
        return a
    h, w = a.shape[0] // k, a.shape[1] // k
    src = a[:h * k, :w * k]
    out = np.zeros((h, w) + a.shape[2:], np.float32)
    for dy in range(k):
        for dx in range(k):
            out += src[dy::k, dx::k]
    return out * np.float32(1.0 / (k * k))


def _box_pass(a, r, axis):
    """Running mean of width ``2r + 1`` along ``axis`` (edge-clamped)."""
    if r < 1:
        return a
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r + 1, r)
    c = np.cumsum(np.pad(a, pad, mode="edge"), axis=axis, dtype=np.float32)
    n = a.shape[axis]
    hi = [slice(None)] * a.ndim
    lo = [slice(None)] * a.ndim
    hi[axis] = slice(2 * r + 1, 2 * r + 1 + n)
    lo[axis] = slice(0, n)
    return (c[tuple(hi)] - c[tuple(lo)]) * np.float32(1.0 / (2 * r + 1))


def _along(axis, ndim, a, b):
    """Index that slices ``a:b`` along ``axis`` and keeps every other axis."""
    return tuple(slice(a, b) if d == axis else slice(None) for d in range(ndim))


def gaussian_blur(arr, sigma):
    """Gaussian blur of a float HxW or HxWxC array (three box passes)."""
    if sigma <= 0.3:
        return arr
    if sigma < 1.0:
        # A single binomial [1 2 1] tap (sigma ~0.7) is plenty for a soft edge.
        out = np.asarray(arr, np.float32)
        for axis in (0, 1):
            p = [(0, 0)] * out.ndim
            p[axis] = (1, 1)
            q = np.pad(out, p, mode="edge")
            n = out.shape[axis]
            taps = [q[_along(axis, out.ndim, a, a + n)] for a in (0, 1, 2)]
            out = 0.25 * taps[0] + 0.5 * taps[1] + 0.25 * taps[2]
        return out
    # Box widths whose three-fold convolution matches the Gaussian variance.
    r = max(1, int(round((np.sqrt(4.0 * sigma * sigma + 1.0) - 1.0) / 2.0)))
    out = np.asarray(arr, np.float32)
    for _ in range(3):
        out = _box_pass(_box_pass(out, r, 0), r, 1)
    return out


def polygon_mask(w, h, polys):
    """Rasterise ``[(points, value)]`` (points in px) into a ``(h, w)`` float
    mask; where polygons overlap the largest value wins (drawn in ascending
    order)."""
    im = Image.new("F", (w, h), 0.0)
    order = sorted(polys, key=lambda pv: pv[1])
    dr = ImageDraw.Draw(im)
    for pts, val in order:
        dr.polygon([(float(x), float(y)) for x, y in pts], fill=float(val))
    return np.asarray(im, np.float32)


def to_uint8(rgb, dither=True, seed=0):
    """8-bit output.  A tiny triangular dither (+-1 LSB) keeps the soft
    background gradients from banding.  The pattern is the *same* every
    frame (``seed``): per-frame noise would make every pixel change every
    frame, which GIF and H.264 cannot compress (50-200 MB GIFs)."""
    v = np.asarray(rgb, np.float32) * 255.0
    if dither:
        rng = np.random.default_rng(seed)
        n = rng.random(v.shape[:2], np.float32) - rng.random(v.shape[:2], np.float32)
        v = v + n[..., None]
    return np.clip(np.round(v), 0, 255).astype(np.uint8)

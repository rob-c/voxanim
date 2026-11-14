""":class:`voxanim.raster.Framebuffer` on a GPU, for large frames (4K and 8K
masters), through PyTorch (see :mod:`voxanim.accel`).

The same methods with the same arguments, so the :class:`~voxanim.scene.Renderer`
draws into either: premultiplied float32 colour, a z-buffer, coverage and
emission, composited "over" in the order the renderer submits primitives.
Sprite atlases and tinted sprite phases are uploaded once and reused (they
arrive as the same numpy arrays frame after frame), so a voxel costs a few
GPU kernels on data already there; lines are rasterised on the device;
quads run their (numpy) shader on the CPU and composite on the device.  The
resolve (:meth:`downsample` and friends) returns numpy arrays, so everything
after it -- the floor, text, HDR encoding -- is unchanged.

The arithmetic is the numpy code's, operation for operation, in float32;
only the device's rounding (fused multiply-adds, reduction order in the
resolve) differs, so frames agree with the numpy renderer to well under one
8-bit level (``tests/test_accel.py``).
"""

from __future__ import annotations

import contextlib
import weakref

import numpy as np

from . import accel


class _Uploads:
    """numpy arrays already on the device, by identity (kept while the array lives)."""

    def __init__(self, dev, limit=20000):
        self.dev, self.limit, self.d = dev, limit, {}

    def get(self, a):
        key = id(a)
        hit = self.d.get(key)
        if hit is not None and hit[0]() is a:
            return hit[1]
        t = accel.to_device(np.ascontiguousarray(a, np.float32), self.dev, accel.torch().float32)
        if len(self.d) > self.limit:
            self.d.clear()
        with contextlib.suppress(TypeError):     # not weak-referenceable: not cached
            self.d[key] = (weakref.ref(a), t)
        return t


_uploads = {}


def uploads(dev):
    """The upload cache of device ``dev`` (shared by every frame of a process)."""
    if dev.type not in _uploads:
        _uploads[dev.type] = _Uploads(dev)
    return _uploads[dev.type]


class TorchFramebuffer:
    """:class:`voxanim.raster.Framebuffer` with its layers on ``dev``."""

    def __init__(self, width, height, track_coverage=False, track_emission=False, dev=None):
        t = accel.torch()
        self.dev = accel.device() if dev is None else dev
        self.w, self.h = width, height
        f32 = t.float32
        self.rgb = t.zeros((height, width, 3), dtype=f32, device=self.dev)
        self.z = t.full((height, width), -float("inf"), dtype=f32, device=self.dev)
        self.cov = t.zeros((height, width), dtype=f32, device=self.dev) if track_coverage else None
        self.emit = (t.zeros((height, width), dtype=f32, device=self.dev) if track_emission
                     else None)
        self.up = uploads(self.dev)

    # the clip rectangle is the numpy Framebuffer's
    def _clip(self, x0, y0, w, h):
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(self.w, x0 + w), min(self.h, y0 + h)
        if fx1 <= fx0 or fy1 <= fy0:
            return None
        return (slice(fy0, fy1), slice(fx0, fx1),
                slice(fy0 - y0, fy1 - y0), slice(fx0 - x0, fx1 - x0))

    def _composite(self, sl, premul, alpha, depth=None, zwrite=False, bias=0.0, emit=None):
        fy, fx = sl
        if depth is not None:
            visible = (depth + bias >= self.z[fy, fx]).to(alpha.dtype)
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
            zr = self.z[fy, fx]
            self.z[fy, fx] = accel.torch().where(alpha > 0.5, depth, zr)

    # ------------------------------------------------------------ voxels
    def blit(self, x0, y0, premul, A, Z, opacity, zc, zwrite):
        """Blit an already-tinted sprite (numpy arrays, uploaded once)."""
        clip = self._clip(x0, y0, A.shape[1], A.shape[0])
        if clip is None:
            return
        fy, fx, sy, sx = clip
        At, Pt, Zt = self.up.get(A)[sy, sx], self.up.get(premul)[sy, sx], self.up.get(Z)[sy, sx]
        if opacity != 1.0:
            Pt = Pt * opacity
            At = At * opacity
        self._composite((fy, fx), Pt, At, Zt + float(zc), zwrite)

    def sprite(self, x0, y0, data, base, white_rgb, opacity, zc, zwrite, white=0.0, glow=0.0):
        """Blit one voxel sprite phase (``data`` is ``(4, h, w)``)."""
        t = accel.torch()
        A, D, W, Z = data
        clip = self._clip(x0, y0, A.shape[1], A.shape[0])
        if clip is None:
            return
        fy, fx, sy, sx = clip
        A, D, W, Z = (self.up.get(v)[sy, sx] for v in (A, D, W, Z))
        base = accel.to_device(np.asarray(base, np.float32), self.dev, t.float32)
        wr = accel.to_device(np.asarray(white_rgb, np.float32), self.dev, t.float32)
        if white > 0.0:
            base = base * (1.0 - white) + white
            D = D * (1.0 - 0.55 * white) + A * (0.55 * white)
        premul = D[..., None] * base + W[..., None] * wr
        if glow > 0.0:
            premul = premul + A[..., None] * glow
        premul = t.minimum(premul, A[..., None])
        alpha = A * opacity
        emit = None
        if self.emit is not None and (glow > 0.0 or white > 0.0):
            emit = A * (opacity * (glow * 2.5 + white * 0.6))
        self._composite((fy, fx), premul * opacity, alpha, Z + float(zc), zwrite, emit=emit)

    # ------------------------------------------------------------ quads and lines
    def quad(self, cam, p0, e1, e2, shader, bias=0.0):
        """A shaded parallelogram: the numpy Framebuffer's geometry and shader
        (on the CPU), composited on the device."""
        from .raster import Framebuffer
        t = accel.torch()
        cap = _Capture()
        Framebuffer.quad(cap.bind(self), cam, p0, e1, e2, shader, bias)
        if cap.args is None:
            return
        sl, premul, alpha, depth, bias, emit = cap.args
        put = lambda v: None if v is None else accel.to_device(v, self.dev, t.float32)  # noqa: E731
        self._composite(sl, put(premul), put(alpha), put(depth), bias=bias, emit=put(emit))

    def line(self, cam, p0, p1, width, color, opacity=1.0, bias=0.0):
        """An anti-aliased, depth-tested segment, rasterised on the device."""
        t = accel.torch()
        a, b = cam.project(p0), cam.project(p1)
        za, zb = cam.depth(p0), cam.depth(p1)
        r = width / 2.0 + 1.0
        x0, y0 = np.floor(np.minimum(a, b) - r).astype(int)
        x1, y1 = np.ceil(np.maximum(a, b) + r).astype(int)
        clip = self._clip(x0, y0, x1 - x0, y1 - y0)
        if clip is None:
            return
        fy, fx, _, _ = clip
        ys = t.arange(fy.start, fy.stop, device=self.dev, dtype=t.float32) + 0.5
        xs = t.arange(fx.start, fx.stop, device=self.dev, dtype=t.float32) + 0.5
        YY, XX = t.meshgrid(ys, xs, indexing="ij")
        d = b - a
        L2 = float(d @ d)
        s = t.clamp(((XX - float(a[0])) * float(d[0]) + (YY - float(a[1])) * float(d[1]))
                    / max(L2, 1e-9), 0, 1)
        dist = t.hypot(XX - (float(a[0]) + s * float(d[0])), YY - (float(a[1]) + s * float(d[1])))
        alpha = t.clamp(width / 2.0 + 0.5 - dist, 0.0, 1.0) * opacity
        depth = float(za) + s * float(zb - za)
        col = accel.to_device(np.asarray(color, np.float32), self.dev, t.float32)
        self._composite((fy, fx), alpha[..., None] * col, alpha, depth, bias=bias)

    # ------------------------------------------------------------ resolve
    def _down(self, a, k):
        if a is None:
            return None
        if k == 1:
            return a.detach().to("cpu").numpy()
        t = accel.torch()
        h, w = a.shape[0] // k, a.shape[1] // k
        src = a[:h * k, :w * k]
        out = t.zeros((h, w) + tuple(a.shape[2:]), dtype=t.float32, device=self.dev)
        for dy in range(k):
            for dx in range(k):
                out += src[dy::k, dx::k]
        return (out * (1.0 / (k * k))).to("cpu").numpy()

    def downsample(self, k):
        """Colour box-filtered down by ``k``, as a numpy array."""
        return self._down(self.rgb, k)

    def downsample_cov(self, k):
        """Coverage box-filtered down by ``k``, as a numpy array."""
        return self._down(self.cov, k)

    def downsample_emit(self, k):
        """Emission box-filtered down by ``k`` (``None`` if not tracked)."""
        return self._down(self.emit, k)


class _Capture:
    """Stands in for a numpy Framebuffer so its quad code runs unchanged and
    hands over what it would composite."""

    def bind(self, fb):
        self.w, self.h, self.emit, self.args = fb.w, fb.h, fb.emit, None
        self._clip = fb._clip
        return self

    def _composite(self, sl, premul, alpha, depth=None, zwrite=False, bias=0.0, emit=None):
        self.args = (sl, premul, alpha, depth, bias, emit)

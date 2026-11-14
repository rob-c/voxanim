"""The voxel rasteriser as one Metal kernel per frame (Apple GPUs, through
PyTorch's ``torch.mps.compile_shader``; see :mod:`voxanim.accel`).

:class:`MetalFramebuffer` has :class:`voxanim.raster.Framebuffer`'s methods,
so the :class:`~voxanim.scene.Renderer` draws into it unchanged, but drawing
only *records* primitives; the frame is composited when it is resolved
(:meth:`~MetalFramebuffer.downsample`):

1. **Record.**  A voxel blit is a reference to its sprite images (the tinted
   colour, coverage and depth arrays, which are the same numpy arrays frame
   after frame) plus its position, opacity and depth.  Lines, quads and
   lit sprites (white, glow, fade) are rasterised by the numpy code itself
   (:class:`~voxanim.raster.Framebuffer`'s own methods, captured) into small
   per-frame images and recorded the same way.
2. **Upload.**  Sprite images go into a pool on the GPU once and stay
   there; the per-frame images and the primitive table are uploaded each frame.
3. **Bin.**  Each primitive is listed under every 16 x 16 tile it touches,
   in submission order (a stable sort on the CPU).
4. **Composite.**  One thread per pixel walks its tile's list in order and
   does exactly what :meth:`voxanim.raster.Framebuffer._composite` does --
   depth test, "over" on colour, coverage and emission, z-write -- with the
   pixel's state held in registers.  The order of primitives at a pixel is
   the order the CPU would have drawn them in, so the result is the same image.

Exactness: the kernel's arithmetic is numpy's float32 arithmetic, operation
for operation, with fused multiply-adds and fast math switched off, and the
resolve sums the same samples in the same order.  Frames of voxels and
lines come out **bit-identical** to the numpy renderer's.  Quads compute
their coverage in float64 on the CPU, and numpy blends them in float64 too;
the kernel blends in float32, so pixels under a quad can differ in the last
bit of a float32 (far below one 8-bit level).  ``tests/test_accel.py`` and
``python3 -m voxanim gpucheck`` measure both.
"""

from __future__ import annotations

import contextlib
import weakref
from collections import OrderedDict

import numpy as np

from . import accel
from .raster import Framebuffer

TILE = 16                         # px: the binning tile (one threadgroup)
LINE_CACHE = 50000                # rasterised lines kept in the GPU pool for reuse
QUAD_CACHE = 20000                # shaded quads kept in the GPU pool for reuse
ASYNC_UPLOAD = True               # per-frame uploads without waiting for the GPU
GPU_LINES = 2 * 3840              # buffer px wide from which lines are rasterised on the GPU
#                                   (a 4K frame supersampled 2x; 960x540 stays bit-identical)
POOL_MAX = 1 << 28                # floats kept in the sprite pool (1 GB) before it restarts

_KERNEL = r"""
#include <metal_stdlib>
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
using namespace metal;

// one primitive: ints  x0 y0 w h kind offP offA offZ offE zwrite
//   kind 0: images in the pool, 1: images in this frame's scratch buffer,
//        2: an anti-aliased line, computed here (large frames),
//        3: a lit sprite (fade, white flash, glow) from the pool's atlas
//           channels: offP = coverage A, offA = diffuse D, offE = sheen W
//   floats (20 each): op zc_hi zc_lo bias | line: ax ay dx dy L2 za zb half r g b opacity
//                     | lit: fade 1-w w 1-.55w .55w glow emitk opacity base_rgb white_rgb
// Every float32 constant is the one numpy would use (Python scalars are
// converted on the CPU exactly as numpy converts them).
// A voxel's depth is its sprite depth (float32) plus its centre's depth, a
// float64 in numpy; it is carried here as hi + lo and added exactly
// (two-sum), so the depth test and the z-write see numpy's float64 value.
// One upload of each kind per frame: ``ibuf`` is a header (W H TX T hasEmit,
// then the offsets of the primitive table, the tile starts, the tile lists
// and -- in fbuf -- the scratch images) followed by those int arrays; ``fbuf``
// is the primitives' floats followed by the scratch images.
kernel void composite(device float* rgb            [[buffer(0)]],
                      device float* zbuf           [[buffer(1)]],
                      device float* cov            [[buffer(2)]],
                      device float* emit           [[buffer(3)]],
                      device const float* pool     [[buffer(4)]],
                      device const float* fbuf     [[buffer(5)]],
                      device const int* ibuf       [[buffer(6)]],
                      uint tid [[thread_position_in_grid]]) {
    device const int* prm = ibuf;
    device const int* pi = ibuf + prm[5];
    device const int* tstart = ibuf + prm[6];
    device const int* tprims = ibuf + prm[7];
    device const float* pf = fbuf;
    device const float* scratch = fbuf + prm[8];
    const int W = prm[0], H = prm[1], TX = prm[2], T = prm[3], hasEmit = prm[4];
    const int tile = (int)tid / (T * T), p = (int)tid % (T * T);
    const int x = (tile % TX) * T + p % T, y = (tile / TX) * T + p / T;
    if (x >= W || y >= H) return;
    const int i = y * W + x;
    float r = rgb[3 * i], g = rgb[3 * i + 1], b = rgb[3 * i + 2];
    float z = zbuf[i], c = cov[i], e = hasEmit ? emit[i] : 0.0f;
    for (int k = tstart[tile]; k < tstart[tile + 1]; k++) {
        const int n = tprims[k];
        device const int* q = pi + 10 * n;
        const int lx = x - q[0], ly = y - q[1];
        if (lx < 0 || ly < 0 || lx >= q[2] || ly >= q[3]) continue;
        device const float* f = pf + 20 * n;
        const float op = f[0], zh = f[1], zl = f[2], bias = f[3];
        float A, pr, pg, pb, Zs, E = 0.0f;
        bool hasE = q[8] >= 0;
        device const float* src = q[4] == 1 ? scratch : pool;
        const int j = ly * q[2] + lx;
        if (q[4] == 2) {
            // raster.Framebuffer.line's formula (it works in float64; this in float32)
            const float X = (float)x + 0.5f, Y = (float)y + 0.5f;
            const float s = clamp(((X - f[4]) * f[6] + (Y - f[5]) * f[7]) / max(f[8], 1e-9f),
                                  0.0f, 1.0f);
            const float ex = X - (f[4] + s * f[6]), ey = Y - (f[5] + s * f[7]);
            const float dist = precise::sqrt(ex * ex + ey * ey);
            A = clamp(f[11] - dist, 0.0f, 1.0f) * f[15];
            pr = A * f[12]; pg = A * f[13]; pb = A * f[14];
            Zs = f[9] + s * (f[10] - f[9]);
        } else if (q[4] == 3) {
            // raster.Framebuffer.sprite, with Renderer._voxel's fade
            float a = pool[q[5] + j], d = pool[q[6] + j], w = pool[q[8] + j];
            Zs = pool[q[7] + j];
            if (f[4] != 1.0f) { a = a * f[4]; d = d * f[4]; w = w * f[4]; }
            float br = f[12], bg = f[13], bb = f[14];
            if (f[6] > 0.0f) {
                br = br * f[5] + f[6]; bg = bg * f[5] + f[6]; bb = bb * f[5] + f[6];
                d = d * f[7] + a * f[8];
            }
            pr = d * br + w * f[15]; pg = d * bg + w * f[16]; pb = d * bb + w * f[17];
            if (f[9] > 0.0f) {
                const float ag = a * f[9];
                pr = pr + ag; pg = pg + ag; pb = pb + ag;
            }
            pr = min(pr, a); pg = min(pg, a); pb = min(pb, a);
            const float o = f[11];
            pr = pr * o; pg = pg * o; pb = pb * o;
            hasE = f[10] >= 0.0f;
            if (hasE) E = a * f[10];
            A = a * o;
        } else {
            A = src[q[6] + j];
            pr = src[q[5] + 3 * j]; pg = src[q[5] + 3 * j + 1]; pb = src[q[5] + 3 * j + 2];
            Zs = src[q[7] + j];
            if (hasE) E = src[q[8] + j];
        }
        if (op != 1.0f) { pr = pr * op; pg = pg * op; pb = pb * op; A = A * op; }
        // depth = Z + (zh + zl), as a float32 dh plus its rounding error dl
        const float s = Zs + zh, bb = s - Zs;
        const float err = ((Zs - (s - bb)) + (zh - bb)) + zl;
        const float depth = s + err, dl = err - (depth - s);
        bool vis;
        if (bias != 0.0f) vis = depth + bias >= z;    // lines and quads: float32 in numpy too
        else vis = depth > z || (depth == z && dl >= 0.0f);
        if (!vis) continue;                           // hidden: numpy multiplies by 0 (exact)
        const float ia = 1.0f - A;
        r = pr + r * ia;  g = pg + g * ia;  b = pb + b * ia;
        c = A + c * ia;
        if (hasEmit) {
            const float ee = e * ia;
            e = hasE ? ee + E : ee;
        }
        if (q[9] && A > 0.5f) z = depth;
    }
    rgb[3 * i] = r; rgb[3 * i + 1] = g; rgb[3 * i + 2] = b;
    zbuf[i] = z; cov[i] = c;
    if (hasEmit) emit[i] = e;
}
"""

_DOWN = r"""
#include <metal_stdlib>
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
using namespace metal;

// raster.box_downsample: each output value is 0 + the k x k samples, added
// row by row in the order the numpy (and torch) code adds them, times 1/k^2.
kernel void down(device const float* src [[buffer(0)]], device float* out [[buffer(1)]],
                 constant int* prm [[buffer(2)]], constant float& inv [[buffer(3)]],
                 uint tid [[thread_position_in_grid]]) {
    const int W = prm[0], C = prm[1], k = prm[2], w = prm[3], h = prm[4];
    if ((int)tid >= w * h * C) return;
    const int c = (int)tid % C, p = (int)tid / C, x = p % w, y = p / w;
    float acc = 0.0f;
    for (int dy = 0; dy < k; dy++)
        for (int dx = 0; dx < k; dx++)
            acc = acc + src[((y * k + dy) * W + (x * k + dx)) * C + c];
    out[tid] = acc * inv;
}
"""

_lib = None
_down_lib = None
_DOWN_PRM = {}                    # the downsampling kernel's parameters, per frame shape


def down_kernel():
    """The compiled downsampling kernel (compiled once per process)."""
    global _down_lib
    if _down_lib is None:
        _down_lib = accel.torch().mps.compile_shader(_DOWN)
    return _down_lib


def kernel():
    """The compiled Metal library (compiled once per process)."""
    global _lib
    if _lib is None:
        _lib = accel.torch().mps.compile_shader(_KERNEL)
    return _lib


def available():
    """True when the Metal kernel can run (PyTorch with an Apple GPU)."""
    t = accel.torch()
    return t is not None and t.backends.mps.is_available() and hasattr(t.mps, "compile_shader")


def split(v):
    """A float64 as two float32s, ``hi + lo``, with ``hi`` the nearest float32."""
    hi = np.float32(v)
    return float(hi), float(np.float32(float(v) - float(hi)))


# ------------------------------------------------------------------ the sprite pool


def _root(a):
    while isinstance(getattr(a, "base", None), np.ndarray):
        a = a.base
    return a


class SpritePool:
    """Sprite images on the GPU, uploaded once.  An array is known by where its
    data lives (pointer, shape, strides) while the array that owns that memory
    is alive, so the atlas's phases (views) and the renderer's cached tinted
    sprites are found again frame after frame."""

    def __init__(self, dev):
        self.dev = dev
        self.known = {}                    # key -> (weakref to the owner, offset)
        self.size = 0                      # floats used
        self.gpu = accel.torch().zeros(1 << 20, dtype=accel.torch().float32, device=dev)
        self.staged, self.staged_size = [], 0
        self.lines = OrderedDict()         # rasterised lines in this pool (see _raster_line_batch)
        self.quads = OrderedDict()         # shaded quads in this pool (see MetalFramebuffer.quad)

    def offset_of(self, key, owner, a):
        """The pool offset of array ``a``, known by ``key`` while ``owner`` lives
        (a cheap key the caller has: an object's identity, an atlas phase)."""
        hit = self.known.get(key)
        if hit is not None and hit[0]() is owner:
            return hit[1]
        off = self._stage(a)
        self.known[key] = (weakref.ref(owner), off)
        return off

    def _stage(self, a):
        flat = np.ascontiguousarray(a, np.float32).ravel()
        off = self.size + self.staged_size
        self.staged.append(flat)
        self.staged_size += flat.size
        return off

    def atlas(self, atlas, ky, kx, ch):
        """The pool offset of channel ``ch`` of atlas phase ``(ky, kx)``."""
        return self.offset_of(("atlas", id(atlas.data), ky, kx, ch), atlas.data,
                              atlas.data[ky, kx, ch])

    def obj(self, a):
        """The pool offset of an array that is the same object frame after
        frame (a cached tinted sprite)."""
        return self.offset_of(("obj", id(a)), a, a)

    def offset(self, a):
        """The pool offset of array ``a`` (staged for upload if it is new)."""
        info = a.__array_interface__
        key = (info["data"][0], a.shape, a.strides, a.dtype.str)
        hit = self.known.get(key)
        if hit is not None and hit[0]() is not None:
            return hit[1]
        off = self._stage(a)
        with contextlib.suppress(TypeError):
            self.known[key] = (weakref.ref(_root(a)), off)
        return off

    def flush(self):
        """Upload what was staged (growing the GPU pool when needed); returns
        the host array copied from (None if nothing was staged)."""
        t = accel.torch()
        if not self.staged:
            return None
        need = self.size + self.staged_size
        if need > self.gpu.numel():
            cap = self.gpu.numel()
            while cap < need:
                cap *= 2
            grown = t.zeros(cap, dtype=t.float32, device=self.dev)
            grown[:self.size] = self.gpu[:self.size]
            self.gpu = grown
        host = np.concatenate(self.staged)
        data = t.from_numpy(host).to(self.dev, non_blocking=ASYNC_UPLOAD)
        self.gpu[self.size:need] = data
        self.size = need
        self.staged, self.staged_size = [], 0
        return host                       # the caller keeps it until the frame is read back

    def full(self):
        """True when the pool should restart (keeps memory bounded)."""
        return self.size + self.staged_size > POOL_MAX


_pools = {}


def pool(dev):
    """The sprite pool of a device, shared by every frame of the process."""
    p = _pools.get(dev.type)
    if p is None or p.full():
        p = _pools[dev.type] = SpritePool(dev)
    return p


# ------------------------------------------------------------------ the framebuffer


class _Capture:
    """Stands in for a numpy Framebuffer so its methods run unchanged and hand
    over what they would composite."""

    def __init__(self, fb):
        self.w, self.h, self.emit = fb.w, fb.h, (0 if fb.track_emission else None)
        self.args = None

    _clip = Framebuffer._clip

    def _composite(self, sl, premul, alpha, depth=None, zwrite=False, bias=0.0, emit=None):
        self.args = (sl, premul, alpha, depth, zwrite, bias, emit)


class MetalFramebuffer:
    """:class:`voxanim.raster.Framebuffer`, composited by one Metal kernel at
    resolve time (see the module docstring)."""

    def __init__(self, width, height, track_coverage=False, track_emission=False, dev=None):
        self.w, self.h = width, height
        self.dev = accel.device() if dev is None else dev
        self.track_coverage, self.track_emission = track_coverage, track_emission
        self.pool = pool(self.dev)
        self.ints, self.floats = [], []       # the primitive table
        self.lines = []                       # lines awaiting rasterisation (their table rows)
        self.scratch, self.scratch_size = [], 0
        self.done = None                      # (rgb, z, cov, emit) tensors once composited

    _clip = Framebuffer._clip

    # -- recording ------------------------------------------------------------------
    def _scratch(self, a, channels):
        """Stage a per-frame image; returns its offset in the scratch buffer."""
        flat = np.ascontiguousarray(a, np.float32).reshape(-1)
        off = self.scratch_size
        self.scratch.append(flat)
        self.scratch_size += flat.size
        assert flat.size % channels == 0
        return off

    def _record_captured(self, cap, zc=0.0):
        if cap.args is None:
            return
        (fy, fx), premul, alpha, depth, zwrite, bias, emit = cap.args
        h, w = fy.stop - fy.start, fx.stop - fx.start
        if depth is None:
            depth = np.full((h, w), np.inf, np.float32)      # untested: always visible
        offP = self._scratch(np.broadcast_to(premul, (h, w, 3)), 3)
        offA = self._scratch(np.broadcast_to(alpha, (h, w)), 1)
        offZ = self._scratch(np.broadcast_to(depth, (h, w)), 1)
        offE = self._scratch(np.broadcast_to(emit, (h, w)), 1) if emit is not None else -1
        self.ints.append((fx.start, fy.start, w, h, 1, offP, offA, offZ, offE, int(bool(zwrite))))
        self.floats.append((1.0, *split(zc), float(bias)) + (0.0,) * 16)

    def add_rows(self, ints, floats):
        """Append a block of primitives (``(n, 10)`` ints, ``(n, 20)`` floats,
        in drawing order): the renderer's bulk voxels."""
        if len(ints):
            self.ints.append(np.asarray(ints, np.int64))
            self.floats.append(np.asarray(floats, np.float64))

    def blit_atlas(self, x0, y0, premul, atlas, ky, kx, opacity, zc, zwrite):
        """:meth:`blit` of atlas phase ``(ky, kx)`` tinted ``premul`` (the
        renderer's fast path: offsets found by identity, no array views)."""
        h, w = atlas.data.shape[-2:]
        if self._clip(x0, y0, w, h) is None:
            return
        p = self.pool
        self.ints.append((x0, y0, w, h, 0, p.obj(premul), p.atlas(atlas, ky, kx, 0),
                          p.atlas(atlas, ky, kx, 3), -1, int(bool(zwrite))))
        self.floats.append((float(np.float32(opacity)), *split(zc), 0.0) + (0.0,) * 16)

    def lit_atlas(self, x0, y0, atlas, ky, kx, fade, base, white_rgb, opacity, zc, zwrite,
                  white=0.0, glow=0.0):
        """:meth:`lit` of atlas phase ``(ky, kx)`` (offsets found by identity)."""
        h, w = atlas.data.shape[-2:]
        if self._clip(x0, y0, w, h) is None:
            return
        p = self.pool
        offs = [p.atlas(atlas, ky, kx, c) for c in range(4)]
        self._lit_row(x0, y0, w, h, offs, fade, base, white_rgb, opacity, zc, zwrite, white,
                      glow)

    def blit(self, x0, y0, premul, A, Z, opacity, zc, zwrite):
        """Record an already-tinted sprite (its images join the GPU pool once)."""
        if self._clip(x0, y0, A.shape[1], A.shape[0]) is None:
            return
        p = self.pool
        self.ints.append((x0, y0, A.shape[1], A.shape[0], 0, p.offset(premul), p.offset(A),
                          p.offset(Z), -1, int(bool(zwrite))))
        self.floats.append((float(np.float32(opacity)), *split(zc), 0.0) + (0.0,) * 16)

    def sprite(self, x0, y0, data, base, white_rgb, opacity, zc, zwrite, white=0.0, glow=0.0):
        """A lit sprite (see :meth:`lit`; unfaded)."""
        self.lit(x0, y0, data, 1.0, base, white_rgb, opacity, zc, zwrite, white, glow)

    def lit(self, x0, y0, data, fade, base, white_rgb, opacity, zc, zwrite, white=0.0,
            glow=0.0):
        """A lit sprite -- :meth:`voxanim.raster.Framebuffer.sprite` of the atlas
        phase ``data`` scaled by ``fade`` (as :meth:`voxanim.scene.Renderer._voxel`
        fades reflections) -- computed by the kernel from the pooled atlas
        channels, with numpy's float32 constants."""
        A, D, W, Z = data
        if self._clip(x0, y0, A.shape[1], A.shape[0]) is None:
            return
        p = self.pool
        offs = [p.offset(A), p.offset(D), p.offset(W), p.offset(Z)]
        self._lit_row(x0, y0, A.shape[1], A.shape[0], offs, fade, base, white_rgb, opacity, zc,
                      zwrite, white, glow)

    def _lit_row(self, x0, y0, w_, h_, offs, fade, base, white_rgb, opacity, zc, zwrite, white,
                 glow):
        f32 = lambda v: float(np.float32(v))  # noqa: E731  (numpy's cast of a Python scalar)
        emitk = (f32(opacity * (glow * 2.5 + white * 0.6))
                 if self.track_emission and (glow > 0.0 or white > 0.0) else -1.0)
        b = np.asarray(base, np.float32)
        w = np.asarray(white_rgb, np.float32)
        oA, oD, oW, oZ = offs
        self.ints.append((x0, y0, w_, h_, 3, oA, oD, oZ, oW, int(bool(zwrite))))
        self.floats.append((1.0, *split(zc), 0.0, f32(fade), f32(1.0 - white), f32(white),
                            f32(1.0 - 0.55 * white), f32(0.55 * white), f32(glow), emitk,
                            f32(opacity), float(b[0]), float(b[1]), float(b[2]), float(w[0]),
                            float(w[1]), float(w[2]), 0.0, 0.0))

    def quad(self, cam, p0, e1, e2, shader, bias=0.0):
        """A shaded quad: shaded by the numpy code, composited by the kernel.

        A quad whose shader carries a ``key`` (everything its pixels depend on:
        the diagram shaders set one) is cached in the GPU pool with its
        geometry, and reused while it does not change -- most of a diagram is
        the same from frame to frame."""
        skey = getattr(shader, "key", None)
        key = None
        if skey is not None:
            key = (skey, np.asarray(p0, np.float64).tobytes(), np.asarray(e1, np.float64).tobytes(),
                   np.asarray(e2, np.float64).tobytes(), float(bias), cam.M.tobytes(),
                   cam.offset.tobytes(), cam.view.tobytes(), self.w, self.h,
                   bool(self.track_emission))
            cache = self.pool.quads
            hit = cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
                if hit[0] is not None:
                    self.ints.append(hit[0])
                    self.floats.append(hit[1])
                return
        cap = _Capture(self)
        Framebuffer.quad(cap, cam, p0, e1, e2, shader, bias)
        if key is None:
            self._record_captured(cap)
            return
        row = self._pooled(cap)
        self.pool.quads[key] = row
        while len(self.pool.quads) > QUAD_CACHE:
            self.pool.quads.popitem(last=False)
        if row[0] is not None:
            self.ints.append(row[0])
            self.floats.append(row[1])

    def _pooled(self, cap):
        """A captured primitive's images staged into the sprite pool (to be
        reused); returns its ``(ints, floats)`` table row, or ``(None, None)``
        when it covers nothing."""
        if cap.args is None:
            return None, None
        (fy, fx), premul, alpha, depth, zwrite, bias, emit = cap.args
        h, w = fy.stop - fy.start, fx.stop - fx.start
        if depth is None:
            depth = np.full((h, w), np.inf, np.float32)      # untested: always visible
        st = self.pool._stage
        offP = st(np.broadcast_to(premul, (h, w, 3)))
        offA = st(np.broadcast_to(alpha, (h, w)))
        offZ = st(np.broadcast_to(depth, (h, w)))
        offE = st(np.broadcast_to(emit, (h, w))) if emit is not None else -1
        return ((fx.start, fy.start, w, h, 0, offP, offA, offZ, offE, int(bool(zwrite))),
                (1.0, 0.0, 0.0, float(bias)) + (0.0,) * 16)

    def line(self, cam, p0, p1, width, color, opacity=1.0, bias=0.0):
        """A line.  Below :data:`GPU_LINES` px wide it is rasterised by the
        numpy code (in float64, so frames stay bit-identical) and composited by
        the kernel; wider frames rasterise it in the kernel, in float32 (an
        8K frame's lines cost seconds on the CPU)."""
        if self.w < GPU_LINES:
            # rasterised by numpy at resolve time, every line of the frame at once
            self.lines.append((len(self.ints), cam, p0, p1, width, color, opacity, bias))
            self.ints.append(None)
            self.floats.append(None)
            return
        a, b = cam.project(p0), cam.project(p1)
        za, zb = cam.depth(p0), cam.depth(p1)
        r = width / 2.0 + 1.0
        x0, y0 = np.floor(np.minimum(a, b) - r).astype(int)
        x1, y1 = np.ceil(np.maximum(a, b) + r).astype(int)
        clip = self._clip(x0, y0, x1 - x0, y1 - y0)
        if clip is None:
            return
        fy, fx, _, _ = clip
        d = b - a
        col = np.asarray(color, np.float32)
        self.ints.append((fx.start, fy.start, fx.stop - fx.start, fy.stop - fy.start, 2, 0, 0, 0,
                          -1, 0))
        self.floats.append((1.0, 0.0, 0.0, float(bias), float(a[0]), float(a[1]), float(d[0]),
                            float(d[1]), float(d @ d), float(za), float(zb), width / 2.0 + 0.5,
                            float(col[0]), float(col[1]), float(col[2]), float(opacity),
                            0.0, 0.0, 0.0, 0.0))

    def _raster_lines(self):
        """Rasterise every recorded line at once: :meth:`voxanim.raster.Framebuffer.line`'s
        float64 arithmetic, pixel by pixel, on all the lines' pixels in one
        array (the same operations on the same values, so the same bits), into
        the scratch buffer; fills the lines' rows of the primitive table."""
        groups = {}
        for ln in self.lines:                     # one batch per camera (a frame has one)
            groups.setdefault(id(ln[1]), []).append(ln)
        self.lines = []
        for lines in groups.values():
            self._raster_line_batch(lines)

    def _raster_line_batch(self, lines):
        rows, cams, P0, P1, width, color, opacity, bias = zip(*lines)
        cam = cams[0]
        P0 = np.asarray([np.asarray(p, np.float64) for p in P0])
        P1 = np.asarray([np.asarray(p, np.float64) for p in P1])
        a, b = cam.project(P0), cam.project(P1)
        za, zb = cam.depth(P0), cam.depth(P1)
        width = np.asarray(width, np.float64)
        op32 = np.asarray(opacity, np.float32)
        col32 = np.asarray(color, np.float32).reshape(-1, 3)
        bias = np.asarray(bias, np.float64)
        # A line is a pure function of these; most lines are the same frame
        # after frame (cages, axes), so their rasterised pieces stay in the GPU
        # pool and are reused (exactly: the same inputs give the same pixels).
        keys = np.concatenate([a, b, za[:, None], zb[:, None], width[:, None],
                               op32[:, None].astype(np.float64), col32.astype(np.float64),
                               bias[:, None], np.full((len(rows), 2), (self.w, self.h))], 1)
        cache = self.pool.lines
        miss = []
        for i, row in enumerate(rows):
            hit = cache.get(keys[i].tobytes())
            if hit is None:
                miss.append(i)
            else:
                cache.move_to_end(keys[i].tobytes())
                self.ints[row], self.floats[row] = hit
        if not miss:
            return
        mi = np.asarray(miss)
        blocks = self._raster_line_pixels(a[mi], b[mi], za[mi], zb[mi], width[mi], op32[mi],
                                          col32[mi], bias[mi])
        for i, (Ib, Fb) in zip(mi, blocks):
            self.ints[rows[i]], self.floats[rows[i]] = Ib, Fb
            cache[keys[i].tobytes()] = (Ib, Fb)
        while len(cache) > LINE_CACHE:
            cache.popitem(last=False)

    def _raster_line_pixels(self, a, b, za, zb, width, op32, col32, bias):
        """Rasterise lines (projected endpoints ``a``, ``b``, depths ``za``, ``zb``)
        into the sprite pool; returns each line's ``(ints, floats)`` block."""
        r = width / 2.0 + 1.0
        x0, y0 = np.floor(np.minimum(a, b) - r[:, None]).astype(int).T
        x1, y1 = np.ceil(np.maximum(a, b) + r[:, None]).astype(int).T
        fx0, fy0 = np.maximum(x0, 0), np.maximum(y0, 0)
        fx1, fy1 = np.minimum(x1, self.w), np.minimum(y1, self.h)
        d = b - a
        L2 = np.array([float(di @ di) for di in d])            # numpy's own dot, line by line
        hw = width / 2.0 + 0.5                                  # coverage is 0 from here out
        # Each line's box cut along the kernel's 16 px grid; only pieces the
        # line reaches are computed.  A piece whose centre is further from the
        # segment than hw plus half its diagonal (and a pixel of margin) has
        # coverage exactly 0 at every pixel, and a pixel of coverage 0 changes
        # nothing in numpy's composite either -- so skipping it is exact.
        T = TILE
        live = (fx1 > fx0) & (fy1 > fy0)
        tx0, ty0 = fx0 // T, fy0 // T
        ntx = np.where(live, (fx1 - 1) // T - tx0 + 1, 0)
        nty = np.where(live, (fy1 - 1) // T - ty0 + 1, 0)
        n_t = ntx * nty
        Lt = np.repeat(np.arange(len(a)), n_t)
        kt = np.arange(len(Lt)) - np.repeat(np.cumsum(n_t) - n_t, n_t)
        ptx = tx0[Lt] + kt % np.maximum(ntx[Lt], 1)
        pty = ty0[Lt] + kt // np.maximum(ntx[Lt], 1)
        px0 = np.maximum(ptx * T, fx0[Lt])
        py0 = np.maximum(pty * T, fy0[Lt])
        px1 = np.minimum((ptx + 1) * T, fx1[Lt])
        py1 = np.minimum((pty + 1) * T, fy1[Lt])
        cx, cy = (px0 + px1) / 2.0, (py0 + py1) / 2.0
        sc = np.clip(((cx - a[Lt, 0]) * d[Lt, 0] + (cy - a[Lt, 1]) * d[Lt, 1])
                     / np.maximum(L2, 1e-9)[Lt], 0, 1)
        dc = np.hypot(cx - (a[Lt, 0] + sc * d[Lt, 0]), cy - (a[Lt, 1] + sc * d[Lt, 1]))
        keep = dc <= hw[Lt] + 0.5 * np.hypot(px1 - px0, py1 - py0) + 1.0
        Lt, px0, py0, px1, py1 = Lt[keep], px0[keep], py0[keep], px1[keep], py1[keep]
        pw, ph = px1 - px0, py1 - py0
        n_px = pw * ph
        Pc = np.repeat(np.arange(len(Lt)), n_px)                # piece of each pixel
        L = Lt[Pc]                                              # line of each pixel
        k = np.arange(len(Pc)) - np.repeat(np.cumsum(n_px) - n_px, n_px)
        XX = (px0[Pc] + k % pw[Pc]) + 0.5
        YY = (py0[Pc] + k // pw[Pc]) + 0.5
        # raster.Framebuffer.line, pixel by pixel (the same float64 operations)
        s = np.clip(((XX - a[L, 0]) * d[L, 0] + (YY - a[L, 1]) * d[L, 1])
                    / np.maximum(L2, 1e-9)[L], 0, 1)
        dist = np.hypot(XX - (a[L, 0] + s * d[L, 0]), YY - (a[L, 1] + s * d[L, 1]))
        alpha = np.clip(hw[L] - dist, 0.0, 1.0).astype(np.float32) * op32[L]
        depth = (za[L] + s * (zb - za)[L]).astype(np.float32)
        premul = alpha[:, None] * col32[L]
        offP = self.pool._stage(premul)
        offA = self.pool._stage(alpha)
        offZ = self.pool._stage(depth)
        start = np.cumsum(n_px) - n_px
        n = len(Lt)
        I = np.zeros((n, 10), np.int64)
        I[:, 0], I[:, 1], I[:, 2], I[:, 3], I[:, 4] = px0, py0, pw, ph, 0     # kind 0: the pool
        I[:, 5], I[:, 6], I[:, 7], I[:, 8] = offP + 3 * start, offA + start, offZ + start, -1
        F = np.zeros((n, 20), np.float64)
        F[:, 0] = 1.0
        F[:, 3] = bias[Lt]
        # the pieces come grouped by line, in line order
        cut = np.searchsorted(Lt, np.arange(1, len(a)))
        return list(zip(np.split(I, cut), np.split(F, cut)))

    def _flatten(self):
        """The primitive table as two arrays, in drawing order: single rows,
        each line's pieces in its place, and blocks of bulk voxels."""
        ints, floats = [], []
        for i, f in zip(self.ints, self.floats):
            if isinstance(i, np.ndarray):
                ints.append(i)
                floats.append(f)
            elif isinstance(i, list):
                if i:
                    ints.append(np.asarray(i, np.int64).reshape(-1, 10))
                    floats.append(np.asarray(f, np.float64).reshape(-1, 20))
            else:
                ints.append(np.asarray([i], np.int64))
                floats.append(np.asarray([f], np.float64))
        self.ints = np.concatenate(ints) if ints else np.zeros((0, 10), np.int64)
        self.floats = np.concatenate(floats) if floats else np.zeros((0, 20), np.float64)

    # -- compositing ------------------------------------------------------------------
    def _bins(self):
        """``(start (tiles + 1,), prims)``: each tile's primitives in submission order."""
        T = TILE
        tx_n, ty_n = -(-self.w // T), -(-self.h // T)
        I = np.asarray(self.ints, np.int64).reshape(-1, 10)
        x0 = np.maximum(I[:, 0], 0)
        y0 = np.maximum(I[:, 1], 0)
        x1 = np.minimum(I[:, 0] + I[:, 2], self.w)
        y1 = np.minimum(I[:, 1] + I[:, 3], self.h)
        tx0, ty0 = x0 // T, y0 // T
        tw = (x1 - 1) // T - tx0 + 1
        th = (y1 - 1) // T - ty0 + 1
        count = np.where((x1 > x0) & (y1 > y0), tw * th, 0)
        total = int(count.sum())
        idx = np.repeat(np.arange(len(I)), count)
        local = np.arange(total) - np.repeat(np.cumsum(count) - count, count)
        tiles = (ty0[idx] + local // tw[idx]) * tx_n + tx0[idx] + local % tw[idx]
        order = np.argsort(tiles, kind="stable")
        start = np.zeros(tx_n * ty_n + 1, np.int64)
        np.cumsum(np.bincount(tiles, minlength=tx_n * ty_n), out=start[1:])
        return start, idx[order], tx_n, ty_n

    def _composite_all(self):
        if self.done is not None:
            return self.done
        t = accel.torch()
        dev, f32 = self.dev, t.float32
        rgb = t.zeros(self.h * self.w * 3, dtype=f32, device=dev)
        z = t.full((self.h * self.w,), -float("inf"), dtype=f32, device=dev)
        cov = t.zeros(self.h * self.w, dtype=f32, device=dev)
        emit = t.zeros(self.h * self.w if self.track_emission else 1, dtype=f32, device=dev)
        self._raster_lines()
        self._flatten()
        if len(self.ints):
            staged = self.pool.flush()
            start, prims, tx_n, ty_n = self._bins()
            pi = np.asarray(self.ints, np.int32).ravel()
            pf = np.asarray(self.floats, np.float32).ravel()
            HEAD = 9
            o_pi = HEAD
            o_start = o_pi + len(pi)
            o_prims = o_start + len(start)
            head = np.array([self.w, self.h, tx_n, TILE, int(self.track_emission), o_pi, o_start,
                             o_prims, len(pf)], np.int32)
            ib = np.concatenate([head, pi, start.astype(np.int32), prims.astype(np.int32)])
            fb = np.concatenate([pf, *self.scratch]) if self.scratch else pf
            n = tx_n * ty_n * TILE * TILE
            fb = np.ascontiguousarray(fb, np.float32)
            # asynchronous copies (the GPU need not finish earlier work first);
            # the arrays stay referenced until the frame is read back
            self._uploaded = (fb, ib, staged)
            kernel().composite(rgb, z, cov, emit, self.pool.gpu,
                               t.from_numpy(fb).to(dev, non_blocking=ASYNC_UPLOAD),
                               t.from_numpy(ib).to(dev, non_blocking=ASYNC_UPLOAD),
                               threads=n, group_size=TILE * TILE)
        self.done = (rgb.reshape(self.h, self.w, 3), z.reshape(self.h, self.w),
                     cov.reshape(self.h, self.w),
                     emit.reshape(self.h, self.w) if self.track_emission else None)
        return self.done

    # -- resolve --------------------------------------------------------------------
    def _down_t(self, a, k):
        """:func:`voxanim.raster.box_downsample` on the device: the same sums, in
        the same order, so the same bits (a tensor)."""
        if a is None or k == 1:
            return a
        t = accel.torch()
        h, w = a.shape[0] // k, a.shape[1] // k
        C = a.shape[2] if a.dim() == 3 else 1
        out = t.empty((h, w) + tuple(a.shape[2:]), dtype=t.float32, device=self.dev)
        key = (self.dev.type, a.shape[1], C, k, w, h)
        prm = _DOWN_PRM.get(key)
        if prm is None:
            prm = _DOWN_PRM[key] = t.tensor([a.shape[1], C, k, w, h], dtype=t.int32,
                                            device=self.dev)
        down_kernel().down(a.contiguous(), out, prm, np.float32(1.0 / (k * k)).item(),
                           threads=w * h * C, group_size=256)
        return out

    def _down(self, a, k):
        out = self._down_t(a, k)
        return None if out is None else out.to("cpu").numpy()

    def resolve_t(self, k):
        """``(colour, coverage, emission or None)`` box-filtered down by ``k``,
        left on the device (the renderer composites and averages them there)."""
        rgb, _z, cov, emit = self._composite_all()
        return self._down_t(rgb, k), self._down_t(cov, k), self._down_t(emit, k)

    def downsample(self, k):
        """Colour box-filtered down by ``k``, as numpy."""
        return self._down(self._composite_all()[0], k)

    def downsample_cov(self, k):
        """Coverage box-filtered down by ``k``, as numpy."""
        return self._down(self._composite_all()[2], k)

    def downsample_emit(self, k):
        """Emission box-filtered down by ``k`` (``None`` if not tracked)."""
        return self._down(self._composite_all()[3], k)

    @property
    def rgb(self):
        """The full-resolution colour layer, as numpy (tests)."""
        return self._composite_all()[0].to("cpu").numpy()

    @property
    def cov(self):
        """The full-resolution coverage layer, as numpy (the reflection pass blurs it)."""
        return self._composite_all()[2].to("cpu").numpy()

    @property
    def z(self):
        """The full-resolution depth layer, as numpy (tests)."""
        return self._composite_all()[1].to("cpu").numpy()

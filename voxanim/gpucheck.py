"""Check the accelerated renderer against the numpy one, frame by frame.

    python3 -m voxanim gpucheck                      # every animation, 3 frames each
    python3 -m voxanim gpucheck conv2d --frames 8 --scale 4
    python3 -m voxanim gpucheck --device cpu         # the torch framebuffer instead

For each animation, frames at evenly spread instants are rendered twice in
one process -- by the numpy :class:`~voxanim.raster.Framebuffer`
(``VOXANIM_DEVICE=numpy``) and by whatever ``--device`` selects (``auto``: the
Metal kernel on an Apple GPU) -- and compared: identical float32 frames,
identical 8-bit frames (what is encoded), the largest differences, and the
time per frame of each.  Both renderers are warmed on the frames they are
timed on, so caches favour neither.  Exits 1 if any 8-bit pixel differs by
more than ``--tolerance`` levels (default 1).
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace

import numpy as np


@dataclass
class Result:
    """One animation's comparison."""

    name: str
    framebuffer: str            # what rendered the accelerated frames
    frames: int
    same_float: int             # frames whose float32 pixels are identical
    same_8bit: int              # frames whose 8-bit pixels are identical
    max_float: float            # largest |difference| of a float32 channel
    max_8bit: int               # largest difference in 8-bit levels
    pixels_8bit: int            # pixels (all frames) differing in 8 bits
    t_numpy: float              # s per frame
    t_accel: float


def check_one(job):
    """Compare one animation (``(name, frames, scale, device)``)."""
    from . import accel, presets
    from .raster import to_uint8
    from .render import framebuffer_kind
    from .scene import QUALITY, Renderer, Style
    name, frames, scale, device = job
    anim = presets.build(name)
    st = Style(**QUALITY["final"])
    st = replace(st, width=int(round(st.width * scale)), height=int(round(st.height * scale)))
    times = [float(u * anim.t_done) for u in np.linspace(0.1, 0.9, frames)]
    out, secs, kind = {}, {}, "numpy"
    for mode in ("numpy", device):
        with accel.forced(mode):
            r = Renderer(anim.cam, st)
            for t in times:                                   # warm: caches favour neither
                r.render_at(anim.scene, t, 1 / 30)
            t0 = time.perf_counter()
            out[mode] = [r.render_at(anim.scene, t, 1 / 30) for t in times]
            secs[mode] = (time.perf_counter() - t0) / frames
            if mode == device:
                kind = framebuffer_kind(st.width)
    same_f = same_8 = px = 0
    max_f, max_8 = 0.0, 0
    for a, b in zip(out["numpy"], out[device]):
        same_f += bool(np.array_equal(a, b))
        max_f = max(max_f, float(np.abs(a - b).max()))
        d8 = np.abs(to_uint8(a).astype(np.int16) - to_uint8(b).astype(np.int16))
        same_8 += not d8.any()
        max_8 = max(max_8, int(d8.max()))
        px += int(d8.any(-1).sum())
    return Result(name, kind, frames, same_f, same_8, max_f, max_8, px, secs["numpy"],
                  secs[device])


def check(names, frames=3, scale=1.0, device="auto", jobs=None, verbose=True):
    """Compare every animation of ``names``; returns the :class:`Result` s."""
    jobs = jobs or min(6, len(names))
    todo = [(n, frames, scale, device) for n in names]
    if jobs <= 1:
        results = list(map(check_one, todo))
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(jobs, mp_context=mp.get_context("spawn")) as ex:
            results = list(ex.map(check_one, todo))
    if verbose:
        report(results)
    return results


def report(results):
    """Print the comparison table and the totals."""
    print("%-34s %-26s %9s %9s %9s %6s %10s" % ("animation", "framebuffer", "float =",
                                              "8-bit =", "max diff", "levels", "speed-up"))
    for r in results:
        print("%-34s %-26s %5d/%-3d %5d/%-3d %9.1e %6d %9.2fx" % (
            r.name, r.framebuffer[:26], r.same_float, r.frames, r.same_8bit, r.frames,
            r.max_float, r.max_8bit, r.t_numpy / max(r.t_accel, 1e-9)))
    n = sum(r.frames for r in results)
    print("\n%d frames: %d identical in float32, %d identical in 8 bits; worst %d level(s) on "
          "%d pixel(s); %.2f s/frame numpy, %.2f s/frame accelerated" % (
              n, sum(r.same_float for r in results), sum(r.same_8bit for r in results),
              max((r.max_8bit for r in results), default=0),
              sum(r.pixels_8bit for r in results),
              np.mean([r.t_numpy for r in results]), np.mean([r.t_accel for r in results])))


def main(names, frames, scale, device, jobs, tolerance):
    """The ``gpucheck`` command: exits 1 if any frame is off by more than ``tolerance``."""
    from . import presets
    names = names or list(presets.REGISTRY)
    results = check(names, frames, scale, device, jobs)
    raise SystemExit(1 if any(r.max_8bit > tolerance for r in results) else 0)


if __name__ == "__main__":                                   # pragma: no cover
    main(None, 3, 1.0, os.environ.get("VOXANIM_DEVICE", "auto"), None, 1)

"""Frame rendering (multi-process) and encoding (ffmpeg)."""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from PIL import Image

from . import presets
from .raster import to_uint8
from .scene import Renderer

ANIM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD = os.path.join(ANIM, "build")
MEDIA = os.path.abspath(os.path.join(ANIM, "..", "media"))


# ------------------------------------------------------------ frame workers

_W = {}


def _init_worker(name, style, opts=None):
    anim = presets.build(name, **(opts or {}))
    _W["anim"] = anim
    _W["renderer"] = Renderer(anim.cam, style)


def _render_frame(job):
    idx, t, dt, path = job
    img = _W["renderer"].render_at(_W["anim"].scene, t, dt)
    Image.fromarray(to_uint8(img)).save(path, compress_level=1)
    return idx


def frame_times(anim, fps):
    n = int(math.ceil(anim.duration * fps - 1e-9))
    return [k / fps for k in range(n)]


def render_frames(name, style, fps, jobs, out_dir, only=None, opts=None):
    opts = opts or {}
    os.makedirs(out_dir, exist_ok=True)
    anim = presets.build(name, **opts)
    times = frame_times(anim, fps)
    for f in os.listdir(out_dir):
        if f.startswith("frame_") and f.endswith(".png"):
            os.remove(os.path.join(out_dir, f))
    todo = [(k, t, 1.0 / fps, os.path.join(out_dir, "frame_%05d.png" % k))
            for k, t in enumerate(times) if only is None or k in only]
    # Build sprite atlases once so the workers find them in the cache.
    Renderer(anim.cam, style)
    t0 = time.time()
    print("== %s: %d frames (%.1fs at %d fps) with %d workers"
          % (name, len(todo), anim.duration, fps, jobs), flush=True)
    if jobs <= 1:
        _init_worker(name, style, opts)
        for n, job in enumerate(todo):
            _render_frame(job)
            if n % 30 == 0:
                print("   %d/%d" % (n, len(todo)), flush=True)
    else:
        # Workers live in this importable module (not __main__) so "spawn" can
        # find them; ProcessPoolExecutor fails loudly if one dies, where
        # multiprocessing.Pool would silently respawn it forever.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(jobs, mp_context=ctx, initializer=_init_worker,
                                 initargs=(name, style, opts)) as pool:
            for n, _ in enumerate(pool.map(_render_frame, todo, chunksize=4)):
                if n % 60 == 0:
                    el = time.time() - t0
                    eta = el / max(n, 1) * (len(todo) - n)
                    print("   %d/%d  (%.0fs, eta %.0fs)" % (n, len(todo), el, eta), flush=True)
    print("   done in %.0fs -> %s" % (time.time() - t0, out_dir), flush=True)
    return len(todo)


# ------------------------------------------------------------ encoding


def encode(frames_dir, fps, out_base, formats, gif_fps=15, gif_width=960):
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    src = ["-framerate", str(fps), "-i", os.path.join(frames_dir, "frame_%05d.png")]
    done = []
    if "mp4" in formats:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, "-c:v", "libx264",
                        "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p",
                        "-movflags", "+faststart", out_base + ".mp4"], check=True)
        done.append(out_base + ".mp4")
    if "webm" in formats:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, "-c:v", "libvpx-vp9",
                        "-crf", "30", "-b:v", "0", "-row-mt", "1", out_base + ".webm"],
                       check=True)
        done.append(out_base + ".webm")
    if "gif" in formats:
        # One palette weighted towards what moves (stats_mode=diff), and error
        # diffusion only inside the rectangle that changed each frame, so the
        # static scene never shimmers.  Cleaner than Bayer at a smaller size;
        # per-frame palettes look best but are ~10x larger.
        vf = ("fps=%d,scale=%d:-1:flags=lanczos,split[a][b];"
              "[a]palettegen=max_colors=256:stats_mode=diff[p];"
              "[b][p]paletteuse=dither=floyd_steinberg:diff_mode=rectangle"
              % (gif_fps, gif_width))
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *src, "-vf", vf,
                        "-loop", "0", out_base + ".gif"], check=True)
        done.append(out_base + ".gif")
    for p in done:
        print("   %7.2f MB  %s" % (os.path.getsize(p) / 1e6, os.path.relpath(p)))



# ------------------------------------------------------------ broadcast masters
#
# Frames stream straight from the workers into ffmpeg as 16-bit RGB: nothing
# touches the disk (one 8K 16-bit frame is ~200 MB).

RESOLUTIONS = {"1080": (1920, 1080), "4k": (3840, 2160), "8k": (7680, 4320)}
CODECS = {
    # 10-bit HEVC Main10 4:2:0 -- the HDR10 / HLG delivery format
    "hevc": dict(ext=".mp4", args=["-c:v", "libx265", "-preset", "medium", "-crf", "12",
                                   "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1"]),
    # ProRes 4444 XQ, 10-bit 4:4:4 -- a mastering / editing intermediate
    "prores": dict(ext=".mov", args=["-c:v", "prores_ks", "-profile:v", "5",
                                     "-pix_fmt", "yuv444p10le", "-vendor", "apl0"]),
    # 10-bit AV1
    "av1": dict(ext=".mp4", args=["-c:v", "libsvtav1", "-preset", "6", "-crf", "16",
                                  "-pix_fmt", "yuv420p10le"]),
}
TRC = {"pq": "smpte2084", "hlg": "arib-std-b67", "sdr": "bt709"}


def _master_frame(job):
    """Render one frame into a shared-memory slot as 16-bit RGB (no pickling
    of 200 MB frames); returns its light levels."""
    from multiprocessing import shared_memory
    from . import hdr
    idx, t, dt, transfer, slot, shape = job
    r = _W["renderer"]
    shm = shared_memory.SharedMemory(name=slot)
    try:
        out = np.ndarray(shape, dtype="<u2", buffer=shm.buf)
        if transfer == "sdr":
            rgb = r.render_at(_W["anim"].scene, t, dt)
            levels = (0.0, 0.0)
            for y in range(0, shape[0], 512):
                out[y:y + 512] = hdr.to_uint16(np.clip(rgb[y:y + 512], 0, 1))
        else:
            rgb, emit = r.render_at(_W["anim"].scene, t, dt, hdr=True)
            cll, fsum = 0.0, 0.0
            for y in range(0, shape[0], 512):           # strips: bounded temporaries
                code = hdr.encode(rgb[y:y + 512], emit[y:y + 512], transfer)
                out[y:y + 512] = hdr.to_uint16(code)
                c, f = hdr.light_levels(rgb[y:y + 512], emit[y:y + 512])
                cll, fsum = max(cll, c), fsum + f * len(code)
            levels = (cll, fsum / shape[0])
        del out
    finally:
        shm.close()
    return idx, levels


def worker_budget(width, height, ss, jobs=None):
    """Workers that fit in RAM.  Measured peaks: 4K at 2x supersampling
    ~2.5 GB, 8K at 1x ~5.3 GB (framebuffer, depth, coverage, emission,
    floor plate, motion-blur accumulators, the 16-bit output frame)."""
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        total = 16 << 30
    per = 0.6e9 + width * height * (ss * ss * 60 + 80)
    encoder = width * height * 110          # x265 with the capped lookahead below
    fit = max(1, int((total * 0.75 - encoder) // per))
    cpu = max(1, (os.cpu_count() or 2) - 1)
    return min(fit, cpu) if jobs is None else max(1, min(jobs, fit))


def light_metadata(name, fps, transfer, every=0.5, opts=None):
    """HDR10 content light levels.  MaxFALL (a frame average) does not depend
    on resolution, so it is measured densely on reference renders, plus 10%.
    MaxCLL is a per-pixel peak, which *does* (small glints concentrate at 8K),
    so it is declared as the grade's ceiling, ``hdr.PEAK``: over-declaring is
    harmless, under-declaring makes TVs clip highlights."""
    from . import hdr
    from .scene import Style
    if transfer != "pq":
        return None
    anim = presets.build(name, **(opts or {}))
    r = Renderer(anim.cam, Style(ss=1, motion_blur=False))
    fall = 0.0
    for t in np.arange(0.0, anim.duration, every):
        rgb, emit = r.render(anim.scene(float(t)), hdr=True)
        fall = max(fall, hdr.light_levels(rgb, emit)[1])
    return int(hdr.PEAK), int(np.ceil(fall * 1.1))


def ffmpeg_master_cmd(path, width, height, fps, transfer, codec, levels=None):
    space = "bt709" if transfer == "sdr" else "bt2020"
    vf = "scale=in_range=pc:out_range=tv:out_color_matrix=%s" % space
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le",
           "-s", "%dx%d" % (width, height), "-r", str(fps), "-i", "-", "-vf", vf,
           *CODECS[codec]["args"],
           "-color_primaries", space, "-color_trc", TRC[transfer],
           "-colorspace", "bt709" if transfer == "sdr" else "bt2020nc", "-color_range", "tv"]
    if codec == "hevc":
        x = ["colorprim=%s" % space, "transfer=%s" % TRC[transfer],
             "colormatrix=%s" % ("bt709" if transfer == "sdr" else "bt2020nc"),
             "range=limited", "repeat-headers=1",
             # bounded memory at 8K: a short lookahead, two frame threads
             "rc-lookahead=12", "frame-threads=2", "aq-mode=3"]
        if transfer == "pq":
            # BT.2020 primaries, D65, a 1000 cd/m2 mastering display
            x += ["hdr10=1", "master-display=G(8500,39850)B(6550,2300)R(35400,14600)"
                  "WP(15635,16450)L(10000000,50)"]
            if levels:
                x.append("max-cll=%d,%d" % levels)
        cmd += ["-x265-params", ":".join(x)]
    elif codec == "av1" and transfer == "pq":
        md = "mastering-display=G(0.17,0.797)B(0.131,0.046)R(0.708,0.292)WP(0.3127,0.329)" \
             "L(1000,0.005)"
        cl = ":content-light=%d,%d" % levels if levels else ""
        cmd += ["-svtav1-params", md + cl]
    cmd += ["-movflags", "+faststart", path] if CODECS[codec]["ext"] == ".mp4" else [path]
    return cmd


def master(name, res="8k", fps=60, transfer="pq", codec="hevc", jobs=None, quality="final",
           out_dir=None, seconds=None, opts=None):
    opts = opts or {}
    """Render ``name`` straight into a broadcast master."""
    from .scene import QUALITY, Style
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")
    W, H = RESOLUTIONS[res]
    style = Style(**QUALITY[quality]).at_scale(W / 960.0)
    p = presets.get(name)
    anim = presets.build(name, **opts)
    times = frame_times(anim, fps)
    if seconds is not None:
        times = times[:int(round(seconds * fps))]
    out_dir = out_dir or os.path.join(BUILD, "masters")
    os.makedirs(out_dir, exist_ok=True)
    label = res + "p" if res.isdigit() else res
    path = os.path.join(out_dir, "%s%s_%s%d_%s%s" % (p.media, presets.options_suffix(name, **opts),
                                                     label, fps,
                                                   "sdr" if transfer == "sdr" else
                                                   "hdr-" + transfer, CODECS[codec]["ext"]))
    levels = light_metadata(name, fps, transfer, opts=opts)
    jobs = worker_budget(W, H, style.ss, jobs)
    print("== %s -> %s\n   %dx%d %d fps %s %s, %d frames, %d workers%s"
          % (name, os.path.relpath(path), W, H, fps, transfer.upper(), codec, len(times), jobs,
             "" if not levels else ", MaxCLL %d / MaxFALL %d" % levels), flush=True)
    Renderer(anim.cam, style)                         # warm the sprite cache once
    ff = subprocess.Popen(ffmpeg_master_cmd(path, W, H, fps, transfer, codec, levels),
                          stdin=subprocess.PIPE)
    from multiprocessing import shared_memory
    shape = (H, W, 3)
    nbytes = H * W * 3 * 2
    window = jobs + 1
    slots = [shared_memory.SharedMemory(create=True, size=nbytes) for _ in range(window)]
    t0 = time.time()
    cll = fall = 0.0
    ctx = mp.get_context("spawn")
    try:
        with ProcessPoolExecutor(jobs, mp_context=ctx, initializer=_init_worker,
                                 initargs=(name, style, opts)) as pool:
            pending = {}

            def submit(k):
                if k < len(times):
                    slot = slots[k % window].name
                    pending[k] = pool.submit(_master_frame, (k, times[k], 1.0 / fps,
                                                             transfer, slot, shape))

            for k in range(window):
                submit(k)
            for k in range(len(times)):
                _idx, (c, f) = pending.pop(k).result()
                ff.stdin.write(slots[k % window].buf[:nbytes])
                cll, fall = max(cll, c), max(fall, f)
                submit(k + window)                     # reuse the slot just written
                if (k + 1) % 30 == 0 or k + 1 == len(times):
                    el = time.time() - t0
                    print("   %d/%d  (%.0fs, eta %.0fs)" % (k + 1, len(times), el,
                                                           el / (k + 1) * (len(times) - k - 1)),
                          flush=True)
    finally:
        for sm in slots:
            sm.close()
            sm.unlink()
    ff.stdin.close()
    if ff.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print("   %.1f MB  %s  (%.0fs)" % (os.path.getsize(path) / 1e6, os.path.relpath(path),
                                      time.time() - t0))
    if levels:
        print("   measured at full res: MaxCLL %.0f, MaxFALL %.0f (declared %d / %d)"
              % (cll, fall, *levels))
    return path

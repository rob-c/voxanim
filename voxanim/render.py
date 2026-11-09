"""Frame rendering (multi-process), encoding (ffmpeg) and render manifests.

Encoder settings are pinned for reproducibility: one encoder thread, and
bit-exact muxing with no copied or build-specific metadata, so the encoded
bytes depend on the frames and the encoder version, not on the machine's CPU
count.  Each web render also records a JSON manifest (:func:`write_manifest`)
of everything that determined its output.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import math
import multiprocessing as mp
import os
import platform
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import PIL
from PIL import Image

from . import presets
from .camera import REF_W
from .raster import to_uint8
from .scene import Renderer

ANIM = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD = os.path.join(ANIM, "build")
MEDIA = os.path.abspath(os.path.join(ANIM, "..", "media"))

PROGRESS_EVERY = 30            # frames between progress lines


def progress(done, total, t0):
    """Print ``done/total`` with the elapsed time and a linear ETA (seconds)."""
    el = time.time() - t0
    eta = el / max(done, 1) * (total - done)
    print("   %d/%d  (%.0fs, eta %.0fs)" % (done, total, el, eta), flush=True)


# ------------------------------------------------------------ ffmpeg

FFMPEG = ["ffmpeg", "-y", "-loglevel", "error"]
# Output options: no encoder/version tags and no metadata copied from the input.
BITEXACT = ["-fflags", "+bitexact", "-flags:v", "+bitexact", "-map_metadata", "-1"]
ONE_THREAD = ["-threads", "1"]          # multi-threaded encoders vary with CPU count


class FFmpegError(RuntimeError):
    """ffmpeg is missing or failed; the message carries ffmpeg's own error."""


def require_ffmpeg():
    """Raise :class:`FFmpegError` unless ``ffmpeg`` is on the ``PATH``."""
    if shutil.which("ffmpeg") is None:
        raise FFmpegError("ffmpeg not found on PATH")


def _failure(code, log):
    """The :class:`FFmpegError` for exit status ``code`` and stderr bytes ``log``."""
    text = log.decode(errors="replace").strip()
    return FFmpegError("ffmpeg exited with status %d%s" % (code, (": " + text) if text else ""))


def run_ffmpeg(args, stdin=None):
    """Run ``ffmpeg -y -loglevel error *args`` to completion, feeding it the
    bytes ``stdin``; raise :class:`FFmpegError` with its stderr on failure."""
    require_ffmpeg()
    res = subprocess.run([*FFMPEG, *args], input=stdin, capture_output=True)
    if res.returncode != 0:
        raise _failure(res.returncode, res.stderr)


@contextlib.contextmanager
def ffmpeg_pipe(cmd):
    """Run ``cmd`` and yield its stdin to stream raw frames into.

    ffmpeg is killed if the body raises, and is never left running; if it
    exits non-zero (or quits early, breaking the pipe) the result is an
    :class:`FFmpegError` with ffmpeg's own error output."""
    require_ffmpeg()
    with tempfile.TemporaryFile() as log:
        ff = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
        broken = False
        try:
            try:
                yield ff.stdin
                ff.stdin.close()
            except BrokenPipeError:
                broken = True                 # ffmpeg quit early: its log says why
            code = ff.wait()
        finally:
            if ff.poll() is None:             # the body raised: stop the encoder
                ff.kill()
                ff.wait()
            with contextlib.suppress(OSError):
                ff.stdin.close()
        if code != 0 or broken:
            log.seek(0)
            raise _failure(code, log.read())


# ------------------------------------------------------------ frame workers

_W = {}


def _init_worker(name, style, opts=None):
    """Pool initializer: build the animation and its renderer once per process."""
    anim = presets.build(name, **(opts or {}))
    _W["anim"] = anim
    _W["renderer"] = Renderer(anim.cam, style)


def _render_frame(job):
    """Render frame ``(idx, t, dt, path)`` to an 8-bit PNG; returns ``idx``."""
    idx, t, dt, path = job
    img = _W["renderer"].render_at(_W["anim"].scene, t, dt)
    Image.fromarray(to_uint8(img)).save(path, compress_level=1)
    return idx


def frame_times(anim, fps):
    """Start time (s) of every frame of ``anim`` at ``fps``: ``ceil(duration * fps)``
    frames, the last one before the loop point."""
    n = int(math.ceil(anim.duration * fps - 1e-9))
    return [k / fps for k in range(n)]


def render_frames(name, style, fps, jobs, out_dir, opts=None):
    """Render every frame of preset ``name`` to ``out_dir/frame_%05d.png``
    (old frames are removed first) with ``jobs`` processes; returns the count."""
    opts = opts or {}
    os.makedirs(out_dir, exist_ok=True)
    anim = presets.build(name, **opts)
    times = frame_times(anim, fps)
    for f in os.listdir(out_dir):
        if f.startswith("frame_") and f.endswith(".png"):
            os.remove(os.path.join(out_dir, f))
    todo = [(k, t, 1.0 / fps, os.path.join(out_dir, "frame_%05d.png" % k))
            for k, t in enumerate(times)]
    # Build sprite atlases once so the workers find them in the cache.
    Renderer(anim.cam, style)
    t0 = time.time()
    print("== %s: %d frames (%.1fs at %d fps) with %d workers"
          % (name, len(todo), anim.duration, fps, jobs), flush=True)
    if jobs <= 1:
        _init_worker(name, style, opts)
        done = map(_render_frame, todo)
        for n, _ in enumerate(done, 1):
            if n % PROGRESS_EVERY == 0:
                progress(n, len(todo), t0)
    else:
        # Workers live in this importable module (not __main__) so "spawn" can
        # find them; ProcessPoolExecutor fails loudly if one dies, where
        # multiprocessing.Pool would silently respawn it forever.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(jobs, mp_context=ctx, initializer=_init_worker,
                                 initargs=(name, style, opts)) as pool:
            for n, _ in enumerate(pool.map(_render_frame, todo, chunksize=4), 1):
                if n % (2 * PROGRESS_EVERY) == 0:
                    progress(n, len(todo), t0)
    print("   done in %.0fs -> %s" % (time.time() - t0, out_dir), flush=True)
    return len(todo)


# ------------------------------------------------------------ web encoding

X264 = ["-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p"]
VP9 = ["-c:v", "libvpx-vp9", "-crf", "30", "-b:v", "0", "-pix_fmt", "yuv420p"]


def encode(frames_dir, fps, out_base, formats, gif_fps=15, gif_width=REF_W):
    """Encode ``frames_dir/frame_%05d.png`` into ``out_base`` + ``.mp4`` /
    ``.webm`` / ``.gif`` for each of ``formats``; returns the files written.
    The GIF is resampled to ``gif_fps`` and ``gif_width`` px wide."""
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    src = ["-framerate", str(fps), "-i", os.path.join(frames_dir, "frame_%05d.png")]
    done = []
    if "mp4" in formats:
        run_ffmpeg([*src, *X264, *ONE_THREAD, *BITEXACT, "-movflags", "+faststart",
                    out_base + ".mp4"])
        done.append(out_base + ".mp4")
    if "webm" in formats:
        run_ffmpeg([*src, *VP9, *ONE_THREAD, *BITEXACT, out_base + ".webm"])
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
        run_ffmpeg([*src, "-vf", vf, "-loop", "0", *BITEXACT, out_base + ".gif"])
        done.append(out_base + ".gif")
    for p in done:
        print("   %7.2f MB  %s" % (os.path.getsize(p) / 1e6, os.path.relpath(p)))
    return done


# ------------------------------------------------------------ manifests


def package_version():
    """voxanim's version (the single source is ``voxanim.__version__``)."""
    from . import __version__
    return __version__


def _command_output(cmd):
    """First line of ``cmd``'s stdout, or ``None`` if it cannot run or fails."""
    try:
        res = subprocess.run(cmd, cwd=ANIM, capture_output=True, text=True)
    except OSError:
        return None
    lines = res.stdout.strip().splitlines()
    return lines[0] if res.returncode == 0 and lines else None


def sha256(path):
    """Hex SHA-256 of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(name, opts, style, fps, outputs, **settings):
    """Everything that determined a render's output, as a JSON-able dict:
    the preset and its options, ``fps`` and other encode ``settings``, the
    :class:`~voxanim.scene.Style` (less its cache path), tool versions,
    platform, git commit (``None`` outside a checkout) and the SHA-256 of
    every file in ``outputs``."""
    st = dataclasses.asdict(style)
    st.pop("cache_dir")
    return {
        "preset": name,
        "options": presets.accepted_options(name, **(opts or {})),
        "fps": fps,
        **settings,
        "style": st,
        "versions": {
            "voxanim": package_version(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": PIL.__version__,
            "ffmpeg": _command_output(["ffmpeg", "-version"]),
        },
        "platform": platform.platform(),
        "git_commit": _command_output(["git", "rev-parse", "HEAD"]),
        "outputs": {os.path.basename(p): sha256(p) for p in outputs},
    }


def write_manifest(path, *args, **kw):
    """Write :func:`manifest` ``(*args, **kw)`` to ``path`` as indented JSON."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest(*args, **kw), f, indent=2, sort_keys=True)
        f.write("\n")
    print("   manifest   %s" % os.path.relpath(path))
    return path


# ------------------------------------------------------------ broadcast masters
#
# Frames stream straight from the workers into ffmpeg as 16-bit RGB: nothing
# touches the disk (one 8K 16-bit frame is ~200 MB).

RESOLUTIONS = {"1080": (1920, 1080), "4k": (3840, 2160), "8k": (7680, 4320)}
CODECS = {
    # 10-bit HEVC Main10 4:2:0 -- the HDR10 / HLG delivery format
    "hevc": dict(ext=".mp4", args=["-c:v", "libx265", "-preset", "medium", "-crf", "12",
                                   "-pix_fmt", "yuv420p10le", "-tag:v", "hvc1",
                                   *ONE_THREAD]),
    # ProRes 4444 XQ, 10-bit 4:4:4 -- a mastering / editing intermediate
    "prores": dict(ext=".mov", args=["-c:v", "prores_ks", "-profile:v", "5",
                                     "-pix_fmt", "yuv444p10le", "-vendor", "apl0"]),
    # 10-bit AV1
    "av1": dict(ext=".mp4", args=["-c:v", "libsvtav1", "-preset", "6", "-crf", "16",
                                  "-pix_fmt", "yuv420p10le", *ONE_THREAD]),
}
TRC = {"pq": "smpte2084", "hlg": "arib-std-b67", "sdr": "bt709"}
STRIP = 512                     # rows per strip when converting a frame (bounds temporaries)


def _master_frame(job):
    """Render one frame into a shared-memory slot as 16-bit RGB (no pickling
    of 200 MB frames); returns ``(idx, (peak, average))`` light levels in cd/m2
    (zeros for SDR)."""
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
            for y in range(0, shape[0], STRIP):
                out[y:y + STRIP] = hdr.to_uint16(np.clip(rgb[y:y + STRIP], 0, 1))
        else:
            rgb, emit = r.render_at(_W["anim"].scene, t, dt, hdr=True)
            cll, fsum = 0.0, 0.0
            for y in range(0, shape[0], STRIP):
                code = hdr.encode(rgb[y:y + STRIP], emit[y:y + STRIP], transfer)
                out[y:y + STRIP] = hdr.to_uint16(code)
                c, f = hdr.light_levels(rgb[y:y + STRIP], emit[y:y + STRIP])
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


def light_metadata(name, transfer, every=0.5, opts=None):
    """HDR10 content light levels ``(MaxCLL, MaxFALL)`` in cd/m2 for PQ
    (``None`` otherwise).  MaxFALL (a frame average) does not depend on
    resolution, so it is measured every ``every`` seconds on reference
    renders, plus 10%.  MaxCLL is a per-pixel peak, which *does* (small glints
    concentrate at 8K), so it is declared as the grade's ceiling,
    ``hdr.PEAK``: over-declaring is harmless, under-declaring makes TVs clip
    highlights."""
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
    """ffmpeg command line encoding raw ``rgb48le`` frames from stdin into
    ``path`` with ``codec`` and the colour signalling for ``transfer``
    (``levels`` = HDR10 ``(MaxCLL, MaxFALL)``, if known)."""
    space = "bt709" if transfer == "sdr" else "bt2020"
    vf = "scale=in_range=pc:out_range=tv:out_color_matrix=%s" % space
    cmd = [*FFMPEG, "-f", "rawvideo", "-pix_fmt", "rgb48le",
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
    cmd += BITEXACT
    cmd += ["-movflags", "+faststart", path] if CODECS[codec]["ext"] == ".mp4" else [path]
    return cmd


def master_path(name, res, fps, transfer, codec, out_dir=None, opts=None):
    """Output file of a master, e.g. ``build/masters/<media>_8k60_hdr-pq.mp4``."""
    out_dir = out_dir or os.path.join(BUILD, "masters")
    os.makedirs(out_dir, exist_ok=True)
    label = res + "p" if res.isdigit() else res
    grade = "sdr" if transfer == "sdr" else "hdr-" + transfer
    return os.path.join(out_dir, "%s%s_%s%d_%s%s" % (
        presets.get(name).media, presets.options_suffix(name, **(opts or {})), label, fps,
        grade, CODECS[codec]["ext"]))


def _stream_frames(sink, name, style, opts, times, fps, transfer, jobs, shape):
    """Render ``times`` in a pool of ``jobs`` workers and write each frame, in
    order, to the file ``sink`` as 16-bit RGB of ``shape``.  Frames travel
    through a ring of ``jobs + 1`` shared-memory slots.  Returns the measured
    ``(MaxCLL, MaxFALL)`` in cd/m2."""
    from multiprocessing import shared_memory
    nbytes = shape[0] * shape[1] * shape[2] * 2
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
                sink.write(slots[k % window].buf[:nbytes])
                cll, fall = max(cll, c), max(fall, f)
                submit(k + window)                     # reuse the slot just written
                if (k + 1) % PROGRESS_EVERY == 0 or k + 1 == len(times):
                    progress(k + 1, len(times), t0)
    finally:
        for sm in slots:
            sm.close()
            sm.unlink()
    return cll, fall


def master(name, res="8k", fps=60, transfer="pq", codec="hevc", jobs=None, quality="final",
           out_dir=None, seconds=None, opts=None):
    """Render ``name`` straight into a broadcast master; returns its path.

    ``res`` is a key of :data:`RESOLUTIONS`, ``transfer`` one of ``pq``,
    ``hlg``, ``sdr``, ``codec`` a key of :data:`CODECS`; ``seconds`` keeps
    only the start (tests).  Raises :class:`FFmpegError` if encoding fails.
    """
    from .scene import QUALITY, Style
    opts = opts or {}
    require_ffmpeg()
    W, H = RESOLUTIONS[res]
    style = Style(**QUALITY[quality]).at_scale(W / REF_W)
    anim = presets.build(name, **opts)
    times = frame_times(anim, fps)
    if seconds is not None:
        times = times[:int(round(seconds * fps))]
    path = master_path(name, res, fps, transfer, codec, out_dir, opts)
    levels = light_metadata(name, transfer, opts=opts)
    jobs = worker_budget(W, H, style.ss, jobs)
    print("== %s -> %s\n   %dx%d %d fps %s %s, %d frames, %d workers%s"
          % (name, os.path.relpath(path), W, H, fps, transfer.upper(), codec, len(times), jobs,
             "" if not levels else ", MaxCLL %d / MaxFALL %d" % levels), flush=True)
    Renderer(anim.cam, style)                         # warm the sprite cache once
    t0 = time.time()
    with ffmpeg_pipe(ffmpeg_master_cmd(path, W, H, fps, transfer, codec, levels)) as sink:
        cll, fall = _stream_frames(sink, name, style, opts, times, fps, transfer, jobs,
                                   (H, W, 3))
    print("   %.1f MB  %s  (%.0fs)" % (os.path.getsize(path) / 1e6, os.path.relpath(path),
                                      time.time() - t0))
    if levels:
        print("   measured at full res: MaxCLL %.0f, MaxFALL %.0f (declared %d / %d)"
              % (cll, fall, *levels))
    return path

"""What a card on the website shows: a poster and a small looping preview.

The site used to make these itself, from the full videos, on every build: it downloaded
600 MB of MP4s and ran ffmpeg on each.  They are made here instead, once, when an
animation is rendered, in each theme it is rendered in (the page shows the previews of
its own theme and streams the full video from the release), and they travel in the
release's ``voxanim-site.tar.gz``, so the site build encodes nothing.

For ``<media>.mp4`` this writes, next to it:

* ``<media>.jpg``: the frame at :data:`POSTER_AT` of the animation, :data:`POSTER_WIDTH`
  pixels wide;
* ``<media>-thumb.mp4``: the first half of the animation at 480 px and 10 frames a
  second, silent, holding its last frame for 2 s before it loops.

A dark video is ``<media>-dark.mp4``, so its previews are ``<media>-dark.jpg`` and
``<media>-dark-thumb.mp4``.

Measured on four animations of 1.10.0 (``conv2d``, ``bump-hunt``, ``gradient-descent``,
``vae-kl-term``): posters 12 to 30 KB (the 960 px posters before were 37 to 60 KB), clips
32 to 78 KB (47 to 101 KB at the 12 frames a second and crf 30 used before).  At 288 px
and 8 frames a second the captions stop being readable.

``python3 -m voxanim previews [NAME ...]`` makes them again from videos that exist,
without rendering.
"""

from __future__ import annotations

import os
import subprocess

from .render import BITEXACT, ONE_THREAD, FFmpegError, require_ffmpeg, run_ffmpeg

POSTER_AT = 0.4          # poster frame, as a fraction of the animation's length
POSTER_WIDTH = 640       # a card is at most ~600 CSS px wide
POSTER_QUALITY = 5       # ffmpeg -q:v for the JPEG (2 best .. 31 worst)
POSTER_SUFFIX = ".jpg"
# The clip the grid plays: ``half`` keeps the first half only, ``hold`` freezes the last
# frame that many seconds so the native loop pauses before it starts again.
# 480 wide: the hero at the top of the page shows this clip large, and at 320 it was soft
CLIP = dict(width=480, fps=10, crf=32, preset="slow", half=True, hold=2.0)
CLIP_SUFFIX = "-thumb.mp4"


def duration(mp4):
    """Length of a video in seconds (ffprobe)."""
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", mp4], check=True, capture_output=True, text=True)
    except FileNotFoundError:
        raise FFmpegError("ffprobe not found on PATH") from None
    except subprocess.CalledProcessError as e:
        raise FFmpegError("ffprobe failed on %s: %s" % (mp4, e.stderr.strip())) from None
    return float(out.stdout)


def paths(mp4):
    """``(poster, clip)``: where the previews of ``mp4`` go."""
    base = os.path.splitext(mp4)[0]
    return base + POSTER_SUFFIX, base + CLIP_SUFFIX


def poster(mp4, jpg, seconds):
    """Write the frame at ``seconds`` of ``mp4`` to ``jpg``."""
    tmp = jpg + ".part.jpg"                 # never leave half a file under the real name
    run_ffmpeg(["-ss", "%.3f" % seconds, "-i", mp4, "-frames:v", "1",
                "-vf", "scale=%d:-2:flags=lanczos" % POSTER_WIDTH,
                "-q:v", str(POSTER_QUALITY), *BITEXACT, tmp])
    os.replace(tmp, jpg)


def clip(mp4, out, seconds):
    """Write the grid's clip of ``mp4`` (``seconds`` long) to ``out``: H.264, silent,
    yuv420p, the index first so it starts at once."""
    c = CLIP
    cut = ["-t", "%.3f" % (seconds / 2)] if c["half"] else []
    tmp = out + ".part.mp4"
    run_ffmpeg([*cut, "-i", mp4, "-vf", "fps=%d,scale=%d:-2,tpad=stop_mode=clone:"
                "stop_duration=%.1f" % (c["fps"], c["width"], c["hold"]),
                "-c:v", "libx264", "-preset", c["preset"], "-crf", str(c["crf"]),
                "-pix_fmt", "yuv420p", "-an", *ONE_THREAD, *BITEXACT,
                "-movflags", "+faststart", tmp])
    os.replace(tmp, out)


def stale(mp4):
    """The previews of ``mp4`` that are missing or older than it."""
    made = os.path.getmtime(mp4)
    return [p for p in paths(mp4) if not os.path.exists(p) or os.path.getmtime(p) < made]


def make(mp4, force=False, quiet=False):
    """Make the poster and the clip of ``mp4`` next to it; returns ``(poster, clip)``.
    Previews newer than the video are kept unless ``force``."""
    require_ffmpeg()
    jpg, small = paths(mp4)
    todo = paths(mp4) if force else stale(mp4)
    if todo:
        seconds = duration(mp4)
        if jpg in todo:
            poster(mp4, jpg, POSTER_AT * seconds)
        if small in todo:
            clip(mp4, small, seconds)
    if not quiet:
        for p in (jpg, small):
            print("   %7.1f KB  %s%s" % (os.path.getsize(p) / 1e3, os.path.relpath(p),
                                         "" if p in todo else "  (kept)"))
    return jpg, small

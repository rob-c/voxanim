"""Command line: ``python3 -m voxanim <command> ...`` (run from ``anim/``).

    list                                  what can be rendered
    info     NAME                         timeline summary
    still    NAME -t 1.5 4 9              single frames -> build/stills/
             (dnn-eval / strided-cnn-classifier: --model FILE --digit N)
    compare  NAME -t 1.5 4                ours next to the original GIF frame
    sheet    NAME                         contact sheet of the whole animation
    render   NAME [NAME ...] | all        frames -> build/NAME/, then encode (web)
    encode   NAME [NAME ...] | all        re-encode existing frames
    master   NAME [NAME ...] | all        broadcast master, streamed (8K HDR by default)
    qa       [NAME ...]                   layout checks: clipped / colliding text, safe areas

Common options: --quality draft|preview|final|max, --scale 2 or --res 4k|8k,
--fps, --jobs.  ``render --install`` writes the encoded files into ../media
(under the site's file names) instead of build/out/; either way a JSON
manifest of the render goes to build/out/.  ``still --hdr pq`` writes the
16-bit PQ frame plus an SDR preview of the HDR grade.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
from PIL import Image

from . import presets
from .camera import REF_W
from .fonts import FontNotFoundError
from .gpt2_anim import PromptLengthError
from .raster import to_uint8
from .render import (
    BUILD,
    MEDIA,
    RESOLUTIONS,
    FFmpegError,
    encode,
    master,
    render_frames,
    run_ffmpeg,
    write_manifest,
)
from .scene import QUALITY, Renderer, Style

THUMB = (480, 270)              # contact-sheet thumbnail size, px
SHEET_COLS = 4


def make_style(args):
    """The :class:`Style` for ``--quality`` at ``--scale`` (or ``--res``)."""
    st = Style(**QUALITY[args.quality])
    scale = args.scale
    if getattr(args, "res", None):
        scale = RESOLUTIONS[args.res][0] / REF_W
    if scale != 1:
        st = st.at_scale(scale)
    return st


def opts(args):
    """Options for parameterised animations (``--model``, ``--digit``, ...)."""
    return {k: getattr(args, k, None) for k in ("model", "digit", "preprocess", "text")}


def _setup(args, subdir):
    """Preset, animation, renderer and output directory (``--out`` or
    ``build/<subdir>``) for a single-NAME command."""
    p = presets.get(args.name)
    anim = presets.build(args.name, **opts(args))
    r = Renderer(anim.cam, make_style(args))
    out = args.out or os.path.join(BUILD, subdir)
    os.makedirs(out, exist_ok=True)
    return p, anim, r, out


def _out_path(args, out, tail):
    """``out/<name><options suffix><tail>``."""
    return os.path.join(out, "%s%s%s" % (args.name, presets.options_suffix(
        args.name, **opts(args)), tail))


def _targets(names):
    """Preset names from the command line (``all`` means every preset)."""
    if names == ["all"]:
        return list(presets.REGISTRY)
    return names


# ------------------------------------------------------------ commands


def cmd_list(args):
    """Print every preset with its duration and one-line description (which
    builds it: may download data or train).  ``--names``: just the names."""
    if args.names:
        print("\n".join(presets.REGISTRY))
        return
    for p in presets.REGISTRY.values():
        anim = p.build()
        print("%-30s %5.1fs  %s" % (p.name, anim.duration, p.about.splitlines()[0]))


def cmd_prepare(args):
    """Download the datasets and train every model, so that later renders and
    the full test suite run offline.  Everything lands in the cache directory
    (``build/.cache`` or ``$VOXANIM_CACHE``); it is safe to run again."""
    names = args.names or list(presets.REGISTRY)
    for n, name in enumerate(names, 1):
        t0 = time.time()
        presets.build(name)
        print("[%2d/%d] %-30s ready (%.1fs)" % (n, len(names), name, time.time() - t0),
              flush=True)


def cmd_info(args):
    """Print an animation's timeline (and, for convolutions, each stage)."""
    p = presets.get(args.name)
    anim = presets.build(args.name, **opts(args))
    print("%s: %.2fs total, done at %.2fs (hold %.1fs, reset %.1fs)"
          % (p.name, anim.duration, anim.t_done, anim.hold, anim.reset))
    for st in getattr(anim, "stages", []):
        if not hasattr(st.spec, "padded_hw"):
            continue
        s = st.spec
        print("  stage %-12s in %s pad %s -> out %s x %d, kernel %s stride %s groups %d"
              % (s.name, s.padded_hw + (s.in_ch,), s.padding, s.out_hw, s.filters,
                 s.kernel, s.stride, s.groups))
        print("    %d steps, %.2fs..%.2fs; swaps at %s"
              % (len(st.steps), st.t_start, st.t_end,
                 ", ".join("%.1f" % a for a, _ in st.swaps.values())))


def cmd_still(args):
    """Render single frames at the times ``-t`` (SDR PNG, or ``--hdr`` 16-bit)."""
    _p, anim, r, out = _setup(args, "stills")
    for t in args.t:
        t0 = time.time()
        path = _out_path(args, out, "_%06.2f.png" % t)
        if args.hdr:
            from . import hdr
            rgb, emit = r.render_at(anim.scene, t, 1.0 / args.fps, hdr=True)
            code = hdr.to_uint16(hdr.encode(rgb, emit, args.hdr))
            hpath = path.replace(".png", "_%s16.png" % args.hdr)
            _save16(hpath, code)
            Image.fromarray(to_uint8(hdr.preview_sdr(rgb, emit))).save(path)
            cll, fall = hdr.light_levels(rgb, emit)
            print("%s  (+ %s: MaxCLL %.0f, FALL %.0f cd/m2)  (%.1fs)"
                  % (path, os.path.basename(hpath), cll, fall, time.time() - t0))
            continue
        img = r.render_at(anim.scene, t, 1.0 / args.fps)
        Image.fromarray(to_uint8(img)).save(path)
        print("%s  (%.2fs)" % (path, time.time() - t0))


def _save16(path, code):
    """16-bit RGB PNG via ffmpeg (Pillow cannot write 48-bit RGB)."""
    h, w = code.shape[:2]
    run_ffmpeg(["-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", "%dx%d" % (w, h), "-i", "-",
                "-frames:v", "1", path], stdin=code.astype("<u2").tobytes())


def reference_frame(p, t, fps=10):
    """Frame of preset ``p``'s original GIF at ``t`` seconds (the originals run
    at ``fps``), or ``None`` if it has none or the file is missing."""
    if not p.reference:
        return None
    path = os.path.join(MEDIA, p.reference)
    if not os.path.exists(path):
        return None
    im = Image.open(path)
    n = getattr(im, "n_frames", 1)
    im.seek(min(n - 1, int(round(t * fps))))
    return im.convert("RGB")


def cmd_compare(args):
    """Our frame above (or ``--overlay``-blended with) the original GIF's."""
    p, anim, r, out = _setup(args, "compare")
    for t in args.t:
        ours = Image.fromarray(to_uint8(r.render_at(anim.scene, t, 1.0 / args.fps)))
        ref = reference_frame(p, t if args.ref_t is None else args.ref_t)
        if ref is None:
            raise SystemExit("%s has no reference GIF" % p.name)
        ref = ref.resize(ours.size, Image.LANCZOS)
        if args.overlay:
            sheet = Image.blend(ours, ref, 0.5)
        else:
            sheet = Image.new("RGB", (ours.width, ours.height * 2), "white")
            sheet.paste(ours, (0, 0))
            sheet.paste(ref, (0, ours.height))
        path = _out_path(args, out, "_%06.2f.png" % t)
        sheet.save(path)
        print(path)


def cmd_sheet(args):
    """A contact sheet of ``--count`` evenly spaced frames."""
    _p, anim, r, out = _setup(args, "sheets")
    n = args.count
    tw, th = THUMB
    ts = np.linspace(0, anim.duration, n, endpoint=False)
    thumbs = []
    for t in ts:
        im = Image.fromarray(to_uint8(r.render_at(anim.scene, float(t), 1.0 / args.fps)))
        thumbs.append(im.resize(THUMB, Image.LANCZOS))
    rows = (n + SHEET_COLS - 1) // SHEET_COLS
    sheet = Image.new("RGB", (tw * SHEET_COLS, th * rows), "white")
    for k, im in enumerate(thumbs):
        sheet.paste(im, ((k % SHEET_COLS) * tw, (k // SHEET_COLS) * th))
    path = _out_path(args, out, ".png")
    sheet.save(path)
    print(path)


def _encode(name, args, style):
    """Encode ``build/<name><suffix>/`` frames into the web formats and write
    the render's manifest to ``build/out/<media><suffix>.json``."""
    p = presets.get(name)
    suffix = presets.options_suffix(name, **opts(args))
    frames = os.path.join(BUILD, name) + suffix
    base = os.path.join(MEDIA if args.install else os.path.join(BUILD, "out"),
                        p.media + suffix)
    gif_width = style.width if args.gif_width is None else args.gif_width
    formats = args.formats.split(",")
    outputs = encode(frames, args.fps, base, formats, gif_fps=args.gif_fps,
                     gif_width=gif_width)
    write_manifest(os.path.join(BUILD, "out", p.media + suffix + ".json"), name, opts(args),
                   style, args.fps, outputs, quality=args.quality, formats=formats,
                   gif_fps=args.gif_fps, gif_width=gif_width)


def cmd_render(args):
    """Render frames to ``build/<name>/`` and encode them for the web."""
    style = make_style(args)
    for name in _targets(args.names):
        frames = os.path.join(BUILD, name) + presets.options_suffix(name, **opts(args))
        render_frames(name, style, args.fps, args.jobs, frames, opts=opts(args))
        _encode(name, args, style)


def cmd_encode(args):
    """Re-encode frames a previous ``render`` left in ``build/<name>/``."""
    style = make_style(args)
    for name in _targets(args.names):
        _encode(name, args, style)


def cmd_master(args):
    """Stream each animation into a broadcast master."""
    for name in _targets(args.names):
        master(name, args.res, args.fps, args.hdr, args.codec, args.jobs, args.quality,
               args.out, args.seconds, opts(args))


def cmd_qa(args):
    """Layout checks; exits 1 if any animation has an issue."""
    from .qa import check
    names = _targets(args.names) if args.names else list(presets.REGISTRY)
    bad = 0
    for name in names:
        bad += check(name, args.samples)
    raise SystemExit(1 if bad else 0)


def parser():
    """The argument parser for every command."""
    ap = argparse.ArgumentParser(prog="voxanim", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def model_opts(sp):
        sp.add_argument("--model", help="model file for dnn-eval (.npz/.pt/.h5/.json)")
        sp.add_argument("--digit", type=int, help="MNIST test image index")
        sp.add_argument("--preprocess", choices=("unit", "standard", "centered", "raw"),
                        help="how the model expects pixels (default: from the file, "
                             "else unit)")
        sp.add_argument("--text", help="prompt for gpt2-attention")

    def common(sp):
        model_opts(sp)
        sp.add_argument("--quality", choices=sorted(QUALITY), default="final")
        sp.add_argument("--scale", type=float, default=1.0)
        sp.add_argument("--res", choices=sorted(RESOLUTIONS))
        sp.add_argument("--fps", type=int, default=30)
        sp.add_argument("--out")

    sp = sub.add_parser("list")
    sp.add_argument("--names", action="store_true", help="names only, without building")
    sp.set_defaults(fn=cmd_list)
    sp = sub.add_parser("prepare", help="download data and train every model (offline after)")
    sp.add_argument("names", nargs="*", help="only these presets (default: all)")
    sp.set_defaults(fn=cmd_prepare)
    sp = sub.add_parser("info")
    sp.add_argument("name")
    model_opts(sp)
    sp.set_defaults(fn=cmd_info)
    for cmd, fn in (("still", cmd_still), ("compare", cmd_compare)):
        sp = sub.add_parser(cmd)
        sp.add_argument("name")
        sp.add_argument("-t", type=float, nargs="+", required=True)
        common(sp)
        if cmd == "compare":
            sp.add_argument("--ref-t", type=float, help="reference time if it differs")
            sp.add_argument("--overlay", action="store_true")
        else:
            sp.add_argument("--hdr", choices=("pq", "hlg"))
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("master")
    sp.add_argument("names", nargs="+")
    sp.add_argument("--res", choices=sorted(RESOLUTIONS), default="8k")
    sp.add_argument("--fps", type=int, default=60)
    sp.add_argument("--hdr", choices=("pq", "hlg", "sdr"), default="pq")
    sp.add_argument("--codec", choices=("hevc", "prores", "av1"), default="hevc")
    sp.add_argument("--jobs", type=int)
    sp.add_argument("--quality", choices=sorted(QUALITY), default="final")
    sp.add_argument("--seconds", type=float, help="only the first N seconds (tests)")
    model_opts(sp)
    sp.add_argument("--out")
    sp.set_defaults(fn=cmd_master)
    sp = sub.add_parser("qa")
    sp.add_argument("names", nargs="*")
    sp.add_argument("--samples", type=int, default=120)
    sp.set_defaults(fn=cmd_qa)
    sp = sub.add_parser("sheet")
    sp.add_argument("name")
    sp.add_argument("--count", type=int, default=16)
    common(sp)
    sp.set_defaults(fn=cmd_sheet)
    for cmd, fn in (("render", cmd_render), ("encode", cmd_encode)):
        sp = sub.add_parser(cmd)
        sp.add_argument("names", nargs="+")
        sp.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
        sp.add_argument("--formats", default="mp4,gif")
        sp.add_argument("--gif-fps", type=int, default=15)
        sp.add_argument("--gif-width", type=int, help="default: the render width")
        sp.add_argument("--install", action="store_true",
                        help="write into $VOXANIM_MEDIA (default ../media) under the "
                             "site's names")
        common(sp)
        sp.set_defaults(fn=fn)
    return ap


def main(argv=None):
    """Run the command line; expected failures (unknown preset, ffmpeg, fonts)
    become a one-line error and exit status 2."""
    args = parser().parse_args(argv)
    try:
        args.fn(args)
    except (presets.UnknownPreset, FFmpegError, FontNotFoundError, PromptLengthError) as e:
        print("voxanim: error: %s" % e, file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()

"""Command line: ``python3 -m voxanim <command> ...`` (run from ``anim/``).

    list                                  what can be rendered
    info     NAME                         timeline summary
    still    NAME -t 1.5 4 9              single frames -> build/stills/
             (dnn-eval / strided-cnn-classifier: --model FILE --digit N)
    compare  NAME -t 1.5 4                ours next to the original GIF frame
    sheet    NAME                         contact sheet of the whole animation
    render   NAME [NAME ...] | all        frames -> build/NAME/, then encode (web)
    encode   NAME                         re-encode existing frames
    master   NAME [NAME ...] | all        broadcast master, streamed (8K HDR by default)
    qa       [NAME ...]                   layout checks: clipped / colliding text, safe areas

Common options: --quality draft|preview|final|max, --scale 2 or --res 4k|8k,
--fps, --jobs.  ``render --install`` writes the encoded files into ../media
(under the site's file names) instead of build/out/.  ``still --hdr pq`` writes
the 16-bit PQ frame plus an SDR preview of the HDR grade.
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace

import numpy as np
from PIL import Image

from . import presets
from .raster import to_uint8
from .render import BUILD, MEDIA, RESOLUTIONS, encode, master, render_frames
from .scene import QUALITY, Renderer, Style


def make_style(args):
    st = Style(**QUALITY[args.quality])
    scale = args.scale
    if getattr(args, "res", None):
        scale = RESOLUTIONS[args.res][0] / 960.0
    if scale != 1:
        st = st.at_scale(scale)
    if getattr(args, "no_watermark", False):
        st = replace(st, watermark=False)
    return st


# ------------------------------------------------------------ commands


def cmd_list(args):
    for p in presets.REGISTRY.values():
        anim = p.build()
        print("%-30s %5.1fs  %s" % (p.name, anim.duration, p.about.splitlines()[0]))


def opts(args):
    """Options for parameterised animations (``--model``, ``--digit``, ...)."""
    return {k: getattr(args, k, None) for k in ("model", "digit", "preprocess", "text")}


def cmd_info(args):
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
    p = presets.get(args.name)
    anim = presets.build(args.name, **opts(args))
    r = Renderer(anim.cam, make_style(args))
    out = args.out or os.path.join(BUILD, "stills")
    os.makedirs(out, exist_ok=True)
    for t in args.t:
        t0 = time.time()
        path = os.path.join(out, "%s%s_%06.2f.png" % (p.name, presets.options_suffix(
            args.name, **opts(args)), t))
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
    import subprocess
    h, w = code.shape[:2]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt",
                    "rgb48le", "-s", "%dx%d" % (w, h), "-i", "-", "-frames:v", "1", path],
                   input=code.astype("<u2").tobytes(), check=True)


def reference_frame(p, t, fps=10):
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
    p = presets.get(args.name)
    anim = presets.build(args.name, **opts(args))
    r = Renderer(anim.cam, make_style(args))
    out = args.out or os.path.join(BUILD, "compare")
    os.makedirs(out, exist_ok=True)
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
        path = os.path.join(out, "%s%s_%06.2f.png" % (p.name, presets.options_suffix(
            args.name, **opts(args)), t))
        sheet.save(path)
        print(path)


def cmd_sheet(args):
    p = presets.get(args.name)
    anim = presets.build(args.name, **opts(args))
    r = Renderer(anim.cam, make_style(args))
    n = args.count
    ts = np.linspace(0, anim.duration, n, endpoint=False)
    thumbs = []
    for t in ts:
        im = Image.fromarray(to_uint8(r.render_at(anim.scene, float(t), 1.0 / args.fps)))
        thumbs.append(im.resize((480, 270), Image.LANCZOS))
    cols = 4
    rows = (n + cols - 1) // cols
    sheet = Image.new("RGB", (480 * cols, 270 * rows), "white")
    for k, im in enumerate(thumbs):
        sheet.paste(im, ((k % cols) * 480, (k // cols) * 270))
    out = args.out or os.path.join(BUILD, "sheets")
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "%s%s.png" % (p.name, presets.options_suffix(args.name,
                                                                            **opts(args))))
    sheet.save(path)
    print(path)


def _targets(names):
    if names == ["all"]:
        return list(presets.REGISTRY)
    return names


def cmd_render(args):
    style = make_style(args)
    for name in _targets(args.names):
        p = presets.get(name)
        frames = os.path.join(BUILD, name)
        suffix = presets.options_suffix(name, **opts(args))
        frames = frames + suffix
        render_frames(name, style, args.fps, args.jobs, frames, opts=opts(args))
        base = os.path.join(MEDIA if args.install else os.path.join(BUILD, "out"),
                            p.media + suffix)
        encode(frames, args.fps, base, args.formats.split(","), gif_fps=args.gif_fps,
               gif_width=style.width if args.gif_width is None else args.gif_width)


def cmd_encode(args):
    for name in _targets(args.names):
        p = presets.get(name)
        base = os.path.join(MEDIA if args.install else os.path.join(BUILD, "out"), p.media)
        encode(os.path.join(BUILD, name), args.fps, base, args.formats.split(","),
               gif_fps=args.gif_fps, gif_width=args.gif_width or 960)


def cmd_master(args):
    for name in _targets(args.names):
        master(name, args.res, args.fps, args.hdr, args.codec, args.jobs, args.quality,
               args.out, args.seconds, opts(args))


def cmd_qa(args):
    from .qa import check
    names = _targets(args.names) if args.names else list(presets.REGISTRY)
    bad = 0
    for name in names:
        bad += check(name, args.samples)
    raise SystemExit(1 if bad else 0)


def main(argv=None):
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
        sp.add_argument("--no-watermark", action="store_true")
        sp.add_argument("--out")

    sub.add_parser("list").set_defaults(fn=cmd_list)
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
        sp.add_argument("--gif-width", type=int)
        sp.add_argument("--install", action="store_true",
                        help="write into ../media under the site's names")
        common(sp)
        sp.set_defaults(fn=fn)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()

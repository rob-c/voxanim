"""Gather the files of a GitHub release into one directory.

    python3 tools/release_assets.py OUT

``OUT`` receives:

* every video as a file of its own, in both themes: ``<media>.mp4`` and
  ``<media>-dark.mp4``, so the website's player can stream any of them straight from the
  release;
* ``voxanim-gifs-light.tar`` and ``voxanim-gifs-dark.tar``: every GIF of each theme.  They
  are stored, not compressed: gzip -9 took 1.8% off twenty of the 1.10.0 GIFs (69.3 MB to
  68.0 MB) and xz 3.7%, which is not worth unpacking 575 MB twice;
* ``voxanim-site.tar.gz``: all the website build needs besides the page itself: each
  animation's poster (``<media>.jpg``) and small preview clip (``<media>-thumb.mp4``) in
  both themes (``<media>-dark.jpg``, ``<media>-dark-thumb.mp4``), as a render or
  ``python3 -m voxanim previews`` makes them, and every render manifest
  (``<media>.json``, ``<media>-dark.json``);
* ``voxanim-weights.tar.gz``: every trained model and recorded run in the
  cache (the ``<name>-<12 hex digits>.npz`` files of ``build/.cache`` or
  ``$VOXANIM_CACHE``);
* ``SHA256SUMS`` over all of the above (``shasum -a 256 -c SHA256SUMS``).

The archives are packed deterministically (sorted, no timestamps), so the same files give
the same archive.  GitHub is understood to cap a release at 1,000 files, and 1.10.0 was at
899 with a video, a GIF and a manifest per animation in one theme: hence the archives.  The
count is printed against the cap, and a release over it is refused.

The build refuses to run when a registered animation has no render in either theme, a
manifest's recorded hash does not match its file, or a poster or preview is missing or
older than its video.  ``--light-only`` allows a release without dark renders;
``--allow-partial`` leaves out animations that have not been rendered at all (to try the
packaging on two or three).  Render first:

    python3 -m voxanim render all --theme both --clean-frames

Then, for example:

    gh release create v1.1.0 --title "voxanim 1.1.0" --notes-file NOTES.md OUT/*
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from voxanim import presets, previews  # noqa: E402
from voxanim.cache import DEFAULT_CACHE  # noqa: E402
from voxanim.render import BUILD  # noqa: E402

WEIGHTS = re.compile(r"-[0-9a-f]{12}\.npz$")
THEMES = {"light": "", "dark": "-dark"}        # theme -> what follows <media> in a file name
FILE_CAP = 1000                                # files GitHub lets one release hold
GIFS = "voxanim-gifs-%s.tar"
SITE = "voxanim-site.tar.gz"
WEIGHTS_FILE = "voxanim-weights.tar.gz"
SUMS = "SHA256SUMS"


def sha256(path):
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def _check_theme(out_dir, p, theme, problems):
    """The verified files of one theme's render of preset ``p``: ``{"json": path, "mp4":
    path, "gif": path, "poster": path, "clip": path}``, or ``None`` with the reasons added
    to ``problems``."""
    base = p.media + THEMES[theme]
    manifest = os.path.join(out_dir, base + ".json")
    with open(manifest, encoding="utf-8") as fh:
        recorded = json.load(fh)
    found, ok = {"json": manifest}, True
    # a manifest from before themes were recorded is a light render
    if recorded.get("theme", "light") != theme:
        problems.append("%s: %s.json records a %s render"
                        % (p.name, base, recorded.get("theme", "light")))
        ok = False
    for ext in ("mp4", "gif"):
        f = "%s.%s" % (base, ext)
        path = os.path.join(out_dir, f)
        if f not in recorded["outputs"]:
            problems.append("%s: no %s in its %s render (its manifest lists none)"
                            % (p.name, ext.upper(), theme))
        elif not os.path.exists(path):
            problems.append("%s: %s is missing" % (p.name, f))
        elif sha256(path) != recorded["outputs"][f]:
            problems.append("%s: %s does not match its manifest" % (p.name, f))
        else:
            found[ext] = path
            continue
        ok = False
    if "mp4" in found:
        found["poster"], found["clip"] = previews.paths(found["mp4"])
        for f in previews.stale(found["mp4"]):
            problems.append("%s: %s is missing or older than its video "
                            "(python3 -m voxanim previews %s)"
                            % (p.name, os.path.basename(f), p.name))
            ok = False
    return found if ok else None


def check_renders(out_dir, registry=None, light_only=False, allow_partial=False):
    """Every animation's render, verified against its manifests: a list of
    ``{"name", "media", "light": files, "dark": files or None}`` with
    ``files`` as :func:`_check_theme` returns them.  Raises ``SystemExit`` naming
    everything that is missing or stale.  ``light_only``: an animation need not have a dark
    render (one that is there is still checked and packed); ``allow_partial``: an
    animation with no render at all is left out instead."""
    registry = presets.REGISTRY if registry is None else registry
    found, problems = [], []
    for p in registry.values():
        have = {t: os.path.exists(os.path.join(out_dir, p.media + sfx + ".json"))
                for t, sfx in THEMES.items()}
        if not have["light"]:
            if not (allow_partial and not have["dark"]):
                problems.append("%s: no light render (expected %s)"
                                % (p.name, os.path.join(out_dir, p.media + ".json")))
            continue
        entry = {"name": p.name, "media": p.media, "dark": None,
                 "light": _check_theme(out_dir, p, "light", problems)}
        if have["dark"]:
            entry["dark"] = _check_theme(out_dir, p, "dark", problems)
        elif not light_only:
            problems.append("%s: no dark render (expected %s; --light-only releases without)"
                            % (p.name, os.path.join(out_dir, p.media + "-dark.json")))
        found.append(entry)
    if problems:
        sys.exit("release: not ready:\n  " + "\n  ".join(problems))
    return found


def pack(path, files, compress=False):
    """Write ``files`` (paths; stored under their base names, sorted) to the tar ``path``,
    gzipped if ``compress``, deterministically: no timestamps, owners or modes that vary,
    so the same files give the same archive.  Returns the names packed."""
    files = sorted(files, key=os.path.basename)
    with open(path, "wb") as raw:
        zipped = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0,
                               compresslevel=9) if compress else None
        with tarfile.open(fileobj=zipped or raw, mode="w", format=tarfile.PAX_FORMAT) as tf:
            for f in files:
                info = tarfile.TarInfo(os.path.basename(f))
                info.size, info.mtime, info.mode = os.path.getsize(f), 0, 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                with open(f, "rb") as fh:
                    tf.addfile(info, fh)
        if zipped:
            zipped.close()
    return [os.path.basename(f) for f in files]


def pack_weights(cache_dir, path):
    """Write the cache's trained models to ``path`` as a deterministic tar.gz;
    returns the file names packed."""
    return pack(path, [os.path.join(cache_dir, n) for n in os.listdir(cache_dir)
                       if WEIGHTS.search(n)], compress=True)


def plan(renders):
    """What a release of ``renders`` holds: ``(videos, archives)`` with ``videos`` the MP4
    paths to copy and ``archives`` a ``{file name: [paths]}`` of what each tar packs."""
    videos, site = [], []
    gifs = {theme: [] for theme in THEMES}
    for r in renders:
        for theme in THEMES:
            if r[theme]:
                videos.append(r[theme]["mp4"])
                gifs[theme].append(r[theme]["gif"])
                site += [r[theme]["poster"], r[theme]["clip"], r[theme]["json"]]
    archives = {GIFS % theme: files for theme, files in gifs.items() if files}
    archives[SITE] = site
    return videos, archives


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="directory to fill (created; must be empty)")
    ap.add_argument("--renders", default=os.path.join(BUILD, "out"),
                    help="where the renders are (default build/out)")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="the model cache to pack")
    ap.add_argument("--light-only", action="store_true",
                    help="release animations that have no dark render too")
    ap.add_argument("--allow-partial", action="store_true",
                    help="leave out animations with no render at all (to try the packaging)")
    args = ap.parse_args(argv)
    if os.path.exists(args.out) and os.listdir(args.out):
        sys.exit("release: %s is not empty" % args.out)
    renders = check_renders(args.renders, light_only=args.light_only,
                            allow_partial=args.allow_partial)
    if not renders:
        sys.exit("release: no renders in %s" % args.renders)
    videos, archives = plan(renders)
    total = len(videos) + len(archives) + 2          # and the weights and SHA256SUMS
    if total > FILE_CAP:
        sys.exit("release: %d files, and a release holds at most %d" % (total, FILE_CAP))
    os.makedirs(args.out, exist_ok=True)
    made = []
    for f in videos:
        shutil.copyfile(f, os.path.join(args.out, os.path.basename(f)))
        made.append(os.path.basename(f))
    for name, files in archives.items():
        pack(os.path.join(args.out, name), files, compress=name.endswith(".gz"))
        made.append(name)
    weights = pack_weights(args.cache, os.path.join(args.out, WEIGHTS_FILE))
    made.append(WEIGHTS_FILE)
    with open(os.path.join(args.out, SUMS), "w", encoding="utf-8") as fh:
        for n in sorted(made):
            fh.write("%s  %s\n" % (sha256(os.path.join(args.out, n)), n))
    dark = sum(1 for r in renders if r["dark"])
    left_out = len(presets.REGISTRY) - len(renders)
    print("%d animations (%d with a dark render%s), %d models packed -> %s"
          % (len(renders), dark, ", %d left out" % left_out if left_out else "",
             len(weights), args.out))
    for name in (*archives, WEIGHTS_FILE):
        print("  %8.1f MB  %s" % (os.path.getsize(os.path.join(args.out, name)) / 1e6, name))
    print("%d files of the %d a release may hold" % (len(made) + 1, FILE_CAP))


if __name__ == "__main__":
    main()

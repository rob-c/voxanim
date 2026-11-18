"""Gather the files of a GitHub release into one directory.

    python3 tools/release_assets.py OUT

``OUT`` receives a folder per repository the release goes to, each with its own
``SHA256SUMS`` (``shasum -a 256 -c SHA256SUMS``).  ``OUT/voxanim-light`` and
``OUT/voxanim-dark``, for the two media repositories, which hold nothing but releases:

* every video of the theme as a file of its own, ``<media>.mp4`` or ``<media>-dark.mp4``,
  so the website's player can stream any of them straight from the release;
* ``voxanim-gifs-light.tar`` or ``voxanim-gifs-dark.tar``: every GIF of the theme.  They
  are stored, not compressed: gzip -9 took 1.8% off twenty of the 1.10.0 GIFs (69.3 MB to
  68.0 MB) and xz 3.7%, which is not worth unpacking 575 MB twice.

And ``OUT/voxanim``, for the release of the source repository itself:

* ``voxanim-site.tar.gz``: all the website build needs besides the page itself: each
  animation's poster (``<media>.jpg``) and small preview clip (``<media>-thumb.mp4``) in
  both themes (``<media>-dark.jpg``, ``<media>-dark-thumb.mp4``), as a render or
  ``python3 -m voxanim previews`` makes them, and every render manifest
  (``<media>.json``, ``<media>-dark.json``);
* ``voxanim-weights.tar.gz``: every trained model and recorded run in the
  cache (the ``<name>-<12 hex digits>.npz`` files of ``build/.cache`` or
  ``$VOXANIM_CACHE``).

The archives are packed deterministically (sorted, no timestamps), so the same files give
the same archive.  GitHub is understood to cap a release at 1,000 files, and 1.10.0 was at
899 with a video, a GIF and a manifest per animation in one theme: hence the archives.
Each folder's count is printed against the cap, and a release over it is refused.

The build refuses to run when a registered animation has no render in either theme, a
manifest's recorded hash does not match its file, or a poster or preview is missing or
older than its video.  ``--light-only`` allows a release without dark renders;
``--allow-partial`` leaves out animations that have not been rendered at all (to try the
packaging on two or three).  Render first:

    python3 -m voxanim render all --theme both --clean-frames

Then, for example:

    gh release create v1.1.0 -R OWNER/voxanim-light --title "1.1.0, light" OUT/voxanim-light/*
    gh release create v1.1.0 -R OWNER/voxanim-dark --title "1.1.0, dark" OUT/voxanim-dark/*
    gh release create v1.1.0 --title "voxanim 1.1.0" --notes-file NOTES.md OUT/voxanim/*

in that order: the website is rebuilt when the last one is published, and plays the videos
of the other two.
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
MAIN = "voxanim"                               # the folder for the source repository's release
MEDIA = "voxanim-%s"                           # and for a theme's media repository


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
    """What a release of ``renders`` holds: ``{folder: (videos, archives)}`` with ``videos``
    the MP4 paths to copy there and ``archives`` a ``{file name: [paths]}`` of what each
    tar there packs.  A theme with no render has no folder."""
    site = []
    videos = {theme: [] for theme in THEMES}
    gifs = {theme: [] for theme in THEMES}
    for r in renders:
        for theme in THEMES:
            if r[theme]:
                videos[theme].append(r[theme]["mp4"])
                gifs[theme].append(r[theme]["gif"])
                site += [r[theme]["poster"], r[theme]["clip"], r[theme]["json"]]
    folders = {MEDIA % theme: (videos[theme], {GIFS % theme: gifs[theme]})
               for theme in THEMES if videos[theme]}
    folders[MAIN] = ([], {SITE: site})
    return folders


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
    folders = plan(renders)
    for folder, (videos, archives) in folders.items():
        total = len(videos) + len(archives) + 1 + (folder == MAIN)   # SHA256SUMS; weights
        if total > FILE_CAP:
            sys.exit("release: %d files for %s, and a release holds at most %d"
                     % (total, folder, FILE_CAP))
    counts, sizes, weights = {}, {}, []
    for folder, (videos, archives) in folders.items():
        dest = os.path.join(args.out, folder)
        os.makedirs(dest)
        made = []
        for f in videos:
            shutil.copyfile(f, os.path.join(dest, os.path.basename(f)))
            made.append(os.path.basename(f))
        for name, files in archives.items():
            pack(os.path.join(dest, name), files, compress=name.endswith(".gz"))
            made.append(name)
        if folder == MAIN:
            weights = pack_weights(args.cache, os.path.join(dest, WEIGHTS_FILE))
            made.append(WEIGHTS_FILE)
        with open(os.path.join(dest, SUMS), "w", encoding="utf-8") as fh:
            for n in sorted(made):
                fh.write("%s  %s\n" % (sha256(os.path.join(dest, n)), n))
        counts[folder] = len(made) + 1
        for n in made:
            if not n.endswith(".mp4"):
                sizes[os.path.join(folder, n)] = os.path.getsize(os.path.join(dest, n))
    dark = sum(1 for r in renders if r["dark"])
    left_out = len(presets.REGISTRY) - len(renders)
    print("%d animations (%d with a dark render%s), %d models packed -> %s"
          % (len(renders), dark, ", %d left out" % left_out if left_out else "",
             len(weights), args.out))
    for name, size in sizes.items():
        print("  %8.1f MB  %s" % (size / 1e6, name))
    for folder, count in counts.items():
        print("%s: %d files of the %d a release may hold" % (folder, count, FILE_CAP))


if __name__ == "__main__":
    main()

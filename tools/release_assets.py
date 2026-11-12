"""Gather the files of a GitHub release into one directory.

    python3 tools/release_assets.py OUT

``OUT`` receives:

* every ``build/out/<name>.mp4``, ``.gif`` and ``.json`` (the renders and
  their manifests; render them first, ``python3 -m voxanim render all``),
* ``voxanim-weights.tar.gz``: every trained model and recorded run in the
  cache (the ``<name>-<12 hex digits>.npz`` files of ``build/.cache`` or
  ``$VOXANIM_CACHE``), packed deterministically (sorted, no timestamps), so
  the same files give the same archive,
* ``SHA256SUMS`` over all of the above (``shasum -a 256 -c SHA256SUMS``).

The build refuses to run when a registered animation has no render, or a
manifest's recorded hash does not match its file.  Then, for example:

    gh release create v1.1.0 --title "voxanim 1.1.0" --notes-file NOTES.md OUT/*
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from voxanim import presets  # noqa: E402
from voxanim.cache import DEFAULT_CACHE  # noqa: E402
from voxanim.render import BUILD  # noqa: E402

WEIGHTS = re.compile(r"-[0-9a-f]{12}\.npz$")


def sha256(path):
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def check_renders(out_dir):
    """The ``(name, [files])`` of every animation's render, verified against its
    manifest; raises ``SystemExit`` naming what is missing or stale."""
    found, problems = [], []
    for p in presets.REGISTRY.values():
        manifest = os.path.join(out_dir, p.media + ".json")
        if not os.path.exists(manifest):
            problems.append("%s: no render (expected %s)" % (p.name, manifest))
            continue
        with open(manifest, encoding="utf-8") as fh:
            outputs = json.load(fh)["outputs"]
        files = [manifest]
        for f, digest in outputs.items():
            path = os.path.join(out_dir, f)
            if not os.path.exists(path):
                problems.append("%s: %s is missing" % (p.name, f))
            elif sha256(path) != digest:
                problems.append("%s: %s does not match its manifest" % (p.name, f))
            else:
                files.append(path)
        found.append((p.name, files))
    if problems:
        sys.exit("release: not ready:\n  " + "\n  ".join(problems))
    return found


def pack_weights(cache_dir, path):
    """Write the cache's trained models to ``path`` as a deterministic tar.gz;
    returns the file names packed."""
    names = sorted(n for n in os.listdir(cache_dir) if WEIGHTS.search(n))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tf:
        for n in names:
            with open(os.path.join(cache_dir, n), "rb") as fh:
                data = fh.read()
            info = tarfile.TarInfo(n)
            info.size, info.mtime, info.mode, info.uid, info.gid = len(data), 0, 0o644, 0, 0
            info.uname = info.gname = ""
            tf.addfile(info, io.BytesIO(data))
    with open(path, "wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0,
                                               compresslevel=9) as gz:
        gz.write(buf.getvalue())
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", help="directory to fill (created; must be empty)")
    ap.add_argument("--renders", default=os.path.join(BUILD, "out"),
                    help="where the renders are (default build/out)")
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="the model cache to pack")
    args = ap.parse_args(argv)
    if os.path.exists(args.out) and os.listdir(args.out):
        sys.exit("release: %s is not empty" % args.out)
    os.makedirs(args.out, exist_ok=True)
    renders = check_renders(args.renders)
    copied = []
    for _name, files in renders:
        for f in files:
            shutil.copyfile(f, os.path.join(args.out, os.path.basename(f)))
            copied.append(os.path.basename(f))
    weights = pack_weights(args.cache, os.path.join(args.out, "voxanim-weights.tar.gz"))
    copied.append("voxanim-weights.tar.gz")
    with open(os.path.join(args.out, "SHA256SUMS"), "w", encoding="utf-8") as fh:
        for n in sorted(copied):
            fh.write("%s  %s\n" % (sha256(os.path.join(args.out, n)), n))
    print("%d animations, %d files, %d models packed -> %s"
          % (len(renders), len(copied) + 1, len(weights), args.out))


if __name__ == "__main__":
    main()

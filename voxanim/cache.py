"""Everything voxanim keeps on disk between runs: downloads and trained weights.

Two kinds of file live in the cache directory (``build/.cache`` next to the
package, or ``$VOXANIM_CACHE``):

* **Downloads** (MNIST, GPT-2): fetched once by :func:`download`, which
  checks a pinned SHA-256 before the file is moved into place, so a
  truncated or substituted file can never be used.
* **Trained weights**: every model is trained by seeded, deterministic numpy
  code and saved by :func:`cached_npz`.  The file name carries a hash of
  *every* setting that affects the result (plus :data:`TRAIN_VERSION`), so
  changing a hyperparameter can never silently load stale weights; the
  settings are also stored inside the file for auditing.

Writes go to a per-process temporary name and are renamed into place, so an
interrupted run or two runs at once never leave a half-written file behind.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE = os.environ.get("VOXANIM_CACHE") or os.path.join(HERE, "..", "build", ".cache")

# Bump when training code changes in a way that changes the weights it produces
# (an initialisation, an optimiser detail): every cached model is retrained.
TRAIN_VERSION = 1


def sha256_of(path, chunk=1 << 20):
    """Hex SHA-256 of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def _tmp(path):
    return "%s.%d.part" % (path, os.getpid())


def download(url, path, sha256, timeout=60):
    """Fetch ``url`` to ``path`` unless it is already there; verify ``sha256``.

    Raises ``RuntimeError`` if the downloaded bytes do not match: nothing is
    written to ``path`` in that case.
    """
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = _tmp(path)
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as out:
            while block := r.read(1 << 20):
                out.write(block)
        got = sha256_of(tmp)
        if got != sha256:
            raise RuntimeError("%s: SHA-256 %s, expected %s (download corrupted or the "
                               "upstream file changed)" % (url, got, sha256))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return path


def settings_key(settings):
    """Short, stable hash of a JSON-able settings dict (plus TRAIN_VERSION)."""
    blob = json.dumps(dict(settings, _train_version=TRAIN_VERSION), sort_keys=True,
                      default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def cached_npz(name, settings, build, cache_dir=None):
    """Load ``<name>-<hash of settings>.npz`` or create it with ``build()``.

    ``build()`` returns a dict of arrays.  The settings are written into the
    file as ``_settings`` (a JSON string) and are not part of the returned
    dict.
    """
    cache_dir = cache_dir or DEFAULT_CACHE
    path = os.path.join(cache_dir, "%s-%s.npz" % (name, settings_key(settings)))
    if os.path.exists(path):
        with np.load(path, allow_pickle=False) as d:
            return {k: d[k] for k in d.files if k != "_settings"}
    arrays = build()
    os.makedirs(cache_dir, exist_ok=True)
    tmp = _tmp(path) + ".npz"
    np.savez(tmp, _settings=np.array(json.dumps(settings, sort_keys=True, default=str)),
             **arrays)
    os.replace(tmp, path)
    return arrays

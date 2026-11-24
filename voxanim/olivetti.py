"""The Olivetti faces (AT&T Laboratories Cambridge, 1992-94): 400 grey photos of 64 x 64, ten
of each of 40 people, taken for face-recognition research. The same file scikit-learn's
``fetch_olivetti_faces`` downloads (``olivettifaces.mat``, 4 MB, pinned by SHA-256), read here
without scipy, and scaled exactly as scikit-learn scales it, so ``load()`` is what the
script's ``fetch_olivetti_faces().images`` gives.
"""

from __future__ import annotations

import functools
import os
import struct
import zlib

import numpy as np

from .cache import DEFAULT_CACHE, download

URL = "https://ndownloader.figshare.com/files/5976027"
SHA256 = "b612fb967f2dc77c9c62d3e1266e0c73d5fca46a4b8906c18e454d41af987794"
FILE = "olivettifaces.mat"
PEOPLE, PHOTOS, SIDE = 40, 10, 64


def path(cache_dir=None):
    return os.path.join(cache_dir or DEFAULT_CACHE, FILE)


def cached(cache_dir=None):
    return os.path.exists(path(cache_dir))


def _matrices(raw):
    """``{name: array}`` of a MATLAB 5 file's numeric matrices (column-major, as stored)."""
    endian = ">" if raw[126:128] == b"MI" else "<"
    types = {1: "i1", 2: "u1", 3: "i2", 4: "u2", 5: "i4", 6: "u4", 7: "f4", 9: "f8"}
    out, off = {}, 128
    while off + 8 <= len(raw):
        tag, n = struct.unpack(endian + "II", raw[off:off + 8])
        body = raw[off + 8:off + 8 + n]
        off += 8 + n + ((8 - n % 8) % 8)
        if tag == 15:                                           # compressed
            body = zlib.decompress(body)
            tag, n = struct.unpack(endian + "II", body[:8])
            body = body[8:8 + n]
        if tag != 14:                                           # not a matrix
            continue
        subs, p = [], 0
        while p < len(body):
            st, sn = struct.unpack(endian + "II", body[p:p + 8])
            if st >> 16:                                        # a small element
                sn, st = st >> 16, st & 0xFFFF
                data, p = body[p + 4:p + 4 + sn], p + 8
            else:
                data = body[p + 8:p + 8 + sn]
                p += 8 + sn + ((8 - sn % 8) % 8)
            subs.append((st, data))
        dims = struct.unpack(endian + "i" * (len(subs[1][1]) // 4), subs[1][1])
        name = subs[2][1].decode()
        st, data = subs[3]
        a = np.frombuffer(data, endian + types[st])
        out[name] = a.reshape(dims[::-1]).T                     # MATLAB's column order
    return out


@functools.lru_cache(maxsize=1)
def load(cache_dir=None):
    """``(images [400, 64, 64] float32 in 0..1, person [400])``, as scikit-learn returns
    them: photo ``i`` is person ``i // 10``."""
    p = download(URL, path(cache_dir), SHA256)
    with open(p, "rb") as fh:
        faces = _matrices(fh.read())["faces"]                   # [4096, 400]
    x = np.float32(faces.T.copy())
    x = x - x.min()
    x /= x.max()
    x = x.reshape((PEOPLE * PHOTOS, SIDE, SIDE)).transpose(0, 2, 1)
    return np.ascontiguousarray(x), np.repeat(np.arange(PEOPLE), PHOTOS)

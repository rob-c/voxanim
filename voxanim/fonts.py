"""Fonts for on-screen text, and the box a piece of text covers.

The fonts ship with the package (``voxanim/fonts/``), so every machine draws
identical text:

* ``regular`` -- Inter 4.1 Regular (SIL Open Font License 1.1),
* ``mono`` -- DejaVu Sans Mono 2.37 (Bitstream Vera licence), used for the
  numbers in matrices and tables and for the few glyphs Inter lacks (the
  superscript T of Kᵀ).

Text is laid out with Pillow's BASIC engine (no libraqm shaping), so the result
does not depend on which optional libraries Pillow was built with.  Each file
is checked against its SHA-256 the first time it is loaded: a different font
would change every frame.
"""

from __future__ import annotations

import hashlib
import os

from PIL import ImageFont

from .camera import REF_H, REF_W

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
FONTS = {
    "regular": ("Inter-Regular.ttf",
                "40d692fce188e4471e2b3cba937be967878f631ad3ebbbdcd587687c7ebe0c82"),
    "mono": ("DejaVuSansMono.ttf",
             "b4a6c3e4faab8773f4ff761d56451646409f29abedd68f05d38c2df667d3c582"),
}
# Inter sets about 7% wider than the Helvetica Neue the layouts were designed
# with; drawing it at 94% keeps every line the width it was laid out for (its
# taller x-height keeps it as legible).
SCALE = {"regular": 0.94, "mono": 1.0}
_cache = {}
_checked = set()


class FontNotFoundError(OSError):
    """A bundled font file is missing or is not the expected file."""


def path(kind):
    """The verified path of the ``kind`` font (a key of :data:`FONTS`)."""
    name, sha = FONTS[kind]
    p = os.path.join(DIR, name)
    if kind not in _checked:
        try:
            with open(p, "rb") as fh:
                got = hashlib.sha256(fh.read()).hexdigest()
        except OSError as e:
            raise FontNotFoundError("bundled font %s is missing" % p) from e
        if got != sha:
            raise FontNotFoundError("%s has SHA-256 %s, expected %s" % (p, got, sha))
        _checked.add(kind)
    return p


def font(kind, px):
    """The ``kind`` font for text ``px`` pixels tall (scaled by :data:`SCALE`),
    rounded to whole pixels and cached."""
    key = font_key(kind, px)
    if key not in _cache:
        _cache[key] = ImageFont.truetype(path(kind), key[1],
                                         layout_engine=ImageFont.Layout.BASIC)
    return _cache[key]


def font_key(kind, px):
    """The key :func:`font` files a font under: its kind and whole-pixel size."""
    return (kind, int(round(px * SCALE[kind])))


_MEASURE_MAX = 200_000            # cached measurements kept before the caches restart
_bboxes, _lengths = {}, {}


def bbox(kind, px, s, anchor):
    """Pillow's ink box of ``s`` in the ``kind`` font at ``px`` (relative to the
    anchor), cached: every frame measures the same captions again, and a
    measurement is a pure function of the font, the string and the anchor."""
    key = (*font_key(kind, px), s, anchor)
    hit = _bboxes.get(key)
    if hit is None:
        if len(_bboxes) > _MEASURE_MAX:
            _bboxes.clear()
        hit = _bboxes[key] = font(kind, px).getbbox(s, anchor=anchor)
    return hit


def length(kind, px, s):
    """Pillow's advance width of ``s`` in the ``kind`` font at ``px``, cached
    like :func:`bbox`."""
    key = (*font_key(kind, px), s)
    hit = _lengths.get(key)
    if hit is None:
        if len(_lengths) > _MEASURE_MAX:
            _lengths.clear()
        hit = _lengths[key] = font(kind, px).getlength(s)
    return hit


def text_box(t, W=REF_W, H=REF_H):
    """Ink box ``(x0, y0, x1, y1)`` of a :class:`~voxanim.scene.Text` in the
    pixels of a ``W x H`` frame (the reference resolution by default), as
    Pillow reports it for the text's font, size and anchor."""
    l, tp, r, b = bbox(t.kind, t.size * H, t.s, t.anchor)
    x, y = t.x * W, t.y * H
    return x + l, y + tp, x + r, y + b

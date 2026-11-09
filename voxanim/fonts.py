"""Fonts for on-screen text, and the box a piece of text covers.

Each type role (``light``, ``regular``, ``mono``) lists font files in order of
preference; the first one that loads is used.  Text is laid out with Pillow's
BASIC engine (no libraqm shaping), so the result does not depend on which
optional libraries Pillow was built with.  A missing font is an error: a
silent fallback to Pillow's bitmap default font would change every frame.
"""

from __future__ import annotations

from PIL import ImageFont

from .camera import REF_H, REF_W

# (path, face index within a .ttc collection); bare names are searched for on
# Pillow's font path.
FONTS = {
    "light": [("/System/Library/Fonts/HelveticaNeue.ttc", 7),
              ("/System/Library/Fonts/Avenir Next.ttc", 10),
              ("/usr/share/fonts/truetype/dejavu/DejaVuSans-ExtraLight.ttf", 0),
              ("DejaVuSans.ttf", 0)],
    "regular": [("/System/Library/Fonts/HelveticaNeue.ttc", 0),
                ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
                ("DejaVuSans.ttf", 0)],
    "mono": [("/System/Library/Fonts/Menlo.ttc", 0),
             ("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 0),
             ("DejaVuSansMono.ttf", 0)],
}
_cache = {}


class FontNotFoundError(OSError):
    """None of the font files listed for a type role could be loaded."""


def font(kind, px):
    """The ``kind`` font (a key of :data:`FONTS`) at ``px`` pixels, rounded to
    whole pixels and cached.  Raises :class:`FontNotFoundError` if no listed
    file loads."""
    key = (kind, int(round(px)))
    if key not in _cache:
        for path, index in FONTS[kind]:
            try:
                _cache[key] = ImageFont.truetype(path, key[1], index=index,
                                                 layout_engine=ImageFont.Layout.BASIC)
                break
            except OSError:
                continue
        else:
            raise FontNotFoundError("no %r font found; tried: %s"
                                    % (kind, ", ".join(p for p, _ in FONTS[kind])))
    return _cache[key]


def text_box(t, W=REF_W, H=REF_H):
    """Ink box ``(x0, y0, x1, y1)`` of a :class:`~voxanim.scene.Text` in the
    pixels of a ``W x H`` frame (the reference resolution by default), as
    Pillow reports it for the text's font, size and anchor."""
    l, tp, r, b = font(t.kind, t.size * H).getbbox(t.s, anchor=t.anchor)
    x, y = t.x * W, t.y * H
    return x + l, y + tp, x + r, y + b

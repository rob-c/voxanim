"""The voxanim mark: a V built of voxels, the point of the V lit.

It is drawn in two places from the one description here: as a small mark in its own colours,
with the name and the project's address in light grey, in the bottom-right corner of every
rendered frame
(:func:`stamp`), and as vector files for the website (:func:`svg`, written by
``tools/make_logo.py``).

The corner mark was asked for on every artefact, "on the bottom right with the project url
in light grey".  It sits in the outer margin, below the line the evidence tags use, and is
part of the frame, so videos, GIFs, stills and posters all carry it.  ``VOXANIM_BRAND=0``
leaves it off (a clean plate).
"""

import functools
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

URL = "rob-c.github.io/voxanim"
NAME = "voxanim"
ROWS = ("X...X", "X...X", ".X.X.", ".X.X.", "..X..")        # the V, top row first
TEAL, ORANGE, INK = "#449696", "#e8622f", "#33383d"
DEPTH, GAP = 0.36, 0.07                                     # how much top and side show
FONT = os.path.join(os.path.dirname(__file__), "fonts", "Inter-Regular.ttf")

# On a frame (fractions of the frame height, so it scales with the resolution)
MARK_H = 0.032              # the V
NAME_H = 0.027              # "voxanim"
URL_H = 0.0225              # the address
MARGIN = 0.009              # from the right and bottom edges
GREY = (0.60, 0.60, 0.60)   # light grey on the #dfdfdf backdrop
GREY_DARK = (0.42, 0.42, 0.42)     # and its counterpart on the dark page
OPACITY = 0.85
SUPER = 4                   # the mark is drawn large and averaged down
MIN_HEIGHT = 180            # frames shorter than this are left unmarked


def cubes():
    """The V's cubes as ``(column, row from the bottom, is the lit one)``, in drawing
    order: bottom rows first and right to left, so nothing covers a face it should not."""
    out = [(c, len(ROWS) - 1 - r, r == len(ROWS) - 1)
           for r, row in enumerate(ROWS) for c, ch in enumerate(row) if ch == "X"]
    return sorted(out, key=lambda q: (q[1], -q[0]))


def faces(col, row, unit=1.0):
    """The three visible faces of one cube, as lists of ``(x, y)`` with y downwards and
    the V's bottom-left at the origin: ``(front, top, side)``."""
    x0, x1 = (col + GAP) * unit, (col + 1 - GAP) * unit
    y0, y1 = -(row + GAP) * unit, -(row + 1 - GAP) * unit
    d = DEPTH * unit
    return ([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            [(x0, y1), (x1, y1), (x1 + d, y1 - d), (x0 + d, y1 - d)],
            [(x1, y0), (x1 + d, y0 - d), (x1 + d, y1 - d), (x1, y1)])


def extent(unit=1.0):
    """``(x0, y0, x1, y1)`` of the whole V at ``unit`` pixels per cube."""
    pts = [p for c, r, _ in cubes() for face in faces(c, r, unit) for p in face]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _shade(hexc, f):
    import colorsys
    r, g, b = (int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    r, g, b = colorsys.hls_to_rgb(h, max(0.0, min(1.0, l + f)), s)
    return "#%02x%02x%02x" % (round(r * 255), round(g * 255), round(b * 255))


def svg(kind="lockup", ink=INK, sub="#a0a0a0", url=True):
    """The logo as an SVG document: ``kind`` is ``"mark"`` (the V alone) or ``"lockup"``
    (the V, the name and, with ``url``, the address)."""
    unit = 16.0
    x0, y0, x1, y1 = extent(unit)
    body = ""
    for c, r, lit in cubes():
        base = ORANGE if lit else TEAL
        for face, fill in zip(faces(c, r, unit), (base, _shade(base, 0.14), _shade(base, -0.15))):
            body += '<polygon points="%s" fill="%s"/>' % (
                " ".join("%.2f,%.2f" % p for p in face), fill)
    if kind == "mark":
        pad = 3.0
        return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="%.1f %.1f %.1f %.1f" '
                'role="img" aria-label="voxanim">%s</svg>'
                % (x0 - pad, y0 - pad, x1 - x0 + 2 * pad, y1 - y0 + 2 * pad, body))
    height = 96.0
    s = (height - 8) / (y1 - y0)
    tx = (x1 - x0) * s + 26
    text = ('<text x="%.1f" y="%d" font-family="Inter, system-ui, -apple-system, \'Segoe UI\', '
            'sans-serif" font-size="50" font-weight="600" letter-spacing="-1.6" fill="%s">%s'
            '</text>' % (tx, 60 if url else 66, ink, NAME))
    if url:
        text += ('<text x="%.1f" y="86" font-family="Inter, system-ui, sans-serif" '
                 'font-size="15.5" letter-spacing="0.2" fill="%s">%s</text>' % (tx + 2, sub, URL))
    return ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %.0f %.0f" role="img" '
            'aria-label="voxanim"><g transform="translate(%.2f %.2f) scale(%.4f)">%s</g>%s</svg>'
            % (tx + 212, height, 4 - x0 * s, 4 - y0 * s, s, body, text))


def _rgb(hexc):
    return tuple(int(hexc[i:i + 2], 16) for i in (1, 3, 5))


@functools.lru_cache(maxsize=16)
def _plate(height, width, dark=False):
    """The corner mark for a frame of this size and theme: ``(colour, alpha, y0, x0)``,
    ``colour`` float32 RGB already multiplied by ``alpha`` (float32 coverage), to be placed
    with its top-left at ``(y0, x0)``.  The V is in its own colours, as on the website; the
    name and the address are a quiet grey that reads on the theme's page."""
    k = SUPER
    mark_h = MARK_H * height * k
    ex0, ey0, ex1, ey1 = extent(1.0)
    unit = mark_h / (ey1 - ey0)
    mark_w = (ex1 - ex0) * unit
    name_font = ImageFont.truetype(FONT, max(4, round(NAME_H * height * k)))
    url_font = ImageFont.truetype(FONT, max(4, round(URL_H * height * k)))
    name_w, url_w = name_font.getlength(NAME), url_font.getlength(URL)
    gap_a, gap_b = 0.30 * mark_h, 0.42 * mark_h
    w = int(np.ceil(mark_w + gap_a + name_w + gap_b + url_w)) + 2 * k
    h = int(np.ceil(mark_h * 1.14)) + 2 * k
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    base = h - k - 0.10 * mark_h                 # the V stands on this line
    for c, r, lit in cubes():
        colour = ORANGE if lit else TEAL
        front, top, side = faces(c, r, unit)
        for face, fill in ((front, colour), (top, _shade(colour, 0.14)),
                           (side, _shade(colour, -0.15))):
            draw.polygon([(k + x - ex0 * unit, base + y) for x, y in face],
                         fill=(*_rgb(fill), 255))
    # one baseline for both words, the V's foot
    grey = tuple(round(255 * v) for v in (GREY_DARK if dark else GREY))
    ink, soft = (*grey, round(255 * OPACITY)), (*grey, round(215 * OPACITY))
    x = k + mark_w + gap_a
    draw.text((x, base), NAME, font=name_font, fill=ink, anchor="ls",
              stroke_width=max(1, round(0.012 * mark_h)), stroke_fill=ink)
    draw.text((x + name_w + gap_b, base), URL, font=url_font, fill=soft, anchor="ls")
    # averaged down with colour weighted by coverage, so edges do not pick up black
    big = np.asarray(img, np.float32) / 255.0
    pre = np.dstack([big[..., :3] * big[..., 3:], big[..., 3]])
    hh, ww = (h // k) * k, (w // k) * k
    small = pre[:hh, :ww].reshape(hh // k, k, ww // k, k, 4).mean((1, 3))
    colour, alpha = small[..., :3].copy(), small[..., 3].copy()
    margin = round(MARGIN * height)
    return colour, alpha, height - margin - alpha.shape[0], width - margin - alpha.shape[1]


def box(height, width):
    """Where the mark sits on a frame of this size: ``(x0, y0, x1, y1)`` in pixels."""
    _colour, alpha, y0, x0 = _plate(height, width)
    return x0, y0, x0 + alpha.shape[1], y0 + alpha.shape[0]


def enabled():
    return os.environ.get("VOXANIM_BRAND", "1") != "0"


def stamp(rgb, dark=False):
    """Composite the corner mark onto float ``rgb`` in place (and return it); ``dark``: the
    frame is in dark mode, and the words are a grey that reads on its page."""
    if not enabled():
        return rgb
    height, width = rgb.shape[:2]
    if height < MIN_HEIGHT:                      # a thumbnail: the address would be dust
        return rgb
    colour, alpha, y0, x0 = _plate(height, width, bool(dark))
    if y0 < 0 or x0 < 0:
        return rgb
    region = rgb[y0:y0 + alpha.shape[0], x0:x0 + alpha.shape[1]]
    region *= (1.0 - alpha[..., None]).astype(rgb.dtype)
    region += colour.astype(rgb.dtype)
    return rgb

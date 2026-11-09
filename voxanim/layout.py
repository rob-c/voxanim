"""Frame furniture shared by every animation: titles, subtitles, the step
indicator -- all inside the broadcast **title-safe** area (5% margins,
SMPTE ST 2046-1) -- and the composition zoom that keeps the geometry inside
the **action-safe** area.  ``python3 -m voxanim qa`` checks both.
"""

from __future__ import annotations

import numpy as np

from .camera import REF_H, REF_W
from .fonts import font, text_box

SAFE = 0.05                     # title-safe margin, as a fraction of the frame
ZOOM = 0.90                     # compositions designed edge to edge -> 0.05 .. 0.95

# Type roles.  Sizes are fractions of the picture height; inks are chosen for
# contrast against the ~#dedede backdrop (WCAG 2 ratios in brackets): body
# text >= 4.5:1, titles higher.
INK = (0.20, 0.22, 0.24)            # titles               (~11:1)
INK_SOFT = (0.30, 0.32, 0.34)       # subtitles            (~7:1)
CAPTION_INK = (0.28, 0.30, 0.32)    # captions, labels     (~7.5:1)
NOTE_INK = (0.38, 0.40, 0.42)       # secondary notes      (~5:1)
STRONG_INK = (0.14, 0.16, 0.18)     # the key number (loss, prediction)
TITLE_SIZE, SUB_SIZE = 0.054, 0.033
CAPTION = 0.031                     # default caption size (~17 px at 540p, 133 px at 8K)
MIN_TEXT = 0.029                    # nothing smaller is ever drawn


def text_width(s, size, kind="regular"):
    """Width of ``s`` (``size`` a fraction of the frame height) as a fraction
    of the frame width, measured at the reference resolution."""
    return font(kind, size * REF_H).getlength(s) / REF_W


def wrap(s, size, width):
    """Split ``s`` into as few lines as fit ``width`` (balanced, at spaces)."""
    s = " ".join(s.split())               # no doubled or edge spaces
    if text_width(s, size) <= width:
        return [s]
    words = s.split(" ")
    best = None
    for cut in range(1, len(words)):
        a, b = " ".join(words[:cut]), " ".join(words[cut:])
        w = max(text_width(a, size), text_width(b, size))
        if w <= width and (best is None or w < best[0]):
            best = (w, [a, b])
    return best[1] if best else [s]


def title_block(scene, title=None, subtitle=None, opacity=1.0, right=None):
    """Title and subtitle at the top-left of the title-safe area.

    ``right`` is where something else (the step indicator) starts on the title
    line: the title steps down in size to keep clear of it.  The subtitle
    breaks at explicit newlines first, then wraps to the safe width."""
    if title:
        avail = (right if right is not None else 1 - SAFE) - SAFE - 0.03
        size = TITLE_SIZE
        while text_width(title, size) > avail and size > 0.040:
            size -= 0.002
        # +0.4%: glyphs like 'A' overhang their origin
        scene.text(SAFE + 0.004, SAFE + (TITLE_SIZE - size) * 0.5, title, size, "regular",
                   INK, opacity, "la")
    if subtitle:
        y = SAFE + (0.078 if title else 0.0)
        lines = [ln for part in subtitle.split("\n")
                 for ln in wrap(part, SUB_SIZE, 1 - 2 * SAFE)]
        for n, line in enumerate(lines):
            scene.text(SAFE + 0.001, y + n * 0.046, line, SUB_SIZE, "regular", INK_SOFT,
                       opacity, "la")


def step_indicator(scene, current, phases, size=0.030):
    """'1 forward  2 loss  ...' right-aligned in the title-safe area."""
    items = ["%d %s" % (n + 1, name) for n, name in enumerate(phases)]
    gap = 0.022
    widths = [text_width(s, size) for s in items]
    # -0.004: the last glyph can overhang its advance width by a pixel
    x = (1 - SAFE) - (sum(widths) + gap * (len(items) - 1)) - 0.004
    left = x
    for n, (s, w) in enumerate(zip(items, widths)):
        if n == current:
            col = (0.12, 0.36, 0.66)            # the step under way
        elif n < current:
            col = INK_SOFT                      # done
        else:
            col = (0.56, 0.58, 0.60)            # still to come (de-emphasised)
        scene.text(x, SAFE + 0.012, s, size, "regular", col, 1.0, "la")
        x += w + gap
    return left


# Voxels at least this opaque, and casting a shadow or reflection, count as
# solid for text collisions (particles and ghosts do not).
SOLID_OPACITY = 0.3
DISK_RADIUS = 0.42      # footprint radius per voxel edge (times px per world unit)


def solid_disks(scene, cam):
    """Screen centres and radii (in ``cam``'s px) of the solid voxels -- not
    the particles, which only cross text for an instant."""
    vis = [v for v in scene.voxels
           if v.visible and v.opacity > SOLID_OPACITY and (v.shadow or v.reflect)]
    if not vis:
        return np.zeros((0, 2)), np.zeros(0)
    c = cam.project(np.array([v.center for v in vis]))
    return c, np.array([v.size for v in vis]) * cam.ppu * cam.zoom * DISK_RADIUS


def box_distance(box, pts):
    """Distance from each point ``pts (n, 2)`` to the box ``(x0, y0, x1, y1)``
    (0 inside it)."""
    x0, y0, x1, y1 = box
    dx = np.maximum(np.maximum(x0 - pts[:, 0], pts[:, 0] - x1), 0)
    dy = np.maximum(np.maximum(y0 - pts[:, 1], pts[:, 1] - y1), 0)
    return np.hypot(dx, dy)


def yield_to_geometry(scene, cam, texts):
    """Fade captions that something solid is passing under (a kernel flying
    to the 'done' row, say): smooth, and a pure function of the scene."""
    c, r = solid_disks(scene, cam)
    if not len(c):
        return
    for t in texts:
        box = text_box(t, cam.width, cam.height)
        clear = float(np.min(box_distance(box, c) - r * 0.6))
        k = min(1.0, max(0.0, (clear + 1.0) / 5.0))
        t.opacity *= k
        t.yielded = k

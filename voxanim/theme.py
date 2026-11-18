"""Dark mode for the renders: the same scene with its neutrals turned over.

Asked for as a second copy of every animation that keeps "the colours primary and vibrant
across the whole set", "mostly inverting text, borders and the background".  That is what
this does: a colour with no chroma in it (the backdrop, text, borders, paper, grey) has its
lightness turned over, a vivid colour is left as it is, and pale tints of a colour become
deeper tints of the same hue.

The renderer (:mod:`voxanim.scene`) applies it to what is page furniture - the backdrop, the
lines, the flat panels and their borders, the text - as each is drawn, and never to the
data: the voxels (a digit's grey pixels, a black "off" cell, a white bar over the chosen
class, a glowing neuron), image textures (a digit on paper, a heat map captioned "darker:
larger") and what is drawn on either (the white frame and flash on a block, a curve across
a map) look exactly as in light mode.  (Turning the finished frame instead put out every
white or lit voxel: ``mlp-4-forward-pass`` showed a teal bar where the light copy had a
white one, and a black voxel's shaded faces came out grey.)  No animation's own drawing
code changes, so the light and dark copies cannot drift apart.

``VOXANIM_THEME=dark`` (or ``--theme dark`` on the command line) turns it on.
"""

import os

import numpy as np

# How much colour a pixel needs before it is left alone.  Chroma here is max - min of the
# sRGB values: 0 for a grey, about 0.3 for the input teal, above 0.5 for the filter colours.
# Below CHROMA_LO a pixel is treated as a neutral and fully turned over; above CHROMA_HI it
# is not touched; in between, in proportion.
CHROMA_LO, CHROMA_HI = 0.10, 0.34
# Where turned-over lightness lands.  The light backdrop (PAGE) goes to NIGHT, close to the
# website's dark page; black ink goes to CHALK, a soft white; and what was lighter than the
# backdrop (paper, a white tile) goes below NIGHT, down to PIT.
PAGE, NIGHT, CHALK, PIT = 0.87, 0.125, 0.93, 0.045


def name():
    """``"light"`` or ``"dark"``: the theme frames are rendered in."""
    return "dark" if os.environ.get("VOXANIM_THEME", "light").lower() == "dark" else "light"


def is_dark():
    return name() == "dark"


def suffix():
    """What a dark render's files carry after the animation's name."""
    return "-dark" if is_dark() else ""


def word(light, dark):
    """The word for the theme being rendered, where a caption names a mark by its shade:
    ``"%s: the truth" % theme.word("black", "white")``.  Dark mode turns ink over, so the
    curve a caption calls black is drawn white there (16 animations said "black: ...",
    "dark mark", "pale block" of something the dark copy showed the other way round).
    Call it while the scene is built, not at import: one process renders both themes."""
    return dark if is_dark() else light


def turn(light):
    """A neutral's lightness turned over: two straight pieces that meet at the backdrop."""
    light = np.asarray(light, np.float32)
    below = CHALK + (NIGHT - CHALK) * (light / PAGE)
    above = NIGHT + (PIT - NIGHT) * ((light - PAGE) / (1.0 - PAGE))
    return np.where(light <= PAGE, below, above)


def unturn(light):
    """The lightness that :func:`turn` sends to ``light`` (its inverse)."""
    light = np.asarray(light, np.float32)
    below = PAGE * (light - CHALK) / (NIGHT - CHALK)
    above = PAGE + (1.0 - PAGE) * (light - NIGHT) / (PIT - NIGHT)
    return np.clip(np.where(light >= NIGHT, below, above), 0.0, 1.0)


def _neutral(chroma):
    w = np.clip((CHROMA_HI - chroma) / (CHROMA_HI - CHROMA_LO), 0.0, 1.0)
    return w * w * (3.0 - 2.0 * w)                       # smooth at both ends


def _relight(rgb, target, light, chroma):
    """``rgb`` moved to lightness ``target`` with hue and chroma kept; a tint pushed past
    black or white is squeezed towards its own lightness, not clipped."""
    out = rgb + (target - light)[..., None]
    over = np.maximum(out.max(-1) - 1.0, 0.0)[..., None]
    under = np.maximum(-out.min(-1), 0.0)[..., None]
    squeeze = np.clip(1.0 - (over + under) / np.maximum(chroma, 1e-6)[..., None], 0.0, 1.0)
    mid = target[..., None]
    return np.clip(mid + (out - mid) * squeeze, 0.0, 1.0)


def to_dark(rgb):
    """``rgb`` (float, display-referred, 0..1) in dark mode; a new array."""
    rgb = np.asarray(rgb, np.float32)
    hi, lo = rgb.max(-1), rgb.min(-1)
    chroma, light = hi - lo, 0.5 * (hi + lo)
    target = light + (turn(light) - light) * _neutral(chroma)
    return _relight(rgb, target, light, chroma)


def keep(rgb):
    """Colours that come out of :func:`to_dark` looking as they went in here: for a flat
    fill that is data and must not be turned over with the page (a grey level that stands
    for a number).  Only near-neutral colours can be kept this way, and only they are
    changed; the renderer does not need it for voxels or images, which it never turns."""
    rgb = np.asarray(rgb, np.float32)
    hi, lo = rgb.max(-1), rgb.min(-1)
    chroma, light = hi - lo, 0.5 * (hi + lo)
    w = _neutral(chroma)
    w = np.where(w > 0.5, w, 0.0)                        # a colour is left to to_dark as it is
    target = light + (unturn(light) - light) * w
    return _relight(rgb, target, light, chroma)


# Thin ink (text, lines) in a dark saturated colour is "vivid", so it is not turned, but a
# navy curve or a bottle-green label that read at 6:1 on the light page fell to 2:1 on the
# dark one.  Ink is lifted until it contrasts with the dark page as it did with the light
# one, and no further than INK_CONTRAST (WCAG's level for body text).
INK_CONTRAST = 4.5
# Where black and white ink tie on a surface of this luminance, by WCAG's ratio (4.6:1
# each).  The ratio flatters dark pairs: on a dulled mid-tone - gradient-descent's
# half-lit neurons, a pale box turned deep - black at 4.6:1 was hard to read where white at
# 4:1 is not, so on a surface that is not vivid the tie is moved up to MURK (white still
# has 3.5:1 there).  A vivid box (a red badge) looks as it does in light mode and keeps
# the ink it has there.
TIE, MURK = 0.179, 0.25


def luminance(rgb):
    """WCAG relative luminance of sRGB ``rgb``."""
    c = np.asarray(rgb, np.float32)
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return lin @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def contrast(a, b):
    """WCAG contrast ratio of two luminances (1 to 21)."""
    hi, lo = np.maximum(a, b), np.minimum(a, b)
    return (hi + 0.05) / (lo + 0.05)


def _wanted(rgb):
    """The contrast ink ``rgb`` had with the light page, up to INK_CONTRAST; none is asked
    of ink lighter than that page."""
    lum, page = luminance(rgb), luminance(np.full(3, PAGE, np.float32))
    return np.where(lum < page, np.minimum(contrast(lum, page), INK_CONTRAST), 1.0)


def _reach(rgb, back, want, tie=TIE, steps=24):
    """``rgb`` moved towards white (on a dark ``back``, a luminance) or black (on a light
    one), hue kept, by the least that gives contrast ``want`` with it; ``tie``: the
    luminance of ``back`` below which it counts as dark."""
    hi, lo = rgb.max(-1), rgb.min(-1)
    chroma, light = hi - lo, 0.5 * (hi + lo)
    up = back < tie                                      # where white and black tie
    end = np.where(up, 1.0, 0.0).astype(np.float32)
    best = _relight(rgb, end, light, chroma)
    for k in range(steps - 1, -1, -1):                   # from the far end back to rgb
        cand = _relight(rgb, light + (end - light) * (k / steps), light, chroma)
        lum = luminance(cand)
        ok = (contrast(lum, back) >= want) & ((lum >= back) == up)
        best = np.where(ok[..., None], cand, best)
    return best


def ink(rgb, under=None):
    """``rgb``, the colour of text or a line, in dark mode.

    On the page (``under`` None): :func:`to_dark`, then lightened (hue kept) while it
    contrasts less with the dark page than it did with the light one.  On a surface of
    colour ``under`` (a label on a box, as that box is now drawn): the ink it was given,
    which was chosen to read on that surface, unless the turned ink reads better there -
    a pale green box is a deep green one in dark mode, and its dark green label sank into
    it (1,674 of 84,945 labels across the set fell under 3:1) - and then moved away from
    the surface until it reads.  Fills are not given this:
    a filled area shows its colour at any lightness, a hairline does not."""
    rgb = np.asarray(rgb, np.float32)
    out, want, tie = to_dark(rgb), _wanted(rgb), TIE
    if under is None:
        back = luminance(np.full(3, NIGHT, np.float32))
    else:
        under = np.clip(np.asarray(under, np.float32), 0.0, 1.0)
        back = luminance(under)
        as_given, turned = contrast(luminance(rgb), back), contrast(luminance(out), back)
        # dark ink whose turned colour reads on a dark surface stays turned, so that a
        # caption written faint is as faint on a deep tint as it is on the page
        stay = (back < TIE) & (luminance(rgb) < luminance(np.full(3, PAGE))) & (turned >= want)
        # ... and dark ink on a dulled mid-tone gives way to light ink (see MURK)
        tie = np.where(under.max(-1) - under.min(-1) < CHROMA_HI, MURK, TIE)
        murky = (back >= TIE) & (back < tie) & (luminance(rgb) < back)
        out = np.where(((as_given >= turned) & ~stay & ~murky)[..., None], rgb, out)
    short = contrast(luminance(out), back) < want
    if not np.any(short):
        return out
    return np.where(short[..., None], _reach(out, back, want, tie), out)


def apply(rgb):
    """The finished frame in the current theme (the same array when light)."""
    return to_dark(rgb) if is_dark() else rgb

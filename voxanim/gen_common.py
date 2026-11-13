"""Drawing helpers shared by the generalisation animations (all flat
:class:`~voxanim.diagram.Diagram` s): digits as small images, a stack of
read-out lines, bars, and the inks the family uses for train / test curves.

Everything here is drawing only; the numbers come from
:mod:`voxanim.generalise` and :mod:`voxanim.deepnet`.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from .diagram import draw_image

TRAIN_INK = tuple(float(v) for v in C.INPUT)        # training curves: teal
TEST_INK = (0.75, 0.22, 0.22)                       # test curves: red
VAL_INK = (0.40, 0.42, 0.46)                        # validation: grey
WRONG_INK = (0.80, 0.25, 0.25)                      # a wrong label, a mistake
DIGIT_INK = (0.16, 0.17, 0.19)                      # the ink of a drawn digit
PAPER = (0.93, 0.93, 0.91)                          # the paper behind it
SIDE = 28


def digit_texture(img, ink=DIGIT_INK, paper=PAPER):
    """A ``(784,)`` image in [0, 1] as a ``(28, 28, 3)`` texture: dark ink on
    pale paper (``ink`` tints the strokes)."""
    v = np.clip(np.asarray(img, np.float64).reshape(SIDE, SIDE), 0.0, 1.0)[..., None]
    return (np.asarray(paper)[None, None] * (1 - v) + np.asarray(ink)[None, None] * v
            ).astype(np.float32)


def draw_digit(anim, sc, c, img, px, op, ink=DIGIT_INK, paper=PAPER, lift=0.1):
    """The digit ``img`` drawn ``px`` wide at world point ``c``."""
    draw_image(anim, sc, c, digit_texture(img, ink, paper), px, px, op, grid=0.0, lift=lift)


def readout(anim, sc, fx, fy, lines, op, step=0.05, size=0.03, anchor="lm"):
    """Lines of ``(text, ink)`` (ink None: the caption ink) stacked from layout
    point ``(fx, fy)``, ``step`` apart (layout fractions)."""
    for n, (text, ink) in enumerate(lines):
        anim.label(sc, anim.P(fx, fy + n * step), text, size, anchor=anchor, ink=ink, opacity=op)


def bar(anim, sc, c_bottom, w, h, fill, op, lift=0.4):
    """A filled bar ``w`` px wide rising ``h`` px from world point ``c_bottom``."""
    if h <= 0.5 or op <= 0.01:
        return
    anim.rect(sc, anim.shift(c_bottom, 0, -h / 2), w, h, fill, op, radius=1.5, lift=lift)


def pct(v):
    """``0.9402`` as ``94.0%``."""
    return "%.1f%%" % (100 * v)


def strong(text):
    """A read-out line in the strong ink."""
    return (text, layout.STRONG_INK)


def note(text):
    """A read-out line in the note ink."""
    return (text, layout.NOTE_INK)

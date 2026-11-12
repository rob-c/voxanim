"""A next-token distribution as a row of horizontal bars, for the behaviour
animations (:mod:`voxanim.halluc_anim`, :mod:`voxanim.condition_anim`,
:mod:`voxanim.calib_anim`).

Each row is one token: its display string on the left, a bar whose length is
its probability (the full length :data:`BAR_PX` is p = 1, so bars of two
distributions drawn side by side are comparable) and the number at the bar's
end.  Everything is measured in reference pixels on a :class:`~voxanim.diagram.Diagram`.
"""

from __future__ import annotations

from . import color as C
from . import layout
from .text import num

BAR_PX = 170.0                  # length of a bar for p = 1 (reference px)
BAR_H = 11.0                    # bar thickness (px)
LABEL_PX = 100.0                # room for the token string, left of the bars (px)
ROW_PX = 24.0                   # vertical pitch of the rows (px)
LABEL_IN = 0.25                 # the token strings fade in over this much of ``grow``
BAR = C.rgb("#1f9e94")          # a bar (teal, as data flowing forward elsewhere)
BAR_DIM = C.rgb("#9fb4b2")      # a bar that is not the one being talked about
MARK = C.rgb("#e0962a")         # the watched token, when there is one


def shown(s):
    """A token string as it is displayed: without the leading space most GPT-2
    tokens carry, and with a newline spelt out as ⏎."""
    return s.strip() + ("⏎" if s.endswith("\n") else "")


def draw_bars(anim, sc, top_left, items, grow=1.0, opacity=1.0, mark=None, size=layout.MIN_TEXT,
              dim=False):
    """Rows of bars for ``items`` ``[(token string, probability)]`` under the world
    point ``top_left`` (the first row's left end, at its vertical centre).

    ``grow`` in [0, 1] draws the bars at that fraction of their length (the
    numbers appear in the last 40%), ``mark`` highlights one token string,
    ``dim`` greys the bars that are not the marked one.
    """
    if opacity <= 0.01 or grow <= 0:
        return
    for n, (tok, p) in enumerate(items):
        row = anim.shift(top_left, 0.0, n * ROW_PX)
        marked = mark is not None and tok == mark
        ink = layout.STRONG_INK if marked else layout.CAPTION_INK
        anim.label(sc, row, shown(tok), size, anchor="lm", ink=ink,
                   opacity=opacity * min(1.0, grow / LABEL_IN), kind="mono")
        length = max(BAR_PX * p * grow, 0.0)
        if length > 0.3:
            c = anim.shift(row, LABEL_PX + length / 2, 0.0)
            fill = MARK if marked else (BAR_DIM if dim else BAR)
            anim.rect(sc, c, length, BAR_H, fill, opacity, radius=2.0, lift=0.4)
        if grow > 0.6:
            anim.label(sc, anim.shift(row, LABEL_PX + length + 6, 0.0), num(p), size,
                       anchor="lm", ink=ink, opacity=opacity * (grow - 0.6) / 0.4, kind="mono")


def bars_height(n):
    """Height in px of ``n`` rows of bars."""
    return n * ROW_PX

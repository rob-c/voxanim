"""Typography for on-screen numbers and symbols.

Labels use proper typographic characters, not ASCII stand-ins: a real minus
sign (U+2212, the width of a plus, so columns of numbers line up), the
multiplication sign and subscript digits.  Every character used here is
covered by both bundled fonts (see :mod:`voxanim.fonts`).
"""

from __future__ import annotations

TIMES = "×"
MINUS = "−"
_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def num(value, digits=2):
    """``value`` with ``digits`` decimals and a typographic minus: ``−0.25``."""
    s = "%.*f" % (digits, value)
    return MINUS + s[1:] if s.startswith("-") else s


def sub(n):
    """Subscript digits for ``n``: ``sub(12) == "₁₂"``."""
    return str(n).translate(_SUBSCRIPTS)

"""The GPT-2 animation's camera path: keyframed log-space zooms with a moving focus.

The camera itself never moves.  Each frame the whole world is scaled about a
*focus* point (see ``GPT2Animation.Z``), so a shot is just ``(focus, span)``:
the model-space point at the centre of attention and how many model units fit
across 86% of the frame width.
"""

from __future__ import annotations

import itertools
import math

from .easing import ease_in_out, remap


class CameraPath:
    """Keyframes ``(time s, focus, span)``, eased between in log space.

    ``focus`` is a (3,) model-space point or a callable ``t -> (3,)`` (a
    tracking shot); ``span`` is in model units.  Times must not decrease.
    """

    def __init__(self, keys):
        self.keys = list(keys)

    def __call__(self, t):
        """``(focus, span)`` at time ``t`` (s); clamped to the first / last key."""
        keys = self.keys
        if t <= keys[0][0]:
            return keys[0][1], keys[0][2]
        for (t0, f0, s0), (t1, f1, s1) in itertools.pairwise(keys):
            if t0 <= t <= t1:
                f0 = f0(t) if callable(f0) else f0
                f1 = f1(t) if callable(f1) else f1
                u = ease_in_out(remap(t, t0, t1))
                span = math.exp(math.log(s0) + (math.log(s1) - math.log(s0)) * u)
                # the focus moves in step with the view size, so the target
                # stays in frame while zooming (a "zoom-pan")
                w = (span - s0) / (s1 - s0) if abs(s1 - s0) > 1.0 else u
                return f0 + (f1 - f0) * w, span
        return keys[-1][1], keys[-1][2]

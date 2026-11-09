"""Text and layout: nothing clipped, nothing colliding, all inside the safe areas."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from conftest import DRAFT, available

from voxanim import presets
from voxanim.scene import Renderer


@pytest.mark.slow
def test_layout_qa_clean():
    """No clipped or colliding text, everything inside the broadcast safe areas."""
    from voxanim.qa import check
    names = [n for n in presets.REGISTRY if available(n)]
    bad = [n for n in names if check(n, samples=24, verbose=False)]
    assert not bad, bad


def test_text_is_not_clipped():
    """Text composited in its own little box must match Pillow drawing the
    same text on the whole frame (a tight box once cut the tops off 7s)."""
    from PIL import Image as Im
    from PIL import ImageDraw as Dr

    from voxanim.scene import Text, font
    anim = presets.get("conv2d").build()
    r = Renderer(anim.cam, replace(DRAFT))
    bg = np.full((540, 960, 3), 0.87, np.float32)
    for size in (0.029, 0.033, 0.054):
        for anchor in ("la", "mt", "rm", "ls"):
            t = Text(0.3123, 0.4337, "784 fgAjy 10 (p) 7", size, "regular", (0, 0, 0), 1.0,
                     anchor)
            # the halo only touches the backdrop; compare ink coverage
            ours = r._draw_texts(bg.copy(), [t]).mean(-1)
            ref = Im.new("L", (960, 540), 0)
            Dr.Draw(ref).text((t.x * 960, t.y * 540), t.s, font=font("regular", size * 540),
                              fill=255, anchor=anchor)
            ink_ref = np.asarray(ref) > 128
            assert (ours[ink_ref] < 0.45).all(), (size, anchor)
            rows = np.nonzero(ink_ref.any(1))[0]
            got = np.nonzero((ours < 0.45).any(1))[0]
            assert rows.min() == got.min() and rows.max() == got.max(), (size, anchor)

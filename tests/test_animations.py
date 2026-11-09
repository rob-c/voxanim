"""Every animation, as a whole: it builds, and it loops without a visible seam."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import CONV_PRESETS, DRAFT, data_marks

from voxanim import presets
from voxanim.scene import Renderer


@pytest.mark.parametrize("name", [pytest.param(n, marks=data_marks(n))
                                  for n in presets.REGISTRY])
def test_loops_seamlessly(name):
    """The last frame flows into the first: the loop reset lands on frame 0.
    (Convolutions are held to a tighter tolerance: nothing in them fades.)"""
    anim = presets.build(name)
    r = Renderer(anim.cam, DRAFT)
    a = r.render(anim.scene(0.0))
    b = r.render(anim.scene(anim.duration))
    tol = (2.0 if name in CONV_PRESETS else 3.0) / 255
    assert np.abs(a - b).max() < tol

"""voxanim.study_resample: the resampling arithmetic and the measured claims."""

from __future__ import annotations

import numpy as np
from conftest import needs_faces

from voxanim import study_resample as R


def test_the_arithmetic():
    k = R.kernel_steps()
    assert k["total"] == 1 and k["kernel"][1, 1] == 0.25 and k["kernel"][0, 0] == 1 / 16
    D = R.down_matrix()
    assert D.shape == (32, 64) and np.allclose(D[1:].sum(1), 1) and np.isclose(D[0].sum(), 0.75)
    x = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    small, skip = R.down(x)
    assert small.shape == skip.shape == (32, 32) and np.array_equal(skip, x[::2, ::2])
    assert np.isclose(small[5, 7], (R.KERNEL * x[9:12, 13:16]).sum(), atol=1e-6)
    got = R.received()
    assert got[0, 0] == 1 and got[-1, 5] == 0.5 and got[-1, -1] == 0.25
    flat = np.full((32, 32), 0.7, np.float32)
    assert np.allclose(R.up_even(flat), 0.7)
    big, blocks = R.up(small)
    assert np.allclose(big[2:-2:2, 2:-2:2], small[1:-1, 1:-1], atol=1e-6)   # kept pixels
    assert np.allclose(blocks[::2, ::2], small)


def test_aliasing_on_the_zone_plate():
    a = R.aliasing()
    assert a["fine_pixels"] > 500
    assert a["skip"] > 3 * a["blur"] and a["blur"] < a["average"] < a["skip"]


@needs_faces
def test_round_trips_on_the_photos():
    r = R.round_trips()
    assert r["blur_bilinear"]["inner"] < 0.5 * r["blur_nearest"]["inner"]
    assert r["blur_even"]["full"] < r["blur_bilinear"]["full"]      # the edge, made even
    assert np.isclose(r["blur_even"]["inner"], r["blur_bilinear"]["inner"], rtol=1e-6)
    assert r["skip_bilinear"]["inner"] < r["blur_bilinear"]["inner"]  # said on screen

"""examples/resample.py's own lines in PyTorch give voxanim.study_resample's pictures: the blur
and skip, the matrix, the transposed convolution and its edge, to float32 rounding."""

from __future__ import annotations

import runpy

import numpy as np
import pytest

from voxanim import blocks
from voxanim import study_resample as R

torch = pytest.importorskip("torch", reason="torch is not installed: the resampling script is "
                                            "checked with it")


@pytest.fixture(scope="module")
def ns():
    return runpy.run_path(str(blocks.RESAMPLE), run_name="test")


def test_the_script_and_the_twin_agree(ns):
    plate = R.zone_plate()
    assert np.allclose(ns["zone_plate"]().numpy(), plate, atol=1e-6)
    small, skip = ns["down"](torch.tensor(plate))
    mine, myskip = R.down(plate)
    np.testing.assert_allclose(small.numpy(), mine, atol=1e-6)
    assert np.array_equal(skip.numpy(), myskip)
    np.testing.assert_allclose(ns["down_matrix"]().numpy(), R.down_matrix())
    run = dict(ns, plate=torch.tensor(plate), small=small, D=ns["down_matrix"]())
    exec("\n".join(blocks.code_lines("rs-same")), run)
    assert float((run["by_matrix"] - small).abs().max()) < 1e-6
    big, blocks_ = ns["up"](small)
    b2, k2 = R.up(mine)
    np.testing.assert_allclose(big.numpy(), b2, atol=1e-6)
    np.testing.assert_allclose(blocks_.numpy(), k2, atol=1e-6)
    np.testing.assert_allclose(ns["up_even"](small).numpy(), R.up_even(mine), atol=1e-6)

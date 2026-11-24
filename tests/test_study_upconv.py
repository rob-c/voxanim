"""voxanim.study_upconv: a kernel bigger than the picture, with padding, as a matrix."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_faces

from voxanim import study_upconv as U


def test_the_kernel_and_the_sizes():
    k = U.kernel()
    assert k.shape == (9, 9) and np.isclose(k.sum(), 1) and np.allclose(k, k.T)
    assert U.OUT == 6 + 2 * 8 - 9 + 1 == 14
    x = np.random.default_rng(0).random((6, 6))
    M = U.matrix()
    assert M.shape == (196, 36)
    assert np.allclose(M @ x.ravel(), U.conv(x, k, 8).ravel())
    assert np.allclose(M.sum(0), 1)                       # each pixel's value, spread, kept
    assert np.allclose(M, U.down_matrix().T)              # the shrinking matrix, on its side
    col = M[:, 0].reshape(14, 14)                         # pixel (0, 0)'s stamp: the kernel
    assert np.allclose(col[:9, :9], k[::-1, ::-1]) and np.allclose(col[9:, :], 0)


@needs_faces
def test_the_picture():
    f = U.facts()
    assert f["same"] < 1e-12 and f["transpose"] == 0
    assert np.isclose(f["total_in"], f["total_out"])
    assert f["nonzero_per_column"] == 81 and f["shape"] == (196, 36)
    assert U.picture().shape == (6, 6)


def test_against_torch():
    torch = pytest.importorskip("torch")
    x = np.random.default_rng(1).random((6, 6))
    k = U.kernel()
    t = torch.nn.functional.conv2d(torch.tensor(x)[None, None], torch.tensor(k)[None, None],
                                   padding=8)[0, 0].numpy()
    assert t.shape == (14, 14) and np.allclose(t, U.conv(x, k, 8))

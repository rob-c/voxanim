"""A kernel bigger than the picture, with padding: conv2d making a picture larger, as a matrix,
for the animation "Bigger by spreading: a kernel larger than the picture" (catalogue 271).

The input is tiny and real: photo 0 of the Olivetti faces (:mod:`voxanim.olivetti`), shrunk to
6 x 6 by averaging (``F.interpolate(mode="area")``, :func:`voxanim.study_faces.resize`). The
kernel is 9 x 9, the binomial row [1, 8, 28, 56, 70, 56, 28, 8, 1] / 256 times itself: a smooth
bell that adds up to 1, wider than the picture. With padding 8, conv2d gives
6 + 2 x 8 - 9 + 1 = 14 pixels a side: the picture grows.

As a matrix (:func:`matrix`): flatten the 6 x 6 picture to 36 numbers and the 14 x 14 output to
196; conv2d is ``out = M @ x`` with M of 196 x 36. Column j of M is the kernel's 9 x 9 stamp
placed where input pixel j sits: each input pixel spreads into a smooth blob, and the output is
the blobs added up. Each column adds up to 1, so the picture's total is kept, only spread.
M is exactly the transpose of the matrix V of the same kernel without padding, which turns a
14 x 14 picture into 6 x 6 (each output pixel an average of 81): shrinking averages many pixels
into one, this spreads one into many. No other smoothing step is used.
"""

from __future__ import annotations

import functools

import numpy as np

from . import olivetti
from . import study_faces as F

SMALL = 6
ROW = np.array([1, 8, 28, 56, 70, 56, 28, 8, 1], np.float64) / 256
K = len(ROW)
PAD = K - 1
OUT = SMALL + 2 * PAD - K + 1                     # 14
PHOTO = 0


def kernel():
    """The 9 x 9 bell: the binomial row times itself (adds up to 1)."""
    return np.outer(ROW, ROW)


@functools.lru_cache(maxsize=1)
def picture():
    """Photo 0 of the Olivetti faces shrunk to 6 x 6 by averaging (float64)."""
    return F.resize(olivetti.load()[0][PHOTO], SMALL).astype(np.float64)


def conv(x, k, pad):
    """``F.conv2d`` (cross-correlation; the bell is symmetric, so it is also a convolution)
    of a square picture with a square kernel, stride 1, zero padding ``pad``."""
    x = np.pad(x, pad)
    n = len(x) - len(k) + 1
    out = np.zeros((n, n))
    for r in range(n):
        for c in range(n):
            out[r, c] = (x[r:r + len(k), c:c + len(k)] * k).sum()
    return out


def matrix(n_in=SMALL, pad=PAD, k=None):
    """The conv2d as a matrix: column j is the output of a picture with a single 1 at pixel j
    (rows and columns flattened row by row)."""
    k = kernel() if k is None else k
    cols = []
    for j in range(n_in * n_in):
        e = np.zeros(n_in * n_in)
        e[j] = 1
        cols.append(conv(e.reshape(n_in, n_in), k, pad).ravel())
    return np.stack(cols, 1)


def up():
    """``dict(x 6 x 6, M 196 x 36, out 14 x 14, by_matrix 14 x 14)``: conv2d and M @ x."""
    x = picture()
    M = matrix()
    return dict(x=x, M=M, out=conv(x, kernel(), PAD), by_matrix=(M @ x.ravel()).reshape(OUT, OUT))


def down_matrix():
    """The same kernel without padding, from 14 x 14 to 6 x 6: V of 36 x 196."""
    return matrix(OUT, 0)


def facts():
    """The exact identities the animation states."""
    u = up()
    M, V = u["M"], down_matrix()
    k = kernel()
    return dict(kernel_sum=float(k.sum()), kernel_max=float(k.max()), out_side=OUT,
                shape=M.shape, column_sums=M.sum(0), row_sums=M.sum(1),
                same=float(np.abs(u["out"] - u["by_matrix"]).max()),
                transpose=float(np.abs(M - V.T).max()),
                total_in=float(u["x"].sum()), total_out=float(u["out"].sum()),
                nonzero_per_column=int((M[:, 0] > 0).sum()),
                nonzero_per_row_middle=int((M[(OUT // 2) * OUT + OUT // 2] > 0).sum()))

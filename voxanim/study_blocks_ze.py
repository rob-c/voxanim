"""The numbers behind animation 271 of "Blocks and code" (group ``ze``, "Resizing pictures"):
a conv2d whose kernel is bigger than the picture, with padding, makes the picture bigger.
Everything comes from :mod:`voxanim.study_upconv` (checked against torch by
``tests/test_study_upconv.py``); this module picks what is shown and checks each claim the
drawing makes with a second, plain computation.

``resample-big-kernel`` ("Bigger: a kernel larger than the picture"): photo 0 of the
Olivetti faces shrunk to 6 x 6 (a real photo, pixels 0 to 1) and a 9 x 9 bell (the binomial
row ``[1, 8, 28, 56, 70, 56, 28, 8, 1] / 256`` times itself, adding up to 1; its middle row
0.001 0.009 0.030 0.060 0.075 0.060 0.030 0.009 0.001). The kernel cannot fit inside the
picture. With padding 8 the picture sits in a 22 x 22 frame of zeros and the kernel fits in
6 + 2 x 8 - 9 + 1 = 14 places a side: 14 x 14 = 196 output pixels. The middle output pixel
(7, 7) has all 36 pixels under the kernel (its other 45 numbers land on zeros): times the
kernel's numbers and added up, 0.61.

As a matrix: x flattened to 36 numbers, the output to 196; conv2d is ``out = M @ x`` with M
of 196 x 36, exactly (0 apart). Column j of M, reshaped to 14 x 14, is the kernel stamped
where pixel j sits (81 numbers that are not 0); the column shown is the brightest pixel's
(:func:`brightest`: pixel (4, 1), column 25, 0.76). The output is the 36 stamps added up,
each times its pixel (:func:`stamps_added`, checked against conv2d). V, the same kernel
without padding, turns 14 x 14 into 6 x 6: each of its 36 rows adds up to 1, a weighted
average of 81; M is exactly V turned on its side (``M == V.T``, 0 apart). Each column of M
adds up to 1, so the total is kept: 22.99 in, 22.99 out. The 196 new pixels hold no new
information: they are the 36 numbers, spread.
"""

from __future__ import annotations

import numpy as np

from . import study_upconv as U
from .study import Panel, Study

ON = "Olivetti faces"
PHOTO = U.PHOTO                    # photo 0 (person 0), named by number only
SOURCE = ("voxanim.study_upconv: F.conv2d with a 9 x 9 bell kernel (the binomial row "
          "[1, 8, 28, 56, 70, 56, 28, 8, 1] / 256 times itself) and padding 8 on photo 0 of "
          "the Olivetti faces (AT&T Laboratories Cambridge, 1992-94) shrunk to 6 x 6 by "
          "averaging, pixels 0 to 1; float64, checked against torch")
WORKED = (U.OUT // 2, U.OUT // 2)  # the output pixel worked out: the middle one, (7, 7)


def num(v, places=2):
    return "%.*f" % (places, float(v))


def brightest(x):
    """The pixel whose column is shown: the brightest of the picture (the first, if tied)."""
    i, j = np.unravel_index(int(np.argmax(x)), x.shape)
    return int(i), int(j)


def stamps_added(x, k=None, pad=U.PAD):
    """The output built the other way round: each input pixel's value times the kernel,
    stamped (flipped, as conv2d's cross-correlation places it) at that pixel's place, all
    added up."""
    k = U.kernel() if k is None else k
    n, kk = len(x), len(k)
    side = n + 2 * pad - kk + 1
    out = np.zeros((side, side))
    for i in range(n):
        for j in range(n):
            r, c = i + pad - kk + 1, j + pad - kk + 1      # the stamp's top-left output pixel
            out[r:r + kk, c:c + kk] += x[i, j] * k[::-1, ::-1]
    return out


def big_kernel_study():
    k = U.kernel()
    u = U.up()
    x, M, out = u["x"], u["M"], u["out"]
    V = U.down_matrix()
    f = U.facts()
    n, side, pad = U.SMALL, U.OUT, U.PAD
    if side != n + 2 * pad - len(k) + 1 or side != 14:
        raise ValueError("the output is not 14 x 14")
    if len(k) <= n:
        raise ValueError("the kernel is not bigger than the picture")
    if f["same"] != 0.0 or f["transpose"] != 0.0:
        raise ValueError("M @ x is not conv2d, or M is not V turned on its side")
    if not np.allclose(stamps_added(x), out, atol=1e-15):
        raise ValueError("the stamps added up are not conv2d")
    if not (np.allclose(M.sum(0), 1) and np.allclose(V.sum(1), 1)):
        raise ValueError("M's columns (V's rows) do not add up to 1")
    padded = np.pad(x, pad)
    wi, wj = WORKED
    window = padded[wi:wi + len(k), wj:wj + len(k)]
    under = int((window != 0).sum())
    worked = float((window * k).sum())
    if under != 36 or abs(worked - out[wi, wj]) > 1e-15 or \
            abs(worked - (x * k[1:7, 1:7]).sum()) > 1e-15:
        raise ValueError("the middle output pixel is not all 36 pixels times the kernel")
    at = brightest(x)
    col = at[0] * n + at[1]
    stamp = M[:, col].reshape(side, side)
    nz = np.argwhere(stamp > 0)
    r0, c0 = (int(v) for v in nz.min(0))
    if (stamp > 0).sum() != 81 or (r0, c0) != at or not np.allclose(
            stamp[r0:r0 + 9, c0:c0 + 9], k[::-1, ::-1]):
        raise ValueError("the column is not the kernel stamped at its pixel")
    if not np.array_equal(V[col].reshape(side, side), stamp):
        raise ValueError("V's row is not M's column")
    total_in, total_out = float(x.sum()), float(out.sum())
    meta = dict(
        source=SOURCE, photo_index=PHOTO, person=PHOTO // 10, small=n, kernel_side=len(k),
        pad=pad, side=side, padded_side=n + 2 * pad, outputs=side * side,
        x=x, x_text=[[num(v) for v in r] for r in x], padded=padded,
        row=U.ROW.tolist(), row256=[1, 8, 28, 56, 70, 56, 28, 8, 1],
        kernel=k, kernel_sum=float(k.sum()), kernel_max=float(k.max()),
        middle_row=k[len(k) // 2].tolist(),
        middle_row_text=[num(v, 3) for v in k[len(k) // 2]],
        equation="%d + 2 × %d − %d + 1 = %d" % (n, pad, len(k), side),
        out=out, out_max=float(out.max()), out_max_text=num(out.max()),
        worked_at=list(WORKED), worked=worked, worked_text=num(worked), under=under,
        on_zeros=len(k) ** 2 - under,
        M=M, M_shape=list(M.shape), V=V, V_shape=list(V.shape),
        by_matrix=u["by_matrix"], same=f["same"], transpose=f["transpose"],
        column=col, column_at=list(at), column_value=float(x[at]),
        column_value_text=num(x[at]), stamp=stamp, stamp_corner=[r0, c0],
        per_column=f["nonzero_per_column"], middle_reads=f["nonzero_per_row_middle"],
        column_sums=M.sum(0), V_row_sums=V.sum(1),
        total_in=total_in, total_out=total_out,
        total_in_text=num(total_in), total_out_text=num(total_out),
        limitations=[
            "the kernel is fixed, not learned: one smooth bell, wider than the picture; a "
            "learned upsampling layer would pick its own numbers",
            "with stride 1 the output's pixels are as far apart as the input's: the picture "
            "grows by 4 pixels past each edge, and the smoothing does the rest; it is not "
            "the 2x upsampling of 264 with stride 2",
            "the output is shaded on its own scale (0 to 0.62) in the animation; on the "
            "input's scale it is darker, since the same total is spread over 196 pixels",
            "the 196 output numbers are made of the 36 input numbers: they add smooth "
            "detail, not information"])
    panels = (
        Panel("The kernel", "9 x 9: the binomial row / 256 times itself, adding up to 1, "
              "wider than the 6 x 6 picture", "matrix", dict(values=k)),
        Panel("Photo 0, bigger", "the 6 x 6 picture (shown where it sits, inside 4 pixels of "
              "black) and conv2d's 14 x 14 output with padding 8", "images",
              dict(values=[np.pad(x, (side - n) // 2), out],
                   labels=["6 x 6, where it sits", "14 x 14 output"])),
        Panel("The matrix M", "196 x 36: column j is the kernel stamped where pixel j sits; "
              "exactly the shrinking matrix V turned on its side", "matrix",
              dict(values=M)),
        Panel("The total, kept", "each column of M adds up to 1: the 36 pixels' total and "
              "the 196 outputs' total", "bars",
              dict(values=np.array([total_in, total_out]),
                   labels=["6 x 6 picture", "14 x 14 output"])),
    )
    return Study("Bigger: a kernel larger than the picture", "A 9 x 9 kernel on a 6 x 6 "
                 "picture with padding 8: 14 x 14, each pixel spread into a blob of 81; M "
                 "(196 x 36) is the shrinking matrix turned on its side", panels, meta)


STUDIES = {"resample-big-kernel": big_kernel_study}

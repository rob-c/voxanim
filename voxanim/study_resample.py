"""Making a picture smaller and bigger with fixed convolution kernels, for the two resampling
animations of "Blocks and code" (catalogue 263-264): ``examples/resample.py`` in numpy.

* **Down** (:func:`down`): the 3 x 3 kernel ``[1, 2, 1] / 4`` times itself, convolved with
  stride 2 and padding 1 (``F.conv2d``): 64 x 64 -> 32 x 32. One axis at a time it is the
  matrix :func:`down_matrix` ``D`` (32 x 64, each row the kernel moved two places on), so the
  result is exactly ``D @ X @ D.T``. Against keeping every second pixel (``skip``).
* **Up** (:func:`up`): ``F.conv_transpose2d`` with the kernel 4 times as large and stride 2:
  the pixels spread out with zeros between them and the gaps blurred shut, bilinear
  interpolation; it is ``4 D.T @ Y @ D``. The last row and column receive half as much as
  the others (:func:`received`), so :func:`up_even` divides by it. Against repeating each
  pixel (nearest, ``blocks``).

Measured: on the zone plate (rings that get finer outwards; exact arithmetic), the false
rings that skipping leaves where the rings are too fine for 32 x 32 (:func:`aliasing`); on
the 400 Olivetti photos, how far each way of going down and back up lands from the photo
(:func:`round_trips`), over the whole picture and inside its 1-pixel border.
"""

from __future__ import annotations

import functools

import numpy as np

from . import olivetti

SIDE = 64
BLUR = np.array([1.0, 2.0, 1.0], np.float32) / 4
KERNEL = np.outer(BLUR, BLUR)


def zone_plate(n=SIDE):
    """``zone_plate`` of the script: (1 + cos(pi r^2 / n)) / 2, 0..1, centred on (n/2, n/2)."""
    y, x = np.meshgrid(np.arange(n) - n / 2, np.arange(n) - n / 2, indexing="ij")
    return ((1 + np.cos(np.pi * (x ** 2 + y ** 2) / n)) / 2).astype(np.float32)


def local_frequency(n=SIDE):
    """How fast the zone plate's rings go at each pixel: r / n cycles per pixel."""
    y, x = np.meshgrid(np.arange(n) - n / 2, np.arange(n) - n / 2, indexing="ij")
    return np.sqrt(x ** 2 + y ** 2) / n


@functools.lru_cache(maxsize=4)
def down_matrix(n=SIDE):
    """``down_matrix`` of the script: row ``i`` holds the kernel on pixels 2i - 1 .. 2i + 1."""
    D = np.zeros((n // 2, n), np.float32)
    for i in range(n // 2):
        for t, w in enumerate(BLUR):
            j = 2 * i - 1 + t
            if 0 <= j < n:
                D[i, j] = w
    return D


def down(img):
    """``down`` of the script: ``(small, skip)``, both 32 x 32."""
    img = np.asarray(img, np.float32)
    D = down_matrix(len(img))
    return (D @ img @ D.T).astype(np.float32), img[::2, ::2].copy()


def up(small):
    """``up`` of the script: ``(big, blocks)``, both 64 x 64."""
    small = np.asarray(small, np.float32)
    D = down_matrix(2 * len(small))
    big = 4 * D.T @ small @ D
    return big.astype(np.float32), np.repeat(np.repeat(small, 2, 0), 2, 1)


def received(n=SIDE // 2):
    """What each pixel of the up-sampled picture receives from a flat picture of ones."""
    return up(np.ones((n, n), np.float32))[0]


def up_even(small):
    """``up_even`` of the script: :func:`up`'s picture divided by :func:`received`."""
    return (up(small)[0] / received(len(small))).astype(np.float32)


def kernel_steps():
    """The kernel as the outer product of [1, 2, 1] / 4 with itself, and its sum."""
    return dict(row=BLUR, kernel=KERNEL, total=float(KERNEL.sum()))


def aliasing():
    """On the zone plate: where the rings are finer than the small grid can hold (more than a
    quarter of a cycle per original pixel), a faithful 32 x 32 picture is plain grey (0.5);
    the false pattern left there, as the root mean square of (value - 0.5), for skip, blur
    and the 2 x 2 average."""
    plate = zone_plate()
    small, skip = down(plate)
    area = plate.reshape(32, 2, 32, 2).mean((1, 3))
    f = local_frequency()[::2, ::2]
    fine = f > 0.25
    rms = lambda a: float(np.sqrt(((a[fine] - 0.5) ** 2).mean()))   # noqa: E731
    return dict(fine_pixels=int(fine.sum()), skip=rms(skip), blur=rms(small),
                average=rms(area), plate=plate, small=small, skipped=skip, area=area)


@functools.lru_cache(maxsize=1)
def round_trips():
    """Over the 400 Olivetti photos: the mean squared difference from the photo after going
    down and back up, for each pair (down: blur or skip; up: bilinear, bilinear with the edge
    made even, or nearest), over the whole picture and inside its 1-pixel border."""
    photos = olivetti.load()[0]
    out = {}
    for dn in ("blur", "skip"):
        for un in ("bilinear", "even", "nearest"):
            full = inner = 0.0
            for img in photos:
                small, skip = down(img)
                s = small if dn == "blur" else skip
                u = {"bilinear": lambda z: up(z)[0], "even": up_even,
                     "nearest": lambda z: up(z)[1]}[un](s)
                d = (u - img).astype(np.float64) ** 2
                full += d.mean()
                inner += d[1:-1, 1:-1].mean()
            out["%s_%s" % (dn, un)] = dict(full=full / len(photos), inner=inner / len(photos))
    return out


def photo(i=0):
    return olivetti.load()[0][i]

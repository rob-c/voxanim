"""The numbers behind animations 263-264 of "Blocks and code" (group ``zb``, "Resizing
pictures"): making a picture smaller and bigger with fixed convolution kernels, nothing
learned. Everything comes from :mod:`voxanim.study_resample`, the numpy twin of
``examples/resample.py`` (checked against torch by ``tests/test_resample_equivalence.py``);
this module picks what is shown and checks each claim the drawing makes with a second,
plain computation (:func:`conv3`: a 3 x 3 kernel slid over a picture, zeros past its edge).

* ``resample-down`` ("Smaller: blur, then skip"): the kernel, ``[1, 2, 1] / 4`` times itself:
  sixteenths 1 2 1 / 2 4 2 / 1 2 1, adding up to 1. It slides over photo 0 (a real photo,
  pixels 0 to 1) two pixels at a time: each pixel of the 32 x 32 picture is a weighted
  average of 9. The pixel worked out is picked by a rule (:func:`worked_at`: in the middle
  half of the picture, the one whose 9 pixels differ most, an eye's edge): 0.53. Then the
  matrix ``D`` (32 x 64): row ``i`` is the kernel's row on pixels 2i - 1 to 2i + 1, the
  kernel moved 2 places a row (row 0's first number falls past the edge, so it holds only
  0.5 and 0.25); ``D @ X @ D.T`` is conv2d's picture, one axis then the other, 1.7881e-07
  apart in float32 (what the script prints). On the zone plate (exact arithmetic), where
  the rings are finer than 32 x 32 can hold (outside a circle of radius 16 pixels: 827 of
  the 1,024 small pixels), a true copy is plain grey; skipping leaves false rings of RMS
  0.354, blur then skip 0.079.
* ``resample-up`` ("Bigger: spread, then blur"): photo 0's 32 x 32 picture spread out with a
  zero between its pixels, then the kernel x 4 slid over it: a kept pixel stays itself
  (weight 1), a gap between two gets half of each, a gap among four a quarter of each:
  bilinear interpolation (:func:`spread` and :func:`conv3` give ``conv_transpose2d``'s
  picture). Nearest repeats each pixel 2 x 2: blocks. The last row and column have one
  neighbour past the edge, so they receive half (0.5, the corner 0.25: ``received()``);
  ``up_even`` divides by what each pixel received. On the 400 photos, down then up, the
  mean squared difference (pixels 0 to 1) inside the 1-pixel border: bilinear 0.00129,
  nearest 0.00302; over the whole picture bilinear 0.00343, with the edge made even
  0.00199. Honest: skip then bilinear lands closer (0.00092 inside): blur's gain is no false
  patterns, not a closer copy.
"""

from __future__ import annotations

import numpy as np

from . import study_resample as R
from .blocks import code_lines, commas
from .study import Panel, Study

ON = "Olivetti faces"              # the evidence tags: "measured on Olivetti faces: ..."
PHOTO = 0                          # the photo shown: the script's photos[0] (person 0)
SOURCE = ("examples/resample.py (voxanim.study_resample, its numpy twin, checked against "
          "torch): fixed kernels, nothing learned; the zone plate is exact arithmetic; the "
          "photos are the 400 Olivetti faces (AT&T Laboratories Cambridge, 1992-94), "
          "64 x 64, pixels 0 to 1")
# what ``print((by_matrix - small).abs().max())`` of region "rs-same" prints in torch
# (float32 rounding; tests/test_study_blocks_zb.py runs it)
PRINTED_SAME = "tensor(1.7881e-07)"
SIXTEENTHS = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
FINE_RADIUS = 16.0                 # where the plate's rings are finer than 32 x 32 holds


def frac(n, d):
    return "%d/%d" % (n, d)


def err(v):
    """A mean squared error on screen: ``0.00129``."""
    return "%.5f" % float(v)


def conv3(img, k, stride=1):
    """A 3 x 3 kernel slid over ``img`` (``stride`` pixels at a time), zeros past the edge
    (padding 1): ``F.conv2d(img, k, stride, padding=1)`` written out, one output pixel a
    weighted sum of the 9 under the kernel."""
    img = np.asarray(img, np.float64)
    k = np.asarray(k, np.float64)
    pad = np.pad(img, 1)
    n = img.shape[0]
    out = np.zeros(((n - 1) // stride + 1,) * 2)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            r, c = i * stride, j * stride
            out[i, j] = (pad[r:r + 3, c:c + 3] * k).sum()
    return out


def spread(small):
    """The small picture spread out: each pixel at an even place, a zero between them (and
    the extra last row and column of ``output_padding=1``)."""
    small = np.asarray(small, np.float32)
    out = np.zeros((2 * len(small),) * 2, np.float32)
    out[::2, ::2] = small
    return out


def worked_at(img=None):
    """The output pixel worked out in 263: in the middle half of the small picture (rows and
    columns 8 to 23), the one whose 9 photo pixels differ most (the first, if tied)."""
    img = R.photo(PHOTO) if img is None else img
    best, at = -1.0, None
    for i in range(8, 24):
        for j in range(8, 24):
            p = img[2 * i - 1:2 * i + 2, 2 * j - 1:2 * j + 2]
            if p.max() - p.min() > best:
                best, at = float(p.max() - p.min()), (i, j)
    return at


def down_study():
    k = R.kernel_steps()
    photo = R.photo(PHOTO)
    small, skip = R.down(photo)
    at = worked_at(photo)
    i, j = at
    patch = photo[2 * i - 1:2 * i + 2, 2 * j - 1:2 * j + 2]
    out = float((patch.astype(np.float64) * k["kernel"]).sum())
    if abs(out - float(small[i, j])) > 1e-6:
        raise ValueError("the worked pixel is not conv2d's")
    if not np.allclose(conv3(photo, R.KERNEL, 2), small, atol=1e-6):
        raise ValueError("the sliding kernel is not down()")
    if not np.allclose(k["kernel"] * 16, SIXTEENTHS):
        raise ValueError("the kernel is not in sixteenths")
    D = R.down_matrix()
    for r in range(1, 32):           # every row is the one before, moved 2 places on
        if not np.array_equal(D[r, 2:], D[r - 1, :-2]):
            raise ValueError("D's rows are not the kernel moved 2 places")
    plate = R.zone_plate()
    psmall, _pskip = R.down(plate)
    dx = (D @ plate).astype(np.float32)
    by_matrix = (dx @ D.T).astype(np.float32)
    a = R.aliasing()
    fine = R.local_frequency()[::2, ::2] > 0.25
    centre = np.hypot(*np.meshgrid(np.arange(0, 64, 2) - 32.0, np.arange(0, 64, 2) - 32.0))
    if not np.array_equal(fine, centre > FINE_RADIUS):
        raise ValueError("the fine rings are not outside the circle")
    meta = dict(
        source=SOURCE, photo_index=PHOTO, person=PHOTO // 10, side=64, small_side=32,
        row=k["row"].tolist(), row_text=[frac(1, 4), frac(2, 4), frac(1, 4)],
        kernel=k["kernel"].tolist(), sixteenths=SIXTEENTHS.tolist(),
        kernel_text=[[frac(v, 16) for v in r] for r in SIXTEENTHS],
        total=k["total"], total16=int(SIXTEENTHS.sum()),
        photo=photo, small=small, skip=skip,
        at=list(at), centre=[2 * i, 2 * j], patch=patch.tolist(),
        patch_text=[["%.2f" % v for v in r] for r in patch], output=out,
        output_text="%.2f" % out, outputs=32 * 32, outputs_text=commas(32 * 32),
        D=D, D_shape=list(D.shape), D_corner=D[:4, :9].tolist(),
        D_corner_text=[["0" if v == 0 else "%g" % v for v in r] for r in D[:4, :9]],
        D_row_sums=D.sum(1).tolist(), D_first_row_sum=float(D[0].sum()),
        plate=plate, plate_small=psmall, plate_dx=dx, by_matrix=by_matrix,
        matrix_gap=float(np.abs(by_matrix - conv3(plate, R.KERNEL, 2)).max()),
        printed_same=PRINTED_SAME,
        fine_pixels=a["fine_pixels"], fine_radius=FINE_RADIUS, fine=fine,
        rms_skip=a["skip"], rms_blur=a["blur"], rms_average=a["average"],
        rms_skip_text="%.3f" % a["skip"], rms_blur_text="%.3f" % a["blur"],
        rms_average_text="%.3f" % a["average"], plate_skip=a["skipped"],
        plate_blur=a["small"], plate_area=a["area"],
        code_kernel=list(code_lines("rs-kernel")), code_down=list(code_lines("rs-down")),
        code_matrix=list(code_lines("rs-matrix")), code_same=list(code_lines("rs-same")),
        limitations=[
            "the kernel is fixed, not learned; it is one common choice of blur, not the "
            "only one",
            "the false rings are measured on one test pattern (exact arithmetic); a photo "
            "has less fine detail, so it shows them less",
            "conv2d pads with zeros: the top row and left column of the small picture are "
            "made from 6 of the 9 numbers, adding to 0.75, so they come out a little dark",
            "D X D.T equals conv2d up to float32 rounding: torch prints 1.7881e-07; numpy's "
            "float32 D X D.T against the kernel slid in float64 differs by about as much"])
    panels = (
        Panel("Photo 0, smaller", "the 64 x 64 photo, blur then skip (32 x 32), and skip "
              "alone", "images", dict(values=[photo, np.kron(small, np.ones((2, 2))),
                                              np.kron(skip, np.ones((2, 2)))],
                                      labels=["photo 0", "blur, then skip", "skip only"])),
        Panel("The matrix D", "32 x 64: row i is the kernel's row on pixels 2i - 1 to "
              "2i + 1", "matrix", dict(values=D)),
        Panel("The zone plate", "rings finer outwards; skip; blur then skip; the 2 x 2 "
              "average (32 x 32 shown at 64)", "images",
              dict(values=[plate] + [np.kron(v, np.ones((2, 2))) for v in
                                     (a["skipped"], a["small"], a["area"])],
                   labels=["plate", "skip only", "blur, then skip", "2 x 2 average"])),
        Panel("False rings", "RMS of (value - 0.5) on the 827 small pixels where the rings "
              "are too fine for 32 x 32", "bars",
              dict(values=np.array([a["skip"], a["blur"], a["average"]]),
                   labels=["skip only", "blur, then skip", "2 x 2 average"])),
    )
    return Study("Smaller: blur, then skip", "A 3 x 3 kernel that adds up to 1, moved 2 "
                 "pixels at a time: on the zone plate, false rings of RMS 0.354 by skipping, "
                 "0.079 by blurring first", panels, meta)


def up_study():
    photo = R.photo(PHOTO)
    small = R.down(photo)[0]
    big, blocks = R.up(small)
    sp = spread(small)
    k4 = 4 * R.KERNEL
    if not np.allclose(conv3(sp, k4), big, atol=1e-6):
        raise ValueError("spread then the kernel x 4 is not up()")
    got = R.received()
    fixed = R.up_even(small)
    rows = slice(16, 21)              # the close-up: small pixels 8-10 x 22-24, spread
    cols = slice(44, 49)
    gap2 = (18, 45)                   # a gap between two kept pixels
    gap4 = (17, 45)                   # a gap among four
    two = (float(sp[18, 44]), float(sp[18, 46]))
    four = (float(sp[16, 44]), float(sp[16, 46]), float(sp[18, 44]), float(sp[18, 46]))
    if abs(big[gap2] - sum(two) / 2) > 1e-6 or abs(big[gap4] - sum(four) / 4) > 1e-6:
        raise ValueError("a gap is not the average of its neighbours")
    if not np.allclose(big[::2, ::2], small, atol=1e-6):
        raise ValueError("a kept pixel is not itself")
    alias = R.aliasing()
    rt = R.round_trips()
    trip = {k: dict(full=float(v["full"]), inner=float(v["inner"])) for k, v in rt.items()}
    edge_rows = got[:-1, -1]
    if not (np.allclose(got[:-1, :-1], 1) and np.allclose(edge_rows, 0.5)
            and np.allclose(got[-1, :-1], 0.5) and got[-1, -1] == 0.25):
        raise ValueError("received() is not 1 inside, 0.5 on the edge, 0.25 at the corner")
    meta = dict(
        source=SOURCE, photo_index=PHOTO, person=PHOTO // 10, side=64, small_side=32,
        photo=photo, small=small, spread=sp, big=big, blocks=blocks, got=got, fixed=fixed,
        kernel4=k4.tolist(), kernel4_text=[["%g" % v for v in r] for r in k4],
        close_rows=[rows.start, rows.stop], close_cols=[cols.start, cols.stop],
        close_small=[[8, 11], [22, 25]],
        spread_close=sp[rows, cols].tolist(), big_close=big[rows, cols].tolist(),
        blocks_close=blocks[rows, cols].tolist(),
        spread_close_text=[["0" if v == 0 else "%.2f" % v for v in r] for r in sp[rows, cols]],
        big_close_text=[["%.2f" % v for v in r] for r in big[rows, cols]],
        blocks_close_text=[["%.2f" % v for v in r] for r in blocks[rows, cols]],
        gap2=list(gap2), gap4=list(gap4), two=list(two), four=list(four),
        gap2_value=float(big[gap2]), gap4_value=float(big[gap4]),
        two_text=["%.2f" % v for v in two], gap2_text="%.2f" % big[gap2],
        gap4_text="%.2f" % big[gap4],
        got_corner=got[-3:, -3:].tolist(),
        got_corner_text=[["%g" % v for v in r] for r in got[-3:, -3:]],
        edge=0.5, corner=0.25, zeros=int((sp == 0).sum()), kept=32 * 32,
        bottom_right=float(big[-1, -1]), bottom_right_fixed=float(fixed[-1, -1]),
        last_row=float(big[-1].mean()), last_row_fixed=float(fixed[-1].mean()),
        last_row_text="%.2f" % big[-1].mean(), last_row_fixed_text="%.2f" % fixed[-1].mean(),
        plate_skip=alias["skipped"], plate_blur=alias["small"],
        rms_skip_text="%.3f" % alias["skip"], rms_blur_text="%.3f" % alias["blur"],
        trips=trip, photos=400,
        inner_bilinear=trip["blur_bilinear"]["inner"],
        inner_nearest=trip["blur_nearest"]["inner"],
        full_bilinear=trip["blur_bilinear"]["full"], full_even=trip["blur_even"]["full"],
        inner_skip_bilinear=trip["skip_bilinear"]["inner"],
        inner_bilinear_text=err(trip["blur_bilinear"]["inner"]),
        inner_nearest_text=err(trip["blur_nearest"]["inner"]),
        full_bilinear_text=err(trip["blur_bilinear"]["full"]),
        full_even_text=err(trip["blur_even"]["full"]),
        inner_skip_bilinear_text=err(trip["skip_bilinear"]["inner"]),
        trip_photo=big,
        code_up=list(code_lines("rs-up")), code_edge=list(code_lines("rs-edge")),
        limitations=[
            "errors are mean squared differences on pixels from 0 to 1, over 400 photos of "
            "64 x 64, going down to 32 x 32 and back: one size, one kind of picture",
            "skip then bilinear lands closer to the photo (0.00092 inside the border) than "
            "blur then bilinear (0.00129): the blur throws fine detail away; what it buys "
            "is no false patterns, not a closer copy",
            "the edge fix only changes the last row and column, so inside the border the "
            "two bilinear errors are the same",
            "bilinear is one choice; bicubic and learned upsampling are others, not shown"])
    names = ["blur, bilinear", "blur, nearest", "skip, bilinear", "skip, nearest"]
    panels = (
        Panel("Photo 0, bigger", "32 x 32, spread out with zeros, bilinear, nearest, and "
              "with the edge made even", "images",
              dict(values=[np.kron(small, np.ones((2, 2))), sp, big, blocks, fixed],
                   labels=["32 x 32", "spread", "bilinear", "nearest", "even edge"])),
        Panel("What each pixel receives", "conv_transpose2d of a flat picture of ones: 1 "
              "inside, 0.5 on the last row and column, 0.25 at the corner", "matrix",
              dict(values=got)),
        Panel("Inside the border", "mean squared difference from the photo after down and "
              "up, inside the 1-pixel border, over 400 photos", "bars",
              dict(values=np.array([trip["blur_bilinear"]["inner"],
                                    trip["blur_nearest"]["inner"],
                                    trip["skip_bilinear"]["inner"],
                                    trip["skip_nearest"]["inner"]]), labels=names)),
        Panel("The whole picture", "the same, over every pixel: making the edge even "
              "(up_even) closes most of the gap", "bars",
              dict(values=np.array([trip["blur_bilinear"]["full"], trip["blur_even"]["full"],
                                    trip["blur_nearest"]["full"],
                                    trip["skip_bilinear"]["full"],
                                    trip["skip_even"]["full"]]),
                   labels=["blur, bilinear", "blur, even edge", "blur, nearest",
                           "skip, bilinear", "skip, even edge"])),
    )
    return Study("Bigger: spread, then blur", "Zeros between the pixels, then the kernel x 4: "
                 "bilinear. On 400 photos, down and up: 0.00129 against nearest's 0.00302 "
                 "inside the border", panels, meta)


STUDIES = {"resample-down": down_study, "resample-up": up_study}

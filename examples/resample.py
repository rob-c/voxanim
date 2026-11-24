"""Making a picture smaller and bigger with conv2d: the program of the resampling animations of
"Blocks and code".

    python examples/resample.py

Two everyday image tasks done with fixed convolution kernels, nothing learned:

1. Down: blur with a 3 x 3 kernel, then keep every second pixel (conv2d with stride 2). Just
   keeping every second pixel, without the blur, lets fine detail turn into false patterns
   (aliasing). One axis at a time, the blur-and-skip is a matrix D with a row per kept pixel,
   each row the kernel moved two places on, so conv2d(image) is D @ image @ D.T exactly.
2. Up: spread the pixels out with zeros between them and blur the gaps shut
   (conv_transpose2d with stride 2): with the kernel [1, 2, 1] / 2 that is bilinear
   interpolation, and its matrix is 2 D.T.

Shown on a test pattern (a zone plate: rings that get finer outwards) and on photos of the
Olivetti faces (scikit-learn downloads them). Needs torch and scikit-learn. As in
mnist_blocks.py, "# region: NAME" ... "# endregion" marks what goes on screen.
"""
# region: rs-imports
import torch
import torch.nn.functional as F
from sklearn.datasets import fetch_olivetti_faces

# endregion

SIDE = 64


def zone_plate(n=SIDE):
    """Rings that get finer towards the edge: cos(pi r^2 / n), 0..1."""
    y, x = torch.meshgrid(torch.arange(n) - n / 2, torch.arange(n) - n / 2, indexing="ij")
    return (1 + torch.cos(torch.pi * (x ** 2 + y ** 2) / n)) / 2


# region: rs-kernel
blur = torch.tensor([1.0, 2.0, 1.0]) / 4  # a little of each neighbour
k = blur[:, None] * blur[None, :]         # 3 x 3, adds up to 1
k = k[None, None]                         # [out, in, 3, 3] for conv2d
# endregion


def down(img):
    # region: rs-down
    x = img[None, None]                   # [1, 1, 64, 64]
    small = F.conv2d(x, k, stride=2,      # blur, keep every 2nd
                     padding=1)           # [1, 1, 32, 32]
    skip = x[..., ::2, ::2]               # just every 2nd: no blur
    # endregion
    return small[0, 0], skip[0, 0]


def down_matrix(n=SIDE):
    # region: rs-matrix
    D = torch.zeros(n // 2, n)            # a row per pixel kept
    for i in range(n // 2):               # row i: centred on pixel 2i
        for t, w in enumerate(blur):      # the kernel's 3 numbers
            j = 2 * i - 1 + t             # on pixels 2i - 1 to 2i + 1
            if 0 <= j < n:                # none past the edge
                D[i, j] = w
    # endregion
    return D


def up(small):
    # region: rs-up
    y = small[None, None]                 # [1, 1, 32, 32]
    big = F.conv_transpose2d(             # spread, then blur
        y, 4 * k, stride=2, padding=1,
        output_padding=1)                 # [1, 1, 64, 64]
    blocks = y.repeat_interleave(2, 2)    # each pixel twice down
    blocks = blocks.repeat_interleave(2, 3)  # and across: blocks
    # endregion
    return big[0, 0], blocks[0, 0]


def up_even(small):
    """Upsampling whose edge is not dimmed: the last row and column receive only half of what
    the others do, so divide every pixel by what it received."""
    big, _blocks = up(small)
    # region: rs-edge
    ones = torch.ones_like(small)         # a flat grey picture
    got, _ = up(ones)                     # what each pixel received
    fixed = big / got                     # 0.5 at the edge: doubled
    # endregion
    return fixed


if __name__ == "__main__":
    plate = zone_plate()
    small, skip = down(plate)
    D = down_matrix()
    # region: rs-same
    by_matrix = D @ plate @ D.T           # one axis, then the other
    print((by_matrix - small).abs().max())  # 0: the same numbers
    # endregion
    photos = torch.tensor(fetch_olivetti_faces().images)
    big, blocks = up(down(photos[0])[0])
    for name, img in (("bilinear", big), ("nearest", blocks)):
        print(name, ((img - photos[0]) ** 2).mean().item())

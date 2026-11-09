"""The convolution teaching set: one idea each, titled and labelled.

They share the originals' look but use fewer filters so each one is short, and
they caption every tensor with its shape.  Each is registered in
:data:`voxanim.presets.REGISTRY` under its media name:

======================================  ==========================================
``lesson-stride-1``, ``-2``, ``-3``     step size: 9x9 becomes 7x7, 4x4, 3x3
``lesson-dilation-2``, ``-3``           gaps between the kernel's taps
``lesson-transposed-conv``              upsampling: zeros between pixels, then conv
``lesson-max-pool``, ``-avg-pool``      2x2 pooling, stride 2, no weights
``lesson-strided-downsample``           a learned stride-2 kernel halves the size
``lesson-pointwise``                    a 1x1 convolution only mixes channels
``lesson-conv3d``                       a 3x3x3 kernel through a 6x6x6 volume
``lesson-conv3d-channels``              conv3d with 2 channels and 2 filters
``lesson-conv1d-stride-2``              conv1d hopping two steps
``lesson-conv1d-dilation-2``            conv1d with taps two steps apart
``lesson-edge-detection``               Sobel x and y, then the magnitude ``|G|``
======================================  ==========================================

The network animations (MLP, CNN, attention, ...) are registered by
:mod:`voxanim.catalog`, imported at the end of this module.
"""

from __future__ import annotations

from functools import partial

import numpy as np

from . import color as C
from .conv import CombineSpec, ConvAnimation, ConvSpec, Layout, Tempo
from .presets import preset
from .text import TIMES

# A roomier default than the originals': labels need space under each block.
EDU = Layout(input=(0.215, 0.46), output=(0.75, 0.62), active=(0.47, 0.52),
             queue=(0.63, 0.30), done=(0.10, 0.90), ppu=22.0 / 540.0)
QUICK = Tempo(slow=1.0, fast=0.16, ramp_in=6, ramp_out=2)

# Seeds of the made-up data, fixed so every render shows the same numbers.
POOL_SEED = 3                   # the pooling lessons' 8x8x4 input
SEQ_SEED = 5                    # the conv1d lessons' 16-step sequence
CONV1D_WEIGHT_SEED = 9          # ... and their two 3-tap filters


def register(name, about=None):
    """Decorator registering a factory as preset ``name`` (its media file is ``name`` too)."""
    return preset(name, name, about=about)


def lesson(name, title, subtitle=None, **spec):
    """A single-stage, titled, labelled animation."""
    spec.setdefault("layout", EDU)
    spec.setdefault("tempo", QUICK)
    return ConvAnimation(ConvSpec(name=name, **spec), title=title, subtitle=subtitle,
                         labels=True)


# ------------------------------------------------------------------ step size


def _stride(s):
    """Stride ``s`` on a 9x9x4 input with two 3x3 filters."""
    out = (9 - 3) // s + 1
    return lesson("stride-%d" % s, in_hw=(9, 9), in_ch=4, filters=2, stride=(s, s),
                  title="Stride %d" % s,
                  subtitle="the kernel steps %d pixel%s at a time: 9%s9 becomes %d%s%d"
                  % (s, "" if s == 1 else "s", TIMES, out, TIMES, out),
                  tempo=Tempo(slow=1.0, fast=0.16 if s == 1 else 0.3, ramp_in=6, ramp_out=2))


for _s in (1, 2, 3):
    register("lesson-stride-%d" % _s,
             about="Step size %d: a 3x3 kernel over a 9x9x4 input." % _s)(partial(_stride, _s))


# ------------------------------------------------------------------ dilation


def _dilation(d):
    """Dilation ``d``: the kernel spreads its taps as it becomes active."""
    ext = 2 * d + 1
    out = 9 - ext + 1
    L = Layout(input=(0.20, 0.47), output=(0.78, 0.62), active=(0.49, 0.53),
               queue=(0.68, 0.29), done=(0.10, 0.90), ppu=20.0 / 540.0)
    return lesson("dilation-%d" % d, in_hw=(9, 9), in_ch=4, filters=2, dilation=(d, d),
                  layout=L,
                  title="Dilation %d" % d,
                  subtitle="%d-pixel gaps between the taps: a 3%s3 kernel sees %d%s%d, "
                  "9%s9 becomes %d%s%d" % (d - 1, TIMES, ext, TIMES, ext, TIMES, out, TIMES, out),
                  tempo=Tempo(slow=1.0, fast=0.2, ramp_in=6, ramp_out=2))


for _d in (2, 3):
    register("lesson-dilation-%d" % _d,
             about="Dilation %d: gaps between the kernel's taps." % _d)(partial(_dilation, _d))


# ------------------------------------------------------------------ scaling


@register("lesson-transposed-conv")
def transposed():
    """Upsampling: zeros between the pixels, then an ordinary convolution."""
    L = Layout(input=(0.20, 0.45), output=(0.76, 0.60), active=(0.50, 0.50),
               queue=(0.64, 0.29), done=(0.10, 0.90), ppu=19.0 / 540.0)
    return lesson("transposed", in_hw=(4, 4), in_ch=3, filters=2, input_dilation=(2, 2),
                  padding=(2, 2, 2, 2), layout=L,
                  tempo=Tempo(slow=0.9, fast=0.09, ramp_in=6, ramp_out=2),
                  title="Transposed convolution (stride 2)",
                  subtitle="spread the pixels apart, fill with zeros, pad, convolve: "
                  "4%s4 is upsampled to 9%s9" % (TIMES, TIMES),
                  labels={"input": "input 4%s4%s3, spread and padded to 11%s11"
                          % (TIMES, TIMES, TIMES)})


def _pool_data(seed=POOL_SEED):
    """An 8x8x4 input, uniform in [-0.9, 0.9]."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.9, 0.9, (8, 8, 4))


def _pool(kind):
    """2x2 pooling, stride 2; ``kind`` is ``"max"`` or ``"avg"``."""
    name = "Max" if kind == "max" else "Average"
    keeps = "largest value" if kind == "max" else "mean"
    return lesson("%s-pool" % kind, in_hw=(8, 8), in_ch=4, kernel=(2, 2), stride=(2, 2),
                  pool=kind, data=_pool_data(),
                  input_colors=[C.rgb("#3f8fb0")],
                  tempo=Tempo(slow=1.1, fast=0.35, ramp_in=4, ramp_out=2),
                  title="%s pooling 2%s2, stride 2" % (name, TIMES),
                  subtitle="no weights: every channel keeps its %s of each window, "
                  "8%s8 becomes 4%s4" % (keeps, TIMES, TIMES))


register("lesson-max-pool",
         about="Max pooling 2x2, stride 2: downsampling without weights.")(partial(_pool, "max"))
register("lesson-avg-pool", about="Average pooling 2x2, stride 2.")(partial(_pool, "avg"))


@register("lesson-strided-downsample")
def strided_down():
    """Downsampling with a learned kernel: stride 2 and same padding."""
    return lesson("down", in_hw=(8, 8), in_ch=4, filters=3, stride=(2, 2),
                  padding=(1, 0, 1, 0), tempo=Tempo(slow=1.0, fast=0.3, ramp_in=5, ramp_out=2),
                  title="Downsampling with a strided convolution",
                  subtitle="stride 2 halves the resolution: 8%s8 becomes 4%s4" % (TIMES, TIMES))


@register("lesson-pointwise")
def pointwise():
    """A 1x1 convolution only mixes channels."""
    return lesson("pointwise", in_hw=(6, 6), in_ch=8, filters=4, kernel=(1, 1),
                  tempo=Tempo(slow=0.9, fast=0.1, ramp_in=6, ramp_out=2),
                  title="1%s1 convolution" % TIMES,
                  subtitle="each output pixel mixes the 8 channels of one input pixel")


# ------------------------------------------------------------------ conv3d


@register("lesson-conv3d")
def conv3d():
    """A 3x3x3 kernel slides along all three axes of a volume."""
    L = Layout(input=(0.21, 0.45), output=(0.75, 0.63), active=(0.48, 0.48),
               queue=(0.62, 0.30), done=(0.10, 0.88), ppu=22.0 / 540.0)
    return lesson("conv3d", in_hw=(6, 6), in_d=6, in_ch=1, filters=1, kernel=(3, 3),
                  kernel_d=3, layout=L, filter_colors=[C.FILTERS[0]],
                  tempo=Tempo(slow=1.1, fast=0.22, ramp_in=6, ramp_out=2),
                  title="Conv3D",
                  subtitle="the kernel slides along depth too: a 6%s6%s6 volume "
                  "becomes 4%s4%s4" % (TIMES, TIMES, TIMES, TIMES))


@register("lesson-conv3d-channels")
def conv3d_channels():
    """Conv3D with 2 input channels and 2 filters (one slab per channel)."""
    L = Layout(input=(0.20, 0.44), output=(0.76, 0.63), active=(0.48, 0.50),
               queue=(0.65, 0.28), done=(0.10, 0.90), ppu=19.0 / 540.0)
    return lesson("conv3d-2ch", in_hw=(5, 5), in_d=5, in_ch=2, filters=2, kernel=(3, 3),
                  kernel_d=3, layout=L,
                  tempo=Tempo(slow=1.1, fast=0.25, ramp_in=6, ramp_out=2),
                  title="Conv3D with channels",
                  subtitle="each channel is a 5%s5%s5 volume; a filter spans both "
                  "channels and 3 slices of depth" % (TIMES, TIMES))


# ------------------------------------------------------------------ conv1d


def _seq(L, Cin, seed):
    """``(L, 1, Cin)``: noisy, phase-shifted sine waves clipped to [-1, 1]."""
    rng = np.random.default_rng(seed)
    t = np.arange(L)[:, None]
    x = 0.7 * np.sin(t * 0.7 + np.arange(Cin)[None] * 1.3) + rng.normal(0, 0.15, (L, Cin))
    return np.clip(x, -1, 1)[:, None, :]


def _conv1d(name, title, subtitle, **kw):
    """Conv1D over a 16-step, 4-channel sequence with two 3-tap filters; ``kw`` go
    to the :class:`ConvSpec` (stride, dilation)."""
    L = Layout(input=(0.24, 0.47), output=(0.74, 0.60), active=(0.48, 0.52),
               queue=(0.63, 0.27), done=(0.10, 0.90), queue_axis="x", spacing=2.2,
               ppu=20.0 / 540.0)
    data = _seq(16, 4, SEQ_SEED)
    w = np.random.default_rng(CONV1D_WEIGHT_SEED).normal(0, 0.5, (2, 3, 1, 4))
    return lesson(name, in_hw=(16, 1), in_ch=4, filters=2, kernel=(3, 1), data=data,
                  weights=w, layout=L, title=title, subtitle=subtitle,
                  tempo=Tempo(slow=1.0, fast=0.22, ramp_in=5, ramp_out=2), **kw)


register("lesson-conv1d-stride-2",
         about="Conv1D with stride 2 over a 16-step, 4-channel sequence.")(
    lambda: _conv1d("conv1d-s2", "Conv1D, stride 2",
                    "the kernel hops two steps: 16 steps become 7", stride=(2, 1)))
register("lesson-conv1d-dilation-2", about="Conv1D with dilation 2 (WaveNet-style).")(
    lambda: _conv1d("conv1d-d2", "Conv1D, dilation 2",
                    "taps two steps apart see 5 steps at once: 16 become 12", dilation=(2, 1)))


# ------------------------------------------------------------------ edge detection


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], float)
SOBEL_Y = SOBEL_X.T.copy()


def edge_image(n=16):
    """A small greyscale picture with straight and curved edges."""
    yy, xx = np.mgrid[0:n, 0:n] + 0.5
    img = np.zeros((n, n))           # 0 = the zero padding: no false border edges
    disc = (xx - n * 0.64) ** 2 + (yy - n * 0.62) ** 2 < (n * 0.26) ** 2
    square = (xx > n * 0.12) & (xx < n * 0.46) & (yy > n * 0.14) & (yy < n * 0.48)
    img[square] = 0.62
    img[disc] = 0.95
    return img


@register("lesson-edge-detection")
def edge_detection():
    """Sobel filters over a greyscale image: Gx and Gy, then |G|."""
    n = 16
    img = edge_image(n)[:, :, None]
    common = dict(in_hw=(n, n), in_ch=1, filters=1, kernel=(3, 3), padding=(1, 1, 1, 1),
                  data=img, pad_color=tuple(C.gray(0.0).tolist()),
                  styles={"input": "gray", "kernel": "diverging", "output": "diverging"},
                  tempo=Tempo(slow=0.9, fast=0.04, ramp_in=8, ramp_out=2),
                  kernel_scale=2.4)
    view = dict(ppu=11.5 / 540.0, azimuth=33.0, elevation=24.0, floor_gap=0.9)
    # Caption each piece of the pipeline (shape labels add little here).
    gx = ConvSpec(name="sobel-x", weights=SOBEL_X[None, :, :, None],
                  labels={"input": "image %d%s%d" % (n, TIMES, n), "kernel": "Sobel x",
                          "output": "Gx"},
                  filter_colors=[C.NEUTRAL],
                  layout=Layout(input=(0.165, 0.50), active=(0.39, 0.25), output=(0.56, 0.28),
                                queue=(0.39, 0.25), done=(0.39, 0.25), **view), **common)
    gy = ConvSpec(name="sobel-y", weights=SOBEL_Y[None, :, :, None], source="first",
                  labels={"kernel": "Sobel y", "output": "Gy"}, filter_colors=[C.NEUTRAL],
                  layout=Layout(active=(0.39, 0.76), output=(0.56, 0.75),
                                queue=(0.39, 0.76), done=(0.39, 0.76), **view), **common)
    mag = CombineSpec(sources=(0, 1), op="magnitude", at=(0.84, 0.52), style="gray",
                      label="|G| = sqrt(Gx² + Gy²)  edges")
    return ConvAnimation([gx, gy], title="Edge detection with Sobel filters",
                         subtitle="orange = positive, blue = negative weights and responses",
                         labels=True, combine=[mag], hold=2.2)

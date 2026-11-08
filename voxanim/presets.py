"""Named animations.

Each preset is a zero-argument factory returning a :class:`ConvAnimation`, so
adding an animation is one function.  ``media`` is the basename the site uses
(under ``../media``) and ``reference`` the original GIF it re-creates, if any.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import import_module

import numpy as np

from . import color as C
from .conv import ConvAnimation, ConvSpec, Layout, Tempo


@dataclass(frozen=True)
class Preset:
    name: str
    build: callable
    media: str
    reference: str | None = None
    about: str = ""


REGISTRY: dict[str, Preset] = {}


def preset(name, media, reference=None, about=None):
    def deco(fn):
        REGISTRY[name] = Preset(name, fn, media, reference,
                                (about or fn.__doc__ or name).strip())
        return fn
    return deco


# ------------------------------------------------------------- conv2d family


@preset("conv2d", "convolution-animation-3x3-kernel",
        "convolution-animation-3x3-kernel.gif")
def conv2d():
    """7x7x8 input, eight 3x3x8 filters, stride 1, no padding -> 5x5x8."""
    return ConvAnimation(ConvSpec())


@preset("conv2d-same-padding", "convolution-animation-3x3-kernel-same-padding",
        "convolution-animation-3x3-kernel-same-padding.gif")
def conv2d_same():
    """[1,1,1,1] ('same') zero padding keeps the output at 7x7."""
    L = Layout(input=(0.232, 0.255), output=(0.722, 0.700))
    return ConvAnimation(ConvSpec(name="conv2d-same", padding=(1, 1, 1, 1), layout=L,
                                  tempo=Tempo(fast=0.15)))


@preset("conv2d-stride-2", "convolution-animation-3x3-kernel-stride-2",
        "convolution-animation-3x3-kernel-stride-2.gif")
def conv2d_stride2():
    """Stride 2 skips every other window: 7x7 -> 3x3."""
    return ConvAnimation(ConvSpec(name="conv2d-stride-2", stride=(2, 2),
                                  tempo=Tempo(slow=1.1, fast=0.38, ramp_in=5)))


@preset("conv2d-stride-2-same-padding",
        "convolution-animation-3x3-kernel-stride-2-same-padding",
        "convolution-animation-3x3-kernel-stride-2-same-padding.gif")
def conv2d_stride2_same():
    """Stride 2 with 'same' padding: 9x9 padded -> 4x4."""
    L = Layout(input=(0.232, 0.255))
    return ConvAnimation(ConvSpec(name="conv2d-stride-2-same", stride=(2, 2),
                                  padding=(1, 1, 1, 1), layout=L,
                                  tempo=Tempo(slow=1.1, fast=0.32, ramp_in=6)))


GROUP_INPUT = [C.rgb("#2f7c7a"), C.rgb("#5fb0ae")]
GROUP_FILTERS = [C.rgb(h) for h in ("#a83736", "#4d7f37", "#2f7d7b", "#6a3fc4",
                                    "#e7615f", "#71c25a", "#4fb3b1", "#9a7cf0")]


@preset("conv2d-2-groups", "convolution-animation-3x3-kernel-2-groups",
        "convolution-animation-3x3-kernel-2-groups.gif")
def conv2d_groups():
    """Two groups: each half of the filters only sees half of the channels."""
    L = Layout(input=(0.17, 0.20), output=(0.86, 0.80), active=(0.47, 0.45),
               queue=(0.66, 0.37), done=(0.13, 0.84))
    return ConvAnimation(ConvSpec(name="conv2d-2-groups", groups=2,
                                  input_colors=GROUP_INPUT, filter_colors=GROUP_FILTERS,
                                  layout=L))


@preset("depthwise", "depthwise-convolution-animation-3x3-kernel",
        "depthwise-convolution-animation-3x3-kernel.gif")
def depthwise():
    """Eight groups: one 3x3x1 filter per channel."""
    L = Layout(input=(0.20, 0.22), output=(0.86, 0.80), active=(0.47, 0.44),
               queue=(0.62, 0.30), done=(0.17, 0.66), spacing=0.8)
    return ConvAnimation(ConvSpec(name="depthwise", groups=8,
                                  filter_colors=C.shades("#d44a48", 8, -0.24, 0.16),
                                  layout=L, tempo=Tempo(fast=0.16), track_groups=False))


@preset("depthwise-separable", "depthwise-separable-convolution-animation-3x3-kernel",
        "depthwise-separable-convolution-animation-3x3-kernel.gif")
def depthwise_separable():
    """A depthwise layer ('same' padded) followed by a 1x1 pointwise layer."""
    L1 = Layout(input=(0.15, 0.215), output=(0.415, 0.56), active=(0.24, 0.40),
                queue=(0.33, 0.21), done=(0.08, 0.53), spacing=0.8)
    L2 = Layout(output=(0.80, 0.81), active=(0.62, 0.66), queue=(0.82, 0.44),
                done=(0.50, 0.86), queue_axis="x", spacing=0.6)
    dw = ConvSpec(name="depthwise", groups=8, padding=(1, 1, 1, 1), track_groups=False,
                  filter_colors=C.shades("#d44a48", 8, -0.24, 0.16), layout=L1,
                  tempo=Tempo(slow=0.9, fast=0.07, ramp_in=8, ramp_out=1.5))
    pw = ConvSpec(name="pointwise", in_hw=(7, 7), kernel=(1, 1), layout=L2,
                  tempo=Tempo(slow=0.8, fast=0.06, ramp_in=6))
    return ConvAnimation([dw, pw])


# ------------------------------------------------------------- conv1d


def _conv1d_data(seed=7):
    """A 12-step, 8-channel sequence with two motifs that filter 0 detects."""
    rng = np.random.default_rng(seed)
    L, Cin, F, K = 12, 8, 4, 3
    x = rng.normal(0.0, 0.22, (L, Cin))
    motif = np.array([[0.9, 0.5, -0.3, -0.8, -0.4, 0.2, 0.7, 0.9],
                      [0.6, 0.9, 0.2, -0.5, -0.9, -0.2, 0.5, 0.6],
                      [0.1, 0.6, 0.8, 0.1, -0.6, -0.7, 0.0, 0.3]])
    x[2:5] += motif
    x[7:10] -= 0.8 * motif
    w = rng.normal(0.0, 0.45, (F, K, 1, Cin))
    w[0, :, 0] = motif
    return np.clip(x, -1, 1)[:, None, :], w / np.abs(w).max()


@preset("conv1d-full-width-scroll", "conv1d-full-width-scroll")
def conv1d_scroll():
    """conv1d over a 12-step sequence with 8 channels: four 3x8 filters slide
    down the time axis (rows) and each window collapses to one value per filter,
    so the output is 10 steps x 4 channels.  Voxel brightness is the value."""
    data, w = _conv1d_data()
    L = Layout(input=(0.235, 0.40), output=(0.735, 0.585), active=(0.47, 0.50),
               queue=(0.60, 0.30), done=(0.10, 0.86), queue_axis="x", spacing=2.2)
    spec = ConvSpec(name="conv1d", in_hw=(12, 1), in_ch=8, filters=4, kernel=(3, 1),
                    data=data, weights=w, layout=L,
                    tempo=Tempo(slow=1.0, fast=0.24, ramp_in=6, ramp_out=2))
    return ConvAnimation(spec)


def _sweep_data(seed=11):
    rng = np.random.default_rng(seed)
    H, W = 8, 8
    x = np.linspace(-0.35, 0.15, W)[None, :] + rng.normal(0, 0.08, (H, W))
    for r in range(H):
        for c in rng.choice(np.arange(1, W - 1), size=2, replace=False):
            x[r, c] = rng.uniform(0.85, 1.0)
    return np.clip(x, -1, 1)[:, :, None], np.array([[[[0.3], [1.0], [0.3]]]])


@preset("conv1d-row-sweep", "conv1d-row-sweep")
def conv1d_sweep():
    """A 1x3 kernel reads each row of an 8x8 single-channel sheet in turn,
    boustrophedon (left->right, then right->left), so every 8-wide row becomes
    6 responses.  The kernel is a matched filter for the bright spikes."""
    data, w = _sweep_data()
    L = Layout(input=(0.27, 0.45), output=(0.75, 0.60), active=(0.52, 0.30),
               queue=(0.70, 0.22), done=(0.15, 0.86), queue_axis="x", ppu=26.0 / 540.0)
    spec = ConvSpec(name="row-sweep", in_hw=(8, 8), in_ch=1, filters=1, kernel=(1, 3),
                    order="serpentine", data=data, weights=w,
                    filter_colors=[C.FILTERS[0]], layout=L,
                    tempo=Tempo(slow=0.9, fast=0.13, ramp_in=6, ramp_out=2))
    return ConvAnimation(spec)


# The teaching set registers itself into REGISTRY (imported last: it uses
# ``preset`` from this module).
import_module(".lessons", __package__)


def build(name, **kw):
    """Build preset ``name``, passing only the options its factory accepts
    (``--model``, ``--digit``, ...)."""
    import inspect
    fn = get(name).build
    params = inspect.signature(fn).parameters
    return fn(**{k: v for k, v in kw.items() if k in params and v is not None})


def options_suffix(name, **kw):
    """File-name suffix for non-default options, e.g. ``_digit3``."""
    import inspect
    params = inspect.signature(get(name).build).parameters
    parts = []
    for k, v in sorted(kw.items()):
        if k in params and v is not None:
            v = os.path.splitext(os.path.basename(str(v)))[0] if k == "model" else v
            if k == "text":                        # a prompt: keep file names tidy
                v = "-" + "-".join("".join(c if c.isalnum() else " " for c in v).split())[:40]
            parts.append("%s%s" % (k, v))
    return ("_" + "_".join(parts)) if parts else ""


def get(name) -> Preset:
    try:
        return REGISTRY[name]
    except KeyError:
        raise SystemExit("unknown animation %r; try: %s" % (name, ", ".join(REGISTRY)))

"""Convolution choreography.

One :class:`ConvSpec` describes a convolution *and* how it is staged; one
:class:`ConvStage` turns it into actors on a timeline:

* the **input** block (optionally zero-padded, optionally with zeros inserted
  between its pixels for a transposed convolution, optionally split into
  channel groups that are pushed apart and ghosted when inactive),
* one **kernel** block per filter, travelling queue -> active slot -> "done"
  row (spread out with gaps between its taps when the kernel is dilated),
* the **output** block inside a wireframe cage, filling one cell per step,
* the white **receptive-field frame** and the **multiply/accumulate wave** that
  sweeps through the window and the kernel along the accumulation axis.

The same grammar covers conv1d (one spatial axis), conv2d, conv3d (a depth
axis, drawn along the channel axis: each input channel becomes a slab of depth
cells), grouped / depthwise convolutions, dilation, transposed convolution and
max / average pooling (no weights: every channel at once).

Every quantity is a pure function of time, so ``scene(t)`` can be evaluated in
any order (parallel rendering, motion-blur sub-frames, previews).

Stages chain: by default a stage reads the previous stage's output (the
depthwise-separable animation), or with ``source="first"`` the first stage's
input (several filters applied to one image, e.g. Sobel x and y).
:class:`CombineSpec` adds a cell-wise step that merges stage outputs (the
gradient magnitude of an edge detector).
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field

import numpy as np

from . import color as C
from . import layout
from .camera import Camera
from .easing import ease_in_out, ease_out, lerp, remap, smoothstep, soft_band
from .scene import Scene
from .tensor import Block


X = "\u00d7"


# ------------------------------------------------------------------ spec


@dataclass
class Tempo:
    """Per-step durations: slow while the idea is introduced, then fast, then
    slowing again for the last few steps -- the originals' pacing."""

    slow: float = 1.0          # seconds for the first step
    fast: float = 0.20         # cruising speed
    ramp_in: float = 9.0       # steps over which it speeds up (e-folding)
    ramp_out: float = 2.5      # steps over which it slows down at the end

    def durations(self, n):
        k = np.arange(n, dtype=np.float64)
        e = np.maximum(np.exp(-k / self.ramp_in), np.exp(-(n - 1 - k) / self.ramp_out))
        return self.fast + (self.slow - self.fast) * e


@dataclass
class Layout:
    """Where things go, as fractions of the frame (x right, y down).

    Anchors are screen positions of bounding-box centres at height 0; the
    renderer is orthographic so this is resolution independent.
    """

    input: tuple = (0.244, 0.262)
    output: tuple = (0.718, 0.690)
    active: tuple = (0.500, 0.495)
    queue: tuple = (0.612, 0.380)      # first waiting kernel
    done: tuple = (0.100, 0.855)       # first finished kernel
    queue_axis: str = "auto"           # "x", "y" or "auto"
    spacing: float = 1.0               # gap between kernels in a row (voxels)
    floor_gap: float = 0.7             # floor distance below the lowest block
    ppu: float = 17.0 / 540.0          # pixels per voxel, as a fraction of height
    azimuth: float = 45.0              # camera (first stage's layout wins)
    elevation: float = 29.4


@dataclass
class ConvSpec:
    name: str = "conv2d"
    in_hw: tuple = (7, 7)
    in_ch: int = 8
    filters: int = 8
    kernel: tuple = (3, 3)
    stride: tuple = (1, 1)
    padding: tuple = (0, 0, 0, 0)      # top, bottom, left, right
    dilation: tuple = (1, 1)           # gaps between kernel taps ("atrous")
    input_dilation: tuple = (1, 1)     # zeros between input pixels (transposed conv)
    groups: int = 1
    # conv3d: a depth axis.  It is drawn along the channel axis -- every input
    # channel becomes a slab ``in_d`` cells deep -- and the kernel slides along
    # it too.
    in_d: int = 1
    kernel_d: int = 1
    stride_d: int = 1
    pad_d: tuple = (0, 0)
    pool: str | None = None            # "max" / "avg": no weights, every channel at once
    order: str = "raster"              # "raster" or "serpentine"
    source: str = "prev"               # chained stages: "prev" output or "first" input

    filter_colors: list | None = None
    input_colors: list | None = None   # one per group
    pad_color: tuple = tuple(C.PAD.tolist())

    # Optional real numbers: input (H, W, C[, D]) and weights
    # (F, kh, kw, C // groups[, kd]).  When given, voxels are coloured by value
    # (see ``styles``) and the output shows the actual convolution.
    data: np.ndarray | None = None
    weights: np.ndarray | None = None
    # How values become colours, per role: "hue" (the role's colour, lighter =
    # larger), "diverging" (warm +, cool -) or "gray" (an image in [0, 1]).
    styles: dict = field(default_factory=dict)
    bias: np.ndarray | None = None     # per filter, added after the convolution
    activation: str | None = None      # "relu": the output keeps only values > 0

    tempo: Tempo = field(default_factory=Tempo)
    transition: tuple = (0.30, 0.80)   # min/max seconds to swap kernels
    group_transition: float = 1.0
    intro: float = 0.4
    explode: float | None = None       # gap between input channel groups
    slab_gap: float = 0.8              # gap between depth slabs (conv3d)
    track_groups: bool = True          # slide the input so the active group stays put
    ghost: float = 0.10                # opacity of inactive groups
    queue_shift: bool | None = None    # waiting kernels advance as one is taken
    layout: Layout = field(default_factory=Layout)
    labels: dict | bool | None = None  # None: automatic shape labels
    kernel_scale: float = 1.0          # draw kernels magnified (to read their weights)

    # -- derived ------------------------------------------------------------
    @property
    def dilated_hw(self):
        (H, W), (a, b) = self.in_hw, self.input_dilation
        return (H - 1) * a + 1, (W - 1) * b + 1

    @property
    def padded_hw(self):
        t, b, l, r = self.padding
        H, W = self.dilated_hw
        return H + t + b, W + l + r

    @property
    def padded_d(self):
        return self.in_d + sum(self.pad_d)

    @property
    def kernel_extent(self):
        (kh, kw), (dh, dw) = self.kernel, self.dilation
        return (kh - 1) * dh + 1, (kw - 1) * dw + 1

    @property
    def out_hw(self):
        (H, W), (kh, kw), (sh, sw) = self.padded_hw, self.kernel_extent, self.stride
        return (H - kh) // sh + 1, (W - kw) // sw + 1

    @property
    def out_d(self):
        return (self.padded_d - self.kernel_d) // self.stride_d + 1

    @property
    def out_ch(self):
        return self.in_ch if self.pool else self.filters

    @property
    def group_ch(self):
        return self.in_ch // self.groups

    @property
    def per_group(self):
        return self.filters // self.groups

    @property
    def passes(self):
        """Kernel passes over the input: one per filter (pooling: just one)."""
        return 1 if self.pool else self.filters

    @property
    def acc_len(self):
        """Length of the multiply/accumulate axis the wave runs along."""
        return (self.group_ch if not self.pool else self.in_ch) * self.kernel_d

    def steps(self):
        """Output positions ``(i, j, k)`` in visiting order (depth outermost)."""
        Ho, Wo = self.out_hw
        out = []
        for k in range(self.out_d):
            for i in range(Ho):
                cols = range(Wo)
                if self.order == "serpentine" and i % 2:
                    cols = reversed(range(Wo))
                out.extend((i, j, k) for j in cols)
        return out

    def validate(self):
        assert self.in_ch % self.groups == 0, "in_ch must divide into groups"
        assert self.filters % self.groups == 0, "filters must divide into groups"
        assert not (self.groups > 1 and self.padded_d > 1), "grouped conv3d not supported"
        assert not (self.pool and self.groups > 1), "pooling has no groups"
        Ho, Wo = self.out_hw
        assert Ho > 0 and Wo > 0 and self.out_d > 0, "kernel larger than (padded) input"


# ------------------------------------------------------------------ reference


def expand_input(spec, data):
    """``(H, W, C[, D])`` -> ``(Hp, Wp, C, Dp)`` with inserted zeros and padding."""
    x = np.asarray(data, np.float64)
    if x.ndim == 3:
        x = x[..., None]
    H, W, Cin, D = x.shape
    a, b = spec.input_dilation
    xd = np.zeros(((H - 1) * a + 1, (W - 1) * b + 1, Cin, D))
    xd[::a, ::b] = x
    t, bo, l, r = spec.padding
    return np.pad(xd, ((t, bo), (l, r), (0, 0), tuple(spec.pad_d)))


def conv_forward(spec, data, weights=None):
    """Reference convolution / pooling (cross-correlation, like every DL
    framework).  Returns ``(Ho, Wo, out_ch, Do)``."""
    x = expand_input(spec, data)
    (kh, kw), (sh, sw), (dh, dw) = spec.kernel, spec.stride, spec.dilation
    kd, sd = spec.kernel_d, spec.stride_d
    Ho, Wo = spec.out_hw
    Do = spec.out_d
    y = np.zeros((Ho, Wo, spec.out_ch, Do))
    if not spec.pool:
        w = np.asarray(weights, np.float64)
        if w.ndim == 4:
            w = w[..., None]
    Cg, Fg = spec.group_ch, spec.per_group
    for i in range(Ho):
        for j in range(Wo):
            for k in range(Do):
                win = x[i * sh:i * sh + (kh - 1) * dh + 1:dh,
                        j * sw:j * sw + (kw - 1) * dw + 1:dw, :, k * sd:k * sd + kd]
                if spec.pool == "max":
                    y[i, j, :, k] = win.max(axis=(0, 1, 3))
                elif spec.pool == "avg":
                    y[i, j, :, k] = win.mean(axis=(0, 1, 3))
                else:
                    for f in range(spec.filters):
                        g = f // Fg
                        y[i, j, f, k] = float((win[:, :, g * Cg:(g + 1) * Cg] * w[f]).sum())
    return y


# ------------------------------------------------------------------ timeline


@dataclass
class Step:
    f: int          # filter (0 when pooling)
    i: int          # output row
    j: int          # output col
    k: int          # output depth
    t: float        # start time
    d: float        # duration

    @property
    def land(self):
        """When the result appears in the output."""
        return self.t + 0.78 * self.d


@dataclass
class Window:
    """What one step reads.

    ``acc[m]`` is the input channel index of accumulation step ``m`` (the
    kernel's channel is ``m`` itself); ``boxes`` are ``(rows, cols, chans, m0)``
    index ranges to outline/light, with ``m0`` the accumulation index at the
    back face of the box.
    """

    rows: tuple
    cols: tuple
    acc: tuple
    boxes: list


def _taps_boxes(rows, cols, ranges, spread):
    """Boxes covering ``ranges`` (chan range, m0) at the tapped rows/cols: one
    box if the taps are contiguous, one per tap otherwise."""
    if not spread:
        rr, cc = (rows[0], rows[-1] + 1), (cols[0], cols[-1] + 1)
        return [(rr, cc, ch, m0) for ch, m0 in ranges]
    return [((r, r + 1), (c, c + 1), ch, m0) for r in rows for c in cols for ch, m0 in ranges]


class ConvStage:
    """One convolution (or pooling) layer, staged."""

    def __init__(self, spec: ConvSpec, cam: Camera, t0=0.0, source: Block | None = None):
        spec.validate()
        self.spec = spec
        self.cam = cam
        self.t0 = t0
        self.last = True               # set by ConvAnimation for chained stages
        self.kernel_label = True
        Hp, Wp = spec.padded_hw
        Dp, kd = spec.padded_d, spec.kernel_d
        Ho, Wo = spec.out_hw
        Do = spec.out_d
        kh, kw = spec.kernel
        Cg = spec.group_ch
        G = spec.groups
        styles = spec.styles

        colors = spec.filter_colors or C.filter_colors(max(spec.filters, 1))
        self.colors = [C.rgb(c) for c in colors]
        gap = spec.explode if spec.explode is not None else (0.0 if G == 1 else
                                                             (0.8 if Cg == 1 else 1.0))
        self.gap = gap
        in_cols = [C.rgb(c) for c in (spec.input_colors or [C.INPUT] * G)]

        # ---- input ----------------------------------------------------------
        if source is not None:
            self.input = source
            self.owns_input = False
            self.real = None
        else:
            if Dp > 1:
                self.input = Block((Hp, Wp, spec.in_ch * Dp), group=Dp,
                                   gap=spec.slab_gap if spec.in_ch > 1 else 0.0, name="input")
            else:
                self.input = Block((Hp, Wp, spec.in_ch), group=Cg, gap=0.0, name="input")
            self.owns_input = True
            for c in range(spec.in_ch):
                self.input.base[:, :, c * Dp:(c + 1) * Dp] = in_cols[(c // Cg) % len(in_cols)]
            # real pixels vs zeros (padding and, for transposed convs, inserted)
            t, _b, l, _r = spec.padding
            a, b = spec.input_dilation
            H, W = spec.in_hw
            real = np.zeros((Hp, Wp), bool)
            real[t:t + (H - 1) * a + 1:a, l:l + (W - 1) * b + 1:b] = True
            self.real = real
            d0 = spec.pad_d[0]
            zero = np.ones((Hp, Wp, spec.in_ch * Dp), bool)
            for c in range(spec.in_ch):
                zero[:, :, c * Dp + d0:c * Dp + d0 + spec.in_d] = ~real[..., None]
            self.zero = zero
            self.input.base[zero] = C.rgb(spec.pad_color)
        self.input_home = None

        # ---- kernels ----------------------------------------------------------
        self.kernels = []
        if not spec.pool:
            for f in range(spec.filters):
                k = Block((kh, kw, Cg * kd), color=self.colors[f], group=kd,
                          gap=spec.slab_gap if (kd > 1 and Cg > 1) else 0.0,
                          pitch=spec.dilation, scale=spec.kernel_scale, name="kernel%d" % f)
                self.kernels.append(k)

        # ---- output -----------------------------------------------------------
        oc = spec.out_ch
        self.output = Block((Ho, Wo, oc * Do), group=Do,
                            gap=spec.slab_gap if (Do > 1 and oc > 1) else 0.0, name="output")
        for c in range(oc):
            col = in_cols[0] if spec.pool else self.colors[c]
            self.output.base[:, :, c * Do:(c + 1) * Do] = col

        # ---- real numbers (optional) ------------------------------------------
        self.values = None
        self.raw = None
        if spec.data is not None and (spec.weights is not None or spec.pool):
            self._apply_values(spec.data, spec.weights, in_cols, styles)

        self._layout()
        self._schedule()

    # -- values -----------------------------------------------------------------
    def _apply_values(self, data, weights, in_cols, styles):
        spec = self.spec
        Dp, Do = spec.padded_d, spec.out_d
        if self.owns_input:
            x = expand_input(spec, data)                  # (Hp, Wp, C, Dp)
            for idx in np.ndindex(x.shape):
                i, j, c, d = idx
                if self.zero[i, j, c * Dp + d]:
                    continue
                base = in_cols[(c // spec.group_ch) % len(in_cols)]
                self.input.base[i, j, c * Dp + d] = C.styled(styles.get("input", "hue"),
                                                             base, x[idx])
        if not spec.pool:
            w = np.asarray(weights, np.float64)
            if w.ndim == 4:
                w = w[..., None]
            wmax = np.abs(w).max() or 1.0
            for f, kb in enumerate(self.kernels):
                for (a, b, c, d), v in np.ndenumerate(w[f] / wmax):
                    kb.base[a, b, c * spec.kernel_d + d] = C.styled(
                        styles.get("kernel", "hue"), self.colors[f], v)
        y = conv_forward(spec, data, weights)
        if spec.bias is not None:
            y = y + np.asarray(spec.bias, np.float64)[None, None, :, None]
        if spec.activation == "relu":
            y = np.maximum(y, 0.0)
        self.raw = y
        ymax = np.abs(y).max() or 1.0
        self.values = y / ymax
        for (i, j, c, k), v in np.ndenumerate(self.values):
            base = in_cols[0] if spec.pool else self.colors[c]
            self.output.base[i, j, c * Do + k] = C.styled(styles.get("output", "hue"), base, v)

    # -- layout -----------------------------------------------------------------
    def _layout(self):
        spec, cam, L = self.spec, self.cam, self.spec.layout
        if self.owns_input:
            self.input.place(cam.frac(*L.input))
            self.input_home = self.input.origin.copy()
        self.output.place(cam.frac(*L.output))
        self.active = cam.frac(*L.active)
        lows = [self.output.origin[2]]
        if self.owns_input:
            lows.append(self.input.origin[2])

        self.queue_shift = True
        self.queue_slots, self.done_slots = [], []
        if self.kernels:
            # Dilated kernels wait compact and only spread their taps when
            # active, so the queue is spaced for the compact size.
            k0 = self.kernels[0]
            k0.pitch = (1.0, 1.0)
            ext = k0.extent
            k0.pitch = tuple(float(v) for v in spec.dilation)
            axis = L.queue_axis
            if axis == "auto":
                axis = "y" if ext[1] < ext[0] else "x"
            unit = np.array([1.0, 0, 0]) if axis == "x" else np.array([0, 1.0, 0])
            step = unit * (ext[0 if axis == "x" else 1] + L.spacing)
            q0 = cam.frac(*L.queue)
            d0 = cam.frac(*L.done)
            G, Fg = spec.groups, spec.per_group
            shift = spec.queue_shift
            if shift is None:
                shift = G == 1 or Fg == 1
            self.queue_shift = shift
            if shift:
                self.queue_slots = [q0 + step * n for n in range(spec.filters)]
            else:
                # A grid: one row per group, behind each other.
                row = np.array([0, -1.0, 0]) if axis == "x" else np.array([1.0, 0, 0])
                row = row * (ext[1 if axis == "x" else 0] + L.spacing)
                self.queue_slots = [q0 + step * (f % Fg) + row * (f // Fg)
                                    for f in range(spec.filters)]
            self.done_slots = [d0 + step * n for n in range(spec.filters)]
            lows.append(self.active[2] - ext[2] / 2)
        self.floor_z = min(lows) - L.floor_gap

        # The kernel's own boxes (for the glass on it): one per depth slab,
        # one per tap if it is dilated.
        if self.kernels:
            kb, kd = self.kernels[0], spec.kernel_d
            ranges = [((c * kd, c * kd + kd), c * kd) for c in range(spec.group_ch)] \
                if kd > 1 else [((0, spec.group_ch), 0)]
            self.kernel_boxes = _taps_boxes(tuple(range(kb.H)), tuple(range(kb.W)), ranges,
                                            spec.dilation != (1, 1))

    # -- schedule ---------------------------------------------------------------
    def _schedule(self):
        spec = self.spec
        steps = spec.steps()
        n = len(steps) * spec.passes
        durs = spec.tempo.durations(n)
        t = self.t0 + spec.intro
        if self.owns_input and spec.groups > 1:
            t += 1.2                                  # time to explode the input
        if self.owns_input and spec.input_dilation != (1, 1):
            t += 1.6                                  # time to spread the pixels apart
        self.t_start = t
        self.swaps = {}            # filter -> (t0, t1) moving into the active slot
        self.steps = []
        k = 0
        for f in range(spec.passes):
            if f > 0:
                d = float(np.clip(2.6 * durs[k], *spec.transition))
                if self.group_of(f) != self.group_of(f - 1):
                    d = max(d, spec.group_transition)
                # let the previous result land before anything moves
                t += 0.25 * self.steps[-1].d
                self.swaps[f] = (t, t + d)
                t += d
            for (i, j, kk) in steps:
                self.steps.append(Step(f, i, j, kk, t, float(durs[k])))
                t += durs[k]
                k += 1
        self.t_end = self.steps[-1].land + 0.35
        self.step_starts = [s.t for s in self.steps]

    def group_of(self, f):
        return 0 if self.spec.pool else f // self.spec.per_group

    @property
    def duration(self):
        return self.t_end - self.t0

    # -- state queries ------------------------------------------------------------
    def step_at(self, t):
        """Index of the step in progress at ``t`` (or the last one started)."""
        return bisect.bisect_right(self.step_starts, t) - 1

    def swap_progress(self, f, t):
        if f == 0:
            return 1.0
        a, b = self.swaps[f]
        return ease_in_out(remap(t, a, b))

    def active_group(self, t):
        """Continuous group index (fractional while the input slides)."""
        g = 0.0
        for f in range(1, self.spec.passes):
            ga, gb = self.group_of(f - 1), self.group_of(f)
            if gb != ga:
                g += (gb - ga) * self.swap_progress(f, t)
        return g

    def window(self, step):
        spec = self.spec
        (kh, kw), (sh, sw), (dh, dw) = spec.kernel, spec.stride, spec.dilation
        Dp, kd, sd = spec.padded_d, spec.kernel_d, spec.stride_d
        rows = tuple(step.i * sh + a * dh for a in range(kh))
        cols = tuple(step.j * sw + b * dw for b in range(kw))
        if spec.pool:
            chans = range(spec.in_ch)
        else:
            g, Cg = self.group_of(step.f), spec.group_ch
            chans = range(g * Cg, (g + 1) * Cg)
        acc = tuple(c * Dp + step.k * sd + d for c in chans for d in range(kd))
        if Dp > 1:
            ranges = [((c * Dp + step.k * sd, c * Dp + step.k * sd + kd), n * kd)
                      for n, c in enumerate(chans)]
        else:
            ranges = [((chans[0], chans[-1] + 1), 0)]
        boxes = _taps_boxes(rows, cols, ranges, (dh, dw) != (1, 1))
        return Window(rows, cols, acc, boxes)

    def out_cells(self, step):
        """Output channel indices a step writes (pooling writes every channel)."""
        Do = self.spec.out_d
        if self.spec.pool:
            return [c * Do + step.k for c in range(self.spec.in_ch)]
        return [step.f * Do + step.k]

    def kernel_visibility(self, t, reset=0.0):
        """Chained stages bring their kernels in when they start and retire
        them when they are done, so only the working layer's kernels show."""
        v = 1.0
        if self.t0 > 0:
            v *= smoothstep(self.t0 - 0.2, self.t0 + 0.5, t) * (1 - smoothstep(0.1, 0.6, reset))
        if not self.last and self.kernels:
            gone = smoothstep(self.t_end + 0.2, self.t_end + 1.0, t)
            v *= 1 - gone * (1 - smoothstep(0.4, 0.9, reset))
        return v

    def kernel_center(self, f, t, reset=0.0):
        F = self.spec.filters
        p_in = self.swap_progress(f, t)
        p_out = self.swap_progress(f + 1, t) if f + 1 < F else 0.0
        if f == 0:
            pos = self.active
        else:
            # Slot in the queue: everything ahead of us that has been taken
            # pulls us one slot closer.
            slot = (f - 1) - sum(self.swap_progress(m, t) for m in range(1, f))
            lo = int(np.floor(slot))
            frac = slot - lo
            if self.queue_shift:
                qa = self.queue_slots[max(lo, 0)]
                qb = self.queue_slots[max(lo + 1, 0)]
                qpos = lerp(qa, qb, frac)
            else:
                qpos = self.queue_slots[f]
            pos = lerp(qpos, self.active, p_in) + self._arc(p_in)
        if p_out > 0:
            pos = lerp(pos, self.done_slots[f], p_out) + self._arc(p_out)
        if reset > 0:
            # Fly home, last-finished first, so the queue refills from the back.
            home = self.active if f == 0 else self.queue_slots[f - 1 if self.queue_shift else f]
            r = self.reset_progress(f, reset)
            pos = lerp(pos, home, r) + self._arc(r, 1.5)
        return np.asarray(pos, np.float64)

    def reset_progress(self, f, reset):
        """How far kernel ``f`` is on its way home during the loop reset."""
        F = self.spec.filters
        delay = 0.5 * (F - 1 - f) / max(F - 1, 1)
        return ease_in_out(remap(reset, delay * 0.8, delay * 0.8 + 0.5))

    def kernel_spread(self, f, t, reset=0.0):
        """0 = compact, 1 = taps fully spread (dilated kernels, while active)."""
        F = self.spec.filters
        p = self.swap_progress(f, t) * (1.0 - (self.swap_progress(f + 1, t) if f + 1 < F
                                               else 0.0))
        if reset > 0:
            p = lerp(p, 1.0 if f == 0 else 0.0, self.reset_progress(f, reset))
        return p

    @staticmethod
    def _arc(p, lift=0.6):
        return np.array([0.0, 0.0, lift * 4 * p * (1 - p)])

    # -- per-frame ------------------------------------------------------------------
    def apply(self, t, scene: Scene, reset=0.0):
        """Write this stage's state for time ``t`` into its blocks and ``scene``.

        ``reset`` (0..1) rewinds everything to the opening state for looping.
        """
        spec = self.spec
        G, Cg = spec.groups, spec.group_ch

        # ---- input groups: explode, ghost, track ----------------------------
        if self.owns_input and G > 1:
            boom = ease_in_out(remap(t, self.t0 + spec.intro * 0.5,
                                     self.t0 + spec.intro * 0.5 + 0.9))
            boom *= 1.0 - ease_in_out(remap(reset, 0.45, 0.95))
            self.input.gap = self.gap * boom
            g_now = self.active_group(t) * (1.0 - ease_in_out(remap(reset, 0.3, 0.9)))
            ghost_in = ease_in_out(remap(t, self.t0 + spec.intro * 0.5 + 0.5,
                                         self.t0 + spec.intro * 0.5 + 1.1))
            ghost_in *= 1.0 - ease_in_out(remap(reset, 0.5, 1.0))
            for g in range(G):
                # Solid only near the active group; the hand-over is kept short
                # so no group lingers half-transparent.
                near = smoothstep(0.55, 0.95, 1.0 - abs(g - g_now))
                op = lerp(1.0, lerp(spec.ghost, 1.0, near), ghost_in)
                self.input.opacity[:, :, g * Cg:(g + 1) * Cg] *= op
            # Framing once exploded.  Tracking: the block slides along the
            # channel axis so the active group always sits where group 0
            # started -- the eye never has to chase it.  Otherwise the exploded
            # block stays centred on the solid one.
            home = self.input_home
            if spec.track_groups:
                y = home[1] - g_now * (Cg + self.input.gap)
            else:
                y = home[1] + (spec.in_ch - self.input.extent[1]) / 2
            self.input.origin = np.array([home[0], y, home[2]])
            scene.probe(("input", id(self)), self.input.origin)

        # ---- transposed conv: the pixels spread apart, zeros appear between --
        if self.owns_input and spec.input_dilation != (1, 1):
            spread = ease_in_out(remap(t, self.t0 + spec.intro, self.t0 + spec.intro + 1.3))
            spread *= 1.0 - ease_in_out(remap(reset, 0.35, 0.9))
            if spread < 1.0:
                self._compact(1.0 - spread)
            self.input.opacity[self.zero] *= spread
            scene.probe(("spread", id(self)), (spread * 8.0, 0.0, 0.0))

        # ---- kernels --------------------------------------------------------
        kvis = self.kernel_visibility(t, reset)
        for f, kb in enumerate(self.kernels):
            kb.opacity *= kvis
            if spec.dilation != (1, 1):
                p = self.kernel_spread(f, t, reset)
                kb.pitch = (lerp(1.0, spec.dilation[0], p), lerp(1.0, spec.dilation[1], p))
            c = self.kernel_center(f, t, reset)
            kb.place(c)
            scene.probe(("kernel", id(self), f), c)

        n = self.step_at(t)
        started = n >= 0 and t >= self.t_start
        cur = self.steps[n] if started else None

        # ---- output cells ---------------------------------------------------
        out = self.output
        out.present[...] = False
        total = len(self.steps)
        for idx, s in enumerate(self.steps[:max(n + 1, 0)]):
            if t < s.land:
                continue
            for ch in self.out_cells(s):
                out.present[s.i, s.j, ch] = True
                k = t - s.land
                out.white[s.i, s.j, ch] = 1.0 - ease_out(remap(k, 0.0, max(0.3, 1.2 * s.d)))
                if reset > 0:
                    # Dissolve in step order: each cell glints and vanishes
                    # quickly, so only a thin front of cells is ever translucent.
                    delay = 0.62 * (idx / max(total - 1, 1))
                    gone = remap(reset, delay, delay + 0.12)
                    out.white[s.i, s.j, ch] = max(out.white[s.i, s.j, ch], 0.8 * gone)
                    out.opacity[s.i, s.j, ch] = 1.0 - ease_in_out(gone)

        # ---- the receptive-field frame ----------------------------------------
        # Before the first step it waits on the first window (the first stage
        # starts with it showing, later stages fade it in); after the last step
        # it stays on the final window if this is the last stage, otherwise it
        # hands over.  The loop reset swaps it back to the first window.
        first = self.window(self.steps[0])
        if cur is None:
            op = 1.0 if self.t0 <= 0 else ease_out(remap(t, self.t_start - 0.35, self.t_start))
            if self.owns_input and spec.input_dilation != (1, 1):
                op *= smoothstep(self.t_start - 0.4, self.t_start, t)   # after the spread
            self._frame(scene, first, op * (1.0 - smoothstep(0.0, 0.3, reset)))
            if reset > 0 and self.t0 <= 0:
                self._frame(scene, first, smoothstep(0.75, 1.0, reset))
            return
        if reset > 0 or t > self.t_end:
            op = 1.0 if self.last else 1.0 - smoothstep(self.t_end, self.t_end + 0.4, t)
            self._frame(scene, self.window(cur), op * (1.0 - smoothstep(0.0, 0.3, reset)))
            if reset > 0 and self.t0 <= 0 and spec.input_dilation == (1, 1):
                self._frame(scene, first, smoothstep(0.75, 1.0, reset))
            return

        u = (t - cur.t) / cur.d
        prev = self.steps[n - 1] if n > 0 else None
        win = self.window(cur)
        nxt = self.steps[n + 1] if n + 1 < len(self.steps) else None
        if u > 1.0 and nxt is not None and nxt.f != cur.f:
            # Between filters: glide back to the next filter's first window
            # while the kernels swap.
            self._frame(scene, win, 1.0, self.window(nxt), self.swap_progress(nxt.f, t))
        elif prev is not None and prev.f == cur.f:
            self._frame(scene, self.window(prev), 1.0, win, ease_in_out(remap(u, 0.0, 0.22)))
        else:
            self._frame(scene, win, 1.0)
        self._wave(scene, cur, u, win)

    def _compact(self, amount):
        """Pull the real input pixels ``amount`` of the way back to a dense
        grid (the transposed-conv input before its zeros are inserted)."""
        spec = self.spec
        Hp, Wp = spec.padded_hw
        H, W = spec.in_hw
        ri, rj = np.nonzero(self.real)
        ci = (Hp - 1) / 2 + (np.arange(H) - (H - 1) / 2)
        cj = (Wp - 1) / 2 + (np.arange(W) - (W - 1) / 2)
        order_i = np.searchsorted(np.unique(ri), ri)
        order_j = np.searchsorted(np.unique(rj), rj)
        dx = (cj[order_j] - rj) * amount
        dz = -(ci[order_i] - ri) * amount
        if self.input.offset is None:
            self.input.offset = np.zeros(self.input.base.shape[:3] + (3,))
        self.input.offset[ri, rj, :, 0] += dx[:, None]
        self.input.offset[ri, rj, :, 2] += dz[:, None]

    def _boxes_world(self, block, win_boxes):
        return [block.box(r, c, ch) for r, c, ch, _m0 in win_boxes]

    def _frame(self, scene, win, opacity, win_to=None, m=0.0):
        """White receptive-field frame around ``win`` (optionally part-way to
        ``win_to``).  Interpolation happens in world space, so it also glides
        correctly between channel groups."""
        if opacity <= 0.003:
            return
        boxes = self._boxes_world(self.input, win.boxes)
        if win_to is not None and m > 0:
            to = self._boxes_world(self.input, win_to.boxes)
            if len(to) == len(boxes):
                boxes = [(lerp(a, c, m), lerp(b, d, m)) for (a, b), (c, d) in zip(boxes, to)]
            elif m > 0.5:
                boxes = to
        width = 1.5 if len(boxes) <= 2 else 1.1
        for lo, hi in boxes:
            scene.wire_box(lo, hi, inflate=0.045, color=(0.93, 0.93, 0.93), width=width,
                           opacity=0.95 * opacity, bias=0.10)
        scene.probe(("frame", id(self)), boxes[0][0], tol=4.0)

    def _wave(self, scene, step, u, win):
        """The white multiply/accumulate pass: a soft band that runs through
        the window and the kernel along the accumulation axis (channels, then
        depth), from the back to the front."""
        L = self.spec.acc_len
        lead = ease_in_out(remap(u, 0.04, 0.56))
        trail = ease_in_out(remap(u, 0.30, 0.80))
        soft = 0.6
        span = L + 2 * soft
        lead_m = -soft + lead * span
        trail_m = -soft + trail * span
        rows = np.array(win.rows)[:, None]
        cols = np.array(win.cols)[None, :]
        kb = self.kernels[step.f] if self.kernels else None
        for m in range(L):
            w = soft_band(m + 0.5, trail_m, lead_m, soft)
            if w <= 0.002:
                continue
            ch = win.acc[m]
            self.input.white[rows, cols, ch] = np.maximum(self.input.white[rows, cols, ch], w)
            if kb is not None:
                kb.white[:, :, m] = np.maximum(kb.white[:, :, m], w)
        # A final glint on the kernel's front face as the result is produced.
        glint = soft_band(u, 0.62, 0.80, 0.08)
        if glint > 0 and kb is not None:
            kb.glow[:, :, L - 1] = np.maximum(kb.glow[:, :, L - 1], 0.35 * glint)
        # The band is fast but soft; a coarse tolerance keeps sub-frames cheap.
        scene.probe(("wave", id(self)), (lead_m, trail_m, 0.0), tol=64.0)

        if lead_m - trail_m <= 0.05:
            return
        view = self.cam.view
        targets = [(self.input, win.boxes)]
        if kb is not None:
            targets.append((kb, self.kernel_boxes))
        for block, boxes in targets:
            for rr, cc, kk, m0 in boxes:
                lo, hi = block.box(rr, cc, kk)
                y0 = block.origin[1] + block.chan_y(kk[0])

                def shader(pts, n, y0=y0, m0=m0):
                    ml = pts[..., 1] - y0 + m0
                    a = np.clip((ml - trail_m) / soft + 0.5, 0, 1) * \
                        np.clip((lead_m - ml) / soft + 0.5, 0, 1)
                    a = a * a * (3 - 2 * a)
                    rgb = np.ones(pts.shape[:-1] + (3,), np.float32) * 0.985
                    return rgb, (0.80 * a).astype(np.float32), 0.9   # emits in HDR

                scene.glass_box(lo, hi, view, shader, inflate=0.05, bias=0.12)

    # -- labels ---------------------------------------------------------------------
    def labels(self):
        """``[(lo, hi, text)]``: world boxes to caption, and the captions."""
        spec = self.spec
        want = spec.labels
        if want is False:
            return []
        want = want if isinstance(want, dict) else {}
        x = "×"

        def shape(h, w, c, d):
            dims = [h, w] + ([d] if d > 1 else [])
            s = x.join(str(v) for v in dims)
            return s + ("  ·  %d ch" % c if d > 1 else x + str(c))

        H, W = spec.in_hw
        Ho, Wo = spec.out_hw
        out = []
        if self.owns_input and want.get("input", True):
            lo, hi = self.input.origin, self.input.origin + self.input.extent
            text = want.get("input") if isinstance(want.get("input"), str) else \
                "input  " + shape(H, W, spec.in_ch, spec.in_d)
            out.append((lo, hi, text))
        if self.kernels and want.get("kernel", True) and self.kernel_label:
            ext = self.kernels[0].extent
            kh, kw = spec.kernel
            text = "%d%s%d%s%d kernel" % (kh, x, kw, x, spec.group_ch)
            if spec.kernel_d > 1:
                text = "%d%s%d%s%d kernel  ·  %d ch" % (spec.kernel_d, x, kh, x, kw,
                                                            spec.group_ch)
            if spec.filters > 1:
                text += "  ·  %d filters" % spec.filters
            if spec.dilation != (1, 1):
                text += "  ·  dilation %d" % spec.dilation[0]
            text = want.get("kernel") if isinstance(want.get("kernel"), str) else text
            out.append((self.active - ext / 2, self.active + ext / 2, text))
        if spec.pool and want.get("kernel", True):
            kh, kw = spec.kernel
            text = "%d%s%d %s, every channel" % (kh, x, kw, spec.pool)
            out.append((self.active, self.active, text))
        if want.get("output", True):
            lo, hi = self.output.origin, self.output.origin + self.output.extent
            text = want.get("output") if isinstance(want.get("output"), str) else \
                "output  " + shape(Ho, Wo, spec.out_ch, spec.out_d)
            out.append((lo, hi, text))
        return out

    def emit(self, scene, view, mview):
        if self.owns_input:
            self.input.emit(scene, view, mview)
        for kb in self.kernels:
            kb.emit(scene, view, mview)
        self.output.emit(scene, view, mview)
        # One cage per output slab when they are spread apart, else one cage.
        o, Do = self.output, self.spec.out_d
        slabs = [(c * Do, (c + 1) * Do) for c in range(self.spec.out_ch)] if o.gap > 0 \
            else [(0, o.C)]
        for k0, k1 in slabs:
            lo, hi = o.box((0, o.H), (0, o.W), (k0, k1))
            scene.wire_box(lo, hi, inflate=0.04, color=C.CAGE, width=1.45, opacity=0.95,
                           bias=0.0)

    def reset_blocks(self):
        if self.owns_input:
            self.input.reset()
        for kb in self.kernels:
            kb.reset()
        self.output.reset()


# ------------------------------------------------------------------ combine


@dataclass
class CombineSpec:
    """A cell-wise merge of stage outputs, e.g. ``|G| = sqrt(Gx^2 + Gy^2)``.

    It sweeps row by row: each row of every source lights up, and the merged
    row lands in a new block.
    """

    sources: tuple = (0, 1)            # stage indices (their outputs must match)
    op: str = "magnitude"              # "magnitude", "sum", "max"
    at: tuple = (0.80, 0.60)           # screen position of the result
    style: str = "gray"
    color: tuple = tuple(C.INPUT.tolist())
    row_time: float = 0.28
    label: str | None = None


class CombineStage:
    def __init__(self, spec: CombineSpec, cam: Camera, t0, stages):
        self.spec = spec
        self.cam = cam
        self.t0 = t0
        self.last = True
        self.srcs = [stages[i].output for i in spec.sources]
        raws = [stages[i].raw for i in spec.sources]
        assert all(r is not None for r in raws), "combine needs stages with data"
        v = np.stack([r[..., 0, 0] if r.shape[2] == 1 else r.sum(axis=(2, 3)) for r in raws])
        if spec.op == "magnitude":
            y = np.sqrt((v ** 2).sum(0))
        elif spec.op == "max":
            y = v.max(0)
        else:
            y = v.sum(0)
        ymax = np.abs(y).max() or 1.0
        self.values = y / ymax
        H, W = y.shape
        self.output = Block((H, W, 1), name="combined")
        for (i, j), val in np.ndenumerate(self.values):
            self.output.base[i, j, 0] = C.styled(spec.style, C.rgb(spec.color), val)
        self.output.place(cam.frac(*spec.at))
        self.t_start = t0 + 0.4
        self.rows = H
        self.t_end = self.t_start + H * spec.row_time + 0.4
        self.floor_z = self.output.origin[2] - 0.7

    def reset_blocks(self):
        self.output.reset()

    def apply(self, t, scene, reset=0.0):
        sp = self.spec
        out = self.output
        out.present[...] = False
        for r in range(self.rows):
            land = self.t_start + (r + 0.8) * sp.row_time
            if t >= land:
                out.present[r] = True
                out.white[r] = 1.0 - ease_out(remap(t - land, 0.0, 0.4))
                if reset > 0:
                    gone = remap(reset, 0.5 * r / self.rows, 0.5 * r / self.rows + 0.15)
                    out.opacity[r] = 1.0 - ease_in_out(gone)
        if reset > 0 or t < self.t_start or t > self.t_end:
            return
        u = (t - self.t_start) / sp.row_time
        r = int(np.clip(np.floor(u), 0, self.rows - 1))
        w = soft_band(u - r, 0.05, 0.75, 0.15)
        for b in self.srcs + [out]:
            if b is not out:
                b.white[r] = np.maximum(b.white[r], 0.75 * w)
            lo, hi = b.box((r, r + 1), (0, b.W), (0, b.C))
            scene.wire_box(lo, hi, inflate=0.045, color=(0.93, 0.93, 0.93), width=1.2,
                           opacity=0.9, bias=0.1)
        scene.probe(("combine", id(self)), out.cell_center(r, 0, 0), tol=4.0)

    def labels(self):
        if self.spec.label is None:
            return []
        o = self.output
        return [(o.origin, o.origin + o.extent, self.spec.label)]

    def emit(self, scene, view, mview):
        self.output.emit(scene, view, mview)
        lo, hi = self.output.box((0, self.output.H), (0, self.output.W), (0, 1))
        scene.wire_box(lo, hi, inflate=0.04, color=C.CAGE, width=1.45, opacity=0.95, bias=0.0)


# ------------------------------------------------------------------ classifier head


@dataclass
class HeadSpec:
    """A dense layer + softmax on the last stage's output: the class scores,
    the probabilities, and the one-hot prediction.

    ``W`` is ``(H * W * C, classes)`` for the output flattened row-major as
    ``(row, col, channel)`` -- numpy / Keras order.
    """

    W: np.ndarray
    b: np.ndarray
    at: tuple = (0.86, 0.56)           # screen position of the logits row
    scale: float = 1.3
    label: int | None = None           # the true class, if known
    title: str = "dense"
    particles: int = 90


class HeadStage:
    def __init__(self, spec: HeadSpec, cam: Camera, t0, source_stage):
        self.spec, self.cam, self.t0, self.last = spec, cam, t0, True
        src = source_stage
        self.src = src.output
        vals = src.raw[..., 0] if src.raw.ndim == 4 else src.raw   # (H, W, C)
        flat = vals.reshape(-1)
        self.logits = flat @ spec.W + spec.b
        e = np.exp(self.logits - self.logits.max())
        self.p = e / e.sum()
        self.pred = int(self.p.argmax())
        n = len(self.p)
        sc = spec.scale
        pitch = (1, 1.75)                       # spaced columns: room for every label
        self.row = Block((1, n, 1), scale=sc, pitch=pitch,
                         name="logits").place(cam.frac(*spec.at))
        lo = self.row.origin
        self.bars = Block((10, n, 1), scale=sc, pitch=pitch, name="bars")
        self.bars.origin = lo + np.array([0.0, 0.0, 3.6 * sc])
        self.hot = Block((1, n, 1), scale=sc, pitch=pitch, name="onehot")
        self.hot.origin = lo - np.array([0.0, 0.0, 2.4 * sc])
        lmax = np.abs(self.logits).max() or 1.0
        for j in range(n):
            self.row.base[0, j, 0] = C.diverging(self.logits[j] / lmax)
        self.heights = np.clip(np.round(self.p * 10).astype(int), 0, 10)
        self.heights[self.p > 0.02] = np.maximum(self.heights[self.p > 0.02], 1)
        # the strongest contributions, for the particles
        H, W_, Cc = vals.shape
        contrib = flat[:, None] * spec.W
        k = min(spec.particles, int((np.abs(contrib) > 0).sum()))
        idx = np.argpartition(-np.abs(contrib).ravel(), k - 1)[:k]
        cmax = np.abs(contrib).max() or 1.0
        self.parts = []
        for m in idx:
            i, j = divmod(int(m), n)
            r, c, ch = np.unravel_index(i, (H, W_, Cc))
            start = 0.55 * ((m * 2654435761) % 1000) / 1000.0
            self.parts.append(((int(r), int(c), int(ch)), j, contrib[i, j] / cmax, start))
        Tm, t = {}, t0 + 0.2
        for name, d in (("dense", 2.4), ("softmax", 1.4), ("hot", 0.8)):
            Tm[name] = (t, t + d)
            t += d + 0.25
        self.Tm = Tm
        self.t_end = t + 0.6
        self.floor_z = self.hot.origin[2] - 0.8

    def p_(self, name, t):
        return remap(t, *self.Tm[name])

    def reset_blocks(self):
        for b in (self.row, self.bars, self.hot):
            b.reset()

    def apply(self, t, scene, reset=0.0):
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.6))
        n = len(self.p)
        dn, sm, ht = self.p_("dense", t), self.p_("softmax", t), self.p_("hot", t)
        self.row.present[0, :, 0] = np.arange(n) < dn * (n + 2) - 1
        self.row.opacity *= back
        # dense: the strongest weighted inputs fly to their class
        if 0 < dn < 1:
            for (r, c, ch), j, v, start in self.parts:
                u = remap(dn, start, start + 0.4)
                if 0 < u < 1:
                    a = self.src.cell_center(r, c, ch)
                    b = self.row.cell_center(0, j, 0)
                    pos = a + (b - a) * ease_in_out(u) + np.array([0, 0, 2.5 * 4 * u * (1 - u)])
                    scene.voxel(pos, C.POSITIVE if v > 0 else C.NEGATIVE, 0.35 + 0.65 * abs(v),
                                glow=0.35, shadow=False, reflect=False, size=0.5)
            scene.probe(("head",), (dn * 40, 0, 0), tol=8.0)
        # softmax: bars grow, the winner glows
        grow = ease_out(sm) * 10
        for j in range(n):
            h = int(min(self.heights[j], np.ceil(grow)))
            self.bars.present[:, j, 0] = np.arange(10) >= 10 - h
            win = j == self.pred
            self.bars.base[:, j, 0] = C.rgb("#c9fff6") if win else C.rgb("#2f8a86")
            self.bars.glow[:, j, 0] = 0.3 * win
        self.bars.opacity *= back
        # one-hot: the prediction
        hot = np.arange(n) == self.pred
        self.hot.base[0, :, 0] = np.where(hot[:, None], C.rgb("#fbfbf4"), C.rgb("#2b2f33"))
        self.hot.glow[0, :, 0] = 0.35 * hot
        self.hot.present[0, :, 0] = ht > 0
        self.hot.opacity *= ease_out(ht) * back
        if self.hot.offset is None:
            self.hot.offset = np.zeros(self.hot.base.shape[:3] + (3,))
        self.hot.offset[0, hot, 0, 1] += 0.35 * self.spec.scale * ease_out(ht)
        # numbers
        cam = self.cam
        op = back
        digits_row = self.hot if ht > 0 else self.row
        for j in range(n):
            if sm > 0 and self.heights[j] > 0:
                top = self.bars.cell_center(10 - self.heights[j], j, 0)
                q = cam.project(top + np.array([0, 0, 0.7 * self.spec.scale]))
                scene.text(q[0] / cam.width, q[1] / cam.height, "%.2f" % self.p[j], 0.029,
                           "regular", layout.INK, min(1, sm * 2) * op, "mb")
            if dn > 0:
                q = cam.project(digits_row.cell_center(0, j, 0) -
                                np.array([0, 0, 1.3 * self.spec.scale]))
                scene.text(q[0] / cam.width, q[1] / cam.height, str(j), 0.029, "regular",
                           layout.CAPTION_INK, min(1, dn * 3) * op, "mt")
        if dn > 0:
            x, y = self._screen_box(self.bars, top=True)
            scene.text(x, y - 0.045, "%s %s 10, softmax" % (self.spec.title, X), 0.026,
                       "regular", layout.CAPTION_INK, min(1, dn * 3) * op, "mb")
        if ht > 0:
            x, y = self._screen_box(self.hot, top=False)
            q = (x * cam.width, (y + 0.065) * cam.height)
            ok = "" if self.spec.label is None else (
                ", correct" if self.spec.label == self.pred else
                ", true label %d" % self.spec.label)
            for n, line in enumerate(("one-hot: %d" % self.pred,
                                      "p = %.2f%s" % (self.p[self.pred], ok))):
                scene.text(q[0] / cam.width, q[1] / cam.height + n * 0.045, line, 0.031,
                           "regular", layout.STRONG_INK if n == 0 else layout.CAPTION_INK,
                           ease_out(ht) * op, "mt")

    def _screen_box(self, block, top):
        """Centre x and top (or bottom) y of a block on screen, as fractions."""
        lo, hi = block.origin, block.origin + block.extent
        c = np.array([[a, b, d] for a in (lo[0], hi[0]) for b in (lo[1], hi[1])
                      for d in (lo[2], hi[2])])
        s = self.cam.project(c)
        y = s[:, 1].min() if top else s[:, 1].max()
        return (s[:, 0].min() + s[:, 0].max()) / 2 / self.cam.width, y / self.cam.height

    def labels(self):
        return []                     # the head captions itself as it appears

    def emit(self, scene, view, mview):
        for b in (self.row, self.bars, self.hot):
            b.emit(scene, view, mview)
        lo, hi = self.row.box((0, 1), (0, self.row.W), (0, 1))
        scene.wire_box(lo, hi, inflate=0.05, color=C.CAGE, width=1.3, opacity=0.9, bias=0.0)


# ------------------------------------------------------------------ animation


class ConvAnimation:
    """One or more chained stages, a hold, and an optional loop reset."""

    def __init__(self, specs, camera=None, hold=1.6, reset=1.4, title=None,
                 subtitle=None, labels=False, combine=None, head=None):
        specs = specs if isinstance(specs, (list, tuple)) else [specs]
        L = specs[0].layout
        cam = camera or Camera(ppu=L.ppu * 540.0, azimuth=L.azimuth, elevation=L.elevation,
                               zoom=layout.ZOOM)
        self.cam = cam
        self.stages = []
        t = 0.0
        for spec in specs:
            src = None
            if self.stages:
                src = self.stages[0].input if spec.source == "first" else self.stages[-1].output
                self.stages[-1].last = False
            st = ConvStage(spec, cam, t0=t, source=src)
            self.stages.append(st)
            t = st.t_end + 0.3
        for cs in (combine or []):
            self.stages[-1].last = False
            st = CombineStage(cs, cam, t, self.stages)
            self.stages.append(st)
            t = st.t_end + 0.3
        if head is not None:
            self.stages[-1].last = False
            st = HeadStage(head, cam, t, self.stages[-1])
            self.stages.append(st)
            t = st.t_end + 0.3
        self.t_done = self.stages[-1].t_end
        self.hold = hold
        self.reset = reset
        self.duration = self.t_done + hold + reset
        self.floor_z = min(s.floor_z for s in self.stages)
        self.title = title
        self.subtitle = subtitle
        self.show_labels = labels
        self._labels = None

    def _label_texts(self):
        """Captions, placed just under each captioned box (computed once)."""
        if self._labels is None:
            self._labels = []
            cam = self.cam
            for st in self.stages:
                for lo, hi, text in st.labels():
                    kind = "kernel" if hasattr(st, "kernels") and st.kernels and \
                        np.allclose((lo + hi) / 2, st.active) else "other"
                    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                    s = cam.project(corners)
                    self._labels.append(((s[:, 0].min() + s[:, 0].max()) / 2 / cam.width,
                                         (s[:, 1].max() + 10) / cam.height, text, st, kind))
        return self._labels

    def scene(self, t):
        t = float(np.clip(t, 0.0, self.duration))
        r = remap(t, self.t_done + self.hold, self.duration) if self.reset > 0 else 0.0
        sc = Scene(floor_z=self.floor_z)
        for st in self.stages:
            st.reset_blocks()
        for st in self.stages:
            st.apply(t, sc, reset=r)
        view = self.cam.view
        mview = self.cam.mirrored().view
        for st in self.stages:
            st.emit(sc, view, mview)
        layout.title_block(sc, self.title, self.subtitle)
        if self.show_labels:
            n0 = len(sc.texts)
            for x, y, text, st, kind in self._label_texts():
                op = st.kernel_visibility(t, r) if kind == "kernel" else 1.0
                if op > 0.01:
                    sc.text(x, y, text, 0.029, "regular", layout.CAPTION_INK, op, "mt")
            layout.yield_to_geometry(sc, self.cam, sc.texts[n0:])
        return sc

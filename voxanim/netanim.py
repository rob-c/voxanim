"""The framework every network animation is built on.

* **Colours** for values: ``lit_colors`` (a value > 0 lights up, zero is
  dark: ReLU and pixel ink), ``div_colors`` (signed: orange +, blue -),
  ``gray_colors`` (a plain greyscale image).
* :class:`Sheet` -- a vector drawn as a sheet of voxels facing the camera.
* :func:`particle` -- a glowing voxel carrying a value along a curve.
* :class:`NetAnimation` -- camera, sheets, captions, title, step indicator
  and the loop reset; a subclass only implements ``draw(scene, t, reset)``.
* :class:`OutputPanel` -- logits, softmax bars, one-hot target and loss.

Every animation is a pure function of time: ``scene(t)`` builds a fresh
:class:`~voxanim.scene.Scene` and nothing carries over between frames.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import layout
from .camera import Camera
from .easing import ease_in_out, ease_out, remap, smoothstep
from .scene import Scene
from .tensor import Block

OFF = C.rgb("#2b2f33")                 # a zero: not lit
INK_LO, INK_HI = C.rgb("#9a9a92"), C.rgb("#fbfbf4")      # lit input pixels
ACT_LO, ACT_HI = C.rgb("#2f8a86"), C.rgb("#c9fff6")      # lit neurons
GRAY_LO, GRAY_HI = C.rgb("#151617"), C.rgb("#f0f0ec")    # plain greyscale image


# ------------------------------------------------------------------ colours


def lit_colors(v, vmax, lo=ACT_LO, hi=ACT_HI):
    """(n,) values -> (n, 3) colours: > 0 lit on a ramp, else OFF."""
    v = np.asarray(v, np.float64)
    a = np.clip(v / (vmax or 1.0), 0.0, 1.0) ** 0.6
    out = lo[None] + (hi - lo)[None] * a[:, None]
    out[v <= 0] = OFF
    return out.astype(np.float32)


def div_colors(v, vmax, mask=None):
    """Signed values on the diverging scale; ``mask`` False cells are OFF."""
    v = np.clip(np.asarray(v, np.float64) / (vmax or 1.0), -1.0, 1.0)
    a = np.abs(v) ** 0.75
    tgt = np.where(v[:, None] >= 0, C.POSITIVE[None], C.NEGATIVE[None])
    out = C.NEUTRAL[None] + (tgt - C.NEUTRAL[None]) * a[:, None]
    if mask is not None:
        out[~mask] = OFF
    return out.astype(np.float32)


def gray_colors(v):
    v = np.clip(np.asarray(v, np.float64), 0.0, 1.0)
    return (GRAY_LO[None] + (GRAY_HI - GRAY_LO)[None] * v[:, None]).astype(np.float32)


def amax(v):
    m = float(np.abs(v).max())
    return m if m > 0 else 1.0


# ------------------------------------------------------------------ sheets


class Sheet:
    """A vector drawn as an ``h x w`` sheet of voxels (row-major)."""

    def __init__(self, cam, shape, at, scale=1.0, name="", pitch=(1, 1)):
        self.h, self.w = shape
        self.n = self.h * self.w
        self.block = Block((self.h, self.w, 1), scale=scale, pitch=pitch, name=name)
        self.block.place(cam.frac(*at))
        self.visible = True
        self.cage = False

    def reset(self):
        self.block.reset()
        self.visible = True
        self.cage = False

    def fill(self, colors, glow=None):
        self.block.base[:, :, 0] = np.asarray(colors).reshape(self.h, self.w, 3)
        if glow is not None:
            self.block.glow[:, :, 0] = np.asarray(glow).reshape(self.h, self.w)

    def light(self, values, amount=1.0):
        """Pop cells with value > 0 towards the viewer and make them glow."""
        on = (np.asarray(values) > 0).reshape(self.h, self.w)
        if not on.any() or amount <= 0:
            return
        if self.block.offset is None:
            self.block.offset = np.zeros(self.block.base.shape[:3] + (3,))
        self.block.offset[on, 0, 1] += 0.32 * self.block.scale * amount
        self.block.glow[:, :, 0] += np.where(on, 0.10 * amount, 0.0)

    def cell(self, n):
        r, c = divmod(int(n), self.w)
        return self.block.cell_center(r, c, 0)

    def show(self, mask):
        self.block.present[:, :, 0] = np.asarray(mask).reshape(self.h, self.w)

    def fade(self, opacity):
        self.block.opacity *= opacity

    def bbox(self, offsets=True):
        """World box of the sheet; with ``offsets`` it includes cells that were
        pushed about (popped, split into patches) -- what captions avoid."""
        b = self.block
        lo, hi = b.origin, b.origin + b.extent
        if offsets and b.offset is not None:
            lo, hi = lo + b.offset.min(axis=(0, 1, 2)), hi + b.offset.max(axis=(0, 1, 2))
        return lo, hi

    @property
    def center(self):
        return self.block.center

    def emit(self, scene, view, mview):
        if not self.visible:
            return
        self.block.emit(scene, view, mview)
        if self.cage:
            lo, hi = self.bbox(offsets=False)
            scene.wire_box(lo, hi, inflate=0.06 * self.block.scale, color=C.CAGE,
                           width=1.3, opacity=0.9 * self.cage, bias=0.0)


def bezier(a, b, u, lift):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ctrl = (a + b) / 2 + np.array([0.0, -0.3 * lift, lift])
    return (1 - u) ** 2 * a + 2 * (1 - u) * u * ctrl + u ** 2 * b


def particle(scene, p0, p1, u, color, size, opacity=1.0, lift=2.0):
    """A small glowing voxel carrying a value from ``p0`` to ``p1``."""
    if not 0.0 < u < 1.0 or opacity <= 0.01:
        return
    pos = bezier(p0, p1, ease_in_out(u), lift)
    scene.voxel(pos, color, opacity, glow=0.35, shadow=False, reflect=False, size=size)


def hash01(n, salt=0):
    """Deterministic pseudo-random number in [0, 1) for staggering."""
    return ((n * 2654435761 + salt * 40503) % 4294967296) / 4294967296.0


# ------------------------------------------------------------------ base


class NetAnimation:
    """Shared plumbing: camera, sheets, captions, titles, the loop reset."""

    hold = 2.0
    reset = 1.0

    def __init__(self, title, subtitle, ppu, azimuth=33.0, elevation=22.0):
        self.cam = Camera(ppu=ppu, azimuth=azimuth, elevation=elevation, zoom=layout.ZOOM)
        self.title, self.subtitle = title, subtitle
        self.sheets = []
        self.blocks_extra = []          # plain Blocks owned by the animation (reset, emitted)
        self._cap = 1.0

    def sheet(self, shape, at, scale=1.0, name="", **kw):
        s = Sheet(self.cam, shape, at, scale, name, **kw)
        self.sheets.append(s)
        return s

    def finish(self, t_done):
        self.t_done = t_done
        self.duration = t_done + self.hold + self.reset
        self.floor_z = min([s.block.origin[2] for s in self.sheets] +
                           [b.origin[2] for b in self.blocks_extra]) - 1.2

    def caption(self, scene, sheet_or_box, text, opacity=1.0, size=0.029, above=False,
                offset=0.0, max_width=0.30, ink=None):
        opacity *= self._cap
        size = max(size, layout.MIN_TEXT)
        if opacity <= 0.01 or not text:
            return
        lo, hi = sheet_or_box.bbox() if isinstance(sheet_or_box, Sheet) else sheet_or_box
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        s = self.cam.project(corners)
        x = (s[:, 0].min() + s[:, 0].max()) / 2 / self.cam.width
        lines = [ln.strip() for part in text.split("\n")
                 for ln in layout.wrap(part.strip(), size, max_width)]
        step = size * 1.3
        for n, line in enumerate(lines):
            if above:
                y = (s[:, 1].min() - 8) / self.cam.height - offset - (len(lines) - 1 - n) * step
                anchor = "mb"
            else:
                y = (s[:, 1].max() + 10) / self.cam.height + offset + n * step
                anchor = "mt"
            scene.text(x, y, line, size, "regular", ink or layout.CAPTION_INK, opacity,
                       anchor)

    def caption_side(self, scene, sheet, text, opacity=1.0, size=0.027):
        """Caption to the right of a (wide, flat) sheet."""
        opacity *= self._cap
        size = max(size, layout.MIN_TEXT)
        if opacity <= 0.01:
            return
        lo, hi = sheet.bbox() if isinstance(sheet, Sheet) else sheet
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        s = self.cam.project(corners)
        scene.text((s[:, 0].max() + 12) / self.cam.width,
                   (s[:, 1].min() + s[:, 1].max()) / 2 / self.cam.height, text, size,
                   "regular", layout.CAPTION_INK, opacity, "lm")

    def step_indicator(self, sc, current, phases):
        self._indicator_left = layout.step_indicator(sc, current, phases)

    def caption_left(self, scene, sheet, text, opacity=1.0, size=0.027):
        """Caption to the left of a sheet, vertically centred."""
        opacity *= self._cap
        size = max(size, layout.MIN_TEXT)
        if opacity <= 0.01:
            return
        lo, hi = sheet.bbox()
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                            for z in (lo[2], hi[2])])
        s = self.cam.project(corners)
        scene.text((s[:, 0].min() - 10) / self.cam.width,
                   (s[:, 1].min() + s[:, 1].max()) / 2 / self.cam.height, text, size,
                   "regular", layout.CAPTION_INK, opacity, "rm")

    def scene(self, t):
        t = float(np.clip(t, 0.0, self.duration))
        r = remap(t, self.t_done + self.hold, self.duration)
        sc = Scene(floor_z=self.floor_z)
        for s in self.sheets:
            s.reset()
        for b in self.blocks_extra:
            b.reset()
        # Captions fade in at the start and out in the reset, so the loop's
        # last frame matches its first.
        self._cap = smoothstep(0.0, 0.5, t) * (1.0 - smoothstep(0.0, 0.5, r))
        self._indicator_left = None
        self.draw(sc, t, r)
        view, mview = self.cam.view, self.cam.mirrored().view
        for s in self.sheets:
            s.emit(sc, view, mview)
        for b in self.blocks_extra:
            b.emit(sc, view, mview)
        layout.yield_to_geometry(sc, self.cam, list(sc.texts))
        layout.title_block(sc, self.title, self.subtitle, right=self._indicator_left)
        return sc

    def draw(self, sc, t, reset):
        raise NotImplementedError


# ------------------------------------------------------------------ output panel


class OutputPanel:
    """Logits, softmax bars, the one-hot row and the result, stacked:

        bars, p over each       softmax p(digit)
        logits row              logits z
        one-hot row             target y (one-hot)
        0 1 2 ... 9             (digit labels)
        prediction / loss

    Columns are spaced ``PITCH`` cells apart so every label has room.
    """

    PITCH = 1.75
    COLUMN_PX = 14.0            # on-screen column spacing: room for a digit label

    def __init__(self, anim, tr, at, scale):
        self.anim, self.tr = anim, tr
        cam = anim.cam
        dx = abs(cam.M[0, 0]) * 960.0 / cam.width          # ref px per unit along x
        pitch = (1, max(self.PITCH, self.COLUMN_PX / (dx * scale)))
        self.logits = anim.sheet((1, 10), at, scale, "logits", pitch=pitch)
        lo = self.logits.block.origin
        self.bars = Block((10, 10, 1), scale=scale, pitch=pitch, name="bars")
        self.bars.origin = lo + np.array([0.0, 0.0, 3.6 * scale])
        self.onehot = anim.sheet((1, 10), at, scale, "onehot", pitch=pitch)
        # far enough below that the two side captions get a clear line each
        # (>= 30 reference px between the rows, whatever the scale)
        px_per_unit = abs(anim.cam.M[1, 2]) * 960.0 / anim.cam.width
        drop = max(2.4 * scale, 30.0 / px_per_unit)
        self.onehot.block.origin = lo - np.array([0.0, 0.0, drop])
        self.set_probs(tr.p, tr.loss)
        self._fit_safe_area()

    SIDE_TEXTS = ("softmax p(digit)", "logits z", "target y (one-hot)")

    def _fit_safe_area(self):
        """Slide the panel left until its right-hand captions end inside the
        title-safe area (they are the widest thing in it)."""
        cam = self.anim.cam
        blocks = (self.bars, self.logits.block, self.onehot.block)
        right = 0.0
        for b in blocks:
            lo, hi = b.origin, b.origin + b.extent
            c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                          for z in (lo[2], hi[2])])
            right = max(right, cam.project(c)[:, 0].max())
        text = max(layout.text_width(s, layout.MIN_TEXT) for s in self.SIDE_TEXTS)
        over = (right + 12) / cam.width + text - (1 - layout.SAFE) + 0.005
        if over <= 0:
            return
        z = self.logits.block.origin[2]
        sx, sy = cam.project(self.logits.block.origin)
        delta = cam.unproject(sx - over * cam.width, sy, z) - cam.unproject(sx, sy, z)
        for b in blocks:
            b.origin = b.origin + delta

    def note(self, sc, text, opacity):
        """A secondary line under the result (e.g. the values before a step)."""
        n = sum(len(layout.wrap(part, 0.031, 0.30)) for part in self.result_text().split("\n"))
        self.anim.caption(sc, self.onehot, text, opacity, size=0.029,
                          offset=0.052 + 0.045 * n + 0.012, max_width=0.5, ink=layout.NOTE_INK)

    def set_probs(self, p, loss):
        """What the bars and the loss read (they change after a training step)."""
        self.p, self.loss = np.asarray(p), float(loss)
        self.heights = np.clip(np.round(self.p * 10).astype(int), 0, 10)
        visible = self.p > 0.02
        self.heights[visible] = np.maximum(self.heights[visible], 1)

    def draw(self, sc, t_bars, t_hot, t_loss, logit_mask=None, logit_colors=None, fade=1.0):
        """Progress values in [0, 1] for each part (0 hides it)."""
        if logit_colors is not None:
            self.logits.fill(logit_colors)
            if logit_mask is not None:
                self.logits.show(logit_mask)
            self.logits.cage = 1.0
        else:
            self.logits.visible = False
        self.logits.fade(fade)
        b = self.bars
        b.reset()
        b.present[...] = False
        if t_bars > 0:
            grow = ease_out(t_bars) * 10
            for c in range(10):
                n = int(min(self.heights[c], np.ceil(grow)))
                b.present[10 - n:, c, 0] = True
                pred = c == int(self.p.argmax())
                b.base[:, c, 0] = ACT_HI if pred else ACT_LO
                b.glow[:, c, 0] = 0.25 if pred else 0.0
            b.opacity *= fade
        if t_hot > 0:
            hot = np.arange(10) == self.hot_index
            self.onehot.fill(np.where(hot[:, None], INK_HI, OFF), glow=np.where(hot, 0.3, 0.0))
            self.onehot.light(hot, ease_out(t_hot))
            self.onehot.fade(ease_out(t_hot) * fade)
            self.onehot.cage = ease_out(t_hot) * fade
        else:
            self.onehot.visible = False

    @property
    def hot_index(self):
        """Which digit the one-hot row marks (the label; a prediction panel
        overrides this)."""
        return self.tr.label

    def emit(self, sc):
        self.bars.emit(sc, self.anim.cam.view, self.anim.cam.mirrored().view)

    def result_text(self):
        pred = int(self.p.argmax())
        return "prediction %d (p = %.2f)\nloss = -log p[%d] = %.2f" % (
            pred, self.p[pred], self.tr.label, self.loss)

    def captions(self, sc, t_bars, t_hot, t_loss, logits_text="logits z",
                 hot_text="target y (one-hot)"):
        a = self.anim
        h = 0.5 * self.bars.scale
        if self.logits.visible:
            a.caption_side(sc, self.logits, logits_text)
        if t_bars > 0:
            op = min(1.0, t_bars * 2)
            for c in np.nonzero(self.heights)[0]:
                p = self.bars.cell_center(10 - self.heights[c], c, 0)
                a.caption(sc, (p - h, p + h), "%.2f" % self.p[c], op, size=0.029, above=True)
            mid = self.bars.cell_center(9, 9, 0)
            a.caption_side(sc, (mid - h, mid + h), "softmax p(digit)", op)
        # the digits label the columns, under the lowest row
        if self.logits.visible or t_bars > 0:
            row = self.onehot if t_hot > 0 else self.logits
            for c in range(10):
                p = row.cell(c)
                a.caption(sc, (p - h, p + h), str(c), size=0.029)
        if t_hot > 0:
            a.caption_side(sc, self.onehot, hot_text, ease_out(t_hot))
        if t_loss > 0:
            a.caption(sc, self.onehot, self.result_text(), ease_out(t_loss), size=0.031,
                      offset=0.052, max_width=0.30)

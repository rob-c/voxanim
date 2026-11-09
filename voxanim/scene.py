"""Scene description and the renderer that turns it into pixels.

A :class:`Scene` is a flat, per-frame list of primitives in world space --
voxels, lines, translucent quads and text.  Animation code builds a fresh one
for every instant ``t``; nothing is mutated between frames, so any frame can
be rendered on its own (which is what makes multi-process rendering and
motion-blur sub-frames trivial).

:class:`Renderer` owns everything that is expensive and reusable: the
supersampled camera, the voxel sprite atlases, the background plate.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np
from PIL import Image, ImageDraw

from . import color as C
from .cache import DEFAULT_CACHE
from .camera import REF_H, REF_W, Camera
from .fonts import font, text_box
from .layout import MIN_TEXT
from .raster import Framebuffer, gaussian_blur, polygon_mask
from .sprites import REFLECTION_LOOK, Look, SpriteAtlas

OPAQUE = 0.999          # opacity at or above which a voxel is solid (writes depth)
INVISIBLE = 0.003       # opacity at or below which a primitive is not drawn
TEXT_COLOR = tuple(C.TEXT.tolist())

# Supersampling is dropped above this output width (see Style.at_scale).
SS_MAX_WIDTH = 3840
# The soft floor work (shadows, reflection) is done at most this wide, then
# upsampled.
FLOOR_MAX_WIDTH = 1920.0

# Background vignette: a squashed ellipse centred a little above the middle.
# r2 = (x * VIGNETTE_SX)^2 + (y * VIGNETTE_SY + VIGNETTE_DY)^2 in [-1, 1]
# frame coordinates; darkening starts at r2 = VIGNETTE_R2 and is
# Style.vignette at full strength.
VIGNETTE_SX, VIGNETTE_SY, VIGNETTE_DY = 0.85, 1.05, 0.15
VIGNETTE_R2 = 0.35

# Contact shadows: each casting voxel drops a square slightly larger than
# itself (by SHADOW_MARGIN world units on each side) onto the floor, fading
# with height over SHADOW_FALLOFF world units.  Voxels fainter than
# SHADOW_MIN_OPACITY cast none.
SHADOW_MARGIN = 0.05
SHADOW_FALLOFF = 9.0
SHADOW_MIN_OPACITY = 0.05
# Reflections fainter than this (after the height falloff) are skipped.
REFLECTION_CUTOFF = 0.04

# Text compositing, all as fractions of the text size (its height in px).
TEXT_PAD = 0.35         # crop margin around Pillow's box (+ TEXT_PAD_PX)
TEXT_PAD_PX = 3
HALO_WIDTH = 0.11       # backdrop-coloured stroke behind the glyphs
HALO_STRENGTH = 0.55    # its opacity, times the text's own


# ------------------------------------------------------------------- style


@dataclass(frozen=True)
class Style:
    """Render settings that are not part of an animation's content."""

    width: int = REF_W
    height: int = REF_H
    ss: int = 2                       # supersampling factor
    look: Look = Look()
    reflection_look: Look = REFLECTION_LOOK
    background: tuple = tuple(C.BACKGROUND.tolist())
    vignette: float = 0.035           # darkening at the frame corners (fraction)
    shadow: float = 0.30              # strength of contact shadows on the floor
    shadow_blur: float = 0.55         # in world units
    reflection: float = 0.60          # strength of the glossy floor
    reflection_falloff: float = 4.0   # world units over which it fades out
    reflection_blur: float = 0.6      # in reference px
    motion_blur: bool = True
    shutter: float = 0.75             # fraction of the frame interval
    blur_px: float = 2.0              # max travel per sub-frame, in *output* px
    blur_max: int = 64                # sub-frame cap
    cache_dir: str = DEFAULT_CACHE

    def at_scale(self, k):
        """The same look at ``k`` times the resolution.

        Sprites, lines and glass carry their own analytic anti-aliasing, so
        supersampling only settles edges *between* primitives.  Above 4K a
        pixel is too small for that to be visible, so 8K renders at 1x.
        """
        width = int(round(self.width * k))
        ss = self.ss if width <= SS_MAX_WIDTH else 1
        return replace(self, width=width, height=int(round(self.height * k)), ss=ss)

    @property
    def ref(self):
        """Output pixels per reference pixel."""
        return self.width / REF_W


# Sprites are anti-aliased analytically, so 2x supersampling only has to
# settle edges *between* primitives: against a 4x reference it differs by
# 0.05 levels on average (3x: 0.035) at half the cost of 3x.
QUALITY = {
    "draft": dict(ss=1, motion_blur=False, reflection=0.0),
    "preview": dict(ss=2, motion_blur=False),
    "final": dict(ss=2),
    "max": dict(ss=3),
}


# ------------------------------------------------------------------- scene


class Voxel(NamedTuple):
    """One voxel, as :meth:`Scene.voxel` records it."""

    center: np.ndarray       # world position (float64, 3)
    color: np.ndarray        # base colour (float32 RGB, 0..1)
    opacity: float
    white: float             # blend towards flat white (0..1)
    glow: float              # added light
    shadow: bool             # casts a contact shadow
    reflect: bool            # shows in the floor reflection
    visible: bool            # seen by the camera
    size: float              # edge length relative to the unit voxel


class Line(NamedTuple):
    """One line segment, as :meth:`Scene.line` records it."""

    p0: np.ndarray           # world end points (float64, 3)
    p1: np.ndarray
    width: float             # reference px
    color: np.ndarray        # float32 RGB, 0..1
    opacity: float
    bias: float              # world units towards the viewer, for the depth test


class Quad(NamedTuple):
    """One shaded parallelogram ``p0 + u e1 + v e2``, as :meth:`Scene.quad` records it."""

    p0: np.ndarray           # world corner (float64, 3)
    e1: np.ndarray           # world edge vectors
    e2: np.ndarray
    shader: object           # points -> (rgb, alpha[, emit]); see Framebuffer.quad
    bias: float


@dataclass
class Text:
    """A line of text placed in frame coordinates (resolution independent)."""

    x: float                 # fraction of the frame width
    y: float                 # fraction of the frame height
    s: str
    size: float              # fraction of the frame height
    kind: str = "regular"    # font role, see voxanim.fonts.FONTS
    color: tuple = TEXT_COLOR
    opacity: float = 1.0
    anchor: str = "la"       # Pillow text anchor
    yielded: float = 1.0     # how much it faded to make room for geometry (QA)


@dataclass
class Scene:
    """Everything visible at one instant, built fresh for every frame."""

    voxels: list = field(default_factory=list)      # [Voxel]
    lines: list = field(default_factory=list)       # [Line]
    quads: list = field(default_factory=list)       # [Quad]
    texts: list = field(default_factory=list)       # [Text]
    probes: dict = field(default_factory=dict)      # key -> (world position, tolerance)
    floor_z: float = -4.0             # height of the glossy floor plane

    def voxel(self, center, color, opacity=1.0, white=0.0, glow=0.0,
              shadow=True, reflect=True, visible=True, size=1.0):
        """One unit voxel centred at ``center``.

        ``white`` blends it towards a flat white (the multiply/accumulate
        highlight) and ``glow`` adds light.  ``visible=False`` voxels only
        appear in the floor reflection (they are hidden from the camera but
        not from below).  ``size`` scales the voxel (magnified kernels).
        """
        self.voxels.append(Voxel(np.asarray(center, np.float64), np.asarray(color, np.float32),
                                 float(opacity), float(white), float(glow), shadow, reflect,
                                 visible, float(size)))

    def line(self, p0, p1, width=1.4, color=(1.0, 1.0, 1.0), opacity=1.0, bias=0.05):
        """A segment between two world points.  ``width`` is in reference px;
        ``bias`` (world units, towards the viewer) lets lines that lie on a
        surface win the depth test.  Invisible lines are dropped."""
        if opacity > INVISIBLE:
            self.lines.append(Line(np.asarray(p0, np.float64), np.asarray(p1, np.float64),
                                   float(width), np.asarray(color, np.float32),
                                   float(opacity), float(bias)))

    def wire_box(self, lo, hi, inflate=0.0, **kw):
        """The twelve edges of the axis-aligned box ``[lo, hi]`` grown by
        ``inflate`` world units; ``kw`` goes to :meth:`line`."""
        lo = np.asarray(lo, np.float64) - inflate
        hi = np.asarray(hi, np.float64) + inflate
        pts = {}
        for i in (0, 1):
            for j in (0, 1):
                for k in (0, 1):
                    pts[i, j, k] = np.array([(lo, hi)[i][0], (lo, hi)[j][1], (lo, hi)[k][2]])
        for a in pts:
            for axis in range(3):
                if a[axis] == 0:
                    b = list(a)
                    b[axis] = 1
                    self.line(pts[a], pts[tuple(b)], **kw)

    def quad(self, p0, e1, e2, shader, bias=0.0):
        """A translucent parallelogram ``p0 + u e1 + v e2`` coloured by
        ``shader`` (see :meth:`voxanim.raster.Framebuffer.quad`)."""
        self.quads.append(Quad(np.asarray(p0, np.float64), np.asarray(e1, np.float64),
                               np.asarray(e2, np.float64), shader, float(bias)))

    def glass_box(self, lo, hi, view, shader, inflate=0.0, bias=0.0):
        """The camera-facing faces of an axis-aligned box, each shaded by
        ``shader(points, face_normal) -> (rgb, alpha)``."""
        lo = np.asarray(lo, np.float64) - inflate
        hi = np.asarray(hi, np.float64) + inflate
        d = hi - lo
        for axis in range(3):
            sign = 1.0 if view[axis] > 0 else -1.0
            n = np.zeros(3)
            n[axis] = sign
            p0 = lo.copy()
            if sign > 0:
                p0[axis] = hi[axis]
            a, b = [ax for ax in range(3) if ax != axis]
            e1, e2 = np.zeros(3), np.zeros(3)
            e1[a], e2[b] = d[a], d[b]
            self.quad(p0, e1, e2, (lambda pts, n=n: shader(pts, n)), bias)

    def text(self, x, y, s, size, kind="regular", color=TEXT_COLOR, opacity=1.0,
             anchor="la"):
        """Add a :class:`Text` (see its fields for units).  ``size`` is raised
        to :data:`voxanim.layout.MIN_TEXT` so nothing is too small to read."""
        self.texts.append(Text(x, y, s, max(size, MIN_TEXT), kind, color, opacity, anchor))

    def probe(self, key, pos, tol=1.0):
        """Register a moving point.  Motion blur takes enough sub-frames that no
        probe travels more than ``tol * Style.blur_px`` pixels in one."""
        self.probes[key] = (np.asarray(pos, np.float64), float(tol))


# ---------------------------------------------------------------- renderer


def _resize(arr, W, H):
    """Bilinear resize of a float ``(h, w)`` or ``(h, w, c)`` array."""
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(np.ascontiguousarray(arr), "F")
                          .resize((W, H), Image.BILINEAR))
    return np.stack([_resize(arr[..., c], W, H) for c in range(arr.shape[-1])], -1)


_WHITE = np.ones(3, np.float32)


class Renderer:
    """Turns scenes into float RGB frames at ``style`` resolution.

    Geometry is drawn premultiplied into a transparent, supersampled,
    z-buffered layer, resolved to output resolution, and only then composited
    over the floor plate (background + contact shadows + reflection), which is
    built at output resolution.  Because "over" is associative this is exact,
    and it keeps the soft floor work off the supersampled buffer.
    """

    TINT_CACHE = 1200           # tinted sprite phases kept (LRU)

    def __init__(self, camera: Camera, style: Style = Style()):
        self.style = style
        self.cam1 = replace(camera, width=style.width, height=style.height,
                            ppu=camera.ppu * style.width / camera.width)
        self.cam = self.cam1.scaled(style.ss)
        # Motion-blur sampling is measured on a reference-resolution camera so
        # it costs the same at every output resolution.
        self.cam_ref = replace(camera, width=REF_W, height=int(round(REF_W * style.height /
                                                                     style.width)),
                               ppu=camera.ppu * REF_W / camera.width)
        # The floor (shadows, glossy reflection) is soft: above 1080p it is
        # rendered at FLOOR_MAX_WIDTH and upsampled.
        self.floor_k = min(1.0, FLOOR_MAX_WIDTH / style.width)
        self.fcam = self.cam1.scaled(self.floor_k) if self.floor_k < 1 else self.cam1
        self.atlas = SpriteAtlas(self.cam.M, self.cam.view, style.look,
                                 cache_dir=style.cache_dir)
        mcam = self.fcam.mirrored()
        self.ratlas = SpriteAtlas(mcam.M, mcam.view, style.reflection_look,
                                  cache_dir=style.cache_dir)
        self.mview = mcam.view
        self._atlases = {(False, 1.0): self.atlas, (True, 1.0): self.ratlas}
        self._bg = None
        self._tints = OrderedDict()

    def atlas_for(self, size, mirror=False):
        """Sprite atlas for voxels ``size`` times the unit (built on demand)."""
        key = (mirror, round(size, 3))
        if key not in self._atlases:
            st = self.style
            cam = self.fcam.mirrored() if mirror else self.cam
            base = st.reflection_look if mirror else st.look
            look = replace(base, size=base.size * size, radius=base.radius * size)
            self._atlases[key] = SpriteAtlas(cam.M, cam.view, look, cache_dir=st.cache_dir)
        return self._atlases[key]

    # -- background -------------------------------------------------------
    def _background(self, W, H):
        """The vignetted backdrop at ``W x H`` (cached for the last size asked)."""
        if self._bg is None or self._bg.shape[:2] != (H, W):
            st = self.style
            ys = np.linspace(-1, 1, H, dtype=np.float32)[:, None]
            xs = np.linspace(-1, 1, W, dtype=np.float32)[None, :]
            base = np.asarray(st.background, np.float32)
            r2 = (xs * VIGNETTE_SX) ** 2 + (ys * VIGNETTE_SY + VIGNETTE_DY) ** 2
            shade = 1.0 - st.vignette * np.clip(r2 - VIGNETTE_R2, 0, 1)
            self._bg = (base[None, None, :] * shade[..., None]).astype(np.float32)
        return self._bg

    # -- text -------------------------------------------------------------
    def _draw_texts(self, rgb, texts):
        """Composite ``texts`` onto ``rgb`` in place (and return it)."""
        for t in texts:
            self._draw_text(rgb, t)
        return rgb

    def _draw_text(self, rgb, t):
        """Composite one :class:`Text` in place, touching only a box around it:
        a soft halo in the backdrop colour, then the glyphs."""
        H, W = rgb.shape[:2]
        px = t.size * H
        f = font(t.kind, px)
        x, y = t.x * W, t.y * H
        bx0, by0, bx1, by1 = text_box(t, W, H)
        # Generous margin: Pillow snaps the (fractional) origin and some
        # glyphs overshoot the reported box -- a tight crop clipped the
        # tops of 7s, fs and 0s by a pixel.
        pad = int(np.ceil(px * TEXT_PAD)) + TEXT_PAD_PX
        x0, y0 = int(np.floor(bx0)) - pad, int(np.floor(by0)) - pad
        x1, y1 = int(np.ceil(bx1)) + pad, int(np.ceil(by1)) + pad
        cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
        if cx1 <= cx0 or cy1 <= cy0:
            return
        org = (x - x0, y - y0)
        sl = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
        reg = rgb[cy0:cy1, cx0:cx1]
        # The halo keeps text crisp where it crosses particles, cage lines or
        # reflections.
        hw = max(1, int(round(px * HALO_WIDTH)))
        halo = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(halo).text(org, t.s, font=f, fill=255, anchor=t.anchor,
                                  stroke_width=hw, stroke_fill=255)
        ha = np.asarray(halo, np.float32)[sl] * np.float32(HALO_STRENGTH * t.opacity / 255.0)
        back = np.asarray(self.style.background, np.float32)
        reg = reg * (1 - ha[..., None]) + back * ha[..., None]
        mask = Image.new("L", (x1 - x0, y1 - y0), 0)
        ImageDraw.Draw(mask).text(org, t.s, font=f, fill=255, anchor=t.anchor)
        a = np.asarray(mask, np.float32)[sl]
        a = (a * np.float32(t.opacity / 255.0))[..., None]
        rgb[cy0:cy1, cx0:cx1] = reg * (1 - a) + np.asarray(t.color, np.float32) * a

    # -- tinted sprites -------------------------------------------------------
    def _tinted(self, atlas, ky, kx, base):
        """Premultiplied colour of an untinted sprite phase, cached (most
        voxels share a handful of colours)."""
        key = (id(atlas), ky, kx, base.tobytes())
        hit = self._tints.get(key)
        if hit is not None:
            self._tints.move_to_end(key)
            return hit
        A, D, W, _Z = atlas.data[ky, kx]
        prem = np.minimum(D[..., None] * base + W[..., None], A[..., None])
        self._tints[key] = prem
        if len(self._tints) > self.TINT_CACHE:
            self._tints.popitem(last=False)
        return prem

    def _voxel(self, fb, atlas, sx, sy, zc, col, op, white, glow, zwrite, fade=1.0):
        """Draw one voxel sprite centred on screen ``(sx, sy)`` at depth ``zc``;
        ``fade`` scales its colour and coverage (reflections)."""
        x0, y0, ky, kx = atlas.locate(sx, sy)
        data = atlas.data[ky, kx]
        if white == 0.0 and glow == 0.0 and fade == 1.0:
            fb.blit(x0, y0, self._tinted(atlas, ky, kx, col), data[0], data[3],
                    op, zc, zwrite)
        else:
            if fade != 1.0:
                data = (data[0] * fade, data[1] * fade, data[2] * fade, data[3])
            fb.sprite(x0, y0, data, col, _WHITE, op, zc, zwrite, white, glow)

    # -- floor: contact shadows and a glossy reflection ---------------------
    def _floor(self, scene):
        """Background plate with shadows and reflection, at output resolution."""
        st, cam = self.style, self.fcam
        plate = self._background(cam.width, cam.height)
        if st.shadow > 0 and scene.voxels:
            plate = self._shadows(scene, plate)
        if st.reflection > 0 and scene.voxels:
            plate = self._reflection(scene, plate)
        if self.floor_k < 1:
            plate = _resize(plate, self.cam1.width, self.cam1.height)
        return plate

    def _shadows(self, scene, plate):
        """``plate`` darkened by the voxels' soft contact shadows."""
        st, cam = self.style, self.fcam
        W, H = cam.width, cam.height
        h = st.look.size / 2 + SHADOW_MARGIN
        fz = scene.floor_z
        cs = [v for v in scene.voxels if v.shadow and v.opacity >= SHADOW_MIN_OPACITY]
        if not cs:
            return plate
        c = np.array([v.center for v in cs])
        size = np.array([v.size for v in cs])
        strength = np.array([v.opacity for v in cs]) * \
            np.exp(-np.maximum(c[:, 2] - fz, 0) / SHADOW_FALLOFF)
        quad = np.array([[-h, -h], [h, -h], [h, h], [-h, h]])
        pts = np.empty((len(c), 4, 3))
        pts[:, :, :2] = c[:, None, :2] + quad[None] * size[:, None, None]
        pts[:, :, 2] = fz
        # Soft by nature: build it at half resolution and upsample.
        scr = cam.project(pts) * 0.5
        hw, hh = (W + 1) // 2, (H + 1) // 2
        mask = polygon_mask(hw, hh, list(zip(scr, strength)))
        mask = gaussian_blur(mask, st.shadow_blur * cam.ppu * 0.5)
        mask = np.asarray(Image.fromarray(mask, "F").resize((W, H), Image.BILINEAR))
        return plate * (1.0 - st.shadow * np.clip(mask, 0.0, 1.0))[..., None]

    def _reflection(self, scene, plate):
        """``plate`` with the solid voxels' mirror image, fading with height."""
        st, cam = self.style, self.fcam
        W, H = cam.width, cam.height
        fz = scene.floor_z
        refl = [v for v in scene.voxels if v.reflect and v.opacity >= OPAQUE]
        if not refl:
            return plate
        c = np.array([v.center for v in refl])
        fade = np.exp(-np.maximum(c[:, 2] - fz, 0) / st.reflection_falloff)
        keep = fade >= REFLECTION_CUTOFF
        if not keep.any():
            return plate
        c, fade = c[keep], fade[keep]
        refl = [v for v, k in zip(refl, keep) if k]
        m = c.copy()
        m[:, 2] = 2 * fz - c[:, 2]
        scr = cam.project(m)
        depth = c @ self.mview
        rfb = Framebuffer(W, H, track_coverage=True)
        for n in np.argsort(depth, kind="stable"):
            v = refl[n]
            self._voxel(rfb, self.atlas_for(v.size, True), scr[n, 0], scr[n, 1],
                        depth[n], v.color, 1.0, v.white, 0.0, True, float(fade[n]))
        k = st.reflection
        blur = st.reflection_blur * W / REF_W
        cov = gaussian_blur(rfb.cov, blur)
        add = gaussian_blur(rfb.rgb, blur)
        return plate * (1.0 - k * cov)[..., None] + k * add

    # -- frame ------------------------------------------------------------------
    def render(self, scene: Scene, hdr=False, floor=None, text=True):
        """Frame as float RGB (display-referred sRGB, 0..1).  With ``hdr`` also
        the emission layer: ``(rgb, emit)``.  ``floor`` reuses a floor plate
        (motion-blur sub-frames share one: it is soft and faint).  ``text=False``
        leaves the text off (motion blur adds it once, sharp)."""
        st, cam = self.style, self.cam
        ss = st.ss
        fb = Framebuffer(cam.width, cam.height, track_coverage=True, track_emission=hdr)

        vis = [v for v in scene.voxels if v.visible]
        if vis:
            centers = np.array([v.center for v in vis])
            scr = cam.project(centers)
            depth = centers @ cam.view
        items = []                      # (depth, kind, index)
        for n, v in enumerate(vis):
            items.append((depth[n], 0 if v.opacity >= OPAQUE else 1, n))
        for n, ln in enumerate(scene.lines):
            items.append((float(((ln.p0 + ln.p1) / 2) @ cam.view), 2, n))
        for n, q in enumerate(scene.quads):
            items.append((float((q.p0 + (q.e1 + q.e2) / 2) @ cam.view), 3, n))

        # Opaque voxels first (they write depth), then everything translucent.
        opaque = sorted((it for it in items if it[1] == 0), key=lambda it: it[0])
        rest = sorted((it for it in items if it[1] != 0), key=lambda it: it[0])
        for zc, kind, n in opaque + rest:
            if kind <= 1:
                v = vis[n]
                self._voxel(fb, self.atlas_for(v.size), scr[n, 0], scr[n, 1], zc, v.color,
                            v.opacity, v.white, v.glow, kind == 0)
            elif kind == 2:
                ln = scene.lines[n]
                fb.line(cam, ln.p0, ln.p1, ln.width * ss * st.width / REF_W, ln.color,
                        ln.opacity, ln.bias)
            else:
                q = scene.quads[n]
                fb.quad(cam, q.p0, q.e1, q.e2, q.shader, q.bias)

        geo = fb.downsample(ss)
        cov = fb.downsample_cov(ss)
        if floor is None:
            floor = self._floor(scene)
        out = floor * (1.0 - cov)[..., None]   # one new frame; then add in place
        out += geo                              # (an 8K frame is 400 MB)
        if text:
            out = self._draw_texts(out, scene.texts)
        return (out, fb.downsample_emit(ss)) if hdr else out

    # -- motion blur --------------------------------------------------------
    def travel(self, a: Scene, b: Scene):
        """Largest tolerance-weighted screen distance (reference px) any probe
        moves between scenes ``a`` and ``b``."""
        best = 0.0
        for key, (pa, tol) in a.probes.items():
            if key not in b.probes:
                continue
            pb = b.probes[key][0]
            d = self.cam_ref.project(pb) - self.cam_ref.project(pa)
            best = max(best, float(np.hypot(*d)) / tol)
        return best

    def subframes(self, build, t, dt):
        """How many sub-frames motion blur needs at ``t``."""
        st = self.style
        if not st.motion_blur or st.shutter <= 0:
            return 1
        half = 0.5 * st.shutter * dt
        # travel is measured in reference px; the budget is in output px, but
        # never finer than half a reference px (1080p and below look the same)
        budget = max(st.blur_px / st.ref, 0.5) if st.ref > 1 else 1.0
        dist = self.travel(build(t - half), build(t + half))
        return int(min(st.blur_max, max(1, math.ceil(dist / budget))))

    def render_at(self, build, t, dt, hdr=False):
        """Render ``build(t)`` with motion blur over ``shutter * dt`` seconds."""
        n = self.subframes(build, t, dt)
        if n == 1:
            return self.render(build(t), hdr)
        half = 0.5 * self.style.shutter * dt
        centre = build(t)
        floor = self._floor(centre)
        acc = None
        for k in range(n):
            tk = t - half + (k + 0.5) * (2 * half) / n
            img = self.render(build(tk), hdr, floor, text=False)
            if acc is None:
                acc = img
            elif hdr:
                acc = (acc[0] + img[0], acc[1] + img[1])
            else:
                acc = acc + img
        # Text is not motion-blurred: it is drawn once, sharp, where it is at
        # the frame's own time (a label riding a pan stays readable).
        if hdr:
            return self._draw_texts(acc[0] / n, centre.texts), acc[1] / n
        return self._draw_texts(acc / n, centre.texts)

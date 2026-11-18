"""Dark mode (voxanim.theme): neutrals turn over, vivid colours and the voxels do not."""
import itertools

import numpy as np
import pytest

from voxanim import color as C
from voxanim import presets, theme
from voxanim.camera import Camera
from voxanim.mech import Mechanism
from voxanim.scene import QUALITY, Renderer, Scene, Style

PALETTE = [C.INPUT, C.POSITIVE, C.NEGATIVE] + list(C.FILTERS)
DRAFT = Style(**QUALITY["draft"])


@pytest.fixture
def themed(monkeypatch):
    """Render the same thing in each theme: ``themed("dark", fn)``."""
    def run(name, fn):
        monkeypatch.setenv("VOXANIM_THEME", name)
        monkeypatch.setenv("VOXANIM_BRAND", "0")         # the corner mark is test_brand's
        return fn()
    return run


def test_turn_and_unturn_are_inverses():
    light = np.linspace(0.0, 1.0, 101, dtype=np.float32)
    assert np.allclose(theme.unturn(theme.turn(light)), light, atol=1e-5)
    dark = np.linspace(theme.PIT, theme.CHALK, 101, dtype=np.float32)
    assert np.allclose(theme.turn(theme.unturn(dark)), dark, atol=1e-5)


def test_the_backdrop_lands_on_night_and_black_ink_on_chalk():
    assert np.allclose(theme.to_dark(np.full(3, theme.PAGE)), theme.NIGHT, atol=1e-5)
    assert np.allclose(theme.to_dark(C.BACKGROUND), theme.NIGHT, atol=0.01)
    assert np.allclose(theme.to_dark(np.zeros(3)), theme.CHALK, atol=1e-5)
    ink = theme.to_dark(C.rgb("#22252a"))                # the diagrams' text
    assert ink.min() > 0.75 and ink.max() - ink.min() < 0.05
    paper = theme.to_dark(np.ones(3))                    # lighter than the page: deeper
    assert np.allclose(paper, theme.PIT, atol=1e-5) and theme.PIT < theme.NIGHT


def test_a_turned_grey_ramp_keeps_its_order_reversed():
    ramp = np.repeat(np.linspace(0, 1, 64, dtype=np.float32)[:, None], 3, 1)
    assert (np.diff(theme.to_dark(ramp)[:, 0]) < 0).all()


@pytest.mark.parametrize("colour", PALETTE, ids=lambda c: "%02x%02x%02x" % tuple(
    int(round(255 * v)) for v in c))
def test_the_palette_comes_through_untouched(colour):
    assert np.abs(theme.to_dark(colour) - colour).max() < 0.01


def test_a_pale_tint_becomes_a_deeper_tint_of_the_same_hue():
    pale = C.rgb("#dce8f6")                              # a soft blue panel
    out = theme.to_dark(pale)
    assert out.mean() < pale.mean() - 0.3
    assert np.argmax(out) == np.argmax(pale) and np.argmin(out) == np.argmin(pale)


def test_keep_then_to_dark_returns_a_neutral_to_where_it_started():
    greys = np.repeat(np.linspace(theme.PIT, theme.CHALK, 40, dtype=np.float32)[:, None], 3, 1)
    assert np.allclose(theme.to_dark(theme.keep(greys)), greys, atol=1e-4)
    for colour in PALETTE:                               # a vivid colour needs no keeping
        assert np.abs(theme.keep(colour) - colour).max() < 0.01


def test_light_mode_is_the_identity(monkeypatch):
    frame = np.random.default_rng(0).random((4, 5, 3), dtype=np.float32)
    monkeypatch.delenv("VOXANIM_THEME", raising=False)
    assert theme.name() == "light" and theme.suffix() == ""
    assert theme.apply(frame) is frame
    monkeypatch.setenv("VOXANIM_THEME", "dark")
    assert theme.name() == "dark" and theme.suffix() == "-dark"
    assert np.allclose(theme.apply(frame), theme.to_dark(frame))


def test_dark_ink_is_lifted_to_the_contrast_it_had_on_the_light_page():
    page, night = (theme.luminance(np.full(3, v)) for v in (theme.PAGE, theme.NIGHT))
    for hexed in ("#1f4e9c", "#0f5a5a", "#7a1f1f", "#1d6b2a", "#22252a", "#6b6f75"):
        colour = C.rgb(hexed)
        was = theme.contrast(theme.luminance(colour), page)
        now = theme.contrast(theme.luminance(theme.ink(colour)), night)
        assert now >= min(was, theme.INK_CONTRAST) - 0.05, hexed
    navy = C.rgb("#1f4e9c")
    lifted = theme.ink(navy)
    assert np.argmax(lifted) == 2 and lifted[2] - lifted[0] > 0.3     # still a vivid blue
    assert np.allclose(theme.ink(C.POSITIVE), C.POSITIVE, atol=0.01)  # bright enough as is
    many = theme.ink(np.stack([navy, C.POSITIVE]))                    # and for an array
    assert np.allclose(many[0], lifted, atol=1e-5) and np.allclose(many[1], C.POSITIVE,
                                                                   atol=0.01)


def test_ink_on_a_surface_is_the_one_that_reads_there():
    dark_green, deep_green = C.rgb("#1d6b2a"), C.rgb("#3d6140")      # label; its box, turned
    on_box = theme.ink(dark_green, deep_green)
    assert theme.contrast(theme.luminance(on_box), theme.luminance(deep_green)) >= 4.4
    assert np.argmax(on_box) == 1                                    # still green
    ink = C.rgb("#22252a")                               # dark ink on an orange box: kept
    assert np.allclose(theme.ink(ink, C.POSITIVE), ink, atol=1e-6)
    white = np.ones(3, np.float32)                       # white on a blue button: kept
    assert np.allclose(theme.ink(white, C.rgb("#2f4ea0")), white, atol=1e-6)
    faint = C.rgb("#a0a0a0")                             # a faint caption stays faint
    on_tint = theme.ink(faint, C.rgb("#1c2a3a"))
    assert np.allclose(on_tint, theme.to_dark(faint), atol=1e-6)


def _toy():
    """A small scene of data (voxels) and furniture (a line, a panel, a caption)."""
    sc = Scene()
    cols = [C.INPUT, (1.0, 1.0, 1.0), (0.13, 0.14, 0.16), (0.5, 0.5, 0.5), (0.79, 1.0, 0.96)]
    for n, col in enumerate(cols):
        sc.voxel((3.0 * n - 6.0, 0.0, 0.0), col, glow=0.25 if n == 4 else 0.0,
                 white=0.6 if n == 0 else 0.0)
    sc.line((-8, 0, 5), (8, 0, 5), width=6.0, color=(0.1, 0.1, 0.1))
    sc.quad((-8, 0, -6), (4, 0, 0), (0, 0, 2), lambda p: (np.ones(3, np.float32),
                                                          np.ones(p.shape[:2], np.float32)))
    kept = lambda p: (np.ones(3, np.float32), np.ones(p.shape[:2], np.float32))  # noqa: E731
    kept.keep_theme = True
    sc.quad((2, 0, -6), (4, 0, 0), (0, 0, 2), kept)
    sc.text(0.5, 0.08, "caption", 0.05, color=(0.13, 0.14, 0.16), anchor="mm")
    return sc


def _at(r, frame, point):
    x, y = r.cam1.project(np.asarray(point, np.float64))
    return frame[int(round(y)), int(round(x))]


def test_voxels_are_drawn_as_in_light_mode_and_the_page_is_turned(themed):
    cam = Camera(width=960, height=540, ppu=24.0)
    r = Renderer(cam, DRAFT)
    sc = _toy()
    light = themed("light", lambda: r.render(sc))
    dark = themed("dark", lambda: r.render(sc))
    for n in range(5):                                   # every voxel, whatever its colour
        centre = (3.0 * n - 6.0, 0.0, 0.0)
        assert np.allclose(_at(r, dark, centre), _at(r, light, centre), atol=1e-6), n
    assert light[5, 5].min() > 0.6 and dark[5, 5].max() < 0.2          # the backdrop
    assert _at(r, light, (0, 0, 5)).max() < 0.2                        # the line: dark ink
    assert _at(r, dark, (0, 0, 5)).min() > 0.8                         # ... turned to chalk
    assert _at(r, dark, (-6, 0, -5)).max() < 0.1                       # the white panel
    assert _at(r, dark, (4, 0, -5)).min() > 0.99                       # the one marked data
    box = slice(int(0.04 * 540), int(0.12 * 540)), slice(380, 580)     # the caption
    assert light[box].min() < 0.3 and dark[box].max() > 0.8


def _flat(colour, alpha=1.0):
    return lambda p: (np.asarray(colour, np.float32), np.full(p.shape[:2], alpha, np.float32))


def test_what_lights_the_voxels_is_kept_and_the_same_thing_on_the_page_is_turned(themed):
    """conv2d's multiply/accumulate wave is a white glass quad and its receptive-field
    frame a white line: turned with the page they swept the kernel black."""
    cam = Camera(width=960, height=540, ppu=24.0)
    r = Renderer(cam, DRAFT)
    sc = Scene()
    for x in range(-2, 3):
        for z in range(-2, 3):
            sc.voxel((float(x) - 8, 0.0, float(z)), C.INPUT)
    view = cam.view
    for cx in (-8.0, 8.0):                               # on the block, and on the page
        mid = np.array([cx, 0.0, 0.0]) + 2.0 * view
        sc.quad(mid - (1.5, 0, 1.5), (3, 0, 0), (0, 0, 1.2), _flat((1, 1, 1), 0.9))
        sc.line(mid + (-1.5, 0, 1.0), mid + (1.5, 0, 1.0), width=5.0, color=(1, 1, 1))
        sc.line(mid + (-1.5, 0, 0.4), mid + (1.5, 0, 0.4), width=5.0, color=(0.1, 0.1, 0.1))
    light = themed("light", lambda: r.render(sc))
    dark = themed("dark", lambda: r.render(sc))

    def at(frame, cx, z):
        return _at(r, frame, np.array([cx, 0.0, z]) + 2.0 * view)
    assert at(dark, -8, -0.9).min() > 0.9 and np.allclose(at(dark, -8, -0.9),
                                                          at(light, -8, -0.9), atol=1e-6)
    assert at(dark, -8, 1.0).min() > 0.95                # the white frame on the block
    assert at(dark, 8, -0.9).max() < 0.15                # the same glass on the page: dark
    assert at(dark, 8, 1.0).max() < 0.15                 # and the white line there
    assert at(dark, 8, 0.4).min() > 0.8                  # a dark line turns on the page
    assert at(dark, -8, 0.4).min() > 0.8                 # ... and on the block: it shows


def test_an_image_is_data_and_what_is_drawn_on_it_keeps_its_ink(themed):
    cam = Camera(width=960, height=540, ppu=24.0)
    r = Renderer(cam, DRAFT)
    tex = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(1, 16, 1).repeat(3, 2)
    tex[0, 0] = C.BACKGROUND                             # a cell not yet filled in

    def image(p):                                        # white 1, black 0, left to right
        u = np.clip((p[..., 0] + 8.0) / 16.0, 0, 1 - 1e-9)
        return tex[0, (u * 16).astype(int)], np.ones(p.shape[:2], np.float32)
    image.key = ("image", "ramp")
    sc = Scene()
    sc.quad((-8, 0, -2), (16, 0, 0), (0, 0, 4), image)
    front = 0.5 * cam.view
    sc.line(np.array([-6.0, 0, 0]) + front, np.array([6.0, 0, 0]) + front, width=6.0,
            color=(0.8, 0.1, 0.1))
    sc.quad(np.array([5.0, 0, 0.8]) + front, (1, 0, 0), (0, 0, 1), _flat((0.05, 0.05, 0.05)))
    light = themed("light", lambda: r.render(sc))
    dark = themed("dark", lambda: r.render(sc))
    for x, z in ((-4.2, -1.0), (0.3, -1.0), (7.2, -1.0),         # the ramp: not turned over
                 (0.0, 0.0), (5.5, 1.3)):                        # the line and the marker
        assert np.allclose(_at(r, dark, (x, 0, z)), _at(r, light, (x, 0, z)), atol=1e-6)
    assert _at(r, dark, (7.2, 0, -1.0)).min() > _at(r, dark, (-4.2, 0, -1.0)).max() + 0.4
    assert _at(r, dark, (-7.5, 0, -1.0)).max() < 0.2     # the unfilled cell is page
    image.keep_theme = False                             # an image may ask to be turned
    turned = themed("dark", lambda: r.render(sc))
    assert _at(r, turned, (7.2, 0, -1.0)).max() < 0.2


def test_the_floor_reflection_is_deep_not_pale_in_dark_mode(themed):
    cam = Camera(width=960, height=540, ppu=24.0)
    r = Renderer(cam, Style(**QUALITY["preview"]))
    sc = Scene()
    for x in range(-3, 4):
        for y in range(-3, 4):
            sc.voxel((float(x), float(y), sc.floor_z + 1.0), C.INPUT)
    light = themed("light", lambda: np.array(r._floor(sc)))      # the plate under the voxels
    dark = themed("dark", lambda: np.array(r._floor(sc)))
    assert np.ptp(light.mean(-1)) > 0.05 and light.min() > 0.5   # shadow and reflection
    assert dark.max() < 0.3                              # no pale ghost of the block
    tinted = dark.max(-1) - dark.min(-1)
    assert tinted.max() > 0.02                           # the block's teal is in the floor


def test_a_lit_white_voxel_stays_white_in_dark_mode(themed):
    """mlp-4-forward-pass: the bar over the predicted class is a pale mint (0.79, 1, 0.96)
    made white by its glow.  Turning the frame, and pre-turning voxel colours, both made
    it teal: its colour sat where "neutral" and "vivid" meet."""
    cam = Camera(width=960, height=540, ppu=24.0)
    r = Renderer(cam, DRAFT)
    sc = Scene()
    for z in range(4):
        sc.voxel((0.0, 0.0, float(z)), (0.79, 1.0, 0.96), glow=0.25)
    light = themed("light", lambda: r.render(sc))
    dark = themed("dark", lambda: r.render(sc))
    x, y = (int(round(v)) for v in r.cam1.project(np.array([0.0, 0.0, 1.5])))
    bar_light, bar_dark = light[y - 6:y + 7, x - 2:x + 3], dark[y - 6:y + 7, x - 2:x + 3]
    assert bar_light.min() > 0.9                         # white in light mode
    assert np.allclose(bar_dark, bar_light, atol=1e-6)   # and the same white in dark


def test_a_dark_frame_of_an_animation_has_a_dark_page_light_title_and_the_same_voxels(themed):
    anim = presets.build("conv2d")
    r = Renderer(anim.cam, DRAFT)
    sc = anim.scene(0.5 * anim.duration)
    assert sc.voxel_count() and sc.texts
    light = themed("light", lambda: r.render(sc))
    dark = themed("dark", lambda: r.render(sc))
    assert light is not dark and light.shape == dark.shape
    edge = np.concatenate([dark[:6].reshape(-1, 3), dark[:, :6].reshape(-1, 3)])
    assert np.median(edge) < 0.2 < 0.7 < np.median(light[:6])          # the page turned over
    # the voxels: where a voxel's centre is seen (not behind another), the same pixel
    V = sc.voxel_arrays()
    front = V["visible"] & (V["opacity"] >= 0.999)
    pts = r.cam1.project(V["center"][front])
    same = total = 0
    for x, y in np.round(pts).astype(int):
        if 0 <= x < dark.shape[1] and 0 <= y < dark.shape[0]:
            total += 1
            same += bool(np.abs(dark[y, x] - light[y, x]).max() < 0.02)
    assert total > 50 and same > 0.9 * total             # a few sit under a line or a frame
    vivid = (light.max(-1) - light.min(-1)) > 0.4        # and the colours are still there
    assert vivid.sum() > 500
    assert np.abs(dark[vivid] - light[vivid]).mean() < 0.02
    # the title: dark ink on the light page, light ink on the dark one
    from voxanim.fonts import text_box
    title = max(sc.texts, key=lambda t: t.size)
    x0, y0, x1, y1 = (int(v) for v in text_box(title, dark.shape[1], dark.shape[0]))
    assert dark[y0:y1, x0:x1].max() > 0.6 > np.median(dark[y0:y1, x0:x1])
    assert light[y0:y1, x0:x1].min() < 0.5 < np.median(light[y0:y1, x0:x1])


def test_rendering_dark_does_not_change_a_later_light_frame(themed):
    anim = presets.build("conv2d")
    r = Renderer(anim.cam, DRAFT)
    sc = anim.scene(1.0)
    before = themed("light", lambda: r.render(sc))
    themed("dark", lambda: r.render(sc))
    assert np.array_equal(themed("light", lambda: r.render(sc)), before)


def test_a_caption_names_a_shade_by_the_theme_it_is_drawn_in(monkeypatch):
    monkeypatch.delenv("VOXANIM_THEME", raising=False)
    assert theme.word("black", "white") == "black"
    monkeypatch.setenv("VOXANIM_THEME", "dark")
    assert theme.word("black", "white") == "white"
    monkeypatch.setenv("VOXANIM_THEME", "light")
    assert "%s: the truth" % theme.word("black", "white") == "black: the truth"


def _caption(name, part, at=None):
    """The on-screen text of animation ``name`` that holds ``part``, searched through its
    run (or only at fraction ``at`` of it)."""
    anim = presets.build(name)
    for k in ([at] if at is not None else [(n + 0.5) / 60 for n in range(60)]):
        for t in anim.scene(anim.duration * k).texts:
            if part in t.s and t.opacity > 0.5:
                return t.s
    return None


def test_the_shade_a_caption_names_is_the_one_on_screen_in_each_theme(monkeypatch):
    """rope-context-extension's attention map is a grey ramp on the page: dark ink for more
    weight in light mode, light for more in dark mode, and the caption says which."""
    monkeypatch.delenv("VOXANIM_THEME", raising=False)
    assert _caption("rope-context-extension", "more weight").startswith("darker: more weight")
    monkeypatch.setenv("VOXANIM_THEME", "dark")
    assert _caption("rope-context-extension", "more weight").startswith("lighter: more weight")


class _Grids(Mechanism):
    """A mask of who may read whom, half filled in, beside a heat map with a scale."""
    TITLE = "grids"
    PHASES = (("only", 4.0),)
    MASK, HEAT = (0.27, 0.55), (0.73, 0.55)

    def phase_1(self, sc, u, op):
        rows = max(1, int(8 * u + 1e-9))
        self.mask = self.grid(sc, np.tril(np.ones((8, 8))), *self.MASK, 200, 200, op,
                              reveal=rows / 8, mask=True)
        self.heat = self.grid(sc, np.linspace(0, 1, 64).reshape(8, 8), *self.HEAT, 200, 200, op)


def test_a_mask_grid_turns_dark_and_a_heat_map_with_a_scale_stays(themed):
    """The masks of the attention animations stood as white panels on the dark page.
    Marked ``mask=True`` they turn: the allowed cells keep their colour, the paper goes
    dark - at every stage of filling in (an automatic rule flipped as they filled)."""
    anim = _Grids()
    r = Renderer(anim.cam, DRAFT)
    H, W = r.cam1.height, r.cam1.width

    def cell(frame, grid, row, col):
        fx, fy = anim.fr(anim.P(*grid(row, col)))        # layout -> frame fractions
        return frame[int(round(fy * H)), int(round(fx * W))]

    for t in (0.6, 1.3, 2.2, 3.5):                       # 1, 2, 4 and 7 rows filled in
        sc = anim.scene(t)
        light = themed("light", lambda sc=sc: r.render(sc))
        dark = themed("dark", lambda sc=sc: r.render(sc))
        # the mask: an allowed cell is as vivid as in light mode, paper and the rows not
        # yet shown are dark
        assert np.allclose(cell(dark, anim.mask, 0, 0), cell(light, anim.mask, 0, 0),
                           atol=0.02), t
        for row, col in ((0, 5), (7, 6)):
            assert cell(light, anim.mask, row, col).min() > 0.85, (t, row, col)
            assert cell(dark, anim.mask, row, col).max() < 0.2, (t, row, col)
        # the heat map: every cell exactly as in light mode, the pale end still pale
        for row, col in ((0, 0), (3, 4), (7, 7)):
            assert np.allclose(cell(dark, anim.heat, row, col), cell(light, anim.heat, row, col),
                               atol=1e-6), (t, row, col)
        assert cell(dark, anim.heat, 0, 0).min() > 0.9


def test_a_kept_image_fades_in_from_the_dark_page(themed):
    """A kept image at part opacity is that much of its light look over the dark page: a
    white sheet half faded is mid-grey, as black ink half faded is on the light page.  It
    is never lighter than the image or darker than the page, and it gets there in order."""
    cam = Camera(width=960, height=540, ppu=24.0)
    r = Renderer(cam, DRAFT)

    def frame(opacity):
        def image(p):
            return np.ones(3, np.float32), np.full(p.shape[:2], opacity, np.float32)
        image.key = ("image", "sheet", opacity)
        sc = Scene()
        sc.quad((-4, 0, -2), (8, 0, 0), (0, 0, 4), image)
        return themed("dark", lambda: r.render(sc))
    page = _at(r, frame(0.0), (0, 0, 0))
    assert page.max() < 0.2
    steps = [_at(r, frame(a), (0, 0, 0)) for a in (0.0, 0.25, 0.5, 0.75, 1.0)]
    for a, got in zip((0.0, 0.25, 0.5, 0.75, 1.0), steps):
        assert np.allclose(got, page * (1 - a) + a, atol=1e-4), a
    assert all(np.all(b >= a) for a, b in itertools.pairwise(steps))


def test_dark_ink_on_a_dulled_mid_tone_gives_way_to_light_ink():
    """gradient-descent writes each neuron's value on a disc tinted by it; half lit, the
    disc is a dull teal in dark mode where black ink (4.4:1) was hard to read."""
    ink = C.rgb("#22252a")
    dull = np.array([0.31, 0.50, 0.47], np.float32)
    assert theme.TIE <= theme.luminance(dull) < theme.MURK
    on = theme.ink(ink, dull)
    assert on.min() > 0.9
    assert theme.contrast(theme.luminance(on), theme.luminance(dull)) >= 3.5
    red = np.array([0.85, 0.31, 0.31], np.float32)       # as light a surface, but vivid:
    assert theme.TIE <= theme.luminance(red) < theme.MURK
    assert theme.ink(ink, red).max() < 0.3               # it keeps its light-mode ink
    lit = np.array([0.45, 0.70, 0.67], np.float32)       # and a light surface keeps dark ink
    assert np.allclose(theme.ink(ink, lit), ink, atol=1e-6)

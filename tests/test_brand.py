"""The voxanim mark (voxanim.brand): the corner mark on every frame and the site's logo files."""
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from voxanim import brand
from voxanim.camera import REF_H, REF_W

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_the_mark_is_a_v_of_nine_cubes_with_its_point_lit():
    cubes = brand.cubes()
    assert len(cubes) == 9 and sum(lit for *_, lit in cubes) == 1
    (col, row, _lit), = [c for c in cubes if c[2]]
    assert (col, row) == (2, 0)                          # the bottom, middle cube
    tops = sorted(c for c, r, _ in cubes if r == 4)
    assert tops == [0, 4]                                # two arms at the top
    assert sorted(cubes, key=lambda q: (q[1], -q[0])) == cubes     # drawing order


@pytest.mark.parametrize("size", [(540, 960), (1080, 1920), (2160, 3840)])
def test_the_corner_mark_sits_in_the_bottom_right_margin_at_every_size(size):
    height, width = size
    x0, y0, x1, y1 = brand.box(height, width)
    assert x1 <= width and y1 <= height
    assert width - x1 == round(brand.MARGIN * height) == height - y1
    assert y0 > 0.93 * height                            # below the line evidence tags use
    assert 0.18 * width < x1 - x0 < 0.30 * width         # legible, and out of the way
    # the same picture at every size, to within resampling
    assert abs((x1 - x0) / height - (brand.box(REF_H, REF_W)[2] - brand.box(REF_H, REF_W)[0])
               / REF_H) < 0.01


def test_stamping_changes_only_the_corner_and_draws_the_v_in_its_colours(monkeypatch):
    frame = np.full((REF_H, REF_W, 3), 0.87, np.float32)
    out = brand.stamp(frame.copy())
    x0, y0, x1, y1 = brand.box(REF_H, REF_W)
    changed = np.abs(out - frame).max(-1) > 1e-6
    assert changed.any() and not changed[:y0].any() and not changed[:, :x0].any()
    assert 0.05 < changed[y0:y1, x0:x1].mean() < 0.6     # a mark, not a block
    patch = out[y0:y1, x0:x1]
    chroma = patch.max(-1) - patch.min(-1)
    cols = np.nonzero((chroma > 0.15).any(0))[0]         # the V: teal and orange
    assert len(cols) and cols.max() < 0.2 * (x1 - x0)    # at the left end of the mark
    teal = patch[(patch[..., 2] > patch[..., 0] + 0.15)]
    orange = patch[(patch[..., 0] > patch[..., 2] + 0.3)]
    assert len(teal) > len(orange) > 0                   # eight cubes to one
    words = patch[:, cols.max() + 3:]                    # the name and the address: grey
    assert (words.max(-1) - words.min(-1)).max() < 0.02
    assert words.min() >= min(brand.GREY) - 0.02 and words.max() <= 0.87 + 1e-6
    # on a dark page the V keeps its colours and the words are a darker grey
    night = np.full((REF_H, REF_W, 3), 0.12, np.float32)
    dark = brand.stamp(night.copy(), dark=True)[y0:y1, x0:x1]
    solid = brand._plate(REF_H, REF_W)[1] > 0.98         # inside a cube, not on its edge
    assert solid.sum() > 20 and np.abs(dark - patch)[solid].max() < 0.02
    assert dark[:, cols.max() + 3:].max() <= max(brand.GREY_DARK) + 0.02
    monkeypatch.setenv("VOXANIM_BRAND", "0")             # a clean plate when asked
    assert np.array_equal(brand.stamp(frame.copy()), frame)


def test_a_frame_too_small_for_the_mark_is_left_alone():
    tiny = np.full((8, 16, 3), 0.5, np.float32)
    assert np.array_equal(brand.stamp(tiny.copy()), tiny)


def test_every_rendered_frame_carries_the_mark(monkeypatch):
    from voxanim import presets
    from voxanim.scene import QUALITY, Renderer, Style
    anim = presets.build("conv2d")
    renderer = Renderer(anim.cam, Style(**QUALITY["draft"]))
    marked = renderer.render(anim.scene(1.0))
    monkeypatch.setenv("VOXANIM_BRAND", "0")
    plain = renderer.render(anim.scene(1.0))
    x0, y0, x1, y1 = brand.box(*marked.shape[:2])
    differs = np.abs(marked - plain).max(-1) > 1e-4
    assert differs[y0:y1, x0:x1].any() and not differs[:y0].any() and not differs[:, :x0].any()


@pytest.mark.parametrize("name", ["logo.svg", "logo-dark.svg", "logo-mark.svg"])
def test_the_site_logo_files_are_what_the_module_draws(name):
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_logo", os.path.join(ROOT, "tools",
                                                                           "make_logo.py"))
    make_logo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(make_logo)
    with open(os.path.join(ROOT, "website", name), encoding="utf-8") as fh:
        on_disk = fh.read()
    assert on_disk == make_logo.FILES[name]() + "\n"      # run tools/make_logo.py after a change
    root = ET.fromstring(on_disk)
    polygons = root.findall(".//{http://www.w3.org/2000/svg}polygon")
    assert len(polygons) == 27                           # nine cubes, three faces each
    fills = {p.get("fill") for p in polygons}
    assert brand.TEAL in fills and brand.ORANGE in fills
    assert ("voxanim" in on_disk) and (("<text" in on_disk) == (name != "logo-mark.svg"))

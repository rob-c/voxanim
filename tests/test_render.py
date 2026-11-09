"""The renderer: camera, culling, determinism, HDR, resolution independence."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
from conftest import DRAFT

from voxanim import presets
from voxanim.camera import Camera
from voxanim.scene import Renderer, Scene
from voxanim.tensor import Block


def test_camera_roundtrip_and_view():
    cam = Camera()
    p = np.array([3.0, -2.0, 1.5])
    sx, sy = cam.project(p)
    q = cam.unproject(sx, sy, z=p[2])
    assert np.allclose(p, q)
    # the view direction is the projection's null space and faces the viewer
    assert np.allclose(cam.M @ cam.view, 0, atol=1e-9)
    assert cam.view[2] > 0 and cam.mirrored().view[2] < 0


def test_culling_is_invisible():
    """Dropping hidden voxels must not change a single pixel."""
    cam = Camera()
    r = Renderer(cam, replace(DRAFT, reflection=0.0, shadow=0.0))
    b = Block((5, 6, 7)).place(cam.frac(0.5, 0.5))
    culled, full = Scene(), Scene()
    b.emit(culled, r.cam.view)
    for i, j, k in np.ndindex(b.present.shape):
        full.voxel(b.cell_center(i, j, k), b.base[i, j, k])
    assert len(culled.voxels) < len(full.voxels)
    diff = np.abs(r.render(culled) - r.render(full)).max()
    assert diff < 1.5 / 255, diff


def test_deterministic():
    anim = presets.get("conv2d").build()
    r = Renderer(anim.cam, DRAFT)
    a = r.render_at(anim.scene, 3.3, 1 / 30)
    b = Renderer(anim.cam, DRAFT).render_at(anim.scene, 3.3, 1 / 30)
    assert np.array_equal(a, b)


def test_hdr_transfer_functions():
    from voxanim import hdr
    # ST 2084 / BT.2100 / BT.2408 reference points
    assert abs(float(hdr.pq_encode(100)) - 0.5081) < 2e-4
    assert abs(float(hdr.pq_encode(1000)) - 0.7518) < 2e-4
    assert abs(float(hdr.pq_encode(hdr.WHITE)) - 0.58) < 1e-3
    assert abs(float(hdr.pq_decode(hdr.pq_encode(203.0))) - 203.0) < 0.05
    white = np.full((1, 1, 3), hdr.WHITE, np.float32)
    assert abs(float(hdr.hlg_encode(white)[0, 0, 0]) - 0.75) < 1e-3
    assert abs(float(hdr.hlg_encode(white * 0 + hdr.PEAK)[0, 0, 0]) - 1.0) < 1e-4
    # SDR white lands on graphics white; BT.709 -> BT.2020 keeps white white
    lw = hdr.display_light(np.ones((1, 1, 3), np.float32))[0, 0]
    assert np.allclose(lw, hdr.WHITE, atol=0.01)


def test_hdr_never_changes_the_sdr_picture():
    anim = presets.get("conv2d").build()
    r = Renderer(anim.cam, DRAFT)
    sc = anim.scene(0.8)                          # the white wave is emissive
    rgb, emit = r.render(sc, hdr=True)
    assert np.array_equal(rgb, r.render(sc)) and emit.max() > 0.1


def test_resolution_independent():
    """A 2x render, box-filtered down, matches the 1x render.  Text is
    excluded: glyphs are rasterised natively (hinted) at every resolution,
    which is what keeps them crisp at 8K."""
    from voxanim.qa import text_boxes
    anim = presets.get("lesson-stride-2").build()
    st = replace(DRAFT, ss=2)
    sc = anim.scene(6.0)
    a = Renderer(anim.cam, st).render(sc)
    b = Renderer(anim.cam, st.at_scale(2)).render(sc)
    b = b.reshape(540, 2, 960, 2, 3).mean((1, 3))
    d = np.abs(a - b).max(-1) * 255
    keep = np.ones(d.shape, bool)
    for _s, (x0, y0, x1, y1) in text_boxes(sc):
        keep[max(0, int(y0) - 2):int(y1) + 3, max(0, int(x0) - 2):int(x1) + 3] = False
    d = d[keep]
    assert d.mean() < 0.3 and np.percentile(d, 99) < 4, (d.mean(), np.percentile(d, 99))


def test_dither_is_static():
    """The anti-banding dither must not change between frames: a still scene
    has to give identical 8-bit frames (else GIFs balloon to 100+ MB)."""
    from voxanim.raster import to_uint8
    img = np.random.default_rng(0).random((32, 48, 3)).astype(np.float32)
    assert np.array_equal(to_uint8(img), to_uint8(img))

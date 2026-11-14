"""voxanim.accel: the device choice, and every PyTorch port against the numpy
code it replaces -- in float64 on the CPU, where the two must agree to
rounding, and the GPU framebuffer within one 8-bit level."""

from __future__ import annotations

import os

import numpy as np
import pytest

from voxanim import accel

torch = pytest.importorskip("torch")


def close(a, b):
    """Equal to float64 rounding: relatively, or absolutely for values that are
    zero up to rounding (a hidden bias under batch norm has no gradient at all,
    only ~1e-17 of noise on either side)."""
    return np.allclose(a, b, rtol=1e-10, atol=1e-13)


# ------------------------------------------------------------------ the device choice


def test_device_choice():
    """numpy mode is plain numpy; auto keeps work that does not gain on numpy;
    an explicit mode forces its device everywhere."""
    with accel.forced("numpy"):
        assert accel.device() is None and accel.torch() is None and accel.describe() == "numpy"
    with accel.forced("auto"):
        assert accel.device(prefer=False) is None
    with accel.forced("cpu"):
        assert accel.device(prefer=False).type == "cpu"
    with accel.forced("bogus"), pytest.raises(ValueError):
        accel.choice()
    assert accel.mlp_work((784, 64, 10), 64) == 64 * (784 * 64 + 64 * 10)
    # the batch-64 MLPs most animations train stay numpy under auto; a wide batch-1024 one moves
    assert accel.mlp_work((784, 256, 256, 10), 64) < accel.TRAIN_WORK
    assert accel.mlp_work((784, 1024, 10), 1024) > accel.TRAIN_WORK


def test_adam_matches_numpy():
    """accel.Adam is optim.Adam's update."""
    from voxanim.optim import Adam
    rng = np.random.default_rng(0)
    P = {"w": rng.normal(0, 1, (5, 3))}
    Pt = {"w": torch.tensor(P["w"])}
    a, b = Adam(P, 0.01), None
    with accel.forced("cpu"):
        b = accel.Adam(Pt, 0.01)
        for _ in range(5):
            g = rng.normal(0, 1, (5, 3))
            a.step(P, {"w": g})
            b.step(Pt, {"w": torch.tensor(g)})
    assert np.allclose(P["w"], Pt["w"].numpy(), rtol=1e-12, atol=1e-14)


# ------------------------------------------------------------------ the ports, in float64


@pytest.mark.parametrize("kind", ["dense", "conv"])
def test_autoencoder_port(kind):
    """The torch autoencoder computes the numpy one: code, reconstruction, loss
    and every gradient, to float64 rounding."""
    from voxanim import autoencoder as A
    rng = np.random.default_rng(5)
    x = rng.random((6, 784)) * (rng.random((6, 784)) > 0.5)
    P = A.init(kind, 3, np.random.default_rng(1))
    for k in P:
        if k.startswith("b"):
            P[k] = rng.normal(0, 0.1, P[k].shape)
    f = A.forward(kind, P, x)
    g = A.backward(kind, P, f, x)
    with accel.forced("cpu"):
        Pt = accel.params_to_device(P, dtype=torch.float64)
        xt = accel.to_device(x, dtype=torch.float64)
        z, r = A.torch_forward(kind, Pt, xt)
        gt = dict(zip(Pt, torch.autograd.grad(A.torch_loss(kind, Pt, xt), list(Pt.values()))))
    assert np.allclose(z.detach().numpy(), f["z"], atol=1e-12)
    assert np.allclose(r.detach().numpy(), f["recon"], atol=1e-12)
    for k in g:
        assert close(gt[k].numpy(), g[k]), k


CASES = [dict(), dict(activation="relu", init="he"), dict(activation="gelu"),
         dict(activation="silu"), dict(activation="elu"), dict(activation="sigmoid"),
         dict(activation="leaky_relu"), dict(norm="batch"), dict(norm="layer"), dict(residual=1),
         dict(residual=2), dict(dropout=0.3), dict(weight_decay=0.02, l1=0.01),
         dict(label_smoothing=0.2), dict(norm="batch", residual=1, dropout=0.2)]


@pytest.mark.parametrize("opts", CASES)
def test_deepnet_port(opts):
    """The torch deepnet's objective and gradients are the numpy model's,
    option by option (dropout with the same masks)."""
    from voxanim import deepnet as D
    from voxanim.deepnet_torch import forward_t, loss_t
    spec = D.Spec((7, 6, 6, 6, 4), **opts)
    rng = np.random.default_rng(2)
    P = D.init(spec, rng)
    for k in P:
        if k.startswith(("b", "beta")):
            P[k] = rng.normal(0, 0.3, P[k].shape)
    x = rng.normal(0, 1, (9, 7))
    y = np.eye(4)[rng.integers(0, 4, 9)]
    masks = ([np.random.default_rng(3).random((9, 6)) >= spec.dropout for _ in range(3)]
             if spec.dropout else None)
    f = D.forward(spec, P, x, train=True, masks=masks, stats={})
    G, _ = D.backward(spec, P, f, y)
    with accel.forced("cpu"):
        Pt = accel.params_to_device(P, dtype=torch.float64)
        mt = [torch.tensor(m, dtype=torch.float64) for m in masks] if masks else None
        obj, _ce = loss_t(spec, Pt, forward_t(spec, Pt, torch.tensor(x), mt, {}, True),
                          torch.tensor(y))
        gt = dict(zip(Pt, torch.autograd.grad(obj, list(Pt.values()))))
    assert obj.item() == pytest.approx(D.objective(spec, P, f, y), abs=1e-12)
    for k in G:
        assert close(gt[k].numpy(), G[k]), k


@pytest.mark.parametrize("opts,kw", [
    (dict(norm="batch", residual=1, dropout=0.2), dict(opt="adam", lr=1e-3)),
    (dict(l1=1e-3, l1_prox=True, max_norm=1.0, label_smoothing=0.1), dict(opt="sgd", lr=0.1))])
def test_deepnet_training_loop(opts, kw, tmp_path, monkeypatch):
    """A whole recorded run through the torch loop (float64, CPU) is the numpy
    run: the same draws, weights, records, probes and batch norm statistics.
    (Hidden biases under batch norm get no gradient, only rounding noise, which
    Adam's normalised step turns into ~1e-10 of drift: hence the atol.)"""
    from voxanim import deepnet as D
    rng = np.random.default_rng(0)
    data = D.Data(rng.random((120, 12)), rng.integers(0, 3, 120), rng.random((40, 12)),
                  rng.integers(0, 3, 40), classes=3)
    spec = D.Spec((12, 10, 10, 3), "relu", "he", **opts)
    common = dict(steps=40, batch=16, every=10, probe_layer=1, sparse=True, weights=("W0",), **kw)
    runs = {}
    for mode in ("numpy", "cpu"):
        monkeypatch.setenv(accel.FLOAT_ENV, "64")
        with accel.forced(mode):
            runs[mode] = D.train(spec, data, cache_dir=str(tmp_path / mode), **common)
    a, b = runs["numpy"], runs["cpu"]
    for k in a.P:
        assert np.allclose(a.P[k], b.P[k], rtol=1e-8, atol=1e-8), k
    for k in ("train_loss", "test_loss", "test_acc", "act_std", "grad_rms", "batch_loss"):
        assert np.allclose(getattr(a, k), getattr(b, k), rtol=1e-8, atol=1e-10), k
    for k in a.probe:
        assert np.allclose(a.probe[k].astype(float), b.probe[k].astype(float), atol=1e-5), k
    for k in a.stats:
        assert np.allclose(a.stats[k], b.stats[k], rtol=1e-8, atol=1e-10), k


def test_float64_refused_on_metal(monkeypatch):
    """Apple GPUs have no float64."""
    monkeypatch.setenv(accel.FLOAT_ENV, "64")
    with accel.forced("mps"), pytest.raises(RuntimeError):
        accel.float_dtype()


# ------------------------------------------------------------------ rendering


def test_gpu_framebuffer_matches_numpy():
    """The torch framebuffer (on the CPU here, as on a GPU) draws a frame of
    voxels, lines and quads within one 8-bit level of the numpy one, and auto
    leaves frames below 8K on numpy, bit for bit."""
    from voxanim import presets
    from voxanim.raster import Framebuffer, to_uint8
    from voxanim.scene import Renderer, Style, framebuffer
    anim = presets.build("conv2d")
    frames = {}
    for mode in ("numpy", "cpu"):
        with accel.forced(mode):
            frames[mode] = Renderer(anim.cam, Style(ss=1)).render_at(anim.scene, 9.0, 1 / 30)
    assert np.abs(frames["numpy"] - frames["cpu"]).max() < 1e-3
    d8 = np.abs(to_uint8(frames["numpy"]).astype(int) - to_uint8(frames["cpu"]).astype(int))
    assert d8.max() <= 1
    from voxanim import raster_metal
    with accel.forced("auto"):
        kind = type(framebuffer(1920, 1080, 960)).__name__
    # auto: the Metal kernel on an Apple GPU, otherwise numpy below 8K
    assert kind == ("MetalFramebuffer" if raster_metal.available() else "Framebuffer")
    with accel.forced("numpy"):
        assert type(framebuffer(1920, 1080, 7680)) is Framebuffer


# ------------------------------------------------------------------ the Metal kernel

def _metal():
    from voxanim import raster_metal
    if not raster_metal.available():
        pytest.skip("no Apple GPU (the Metal kernel needs one)")
    return raster_metal


class _Cam:
    """The two camera methods a line needs: screen position and depth."""

    def project(self, p):
        return np.asarray(p, np.float64)[..., :2]

    def depth(self, p):
        return np.asarray(p, np.float64)[..., 2]


def _sprites(rng, n, size=9):
    """``n`` random sprites: tinted premul <= coverage, a depth image, and a
    float64 centre depth drawn near its neighbours' (near ties, where float32
    alone would decide differently)."""
    out = []
    for _ in range(n):
        A = (rng.random((size, size)) * (rng.random((size, size)) > 0.2)).astype(np.float32)
        P = (A[..., None] * rng.random((size, size, 3))).astype(np.float32)
        Z = rng.normal(0, 0.5, (size, size)).astype(np.float32)
        zc = np.float64(rng.normal(0, 0.3)) + 1e-9 * rng.normal()
        out.append((int(rng.integers(-4, 40)), int(rng.integers(-4, 30)), P, A, Z,
                    float(rng.choice([1.0, 0.7, 0.35])), zc, bool(rng.random() < 0.7)))
    return out


def _draw(fb, sprites, lines=(), glow=None):
    for x0, y0, P, A, Z, op, zc, zw in sprites:
        fb.blit(x0, y0, P, A, Z, op, zc, zw)
    for p0, p1, w, col, op, bias in lines:
        fb.line(_Cam(), p0, p1, w, col, op, bias)
    if glow is not None:
        x0, y0, data, base, zc = glow
        fb.sprite(x0, y0, data, base, np.ones(3, np.float32), 0.8, zc, True, white=0.3, glow=0.5)


def test_metal_kernel_is_bit_identical():
    """Sprites (opacity, z-writes, float64 depth near-ties), lines and a lit
    sprite, with HDR emission: the kernel's layers equal numpy's bit for bit,
    and the same frame twice is the same bytes."""
    M = _metal()
    from voxanim.raster import Framebuffer
    rng = np.random.default_rng(7)
    sprites = _sprites(rng, 300)
    lines = [((3.2, 4.1, 0.2), (40.7, 25.3, -0.4), 2.5, (0.9, 0.2, 0.1), 0.8, 0.0),
             ((45.0, 2.0, 0.5), (1.0, 33.0, 0.1), 1.2, (0.1, 0.4, 0.9), 1.0, 0.05)]
    A = rng.random((9, 9)).astype(np.float32)
    data = np.stack([A, A * 0.8, A * 0.1, rng.normal(0, 0.3, (9, 9))]).astype(np.float32)
    glow = (10, 12, data, np.array([0.3, 0.6, 0.9], np.float32), np.float64(0.123456789))
    ref = Framebuffer(48, 36, track_coverage=True, track_emission=True)
    _draw(ref, sprites, lines, glow)
    frames = []
    for _ in range(2):
        fb = M.MetalFramebuffer(48, 36, track_coverage=True, track_emission=True,
                                dev=torch.device("mps"))
        _draw(fb, sprites, lines, glow)
        frames.append((fb.rgb, fb.z, *[fb._down(x, 1) for x in fb._composite_all()[2:]]))
    got = frames[0]
    assert np.array_equal(got[0], ref.rgb) and np.array_equal(got[1], ref.z)
    assert np.array_equal(got[2], ref.cov) and np.array_equal(got[3], ref.emit)
    assert all(np.array_equal(a, b) for a, b in zip(*frames))
    for k in (1, 2, 3):
        assert np.array_equal(M.MetalFramebuffer.downsample(fb, k) if k > 1 else got[0],
                              ref.downsample(k))


def test_metal_depth_needs_float64():
    """A sprite a hair (2e-12) behind one already drawn: its float64 depth is
    below the stored float32 depth, so numpy hides it, though in float32 alone
    the two would tie and it would show.  The kernel's two-float depth hides it."""
    M = _metal()
    from voxanim.raster import Framebuffer
    A = np.ones((4, 4), np.float32)
    Z = np.full((4, 4), 1.0, np.float32)
    zc1, zc2 = np.float64(0.5), np.float64(0.5) - 2e-12           # equal in float32
    assert np.float32(1.0 + zc1) == np.float32(1.0 + zc2) and 1.0 + zc2 < 1.0 + zc1
    red, blue = A[..., None] * np.float32([1, 0, 0]), A[..., None] * np.float32([0, 0, 1])
    ref = Framebuffer(4, 4, track_coverage=True)
    fb = M.MetalFramebuffer(4, 4, track_coverage=True, dev=torch.device("mps"))
    for f in (ref, fb):
        f.blit(0, 0, red, A, Z, 1.0, zc1, True)          # opaque: writes depth 1.5 ...
        f.blit(0, 0, blue, A, Z, 0.5, zc2, True)         # ... this one is 2e-12 behind it
    assert np.array_equal(fb.rgb, ref.rgb) and np.array_equal(fb.z, ref.z)
    assert ref.rgb[0, 0, 2] == 0                          # numpy hid the blue one


def test_metal_quads_and_gpu_lines(monkeypatch):
    """Quads (blended in float64 by numpy) agree to float32 rounding; lines
    rasterised on the GPU (large frames) agree within one 8-bit level."""
    M = _metal()
    from voxanim import presets
    from voxanim.raster import to_uint8
    from voxanim.scene import Renderer, Style
    anim = presets.build("conv2d")
    frames = {}
    for mode, gpu_lines in (("numpy", None), ("mps", None), ("mps", 0)):
        if gpu_lines is not None:
            monkeypatch.setattr(M, "GPU_LINES", gpu_lines)
        with accel.forced(mode):
            frames[(mode, gpu_lines)] = Renderer(anim.cam, Style(ss=2)).render_at(
                anim.scene, 30.0, 1 / 30)
    ref = frames[("numpy", None)]
    assert np.abs(frames[("mps", None)] - ref).max() < 1e-6
    d8 = np.abs(to_uint8(frames[("mps", 0)]).astype(int) - to_uint8(ref).astype(int))
    assert d8.max() <= 1


def test_gpucheck_reports():
    """The gpucheck tool compares an animation and reports what it found."""
    _metal()
    from voxanim.gpucheck import check
    (r,) = check(["lesson-max-pool"], frames=2, jobs=1, verbose=False)
    assert r.framebuffer.startswith("metal-kernel") and r.frames == 2
    assert r.max_8bit <= 1 and r.same_8bit >= 1 and r.t_numpy > 0 and r.t_accel > 0


# ------------------------------------------------------------------ the render pipeline

def test_one_pool_renders_the_same_frames(tmp_path):
    """render_many (one pool for every animation) writes the frames that
    render_frames writes, byte for byte, and reports each animation once."""
    from voxanim.render import render_frames, render_many
    from voxanim.scene import Style
    st = Style(ss=1, motion_blur=False)
    names = ["lesson-max-pool", "lesson-pointwise"]
    done = []
    render_many(names, st, 1, 2, lambda n: str(tmp_path / "many" / n), done.append)
    assert sorted(done) == sorted(names)
    for n in names:
        render_frames(n, st, 1, 1, str(tmp_path / "one" / n))
        a = sorted(os.listdir(tmp_path / "many" / n))
        assert a == sorted(os.listdir(tmp_path / "one" / n)) and len(a) > 5
        for f in a:
            many, one = tmp_path / "many" / n / f, tmp_path / "one" / n / f
            assert many.read_bytes() == one.read_bytes()


def test_bulk_voxels_and_line_cache_change_nothing(monkeypatch):
    """Voxels submitted as whole blocks equal voxels submitted one by one, and
    a frame drawn from cached lines equals one drawn from scratch."""
    M = _metal()
    from voxanim import presets
    from voxanim.scene import Renderer, Style
    anim = presets.build("cnn-training-step")
    t = 0.4 * anim.t_done

    def frame():
        return Renderer(anim.cam, Style(ss=2)).render_at(anim.scene, t, 1 / 30)

    with accel.forced("mps"):
        M.pool(torch.device("mps")).lines.clear()
        cold = frame()                                  # every line rasterised
        warm = frame()                                  # every line from the cache
        monkeypatch.delattr(M.MetalFramebuffer, "add_rows")
        one_by_one = frame()                            # the per-voxel path
    assert np.array_equal(cold, warm) and np.array_equal(cold, one_by_one)


def test_metal_downsample_is_numpys():
    """The downsampling kernel adds the samples in numpy's order: same bits."""
    M = _metal()
    from voxanim.raster import box_downsample
    rng = np.random.default_rng(0)
    fb = M.MetalFramebuffer(8, 8, dev=torch.device("mps"))
    for shape, k in (((108, 192, 3), 2), ((101, 77), 2), ((99, 66, 3), 3)):
        a = rng.random(shape, dtype=np.float32)
        got = fb._down_t(torch.from_numpy(a).to("mps"), k).cpu().numpy()
        assert np.array_equal(got, box_downsample(a, k))


def test_probe_only_scenes_have_the_same_probes():
    """Motion blur measures travel from probe-only builds: they register
    exactly the probes a full build does (so the sub-frames are the same)."""
    from voxanim import presets
    for name in ("conv2d", "mlp-9-training-step", "cnn-training-step", "activation-race"):
        anim = presets.build(name)
        for u in np.linspace(0, 1, 15):
            t = float(u * anim.duration)
            a, b = anim.scene(t).probes, anim.scene(t, probes_only=True).probes
            assert a.keys() == b.keys(), (name, t)
            for k in a:
                assert np.array_equal(a[k][0], b[k][0]) and a[k][1] == b[k][1], (name, t, k)
        assert anim.scene(1.0, probes_only=True).voxel_count() == 0


def test_quad_cache_changes_nothing():
    """Diagram quads drawn from the GPU pool's cache equal quads shaded afresh,
    and a shader's key covers what it draws (another colour, another key)."""
    M = _metal()
    from voxanim import presets
    from voxanim.scene import Renderer, Style
    for name in ("mlp-classifier", "gpt2-transformer-block"):
        anim = presets.build(name)
        t = 0.5 * anim.t_done
        with accel.forced("mps"):
            M.pool(torch.device("mps")).quads.clear()
            cold = Renderer(anim.cam, Style(ss=2)).render_at(anim.scene, t, 1 / 30)
            warm = Renderer(anim.cam, Style(ss=2)).render_at(anim.scene, t, 1 / 30)
        assert np.array_equal(cold, warm), name
    from voxanim.diagram import Diagram
    d = Diagram("t", "")
    from voxanim.scene import Scene
    sc = Scene()
    d.disk(sc, d.P(0.5, 0.5), 5.0, (1.0, 0.0, 0.0))
    d.disk(sc, d.P(0.5, 0.5), 5.0, (0.0, 1.0, 0.0))
    assert sc.quads[0].shader.key != sc.quads[1].shader.key



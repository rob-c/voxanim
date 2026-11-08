"""Regression tests for the animation stack.

    python3 tests/test_voxanim.py          # from anim/, no dependencies
    python3 -m pytest tests                # if pytest is installed

They guard the things that silently go wrong in this kind of tooling: the
numbers on screen, the schedule, hidden-voxel culling, loop seams, and
determinism.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import itertools

from voxanim import presets
from voxanim.camera import Camera
from voxanim.conv import ConvSpec, ConvStage, conv_forward
from voxanim.scene import Renderer, Scene, Style
from voxanim.tensor import Block

DRAFT = Style(ss=2, motion_blur=False)


def test_conv_forward_matches_einsum():
    rng = np.random.default_rng(0)
    for spec in (ConvSpec(), ConvSpec(padding=(1, 1, 1, 1), stride=(2, 2)),
                 ConvSpec(groups=2), ConvSpec(groups=8, filters=8),
                 ConvSpec(in_hw=(12, 1), kernel=(3, 1), filters=4)):
        x = rng.normal(size=spec.in_hw + (spec.in_ch,))
        w = rng.normal(size=(spec.filters,) + spec.kernel + (spec.group_ch,))
        got = conv_forward(spec, x, w)[..., 0]
        t, b, l, r = spec.padding
        xp = np.pad(x, ((t, b), (l, r), (0, 0)))
        (kh, kw), (sh, sw) = spec.kernel, spec.stride
        Ho, Wo = spec.out_hw
        want = np.zeros((Ho, Wo, spec.filters))
        for f in range(spec.filters):
            g = f // spec.per_group
            xs = xp[..., g * spec.group_ch:(g + 1) * spec.group_ch]
            for i in range(Ho):
                for j in range(Wo):
                    want[i, j, f] = np.einsum(
                        "abc,abc->", xs[i * sh:i * sh + kh, j * sw:j * sw + kw], w[f])
        assert np.allclose(got, want), spec


def _brute(spec, x, w):
    """Tap-by-tap reference, written independently of conv_forward."""
    H, W, Cin, D = x.shape
    a, b = spec.input_dilation
    t, _bo, l, _r = spec.padding
    Hp, Wp = spec.padded_hw
    xp = np.zeros((Hp, Wp, Cin, spec.padded_d))
    for i in range(H):
        for j in range(W):
            xp[t + i * a, l + j * b, :, spec.pad_d[0]:spec.pad_d[0] + D] = x[i, j]
    (kh, kw), (sh, sw), (dh, dw) = spec.kernel, spec.stride, spec.dilation
    Ho, Wo = spec.out_hw
    y = np.zeros((Ho, Wo, spec.out_ch, spec.out_d))
    for i, j, k in np.ndindex(Ho, Wo, spec.out_d):
        taps = [(i * sh + p * dh, j * sw + q * dw, k * spec.stride_d + r)
                for p in range(kh) for q in range(kw) for r in range(spec.kernel_d)]
        if spec.pool:
            vals = np.array([xp[u, v, :, z] for u, v, z in taps])
            y[i, j, :, k] = vals.max(0) if spec.pool == "max" else vals.mean(0)
            continue
        for f in range(spec.filters):
            g = f // spec.per_group
            for (u, v, z), (p, q, r) in zip(taps, np.ndindex(kh, kw, spec.kernel_d)):
                for c in range(spec.group_ch):
                    y[i, j, f, k] += xp[u, v, g * spec.group_ch + c, z] * w[f, p, q, c, r]
    return y


def test_conv_forward_variants_match_brute_force():
    rng = np.random.default_rng(1)
    for spec in (ConvSpec(in_hw=(9, 9), in_ch=2, filters=2, dilation=(2, 2)),
                 ConvSpec(in_hw=(4, 4), in_ch=2, filters=2, input_dilation=(2, 2),
                          padding=(2, 2, 2, 2)),
                 ConvSpec(in_hw=(5, 5), in_d=5, in_ch=2, filters=2, kernel_d=3,
                          stride=(2, 1), stride_d=2),
                 ConvSpec(in_hw=(8, 8), in_ch=3, kernel=(2, 2), stride=(2, 2), pool="max"),
                 ConvSpec(in_hw=(8, 8), in_ch=3, kernel=(2, 2), stride=(2, 2), pool="avg")):
        x = rng.normal(size=spec.in_hw + (spec.in_ch, spec.in_d))
        w = rng.normal(size=(spec.filters,) + spec.kernel + (spec.group_ch, spec.kernel_d))
        got = conv_forward(spec, x, None if spec.pool else w)
        assert got.shape == spec.out_hw + (spec.out_ch, spec.out_d), spec
        assert np.allclose(got, _brute(spec, x, w)), spec
    # by hand: ones everywhere -> every output counts the taps it read
    s = ConvSpec(in_hw=(4, 4), in_d=4, in_ch=1, filters=1, kernel_d=3)
    y = conv_forward(s, np.ones((4, 4, 1, 4)), np.ones((1, 3, 3, 1, 3)))
    assert y.shape == (2, 2, 1, 2) and np.all(y == 27)


def test_presets_cover_every_output_cell_once():
    for name, p in presets.REGISTRY.items():
        if name.startswith(("mlp-", "cnn-", "attention-", "rnn-", "lstm-", "dnn-",
                            "strided-", "autoencoder-", "vae-", "unet-", "dec-",
                            "idec-", "gpt2-", "gradient-", "decision-")):
            continue                       # not a ConvAnimation
        anim = p.build()
        for st in anim.stages:
            if not isinstance(st, ConvStage):
                anim.scene(st.t_end + 0.01)
                assert st.output.present.all(), name       # a combine step
                continue
            s = st.spec
            Ho, Wo = s.out_hw
            cells = [(x.i, x.j, x.k, x.f) for x in st.steps]
            assert len(cells) == len(set(cells)) == Ho * Wo * s.out_d * s.passes, name
            starts = [x.t for x in st.steps]
            assert all(b > a for a, b in itertools.pairwise(starts)), name
            # every output cell is on screen once the stage has finished
            anim.scene(st.t_end + 0.01)
            assert st.output.present.all(), name


def test_serpentine_order():
    s = ConvSpec(in_hw=(3, 5), kernel=(1, 3), order="serpentine")
    assert s.steps()[:6] == [(0, 0, 0), (0, 1, 0), (0, 2, 0), (1, 2, 0), (1, 1, 0), (1, 0, 0)]


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
    r = Renderer(cam, replace_style(DRAFT, reflection=0.0, shadow=0.0, watermark=False))
    b = Block((5, 6, 7)).place(cam.frac(0.5, 0.5))
    culled, full = Scene(), Scene()
    b.emit(culled, r.cam.view)
    for i, j, k in np.ndindex(b.present.shape):
        full.voxel(b.cell_center(i, j, k), b.base[i, j, k])
    assert len(culled.voxels) < len(full.voxels)
    diff = np.abs(r.render(culled) - r.render(full)).max()
    assert diff < 1.5 / 255, diff


def test_loop_closes():
    """The last frame flows into the first: the reset lands on frame 0."""
    for name in ("conv2d", "conv2d-2-groups", "depthwise-separable"):
        anim = presets.get(name).build()
        r = Renderer(anim.cam, DRAFT)
        a = r.render(anim.scene(0.0))
        b = r.render(anim.scene(anim.duration))
        assert np.abs(a - b).max() < 2.0 / 255, name


def test_deterministic():
    anim = presets.get("conv2d").build()
    r = Renderer(anim.cam, DRAFT)
    a = r.render_at(anim.scene, 3.3, 1 / 30)
    b = Renderer(anim.cam, DRAFT).render_at(anim.scene, 3.3, 1 / 30)
    assert np.array_equal(a, b)


def _mnist_cached():
    from voxanim.scene import DEFAULT_CACHE
    return os.path.exists(os.path.join(DEFAULT_CACHE, "mnist.npz"))


def test_mlp_gradients_match_finite_differences():
    """Every number the network animations draw is the real gradient."""
    if not _mnist_cached():
        print("   (skipped: MNIST not downloaded yet)")
        return
    from voxanim import mnist
    tr = mnist.trace(18)

    def loss(x, Ws):
        a = x
        for k, (W, b) in enumerate(zip(Ws, tr.bs)):
            z = a @ W + b
            a = np.maximum(z, 0) if k < 2 else z
        z = a - a.max()
        return -np.log(np.exp(z[tr.label]) / np.exp(z).sum())

    assert abs(loss(tr.x, tr.Ws) - tr.loss) < 1e-9
    e = 1e-6
    for k in range(3):
        i, j = np.unravel_index(np.argmax(np.abs(tr.dWs[k])), tr.dWs[k].shape)
        Wp = [W.copy() for W in tr.Ws]
        Wm = [W.copy() for W in tr.Ws]
        Wp[k][i, j] += e
        Wm[k][i, j] -= e
        fd = (loss(tr.x, Wp) - loss(tr.x, Wm)) / (2 * e)
        assert abs(fd - tr.dWs[k][i, j]) < 1e-6 * max(1, abs(fd)), (k, fd, tr.dWs[k][i, j])
    # weight gradients of layer 1 vanish exactly where the pixel is 0
    assert np.all(tr.dWs[0][tr.x == 0] == 0) and np.any(tr.dWs[0][tr.x > 0] != 0)
    # and ReLU blocks the error at neurons that were off
    assert np.all(tr.deltas[0][tr.zs[0] <= 0] == 0)


def test_mlp_animations_loop():
    if not _mnist_cached():
        return
    for name in [n for n in presets.REGISTRY if n.startswith(("mlp-", "cnn-", "attention-", "rnn-", "lstm-"))]:
        anim = presets.get(name).build()
        r = Renderer(anim.cam, DRAFT)
        a = r.render(anim.scene(0.0))
        b = r.render(anim.scene(anim.duration))
        assert np.abs(a - b).max() < 3.0 / 255, name


def test_cnn_gradients_match_finite_differences():
    if not _mnist_cached():
        return
    from voxanim import mnist
    tr = mnist.cnn_trace(18)

    def loss(params):
        f = mnist._cnn_forward(params, tr.x[None])
        return -np.log(f["p"][0, tr.label])

    assert abs(loss(tr.params) - tr.loss) < 1e-9
    e = 1e-6
    for (k, j), G in (((0, 0), tr.grad["dWc"]), ((0, 1), tr.grad["dbc"]),
                      ((1, 0), tr.grad["dWd"])):
        for idx in (np.unravel_index(np.argmax(np.abs(G)), G.shape),
                    tuple(np.array(G.shape) // 2)):
            bump = []
            for sgn in (1, -1):
                q = [[a.copy() for a in layer] for layer in tr.params]
                q[k][j][idx] += sgn * e
                bump.append(loss(q))
            fd = (bump[0] - bump[1]) / (2 * e)
            assert abs(fd - G[idx]) < 1e-6 * max(1, abs(fd)), (k, j, idx, fd, G[idx])
    # max-pool routes each window's gradient to exactly one cell (its winner),
    # and ReLU then blocks it where the map was off
    da1 = tr.grad["da1"].reshape(13, 2, 13, 2, -1)
    assert ((da1 != 0).sum(axis=(1, 3)) <= 1).all()
    assert np.all(tr.grad["d1"][tr.fwd["z1"] <= 0] == 0)
    # and one SGD step on the image helps it
    assert mnist.cnn_sgd_step(tr, 0.005).loss < tr.loss


def _fd_check(loss, P, G, keys, e=1e-6, tol=1e-5):
    """Central differences at the largest and a middle entry of each gradient."""
    for k in keys:
        g = np.asarray(G[k])
        for idx in {np.unravel_index(np.argmax(np.abs(g)), g.shape),
                    tuple(np.array(g.shape) // 2)}:
            plus = {a: np.array(b) for a, b in P.items()}
            minus = {a: np.array(b) for a, b in P.items()}
            plus[k][idx] += e
            minus[k][idx] -= e
            fd = (loss(plus) - loss(minus)) / (2 * e)
            assert abs(fd - g[idx]) <= tol * max(1e-3, abs(fd)), (k, idx, fd, g[idx])


def test_attention_gradients_match_finite_differences():
    if not _mnist_cached():
        return
    from voxanim import vit
    tr = vit.trace(15)

    def loss(P):
        return -np.log(vit.forward(P, tr.x[None])["p"][0, tr.label])

    assert abs(loss(tr.P) - tr.loss) < 1e-9
    _fd_check(loss, tr.P, tr.g, vit.NAMES)
    # only the CLS row is read: everything above the heads gets gradient on row 0 only
    assert np.all(tr.d["dO"][1:] == 0) and np.all(tr.d["dA"][:, 1:] == 0)
    assert np.all(tr.d["dQ"][1:] == 0) and np.all(tr.d["dK"][1:] != 0)
    assert vit.sgd_step(tr, 0.001).loss < tr.loss


def test_recurrent_gradients_match_finite_differences():
    from voxanim import sequence as S
    rng = np.random.default_rng(3)
    for kind in ("rnn", "lstm"):
        M = S.MODELS[kind]
        P = {k: np.array(v + rng.normal(0, 0.1, np.shape(v)))
             for k, v in M["init"](rng, M["n"]).items()}
        x, tgt = S.sine_batch(rng, 3)
        g, _ = M["bwd"](P, M["fwd"](P, x), tgt)
        _fd_check(lambda Q: 0.5 * np.mean((M["fwd"](Q, x)["y"] - tgt) ** 2), P, g, list(P))
        tr = S.trace(kind)
        assert S.sgd_step(tr, S.STEP_LR[kind]).loss < tr.loss


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
    st = replace_style(DRAFT, ss=2)
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


def test_layout_qa_clean():
    """No clipped or colliding text, everything inside the broadcast safe areas."""
    from voxanim.qa import check
    bad = [n for n in presets.REGISTRY if check(n, samples=24, verbose=False)]
    assert not bad, bad


def test_model_loader_formats(tmp=None):
    """npz (ours), PyTorch-style (out, in) names, and JSON all load the same."""
    import json
    import tempfile

    from voxanim import model_io
    rng = np.random.default_rng(4)
    layers = [(rng.normal(size=(784, 100)), rng.normal(size=100)),
              (rng.normal(size=(100, 100)), rng.normal(size=100)),
              (rng.normal(size=(100, 10)), rng.normal(size=10))]
    x = rng.random(784)
    want = model_io.forward(layers, x)[2]
    d = tempfile.mkdtemp()
    model_io.save_dense(os.path.join(d, "a.npz"), layers, "standard")
    got, pre = model_io.load_dense(os.path.join(d, "a.npz"))
    assert pre == "standard" and np.allclose(model_io.forward(got, x)[2], want, atol=1e-5)
    # PyTorch nn.Sequential state_dict layout: layers.N.weight is (out, in)
    torch_like = {"net.%d.weight" % (2 * n): W.T for n, (W, _b) in enumerate(layers)}
    torch_like.update({"net.%d.bias" % (2 * n): b for n, (_W, b) in enumerate(layers)})
    np.savez(os.path.join(d, "b.npz"), **torch_like)
    got, _ = model_io.load_dense(os.path.join(d, "b.npz"))
    assert np.allclose(model_io.forward(got, x)[2], want)
    with open(os.path.join(d, "c.json"), "w") as f:
        json.dump({"layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in layers]}, f)
    got, _ = model_io.load_dense(os.path.join(d, "c.json"))
    assert np.allclose(model_io.forward(got, x)[2], want)


def test_strided_classifier_matches_the_model():
    """The animation's convolutions and logits are the trained model's."""
    if not _mnist_cached():
        return
    from voxanim import strided
    from voxanim.mnist import load
    rng = np.random.default_rng(2)
    P = strided.init(rng)
    P["b0"], P["b1"] = rng.normal(0, 0.05, 4), rng.normal(0, 0.05, 8)   # off the ReLU kink
    xtr, ytr, _, _ = load()
    x, y = xtr[:3], np.eye(10)[ytr[:3]]
    g = strided.backward(P, strided.forward(P, x), y)
    _fd_check(lambda Q: -np.mean(np.log((strided.forward(Q, x)["p"] * y).sum(1))), P, g,
              list(P))
    anim = presets.build("strided-cnn-classifier", digit=3)
    tr = strided.trace(3)
    s1, s2, head = anim.stages
    assert np.allclose(s1.raw[..., 0], tr.acts[1]) and np.allclose(s2.raw[..., 0], tr.acts[2])
    assert np.allclose(head.logits, tr.logits) and head.pred == tr.prediction


def test_dnn_eval_matches_the_model():
    if not _mnist_cached():
        return
    from voxanim import mnist, model_io
    layers, _pre = model_io.load_dense(os.path.join(os.path.dirname(__file__), "..", "models",
                                                    "mnist_dnn_100x3.npz"))
    assert [W.shape for W, _ in layers] == [(784, 100), (100, 100), (100, 100), (100, 10)]
    _a, _b, xte, yte = mnist.load()
    _acts, _pre, p = mnist._forward([[W, b] for W, b in layers], xte[:200])
    acc = (p.argmax(1) == yte[:200]).mean()
    assert acc > 0.95, acc
    anim = presets.build("dnn-eval", digit=8)
    assert np.allclose(anim.p, p[8]) and anim.ev.prediction == int(p[8].argmax())


def test_new_animations_loop():
    if not _mnist_cached():
        return
    for name in ("strided-cnn-classifier", "dnn-eval"):
        anim = presets.build(name)
        r = Renderer(anim.cam, DRAFT)
        a = r.render(anim.scene(0.0))
        b = r.render(anim.scene(anim.duration))
        assert np.abs(a - b).max() < 3.0 / 255, name


def test_text_is_not_clipped():
    """Text composited in its own little box must match Pillow drawing the
    same text on the whole frame (a tight box once cut the tops off 7s)."""
    from PIL import Image as Im
    from PIL import ImageDraw as Dr

    from voxanim.scene import Text, font
    anim = presets.get("conv2d").build()
    r = Renderer(anim.cam, replace_style(DRAFT, watermark=False))
    bg = np.full((540, 960, 3), 0.87, np.float32)
    for size in (0.029, 0.033, 0.054):
        for anchor in ("la", "mt", "rm", "ls"):
            t = Text(0.3123, 0.4337, "784 fgAjy 10 (p) 7", size, "regular", (0, 0, 0), 1.0,
                     anchor)
            # the halo only touches the backdrop; compare ink coverage
            ours = r._draw_texts(bg.copy(), [t]).mean(-1)
            ref = Im.new("L", (960, 540), 0)
            Dr.Draw(ref).text((t.x * 960, t.y * 540), t.s, font=font("regular", size * 540),
                              fill=255, anchor=anchor)
            ink_ref = np.asarray(ref) > 128
            assert (ours[ink_ref] < 0.45).all(), (size, anchor)
            rows = np.nonzero(ink_ref.any(1))[0]
            got = np.nonzero((ours < 0.45).any(1))[0]
            assert rows.min() == got.min() and rows.max() == got.max(), (size, anchor)


def test_autoencoder_and_rnn2_gradients():
    from voxanim import autoencoder as A
    from voxanim import sequence as S
    rng = np.random.default_rng(1)
    x = rng.random((2, 784))
    for kind in ("dense", "conv"):
        P = A.init(kind, 3, rng)
        P = {k: (v + rng.normal(0, 0.02, v.shape) if k.startswith("b") else v)
             for k, v in P.items()}
        g = A.backward(kind, P, A.forward(kind, P, x), x)
        _fd_check(lambda Q: A.loss(A.forward(kind, Q, x), x), P, g, list(P))
    P = {k: np.array(v + rng.normal(0, 0.1, np.shape(v)))
         for k, v in S.rnn2_init(rng).items()}
    xs, tg = S.sine_batch(rng, 3)
    g = S.rnn2_backward(P, S.rnn2_forward(P, xs), tg)
    _fd_check(lambda Q: 0.5 * np.mean((S.rnn2_forward(Q, xs)["y"] - tg) ** 2), P, g, list(P))
    run = S.rnn2_training(steps=40)
    assert run.losses[-5:].mean() < run.losses[:5].mean()      # it learns


def test_autoencoder_and_batch_animations_loop():
    from voxanim.scene import DEFAULT_CACHE
    names = ["rnn-batch-training"]
    for kind in ("dense", "conv"):
        for lat in (2, 3):
            if os.path.exists(os.path.join(DEFAULT_CACHE, "ae_%s_l%d_e%d_s0.npz"
                                           % (kind, lat, {"dense": 12, "conv": 4}[kind]))):
                names.append("autoencoder-%s-%dd" % (kind, lat))
    for name in names:
        anim = presets.build(name)
        r = Renderer(anim.cam, DRAFT)
        a = r.render(anim.scene(0.0))
        b = r.render(anim.scene(anim.duration))
        assert np.abs(a - b).max() < 3.0 / 255, name


def test_vae_unet_dec_gradients():
    from voxanim import deep_models as D
    rng = np.random.default_rng(2)
    x = rng.random((3, 784))

    def jitter(P):
        return {k: (v + rng.normal(0, 0.02, v.shape) if k.startswith("b") else v)
                for k, v in P.items()}
    P = jitter(D.vae_init(3, rng))
    eps = rng.normal(size=(3, 3))
    _fd_check(lambda Q: D.vae_loss(D.vae_forward(Q, x, eps))[0], P,
              D.vae_backward(P, D.vae_forward(P, x, eps)), list(P))
    for skips in (True, False):
        P = jitter(D.unet_init(rng, skips))
        xn = D.noisy(x, rng)
        _fd_check(lambda Q: float(np.mean((D.unet_forward(Q, xn, skips)["out"] - x) ** 2)), P,
                  D.unet_backward(P, D.unet_forward(P, xn, skips), x, skips), list(P))
    z, mu = rng.normal(size=(5, 3)), rng.normal(size=(10, 3))
    q, k = D.soft_assign(z, mu)
    p = D.target(q)
    dz, dmu = D.kl_grads(z, mu, p, q, k)
    _fd_check(lambda Q: D.kl_loss(p, D.soft_assign(Q["z"], Q["mu"])[0]), {"z": z, "mu": mu},
              {"z": dz, "mu": dmu}, ["z", "mu"])
    assert np.allclose(p.sum(1), 1) and (p.max(1) >= q.max(1) - 1e-12).all()   # p is sharper


def test_new_model_animations_loop():
    from voxanim.scene import DEFAULT_CACHE
    need = {"vae-dense-3d": "vae_l3_e15_s0.npz", "unet-dense": "unet_skips_e8_s0.npz",
            "dec-dense": "dec_i1500_s0.npz", "idec-dense": "idec_i1500_s0.npz"}
    for name, f in need.items():
        if not os.path.exists(os.path.join(DEFAULT_CACHE, f)):
            continue
        anim = presets.build(name)
        r = Renderer(anim.cam, DRAFT)
        a = r.render(anim.scene(0.0))
        b = r.render(anim.scene(anim.duration))
        assert np.abs(a - b).max() < 3.0 / 255, name


def _gpt2_cached():
    from voxanim import gpt2
    return os.path.exists(os.path.join(gpt2.DIR, "model.safetensors"))


def test_gpt2_numpy_matches_known_behaviour():
    """The numpy GPT-2 tokenises like GPT-2, predicts what GPT-2 small predicts,
    and layer 5 head 12 is the well-known previous-token head."""
    if not _gpt2_cached():
        return
    from voxanim import gpt2
    tok = gpt2.Tokenizer()
    ids = tok.encode("The cat sat on the")
    assert ids == [464, 3797, 3332, 319, 262]
    assert tok.decode(ids) == "The cat sat on the"
    tr = gpt2.run("One, two, three,")
    assert tr.top[0][0] == " four" and 0.4 < tr.top[0][1] < 0.5
    a = tr.attn[4, 11]
    assert all(a[i, i - 1] > 0.95 for i in range(1, len(tr.ids)))
    assert np.allclose(np.triu(tr.attn, 1), 0)                 # causal
    assert np.allclose(tr.attn.sum(-1), 1, atol=1e-5)


def test_gpt2_panel_mips_are_per_axis():
    from voxanim.gpt2_anim import Panel
    p = Panel(np.random.default_rng(0).random((6, 3072, 3)).astype(np.float32), (0, 0, 0))
    assert p.mip(0, 5).shape == (6, 96, 3)           # columns averaged, the 6 rows kept
    assert p.mip(1, 0).shape == (3, 3072, 3)
    assert np.allclose(p.mip(0, 1)[:, 0], p.tex[:, :2].mean(1))


def test_gpt2_animation_loops():
    if not _gpt2_cached():
        return
    anim = presets.build("gpt2-attention")
    r = Renderer(anim.cam, DRAFT)
    a = r.render(anim.scene(0.0))
    b = r.render(anim.scene(anim.duration))
    assert np.abs(a - b).max() < 3.0 / 255


def test_xor_gradients_match_finite_differences():
    from voxanim import gd_anim as G
    rng = np.random.default_rng(1)
    p = rng.normal(0, 1, 13)
    for x, y in zip(G.X, G.Y):
        g, _ = G.grads(p, x, y)
        num = np.zeros_like(p)
        for i in range(len(p)):
            e = np.zeros_like(p)
            e[i] = 1e-6
            L = [0.5 * (G.forward(p + s * e, x)[3] - y) ** 2 for s in (1, -1)]
            num[i] = (L[0] - L[1]) / 2e-6
        assert np.allclose(g, num, atol=1e-7), (g, num)
    hist, losses = G.train()
    yh = [G.forward(hist[-1], x)[3] for x in G.X]
    assert losses[-1] < 0.01 and np.all(np.round(yh) == G.Y)    # it learns XOR


def test_decision_tree_splits():
    from voxanim import dtree as D
    X, y, root, order = D.model()
    for n in order:                        # every split lowers the impurity, sizes add up
        assert n.split_gini < n.gini
        assert sum(len(c.idx) for c in n.children) == len(n.idx)
        f, t = n.feature, n.threshold
        assert np.all(X[n.children[0].idx, f] <= t) and np.all(X[n.children[1].idx, f] > t)
    leaves = [n for n in D.all_nodes(root) if n.leaf]
    assert sum(len(n.idx) for n in leaves) == len(y)
    # brute force agrees with the root's choice
    best = min((D.gini(y[X[:, f] <= t]) * (X[:, f] <= t).sum() +
                D.gini(y[X[:, f] > t]) * (X[:, f] > t).sum(), f, t)
               for f in (0, 1) for t in D._sweep(X, y, f)[0]
               if min((X[:, f] <= t).sum(), (X[:, f] > t).sum()) >= D.MIN_LEAF)
    assert best[1] == root.feature and abs(best[2] - root.threshold) < 1e-12


def test_diagram_animations_loop():
    names = ["gradient-descent", "decision-tree"]
    if _mnist_cached():
        names.append("mlp-classifier")
    for name in names:
        anim = presets.build(name)
        r = Renderer(anim.cam, DRAFT)
        a = r.render(anim.scene(0.0))
        b = r.render(anim.scene(anim.duration))
        assert np.abs(a - b).max() < 3.0 / 255, name


def test_dither_is_static():
    """The anti-banding dither must not change between frames: a still scene
    has to give identical 8-bit frames (else GIFs balloon to 100+ MB)."""
    from voxanim.raster import to_uint8
    img = np.random.default_rng(0).random((32, 48, 3)).astype(np.float32)
    assert np.array_equal(to_uint8(img), to_uint8(img))


def replace_style(st, **kw):
    from dataclasses import replace
    return replace(st, **kw)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print("ok    ", name)
            except Exception as e:
                failed += 1
                print("FAIL  ", name, "--", type(e).__name__, e)
    sys.exit(1 if failed else 0)

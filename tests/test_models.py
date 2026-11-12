"""The models behind the animations: gradients against finite differences,
loaders, and known behaviour."""

from __future__ import annotations

import os

import numpy as np
import pytest
from conftest import fd_check, needs_gpt2, needs_mnist

from voxanim import presets


@needs_mnist
def test_mlp_gradients_match_finite_differences():
    """Every number the network animations draw is the real gradient."""
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


@needs_mnist
def test_cnn_gradients_match_finite_differences():
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


@needs_mnist
def test_attention_gradients_match_finite_differences():
    from voxanim import vit
    tr = vit.trace(15)

    def loss(P):
        return -np.log(vit.forward(P, tr.x[None])["p"][0, tr.label])

    assert abs(loss(tr.P) - tr.loss) < 1e-9
    fd_check(loss, tr.P, tr.g, vit.NAMES)
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
        fd_check(lambda Q, M=M, x=x, tgt=tgt: 0.5 * np.mean((M["fwd"](Q, x)["y"] - tgt) ** 2), P,
                 g, list(P))
        tr = S.trace(kind)
        assert S.sgd_step(tr, S.STEP_LR[kind]).loss < tr.loss


def test_continue_training_is_self_consistent():
    """The recorded run agrees with itself: a step's stored weights reproduce
    its stored predictions and loss; and training on does help."""
    from voxanim import sequence as S
    x, tgt = S.example()
    for kind in ("rnn", "lstm"):
        c = S.continue_training(kind)
        M = S.MODELS[kind]
        assert c.y.shape == (S.MORE_STEPS, S.T_STEPS)
        for k in (0, 99, S.MORE_STEPS - 1):
            P = {key: np.asarray(c.P[key][k], np.float64) for key in c.P}
            y = M["fwd"](P, x[None])["y"][0]
            assert np.allclose(y, c.y[k], atol=1e-4)
            assert abs(S.mse_loss(y, tgt) - c.losses[k]) < 1e-4
        tr2 = S.sgd_step(S.trace(kind), S.STEP_LR[kind])
        assert c.losses[-1] < tr2.loss / 4


def test_memory_task_gradients_and_outcome():
    """The masked gradient of the memory task matches finite differences, and
    the trained pair behaves as the animation says: the LSTM answers the cue,
    the RNN does not, and the gradient reaching the cue differs by orders."""
    from voxanim import memory as Mem
    from voxanim import sequence as S
    rng = np.random.default_rng(4)
    for kind in ("rnn", "lstm"):
        M = S.MODELS[kind]
        P = M["init"](rng, 5)
        x, cue = Mem.memory_batch(rng, 3, 6)
        f = M["fwd"](P, x)
        g, _ = Mem.backward_last(kind, P, f, cue)
        fd_check(lambda Q, M=M, x=x, cue=cue: Mem.loss_last(M["fwd"](Q, x)["y"], cue), P, g,
                 list(P))
    rnn, lstm = Mem.trace("rnn"), Mem.trace("lstm", Mem.FORGET_BIAS)
    leaky = Mem.train("lstm", Mem.LEAKY_BIAS)
    # loose bounds: trained on another CPU the weights differ in the last bits,
    # and 8000 Adam steps amplify that (the README's reproducibility caveat)
    assert abs(lstm.y - lstm.cue) < 0.3 and abs(lstm.y2 + lstm.cue) < 0.3
    assert abs(rnn.y - rnn.cue) > 0.5
    assert lstm.losses[-200:].mean() < 0.02
    assert min(rnn.losses[-200:].mean(), leaky.losses[-200:].mean()) > 0.1
    assert np.abs(lstm.d["dc"][0]).max() > 100 * np.abs(rnn.d["dh"][0]).max()


def test_snapshot_steps():
    from voxanim import autoencoder as A
    assert A.snapshot_steps(20, 1.5).tolist() == [0, 1, 2, 3, 4, 6, 9, 13, 19, 20]
    s = A.snapshot_steps(5628)
    assert s[0] == 0 and s[-1] == 5628 and np.all(np.diff(s) > 0) and len(s) < 120


@needs_mnist
@pytest.mark.slow
def test_autoencoder_training_run_matches_the_trained_model():
    """The snapshotted run makes the same draws as train(): its last snapshot
    is the cached model, byte for byte, and the cloud it encodes is the one
    autoencoder-dense-2d shows."""
    from voxanim import autoencoder as A
    run = A.training_run("dense", 2)
    P, _ = A.train("dense", 2)
    assert all(np.array_equal(run.P[k], P[k]) for k in P)
    # trace() encodes in chunks of 200, the run all 600 at once: BLAS sums differ in the last bits
    assert np.allclose(run.cloud[-1], A.trace("dense", 2).cloud, rtol=0, atol=1e-9)
    assert run.steps[0] == 0 and run.steps[-1] == A.EPOCHS["dense"] * run.steps_per_epoch
    assert run.losses[-1] < run.losses[0] / 4
    assert run.acts[0].shape == (len(run.steps), 3, 784)


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
        fd_check(lambda Q, kind=kind: A.loss(A.forward(kind, Q, x), x), P, g, list(P))
    P = {k: np.array(v + rng.normal(0, 0.1, np.shape(v)))
         for k, v in S.rnn2_init(rng).items()}
    xs, tg = S.sine_batch(rng, 3)
    g = S.rnn2_backward(P, S.rnn2_forward(P, xs), tg)
    fd_check(lambda Q: 0.5 * np.mean((S.rnn2_forward(Q, xs)["y"] - tg) ** 2), P, g, list(P))
    run = S.rnn2_training(steps=40)
    assert run.losses[-5:].mean() < run.losses[:5].mean()      # it learns


def test_vae_unet_dec_gradients():
    from voxanim import deep_models as D
    rng = np.random.default_rng(2)
    x = rng.random((3, 784))

    def jitter(P):
        return {k: (v + rng.normal(0, 0.02, v.shape) if k.startswith("b") else v)
                for k, v in P.items()}
    P = jitter(D.vae_init(3, rng))
    eps = rng.normal(size=(3, 3))
    fd_check(lambda Q: D.vae_loss(D.vae_forward(Q, x, eps))[0], P,
              D.vae_backward(P, D.vae_forward(P, x, eps)), list(P))
    for skips in (True, False):
        P = jitter(D.unet_init(rng, skips))
        xn = D.noisy(x, rng)
        fd_check(lambda Q, xn=xn, skips=skips: float(np.mean((D.unet_forward(Q, xn, skips)["out"]
                                                              - x) ** 2)), P,
                  D.unet_backward(P, D.unet_forward(P, xn, skips), x, skips), list(P))
    z, mu = rng.normal(size=(5, 3)), rng.normal(size=(10, 3))
    q, k = D.soft_assign(z, mu)
    p = D.target(q)
    dz, dmu = D.kl_grads(z, mu, p, q, k)
    fd_check(lambda Q: D.kl_loss(p, D.soft_assign(Q["z"], Q["mu"])[0]), {"z": z, "mu": mu},
              {"z": dz, "mu": dmu}, ["z", "mu"])
    assert np.allclose(p.sum(1), 1) and (p.max(1) >= q.max(1) - 1e-12).all()   # p is sharper


def test_xor_gradients_match_finite_differences():
    from voxanim import gd_anim as G
    rng = np.random.default_rng(1)
    p = rng.normal(0, 1, 13)
    for x, y in zip(G.XOR_X, G.XOR_Y):
        g, _ = G.grads(p, x, y)
        num = np.zeros_like(p)
        for i in range(len(p)):
            e = np.zeros_like(p)
            e[i] = 1e-6
            L = [0.5 * (G.forward(p + s * e, x)[3] - y) ** 2 for s in (1, -1)]
            num[i] = (L[0] - L[1]) / 2e-6
        assert np.allclose(g, num, atol=1e-7), (g, num)
    hist, losses = G.train()
    yh = [G.forward(hist[-1], x)[3] for x in G.XOR_X]
    assert losses[-1] < 0.01 and np.all(np.round(yh) == G.XOR_Y)    # it learns XOR


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


@needs_mnist
def test_strided_classifier_matches_the_model():
    """The animation's convolutions and logits are the trained model's."""
    from voxanim import strided
    from voxanim.mnist import load
    rng = np.random.default_rng(2)
    P = strided.init(rng)
    P["b0"], P["b1"] = rng.normal(0, 0.05, 4), rng.normal(0, 0.05, 8)   # off the ReLU kink
    xtr, ytr, _, _ = load()
    x, y = xtr[:3], np.eye(10)[ytr[:3]]
    g = strided.backward(P, strided.forward(P, x), y)
    fd_check(lambda Q: -np.mean(np.log((strided.forward(Q, x)["p"] * y).sum(1))), P, g,
              list(P))
    anim = presets.build("strided-cnn-classifier", digit=3)
    tr = strided.trace(3)
    s1, s2, head = anim.stages
    assert np.allclose(s1.raw[..., 0], tr.acts[1]) and np.allclose(s2.raw[..., 0], tr.acts[2])
    assert np.allclose(head.logits, tr.logits) and head.pred == tr.prediction


@needs_mnist
def test_dnn_eval_matches_the_model():
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


@needs_gpt2
def test_gpt2_numpy_matches_known_behaviour():
    """The numpy GPT-2 tokenises like GPT-2, predicts what GPT-2 small predicts,
    and layer 5 head 12 is the well-known previous-token head."""
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

"""The maths animations: the autodiff engine against finite differences, the
softmax / cross-entropy gradient p − y, the dead-ReLU claims, and the three
presets' durations."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import data_marks, needs_gpt2, needs_mnist

from voxanim import autodiff as A
from voxanim import softmax_anim as S

PRESETS = ("autodiff-graph", "gpt2-softmax-cross-entropy", "activations")


# ------------------------------------------------------------------ autodiff


def _loss(**kw):
    return A.neuron_loss(**kw).L.data


def test_autodiff_matches_finite_differences():
    """Reverse mode gives the same gradient as central differences, for every
    leaf, at the worked example and at random points."""
    rng = np.random.default_rng(0)
    params, inputs = A.neuron_example()
    points = [dict(params, **inputs)]
    for _ in range(5):
        points.append(dict(zip(("w1", "w2", "b", "x1", "x2", "y"), rng.normal(0, 1.5, 6))))
    for pt in points:
        g = A.neuron_loss(**pt)
        A.backward(g.L)
        for k in ("w1", "w2", "b", "x1", "x2", "y"):
            fd = A.numeric_gradient(_loss, pt, k)
            assert abs(fd - g[k].grad) < 1e-6 * max(1.0, abs(fd)), (k, fd, g[k].grad)


def test_autodiff_steps_are_the_chain_rule():
    """Every step multiplies the incoming gradient by a local derivative, and
    the parent gradients are the sums of the products into them."""
    params, inputs = A.neuron_example()
    g = A.neuron_loss(**params, **inputs)
    steps = A.backward(g.L)
    assert len(steps) == 12 and g.L.grad == 1.0
    got = dict.fromkeys(g.nodes, 0.0)
    got["L"] = 1.0
    for st in steps:
        assert st.product == st.incoming * st.local
        got[st.parent.name] += st.product
    for k, v in g.nodes.items():
        assert got[k] == pytest.approx(v.grad, abs=1e-15), k
    # children first: a node's gradient is complete before it is passed on
    order = [st.node.name for st in steps]
    for st in steps:
        assert all(order.index(c.name) < order.index(st.node.name) or c.name == st.node.name
                   for c in g.nodes.values() if st.node in c.parents)


def test_autodiff_example_agrees_with_gradient_descent():
    """The worked example is gd_anim's hidden neuron h1 on x = (1, 0), y = 1:
    the forward numbers agree; one step lowers the loss; w2 gets exactly 0."""
    from voxanim import gd_anim as G
    params, inputs = A.neuron_example()
    g = A.neuron_loss(**params, **inputs)
    z, h, _o, _yh = G.forward(G.train()[0][0], G.XOR_X[G.EX])
    assert g["z"].data == pytest.approx(z[0]) and g["a"].data == pytest.approx(h[0])
    assert inputs == dict(x1=1.0, x2=0.0, y=1.0)
    grads = A.gradients(params, inputs)
    assert grads["w2"] == 0.0 and grads["w1"] == pytest.approx(grads["b"])
    after = A.sgd_step(params, inputs)
    assert A.neuron_loss(**after, **inputs).L.data < g.L.data
    # the numbers the subtitles quote
    assert round(g.L.data, 3) == 0.110 and round(grads["w1"], 3) == -0.117
    assert round(A.neuron_loss(**after, **inputs).L.data, 3) == 0.063


# ------------------------------------------------------------------ softmax / cross-entropy


def test_softmax_cross_entropy_gradient_is_p_minus_y():
    """d(−log softmax(z)[k]) / dz = p − y, against central differences; the
    gradients sum to 0; softmax is shift invariant."""
    rng = np.random.default_rng(1)
    for n in (10, 300):
        z = rng.normal(0, 3, n) - 90.0                   # GPT-2-sized offsets too
        k = int(rng.integers(n))
        p = S.softmax(z)
        g = S.grad_logits(p, k)
        assert abs(g.sum()) < 1e-12
        for j in {k, int(np.argmax(p)), 0, n - 1}:
            e = 1e-6
            zp, zm = z.copy(), z.copy()
            zp[j] += e
            zm[j] -= e
            fd = (S.cross_entropy(S.softmax(zp), k) - S.cross_entropy(S.softmax(zm), k)) / (2 * e)
            assert abs(fd - g[j]) < 1e-6 * max(1.0, abs(fd)), (n, j, fd, g[j])
        assert np.allclose(S.softmax(z + 123.4), p)
        assert p.sum() == pytest.approx(1.0)


def test_temperature_and_entropy():
    """Melting raises the entropy, sharpening lowers it; the bounds hold."""
    z = np.array([2.0, 1.0, 0.5, -1.0, -3.0])
    h1, h2, h05 = (S.entropy(S.softmax(z, T)) for T in (1.0, S.T_MELT, S.T_SHARP))
    assert h05 < h1 < h2 < np.log(len(z))
    assert S.entropy(np.array([1.0, 0.0, 0.0])) == 0.0


@needs_mnist
def test_mlp_logits_numbers():
    """The MLP panel's numbers: a 3 at p = 0.62, loss 0.48, an 8 next."""
    m = S.mlp_logits()
    p = m.p()
    assert m.target == 3 and int(p.argmax()) == 3
    assert round(float(p[3]), 2) == 0.62 and round(S.cross_entropy(p, 3), 2) == 0.48
    assert int(np.argsort(-p)[1]) == 8
    assert S.grad_logits(p, 3)[3] == pytest.approx(p[3] - 1)


@needs_gpt2
def test_gpt2_logits_numbers():
    """The GPT-2 panel's numbers: 50,257 logits, ' Paris' ranked fifth at
    p = 0.032, loss 3.43, the top ten holding 0.36."""
    m = S.gpt2_top()
    p = m.p()
    assert m.n == 50257 and len(m.rows) == 10 and m.row_of_target == 4
    assert m.labels[m.row_of_target] == "Paris" and m.labels[0] == "the"
    assert round(float(p[m.target]), 3) == 0.032 and round(S.cross_entropy(p, m.target), 2) == 3.43
    assert round(float(p[m.rows].sum()), 2) == 0.36
    assert S.entropy(m.p(S.T_SHARP)) < S.entropy(p) < S.entropy(m.p(S.T_MELT)) < np.log(m.n)


# ------------------------------------------------------------------ activations


def test_activation_derivatives():
    """Each drawn derivative matches central differences of its function
    (ReLU away from 0, where it jumps)."""
    from voxanim import activations_anim as AC
    z = np.linspace(-4.3, 4.3, 87)
    z = z[np.abs(z) > 1e-3]                            # ReLU' jumps at 0
    e = 1e-6
    for _k, _name, f, df, _yr, _s in AC.FUNCTIONS:
        fd = (f(z + e) - f(z - e)) / (2 * e)
        assert np.allclose(fd, df(z), atol=1e-6), _k
    assert AC.d_relu(np.array([-1e-9, 0.0, 1e-9])).tolist() == [0.0, 0.0, 1.0]


@needs_mnist
def test_layer_preactivations_claims():
    """The real layer: 64 pre-activations, some negative (ReLU zeros them exactly),
    and the saturation counts the subtitles quote are what the functions give."""
    from voxanim import activations_anim as AC
    z, label = AC.layer_preactivations()
    assert z.shape == (64,) and label == 3
    neg = int((z < 0).sum())
    assert 0 < neg < 64
    assert int((AC.relu(z) == 0).sum()) == neg and int((AC.gelu(z) < 0).sum()) == neg
    assert int((np.abs(AC.d_sigmoid(z)) < AC.SATURATED).sum()) > 0


@needs_mnist
def test_dead_relus():
    """At lr 1.0 a good share of the hidden units die within the first 30 steps
    and none revives; the gradient through a dead unit is exactly 0 on every
    test image; at lr 0.3 none dies and the accuracy after one epoch is the same."""
    from voxanim import activations_anim as AC
    from voxanim import mnist
    hot, sane = AC.dead_relu_run(AC.LR_HOT), AC.dead_relu_run(AC.LR_SANE)
    assert 20 <= hot.n_dead <= 40 and sane.n_dead == 0 and sane.counts.max() == 0
    assert hot.counts.max() == hot.n_dead and hot.steps[np.argmax(hot.counts == hot.n_dead)] <= 30
    # dead for good: once dead on the whole test set, dead at every later checkpoint
    D = hot.dead
    assert all(not (D[i] & ~D[i + 1]).any() for i in range(len(D) - 1))
    # exactly zero gradient, recomputed from the weights on the whole test set
    _xtr, _ytr, xte, yte = mnist.load()
    _g, d1 = AC._grads(hot.P, xte, np.eye(10)[yte])
    assert np.all(d1[:, D[-1]] == 0.0) and np.all(np.abs(d1[:, ~D[-1]]).max(0) > 0)
    assert np.all(hot.grad_max[D[-1]] == 0.0) and np.all(hot.z_max[D[-1]] <= 0)
    assert abs(hot.accuracy - sane.accuracy) < 0.01 and hot.accuracy > 0.94


# ------------------------------------------------------------------ the presets


@pytest.mark.parametrize("name", [pytest.param(n, marks=data_marks(n)) for n in PRESETS])
def test_presets_build_and_last_35_to_55_s(name):
    from voxanim import presets
    anim = presets.build(name)
    assert 35.0 <= anim.duration <= 55.0, anim.duration
    assert anim.step_of(0.0) == 0 and anim.step_of(anim.t_done) == len(anim.PHASES) - 1

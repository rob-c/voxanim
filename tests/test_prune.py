"""The trimming family (:mod:`voxanim.prune`, :mod:`voxanim.prune_anim`): masks
and energies as computed, retraining that respects the mask exactly, the
distillation gradient against finite differences, and the claims the four
animations make (as relative claims: CI retrains on another CPU)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fd_check, needs_mnist

from voxanim import prune as PR
from voxanim.mnist import CLASSES, CNN_FILTERS, _cnn_forward, _cnn_grads, _cnn_params, _init

# ------------------------------------------------------------------ small problems


def _mlp(sizes=(12, 8, 6, 5), seed=0):
    """A small random MLP as a dict, with random biases (no ReLU on its kink)."""
    rng = np.random.default_rng(seed)
    P = PR.layers_to_dict(_init(rng, sizes))
    for k in P:
        if k.startswith("b"):
            P[k] = rng.normal(0, 0.3, P[k].shape)
    return P, rng


def _batch(rng, n, d, classes):
    x = rng.random((n, d))
    y = rng.integers(0, classes, n)
    return x, y, np.eye(classes)[y]


# ------------------------------------------------------------------ masks


def test_magnitude_mask_drops_exactly_the_smallest():
    P, _rng = _mlp()
    for f in (0.0, 0.3, 0.5, 0.9):
        m = PR.magnitude_mask(P, f)
        w = np.abs(PR.all_weights(P))
        want = int(round(f * w.size))
        assert PR.count_pruned(m) == (want, w.size)
        kept = np.concatenate([np.abs(P[k][m[k]]).ravel() for k in PR.weight_keys(P)])
        dropped = np.concatenate([np.abs(P[k][~m[k]]).ravel() for k in PR.weight_keys(P)])
        if want:
            assert dropped.max() <= kept.min()
            assert PR.magnitude_threshold(P, f) == kept.min()
        cut = PR.apply_mask(P, m)
        assert all((cut[k][~m[k]] == 0).all() for k in m)
        assert all(np.array_equal(cut[k][m[k]], P[k][m[k]]) for k in m)
        assert all(np.array_equal(cut[k], P[k]) for k in P if k.startswith("b"))
    assert abs(PR.density(PR.magnitude_mask(P, 0.8)) - 0.2) < 2.0 / w.size


def test_random_mask_matches_the_per_layer_density():
    P, rng = _mlp()
    like = PR.magnitude_mask(P, 0.8)
    rand = PR.random_mask(P, like, rng)
    for k in like:
        assert rand[k].shape == like[k].shape and rand[k].sum() == like[k].sum()
    assert any(not np.array_equal(rand[k], like[k]) for k in like)


def test_unit_and_filter_masks_silence_the_unit():
    """A removed hidden unit's activation is exactly 0, so its outgoing weights
    do nothing; a removed conv filter's map is exactly 0."""
    P, rng = _mlp()
    x, _y, _oh = _batch(rng, 5, 12, 5)
    cut = PR.apply_mask(P, PR.unit_masks(P, [1, 4]))
    acts = PR._forward(PR._as_layers(cut, 3), x)[0]
    assert (acts[1][:, [1, 4]] == 0).all() and (acts[1][:, [0, 2, 3]] != 0).any()
    # the outgoing row cannot matter once the unit is silent
    other = dict(cut, W1=cut["W1"] + rng.normal(0, 1, cut["W1"].shape) * np.isin(
        np.arange(8), [1, 4])[:, None])
    assert np.allclose(PR.mlp_logits(other, x), PR.mlp_logits(cut, x))
    Pc = dict(Wc=rng.normal(0, 0.5, (3, 3, CNN_FILTERS)), bc=rng.normal(0, 0.1, CNN_FILTERS),
              Wd=rng.normal(0, 0.1, (676, CLASSES)), bd=np.zeros(CLASSES))
    cutc = PR.apply_mask(Pc, PR.filter_masks(Pc, [2]))
    a1 = _cnn_forward(_cnn_params(cutc), rng.random((3, 784)))["a1"]
    assert (a1[..., 2] == 0).all() and (a1[..., 0] != 0).any()


# ------------------------------------------------------------------ energies


def test_energies_are_mean_squared_activations():
    P, rng = _mlp((12, 8, 6, 5))
    x = rng.random((2500, 12))                       # more than one evaluation chunk
    h = PR._forward(PR._as_layers(P, 3), x)[0][1]
    assert np.allclose(PR.unit_energy(P, x), (h * h).mean(0))
    Pc = dict(Wc=rng.normal(0, 0.5, (3, 3, CNN_FILTERS)), bc=np.zeros(CNN_FILTERS),
              Wd=rng.normal(0, 0.1, (676, CLASSES)), bd=np.zeros(CLASSES))
    xi = rng.random((1200, 784))
    a1 = _cnn_forward(_cnn_params(Pc), xi)["a1"]
    assert np.allclose(PR.filter_energy(Pc, xi), (a1 * a1).mean(axis=(0, 1, 2)))


# ------------------------------------------------------------------ retraining under a mask


def test_fit_masked_keeps_pruned_weights_exactly_zero():
    P, rng = _mlp()
    x, y, _oh = _batch(rng, 300, 12, 5)
    masks = PR.magnitude_mask(P, 0.7)
    masks.update(PR.unit_masks(P, [3]))              # a structured mask on top, with its bias
    Q = PR.apply_mask(P, masks)
    before = {k: v.copy() for k, v in Q.items()}
    steps = []
    PR.fit_masked(Q, PR._mlp_grads, x, y, 40, rng, masks, batch=32, lr=1e-2,
                  record=steps.append, every=10, classes=5)
    assert steps == [0, 10, 20, 30, 40]
    for k, m in masks.items():
        assert (Q[k][~m] == 0).all()                 # exactly, not approximately
        assert (Q[k][m] != before[k][m]).any()       # and the rest trained
    assert (Q["W2"] != before["W2"]).any()           # unmasked arrays train too


def test_fit_masked_with_cnn_masks():
    rng = np.random.default_rng(1)
    Pc = dict(Wc=rng.normal(0, 0.5, (3, 3, CNN_FILTERS)), bc=np.zeros(CNN_FILTERS),
              Wd=rng.normal(0, 0.1, (676, CLASSES)), bd=np.zeros(CLASSES))
    masks = PR.filter_masks(Pc, [0, 3])
    Q = PR.apply_mask(Pc, masks)
    x, y, _oh = _batch(rng, 40, 784, CLASSES)
    PR.fit_masked(Q, _cnn_grads, x, y, 5, rng, masks, batch=8, lr=1e-2)
    assert (Q["Wc"][..., [0, 3]] == 0).all() and (Q["bc"][[0, 3]] == 0).all()
    assert (Q["Wc"][..., [1, 2]] != Pc["Wc"][..., [1, 2]]).any()


# ------------------------------------------------------------------ distillation


@pytest.mark.parametrize("alpha", [0.0, 1.0, 0.5])
@pytest.mark.parametrize("T", [1.0, 4.0])
def test_distillation_gradient_matches_finite_differences(alpha, T):
    P, rng = _mlp((12, 8, 5))
    x, _y, oh = _batch(rng, 6, 12, 5)
    zt = rng.normal(0, 3, (6, 5))                    # a confident "teacher"
    G = PR.kd_grads(P, x, oh, zt, T, alpha)
    assert set(G) == set(P)
    fd_check(lambda Q: PR.kd_loss(Q, x, oh, zt, T, alpha), P, G, list(P), tol=1e-5)


def test_distillation_loss_limits():
    """alpha = 0 is the plain cross-entropy; the soft term is 0 when the
    student's logits are the teacher's."""
    P, rng = _mlp((12, 8, 5))
    x, _y, oh = _batch(rng, 6, 12, 5)
    z = PR.mlp_logits(P, x)
    p = PR.softmax(z)
    ce = -np.mean(np.log(np.sum(p * oh, 1)))
    assert np.isclose(PR.kd_loss(P, x, oh, z * 0 + 1, 4.0, 0.0), ce)
    assert np.isclose(PR.kd_loss(P, x, oh, z, 4.0, 1.0), 0.0)
    assert PR.kd_loss(P, x, oh, z + rng.normal(0, 1, z.shape), 4.0, 1.0) > 0


# ------------------------------------------------------------------ the animations' claims


@needs_mnist
def test_magnitude_pruning_claims():
    """Accuracy falls as more is pruned, collapses past 90% without retraining,
    is recovered by the fine-tune, and iterative pruning keeps more at 99%."""
    d = PR.prune_sweep()
    pruned, tuned, it = d["acc_pruned"], d["acc_tuned"], d["acc_iter"]
    dense = float(d["acc_dense"])
    assert list(d["levels"]) == list(PR.LEVELS)
    assert (np.diff(pruned) < 0).all()
    assert pruned[0] > dense - 0.02                              # 50%: barely noticed
    assert pruned[2] < 0.8 and pruned[-1] < 0.2                  # 90%, 99%: collapse
    assert (tuned[1:] > pruned[1:] + 0.02).all()                 # the fine-tune recovers
    assert tuned[2] > 0.93                                       # 90% pruned, fine-tuned
    assert it[-1] > tuned[-1] + 0.1 and it[-2] > tuned[-2]       # iterative wins where it bites
    # the masks the sheets draw are the ones that were trained under
    from voxanim import presets
    anim = presets.build("magnitude-pruning")
    for n, f in enumerate(PR.LEVELS):
        assert np.isclose(anim.threshold(f), PR.magnitude_threshold(anim.P, f))
        assert int(round(f * anim.total)) == int(d["pruned"][n])
    for k in PR.weight_keys(anim.P):
        assert (anim.tuned[k][np.abs(anim.P[k]) < d["thresholds"][2]] == 0).all()
        assert np.mean(anim.iter[k] == 0) > 0.9 or k != "W0"


@needs_mnist
def test_lottery_ticket_claims():
    """In both seeds the ticket beats a random mask of the same density, keeps
    pace with the dense network, and is ahead from the first recorded steps."""
    for seed in (0, 1):
        d = PR.lottery(seed=seed)
        ticket, rand, dense = d["acc_ticket"], d["acc_random"], d["acc_dense"]
        masks = {k[5:]: v for k, v in d.items() if k.startswith("mask_")}
        assert abs(PR.density(masks) - PR.TICKET_DENSITY) < 1e-3
        assert ticket[-1] > rand[-1] + 0.01
        assert ticket[-1] > dense[-1] - 0.01
        assert (ticket[1:5] > rand[1:5]).all() and (ticket[1:5] > dense[1:5]).all()
        # the ticket's weights are the initialisation's where kept, 0 elsewhere
        for k, m in masks.items():
            assert (d["final_ticket_" + k][~m] == 0).all()
            assert (d["final_random_" + k][~d["rand_" + k]] == 0).all()
        P0 = PR.layers_to_dict(_init(np.random.default_rng(seed), PR.SIZES))
        assert all(np.array_equal(P0[k], d["init_" + k]) for k in P0)


@needs_mnist
def test_structured_pruning_claims():
    """Removing a filter without retraining is a disaster the fine-tune mends;
    half the hidden units can go with a fine-tune after each cut."""
    d = PR.structured_sweep()
    fa, ft, ua, ut = d["filter_acc"], d["filter_acc_tuned"], d["unit_acc"], d["unit_acc_tuned"]
    e = d["filter_energy"]
    assert list(d["filter_order"]) == list(np.argsort(e))
    assert list(d["unit_order"]) == list(np.argsort(d["unit_energy"]))
    assert fa[1] < 0.8 and ft[1] > 0.9 and ft[2] > 0.9 and ft[3] > 0.8
    assert (ft[1:] > fa[1:] + 0.2).all()
    h = len(ua) // 2
    assert ua[h] < ua[0] - 0.2 and ut[h] > ua[0] - 0.02
    assert ut[h] > ua[h] + 0.2
    assert ua[-1] < 0.3                                          # one unit left
    # the half-way networks are what the sheets show: half the units silent
    assert int((d["mlp_half_W0"] == 0).all(0).sum()) == h
    assert int((d["cnn_half_Wc"] == 0).all((0, 1)).sum()) == CNN_FILTERS // 2


@needs_mnist
def test_distillation_claims():
    """On 3,000 images the soft targets beat the hard labels; the wrong classes
    carry the dark knowledge; on the whole set the gain is gone."""
    d = PR.distil()
    full = PR.distil(n_train=60000, steps=1500)
    assert float(d["acc_teacher"]) > 0.97
    assert d["acc_soft"][-1] > d["acc_hard"][-1] + 0.005
    assert d["acc_both"][-1] > d["acc_hard"][-1] + 0.005
    gain_small = d["acc_soft"][-1] - d["acc_hard"][-1]
    gain_full = full["acc_soft"][-1] - full["acc_hard"][-1]
    assert gain_small > gain_full + 0.01
    from voxanim.mnist import load
    from voxanim.prune_anim import DS_IMAGE
    _xtr, _ytr, _xte, yte = load()
    z = d["teacher_logits_test"][DS_IMAGE]
    p1, p4 = PR.softmax(z[None])[0], PR.softmax(z[None] / PR.TEMPERATURE)[0]
    label = int(yte[DS_IMAGE])
    assert label == 3 and p1.argmax() == 3 and p4.argmax() == 3
    wrong = [i for i in np.argsort(-p4) if i != label]
    assert wrong[0] == 8 and p4[8] > 0.1 and p4[8] > p1[8]
    assert p1[label] > 0.8 and p4[label] < 0.5

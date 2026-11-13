"""Fine-tuning and transfer (:mod:`voxanim.transfer`): the split, the network's
gradients, freezing and per-layer rates, rehearsal, the learning-rate finder,
and the relative claims the three animations make."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import fd_check, needs_mnist

from voxanim import transfer as T
from voxanim.optim import Adam

# ------------------------------------------------------------------ no data needed


def _small(seed=0):
    rng = np.random.default_rng(seed)
    x = rng.random((5, 784))
    y = np.eye(10)[rng.integers(0, 10, 5)]
    P = T.init(rng, filters=(2, 3))
    for k in ("b1", "b2", "bo"):               # no ReLU sits exactly on its kink
        P[k] = rng.normal(0, 0.2, P[k].shape)
    return x, y, P


def test_gradients_match_finite_differences():
    """Every parameter of the two-conv network, through both pools and ReLUs."""
    x, y, P = _small()
    G = T.grads(P, x, y)
    fd_check(lambda Q: T.cross_entropy(T.forward(Q, x)["p"], y), P, G, list(P))


def test_eval_forward_matches_training_forward():
    """The evaluation path (no pool winners) gives the same probabilities."""
    x, _y, P = _small(1)
    assert np.array_equal(T.forward(P, x)["p"], T.forward(P, x, train=False)["p"])


def test_per_layer_adam_is_adam_and_freezes_exactly():
    """With one rate everywhere PerLayerAdam is Adam bit for bit; a rate of 0
    leaves the array untouched."""
    x, y, P = _small(2)
    Q, R = ({k: v.copy() for k, v in P.items()} for _ in range(2))
    a, b = Adam(Q, 1e-3), T.PerLayerAdam(R, dict.fromkeys(R, 1e-3))
    for _ in range(5):
        g = T.grads(Q, x, y)
        a.step(Q, g)
        b.step(R, g)
    assert all(np.array_equal(Q[k], R[k]) for k in Q)
    S = {k: v.copy() for k, v in P.items()}
    opt = T.PerLayerAdam(S, T.rates(0.0, 1e-3))
    for _ in range(5):
        opt.step(S, T.grads(S, x, y))
    for k in T.EARLY:
        assert np.array_equal(S[k], P[k])
    for k in T.HEAD:
        assert not np.array_equal(S[k], P[k])


def test_lr_analysis_on_a_synthetic_curve():
    """Knee before the minimum at the steepest fall, divergence where the loss
    is back above its start."""
    lrs = np.array([T.range_lr(k) for k in range(T.LR_STEPS)])
    u = np.log10(lrs)
    loss = 2.4 - 2.0 / (1 + np.exp(-(u + 3.0) * 3)) + 3.0 * np.clip(u + 1.0, 0, None) ** 2
    an = T.lr_analysis(lrs, loss, window=1)
    assert an.knee < an.minimum < an.diverge
    assert abs(u[an.knee] + 3.0) < 0.1                    # the sigmoid's steepest point
    assert loss[an.diverge] > loss[0] >= loss[an.diverge - 1]


def test_cosine_restarts_shape():
    lr = T.cosine_restarts(300, 3, 0.1)
    assert lr(0) == 0.1 and lr(100) == 0.1 and lr(200) == 0.1
    assert lr(99) < 0.1 * 0.001 and lr(50) == pytest.approx(0.05)
    one = T.cosine_restarts(300, 1, 0.1)
    assert one(150) == pytest.approx(0.05) and one(299) < 1e-5


def test_centred_mean():
    x = np.arange(10.0)
    assert np.allclose(T.centred_mean(x, 3)[1:-1], x[1:-1])
    assert T.centred_mean(x, 3)[0] == 0.5 and T.centred_mean(x, 3)[-1] == 8.5


# ------------------------------------------------------------------ the data and the runs


@pytest.fixture(scope="module")
def data():
    return T.split()


@needs_mnist
def test_split(data):
    """Old and new digits apart; the few-shot set and the buffer balanced and
    relabelled as stated; the eval set balanced."""
    assert set(np.unique(data.y_old)) == set(T.OLD) and set(np.unique(data.y_new)) == set(T.NEW)
    assert len(data.x_old) + len(data.x_new) == 60000
    assert len(data.x_few) == T.N_FEW * len(T.NEW)
    assert np.array_equal(np.bincount(data.y_few), [T.N_FEW] * len(T.NEW))   # labels 0-4
    assert np.array_equal(np.bincount(data.y_buffer), [T.N_BUFFER] * len(T.OLD))
    assert np.array_equal(np.bincount(data.y_eval), [T.EVAL_PER_DIGIT] * T.CLASSES)
    # the few-shot images are new digits of the training set, relabelled by -5
    assert all(any(np.array_equal(x, xn) for xn in data.x_new[data.y_new == y + T.NEW[0]][:3000])
               for x, y in zip(data.x_few[:3], data.y_few[:3]))


@needs_mnist
def test_pretrained_knows_only_the_old_digits(data):
    pre = T.pretrain(data)
    assert pre.old_acc > 0.95
    assert pre.acc[list(T.NEW)].max() == 0.0                 # never a target: never predicted
    assert pre.steps[-1] == T.PRE["epochs"] * -(-len(data.x_old) // T.PRE["batch"])
    assert pre.W1.shape == (len(pre.steps), 3, 3, T.FILTERS[0])
    assert np.array_equal(pre.W1[-1], T.filters(pre.P["W1"]))


@needs_mnist
def test_transfer_claims(data):
    """Frozen features beat scratch with 100 a class, fine-tuning is at least as
    good as scratch, and the frozen layers are the pretrained ones exactly."""
    pre = T.pretrain(data)
    runs = T.transfer_runs(data)
    for k in T.EARLY:
        assert np.array_equal(runs["frozen"].P[k], pre.P[k])
    assert np.array_equal(runs["frozen"].W1[0], runs["frozen"].W1[-1])
    assert not np.array_equal(runs["finetune"].P["W1"], pre.P["W1"])
    assert runs["frozen"].test_acc > runs["scratch"].test_acc
    assert runs["finetune"].test_acc > runs["scratch"].test_acc
    for r in runs.values():
        assert r.test_acc > 0.9 and r.acc[0] < 0.4 and r.acc[-1] > 0.9
        assert r.steps[-1] == T.STEPS and len(r.acc) == len(r.steps) == len(r.W1)


@needs_mnist
def test_forgetting_claims(data):
    """Plain fine-tuning drops the old digits by a large margin; rehearsal keeps
    most of it; lower rates keep it too, at the price of slower learning; the
    rehearsal minibatches mix as stated."""
    pre = T.pretrain(data)
    runs = T.forgetting_runs(data)
    plain, reh, low = (runs[k] for k in T.REMEDIES)
    assert plain.old(0) == pytest.approx(pre.acc[list(T.OLD)].mean(), abs=0.05)
    assert plain.test_old < pre.old_acc - 0.5                  # catastrophic
    assert plain.test_new > 0.9
    assert reh.test_old > 0.8 * pre.old_acc                    # most of it kept
    assert reh.test_old > plain.test_old + 0.5
    assert reh.test_new > 0.9
    assert low.test_old > plain.test_old + 0.4                 # much more kept ...
    assert low.test_new < plain.test_new - 0.05                # ... at a price
    # the minibatches: all new digits, except with rehearsal, where about
    # REHEARSE / FT_BATCH of the recorded images are old digits
    assert np.isin(plain.shown_y, T.NEW).all() and np.isin(low.shown_y, T.NEW).all()
    old_share = np.isin(reh.shown_y, T.OLD).mean()
    expected = T.REHEARSE / T.FT_BATCH
    assert abs(old_share - expected) < 0.12, old_share
    assert np.isin(reh.shown_y[0], T.OLD).any()


@needs_mnist
def test_lr_finder_claims():
    """The knee lies in Adam's usual decade, below the minimum, with the
    divergence more than ten times higher."""
    lrs, losses = T.lr_range_test()
    an = T.lr_analysis(lrs, losses)
    assert 2e-4 < lrs[an.knee] < 5e-3
    assert an.knee < an.minimum < an.diverge
    assert lrs[an.diverge] > 10 * lrs[an.knee]
    assert an.smooth[an.minimum] < 0.5 < 2.0 < an.smooth[0]
    assert an.smooth[an.diverge] > an.smooth[0]
    assert an.slope[an.knee] == an.slope[:an.minimum].min() < -1.5    # per decade


@needs_mnist
def test_restart_claims():
    """The loss rises after every restart and settles lower; both runs end well
    trained; the landscape walker leaves the floor at each restart and comes back."""
    peak, mlp = T.restart_runs()
    r = mlp["restarts"]
    length = r.steps // T.CYCLES
    assert r.lrs[0] == peak and r.lrs[length] == peak and r.lrs[length - 1] < peak / 1000
    ends = [r.sub_loss[np.searchsorted(r.sub_steps, c * length)] for c in range(1, T.CYCLES + 1)]
    for c in range(1, T.CYCLES):
        at = np.searchsorted(r.sub_steps, c * length)
        # at the knee rate a restart is gentle: the loss stops falling for a
        # while (and rises a little at the later restarts), then ends lower
        assert r.sub_loss[at + 1:at + 7].max() >= r.sub_loss[at]
        assert ends[c] < ends[c - 1]
    at = np.searchsorted(r.sub_steps, (T.CYCLES - 1) * length)
    assert r.sub_loss[at + 1:at + 7].max() > r.sub_loss[at] * 1.02
    assert r.acc > 0.95 and mlp["cosine"].acc > 0.95
    assert np.all(np.diff(r.cycle_acc) > 0)
    run = T.landscape_restarts()
    from voxanim import landscape as LS
    prob = LS.problem()
    lmin = prob.loss(prob.minimum())
    L = T.LAND_STEPS // T.LAND_CYCLES
    for c in range(T.LAND_CYCLES):
        seg = run.loss[c * L:(c + 1) * L + 1]
        assert seg[-1] < lmin + 0.02                                  # settled
        if c:
            assert seg[1:].max() > run.loss[c * L] + 0.03             # kicked out
    assert np.all(run.w >= -7) and np.all(run.w <= 3)                # inside the drawn box


@needs_mnist
@pytest.mark.parametrize("name", ["transfer-learning", "catastrophic-forgetting",
                                  "learning-rate-finder-and-restarts"])
def test_durations_and_phases(name):
    from voxanim import presets
    anim = presets.build(name)
    assert 40 <= anim.duration <= 60, anim.duration
    assert anim.scene(0.0).texts and anim.subtitle

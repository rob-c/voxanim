"""The vision animations (51-58): layout, the numbers they show, and the data
they compute beyond their studies."""

from functools import cache

import numpy as np
import pytest
from conftest import available

from voxanim import mech_vision as mv
from voxanim import next_vision as nv
from voxanim import presets, qa
from voxanim import study_vision as sv
from voxanim.next_catalog import build_study

NAMES = tuple(mv.ANIMATIONS)
pytestmark = pytest.mark.skipif(not all(available(n) for n in NAMES), reason="MNIST not cached")


@cache
def built(name):
    return presets.get(name).build()


@cache
def study(name):
    return build_study("next_vision", name)


def subtitles(anim):
    return [anim.subtitle_for(k) for k in range(len(anim.PHASES))]


def test_all_eight_are_drawn():
    assert NAMES == ("receptive-field", "vit-attention", "clip-contrastive",
                     "adversarial-examples", "object-detection-basics", "semantic-segmentation",
                     "batchnorm-in-cnns", "style-transfer")
    for name in NAMES:
        assert isinstance(built(name), mv.ANIMATIONS[name])


@pytest.mark.parametrize("name", NAMES)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", NAMES)
def test_runs_45_to_60_seconds_and_closes_its_loop(name):
    anim = built(name)
    assert 45 <= anim.t_done <= 60
    first, last = anim.scene(0.0), anim.scene(anim.duration)
    assert [t.s for t in first.texts] == [t.s for t in last.texts]
    assert len(first.quads) == len(last.quads) and len(first.lines) == len(last.lines)
    assert all(s and len(s) < 260 for s in subtitles(anim))


# ---------------------------------------------------------------------- 51
def test_receptive_field_numbers_are_the_studys():
    anim, s = built("receptive-field"), study("receptive-field")
    sizes = [int(v) for v in s.panels[1].data["values"]]
    assert anim.sizes == sizes == [1, 3, 4, 8, 9, 28]
    assert list(nv.receptive_field_sizes([3, 2, 3], [1, 2, 1], [1, 1, 1])) == sizes[:4]
    assert s.panels[3].notes[0] == "nonzero pixels: %d / 784" % anim.nonzero
    assert "non-zero on %d of 784" % anim.nonzero in subtitles(anim)[3]
    assert anim.share[anim.half] >= 0.5 > anim.share[anim.half - 1]


def test_receptive_field_pool_gradient_matches_the_study_and_the_network():
    from voxanim import mnist

    anim, s = built("receptive-field"), study("receptive-field")
    f = mnist._cnn_forward(anim.params, anim.x.reshape(1, 784))
    # the study's own cell, by this module's function
    mine = mv.ReceptiveField.pool_gradient(anim.params, f, 6, 6)
    np.testing.assert_allclose(nv.normalize(mine), s.panels[2].data["values"][1])
    # the cell on screen: a finite difference of the pooled output agrees
    pr, pc = anim.cell
    assert np.count_nonzero(anim.support) == 9 and anim.four.max() == f["pooled"][0, pr, pc, 0]
    r, c = np.argwhere(anim.support > 0)[4]
    x = anim.x.reshape(1, 784).copy()
    x[0, r * 28 + c] += 1e-4
    moved = (mnist._cnn_forward(anim.params, x)["pooled"][0, pr, pc, 0] - anim.four.max()) / 1e-4
    assert abs(moved) == pytest.approx(anim.support[r, c], rel=1e-3)


# ---------------------------------------------------------------------- 52
def test_vit_numbers_are_the_studys():
    anim, s = built("vit-attention"), study("vit-attention")
    reach = [float(v) for v in s.panels[2].data["values"]]
    assert anim.reach == reach[:3] and anim.uniform == reach[3]
    assert anim.uniform == pytest.approx(sv.patch_distance().mean())      # exact, 7 x 7 grid
    assert "%.2f, %.2f and %.2f patch widths" % tuple(reach[:3]) in subtitles(anim)[4]
    read = [float(v) for v in s.panels[3].data["values"]]
    assert anim.read == read[:3] and read[3] == 49
    assert "about %.0f of the 49 patches in block 1, then %.0f and %.0f" % tuple(
        read[:3]) in subtitles(anim)[3]
    assert s.panels[3].notes[0].startswith("test accuracy %.1f%%" % (
        100 * np.mean(anim.accuracy)))
    np.testing.assert_allclose(anim.cls[-1, 0], s.panels[1].data["values"])
    np.testing.assert_allclose(anim.cls.sum(-1), 1.0)
    assert anim.said[0] == 7 and all(anim.has_ink[0][q] for q in anim.queries)


def test_vit_measures_follow_from_the_attention_drawn():
    d = sv.vit_run()
    xt, yt = sv.mnist_data()[2:]
    np.testing.assert_array_equal(d["example"], xt[:sv.VIT_SHOWN])     # held-out digits
    # the stored weights give the stored attention and readings
    logits, attention = sv.vit_loss(sv.part(d, "weights_"), xt[:sv.VIT_SHOWN],
                                    yt[:sv.VIT_SHOWN], False)[2:]
    np.testing.assert_allclose(attention, d["attention"], atol=1e-9)
    np.testing.assert_array_equal(logits.argmax(1), d["read"])
    np.testing.assert_allclose(attention.sum(-1), 1.0)
    # attention spread evenly measures as the baselines drawn beside the bars
    ink = sv.has_ink(xt[:sv.VIT_PROBE])
    even = sv.attention_measures(np.full((1, sv.VIT_PROBE, 1, 50, 50), 1 / 50), ink)
    assert even["reach"][0, 0] == pytest.approx(sv.patch_distance().mean())
    assert even["spread"][0, 0] == pytest.approx(49) == pytest.approx(even["cls_spread"][0, 0])
    assert even["cls_ink"][0, 0] == pytest.approx(float(d["ink_share"]))
    # attention on one neighbouring patch only would measure as near
    near = np.zeros((1, 1, 1, 50, 50))
    for i in range(49):
        near[0, 0, 0, i + 1, (i + 1 if i % 7 < 6 else i - 1) + 1] = 1.0
    near[0, 0, 0, 0, 1] = 1.0
    assert sv.attention_measures(near, ink[:1])["reach"][0, 0] == pytest.approx(1.0)


def test_vit_attention_is_concentrated_but_not_local():
    """What the redesigned study supports, in all three seeds: the class token reads far
    fewer than 49 patches, fewer after block 1; reach stays that of even attention."""
    d = sv.vit_run()
    assert min(d["accuracy"]) > 0.95
    read = np.asarray(d["cls_spread"]).mean(2)                # seed, block
    assert (read < 49 - 12).all()
    assert (read[:, 1:].max(1) < read[:, 0] - 3).all()
    reach = np.asarray(d["reach"])                            # seed, block, head
    assert np.abs(reach - sv.patch_distance().mean()).max() < 0.3      # the null, kept
    assert np.asarray(d["cls_ink"]).mean() > float(d["ink_share"]) + 0.03


# ---------------------------------------------------------------------- 53
def test_clip_numbers_are_the_studys():
    anim, s = built("clip-contrastive"), study("clip-contrastive")
    assert s.panels[3].notes[0] == "test accuracy %.1f%%" % (100 * anim.accuracy)
    assert "%.1f%% of 1,000 images" % (100 * anim.accuracy) in subtitles(anim)[3]
    assert anim.entropy == [float(v) for v in s.panels[3].data["values"]]
    # the temperature phase's arithmetic gives the study's entropies at its five stops
    cosine = anim.scores[0] * anim.TEMP
    for t, want in zip(anim.temps, anim.entropy):
        p = np.exp(cosine / t - (cosine / t).max())
        p /= p.sum()
        assert -np.sum(p * np.log(p + 1e-20)) == pytest.approx(want, abs=1e-9)


def test_clip_rerun_reproduces_the_studys_training():
    s = study("clip-contrastive")
    steps, snaps, losses, _ = mv.clip_run()
    np.testing.assert_allclose(losses, s.panels[1].data["ys"][0])
    np.testing.assert_allclose(snaps[0], s.panels[0].data["values"])
    np.testing.assert_allclose(snaps[list(steps).index(900)], s.panels[2].data["values"])


# ---------------------------------------------------------------------- 54
def test_adversarial_numbers_are_the_studys():
    anim, s = built("adversarial-examples"), study("adversarial-examples")
    fgsm, pgd = np.asarray(s.panels[2].data["ys"], float)
    bars = [float(v) for v in s.panels[3].data["values"]]
    text = subtitles(anim)
    assert "%.1f%% after one step (FGSM) and %.1f%% after twelve" % (
        100 * fgsm[3], 100 * pgd[3]) in text[2]
    assert "rises from %.1f%% to %.1f%%" % (100 * bars[2], 100 * bars[3]) in text[3]
    assert "falls from %.1f%% to %.1f%%" % (100 * bars[0], 100 * bars[1]) in text[3]
    assert (anim.truth, anim.flip) == (7, 3)


def test_adversarial_steps_end_at_the_studys_example_within_budget():
    anim, s = built("adversarial-examples"), study("adversarial-examples")
    np.testing.assert_allclose(anim.traj[-1].reshape(28, 28), s.panels[0].data["values"][1],
                               atol=1e-12)
    np.testing.assert_allclose(anim.probs[0], s.panels[1].data["values"][0], atol=1e-12)
    np.testing.assert_allclose(anim.probs[-1], s.panels[1].data["values"][1], atol=1e-12)
    assert np.abs(anim.traj - anim.clean).max() <= anim.EPS + 1e-12
    assert np.abs(anim.attacked - anim.batch).max() <= anim.EPS + 1e-12


# ---------------------------------------------------------------------- 55
def test_detection_numbers_are_the_studys():
    anim, s = built("object-detection-basics"), study("object-detection-basics")
    assert s.panels[3].notes == ("mean held-out IoU %.3f" % anim.mean_iou,
                                 "kept %d / %d proposals" % (len(anim.keep), len(anim.proposals)))
    assert "averages %.3f" % anim.mean_iou in subtitles(anim)[2]
    np.testing.assert_allclose(anim.scenes[0], s.panels[0].data["values"][0])
    # the boxes drawn for the held-out scenes score what the study recorded
    mine = [[nv.iou(anim.truth[i, j], anim.pred[i, j:j + 1])[0] for j in range(2)]
            for i in range(80)]
    np.testing.assert_allclose(mine, anim.overlaps)


def test_detection_rerun_and_suppression_match_the_study():
    anim, s = built("object-detection-basics"), study("object-detection-basics")
    _steps, path, losses, _ = mv.detector_run()
    np.testing.assert_allclose(losses, s.panels[1].data["ys"][0])
    np.testing.assert_allclose(path[-1], anim.pred[0])
    np.testing.assert_allclose(path[-1], anim.proposals[[0, 3]])
    assert [kept for kept, _ in anim.trace] == anim.keep == [0, 3]
    removed = {j for _, others in anim.trace for j, _, gone in others if gone}
    assert removed == {1, 2, 4, 5}


# ---------------------------------------------------------------------- 56
def test_segmentation_numbers_are_the_studys():
    anim, s = built("semantic-segmentation"), study("semantic-segmentation")
    iou = [float(v) for v in s.panels[3].data["values"]]
    assert "background %.3f, the one %.3f, the seven %.3f" % tuple(iou) in subtitles(anim)[3]
    np.testing.assert_allclose(mv.class_iou(anim.pred, anim.truth), iou)
    np.testing.assert_array_equal(anim.pred[:3], np.round(
        np.asarray(s.panels[2].data["values"]) * 2).astype(int))


def test_segmentation_forward_pass_is_the_studys_and_both_paths_matter():
    from voxanim import next_vision_models as vm

    anim = built("semantic-segmentation")
    prob = vm.unet_loss(anim.p, anim.images, anim.truth)[2]
    np.testing.assert_array_equal(anim.pred, prob.argmax(-1))
    assert anim.h0.shape == (80, 16, 16, 4) and anim.h1.shape == (80, 8, 8, 8)
    for cut in (anim.iou_no_skip, anim.iou_no_coarse):
        assert cut[1] < anim.iou[1] and cut[2] < anim.iou[2]


# ---------------------------------------------------------------------- 57
def test_batchnorm_numbers_are_the_studys():
    anim, s = built("batchnorm-in-cnns"), study("batchnorm-in-cnns")
    text = subtitles(anim)
    assert "filter 1 has mean %.2f and variance %.2f" % (anim.mu[0], anim.var[0]) in text[0]
    np.testing.assert_allclose(anim.stored, np.asarray(s.panels[0].data["values"])[:, :4])
    assert "one image alone gives %.2f" % anim.means[2][1] in text[2]
    accuracy = np.asarray(s.panels[2].data["seeds"], float)
    np.testing.assert_allclose(s.panels[2].data["values"], accuracy.mean(0))
    assert "accuracy falls from %.1f%% to %.1f%%" % (
        100 * accuracy[:, 0].mean(), 100 * accuracy[:, -1].mean()) in text[3]
    assert s.panels[2].data["labels"] == ["stored"] + ["batch of %d" % b for b in anim.sizes]
    np.testing.assert_allclose(anim.changed, s.panels[3].data["seeds"])


def test_batchnorm_statistics_and_maps_are_recomputed_correctly():
    anim = built("batchnorm-in-cnns")
    mean, var, hists = anim.partial[-1]                       # after all 256 images
    np.testing.assert_allclose(mean, anim.mu)
    np.testing.assert_allclose(var, anim.var)
    assert all(0.95 < h.sum() <= 1.0 + 1e-9 for h in hists)      # shares; far tails fall outside
    # normalising with the batch's own statistics: mean 0, variance 1
    z = (anim.train - anim.mu) / np.sqrt(anim.var + 1e-5)
    np.testing.assert_allclose(z.mean((0, 1, 2)), 0, atol=1e-9)
    np.testing.assert_allclose(z.var((0, 1, 2)), 1, atol=1e-3)
    # which is what the network itself does to a batch, and per image to a batch of one
    d = sv.batchnorm_run()
    p = sv.part(d, "weights_")
    x = sv.mnist_data()[0][:256]
    f = sv.cnn_forward(p, x)
    np.testing.assert_allclose(f["hat1"][..., :4], z, atol=1e-9)
    alone = sv.cnn_forward(p, x[:8], batch=1)["hat1"]
    np.testing.assert_allclose(alone.mean((1, 2)), 0, atol=1e-9)
    np.testing.assert_allclose(alone[3:4], sv.cnn_forward(p, x[3:4])["hat1"], atol=1e-9)
    # the stored statistics are a running average of the batches'
    running, batch = np.asarray(d["running_mean"]), np.asarray(d["batch_mean"])
    np.testing.assert_allclose(running[1:], sv.MOMENTUM * running[:-1]
                               + (1 - sv.MOMENTUM) * batch[1:], atol=1e-12)


def test_batchnorm_bug_costs_accuracy_in_every_seed():
    """What the redesigned study supports: with the batch's own statistics at test time,
    large batches lose nothing and a batch of one loses accuracy, in each of three seeds."""
    d = sv.batchnorm_run()
    stored, own = np.asarray(d["stored"]), np.asarray(d["own"])        # seed; seed, batch size
    assert len(stored) == 3 and sv.BN_BATCHES == (256, 32, 8, 2, 1)
    assert (own[:, -1] < stored - 0.005).all()
    assert own[:, -1].mean() < stored.mean() - 0.02
    assert np.abs(own[:, 0] - stored).max() < 0.003
    assert (np.diff(own.mean(0)) < 0.001).all()               # smaller batches: never better
    assert np.asarray(d["one_digit"]).mean() < stored.mean() - 0.02
    # the test images: distinct, from the test split, and scored as stored (first seed)
    x, y, order = sv.bn_test_set()
    assert len(set(order.tolist())) == sv.BN_TEST == len(x)
    p, stats = sv.part(d, "weights_"), sv.part(d, "stats_")
    assert np.mean(sv.cnn_predict(p, x, stats) == y) == pytest.approx(stored[0])
    assert np.mean(sv.cnn_predict(p, x, batch=1) == y) == pytest.approx(own[0, -1])


def test_cnn_gradients_match_finite_differences():
    rng = np.random.default_rng(0)
    for norm, pad in ((True, False), (False, True)):
        p = sv.cnn_init(rng, (3, 4), norm, pad)
        x = rng.random((5, 784))

        def loss(p=p, x=x, pad=pad):
            f = sv.cnn_forward(p, x, pad=pad)
            return 0.5 * (f["logits"] ** 2).sum() + 0.1 * (f["a1"] ** 2).sum(), f
        f = loss()[1]
        g = sv.cnn_backward(p, f, f["logits"], {1: 0.2 * f["a1"]}, pad=pad)
        for key in list(p) + ["x"]:
            target = x if key == "x" else p[key]
            at = tuple(int(rng.integers(0, n)) for n in target.shape)
            old = target[at]
            target[at] = old + 1e-5
            up = loss()[0]
            target[at] = old - 1e-5
            down = loss()[0]
            target[at] = old
            assert g[key].reshape(target.shape)[at] == pytest.approx(
                (up - down) / 2e-5, rel=1e-4, abs=1e-6), (norm, key)


# ---------------------------------------------------------------------- 58
def test_style_numbers_are_the_studys():
    anim, s = built("style-transfer"), study("style-transfer")
    text = subtitles(anim)
    assert "falls from %.2f to %.2f" % (anim.loss[0], anim.loss[-1]) in text[2]
    full = np.asarray(s.panels[1].data["values"], float)
    f = anim.picked
    assert "Filters %d and %d give %.2f" % (f[1] + 1, f[3] + 1, full[f[1], f[3]]) in text[1]
    a = anim.maps[1].reshape(-1, 4)                           # the style digit's maps on screen
    np.testing.assert_allclose(a.T @ a / len(a), anim.gram)
    np.testing.assert_allclose(anim.gram, full[np.ix_(f, f)])
    ink = s.metadata["ink"]
    assert "from %.0f pixels' worth to %.0f" % (ink["content"], ink["thin_result"]) in text[3]
    np.testing.assert_allclose(anim.results[:, 0], np.asarray(s.panels[2].data["values"])[1:])
    np.testing.assert_allclose(anim.mean[1].mean(), np.asarray(
        sv.style_run()["mismatch"])[1].mean())


def test_style_rerun_reproduces_the_studys_optimisation():
    d = sv.style_run()
    p = sv.part(d, "weights_")
    xt = sv.mnist_data()[2]
    # the styles are real test digits, untouched, and none of them is a content digit
    index = [int(v) for v in d["style_index"]]
    np.testing.assert_array_equal(d["style"], xt[index].reshape(-1, 28, 28))
    np.testing.assert_array_equal(d["content"], xt[:sv.CONTENT].reshape(-1, 28, 28))
    assert min(index) >= sv.CONTENT
    weight = sv.stroke_weight(xt[sv.STYLE_POOL[0]:sv.STYLE_POOL[1]])
    assert index == [sv.STYLE_POOL[0] + int(weight.argmin()), sv.STYLE_POOL[0] + int(
        weight.argmax())]
    # the heavy style on the first two content digits, again
    z, mismatch, _, snaps, read = sv.style_optimise(p, xt[:2], xt[index[1]:index[1] + 1])
    np.testing.assert_allclose(z, np.asarray(d["result"])[1, :2], atol=1e-6)
    np.testing.assert_allclose(mismatch, np.asarray(d["mismatch"])[1][:, :2], atol=1e-6)
    np.testing.assert_allclose(snaps[0], xt[:2].reshape(-1, 28, 28))      # starts at the content
    np.testing.assert_array_equal(read, np.asarray(d["read_after"])[1, :2])
    assert z.min() >= 0 and z.max() <= 1


def test_style_is_taken_and_the_digit_kept():
    """What the redesigned study supports, over twenty content digits: each comes out with
    less ink in the thin style and more in the heavy one, the Gram mismatch falls by more
    than half, and the network still reads most results as the same digit."""
    d = sv.style_run()
    ink = lambda v: np.asarray(v, float).reshape(len(v), -1).sum(1)     # noqa: E731
    content, (thin, heavy) = ink(d["content"]), (ink(r) for r in d["result"])
    assert len(content) == 20
    assert (thin < content - 5).all() and (heavy > content + 30).all()
    assert thin.mean() < content.mean() - 30
    mismatch = np.asarray(d["mismatch"])                      # style, step, digit
    assert (mismatch[:, -1] < 0.5 * mismatch[:, 0]).all()
    same = (np.asarray(d["read_after"]) == np.asarray(d["read_before"])).sum(1)
    assert (same >= 16).all()
    assert float(d["accuracy"]) > 0.97

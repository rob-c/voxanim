"""Animations 106-109 (voxanim.mech_labels) and their experiments (voxanim.study_labels):
the layout is clean, the loop closes, the numbers on screen are the study's, each
study's claim holds as a relative claim with a margin over its seeds, and the recorded
arrays are what they say they are (one run of each experiment is repeated)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import mnist_cached, needs_mnist

from voxanim import mech_labels as M
from voxanim import presets, qa
from voxanim import study_labels as S
from voxanim.next_catalog import build_study

NAMES = ("label-embedding", "label-embedding-classifier", "label-embedding-autoencoder",
         "label-embedding-generation")
pytestmark = needs_mnist


def needs():
    if not mnist_cached():
        pytest.skip("MNIST not downloaded")


def study(name):
    return build_study("study_labels", name)


def subtitles(anim):
    return [anim.subtitle_for(k) for k in range(len(anim.PHASES))]


def texts(anim, t):
    return [x.s for x in anim.scene(t).texts]


# ---------------------------------------------------------------------- all four
@pytest.mark.parametrize("name", NAMES)
def test_layout_is_clean(name):
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", NAMES)
def test_drawn_by_this_module_and_loop_closes(name):
    anim = presets.get(name).build()
    assert type(anim) is M.ANIMATIONS[name]
    assert 45.0 <= anim.t_done <= 60.0

    def summary(t):
        sc = anim.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(anim.duration)
    for a, b in anim.spans:                           # every phase draws things, not only text
        assert len(anim.scene((a + b) / 2).quads) >= 3


@pytest.mark.parametrize("name", NAMES)
def test_subtitles_fit_two_lines(name):
    from voxanim import layout

    anim = presets.get(name).build()
    for sub in subtitles(anim):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE


@pytest.mark.parametrize("name", NAMES)
def test_study_is_a_fair_comparison_on_disjoint_data(name):
    s = study(name)
    m = s.metadata
    assert len(s.panels) == 4 and len(m["seeds"]) >= 3 and m["designs_tried"]
    data = m["dataset"]
    assert data["train"]["official_partition"] == "train"
    assert data["held_out"]["official_partition"] == "test"
    assert data["train"]["examples"] == S.N_TRAIN and data["held_out"]["examples"] == S.N_TEST


def test_data_split_and_judge():
    needs()
    from voxanim import mnist, realdata

    x, y, xt, yt, _ = S.data()
    _, _, m = realdata.mnist_split(S.N_TRAIN, "train", 28, S.SEED)
    _, _, mt = realdata.mnist_split(S.N_TEST, "test", 28, S.SEED)
    assert len(set(m["example_indices"])) == S.N_TRAIN and max(m["example_indices"]) < 50000
    assert len(set(mt["example_indices"])) == S.N_TEST
    xa, ya, xb, _yb = mnist.load()
    assert np.array_equal(x, xa[m["example_indices"]])
    assert np.array_equal(y, ya[m["example_indices"]])
    assert np.array_equal(xt, xb[mt["example_indices"]])
    read, acc = S.judge()
    assert acc > 0.95 and (read(xt).argmax(1) == yt).mean() > 0.95


def test_rank_correlation_and_distances():
    rng = np.random.default_rng(0)
    a = rng.normal(size=45)
    assert np.isclose(S.spearman(a, 3 * a + 1), 1.0) and np.isclose(S.spearman(a, -a ** 3), -1.0)
    assert S.spearman(a, np.ones(45)) == 0.0
    assert np.allclose(S.ranks([3.0, 1.0, 3.0, 2.0]), [2.5, 0.0, 2.5, 1.0])
    one_hot = S.distances(np.eye(10))
    assert np.allclose(one_hot[S.PAIRS], 1.0) and np.allclose(np.diag(one_hot), 0.0)
    t = rng.normal(size=(10, 5))
    assert np.allclose(S.cosines(t), S.cosines(t + 7.0)) and np.allclose(np.diag(S.cosines(t)), 1)


# ---------------------------------------------------------------------- 106
def test_table_claims():
    needs()
    s, d = study("label-embedding"), S.table_data()
    agree = np.asarray(s.panels[1].data["seeds"])              # (4, width, seed)
    w8 = S.TAB_WIDTHS.index(8)
    # a one-hot marker puts every pair equally far; a trained table does not
    assert np.ptp(s.panels[0].data["one_hot"][S.PAIRS]) < 1e-12
    assert np.ptp(s.panels[0].data["values"][S.PAIRS]) > 0.5
    # trained geometry follows the mean images and the judge's confusion; untrained does not
    for w in range(1, len(S.TAB_WIDTHS)):                      # every width from 4 up, every seed
        assert agree[0, w].min() > agree[2, w].max() + 0.2
        assert agree[1, w].mean() > agree[3, w].mean() + 0.3
    assert agree[0, w8].mean() > 0.6 and abs(agree[2].mean()) < 0.15
    # wider tables agree more with the images and between seeds; the numbers never agree
    geometry = np.asarray(s.panels[2].data["geometry"])
    numbers = np.asarray(s.panels[2].data["numbers"])
    assert agree[0, -1].mean() > agree[0, 0].mean() + 0.2
    assert geometry[-1].mean() > geometry[0].mean() + 0.3
    assert geometry[w8].mean() > numbers[w8].mean() + 0.3 and np.abs(numbers.mean(1)).max() < 0.3
    # the start: a table started large stays its start, one started small becomes the task's
    sweep = np.asarray(s.panels[2].data["start_sweep_seeds"])  # (3, start, seed)
    assert sweep[2, 0].min() > 0.8 and sweep[0, 0].max() < 0.4
    assert sweep[0, 2].min() > sweep[0, 0].max() + 0.2 and sweep[2, 2].max() < 0.5
    # 4 and 9 are each other's nearest label in most seeds at width 8
    near = np.asarray(s.metadata["nearest_label_counts_width_8"])
    assert near[4, 9] + near[9, 4] >= len(S.TAB_SEEDS)
    # the drawing task ends close to its floor, the mean images
    loss = np.asarray(s.panels[3].data["loss"])
    assert loss.min() >= d["floor"] and loss[w8].max() < d["floor"] * 1.02


def test_table_data_is_what_it_says():
    needs()
    s, d = study("label-embedding"), S.table_data()
    x, y, xt, yt, _ = S.data()
    means = np.array([x[y == c].mean(0) for c in range(10)])
    assert np.allclose(d["means"], means)
    assert np.isclose(d["floor"], np.mean((means[yt] - xt) ** 2))
    assert np.allclose(d["mean_dist"], S.distances(means)) and np.isclose(
        d["mean_dist"][S.PAIRS].mean(), 1.0)
    soft = d["soft"]
    assert np.allclose(soft, soft.T) and np.argmax(soft.sum(0)) in range(10)
    # one run again: same table, same states along the way
    again = S.tab_run(2, S.TAB_SEEDS[0])
    assert np.allclose(again["table"], d["table_2"][0])
    assert np.allclose(again["snaps"], d["snaps_2"][0])
    assert np.allclose(again["curve"], d["curve_2"][0])
    assert np.allclose(d["snaps_2"][0, 0], again["start"]) and np.allclose(
        d["snaps_2"][0, -1], d["table_2"][0])
    assert np.abs(again["start"]).max() < 0.05          # started small
    assert np.allclose(d["dist"][0, 0], S.distances(d["table_2"][0]))
    # the sweep at the main start size is the main run
    i = S.TAB_STARTS.index(S.TAB_INIT)
    assert np.allclose(d["sweep_dist"][i], d["dist"][S.TAB_WIDTHS.index(S.TAB_START_WIDTH)])
    assert np.allclose(s.panels[3].data["values"][:10].reshape(10, 784), d["drawn"])


def test_table_numbers_on_screen():
    needs()
    s = study("label-embedding")
    anim = M.LabelEmbedding(s)
    sub, m = subtitles(anim), s.metadata
    w8 = S.TAB_WIDTHS.index(8)
    agree = np.asarray(s.panels[1].data["ys"])
    assert "as the average images do (%.2f)" % agree[0, w8] in sub[3]
    assert "mix-ups do (%.2f); untrained, %.2f" % (agree[1, w8], agree[2, w8]) in sub[3]
    a, b = m["closest_pairs_width_8"][0]
    assert "closest labels are %d and %d" % (a, b) in sub[3]
    assert "(correlation %.2f at 8 a row)" % np.asarray(s.panels[2].data["ys"])[1, w8] in sub[4]
    assert "%.2f with 32" % np.asarray(s.panels[2].data["ys"])[0, -1] in sub[4]
    sweep = np.asarray(s.panels[2].data["start_sweep"])
    assert "(%.2f like its start, %.2f like the images)" % (sweep[2, 0], sweep[0, 0]) in sub[5]
    assert "the averages' own %.3f" % anim.d["floor"] in sub[2]
    t = anim.spans[3][1] - 1.0
    assert "table agrees %.2f" % agree[0, w8] in texts(anim, t)
    assert "table agrees %.2f" % agree[1, w8] in texts(anim, t)


# ---------------------------------------------------------------------- 107
def test_classifier_claims():
    needs()
    s = study("label-embedding-classifier")
    acc = np.asarray(s.panels[0].data["seeds"])                # (variant, width, seed)
    onehot, smooth, learned, random = range(4)
    wide = len(S.CLF_WIDTHS) - 1
    # accuracy: with wide rows, learning the table is worth less than the seeds differ
    assert abs(acc[learned, wide].mean() - acc[random, wide].mean()) < 0.005
    assert np.ptp(acc[:, wide].mean(1)) < 0.02 and acc[:, wide].min() > 0.94
    # ... with 2-number rows it is worth a lot, in every seed
    assert (acc[learned, 0] > acc[random, 0] + 0.03).all()
    # geometry: trained tables follow the judge's confusion, the fixed random one cannot
    cc = np.asarray(s.panels[2].data["seeds"])                 # (variant, width, seed)
    assert cc[onehot, wide].min() > cc[random, wide].max() + 0.2
    assert cc[learned, wide].min() > cc[random, wide].max() + 0.2
    assert abs(cc[random].mean()) < 0.15
    # label smoothing evens the rows out: less agreement and less spread, in the mean
    spread = np.asarray(s.panels[2].data["spread"])
    assert cc[smooth, wide].mean() < cc[onehot, wide].mean() - 0.3
    assert spread[smooth, wide].mean() < spread[onehot, wide].mean() - 0.03
    # 4 and 9 point more alike in the trained tables than in the smoothed or random ones
    fn = np.asarray(s.panels[2].data["four_nine"]).mean(2)
    assert min(fn[onehot, wide], fn[learned, wide]) > max(fn[smooth, wide], fn[random, wide]) + 0.15
    # mistakes: wide, every variant errs on the pairs one-hot errs on; with 2-number rows
    # the random table's mistakes are its own
    same = np.asarray(s.panels[3].data["same_pairs"])
    assert same[1:, wide].min() > 0.85
    assert same[random, 0] < same[learned, 0] - 0.3 and same[random, 0] < same[random, wide] - 0.5


def test_classifier_data_is_what_it_says():
    needs()
    s, d = study("label-embedding-classifier"), S.classifier_data()
    _x, _y, _xt, _yt, _ = S.data()
    wide, n = len(S.CLF_WIDTHS) - 1, S.N_TEST
    assert (d["confusion"].sum((-1, -2)) == n).all()
    assert np.allclose(d["confusion"].trace(axis1=-2, axis2=-1) / n, d["acc"])
    # every variant of a seed starts from the same table; the random one ends as it started
    for i, w in enumerate(S.CLF_WIDTHS):
        for j, seed in enumerate(S.CLF_SEEDS):
            start = S.clf_init(w, seed)[0]["T"]
            assert np.allclose(d["start"][i, j, :, :w], start)
            assert np.allclose(d["table"][3, i, j, :, :w], start)
            assert not np.allclose(d["table"][2, i, j, :, :w], start)
    # the scores drawn are the feature times each row, plus the bias
    table = S.clf_table(d, "one-hot", S.CLF_WIDTHS[wide], 0)
    assert np.allclose(d["feats"][0, wide] @ table.T + d["bias"][0, wide, 0], d["scores"][0, wide])
    anim = M.LabelEmbeddingClassifier(s)
    assert anim.truth == 4 and int(anim.order[0]) == int(np.argmax(anim.scores))
    # one run again
    again = S.clf_run("learned", 4, S.CLF_SEEDS[1])
    assert np.isclose(again["acc"], d["acc"][2, S.CLF_WIDTHS.index(4), 1])
    assert np.allclose(again["table"], d["table"][2, S.CLF_WIDTHS.index(4), 1])
    assert np.allclose(s.panels[3].data["by_variant"],
                       d["confusion"][:, wide].sum(1) * (1 - np.eye(10)))


def test_classifier_gradients():
    needs()
    from conftest import fd_check

    x, y, _, _, _ = S.data()
    xb, yb = x[:12], y[:12]
    for variant in S.CLF_VARIANTS:
        P, _ = S.clf_init(4, 3)
        P["T"] = P["T"] * 50                          # away from the tiny start

        def loss(Q, variant=variant):
            target = np.eye(10)[yb]
            if variant == "smoothed":
                target = target * (1 - S.CLF_SMOOTH) + S.CLF_SMOOTH / 10
            p = S.softmax(S.clf_scores(variant, Q, S.clf_feature(Q, xb)[1]))
            return -(target * np.log(p)).sum() / len(xb)
        keys = ["W0", "W1"] + ([] if variant == "random" else ["T"])
        fd_check(loss, P, S.clf_grads(variant, P, xb, yb), keys, tol=1e-4)


def test_classifier_numbers_on_screen():
    needs()
    s = study("label-embedding-classifier")
    anim = M.LabelEmbeddingClassifier(s)
    sub, _m = subtitles(anim), s.metadata
    acc = np.asarray(s.panels[0].data["ys"])
    assert "%s, %s, %s and %s" % tuple(M.pct(a) for a in acc[:, -1]) in sub[2]
    assert "buys %.1f points" % (100 * (acc[2, -1] - acc[3, -1])) in sub[2]
    cc = np.asarray(s.panels[2].data["ys"])
    assert "%.2f one-hot, %.2f learned, %.2f random" % (cc[0, -1], cc[2, -1], cc[3, -1]) in sub[3]
    assert "evens the rows out: %.2f" % cc[1, -1] in sub[3]
    same = np.asarray(s.panels[3].data["same_pairs"])
    assert "(%.2f)" % same[3, -1] in sub[4] and "(%.2f)" % same[3, 0] in sub[4]
    assert "here 4 wins with %.1f" % anim.scores[4] in sub[0]
    shown = texts(anim, anim.spans[3][1] - 1.0)
    for v in range(4):
        assert "agrees %.2f" % cc[v, -1] in shown


# ---------------------------------------------------------------------- 108
def test_autoencoder_claims():
    needs()
    s = study("label-embedding-autoencoder")
    loss = np.asarray(s.panels[0].data["seeds"])               # (variant, code, seed)
    knn = np.asarray(s.panels[1].data["seeds"])
    probe = np.asarray(s.panels[1].data["probe_seeds"])
    plain, dec, both, shuf = range(4)
    # telling the decoder lowers the held-out error at every code size, in every seed ...
    assert (loss[dec].max(1) < loss[plain].min(1)).all()
    assert loss[dec, 0].mean() < 0.9 * loss[plain, 0].mean()
    # ... by less and less as the code grows
    gain = 1 - loss[dec].mean(1) / loss[plain].mean(1)
    assert (np.diff(gain) < 0).all() and gain[0] > gain[-1] + 0.1
    # the same network handed a random label's row is no better than plain
    assert np.abs(loss[shuf].mean(1) / loss[plain].mean(1) - 1).max() < 0.03
    # telling the encoder too changes little next to that
    assert np.abs(loss[both].mean(1) / loss[dec].mean(1) - 1).max() < 0.03
    # the digit leaves a small code when the decoder is told it (both probes, every seed)
    for small in (0, 1):
        assert knn[dec, small].max() < knn[plain, small].min() - 0.2
        assert probe[dec, small].max() < probe[plain, small].min() - 0.15
        assert knn[both, small].max() < knn[plain, small].min() - 0.2
    # ... and stays in a big one
    assert abs(knn[dec, -1].mean() - knn[plain, -1].mean()) < 0.03
    assert np.abs(knn[shuf].mean(1) - knn[plain].mean(1)).max() < 0.1


def test_autoencoder_data_is_what_it_says():
    needs()
    s, a = study("label-embedding-autoencoder"), S.autoencoder_data()
    _x, _y, xt, yt, _ = S.data()
    assert np.array_equal(a["first"], S.first_of_each(yt))
    assert list(yt[a["first"]]) == list(range(10))
    assert np.allclose(a["sources"], xt[a["first"]])
    assert np.array_equal(a["cloud_y"], yt[:S.AE_CLOUD])
    # one run again: the same error, probe, codes and pictures
    again = S.ae_run("decoder", 2, S.AE_SEEDS[0])
    assert np.isclose(again["loss"], a["loss"][1, 0, 0])
    assert np.isclose(again["knn"], a["knn"][1, 0, 0])
    assert np.allclose(again["cloud"], a["cloud"][1, 0], atol=1e-5)
    assert np.allclose(again["pictures"], a["pictures"][1, 0], atol=1e-5)
    assert np.isclose(again["swap_as_asked"], a["swap_as_asked"][1, 0, 0])
    # the panels are those arrays
    assert np.allclose(s.panels[2].data["decoder"], a["cloud"][1, 0, :, :2])
    values = s.panels[3].data["values"].reshape(5, 10, 784)
    assert np.allclose(values[0], a["sources"])
    assert np.allclose(values[2], a["pictures"][1, 0][np.arange(10), np.arange(10)])
    # the probes read the digit when it is there and not when it is not
    rng = np.random.default_rng(0)
    yy = rng.integers(0, 10, 600)
    told = np.eye(10)[yy] + 0.05 * rng.normal(size=(600, 10))
    noise = rng.normal(size=(600, 10))
    assert S.knn_accuracy(told[:400], yy[:400], told[400:], yy[400:]) > 0.95
    assert S.linear_probe(told[:400], yy[:400], told[400:], yy[400:]) > 0.95
    assert S.knn_accuracy(noise[:400], yy[:400], noise[400:], yy[400:]) < 0.25


def test_autoencoder_gradients():
    needs()
    from conftest import fd_check

    x, y, _, _, _ = S.data()
    for variant in S.AE_VARIANTS:
        P, _ = S.ae_init(variant, 4, 1)
        keys = ["W0", "W2", "W3", "W5"] + ([] if variant == "plain" else ["E"])
        fd_check(lambda Q, v=variant: S.ae_loss(v, Q, x[:12], y[:12]), P,
                 S.ae_grads(variant, P, x[:12], y[:12]), keys, tol=1e-4)


def test_autoencoder_numbers_on_screen():
    needs()
    s = study("label-embedding-autoencoder")
    anim = M.LabelEmbeddingAutoencoder(s)
    sub = subtitles(anim)
    loss, knn = np.asarray(s.panels[0].data["ys"]), np.asarray(s.panels[1].data["ys"])
    assert "%.4f plain and %.4f told" % (loss[0, 0], loss[1, 0]) in sub[3]
    assert "%.4f against %.4f with 16 numbers" % (loss[0, -1], loss[1, -1]) in sub[3]
    assert "Plain: %s with 2 numbers. Decoder told: %s." % (
        M.pct(knn[0, 0], 0), M.pct(knn[1, 0], 0)) in sub[5]
    assert "%s and %s" % (M.pct(knn[0, -1], 0), M.pct(knn[1, -1], 0)) in sub[5]
    shown = texts(anim, anim.spans[4][1] - 1.0)
    assert "neighbours name the digit: %s" % M.pct(knn[0, 0], 0) in shown
    assert "neighbours name the digit: %s" % M.pct(knn[1, 0], 0) in shown
    # the code drawn in the pipeline is the recorded code of that held-out image
    a = anim.a
    assert np.allclose(anim.code_of(1, 2, 4), a["cloud"][1, 0, a["first"][4], :2])


# ---------------------------------------------------------------------- 109
def test_generation_claims():
    needs()
    s = study("label-embedding-generation")
    asked = np.asarray(s.panels[1].data["seeds"])     # (decoder, both, shuffled; code; seed)
    plain = np.asarray(s.panels[1].data["plain"])
    dec, both, shuf = range(3)
    # with a small code the swap works, in every seed; nothing to swap without a label
    assert asked[dec, 0].min() > 0.8 and asked[dec, 1].min() > 0.5
    assert asked[shuf].max() < 0.1 and plain.max() < 0.1
    assert asked[dec, 0].min() > asked[shuf, 0].max() + 0.7
    # the bigger the code, the less the label is obeyed
    assert (np.diff(asked[dec].mean(1)) < -0.1).all() and asked[dec, -1].max() < 0.1
    assert (np.diff(asked[both].mean(1)) < -0.1).all()
    # style: source and swap agree far above the floor, and more with a bigger code
    style = np.asarray(s.panels[2].data["seeds"])              # (code, seed, measure)
    floor = np.asarray(s.panels[2].data["floor"])              # (measure, code)
    assert np.abs(floor).max() < 0.1
    assert style[:, :, 1].min() > 0.4                          # slant, every code and seed
    assert style[1:, :, 0].min() > 0.8 and style[1:, :, 2].min() > 0.5
    assert (np.diff(style.mean(1), axis=0) > 0).all()
    # in between: the reading turns over from source to asked, with blends on the way
    mix = np.asarray(s.panels[3].data["ys"])                   # (source, asked, neither; alpha)
    assert mix[0, 0] > 0.95 and mix[1, 0] < 0.02
    assert mix[1, -1] > mix[1, 0] + 0.5 and mix[0, -1] < mix[0, 0] - 0.5
    assert (np.diff(mix[1]) > -0.01).all() and (np.diff(mix[0]) < 0.01).all()
    assert mix[2].max() > mix[2, 0] + 0.1
    assert np.allclose(mix.sum(0), 1.0)


def test_generation_data_is_what_it_says():
    needs()
    s, a = study("label-embedding-generation"), S.autoencoder_data()
    read, _ = S.judge()
    dec, ci = 1, S.AE_CODES.index(4)
    grid = s.panels[0].data["values"].reshape(10, 10, 784)
    assert np.allclose(grid, a["pictures"][dec, ci])
    # the judge's reading of every drawn picture, again
    assert np.array_equal(read(grid.reshape(100, 784)).argmax(1).reshape(10, 10),
                          a["read_pictures"][dec, ci])
    # a strip starts as the source's own reconstruction and ends as the swap in the grid
    for k, (lo, hi) in enumerate(S.AE_STRIPS):
        assert np.allclose(a["strips"][dec, ci, k, 0], a["pictures"][dec, ci, lo, lo], atol=1e-5)
        assert np.allclose(a["strips"][dec, ci, k, -1], a["pictures"][dec, ci, lo, hi], atol=1e-5)
    # the style measures: more ink is more ink, a right-leaning stroke has a positive slant
    img = np.zeros((3, 28, 28))
    for r in range(6, 22):
        img[0, r, 13:15] = 1.0                        # upright, 2 wide
        img[1, r, 11:17] = 1.0                        # upright, 6 wide
        c = 14 + (14 - r) // 2
        img[2, r, c:c + 2] = 1.0                      # leaning right
    ink, slant, thick = S.ink_measures(img.reshape(3, 784))
    assert ink[1] > 2.5 * ink[0] and thick[1] > thick[0]
    assert abs(slant[0]) < 1e-9 and slant[2] > 0.3
    assert np.allclose(a["source_measures"], S.ink_measures(a["sources"]))


def test_generation_numbers_on_screen():
    needs()
    s = study("label-embedding-generation")
    anim = M.LabelEmbeddingGeneration(s)
    sub = subtitles(anim)
    asked = np.asarray(s.panels[1].data["ys"])[0]
    assert "read as asked %s with 2-number codes, %s with 4, %s with 8, %s with 16" % tuple(
        M.pct(v, 0) for v in asked) in sub[2]
    read = np.asarray(s.panels[0].data["read"])[1, S.AE_CODES.index(4)]
    right = int(sum(read[r, c] == c for r in range(10) for c in range(10) if r != c))
    assert "reads %d of these 90 swaps" % right in sub[1]
    assert "%d of 90 swaps" % right in texts(anim, anim.spans[1][1] - 1.0)
    slant = np.asarray(s.panels[2].data["ys"])[1]
    assert "slant agrees %.2f with 2-number codes and %.2f with 8" % (slant[0], slant[2]) in sub[3]
    mix = np.asarray(s.panels[3].data["ys"])
    assert "%s of pictures still read as the source, %s as the digit asked and %s as neither" % (
        M.pct(mix[0, 5], 0), M.pct(mix[1, 5], 0), M.pct(mix[2, 5], 0)) in sub[4]
    # the swap phase shows the judge's actual reading of the picture on screen
    t = anim.spans[0][0] + 0.5 * (anim.spans[0][1] - anim.spans[0][0])
    turn = int(min(0.5 / 0.92 * 10, 9))
    lab = anim.sequence[turn]
    assert "the classifier reads %d" % read[anim.SOURCE, lab] in texts(anim, t)

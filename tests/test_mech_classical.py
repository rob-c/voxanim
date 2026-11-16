"""Animations 67-74 of the "next 100": what they draw is what was measured."""

import numpy as np
import pytest
from conftest import available

from voxanim import mech_classical as M
from voxanim import next_vision as nv
from voxanim import presets, qa, toy2d
from voxanim import study_classical as S
from voxanim.next_catalog import build_study

NAMES = list(M.ANIMATIONS)
needs_data = pytest.mark.skipif(not all(available(n) for n in NAMES), reason="MNIST not downloaded")
pytestmark = needs_data

_built = {}


def built(name):
    """``(animation, study)`` of catalogue ``name``, built once."""
    if name not in _built:
        _built[name] = (presets.get(name).build(), build_study("next_vision", name))
    return _built[name]


def shown(anim, t):
    """Every string on screen at ``t``, with the live subtitle."""
    scene = anim.scene(float(t))
    return [tx.s for tx in scene.texts if tx.opacity > 0.5] + [anim.subtitle]


def phase_time(anim, k, u):
    a, b = anim.spans[k]
    return a + (b - a) * u


def has(strings, part):
    return any(part in s for s in strings)


def test_all_eight_are_registered():
    assert NAMES == ["bias-variance", "cross-validation", "svm-and-kernels", "gradient-boosting",
                     "gaussian-mixture-em", "curse-of-dimensionality", "tsne-and-umap",
                     "roc-and-calibration"]


@pytest.mark.parametrize("name", NAMES)
def test_registered_laid_out_and_looping(name):
    anim, _study = built(name)
    assert isinstance(anim, M.ANIMATIONS[name])
    assert 45.0 <= anim.spans[-1][1] <= 60.0
    assert qa.check(name, samples=24, verbose=False) == 0
    first, last = anim.scene(0.0), anim.scene(float(anim.duration))

    def key(sc):
        return sorted((t.s, round(float(t.opacity), 3)) for t in sc.texts)

    assert key(first) == key(last)
    for k in range(len(anim.PHASES)):
        assert anim.subtitle_for(k)


@pytest.mark.parametrize("name", NAMES)
def test_kept_arrays_are_the_studys(name):
    # every cache key in KEPT loads: the settings still match next_vision.py
    for key in M.KEPT:
        assert M.kept(key)
    # and the four redesigned studies come from study_classical, not next_vision
    assert set(S.STUDIES) == {"bias-variance", "gaussian-mixture-em", "curse-of-dimensionality",
                              "tsne-and-umap"}
    assert not set(M.KEPT) & {"bias_variance_mnist", "gmm_mnist", "dimension_mnist", "tsne"}


# ---------------------------------------------------------------------- 67 bias-variance
def test_bias_variance_shows_the_studys_decomposition():
    anim, study = built("bias-variance")
    bias, spread, noise, mse = (np.asarray(v) for v in study.panels[2].data["ys"])
    assert np.allclose(bias + spread + noise, mse)          # exact on the test images
    assert np.max(study.panels[3].data["values"]) < 1e-12
    best = int(np.argmin(mse))
    strings = shown(anim, phase_time(anim, 3, 0.97))
    assert has(strings, "bias   %s" % M.small(bias[-1]))
    assert has(strings, "+ spread   %s" % M.small(spread[-1]))
    assert has(strings, "= error   %s" % M.small(mse[-1]))
    assert has(strings, "lowest at degree %d" % study.panels[2].data["x"][best])
    assert has(strings, "It is lowest at degree %d: %s" % (best + 1, M.small(mse[best])))
    assert has(shown(anim, phase_time(anim, 2, 0.05)), "bias %s, falling to %s" % (
        M.small(bias[0]), M.small(bias[best])))


def test_bias_falls_variance_rises_and_the_error_has_a_minimum_between():
    d = S.bias_variance_data()
    bias, variance, mse = d["bias2"], d["variance"], d["mse"]
    best = int(np.argmin(mse))
    assert 0 < best < len(mse) - 1                          # neither the stiffest nor the loosest
    assert bias[best] < 0.1 * bias[0]                       # the line's bias is gone ...
    assert variance[-1] > 10 * variance[0]                  # ... and the spread has taken over
    assert mse[best] < 0.6 * mse[0] and mse[best] < 0.6 * mse[-1]
    # the rise in spread is not only a few wild fits: the median column's spread rises too
    assert d["median_variance"][-1] > 2 * d["median_variance"][0]
    # the low-degree error is mostly bias, the high-degree error mostly spread
    assert bias[0] > 10 * variance[0] and variance[-1] > 10 * bias[-1]


def test_bias_variance_fits_are_real_resamples_of_real_measurements():
    anim, _ = built("bias-variance")
    d = S.bias_variance_data()
    x, _, xt, _ = M.nv.mnist_data()
    assert np.allclose(d["pool"], x[:5000].reshape(-1, 28, 28).mean(1))
    assert np.allclose(d["truth"], xt.reshape(-1, 28, 28).mean(1).mean(0))    # the test split
    assert d["sampled"].shape == (10, 200, 2, 40)
    assert d["sampled"][:, :, 0].max() < 5000 and d["sampled"][:, :, 1].max() == 27
    # a stored curve is the least-squares fit to its stored draw
    for degree, resample in ((0, 3), (3, 7), (9, 150)):
        images, columns = d["sampled"][degree, resample]
        again = S.fit_columns(columns, d["pool"][images, columns], degree + 1)(d["fine"])
        assert np.allclose(again, d["curves"][degree, resample])
    assert anim.curves.shape == (10, 200, len(anim.fine))
    # the measurement drawn on a digit is that image's column
    image, column = anim.examples[0]
    assert anim.images[image].reshape(28, 28)[:, column].mean() == pytest.approx(
        anim.pool[image, column])


# ---------------------------------------------------------------------- 68 cross-validation
def test_cross_validation_numbers_are_the_studys():
    anim, study = built("cross-validation")
    scores = np.asarray(study.panels[2].data["values"])
    outer = np.asarray(study.panels[3].data["values"])
    assert study.panels[3].notes[0] == "optimism gap %+.1f pp" % (
        100 * (anim.optimistic - anim.nested))
    assert study.panels[3].notes[1] == "external test accuracy %.1f%%" % (100 * anim.test_accuracy)
    strings = shown(anim, phase_time(anim, 4, 0.9))
    for value in (scores.max(0).mean(), outer.mean(), anim.test_accuracy):
        assert has(strings, "%s%%" % M.number(100 * value))
    assert has(shown(anim, phase_time(anim, 3, 0.93)), "average of the five: %s%%" % M.number(
        100 * outer.mean()))


def test_cross_validation_recomputed_folds_match():
    anim, study = built("cross-validation")
    d = M.kept("cross_validation_mnist")
    scores = np.asarray(study.panels[2].data["values"])
    # the per-image right/wrong flags drawn as red rings reproduce the score table
    assert np.allclose(anim.correct.sum(2) / 40.0, scores)
    assert not anim.correct[:, 0, anim.fold != 0].any()
    # the inner folds choose the settings the study recorded
    inner = M.inner_scores(d["x"], d["y"], d["gammas"])
    assert np.allclose(d["gammas"][inner.argmax(1)], d["selected"])
    assert anim.test_right.mean() == pytest.approx(float(d["test_accuracy"]))
    sheet = M.thumbs(d["x"], d["y"], (anim.fold == 0).astype(int), 20)
    assert sheet.shape == (80, 160, 3)


# ---------------------------------------------------------------------- 69 svm-and-kernels
def test_svm_numbers_are_the_studys():
    anim, study = built("svm-and-kernels")
    strings = shown(anim, phase_time(anim, 4, 0.95))
    assert has(strings, "%s%%" % M.number(100 * study.metadata["linear_accuracy"]))
    assert has(strings, "%s%%" % M.number(100 * study.metadata["lifted_accuracy"]))
    counts = [int(v) for v in study.panels[3].data["values"]]
    strings = shown(anim, phase_time(anim, 3, 0.97))
    for count in counts:
        assert has(strings, "%d of 200" % count)
    assert [int((m["a"] > 1e-6).sum()) for m in anim.models] == counts


def test_svm_line_and_boundaries_are_the_fitted_models():
    anim, _ = built("svm-and-kernels")
    d = M.kept("svm_mnist")
    # the line shown settling ends at the study's linear SVM
    assert np.allclose(anim.trail[-1], np.r_[d["linear_w"], d["linear_b"]])
    assert int(anim.in_street.sum()) == 99 and int(anim.wrong_side.sum()) == 24
    # the drawn outlines lie on the models' decision boundaries
    mid = anim.lifted_edge.mean(1)
    assert np.abs(anim.lifted_score(mid)).max() < 0.02
    mid = anim.rbf_edges[2].mean(1)
    assert np.abs(toy2d.svm_predict(anim.models[2], mid)).max() < 0.05
    # the lifted model is the study's: its accuracy on the test points
    lift = np.column_stack((anim.test, (anim.test ** 2).sum(1)))
    assert np.mean((lift @ anim.lw + anim.lb > 0) == anim.test_y) == pytest.approx(anim.lifted_acc)


def test_contour_finds_a_circle():
    edge = M.contour(lambda q: (q * q).sum(1) - 1.0, (-2, 2), (-2, 2), 60)
    assert len(edge) > 40
    assert np.allclose(np.hypot(edge[..., 0], edge[..., 1]), 1.0, atol=0.01)


# ---------------------------------------------------------------------- 70 gradient-boosting
def test_boosting_numbers_are_the_studys():
    anim, study = built("gradient-boosting")
    slow, fast, bag = (float(v) for v in study.panels[3].data["values"])
    strings = shown(anim, phase_time(anim, 4, 0.97))
    for value in (slow, fast, bag):
        assert has(strings, "%.5f" % value)
    assert has(shown(anim, phase_time(anim, 2, 0.5)), "%.5f" % study.panels[1].data["ys"][0][-1])


def test_boosting_replayed_trees_reproduce_the_curves():
    anim, study = built("gradient-boosting")
    train, held = (np.asarray(v) for v in study.panels[1].data["ys"])
    assert np.allclose(((anim.fit[1:] - anim.y) ** 2).mean(1), train)
    assert np.allclose(((anim.held[1:] - anim.yt) ** 2).mean(1), held)
    assert np.allclose(((anim.held_fast[1:] - anim.yt) ** 2).mean(1), anim.fast_mse)
    assert np.allclose(anim.fit[0], anim.y.mean())
    # the path drawn down a tree ends at the leaf whose value the tree predicts
    tree = anim.trees[0]
    for i in (0, 17, 300):
        node = tree
        for go in M.route(tree, anim.x[i]):
            node = node[2 + go]
        assert node == pytest.approx(toy2d.tree_predict(tree, anim.x[i:i + 1])[0])
    assert set(map(len, M.spots(tree, 0, 1, 0, 0.1))) <= {0, 1, 2, 3}


# ---------------------------------------------------------------------- 71 gaussian-mixture-em
def test_mixture_numbers_are_the_studys():
    anim, study = built("gaussian-mixture-em")
    a, b = (np.asarray(v) for v in study.panels[1].data["ys"])
    d = S.gaussian_mixture_data()
    best, worst = anim.chosen
    strings = shown(anim, phase_time(anim, 3, 0.9))
    assert has(strings, "log-likelihood %.2f, not %.2f" % (b[-1], a[-1]))
    assert has(strings, "%.2f against %.2f" % (d["test_ll"][0, worst], d["test_ll"][0, best]))
    assert has(shown(anim, phase_time(anim, 2, 0.5)), "from %.2f to %.2f" % (a[0], a[-1]))
    assert (np.diff(a) > -1e-9).all() and (np.diff(b) > -1e-9).all()      # "never worse"
    reached = np.asarray(study.panels[3].data["values"])
    strings = shown(anim, phase_time(anim, 4, 0.97))
    for share in reached:
        assert has(strings, "%.0f%% reach the best fit" % (100 * share))
    assert np.allclose(reached, study.metadata["reached_best"])
    assert (anim.fate == 0).mean(1) == pytest.approx(reached)
    assert (anim.fate == 2).mean(1) == pytest.approx(study.metadata["clearly_worse_heldout"])


def test_the_start_decides_where_em_ends_and_a_kmeans_start_helps():
    d = S.gaussian_mixture_data()
    s = S.mixture_summary(d)
    random, seeded, kmeans = range(3)
    # random starts: most reach the best optimum, a real share do not, and some end
    # clearly worse on images the fit never saw
    assert 0.5 < s["reached"][random] < 0.95
    assert s["bad"][random] >= 0.02
    assert s["worst"][random] < s["usual"] - 0.1
    # the fit the animation shows as "stuck" is worse on training AND held-out images
    best, worst = s["best_start"][random], s["worst_start"][random]
    assert d["train_ll"][random, worst] < d["train_ll"][random, best] - 0.1
    assert d["test_ll"][random, worst] < d["test_ll"][random, best] - 0.1
    # the remedy that works here: start from a k-means solution
    assert s["reached"][kmeans] > s["reached"][random] + 0.05
    assert s["bad"][kmeans] < s["bad"][random] - 0.02
    # k-means++ seeding alone is NOT shown to help (and the animation says so)
    assert abs(s["reached"][seeded] - s["reached"][random]) < 0.06


def test_mixture_data_and_history_are_the_studys():
    anim, _ = built("gaussian-mixture-em")
    d = S.gaussian_mixture_data()
    assert d["train"].shape == (1200, 2) and d["test"].shape == (600, 2)
    assert set(np.unique(d["train_labels"])) == {0, 1, 6, 7} == set(np.unique(d["test_labels"]))
    _, y, _, yt = M.nv.mnist_data()                        # train and test: the official splits
    assert np.array_equal(y[d["train_indices"]], d["train_labels"])
    assert np.array_equal(yt[d["test_indices"]], d["test_labels"])
    assert d["silhouette"] > 0.4                           # distinct clusters, unlike the first study  # noqa: E501
    # random and k-means++ starts are training points
    for kind in (0, 1):
        for start in d["starts"][kind, :5].reshape(-1, 2):
            assert np.isclose(d["train"], start).all(1).any()
    for run, i in zip(anim.runs, anim.chosen):
        fit = np.concatenate([run[key][-1].ravel() for key in ("mu", "cov", "pi")])
        assert np.allclose(fit, d["final"][0, i])
        assert np.allclose(run["loss"], d["losses"][0, i])
        assert np.allclose(run["r"].sum(2), 1.0)
        assert S.mean_log_density(d["train"], run["mu"][-1], run["cov"][-1], run["pi"][-1]) \
            == pytest.approx(d["train_ll"][0, i])
    shape = M.ellipse(np.zeros(2), np.diag([4.0, 1.0]), sigmas=2.0)
    assert np.allclose((shape[:, 0] / 4) ** 2 + (shape[:, 1] / 2) ** 2, 1.0)


# ---------------------------------------------------------------------- 72 curse-of-dimensionality
def test_curse_numbers_are_the_studys():
    anim, study = built("curse-of-dimensionality")
    ratio = np.asarray(study.panels[0].data["ys"][0])
    accuracy = np.asarray(study.panels[1].data["ys"][0])
    table = np.asarray(study.panels[2].data["values"])
    assert has(shown(anim, phase_time(anim, 0, 0.5)), "%.0f%% as far" % (100 * ratio[0]))
    assert has(shown(anim, phase_time(anim, 1, 0.9)), "nearest ÷ farthest: %.0f%%" % (
        100 * ratio[-1]))
    strings = shown(anim, phase_time(anim, 3, 0.9))
    assert has(strings, "the same digit %.0f%% of the time. With 49 shuffled ones added, %.0f%%"
               % (100 * accuracy[0], 100 * accuracy[1]))
    strings = shown(anim, phase_time(anim, 4, 0.97))
    assert has(strings, "+49, 8,000 images: %.0f%%" % (100 * table[1, -1]))
    assert has(strings, "49 real, 125 images: %.0f%%" % (100 * table[0, 0]))
    assert np.allclose(table[:, 3], accuracy)


def test_empty_coordinates_hurt_and_coordinates_with_signal_help():
    d = S.curse_data()
    acc, ratio = d["accuracy"], d["ratio"]                  # [shuffle, level(, size)]
    at = S.CURSE["sizes"].index(1000)
    assert acc.shape == (5, 6, 7)
    # in every one of the five shuffles: 49 empty coordinates cost at least 15 points
    assert acc[:, 1, at].max() < acc[0, 0, at] - 0.15
    assert (np.diff(acc[:, :, at], axis=1) < 0).all()       # and every further block costs more
    assert acc[:, -1, at].max() < 0.3                       # ending near the 10% of guessing
    # distances concentrate, in every shuffle
    assert (np.diff(ratio, axis=1) > 0).all() and ratio[:, -1].min() > ratio[0, 0] + 0.5
    # eight times the data does not buy back what 49 empty coordinates cost
    assert acc[:, 1, -1].max() < acc[0, 0, at] - 0.1
    # the contrast: coordinates that carry signal help, although distances concentrate too
    pca = np.asarray(d["pca_accuracy"])
    assert pca[-1] > pca[0] + 0.3 and d["pca_ratio"][-1] > d["pca_ratio"][0]
    # all 49 PCA coordinates are a rotation of the 49 pixels: the same neighbours
    assert pca[-1] == pytest.approx(acc[0, 0, at])


def test_curse_coordinates_are_what_they_are_said_to_be():
    anim, _ = built("curse-of-dimensionality")
    d = S.curse_data()
    x, y, xt, yt = M.nv.mnist_data()
    train, test = S.downsample(x[:1000]), S.downsample(xt[:1000])
    dist = np.sqrt(np.maximum(S.squared_distances(test, train), 0))
    # the first level is the real pixels alone
    assert np.array_equal(dist.argmin(1), d["nearest"][0])
    assert np.allclose(dist.min(1), d["near_distance"]) and np.allclose(dist.max(1),
                                                                        d["far_distance"])
    assert np.mean(y[:1000][d["nearest"][0]] == yt[:1000]) == pytest.approx(
        d["accuracy"][0, 0, S.CURSE["sizes"].index(1000)])
    assert np.mean(dist.min(1) / dist.max(1)) == pytest.approx(d["ratio"][0, 0])
    # a shuffled block: every column keeps its values and loses its images
    block = S.shuffled_block(np.random.default_rng(3), train)
    assert np.allclose(np.sort(block, 0), np.sort(train, 0))
    assert abs(np.corrcoef(block[:, 24], train[:, 24])[0, 1]) < 0.1
    means = np.array([block[y[:1000] == c].mean(0) for c in range(10)])
    real = np.array([train[y[:1000] == c].mean(0) for c in range(10)])
    assert means.std(0).mean() < 0.3 * real.std(0).mean()   # no class structure left
    assert np.allclose(np.sort(d["noise_tiles"][0], 0)[-1], np.sort(test, 0)[-1], atol=1) \
        and d["noise_tiles"].shape == (3, 16, 49)
    assert np.allclose([M.ball_in_box(v) for v in anim.BALL], d["volume"])
    assert M.ball_in_box(2) == pytest.approx(np.pi / 4)
    assert np.allclose(np.asarray(d["hist"]).sum(1), 1.0)
    assert np.allclose(anim.rug[0].min(), d["near_distance"][0])


# ---------------------------------------------------------------------- 73 tsne-and-umap
def test_both_methods_maps_are_the_studys():
    anim, study = built("tsne-and-umap")
    d = S.maps_data()
    assert anim.TITLE == "t-SNE and UMAP"
    for k in range(3):
        assert np.allclose(anim.maps[k], d["tsne"][k, 0])
    assert np.allclose(study.panels[0].data["xy"], d["tsne"][1, 0])
    assert np.allclose(study.panels[1].data["xy"], d["umap"][1, 0])
    assert np.allclose(anim.frames[-1], d["tsne"][1, 0])            # the t-SNE path replayed
    assert np.abs(anim.frames[0]).max() < 0.01                      # the heap it starts from
    assert np.allclose(anim.umap_frames[-1], anim.umap)             # UMAP's path ends at its map
    assert has(shown(anim, phase_time(anim, 1, 0.44)), "step 200 of 400")
    assert has(shown(anim, phase_time(anim, 3, 0.95)), "round 500 of 500")
    strings = shown(anim, phase_time(anim, 4, 0.97))
    faith = d["faith"]
    assert has(strings, "t-SNE keeps %.0f%% nearby and UMAP %.0f%%" % (
        100 * faith[0, 1, :, 0].mean(), 100 * faith[1, 1, :, 0].mean()))
    assert has(strings, "%.2f  (%.2f to %.2f)" % (
        faith[1, 1, :, 1].mean(), faith[1, 1, :, 1].min(), faith[1, 1, :, 1].max()))


def test_tsne_affinities_and_faithfulness():
    anim, _ = built("tsne-and-umap")
    x = anim.x
    p = M.conditional_affinities(x, 30)
    assert np.allclose(p.sum(1), 1.0)
    assert np.allclose((p + p.T) / (2 * len(x)), nv.tsne_affinity(x, 30))
    assert np.allclose(anim.share[1], p[anim.WHO])
    many = [int((row > 0.01).sum()) for row in anim.share]
    assert many == sorted(many)                                     # more perplexity, more sharers
    assert has(shown(anim, phase_time(anim, 0, 0.1)), "here %d, %d, then %d" % tuple(many))
    # a map that is the data keeps everything
    assert M.map_faithfulness(x[:, :2] * 1.0, x[:, :2] * 3.0) == pytest.approx((1.0, 1.0))


def test_umap_is_umap_and_keeps_about_what_tsne_keeps():
    d = S.maps_data()
    x, faith = d["pixels"], d["faith"]                              # [method, setting, seed, measure]  # noqa: E501
    # the fuzzy graph: 15 links a row, the nearest at full strength, each row summing to log2(15)
    v = S.fuzzy_memberships(x, 15)
    assert ((v > 0).sum(1) == 15).all() and np.allclose(v.max(1), 1.0)
    assert np.allclose(v.sum(1), np.log2(15), atol=1e-3)
    nearest = S.pair_distances(x).argsort(1, kind="stable")[:, 1]
    assert np.allclose(v[np.arange(len(x)), nearest], 1.0)
    graph = S.fuzzy_graph(x, 15)
    assert np.allclose(graph, d["graph"]) and np.allclose(graph, graph.T)
    assert graph.min() >= 0 and graph.max() <= 1 + 1e-12 and (graph >= v - 1e-12).all()
    # the layout is seeded, and starts from the graph's spectral embedding
    a, b = (S.umap_layout(graph, [51, 15, 0], epochs=20) for _ in range(2))
    assert np.array_equal(a, b)
    assert np.allclose(a[0], d["umap_frames"][0]) and np.abs(a[0]).max() == pytest.approx(
        10, abs=0.01)
    # the layout does the work: neighbourhoods kept rise well above the spectral start's
    assert faith[1, 1, :, 0].min() > d["spectral_faith"][0] + 0.15
    # every map of either method keeps far more than chance (10 of 239) and far from all
    assert faith[..., 0].min() > 0.4 and faith[..., 0].max() < 0.7
    # at the middle settings, in every seed: the two methods keep nearly the same
    # neighbourhoods, and t-SNE's distances track the real ones better than UMAP's - loosely
    assert abs(faith[0, 1, :, 0].mean() - faith[1, 1, :, 0].mean()) < 0.08
    assert faith[0, 1, :, 1].min() > faith[1, 1, :, 1].max() + 0.02
    assert faith[..., 1].max() < 0.75
    assert d["tsne"].shape == d["umap"].shape == (3, 3, 240, 2)


# ---------------------------------------------------------------------- 74 roc-and-calibration
def test_roc_numbers_are_the_studys():
    anim, study = built("roc-and-calibration")
    (tn, fp), (fn, tp) = np.asarray(study.panels[0].data["values"])
    # at the moment the sweep rests on 50%, the four boxes are the study's matrix
    strings = shown(anim, phase_time(anim, 1, 0.52))
    for what, count in (("caught", tp), ("missed", fn), ("false alarm", fp), ("left alone", tn)):
        assert has(strings, "%s  %s" % (what, f"{int(count):,}"))
    assert has(strings, "catches %d sevens, misses %d" % (tp, fn))
    assert has(shown(anim, phase_time(anim, 2, 0.95)), study.panels[1].notes[0].split()[-1])
    assert study.panels[2].notes[0] == "positive prevalence %.1f%%" % (100 * anim.prevalence)


def test_roc_recomputed_scores_match():
    anim, study = built("roc-and-calibration")
    (tn, fp), (fn, tp) = np.asarray(study.panels[0].data["values"])
    assert anim._counts(0.0) == (tp, fn, fp, tn)
    assert anim._counts(anim.Z) == (0, tp + fn, 0, tn + fp)
    assert anim._counts(-anim.Z) == (tp + fn, 0, tn + fp, 0)
    # the calibration dots are the study's bins
    conf = np.asarray(study.panels[3].data["x"])
    freq = np.asarray(study.panels[3].data["ys"][0])
    assert np.allclose([b[2] for b in anim.bins], conf)
    assert np.allclose([b[3] for b in anim.bins], freq)
    assert sum(b[4] for b in anim.bins) == 2000
    # precision at the measured base rate is the counted one; rarer sevens lower it
    assert M.precision_at(*anim.rates, anim.prevalence) == pytest.approx(tp / (tp + fp))
    assert M.precision_at(*anim.rates, 0.01) < M.precision_at(*anim.rates, anim.prevalence)
    assert has(shown(anim, phase_time(anim, 3, 0.9)), "%.1f%% sevens" % (
        100 * M.precision_at(*anim.rates, 0.01)))
    assert np.allclose(anim.share[0].sum(), 1.0) and np.allclose(anim.share[1].sum(), 1.0)

"""Animations 81-85 of the "next 100": what they draw is what was measured."""

import numpy as np
import pytest
from conftest import available

from voxanim import mech_retrieval as M
from voxanim import presets, qa
from voxanim import study_retrieval as SR
from voxanim.next_catalog import build_study

PRESETS = {
    "retrieval-augmented-generation": "gpt2-retrieval-augmented-generation",
    "vector-search-hnsw": "vector-search-hnsw",
    "infonce-temperature": "infonce-temperature",
    "hard-negatives": "hard-negatives",
    "dimensionality-of-embeddings": "dimensionality-of-embeddings",
}


def needs(name):
    return pytest.mark.skipif(not available(PRESETS[name]), reason="data not downloaded")


_built = {}


def built(name):
    """``(animation, study)`` of catalogue ``name``, built once."""
    if name not in _built:
        _built[name] = (presets.get(PRESETS[name]).build(), build_study("next_systems", name))
    return _built[name]


def shown(anim, t):
    """Every string on screen at ``t``, with the live subtitle."""
    scene = anim.scene(float(t))
    return [tx.s for tx in scene.texts if tx.opacity > 0.5] + [anim.subtitle]


def middle(anim, k, at=0.9):
    a, b = anim.spans[k]
    return a + (b - a) * at


@pytest.mark.parametrize("name", [pytest.param(n, marks=needs(n)) for n in PRESETS])
def test_drawn_by_this_module_and_laid_out(name):
    anim, _ = built(name)
    assert type(anim) is M.ANIMATIONS[name]
    assert 45 <= anim.spans[-1][1] <= 60
    assert qa.check(PRESETS[name], samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", [pytest.param(n, marks=needs(n)) for n in PRESETS])
def test_loop_closes_on_an_empty_frame(name):
    anim, _ = built(name)
    first = {(t.s, round(t.opacity, 3)) for t in anim.scene(0.0).texts}
    last = {(t.s, round(t.opacity, 3)) for t in anim.scene(anim.duration).texts}
    assert first == last
    assert not anim.scene(0.0).lines and not anim.scene(anim.duration).lines


# ---------------------------------------------------------------------- 81
@needs("retrieval-augmented-generation")
def test_retrieval_study_helps_only_with_what_the_file_holds():
    _, study = built("retrieval-augmented-generation")
    settings, arrays = M.evidence(study)
    store, asked = settings["store"], settings["asked"]
    prob = np.asarray(study.panels[2].data["values"], float)
    assert prob.shape == (len(asked), len(SR.CONDITIONS)) and len(asked) >= 50
    assert np.all((prob > 0) & (prob < 1))
    mean = dict(zip(SR.CONDITIONS, prob.mean(0)))
    # with its own line on file, either retriever lifts the true next word a long way ...
    assert mean["words_in"] > mean["alone"] + 0.3
    assert mean["embedding_in"] > mean["alone"] + 0.3
    assert study.metadata["helped"]["words_in"] > 0.9
    # ... and with that line removed almost none of the lift is left
    for key in ("words_out", "embedding_out", "random"):
        assert mean[key] < mean["alone"] + 0.05
        assert mean["words_in"] > mean[key] + 0.3
    assert study.metadata["source_found_first"]["words"] > 0.9
    assert study.metadata["source_found_first"]["embedding"] > 0.7
    # the data: distinct held-out lines, each question the start of its own line, and the
    # prompts that are meant to lack that line do lack it
    assert len(store) == len(set(store)) == SR.STORE
    held_out, train = M.R.text("test"), M.R.text("train")
    assert all(line in held_out for line in store)
    assert sum(line in train for line in store) <= 3        # a few stock lines recur
    pick = arrays["history_pick"]
    for n, i in enumerate(asked):
        assert store[i].startswith(study.metadata["questions"][n])
        columns = {c: int(pick[n, k]) for k, c in enumerate(SR.CONDITIONS)}
        assert i not in (columns["words_out"], columns["embedding_out"], columns["random"])
    assert study.metadata["designs_tried"]


@needs("retrieval-augmented-generation")
def test_retrieval_shows_the_studys_scores_and_probabilities():
    anim, study = built("retrieval-augmented-generation")
    prob = np.asarray(study.panels[2].data["values"], float)
    one = dict(zip(SR.CONDITIONS, prob[0]))
    alone, top, off = one["alone"], one["words_in"], one["words_out"]
    assert "%.2f" % anim.scores.max() in anim.subtitle_for(1)
    assert M.percent(top) in anim.subtitle_for(2) and M.percent(alone) in anim.subtitle_for(2)
    assert M.percent(off) in anim.subtitle_for(3)
    texts = shown(anim, middle(anim, 1))
    for v in anim.scores:
        assert "%.2f" % v in texts
    texts = shown(anim, middle(anim, 3))
    for v in (alone, top, off):
        assert M.percent(v) in texts
    # the candidate bars are GPT-2's rows, the true next word first
    assert anim.tops[0, 0] == pytest.approx(alone) and anim.tops[1, 0] == pytest.approx(top)
    assert anim.tops[2, 0] == pytest.approx(off)
    # the last phase: every prompt's mean, and how many questions it helped and hurt
    texts = shown(anim, anim.spans[4][1] + 1.0)
    for k, key in enumerate(SR.CONDITIONS):
        assert M.percent(prob[:, k].mean()) in texts
        assert M.percent(prob[:, k].mean()) in anim.subtitle_for(4) or key == "random"
        if key != "alone":
            assert "%d : %d" % ((prob[:, k] > prob[:, 0]).sum(),
                                (prob[:, k] < prob[:, 0]).sum()) in texts
    assert anim.helped["words_in"] == round(study.metadata["helped"]["words_in"] * len(prob))


@needs("retrieval-augmented-generation")
def test_retrieval_weights_and_matches_are_the_retrievers():
    anim, study = built("retrieval-augmented-generation")
    settings, arrays = M.evidence(study)
    words, idf = M.tfidf_table()
    np.testing.assert_allclose(idf, arrays["parameter_idf"])
    q = M.tfidf_vector(anim.query)
    scores = np.array([float(M.tfidf_vector(p) @ q) for p in settings["store"]])
    np.testing.assert_allclose(scores, arrays["history_word_scores"][0], atol=1e-12)
    # the five lines drawn are the five best of the whole file, with their own scores
    assert sorted(anim.on_file) == sorted(int(j) for j in np.argsort(-scores, kind="stable")[:5])
    np.testing.assert_allclose(anim.scores, scores[anim.on_file], atol=1e-12)
    # a line with no shared word scores nothing, and the shown weights are the fitted ones
    for shared, score in zip(anim.shared, anim.scores):
        assert bool(shared) == (score > 0)
    assert anim.qidf == [float(idf[words.index(w)]) for w in anim.qwords]
    assert anim.found and anim.passages[anim.best].startswith(anim.query)
    assert not anim.passages[anim.second].startswith(anim.query)
    assert anim.scores[anim.second] == np.sort(anim.scores)[-2]


# ---------------------------------------------------------------------- 82
@needs("vector-search-hnsw")
def test_hnsw_search_steps_are_the_studys_search():
    anim, study = built("vector-search-hnsw")
    ix = anim.ix
    np.testing.assert_allclose(ix["xy"], study.panels[0].data["xy"], atol=1e-12)
    assert np.array_equal(ix["graph"].levels, study.panels[0].data["classes"])
    efs = [int(v) for v in study.panels[3].data["x"]]
    recall, work = [], []
    for ef in efs:
        hits, counts = [], []
        for q, exact in zip(ix["q"], ix["exact"]):
            events, answer, count = M.search_trace(ix["graph"], q, ef)
            assert (answer, count) == ix["graph"].query(q, ef)[:2]
            assert events[-1]["n"] == count
            hits.append(answer == exact)
            counts.append(count)
        recall.append(np.mean(hits))
        work.append(np.mean(counts) / len(ix["u"]))
    np.testing.assert_allclose(recall, study.panels[3].data["ys"][0])
    np.testing.assert_allclose(work, study.panels[3].data["ys"][1])
    # the path the study drew (query 0, width 8): the hops and the candidates opened
    events = M.search_trace(ix["graph"], ix["q"][0], 8)[0]
    path = [e["node"] for e in events if e["kind"] in ("hop", "expand")]
    np.testing.assert_allclose(ix["xy"][path], study.panels[2].data["xy"], atol=1e-12)


@needs("vector-search-hnsw")
def test_hnsw_shows_the_measured_numbers_and_true_distances():
    anim, study = built("vector-search-hnsw")
    recall, work = np.asarray(study.panels[3].data["ys"], float)
    assert M.percent(recall[0]) in anim.subtitle_for(3)
    assert M.percent(work[0]) in anim.subtitle_for(3)
    texts = shown(anim, anim.spans[3][1] + 1.0)
    assert any(M.percent(recall[-1]) in s for s in texts)
    assert any(M.percent(work[-1]) in s for s in texts)
    # around the query each point sits at its true distance, and every hop comes nearer
    q = anim.ix["q"][anim.QUERY]
    true = np.sqrt(((anim.ix["u"] - q) ** 2).sum(1))
    np.testing.assert_allclose(np.linalg.norm(anim.around, axis=1) * anim.reach, true)
    events, _answer, count = anim.traces[1]
    hops = [e["d"] for e in events if e["kind"] in ("enter", "hop")]
    assert hops == sorted(hops, reverse=True)
    assert "%d of %d" % (count, len(true)) in shown(anim, middle(anim, 1, 0.96))
    assert anim.traces[8][1] == anim.exact == int(np.argmin(true))


# ---------------------------------------------------------------------- 83
@needs("infonce-temperature")
def test_temperature_study_has_a_sweet_spot_in_every_seed():
    _, study = built("infonce-temperature")
    acc, unif, hard = (np.asarray(study.metadata[k], float) for k in (
        "accuracy", "uniformity", "hardest_negative_share"))
    temps = study.metadata["design"]["temperatures"]
    assert acc.shape == (len(temps), 3) and len(temps) >= 6 and temps[0] <= 0.02 < 1 < temps[-1]
    best = int(np.argmax(acc.mean(1)))
    assert 0 < best < len(temps) - 1
    # every seed of the best temperature beats every seed of both ends, with room to spare
    assert acc[best].min() > acc[0].max() + 0.02
    assert acc[best].min() > acc[-1].max() + 0.1
    # the embeddings are most spread near the same place, and crowd towards both ends
    spread = int(np.argmin(unif.mean(1)))
    assert abs(spread - best) <= 1
    assert unif[spread].max() < unif[0].min() - 0.5
    assert unif[spread].max() < unif[-1].min() - 0.5
    # a low temperature puts the push on the hardest negative: its share falls all the way
    assert np.all(np.diff(hard.mean(1)) < 0)
    assert hard[0].min() > 0.5 and hard[-1].max() < 0.05
    # the panels are those runs: mean, lowest and highest seed
    for k, values in ((1, study.metadata["alignment"]), (2, unif), (3, acc)):
        np.testing.assert_allclose(study.panels[k].data["ys"], SR.spread(values))
    meta = study.metadata["training_provenance"]["dataset"]
    assert meta["train"]["official_partition"] == "train"
    assert meta["test"]["official_partition"] == "test"
    assert study.metadata["designs_tried"]


@needs("infonce-temperature")
def test_temperature_shows_the_studys_numbers():
    anim, study = built("infonce-temperature")
    acc = np.asarray(study.panels[3].data["ys"], float)[0]
    unif = np.asarray(study.panels[2].data["ys"], float)[0]
    texts = shown(anim, anim.spans[4][1] + 1.0)
    for v in acc:
        assert M.percent(v) in texts
    assert M.percent(acc.max()) in anim.subtitle_for(4)
    assert M.percent(study.metadata["pixel_accuracy"]) in " ".join(texts)
    assert M.percent(study.metadata["untrained"]["accuracy"]) in " ".join(texts)
    for i in anim.at:
        assert "crowding %.2f" % unif[i] in shown(anim, middle(anim, 2))
        assert "%.2f" % unif[i] in anim.subtitle_for(2)
    assert "crowding %.2f" % unif[-1] in " ".join(shown(anim, middle(anim, 3, 0.95)))


@needs("infonce-temperature")
def test_temperature_encoders_and_shares_are_the_studys():
    anim, study = built("infonce-temperature")
    held = SR.eval_views()
    for i, w in zip(anim.at, anim.encoders):            # the first seed's encoders
        measured = SR.measures(w, held)
        for key, value in zip(("accuracy", "alignment", "uniformity"), measured):
            assert value == pytest.approx(study.metadata[key][i][0], rel=1e-5)
    for apart, same in anim.pairs:
        assert apart.sum() == pytest.approx(1.0) and same.sum() == pytest.approx(1.0)
    # the shares drawn are the loss's own softmax over the same similarities
    for t in anim.temps:
        p = M.shares(anim.sim, t)
        np.testing.assert_allclose(p.sum(1), 1.0)
        scores = anim.sim / t
        expected = np.exp(scores[anim.ROW]) / np.exp(scores[anim.ROW]).sum()
        np.testing.assert_allclose(p[anim.ROW], expected)
        assert M.percent(M.push(anim.sim, t, anim.ROW), 0) in anim.subtitle_for(1)
    assert anim.share(anim.temps[0]) > anim.share(anim.temps[1]) > anim.share(anim.temps[2])
    assert M.push(anim.sim, anim.temps[0], anim.ROW) > M.push(anim.sim, anim.temps[1], anim.ROW) \
        > M.push(anim.sim, anim.temps[2], anim.ROW)
    np.testing.assert_allclose(np.linalg.norm(anim.za, axis=1), 1.0, atol=1e-5)
    # the two views of a digit differ, and are views of that test digit
    assert not np.allclose(anim.a, anim.b)
    assert anim.partner > anim.other + 0.3


# ---------------------------------------------------------------------- 84
@needs("hard-negatives")
def test_hard_negatives_study_mining_helps_and_its_false_negatives_cost():
    _, study = built("hard-negatives")
    acc = {k: np.asarray(v, float) for k, v in study.metadata["accuracy"].items()}
    share = study.metadata["false_negative_share"]
    first = {k: np.mean(v) for k, v in share["first_100_updates"].items()}
    last = {k: np.mean(v) for k, v in share["last_100_updates"].items()}
    assert all(len(v) == 3 for v in acc.values())
    # mined negatives beat random ones, and beat them more once the false ones are removed
    assert acc["hardest"].mean() > acc["random"].mean() + 0.015
    assert acc["hardest"].min() > acc["random"].max()
    assert acc["clean"].min() > acc["hardest"].max() + 0.03
    # the mined are increasingly the anchor's own digit; the drawn stay at one in ten
    assert last["hardest"] > first["hardest"] + 0.1 and first["hardest"] > first["random"] + 0.05
    assert 0.08 < first["random"] < 0.12 and 0.08 < last["random"] < 0.12
    assert last["clean"] == 0.0
    assert first["random"] < last["skip"] < last["hardest"] - 0.15
    np.testing.assert_allclose(study.panels[3].data["values"],
                               [acc[m].mean() for m in SR.MODES])
    assert study.metadata["designs_tried"]


@needs("hard-negatives")
def test_hard_negatives_are_mined_without_the_labels():
    anim, _ = built("hard-negatives")
    w = M.encoder(M.evidence(anim.study)[1], "hardest0")
    u, v = SR.embed(w, anim.a), SR.embed(w, anim.b)
    shuffled = np.random.default_rng(0).permutation(anim.labels)
    for mode in ("hardest", "skip"):
        mask, _ = SR.choose(u, v, mode, None, anim.labels)
        np.testing.assert_array_equal(mask, SR.choose(u, v, mode, None, shuffled)[0])
        assert np.all(mask.sum(1) == SR.NEGATIVES + 1) and np.all(np.diag(mask))
    # the control does read them: none of its negatives is the anchor's digit
    mask, false = SR.choose(u, v, "clean", None, anim.labels)
    same = anim.labels[:, None] == anim.labels[None]
    assert false == 0.0 and not (mask & same & ~np.eye(len(mask), dtype=bool)).any()


@needs("hard-negatives")
def test_hard_negatives_batch_is_the_studys_first_batch():
    anim, study = built("hard-negatives")
    _settings, arrays = M.evidence(study)
    false = arrays["history_false"]                      # mode, seed, update
    # mined before training, the batch drawn is the hardest run's first update
    assert anim.mined_false[0] == pytest.approx(float(false[1, 0, 0]))
    assert np.mean(anim.labels[anim.lot] == anim.digit_of) * SR.NEGATIVES == \
        sum(anim.labels[j] == anim.digit_of for j in anim.lot)
    assert len(anim.lot) == SR.NEGATIVES and anim.ANCHOR not in anim.lot
    for at in (0, anim.updates):
        cos = anim.cos[at]
        assert anim.ANCHOR not in anim.rank[at]
        assert list(cos) == sorted(cos, reverse=True)
        assert anim.same[at] == [anim.labels[j] == anim.digit_of for j in anim.rank[at]]
    # the pair drawn is two different views of one training digit
    assert not np.allclose(anim.a[anim.ANCHOR], anim.b[anim.ANCHOR])
    assert anim.mined_false[anim.updates] > anim.mined_false[0]


@needs("hard-negatives")
def test_hard_negatives_shows_the_studys_numbers():
    anim, study = built("hard-negatives")
    acc = [float(v) for v in study.panels[3].data["values"]]
    texts = shown(anim, anim.spans[4][1] + 1.0)
    for v in acc:
        assert M.percent(v) in texts
    for m in (0, 1, 3):
        assert M.percent(acc[m]) in anim.subtitle_for(4)
    for m in (0, 1, 2):
        assert M.percent(acc[m]) in anim.subtitle_for(3)
        assert M.percent(anim.trace[m, -1]) in shown(anim, middle(anim, 3, 0.95))
    texts = shown(anim, middle(anim, 2, 0.95))
    assert "%s, then %s" % (M.percent(anim.first[1], 0), M.percent(anim.last[1], 0)) in texts
    assert M.percent(anim.last[1], 0) in anim.subtitle_for(2)
    assert "every seed" in anim.subtitle_for(4) and "hold it back" in anim.subtitle_for(4)


# ---------------------------------------------------------------------- 85
@needs("dimensionality-of-embeddings")
def test_width_neighbours_reproduce_the_studys_accuracy():
    anim, study = built("dimensionality-of-embeddings")
    dims = [int(v) for v in study.panels[1].data["x"]]
    separate = np.asarray(study.panels[1].data["ys"][0], float)
    nested = np.asarray(study.panels[2].data["ys"][1], float)
    for n, d in enumerate(dims):
        assert np.mean(anim.y[anim.near[d]] == anim.yt) == pytest.approx(separate[n])
        assert np.mean(anim.y[anim.near_cut[d]] == anim.yt) == pytest.approx(nested[n])
        assert len(anim.vectors[d]) == d and anim.bytes[n] == 4 * d
    assert len(anim.wide) == dims[-1]


@needs("dimensionality-of-embeddings")
def test_width_shows_the_studys_numbers():
    anim, study = built("dimensionality-of-embeddings")
    dims = [int(v) for v in study.panels[1].data["x"]]
    separate = np.asarray(study.panels[1].data["ys"][0], float)
    nested = np.asarray(study.panels[2].data["ys"][1], float)
    assert M.percent(separate[0]) in anim.subtitle_for(1)
    assert M.percent(separate.max()) in anim.subtitle_for(2)
    i16 = dims.index(16)
    assert M.percent(nested[i16]) in anim.subtitle_for(3)
    assert M.percent(separate[i16]) in anim.subtitle_for(3)
    texts = shown(anim, middle(anim, 2, 0.95))
    for v, b in zip(separate, study.panels[3].data["values"]):
        assert M.percent(v) in texts and f"{int(b):,}" in texts
    texts = shown(anim, anim.spans[3][1] + 1.0)
    assert "cut to %d: %s right" % (dims[-1], M.percent(nested[-1])) in texts
    assert "built at %d: %s right" % (dims[-1], M.percent(separate[-1])) in texts


def test_log_sweep_reaches_each_value_in_turn():
    xs = [1, 2, 4, 8, 16, 32]
    assert M.log_sweep(xs, 0.0) == (0.0, 0)
    assert M.log_sweep(xs, 1.0) == (pytest.approx(1.0), 5)
    assert [M.log_sweep(xs, k / 5)[1] for k in range(6)] == list(range(6))
    assert M.log_sweep(xs, 0.5)[1] == 2


def test_clip_cuts_a_long_line_at_a_word_and_says_so():
    from voxanim import layout

    assert M.clip("a short line", 0.029, 0.5) == "a short line"
    assert M.clip("a short line", 0.029) == "a short line"
    line = " ".join(["word"] * 40)
    cut = M.clip(line, 0.029, 0.2)
    assert cut.endswith(" ...") and line.startswith(cut[:-4])
    assert layout.text_width(cut, 0.029) <= 0.2

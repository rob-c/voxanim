"""Animations 21-28 of the "next 100" (voxanim.mech_training): their layout,
that the numbers they show are the study's, and that what they compute
themselves is what the study's model, data or rule gives."""

import numpy as np
import pytest
from conftest import available

from voxanim import mech_training as M
from voxanim import presets, qa

PRESETS = {
    "teacher-forcing": "teacher-forcing",
    "perplexity-explained": "gpt2-perplexity-explained",
    "packing-and-padding": "packing-and-padding",
    "warmup-and-clipping": "warmup-and-clipping",
    "grokking": "grokking",
    "scaling-laws": "scaling-laws",
    "curriculum-and-data-mixing": "curriculum-and-data-mixing",
    "tokenizer-effects": "tokenizer-effects",
}


def needs(name):
    return pytest.mark.skipif(not available(PRESETS[name]), reason="data for %s not downloaded" % name)  # noqa: E501


def build(name):
    return presets.get(PRESETS[name]).build()


def subtitles(anim):
    return [anim.subtitle_for(k) for k in range(len(anim.PHASES))]


@pytest.mark.parametrize("name", [pytest.param(n, marks=needs(n)) for n in PRESETS])
def test_layout_and_loop(name):
    assert set(PRESETS) == set(M.ANIMATIONS)
    assert qa.check(PRESETS[name], samples=24, verbose=False) == 0
    anim = build(name)
    assert isinstance(anim, M.ANIMATIONS[name])
    assert 45.0 <= anim.t_done <= 60.0
    first, last = anim.scene(0.0), anim.scene(anim.duration)
    assert [t.s for t in first.texts if t.opacity > 0.02] == [t.s for t in last.texts if t.opacity > 0.02]  # noqa: E501
    assert all(s and len(s) < 230 for s in subtitles(anim))


@needs("teacher-forcing")
def test_teacher_forcing():
    from voxanim import next_language as nl
    from voxanim import tinylm as lm

    anim = build("teacher-forcing")
    s = anim.study
    by_pos = np.asarray(s.panels[3].data["ys"])
    assert "%.2f nats reading the recording, %.2f reading" % (
        by_pos[0, 1:].mean(), by_pos[1, 1:].mean()) in anim.subtitle_for(4)
    assert "from %.2f to %.2f" % (s.panels[2].data["ys"][0][0], s.panels[2].data["ys"][0][-1]) \
        in anim.subtitle_for(2)
    assert np.array_equal(anim.mask, lm.attention_mask(8, lm.Config()))
    # the rollout: the checkpoint is the study's (it reproduces its held-out losses), every
    # free-running character is the model's own greedy choice given the ones before it
    r = M.teacher_rollout()
    seq = nl._text_windows(256, 16)
    c = lm.Config(vocab=nl._corpus()["vocab"], width=24, heads=4, kv_heads=4)
    P = lm.train(c, seq[:, :-1], seq[:, 1:], steps=350, seed=nl.SEED, name="real-teacher-v8")["P"]
    test = nl._text_windows(96, 16, "test", nl.SEED + 1)
    assert np.allclose(nl.nll(lm.forward(P, test[:, :-1], c), test[:, 1:]).mean(0), by_pos[0])
    assert np.array_equal(r["recorded"], test[1])
    assert r["free"][0] == test[1, 0]
    for i in range(1, 16):
        assert r["free"][i] == lm.forward(P, r["free"][None, :i], c)[0, -1].argmax()
        assert r["forced"][i] == lm.forward(P, test[1:2, : i + 1], c)[0, -1].argmax()


@needs("perplexity-explained")
def test_perplexity():
    from voxanim import gpt2_hooks
    from voxanim import next_language as nl

    anim = build("perplexity-explained")
    s = anim.study
    ppl = s.panels[2].data["values"]
    assert "perplexity %s, a bigram count %s, GPT-2 %.0f" % (
        f"{ppl[0]:,.0f}", f"{ppl[1]:,.0f}", ppl[2]) in anim.subtitle_for(2)
    assert "%.2f nats" % s.panels[1].data["values"][2] in anim.subtitle_for(1)
    assert np.isclose(np.exp(anim.mean[2]), anim.ppl[2])
    # the tokens drawn are the ones whose surprise the study recorded
    words = M.perplexity_tokens()
    assert [w.strip()[:10] or "space" for w in words[1:]] == list(s.panels[0].data["labels"])
    ids = nl._tok().encode(nl._corpus()["test"][:100])
    assert [nl._tok().decode([i]) for i in ids[:9]] == words
    again = nl.nll(gpt2_hooks.forward(ids)["logits"][:-1], ids[1:])[:8]
    assert np.allclose(again, anim.surprise, atol=1e-6)
    assert M.one_in(np.exp(-anim.surprise[7])) == "1 in %s" % M.number(np.exp(anim.surprise[7]))


@needs("packing-and-padding")
def test_packing():
    from voxanim import next_language as nl

    anim = build("packing-and-padding")
    s = anim.study
    assert "%.3f nats one at a time, %.3f packed with no resets, %.3f packed with both" % tuple(
        s.panels[3].data["values"]) in anim.subtitle_for(4)
    used, wasted = (int(v) for v in s.panels[1].data["values"])
    assert "%d characters of text and %d cells of padding" % (used, wasted) in anim.subtitle_for(0)
    # the lines drawn are the study's twelve; the masks are its packing rule
    assert [len(line) for line in anim.source] == list(s.panels[0].data["values"])
    assert sum(len(line) for line in anim.source) == used
    causal, leak = M.packed_masks(8)
    assert np.array_equal(causal & ~leak, np.asarray(s.panels[2].data["values"]) > 0.5)
    assert leak.sum() == 64 and causal.sum() == 136
    test = nl._text_windows(64, 8, "test", nl.SEED + 1)
    ex = M.packing_example()
    assert np.array_equal(ex["a"], test[1, :8]) and np.array_equal(ex["b"], test[33, :8])


@needs("warmup-and-clipping")
def test_warmup():
    from voxanim import study_training as st

    anim = build("warmup-and-clipping")
    s = anim.study
    mean = s.panels[3].data["values"]
    assert "fixed rate %.2f, warm-up %.2f, warm-up and clipping %.2f nats" % tuple(
        mean[:3]) in anim.subtitle_for(3)
    assert "also trains (%.2f)" % mean[3] in anim.subtitle_for(3)
    assert anim.ramp_len == 80 and anim.peak == 0.006 and anim.layers == 6
    r = st.warmup_runs()
    # the gradient lengths and clip ratios drawn are the ones the clipped run recorded and applied
    assert np.array_equal(anim.norm, r["norm"][2, 0]) and np.array_equal(anim.ratio, r["ratio"][2, 0])  # noqa: E501
    assert np.allclose(M.clip_ratios(r["norm"][2]), r["ratio"][2])
    assert np.all(r["ratio"][[0, 1, 3]] == 1.0)
    assert anim.clipped == int((r["ratio"][2, 0] < 1).sum()) > 0
    assert "%d of the 300 updates" % anim.clipped in anim.subtitle_for(1)
    # the counts on screen follow the study's rule for a run that learned
    learned = [int(v) for v in s.panels[3].data["learned"]]
    assert "rate %d of 5 get under" % learned[0] in anim.subtitle_for(2)
    assert "with warm-up %d of 5" % learned[1] in anim.subtitle_for(2)
    assert [anim.below(arm, len(anim.held_at) - 1) for arm in range(4)] == [
        int((r["held"][arm, :, -1] < r["unigram"] - 0.2).sum()) for arm in range(4)]


@needs("warmup-and-clipping")
def test_warmup_study_fixed_rate_stays_on_the_plateau_and_warmup_leaves_it():
    from voxanim import realdata
    from voxanim import study_training as st

    s = st.STUDIES["warmup-and-clipping"]().validate()
    r = st.warmup_runs()
    best = np.asarray(s.panels[3].data["each"])              # arm, seed: each run's best point
    fixed, warm, both, half = best
    assert best.shape == (4, 5) and np.array_equal(best, r["held"].min(-1))
    # the claim, seed by seed and with room: at 0.006 no fixed-rate run learns, every warmed-up
    # run does, and clipping on top does not cost anything
    assert fixed.min() > max(warm.max(), both.max()) + 0.5
    assert fixed.min() > r["unigram"] - 0.2 and max(warm.max(), both.max()) < r["unigram"] - 0.6
    assert list(s.panels[3].data["learned"]) == [0, 5, 5, 5]
    assert both.mean() < warm.mean() + 0.05
    # warm-up buys the higher rate; here it is not better than half the rate without it
    assert abs(warm.mean() - half.mean()) < 0.1
    # integrity: arms share each seed's start, the schedules are the ones named, the plateau is
    # the character-frequency loss, and training and held-out text do not overlap
    assert np.allclose(r["held"][:, :, 0], r["held"][0, :, 0])
    assert np.allclose(r["rate"][0], 0.006) and np.allclose(r["rate"][3], 0.003)
    assert np.allclose(r["rate"][1], 0.006 * np.minimum(1, np.arange(1, 301) / 80))
    assert np.array_equal(r["rate"][1], r["rate"][2])
    assert np.isfinite(r["loss"]).all() and np.isfinite(r["norm"]).all()
    held = realdata.text_sequences(256, 17, "test", st.SEED + 1)
    assert np.isclose(st.unigram_loss(held), float(r["unigram"])) and 3.2 < r["unigram"] < 3.5
    train, test = (realdata.text_metadata(k)["character_range"] for k in ("train", "test"))
    assert train[1] <= test[0]
    assert len(s.metadata["designs_tried"]) >= 3


@needs("grokking")
def test_grokking():
    from voxanim import study_training as st

    anim = build("grokking")
    s = anim.study
    r = st.grokking_runs()
    d = s.panels[3].data
    assert "training at 99%% by update %s, test at 85%% by update %s, %.0f times later" % (
        "{:,.0f}".format(d["fitted"][0].mean()), "{:,.0f}".format(d["generalised"][0].mean()),
        d["generalised"][0].mean() / d["fitted"][0].mean()) in anim.subtitle_for(4)
    assert "passes 85%% near update %.0f, before training reaches 99%% near update %.0f" % (
        d["generalised"][1].mean(), d["fitted"][1].mean()) in anim.subtitle_for(3)
    # the walls are seed 0 of each arm: their lit cells are the accuracies the study plots
    assert np.array_equal(r["step"], np.asarray(s.panels[1].data["x"]).astype(int))
    for arm in (0, 1):
        assert np.allclose(r["train_ok"][arm].mean(1), s.panels[1].data["train"][arm, 0])
        assert np.allclose(r["test_ok"][arm].mean(1), s.panels[1].data["test"][arm, 0])
    k = anim.fit_k
    assert anim.steps[k] == d["fitted"][0, 0] and r["train_ok"][0, k].mean() >= 0.99
    assert r["train_ok"][0, k - 1].mean() < 0.99
    assert "%d of the 1,000 training images right, but only %d of the 2,000 test" % (
        r["train_ok"][0, k].sum(), r["test_ok"][0, k].sum()) in anim.subtitle_for(1)
    assert anim.n_train == 1000 and anim.n_test == 2000 and anim.sizes == [196, 200, 200, 10]


@needs("grokking")
def test_grokking_study_test_accuracy_arrives_late_only_from_the_large_start():
    from voxanim import study_training as st

    s = st.STUDIES["grokking"]().validate()
    r = st.grokking_runs()
    d = st.grokking_delays(r)
    large, default = 0, 1
    assert d["reached"].all()
    # the claim, in every seed and with room: from the large start the test set is still mostly
    # wrong when the training set is learned and gets to 85% several times later ...
    assert np.all(d["generalised"][large] >= 3 * d["fitted"][large])
    assert d["test_then"][large].max() < 0.6
    assert r["test"][large, :, -1].min() > 0.85
    # ... while the weights shrink (by more than a third after the training set is learned) ...
    for seed in range(3):
        k = int(np.searchsorted(r["step"], d["fitted"][large, seed]))
        assert r["norm"][large, seed, -1] < 0.67 * r["norm"][large, seed, k]
    # ... and from the default start there is no wait: the test level comes first
    assert np.all(d["generalised"][default] <= d["fitted"][default])
    assert d["test_then"][default].min() > 0.85
    # integrity: same start up to the scale, official partitions, the run that was described
    assert np.allclose(r["norm"][large, :, 0], 3.0 * r["norm"][default, :, 0])
    assert r["step"][0] == 0 and r["step"][-1] == 12000 and np.all(np.diff(r["step"]) > 0)
    assert r["train_ok"].shape[2] == 1000 and r["test_ok"].shape[2] == 2000
    meta = s.metadata["dataset_provenance"]
    assert meta["train"]["official_partition"] == "train" and meta["test"]["official_partition"] == "test"  # noqa: E501
    assert max(meta["train"]["example_indices"]) < 50000
    assert np.array_equal(s.panels[3].data["fitted"], d["fitted"])
    assert len(s.metadata["designs_tried"]) >= 3


@needs("scaling-laws")
def test_scaling():
    from voxanim import study_training as st
    from voxanim import tinylm as lm

    anim = build("scaling-laws")
    s = anim.study
    loss = np.asarray(s.panels[0].data["values"])
    ran = np.asarray(s.panels[0].data["ran"])
    assert "falls from %.2f to %.2f nats" % (loss.max(), loss[ran].min()) in anim.subtitle_for(1)
    assert "%d runs" % ran.sum() in anim.subtitle_for(1)
    assert anim.best == [int(np.where(ran[:, c], loss[:, c], np.inf).argmin())
                         for c in range(loss.shape[1])]
    assert anim.front_y[-1] == loss[anim.best[-1], -1]
    # the weights shown are the models' real sizes; the shapes of phase 1 have equal areas
    for width, weights in zip(anim.widths, anim.weights):
        assert weights == sum(v.size for v in lm.init(st._scaling_config(width), 0).values())
    areas = np.asarray(anim.weights) * anim.characters[:, anim.SHOWN]
    assert anim.ran[:, anim.SHOWN].all() and areas.max() / areas.min() < 1.02
    assert "width 8 reads %s characters and width 64 only %s" % (
        f"{anim.characters[0, 4]:,.0f}", f"{anim.characters[3, 4]:,.0f}") \
        in anim.subtitle_for(0)
    # the line drawn is the least-squares power law of the seven cheapest budgets only
    lx, (ly, fit) = np.asarray(s.panels[2].data["x"]), np.asarray(s.panels[2].data["ys"])
    slope, intercept = np.polyfit(lx[:7], ly[:7], 1)
    assert np.isclose(anim.slope, slope) and np.allclose(fit, intercept + slope * lx)
    assert "predicts %.2f nats at the dearest, and the run there measured %.2f" % (
        np.exp(fit[-1]), np.exp(ly[-1])) in anim.subtitle_for(2)
    assert M.stretches([1, 0, 0, 1, 2, 2]) == [(1, 0, 0), (0, 1, 2), (1, 3, 3), (2, 4, 5)]
    assert anim.best_by_budget() in anim.subtitle_for(3)


@needs("scaling-laws")
def test_scaling_study_loss_falls_with_compute_and_the_best_size_grows():
    from voxanim import realdata
    from voxanim import study_training as st

    s = st.STUDIES["scaling-laws"]().validate()
    r = st.scaling_runs()
    f = st.scaling_frontier(r)
    ran, each = f["ran"], r["loss"]                          # each: seed, width, budget
    assert each.shape == (3, 4, 10) and ran.sum() == s.metadata["settings"]["runs"] >= 27
    # compute really varies: budgets double over a 512-fold range and every size meets them
    assert np.allclose(r["compute"], 6.0 * r["weights"][:, None] * r["characters"])
    assert np.allclose(f["compute"][1:] / f["compute"][:-1], 2.0, rtol=0.05)
    for j in range(10):
        spent = r["compute"][ran[:, j], j]
        assert spent.max() / spent.min() < 1.06
    # loss falls with compute: along the frontier at every step, by more than a nat in all, and
    # for every model on its own
    assert np.all(np.diff(f["loss"]) < 0) and f["loss"][0] - f["loss"][-1] > 1.0
    for w in range(4):
        assert np.all(np.diff(f["mean"][w][ran[w]]) < 0)
    # the size-versus-data frontier, in every seed and with room: with little compute the
    # smallest model (more text) beats the next size, with 16 times more it is the other way
    # round; and the widest model never beats the one below it at the same budget
    assert np.all(each[:, 0, 3] < each[:, 1, 3] - 0.03)
    assert np.all(each[:, 1, 6] < each[:, 0, 6] - 0.03)
    assert np.all(each[:, 2, 3:] < each[:, 3, 3:] - 0.02)
    assert f["best"][3] == 0 and f["best"][6] == 1 and f["best"][-1] == 2
    # the power law is fitted to the seven cheapest budgets; on the three it never saw it is too
    # optimistic, by more the further it is carried (the curve bends)
    assert -0.2 < f["slope"] < -0.05
    assert np.all(f["predicted"][7:] < f["loss"][7:] - 0.05) and np.all(np.diff(f["error"]) > 0)
    assert np.array_equal(s.panels[2].data["held_out_error"], f["error"])
    # integrity: nothing unrun is counted, the characters are those the updates read, and
    # training and held-out text do not overlap
    assert np.all(each[:, ~ran] == 0) and np.all(each[:, ran] > 1.0)
    assert np.array_equal(r["characters"], r["updates"] * 32 * 16)
    train, test = (realdata.text_metadata(k)["character_range"] for k in ("train", "test"))
    assert train[1] <= test[0]
    assert len(s.metadata["designs_tried"]) >= 3


@needs("curriculum-and-data-mixing")
def test_mixing():
    anim = build("curriculum-and-data-mixing")
    s = anim.study
    assert "90/10 %.2f, 50/50 %.2f, curriculum %.2f nats" % tuple(
        s.panels[3].data["values"]) in anim.subtitle_for(4)
    assert "%.2f for 90/10" % s.panels[1].data["ys"][0][-1] in anim.subtitle_for(2)
    # the batches drawn follow the study's rule
    for step in (0, 100, 319):
        assert M.mix_counts("90/10", step) == 22 and M.mix_counts("50/50", step) == 12
        assert M.mix_counts("curriculum", step) == int(round(24 * (0.9 - 0.8 * step / 320)))
    assert M.mix_counts("curriculum", 0) == 22 and M.mix_counts("curriculum", 320) == 2


@needs("tokenizer-effects")
def test_tokenizers():
    from voxanim import next_language as nl
    from voxanim import toycorpus

    anim = build("tokenizer-effects")
    s = anim.study
    per_char = s.panels[2].data["values"]
    assert "%.2f, %.2f and %.2f for BPE" % tuple(per_char[1:4]) in anim.subtitle_for(3)
    assert "{:,}".format(int(s.panels[0].data["values"][3])) in anim.subtitle_for(2)
    # the merges drawn are the study's: they give its token counts and table sizes
    d = M.bpe_demo()
    train = nl._corpus()["train"][:18000]
    for merges, count, table in zip((16, 48, 96), s.panels[0].data["values"][1:4],
                                    s.panels[1].data["values"][1:4]):
        assert len(toycorpus.bpe_apply(d["test"], d["rules"][:merges])) == count
        # a table has a row for every token the training text still uses, plus one for unknowns
        used = set(toycorpus.bpe_apply(train, d["rules"][:merges]))
        assert (len(used) + 1) * 16 == table and len(used) <= d["alphabet"] + merges
    assert (d["alphabet"] + 1) * 16 == s.panels[1].data["values"][0]
    assert all("".join(seg) == d["phrase"] for seg in d["segments"])
    assert d["segments"][0] == list(d["phrase"])

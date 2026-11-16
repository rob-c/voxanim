"""Animations 43-50 of the "next 100": what they draw is what was measured."""

import itertools

import numpy as np
import pytest
from conftest import available

from voxanim import mech_interp as M
from voxanim import presets, qa
from voxanim import study_interp as SI
from voxanim.next_catalog import build_study

PRESETS = {
    "superposition": "superposition",
    "sparse-autoencoder-features": "gpt2-sparse-autoencoder-features",
    "linear-probes": "gpt2-linear-probes",
    "activation-patching": "gpt2-activation-patching",
    "induction-heads": "gpt2-induction-heads",
    "head-ablation": "gpt2-head-ablation",
    "saliency-and-integrated-gradients": "saliency-and-integrated-gradients",
    "feature-visualisation": "feature-visualisation",
}


def needs(name):
    return pytest.mark.skipif(not available(PRESETS[name]), reason="data not downloaded")


_built = {}


def built(name):
    """``(animation, study)`` of catalogue ``name``, built once."""
    if name not in _built:
        _built[name] = (presets.get(PRESETS[name]).build(), build_study("next_language", name))
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


@pytest.mark.parametrize("name", [pytest.param(n, marks=needs(n)) for n in PRESETS])
def test_registered_laid_out_and_looping(name):
    anim, _study = built(name)
    assert isinstance(anim, M.ANIMATIONS[name])
    assert 45.0 <= anim.spans[-1][1] <= 60.0
    assert qa.check(PRESETS[name], samples=24, verbose=False) == 0
    first, last = anim.scene(0.0), anim.scene(float(anim.duration))
    key = lambda sc: sorted((t.s, round(float(t.opacity), 3)) for t in sc.texts)  # noqa: E731
    assert key(first) == key(last)


# ---------------------------------------------------------------------- 43
@needs("superposition")
def test_superposition_directions_are_the_studys():
    anim, study = built("superposition")
    d = M.superposition_runs()
    # the trainings run again here end where the study's did
    np.testing.assert_allclose(d["W"][0, -1], study.panels[0].data["xy"].T, atol=1e-12)
    np.testing.assert_allclose(d["W"][3, -1], study.panels[1].data["xy"].T, atol=1e-12)
    np.testing.assert_allclose(d["heldout_loss"], study.metadata["heldout_loss"], atol=1e-12)
    # and the interference matrix drawn is the Gram matrix of the sparse run's directions
    np.testing.assert_allclose(d["W"][3, -1].T @ d["W"][3, -1], anim.gram, atol=1e-12)
    assert d["W"].shape == (4, 101, 2, 5)
    assert np.allclose(d["W"][:, 0], d["W"][0, 0])            # every run starts the same
    held = study.metadata["heldout_loss"]
    zeros = study.metadata["actual_zero_fraction"]
    assert "%.3f to %.4f" % (held[0], held[3]) in anim.subtitle_for(4)
    assert "%.0f%% to %.0f%%" % (100 * zeros[0], 100 * zeros[3]) in anim.subtitle_for(4)
    late = shown(anim, phase_time(anim, 4, 0.95))
    for z in zeros:
        assert has(late, "%.0f%% zeros" % (100 * z))
    # the squeezed digit: two numbers in, the dense model's own reconstruction out
    x = np.where(d["features"] > d["thresholds"][0], d["features"], 0)
    np.testing.assert_allclose(anim.h, d["W"][0, -1] @ x)
    assert has(shown(anim, phase_time(anim, 0, 0.97)), "%.2f" % anim.h[1])


@needs("superposition")
def test_superposition_pentagon_is_measured():
    anim, study = built("superposition")
    W = np.asarray(study.panels[1].data["xy"], float).T
    angles = np.sort(np.degrees(np.arctan2(W[1], W[0])) % 360)
    gaps = np.diff(np.r_[angles, angles[0] + 360])
    assert abs(gaps.sum() - 360) < 1e-9
    assert "%.0f to %.0f degrees" % (gaps.min(), gaps.max()) in anim.subtitle_for(2)
    gram = np.asarray(study.panels[2].data["values"])
    off = gram[~np.eye(5, dtype=bool)]
    assert "%.2f" % off.max() in anim.subtitle_for(3)
    assert has(shown(anim, phase_time(anim, 3, 0.93)), "%+.2f" % gram[4, 0])


# ---------------------------------------------------------------------- 44
@needs("sparse-autoencoder-features")
def test_sae_features_are_alive_and_some_read_as_one_word():
    """The redesigned study's claims, all on held-out text, with margin."""
    _, study = built("sparse-autoencoder-features")
    d = SI.sae_data()
    m = study.metadata["summary"]
    assert m == SI.sae_summary(d)
    assert m["dead"] < 0.05 and m["alive"] > 0.95 * len(d["fires"])
    assert float(d["unexplained"]) < 0.35                       # most of the variance rebuilt
    assert float(d["unexplained"]) < float(d["train_unexplained"]) + 0.15
    # 20 random tokens agree on about one in seven; a readable feature's top 20 on four in five
    assert m["chance"] < 0.25 and m["readable_share"] > 0.10
    k, left = [int(v) for v in d["sweep_k"]], [float(v) for v in d["sweep_unexplained"]]
    assert k == sorted(k) and all(a > b + 0.02 for a, b in itertools.pairwise(left))
    np.testing.assert_allclose(study.panels[1].data["ys"][0], left)
    np.testing.assert_allclose(study.panels[3].data["values"][0], d["unexplained"])
    assert study.panels[2].data["values"].sum() == m["scored"]


@needs("sparse-autoencoder-features")
def test_sae_data_is_held_out_and_first_tokens_are_left_out():
    from voxanim import realdata

    _, study = built("sparse-autoencoder-features")
    d = SI.sae_data()
    c = SI.SAE
    assert int(d["train_tokens"]) == c["train_windows"] * (c["tokens"] - 1)
    assert int(d["heldout_tokens"]) == c["heldout_windows"] * (c["tokens"] - 1)
    assert study.metadata["first_token_norm_ratio"] > 10        # why they are left out
    test, train = realdata.text("test"), realdata.text("train")
    held = str(d["heldout_text"])
    assert held in test and held not in train
    assert held.endswith("".join(str(t) for t in d["tokens"]))
    # a top-k code: never more than k features on a token, and the shown ones are the encoder's
    assert ((d["h"] > 0).sum(1) <= c["k"]).all() and (d["h"] >= 0).all()
    for j in study.metadata["summary"]["picks"]:
        assert d["fires"][j] >= c["top"]
        assert np.all(np.diff(d["top_values"][j]) <= 0) and d["top_values"][j][0] == d["peak"][j]
        for row in d["contexts"][j]:
            assert "".join(str(w) for w in row) in test


@needs("sparse-autoencoder-features")
def test_sae_animation_states_the_measurements():
    anim, study = built("sparse-autoencoder-features")
    d = SI.sae_data()
    m = study.metadata["summary"]
    assert sum(anim.counts) == len(d["fires"]) == 1024
    assert anim.counts[0] == round(m["dead"] * 1024) and anim.counts[1] == m["readable"]
    assert "%.0f%% of the variance" % (100 * (1 - float(d["unexplained"]))) in anim.subtitle_for(1)
    assert "%d of %d features fire on one word" % (m["readable"], m["scored"]) \
        in anim.subtitle_for(2)
    assert "%d of the 1024 features are silent" % anim.counts[0] in anim.subtitle_for(3)
    # the token rewritten on screen is the autoencoder's own code for it
    i = anim.TURNS[0] - 1
    late = shown(anim, phase_time(anim, 1, 0.9 / (len(anim.TURNS) * 1.03)))
    assert has(late, "%d of 1024 fire" % (anim.h[i] > 0).sum())
    assert anim.words == [M.show(t) for t in d["tokens"][anim.FROM:anim.FROM + 10]]
    mid = shown(anim, phase_time(anim, 2, 0.97))
    for j in m["picks"]:
        assert has(mid, "feature %d" % j) and has(mid, "fires on %d held-out" % d["fires"][j])
    end = shown(anim, phase_time(anim, 3, 0.97))
    assert has(end, "%d" % anim.counts[1]) and has(end, "%d" % anim.counts[2])
    for k, v in zip(d["sweep_k"], d["sweep_unexplained"]):
        assert has(end, "%d features per token: %.0f%%" % (k, 100 * v))


# ---------------------------------------------------------------------- 45
@needs("linear-probes")
def test_probe_run_is_the_studys():
    anim, study = built("linear-probes")
    d = M.probe_run()
    accuracy = study.panels[1].data["ys"][0]
    np.testing.assert_allclose(d["accuracy"], accuracy, atol=1e-12)
    np.testing.assert_allclose(d["control"], study.panels[2].data["ys"][0], atol=1e-12)
    np.testing.assert_allclose(d["labels"].mean(), study.panels[0].data["values"][1])
    # the dots drawn give the accuracy drawn: balanced over the two classes
    y = d["labels"].astype(bool)
    for layer in (0, 5, 12):
        pred = d["scores"][layer] > 0
        assert abs(np.mean([pred[y].mean(), (~pred[~y]).mean()]) - accuracy[layer]) < 1e-12
    assert "%.2f at layer 0" % accuracy[0] in anim.subtitle_for(1)
    assert "%.2f" % study.panels[2].data["ys"][0][0] in anim.subtitle_for(3)
    assert has(shown(anim, phase_time(anim, 1, 0.95)), "balanced accuracy %.2f" % accuracy[12])
    assert has(shown(anim, phase_time(anim, 2, 0.95)), "%.2f" % accuracy.max())


@needs("linear-probes")
def test_probe_labels_on_screen_follow_the_text():
    anim, _ = built("linear-probes")
    d = M.probe_run()
    words = [str(t) for t in d["tokens"][2:14]]
    for i in range(1, len(words)):
        assert bool(anim.word_yes[i - 1]) == any(ch in words[i - 1] for ch in ".,;:!?")
    assert anim.words == [M.show(w) for w in words[1:]]


# ---------------------------------------------------------------------- 46
@needs("activation-patching")
def test_patching_prompts_and_numbers_are_the_studys():
    anim, study = built("activation-patching")
    clean, corrupt, target = M.patching_prompts()
    s = study.metadata["settings"]
    assert "".join(clean) == s["clean"] and "".join(corrupt) == s["corrupted"]
    assert target == s["target"]
    assert anim.swapped == [2, 3]
    p_clean, p_bad = study.panels[0].data["values"]
    assert "%.3f" % p_clean in anim.subtitle_for(0) and "%.3f" % p_bad in anim.subtitle_for(0)
    matrix = np.asarray(study.panels[1].data["values"])
    recovery = np.asarray(study.panels[2].data["values"])
    np.testing.assert_allclose(recovery, (matrix - p_bad) / (p_clean - p_bad), atol=1e-9)
    # the single patches followed land on the study's probabilities
    for turn, (layer, k) in enumerate(anim.SHOWN):
        late = shown(anim, phase_time(anim, 1, (turn + 0.97) / (3 * 1.02)))
        assert has(late, "p = %.3f" % matrix[layer, k])
        assert has(late, "%.0f%% of the gap recovered" % (100 * recovery[layer, k]))
    # tokens before the swap are the same in both runs: patching them changes nothing
    assert np.allclose(recovery[:, :2], 0)


@needs("activation-patching")
def test_patching_path_is_the_argmax_of_the_heatmap():
    anim, study = built("activation-patching")
    recovery = np.asarray(study.panels[2].data["values"])
    assert list(anim.path) == list(recovery.argmax(1))
    covered = [layer for _, a, b in anim.runs for layer in range(a, b + 1)]
    assert covered == list(range(13))
    for k, a, b in anim.runs:
        assert all(recovery[layer].argmax() == k for layer in range(a, b + 1))
        assert "layers %d to %d" % (a, b) in anim.subtitle_for(3)


# ---------------------------------------------------------------------- 47
@needs("induction-heads")
def test_induction_attention_is_the_studys():
    anim, study = built("induction-heads")
    d = M.induction_run()
    mass = np.asarray(study.panels[3].data["values"])
    np.testing.assert_allclose(d["mass"], mass, atol=1e-7)
    # the head followed is one of the study's five strongest, and its map is GPT-2's
    assert anim.head in anim.best
    assert np.allclose(anim.attn.sum(-1), 1.0, atol=1e-4)
    assert np.allclose(np.triu(anim.attn, 1), 0)
    targets = M.successor_targets([int(i) for i in d["ids"]])
    score = np.mean([anim.attn[t, js].sum() for t, js in targets.items()])
    assert abs(score - mass[anim.head]) < 1e-6
    # the repeat drawn is in the text: each query's token appeared 17 back
    for q in anim.QUERIES:
        assert d["ids"][q] == d["ids"][q - 17] and targets[q][-1] == q - 16
    assert "%.2f" % anim.attn[24, 8] in anim.subtitle_for(1)
    assert "%.2f" % mass.max() in anim.subtitle_for(2)
    late = shown(anim, phase_time(anim, 1, (1 + 0.9) / (5 * 1.02)))
    assert has(late, "puts %.2f of its attention" % anim.attn[24, 8])


@needs("induction-heads")
def test_small_model_grows_an_induction_head_only_on_repeated_lines():
    """Three seeds each: trained on repeated lines the head forms, on ordinary text not."""
    _, study = built("induction-heads")
    m = study.metadata["summary"]
    twice, plain = m["twice"], m["plain"]
    assert len(twice["best_head"]) == len(plain["best_head"]) == 3
    assert min(twice["best_head"]) > max(plain["best_head"]) + 0.3
    assert max(plain["best_head"]) < 0.1
    assert max(twice["later_loss"]) < min(plain["later_loss"]) - 1.0
    assert min(twice["later_right"]) > max(plain["later_right"]) + 0.3
    # the repeat is what it uses: a first reading is no easier than for the control
    assert min(twice["first_loss"]) > max(twice["later_loss"]) + 1.0
    assert all(0 < step < study.metadata["settings"]["updates"] for step in m["appears_at"])
    d = SI.induction_data()
    assert m == SI.induction_summary(d)
    np.testing.assert_allclose(study.panels[1].data["ys"][0], d["later"][0].mean(0))
    np.testing.assert_allclose(study.panels[2].data["ys"][:3], d["copy"][0, :, :, 1].max(-1))
    # every model starts from the same loss and no induction head
    assert np.all(d["copy"][:, :, 0].max((-1, -2)) < 0.1)


@needs("induction-heads")
def test_small_induction_windows_are_real_held_out_text_read_again():
    from voxanim import realdata

    anim, study = built("induction-heads")
    d = SI.induction_data()
    test = realdata.text("test")
    held, lengths = d["held"], d["held_len"]
    lo, hi = SI.INDUCTION["repeat"]
    assert lengths.min() >= lo and lengths.max() <= hi and len(set(lengths.tolist())) > 5
    for row, length in zip(held, lengths):
        assert np.array_equal(row[length:], row[:-length])           # the stretch, repeated
        assert realdata.decode(row[:length]) in test
    np.testing.assert_array_equal(study.panels[0].data["values"], held[:8])
    assert np.allclose(d["attention"].sum(-1), 1.0)
    # what is drawn: a character of the second reading, and what followed it the first time
    q, length = anim.small_query, anim.length
    assert length <= q < 2 * length and anim.small_targets == [q - length + 1]
    assert anim.window[q - length] == anim.window[q]
    assert anim.window[anim.small_targets[0]] == anim.window[q + 1]
    assert abs(anim.small_row[anim.small_targets].sum() - anim.small_share) < 1e-12
    assert "%.2f of its" % anim.small_share in anim.subtitle_for(3)
    late = shown(anim, phase_time(anim, 3, 0.95))
    assert has(late, "probability %.2f" % anim.p_next[0])
    assert has(late, "%.2f of its attention there" % anim.control_share)
    m = study.metadata["summary"]
    assert "falls to %.2f nats" % np.mean(m["twice"]["later_loss"]) in anim.subtitle_for(4)
    assert "scores %.2f" % max(m["plain"]["best_head"]) in anim.subtitle_for(4)


# ---------------------------------------------------------------------- 48
@needs("head-ablation")
def test_ablation_ranking_holds_across_passages():
    """The redesigned study's claim: with 64 passages the per-head changes are stable."""
    _, study = built("head-ablation")
    d = SI.ablation_data()
    m = SI.ablation_summary(d)
    change = d["change"]
    assert change.shape == (64, 12, 12)
    np.testing.assert_allclose(study.panels[1].data["values"], change.mean(0), atol=1e-12)
    np.testing.assert_allclose(study.panels[2].data["values"],
                               change.std(0, ddof=1) / np.sqrt(len(change)), atol=1e-12)
    np.testing.assert_allclose(study.panels[3].data["xy"],
                               np.c_[change[0::2].mean(0).ravel(), change[1::2].mean(0).ravel()])
    assert study.metadata["rank_correlation"] == m["rank_correlation"] > 0.6
    assert m["random_halves"].min() > 0.4                    # not an accident of odd and even
    # one passage alone ranks the heads far worse than half of them do
    assert study.metadata["single_passage_rank_correlation"] < m["rank_correlation"] - 0.2
    mean, sem = m["mean"], m["sem"]
    worst, eased = np.unravel_index(mean.argmax(), mean.shape), np.unravel_index(
        mean.argmin(), mean.shape)
    assert mean[worst] > 4 * sem[worst] and -mean[eased] > 4 * sem[eased]
    assert study.metadata["heads_beyond_two_standard_errors"] == int(m["clear"].sum()) >= 36


@needs("head-ablation")
def test_ablation_passages_are_held_out_and_the_fast_forward_is_gpt2():
    from voxanim import gpt2, gpt2_hooks, realdata

    anim, study = built("head-ablation")
    d = SI.ablation_data()
    test = realdata.text("test")
    passages = study.metadata["settings"]["passages"]
    assert len(set(passages)) == len(passages) == 64 and all(p in test for p in passages)
    assert len(set(d["starts"].tolist())) == 64
    np.testing.assert_allclose(study.panels[0].data["values"], d["base"].mean(1))
    # the batched forward used for the 9,216 ablations against the instrumented one
    ids = d["ids"][0]
    layer, head = anim.SHOWN[0]

    def loss(**kw):
        p = gpt2.softmax(gpt2_hooks.forward(ids[:-1], **kw)["logits"])
        return -np.log(p[np.arange(len(ids) - 1), ids[1:]]).mean()
    base = loss()
    assert abs(base - d["base"][0].mean()) < 2e-3
    assert abs(loss(ablate={(layer, head): 0}) - base - d["change"][0, layer, head]) < 2e-3


@needs("head-ablation")
def test_ablation_animation_states_the_measurements():
    anim, study = built("head-ablation")
    d = SI.ablation_data()
    change = np.asarray(study.panels[1].data["values"])
    sem = np.asarray(study.panels[2].data["values"])
    base = float(d["base"][0].mean())
    assert "is %.2f nats" % base in anim.subtitle_for(0)
    layer, head = anim.SHOWN[0]
    assert "from %.2f to %.2f" % (base, base + d["change"][0, layer, head]) \
        in anim.subtitle_for(1)
    late = shown(anim, phase_time(anim, 1, 0.97 / (3 * 1.02)))
    assert has(late, "change: %+.2f nats" % d["change"][0, layer, head])
    assert anim.words == [M.show(t) for t in d["tokens"][0]]
    # the map settles on the study's means, and the spread drawn is the passages'
    np.testing.assert_allclose(anim.running[-1], change, atol=1e-12)
    end = shown(anim, phase_time(anim, 2, 0.97))
    assert has(end, "mean loss change over 64 passages")
    assert has(end, "without head %d.%d: mean %+.2f nats" % (layer, head, change[layer, head]))
    assert "correlation %.2f" % study.metadata["single_passage_rank_correlation"] \
        in anim.subtitle_for(2)
    order = np.argsort(change.ravel())
    assert anim.worst[0] == divmod(int(order[-1]), 12)
    assert anim.eased[0] == divmod(int(order[0]), 12)
    assert "correlation %.2f" % study.metadata["rank_correlation"] in anim.subtitle_for(3)
    last = shown(anim, phase_time(anim, 3, 0.97))
    assert has(last, "%+.2f ± %.2f" % (change.max(), sem[anim.worst[0]]))
    assert has(last, "%+.2f ± %.2f" % (change.min(), sem[anim.eased[0]]))
    assert has(last, "rank correlation %.2f" % study.metadata["rank_correlation"])


# ---------------------------------------------------------------------- 49
@needs("saliency-and-integrated-gradients")
def test_saliency_run_is_the_studys():
    anim, study = built("saliency-and-integrated-gradients")
    d = M.saliency_run()
    gap, total, err = study.panels[3].data["values"]
    assert abs(d["ig"].sum() - total) < 1e-9
    assert abs((d["path_logit"][-1] - d["path_logit"][0]) - gap) < 1e-9
    np.testing.assert_allclose(d["x"].reshape(28, 28), study.panels[0].data["values"][0])
    # the running attribution ends at integrated gradients, and tracks the logit on the way
    np.testing.assert_allclose(d["running"][-1], d["ig"], atol=1e-12)
    rise = d["path_logit"] - d["path_logit"][0]
    assert np.abs(d["running"].sum(1) - rise[1:]).max() < 0.02
    np.testing.assert_allclose(d["times_input"], d["gradient"] * d["x"])
    for part in ("%.3f" % total, "%.3f" % gap, "%.3f" % err):
        assert part in anim.subtitle_for(3)
    end = shown(anim, phase_time(anim, 3, 0.97))
    assert has(end, "sums to %.3f" % total) and has(end, "rose %.3f" % gap)
    mid = shown(anim, phase_time(anim, 2, 0.85 * 0.5))
    assert has(mid, "shares sum to %.2f" % d["running"][255].sum())


# ---------------------------------------------------------------------- 50
@needs("feature-visualisation")
def test_feature_regularisers_make_the_image_more_like_a_digit():
    """The redesigned study's claims, over ten units and three starting noises."""
    _, study = built("feature-visualisation")
    d = SI.feature_data()
    m = study.metadata["summary"]
    assert m == SI.feature_summary(d)
    plain, regular = m["plain"], m["regularised"]
    assert m["trials"] == 30
    # both climbs excite the unit more than any real digit does
    assert plain["objective"] > regular["objective"] > m["real_best"] + 5
    # a network that took no part in the climb reads the regularised images more often
    assert regular["judge_right"] >= plain["judge_right"] + 3
    assert all(r > p for r, p in zip(regular["judge_right_per_seed"],
                                     plain["judge_right_per_seed"]))
    # and they are closer to the average real digit, in every seed
    assert min(regular["likeness_per_seed"]) > max(plain["likeness_per_seed"]) + 0.05
    assert regular["likeness"] < 0.5                 # strokes, not well-formed digits
    assert d["cnn_right"].all()                      # the CNN itself is sure of every one
    assert study.metadata["judge_accuracy"] > 0.95


@needs("feature-visualisation")
def test_feature_data_is_the_cnns():
    from voxanim import mnist

    _, study = built("feature-visualisation")
    d = SI.feature_data()
    params, _ = mnist.cnn_train()
    _, _, test, _ = mnist.load()
    # the gradient climbed is the gradient of the response
    rng = np.random.default_rng(0)
    x = rng.uniform(0.2, 0.8, (1, 28, 28))
    value, grad = SI.class_objective(params, x, 3)
    logits = SI.cnn_logits(params, x)[0]
    assert abs(value[0] - (logits[3] - np.delete(logits, 3).mean())) < 1e-9
    for i, j in ((5, 5), (14, 14), (20, 9)):
        hi, lo = x.copy(), x.copy()
        hi[0, i, j] += 1e-5
        lo[0, i, j] -= 1e-5
        slope = (SI.class_objective(params, hi, 3)[0][0]
                 - SI.class_objective(params, lo, 3)[0][0]) / 2e-5
        assert abs(slope - grad[0, i, j]) < 1e-6
    finals = d["finals"]
    assert finals.min() >= 0 and finals.max() <= 1
    for run in range(2):
        for digit in range(3):
            np.testing.assert_array_equal(d["frames"][run, digit, -1], finals[run, 0, digit])
            np.testing.assert_array_equal(d["frames"][run, digit, 0], d["starts"][0])
            assert abs(SI.class_objective(params, finals[run, 0, digit][None], digit)[0][0]
                       - d["traces"][run, 0, digit, -1]) < 1e-9
    # the real digits shown are the test set's highest scorers on each unit
    for digit in (0, 7):
        scores = np.concatenate([SI.class_objective(params, test[i:i + 2000], digit)[0]
                                 for i in range(0, len(test), 2000)])
        assert int(scores.argmax()) == int(d["real_best"][digit, 0])
        assert abs(scores.max() - d["real_best_value"][digit, 0]) < 1e-9
    np.testing.assert_array_equal(study.panels[0].data["values"], finals[0, 0])
    np.testing.assert_array_equal(study.panels[1].data["values"], finals[1, 0])
    np.testing.assert_allclose(study.panels[3].data["ys"][1], d["traces"][1].mean((0, 1)))


@needs("feature-visualisation")
def test_feature_animation_states_the_measurements():
    anim, study = built("feature-visualisation")
    d = SI.feature_data()
    m = study.metadata["summary"]
    g = anim.DIGIT
    t = d["traces"][:, 0, g]
    assert "a response of %.1f" % d["real_best_value"][g, 0] in anim.subtitle_for(0)
    assert "climbs to %.0f" % t[0, -1] in anim.subtitle_for(1)
    assert "stops at %.0f" % t[1, -1] in anim.subtitle_for(2)
    assert "%d of 30 plain images and %d of 30 regularised" % (
        m["plain"]["judge_right"], m["regularised"]["judge_right"]) in anim.subtitle_for(3)
    assert has(shown(anim, phase_time(anim, 1, 0.84)), "step 197")
    for run, phase in ((0, 1), (1, 2)):
        late = shown(anim, phase_time(anim, phase, 0.9))
        assert has(late, "response %.1f" % t[run, -1])
        assert has(late, "a second network reads: %d" % d["judge"][run, 0, g].argmax())
    end = shown(anim, phase_time(anim, 3, 0.97))
    for name in ("plain", "regularised"):
        assert has(end, "read right %d of 30" % m[name]["judge_right"])
        assert has(end, "response %.0f" % m[name]["objective"])
    assert has(end, "plain %.2f, regularised %.2f" % (m["plain"]["likeness"],
                                                      m["regularised"]["likeness"]))

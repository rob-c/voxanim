"""Animations 35-42 of the "next 100": what they draw is what was measured."""

import itertools

import numpy as np
import pytest
from conftest import available

from voxanim import mech_finetune as M
from voxanim import presets, qa, realdata
from voxanim import study_finetune as F
from voxanim.next_catalog import build_study

PRESETS = {
    "instruction-tuning": "instruction-tuning",
    "reward-model": "reward-model",
    "rlhf-ppo": "rlhf-ppo",
    "dpo": "dpo",
    "reward-hacking": "reward-hacking",
    "prompt-and-prefix-tuning": "prompt-and-prefix-tuning",
    "model-merging": "model-merging",
    "knowledge-editing": "gpt2-knowledge-editing",
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


def on_screen(anim, k, u, text):
    return any(text in s for s in shown(anim, phase_time(anim, k, u)))


@pytest.mark.parametrize("name", [pytest.param(n, marks=needs(n)) for n in PRESETS])
def test_registered_laid_out_and_looping(name):
    anim, _study = built(name)
    assert isinstance(anim, M.ANIMATIONS[name])
    assert 45.0 <= anim.spans[-1][1] <= 60.0
    assert qa.check(PRESETS[name], samples=24, verbose=False) == 0
    first, last = anim.scene(0.0), anim.scene(float(anim.duration))
    key = lambda sc: sorted((t.s, round(float(t.opacity), 3)) for t in sc.texts)  # noqa: E731
    assert key(first) == key(last)
    assert len(first.lines) == len(last.lines)
    for k in range(len(anim.PHASES)):
        assert anim.subtitle_for(k)


# ---------------------------------------------------------------------- 35
@needs("instruction-tuning")
def test_instruction_examples_are_the_play_and_its_answer():
    r = F.instruction_runs()
    caps, test = F.capitals(), realdata.text("test")
    for x, y, block in ((r["example_x"], r["example_y"], 6), (r["show_x"], r["show_y"], 6)):
        seq = np.c_[x, y[:, -1:]]
        assert seq.shape == (block, 5 + 2 * F.LINE + 1)
        for row in seq:
            head, line, answer = realdata.decode(row[:5]), row[5:13], row[14:]
            assert head in F.INSTRUCTIONS and realdata.decode(row[13:14]) == "\n"
            assert realdata.decode(line) in test                 # held-out text, as the play has it
            want = caps[line] if head == "Caps:" else line
            np.testing.assert_array_equal(answer, want)
    # the three shown lines are each asked both ways, and sit inside one line of the play
    assert np.array_equal(r["show_x"][0::2, 5:13], r["show_x"][1::2, 5:13])
    assert "\n" not in realdata.decode(r["show_x"][:, 5:13].ravel())
    assert np.array_equal(r["mask"].sum(1), np.full(6, F.LINE))
    # no window of the training split is in the test split's stretch of the file
    spans = [realdata.text_metadata(k)["character_range"] for k in ("train", "validation", "test")]
    assert spans[0][1] <= spans[1][0] and spans[1][1] <= spans[2][0]


@needs("instruction-tuning")
def test_instruction_tuning_teaches_the_task_and_the_mask_matters():
    """The claim: fine-tuning turns a base that cannot answer into one that can, and the
    answer-only mask gets there far sooner than grading every character."""
    _anim, study = built("instruction-tuning")
    r = F.instruction_runs()
    trace, stop = r["trace"], r["stop"]
    assert trace.shape[:3] == (len(F.SEEDS), 2, 4)
    at_stop = np.take_along_axis(trace, stop[:, :, None, None], 3)[..., 0]     # seed, run, metric
    base = r["base"]
    assert np.all(at_stop[:, 0, 2] > base[:, 2] + 0.5)          # accuracy, every seed
    assert np.all(r["written"][:, 0] > 0.9) and np.all(r["base_written"] < 0.05)
    reach = F.reached(trace)
    assert np.all(reach[:, 0] <= F.TUNE_STEPS)                  # answer-only: every seed gets there
    assert np.all(reach[:, 1] >= reach[:, 0] + 500)             # all-token: much later, or never
    assert np.all(at_stop[:, 0, 2] > at_stop[:, 1, 2] + 0.2)
    # what the all-token loss bought instead: it still predicts the line, the masked one cannot
    assert np.all(trace[:, 1, 3, -1] < trace[:, 0, 3, -1] - 1.0)
    # the study's panels are the seeds' mean and the seeds themselves
    np.testing.assert_allclose(study.panels[3].data["ys"], trace[:, :, 2].mean(0), atol=1e-12)
    np.testing.assert_allclose(study.panels[3].data["each"], trace[:, :, 2], atol=1e-12)
    assert study.metadata["designs_tried"] and study.metadata["seeds"] == list(F.SEEDS)


@needs("instruction-tuning")
def test_instruction_tuning_shows_the_measured_numbers():
    anim, study = built("instruction-tuning")
    r = F.instruction_runs()
    acc = np.asarray(study.panels[3].data["ys"])
    mask = np.asarray(study.panels[1].data["values"])
    for row in anim.rows:
        assert "".join(row[:5]) in F.INSTRUCTIONS and len(row) == mask.shape[1] + 1
    assert "all %d guesses" % mask.shape[1] in anim.subtitle_for(1)
    assert "just the %d" % mask[0].sum() in anim.subtitle_for(1)
    reach = F.reached(r["trace"])
    assert "by update %d on average" % (round(reach[:, 0].mean() / 10) * 10) in anim.subtitle_for(2)
    if np.all(reach[:, 1] > F.TUNE_STEPS):
        assert "not within %d updates in any run" % F.TUNE_STEPS in anim.subtitle_for(2)
    assert M.pct(r["written"][:, 0].mean()) in anim.subtitle_for(3)
    # the counts under the sweep end at 21 and 8, and the last frame's accuracies are the study's
    end = shown(anim, phase_time(anim, 1, 0.95))
    assert "%d" % mask.shape[1] in end and "%d" % mask[0].sum() in end
    assert on_screen(anim, 3, 0.95, "answer-only  %s" % M.pct(acc[0][-1]))
    assert on_screen(anim, 3, 0.95, "all-token  %s" % M.pct(acc[1][-1]))
    assert on_screen(anim, 2, 0.95, "answer-only  %d to %d" % (reach[:, 0].min(),
                                                              reach[:, 0].max()))
    # the answers drawn at the last checkpoint are what that model wrote, character by character
    texts = shown(anim, phase_time(anim, 3, 0.95))
    for e in anim.HELD:
        for ch in realdata.decode(r["guess"][0, -1, e]):
            assert M.shown(ch) in texts


# ---------------------------------------------------------------------- 36
@needs("reward-model")
def test_preference_pairs_are_the_text():
    d = M.preferences()
    words = d["words"]
    train, test = F._words("train"), F._words("test")
    bigrams = {(a, b) for a, b in itertools.pairwise(train)}
    assert all((words[c], words[w]) in bigrams for c, w in d["train_pairs"])
    held = {(a, b) for a, b in itertools.pairwise(test)}
    assert all((words[c], words[w]) in held for c, w in d["test_pairs"][:500])
    assert np.all(d["train_rejected"] != d["train_pairs"][:, 1])
    assert np.all(d["test_rejected"] != d["test_pairs"][:, 1])
    # after 'the', both words of a pair are among the 48 candidates
    the = d["test_pairs"][:, 0] == 0
    assert words[0] == "the" and the.sum() == d["counts"][4] > 100
    assert d["test_pairs"][the, 1].max() < F.CANDIDATES > d["test_rejected"][the].max()
    assert d["losers"].shape == (len(F.SEEDS), len(d["winners"]))
    assert len(d["candidates"]) == F.CANDIDATES and d["candidates"] == words[:F.CANDIDATES]


@needs("reward-model")
def test_reward_model_learns_the_preferences():
    """The claim: given the words (not their spelling) the scorer ranks held-out pairs far
    above chance, and seeing the previous word helps further - in every seed."""
    _anim, study = built("reward-model")
    d = M.preferences()
    acc = d["accuracy"]                                   # kind (context, word, spelling), seed
    assert acc.shape == (3, len(F.SEEDS))
    assert acc[0].min() > 0.85
    assert acc[0].min() > acc[1].max() + 0.05
    assert acc[1].min() > acc[2].max() + 0.05
    assert d["test_loss"][0].max() < np.log(2) - 0.3
    np.testing.assert_allclose(study.panels[3].data["values"][:3], acc.mean(1), atol=1e-12)
    # the stored scores are the stored model's, and they rank the held-out pairs as reported
    table = d["u"] @ d["v"].T + d["b"]
    np.testing.assert_allclose(table[0, :F.CANDIDATES], d["rewards"][0], atol=1e-12)
    c, w = d["test_pairs"].T
    assert (table[c, w] > table[c, d["test_rejected"]]).mean() == pytest.approx(acc[0, 0])
    assert M.F.rank_correlation(d["rewards"][0], d["true"]) > 0.2


@needs("reward-model")
def test_reward_model_shows_the_measured_numbers():
    anim, study = built("reward-model")
    accuracy = np.asarray(study.panels[3].data["values"])
    loss = np.asarray(study.panels[1].data["ys"])[0]
    for v in accuracy[:3]:
        assert "%.1f%%" % (100 * v) in anim.subtitle_for(3)
        assert on_screen(anim, 3, 0.95, "%.1f%%" % (100 * v))
    assert "from %.3f to %.3f" % (loss[0], loss[-1]) in anim.subtitle_for(2)
    d = M.preferences()
    # the pairs drawn are the study's first pairs, by name
    texts = shown(anim, phase_time(anim, 0, 0.95))
    for (c, w), l in zip(d["train_pairs"][:5], d["train_rejected"][:5]):
        assert all(d["words"][i] in texts for i in (c, w, l))
    # the score built up on screen is the model's score for that pair
    c, w = d["train_pairs"][anim.scored[0]]
    assert on_screen(anim, 2, 0.95, "score %s" % M.signed(anim.table[c, w]))


@needs("reward-model")
def test_reward_model_reads_the_gap_off_the_studys_curve():
    anim, study = built("reward-model")
    x, y = study.panels[0].data["x"], np.asarray(study.panels[0].data["ys"])[0]
    np.testing.assert_allclose(M.sigmoid(x), y, atol=1e-12)
    pair = anim.scored[2]                              # one the scorer has the wrong way round
    gap = anim.gap(pair)
    assert gap < 0
    t = anim.spans[1][0] + (anim.spans[1][1] - anim.spans[1][0]) * (2.85 / 4)
    texts = shown(anim, t)
    assert "gap between the scores: %s" % M.signed(gap) in texts
    assert "%s sure of the preferred word" % M.pct(M.sigmoid(gap)) in texts


# ---------------------------------------------------------------------- 37
@needs("rlhf-ppo")
def test_rlhf_policies_are_the_studys():
    anim, study = built("rlhf-ppo")
    final = np.asarray(study.panels[3].data["ys"])
    kl = np.asarray(study.panels[1].data["ys"])
    for run, dist in enumerate(anim.dists):
        assert dist.shape == (181, 48)
        np.testing.assert_allclose(dist[0], final[3], atol=1e-12)          # starts at the reference
        np.testing.assert_allclose(dist[-1], final[run], atol=1e-12)
        drift = (dist[1:] * np.log(dist[1:] / final[3])).sum(1)
        np.testing.assert_allclose(drift, kl[run], atol=1e-9)
    # and the runs are PPO against the stored reward model
    d = M.preferences()
    again, _p = F.ppo_trace(d["rewards"][0], F.BETAS[2])
    np.testing.assert_allclose(again[2], kl[2], atol=1e-9)


@needs("rlhf-ppo")
def test_the_leash_trades_reward_for_drift():
    """The claim: a larger KL penalty leaves the policy nearer the reference and with less
    reward-model score - for all three reward models."""
    _anim, study = built("rlhf-ppo")
    score = np.asarray(study.panels[0].data["each"])[:, :, -1]         # seed, beta
    kl = np.asarray(study.panels[1].data["each"])[:, :, -1]
    assert score.shape == (len(F.SEEDS), len(F.BETAS))
    assert np.all(kl[:, 0] > kl[:, 1] + 0.05) and np.all(kl[:, 1] > kl[:, 2] + 0.5)
    assert np.all(score[:, 0] > score[:, 1] + 0.01) and np.all(score[:, 1] > score[:, 2] + 0.3)
    start = M.preferences()["ref"] @ M.preferences()["rewards"].T
    assert np.all(score[:, 2] > start + 0.5)                    # even leashed, it gains reward


@needs("rlhf-ppo")
def test_rlhf_shows_the_measured_numbers():
    anim, study = built("rlhf-ppo")
    score = np.asarray(study.panels[0].data["ys"])[:, -1]
    kl = np.asarray(study.panels[1].data["ys"])[:, -1]
    final = np.asarray(study.panels[3].data["ys"])
    assert "%.2f nats" % kl[0] in anim.subtitle_for(1)
    assert M.pct(final[0].max()) in anim.subtitle_for(1)
    assert "score %.2f for %.2f nats" % (score[0], kl[0]) in anim.subtitle_for(3)
    for run in range(3):
        assert on_screen(anim, 3, 0.95, "score %.2f  drift %.2f" % (score[run], kl[run]))
    king = anim.words.index("king")
    assert on_screen(anim, 1, 0.95, M.pct(final[0][king]))


# ---------------------------------------------------------------------- 38
@needs("dpo")
def test_dpo_run_is_the_studys():
    _anim, study = built("dpo")
    d = M.preferences()
    p = d["dpo_p"]
    np.testing.assert_allclose(p[0], d["ref"], atol=1e-12)
    np.testing.assert_allclose(p.sum(1), 1.0, atol=1e-12)
    drift = (p[1:] * np.log(p[1:] / d["ref"])).sum(1)
    np.testing.assert_allclose(drift, np.asarray(study.panels[3].data["ys"])[0], atol=1e-9)
    np.testing.assert_allclose(F.DPO_BETA * np.log(p[-1] / d["ref"]),
                               np.asarray(study.panels[1].data["ys"])[0], atol=1e-9)
    again, _p = F.dpo_trace(d["winners"], d["losers"][0])
    np.testing.assert_allclose(again[0], np.asarray(study.panels[0].data["ys"])[0], atol=1e-12)


@needs("dpo")
def test_dpo_lands_where_ppo_does():
    """The claim: with no reward model, DPO moves the policy as far from the reference as
    PPO does against one, and to nearly the same place - in every seed."""
    _anim, study = built("dpo")
    kl = np.asarray(study.panels[3].data["each"])[:, :, -1]            # seed, DPO | PPO
    held = np.asarray(study.panels[2].data["each"])[:, :, -1]
    between = np.asarray(study.panels[3].data["between"])
    assert np.all(kl > 0.5)                                    # both really moved
    assert np.all(between < 0.1) and np.all(between < kl.min(1) / 5)
    assert np.all(np.abs(held[:, 0] - held[:, 1]) < 0.3)
    start = float(M.preferences()["ref"] @ M.preferences()["true"])
    assert np.all(held > start + 0.5)                          # and both gained held-out score
    loss = np.asarray(study.panels[0].data["each"])
    assert np.all(loss[:, -1] < loss[:, 0] - 0.05)


@needs("dpo")
def test_dpo_shows_the_measured_numbers():
    anim, study = built("dpo")
    held = np.asarray(study.panels[2].data["ys"])
    between = np.asarray(study.panels[3].data["between"])
    assert "DPO ends at %.2f and PPO at %.2f" % (held[0][-1], held[1][-1]) in anim.subtitle_for(3)
    assert "%.2f nats apart" % between[0] in anim.subtitle_for(3)
    assert on_screen(anim, 3, 0.95, "DPO   %.2f" % held[0][-1])
    # the pair's loss on screen is the DPO loss of that pair under the final policy
    w, l = anim.winners[anim.pair], anim.losers[anim.pair]
    implicit = np.log(anim.p[-1] / anim.ref)
    loss = np.logaddexp(0, -anim.beta * (implicit[w] - implicit[l]))
    assert on_screen(anim, 1, 0.95, "loss on this pair  %.3f" % loss)
    assert on_screen(anim, 1, 0.95, "= " + M.signed(implicit[w]))


# ---------------------------------------------------------------------- 39
@needs("reward-hacking")
def test_weak_reward_models_are_the_ones_that_get_hacked():
    """The claim: pushing a policy along a proxy always raises the proxy score; whether the
    held-out score follows depends on how good the proxy is."""
    _anim, study = built("reward-hacking")
    last = study.panels[3].data
    held, proxy = np.asarray(last["each"]), np.asarray(last["proxy"])  # size, seed, update
    assert held.shape == (len(F.SIZES), len(F.PUSH_SEEDS), F.PUSH_STEPS + 1)
    assert np.all(proxy[:, :, -1] > proxy[:, :, 0] + 0.3)       # every proxy score rose
    assert np.all(np.diff(proxy, axis=2) > -1e-9)
    worse = (held[:, :, -1] < held[:, :, 0] - 0.05).sum(1)
    assert list(worse) == [int(v) for v in last["ends_worse"]]
    assert worse[0] + worse[1] >= 2 and worse[2] == 0           # only weak proxies backfire
    accuracy = np.asarray(last["accuracy"])[:, :, 0]
    assert accuracy[2].min() > accuracy[1].max() + 0.03 > accuracy[0].max() + 0.06
    end = held[:, :, -1]
    assert end[2].min() > end[0].mean() + 0.5 and end[2].min() > end[1].mean() + 0.3
    assert end[2].max() - end[2].min() < 0.05 < end[0].max() - end[0].min()
    # the run drawn is the one that fell furthest from its best, of all thirty
    fall = held.max(2) - held[:, :, -1]
    s = study.metadata["settings"]
    k, j = F.SIZES.index(s["drawn_size"]), F.PUSH_SEEDS.index(s["drawn_seed"])
    assert fall[k, j] == fall.max() > 0.5
    np.testing.assert_allclose(np.asarray(study.panels[1].data["ys"])[1], held[k, j], atol=1e-12)


@needs("reward-hacking")
def test_reward_hacking_shows_the_measured_numbers():
    anim, study = built("reward-hacking")
    start, best, end = (float(v) for v in study.panels[3].data["marks"])
    values = np.asarray(study.panels[1].data["ys"])
    assert anim.peak == int(values[1].argmax()) and values[1][anim.peak] == best
    assert 0 < anim.peak < len(values[1]) - 1 and best > start > end
    assert "from %.3f to its best, %.3f, at update %d" % (start, best, anim.peak) in \
        anim.subtitle_for(2)
    assert "falls to %.3f" % end in anim.subtitle_for(2)
    assert np.all(np.diff(values[0]) > 0)                  # "its proxy score only rises"
    texts = shown(anim, phase_time(anim, 2, 0.95))
    for v in (start, best, end):
        assert "held-out %s" % M.signed(v, 3) in texts
    # the bars are rows of the study's recorded policy
    dist = np.asarray(study.panels[2].data["values"])
    assert anim.every == 6 and len(dist) == 41
    favourite = int(np.argmax(np.asarray(study.panels[0].data["ys"])[0]))
    assert anim.words[favourite] in anim.subtitle_for(3)
    assert on_screen(anim, 1, 0.95, M.pct(dist[-1][favourite]))
    np.testing.assert_allclose(dist @ np.asarray(study.panels[0].data["ys"])[0],
                               values[0][::6], atol=1e-9)
    worse = [int(v) for v in study.panels[3].data["ends_worse"]]
    for k, n in enumerate(worse):  # noqa: B007
        assert on_screen(anim, 4, 0.95, "ends below its start: %d of %d" % (n, len(F.PUSH_SEEDS)))
    assert "%d of 10 on all" % worse[2] in anim.subtitle_for(4)
    assert "worst of thirty" in anim.subtitle_for(4)


# ---------------------------------------------------------------------- 40
@needs("prompt-and-prefix-tuning")
def test_peft_counts_are_exact():
    anim, study = built("prompt-and-prefix-tuning")
    sizes = [int(v) for v in study.panels[0].data["values"]]
    c, r = anim.config, anim.rank
    assert sum(anim.block_sizes) == sizes[3]
    assert r * c.width == sizes[0]
    assert 2 * r * c.heads * (c.width // c.heads) == sizes[1]
    assert c.width * r + r * c.vocab == sizes[2]
    texts = shown(anim, phase_time(anim, 1, 0.6))
    assert "%d + %d = %d numbers" % (c.width * r, r * c.vocab, sizes[2]) in texts


@needs("prompt-and-prefix-tuning")
def test_adapters_go_part_of_the_way_to_full_tuning():
    """The claim: each add-on, training under 1% of the numbers, takes the base a real part
    of the way to full tuning on held-out text, and full tuning stays ahead - every seed."""
    _anim, study = built("prompt-and-prefix-tuning")
    r = F.peft_runs()
    final, base = r["final"], r["base"]                        # seed, method
    assert final.shape == (len(F.SEEDS), 4)
    assert np.all(final[:, :3] < base[:, None] - 0.5)           # every add-on helps
    assert np.all(final[:, 3:] < final[:, :3] - 0.3)            # full tuning is ahead of each
    closed = F.gap_closed(r)
    assert np.all(closed[:, :3] > 0.4) and np.all(closed[:, :3] < 0.85)
    assert np.all(r["sizes"][:3] / r["sizes"][3] < 0.01)
    # the base it starts from is bad at this text and good at the text it was trained on
    c = F.config()
    plain = F.windows("test", 256, np.random.default_rng(0))
    from voxanim import tinylm as lm
    assert lm.loss(F.base_model(0), plain[:, :-1], plain[:, 1:], c) < base[0] - 1.0
    # LoRA starts as the base exactly (its B is zero), on the same held-out windows
    np.testing.assert_allclose(r["trace"][:, 2, 1, 0], base, atol=1e-9)
    np.testing.assert_allclose(study.panels[2].data["values"], final.mean(0), atol=1e-12)
    assert set(r["rate"]) <= set(F.RATES)
    # held-out windows are capitals of the test split, never of the training split
    sample = realdata.decode(r["sample"][0])
    assert sample == sample.upper() and sample.upper() in realdata.text("test").upper()


@needs("prompt-and-prefix-tuning")
def test_peft_shows_the_measured_numbers():
    anim, study = built("prompt-and-prefix-tuning")
    final = [float(v) for v in study.panels[2].data["values"]]
    traces = np.asarray(study.panels[1].data["ys"])
    closed = [float(v) for v in study.panels[2].data["closed"]]
    assert "soft prompt %.2f, prefix %.2f, LoRA %.2f, full tuning %.2f" % tuple(final) in \
        anim.subtitle_for(4)
    assert "%.2f untouched" % anim.base in anim.subtitle_for(4)
    texts = shown(anim, phase_time(anim, 4, 0.95))
    for v, share in zip(final, closed):
        assert "%.2f" % v in texts and "%.0f%% of the way" % (100 * share) in texts
    assert on_screen(anim, 3, 0.95, "%.3f" % traces[3][-1])


# ---------------------------------------------------------------------- 41
@needs("model-merging")
def test_merging_plane_holds_the_named_mixes():
    anim, study = built("model-merging")
    r = F.merging_runs()
    assert -1.0 < float(r["cosine"][0]) < 1.0
    for column, (name, a, b, _ink) in enumerate(anim.MIXES):
        i, j = list(F.MERGE_B).index(b), list(F.MERGE_A).index(a)
        np.testing.assert_allclose(r["plane"][:, i, j], r["scores"][0, :, column], atol=1e-9)
        assert name == study.panels[2].data["labels"][column] == F.MIXES[column][0]
    np.testing.assert_allclose(study.panels[2].data["values"], r["scores"][:, 0].mean(0))
    np.testing.assert_allclose(study.panels[3].data["values"], r["scores"][:, 1].mean(0))
    # the map's base cell is zero by construction, and its best cell is well below the base
    base = anim.rise[len(F.MERGE_A) - 1 - list(F.MERGE_A).index(0.0), list(F.MERGE_B).index(0.0)]
    assert base == 0.0 and anim.rise.min() < -0.5
    # the two tasks' windows are the same play: capitals of it, and it backwards
    a, b = (realdata.decode(row) for row in r["sample"][:, 0])
    train = realdata.text("train")
    assert a == a.upper() and a in train.upper() and b[::-1] in train


@needs("model-merging")
def test_task_vectors_add_and_subtract():
    """The claim, for every seed: each fine-tune learns its own task and harms the other;
    half of each vector gives both skills, less of each; subtracting a vector un-learns."""
    _anim, _study = built("model-merging")
    r = F.merging_runs()
    name = [m[0] for m in F.MIXES]
    A, B, plain = (dict(zip(name, r["scores"][:, text].T)) for text in range(3))
    assert np.all(A["tuned on A"] < A["base"] - 1.0) and np.all(B["tuned on B"] < B["base"] - 1.0)
    assert np.all(B["tuned on A"] > B["base"] + 0.3)            # A's vector harms B
    for T, own in ((A, "tuned on A"), (B, "tuned on B")):
        assert np.all(T["average"] < T["base"] - 0.5)           # the merge has both skills
        assert np.all(T["average"] > T[own] + 0.3)              # each less than its specialist
        assert np.all(T["sum"] > T["average"] + 0.3)            # the full sum overshoots
    assert np.all(B["minus B"] > B["base"] + 0.5)               # subtraction un-learns B
    assert np.all(B["A minus B"] > B["base"] + 0.5)
    assert np.all(plain["base"] < plain["tuned on A"] - 0.5)    # what the fine-tunes cost
    assert np.all(np.abs(r["cosine"]) < 0.3)
    assert np.all(r["stop"] > 0)


@needs("model-merging")
def test_merging_shows_the_measured_numbers():
    anim, study = built("model-merging")
    a = [float(v) for v in study.panels[2].data["values"]]
    b = [float(v) for v in study.panels[3].data["values"]]
    assert "lengths %.2f and %.2f" % tuple(anim.norms) in anim.subtitle_for(1)
    assert "base %.2f and %.2f" % (a[0], b[0]) in anim.subtitle_for(3)
    assert "the average %.2f and %.2f" % (a[3], b[3]) in anim.subtitle_for(3)
    texts = shown(anim, phase_time(anim, 3, 0.95))
    for v in a + b:
        assert "%.2f" % v in texts
    assert any("un-learns B (%.2f → %.2f)" % (b[0], b[6]) in t for t in texts)


# ---------------------------------------------------------------------- 42
@needs("knowledge-editing")
def test_edit_repeats_the_study():
    anim, study = built("knowledge-editing")
    e = M.edit_trace(anim.layer, float(anim.scales[-1]))
    response = np.asarray(study.panels[2].data["ys"])
    np.testing.assert_allclose(e["p_after"], response[:, -1], rtol=1e-5)
    np.testing.assert_allclose(e["p_before"], response[:, 0], rtol=1e-5)
    np.testing.assert_allclose(e["p_before"][1], study.panels[1].data["values"][0], rtol=1e-5)
    side = np.asarray(study.panels[3].data["ys"])[0]
    np.testing.assert_allclose([e["c_before"], e["c_after"]], [side[0], side[-1]], rtol=1e-5)
    # rank one: every entry drawn is its column entry times its row entry
    np.testing.assert_allclose(e["block"], np.outer(e["column"], e["push"]), rtol=1e-6,
                               atol=1e-12)
    assert float(e["moved"]) == pytest.approx(float(anim.scales[-1]))
    assert str(e["target"][0]) == study.metadata["settings"]["target"]


@needs("knowledge-editing")
def test_knowledge_editing_shows_the_measured_numbers():
    anim, study = built("knowledge-editing")
    patch = np.asarray(study.panels[0].data["values"])
    response = np.asarray(study.panels[2].data["ys"])
    assert anim.row == int(np.argmax(patch[1:12].max(1))) + 1
    assert patch[anim.row].argmax() == patch.shape[1] - 1          # the ringed cell
    assert "%s and %s" % (M.pct(response[0][-1]), M.pct(response[1][-1])) in anim.subtitle_for(3)
    assert M.pct(patch[anim.row, -1]) in anim.subtitle_for(1)
    texts = shown(anim, phase_time(anim, 3, 0.95))
    assert M.pct(response[0][-1]) in texts and M.pct(response[1][-1]) in texts
    assert "edit strength %d" % anim.scales[-1] in texts

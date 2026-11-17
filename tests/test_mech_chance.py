"""Animations 115-117 (voxanim.mech_chance) and their experiments (voxanim.study_chance):
the layout is clean, the loop closes, the numbers on screen are the study's, each
study's claim holds as a relative claim with a margin, and the recorded arrays are what
they say they are (a chain refitted, a training run and a sampled text repeated)."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import needs_gpt2, needs_mnist, needs_text

from voxanim import mech_chance as M
from voxanim import presets, qa
from voxanim import study_chance as S
from voxanim.next_catalog import build_study

MARKOV, SEEDS, TEXT = "markov-chains", "seeds-and-training-outcomes", "seeds-and-sampled-text"
PRESET = {MARKOV: MARKOV, SEEDS: SEEDS, TEXT: "gpt2-" + TEXT}
EACH = [pytest.param(MARKOV, marks=needs_text), pytest.param(SEEDS, marks=needs_mnist),
        pytest.param(TEXT, marks=[needs_gpt2, needs_text])]


def study(name):
    return build_study("study_chance", name)


def anim(name):
    return presets.get(PRESET[name]).build()


def subtitles(a):
    return [a.subtitle_for(k) for k in range(len(a.PHASES))]


def texts(a, t):
    return [x.s for x in a.scene(t).texts]


def at(a, phase, u):
    lo, hi = a.spans[phase]
    return lo + (hi - lo) * u


# ---------------------------------------------------------------------- all three
@pytest.mark.parametrize("name", EACH)
def test_layout_is_clean(name):
    assert qa.check(PRESET[name], samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", EACH)
def test_drawn_by_this_module_and_loop_closes(name):
    a = anim(name)
    assert type(a) is M.ANIMATIONS[name]
    assert 45.0 <= a.t_done <= 60.0

    def summary(t):
        sc = a.scene(t)
        return len(sc.lines), len(sc.quads), sorted((x.s, round(x.opacity, 3)) for x in sc.texts)
    assert summary(0.0) == summary(a.duration)
    for lo, hi in a.spans:                            # every phase draws things, not only text
        assert len(a.scene((lo + hi) / 2).quads) >= 2


@pytest.mark.parametrize("name", EACH)
def test_subtitles_fit_two_lines(name):
    from voxanim import layout

    for sub in subtitles(anim(name)):
        lines = layout.wrap(sub, layout.SUB_SIZE, 1 - 2 * layout.SAFE)
        assert len(lines) <= 2
        assert max(layout.text_width(ln, layout.SUB_SIZE) for ln in lines) <= 1 - 2 * layout.SAFE


@pytest.mark.parametrize("name", EACH)
def test_study_says_what_was_run(name):
    s = study(name)
    m = s.metadata
    assert len(s.panels) == 4 and len(m["seeds"]) >= 3
    assert m["designs_tried"] and m["limitations"] and "default_rng" in m["where_the_seed_enters"]


# ---------------------------------------------------------------------- no data needed
def test_draw_is_the_inverse_of_the_running_sum():
    p = np.array([0.2, 0.0, 0.5, 0.3, 0.0])
    assert [S.draw(p, u) for u in (0.0, 0.19, 0.2, 0.69, 0.7, 0.999999)] == [0, 0, 2, 2, 3, 3]
    assert S.draw(p * 7, 0.5) == 2                    # counts work as well as chances
    rng = np.random.default_rng(0)
    picks = np.bincount([S.draw(p, u) for u in rng.random(20000)], minlength=5) / 20000
    assert np.abs(picks - p).max() < 0.015


def test_entropy_and_distance():
    assert S.entropy([0.5, 0.5]) == 1.0 and S.entropy([1.0, 0.0]) == 0.0
    assert np.isclose(S.entropy(np.full(65, 1 / 65)), np.log2(65))
    assert S.tv([1, 0], [0, 1]) == 1.0 and S.tv([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_chain_arithmetic_on_a_chain_known_by_hand():
    P = np.array([[0.9, 0.1], [0.5, 0.5]])            # fixed vector (5/6, 1/6)
    pi = S.stationary(P)
    assert np.allclose(pi, [5 / 6, 1 / 6]) and np.allclose(pi @ P, pi)
    rate = 5 / 6 * S.entropy([0.9, 0.1]) + 1 / 6 * 1.0
    assert np.isclose(S.entropy_rate(P, pi), rate) and rate < S.entropy(pi)
    far, dist = S.forgetting(P, pi, [0, 1], 6)
    assert np.allclose(far[:, 1:] / far[:, :-1], 0.4)  # shrinks by the second eigenvalue
    assert np.allclose(dist[1, 1], P[1]) and np.allclose(dist.sum(-1), 1)
    path = S.walk(P, 0, np.array([0.95, 0.2, 0.6, 0.89]))
    assert path.tolist() == [0, 1, 0, 0, 0]
    seq = np.array([0, 1, 0, 1, 0, 1, 1])
    assert S.transitions(seq, 2).tolist() == [[0, 3], [2, 1]]
    model = S.fit_order(np.array([0, 1] * 20), 1, 2)
    assert S.order_bits(model, np.array([0, 1] * 5), 1, 2, 0.0, 1) == (0.0, 0.0)
    second = S.fit_order(np.array([0, 1] * 20), 2, 2)
    bits, unseen = S.order_bits(second, np.array([1, 1, 0]), 2, 2, 1.0, 2)
    assert unseen == 1.0 and np.isclose(bits, 1.0)    # a context never seen: a coin
    new, dead = S.generate(model, 1, 2, [0], np.array([0.3, 0.9, 0.1]))
    assert new.tolist() == [1, 0, 1] and dead == 0


def test_reshaping_a_distribution():
    p = np.array([0.5, 0.3, 0.15, 0.05])
    assert S.reshape(p, (0.0, 0, 0.0)).tolist() == [1, 0, 0, 0]
    assert np.allclose(S.reshape(p, (1.0, 0, 0.0)), p)
    cold, hot = S.reshape(p, (0.5, 0, 0.0)), S.reshape(p, (2.0, 0, 0.0))
    assert np.allclose(cold, p ** 2 / (p ** 2).sum()) and cold[0] > p[0] > hot[0]
    assert S.entropy(cold) < S.entropy(p) < S.entropy(hot)
    assert np.allclose(S.reshape(p, (1.0, 2, 0.0)), [0.625, 0.375, 0, 0])
    assert np.allclose(S.reshape(p, (1.0, 0, 0.9)), [0.5 / 0.95, 0.3 / 0.95, 0.15 / 0.95, 0])
    assert np.allclose(S.reshape(p, (1.0, 0, 0.4)), [1, 0, 0, 0])
    assert np.array_equal(S.uniforms(7), S.uniforms(7))
    assert not np.array_equal(S.uniforms(7), S.uniforms(8))


def test_agreement_and_resampling():
    tokens = np.array([[1, 2, 3], [1, 2, 4], [1, 5, 3], [1, 2, 3]])
    share, first = S.agreement(tokens)
    assert share.tolist() == [1.0, 0.5, 1 / 6] and first.tolist() == [2, 1, 3, 1, 2, 1]
    assert S.distinct(tokens) == 3
    rng = np.random.default_rng(0)
    low, high = np.array([0.1, 0.2, 0.3]), np.array([0.5, 0.6])
    assert S.wrong_winner(high, low, (1, 4), 500, rng).tolist() == [0.0, 0.0]
    assert S.wrong_winner(low, low, (1,), 500, rng)[0] == 0.5
    mixed = S.wrong_winner(np.array([0.0, 1.0, 1.0]), np.array([0.0, 0.0, 1.0]), (1, 8, 64),
                           4000, rng)
    assert np.isclose(mixed[0], (1 + 0.5 * 4) / 9) and mixed[0] > mixed[1] > mixed[2]


# ---------------------------------------------------------------------- 115
@needs_text
def test_markov_chain_is_counted_from_the_training_split():
    from voxanim import realdata

    d = S.markov_data()
    chars, train = S.alphabet(), realdata.text("train")
    assert str(d["alphabet"]) == chars and len(chars) == 65
    assert set(realdata.text("test")) <= set(chars)
    a, b = chars.index("q"), chars.index("u")
    assert d["counts"][a, b] == train.count("qu") and d["counts"].sum() == len(train) - 1
    assert np.allclose(d["P"].sum(1), 1) and np.allclose(d["P4"].sum(1), 1)
    kinds = d["classes"]
    assert [int(kinds[chars.index(c)]) for c in "eEtT \n,3"] == [0, 0, 1, 1, 2, 3, 3, 3]
    assert d["counts4"].sum() == len(train) - 1
    assert d["counts4"][2, 2] == sum(train.startswith("  ", i) for i in range(len(train)))
    for split in ("train", "validation", "test"):
        assert "".join(chars[i] for i in S.char_ids(split)[:200]) == realdata.text(split)[:200]


@needs_text
def test_markov_stationary_vector_is_exact_and_the_walk_finds_it():
    s, d = study(MARKOV), S.markov_data()
    for P, pi in ((d["P"], d["pi"]), (d["P4"], d["pi4"])):
        assert np.allclose(pi @ P, pi, atol=1e-12) and np.isclose(pi.sum(), 1) and pi.min() > 0
    # the fixed vector of a chain counted from a text is the text's character frequencies
    assert s.metadata["stationary_vs_frequency"] < 1e-5
    walk_tv = np.asarray(s.panels[1].data["walk_tv"])
    assert walk_tv.shape == (len(S.WALK_SEEDS), len(S.WALK_AT))
    assert (walk_tv[:, -1] < 0.01).all() and (walk_tv[:, 0] > 5 * walk_tv[:, -1]).all()
    us = np.random.default_rng(S.MARKOV_SEED).random(200)
    path = S.walk(d["P4"], d["classes"][S.char_ids("train")[0]], us)
    assert np.array_equal(path, d["walk_path"]) and np.array_equal(us, d["walk_us"])


@needs_text
def test_markov_chain_forgets_its_start_and_knows_more_than_single_characters():
    s = study(MARKOV)
    far = np.asarray(s.panels[1].data["ys"])
    assert (far[:, 0] > 0.8).all() and (far[:, -1] < 0.01).all()
    assert (np.diff(far, axis=1) < 1e-12).all()
    dist = np.asarray(s.panels[1].data["distributions"])
    chars = S.alphabet()
    assert dist[0, 1, chars.index("u")] > 0.99        # after q comes u
    uniform, single, rate = s.panels[2].data["values"]
    assert rate < single - 1.0 and single < uniform - 1.0
    m = s.metadata
    # the order-1 entropy rate is the order-1 chain's score on its own text
    assert abs(rate - m["fitted_text_bits"][1]) < 1e-3
    assert abs(single - m["fitted_text_bits"][0]) < 1e-3
    assert m["second_eigenvalue"]["characters"] < 1.0


@needs_text
def test_markov_order_helps_then_hurts_on_held_out_text():
    s = study(MARKOV)
    m = s.metadata
    held, fitted = (np.asarray(v) for v in s.panels[3].data["ys"])
    best = m["best_order"]
    assert 2 <= best < max(S.ORDERS)
    assert (np.diff(held[:best + 1]) < -0.001).all() and held[0] - held[3] > 1.0
    assert held[-1] > held[best] + 0.3                # too-rare contexts: it turns back up
    assert (np.diff(fitted) < -0.2).all()             # on its own text it only falls
    assert (fitted[1:] < held[1:]).all() and fitted[-1] < held[-1] - 1.0
    assert (np.diff(m["contexts_seen"]) > 0).all() and m["contexts_seen"][1] == 65
    assert (np.diff(m["unseen_context_share"]) >= 0).all()
    assert m["unseen_context_share"][-1] > 0.1 and m["dead_ends"] == 0


@needs_text
def test_markov_order_two_refitted_gives_the_recorded_numbers_and_text():
    m = study(MARKOV).metadata
    train, test = S.char_ids("train"), S.char_ids("test")
    model = S.fit_order(train, 2, 65)
    bits, unseen = S.order_bits(model, test, 2, 65, m["smoothing"][2], max(S.ORDERS))
    assert np.isclose(bits, m["held_out_bits"][2]) and np.isclose(
        unseen, m["unseen_context_share"][2])
    us = np.random.default_rng(S.MARKOV_SEED).random(S.TEXT_LENGTH)
    new, _ = S.generate(model, 2, 65, train[:2], us)
    assert "".join(S.alphabet()[c] for c in new) == m["generated"]["2"]


@needs_text
def test_markov_numbers_on_screen():
    a = anim(MARKOV)
    m = a.study.metadata
    d = S.markov_data()
    assert a.head_counts[-1].sum() == S.HEAD - 1
    assert np.array_equal(a.head_counts[-1], S.transitions(np.array(a.head_kinds), 4))
    subs = subtitles(a)
    assert f"{int(d['counts4'].sum()):,}" in subs[0]
    assert "%.2f" % m["entropy_bits"]["rate"] in subs[3]
    assert "%.2f" % m["entropy_bits"]["single"] in subs[3]
    assert "%.2f" % min(m["held_out_bits"]) in subs[4]
    assert "order %d" % m["best_order"] in subs[4]
    shown = texts(a, at(a, 0, 0.9))
    assert "%.2f" % d["P4"][1, 0] in shown and f"{int(d['counts4'][1, 0]):,}" in shown
    shown = texts(a, at(a, 1, 0.99))
    assert "share of time after 20,000 hops" in shown
    assert "%.1f%%" % (100 * d["walk_shares"][-1, 1]) in shown
    shown = texts(a, at(a, 3, 0.95))
    assert "%.2f: the entropy rate" % m["entropy_bits"]["rate"] in shown
    shown = texts(a, at(a, 4, 0.95))
    assert f"{m['contexts_seen'][5]:,}" in shown
    assert any("%.2f bits" % min(m["held_out_bits"]) in x for x in shown)
    assert any(x and x in m["generated"]["3"].replace("\n", "↵") for x in shown if len(x) > 30)


# ---------------------------------------------------------------------- 116
@needs_mnist
def test_seeds_data_is_disjoint_and_the_plan_is_what_it_says():
    from voxanim import mnist, realdata

    x, y, xt, _, meta = S.mnist_data()
    _, _, m = realdata.mnist_split(S.N_TRAIN, "train", 28, S.SPLIT_SEED)
    _, _, mt = realdata.mnist_split(10000, "test", 28, S.SPLIT_SEED)
    assert len(set(m["example_indices"])) == S.N_TRAIN and max(m["example_indices"]) < 50000
    assert sorted(mt["example_indices"]) == list(range(10000))
    xa, ya, xb, _ = mnist.load()
    assert np.array_equal(x, xa[m["example_indices"]])
    assert np.array_equal(y, ya[m["example_indices"]])
    assert np.array_equal(xt, xb[mt["example_indices"]])
    assert meta["train"]["official_partition"] == "train"
    assert meta["held_out"]["official_partition"] == "test"
    plan, groups = S.run_plan()
    assert len(plan) == 2 * S.N_RUNS + 6 * S.N_SOURCE == len(S.seeds_data()["acc"])
    lo, hi = groups["order/1"]
    assert {job[0] for job in plan[lo:hi]} == {S.WIDTH}
    assert {(job[1], job[3]) for job in plan[lo:hi]} == {(1, 1)}
    assert len({job[2] for job in plan[lo:hi]}) == S.N_SOURCE
    assert all(job[1] == job[2] == job[3] for job in plan[slice(*groups["all"])])


@needs_mnist
def test_seeds_three_generators_are_independent():
    a = S.initial_weights(S.generators(3, 0, 0)[0], S.WIDTH)
    b = S.initial_weights(S.generators(3, 5, 9)[0], S.WIDTH)
    c = S.initial_weights(S.generators(4, 0, 0)[0], S.WIDTH)
    assert S.weights_hash(a) == S.weights_hash(b) != S.weights_hash(c)
    assert np.array_equal(S.generators(0, 6, 0)[1].permutation(50),
                          S.generators(8, 6, 2)[1].permutation(50))
    assert not np.array_equal(S.generators(0, 6, 0)[1].permutation(50),
                              S.generators(0, 7, 0)[1].permutation(50))
    assert np.array_equal(S.generators(1, 2, 6)[2].random(9), S.generators(0, 0, 6)[2].random(9))
    assert not np.array_equal(S.generators(1, 2, 6)[2].random(9),
                              S.generators(1, 2, 6)[0].random(9))


@needs_mnist
def test_seeds_same_seed_twice_gives_identical_weights():
    d = S.seeds_data()
    plan, groups = S.run_plan()
    k = groups["init/0"][0] + 3                       # a run with three different seeds
    first, curve, wrong = S.train_run(*plan[k])
    again, _, _ = S.train_run(*plan[k])
    assert all(np.array_equal(first[key], again[key]) for key in first)
    assert S.weights_hash(first) == S.weights_hash(again)
    other, _, _ = S.train_run(S.WIDTH, plan[k][1], plan[k][2], plan[k][3] + 1)
    assert S.weights_hash(other) != S.weights_hash(first)
    assert np.array_equal(other["W0"].shape, first["W0"].shape)
    # the recorded run (perhaps trained on another machine: compared to within rounding)
    assert abs((1 - wrong.mean()) - d["acc"][k]) < 0.004
    assert np.abs(curve - d["curves"][k]).max() < 0.01
    m = study(SEEDS).metadata["reproducible"]
    assert m["same_seed_bit_identical"] and m["same_seed_max_difference"] == 0.0
    assert m["sha256"][0] == m["sha256"][1] != m["sha256"][2]


@needs_mnist
def test_seeds_spread_mistakes_and_sources():
    s = study(SEEDS)
    m = s.metadata
    acc, mis = m["accuracy"], m["mistakes"]
    final = np.asarray(s.panels[0].data["final"])
    assert len(final) == S.N_RUNS >= 20 and np.isclose(final.std(ddof=1), acc["std"])
    assert acc["max"] - acc["min"] > 0.004 and 0.9 < acc["mean"] < 0.97
    assert np.allclose(np.asarray(s.panels[0].data["ys"])[:, -1], final)
    wrong = np.asarray(s.panels[1].data["values"])
    assert wrong.shape == (S.N_RUNS, mis["wrong_in_some_run"])
    assert np.allclose(1 - final, wrong.sum(1) / 10000)
    times = np.asarray(s.panels[1].data["times"])
    assert np.array_equal(times, wrong.sum(0)) and (np.diff(times) <= 0).all()
    assert (times == S.N_RUNS).sum() == mis["wrong_in_every_run"] > 20
    # runs do not make the same mistakes: far more digits are missed by some run than by one
    assert mis["wrong_in_every_run"] < 0.6 * mis["mean_per_run"]
    assert mis["wrong_in_some_run"] > 1.5 * mis["mean_per_run"]
    assert 0.4 < mis["overlap_range"][0] <= mis["overlap"] <= mis["overlap_range"][1] < 0.9
    # two runs disagree on far more digits than their accuracies differ by
    assert mis["disagree"] > 5 * acc["std"]
    # each source alone moves accuracy; none alone exceeds what all three do by much
    for name in S.SOURCES:
        for spread in m["spread_by_source"][name]:
            assert 0.0005 < spread < 1.5 * m["spread_all"]
    total = sum(np.mean(m["spread_by_source"][name]) ** 2 for name in S.SOURCES)
    assert total > 1.3 * m["spread_all"] ** 2         # the variances do not add up
    per = np.asarray(s.panels[2].data["per_base"])
    assert np.isclose(np.asarray(s.panels[2].data["init"])[1].std(ddof=1), per[0, 1])


@needs_mnist
def test_seeds_one_seed_each_often_names_the_wrong_winner():
    s = study(SEEDS)
    c = s.metadata["comparison"]
    share = np.asarray(c["wrong_winner"])
    a, b = np.asarray(s.panels[3].data["width_a"]), np.asarray(s.panels[3].data["width_b"])
    assert c["gap"] > 3 * c["gap_standard_error"]     # the difference is real
    assert c["gap"] < 2 * max(c["std"])               # and small beside the seed spread
    assert share[0] > 0.1 and share[-1] < 0.01 and (np.diff(share) < 0).all()
    better, worse = (b, a) if b.mean() > a.mean() else (a, b)
    exact = (worse[:, None] > better[None]).mean() + 0.5 * (worse[:, None] == better[None]).mean()
    assert np.isclose(share[0], exact)
    n = c["seeds_for_under_5_percent"]
    assert n is not None and 2 <= n <= 16 and share[c["n"].index(n)] < 0.05
    again = S.wrong_winner(better, worse, S.COMPARE_N, S.COMPARE_DRAWS,
                           np.random.default_rng(S.COMPARE_SEED))
    assert np.allclose(again, share)


@needs_mnist
def test_seeds_numbers_on_screen():
    a = anim(SEEDS)
    m = a.study.metadata
    subs = subtitles(a)
    assert "%.2f%%" % (100 * m["accuracy"]["min"]) in subs[1]
    assert "%.2f points" % (100 * m["accuracy"]["std"]) in subs[1]
    assert "share %d%%" % round(100 * m["mistakes"]["overlap"]) in subs[2]
    assert "%.2f. The spreads" % (100 * m["spread_all"]) in subs[3]
    assert "ahead %d%%" % round(100 * m["comparison"]["wrong_winner"][0]) in subs[4]
    assert a.weights == 784 * 64 + 64 + 64 * 10 + 10
    shown = texts(a, at(a, 0, 0.3))
    assert "%.2f%%" % (100 * a.final[0]) in shown and "generator [0, 2]" in shown
    assert "%.2f%%" % (100 * a.final[1]) in texts(a, at(a, 0, 0.66))
    shown = texts(a, at(a, 3, 0.97))
    for k, name in enumerate(S.SOURCES):
        assert "%.2f and %.2f" % tuple(100 * v for v in m["spread_by_source"][name]) in shown
        assert np.allclose(a.spread[k], m["spread_by_source"][name])
    shown = texts(a, at(a, 4, 0.97))
    assert "really %.2f points apart" % (100 * m["comparison"]["gap"]) in shown
    d = S.seeds_data()
    assert np.array_equal(d["init_0"], S.initial_weights(S.generators(0, 0, 0)[0],
                                                         S.WIDTH)["W0"][406:414, :8])
    assert d["mask_0"].shape == (8, 8) and d["order_1"].shape == (6, 28, 28)


# ---------------------------------------------------------------------- 117
@needs_gpt2
@needs_text
def test_sampling_prompts_are_held_out_text():
    from voxanim import realdata

    d = S.sampling_data()
    m = study(TEXT).metadata
    assert d["tokens"].shape == (S.PROMPTS, len(S.SETTINGS), len(S.SAMPLE_SEEDS), S.NEW_TOKENS)
    test, train = realdata.text("test"), realdata.text("train")
    for prompt in m["prompts"]:
        assert prompt in test and prompt not in train
    assert m["gpt2_passes"] < m["passes_without_sharing"]
    assert np.array_equal(d["uniforms"], [S.uniforms(s) for s in S.SAMPLE_SEEDS])
    q = d["first_q"]
    assert np.allclose(q[3], d["first_p"]) and q[0, 0] == 1.0 and (np.diff(q, axis=1) <= 0).all()
    assert d["first_kept"][5] == 40 == (q[5] > 0).sum() and np.isclose(q[5].sum(), 1)
    assert d["first_kept"][0] == 1 and d["first_kept"][3] == 50257
    assert d["first_kept"][6] < d["first_kept"][3]
    # the first token of each seed is the one its first number picks from the recorded bar
    for a in (1, 3, 5):
        for b in range(6):
            if d["uniforms"][b, 0] < q[a].sum() - 1e-9:
                k = int(np.searchsorted(np.cumsum(q[a]), d["uniforms"][b, 0], side="right"))
                assert d["first_order"][k] == d["tokens"][S.DRAWN_PROMPT, a, b, 0]


@needs_gpt2
@needs_text
def test_sampling_the_same_seed_gives_the_same_text():
    from voxanim import study_decoding as D

    d = S.sampling_data()
    prompt = D.heldout_prompts(S.PROMPTS, S.PROMPT_TOKENS)[S.DRAWN_PROMPT]
    assert np.array_equal(prompt, d["prompts"][S.DRAWN_PROMPT])
    for a, b in ((3, 3), (6, 1)):                     # plain, then nucleus sampling
        tokens, logp = S.sample_plain(prompt, S.SAMPLE_SEEDS[b], S.SETTINGS[a])
        assert np.array_equal(tokens, d["tokens"][S.DRAWN_PROMPT, a, b])
        assert np.allclose(logp, d["logp"][S.DRAWN_PROMPT, a, b], atol=1e-3)
    m = study(TEXT).metadata["reproducible"]
    assert m["same_seed_same_text"] and m["plain_sampler_equals_shared_passes"]
    assert not m["other_seed_same_text"]


@needs_gpt2
@needs_text
def test_sampling_temperature_sets_how_soon_seeds_part():
    s = study(TEXT)
    m = s.metadata
    d = S.sampling_data()
    tokens = d["tokens"]
    assert (tokens[:, 0] == tokens[:, 0, :1]).all()   # temperature 0: every seed, same text
    assert m["distinct"][0] == 1.0 and m["agreement_at_the_end"][0] == 1.0
    one, first = np.asarray(m["agreement_after_one_token"]), np.asarray(m["first_difference"])
    assert (np.diff(one[:S.N_TEMPS]) < -0.02).all() and one[1] > one[4] + 0.15
    assert (np.diff(first[:S.N_TEMPS]) < 0).all()
    agree = np.asarray(s.panels[0].data["ys"])
    assert (np.diff(agree, axis=1) <= 1e-12).all()    # texts that part do not meet again
    kinds = np.asarray(m["distinct"])
    assert kinds[1] < kinds[3] - 2 and kinds[3] > 19
    logp, bits = np.asarray(m["log_probability"]), np.asarray(m["sampled_entropy_bits"])
    assert (np.diff(logp[:S.N_TEMPS]) < -0.2).all() and (np.diff(bits[:S.N_TEMPS]) > 0.5).all()
    per_prompt = np.asarray(s.panels[2].data["per_prompt"])
    assert (per_prompt[:, 1] > per_prompt[:, 4] + 2).all()      # on every prompt


@needs_gpt2
@needs_text
def test_sampling_cutting_the_tail_keeps_variety_and_drops_unlikely_tokens():
    s = study(TEXT)
    m = s.metadata
    kinds, rare = np.asarray(m["distinct"]), np.asarray(m["texts_with_an_unlikely_token"])
    logp = np.asarray(m["log_probability"])
    plain, top_k, top_p, cool = 3, 5, 6, 2
    assert rare[top_k] < rare[plain] - 0.5 and rare[top_k] < rare[cool] - 0.2
    assert kinds[top_k] > kinds[cool] and kinds[top_k] > kinds[plain] - 1
    assert logp[top_k] > logp[plain] + 0.5 and logp[top_p] > logp[plain] + 0.3
    assert rare[top_p] < rare[plain]
    per = np.asarray(s.panels[3].data["per_prompt"])
    assert (per[:, top_k] < per[:, plain]).all()                # on every prompt
    d = S.sampling_data()
    assert (d["logp"][:, top_k] > np.log(1e-5)).all()
    assert (d["bits"][:, 0] == 0).all() and (d["bits"][:, top_k] <= np.log2(40) + 1e-9).all()


@needs_gpt2
@needs_text
def test_sampling_numbers_on_screen():
    a = anim(TEXT)
    m = a.study.metadata
    d = S.sampling_data()
    subs = subtitles(a)
    assert "%d%% → %d%%" % (round(100 * m["agreement_after_one_token"][1]),
                            round(100 * m["agreement_after_one_token"][4])) in subs[3]
    assert "gives %.1f distinct" % m["distinct"][5] in subs[4]
    assert "only %d%%" % round(100 * m["texts_with_an_unlikely_token"][5]) in subs[4]
    shown = texts(a, at(a, 0, 0.95))
    for b in range(6):
        assert "u = %.3f" % S.uniforms(b)[0] in shown
        assert str(d["words"][3, b, 0])[:9] in shown
    six = d["tokens"][S.DRAWN_PROMPT, M.TREE_SETTING, :6]
    for i in range(6):
        for j in range(6):
            same = np.cumprod(six[i] == six[j]).sum()
            assert a.first6[i, j] == (0 if i == j else same + 1)
    assert "%d of the 15" % (a.first6[np.triu_indices(6, 1)] == 1).sum() in subs[1]
    shown = texts(a, at(a, 2, 0.05))                  # temperature 0: one text, five times
    assert shown.count(str(d["texts"][0, 0]).replace("\n", "↵").strip()[:74]) == 5
    shown = texts(a, at(a, 4, 0.9))
    assert "%.1f" % m["distinct"][5] in shown
    assert "%.0f%%" % (100 * m["texts_with_an_unlikely_token"][3]) in shown
    assert "−%.2f" % -m["log_probability"][6] in shown

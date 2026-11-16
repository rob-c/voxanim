"""The reinforcement-learning animations (91-96): layout, the numbers they
show against their studies, the replays they draw against the studies' own
curves, and what the redesigned glimpse-game studies (92, 93, 94, 96) support."""

import itertools

import numpy as np
import pytest
from conftest import available, needs_mnist

from voxanim import mech_rl as M
from voxanim import presets, qa
from voxanim import study_rl as G
from voxanim.next_catalog import build_study

NAMES = sorted(M.ANIMATIONS)


def study(name):
    return build_study("next_systems", name)


def built(name):
    if not available(name):
        pytest.skip("needs data that is not cached")
    return presets.get(name).build()


@pytest.mark.parametrize("name", NAMES)
def test_layout_is_clean(name):
    if not available(name):
        pytest.skip("needs data that is not cached")
    assert qa.check(name, samples=24, verbose=False) == 0


@pytest.mark.parametrize("name", NAMES)
def test_built_by_this_module_and_loop_closes(name):
    anim = built(name)
    assert type(anim) is M.ANIMATIONS[name]
    assert 45.0 <= anim.t_done <= 60.0
    first, last = anim.scene(0.0), anim.scene(anim.duration)
    assert sorted((t.s, round(t.opacity, 3)) for t in first.texts) == sorted(
        (t.s, round(t.opacity, 3)) for t in last.texts)
    assert len(first.quads) == len(last.quads) and len(first.lines) == len(last.lines)
    for k in range(len(anim.PHASES)):
        assert anim.subtitle_for(k)


# ---------------------------------------------------------------------- 91
@needs_mnist
def test_bandit_replay_is_the_study():
    s, d = study("multi-armed-bandits"), M.bandit_replay()
    assert np.allclose(d["means"], s.panels[0].data["values"])
    assert np.allclose(d["regret"].mean(1), s.panels[1].data["ys"])
    counts = np.array([[np.bincount(run, minlength=10) for run in method]
                       for method in d["arm"]]).mean(1)
    assert np.allclose(counts, s.panels[2].data["values"])
    assert d["pixels"].sum(1).tolist() == list(M.PIXEL_COUNTS)
    # Thompson's recorded guesses do pick the arm it pulled
    assert np.array_equal(d["draws"].argmax(1), d["arm"][2, 0])
    # and the digits shown in the first phase are the ones those pulls drew
    assert np.array_equal(d["table"][d["arm"][2, 0], d["picked"]], d["outcome"][2, 0])


@needs_mnist
def test_bandit_numbers_shown():
    s, anim = study("multi-armed-bandits"), built("multi-armed-bandits")
    final = np.asarray(s.panels[1].data["ys"])[:, -1]
    text = anim.subtitle_for(4)
    assert "Thompson %.0f, ε-greedy %.0f, UCB %.0f" % (final[2], final[0], final[1]) in text
    means = np.asarray(s.panels[0].data["values"])
    assert "%d%% to %d%%" % (round(100 * means.min()), round(100 * means.max())) in \
        anim.subtitle_for(0)
    # the state the replay draws: pulls before pull t add up to t
    assert np.allclose(anim.n.sum(2), np.arange(1001)[None])
    assert anim._clock(1.0, 0)[0] == 1000 and anim._clock(0.0, 1)[0] == 0


# ---------------------------------------------------------------------- the glimpse game
@needs_mnist
def test_glimpse_data_is_real_and_held_out():
    x, y, xt, yt, meta = G.digits()
    assert x.shape == (4096, 196) and xt.shape == (1000, 196)
    assert meta["train"]["official_partition"] == "train"
    assert meta["test"]["official_partition"] == "test"          # disjoint by construction
    assert meta["train"]["dataset"] == "MNIST" and x.min() >= 0 and x.max() <= 1
    assert set(np.unique(y)) == set(range(10)) == set(np.unique(yt))
    # the four quadrants tile the image, and a covered quadrant shows nothing of the digit
    assert np.array_equal(G.QUAD.sum(0), np.ones(196))
    mask = np.array([[1.0, 0, 0, 1]] * 5)
    obs = G.observe(x[:5], mask, 0.5)
    assert obs.shape == (5, G.IN) and np.array_equal(obs[:, 196:200], mask)
    assert not obs[:, :196][:, (G.QUAD[1] + G.QUAD[2]) > 0].any()
    assert np.array_equal(obs[:, :196][:, G.QUAD[0] > 0], x[:5][:, G.QUAD[0] > 0])


@needs_mnist
def test_runs_are_seeded():
    a, b = G.pg_run("critic", 1, updates=40, every=20), G.pg_run("critic", 1, updates=40, every=20)
    assert np.array_equal(a["reward"], b["reward"]) and np.array_equal(a["accuracy"], b["accuracy"])
    a, b = (G.q_run(1, shaping="potential", iterations=200) for _ in range(2))
    assert np.array_equal(a["score"], b["score"]) and np.array_equal(a["loss"], b["loss"])


@needs_mnist
@pytest.mark.parametrize("kind", ["baseline", "critic"])
def test_demo_episodes_are_the_game(kind):
    ep = M.demo(kind)
    act = ep["act"]
    assert ((act[:2] >= 0) & (act[:2] < 4)).all() and ((act[2] >= 4) & (act[2] < 14)).all()
    assert np.allclose(ep["probs"].sum(2), 1.0)
    assert not ep["probs"][:2, :, 4:].any() and not ep["probs"][2, :, :4].any()
    n = np.arange(act.shape[1])
    assert not ep["mask"][0].any() and ep["mask"][1][n, act[0]].all()
    assert np.array_equal(ep["mask"][1].sum(1), np.ones(len(n)))
    assert ep["mask"][2][n, act[1]].all()
    # rewards: 2 points for the right name, 1 for a wrong one
    assert np.array_equal(ep["reward"], 1.0 + (act[2] - 4 == ep["label"]))
    # the three multipliers: the reward, that minus the expectation, and the one-step
    # changes, which add up over an episode to the reward minus the first expectation
    assert np.allclose(ep["return"], ep["reward"][None])
    assert np.allclose(ep["baseline"], ep["reward"][None] - ep["value"])
    assert np.allclose(ep["critic"].sum(0), ep["reward"] - ep["value"][0])


# ---------------------------------------------------------------------- 92
@needs_mnist
def test_a_baseline_cuts_gradient_noise_and_learns_faster():
    s, d = study("policy-gradient"), G.pg_runs()
    assert np.allclose(s.panels[0].data["ys"], d["accuracy"][:2].mean(1))
    assert np.allclose(s.panels[2].data["ys"], d["noise_b"].mean(0)[:, :2].T)
    final = np.asarray(s.panels[3].data["values"]).reshape(2, G.SEEDS)
    assert np.array_equal(final, d["accuracy"][:2, :, -1])
    # the claim: with the baseline, held-out accuracy is higher (mean of five seeds, with
    # margin), and each batch's gradient is several times nearer the true gradient
    assert final[1].mean() > final[0].mean() + 0.10
    assert final[1].min() > final[0].mean()
    ratio = d["noise_b"][:, :, 0] / d["noise_b"][:, :, 1]              # (seed, probe)
    assert ratio.mean(1).min() > 3.0
    assert np.allclose(ratio.mean(1), s.metadata["noise_ratio_per_seed"])
    # what the noise is measured against is small beside it: the estimate is mostly noise
    assert (d["signal_b"] < d["noise_b"][:, :, 1]).all()
    assert s.metadata["seeds"] == 5 and "designs_tried" in s.metadata


@needs_mnist
def test_policy_gradient_numbers_shown():
    s, anim = study("policy-gradient"), built("policy-gradient")
    final = np.asarray(s.panels[3].data["values"]).reshape(2, -1)
    noise = np.asarray(s.panels[2].data["ys"]).mean(1)
    assert "%s without the baseline, %s with it" % (M._spread(final[0]), M._spread(
        final[1])) in anim.subtitle_for(4)
    assert "lies %.2f from" % noise[0] in anim.subtitle_for(3)
    assert "and %.2f with it, %.1f times closer" % (noise[1], noise[0] / noise[1]) in \
        anim.subtitle_for(3)
    # the two episodes drawn: one wrong, one right, and what the baseline does to their push
    ep = anim.ep
    wrong, right = anim.shown
    assert ep["reward"][wrong] == 1 and ep["reward"][right] == 2
    assert (ep["return"][:, wrong] > 0).all()                 # pushed up although wrong
    assert ep["baseline"][2, wrong] < 0 < ep["baseline"][2, right]
    assert np.allclose(M.rolling([1, 2, 3, 4], 2), [1, 1.5, 2.5, 3.5])
    assert M._spread([0.5, 0.6, 0.7]) == "60% (50 to 70)"


# ---------------------------------------------------------------------- 93
@needs_mnist
def test_the_critic_lowers_the_noise_of_the_actors_update():
    s, d = study("actor-critic"), G.pg_runs()
    assert np.allclose(s.panels[0].data["ys"], d["accuracy"].mean(1))
    assert np.allclose(s.panels[2].data["ys"], d["noise_c"].mean(0).T)
    noise = d["noise_c"].mean(1)                              # (seed, estimator)
    # against the raw return (REINFORCE), on the same policies and episodes: every seed
    assert (noise[:, 2] < 0.5 * noise[:, 0]).all()
    # and below the return minus a baseline, on the mean of the seeds
    assert noise[:, 2].mean() < 0.95 * noise[:, 1].mean()
    # what it costs: the one-step estimate's mean is off the true gradient by far less
    # than a single batch's noise
    assert (d["bias_c"].mean(1)[:, 2] < 0.1 * noise[:, 2]).all()
    final = d["accuracy"][:, :, -1]
    assert np.array_equal(np.asarray(s.panels[3].data["values"]).reshape(3, -1), final)
    assert final[2].mean() > final[0].mean() + 0.10          # learns more than REINFORCE
    assert final[2].mean() > final[1].mean() - 0.03          # and no less than the baseline


@needs_mnist
def test_actor_critic_numbers_shown():
    s, anim = study("actor-critic"), built("actor-critic")
    final = np.asarray(s.panels[3].data["values"]).reshape(3, -1)
    noise = np.asarray(s.panels[2].data["ys"]).mean(1)
    assert "actor-critic %s" % M._spread(final[2]) in anim.subtitle_for(3)
    assert "the raw reward %s" % M._spread(final[0]) in anim.subtitle_for(3)
    assert "%.2f using the critic's change" % noise[2] in anim.subtitle_for(2)
    ep, j = anim.ep, anim.shown[1]
    # the push drawn for the second look: the critic's expectation after it minus before
    assert np.isclose(ep["critic"][1, j], ep["value"][2, j] - ep["value"][1, j])
    assert np.isclose(ep["critic"][2, j], ep["reward"][j] - ep["value"][2, j])
    assert "from %.2f to %.2f: push %+.2f" % (ep["value"][1, j], ep["value"][2, j],
                                             ep["critic"][1, j]) in anim.subtitle_for(1)


# ---------------------------------------------------------------------- 94
@needs_mnist
def test_replay_and_target_network_under_correlated_experience():
    s, d = study("dqn-replay-buffer"), G.dqn_runs()
    assert np.allclose(s.panels[0].data["ys"], d["right"].mean(1))
    late = np.asarray(s.panels[1].data["values"]).reshape(5, G.SEEDS)
    assert np.allclose(late, G.last_pass(d["right"]).mean(2))
    neither, replay, target, both, shuffled = late
    # digits in runs, newest steps only: the learner is at chance, with or without a target
    # network; the same learner on shuffled digits is fine, so the correlation is the cause
    assert neither.max() < 0.2 and target.max() < 0.2 and shuffled.min() > 0.5
    # replay repairs most of it, and the target network adds to replay
    assert replay.mean() > neither.mean() + 0.25
    assert both.mean() > replay.mean() + 0.05
    assert G.last_pass(d["right"])[3].min(1).mean() > G.last_pass(d["right"])[1].min(1).mean() + 0.1
    # without replay the network also believes it is doing well
    excess = G.last_pass(d["value"] - d["score"]).mean((1, 2))
    assert excess[0] > 0.5 and excess[3] < 0.25
    # the batches: one class without replay, many with it (first seed, after the start)
    classes = (d["batch_classes"][:, 0, 5:] > 0).sum(2)
    assert classes[0].mean() < 2.0 and classes[3].mean() > 6.0
    assert d["batch_classes"].sum(3).max() == G.Q["batch"]
    # the matrix drawn is the first seed's share right per class, neither then both
    assert np.allclose(s.panels[2].data["values"][:10], d["by_class"][0, 0].T)


@needs_mnist
def test_dqn_numbers_shown():
    s, anim = study("dqn-replay-buffer"), built("dqn-replay-buffer")
    late = np.asarray(s.panels[1].data["values"]).reshape(5, -1)
    text = anim.subtitle_for(3)
    assert "replay only %s" % M._spread(late[1]) in text and "both %s" % M._spread(late[3]) in text
    assert "from %d%% to %d%%" % (round(100 * late[1].mean()), round(100 * late[3].mean())) in \
        anim.subtitle_for(2)
    # the step drawn: a look, whose target is 0.9 x the best value at the next step
    ep, j = anim.ep, anim.shown
    assert ep["right"][j] and ep["act"][1, j] < 4 and ep["act"][1, j] == ep["q"][1, j].argmax()
    assert ep["mask"][2, j].sum() == 2


# ---------------------------------------------------------------------- 96
def test_shaping_arithmetic_is_exact():
    (_, new), (_, loop) = M.RewardShaping.PLANS
    game_a, potential_a, bonus_a = M.extras(new)
    game_b, potential_b, bonus_b = M.extras(loop)
    assert np.isclose(game_a, 0.9 ** 3) and np.isclose(game_b, 0.9 ** 5)
    # potential-based shaping adds nothing to any whole episode; the bonus pays the loop more
    assert abs(potential_a) < 1e-12 and abs(potential_b) < 1e-12
    assert game_a > game_b and game_a + bonus_a < game_b + bonus_b
    assert np.isclose(bonus_b, 0.25 * sum(0.9 ** t for t in range(5)))


@needs_mnist
def test_potential_shaping_keeps_the_policy_and_a_bonus_is_farmed():
    s, d = study("reward-shaping"), G.shaping_runs()
    assert np.allclose(s.panels[0].data["ys"], d["score"].mean(1))
    score = np.asarray(s.panels[2].data["values"]).reshape(3, G.SEEDS)
    assert np.allclose(score, G.last_pass(d["score"]).mean(2))
    steps = G.last_pass(d["steps"]).mean(2)
    repeats = G.last_pass(d["repeats"]).mean(2)
    # potential-based shaping: the same score on the game itself, the same length of episode
    assert abs(score[1].mean() - score[0].mean()) < 0.03
    assert abs(steps[1].mean() - steps[0].mean()) < 0.5
    # the bonus per look: every seed scores lower on the game, uses nearly all six steps
    # and looks again at uncovered quadrants
    assert score[2].max() < score[0].min() - 0.10
    assert steps[2].min() > steps[0].max() + 1.5 and repeats[2].min() > 1.0 > 2 * repeats[0].max()
    # and the bonus learner's own estimate says it is doing better than the others
    value = G.last_pass(d["value"]).mean((1, 2))
    assert value[2] > value[0] + 0.5
    assert np.allclose(s.metadata["steps_to_reach"][1], [G._first(run, s.metadata["reach_level"],
                       np.asarray(s.panels[0].data["x"])) for run in d["score"][1]])


@needs_mnist
def test_reward_shaping_numbers_shown():
    s, anim = study("reward-shaping"), built("reward-shaping")
    score = np.asarray(s.panels[2].data["values"]).reshape(3, -1).mean(1)
    text = anim.subtitle_for(3)
    assert "%.2f with the bonus per look, %.2f on the game's reward, %.2f with" % (
        score[2], score[0], score[1]) in text
    # it says whether shaping was faster, as measured
    reach = np.asarray(s.metadata["steps_to_reach"]).mean(1)
    assert ("was faster" in text) == (reach[1] < 0.9 * reach[0])
    assert "%.1f steps and %.1f repeated looks" % (anim.late_steps[2], anim.late_repeats[2]) in \
        anim.subtitle_for(2)
    # the digit drawn: the bonus learner looks again at a quadrant it had uncovered
    j = anim.shown
    assert anim.eps[2]["repeats"][j] >= 1 and anim.eps[0]["right"][j]
    assert anim.eps[2]["steps"][j] > anim.eps[0]["steps"][j]


# ---------------------------------------------------------------------- 95
def test_mcts_replay_is_the_study():
    s, d = study("mcts-tic-tac-toe"), M.mcts_replay()
    assert np.array_equal(d["counts"], s.panels[1].data["ys"])
    assert np.allclose(d["values"], s.panels[2].data["ys"])
    assert d["moves"] == [int(m) for m in np.flatnonzero(np.array(M.START) == 0)]
    for n, sim in enumerate(d["sims"]):
        # each path starts at the root and each step adds one mark; playouts end the game
        boards = sim["path"] + sim["rollout"]
        assert boards[0] == M.START
        for a, b in itertools.pairwise(boards):
            assert int(np.sum(np.array(a) != np.array(b))) == 1
        winner = M.S.tictactoe_winner(boards[-1])
        assert winner != 0 and sim["reward"] == (0.0 if winner == 2 else float(winner))
        if d["ucb"][n] is not None and sim["path"][1] in d["kids"]:
            totals = [v + b for v, b in d["ucb"][n]]
            assert d["kids"].index(sim["path"][1]) == int(np.argmax(totals))


def test_mcts_numbers_shown():
    s, anim = study("mcts-tic-tac-toe"), built("mcts-tic-tac-toe")
    final = np.asarray(s.panels[3].data["values"])
    text = anim.subtitle_for(3)
    assert "has %d visits" % final.max() in text and "after %d simulations" % 501 in text
    assert M.MOVE_NAMES[anim.moves[int(final.argmax())]] == "top right"
    assert M.S.tictactoe_winner(anim.data["kids"][int(final.argmax())]) == 1

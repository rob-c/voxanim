"""Q-learning in the gridworld (:mod:`voxanim.gridworld`) and its animation:
the rules, the update, convergence to the shortest path, the subtitles' numbers."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from voxanim import gridworld as G


def test_world_rules():
    w = G.World()
    assert (w.h, w.w) == (6, 8)
    assert w.start == (5, 0) and w.goal == (0, 7) and w.penalty == (2, 5)
    # walls and edges hold the agent in place, for no reward
    assert w.step((5, 0), 2) == ((5, 0), 0.0)            # down, off the grid
    assert w.step((5, 1), 1) == ((5, 1), 0.0)            # right, into the wall at (5, 2)
    assert w.step((5, 0), 0) == ((4, 0), 0.0)            # up
    assert w.step((1, 7), 0) == ((0, 7), G.GOAL_REWARD)
    assert w.step((2, 4), 1) == ((2, 5), G.PENALTY_REWARD)
    path = w.shortest_path()
    assert path[0] == w.start and path[-1] == w.goal and len(path) - 1 == 14
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in pairwise(path))


def test_every_recorded_update_is_the_q_learning_rule():
    """Replaying the recorded steps with the rule reproduces every stored Q."""
    run = G.run()
    Q = np.zeros_like(run.Q[0])
    for k in range(run.episodes):
        for st in run.episode(k):
            assert st.q_before == Q[st.s][st.a]
            assert st.max_next == (0.0 if run.world.terminal(st.s2) else Q[st.s2].max())
            Q[st.s][st.a] += G.ALPHA * (st.r + G.GAMMA * st.max_next - st.q_before)
            assert abs(Q[st.s][st.a] - st.q_after) < 1e-12
        assert np.array_equal(Q, run.Q[k + 1])
    assert sum(len(run.episode(k)) for k in range(run.episodes)) == len(run.steps)


def test_epsilon_decays():
    assert G.epsilon(0) == 1.0
    assert G.epsilon(100) < 0.4 and G.epsilon(G.EPISODES - 1) == G.EPS_MIN


def test_converges_to_the_shortest_path():
    """The greedy path is a shortest path from episode 102 on, and the first
    episodes are what the animation says they are."""
    run = G.run()
    w = run.world
    shortest = len(w.shortest_path()) - 1
    for k in range(101, run.episodes):
        p = w.greedy_path(run.Q[k + 1])
        assert p[-1] == w.goal and len(p) - 1 == shortest, k
    assert run.lengths[0] == 21 and run.returns[0] == -1.0
    assert run.returns[:5].tolist() == [-1.0] * 5
    assert int(np.argmax(run.returns > 0)) + 1 == 41        # the goal is first reached
    assert (run.returns[-50:] == 1.0).sum() >= 45             # late episodes reach it (eps 0.05)
    # the first update that is not zero is the one into the penalty cell
    ep1 = run.episode(0)
    assert all(st.q_after == 0.0 for st in ep1[:-1])
    assert ep1[-1].s2 == w.penalty and ep1[-1].q_after == G.ALPHA * G.PENALTY_REWARD
    # values: 1 next to the goal, decaying by gamma per step along the path
    V = run.Q[-1].max(-1)
    path = w.greedy_path(run.Q[-1])
    for a, b in pairwise(path[:-1]):
        assert abs(V[a] - G.GAMMA * V[b]) < 0.05
    assert abs(V[path[-2]] - 1.0) < 1e-6


def test_any_seed_converges():
    """Not a lucky seed: a handful of other seeds end on a shortest path too."""
    w = G.World()
    shortest = len(w.shortest_path()) - 1
    for seed in (0, 1, 2, 3):
        p = w.greedy_path(G.run(seed=seed).Q[-1])
        assert p[-1] == w.goal and len(p) - 1 == shortest, seed


def test_animation_claims():
    """What the subtitles quote, and the timeline."""
    from voxanim.qlearning_anim import QLearning
    anim = QLearning(G.run())
    assert 40 <= anim.duration <= 60
    assert anim.first_goal == 41 and anim.converged == 102
    sc = anim.scene(anim.Tm["world"][1] - 0.1)
    assert sc.voxels and sc.texts

"""Tabular Q-learning in a small gridworld, with every episode recorded.

The world (:data:`MAP`): 6 rows by 8 columns, walls (``#``), a start (``S``),
a goal (``G``, reward +1, ends the episode) and a penalty cell (``X``, reward
-1, ends the episode).  Four actions (up, right, down, left) move one cell;
moving into a wall or off the grid leaves the agent where it is.  Every
other step is worth 0, so with a discount ``gamma < 1`` the best policy is
the shortest path to the goal.

The agent keeps a table ``Q[row, col, action]`` (zeros at the start) and
after every move updates one entry:

    Q(s, a) <- Q(s, a) + alpha (r + gamma max_a' Q(s', a') - Q(s, a))

where ``max_a' Q(s', a')`` is 0 when ``s'`` ends the episode.  It explores
epsilon-greedily: with probability ``epsilon`` a random action, otherwise the
best one (ties broken at random), and ``epsilon`` decays from 1 towards
:data:`EPS_MIN` as the episodes go by.

:func:`run` plays :data:`EPISODES` episodes from a seeded generator and records
every step (state, action, reward, next state, the Q entry before and after)
and the whole Q-table after each episode, so :mod:`voxanim.qlearning_anim`
can replay the learning exactly.  It takes well under a second, so it is not
cached.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

MAP = (
    "......#G",
    ".##...#.",
    ".....X..",
    "..#.....",
    "..#.##..",
    "S.#.....",
)
ACTIONS = ((-1, 0), (0, 1), (1, 0), (0, -1))     # up, right, down, left (row, col)
ACTION_NAMES = ("up", "right", "down", "left")
GOAL_REWARD, PENALTY_REWARD, STEP_REWARD = 1.0, -1.0, 0.0
ALPHA = 0.5                  # learning rate
GAMMA = 0.9                  # discount
EPS_MIN = 0.05               # exploration never falls below this
EPS_DECAY = 0.99             # epsilon of episode k (from 0) is max(EPS_MIN, EPS_DECAY ** k)
EPISODES = 400
MAX_STEPS = 200              # an episode is cut off after this many moves
SEED = 25                    # chosen so the first episode is short enough to watch (see run)


def epsilon(episode):
    """Exploration rate of ``episode`` (counted from 0)."""
    return max(EPS_MIN, EPS_DECAY ** episode)


class World:
    """The gridworld of :data:`MAP` (or another map in the same notation)."""

    def __init__(self, rows=MAP):
        self.map = tuple(rows)
        self.h, self.w = len(rows), len(rows[0])
        self.walls = np.array([[ch == "#" for ch in r] for r in rows])
        self.start = self._find("S")
        self.goal = self._find("G")
        self.penalty = self._find("X")

    def _find(self, ch):
        for r, row in enumerate(self.map):
            if ch in row:
                return (r, row.index(ch))
        raise ValueError("no %r in the map" % ch)

    def terminal(self, s):
        """True at the goal and the penalty cell."""
        return s == self.goal or s == self.penalty

    def step(self, s, a):
        """``(next state, reward)`` of action ``a`` (0..3) in state ``s = (row, col)``."""
        dr, dc = ACTIONS[a]
        r, c = s[0] + dr, s[1] + dc
        if not (0 <= r < self.h and 0 <= c < self.w) or self.walls[r, c]:
            r, c = s
        s2 = (r, c)
        if s2 == self.goal:
            return s2, GOAL_REWARD
        if s2 == self.penalty:
            return s2, PENALTY_REWARD
        return s2, STEP_REWARD

    def shortest_path(self):
        """The cells of a shortest start-to-goal path (breadth-first search),
        the start first and the goal last."""
        prev = {self.start: None}
        queue = deque([self.start])
        while queue:
            s = queue.popleft()
            if s == self.goal:
                break
            for a in range(len(ACTIONS)):
                s2, _r = self.step(s, a)
                if s2 not in prev and s2 != self.penalty:
                    prev[s2] = s
                    queue.append(s2)
        path, s = [], self.goal
        while s is not None:
            path.append(s)
            s = prev[s]
        return path[::-1]

    def greedy_path(self, Q, limit=MAX_STEPS):
        """Cells visited by following ``Q``'s best action from the start, until a
        terminal cell or ``limit`` moves (the first tie-free argmax is taken)."""
        s, path = self.start, [self.start]
        for _ in range(limit):
            if self.terminal(s):
                break
            s, _r = self.step(s, int(np.argmax(Q[s])))
            path.append(s)
        return path


@dataclass
class Step:
    """One move and the Q update it caused."""

    episode: int              # from 0
    s: tuple                  # (row, col) before
    a: int                    # action
    r: float                  # reward
    s2: tuple                 # (row, col) after
    q_before: float           # Q(s, a) before the update
    max_next: float           # max_a' Q(s', a') (0 when s' is terminal)
    q_after: float            # Q(s, a) after the update


@dataclass
class Run:
    """A :func:`run`: every step, and the Q-table after every episode."""

    world: World
    steps: list               # [Step], all episodes in order
    Q: np.ndarray             # (episodes + 1, h, w, 4): Q[0] is all zeros
    lengths: np.ndarray       # (episodes,) moves per episode
    returns: np.ndarray       # (episodes,) the final reward of each episode (+1, -1 or 0)

    def episode(self, k):
        """The steps of episode ``k``."""
        return [st for st in self.steps if st.episode == k]

    @property
    def episodes(self):
        return len(self.lengths)


def choose(rng, Q, s, eps):
    """Epsilon-greedy action in state ``s``: random with probability ``eps``,
    otherwise the best (ties broken at random).  Draws from ``rng`` in a
    fixed order so a run is reproducible."""
    if rng.random() < eps:
        return int(rng.integers(len(ACTIONS)))
    q = Q[s]
    best = np.flatnonzero(q == q.max())
    return int(best[rng.integers(len(best))])


def run(episodes=EPISODES, seed=SEED, alpha=ALPHA, gamma=GAMMA, world=None) -> Run:
    """Q-learning for ``episodes`` episodes from a seeded generator.

    The seed was picked so that the first episode is short enough to watch
    move by move (21 steps, ending at the penalty cell); the outcome does not
    depend on it: with these settings 40 seeds out of 40 end with the greedy
    path being a shortest path.  With :data:`SEED` the goal is first reached
    in episode 41 and the greedy path is shortest from episode 102 on.
    """
    world = world or World()
    rng = np.random.default_rng(seed)
    Q = np.zeros((world.h, world.w, len(ACTIONS)))
    steps, tables, lengths, returns = [], [Q.copy()], [], []
    for k in range(episodes):
        s, eps = world.start, epsilon(k)
        n, last = 0, 0.0
        while not world.terminal(s) and n < MAX_STEPS:
            a = choose(rng, Q, s, eps)
            s2, r = world.step(s, a)
            q0 = Q[s][a]
            mx = 0.0 if world.terminal(s2) else float(Q[s2].max())
            Q[s][a] = q0 + alpha * (r + gamma * mx - q0)
            steps.append(Step(k, s, a, r, s2, float(q0), mx, float(Q[s][a])))
            s, n, last = s2, n + 1, r
        tables.append(Q.copy())
        lengths.append(n)
        returns.append(last)
    return Run(world, steps, np.array(tables), np.array(lengths), np.array(returns))

"""Reinforcement learning: animations 91-96 of the "next 100" (see :mod:`voxanim.mech`).

Each draws the mechanism first and the measurement after it. The numbers are
the study's: bandits over ten MNIST classifiers (:mod:`voxanim.next_systems_real`),
an exact Monte Carlo tree search on tic-tac-toe (:mod:`voxanim.next_systems`), and
policy-gradient, actor-critic, Q-learning and reward-shaping runs on the glimpse
game over MNIST digits (:mod:`voxanim.study_rl`, five seeds per arm).

The bandit and tree-search studies keep their curves but not the steps behind
them (which arm was pulled, which move was tried), so those runs are made again
here with the steps recorded, and the tests hold each replay to the study's own
curve. The glimpse-game episodes drawn are played here by the networks the
study's first seed trained.
"""

from __future__ import annotations

import itertools
from functools import lru_cache

import numpy as np

from . import next_systems as S
from . import next_systems_real as SR
from . import realdata as R
from . import study_rl as G
from .cache import cached_npz
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIM,
    GREEN,
    MAGENTA,
    OCHRE,
    PAPER,
    RED,
    TEAL,
    TRACK,
    VIOLET,
    WHITE,
    Mechanism,
    dark,
    fit,
    soft,
)
from .optim import Adam

INK = (0.16, 0.17, 0.19)
SIDE = 7                      # the bandit study's digits are MNIST pooled to 7 x 7
PIXEL_COUNTS = (1, 2, 4, 7, 10, 15, 21, 28, 40, 49)      # the ten bandit arms


def _cached(name, build, **settings):
    source = R.source_hash("mech_rl.py", "next_systems_real.py", "next_systems.py",
                           "realdata.py", "optim.py")
    return {k: np.asarray(v) for k, v in cached_npz(
        "mech-rl-" + name, dict(version=1, source=source, **settings), build).items()}


# ---------------------------------------------------------------------- bandits, replayed
@lru_cache(maxsize=1)
def bandit_replay():
    """The bandit study again (:func:`voxanim.next_systems_real.bandits`), with
    every pull kept.

    Returns ``pixels`` (10, 49): the pixels each arm's classifier reads;
    ``table`` (10, 1000): whether each classifier got each validation digit
    right; ``means``; and per method (ε-greedy, UCB, Thompson) and seed:
    ``arm`` and ``outcome`` of every pull (3, 8, 1000) and the cumulative
    ``regret``; for the first seed, ``draws`` (1000, 10): the value Thompson
    sampling drew from each arm's posterior before each pull.
    """
    def build():
        x, y, _xt, _yt, _ = SR.digits()
        xv, yv, _ = R.mnist_split(1000, "validation", seed=19)
        rng = np.random.default_rng(17)
        table, pixels = [], np.zeros((10, SIDE * SIDE))
        for j, count in enumerate(PIXEL_COUNTS):
            selected = np.argsort(x.var(0))[-count:]
            pixels[j, selected] = 1
            a = np.c_[x[:, selected], np.ones(len(x))]
            av = np.c_[xv[:, selected], np.ones(len(xv))]
            parameters = {"w": np.zeros((count + 1, 10))}
            opt = Adam(parameters, 0.03)
            for _ in range(100):
                ids = rng.choice(len(a), 64, False)
                _, g = SR._linear_loss(parameters["w"], a[ids], y[ids])
                opt.step(parameters, {"w": g})
            table.append((av @ parameters["w"]).argmax(1) == yv)
        table = np.array(table)
        means = table.mean(1)
        arms = np.zeros((3, 8, 1000), int)
        outcomes = np.zeros((3, 8, 1000), int)
        regrets = np.zeros((3, 8, 1000))
        draws = np.zeros((1000, 10))
        picked = np.zeros(1000, int)                    # the validation digit of each pull
        for method in range(3):
            for seed in range(8):
                r = np.random.default_rng(seed + 17)
                n, wins, total = np.zeros(10), np.zeros(10), 0.0
                for t in range(1000):
                    q = wins / np.maximum(n, 1)
                    if method == 0:
                        arm = int(r.integers(10)) if r.random() < 0.1 else int(q.argmax())
                    elif method == 1:
                        arm = (int(np.flatnonzero(n == 0)[0]) if np.any(n == 0)
                               else int(np.argmax(q + np.sqrt(2 * np.log(t + 2) / n))))
                    else:
                        sample = r.beta(wins + 1, n - wins + 1)
                        arm = int(np.argmax(sample))
                        if seed == 0:
                            draws[t] = sample
                    which = int(r.integers(table.shape[1]))
                    outcome = table[arm, which]
                    if method == 2 and seed == 0:
                        picked[t] = which
                    n[arm] += 1
                    wins[arm] += outcome
                    total += means.max() - means[arm]
                    arms[method, seed, t], outcomes[method, seed, t] = arm, outcome
                    regrets[method, seed, t] = total
        return dict(pixels=pixels, table=table, means=means, arm=arms, outcome=outcomes,
                    regret=regrets, draws=draws, picked=picked, digits=xv[picked],
                    labels=yv[picked])

    return _cached("bandits", build, arms=10, pulls=1000, seeds=8)


# ---------------------------------------------------------------------- tree search, replayed
START = (1, 1, 0, -1, -1, 0, 0, 0, 0)      # the study's position: X X . / O O . / . . .


class RecordingMCTS(S.MCTS):
    """The study's search, with each simulation's steps kept: the ``path`` of
    boards from the root (the last one newly expanded, or a finished game),
    the boards of the random ``rollout`` after it, and the ``reward``."""

    def simulate(self):
        board, player = self.root, self.player
        path, rollout, expanded = [], [], False
        while True:
            key = (board, player)
            path.append(key)
            winner = S.tictactoe_winner(board)
            if winner:
                break
            moves = np.flatnonzero(np.array(board) == 0)
            if key not in self.children:
                self.children[key] = []
                for a in moves:
                    b = list(board)
                    b[a] = player
                    self.children[key].append((tuple(b), -player))
                child = self.children[key][int(self.rng.integers(len(moves)))]
                board, player = child
                path.append(child)
                expanded = True
                while not (winner := S.tictactoe_winner(board)):
                    a = int(self.rng.choice(np.flatnonzero(np.array(board) == 0)))
                    b = list(board)
                    b[a] = player
                    board = tuple(b)
                    player = -player
                    rollout.append(board)
                break
            children = self.children[key]
            unvisited = [c for c in children if self.n.get(c, 0) == 0]
            if unvisited:
                board, player = unvisited[int(self.rng.integers(len(unvisited)))]
                continue
            sign = 1 if player == self.player else -1

            def ucb(c, sign=sign, key=key):
                return sign * self.v[c] / self.n[c] + np.sqrt(
                    2 * np.log(max(self.n.get(key, 1), 1)) / self.n[c])

            board, player = max(children, key=ucb)
        reward = 0.0 if winner == 2 else float(winner == self.player) * 2 - 1
        for key in path:
            self.n[key] = self.n.get(key, 0) + 1
            self.v[key] = self.v.get(key, 0) + reward
        return dict(path=[k[0] for k in path], rollout=rollout, reward=reward, expanded=expanded)


@lru_cache(maxsize=1)
def mcts_replay(simulations=501):
    """The study's 501 simulations from :data:`START`: per simulation its
    record (see :class:`RecordingMCTS`) and, after it, the root's ``counts``
    and mean ``values`` per legal move, as the study keeps them; ``ucb``: the
    score of each root move the selection saw before that simulation (mean
    value, exploration bonus; None while a move is still unvisited)."""
    tree = RecordingMCTS(START)
    moves = [int(m) for m in np.flatnonzero(np.array(START) == 0)]
    kids = []
    for a in moves:
        b = list(START)
        b[a] = 1
        kids.append((tuple(b), -1))
    root = (START, 1)
    sims, counts, values, scores = [], [], [], []
    for _ in range(simulations):
        ns = [tree.n.get(k, 0) for k in kids]
        if min(ns) > 0:
            total = max(tree.n.get(root, 1), 1)
            scores.append([(tree.v[k] / tree.n[k], float(np.sqrt(2 * np.log(total) / tree.n[k])))
                           for k in kids])
        else:
            scores.append(None)
        sims.append(tree.simulate())
        counts.append([tree.n.get(k, 0) for k in kids])
        values.append([tree.v.get(k, 0) / max(tree.n.get(k, 0), 1) for k in kids])
    return dict(moves=moves, kids=[k[0] for k in kids], sims=sims, counts=np.array(counts).T,
                values=np.array(values).T, ucb=scores, tree=tree)

# ---------------------------------------------------------------------- shared drawing
class RL(Mechanism):
    """What the six animations draw with: upright bars and straight lines."""

    def vbar(self, sc, fx, base, w, h, ink, op, fill_op=1.0, border=None, lift=0.5):
        """An upright bar ``w`` px wide rising ``h`` px (negative: hanging) from
        layout ``(fx, base)``."""
        if op <= 0.01:
            return
        h = h if abs(h) >= 1.0 else (1.0 if h >= 0 else -1.0)
        dy = self.frac(0, h)[1]
        self.rect(sc, self.P(float(fx), float(base - dy / 2)), w, abs(h), ink, op,
                  radius=min(3.0, abs(h) / 2, w / 2), fill_op=fill_op, border=border, lift=lift)

    def seg(self, sc, a, b, ink, op, width=2.0, lift=1.0):
        """A straight line between two layout points, in front of the boxes."""
        if op > 0.01:
            self.polyline(sc, [self.P(float(a[0]), float(a[1]), lift),
                               self.P(float(b[0]), float(b[1]), lift)], width, ink, op, bias=0.3)

    def bars(self, sc, values, x0, x1, base, height, op, ink, labels=None, chosen=None,
             truth=None, w=None, top=1.0, size=0.034, chosen_ink=AMBER, numbers=False):
        """A row of upright bars (``values`` out of ``top``) between ``x0`` and
        ``x1`` standing on ``base``, ``height`` px tall at ``top``, each on its
        track with its label underneath; ``chosen``: the bar ringed;
        ``truth``: the bar whose label is marked as the right answer. Returns
        the bars' x fractions."""
        n = len(values)
        xs = [x0 + (x1 - x0) * (i + 0.5) / n for i in range(n)]
        w = w if w is not None else self.span((x1 - x0) / n)[0] - 8.0
        for i, (v, x) in enumerate(zip(values, xs)):
            self.vbar(sc, x, base, w, height, TRACK, 0.45 * op, lift=0.2)
            self.vbar(sc, x, base, w, height * float(np.clip(v / top, 0, 1)),
                      ink[i] if isinstance(ink, list) else ink, op)
            if chosen is not None and i == chosen:
                dy = self.frac(0, height)[1]
                self.rect(sc, self.P(x, base - dy / 2), w + 8, height + 8, chosen_ink, op,
                          border=chosen_ink, border_w=3.0, radius=5, lift=0.8, fill_op=0.0)
            if labels is not None:
                right = truth is not None and i == truth
                self.text(sc, x, base + 0.038, labels[i], op, size=size, strong=right,
                          ink=None if right else AXIS)
                if right:
                    self.seg(sc, (x - self.frac(w / 2)[0], base + 0.066),
                             (x + self.frac(w / 2)[0], base + 0.066), INK, op, 2.6)
        return xs


def _pct(v):
    return "%d%%" % round(100 * float(v))


# ---------------------------------------------------------------------- 91
class MultiArmedBandits(RL):
    """Ten slot-machine arms that are ten digit classifiers; three ways of
    deciding which to pull next, each replayed pull by pull on its estimates;
    then the regret each one ran up."""

    TITLE = "Multi-armed bandits"
    PHASES = (("ten arms", 11.0), ("ε-greedy", 10.0), ("UCB", 11.0), ("Thompson", 12.0),
              ("regret", 11.0))
    X = (0.06, 0.94)
    TRACK_Y = (0.465, 0.80)            # the top and the bottom (value 0) of every arm's scale
    MASK_Y, COUNT_Y, PAID_Y, NOTE_Y, HEAD_Y = 0.385, 0.845, 0.888, 0.93, 0.292
    TOPS = (1.0, 2.6, 1.0)             # the value at the top of the scale, per method
    METHODS = (("ε-greedy", RED), ("UCB", BLUE), ("Thompson", VIOLET))
    SLOW_FROM = (0, 10, 0)             # the first pull shown slowly, per method

    def setup(self):
        s = self.study
        d = self.data = bandit_replay()
        self.means = np.asarray(s.panels[0].data["values"], float)
        self.regret = np.asarray(s.panels[1].data["ys"], float)
        self.counts = np.asarray(s.panels[2].data["values"], float)
        self.pulls = self.regret.shape[1]
        self.best = int(self.means.argmax())
        # the first seed's run of each method: pulls and wins of every arm before pull t
        hot = np.eye(10)[d["arm"][:, 0]]                                  # (3, 1000, 10)
        self.n = np.concatenate([np.zeros((3, 1, 10)), hot.cumsum(1)], 1)
        self.wins = np.concatenate(
            [np.zeros((3, 1, 10)), (hot * d["outcome"][:, 0][..., None]).cumsum(1)], 1)
        self.run_regret = np.concatenate([np.zeros((3, 1)), d["regret"][:, 0]], 1)
        self.xs = [self.X[0] + (self.X[1] - self.X[0]) * (i + 0.5) / 10 for i in range(10)]

    def subtitle_for(self, k):
        final = self.regret[:, -1]
        share = self.counts[:, self.best] / self.pulls
        return [
            "ten arms to pull: ten digit classifiers that read from 1 to 49 pixels. A pull pays 1 "
            "when the classifier names a digit correctly; their true rates run from %s to %s"
            % (_pct(self.means.min()), _pct(self.means.max())),
            "ε-greedy pulls the arm with the best average so far, except one pull in ten, which "
            "goes to an arm picked at random",
            "UCB adds a bonus to each average that is large for arms it has pulled little, and "
            "pulls the highest total; the bonus shrinks as an arm is pulled",
            "Thompson sampling keeps a belief about each arm's rate, draws one guess from each "
            "belief and pulls the highest guess; the beliefs narrow as pulls come in",
            "regret is the reward lost against always pulling the best arm. After %s pulls: "
            "Thompson %.0f, ε-greedy %.0f, UCB %.0f; Thompson gave the best arm %s of its pulls"
            % (f"{self.pulls:,}", final[2], final[0], final[1], _pct(share[2])),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _y(self, v, top=1.0):
        y0, y1 = self.TRACK_Y
        return y1 - (y1 - y0) * float(min(v, top)) / top

    def _arms(self, sc, op, shown=10, top=1.0, track=True):
        """The ten arms: the pixels each classifier reads, over its scale."""
        px = self.span(0, self.TRACK_Y[1] - self.TRACK_Y[0])[1]
        for j, x in enumerate(self.xs[:shown]):
            self.grid(sc, self.data["pixels"][j].reshape(SIDE, SIDE), x, self.MASK_Y, 48, 48, op,
                      vmax=1.0, hot=TEAL, lines=0.1)
            if track:
                self.vbar(sc, x, self.TRACK_Y[1], 30, px, TRACK, 0.45 * op, lift=0.2)
        for v in (0, 1):
            self.text(sc, 0.058, self._y(v, top), "%d" % v, op, ink=AXIS)
        return px

    def _truth(self, sc, op, top=1.0):
        for j, x in enumerate(self.xs):
            y = self._y(self.means[j], top)
            half = self.frac(24)[0]
            self.seg(sc, (x - half, y), (x + half, y), dark(TEAL, 0.35), op, 3.4, lift=1.2)

    def phase_1(self, sc, u, op):
        shown = int(np.clip(u / 0.36 * 10 + 1, 1, 10))
        px = self._arms(sc, op, shown)
        for j, x in enumerate(self.xs[:shown]):
            rise = smoothstep(0.0, 1.0, u / 0.36 * 10 - j)
            self.vbar(sc, x, self.TRACK_Y[1], 30, px * self.means[j] * rise, TEAL, op)
            self.text(sc, x, self.COUNT_Y, _pct(self.means[j]), op * rise, size=0.034, strong=True)
            self.text(sc, x, self.PAID_Y, "%d px" % PIXEL_COUNTS[j], op * rise, ink=AXIS)
        self.text(sc, self.X[0], self.HEAD_Y, "each arm: a classifier that reads only the coloured "
                  "pixels of a 7 x 7 digit", op, anchor="lm", strong=True, size=0.034)
        # four real pulls: the digit, what that arm's classifier sees of it, right or wrong
        d = self.data
        k = (u - 0.42) / 0.56 * 4
        if 0 <= k < 4:
            i, w = int(k), k - int(k)
            arm, x = int(d["arm"][2, 0, i]), self.xs[int(d["arm"][2, 0, i])]
            seen = smoothstep(0.25, 0.5, w)
            image = d["digits"][i] * (1 - seen + seen * d["pixels"][arm])
            show = op * smoothstep(0.0, 0.12, w) * (1 - smoothstep(0.9, 1.0, w))
            self.digit(sc, image, x, self.MASK_Y, 48, show, frame=AMBER)
            won = bool(d["outcome"][2, 0, i])
            self.text(sc, 0.5, self.NOTE_Y, "a pull: a real digit (a %d) goes to arm %d, which "
                      "sees %d of its pixels and gets it %s: reward %d" % (
                          d["labels"][i], arm + 1, PIXEL_COUNTS[arm],
                          "right" if won else "wrong", won),
                      show * smoothstep(0.45, 0.6, w), size=0.032,
                      ink=dark(GREEN) if won else dark(RED), strong=False)
        elif u < 0.42:
            self.text(sc, 0.5, self.NOTE_Y, "bar: the share of 1,000 validation digits the "
                      "classifier gets right (the player is not told)",
                      op * (1 - smoothstep(0.38, 0.42, u)), size=0.032)

    def _clock(self, u, method, slow=6, part=0.56):
        """The pull being shown at ``u``: ``(t, w)`` with ``w`` the fraction of
        the way through it, or None once the replay runs fast."""
        start = self.SLOW_FROM[method]
        lead = 0.10 if start else 0.0
        if u < lead:
            return int(u / lead * start), None
        if u < part:
            k = (u - lead) / (part - lead) * slow
            return start + min(int(k), slow - 1), k - int(k)
        f = min((u - part) / (0.94 - part), 1.0)
        first = start + slow
        return int(min(first * (self.pulls / first) ** f, self.pulls)), None

    def _play(self, sc, u, op, method):
        name, ink = self.METHODS[method]
        d = self.data
        top = self.TOPS[method]
        px = self._arms(sc, op, top=top, track=method < 2)
        self._truth(sc, op, top)
        t, w = self._clock(u, method)
        live = w is not None
        done = smoothstep(0.72, 0.9, w) if live else 0.0
        n = self.n[method, t] + (self.n[method, t + 1] - self.n[method, t]) * done if live \
            else self.n[method, t]
        wins = self.wins[method, t] + (self.wins[method, t + 1] - self.wins[method, t]) * done \
            if live else self.wins[method, t]
        q = wins / np.maximum(n, 1)
        n0, w0 = self.n[method, min(t, self.pulls - 1)], self.wins[method, min(t, self.pulls - 1)]
        arm = int(d["arm"][method, 0, min(t, self.pulls - 1)])
        decide = smoothstep(0.08, 0.3, w) if live else 0.0
        half = self.frac(15)[0]
        for j, x in enumerate(self.xs):
            if method < 2:
                self.vbar(sc, x, self.TRACK_Y[1], 30, px * q[j] / top, ink,
                          op if n[j] else 0.0)
            if method == 1 and n[j] > 0:
                total = q[j] + float(np.sqrt(2 * np.log(t + 2) / max(n[j], 1)))
                self.seg(sc, (x, self._y(q[j], top)), (x, self._y(total, top)), dark(ink), op, 3.0)
                self.seg(sc, (x - half * 0.8, self._y(total, top)),
                         (x + half * 0.8, self._y(total, top)), dark(ink), op, 3.0)
            if method == 2:
                a, b = wins[j] + 1, n[j] - wins[j] + 1
                p = (np.arange(36) + 0.5) / 36
                dens = np.exp((a - 1) * np.log(p) + (b - 1) * np.log(1 - p))
                dens = dens / dens.max()
                for pk, dk in zip(p, dens):
                    if dk > 0.03:
                        self.rect(sc, self.P(x, self._y(pk)), 4 + 62 * dk, px / 36, ink, op,
                                  radius=0.0, lift=0.3)
                if live:
                    self.disk(sc, self.P(x, self._y(d["draws"][t, j])), 6.5, WHITE if j != arm
                              else AMBER, op * decide * (1 - done), lift=1.6, outline=0.5)
            self.text(sc, x, self.COUNT_Y, "%d" % round(n[j]), op, size=0.034, strong=True)
        if live:
            # the arm the rule picks, and what the pull paid
            dy = self.TRACK_Y[1] - self.TRACK_Y[0]
            self.rect(sc, self.P(self.xs[arm], self.TRACK_Y[0] + dy / 2), 72, px + 14, AMBER,
                      op * decide * (1 - smoothstep(0.9, 1.0, w)), border=AMBER, border_w=3.0,
                      radius=6, lift=0.9, fill_op=0.0)
            won = bool(d["outcome"][method, 0, t])
            self.text(sc, self.xs[arm], self.PAID_Y, "paid %d" % won,
                      op * smoothstep(0.45, 0.6, w) * (1 - smoothstep(0.9, 1.0, w)), size=0.032,
                      ink=dark(GREEN) if won else dark(RED))
            greedy = int(np.argmax(w0 / np.maximum(n0, 1)))
            why = ["the best average so far" if arm == greedy else "a random arm (the 1 in 10)",
                   "the highest average + bonus", "the highest guess"][method]
            self.text(sc, self.X[0], self.HEAD_Y, "%s pulls arm %d: %s" % (name, arm + 1, why), op,
                      anchor="lm", strong=True, size=0.034)
        else:
            self.text(sc, self.X[0], self.HEAD_Y, name + [
                ": bar = average reward of the arm so far",
                ": bar = average so far, whisker = its bonus",
                ": each shape = the belief about that arm's rate"][method], op, anchor="lm",
                strong=True, size=0.034)
        self.text(sc, self.X[1], self.HEAD_Y, "pull %s of %s    regret %.1f" % (
            f"{min(t + 1, self.pulls):,}", f"{self.pulls:,}",
            self.run_regret[method, min(t + (1 if live else 0), self.pulls)]), op, anchor="rm",
            size=0.034, ink=dark(ink))
        self.text(sc, 0.5, self.NOTE_Y, "number: pulls of that arm so far    dark line: its true "
                  "rate, which the player never sees", op, size=0.030)

    def phase_2(self, sc, u, op):
        self._play(sc, u, op, 0)

    def phase_3(self, sc, u, op):
        self._play(sc, u, op, 1)

    def phase_4(self, sc, u, op):
        self._play(sc, u, op, 2)

    def phase_5(self, sc, u, op):
        steps = np.arange(1, self.pulls + 1)
        chart = self.chart(((0.10, 0.56), (0.35, 0.84)), steps, self.regret)
        self.axes(sc, chart, op, "pull", "regret (mean of 8 runs)")
        g = u / 0.6
        for (name, ink), trace in zip(self.METHODS, self.regret):  # noqa: B007
            self.plot(sc, chart, steps, trace, ink, op, g)
        now = int(min(g, 1.0) * (self.pulls - 1))
        self.lines(sc, 0.12, 0.40, [("%s  %.0f" % (name, trace[now]), dark(ink))
                                    for (name, ink), trace in zip(self.METHODS, self.regret)],
                   op, step=0.05, size=0.034)
        # where each method's 1,000 pulls went
        late = op * smoothstep(0.35, 0.55, u)
        self.text(sc, 0.79, 0.30, "where the %s pulls went" % f"{self.pulls:,}", late,
                  strong=True, size=0.034)
        for row, ((name, ink), counts) in enumerate(zip(self.METHODS, self.counts)):
            y = 0.44 + row * 0.175
            self.text(sc, 0.64, y - 0.072, name, late, anchor="lm", ink=dark(ink), size=0.032)
            self.bars(sc, counts, 0.64, 0.94, y + 0.06, 54, late, ink, top=self.counts.max(),
                      w=22.0)
            self.text(sc, 0.94, y - 0.072, "best arm: %s" % _pct(counts[self.best] / self.pulls),
                      late, anchor="rm", size=0.030)
        self.text(sc, 0.79, 0.915, "arms 1 to 10, worst to best", late, ink=AXIS)


# ---------------------------------------------------------------------- the glimpse game, drawn
DIGITS = [str(i) for i in range(10)]


def rolling(v, n=25):
    """The mean of the last ``n`` values (fewer at the start): exact arithmetic
    on a measured trace, to make its level readable."""
    v = np.asarray(v, float)
    c = np.cumsum(np.insert(v, 0, 0.0))
    i = np.arange(1, len(v) + 1)
    lo = np.maximum(i - n, 0)
    return (c[i] - c[lo]) / (i - lo)


def _spread(v):
    """Mean and range of a measurement over seeds, as percentages."""
    v = np.asarray(v, float)
    return "%s (%d to %d)" % (_pct(v.mean()), round(100 * v.min()), round(100 * v.max()))


@lru_cache(maxsize=4)
def demo(kind, update=300, n=256):
    """Real episodes of look, look, name (:func:`voxanim.study_rl.rollout`): the first
    seed's ``kind`` policy as it was after ``update`` updates, playing the first ``n``
    training digits. Adds ``image``, ``label``, ``right``, the critic's ``value`` (3, n)
    and each estimator's multiplier (``return``, ``baseline``, ``critic``: (3, n))."""
    P, C = G.kept(G.pg_runs(), kind, update)
    x, y = G.digits()[:2]
    ep = G.rollout(P, x[:n], y[:n], np.random.default_rng(update + 1))
    v = G.values(C, ep)
    ep.update(image=x[:n], label=y[:n], value=v, right=ep["reward"] == G.POINTS[1],
              **{k: G.weights(k, ep, v) for k in G.ESTIMATORS})
    return ep


def pick(ep, right):
    """The first kept episode that named its digit right (or wrong) after uncovering two
    different quadrants."""
    ok = (ep["right"] == right) & (ep["act"][0] != ep["act"][1])
    return int(np.flatnonzero(ok)[0])


@lru_cache(maxsize=4)
def free_demo(which, n=200):
    """Greedy episodes of the free game (:func:`voxanim.study_rl.play`) on the first ``n``
    held-out digits, by a first-seed network: ``dqn`` (replay and target network, digits
    in runs) or 0, 1, 2 (the reward-shaping arms)."""
    if which == "dqn":
        d = G.dqn_runs()
        P = {k: d["P_" + k] for k in "WbUc"}
    else:
        d = G.shaping_runs()
        P = {k: d["P%d_%s" % (which, k)] for k in "WbUc"}
    xt, yt = G.digits()[2:4]
    ep = G.play(P, xt[:n], yt[:n])
    ep.update(image=xt[:n], label=yt[:n])
    return ep


class Glimpse(RL):
    """What the four glimpse-game animations share: a digit with covered quadrants, the
    bars over its actions, an episode of look, look, name as three columns, and curves
    with one thin line per seed."""

    COLS = (0.06, 0.33, 0.60)           # where each step of look, look, name starts
    HEAD_Y, MID_Y, BASE, TALL, IMG, NOTE_Y = 0.292, 0.462, 0.56, 104.0, 92.0, 0.93
    SAID_Y = 0.655
    STEP_NAMES = ("1  uncover a quadrant", "2  uncover another", "3  name the digit")

    def covered(self, sc, image, mask, fx, fy, size, op, ring=None, ring_ink=AMBER):
        """A 14 x 14 digit with the quadrants not in ``mask`` covered; ``ring``: a
        quadrant outlined (the one being uncovered)."""
        mask = np.asarray(mask, float)
        self.digit(sc, np.asarray(image) * (mask @ G.QUAD), fx, fy, size, op, frame=AXIS)
        dw, dh = self.frac(size, size)
        for q in range(4):
            c = self.P(fx + (q % 2 - 0.5) * dw / 2, fy + (q // 2 - 0.5) * dh / 2)
            if not mask[q]:
                self.rect(sc, c, size / 2 - 2, size / 2 - 2, DIM, op, radius=3, lift=0.6)
            if ring == q:
                self.rect(sc, c, size / 2 + 2, size / 2 + 2, ring_ink, op, border=ring_ink,
                          border_w=3.0, radius=3, lift=0.9, fill_op=0.0)

    def icon(self, sc, q, fx, fy, op, size=16.0, ink=TEAL):
        """The action "uncover quadrant ``q``": a small square with that quarter filled."""
        self.rect(sc, self.P(fx, fy), size, size, PAPER, op, border=AXIS, border_w=1.0,
                  radius=2, lift=0.3)
        dw, dh = self.frac(size, size)
        self.rect(sc, self.P(fx + (q % 2 - 0.5) * dw / 2, fy + (q // 2 - 0.5) * dh / 2),
                  size / 2, size / 2, ink, op, radius=0, lift=0.5)

    def look_bars(self, sc, values, x0, base, tall, op, chosen=None, top=1.0, pitch=0.034,
                  ink=TEAL):
        xs = self.bars(sc, np.clip(values, 0, None), x0, x0 + 4 * pitch, base, tall, op, ink,
                       chosen=chosen, top=top)
        for q, x in enumerate(xs):
            self.icon(sc, q, x, base + 0.04, op)
        return xs

    def strip(self, sc, ep, j, op, ink, w=1.0):
        """Episode ``j`` of ``ep`` as three columns, appearing left to right as ``w``
        goes 0..1: the digit as covered at that step, the policy's probabilities, the
        sampled action ringed; then what the name earned."""
        label = int(ep["label"][j])
        for t, x0 in enumerate(self.COLS):
            shown = op * smoothstep(0.0, 0.1, w - 0.25 * t)
            picked = smoothstep(0.12, 0.2, w - 0.25 * t) > 0.5
            a = int(ep["act"][t, j])
            self.text(sc, x0, self.HEAD_Y, self.STEP_NAMES[t], shown, anchor="lm", strong=True,
                      size=0.032)
            self.covered(sc, ep["image"][j], ep["mask"][t, j], x0 + 0.052, self.MID_Y, self.IMG,
                         shown, ring=a if t < 2 and picked else None)
            if t < 2:
                self.look_bars(sc, ep["probs"][t, j, :4], x0 + 0.115, self.BASE, self.TALL,
                               shown, chosen=a if picked else None)
            else:
                self.bars(sc, ep["probs"][2, j, 4:], x0 + 0.115, 0.94, self.BASE, self.TALL,
                          shown, ink, labels=DIGITS, truth=label,
                          chosen=a - 4 if picked else None)
                right = a - 4 == label
                self.text(sc, x0 + 0.115, self.SAID_Y, "names %d, %s: %d point%s" % (
                    a - 4, "right" if right else "it is a %d" % label, ep["reward"][j],
                    "s" if right else ""), shown * float(picked), anchor="lm", size=0.032,
                    ink=dark(GREEN) if right else dark(RED))

    def signed(self, sc, fx, fy, v, op, per=0.075, grow=1.0):
        """A push as a bar from a zero line at ``fx``: right and green when positive."""
        self.seg(sc, (fx, fy - 0.022), (fx, fy + 0.022), AXIS, op, 1.6)
        w = self.span(per * abs(float(v)) * grow)[0]
        if w > 0.5:
            self.rect(sc, self.P(fx + np.sign(v) * self.frac(w / 2)[0], fy), w, 16,
                      GREEN if v > 0 else RED, op, radius=min(3.0, w / 2))

    def seeds(self, sc, chart, x, runs, ink, op, g):
        """One thin line per seed and their mean, heavier."""
        for run in runs:
            self.plot(sc, chart, x, run, ink, 0.3 * op, g, width=1.0, dot=False)
        return self.plot(sc, chart, x, np.mean(runs, 0), ink, op, g, width=2.6)

    def seed_bars(self, sc, values, x0, y, op, ink, name, top=1.0, width=0.30):
        """A measurement of five seeds: the mean as a bar, each seed a dot on it."""
        full = self.span(width)[0]
        v = np.asarray(values, float)
        self.text(sc, x0, y - 0.047, name, op, anchor="lm", ink=dark(ink), size=0.032)
        self.hbar(sc, x0, y, full * v.mean() / top, ink, op, "", "", thick=24.0, track=full)
        for s in v:
            self.disk(sc, self.P(x0 + width * s / top, y), 4.0, WHITE, op, lift=1.4, outline=0.6)


class LookLookName(Glimpse):
    """The policy-gradient and actor-critic animations: episodes of look, look, name with
    each estimator's push under every action, then the gradient noise and the runs."""

    ARMS = ()                           # (estimator, name, ink) of the runs compared
    PROBED = "b"                        # whose policies the gradient noise was measured on
    PUSH_Y = 0.83

    def setup(self):
        s, d = self.study, G.pg_runs()
        rows = [G.ESTIMATORS.index(k) for k, _n, _i in self.ARMS]
        self.acc = d["accuracy"][rows]                              # (arm, seed, evaluation)
        self.acc_x = np.asarray(s.panels[0].data["x"], float)
        self.mean_acc = np.asarray(s.panels[0].data["ys"], float)
        self.noise = np.asarray(s.panels[2].data["ys"], float)      # (arm, probe)
        self.noise_x = np.asarray(s.panels[2].data["x"], float)
        self.noise_seeds = d["noise_" + self.PROBED].mean(1)[:, rows].T    # (arm, seed)
        self.final = np.asarray(s.panels[3].data["values"], float).reshape(len(rows), -1)
        self.updates = int(self.acc_x[-1])
        self.ep = demo(self.ARMS[-1][0])
        self.shown = [pick(self.ep, False), pick(self.ep, True)]

    def pushes(self, sc, j, kind, op, grow=1.0):
        """Under each column of episode ``j``: what its action's push is made of."""
        ep = self.ep
        for t, x0 in enumerate(self.COLS):
            v, w, r = float(ep["value"][t, j]), float(ep[kind][t, j]), float(ep["reward"][j])
            if kind == "return":
                rows = [("the reward that followed: %d" % r, None)]
            elif kind == "baseline":
                rows = [("reward %d - expected %.2f" % (r, v), dark(VIOLET))]
            elif t < 2:
                rows = [("expected %.2f, now %.2f" % (v, ep["value"][t + 1, j]),
                         dark(VIOLET))]
            else:
                rows = [("reward %d - expected %.2f" % (r, v), dark(VIOLET))]
            self.lines(sc, x0, 0.725, rows, op, size=0.030)
            self.text(sc, x0, 0.775, "push %+.2f" % w, op, anchor="lm", size=0.034,
                      strong=True)
            self.signed(sc, x0 + 0.115, self.PUSH_Y, w, op, grow=grow)

    def several(self, sc, u, op, kind, ink):
        k = min(int(u * 2), 1)
        w = u * 2 - k
        fade = (smoothstep(0.0, 0.06, w) if k else 1.0) * (1 - smoothstep(0.94, 1.0, w)
                                                          if k == 0 else 1.0)
        j = self.shown[k]
        self.strip(sc, self.ep, j, op * fade, ink, 1.0 if k or u > 0.02 else w)
        self.pushes(sc, j, kind, op * fade * smoothstep(0.12, 0.22, w),
                    smoothstep(0.25, 0.5, w))
        self.text(sc, 0.5, self.NOTE_Y, "bar: how far the sampled action's score is pushed "
                  "(right, green: likelier next time; left, red: less likely)", op * fade,
                  size=0.030)

    def noise_phase(self, sc, u, op, head):
        self.text(sc, 0.06, self.HEAD_Y, head, op, anchor="lm", strong=True, size=0.032)
        top = self.noise_seeds.max()
        for row, ((_k, name, ink), v) in enumerate(zip(self.ARMS, self.noise_seeds)):
            y = 0.41 + row * 0.115
            self.seed_bars(sc, v * smoothstep(0.0, 0.3, u), 0.06, y, op, ink,
                           "%s: %.2f" % (name, v.mean()), top=top)
        self.lines(sc, 0.06, 0.41 + len(self.ARMS) * 0.115 - 0.02, [
            "bar: mean of 5 runs    dot: one run",
            "true gradient: from 8,192 episodes",
            "estimate: one batch of 16 episodes",
            "all estimates on the same 200 batches"], op, size=0.030, step=0.043)
        chart = self.chart(((0.50, 0.94), (0.36, 0.84)), self.noise_x, [0.0, self.noise.max()])
        self.axes(sc, chart, op, "update", "squared distance from the true gradient")
        for (_k, _n, ink), trace in zip(self.ARMS, self.noise):
            self.plot(sc, chart, self.noise_x, trace, ink, op, u / 0.7)

    def learning_phase(self, sc, u, op):
        g = min(u / 0.8, 1.0)
        k = int(g * (len(self.acc_x) - 1))
        self.text(sc, 0.06, self.HEAD_Y, "held-out digits named, update %d" % self.acc_x[k], op,
                  anchor="lm", strong=True, size=0.032)
        for row, ((_k, name, ink), runs) in enumerate(zip(self.ARMS, self.acc)):
            y = 0.41 + row * 0.115
            self.seed_bars(sc, runs[:, k], 0.06, y, op, ink, "%s: %s" % (name, _pct(
                runs[:, k].mean())))
        self.text(sc, 0.06, 0.41 + len(self.ARMS) * 0.115 - 0.02,
                  "bar: mean of 5 runs    dot: one run", op, anchor="lm", size=0.030)
        chart = self.chart(((0.50, 0.94), (0.36, 0.84)), self.acc_x, [0.0, self.acc.max()])
        self.axes(sc, chart, op, "update", "held-out accuracy (thin: each run)")
        for (_k, _n, ink), runs in zip(self.ARMS, self.acc):
            self.seeds(sc, chart, self.acc_x, runs, ink, op, g)


# ---------------------------------------------------------------------- 92
class PolicyGradient(LookLookName):
    """REINFORCE on look, look, name: an episode sampled from the policy, the push its
    actions get from the reward that followed, the same push with a learned baseline
    subtracted, how far each leaves a batch's gradient from the true one, and five runs
    of each."""

    TITLE = "Policy gradient (REINFORCE)"
    PHASES = (("an episode", 10.0), ("the push", 12.0), ("a baseline", 12.0),
              ("gradient noise", 11.0), ("five runs", 11.0))
    ARMS = (("return", "no baseline", RED), ("baseline", "learned baseline", BLUE))

    def subtitle_for(self, k):
        n, f = self.noise.mean(1), self.final
        clear = " Every run with the baseline beat every run without" if f[1].min() > f[
            0].max() else ""
        return [
            "a real digit starts covered. The policy picks a quadrant to uncover, twice, then "
            "names the digit. It scores 1 point for an answer, 2 for the right one, paid at the "
            "end",
            "REINFORCE: make every action of the episode likelier in proportion to the reward "
            "that followed. Every reward here is positive, so a wrong episode is pushed up too, "
            "only less",
            "a baseline: a second network learns the reward to expect, and the push becomes "
            "reward minus expected. Now a wrong answer pushes its actions down, a right one up",
            "measured on the same policies and episodes: a batch's gradient lies %.2f from the "
            "true gradient without the baseline and %.2f with it, %.1f times closer" % (
                n[0], n[1], n[0] / n[1]),
            "five runs each, %s updates. Held-out digits named: %s without the baseline, %s "
            "with it.%s" % (f"{self.updates:,}", _spread(f[0]), _spread(f[1]), clear),
        ][k]

    def phase_1(self, sc, u, op):
        j = self.shown[1]
        self.strip(sc, self.ep, j, op, BLUE, u * 1.6)
        late = op * smoothstep(0.5, 0.62, u)
        self.lines(sc, 0.5, 0.75, [
            "bars: the probability the policy gives each action    ring: the one it sampled",
            "a real training digit (a %d), played by the policy as it was after 300 updates"
            % self.ep["label"][j]], late, size=0.032, step=0.055, anchor="mm")

    def phase_2(self, sc, u, op):
        self.several(sc, u, op, "return", BLUE)

    def phase_3(self, sc, u, op):
        self.several(sc, u, op, "baseline", BLUE)

    def phase_4(self, sc, u, op):
        self.noise_phase(sc, u, op, "how far a batch's gradient is from the true one")

    def phase_5(self, sc, u, op):
        self.learning_phase(sc, u, op)


# ---------------------------------------------------------------------- 93
class ActorCritic(LookLookName):
    """An actor (the policy) and a critic (the points it expects) on look, look, name:
    the critic's expectation at each step, the push each action gets from the change in
    it, the gradient noise of that push against REINFORCE's, and five runs of each."""

    TITLE = "Actor-critic"
    PHASES = (("actor, critic", 11.0), ("the advantage", 15.0), ("gradient noise", 12.0),
              ("five runs", 11.0))
    ARMS = (("return", "raw reward (REINFORCE)", RED), ("baseline", "reward - baseline", BLUE),
            ("critic", "actor-critic", VIOLET))
    PROBED = "c"

    def subtitle_for(self, k):
        ep, j = self.ep, self.shown[1]
        n, f, s = self.noise.mean(1), self.final, self.noise_seeds
        every = ", lower in all five runs" if (s[2] < s[1]).all() and (s[2] < s[0]).all() \
            else ""
        return [
            "two networks play each digit. The actor picks the actions; the critic says how "
            "many points it expects from here on, and its expectation moves as quadrants are "
            "uncovered",
            "actor-critic: an action is pushed by the change it made to the critic's "
            "expectation, not by the final reward. Here the second look moved it from %.2f to "
            "%.2f: push %+.2f" % (ep["value"][1, j], ep["value"][2, j], ep["critic"][1, j]),
            "the same batches, three pushes. A batch's gradient lies %.2f from the true one "
            "using the raw reward, %.2f using reward minus a baseline, %.2f using the critic's "
            "change%s" % (n[0], n[1], n[2], every),
            "five runs each, held-out digits named: actor-critic %s, reward minus a baseline "
            "%s, the raw reward %s" % (_spread(f[2]), _spread(f[1]), _spread(f[0])),
        ][k]

    def critic(self, sc, j, op, upto=3.0):
        """The critic's expectation under each column, on a scale from 1 to 2 points."""
        for t, x0 in enumerate(self.COLS):
            shown = op * smoothstep(0.0, 0.5, upto - t)
            v = float(np.clip(self.ep["value"][t, j], 1.0, 2.0))
            full = self.span(0.12)[0]
            self.hbar(sc, x0, 0.725, full * (v - 1.0), VIOLET, shown, "", "", thick=18.0,
                      track=full)
            self.text(sc, x0 + 0.13, 0.725, "expects %.2f" % self.ep["value"][t, j], shown,
                      anchor="lm", ink=dark(VIOLET), size=0.030)
            for tick in (1, 2):
                self.text(sc, x0 + 0.12 * (tick - 1), 0.768, str(tick), shown, ink=AXIS)

    def phase_1(self, sc, u, op):
        j = self.shown[1]
        self.strip(sc, self.ep, j, op, BLUE, u * 2.0)
        self.critic(sc, j, op, (u - 0.3) / 0.12)
        late = op * smoothstep(0.7, 0.8, u)
        self.lines(sc, 0.5, 0.84, [
            "teal, blue: the actor's probabilities    violet: the points the critic expects "
            "from this step on",
            "a real training digit (a %d), played by the actor as it was after 300 updates"
            % self.ep["label"][j]], late, size=0.030, step=0.047, anchor="mm")

    def phase_2(self, sc, u, op):
        self.several(sc, u, op, "critic", BLUE)

    def phase_3(self, sc, u, op):
        self.noise_phase(sc, u, op, "how far a batch's gradient is from the true one")

    def phase_4(self, sc, u, op):
        self.learning_phase(sc, u, op)


# ---------------------------------------------------------------------- the free game, drawn
class FreeGame(Glimpse):
    """The free glimpse game (look or name at every step, six steps at most)."""

    def state(self, sc, ep, j, t, fx, fy, size, op, said=True):
        """Step ``t`` of greedy episode ``j``: the digit as covered then, the quadrant it
        looks at ringed (red: one already uncovered), and what it did underneath."""
        a = int(ep["act"][t, j])
        again = a < 4 and ep["mask"][t, j, a] == 1
        self.covered(sc, ep["image"][j], ep["mask"][t, j], fx, fy, size, op,
                     ring=a if a < 4 else None, ring_ink=RED if again else AMBER)
        if said:
            dy = self.frac(0, size / 2)[1] + 0.036
            what = "again" if again else "looks" if a < 4 else "names %d" % (a - 4)
            self.text(sc, fx, fy + dy, what, op, size=0.030,
                      ink=dark(RED) if again else None, strong=a >= 4)

    def values(self, sc, q, x0, base, tall, op, ink, chosen=None, truth=None):
        """The fourteen action values: four looks, ten names."""
        self.look_bars(sc, q[:4], x0, base, tall, op, chosen=chosen if chosen is not None
                       and chosen < 4 else None, pitch=0.026)
        self.bars(sc, np.clip(q[4:], 0, None), x0 + 0.115, x0 + 0.335, base, tall, op, ink,
                  labels=DIGITS, truth=truth, chosen=chosen - 4 if chosen is not None
                  and chosen >= 4 else None)


# ---------------------------------------------------------------------- 94
class DQNReplayBuffer(FreeGame):
    """Q-learning with a small network on the free glimpse game, its digits arriving one
    class at a time: a step and the target built from the next one, what a replay buffer
    changes about the batch, what a target network changes about the target, and five
    runs of each combination."""

    TITLE = "DQN: replay buffer and target network"
    PHASES = (("a target", 12.0), ("replay buffer", 14.0), ("target network", 12.0),
              ("five runs", 12.0))
    RUNS = (("neither", RED), ("replay only", BLUE), ("target network only", OCHRE),
            ("both", VIOLET), ("neither, digits shuffled", GREEN))

    def setup(self):
        s, d = self.study, G.dqn_runs()
        self.d = d
        self.x = np.asarray(s.panels[0].data["x"], float)
        self.right = np.asarray(s.panels[0].data["ys"], float)            # (arm, evaluation)
        self.late = np.asarray(s.panels[1].data["values"], float).reshape(5, -1)
        self.by_class = np.asarray(s.panels[2].data["values"], float)     # (20, evaluation)
        self.excess = np.asarray(s.panels[3].data["ys"], float)
        self.value = G.last_pass(d["value"]).mean((1, 2))
        self.score = G.last_pass(d["score"]).mean((1, 2))
        ep = self.ep = free_demo("dqn")
        # a held-out episode that looked at least twice and named its digit right
        self.shown = int(np.flatnonzero(ep["right"] & (ep["steps"] >= 3)
                                        & (ep["repeats"] == 0))[0])

    def subtitle_for(self, k):
        f = self.late
        return [
            "at each step the player uncovers a quadrant or names the digit. Q-learning gives "
            "every action a value, the reward it expects; an action's target is its reward "
            "plus 0.9 x the best value at the next step",
            "here the digits arrive one class at a time. The newest 32 steps are all one digit, "
            "and a network trained on them forgets the others; a replay buffer trains on 32 "
            "drawn from the last 20,000 steps",
            "the target comes from the network being trained, so it moves with every update. A "
            "target network is a copy refreshed every 200 updates: with replay it lifts "
            "held-out digits named from %s to %s" % (_pct(f[1].mean()), _pct(f[3].mean())),
            "five runs each, held-out digits named: neither %s, replay only %s, target network "
            "only %s, both %s. With the digits shuffled the plain learner manages %s" % (
                _pct(f[0].mean()), _spread(f[1]), _pct(f[2].mean()), _spread(f[3]),
                _spread(f[4])),
        ][k]

    def phase_1(self, sc, u, op):
        ep, j, t = self.ep, self.shown, 1
        a, label = int(ep["act"][t, j]), int(ep["label"][j])
        nxt = ep["q"][t + 1, j]
        cols = ((0.06, "now: one quadrant uncovered", t, AMBER, 0.0),
                (0.535, "next: after that look", t + 1, TEAL, 0.32))
        for x0, head, step, ink, at in cols:
            shown = op * smoothstep(at, at + 0.08, u)
            self.text(sc, x0, self.HEAD_Y, head, shown, anchor="lm", strong=True, size=0.032)
            self.covered(sc, ep["image"][j], ep["mask"][step, j], x0 + 0.045, self.MID_Y, 80.0,
                         shown, ring=a if step == t and u > 0.16 else None)
            self.values(sc, ep["q"][step, j], x0 + 0.10, self.BASE, self.TALL, shown, ink,
                        chosen=int(ep["q"][step, j].argmax()) if u > at + 0.14 else None,
                        truth=label)
        self.text(sc, 0.06, self.SAID_Y, "its best action: uncover this quadrant, value %.2f"
                  % ep["q"][t, j, a], op * smoothstep(0.16, 0.22, u), anchor="lm", size=0.032,
                  ink=dark(AMBER, 0.35))
        self.text(sc, 0.535, self.SAID_Y, "best value next: %.2f" % nxt.max(),
                  op * smoothstep(0.46, 0.52, u), anchor="lm", size=0.032, ink=dark(TEAL))
        self.text(sc, 0.06, 0.74, "bars: the value the network gives each action (4 looks, 10 "
                  "names)", op * smoothstep(0.2, 0.3, u), anchor="lm", size=0.030)
        self.text(sc, 0.06, 0.80, "target for that look = reward 0 + 0.9 x %.2f = %.2f" % (
            nxt.max(), 0.9 * nxt.max()), op * smoothstep(0.58, 0.65, u), anchor="lm",
            size=0.034, strong=True)
        self.text(sc, 0.06, 0.855, "its value now is %.2f: the update moves it toward the "
                  "target" % ep["q"][t, j, a], op * smoothstep(0.72, 0.79, u), anchor="lm",
                  size=0.034, ink=dark(AMBER, 0.35))
        self.text(sc, 0.94, self.NOTE_Y, "a real held-out digit (a %d), played by the trained "
                  "network" % label, op, anchor="rm", ink=AXIS)

    def phase_2(self, sc, u, op):
        d = self.d
        g = min(u / 0.85, 1.0)
        k = int(g * (len(self.x) - 1))
        dealt = (max(int(self.x[k]) - 1, 0) // G.Q["block"]) % 10
        self.text(sc, 0.06, self.HEAD_Y, "step %s: the game is dealing %ds" % (
            f"{int(self.x[k]):,}", dealt), op, anchor="lm", strong=True, size=0.032)
        rows = ((0, 0.525, "no replay: the newest 32 steps", RED),
                (3, 0.80, "replay: 32 drawn from the last 20,000", VIOLET))
        for arm, base, name, ink in rows:
            self.text(sc, 0.06, base - 0.155, name, op, anchor="lm", ink=dark(ink), size=0.032)
            self.bars(sc, d["batch_classes"][arm, 0, k], 0.06, 0.42, base, 68.0, op, ink,
                      labels=DIGITS, top=32.0)
        self.text(sc, 0.24, 0.885, "the digits in this update's training batch", op, ink=AXIS)
        # what each network can do, class by class, as training goes
        self.text(sc, 0.50, self.HEAD_Y, "held-out digits named, by class (first run)", op,
                  anchor="lm", strong=True, size=0.032)
        for half, (y, name, ink) in enumerate(((0.47, "no replay", RED),
                                               (0.745, "replay + target network", VIOLET))):
            m = self.by_class[half * 10:half * 10 + 10].copy()
            m[:, k + 1:] = 0
            cell = self.grid(sc, m, 0.735, y, 372.0, 90.0, op, vmax=1.0, hot=ink, lines=0.0)
            self.text(sc, 0.52, y - 0.125, name, op, anchor="lm", ink=dark(ink), size=0.030)
            for c in (0, 9):
                self.text(sc, 0.502, cell(c, 0)[1], str(c), op, ink=AXIS)
        self.text(sc, 0.735, 0.885, "row: a digit    across: training steps", op, ink=AXIS)

    def phase_3(self, sc, u, op):
        g = min(u / 0.85, 1.0)
        k = int(g * (len(self.x) - 1))
        int(self.x[k]) % G.Q["sync"] if g < 1 else 0
        flash = max(0.0, 1.0 - (g * 12 % 1.0) * 3)
        self.box(sc, 0.20, 0.40, 250, 50, soft(AMBER), op, "the network being trained",
                 border=AMBER, size=0.032)
        self.box(sc, 0.20, 0.64, 250, 50, soft(TEAL), op, "its copy: computes the targets",
                 border=TEAL, size=0.032)
        self.arrow(sc, (0.20, 0.455), (0.20, 0.585), AMBER, op * max(0.25, flash), width=3.4)
        self.text(sc, 0.215, 0.52, "copied every 200 updates", op, anchor="lm", size=0.030)
        self.lines(sc, 0.06, 0.75, [
            "without the copy, each update also moves",
            "the target it is chasing"], op, size=0.030, step=0.045)
        chart = self.chart(((0.50, 0.94), (0.43, 0.84)), self.x, [0.0, self.d["right"].max()])
        self.axes(sc, chart, op, "training step", "held-out digits named (thin: each run)")
        for arm in (1, 3):
            self.seeds(sc, chart, self.x, self.d["right"][arm], self.RUNS[arm][1], op, g)
        for arm, x in ((1, 0.50), (3, 0.70)):
            self.text(sc, x, self.HEAD_Y, "%s: %s" % (self.RUNS[arm][0], _pct(
                self.right[arm, k])), op, anchor="lm", ink=dark(self.RUNS[arm][1]), size=0.032)

    def phase_4(self, sc, u, op):
        self.text(sc, 0.06, self.HEAD_Y, "held-out digits named, last pass through the classes",
                  op, anchor="lm", strong=True, size=0.032)
        for row, ((name, ink), v) in enumerate(zip(self.RUNS, self.late)):
            self.seed_bars(sc, v * smoothstep(0.0, 0.3, u), 0.06, 0.395 + row * 0.098, op, ink,
                           "%s: %s" % (name, _pct(v.mean())), width=0.34)
        chart = self.chart(((0.52, 0.94), (0.36, 0.84)), self.x, [0.0, self.right.max()])
        self.axes(sc, chart, op, "training step", "held-out digits named (mean of 5 runs)")
        for (name, ink), trace in zip(self.RUNS, self.right):  # noqa: B007
            self.plot(sc, chart, self.x, trace, ink, op, u / 0.75, width=1.8)
        self.lines(sc, 0.06, 0.85, [
            "bar: mean of 5 runs    dot: one run",
            "score expected at the start, and got:",
            "neither %.2f, %.2f    both %.2f, %.2f" % (
                self.value[0], self.score[0], self.value[3], self.score[3])], op, size=0.030,
            step=0.04)


# ---------------------------------------------------------------------- 96
def extras(looks, name_right=True):
    """Exact arithmetic for one episode of the free game that makes ``looks`` (a list:
    True for a look at a new quadrant, False for a repeat) and then names the digit
    right: the game's reward, the discounted total that potential-based shaping adds, and
    the one a bonus per look adds (:data:`voxanim.study_rl.SHAPE`)."""
    g, phi, potential, bonus = G.GAMMA, 0.0, 0.0, 0.0
    for t, new in enumerate(looks):
        after = phi + G.SHAPE["potential"] * new
        potential += g ** t * (g * after - phi)
        bonus += g ** t * G.SHAPE["bonus"]
        phi = after
    potential += g ** len(looks) * (0.0 - phi)          # the name ends the episode
    return float(name_right) * g ** len(looks), potential, bonus


class RewardShaping(FreeGame):
    """The free glimpse game learned on three rewards: its own, that plus potential-based
    shaping, and that plus a bonus for every look. What each extra adds to two episodes
    (exact), an episode by each trained network, and the game's own score over five runs."""

    TITLE = "Reward shaping"
    PHASES = (("the game", 10.0), ("two extras", 14.0), ("what each learned", 13.0),
              ("the game's score", 12.0))
    RUNS = (("the game's reward only", BLUE), ("+ potential-based shaping", GREEN),
            ("+ 0.25 for every look", RED))
    PLANS = (("three new quadrants, then the name", (True, True, True)),
             ("the same quadrant five times, then the name", (True, False, False, False,
                                                             False)))

    def setup(self):
        s, d = self.study, G.shaping_runs()
        self.d = d
        self.x = np.asarray(s.panels[0].data["x"], float)
        self.score = np.asarray(s.panels[0].data["ys"], float)
        self.steps = np.asarray(s.panels[1].data["ys"], float)
        self.late = np.asarray(s.panels[2].data["values"], float).reshape(3, -1)
        self.repeats = np.asarray(s.panels[3].data["ys"], float)
        n = len(G.last_pass(self.x))
        self.late_steps = self.steps[:, -n:].mean(1)
        self.late_repeats = self.repeats[:, -n:].mean(1)
        self.reach = np.asarray(s.metadata["steps_to_reach"], float)[:2].mean(1)
        self.eps = [free_demo(k) for k in range(3)]
        # one held-out digit on which every network plays its usual number of steps
        usual = [int(np.round(np.median(ep["steps"]))) for ep in self.eps]
        ok = np.all([ep["steps"] == n for ep, n in zip(self.eps, usual)], 0) & self.eps[0][
            "right"] & (self.eps[2]["repeats"] >= 1)
        self.shown = int(np.flatnonzero(ok)[0])

    def subtitle_for(self, k):
        s, r, f = self.late_steps, self.late_repeats, self.late.mean(1)
        faster = self.reach[1] < 0.9 * self.reach[0]
        return [
            "the game's own reward: 1 for the right name, worth 0.9 times less for every step "
            "taken, and nothing after six steps. Looking helps, but it costs time",
            "two ways to encourage looking. Potential-based: pay for each newly uncovered "
            "quadrant and take it back at the end, so an episode's extra always sums to 0. Or "
            "simply 0.25 for every look",
            "five runs on each reward. Paid 0.25 per look, the learner keeps looking, also at "
            "quadrants already uncovered: %.1f steps and %.1f repeated looks per episode, "
            "against %.1f and %.1f on the game's reward" % (s[2], r[2], s[0], r[0]),
            "the game's own score: %.2f with the bonus per look, %.2f on the game's reward, "
            "%.2f with potential shaping, which kept the result %s: %s steps to reach "
            "%.2f, against %s" % (f[2], f[0], f[1], "and was faster" if faster
                                  else "but was no faster", f"{self.reach[1]:,.0f}",
                                  self.study.metadata["reach_level"],
                                  f"{self.reach[0]:,.0f}"),
        ][k]

    def episode(self, sc, ep, j, x0, y, size, op, upto=6.0, pitch=0.092):
        """Greedy episode ``j`` as a row of states; returns the x after the last."""
        steps = int(ep["steps"][j])
        for t in range(steps):
            self.state(sc, ep, j, t, x0 + t * pitch, y, size, op * smoothstep(0.0, 0.6, upto - t))
        return x0 + steps * pitch

    def phase_1(self, sc, u, op):
        ep, j = self.eps[0], self.shown
        steps = int(ep["steps"][j])
        self.text(sc, 0.06, self.HEAD_Y, "one episode: look or name at every step, six steps "
                  "at most", op, anchor="lm", strong=True, size=0.032)
        end = self.episode(sc, ep, j, 0.13, 0.47, 96.0, op, u / 0.5 * steps, pitch=0.15)
        told = op * smoothstep(0.55, 0.65, u)
        self.text(sc, min(end, 0.80), 0.47, "right: reward 0.9^%d = %.2f" % (
            steps - 1, ep["score"][j]), told, anchor="lm", size=0.034, ink=dark(GREEN))
        self.lines(sc, 0.06, 0.70, [
            "a real held-out digit (a %d), played by a network trained on this reward alone"
            % ep["label"][j],
            "ring: the quadrant it uncovers next    grey: still covered",
            "a name at the first step could earn 1, but the digit is still covered",
            "every look makes the name surer and the reward smaller"],
            told, size=0.032, step=0.055)

    def phase_2(self, sc, u, op):
        self.text(sc, 0.06, self.HEAD_Y, "what each reward pays for two episodes (exact "
                  "arithmetic)", op, anchor="lm", strong=True, size=0.032)
        for col, (name, looks) in enumerate(self.PLANS):
            x0 = 0.06 + col * 0.46
            shown = op * smoothstep(0.0, 0.1, u - 0.25 * col)
            self.text(sc, x0, 0.36, name, shown, anchor="lm", size=0.032, strong=True)
            q = 0
            for t, new in enumerate(looks):
                q = q + 1 if new and t else q
                self.icon(sc, q, x0 + 0.02 + t * 0.045, 0.425, shown, size=26.0,
                          ink=TEAL if new else RED)
            self.box(sc, x0 + 0.045 + len(looks) * 0.045, 0.425, 58, 28, soft(GREEN), shown,
                     "name", border=GREEN, size=0.030)
            game, potential, bonus = extras(looks)
            rows = ((self.RUNS[0], game, 0.0, 0.2), (self.RUNS[1], game, potential, 0.4),
                    (self.RUNS[2], game, bonus, 0.6))
            for row, ((what, ink), base, extra, at) in enumerate(rows):
                y = 0.535 + row * 0.115
                told = shown * smoothstep(at, at + 0.1, u)
                self.text(sc, x0, y - 0.045, what, told, anchor="lm", ink=dark(ink), size=0.030)
                full = self.span(0.25)[0]
                self.hbar(sc, x0, y, full * (base + extra) / 1.7, ink, told, "", "", thick=22.0,
                          track=full)
                self.text(sc, x0 + 0.26, y, "%.2f" % (base + extra) if row == 0 else
                          "%.2f %s %.2f = %.2f" % (base, "+" if extra >= 0 else "-", abs(extra),
                                                   base + extra), told, anchor="lm",
                          strong=True, size=0.030)
        self.text(sc, 0.5, 0.875, "red: a look at a quadrant that is already uncovered. Only "
                  "the bonus per look pays more for the loop", op * smoothstep(0.75, 0.85, u),
                  size=0.032)

    def phase_3(self, sc, u, op):
        j = self.shown
        self.text(sc, 0.06, self.HEAD_Y, "the same held-out digit (a %d), played by a network "
                  "trained on each reward" % self.eps[0]["label"][j], op, anchor="lm",
                  strong=True, size=0.032)
        for row, ((name, ink), ep) in enumerate(zip(self.RUNS, self.eps)):  # noqa: B007
            y = 0.42 + row * 0.185
            start = row * 0.22
            shown = op * smoothstep(start, start + 0.06, u)
            self.lines(sc, 0.06, y - 0.02, fit(name, 0.030, 0.17), shown, size=0.030,
                       step=0.04)
            steps = int(ep["steps"][j])
            end = self.episode(sc, ep, j, 0.29, y - 0.012, 58.0, shown,
                               (u - start) / 0.3 * steps, pitch=0.088)
            told = shown * smoothstep(start + 0.3, start + 0.36, u)
            right = bool(ep["right"][j])
            self.text(sc, max(end, 0.82) - 0.02, y - 0.012, "score %.2f" % ep["score"][j] if right
                      else "score 0", told, anchor="lm", size=0.034, strong=True)
        self.text(sc, 0.5, self.NOTE_Y, "one digit; over 1,000 held-out digits: %s steps per "
                  "episode" % ", ".join("%.1f" % v for v in self.late_steps), op, size=0.030)

    def phase_4(self, sc, u, op):
        g = min(u / 0.8, 1.0)
        k = int(g * (len(self.x) - 1))
        self.text(sc, 0.06, self.HEAD_Y, "the game's own score on held-out digits, step %s"
                  % f"{int(self.x[k]):,}", op, anchor="lm", strong=True, size=0.032)
        for row, (name, ink) in enumerate(self.RUNS):
            v = self.d["score"][row, :, k]
            self.seed_bars(sc, v, 0.06, 0.41 + row * 0.115, op, ink, "%s: %.2f" % (
                name, v.mean()), top=0.8)
        self.lines(sc, 0.06, 0.735, [
            "bar: mean of 5 runs    dot: one run",
            "steps per episode: %s" % ", ".join("%.1f" % v for v in self.steps[:, k]),
            "repeated looks: %s" % ", ".join("%.1f" % v for v in self.repeats[:, k])], op,
            size=0.030, step=0.045)
        chart = self.chart(((0.50, 0.94), (0.36, 0.84)), self.x, [0.0, self.d["score"].max()])
        self.axes(sc, chart, op, "training step", "score (thin: each run)")
        for (_n, ink), runs in zip(self.RUNS, self.d["score"]):
            self.seeds(sc, chart, self.x, runs, ink, op, g)


# ---------------------------------------------------------------------- 95
MOVE_NAMES = {2: "top right", 5: "middle right", 6: "bottom left", 7: "bottom middle",
              8: "bottom right"}


class MCTSTicTacToe(RL):
    """Monte Carlo tree search on a tic-tac-toe position with a win in one:
    the position and its five moves, simulations one at a time (select,
    expand, play out at random, back the result up), the rule that picks
    which move to try next, and where 501 simulations went."""

    TITLE = "Monte Carlo tree search"
    PHASES = (("the position", 9.0), ("a simulation", 15.0), ("which move next", 15.0),
              ("501 simulations", 13.0))
    ROOT = (0.5, 0.36, 78.0)
    KIDS_Y, KID = 0.595, 74.0
    ROW3_Y, SMALL, STEP = 0.81, 64.0, 0.083
    KID_X = (0.105, 0.29, 0.475, 0.66, 0.845)
    MOVE_INKS = (GREEN, VIOLET, TEAL, MAGENTA, OCHRE)

    def setup(self):
        s = self.study
        d = self.data = mcts_replay()
        self.moves = [int(m) for m in d["moves"]]
        self.counts = np.asarray(s.panels[1].data["ys"], float)
        self.values = np.asarray(s.panels[2].data["ys"], float)
        self.final = np.asarray(s.panels[3].data["values"], float)
        self.sims = self.counts.shape[1]
        self.best = int(self.final.argmax())

    def subtitle_for(self, k):
        name = MOVE_NAMES[self.moves[self.best]]
        return [
            "X to move, with five empty squares. One of them, the %s, completes a row and wins "
            "at once. The search is told only the rules, not which move is good" % name,
            "one simulation: pick a move, add one new position to the tree, play the rest out "
            "with random moves, and send the result (+1 X wins, 0 draw, -1 O wins) back up",
            "which move next: each move's average result plus a bonus that is large for moves "
            "tried little. The highest total is tried, so good moves get most visits and no "
            "move is forgotten",
            "after %d simulations the %s has %d visits and an average result of %+.2f; no "
            "other move has more than %d. The search plays the most visited move" % (
                self.sims, name, self.final[self.best], self.values[self.best, -1],
                np.delete(self.final, self.best).max()),
        ][k]

    # ------------------------------------------------------------------ drawing
    def board(self, sc, b, fx, fy, size, op, new=None, ring=None, solid=True):
        """A tic-tac-toe position ``b`` (9 values: 1 X, -1 O, 0 empty), ``size``
        px; ``new``: a square to tint (the move just made); ``ring``: a border
        colour."""
        if op <= 0.01:
            return
        c = self.P(float(fx), float(fy))
        self.rect(sc, c, size + 6, size + 6, PAPER, op, border=ring if ring is not None else (
            AXIS if solid else DIM), border_w=3.0 if ring is not None else 1.4, radius=5,
            lift=0.3)
        cell = size / 3.0
        dw, dh = self.frac(size, size)
        if new is not None:
            r, k = divmod(int(new), 3)
            self.rect(sc, self.P(fx + (k - 1) * dw / 3, fy + (r - 1) * dh / 3), cell - 2, cell - 2,
                      soft(AMBER, 0.45), op, radius=2, lift=0.4)
        for k in (1, 2):
            gx, gy = fx - dw / 2 + k * dw / 3, fy - dh / 2 + k * dh / 3
            self.seg(sc, (gx, fy - dh / 2), (gx, fy + dh / 2), AXIS, op * 0.8, 1.0, lift=0.6)
            self.seg(sc, (fx - dw / 2, gy), (fx + dw / 2, gy), AXIS, op * 0.8, 1.0, lift=0.6)
        m = 0.30
        for i, v in enumerate(b):
            if not v:
                continue
            r, k = divmod(i, 3)
            x, y = fx + (k - 1) * dw / 3, fy + (r - 1) * dh / 3
            if v > 0:
                ex, ey = dw / 3 * m, dh / 3 * m
                self.seg(sc, (x - ex, y - ey), (x + ex, y + ey), BLUE, op, max(2.2, size / 22))
                self.seg(sc, (x - ex, y + ey), (x + ex, y - ey), BLUE, op, max(2.2, size / 22))
            else:
                dia = cell * 0.62
                self.rect(sc, self.P(x, y), dia, dia, RED, op, border=RED,
                          border_w=max(2.2, size / 22), radius=dia / 2, lift=0.7, fill_op=0.0)

    def _edge(self, sc, a, b, ink, op, u=1.0, width=2.0):
        self.arc(sc, a, b, ink, op, u, 0.0, width, head=False)

    def _tree(self, sc, op, counts, values, chosen=None, select=1.0, kids=5, stats=True):
        """The root, its five moves and what is known of each."""
        rx, ry, rs = self.ROOT
        self.board(sc, START, rx, ry, rs, op)
        top = ry + self.frac(0, rs / 2 + 5)[1]
        for j, x in enumerate(self.KID_X[:kids]):
            picked = chosen == j
            kid_top = self.KIDS_Y - self.frac(0, self.KID / 2 + 5)[1]
            self._edge(sc, (rx, top), (x, kid_top), AMBER if picked else AXIS,
                       op * (1.0 if picked else 0.45), select if picked else 1.0,
                       3.2 if picked else 1.4)
            self.board(sc, self.data["kids"][j], x, self.KIDS_Y, self.KID, op,
                       new=self.moves[j], ring=AMBER if picked and select > 0.9 else None)
            if stats:
                n = int(counts[j])
                self.text(sc, x + 0.05, self.KIDS_Y - 0.024, "%d visit%s" % (
                    n, "" if n == 1 else "s"), op, anchor="lm", size=0.032, strong=True)
                self.text(sc, x + 0.05, self.KIDS_Y + 0.024, "avg %+.2f" % values[j] if n
                          else "untried", op, anchor="lm", size=0.030, ink=AXIS)

    def phase_1(self, sc, u, op):
        kids = int(np.clip(u / 0.5 * 5 + 1, 1, 5)) if u > 0.08 else 0
        self._tree(sc, op, None, None, kids=kids, stats=False)
        rx, ry, _ = self.ROOT
        self.text(sc, rx - 0.05, ry, "X to move", op, anchor="rm", strong=True, size=0.034,
                  ink=dark(BLUE))
        self.text(sc, rx + 0.05, ry, "five empty squares: five moves",
                  op * smoothstep(0.08, 0.2, u), anchor="lm", size=0.032)
        for j, x in enumerate(self.KID_X[:kids]):
            won = S.tictactoe_winner(self.data["kids"][j]) == 1
            self.text(sc, x, self.KIDS_Y + 0.11, "three in a row: X wins" if won
                      else "the game goes on", op * smoothstep(0.55, 0.7, u), size=0.030,
                      ink=dark(GREEN) if won else AXIS, strong=False)
        self.lines(sc, 0.5, 0.80, ["the search does not know this yet",
                                   "it will find out by trying moves and playing the games out "
                                   "at random"], op * smoothstep(0.7, 0.85, u), size=0.034,
                   step=0.055, anchor="mm")

    def _simulation(self, sc, n, w, op, ucb=False):
        d = self.data
        sim = d["sims"][n]
        before = self.counts[:, n - 1] if n else np.zeros(5)
        vbefore = self.values[:, n - 1] if n else np.zeros(5)
        landed = smoothstep(0.88, 0.95, w)
        counts = before if landed < 0.5 else self.counts[:, n]
        values = vbefore if landed < 0.5 else self.values[:, n]
        j = d["kids"].index(sim["path"][1])
        x = self.KID_X[j]
        self._tree(sc, op, counts, values, chosen=j, select=smoothstep(0.05, 0.22, w))
        # the rule's totals, while it chooses
        if ucb and d["ucb"][n] is not None:
            show = op * smoothstep(0.0, 0.05, w) * (1 - smoothstep(0.26, 0.32, w))
            totals = [v + b for v, b in d["ucb"][n]]
            for i, (v, b) in enumerate(d["ucb"][n]):
                top = i == int(np.argmax(totals))
                self.text(sc, self.KID_X[i] + 0.02, self.KIDS_Y + 0.115, "%+.2f + %.2f = %.2f" % (
                    v, b, v + b), show, size=0.030, strong=top, ink=None if top else AXIS)
            self.text(sc, 0.5, self.KIDS_Y + 0.18, "average result + bonus for being tried "
                      "little = total: the highest total is tried", show, size=0.032)
        # below the chosen move: the rest of the path in the tree, then the random playout
        deeper = list(sim["path"][2:])
        boards = deeper + list(sim["rollout"])
        way = 1 if x < 0.6 else -1
        px, py = x, self.KIDS_Y + self.frac(0, self.KID / 2 + 5)[1]
        last = (x, self.KIDS_Y)
        prev = sim["path"][1]
        for k, b in enumerate(boards):
            tree = k < len(deeper)
            at = 0.32 + 0.36 * k / max(len(boards), 1)
            shown = op * smoothstep(at, at + 0.07, w)
            bx = x + way * self.STEP * k
            by = self.ROW3_Y
            moved = int(np.flatnonzero(np.array(b) != np.array(prev))[0])
            self.board(sc, b, bx, by, self.SMALL, shown, new=moved, solid=tree,
                       ring=AMBER if tree and k == len(deeper) - 1 and sim["expanded"] else None)
            if k == 0:
                self._edge(sc, (px, py), (bx, by - self.frac(0, self.SMALL / 2 + 5)[1]),
                           AMBER if tree else AXIS, shown, 1.0, 2.6 if tree else 1.6)
            else:
                half = self.frac(self.SMALL / 2 + 4)[0]
                self.seg(sc, (bx - way * (self.STEP - half), by), (bx - way * half, by),
                         AMBER if tree else AXIS, shown, 2.6 if tree else 1.6)
            if tree:
                last = (bx, by)
            prev = b
        result = {1.0: "X wins: +1", 0.0: "a draw: 0", -1.0: "O wins: -1"}[sim["reward"]]
        ink = {1.0: GREEN, 0.0: AXIS, -1.0: RED}[sim["reward"]]
        told = op * smoothstep(0.70, 0.76, w)
        ex = x + way * self.STEP * max(len(boards) - 1, 0) if boards else x
        self.text(sc, min(max(ex, 0.13), 0.87), 0.91 if boards else self.KIDS_Y + 0.115, result,
                  told, size=0.034, strong=False, ink=dark(ink) if sim["reward"] else INK)
        # the result goes back up the path
        rx, ry, _ = self.ROOT
        hops = [last, (x, self.KIDS_Y), (rx, ry)] if last != (x, self.KIDS_Y) else [
            (x, self.KIDS_Y), (rx, ry)]
        for h, (a, b) in enumerate(itertools.pairwise(hops)):
            self.pulse(sc, a, b, (w - 0.78 - 0.07 * h) / 0.10, ink if sim["reward"] else INK, op,
                       r=7.0)
        stage = ("1 select a move" if w < 0.3 else "2 add a new position" if w < 0.45 and deeper
                 else "3 play out at random" if w < 0.70 and sim["rollout"]
                 else "4 send the result back up" if w >= 0.70 else "the game is already over")
        self.text(sc, 0.06, 0.30, "simulation %d" % (n + 1), op, anchor="lm", strong=True,
                  size=0.034)
        self.text(sc, 0.06, 0.345, stage, op, anchor="lm", size=0.032, ink=dark(AMBER, 0.35))

    def _run(self, sc, u, op, first, count, ucb):
        k = min(int(u * count), count - 1)
        w = u * count - k
        fade = (smoothstep(0.0, 0.04, w) if k else 1.0) * (
            1 - smoothstep(0.97, 1.0, w) if k < count - 1 else 1.0)
        self._simulation(sc, first + k, w, op * (0.35 + 0.65 * fade), ucb)

    def phase_2(self, sc, u, op):
        self._run(sc, u, op, 0, 5, False)
        self.text(sc, 0.94, 0.30, "untried moves are tried first", op, anchor="rm", ink=AXIS)

    def phase_3(self, sc, u, op):
        self._run(sc, u, op, 5, 6, True)

    def phase_4(self, sc, u, op):
        g = min(u / 0.8, 1.0)
        n = int(g * (self.sims - 1))
        full = self.span(0.24)[0]
        self.text(sc, 0.06, 0.292, "simulation %d of %d: visits of each move" % (n + 1, self.sims),
                  op, anchor="lm", strong=True, size=0.032)
        for j, (kid, ink) in enumerate(zip(self.data["kids"], self.MOVE_INKS)):
            y = 0.385 + j * 0.113
            self.board(sc, kid, 0.09, y, 46, op, new=self.moves[j],
                       ring=ink if j == self.best and g >= 1.0 else None)
            self.hbar(sc, 0.14, y, full * self.counts[j, n] / self.sims, ink, op, "",
                      "%d" % self.counts[j, n], thick=26.0, track=full)
        sims = np.arange(1, self.sims + 1)
        chart = self.chart(((0.56, 0.945), (0.40, 0.84)), sims, self.values)
        self.axes(sc, chart, op, "simulation", "average result of each move (+1: X wins)")
        for trace, ink in zip(self.values, self.MOVE_INKS):
            self.plot(sc, chart, sims, trace, ink, op, g, width=1.6)
        self.text(sc, 0.94, 0.292, "%s: %+.2f" % (MOVE_NAMES[self.moves[self.best]],
                                                 self.values[self.best, n]), op, anchor="rm",
                  ink=dark(self.MOVE_INKS[self.best]), size=0.032)


ANIMATIONS = {
    "multi-armed-bandits": MultiArmedBandits,
    "policy-gradient": PolicyGradient,
    "actor-critic": ActorCritic,
    "dqn-replay-buffer": DQNReplayBuffer,
    "mcts-tic-tac-toe": MCTSTicTacToe,
    "reward-shaping": RewardShaping,
}

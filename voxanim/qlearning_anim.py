"""Q-learning in a gridworld: the recorded run of :mod:`voxanim.gridworld`.

The world is a slab of floor tiles (6 x 8), walls a cube high, the goal
tile green and the trap red; the agent is a teal voxel that moves cell by
cell.  The Q-table is drawn *on* the world: each cell carries four small
columns of voxels, one at each side, whose height is ``Q(s, a)`` for the
action that leaves through that side (orange above zero, blue below), so the
table fills in as a height-field and the tallest column of a cell is its
greedy action.  A panel on the right keeps the episode counter, epsilon and
the update rule with the numbers of the move just made.

Five phases:

1. the world -- tiles, walls, the start, the rewards,
2. episode 1 -- every move random (epsilon = 1); the update rule is applied
   after each move and changes nothing until the agent walks into the trap:
   ``Q <- 0 + 0.5 (-1 + 0.9 x 0 - 0) = -0.5``, the first entry of the table,
3. learning -- episodes 2 to 400 fast-forwarded with the counter: each
   episode's path flashes, the columns rise one cell further back from the
   rewards at every visit, epsilon decays; the goal is first reached in
   episode 41,
4. policy -- the greedy action in each cell as an arrow on the tallest
   column; from episode 102 on it is a shortest path,
5. greedy path -- the agent follows the arrows: 14 moves, round the trap.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from . import color as C
from . import layout
from .conv import Tempo
from .easing import ease_in_out, remap, smoothstep
from .gridworld import ACTION_NAMES, ACTIONS, ALPHA, EPS_MIN, GAMMA, epsilon
from .netanim import NetAnimation, div_colors
from .tensor import Block
from .text import MINUS, TIMES, num

PPU = 40.0                       # pixels per world unit
GRID_AT = (0.31, 0.63)           # layout fraction of the grid's centre
PITCH = 1.2                      # tile pitch (a gap of 0.2 between tiles)
COL_SIZE = 0.36                  # the Q columns' voxel size
COL_OFF = 0.30                   # their offset from the tile centre, towards the action's side
HMAX = 8                         # voxels of column for |Q| = 1
Q_VISIBLE = 0.005                # |Q| above this gets at least one voxel
AGENT = C.INPUT
AGENT_SIZE = 0.7
FLOOR = C.rgb("#c4c8cc")
WALL = C.rgb("#4a4e52")
GOAL = C.FILTERS[3]              # green
TRAP = C.FILTERS[0]              # red
TRAIL = (0.30, 0.62, 0.62)
ARROW = (0.97, 0.97, 0.95)
FRAME = (0.96, 0.96, 0.96)
PANEL_X = 0.60                   # frame x of the right-hand panel
PANEL_Y = 0.30                   # its first line
LINE = 0.046                     # line spacing in the panel
FF_SLOW, FF_FAST, FF_RAMP = 2.6, 0.025, 2.0     # fast-forward: seconds per episode, e-folding
FF_NUMBERS = 0.6                 # the per-move numbers are shown while an episode lasts this long
WALK_STEP = 0.3                  # seconds per move of the final greedy walk


_SUPERSCRIPTS = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")


def fmt(v):
    """A Q value with two decimals and a typographic minus."""
    return num(v, 2)


def sup(n):
    """Superscript digits for ``n``: ``sup(13) == "¹³"``."""
    return str(n).translate(_SUPERSCRIPTS)


class QLearning(NetAnimation):
    """The recorded :class:`voxanim.gridworld.Run` as an animation."""

    PHASES = ("the world", "episode 1", "learning", "policy", "greedy path")

    def __init__(self, run):
        super().__init__("Q-learning in a gridworld", "", PPU, azimuth=33.0, elevation=36.0)
        self.run, self.w = run, run.world
        w = self.w
        self.hold = 2.5
        # the tiles: one block, columns along x, rows along y (the back row is row 0)
        self.tiles = Block((1, w.w, w.h), color=FLOOR, pitch=(1, PITCH), name="tiles")
        self.tiles.group, self.tiles.gap = 1, PITCH - 1.0           # rows spaced like columns
        self.tiles.place(self.cam.frac(*GRID_AT))
        self.blocks_extra.append(self.tiles)
        for r in range(w.h):
            for c in range(w.w):
                if w.walls[r, c]:
                    self.tiles.base[0, c, r] = WALL
        self.tiles.base[0, w.goal[1], w.goal[0]] = GOAL
        self.tiles.base[0, w.penalty[1], w.penalty[0]] = TRAP
        # the story's numbers
        self.first_goal = int(np.argmax(run.returns > 0)) + 1
        self.converged = self._converged_from()
        self.greedy = w.greedy_path(run.Q[-1])
        self.shortest = len(w.shortest_path()) - 1
        self.v_start = float(run.Q[-1][w.start].max())
        self.first_move = ACTION_NAMES[int(np.argmax(run.Q[-1][w.start]))]
        self.ep1 = run.episode(0)

        # ---- timeline -----------------------------------------------------------------
        Tm, t = {}, 0.0

        def add(name, dur, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + dur)
            t += dur + gap

        add("world", 4.5)
        n1 = len(self.ep1)
        d1 = Tempo(slow=0.9, fast=0.35, ramp_in=4.0, ramp_out=1.5).durations(n1)
        self.s1_starts, self.s1_durs = t + np.concatenate([[0.0], np.cumsum(d1)[:-1]]), d1
        add("episode1", float(d1.sum()) + 1.8, 0.3)       # + a pause on the first update
        E = run.episodes
        k = np.arange(1, E)                                 # episodes 2 .. E (index 1 ..)
        dff = FF_FAST + (FF_SLOW - FF_FAST) * np.exp(-(k - 1) / FF_RAMP)
        self.ff_starts, self.ff_durs = t + np.concatenate([[0.0], np.cumsum(dff)[:-1]]), dff
        add("learning", float(dff.sum()), 0.5)
        add("policy", 3.5)
        add("walk", WALK_STEP * (len(self.greedy) - 1) + 0.6, 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["episode1"][0], Tm["learning"][0], Tm["policy"][0],
                      Tm["walk"][0]]

    def _converged_from(self):
        """The first episode (1-based) from which the greedy path stays a shortest path."""
        w, run = self.w, self.run
        n = len(w.shortest_path())
        ok = [(p[-1] == w.goal and len(p) == n)
              for p in (w.greedy_path(run.Q[k + 1]) for k in range(run.episodes))]
        k = run.episodes
        while k > 0 and ok[k - 1]:
            k -= 1
        return k + 1

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    # -- geometry -----------------------------------------------------------------------
    def tile_top(self, s):
        """World point at the centre of the top face of cell ``s = (row, col)``."""
        c = self.tiles.cell_center(0, s[1], s[0])
        return c + np.array([0.0, 0.0, 0.5])

    def column_base(self, s, a):
        """World centre of the lowest voxel of the ``Q(s, a)`` column."""
        dr, dc = ACTIONS[a]
        return self.tile_top(s) + np.array([COL_OFF * dc, COL_OFF * dr, COL_SIZE / 2])

    def voxels_for(self, q):
        """How many voxels a Q value is drawn as."""
        if abs(q) <= Q_VISIBLE:
            return 0
        return max(1, int(round(abs(q) * HMAX)))

    def over(self, s, Q):
        """World point above cell ``s``: its tile top, raised over its tallest Q column."""
        h = max(self.voxels_for(q) for q in Q[s]) * COL_SIZE
        return self.tile_top(s) + np.array([0.0, 0.0, h])

    # -- where the agent is ----------------------------------------------------------------
    def _state(self, t):
        """The run's state at time ``t``: a dict with ``Q`` (the table as it
        stands), ``episode`` (1-based, shown in the counter), ``pos`` (the
        agent's world point, or None), ``path`` (cells of the episode so far),
        ``step`` (the last :class:`~voxanim.gridworld.Step` applied, or None),
        ``eps``, ``fresh`` (0..1: how recently the last update landed),
        ``trail`` (the path's opacity) and ``numbers`` (the per-move numbers'
        opacity: they go when the episodes get too fast to read)."""
        run, w = self.run, self.w
        st = dict(Q=run.Q[0], episode=1, pos=None, path=[], step=None, eps=epsilon(0), fresh=0.0,
                  trail=1.0, numbers=1.0)
        if t < self.Tm["episode1"][0]:
            st["pos"] = self.over(w.start, st["Q"])
            return st
        if t < self.Tm["learning"][0]:
            return self._episode_state(t, 0, self.s1_starts, self.s1_durs, st)
        if t < self.Tm["policy"][0] - 0.25:
            k = int(np.searchsorted(self.ff_starts, t, side="right") - 1)
            k = min(max(k, 0), run.episodes - 2) + 1          # episode index 1 .. E-1
            steps = run.episode(k)
            u = (t - self.ff_starts[k - 1]) / self.ff_durs[k - 1]
            each = self.ff_durs[k - 1] / len(steps)
            starts = self.ff_starts[k - 1] + np.arange(len(steps)) * each
            durs = np.full(len(steps), each)
            st = self._episode_state(t, k, starts, durs, st)
            st["trail"] = 1.0 - smoothstep(0.85, 1.0, u)     # the path flashes, then fades
            st["numbers"] = smoothstep(FF_NUMBERS * 0.7, FF_NUMBERS, float(self.ff_durs[k - 1]))
            return st
        # the final table; the agent rests at the start, then walks the greedy path
        st["Q"], st["episode"], st["eps"] = run.Q[-1], run.episodes, epsilon(run.episodes - 1)
        u = self.progress("walk", t)
        moves = len(self.greedy) - 1
        x = u * (WALK_STEP * moves + 0.6) / WALK_STEP          # moves done, fractional
        i = min(int(x), moves - 1)
        st["path"] = self.greedy[:i + 1]
        Q = run.Q[-1]
        a, b = self.over(self.greedy[i], Q), self.over(self.greedy[min(i + 1, moves)], Q)
        st["pos"] = a + (b - a) * ease_in_out(min(x - i, 1.0)) if u > 0 else self.over(w.start, Q)
        return st

    def _episode_state(self, t, k, starts, durs, st):
        """State during episode index ``k`` whose moves start at ``starts``
        and last ``durs`` seconds (see :meth:`_state`)."""
        run = self.run
        steps = run.episode(k)
        i = int(np.searchsorted(starts, t, side="right") - 1)
        i = min(max(i, 0), len(steps) - 1)
        u = (t - starts[i]) / durs[i]
        Q = run.Q[k].copy()
        done = i + 1 if u >= 0.5 else i                    # the update lands mid-move
        for s_ in steps[:done]:
            Q[s_.s][s_.a] = s_.q_after
        a, b = self.over(steps[i].s, Q), self.over(steps[i].s2, Q)
        st.update(Q=Q, episode=k + 1, eps=epsilon(k), path=[s_.s for s_ in steps[:i + 1]],
                  pos=a + (b - a) * ease_in_out(min(u, 1.0)),
                  step=steps[done - 1] if done > 0 else None,
                  fresh=1.0 - smoothstep(0.5, 1.0, u) if done == i + 1 else 0.0)
        if u >= 1.0:                                        # the pause after the last move
            st["pos"] = b
            st["fresh"] = 0.0
        return st

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        run = self.run
        ph = 0 if reset > 0 else self.step_of(t)
        self.subtitle = [
            "the world: 6 %s 8 cells and walls; the goal pays +1, the trap %s1, both end the "
            "episode; every other move pays 0" % (TIMES, MINUS),
            "episode 1: ε = 1, every move is random; Q is all 0, so the update changes nothing "
            "until a reward arrives",
            "episodes 2 to %d: ε decays to %.2f; every visit carries a reward one cell further "
            "back; the goal is first reached in episode %d" % (run.episodes, EPS_MIN,
                                                               self.first_goal),
            "greedy policy: the arrow in each cell is the action with the largest Q; from "
            "episode %d on it is a shortest path" % self.converged,
            "the greedy path from the start: %d moves, the shortest, clear of the trap; "
            "Q(start, %s) = %.2f = %.1f%s" % (self.shortest, self.first_move, self.v_start, GAMMA,
                                              sup(self.shortest - 1)),
        ][ph]
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        world = self.progress("world", t)
        st = self._state(t)
        if reset > 0:                                      # the loop returns to the empty table
            st = dict(st, Q=run.Q[0] + (st["Q"] - run.Q[0]) * back, trail=back)
        self._tiles(sc, world, back)
        self._columns(sc, st, back)
        policy = smoothstep(0.0, 0.6, self.progress("policy", t)) * back
        if policy > 0:
            self._arrows(sc, st["Q"], policy)
        self._trail(sc, st, back)
        self._agent(sc, st, back * smoothstep(0.55, 0.75, world))
        self._panel(sc, t, st, world, back)
        self.step_indicator(sc, ph, self.PHASES)

    # -- the world ---------------------------------------------------------------------------
    def _tiles(self, sc, world, back):
        """The floor tiles (appearing row by row), the walls and the start frame."""
        w, b = self.w, self.tiles
        rows = np.arange(w.h)
        b.present[0, :, :] = (rows < world * (w.h + 2) * 1.0)[None, :]
        b.opacity *= back
        b.glow[0, w.goal[1], w.goal[0]] = 0.25
        for r in range(w.h):
            for c in range(w.w):
                if w.walls[r, c] and b.present[0, c, r]:
                    sc.voxel(self.tile_top((r, c)) + np.array([0, 0, 0.5]), WALL, back)
        if b.present[0, w.start[1], w.start[0]]:
            lo, hi = b.box((0, 1), (w.start[1], w.start[1] + 1), (w.start[0], w.start[0] + 1))
            sc.wire_box(lo, hi, inflate=0.05, color=FRAME, width=1.3, opacity=0.9 * back, bias=0.1)
            self.caption(sc, (lo, hi), "start", back * smoothstep(0.6, 0.9, world), size=0.027)

    def _columns(self, sc, st, back):
        """The Q-table: four columns per cell, ``Q(s, a)`` voxels high each."""
        w, Q = self.w, st["Q"]
        vmax = max(1.0, float(np.abs(Q).max()))
        fresh = st["fresh"]
        last = st["step"]
        for r in range(w.h):
            for c in range(w.w):
                if w.walls[r, c] or w.terminal((r, c)):
                    continue
                for a in range(len(ACTIONS)):
                    q = float(Q[r, c, a])
                    n = self.voxels_for(q)
                    if n == 0:
                        continue
                    col = div_colors([q], vmax)[0]
                    base = self.column_base((r, c), a)
                    hot = fresh if last is not None and (r, c) == last.s and a == last.a else 0.0
                    for i in range(n):
                        p = base + np.array([0.0, 0.0, i * COL_SIZE])
                        sc.voxel(p, col, back, white=0.6 * hot, glow=0.3 * hot, size=COL_SIZE)

    def _arrows(self, sc, Q, op):
        """The greedy action of every cell that has a value, as an arrow on its tallest column."""
        w = self.w
        for r in range(w.h):
            for c in range(w.w):
                if w.walls[r, c] or w.terminal((r, c)) or Q[r, c].max() <= Q_VISIBLE:
                    continue
                a = int(np.argmax(Q[r, c]))
                dr, dc = ACTIONS[a]
                d = np.array([dc, dr, 0.0])
                side = np.array([dr, -dc, 0.0])
                top = self.column_base((r, c), a) + np.array(
                    [0.0, 0.0, (self.voxels_for(Q[r, c, a]) - 0.5) * COL_SIZE + 0.08])
                tail = top - d * 0.6
                head = top + d * 0.2
                kw = dict(width=2.6, color=ARROW, opacity=op, bias=0.2)
                sc.line(tail, head, **kw)
                sc.line(head, head - d * 0.26 + side * 0.22, **kw)
                sc.line(head, head - d * 0.26 - side * 0.22, **kw)

    def _trail(self, sc, st, back):
        """The current episode's path so far, as a line over the tiles."""
        path = st["path"]
        op = 0.85 * back * st["trail"]
        if len(path) < 1 or op <= 0.01:
            return
        lift = np.array([0.0, 0.0, 0.12])
        pts = [self.over(s, st["Q"]) + lift for s in path]
        if st["pos"] is not None:
            pts.append(np.array(st["pos"]) + lift)
        for a, b in pairwise(pts):
            if np.any(a != b):
                sc.line(a, b, width=1.6, color=TRAIL, opacity=op, bias=0.1)

    def _agent(self, sc, st, op):
        """The agent: a teal voxel hovering over its cell (and its Q columns)."""
        if st["pos"] is None or op <= 0.01:
            return
        p = np.array(st["pos"]) + np.array([0.0, 0.0, AGENT_SIZE / 2 + 0.2])
        sc.voxel(p, AGENT, op, glow=0.3, size=AGENT_SIZE)
        sc.probe(("agent",), p, tol=3.0)

    # -- the panel ----------------------------------------------------------------------------
    def _panel(self, sc, t, st, world, back):
        """Counter, epsilon, the rules and the update with its numbers, right of the grid."""
        op = back * self._cap
        if op <= 0.01:
            return
        x, y = PANEL_X, PANEL_Y
        run = self.run
        lines = []
        if t < self.Tm["episode1"][0]:
            o = smoothstep(0.3, 0.7, world)
            lines += [("4 actions: up, right, down, left", 0.027, layout.CAPTION_INK, o),
                      ("goal (green): +1      trap (red): %s1" % MINUS, 0.027,
                       layout.CAPTION_INK, o),
                      ("learning rate α = %g, discount γ = %g" % (ALPHA, GAMMA), 0.027,
                       layout.CAPTION_INK, o),
                      ("Q(s, a) = 0 for every cell and action", 0.027, layout.CAPTION_INK,
                       smoothstep(0.7, 1.0, world))]
        else:
            walking = t >= self.Tm["policy"][0] - 0.25
            ep = st["episode"]
            head = ("episode %d of %d" % (ep, run.episodes)) if not walking else \
                "after %d episodes" % run.episodes
            lines.append((head, 0.031, layout.STRONG_INK, 1.0))
            lines.append(("ε = %.2f" % st["eps"] + ("  (every move random)" if st["eps"] >= 1
                                                      else ""), 0.027, layout.CAPTION_INK, 1.0))
            if not walking:
                goals = int((run.returns[:ep - 1] > 0).sum())
                traps = int((run.returns[:ep - 1] < 0).sum())
                lines.append(("so far: goal %d %s, trap %d %s" % (
                    goals, "time" if goals == 1 else "times", traps,
                    "time" if traps == 1 else "times"), 0.027, layout.CAPTION_INK, 1.0))
                lines.append(("", 0.027, layout.CAPTION_INK, 1.0))
                lines.append(("Q(s, a) ← Q(s, a) + α (r + γ max Q(s′, ·) %s Q(s, a))" % MINUS,
                              0.027, layout.CAPTION_INK, 1.0))
                s_ = st["step"]
                if s_ is not None and st["numbers"] > 0.01:   # too fast to read: left out
                    lines.append(("move %s: r = %s" % (ACTION_NAMES[s_.a], fmt(s_.r)
                                                        if s_.r else "0"),
                                  0.027, layout.CAPTION_INK, st["numbers"]))
                    lines.append(("%s + %g (%s + %g %s %s %s %s) = %s" % (
                        fmt(s_.q_before), ALPHA, fmt(s_.r) if s_.r else "0", GAMMA, TIMES,
                        fmt(s_.max_next), MINUS, fmt(s_.q_before), fmt(s_.q_after)),
                        0.029, layout.STRONG_INK if s_.q_after != s_.q_before else
                        layout.NOTE_INK, st["numbers"]))
            else:
                lines.append(("the height of a column is Q(s, a)", 0.027, layout.CAPTION_INK,
                              1.0))
                lines.append(("the arrow is the largest of the four", 0.027,
                              layout.CAPTION_INK, 1.0))
        for n, (text, size, ink, o) in enumerate(lines):
            if text:
                sc.text(x, y + n * LINE, text, size, "regular", ink, op * o, "lm")

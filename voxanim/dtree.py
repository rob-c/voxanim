"""A decision tree (CART), grown on a small 2-D dataset, then used.

    feature space (the data, the splits, the regions)     the tree

1. **data** -- 120 points, two features x1 and x2, three classes,
2. **best split** -- at the root, every threshold on x1 and then on x2 is
   tried; for each, the Gini impurity of the two halves (weighted by size)
   is plotted along the axis.  The lowest wins and becomes the root's
   question, "x1 <= t?",
3. **grow** -- the same search inside each half, until a node is pure, too
   small, or three questions deep,
4. **regions** -- each leaf predicts its majority class: the tree cuts the
   plane into axis-aligned boxes (a staircase where the true boundary is
   diagonal),
5. **predict** -- new points fall down the tree, one question at a time.

Everything is computed here, in numpy (:func:`grow`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache

import numpy as np

from . import color as C
from . import layout
from .diagram import EDGE, Diagram
from .easing import remap, smoothstep
from .text import num, sub

# ------------------------------------------------------------------ the data and the tree

CLASS_COLORS = [C.rgb("#e0962a"), C.rgb("#3f7fd0"), C.rgb("#3fa66a")]
CLASS_NAMES = ("A", "B", "C")
N, SEED = 120, 6                            # training points; dataset seed
MAX_DEPTH, MIN_LEAF, MIN_GAIN = 3, 5, 0.015 # stopping rules (MIN_GAIN: weighted Gini drop)
NEW_POINTS = [(0.62, 0.72), (0.27, 0.13)]   # the points predicted at the end
NEW_SIDE = (1, -1)                          # which side of each new point its label goes


def dataset(seed=SEED, n=N):
    """``(X (n, 2) in [0, 1], y (n,) in {0, 1, 2})``.

    Class 2 lies below a diagonal line, the rest is split between class 0
    (left) and class 1 (right) by a slanted boundary; the points are then
    jittered, so the classes overlap a little near the boundaries.
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.04, 0.96, (n, 2))
    y = np.where(X[:, 1] < 0.18 + 0.42 * X[:, 0], 2,
                 np.where(X[:, 0] < 0.45 + 0.25 * (X[:, 1] - 0.5), 0, 1))
    X = np.clip(X + rng.normal(0, 0.035, X.shape), 0.02, 0.98)
    return X, y


def gini(y, k=3):
    """Gini impurity 1 - sum p_c^2 of the labels ``y`` (0 for an empty set)."""
    if len(y) == 0:
        return 0.0
    p = np.bincount(y, minlength=k) / len(y)
    return float(1 - (p ** 2).sum())


@dataclass
class Node:
    """One node of the tree.  ``path`` is its route from the root ("" for the
    root, then "L"/"R" per question), which also identifies it."""

    path: str
    idx: np.ndarray                         # the training points that reach it
    counts: np.ndarray                      # points per class
    gini: float
    box: tuple                              # its region: (x1lo, x1hi, x2lo, x2hi)
    feature: int = -1                       # the question, "x[feature] <= threshold?"
    threshold: float = 0.0                  # (feature -1: a leaf)
    split_gini: float = 0.0                 # weighted Gini of the two children
    sweeps: list = field(default_factory=list)   # per feature: (thresholds, weighted gini)
    children: list = field(default_factory=list)

    @property
    def leaf(self):
        """True when the node asks no question."""
        return not self.children

    @property
    def depth(self):
        """Number of questions above it."""
        return len(self.path)

    @property
    def majority(self):
        """The class it predicts: the most common one among its points."""
        return int(np.argmax(self.counts))


def _sweep(X, y, f):
    """Every candidate threshold on feature ``f`` (midpoints between distinct
    values) and the weighted Gini of the split it makes; NaN where a side
    would have fewer than MIN_LEAF points."""
    xs = np.unique(X[:, f])
    th = (xs[1:] + xs[:-1]) / 2
    g = []
    for t in th:
        m = X[:, f] <= t
        nl, nr = int(m.sum()), int((~m).sum())
        if nl < MIN_LEAF or nr < MIN_LEAF:
            g.append(np.nan)
        else:
            g.append((nl * gini(y[m]) + nr * gini(y[~m])) / len(y))
    return th, np.array(g)


def grow(X, y):
    """CART on Gini impurity.  Returns ``(root, order)``: the tree, and its
    internal nodes in the order they are split (breadth first)."""
    order = []

    def build(idx, path, box):
        Xi, yi = X[idx], y[idx]
        node = Node(path, idx, np.bincount(yi, minlength=3), gini(yi), box)
        if node.gini > 0 and len(path) < MAX_DEPTH:
            node.sweeps = [_sweep(Xi, yi, f) for f in (0, 1)]
            best = None
            for f, (th, g) in enumerate(node.sweeps):
                if np.all(np.isnan(g)):
                    continue
                k = int(np.nanargmin(g))
                if best is None or g[k] < best[2] - 1e-12:
                    best = (f, th[k], g[k])
            if best is not None and (node.gini - best[2]) * len(idx) / len(y) >= MIN_GAIN:
                node.feature, node.threshold, node.split_gini = best
        return node

    root = build(np.arange(len(y)), "", (0.0, 1.0, 0.0, 1.0))
    queue = [root]
    while queue:
        node = queue.pop(0)
        if node.feature < 0:
            continue
        order.append(node)
        f, t = node.feature, node.threshold
        m = X[node.idx, f] <= t
        lo, hi = list(node.box), list(node.box)
        lo[2 * f + 1] = t
        hi[2 * f] = t
        node.children = [build(node.idx[m], node.path + "L", tuple(lo)),
                         build(node.idx[~m], node.path + "R", tuple(hi))]
        queue += node.children
    return root, order


def walk(root, p):
    """The nodes point ``p`` passes through, root to leaf."""
    out = [root]
    while not out[-1].leaf:
        n = out[-1]
        out.append(n.children[0 if p[n.feature] <= n.threshold else 1])
    return out


def all_nodes(root):
    """Every node, depth first, left before right."""
    out, stack = [], [root]
    while stack:
        n = stack.pop()
        out.append(n)
        stack += n.children[::-1]
    return out


@cache
def model():
    """``(X, y, root, order)`` for the default dataset."""
    X, y = dataset()
    root, order = grow(X, y)
    return X, y, root, order


# ------------------------------------------------------------------ layout

PLOT_C, PLOT_PX = (0.22, 0.63), 270.0       # the feature-space plot: centre, size (px)
TREE_X, TREE_Y = (0.50, 0.945), (0.33, 0.50, 0.67, 0.84)   # leaf span; y per depth
BOX_W, BOX_H = 86.0, 56.0                   # tree node box (px)
STRIP = 44.0                                # px: the Gini curves beside the plot
FOCUS = (0.12, 0.36, 0.66)                  # the node / threshold / point in play
SPLIT_INK = (0.25, 0.27, 0.29)
GREY_EDGE = (0.55, 0.57, 0.59)
SEARCH = (3.6, 3.0, 1.6)                    # root: sweep x1, sweep x2, settle (s)
SEARCH_FAST = (1.3, 1.3, 0.9)               # every later node
PREDICT_S, PREDICT_GAP = 6.0, 0.4           # each new point's descent, and the pause after


@dataclass
class Frame:
    """What one frame shows (see :meth:`DecisionTree._frame`)."""

    back: float                 # 1, fading to 0 in the loop reset
    intro: float
    done: set                   # paths of the nodes already split
    active: tuple | None        # (node, x1 sweep, x2 sweep, settle) progress, or None
    regions: float              # the leaves' colours fading in
    pk: int                     # which new point is falling (-1: none yet)
    pu: float                   # progress of its descent
    path: list                  # the nodes it passes through ([] if none)
    step: int                   # how far down that path it is (-1 if none)

    @property
    def pred(self):
        """The new point on show, or None."""
        return np.array(NEW_POINTS[self.pk]) if self.pk >= 0 else None

    @property
    def here(self):
        """The node the new point has reached, or the node being searched."""
        if self.path:
            return self.path[self.step]
        return self.active[0] if self.active else None


class DecisionTree(Diagram):
    """The tree grows split by split beside the feature space it carves up."""

    PHASES = ("data", "best split", "grow", "regions", "predict")

    def __init__(self):
        super().__init__("Growing a decision tree", "")
        self.X, self.y, self.root, self.order = model()
        self.nodes = all_nodes(self.root)
        leaves = [n for n in self.nodes if n.leaf]
        self.plot = self.P(*PLOT_C)
        self.pos = {k: self.P(*v) for k, v in self._tree_layout(leaves).items()}
        self._timeline()
        self.hold, self.reset = 3.0, 1.2
        self.finish(self.Tm["predict"][1])
        self.acc = sum(int(n.counts.max()) for n in leaves) / len(self.y)

    def _tree_layout(self, leaves):
        """Leaves evenly spaced in order, every parent centred over its children."""
        at = {}
        for k, n in enumerate(leaves):
            fx = TREE_X[0] + (TREE_X[1] - TREE_X[0]) * k / max(1, len(leaves) - 1)
            at[n.path] = [fx, TREE_Y[n.depth]]
        for n in sorted(self.nodes, key=lambda n: -n.depth):
            if not n.leaf:
                xs = [at[c.path][0] for c in n.children]
                at[n.path] = [sum(xs) / 2, TREE_Y[n.depth]]
        return at

    def _timeline(self):
        """``self.Tm`` (beats), ``self.split_t`` (each split's sweep x1 / sweep
        x2 / settle boundaries) and ``self.pred_t`` (each new point's descent)."""
        T = dict(intro=(0.2, 1.2), points=(1.2, 2.8))
        t = 3.0
        self.split_t = {}
        for k, n in enumerate(self.order):
            d = SEARCH if k == 0 else SEARCH_FAST
            self.split_t[n.path] = (t, t + d[0], t + d[0] + d[1], t + sum(d))
            t += sum(d) + (0.6 if k == 0 else 0.3)
        T["grow"] = (self.split_t[self.order[1].path][0] if len(self.order) > 1 else t, t)
        T["regions"] = (t + 0.2, t + 2.6)
        t += 3.0
        self.pred_t = []
        for _ in NEW_POINTS:
            self.pred_t.append((t, t + PREDICT_S))
            t += PREDICT_S + PREDICT_GAP
        T["predict"] = (self.pred_t[0][0], t)
        self.Tm = T


    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        for k, name in ((4, "predict"), (3, "regions"), (2, "grow")):
            if t >= self.Tm[name][0]:
                return k
        return 1 if t >= self.split_t[""][0] else 0

    # ------------------------------------------------------------------ geometry and state
    def pp(self, x1, x2, lift=0.0):
        """Feature space (0..1, 0..1) to a world point on the plot."""
        return self.shift(self.plot, (x1 - 0.5) * PLOT_PX, (0.5 - x2) * PLOT_PX) + \
            self.view * lift

    def _frame(self, t, reset):
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        done, active = set(), None
        for n in self.order:
            t0, t1, t2, t3 = self.split_t[n.path]
            if t >= t3:
                done.add(n.path)
            elif t >= t0:
                active = (n, remap(t, t0, t1), remap(t, t1, t2), remap(t, t2, t3))
        pk, pu = -1, 0.0
        for k, (a, b) in enumerate(self.pred_t):
            if a <= t:
                pk, pu = k, remap(t, a, b)
        path, step = [], -1
        if pk >= 0:
            path = walk(self.root, np.array(NEW_POINTS[pk]))
            step = min(len(path) - 1, int(pu * 1.25 * len(path)))
        return Frame(back, smoothstep(0.0, 1.0, self.progress("intro", t)) * back, done, active,
                     smoothstep(0.0, 1.0, self.progress("regions", t)) * back, pk, pu, path, step)

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        f = self._frame(t, reset)
        self.rect(sc, self.plot, PLOT_PX + 2, PLOT_PX + 2, (0.975, 0.975, 0.97), f.intro,
                  border=GREY_EDGE, border_w=1.0, radius=2, lift=0.0)
        if f.regions > 0:
            self._regions(sc, f.regions)
        self._splits(sc, f)
        self._points(sc, t, f)
        if f.active:
            self._search(sc, *f.active, f.back)
        if f.pk >= 0:
            self._predict(sc, f.pred, f.pu, f.back, NEW_SIDE[f.pk])
        self._tree(sc, t, f)
        self._axes(sc, f.intro)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, f, reset)

    def _splits(self, sc, f):
        """The splits made so far as lines across their regions, and an outline
        around the region in play."""
        active = f.active
        for n in self.order:
            if n.path in f.done or (active and active[0] is n and active[3] > 0):
                lock = 1.0 if n.path in f.done else smoothstep(0.0, 0.5, active[3])
                a, b = self._split_line(n, n.threshold)
                sc.line(a, b, width=2.2, color=SPLIT_INK, opacity=0.9 * lock * f.back,
                        bias=0.3)
        here = f.here
        if here is not None:
            x1a, x1b, x2a, x2b = here.box
            c = self.pp((x1a + x1b) / 2, (x2a + x2b) / 2, 0.2)
            self.rect(sc, c, (x1b - x1a) * PLOT_PX, (x2b - x2a) * PLOT_PX, (1.0, 1.0, 1.0),
                      0.9 * f.back, border=FOCUS, border_w=1.6, radius=1, lift=0.15,
                      fill_op=0.0)

    def _points(self, sc, t, f):
        """The training points, popping in; those outside the region in play dim."""
        pop = self.progress("points", t)
        here = f.here
        inside = set(here.idx.tolist()) if here is not None else None
        n = len(self.y)
        for i, (p, k) in enumerate(zip(self.X, self.y)):
            on = smoothstep(0.0, 0.2, remap(pop, 0.6 * i / n, 0.6 * i / n + 0.4)) * f.back
            dim = 1.0 if inside is None or i in inside else 0.3
            self.disk(sc, self.pp(*p), 3.6, CLASS_COLORS[k], on * dim, lift=0.8, outline=0.3)

    def _split_line(self, n, th, f=None):
        """End points of the line ``x[f] = th`` across node ``n``'s region."""
        f = n.feature if f is None else f
        x1a, x1b, x2a, x2b = n.box
        if f == 0:
            return self.pp(th, x2a, 0.5), self.pp(th, x2b, 0.5)
        return self.pp(x1a, th, 0.5), self.pp(x1b, th, 0.5)

    def _regions(self, sc, op):
        """Each leaf's region tinted with the class it predicts."""
        for n in (n for n in self.nodes if n.leaf):
            x1a, x1b, x2a, x2b = n.box
            c = self.pp((x1a + x1b) / 2, (x2a + x2b) / 2, 0.1)
            col = C.mix(CLASS_COLORS[n.majority], (1, 1, 1), 0.55)
            self.rect(sc, c, (x1b - x1a) * PLOT_PX, (x2b - x2a) * PLOT_PX, col, op * 0.75,
                      radius=0.5, lift=0.1)

    # ------------------------------------------------------------------ the threshold search
    def _curve_point(self, f, th, g, glo, ghi):
        """A point of feature ``f``'s Gini curve: along the plot's top edge (x1)
        or its right edge (x2), further out for higher Gini."""
        k = (g - glo) / max(ghi - glo, 1e-9)
        if f == 0:
            return self.pp(th, 1.0, 0.3) + self.Mi @ np.array([0.0, -6 - STRIP * k])
        return self.pp(1.0, th, 0.3) + self.Mi @ np.array([6 + STRIP * k, 0.0])

    def _strip(self, sc, f, fade):
        """The baseline of feature ``f``'s Gini curve and its label."""
        if f == 0:
            a, b = (q + self.Mi @ np.array([0.0, -6.0])
                    for q in (self.pp(0, 1.0, 0.2), self.pp(1, 1.0, 0.2)))
        else:
            a, b = (q + self.Mi @ np.array([6.0, 0.0])
                    for q in (self.pp(1.0, 0, 0.2), self.pp(1.0, 1, 0.2)))
        sc.line(a, b, width=1.0, color=EDGE, opacity=0.4 * fade, bias=0.0)
        if f == 0:
            self.label(sc, self.pp(0.0, 1.0), "Gini", layout.MIN_TEXT, anchor="rm",
                       dx=-6, dy=-6 - STRIP / 2, opacity=fade, ink=layout.NOTE_INK)
        else:
            self.label(sc, self.pp(1.0, 0.0), "Gini", layout.MIN_TEXT, anchor="mt",
                       dx=6 + STRIP / 2, dy=6, opacity=fade, ink=layout.NOTE_INK)

    def _search(self, sc, n, u0, u1, u2, back):
        """Sweep the threshold over x1 (``u0``) then x2 (``u1``), drawing each
        split's weighted Gini as a curve; ``u2`` settles on the best."""
        # both features on one scale: the lowest Gini touches the baseline
        both = np.concatenate([gg[~np.isnan(gg)] for _t, gg in n.sweeps])
        glo, ghi = float(both.min()), max(n.gini, float(both.max()))
        fade = (1.0 - smoothstep(0.6, 1.0, u2)) * back
        for f, u in ((0, u0), (1, u1)):
            th, g = n.sweeps[f]
            ok = ~np.isnan(g)
            if not ok.any() or u <= 0:
                continue
            th, g = th[ok], g[ok]
            m = max(1, int(np.ceil(len(th) * min(u, 1.0))))
            self._strip(sc, f, fade)
            pts = [self._curve_point(f, th[i], g[i], glo, ghi) for i in range(m)]
            if len(pts) > 1:
                self.polyline(sc, pts, width=1.8, color=FOCUS, opacity=0.85 * fade)
            if u < 1:                               # the threshold being tried
                a, b = self._split_line(n, th[m - 1], f)
                sc.line(a, b, width=1.6, color=FOCUS, opacity=0.8 * back, bias=0.3)
                self.disk(sc, self._curve_point(f, th[m - 1], g[m - 1], glo, ghi), 3.2, FOCUS,
                          fade, lift=0.6, outline=0.0)
            else:                                   # this feature's best
                k, win = int(np.argmin(g)), n.feature == f
                self.disk(sc, self._curve_point(f, th[k], g[k], glo, ghi), 4.0 if win else 3.0,
                          FOCUS if win else GREY_EDGE, fade, lift=0.6, outline=0.0)

    def _predict(self, sc, p, u, back, side=1):
        """The new point, ringed and labelled, while it falls down the tree."""
        on = smoothstep(0.0, 0.08, u) * (1 - smoothstep(0.9, 1.0, u)) * back
        self.disk(sc, self.pp(*p), 6.5, (1.0, 1.0, 1.0), on, ring=FOCUS, ring_w=2.4,
                  ring_op=1.0, lift=1.2)
        self.label(sc, self.pp(*p), "new", layout.MIN_TEXT, anchor="lm" if side > 0 else "rm",
                   dx=13 * side, opacity=on, ink=FOCUS)

    # ------------------------------------------------------------------ the tree
    def _box_text(self, n, regions):
        if n.leaf:
            return "class " + CLASS_NAMES[n.majority] if regions > 0.5 else "leaf"
        return "x%s ≤ %s" % (sub(n.feature + 1), num(n.threshold))

    def _tree(self, sc, t, f):
        """Nodes appear as their parent is split; each box shows its question
        (or class), its class mix and its size.  The new point's path lights up."""
        shown = {""} if t >= self.split_t[""][0] else set()
        for n in self.nodes:
            if n.path in f.done:
                shown.update(c.path for c in n.children)
        appear = {}
        for n in self.nodes:
            if n.path == "":
                appear[""] = smoothstep(0.0, 0.5, t - self.split_t[""][0])
            elif n.path[:-1] in f.done:
                appear[n.path] = smoothstep(0.0, 0.4, t - self.split_t[n.path[:-1]][3])
        on_path = {m.path for m in f.path[:f.step + 1]}
        self._tree_edges(sc, shown, appear, on_path, f.intro)
        for n in self.nodes:
            if n.path in shown:
                self._box(sc, n, appear[n.path] * f.intro, f, on_path)

    def _tree_edges(self, sc, shown, appear, on_path, op):
        for n in self.nodes:
            for k, c in enumerate(n.children):
                if c.path not in shown:
                    continue
                a = self.shift(self.pos[n.path], 0, BOX_H / 2)
                b = self.shift(self.pos[c.path], 0, -BOX_H / 2)
                hot = n.path in on_path and c.path in on_path
                sc.line(a, b, width=2.6 if hot else 1.3, color=FOCUS if hot else EDGE,
                        opacity=(0.95 if hot else 0.45) * appear[c.path] * op, bias=0.0)
                if n.path == "":
                    self.label(sc, (a + b) / 2, "yes" if k == 0 else "no", layout.MIN_TEXT,
                               anchor="rb" if k == 0 else "lb", dx=-10 if k == 0 else 10,
                               dy=-4, opacity=appear[c.path] * op, ink=layout.NOTE_INK)

    def _box(self, sc, n, a, f, on_path):
        c = self.pos[n.path]
        searching = f.active is not None and f.active[0] is n
        split = (not n.leaf) and (n.path in f.done or (searching and f.active[3] > 0.3))
        fill = (0.985, 0.985, 0.98)
        if n.leaf and f.regions > 0:
            fill = C.mix(fill, C.mix(CLASS_COLORS[n.majority], (1, 1, 1), 0.55), f.regions)
        hot = searching or n.path in on_path
        self.rect(sc, c, BOX_W, BOX_H, fill, a, border=FOCUS if hot else GREY_EDGE,
                  border_w=2.0 if hot else 1.1, radius=7, lift=0.6)
        text = self._box_text(n, f.regions) if (split or n.leaf) else "?"
        self.label(sc, self.shift(c, 0, -15), text, layout.MIN_TEXT, opacity=a,
                   ink=layout.STRONG_INK)
        self._bar(sc, self.shift(c, 0, 3), n.counts, a)
        self.label(sc, self.shift(c, 0, 18), "n = %d" % len(n.idx), layout.MIN_TEXT,
                   opacity=a, ink=layout.NOTE_INK)

    def _bar(self, sc, c, counts, op):
        """Class proportions as a stacked bar."""
        w = BOX_W - 22
        x = -w / 2
        tot = counts.sum()
        for k, n in enumerate(counts):
            if n == 0:
                continue
            seg = w * n / tot
            self.rect(sc, self.shift(c, x + seg / 2, 0), max(seg, 1.5), 6, CLASS_COLORS[k], op,
                      radius=0.5, lift=0.75)
            x += seg

    def _axes(self, sc, op):
        """Axis names and the class legend under the plot."""
        self.label(sc, self.pp(0.5, 0.0), "x" + sub(1), layout.MIN_TEXT, anchor="mt", dy=8,
                   opacity=op)
        self.label(sc, self.pp(0.0, 0.5), "x" + sub(2), layout.MIN_TEXT, anchor="rm", dx=-8,
                   opacity=op)
        for k in range(3):
            q = self.pp(0.0 + 0.2 * k, 0.0)
            self.disk(sc, self.shift(q, 8, 34), 4.0, CLASS_COLORS[k], op, lift=0.8, outline=0.3)
            self.label(sc, self.shift(q, 16, 34), CLASS_NAMES[k], layout.MIN_TEXT, anchor="lm",
                       opacity=op)

    # ------------------------------------------------------------------ words
    def _words(self, t, f, reset):
        """The subtitle at ``t``, with the numbers of the split being tried."""
        if reset > 0 or t < self.split_t[""][0]:
            return ("%d training points, 2 features, 3 classes. A decision tree asks "
                    "yes/no questions about one feature at a time" % len(self.y))
        if f.active:
            return self._search_words(*f.active)
        if t < self.Tm["regions"][0] - 0.1:
            return "grow: the same search inside each part"
        if f.pk < 0:
            return ("each leaf predicts its most common class; the tree cuts the plane into "
                    "boxes. Training accuracy %.0f%%" % (100 * self.acc))
        p, n = NEW_POINTS[f.pk], f.path[f.step]
        if n.leaf:
            return "new point (%s, %s): a leaf, so the answer is class %s" % (
                num(p[0]), num(p[1]), CLASS_NAMES[n.majority])
        v = p[n.feature]
        return ("new point (%s, %s): is x%s = %s ≤ %s? %s"
                % (num(p[0]), num(p[1]), sub(n.feature + 1), num(v), num(n.threshold),
                   "yes, go left" if v <= n.threshold else "no, go right"))

    def _search_words(self, n, u0, u1, u2):
        where = "all %d points" % len(n.idx) if n.path == "" else \
            "the %d points in this box" % len(n.idx)
        if u2 > 0:
            return ("best: x%s ≤ %s. Gini impurity %s before, %s after (weighted by size)"
                    % (sub(n.feature + 1), num(n.threshold), num(n.gini, 3),
                       num(n.split_gini, 3)))
        f = 0 if u1 <= 0 else 1
        if n.path != "":
            return "grow: the same search for %s, on x%s then x%s" % (where, sub(1), sub(2))
        th, g = n.sweeps[f]
        ok = ~np.isnan(g)
        th, g = th[ok], g[ok]
        m = max(1, int(np.ceil(len(th) * min(u0 if f == 0 else u1, 1.0)))) - 1
        left = self.X[n.idx, f] <= th[m]
        return ("try every threshold on x%s for %s: x%s ≤ %s splits them %d | %d; "
                "Gini after = %s (lower = purer)"
                % (sub(f + 1), where, sub(f + 1), num(th[m]), int(left.sum()),
                   int((~left).sum()), num(g[m], 3)))

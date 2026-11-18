"""One decision tree, why it is not enough, and what averaging many does: animations
160-164 of the "next" catalogue (see :mod:`voxanim.mech`).

* :class:`SplitsAndImpurity`: how a node chooses its question: every threshold of two
  features scored by the impurity it removes;
* :class:`PruningAndDepth`: a tree grown to the end, then held back by depth, leaf
  size, gain and cost-complexity pruning, the setting chosen by cross-validation;
* :class:`Instability`: the same recipe on resamples of the same events gives
  different trees; bias and variance against depth;
* :class:`Bagging`: bootstrap resamples, a tree on each, and their vote;
* :class:`RandomForest`: bagging with a random handful of features at each split, and
  reading which features the forest uses.

They draw the experiments of :mod:`voxanim.study_trees`. No physics dataset is used:
real handwritten digits stand in for events (4s for signal, 9s for background), which
the first phase of each says. Every phase carries, at the bottom, what kind of evidence
it shows: exact arithmetic on the events drawn, or a measurement on MNIST.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import study_trees as S
from . import theme
from .diagram import draw_image
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    BLUE,
    DIGIT_INK,
    DIM,
    GREEN,
    MAGENTA,
    OCHRE,
    PAPER,
    RED,
    TEAL,
    VIOLET,
    dark,
    soft,
)
from .mech_phys_apps import GREY, Phys, commas, pct

MEASURED = "measured on MNIST, held-out"
TRAINING = "measured on MNIST, training events"
BIG, HEAD, SMALL, TEXT_S = 0.036, 0.033, 0.030, 0.029
SIG, BKG = BLUE, RED              # a 4 (signal) and a 9 (background), everywhere
ASK, GAIN, TRAIN, HELD = AMBER, GREEN, VIOLET, GREEN
SHORT = ("top", "spread")         # the two features inside a question
AXES = ("ink at the top", "vertical spread (pixels)")
X_TICKS = [(v, "%g" % v) for v in (0, 0.2, 0.4, 0.6, 0.8)]
Y_TICKS = [(v, "%g" % v) for v in (4, 5, 6)]
SCORE_INKS = dict(gini=ASK, entropy=MAGENTA, error=OCHRE)


def exact(n):
    """The tag of a phase that works something out from ``n`` events on screen."""
    return "exact arithmetic on %s events" % commas(n)


def is_evidence(text):
    """True for the tag every phase carries."""
    return text.startswith(("exact arithmetic on ", "measured on MNIST"))


def question(feature, threshold):
    """A node's question about one of the two features."""
    return "%s ≤ %.2f?" % (SHORT[feature], threshold)


def class_ink(share):
    """The colour of a node whose share of 4s is ``share``."""
    return tuple(float(v) for v in C.mix(BKG, SIG, float(share)))


def parents(tree):
    """Each node's parent (-1 for the root)."""
    up = np.full(len(tree["feature"]), -1, np.int64)
    for t in np.flatnonzero(tree["feature"] >= 0):
        up[tree["left"][t]] = up[tree["right"][t]] = t
    return up


def tree_layout(tree, box, internal=None, max_depth=None, depth_span=None):
    """Where each node of a tree is drawn in the layout box ``((x0, x1), (top,
    bottom))``: leaves evenly spaced from left to right, every other node midway
    between its children, one row per level. Returns ``{node: (fx, fy)}``."""
    asks = S._asks(tree, internal, max_depth)
    (x0, x1), (y0, y1) = box
    order, stack = [], [0]
    while stack:                                       # leaves, left to right
        t = stack.pop()
        if asks[t]:
            stack += [int(tree["right"][t]), int(tree["left"][t])]
        else:
            order.append(t)
    fx = {t: x0 + (i + 0.5) * (x1 - x0) / len(order) for i, t in enumerate(order)}
    for t in range(len(asks) - 1, -1, -1):             # children before parents
        if asks[t] and int(tree["left"][t]) in fx:
            fx[t] = (fx[int(tree["left"][t])] + fx[int(tree["right"][t])]) / 2
    deepest = depth_span or max(int(tree["depth"][t]) for t in fx) or 1
    return {t: (x, y0 + (y1 - y0) * int(tree["depth"][t]) / deepest) for t, x in fx.items()}


class Trees(Phys):
    """What the five animations share: the feature plane with its events and a tree's
    partition of it, trees drawn as trees, and maps of answers over the plane."""

    def plane(self, sc, box, op, labels=True, ticks=True):
        ch = self.pane(box, *S.PLANE)
        self.axes(sc, ch, op, AXES[0] if labels else "", AXES[1] if labels else "",
                  X_TICKS if ticks else [], Y_TICKS if ticks else [], grid=False)
        return ch

    def events(self, sc, ch, xy, classes, op, r=3.0, upto=None, weights=None):
        """Events as points in their class's colour (``weights``: a resample's counts;
        those left out are grey and small, those picked more than once larger)."""
        if op <= 0.01:
            return
        (x0, x1), (y0, y1) = ch.x_range, ch.y_range
        for i, ((x, y), c) in enumerate(zip(xy[:upto], classes[:upto])):
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                continue
            ink, size = (SIG if c else BKG), r
            if weights is not None:
                w = weights[i]
                ink, size = (DIM, 0.7 * r) if w == 0 else (ink, r * (0.75 + 0.3 * w))
            self.disk(sc, self.P(ch.fx(float(x)), ch.fy(float(y))), size, ink, op, lift=1.1)

    def partition(self, sc, ch, tree, cuts, op, internal=None, max_depth=None, fill=0.5):
        """The rectangles a two-feature tree cuts the plane into, each in the colour of
        its leaf's majority, deeper the purer the leaf."""
        if op <= 0.01:
            return
        for t, x0, x1, y0, y1 in S.regions(tree, cuts, S.PLANE, internal, max_depth):
            t = int(t)
            if x1 - x0 < 1e-9 or y1 - y0 < 1e-9:
                continue
            p = tree["n1"][t] / max(tree["n"][t], 1e-12)
            ink = soft(SIG if p > 0.5 else BKG, 0.55 - 0.9 * abs(p - 0.5))
            a, b = (ch.fx(x0), ch.fy(y1)), (ch.fx(x1), ch.fy(y0))
            w, h = self.span(b[0] - a[0], b[1] - a[1])
            self.rect(sc, self.P((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), w, h, ink, op,
                      radius=0.0, lift=0.15, fill_op=fill, border=PAPER, border_w=0.8)

    def tree_dots(self, sc, tree, pos, op, live=None, mark=(), r=(2.4, 9.0), grow=None):
        """A tree as a node per disk (its colour the node's mix of classes, its size
        its events) joined to its parent. ``live``: the nodes still in the tree (the
        rest are drawn faint); ``mark``: nodes ringed; ``grow``: levels shown so far."""
        if op <= 0.01:
            return
        up, top = parents(tree), tree["n"][0]
        for t, (fx, fy) in pos.items():
            d = int(tree["depth"][t])
            show = 1.0 if grow is None else float(np.clip(grow - d + 1, 0.0, 1.0))
            if show <= 0.01:
                continue
            on = live is None or live[t]
            a = op * show * (1.0 if on else 0.16)
            if up[t] >= 0:
                self.polyline(sc, [self.P(*pos[int(up[t])], 0.4), self.P(fx, fy, 0.4)], 1.3,
                              AXIS, a * 0.8, bias=0.2)
            size = r[0] + (r[1] - r[0]) * np.sqrt(tree["n"][t] / top)
            ink = class_ink(tree["n1"][t] / max(tree["n"][t], 1e-12))
            ring = ASK if t in mark else None
            self.disk(sc, self.P(fx, fy), size, ink, a, ring=ring, ring_w=3.0, lift=1.0 + 0.01 * d)

    def tree_boxes(self, sc, tree, cuts, pos, op, internal=None, max_depth=None, grow=None,
                   w=(104, 62), h=30, size=0.029, mark=()):
        """A small tree as boxes: a question in each node that asks one, and in each
        leaf how many 4s and 9s of the training events reach it."""
        if op <= 0.01:
            return
        asks = S._asks(tree, internal, max_depth)
        up = parents(tree)
        for t, (fx, fy) in pos.items():
            d = int(tree["depth"][t])
            show = op * (1.0 if grow is None else float(np.clip(grow - d + 1, 0.0, 1.0)))
            if show <= 0.01:
                continue
            if up[t] >= 0:
                px, py = pos[int(up[t])]
                gap = self.frac(0, h / 2 + 2)[1]
                self.arrow(sc, (px + (0.012 if fx > px else -0.012), py + gap),
                           (fx, fy - gap), AXIS, show, width=1.8)
            if asks[t]:
                f = int(tree["feature"][t])
                self.box(sc, fx, fy, w[0], h, soft(ASK, 0.55), show,
                         question(f, cuts[f, tree["cut"][t]]), border=ASK, size=size,
                         glow=0.5 if t in mark else 0.0)
            else:
                n1 = int(round(tree["n1"][t]))
                n0 = int(round(tree["n"][t])) - n1
                ink = SIG if n1 > n0 else BKG
                self.box(sc, fx, fy, w[1], h, soft(ink, 0.5), show, "%d | %d" % (n1, n0),
                         border=ink, size=size)

    def answers(self, sc, values, fx, fy, size, op, frame=True):
        """A map over the plane of the share of trees answering 4 (0..1): blue where
        they say 4, red where they say 9, pale where they are divided."""
        v = np.asarray(values, float)[..., None]
        mid = PAPER * 0.985
        tex = np.where(v >= 0.5, mid + (np.asarray(SIG) - mid) * (2 * v - 1) * 0.8,
                       mid + (np.asarray(BKG) - mid) * (1 - 2 * v) * 0.8)
        c = self.P(float(fx), float(fy))
        if frame:
            self.rect(sc, c, size + 6, size + 6, PAPER, op, border=AXIS, border_w=1.2, radius=4,
                      lift=0.1)
        draw_image(self, sc, c, tex.astype(np.float32), size, size, op, grid=0)

    def note(self, sc, kind, op):
        """The evidence tag at the bottom left (when the right is taken)."""
        self.text(sc, 0.05, 0.925, kind, op, anchor="lm", ink=dark(TEAL, 0.25))

    def classes_key(self, sc, fx, fy, op, step=0.115):
        self.key(sc, fx, fy, [("a 4: signal", SIG)], op)
        self.key(sc, fx + step, fy, [("a 9: background", BKG)], op)


# ====================================================================== 160: one split
def fours(n1, n0, lead=""):
    """Counts of a group in words: ``5 fours, 0 nines``."""
    return "%s%d four%s, %d nine%s" % (lead, n1, "" if n1 == 1 else "s", n0,
                                      "" if n0 == 1 else "s")


def gini_line(name, n1, n0, value):
    """One group's Gini impurity written out from its counts."""
    n = n1 + n0
    return "%s 1 − (%d/%d)² − (%d/%d)² = %.3f" % (name, n1, n, n0, n, value)


class SplitsAndImpurity(Trees):
    """Twelve real events as two numbers each; one threshold sliding along a feature
    with the counts either side and the Gini arithmetic of the best one; every
    threshold of both features as bars of gain and the peak taken; the same on 2,000
    events as curves; and the three impurities compared on a node where the error sees
    no gain, and as whole trees on the 49 cells."""

    TITLE = "Choosing a split"
    PHASES = (("one node", 9.0), ("a threshold", 13.0), ("every threshold", 10.0),
              ("2,000 events", 10.0), ("three scores", 12.0))
    LEFT = ((0.08, 0.40), (0.40, 0.73))
    RIGHT = ((0.60, 0.93), (0.38, 0.76))
    UPPER, LOWER = ((0.52, 0.94), (0.36, 0.53)), ((0.52, 0.94), (0.63, 0.80))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1 = s.panels[0].data, s.panels[1].data
        self.xy, self.cls = np.asarray(p0["xy"], float), np.asarray(p0["classes"])
        self.cuts, self.gains = np.asarray(p0["cuts"], float), np.asarray(p0["gains"], float)
        self.images = np.asarray(p0["images"])
        self.hand = m["handful"]
        self.f = int(self.hand["feature"])
        self.c = int(np.argmin(np.abs(self.cuts[self.f] - self.hand["threshold"])))
        self.grid, self.node_gains = np.asarray(p1["grid"]), np.asarray(p1["gains"])
        self.node_xy, self.node_cls = np.asarray(p1["xy"]), np.asarray(p1["classes"])
        self.node, self.blind, self.crit = m["node"], m["blind"], m["criteria"]
        self.shown = S.HANDFUL                       # the first 9: the digit explained
        tie = np.flatnonzero(np.isclose(self.gains[0], self.gains[0].max()).ravel())
        self.ties = [divmod(int(k), self.cuts.shape[1]) for k in tie]
        self.same_peak = int(self.gains[1].argmax()) == int(self.gains[0].argmax())

    def subtitle_for(self, k):
        h, n, b = self.hand, self.node, self.blind
        g, e = self.crit["Gini"]["held"], self.crit["misclassification"]["held"]
        return [
            "handwritten digits stand in for events: 4s for signal, 9s for background, "
            "as in an analysis. Each is read as two numbers, and a node of a tree must "
            "choose one question about one of them",
            "a question is a threshold on one feature. The Gini impurity of a group is 0 "
            "when it is all one class and 0.5 at half and half: this threshold takes it "
            "from %.3f to %.3f, a gain of %.3f"
            % (h["gini"]["node"], h["gini"]["after"], h["gini"]["gain"]),
            "every threshold of both features is tried, %d on each, and scored the same "
            "way. The node takes the peak: %s, gain %.3f. Entropy, another impurity, "
            "%s" % (self.cuts.shape[1], question(self.f, h["threshold"]).rstrip("?"),
                    h["gini"]["gain"], "peaks at the same threshold" if self.same_peak
                    else "peaks at another threshold"),
            "the same node with %s training events: the gain is now a curve along each "
            "feature. Its peak is %s, Gini gain %.3f; the best threshold on the ink at "
            "the top would give %.3f"
            % (commas(n["events"]), question(n["feature"], n["threshold"]).rstrip("?"),
               n["gini"]["gain"], n["best_other"]),
            "all three scores pick the same root. Further down the error sees no gain "
            "(%d wrong before, %d + %d after) where Gini sees %.3f: a tree grown on it "
            "stops early, at %s held-out error against %s"
            % (b["wrong"][0], b["wrong"][1], b["wrong"][2], b["gini"]["gain"],
               pct(e["mean"], 1), pct(g["mean"], 1)),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _line(self, sc, ch, f, v, ink, op, width=2.6, dashed=False):
        """A threshold on feature ``f`` across the plane."""
        if f == 0:
            self.vline(sc, ch, v, ink, op, width=width, dashed=dashed)
        else:
            self.hline(sc, ch, v, ink, op, width=width, dashed=dashed)

    def _points(self, sc, ch, op, r=6.0):
        self.events(sc, ch, self.xy, self.cls, op, r=r)

    def _gain_charts(self, sc, op, top, ranges=S.PLANE):
        out = []
        for f, box in enumerate((self.UPPER, self.LOWER)):
            ch = self.pane(box, ranges[f], (0.0, top))
            ticks = [(v, "%g" % v) for v in ((0.2, 0.4, 0.6) if top > 0.5 else (0.2, 0.4))]
            marks = (0, 0.2, 0.4, 0.6, 0.8) if f == 0 else (3.5, 4, 4.5, 5, 5.5, 6)
            self.axes(sc, ch, op, AXES[f] if f else "", "gain of each threshold on the "
                      + ("vertical spread" if f else "ink at the top"),
                      [(v, "%g" % v) for v in marks if ranges[f][0] <= v <= ranges[f][1]],
                      ticks, grid=False)
            out.append(ch)
        return out

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        self.text(sc, 0.05, 0.315, "one node of a tree: twelve events, two numbers each", op,
                  anchor="lm", strong=True, size=HEAD)
        ch = self.plane(sc, self.RIGHT, op)
        for i, ((x, y), c) in enumerate(zip(self.xy, self.cls)):
            fx, fy = 0.075 + 0.06 * (i % 6), 0.39 + 0.10 * (i // 6)
            ink = SIG if c else BKG
            self.digit(sc, self.images[i], fx, fy, 42, op, frame=soft(ink, 0.3))
            go = (u - 0.3 - 0.035 * i) / 0.14
            at = (ch.fx(x), ch.fy(y))
            self.pulse(sc, (fx, fy), at, go, ink, op, bend=0.06, r=5.0)
            if go >= 0.9:
                self.disk(sc, self.P(*at), 6.0, ink, op * min(1.0, (go - 0.9) * 10), lift=1.1)
        self.classes_key(sc, 0.06, 0.562, op, step=0.13)
        k = self.shown
        img, (fx, fy), px = self.images[k], (0.105, 0.735), 104 / 28
        self.digit(sc, img, fx, fy, 104, op)
        c = self.P(fx, fy)
        one = op * smoothstep(0.05, 0.15, u)
        rows, cols = S.TOP_ROWS, S.TOP_COLS
        self.rect(sc, self.shift(c, (sum(cols) / 2 - 14) * px, (sum(rows) / 2 - 14) * px),
                  (cols[1] - cols[0]) * px, (rows[1] - rows[0]) * px, ASK, one, border=ASK,
                  border_w=2.4, radius=2, lift=0.9, fill_op=0.0)
        self.text(sc, 0.185, 0.665, "%s = %.2f" % (AXES[0], self.xy[k, 0]), one, anchor="lm",
                  ink=dark(ASK, 0.4), size=HEAD)
        self.text(sc, 0.185, 0.705, "the mean ink in the amber patch:", one, anchor="lm",
                  size=SMALL)
        self.text(sc, 0.185, 0.747, "a 9 is closed there, a 4 open", one, anchor="lm",
                  size=SMALL)
        two = op * smoothstep(0.16, 0.26, u)
        mass = img.sum()
        mean = float((img.sum(1) * np.arange(28)).sum() / mass)
        s = float(self.xy[k, 1])
        a = self.shift(c, 62, (mean - s + 0.5 - 14) * px)
        b = self.shift(c, 62, (mean + s + 0.5 - 14) * px)
        self.polyline(sc, [self.shift(a, -6, 0), self.shift(a, 6, 0)], 2.6, TEAL, two)
        self.polyline(sc, [a, b], 2.6, TEAL, two)
        self.polyline(sc, [self.shift(b, -6, 0), self.shift(b, 6, 0)], 2.6, TEAL, two)
        self.text(sc, 0.185, 0.805, "vertical spread = %.2f pixels" % s, two, anchor="lm",
                  ink=dark(TEAL, 0.3), size=HEAD)
        self.text(sc, 0.185, 0.847, "how far the ink lies from its mean row", two,
                  anchor="lm", size=SMALL)
        self.stand_in(sc, op)
        self.tag(sc, exact(len(self.cls)), op)

    def _threshold(self, u):
        """The threshold on the best feature as it slides: through all of them, then
        back to the best."""
        cuts = self.cuts[self.f]
        n = len(cuts)
        if u < 0.46:
            return float(np.interp(np.clip((u - 0.05) / 0.38, 0, 1) * (n - 1), np.arange(n),
                                   cuts))
        w = smoothstep(0.46, 0.52, u)
        return float(cuts[-1] + (cuts[self.c] - cuts[-1]) * w)

    def phase_2(self, sc, u, op):
        ch = self.plane(sc, self.LEFT, op)
        self._points(sc, ch, op)
        f, v = self.f, self._threshold(u)
        self._line(sc, ch, f, v, ASK, op, width=3.0)
        yes = self.xy[:, f] <= v
        n = len(self.cls)
        l1, l0 = int(self.cls[yes].sum()), int((1 - self.cls[yes]).sum())
        r1, r0 = int(self.cls.sum()) - l1, int((1 - self.cls).sum()) - l0
        gain = float(S.split_gain(n, self.cls.sum(), l1 + l0, l1))
        x, top = 0.70, 0.335
        self.box(sc, x, top, 236, 30, PAPER, op, fours(r1 + l1, r0 + l0, "the node: "),
                 border=AXIS, size=SMALL)
        self.box(sc, x, top + 0.068, 150, 30, soft(ASK, 0.5), op, question(f, v), border=ASK,
                 size=SMALL)
        for side, (fx, a, b, word) in enumerate(((x - 0.12, l1, l0, "yes: "),
                                                 (x + 0.12, r1, r0, "no: "))):
            ink = GREY if a == b else (SIG if a > b else BKG)
            self.arrow(sc, (x + (-0.03 if side == 0 else 0.03), top + 0.099),
                       (fx, top + 0.139), AXIS, op, width=2.0)
            self.box(sc, fx, top + 0.17, 204, 30, soft(ink, 0.55), op, fours(a, b, word),
                     border=ink, size=SMALL)
        live = op * (1 - smoothstep(0.50, 0.54, u))
        self.text(sc, x, top + 0.25, "Gini gain of this threshold: %.3f" % gain, live,
                  ink=dark(GAIN, 0.3), size=HEAD)
        self.text(sc, x, top + 0.30, "slide it through all %d places between the events"
                  % len(self.cuts[f]), live, size=SMALL)
        h = self.hand
        g, e = h["gini"], h["entropy"]
        (a1, a0), (b1, b0), (c1, c0) = h["counts"]
        rows = [
            ("Gini impurity  G = 1 − (share of 4s)² − (share of 9s)²", None, True),
            (gini_line("node:", a1, a0, g["node"]), None, False),
            (gini_line("yes:", b1, b0, g["left"]), None, False),
            (gini_line("no:", c1, c0, g["right"]), None, False),
            ("after:  %d/%d × %.3f + %d/%d × %.3f = %.3f"
             % (b1 + b0, n, g["left"], c1 + c0, n, g["right"], g["after"]), None, False),
            ("gain = %.3f − %.3f = %.3f" % (g["node"], g["after"], g["gain"]),
             dark(GAIN, 0.3), True),
            ("entropy, −Σ p log₂ p, in bits: %.3f → %.3f, gain %.3f"
             % (e["node"], e["after"], e["gain"]), dark(MAGENTA, 0.3), False),
        ]
        for i, (row, ink, strong) in enumerate(rows):
            show = op * smoothstep(0.54 + 0.05 * i, 0.58 + 0.05 * i, u)
            self.text(sc, 0.475, top + 0.245 + 0.047 * i, row, show, anchor="lm", ink=ink,
                      strong=strong and ink is None, size=HEAD if i == 5 else SMALL)
        self.tag(sc, exact(n), op)

    def phase_3(self, sc, u, op):
        ch = self.plane(sc, self.LEFT, op)
        self._points(sc, ch, op)
        charts = self._gain_charts(sc, op, 0.72, ((0.0, 0.7), (4.2, 5.3)))
        n = self.cuts.shape[1]
        for f in range(2):
            go = float(np.clip((u - 0.06 - 0.4 * f) / 0.34, 0.0, 1.0))
            if 0.0 < go < 1.0:
                v = float(np.interp(go * (n - 1), np.arange(n), self.cuts[f]))
                self._line(sc, ch, f, v, ASK, op, width=3.0)
            for k in range(int(np.floor(go * (n - 1) + 1e-9)) + 1 if go > 0 else 0):
                c = charts[f]
                at = c.fx(self.cuts[f, k])
                top = c.fy(self.gains[0, f, k])
                h = self.span(0, c.y1 - top)[1]
                if h > 0.5:
                    self.rect(sc, self.P(at, (top + c.y1) / 2), 9, h, ASK, op, radius=1.5,
                              lift=0.5)
                self.disk(sc, self.P(at, c.fy(self.gains[1, f, k])), 3.4, MAGENTA, op,
                          lift=1.2)
        self.key(sc, 0.745, 0.378, [("bars: Gini gain", ASK), ("dots: entropy gain, bits",
                                                             MAGENTA)], op, step=0.038)
        end = op * smoothstep(0.84, 0.92, u)
        c = charts[self.f]
        self.disk(sc, self.P(c.fx(self.cuts[self.f, self.c]),
                             c.fy(self.gains[0, self.f, self.c])), 7.0, ASK, end, ring=GAIN,
                  ring_w=3.0, lift=1.5)
        self._line(sc, ch, self.f, self.cuts[self.f, self.c], GAIN, end, width=3.0)
        h = self.hand
        self.text(sc, 0.05, 0.84, "the peak: %s  Gini gain %.3f"
                  % (question(self.f, h["threshold"]), h["gini"]["gain"]), end, anchor="lm",
                  ink=dark(GAIN, 0.3), size=HEAD)
        other = [t for t in self.ties if t != (self.f, self.c)]
        if other:
            f2, c2 = other[0]
            self.text(sc, 0.05, 0.884, "%s ties with it: the first found is taken"
                      % question(f2, self.cuts[f2, c2]).rstrip("?"), end, anchor="lm",
                      size=SMALL)
        self.tag(sc, exact(len(self.cls)), op)

    def phase_4(self, sc, u, op):
        n = self.node
        ch = self.plane(sc, self.LEFT, op)
        self.events(sc, ch, self.node_xy, self.node_cls, op, r=2.6)
        charts = self._gain_charts(sc, op, 0.46)
        go = smoothstep(0.08, 0.6, u)
        for f in range(2):
            for k, name in enumerate(S.CRITERIA):
                ink = SCORE_INKS[name]
                self.path(sc, charts[f], self.grid[f], self.node_gains[k, f], ink, op, go,
                          width=3.2 if k == 0 else 2.0)
        self.key(sc, 0.80, 0.378, [("Gini", ASK), ("entropy, bits", MAGENTA),
                                   ("error", OCHRE)], op, step=0.038)
        end = op * smoothstep(0.62, 0.72, u)
        f, v = int(n["feature"]), n["threshold"]
        c = charts[f]
        self.disk(sc, self.P(c.fx(v), c.fy(n["gini"]["gain"])), 6.0, ASK, end, ring=GAIN,
                  ring_w=3.0, lift=1.5)
        self._line(sc, ch, f, v, GAIN, end, width=3.0)
        self._line(sc, ch, 1 - f, n["best_other_threshold"], ASK, 0.8 * end, width=2.0,
                   dashed=True)
        self.text(sc, 0.05, 0.315, "now %s training events (%d drawn)"
                  % (commas(n["events"]), len(self.node_xy)), op, anchor="lm", strong=True,
                  size=HEAD)
        (_, _), (b1, b0), (c1, c0) = n["counts"]
        self.text(sc, 0.05, 0.84, "the peak: %s  Gini gain %.3f"
                  % (question(f, v), n["gini"]["gain"]), end, anchor="lm",
                  ink=dark(GAIN, 0.3), size=HEAD)
        self.text(sc, 0.05, 0.884, "%s;  %s" % (fours(b1, b0, "yes: "), fours(c1, c0, "no: ")),
                  end, anchor="lm", size=SMALL)
        self.tag(sc, exact(n["events"]), op)

    def phase_5(self, sc, u, op):
        b = self.blind
        (a1, a0), (b1, b0), (c1, c0) = b["counts"]
        self.text(sc, 0.05, 0.315, "further down the tree: %s" % fours(a1, a0), op,
                  anchor="lm", strong=True, size=HEAD)
        self.text(sc, 0.05, 0.358, "%s  %s;  %s" % (question(b["feature"], b["threshold"]),
                                                   fours(b1, b0, "yes: "), fours(c1, c0, "no: ")),
                  op, anchor="lm", size=SMALL)
        ch = self.pane(((0.08, 0.41), (0.435, 0.715)), (0.0, 0.5), (0.0, 0.55))
        self.axes(sc, ch, op, "share of 9s in a group", "impurity",
                  [(v, "%g" % v) for v in (0, 0.1, 0.2, 0.3, 0.4, 0.5)],
                  [(v, "%g" % v) for v in (0.1, 0.3, 0.5)], grid=False)
        p = np.linspace(0.0, 0.5, 60)
        go = smoothstep(0.03, 0.25, u)
        curves = (("gini", S.impurity(p)), ("entropy", S.impurity(p, "entropy") / 2),
                  ("error", S.impurity(p, "error")))
        for name, ys in curves:
            self.path(sc, ch, p, ys, SCORE_INKS[name], op, go, width=2.6)
        self.key(sc, 0.10, 0.455, [("Gini", ASK), ("entropy ÷ 2", MAGENTA), ("error", OCHRE)],
                 op, step=0.038)
        shares = [a0 / (a1 + a0), b0 / (b1 + b0), c0 / (c1 + c0)]
        mark = op * smoothstep(0.28, 0.4, u)
        for name in ("gini", "error"):
            ink = SCORE_INKS[name]
            at = [(ch.fx(q), ch.fy(float(S.impurity(q, name)))) for q in shares]
            self.polyline(sc, [self.P(*at[1], 1.0), self.P(*at[2], 1.0)], 2.0, DIGIT_INK,
                          0.8 * mark)
            for k, pt in enumerate(at):
                self.disk(sc, self.P(*pt), 6.0 if k == 0 else 4.5, ink, mark,
                          ring=DIGIT_INK if k == 0 else None, ring_w=1.6, lift=1.5)
            if name == "gini":
                after = ch.fy(b["gini"]["after"])
                self.polyline(sc, [self.P(at[0][0], at[0][1], 1.3), self.P(at[0][0], after, 1.3)],
                              4.0, GAIN, mark)
        gap = op * smoothstep(0.42, 0.52, u)
        self.text(sc, 0.05, 0.825, "Gini: %.3f before, %.3f after (the chord): gain %.3f"
                  % (b["gini"]["node"], b["gini"]["after"], b["gini"]["gain"]), gap,
                  anchor="lm", ink=dark(ASK, 0.4), size=SMALL)
        self.text(sc, 0.05, 0.868, "error: %d wrong before, %d + %d after: no gain"
                  % tuple(b["wrong"]), gap, anchor="lm", ink=dark(OCHRE, 0.3), size=SMALL)
        x = 0.60
        self.text(sc, x, 0.315, "whole trees on 49 cells, by score", op,
                  anchor="lm", strong=True, size=HEAD)
        self.text(sc, x, 0.36, "held-out error, mean of %d seeds" % len(self.meta["seeds"]),
                  op, anchor="lm", size=SMALL)
        full = self.span(0.13)[0]
        worst = max(v["held"]["mean"] for v in self.crit.values())
        for r, (name, key) in enumerate((("Gini", "gini"), ("entropy", "entropy"),
                                         ("misclassification", "error"))):
            v = self.crit[name]
            self.hbar(sc, x + 0.085, 0.425 + 0.062 * r, full * v["held"]["mean"] / worst,
                      SCORE_INKS[key], op, "error" if key == "error" else name,
                      "%s, %d leaves" % (pct(v["held"]["mean"], 1), round(v["leaves"]["mean"])),
                      u=(u - 0.5 - 0.06 * r) / 0.15, thick=22.0)
        same = self.meta["same_split"]
        self.lines(sc, x, 0.64, [
            "entropy takes Gini's split at %s of the nodes" % pct(same["entropy"]["mean"]),
            "at %s of them no threshold lowers the error:"
            % pct(same["error_sees_nothing"]["mean"]),
            "a tree grown on the error stops there, with",
            "%s of its training events still wrong"
            % pct(self.crit["misclassification"]["train"]["mean"], 1)],
            op * smoothstep(0.72, 0.82, u), size=SMALL, step=0.044)
        self.note(sc, exact(a1 + a0), op)
        self.tag(sc, MEASURED, op)


# ====================================================================== 161: holding back
class PruningAndDepth(Trees):
    """A tree grown level by level until every leaf is pure, with its partition of the
    plane and its error on training and held-out events; the same tree cut at a depth;
    regrown with a minimum leaf size and with a minimum gain; pruned weakest link
    first; the setting chosen by cross-validation; and every control over seeds."""

    TITLE = "Holding a tree back"
    PHASES = (("to the end", 10.0), ("depth", 8.0), ("leaf size, gain", 10.0),
              ("pruning", 12.0), ("cross-validation", 9.0), ("over seeds", 9.0))
    LEFT = ((0.07, 0.36), (0.40, 0.70))
    TREE = ((0.42, 0.94), (0.34, 0.56))
    CHART = ((0.50, 0.76), (0.69, 0.81))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.xy, self.cls = np.asarray(p0["xy"], float), np.asarray(p0["classes"])
        self.tree, self.cuts = S.tree_from(p0, "tree_"), np.asarray(p0["cuts"])
        self.pos = tree_layout(self.tree, self.TREE)
        self.deep = int(self.tree["depth"].max())
        self.other = {}
        for name in ("leaf", "gain"):
            t = S.tree_from(p0, name + "_tree_")
            self.other[name] = (t, np.asarray(p0[name + "_cuts"]), tree_layout(t, self.TREE))
        self.depths = np.asarray(p1["x"], float)
        self.train, self.held = (np.asarray(v, float) for v in p1["ys"])
        self.leaves = np.asarray(p1["leaves"])
        self.alphas, self.internals = np.asarray(p2["alphas"]), np.asarray(p2["internals"])
        self.links, self.seq = np.asarray(p2["links"]), np.asarray(p2["seq"], float)
        self.grid = np.asarray(p2["x"], float)
        self.cv, self.cv_held = np.asarray(p2["ys"][0]), np.asarray(p2["ys"][1])
        self.p3 = p3
        self.chosen, self.grown, self.over = m["chosen"], m["grown"], m["over_seeds"]
        self.steps = m["sequence"]

    def subtitle_for(self, k):
        m, g, c, o = self.meta, self.grown, self.chosen, self.over
        d = m["shown"]["depth"]
        leaf = self.p3["two_leaf"][0, S.LEAF_SIZES.index(m["shown"]["leaf"])]
        gain = self.p3["two_gain"][0, S.MIN_GAINS.index(m["shown"]["gain"])]
        return [
            "digits stand in for events, as in an analysis: 4s signal, 9s background. "
            "Grown until every leaf is pure, a tree cuts the plane into %d pieces: no "
            "training event is wrong, but %s of the held-out ones are"
            % (g["leaves"], pct(g["held"], 1)),
            "the first control is a maximum depth. Cut at depth %d, the same tree has %d "
            "leaves: it gets %s of its training events wrong and %s of the held-out "
            "ones, down from %s"
            % (d, self.leaves[d - 1], pct(self.train[d - 1], 1), pct(self.held[d - 1], 1),
               pct(g["held"], 1)),
            "two more controls stop a split while the tree grows: every leaf must keep "
            "at least %d events (held-out error %s), or a split must remove at least "
            "%g of the impurity (%s)"
            % (m["shown"]["leaf"], pct(leaf[1], 1), m["shown"]["gain"], pct(gain[1], 1)),
            "pruning grows the whole tree, then cuts the weakest link, the cut that adds "
            "fewest errors for each leaf it removes, again and again. A price α per "
            "leaf says where to stop: %d trees, from %d leaves down to 1"
            % (len(self.alphas), self.steps["leaves"][0]),
            "cross-validation chooses without the held-out events: grow on four fifths "
            "of the training events, judge on the rest. It picks α = %g: %d leaves, "
            "held-out error %s against %s grown to the end"
            % (c["alpha"]["setting"], c["alpha"]["leaves"], pct(c["alpha"]["held"], 1),
               pct(g["held"], 1)),
            "over %d seeds, two features: held-out error turns up from %s at the best "
            "depth to %s at the end. On 49 cells and %s events it only flattens (%s, "
            "then %s); pruning gains a little there (%s)"
            % (len(m["seeds"]), pct(o["two_depth"]["best"], 1), pct(o["two_depth"]["end"], 1),
               commas(2 * S.CELLS_N), pct(o["cells_depth"]["best"], 1),
               pct(o["cells_depth"]["end"], 1), pct(o["cells_alpha"]["best"], 1)),
        ][k]

    # ------------------------------------------------------------------ shared drawing
    def _plane(self, sc, op, tree=None, cuts=None, **how):
        ch = self.plane(sc, self.LEFT, op)
        tree, cuts = (self.tree, self.cuts) if tree is None else (tree, cuts)
        self.partition(sc, ch, tree, cuts, op, **how)
        self.events(sc, ch, self.xy, self.cls, op, r=2.2)
        return ch

    def _numbers(self, sc, op, head, leaves, train, held, x=0.80):
        self.text(sc, x, 0.69, head, op, anchor="lm", strong=True, size=HEAD)
        self.text(sc, x, 0.735, "%d leaves" % leaves, op, anchor="lm", size=SMALL)
        self.key(sc, x + 0.008, 0.782, [("training %s" % pct(train, 1), TRAIN),
                                        ("held-out %s" % pct(held, 1), HELD)], op,
                 step=0.044, size=SMALL)

    def _depth_chart(self, sc, op, go=1.0, mark=None):
        last = self.deep
        ch = self.pane(self.CHART, (1, last), (0.0, 0.2))
        self.axes(sc, ch, op, "depth", "share of events wrong",
                  [(v, "%d" % v) for v in (1, 4, 8, last)],
                  [(v, pct(v)) for v in (0.0, 0.1, 0.2)], grid=False)
        xs = self.depths[:last]
        self.path(sc, ch, xs, self.train[:last], TRAIN, op, go, width=2.8)
        self.path(sc, ch, xs, self.held[:last], HELD, op, go, width=3.2, dot=True)
        if mark is not None:
            self.vline(sc, ch, mark, ASK, op, width=2.0, dashed=True)
        return ch

    def _at(self, d):
        """Training error, held-out error and leaves of the tree cut at depth ``d``
        (fractional: between the two whole depths)."""
        k = int(np.clip(np.floor(d), 1, self.deep)) - 1
        w = float(np.clip(d - 1 - k, 0.0, 1.0))
        j = min(k + 1, self.deep - 1)
        return (self.train[k] + (self.train[j] - self.train[k]) * w,
                self.held[k] + (self.held[j] - self.held[k]) * w)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        d = 1 + (self.deep - 1) * smoothstep(0.08, 0.78, u)
        whole = int(np.floor(d + 1e-9))
        self.text(sc, 0.05, 0.315, "grow until every leaf is pure", op, anchor="lm",
                  strong=True, size=HEAD)
        self._plane(sc, op, max_depth=whole)
        self.tree_dots(sc, self.tree, self.pos, op, grow=d)
        self.text(sc, 0.68, 0.598, "each disk a node: blue where 4s reach it, red 9s", op,
                  size=SMALL)
        self._depth_chart(sc, op, (d - 1) / (self.deep - 1))
        train, held = self._at(d)
        self._numbers(sc, op, "depth %d" % whole, self.leaves[whole - 1], train, held)
        self.stand_in(sc, op)
        self.tag(sc, MEASURED, op)

    def phase_2(self, sc, u, op):
        stop = self.meta["shown"]["depth"]
        d = self.deep + (stop - self.deep) * smoothstep(0.12, 0.62, u)
        whole = int(np.ceil(d - 1e-9))
        self.text(sc, 0.05, 0.315, "control 1: a maximum depth", op, anchor="lm",
                  strong=True, size=HEAD)
        self._plane(sc, op, max_depth=whole)
        self.tree_dots(sc, self.tree, self.pos, op, live=S.reached(self.tree, max_depth=whole))
        y = self.pos[0][1] + (self.TREE[1][1] - self.TREE[1][0]) * (d + 0.5) / self.deep
        self.dashes(sc, (self.TREE[0][0], y), (self.TREE[0][1], y), ASK, op, n=30, width=2.2)
        self.text(sc, 0.68, 0.598, "nothing below depth %d is asked" % whole, op,
                  ink=dark(ASK, 0.4), size=SMALL)
        self._depth_chart(sc, op, 1.0, mark=d)
        self._numbers(sc, op, "depth %d" % whole, self.leaves[whole - 1],
                      self.train[whole - 1], self.held[whole - 1])
        self.tag(sc, MEASURED, op)

    def _control_chart(self, sc, op, name, values, shown, label, fmt):
        rows = np.asarray(self.p3["two_" + name][0], float)       # setting: train, held, leaves
        x = np.arange(len(values), dtype=float)
        ch = self.pane(self.CHART, (0, len(values) - 1), (0.0, 0.2))
        pick = sorted({0, len(values) // 3, values.index(shown), len(values) - 1})
        self.axes(sc, ch, op, label, "share of events wrong",
                  [(float(i), fmt % values[i]) for i in pick],
                  [(v, pct(v)) for v in (0.0, 0.1, 0.2)], grid=False)
        self.path(sc, ch, x, rows[:, 0], TRAIN, op, width=2.8)
        self.path(sc, ch, x, rows[:, 1], HELD, op, width=3.2)
        k = values.index(shown)
        self.vline(sc, ch, k, ASK, op, width=2.0, dashed=True)
        return rows[k]

    def phase_3(self, sc, u, op):
        shown = self.meta["shown"]
        parts = (("leaf", S.LEAF_SIZES, shown["leaf"], "events a leaf must keep", "%d",
                  "control 2: a minimum leaf size", "at least %d events in every leaf"
                  % shown["leaf"], 1 - smoothstep(0.42, 0.46, u)),
                 ("gain", S.MIN_GAINS, shown["gain"], "gain a split must reach", "%g",
                  "control 3: a minimum gain", "a split must remove %g of the impurity"
                  % shown["gain"], smoothstep(0.46, 0.5, u)))
        for name, values, at, label, fmt, head, line, on in parts:
            a = op * on
            if a <= 0.01:
                continue
            tree, cuts, pos = self.other[name]
            self.text(sc, 0.05, 0.315, head, a, anchor="lm", strong=True, size=HEAD)
            self._plane(sc, a, tree, cuts)
            self.tree_dots(sc, tree, pos, a, r=(3.0, 10.0))
            self.text(sc, 0.68, 0.598, line, a, ink=dark(ASK, 0.4), size=SMALL)
            row = self._control_chart(sc, a, name, list(values), at, label, fmt)
            self._numbers(sc, a, (fmt % at) + (" events" if name == "leaf" else " gain"),
                          int(row[2]), row[0], row[1])
        self.tag(sc, MEASURED, op)

    def phase_4(self, sc, u, op):
        last = len(self.alphas) - 2                     # the last tree drawn: two leaves
        s = float(np.clip((u - 0.1) / 0.72, 0.0, 1.0)) * last
        k = min(int(s), last)
        self.text(sc, 0.05, 0.315, "control 4: grow it all, then prune", op, anchor="lm",
                  strong=True, size=HEAD)
        self._plane(sc, op, internal=self.internals[k])
        nxt = int(self.links[k + 1]) if k < last else -1
        mark = (nxt,) if nxt >= 0 else ()
        self.tree_dots(sc, self.tree, self.pos, op,
                       live=S.reached(self.tree, internal=self.internals[k]), mark=mark)
        self.text(sc, 0.68, 0.598, "ringed: the weakest link, cut next", op,
                  ink=dark(ASK, 0.4), size=SMALL)
        top = float(self.alphas[last]) * 1.05
        ch = self.pane(self.CHART, (0.0, top), (0.0, 0.2))
        self.axes(sc, ch, op, "price of a leaf, α", "share of events wrong",
                  [(v, "%g" % v) for v in (0, 0.004, 0.008, 0.012)],
                  [(v, pct(v)) for v in (0.0, 0.1, 0.2)], grid=False)
        go = (k + 1e-6) / last
        self.path(sc, ch, self.alphas[:last + 1], self.seq[:last + 1, 0], TRAIN, op, go,
                  width=2.8)
        self.path(sc, ch, self.alphas[:last + 1], self.seq[:last + 1, 1], HELD, op, go,
                  width=3.2, dot=True)
        self._numbers(sc, op, "α = %.4f" % self.alphas[k], int(self.seq[k, 2]), self.seq[k, 0],
                      self.seq[k, 1])
        self.text(sc, 0.05, 0.805, "cost = training error + α × number of leaves", op,
                  anchor="lm", strong=True, size=SMALL)
        self.text(sc, 0.05, 0.848, "weakest link: fewest new errors per leaf cut", op,
                  anchor="lm", size=SMALL)
        if k < last:
            wrong, below, leaves = self.steps["cuts"][k]
            n = len(self.cls)
            self.text(sc, 0.05, 0.891, "next: (%d − %d) / %d ÷ (%d − 1) = %.4f"
                      % (wrong, below, n, leaves, (wrong - below) / n / (leaves - 1)), op,
                      anchor="lm", ink=dark(ASK, 0.4), size=SMALL)
        self.tag(sc, MEASURED, op)

    def phase_5(self, sc, u, op):
        c = self.chosen["alpha"]
        internal = self.internals[c["tree"]]
        self.text(sc, 0.05, 0.315, "which setting? cross-validation", op, anchor="lm",
                  strong=True, size=HEAD)
        self._plane(sc, op, internal=internal)
        pos = tree_layout(self.tree, ((0.43, 0.95), (0.335, 0.60)), internal=internal)
        self.tree_boxes(sc, self.tree, self.cuts, pos, op * smoothstep(0.55, 0.7, u),
                        internal=internal, w=(102, 58), h=25)
        turn = int(u * 10) % S.FOLDS
        for i in range(S.FOLDS):
            out = i == turn
            self.box(sc, 0.10 + 0.058 * i, 0.815, 50, 26, soft(TEAL, 0.2) if out else PAPER,
                     op, "judge" if out else "grow", border=TEAL if out else AXIS, size=TEXT_S)
        self.text(sc, 0.05, 0.868, "%d folds of the training events, each held" % S.FOLDS,
                  op, anchor="lm", size=SMALL)
        self.text(sc, 0.05, 0.906, "back once", op, anchor="lm", size=SMALL)
        x = np.arange(len(self.grid), dtype=float)
        ch = self.pane(self.CHART, (0, len(x) - 1), (0.06, 0.16))
        k = int(c["index"])
        self.axes(sc, ch, op, "price of a leaf, α", "share of events wrong",
                  [(float(i), "%g" % self.grid[i]) for i in sorted({0, k, 8, len(x) - 1})],
                  [(v, pct(v)) for v in (0.08, 0.12, 0.16)], grid=False)
        go = smoothstep(0.05, 0.45, u)
        self.path(sc, ch, x, self.cv_held, HELD, op, go, width=2.4)
        self.path(sc, ch, x, self.cv, TEAL, op, go, width=3.2, dot=True)
        pick = op * smoothstep(0.48, 0.56, u)
        self.disk(sc, self.P(ch.fx(k), ch.fy(self.cv[k])), 6.0, TEAL, pick, ring=ASK,
                  ring_w=3.0, lift=1.5)
        xx = 0.79
        self.key(sc, xx + 0.008, 0.675, [("cross-validated", TEAL), ("held-out", HELD)], op,
                 step=0.04, size=SMALL)
        self.text(sc, xx, 0.765, "chosen: α = %g" % c["setting"], pick, anchor="lm",
                  strong=True, size=HEAD)
        self.text(sc, xx, 0.808, "%d leaves: %s" % (c["leaves"], pct(c["held"], 1)),
                  pick, anchor="lm", size=SMALL)
        self.text(sc, xx, 0.848, "to the end: %s" % pct(self.grown["held"], 1), pick,
                  anchor="lm", size=SMALL)
        self.tag(sc, MEASURED, op)

    def phase_6(self, sc, u, op):
        p3 = self.p3
        seeds = len(self.meta["seeds"])
        self.text(sc, 0.05, 0.315, "every control, %d seeds" % seeds, op, anchor="lm",
                  strong=True, size=HEAD)
        charts = (("depth", "maximum depth", list(S.DEPTHS), "%d", (0, 3, 7, 15)),
                  ("leaf", "events a leaf must keep", list(S.LEAF_SIZES), "%d", (0, 3, 6, 10)),
                  ("alpha", "price of a leaf, α", list(S.ALPHAS), "%g", (0, 4, 8, 11)))
        go = smoothstep(0.05, 0.55, u)
        for i, (name, label, values, fmt, pick) in enumerate(charts):
            x0 = 0.09 + 0.305 * i
            ch = self.pane(((x0, x0 + 0.24), (0.40, 0.70)), (0, len(values) - 1), (0.0, 0.26))
            ticks = [(float(k), "end" if name == "depth" and k == 15 else fmt % values[k])
                     for k in pick]
            self.axes(sc, ch, op, label, "share wrong" if i == 0 else "", ticks,
                      [(v, pct(v)) for v in (0.0, 0.1, 0.2)] if i == 0 else [], grid=False)
            x = np.arange(len(values), dtype=float)
            two, cel = np.asarray(p3["two_" + name]), np.asarray(p3["cells_" + name])
            for row in two[:, :, 1]:
                self.path(sc, ch, x, row, soft(HELD, 0.4), op, go, width=1.2)
            self.path(sc, ch, x, two[:, :, 0].mean(0), TRAIN, op, go, width=2.6)
            self.path(sc, ch, x, cel[:, :, 1].mean(0), OCHRE, op, go, width=2.8)
            self.path(sc, ch, x, two[:, :, 1].mean(0), HELD, op, go, width=3.2)
        y = 0.835
        for j, (name, ink) in enumerate((("two features, training", TRAIN),
                                         ("two features, held-out", HELD),
                                         ("49 cells, held-out", OCHRE))):
            self.key(sc, 0.09 + 0.27 * j, y, [(name, ink)], op, size=SMALL)
        o = self.over
        self.text(sc, 0.05, 0.882, "49 cells: %s grown to the end; best depth %s, best "
                  "leaf size %s, best α %s"
                  % (pct(o["cells_depth"]["end"], 1), pct(o["cells_depth"]["best"], 1),
                     pct(o["cells_leaf"]["best"], 1), pct(o["cells_alpha"]["best"], 1)),
                  op * smoothstep(0.6, 0.7, u), anchor="lm", size=SMALL)
        self.tag(sc, MEASURED, op)


# ====================================================================== 162: instability
DEPTH_TICKS = [(float(i), "end" if d >= 99 else "%d" % d) for i, d in enumerate(S.INST_DEPTHS)]
KIND_INKS = dict(two=TEAL, cells=OCHRE)
KIND_NAMES = dict(two="two features", cells="49 cells")
BIAS, VARIANCE = AMBER, MAGENTA


class Instability(Trees):
    """The same 500 events resampled and a tree grown on each resample; four such trees
    side by side with their first questions and partitions; their answers to held-out
    events; how often two trees disagree against depth and after a few events are
    swapped; and the squared error split into bias and variance."""

    TITLE = "One tree is unstable"
    PHASES = (("resamples", 10.0), ("four trees", 12.0), ("they disagree", 10.0),
              ("how often", 9.0), ("bias and variance", 12.0))
    LEFT = ((0.07, 0.40), (0.40, 0.73))
    COLUMNS = (0.165, 0.39, 0.615, 0.84)

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.xy, self.cls = np.asarray(p0["xy"], float), np.asarray(p0["classes"])
        self.weights = np.asarray(p0["weights"])
        self.trees = [(S.tree_from(p0, "tree%d_" % b), np.asarray(p0["cuts%d" % b]))
                      for b in range(len(self.weights))]
        self.votes, self.labels = np.asarray(p1["values"]), np.asarray(p1["labels"])
        self.images = np.asarray(p1["images"])
        self.points, self.points_y = np.asarray(p1["points_xy"]), np.asarray(p1["points_y"])
        self.point_votes = np.asarray(p1["point_votes"])
        self.x = np.asarray(p2["x"], float)
        self.disagree, self.per_seed = np.asarray(p2["ys"]), np.asarray(p2["per_seed"])
        self.parts = np.asarray(p3["ys"], float)
        self.one, self.one_image = np.asarray(p3["one"], float), np.asarray(p3["one_image"])
        self.shown = m["shown"]

    def subtitle_for(self, k):
        m = self.meta
        first = [t["threshold"][0] for t in self.shown]
        d, sw, p = m["disagreement"], m["swapped"], m["parts"]["cells"]
        errs = m["shown_errors"]
        return [
            "digits stand in for events, as in an analysis: 4s signal, 9s background. "
            "Draw the same %d training events again with replacement, some twice and "
            "about a third not at all, and grow the same kind of tree"
            % len(self.cls),
            "four resamples, four trees. The first question keeps its feature and its "
            "threshold moves a little (%.2f to %.2f); from the second question down the "
            "trees part ways, and so do their partitions" % (min(first), max(first)),
            "so they answer differently. On %s of held-out events the four trees do not "
            "all agree, though each alone is right about as often: %s to %s wrong"
            % (pct(m["split_share"]), pct(min(errs), 1), pct(max(errs), 1)),
            "grown to the end, two trees from different resamples disagree on %s of "
            "held-out events (%s with 49 cells). Swapping just %s of the training "
            "events for new ones changes %s of the answers"
            % (pct(d["two"]["end"]["mean"], 1), pct(d["cells"]["end"]["mean"], 1),
               pct(sw["cells"]["shares"][1]), pct(sw["cells"]["changed"][1], 1)),
            "squared error = bias² (how wrong the average tree is) + variance (the "
            "scatter around it). With depth bias² falls from %.3f to %.3f, variance "
            "grows from %.3f to %.3f: bagging removes variance, boosting bias"
            % (p["bias2"][0], p["bias2"][-1], p["variance"][0], p["variance"][-1]),
        ][k]

    def _top(self, sc, b, fx, fy, op, grow=2.0):
        """Tree ``b``'s first two levels of questions."""
        t = self.shown[b]
        self.box(sc, fx, fy, 108, 27, soft(ASK, 0.5), op * min(grow, 1.0),
                 question(t["feature"][0], t["threshold"][0]), border=ASK, size=TEXT_S)
        low = op * float(np.clip(grow - 1.0, 0.0, 1.0))
        for side in (0, 1):
            x = fx + (-0.053 if side == 0 else 0.053)
            self.arrow(sc, (fx + (-0.02 if side == 0 else 0.02), fy + 0.03), (x, fy + 0.052),
                       AXIS, low, width=1.8)
            self.box(sc, x, fy + 0.082, 92, 27, soft(ASK, 0.5), low,
                     question(t["feature"][1 + side], t["threshold"][1 + side]), border=ASK,
                     size=TEXT_S)

    def phase_1(self, sc, u, op):
        n = len(self.weights)
        b = min(int(u * n), n - 1)
        within = u * n - b
        w = self.weights[b]
        tree, cuts = self.trees[b]
        ch = self.plane(sc, self.LEFT, op)
        self.partition(sc, ch, tree, cuts, op * smoothstep(0.35, 0.55, within))
        self.events(sc, ch, self.xy, self.cls, op, r=2.4, weights=w)
        x = 0.48
        self.text(sc, x, 0.33, "the same %d events, resampled" % len(w), op, anchor="lm",
                  strong=True, size=HEAD)
        self.lines(sc, x, 0.385, ["pick %d with replacement: some events come" % len(w),
                                  "twice or more (larger), others not at all (grey)"], op,
                   size=SMALL, step=0.042)
        self.text(sc, x, 0.495, "resample %d: %d events left out, %d picked more than once"
                  % (b + 1, int((w == 0).sum()), int((w > 1).sum())), op, anchor="lm",
                  ink=dark(ASK, 0.4), size=SMALL)
        grow = 2.0 * smoothstep(0.3, 0.7, within)
        self.text(sc, x, 0.565, "its tree starts with", op * min(grow, 1.0), anchor="lm",
                  size=SMALL)
        self._top(sc, b, 0.74, 0.625, op, grow)
        self.text(sc, x, 0.80, "and goes on to %d leaves" % self.shown[b]["leaves"],
                  op * smoothstep(0.6, 0.75, within), anchor="lm", size=SMALL)
        self.stand_in(sc, op)
        self.tag(sc, TRAINING, op)

    def phase_2(self, sc, u, op):
        for b, fx in enumerate(self.COLUMNS):
            on = op * smoothstep(0.04 + 0.13 * b, 0.12 + 0.13 * b, u)
            self.text(sc, fx, 0.315, "resample %d" % (b + 1), on, strong=True, size=SMALL)
            self._top(sc, b, fx, 0.365, on)
            ch = self.plane(sc, ((fx - 0.095, fx + 0.095), (0.50, 0.80)), on, labels=False,
                            ticks=False)
            tree, cuts = self.trees[b]
            self.partition(sc, ch, tree, cuts, on, fill=0.75)
            self.text(sc, fx, 0.83, "%d leaves" % self.shown[b]["leaves"], on, size=SMALL)
        end = op * smoothstep(0.62, 0.72, u)
        first = [t["threshold"][0] for t in self.shown]
        second = [t["threshold"][1] for t in self.shown]
        self.text(sc, 0.05, 0.88, "first threshold %.2f to %.2f; second %.2f to %.2f; "
                  "below that, different questions" % (min(first), max(first), min(second),
                                                       max(second)), end, anchor="lm",
                  size=SMALL)
        self.tag(sc, TRAINING, op)

    def phase_3(self, sc, u, op):
        ch = self.plane(sc, self.LEFT, op)
        split = self.point_votes.min(0) != self.point_votes.max(0)
        for (x, y), c, s in zip(self.points, self.points_y, split):
            if not (S.PLANE[0][0] <= x <= S.PLANE[0][1] and S.PLANE[1][0] <= y <= S.PLANE[1][1]):
                continue
            ink = SIG if c else BKG
            if s:
                self.disk(sc, self.P(ch.fx(x), ch.fy(y)), 3.6, ink,
                          op * smoothstep(0.1, 0.25, u), ring=ASK, ring_w=2.0, lift=1.3)
            else:
                self.disk(sc, self.P(ch.fx(x), ch.fy(y)), 2.2, soft(ink, 0.45), op, lift=1.0)
        self.text(sc, 0.05, 0.315, "%d held-out events; ringed: the four trees differ"
                  % len(self.points), op, anchor="lm", strong=True, size=SMALL)
        x0, dx = 0.555, 0.072
        self.text(sc, 0.47, 0.315, "six of them, and each tree's answer", op, anchor="lm",
                  strong=True, size=SMALL)
        for i, img in enumerate(self.images):
            fx = x0 + dx * i
            ink = SIG if self.labels[i] else BKG
            self.digit(sc, img, fx, 0.405, 54, op, frame=soft(ink, 0.3))
        for b in range(len(self.votes)):
            fy = 0.515 + 0.058 * b
            on = op * smoothstep(0.15 + 0.12 * b, 0.25 + 0.12 * b, u)
            self.text(sc, 0.47, fy, "tree %d" % (b + 1), on, anchor="lm", size=SMALL)
            for i in range(len(self.images)):
                v = int(self.votes[b, i])
                ink = SIG if v else BKG
                wrong = v != int(self.labels[i])
                self.box(sc, x0 + dx * i, fy, 46, 26, soft(ink, 0.5), on, "4" if v else "9",
                         border=DIGIT_INK if wrong else ink, size=SMALL)
        m = self.meta
        end = op * smoothstep(0.66, 0.76, u)
        self.lines(sc, 0.47, 0.775, [
            "%s outline: a wrong answer" % theme.word("dark", "white"),
            "the four do not all agree on %s of held-out events" % pct(m["split_share"]),
            "(these six were picked from those)"], end, size=SMALL, step=0.042)
        self.tag(sc, MEASURED, op)

    def phase_4(self, sc, u, op):
        m = self.meta
        ch = self.pane(((0.09, 0.47), (0.41, 0.72)), (0, len(self.x) - 1), (0.0, 0.16))
        self.axes(sc, ch, op, "depth of the trees", "share of held-out events two trees "
                  "disagree on", DEPTH_TICKS, [(v, pct(v)) for v in (0.0, 0.05, 0.10, 0.15)])
        go = smoothstep(0.05, 0.5, u)
        for k, kind in enumerate(S.KINDS):
            ink = KIND_INKS[kind]
            for row in self.per_seed[k]:
                self.path(sc, ch, self.x, row, soft(ink, 0.45), op, go, width=1.2)
            self.path(sc, ch, self.x, self.disagree[k], ink, op, go, width=3.2, dot=True)
        self.key(sc, 0.10, 0.835, [(KIND_NAMES[k] + ", %d seeds" % len(m["seeds"]),
                                    KIND_INKS[k]) for k in S.KINDS], op, step=0.042,
                 size=SMALL)
        x = 0.56
        self.text(sc, x, 0.33, "swap a few training events for new ones:", op, anchor="lm",
                  strong=True, size=SMALL)
        self.text(sc, x, 0.372, "how many held-out answers change?", op, anchor="lm",
                  strong=True, size=SMALL)
        full = self.span(0.17)[0]
        sw = m["swapped"]
        top = max(max(sw[k]["changed"]) for k in S.KINDS)
        for i, share_ in enumerate(sw["two"]["shares"]):
            fy = 0.455 + 0.095 * i
            self.text(sc, x + 0.115, fy + 0.016, "%s swapped" % pct(share_, 1 if share_ < 0.01
                                                                   else 0), op, anchor="rm",
                      size=SMALL)
            for j, kind in enumerate(S.KINDS):
                v = sw[kind]["changed"][i]
                self.hbar(sc, x + 0.13, fy + 0.034 * j, full * v / top, KIND_INKS[kind], op,
                          "", pct(v, 1), u=(u - 0.4 - 0.07 * i) / 0.2, thick=15.0)
        self.tag(sc, MEASURED, op)

    def _strip(self, sc, fx0, fx1, fy, values, label, row, op, u0, u):
        """Every tree's answer for one event as dots along a line from 0 to 1."""
        self.polyline(sc, [self.P(fx0, fy), self.P(fx1, fy)], 1.4, AXIS, op)
        for v, name in ((0.0, "0: a 9"), (1.0, "1: a 4")):
            x = fx0 + (fx1 - fx0) * v
            self.polyline(sc, [self.P(x, fy - 0.008), self.P(x, fy + 0.008)], 1.4, AXIS, op)
            self.text(sc, x, fy + 0.035, name, op, size=TEXT_S)
        self.text(sc, fx0 - 0.01, fy - 0.135, label, op, anchor="lm", strong=True, size=SMALL)
        stack = {}
        shown = int(len(values) * smoothstep(u0, u0 + 0.2, u))
        for v in values[:shown]:
            key = round(float(v), 2)
            k = stack.get(key, 0)
            stack[key] = k + 1
            self.disk(sc, self.P(fx0 + (fx1 - fx0) * float(v) + 0.0075 * (k // 8),
                                 fy - 0.012 - 0.011 * (k % 8)), 2.8, VARIANCE, op, lift=1.1)
        mean = float(np.mean(values))
        x = fx0 + (fx1 - fx0) * mean
        done = op * smoothstep(u0 + 0.2, u0 + 0.26, u)
        self.polyline(sc, [self.P(x, fy - 0.10, 1.2), self.P(x, fy + 0.012, 1.2)], 2.6, BIAS,
                      done)
        self.text(sc, fx1 + 0.03, fy - 0.10, "mean answer %.2f, truth 1" % mean, done,
                  anchor="lm", size=SMALL)
        self.text(sc, fx1 + 0.03, fy - 0.057, "bias² = (1 − %.2f)² = %.2f"
                  % (mean, row["bias2"]), done, anchor="lm", ink=dark(BIAS, 0.4), size=SMALL)
        self.text(sc, fx1 + 0.03, fy - 0.014, "variance around it = %.2f" % row["variance"],
                  done, anchor="lm", ink=dark(VARIANCE, 0.3), size=SMALL)

    def phase_5(self, sc, u, op):
        m = self.meta
        one = m["one_event"]
        self.text(sc, 0.05, 0.315, "one held-out 4, %d trees from %d separate samples"
                  % (m["samples"], m["samples"]), op, anchor="lm", strong=True, size=SMALL)
        self.digit(sc, self.one_image, 0.085, 0.47, 60, op, frame=soft(SIG, 0.3))
        self._strip(sc, 0.16, 0.28, 0.51, self.one[0], "depth 1", one["shallow"], op, 0.04, u)
        self._strip(sc, 0.16, 0.28, 0.725, self.one[1], "grown to the end", one["deep"], op,
                    0.22, u)
        self.text(sc, 0.05, 0.812, "each dot: one tree's answer, the share of 4s in its leaf",
                  op, anchor="lm", size=TEXT_S)
        ch = self.pane(((0.65, 0.94), (0.40, 0.70)), (0, len(self.x) - 1), (0.0, 0.19))
        self.axes(sc, ch, op, "depth of the trees", "49 cells, all held-out events",
                  DEPTH_TICKS, [(v, "%g" % v) for v in (0.0, 0.05, 0.1, 0.15)], grid=False)
        go = smoothstep(0.45, 0.75, u)
        self.path(sc, ch, self.x, self.parts[0], DIGIT_INK, op, go, width=2.2)
        self.path(sc, ch, self.x, self.parts[1], BIAS, op, go, width=3.2)
        self.path(sc, ch, self.x, self.parts[2], VARIANCE, op, go, width=3.2)
        self.key(sc, 0.80, 0.415, [("squared error", DIGIT_INK), ("bias²", BIAS),
                                   ("variance", VARIANCE)], op, step=0.043, size=TEXT_S)
        end = op * smoothstep(0.78, 0.88, u)
        self.text(sc, 0.05, 0.875, "variance: what bagging removes", end, anchor="lm",
                  ink=dark(VARIANCE, 0.3), size=HEAD)
        self.text(sc, 0.50, 0.875, "bias: what boosting removes", end, anchor="lm",
                  ink=dark(BIAS, 0.4), size=HEAD)
        self.tag(sc, MEASURED, op)


# ====================================================================== 163: bagging
TREE_TICKS = [(v, "%d" % v) for v in (1, 3, 10, 30, 100)]
OOB = TEAL


class Bagging(Trees):
    """Sixteen events and a bootstrap resample of them, pick by pick; one jagged tree
    for each resample of the two-feature sample; their vote over the plane smoothing
    as trees are added; held-out error against the number of trees on the 49 cells;
    the variance of the vote against the formula; and the out-of-bag error."""

    TITLE = "Bagging"
    PHASES = (("the bootstrap", 11.0), ("a tree each", 8.0), ("the vote", 10.0),
              ("more trees", 9.0), ("variance", 11.0), ("out-of-bag", 9.0))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.picked_cls, self.picks = np.asarray(p0["classes"]), np.asarray(p0["picks"])
        self.counts, self.images = np.asarray(p0["counts"]), np.asarray(p0["images"])
        self.shares, self.maps = np.asarray(p1["values"]), np.asarray(p1["maps"])
        self.plane_single = np.asarray(p1["single"], float)
        self.plane_together = np.asarray(p1["together"], float)
        self.steps = np.asarray(p2["x"], float)
        self.held, self.oob = (np.asarray(v, float) for v in p2["ys"])
        self.per_seed = np.asarray(p2["per_seed"], float)
        self.var_measured, self.var_formula = (np.asarray(v, float) for v in p3["ys"])
        self.var = m["variance"]
        self.xs = [0.085 + 0.0553 * i for i in range(len(self.counts))]
        self.cells = S.cells(self.images[0], S.BLOCK).reshape(S.SIDE, S.SIDE)

    def subtitle_for(self, k):
        m = self.meta
        pk, v, held = m["picked"], self.var, m["held"]
        return [
            "digits stand in for events, as in an analysis. A bootstrap resample picks "
            "as many events as there are, with replacement: here %d of %d come up, some "
            "twice, and %d are left out. In a large sample 63%% come up"
            % (pk["unique"], pk["events"], pk["left_out"]),
            "a tree is grown to the end on each resample. With two features its answers "
            "over the plane can be drawn: every tree is jagged, each in its own way, and "
            "wrong on %s of held-out events" % pct(m["plane"]["single"], 1),
            "bagging lets them vote. As trees are added the jagged edges average into a "
            "smooth boundary, and the majority of %d trees is wrong on %s of held-out "
            "events" % (m["trees"], pct(m["plane"]["together"], 1)),
            "49 cells and %s training events: one tree is wrong on %s of held-out "
            "events, the vote of 10 on %s, of 100 on %s (mean of %d training samples). "
            "The error falls, then flattens"
            % (commas(m["unique_share"]["events"]), pct(held[1]["mean"], 1),
               pct(held[10]["mean"], 1), pct(held[100]["mean"], 1), len(m["seeds"])),
            "it flattens because the trees are alike. The variance of the vote is "
            "ρσ² + (1 − ρ)σ²/B: with the measured ρ = %.3f and σ² = %.3f it follows the "
            "measured variance within %s, and never goes below ρσ²"
            % (v["rho"], v["sigma2"], pct(v["worst_ratio"])),
            "a tree never saw the events its resample left out, so they test it for "
            "free. This out-of-bag error is %s after 100 trees against %s held-out: a "
            "little high, as each event is judged by about %d of the trees"
            % (pct(m["oob"][100]["mean"], 1), pct(held[100]["mean"], 1),
               round(m["held_at_share_of_trees"]["trees"])),
        ][k]

    # ------------------------------------------------------------------ phases
    def _sources(self, sc, fy, size, op, counts=None, mark=None):
        for i, fx in enumerate(self.xs):
            ink = SIG if self.picked_cls[i] else BKG
            out = counts is not None and counts[i] == 0
            frame = (mark if mark is not None and out else soft(ink, 0.3))
            self.digit(sc, self.images[i], fx, fy, size, op * (0.4 if out and mark is None
                                                               else 1.0), frame=frame)

    def phase_1(self, sc, u, op):
        n = len(self.picks)
        top, low = 0.395, 0.615
        self.text(sc, 0.05, 0.315, "%d training events" % n, op, anchor="lm", strong=True,
                  size=HEAD)
        self.text(sc, 0.05, 0.535, "a bootstrap resample: %d picks, each from all %d" % (n, n),
                  op, anchor="lm", strong=True, size=HEAD)
        done = [u >= 0.10 + 0.036 * j for j in range(n)]
        seen = np.bincount(self.picks[np.flatnonzero(done)], minlength=n)
        final = u > 0.72
        self._sources(sc, top, 40, op, counts=seen if final else None)
        for i, fx in enumerate(self.xs):
            if seen[i] or final:
                self.text(sc, fx, top + 0.058, "×%d" % seen[i], op, size=TEXT_S,
                          ink=dark(ASK, 0.4) if seen[i] > 1 else None)
        for j, fx in enumerate(self.xs):
            src = int(self.picks[j])
            start = 0.06 + 0.036 * j
            self.pulse(sc, (self.xs[src], top + 0.04), (fx, low - 0.04), (u - start) / 0.04,
                       ASK, op, bend=-0.03, r=5.0)
            if done[j]:
                ink = SIG if self.picked_cls[src] else BKG
                self.digit(sc, self.images[src], fx, low, 40, op, frame=soft(ink, 0.3))
            else:
                self.box(sc, fx, low, 44, 44, PAPER, 0.6 * op, border=AXIS)
        pk = self.meta["picked"]
        end = op * smoothstep(0.74, 0.82, u)
        us = self.meta["unique_share"]
        self.lines(sc, 0.05, 0.71, [
            ("%d different events came up, %d were left out, the most picked came %d times"
             % (pk["unique"], pk["left_out"], pk["most"]), dark(ASK, 0.4)),
            "expected share that comes up: 1 − (1 − 1/%d)¹⁶ = %s;  in a large sample "
            "1 − 1/e = %s" % (n, pct(pk["expected_share"]), pct(us["limit"], 1)),
            "measured on the resamples of %s events used below: %s"
            % (commas(us["events"]), pct(us["measured"], 1))], end, size=SMALL, step=0.044)
        self.stand_in(sc, op)
        self.tag(sc, exact(n), op)

    def phase_2(self, sc, u, op):
        self.text(sc, 0.05, 0.315, "one tree for each resample of 500 events, grown to "
                  "the end", op, anchor="lm", strong=True, size=HEAD)
        for i, grid in enumerate(self.maps):
            fx = 0.125 + 0.15 * i
            on = op * smoothstep(0.05 + 0.1 * i, 0.13 + 0.1 * i, u)
            self.text(sc, fx, 0.385, "tree %d" % (i + 1), on, size=SMALL)
            self.answers(sc, grid, fx, 0.535, 122, on)
            self.text(sc, fx, 0.685, "%s wrong" % pct(self.plane_single[i], 1), on,
                      size=SMALL)
        self.classes_text(sc, 0.05, 0.76, op)
        self.text(sc, 0.05, 0.805, "across: ink at the top; up: vertical spread. Blue: the "
                  "tree answers 4 there; red: 9", op, anchor="lm", size=SMALL)
        self.text(sc, 0.05, 0.85, "held-out error of the first %d trees: %s to %s"
                  % (len(self.maps), pct(self.plane_single[:len(self.maps)].min(), 1),
                     pct(self.plane_single[:len(self.maps)].max(), 1)),
                  op * smoothstep(0.7, 0.8, u), anchor="lm", size=SMALL)
        self.tag(sc, MEASURED, op)

    def classes_text(self, sc, fx, fy, op):
        self.text(sc, fx, fy, "every tree divides the same plane differently", op,
                  anchor="lm", ink=dark(ASK, 0.4), size=SMALL)

    def phase_3(self, sc, u, op):
        last = len(self.steps) - 1
        k = int(round(last * smoothstep(0.06, 0.78, u)))
        b = int(self.steps[k])
        self.text(sc, 0.05, 0.315, "the share of %d tree%s answering 4" % (b, "" if b == 1
                                                                            else "s"),
                  op, anchor="lm", strong=True, size=HEAD)
        self.answers(sc, self.shares[k], 0.19, 0.58, 208, op)
        self.text(sc, 0.19, 0.815, "across: ink at the top", op, size=TEXT_S)
        self.text(sc, 0.05, 0.868, "pale: the trees are divided", op, anchor="lm", size=SMALL)
        self.text(sc, 0.318, 0.58, "up: spread", op, anchor="lm", size=TEXT_S)
        ch = self.pane(((0.52, 0.93), (0.41, 0.70)), (1, self.steps[-1]), (0.10, 0.16),
                       xlog=True)
        self.axes(sc, ch, op, "trees voting", "held-out events the majority gets wrong",
                  TREE_TICKS, [(v, pct(v)) for v in (0.10, 0.12, 0.14, 0.16)])
        single = float(self.plane_single.mean())
        self.hline(sc, ch, single, GREY, op)
        self.text(sc, 0.93, ch.fy(single) - 0.022, "one tree, on average", op, anchor="rm",
                  size=TEXT_S)
        self.path(sc, ch, self.steps, self.plane_together, HELD, op, k / last, width=3.2,
                  dot=True)
        self.text(sc, 0.52, 0.845, "%d tree%s: %s wrong" % (b, "" if b == 1 else "s",
                                                           pct(self.plane_together[k], 1)),
                  op, anchor="lm", ink=dark(HELD, 0.3), size=BIG)
        self.tag(sc, MEASURED, op)

    def phase_4(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.05, 0.315, "now 49 cells an event", op, anchor="lm", strong=True,
                  size=HEAD)
        self.digit(sc, self.images[0], 0.105, 0.46, 96, op, frame=soft(SIG, 0.3))
        self.grid(sc, self.cells, 0.245, 0.46, 96, 96, op, vmax=1.0, hot=SIG)
        self.arrow(sc, (0.162, 0.46), (0.188, 0.46), AXIS, op)
        self.lines(sc, 0.05, 0.60, ["%s training events" % commas(m["unique_share"]["events"]),
                                    "%d resamples, %d trees" % (m["trees"], m["trees"]),
                                    "repeated on %d training" % len(m["seeds"]),
                                    "samples (thin lines)"], op, size=SMALL, step=0.044)
        ch = self.pane(((0.42, 0.93), (0.40, 0.74)), (1, self.steps[-1]), (0.03, 0.125),
                       xlog=True)
        self.axes(sc, ch, op, "trees voting", "held-out events the majority gets wrong",
                  TREE_TICKS, [(v, pct(v)) for v in (0.04, 0.06, 0.08, 0.10, 0.12)])
        go = smoothstep(0.08, 0.7, u)
        for row in self.per_seed[0]:
            self.path(sc, ch, self.steps, row, soft(HELD, 0.45), op, go, width=1.1)
        self.path(sc, ch, self.steps, self.held, HELD, op, go, width=3.4, dot=True)
        single = m["single"]["mean"]
        self.hline(sc, ch, single, GREY, op)
        self.text(sc, 0.93, ch.fy(single) - 0.022, "one tree, on average: %s"
                  % pct(single, 1), op, anchor="rm", size=TEXT_S)
        end = op * smoothstep(0.72, 0.82, u)
        self.text(sc, 0.42, 0.875, "1 tree %s,  10 trees %s,  100 trees %s"
                  % tuple(pct(m["held"][b]["mean"], 1) for b in (1, 10, 100)), end,
                  anchor="lm", ink=dark(HELD, 0.3), size=HEAD)
        self.tag(sc, MEASURED, op)

    def phase_5(self, sc, u, op):
        v = self.var
        rows = [
            ("why it flattens: the trees are alike", None, True, HEAD),
            ("one tree's answer for an event changes", None, False, SMALL),
            ("from sample to sample: variance σ² = %.3f" % v["sigma2"], None, False, SMALL),
            ("two trees of a bag change together:", None, False, SMALL),
            ("correlation ρ = %.3f" % v["rho"], None, False, SMALL),
            ("variance of the vote of B trees", None, True, SMALL),
            ("ρσ² + (1 − ρ)σ²/B", dark(ASK, 0.4), False, BIG),
        ]
        y = 0.315
        for i, (row, ink, strong, size) in enumerate(rows):
            self.text(sc, 0.05, y, row, op * smoothstep(0.03 + 0.05 * i, 0.08 + 0.05 * i, u),
                      anchor="lm", ink=ink, strong=strong, size=size)
            y += 0.052 if i in (0, 2, 4, 5) else 0.042
        ch = self.pane(((0.53, 0.93), (0.40, 0.72)), (1, self.steps[-1]), (0.0, 0.08),
                       xlog=True)
        self.axes(sc, ch, op, "trees voting, B", "variance of the share answering 4",
                  TREE_TICKS, [(v_, "%g" % v_) for v_ in (0.0, 0.02, 0.04, 0.06, 0.08)],
                  grid=False)
        go = smoothstep(0.4, 0.75, u)
        self.hline(sc, ch, v["floor"], ASK, op * smoothstep(0.36, 0.42, u), width=2.0)
        self.path(sc, ch, self.steps, self.var_formula, DIGIT_INK, op, go, width=2.2)
        shown = int(np.ceil(len(self.steps) * go))
        self.dots(sc, ch, self.steps[:shown], self.var_measured[:shown], MAGENTA, op, r=4.6,
                  lift=1.4)
        self.key(sc, 0.70, 0.43, [("measured, %d samples" % len(self.meta["seeds"]), MAGENTA),
                                  ("the formula", DIGIT_INK), ("its floor, ρσ²", ASK)], op,
                 step=0.043, size=TEXT_S)
        end = op * smoothstep(0.78, 0.86, u)
        self.text(sc, 0.05, 0.84, "B = 1: %.3f     B = 100: %.4f" % (self.var_formula[0],
                                                                      self.var_formula[-1]),
                  end, anchor="lm", size=SMALL)
        self.text(sc, 0.05, 0.882, "never below ρσ² = %.4f, however many trees" % v["floor"],
                  end, anchor="lm", ink=dark(ASK, 0.4), size=SMALL)
        self.tag(sc, MEASURED, op)

    def phase_6(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.05, 0.315, "left out of the resample: never seen by its tree", op,
                  anchor="lm", strong=True, size=HEAD)
        self._sources(sc, 0.395, 40, op, counts=self.counts, mark=OOB)
        self.text(sc, 0.05, 0.47, "framed: out of the bag for this tree. An event is judged "
                  "only by the trees that left it out", op, anchor="lm", size=SMALL)
        ch = self.pane(((0.09, 0.52), (0.57, 0.80)), (1, self.steps[-1]), (0.03, 0.125),
                       xlog=True)
        self.axes(sc, ch, op, "trees in the bag", "share wrong, 49 cells",
                  TREE_TICKS, [(v, pct(v)) for v in (0.04, 0.08, 0.12)])
        go = smoothstep(0.1, 0.6, u)
        self.path(sc, ch, self.steps, self.held, HELD, op, go, width=3.0)
        self.path(sc, ch, self.steps, self.oob, OOB, op, go, width=3.4, dot=True)
        x = 0.58
        self.key(sc, x + 0.008, 0.575, [("out-of-bag, from the training events", OOB),
                                        ("held-out events", HELD)], op, step=0.044,
                 size=SMALL)
        end = op * smoothstep(0.62, 0.72, u)
        gap = m["oob_minus_held"]
        lag = m["held_at_share_of_trees"]
        self.lines(sc, x, 0.68, [
            "100 trees: out-of-bag %s, held-out %s" % (pct(m["oob"][100]["mean"], 1),
                                                       pct(m["held"][100]["mean"], 1)),
            "an event is out of the bag for 37% of",
            "the trees: %d of 100 judge it, and %d trees" % (round(lag["trees"]),
                                                             round(lag["trees"])),
            "get %s of held-out events wrong" % pct(lag["held"], 1),
            "sample by sample they differ by %s" % pct(gap["mean_abs"], 1)], end,
            size=SMALL, step=0.043)
        self.tag(sc, MEASURED, op)


# ====================================================================== 164: random forest
MTRY_TICKS = [(float(v), "%d" % v) for v in S.MTRY]
PERM = TEAL


def cell_name(cell):
    """A cell of the 7 x 7 read-out by its row and column, counted from 1."""
    return "row %d, column %d" % (cell // S.SIDE + 1, cell % S.SIDE + 1)


def where(cell):
    return cell // S.SIDE, cell % S.SIDE


class RandomForest(Trees):
    """The cells drawn for the first nodes of a tree and the best of each handful
    taken; the cell each tree starts from, with all, seven and one tried, and trees
    growing less alike and weaker; the forest's error against the size of the handful;
    importance of each cell by gain and by permutation drawn on a digit; and where the
    two measures part."""

    TITLE = "A random forest"
    PHASES = (("a handful of cells", 12.0), ("less alike", 10.0), ("how many to try", 9.0),
              ("which cells matter", 12.0), ("two measures", 10.0))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.tried, self.tried_gain = np.asarray(p0["values"]).astype(int), np.asarray(p0["gains"])
        self.taken, self.cut = np.asarray(p0["taken"]), np.asarray(p0["cut"])
        self.counts = np.asarray(p0["counts"])
        self.images = np.asarray(p0["images"])
        self.cells = S.cells(self.images[0], S.BLOCK).reshape(S.SIDE, S.SIDE)
        self.mtry = np.asarray(p1["x"], float)
        self.forest, self.single, self.alike = (np.asarray(v, float) for v in p1["ys"])
        self.per_seed, self.first = np.asarray(p1["per_seed"]), np.asarray(p1["first"])
        self.gain, self.perm = np.asarray(p2["values"], float), np.asarray(p3["values"], float)
        self.share = np.asarray(p3["share"], float)
        self.imp = m["importance"]

    def subtitle_for(self, k):
        m, imp = self.meta, self.imp
        c, sgl, f, best = m["correlation"], m["single"], m["forest"], m["best"]
        return [
            "digits stand in for events, as in an analysis, each read as 49 cells. A "
            "random forest is bagging with one change: at every split only a random "
            "handful of cells, here %d, may be tried, and the best of them is taken"
            % m["read"],
            "with all 49 tried, each of 100 trees asks about the same cell first; with "
            "%d tried they start from %d different cells. Trees become less alike "
            "(correlation %.2f down to %.2f) and each weaker (%s wrong up to %s)"
            % (m["read"], m["first_cells"][m["read"]], c[49]["mean"], c[1]["mean"],
               pct(sgl[49]["mean"]), pct(sgl[1]["mean"])),
            "the forest is best in between: %s of held-out events wrong with %d cells "
            "tried, against %s with one and %s with all 49, which is bagging. It is so "
            "in %s of %d seeds"
            % (pct(best["held"], 1), best["mtry"], pct(f[1]["mean"], 1), pct(f[49]["mean"], 1),
               "each" if m["best_beats_ends_in_every_seed"] else "some", len(m["seeds"])),
            "which cells does the forest use? Add up the gain of every split on a cell; "
            "or shuffle that cell's readings among held-out events and watch the error "
            "rise: from %s to %s for the cell at the top centre"
            % (pct(imp["base"], 1), pct(imp["base"] + imp["top_rise"][0], 1)),
            "the two agree on the top cell, not below it: %d cells share %s of the gain, "
            "yet shuffling any one of them costs under 0.1 point. A neighbouring cell "
            "says much the same, so the forest does without"
            % (imp["quiet_cells"], pct(imp["quiet_gain"])),
        ][k]

    def _on_digit(self, sc, values, fx, fy, size, op, ink, image=1, ring=()):
        """A 7 x 7 map of importance drawn back on a digit: each cell tinted by its
        value, the digit's ink dark on top."""
        v = np.asarray(values, float)
        v = np.sqrt(np.clip(v / max(float(v.max()), 1e-12), 0.0, 1.0))
        v = np.kron(v, np.ones((S.BLOCK, S.BLOCK)))[..., None]
        tint = PAPER + (np.asarray(ink) - PAPER) * v
        img = np.clip(self.images[image], 0.0, 1.0)[..., None]
        tex = tint * (1 - 0.8 * img) + np.asarray(DIGIT_INK) * 0.8 * img
        c = self.P(fx, fy)
        self.rect(sc, c, size + 8, size + 8, PAPER, op, border=AXIS, border_w=1.2, radius=5,
                  lift=0.1)
        draw_image(self, sc, c, tex.astype(np.float32), size, size, op, grid=0)
        for cell in ring:
            r, k = where(cell)
            self.rect(sc, self.shift(c, (k + 0.5 - S.SIDE / 2) * size / S.SIDE,
                                     (r + 0.5 - S.SIDE / 2) * size / S.SIDE),
                      size / S.SIDE, size / S.SIDE, ASK, op, border=ASK, border_w=2.4,
                      radius=2, lift=0.9, fill_op=0.0)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        n = len(self.tried)
        s = float(np.clip((u - 0.04) / 0.92, 0.0, 0.9999)) * n
        j, w = int(s), s - int(s)
        self.text(sc, 0.05, 0.315, "an event, read as 7 × 7 cells", op, anchor="lm",
                  strong=True, size=HEAD)
        self.digit(sc, self.images[0], 0.105, 0.525, 100, op, frame=soft(SIG, 0.3))
        self.arrow(sc, (0.165, 0.525), (0.19, 0.525), AXIS, op)
        cells, gains = self.tried[j], self.tried_gain[j]
        shown = int(np.ceil(len(cells) * smoothstep(0.05, 0.45, w)))
        done = smoothstep(0.55, 0.65, w)
        took = int(self.taken[j])
        self.grid(sc, self.cells, 0.295, 0.525, 161, 161, op, vmax=1.0, hot=SIG,
                  ring=[where(c) for c in cells[:shown]], ring_ink=ASK)
        if done > 0:
            r, k = where(took)
            size = 161 / S.SIDE
            self.rect(sc, self.shift(self.P(0.295, 0.525), (k + 0.5 - S.SIDE / 2) * size,
                                     (r + 0.5 - S.SIDE / 2) * size), size + 4, size + 4, GAIN,
                      op * done, border=GAIN, border_w=3.4, radius=2, lift=1.2, fill_op=0.0)
        self.text(sc, 0.05, 0.735, "ringed: the %d cells drawn for this node" % len(cells), op,
                  anchor="lm", ink=dark(ASK, 0.4), size=SMALL)
        self.text(sc, 0.05, 0.778, "a new handful is drawn at every node", op, anchor="lm",
                  size=SMALL)
        x = 0.47
        names = ("node 1, the root", "node 2, its yes side", "node 3, its no side")
        self.text(sc, x, 0.315, "%s: %s events" % (names[j], commas(self.counts[0, j])), op,
                  anchor="lm", strong=True, size=HEAD)
        self.text(sc, x, 0.36, "the best gain each of its %d cells could give" % len(cells),
                  op, anchor="lm", size=SMALL)
        full = self.span(0.20)[0]
        top = float(self.tried_gain.max())
        for i, (c, g) in enumerate(zip(cells[:shown], gains[:shown])):
            best = c == took and done > 0
            self.hbar(sc, x + 0.165, 0.415 + 0.05 * i, max(full * g / top, 1.0),
                      GAIN if best else ASK, op, cell_name(int(c)), "%.3f" % g, thick=17.0)
        thr = (self.cut[j] + 1) / S.CELL_BINS
        self.text(sc, x, 0.79, "taken: is the cell in %s ≤ %.2f?" % (cell_name(took), thr),
                  op * done, anchor="lm", ink=dark(GAIN, 0.3), size=HEAD)
        self.stand_in(sc, op)
        self.tag(sc, TRAINING, op)

    def phase_2(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.05, 0.315, "the cell each of %d trees asks about first" % m["trees"],
                  op, anchor="lm", strong=True, size=HEAD)
        for i, tried in enumerate((49, m["read"], 1)):
            fx = 0.125 + 0.165 * i
            on = op * smoothstep(0.04 + 0.12 * i, 0.14 + 0.12 * i, u)
            first = self.first[S.MTRY.index(tried)]
            count = np.bincount(first, minlength=S.CELLS).reshape(S.SIDE, S.SIDE)
            self.text(sc, fx, 0.38, "all 49 tried" if tried == 49 else "%d tried" % tried, on,
                      strong=True, size=SMALL)
            self.grid(sc, np.sqrt(count / count.max()), fx, 0.53, 120, 120, on, vmax=1.0,
                      hot=ASK)
            self.text(sc, fx, 0.68, "%d different" % m["first_cells"][tried], on, size=SMALL)
            self.text(sc, fx, 0.718, "cell%s" % ("" if m["first_cells"][tried] == 1 else "s"),
                      on, size=SMALL)
        ch = self.pane(((0.62, 0.93), (0.40, 0.70)), (1, 49), (0.0, 0.34), xlog=True)
        self.axes(sc, ch, op, "cells tried at each split", "", MTRY_TICKS,
                  [(v, "%g" % v) for v in (0.0, 0.1, 0.2, 0.3)])
        go = smoothstep(0.4, 0.8, u)
        self.path(sc, ch, self.mtry, self.alike, ASK, op, go, width=3.2, dot=True)
        self.path(sc, ch, self.mtry, self.single, TRAIN, op, go, width=3.2, dot=True)
        self.key(sc, 0.63, 0.34, [("correlation between two trees", ASK)], op, size=TEXT_S)
        self.key(sc, 0.63, 0.378, [("share one tree gets wrong", TRAIN)], op, size=TEXT_S)
        c, sgl = m["correlation"], m["single"]
        end = op * smoothstep(0.8, 0.88, u)
        self.text(sc, 0.05, 0.80, "fewer cells tried: trees less alike, correlation %.2f "
                  "down to %.2f" % (c[49]["mean"], c[1]["mean"]), end, anchor="lm",
                  ink=dark(ASK, 0.4), size=SMALL)
        self.text(sc, 0.05, 0.843, "but each tree weaker: %s of held-out events wrong, up "
                  "to %s" % (pct(sgl[49]["mean"], 1), pct(sgl[1]["mean"], 1)), end,
                  anchor="lm", ink=dark(TRAIN, 0.3), size=SMALL)
        self.tag(sc, MEASURED, op)

    def phase_3(self, sc, u, op):
        m = self.meta
        best = m["best"]
        ch = self.pane(((0.09, 0.52), (0.40, 0.72)), (1, 49), (0.035, 0.065), xlog=True)
        self.axes(sc, ch, op, "cells tried at each split", "held-out events a forest of "
                  "%d trees gets wrong" % m["trees"], MTRY_TICKS,
                  [(v, pct(v)) for v in (0.04, 0.05, 0.06)])
        go = smoothstep(0.05, 0.55, u)
        for row in self.per_seed[:, :, 0]:
            self.path(sc, ch, self.mtry, row, soft(HELD, 0.45), op, go, width=1.2)
        self.path(sc, ch, self.mtry, self.forest, HELD, op, go, width=3.4)
        pick = op * smoothstep(0.56, 0.64, u)
        self.disk(sc, self.P(ch.fx(best["mtry"]), ch.fy(best["held"])), 6.5, HELD, pick,
                  ring=ASK, ring_w=3.0, lift=1.5)
        self.text(sc, 0.09, 0.845, "thin: each of %d seeds; ringed: the best, %d tried"
                  % (len(m["seeds"]), best["mtry"]), op, anchor="lm", size=SMALL)
        x = 0.60
        self.text(sc, x, 0.33, "held-out events wrong", op, anchor="lm", strong=True,
                  size=HEAD)
        rows = (("one tree alone", GREY, m["single"][49]["mean"]),
                ("1 cell tried", soft(HELD, 0.3), m["forest"][1]["mean"]),
                ("all 49: bagging", soft(HELD, 0.3), m["forest"][49]["mean"]),
                ("%d tried" % m["read"], soft(HELD, 0.3), m["usual"]),
                ("%d tried" % best["mtry"], HELD, best["held"]))
        full = self.span(0.16)[0]
        for r, (name, ink, v) in enumerate(rows):
            self.hbar(sc, x + 0.135, 0.41 + 0.07 * r, full * v / rows[0][2], ink, op, name,
                      pct(v, 1), u=(u - 0.3 - 0.07 * r) / 0.2, thick=22.0)
        self.text(sc, x, 0.79, "between one cell and all of them", pick, anchor="lm",
                  ink=dark(HELD, 0.3), size=SMALL)
        self.tag(sc, MEASURED, op)

    def phase_4(self, sc, u, op):
        imp = self.imp
        top = imp["top"][0]
        a = op
        self.text(sc, 0.17, 0.315, "by total gain", a, strong=True, size=HEAD)
        self._on_digit(sc, self.gain, 0.17, 0.555, 196, a, ASK, ring=(top,))
        self.text(sc, 0.17, 0.795, "share of all the impurity", a, size=TEXT_S)
        self.text(sc, 0.17, 0.833, "its splits removed: %s at the top"
                  % pct(imp["top_gain"][0]), a, size=TEXT_S)
        b = op * smoothstep(0.3, 0.4, u)
        self.text(sc, 0.49, 0.315, "by permutation", b, strong=True, size=HEAD)
        full = smoothstep(0.62, 0.74, u)
        only = np.zeros_like(self.perm)
        only.ravel()[top] = self.perm.ravel()[top]
        self._on_digit(sc, np.clip(only + (self.perm - only) * full, 0, None), 0.49, 0.555, 196,
                       b, PERM, ring=(top,))
        self.text(sc, 0.49, 0.795, "rise of the held-out error", b, size=TEXT_S)
        self.text(sc, 0.49, 0.833, "when the cell is shuffled", b, size=TEXT_S)
        x = 0.68
        rise = imp["top_rise"][0] * smoothstep(0.42, 0.6, u)
        self.lines(sc, x, 0.40, ["shuffle the ringed cell", "among the held-out events:",
                                 "the forest's error goes from"], b, size=SMALL, step=0.043)
        self.text(sc, x, 0.545, "%s to %s" % (pct(imp["base"], 1), pct(imp["base"] + rise, 1)),
                  b, anchor="lm", ink=dark(PERM, 0.3), size=BIG)
        self.lines(sc, x, 0.62, ["then every cell in turn", "(forest with %d cells tried,"
                                 % self.meta["read"], "mean of %d seeds)"
                                 % len(self.meta["seeds"])],
                   op * smoothstep(0.62, 0.7, u), size=SMALL, step=0.043)
        self.tag(sc, MEASURED, op)

    def phase_5(self, sc, u, op):
        imp = self.imp
        gain, share_ = self.gain.ravel(), self.share.ravel()
        ch = self.pane(((0.09, 0.43), (0.40, 0.72)), (0.0, 0.2), (0.0, 0.4))
        self.axes(sc, ch, op, "share of the total gain", "share of the total rise when "
                  "shuffled", [(v, pct(v)) for v in (0.05, 0.1, 0.15, 0.2)],
                  [(v, pct(v)) for v in (0.0, 0.1, 0.2, 0.3, 0.4)])
        self.polyline(sc, [self.P(ch.fx(0), ch.fy(0), 0.5), self.P(ch.fx(0.2), ch.fy(0.2), 0.5)],
                      1.6, AXIS, 0.8 * op)
        quiet = (gain > 0.01) & (self.perm.ravel() < 0.001)
        shown = int(np.ceil(len(gain) * smoothstep(0.04, 0.4, u)))
        for k in np.argsort(-gain)[:shown]:
            plain = not quiet[k] and k != imp["top"][0]
            ink = GREY if plain else (ASK if quiet[k] else PERM)
            self.disk(sc, self.P(ch.fx(gain[k]), ch.fy(share_[k])), 3.6 if plain else 5.0,
                      ink, op, lift=1.3)
        self.text(sc, 0.09, 0.845, "each dot a cell; on the line the two measures agree", op,
                  anchor="lm", size=TEXT_S)
        on = op * smoothstep(0.45, 0.55, u)
        self._on_digit(sc, np.where(quiet, gain, 0.0).reshape(S.SIDE, S.SIDE), 0.555, 0.50,
                       150, on, ASK)
        self.text(sc, 0.555, 0.675, "amber: the %d cells" % imp["quiet_cells"], on,
                  ink=dark(ASK, 0.4), size=TEXT_S)
        self.text(sc, 0.555, 0.712, "below the line", on, ink=dark(ASK, 0.4), size=TEXT_S)
        x = 0.68
        self.lines(sc, x, 0.36, [
            ("top cell: %s of the gain," % pct(imp["top_share_gain"]), dark(PERM, 0.3)),
            ("%s of the rise" % pct(imp["top_share_permutation"]), dark(PERM, 0.3)),
            ("%d cells with over 1%% of the" % imp["quiet_cells"], dark(ASK, 0.4)),
            ("gain each, %s together:" % pct(imp["quiet_gain"]), dark(ASK, 0.4)),
            ("shuffling one costs under", dark(ASK, 0.4)),
            ("0.1 point of error", dark(ASK, 0.4)),
            "gain counts every use of",
            "a cell; permutation asks if",
            "the forest could do without it"],
            op * smoothstep(0.6, 0.7, u), size=SMALL, step=0.043)
        self.tag(sc, MEASURED, op)


ANIMATIONS = {
    "tree-splits-and-impurity": SplitsAndImpurity,
    "tree-pruning-and-depth": PruningAndDepth,
    "tree-instability": Instability,
    "bagging": Bagging,
    "random-forest": RandomForest,
}

"""Animations 172-173 of the "next" catalogue (see :mod:`voxanim.mech`): entropy and the
Gini index in a decision tree's nodes, on all ten MNIST digits.

* :class:`EntropyAndGiniByHand`: the root of a tree as a bag of 10,000 training digits;
  its entropy and its Gini index worked out term by term from the ten counts; what each
  one counts (pairs of digits that differ; the yes/no questions of a Huffman code); both
  falling after the first question and further down; the two-class curves, and in nats
  entropy above Gini.
* :class:`EntropyDownATree`: a tree grown by information gain, its top levels coloured by
  entropy; the entropy left at every depth and the gains that remove it; the tree read as
  a probability model, whose training loss in nats is exactly the entropy left; the
  held-out digits it gives probability zero, smoothing, and the held-out loss against
  depth.

Their neighbour :class:`voxanim.mech_trees.SplitsAndImpurity` scores the thresholds of
two features for 4s against 9s; these two measure impurity itself, over all ten digits,
and follow it through a whole tree. They draw the studies of :mod:`voxanim.study_info`
(and its cached tree); every number on screen is the study's, or is worked out here from
its cached tree and counts with its own functions. Every phase says, bottom right, what
kind of evidence it shows: exact arithmetic on counts, or a measurement on MNIST.
"""

from __future__ import annotations

import numpy as np

from . import color as C
from . import study_info as S
from .diagram import draw_image, shrink_columns
from .easing import smoothstep
from .mech import (
    AMBER,
    AXIS,
    DIGIT_INK,
    GREEN,
    MAGENTA,
    OCHRE,
    PAPER,
    TEAL,
    VIOLET,
    dark,
    soft,
)
from .mech_families import TEN, WHITE
from .mech_phys_apps import Phys, commas, pct

ENT, GIN, ERR = MAGENTA, AMBER, OCHRE      # entropy, Gini, error: as in mech_trees
GAIN, HELD, TRAIN, ASK = GREEN, GREEN, VIOLET, AMBER
BIG, HEAD, SMALL = 0.036, 0.033, 0.030
LN2 = float(np.log(2.0))
MEASURED = "measured on MNIST: %s held-out digits"
TRAINING = "measured on MNIST: %s training digits"


def exact(n):
    """The tag of a phase that works something out from the counts of ``n`` digits."""
    return "exact arithmetic on the counts of %s training digits" % commas(n)


def is_evidence(text):
    """True for the tag every phase carries."""
    return text.startswith(("exact arithmetic on ", "measured on MNIST"))


def bits(v):
    return "%.3f bits" % float(v)


def threshold(cut):
    """The reading a node's question compares with: left (yes) is ``bin <= cut``, which is
    ``reading <= CUTS[cut]``."""
    return float(S.CUTS[int(cut)])


def ask(cut):
    """A node's question about its cell, in words."""
    return "ink ≤ %.3g?" % threshold(cut)


def cell_name(cell):
    """A cell of the 7 x 7 read-out by its row and column, counted from 1."""
    side = 28 // S.BLOCK
    return "row %d, column %d" % (int(cell) // side + 1, int(cell) % side + 1)


def code_words(lengths):
    """Codewords with these lengths (0: no codeword), assigned in the canonical way
    (shortest first, then by digit): a prefix code, so with a Huffman code's lengths it is
    a Huffman code."""
    words, code, prev = [""] * len(lengths), 0, 0
    for k in sorted((k for k in range(len(lengths)) if lengths[k] > 0),
                    key=lambda k: (lengths[k], k)):
        code <<= int(lengths[k]) - prev
        prev = int(lengths[k])
        words[k] = format(code, "0%db" % prev)
        code += 1
    return words


def running_share(counts, n=S.PAIRS, seed=S.PAIR_SEED):
    """The share of the first ``k`` pairs that differ, for every ``k``: the same draws as
    :func:`voxanim.study_info.pairs_differ` (so the last value is its share), and the
    first pairs themselves."""
    rng = np.random.default_rng(seed)
    p = S.proportions(counts)
    a = rng.choice(S.CLASSES, n, p=p)
    b = rng.choice(S.CLASSES, n, p=p)
    differ = np.cumsum(a != b) / np.arange(1, n + 1)
    return differ, a[:12], b[:12]


def entropy_ink(h, top=S.LOG2_10):
    """The colour of a node with entropy ``h`` bits: pale when nearly pure, deep magenta
    at an even mix of the ten digits."""
    return tuple(float(v) for v in C.mix(PAPER, ENT, entropy_tint(h, top)))


def entropy_tint(h, top=S.LOG2_10):
    """How far :func:`entropy_ink` moves from the paper towards magenta (works on arrays)."""
    return 0.04 + 0.62 * np.clip(np.asarray(h, float) / top, 0.0, 1.0)


def preorder(tree):
    """Each node's place in a depth-first walk, yes side first."""
    rank, stack, order = np.zeros(len(tree["feature"]), np.int64), [0], 0
    while stack:
        t = stack.pop()
        rank[t] = order
        order += 1
        if tree["feature"][t] >= 0:
            stack += [int(tree["right"][t]), int(tree["left"][t])]
    return rank


def depth_map(tree, xb, depths):
    """``(depths, digits)``: the entropy (bits) of the node each training digit sits in
    with the tree cut at each depth, the digits in the tree's order (so every node's
    digits lie side by side, the yes side to the left). A row's mean is the entropy left
    at that depth."""
    H = S.impurity_of(tree["counts"], "entropy")
    leaf = S.path_nodes(tree, xb, max(depths))
    order = np.argsort(preorder(tree)[leaf], kind="stable")
    return np.array([H[S.path_nodes(tree, xb, d)[order]] for d in depths])


def node_layout(tree, depth, x0, x1):
    """``{node: fx}`` for the nodes down to ``depth``: the deepest row's nodes (and leaves
    above it) evenly spread, every other node midway between its children."""
    order, stack = [], [0]
    while stack:
        t = stack.pop()
        if tree["feature"][t] >= 0 and tree["depth"][t] < depth:
            stack += [int(tree["right"][t]), int(tree["left"][t])]
        else:
            order.append(t)
    fx = {t: x0 + (i + 0.5) * (x1 - x0) / len(order) for i, t in enumerate(order)}
    for t in sorted(range(len(tree["feature"])), key=lambda t: -tree["depth"][t]):
        if tree["depth"][t] < depth and tree["feature"][t] >= 0 and int(tree["left"][t]) in fx:
            fx[t] = (fx[int(tree["left"][t])] + fx[int(tree["right"][t])]) / 2
    return fx


class InfoTrees(Phys):
    """What the two animations share: the ten digits as coloured chips, a node drawn as
    the ten bars of its shares, the evidence tag and the cell a question asks about."""

    def chip(self, sc, digit, fx, fy, op, size=22.0):
        """A digit as a chip in its colour."""
        self.box(sc, fx, fy, size, size - 2, TEN[int(digit)], op, str(int(digit)), ink=WHITE,
                 size=0.029, radius=4)

    def shares(self, sc, counts, fx, fy, w, h, op, frame=None, fill=None):
        """A node as the ten bars of its shares, one per digit in its colour, the full
        height of the box a share of 1 and each bar as high as the square root of its
        share (so a tenth still shows): an even mix is ten low bars, a pure node one bar
        to the top."""
        if op <= 0.01:
            return
        p = S.proportions(counts)
        self.rect(sc, self.P(fx, fy), w + 8, h + 8, fill if fill is not None else PAPER, op,
                  border=frame if frame is not None else AXIS, border_w=1.8, radius=4,
                  lift=0.3)
        bw = w / S.CLASSES
        base = self.shift(self.P(fx, fy), -w / 2, h / 2)
        for k, v in enumerate(np.sqrt(p) * h):
            if v < 0.6:
                continue
            self.rect(sc, self.shift(base, (k + 0.5) * bw, -v / 2), max(bw - 1.2, 1.0), v,
                      TEN[k], op, radius=0.8, lift=0.55)

    def cell_on_digit(self, sc, img, cell, fx, fy, size, op):
        """A digit with the cell a question reads ringed in amber."""
        side = 28 // S.BLOCK
        self.digit(sc, img, fx, fy, size, op, frame=AXIS)
        r, k = divmod(int(cell), side)
        c = self.P(fx, fy)
        self.rect(sc, self.shift(c, (k + 0.5 - side / 2) * size / side,
                                 (r + 0.5 - side / 2) * size / side),
                  size / side + 2, size / side + 2, ASK, op, border=ASK, border_w=2.4, radius=2,
                  lift=0.9, fill_op=0.0)


# ====================================================================== 172 by hand
ROW0, STEP = 0.392, 0.0445       # the ten rows of the table, digit 0 at the top
CHIP_X, COL_X, BAR_X = 0.075, 0.172, 0.192
BAR_PX = 150.0


def row(k):
    return ROW0 + STEP * k


class EntropyAndGiniByHand(InfoTrees):
    """The root of a tree as ten counts; entropy and Gini term by term against their
    maxima; what each counts (pairs that differ, a Huffman code's questions); the first
    question and a node further down; the two-class curves and nats against Gini."""

    TITLE = "Entropy and Gini by hand"
    PHASES = (("a node", 8.0), ("entropy", 10.0), ("Gini", 12.0), ("questions", 10.0),
              ("a split", 11.0), ("two classes", 9.0))

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        self.root, self.left, self.right, self.deep = m["nodes"]
        r = self.root
        self.n = r["n"]
        self.counts = np.asarray(r["counts"], float)
        self.p = np.asarray(r["shares"], float)
        self.hterms = np.asarray(r["entropy_terms"], float)
        self.gterms = np.asarray(r["gini_terms"], float)
        self.code = code_words(r["huffman_lengths"])
        self.deep_code = code_words(self.deep["huffman_lengths"])
        runs = S.tree_runs()
        self.images = np.asarray(runs["root_digits"], float)
        self.labels = S.tree_data()[1][:len(self.images)]
        self.yes_path = [int(t) for t in runs["hand_path"]]
        T = S.tree_of("entropy")
        self.path_counts = [np.asarray(T["counts"][t], float) for t in self.yes_path]
        self.differ, self.pa, self.pb = running_share(self.counts)
        two = s.panels[2].data
        self.two_p, self.two = np.asarray(two["x"], float), np.asarray(two["ys"], float)
        self.most = int(np.argmax(self.deep["counts"]))
        self.often, self.rare = int(np.argmax(self.p)), int(np.argmin(self.p))

    def subtitle_for(self, k):
        m, r, a, b, d = self.meta, self.root, self.left, self.right, self.deep
        return [
            "a node of a tree holds digits of every kind. At the root are all %s training "
            "digits, close to an even mix; impurity is worked out from these ten counts alone"
            % commas(self.n),
            "entropy adds −p log₂ p over the ten shares: %.3f bits here, all but the %.3f of "
            "an even mix. −log₂ p is how surprising one digit is; entropy is its mean"
            % (r["entropy"], m["max_entropy"]),
            "Gini adds p(1 − p): %.3f here, its most is %g. It is the chance that two digits "
            "drawn from the node differ: of %s pairs drawn, %.4f ± %.4f differ"
            % (r["gini"], m["max_gini"], commas(m["pairs"]), r["pairs_differ"],
               r["pairs_error"]),
            "entropy counts yes/no questions: a Huffman code for these counts needs %.3f a "
            "digit on average. No code needs fewer than the entropy, %.3f, and this one is "
            "within one bit" % (r["huffman_mean"], r["entropy"]),
            "the first question splits the root %s | %s: %.3f and %.3f bits, %.3f weighted, a "
            "gain of %.3f. %d questions down, %s digits, %s of them %ds: %.3f bits, Gini %.3f"
            % (commas(a["n"]), commas(b["n"]), a["entropy"], b["entropy"],
               m["weighted_children_entropy"], m["first_gain"], d["depth"], commas(d["n"]),
               commas(d["counts"][self.most]), self.most, d["entropy"], d["gini"]),
            "for two kinds the three measures have one shape: 0 when pure, highest at an "
            "even mix. In nats, entropy is never below Gini: term by term, −ln p ≥ 1 − p",
        ][k]

    # ------------------------------------------------------------------ the table
    def _chips(self, sc, op):
        self.text(sc, CHIP_X, 0.347, "digit", op, size=0.029)
        for k in range(S.CLASSES):
            self.chip(sc, k, CHIP_X, row(k), op)

    def _column(self, sc, head, values, op):
        self.text(sc, COL_X, 0.347, head, op, anchor="rm", size=0.029)
        for k, v in enumerate(values):
            self.text(sc, COL_X, row(k), v, op, anchor="rm", size=0.029)

    def _bars(self, sc, values, top, ink, op, u, head, fmt="%.3f", start=0.0, span=0.5):
        """A bar per digit growing in turn from ``start`` over ``span`` of the phase."""
        self.text(sc, BAR_X, 0.347, head, op, anchor="lm", size=0.029, ink=dark(ink, 0.3))
        for k, v in enumerate(values):
            go = (u - start - span * k / S.CLASSES) / (span / S.CLASSES * 2.0)
            if go <= 0:
                continue
            self.hbar(sc, BAR_X, row(k), BAR_PX * v / top, ink, op, value=fmt % v,
                      u=min(go, 1.0), thick=15.0)

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        self.text(sc, 0.05, 0.300, "a node: the %s training digits at the root of a tree"
                  % commas(self.n), op, anchor="lm", strong=True, size=HEAD)
        self._chips(sc, op)
        self._column(sc, "count", [commas(c) for c in self.counts], op)
        self._bars(sc, self.p, 0.12, AXIS, op, u, "share p = count / %s" % commas(self.n),
                   fmt="%.3f", start=0.05, span=0.45)
        for i, (img, y) in enumerate(zip(self.images[:12], self.labels[:12])):
            fx, fy = 0.585 + 0.062 * (i % 6), 0.40 + 0.115 * (i // 6)
            a = op * smoothstep(0.02 + 0.02 * i, 0.08 + 0.02 * i, u)
            self.digit(sc, img, fx, fy, 50, a, frame=soft(TEN[int(y)], 0.25))
        b = op * smoothstep(0.55, 0.65, u)
        lo, hi = int(self.counts.min()), int(self.counts.max())
        self.lines(sc, 0.55, 0.64, [
            "the first of them, framed in their digit's colour",
            ("%s to %s of each kind: close to even" % (commas(lo), commas(hi)), None),
            ("an even mix would be p = %g for every digit" % (1 / S.CLASSES), None),
            ("how impure is this node? only the counts matter", dark(ENT, 0.3))],
            b, size=SMALL, step=0.046)
        self.tag(sc, exact(self.n), op)

    def phase_2(self, sc, u, op):
        r, m = self.root, self.meta
        self.text(sc, 0.05, 0.300, "entropy, term by term", op, anchor="lm", strong=True,
                  size=HEAD)
        self._chips(sc, op)
        self._column(sc, "p", ["%.3f" % v for v in self.p], op)
        self._bars(sc, self.hterms, 0.40, ENT, op, u, "−p log₂ p", start=0.04, span=0.40)
        x = 0.55
        self.text(sc, x, 0.355, "H = −Σ p log₂ p", op, anchor="lm", ink=dark(ENT, 0.3),
                  size=BIG)
        added = int(np.clip(np.floor((u - 0.04) / 0.40 * S.CLASSES + 0.5), 0, S.CLASSES))
        total = float(self.hterms[:added].sum())
        self.text(sc, x, 0.415, "the terms added so far: %s" % bits(total), op, anchor="lm",
                  size=SMALL)
        b = op * smoothstep(0.48, 0.58, u)
        full = self.span(0.19)[0]
        self.hbar(sc, x + 0.135, 0.49, full * r["entropy"] / m["max_entropy"], ENT, b,
                  "this node", "%.3f" % r["entropy"], thick=18.0, track=full)
        self.hbar(sc, x + 0.135, 0.545, full, soft(ENT, 0.5), b, "an even mix",
                  "%.3f" % m["max_entropy"], thick=18.0)
        self.text(sc, x, 0.60, "an even mix of ten: log₂ 10 = %.3f bits, the most"
                  % m["max_entropy"], b, anchor="lm", size=SMALL)
        c = op * smoothstep(0.62, 0.72, u)
        po, pr = self.p[self.often], self.p[self.rare]
        self.lines(sc, x, 0.675, [
            ("−log₂ p: the surprise of one digit, in bits", dark(ENT, 0.3)),
            "the commonest, a %d (p = %.3f): %.2f bits" % (self.often, po, -np.log2(po)),
            "the rarest, a %d (p = %.3f): %.2f bits" % (self.rare, pr, -np.log2(pr)),
            "entropy is the mean surprise of the node's digits"],
            c, size=SMALL, step=0.046)
        self.tag(sc, exact(self.n), op)

    def phase_3(self, sc, u, op):
        r, m = self.root, self.meta
        self.text(sc, 0.05, 0.300, "Gini, term by term", op, anchor="lm", strong=True,
                  size=HEAD)
        self._chips(sc, op)
        self._column(sc, "p", ["%.3f" % v for v in self.p], op)
        self._bars(sc, self.gterms, 0.12, GIN, op, u, "p(1 − p)", start=0.03, span=0.30)
        x = 0.55
        self.text(sc, x, 0.355, "G = Σ p(1 − p) = 1 − Σ p²", op, anchor="lm",
                  ink=dark(GIN, 0.4), size=BIG)
        added = int(np.clip(np.floor((u - 0.03) / 0.30 * S.CLASSES + 0.5), 0, S.CLASSES))
        self.text(sc, x, 0.413, "the terms added so far: %.3f; the most, 1 − 1/10 = %g"
                  % (float(self.gterms[:added].sum()), m["max_gini"]), op, anchor="lm",
                  size=SMALL)
        b = op * smoothstep(0.36, 0.44, u)
        self.text(sc, x, 0.475, "what it counts: draw two digits, each put back:",
                  b, anchor="lm", size=SMALL)
        self.text(sc, x, 0.517, "how often are they different digits?", b, anchor="lm",
                  size=SMALL, ink=dark(GIN, 0.4))
        for i in range(5):
            a = b * smoothstep(0.40 + 0.03 * i, 0.44 + 0.03 * i, u)
            fx = x + 0.012 + 0.078 * i
            self.chip(sc, self.pa[i], fx, 0.57, a, size=20.0)
            self.text(sc, fx + 0.0235, 0.57, "=" if self.pa[i] == self.pb[i] else "≠", a,
                      size=0.029, ink=None if self.pa[i] == self.pb[i] else dark(GIN, 0.4))
            self.chip(sc, self.pb[i], fx + 0.047, 0.57, a, size=20.0)
        c = op * smoothstep(0.55, 0.6, u)
        ch = self.pane(((x + 0.06, 0.92), (0.64, 0.80)), (10, m["pairs"]), (0.80, 1.0),
                       xlog=True)
        self.axes(sc, ch, c, "", "", [(10, "10"), (1000, "1,000"),
                                      (m["pairs"], "%s pairs" % commas(m["pairs"]))],
                  [(0.9, "0.9"), (1.0, "1")])
        self.hline(sc, ch, r["gini"], GIN, c, width=1.6, dashed=True)
        k = np.unique(np.geomspace(10, m["pairs"], 160).astype(int))
        go = smoothstep(0.6, 0.9, u)
        self.path(sc, ch, k, self.differ[k - 1], DIGIT_INK, c, go, width=2.0,
                  dot=True)
        d = op * smoothstep(0.88, 0.94, u)
        self.text(sc, x, 0.865, "share that differ: %.4f ± %.4f; Gini %.4f"
                  % (r["pairs_differ"], r["pairs_error"], r["gini"]), d, anchor="lm",
                  size=SMALL, ink=dark(GIN, 0.4))
        self.tag(sc, exact(self.n) + "; pairs drawn at random", op)

    def _code(self, sc, word, fx, fy, op, ink, size=16.0):
        for j, bit in enumerate(word):
            self.box(sc, fx + self.frac(size + 2.0)[0] * (j + 0.5), fy, size, size, soft(ink, 0.6),
                     op, bit, border=ink, size=0.029, radius=3)

    def phase_4(self, sc, u, op):
        r, d = self.root, self.deep
        self.text(sc, 0.05, 0.300, "what entropy counts: yes/no questions", op, anchor="lm",
                  strong=True, size=HEAD)
        self._chips(sc, op)
        self._column(sc, "p", ["%.3f" % v for v in self.p], op)
        self.text(sc, BAR_X, 0.347, "a Huffman code", op, anchor="lm", size=0.029,
                  ink=dark(TEAL, 0.3))
        self.text(sc, 0.405, 0.347, "−log₂ p", op, anchor="rm", size=0.029, ink=dark(ENT, 0.3))
        for k in range(S.CLASSES):
            a = op * smoothstep(0.04 + 0.025 * k, 0.09 + 0.025 * k, u)
            self._code(sc, self.code[k], BAR_X, row(k), a, TEAL)
            self.text(sc, 0.405, row(k), "%.2f" % -np.log2(self.p[k]), a, anchor="rm",
                      size=0.029)
        x = 0.50
        b = op * smoothstep(0.32, 0.4, u)
        self.lines(sc, x, 0.355, [
            "each answer is one bit: 0 for no, 1 for yes",
            "no codeword begins another, so a run of answers",
            "names the digits one after another"], b, size=SMALL, step=0.044)
        c = op * smoothstep(0.45, 0.53, u)
        full = self.span(0.25)[0]
        top = 3.5
        x2 = 0.69
        self.text(sc, x, 0.525, "at the root", c, anchor="lm", strong=True, size=SMALL)
        self.hbar(sc, x2, 0.575, full * r["entropy"] / top, ENT, c, "entropy H",
                  "%.3f" % r["entropy"], thick=16.0)
        self.hbar(sc, x2, 0.62, full * r["huffman_mean"] / top, TEAL, c, "mean questions L",
                  "%.3f" % r["huffman_mean"], thick=16.0)
        e = op * smoothstep(0.62, 0.7, u)
        self.text(sc, x, 0.68, "%s digits, %s of them %ds" % (
            commas(d["n"]), commas(d["counts"][self.most]), self.most), e, anchor="lm",
            strong=True, size=SMALL)
        self.hbar(sc, x2, 0.73, full * d["entropy"] / top, ENT, e, "entropy H",
                  "%.3f" % d["entropy"], thick=16.0)
        self.hbar(sc, x2, 0.775, full * d["huffman_mean"] / top, TEAL, e, "mean questions L",
                  "%.3f" % d["huffman_mean"], thick=16.0)
        f = op * smoothstep(0.78, 0.86, u)
        self.text(sc, x, 0.835, "H ≤ L < H + 1: no code asks less than one question a digit",
                  f, anchor="lm", size=SMALL, ink=dark(TEAL, 0.3))
        self.tag(sc, exact(self.n), op)

    def _node(self, sc, counts, fx, fy, op, head, lines, w=96.0, h=40.0):
        """A node: its shares as bars, a head line above and its numbers below."""
        H = S.entropy(counts)
        self.shares(sc, counts, fx, fy, w, h, op, frame=entropy_ink(H, 3.0))
        self.text(sc, fx, fy - self.frac(0, h / 2 + 14)[1], head, op, size=0.029, strong=True)
        for i, s in enumerate(lines):
            self.text(sc, fx, fy + self.frac(0, h / 2 + 16)[1] + 0.04 * i, s, op, size=0.029)

    def phase_5(self, sc, u, op):
        r, a, b, m = self.root, self.left, self.right, self.meta
        self.text(sc, 0.05, 0.300, "the first question", op, anchor="lm", strong=True,
                  size=HEAD)
        rx, ry = 0.25, 0.40
        self._node(sc, r["counts"], rx, ry, op, "", ["%s · Gini %.3f" % (bits(r["entropy"]),
                                                                       r["gini"])])
        q = op * smoothstep(0.06, 0.14, u)
        self.cell_on_digit(sc, self.images[0], r["feature"], rx + 0.135, ry, 52, q)
        self.text(sc, rx + 0.175, ry - 0.025, "the ringed cell:", q, anchor="lm", size=0.029)
        self.text(sc, rx + 0.175, ry + 0.015, ask(r["cut"]), q, anchor="lm", size=0.029,
                  ink=dark(ASK, 0.4))
        kids = op * smoothstep(0.14, 0.24, u)
        for (fx, node, word) in ((0.12, a, "yes"), (0.38, b, "no")):
            self.arrow(sc, (rx + (-0.03 if fx < rx else 0.03), ry + 0.10),
                       (fx, 0.54), AXIS, kids, width=1.8)
            self._node(sc, node["counts"], fx, 0.64, kids, "%s: %s digits" % (
                word, commas(node["n"])), [bits(node["entropy"]), "Gini %.3f" % node["gini"]])
        w = op * smoothstep(0.30, 0.38, u)
        self.text(sc, 0.05, 0.815, "weighted: %.4f × %.3f + %.4f × %.3f = %s" % (
            a["n"] / r["n"], a["entropy"], b["n"] / r["n"], b["entropy"],
            bits(m["weighted_children_entropy"])), w, anchor="lm", size=SMALL)
        self.text(sc, 0.05, 0.86, "gain = entropy before − weighted after = %s"
                  % bits(m["first_gain"]), w, anchor="lm", size=SMALL, ink=dark(GAIN, 0.3))
        # down the yes side to the node drawn by hand
        x = 0.585
        c = op * smoothstep(0.42, 0.5, u)
        self.text(sc, x - 0.02, 0.355, "down the yes side", c, anchor="lm", strong=True,
                  size=SMALL)
        self.text(sc, 0.95, 0.355, "bits   Gini", c, anchor="rm", size=0.029)
        for i, cnt in enumerate(self.path_counts):
            on = op * smoothstep(0.44 + 0.06 * i, 0.5 + 0.06 * i, u)
            fy = 0.41 + 0.083 * i
            H = S.entropy(cnt)
            self.text(sc, x, fy, "depth %d" % i, on, anchor="lm", size=0.029)
            self.shares(sc, cnt, x + 0.135, fy, 70, 26, on, frame=entropy_ink(H, 3.0))
            self.text(sc, x + 0.19, fy, commas(cnt.sum()), on, anchor="lm", size=0.029)
            self.text(sc, 0.95, fy, "%.3f   %.3f" % (H, S.gini(cnt)), on, anchor="rm",
                      size=0.029, ink=dark(ENT, 0.3) if i == len(self.yes_path) - 1 else None)
        self.tag(sc, exact(self.n), op)

    def phase_6(self, sc, u, op):
        m = self.meta
        ch = self.pane(((0.10, 0.50), (0.37, 0.78)), (0.0, 1.0), (0.0, 1.05))
        self.axes(sc, ch, op, "share of one of two kinds, p", "", [(v, "%g" % v) for v in
                                                                   (0, 0.25, 0.5, 0.75, 1)],
                  [(v, "%g" % v) for v in (0, 0.5, 1)])
        w = smoothstep(0.45, 0.65, u)
        go = smoothstep(0.05, 0.35, u)
        ent = self.two[0] * (1 + (LN2 - 1) * w)
        gin = self.two[1] * (1 - 0.5 * w)
        self.path(sc, ch, self.two_p, self.two[2], ERR, op * (1 - w), go, width=2.4)
        self.path(sc, ch, self.two_p, gin, GIN, op, go, width=2.8)
        self.path(sc, ch, self.two_p, ent, ENT, op, go, width=2.8)
        x = 0.56
        before, after = op * float(np.clip(1 - 2 * w, 0, 1)), op * float(np.clip(2 * w - 1, 0, 1))
        self.key(sc, x, 0.36, [("entropy, bits", ENT), ("2 × Gini", GIN), ("2 × error", ERR)],
                 before, size=SMALL)
        self.key(sc, x, 0.36, [("entropy, nats", ENT), ("Gini", GIN)], after, size=SMALL)
        self.text(sc, x, 0.50, "each 0 when pure and 1 at half and half", before,
                  anchor="lm", size=SMALL)
        c = op * smoothstep(0.62, 0.7, u)
        self.text(sc, x, 0.465, "in nats entropy lies above Gini", c, anchor="lm", size=SMALL,
                  ink=dark(ENT, 0.3))
        self.text(sc, x, 0.535, "node", c, anchor="lm", size=0.029)
        self.text(sc, 0.82, 0.535, "nats", c, anchor="rm", size=0.029, ink=dark(ENT, 0.3))
        self.text(sc, 0.93, 0.535, "Gini", c, anchor="rm", size=0.029, ink=dark(GIN, 0.4))
        names = ("root", "yes side", "no side", "depth %d" % self.deep["depth"])
        for i, (name, v) in enumerate(zip(names, m["nats_vs_gini"])):
            fy = 0.58 + 0.042 * i
            self.text(sc, x, fy, name, c, anchor="lm", size=0.029)
            self.text(sc, 0.82, fy, "%.3f" % v["nats"], c, anchor="rm", size=0.029)
            self.text(sc, 0.93, fy, "%.3f" % v["gini"], c, anchor="rm", size=0.029)
        self.text(sc, x, 0.78, "−p ln p ≥ p(1 − p) for every digit,", c, anchor="lm",
                  size=SMALL)
        self.text(sc, x, 0.822, "since −ln p ≥ 1 − p", c, anchor="lm", size=SMALL)
        self.tag(sc, "exact arithmetic on two kinds, and the counts of %s training digits"
                 % commas(self.n), op)


# ====================================================================== 173 down a tree
class EntropyDownATree(InfoTrees):
    """The tree's top levels coloured by entropy; the entropy left at every depth and the
    gains that remove it; a leaf read as a model, and the training loss in nats; the
    held-out digits given probability zero, and smoothing; held-out loss against depth."""

    TITLE = "Entropy down a tree"
    PHASES = (("the tree", 11.0), ("depth by depth", 12.0), ("a model", 12.0),
              ("zero", 10.0), ("held out", 12.0))
    TOP = 3                      # levels drawn as nodes

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        self.T = T = S.tree_of("entropy")
        self.D = m["depths"]
        self.G = m["gini_depths"]
        self.depth = np.array([r["depth"] for r in self.D], float)
        self.H = np.array([r["entropy"] for r in self.D], float)
        self.gains = np.asarray(m["gain_by_depth"], float)
        self.n = float(T["counts"][0].sum())
        self.Hn = S.impurity_of(T["counts"], "entropy")
        self.pos = node_layout(T, self.TOP, 0.07, 0.75)
        xb, _y, xbt, yt, images = S.tree_data()
        self.root_image = images[0]
        self.map = depth_map(T, xb, range(m["max_depth"] + 1))
        # a node drawn as a model: the one 172 works by hand, further down the yes side
        self.leaf = int(S.tree_runs()["hand_nodes"][-1])
        self.leaf_counts = np.asarray(T["counts"][self.leaf], float)
        # a held-out digit the unsmoothed tree, cut at full depth, gives probability zero
        node = S.path_nodes(T, xbt, m["max_depth"])
        zero = np.flatnonzero(T["counts"][node, yt] == 0)
        self.zero = int(zero[np.argmax(T["counts"][node[zero]].sum(1))])    # the largest leaf
        self.zero_label = int(yt[self.zero])
        self.zero_node = int(node[self.zero])
        self.zero_counts = np.asarray(T["counts"][self.zero_node], float)
        from . import mnist
        self.zero_image = mnist.load()[2][self.zero]
        self.best = int(m["best_depth"])
        self.n_held = len(yt)

    def subtitle_for(self, k):
        m, D = self.meta, self.D
        last, b = D[-1], D[self.best]
        return [
            "a tree grown on the 49 cells of %s training digits, all ten kinds: each node "
            "asks the question of most information gain. Colour: a node's entropy"
            % commas(self.n),
            "entropy left at each depth, the digits' mean: %.3f bits at the root, %.3f at "
            "depth %d. The gains of each depth, weighted by node size, add up to %.3f bits"
            % (D[0]["entropy"], last["entropy"], last["depth"], m["total_gain"]),
            "as a model, a leaf's shares are its probabilities and a digit's loss is −ln p of "
            "its own kind. In nats the entropy left is exactly the training loss: %.3f at the "
            "root" % D[0]["train_nats_unsmoothed"],
            "a held-out digit can reach a leaf with none of its kind: at depth %d the tree "
            "gives %s held-out digits probability 0, an infinite loss. So each leaf gets one "
            "imaginary digit of every kind"
            % (last["depth"], commas(last["test_zero_probability"])),
            "held-out loss falls from %.3f nats, about ln 10, to %.3f at depth %d (%s right), "
            "then rises to %.3f at depth %d while training entropy falls on to %.3f bits"
            % (D[0]["test_nats"], b["test_nats"], b["depth"], pct(b["test_accuracy"]),
               last["test_nats"], last["depth"], last["entropy"]),
        ][k]

    # ------------------------------------------------------------------ phases
    def _key(self, sc, fx, fy, op, w=150.0):
        """The colour scale of entropy, 0 to the root's."""
        steps = np.linspace(0.0, S.LOG2_10, 64)
        tex = np.array([[entropy_ink(h) for h in steps]], np.float32)
        draw_image(self, sc, self.P(fx + self.frac(w / 2)[0], fy), tex, w, 12, op, grid=0,
                   keep_theme=False)
        self.text(sc, fx, fy + 0.035, "0", op, anchor="lm", size=0.029)
        self.text(sc, fx + self.frac(w)[0], fy + 0.035, "%.2f bits" % S.LOG2_10, op,
                  anchor="rm", size=0.029)

    def phase_1(self, sc, u, op):
        T, m = self.T, self.meta
        ys = [0.37 + 0.142 * d for d in range(self.TOP + 1)]
        up = {}
        for t in np.flatnonzero(T["feature"] >= 0):
            up[int(T["left"][t])] = up[int(T["right"][t])] = int(t)
        for t, fx in self.pos.items():
            d = int(T["depth"][t])
            show = op * smoothstep(0.05 + 0.15 * d, 0.13 + 0.15 * d, u)
            if show <= 0.01:
                continue
            fy = ys[d]
            if t in up:
                px = self.pos[up[t]]
                self.arrow(sc, (px + (0.02 if fx > px else -0.02), ys[d - 1] + 0.085),
                           (fx, fy - 0.04), AXIS, show * 0.8, width=1.5)
            H = float(self.Hn[t])
            w = 64.0 if d == self.TOP else 78.0
            self.shares(sc, T["counts"][t], fx, fy, w, 30, show, frame=dark(ENT, 0.1),
                        fill=entropy_ink(H))
            self.text(sc, fx, fy + 0.058, "%.2f" % H, show, size=0.029)
        r = op * smoothstep(0.02, 0.1, u)
        self.text(sc, self.pos[0] - 0.075, ys[0], "%s digits" % commas(self.n), r,
                  anchor="rm", size=0.029)
        x = 0.79
        self.lines(sc, x, 0.325, ["each node: its", "ten shares; its", "entropy in bits"],
                   op, size=0.029, step=0.04)
        self._key(sc, x, 0.45, op)
        q = op * smoothstep(0.22, 0.3, u)
        self.text(sc, x, 0.53, "the root asks:", q, anchor="lm", size=0.029)
        self.cell_on_digit(sc, self.root_image, T["feature"][0], x + 0.045, 0.615, 56, q)
        self.text(sc, x, 0.70, ask(T["cut"][0]), q, anchor="lm", size=0.029, ink=dark(ASK, 0.4))
        e = op * smoothstep(0.7, 0.78, u)
        self.lines(sc, x, 0.765, ["grown to depth %d:" % m["max_depth"],
                                  "%s nodes," % commas(m["nodes"]),
                                  "%s leaves" % commas(m["leaves"])], e, size=0.029, step=0.04)
        self.tag(sc, exact(self.n), op)

    def phase_2(self, sc, u, op):
        m = self.meta
        rows = len(self.map)
        x0, x1, y0, y1 = 0.10, 0.46, 0.36, 0.80
        grow = smoothstep(0.03, 0.4, u)
        tex = np.asarray(PAPER, np.float32) + (np.asarray(ENT, np.float32) - PAPER) * \
            entropy_tint(self.map)[..., None]
        shown = int(np.ceil(rows * grow - 1e-9))
        tex[shown:] = PAPER
        w, h = self.span(x1 - x0, y1 - y0)
        draw_image(self, sc, self.P((x0 + x1) / 2, (y0 + y1) / 2),
                   shrink_columns(tex.astype(np.float32), int(w)), w, h, op, grid=0,
                   keep_theme=False)
        self.text(sc, x0, 0.33, "every training digit's node, depth by depth", op,
                  anchor="lm", size=0.029, strong=True)
        for d in (0, 5, 10, 14):
            if d < rows:
                self.text(sc, x0 - 0.01, y0 + (d + 0.5) * (y1 - y0) / rows, "depth %d" % d, op,
                          anchor="rm", size=0.029)
        self.text(sc, (x0 + x1) / 2, 0.83, "the %s digits, side by side as the tree"
                  % commas(self.n), op, size=0.029)
        self.text(sc, (x0 + x1) / 2, 0.87, "sorts them; colour: their node's entropy", op,
                  size=0.029)
        # the entropy left and the gains, as bars
        ch = self.pane(((0.58, 0.93), (0.38, 0.74)), (-0.6, 16.6), (0.0, 3.5))
        self.axes(sc, ch, op, "depth", "", [(v, "%d" % v) for v in (0, 5, 10, 14)],
                  [(v, "%g" % v) for v in (0, 1, 2, 3)])
        self.text(sc, 0.58, 0.345, "entropy left, bits", op, anchor="lm", size=0.029,
                  ink=dark(ENT, 0.3))
        go = smoothstep(0.05, 0.45, u) * len(self.H)
        for d, v in enumerate(self.H):
            if go <= d:
                break
            self.bars(sc, ch, [d - 0.38, d + 0.38], [v], soft(ENT, 0.25), op, fill=0.9)
            if d + 1 < len(self.H):
                g = op * smoothstep(0.45, 0.6, u)
                self.bars(sc, ch, [d + 1 - 0.38, d + 1 + 0.38], [self.gains[d]], GAIN, g,
                          fill=0.85, base=[self.H[d + 1]])
        st = op * smoothstep(0.6, 0.7, u)
        base = float(self.H[-1])
        fill = smoothstep(0.62, 0.85, u)
        for d, g in enumerate(self.gains):
            a = st * float(np.clip(fill * len(self.gains) - d, 0, 1))
            self.bars(sc, ch, [15.6, 16.6], [g], GAIN if d % 2 == 0 else dark(GAIN, 0.25), a,
                      fill=0.9, base=[base])
            base += g
        self.bars(sc, ch, [15.6, 16.6], [self.H[-1]], soft(ENT, 0.25), st, fill=0.9)
        self.text(sc, 0.93, 0.33, "all gains", st, anchor="rm", size=0.029, ink=dark(GAIN, 0.3))
        e = op * smoothstep(0.82, 0.9, u)
        self.text(sc, 0.93, 0.865, "gains %.3f = %.3f − %.3f left" % (
            m["total_gain"], self.H[0], self.H[-1]), e, anchor="rm", size=SMALL,
            ink=dark(GAIN, 0.3))
        self.tag(sc, TRAINING % commas(self.n), op)

    def phase_3(self, sc, u, op):
        D, c = self.D, self.leaf_counts
        n = c.sum()
        k = int(np.argmax(c))
        other = int(np.argsort(-c)[1])
        p = c / n
        self.text(sc, 0.05, 0.30, "a node as a model: depth %d, %s digits" % (
            self.T["depth"][self.leaf], commas(n)), op, anchor="lm", strong=True, size=HEAD)
        ch = self.pane(((0.09, 0.44), (0.40, 0.72)), (-0.5, 9.5), (0.0, 1.12))
        self.axes(sc, ch, op, "", "", [], [(v, "%g" % v) for v in (0, 0.5, 1)], grid=False)
        self.text(sc, 0.09, 0.355, "its shares, read as probabilities", op, anchor="lm",
                  size=0.029)
        go = smoothstep(0.03, 0.2, u)
        for d in range(S.CLASSES):
            self.bars(sc, ch, [d - 0.38, d + 0.38], [p[d]], TEN[d], op, fill=0.9, grow=go)
            self.chip(sc, d, ch.fx(d), 0.755, op, size=20.0)
        for d in (k, other):
            self.text(sc, ch.fx(d), ch.fy(p[d] * go) - 0.03, "%.3f" % p[d], op * go,
                      size=0.029, ink=dark(TEN[d], 0.3))
        a = op * smoothstep(0.22, 0.3, u)
        self.lines(sc, 0.05, 0.815, [
            "a %d in it: −ln(%d/%d) = %.3f nats;  a %d: −ln(%d/%d) = %.3f nats"
            % (k, c[k], n, -np.log(p[k]), other, c[other], n, -np.log(p[other])),
            ("mean over its digits: %.3f nats = %.3f bits × ln 2, its entropy"
             % (S.entropy(c, np.e), S.entropy(c)), dark(ENT, 0.3))], a, size=SMALL,
            step=0.044)
        b = op * smoothstep(0.4, 0.48, u)
        x = 0.53
        self.text(sc, x, 0.36, "every node at a depth", b, anchor="lm", strong=True, size=SMALL)
        heads = ("depth", "entropy left", "× ln 2", "training loss")
        cols = (x, 0.70, 0.80, 0.93)
        for j, (s, cx) in enumerate(zip(heads, cols)):
            self.text(sc, cx, 0.41, s, b, anchor="lm" if j == 0 else "rm", size=0.029,
                      ink=dark(TRAIN, 0.3) if j == 3 else None)
        self.text(sc, 0.70, 0.45, "bits", b, anchor="rm", size=0.029)
        self.text(sc, 0.80, 0.45, "nats", b, anchor="rm", size=0.029)
        self.text(sc, 0.93, 0.45, "nats", b, anchor="rm", size=0.029)
        for i, d in enumerate((0, 1, 5, 10, 14)):
            r = D[d]
            on = op * smoothstep(0.48 + 0.06 * i, 0.54 + 0.06 * i, u)
            fy = 0.495 + 0.045 * i
            vals = ("%d" % d, "%.4f" % r["entropy"], "%.4f" % (r["entropy"] * LN2),
                    "%.4f" % r["train_nats_unsmoothed"])
            for j, (s, cx) in enumerate(zip(vals, cols)):
                self.text(sc, cx, fy, s, on, anchor="lm" if j == 0 else "rm", size=0.029)
        e = op * smoothstep(0.8, 0.88, u)
        self.text(sc, x, 0.73, "the same at every depth: the loss of the", e, anchor="lm",
                  size=SMALL, ink=dark(TRAIN, 0.3))
        self.text(sc, x, 0.772, "unsmoothed shares is the entropy left", e, anchor="lm",
                  size=SMALL, ink=dark(TRAIN, 0.3))
        self.tag(sc, TRAINING % commas(self.n), op)

    def phase_4(self, sc, u, op):
        m, D = self.meta, self.D
        c = self.zero_counts
        n = c.sum()
        z = self.zero_label
        held = commas(self.n_held)
        self.text(sc, 0.05, 0.30, "a held-out %d meets a leaf with no %ds" % (z, z), op,
                  anchor="lm", strong=True, size=HEAD)
        kinds = np.flatnonzero(c)
        self.digit(sc, self.zero_image, 0.11, 0.43, 70, op, frame=soft(TEN[z], 0.25))
        self.lines(sc, 0.11, 0.52, ["held-out", "digit no. %d" % self.zero], op, size=0.029,
                   step=0.038, anchor="mm")
        self.arrow(sc, (0.17, 0.43), (0.215, 0.43), AXIS, op)
        self.shares(sc, c, 0.30, 0.43, 110, 46, op, frame=dark(ENT, 0.1),
                    fill=entropy_ink(S.entropy(c)))
        self.lines(sc, 0.30, 0.52, ["its leaf, at depth %d:" % self.T["depth"][self.zero_node],
                                    "%s training digits%s" % (commas(n), ", all %ds" % kinds[0]
                                                              if len(kinds) == 1 else "")],
                   op, size=0.029, step=0.038, anchor="mm")
        a = op * smoothstep(0.12, 0.2, u)
        self.lines(sc, 0.05, 0.63, [
            "p(%d) = 0/%d = 0: loss −ln 0, infinite" % (z, n),
            ("smoothed, one imaginary digit of each kind:", dark(HELD, 0.3)),
            ("p(%d) = (0 + 1)/(%d + 10) = %.3f: loss %.2f nats" % (
                z, n, 1 / (n + S.CLASSES), np.log(n + S.CLASSES)), dark(HELD, 0.3))],
            a, size=SMALL, step=0.046)
        b = op * smoothstep(0.3, 0.38, u)
        zeros = np.array([r["test_zero_probability"] for r in D], float)
        ch = self.pane(((0.60, 0.93), (0.38, 0.76)), (0, m["max_depth"]), (0, 1800))
        self.axes(sc, ch, b, "depth", "", [(v, "%d" % v) for v in (0, 5, 10, 14)],
                  [(v, commas(v)) for v in (0, 500, 1000, 1500)])
        self.text(sc, 0.60, 0.345, "held-out digits given p = 0", b, anchor="lm", size=0.029,
                  ink=dark(HELD, 0.3))
        end = self.path(sc, ch, self.depth, zeros, HELD, b, smoothstep(0.38, 0.75, u),
                        width=2.6, dot=True)
        e = op * smoothstep(0.75, 0.82, u)
        if end is not None:
            self.text(sc, end[0] - 0.012, end[1] - 0.04, commas(zeros[-1]), e, anchor="rm",
                      size=0.029, ink=dark(HELD, 0.3))
        self.text(sc, 0.60, 0.865, "of %s held-out digits, unsmoothed" % held, e,
                  anchor="lm", size=0.029)
        self.tag(sc, MEASURED % held, op)

    def phase_5(self, sc, u, op):
        m, D, G = self.meta, self.D, self.G
        b = D[self.best]
        ch = self.pane(((0.10, 0.58), (0.37, 0.78)), (0, m["max_depth"]), (0.0, 2.4))
        self.axes(sc, ch, op, "depth of the tree", "", [(v, "%d" % v) for v in
                                                         (0, 2, 4, 6, 8, 10, 12, 14)],
                  [(v, "%g" % v) for v in (0, 0.5, 1, 1.5, 2)])
        self.text(sc, 0.10, 0.335, "nats", op, anchor="lm", size=0.029)
        go = smoothstep(0.04, 0.5, u)
        train = np.array([r["train_nats_unsmoothed"] for r in D])
        test = np.array([r["test_nats"] for r in D])
        gtest = np.array([r["test_nats"] for r in G])
        self.hline(sc, ch, np.log(S.CLASSES), AXIS, op, width=1.4, dashed=True)
        self.path(sc, ch, self.depth, train, ENT, op, go, width=2.6, dot=True)
        self.path(sc, ch, self.depth, test, HELD, op, go, width=2.8, dot=True)
        g = op * smoothstep(0.62, 0.72, u)
        self.path(sc, ch, self.depth, gtest, soft(GIN, 0.2), g, 1.0, width=1.6)
        x = 0.64
        self.key(sc, x, 0.36, [("training entropy left, in nats", ENT),
                               ("held-out loss, smoothed", HELD)], op, size=0.029)
        self.key(sc, x, 0.448, [("held out, a tree grown by Gini", soft(GIN, 0.2))], g,
                 size=0.029)
        self.text(sc, x, 0.51, "dashed: ln 10 = %.3f, a uniform guess" % np.log(S.CLASSES),
                  op, anchor="lm", size=0.029)
        a = op * smoothstep(0.5, 0.58, u)
        at = (ch.fx(self.best), ch.fy(b["test_nats"]))
        self.disk(sc, self.P(*at), 7.0, HELD, a, ring=ASK, ring_w=3.0, lift=1.5)
        self.text(sc, at[0], at[1] - 0.08, "%.3f" % b["test_nats"], a, size=0.029,
                  ink=dark(HELD, 0.3))
        self.lines(sc, x, 0.58, [
            ("root: %.3f nats; best: %.3f at depth %d" % (D[0]["test_nats"], b["test_nats"],
                                                          b["depth"]), dark(HELD, 0.3)),
            ("then up to %.3f at depth %d" % (D[-1]["test_nats"], D[-1]["depth"]),
             dark(HELD, 0.3)),
            "held-out accuracy: %s at the root," % pct(D[0]["test_accuracy"]),
            "%s at depth %d, %s at %d" % (pct(b["test_accuracy"]), b["depth"],
                                          pct(D[-1]["test_accuracy"]), D[-1]["depth"]),
            ("Gini tree: best %.3f at depth %d" % (min(gtest), int(np.argmin(gtest))),
             dark(GIN, 0.4))],
            a, size=0.029, step=0.042)
        self.tag(sc, MEASURED % commas(self.n_held), op)


ANIMATIONS = {
    "entropy-and-gini-by-hand": EntropyAndGiniByHand,
    "entropy-down-a-tree": EntropyDownATree,
}

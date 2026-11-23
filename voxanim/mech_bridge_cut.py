"""Animation 178 of the "next" catalogue (see :mod:`voxanim.mech`): the bridge from Part 1's
cuts and tests to Part 2's decision trees.

* :class:`FromACutToATree`: one feature of MNIST 4s (signal) and 9s (background), a cut
  moved along it and chosen by ``s / sqrt(s + b)``, as in "Moving a cut"; the same
  threshold scored by Gini gain, as a tree's root scores it, and the cut drawn as a tree
  of depth one; cuts on cuts, the tree's boxes in the plane of two features; held-out ROC
  curves of the cut, the tree at several depths and the likelihood ratio of two 2-D
  histograms, as in "The likelihood ratio is the best test"; and how deep is deep enough.

It draws :mod:`voxanim.study_bridge_cut`'s study: every number on screen is the study's
metadata or panels, and the boxes are worked out from its cached tree with
:func:`voxanim.study_trees.regions`. Every phase says, bottom right, what kind of evidence
it shows: exact arithmetic on the training events, or a measurement on MNIST.
"""

from __future__ import annotations

import numpy as np

from . import study_bridge_cut as S
from . import study_trees as T
from .easing import smoothstep
from .mech import AMBER, AXIS, GREEN, MAGENTA, OCHRE, TEAL, VIOLET, dark, soft
from .mech_phys_apps import GREY, commas, pct
from .mech_trees import BKG, SIG, Trees, tree_layout

BIG, HEAD, SMALL, TEXT_S = 0.036, 0.033, 0.030, 0.029
FOM, GINI = TEAL, AMBER                    # s / sqrt(s + b) and the Gini gain
TRAIN, HELD, RATIO = VIOLET, GREEN, MAGENTA
FROM_CUT = "from: Moving a cut"
FROM_TEST = "from: The likelihood ratio is the best test"
NEXT = "next: Growing a decision tree · Choosing a split"
SPREAD = "vertical spread (pixels)"


def exact(n):
    """The tag of a phase that works something out from the training events' counts."""
    return "exact arithmetic on %s training events" % commas(n)


def grown(n):
    return "measured on MNIST: a tree grown on %s training events" % commas(n)


def held(n):
    return "measured on MNIST: %s held-out events" % commas(n)


def is_evidence(text):
    """True for the tag every phase carries."""
    return text.startswith(("exact arithmetic on ", "measured on MNIST"))


def px(v):
    """A threshold on the spread, in pixels."""
    return "%.2f" % float(v)


def agree(f, g):
    """What Gini's cut is beside the cut of s / sqrt(s + b) (two records of the study)."""
    if f["cut"] == g["cut"]:
        return "picks the same cut, %s pixels" % px(g["cut"])
    return "picks %s, %.2f pixels from s / √(s + b)" % (px(g["cut"]), abs(f["cut"] - g["cut"]))


def moved(r):
    """Where the cuts go with the signal made rare."""
    if r["same"]:
        return "both move to %s" % px(r["by_fom"]["cut"])
    return "s / √(s + b) moves to %s and Gini to %s" % (px(r["by_fom"]["cut"]),
                                                       px(r["by_gini"]["cut"]))


def picks(r):
    if r["same"]:
        return "both pick %s" % px(r["by_fom"]["cut"])
    return "%s and %s" % (px(r["by_fom"]["cut"]), px(r["by_gini"]["cut"]))


def crop(xs, ys, top):
    """The part of a rising curve below ``top``, ending where it crosses it."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    over = np.flatnonzero(ys > top)
    if not len(over):
        return xs, ys
    k = int(over[0])
    if k == 0:
        return xs[:1], np.minimum(ys[:1], top)
    f = (top - ys[k - 1]) / (ys[k] - ys[k - 1])
    return np.r_[xs[:k], xs[k - 1] + f * (xs[k] - xs[k - 1])], np.r_[ys[:k], top]


class FromACutToATree(Trees):
    """A cut on one feature chosen by s / sqrt(s + b); the same cut chosen by Gini, a
    stump; cuts on cuts in the plane; held-out ROC curves against the likelihood ratio;
    the area under them against depth."""

    TITLE = "From a cut to a tree"
    PHASES = (("a cut", 11.0), ("Gini's cut", 11.0), ("cuts on cuts", 12.0),
              ("against the ratio", 12.0), ("how deep", 10.0))
    HIST = ((0.09, 0.50), (0.395, 0.56))
    SCORE = ((0.09, 0.50), (0.665, 0.785))
    LEFT = ((0.08, 0.41), (0.395, 0.80))
    ROC = ((0.09, 0.47), (0.395, 0.80))
    DEPTH = ((0.09, 0.53), (0.37, 0.80))
    ROC_X, ROC_Y = (0.5, 1.0), (0.0, 0.4)

    def setup(self):
        s = self.study
        m = self.meta = s.metadata
        p0, p1, p2, p3 = (p.data for p in s.panels)
        self.cuts_x = np.asarray(p0["x"], float)
        self.s, self.b = np.asarray(p0["s"], float), np.asarray(p0["b"], float)
        self.fom, self.gain = np.asarray(p0["fom"], float), np.asarray(p0["gain"], float)
        self.fom_n, self.gain_n = (np.asarray(v, float) for v in p0["ys"])
        self.edges = np.asarray(p0["edges"], float)
        self.h_sig, self.h_bkg = np.asarray(p0["signal"], float), np.asarray(p0["background"],
                                                                            float)
        every = max(1, len(self.cuts_x) // 400)       # the score curves, thinned to draw
        self.thin = np.r_[np.arange(0, len(self.cuts_x), every), len(self.cuts_x) - 1]
        self.xy, self.cls = np.asarray(p1["xy"], float), np.asarray(p1["classes"])
        self.boxes = {d: np.asarray(p1["boxes_%d" % d], float) for d in m["plane_depths"]}
        self.tree, self.tcuts = S.tree()
        self.grid = np.asarray(p2["x"], float)
        self.curves = np.asarray(p2["ys"], float)       # cut, depths shown..., ratio
        self.depths = np.asarray(p3["x"], float)
        self.train_area, self.held_area = (np.asarray(v, float) for v in p3["ys"])
        self.leaves = np.asarray(p3["leaves"])
        self.one, self.rare, self.root = m["one_feature"], m["rare"], m["root"]
        self.by_fom, self.by_gini = self.one["by_fom"], self.one["by_gini"]
        self.n_train, self.n_held = m["n_train"], m["n_held"]
        self.at = {d["depth"]: d for d in m["depths"]}
        self.pos = tree_layout(self.tree, ((0.485, 0.95), (0.37, 0.55)), max_depth=2)
        self.stump_pos = tree_layout(self.tree, ((0.60, 0.90), (0.55, 0.67)), max_depth=1)

    # ------------------------------------------------------------------ words
    def subtitle_for(self, k):
        m, f, g, r = self.meta, self.by_fom, self.by_gini, self.rare
        best = self.at[m["best_depth"]]
        last = self.at[m["deepest"]]
        return [
            "as in Moving a cut, keep the events on one side of a cut on one variable, here "
            "how far a digit's ink spreads. s / √(s + b) of the %s training events is "
            "best at %s pixels" % (commas(self.n_train), px(f["cut"])),
            "a tree's root scores the same thresholds by Gini gain and %s. With the 4s made "
            "rare %s: a cut is a tree of depth one" % (agree(f, g), moved(r)),
            "a tree asks again in each half, with either feature: cuts on cuts. At depth %d "
            "its %d boxes each give the share of training 4s inside as the event's score"
            % (m["best_depth"], best["leaves"]),
            "on held-out events, where the cut keeps %s of 4s it keeps %s of 9s; the depth-%d "
            "tree %s and the likelihood ratio of two 2-D histograms %s"
            % (pct(m["stump_held"]["efficiency"]), pct(m["stump_held"]["background"], 1),
               m["best_depth"], pct(best["background_at_stump"], 1),
               pct(m["ratio"]["background_at_stump"], 1)),
            "deeper fits the training events better every time, but held-out events stop "
            "improving at depth %d (%.3f against the ratio's %.3f) and fall to %.3f at "
            "depth %d" % (m["best_depth"], m["best_area"], m["ratio"]["held_area"],
                          last["held_area"], m["deepest"]),
        ][k]

    # ------------------------------------------------------------------ pieces
    def _sweep(self, u):
        """The index of the threshold shown at ``u``: from a loose cut across to a tight
        one and back to the best by s / sqrt(s + b)."""
        lo = int(np.searchsorted(self.cuts_x, 5.6))
        hi = int(np.searchsorted(self.cuts_x, 4.2))
        best = int(np.argmax(self.fom))
        if u < 0.35:
            f = lo + (hi - lo) * smoothstep(0.05, 0.35, u)
        else:
            f = hi + (best - hi) * smoothstep(0.38, 0.62, u)
        return int(round(f))

    def _histogram(self, sc, op, box, cut=None, keep=True):
        top = max(self.h_sig.max(), self.h_bkg.max()) * 1.15
        ch = self.pane(box, T.PLANE[S.FEATURE], (0.0, top))
        self.axes(sc, ch, op, "", "training events",
                  [(v, "%g" % v) for v in (3.5, 4, 4.5, 5, 5.5, 6)],
                  [(v, commas(v)) for v in (0, 500, 1000) if v < top], grid=False)
        self.bars(sc, ch, self.edges, self.h_bkg, BKG, op, fill=0.45)
        self.bars(sc, ch, self.edges, self.h_sig, SIG, op, fill=0.45)
        if cut is not None and keep:
            a, b = ch.fx(T.PLANE[S.FEATURE][0]), ch.fx(cut)
            w, h = self.span(b - a, ch.y1 - ch.y0)
            self.rect(sc, self.P((a + b) / 2, (ch.y0 + ch.y1) / 2), w, h, soft(FOM, 0.75),
                      op * 0.5, radius=0.0, lift=0.05)
        return ch

    def _scores(self, sc, op, gini=0.0):
        """s / sqrt(s + b) (and, as far as ``gini``, the Gini gain) against the threshold,
        each as a share of its best."""
        ch = self.pane(self.SCORE, T.PLANE[S.FEATURE], (0.0, 1.08))
        self.axes(sc, ch, op, SPREAD, "share of best",
                  [(v, "%g" % v) for v in (3.5, 4, 4.5, 5, 5.5, 6)],
                  [(0.0, "0"), (0.5, "0.5"), (1.0, "1")], grid=False)
        xs = self.cuts_x[self.thin]
        self.path(sc, ch, xs, self.fom_n[self.thin], FOM, op, width=2.6)
        if gini > 0:
            self.path(sc, ch, xs, self.gain_n[self.thin], GINI, op, gini, width=2.6)
        return ch

    # ------------------------------------------------------------------ phases
    def phase_1(self, sc, u, op):
        k = self._sweep(u)
        c = float(self.cuts_x[k])
        self.text(sc, 0.05, 0.285, FROM_CUT, op, anchor="lm", strong=True, size=HEAD,
                  ink=dark(FOM, 0.35))
        ch = self._histogram(sc, op, self.HIST, cut=c)
        self.vline(sc, ch, c, FOM, op, width=2.6)
        self.text(sc, ch.fx(c) - 0.006, ch.y0 + 0.02, "keep ≤ %s" % px(c), op, anchor="rm",
                  size=TEXT_S, ink=dark(FOM, 0.4))
        ch2 = self._scores(sc, op)
        self.disk(sc, self.P(ch2.fx(c), ch2.fy(self.fom_n[k])), 5.5, FOM, op, glow=0.4,
                  lift=1.4, outline=0.0)
        self.classes_key(sc, 0.58, 0.36, op, step=0.13)
        self.lines(sc, 0.58, 0.43, [
            "one variable: how far the ink spreads up and down",
            "keep every event below the cut"], op, size=SMALL, step=0.042)
        self.text(sc, 0.58, 0.54, "cut at %s pixels" % px(c), op, anchor="lm", size=HEAD,
                  ink=dark(FOM, 0.4))
        self.lines(sc, 0.58, 0.59, [
            "s = %s 4s kept of %s" % (commas(self.s[k]), commas(self.meta["train_signal"])),
            "b = %s 9s kept of %s" % (commas(self.b[k]), commas(self.meta["train_background"])),
            ("s / √(s + b) = %.1f" % self.fom[k], dark(FOM, 0.4))], op, size=SMALL, step=0.044)
        end = op * smoothstep(0.66, 0.74, u)
        self.text(sc, 0.58, 0.76, "best: %s pixels, s / √(s + b) = %.1f"
                  % (px(self.by_fom["cut"]), self.by_fom["fom"]), end, anchor="lm",
                  strong=True, size=SMALL)
        self.stand_in(sc, op)
        self.tag(sc, exact(self.n_train), op)

    def phase_2(self, sc, u, op):
        f, g = self.by_fom, self.by_gini
        self.text(sc, 0.05, 0.285, "a tree's root scores the same thresholds by Gini gain",
                  op, anchor="lm", strong=True, size=HEAD)
        ch = self._histogram(sc, op, self.HIST)
        both = op * smoothstep(0.30, 0.40, u)
        self.vline(sc, ch, f["cut"], FOM, op, width=2.4)
        self.vline(sc, ch, g["cut"], GINI, both, width=2.4, dashed=True)
        grow = smoothstep(0.05, 0.30, u)
        ch2 = self._scores(sc, op, gini=grow)
        for v, ink in ((f["cut"], FOM), (g["cut"], GINI)):
            a = op if ink is FOM else both
            self.disk(sc, self.P(ch2.fx(v), ch2.fy(1.0)), 5.5, ink, a, glow=0.4, lift=1.4,
                      outline=0.0)
        self.key(sc, 0.565, 0.37, [("s / √(s + b) picks %s pixels" % px(f["cut"]), FOM)], op,
                 size=SMALL)
        self.key(sc, 0.565, 0.415, [("Gini gain picks %s pixels" % px(g["cut"]), GINI)], both,
                 size=SMALL)
        self.text(sc, 0.555, 0.465, "held out: %s and %s of 4s kept" % (
            pct(f["held"]["efficiency"]), pct(g["held"]["efficiency"])), both, anchor="lm",
            size=SMALL)
        stump = op * smoothstep(0.50, 0.60, u)
        self.tree_boxes(sc, self.tree, self.tcuts, self.stump_pos, stump, max_depth=1,
                        w=(130, 92), size=TEXT_S)
        self.text(sc, 0.75, 0.73, "in each leaf: training 4s | 9s", stump, size=TEXT_S)
        self.text(sc, 0.75, 0.775, "a cut is a tree of depth one: a stump", stump,
                  size=SMALL, strong=True)
        rare = op * smoothstep(0.74, 0.82, u)
        r = self.rare
        self.text(sc, 0.555, 0.835, "4s rare, %d per %s 9s: %s" % (
            r["signal"], commas(r["background"]), picks(r)), rare,
            anchor="lm", size=SMALL, ink=dark(FOM, 0.4))
        self.tag(sc, exact(self.n_train), op)

    def phase_3(self, sc, u, op):
        shown = self.meta["plane_depths"]
        step = int(np.clip(np.floor(smoothstep(0.04, 0.80, u) * len(shown)), 0,
                           len(shown) - 1))
        d = shown[step]
        self.text(sc, 0.05, 0.285, "cuts on cuts: ask again in each half", op, anchor="lm",
                  strong=True, size=HEAD)
        ch = self.plane(sc, self.LEFT, op)
        self.partition(sc, ch, self.tree, self.tcuts, op, max_depth=d, fill=0.55)
        self.events(sc, ch, self.xy, self.cls, op, r=2.0)
        upto = min(d, 2)
        pos = {t: v for t, v in self.pos.items() if self.tree["depth"][t] <= upto}
        self.tree_boxes(sc, self.tree, self.tcuts, pos, op, max_depth=upto,
                        w=(116, 90), size=TEXT_S)
        self.text(sc, 0.56, 0.62, "depth %d: %d boxes" % (d, self.at[d]["leaves"]), op,
                  anchor="lm", size=HEAD, ink=dark(AMBER, 0.4))
        self.lines(sc, 0.56, 0.68, [
            "each box scores an event by its",
            "share of training 4s: blue mostly 4s",
            "leaves drawn above: training 4s | 9s"], op, size=SMALL, step=0.047)
        self.tag(sc, grown(self.n_train), op)

    def _roc(self, sc, op, u):
        ch = self.pane(self.ROC, self.ROC_X, self.ROC_Y)
        self.axes(sc, ch, op, "share of held-out 4s kept", "share of 9s kept",
                  [(v, pct(v)) for v in (0.6, 0.8, 1.0)],
                  [(v, pct(v)) for v in (0.0, 0.1, 0.2, 0.3, 0.4)], grid=True)
        names = self._roc_names()
        for i, (ys, (_, ink)) in enumerate(zip(self.curves, names)):
            go = smoothstep(0.06 + 0.08 * i, 0.20 + 0.08 * i, u)
            if go <= 0:
                continue
            xs, yc = crop(self.grid, ys, self.ROC_Y[1])
            keep = xs >= self.ROC_X[0]
            if i == 1:                                    # the stump: one corner
                e, b = self.meta["stump_held"]["efficiency"], self.meta["stump_held"][
                    "background"]
                self.disk(sc, self.P(ch.fx(e), ch.fy(b)), 6.5, ink, op * go, glow=0.4,
                          lift=1.5, outline=0.0)
                continue
            self.path(sc, ch, xs[keep], yc[keep], ink, op, go,
                      width=3.0 if ink is RATIO or ink is GREEN else 2.2)
        e = self.meta["stump_held"]["efficiency"]
        self.vline(sc, ch, e, AXIS, op * smoothstep(0.62, 0.70, u), width=1.6, dashed=True)
        return ch

    def _roc_names(self):
        m = self.meta
        rows = [("spread alone, every cut", GREY)]
        for i, d in enumerate(m["shown_depths"]):
            name = "the stump" if d == 1 else ("tree, depth %d" % d)
            ink = AMBER if d == 1 else (GREEN if d == m["best_depth"] else
                                        (VIOLET if d < m["best_depth"] else OCHRE))
            rows.append((name, ink if i else AMBER))
        rows.append(("likelihood ratio", RATIO))
        return rows

    def phase_4(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.05, 0.285, FROM_TEST, op, anchor="lm", strong=True, size=HEAD,
                  ink=dark(RATIO, 0.35))
        self._roc(sc, op, u)
        e = m["stump_held"]["efficiency"]
        names = self._roc_names()
        values = [m["cut_background_at_stump"], m["stump_held"]["background"]]
        values += [self.at[d]["background_at_stump"] for d in m["shown_depths"][1:]]
        values += [m["ratio"]["background_at_stump"]]
        self.text(sc, 0.535, 0.36, "9s kept where %s of 4s are" % pct(e), op, anchor="lm",
                  size=SMALL, strong=True)
        for i, ((name, ink), v) in enumerate(zip(names, values)):
            a = op * smoothstep(0.06 + 0.08 * i, 0.16 + 0.08 * i, u)
            self.key(sc, 0.545, 0.41 + 0.044 * i, [("%s: %s" % (name, pct(v, 1)), ink)], a,
                     size=TEXT_S)
        late = op * smoothstep(0.70, 0.80, u)
        self.lines(sc, 0.535, 0.74, [
            "the ratio: two histograms of training events,",
            "%d × %d bins, divided cell by cell" % (m["ratio"]["bins"], m["ratio"]["bins"]),
            ("a tree's boxes: bins it chose itself", dark(GREEN, 0.4))], late, size=TEXT_S,
            step=0.040)
        self.tag(sc, held(self.n_held), op)

    def phase_5(self, sc, u, op):
        m = self.meta
        go = smoothstep(0.04, 0.50, u)
        ch = self.pane(self.DEPTH, (1, m["deepest"]), (0.84, 1.005))
        self.axes(sc, ch, op, "depth of the tree", "area under the ROC curve",
                  [(v, "%d" % v) for v in (1, 5, 10, 15, 20, 25) if v <= m["deepest"]],
                  [(v, "%.2f" % v) for v in (0.85, 0.9, 0.95, 1.0)], grid=True)
        self.hline(sc, ch, m["ratio"]["held_area"], RATIO, op, width=2.0)
        self.path(sc, ch, self.depths, self.train_area, TRAIN, op, go, width=2.6, dot=True)
        self.path(sc, ch, self.depths, self.held_area, HELD, op, go, width=3.0, dot=True)
        mark = op * smoothstep(0.50, 0.58, u)
        best = m["best_depth"]
        self.disk(sc, self.P(ch.fx(best), ch.fy(m["best_area"])), 7.0, HELD, mark,
                  ring=AMBER, ring_w=3.0, lift=1.6)
        self.key(sc, 0.59, 0.37, [("training events", TRAIN), ("held-out events", HELD),
                                  ("likelihood ratio, held out", RATIO)], op, size=SMALL)
        last = self.at[m["deepest"]]
        self.lines(sc, 0.575, 0.535, [
            ("held out, best at depth %d: %.3f" % (best, m["best_area"]), dark(HELD, 0.4)),
            ("(%d boxes; the ratio %.3f)" % (self.at[best]["leaves"], m["ratio"]["held_area"]),
             None),
            ("grown to the end, depth %d:" % m["deepest"], None),
            ("%.3f held out, %.3f training" % (last["held_area"], last["train_area"]),
             dark(TRAIN, 0.3)),
            ("the cut alone: %.3f" % m["cut_area"], None)], mark, size=SMALL, step=0.042)
        end = op * smoothstep(0.70, 0.78, u)
        self.text(sc, 0.575, 0.78, "deeper stops helping", end, anchor="lm", size=HEAD,
                  strong=True)
        self.text(sc, 0.575, 0.835, NEXT, end, anchor="lm", size=SMALL, ink=dark(GREEN, 0.4))
        self.tag(sc, held(self.n_held), op)


ANIMATIONS = {S.NAME: FromACutToATree}
